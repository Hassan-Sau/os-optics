"""Golden case 14: x/y-independent Gaussian q-parameters, simple astigmatism, tilted spherical surfaces.

Independent references:
* astigmatic source through an ideal lens: two separate plain-numpy ABCD propagations;
* tilted spherical mirror: textbook f_t = (R/2) cos(theta), f_s = (R/2)/cos(theta);
* tilted mirror / thick tilted singlet: the 2x2 ray-transfer matrices are extracted by *exact geometric ray tracing written in
  this test* (own sphere-intersection / vector-Snell code, central finite differences) and applied as q' = (A q + B)/(C q + D).
  This verifies the engine's Coddington-based Gaussian code against pure ray geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import ConstantIndexMaterial, JonesVector, ProjectSnapshot

from helpers import abcd_apply, arrivals, free, issue_codes, one_arrival, rayleigh_range, run, thin

pytestmark = pytest.mark.golden
NM = 1e-6


def q_of(w0, lam, m2=1.0, z=0.0, n=1.0):
    return complex(z, math.pi * n * w0**2 / (m2 * lam * NM))


# ---------------------------------------------------------------------------------------------------------------
# x/y independent q with different waists and M^2 through an ideal lens (plain per-axis ABCD)
# ---------------------------------------------------------------------------------------------------------------
def test_golden_xy_independent_gaussian_q_astigmatic_source_and_lens():
    lam, f = 532.0, 120.0
    wx, wy, m2x, m2y = 0.3, 0.6, 1.1, 1.7
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=wx, waist_radius_y_mm=wy, m2=m2x, m2_y=m2y, position=(0, 0, 20),
                    direction=(1, 0, 0), waist_position_z_mm=0.0),
            B.thin_lens("Lens", focal_length_mm=f, clear_radius_mm=12.7, position=(80, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("d", position=(260, 0, 20), normal=(1, 0, 0), radius_mm=12.0),),
    )
    g = one_arrival(run(proj), "d").beam.gaussian
    for axis, w0, m2, q_engine, radius in (("x", wx, m2x, g.q_x_mm, g.beam_radius_x_mm), ("y", wy, m2y, g.q_y_mm, g.beam_radius_y_mm)):
        mat = free(180.0) @ thin(f) @ free(80.0)
        q_ref = abcd_apply(mat, q_of(w0, lam, m2))
        assert q_engine[0] == pytest.approx(q_ref.real, rel=1e-11, abs=1e-9)
        assert q_engine[1] == pytest.approx(q_ref.imag, rel=1e-11)
        w_ref = math.sqrt(m2 * lam * NM * abs(q_ref) ** 2 / (math.pi * q_ref.imag))
        assert radius == pytest.approx(w_ref, rel=1e-11)
    # the two axes genuinely differ (astigmatic beam)
    assert g.beam_radius_x_mm != pytest.approx(g.beam_radius_y_mm, rel=1e-3)
    assert g.m2_x == 1.1 and g.m2_y == 1.7


# ---------------------------------------------------------------------------------------------------------------
# helpers: exact geometric ray tracing (independent of the engine)
# ---------------------------------------------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _sphere_hit(o, d, c, r, root):
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    assert disc >= 0
    s = math.sqrt(disc)
    t = -b - s if root == "near" else -b + s
    return t


def _refract(d, n, n1, n2):
    n = n if d @ n < 0 else -n
    eta = n1 / n2
    c1 = -d @ n
    k = 1.0 - eta**2 * (1.0 - c1**2)
    assert k > 0
    return _unit(eta * d + (eta * c1 - math.sqrt(k)) * n)


def numeric_abcd(trace, chief_o, chief_d, e_t_in, e_s_in, delta_pos=1e-5, delta_ang=1e-7):
    """ABCD (physical y, theta) of a system from ray tracing. ``trace(o, d)`` returns ``(point, direction, plane_normal_reference)``.
    Output axes are the images of the input axes through the *chief* ray Jacobian: t' = component of the perturbed output
    ray in the plane spanned by the output chief and the input-tangential image."""
    p0, d0 = trace(chief_o, chief_d)

    def out_coords(p, d, e_t_out, e_s_out):
        # transverse offset of the ray in the plane through p0 perpendicular to d0, and small angles vs the chief
        s = ((p0 - p) @ d0) / (d @ d0)
        pt = p + s * d
        return np.array([(pt - p0) @ e_t_out, (pt - p0) @ e_s_out]), np.array([d @ e_t_out, d @ e_s_out])

    # output tangential axis: perturb the input by a small tangential displacement and read the direction of change
    # (tangential = in the plane of incidence). Sagittal axis is perpendicular to the plane of incidence and unchanged.
    e_s_out = e_s_in
    e_t_out = _unit(np.cross(e_s_out, d0))
    # orientation: choose the sign so that the tangential coordinate has the same handedness as the input (unfolded picture):
    # e_t_out must be the image of e_t_in under the small-angle mapping; determine by tracing a tangentially tilted ray.
    o1, d1 = trace(chief_o, _unit(chief_d + delta_ang * e_t_in))
    if (d1 - d0) @ e_t_out < 0:
        e_t_out = -e_t_out
    mats = {}
    for name, e_in, e_out in (("t", e_t_in, e_t_out), ("s", e_s_in, e_s_out)):
        cols = []
        for pos, ang in ((delta_pos, 0.0), (0.0, delta_ang)):
            res = []
            for sign in (+1.0, -1.0):
                o = chief_o + sign * pos * e_in
                d = _unit(chief_d + sign * ang * e_in)
                p, dd = trace(o, d)
                y, th = out_coords(p, dd, e_t_out, e_s_out)
                yy = y[0] if name == "t" else y[1]
                tt = th[0] if name == "t" else th[1]
                res.append((yy, tt))
            (yp, tp), (ym, tm) = res
            cols.append(((yp - ym) / (2 * (pos or ang)), (tp - tm) / (2 * (pos or ang))))
        (a, c), (b, d_) = cols  # columns: response to position, response to angle
        mats[name] = np.array([[a, b], [c, d_]])
    return mats, p0, d0, e_t_out


# ---------------------------------------------------------------------------------------------------------------
# tilted spherical mirror
# ---------------------------------------------------------------------------------------------------------------
def _mirror_trace_factory(vertex, normal_v, radius):
    """Concave (radius>0) / convex (radius<0) spherical mirror: centre = vertex + radius * normal_v (normal_v faces the beam)."""
    centre = vertex + radius * normal_v
    r = abs(radius)

    def trace(o, d):
        # concave: the ray starts inside the sphere -> far root; convex: outside -> near root
        t = _sphere_hit(o, d, centre, r, "far" if radius > 0 else "near")
        p = o + t * d
        n = (p - centre) / r
        return p, _unit(d - 2 * (d @ n) * n)

    return trace


@pytest.mark.parametrize("aoi_deg,radius", [(45.0, 200.0), (30.0, 250.0), (20.0, -300.0), (55.0, 400.0)])
def test_golden_tilted_spherical_mirror_astigmatism(aoi_deg, radius):
    lam, w0 = 633.0, 0.5
    a = math.radians(aoi_deg)
    normal_v = np.array([-math.cos(a), math.sin(a), 0.0])  # faces the incoming beam (beam along +x)
    vertex = np.array([100.0, 0.0, 20.0])
    d_in = np.array([1.0, 0.0, 0.0])
    # engine
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=0.0),
            B.spherical_mirror("M", position=tuple(vertex), normal=tuple(normal_v), radius_of_curvature_mm=radius, clear_radius_mm=10.0),
        ),
        observations=(),
    )
    # output direction by the exact mirror law at the vertex (chief hits the vertex)
    d_out = d_in - 2 * (d_in @ normal_v) * normal_v
    dist = 60.0
    obs_pos = vertex + dist * d_out
    proj = proj.model_copy(update={"observations": (B.observation_plane("d", position=tuple(obs_pos), normal=tuple(-d_out), radius_mm=20.0),)})
    res = run(proj)
    # at small AOI the incident beam also crosses the (virtual) plane: select the reflected beam by direction
    refl = [a for a in arrivals(res, "d") if a.beam.direction[0] * d_out[0] + a.beam.direction[1] * d_out[1] > 0.99]
    assert len(refl) == 1
    g = refl[0].beam.gaussian
    assert g.valid, res.issues
    # 1) textbook closed form
    f_t = (radius / 2.0) * math.cos(a)
    f_s = (radius / 2.0) / math.cos(a)
    q_in = q_of(w0, lam, z=100.0)  # 100 mm after the waist at the source
    for q_engine, f in ((g.q_x_mm, f_t), (g.q_y_mm, f_s)):
        q_ref = abcd_apply(free(dist) @ thin(f), q_in)
        assert q_engine[0] == pytest.approx(q_ref.real, rel=1e-11, abs=1e-8)
        assert q_engine[1] == pytest.approx(q_ref.imag, rel=1e-11)
    # 2) numerical ABCD from exact ray tracing
    trace = _mirror_trace_factory(vertex, normal_v, radius)
    mats, p0, d0, _ = numeric_abcd(trace, vertex - 30 * d_in, d_in, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    for q_engine, key in ((g.q_x_mm, "t"), (g.q_y_mm, "s")):
        m = mats[key]
        # the reference planes are 30 mm before / at the mirror; propagate the numeric matrix accordingly
        # rays are launched from the plane 30 mm upstream of the mirror, so `m` already contains that free propagation and
        # maps to the reference plane at the mirror hit point; then propagate `dist` to the observation plane.
        q_before = q_of(w0, lam, z=100.0 - 30.0)
        q_ref = abcd_apply(free(dist) @ m, q_before)
        assert q_engine[0] == pytest.approx(q_ref.real, rel=2e-6, abs=1e-5)
        assert q_engine[1] == pytest.approx(q_ref.imag, rel=2e-6)
    # x and y differ: astigmatism (unless cos = 1)
    assert abs(g.q_x_mm[0] - g.q_y_mm[0]) > 1e-3


def test_normal_incidence_mirror_is_stigmatic():
    lam, w0 = 633.0, 0.5
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.spherical_mirror("M", position=(100, 0, 20), normal=(-1, 0, 0), radius_of_curvature_mm=300.0, clear_radius_mm=10.0)),
        observations=(B.observation_plane("d", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
    )
    res = run(proj)
    # the plane at x = 40 is crossed twice (outbound and after the mirror): take the returning beam
    back = [a for a in next(o for o in res.observations if o.observation_id == "d").arrivals if a.beam.direction[0] < 0]
    assert len(back) == 1
    g = back[0].beam.gaussian
    assert g.q_x_mm == pytest.approx(g.q_y_mm, rel=1e-12)
    f = 150.0
    q_ref = abcd_apply(free(60.0) @ thin(f), q_of(w0, lam, z=100.0))  # unfolded: 100 mm to the mirror, 60 mm back to x = 40
    assert g.q_x_mm[0] == pytest.approx(q_ref.real, rel=1e-11, abs=1e-8)


# ---------------------------------------------------------------------------------------------------------------
# thick tilted singlet lens (Coddington on two refracting surfaces + propagation in glass)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tilt_deg,r1,r2", [(0.0, 120.0, math.inf), (15.0, 120.0, math.inf), (25.0, 100.0, -150.0), (-20.0, math.inf, -80.0)])
def test_golden_tilted_singlet_lens_gaussian_vs_ray_abcd(tilt_deg, r1, r2):
    lam, w0, n_glass, thick, clear = 633.0, 0.3, 1.5, 6.0, 12.0
    glass = ConstantIndexMaterial(material_id="glass", valid_wavelength_nm=(300.0, 900.0), refractive_index=n_glass)
    tilt = math.radians(tilt_deg)
    axis = np.array([math.cos(tilt), math.sin(tilt), 0.0])  # lens optical axis; beam along +x through the mid-plane centre
    centre = np.array([100.0, 0.0, 20.0])
    half = thick / 2.0
    lens = B.singlet_lens("Lens", r1_mm=r1, r2_mm=r2, thickness_mm=thick, clear_radius_mm=clear, material="glass",
                          position=tuple(centre), axis=tuple(axis))
    # ---- independent geometry (world coordinates)
    v1 = centre - half * axis
    v2 = centre + half * axis

    def surface_hit(o, d, vertex, radius, into_glass):
        if math.isinf(radius):
            t = ((vertex - o) @ axis) / (d @ axis)
            return o + t * d, axis.copy()
        c = vertex + radius * axis  # radius > 0: centre on the +axis side
        # front surface convex (radius>0) hit from outside -> near; back surface with centre on the source side (radius<0),
        # hit from inside the sphere -> far; concave cases follow the same rule: use the root whose point is nearest the vertex
        t1, t2 = _sphere_hit(o, d, c, abs(radius), "near"), _sphere_hit(o, d, c, abs(radius), "far")
        best = min((t for t in (t1, t2) if t > 1e-9), key=lambda t: np.linalg.norm(o + t * d - vertex))
        p = o + best * d
        return p, (p - c) / abs(radius)

    def trace(o, d):
        p1, n1 = surface_hit(o, d, v1, r1, True)
        d1 = _refract(d, n1, 1.0, n_glass)
        p2, n2 = surface_hit(p1, d1, v2, r2, False)
        d2 = _refract(d1, n2, n_glass, 1.0)
        return p2, d2

    d_in = np.array([1.0, 0.0, 0.0])
    src = np.array([0.0, 0.0, 20.0])
    start = src + 40.0 * d_in  # numeric ABCD input plane, 40 mm after the source (waist)
    mats, p0, d0, e_t_out = numeric_abcd(trace, start, d_in, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    # front-hit distance along the chief from `start` is inside the numeric matrix (rays start at `start`): free-space included.
    dist_out = 70.0
    obs_pos = p0 + dist_out * d0
    proj = ProjectSnapshot(
        materials=(glass,),
        components=(B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=tuple(src), direction=(1, 0, 0), waist_position_z_mm=0.0), lens),
        observations=(B.observation_plane("d", position=tuple(obs_pos), normal=tuple(-d0), radius_mm=40.0),),
    )
    res = run(proj)
    g = one_arrival(res, "d").beam.gaussian
    assert g.valid, [i.message for i in res.issues]
    q_start = q_of(w0, lam, z=40.0)
    for q_engine, key in ((g.q_x_mm, "t"), (g.q_y_mm, "s")):
        q_ref = abcd_apply(free(dist_out) @ mats[key], q_start)
        assert q_engine[0] == pytest.approx(q_ref.real, rel=5e-6, abs=1e-4), (key, tilt_deg)
        assert q_engine[1] == pytest.approx(q_ref.imag, rel=5e-6), (key, tilt_deg)
    if abs(tilt_deg) > 1.0:
        assert abs(g.q_x_mm[0] - g.q_y_mm[0]) > 1e-3  # tilt induces astigmatism
    else:
        assert g.q_x_mm == pytest.approx(g.q_y_mm, rel=1e-9)


# ---------------------------------------------------------------------------------------------------------------
# scope: twisted astigmatism is refused, round beams are re-aligned exactly
# ---------------------------------------------------------------------------------------------------------------
def _rotated_source_project(rotation_deg, wx, wy, mirror_radius=None):
    from os_optics.contracts import Pose3D

    src = B.laser("L", wavelength_nm=633.0, waist_radius_mm=wx, waist_radius_y_mm=wy, position=(0, 0, 20), direction=(1, 0, 0))
    # rotate the source frame about its propagation axis: beam axes no longer align with the plane of incidence of a 45 deg fold
    rot = Pose3D.from_axis_direction((0, 0, 20), (1, 0, 0)).to_pose()
    from os_optics.math import quat_from_axis_angle, quat_multiply

    q_extra = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(rotation_deg))  # local z is the propagation axis
    q_new = quat_multiply(np.array(src.pose.quaternion_wxyz), q_extra)
    src = src.model_copy(update={"pose": Pose3D(position_mm=(0, 0, 20), quaternion_wxyz=tuple(float(v) for v in q_new))})
    inv = 1.0 / math.sqrt(2.0)
    mirror = (B.mirror("M", position=(50, 0, 20), normal=(-inv, inv, 0), clear_radius_mm=10.0) if mirror_radius is None else
              B.spherical_mirror("M", position=(50, 0, 20), normal=(-inv, inv, 0), radius_of_curvature_mm=mirror_radius, clear_radius_mm=10.0))
    return ProjectSnapshot(
        components=(src, mirror),
        observations=(B.observation_plane("d", position=(50, 80, 20), normal=(0, -1, 0), radius_mm=30.0),),
    )


@pytest.mark.parametrize("rotation", [30.0, 45.0, 75.0])
def test_twisted_astigmatism_is_out_of_model_scope(rotation):
    res = run(_rotated_source_project(rotation, 0.3, 0.6))
    g = one_arrival(res, "d").beam.gaussian
    assert not g.valid and g.status.value == "OUT_OF_MODEL_SCOPE" and g.reason == "GAUSSIAN_TWISTED_ASTIGMATISM"
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" in issue_codes(res)
    assert res.status.value == "OUT_OF_MODEL_SCOPE"
    assert g.beam_radius_x_mm is None  # no pseudo-exact numbers


@pytest.mark.parametrize("rotation", [0.0, 90.0, -90.0, 180.0])
def test_aligned_astigmatic_axes_are_supported(rotation):
    res = run(_rotated_source_project(rotation, 0.3, 0.6))
    g = one_arrival(res, "d").beam.gaussian
    assert g.valid, res.issues
    # a plane mirror leaves q untouched and the axis *labels follow the physical axes*: x keeps its waist 0.3, y keeps 0.6
    assert g.waist_radius_x_mm == pytest.approx(0.3, rel=1e-12)
    assert g.waist_radius_y_mm == pytest.approx(0.6, rel=1e-12)
    # the beam frame is carried through the 45 deg fold (plane of incidence = xy-plane, output along +y): a source frame rotated
    # by 0/180 deg has its x axis in the plane of incidence -> global x after the fold; by +-90 deg it is out of plane -> global z
    e1 = np.array(g.e1)
    expect_axis = np.array([0.0, 0.0, 1.0]) if abs(rotation) % 180.0 == 90.0 else np.array([1.0, 0.0, 0.0])
    assert abs(abs(e1 @ expect_axis) - 1.0) < 1e-12


def test_round_beam_at_any_frame_rotation_is_supported_and_realigned():
    res = run(_rotated_source_project(37.0, 0.4, 0.4, mirror_radius=250.0))
    g = one_arrival(res, "d").beam.gaussian
    assert g.valid, res.issues
    assert g.q_x_mm != pytest.approx(g.q_y_mm, rel=1e-6)  # the tilted spherical mirror made it astigmatic in (t, s)
