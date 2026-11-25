"""Sag surfaces (``SagSurface``: conic, even asphere, cylinder; off-axis sections): ray intersection, refraction / reflection,
Gaussian beam at local principal curvatures, apertures, determinism.

Every expected value is independent of the production intersection code: closed-form geometry (paraboloid focus, thick
lens ABCD matrices, conic curvature formulas), an equivalent ``SphereSurface`` traced by the existing sphere code, or a
plain-Python high-precision reference (``decimal``, 40 digits: bisection on the sag equation, vector Snell) written below.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from os_optics import EngineConfiguration, ValidationRequest, builders as B
from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.contracts import (
    CircularAperture,
    ConstantIndexMaterial,
    EngineeringComponentSnapshot,
    MirrorInteraction,
    PlaneSurface,
    Pose3D,
    ProjectSnapshot,
    RectangularAperture,
    RefractInteraction,
    SagSurface,
    SphereSurface,
)
from os_optics.errors import EngineInputError
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import Tracer
from os_optics.optics.surfaces import intersect
from os_optics.optics.surfaces.sag import principal_curvatures
from os_optics.scene.lower import lower_scene
from os_optics.validation.ray_paths import ray_paths_document, validate_with_ray_paths

from helpers import abcd_apply, free, issue_codes, one_arrival, rayleigh_range, run

Z = (0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)
Y_UP = (0.0, 1.0, 0.0)
GLASS_N = 1.5
GLASS = ConstantIndexMaterial(material_id="glass", valid_wavelength_nm=(300.0, 2000.0), refractive_index=GLASS_N)
LAM = 633.0


# ------------------------------------------------------------------------------------------------------------- helpers
def sag(sid: str, *, z: float = 0.0, origin=None, normal=Z, u_axis=X, c: float = 0.0, k: float = 0.0, coeffs=(), cyl=False,
        aperture=None, margin: float = 0.0, pos: str = "air", neg: str = "air", interaction=None) -> SagSurface:
    return SagSurface(surface_id=sid, origin_mm=origin or (0.0, 0.0, z), normal=normal, u_axis=u_axis, curvature_per_mm=c, conic=k,
                      aspheric_coefficients=tuple(coeffs), cylindrical=cyl, aperture=aperture or CircularAperture(radius_mm=10.0),
                      mount_margin_mm=margin, medium_positive=pos, medium_negative=neg,
                      interaction=interaction or RefractInteraction())


def element(cid: str, *surfaces, role: str = "lens") -> EngineeringComponentSnapshot:
    return EngineeringComponentSnapshot(instance_id=cid, role=role, pose=Pose3D(), surfaces=tuple(surfaces))


def trace(project: ProjectSnapshot):
    lib = MaterialLibrary(project.materials, CFG)
    scene, _ = lower_scene(project, CFG, lib)
    return Tracer(scene, lib, CFG).run(), scene


def record(out, observation_id: str):
    recs = [r for r in out.arrivals if r.observation_id == observation_id]
    assert len(recs) == 1, [(r.observation_id, r.path_id) for r in out.arrivals]
    return recs[0]


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def snell(d: np.ndarray, n: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Vector Snell (plain numpy on single vectors; ``n`` any orientation)."""
    if d @ n > 0:
        n = -n
    eta, c1 = n1 / n2, -float(d @ n)
    k = 1.0 - eta * eta * (1.0 - c1 * c1)
    return eta * d + (eta * c1 - math.sqrt(k)) * n


# ------------------------------------------------------------------- high-precision reference (decimal, plain Python)
PREC = 40


def _d(x) -> Decimal:
    return Decimal(repr(float(x)))


class RefSag:
    """Axis-aligned sag surface (vertex (0, 0, z_v), axis +z, u = +x) evaluated in 40-digit decimal arithmetic."""

    def __init__(self, z_v: float, c: float, k: float, coeffs=(), cyl: bool = False):
        self.z_v, self.c, self.k, self.coeffs, self.cyl = _d(z_v), _d(c), _d(k), [_d(a) for a in coeffs], cyl

    def s(self, a: Decimal, b: Decimal) -> Decimal:
        return a * a if self.cyl else a * a + b * b

    def z(self, s: Decimal) -> Decimal | None:
        arg = 1 - (1 + self.k) * self.c * self.c * s
        if arg < 0:
            return None
        out = self.c * s / (1 + arg.sqrt())
        for j, a in enumerate(self.coeffs):
            out += a * s ** (j + 2)
        return out

    def dz_ds(self, s: Decimal) -> Decimal:
        big_s = (1 - (1 + self.k) * self.c * self.c * s).sqrt()
        out = self.c / (2 * big_s)
        for j, a in enumerate(self.coeffs):
            out += (j + 2) * a * s ** (j + 1)
        return out

    def residual(self, p, t: Decimal) -> Decimal | None:
        (ox, oy, oz), (dx, dy, dz) = p
        x, y, z = ox + t * dx, oy + t * dy, oz + t * dz
        zs = self.z(self.s(x, y))
        return None if zs is None else (z - self.z_v) - zs

    def hit(self, o, d, t_max: float, steps: int = 400):
        """First root of the sag equation along ``o + t d`` for ``1e-6 < t < t_max``: scan, then 120 bisections. Returns the
        point and the unit normal toward +z (floats) or ``None``."""
        with localcontext() as ctx:
            ctx.prec = PREC
            p = (tuple(_d(v) for v in o), tuple(_d(v) for v in d))
            t_prev, f_prev = _d(1e-6), self.residual(p, _d(1e-6))
            for i in range(1, steps + 1):
                t = _d(1e-6) + (_d(t_max) - _d(1e-6)) * i / steps
                f = self.residual(p, t)
                if f is not None and f_prev is not None and f * f_prev <= 0:
                    lo, hi, f_lo = t_prev, t, f_prev
                    for _ in range(120):
                        mid = (lo + hi) / 2
                        f_mid = self.residual(p, mid)
                        if (f_mid < 0) == (f_lo < 0):
                            lo, f_lo = mid, f_mid
                        else:
                            hi = mid
                    t_root = (lo + hi) / 2
                    x, y, z = (p[0][i] + t_root * p[1][i] for i in range(3))
                    g = self.dz_ds(self.s(x, y))
                    nx, ny = -2 * x * g, (Decimal(0) if self.cyl else -2 * y * g)
                    norm = (nx * nx + ny * ny + 1).sqrt()
                    return np.array([float(x), float(y), float(z)]), np.array([float(nx / norm), float(ny / norm), float(1 / norm)])
                t_prev, f_prev = t, f
        return None


def ref_snell(d, n, n1: float, n2: float) -> np.ndarray:
    """Vector Snell in decimal arithmetic."""
    with localcontext() as ctx:
        ctx.prec = PREC
        dd, nn = [_d(v) for v in d], [_d(v) for v in n]
        dot = sum(a * b for a, b in zip(dd, nn))
        if dot > 0:
            nn, dot = [-v for v in nn], -dot
        eta, c1 = _d(n1) / _d(n2), -dot
        k = 1 - eta * eta * (1 - c1 * c1)
        out = [eta * a + (eta * c1 - k.sqrt()) * b for a, b in zip(dd, nn)]
        norm = sum(v * v for v in out).sqrt()
        return np.array([float(v / norm) for v in out])


def ref_plane(o, d, z: float) -> np.ndarray:
    with localcontext() as ctx:
        ctx.prec = PREC
        t = (_d(z) - _d(o[2])) / _d(d[2])
        return np.array([float(_d(o[i]) + t * _d(d[i])) for i in range(3)])


# ===================================================================================================================
# 1. k = 0 without terms is a sphere: SagSurface and SphereSurface give the same trace (refraction, reflection, Fresnel
#    reflection branch, Gaussian, pupil rays, first-order transfer) at several angles of incidence.
# ===================================================================================================================
def _sphere(sid: str, vertex_z: float, radius: float, before: str, after: str, clear: float, margin: float, fresnel=False):
    r = abs(radius)
    inside, outside = (after, before) if radius > 0 else (before, after)
    return SphereSurface(surface_id=sid, center_mm=(0.0, 0.0, vertex_z + radius), radius_mm=r, cap_axis=(0.0, 0.0, -1.0 if radius > 0 else 1.0),
                         aperture_radius_mm=clear, mount_margin_mm=margin, medium_positive=outside, medium_negative=inside,
                         interaction=RefractInteraction(trace_fresnel_reflection=fresnel))


def _lens(kind: str):
    r1, r2, half, clear, margin = 40.0, -60.0, 3.0, 10.0, 1.0
    if kind == "sphere":
        return element("Lens", _sphere("front", -half, r1, "air", "N-BK7", clear, margin, fresnel=True),
                       _sphere("back", half, r2, "N-BK7", "air", clear, margin))
    ap = CircularAperture(radius_mm=clear)
    return element("Lens", sag("front", z=-half, c=1.0 / r1, aperture=ap, margin=margin, pos="N-BK7", neg="air",
                               interaction=RefractInteraction(trace_fresnel_reflection=True)),
                   sag("back", z=half, c=1.0 / r2, aperture=ap, margin=margin, pos="air", neg="N-BK7"))


def _mirror(kind: str, radius: float):
    if kind == "sphere":
        concave = radius > 0
        surf = SphereSurface(surface_id="face", center_mm=(0.0, 0.0, radius), radius_mm=abs(radius), cap_axis=(0.0, 0.0, -1.0 if concave else 1.0),
                             aperture_radius_mm=12.0, interaction=MirrorInteraction(reflective_side="negative" if concave else "positive"))
        return element("M", surf, role="mirror")
    return element("M", sag("face", c=1.0 / radius, aperture=CircularAperture(radius_mm=12.0),
                            interaction=MirrorInteraction(reflective_side="positive")), role="mirror")


def _signature(result):
    """Every number of every arrival (``None`` -> NaN), in a stable order."""
    rows = []
    for obs in result.observations:
        for a in sorted(obs.arrivals, key=lambda a: (a.beam.path_id, a.crossing_index)):
            g = a.beam.gaussian
            gauss = ([*g.q_x_mm, *g.q_y_mm, g.beam_radius_x_mm, g.beam_radius_y_mm] if g is not None and g.valid else [math.nan] * 6)
            first = list(np.ravel(a.first_order.matrix)) if a.first_order is not None else [math.nan] * 16
            rows.append([*a.beam.position_mm, *a.beam.direction, a.beam.power_W, a.beam.optical_path_length_mm, a.beam.geometric_length_mm,
                         *gauss, math.nan if a.pupil_rms_radius_mm is None else a.pupil_rms_radius_mm, *a.uv_mm, *first])
    return np.array(rows, dtype=float)


def _assert_same_trace(actual, desired):
    """Same arrivals to 1e-9 relative; the first-order matrices (central differences over 1e-6 rad / 1e-4 mm steps, so
    rounding-level differences of the two surface codes are amplified ~1e6 times) to 1e-6."""
    a, b = _signature(actual), _signature(desired)
    np.testing.assert_allclose(a[:, :18], b[:, :18], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a[:, 18:], b[:, 18:], rtol=1e-6, atol=1e-7)


# hit_y = 0: meridional incidence (the Gaussian beam stays valid); hit_y = -1.5: skew incidence on the second surface turns
# the beam's astigmatism twisted (out of model scope for both surface kinds alike)
@pytest.mark.parametrize("hit_y", [0.0, -1.5])
@pytest.mark.parametrize("theta_deg", [0.0, 8.0, 20.0])
def test_k0_sag_lens_matches_the_sphere_lens(theta_deg, hit_y):
    th = math.radians(theta_deg)
    d = (math.sin(th), 0.0, math.cos(th))
    hit = np.array([1.0, hit_y, -3.0])
    src = tuple(hit - 60.0 * np.array(d))

    def project(kind):
        return ProjectSnapshot(
            components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.3, position=src, direction=d, up=Y_UP), _lens(kind)),
            observations=(B.observation_plane("out", position=(0.0, 0.0, 80.0), normal=Z, radius_mm=60.0, up=Y_UP),
                          B.observation_plane("back", position=(0.0, 0.0, -30.0), normal=(0.0, 0.0, -1.0), radius_mm=60.0, up=Y_UP)),
        )

    sph, sg_ = run(project("sphere")), run(project("sag"))
    assert sph.status == sg_.status and issue_codes(sph) == issue_codes(sg_)
    assert len(sph.observations[1].arrivals) == 2  # the incoming beam and the Fresnel reflection of the front surface
    if hit_y == 0.0:
        assert all(a.beam.gaussian.valid for o in sg_.observations for a in o.arrivals)
    _assert_same_trace(sg_, sph)
    for p_sph, p_sag in zip(sph.paths, sg_.paths):
        assert p_sph.lineage == p_sag.lineage
        assert p_sph.optical_path_length_mm == pytest.approx(p_sag.optical_path_length_mm, rel=1e-12)


@pytest.mark.parametrize("radius", [200.0, -150.0])
@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 25.0])
def test_k0_sag_mirror_matches_the_sphere_mirror(radius, theta_deg):
    th = math.radians(theta_deg)
    d = (math.sin(th), 0.0, -math.cos(th))
    hit = np.array([1.0, 0.0, 0.0])
    src = tuple(hit - 60.0 * np.array(d))

    def project(kind):
        return ProjectSnapshot(
            components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, waist_radius_y_mm=0.8, position=src, direction=d, up=Y_UP),
                        _mirror(kind, radius)),
            observations=(B.observation_plane("out", position=(0.0, 0.0, 100.0), normal=Z, radius_mm=80.0, up=Y_UP),),
        )

    sph, sg_ = run(project("sphere")), run(project("sag"))
    assert issue_codes(sph) == issue_codes(sg_)
    g = one_arrival(sg_, "out").beam.gaussian
    assert g.valid and (theta_deg == 0.0 or abs(g.q_x_mm[0] - g.q_y_mm[0]) > 1e-3)  # tilted: astigmatic (tangential != sagittal)
    _assert_same_trace(sg_, sph)


# ===================================================================================================================
# 2. Paraboloid mirror (k = -1): an on-axis collimated beam focuses exactly at f = R/2.
# ===================================================================================================================
def _paraboloid_project(kind: str = "sag", *, w0: float = 3.0):
    big_r = 200.0
    mirror = (element("M", sag("face", c=1.0 / big_r, k=-1.0, aperture=CircularAperture(radius_mm=25.0),
                               interaction=MirrorInteraction()), role="mirror")
              if kind == "sag" else _mirror("sphere", big_r))
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0, position=(0.0, 0.0, 50.0), direction=(0.0, 0.0, -1.0), up=Y_UP),
                    mirror),
        observations=(B.observation_plane("focus", position=(0.0, 0.0, big_r / 2.0), normal=Z, radius_mm=10.0, up=Y_UP),),
    )


def test_paraboloid_mirror_focuses_a_collimated_beam_at_half_the_radius():
    out, _ = trace(_paraboloid_project())
    rec = record(out, "focus")
    assert rec.pupil_points.shape[0] == rec.pupil_total == 96
    focus = np.array([0.0, 0.0, 100.0])
    assert np.max(np.linalg.norm(rec.pupil_points - focus, axis=1)) <= 1e-9
    # every reflected ray points at the focus from its hit point; the spherical mirror of the same vertex radius does not focus
    sphere_rec = record(trace(_paraboloid_project("sphere"))[0], "focus")
    assert np.max(np.linalg.norm(sphere_rec.pupil_points - focus, axis=1)) > 1e-4


def test_paraboloid_gaussian_waist_vs_analytic_abcd_and_no_aberration_flag():
    big_r, w0 = 200.0, 3.0
    result = run(_paraboloid_project(w0=w0))
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in issue_codes(result)
    a = one_arrival(result, "focus")
    g = a.beam.gaussian
    assert g.valid
    zr = rayleigh_range(w0, LAM)
    mirror = np.array([[1.0, 0.0], [-2.0 / big_r, 1.0]])
    q_ref = abcd_apply(free(big_r / 2.0) @ mirror @ free(50.0), complex(0.0, zr))
    for q in (g.q_x_mm, g.q_y_mm):
        assert q[0] == pytest.approx(q_ref.real, rel=1e-10, abs=1e-9)
        assert q[1] == pytest.approx(q_ref.imag, rel=1e-10)
    # the waist sits at the focus within paraxial accuracy (zR >> f: offset f / (1 + (zR/f)^2)-scale)
    assert abs(g.waist_distance_x_mm) < 1e-3 * big_r / 2.0
    assert a.pupil_rms_radius_mm == pytest.approx(0.0, abs=1e-9)


# ===================================================================================================================
# 3. 90-degree off-axis paraboloid: parent focal length f, aperture decentred by 2f; reflected focal length 2f.
# ===================================================================================================================
F_PARENT = 50.0


def _oap_project(*, w0x: float = 2.0, w0y: float | None = None, aperture=None):
    f = F_PARENT
    oap = sag("face", c=1.0 / (2.0 * f), k=-1.0, aperture=aperture or CircularAperture(radius_mm=12.7, center_offset_mm=(2.0 * f, 0.0)),
              interaction=MirrorInteraction())
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0x, waist_radius_y_mm=w0y, position=(2.0 * f, 0.0, 200.0),
                            direction=(0.0, 0.0, -1.0), up=Y_UP),
                    element("OAP", oap, role="mirror")),
        observations=(B.observation_plane("focus", position=(0.0, 0.0, f), normal=(-1.0, 0.0, 0.0), radius_mm=10.0, up=Y_UP),),
    )


def test_off_axis_paraboloid_focuses_at_the_parent_focus_with_reflected_focal_length_2f():
    f = F_PARENT
    out, _ = trace(_oap_project())
    rec = record(out, "focus")
    focus = np.array([0.0, 0.0, f])
    np.testing.assert_allclose(rec.position, focus, atol=1e-12)
    np.testing.assert_allclose(rec.direction, (-1.0, 0.0, 0.0), atol=1e-15)
    assert rec.length_mm == pytest.approx((200.0 - f) + 2.0 * f, rel=1e-14)  # source -> mirror, then the reflected focal length
    assert rec.pupil_points.shape[0] == rec.pupil_total
    assert np.max(np.linalg.norm(rec.pupil_points - focus, axis=1)) <= 1e-9


@pytest.mark.parametrize("w0x,w0y", [(2.0, None), (1.2, 2.5)])
def test_off_axis_paraboloid_gaussian_tangential_and_sagittal_vs_reference(w0x, w0y):
    f = F_PARENT
    result = run(_oap_project(w0x=w0x, w0y=w0y))
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" not in issue_codes(result)
    g = one_arrival(result, "focus").beam.gaussian
    assert g.valid
    # independent reference: at the hit the paraboloid's principal curvatures (meridional c / (1 + c^2 rho^2)^(3/2), parallel
    # c / (1 + c^2 rho^2)^(1/2), c = 1/(2f), rho = 2f) at 45 deg incidence give mirror powers 2 k_m / cos45 = 2 k_p cos45 = 1/(2f):
    # both axes focus with the reflected focal length 2f.
    c = 1.0 / (2.0 * f)
    k_m, k_p = c / (1.0 + (c * 2.0 * f) ** 2) ** 1.5, c / math.sqrt(1.0 + (c * 2.0 * f) ** 2)
    cos45 = math.sqrt(0.5)
    p_t, p_s = 2.0 * k_m / cos45, 2.0 * k_p * cos45
    assert p_t == pytest.approx(1.0 / (2.0 * f), rel=1e-15) and p_s == pytest.approx(1.0 / (2.0 * f), rel=1e-15)
    for q, w0 in ((g.q_x_mm, w0x), (g.q_y_mm, w0y or w0x)):
        q_in = complex(200.0 - f, rayleigh_range(w0, LAM))
        q_ref = abcd_apply(free(2.0 * f) @ np.array([[1.0, 0.0], [-p_t, 1.0]]), q_in)
        assert q[0] == pytest.approx(q_ref.real, rel=1e-10, abs=1e-9)
        assert q[1] == pytest.approx(q_ref.imag, rel=1e-10)
    # waist position: the Gaussian waist distance before the plane matches the reference
    q_ref = abcd_apply(free(2.0 * f) @ np.array([[1.0, 0.0], [-p_t, 1.0]]), complex(200.0 - f, rayleigh_range(w0x, LAM)))
    assert g.waist_distance_x_mm == pytest.approx(q_ref.real, rel=1e-9, abs=1e-9)


def test_off_axis_paraboloid_skew_incidence_is_twisted_astigmatism_even_for_a_round_beam():
    """A beam tilted out of the meridional plane meets the OAP with a plane of incidence that is not along the surface's
    principal directions (k1 != k2 there): the reflected beam is generally astigmatic, outside the x/y model."""
    f = F_PARENT

    def project(direction):
        oap = sag("face", c=1.0 / (2.0 * f), k=-1.0, aperture=CircularAperture(radius_mm=12.7, center_offset_mm=(2.0 * f, 0.0)),
                  interaction=MirrorInteraction())
        hit = np.array([2.0 * f, 0.0, f])
        return ProjectSnapshot(
            components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=1.0, position=tuple(hit - 150.0 * np.array(direction)),
                                direction=direction, up=Y_UP),
                        element("OAP", oap, role="mirror")),
            observations=(B.observation_plane("far", position=(-100.0, 0.0, f), normal=(-1.0, 0.0, 0.0), radius_mm=80.0, up=Y_UP),),
        )

    skew = run(project(tuple(unit((0.0, 0.05, -1.0)))))
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" in issue_codes(skew)
    meridional = run(project(tuple(unit((0.05, 0.0, -1.0)))))  # tilted inside the meridional plane: aligned
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" not in issue_codes(meridional)
    assert one_arrival(meridional, "far").beam.gaussian.valid


def test_curved_beam_splitter_reflects_like_the_same_sag_mirror():
    from os_optics.contracts import BeamSplitterInteraction, ConstantCoating

    def project(interaction):
        face = sag("face", c=1.0 / 300.0, k=-0.4, aperture=CircularAperture(radius_mm=12.0), interaction=interaction)
        th = math.radians(12.0)
        d = (math.sin(th), 0.0, -math.cos(th))
        return ProjectSnapshot(
            components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.7, position=tuple(-80.0 * np.array(d) + np.array([2.0, 0.0, 0.0])),
                                direction=d, up=Y_UP),
                        element("S", face, role="mirror")),
            observations=(B.observation_plane("r", position=(0.0, 0.0, 120.0), normal=Z, radius_mm=80.0, up=Y_UP),
                          B.observation_plane("t", position=(0.0, 0.0, -120.0), normal=(0.0, 0.0, -1.0), radius_mm=80.0, up=Y_UP)),
        )

    half = ConstantCoating(reflectance_s=0.5, reflectance_p=0.5, transmittance_s=0.5, transmittance_p=0.5)
    split = run(project(BeamSplitterInteraction(coating=half)))
    mirror = run(project(MirrorInteraction(coating=ConstantCoating(reflectance_s=0.5, reflectance_p=0.5, transmittance_s=0.0,
                                                                   transmittance_p=0.0))))
    a, b = one_arrival(split, "r"), one_arrival(mirror, "r")
    np.testing.assert_allclose(a.beam.position_mm, b.beam.position_mm, atol=1e-12)
    np.testing.assert_allclose(a.beam.direction, b.beam.direction, atol=1e-15)
    assert a.beam.power_W == pytest.approx(b.beam.power_W, rel=1e-12)
    for qa, qb in ((a.beam.gaussian.q_x_mm, b.beam.gaussian.q_x_mm), (a.beam.gaussian.q_y_mm, b.beam.gaussian.q_y_mm)):
        assert qa == pytest.approx(qb, rel=1e-12)
    assert a.beam.gaussian.q_x_mm != pytest.approx(a.beam.gaussian.q_y_mm, rel=1e-6)  # tilted curved reflection: astigmatic
    t = one_arrival(split, "t")  # the transmitted branch crosses the thin splitter undeviated
    np.testing.assert_allclose(t.beam.direction, (math.sin(math.radians(12.0)), 0.0, -math.cos(math.radians(12.0))), atol=1e-15)


def test_off_axis_paraboloid_principal_curvatures_and_directions():
    f = F_PARENT
    _, scene = trace(_oap_project())
    surf = scene.surfaces[scene.surface_index["OAP/face"]]
    c, rho = 1.0 / (2.0 * f), 2.0 * f
    geo = principal_curvatures(surf, np.array([rho, 0.0, c * rho * rho / 2.0]))
    assert geo.k1 == pytest.approx(c / (1.0 + c * c * rho * rho) ** 1.5, rel=1e-14)
    assert geo.k2 == pytest.approx(c / math.sqrt(1.0 + c * c * rho * rho), rel=1e-14)
    np.testing.assert_allclose(geo.normal, unit((-c * rho, 0.0, 1.0)), atol=1e-15)
    np.testing.assert_allclose(geo.dir1, unit((1.0, 0.0, c * rho)), atol=1e-15)  # meridian
    np.testing.assert_allclose(geo.dir2, (0.0, 1.0, 0.0), atol=1e-15)  # parallel circle


# ===================================================================================================================
# 4. Plano-convex cylindrical lens: line focus at the paraxial focal length of the curved axis, no power along the straight
#    axis; astigmatic Gaussian per axis vs analytic thick-lens ABCD; rotated cylinder vs astigmatic / round beam.
# ===================================================================================================================
CYL_R, CYL_T, CYL_SRC = 50.0, 5.0, 50.0


def _cyl_matrices():
    """Thick plano-convex lens (curved face first, radius CYL_R, centre ahead): curved axis / straight axis ABCD from the
    first vertex to just after the flat face, in (height, angle in the local medium)."""
    n = GLASS_N

    def lens(radius):
        curved = np.array([[1.0, 0.0], [0.0 if math.isinf(radius) else (1.0 - n) / (n * radius), 1.0 / n]])
        return np.array([[1.0, 0.0], [0.0, n]]) @ free(CYL_T) @ curved

    return lens(CYL_R), lens(math.inf)


def _cyl_project(*, w0x=0.4, w0y=0.6, rotation_deg=0.0, plane_z=None):
    rot = math.radians(rotation_deg)
    u = (math.cos(rot), math.sin(rot), 0.0)
    m_u, _ = _cyl_matrices()
    bfl = -m_u[0, 0] / m_u[1, 0]
    front = sag("front", z=0.0, u_axis=u, c=1.0 / CYL_R, cyl=True, aperture=RectangularAperture(half_width_mm=8.0, half_height_mm=8.0),
                pos="glass", neg="air")
    back = PlaneSurface(surface_id="back", origin_mm=(0.0, 0.0, CYL_T), normal=Z, aperture=CircularAperture(radius_mm=15.0),
                        medium_positive="air", medium_negative="glass", interaction=RefractInteraction())
    return ProjectSnapshot(
        materials=(GLASS,),
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0x, waist_radius_y_mm=w0y, position=(0.0, 0.0, -CYL_SRC), direction=Z,
                            up=Y_UP),
                    element("Cyl", front, back)),
        observations=(B.observation_plane("line", position=(0.0, 0.0, plane_z if plane_z is not None else CYL_T + bfl), normal=Z,
                                          radius_mm=20.0, up=Y_UP),),
    ), bfl


def test_cylindrical_lens_line_focus_first_order_matches_thick_lens_abcd():
    project, bfl = _cyl_project()
    m_u, m_v = _cyl_matrices()
    assert -1.0 / m_u[1, 0] == pytest.approx(CYL_R / (GLASS_N - 1.0), rel=1e-14)  # EFL of the curved axis: R / (n - 1)
    result = run(project)
    a = one_arrival(result, "line")
    total_u = free(bfl) @ m_u @ free(CYL_SRC)
    total_v = free(bfl) @ m_v @ free(CYL_SRC)
    m = np.array(a.first_order.matrix)
    expected = np.zeros((4, 4))
    expected[np.ix_([0, 2], [0, 2])] = total_u
    expected[np.ix_([1, 3], [1, 3])] = total_v
    np.testing.assert_allclose(m, expected, atol=2e-6)
    assert abs(m[0, 0]) < 1e-6  # line focus: parallel rays displaced along the curved axis converge
    assert m[1, 1] == pytest.approx(1.0, abs=1e-9) and abs(m[3, 1]) < 1e-9  # straight axis: no focusing at all


def test_cylindrical_lens_pupil_rays_keep_their_straight_axis_coordinate():
    project, _ = _cyl_project(w0x=0.4, w0y=0.6)
    out, _ = trace(project)
    rec = record(out, "line")
    assert rec.pupil_points.shape[0] == rec.pupil_total
    launch_v = rec.pupil_launch[:, 1] * 0.6  # collimated (waist at the source): the launch offsets along y in mm
    np.testing.assert_allclose(rec.pupil_points[:, 1], launch_v, atol=1e-12)
    np.testing.assert_allclose(rec.pupil_directions[:, 1], 0.0, atol=1e-15)
    # along the curved axis: an exact 2D trace (circle of radius R centred R behind the vertex, Snell, flat face, plane)
    plane_z = float(rec.position[2])
    for (x, y, _), launch in zip(rec.pupil_points, rec.pupil_launch):
        a = launch[0] * 0.4
        z1 = CYL_R - math.sqrt(CYL_R**2 - a * a)
        d1 = snell(np.array([0.0, 0.0, 1.0]), np.array([a, 0.0, z1 - CYL_R]) / CYL_R, 1.0, GLASS_N)
        x2 = a + d1[0] / d1[2] * (CYL_T - z1)
        d2 = snell(d1, np.array([0.0, 0.0, 1.0]), GLASS_N, 1.0)
        assert x == pytest.approx(x2 + d2[0] / d2[2] * (plane_z - CYL_T), abs=1e-12)


def test_cylindrical_lens_astigmatic_gaussian_per_axis_vs_abcd():
    w0x, w0y = 0.4, 0.6
    project, bfl = _cyl_project(w0x=w0x, w0y=w0y)
    g = one_arrival(run(project), "line").beam.gaussian
    assert g.valid
    m_u, m_v = _cyl_matrices()
    np.testing.assert_allclose(np.abs(g.e1), (1.0, 0.0, 0.0), atol=1e-15)
    for q, w0, mat in ((g.q_x_mm, w0x, m_u), (g.q_y_mm, w0y, m_v)):
        q_ref = abcd_apply(free(bfl) @ mat, complex(CYL_SRC, rayleigh_range(w0, LAM)))
        assert q[0] == pytest.approx(q_ref.real, rel=1e-10, abs=1e-9)
        assert q[1] == pytest.approx(q_ref.imag, rel=1e-10)


def test_rotated_cylinder_twists_an_astigmatic_beam_but_handles_a_round_one():
    project, _ = _cyl_project(w0x=0.4, w0y=0.6, rotation_deg=30.0)
    result = run(project)
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" in issue_codes(result)
    assert not one_arrival(result, "line").beam.gaussian.valid
    # a round beam has no preferred axes: its frame is turned onto the cylinder's axes and stays x/y independent
    w0 = 0.5
    project, bfl = _cyl_project(w0x=w0, w0y=w0, rotation_deg=30.0)
    result = run(project)
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" not in issue_codes(result)
    g = one_arrival(result, "line").beam.gaussian
    assert g.valid
    u_rot = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0])
    assert abs(float(np.array(g.e1) @ u_rot)) == pytest.approx(1.0, abs=1e-12)
    m_u, m_v = _cyl_matrices()
    for q, mat in ((g.q_x_mm, m_u), (g.q_y_mm, m_v)):
        q_ref = abcd_apply(free(bfl) @ mat, complex(CYL_SRC, rayleigh_range(w0, LAM)))
        assert q[0] == pytest.approx(q_ref.real, rel=1e-10, abs=1e-9)
        assert q[1] == pytest.approx(q_ref.imag, rel=1e-10)


# ===================================================================================================================
# 5. Even-asphere singlet: a plano-asphere (flat face first) whose A4..A10 terms bend a k = -0.6 base conic into the
#    Taylor expansion (to rho^10) of the aberration-free hyperboloid k = -n^2 (vertex radius 10 mm, n = 1.5, f = 20 mm).
#    Coefficient magnitudes are those of molded collimating aspheres (A4 ~ 2e-4, A6 ~ 9e-7, A8 ~ 8e-9, A10 ~ 7e-11).
# ===================================================================================================================
ASPH_C, ASPH_K0, ASPH_T = -0.1, -0.6, 4.0


def _asphere_coefficients() -> tuple[float, ...]:
    # conic sag series: c s/2 + (1+k) c^3 s^2/8 + (1+k)^2 c^5 s^3/16 + 5 (1+k)^3 c^7 s^4/128 + 7 (1+k)^4 c^9 s^5/256 + ...
    series = (1.0 / 8.0, 1.0 / 16.0, 5.0 / 128.0, 7.0 / 256.0)
    kt = -GLASS_N**2
    return tuple(series[m - 2] * ASPH_C ** (2 * m - 1) * ((1.0 + kt) ** (m - 1) - (1.0 + ASPH_K0) ** (m - 1)) for m in range(2, 6))


def _asphere_lens(aspheric: bool = True, margin: float = 0.5):
    front = PlaneSurface(surface_id="front", origin_mm=(0.0, 0.0, 0.0), normal=Z, aperture=CircularAperture(radius_mm=6.0),
                         medium_positive="glass", medium_negative="air", interaction=RefractInteraction())
    back = sag("back", z=ASPH_T, c=ASPH_C, k=ASPH_K0 if aspheric else 0.0, coeffs=_asphere_coefficients() if aspheric else (),
               aperture=CircularAperture(radius_mm=5.0), margin=margin, pos="air", neg="glass")
    return element("Asph", front, back)


def _asphere_project(*, w0=1.0, theta_deg=0.0, aspheric=True, plane_z=ASPH_T + 20.0):
    th = math.radians(theta_deg)
    d = (math.sin(th), 0.0, math.cos(th))
    src = (-20.0 * d[0], 0.0, -20.0 * d[2])
    return ProjectSnapshot(
        materials=(GLASS,),
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0, position=src, direction=d, up=Y_UP), _asphere_lens(aspheric)),
        observations=(B.observation_plane("f", position=(0.0, 0.0, plane_z), normal=Z, radius_mm=15.0, up=Y_UP),),
    )


def _ref_asphere() -> RefSag:
    return RefSag(ASPH_T, ASPH_C, ASPH_K0, _asphere_coefficients())


def test_asphere_intersections_and_normals_match_the_high_precision_reference():
    _, scene = trace(_asphere_project())
    surf = scene.surfaces[scene.surface_index["Asph/back"]]
    ref = _ref_asphere()
    rays = []
    for a0 in (-4.6, -2.0, 0.0, 1.3, 3.7):
        for b0 in (-2.5, 0.0, 1.9):
            for tilt in (-0.15, 0.0, 0.2):
                rays.append(((a0, b0, 0.5), unit((tilt, 0.5 * tilt, 1.0))))  # inside the glass, toward the asphere
    rays.append(((3.0, 1.0, 30.0), unit((-0.1, 0.0, -1.0))))  # from the air side, travelling back toward the lens
    o = np.array([r[0] for r in rays])
    d = np.array([r[1] for r in rays])
    hits = intersect(surf, o, d, CFG)
    for i, (oi, di) in enumerate(rays):
        expected = ref.hit(oi, di, 40.0)
        assert expected is not None
        point, normal = expected
        rho = math.hypot(point[0], point[1])
        if rho > 5.5:  # beyond the clear aperture (5) and its 0.5 mm mount ring: the ray passes by
            assert np.isinf(hits.t[i])
            continue
        assert np.isfinite(hits.t[i]) and bool(hits.inside[i]) == (rho <= 5.0)
        np.testing.assert_allclose(hits.point[i], point, atol=1e-10)
        np.testing.assert_allclose(hits.normal[i], normal, atol=1e-12)
        assert (hits.u[i], hits.v[i]) == pytest.approx((point[0], point[1]), abs=1e-10)


@pytest.mark.parametrize("theta_deg", [0.0, 2.0])
def test_asphere_traced_pupil_rays_match_the_high_precision_reference(theta_deg):
    w0 = 1.0
    out, _ = trace(_asphere_project(w0=w0, theta_deg=theta_deg))
    rec = record(out, "f")
    th = math.radians(theta_deg)
    d0 = np.array([math.sin(th), 0.0, math.cos(th)])
    src = -20.0 * d0
    ref = _ref_asphere()
    launches = np.vstack([[0.0, 0.0], rec.pupil_launch])
    points = np.vstack([rec.position, rec.pupil_points])
    directions = np.vstack([rec.direction, rec.pupil_directions])
    assert rec.pupil_points.shape[0] == rec.pupil_total
    for launch, point, direction in zip(launches, points, directions):
        o = src + w0 * (launch[0] * rec.source_e1 + launch[1] * rec.source_e2)
        p1 = ref_plane(o, d0, 0.0)
        d1 = ref_snell(d0, np.array([0.0, 0.0, 1.0]), 1.0, GLASS_N)
        p2, n2 = ref.hit(p1, d1, 10.0)
        d2 = ref_snell(d1, n2, GLASS_N, 1.0)
        p3 = ref_plane(p2, d2, ASPH_T + 20.0)
        np.testing.assert_allclose(point, p3, atol=1e-9)
        np.testing.assert_allclose(direction, d2, atol=1e-12)


def test_asphere_spherical_aberration_is_far_below_the_equivalent_spherical_lens():
    asph = run(_asphere_project(w0=1.0))
    sphere = run(_asphere_project(w0=1.0, aspheric=False))
    a_rms = one_arrival(asph, "f").pupil_rms_radius_mm
    s_rms = one_arrival(sphere, "f").pupil_rms_radius_mm
    assert s_rms > 1e-3 and a_rms < 1e-6 * s_rms  # sub-nanometre vs ~8 micrometres
    # the Engine's aberration check: the spherical lens is aberration dominated at its focus, the asphere is not
    assert "GAUSSIAN_ABERRATION_DOMINATED" in issue_codes(sphere)
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in issue_codes(asph)
    g = one_arrival(asph, "f").beam.gaussian
    assert g.valid and abs(g.waist_distance_x_mm) < 1e-3


def test_principal_curvatures_of_the_asphere_match_finite_differences_of_the_reference_sag():
    _, scene = trace(_asphere_project())
    surf = scene.surfaces[scene.surface_index["Asph/back"]]
    ref = _ref_asphere()
    for a, b in ((0.0, 0.0), (2.0, 0.0), (1.5, -2.5), (-3.0, 3.2)):
        with localcontext() as ctx:
            ctx.prec = PREC
            h = Decimal("1e-5")
            da, db = _d(a), _d(b)

            def f(x, y):
                return ref.z(ref.s(x, y))

            fa = (f(da + h, db) - f(da - h, db)) / (2 * h)
            fb = (f(da, db + h) - f(da, db - h)) / (2 * h)
            faa = (f(da + h, db) - 2 * f(da, db) + f(da - h, db)) / (h * h)
            fbb = (f(da, db + h) - 2 * f(da, db) + f(da, db - h)) / (h * h)
            fab = (f(da + h, db + h) - f(da + h, db - h) - f(da - h, db + h) + f(da - h, db - h)) / (4 * h * h)
            z = f(da, db)
        fa, fb, faa, fbb, fab = (float(v) for v in (fa, fb, faa, fbb, fab))
        w = math.sqrt(1.0 + fa * fa + fb * fb)
        first = np.array([[1.0 + fa * fa, fa * fb], [fa * fb, 1.0 + fb * fb]])
        second = np.array([[faa, fab], [fab, fbb]]) / w
        expected = np.sort(np.linalg.eigvals(np.linalg.solve(first, second)).real)
        geo = principal_curvatures(surf, np.array([a, b, ASPH_T + float(z)]))
        np.testing.assert_allclose(np.sort([geo.k1, geo.k2]), expected, rtol=1e-7)
        np.testing.assert_allclose(geo.normal, unit((-fa, -fb, 1.0)), atol=1e-9)
        assert abs(float(geo.dir1 @ geo.normal)) < 1e-15 and abs(float(geo.dir2 @ geo.normal)) < 1e-15


# ===================================================================================================================
# 6. Misses, sag-domain edge, rectangular / decentred apertures, mount ring, grazing incidence, TIR inside a sag-bounded glass.
# ===================================================================================================================
def _lowered(surface: SagSurface, materials=()):
    project = ProjectSnapshot(materials=materials, components=(element("E", surface),))
    lib = MaterialLibrary(project.materials, CFG)
    scene, _ = lower_scene(project, CFG, lib)
    return scene.surfaces[0]


@pytest.mark.parametrize("coeffs", [(), (1e-7,)])
def test_misses_and_the_sag_domain_edge(coeffs):
    # sphere-like profile, domain rho <= 10; clear aperture 9.5 + a 1 mm mount ring reaching beyond the domain
    surf = _lowered(sag("s", c=0.1, coeffs=coeffs, aperture=CircularAperture(radius_mm=9.5), margin=1.0))
    o = np.array([[0.0, 0.0, -20.0], [9.8, 0.0, -20.0], [10.2, 0.0, -20.0], [12.0, 0.0, -20.0], [0.0, 0.0, -20.0], [2.0, 0.0, -20.0]])
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    hits = intersect(surf, o, d, CFG)
    assert hits.t[0] == pytest.approx(20.0, abs=1e-12) and bool(hits.inside[0])
    z_ring = 10.0 - math.sqrt(100.0 - 9.8**2) + (1e-7 * 9.8**4 if coeffs else 0.0)
    assert hits.t[1] == pytest.approx(20.0 + z_ring, abs=1e-9) and not bool(hits.inside[1])  # on the mount ring
    assert np.isinf(hits.t[2:]).all()  # beyond the domain, beyond the support, pointing away, parallel to the vertex plane
    assert np.isfinite(hits.normal).all() and np.isfinite(hits.point).all()


def test_lowering_rejects_an_aperture_beyond_the_domain_and_takes_the_in_plane_part_of_a_u_axis():
    """A ``u_axis`` off the surface plane is taken by its component in the plane (its azimuth about the normal), which is
    the surface lowered with that in-plane direction; only a ``u_axis`` along the normal, which has no such component, is
    refused."""
    with pytest.raises(EngineInputError, match="domain"):
        _lowered(sag("s", c=0.1, aperture=CircularAperture(radius_mm=10.5)))
    tilted = _lowered(sag("s", c=0.1, cyl=True, u_axis=tuple(unit((1.0, 1.0, 0.3))), aperture=CircularAperture(radius_mm=5.0)))
    in_plane = _lowered(sag("s", c=0.1, cyl=True, u_axis=tuple(unit((1.0, 1.0, 0.0))), aperture=CircularAperture(radius_mm=5.0)))
    np.testing.assert_allclose(tilted.u_axis, in_plane.u_axis, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(tilted.v_axis, in_plane.v_axis, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(tilted.u_axis, [math.sqrt(0.5), math.sqrt(0.5), 0.0], rtol=0.0, atol=1e-15)
    with pytest.raises(EngineInputError, match="u_axis is parallel to the normal"):
        _lowered(sag("s", c=0.1, u_axis=(0.0, 0.0, -2.0), aperture=CircularAperture(radius_mm=5.0)))
    _lowered(sag("s", c=0.1, aperture=CircularAperture(radius_mm=10.0)))  # exactly the hemisphere: accepted


def _notices(surface: SagSurface):
    project = ProjectSnapshot(components=(element("E", surface),))
    scene, _ = lower_scene(project, CFG, MaterialLibrary(project.materials, CFG))
    return [n for n in scene.notices if n.code.value == "U_AXIS_PROJECTED"]


def test_a_projected_u_axis_is_stated_in_a_notice_with_the_angle_removed():
    """A ``u_axis`` tilted out of the surface plane is projected onto it (the test above); the run states it: a notice
    (``U_AXIS_PROJECTED``, NOT_APPLICABLE: it counts toward no status) at the surface, with the angle between the stated axis
    and the plane, asin(|u . n|) for a unit u (independent: 0.3 / |(1, 1, 0.3)|). A tilt at rounding level (up to
    ``unit_vector_tol``, the level below which a stated direction is used as given) gives none, nor does an in-plane axis."""
    notice, = _notices(sag("s", c=0.1, cyl=True, u_axis=(1.0, 1.0, 0.3), aperture=CircularAperture(radius_mm=5.0)))
    expected = math.degrees(math.asin(0.3 / math.sqrt(2.09)))
    assert notice.status.value == "NOT_APPLICABLE" and notice.location == "component:E/surface:s"
    assert notice.numerical_context["tilt_deg"] == pytest.approx(expected, rel=1e-12)
    assert notice.numerical_context["tolerance_rad"] == CFG.unit_vector_tol
    assert f"u_axis lies {expected:.6g} deg out of the surface plane" in notice.message
    assert _notices(sag("s", c=0.1, u_axis=(1.0, 0.0, 1e-12), aperture=CircularAperture(radius_mm=5.0))) == []
    assert _notices(sag("s", c=0.1, u_axis=(1.0, 0.0, 0.0), aperture=CircularAperture(radius_mm=5.0))) == []
    # (a run carries it among its issues, as a notice)
    project = ProjectSnapshot(components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.2, position=(0.0, 0.0, -50.0), direction=Z, up=Y_UP),
                                          element("E", sag("s", c=0.01, u_axis=(1.0, 0.0, 0.5), aperture=CircularAperture(radius_mm=5.0)))))
    issue, = [i for i in run(project).issues if i.code.value == "U_AXIS_PROJECTED"]
    assert issue.numerical_context["tilt_deg"] == pytest.approx(math.degrees(math.atan(0.5)), rel=1e-12)


def test_a_projected_aperture_u_axis_of_a_plane_is_stated_in_the_same_notice():
    """A plane's ``aperture_u_axis`` (the u direction of a rectangular or polygonal opening) off the surface plane is projected
    onto it the same way: the same notice, naming the field, with asin(|u . n|) / |u| as the angle (0.3 / |(1, 1, 0.3)|);
    an in-plane axis or a rounding-level tilt gives none."""

    def notices(u_axis):
        plane = PlaneSurface(surface_id="p", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                             aperture=RectangularAperture(half_width_mm=4.0, half_height_mm=2.0), aperture_u_axis=u_axis,
                             interaction=MirrorInteraction())
        project = ProjectSnapshot(components=(element("E", plane, role="mirror"),))
        scene, _ = lower_scene(project, CFG, MaterialLibrary(project.materials, CFG))
        return [n for n in scene.notices if n.code.value == "U_AXIS_PROJECTED"]

    notice, = notices((1.0, 1.0, 0.3))
    expected = math.degrees(math.asin(0.3 / math.sqrt(2.09)))
    assert notice.status.value == "NOT_APPLICABLE" and notice.location == "component:E/surface:p"
    assert notice.numerical_context["tilt_deg"] == pytest.approx(expected, rel=1e-12)
    assert f"aperture_u_axis lies {expected:.6g} deg out of the surface plane" in notice.message
    assert notices((2.0, 0.0, 2e-12)) == [] and notices((1.0, 1.0, 0.0)) == []


@pytest.mark.parametrize("cyl", [False, True])
def test_rectangular_decentred_aperture_and_mount_ring_on_a_sag_surface(cyl):
    ap = RectangularAperture(half_width_mm=4.0, half_height_mm=2.0, center_offset_mm=(1.0, -0.5))
    surf = _lowered(sag("s", c=0.02, k=-0.5, cyl=cyl, aperture=ap, margin=0.5))
    pts = [(4.8, -2.4), (5.3, -0.5), (1.0, 1.8), (5.6, 0.0), (1.0, 1.9), (-3.2, -0.5)]
    o = np.array([[a, b, -10.0] for a, b in pts])
    d = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    hits = intersect(surf, o, d, CFG)
    hit = np.isfinite(hits.t)
    np.testing.assert_array_equal(hit, [True, True, True, False, True, True])
    np.testing.assert_array_equal(hits.inside[hit], [True, False, False, False, False])
    np.testing.assert_allclose(np.column_stack([hits.u, hits.v])[hit], np.array(pts)[hit], atol=1e-12)
    for i in np.flatnonzero(hit):
        a, b = pts[i]
        s = a * a if cyl else a * a + b * b
        assert hits.point[i, 2] == pytest.approx(0.02 * s / (1.0 + math.sqrt(1.0 - 0.5 * 0.02**2 * s)), abs=1e-12)


def test_chief_ray_on_the_mount_ring_of_a_sag_mirror_is_clipped():
    oap = sag("face", c=0.01, k=-1.0, aperture=CircularAperture(radius_mm=5.0, center_offset_mm=(100.0, 0.0)), margin=2.0,
              interaction=MirrorInteraction())
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.2, position=(106.0, 0.0, 200.0), direction=(0.0, 0.0, -1.0), up=Y_UP),
                    element("OAP", oap, role="mirror")),
    )
    result = run(project)
    assert "CLIPPED" in issue_codes(result)
    assert result.paths[0].termination == "CLIPPED" and result.paths[0].termination_surface_id == "OAP/face"


def test_grazing_incidence_on_a_steep_cylinder_terminates():
    a = 10.0 * math.sin(math.radians(89.95))  # AOI 89.95 deg on a radius-10 cylinder hit parallel to its axis
    mirror = sag("face", c=0.1, cyl=True, aperture=RectangularAperture(half_width_mm=10.0, half_height_mm=5.0), interaction=MirrorInteraction())
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=1e-3, position=(a, 0.0, 50.0), direction=(0.0, 0.0, -1.0), up=Y_UP),
                    element("M", mirror, role="mirror")),
    )
    result = run(project)
    assert "GRAZING_INCIDENCE" in issue_codes(result)
    assert result.paths[0].termination == "GRAZING_INCIDENCE"


def test_total_internal_reflection_inside_a_sag_bounded_glass():
    # glass between a flat entrance (z = 0) and a convex sag exit (vertex z = 10, centre of curvature at the origin, R = 10):
    # a ray parallel to the axis at x = 8 meets the exit at (8, 0, 6) with normal (0.8, 0, 0.6): AOI 53.13 deg > 41.81 deg
    front = PlaneSurface(surface_id="front", origin_mm=(0.0, 0.0, 0.0), normal=Z, aperture=CircularAperture(radius_mm=12.0),
                         medium_positive="glass", medium_negative="air", interaction=RefractInteraction())
    back = sag("back", z=10.0, c=-0.1, aperture=CircularAperture(radius_mm=9.0), pos="air", neg="glass")
    project = ProjectSnapshot(
        materials=(GLASS,),
        components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.05, position=(8.0, 0.0, -20.0), direction=Z, up=Y_UP),
                    element("Block", front, back)),
    )
    result = run(project)
    path = result.paths[0]
    assert path.lineage[:2] == ("Block/front:refract", "Block/back:tir")
    seg_in, seg_out = path.segments[1], path.segments[2]
    np.testing.assert_allclose(seg_in.end_mm, (8.0, 0.0, 6.0), atol=1e-12)
    assert seg_in.end_event == "tir" and seg_in.angle_of_incidence_deg == pytest.approx(math.degrees(math.asin(0.8)), abs=1e-10)
    n = np.array([0.8, 0.0, 0.6])
    d_ref = np.array([0.0, 0.0, 1.0]) - 2.0 * n[2] * n  # reflection law
    direction = (np.array(seg_out.end_mm) - np.array(seg_out.start_mm)) / seg_out.length_mm
    np.testing.assert_allclose(direction, d_ref, atol=1e-12)
    assert seg_out.medium_id == "glass" and seg_in.power_end_W == pytest.approx(seg_out.power_start_W, rel=1e-15)


# ===================================================================================================================
# 7. Determinism: the same Project gives the same result hash, trace hash and ray-path document twice.
# ===================================================================================================================
@pytest.mark.parametrize("name", ["asphere", "oap", "cylinder"])
def test_sag_traces_are_deterministic(name):
    def make():
        return {"asphere": lambda: _asphere_project(theta_deg=2.0), "oap": _oap_project, "cylinder": lambda: _cyl_project()[0]}[name]()

    first = validate_with_ray_paths(ValidationRequest(project=make(), configuration=EngineConfiguration()))
    second = validate_with_ray_paths(ValidationRequest(project=make(), configuration=EngineConfiguration()))
    assert first[0].manifest.result_hash == second[0].manifest.result_hash
    assert first[0].manifest.trace_hash == second[0].manifest.trace_hash
    assert ray_paths_document(first[1]) == ray_paths_document(second[1])
