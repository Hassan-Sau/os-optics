"""Gradient-index media beyond the radial rod with plane faces across its axis: the axial medium (``n(z) = n0 + a1 z + a2 z^2 +
a3 z^3``, ``optics/rays/gradient.py`` ``AxialLegs``) and rods whose end faces are slanted planes (a ``SURFACE_SET`` with a
round body) or spheres / conics (a ``SURFACE_SEQUENCE``), each from the Project through lowering and the trace to the results.

Independent references, computed here:

* the ray equation ``d/ds (n t) = grad n`` integrated in arc length by SciPy's ``solve_ivp`` (DOP853, rtol 1e-12) with the
  optical path ``int n ds``, the length and the polarization carried by ``de/ds = -(e . dt/ds) t`` -- for the radial
  profile ``n^2 = n0^2 (1 - g^2 r^2)`` and the axial polynomial, stopped by an event on the face (a plane, a sphere);
* the faces' refraction by vector Snell's law at the local index, the straight legs in air by hand, and the p-wave Fresnel
  transmittances of the two faces;
* the paraxial reduced ABCD matrices by hand: an axial gap ``((1, int dz / n), (0, 1))`` with ``int dz / n`` by SciPy's
  ``quad``, a GRIN gap ``((cos gL, sin gL / (n0 g)), (-n0 g sin gL, cos gL))``, a face ``((1, 0), (-(n2 - n1) / R, 1))``,
  applied to the Gaussian beam's ``q``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.contracts import AxialGradientIndexMaterial
from os_optics.local import LocalRunError, lower_project, run_optical_project
from os_optics.local.compile import ProjectResolver, compile_declared, paraxial_matrix
from os_optics.optics.rays import gradient
from os_optics.scene.model import GradientFaceG

from catalog3_products import d
from local_helpers import ALONG_X, plane, project, quat_multiply, source, turn_about_z

N0, G = 1.6, 0.3
TERMS = (-0.01, 2e-4, -3e-6)
WAVELENGTH = 633.0
APPROXIMATIONS = ["NOMINAL_PRESCRIPTION", "IDEAL_SURFACE_FIGURE", "GHOST_REFLECTIONS_NOT_TRACED", "EDGE_AS_OPAQUE_RIM"]


def axial_n(r) -> float:
    z = r[2]
    return N0 + TERMS[0] * z + TERMS[1] * z * z + TERMS[2] * z ** 3


def axial_grad(r) -> np.ndarray:
    z = r[2]
    return np.array([0.0, 0.0, TERMS[0] + 2.0 * TERMS[1] * z + 3.0 * TERMS[2] * z * z])


def radial_n(r) -> float:
    return N0 * math.sqrt(1.0 - G * G * (r[0] ** 2 + r[1] ** 2))


def radial_grad(r) -> np.ndarray:
    n = radial_n(r)
    return np.array([-N0 * N0 * G * G * r[0] / n, -N0 * N0 * G * G * r[1] / n, 0.0])


def integrate(index, grad, start, direction, event, e=None) -> dict:
    """The ray from ``start`` along ``direction`` (unit) until ``event(r)`` changes sign: ``dr/ds = t``, ``d(n t)/ds = grad n``,
    ``de/ds = -(e . dt/ds) t``."""

    def rhs(_s, y):
        r, p, ev = y[:3], y[3:6], y[6:9]
        n = index(r)
        g = grad(r)
        t = p / n
        dt = (g - np.dot(g, t) * t) / n
        return np.concatenate([t, g, -np.dot(ev, dt) * t, [n]])

    def stop(_s, y):
        return event(y[:3])

    stop.terminal = True
    y0 = np.concatenate([start, index(start) * direction, e if e is not None else np.zeros(3), [0.0]])
    sol = solve_ivp(rhs, (0.0, 500.0), y0, method="DOP853", rtol=1e-12, atol=1e-14, events=stop)
    y = sol.y_events[0][0]
    return {"point": y[:3], "direction": y[3:6] / index(y[:3]), "length": float(sol.t_events[0][0]), "opl": float(y[9]), "e": y[6:9]}


def snell(t: np.ndarray, m: np.ndarray, n1: float, n2: float) -> np.ndarray:
    m = m if np.dot(t, m) > 0.0 else -m
    mu, cos_i = n1 / n2, float(np.dot(t, m))
    cos_t = math.sqrt(1.0 - mu * mu * (1.0 - cos_i * cos_i))
    return mu * t + (cos_t - mu * cos_i) * m


def transmit(e: np.ndarray, t_in: np.ndarray, t_out: np.ndarray, m: np.ndarray, n1: float, n2: float) -> tuple[float, np.ndarray]:
    """A face's power transmittance for the unit field ``e`` (across ``t_in``) and the unit field behind it: its s and p parts
    (across and in the plane of incidence) scaled by the Fresnel amplitude coefficients ``t_s``, ``t_p``, the power by
    ``n2 cos t / (n1 cos i)``."""

    ci, ct = abs(float(np.dot(t_in, m))), abs(float(np.dot(t_out, m)))
    s_hat = np.cross(t_in, m)
    if np.linalg.norm(s_hat) < 1e-15:  # (normal incidence: s and p alike)
        s_hat = np.cross(t_in, [1.0, 0.0, 0.0] if abs(t_in[0]) < 0.9 else [0.0, 1.0, 0.0])
    s_hat = s_hat / np.linalg.norm(s_hat)
    ts, tp = 2.0 * n1 * ci / (n1 * ci + n2 * ct), 2.0 * n1 * ci / (n2 * ci + n1 * ct)
    es, ep = float(np.dot(e, s_hat)), float(np.dot(e, np.cross(s_hat, t_in)))
    out = ts * es * s_hat + tp * ep * np.cross(s_hat, t_out)
    power = n2 * ct / (n1 * ci) * float(np.dot(out, out))
    return power, out / np.linalg.norm(out)


# -------------------------------------------------------------------------------------------------- the axial medium's legs


@pytest.mark.parametrize("start, tilt", [((0.3, 0.0), (0.0, 0.0)), ((0.2, -0.1), (0.05, 0.08)), ((0.0, 0.4), (-0.3, 0.0)),
                                         ((0.5, 0.3), (-0.2, 0.25))])
def test_axial_legs_match_the_integrated_ray_equation(start, tilt) -> None:
    """Meridional and skew rays through 12 mm of a cubic axial profile: point, direction, optical path and length against the
    integrated ray equation; the polarization carried in closed form against parallel transport; the polyline on the ray."""

    z_end = 12.0
    profile = gradient.AxialProfile(N0, TERMS)
    direction = np.array([tilt[0], tilt[1], 1.0])
    direction = direction / np.linalg.norm(direction)
    e = np.cross(direction, [0.3, 1.0, 0.2])
    e = e / np.linalg.norm(e)
    leg, along = gradient.legs(profile, np.array([start[0]]), np.array([start[1]]), np.array([0.0]), direction[None, :], 0.0, z_end)
    assert isinstance(leg, gradient.AxialLegs) and bool(along[0]) and not bool(leg.turns()[0])
    x, y, z, dx, dy = leg.at(leg.z_len)
    reference = integrate(axial_n, axial_grad, np.array([start[0], start[1], 0.0]), direction, lambda r: r[2] - z_end, e)
    np.testing.assert_allclose([x[0], y[0], z[0]], reference["point"], atol=1e-9)
    out = np.array([dx[0], dy[0], 1.0])
    out = out / np.linalg.norm(out)
    np.testing.assert_allclose(out, reference["direction"], atol=1e-10)
    # (the integration's own error is ~2e-11 of the path: rel 1e-10)
    assert float(leg.optical_path()[0]) == pytest.approx(reference["opl"], rel=1e-10)
    assert float(leg.lengths()[0]) == pytest.approx(reference["length"], rel=1e-10)
    if tilt == (0.0, 0.0):  # (along the axis: int n dz = n0 L + a1 L^2 / 2 + a2 L^3 / 3 + a3 L^4 / 4 exactly)
        exact = N0 * z_end + TERMS[0] * z_end ** 2 / 2 + TERMS[1] * z_end ** 3 / 3 + TERMS[2] * z_end ** 4 / 4
        assert float(leg.optical_path()[0]) == pytest.approx(exact, rel=1e-14)
    carried = gradient.planar_transport(e[None, :], direction[None, :], out[None, :], np.array([0.0, 0.0, 1.0]))[0]
    assert abs(float(np.dot(carried, reference["e"] / np.linalg.norm(reference["e"])))) == pytest.approx(1.0, abs=1e-9)
    points, _tangents = leg.points(leg.point_count())
    for k in (8, 16, 24):
        z_k = float(points[k, 0, 2])
        ray = integrate(axial_n, axial_grad, np.array([start[0], start[1], 0.0]), direction, lambda r, z_k=z_k: r[2] - z_k)
        np.testing.assert_allclose(points[k, 0], ray["point"], atol=1e-9)


def test_an_axial_ray_near_its_turning_point_is_not_followed() -> None:
    """``n`` falls from 1.6 to 1.1 over 10 mm: a ray at 0.95 rad would turn back (its ``|k| = 1.6 sin 0.95`` exceeds 1.1 times
    the cosine margin), so it is marked; a ray at 0.3 rad is not."""

    profile = gradient.AxialProfile(N0, (-0.05,))
    rays = np.array([[math.sin(0.95), 0.0, math.cos(0.95)], [math.sin(0.3), 0.0, math.cos(0.3)]])
    leg, _ = gradient.legs(profile, np.zeros(2), np.zeros(2), np.zeros(2), rays, 0.0, 10.0)
    assert leg.turns().tolist() == [True, False]


# ----------------------------------------------------------------------------------- faces that are not planes across the axis


def _slanted(deg: float, z: float) -> GradientFaceG:
    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    radius = 1.0
    return GradientFaceG("plane", (0.0, 0.0, z), (0.0, s, c), z - radius * abs(s) / c, z + radius * abs(s) / c, abs(s) / c)


def _conic(z: float, radius_of_curvature: float, conic: float) -> GradientFaceG:
    c, rod = 1.0 / radius_of_curvature, 1.0
    root = math.sqrt(1.0 - (1.0 + conic) * c * c * rod * rod)
    sag = c * rod * rod / (1.0 + root)
    return GradientFaceG("conic", (0.0, 0.0, z), (0.0, 0.0, 1.0), z + min(0.0, sag), z + max(0.0, sag), abs(c) * rod / root, c, conic)


def _face_event(face: GradientFaceG):
    def event(r):
        return r[2] - float(gradient.face_z(face, np.array([r[0]]), np.array([r[1]]))[0])
    return event


@pytest.mark.parametrize("face", [_slanted(8.0, 6.0), _slanted(-25.0, 6.0), _conic(6.0, 3.0, 0.0), _conic(6.0, -2.5, -0.6)],
                         ids=["slant8", "slant-25", "sphere", "conic"])
@pytest.mark.parametrize("axial", [False, True], ids=["radial", "axial"])
def test_a_ray_meets_a_slanted_or_curved_face_where_the_ray_equation_does(face: GradientFaceG, axial: bool) -> None:
    profile = gradient.AxialProfile(N0, TERMS) if axial else gradient.Profile(N0, G)
    index, grad = (axial_n, axial_grad) if axial else (radial_n, radial_grad)
    starts = np.array([[0.3, 0.0], [0.2, -0.4], [-0.5, 0.1]])
    tilts = np.array([[0.0, 0.0], [0.05, 0.08], [-0.05, 0.06]])
    directions = np.column_stack([tilts, np.ones(3)])
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    leg, _ = gradient.legs(profile, starts[:, 0], starts[:, 1], np.zeros(3), directions, 0.0, 6.0)
    leg, found = gradient.to_face(leg, face)
    assert found.all()
    x, y, z, _dx, _dy = leg.at(leg.z_len)
    for k in range(3):
        reference = integrate(index, grad, np.array([starts[k, 0], starts[k, 1], 0.0]), directions[k], _face_event(face))
        np.testing.assert_allclose([x[k], y[k], z[k]], reference["point"], atol=1e-9)
        assert float(leg.optical_path(profile)[k]) == pytest.approx(reference["opl"], rel=1e-10)


def test_a_ray_too_oblique_to_cross_a_steep_face_once_is_not_found() -> None:
    """A face slanted 80 deg (largest slope tan 80 = 5.7) and a ray crossing the axis at 0.3 rad: ``slope s >= 1``, so its
    single crossing is not proven and the ray is marked."""

    face = _slanted(80.0, 6.0)
    profile = gradient.Profile(N0, G)
    direction = np.array([[math.sin(0.3), 0.0, math.cos(0.3)]])
    leg, _ = gradient.legs(profile, np.zeros(1), np.zeros(1), np.zeros(1), direction, 0.0, 6.0)
    _leg, found = gradient.to_face(leg, face)
    assert found.tolist() == [False]


# ------------------------------------------------------------------------------------------------------- the Project path


def axial_material() -> dict:
    return {"material_id": "AGRIN", "display_name": "Test axial gradient glass", "basis": "test values (not a supplier's product)",
            "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": str(N0)},
            "absorption_coefficient_per_mm": "0",  # Ideal test rod: isolate the ray equation and Fresnel faces.
            "valid_band_nm": {"min_nm": "400", "max_nm": "1700", "status": "SPECIFIED"},
            "axial_gradient": {"kind": "AXIAL_POLYNOMIAL", "index_terms": [d(a) for a in TERMS]}}


def radial_material() -> dict:
    return {"material_id": "GRIN", "display_name": "Test gradient glass", "basis": "test values (not a supplier's product)",
            "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": str(N0)},
            "absorption_coefficient_per_mm": "0",  # Ideal test rod: isolate the ray equation and Fresnel faces.
            "valid_band_nm": {"min_nm": "400", "max_nm": "1700", "status": "SPECIFIED"},
            "gradient": {"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": str(G)}}


def sequence_rod(medium: str, length: str, r1: str | None = None, r2: str | None = None, *, edge: str = "1", clear: str = "0.9",
                 conic2: str | None = None) -> dict:
    faces = [{"surface_id": "S1", "vertex_z_mm": "0"}, {"surface_id": "S2", "vertex_z_mm": length}]
    for face, radius in zip(faces, (r1, r2)):
        face.update(clear_radius_mm=clear, traced="TRANSMITTED", status="SPECIFIED")
        if radius is not None:
            face["radius_mm"] = radius
    if conic2 is not None:
        faces[1]["conic"] = conic2
    return {"model_id": "ROD", "model": {
        "model_kind": "SURFACE_SEQUENCE", "fidelity": "SURFACE_PRESCRIPTION", "surfaces": faces, "media": [medium],
        "edge_radius_mm": edge, "valid_band_nm": {"min_nm": "500", "max_nm": "1100", "status": "SPECIFIED"},
        "approximations": APPROXIMATIONS, "status": "SPECIFIED"}}


def slanted_rod(medium: str, deg: float, length: float = 5.0) -> dict:
    """A round body of ``medium`` (radius 1 mm) from local z = 0 to ``length``: the entrance face through the origin tilted by
    ``deg`` about local X (outward normal ``(0, sin, -cos)``), the exit face across the axis."""

    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    faces = [{"point_mm": ["0", "0", d(length)], "outward_normal": ["0", "0", "1"]},
             {"point_mm": ["0", "0", "0"], "outward_normal": ["0", d(s), d(-c)]}]
    common = {"u_axis": ["1", "0", "0"], "medium_front": "AIR", "medium_back": medium, "traced": "TRANSMITTED", "status": "SPECIFIED",
              "clear_radius_mm": "0.9"}
    surfaces = [{"surface_id": "S1", "origin_mm": ["0", "0", "0"], "normal": ["0", d(s), d(-c)], **common},
                {"surface_id": "S2", "origin_mm": ["0", "0", d(length)], "normal": ["0", "0", "1"], **common}]
    return {"model_id": "ROD", "model": {
        "model_kind": "SURFACE_SET", "fidelity": "SURFACE_PRESCRIPTION", "surfaces": surfaces,
        "body": {"medium": medium, "faces": faces, "side_radius_mm": "1"},
        "valid_band_nm": {"min_nm": "500", "max_nm": "1100", "status": "SPECIFIED"},
        "approximations": ["NOMINAL_PRESCRIPTION", "EDGE_AS_OPAQUE_RIM"], "status": "SPECIFIED"}}


def rod_project(model: dict, material: dict, *, start=(0.0, 0.0), tilt: float = 0.0, waist: float = 0.05, out_at: float = 8.0) -> dict:
    """The rod from x = 0 along design +X (its local Z; local X = design Y, local Y = design Z), a 633 nm Gaussian source 20 mm
    before it at local ``start`` tilted by ``tilt`` rad about design Z (toward local X), its waist on the entrance plane, and
    the plane OUT at design x = ``out_at``."""

    quaternion = quat_multiply(turn_about_z(math.cos(tilt / 2), math.sin(tilt / 2)), ALONG_X)
    position = (-20.0 * math.cos(tilt), start[0] - 20.0 * math.sin(tilt), start[1])
    src = source(position=position, wavelength_nm=WAVELENGTH, waist_mm=waist, waist_position_mm=20.0, quaternion=quaternion)
    comp = {"instance_id": "G1", "optical_role": "lens", "custom_part": {"custom_part_key": "grin", "description": "gradient rod"},
            "optical_model_id": "ROD", "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": list(ALONG_X)}}
    document = project([src, comp], [plane("OUT", (out_at, 0.0, 0.0), radius_mm=5.0)])
    document["optical_models"] = [model]
    document["model_materials"] = [material]
    return document


def to_design(v) -> np.ndarray:
    return np.array([v[2], v[0], v[1]])


def reference_path(start, tilt, entry, exit_, index, grad) -> dict:
    """The chief ray by hand in the rod's local frame: straight from the source to the entrance face (``entry``: ``(point on
    the face, its normal there)`` of a ray), Snell into the medium, the integrated ray equation to the exit face (``exit_``:
    ``(event, normal(r))``), Snell out."""

    origin = np.array([start[0] - 20.0 * math.sin(tilt), start[1], -20.0 * math.cos(tilt)])
    t0 = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    e0 = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])  # (the source's field: along its local x, design Y turned by the tilt)
    p1, m1 = entry(origin, t0)
    t1 = snell(t0, m1, 1.0, index(p1))
    power1, e1 = transmit(e0, t0, t1, m1, 1.0, index(p1))
    inside = integrate(index, grad, p1, t1, exit_[0], e1)
    p2 = inside["point"]
    m2 = exit_[1](p2)
    t2 = snell(inside["direction"], m2, index(p2), 1.0)
    e2 = inside["e"] / np.linalg.norm(inside["e"])
    power2, _e3 = transmit(e2, inside["direction"], t2, m2, index(p2), 1.0)
    return {"entry": p1, "exit": p2, "out": t2, "opl": inside["opl"], "length": inside["length"], "power": power1 * power2 * 1e-3}


def plane_entry(point, normal):
    point, normal = np.asarray(point, float), np.asarray(normal, float)

    def entry(origin, t0):
        return origin + (np.dot(point - origin, normal) / np.dot(t0, normal)) * t0, normal
    return entry


def sphere_entry(vertex_z: float, radius: float):
    centre = np.array([0.0, 0.0, vertex_z + radius])

    def entry(origin, t0):
        rel = origin - centre
        b, c = np.dot(rel, t0), np.dot(rel, rel) - radius * radius
        root = math.sqrt(b * b - c)
        t = -b - root if radius > 0 else -b + root
        p = origin + t * t0
        return p, (p - centre) / abs(radius)
    return entry


def sphere_exit(vertex_z: float, radius: float):
    centre = np.array([0.0, 0.0, vertex_z + radius])
    return (lambda r: (np.dot(r - centre, r - centre) - radius * radius) * (-1.0 if radius > 0 else 1.0),
            lambda r: (r - centre) / abs(radius))


def plane_exit(z: float):
    return (lambda r: r[2] - z, lambda r: np.array([0.0, 0.0, 1.0]))


def inside_segments(result, medium: str):
    (path,) = result.paths
    return [s for s in path.segments if s.medium_id == medium]


def check_chief(run, medium: str, reference: dict) -> None:
    result = run.result
    assert result.power_budget.conserved
    inside = inside_segments(result, medium)
    np.testing.assert_allclose(inside[0].start_mm, to_design(reference["entry"]), atol=1e-9)
    np.testing.assert_allclose(inside[-1].end_mm, to_design(reference["exit"]), atol=1e-9)
    assert sum(s.optical_path_length_mm for s in inside) == pytest.approx(reference["opl"], rel=1e-10)
    assert sum(s.length_mm for s in inside) == pytest.approx(reference["length"], rel=1e-10)
    (arrival,) = next(o for o in result.observations if o.observation_id == "OUT").arrivals
    np.testing.assert_allclose(arrival.beam.direction, to_design(reference["out"]), atol=1e-9)
    assert arrival.beam.power_W == pytest.approx(reference["power"], rel=1e-9)


def test_an_axial_rod_lowers_to_an_axial_gradient_material_with_its_entrance_on_the_axis() -> None:
    request = lower_project(rod_project(sequence_rod("AGRIN", "10", edge="2", clear="1.9"), axial_material()), catalog=None).request
    (rod,) = [m for m in request.project.materials if isinstance(m, AxialGradientIndexMaterial)]
    assert rod.material_id == "project:AGRIN@G1" and rod.axis_component == "G1"
    assert (rod.axis_point_mm, rod.axis_direction, rod.radius_mm, rod.coefficients_per_mm) == ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, TERMS)


@pytest.mark.parametrize("start, tilt", [((0.3, 0.0), 0.0), ((0.3, 0.0), 0.05), ((-0.4, 0.2), -0.08)])
def test_a_beam_through_an_axial_rod_follows_the_ray_equation(start, tilt: float) -> None:
    run = run_optical_project(rod_project(sequence_rod("AGRIN", "10", edge="2", clear="1.9"), axial_material(), start=start, tilt=tilt,
                                          out_at=13.0), catalog=None)
    reference = reference_path(start, tilt, plane_entry((0, 0, 0), (0, 0, -1)), plane_exit(10.0), axial_n, axial_grad)
    check_chief(run, "project:AGRIN@G1", reference)
    assert run.model_scope_issues == () and run.status == "PASS"


def test_the_gaussian_beam_through_an_axial_rod_takes_its_paraxial_matrix() -> None:
    """On the axis, the waist on the entrance face: ``q = i z_R + int_0^L dz / n(z) + delta`` behind it (a plane face keeps
    ``q / n``; the reduced slope ``n u`` is conserved in the medium)."""

    run = run_optical_project(rod_project(sequence_rod("AGRIN", "10", edge="2", clear="1.9"), axial_material(), waist=0.1, out_at=12.0),
                              catalog=None)
    (arrival,) = next(o for o in run.result.observations if o.observation_id == "OUT").arrivals
    z_r = math.pi * 0.1 * 0.1 / (WAVELENGTH * 1e-6)
    reduced = quad(lambda z: 1.0 / axial_n((0, 0, z)), 0.0, 10.0, epsabs=1e-13, epsrel=1e-13)[0]
    expected = complex(reduced + 2.0, z_r)
    assert arrival.beam.gaussian.valid
    for q in (arrival.beam.gaussian.q_x_mm, arrival.beam.gaussian.q_y_mm):
        assert complex(*q) == pytest.approx(expected, rel=1e-9)


def test_the_paraxial_matrix_of_an_axial_lens_with_curved_faces() -> None:
    """Faces R1 = 8 mm and R2 = -12 mm: ``((1, 0), (-(1 - n(L)) / R2, 1)) ((1, int dz / n), (0, 1)) ((1, 0), (-(n0 - 1) / R1, 1))``."""

    document = rod_project(sequence_rod("AGRIN", "10", "8", "-12", edge="2", clear="1.9"), axial_material())
    model = ProjectDocumentV2.model_validate(document)
    element = compile_declared("G1", model.optical_model("ROD"), ProjectResolver(model))
    n_l = axial_n((0, 0, 10.0))
    gap = quad(lambda z: 1.0 / axial_n((0, 0, z)), 0.0, 10.0, epsabs=1e-13, epsrel=1e-13)[0]
    first = np.array([[1.0, 0.0], [-(N0 - 1.0) / 8.0, 1.0]])
    middle = np.array([[1.0, gap], [0.0, 1.0]])
    last = np.array([[1.0, 0.0], [-(1.0 - n_l) / -12.0, 1.0]])
    expected = last.dot(middle).dot(first)
    np.testing.assert_allclose(np.array(paraxial_matrix(element, WAVELENGTH)), expected, rtol=1e-12, atol=1e-14)


def test_a_beam_through_an_axial_rod_with_curved_faces_follows_the_ray_equation() -> None:
    run = run_optical_project(rod_project(sequence_rod("AGRIN", "10", "8", "-12", edge="2", clear="1.9"), axial_material(), start=(0.4, -0.2),
                                          out_at=14.0), catalog=None)
    reference = reference_path((0.4, -0.2), 0.0, sphere_entry(0.0, 8.0), sphere_exit(10.0, -12.0), axial_n, axial_grad)
    check_chief(run, "project:AGRIN@G1", reference)
    assert run.status == "PASS"


@pytest.mark.parametrize("r1, r2, start", [("3", "-4", (0.2, 0.0)), ("-5", "2.5", (0.15, -0.1))])
def test_a_beam_through_a_radial_rod_with_spherical_faces_follows_the_ray_equation(r1: str, r2: str, start) -> None:
    run = run_optical_project(rod_project(sequence_rod("GRIN", "5", r1, r2), radial_material(), start=start), catalog=None)
    reference = reference_path(start, 0.0, sphere_entry(0.0, float(r1)), sphere_exit(5.0, float(r2)), radial_n, radial_grad)
    check_chief(run, "project:GRIN@G1", reference)
    assert run.model_scope_issues == () and run.status == "PASS"


def test_the_gaussian_beam_through_a_radial_rod_with_spherical_faces_takes_the_matrix_product() -> None:
    """On the axis: the faces' refraction and the GRIN gap applied to the reduced ``q`` (a waist on the entrance vertex plane:
    ``q = i z_R`` in air there), then 3 mm of air."""

    length = 5.0
    run = run_optical_project(rod_project(sequence_rod("GRIN", "5", "3", "-4"), radial_material(), waist=0.05), catalog=None)
    (arrival,) = next(o for o in run.result.observations if o.observation_id == "OUT").arrivals
    face1 = np.array([[1.0, 0.0], [-(N0 - 1.0) / 3.0, 1.0]])
    rod = np.array([[math.cos(G * length), math.sin(G * length) / (N0 * G)], [-N0 * G * math.sin(G * length), math.cos(G * length)]])
    face2 = np.array([[1.0, 0.0], [-(1.0 - N0) / -4.0, 1.0]])
    (a, b), (c, dd) = face2.dot(rod).dot(face1)
    z_r = math.pi * 0.05 * 0.05 / (WAVELENGTH * 1e-6)
    q = complex(0.0, z_r)
    expected = (a * q + b) / (c * q + dd) + 3.0
    assert arrival.beam.gaussian.valid
    assert complex(*arrival.beam.gaussian.q_x_mm) == pytest.approx(expected, rel=1e-7)
    document = rod_project(sequence_rod("GRIN", "5", "3", "-4"), radial_material())
    model = ProjectDocumentV2.model_validate(document)
    element = compile_declared("G1", model.optical_model("ROD"), ProjectResolver(model))
    np.testing.assert_allclose(np.array(paraxial_matrix(element, WAVELENGTH)), face2.dot(rod).dot(face1), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("medium, material", [("GRIN", radial_material()), ("AGRIN", axial_material())], ids=["radial", "axial"])
@pytest.mark.parametrize("deg, start", [(8.0, (0.0, 0.0)), (8.0, (0.0, 0.3)), (-12.0, (0.0, -0.25))])
def test_a_beam_through_a_rod_with_a_slanted_entrance_face_follows_the_ray_equation(medium: str, material: dict, deg: float, start) -> None:
    """The entrance face tilted about local X, the beam in the plane of the tilt (local Y Z): the chief ray against the
    integrated ray equation, the exit direction against Snell's law at the local index; the transfer stays decoupled on that
    plane, so the Gaussian state stays valid."""

    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    run = run_optical_project(rod_project(slanted_rod(medium, deg), material, start=start), catalog=None)
    index, grad = (radial_n, radial_grad) if medium == "GRIN" else (axial_n, axial_grad)
    reference = reference_path(start, 0.0, plane_entry((0, 0, 0), (0, s, -c)), plane_exit(5.0), index, grad)
    check_chief(run, f"project:{medium}@G1", reference)
    (arrival,) = next(o for o in run.result.observations if o.observation_id == "OUT").arrivals
    assert arrival.beam.gaussian.valid
    assert run.model_scope_issues == () and run.status == "PASS"


def test_a_skew_chief_ray_behind_a_slanted_face_is_twisted_astigmatism() -> None:
    """The beam off the plane of the tilt: the chief ray is skew in the rod, the first-order transfer couples the beam's axes
    and the Gaussian state is withheld (``GAUSSIAN_TWISTED_ASTIGMATISM``); the ray data stay exact."""

    run = run_optical_project(rod_project(slanted_rod("GRIN", 8.0), radial_material(), start=(0.2, 0.0)), catalog=None)
    s, c = math.sin(math.radians(8.0)), math.cos(math.radians(8.0))
    reference = reference_path((0.2, 0.0), 0.0, plane_entry((0, 0, 0), (0, s, -c)), plane_exit(5.0), radial_n, radial_grad)
    check_chief(run, "project:GRIN@G1", reference)
    assert "GAUSSIAN_TWISTED_ASTIGMATISM" in {i.code.value for i in run.result.issues}


def test_a_chief_ray_too_oblique_to_a_steep_face_ends_out_of_model_scope() -> None:
    """A curved exit face of radius 1.01 mm on a rod of radius 1 mm (largest slope 7.05) and a beam 0.6 mm off the axis (its
    slope across the axis up to ``g r0 = 0.18``): the chief ray's slope bound times the face's exceeds 1, so its single crossing
    is not proven: the path ends ``GRADIENT_INDEX_UNSUPPORTED``."""

    run = run_optical_project(rod_project(sequence_rod("GRIN", "5", None, "-1.01"), radial_material(), start=(0.6, 0.0), waist=0.02),
                              catalog=None)
    issues = [i for i in run.result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    assert issues and "too obliquely" in issues[0].message
    assert run.result.power_budget.conserved and run.status == "OUT_OF_MODEL_SCOPE"


@pytest.mark.parametrize("change, match", [
    (lambda doc: doc["optical_models"][0]["model"]["surfaces"][1].update(aspheric_coefficients=["0.001"]), "not a plane, sphere or conic"),
    (lambda doc: doc["optical_models"][0]["model"]["surfaces"][1].update(radius_mm="-0.85"), "profile ends inside the rod's radius"),
    (lambda doc: doc["model_materials"][0]["axial_gradient"].update(index_terms=["-0.07"]), "falls below index 1"),
    (lambda doc: [doc["optical_models"][0]["model"]["surfaces"][0].update(radius_mm="1"),
                  doc["optical_models"][0]["model"]["surfaces"][1].update(radius_mm="-1", vertex_z_mm="1")], "reach past each other"),
])
def test_a_rod_outside_the_rules_is_refused(change, match: str) -> None:
    document = rod_project(sequence_rod("AGRIN", "10", "8", "-12", edge="0.9", clear="0.8"), axial_material())
    change(document)
    with pytest.raises(LocalRunError, match=match):
        lower_project(document, catalog=None)


def test_a_rod_in_a_surface_set_needs_its_round_body() -> None:
    document = rod_project(slanted_rod("GRIN", 8.0), radial_material())
    del document["optical_models"][0]["model"]["body"]["side_radius_mm"]
    with pytest.raises(LocalRunError):
        lower_project(document, catalog=None)


@pytest.mark.parametrize("which", ["axial_curved", "slanted"])
def test_a_bench_turned_about_an_oblique_axis_traces_the_same(which: str) -> None:
    """The two Pyodide parity Projects, turned and not: the same status, power and arrival (turned back) -- a face across the
    axis stays across it after the turn's rounding (``scene/lower.py`` takes a normal within 1e-12 of the axis as the axis)."""
    from test_packaging_and_pyodide import OBLIQUE_TURN, _turned

    document = (rod_project(sequence_rod("AGRIN", "10", "8", "-12", edge="2", clear="1.9"), axial_material(), start=(0.4, -0.2), out_at=14.0)
                if which == "axial_curved" else rod_project(slanted_rod("GRIN", 8.0), radial_material(), start=(0.0, 0.3)))
    plain = run_optical_project(document, catalog=None)
    turned = run_optical_project(_turned(document), catalog=None)
    assert plain.status == turned.status == "PASS"
    (a,) = next(o for o in plain.result.observations if o.observation_id == "OUT").arrivals
    (b,) = next(o for o in turned.result.observations if o.observation_id == "OUT").arrivals
    assert b.beam.power_W == pytest.approx(a.beam.power_W, rel=1e-12)
    w, x, y, z = OBLIQUE_TURN
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    np.testing.assert_allclose(np.array(b.beam.direction), rot.dot(np.array(a.beam.direction)), atol=1e-12)
