"""Radial gradient-index media (GRIN rods): the closed-form leg (``optics/rays/gradient.py``), the Project path (a declared
``gradient`` material in a round ``SURFACE_SEQUENCE``), the Gaussian beam and the paraxial matrix.

Independent references, computed here:

* the ray equation ``d/ds (n dr/ds) = grad n`` for ``n^2 = n0^2 (1 - g^2 r^2)`` integrated in arc length by SciPy's
  ``solve_ivp`` (DOP853, rtol 1e-12) from the ray's start, with the optical path ``int n ds``, the length ``int ds`` and the
  polarization carried by ``de/ds = -(e . dt/ds) t`` (parallel transport) -- not the closed form under test;
* the paraxial period ``2 pi / g`` and the GRIN matrix ``((cos gL, sin gL / (n0 g)), (-n0 g sin gL, cos gL))`` (E. W. Marchand,
  *Gradient Index Optics*, 1978, ch. 3) applied to the Gaussian beam's ``q`` by hand;
* the focal length of a rod in air, ``f = 1 / (n0 g sin gL)``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index
from scipy.integrate import solve_ivp

from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.contracts import GradientIndexMaterial
from os_optics.local import LocalRunError, lower_project, run_optical_project
from os_optics.local.compile import ProjectResolver, compile_declared, paraxial_matrix
from os_optics.optics.rays import gradient

from local_helpers import ALONG_X, plane, project, source

N0, G, RADIUS = 1.6, 0.3, 1.0
WAVELENGTH = 633.0
QUARTER = math.pi / (2.0 * G)


def ode_ray(start: np.ndarray, direction: np.ndarray, z_end: float, e: np.ndarray | None = None) -> dict:
    """The ray from ``start`` along ``direction`` (unit) to the plane ``z = z_end``, by integrating the ray equation in arc length:
    ``dr/ds = t``, ``d(n t)/ds = grad n``, ``de/ds = -(e . dt/ds) t``; ``n^2 = N0^2 (1 - G^2 (x^2 + y^2))``."""

    def index(r):
        return N0 * math.sqrt(1.0 - G * G * (r[0] ** 2 + r[1] ** 2))

    def rhs(_s, y):
        r, p, ev = y[:3], y[3:6], y[6:9]
        n = index(r)
        grad = np.array([-N0 * N0 * G * G * r[0] / n, -N0 * N0 * G * G * r[1] / n, 0.0])
        t = p / n
        dt = (grad - np.dot(grad, t) * t) / n  # d t / ds
        return np.concatenate([t, grad, -np.dot(ev, dt) * t, [n]])

    def plane_reached(_s, y):
        return y[2] - z_end

    plane_reached.terminal, plane_reached.direction = True, 1
    n_start = index(start)
    y0 = np.concatenate([start, n_start * direction, e if e is not None else np.zeros(3), [0.0]])
    sol = solve_ivp(rhs, (0.0, 10.0 * z_end + 10.0), y0, method="DOP853", rtol=1e-12, atol=1e-14, events=plane_reached)
    y = sol.y_events[0][0]
    n_end = index(y[:3])
    return {"point": y[:3], "direction": y[3:6] / n_end, "length": float(sol.t_events[0][0]), "opl": float(y[9]), "e": y[6:9]}


def profile() -> gradient.Profile:
    return gradient.Profile(N0, G)


@pytest.mark.parametrize("start, tilt", [((0.3, 0.0), (0.0, 0.0)), ((0.2, -0.1), (0.05, 0.08)), ((0.0, 0.4), (-0.12, 0.0)),
                                         ((0.5, 0.3), (-0.1, 0.2))])
def test_the_closed_form_leg_matches_the_integrated_ray_equation(start, tilt) -> None:
    """Meridional and skew rays over 1.7 periods: point, direction, optical path and length against the integrated ray
    equation, and the polarization carried along against parallel transport."""

    z_end = 1.7 * 2.0 * math.pi / G
    d = np.array([tilt[0], tilt[1], 1.0])
    d = d / np.linalg.norm(d)
    e = np.cross(d, [0.3, 1.0, 0.2])
    e = e / np.linalg.norm(e)
    leg, along = gradient.legs(profile(), np.array([start[0]]), np.array([start[1]]), np.array([0.0]), d[None, :], 0.0, z_end)
    assert bool(along[0])
    x, y, z, dx, dy = leg.at(leg.z_len)
    direction = np.array([dx[0], dy[0], 1.0]) / math.hypot(dx[0], dy[0], 1.0)
    reference = ode_ray(np.array([start[0], start[1], 0.0]), d, z_end, e)
    np.testing.assert_allclose([x[0], y[0], z[0]], reference["point"], atol=1e-9)
    np.testing.assert_allclose(direction, reference["direction"], atol=1e-9)
    assert float(leg.optical_path(profile())[0]) == pytest.approx(reference["opl"], rel=1e-11)
    assert float(leg.lengths(profile())[0]) == pytest.approx(reference["length"], rel=1e-11)
    points, tangents = leg.points(leg.point_count())
    carried = gradient.transported(points, tangents, e[None, :])[0]
    expected = reference["e"] / np.linalg.norm(reference["e"])
    assert abs(float(np.dot(carried, expected))) == pytest.approx(1.0, abs=1e-9)
    sampled = float(np.max(np.sum(points[:, 0, :2] ** 2, axis=-1)))  # (the polyline's points: at most the largest r^2)
    assert sampled <= float(leg.largest_r2()[0]) <= sampled * (1.0 + 1e-4)


def test_a_paraxial_ray_has_the_period_two_pi_over_g() -> None:
    """A ray 1 um off the axis, parallel to it: it crosses the axis after a quarter period ``pi / (2 g)`` and comes back after
    one period ``2 pi / g`` (to the paraxial order of its height)."""

    for length, expected in ((QUARTER, 0.0), (4.0 * QUARTER, 1e-3), (2.0 * QUARTER, -1e-3)):
        leg, _ = gradient.legs(profile(), np.array([1e-3]), np.array([0.0]), np.array([0.0]), np.array([[0.0, 0.0, 1.0]]), 0.0, length)
        x, *_ = leg.at(leg.z_len)
        assert float(x[0]) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------------------------------- the Project path


def material(constant: str = str(G)) -> dict:
    return {"material_id": "GRIN", "display_name": "Test gradient glass", "basis": "test values (not a supplier's product)",
            "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": str(N0)},
            "absorption_coefficient_per_mm": "0",  # Ideal test rod: isolate GRIN refraction from bulk absorption.
            "valid_band_nm": {"min_nm": "400", "max_nm": "1700", "status": "SPECIFIED"},
            "gradient": {"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": constant}}


def rod(length: float) -> dict:
    face = {"clear_radius_mm": "0.9", "traced": "TRANSMITTED", "status": "SPECIFIED"}
    return {"model_id": "ROD", "model": {
        "model_kind": "SURFACE_SEQUENCE", "fidelity": "SURFACE_PRESCRIPTION",
        "surfaces": [{"surface_id": "S1", "vertex_z_mm": "0", **face}, {"surface_id": "S2", "vertex_z_mm": repr(round(length, 9)), **face}],
        "media": ["GRIN"], "edge_radius_mm": "1", "valid_band_nm": {"min_nm": "500", "max_nm": "1100", "status": "SPECIFIED"},
        "approximations": ["NOMINAL_PRESCRIPTION", "IDEAL_SURFACE_FIGURE", "GHOST_REFLECTIONS_NOT_TRACED", "EDGE_AS_OPAQUE_RIM"],
        "status": "SPECIFIED"}}


def grin_project(length: float = QUARTER, *, height: float = 0.0, waist: float = 0.1, planes=None, quaternion=ALONG_X) -> dict:
    """The rod from x = 0 along design +X (its local Z), a 633 nm Gaussian source 20 mm before it at ``height`` along design Y,
    its waist on the entrance face, and the plane OUT 1 um behind the exit face."""

    src = source(position=(-20.0, height, 0.0), wavelength_nm=WAVELENGTH, waist_mm=waist, waist_position_mm=20.0)
    comp = {"instance_id": "G1", "optical_role": "lens", "custom_part": {"custom_part_key": "grin", "description": "GRIN rod"},
            "optical_model_id": "ROD", "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": list(quaternion)}}
    document = project([src, comp], planes or [plane("OUT", (round(length, 9) + 0.001, 0.0, 0.0), radius_mm=5.0)])
    document["optical_models"] = [rod(length)]
    document["model_materials"] = [material()]
    return document


def arrival(result, oid: str = "OUT"):
    (only,) = next(o for o in result.observations if o.observation_id == oid).arrivals
    return only


def test_a_declared_rod_lowers_to_a_gradient_index_material_about_its_axis() -> None:
    request = lower_project(grin_project(), catalog=None).request
    (grin,) = [m for m in request.project.materials if isinstance(m, GradientIndexMaterial)]
    assert grin.material_id == "project:GRIN@G1" and grin.axis_component == "G1"
    assert (grin.axis_point_mm, grin.axis_direction, grin.radius_mm, grin.gradient.per_mm) == ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), RADIUS, G)


def test_a_ray_through_a_declared_rod_follows_the_ray_equation() -> None:
    """A beam 0.25 mm off the axis through a rod of 0.8 pitch: the chief ray's exit point inside the rod, its optical path and
    length in the rod against the integrated ray equation; the pieces of the recorded path lie on the curved ray."""

    length = 0.8 * 2.0 * math.pi / G
    result = run_optical_project(grin_project(length, height=0.25), catalog=None).result
    assert result.power_budget.conserved
    (path,) = result.paths
    inside = [s for s in path.segments if s.medium_id == "project:GRIN@G1"]
    assert len(inside) == math.ceil(0.8 * 16)
    reference = ode_ray(np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), round(length, 9))
    # (design frame: local Z = design X, local X = design Y)
    np.testing.assert_allclose(inside[-1].end_mm, (reference["point"][2], reference["point"][0], reference["point"][1]), atol=1e-9)
    assert sum(s.optical_path_length_mm for s in inside) == pytest.approx(reference["opl"], rel=1e-11)
    assert sum(s.length_mm for s in inside) == pytest.approx(reference["length"], rel=1e-11)
    for segment in inside:
        y = segment.end_mm[1]
        z = segment.end_mm[0]
        ray = ode_ray(np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), z) if z > 0 else None
        if ray is not None:
            assert y == pytest.approx(ray["point"][0], abs=1e-9)
    out = arrival(result)
    exit_slope = reference["direction"][0]
    expected = np.array([math.sqrt(1.0 - (N0 * math.sqrt(1 - G * G * reference["point"][0] ** 2) * exit_slope) ** 2),
                         N0 * math.sqrt(1 - G * G * reference["point"][0] ** 2) * exit_slope, 0.0])
    np.testing.assert_allclose(out.beam.direction, expected, atol=1e-9)


def gaussian_q_after(length: float, waist: float, delta: float) -> complex:
    """The on-axis beam's q in air ``delta`` behind the rod's exit face, by hand: a waist on the entrance face, reduced
    ``q = i z_R`` (continuous across a flat face), the GRIN matrix, then ``delta`` of air."""

    z_r = math.pi * waist * waist / (WAVELENGTH * 1e-6)
    a, b = math.cos(G * length), math.sin(G * length) / (N0 * G)
    c, d = -N0 * G * math.sin(G * length), math.cos(G * length)
    q = complex(0.0, z_r)
    return (a * q + b) / (c * q + d) + delta


@pytest.mark.parametrize("pitch", [0.25, 0.23, 0.5, 0.29])
def test_the_gaussian_beam_through_a_rod_takes_the_grin_matrix(pitch: float) -> None:
    length = pitch * 2.0 * math.pi / G
    result = run_optical_project(grin_project(length), catalog=None).result
    beam = arrival(result).beam
    assert beam.gaussian.valid
    expected = gaussian_q_after(round(length, 9), 0.1, 0.001)
    for q in (beam.gaussian.q_x_mm, beam.gaussian.q_y_mm):
        assert complex(*q) == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_an_off_axis_beam_keeps_a_valid_gaussian_state_on_its_meridional_plane() -> None:
    """0.2 mm off the axis: the transfer about the chief ray decouples on its meridional plane; the beam's waist behind a
    quarter pitch is close to the on-axis one (the off-axis ray's invariant beta differs from n0 by (g r)^2 / 2)."""

    beam = arrival(run_optical_project(grin_project(height=0.2), catalog=None).result).beam
    on_axis = arrival(run_optical_project(grin_project(), catalog=None).result).beam
    assert beam.gaussian.valid
    assert beam.gaussian.beam_radius_x_mm == pytest.approx(on_axis.gaussian.beam_radius_x_mm, rel=0.05)


def test_the_paraxial_matrix_of_a_rod_is_the_grin_matrix() -> None:
    for pitch in (0.23, 0.25, 0.5):
        length = round(pitch * 2.0 * math.pi / G, 9)
        project_ = ProjectDocumentV2.model_validate(grin_project(length))
        element = compile_declared("G1", project_.optical_model("ROD"), ProjectResolver(project_))
        (a, b), (c, d) = paraxial_matrix(element, WAVELENGTH)
        assert (a, b, c, d) == pytest.approx((math.cos(G * length), math.sin(G * length) / (N0 * G), -N0 * G * math.sin(G * length),
                                              math.cos(G * length)), abs=1e-12)
        if pitch != 0.5:
            assert -1.0 / c == pytest.approx(1.0 / (N0 * G * math.sin(G * length)), rel=1e-12)


def _turned_about_z(angle_rad: float, quaternion) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2)
    w2, x2, y2, z2 = quaternion
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def test_the_rod_s_side_and_surfaces_inside_it_end_the_path_out_of_model_scope() -> None:
    """A beam entering 0.6 mm off the axis at 0.4 rad: its ray's amplitude in the rod exceeds the 1 mm radius, so it reaches the
    side; an observation plane across the rod is met inside it. Both end the path out of model scope."""

    document = grin_project(height=0.0)
    src = document["components"][0]
    src["nominal_pose"]["quaternion_wxyz"] = list(_turned_about_z(0.4, ALONG_X))
    src["nominal_pose"]["position_mm"] = [-1.0, 0.6 - math.tan(0.4), 0.0]
    src["source_model"].update(waist_position_mm=1.0 / math.cos(0.4), waist_radius_mm=0.02)
    run = run_optical_project(document, catalog=None)
    result = run.result
    side = [i for i in result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    assert side and "reaches the rod's side" in side[0].message
    assert result.power_budget.conserved
    assert run.status == "OUT_OF_MODEL_SCOPE"
    inside = grin_project(QUARTER, planes=[plane("OUT", (QUARTER + 0.001, 0.0, 0.0), radius_mm=5.0), plane("MID", (1.0, 0.0, 0.0), radius_mm=5.0)])
    result = run_optical_project(inside, catalog=None).result
    issues = [i for i in result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    assert issues and "observation/MID" in issues[0].message
    assert result.paths[0].termination == "GRADIENT_INDEX_UNSUPPORTED"


def test_an_observation_plane_in_a_gradient_medium_is_refused() -> None:
    document = grin_project()
    document["observations"][0]["medium"] = "GRIN"
    with pytest.raises(LocalRunError, match="lies inside the gradient-index medium GRIN"):
        lower_project(document, catalog=None)


def test_a_rod_turned_and_placed_traces_the_same() -> None:
    """The same rod and source turned 30 deg about design Z and moved: the beam's power, path length and q are the same."""

    base = arrival(run_optical_project(grin_project(height=0.2), catalog=None).result)
    angle = math.radians(30.0)
    turn = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    def rotate(v):
        c, s = math.cos(angle), math.sin(angle)
        return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]

    document = grin_project(height=0.2)
    shift = np.array([3.0, -2.0, 1.0])
    for comp in document["components"]:
        pose = comp["nominal_pose"]
        pose["position_mm"] = list(np.array(rotate(pose["position_mm"])) + shift)
        pose["quaternion_wxyz"] = list(mul(turn, tuple(pose["quaternion_wxyz"])))
    for obs in document["observations"]:
        obs["pose"]["position_mm"] = list(np.array(rotate(obs["pose"]["position_mm"])) + shift)
        obs["pose"]["quaternion_wxyz"] = list(mul(turn, tuple(obs["pose"]["quaternion_wxyz"])))
    moved = arrival(run_optical_project(document, catalog=None).result)
    assert moved.beam.power_W == pytest.approx(base.beam.power_W, rel=1e-12)
    assert moved.beam.gaussian.beam_radius_x_mm == pytest.approx(base.beam.gaussian.beam_radius_x_mm, rel=1e-6)
    np.testing.assert_allclose(moved.beam.direction, rotate(base.beam.direction), atol=1e-9)


def test_the_native_rod_states_its_pitch_and_focal_length_and_focuses_on_its_exit_face() -> None:
    from os_optics.native_library import NativeComponentError, add_to_project, build_native_model

    parameters = {"material": material(), "design_wavelength_nm": "633", "diameter_mm": "1.8", "pitch": "0.25"}
    build = build_native_model("lens.grin_rod", parameters)
    length = float(build.design["length_mm"])
    assert length == pytest.approx(math.ceil(QUARTER * 1000.0) / 1000.0, abs=1e-12)
    # (the native library designs for air at 20 C and 101325 Pa: its focal lengths take the axis index relative to that air)
    n0 = N0 / air_index(633.0)
    assert float(build.design["effective_focal_length_mm"]) == pytest.approx(1.0 / (n0 * G * math.sin(G * length)), rel=1e-6)
    assert float(build.design["back_focal_distance_mm"]) == pytest.approx(math.cos(G * length) / (n0 * G * math.sin(G * length)), abs=1e-6)
    src = source(position=(-20.0, 0.0, 0.0), wavelength_nm=WAVELENGTH, waist_mm=0.1, waist_position_mm=20.0)
    document = add_to_project(project([src], [plane("OUT", (length + 0.001, 0.0, 0.0), radius_mm=5.0)]), build, instance_id="G1",
                              position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)
    run = run_optical_project(document, catalog=None)
    beam = arrival(run.result).beam
    assert complex(*beam.gaussian.q_x_mm) == pytest.approx(gaussian_q_after(length, 0.1, 0.001), rel=1e-7, abs=1e-9)
    assert run.model_scope_issues == () and run.status == run.engine_status
    with pytest.raises(NativeComponentError, match="exactly one of length_mm"):
        build_native_model("lens.grin_rod", {k: v for k, v in parameters.items() if k != "pitch"})
    with pytest.raises(NativeComponentError, match="a GRIN rod's material is a gradient-index material"):
        build_native_model("lens.grin_rod", {**parameters, "material": {"constant_index": "1.6", "valid_band_nm": [400, 1700]}})


# ------------------------------------------------------------------------------------ the run's model scope (TR-1)


@pytest.mark.parametrize("pitch, height, behind", [(0.25, 0.0, 0.001), (0.8, 0.25, 0.001), (0.25, 0.0, 5.0), (0.5, 0.0, 30.0)])
def test_a_beam_through_a_rod_and_out_of_its_end_face_is_within_the_model_scope(pitch: float, height: float, behind: float) -> None:
    """The chief ray in the rod is recorded as straight pieces along its curved path (all but the last end inside the
    medium, where the next piece starts); it leaves through the end face S2, so the run is graded on the Engine's own result."""

    length = pitch * 2.0 * math.pi / G
    run = run_optical_project(grin_project(length, height=height, planes=[plane("OUT", (round(length, 9) + behind, 0.0, 0.0), radius_mm=5.0)]),
                              catalog=None)
    (path,) = run.result.paths
    inside = [s for s in path.segments if s.medium_id == "project:GRIN@G1"]
    assert len(inside) > 1 and [s.end_surface_id for s in inside] == [None] * (len(inside) - 1) + ["G1/S2"]
    assert run.model_scope_issues == ()
    assert run.engine_status == "PASS" and run.status == "PASS"


def test_a_piece_of_the_rod_s_path_that_does_not_continue_is_an_edge_exit() -> None:
    """The same check on the recorded path cut after a piece that ends inside the rod (nothing continues from there, as for a
    beam that left the medium without meeting a surface): the light leaves the rod through its edge."""

    from os_optics.local.validity import element_scope_issues

    document = grin_project(0.8 * 2.0 * math.pi / G, height=0.25)
    lowered = lower_project(document, catalog=None)
    result = run_optical_project(document, catalog=None).result
    (path,) = result.paths
    cut = next(k for k, s in enumerate(path.segments) if s.medium_id == "project:GRIN@G1" and s.end_surface_id is None)
    segments = path.segments[:cut + 2]
    segments = (*segments[:-1], segments[-1].model_copy(update={"medium_id": "air"}))  # (the piece after it is not in the rod)
    broken = result.model_copy(update={"paths": (path.model_copy(update={"segments": segments}),)})
    codes = [(i.instance_id, i.surface_id, i.code) for i in element_scope_issues(broken, lowered.elements)]
    assert ("G1", "S1", "RAY_LEFT_ELEMENT_THROUGH_EDGE") in codes
    assert element_scope_issues(result, lowered.elements) == ()


# ------------------------------------------------------------------------------- light leaving through the rod's side (BND-2)


def _side_share(tilt: float, waist: float, rays: int = 12000, seed: int = 7) -> float:
    """Independent reference: the share of a TEM00 beam (waist ``waist`` on the entrance face's centre, tilted ``tilt`` rad in
    the design XY plane) whose rays reach the rod's side (r > ``RADIUS``) within one pitch. Rays sampled from the beam's Wigner
    distribution at its waist (positions sigma w0/2, angles sigma lambda/(2 pi w0) per axis), carried straight to the face
    plane, refracted by vector Snell into n(r), then the ray equation d(n t)/ds = grad n integrated in arc length by
    fixed-step RK4 (not the closed form under test)."""
    rng = np.random.default_rng(seed)
    lam = WAVELENGTH * 1e-6
    b = np.array([math.cos(tilt), math.sin(tilt), 0.0])
    e1, e2 = np.array([-math.sin(tilt), math.cos(tilt), 0.0]), np.array([0.0, 0.0, 1.0])
    u, v = rng.normal(0.0, waist / 2.0, rays), rng.normal(0.0, waist / 2.0, rays)
    a1, a2 = rng.normal(0.0, lam / (2.0 * math.pi * waist), (2, rays))
    p = u[:, None] * e1 + v[:, None] * e2
    d = b[None, :] + a1[:, None] * e1 + a2[:, None] * e2
    d /= np.linalg.norm(d, axis=1)[:, None]
    q = p - (p[:, 0] / d[:, 0])[:, None] * d  # (to the face plane x = 0)

    def index(r):
        return N0 * np.sqrt(1.0 - G * G * (r[:, 1] ** 2 + r[:, 2] ** 2))

    n1 = index(q)
    cos_i = d[:, 0]
    eta = 1.0 / n1
    cos_t = np.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))
    t = eta[:, None] * d + (eta * cos_i - cos_t)[:, None] * np.array([-1.0, 0.0, 0.0])
    r, mom = q.copy(), n1[:, None] * t

    def rhs(r, mom):
        n = index(r)
        grad = np.zeros_like(r)
        grad[:, 1], grad[:, 2] = -N0 * N0 * G * G * r[:, 1] / n, -N0 * N0 * G * G * r[:, 2] / n
        return mom / n[:, None], grad

    length = 2.0 * math.pi / G
    reach = np.zeros(rays)
    ds = 0.01
    for _ in range(int(3.0 * length / ds)):
        live = r[:, 0] < length
        if not live.any():
            break
        k1r, k1p = rhs(r, mom)
        k2r, k2p = rhs(r + 0.5 * ds * k1r, mom + 0.5 * ds * k1p)
        k3r, k3p = rhs(r + 0.5 * ds * k2r, mom + 0.5 * ds * k2p)
        k4r, k4p = rhs(r + ds * k3r, mom + ds * k3p)
        step_r = ds / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        step_p = ds / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        r = np.where(live[:, None], r + step_r, r)
        mom = np.where(live[:, None], mom + step_p, mom)
        inside = live & (r[:, 0] < length)
        reach = np.where(inside, np.maximum(reach, r[:, 1] ** 2 + r[:, 2] ** 2), reach)
        r[:, 1:] = np.where((reach >= RADIUS * RADIUS)[:, None], 0.0, r[:, 1:])  # (gone: kept inside the index's domain)
    return float(np.mean(reach >= RADIUS * RADIUS))


def _tilted(tilt: float, waist: float, *, stop=None) -> dict:
    """``grin_project`` over one pitch with the beam's waist on the entrance face's centre, tilted ``tilt`` rad in the design XY
    plane; a ``power_W`` minimum of 0.8 mW on OUT; ``stop``: (distance before the face, opening radius) of a stop across the
    beam."""
    from declared_helpers import stop_model

    length = 2.0 * math.pi / G
    document = grin_project(length, planes=[plane("OUT", (round(length, 9) + 0.001, 0.0, 0.0), radius_mm=5.0)])
    src = document["components"][0]
    src["nominal_pose"]["quaternion_wxyz"] = list(_turned_about_z(tilt, ALONG_X))
    src["nominal_pose"]["position_mm"] = [-1.0, -math.tan(tilt), 0.0]
    src["source_model"].update(waist_position_mm=1.0 / math.cos(tilt), waist_radius_mm=waist)
    if stop is not None:
        back, opening = stop
        document["components"].insert(1, {
            "instance_id": "STOP", "optical_role": "aperture", "custom_part": {"custom_part_key": "own-stop", "description": "Stop."},
            "optical_model_id": "stop", "nominal_pose": {"position_mm": [-back * math.cos(tilt), -back * math.sin(tilt), 0.0],
                                                         "quaternion_wxyz": list(_turned_about_z(tilt, ALONG_X))}})
        document["optical_models"].append(stop_model("stop", opening_radius_mm=repr(opening), edge_radius_mm="0.3"))
    document["targets"] = [{"target_id": "min_out", "metric": "power_W", "observation_id": "OUT", "minimum": 8e-4, "unit": "W"}]
    return document


def test_light_leaving_the_rod_through_its_side_puts_the_power_behind_it_out_of_model_scope() -> None:
    """The beam tilted 0.5 rad with a 0.02 mm waist: its chief ray peaks 0.9988 mm from the axis of the 1 mm rod and stays in,
    but 44 % of the beam (the reference) reaches the side. Its pupil rays that do are not followed, so the power behind the
    rod is out of model scope -- the plane's power, the run, and a ``power_W`` minimum the whole beam would meet (it read PASS
    and MET with the full 9.28e-4 W) -- and the issue states the samples' share against the reference. A tail beyond the
    samples is caught by the beam's envelope."""

    share = _side_share(0.5, 0.02)
    assert 0.40 < share < 0.50
    run = run_optical_project(_tilted(0.5, 0.02), catalog=None)
    result = run.result
    (side,) = [i for i in result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    context = side.numerical_context
    assert side.status.value == "OUT_OF_MODEL_SCOPE" and "reach the rod's side" in side.message
    assert context["pupil_rays_at_side"] == context["pupil_rays_left"] > 0
    assert abs(context["left_fraction"] - share) < 0.1
    (only,) = result.observations[0].arrivals
    assert "G1/S2" in only.power_out_of_scope_at
    assert result.observations[0].quantity_state == "OUT_OF_MODEL_SCOPE"
    (target,) = result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and target.assessment.value == "INDETERMINATE"
    upper = result.observations[0].power_upper_bound_W
    assert upper is None or upper >= only.beam.power_W * (1.0 - share)
    assert result.power_budget.conserved
    assert run.status == "OUT_OF_MODEL_SCOPE"
    # (0.49 rad: 1.7 % of the beam reaches the side (the reference), all of it beyond the pupil samples: the beam's envelope
    # does -- it read PASS)
    tail = _side_share(0.49, 0.02)
    assert 0.01 < tail < 0.03
    run = run_optical_project(_tilted(0.49, 0.02), catalog=None)
    (side,) = [i for i in run.result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    assert side.numerical_context["pupil_rays_left"] == 0 and side.numerical_context["envelope_sigmas"] > 3.0
    assert "envelope" in side.message and run.status == "OUT_OF_MODEL_SCOPE"
    assert run.result.targets[0].assessment.value == "INDETERMINATE"
    # (a beam the rod holds whole: 0.45 rad, the reference finds no ray at the side; nothing is flagged)
    assert _side_share(0.45, 0.02) == 0.0
    run = run_optical_project(_tilted(0.45, 0.02), catalog=None)
    assert not [i for i in run.result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    assert run.status == "PASS" and run.result.targets[0].assessment.value == "MET"


def test_on_the_pupil_sample_basis_the_share_leaving_the_side_is_booked() -> None:
    """The same beam behind a stop that cuts 15 % of it (no valid Gaussian state behind it: the pupil samples carry the power):
    the samples that reach the rod's side take their share out of the beam, booked as light whose fate is not followed
    (``TERMINATED_OUT_OF_MODEL``), the ledger closes, and the power behind the rod stays out of model scope."""

    result = run_optical_project(_tilted(0.5, 0.02, stop=(0.5, 0.02)), catalog=None).result
    (side,) = [i for i in result.issues if i.code.value == "GRADIENT_INDEX_UNSUPPORTED"]
    context = side.numerical_context
    (only,) = result.observations[0].arrivals
    assert only.beam.gaussian is None or not only.beam.gaussian.valid
    assert context["pupil_rays_at_side"] == context["pupil_rays_left"] > 0 and context["left_power_W"] > 0.0
    (line,) = result.power_budget.lines
    # (the samples' share of the power that entered the rod leaves the beam there; the rest goes on to the exit face)
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(context["left_power_W"], rel=1e-12)
    entered = context["left_power_W"] / context["left_fraction"]
    assert only.beam.power_W <= entered * (1.0 - context["left_fraction"]) * (1.0 + 1e-9)
    assert "G1/S2" in only.power_out_of_scope_at
    assert result.power_budget.conserved and line.conserved
