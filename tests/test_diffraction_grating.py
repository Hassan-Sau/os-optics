"""The diffraction grating (``DIFFRACTION_GRATING``), against grating equations derived here, not through Engine functions.

Sign convention under test (``DiffractionGratingModelV1``): with ``K`` the unit grating vector (local +X at the default angle),
``L`` the period and ``lambda`` the vacuum wavelength, an order ``m`` leaves with ``n_exit t_exit = n_in t_in + m (lambda / L) K``,
``t`` being a direction's component in the plane. With an angle measured in the plane of ``K`` and the normal, positive toward
``+K`` on the side the light leaves or arrives:

    n_exit sin(theta_m) = n_in sin(theta_i) + m lambda / L,   sin(theta_i) = d_in . K.

The reference planes of the issue's cases follow from this by hand (every number below is computed in this file with ``math``):

* G1, reflection, 600 lines/mm, theta_i = +10 deg, a screen 120 mm from the grating across the specular direction: the screen
  coordinate is ``u_m = 120 tan(theta_m - theta_0)`` with ``theta_0 = theta_i``. Published reference: +1 at 35.174 / 44.250 /
  54.203 mm for 450 / 550 / 650 nm, separation 19.029 mm.
* G2, transmission, 300 lines/mm, normal incidence, a screen 120 mm behind it: ``u_m = 120 tan(theta_m)``, +-16.35 / +-20.08 /
  +-23.86 mm.
* G3, Littrow, 600 lines/mm: ``2 sin(theta_L) = lambda / L``, 7.7586 / 9.4972 / 11.2447 deg; a pose fixed at the 550 nm angle sends
  450 / 550 / 650 nm in order -1 at ``|theta_m| - theta_L`` from the retro direction, -3.4701 / 0 / +3.5056 deg.
* G4, grism, derived here: a glass prism of apex ``A`` entered at normal incidence and a grating in air behind it whose plane is
  across the undeviated axis (see ``test_g4_grism_prism_then_grating_on_one_path``).

Tolerances, chosen before running. Positions on a screen: 1e-9 mm of the closed form (double-precision rays over 100 mm carry
about 1e-13 mm per surface; the margin is for the few operations between). Angles: 1e-9 deg. The published reference numbers are
stated to three (G1, G3 four) decimals: 0.0005 mm and 0.00005 deg. Powers: 1e-12 relative (double-precision products of a
few factors). A tolerance of the optical bench does not enter: these are model-to-closed-form comparisons of one
ideal plane grating. Not validated here: any efficiency of a real grating; no measured or independently computed response
data exists in this repository, so the tables below are explicit fixtures (``ASSUMED``) or test the provenance handling only.
"""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from os_optics._vendor.shared_contracts.optical_design import ProjectOpticalModelV1
from os_optics.contracts.optics import DiffractionGratingInteraction, GratingOrderTable
from os_optics.local import PROJECT_OPS, json_api, run_optical_project
from os_optics.optical_view import optic_mesh, section
from os_optics.results import load_run

from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import ALONG_X, plane, project, source

MM = 1e-9
DEG = 1e-9
REL = 1e-12
POWER_W = 0.002
SHA = "cd" * 32
EVIDENCE = {"method": "fixture values standing for a measurement", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": SHA,
            "fit_model": "the stated rows", "validation_summary": "fixture", "date": "2026-10-30"}


# ------------------------------------------------------------------------------------------------- hand geometry


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _quat(x, y, z) -> tuple[float, float, float, float]:
    """The unit quaternion of the rotation with columns ``x, y, z``."""

    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def _vec(a, scale_a, b=None, scale_b=0.0):
    return tuple(scale_a * a[i] + (0.0 if b is None else scale_b * b[i]) for i in range(3))


def reflection_pose(theta_i_deg: float):
    """A reflection grating at the origin met by a beam along design +X at signed incidence ``theta_i``: its ruled face's normal
    ``N`` faces the beam, the grating vector ``K`` lies in the design XY plane and ``d . K = sin(theta_i)``. Returns the
    pose's quaternion, ``K`` and ``N``."""

    phi = math.radians(theta_i_deg)
    normal, k = (-math.cos(phi), math.sin(phi), 0.0), (math.sin(phi), math.cos(phi), 0.0)
    return _quat(k, _cross(normal, k), normal), k, normal


def screen_across(direction, k, normal, distance: float, name: str = "SCREEN", radius: float = 300.0) -> dict:
    """A screen ``distance`` from the origin across ``direction`` (in the plane of ``K`` and ``N``); its u axis turns from ``direction``
    toward larger angles from the normal."""

    u = _vec(k, float(sum(direction[i] * normal[i] for i in range(3))), normal, -float(sum(direction[i] * k[i] for i in range(3))))
    return plane(name, _vec(direction, distance), _quat(u, _cross(direction, u), direction), radius_mm=radius)


def grating(model_id: str = "g", *, mode: str = "REFLECTION", density: str = "600", orders=(-1, 0, 1), **extra) -> dict:
    model = {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": mode, "line_density_per_mm": density, "orders": list(orders),
             "clear_half_widths_mm": ["12", "12"], "edge_half_widths_mm": ["12.5", "12.5"], "status": "SPECIFIED", **extra}
    if "period_nm" in extra:
        del model["line_density_per_mm"]
    if "all_propagating_orders" in extra:
        del model["orders"]
    return {"model_id": model_id, "model": model}


def laser(wavelength_nm: float, name: str = "SRC", *, position=(-100.0, 0.0, 0.0), polarization_deg: float = 0.0, power_w: float = POWER_W) -> dict:
    return source(name, position, wavelength_nm=wavelength_nm, waist_mm=0.5, polarization_deg=polarization_deg, power_w=power_w)


def run(components, observations, models, targets=None, materials=None, **kwargs):
    document = with_models(project(components, observations, targets), models, materials, design_intent="OPTICAL_SYSTEM")
    return run_optical_project(document, catalog=None, **kwargs)


def order_of(arrival) -> int:
    (step,) = arrival.grating_orders
    return step.order


def by_order(result, observation_id: str = "SCREEN") -> dict[int, object]:
    obs = next(o for o in result.observations if o.observation_id == observation_id)
    return {order_of(a): a for a in obs.arrivals}


def orders_table(result) -> dict[int, object]:
    return {row.order: row for row in result.grating_orders}


def theta_m(theta_i_deg: float, m: int, wavelength_nm: float, density_per_mm: float, n_in: float = 1.0, n_out: float = 1.0) -> float:
    """Deg, from ``n_out sin(theta_m) = n_in sin(theta_i) + m lambda / L``."""

    return math.degrees(math.asin((n_in * math.sin(math.radians(theta_i_deg)) + m * wavelength_nm * 1e-6 * density_per_mm) / n_out))


# ---------------------------------------------------------------------------------------------- reference cases


@pytest.mark.parametrize("wavelength_nm, published", [(450.0, 35.174), (550.0, 44.250), (650.0, 54.203)])
def test_g1_reflection_positions_on_a_screen_across_the_specular_direction(wavelength_nm, published):
    pose, k, normal = reflection_pose(10.0)
    specular = _vec(k, math.sin(math.radians(10.0)), normal, math.cos(math.radians(10.0)))
    result = run([laser(wavelength_nm), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)],
                 [screen_across(specular, k, normal, 120.0)], [grating()]).result
    arrivals = by_order(result)
    for m, arrival in arrivals.items():
        expected = 120.0 * math.tan(math.radians(theta_m(10.0, m, wavelength_nm, 600.0) - 10.0))
        assert arrival.uv_mm[0] == pytest.approx(expected, abs=MM), m
        assert arrival.uv_mm[1] == pytest.approx(0.0, abs=MM)
    assert arrivals[1].uv_mm[0] == pytest.approx(published, abs=0.0005)
    assert arrivals[0].uv_mm[0] == pytest.approx(0.0, abs=MM)  # the zero order is the specular reflection, at any wavelength
    assert orders_table(result)[1].exit_angle_deg == pytest.approx(theta_m(10.0, 1, wavelength_nm, 600.0), abs=DEG)


def test_g1_separation_of_the_first_order_across_the_band():
    pose, k, normal = reflection_pose(10.0)
    specular = _vec(k, math.sin(math.radians(10.0)), normal, math.cos(math.radians(10.0)))
    lasers = [laser(w, f"L{int(w)}") for w in (450.0, 650.0)]
    result = run([*lasers, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [screen_across(specular, k, normal, 120.0)], [grating(orders=(1,))]).result
    (obs,) = result.observations
    positions = sorted(a.uv_mm[0] for a in obs.arrivals)
    assert positions[1] - positions[0] == pytest.approx(19.029, abs=0.001)


def test_the_separation_of_two_lines_is_a_target_with_no_efficiency_stated():
    """A geometric pair metric reads two orders' positions: the G1 separation, decided though neither order's power is known."""

    pose, k, normal = reflection_pose(10.0)
    specular = _vec(k, math.sin(math.radians(10.0)), normal, math.cos(math.radians(10.0)))
    lasers = [laser(w, f"L{int(w)}") for w in (450.0, 650.0)]
    target = {"target_id": "SEP", "observation_id": "SCREEN", "metric": "centroid_separation_mm", "unit": "mm", "source_id": "L450", "source_id_b": "L650",
              "target_value": 19.029, "tolerance_abs": 0.001}
    result = run([*lasers, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [screen_across(specular, k, normal, 120.0)], [grating(orders=(1,))], targets=[target]).result
    (found,) = result.targets
    assert found.status.value == "PASS"
    assert found.value == pytest.approx(120.0 * (math.tan(math.radians(theta_m(10.0, 1, 650.0, 600.0) - 10.0)) - math.tan(math.radians(theta_m(10.0, 1, 450.0, 600.0) - 10.0))), abs=MM)


@pytest.mark.parametrize("wavelength_nm, published", [(450.0, 16.35), (550.0, 20.08), (650.0, 23.86)])
def test_g2_transmission_positions_both_signs_and_zero_order(wavelength_nm, published):
    # local +Z along design +X: local +X (the grating vector) is design +Y
    result = run([laser(wavelength_nm), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)],
                 [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [grating(mode="TRANSMISSION", density="300")]).result
    arrivals = by_order(result)
    for m in (-1, 1):
        expected = 120.0 * math.tan(math.radians(theta_m(0.0, m, wavelength_nm, 300.0)))
        assert arrivals[m].uv_mm[0] == pytest.approx(expected, abs=MM)
        assert arrivals[m].uv_mm[0] == pytest.approx(m * published, abs=0.005)
    assert arrivals[0].uv_mm == pytest.approx((0.0, 0.0), abs=MM)


@pytest.mark.parametrize("wavelength_nm, littrow, offset", [(450.0, 7.7586, -3.4701), (550.0, 9.4972, 0.0), (650.0, 11.2447, 3.5056)])
def test_g3_littrow_angles_and_the_fixed_pose_offsets_from_retro(wavelength_nm, littrow, offset):
    assert math.degrees(math.asin(wavelength_nm * 1e-6 * 600.0 / 2.0)) == pytest.approx(littrow, abs=0.00005)
    theta_l = math.degrees(math.asin(550.0 * 1e-6 * 600.0 / 2.0))
    # order -1 with a positive incidence angle returns toward the source side
    pose, k, normal = reflection_pose(theta_l)
    retro = (-1.0, 0.0, 0.0)
    result = run([laser(wavelength_nm), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)],
                 [plane("BACK", (-40.0, 0.0, 0.0), _quat((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), radius_mm=300.0)], [grating(orders=(-1,))]).result
    row = orders_table(result)[-1]
    expected = theta_m(theta_l, -1, wavelength_nm, 600.0)
    assert row.exit_angle_deg == pytest.approx(expected, abs=DEG)
    assert abs(expected) - theta_l == pytest.approx(offset, abs=0.00005)
    # the design-frame direction reads the same offset from retro (acos near 1 resolves about 1e-6 deg)
    angle = math.degrees(math.acos(min(1.0, sum(row.direction[i] * retro[i] for i in range(3)))))
    assert angle == pytest.approx(abs(abs(expected) - theta_l), abs=1e-5)
    (arrival,) = [a for a in next(o for o in result.observations if o.observation_id == "BACK").arrivals if a.grating_orders]
    if wavelength_nm == 550.0:  # the Littrow line returns on itself: the plane reads it on the incident axis
        assert arrival.uv_mm[0] == pytest.approx(0.0, abs=MM)


def test_g4_grism_prism_then_grating_on_one_path():
    """A wedge prism (``prism.wedge``: entry face across the beam, exit face tilted by apex ``A``, of constant index ``n``) and a
    transmission grating in air behind it, its plane across the design axis. Hand derivation, for a ray along +X through the
    wedge's axis:

    * entry at normal incidence: undeviated, along +X inside the glass;
    * exit, at ``x_e`` on the axis: incidence ``A``, Snell ``sin(r) = n sin(A)``; it leaves at ``delta = r - A`` from +X toward the
      wedge's base, local -X, which is design -Y for a component placed ``ALONG_X``;
    * grating, normal +X, its grating vector ``K`` local +X (design +Y): the signed incidence is ``-delta``, so
      ``sin(theta_m) = sin(-delta) + m lambda / L``, reached at ``y_g = (x_g - x_e) tan(-delta)``;
    * a screen across +X at ``x_s``: ``y = y_g + (x_s - x_g) tan(theta_m)``.

    The issue's prism-only numbers carry an unexplained 0.105-0.108 deg offset and are not used; this case is derived whole here. The
    wedge's own deviation is read from the case too: with the grating's order 0 the angle leaving the grating is ``-delta``."""

    n, apex = 1.5, math.radians(12.0)
    grating_x, screen_x = 40.0, 160.0
    built = json_api.build_native_model("prism.wedge", {"material": {"constant_index": "1.5", "valid_band_nm": ["400", "800"]}, "wedge_angle_deg": "12", "diameter_mm": "20",
                                                        "design_wavelength_nm": "550", "model_id": "wedge"})
    assert built["ok"] is True, built
    x_exit = float(built["optical_model"]["model"]["surfaces"][1]["origin_mm"][2])
    delta = math.asin(n * math.sin(apex)) - apex
    for wavelength_nm in (500.0, 600.0):
        document = with_models(project(
            [laser(wavelength_nm), declared_component("W", "other_optical", "wedge", (0.0, 0.0, 0.0), ALONG_X),
             declared_component("G", "other_optical", "g", (grating_x, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (screen_x, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
            [built["optical_model"], grating(mode="TRANSMISSION", density="300")], built["model_materials"], design_intent="OPTICAL_SYSTEM")
        result = run_optical_project(document, catalog=None).result
        arrivals, rows = by_order(result), orders_table(result)
        assert set(arrivals) == {-1, 0, 1}
        for m, arrival in arrivals.items():
            theta = math.asin(math.sin(-delta) + m * wavelength_nm * 1e-6 * 300.0)
            y_grating = (grating_x - x_exit) * math.tan(-delta)
            assert arrival.uv_mm[0] == pytest.approx(y_grating + (screen_x - grating_x) * math.tan(theta), abs=MM), (wavelength_nm, m)
            assert rows[m].incidence_deg == pytest.approx(math.degrees(-delta), abs=DEG)
            assert rows[m].exit_angle_deg == pytest.approx(math.degrees(theta), abs=DEG)
        assert rows[0].exit_angle_deg == pytest.approx(math.degrees(-delta), abs=DEG)


def test_grating_with_an_incident_medium_of_index_n():
    """Light in a medium of index ``n`` meets a grating leaving into air: ``sin(theta_m) = n sin(theta_i) + m lambda / L``. The medium is
    entered through the zero order of a first grating at normal incidence (undeviated, ``n`` on its far side); the second grating's
    normal leans by ``phi`` inside it."""

    n, phi = 1.6, 5.0
    medium = {"material_id": "GEL", "display_name": "Fixture medium", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.6"},
              "valid_band_nm": {"min_nm": "400", "max_nm": "800", "status": "SPECIFIED"}}
    entry = grating("entry", mode="TRANSMISSION", density="300", orders=(0,), medium_positive_z="GEL")
    exit_grating = grating("exit", mode="TRANSMISSION", density="300", orders=(-1, 0, 1), medium_negative_z="GEL")
    p_rad = math.radians(phi)
    normal, k = (math.cos(p_rad), -math.sin(p_rad), 0.0), (math.sin(p_rad), math.cos(p_rad), 0.0)  # d . K = sin(phi)
    pose = _quat(k, _cross(normal, k), normal)
    result = run([laser(550.0), declared_component("E", "other_optical", "entry", (0.0, 0.0, 0.0), ALONG_X), declared_component("X", "other_optical", "exit", (30.0, 0.0, 0.0), pose)],
                 [], [entry, exit_grating], materials=[medium]).result
    rows = {row.order: row for row in result.grating_orders if row.surface_id == "X/grating"}
    for m in (-1, 0, 1):
        assert rows[m].incidence_deg == pytest.approx(phi, abs=DEG)
        assert rows[m].exit_angle_deg == pytest.approx(theta_m(phi, m, 550.0, 300.0, n, 1.0), abs=DEG), m


def _dec(value: float) -> str:
    return f"{value:.15f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------------------------- the vector equation itself


def test_conical_mount_follows_the_vector_equation_by_hand():
    """A beam with a component along the grooves: the tangential vector is conserved along the grooves and shifted along K.
    Here the grating lies in the design YZ plane, normal -X, K along +Y (grooves along Z)."""

    phi, psi = math.radians(8.0), math.radians(15.0)  # the beam's direction components along K and the grooves
    d = (math.sqrt(1.0 - math.sin(phi) ** 2 - math.sin(psi) ** 2), math.sin(phi), math.sin(psi))
    normal, k = (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    pose = _quat(k, _cross(normal, k), normal)
    # a source pose whose local +Z is d
    e1 = tuple(c / math.hypot(d[0], d[1]) for c in (-d[1], d[0], 0.0))
    position = _vec(d, -100.0)
    comp = laser(633.0, position=position)
    comp["nominal_pose"]["quaternion_wxyz"] = list(_quat(e1, _cross(d, e1), d))
    table = {}
    for mode, sign in (("REFLECTION", 1.0), ("TRANSMISSION", -1.0)):
        model = grating(mode=mode, density="600", orders=(-1, 0, 1))
        result = run([comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [], [model]).result
        table[mode] = orders_table(result)
        for m in (-1, 0, 1):
            along_k = d[1] + m * 633.0 * 1e-6 * 600.0
            along_g = d[2]
            normal_part = math.sqrt(1.0 - along_k * along_k - along_g * along_g)
            # reflection: back on the -X side (toward +normal); transmission: on on the +X side
            expected = (-normal_part if sign > 0.0 else normal_part, along_k, along_g)
            assert table[mode][m].direction == pytest.approx(expected, abs=1e-12), (mode, m)
            assert table[mode][m].conical_angle_deg == pytest.approx(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), abs=DEG)


def test_a_refractive_exit_medium_enters_the_equation():
    """A transmission grating from air into a medium of index ``n``: ``n sin(theta_m) = sin(theta_i) + m lambda / L``; the normal part
    is ``cos(theta_m)``."""

    n = 1.6
    medium = {"material_id": "GEL", "display_name": "Fixture medium", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.6"},
              "valid_band_nm": {"min_nm": "400", "max_nm": "800", "status": "SPECIFIED"}}
    model = grating(mode="TRANSMISSION", density="600", orders=(-1, 0, 1), medium_positive_z="GEL")
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [], [model], materials=[medium]).result
    rows = orders_table(result)
    for m in (-1, 0, 1):
        assert rows[m].exit_angle_deg == pytest.approx(theta_m(0.0, m, 550.0, 600.0, 1.0, n), abs=DEG)


def test_nonpropagating_orders_are_reported_and_not_traced():
    """At 650 nm and 1200 lines/mm, normal incidence: ``lambda / L = 0.78``: orders 0 and +-1 propagate, +-2 do not."""

    model = grating(mode="TRANSMISSION", density="1200", orders=(-2, -1, 0, 1, 2))
    result = run([laser(650.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)],
                 [plane("SCREEN", (50.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [model]).result
    rows = orders_table(result)
    assert {m: rows[m].propagation for m in rows} == {-2: "NONPROPAGATING", -1: "PROPAGATING", 0: "PROPAGATING", 1: "PROPAGATING", 2: "NONPROPAGATING"}
    for m in (-2, 2):
        row = rows[m]
        assert row.direction is None and row.order_path_id is None and row.followed is False
        assert row.power_state.value == row.polarization_state.value == row.phase_state.value == "NOT_APPLICABLE"
        assert row.power_W is None and row.power_upper_bound_W is None
    assert set(by_order(result)) == {-1, 0, 1}
    assert {i.numerical_context["order"] for i in result.issues if i.code.value == "GRATING_ORDER_NONPROPAGATING"} == {-2, 2}
    assert all(i.status.value == "WARNING" for i in result.issues if i.code.value == "GRATING_ORDER_NONPROPAGATING")


def test_all_propagating_range_follows_what_propagates_and_says_nothing_of_the_rest():
    model = grating(mode="TRANSMISSION", density="1200", all_propagating_orders=(-3, 3))
    result = run([laser(650.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (50.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [model]).result
    assert set(by_order(result)) == {-1, 0, 1}
    assert not [i for i in result.issues if i.code.value == "GRATING_ORDER_NONPROPAGATING"]


def test_a_propagating_order_outside_the_declared_ones_is_reported_as_not_followed():
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)],
                 [plane("SCREEN", (50.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [grating(mode="TRANSMISSION", density="300", orders=(1,))]).result
    rows = orders_table(result)
    assert rows[1].followed is True and rows[0].followed is False and rows[0].propagation == "PROPAGATING"
    assert rows[0].direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12) and rows[0].order_path_id is None
    assert set(by_order(result)) == {1}


def test_the_grating_vector_angle_turns_the_dispersion_about_the_normal():
    """K at 90 deg: local +Y, design +Z at ALONG_X: the order leaves along v (screen's local Y) and not u."""

    model = grating(mode="TRANSMISSION", density="300", orders=(1,), grating_vector_angle_deg="90")
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [model]).result
    arrival = by_order(result)[1]
    assert arrival.uv_mm[0] == pytest.approx(0.0, abs=MM)
    assert arrival.uv_mm[1] == pytest.approx(120.0 * math.tan(math.radians(theta_m(0.0, 1, 550.0, 300.0))), abs=MM)


def test_period_and_line_density_are_the_same_grating():
    a = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [], [grating(mode="TRANSMISSION", density="500", orders=(1,))]).result
    model = grating(mode="TRANSMISSION", period_nm="2000", orders=(1,))
    b = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [], [model]).result
    assert orders_table(a)[1].direction == orders_table(b)[1].direction


def test_full_system_detector_position_behind_a_lens():
    """Grating, then an ideal lens of focal length ``f`` one focal length behind it, a detector at its focus. An ideal lens sends a
    collimated order at angle ``theta`` to ``y = f tan(theta)`` on its focal plane: every wavelength's spot lies at the closed form, the
    spectrum a spectrometer reads, and the zero order on the axis. The source is collimated and the order's power is not known: the
    geometry still gives the detector position."""

    f = 100.0
    lens = {"model_id": "lens", "model": {"model_kind": "IDEAL_THIN_LENS", "model_version": "1", "focal_length_mm": "100", "clear_radius_mm": "40", "edge_radius_mm": "45",
                                          "valid_band_nm": {"min_nm": "300", "max_nm": "900", "status": "SPECIFIED"}, "status": "SPECIFIED"}}
    for wavelength_nm in (450.0, 550.0, 650.0):
        result = run([laser(wavelength_nm), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X),
                      declared_component("L", "lens", "lens", (f, 0.0, 0.0), ALONG_X)],
                     [plane("DET", (2.0 * f, 0.0, 0.0), ALONG_X, radius_mm=60.0)], [grating(mode="TRANSMISSION", density="300", orders=(0, 1)), lens]).result
        (obs,) = result.observations
        got = {order_of(a): a.uv_mm[0] for a in obs.arrivals}
        assert got[0] == pytest.approx(0.0, abs=MM)
        assert got[1] == pytest.approx(f * math.tan(math.radians(theta_m(0.0, 1, wavelength_nm, 300.0))), abs=1e-6)


def test_the_gaussian_beam_follows_the_anamorphic_magnification():
    """In the plane of incidence the beam width scales by ``cos(theta_m) / cos(theta_i)`` (with the source's waist at the grating, the
    plane right behind it); across the plane it does not change."""

    pose, k, normal = reflection_pose(20.0)
    theta = theta_m(20.0, 1, 550.0, 600.0)
    direction = _vec(k, math.sin(math.radians(theta)), normal, math.cos(math.radians(theta)))
    comp = laser(550.0, position=(-200.0, 0.0, 0.0))
    comp["source_model"]["waist_position_mm"] = 200.0  # the waist on the grating
    result = run([comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [screen_across(direction, k, normal, 1e-3, "NEAR")], [grating(orders=(1,))]).result
    (obs,) = result.observations
    (arrival,) = [a for a in obs.arrivals if a.grating_orders]  # (the incident beam crosses this plane too)
    g = arrival.beam.gaussian
    widths = sorted((g.beam_radius_x_mm, g.beam_radius_y_mm))
    ratio = math.cos(math.radians(theta)) / math.cos(math.radians(20.0))
    assert sorted((0.5 * ratio, 0.5)) == pytest.approx(widths, abs=1e-6)
    # and the pupil rays, traced by the vector equation, spread with it: no aberration is reported at this plane
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in {i.code.value for i in result.issues}


def test_a_round_beam_in_a_conical_mount_rotates_onto_the_maps_axes():
    """A conical pass of a round beam stays valid, with the waists the map's determinant gives."""

    phi, psi = math.radians(10.0), math.radians(30.0)
    d = (math.sqrt(1.0 - math.sin(phi) ** 2 - math.sin(psi) ** 2), math.sin(phi), math.sin(psi))
    normal, k = (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    pose = _quat(k, _cross(normal, k), normal)
    e1 = tuple(c / math.hypot(d[0], d[1]) for c in (-d[1], d[0], 0.0))
    comp = laser(633.0, position=_vec(d, -100.0))
    comp["nominal_pose"]["quaternion_wxyz"] = list(_quat(e1, _cross(d, e1), d))
    a = run([comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [], [grating(orders=(1,))]).result
    direction = orders_table(a)[1].direction
    outward = plane("NEAR", _vec(direction, 50.0), _quat(_perp(direction), _cross(direction, _perp(direction)), direction), radius_mm=100.0)
    b = run([comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [outward], [grating(orders=(1,))]).result
    (arrival,) = b.observations[0].arrivals
    assert arrival.beam.gaussian.valid
    # the product of the waists scales by cos(exit) / cos(incidence), whatever the conical angle (the map's determinant)
    assert sorted((arrival.beam.gaussian.waist_radius_x_mm, arrival.beam.gaussian.waist_radius_y_mm))[0] * sorted(
        (arrival.beam.gaussian.waist_radius_x_mm, arrival.beam.gaussian.waist_radius_y_mm))[1] == pytest.approx(0.5 * 0.5 * abs(direction[0]) / d[0], rel=1e-9)


def _perp(d):
    u = _cross(d, (0.0, 0.0, 1.0))
    norm = math.sqrt(sum(c * c for c in u))
    return tuple(c / norm for c in u)


# ---------------------------------------------------------------------------------------- power without a response


def _transmission_with_screen(*, absorbing: bool = False, **grating_fields):
    observation = plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)
    observation["absorbing"] = absorbing
    return run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [observation],
               [grating(mode="TRANSMISSION", density="300", **grating_fields)])


def test_missing_efficiency_books_the_parent_once_and_never_reads_zero_or_an_equal_split():
    result = _transmission_with_screen().result
    (line,) = result.power_budget.lines
    assert line.sinks == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=REL)}, "booked once, not once per order"
    assert result.power_budget.conserved
    assert line.observed_W == {}, "the plane observes nothing known"
    arrivals = by_order(result)
    assert set(arrivals) == {-1, 0, 1}
    for arrival in arrivals.values():
        assert arrival.beam.power_W is None, "not 0 W, not P / 3"
        assert arrival.detected_power_W is None
        assert arrival.power_upper_bound_W == pytest.approx(POWER_W, rel=REL)
        assert arrival.beam.polarization is None and arrival.polarization_state is None and arrival.beam.phase_rad is None
        assert arrival.ray_opl_defined is False
    (obs,) = result.observations
    assert obs.total_power_W is None and obs.power_complete is False and obs.status.value == "UNKNOWN"
    assert obs.quantity_state.value == "UNKNOWN"
    assert all(s.power_W is None and s.complete is False for s in obs.power_by_wavelength)
    assert result.status.value == "UNKNOWN"
    (issue,) = [i for i in result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"]
    assert issue.numerical_context["reason"] == "NO_RESPONSE"


def test_paths_of_unknown_power_read_null_not_zero():
    result = _transmission_with_screen().result
    children = [p for p in result.paths if p.parent_path_id is not None]
    assert len(children) == 3
    for path in children:
        assert path.final_power_W is None and path.branch_kind.startswith("grating_order_")
        assert all(s.power_start_W is None and s.power_end_W is None for s in path.segments)
    (parent,) = [p for p in result.paths if p.parent_path_id is None]
    assert parent.termination == "GRATING_ORDERS"


def test_an_absorbing_plane_reads_the_bound_and_a_transparent_one_has_none():
    absorbing = _transmission_with_screen(absorbing=True).result
    (obs,) = absorbing.observations
    assert obs.total_power_W is None and obs.power_upper_bound_W == pytest.approx(POWER_W, rel=REL)
    assert obs.quantity_state.value == "BOUNDED"
    assert all(a.beam.power_W is None and a.detected_power_W is None for a in obs.arrivals)
    transparent = _transmission_with_screen(absorbing=False).result
    (obs,) = transparent.observations
    assert obs.power_upper_bound_W is None and "may cross it more than once" in obs.message


def _power_target(observation: str = "SCREEN", **fields) -> dict:
    return {"target_id": "T", "observation_id": observation, "metric": "power_W", "unit": "W", **fields}


@pytest.mark.parametrize("fields, status", [({"minimum": 1e-6}, "UNKNOWN"), ({"maximum": 0.1}, "PASS"), ({"minimum": 1.0}, "FAIL")])
def test_power_targets_on_an_absorbing_plane_grade_on_the_interval(fields, status):
    observation = plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)
    observation["absorbing"] = True
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [observation],
                 [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], targets=[_power_target(**fields)]).result
    (target,) = result.targets
    assert target.status.value == status
    assert target.value is None, "the plane's power has no value"


def test_a_power_target_on_a_transparent_plane_stays_unknown_and_a_position_target_is_decided():
    run_ = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)],
               [grating(mode="TRANSMISSION", density="300", orders=(1,))],
               targets=[_power_target(minimum=1e-9), {"target_id": "POS", "observation_id": "SCREEN", "metric": "position_u_mm", "unit": "mm", "target_value": 20.0, "tolerance_abs": 0.1}])
    by_id = {t.target_id: t for t in run_.result.targets}
    assert by_id["T"].status.value == "UNKNOWN" and by_id["T"].value is None
    assert by_id["POS"].status.value == "PASS"
    assert by_id["POS"].value == pytest.approx(120.0 * math.tan(math.radians(theta_m(0.0, 1, 550.0, 300.0))), abs=MM)


def test_an_order_chosen_by_path_is_bounded_by_its_own_upper_bound():
    first = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)],
                [grating(mode="TRANSMISSION", density="300", orders=(0, 1))]).result
    path = orders_table(first)[1].order_path_id
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)],
                 [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], targets=[_power_target(maximum=0.5, parameters=[{"key": "path_id", "value": path}])]).result
    (target,) = result.targets
    assert target.status.value == "PASS" and target.value is None


def test_geometry_only_light_is_followed_through_a_splitter_and_a_mirror():
    """Behind a grating without a response, a 50/50 plate splitter's two branches and a fold mirror's reflection all reach their
    planes: an unpowered branch is followed unconditionally, and the probes' power floor does not stop it."""

    splitter = json_api.build_native_model("splitter.plate", {"diameter_mm": "25.4", "reflectance": "0.5", "valid_band_nm": ["400", "1000"], "model_id": "bs"})
    assert splitter["ok"], splitter
    # the splitter's face normal faces the beam at 45 deg: -X + Y over sqrt 2
    s = math.sqrt(0.5)
    normal, x_axis = (-s, s, 0.0), (s, s, 0.0)
    pose = _quat(x_axis, _cross(normal, x_axis), normal)
    document = with_models(project(
        [laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), declared_component("BS", "splitter", "bs", (40.0, 0.0, 0.0), pose)],
        [plane("THROUGH", (120.0, 0.0, 0.0), ALONG_X, radius_mm=100.0), plane("FOLDED", (40.0, 80.0, 0.0), _quat((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), radius_mm=100.0)]),
        [grating(mode="TRANSMISSION", density="300", orders=(0,)), splitter["optical_model"]], coatings=splitter["model_coatings"], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    arrivals = {o.observation_id: o.arrivals for o in result.observations}
    assert len(arrivals["THROUGH"]) == 1 and len(arrivals["FOLDED"]) == 1
    assert arrivals["THROUGH"][0].beam.power_W is None and arrivals["FOLDED"][0].beam.power_W is None
    assert arrivals["FOLDED"][0].power_upper_bound_W == pytest.approx(POWER_W, rel=REL)
    assert result.power_budget.lines[0].sinks == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=REL)}


def test_two_gratings_in_series_book_the_power_once():
    result = run([laser(550.0), declared_component("G1", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), declared_component("G2", "other_optical", "g", (40.0, 0.0, 0.0), ALONG_X)],
                 [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [grating(mode="TRANSMISSION", density="100", orders=(0, 1))]).result
    assert result.power_budget.lines[0].sinks == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=REL)}
    (obs,) = result.observations
    assert len(obs.arrivals) == 4
    positions = sorted(a.uv_mm[0] for a in obs.arrivals)
    s1 = 550.0e-6 * 100.0
    t = math.asin(s1)
    t2 = math.asin(2.0 * s1)
    # (m1, m2): both 0; (0, 1): turned at the second grating; (1, 0): turned at the first; (1, 1): twice
    expected = sorted((0.0, 80.0 * math.tan(t), 120.0 * math.tan(t), 40.0 * math.tan(t) + 80.0 * math.tan(t2)))
    assert positions == pytest.approx(expected, abs=MM)
    assert all(len(a.grating_orders) == 2 and a.power_upper_bound_W == pytest.approx(POWER_W, rel=REL) for a in obs.arrivals)


def test_the_back_of_a_reflection_grating_blocks_and_a_clipped_order_is_a_warning():
    # the beam along +X meets a reflection grating facing +X (its back)
    pose = _quat((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], [], [grating()]).result
    assert "BLOCKED_BY_BACK_SIDE" in {i.code.value for i in result.issues}
    assert result.power_budget.lines[0].sinks == {"BLOCKED_BACK_SIDE": pytest.approx(POWER_W, rel=REL)}
    # a stop's rim in front of the +1 order: it is clipped with no power to book, and the run does not fail
    stop = {"model_id": "stop", "model": {"model_kind": "APERTURE_STOP", "model_version": "1", "opening_radius_mm": "3", "edge_radius_mm": "60", "status": "SPECIFIED"}}
    clipped = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), declared_component("S", "other_optical", "stop", (60.0, 0.0, 0.0), ALONG_X)],
                  [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)], [grating(mode="TRANSMISSION", density="300", orders=(0, 1)), stop]).result
    assert [i.status.value for i in clipped.issues if i.code.value == "CLIPPED"] == ["WARNING"]
    assert set(by_order(clipped)) == {0}


def test_a_network_through_a_grating_is_refused_and_phase_targets_are_undefined():
    observation = plane("DET", (50.0, 0.0, 0.0), ALONG_X)
    observation.update(absorbing=True, medium="air")
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [observation]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0,))], design_intent="OPTICAL_SYSTEM")
    answered = json_api.linear_optical_network(document)
    assert answered["ok"] is False
    assert {p["code"] for p in answered["error"]["problems"]} == {"PHASE_UNDEFINED_GRATING"}
    result = run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("DET", (50.0, 0.0, 0.0), ALONG_X)],
                 [grating(mode="TRANSMISSION", density="300", orders=(0,))], targets=[{"target_id": "P", "observation_id": "DET", "metric": "phase_rad", "unit": "rad", "minimum": -10.0}]).result
    (target,) = result.targets
    assert target.value is None and target.status.value == "UNKNOWN" and "PHASE_UNDEFINED" in target.issue_codes


# ------------------------------------------------------------------------------------------------- the response table


def table(rows, *, normalization="ABSOLUTE", basis="TE_TM", interpolation="LINEAR", kind="ASSUMED", absorption=(), **extra) -> dict:
    return {"normalization": normalization, "polarization_basis": basis, "interpolation": interpolation, "rows": rows, "absorption": list(absorption),
            "data_kind": kind, "basis": "fixture values for this test", **extra}


def te_tm(w: str, a: str, m: int, te: str, tm: str) -> dict:
    return {"wavelength_nm": w, "aoi_deg": a, "order": m, "efficiency_te": te, "efficiency_tm": tm}


#: Fixture response at normal incidence: every value is this test's choice, not a grating's.
NORMAL_ROWS = [te_tm("500", "0", -1, "0.1", "0.2"), te_tm("500", "0", 0, "0.3", "0.1"), te_tm("500", "0", 1, "0.4", "0.5"),
               te_tm("600", "0", -1, "0.2", "0.3"), te_tm("600", "0", 0, "0.1", "0.2"), te_tm("600", "0", 1, "0.5", "0.4")]


def _table_run(response, wavelength_nm=550.0, polarization_deg=90.0, *, orders=(-1, 0, 1), observations=None, targets=None, components=None):
    """Transmission grating along +X; at ALONG_X local +Y is design +Z and local +X design +Y: the grooves run along local +Y, so the
    source's local-X polarization (``polarization_deg`` 0: along design +Y) is TM, 90 deg (along design +Z) is TE."""

    observations = observations if observations is not None else [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]
    comps = components or [laser(wavelength_nm, polarization_deg=polarization_deg), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)]
    return run(comps, observations, [grating(mode="TRANSMISSION", density="300", orders=orders, response=response)], targets=targets).result


def test_order_powers_follow_the_table_with_linear_interpolation_te():
    """At 550 nm, halfway in the table: TE efficiencies 0.15, 0.2, 0.45; the rest (0.2) is not represented."""

    result = _table_run(table(NORMAL_ROWS))
    arrivals = by_order(result)
    for m, expected in {-1: 0.15, 0: 0.2, 1: 0.45}.items():
        assert arrivals[m].beam.power_W == pytest.approx(POWER_W * expected, rel=REL), m
        assert orders_table(result)[m].efficiency == pytest.approx(expected, rel=REL)
        assert orders_table(result)[m].power_state.value == "KNOWN"
        assert orders_table(result)[m].response_basis == "ASSUMED"
        assert orders_table(result)[m].phase_state.value == "UNKNOWN"
        assert orders_table(result)[m].polarization_state.value == "KNOWN"
    (line,) = result.power_budget.lines
    # the ledger closes: what the orders carry leaves them at their ends, the rest is the element's unmodelled response
    assert line.sinks["UNMODELLED_RESPONSE"] == pytest.approx(POWER_W * 0.2, rel=1e-9)
    assert line.sinks["ESCAPED"] == pytest.approx(POWER_W * 0.8, rel=1e-9)
    assert result.power_budget.conserved
    assert line.observed_W["SCREEN"] == pytest.approx(POWER_W * 0.8, rel=1e-9)


def test_order_powers_follow_the_table_tm_and_an_absorbing_screen():
    observation = plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)
    observation["absorbing"] = True
    result = _table_run(table(NORMAL_ROWS), polarization_deg=0.0, observations=[observation])
    arrivals = by_order(result)
    for m, expected in {-1: 0.25, 0: 0.15, 1: 0.45}.items():
        assert arrivals[m].beam.power_W == pytest.approx(POWER_W * expected, rel=REL)
        assert arrivals[m].detected_power_W == pytest.approx(POWER_W * expected, rel=REL)
    (line,) = result.power_budget.lines
    assert line.sinks["DETECTED"] == pytest.approx(POWER_W * 0.85, rel=1e-9)
    assert line.sinks["UNMODELLED_RESPONSE"] == pytest.approx(POWER_W * 0.15, rel=1e-9)
    (obs,) = result.observations
    assert obs.total_power_W == pytest.approx(POWER_W * 0.85, rel=1e-9)
    # the rest may reach it: the plane's power is bounded by the part the response does not give
    assert obs.power_upper_bound_W == pytest.approx(POWER_W, rel=1e-9)
    assert obs.quantity_state.value == "BOUNDED"


def test_a_table_that_accounts_for_every_watt_leaves_nothing_unknown():
    rows = [te_tm("550", "0", 0, "0.2", "0.2"), te_tm("550", "0", 1, "0.7", "0.6")]
    absorption = [{"wavelength_nm": "550", "aoi_deg": "0", "absorption_te": "0.1", "absorption_tm": "0.2"}]
    observation = plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)
    observation["absorbing"] = True
    result = _table_run(table(rows, absorption=absorption), orders=(0, 1), observations=[observation])
    (line,) = result.power_budget.lines
    assert line.sinks == {"DETECTED": pytest.approx(POWER_W * 0.9, rel=1e-9), "ELEMENT_TRANSMISSION_LOSS": pytest.approx(POWER_W * 0.1, rel=1e-9)}
    assert result.status.value == "PASS"
    (obs,) = result.observations
    assert obs.power_complete is True and obs.quantity_state.value == "KNOWN"
    assert obs.total_power_W == pytest.approx(POWER_W * 0.9, rel=1e-9)


def test_a_table_order_not_followed_goes_to_the_unmodelled_response_and_an_unrepresented_followed_order_is_bounded():
    rows = [te_tm("550", "0", 0, "0.2", "0.2"), te_tm("550", "0", 1, "0.5", "0.5")]
    result = _table_run(table(rows), orders=(-1, 1))  # 0 listed, not followed; -1 followed, not listed
    arrivals = by_order(result)
    assert arrivals[1].beam.power_W == pytest.approx(POWER_W * 0.5, rel=REL)
    assert arrivals[-1].beam.power_W is None
    assert arrivals[-1].power_upper_bound_W == pytest.approx(POWER_W * 0.3, rel=1e-9), "what the represented orders leave"
    (line,) = result.power_budget.lines
    assert line.sinks["UNMODELLED_RESPONSE"] == pytest.approx(POWER_W * 0.5, rel=1e-9)
    rows_out = orders_table(result)
    assert rows_out[0].followed is False and rows_out[0].power_W == pytest.approx(POWER_W * 0.2, rel=REL), "known, not followed"


def test_mixed_polarization_adds_te_and_tm_powers_and_leaves_the_polarization_unknown():
    """At 45 deg half the power is TE, half TM: an order carries the mean of its efficiencies, whatever their relative phase."""

    result = _table_run(table(NORMAL_ROWS), polarization_deg=45.0)
    arrivals = by_order(result)
    for m, expected in {-1: 0.2, 0: 0.175, 1: 0.45}.items():
        assert arrivals[m].beam.power_W == pytest.approx(POWER_W * expected, rel=1e-9)
        assert arrivals[m].beam.polarization is None and arrivals[m].polarization_state is None
        assert orders_table(result)[m].polarization_state.value == "UNKNOWN"


def test_a_mixed_order_loses_its_power_at_a_polarizer_and_keeps_it_at_a_thin_lens():
    pol = {"model_id": "pol", "model": {
        "model_kind": "POLARIZATION_ELEMENT", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
        "transmittance": {"value": "0.9", "status": "SPECIFIED"}, "extinction_ratio": {"value": "10000", "status": "SPECIFIED"},
        "clear_radius_mm": "40", "edge_radius_mm": "45", "valid_band_nm": {"min_nm": "400", "max_nm": "800", "status": "SPECIFIED"},
        "valid_aoi_deg": {"min_deg": "0", "max_deg": "30", "status": "SPECIFIED"}, "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}}
    rows = [te_tm("550", "0", 0, "0.6", "0.2")]
    document = with_models(project([laser(550.0, polarization_deg=45.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X),
                                    declared_component("P", "polarizer", "pol", (40.0, 0.0, 0.0), ALONG_X)],
                                   [plane("BEFORE", (20.0, 0.0, 0.0), ALONG_X), plane("AFTER", (80.0, 0.0, 0.0), ALONG_X)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0,), response=table(rows)), pol], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    arrivals = {o.observation_id: o.arrivals[0] for o in result.observations}
    assert arrivals["BEFORE"].beam.power_W == pytest.approx(POWER_W * 0.4, rel=1e-9)
    assert arrivals["AFTER"].beam.power_W is None, "the polarizer's share depends on the relative phase the table does not state"
    assert arrivals["AFTER"].power_upper_bound_W == pytest.approx(POWER_W * 0.4, rel=1e-9)
    assert any(i.code.value == "GRATING_RESPONSE_UNAVAILABLE" and "share" in i.message for i in result.issues)
    assert result.power_budget.conserved


def test_a_response_that_does_not_depend_on_the_polarization_keeps_the_state_that_arrives():
    """TE = TM = 0.7 at 45 deg polarization: the order keeps the arriving state, and an ideal polarizer along the local x axis of the source passes half of the 0.7:
    0.35 of the power, times its transmittance 0.9."""

    pol = {"model_id": "pol", "model": {
        "model_kind": "POLARIZATION_ELEMENT", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
        "transmittance": {"value": "0.9", "status": "SPECIFIED"}, "extinction_ratio": {"value": "1000000", "status": "SPECIFIED"},
        "clear_radius_mm": "40", "edge_radius_mm": "45", "valid_band_nm": {"min_nm": "400", "max_nm": "800", "status": "SPECIFIED"},
        "valid_aoi_deg": {"min_deg": "0", "max_deg": "30", "status": "SPECIFIED"}, "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}}
    rows = [te_tm("550", "0", 0, "0.7", "0.7"), te_tm("550", "0", 1, "0.1", "0.1")]
    document = with_models(project([laser(550.0, polarization_deg=45.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X),
                                    declared_component("P", "polarizer", "pol", (40.0, 0.0, 0.0), ALONG_X)],
                                   [plane("AFTER", (80.0, 0.0, 0.0), ALONG_X)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 1), response=table(rows)), pol], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    assert orders_table(result)[0].polarization_state.value == "KNOWN"
    (after, *_) = [a for a in result.observations[0].arrivals if abs(a.uv_mm[0]) + abs(a.uv_mm[1]) < 1e-6]
    assert after.beam.polarization is not None
    assert after.beam.power_W == pytest.approx(POWER_W * 0.7 * 0.5 * 0.9, rel=1e-5)
    assert "GRATING_RESPONSE_UNAVAILABLE" not in {i.code.value for i in result.issues}


def test_one_order_that_depends_on_the_polarization_opens_every_order_of_the_grating():
    rows = [te_tm("550", "0", 0, "0.7", "0.7"), te_tm("550", "0", 1, "0.1", "0.2")]
    result = _table_run(table(rows), polarization_deg=45.0, orders=(0, 1))
    assert {m: orders_table(result)[m].polarization_state.value for m in (0, 1)} == {0: "UNKNOWN", 1: "UNKNOWN"}


def test_rays_through_a_polarization_independent_response_keep_their_power_at_an_oblique_lens():
    """An emitter, a grating of TE = TM = 0.7 and a glass lens the rays meet off normal: the power on the plane is 0.7 of the same chain without the grating, within the
    rays the wider grating changes nothing for (tolerance 1e-9, from the same sequence), and nothing is left to the unmodelled response beyond the 0.3 the table gives away."""

    rows = [te_tm("550", a, 0, "0.7", "0.7") for a in ("-89.99", "89.99")]
    response = table(rows, max_conical_angle_deg="89.99")
    lens = {"model_id": "L", "model": sequence([surface(1, "0", radius="80", clear="19"), surface(2, "5", clear="19")], ["N-BK7"], edge="20")}
    screen = {"observation_id": "SCREEN", "pose": {"position_mm": [40.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}, "absorbing": True}
    lens_part = declared_component("LENS", "lens", "L", (10.0, 0.0, 0.0), ALONG_X)
    bare = run([_emitter(), lens_part], [screen], [lens], materials=[nbk7()]).result
    both = run([_emitter(), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), lens_part], [screen],
               [grating(mode="TRANSMISSION", density="300", orders=(0,), response=response, **WIDE), lens], materials=[nbk7()]).result
    (light,), (reference,) = both.observations[0].sampled_light, bare.observations[0].sampled_light
    assert reference.power_W > 0.0 and light.power_W == pytest.approx(0.7 * reference.power_W, rel=1e-9)
    (line,) = both.power_budget.lines
    assert line.sinks["UNMODELLED_RESPONSE"] == pytest.approx(0.3 * line.input_power_W, rel=1e-9)
    assert both.power_budget.conserved


def _two_mixed_orders(targets):
    """Order 0 and order 1 of a 45 deg beam on one plane: the table states their powers and, with TE != TM, not their polarization."""

    rows = [te_tm("550", "0", 0, "0.6", "0.2"), te_tm("550", "0", 1, "0.4", "0.8")]  # (they sum to 1: the ledger is complete, and the plane's power is known)
    return _table_run(table(rows), polarization_deg=45.0, orders=(0, 1), targets=targets)


@pytest.mark.parametrize("metric, unit", [("fringe_visibility", None), ("peak_irradiance_W_per_mm2", "W/mm^2")])
def test_a_pair_of_beams_of_unknown_polarization_is_not_known_rather_than_absent(metric, unit):
    """Two beams reach the plane: "no pair of beams reaches it" would be false, and the coherent power of the same plane already reads UNKNOWN."""

    targets = [{"target_id": "pair", "observation_id": "SCREEN", "metric": metric, "minimum": 0.0, "unit": unit},
               {"target_id": "sum", "observation_id": "SCREEN", "metric": "coherent_power_W", "minimum": 0.0, "unit": "W"}]
    result = _two_mixed_orders(targets)
    assert len(result.observations[0].arrivals) == 2
    found = {t.target_id: t for t in result.targets}
    assert found["sum"].status.value == "UNKNOWN"
    assert found["pair"].status.value == "UNKNOWN" and found["pair"].value is None
    assert found["pair"].issue_codes == ("PHASE_UNDEFINED",) and "polarization" in found["pair"].message
    assert "OBSERVATION_NOT_REACHED" not in found["pair"].issue_codes
    assert result.status.value != "FAIL"


def test_a_pair_named_by_its_paths_is_read_when_the_other_beams_are_of_unknown_polarization():
    """A third beam of unknown polarization does not take the answer from a pair named by its two paths; and with no selector it does: it may be the pair asked for."""

    first = laser(550.0, "A", position=(-100.0, 0.0, 1.0))
    second = laser(550.0, "B", position=(50.0, 0.0, -1.0))
    other = laser(550.0, "C", position=(-100.0, 0.0, 0.0), polarization_deg=45.0)
    # (A and C pass the grating; A's pure polarization stays known, C's mixed state is open behind a table with TE != TM)
    rows = [te_tm("550", "0", 0, "0.6", "0.2")]
    pair = {"target_id": "pair", "observation_id": "SCREEN", "metric": "peak_irradiance_W_per_mm2", "minimum": 0.0, "unit": "W/mm^2"}
    named = {**pair, "target_id": "named", "parameters": [{"key": "path_id_a", "value": "A#0.0"}, {"key": "path_id_b", "value": "B#0"}]}
    result = _table_run(table(rows), orders=(0,), targets=[pair, named], components=[first, second, other, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)])
    found = {t.target_id: t for t in result.targets}
    assert found["pair"].status.value == "UNKNOWN" and found["pair"].issue_codes == ("PHASE_UNDEFINED",)
    assert "PHASE_UNDEFINED" not in found["named"].issue_codes and "INCOHERENT_SOURCES" in found["named"].issue_codes  # (its own answer: two lasers do not interfere)


def test_a_dark_plane_still_fails_a_pair_target():
    """The grating sends both orders away from a small plane: no beam reaches it, and the ledger is complete."""

    rows = [te_tm("550", "0", 1, "1", "1")]
    target = {"target_id": "pair", "observation_id": "SCREEN", "metric": "fringe_visibility", "minimum": 0.0}
    result = _table_run(table(rows), polarization_deg=45.0, orders=(1,), targets=[target], observations=[plane("SCREEN", (100.0, 0.0, 0.0), ALONG_X, radius_mm=1.0)])
    (found,) = result.targets
    assert found.status.value == "FAIL" and "OBSERVATION_NOT_REACHED" in found.issue_codes


def test_a_beam_of_unknown_power_outside_the_acceptance_cone_reports_no_zero_incident_power():
    """Order 5 misses the 1 deg cone: the plane detects nothing of it, and what it brings is not known, at most the incident power."""

    screen = {**plane("SCREEN", (100.0, 0.0, 0.0), ALONG_X, radius_mm=300.0), "absorbing": True, "acceptance_half_angle_deg": 1.0}
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [screen]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 5))], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    (missed, *_) = [i for i in result.issues if i.code.value == "MONITOR_MISSED"]
    assert "incident_power_W" not in missed.numerical_context
    assert missed.numerical_context["power_upper_bound_W"] == pytest.approx(POWER_W)


def test_a_beam_of_known_power_outside_the_acceptance_cone_reports_its_power():
    rows = [te_tm("550", "0", 0, "0.5", "0.5"), te_tm("550", "0", 5, "0.5", "0.5")]
    screen = {**plane("SCREEN", (100.0, 0.0, 0.0), ALONG_X, radius_mm=300.0), "absorbing": True, "acceptance_half_angle_deg": 1.0}
    result = _table_run(table(rows), orders=(0, 5), observations=[screen])
    (missed, *_) = [i for i in result.issues if i.code.value == "MONITOR_MISSED"]
    assert missed.numerical_context["incident_power_W"] == pytest.approx(POWER_W * 0.5, rel=1e-9)
    assert "power_upper_bound_W" not in missed.numerical_context


def test_a_beam_of_unknown_power_beside_the_active_area_is_not_a_zero():
    screen = plane("SCREEN", (100.0, 0.0, 0.0), ALONG_X, radius_mm=1.0)
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [screen]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    (outside,) = result.observations[0].outside_active_area
    assert outside.power_W is None and outside.power_upper_bound_W == pytest.approx(POWER_W)
    assert outside.model_dump(mode="json")["power_W"] is None


def test_pure_te_order_keeps_its_polarization_along_the_grooves_and_a_polarizer_reads_it():
    rows = [te_tm("550", "0", 1, "0.6", "0.2")]
    result = _table_run(table(rows), orders=(1,))
    arrival = by_order(result)[1]
    stokes = arrival.polarization_state
    # TE: the field along the grooves (design Z at this pose): the major axis is the groove axis
    assert stokes.handedness == "LINEAR"
    assert abs(stokes.major_axis[2]) == pytest.approx(1.0, abs=1e-12)


def test_tm_order_field_lies_in_the_plane_of_incidence_across_the_leaving_direction():
    rows = [te_tm("550", "0", 1, "0.6", "0.2")]
    result = _table_run(table(rows), orders=(1,), polarization_deg=0.0)
    arrival = by_order(result)[1]
    axis = arrival.polarization_state.major_axis
    direction = arrival.beam.direction
    assert abs(sum(a * b for a, b in zip(axis, direction))) == pytest.approx(0.0, abs=1e-12)
    assert axis[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("wavelength_nm", [499.0, 601.0])
def test_outside_the_table_the_response_is_unavailable_and_geometry_goes_on(wavelength_nm):
    result = _table_run(table(NORMAL_ROWS), wavelength_nm=wavelength_nm)
    arrivals = by_order(result)
    assert set(arrivals) == {-1, 0, 1}
    assert all(a.beam.power_W is None and a.power_upper_bound_W == pytest.approx(POWER_W, rel=REL) for a in arrivals.values())
    (issue,) = [i for i in result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"]
    assert issue.numerical_context["reason"] == "OUT_OF_DOMAIN"
    assert arrivals[1].uv_mm[0] == pytest.approx(120.0 * math.tan(math.radians(theta_m(0.0, 1, wavelength_nm, 300.0))), abs=MM)
    assert result.power_budget.lines[0].sinks == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=REL)}


def test_points_only_answers_the_grid_exactly():
    response = table(NORMAL_ROWS, interpolation="POINTS_ONLY")
    assert by_order(_table_run(response, wavelength_nm=500.0))[0].beam.power_W == pytest.approx(POWER_W * 0.3, rel=REL)
    assert by_order(_table_run(response, wavelength_nm=550.0))[0].beam.power_W is None


def test_a_table_over_angles_interpolates_in_the_signed_incidence():
    """Reflection at +10 deg in a table at 0 and +20 deg: the mean. At -10 deg the same grating is out of the domain."""

    rows = [te_tm("550", a, 0, te, "0") for a, te in (("0", "0.2"), ("20", "0.6"))]
    pose, _k, _normal = reflection_pose(10.0)
    comps = [laser(550.0, polarization_deg=90.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)]
    result = run(comps, [], [grating(orders=(0,), response=table(rows))]).result
    row = orders_table(result)[0]
    assert row.incidence_deg == pytest.approx(10.0, abs=DEG)
    assert row.power_W == pytest.approx(POWER_W * 0.4, rel=1e-9)
    pose, _k, _normal = reflection_pose(-10.0)
    comps = [laser(550.0, polarization_deg=90.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)]
    result = run(comps, [], [grating(orders=(0,), response=table(rows))]).result
    row = orders_table(result)[0]
    assert row.power_W is None and row.power_state.value == "BOUNDED"


def test_a_relative_table_gives_no_absolute_power_and_reports_its_values():
    result = _table_run(table(NORMAL_ROWS, normalization="RELATIVE"))
    rows = orders_table(result)
    assert rows[1].power_W is None and rows[1].relative_response == pytest.approx(0.45, rel=REL)
    assert rows[1].power_upper_bound_W == pytest.approx(POWER_W, rel=REL)
    assert by_order(result)[1].beam.power_W is None
    assert [i.numerical_context["reason"] for i in result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"] == ["RELATIVE"]


def test_an_unpolarized_table_bounds_a_polarized_beam_without_giving_its_power():
    rows = [{"wavelength_nm": "550", "aoi_deg": "0", "order": m, "efficiency": v} for m, v in ((0, "0.2"), (1, "0.7"))]
    result = _table_run(table(rows, basis="UNPOLARIZED"), orders=(0, 1))
    arrivals = by_order(result)
    assert arrivals[0].beam.power_W is None and arrivals[0].power_upper_bound_W == pytest.approx(POWER_W * 0.4, rel=REL), "twice the mean"
    assert arrivals[1].beam.power_W is None and arrivals[1].power_upper_bound_W == pytest.approx(POWER_W, rel=REL), "at most the incident power"
    assert orders_table(result)[0].efficiency == pytest.approx(0.2, rel=REL)
    assert [i.numerical_context["reason"] for i in result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"] == ["UNPOLARIZED"]


def test_a_conical_mount_is_outside_a_planar_table_and_inside_one_that_states_its_angle():
    psi = math.radians(5.0)
    d = (math.cos(psi), 0.0, math.sin(psi))
    e1 = (0.0, 1.0, 0.0)
    comp = laser(550.0, position=_vec(d, -100.0), polarization_deg=0.0)
    comp["nominal_pose"]["quaternion_wxyz"] = list(_quat(e1, _cross(d, e1), d))
    rows = [te_tm("550", "0", 0, "0.5", "0.5")]
    outside = _table_run(table(rows), orders=(0,), components=[comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], observations=[])
    assert orders_table(outside)[0].power_W is None
    assert [i.numerical_context["reason"] for i in outside.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"] == ["CONICAL"]
    inside = _table_run(table(rows, max_conical_angle_deg="6"), orders=(0,), components=[comp, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], observations=[])
    assert orders_table(inside)[0].power_W == pytest.approx(POWER_W * 0.5, rel=REL)


def test_a_transmission_table_holds_for_light_along_plus_z_only():
    against = _quat((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))  # local +Z along -X
    result = _table_run(table(NORMAL_ROWS), components=[laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), against)], observations=[])
    assert orders_table(result)[1].power_W is None
    assert [i.numerical_context["reason"] for i in result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"] == ["OTHER_SIDE"]


def test_provenance_is_kept_with_the_run_and_assumed_data_is_told_apart():
    measured = table(NORMAL_ROWS, kind="MEASURED", evidence=dict(EVIDENCE))
    computed = table(NORMAL_ROWS, kind="INDEPENDENTLY_COMPUTED", evidence=dict(EVIDENCE))
    assumed = table(NORMAL_ROWS)
    for response, basis in ((measured, "MEASURED"), (computed, "INDEPENDENTLY_COMPUTED"), (assumed, "ASSUMED")):
        document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], []),
                               [grating(mode="TRANSMISSION", density="300", response=response)], design_intent="OPTICAL_SYSTEM")
        built = run_optical_project(document, catalog=None)
        assert {row.response_basis for row in built.result.grating_orders} == {basis}
        (element,) = [e for e in built.element_models if e.get("optical_model_id") == "g"]
        stated = element["response"]["order_response"]
        assert stated["data_kind"] == basis and stated["normalization"] == "ABSOLUTE" and stated["polarization_basis"] == "TE_TM"
        assert stated["interpolation"] == "LINEAR" and len(stated["rows"]) == len(NORMAL_ROWS)
        assert (stated["evidence"] is None) == (basis == "ASSUMED")
        assert element["response"]["phase"] == "not stated"
        assert element["fidelity"] == "IDEAL_THIN_ELEMENT"


# ------------------------------------------------------------------------------------------------- contract


def _model(**fields) -> ProjectOpticalModelV1:
    return ProjectOpticalModelV1.model_validate(grating(**fields))


def test_contract_refuses_a_table_that_makes_more_than_the_incident_power():
    with pytest.raises(ValidationError, match="sum to more than 1"):
        _model(response=table([te_tm("550", "0", 0, "0.6", "0.1"), te_tm("550", "0", 1, "0.5", "0.1")]))
    with pytest.raises(ValidationError, match="sum to more than 1"):
        _model(response=table([te_tm("550", "0", 0, "0.6", "0.1")], absorption=[{"wavelength_nm": "550", "aoi_deg": "0", "absorption_te": "0.5", "absorption_tm": "0"}]))
    # a relative table is not a fraction of the incident power: its values may exceed 1 and it has no absorption
    _model(response=table([te_tm("550", "0", 0, "1.6", "1.1")], normalization="RELATIVE"))
    with pytest.raises(ValidationError, match="RELATIVE"):
        _model(response=table([te_tm("550", "0", 0, "0.6", "0.1")], normalization="RELATIVE", absorption=[{"wavelength_nm": "550", "aoi_deg": "0", "absorption_te": "0.1", "absorption_tm": "0"}]))


def test_contract_refuses_an_incomplete_grid_and_missing_provenance():
    with pytest.raises(ValidationError, match="full grid"):
        _model(response=table(NORMAL_ROWS[:-1]))
    with pytest.raises(ValidationError, match="repeat"):
        _model(response=table([te_tm("550", "0", 0, "0.6", "0.1"), te_tm("550", "0", 0, "0.2", "0.1")]))
    with pytest.raises(ValidationError, match="evidence"):
        _model(response=table(NORMAL_ROWS, kind="MEASURED"))
    with pytest.raises(ValidationError, match="evidence"):
        _model(response=table(NORMAL_ROWS, kind="ASSUMED", evidence=dict(EVIDENCE)))
    with pytest.raises(ValidationError, match="TE_TM"):
        _model(response=table([{"wavelength_nm": "550", "aoi_deg": "0", "order": 0, "efficiency": "0.5"}]))


@pytest.mark.parametrize("fields, text", [
    ({"period_nm": "1000", "line_density_per_mm": "600"}, "exactly one"),
    ({"orders": (1, 0)}, "strictly increasing"),
    ({"medium_negative_z": "GLASS"}, "reflection grating"),
    ({"clear_half_widths_mm": ["20", "20"]}, "inside"),
])
def test_contract_refuses_a_grating_that_says_two_things(fields, text):
    document = grating(**{k: v for k, v in fields.items() if k != "line_density_per_mm"})
    document["model"].update({k: v for k, v in fields.items() if k == "line_density_per_mm"})
    with pytest.raises(ValidationError, match=text):
        ProjectOpticalModelV1.model_validate(document)


def test_engine_table_shape_is_checked():
    with pytest.raises(ValidationError):
        GratingOrderTable(normalization="absolute", polarization_basis="te_tm", interpolation="linear", data_kind="assumed", wavelength_nm=(500.0,), aoi_deg=(0.0,),
                          orders=(0, 1), efficiency_te=(0.1,), efficiency_tm=(0.1, 0.2))
    DiffractionGratingInteraction(mode="reflection", period_nm=1000.0, grating_vector_local=(1.0, 0.0, 0.0), orders=(1,))


# ----------------------------------------------------------------------------------------- authoring and artifacts


def test_native_class_discovery_and_build():
    (described,) = [c for c in json_api.native_components()["classes"] if c["class_id"] == "grating.plane"]
    assert described["model_kind"] == "DIFFRACTION_GRATING" and described["category"] == "GRATING" and "grating" in described["keywords"]
    assert {"mode", "line_density_per_mm", "period_nm", "grating_vector_angle_deg", "orders", "all_propagating_orders", "response"} <= {p["name"] for p in described["parameters"]}
    built = json_api.build_native_model("grating.plane", {"mode": "TRANSMISSION", "period_nm": "2000", "all_propagating_orders": [-2, 2], "width_mm": "25", "height_mm": "25", "model_id": "g"})
    assert built["ok"] is True, built
    assert built["optical_model"]["model"]["model_kind"] == "DIFFRACTION_GRATING"
    assert built["design"]["order_powers"].startswith("not known")
    assert json_api.build_native_model("grating.plane", {"mode": "TRANSMISSION", "width_mm": "25", "height_mm": "25", "orders": [0]})["ok"] is False


def test_the_command_line_lists_and_describes_the_grating(capsys):
    from os_optics.cli import main

    assert main(["native", "list", "--category", "GRATING"]) == 0
    assert [c["class_id"] for c in json.loads(capsys.readouterr().out)["classes"]] == ["grating.plane"]
    assert main(["native", "describe", "grating"]) == 0
    assert json.loads(capsys.readouterr().out)["model_kind"] == "DIFFRACTION_GRATING"
    assert main(["schema", "project"]) == 0
    assert "DIFFRACTION_GRATING" in capsys.readouterr().out


def test_project_ops_add_and_the_capability_is_listed():
    assert "diffraction_gratings" in json_api.capabilities()["project_features"]
    assert "DIFFRACTION_GRATING" in PROJECT_OPS["add_optical_model"]["summary"]
    document = project([laser(550.0)], [])
    updated = json_api.update_project(document, [{"op": "add_optical_model", "optical_model": grating(mode="TRANSMISSION", density="300")},
                                                 {"op": "add_component", "component": declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)}])
    assert updated["ok"] is True, updated
    assert updated["project"]["optical_models"][0]["model"]["mode"] == "TRANSMISSION"


def test_saved_artifacts_keep_null_powers_bounds_and_the_order_table(tmp_path):
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(-2, -1, 0, 1, 2))], design_intent="OPTICAL_SYSTEM")
    built = run_optical_project(document, catalog=None, output_dir=tmp_path / "run")
    results = load_run(tmp_path / "run")
    rows = results.grating_orders()
    assert {r["order"] for r in rows} >= {-2, -1, 0, 1, 2}
    assert all(r["power_W"] is None if "power_W" in r else True for r in rows)
    assert results.grating_orders("G") == rows
    monitor = results.monitor("SCREEN")
    assert monitor["total_power_W"] is None
    (row,) = [a for a in results.arrivals("SCREEN") if a["path_id"] == next(r["order_path_id"] for r in rows if r["order"] == 1)]
    assert row["power_W"] in (None, "") and row["position_u_mm" if "position_u_mm" in row else "u_mm"] == pytest.approx(
        120.0 * math.tan(math.radians(theta_m(0.0, 1, 550.0, 300.0))), abs=MM)
    # the stored result reads the same as the run that wrote it
    assert json.loads((tmp_path / "run" / "simulation_result.json").read_text(encoding="utf-8"))["grating_orders"] == [r.model_dump(mode="json") for r in built.result.grating_orders]
    assert results.handoff("SCREEN")["beams"][0]["power_W"] is None


def test_json_api_returns_the_same_meanings():
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], design_intent="OPTICAL_SYSTEM")
    answered = json_api.run_project(document)
    assert answered["ok"] is True, answered
    observations = answered["summary"]["observations"]
    (screen,) = observations
    assert all(a["power_W"] is None for a in screen["arrivals"])
    assert screen["quantity_state"] == "UNKNOWN"
    assert answered["status"] == "UNKNOWN"


# -------------------------------------------------------------------------------------------------- views


def test_the_body_reads_as_a_ruled_plate():
    built = json_api.build_native_model("grating.plane", {"mode": "REFLECTION", "line_density_per_mm": "600", "orders": [1], "width_mm": "25", "height_mm": "20", "model_id": "g"})
    pose, _k, _normal = reflection_pose(10.0)
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), pose)], []), [built["optical_model"]], design_intent="OPTICAL_SYSTEM")
    scene = run_optical_project(document, catalog=None).optical_view
    (element,) = [e for e in scene.elements if e.element_id == "G"]
    geom = element.geometry
    assert geom.model_kind == "DIFFRACTION_GRATING" and geom.surfaces[0].role.value == "GRATING" and geom.shape == "BOX"
    assert geom.grating.mode == "REFLECTION" and geom.grating.period_nm == pytest.approx(1e6 / 600.0)
    assert geom.reference_axis.value == "FACE_NORMAL"
    drawing = optic_mesh.draw_body(geom)
    materials = {name for name, _ in drawing.body}
    assert materials == {"grating_body", "grating_rule"}
    rules = next(mesh for name, mesh in drawing.body if name == "grating_rule")
    xs = [p[0] for p in rules.positions]
    ys = [p[1] for p in rules.positions]
    # the rules run along the grooves (local Y), across the aperture along local X, and stay above the plate
    assert max(ys) - min(ys) == pytest.approx(18.0, abs=1e-9)
    assert 0.0 < min(p[2] for p in rules.positions)
    assert max(xs) <= 11.25 + 1e-9 and min(xs) >= -11.25 - 1e-9
    assert len({round(x, 6) for x in xs}) >= 2 * optic_mesh.GRATING_RULES - 4
    section_entry = section.element_section(geom, "XZ")
    assert section_entry["surfaces"][0]["role"] == "GRATING"
    assert "plate" in section_entry
    # the section across the grooves carries their mark over the clear aperture; the one along them does not
    assert len(section_entry["symbol"]) == section.GRATING_SECTION_RULES
    assert all(-11.25 <= line[0][0] <= 11.25 for line in section_entry["symbol"])
    assert section.element_section(geom, "YZ")["symbol"] == []


def test_a_turned_grating_rules_run_across_its_grating_vector():
    built = json_api.build_native_model("grating.plane", {"mode": "TRANSMISSION", "line_density_per_mm": "600", "orders": [0], "width_mm": "20", "height_mm": "20",
                                                          "grating_vector_angle_deg": "90", "model_id": "g"})
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], []), [built["optical_model"]], design_intent="OPTICAL_SYSTEM")
    scene = run_optical_project(document, catalog=None).optical_view
    (element,) = [e for e in scene.elements if e.element_id == "G"]
    drawing = optic_mesh.draw_body(element.geometry)
    rules = next(mesh for name, mesh in drawing.body if name == "grating_rule")
    # K along local Y: the rules run along local X
    assert max(p[0] for p in rules.positions) - min(p[0] for p in rules.positions) == pytest.approx(18.0, abs=1e-9)


def test_scene_beams_of_unknown_power_are_drawn_with_a_null_power():
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], design_intent="OPTICAL_SYSTEM")
    built = run_optical_project(document, catalog=None)
    scene = built.optical_view
    children = [p for p in scene.beam_paths if p.parent_path_id is not None]
    assert len(children) == 2 and all(p.final_power_W is None for p in children)
    assert {r.value for p in children for r in p.segment_roles} == {"PRIMARY"}
    (screen,) = scene.observation_planes
    assert all(f.power_W is None for f in screen.footprints)
    assert built.binding_problems() == ()


# --------------------------------------------------------------------------------------------- other results of a run


def test_every_result_table_reads_a_run_of_unknown_power(tmp_path):
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(-2, 0, 1))], design_intent="OPTICAL_SYSTEM")
    run_optical_project(document, catalog=None, output_dir=tmp_path / "run")
    results = load_run(tmp_path / "run")
    for kind in ("monitors", "arrivals", "rays", "paths", "segments", "power-budget", "superpositions", "exposures", "grating-orders", "handoff"):
        filters = {"monitor": "SCREEN"} if kind == "handoff" else {}
        assert results.table(kind, **filters) is not None, kind
        assert results.export(kind, "csv", **filters), kind
    rows = {row["order"]: row for row in results.table("grating-orders")[0]}
    assert rows[1]["direction_x"] == pytest.approx(math.cos(math.radians(theta_m(0.0, 1, 550.0, 300.0))), abs=1e-12)
    assert rows[1]["power_W"] is None and rows[1]["power_state"] == "BOUNDED" and rows[1]["response_basis"] == "NONE"
    wavefront = results.wavefront("SCREEN", rows[1]["order_path_id"])
    assert wavefront["reason"] == "RAY_OPL_UNDEFINED"
    (path,) = [p for p in results.paths() if p["path_id"] == rows[1]["order_path_id"]]
    assert path["branch_kind"] == "grating_order_+1" and path["final_power_W"] is None


def test_the_coherent_sum_of_two_known_orders_is_open_over_every_phase():
    """Orders 0 and +1 of one line reach one plane in the same direction after a second grating folds them together: their relative phase
    is not stated: the pair has no phase difference, and the plane's coherent power is the sum on the trace's conventions (no added
    delay at the order passes, the line taken as monochromatic) with its entry -- an estimate that decides nothing."""

    first = grating("first", mode="TRANSMISSION", density="100", orders=(0, 1), response=table([te_tm("550", "0", 0, "0.5", "0.5"), te_tm("550", "0", 1, "0.5", "0.5")]))
    result = run([laser(550.0, polarization_deg=90.0), declared_component("G", "other_optical", "first", (0.0, 0.0, 0.0), ALONG_X)],
                 [plane("SCREEN", (1.0, 0.0, 0.0), ALONG_X, radius_mm=50.0)], [first],
                 targets=[{"target_id": "C", "observation_id": "SCREEN", "metric": "coherent_power_W", "unit": "W", "maximum": 1.0}]).result
    (obs,) = result.observations
    assert [a.beam.power_W for a in obs.arrivals] == pytest.approx([POWER_W * 0.5, POWER_W * 0.5], rel=REL)
    (target,) = result.targets
    assert target.value == pytest.approx(POWER_W, rel=REL) and (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert target.value_basis["value"].kind.value == "ESTIMATE"
    (pair,) = result.interference
    assert pair.phase_difference_rad is None and "PHASE_UNDEFINED" in pair.issue_codes
    assert pair.beam_crossing_angle_deg == pytest.approx(theta_m(0.0, 1, 550.0, 100.0), abs=1e-6), "geometry stays available"


def test_no_pair_is_formed_from_a_beam_of_unknown_field():
    result = _transmission_with_screen(orders=(0, 1)).result
    assert result.interference == () and result.observations[0].superpositions == ()


def test_a_rated_element_behind_unknown_orders_reads_unknown_and_a_rated_grating_reads_its_incident_power():
    rating = {"regime": "CW", "value": "100", "unit": "W/cm2", "wavelength_nm": "550", "status": "DECLARED", "basis": "fixture"}
    rated = grating(mode="TRANSMISSION", density="300", orders=(0,))
    rated["power_ratings"] = [rating]
    window = json_api.build_native_model("plate.window", {"material": {"constant_index": "1.5", "valid_band_nm": ["400", "800"]}, "thickness_mm": "3", "diameter_mm": "25.4", "model_id": "w"})
    assert window["ok"] is True, window
    behind = dict(window["optical_model"], power_ratings=[rating])
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), declared_component("W", "window", "w", (30.0, 0.0, 0.0), ALONG_X)], []),
                           [rated, behind], window["model_materials"], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    surfaces = {s.surface_id: s for s in result.exposures.surfaces}
    assert surfaces["G/grating"].incident_power_W == pytest.approx(POWER_W, rel=REL)
    assert surfaces["W/S1"].hits == () and surfaces["W/S1"].complete is False, "its light's power is not known: not a zero exposure"


def test_a_probe_of_untraced_light_meeting_a_grating_is_of_unknown_fate():
    """A ghost of an uncoated window's face meets a grating without a response: its light may reach any plane."""

    window = json_api.build_native_model("plate.window", {"material": {"constant_index": "1.5", "valid_band_nm": ["400", "800"]}, "thickness_mm": "3", "diameter_mm": "25.4", "model_id": "w"})
    assert window["ok"] is True, window
    document = with_models(project([laser(550.0, position=(40.0, 0.0, 0.0)), declared_component("W", "window", "w", (60.0, 0.0, 0.0), ALONG_X),
                                    declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), _quat((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))],
                                   [plane("DET", (120.0, 0.0, 0.0), ALONG_X, radius_mm=30.0), plane("FAR", (0.0, 500.0, 0.0), _quat((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), radius_mm=5.0)]),
                           [window["optical_model"], grating(orders=(0,))], window["model_materials"], design_intent="OPTICAL_SYSTEM")
    result = run_optical_project(document, catalog=None).result
    obs = {o.observation_id: o for o in result.observations}
    assert obs["FAR"].total_power_W is None and obs["FAR"].power_complete is False
    assert result.grating_orders == (), "a probe's pass is not an order of the traced light"


def test_the_model_row_and_issue_texts_name_what_they_state():
    result = _transmission_with_screen()
    (element,) = [e for e in result.element_models if e.get("optical_model_id") == "g"]
    assert element["response"]["direction_model"].startswith("n_exit t_exit") and element["response"]["order_response"] is None
    (issue,) = [i for i in result.result.issues if i.code.value == "GRATING_RESPONSE_UNAVAILABLE"]
    assert "state the grating's response" in issue.suggested_action
    assert "directions are traced" in issue.message
    for text in (issue.message, issue.suggested_action, result.result.observations[0].message):
        assert "#" not in text and "should" not in text
    # a grating has no model-scope issue: its directions hold wherever it is met
    assert result.model_scope_issues == ()


def test_a_grating_inside_a_compound_unit_turns_with_the_member():
    """Member pose: a quarter turn about the unit's Z takes the grating vector from unit +X to unit +Y; placed along +X, the unit's +Y is design +Z."""

    half = math.sqrt(0.5)
    unit = {"model_id": "unit", "model": {"model_kind": "COMPOUND", "model_version": "1", "members": [
        {"member_id": "G", "model_id": "g", "pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [half, 0.0, 0.0, half]}}],
        "envelope": {"kind": "CYLINDER", "diameter_mm": "40", "z_min_mm": "-1", "z_max_mm": "1", "status": "NOMINAL"}}}
    result = run([laser(550.0), declared_component("U", "other_optical", "unit", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)],
                 [grating(mode="TRANSMISSION", density="300", orders=(1,)), unit]).result
    (obs,) = result.observations
    (arrival,) = obs.arrivals
    assert arrival.uv_mm[0] == pytest.approx(0.0, abs=MM)
    assert arrival.uv_mm[1] == pytest.approx(120.0 * math.tan(math.radians(theta_m(0.0, 1, 550.0, 300.0))), abs=MM)


def test_a_stated_frame_names_ports_only():
    ports = {"origin": "APERTURE_CENTER", "nominal_propagation": "PORTS", "reference_axis": "PORTS", "symmetry": "NONE",
             "ports": [{"port_id": "IN", "role": "INPUT", "point_mm": ["0", "0", "5"], "direction": ["0", "0", "-1"], "status": "NOMINAL"},
                       {"port_id": "OUT", "role": "OUTPUT", "point_mm": ["0", "0", "0"], "direction": ["0", "1", "0"], "status": "DERIVED"}]}
    model = dict(grating(), frame=ports)
    run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), reflection_pose(10.0)[0])], [], [model])
    wrong = dict(grating(), frame={"origin": "FIRST_SURFACE_VERTEX", "symmetry": "NONE"})
    with pytest.raises(Exception, match="follows from its mode"):
        run([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), reflection_pose(10.0)[0])], [], [wrong])


# ----------------------------------------------------------------------------------------------------- extended emitters


def _emitter(power_w: float = POWER_W) -> dict:
    """A small Lambertian disc 1 mm before a grating that holds its whole hemisphere; 64 sampled rays."""

    return {"instance_id": "E", "optical_role": "source", "custom_part": {"custom_part_key": "test-led", "description": "Test emitter."},
            "nominal_pose": {"position_mm": [-1.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)},
            "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 550.0, "power_w": power_w, "emitting_surface": {"shape": "DISC", "diameter_mm": 0.02},
                             "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 64, "seed": 7}}}


WIDE = {"clear_half_widths_mm": ["5000", "5000"], "edge_half_widths_mm": ["5001", "5001"]}


def test_rays_of_an_extended_emitter_book_a_grating_without_a_response_and_add_no_zero():
    """A sampled result is a sum of powers: a ray of unknown power adds none, the plane's power is not known, and no per-ray order table grows."""

    observation = {"observation_id": "SCREEN", "pose": {"position_mm": [1.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}, "absorbing": True}
    result = run([_emitter(), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [observation], [grating(mode="TRANSMISSION", density="300", orders=(0,), **WIDE)]).result
    (line,) = result.power_budget.lines
    assert line.sinks == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=1e-9)} and result.power_budget.conserved
    (obs,) = result.observations
    assert all(not s.counted_crossing_count for s in obs.sampled_light)
    # (an absorbing plane no sampled ray reaches is bounded by what the ledger leaves open and by the rays' resolution)
    assert obs.total_power_W is None and obs.power_complete is False and obs.power_upper_bound_W >= POWER_W
    assert result.grating_orders == ()
    assert "GRATING_RESPONSE_UNAVAILABLE" in {i.code.value for i in result.issues}


def test_rays_of_an_extended_emitter_take_the_orders_a_response_gives():
    """TE = TM efficiencies, so each ray's state carries the same share: the plane's power is the table's 0.7 of the emitted power in order 0
    (which propagates at every angle), and the rest is the element's unmodelled response."""

    rows = [te_tm("550", a, 0, "0.7", "0.7") for a in ("-89.99", "89.99")]
    observation = {"observation_id": "SCREEN", "pose": {"position_mm": [1.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}, "absorbing": True}
    response = table(rows, max_conical_angle_deg="89.99")
    result = run([_emitter(), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [observation],
                 [grating(mode="TRANSMISSION", density="300", orders=(0,), response=response, **WIDE)]).result
    (line,) = result.power_budget.lines
    assert line.sinks["DETECTED"] == pytest.approx(POWER_W * 0.7, rel=1e-9) and result.power_budget.conserved
    (obs,) = result.observations
    (light,) = obs.sampled_light
    assert light.power_W == pytest.approx(POWER_W * 0.7, rel=1e-9)
    assert obs.total_power_W == pytest.approx(POWER_W * 0.7, rel=1e-9)


def test_an_irradiance_map_of_beams_of_unknown_power_says_where_they_are_and_not_that_none_arrives(tmp_path):
    document = with_models(project([laser(550.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)]),
                           [grating(mode="TRANSMISSION", density="300", orders=(0, 1))], design_intent="OPTICAL_SYSTEM")
    run_optical_project(document, catalog=None, output_dir=tmp_path / "run")
    answered = load_run(tmp_path / "run").irradiance_map("SCREEN")
    assert (answered["status"], answered["reason"]) == ("UNAVAILABLE", "BEAM_POWER_UNKNOWN")
    assert "2 beams reach this plane" in answered["message"]
    (axial, diffracted) = sorted(answered["omitted_beams"], key=lambda beam: abs(beam["uv_mm"][0]) + abs(beam["uv_mm"][1]))
    # sin(theta) = m lambda / period, on a plane 120 mm behind the grating
    sine = 550e-6 * 300.0
    assert abs(diffracted["uv_mm"][0]) + abs(diffracted["uv_mm"][1]) == pytest.approx(120.0 * sine / math.sqrt(1.0 - sine * sine), rel=1e-9)
    assert axial["uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert axial["power_upper_bound_W"] == diffracted["power_upper_bound_W"] == pytest.approx(POWER_W)
