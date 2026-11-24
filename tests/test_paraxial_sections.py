"""Paraxial analysis in one section frame for the whole system (cylinders), the meaning of ``solve: "collimate"`` and the
source checks (``os_optics.local.paraxial``).

Every expected number is computed here independently: thick-lens matrices of the cylinder from its prescription with a
Sellmeier N-BK7 index written in the test, Gaussian q propagation through them; the solved layouts are run and the traced
far-field divergences compared with the predictions.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.local import ParaxialError, paraxial_analysis

from declared_helpers import BAND, NBK7_SELLMEIER, declared_component, sequence, surface, with_models
from local_helpers import (
    AGAINST_X,
    ALONG_X,
    PCX_F50,
    PCX_F100,
    catalog_document,
    element,
    free,
    lens_matrix,
    plane,
    project,
    q_through,
    quat_multiply,
    rayleigh_range,
    run,
    source,
    turn_about_z,
    waist_radius,
)

NM = 1e-6
CYL_R, CYL_T = 51.5, 4.0
CYLINDER = {"model_id": "cyl", "model": sequence([surface(1, "0", radius="51.5", cylinder_axis="Y", clear="9"), surface(2, "4", clear="9")],
                                                  ["N-BK7"], edge="12.7")}
#: ALONG_X rolled 90 deg about the optical axis (+X): the cylinder's local X (its curved axis) points along design Z.
ROLL90 = quat_multiply(ALONG_X, turn_about_z(math.sqrt(0.5), math.sqrt(0.5)))
#: ALONG_X rolled 45 deg about the optical axis.
ROLL45 = quat_multiply(ALONG_X, turn_about_z(math.cos(math.pi / 8), math.sin(math.pi / 8)))
NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)


def _nbk7(wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1 + sum(b * lam2 / (lam2 - c) for b, c in zip(NBK7_B, NBK7_C)))


def _refract(radius: float, n1: float, n2: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-(n2 - n1) / radius, 1.0]])


def _cylinder_matrices(wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent thick-lens matrices of the plano-convex N-BK7 cylinder: (curved section, straight section)."""

    n = _nbk7(wavelength_nm)
    return free(CYL_T / n) @ _refract(CYL_R, 1.0, n), free(CYL_T / n)


def _afocal_gap(m1: np.ndarray, m2: np.ndarray) -> float:
    c = [(m2 @ free(d) @ m1)[1, 0] for d in (0.0, 1.0)]
    return -c[0] / (c[1] - c[0])


def _material() -> dict:
    return {"material_id": "N-BK7", "display_name": "N-BK7", "dispersion": dict(NBK7_SELLMEIER), "valid_band_nm": dict(BAND),
            "basis": "Schott N-BK7 Sellmeier formula"}


def _crossed(x2: float, *, waist_mm: float = 0.05, quaternion2=ROLL90, wavelength_nm: float = 780.0) -> dict:
    """C1 (power along design Y: section X of its own frame) at x = 0 and C2 (rolled: power along design Z) at x2."""

    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=waist_mm, wavelength_nm=wavelength_nm),
                   declared_component("C1", "lens", "cyl", (0.0, 0.0, 0.0), ALONG_X),
                   declared_component("C2", "lens", "cyl", (x2, 0.0, 0.0), quaternion2)],
                  [plane("far", (3000.0, 0.0, 0.0), radius_mm=60.0)])
    return with_models(doc, [CYLINDER], [_material()])


def _expander(x2: float = 160.0) -> dict:
    return project([source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, wavelength_nm=780.0), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0)),
                    element("L2", "lens", PCX_F100, (x2, 0.0, 0.0))], [plane("far", (x2 + 2000.0, 0.0, 0.0), radius_mm=40.0)])


def _far(result):
    return next(o for o in result.observations if o.observation_id == "far").arrivals[0].beam.gaussian


def _codes(document: dict, request: dict, **kwargs) -> list[dict]:
    with pytest.raises(ParaxialError) as info:
        paraxial_analysis(document, request, **kwargs)
    return list(info.value.problems)


# ------------------------------------------------------------------------------------------------ crossed cylinders


def test_crossed_cylinders_use_each_elements_own_section_matrices():
    """B3: C2 rolled 90 deg has its power in the section where C1 has none. Each section's system is the independent product
    of the curved matrix of the cylinder powered there and the plate matrix of the other -- never both powers in one section."""

    lam = 780.0
    curved, straight = _cylinder_matrices(lam)
    gap = 60.0 - CYL_T
    doc = _crossed(60.0)
    expected = {"X": straight @ free(gap) @ curved, "Y": curved @ free(gap) @ straight}
    for axis in ("X", "Y"):
        reply = paraxial_analysis(doc, {"components": ["C1", "C2"], "wavelength_nm": lam, "axis": axis})
        assert reply["section_frame"]["reference"] == "component C1"
        assert reply["section_frame"]["x"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-15)
        assert np.array(reply["system"]["abcd"]) == pytest.approx(expected[axis], abs=1e-12)
        other = "Y" if axis == "X" else "X"
        assert [e["element_section"] for e in reply["elements"]] == [axis, other]
        powered, plain = (0, 1) if axis == "X" else (1, 0)
        assert reply["elements"][powered]["efl_mm"] == pytest.approx(-1 / curved[1, 0], rel=1e-12)
        assert reply["elements"][plain]["efl_mm"] is None
        for section, matrix in expected.items():
            assert np.array(reply["sections"][section]["abcd"]) == pytest.approx(matrix, abs=1e-12)


def test_crossed_cylinders_have_no_afocal_solution_in_one_section():
    """The rv2 reproducer: the old analysis summed both powers into section X and 'solved' an afocal gap at which the traced
    beam diverged 40x more than the input. Section X has one powered element: no afocal solution."""

    problems = _codes(_crossed(100.0, waist_mm=1.0), {"components": ["C1", "C2"], "wavelength_nm": 780, "axis": "X", "solve": "afocal"})
    assert [p["code"] for p in problems] == ["PARAXIAL_NO_SOLUTION"]
    assert "'C2' has none in section X" in problems[0]["message"]


def test_parallel_cylinder_pair_is_afocal_at_the_independent_gap_and_states_the_other_section():
    """Two cylinders powered in the same section (C2 turned around: its local X reversed, still parallel): afocal in X at the
    independent gap; section Y (two plates) is afocal too. The traced beam at the solution: x divergence = input / |A|."""

    lam = 780.0
    curved, _straight = _cylinder_matrices(lam)
    doc = _crossed(150.0, waist_mm=1.0, quaternion2=AGAINST_X)
    reply = paraxial_analysis(doc, {"components": ["C1", "C2"], "wavelength_nm": lam, "axis": "X", "solve": "afocal"})
    turned = np.array([[curved[1, 1], curved[0, 1]], [curved[1, 0], curved[0, 0]]])
    gap = _afocal_gap(curved, turned)
    solution = reply["solution"]
    assert solution["section"] == "X" and solution["vertex_gap_mm"] == pytest.approx(gap, abs=1e-9)
    assert solution["other_section"]["afocal"] is True and solution["notes"] == []
    moved = copy.deepcopy(doc)
    moved["components"][2]["nominal_pose"]["position_mm"] = solution["position_mm"]
    g = _far(run(moved).result)
    theta_in = lam * NM / (math.pi * 1.0)
    assert g.far_field_half_angle_x_rad == pytest.approx(theta_in / abs(solution["beam_size_magnification"]), rel=1e-6)
    assert g.far_field_half_angle_y_rad == pytest.approx(theta_in, rel=1e-6)


def test_crossed_pair_afocal_in_one_section_notes_the_other():
    """A crossed pair with a spherical second lens: afocal in X does not make Y afocal -- the reply says so."""

    doc = _crossed(150.0, waist_mm=1.0)
    doc["components"][2] = element("C2", "lens", PCX_F100, (150.0, 0.0, 0.0))
    solution = paraxial_analysis(doc, {"components": ["C1", "C2"], "wavelength_nm": 780, "axis": "X", "solve": "afocal"},
                                 catalog=catalog_document())["solution"]
    assert solution["other_section"]["afocal"] is False
    assert [n["code"] for n in solution["notes"]] == ["OTHER_SECTION_NOT_AFOCAL"]


def test_a_cylinder_turned_45_degrees_about_the_axis_is_refused():
    problems = _codes(_crossed(60.0, quaternion2=ROLL45), {"components": ["C1", "C2"], "wavelength_nm": 780, "axis": "X"})
    assert [p["code"] for p in problems] == ["PARAXIAL_CYLINDER_ROTATED"]
    assert "45 deg" in problems[0]["message"] and problems[0]["location"] == "component C2"


def test_collimate_in_one_section_propagates_the_other_source_axis_through_its_own_section():
    """M6 / B3 with a trace: crossed cylinders, collimate in section X with the source's waist. The x output is the collimated
    beam of C1's front focal point; y goes through section Y (C2's power) and is reported UNCOLLIMATED. Both predicted
    far-field divergences equal the Engine's traced ones."""

    lam, w0 = 780.0, 0.05
    doc = _crossed(60.0, waist_mm=w0)
    reply = paraxial_analysis(doc, {"components": ["C1", "C2"], "wavelength_nm": lam, "axis": "X", "solve": "collimate", "source_id": "SRC"})
    solution = reply["solution"]
    assert reply["section_frame"]["reference"] == "source SRC"
    curved, straight = _cylinder_matrices(lam)
    gap = 60.0 - CYL_T
    mx, my = straight @ free(gap) @ curved, curved @ free(gap) @ straight
    ffd = -mx[1, 1] / mx[1, 0]
    assert solution["object_distance_mm"] == pytest.approx(ffd, rel=1e-12)
    x, y = solution["output_beam"]["x"], solution["output_beam"]["y"]
    assert (x["section"], x["state"], y["section"], y["state"]) == ("X", "COLLIMATED", "Y", "UNCOLLIMATED")
    assert [n["code"] for n in solution["notes"]] == ["OTHER_SECTION_NOT_COLLIMATED"]
    zr = rayleigh_range(w0, lam)
    for beam, matrix in ((x, mx), (y, my)):
        assert beam["waist_radius_mm"] == pytest.approx(waist_radius(q_through(matrix, complex(ffd, zr)), lam), rel=1e-9)
    moved = copy.deepcopy(doc)
    for comp in moved["components"][1:]:
        comp["nominal_pose"]["position_mm"] = solution["positions_mm"][comp["instance_id"]]
    g = _far(run(moved).result)
    assert g.far_field_half_angle_x_rad == pytest.approx(x["far_field_half_angle_rad"], rel=1e-6)
    assert g.far_field_half_angle_y_rad == pytest.approx(y["far_field_half_angle_rad"], rel=1e-6)


def test_single_cylinder_collimate_leaves_the_other_axis_unfocused():
    """M6 reproducer: one cylinder, axis X: y has no power, so the predicted y beam is the free-space beam (= the trace)."""

    lam, w0 = 780.0, 0.05
    doc = with_models(project([source(position=(-200.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam),
                               declared_component("C1", "lens", "cyl", (0.0, 0.0, 0.0), ALONG_X)],
                              [plane("far", (3000.0, 0.0, 0.0), radius_mm=60.0)]), [CYLINDER], [_material()])
    solution = paraxial_analysis(doc, {"components": ["C1"], "wavelength_nm": lam, "axis": "X", "solve": "collimate", "source_id": "SRC"})["solution"]
    y = solution["output_beam"]["y"]
    assert y["state"] == "UNCOLLIMATED" and y["waist_radius_mm"] == pytest.approx(w0, rel=1e-12)
    assert y["far_field_half_angle_rad"] == pytest.approx(lam * NM / (math.pi * w0), rel=1e-12)
    moved = copy.deepcopy(doc)
    moved["components"][1]["nominal_pose"]["position_mm"] = solution["positions_mm"]["C1"]
    g = _far(run(moved).result)
    assert g.far_field_half_angle_y_rad == pytest.approx(y["far_field_half_angle_rad"], rel=1e-6)
    assert g.far_field_half_angle_x_rad == pytest.approx(solution["output_beam"]["x"]["far_field_half_angle_rad"], rel=1e-6)


# ------------------------------------------------------------------------------------------------ collimate meaning


def test_collimate_reports_a_focus_when_the_source_is_already_collimated():
    """A 1 mm waist at 632.8 nm (Rayleigh range ~ 5 m) at the front focal point of an f ~ 50 mm lens is focused, not
    collimated: state FOCUSED with INPUT_ALREADY_COLLIMATED; output waist lambda f / (pi w0), output Rayleigh range f^2 / z_R."""

    lam, w0 = 632.8, 1.0
    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0))],
                  [plane("far", (3000.0, 0.0, 0.0), radius_mm=40.0)])
    solution = paraxial_analysis(doc, {"components": ["L1"], "wavelength_nm": lam, "solve": "collimate", "source_id": "SRC"},
                                 catalog=catalog_document())["solution"]
    efl = -1 / lens_matrix(PCX_F50, lam)[1, 0]
    beam = solution["output_beam"]["x"]
    assert beam["input_rayleigh_range_mm"] == pytest.approx(rayleigh_range(w0, lam), rel=1e-12) and beam["input_rayleigh_range_mm"] > abs(efl)
    assert beam["state"] == "FOCUSED" and solution["output_beam"]["y"]["state"] == "FOCUSED"
    assert {n["code"] for n in solution["notes"]} == {"INPUT_ALREADY_COLLIMATED"}
    assert beam["waist_radius_mm"] == pytest.approx(lam * NM * efl / (math.pi * w0), rel=1e-6)
    assert beam["rayleigh_range_mm"] == pytest.approx(efl * efl / rayleigh_range(w0, lam), rel=1e-6) and beam["rayleigh_range_mm"] < abs(efl)
    assert "only when the source's Rayleigh range is at most |EFL|" in solution["meaning"]


def test_collimate_with_a_short_rayleigh_range_is_collimated():
    lam, w0 = 780.0, 0.005
    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0))],
                  [plane("far", (3000.0, 0.0, 0.0), radius_mm=40.0)])
    solution = paraxial_analysis(doc, {"components": ["L1"], "wavelength_nm": lam, "solve": "collimate", "source_id": "SRC"},
                                 catalog=catalog_document())["solution"]
    assert solution["output_beam"]["x"]["state"] == "COLLIMATED" and solution["notes"] == []


# ------------------------------------------------------------------------------------------------ source checks


@pytest.mark.parametrize("components", [["L1"], ["L1", "L2"]], ids=["one", "two"])
def test_an_off_axis_source_is_reported_by_its_lateral_offset(components):
    """N7: a source 3 mm off the axis is named with its offset (not as a tilted element), with one or several components."""

    doc = _expander()
    doc["components"][0]["nominal_pose"]["position_mm"] = [-200.0, 3.0, 0.0]
    problems = _codes(doc, {"components": components, "wavelength_nm": 780, "source_id": "SRC"}, catalog=catalog_document())
    assert problems[0]["code"] == "PARAXIAL_NOT_COAXIAL" and problems[0]["location"] == "source_id"
    assert "source 'SRC' is 3 mm off the optical axis" in problems[0]["message"]


def test_a_source_emitting_away_from_the_components_is_refused():
    doc = _expander()
    doc["components"][0]["nominal_pose"]["quaternion_wxyz"] = list(AGAINST_X)
    one = _codes(doc, {"components": ["L1"], "wavelength_nm": 780, "source_id": "SRC", "solve": "collimate"}, catalog=catalog_document())
    assert one[0]["code"] == "PARAXIAL_ORDER_INVALID" and "does not reach" in one[0]["message"]
    two = _codes(doc, {"components": ["L1", "L2"], "wavelength_nm": 780, "source_id": "SRC"}, catalog=catalog_document())
    assert two[0]["code"] == "PARAXIAL_NOT_COAXIAL" and "emits against the optical axis" in two[0]["message"]
