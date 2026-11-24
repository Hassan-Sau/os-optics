"""Paraxial analysis of placed components (``os_optics.local.paraxial``).

Every expected number is computed here independently: thick-lens ABCD matrices from the prescriptions with NumPy and a
Sellmeier index written in the test (``local_helpers.lens_matrix`` / ``sellmeier_index``, or the BB1-02 prescriptions
below), Gaussian q propagation through them, and the textbook afocal / collimation conditions. The Engine runs of the
solved layouts are then checked against those references (output divergence, beam size, the BB1-02 targets).
"""

from __future__ import annotations

import copy
import io
import json
import math
from contextlib import redirect_stdout

import numpy as np
import pytest

from os_optics.cli import main as cli_main
from os_optics.local import ParaxialError, json_api, paraxial_analysis

from declared_helpers import BAND, NBK7_SELLMEIER, declared_component, sequence, surface, with_models
from local_helpers import (
    AGAINST_X,
    ALONG_X,
    ALONG_Y,
    PCV_FN50,
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


def _afocal_gap(m1: np.ndarray, m2: np.ndarray) -> float:
    """Vertex gap d with C = 0 for M2 T(d) M1 (C is linear in d: two evaluations)."""

    c = [(m2 @ free(d) @ m1)[1, 0] for d in (0.0, 1.0)]
    return -c[0] / (c[1] - c[0])


def _expander(sku1: str, sku2: str, x2: float, *, waist_mm: float = 1.0, wavelength_nm: float = 780.0, turned2: bool = False) -> dict:
    return project([source(position=(-200.0, 0.0, 0.0), waist_mm=waist_mm, wavelength_nm=wavelength_nm),
                    element("L1", "lens", sku1, (0.0, 0.0, 0.0)),
                    element("L2", "lens", sku2, (x2, 0.0, 0.0), AGAINST_X if turned2 else ALONG_X)],
                   [plane("near", (x2 + 30.0, 0.0, 0.0), radius_mm=20.0), plane("far", (x2 + 2000.0, 0.0, 0.0), radius_mm=40.0)])


def _far_arrival(result, observation_id: str = "far"):
    obs = next(o for o in result.observations if o.observation_id == observation_id)
    return obs.arrivals[0].beam.gaussian


# ------------------------------------------------------------------------------------------------ matrices


def test_element_and_system_matrices_equal_an_independent_thick_lens_abcd():
    doc = _expander(PCX_F50, PCX_F100, 160.0)
    reply = paraxial_analysis(doc, {"components": ["L1", "L2"], "wavelength_nm": 633}, catalog=catalog_document())
    m1, m2 = lens_matrix(PCX_F50, 633.0), lens_matrix(PCX_F100, 633.0)
    assert np.array(reply["elements"][0]["abcd"]) == pytest.approx(m1, abs=1e-12)
    assert np.array(reply["elements"][1]["abcd"]) == pytest.approx(m2, abs=1e-12)
    gap = 160.0 - 5.3  # L1's exit vertex (5.3 mm) to L2's entrance vertex
    system = m2 @ free(gap) @ m1
    out = reply["system"]
    assert out["vertex_gaps_mm"] == pytest.approx([gap], abs=1e-12)
    assert np.array(out["abcd"]) == pytest.approx(system, abs=1e-12)
    (a, _b), (c, d) = system
    assert out["efl_mm"] == pytest.approx(-1 / c, rel=1e-12)
    assert out["bfl_mm"] == pytest.approx(-a / c, rel=1e-12)
    assert out["ffd_mm"] == pytest.approx(-d / c, rel=1e-12)
    assert out["rear_principal_plane_mm"] == pytest.approx((1 - a) / c, rel=1e-12)
    assert out["front_principal_plane_mm"] == pytest.approx((d - 1) / c, rel=1e-12)
    # focal points in the design frame: before the entrance vertex at x = 0, after the exit vertex at x = 160 + 3.6
    assert out["front_focal_point_mm"] == pytest.approx([0.0 + d / c, 0.0, 0.0], abs=1e-9)
    assert out["rear_focal_point_mm"] == pytest.approx([163.6 - a / c, 0.0, 0.0], abs=1e-9)
    single = reply["elements"][0]
    assert single["efl_mm"] == pytest.approx(-1 / m1[1, 0], rel=1e-12) and single["turned"] is False
    assert single["entrance_vertex_mm"] == [0.0, 0.0, 0.0] and single["exit_vertex_mm"] == pytest.approx([5.3, 0.0, 0.0])


def test_a_turned_element_uses_the_reversed_prescription():
    doc = _expander(PCX_F50, PCX_F100, 160.0, turned2=True)
    reply = paraxial_analysis(doc, {"components": ["L1", "L2"], "wavelength_nm": 780}, catalog=catalog_document())
    l2 = reply["elements"][1]
    assert l2["turned"] is True
    assert np.array(l2["abcd"]) == pytest.approx(lens_matrix(PCX_F100, 780.0, turned=True), abs=1e-12)
    # turned around, the plano face (local z = 3.6) meets the light first
    assert l2["entrance_vertex_mm"] == pytest.approx([156.4, 0.0, 0.0]) and l2["exit_vertex_mm"] == pytest.approx([160.0, 0.0, 0.0])
    assert reply["system"]["vertex_gaps_mm"] == pytest.approx([156.4 - 5.3])


# ------------------------------------------------------------------------------------------------ afocal


@pytest.mark.parametrize("sku1, sku2", [(PCX_F50, PCX_F100), (PCV_FN50, PCX_F100)], ids=["keplerian", "galilean"])
def test_afocal_spacing_matches_the_independent_condition_and_collimates_in_a_run(sku1, sku2):
    """Keplerian (f+50, f+100) and Galilean (f-50, f+100) 2x expanders at 780 nm: the helper's vertex gap is the independent C = 0
    gap; at it the Engine's output beam is collimated: its far-field divergence is the input's divided by |A| (an afocal map
    q' = A^2 q + A B), with |A| ~ 2 and the image inverted (A < 0) only for the Keplerian pair."""

    lam = 780.0
    doc = _expander(sku1, sku2, 140.0)
    reply = paraxial_analysis(doc, {"components": ["L1", "L2"], "wavelength_nm": lam, "solve": "afocal"}, catalog=catalog_document())
    m1, m2 = lens_matrix(sku1, lam), lens_matrix(sku2, lam)
    gap = _afocal_gap(m1, m2)
    solution = reply["solution"]
    assert solution["kind"] == "AFOCAL" and solution["vertex_gap_mm"] == pytest.approx(gap, abs=1e-9)
    system = m2 @ free(gap) @ m1
    assert abs(system[1, 0]) < 1e-15
    assert solution["beam_size_magnification"] == pytest.approx(system[0, 0], rel=1e-9)
    assert solution["angular_magnification"] == pytest.approx(system[1, 1], rel=1e-9)
    assert abs(system[0, 0]) == pytest.approx(2.0, rel=0.02)
    assert (system[0, 0] < 0) == (sku1 == PCX_F50)
    exit_l1 = 5.3 if sku1 == PCX_F50 else 3.5
    assert solution["position_mm"] == pytest.approx([exit_l1 + gap, 0.0, 0.0], abs=1e-9)
    assert solution["move_along_axis_mm"] == pytest.approx(exit_l1 + gap - 140.0, abs=1e-9)

    solved = run(_expander(sku1, sku2, solution["position_mm"][0]))
    g = _far_arrival(solved.result)
    theta_in = lam * NM / (math.pi * 1.0)
    assert g.far_field_half_angle_x_rad == pytest.approx(theta_in / abs(system[0, 0]), rel=1e-6)
    # off the solution the output is no longer collimated: the divergence grows
    for offset in (-2.0, 2.0):
        detuned = run(_expander(sku1, sku2, solution["position_mm"][0] + offset))
        assert _far_arrival(detuned.result).far_field_half_angle_x_rad > 1.5 * g.far_field_half_angle_x_rad


# ------------------------------------------------------------------------------------------------ collimation


def test_collimation_distance_puts_the_waist_at_the_front_focal_point_and_minimizes_the_output_divergence():
    """A 5 um waist (a fibre-like point source) at 780 nm in front of the f+50 lens: the helper's object distance is the
    independent front focal distance; the predicted output beam equals the independent q propagation and the Engine's run;
    moving the lens either way increases the output divergence (a Gaussian output waist is largest at s = FFD)."""

    lam, w0 = 780.0, 0.005
    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0))],
                  [plane("far", (3000.0, 0.0, 0.0), radius_mm=40.0)])
    reply = paraxial_analysis(doc, {"components": ["L1"], "wavelength_nm": lam, "solve": "collimate", "source_id": "SRC"},
                              catalog=catalog_document())
    m = lens_matrix(PCX_F50, lam)
    ffd = -m[1, 1] / m[1, 0]
    solution = reply["solution"]
    assert solution["object_distance_mm"] == pytest.approx(ffd, rel=1e-12)
    assert solution["current_object_distance_mm"] == pytest.approx(200.0)
    assert solution["move_along_axis_mm"] == pytest.approx(ffd - 200.0, abs=1e-9)
    x_lens = solution["positions_mm"]["L1"][0]
    assert x_lens == pytest.approx(-200.0 + ffd, abs=1e-9)
    q_out = q_through(m, complex(ffd, rayleigh_range(w0, lam)))
    w_out = waist_radius(q_out, lam)
    beam = solution["output_beam"]["x"]
    assert beam["waist_radius_mm"] == pytest.approx(w_out, rel=1e-9)
    assert beam["waist_after_output_plane_mm"] == pytest.approx(-q_out.real, rel=1e-9)
    assert beam["far_field_half_angle_rad"] == pytest.approx(lam * NM / (math.pi * w_out), rel=1e-9)

    def divergence(x: float) -> float:
        moved = copy.deepcopy(doc)
        moved["components"][1]["nominal_pose"]["position_mm"][0] = x
        return _far_arrival(run(moved).result).far_field_half_angle_x_rad

    at_solution = divergence(x_lens)
    assert at_solution == pytest.approx(beam["far_field_half_angle_rad"], rel=1e-6)
    assert divergence(x_lens - 0.5) > at_solution and divergence(x_lens + 0.5) > at_solution


# ------------------------------------------------------------------------------------------------ BB1-02


NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)


def _nbk7(wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1 + sum(b * lam2 / (lam2 - c) for b, c in zip(NBK7_B, NBK7_C)))


def _bb102_project(x2: float) -> dict:
    """Audit case BB1-02: a 3x expander at 780 nm (w0 = 1 mm source 200 mm before L1) from the plano-convex N-BK7 lenses
    R 25.8 mm / 5.3 mm and R 77.3 mm / 3.1 mm (the published prescriptions of the two catalog lenses the audit used), L2 turned so
    its plano face meets the light; targets: 3 mm (+-2 %) beam radius 20 mm and 400 mm after L2, divergence below 0.5 mrad."""

    l1 = {"model_id": "pcx_r25p8", "model": sequence([surface(1, "0", radius="25.8", clear="11.43"), surface(2, "5.3", clear="11.43")],
                                                     ["N-BK7"], edge="12.7")}
    l2 = {"model_id": "pcx_r77p3", "model": sequence([surface(1, "0", radius="77.3", clear="11.43"), surface(2, "3.1", clear="11.43")],
                                                     ["N-BK7"], edge="12.7")}
    material = {"material_id": "N-BK7", "display_name": "N-BK7", "dispersion": dict(NBK7_SELLMEIER), "valid_band_nm": dict(BAND),
                "basis": "Schott N-BK7 Sellmeier formula"}
    targets = [
        {"target_id": "near_radius", "metric": "beam_radius_x_mm", "observation_id": "near_field", "target_value": 3.0, "tolerance_rel": 0.02,
         "unit": "mm"},
        {"target_id": "near_radius_y", "metric": "beam_radius_y_mm", "observation_id": "near_field", "target_value": 3.0,
         "tolerance_rel": 0.02, "unit": "mm"},
        {"target_id": "far_radius", "metric": "beam_radius_x_mm", "observation_id": "target_400mm", "target_value": 3.0,
         "tolerance_rel": 0.02, "unit": "mm"},
        {"target_id": "far_divergence", "metric": "divergence_half_angle_x_rad", "observation_id": "target_400mm", "maximum": 0.0005,
         "unit": "rad"},
    ]
    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, wavelength_nm=780.0),
                   declared_component("L1", "lens", "pcx_r25p8", (0.0, 0.0, 0.0), ALONG_X),
                   declared_component("L2", "lens", "pcx_r77p3", (x2, 0.0, 0.0), AGAINST_X)],
                  [plane("near_field", (x2 + 20.0, 0.0, 0.0), radius_mm=15.0), plane("target_400mm", (x2 + 400.0, 0.0, 0.0), radius_mm=15.0)],
                  targets)
    return with_models(doc, [l1, l2], [material])


def _bb102_reference_position() -> float:
    """The afocal L2 vertex position from an independent thick-lens ABCD (the audit's own reference gives 204.5303 mm)."""

    n = _nbk7(780.0)

    def refract(radius: float, n1: float, n2: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [-(n2 - n1) / radius, 1.0]])  # reduced (y, n u) form

    def pair(x2: float) -> np.ndarray:
        m = free(5.3 / n) @ refract(25.8, 1.0, n)  # L1: curved face at x = 0, plano at 5.3
        m = free(x2 - 3.1 - 5.3) @ m  # air to L2's plano face at x2 - 3.1
        return refract(-77.3, n, 1.0) @ free(3.1 / n) @ m  # L2 turned: plano face first, curved vertex at x2 (centre upstream)

    c0, c1 = pair(204.0)[1, 0], pair(205.0)[1, 0]
    return 204.0 - c0 / (c1 - c0)


def test_bb1_02_thick_lens_afocal_spacing_passes_where_thin_lens_spacings_fail():
    """Reproduces the BB1-02 finding: thin-lens spacings (L2 at 200.0 mm, and 202.2 mm from thin-lens focal lengths at 780 nm)
    miss the customer's targets; the helper's thick-lens afocal position (= the independent reference, 204.530 mm) passes all
    four targets in an Engine run."""

    reference = _bb102_reference_position()
    assert reference == pytest.approx(204.5303, abs=1e-4)  # the audit's independent value
    reply = json_api.paraxial_analysis(_bb102_project(200.0), {"components": ["L1", "L2"], "wavelength_nm": 780, "solve": "afocal"})
    assert reply["ok"] is True
    solution = reply["solution"]
    assert solution["position_mm"][0] == pytest.approx(reference, abs=1e-9)
    assert solution["beam_size_magnification"] == pytest.approx(-2.99612, abs=1e-5)
    outcomes = {}
    for x2 in (200.0, 202.2, solution["position_mm"][0]):
        result = run(_bb102_project(x2))
        outcomes[x2] = (result.status.value, {t.target_id: t.status.value for t in result.result.targets},
                        {(issue.code.value, issue.status.value) for issue in result.result.issues})
    assert outcomes[200.0][0] == "FAIL" and set(outcomes[200.0][1].values()) == {"FAIL"}
    assert outcomes[202.2][0] == "FAIL" and outcomes[202.2][1]["far_radius"] == "FAIL"
    solved = outcomes[solution["position_mm"][0]]
    # All four independently referenced *geometric* targets pass at the same thick-lens spacing. The catalog's
    # glass has no internal-loss data, so the aggregate run cannot claim known transmitted power or a full PASS.
    assert solved[0] == "UNKNOWN" and set(solved[1].values()) == {"PASS"}
    # Unknown bulk loss is mandatory; a NOT_APPLICABLE withheld-bound notice is conditional on
    # an unbounded return/closed path, not on the absence of the glass's absorption data alone.
    assert ("BULK_LOSS_NOT_ESTABLISHED", "WARNING") in solved[2]
    assert solved[2] <= {("BULK_LOSS_NOT_ESTABLISHED", "WARNING"), ("EXPOSURE_BOUND_WITHHELD", "NOT_APPLICABLE")}


# ------------------------------------------------------------------------------------------------ refusals, CLI


def test_refusals_are_structured():
    doc = _expander(PCX_F50, PCX_F100, 160.0)
    cat = catalog_document()

    def codes(request: dict, document: dict = doc) -> list[str]:
        with pytest.raises(ParaxialError) as info:
            paraxial_analysis(document, request, catalog=cat)
        return [p["code"] for p in info.value.problems]

    assert codes({"components": ["L1"], "wavelength_nm": 780, "extra": 1}) == ["PARAXIAL_REQUEST_INVALID"]
    assert codes({"components": ["L1", "L1"], "wavelength_nm": 780}) == ["PARAXIAL_REQUEST_INVALID"]
    assert codes({"components": ["L1"], "wavelength_nm": -1}) == ["PARAXIAL_REQUEST_INVALID"]
    assert codes({"components": ["L1"], "wavelength_nm": 780, "solve": "afocal"}) == ["PARAXIAL_REQUEST_INVALID"]
    assert codes({"components": ["NOPE"], "wavelength_nm": 780}) == ["PARAXIAL_COMPONENT_UNKNOWN"]
    assert codes({"components": ["SRC"], "wavelength_nm": 780}) == ["PARAXIAL_ELEMENT_UNSUPPORTED"]
    # listed the other way round: the Project's source lights L1 first along +X, so the list is refused as with source_id
    assert codes({"components": ["L2", "L1"], "wavelength_nm": 780}) == ["PARAXIAL_NOT_COAXIAL"]
    assert codes({"components": ["L2", "L1"], "wavelength_nm": 780, "source_id": "SRC"}) == ["PARAXIAL_NOT_COAXIAL"]
    # with no source on the line the direction is assumed and labelled: light travelling -X, both lenses turned around (a reverse analysis)
    sourceless = copy.deepcopy(doc)
    sourceless["components"] = [c for c in sourceless["components"] if c["instance_id"] != "SRC"]
    reverse = paraxial_analysis(sourceless, {"components": ["L2", "L1"], "wavelength_nm": 780}, catalog=cat)
    assert [e["turned"] for e in reverse["elements"]] == [True, True]
    assert reverse["light_direction"]["rule"] == "ASSUMED" and reverse["light_direction"]["direction"] == pytest.approx([-1.0, 0.0, 0.0])
    tilted = copy.deepcopy(doc)
    tilted["components"][2]["nominal_pose"]["quaternion_wxyz"] = list(ALONG_Y)
    assert codes({"components": ["L1", "L2"], "wavelength_nm": 780}, tilted) == ["PARAXIAL_NOT_COAXIAL"]
    off_axis = copy.deepcopy(doc)
    off_axis["components"][2]["nominal_pose"]["position_mm"] = [160.0, 0.5, 0.0]
    assert codes({"components": ["L1", "L2"], "wavelength_nm": 780}, off_axis) == ["PARAXIAL_NOT_COAXIAL"]
    overlapping = _expander(PCX_F50, PCX_F100, 4.0)
    assert codes({"components": ["L1", "L2"], "wavelength_nm": 780}, overlapping) == ["PARAXIAL_ORDER_INVALID"]
    assert codes({"components": ["L1"], "wavelength_nm": 5000}) == ["PARAXIAL_ELEMENT_UNSUPPORTED"]  # outside N-BK7's data
    refused = json_api.paraxial_analysis(doc, {"components": ["NOPE"], "wavelength_nm": 780}, catalog=cat)
    assert refused["ok"] is False and refused["error"]["code"] == "PARAXIAL_ANALYSIS_INVALID"


def test_cli_prints_the_analysis_and_example_help_points_to_it(tmp_path):
    path = tmp_path / "project.json"
    path.write_text(json.dumps(_bb102_project(200.0)), encoding="utf-8")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "paraxial", str(path), "--components", "L1", "L2", "--wavelength-nm", "780", "--solve", "afocal"])
    assert code == 0
    reply = json.loads(out.getvalue())
    assert reply["solution"]["position_mm"][0] == pytest.approx(_bb102_reference_position(), abs=1e-9)
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "paraxial", str(path), "--components", "L9", "--wavelength-nm", "780"])
    assert code == 65 and json.loads(out.getvalue())["error"]["problems"][0]["code"] == "PARAXIAL_COMPONENT_UNKNOWN"
    help_text = io.StringIO()
    with redirect_stdout(help_text), pytest.raises(SystemExit):
        cli_main(["project", "example", "--help"])
    assert "os-optics project paraxial" in help_text.getvalue()
    caps = json_api.capabilities()
    assert caps["paraxial_analysis"]["solves"] == ["afocal", "collimate"]


def test_a_project_without_a_light_source_is_analysed_when_the_request_names_none():
    """PV-O1: lens spacing can be found before a source is placed. Without ``source_id`` the analysis needs no source in the
    Project (the same afocal answer as with one); naming a source the Project lacks is still refused."""

    with_source = _bb102_project(200.0)
    without = copy.deepcopy(with_source)
    without["components"] = [c for c in without["components"] if c["optical_role"] != "source"]
    without["targets"] = []
    request = {"components": ["L1", "L2"], "wavelength_nm": 780, "solve": "afocal"}
    reply = json_api.paraxial_analysis(without, request)
    assert reply["ok"] is True
    lit = json_api.paraxial_analysis(with_source, request)
    # the same answer; only where the light's direction comes from differs (the Project's source, or assumed from L1 to L2)
    assert (reply["light_direction"]["rule"], lit["light_direction"]["rule"]) == ("ASSUMED", "PROJECT_SOURCE")
    assert reply["light_direction"]["direction"] == lit["light_direction"]["direction"]
    assert {k: v for k, v in reply.items() if k != "light_direction"} == {k: v for k, v in lit.items() if k != "light_direction"}
    refused = json_api.paraxial_analysis(without, {**request, "source_id": "SRC"})
    assert refused["ok"] is False
    assert [p["code"] for p in refused["error"]["problems"]] == ["SOURCE_MISSING"]


# ------------------------------------------------------------------------------------------------ the coaxial tolerance


def _three_lenses(middle_position=(80.0, 0.0, 0.0), middle_tilt_rad: float = 0.0) -> dict:
    """Three catalog lenses along +X at x = 0, 80, 160 mm, no source; the middle one moved and turned about global Z."""

    half = middle_tilt_rad / 2.0
    pose = quat_multiply(turn_about_z(math.cos(half), math.sin(half)), ALONG_X)
    return project([element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0)), element("L2", "lens", PCX_F100, middle_position, pose),
                    element("L3", "lens", PCX_F50, (160.0, 0.0, 0.0))], [plane("far", (400.0, 0.0, 0.0), radius_mm=40.0)])


def test_a_stated_coaxial_tolerance_answers_a_slightly_misaligned_layout_on_the_nominal_axis_and_states_what_was_measured():
    """The middle lens 5e-6 mm off the axis and tilted by 3e-6 rad: outside the Engine's tolerance (1e-6 mm; 1 - cos <= 1e-12, about
    1.414e-6 rad), so the layout is refused by default, the message naming the measured value, the tolerance and the request field.
    A request that states coaxial_tolerance accepts it: the analysis is the nominal-axis one (the aligned layout's matrix to 1e-9),
    every element states its measured decentre and tilt, and the reply the tolerance used."""

    cat = catalog_document()
    request = {"components": ["L1", "L2", "L3"], "wavelength_nm": 780}
    aligned = paraxial_analysis(_three_lenses(), request, catalog=cat)
    assert aligned["coaxial_tolerance"] == {"lateral_mm": 1e-6, "tilt_rad": math.sqrt(2e-12)}
    assert [(e["decentre_mm"], e["tilt_rad"]) for e in aligned["elements"]] == [(0.0, 0.0)] * 3
    moved = _three_lenses((80.0, 5e-6, 0.0), 3e-6)
    with pytest.raises(ParaxialError) as info:
        paraxial_analysis(moved, request, catalog=cat)
    (problem,) = info.value.problems
    assert problem["code"] == "PARAXIAL_NOT_COAXIAL" and "3e-06 rad" in problem["message"] and "coaxial_tolerance.tilt_rad is 1.41421e-06 rad" in problem["message"]
    with pytest.raises(ParaxialError) as info:  # (the tilt accepted, the decentre not)
        paraxial_analysis(moved, {**request, "coaxial_tolerance": {"tilt_rad": 5e-6}}, catalog=cat)
    (problem,) = info.value.problems
    assert "'L2' is 5e-06 mm off the optical axis" in problem["message"] and "coaxial_tolerance.lateral_mm is 1e-06 mm" in problem["message"]
    reply = paraxial_analysis(moved, {**request, "coaxial_tolerance": {"lateral_mm": 1e-5, "tilt_rad": 5e-6}}, catalog=cat)
    assert reply["coaxial_tolerance"] == {"lateral_mm": 1e-5, "tilt_rad": 5e-6}
    middle = reply["elements"][1]
    assert middle["decentre_mm"] == pytest.approx(5e-6, rel=1e-6) and middle["tilt_rad"] == pytest.approx(3e-6, rel=1e-6)
    assert reply["elements"][0]["decentre_mm"] == 0.0 and reply["elements"][2]["tilt_rad"] == 0.0
    assert reply["system"]["efl_mm"] == pytest.approx(aligned["system"]["efl_mm"], rel=1e-9)
    assert np.array(reply["system"]["abcd"]) == pytest.approx(np.array(aligned["system"]["abcd"]), rel=1e-9, abs=1e-12)
    for stated in ({"lateral_mm": -1.0}, {"tilt_rad": 2.0}, {"radius": 1.0}, 3):
        with pytest.raises(ParaxialError) as info:
            paraxial_analysis(moved, {**request, "coaxial_tolerance": stated}, catalog=cat)
        assert info.value.problems[0]["code"] == "PARAXIAL_REQUEST_INVALID"
    answered = json_api.paraxial_analysis(moved, {**request, "coaxial_tolerance": {"lateral_mm": "1e-5", "tilt_rad": "5e-6"}}, catalog=cat)
    assert answered["ok"] and answered["coaxial_tolerance"] == {"lateral_mm": 1e-5, "tilt_rad": 5e-6}


def test_a_project_source_beyond_the_first_entrance_vertex_is_set_aside_unless_the_request_names_it():
    """A source between the two lenses, emitting along the axis: it lights L2 only, not the list [L1, L2] in its order. The request
    names no source, so it is set aside with a note and the reply is the sourceless Project's (the assumed direction); named, the
    request is refused as before."""

    cat = catalog_document()
    doc = _expander(PCX_F50, PCX_F100, 160.0)
    doc["components"][0]["nominal_pose"]["position_mm"] = [80.0, 0.0, 0.0]
    request = {"components": ["L1", "L2"], "wavelength_nm": 780}
    sourceless = copy.deepcopy(doc)
    sourceless["components"] = [c for c in sourceless["components"] if c["instance_id"] != "SRC"]
    plain = paraxial_analysis(sourceless, request, catalog=cat)
    reply = paraxial_analysis(doc, request, catalog=cat)
    assert reply["light_direction"]["rule"] == "ASSUMED" and reply["light_direction"]["source_ids"] == []
    assert reply["light_direction"]["notes"] == [{"code": "PARAXIAL_SOURCE_NOT_BEFORE_COMPONENTS", "source_ids": ["SRC"]}]
    assert {k: v for k, v in reply.items() if k != "light_direction"} == {k: v for k, v in plain.items() if k != "light_direction"}
    with pytest.raises(ParaxialError) as info:
        paraxial_analysis(doc, {**request, "source_id": "SRC"}, catalog=cat)
    assert [p["code"] for p in info.value.problems] == ["PARAXIAL_ORDER_INVALID"]
    lit = paraxial_analysis(_expander(PCX_F50, PCX_F100, 160.0), request, catalog=cat)
    assert lit["light_direction"]["rule"] == "PROJECT_SOURCE" and lit["light_direction"]["notes"] == []
