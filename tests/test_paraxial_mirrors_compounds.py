"""Paraxial analysis of mirrors, folded axes and compound units (``os_optics.local.paraxial``).

Every expected number is derived here by hand or computed independently of the code under test: the two-mirror formulas
``phi = 1/f1 + 1/f2 - d/(f1 f2)``, ``BFL = (1 - d/f1) / phi``, a y-nu trace with a Sellmeier index written in the test, Gaussian q
propagation through matrices written in the test. The traced runs then check the positions in the design frame.
"""

from __future__ import annotations

import copy
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import BAND, NBK7_SELLMEIER, cylinder, declared_component, sequence, surface, thin_lens_model, with_models
from local_helpers import (
    AGAINST_X,
    ALONG_X,
    ALONG_Y,
    CONCAVE_MIRROR,
    FLAT_MIRROR,
    FOLD_X_TO_Y,
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
)
from os_optics.cli import main as cli_main
from os_optics.local import ParaxialError, json_api, load_public_catalog, paraxial_analysis, run_optical_project
from os_optics.local import compile as C
from os_optics.prescription import import_prescription, merge_into_project

HR = next(c for c in catalog_document()["coatings"] if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
#: 180 deg about the unit's X: a member whose local +Z points back along the unit's +Z.
FACE_BACK = (0.0, 1.0, 0.0, 0.0)
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prescriptions"


def _codes(document: dict, request: dict, **kwargs) -> list[dict]:
    with pytest.raises(ParaxialError) as info:
        paraxial_analysis(document, request, **kwargs)
    return list(info.value.problems)


def _mirror(model_id: str, radius: str | None, clear: str = "60", **extra) -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", radius=radius, clear=clear, traced="REFLECTED", coating_id=HR["coating_id"], **extra)],
                                                    [], edge=str(int(clear) + 1))}


def _material() -> dict:
    return {"material_id": "N-BK7", "display_name": "N-BK7", "dispersion": dict(NBK7_SELLMEIER), "valid_band_nm": dict(BAND),
            "basis": "Schott N-BK7 Sellmeier formula"}


# ------------------------------------------------------------------------------------------------ the Cassegrain


def _cassegrain(image_x: float = 200.0, waist_mm: float = 20.0, *, m2_quaternion=AGAINST_X) -> dict:
    """The two-mirror system: concave R = +800 mm at x = 0, convex R = -150 mm at x = 350 mm; a collimated beam along -X from a source
    between them (a source has no body), and an image plane. Hand values: f1 = 400, f2 = -75, phi = 1/400 - 1/75 + 350/30000 =
    1/1200, so EFL = 1200, BFL = (1 - 350/400) * 1200 = 150, the image at x = 350 - 150 = 200."""

    doc = project([source(position=(340.0, 0.0, 0.0), quaternion=AGAINST_X, waist_mm=waist_mm, wavelength_nm=633.0),
                   declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X),
                   declared_component("M2", "mirror", "m2", (350.0, 0.0, 0.0), m2_quaternion)],
                  [plane("image", (image_x, 0.0, 0.0), AGAINST_X, radius_mm=60.0)])
    return with_models(doc, [_mirror("m1", "800"), _mirror("m2", "-150", "15")], coatings=[HR])


def _focus_pass(document: dict):
    """The arrival of the third crossing of the image plane: towards M1 after both mirrors (the first two are the beam on its way to M2)."""

    result = run_optical_project(document).result
    arrivals = next(o for o in result.observations if o.observation_id == "image").arrivals
    third = arrivals[2]
    assert third.beam.direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)
    return third


def test_paraxial_matrix_of_a_mirror():
    catalog = load_public_catalog(catalog_document())
    assert C.paraxial_matrix(C.compile_element("F", catalog.get(FLAT_MIRROR), catalog), 633.0) == ((1.0, 0.0), (0.0, 1.0))
    (a, b), (c, d) = C.paraxial_matrix(C.compile_element("C", catalog.get(CONCAVE_MIRROR), catalog), 633.0)
    assert (a, b, d) == (1.0, 0.0, 1.0) and c == pytest.approx(-2.0 / 200.0, rel=1e-15)


def test_cassegrain_reports_the_hand_values_and_states_the_unfold_convention():
    reply = paraxial_analysis(_cassegrain(), {"components": ["M1", "M2"], "wavelength_nm": 633, "source_id": "SRC"})
    out = reply["system"]
    m1, m2 = np.array([[1.0, 0.0], [-1.0 / 400.0, 1.0]]), np.array([[1.0, 0.0], [1.0 / 75.0, 1.0]])
    system = m2 @ free(350.0) @ m1
    assert system[0, 0] == pytest.approx(0.125, rel=1e-12) and system[1, 0] == pytest.approx(-1.0 / 1200.0, rel=1e-12)
    assert np.array(out["abcd"]) == pytest.approx(system, abs=1e-12)
    assert out["efl_mm"] == pytest.approx(1200.0, rel=1e-12) and out["bfl_mm"] == pytest.approx(150.0, rel=1e-12)
    assert out["ffd_mm"] == pytest.approx(-system[1, 1] / system[1, 0], rel=1e-12)
    assert out["vertex_gaps_mm"] == pytest.approx([350.0])
    assert out["rear_focal_point_mm"] == pytest.approx([200.0, 0.0, 0.0], abs=1e-9)
    # the front focal point lies on the incoming ray, upstream of M1 along +X
    assert out["front_focal_point_mm"] == pytest.approx([out["ffd_mm"], 0.0, 0.0], abs=1e-9)
    assert reply["optical_axis"] == [-1.0, 0.0, 0.0]
    assert [s["direction"] for s in reply["axis_path"]] == [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert [(s["start_along_axis_mm"], s["end_along_axis_mm"]) for s in reply["axis_path"]] == [(0.0, 0.0), (0.0, 350.0), (350.0, 350.0)]
    assert [(e["reflector"], e["turned"], e["efl_mm"]) for e in reply["elements"]] == [(True, False, 400.0), (True, False, -75.0)]
    assert reply["elements"][1]["exit_vertex_along_axis_mm"] == 350.0 and reply["elements"][1]["exit_vertex_mm"] == [350.0, 0.0, 0.0]
    assert "unfolded" in reply["conventions"]["unfolded_axis"] and "R / 2" in reply["conventions"]["mirrors"]


def test_cassegrain_image_is_where_the_trace_focuses():
    """A real trace, the plane scanned in 1 mm steps: the smallest ray spot of a 20 mm beam lies within 2 % of the paraxial image at
    x = 200 mm (spherical mirrors: its marginal rays focus elsewhere). A 2 mm beam, at the paraxial limit, is sharpest at the paraxial
    image itself; its Gaussian waist is where the q of a matrix written here puts it."""

    scan = np.arange(196.0, 205.0, 1.0)
    spots = [_focus_pass(_cassegrain(float(x))).pupil_rms_radius_mm for x in scan]
    best = float(scan[int(np.argmin(spots))])
    assert abs(best - 200.0) / 200.0 < 0.02
    assert min(spots) < 0.01  # mm: a focus, not a coincidence of the scan

    small = {x: _focus_pass(_cassegrain(x, 2.0)).pupil_rms_radius_mm for x in (199.5, 200.0, 200.5)}
    assert small[200.0] < 1e-4 and small[200.0] < 0.1 * min(small[199.5], small[200.5])

    lam, w0 = 633.0, 20.0
    system = np.array([[1.0, 0.0], [1.0 / 75.0, 1.0]]) @ free(350.0) @ np.array([[1.0, 0.0], [-1.0 / 400.0, 1.0]])
    q_out = q_through(system, complex(340.0, rayleigh_range(w0, lam)))  # the waist 340 mm before M1
    waist_x = 350.0 + q_out.real  # q_out = z - z_waist along the outgoing -X from x = 350
    gaussian = _focus_pass(_cassegrain(200.0)).beam.gaussian
    assert gaussian.waist_distance_x_mm == pytest.approx(waist_x - 200.0, abs=1e-6)


def test_the_cli_answers_a_cassegrain_and_refuses_a_decentred_member_with_its_reason(tmp_path):
    path = tmp_path / "cassegrain.json"
    path.write_text(json.dumps(_cassegrain()), encoding="utf-8")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "paraxial", str(path), "--components", "M1", "M2", "--wavelength-nm", "633", "--source", "SRC"])
    reply = json.loads(out.getvalue())
    assert code == 0 and reply["system"]["efl_mm"] == pytest.approx(1200.0) and reply["system"]["bfl_mm"] == pytest.approx(150.0)
    assert "unfolded_axis" in reply["conventions"] and len(reply["axis_path"]) == 3

    path.write_text(json.dumps(_triplet_project(offsets={1: (0.5, 0.0)})), encoding="utf-8")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "paraxial", str(path), "--components", "U", "--wavelength-nm", "587.6"])
    problem = json.loads(out.getvalue())["error"]["problems"][0]
    assert code == 65 and problem["code"] == "PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL" and "'E2'" in problem["message"]


def test_a_mirror_system_needs_no_source():
    """No source: light arrives at a curved first mirror along its axis: the same numbers as with the source."""

    with_source = paraxial_analysis(_cassegrain(), {"components": ["M1", "M2"], "wavelength_nm": 633, "source_id": "SRC"})
    without = paraxial_analysis(_cassegrain(), {"components": ["M1", "M2"], "wavelength_nm": 633})
    assert without["system"] == with_source["system"] and without["optical_axis"] == [-1.0, 0.0, 0.0]
    assert without["section_frame"]["reference"] == "component M1"


def test_a_convex_mirror_has_a_virtual_focus_behind_it():
    """f = R/2 = -100 mm at x = 0, light leaving along +X: BFL = -100 mm, the virtual focus at x = -100, behind the mirror."""

    doc = with_models(project([declared_component("M", "mirror", "m", (0.0, 0.0, 0.0), ALONG_X)], [plane("p", (50.0, 0.0, 0.0))]),
                      [_mirror("m", "-200")], coatings=[HR])
    out = paraxial_analysis(doc, {"components": ["M"], "wavelength_nm": 633})["system"]
    assert out["efl_mm"] == -100.0 and out["bfl_mm"] == -100.0
    assert out["rear_focal_point_mm"] == [-100.0, 0.0, 0.0]


def test_afocal_mirror_pair_moves_the_second_mirror_along_its_incoming_direction():
    """A Mersenne pair: afocal at d = f1 + f2 = 400 - 75 = 325 mm, beam magnification -f2/f1 = 75/400 (reduced, upright). The traced
    output at the solved position keeps a 1 mm-waist beam's collimation: divergence = input / |A|."""

    doc = _cassegrain(100.0, 1.0)
    solution = paraxial_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 633, "solve": "afocal"})["solution"]
    assert solution["vertex_gap_mm"] == pytest.approx(325.0, rel=1e-12)
    assert solution["move_along_axis_mm"] == pytest.approx(-25.0, rel=1e-12) and solution["move_direction"] == [1.0, 0.0, 0.0]
    assert solution["position_mm"] == pytest.approx([325.0, 0.0, 0.0], abs=1e-9)
    assert solution["beam_size_magnification"] == pytest.approx(75.0 / 400.0, rel=1e-12)

    moved = copy.deepcopy(doc)
    moved["components"][2]["nominal_pose"]["position_mm"] = solution["position_mm"]
    moved["components"][0]["nominal_pose"]["position_mm"] = [320.0, 0.0, 0.0]
    obs = run_optical_project(moved).result.observations[0]
    third = next(a for a in obs.arrivals if a.beam.direction[0] < 0 and a.crossing_index == 2)
    theta_in = 633.0e-6 / (math.pi * 1.0)
    assert third.beam.gaussian.far_field_half_angle_x_rad == pytest.approx(theta_in / (75.0 / 400.0), rel=1e-6)


# ------------------------------------------------------------------------------------------------ fold mirrors


def _folded(*, source_on: bool = True, l2_position=(30.0, 40.0, 0.0), l2_quaternion=ALONG_Y, mirror_quaternion=FOLD_X_TO_Y) -> dict:
    """Thin lens f1 = 100 at x = 0, a plane mirror at x = 30 folding +X to +Y, thin lens f2 = 50 at (30, 40, 0)."""

    components = [declared_component("L1", "lens", "f100", (0.0, 0.0, 0.0), ALONG_X),
                  element("FM", "mirror", FLAT_MIRROR, (30.0, 0.0, 0.0), mirror_quaternion),
                  declared_component("L2", "lens", "f50", l2_position, l2_quaternion)]
    if source_on:
        components.insert(0, source(position=(-200.0, 0.0, 0.0), waist_mm=2.0, wavelength_nm=633.0))
    doc = project(components, [plane("image", (30.0, 58.75, 0.0), ALONG_Y, radius_mm=20.0)])
    return with_models(doc, [thin_lens_model("f100", "100"), thin_lens_model("f50", "50")])


def test_a_fold_mirror_unfolds_a_lens_pair_to_hand_derived_numbers():
    """Hand derivation with the unfolded gap d = 30 + 40 = 70 mm: phi = 1/100 + 1/50 - 70/5000 = 0.016, EFL = 62.5 mm,
    BFL = (1 - 70/100) / phi = 18.75 mm, FFD = (1 - 70/50) / phi = -25 mm, so the rear focal point is 18.75 mm along +Y from L2:
    (30, 58.75, 0), and the front focal point 25 mm after L1 on the incoming line: (25, 0, 0)."""

    reply = paraxial_analysis(_folded(), {"components": ["L1", "FM", "L2"], "wavelength_nm": 633, "source_id": "SRC"}, catalog=catalog_document())
    out = reply["system"]
    assert out["efl_mm"] == pytest.approx(62.5, rel=1e-12) and out["bfl_mm"] == pytest.approx(18.75, rel=1e-12)
    assert out["ffd_mm"] == pytest.approx(-25.0, rel=1e-12)
    assert out["vertex_gaps_mm"] == pytest.approx([30.0, 40.0], abs=1e-12)
    assert out["rear_focal_point_mm"] == pytest.approx([30.0, 58.75, 0.0], abs=1e-9)
    assert out["front_focal_point_mm"] == pytest.approx([25.0, 0.0, 0.0], abs=1e-9)
    assert out["output_plane_mm"] == pytest.approx([30.0, 40.0, 0.0], abs=1e-9)
    assert [s["direction"] for s in reply["axis_path"]] == [pytest.approx([1.0, 0.0, 0.0]), pytest.approx([0.0, 1.0, 0.0], abs=1e-12)]
    assert reply["axis_path"][1]["start_along_axis_mm"] == pytest.approx(30.0) and reply["axis_path"][1]["end_along_axis_mm"] == pytest.approx(70.0)
    assert reply["axis_path"][1]["section_x"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert reply["axis_path"][1]["section_y"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert reply["elements"][2]["exit_vertex_along_axis_mm"] == pytest.approx(70.0)


def test_the_folded_focus_is_where_the_trace_focuses():
    """The plane at the predicted rear focal point (30, 58.75, 0): a 2 mm collimated beam traced through the real fold mirror is sharpest there."""

    def spot(y: float) -> float:
        doc = _folded()
        doc["observations"][0]["pose"]["position_mm"] = [30.0, y, 0.0]
        return next(o for o in run(doc).result.observations if o.observation_id == "image").arrivals[0].pupil_rms_radius_mm

    assert spot(58.75) < 1e-6
    assert spot(58.25) > 100 * spot(58.75) and spot(59.25) > 100 * spot(58.75)


def test_a_first_plane_mirror_without_a_source_takes_the_incoming_direction_from_the_next_element():
    """Mirror first: the outgoing direction is towards L2 (+Y), so the light arrives along its mirror image, +X. Section axes
    lie across the light."""

    doc = project([element("FM", "mirror", FLAT_MIRROR, (30.0, 0.0, 0.0), FOLD_X_TO_Y), declared_component("L2", "lens", "f50", (30.0, 40.0, 0.0), ALONG_Y)],
                  [plane("image", (30.0, 58.75, 0.0), ALONG_Y)])
    doc = with_models(doc, [thin_lens_model("f50", "50")])
    reply = paraxial_analysis(doc, {"components": ["FM", "L2"], "wavelength_nm": 633}, catalog=catalog_document())
    assert reply["optical_axis"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    frame = reply["section_frame"]
    assert np.dot(frame["x"], reply["optical_axis"]) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(frame["y"], reply["optical_axis"]) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(frame["x"], frame["y"]) == pytest.approx(0.0, abs=1e-12)
    assert reply["system"]["efl_mm"] == pytest.approx(50.0, rel=1e-12)
    assert reply["system"]["rear_focal_point_mm"] == pytest.approx([30.0, 90.0, 0.0], abs=1e-9)


def test_thick_lenses_around_a_fold_equal_the_straight_system():
    """Catalog lenses around a fold: the matrix of the straight system of the same unfolded gap, built from the prescriptions here."""

    lam = 780.0
    doc = project([element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0)), element("FM", "mirror", FLAT_MIRROR, (60.0, 0.0, 0.0), FOLD_X_TO_Y),
                   element("L2", "lens", PCX_F100, (60.0, 80.0, 0.0), ALONG_Y)], [plane("p", (60.0, 300.0, 0.0), ALONG_Y)])
    reply = paraxial_analysis(doc, {"components": ["L1", "FM", "L2"], "wavelength_nm": lam}, catalog=catalog_document())
    expected = lens_matrix(PCX_F100, lam) @ free(80.0) @ free(60.0 - 5.3) @ lens_matrix(PCX_F50, lam)
    assert np.array(reply["system"]["abcd"]) == pytest.approx(expected, abs=1e-12)
    assert reply["system"]["output_plane_mm"] == pytest.approx([60.0, 83.6, 0.0], abs=1e-9)


def test_afocal_past_a_fold_moves_the_second_lens_along_the_folded_axis():
    """A unit of a thin lens f1 = 100 and a fold mirror 30 mm after it, then a thin lens f2 = 50: afocal at f1 + f2 = 150 mm along the
    unfolded axis, so 120 mm from the unit's exit vertex (the mirror), along +Y."""

    solution = paraxial_analysis(project_of_pair(_folded()), {"components": ["FOLDED_L1", "FOLDED_L2"], "wavelength_nm": 633, "solve": "afocal"},
                                 catalog=catalog_document())["solution"]
    assert solution["vertex_gap_mm"] == pytest.approx(120.0, rel=1e-12)
    assert solution["move_direction"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert solution["position_mm"] == pytest.approx([30.0, 120.0, 0.0], abs=1e-9)


def project_of_pair(doc: dict) -> dict:
    """A compound unit holding the first lens and the fold mirror, and the second lens as its own component."""

    out = copy.deepcopy(doc)
    envelope = cylinder("80", "-1", "60")
    mirror = {"model_id": "fm", "model": sequence([surface(1, "0", clear="11", traced="REFLECTED", coating_id=HR["coating_id"])], [], edge="12")}
    unit = {"model_id": "unit", "model": {"model_kind": "COMPOUND", "model_version": "1", "envelope": envelope, "members": [
        {"member_id": "L", "model_id": "f100", "pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(IDENTITY_Q)}},
        {"member_id": "F", "model_id": "fm", "pose": {"position_mm": [0.0, 0.0, 30.0],
                                                     "quaternion_wxyz": list(quat_multiply(_inverse(ALONG_X), FOLD_X_TO_Y))}},
    ]}}
    out["optical_models"] = [*out["optical_models"], mirror, unit]
    out["model_coatings"] = [HR]
    out["components"] = [declared_component("FOLDED_L1", "lens", "unit", (0.0, 0.0, 0.0), ALONG_X),
                         declared_component("FOLDED_L2", "lens", "f50", (30.0, 40.0, 0.0), ALONG_Y)]
    return out


def _inverse(q) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def test_collimate_before_a_fold_moves_the_whole_train_along_the_incoming_axis():
    """A thin lens f = 100 mm at x = 0, a fold mirror at x = 30: a 50 um waist 200 mm before the lens is collimated by moving the lens and the
    mirror together by 100 - 200 = -100 mm along +X. In the run, the far-field divergence past the fold is the predicted one, lambda / (pi w_out)
    with w_out = lambda f / (pi w_in)."""

    lam, w0, f = 633.0, 0.05, 100.0
    doc = project([source(position=(-200.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam), declared_component("L1", "lens", "f100", (0.0, 0.0, 0.0), ALONG_X),
                   element("FM", "mirror", FLAT_MIRROR, (30.0, 0.0, 0.0), FOLD_X_TO_Y)], [plane("far", (-70.0, 3000.0, 0.0), ALONG_Y, radius_mm=40.0)])
    doc = with_models(doc, [thin_lens_model("f100", "100")])
    solution = paraxial_analysis(doc, {"components": ["L1", "FM"], "wavelength_nm": lam, "solve": "collimate", "source_id": "SRC"},
                                 catalog=catalog_document())["solution"]
    assert solution["object_distance_mm"] == pytest.approx(f, rel=1e-12) and solution["move_along_axis_mm"] == pytest.approx(-100.0, rel=1e-12)
    assert solution["positions_mm"] == {"L1": pytest.approx([-100.0, 0.0, 0.0]), "FM": pytest.approx([-70.0, 0.0, 0.0])}
    beam = solution["output_beam"]["x"]
    assert beam["state"] == "COLLIMATED"
    w_out = lam * 1e-6 * f / (math.pi * w0)
    assert beam["waist_radius_mm"] == pytest.approx(w_out, rel=1e-9)
    moved = copy.deepcopy(doc)
    for component in moved["components"][1:]:
        component["nominal_pose"]["position_mm"] = solution["positions_mm"][component["instance_id"]]
    gaussian = run(moved).result.observations[0].arrivals[0].beam.gaussian
    assert gaussian.far_field_half_angle_x_rad == pytest.approx(lam * 1e-6 / (math.pi * w_out), rel=1e-6)
    assert gaussian.far_field_half_angle_y_rad == pytest.approx(lam * 1e-6 / (math.pi * w_out), rel=1e-6)


def test_section_axes_follow_the_fold_for_a_cylinder_after_it():
    """A cylinder curved along local X (declared helper) before and after a fold: section X = design Y becomes design +X past the mirror
    (v - 2 (v . n) n with n = (-1, 1, 0) / sqrt(2)), the local X of the second cylinder, so both have their power in section X, the
    plane of the fold; rolled 90 deg about the folded axis, the second has it in section Y. Numbers: thin-element power of the
    independent matrix through the unfolded gap."""

    n = _nbk7(633.0)
    curved = free(4.0 / n) @ np.array([[1.0, 0.0], [-(n - 1.0) / 51.5, 1.0]])
    plate = free(4.0 / n)
    cyl = {"model_id": "cyl", "model": sequence([surface(1, "0", radius="51.5", cylinder_axis="Y", clear="9"), surface(2, "4", clear="9")], ["N-BK7"], edge="12.7")}

    def doc(quaternion) -> dict:
        d = project([source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, wavelength_nm=633.0), declared_component("C1", "lens", "cyl", (0.0, 0.0, 0.0), ALONG_X),
                     element("FM", "mirror", FLAT_MIRROR, (30.0, 0.0, 0.0), FOLD_X_TO_Y), declared_component("C2", "lens", "cyl", (30.0, 40.0, 0.0), quaternion)],
                    [plane("p", (30.0, 300.0, 0.0), ALONG_Y)])
        return with_models(d, [cyl], [_material()])

    gap = free(30.0 - 4.0 + 40.0)
    same = paraxial_analysis(doc(ALONG_Y), {"components": ["C1", "FM", "C2"], "wavelength_nm": 633, "axis": "X", "source_id": "SRC"}, catalog=catalog_document())
    assert same["elements"][2]["element_section"] == "X"
    assert np.array(same["sections"]["X"]["abcd"]) == pytest.approx(curved @ gap @ curved, abs=1e-12)
    assert np.array(same["sections"]["Y"]["abcd"]) == pytest.approx(plate @ gap @ plate, abs=1e-12)
    rolled = quat_multiply(ALONG_Y, turn_about_z(math.sqrt(0.5), math.sqrt(0.5)))
    crossed = paraxial_analysis(doc(rolled), {"components": ["C1", "FM", "C2"], "wavelength_nm": 633, "axis": "X", "source_id": "SRC"}, catalog=catalog_document())
    assert crossed["elements"][2]["element_section"] == "Y"
    assert np.array(crossed["sections"]["X"]["abcd"]) == pytest.approx(plate @ gap @ curved, abs=1e-12)
    assert np.array(crossed["sections"]["Y"]["abcd"]) == pytest.approx(curved @ gap @ plate, abs=1e-12)


# ------------------------------------------------------------------------------------------------ the light path decides the direction


def _sourceless(document: dict) -> dict:
    out = copy.deepcopy(document)
    out["components"] = [c for c in out["components"] if c["optical_role"] != "source"]
    return out


def _cli(document: dict, tmp_path, *args: str) -> tuple[int, dict]:
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "paraxial", str(path), *args])
    return code, json.loads(out.getvalue())


def test_a_reversed_mirror_list_gets_the_same_verdict_with_and_without_the_source(tmp_path):
    """The Cassegrain's source lights M1 along -X. [M2, M1] would be the pair seen by light travelling +X (EFL 1200, BFL 6800 by hand, the
    compound test above): with the Project's source on the line the verdict is the source's, PARAXIAL_REFLECTOR_BACK_FACE, whether or not the request
    names it; in the CLI too (exit 65 both ways)."""

    doc = _cassegrain()
    for request in ({"components": ["M2", "M1"], "wavelength_nm": 633}, {"components": ["M2", "M1"], "wavelength_nm": 633, "source_id": "SRC"}):
        (problem,) = _codes(doc, request)
        assert problem["code"] == "PARAXIAL_REFLECTOR_BACK_FACE" and problem["location"] == "component M2"
        assert "'SRC'" in problem["message"]
    for args in ([], ["--source", "SRC"]):
        code, reply = _cli(doc, tmp_path, "--components", "M2", "M1", "--wavelength-nm", "633", *args)
        assert code == 65 and reply["error"]["problems"][0]["code"] == "PARAXIAL_REFLECTOR_BACK_FACE"

    forward = paraxial_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 633})
    assert forward["system"]["efl_mm"] == pytest.approx(1200.0, rel=1e-12) and forward["system"]["bfl_mm"] == pytest.approx(150.0, rel=1e-12)
    assert forward["system"]["rear_focal_point_mm"] == pytest.approx([200.0, 0.0, 0.0], abs=1e-9)
    assert forward["light_direction"]["rule"] == "PROJECT_SOURCE" and forward["light_direction"]["source_ids"] == ["SRC"]
    assert forward["light_direction"]["direction"] == [-1.0, 0.0, 0.0]
    named = paraxial_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 633, "source_id": "SRC"})
    assert named["light_direction"]["rule"] == "SOURCE" and named["system"] == forward["system"]


def test_without_a_source_on_the_line_a_reversed_list_is_an_assumed_direction_and_says_so(tmp_path):
    """No source: the curved first mirror is assumed to be met on its reflecting side, so [M2, M1] is the pair met by light travelling +X,
    EFL 1200 and BFL 6800 by hand (M1 T(350) M2: A = 1 + 350/75), and the reply labels the direction as assumed."""

    doc = _sourceless(_cassegrain())
    reverse = paraxial_analysis(doc, {"components": ["M2", "M1"], "wavelength_nm": 633})
    assert reverse["system"]["efl_mm"] == pytest.approx(1200.0, rel=1e-12) and reverse["system"]["bfl_mm"] == pytest.approx(6800.0, rel=1e-12)
    assert reverse["optical_axis"] == [1.0, 0.0, 0.0]
    light = reverse["light_direction"]
    assert light["rule"] == "ASSUMED" and light["source_ids"] == [] and light["direction"] == [1.0, 0.0, 0.0] and "reverse order" in light["meaning"]
    assert "ASSUMED" in reverse["conventions"]["light_direction"]
    code, reply = _cli(doc, tmp_path, "--components", "M2", "M1", "--wavelength-nm", "633")
    assert code == 0 and reply["light_direction"]["rule"] == "ASSUMED"


def test_a_mirror_listed_twice_is_refused_with_or_without_the_source():
    """[M1, M2, M1] would read EFL 2400: a light path of this analysis meets each mirror once; light returning to a mirror is a cavity."""

    for doc in (_cassegrain(), _sourceless(_cassegrain())):
        for extra in ({}, {"source_id": "SRC"}) if doc["components"][0]["optical_role"] == "source" else ({},):
            (problem,) = _codes(doc, {"components": ["M1", "M2", "M1"], "wavelength_nm": 633, **extra})
            assert problem["code"] == "PARAXIAL_REFLECTOR_REPEATED" and problem["location"] == "component M1"
            assert "'M1'" in problem["message"] and "cavity" in problem["message"]
    # a lens is still listed again after a mirror (the double pass)
    assert paraxial_analysis(_double_pass(), {"components": ["L1", "M1", "L1"], "wavelength_nm": 633})["system"]["efl_mm"] == pytest.approx(80.0, rel=1e-12)


def test_sources_on_the_line_that_disagree_are_set_aside_with_a_note_and_the_direction_is_assumed():
    """Two sources light M1 along the line through it from opposite sides. The request names neither, so they do not decide the
    light's direction: the reply is the sourceless Project's (the assumed direction), and light_direction.notes names both. Naming
    one decides it; a source beyond the list that emits away from it lights nothing of it and changes nothing."""

    doc = _cassegrain()
    doc["components"].append(source("SRC2", position=(-50.0, 0.0, 0.0), quaternion=ALONG_X, waist_mm=1.0, wavelength_nm=633.0))
    request = {"components": ["M1", "M2"], "wavelength_nm": 633}
    sourceless = _cassegrain()
    sourceless["components"] = [c for c in sourceless["components"] if c["optical_role"] != "source"]
    plain = paraxial_analysis(sourceless, request)
    assert plain["light_direction"]["rule"] == "ASSUMED" and plain["light_direction"]["notes"] == []
    reply = paraxial_analysis(doc, request)
    light = reply["light_direction"]
    assert light["rule"] == "ASSUMED" and light["source_ids"] == []
    assert light["notes"] == [{"code": "PARAXIAL_LIGHT_DIRECTION_AMBIGUOUS", "source_ids": ["SRC", "SRC2"]}]
    assert {k: v for k, v in reply.items() if k != "light_direction"} == {k: v for k, v in plain.items() if k != "light_direction"}
    assert light["direction"] == plain["light_direction"]["direction"]
    named = paraxial_analysis(doc, {**request, "source_id": "SRC"})
    assert named["system"]["bfl_mm"] == pytest.approx(150.0, rel=1e-12) and named["light_direction"]["notes"] == []
    # a source past M1 on the same line that emits away from it: M1 is behind it, so it is no light of the list
    away = _cassegrain()
    lit = paraxial_analysis(away, request)
    position = [c for c in away["components"] if c["instance_id"] == "SRC"][0]["nominal_pose"]["position_mm"]
    away["components"].append(source("AWAY", position=(position[0] + 500.0, 0.0, 0.0), quaternion=ALONG_X, waist_mm=1.0, wavelength_nm=633.0))
    assert paraxial_analysis(away, request) == lit
    (problem,) = _codes(away, {**request, "source_id": "AWAY"})  # (named: every check and refusal stands)
    assert problem["code"] in ("PARAXIAL_NOT_COAXIAL", "PARAXIAL_ORDER_INVALID", "PARAXIAL_REFLECTOR_BACK_FACE")
    # sources that agree: the light path is decided
    agree = _cassegrain()
    agree["components"].append(source("SRC2", position=(345.0, 0.0, 0.0), quaternion=AGAINST_X, waist_mm=1.0, wavelength_nm=633.0))
    reply = paraxial_analysis(agree, {"components": ["M1", "M2"], "wavelength_nm": 633})
    assert reply["light_direction"]["source_ids"] == ["SRC", "SRC2"] and reply["system"]["bfl_mm"] == pytest.approx(150.0, rel=1e-12)


# ------------------------------------------------------------------------------------------------ mirror refusals


def test_a_mirror_met_on_its_back_is_refused():
    doc = _cassegrain(m2_quaternion=ALONG_X)
    problems = _codes(doc, {"components": ["M1", "M2"], "wavelength_nm": 633})
    assert [p["code"] for p in problems] == ["PARAXIAL_REFLECTOR_BACK_FACE"]
    assert problems[0]["location"] == "component M2" and "'M2'" in problems[0]["message"]


def test_a_tilted_curved_mirror_is_refused():
    doc = project([source(position=(-200.0, 0.0, 0.0), wavelength_nm=633.0), element("CM", "mirror", CONCAVE_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                  [plane("p", (0.0, 100.0, 0.0), ALONG_Y)])
    problems = _codes(doc, {"components": ["CM"], "wavelength_nm": 633, "source_id": "SRC"}, catalog=catalog_document())
    assert [p["code"] for p in problems] == ["PARAXIAL_CURVED_REFLECTOR_TILTED"]
    assert "45 deg" in problems[0]["message"]
    # a stated tilt tolerance, generous or small, does not admit it: a tilted curved mirror has no single 2 x 2 matrix (its
    # tangential and sagittal focal lengths are R cos(45 deg) / 2 = 70.7 mm and R / (2 cos(45 deg)) = 141.4 mm, not R / 2)
    for tilt_rad in (0.8, 5e-6):
        problems = _codes(doc, {"components": ["CM"], "wavelength_nm": 633, "source_id": "SRC", "coaxial_tolerance": {"tilt_rad": tilt_rad}},
                          catalog=catalog_document())
        assert [p["code"] for p in problems] == ["PARAXIAL_CURVED_REFLECTOR_TILTED"], tilt_rad
        assert "45 deg" in problems[0]["message"] and "whatever coaxial_tolerance says" in problems[0]["message"]


def test_an_element_off_the_folded_axis_is_refused():
    problems = _codes(_folded(l2_position=(30.0, 40.0, 3.0)), {"components": ["L1", "FM", "L2"], "wavelength_nm": 633, "source_id": "SRC"}, catalog=catalog_document())
    assert problems[0]["code"] == "PARAXIAL_NOT_COAXIAL" and "'L2' is 3 mm off the optical axis" in problems[0]["message"]
    problems = _codes(_folded(l2_position=(70.0, 0.0, 0.0), l2_quaternion=ALONG_X), {"components": ["L1", "FM", "L2"], "wavelength_nm": 633, "source_id": "SRC"},
                      catalog=catalog_document())
    assert problems[0]["code"] == "PARAXIAL_NOT_COAXIAL"  # a straight-through position past the mirror is not on the folded axis


def test_an_element_behind_the_mirror_on_the_folded_axis_is_refused_as_out_of_order():
    problems = _codes(_folded(l2_position=(30.0, -40.0, 0.0)), {"components": ["L1", "FM", "L2"], "wavelength_nm": 633, "source_id": "SRC"}, catalog=catalog_document())
    assert problems[0]["code"] == "PARAXIAL_ORDER_INVALID" and "'L2' begins 40 mm before 'FM' ends" in problems[0]["message"]


def test_a_reflector_behind_other_surfaces_is_refused_and_a_first_surface_mirror_on_a_substrate_is_used():
    """Second-surface: the light meets glass first. On a substrate: the reflecting face is on the +Z side, the glass behind it plays no part."""

    second_surface = sequence([surface(1, "0", radius="800", traced="REFLECTED", coating_id=HR["coating_id"]), surface(2, "5")], ["N-BK7"])
    substrate = sequence([surface(1, "-5"), surface(2, "0", radius="800", traced="REFLECTED", coating_id=HR["coating_id"])], ["N-BK7"])
    base = project([declared_component("M", "mirror", "m", (0.0, 0.0, 0.0), ALONG_X)], [plane("p", (100.0, 0.0, 0.0))])
    refused = with_models(base, [{"model_id": "m", "model": second_surface}], [_material()], [HR])
    problems = _codes(refused, {"components": ["M"], "wavelength_nm": 633})
    assert [p["code"] for p in problems] == ["PARAXIAL_REFLECTOR_BEHIND_SURFACES"] and "surface S1" in problems[0]["message"]
    used = with_models(base, [{"model_id": "m", "model": substrate}], [_material()], [HR])
    reply = paraxial_analysis(used, {"components": ["M"], "wavelength_nm": 633})
    assert reply["system"]["efl_mm"] == pytest.approx(400.0, rel=1e-12) and reply["elements"][0]["vertex_span_mm"] == 0.0


def test_a_surface_set_stays_refused():
    from catalog3_products import CUBE, catalog3_document

    doc = project([element("C", "splitter", CUBE, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None)], [plane("p", (100.0, 0.0, 0.0))])
    problems = _codes(doc, {"components": ["C"], "wavelength_nm": 633}, catalog=catalog3_document())
    assert [p["code"] for p in problems] == ["PARAXIAL_ELEMENT_UNSUPPORTED"] and "SURFACE_SET" in problems[0]["message"]


# ------------------------------------------------------------------------------------------------ compound units


NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)
#: Three N-BK7 singlets: (radius of the front, radius of the back, thickness, air gap after).
TRIPLET = ((60.0, -80.0, 5.0, 3.0), (-40.0, 45.0, 2.0, 4.0), (70.0, -55.0, 6.0, 0.0))
SOURCE = source(position=(-100.0, 0.0, 0.0), wavelength_nm=587.6)


def _nbk7(wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1 + sum(b * lam2 / (lam2 - c) for b, c in zip(NBK7_B, NBK7_C)))


def _text(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _ynu_matrix(wavelength_nm: float) -> np.ndarray:
    """The triplet's ABCD by two y-nu traces, surface by surface, written here."""

    n = _nbk7(wavelength_nm)
    columns = []
    for y, nu in ((1.0, 0.0), (0.0, 1.0)):
        for index, (front, back, thickness, gap) in enumerate(TRIPLET):
            nu -= y * (n - 1.0) / front
            y += thickness * nu / n
            nu -= y * (1.0 - n) / back
            if index < len(TRIPLET) - 1:
                y += gap * nu
        columns.append((y, nu))
    return np.array([[columns[0][0], columns[1][0]], [columns[0][1], columns[1][1]]])


def _singlet_model(index: int, front: float, back: float, thickness: float) -> dict:
    return {"model_id": f"s{index}", "model": sequence([surface(1, "0", radius=_text(front)), surface(2, _text(thickness), radius=_text(back))], ["N-BK7"])}


def _triplet_project(*, order=(0, 1, 2), offsets: dict[int, tuple] | None = None, turns: dict[int, tuple] | None = None, quaternion=ALONG_X) -> dict:
    offsets, turns = offsets or {}, turns or {}
    z, members = 0.0, []
    for index, (_front, _back, thickness, gap) in enumerate(TRIPLET):
        x, y = offsets.get(index, (0.0, 0.0))
        members.append({"member_id": f"E{index + 1}", "model_id": f"s{index}", "pose": {"position_mm": [x, y, z], "quaternion_wxyz": list(turns.get(index, IDENTITY_Q))}})
        z += thickness + gap
    unit = {"model_id": "unit", "model": {"model_kind": "COMPOUND", "model_version": "1", "envelope": cylinder("40", "-1", "40"), "members": [members[k] for k in order]}}
    models = [_singlet_model(i, *row[:3]) for i, row in enumerate(TRIPLET)]
    doc = project([SOURCE, declared_component("U", "lens", "unit", (10.0, 0.0, 0.0), quaternion)], [plane("p", (200.0, 0.0, 0.0))])
    return with_models(doc, [*models, unit], [_material()])


def _sequence_project(quaternion=ALONG_X) -> dict:
    surfaces, media, z = [], [], 0.0
    for index, (front, back, thickness, gap) in enumerate(TRIPLET):
        surfaces.append(surface(2 * index + 1, _text(z), radius=_text(front)))
        media.append("N-BK7")
        z += thickness
        surfaces.append(surface(2 * index + 2, _text(z), radius=_text(back)))
        if index < len(TRIPLET) - 1:
            media.append("AIR")
            z += gap
    doc = project([SOURCE, declared_component("U", "lens", "seq", (10.0, 0.0, 0.0), quaternion)], [plane("p", (200.0, 0.0, 0.0))])
    return with_models(doc, [{"model_id": "seq", "model": sequence(surfaces, media)}], [_material()])


FIRST_ORDER = ("efl_mm", "bfl_mm", "ffd_mm", "front_principal_plane_mm", "rear_principal_plane_mm")


@pytest.mark.parametrize("quaternion", [ALONG_X, AGAINST_X], ids=["along", "turned"])
def test_a_coaxial_compound_equals_the_equivalent_surface_sequence(quaternion):
    request = {"components": ["U"], "wavelength_nm": 587.6, "source_id": "SRC"}
    unit = paraxial_analysis(_triplet_project(quaternion=quaternion), request)
    single = paraxial_analysis(_sequence_project(quaternion), request)
    for key in FIRST_ORDER:
        assert unit["system"][key] == pytest.approx(single["system"][key], rel=1e-6)
    for key in ("input_plane_mm", "output_plane_mm", "front_focal_point_mm", "rear_focal_point_mm"):
        assert unit["system"][key] == pytest.approx(single["system"][key], abs=1e-6)
    assert unit["elements"][0]["vertex_span_mm"] == pytest.approx(single["elements"][0]["vertex_span_mm"], rel=1e-12)
    assert unit["elements"][0]["turned"] is (quaternion == AGAINST_X)
    assert [m["member_id"] for m in unit["elements"][0]["members"]] == (["E1", "E2", "E3"] if quaternion == ALONG_X else ["E3", "E2", "E1"])


def test_a_compound_equals_an_independent_ynu_trace():
    lam = 587.6
    reply = paraxial_analysis(_triplet_project(), {"components": ["U"], "wavelength_nm": lam})
    assert np.array(reply["system"]["abcd"]) == pytest.approx(_ynu_matrix(lam), abs=1e-12)
    assert np.array(reply["elements"][0]["abcd"]) == pytest.approx(_ynu_matrix(lam), abs=1e-12)


def test_a_compound_takes_its_lenses_in_pose_order_whatever_order_it_declares():
    lam = 587.6
    reply = paraxial_analysis(_triplet_project(order=(2, 0, 1)), {"components": ["U"], "wavelength_nm": lam})
    assert np.array(reply["system"]["abcd"]) == pytest.approx(_ynu_matrix(lam), abs=1e-12)
    assert [m["member_id"] for m in reply["elements"][0]["members"]] == ["E1", "E2", "E3"]


def test_a_member_turned_around_uses_the_reversed_prescription():
    """E2 posed facing back with its origin at z = 10: its glass still occupies z = 8 .. 10, but the light meets its back surface first
    (the second singlet's surfaces reversed, radii negated). E3 keeps its place after the 4 mm gap, at z = 14."""

    lam = 587.6
    n = _nbk7(lam)
    front, back, thickness, gap = TRIPLET[1]
    doc = _triplet_project(turns={1: FACE_BACK})
    unit = next(m for m in doc["optical_models"] if m["model_id"] == "unit")
    unit["model"]["members"][1]["pose"]["position_mm"][2] = 8.0 + thickness
    unit["model"]["members"][2]["pose"]["position_mm"][2] = 8.0 + thickness + gap
    reply = paraxial_analysis(doc, {"components": ["U"], "wavelength_nm": lam})

    def singlet(r1: float, r2: float, t: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [-(1.0 - n) / r2, 1.0]]) @ free(t / n) @ np.array([[1.0, 0.0], [-(n - 1.0) / r1, 1.0]])

    first, third = TRIPLET[0], TRIPLET[2]
    expected = singlet(third[0], third[1], third[2]) @ free(gap) @ singlet(-back, -front, thickness) @ free(first[3]) @ singlet(first[0], first[1], first[2])
    assert np.array(reply["system"]["abcd"]) == pytest.approx(expected, abs=1e-12)
    assert [m["turned"] for m in reply["elements"][0]["members"]] == [False, True, False]


def test_a_decentred_member_is_refused_by_name_and_pose():
    problems = _codes(_triplet_project(offsets={1: (0.5, 0.0)}), {"components": ["U"], "wavelength_nm": 587.6})
    assert [p["code"] for p in problems] == ["PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL"]
    assert problems[0]["location"] == "component U member E2"
    message = problems[0]["message"]
    assert "member 'E2' of 'U'" in message and "position_mm [0.5, 0.0, 8.0]" in message and "quaternion_wxyz [1.0, 0.0, 0.0, 0.0]" in message
    assert "0.5 mm off the optical axis" in message


def test_a_tilted_member_is_refused_by_name_and_pose():
    tilt = (math.cos(math.radians(5.0)), math.sin(math.radians(5.0)), 0.0, 0.0)
    problems = _codes(_triplet_project(turns={2: tilt}), {"components": ["U"], "wavelength_nm": 587.6})
    assert [p["code"] for p in problems] == ["PARAXIAL_COMPOUND_MEMBER_NOT_COAXIAL"]
    assert "member 'E3' of 'U'" in problems[0]["message"] and "tilted by 10 deg" in problems[0]["message"]


def test_overlapping_members_are_refused():
    doc = _triplet_project()
    unit = next(m for m in doc["optical_models"] if m["model_id"] == "unit")
    unit["model"]["members"][1]["pose"]["position_mm"][2] = 3.0  # the second singlet begins inside the first
    problems = _codes(doc, {"components": ["U"], "wavelength_nm": 587.6})
    assert [p["code"] for p in problems] == ["PARAXIAL_ORDER_INVALID"] and "member 'E2' of 'U' begins 2 mm before member 'E1' of 'U' ends" in problems[0]["message"]


def _cassegrain_unit(order=("M1", "M2")) -> dict:
    members = {"M1": {"member_id": "M1", "model_id": "m1", "pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(IDENTITY_Q)}},
               "M2": {"member_id": "M2", "model_id": "m2", "pose": {"position_mm": [0.0, 0.0, 350.0], "quaternion_wxyz": list(FACE_BACK)}}}
    unit = {"model_id": "unit", "model": {"model_kind": "COMPOUND", "model_version": "1", "envelope": cylinder("124", "-1", "351"), "members": [members[k] for k in order]}}
    doc = project([declared_component("TEL", "mirror", "unit", (0.0, 0.0, 0.0), ALONG_X)], [plane("image", (200.0, 0.0, 0.0), AGAINST_X, radius_mm=60.0)])
    return with_models(doc, [_mirror("m1", "800"), _mirror("m2", "-150", "15"), unit], coatings=[HR])


def test_a_compound_with_mirrors_follows_its_declared_order():
    """The Cassegrain as one unit: the hand values, and both stretches of the axis inside it. Declared the other way round, light meets the
    secondary first: M = M1 T(350) M2 with A = 1 + 350/75, so EFL is unchanged (1200) and BFL = A * 1200 = 6800 by hand."""

    reply = paraxial_analysis(_cassegrain_unit(), {"components": ["TEL"], "wavelength_nm": 633})
    out, unit = reply["system"], reply["elements"][0]
    assert out["efl_mm"] == pytest.approx(1200.0, rel=1e-12) and out["bfl_mm"] == pytest.approx(150.0, rel=1e-12)
    assert out["rear_focal_point_mm"] == pytest.approx([200.0, 0.0, 0.0], abs=1e-9)
    assert unit["reflector"] is True and unit["vertex_span_mm"] == pytest.approx(350.0)
    assert [m["member_id"] for m in unit["members"]] == ["M1", "M2"]
    reverse = paraxial_analysis(_cassegrain_unit(("M2", "M1")), {"components": ["TEL"], "wavelength_nm": 633})["system"]
    assert reverse["efl_mm"] == pytest.approx(1200.0, rel=1e-12) and reverse["bfl_mm"] == pytest.approx(6800.0, rel=1e-12)


# ------------------------------------------------------------------------------------------------ the double pass


def _double_pass(mirror_x: float = 50.0, mirror_radius: str | None = "200", *, lens_at: float = 0.0, extra: tuple = (), models: tuple = ()) -> dict:
    """A thin lens f = 100 mm at ``lens_at``, then a mirror on the same axis that sends the light back through it."""

    doc = project([declared_component("L1", "lens", "f100", (lens_at, 0.0, 0.0), ALONG_X), declared_component("M1", "mirror", "m", (mirror_x, 0.0, 0.0), AGAINST_X), *extra],
                  [plane("image", (-300.0, 0.0, 0.0), AGAINST_X, radius_mm=60.0)])
    return with_models(doc, [thin_lens_model("f100", "100"), _mirror("m", mirror_radius), *models], coatings=[HR])


def test_a_lens_met_twice_is_listed_twice_and_gives_the_hand_values():
    """L T M T L with f = 100, d = 50, R = 200: A = -1/4, B = 75, C = -1/80, D = -1/4, so EFL = 80, BFL = -20: 20 mm behind the lens's output
    plane along the outgoing -X, at x = +20. A single pass would read EFL 66.67."""

    reply = paraxial_analysis(_double_pass(), {"components": ["L1", "M1", "L1"], "wavelength_nm": 633})
    out = reply["system"]
    lens, gap, mirror = np.array([[1.0, 0.0], [-1.0 / 100.0, 1.0]]), free(50.0), np.array([[1.0, 0.0], [-2.0 / 200.0, 1.0]])
    system = lens @ gap @ mirror @ gap @ lens
    assert system == pytest.approx(np.array([[-0.25, 75.0], [-0.0125, -0.25]]), abs=1e-12)
    assert np.array(out["abcd"]) == pytest.approx(system, abs=1e-12)
    assert out["efl_mm"] == pytest.approx(80.0, rel=1e-12) and out["bfl_mm"] == pytest.approx(-20.0, rel=1e-12)
    assert out["rear_focal_point_mm"] == pytest.approx([20.0, 0.0, 0.0], abs=1e-9)
    assert out["vertex_gaps_mm"] == pytest.approx([50.0, 50.0])
    assert [e["instance_id"] for e in reply["elements"]] == ["L1", "M1", "L1"] and [e["turned"] for e in reply["elements"]] == [False, False, True]
    assert out["output_plane_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_a_cats_eye_is_afocal_with_a_magnification_of_minus_one():
    """A plane mirror at the lens's focus: L T(f) T(f) L = ((-1, 2 f), (0, -1)), by hand."""

    out = paraxial_analysis(_double_pass(100.0, None), {"components": ["L1", "M1", "L1"], "wavelength_nm": 633})["system"]
    assert out["afocal"] is True and out["efl_mm"] is None
    assert np.array(out["abcd"]) == pytest.approx(np.array([[-1.0, 200.0], [0.0, -1.0]]), abs=1e-12)


def test_a_thick_lens_on_its_way_back_is_used_turned_around():
    """A thick biconvex lens of distinct radii: the second pass is the element's own matrix reversed, ((D, B), (C, A)), its gap measured to the same vertex it left by."""

    biconvex = {"model_id": "thick", "model": sequence([surface(1, "0", radius="100", clear="12"), surface(2, "6", radius="-40", clear="12")], ["N-BK7"], edge="13")}
    doc = project([declared_component("L1", "lens", "thick", (0.0, 0.0, 0.0), ALONG_X), declared_component("M1", "mirror", "m", (56.0, 0.0, 0.0), AGAINST_X)],
                  [plane("image", (-300.0, 0.0, 0.0), AGAINST_X, radius_mm=60.0)])
    doc = with_models(doc, [biconvex, _mirror("m", "200")], [_material()], [HR])
    (a, b), (c, d) = paraxial_analysis(doc, {"components": ["L1"], "wavelength_nm": 633})["system"]["abcd"]
    forward, back = np.array([[a, b], [c, d]]), np.array([[d, b], [c, a]])
    reply = paraxial_analysis(doc, {"components": ["L1", "M1", "L1"], "wavelength_nm": 633})
    gap, mirror = free(50.0), np.array([[1.0, 0.0], [-2.0 / 200.0, 1.0]])
    assert reply["system"]["vertex_gaps_mm"] == pytest.approx([50.0, 50.0])
    assert np.array(reply["system"]["abcd"]) == pytest.approx(back @ gap @ mirror @ gap @ forward, abs=1e-12)


def test_the_double_pass_focus_is_where_the_trace_focuses():
    """The focus in front of the mirror system, traced: a collimated beam through a lens f = 100 mm, a plane mirror 150 mm behind it and back through the lens:
    L T(300) L = ((-2, 300), (1/100, -2)) by hand: EFL = -100, BFL = -A / C = 200 mm from the lens along -X, at x = -200."""

    reply = paraxial_analysis(_double_pass(150.0, None), {"components": ["L1", "M1", "L1"], "wavelength_nm": 633})["system"]
    assert reply["efl_mm"] == pytest.approx(-100.0, rel=1e-12) and reply["bfl_mm"] == pytest.approx(200.0, rel=1e-12)
    assert reply["rear_focal_point_mm"] == pytest.approx([-200.0, 0.0, 0.0], abs=1e-9)

    def spot(x: float) -> float:
        doc = _double_pass(150.0, None, extra=(source(position=(-100.0, 0.0, 0.0), waist_mm=1.5, wavelength_nm=633.0),))
        doc["observations"][0]["pose"]["position_mm"] = [x, 0.0, 0.0]
        arrivals = run_optical_project(doc).result.observations[0].arrivals
        back = [a for a in arrivals if a.beam.direction[0] < 0.0][-1]
        return back.beam.gaussian.beam_radius_x_mm

    assert spot(-200.0) < spot(-190.0) and spot(-200.0) < spot(-210.0)


def test_a_lens_the_returning_light_crosses_must_be_listed_again():
    """The list leaves out the second pass, which would read EFL 66.67 against the true 80: refused, naming the lens and the list to give."""

    (problem,) = _codes(_double_pass(), {"components": ["L1", "M1"], "wavelength_nm": 633})
    assert problem["code"] == "PARAXIAL_ELEMENT_CROSSED_AGAIN" and problem["location"] == "component L1"
    assert "'L1'" in problem["message"] and "['L1', 'M1', 'L1']" in problem["message"]


def test_a_lens_listed_only_on_the_way_back_is_refused_too():
    # L2 lies between L1 and the mirror: the light meets it on both stretches
    extra = (declared_component("L2", "lens", "f100", (25.0, 0.0, 0.0), ALONG_X),)
    (problem,) = _codes(_double_pass(extra=extra), {"components": ["L1", "M1", "L2", "L1"], "wavelength_nm": 633})
    assert problem["code"] == "PARAXIAL_ELEMENT_CROSSED_AGAIN" and "'L2'" in problem["message"] and "['L1', 'L2', 'M1', 'L2', 'L1']" in problem["message"]
    assert paraxial_analysis(_double_pass(extra=extra), {"components": ["L1", "L2", "M1", "L2", "L1"], "wavelength_nm": 633})["system"]["vertex_gaps_mm"] == pytest.approx([25.0, 25.0, 25.0, 25.0])


def test_a_lens_beside_the_return_path_needs_no_second_pass():
    """A fold mirror turns the light away from the lens: a wide lens next to the outgoing stretch is not crossed."""

    doc = project([element("L1", "lens", PCX_F100, (0.0, 0.0, 0.0), ALONG_X), element("FM", "mirror", FLAT_MIRROR, (30.0, 0.0, 0.0), FOLD_X_TO_Y)],
                  [plane("image", (30.0, 60.0, 0.0), ALONG_Y, radius_mm=20.0)])
    reply = paraxial_analysis(doc, {"components": ["L1", "FM"], "wavelength_nm": 633}, catalog=catalog_document())
    assert reply["system"]["efl_mm"] is not None


def test_a_lens_upstream_of_the_first_listed_mirror_is_crossed_on_the_way_in():
    """The source lies before L1, and the list starts at the mirror: the light meets L1 first, so [M1, L1] would read 66.67 against the true 80. The
    Project's source decides the light path whether or not the request names it; with no source on the line the analysis starts at the mirror."""

    doc = _double_pass(extra=(source(position=(-100.0, 0.0, 0.0), waist_mm=1.5, wavelength_nm=633.0),))
    for request in ({"components": ["M1", "L1"], "wavelength_nm": 633, "source_id": "SRC"}, {"components": ["M1", "L1"], "wavelength_nm": 633}):
        (problem,) = _codes(doc, request)
        assert problem["code"] == "PARAXIAL_ELEMENT_CROSSED_AGAIN" and "['L1', 'M1', 'L1']" in problem["message"]
    assert paraxial_analysis(doc, {"components": ["L1", "M1", "L1"], "wavelength_nm": 633, "source_id": "SRC"})["system"]["efl_mm"] == pytest.approx(80.0, rel=1e-12)
    reply = paraxial_analysis(_double_pass(), {"components": ["M1", "L1"], "wavelength_nm": 633})
    assert reply["system"]["efl_mm"] == pytest.approx(200.0 / 3.0, rel=1e-12) and reply["light_direction"]["rule"] == "ASSUMED"


def test_a_source_between_the_lens_and_the_mirror_does_not_cross_the_lens_on_the_way_in():
    doc = _double_pass(extra=(source(position=(25.0, 0.0, 0.0), waist_mm=1.5, wavelength_nm=633.0),))
    assert paraxial_analysis(doc, {"components": ["M1", "L1"], "wavelength_nm": 633, "source_id": "SRC"})["system"]["efl_mm"] == pytest.approx(200.0 / 3.0, rel=1e-12)


def test_a_repeat_needs_a_mirror_between():
    extra = (declared_component("L2", "lens", "f100", (25.0, 0.0, 0.0), ALONG_X),)
    (problem,) = _codes(_double_pass(), {"components": ["L1", "M1", "L1", "L1"], "wavelength_nm": 633})  # (the repeat at the end: EFL 100, gaps [50, 50, 0], would be a power doubled)
    assert problem["code"] == "PARAXIAL_REQUEST_INVALID" and "'L1'" in problem["message"]
    for components in (["L1", "L2", "L1"], ["L1", "L1"], ["L2", "L1", "L1"]):
        (problem,) = _codes(_double_pass(extra=extra), {"components": components, "wavelength_nm": 633})
        assert problem["code"] == "PARAXIAL_REQUEST_INVALID" and "'L1'" in problem["message"] and "mirror" in problem["message"], components


def test_a_mirror_unit_whose_light_returns_through_its_own_lens_is_refused():
    """A unit cannot name a member twice, so its second pass has no statement: refused, naming the member."""

    members = [{"member_id": "L", "model_id": "f100", "pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(IDENTITY_Q)}},
               {"member_id": "M", "model_id": "m", "pose": {"position_mm": [0.0, 0.0, 50.0], "quaternion_wxyz": list(FACE_BACK)}}]
    unit = {"model_id": "unit", "model": {"model_kind": "COMPOUND", "model_version": "1", "envelope": cylinder("124", "-1", "51"), "members": members}}
    doc = with_models(project([declared_component("CAT", "other_optical", "unit", (0.0, 0.0, 0.0), ALONG_X)], [plane("image", (-300.0, 0.0, 0.0), AGAINST_X, radius_mm=60.0)]),
                      [thin_lens_model("f100", "100"), _mirror("m", "200"), unit], coatings=[HR])
    (problem,) = _codes(doc, {"components": ["CAT"], "wavelength_nm": 633})
    assert problem["code"] == "PARAXIAL_ELEMENT_CROSSED_AGAIN" and "member 'L'" in problem["message"] and "separate components" in problem["message"]


def test_a_collimate_solve_of_a_double_pass_lists_each_component_once():
    doc = _double_pass(extra=(source(position=(-100.0, 0.0, 0.0), waist_mm=1.5, wavelength_nm=633.0),))
    solution = paraxial_analysis(doc, {"components": ["L1", "M1", "L1"], "wavelength_nm": 633, "solve": "collimate", "source_id": "SRC"})["solution"]
    assert solution["components"] == ["L1", "M1"] and sorted(solution["positions_mm"]) == ["L1", "M1"]
    assert solution["object_distance_mm"] == pytest.approx(-20.0, rel=1e-12)  # -D / C, by hand above


def test_an_imported_prescription_is_analysed_end_to_end():
    """The refused case: a Cooke triplet from import-prescription, as the imported COMPOUND. EFL and BFD against a y-nu trace written here,
    with the fixture's radii and thicknesses and the glass data of the fixture (XSK16, F2)."""

    glass = {"XSK16": ((1.34317774, 0.241144399, 0.994317969), (0.00704687339, 0.0229005, 92.7508526)),
             "F2": ((1.34533359, 0.209073176, 0.937357162), (0.00997743871, 0.0470450767, 111.886764))}
    rows = [(22.01359, 3.25896, "XSK16"), (-435.7604, 6.00756, None), (-22.21328, 0.99997, "F2"), (20.29192, 2.0, None),
            (None, 2.75, None), (79.6836, 2.95208, "XSK16"), (-18.39533, None, None)]
    lam = 587.5618

    def n(medium) -> float:
        if medium is None:
            return 1.0
        b, c = glass[medium]
        return math.sqrt(1.0 + sum(bi * (lam / 1000.0) ** 2 / ((lam / 1000.0) ** 2 - ci) for bi, ci in zip(b, c)))

    y, nu, before = 1.0, 0.0, None
    for index, (radius, _t, after) in enumerate(rows):
        if index:
            y += rows[index - 1][1] * nu / n(rows[index - 1][2])
        if radius is not None:
            nu -= y * (n(after) - n(before)) / radius
        before = after
    efl, bfd = -1.0 / nu, -y / nu

    catalog = load_public_catalog(REAL_CATALOG.read_bytes())
    imported = import_prescription((FIXTURES / "cooke_triplet.csv").read_text(encoding="utf-8"), model_id="cooke", catalog=catalog)
    base = project([], [plane("image", (0.0, 0.0, 100.0), IDENTITY_Q, radius_mm=30.0)])
    document = merge_into_project(base, imported, instance_id="COOKE", pose={"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(IDENTITY_Q)})
    answer = json_api.paraxial_analysis(document.payload(), {"components": ["COOKE"], "wavelength_nm": lam}, catalog=REAL_CATALOG.read_bytes())
    assert answer["ok"] is True
    assert answer["system"]["efl_mm"] == pytest.approx(efl, rel=1e-9) and answer["system"]["bfl_mm"] == pytest.approx(bfd, rel=1e-9)
    assert [m["member_id"] for m in answer["elements"][0]["members"]] == ["L1", "L2", "STOP", "L3"]
