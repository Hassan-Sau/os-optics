"""Coatings a prescription defines itself (``[coatings]`` of csv-surface-table/1, ``ir.CoatingDefinition``): spectra (with
s / p), angle-resolved tables and stated phases become the same ``PublicCoatingV1`` a Project would declare by hand, the
imported unit simulates exactly like that hand-declared Project, identical coatings are shared when merged and different
ones refused, and incomplete or impossible definitions are refused with located ``COATING_INVALID`` problems.

Fixture data: every coating value and phase below is made up for the test.
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal

import pytest

from os_optics.local import lower_project, run_optical_project
from os_optics.prescription import PrescriptionErrorCode as E, PrescriptionImportError, merge_into_project
from os_optics._vendor.shared_contracts.design_catalog import PublicCoatingV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2

from test_prescription_import import SEED, base_project, imported

LENS = """#! wavelength_min_nm = 500
#! wavelength_max_nm = 800
surface,radius,thickness,material,semi_diameter,coating
1,50,5,N-BK7,10,TABLE_AR
2,-50,10,,10,SP_AR
"""
COATINGS = """[coatings]
name,kind,wavelength_nm,aoi_deg,r_s,r_p,t_s,t_p,reflection_phase_s,reflection_phase_p,transmission_phase_s,transmission_phase_p,basis,min_aoi_deg,max_aoi_deg
TABLE_AR,ANTI_REFLECTION,400,0,0.01,0.01,0.99,0.99,,,0.1,0.3,Fixture table,,
table_ar,,400,30,0.02,0.005,0.97,0.994,,,0.1,0.3,,,
TABLE_AR,,1100,0,0.012,0.012,0.988,0.988,,,0.2,0.5,,,
TABLE_AR,,1100,30,0.022,0.006,0.968,0.993,,,0.2,0.5,,,
SP_AR,ANTI_REFLECTION,400,,0.02,0.01,0.97,0.98,1.5,1.2,0,0,,0,10
SP_AR,,1100,,0.03,0.02,0.96,0.97,1.5,1.2,0,0,,,
"""
DOCUMENT = LENS + COATINGS


def _coatings(result) -> dict[str, dict]:
    return {c.coating_id: c.model_dump(mode="json", exclude_none=True) for c in result.model_coatings}


def test_defined_coatings_become_the_coatings_a_project_declares():
    result = imported(DOCUMENT)
    coatings = _coatings(result)
    table = coatings["TABLE_AR"]
    assert table["angular_spectrum"] == {
        "wavelength_nm": ["400", "1100"], "aoi_deg": ["0", "30"], "reflectance_s": [["0.01", "0.02"], ["0.012", "0.022"]],
        "reflectance_p": [["0.01", "0.005"], ["0.012", "0.006"]], "transmittance_s": [["0.99", "0.97"], ["0.988", "0.968"]],
        "transmittance_p": [["0.99", "0.994"], ["0.988", "0.993"]], "interpolation": "BILINEAR", "aoi_axis": "INCIDENT_MEDIUM",
        "status": "DECLARED"}
    assert table["phase"] == {"wavelength_nm": ["400", "1100"], "transmission_s_rad": ["0.1", "0.2"], "transmission_p_rad": ["0.3", "0.5"],
                              "convention": "S_P_TRIAD_EXP_MINUS_I_OMEGA_T", "status": "DECLARED"}
    assert (table["aoi_reference"], table["valid_aoi_deg"]["min_deg"], table["valid_aoi_deg"]["max_deg"]) == ("INCIDENT_MEDIUM", "0", "30")
    assert table["basis"] == "Fixture table"
    spectrum = coatings["SP_AR"]
    assert spectrum["spectrum"]["reflectance"] == ["0.015", "0.025"] and spectrum["spectrum"]["transmittance"] == ["0.975", "0.965"]
    assert spectrum["spectrum"]["reflectance_s"] == ["0.02", "0.03"] and spectrum["aoi_reference"] == "AIR_EQUIVALENT"
    # a phase with one value on every row is constant
    assert spectrum["phase"] == {"reflection_s_rad": "1.5", "reflection_p_rad": "1.2", "transmission_s_rad": "0",
                                 "transmission_p_rad": "0", "convention": "S_P_TRIAD_EXP_MINUS_I_OMEGA_T", "status": "DECLARED"}
    lens = next(m.model for m in result.optical_models if m.model_id == "prescription-L1")
    assert "COATING_PHASE_NOT_MODELLED" not in [a.value for a in lens.approximations]  # every coating states its phases
    assert result.report["counts"]["coatings"] == 2


def _hand_declared(result) -> dict:
    """The same unit written by hand: the imported models and coatings, declared directly in a Project."""

    document = base_project()
    document["components"].append({"instance_id": "OBJ", "optical_role": "lens", "custom_part": {"custom_part_key": "obj", "description": "Lens."},
                                   "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                                   "optical_model_id": result.model_id})
    document["optical_models"] = [m.model_dump(mode="json", exclude_none=True) for m in result.optical_models]
    document["model_materials"] = [m.model_dump(mode="json", exclude_none=True) for m in result.model_materials]
    document["model_coatings"] = [json.loads(json.dumps(c)) for c in _coatings(result).values()]
    return document


def test_an_imported_unit_with_defined_coatings_simulates_like_the_hand_declared_project():
    result = imported(DOCUMENT)
    merged = merge_into_project(base_project(), result, instance_id="OBJ", pose={"position_mm": [0.0, 0.0, 0.0],
                                                                                 "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    by_import, by_hand = lower_project(merged), lower_project(_hand_declared(result))
    assert by_import.engine_input_hash == by_hand.engine_input_hash
    request = by_import.request.model_dump_json()
    assert '"phases"' in request and '"aoi_deg":[0.0,30.0,90.0]' in request  # the table and the phases reach the Engine
    run = run_optical_project(merged)
    hand_run = run_optical_project(_hand_declared(result))
    assert run.model_scope_issues == hand_run.model_scope_issues == ()
    # The imported N-BK7 states no internal loss. The coating table and phases still trace geometrically, but neither
    # spelling of the same Engine input may claim measured transmission from absent bulk-loss evidence.
    assert run.status.value == hand_run.status.value == "UNKNOWN"
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
    arrivals = [a for o in run.result.observations for a in o.arrivals]
    hand_arrivals = [a for o in hand_run.result.observations for a in o.arrivals]
    assert [(a.beam.path_id, a.beam.optical_path_length_mm, a.beam.power_W, a.power_upper_bound_W) for a in arrivals] == [
        (a.beam.path_id, a.beam.optical_path_length_mm, a.beam.power_W, a.power_upper_bound_W) for a in hand_arrivals]
    assert any(a.beam.power_W is None and a.power_upper_bound_W > 0 for a in arrivals)


def test_identical_coatings_are_shared_and_different_ones_refused_on_merge():
    result = imported(DOCUMENT)
    once = merge_into_project(base_project(), result, instance_id="A")
    second = imported(DOCUMENT, model_id="other")
    twice = merge_into_project(once, second, instance_id="B", pose={"position_mm": [0.0, 30.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    assert sorted(c.coating_id for c in twice.model_coatings) == ["SP_AR", "TABLE_AR"]  # one definition each, shared
    changed = imported(DOCUMENT.replace("1100,,0.03,0.02,0.96,0.97", "1100,,0.03,0.02,0.95,0.97"), model_id="third")
    with pytest.raises(PrescriptionImportError) as info:
        merge_into_project(once, changed, instance_id="C")
    assert [p.code for p in info.value.problems] == [E.PROJECT_INVALID] and "SP_AR" in info.value.problems[0].message


def _problems(doc: str) -> list[dict]:
    with pytest.raises(PrescriptionImportError) as info:
        imported(doc)
    return [p.as_dict() for p in info.value.problems]


@pytest.mark.parametrize("case, doc, words", [
    ("grid-point-missing", DOCUMENT.replace("TABLE_AR,,1100,30,0.022,0.006,0.968,0.993,,,0.2,0.5,,,\n", ""), "missing 1100 nm at 30 deg"),
    ("r-plus-t-above-1", DOCUMENT.replace("TABLE_AR,,1100,0,0.012,0.012,0.988", "TABLE_AR,,1100,0,0.12,0.012,0.988"), "cannot exceed 1"),
    ("phase-differs-between-angles", DOCUMENT.replace(",0.1,0.3,,,\n", ",0.15,0.3,,,\n", 1), "differs between angles"),
    ("spectrum-without-angles", DOCUMENT.replace(",,0,10\n", ",,,\n"), "states the angles of incidence it holds for"),
    ("mixed-angles", DOCUMENT.replace("SP_AR,,1100,,", "SP_AR,,1100,5,"), "on every row (an angle-resolved table) or on none"),
    ("value-missing", DOCUMENT.replace("SP_AR,,1100,,0.03,0.02,0.96,", "SP_AR,,1100,,0.03,,0.96,"), "on every row"),
    ("attribute-twice", DOCUMENT.replace("TABLE_AR,,1100,0,", "TABLE_AR,BEAMSPLITTER,1100,0,"), "state it once"),
], ids=lambda value: value if isinstance(value, str) and "\n" not in value and len(value) < 40 else "")
def test_incomplete_or_impossible_coating_definitions_are_refused(case, doc, words):
    problems = _problems(doc)
    assert {p["code"] for p in problems} == {E.COATING_INVALID.value}, problems
    assert any(words in p["message"] for p in problems), problems
    assert all(p["location"].get("section") == "coatings" for p in problems)


def test_a_defined_coating_no_surface_uses_is_reported_and_not_imported():
    doc = DOCUMENT + "SPARE,ANTI_REFLECTION,400,,0.01,0.01,0.99,0.99,,,,,,0,5\nSPARE,,1100,,0.01,0.01,0.99,0.99,,,,,,,\n"
    result = imported(doc)
    assert "SPARE" not in _coatings(result)
    assert [w["code"] for w in result.report["warnings"]] == ["COATING_UNUSED"]


def test_a_defined_coating_takes_precedence_over_a_published_id_and_bands_bound_the_models():
    """A [coatings] definition named like a published coating is the file's own; the imported models' derived band lies
    inside every coating span and every tabulated phase span (here 400-1100 nm, so the file's 500-800 nm band stands)."""

    published = SEED.coatings[0].coating_id
    doc = DOCUMENT.replace("TABLE_AR", published).replace("table_ar", published.lower())
    result = imported(doc)
    assert _coatings(result)[published]["basis"] == "Fixture table"
    narrow = imported(DOCUMENT.replace("#! wavelength_min_nm = 500\n#! wavelength_max_nm = 800\n", ""))
    band = narrow.report["wavelength_band_nm"]
    assert (Decimal(band["min_nm"]), Decimal(band["max_nm"])) == (Decimal(400), Decimal(1100))


def test_coatings_a_project_declares_by_hand_round_trip_through_the_project_contract():
    result = imported(DOCUMENT)
    document = _hand_declared(result)
    project = ProjectDocumentV2.model_validate(copy.deepcopy(document))
    assert [PublicCoatingV1.model_validate(c) for c in document["model_coatings"]] == list(project.model_coatings)
