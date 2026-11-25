"""Prescription import resolves glass names with the shared resolver of the public contracts, and a Project keeps the
complete copy (assignment section 2).

On a synthetic catalog-schema-3.6 release (``material_catalog36``): ``D-K9-25`` and ``S-FPL53`` import verbatim (their
complete optical definition and ``copied_from``); a bare ``F5`` two makers publish is ambiguous and names both; a
manufacturer-qualified ``F5`` and the explicit id ``SCHOTT_F5`` are that record; bare ``F2`` -- a released stable id
that CDGM's F2 shares as a designation -- names both candidates, CDGM's F2 in a CDGM-first prescription
(``glass_catalogs``), and SCHOTT's unchanged stable id only with ``ID:F2`` or an explicit maker/order. The imported
Project, saved as JSON, reopens and is checked and analysed with no catalog on the device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import material_catalog36 as M
from os_optics.catalog_copy import copy_catalog_material, material_conflict, physics_difference
from os_optics.local import LocalRunError, json_api, load_public_catalog
from os_optics._vendor.shared_contracts.design_catalog.material_physics import MaterialNameIndex
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.prescription import PrescriptionImportError, import_prescription, merge_into_project

CATALOG = M.catalog()


def lens(glass: str, *, directives: str = "", second: str = "", maker_column: str | None = None) -> str:
    """A cemented doublet whose first glass is ``glass`` (and second ``second``, default the first)."""

    header = "surface,radius,thickness,material,semi_diameter" + (",glass_manufacturer" if maker_column is not None else "")
    extra = f",{maker_column}" if maker_column is not None else ""
    blank = "," if maker_column is not None else ""
    second = second or glass
    return (directives + header + "\n" + f"1,60,6,{glass},10{extra}\n" + f"2,-40,3,{second},10{blank}\n" + f"3,-120,90,,10{blank}\n")


def imported(text: str, **kwargs):
    kwargs.setdefault("catalog", CATALOG)
    return import_prescription(text, **kwargs)


def problems(text: str, **kwargs) -> list[dict]:
    with pytest.raises(PrescriptionImportError) as exc:
        imported(text, **kwargs)
    return [p.as_dict() for p in exc.value.problems]


def glasses(result) -> dict[str, dict]:
    return {g["written"]: g for g in result.report["glasses"]}


def warning_codes(result) -> list[str]:
    return [w["code"] for w in result.report["warnings"]]


def published(material_id: str) -> dict:
    return CATALOG.material(material_id).model_dump(mode="json", by_alias=True, exclude_none=True)


def copy_of(result, material_id: str) -> dict:
    return next(m for m in result.model_materials if m.material_id == material_id).model_dump(mode="json", by_alias=True, exclude_none=True)


# ------------------------------------------------------------------------------------------- the resolver


def test_the_engine_uses_the_shared_resolver_of_the_contracts():
    index = MaterialNameIndex(CATALOG.materials)
    assert index.resolve("F5").candidates == ("CDGM_F5", "SCHOTT_F5")
    shared = index.resolve("F2")
    assert shared.ambiguous and shared.resolved is None and shared.candidates == ("CDGM_F2", "F2")
    assert shared.also_matches == ()
    assert index.resolve("F2", catalogue_order=["CDGM"]).resolved == "CDGM_F2"
    assert index.resolve("F2", manufacturer="SCHOTT").resolved == "F2"
    assert index.resolve("ID:F2").resolved == "F2"


@pytest.mark.parametrize("glass, material_id", [("D-K9-25", "CDGM_D-K9-25"), ("S-FPL53", "OHARA_S-FPL53")])
def test_a_designation_imports_its_record_verbatim(glass, material_id):
    result = imported(lens(glass))
    assert [m.material_id for m in result.model_materials] == [material_id]
    assert glasses(result)[glass]["how"] == "designation" and glasses(result)[glass]["material_id"] == material_id
    copy, source = copy_of(result, material_id), published(material_id)
    for field in ("dispersion", "valid_band_nm", "index_reference", "bulk_loss", "thermo_optic", "display_name"):
        assert copy.get(field) == source.get(field), field
    assert copy["copied_from"] == {"catalog_version": M.CATALOG_VERSION, "material_id": material_id,
                                   "source_edition": source["identity"]["source_edition"]}
    assert "identity" not in copy and copy["basis"].endswith(f"copied from public optical catalog {M.CATALOG_VERSION}")
    assert result.model_materials[0] == copy_catalog_material(CATALOG.material(material_id), M.CATALOG_VERSION)
    assert result.report["warnings"] == []


def test_the_physics_the_copies_carry():
    """The anneal variant keeps its thermo-optic data and its own law (not the base record's); S-FPL53 keeps its air
    convention (with the vacuum band) and its measured internal transmittance."""

    variant = copy_of(imported(lens("D-K9-25")), "CDGM_D-K9-25")
    assert variant["thermo_optic"]["kind"] == "SCHOTT_DN_DT" and variant["dispersion"]["b"][0] == "1.0396"
    fpl53 = copy_of(imported(lens("S-FPL53")), "OHARA_S-FPL53")
    assert fpl53["index_reference"]["wavelength"] == "AIR" and fpl53["index_reference"]["index"] == "RELATIVE_TO_AIR"
    assert (fpl53["valid_band_nm"]["min_nm"], fpl53["valid_band_nm"]["max_nm"]) == M.FPL53_BAND
    assert fpl53["bulk_loss"]["kind"] == "INTERNAL_TRANSMITTANCE" and fpl53["bulk_loss"]["thickness_mm"] == ["10", "10", "10"]


def test_a_bare_designation_two_makers_publish_is_ambiguous_and_names_both():
    (problem,) = problems(lens("F5", second="N-BK7"))
    assert problem["code"] == "GLASS_AMBIGUOUS"
    assert "CDGM_F5 (CDGM F5)" in problem["message"] and "SCHOTT_F5 (SCHOTT F5)" in problem["message"]
    assert "MAKER:DESIGNATION" in problem["message"] and "glass_catalogs" in problem["message"]
    assert problem["location"] == {"row": 2, "column": "material", "surface": "1", "section": "surfaces"}


@pytest.mark.parametrize("glass, maker, material_id", [
    ("SCHOTT:F5", None, "SCHOTT_F5"), ("CDGM:F5", None, "CDGM_F5"), ("cdgm : f5", None, "CDGM_F5"),
    ("F5", "SCHOTT", "SCHOTT_F5"), ("F5", "CDGM", "CDGM_F5"), ("CDGM:F5", "cdgm", "CDGM_F5")])
def test_a_manufacturer_qualified_designation_is_that_makers_record(glass, maker, material_id):
    result = imported(lens(glass, maker_column=maker, second="N-BK7"))
    assert {m.material_id for m in result.model_materials} == {material_id, "N-BK7"}
    record = next(g for g in result.report["glasses"] if g["material_id"] == material_id)
    assert record["how"] == "manufacturer designation" and record["designation"] == "F5"
    assert next(m.model.media for m in result.optical_models if m.model_id == "prescription-L1")[0] == material_id


def test_manufacturer_syntax_problems():
    (conflict,) = problems(lens("SCHOTT:F5", maker_column="CDGM", second="N-BK7"))
    assert conflict["code"] == "VALUE_INVALID" and "state the maker once" in conflict["message"]
    (unknown,) = problems(lens("HIKARI:F5", second="N-BK7"))
    assert unknown["code"] == "GLASS_UNKNOWN" and "no material of manufacturer or catalogue 'HIKARI'" in unknown["message"]
    assert "CDGM" in unknown["message"] and "SCHOTT" in unknown["message"]
    (missing,) = problems(lens("OHARA:F5", second="N-BK7"))
    assert missing["code"] == "GLASS_UNKNOWN" and "closest published" in missing["message"]
    (bad,) = problems(lens(":F5", second="N-BK7"))
    assert bad["code"] == "VALUE_INVALID"
    air_maker = "surface,radius,thickness,material,semi_diameter,glass_manufacturer\n1,60,6,N-BK7,10,\n2,-40,90,,10,SCHOTT\n"
    assert [p["code"] for p in problems(air_maker)] == ["VALUE_INVALID"]


def test_an_explicit_stable_id_is_authoritative():
    result = imported(lens("SCHOTT_F5"))
    assert [m.material_id for m in result.model_materials] == ["SCHOTT_F5"]
    assert glasses(result)["SCHOTT_F5"]["how"] == "material_id" and result.report["warnings"] == []
    assert [m.material_id for m in imported(lens("ID:CDGM_F5")).model_materials] == ["CDGM_F5"]
    (unknown,) = problems(lens("ID:F5", second="N-BK7"))  # ID: never looks up a designation
    assert unknown["code"] == "GLASS_UNKNOWN" and "stable id F5" in unknown["message"]


def test_a_released_id_shared_as_a_designation_and_the_catalogue_order():
    (bare,) = problems(lens("F2", second="N-BK7"))
    assert bare["code"] == "GLASS_AMBIGUOUS"
    assert "CDGM_F2 (CDGM F2)" in bare["message"] and "F2 (SCHOTT F2)" in bare["message"]
    assert "ID:<material id>" in bare["message"]

    cdgm_first = imported(lens("F2", directives="#! glass_catalogs = CDGM, SCHOTT\n"))
    assert [m.material_id for m in cdgm_first.model_materials] == ["CDGM_F2"]
    record = glasses(cdgm_first)["F2"]
    assert record["how"] == "catalogue order" and record["candidates"] == ["CDGM_F2", "F2"] and record["catalogue_order"] == ["CDGM", "SCHOTT"]
    assert cdgm_first.report["warnings"] == []
    schott_first = imported(lens("F2", directives="#! gcat = SCHOTT CDGM\n"))  # the Zemax GCAT spelling
    assert [m.material_id for m in schott_first.model_materials] == ["F2"] and warning_codes(schott_first) == []
    schott_maker = imported(lens("SCHOTT:F2"))
    assert [m.material_id for m in schott_maker.model_materials] == ["F2"] and warning_codes(schott_maker) == []
    assert [m.material_id for m in imported(lens("F5", directives="#! glass_catalogs = CDGM SCHOTT\n")).model_materials] == ["CDGM_F5"]
    # an order naming no candidate's catalogue does not decide; ID: is never re-read through the order
    assert problems(lens("F5", directives="#! glass_catalogs = OHARA\n"))[0]["code"] == "GLASS_AMBIGUOUS"
    assert problems(lens("F2", directives="#! glass_catalogs = OHARA\n"))[0]["code"] == "GLASS_AMBIGUOUS"
    explicit = imported(lens("ID:F2", directives="#! glass_catalogs = CDGM\n"))
    assert [m.material_id for m in explicit.model_materials] == ["F2"] and warning_codes(explicit) == []
    assert glasses(explicit)["ID:F2"]["how"] == "material_id (explicit)"
    assert problems(lens("F2", directives="#! glass_catalogs = CDGM *\n"))[0]["code"] == "DIRECTIVE_INVALID"


def test_a_glass_resolved_outside_the_stated_catalogues_is_a_reported_substitution():
    """Zemax MISC PMMA: the unique designation resolves to a literature model no listed catalogue holds -- reported and
    marked; inside the order, without an order, or qualified by maker: no such warning."""

    misc = imported(lens("PMMA", second="N-BK7", directives="#! glass_catalogs = MISC\n"))
    assert glasses(misc)["PMMA"]["material_id"] == "PMMA_SULTANOVA2009" and glasses(misc)["PMMA"]["outside_catalogue_order"] is True
    (warning,) = [w for w in misc.report["warnings"] if w["code"] == "GLASS_OUTSIDE_CATALOGUES"]
    assert "catalogues are MISC" in warning["message"] and "PMMA_SULTANOVA2009" in warning["message"]
    assert "Sultanova" in warning["message"] and "substitution" in warning["message"]
    assert "N-BK7" not in glasses(misc) or "outside_catalogue_order" not in glasses(misc)["N-BK7"]  # no identity: not checked
    unordered = imported(lens("PMMA", second="N-BK7"))
    assert "outside_catalogue_order" not in glasses(unordered)["PMMA"] and "GLASS_OUTSIDE_CATALOGUES" not in warning_codes(unordered)
    inside = imported(lens("S-FPL53", directives="#! glass_catalogs = OHARA\n"))
    assert glasses(inside)["S-FPL53"]["outside_catalogue_order"] is False and inside.report["warnings"] == []
    outside = imported(lens("S-FPL53", directives="#! glass_catalogs = SCHOTT CDGM\n"))
    assert warning_codes(outside) == ["GLASS_OUTSIDE_CATALOGUES"] and "(OHARA)" in outside.report["warnings"][0]["message"]
    qualified = imported(lens("OHARA:S-FPL53", directives="#! glass_catalogs = SCHOTT\n"))
    assert qualified.report["warnings"] == []


def test_a_loose_spelling_resolves_only_when_unique_and_is_reported():
    result = imported(lens("dk9 25"))
    assert [m.material_id for m in result.model_materials] == ["CDGM_D-K9-25"]
    assert glasses(result)["dk9 25"]["how"] == "loose match" and warning_codes(result) == ["GLASS_NAME_MATCHED"]
    assert [m.material_id for m in imported(lens("sfpl53")).model_materials] == ["OHARA_S-FPL53"]  # its documented alias


def test_a_catalog_without_identities_resolves_by_id_as_before():
    fixture = load_public_catalog()  # schema 3.0, no identities
    result = import_prescription(lens("N-BK7"), catalog=fixture)
    assert [m.material_id for m in result.model_materials] == ["N-BK7"]
    assert result.model_materials[0].copied_from is None and glasses(result)["N-BK7"]["how"] == "material_id"
    loose = import_prescription(lens("nbk7"), catalog=fixture)
    assert warning_codes(loose) == ["GLASS_NAME_MATCHED"]


def test_a_defined_material_and_a_published_one_of_the_same_id_are_refused():
    text = lens("SCHOTT:F5", second="SCHOTT_F5") + "[materials]\nname,index,min_nm,max_nm\nSCHOTT_F5,1.6,400,700\n"
    assert {p["code"] for p in problems(text)} == {"MATERIAL_INVALID"}


# ---------------------------------------------------------------------------------- the Project copy


def test_physics_difference_compares_the_new_physics_and_not_the_provenance():
    material = CATALOG.material("OHARA_S-FPL53")
    copy = copy_catalog_material(material, M.CATALOG_VERSION)
    other_release = copy_catalog_material(material, "2027-01-01.1")
    assert copy.copied_from != other_release.copied_from and physics_difference(copy, other_release) is None
    data = copy.model_dump(mode="json", exclude_none=True)
    lossless = {k: v for k, v in data.items() if k != "bulk_loss"}
    assert physics_difference(lossless, copy) == "the spectral bulk loss (not stated vs stated)"
    lossier = {**data, "bulk_loss": {**data["bulk_loss"], "internal_transmittance": ["0.98", "0.998", "0.997"]}}
    assert physics_difference(lossier, copy) == "the spectral bulk loss"
    reworded = {**data, "bulk_loss": {**data["bulk_loss"], "basis": "Another text."},
                "index_reference": {**data["index_reference"], "basis": "Elsewhere.", "stated_by": "PUBLISHER"}}
    assert physics_difference(reworded, copy) is None
    vacuum_law = {**data, "index_reference": None}
    assert "the index convention (index_reference) (not stated vs stated)" in physics_difference(vacuum_law, copy)
    variant = copy_catalog_material(CATALOG.material("CDGM_D-K9-25"), M.CATALOG_VERSION).model_dump(mode="json", exclude_none=True)
    warmer = {**variant, "thermo_optic": {**variant["thermo_optic"], "d0": "0.000002"}}
    assert physics_difference(warmer, variant) == "the thermo-optic data"
    assert "different physics" in material_conflict(warmer, variant)


def base_project() -> dict:
    """A source and an image plane (no optics; ``test_prescription_import.base_project``)."""

    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Test laser."},
        "nominal_pose": {"position_mm": [0.0, 0.0, -50.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 1.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "image", "pose": {"position_mm": [0.0, 0.0, 81.28], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                          "aperture": {"shape": "circle", "radius_mm": 10.0}}]}


def _project_with(result) -> ProjectDocumentV2:
    return merge_into_project(base_project(), result, instance_id="RX")


def test_the_saved_project_reopens_and_runs_with_no_catalog(tmp_path: Path, empty_catalog_store):
    """Import -> save the Project as JSON -> on a device with no catalog: the Project loads, is checked and analysed, and
    its materials are the copies chosen at import (no second name lookup: the catalog is not there to look in)."""

    result = imported(lens("D-K9-25", second="SCHOTT:F5", directives="#! name = offline doublet\n"), model_id="rx")
    project = _project_with(result)
    saved = tmp_path / "project.json"
    saved.write_text(json.dumps(project.payload(), indent=2), encoding="utf-8")

    with pytest.raises(LocalRunError):
        load_public_catalog()  # nothing on this device
    reopened = json.loads(saved.read_text(encoding="utf-8"))
    by_id = {m["material_id"]: m for m in reopened["model_materials"]}
    assert set(by_id) == {"CDGM_D-K9-25", "SCHOTT_F5"}
    assert by_id["CDGM_D-K9-25"]["copied_from"]["material_id"] == "CDGM_D-K9-25"
    assert by_id["CDGM_D-K9-25"]["dispersion"] == published("CDGM_D-K9-25")["dispersion"]
    checked = json_api.check_project(reopened)
    assert checked["ok"] is True, checked
    assert checked["catalog_version"] is None and checked["project_hash"] == project.content_hash()
    reply = json_api.paraxial_analysis(reopened, {"components": ["RX"], "wavelength_nm": 587.6})
    assert reply["ok"] is True, reply
    efl = reply["system"]["efl_mm"]
    assert efl is not None and 50.0 < abs(efl) < 1000.0
