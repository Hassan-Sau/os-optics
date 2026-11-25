"""Material discovery for agents: ``os-optics catalog materials`` / ``catalog material`` (sections 2
and 3).

On the synthetic schema-3.6 release (``material_catalog36``): list by maker, by designation (exact and loose), by the
wavelength range a material covers and with measured bulk loss there; resolve one name as prescription import does;
and select on infrared properties through the Engine's own evaluation -- the index at a wavelength, the dispersion figure
``(n_mid - 1) / (n_lo - n_hi)`` across a band and the internal transmittance of a thickness, which is ``null`` with a
reason (never 100 %) where no loss is stated. The transmittance reference is computed here from the fixture's own
extinction coefficients (linear in measured k between rows, then ``alpha = 4 pi k / lambda_vacuum``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import material_catalog36 as M
from os_optics.catalog_materials import find_materials, material_summary, resolve_name
from os_optics.cli import main

CATALOG = M.catalog()


@pytest.fixture
def catalog_file(tmp_path: Path) -> str:
    path = tmp_path / "catalog36.json"
    path.write_text(json.dumps(M.catalog_document()), encoding="utf-8")
    return str(path)


def run(capsys, catalog_file: str, *argv: str) -> tuple[int, dict]:
    code = main(["catalog", "--catalog", catalog_file, *argv])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


def ids(document: dict) -> list[str]:
    return [m["material_id"] for m in document["materials"]]


def test_list_by_maker_designation_and_search(capsys, catalog_file):
    code, cdgm = run(capsys, catalog_file, "materials", "--manufacturer", "CDGM")
    assert code == 0 and ids(cdgm) == ["CDGM_D-K9", "CDGM_D-K9-25", "CDGM_F2", "CDGM_F5"] and cdgm["total"] == 4
    variant = next(m for m in cdgm["materials"] if m["material_id"] == "CDGM_D-K9-25")["identity"]
    assert variant["variant_of"] == "CDGM_D-K9" and variant["ordering_status"]["design_availability"] == "CURRENT"
    assert variant["source_edition"] == "CDGM 2026-09"
    assert ids(run(capsys, catalog_file, "materials", "--designation", "F5")[1]) == ["CDGM_F5", "SCHOTT_F5"]
    assert ids(run(capsys, catalog_file, "materials", "--designation", "F2")[1]) == ["CDGM_F2", "F2"]
    assert ids(run(capsys, catalog_file, "materials", "--designation", "sfpl 53")[1]) == []
    assert ids(run(capsys, catalog_file, "materials", "--designation", "sfpl 53", "--loose")[1]) == ["OHARA_S-FPL53"]
    assert ids(run(capsys, catalog_file, "materials", "--search", "dk9")[1]) == ["CDGM_D-K9", "CDGM_D-K9-25"]
    paged = run(capsys, catalog_file, "materials", "--manufacturer", "CDGM", "--offset", "1", "--limit", "2")[1]
    assert ids(paged) == ["CDGM_D-K9-25", "CDGM_F2"] and paged["total"] == 4


def test_list_by_wavelength_coverage_and_conventions(capsys, catalog_file):
    covering = run(capsys, catalog_file, "materials", "--covers-nm", "8000", "12000")[1]
    assert ids(covering) == ["TESTIR"]
    assert ids(run(capsys, catalog_file, "materials", "--covers-nm", "8000", "12000", "--with-bulk-loss")[1]) == ["TESTIR"]
    assert ids(run(capsys, catalog_file, "materials", "--covers-nm", "2000", "12000", "--with-bulk-loss")[1]) == []  # loss from 8 um only
    visible = run(capsys, catalog_file, "materials", "--wavelength-nm", "380")[1]
    assert "OHARA_S-FPL53" in ids(visible) and "TESTIR" not in ids(visible)
    fpl53 = next(m for m in visible["materials"] if m["material_id"] == "OHARA_S-FPL53")
    assert fpl53["convention"]["wavelength"] == "AIR" and fpl53["convention"]["reference_air"]["temperature_c"] == "20"
    assert fpl53["convention"]["source_band_nm"] == {"min_nm": "365", "max_nm": "1014", "on": "air wavelengths"}
    assert (fpl53["valid_band_nm"]["min_nm"], fpl53["valid_band_nm"]["max_nm"]) == M.FPL53_BAND
    assert fpl53["bulk_loss"]["band_nm"]["min_nm"] != "400"  # the loss rows are on air wavelengths: their vacuum band
    bk7 = next(m for m in visible["materials"] if m["material_id"] == "N-BK7")
    assert bk7["identity"] is None and bk7["convention"]["wavelength"] == "NOT_STATED" and bk7["bulk_loss"] is None
    assert "not a zero loss" in bk7["bulk_loss_note"]
    assert run(capsys, catalog_file, "materials", "--covers-nm", "900", "500")[0] == 64


def test_resolve_one_name_as_the_import_does(capsys, catalog_file):
    code, ambiguous = run(capsys, catalog_file, "material", "F5")
    assert code == 1 and ambiguous["resolution"]["how"] == "ambiguous"
    assert [c["material_id"] for c in ambiguous["candidates"]] == ["CDGM_F5", "SCHOTT_F5"] and "MAKER:DESIGNATION" in ambiguous["next_step"]
    code, cdgm = run(capsys, catalog_file, "material", "CDGM:F5")
    assert code == 0 and cdgm["resolution"]["resolved"] == "CDGM_F5" and cdgm["material"]["identity"]["manufacturer"] == "CDGM"
    assert run(capsys, catalog_file, "material", "F5", "--manufacturer", "SCHOTT")[1]["resolution"]["resolved"] == "SCHOTT_F5"
    code, shared = run(capsys, catalog_file, "material", "F2")
    assert code == 1 and shared["resolution"]["resolved"] is None and shared["resolution"]["how"] == "ambiguous"
    assert shared["resolution"]["candidates"] == ["CDGM_F2", "F2"] and shared["resolution"]["also_matches"] == []
    assert [m["identity"]["manufacturer"] for m in shared["candidates"]] == ["CDGM", "SCHOTT"]
    ordered = run(capsys, catalog_file, "material", "F2", "--glass-catalogs", "CDGM SCHOTT")[1]["resolution"]
    assert ordered["resolved"] == "CDGM_F2" and ordered["how"] == "catalogue order"
    assert run(capsys, catalog_file, "material", "F2", "--manufacturer", "SCHOTT")[1]["resolution"]["resolved"] == "F2"
    stable = run(capsys, catalog_file, "material", "ID:F2", "--glass-catalogs", "CDGM")[1]["resolution"]
    assert stable["resolved"] == "F2" and stable["how"] == "material_id (explicit)" and stable["also_matches"] == []
    loose = run(capsys, catalog_file, "material", "dk9-25")[1]["resolution"]
    assert loose["resolved"] == "CDGM_D-K9-25" and loose["how"] == "loose match"
    code, unknown = run(capsys, catalog_file, "material", "UNOBTAINIUM")
    assert code == 1 and unknown["resolution"]["resolved"] is None and unknown["candidates"] == []


def test_infrared_selection_uses_the_engine_evaluation(capsys, catalog_file):
    pytest.importorskip("os_optics.optics.materials.physics")
    code, document = run(capsys, catalog_file, "material", "TESTIR", "--at-nm", "10000", "--band-nm", "8000", "12000",
                         "--thickness-mm", "2")
    assert code == 0
    ir = document["ir"]
    assert ir["available"] is True and ir["at"]["n"] == pytest.approx(2.498, abs=1e-9)  # a measured table's own point
    assert "derivative_limitation" in ir["at"] and "n_group" not in ir["at"]  # no GDD claim from a measured table
    band = ir["band"]
    assert [p["n"] for p in band["points"]] == pytest.approx([2.502, 2.498, 2.494], abs=1e-9)
    assert band["dispersion_figure"] == pytest.approx((2.498 - 1.0) / (2.502 - 2.494), rel=1e-4)
    # alpha = 4 pi k / lambda (lambda in mm), T = exp(-alpha L)
    expected = {nm: math.exp(-4.0 * math.pi * k / (nm * 1e-6) * 2.0) for nm, k in ((8000.0, 1e-5), (10000.0, 5e-6), (12000.0, 1e-5))}
    assert ir["at"]["transmission"]["internal_transmittance"] == pytest.approx(expected[10000.0], abs=2e-6)
    assert [t["internal_transmittance"] for t in band["transmission"]] == pytest.approx([expected[8000.0], expected[10000.0], expected[12000.0]], abs=2e-6)


def test_missing_loss_is_not_established_never_full_transmission(capsys, catalog_file):
    pytest.importorskip("os_optics.optics.materials.physics")
    bk7 = run(capsys, catalog_file, "material", "N-BK7", "--at-nm", "587.6", "--thickness-mm", "10")[1]["ir"]["at"]
    assert bk7["n"] == pytest.approx(1.5168, abs=2e-4)
    assert bk7["transmission"]["internal_transmittance"] is None and bk7["transmission"]["reason"]
    outside = run(capsys, catalog_file, "material", "TESTIR", "--at-nm", "4000", "--thickness-mm", "2")[1]["ir"]["at"]
    assert outside["n"] is not None and outside["transmission"]["internal_transmittance"] is None  # loss rows start at 8 um
    beyond = run(capsys, catalog_file, "material", "N-BK7", "--at-nm", "10000")[1]["ir"]["at"]
    assert beyond["n"] is None and beyond["reason"]  # outside its valid band: not extrapolated
    assert main(["catalog", "--catalog", catalog_file, "material", "N-BK7", "--thickness-mm", "2"]) == 64


def test_the_json_api_mirrors_the_commands():
    from os_optics.local import json_api

    json_api.install_catalog(json.dumps(M.catalog_document()).encode("utf-8"))
    found = json_api.catalog_find_materials({"manufacturer": "CDGM", "designation": "F5"})
    assert found["ok"] is True and [m["material_id"] for m in found["materials"]] == ["CDGM_F5"]
    ir = json_api.catalog_find_materials({"covers_nm": [8000, 12000], "with_bulk_loss": True, "at_nm": 10000, "thickness_mm": 2})
    assert [m["material_id"] for m in ir["materials"]] == ["TESTIR"]
    assert ir["materials"][0]["ir"]["at"]["transmission"]["internal_transmittance"] is not None
    one = json_api.catalog_find_materials({"name": "F2", "glass_catalogs": ["CDGM"]})
    assert one["ok"] is True and one["resolution"]["resolved"] == "CDGM_F2" and one["material"]["identity"]["manufacturer"] == "CDGM"
    ambiguous = json_api.catalog_find_materials({"name": "F5"})
    assert ambiguous["ok"] is True and ambiguous["resolution"]["resolved"] is None and len(ambiguous["candidates"]) == 2
    shared = json_api.catalog_find_materials({"name": "F2"})
    assert shared["ok"] is True and shared["resolution"]["resolved"] is None
    assert [m["material_id"] for m in shared["candidates"]] == ["CDGM_F2", "F2"]
    explicit = json_api.catalog_find_materials({"name": "ID:F2"})
    assert explicit["ok"] is True and explicit["resolution"]["resolved"] == "F2"
    for refused in (json_api.catalog_find_materials({"colour": "red"}), json_api.catalog_find_materials({"thickness_mm": 2}),
                    json_api.catalog_find_materials({"name": "F2", "band_nm": [12000, 8000]}), json_api.catalog_find_materials({"name": "F2", "covers_nm": [1, 2]})):
        assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "CATALOG_QUERY_INVALID", refused
    methods = json_api.capabilities()["json_api_methods"]
    assert "catalog_find_materials" in methods


def test_the_library_functions():
    assert [m.material_id for m in find_materials(CATALOG.materials, manufacturer="SCHOTT")] == ["F2", "SCHOTT_F5"]
    assert [m.material_id for m in find_materials(CATALOG.materials, availability="NOT_STATED")] == ["FUSED_SILICA", "N-BK7", "PMMA_SULTANOVA2009"]
    assert resolve_name(CATALOG.materials, "S-FPL53")["resolved"] == "OHARA_S-FPL53"
    summary = material_summary(CATALOG.material("CDGM_D-K9-25"))
    assert summary["thermo_optic"]["kind"] == "SCHOTT_DN_DT" and summary["dispersion"] == {"kind": "SELLMEIER", "terms": 3}
