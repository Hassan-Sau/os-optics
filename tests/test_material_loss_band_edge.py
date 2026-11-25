"""Selection and Engine evaluation agree at every published bulk-loss boundary.

The public material definitions come from the committed public material records
(``tests/fixtures/materials/redistributable_materials.json``) and the synthetic design materials of
``material_catalog36``, not from an agent's scratch candidate or a package on sys.path. The Engine public API and
selection are exercised independently at both sides of each edge, for both kinds of bulk loss (extinction coefficients and
internal transmittance) and for lower and upper edges.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import material_catalog36 as M
from os_optics._vendor.shared_contracts.design_catalog import PublicMaterialV1
from os_optics.catalog_materials import list_materials, loss_band_nm, material_summary
from os_optics.cli import main
from os_optics.optics.materials.physics import bulk_alpha_per_mm

LIBRARY = Path(__file__).resolve().parent / "fixtures" / "materials" / "redistributable_materials.json"
EDGE_STEP_NM = Decimal("0.00001")  # > the Engine's 1e-9 nm range slack


@pytest.fixture(scope="module")
def materials():
    records = json.loads(LIBRARY.read_text(encoding="utf-8"))["records"]
    return [PublicMaterialV1.model_validate(r["public"]) for r in records if r["public"].get("bulk_loss")]


def test_hws9_lower_edge_is_not_rounded_down_into_false_loss_coverage(materials) -> None:
    """CDGM HWS9's internal transmittance starts at 2500 nm in air: the vacuum edge is 2500.6704516 nm, and a query just
    below it is not loss-covered (it was once rounded down into false coverage)."""
    material = next(m for m in materials if m.material_id == "CDGM_HWS9")
    lower, upper = loss_band_nm(material)
    assert abs(lower - Decimal("2500.6704516")) < Decimal("0.0000001")
    assert bulk_alpha_per_mm(material, 2500.670)[0] is None
    catalog = SimpleNamespace(catalog_version="2026-11-23.1", schema_version="3.6", materials=[material])
    before = list_materials(catalog, wavelength_nm="2500.670", with_bulk_loss=True)
    after = list_materials(catalog, wavelength_nm="2500.670452", with_bulk_loss=True)
    assert before["materials"] == [] and before["total"] == 0
    assert after["total"] == 1 and after["materials"][0]["material_id"] == "CDGM_HWS9"
    assert bulk_alpha_per_mm(material, 2500.670452)[0] is not None
    assert Decimal(material_summary(material)["bulk_loss"]["band_nm"]["min_nm"]) == lower
    assert Decimal(material_summary(material)["bulk_loss"]["band_nm"]["max_nm"]) == upper


def test_catalog_cli_excludes_the_false_lower_edge_and_includes_the_first_measured_point(materials, tmp_path, capsys) -> None:
    hws9 = next(m for m in materials if m.material_id == "CDGM_HWS9")
    document = M.catalog_document()
    document["materials"] = sorted([*document["materials"], hws9.model_dump(mode="json", exclude_none=True)],
                                   key=lambda material: material["material_id"])
    path = tmp_path / "public-catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    for nm, expected in (("2500.670", 0), ("2500.670452", 1)):
        code = main(["catalog", "--catalog", str(path), "materials", "--manufacturer", "CDGM",
                     "--designation", "HWS9", "--wavelength-nm", nm, "--with-bulk-loss"])
        assert code == 0
        found = json.loads(capsys.readouterr().out)
        assert found["total"] == expected
        if expected:
            assert found["materials"][0]["material_id"] == "CDGM_HWS9"


def test_every_published_loss_edge_has_consistent_selection_and_evaluation(materials) -> None:
    synthetic = [PublicMaterialV1.model_validate(m) for m in M.catalog_document()["materials"] if m.get("bulk_loss")]
    checked = set()
    for material in [*materials, *synthetic]:
        band = loss_band_nm(material)
        assert band is not None
        for side, (boundary, inside, outside) in (("lower", (band[0], band[0] + EDGE_STEP_NM, band[0] - EDGE_STEP_NM)),
                                                  ("upper", (band[1], band[1] - EDGE_STEP_NM, band[1] + EDGE_STEP_NM))):
            # Some sources measure loss beyond their index law's band. Only assert on wavelengths on which the
            # material index itself is valid; a queried catalog material requires BOTH index and loss there.
            model_low = Decimal(material.valid_band_nm.min_nm)
            model_high = Decimal(material.valid_band_nm.max_nm)
            if not model_low <= inside <= model_high or not model_low <= outside <= model_high:
                continue
            catalog = SimpleNamespace(catalog_version="2026-11-23.1", schema_version="3.6", materials=[material])
            for point in (inside, outside):
                numeric = float(point)
                actual = bulk_alpha_per_mm(material, numeric)[0] is not None
                listed = list_materials(catalog, wavelength_nm=str(point), with_bulk_loss=True)["total"] == 1
                assert listed == actual, (material.material_id, boundary, point, actual, listed)
            kind = getattr(material.bulk_loss.kind, "value", material.bulk_loss.kind)
            checked.add((kind, side, material.material_id))
    covered = {(kind, side) for kind, side, _ in checked}
    assert covered == {(kind, side) for kind in ("EXTINCTION_COEFFICIENT", "INTERNAL_TRANSMITTANCE")
                       for side in ("lower", "upper")}, sorted(checked)
