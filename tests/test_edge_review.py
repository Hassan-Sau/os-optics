"""Edge-thickness review band and crossing profiles inside a clear aperture (``optical_solids.checks``).

The reconstructed case (made-up numbers in the shape of the reported one): a biconvex element, front R = 14.124 mm with a
clear semi-diameter of 12 mm (the element outline), back R = -30 mm with a clear semi-diameter of 8 mm edged at 8.124 mm,
8 mm thick. Held flat beyond its edge, the back leaves 8 - 1.121 - 6.675 = 0.204 mm of glass at the outline; the two
profiles themselves cross inside the front's clear aperture (at 12 mm they are 8 - 2.505 - 6.675 = -1.18 mm apart). The
references below are the closed-form sphere sags; the 1D search is refined to well below 1e-9 mm.

Edge band: an edge thickness above 1.5 of the clear diameter (the published catalog's largest is 1.30) is a warning for
any model; a declared model between its review ratio (0.35) and 1.5 is an INFO finding; no catalog product meets either
finding (``test_optical_solids.py`` census).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import sequence, surface
from os_optics._vendor.shared_contracts.design_catalog import SurfaceSequenceModelV1
from os_optics.cli import main as cli_main
from os_optics.local import json_api, load_public_catalog
from os_optics.local.compile import compile_element
from os_optics.optical_solids import EdgeThicknessBand, body_of, element_issues
from os_optics.optical_solids.checks import EDGE_THICKNESS_RATIO_LIMIT, EDGE_THICKNESS_REVIEW_RATIO
from os_optics.prescription import import_prescription

ENGINE = Path(__file__).resolve().parents[1]
R1, R2, CT, CLEAR1, CLEAR2, EDGE2 = 14.124, -30.0, 8.0, 12.0, 8.0, 8.124
SINGLE_MATERIALS = "[materials]\nname,kind,index,min_nm,max_nm\nGLASS,CONSTANT_INDEX,1.5,400,800\n"
CROSSING_CSV = ("surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n"
                f"1,{R1},{CT:g},GLASS,{CLEAR1:g},{CLEAR1:g}\n2,{R2:g},10,,{CLEAR2:g},{EDGE2}\n" + SINGLE_MATERIALS)


def _sag(r: float, rho: float) -> float:
    return r - math.copysign(math.sqrt(r * r - rho * rho), r)


def _gallery():
    spec = importlib.util.spec_from_file_location("lens_prescription_gallery_edges", ENGINE / "scripts" / "lens_prescription_gallery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _d(value: float) -> str:
    return f"{value:g}"


def _model(front: dict | None = None, back_edge: str | None = _d(EDGE2)) -> SurfaceSequenceModelV1:
    """The reconstructed element; ``front``: fields of the front surface (``clear``: its clear radius, None: none)."""

    front = dict(front or {})
    clear = front.pop("clear", _d(CLEAR1))
    s1 = surface(1, "0", radius=_d(R1), clear=clear or "1", **front)
    if clear is None:
        s1.pop("clear_radius_mm")
    s2 = surface(2, _d(CT), radius=_d(R2), clear=_d(CLEAR2), **({"edge_radius_mm": back_edge} if back_edge else {}))
    return SurfaceSequenceModelV1.model_validate(sequence([s1, s2], ["N-BK7"], edge=_d(CLEAR1)))


def _issues(model: SurfaceSequenceModelV1, *, catalog: bool = False, band: EdgeThicknessBand | None = None):
    element = SimpleNamespace(instance_id="C3", model=model, identity=SimpleNamespace(is_catalog=catalog))
    return element_issues(body_of(element), *(() if band is None else (band,)))


def test_profiles_crossing_inside_a_clear_aperture_are_an_error_not_a_thin_drawn_edge():
    issues, gaps = _issues(_model())
    unheld = CT + _sag(R2, CLEAR1) - _sag(R1, CLEAR1)
    held = CT + _sag(R2, EDGE2) - _sag(R1, CLEAR1)
    assert unheld == pytest.approx(-1.18, abs=5e-3) and held == pytest.approx(0.204, abs=5e-3)
    (gap,) = gaps
    assert gap["edge_thickness_mm"] == pytest.approx(held, abs=1e-6)  # the drawn body, held flat beyond the back's edge
    assert gap["clear_aperture_min_thickness_mm"] == pytest.approx(unheld, abs=1e-6)
    (found,) = [i for i in issues if i.code == "SURFACES_INTERSECT"]
    assert found.severity == "ERROR" and found.surface_id == "S1" and found.other_surface_id == "S2"
    assert found.value_mm == pytest.approx(unheld, abs=1e-6)
    assert math.hypot(*found.local_position_mm[:2]) == pytest.approx(CLEAR1, abs=1e-6)
    assert "inside the clear aperture of S1" in found.message and f"{held:.3f}"[:4] in found.message and "S2 is held flat" in found.message


def test_without_the_small_back_edge_the_old_check_already_refuses_it():
    """The same profiles reaching the outline on both surfaces: the drawn body itself crosses (the existing finding)."""

    issues, _ = _issues(_model(back_edge=None))
    (found,) = [i for i in issues if i.code == "SURFACES_INTERSECT"]
    assert "between the clear aperture and the outline" in found.message


def test_a_front_clear_aperture_inside_the_crossing_is_valid():
    issues, gaps = _issues(_model(front={"clear": "11"}))
    assert "SURFACES_INTERSECT" not in [i.code for i in issues]
    expected = CT + _sag(R2, 11.0) - _sag(R1, 11.0)
    assert expected > 0.0 and gaps[0]["clear_aperture_min_thickness_mm"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("front, crosses", [
    ({"clear": None, "clear_half_widths_mm": ["8.2", "8.2"]}, True),  # corners 11.60 mm out, past the 11.35 mm crossing
    ({"clear": None, "clear_half_widths_mm": ["8.2", "8.2"], "clear_shape": {"kind": "ELLIPSE"}}, False),  # inscribed: 8.2 mm
    ({"clear_shape": {"kind": "POLYGON", "vertices_mm": [["11.5", "0"], ["-5", "5"], ["-5", "-5"]]}}, True),
    ({"clear_shape": {"kind": "POLYGON", "vertices_mm": [["11", "0"], ["-5", "5"], ["-5", "-5"]]}}, False),
    ({"clear_shape": {"kind": "D_SHAPE", "flat_mm": "4", "flat_side": "+X"}}, True),  # the round side still reaches 12 mm
    ({"holes": [{"center_mm": ["0", "0"], "radius_mm": "2"}]}, True),
])
def test_the_exact_clear_shape_decides(front, crosses):
    issues, _ = _issues(_model(front=front))
    assert ("SURFACES_INTERSECT" in [i.code for i in issues]) is crosses


def _thick(thickness: str) -> SurfaceSequenceModelV1:
    """A plano-convex-ish element (R 200 mm front), 5.9 mm clear diameter, outline 4 mm."""

    return SurfaceSequenceModelV1.model_validate(
        sequence([surface(1, "0", radius="200", clear="2.95"), surface(2, thickness, clear="2.95")], ["N-BK7"], edge="4"))


def test_a_declared_model_above_the_review_band_is_a_fact_with_its_ratio_and_warned_beyond_the_catalog_limit():
    issues, _ = _issues(_thick("3.1"))  # 3.06 mm at the edge, 52 % of 5.9 mm: inside every published product's range
    (found,) = [i for i in issues if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert found.severity == "INFO" and found.instance_id == "C3"
    assert found.message.startswith("element C3: ") and found.message.endswith("52% of its clear diameter 5.9 mm, above the review ratio 35%")
    assert found.value_mm == pytest.approx(3.1 - _sag(200.0, 4.0), abs=1e-9)  # the glass at the 4 mm outline (closed-form sag)
    assert found.limit_mm == pytest.approx(EDGE_THICKNESS_REVIEW_RATIO * 5.9)
    assert not [i for i in _issues(_thick("1.9"))[0] if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]  # 31 %
    assert not [i for i in _issues(_thick("3.1"), band=EdgeThicknessBand(review_ratio=0.6))[0] if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    # beyond the catalog-calibrated 1.5 (no published product): a warning for a declared model too
    (beyond,) = [i for i in _issues(_thick("12.2"))[0] if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert beyond.severity == "WARNING" and "206%" in beyond.message and "(more than 150%)" in beyond.message
    assert beyond.limit_mm == pytest.approx(EDGE_THICKNESS_RATIO_LIMIT * 5.9)


def test_a_catalog_product_keeps_the_catalog_calibrated_limit():
    assert not [i for i in _issues(_thick("3.1"), catalog=True)[0] if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    (found,) = [i for i in _issues(_thick("12.2"), catalog=True)[0] if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert found.limit_mm == pytest.approx(EDGE_THICKNESS_RATIO_LIMIT * 5.9) and "206%" in found.message


def test_a_catalog_products_body_knows_it_is_one():
    """``test_optical_solids.py`` checks that no published product has a finding (the crossing and the catalog limit
    included); here: its bodies are the catalog's, and its glass is positive over every clear aperture."""

    catalog = load_public_catalog(REAL_CATALOG)
    product = next(p for p in sorted(catalog.products, key=lambda p: p.optical_sku) if p.simulation_models[0].model_kind == "SURFACE_SEQUENCE")
    element = compile_element("E", product, catalog, model_kind="SURFACE_SEQUENCE")
    body = body_of(element, design_geometry=product.design_geometry)
    assert body.catalog and not body_of(SimpleNamespace(instance_id="E", model=_thick("3"))).catalog
    _, gaps = element_issues(body)
    assert all(g["clear_aperture_min_thickness_mm"] > 0.0 for g in gaps if g["medium"] != "AIR")


def test_the_project_check_refuses_the_reconstructed_case_and_names_the_crossing():
    imported = {"ok": True, **import_prescription(CROSSING_CSV, model_id="c3", glass_source="inline").as_dict()}
    assert not [w for w in imported["report"]["warnings"] if w["code"] == "EDGE_THICKNESS_IMPLAUSIBLE"]  # the drawn edge is thin
    reply = json_api.check_project(_gallery().project_compound(imported))
    assert reply["ok"] is False
    (problem,) = [p for p in reply["error"]["problems"] if p["code"] == "SURFACES_INTERSECT"]
    assert "cross inside the clear aperture of S1" in problem["message"] and problem["details"]["value_mm"] == pytest.approx(-1.18, abs=5e-3)
    assert reply["optical_solids"]["status"] == "INVALID"


def test_project_solids_takes_another_review_ratio(tmp_path, capsys):
    text = ("surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n1,200,3.1,GLASS,2.95,4\n2,inf,10,,2.95,4\n" + SINGLE_MATERIALS)
    imported = {"ok": True, **import_prescription(text, model_id="thick", glass_source="inline").as_dict()}
    # 52 %: a fact of the import report, not a warning
    assert imported["report"]["warnings"] == []
    (row,) = imported["report"]["members"][0]["edge_thickness"]
    assert row["above_review_ratio"] is True and row["ratio_to_clear_diameter"] == "0.5186"  # (3.1 - sag(4 mm)) / 5.9
    assert float(row["edge_thickness_mm"]) == pytest.approx(3.1 - _sag(200.0, 4.0), abs=1e-4)
    relaxed = import_prescription(text, model_id="thick", glass_source="inline", edge_review_ratio=0.6).report
    assert "above_review_ratio" not in relaxed["members"][0]["edge_thickness"][0]
    path = tmp_path / "project.json"
    project = _gallery().project_compound(imported)
    path.write_text(json.dumps(project), encoding="utf-8")
    assert cli_main(["project", "solids", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "VALID"
    assert [(i["code"], i["severity"]) for i in report["issues"] if i["code"] == "EDGE_THICKNESS_IMPLAUSIBLE"] == [("EDGE_THICKNESS_IMPLAUSIBLE", "INFO")]
    assert cli_main(["project", "solids", str(path), "--edge-review-ratio", "0.6"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "EDGE_THICKNESS_IMPLAUSIBLE" not in [i["code"] for i in report["issues"]]
    assert cli_main(["project", "solids", str(path), "--edge-review-ratio", "-1"]) == 64
    # the JSON API takes the same ratio
    for call in (json_api.inspect_optical_solids, json_api.check_project):
        default, relaxed_reply = call(project), call(project, edge_review_ratio=0.6)
        solids = default if call is json_api.inspect_optical_solids else default["optical_solids"]
        relaxed_solids = relaxed_reply if call is json_api.inspect_optical_solids else relaxed_reply["optical_solids"]
        assert "EDGE_THICKNESS_IMPLAUSIBLE" in [i["code"] for i in solids["issues"]]
        assert "EDGE_THICKNESS_IMPLAUSIBLE" not in [i["code"] for i in relaxed_solids["issues"]]
        assert call(project, edge_review_ratio=0)["error"]["problems"][0]["code"] == "PROJECT_INVALID"
