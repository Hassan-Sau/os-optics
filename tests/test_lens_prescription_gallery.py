"""The Zemax-style prescriptions of ``fixtures/prescriptions`` (``lens_*``, ``double_gauss_6``): every construction route runs,
apertures and edges are what the file declares, and a lens seen side-on has a curved silhouette, not a rectangle.

The acceptance of how lenses look is a person looking at real browser screenshots (``scripts/lens_prescription_gallery.py``);
the pixel test here only guards the result with the Engine's own numpy snapshot.

Limits chosen before running: ray data of the routes agrees within 1e-9 mm (one model, the same surfaces); a drawn wall spans
the reported edge thickness within the chord tolerance 0.002 mm; side-on, the centre and rim widths differ by a factor above 1.5
for the two singlets (their centre / edge thickness ratios are 2.5 and 2.3 by hand from the fixtures: e.g. biconvex R 51.5, rim
radius 12.7: sag 1.590 mm, edge 5.3 - 2 x 1.590 = 2.12 mm, centre 5.3 mm) and below 1.1 for a flat plate.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from os_optics.local import json_api
from os_optics.optical_view.section import optical_section, scene_section_elements
from os_optics.optical_view.surface_mesh import CHORD_TOLERANCE_MM, profile_stations
from os_optics.prescription import PrescriptionImportError, import_prescription
from os_optics.viewer import optical_view_scene
from os_optics.viewer.raster import render_view

ENGINE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "prescriptions"
SINGLE_MATERIALS = "[materials]\nname,kind,index,min_nm,max_nm\nGLASS,CONSTANT_INDEX,1.5,400,800\n"
RAY_TOLERANCE_MM = 1e-9
WALL_TOLERANCE_MM = 0.002
CURVED_WIDTH_RATIO = 1.5
RECTANGLE_WIDTH_RATIO = 1.1
IMAGE_SIZE = 900
#: A pixel is the lens when its channels differ from the backdrop's by more than this in sum (of 765).
BACKGROUND_DIFFERENCE = 6
#: Rows of the silhouette compared: near the rim, away from the antialiased end rows.
RIM_ROW_SHARE = 0.04


def _gallery():
    spec = importlib.util.spec_from_file_location("lens_prescription_gallery", ENGINE / "scripts" / "lens_prescription_gallery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GALLERY = _gallery()
SELF_CONTAINED = [k for k in GALLERY.KINDS if k != "cooke_triplet"]  # the Cooke fixture takes its flint from the public catalog


def _text(kind: str) -> str:
    return (FIXTURES / f"{kind}.csv").read_text(encoding="utf-8")


def _imported(text: str) -> dict:
    return {"ok": True, **import_prescription(text, model_id="lens", glass_source="inline").as_dict()}


def _models(text: str) -> list[dict]:
    """The native models alone (their display names cite the source digest and rows)."""

    return [m["model"] for m in _imported(text)["optical_models"]]


def _run(document: dict, out: Path) -> dict:
    reply = json_api.run_project(document, output_dir=str(out))
    assert reply.get("ok"), reply
    return reply


#: The kinds whose glass is thicker at its edge than the review band (0.35 of the clear diameter): the over-thick one
#: (2.07), a near-ball lens (0.40) and the double Gauss's two thick inner elements (0.40 each). Only the over-thick one is
#: beyond the catalog-calibrated 1.5 (a warning); the others are facts of their rows (``above_review_ratio``).
EDGE_REVIEWED = ("lens_implausible_thick", "lens_near_ball", "double_gauss_6")
EDGE_WARNED = ("lens_implausible_thick",)


@pytest.mark.parametrize("kind", SELF_CONTAINED)
def test_every_kind_imports_with_only_the_over_thick_one_warned_and_the_thick_ones_flagged(kind):
    report = _imported(_text(kind))["report"]
    codes = {w["code"] for w in report["warnings"]}
    assert codes == ({"EDGE_THICKNESS_IMPLAUSIBLE"} if kind in EDGE_WARNED else set()), report["warnings"]
    for warning in report["warnings"]:
        assert "% of its clear diameter" in warning["message"] and "(above 150%)" in warning["message"]
    rows = [g for member in report["members"] if member["model_kind"] == "SURFACE_SEQUENCE" for g in member["edge_thickness"]]
    flagged = [g for g in rows if g.get("above_review_ratio")]
    assert bool(flagged) is (kind in EDGE_REVIEWED and kind not in EDGE_WARNED)
    assert all(0.35 < float(g["ratio_to_clear_diameter"]) <= 1.5 for g in flagged)
    for member in report["members"]:
        if member["model_kind"] == "SURFACE_SEQUENCE":
            assert all(Decimal(g["edge_thickness_mm"]) > 0 for g in member["edge_thickness"])
            assert all(Decimal(m["ratio"]) >= 1 for m in member["mechanical_to_clear"])


def test_edge_semi_diameter_is_the_mechanical_column():
    text = _text("lens_airspaced_doublet")
    assert _models(text.replace("mech_semi_diameter", "edge_semi_diameter")) == _models(text)


def test_the_air_spaced_doublet_keeps_its_declared_apertures_and_spacings(tmp_path):
    """No thickening: every vertex is at the sum of the file's thicknesses, every clear and edge radius is the file's."""

    imported = _imported(_text("lens_airspaced_doublet"))
    by_id = {m["model_id"]: m["model"] for m in imported["optical_models"]}
    l1, l2 = by_id["lens-L1"], by_id["lens-L2"]
    assert [s["vertex_z_mm"] for s in l1["surfaces"]] == ["0", "4"] and [s["vertex_z_mm"] for s in l2["surfaces"]] == ["0", "2.2"]
    assert [m["pose"]["position_mm"][2] for m in by_id["lens"]["members"]] == [0.0, 4.4]
    assert [s["clear_radius_mm"] for s in l1["surfaces"] + l2["surfaces"]] == ["11", "10.4", "10.2", "10"]
    assert (l1["edge_radius_mm"], l2["edge_radius_mm"]) == ("12.5", "12")
    assert "edge_radius_mm" not in l2["surfaces"][0] and l2["surfaces"][1]["edge_radius_mm"] == "11"
    reply = _run(GALLERY.project_compound(imported), tmp_path / "run")
    assert reply["local_validation"]["simulation"]["outcome"]["execution"]["complete"]


def test_the_drawn_wall_spans_the_real_edge_thickness(tmp_path):
    """Read back from ``optical_scene.glb``: each element's wall is as long as its edge thickness and at its own outline radius."""

    imported = _imported(_text("lens_airspaced_doublet"))
    _run(GALLERY.project_compound(imported), tmp_path / "run")
    part = next(p for p in optical_view_scene(tmp_path / "run").parts if p.part_id == "LENS")
    walls = sorted((m.positions for m in part.meshes if m.material == "optic_edge" and m.surface_id is None), key=lambda p: p[:, 0].min())
    members = [m for m in imported["report"]["members"] if m["model_kind"] == "SURFACE_SEQUENCE"]
    assert len(walls) == len(members) == 2
    for wall, member, radius in zip(walls, members, (12.5, 12.0), strict=True):
        assert float(np.ptp(wall[:, 0])) == pytest.approx(float(member["edge_thickness"][0]["edge_thickness_mm"]), abs=WALL_TOLERANCE_MM)
        assert np.hypot(wall[:, 1], wall[:, 2]) == pytest.approx(radius, abs=WALL_TOLERANCE_MM)


def test_the_207_percent_element_is_reported_by_the_import_and_by_the_check():
    imported = _imported(_text("lens_implausible_thick"))
    (warning,) = imported["report"]["warnings"]
    assert warning["code"] == "EDGE_THICKNESS_IMPLAUSIBLE" and "element L1" in warning["message"] and "207%" in warning["message"]
    assert imported["report"]["members"][0]["edge_thickness"][0]["ratio_to_clear_diameter"] == "2.0678"
    reply = json_api.check_project(GALLERY.project_compound(imported))
    found = [i for i in reply["optical_solids"]["issues"] if i["code"] == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert len(found) == 1 and "207%" in found[0]["message"] and found[0]["member_id"] == "L1"


def test_glass_far_beyond_a_clear_aperture_is_reported_by_the_import():
    text = "surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n1,100,4,GLASS,10,10\n2,-100,5,,2,\n" + SINGLE_MATERIALS
    (warning,) = [w for w in _imported(text)["report"]["warnings"] if w["code"] != "EDGE_THICKNESS_IMPLAUSIBLE"]  # 4 mm of glass on 4 mm
    assert warning["code"] == "EDGE_FAR_BEYOND_CLEAR_APERTURE" and "element L1, surface 2" in warning["message"] and "5.00 times" in warning["message"]


@pytest.mark.parametrize("directive, needle", [
    ("edge_thickness_max_mm = 2", "above the declared edge_thickness_max_mm 2"),
    ("edge_thickness_min_mm = 3", "below the declared edge_thickness_min_mm 3"),
    ("mech_to_clear_ratio_max = 1.05", "above the declared mech_to_clear_ratio_max 1.05"),
    ("mech_to_clear_ratio_min = 1.2", "below the declared mech_to_clear_ratio_min 1.2"),
])
def test_an_element_outside_a_declared_range_is_refused(directive, needle):
    text = _text("lens_biconvex").replace("#! units = mm", f"#! units = mm\n#! {directive}")
    with pytest.raises(PrescriptionImportError) as refused:
        import_prescription(text, glass_source="inline")
    assert any(p.code.value == "EDGE_INVALID" and needle in p.message for p in refused.value.problems)


def test_a_declared_range_the_element_meets_changes_nothing():
    text = _text("lens_biconvex")
    declared = text.replace("#! units = mm", "#! units = mm\n#! edge_thickness_min_mm = 2\n#! edge_thickness_max_mm = 3\n#! mech_to_clear_ratio_max = 1.2")
    assert _models(declared) == _models(text)


def test_the_edge_must_lie_inside_the_profile():
    text = "surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n1,6,10,GLASS,4,6.5\n2,-6,5,,4,6.5\n" + SINGLE_MATERIALS
    with pytest.raises(PrescriptionImportError) as refused:
        import_prescription(text, glass_source="inline")
    assert [p.code.value for p in refused.value.problems] == ["EDGE_INVALID"] and "does not reach" in refused.value.problems[0].message


def test_a_section_across_a_steep_face_edged_inside_its_outline_completes(tmp_path):
    """The double Gauss's concave faces end (R 11.14 mm) inside the outline of their element (11.8 mm): their flat flange is drawn,
    and sampling beyond the profile's domain stays a single interval (it once split to 2^40 of them)."""

    assert profile_stations(lambda _t: None, 0.0, 1.0, CHORD_TOLERANCE_MM) == [0.0, 1.0]
    _run(GALLERY.project_compound(_imported(_text("double_gauss_6"))), tmp_path / "run")
    source = optical_view_scene(tmp_path / "run").source_scene
    section = optical_section(scene_section_elements(source), "LENS", "XZ", beam_paths=source.beam_paths)
    members = {e["member_id"]: e for e in section["elements"]}
    assert len(members["L2"]["media"]) == 2 and members["L2"]["surfaces"][2]["edge_mm"] == [-9.0, 9.0]
    points = [p for segment in members["L2"]["surfaces"][2]["segments"] for p in segment["points"]]
    flange = [z for t, z in points if abs(t) >= 9.0]
    assert max(abs(t) for t, _ in points) == 11.8 and max(flange) - min(flange) < 1e-6


def _rays(reply: dict) -> list[np.ndarray]:
    """Every traced chief and pupil polyline of a run, in path order."""

    data = json.loads(reply["files"]["ray_data.json"])
    rays = []
    for path in data["paths"]:
        rays.append(np.asarray(path["chief"]["points_mm"]))
        rays += [np.asarray(ray["points_mm"]) for ray in path["pupil"]["rays"]]
    return rays


@pytest.mark.parametrize("kind", ["lens_airspaced_doublet", "double_gauss_6"])
def test_the_three_routes_trace_the_same_rays(kind, tmp_path):
    """One model each per element, one per group with no air-gap inflation, and the imported unit: the same ray data."""

    imported = _imported(_text(kind))
    reference = None
    for route, build in GALLERY.ROUTE_BUILDERS.items():
        reply = _run(build(imported), tmp_path / route)
        assert not [i for i in reply["summary"].get("issues", ()) if "SOLID" in str(i)], reply["summary"]
        rays = _rays(reply)
        assert len(rays) > 1
        if reference is None:
            reference = rays
            continue
        assert len(rays) == len(reference), route
        for ray, expected in zip(rays, reference, strict=True):
            assert ray == pytest.approx(expected, abs=RAY_TOLERANCE_MM), route


def _widths(tmp_path: Path, text: str, *, cut: bool = False) -> np.ndarray:
    """Width in pixels of the lens per image row, rim to rim, from the Engine's numpy snapshot looking square-on at its side (the
    lens alone in the picture: source, plane and beams left out). ``cut``: only the triangles the plane through the axis
    crosses, the lens's own drawn profile (the whole lens's outline is its rim: a biconcave lens seen from the side is a
    rectangle, its hollows behind the rim)."""

    imported = _imported(text)
    run = tmp_path / "run"
    _run(GALLERY.project_compound(imported), run)
    scene = optical_view_scene(run)
    lens = next(p for p in scene.parts if p.part_id == "LENS")
    low, high = lens.bounds
    if cut:
        def crossing(mesh):
            y = mesh.positions[:, 1][mesh.triangles]
            return replace(mesh, triangles=mesh.triangles[(y.min(axis=1) <= 0.0) & (y.max(axis=1) >= 0.0)])

        lens = replace(lens, meshes=tuple(crossing(m) for m in lens.meshes))
    scene = replace(scene, parts=(lens,), beams=(), markers=(), focus_min=low, focus_max=high, bounds_min=low, bounds_max=high, notes=())
    image, _ = render_view(scene, "front", IMAGE_SIZE, IMAGE_SIZE, labels=False)
    middle = IMAGE_SIZE // 2
    window = image[:, IMAGE_SIZE // 4:3 * IMAGE_SIZE // 4].astype(int)  # the axes triad and the scale bar lie outside it, the title above the gap
    background = image[:, IMAGE_SIZE // 10].astype(int)[:, None, :]  # per row: a backdrop may shade from top to bottom
    drawn = np.abs(window - background).sum(axis=2) > BACKGROUND_DIFFERENCE
    top = bottom = next(r for r in range(middle, 0, -1) if drawn[r].any())
    while top > 0 and drawn[top - 1].any():
        top -= 1
    while bottom < IMAGE_SIZE - 1 and drawn[bottom + 1].any():
        bottom += 1
    return np.array([np.flatnonzero(r).max() - np.flatnonzero(r).min() + 1 for r in drawn[top:bottom + 1]], dtype=float)


def _centre_and_rim(widths: np.ndarray) -> tuple[float, float]:
    n = len(widths)
    skip = max(1, math.ceil(n * RIM_ROW_SHARE))
    return float(widths[n // 2]), float((widths[skip] + widths[n - 1 - skip]) / 2.0)


def test_a_biconvex_lens_side_on_is_wider_at_the_centre_than_at_the_rim(tmp_path):
    centre, rim = _centre_and_rim(_widths(tmp_path, _text("lens_biconvex")))
    assert centre > CURVED_WIDTH_RATIO * rim


def test_the_drawn_profile_of_a_biconvex_lens_is_wider_at_the_centre(tmp_path):
    centre, rim = _centre_and_rim(_widths(tmp_path, _text("lens_biconvex"), cut=True))
    assert centre > CURVED_WIDTH_RATIO * rim


def test_the_drawn_profile_of_a_biconcave_lens_is_wider_at_the_rim(tmp_path):
    centre, rim = _centre_and_rim(_widths(tmp_path, _text("lens_biconcave"), cut=True))
    assert rim > CURVED_WIDTH_RATIO * centre


@pytest.mark.parametrize("cut", [False, True])
def test_a_flat_plate_side_on_is_a_rectangle(tmp_path, cut):
    """The control: the measure above does say when a silhouette is a rectangle."""

    plate = ("#! units = mm\nsurface,radius,thickness,material,semi_diameter,mech_semi_diameter\n"
             "1,inf,5.3,GLASS,11.4,12.7\n2,inf,48,,11.4,12.7\n" + SINGLE_MATERIALS)
    centre, rim = _centre_and_rim(_widths(tmp_path, plate, cut=cut))
    assert max(centre, rim) < RECTANGLE_WIDTH_RATIO * min(centre, rim)
