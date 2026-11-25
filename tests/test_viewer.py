"""The local 3D viewer (``os_optics.viewer``, ``os-optics view``): scene model, summary, snapshots, page and server.

Runs the bundled reference designs (focus and dichroic routing) through the real local workflow, and opens the Assembly
Preview Package reference fixture in memory.
"""

from __future__ import annotations

import builtins
import http.client
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from os_optics.assembly import load_assembly_preview
from os_optics.assembly.mesh import read_part_mesh
from os_optics.assembly.transforms import Rigid
from os_optics.cli import main
from os_optics.local import example_project_of_kind, run_optical_project

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import declared_component, with_models
from local_helpers import ALONG_X, plane, project, source
from os_optics.optical_view import parse_glb
from os_optics.viewer import (
    PRESET_ORDER,
    ViewerInputError,
    assembly_view_scene,
    optical_view_scene,
    render_snapshots,
    scene_summary,
    snapshot_images,
    view_assembly,
    view_document,
    view_run,
)
from os_optics.viewer.page import ASSETS, asset_bytes, page_files, standalone_html
from os_optics.viewer.png import read_png_rgb
from os_optics.viewer.server import LOOPBACK, TOKEN_BYTES

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "tests/fixtures/assembly_preview/reference.assembly-preview.zip"
MULTI_PREVIEW = ROOT / "tests/fixtures/assembly_drawing/four_lenses.assembly-preview.zip"
SMALL = {"width": 480, "height": 360}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    out = {}
    for kind in ("focus", "dichroic_routing"):
        directory = tmp_path_factory.mktemp(kind)
        run_optical_project(example_project_of_kind(kind, REAL_CATALOG), catalog=REAL_CATALOG, output_dir=directory)
        out[kind] = directory
    return out


@pytest.fixture(scope="module")
def preview():
    return load_assembly_preview(PREVIEW)


# ------------------------------------------------------------------------------------------------ scene model / summary


def test_the_optical_summary_names_parts_beams_units_axes_and_presets(runs):
    summary = scene_summary(runs["focus"])
    assert summary["view_kind"] == "OPTICAL_SIMULATION_VIEW" and summary["frame"] == "PROJECT_DESIGN_FRAME"
    assert summary["units"] == {"length": "mm"} and summary["axes"]["up"] == "+Z"
    assert set(summary["parts_by_class"]) == {"SOURCE", "LENS", "TARGET_PLANE"}
    (lens,) = summary["parts_by_class"]["LENS"]
    assert lens["id"] == "L1" and lens["sku"].startswith("OS-OPT-")
    size = lens["bounds"]["size_mm"]
    assert 10.0 < max(size) < 30.0  # a 12.7 mm lens, in millimetres (not metres)
    assert [b["id"] for b in summary["beams"]] == ["SRC#0"] and summary["beams"][0]["display_role"] == "PRIMARY"
    assert set(summary["camera_presets"]) == set(PRESET_ORDER)
    top = summary["camera_presets"]["top"]
    assert top["view_direction"] == [0.0, 0.0, 1.0] and top["up"] == [0.0, 1.0, 0.0]


def test_the_viewer_places_elements_in_the_design_frame_not_the_gltf_frame(runs):
    scene = optical_view_scene(runs["focus"])
    project = json.loads((runs["focus"] / "project.json").read_text(encoding="utf-8"))
    for component in project["components"]:
        part = next(p for p in scene.parts if p.part_id == component["instance_id"])
        lo, hi = part.bounds
        position = np.asarray(component["nominal_pose"]["position_mm"])
        assert np.linalg.norm((lo + hi) / 2 - position) < 30.0  # millimetres, +Z up: the element sits at its own pose


def test_weak_branches_are_secondary_and_main_beams_primary(runs):
    scene_doc = json.loads((runs["dichroic_routing"] / "optical_scene.json").read_text(encoding="utf-8"))
    roles = {p["path_id"]: (p["segment_roles"], p["display_role"]) for p in scene_doc["beam_paths"]}
    # 1064 nm: reflected by the dichroic (main), 5 % leaks through; 532 nm: transmitted (main), 10 % leaks by reflection
    assert roles["SRC_R#0"] == (["PRIMARY", "SECONDARY"], "SECONDARY")
    assert roles["SRC_R#0.0"] == (["PRIMARY"], "PRIMARY")
    assert roles["SRC_T#0"] == (["PRIMARY", "PRIMARY"], "PRIMARY")
    assert roles["SRC_T#0.0"] == (["SECONDARY"], "SECONDARY")
    gltf, _ = parse_glb((runs["dichroic_routing"] / "optical_scene.glb").read_bytes())
    names = {m["name"] for m in gltf["materials"]}
    assert any(n.startswith("beam_secondary_chief_") for n in names) and any(n.startswith("beam_chief_") for n in names)
    beam_nodes = [n for n in gltf["nodes"] if n.get("extras", {}).get("kind") == "BEAM_PATH"]
    assert {n["extras"]["display_role"] for n in beam_nodes} == {"PRIMARY", "SECONDARY"}
    elements = [n for n in gltf["nodes"] if n.get("extras", {}).get("kind") == "OPTICAL_ELEMENT"]
    assert {n["extras"]["part_class"] for n in elements} == {"SOURCE", "FILTER"}


def test_the_view_document_carries_exact_float32_geometry(runs):
    scene = optical_view_scene(runs["focus"])
    doc = view_document(scene)
    import base64

    for part, entry in zip(scene.parts, doc["parts"], strict=True):
        for mesh, m in zip(part.meshes, entry["meshes"], strict=True):
            positions = np.frombuffer(base64.b64decode(m["positions"]), dtype="<f4").reshape(-1, 3)
            assert np.allclose(positions, mesh.positions, atol=1e-3)
            indices = np.frombuffer(base64.b64decode(m["indices"]), dtype="<u4").reshape(-1, 3)
            assert np.array_equal(indices, mesh.triangles)


def test_a_directory_without_a_view_is_refused(tmp_path):
    with pytest.raises(ViewerInputError):
        optical_view_scene(tmp_path)


# ---------------------------------------------------------------------------------------------------------- assembly


def test_the_assembly_is_drawn_exactly_at_its_package_poses(preview):
    scene = assembly_view_scene(preview)
    parts = {p.part_ref: p for p in preview.scene.parts}
    assert {p.part_id for p in scene.parts} == set(parts)
    for view_part in scene.parts:
        part = parts[view_part.part_id]
        mesh = read_part_mesh(preview.files[part.geometry.asset_path])
        placed = np.asarray(Rigid.from_pose(part.pose_in_world.position_mm, part.pose_in_world.quaternion_wxyz).apply_all(mesh.vertices))
        lo, hi = view_part.bounds
        assert np.allclose(lo, placed.min(axis=0), atol=1e-6) and np.allclose(hi, placed.max(axis=0), atol=1e-6)
    classes = {p.part_id: p.part_class for p in scene.parts}
    assert classes["b01-optic"] == "OPTIC" and classes["b02-optic-mount"] == "MOUNT" and classes["b05-breadboard"] == "BREADBOARD"
    assert {m.marker_class for m in scene.markers} == {"CUSTOMER_SUPPLIED", "OBSERVATION"}


def test_the_assembly_summary_lists_chains_and_the_beam_axis(preview):
    summary = scene_summary(preview)
    assert summary["view_kind"] == "EXPERIMENTAL_LAB_ASSEMBLY" and summary["frame"] == "ASSEMBLY_WORLD"
    optic = summary["parts_by_class"]["OPTIC"][0]
    assert optic["carried_by"] == "b02-optic-mount" and optic["chain"] == "b04-post-holder-base"
    assert summary["beams"][0]["runs"][0]["role"] == "DESIGN_AXIS"
    assert all(len(p) == 3 for p in summary["beams"][0]["runs"][0]["points_mm"])


def test_view_assembly_serves_from_memory_and_writes_nothing(preview, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"view_assembly wrote {file!r}")
        return real_open(file, mode, *args, **kwargs)

    def no_write(*_args, **_kwargs):
        raise AssertionError("view_assembly wrote a file")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "write_bytes", no_write)
    monkeypatch.setattr(Path, "write_text", no_write)
    monkeypatch.setattr(os, "open", no_write)
    server = view_assembly(preview, open_browser=False)
    try:
        status, _, body = _get(server, server.url + "view.json")
        doc = json.loads(body)
        assert status == 200 and doc["view_kind"] == "EXPERIMENTAL_LAB_ASSEMBLY" and len(doc["parts"]) == len(preview.scene.parts)
    finally:
        server.close()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------------------- snapshots


def _background_fraction(image: np.ndarray) -> float:
    return float(np.mean(np.all(image == image[TOP_OFFSET_PX, -1], axis=2)))


TOP_OFFSET_PX = 200  # a background pixel away from the title band and the legend band


def test_snapshots_are_deterministic_non_empty_and_show_the_beam(runs):
    scene = optical_view_scene(runs["focus"])
    first = snapshot_images(scene, ("iso", "top"), **SMALL)
    second = snapshot_images(scene, ("iso", "top"), **SMALL)
    assert {k: v[0] for k, v in first.items()} == {k: v[0] for k, v in second.items()}
    beam = np.array(scene.beams[0].rgb) * 255.0
    for view, (data, drawn) in first.items():
        image = read_png_rgb(data)
        assert image.shape == (SMALL["height"], SMALL["width"], 3)
        assert _background_fraction(image) < 0.97, view  # something is drawn
        close = np.all(np.abs(image.astype(float) - beam) < 40.0, axis=2)
        assert close.sum() > 50, f"{view}: no beam-coloured pixels"
        assert drawn["scale_bar_mm"] > 0 and drawn["pixels_per_mm"] > 0


def test_top_snapshot_keeps_the_design_frame_orientation(runs):
    """Top view: +X to the right, +Y up the image. The focus design runs along +X, so the source is left of the lens."""

    scene = optical_view_scene(runs["focus"])
    from os_optics.viewer.raster import SUPERSAMPLE, make_camera

    cam = make_camera(scene, "top", **SMALL)
    src = next(p for p in scene.parts if p.part_id == "SRC")
    lens = next(p for p in scene.parts if p.part_id == "L1")
    x_src = cam.project(np.array([sum(src.bounds) / 2]))[0][0] / SUPERSAMPLE
    x_lens = cam.project(np.array([sum(lens.bounds) / 2]))[0][0] / SUPERSAMPLE
    assert x_src < x_lens
    assert list(cam.right) == [1.0, 0.0, 0.0] and list(cam.up) == [0.0, 1.0, 0.0]


def test_render_snapshots_writes_images_summary_and_legend(preview, tmp_path):
    written = render_snapshots(preview, tmp_path / "shots", views=("iso", "front"), **SMALL)
    assert sorted(p.name for p in (tmp_path / "shots").iterdir()) == ["front.png", "iso.png", "legend.md", "scene_summary.json"]
    summary = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert set(summary["snapshots"]) == {"iso.png", "front.png"} and summary["snapshots"]["iso.png"]["projection"] == "orthographic"
    legend = written["legend"].read_text(encoding="utf-8")
    assert "b01-optic" in legend and "units mm" in legend
    for name in ("iso.png", "front.png"):
        assert _background_fraction(read_png_rgb((tmp_path / "shots" / name).read_bytes())) < 0.97


def test_multi_chain_snapshot_renders(tmp_path):
    written = render_snapshots(load_assembly_preview(MULTI_PREVIEW), tmp_path, views=("top",), **SMALL)
    assert written["top"].stat().st_size > 1000


def test_unknown_views_and_sizes_are_refused(runs):
    scene = optical_view_scene(runs["focus"])
    with pytest.raises(ValueError):
        snapshot_images(scene, ("perspective",))
    with pytest.raises(ValueError):
        snapshot_images(scene, ("iso",), width=10, height=10)


# ------------------------------------------------------------------------------------------------------ page / server


def _get(server, url: str, *, method: str = "GET", host: str | None = None):
    path = url.split(f"{server.port}", 1)[1]
    connection = http.client.HTTPConnection(LOOPBACK, server.port, timeout=10)
    try:
        connection.putrequest(method, path, skip_host=host is not None)
        if host is not None:
            connection.putheader("Host", host)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_the_server_binds_loopback_under_an_unguessable_path(runs):
    first = view_run(runs["focus"], open_browser=False)
    second = view_run(runs["focus"], open_browser=False)
    try:
        for server in (first, second):
            assert server.host == LOOPBACK and server._httpd.server_address[0] == LOOPBACK
            token = server.url.rstrip("/").rsplit("/", 1)[1]
            assert len(token) >= TOKEN_BYTES and re.fullmatch(r"[A-Za-z0-9_-]+", token)
        assert first.url != second.url
        status, headers, body = _get(first, first.url)
        assert status == 200 and b"three.bundle.js" in body and headers["Cache-Control"] == "no-store"
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        for name in ("viewer.js", "viewer.css", "three.bundle.js", "view.json"):
            assert _get(first, first.url + name)[0] == 200
        base = f"http://{LOOPBACK}:{first.port}/"
        assert _get(first, base)[0] == 404
        assert _get(first, base + "view.json")[0] == 404
        assert _get(first, first.url + "../index.html")[0] == 404
        assert _get(first, first.url + "missing.js")[0] == 404
        assert _get(first, first.url, method="POST")[0] == 405
        assert _get(first, first.url, host="attacker.example:80")[0] == 403
    finally:
        first.close()
        second.close()


def test_the_standalone_page_is_self_contained(runs, tmp_path):
    assert main(["view", str(runs["focus"]), "--write-html", str(tmp_path / "view.html")]) == 0
    html = (tmp_path / "view.html").read_text(encoding="utf-8")
    assert 'id="opensci-view-document"' in html and "OPENSCI_THREE" in html
    assert not re.search(r"<(script|link)[^>]+(src|href)=\"(?!data:)", html)
    assert html.count("</script>") == html.count("<script")
    served = page_files(view_document(optical_view_scene(runs["focus"])))
    assert set(served) == {*ASSETS, "view.json"} and b"view.json" not in served["index.html"]


def test_cli_snapshot_and_errors(runs, tmp_path, capsys):
    assert main(["view", str(runs["dichroic_routing"]), "--snapshot", str(tmp_path / "s"), "--views", "iso", "--size", "400x300"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["view_kind"] == "OPTICAL_SIMULATION_VIEW" and (tmp_path / "s" / "iso.png").is_file()
    assert main(["view", str(tmp_path / "nothing-here")]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "VIEW_INPUT_INVALID"
    assert main(["view", str(runs["focus"]), "--snapshot", str(tmp_path / "t"), "--size", "big"]) == 64


# ------------------------------------------------------------------------------------------------ assets and wording


def test_the_vendored_three_bundle_is_the_pinned_release_with_its_licence():
    lock = json.loads((ROOT / "scripts/viewer-bundle/package-lock.json").read_text(encoding="utf-8"))
    pinned = lock["packages"]["node_modules/three"]["version"]
    head = asset_bytes("three.bundle.js")[:200].decode("utf-8")
    assert f"npm three@{pinned}" in head and "MIT License" in head
    licence = asset_bytes("THREE_LICENSE.txt").decode("utf-8")
    assert "The MIT License" in licence and "three.js authors" in licence


def test_a_beam_of_unknown_power_reads_not_known_and_a_root_beam_has_no_parent_row(tmp_path):
    """The panel prints every fact of a beam through valueText, which looks a value up as val_<value>: a null power needs its own words, and a root beam has no parent to state."""

    grating = {"model_id": "g", "model": {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "100", "orders": [0, 1],
                                          "clear_half_widths_mm": ["10", "10"], "edge_half_widths_mm": ["11", "11"], "status": "SPECIFIED"}}
    document = with_models(project([source("A", (-100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)],
                                   [plane("SCREEN", (100.0, 0.0, 0.0), radius_mm=60.0)]), [grating], design_intent="OPTICAL_SYSTEM")
    run_optical_project(document, catalog=None, output_dir=tmp_path / "run")
    beams = optical_view_scene(tmp_path / "run").beams
    orders = [b for b in beams if b.facts.get("parent_path_id") is not None]
    assert orders and all(b.facts["final_power_W"] is None for b in orders)
    assert [b for b in beams if "parent_path_id" not in b.facts]
    assert all(value is not None for b in beams for key, value in b.facts.items() if key != "final_power_W")
    source_text = asset_bytes("viewer.js").decode("utf-8")
    assert 'var key = "val_" + String(v);' in source_text and 'val_null: "not known"' in source_text and 'val_null: "未知"' in source_text


def test_the_viewer_ui_speaks_english_and_chinese_with_the_same_keys():
    source = asset_bytes("viewer.js").decode("utf-8")
    blocks = re.search(r"en: \{(.*?)\n    \},\n    \"zh-CN\": \{(.*?)\n    \}\n", source, re.S)
    assert blocks is not None
    key = re.compile(r"(?:^|[,{]\s*)([A-Za-z_]+):", re.M)
    en, zh = set(key.findall(blocks.group(1))), set(key.findall(blocks.group(2)))
    assert en == zh and {"iso", "top", "front", "side", "spacedBanner", "PRIMARY", "SECONDARY"} <= en


FORBIDDEN_WORDING = ("proxy", "reconstructed", "approximate", "not guaranteed", "not verified", "engineering review", "supplier",
                     "in this version", "not yet")


def test_viewer_text_states_product_facts_only():
    texts = [asset_bytes(name).decode("utf-8") for name in ("index.html", "viewer.js", "viewer.css")]
    texts += [p.read_text(encoding="utf-8") for p in (ROOT / "src/os_optics/viewer").glob("*.py")]
    texts.append((ROOT / "src/os_optics/cli_view.py").read_text(encoding="utf-8"))
    for text in texts:
        lowered = text.lower()
        assert not [w for w in FORBIDDEN_WORDING if w in lowered]


def test_a_written_run_says_how_to_open_its_3d_view(tmp_path):
    run = run_optical_project(example_project_of_kind("focus"), output_dir=tmp_path / "run")
    assert run.summary()["optical_view"]["view_command"].startswith("os-optics view <run directory>")
    assert "view_command" not in run_optical_project(example_project_of_kind("focus")).summary()["optical_view"]


def test_every_optical_surface_is_pickable_and_hideable_in_the_page(runs):
    """The view document names the optical surface every element mesh draws and lists each element's surfaces with their
    published facts; the page offers per-part and per-surface visibility and surface picking."""
    scene = optical_view_scene(runs["focus"])
    doc = view_document(scene)
    lenses = [p for p in doc["parts"] if p["class"] == "LENS"]
    assert lenses
    for part in lenses:
        ids = [s["surface_id"] for s in part["surfaces"]]
        assert ids and ids == sorted(ids)
        assert {m["surface"] for m in part["meshes"] if "surface" in m} == set(ids)
        for facts in part["surfaces"]:
            assert {"surface_id", "role", "clear_radius_mm", "medium_before", "medium_after", "chord_tolerance_mm"} <= set(facts)
        assert {m["material"] for m in part["meshes"]} >= {"optic_glass", "optic_rim", "aperture_outline"}
    source = asset_bytes("viewer.js").decode("utf-8")
    assert "hiddenParts" in source and "hiddenSurfaces" in source and "surfacePanel" in source
    assert 'id="part-legend"' in asset_bytes("index.html").decode("utf-8")
