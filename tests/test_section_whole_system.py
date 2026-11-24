"""The 2D layout of a lens: ``view --section ALL`` and ``--frame``, on a six-element objective assembled as six
``SURFACE_SEQUENCE`` components, and the refusal of ``layout draw`` without an Optical Layout Package.

References chosen before running: the profiles are the closed-form sphere within the section's rounding (1e-6 mm); the
image's scale is measured from its own pixels (the first lens's front vertex to the image plane, at the design's
distance, within 1.5 px; the scale bar's length within 1.5 px); a box's image scale is the closed-form fit of that box."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from six_singlets import CSV, ELEMENTS, observation, six_singlet_project

from os_optics.cli import main as cli_main
from os_optics.local import json_api, run_optical_project
from os_optics.optical_layout import OpticalLayoutError, OpticalLayoutErrorCode, load_layout_package
from os_optics.optical_view.section import (
    MAX_RAY_LUMINANCE,
    SVG_BOTTOM_PX,
    SVG_MARGIN_PX,
    SVG_TOP_PX,
    SectionError,
    _bounds,
    _legible,
    ray_key,
    ray_paints,
    scene_section_elements,
    section_layout,
    section_svg,
    system_sections,
)
from os_optics.viewer import optical_view_scene
from os_optics.viewer.png import read_png_rgb
from os_optics.viewer.raster import clip_long_segments, render_view
from os_optics.viewer.section_render import section_png

ROUNDING = 1e-6
PIXEL_TOLERANCE = 1.5
IMAGE_X_MM = 140.1
L1_R1_MM = 54.0
L1_EDGE_MM = 26.0
LIGHT_ON_WHITE = 250


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("six") / "run"
    run_optical_project(six_singlet_project()).write_artifacts(out)
    return out


@pytest.fixture(scope="module")
def scene(run_dir):
    return optical_view_scene(run_dir).source_scene


@pytest.fixture(scope="module")
def whole(scene):
    (section,) = system_sections(scene_section_elements(scene), "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)
    return section


def _view(run_dir: Path, *args: str, capsys) -> tuple[int, dict | None]:
    code = cli_main(["view", str(run_dir), *args])
    out = capsys.readouterr().out
    return code, json.loads(out) if code == 0 else None


def _orange_columns(image: np.ndarray) -> np.ndarray:
    return np.nonzero(((image[:, :, 0] > 170) & (image[:, :, 1] < 120) & (image[:, :, 2] < 60)).sum(axis=0) > 20)[0]


def test_the_section_of_all_is_one_drawing_of_the_six_elements_the_stop_and_the_image_plane(whole):
    assert [e["element_id"] for e in whole["elements"]] == ["L1", "L2", "L3", "STOP", "L4", "L5", "L6"]
    assert {e["model_kind"] for e in whole["elements"] if e["element_id"] in ELEMENTS} == {"SURFACE_SEQUENCE"}
    assert whole["elements"][3]["model_kind"] == "APERTURE_STOP" and whole["excluded"] == []
    assert whole["scope"] == "ALL" and whole["reference_element_id"] == "L1"
    assert [(p["observation_id"], p["z_mm"], p["t_mm"]) for p in whole["observation_planes"]] == [("image", IMAGE_X_MM, [-25.0, 25.0])]


def test_a_lens_is_its_true_curved_profile_not_a_plate(whole):
    l1 = whole["elements"][0]
    assert "plate" not in l1 and "substrate" not in l1
    for t, z in [p for seg in l1["surfaces"][0]["segments"] for p in seg["points"]]:
        assert z == pytest.approx(L1_R1_MM - math.sqrt(L1_R1_MM ** 2 - t * t), abs=ROUNDING)
    front = [z for t, z in l1["media"][0]["polygon"] if abs(t) == L1_EDGE_MM]
    assert max(front) - min(front) > 1.0  # the edge of the glass is not its centre thickness: the outline curves


def test_a_thin_lens_is_its_symbol_not_a_rectangle():
    for focal, outward in (("50", True), ("-25", False)):
        built = json_api.build_native_model("lens.ideal_thin", {"focal_length_mm": focal, "clear_aperture_diameter_mm": "20"},
                                            {"into": six_singlet_project(), "instance_id": "THIN", "position_mm": [-50.0, 0.0, 0.0],
                                             "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]})
        assert built["ok"], built
        section = json_api.optical_section(built["project"], "THIN", "XZ")["section"]
        thin = next(e for e in section["elements"] if e["element_id"] == "THIN")
        assert "plate" not in thin and len(thin["symbol"]) == 2
        tip_t = abs(thin["symbol"][0][1][0])
        foot_t = abs(thin["symbol"][0][0][0])
        assert (tip_t > foot_t) is outward
        assert section_svg(section).count("stroke-width") > 0


def test_the_png_has_the_design_distances_at_one_scale_and_a_readable_scale_bar(whole):
    layout = section_layout(whole)
    image = read_png_rgb(section_png(whole))
    assert image.shape[:2] == (layout.height, layout.width)
    # true scale, measured from the pixels: from the first lens's front vertex on the axis to the image plane
    _, axis_y = layout.to_px(0.0, 0.0)
    row = image[int(round(axis_y)) - 8]  # just off the axis line and the chief ray
    glass = np.nonzero((row[:, 2] > row[:, 0] + 10) & (row[:, 2] < LIGHT_ON_WHITE))[0]
    vertex = glass[0]
    plane = _orange_columns(image)
    assert plane.size and abs((plane.mean() - vertex) - IMAGE_X_MM * layout.scale) <= PIXEL_TOLERANCE, (plane.mean(), vertex, IMAGE_X_MM * layout.scale)
    # the scale bar
    y = layout.height - 30
    dark = np.nonzero(image[y].sum(axis=1) < 200)[0]
    assert dark.size and abs((dark.max() - dark.min()) - layout.bar_mm * layout.scale) <= PIXEL_TOLERANCE, (dark.min(), dark.max(), layout.bar_mm * layout.scale)
    assert layout.bar_mm == 20.0
    # the optics fill the frame: they span nearly all of the image's width, nothing cuts them above or below
    assert IMAGE_X_MM * layout.scale > 0.85 * layout.width
    assert layout.to_px(0.0, L1_EDGE_MM)[1] >= SVG_TOP_PX
    assert layout.to_px(0.0, -L1_EDGE_MM)[1] <= layout.height - SVG_BOTTOM_PX


def test_every_label_is_in_the_svg_and_labels_do_not_overlap(whole):
    svg = section_svg(whole)
    labels = re.findall(r'font-size="14"[^>]*>([^<]+)<', svg)
    assert sorted(labels) == sorted([*ELEMENTS, "STOP", "image"])
    assert "of the system on the axis of L1" in svg


def test_the_cli_writes_one_section_for_all_with_its_numbers(run_dir, tmp_path, capsys):
    code, doc = _view(run_dir, "--section", "ALL", "--snapshot", str(tmp_path / "a"), capsys=capsys)
    assert code == 0
    assert set(doc["sections"]) == {"ALL"}
    entry = doc["sections"]["ALL"]
    assert entry["elements"] == ["L1", "L2", "L3", "STOP", "L4", "L5", "L6"] and entry["observation_planes"] == ["image"]
    assert (tmp_path / "a" / "section-ALL-XZ.png").is_file() and (tmp_path / "a" / "section-ALL-XZ.svg").is_file()
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["section-ALL-XZ.json", "section-ALL-XZ.png", "section-ALL-XZ.svg"]
    saved = json.loads((tmp_path / "a" / "section-ALL-XZ.json").read_text(encoding="utf-8"))
    assert saved["reference_element_id"] == "L1"


def test_section_takes_repeated_flags_and_comma_lists(run_dir, tmp_path, capsys):
    code, doc = _view(run_dir, "--section", "L1", "--section", "L3,L6", "--snapshot", str(tmp_path / "r"), capsys=capsys)
    assert code == 0 and set(doc["sections"]) == {"L1", "L3", "L6"}
    assert doc["sections"]["L6"]["reference_element_id"] == "L6"


def test_the_default_section_frame_is_the_optics_not_the_source(whole):
    source_z = whole["bounds"]["z_mm"][0]
    assert source_z == -200.0
    default, scene_wide = section_layout(whole), section_layout(whole, frame="scene")
    assert default.box[0] > -10.0 and scene_wide.box[0] < -200.0
    assert default.scale > 2.0 * scene_wide.scale
    assert scene_wide.bar_mm > default.bar_mm
    assert scene_wide.height < default.height  # the aspect follows the box


def test_an_explicit_box_sets_the_image_aspect_and_scale(whole):
    # design frame: L1's z is +X, its t is +Y; the box's Z has no bearing on this section
    box = (30.0, -10.0, -30.0, 70.0, 10.0, 30.0)
    layout = section_layout(whole, 1400, 760, box)
    assert layout.box == pytest.approx((30.0, 70.0, -10.0, 10.0))
    top = SVG_TOP_PX + 34
    expected = min((1400 - 2 * SVG_MARGIN_PX) / 40.0, (760 - top - SVG_BOTTOM_PX) / 20.0)
    assert layout.scale == pytest.approx(expected)
    assert layout.to_px(30.0, 10.0)[1] == pytest.approx(top, abs=1.0)
    assert layout.bar_mm <= (layout.width - 2 * SVG_MARGIN_PX) / layout.scale / 2.0
    assert section_png(whole, frame=box) == section_png(whole, frame=box)
    with pytest.raises(SectionError, match="no extent"):
        section_layout(whole, frame=(30.0, 5.0, 0.0, 70.0, 5.0, 0.0))


def test_the_frame_flag_on_sections_and_snapshots(run_dir, tmp_path, capsys):
    code, doc = _view(run_dir, "--section", "ALL", "--frame=30,-10,-10,70,10,10", "--snapshot", str(tmp_path / "b"), capsys=capsys)
    assert code == 0
    entry = doc["sections"]["ALL"]
    assert entry["scale_bar_mm"] == 5.0 and entry["pixels_per_mm"] > 20.0
    assert _view(run_dir, "--section", "ALL", "--frame", "wide", "--snapshot", str(tmp_path / "c"), capsys=capsys)[0] == 64
    assert _view(run_dir, "--section", "ALL", "--frame=1,2,3", "--snapshot", str(tmp_path / "c"), capsys=capsys)[0] == 64
    assert _view(run_dir, "--section", "ALL", "--frame=9,0,0,1,1,1", "--snapshot", str(tmp_path / "c"), capsys=capsys)[0] == 64
    assert _view(run_dir, "--section", "ALL", "--frame=9,0,0,9,0,0", "--snapshot", str(tmp_path / "c"), capsys=capsys)[0] == 64
    assert cli_main(["view", str(run_dir), "--frame", "optics"]) == 64
    capsys.readouterr()
    code, doc = _view(run_dir, "--views", "front", "--frame", "optics", "--snapshot", str(tmp_path / "s"), capsys=capsys)
    assert code == 0
    summary = json.loads((tmp_path / "s" / "scene_summary.json").read_text(encoding="utf-8"))
    info = summary["snapshots"]["front.png"]
    image = read_png_rgb((tmp_path / "s" / "front.png").read_bytes())
    assert image.shape[:2] == (info["height_px"], info["width_px"])
    assert info["height_px"] < 960  # the image takes the box's aspect within --size


def test_snapshot_frames_fit_what_they_name(run_dir):
    view = optical_view_scene(run_dir)
    _, default = render_view(view, "front", 1280, 960)
    _, optics = render_view(view, "front", 1280, 960, frame="optics")
    _, everything = render_view(view, "front", 1280, 960, frame="scene")
    # the optics: L1's front vertex (x = 0) to the image plane (x = 140.1) across the usable width
    assert optics["pixels_per_mm"] == pytest.approx((1280 - 72) / IMAGE_X_MM, rel=1e-3)
    assert default["pixels_per_mm"] < optics["pixels_per_mm"] / 2.0
    assert everything["pixels_per_mm"] <= default["pixels_per_mm"] * (1.0 + 1e-9)
    assert optics["scale_bar_mm"] < default["scale_bar_mm"]
    image, box = render_view(view, "front", 1280, 960, frame=(30.0, -10.0, -10.0, 70.0, 10.0, 10.0))
    assert box["pixels_per_mm"] > optics["pixels_per_mm"] and image.shape[0] == box["height_px"]
    with pytest.raises(ValueError, match="no extent"):
        render_view(view, "front", 1280, 960, frame=(30.0, 0.0, 5.0, 70.0, 0.0, 5.0))


def test_a_far_plane_stays_out_of_the_optics_frame_and_is_named(tmp_path, capsys):
    project = six_singlet_project(extra_observations=[observation("far", 3000.0)])
    run_optical_project(project).write_artifacts(tmp_path / "run")
    code, doc = _view(tmp_path / "run", "--section", "ALL", "--snapshot", str(tmp_path / "f"), capsys=capsys)
    assert code == 0
    entry = doc["sections"]["ALL"]
    assert [p["observation_id"] for p in entry["beyond_frame"]] == ["far"] and entry["pixels_per_mm"] > 8.0
    svg = (tmp_path / "f" / "section-ALL-XZ.svg").read_text(encoding="utf-8")
    assert "beyond this frame: far (" in svg
    code, doc = _view(tmp_path / "run", "--section", "ALL", "--frame", "scene", "--snapshot", str(tmp_path / "g"), capsys=capsys)
    assert doc["sections"]["ALL"]["beyond_frame"] == [] and doc["sections"]["ALL"]["pixels_per_mm"] < 1.0


def test_a_plane_off_the_section_is_listed_and_an_image_height_widens_the_frame():
    elements = [{"element_id": "L", "surfaces": [{"segments": [{"zone": "CLEAR_APERTURE", "points": [[-5.0, 0.0], [5.0, 0.0]]}]}], "media": []}]
    rays = [{"points": [[0.0, -10.0], [8.0, 10.0], [12.0, 20.0]], "display_role": "PRIMARY"}]
    bounds = _bounds(elements, rays, [{"observation_id": "p", "z_mm": 10.0, "t_mm": [-20.0, 20.0]}])
    assert bounds["frame_t_mm"] == [-5.0, 8.0] and bounds["frame_z_mm"] == [0.0, 10.0]
    outside = _bounds(elements, rays, [{"observation_id": "p", "z_mm": 10.0, "t_mm": [-6.0, 6.0]}])  # the ray misses the plane's aperture
    assert outside["frame_t_mm"] == [-5.0, 5.0]
    project = six_singlet_project(extra_observations=[{**observation("aside", 60.0), "pose": {"position_mm": [60.0, 0.0, 40.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}])
    scene = run_optical_project(project).optical_view
    (section,) = system_sections(scene_section_elements(scene), "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)
    assert {"element_id": "aside", "reason": "NOT_ON_AXIS"} in section["excluded"]
    assert "not in this section: aside (not on axis)" in section_svg(section)


def test_all_is_referred_to_the_element_the_light_reaches_first_whatever_the_listing_order():
    project = six_singlet_project(order=["L6", "L4", "STOP", "L2", "L1", "L3", "L5"])
    scene = run_optical_project(project).optical_view
    (section,) = system_sections(scene_section_elements(scene), "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)
    assert section["reference_element_id"] == "L1"
    assert [e["element_id"] for e in sorted(section["elements"], key=lambda e: e["axial_offset_mm"])][0] == "L1"


def test_a_system_on_two_axes_gives_one_section_per_axis(tmp_path, capsys):
    project = six_singlet_project()
    aside = next(c for c in project["components"] if c["instance_id"] == "L6")
    aside["nominal_pose"]["position_mm"] = [67.3, 60.0, 0.0]
    reply = json_api.optical_section(project, "ALL", "XZ", svg=True)
    assert reply["ok"] and [e["element_id"] for e in reply["section"]["elements"]] == ["L1", "L2", "L3", "STOP", "L4", "L5"]
    assert [[e["element_id"] for e in o["section"]["elements"]] for o in reply["other_axes"]] == [["L6"]]
    assert "svg" in reply["other_axes"][0] and "L6 (not on axis)" in reply["svg"]
    assert json_api.optical_section(six_singlet_project(), "ALL", "XZ")["ok"] and "other_axes" not in json_api.optical_section(six_singlet_project(), "ALL", "XZ")


def test_a_compound_import_gives_the_same_picture(tmp_path, capsys):
    project = six_singlet_project()
    into = {k: v for k, v in project.items() if k not in ("optical_models", "model_materials", "model_coatings")}
    into["components"] = [c for c in project["components"] if c["optical_role"] == "source"]
    imported = json_api.import_prescription(CSV.read_text(encoding="utf-8"), {"model_id": "gauss", "into": into, "instance_id": "OBJ",
                                                                              "pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}})
    assert imported["ok"], imported
    run_optical_project(imported["project"]).write_artifacts(tmp_path / "run")
    code, doc = _view(tmp_path / "run", "--section", "ALL", "--snapshot", str(tmp_path / "s"), capsys=capsys)
    assert code == 0 and doc["sections"]["ALL"]["reference_element_id"] == "OBJ"
    assert [e.split("/")[1] for e in doc["sections"]["ALL"]["elements"]] == ["L1", "L2", "L3", "STOP", "L4", "L5", "L6"]
    labels = re.findall(r'font-size="14"[^>]*>([^<]+)<', (tmp_path / "s" / "section-ALL-XZ.svg").read_text(encoding="utf-8"))
    assert sorted(labels) == sorted([*ELEMENTS, "STOP", "image"])


def test_a_light_ray_colour_is_darkened_and_fields_are_coloured_by_source():
    yellow = (1.0, 0.9, 0.0)
    r, g, b = _legible(yellow)
    assert 0.2126 * r + 0.7152 * g + 0.0722 * b == pytest.approx(MAX_RAY_LUMINANCE) and r / g == pytest.approx(1.0 / 0.9)
    assert _legible((0.0, 0.0, 0.5)) == (0.0, 0.0, 0.5)
    project = six_singlet_project((0.0, 4.0))
    scene = run_optical_project(project).optical_view
    (section,) = system_sections(scene_section_elements(scene), "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)
    assert [s for s, _ in ray_key(section)] == ["FIELD0", "FIELD1"]
    paints = ray_paints(section)
    colours = {r["source_element_id"]: p.rgb for r, p in zip(section["rays"], paints, strict=True)}
    assert colours["FIELD0"] != colours["FIELD1"]
    chief = next(p for r, p in zip(section["rays"], paints, strict=True) if r["role"] == "CHIEF")
    pale = [p for r, p in zip(section["rays"], paints, strict=True) if r["role"] == "PUPIL"]
    assert chief.opacity == 1.0 and len({p.opacity for p in pale}) == 2
    assert ">FIELD1<" in section_svg(section)


def test_a_tight_frame_samples_only_what_shows():
    p0 = np.array([[-1e7, 50.0, 0.0], [10.0, 10.0, 1.0], [-1e7, -500.0, 0.0]])
    p1 = np.array([[1e7, 50.0, 2.0], [20.0, 20.0, 1.0], [1e7, -500.0, 0.0]])
    a, b = clip_long_segments(p0, p1, 100.0, 100.0)
    assert a[0] == pytest.approx([0.0, 50.0, 1.0], abs=1e-3) and b[0] == pytest.approx([100.0, 50.0, 1.0], abs=1e-3)
    assert (a[1] == p0[1]).all() and (b[1] == p1[1]).all()  # short: untouched
    assert (a[2] == b[2]).all()  # misses the canvas


@pytest.mark.parametrize("args", [[], ["{missing}", "--out", "{out}"], ["{run}", "--out", "{out}"], ["{project}", "--out", "{out}"]])
def test_layout_draw_without_a_package_names_it_and_what_produces_it(args, run_dir, tmp_path, capsys):
    project = tmp_path / "project.json"
    project.write_text(json.dumps(six_singlet_project()), encoding="utf-8")
    names = {"{missing}": str(tmp_path / "nothing.json"), "{run}": str(run_dir), "{project}": str(project), "{out}": str(tmp_path / "out")}
    assert cli_main(["layout", "draw", *[names.get(a, a) for a in args]]) == 65
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "OPTICAL_LAYOUT_PACKAGE_REQUIRED"
    text = " ".join(p["message"] for p in error["problems"])
    assert "Optical Layout Package (optical-layout.json)" in text and "order layout" in text and "resolution layout" in text
    assert "--section ALL" in text
    if args and args[0] in ("{missing}", "{run}"):
        assert names[args[0]] in text
    assert not (tmp_path / "out").exists()


def test_layout_refusals_at_the_loader_and_usage():
    with pytest.raises(OpticalLayoutError) as caught:
        load_layout_package(Path("no-such-file.json"))
    assert caught.value.code is OpticalLayoutErrorCode.PACKAGE_REQUIRED
    assert json_api.generate_optical_layout({"schema": "opensci.project"})["error"]["code"] == "OPTICAL_LAYOUT_PACKAGE_REQUIRED"
    assert cli_main(["layout", "draw", "something.json"]) == 64  # --out is still asked for


def test_the_help_names_the_requirement_next_to_the_section_view(capsys):
    for argv in (["view", "--help"], ["layout", "draw", "--help"]):
        with pytest.raises(SystemExit):
            cli_main(argv)
        text = " ".join(capsys.readouterr().out.split())
        assert "--section ALL --snapshot DIR" in text and "Optical Layout Package" in text
