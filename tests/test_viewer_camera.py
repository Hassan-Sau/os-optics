"""Snapshots from a stated camera, framed on chosen parts, on a chosen background, in batches, and the pixel size they
record (``os-optics view --camera / --camera-file / --focus / --background / --size-mode``; ``viewer.model.camera_from_angles``).

References chosen before running: the camera direction is the closed form ``-a cos A + (s cos E + l sin E) sin A`` of the
convention (axis ``a`` = design +X for the six-singlet objective, ``s`` = -Y, ``l`` = +Z), so the angle measured back from
the recorded view direction is the requested one to 1e-9 degree; a roll of +90 degrees turns the picture clockwise, so the
axis then points down the screen; an orthographic image's pose does not depend on its pixel size; an image framed with
``fit`` is never larger than ``--size`` and fills it on one side; ``exact`` is ``--size``; a transparent image has alpha 0
where nothing is drawn and 255 under the title band.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from six_singlets import six_singlet_project

from os_optics.cli import main as cli_main
from os_optics.local import run_optical_project
from os_optics.viewer import (
    PartsFrame,
    Shot,
    axis_frame,
    camera_angles,
    camera_from_angles,
    focus_frame,
    optical_view_scene,
    render_snapshots,
    snapshot_images,
    view_document,
)
from os_optics.viewer.model import frame_points
from os_optics.viewer.png import COLOR_TYPE_RGBA, read_png
from os_optics.viewer.raster import FRAMED_OVERLAY_PX, MARGIN_PX, TOP_RESERVED_PX, _fit_line, _framed_size, render_view, text_size

ANGLE_TOLERANCE_DEG = 1e-9
SMALL = (480, 360)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("six") / "run"
    run_optical_project(six_singlet_project()).write_artifacts(out)
    return out


@pytest.fixture(scope="module")
def scene(run_dir):
    return optical_view_scene(run_dir)


def _view(*args: str, capsys) -> tuple[int, dict | None]:
    code = cli_main(["view", *args])
    out = capsys.readouterr().out
    return code, json.loads(out) if code == 0 else None


def _summary(directory: Path) -> dict:
    return json.loads((directory / "scene_summary.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ the convention


def test_the_camera_frame_is_the_main_beams_axis(scene):
    axis, side, lift = axis_frame(scene)
    assert axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert side == pytest.approx([0.0, -1.0, 0.0], abs=1e-12) and lift == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    doc = view_document(scene)
    assert doc["camera_frame"] == {"axis": [1.0, 0.0, 0.0], "side": [0.0, -1.0, 0.0], "lift": [0.0, 0.0, 1.0]}


@pytest.mark.parametrize("angle, elevation, roll", [(86.0, 8.0, 0.0), (30.0, -45.0, 12.5), (90.0, 0.0, 0.0), (150.0, 170.0, -30.0)])
def test_a_camera_by_angles_is_the_closed_form_and_measures_back(scene, angle, elevation, roll):
    direction, up = camera_from_angles(scene, angle, elevation, roll)
    a, e = math.radians(angle), math.radians(elevation)
    expected = np.array([-math.cos(a), -math.cos(e) * math.sin(a), math.sin(e) * math.sin(a)])
    assert direction == pytest.approx(expected, abs=1e-15)
    assert math.degrees(math.acos(-float(direction @ np.array([1.0, 0.0, 0.0])))) == pytest.approx(angle, abs=ANGLE_TOLERANCE_DEG)
    assert abs(float(direction @ up)) < 1e-15 and float(np.linalg.norm(up)) == pytest.approx(1.0, abs=1e-15)
    measured = camera_angles(scene, direction, up)
    assert (measured["angle_to_axis_deg"], measured["elevation_deg"], measured["roll_deg"]) == pytest.approx((angle, elevation, roll), abs=1e-6)


def test_at_roll_zero_the_axis_runs_level_and_a_positive_roll_turns_the_picture_clockwise(scene):
    axis = np.array([1.0, 0.0, 0.0])
    for angle, elevation in ((86.0, 8.0), (40.0, 25.0)):
        direction, up = camera_from_angles(scene, angle, elevation)
        right = np.cross(-direction, up)  # the rasteriser's screen right: forward x up
        assert abs(float(axis @ up)) < 1e-12 and float(axis @ right) > 0.0  # level, from left to right
    direction, up = camera_from_angles(scene, 90.0, 0.0, 90.0)
    assert float(axis @ up) == pytest.approx(-1.0, abs=1e-12)  # the axis now points down the screen


def test_the_square_on_camera_is_the_path_horizontal_preset(scene):
    _, path = render_view(scene, "path_horizontal", *SMALL)
    preset = path["camera"]
    assert (preset["angle_to_axis_deg"], preset["elevation_deg"], preset["roll_deg"]) == (90.0, 0.0, 0.0)
    _, top = render_view(scene, "top", *SMALL)
    assert (top["camera"]["angle_to_axis_deg"], top["camera"]["elevation_deg"]) == (90.0, 90.0)


# ------------------------------------------------------------------------------------------------ snapshots


def test_camera_86_8_is_86_degrees_to_the_axis_and_recorded(run_dir, tmp_path, capsys):
    code, doc = _view(str(run_dir), "--snapshot", str(tmp_path / "s"), "--camera", "86,8", "--size", "480x360", capsys=capsys)
    assert code == 0 and set(doc["files"]) == {"camera", "summary", "legend"}  # without --views only the camera image
    info = _summary(tmp_path / "s")["snapshots"]["camera.png"]
    camera = info["camera"]
    assert camera["angle_to_axis_deg"] == 86.0 and camera["elevation_deg"] == 8.0 and camera["roll_deg"] == 0.0
    v = np.array(info["view_direction"])
    assert math.degrees(math.acos(-v[0])) == pytest.approx(86.0, abs=ANGLE_TOLERANCE_DEG)
    assert camera["axis"] == [1.0, 0.0, 0.0] and camera["projection"] == "orthographic"
    assert np.linalg.norm(np.array(camera["eye_mm"]) - np.array(camera["target_mm"])) == pytest.approx(camera["distance_mm"], rel=1e-6)
    assert info["view"] == "camera" and "86 deg to the axis" in info["title"]
    assert "86 deg to the axis" in (tmp_path / "s" / "legend.md").read_text(encoding="utf-8")


def test_the_pose_does_not_depend_on_the_image_size(scene):
    shot = [Shot("side", angle_to_axis_deg=86.0, elevation_deg=8.0, roll_deg=5.0)]
    small = snapshot_images(scene, (), width=320, height=240, shots=shot)["side"][1]["camera"]
    large = snapshot_images(scene, (), width=900, height=500, shots=shot)["side"][1]["camera"]
    for key in ("view_direction", "up", "angle_to_axis_deg", "elevation_deg", "roll_deg", "distance_mm"):
        assert small[key] == large[key]


def test_a_camera_shot_is_deterministic(scene):
    shot = [Shot("a", angle_to_axis_deg=60.0, elevation_deg=20.0, roll_deg=10.0, distance="3.5x")]
    first = snapshot_images(scene, (), width=320, height=240, shots=shot, frame="optics")["a"]
    again = snapshot_images(scene, (), width=320, height=240, shots=shot, frame="optics")["a"]
    assert first[0] == again[0] and first[1] == again[1]
    points = frame_points(scene, "optics")
    size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    assert first[1]["camera"]["distance_mm"] == pytest.approx(3.5 * size, rel=1e-6)


def test_focus_frames_the_named_parts(scene):
    l2 = focus_frame(scene, "element:L2")
    assert l2 == PartsFrame(("L2",), "element:L2")
    assert np.array_equal(frame_points(scene, l2), np.concatenate([m.positions for p in scene.parts if p.part_id == "L2" for m in p.meshes]))
    image, drawn = render_view(scene, "front", 1280, 960, frame=l2)
    _, optics = render_view(scene, "front", 1280, 960, frame="optics")
    assert drawn["pixels_per_mm"] > 1.5 * optics["pixels_per_mm"]
    x0, y0, x1, y1 = drawn["content_box_px"]
    assert 0.0 <= x0 < x1 <= drawn["image_px"][0] and TOP_RESERVED_PX <= y0 < y1 <= drawn["image_px"][1]
    assert image.shape[:2] == (drawn["image_px"][1], drawn["image_px"][0])
    on_axis = focus_frame(scene, "axis:L1")
    assert on_axis.part_ids == ("L1", "L2", "L3", "STOP", "L4", "L5", "L6", "image")
    assert focus_frame(scene, "element:L2+image").part_ids == ("L2", "image")
    for bad in ("element:", "L2", "part:L2", "axis:L1+L2", "element:NOPE"):
        with pytest.raises(ValueError):
            focus_frame(scene, bad)


def test_a_transparent_background_writes_rgba_with_coverage_alpha(scene, tmp_path):
    written = render_snapshots(scene, tmp_path, views=("front",), width=480, height=360, frame="optics", background="transparent")
    data = written["front"].read_bytes()
    assert data[25] == COLOR_TYPE_RGBA  # the IHDR colour type
    image = read_png(data)
    info = _summary(tmp_path)["snapshots"]["front.png"]
    assert image.shape == (info["image_px"][1], info["image_px"][0], 4) and info["background"] == "transparent"
    x0, y0, x1, y1 = info["content_box_px"]
    assert float((image[:, :, 3] == 0).mean()) > 0.2  # where nothing is drawn
    assert image[2, 2, 3] == 255  # the title band
    assert image[int((y0 + y1) / 2), int((x0 + x1) / 2), 3] > 0  # the optics
    studio = render_view(scene, "front", 480, 360, frame="optics")[0]
    default = render_view(scene, "front", 480, 360, frame="optics", background="studio")[0]
    assert studio.shape[2] == 3 and np.array_equal(studio, default)


def test_a_flat_background_colour(scene):
    image, drawn = render_view(scene, "front", 480, 360, frame="optics", background=(1.0, 1.0, 1.0))
    assert drawn["background"] == "#ffffff"
    assert float((image == 255).all(axis=2).mean()) > 0.5


def test_a_title_wider_than_a_small_image_is_shortened_not_cut_by_the_edge(scene):
    """A camera title ("Optical Simulation View - CAMERA side: 86 deg to the axis, elevation 8 deg, orthographic") is longer
    than 800 px: it loses the scene's name, then ends in an ellipsis, and always lies inside the image (a 480 px image too)."""

    words = "CAMERA a1_side_86_8: 86 deg to the axis, elevation 8 deg"
    full = f"Optical Simulation View - {words}, orthographic"
    candidates = (full, f"Optical Simulation View - {words}", words)
    assert _fit_line(candidates, 5000.0) == full
    assert _fit_line(candidates, text_size(full)[0] - 1.0) == candidates[1]
    assert _fit_line(candidates, text_size(candidates[1])[0] - 1.0) == words
    for room in (text_size(words)[0] - 1.0, 300.0, 120.0):
        line = _fit_line(candidates, room)
        assert line.endswith("...") and text_size(line)[0] <= room and words.startswith(line[:-3].rstrip())
    shot = [Shot("side", angle_to_axis_deg=86.0, elevation_deg=8.0)]
    for width in (480, 800):
        image = read_png(snapshot_images(scene, (), width=width, height=360, shots=shot, frame="optics")["side"][0])
        band = image[: TOP_RESERVED_PX - 6, :, :3].astype(int)
        ink = np.nonzero((band.sum(axis=2) < 3 * 200).any(axis=0))[0]  # dark pixels of the title lines
        assert ink.size and ink.max() <= image.shape[1] - MARGIN_PX // 3  # clear of the right edge, like the left margin


# ------------------------------------------------------------------------------------------------ pixel size


def test_a_fitted_image_is_never_larger_than_the_requested_size(scene):
    """``ceil`` of a span times its own fill scale could give one pixel more than asked for."""

    for k in range(1, 120):
        box = (0.0, -1.0 - 0.137 * k, -1.0, 3.0 + 0.731 * k, 1.0, 1.0)
        for width, height in ((1920, 1080), (1280, 960), (777, 700)):
            w, h = _framed_size(scene, "front", box, width, height, FRAMED_OVERLAY_PX)
            assert w <= width and h <= height


@pytest.mark.parametrize("view, frame, fills", [("front", "optics", "width"), ("front", "element:L1", "height"), ("side", "element:L1", "height")])
def test_fit_and_exact_sizes_are_stated_and_true(scene, view, frame, fills):
    framing = focus_frame(scene, frame) if frame.startswith("element:") else frame
    image, fit = render_view(scene, view, 960, 540, frame=framing)
    w, h = fit["image_px"]
    assert image.shape[:2] == (h, w) and w <= 960 and h <= 540 and fit["requested_px"] == [960, 540]
    assert (w == 960) if fills == "width" else (h == 540)
    assert fit["size_mode"] == "fit"
    image, exact = render_view(scene, view, 960, 540, frame=framing, size_mode="exact")
    assert image.shape[:2] == (540, 960) and exact["image_px"] == [960, 540] and exact["size_mode"] == "exact"
    x0, _, x1, _ = exact["content_box_px"]
    assert (x0 + x1) / 2.0 == pytest.approx(480.0, abs=1.0)  # the fitted content is centred
    with pytest.raises(ValueError, match="size mode"):
        render_view(scene, view, 480, 360, frame=framing, size_mode="stretch")


def test_the_cli_records_image_px_and_writes_a_fixed_canvas_on_request(run_dir, tmp_path, capsys):
    code, _ = _view(str(run_dir), "--snapshot", str(tmp_path / "fit"), "--views", "front", "--frame", "optics", "--size", "960x540", capsys=capsys)
    assert code == 0
    info = _summary(tmp_path / "fit")["snapshots"]["front.png"]
    image = read_png((tmp_path / "fit" / "front.png").read_bytes())
    assert [image.shape[1], image.shape[0]] == info["image_px"] == [info["width_px"], info["height_px"]]
    w, h = info["image_px"]
    assert w <= 960 and h <= 540 and (w == 960 or h == 540) and {"pixels_per_mm", "scale_bar_mm", "content_box_px"} <= set(info)
    code, _ = _view(str(run_dir), "--snapshot", str(tmp_path / "exact"), "--views", "front", "--frame", "optics", "--size", "960x540",
                    "--size-mode", "exact", capsys=capsys)
    assert code == 0
    assert read_png((tmp_path / "exact" / "front.png").read_bytes()).shape[:2] == (540, 960)
    assert _summary(tmp_path / "exact")["snapshots"]["front.png"]["image_px"] == [960, 540]


# ------------------------------------------------------------------------------------------------ batches and refusals


def test_one_command_captures_several_runs_with_a_camera_file(run_dir, tmp_path, capsys):
    runs = []
    for name in ("designA", "designB"):  # two run directories of their own names
        target = tmp_path / name
        target.mkdir()
        for f in run_dir.iterdir():
            if f.is_file():
                (target / f.name).write_bytes(f.read_bytes())
        runs.append(str(target))
    shots = tmp_path / "shots.json"
    shots.write_text(json.dumps([["side86", 86, 8], ["low", 30, -10, 5, "2x"], {"name": "top", "view": "top"},
                                 {"name": "l2", "angle_to_axis_deg": 70, "elevation_deg": 15, "focus": "element:L2"}]), encoding="utf-8")
    code, doc = _view(*runs, "--snapshot", str(tmp_path / "out"), "--camera-file", str(shots), "--size", "320x240", "--frame", "optics",
                      capsys=capsys)
    assert code == 0 and set(doc["runs"]) == {"designA", "designB"}
    for name in ("designA", "designB"):
        assert set(doc["runs"][name]["files"]) == {"side86", "low", "top", "l2", "summary", "legend"}
        snapshots = _summary(tmp_path / "out" / name)["snapshots"]
        assert snapshots["side86.png"]["camera"]["angle_to_axis_deg"] == 86.0
        assert snapshots["low.png"]["camera"]["roll_deg"] == 5.0 and snapshots["low.png"]["frame"] == "optics"
        assert snapshots["top.png"]["view"] == "top" and snapshots["l2.png"]["frame"] == "element:L2"
    assert (tmp_path / "out" / "designA" / "side86.png").read_bytes() == (tmp_path / "out" / "designB" / "side86.png").read_bytes()


@pytest.mark.parametrize("args, needle", [
    (["--camera", "86,8"], "--camera applies to --snapshot"),
    (["--focus", "element:L2"], "--focus applies"),
    (["--background", "transparent"], "--background applies"),
    (["--snapshot", "S", "--camera", "200,0"], "0..180"),
    (["--snapshot", "S", "--camera", "86"], "ANGLE,ELEVATION"),
    (["--snapshot", "S", "--camera", "86,8,0,-3"], "positive"),
    (["--snapshot", "S", "--background", "#12345"], "#rrggbb"),
    (["--snapshot", "S", "--focus", "element:NOPE"], "no part NOPE"),
    (["--snapshot", "S", "--focus", "element:L2", "--frame", "optics"], "not both"),
])
def test_usage_errors(run_dir, tmp_path, capsys, args, needle):
    args = [str(tmp_path / "s") if a == "S" else a for a in args]
    assert cli_main(["view", str(run_dir), *args]) == 64
    assert needle in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli_main(["view", str(run_dir), "--snapshot", str(tmp_path / "s"), "--size-mode", "stretch"])


@pytest.mark.parametrize("entries, needle", [
    ([], "non-empty JSON list"),
    ([["a", 86, 8], ["a", 40, 0]], "more than once"),
    ([["../x", 86, 8]], "not file names"),
    ([{"name": "a", "angle": 3}], "unknown field"),
    ([{"name": "a", "view": "top", "elevation_deg": 3}], "takes no angles"),
    ([{"name": "a", "view": "sideways"}], "unknown view"),
    ([{"name": "a", "angle_to_axis_deg": 80, "focus": "element:L2", "frame": "optics"}], "not both"),
    ([[86, 8]], "starts with the image name"),
])
def test_camera_file_refusals(run_dir, tmp_path, capsys, entries, needle):
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    assert cli_main(["view", str(run_dir), "--snapshot", str(tmp_path / "s"), "--camera-file", str(path)]) == 64
    assert needle in capsys.readouterr().err
    assert not (tmp_path / "s").exists()  # nothing is written before every camera is known


def test_several_runs_need_a_snapshot_directory_and_distinct_names(run_dir, tmp_path, capsys):
    assert cli_main(["view", str(run_dir), str(run_dir)]) == 64
    assert cli_main(["view", str(run_dir), str(run_dir), "--snapshot", str(tmp_path / "s")]) == 64
    assert "same name" in capsys.readouterr().err


def test_the_page_takes_the_camera_from_its_address():
    from os_optics.viewer.page import asset_bytes

    source = asset_bytes("viewer.js").decode("utf-8")
    for needle in ("Viewer.prototype.setCamera", "Viewer.prototype.cameraFromAngles", "Viewer.prototype.applyHash",
                   "#camera=ANGLE,ELEVATION", "window.__opensciViewer = viewer", "camera_frame"):
        assert needle in source
