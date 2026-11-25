"""The camera pose of the self-contained viewer page does not depend on the window size or the browser's device pixel ratio.

Needs a real browser (``playwright`` and a Chromium it can start); skipped where there is none, unless the environment
variable ``OPENSCI_BROWSER_TESTS`` is ``required`` (CI's ``engine-browser`` job, ``scripts/run_browser_tests.py --require``):
then a missing playwright or Chromium fails instead of skipping. The expected pose is the
Engine's own: ``Shot`` poses recorded by ``snapshot_images`` (closed form of ``camera_from_angles``), so the page (JavaScript)
and the snapshots (Python) are two independent computations of one convention. Chosen before running: the line of sight and
the screen-up of the page's camera equal the closed form to 1e-9 (and the recorded ``view_direction`` and ``up``, rounded to 6 decimals) at every combination of window size
(by loading the page at that size, and by resizing it afterwards) and device pixel ratio 1 and 2; the drawing buffer is the
CSS size times the pixel ratio; a stated eye distance fixes the eye position exactly (the page then does not refit); the
picture at pixel ratio 2, reduced to the CSS size, is the picture at pixel ratio 1.

    uv run --with playwright python -m pytest tests/test_viewer_browser_pose.py
    python scripts/run_browser_tests.py [--require]     # the same, reporting executed and skipped tests
"""

from __future__ import annotations

import base64
import math
import os
from pathlib import Path

import numpy as np
import pytest
from six_singlets import six_singlet_project

from os_optics.cli import main as cli_main
from os_optics.local import run_optical_project
from os_optics.viewer import Shot, optical_view_scene, snapshot_images

#: ``OPENSCI_BROWSER_TESTS=required``: a missing playwright or Chromium is a failure, not a skip.
REQUIRED = os.environ.get("OPENSCI_BROWSER_TESTS", "").strip().lower() == "required"
if REQUIRED:
    from playwright import sync_api  # (absent: the module fails to import, every test errors)
else:
    sync_api = pytest.importorskip("playwright.sync_api")

#: ``(angle to the axis, elevation, roll)`` of the eight cameras of the batch campaign.
ANGLES = ((86.0, 8.0, 0.0), (90.0, 0.0, 0.0), (90.0, 90.0, 0.0), (45.0, 25.0, 0.0), (135.0, -20.0, 0.0), (0.0, 0.0, 0.0),
          (60.0, 15.0, 30.0), (120.0, 35.0, -15.0))
#: ``(window width, height)`` in CSS pixels and the device pixel ratios the page is loaded at.
WINDOWS = ((1400, 900), (960, 720), (700, 500))
RATIOS = (1, 2)
POSE_TOLERANCE = 1e-9
RECORD_TOLERANCE = 2e-6  # the pose recorded in scene_summary.json is rounded to 6 decimals
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
PAGE_TIMEOUT_MS = 60_000
#: Mean absolute difference (of 255) between the ratio-1 picture and the ratio-2 picture reduced to the same size.
MAX_MEAN_DIFFERENCE = 6.0

READ_POSE = """() => {
  const v = window.__opensciViewer, c = v.camera, t = v.controls.target;
  c.updateMatrixWorld();
  const m = c.matrixWorld.elements, canvas = v.renderer.domElement;
  const dir = c.position.clone().sub(t).normalize();
  return {dir: dir.toArray(), up: [m[4], m[5], m[6]], target: t.toArray(), position: c.position.toArray(), kind: c.type,
          css: [v.width, v.height], buffer: [canvas.width, canvas.height], ratio: v.renderer.getPixelRatio(), dpr: window.devicePixelRatio,
          camera_up: c.up.toArray()};
}"""
SET_CAMERA = "(o) => { window.__opensciViewer.setCamera(o); window.__opensciViewer.renderNow(); }"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("six") / "run"
    run_optical_project(six_singlet_project()).write_artifacts(out)
    return out


@pytest.fixture(scope="module")
def page_url(run_dir, tmp_path_factory) -> str:
    html = tmp_path_factory.mktemp("page") / "view.html"
    assert cli_main(["view", str(run_dir), "--write-html", str(html)]) == 0
    return html.as_uri()


@pytest.fixture(scope="module")
def expected(run_dir) -> list[dict]:
    """The recorded pose of each of ``ANGLES``, from the Engine's snapshots (a pose does not depend on the image size)."""

    scene = optical_view_scene(run_dir)
    shots = [Shot(f"a{k}", angle_to_axis_deg=a, elevation_deg=e, roll_deg=r) for k, (a, e, r) in enumerate(ANGLES)]
    images = snapshot_images(scene, (), width=320, height=240, shots=shots)
    return [images[s.name][1]["camera"] for s in shots]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        # skipped only where playwright has no Chromium installed; an installed Chromium that fails to start (a flag or
        # renderer regression) fails the test
        if not Path(playwright.chromium.executable_path).is_file():
            missing = f"no Chromium for playwright at {playwright.chromium.executable_path}"
            if REQUIRED:
                pytest.fail(f"{missing} (OPENSCI_BROWSER_TESTS=required; install it: python -m playwright install chromium)")
            pytest.skip(missing)
        chromium = playwright.chromium.launch(args=CHROMIUM_ARGS)
        yield chromium
        chromium.close()


def _open(browser, url: str, size: tuple[int, int], ratio: int):
    context = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=ratio)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(url)
    page.wait_for_function("window.__opensciViewer !== undefined", timeout=PAGE_TIMEOUT_MS)
    return context, page, errors


def _closed_form(angle: float, elevation: float, roll: float) -> tuple[list[float], list[float]]:
    """The line of sight and screen-up for the six-singlet objective (axis +X, side -Y, lift +Z), by the convention's formulas."""

    a, e, r = (math.radians(v) for v in (angle, elevation, roll))
    direction = np.array([-math.cos(a), -math.cos(e) * math.sin(a), math.sin(e) * math.sin(a)])
    level = np.array([0.0, math.sin(e), math.cos(e)])
    up = level * math.cos(r) + np.cross(direction, level) * math.sin(r)
    return direction.tolist(), up.tolist()


def _assert_pose(got: dict, camera: dict, label: str, angles: tuple[float, float, float] | None = None) -> None:
    """The page's pose is the recorded one (to its rounding), and the closed form of the stated angles (to 1e-9)."""

    assert got["dir"] == pytest.approx(camera["view_direction"], abs=RECORD_TOLERANCE), label
    assert got["up"] == pytest.approx(camera["up"], abs=RECORD_TOLERANCE), label
    if angles is not None:
        direction, up = _closed_form(*angles)
        assert got["dir"] == pytest.approx(direction, abs=POSE_TOLERANCE), label
        assert got["up"] == pytest.approx(up, abs=POSE_TOLERANCE), label


def test_the_pose_is_the_same_in_every_window_size_and_pixel_ratio(browser, page_url, expected):
    seen: dict[tuple, dict] = {}
    for ratio in RATIOS:
        for size in WINDOWS:
            context, page, errors = _open(browser, page_url, size, ratio)
            try:
                for k, (angle, elevation, roll) in enumerate(ANGLES):
                    page.evaluate(SET_CAMERA, {"angle_to_axis_deg": angle, "elevation_deg": elevation, "roll_deg": roll})
                    got = page.evaluate(READ_POSE)
                    label = f"{angle},{elevation},{roll} at {size} x{ratio}"
                    _assert_pose(got, expected[k], label, ANGLES[k])
                    assert math.degrees(math.acos(-got["dir"][0])) == pytest.approx(angle, abs=1e-6), label  # angle to the axis (+X)
                    assert got["dpr"] == ratio and got["ratio"] == ratio and got["kind"] == "PerspectiveCamera"
                    assert got["buffer"] == [got["css"][0] * ratio, got["css"][1] * ratio], label
                    seen.setdefault((angle, elevation, roll), {})[(size, ratio)] = got["target"]
                assert not errors
            finally:
                context.close()
    for key, targets in seen.items():  # the framed centre is the same point everywhere
        first = next(iter(targets.values()))
        for target in targets.values():
            assert target == pytest.approx(first, abs=POSE_TOLERANCE), key


def test_the_address_camera_and_a_resize_keep_the_pose(browser, page_url, expected):
    """``#camera=86,8`` at load, then the window is resized: the line of sight, the screen-up and the centre stay."""

    poses = []
    for ratio in RATIOS:
        context, page, errors = _open(browser, page_url + "#camera=86,8", WINDOWS[0], ratio)
        try:
            for size in WINDOWS:
                page.set_viewport_size({"width": size[0], "height": size[1]})
                page.wait_for_function(f"window.__opensciViewer.width > 0 && Math.abs(window.__opensciViewer.renderer.domElement.width / {ratio} - window.__opensciViewer.width) < 1")
                got = page.evaluate(READ_POSE)
                _assert_pose(got, expected[0], f"resized to {size} x{ratio}", ANGLES[0])
                poses.append(got)
            assert not errors
        finally:
            context.close()
    for got in poses:
        assert got["target"] == pytest.approx(poses[0]["target"], abs=POSE_TOLERANCE)
        assert got["camera_up"] == pytest.approx(poses[0]["camera_up"], abs=POSE_TOLERANCE)


def test_a_stated_eye_distance_fixes_the_eye_position(browser, page_url, expected):
    """``#camera=86,8,0,DISTANCE_MM``: the eye is DISTANCE_MM from the centre on the line of sight in every window and ratio."""

    distance = 450.0
    eyes = []
    for ratio in RATIOS:
        for size in WINDOWS:
            context, page, _ = _open(browser, page_url + f"#camera=86,8,0,{distance:g}", size, ratio)
            try:
                got = page.evaluate(READ_POSE)
                _assert_pose(got, expected[0], f"{size} x{ratio}", ANGLES[0])
                offset = np.array(got["position"]) - np.array(got["target"])
                assert float(np.linalg.norm(offset)) == pytest.approx(distance, abs=1e-9)
                eyes.append(got["position"])
            finally:
                context.close()
    for eye in eyes:
        assert eye == pytest.approx(eyes[0], abs=POSE_TOLERANCE)


READ_PIXELS = """() => {
  const v = window.__opensciViewer; v.renderNow();
  const src = v.renderer.domElement, copy = document.createElement("canvas");
  copy.width = src.width; copy.height = src.height;
  const g = copy.getContext("2d"); g.drawImage(src, 0, 0);
  const data = g.getImageData(0, 0, copy.width, copy.height).data;
  let binary = ""; for (let i = 0; i < data.length; i += 8192) binary += String.fromCharCode.apply(null, data.subarray(i, i + 8192));
  return {width: copy.width, height: copy.height, base64: btoa(binary)};
}"""


def _picture(page) -> np.ndarray:
    """The drawing buffer of the page's canvas, (height, width, 4) uint8."""

    page.wait_for_timeout(300)
    got = page.evaluate(READ_PIXELS)
    return np.frombuffer(base64.b64decode(got["base64"]), dtype=np.uint8).reshape(got["height"], got["width"], 4)


def test_the_picture_at_ratio_2_is_the_picture_at_ratio_1_at_twice_the_pixels(browser, page_url):
    size = WINDOWS[1]
    pictures = {}
    for ratio in RATIOS:
        context, page, _ = _open(browser, page_url + "#camera=86,8", size, ratio)
        try:
            pictures[ratio] = _picture(page)
        finally:
            context.close()
    one, two = pictures[1].astype(float), pictures[2].astype(float)
    assert two.shape[:2] == (one.shape[0] * 2, one.shape[1] * 2)  # twice the pixels, the same CSS size
    reduced = two[: one.shape[0] * 2, : one.shape[1] * 2].reshape(one.shape[0], 2, one.shape[1], 2, -1).mean(axis=(1, 3))
    assert float(np.abs(reduced[:, :, :3] - one[:, :, :3]).mean()) < MAX_MEAN_DIFFERENCE
