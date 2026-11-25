"""Render the Zemax-style lens prescriptions in the real viewer and save screenshots for a person to look at.

Development tooling only (the Engine core never starts a browser). For every fixture of
``tests/fixtures/prescriptions`` it builds the three construction routes, runs them with a real trace and renders the run:

* ``compound``: ``project import-prescription`` -> one ``COMPOUND`` unit;
* ``elements``: one ``SURFACE_SEQUENCE`` component per physical element, each with its own pose;
* ``groups``: one ``SURFACE_SEQUENCE`` per group between stops (air gaps inside the model), per-surface edge radii.

Rendering: ``os-optics view RUN --write-html`` in real Chrome, driven through the page's own buttons and a real mouse drag,
plus ``view --snapshot`` PNGs and ``--section`` output of the engine's own renderers. Nothing here reads the geometry back.

    uv run --no-sync --with playwright python scripts/lens_prescription_gallery.py --out DIR [--kinds a,b] [--routes compound]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ENGINE_ROOT / "tests" / "fixtures" / "prescriptions"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CHROME_ARGS = ["--headless=new", "--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--disable-gpu-sandbox"]
VIEWPORT = {"width": 1400, "height": 1000}
COMMAND_TIMEOUT_S = 240
#: Local +Z along world +X: the optical axis runs left to right, square-on to the viewer's +Z up.
AXIS_ALONG_X = [0.5, 0.5, 0.5, 0.5]
DEFAULT_IMAGE_DISTANCE_MM = 60.0
BEAM_SHARE_OF_CLEAR = 0.3
WAVELENGTH_NM = 587.5618
KINDS = ("lens_biconvex", "lens_plano_convex", "lens_convex_plano", "lens_positive_meniscus", "lens_negative_meniscus", "lens_biconcave",
         "lens_plano_concave", "lens_cemented_doublet", "lens_cemented_triplet", "lens_airspaced_doublet", "lens_even_asphere",
         "lens_near_ball", "lens_cylindrical", "lens_implausible_thick", "cooke_triplet", "double_gauss_6")
ROUTES = ("compound", "elements", "groups")
#: Screenshots of one kind: (file stem, preset button, mouse drag in pixels applied after it). With the optical axis along
#: world +X the "Front" preset looks square-on at the lens's side. About 2.6 px of drag per degree at this window size:
#: ``side_86`` is the owner's near-side view (the eye 86 deg off the axis, slightly above).
OVERVIEW_SHOTS = (
    ("side_86", "Front", (10, 20)),
    ("iso", "Iso", (0, 0)),
    ("top", "Top", (0, 0)),
    ("front", "Front", (0, 0)),
    ("along_axis", "Along beam", (0, 0)),
    ("oblique_a", "Iso", (160, -40)),
    ("oblique_b", "Iso", (-210, 90)),
)
#: The same, wheel-zoomed onto the lens with the source, plane and beams switched off in the legend.
CLOSEUP_SHOTS = (
    ("closeup_side_86", "Front", (10, 20)),
    ("closeup_three_quarter", "Front", (-95, 45)),
    ("closeup_oblique", "Front", (120, -60)),
    ("closeup_face", "Front", (190, 30)),
    ("closeup_iso", "Iso", (0, 0)),
    ("closeup_path_horizontal", "Path horizontal", (0, 0)),
    ("closeup_plane_xy", "XY plane", (0, 0)),  # the axis along world +X: square-on to the lens's side, orthographic
)
HIDDEN_IN_CLOSEUP = ("Light sources", "Observation planes", "Main beam", "Pupil rays", "Weak branch", "Beam (design axis)")
WHEEL_NOTCH = -240
#: One wheel notch of that size scales the eye's distance by about this much; a close-up fills this share of the window.
NOTCH_SCALE = 0.884
CLOSEUP_FILL = 0.62


@dataclass(frozen=True)
class Engine:
    """How to run the Engine CLI: this interpreter, ``-m os_optics``, from the Engine package."""

    python: str = sys.executable

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        done = subprocess.run([self.python, "-m", "os_optics", *args], capture_output=True, text=True, encoding="utf-8", cwd=ENGINE_ROOT,
                              timeout=COMMAND_TIMEOUT_S)
        if check and done.returncode != 0:
            raise RuntimeError(f"os-optics {' '.join(args)} exited {done.returncode}:\n{done.stdout[-3000:]}\n{done.stderr[-2000:]}")
        return done


# ---------------------------------------------------------------------------------------------------- the three routes


def _base_project(imported: dict) -> dict:
    """A Gaussian source and an observation plane on the world X axis, the unit's axial middle at the origin (so the viewer's
    framing, and its zoom, centres on the lens)."""

    clear = _clear_radius(imported)
    origin = _unit_origin_x(imported)
    image = origin + _image_z(imported)
    return {
        "schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
        "components": [{
            "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Gallery source."},
            "nominal_pose": {"position_mm": [-image, 0.0, 0.0], "quaternion_wxyz": AXIS_ALONG_X},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM, "power_w": 0.001,
                             "waist_radius_mm": round(clear * BEAM_SHARE_OF_CLEAR, 4), "waist_position_mm": 0.0, "m2": 1.0,
                             "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "image", "pose": {"position_mm": [image, 0.0, 0.0], "quaternion_wxyz": AXIS_ALONG_X},
                          "aperture": {"shape": "circle", "radius_mm": max(clear, 5.0)}}],
    }


def _members(imported: dict) -> list[dict]:
    """The imported members in axis order, each with its model."""

    models = {m["model_id"]: m for m in imported["optical_models"]}
    unit = models[imported["model_id"]]["model"]
    return [{"member_id": member["member_id"], "z": member["pose"]["position_mm"][2], "model": models[member["model_id"]]["model"],
             "model_id": member["model_id"], "display_name": models[member["model_id"]]["display_name"]} for member in unit["members"]]


def _clear_radius(imported: dict) -> float:
    return max(float(s["clear_radius_mm"]) for m in _members(imported) for s in m["model"].get("surfaces", ()) if "clear_radius_mm" in s)


def _length(imported: dict) -> float:
    """Axial length of the unit, first to last surface vertex."""

    return max(m["z"] + float(s["vertex_z_mm"]) for m in _members(imported) for s in m["model"].get("surfaces", ()))


def _unit_origin_x(imported: dict) -> float:
    return -_length(imported) / 2.0


def _image_z(imported: dict) -> float:
    z = imported["report"]["image_plane_z_mm"]
    return float(z) if z is not None and float(z) > 0.0 else DEFAULT_IMAGE_DISTANCE_MM


def _outline_radius(imported: dict) -> float:
    return max(float(m["model"]["edge_radius_mm"]) for m in _members(imported))


def project_compound(imported: dict) -> dict:
    project = _base_project(imported)
    project["optical_models"] = imported["optical_models"]
    project["model_materials"] = imported["model_materials"]
    project["components"].append({**imported["component"], "instance_id": "LENS", "nominal_pose": {"position_mm": [_unit_origin_x(imported), 0.0, 0.0], "quaternion_wxyz": AXIS_ALONG_X}})
    return project


def _component(instance_id: str, model_id: str, x_mm: float, role: str, description: str) -> dict:
    return {"instance_id": instance_id, "optical_role": role, "custom_part": {"custom_part_key": model_id, "description": description[:200]},
            "nominal_pose": {"position_mm": [x_mm, 0.0, 0.0], "quaternion_wxyz": AXIS_ALONG_X}, "optical_model_id": model_id}


def project_elements(imported: dict) -> dict:
    """One component per member (its model unchanged), each at its own pose."""

    project = _base_project(imported)
    project["optical_models"], project["model_materials"] = [], imported["model_materials"]
    origin = _unit_origin_x(imported)
    for member in _members(imported):
        project["optical_models"].append({"model_id": member["model_id"], "display_name": member["display_name"], "model": member["model"]})
        role = "aperture" if member["model"]["model_kind"] == "APERTURE_STOP" else "lens"
        project["components"].append(_component(member["member_id"], member["model_id"], origin + member["z"], role, member["display_name"]))
    return project


def _plain(value: Decimal) -> str:
    """A contract decimal string: digits and a point, no exponent, no trailing zeros."""

    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text


def _merged_sequence(run: list[dict]) -> dict:
    """Consecutive lens members as one ``SURFACE_SEQUENCE``: air gaps between them as media, every surface keeps its edge."""

    surfaces, media = [], []
    start = Decimal(repr(run[0]["z"]))
    outline = max(Decimal(m["model"]["edge_radius_mm"]) for m in run)
    for position, member in enumerate(run):
        model = member["model"]
        for surface in model["surfaces"]:
            vertex = Decimal(repr(member["z"])) - start + Decimal(surface["vertex_z_mm"])
            own = Decimal(surface.get("edge_radius_mm", model["edge_radius_mm"]))
            merged = {**surface, "surface_id": f"S{len(surfaces) + 1}", "vertex_z_mm": _plain(vertex)}
            merged.pop("edge_radius_mm", None)
            if own != outline:
                merged["edge_radius_mm"] = _plain(own)
            surfaces.append(merged)
        media += list(model["media"])
        if position < len(run) - 1:
            media.append("AIR")
    first = run[0]["model"]
    return {"model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": first["fidelity"], "surfaces": surfaces, "media": media,
            "edge_radius_mm": _plain(outline), "valid_band_nm": first["valid_band_nm"], "approximations": first["approximations"], "status": first["status"]}


def project_groups(imported: dict) -> dict:
    """One ``SURFACE_SEQUENCE`` per group: the lens members between two stops merged into one model."""

    project = _base_project(imported)
    project["optical_models"], project["model_materials"] = [], imported["model_materials"]
    origin = _unit_origin_x(imported)
    runs: list[list[dict]] = []
    current: list[dict] = []
    for member in _members(imported):
        if member["model"]["model_kind"] == "APERTURE_STOP":
            if current:
                runs.append(current)
                current = []
            runs.append([member])
        else:
            current.append(member)
    if current:
        runs.append(current)
    group = 0
    for run in runs:
        if run[0]["model"]["model_kind"] == "APERTURE_STOP":
            member = run[0]
            project["optical_models"].append({"model_id": member["model_id"], "display_name": member["display_name"], "model": member["model"]})
            project["components"].append(_component(member["member_id"], member["model_id"], origin + member["z"], "aperture", member["display_name"]))
            continue
        group += 1
        model_id = f"group-{group}"
        project["optical_models"].append({"model_id": model_id, "display_name": f"Group {group}", "model": _merged_sequence(run)})
        project["components"].append(_component(f"G{group}", model_id, origin + run[0]["z"], "lens", f"Group {group}"))
    return project


ROUTE_BUILDERS = {"compound": project_compound, "elements": project_elements, "groups": project_groups}


# ------------------------------------------------------------------------------------------------------------- rendering


def build_runs(engine: Engine, kind: str, routes: tuple[str, ...], out: Path) -> dict[str, Path]:
    """Import ``kind``, write and run every route; the run directories that exist afterwards (a refused route is recorded)."""

    work = out / kind
    work.mkdir(parents=True, exist_ok=True)
    done = engine.run("project", "import-prescription", str(FIXTURES / f"{kind}.csv"), "--model-id", "lens", check=False)
    (work / "import.json").write_text(done.stdout, encoding="utf-8")
    imported = json.loads(done.stdout)
    if not imported.get("ok"):
        raise RuntimeError(f"{kind}: import refused:\n{done.stdout[:3000]}")
    runs: dict[str, Path] = {}
    for route in routes:
        route_dir = work / route
        route_dir.mkdir(exist_ok=True)
        project_file = route_dir / "project.json"
        project_file.write_text(json.dumps(ROUTE_BUILDERS[route](imported), indent=1), encoding="utf-8")
        check = engine.run("project", "check", str(project_file), check=False)
        (route_dir / "check.json").write_text(check.stdout, encoding="utf-8")
        run_dir = route_dir / "run"
        run = engine.run("project", "run", str(project_file), "--out-dir", str(run_dir), "-o", str(route_dir / "result.json"), check=False)
        (route_dir / "run.log").write_text(f"exit {run.returncode}\n{run.stdout[-4000:]}\n{run.stderr[-4000:]}", encoding="utf-8")
        if not (run_dir / "optical_scene.glb").exists():
            print(f"  {kind}/{route}: no optical view (exit {run.returncode}); see {route_dir / 'run.log'}")
            continue
        print(f"  {kind}/{route}: run exit {run.returncode}")
        runs[route] = run_dir
    return runs


def engine_renders(engine: Engine, run_dir: Path, out: Path, section_ids: list[str]) -> None:
    """The Engine's own numpy snapshot PNGs and section drawings of one run."""

    views = "iso,top,front,side,beam_axis"
    engine.run("view", str(run_dir), "--snapshot", str(out / "snapshot"), "--views", views, "--size", "1000x750")
    if section_ids:
        try:
            done = engine.run("view", str(run_dir), "--snapshot", str(out / "section"), "--section", ",".join(section_ids), check=False)
        except subprocess.TimeoutExpired:
            print(f"  section of {section_ids}: stopped after {COMMAND_TIMEOUT_S} s")
            return
        if done.returncode != 0:
            print(f"  section of {section_ids}: exit {done.returncode}: {done.stderr[-400:]}")


def _shoot(page, canvas, centre, out: Path, shots, wheel: int) -> None:
    cx, cy = centre
    for stem, preset, drag in shots:
        page.get_by_role("button", name=preset, exact=True).click()
        page.wait_for_timeout(120)
        if drag != (0, 0):
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + drag[0], cy + drag[1], steps=12)
            page.mouse.up()
        if wheel:
            page.mouse.move(cx, cy)
            for _ in range(wheel):
                page.mouse.wheel(0, WHEEL_NOTCH)
                page.wait_for_timeout(25)
        page.mouse.move(2, 2)  # no hover tooltip over the picture
        page.wait_for_timeout(200)
        canvas.screenshot(path=str(out / f"{stem}.png"))


def browser_shots(page, html: Path, out: Path, closeup_notches: int) -> None:
    """Open the standalone page and take every shot through its visible controls."""

    out.mkdir(parents=True, exist_ok=True)
    page.goto(html.as_uri())
    page.wait_for_selector("#stage canvas")
    page.wait_for_timeout(400)
    labels = page.get_by_role("button", name="Labels").first
    labels.click()  # key parts -> all
    labels.click()  # all -> off: the shape is what is being looked at
    canvas = page.locator("#stage canvas")
    box = canvas.bounding_box()
    centre = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    _shoot(page, canvas, centre, out, OVERVIEW_SHOTS, 0)
    for name in HIDDEN_IN_CLOSEUP:
        boxes = page.get_by_label(name)
        for k in range(boxes.count()):
            boxes.nth(k).uncheck()
    _shoot(page, canvas, centre, out, CLOSEUP_SHOTS, closeup_notches)


def closeup_notches(imported: dict) -> int:
    """Wheel notches that bring the unit from the whole-scene framing to about ``CLOSEUP_FILL`` of the window."""

    span = 2.0 * (_unit_origin_x(imported) + _image_z(imported))
    size = 2.0 * max(_outline_radius(imported), _length(imported) / 2.0)
    factor = CLOSEUP_FILL * span / size
    return max(0, round(math.log(factor) / -math.log(NOTCH_SCALE))) if factor > 1.0 else 0


def take_screenshots(built: list[tuple[str, str, Path, int]]) -> None:
    from playwright.sync_api import sync_playwright  # only the screenshot step needs it

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=False, args=CHROME_ARGS)
        context = browser.new_context(viewport=VIEWPORT, locale="en-US", device_scale_factor=1)
        page = context.new_page()
        problems: list[str] = []
        page.on("pageerror", lambda exc: problems.append(str(exc)))
        for kind, route, html, notches in built:
            browser_shots(page, html, html.parent / "chrome", notches)
            print(f"  shots {kind}/{route}", flush=True)
        if problems:
            print("page errors:", problems[:5])
        browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for the runs and images (outside Git)")
    parser.add_argument("--kinds", default=",".join(KINDS), help="comma-separated fixture names")
    parser.add_argument("--routes", default=",".join(ROUTES), help="comma-separated routes")
    parser.add_argument("--no-browser", action="store_true", help="only the runs and the Engine's own renders (a machine without Chrome)")
    parser.add_argument("--shots-only", action="store_true", help="only the browser screenshots, of the view.html files already under --out")
    args = parser.parse_args(argv)
    out = Path(args.out)
    routes = tuple(r for r in args.routes.split(",") if r)
    kinds = [k for k in args.kinds.split(",") if k]
    built: list[tuple[str, str, Path, int]] = []
    if args.shots_only:
        for kind in kinds:
            for route in routes:
                html = out / kind / route / "view.html"
                if html.exists():
                    built.append((kind, route, html, closeup_notches(json.loads((out / kind / "import.json").read_text(encoding="utf-8")))))
    else:
        engine = Engine()
        for kind in kinds:
            print(kind, flush=True)
            for route, run_dir in build_runs(engine, kind, routes, out).items():
                view_dir = out / kind / route
                html = view_dir / "view.html"
                engine.run("view", str(run_dir), "--write-html", str(html))
                scene = json.loads((run_dir / "optical_scene.json").read_text(encoding="utf-8"))
                sections = [e["element_id"] for e in scene["elements"] if e["optical_role"] == "lens"][:1]
                engine_renders(engine, run_dir, view_dir, sections)
                built.append((kind, route, html, closeup_notches(json.loads((out / kind / "import.json").read_text(encoding="utf-8")))))
    if not args.no_browser and built:
        take_screenshots(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
