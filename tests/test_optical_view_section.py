"""The exact section view (``os_optics.optical_view.section``) of a representative compound objective, against
closed-form references: every profile point is the exact spherical sag of the published radius, every chord between
two points stays within the chord tolerance (the circle's sagitta over the chord), the media polygons close at the
closed-form edge thickness, the clear-aperture ticks sit on the profile at the published clear radius, every element
on the axis sits at its design offset (turned around when it faces the other way), an element off the axis is listed
as excluded, and the Engine's chief ray in the plane is reproduced exactly. SVG / PNG / CLI / JSON API outputs."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal

import pytest

from catalog_store_helpers import REAL_CATALOG
from compound_objective import AGAINST_X, ALONG_X, lens, objective_parts, objective_project, project_of, source
from os_optics.local import load_public_catalog, run_optical_project
from os_optics.optical_view.section import optical_section, scene_section_elements, section_svg
from os_optics.optical_view.surface_mesh import CHORD_TOLERANCE_MM
from os_optics.viewer.png import read_png_rgb
from os_optics.viewer.section_render import section_png

ROUNDING = 1e-6  # the section's numbers carry 6 decimals


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


@pytest.fixture(scope="module")
def run(catalog):
    return run_optical_project(objective_project(catalog), catalog=REAL_CATALOG)


@pytest.fixture(scope="module")
def section(run):
    scene = run.optical_view
    return optical_section(scene_section_elements(scene), "L1", "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)


def _published(catalog, run):
    products = {e.element_id: catalog.get(e.optical_sku) for e in run.optical_view.elements if e.optical_sku}
    return {k: p.simulation_models[0] for k, p in products.items()}


def _sphere_z(vertex: float, radius: float | None, t: float) -> float:
    if radius is None:
        return vertex
    return vertex + radius - math.copysign(math.sqrt(radius * radius - t * t), radius)


def test_every_profile_point_is_exact_and_every_chord_within_the_tolerance(catalog, run, section):
    models = _published(catalog, run)
    assert [e["element_id"] for e in section["elements"]] == ["L1", "L2", "L3"] and section["excluded"] == []
    for element in section["elements"]:
        model = models[element["element_id"]]
        offset = element["axial_offset_mm"]
        for surface, published in zip(element["surfaces"], model.surfaces, strict=True):
            vertex = float(Decimal(published.vertex_z_mm))
            radius = None if published.radius_mm is None else float(Decimal(published.radius_mm))
            points = [p for seg in surface["segments"] for p in seg["points"]]
            ts = [t for t, _ in points]
            assert min(ts) == pytest.approx(-float(Decimal(model.edge_radius_mm))) and max(ts) == pytest.approx(float(Decimal(model.edge_radius_mm)))
            for t, z in points:
                assert z - offset == pytest.approx(_sphere_z(vertex, radius, t), abs=ROUNDING), (element["element_id"], surface["surface_id"])
            if radius is not None:
                for seg in surface["segments"]:
                    for (t0, z0), (t1, z1) in zip(seg["points"], seg["points"][1:]):
                        half = math.hypot(t1 - t0, z1 - z0) / 2.0
                        sagitta = abs(radius) - math.sqrt(radius * radius - half * half)
                        assert sagitta <= CHORD_TOLERANCE_MM + ROUNDING
            clear = float(Decimal(published.clear_radius_mm))
            assert surface["clear_aperture_mm"] == pytest.approx([-clear, clear])
            for t, z in surface["clear_aperture_ticks"]:
                assert abs(t) == pytest.approx(clear) and z - offset == pytest.approx(_sphere_z(vertex, radius, t), abs=ROUNDING)
            zones = [seg["zone"] for seg in surface["segments"]]
            assert zones == ["RIM", "CLEAR_APERTURE", "RIM"]


def test_the_media_close_at_the_closed_form_edge_thickness(catalog, run, section):
    models = _published(catalog, run)
    for element in section["elements"]:
        model = models[element["element_id"]]
        edge = float(Decimal(model.edge_radius_mm))
        glass = [k for k, m in enumerate(model.media) if m != "AIR"]
        assert [m["medium"] for m in element["media"]] == [model.media[k] for k in glass]
        for k, medium in zip(glass, element["media"], strict=True):
            front, back = model.surfaces[k], model.surfaces[k + 1]

            def z(surface, t):
                return _sphere_z(float(Decimal(surface.vertex_z_mm)), None if surface.radius_mm is None else float(Decimal(surface.radius_mm)), t)

            expected = z(back, edge) - z(front, edge)
            assert medium["edge_thickness_mm"] == pytest.approx([expected, expected], abs=ROUNDING)
            assert medium["centre_thickness_mm"] == pytest.approx(float(Decimal(back.vertex_z_mm) - Decimal(front.vertex_z_mm)))


def test_the_elements_sit_at_their_design_offsets_and_the_chief_ray_is_the_engines(run, section):
    assert [e["axial_offset_mm"] for e in section["elements"]] == [0.0, 40.0, 75.0]
    assert not any(e["flipped"] for e in section["elements"])
    scene = run.optical_view
    chief = [r for r in section["rays"] if r["role"] == "CHIEF"]
    assert len(chief) == len(scene.beam_paths)
    for ray, path in zip(chief, scene.beam_paths, strict=True):
        assert ray["max_out_of_plane_mm"] == 0.0  # the chief ray runs along the axis, in the plane
        # design frame: L1 at the origin with local +Z = +X and local +X = -Z (ALONG_X), so t = -z_design, z = x_design
        assert ray["points"] == [[pytest.approx(-p[2], abs=ROUNDING), pytest.approx(p[0], abs=ROUNDING)] for p in path.chief_ray.points_mm]
    zs = [z for e in section["elements"] for s in e["surfaces"] for seg in s["segments"] for _, z in seg["points"]]
    assert section["bounds"]["optics_z_mm"] == [min(zs), max(zs)]
    rays = [z for r in section["rays"] for _, z in r["points"]]
    assert section["bounds"]["z_mm"] == [min(zs + rays), max(zs + rays)]


def test_an_element_turned_around_is_flipped_and_one_off_the_axis_is_excluded(catalog):
    a, b, c = objective_parts(catalog)
    project = project_of([source(), lens("L1", a, (0.0, 0.0, 0.0)), lens("L2", b, (50.0, 0.0, 0.0), AGAINST_X),
                          lens("L3", c, (80.0, 1.0, 0.0))])
    from os_optics.local import json_api

    json_api.install_catalog(REAL_CATALOG.read_bytes())
    reply = json_api.optical_section(project.payload(), "L1", "XZ")
    assert reply["ok"]
    section = reply["section"]
    assert [e["element_id"] for e in section["elements"]] == ["L1", "L2"]
    assert section["excluded"] == [{"element_id": "L3", "reason": "NOT_ON_AXIS"}]
    l2 = section["elements"][1]
    assert l2["flipped"] and l2["axial_offset_mm"] == 50.0
    model = b.simulation_models[0]
    for surface, published in zip(l2["surfaces"], model.surfaces, strict=True):
        vertex = float(Decimal(published.vertex_z_mm))
        radius = None if published.radius_mm is None else float(Decimal(published.radius_mm))
        for seg in surface["segments"]:
            for t, z in seg["points"]:
                assert z == pytest.approx(50.0 - _sphere_z(vertex, radius, t), abs=ROUNDING)  # turned around: z = 50 - z_local
    assert section["rays"] == []  # a Project document: geometry only


def test_svg_png_cli_and_json_api_outputs(run, section, tmp_path, catalog):
    svg = section_svg(section)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") and "scale(" not in svg  # one scale, no stretching transform
    assert "true scale" in svg
    png = section_png(section)
    assert png == section_png(section)  # deterministic
    image = read_png_rgb(png)
    assert image.shape[1] == 1400 and image.shape[0] <= 760 and (image < 250).any()
    out = run.write_artifacts(tmp_path / "run")
    del out
    done = subprocess.run([sys.executable, "-m", "os_optics", "view", str(tmp_path / "run"), "--section", "L1,L3", "--section-plane", "YZ",
                           "--snapshot", str(tmp_path / "shots")], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    files = json.loads(done.stdout)["sections"]
    assert set(files) == {"L1", "L3"}
    l3 = json.loads((tmp_path / "shots" / "section-L3-YZ.json").read_text(encoding="utf-8"))
    assert l3["reference_element_id"] == "L3" and l3["plane"] == "YZ" and [e["element_id"] for e in l3["elements"]] == ["L1", "L2", "L3"]
    assert [e["axial_offset_mm"] for e in l3["elements"]] == [-75.0, -35.0, 0.0]
    bad = subprocess.run([sys.executable, "-m", "os_optics", "view", str(tmp_path / "run"), "--section", "SRC", "--snapshot", str(tmp_path / "x")],
                         capture_output=True, text=True, timeout=600)
    assert bad.returncode == 64 and "no optical body" in bad.stderr
    from os_optics.local import json_api

    from_run = json_api.optical_section(str(tmp_path / "run"), "L1", "XZ", svg=True)
    assert from_run["ok"] and from_run["section"] == section and from_run["svg"] == svg
    assert json_api.optical_section(str(tmp_path / "run"), "L1", "XY")["error"]["code"] == "SECTION_INVALID"
