"""The local viewer's legend shows what is drawn.

* a compound unit is listed as a compound unit whose swatches are its members' drawn colours (lens glass, stop plate),
  an aperture stop as an aperture in the stop plate's colour -- not both as grey "other optical elements";
* the beam legend's reference-axis entry is one grey dashed swatch, as the axis is drawn;
* the parts list names sources and planes by their class in the page's language (no English label in zh-CN).
"""

from __future__ import annotations

import re

import pytest

from declared_helpers import compound_model, cylinder, declared_component, doublet_model, flint, nbk7, plano_convex_model, stop_model, with_models
from local_helpers import plane, project, source
from os_optics.local import run_optical_project
from os_optics.optical_view import parse_glb, scene_glb_bytes
from os_optics.optical_view.glb import STATIC_MATERIALS
from os_optics import viewer
from os_optics.viewer.model import view_document
from os_optics.viewer.optical import optical_scene_from
from os_optics.viewer.raster import _legend_rows

ASSETS = __import__("pathlib").Path(viewer.__file__).parent / "assets"


@pytest.fixture(scope="module")
def scene():
    unit = compound_model("unit", [("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)), ("L2", "pcx", (0, 0, 20.5), (1, 0, 0, 0)),
                                   ("STOP", "stop", (0, 0, 27.0), (1, 0, 0, 0))], cylinder("22", "-1", "30"))
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), declared_component("OBJ", "lens", "unit"),
                        declared_component("A1", "aperture", "iris", position=(60.0, 0.0, 0.0))], [plane("P", (90.0, 0.0, 0.0), radius_mm=20.0)])
    document = with_models(document, [doublet_model(), plano_convex_model(), stop_model(opening_radius_mm="8", edge_radius_mm="10"),
                                      stop_model("iris", opening_radius_mm="3", edge_radius_mm="12.7"), unit], [nbk7(), flint()])
    run = run_optical_project(document)
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    return optical_scene_from(run.optical_view, gltf, binary)


def _colour(material: str) -> list[float]:
    return [round(c, 4) for c in STATIC_MATERIALS[material][0]]


def test_a_unit_and_a_stop_are_listed_in_the_colours_they_are_drawn_in(scene):
    classes = {c["key"]: c for c in view_document(scene)["classes"]}
    assert "OTHER_OPTICAL" not in classes
    parts = {p.part_id: p.part_class for p in scene.parts}
    assert (parts["OBJ"], parts["A1"]) == ("OPTICAL_UNIT", "APERTURE")
    drawn_unit = {tuple(m.rgb) for p in scene.parts if p.part_id == "OBJ" for m in p.meshes if m.material in ("optic_glass", "stop_plate")}
    swatches = {tuple(round(c, 4) for c in rgb) for rgb in classes["OPTICAL_UNIT"]["swatches"]}
    assert swatches == {tuple(_colour("optic_glass")), tuple(_colour("stop_plate"))} and swatches == {tuple(round(c, 4) for c in rgb) for rgb in drawn_unit}
    stop_drawn = {tuple(m.rgb) for p in scene.parts if p.part_id == "A1" for m in p.meshes}
    assert stop_drawn == {tuple(round(c, 4) for c in classes["APERTURE"]["rgb"])}
    # the snapshot legend uses the same colours
    rows = {label: colours for label, colours, kind in _legend_rows(scene) if kind == "boxes"}
    assert {tuple(round(c, 4) for c in rgb) for rgb in rows["Compound optical units"]} == swatches


def test_the_page_draws_the_reference_axis_swatch_grey_and_dashed_and_localizes_the_parts_list():
    script = (ASSETS / "viewer.js").read_text(encoding="utf-8")
    # the reference axis is drawn 0x737884 (the palette's REFERENCE colour) and dashed; its legend swatch is the same
    assert re.search(r'role === "REFERENCE"\) \{\s*// the reference axis is drawn grey and dashed', script)
    assert 'el("span", "swatch line dashed")' in script and "styles.REFERENCE.rgb" in script
    # sources and planes in the parts list: their class in the page's language
    for key, zh in (("part_SOURCE", "光源"), ("part_OBSERVATION_PLANE", "观测面"), ("part_TARGET_PLANE", "目标面")):
        assert f'{key}: "{zh}"' in script
    assert 't("part_" + p["class"])' in script
