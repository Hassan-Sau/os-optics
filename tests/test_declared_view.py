"""The Optical Simulation View of declared optical models: bound to the Engine's compilation of the declared model, drawn
as an ``OPTIC_BODY`` (``basis = PROJECT_DECLARED_MODEL``) or, for a ``COMPOUND`` unit, as ONE ``OPTICAL_UNIT`` element
whose members are child nodes of its GLB node; an aperture stop is its opaque plate around an open opening. Free text of
the Project (custom part descriptions, model display names) never reaches the scene or the GLB."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from os_optics import validate
from os_optics.contracts import AbsorberInteraction
from os_optics.local import lower_project, run_optical_project
from os_optics.optical_view import OpticalViewBindingError, build_optical_simulation_scene, parse_glb, read_accessor, scene_glb_bytes
from os_optics.viewer.optical import optical_scene_from

from declared_helpers import (
    compound_model,
    cylinder,
    declared_component,
    doublet_model,
    flint,
    nbk7,
    objective_model,
    plano_convex_model,
    stop_model,
    with_models,
)
from local_helpers import plane, project, source

PRIVATE_TEXT = "Private bench note 7731"


def _design(components: list[dict], models: list[dict]) -> dict:
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), *components], [plane("P", (90.0, 0.0, 0.0), radius_mm=20.0)])
    for component in document["components"][1:]:
        component["custom_part"]["description"] = PRIVATE_TEXT
    for model in models:
        model["display_name"] = PRIVATE_TEXT
    return with_models(document, models, [nbk7(), flint()])


def _unit_design() -> dict:
    unit = compound_model("unit", [("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)), ("L2", "pcx", (0, 0, 20.5), (1, 0, 0, 0)),
                                   ("STOP", "stop", (0, 0, 27.0), (1, 0, 0, 0))], cylinder("22", "-1", "30"))
    return _design([declared_component("OBJ", "lens", "unit")],
                   [doublet_model(), plano_convex_model(), stop_model(opening_radius_mm="8", edge_radius_mm="10"), unit])


@pytest.fixture(scope="module")
def unit_run():
    return run_optical_project(_unit_design())


def test_a_declared_model_is_drawn_as_its_optic_body():
    run = run_optical_project(_design([declared_component("OBJ", "lens", "objective")], [objective_model()]))
    element = next(e for e in run.optical_view.elements if e.element_id == "OBJ")
    body = element.geometry
    assert (element.optical_sku, element.optical_model_id, element.label) == (None, "objective", "objective")
    assert body.kind == "OPTIC_BODY" and body.basis.value == "PROJECT_DECLARED_MODEL" and body.category is None
    assert [s.surface_id for s in body.surfaces] == ["S1", "S2", "S3", "S4", "S5"] and body.media == ("N-BK7", "FLINT", "AIR", "N-BK7")
    # the derived body: the edge diameter, from the first vertex to the last surface's rim
    assert (body.diameter_mm, body.z_min_mm, body.z_max_mm) == (20.0, 0.0, 25.0)
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n["name"] == "element:OBJ"]
    assert node["extras"]["optical_model_id"] == "objective" and node["extras"]["part_class"] == "LENS"
    assert node["extras"]["geometry_basis"] == "PROJECT_DECLARED_MODEL" and "category" not in node["extras"]
    for prim in gltf["meshes"][node["mesh"]]["primitives"]:  # every drawn point lies in the derived body
        points = np.asarray(read_accessor(gltf, binary, prim["attributes"]["POSITION"]))
        assert np.all(np.hypot(points[:, 0], points[:, 1]) <= 10.0 + 1e-4) and np.all((points[:, 2] >= -1e-4) & (points[:, 2] <= 25.0 + 1e-4))


def test_a_compound_unit_is_one_element_with_its_member_bodies(unit_run):
    element = next(e for e in unit_run.optical_view.elements if e.element_id == "OBJ")
    unit = element.geometry
    assert unit.kind == "OPTICAL_UNIT" and element.optical_model_id == "unit"
    assert [(m.member_id, m.optical_model_id, m.body.model_kind) for m in unit.members] == [
        ("L1", "doublet", "SURFACE_SEQUENCE"), ("L2", "pcx", "SURFACE_SEQUENCE"), ("STOP", "stop", "APERTURE_STOP")]
    assert unit.members[2].pose.position_mm == (0.0, 0.0, 27.0) and (unit.diameter_mm, unit.z_max_mm) == (22.0, 30.0)
    assert unit.members[2].body.surfaces[0].clear_radius_mm == 8.0 and unit.members[2].body.z_min_mm == unit.members[2].body.z_max_mm == 0.0
    gltf, binary = parse_glb(scene_glb_bytes(unit_run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n["name"] == "element:OBJ"]
    assert "mesh" not in node and node["extras"]["members"] == ["L1", "L2", "STOP"]
    children = [gltf["nodes"][k] for k in node["children"]]
    assert [c["extras"]["member_id"] for c in children] == ["L1", "L2", "STOP"]
    assert children[2]["translation"] == [0.0, 0.0, 27.0] and children[2]["extras"]["model_kind"] == "APERTURE_STOP"
    # the viewer places the member meshes with the member pose, then the unit pose (along design +X here)
    parts = {p.part_id: p for p in optical_scene_from(unit_run.optical_view, gltf, binary).parts}
    points = np.concatenate([m.positions for m in parts["OBJ"].meshes])
    assert points[:, 0].min() == pytest.approx(0.0, abs=0.5) and points[:, 0].max() == pytest.approx(27.0, abs=1e-3)


def test_an_aperture_stop_is_drawn_as_its_plate_around_an_open_opening(unit_run):
    gltf, binary = parse_glb(scene_glb_bytes(unit_run.optical_view))
    (member,) = [n for n in gltf["nodes"] if n.get("extras", {}).get("kind") == "OPTICAL_UNIT_MEMBER" and n["extras"]["member_id"] == "STOP"]
    assert "mesh" not in member  # a stop has no body beyond its plane: its plate is its optical surface node
    (stop,) = [gltf["nodes"][c] for c in member["children"]]
    assert stop["name"].endswith("/STOP-stop") and stop["extras"]["role"] == "APERTURE_STOP"
    (prim,) = gltf["meshes"][stop["mesh"]]["primitives"]
    assert gltf["materials"][prim["material"]]["name"] == "stop_plate"
    points = np.asarray(read_accessor(gltf, binary, prim["attributes"]["POSITION"]))
    radii = np.hypot(points[:, 0], points[:, 1])
    assert radii.min() == pytest.approx(8.0, abs=1e-4) and radii.max() == pytest.approx(10.0, abs=1e-4)  # nothing inside the opening


def test_the_view_binds_a_declared_element_to_its_compiled_model():
    design = _design([declared_component("OBJ", "lens", "objective")], [objective_model()])
    lowered = lower_project(design)
    request = lowered.request
    (component,) = [c for c in request.project.components if c.instance_id == "OBJ"]
    first = component.surfaces[0]
    tampered = request.model_copy(update={"project": request.project.model_copy(update={"components": tuple(
        c.model_copy(update={"surfaces": (first.model_copy(update={"interaction": AbsorberInteraction()}), *c.surfaces[1:])})
        if c.instance_id == "OBJ" else c for c in request.project.components)})})
    from os_optics.local import load_project

    with pytest.raises(OpticalViewBindingError, match=r"model objective\): the Engine element is not its declared model: surface S1 differs"):
        build_optical_simulation_scene(project=load_project(design), catalog=None, engine_request=tampered, engine_result=validate(tampered))


def test_project_free_text_never_reaches_the_view(unit_run):
    assert PRIVATE_TEXT not in json.dumps(unit_run.optical_view.payload())
    assert PRIVATE_TEXT.encode() not in scene_glb_bytes(unit_run.optical_view)


def test_the_unit_scene_is_bound_and_deterministic(unit_run):
    again = run_optical_project(_unit_design())
    assert again.optical_view.content_hash() == unit_run.optical_view.content_hash()
    assert scene_glb_bytes(again.optical_view) == scene_glb_bytes(unit_run.optical_view)
    bounds = unit_run.optical_view.bounds
    assert bounds.max_mm[0] >= 27.0 and math.isfinite(bounds.min_mm[0])
