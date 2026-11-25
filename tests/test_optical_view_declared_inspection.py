"""True-scale inspection and optical-solid validity of declared optical models and compound units.

* every member of a ``COMPOUND`` unit carries its own optical-surface nodes (``surface:<unit>/<member>-<surface>``) whose
  vertices lie on the member's declared profile; an aperture stop is its plate around the open opening plus its central
  obstruction; an ideal thin lens is one plane (clear aperture, band, rim);
* the exact section through a whole unit places every member at its member pose along the unit axis and draws each
  surface at its closed-form sag; the stop's opening is open;
* the lowering refuses overlapping members and a member outside its unit's envelope (structured, localized), and keeps
  valid units valid; a touching stop is a contact, not an error;
* a chief ray that falls on a central obstruction is drawn as the beam's reference axis behind it (scene, GLB, local
  viewer, section), while the annulus's pupil rays continue.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from declared_helpers import (
    compound_model,
    cylinder,
    declared_component,
    doublet_model,
    flint,
    nbk7,
    plano_convex_model,
    stop_model,
    thin_lens_model,
    with_models,
)
from local_helpers import plane, project, source
from os_optics.local import LocalRunError, LocalRunErrorCode, json_api, lower_project, run_optical_project
from os_optics.optical_view import parse_glb, read_accessor, scene_glb_bytes
from os_optics.optical_view.section import optical_section, scene_section_elements
from os_optics.viewer.optical import optical_scene_from

UNIT_MEMBERS = [("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)), ("L2", "pcx", (0, 0, 20.5), (1, 0, 0, 0)), ("STOP", "stop", (0, 0, 27.0), (1, 0, 0, 0))]


def _design(components: list[dict], models: list[dict], *, plane_x: float = 90.0, waist: float | None = None) -> dict:
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), *components], [plane("P", (plane_x, 0.0, 0.0), radius_mm=20.0)])
    if waist is not None:
        document["components"][0]["source_model"]["waist_radius_mm"] = waist
    return with_models(document, models, [nbk7(), flint()])


def _unit_design(members=UNIT_MEMBERS, envelope=("22", "-1", "30"), stop: dict | None = None) -> dict:
    unit = compound_model("unit", members, cylinder(*envelope))
    return _design([declared_component("OBJ", "lens", "unit")],
                   [doublet_model(), plano_convex_model(), stop if stop is not None else stop_model(opening_radius_mm="8", edge_radius_mm="10"), unit])


@pytest.fixture(scope="module")
def unit_run():
    return run_optical_project(_unit_design())


def _sag(radius: float | None, rho: float) -> float:
    return 0.0 if radius is None else radius - math.copysign(math.sqrt(radius * radius - rho * rho), radius)


def test_every_member_surface_is_its_own_node_on_its_declared_profile(unit_run):
    gltf, binary = parse_glb(scene_glb_bytes(unit_run.optical_view))
    unit = next(e for e in unit_run.optical_view.elements if e.element_id == "OBJ")
    members = {m.member_id: m for m in unit.geometry.members}
    surfaces = [n for n in gltf["nodes"] if n.get("extras", {}).get("kind") == "OPTICAL_SURFACE"]
    names = sorted(n["name"] for n in surfaces)
    assert names == sorted(f"surface:OBJ/{m}-{s.surface_id}" for m, member in members.items() for s in member.body.surfaces)
    for node in surfaces:
        extras = node["extras"]
        member = members[extras["member_id"]]
        assert extras["engine_surface_id"] == f"{extras['member_id']}-{extras['surface_id']}"
        surface = next(s for s in member.body.surfaces if s.surface_id == extras["surface_id"])
        for prim in gltf["meshes"][node["mesh"]]["primitives"]:
            for x, y, z in read_accessor(gltf, binary, prim["attributes"]["POSITION"]):  # member frame (node below the member node)
                assert z == pytest.approx(surface.vertex_z_mm + _sag(surface.radius_mm, math.hypot(x, y)), abs=2e-5)
    # the local viewer lists the unit's surfaces under their Engine ids and keeps them pickable
    scene = optical_scene_from(unit_run.optical_view, gltf, binary)
    part = next(p for p in scene.parts if p.part_id == "OBJ")
    assert {f["surface_id"] for f in part.surfaces} == {f"{m}-{s.surface_id}" for m, member in members.items() for s in member.body.surfaces}
    assert {m.surface_id for m in part.meshes if m.surface_id} == {f["surface_id"] for f in part.surfaces}


def test_a_stop_is_its_plate_around_the_open_opening_and_its_obstruction():
    stop = stop_model(opening_radius_mm="6", obstruction_radius_mm="2", edge_radius_mm="10")
    run = run_optical_project(_design([declared_component("S", "aperture", "stop", position=(0.0, 0.0, 0.0))], [stop]))
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n.get("name") == "surface:S/stop"]
    radii = []
    for prim in gltf["meshes"][node["mesh"]]["primitives"]:
        assert gltf["materials"][prim["material"]]["name"] == "stop_plate"
        radii += [math.hypot(x, y) for x, y, _ in read_accessor(gltf, binary, prim["attributes"]["POSITION"])]
    radii = np.asarray(radii)
    assert radii.max() == pytest.approx(10.0, abs=1e-5)
    assert not ((radii > 2.0 + 1e-5) & (radii < 6.0 - 1e-5)).any()  # the open ring between obstruction and opening edge
    assert (np.abs(radii - 2.0) < 1e-5).any() and (np.abs(radii - 6.0) < 1e-5).any()


def test_a_thin_lens_is_one_plane_with_its_clear_aperture_and_rim():
    run = run_optical_project(_design([declared_component("T", "lens", "thin", position=(0.0, 0.0, 0.0))], [thin_lens_model(clear="10", edge="12.7")]))
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n.get("name") == "surface:T/lens"]
    assert node["extras"]["focal_length_mm"] == 100.0 and node["extras"]["role"] == "THIN_LENS"
    materials = {gltf["materials"][p["material"]]["name"]: read_accessor(gltf, binary, p["attributes"]["POSITION"]) for p in gltf["meshes"][node["mesh"]]["primitives"]}
    assert set(materials) == {"thin_lens", "aperture_outline", "optic_rim"}
    assert {round(z, 9) for points in materials.values() for _, _, z in points} == {0.0}  # zero thickness
    assert max(math.hypot(x, y) for x, y, _ in materials["thin_lens"]) == pytest.approx(10.0, abs=1e-5)


def test_the_section_of_a_unit_places_every_member_at_its_pose(unit_run):
    scene = unit_run.optical_view
    section = optical_section(scene_section_elements(scene), "OBJ", "XZ", beam_paths=scene.beam_paths)
    assert [(e["element_id"], e["axial_offset_mm"]) for e in section["elements"]] == [("OBJ/L1", 0.0), ("OBJ/L2", 20.5), ("OBJ/STOP", 27.0)]
    assert {e["member_id"] for e in section["elements"]} == {"L1", "L2", "STOP"}
    unit = next(e for e in scene.elements if e.element_id == "OBJ")
    for entry in section["elements"]:
        body = next(m for m in unit.geometry.members if m.member_id == entry["member_id"]).body
        for surface, drawn in zip(body.surfaces, entry["surfaces"], strict=True):
            for seg in drawn["segments"]:
                for t, z in seg["points"]:
                    assert z - entry["axial_offset_mm"] == pytest.approx(surface.vertex_z_mm + _sag(surface.radius_mm, abs(t)), abs=1e-6)
    stop = next(e for e in section["elements"] if e["member_id"] == "STOP")
    plate = [t for seg in stop["surfaces"][0]["segments"] for t, _ in seg["points"]]
    assert all(abs(t) >= 8.0 - 1e-9 for t in plate)  # the opening is open
    reply = json_api.optical_section(_unit_design(), "OBJ", "XZ")
    assert reply["ok"] and [e["element_id"] for e in reply["section"]["elements"]] == ["OBJ/L1", "OBJ/L2", "OBJ/STOP"]


def test_the_lowering_checks_the_members_of_a_unit():
    assert lower_project(_unit_design()).solids.valid
    overlapping = copy.deepcopy(UNIT_MEMBERS)
    overlapping[1] = ("L2", "pcx", (0, 0, 7.0), (1, 0, 0, 0))  # into the doublet's glass
    with pytest.raises(LocalRunError) as refused:
        lower_project(_unit_design(overlapping))
    assert [(p.code, p.location) for p in refused.value.problems] == [(LocalRunErrorCode.OPTICAL_BODIES_OVERLAP, "component OBJ member L1")]
    with pytest.raises(LocalRunError) as outside:
        lower_project(_unit_design(envelope=("22", "-1", "24")))
    assert {(p.code, p.location) for p in outside.value.problems} == {(LocalRunErrorCode.MEMBER_OUTSIDE_ENVELOPE, "component OBJ member L2"),
                                                                       (LocalRunErrorCode.MEMBER_OUTSIDE_ENVELOPE, "component OBJ member STOP")}
    report = json_api.inspect_optical_solids(_unit_design(overlapping))  # the full report of an invalid design
    assert report["ok"] and report["status"] == "INVALID" and report["issues"][0]["member_id"] == "L1"
    assert json_api.check_project(_unit_design())["optical_solids"]["status"] == "VALID"


def test_an_obstructed_chief_ray_is_the_reference_axis_behind_the_obstruction():
    stop = stop_model(opening_radius_mm="6", obstruction_radius_mm="2", edge_radius_mm="10")
    run = run_optical_project(_design([declared_component("S", "aperture", "stop", position=(0.0, 0.0, 0.0))], [stop], waist=3.0))
    (path,) = run.optical_view.beam_paths
    k = path.chief_reference_from
    assert k is not None and path.chief_ray.points_mm[k][0] == pytest.approx(0.0, abs=1e-9)  # from the stop plane on
    assert all(len(r.points_mm) > k + 1 for r in path.pupil_rays)  # the annulus continues
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    names = {gltf["materials"][p["material"]]["name"] for n in gltf["nodes"] if n.get("name", "").startswith("beam:")
             for p in gltf["meshes"][n["mesh"]]["primitives"]}
    assert "beam_reference_axis" in names
    beam = optical_scene_from(run.optical_view, gltf, binary).beams[0]
    roles = [role for role, _ in beam.runs]
    assert roles[0] == "PRIMARY" and "REFERENCE" in roles
    reference = next(pts for role, pts in beam.runs if role == "REFERENCE")
    assert reference[0][0] == pytest.approx(0.0, abs=1e-6)
    section = optical_section(scene_section_elements(run.optical_view), "S", "XZ", beam_paths=run.optical_view.beam_paths)
    (chief,) = [r for r in section["rays"] if r["role"] == "CHIEF"]
    assert chief["reference_from"] == k
    clear = run_optical_project(_design([declared_component("S", "aperture", "stop", position=(0.0, 0.0, 0.0))],
                                        [stop_model(opening_radius_mm="6", edge_radius_mm="10")], waist=3.0))
    assert all(p.chief_reference_from is None for p in clear.optical_view.beam_paths)
