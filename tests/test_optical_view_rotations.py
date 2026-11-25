"""Nontrivial poses: the view follows the Project nominal pose (right-handed, mm, element local +Z = optical axis) for a
rigidly rotated design, a lens tilted against the beam and a component-attached plane, and the GLB carries exactly
those poses. Quaternions are built with sqrt only (no libm trig), so the fixtures are platform-independent. (Moved from
the earlier ``opensci-optical-view`` package; the Engine requests come from the real local-runtime lowering.)"""

from __future__ import annotations

import math

import pytest
from optical_view_reference import build_scene, element_primitives, reference_project, rigidly_moved, to_local, unit_quaternion
from optical_view_reference import catalog, engine_request, engine_result, engine_run, project, ray_paths, scene  # noqa: F401  (pytest fixtures)

from os_optics import trace_ray_paths
from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, ObservationV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.optical_view import geometry as g
from os_optics.optical_view import parse_glb, read_accessor, scene_glb_bytes

RIGID_Q = unit_quaternion(0.8, 0.3, -0.4, 0.33)  # a general rotation: no axis-aligned shortcut survives it
RIGID_T = (40.0, -25.0, 60.0)
TILT_Q = unit_quaternion(10.0, 1.0, 0.0, 0.0)  # ~11.4 deg about the lens-local X axis


def _build(project: ProjectDocumentV2, catalog):
    return build_scene(project, catalog)


def _moved(p):
    return g.apply_pose(RIGID_T, RIGID_Q, p)


def _steps(request, prefix):
    (path,) = [p for p in trace_ray_paths(request).paths if p.parent_path_id is None]
    return {sid.split("/", 1)[1]: k for k, sid in enumerate(path.chief_surface_ids) if sid and sid.startswith(prefix)}


def test_a_rigidly_rotated_design_gives_the_rigidly_rotated_view(catalog, scene):
    rotated, _ = _build(rigidly_moved(reference_project(catalog), RIGID_T, RIGID_Q), catalog)
    for base, moved in zip(scene.elements, rotated.elements, strict=True):
        assert moved.pose.position_mm == pytest.approx(_moved(base.pose.position_mm), abs=1e-6)
        assert g.same_rotation(moved.pose.quaternion_wxyz, g.quat_multiply(RIGID_Q, base.pose.quaternion_wxyz), 1e-12)
        assert moved.geometry == base.geometry  # element-frame geometry does not depend on the pose
    for base, moved in zip(scene.observation_planes, rotated.observation_planes, strict=True):
        assert moved.pose.position_mm == pytest.approx(_moved(base.pose.position_mm), abs=1e-6)
    # The Engine rays of the rotated design are the rotated rays of the original design.
    for base_path, moved_path in zip(scene.beam_paths, rotated.beam_paths, strict=True):
        for base_ray, moved_ray in zip((base_path.chief_ray, *base_path.pupil_rays), (moved_path.chief_ray, *moved_path.pupil_rays), strict=True):
            assert len(base_ray.points_mm) == len(moved_ray.points_mm)
            for a, b in zip(base_ray.points_mm, moved_ray.points_mm, strict=True):
                assert b == pytest.approx(_moved(a), abs=1e-5)


def test_in_a_rotated_design_rays_meet_the_published_surfaces_in_the_lens_frame(catalog):
    rotated, request = _build(rigidly_moved(reference_project(catalog), RIGID_T, RIGID_Q), catalog)
    lens = next(e for e in rotated.elements if e.element_id == "L1")
    steps = _steps(request, "L1/")
    (path,) = rotated.beam_paths
    for surface in lens.geometry.surfaces:
        k = steps[surface.surface_id]
        assert to_local(lens, path.chief_ray.points_mm[k]) == pytest.approx((0.0, 0.0, surface.vertex_z_mm), abs=1e-5)
        for ray in path.pupil_rays:
            x, y, z = to_local(lens, ray.points_mm[k])
            assert math.hypot(x, y) < surface.clear_radius_mm
            assert z == pytest.approx(surface.vertex_z_mm + surface.sag_at(math.hypot(x, y)), abs=1e-5)


def test_the_glb_places_the_rotated_lens_body_across_the_rotated_beam(catalog):
    rotated, _ = _build(rigidly_moved(reference_project(catalog), RIGID_T, RIGID_Q), catalog)
    lens = next(e for e in rotated.elements if e.element_id == "L1")
    geom = lens.geometry
    gltf, binary = parse_glb(scene_glb_bytes(rotated))
    (node,) = [n for n in gltf["nodes"] if n["name"] == "element:L1"]
    w, x, y, z = lens.pose.quaternion_wxyz
    assert node["translation"] == list(lens.pose.position_mm)
    assert node["rotation"] == pytest.approx([x, y, z, w], abs=1e-15)
    q = (node["rotation"][3], *node["rotation"][:3])
    axis = g.rotate(q, (0.0, 0.0, 1.0))
    (path,) = rotated.beam_paths
    beam_dir = g.unit(g.sub(path.chief_ray.points_mm[1], path.chief_ray.points_mm[0]))
    assert g.dot(axis, beam_dir) == pytest.approx(1.0, abs=1e-9)  # the lens faces the beam
    rims = [g.surface_rim(s.vertex_z_mm, s.radius_mm, s.clear_radius_mm, geom.edge_radius_mm)[1] for s in geom.surfaces]
    lo, hi = min(s.vertex_z_mm for s in geom.surfaces), max(s.vertex_z_mm for s in geom.surfaces)
    radii = []
    for points in element_primitives(gltf, binary, "L1").values():
        for p in points:
            world = g.apply_pose(tuple(node["translation"]), q, p)
            rel = g.sub(world, lens.pose.position_mm)
            along = g.dot(rel, axis)
            assert min(lo, *rims) - 1e-4 <= along <= max(hi, *rims) + 1e-4
            radii.append(g.norm(g.sub(rel, g.scale(axis, along))))
    assert max(radii) == pytest.approx(geom.edge_radius_mm, abs=1e-4)


def test_a_lens_tilted_against_the_beam_is_drawn_tilted_and_the_rays_are_the_engines(catalog):
    base = reference_project(catalog)
    components = []
    for c in base.components:
        if c.instance_id == "L1":
            c = c.model_copy(update={"nominal_pose": PoseV1(position_mm=c.nominal_pose.position_mm,
                                                             quaternion_wxyz=g.quat_multiply(c.nominal_pose.quaternion_wxyz, TILT_Q))})
        components.append(c)
    tilted_project = ProjectDocumentV2.model_validate(base.model_copy(update={"components": tuple(components)}).payload())
    tilted, request = _build(tilted_project, catalog)
    lens = next(e for e in tilted.elements if e.element_id == "L1")
    axis = g.rotate(lens.pose.quaternion_wxyz, (0.0, 0.0, 1.0))
    tilt_deg = math.degrees(math.acos(g.dot(axis, (1.0, 0.0, 0.0))))
    assert tilt_deg == pytest.approx(2 * math.degrees(math.atan(0.1)), abs=1e-9)
    (path,) = tilted.beam_paths
    # the chief ray meets the tilted first surface where the lens body places it (on its published profile) ...
    first = lens.geometry.surfaces[0]
    hit = to_local(lens, path.chief_ray.points_mm[_steps(request, "L1/")[first.surface_id]])
    assert hit[2] == pytest.approx(first.vertex_z_mm + first.sag_at(math.hypot(hit[0], hit[1])), abs=1e-5)
    # ... the tilted lens deflects the beam off the original axis ...
    after = path.chief_ray.points_mm[-1]
    assert math.hypot(after[1], after[2]) > 1e-3
    # ... and the rays are exactly the Engine's rays, not a view-side reconstruction
    (engine_path,) = trace_ray_paths(request).paths
    assert path.chief_ray.points_mm == tuple(tuple(round(c, 6) + 0.0 for c in p) for p in engine_path.chief_points_mm)


def test_a_component_attached_plane_is_composed_with_its_component_pose(catalog):
    base = reference_project(catalog)
    attached = ObservationV1(observation_id="behind_lens", device_instance_id="L1",
                             pose=PoseV1(position_mm=(0.0, 0.0, 20.0)), aperture=CircularApertureV1(radius_mm=8.0))
    project = ProjectDocumentV2.model_validate(base.model_copy(update={"observations": (*base.observations, attached)}).payload())
    moved = rigidly_moved(project, RIGID_T, RIGID_Q)
    view, _ = _build(moved, catalog)
    lens = next(e for e in view.elements if e.element_id == "L1")
    plane = next(p for p in view.observation_planes if p.observation_id == "behind_lens")
    assert plane.pose.position_mm == pytest.approx(g.apply_pose(lens.pose.position_mm, lens.pose.quaternion_wxyz, (0.0, 0.0, 20.0)), abs=1e-6)
    assert g.same_rotation(plane.pose.quaternion_wxyz, lens.pose.quaternion_wxyz, 1e-12)
    # the Engine chief ray crosses exactly this plane: its hit point lies in it
    (path,) = view.beam_paths
    normal = g.rotate(plane.pose.quaternion_wxyz, (0.0, 0.0, 1.0))
    distances = [abs(g.dot(g.sub(p, plane.pose.position_mm), normal)) for p in path.chief_ray.points_mm]
    assert min(distances) < 1e-5
    (footprint,) = plane.footprints
    assert footprint.path_id == path.path_id
