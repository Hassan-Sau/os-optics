"""Acceptance: the reference optical design (a catalog converging lens found by query, at 633 nm) yields a local Optical
Simulation View with a source, the lens as one generic ``OPTIC_BODY`` carrying its published surfaces, the Engine beam
through the lens surfaces and beyond, and the target plane. Nothing here depends on a particular SKU: the assertions are
about the published model of whatever product the query returns."""

from __future__ import annotations

import math

import pytest

from optical_view_reference import BEAM_ALONG_X, SOURCE_WAIST_MM, SOURCE_X_MM, focal_plane_x_mm, reference_lens, to_local
from optical_view_reference import catalog, engine_request, engine_result, engine_run, project, ray_paths, scene  # noqa: F401  (pytest fixtures)

from os_optics import trace_ray_paths
from os_optics._vendor.shared_contracts.design_catalog import SurfaceSequenceModelV1
from os_optics._vendor.shared_contracts.optical_view import (
    GeometryBasisV1,
    GeometryStatusV1,
    ObservationPlaneKindV1,
    OpticalSimulationSceneV1,
    OpticBodyGeometryV1,
    RayEndV1,
    SourceEmitterGeometryV1,
)

from os_optics.optical_view import optical_view_files, parse_glb


def _lens(scene):
    return next(e for e in scene.elements if e.element_id == "L1")


def test_scene_is_an_optical_simulation_view_of_the_project(scene, project, catalog):
    assert scene.view_kind == "OPTICAL_SIMULATION_VIEW"
    assert scene.view_notice.startswith("Optical Simulation View of the design")
    assert scene.project.project_hash == project.content_hash_ref()
    assert scene.catalog_version == catalog.catalog_version
    assert scene.schema_version == "3.0"
    assert [e.element_id for e in scene.elements] == ["SRC", "L1"]
    assert OpticalSimulationSceneV1.model_validate_json(scene.model_dump_json(by_alias=True)) == scene


def test_the_lens_is_one_optic_body_with_its_published_surfaces(scene, catalog):
    lens = _lens(scene)
    product = reference_lens(catalog)
    (model,) = product.simulation_models
    assert isinstance(model, SurfaceSequenceModelV1)
    assert lens.optical_sku == product.optical_sku and lens.label == product.display_name
    assert lens.pose.quaternion_wxyz == BEAM_ALONG_X and lens.pose.position_mm == (0.0, 0.0, 0.0)
    geom = lens.geometry
    assert isinstance(geom, OpticBodyGeometryV1) and geom.kind == "OPTIC_BODY"
    body = product.design_geometry
    assert (geom.diameter_mm, geom.z_min_mm, geom.z_max_mm) == (float(body.diameter_mm), float(body.z_min_mm), float(body.z_max_mm))
    assert (geom.category, geom.form, geom.reference_axis) == (product.category, product.form, None)
    assert (geom.model_kind, geom.fidelity, geom.edge_radius_mm) == ("SURFACE_SEQUENCE", model.fidelity, float(model.edge_radius_mm))
    assert geom.media == model.media and geom.polarization is None
    assert [(s.surface_id, s.role.value, s.vertex_z_mm, s.radius_mm, s.clear_radius_mm) for s in geom.surfaces] == [
        (s.surface_id, "REFRACT", float(s.vertex_z_mm), None if s.radius_mm is None else float(s.radius_mm), float(s.clear_radius_mm))
        for s in model.surfaces
    ]
    for surface, published in zip(geom.surfaces, model.surfaces, strict=True):
        coating = catalog.coating(published.coating_id) if published.coating_id else None
        assert surface.coating_kind == (coating.kind if coating else None)
    assert geom.status is GeometryStatusV1.NOMINAL and geom.basis is GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY
    assert "effective_plane_z_mm" not in scene.payload()["elements"][1]["geometry"]  # no thin-lens plane any more


def test_source_and_target_planes(scene, catalog, engine_result):
    source = next(e for e in scene.elements if e.element_id == "SRC")
    assert isinstance(source.geometry, SourceEmitterGeometryV1) and source.geometry.emitter_radius_mm == SOURCE_WAIST_MM
    focal, screen = scene.observation_planes
    assert (focal.observation_id, focal.kind, focal.target_ids) == ("focal", ObservationPlaneKindV1.TARGET, ("power_at_focal",))
    assert (screen.observation_id, screen.kind) == ("screen", ObservationPlaneKindV1.OBSERVATION)
    assert focal.pose.position_mm == (round(focal_plane_x_mm(catalog), 6), 0.0, 0.0)
    (spot,) = focal.footprints
    (wide,) = screen.footprints
    # The view must not draw the old lossless N-BK7 estimate as measured optical power.
    assert spot.power_W is wide.power_W is None
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in engine_result.issues}
    (focal_arrival,) = next(o for o in engine_result.observations if o.observation_id == "focal").arrivals
    (screen_arrival,) = next(o for o in engine_result.observations if o.observation_id == "screen").arrivals
    assert focal_arrival.beam.power_W is screen_arrival.beam.power_W is None
    assert 0 < screen_arrival.power_upper_bound_W <= focal_arrival.power_upper_bound_W <= 1e-3


def test_the_beam_crosses_the_published_surfaces_where_the_engine_traced_it(scene, engine_request):
    """Every ray meets each lens surface on that surface's published profile: the view's surface annotations and the
    Engine's surfaces describe the same element (both come from the one public model)."""
    lens = _lens(scene)
    (path,) = scene.beam_paths
    (traced,) = trace_ray_paths(engine_request).paths
    steps = {surface_id: k for k, surface_id in enumerate(traced.chief_surface_ids) if surface_id is not None}
    assert [sid for sid in traced.chief_surface_ids if sid and sid.startswith("L1/")] == [f"L1/{s.surface_id}" for s in lens.geometry.surfaces]
    assert path.source_element_id == "SRC" and path.wavelength_nm == 633.0 and path.chief_ray.points_mm[0] == (SOURCE_X_MM, 0.0, 0.0)
    for surface in lens.geometry.surfaces:
        k = steps[f"L1/{surface.surface_id}"]
        assert to_local(lens, path.chief_ray.points_mm[k]) == pytest.approx((0.0, 0.0, surface.vertex_z_mm), abs=1e-6)
        for ray in path.pupil_rays:
            x, y, z = to_local(lens, ray.points_mm[k])
            rho = math.hypot(x, y)
            assert rho < surface.clear_radius_mm
            assert z == pytest.approx(surface.vertex_z_mm + surface.sag_at(rho), abs=1e-5)
    assert path.pupil_ray_count_traced > 0 and len(path.pupil_rays) > 0  # default: the outermost ring
    assert all(ray.end is RayEndV1.PATH_END for ray in path.pupil_rays)


def test_a_converging_lens_brings_the_outer_ring_towards_the_axis(scene):
    (path,) = scene.beam_paths
    for ray in path.pupil_rays:
        start, focal = ray.points_mm[0], ray.points_mm[-2]  # the source and the target plane before the screen
        assert math.hypot(focal[1], focal[2]) < math.hypot(start[1], start[2])


def test_every_ray_point_is_an_engine_point(scene, engine_request):
    trace = trace_ray_paths(engine_request)
    engine = {(p.path_id, 0): p.chief_points_mm for p in trace.paths}
    engine.update({(p.path_id, r.ray_index): r.points_mm for p in trace.paths for r in p.pupil_rays})
    d = scene.builder.settings.coordinate_decimals
    for path in scene.beam_paths:
        for ray in (path.chief_ray, *path.pupil_rays):
            expected = engine[(path.path_id, ray.ray_index)]
            assert len(ray.points_mm) == len(expected)
            for got, want in zip(ray.points_mm, expected, strict=True):
                assert got == tuple(round(c, d) + 0.0 for c in want)


def test_engine_run_binding_matches_the_validation_manifest(scene, engine_result, ray_paths):
    manifest = engine_result.manifest
    assert scene.engine_run.input_hash.digest == manifest.input_hash == ray_paths.input_hash
    assert scene.engine_run.result_hash.digest == manifest.result_hash
    assert scene.engine_run.trace_hash == manifest.trace_hash == ray_paths.trace_hash
    assert scene.engine_run.engine_version == manifest.engine_version
    assert scene.engine_run.ray_trace_version == ray_paths.ray_paths_version
    assert (scene.builder.builder_package, scene.builder.builder_version) == ("os-optics", manifest.engine_version)
    assert scene.engine_run.engine_package == "os-optics"


def test_the_bounds_hold_every_element_and_ray(scene):
    lo, hi = scene.bounds.min_mm, scene.bounds.max_mm
    for path in scene.beam_paths:
        for ray in (path.chief_ray, *path.pupil_rays):
            for point in ray.points_mm:
                assert all(lo[i] - 1e-6 <= point[i] <= hi[i] + 1e-6 for i in range(3))
    geom = _lens(scene).geometry
    assert hi[1] >= geom.diameter_mm / 2 - 1e-6 and lo[1] <= -geom.diameter_mm / 2 + 1e-6


def test_artifacts_are_complete_and_the_glb_is_valid(scene):
    files = optical_view_files(scene)
    assert set(files) == {"optical_scene.json", "optical_scene.glb"}
    gltf, _ = parse_glb(files["optical_scene.glb"])
    root = gltf["nodes"][gltf["scenes"][0]["nodes"][0]]
    names = [gltf["nodes"][i]["name"] for i in root["children"]]
    assert names == ["element:SRC", "element:L1", "observation:focal", "observation:screen", "beam:SRC#0", "optical_axes"]
    assert root["extras"]["opensci_view"] == "OPTICAL_SIMULATION_VIEW"
    assert root["extras"]["scene_content_hash"] == scene.content_hash() and root["extras"]["scene_schema_version"] == "3.0"
    lens_node = next(n for n in gltf["nodes"] if n["name"] == "element:L1")
    assert lens_node["extras"]["geometry_kind"] == "OPTIC_BODY" and lens_node["extras"]["category"] == "LENS"
    assert lens_node["extras"]["surfaces"] == [f"{s.surface_id}:REFRACT" for s in _lens(scene).geometry.surfaces]


def test_the_glb_camera_looks_from_the_hint_at_the_scene(scene):
    from os_optics.optical_view import geometry as g

    gltf, _ = parse_glb(optical_view_files(scene)["optical_scene.glb"])
    camera = gltf["nodes"][gltf["scenes"][0]["nodes"][1]]
    assert gltf["cameras"][camera["camera"]]["type"] == "perspective"

    def to_gltf(p):
        return (p[0] / 1000, p[2] / 1000, -p[1] / 1000)

    eye, target = to_gltf(scene.camera_hint.eye_mm), to_gltf(scene.camera_hint.target_mm)
    assert camera["translation"] == pytest.approx(list(eye), abs=1e-12)
    x, y, z, w = camera["rotation"]
    looking = g.rotate((w, x, y, z), (0.0, 0.0, -1.0))
    assert looking == pytest.approx(g.unit(g.sub(target, eye)), abs=1e-9)
    assert g.rotate((w, x, y, z), (0.0, 1.0, 0.0))[1] > 0.0  # design +Z (glTF +Y) stays up
