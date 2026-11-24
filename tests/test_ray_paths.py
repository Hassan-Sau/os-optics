"""``validate_with_ray_paths`` / ``trace_ray_paths``: the chief and pupil ray paths of the Engine's own trace.

They must (1) come from the very run ``validate`` performs (identical result, input and trace hashes, chief polylines equal
to the ``OpticalPath`` segments), (2) never change that run (the recorder only observes), and (3) report where the Engine
carries and stops each pupil ray. Scenes are built with the Engine's public builders."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from os_optics import (
    RAY_PATHS_VERSION,
    EngineConfiguration,
    NumericalConfig,
    RayPathTrace,
    ValidationRequest,
    trace_ray_paths,
    validate,
    validate_with_ray_paths,
)
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import Tracer
from os_optics.scene.lower import lower_scene
from os_optics.validation.ray_paths import _RayPathRecorder

REQUESTS = Path(__file__).resolve().parent / "fixtures" / "requests"
FIXTURE_FILES = sorted(REQUESTS.glob("*.json"))


def _request(components, observations, **numerical) -> ValidationRequest:
    project = ProjectSnapshot(components=tuple(components), observations=tuple(observations))
    return ValidationRequest(project=project, configuration=EngineConfiguration(
        run_layout=False, run_control=False, numerical=NumericalConfig(**numerical)))


def _keplerian():
    return _request(
        [B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=20.0),
         B.thin_lens("L1", focal_length_mm=25.0, clear_radius_mm=12.7, position=(20.0, 0, 20), axis=(1, 0, 0)),
         B.thin_lens("L2", focal_length_mm=100.0, clear_radius_mm=12.7, position=(145.0, 0, 20), axis=(1, 0, 0))],
        [B.observation_plane("out", position=(295.0, 0, 20), normal=(1, 0, 0), radius_mm=20.0)],
    )


def _stop_then_splitter():
    return _request(
        [B.laser("SRC", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 0.0, 20.0), direction=(1.0, 0.0, 0.0)),
         B.aperture_stop("Stop", position=(30.0, 0.0, 20.0), axis=(1.0, 0.0, 0.0), radius_mm=0.8),
         B.npbs("BS", position=(60.0, 0.0, 20.0), normal=B.unit((-1.0, 1.0, 0.0)))],
        [B.observation_plane("T", position=(100.0, 0.0, 20.0), normal=(1.0, 0.0, 0.0), radius_mm=20.0),
         B.observation_plane("R", position=(60.0, 40.0, 20.0), normal=(0.0, 1.0, 0.0), radius_mm=20.0)],
    )


def _mirror_window_dump():
    return _request(
        [B.laser("SRC", wavelength_nm=780.0, waist_radius_mm=0.8, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
         B.window("W", position=(30.0, 0.0, 0.0), axis=B.unit((1.0, 0.2, 0.0)), thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7",
                  trace_fresnel_reflection=True),
         B.mirror_for_reflection("M", position=(80.0, 0.0, 0.0), d_in=(1.0, 0.0, 0.0), d_out=(0.0, 1.0, 0.0)),
         B.beam_dump("D", position=(80.0, 60.0, 0.0), axis=(0.0, 1.0, 0.0))],
        [B.observation_plane("mid", position=(80.0, 30.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=15.0)],
    )


OBLIQUE_AXIS = B.unit((3.0, -2.0, 1.5))
OBLIQUE_UP = B.unit((0.0, 0.3, 1.0))
OBLIQUE_START = (5.0, 7.0, 11.0)
OBLIQUE_F, OBLIQUE_D_LENS = 60.0, 40.0


def _oblique_lens_center_and_focus():
    a = np.array(OBLIQUE_AXIS)
    lens_center = tuple(float(v) for v in (np.array(OBLIQUE_START) + OBLIQUE_D_LENS * a))
    focus = tuple(float(v) for v in (np.array(lens_center) + OBLIQUE_F * a))
    return lens_center, focus


def _oblique_focus():
    lens_center, focus = _oblique_lens_center_and_focus()
    return _request(
        [B.laser("SRC", wavelength_nm=780.0, waist_radius_mm=1.0, position=OBLIQUE_START, direction=OBLIQUE_AXIS, up=OBLIQUE_UP),
         B.thin_lens("L1", focal_length_mm=OBLIQUE_F, clear_radius_mm=11.43, position=lens_center, axis=OBLIQUE_AXIS, up=OBLIQUE_UP)],
        [B.observation_plane("focal", position=focus, normal=OBLIQUE_AXIS, radius_mm=5.0, up=OBLIQUE_UP)],
    )


def _stop_only():
    return _request(
        [B.laser("SRC", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 0.0, 20.0), direction=(1.0, 0.0, 0.0)),
         B.aperture_stop("Stop", position=(40.0, 0.0, 20.0), axis=(1.0, 0.0, 0.0), radius_mm=0.8)],
        [B.observation_plane("after", position=(70.0, 0.0, 20.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0)],
    )


def _tilted_not_followed():
    return _request(
        [B.laser("SRC", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
         B.aperture_stop("Tilted", position=(50.5, 0.0, 0.0), axis=B.unit((1.0, 1.0, 0.0)), radius_mm=40.0)],
        [B.observation_plane("obs", position=(50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=40.0)],
    )


def _path_limited():
    """Two sources but room for one path: the tracer discards the second root at its end (max_paths)."""
    return _request(
        [B.laser("A", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 0.0, 20.0), direction=(1.0, 0.0, 0.0)),
         B.laser("B", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 10.0, 20.0), direction=(1.0, 0.0, 0.0))],
        [B.observation_plane("obs", position=(50.0, 0.0, 20.0), normal=(1.0, 0.0, 0.0), radius_mm=40.0)],
        max_paths=1,
    )


def _run_physics_off():
    request = _keplerian()
    return request.model_copy(update={"configuration": request.configuration.model_copy(update={"run_physics": False})})


def _bad_schema():
    request = _keplerian()
    return request.model_copy(update={"project": request.project.model_copy(update={"project_schema_version": "999.0.0"})})


def _fixture(name: str) -> ValidationRequest:
    return ValidationRequest.model_validate_json((REQUESTS / f"{name}.json").read_text(encoding="utf-8"))


SCENES = {"keplerian": _keplerian, "stop_then_splitter": _stop_then_splitter, "mirror_window_dump": _mirror_window_dump}
EXTRA = {"oblique_focus": _oblique_focus, "stop_only": _stop_only, "tilted_not_followed": _tilted_not_followed,
         "path_limited": _path_limited, "run_physics_off": _run_physics_off, "bad_schema": _bad_schema}
ALL = {**{f"fixture:{p.stem}": (lambda p=p: _fixture(p.stem)) for p in FIXTURE_FILES}, **SCENES, **EXTRA}


def _chief_polyline(path, ray_path):
    if not path.segments:  # a branch that ended at its branch point (e.g. a zero-power reflection)
        return [ray_path.chief_points_mm[0]]
    points = [seg.start_mm for seg in path.segments]
    if path.segments[-1].end_mm is not None:
        points.append(path.segments[-1].end_mm)
    return points


@pytest.mark.parametrize("name", sorted(ALL))
def test_the_result_is_exactly_validates_and_the_rays_are_its_trace(name):
    request = ALL[name]()
    result, trace = validate_with_ray_paths(request)
    reference = validate(request)
    assert result.model_dump_json() == reference.model_dump_json()
    assert isinstance(trace, RayPathTrace) and trace.ray_paths_version == RAY_PATHS_VERSION == "2.0"
    assert (trace.input_hash, trace.trace_hash) == (reference.manifest.input_hash, reference.manifest.trace_hash)
    assert trace_ray_paths(request) == trace
    if trace.status == "NOT_TRACED":
        assert trace.trace_hash is None and trace.paths == () and trace.message
        return
    assert trace.status == "TRACED" and trace.message == "" and trace.trace_hash is not None
    assert [p.path_id for p in trace.paths] == sorted(p.path_id for p in trace.paths)
    assert len(trace.paths) == result.summary.path_count
    if request.configuration.include_paths:
        assert [p.path_id for p in trace.paths] == [p.path_id for p in result.paths]
        for ray_path, path in zip(trace.paths, result.paths, strict=True):
            assert (ray_path.source_id, ray_path.spectral_index, ray_path.wavelength_nm) == (path.source_id, path.spectral_index, path.wavelength_nm)
            assert (ray_path.parent_path_id, ray_path.termination, ray_path.termination_surface_id) == (
                path.parent_path_id, path.termination, path.termination_surface_id)
            assert list(ray_path.chief_points_mm) == _chief_polyline(path, ray_path)
            assert ray_path.chief_surface_ids[0] == path.branch_surface_id
            assert list(ray_path.chief_surface_ids[1:]) == [s.end_surface_id for s in path.segments if s.end_surface_id is not None]
            assert (ray_path.chief_exit_direction is not None) == (path.termination == "ESCAPED")
    by_id = {p.path_id: p for p in trace.paths}
    for ray_path in trace.paths:
        assert len(ray_path.chief_points_mm) == len(ray_path.chief_surface_ids)
        assert len(ray_path.pupil_rays) <= ray_path.pupil_ray_count
        assert all(1 <= r.ray_index <= ray_path.pupil_ray_count for r in ray_path.pupil_rays)
        if ray_path.parent_path_id is not None and ray_path.parent_path_id in by_id:
            parent = by_id[ray_path.parent_path_id]
            assert ray_path.path_id.startswith(parent.path_id + ".") and ray_path.source_id == parent.source_id
            assert {r.ray_index for r in ray_path.pupil_rays} <= {r.ray_index for r in parent.pupil_rays}
            assert ray_path.chief_points_mm[0] in parent.chief_points_mm
            assert ray_path.chief_surface_ids[0] in parent.chief_surface_ids


@pytest.mark.parametrize("name", sorted(ALL))
def test_recording_never_changes_the_engine_trace(name):
    request = ALL[name]()
    project, cfg = request.project, request.configuration.numerical
    outputs = []
    for recorder in (None, _RayPathRecorder()):
        library = MaterialLibrary(project.materials, cfg)
        try:
            scene, _ = lower_scene(project, cfg, library)
        except Exception:  # a request the Engine refuses before tracing: nothing to compare
            return
        outputs.append(Tracer(scene, library, cfg, recorder=recorder).run())
    plain, recorded = outputs
    assert recorded.paths == plain.paths and recorded.trace_events == plain.trace_events
    assert [i.model_dump() for i in recorded.issues] == [i.model_dump() for i in plain.issues]
    assert len(recorded.arrivals) == len(plain.arrivals)
    for a, b in zip(recorded.arrivals, plain.arrivals, strict=True):
        assert (a.observation_id, a.path_id, a.power_W, a.uv) == (b.observation_id, b.path_id, b.power_W, b.uv)
        assert np.array_equal(a.pupil_uv, b.pupil_uv) and np.array_equal(a.direction, b.direction)


def test_trace_is_deterministic():
    assert trace_ray_paths(_keplerian()) == trace_ray_paths(_keplerian())


def test_pupil_rays_of_a_collimated_beam_meet_at_the_thin_lens_focus_on_an_oblique_axis():
    a = np.array(OBLIQUE_AXIS)
    _lens_center, focus = _oblique_lens_center_and_focus()
    (ray_path,) = trace_ray_paths(_oblique_focus()).paths
    assert ray_path.source_component_id == "SRC" and ray_path.chief_surface_ids == (None, "L1/lens", "observation/focal")
    assert ray_path.pupil_ray_count == 96 and len(ray_path.pupil_rays) == 96
    assert {(r.ring_index, r.ring_position_index) for r in ray_path.pupil_rays} == {(k, m) for k in range(8) for m in range(12)}
    for ray in ray_path.pupil_rays:
        assert ray.end == "PATH_END" and ray.end_surface_id is None and ray.exit_direction is not None
        start, at_lens, at_focus = (np.array(p) for p in ray.points_mm)
        offset = start - np.array(ray_path.chief_points_mm[0])
        assert abs(offset @ a) < 1e-12
        assert np.linalg.norm(at_lens - start - OBLIQUE_D_LENS * a) < 1e-9
        assert np.linalg.norm(at_focus - np.array(focus)) < 1e-9  # the Engine's thin lens sends it through the focus
        assert np.array(ray.exit_direction) @ offset < 0.0


def test_rays_outside_an_aperture_stop_end_where_the_engine_removes_them():
    (ray_path,) = trace_ray_paths(_stop_only()).paths
    stopped = [r for r in ray_path.pupil_rays if r.end == "STOPPED_AT_SURFACE"]
    passed = [r for r in ray_path.pupil_rays if r.end == "PATH_END"]
    assert stopped and passed and len(stopped) + len(passed) == ray_path.pupil_ray_count
    for ray in stopped:
        assert ray.end_surface_id == "Stop/stop" and len(ray.points_mm) == 2 and ray.exit_direction is None
        assert math.hypot(ray.points_mm[-1][1], ray.points_mm[-1][2] - 20.0) > 0.8
    for ray in passed:
        assert len(ray.points_mm) == 3 and ray.exit_direction is not None
        assert math.hypot(ray.points_mm[1][1], ray.points_mm[1][2] - 20.0) <= 0.8


def test_a_branch_after_a_stop_carries_only_the_rays_that_reached_the_branch_point():
    """Regression: rays removed before a splitter belong to the parent path only."""
    trace = trace_ray_paths(_stop_then_splitter())
    by_id = {p.path_id: p for p in trace.paths}
    (root,) = [p for p in trace.paths if p.parent_path_id is None]
    children = [p for p in trace.paths if p.parent_path_id is not None]
    assert children
    passed_stop = {r.ray_index for r in root.pupil_rays if r.end != "STOPPED_AT_SURFACE"}
    assert 0 < len(passed_stop) < root.pupil_ray_count
    for child in children:
        parent = by_id[child.parent_path_id]
        assert {r.ray_index for r in child.pupil_rays} <= passed_stop and child.pupil_ray_count == root.pupil_ray_count
        k = parent.chief_surface_ids.index(child.chief_surface_ids[0])
        assert child.chief_points_mm[0] == parent.chief_points_mm[k]
        for ray in child.pupil_rays:
            (parent_ray,) = [r for r in parent.pupil_rays if r.ray_index == ray.ray_index]
            assert ray.points_mm[0] == parent_ray.points_mm[k]


def test_marginal_rays_that_meet_the_chiefs_next_surface_before_a_plane_skip_the_plane_and_go_on():
    """The rays above y = 0.5 mm meet the tilted stop (the chief ray's next surface) before the plane the chief ray crosses
    first: an observation plane never retires a ray, so they have no point on the plane, reach the stop and go on with the
    beam (1, M1: they used to be retired at the plane, NOT_FOLLOWED)."""
    (ray_path,) = trace_ray_paths(_tilted_not_followed()).paths
    assert ray_path.chief_surface_ids == (None, "observation/obs", "Tilted/stop")
    skipped = [r for r in ray_path.pupil_rays if 1 not in r.steps]
    assert skipped and len(skipped) < ray_path.pupil_ray_count
    for ray in skipped:
        assert ray.points_mm[0][1] > 0.5
        assert ray.surface_ids == (None, "Tilted/stop") and ray.end == "PATH_END"
    assert {r.end for r in ray_path.pupil_rays} == {"PATH_END"}


def test_absorbed_paths_stop_every_ray_at_the_absorber():
    trace = trace_ray_paths(_mirror_window_dump())
    absorbed = [p for p in trace.paths if p.termination == "ABSORBED"]
    assert absorbed
    for path in absorbed:
        assert path.termination_surface_id == "D/dump" and path.chief_exit_direction is None
        assert all(r.end == "STOPPED_AT_SURFACE" and r.end_surface_id == "D/dump" and r.exit_direction is None for r in path.pupil_rays)


def test_the_fresnel_branch_starts_at_the_window_and_escapes_with_exit_directions():
    trace = trace_ray_paths(_mirror_window_dump())
    branches = [p for p in trace.paths if p.parent_path_id is not None]
    assert branches
    for branch in branches:
        assert branch.chief_surface_ids[0].startswith("W/")
        if branch.termination == "ESCAPED":
            assert branch.chief_exit_direction is not None
            assert all(r.exit_direction is not None for r in branch.pupil_rays if r.end == "PATH_END")


def test_a_path_discarded_by_the_path_limit_is_not_reported():
    result, trace = validate_with_ray_paths(_path_limited())
    assert trace.status == "TRACED" and [p.path_id for p in trace.paths] == [p.path_id for p in result.paths] == ["A#0"]


def test_not_traced_instead_of_raising():
    off = _run_physics_off()
    disabled = trace_ray_paths(off)
    assert disabled.status == "NOT_TRACED" and disabled.trace_hash is None and "run_physics" in disabled.message
    assert disabled.input_hash == validate(off).manifest.input_hash

    schema = trace_ray_paths(_bad_schema())
    assert schema.status == "NOT_TRACED" and schema.message == "unsupported project_schema_version" and schema.paths == ()

    invalid = _fixture("invalid_zero_normal")
    refused = trace_ray_paths(invalid)
    (issue,) = validate(invalid).issues
    assert refused.status == "NOT_TRACED" and refused.message == f"{issue.code.value}: {issue.message}"
    assert refused.message.startswith("ZERO_LENGTH_VECTOR: ")


def test_a_run_that_fails_after_tracing_reports_no_rays():
    """The rays are bound to the manifest's trace_hash; a run that fails later (here: an unknown actuator in the control
    analysis) has none, so neither does its ray trace."""
    request = _fixture("control_recoverability")
    spec = request.project.control_spec
    project = request.project.model_copy(update={"control_spec": spec.model_copy(update={"actuator_refs": ("Nobody/x",)})})
    request = request.model_copy(update={"project": project})
    result, trace = validate_with_ray_paths(request)
    assert result.manifest.trace_hash is None and result.status.value == "FAIL"
    assert trace.status == "NOT_TRACED" and trace.paths == () and trace.message.startswith("UNKNOWN_REFERENCE: ")
