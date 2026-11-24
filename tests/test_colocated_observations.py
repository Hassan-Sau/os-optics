"""Co-located observation planes sample each arriving path without hiding one another."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, builders as B, validate_with_ray_paths
from os_optics.contracts import ProjectSnapshot
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.decisions import SceneIndex, observation_crossing
from os_optics.optics.rays.tracer import Tracer
from os_optics.scene.lower import lower_scene

from helpers import arrivals, issue_codes, one_arrival, run


def _laser():
    return B.laser(
        "L", wavelength_nm=532.0, waist_radius_mm=0.5, power_W=10e-3,
        position=(0, 0, 20), direction=(1, 0, 0),
    )


def _observer(observation_id: str, x: float = 100.0, *, absorbing: bool = False):
    return B.observation_plane(
        observation_id, position=(x, 0, 20), normal=(1, 0, 0),
        radius_mm=10.0, absorbing=absorbing,
    )


def test_identical_nonabsorbing_planes_record_the_same_beam_and_continue():
    first = _observer("first")
    second = _observer("second")
    downstream = _observer("downstream", 200.0)
    baseline = run(ProjectSnapshot(components=(_laser(),), observations=(first, downstream)))
    result = run(ProjectSnapshot(components=(_laser(),), observations=(first, second, downstream)))

    reference = one_arrival(baseline, "first")
    assert one_arrival(result, "first") == reference
    assert one_arrival(result, "second") == reference
    assert reference.beam.geometric_length_mm == pytest.approx(100.0)
    assert reference.beam.power_W == pytest.approx(10e-3)
    assert reference.pupil_rays_arrived == reference.pupil_ray_count > 0
    assert one_arrival(result, "downstream") == one_arrival(baseline, "downstream")
    assert len(result.paths) == 1
    assert result.paths[0].termination == "ESCAPED"
    assert result.paths[0].geometric_path_length_mm == pytest.approx(200.0)


def test_colocated_arrivals_do_not_depend_on_observation_declaration_order():
    observers = (_observer("first"), _observer("second"), _observer("downstream", 200.0))
    reference = None
    for order in permutations(observers):
        result = run(ProjectSnapshot(components=(_laser(),), observations=order))
        by_id = {observation.observation_id: observation for observation in result.observations}
        assert all(len(observation.arrivals) == 1 for observation in by_id.values())
        if reference is None:
            reference = by_id
        else:
            assert by_id == reference


@pytest.mark.parametrize("offset_y", (0.0, 5.0))
def test_colocated_observers_keep_independent_active_areas(offset_y: float):
    narrow = B.observation_plane("narrow", position=(100, offset_y, 20), normal=(1, 0, 0), radius_mm=0.3)
    wide = B.observation_plane("wide", position=(100, offset_y, 20), normal=(1, 0, 0), radius_mm=10.0)
    downstream = _observer("downstream", 200.0)
    narrow_only = run(ProjectSnapshot(components=(_laser(),), observations=(narrow,)))
    wide_only = run(ProjectSnapshot(components=(_laser(),), observations=(wide, downstream)))
    result = run(ProjectSnapshot(components=(_laser(),), observations=(narrow, wide, downstream)))

    assert arrivals(result, "narrow") == arrivals(narrow_only, "narrow")
    assert one_arrival(result, "wide") == one_arrival(wide_only, "wide")
    assert one_arrival(result, "downstream") == one_arrival(wide_only, "downstream")
    if offset_y == 0.0:
        small = one_arrival(result, "narrow")
        large = one_arrival(result, "wide")
        assert small.active_area_power_fraction < large.active_area_power_fraction
        assert small.pupil_rays_arrived == large.pupil_rays_arrived
    else:
        assert arrivals(result, "narrow") == []


@pytest.mark.parametrize("reverse", (False, True))
def test_colocated_observers_keep_their_own_plane_coordinates(reverse: bool):
    observers = (
        B.observation_plane("forward", position=(100, 1, 22), normal=(1, 0, 0), radius_mm=10.0),
        B.observation_plane("flipped", position=(100, 1, 22), normal=(-1, 0, 0), radius_mm=10.0),
        B.observation_plane("rotated", position=(100, 1, 22), normal=(1, 0, 0), up=(0, 1, 0), radius_mm=10.0),
    )
    expected_uv = {"forward": (-1.0, -2.0), "flipped": (1.0, -2.0), "rotated": (2.0, -1.0)}
    result = run(ProjectSnapshot(components=(_laser(),), observations=observers[::-1] if reverse else observers))

    for observer in observers:
        observation_id = observer.observation_id
        isolated = run(ProjectSnapshot(components=(_laser(),), observations=(observer,)))
        reference = one_arrival(isolated, observation_id)
        actual = one_arrival(result, observation_id)
        assert actual.uv_mm == pytest.approx(expected_uv[observation_id])
        assert actual.aoi_deg == pytest.approx(reference.aoi_deg)
        assert actual.pupil_centroid_uv_mm == pytest.approx(reference.pupil_centroid_uv_mm)
        assert actual.pupil_rms_radius_mm == pytest.approx(reference.pupil_rms_radius_mm)
        assert actual.pupil_rays_arrived == reference.pupil_rays_arrived == reference.pupil_ray_count
        assert actual.beam.position_mm == pytest.approx(reference.beam.position_mm)
        assert actual.beam.optical_path_length_mm == pytest.approx(reference.beam.optical_path_length_mm)
        assert actual.beam.power_W == pytest.approx(reference.beam.power_W)


@pytest.mark.parametrize("reverse", (False, True))
def test_each_split_path_arrives_at_both_colocated_observers(reverse: bool):
    # Both arms cross x + y = 200, then x + y = 300, at different points.
    components = (
        _laser(),
        B.npbs("BS", position=(50, 0, 20), normal=(-1, 1, 0), reflectance=0.3, clear_radius_mm=10.0),
    )
    first = B.observation_plane("first", position=(100, 100, 20), normal=(1, 1, 0))
    second = B.observation_plane("second", position=(100, 100, 20), normal=(1, 1, 0))
    downstream = B.observation_plane("downstream", position=(150, 150, 20), normal=(1, 1, 0))
    baseline = run(ProjectSnapshot(components=components, observations=(first, downstream)))
    observers = (first, second, downstream)
    result = run(ProjectSnapshot(components=components, observations=observers[::-1] if reverse else observers))

    reference = {arrival.beam.path_id: arrival for arrival in arrivals(baseline, "first")}
    assert len(reference) == 2
    assert sorted(arrival.beam.power_W for arrival in reference.values()) == pytest.approx([3e-3, 7e-3])
    for observation_id in ("first", "second"):
        actual = arrivals(result, observation_id)
        assert len(actual) == 2
        assert {arrival.beam.path_id: arrival for arrival in actual} == reference
        assert all(arrival.beam.source_id == "L" for arrival in actual)
        assert [arrival.beam.geometric_length_mm for arrival in actual] == pytest.approx([200.0, 200.0])
        assert all(arrival.pupil_rays_arrived == arrival.pupil_ray_count > 0 for arrival in actual)
    assert arrivals(result, "downstream") == arrivals(baseline, "downstream")
    assert len(result.paths) == 2
    assert {path.termination for path in result.paths} == {"ESCAPED"}
    assert {path.path_id for path in result.paths} == set(reference)
    assert sum(path.parent_path_id is not None for path in result.paths) == 1


def test_absorbing_detector_still_terminates_the_path():
    detector = _observer("detector", absorbing=True)
    result = run(ProjectSnapshot(
        components=(_laser(),), observations=(detector, _observer("downstream", 200.0)),
    ))

    assert one_arrival(result, "detector").beam.power_W == pytest.approx(10e-3)
    assert arrivals(result, "downstream") == []
    assert len(result.paths) == 1
    assert result.paths[0].termination == "DETECTED"
    assert result.paths[0].geometric_path_length_mm == pytest.approx(100.0)


@pytest.mark.parametrize("reverse", (False, True))
def test_colocated_observers_are_recorded_before_detector_termination(reverse: bool):
    # Put the absorber between the passive observers in ID order as well as declaration order.
    detector = _observer("M_detector", absorbing=True)
    observers = (_observer("A_observer"), detector, _observer("Z_observer"), _observer("downstream", 200.0))
    baseline = run(ProjectSnapshot(components=(_laser(),), observations=(detector,)))
    result = run(ProjectSnapshot(
        components=(_laser(),), observations=observers[::-1] if reverse else observers,
    ))

    reference = one_arrival(baseline, "M_detector")
    assert reference.detected_power_W == pytest.approx(10e-3)
    assert one_arrival(result, "M_detector") == reference
    # the transparent observers see the same beam; only the detector absorbs (detects) it
    for observation_id in ("A_observer", "Z_observer"):
        assert one_arrival(result, observation_id) == reference.model_copy(update={"detected_power_W": None})
    assert arrivals(result, "downstream") == []
    assert len(result.paths) == 1
    assert result.paths[0].termination == "DETECTED"
    assert result.paths[0].termination_surface_id == baseline.paths[0].termination_surface_id
    assert result.paths[0].geometric_path_length_mm == pytest.approx(100.0)


def test_nearby_planes_with_large_tangential_offsets_remain_distinct():
    observers = tuple(
        B.observation_plane(observation_id, position=(x, 1e12, 20), normal=(1, 0, 0))
        for observation_id, x in (("first", 100.0), ("second", 100.0001))
    )
    result = run(ProjectSnapshot(components=(_laser(),), observations=observers))

    for observation_id, x in (("first", 100.0), ("second", 100.0001)):
        arrival = one_arrival(result, observation_id)
        assert arrival.beam.geometric_length_mm == pytest.approx(x, rel=0, abs=1e-9)
        assert arrival.beam.position_mm[0] == pytest.approx(x, rel=0, abs=1e-9)


def test_colocated_observers_do_not_use_the_per_path_interaction_limit():
    """Observation planes are virtual : crossing them is not a physical interaction, so even a limit of one
    interaction lets the beam cross the transparent observer and reach the coplanar detector (this used to stop the path
    after the first observer, MAX_INTERACTIONS, with the detector never reached)."""
    project = ProjectSnapshot(
        components=(_laser(),), observations=(_observer("A"), _observer("B", absorbing=True)),
    )
    result = run(project, numerical=NumericalConfig(max_interactions_per_path=1))

    assert one_arrival(result, "A").beam.geometric_length_mm == pytest.approx(100.0)
    assert one_arrival(result, "B").detected_power_W == pytest.approx(10e-3)
    assert len(result.paths) == 1
    assert result.paths[0].termination == "DETECTED"
    assert "MAX_INTERACTIONS" not in issue_codes(result)


def test_tilted_planes_through_one_point_each_sample_their_own_pupil_crossings():
    """Two planes through one point, 0.29 deg apart, under a loose parallel_dot_tol: both belong to the chief ray's
    crossing (neither is left behind the ray-advance guard), and each records every pupil ray where that ray crosses
    that plane -- the loose tolerance never puts one plane's rays on the other."""
    project = ProjectSnapshot(components=(_laser(),), observations=(
        B.observation_plane("flat", position=(100, 0, 20), normal=(1, 0, 0)),
        B.observation_plane("tilted", position=(100, 0, 20), normal=(1, 0.005, 0)),
    ))
    cfg = NumericalConfig(parallel_dot_tol=0.01)
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    flat, tilted = scene.surface_index["observation/flat"], scene.surface_index["observation/tilted"]
    times = np.full((len(scene.surfaces), 1), np.inf)
    times[[flat, tilted], 0] = 100.0
    assert observation_crossing(SceneIndex.of(scene.surfaces), flat, times, cfg) == [flat, tilted]
    out = Tracer(scene, library, cfg).run()
    by_id = {arrival.observation_id: arrival for arrival in out.arrivals}
    assert sorted(by_id) == ["flat", "tilted"]
    for arrival in by_id.values():
        assert len(arrival.pupil_points) == arrival.pupil_total > 0
        on_plane = (arrival.pupil_points - arrival.plane_origin) @ arrival.plane_normal
        assert np.max(np.abs(on_plane)) <= 1e-9
    assert np.max(np.abs(by_id["flat"].pupil_points - by_id["tilted"].pupil_points)) > 1e-4


def test_colocated_observers_do_not_spuriously_retire_recorded_pupil_rays():
    project = ProjectSnapshot(
        components=(_laser(),),
        observations=(_observer("first"), _observer("second"), _observer("downstream", 200.0)),
    )
    request = ValidationRequest(
        project=project, configuration=EngineConfiguration(run_layout=False, run_control=False),
    )
    result, trace = validate_with_ray_paths(request)

    assert result == run(project)
    assert len(trace.paths) == 1
    path = trace.paths[0]
    assert len(path.pupil_rays) == path.pupil_ray_count > 0
    assert {ray.end for ray in path.pupil_rays} == {"PATH_END"}
    for observation in project.observations:
        assert path.chief_surface_ids.count(f"observation/{observation.observation_id}") == 1
