"""Observation planes that meet the beam at one crossing, or within a hair of each other.

Observation planes are virtual: each samples every ray of a beam where that ray crosses that plane, and none moves,
turns, retires or counts a ray. Two transparent planes through one point with normals +X and (X + Y) / sqrt(2)
recorded the 1 mW beam on one of them only (which one depended on the declaration order); the other was left behind the
ray-advance guard and read as a known 0 W, so a 1 nW maximum on it was MET. These tests pin the physical invariants of
every case of the issue's acceptance: power conservation per arrival, independence of the declaration order (all
permutations, compared bit for bit), the crossing index, each arrival on its own plane, and each pupil ray's crossing
point and optical path length against the straight-line crossing of that ray with that plane.
"""

from __future__ import annotations

import copy
import math
from itertools import permutations

import numpy as np
import pytest

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, builders as B, validate_with_ray_paths
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import Tracer
from os_optics.scene.lower import lower_scene

from helpers import arrivals, issue_codes, one_arrival, run

SOURCE_W = 1e-3
LAM = 780.0
X = (1.0, 0.0, 0.0)
MX = (-1.0, 0.0, 0.0)
Q = math.sqrt(0.5)
DIAGONAL = (Q, Q, 0.0)  # the issue's second normal, (X + Y) / sqrt(2)
R_MIRROR = 0.9


def _laser(**kwargs):
    kwargs.setdefault("waist_radius_mm", 0.5)
    return B.laser("L", wavelength_nm=LAM, power_W=SOURCE_W, position=(-10.0, 0.0, 0.0), direction=X, **kwargs)


def _plane(observation_id: str, position=(0.0, 0.0, 0.0), normal=X, **kwargs):
    kwargs.setdefault("radius_mm", 50.0)
    return B.observation_plane(observation_id, position=position, normal=normal, **kwargs)


def _unit(v) -> tuple[float, float, float]:
    a = np.array(v, dtype=float)
    a = a / math.sqrt(float(a @ a))
    return float(a[0]), float(a[1]), float(a[2])


def _in_xy(deg: float) -> tuple[float, float, float]:
    """A plane normal turned ``deg`` from +X toward +Y."""
    return math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0


def _trace(project: ProjectSnapshot, cfg: NumericalConfig | None = None):
    cfg = cfg or NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    return Tracer(scene, library, cfg).run()


def _records(out, observation_id: str):
    return [a for a in out.arrivals if a.observation_id == observation_id]


def _fingerprint(record) -> tuple:
    """Every number of one arrival record that the order of the declarations must not change, bit for bit."""
    return (record.path_id, record.crossing_index, record.power_W, record.opl_mm, record.length_mm, record.position.tobytes(),
            record.direction.tobytes(), record.uv, record.pupil_points.tobytes(), record.pupil_launch.tobytes(),
            None if record.transfer is None else record.transfer.tobytes(), record.counted, record.detected_power_W)


def _by_plane(out) -> dict[str, list[tuple]]:
    planes: dict[str, list[tuple]] = {}
    for record in out.arrivals:
        planes.setdefault(record.observation_id, []).append(_fingerprint(record))
    return {oid: sorted(entries, key=lambda e: (e[0], e[1])) for oid, entries in planes.items()}


def _assert_whole_beam_on_its_plane(record, *, power_W: float = SOURCE_W) -> None:
    """The arrival carries the beam's power, every pupil ray and the first-order transfer, its chief and pupil points lie on
    its own plane, and the pupil footprint is centred on the chief ray (the collimated beam is symmetric about it)."""
    assert record.power_W == pytest.approx(power_W, rel=1e-12)
    assert len(record.pupil_points) == record.pupil_total > 0
    assert record.transfer is not None, record.transfer_issue
    assert abs(float((record.position - record.plane_origin) @ record.plane_normal)) <= 1e-12
    assert np.max(np.abs((record.pupil_points - record.plane_origin) @ record.plane_normal)) <= 1e-12
    centroid = record.pupil_points.mean(axis=0) - record.position
    assert np.max(np.abs(centroid)) <= 1e-12


# ----------------------------------------------------------------------------------------------- 1. the issue's fixture

# intersecting-order-1.json (recorded 2026-10-24; SHA-256 2a7715b5...): B declared before A
_FIXTURE = {
    "schema": "opensci.project", "project_schema_version": "2.0",
    "components": [{
        "instance_id": "S", "optical_role": "source", "custom_part": {"custom_part_key": "S", "description": "S"},
        "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 852, "power_w": 0.001, "waist_radius_mm": 0.5,
                         "waist_position_mm": 0, "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}},
    }],
    "observations": [
        {"observation_id": "B", "pose": {"position_mm": [100.0, 0.0, 0.0],
                                         "quaternion_wxyz": [0.27059805007309845, 0.2705980500730985, 0.6532814824381882, 0.6532814824381882]},
         "aperture": {"shape": "circle", "radius_mm": 12}, "absorbing": False},
        {"observation_id": "A", "pose": {"position_mm": [100.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
         "aperture": {"shape": "circle", "radius_mm": 12}, "absorbing": False},
    ],
    "targets": [{"target_id": "strict", "observation_id": "A", "metric": "power_W", "maximum": 1e-09}],
}


@pytest.mark.parametrize("reverse", (False, True), ids=("B,A", "A,B"))
def test_the_issue_fixture_records_the_milliwatt_on_both_planes_and_the_nanowatt_maximum_is_unmet(reverse: bool):
    from local_helpers import arrivals as local_arrivals, run as local_run  # noqa: PLC0415

    document = copy.deepcopy(_FIXTURE)
    if reverse:
        document["observations"].reverse()
    local = local_run(document)
    for observation_id in ("A", "B"):
        (arrival,) = local_arrivals(local.result, observation_id)
        assert arrival.beam.power_W == pytest.approx(1e-3, rel=1e-12)
        assert arrival.beam.geometric_length_mm == pytest.approx(100.0, abs=1e-9)
        assert arrival.crossing_index == 0
        assert arrival.pupil_rays_arrived == arrival.pupil_ray_count > 0
    (target,) = local.summary()["targets"]
    assert target["value"] == pytest.approx(1e-3, rel=1e-12)
    assert target["assessment"] == "UNMET"
    assert "NO_TRACED_ARRIVAL" not in target["issue_codes"]
    assert "OBSERVATION_NOT_REACHED" not in issue_codes(local.result)


@pytest.mark.parametrize("deg", (0.5, 10.0, 45.0, 60.0, 80.0))
def test_two_planes_through_one_point_each_record_the_whole_beam_in_either_order(deg: float):
    reference = None
    for order in (("A", "B"), ("B", "A")):
        planes = {"A": _plane("A"), "B": _plane("B", normal=_in_xy(deg))}
        out = _trace(ProjectSnapshot(components=(_laser(),), observations=tuple(planes[o] for o in order)))
        for observation_id in ("A", "B"):
            (record,) = _records(out, observation_id)
            _assert_whole_beam_on_its_plane(record)
            assert record.crossing_index == 0
            assert record.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
            assert record.length_mm == pytest.approx(10.0, abs=1e-12)
        if reference is None:
            reference = _by_plane(out)
        else:
            assert _by_plane(out) == reference
    # the beam goes on unchanged: one path, out of the scene
    assert [p.termination for p in out.paths] == ["ESCAPED"]


def test_a_star_of_three_planes_through_one_point_records_the_beam_three_times_in_every_order():
    planes = (_plane("A"), _plane("B", normal=DIAGONAL), _plane("C", normal=_unit((1.0, -0.3, 0.8))))
    reference = None
    for order in permutations(planes):
        out = _trace(ProjectSnapshot(components=(_laser(),), observations=order))
        assert sorted(r.observation_id for r in out.arrivals) == ["A", "B", "C"]
        for record in out.arrivals:
            _assert_whole_beam_on_its_plane(record)
        if reference is None:
            reference = _by_plane(out)
        else:
            assert _by_plane(out) == reference


def test_the_issue_geometry_through_validate_fails_the_nanowatt_maximum_in_either_order():
    target = Target(target_id="strict", metric="power_W", observation_id="A", maximum=1e-9, unit="W")
    for order in (("B", "A"), ("A", "B")):
        planes = {"A": _plane("A"), "B": _plane("B", normal=DIAGONAL)}
        result = run(ProjectSnapshot(components=(_laser(),), observations=tuple(planes[o] for o in order), targets=(target,)))
        assert one_arrival(result, "A").beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)
        assert one_arrival(result, "B").beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)
        (evaluated,) = result.targets
        assert evaluated.value == pytest.approx(SOURCE_W, rel=1e-12)
        assert evaluated.assessment.value == "UNMET"
        assert "OBSERVATION_NOT_REACHED" not in issue_codes(result)


# ---------------------------------------------------------------------------- 2. coplanar and opposite planes at one place


def test_coplanar_opposite_and_tilted_planes_at_one_place_in_every_order():
    planes = (_plane("same_1"), _plane("same_2"), _plane("flipped", normal=MX), _plane("tilted", normal=DIAGONAL))
    reference = None
    for order in permutations(planes):
        result = run(ProjectSnapshot(components=(_laser(),), observations=order))
        by_id = {o.observation_id: o.arrivals for o in result.observations}
        assert all(len(found) == 1 for found in by_id.values())
        same_1, same_2, flipped = by_id["same_1"][0], by_id["same_2"][0], by_id["flipped"][0]
        assert same_1 == same_2
        assert flipped.beam.power_W == same_1.beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)
        assert flipped.beam.optical_path_length_mm == same_1.beam.optical_path_length_mm
        assert flipped.direction_uvn[2] == pytest.approx(-same_1.direction_uvn[2])
        assert all(found[0].pupil_rays_arrived == found[0].pupil_ray_count for found in by_id.values())
        if reference is None:
            reference = by_id
        else:
            assert by_id == reference


# ------------------------------------------------------------------------------------------ 3. nearby but distinct planes


@pytest.mark.parametrize("offset_mm", (5e-8, 1e-6, 1e-3, 1e-2), ids=("0.05nm", "1nm", "1um", "10um"))
@pytest.mark.parametrize("normal", (X, DIAGONAL), ids=("parallel", "tilted"))
def test_nearby_planes_each_record_their_own_crossing(offset_mm: float, normal):
    """B is ``offset_mm`` farther along the beam than A (0.05 nm is inside the 0.1 nm ray-advance guard, which used to hide
    it): each records its own crossing -- not collapsed onto A's -- with the whole beam, in either order."""
    reference = None
    for order in (("A", "B"), ("B", "A")):
        planes = {"A": _plane("A"), "B": _plane("B", position=(offset_mm, 0.0, 0.0), normal=normal)}
        out = _trace(ProjectSnapshot(components=(_laser(),), observations=tuple(planes[o] for o in order)))
        (a,) = _records(out, "A")
        (b,) = _records(out, "B")
        _assert_whole_beam_on_its_plane(a)
        _assert_whole_beam_on_its_plane(b)
        assert a.length_mm == pytest.approx(10.0, abs=1e-12)
        assert b.length_mm == pytest.approx(10.0 + offset_mm, abs=1e-12)
        assert b.opl_mm - a.opl_mm == pytest.approx(offset_mm, abs=1e-12)
        assert b.position[0] - a.position[0] == pytest.approx(offset_mm, abs=1e-12)
        if reference is None:
            reference = _by_plane(out)
        else:
            assert _by_plane(out) == reference


# --------------------------------------------------------------------------------- 4. coincidence with physical surfaces


def _arrival_keys(result, observation_id: str):
    return sorted((a.beam.path_id, a.crossing_index, round(a.beam.power_W, 15), tuple(round(v, 9) for v in a.beam.direction))
                  for a in arrivals(result, observation_id))


@pytest.mark.parametrize("reverse", (False, True))
def test_two_nonparallel_planes_through_a_mirror_hit_point_record_as_if_on_its_incident_side(reverse: bool):
    """A 45 deg fold mirror at the origin; A (normal +X) and T (normal tilted, the reflected beam crosses it back) both
    through the hit point. Each records what it records 1 um before the mirror on the incident side, in either order."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=R_MIRROR)
    tilt = _unit((1.0, -0.5, 0.0))

    def project(shift: float) -> ProjectSnapshot:
        planes = (_plane("A", position=(-shift, 0.0, 0.0)), _plane("T", position=(-shift, 0.0, 0.0), normal=tilt))
        return ProjectSnapshot(components=(_laser(), fold), observations=planes[::-1] if reverse else planes)

    at, before = run(project(0.0)), run(project(1e-3))
    assert "OBSERVATION_NOT_REACHED" not in issue_codes(at)
    for observation_id in ("A", "T"):
        assert _arrival_keys(at, observation_id) == _arrival_keys(before, observation_id)
    assert [(a.crossing_index, a.beam.power_W) for a in arrivals(at, "A")] == [(0, pytest.approx(SOURCE_W))]
    assert sorted((a.crossing_index, a.beam.power_W) for a in arrivals(at, "T")) == [
        (0, pytest.approx(SOURCE_W)), (1, pytest.approx(SOURCE_W * R_MIRROR))]


# ------------------------------------------------------------------------------ 5. pupil-ray crossing geometry, per ray


def _straight_line_crossings(ray_path, plane_id: str, origin, normal):
    """For every pupil ray of ``ray_path`` that reached ``plane_id``: its recorded point and OPL there, and the crossing of
    the straight line from its launch point along its launch direction with the plane (point, OPL = launch OPL + t)."""
    origin, normal = np.array(origin, dtype=float), np.array(normal, dtype=float)
    rows = []
    for ray in ray_path.pupil_rays:
        if f"observation/{plane_id}" not in ray.surface_ids:
            continue
        k = ray.surface_ids.index(f"observation/{plane_id}")
        p0, d0 = np.array(ray.points_mm[0]), np.array(ray.directions[0])
        t = float((origin - p0) @ normal) / float(d0 @ normal)
        rows.append((np.array(ray.points_mm[k]), ray.opl_mm[k], p0 + t * d0, ray.opl_mm[0] + t))
    return rows


@pytest.mark.parametrize("normal", (DIAGONAL, _unit((1.0, 0.4, -0.7))))
def test_each_pupil_ray_is_recorded_where_it_crosses_its_plane(normal):
    """A diverging beam (waist 0.05 mm, 20 mm before the source: wavefront radius 25 mm at the source) crosses A (+X) and a
    tilted plane through the same point: every pupil ray on each plane at its own straight-line crossing, with its own
    optical path length (air, n = 1) -- in either declaration order."""
    for order in (("A", "T"), ("T", "A")):
        planes = {"A": _plane("A"), "T": _plane("T", normal=normal)}
        source = _laser(waist_radius_mm=0.05, waist_position_z_mm=-20.0)
        project = ProjectSnapshot(components=(source,), observations=tuple(planes[o] for o in order))
        result, trace = validate_with_ray_paths(ValidationRequest(
            project=project, configuration=EngineConfiguration(run_layout=False, run_control=False)))
        (ray_path,) = trace.paths
        assert {ray.end for ray in ray_path.pupil_rays} == {"PATH_END"}
        for plane_id, plane_normal in (("A", X), ("T", normal)):
            rows = _straight_line_crossings(ray_path, plane_id, (0.0, 0.0, 0.0), plane_normal)
            assert len(rows) == ray_path.pupil_ray_count > 0
            spread = max(float(np.linalg.norm(point - rows[0][0])) for point, *_ in rows)
            assert spread > 0.01  # the rays meet the plane at different points
            for point, opl, expected_point, expected_opl in rows:
                assert point == pytest.approx(expected_point, abs=1e-9)
                assert opl == pytest.approx(expected_opl, abs=1e-9)
            arrival = one_arrival(result, plane_id)
            assert arrival.pupil_rays_arrived == arrival.pupil_ray_count


def test_the_tilted_plane_footprint_is_the_beam_projected_onto_it():
    """On a 45 deg plane through the crossing, the collimated beam's pupil footprint is stretched by sqrt(2) across the
    tilt: its RMS radius is sqrt(3/2) times the one on the beam-normal plane through the same point."""
    result = run(ProjectSnapshot(components=(_laser(),), observations=(_plane("B", normal=DIAGONAL), _plane("A"))))
    a, b = one_arrival(result, "A"), one_arrival(result, "B")
    assert b.pupil_rms_radius_mm == pytest.approx(a.pupil_rms_radius_mm * math.sqrt(1.5), rel=1e-9)
    assert b.beam.optical_path_length_mm == pytest.approx(a.beam.optical_path_length_mm, abs=1e-12)


# --------------------------------------------------------------------------------------- 6. absorbing-plane precedence


@pytest.mark.parametrize("detector_normal", (X, DIAGONAL), ids=("parallel", "tilted"))
@pytest.mark.parametrize("ids", (("A_monitor", "Z_detector"), ("Z_monitor", "A_detector")))
@pytest.mark.parametrize("reverse", (False, True))
def test_a_transparent_plane_tied_with_a_detector_samples_first_and_the_detector_stops_the_beam(detector_normal, ids, reverse):
    """Tied with a parallel detector, the monitor records the whole beam. Tied with a tilted one (through the same point),
    the rays on one side meet the detector first: the monitor records only the rays that reach it no later than the
    detector, and its power -- the beam's, which follows the chief ray -- is bounded below by the beam's less the share the
    detector takes first (review m1). Either way the
    detector detects the beam and nothing goes on behind it."""
    monitor_id, detector_id = ids
    planes = (_plane(monitor_id), _plane(detector_id, normal=detector_normal, absorbing=True), _plane("behind", position=(50.0, 0.0, 0.0)))
    targets = (Target(target_id="mon", metric="power_W", observation_id=monitor_id, minimum=0.9e-3, unit="W"),)
    result = run(ProjectSnapshot(components=(_laser(),), observations=planes[::-1] if reverse else planes, targets=targets))
    monitor, detector = one_arrival(result, monitor_id), one_arrival(result, detector_id)
    assert monitor.detected_power_W is None
    (target,) = result.targets
    if detector_normal == X:
        assert monitor.beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)
        assert monitor.pupil_rays_arrived == monitor.pupil_ray_count
        assert monitor.power_out_of_scope_at == ()
        assert target.assessment.value == "MET"
    else:
        assert 0 < monitor.pupil_rays_arrived < monitor.pupil_ray_count
        # the half of the beam the detector meets first is a bounded share: the monitor's power lies in [P - S, P]
        assert monitor.power_out_of_scope_at == () and monitor.power_taken_before_at == (f"observation/{detector_id}",)
        assert monitor.power_taken_before_upper_W >= 0.5 * SOURCE_W
        assert target.assessment.value == "INDETERMINATE" and "POWER_TAKEN_BEFORE_PLANE" in target.issue_codes
        assert f"observation/{detector_id} takes at most" in target.message
    assert detector.detected_power_W == pytest.approx(SOURCE_W, rel=1e-12)
    assert arrivals(result, "behind") == []
    (path,) = result.paths
    assert path.termination == "DETECTED"
    assert path.termination_surface_id == f"observation/{detector_id}"
    assert dict(result.power_budget.total_sinks)["DETECTED"] == pytest.approx(SOURCE_W, rel=1e-12)


def test_a_detector_the_beam_misses_does_not_stop_its_rays_on_the_way_to_the_next_surface():
    """An absorbing plane whose active area the beam misses passes it unchanged: its pupil rays, which the plane still
    crosses ahead of where they left the source, travel on with the chief ray through a lens to a plane behind it."""
    detector = _plane("dump", position=(-3.0, 3.0, 0.0), normal=DIAGONAL, radius_mm=1.0, absorbing=True)
    lens = B.thin_lens("F", position=(20.0, 0.0, 0.0), focal_length_mm=100.0, clear_radius_mm=10.0)
    behind = _plane("behind", position=(40.0, 0.0, 0.0))
    out = _trace(ProjectSnapshot(components=(_laser(), lens), observations=(detector, behind)))
    (dump,) = _records(out, "dump")
    assert not dump.counted and dump.detected_power_W is None  # the beam is beside its active area: it passes
    (arrival,) = _records(out, "behind")
    assert arrival.power_W == pytest.approx(SOURCE_W, rel=1e-12)
    assert len(arrival.pupil_points) == arrival.pupil_total > 0
    assert [p.termination for p in out.paths] == ["ESCAPED"]


# --------------------------------------------------------------------------------------- 7. crossing index, double pass


@pytest.mark.parametrize("reverse", (False, True))
def test_a_retro_reflected_beam_crosses_both_tied_planes_twice(reverse: bool):
    retro = B.mirror("M", position=(20.0, 0.0, 0.0), normal=MX, reflectance=R_MIRROR)
    planes = (_plane("A"), _plane("B", normal=DIAGONAL))
    out = _trace(ProjectSnapshot(components=(_laser(), retro), observations=planes[::-1] if reverse else planes))
    for observation_id in ("A", "B"):
        records = sorted(_records(out, observation_id), key=lambda r: r.crossing_index)
        assert [r.crossing_index for r in records] == [0, 1]
        _assert_whole_beam_on_its_plane(records[0], power_W=SOURCE_W)
        _assert_whole_beam_on_its_plane(records[1], power_W=SOURCE_W * R_MIRROR)
        assert records[0].length_mm == pytest.approx(10.0, abs=1e-12)
        assert records[1].length_mm == pytest.approx(50.0, abs=1e-12)
        assert float(records[0].direction @ records[1].direction) == pytest.approx(-1.0)
    assert len(out.arrivals) == 4  # one arrival per plane per pass


# ------------------------------------------------------------------------- 8. no double count, no interaction budget


def _fold_chain(planes: tuple = ()) -> ProjectSnapshot:
    """+X onto M1 at x = 20 (to +Y), onto M2 at (20, 20) (back to +X), to the plane D at x = 60; ``planes`` are crossed
    on the way (tied pairs before M1 and between the mirrors)."""
    m1 = B.mirror("M1", position=(20.0, 0.0, 0.0), normal=(-Q, Q, 0.0))
    m2 = B.mirror("M2", position=(20.0, 20.0, 0.0), normal=(Q, -Q, 0.0))
    return ProjectSnapshot(components=(_laser(), m1, m2), observations=(*planes, _plane("D", position=(60.0, 20.0, 0.0))))


_TIED = (_plane("A1", position=(10.0, 0.0, 0.0)), _plane("B1", position=(10.0, 0.0, 0.0), normal=DIAGONAL),
         _plane("A2", position=(20.0, 10.0, 0.0), normal=(0.0, 1.0, 0.0)), _plane("B2", position=(20.0, 10.0, 0.0), normal=DIAGONAL))


@pytest.mark.parametrize("limit", (1, 2))
def test_virtual_planes_do_not_change_a_path_at_the_interaction_limit(limit: int):
    cfg = NumericalConfig(max_interactions_per_path=limit)
    plain, monitored = run(_fold_chain(), numerical=cfg), run(_fold_chain(_TIED), numerical=cfg)
    assert [(p.termination, p.termination_surface_id) for p in monitored.paths] == [
        (p.termination, p.termination_surface_id) for p in plain.paths]
    assert monitored.paths[0].geometric_path_length_mm == pytest.approx(plain.paths[0].geometric_path_length_mm, abs=1e-12)
    assert dict(monitored.power_budget.total_sinks) == pytest.approx(dict(plain.power_budget.total_sinks), abs=1e-18)
    assert ("MAX_INTERACTIONS" in issue_codes(monitored)) == ("MAX_INTERACTIONS" in issue_codes(plain)) == (limit == 1)
    if limit == 1:  # the path stops at M2, after the planes between the mirrors
        assert arrivals(monitored, "D") == arrivals(plain, "D") == []
    else:
        far, reference = one_arrival(monitored, "D"), one_arrival(plain, "D")
        assert far.beam.power_W == pytest.approx(reference.beam.power_W, rel=1e-12)
        assert far.beam.geometric_length_mm == pytest.approx(reference.beam.geometric_length_mm, abs=1e-12)
        assert far.beam.position_mm == pytest.approx(reference.beam.position_mm, abs=1e-12)
        assert far.pupil_rays_arrived == reference.pupil_rays_arrived == reference.pupil_ray_count
    for observation_id in ("A1", "B1", "A2", "B2"):
        (arrival,) = arrivals(monitored, observation_id)  # one arrival each, whatever the limit
        assert arrival.beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)
        assert arrival.pupil_rays_arrived == arrival.pupil_ray_count
