"""Numerical robustness: parallel / near-parallel rays, grazing incidence, zero-length vectors, degenerate
surfaces, TIR boundary, aperture edge, floating-point tolerance boundary, invalid material range, zero power, missing
coherence. Errors are structured; NaN / Inf never propagate."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from os_optics import SCHEMA_VERSION, EngineConfiguration, ValidationRequest, validate, validate_json
from os_optics import builders as B
from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.contracts import (
    CircularAperture, EngineeringComponentSnapshot, JonesVector, MirrorInteraction, PlaneSurface, Pose3D, ProjectSnapshot,
    RefractInteraction, SphereSurface,
)

from helpers import issue_codes, one_arrival, run

INV = 1.0 / math.sqrt(2.0)


def _walls(res):
    """Assert the result serializes to strict JSON (no NaN/Infinity tokens)."""
    text = res.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c} in result"))


# ---- parallel / near-parallel rays -------------------------------------------------------------------------------------------
def test_ray_parallel_to_a_surface_plane_never_hits_it():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.mirror("M", position=(50, 0, 20), normal=(0, 1, 0), clear_radius_mm=10.0)),  # mirror plane contains the ray
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    assert one_arrival(res, "d").beam.geometric_length_mm == pytest.approx(100.0)  # passes by, no NaN
    _walls(res)


def test_near_parallel_grazing_incidence_warn_and_fail_thresholds():
    def outcome(aoi_deg):
        a = math.radians(aoi_deg)
        n = (-math.cos(a), math.sin(a), 0.0)  # angle between beam (+x) and the normal's opposite = aoi
        proj = ProjectSnapshot(
            components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.1, position=(0, 0, 20), direction=(1, 0, 0)),
                        B.mirror("M", position=(50, 0, 20), normal=n, clear_radius_mm=200.0)),
            observations=(B.observation_plane("d", position=(50, 100, 20), normal=(0, -1, 0), radius_mm=1e4),),
        )
        return run(proj)

    warn = outcome(86.0)
    assert "GRAZING_INCIDENCE" in issue_codes(warn) and warn.issues[0].status.value == "WARNING"
    assert warn.observations[0].arrivals  # still traced
    fail = outcome(89.95)
    codes = {(i.code.value, i.status.value) for i in fail.issues}
    # (a model limit for the light behind the surface, not a contradiction in the scene: OUT_OF_MODEL_SCOPE)
    assert ("GRAZING_INCIDENCE", "OUT_OF_MODEL_SCOPE") in codes and fail.paths[0].termination == "GRAZING_INCIDENCE"
    assert fail.outcome.execution.outcome.value == "COMPLETED"
    for r in (warn, fail):
        _walls(r)


def test_ray_exactly_parallel_to_thin_lens_plane_is_a_structured_error():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.1, position=(0, 5, 20), direction=(1, 0, 0)),
                    B.thin_lens("Lens", focal_length_mm=50.0, clear_radius_mm=200.0, position=(50, 0, 20), axis=(0, 1, 0))),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    # lens plane contains the beam direction: no intersection at all (parallel), the beam passes
    assert res.observations[0].arrivals
    _walls(res)


# ---- zero-length / non-unit vectors, degenerate surfaces ----------------------------------------------------------------------
def _bad_plane(normal):
    return EngineeringComponentSnapshot(
        instance_id="X", surfaces=(PlaneSurface(surface_id="s", origin_mm=(10, 0, 20), normal=normal, aperture=CircularAperture(radius_mm=5.0),
                                                interaction=MirrorInteraction()),))


def test_zero_length_surface_normal_is_rejected_with_code_and_location():
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                                       _bad_plane((0.0, 0.0, 0.0))))
    res = run(proj)
    assert res.status.value == "FAIL" and res.issues[0].code.value == "ZERO_LENGTH_VECTOR"
    assert res.issues[0].location == "component:X/surface:s" and res.manifest is not None
    _walls(res)


def _rounded_scene(normal, quaternion, axis, jones):
    """A source (Jones vector ``jones``) folded by a mirror (``normal``) through a polarizer (pose ``quaternion``,
    transmission axis ``axis``) onto a plane: every stated direction reaches the result."""
    from os_optics.contracts import PolarizerInteraction

    mirror = EngineeringComponentSnapshot(instance_id="M", pose=Pose3D(position_mm=(50.0, 0.0, 20.0)), surfaces=(
        PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=normal, aperture=CircularAperture(radius_mm=5.0),
                     interaction=MirrorInteraction()),))
    polarizer = EngineeringComponentSnapshot(
        instance_id="P", pose=Pose3D(position_mm=(50.0, 40.0, 20.0), quaternion_wxyz=quaternion), surfaces=(
            PlaneSurface(surface_id="p", origin_mm=(0, 0, 0), normal=(0.0, 1.0, 0.0), aperture=CircularAperture(radius_mm=5.0),
                         interaction=PolarizerInteraction(axis_local=axis)),))
    laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0),
                    polarization=JonesVector(ex=(jones[0], 0.0), ey=(jones[1], 0.0)))
    return ProjectSnapshot(components=(laser, mirror, polarizer),
                           observations=(B.observation_plane("d", position=(50, 100, 20), normal=(0, -1, 0), radius_mm=10.0),))


def _assert_numbers_agree(a, b, rel):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_numbers_agree(a[key], b[key], rel)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_numbers_agree(x, y, rel)
    elif isinstance(a, float) and not isinstance(a, bool):
        assert y_close(a, b, rel), (a, b)
    else:
        assert a == b


def y_close(a: float, b: float, rel: float) -> bool:
    # relative, with an absolute floor for numbers that are differences of nearly equal ones (offsets near zero: 1e-15 of
    # the scene's 100 mm)
    return abs(a - b) <= rel * max(abs(a), abs(b)) + 1e-13


def test_a_rounded_direction_quaternion_and_jones_vector_are_taken_by_their_direction():
    # written with four decimals (|v| off 1 by ~1e-5, beyond the 1e-9 / 1e-6 tolerances): the result equals the one of the
    # full-precision normalized inputs to 1e-12
    q = (0.9239, 0.0, 0.3827, 0.0)
    qn = tuple(c / math.sqrt(sum(x * x for x in q)) for c in q)
    rounded = run(_rounded_scene((-0.7071, 0.7071, 0.0), q, (0.7071, 0.0, 0.7071), (0.7071, 0.7071)))
    exact = run(_rounded_scene((-INV, INV, 0.0), qn, (INV, 0.0, INV), (INV, INV)))
    assert rounded.status.value == exact.status.value and not {"NON_UNIT_VECTOR", "NON_UNIT_QUATERNION", "INVALID_INPUT"} & issue_codes(rounded)
    arrival = one_arrival(rounded, "d")
    # independent reference: the polarizer passes cos^2 of the angle between the field and its axis (both normalized)
    assert 0.0 < arrival.beam.power_W < 1.0
    _assert_numbers_agree(json.loads(rounded.observations[0].model_dump_json()),
                          json.loads(exact.observations[0].model_dump_json()), 1e-12)
    # a direction stated at another length is the same direction
    long = run(_rounded_scene((-2.0, 2.0, 0.0), tuple(3.0 * c for c in qn), (5.0, 0.0, 5.0), (3.0, 3.0)))
    _assert_numbers_agree(json.loads(long.observations[0].model_dump_json()),
                          json.loads(exact.observations[0].model_dump_json()), 1e-12)


def test_a_zero_quaternion_or_jones_vector_is_refused_with_code_and_location():
    from os_optics.contracts.geometry import Pose3D as P

    zero_q = P.model_construct(position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=(0.0, 0.0, 0.0, 0.0))
    comp = EngineeringComponentSnapshot.model_construct(**{**EngineeringComponentSnapshot(instance_id="X").__dict__, "pose": zero_q})
    res = run(ProjectSnapshot(components=(comp,)))
    assert res.issues[0].code.value == "ZERO_LENGTH_VECTOR" and res.issues[0].location == "component:X"
    res = run(_rounded_scene((-INV, INV, 0.0), (1.0, 0.0, 0.0, 0.0), (INV, 0.0, INV), (0.0, 0.0)))
    assert res.issues[0].code.value == "INVALID_INPUT" and "non-zero" in res.issues[0].message


def test_degenerate_aperture_axis_parallel_to_normal():
    from os_optics.contracts import RectangularAperture

    s = PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=(0, 0, 1), aperture=RectangularAperture(half_width_mm=1.0, half_height_mm=1.0),
                     aperture_u_axis=(0.0, 0.0, 1.0), interaction=MirrorInteraction())
    res = run(ProjectSnapshot(components=(EngineeringComponentSnapshot(instance_id="X", surfaces=(s,)),)))
    assert res.issues[0].code.value == "DEGENERATE_SURFACE"


def test_zero_length_cap_axis_and_polarizer_axis_rejected():
    sph = SphereSurface(surface_id="s", center_mm=(0, 0, 0), radius_mm=10.0, cap_axis=(0.0, 0.0, 0.0), aperture_radius_mm=5.0,
                        interaction=RefractInteraction())
    res = run(ProjectSnapshot(components=(EngineeringComponentSnapshot(instance_id="X", surfaces=(sph,)),)))
    assert res.issues[0].code.value == "ZERO_LENGTH_VECTOR"
    from os_optics.contracts import PolarizerInteraction

    pol = PlaneSurface(surface_id="p", origin_mm=(0, 0, 0), normal=(0, 0, 1), interaction=PolarizerInteraction(axis_local=(0.0, 0.0, 0.0)))
    res = run(ProjectSnapshot(components=(EngineeringComponentSnapshot(instance_id="X", surfaces=(pol,)),)))
    assert res.issues[0].code.value == "ZERO_LENGTH_VECTOR"


def test_polarizer_axis_parallel_to_ray_is_degenerate_not_nan():
    from os_optics.contracts import PolarizerInteraction

    surf = PlaneSurface(surface_id="p", origin_mm=(0, 0, 0), normal=(0, 0, 1), aperture=CircularAperture(radius_mm=5.0),
                        interaction=PolarizerInteraction(axis_local=(0.0, 0.0, 1.0)))  # axis along the beam
    comp = EngineeringComponentSnapshot(instance_id="P", pose=Pose3D.from_axis_direction((20, 0, 20), (1, 0, 0)), surfaces=(surf,))
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)), comp),
                           observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=5.0),))
    res = run(proj)
    assert "DEGENERATE_SURFACE" in issue_codes(res) and res.paths[0].termination == "DEGENERATE_SURFACE"
    _walls(res)


def test_nan_and_inf_inputs_are_rejected_by_the_public_contract():
    good = ValidationRequest(project=ProjectSnapshot()).model_dump_json()
    doc = json.loads(good)
    doc["project"]["components"] = [{"instance_id": "A", "pose": {"position_mm": [0, 0, 1e999], "quaternion_wxyz": [1, 0, 0, 0]}}]
    out = json.loads(validate_json(json.dumps(doc)))
    assert out["status"] == "FAIL" and out["issues"][0]["code"] == "INVALID_INPUT"
    assert out["manifest"]["result_hash"]
    out2 = json.loads(validate_json('{"project": {"components": [{"instance_id": "A", "pose": {"position_mm": [NaN, 0, 0]}}]}}'))
    assert out2["status"] == "FAIL"
    out3 = json.loads(validate_json("not json at all"))
    assert out3["status"] == "FAIL" and out3["issues"][0]["code"] == "INVALID_INPUT"
    out4 = json.loads(validate_json('{"project": {}, "surprise": 1}'))
    assert out4["status"] == "FAIL"  # unknown top-level field rejected (extra=forbid)


def test_unsupported_schema_version():
    major = int(SCHEMA_VERSION.split(".")[0])
    res = validate(ValidationRequest(project=ProjectSnapshot(project_schema_version=f"{major + 1}.0.0")))
    assert res.status.value == "FAIL" and res.issues[0].code.value == "UNSUPPORTED_SCHEMA_VERSION"
    ok = validate(ValidationRequest(project=ProjectSnapshot(project_schema_version=f"{major}.0.3")))  # same major: accepted
    assert ok.issues == ()


# ---- aperture edge / floating-point tolerance boundary ---------------------------------------------------------------------------
@pytest.mark.parametrize("delta,clipped", [(-1e-6, False), (0.0, False), (0.5e-9, False), (2e-9, True), (1e-6, True)])
def test_aperture_edge_is_inclusive_within_declared_tolerance(delta, clipped):
    a = 2.0
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.05, position=(0, a + delta, 20), direction=(1, 0, 0)),
                    B.aperture_stop("Stop", position=(30, 0, 20), axis=(1, 0, 0), radius_mm=a)),
        observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    assert ("CLIPPED" in issue_codes(res)) is clipped
    (arrival,) = res.observations[0].arrivals
    if clipped:
        # the chief ray just beyond the edge: the half of the beam inside the opening goes on, carried by its pupil rays
        # that pass, one of them its reference ray from the stop on
        assert arrival.reference_ray_promoted_at == ("Stop/stop",)
        assert 0.45 < arrival.beam.power_W / 1e-3 < 0.5
    else:  # (the chief ray on the edge within the tolerance: the Gaussian state passes about half the beam on its own)
        assert arrival.reference_ray_promoted_at == () and 0.45 < arrival.beam.power_W / 1e-3 < 0.5


@pytest.mark.parametrize("gap,hit", [(5e-8, False), (2e-7, True), (1e-3, True)])
def test_ray_min_advance_boundary(gap, hit):
    """A surface closer than ray_min_advance_mm (1e-7 mm) to the start point is not re-hit (self-intersection guard)."""
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.1, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.mirror("M", position=(gap, 0, 20), normal=(-1, 0, 0), clear_radius_mm=5.0)),
        observations=(B.observation_plane("back", position=(-10, 0, 20), normal=(1, 0, 0), radius_mm=5.0),),
    )
    res = run(proj)
    reflected = [a for a in res.observations[0].arrivals if a.beam.direction[0] < 0]
    assert bool(reflected) is hit
    _walls(res)


# ---- zero power / blocked paths -----------------------------------------------------------------------------------------------------
def test_orthogonal_analyzer_zero_power_terminates_path_cleanly():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0), polarization=JonesVector.linear(90.0)),
                    B.polarizer("P", position=(20, 0, 20), axis=(1, 0, 0), angle_deg=0.0, t_perp=0.0)),
        observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
        targets=(),
    )
    res = run(proj)
    assert res.paths[0].termination == "ZERO_POWER" and res.paths[0].final_power_W <= CFG.power_epsilon_W
    # cos(90 degrees) leaves a positive floating-point remainder far below the line's negligible level: the stated truncation
    # of a known zero, its finite bound kept beside it (decision B4)
    assert "OBSERVATION_NOT_REACHED" in issue_codes(res) and res.status.value == "PASS"
    assert res.observations[0].total_power_W == 0.0 and res.observations[0].power_complete
    assert res.observations[0].quantity_state.value == "KNOWN_ZERO"
    assert 0 < res.observations[0].power_upper_bound_W < 1e-30
    _walls(res)


def test_back_side_of_a_mirror_blocks_and_is_reported():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.mirror("M", position=(30, 0, 20), normal=(1, 0, 0), clear_radius_mm=10.0)),  # reflective side faces away
        observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    assert "BLOCKED_BY_BACK_SIDE" in issue_codes(res) and res.paths[0].termination == "BLOCKED_BY_BACK_SIDE"


def test_cavity_loop_is_bounded_by_interaction_limit():
    """Two parallel mirrors facing each other at normal incidence: the beam would bounce forever -> MAX_INTERACTIONS."""
    m1 = B.mirror("M1", position=(50, 0, 20), normal=(-1, 0, 0), clear_radius_mm=10.0, reflectance=0.999999)
    m2 = B.mirror("M2", position=(20, 0, 20), normal=(1, 0, 0), clear_radius_mm=10.0, reflectance=0.999999)
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.2, position=(30, 0, 20), direction=(1, 0, 0)), m1, m2),
        observations=(),
    )
    res = run(proj)
    assert "MAX_INTERACTIONS" in issue_codes(res) and res.paths[0].termination == "MAX_INTERACTIONS"
    assert len(res.paths[0].lineage) == CFG.max_interactions_per_path


# ---- seeded fuzz: no exceptions, no NaN, valid statuses ---------------------------------------------------------------------------
def _random_project(rng: np.random.Generator) -> ProjectSnapshot:
    def vec():
        v = rng.normal(size=3)
        return tuple(float(x) for x in v / np.linalg.norm(v))

    comps = [B.laser("L", wavelength_nm=float(rng.uniform(350, 900)), waist_radius_mm=float(rng.uniform(0.05, 2.0)), position=(0, 0, 0),
                     direction=vec(), power_W=float(rng.uniform(1e-4, 1e-1)), polarization=JonesVector.linear(float(rng.uniform(0, 180))),
                     coherence_length_mm=float(rng.uniform(1, 1000)))]
    pos = np.zeros(3)
    for i in range(int(rng.integers(2, 7))):
        pos = pos + rng.uniform(5, 80) * np.array(vec())
        kind = int(rng.integers(0, 9))
        p = tuple(float(x) for x in pos)
        name = f"E{i}"
        if kind == 0:
            comps.append(B.mirror(name, position=p, normal=vec(), clear_radius_mm=float(rng.uniform(1, 15))))
        elif kind == 1:
            comps.append(B.thin_lens(name, focal_length_mm=float(rng.choice([-1, 1]) * rng.uniform(10, 300)), clear_radius_mm=float(rng.uniform(2, 15)), position=p, axis=vec()))
        elif kind == 2:
            comps.append(B.npbs(name, position=p, normal=vec(), reflectance=float(rng.uniform(0.05, 0.95))))
        elif kind == 3:
            comps.append(B.window(name, thickness_mm=float(rng.uniform(1, 10)), clear_radius_mm=float(rng.uniform(3, 15)), material="N-BK7", position=p, axis=vec()))
        elif kind == 4:
            comps.append(B.polarizer(name, position=p, axis=vec(), angle_deg=float(rng.uniform(0, 180))))
        elif kind == 5:
            comps.append(B.half_waveplate(name, position=p, axis=vec(), fast_axis_angle_deg=float(rng.uniform(0, 180)), design_wavelength_nm=532.0))
        elif kind == 6:
            comps.append(B.aperture_stop(name, position=p, axis=vec(), radius_mm=float(rng.uniform(0.3, 6))))
        elif kind == 7:
            comps.append(B.spherical_mirror(name, position=p, normal=vec(), radius_of_curvature_mm=float(rng.choice([-1, 1]) * rng.uniform(30, 500)),
                                            clear_radius_mm=float(rng.uniform(2, 12))))
        else:
            comps.append(B.pbs(name, position=p, normal=vec()))
    d0 = np.array(comps[0].pose.to_pose().rotation[:, 2])
    # one plane right behind the source is always reached; extra random planes usually are not (exercises FAIL / warnings)
    obs = [B.observation_plane("O0", position=tuple(float(x) for x in 2.0 * d0), normal=tuple(float(x) for x in -d0), radius_mm=20.0)]
    if rng.random() < 0.5:
        obs += [B.observation_plane(f"O{j}", position=tuple(float(x) for x in rng.uniform(-100, 300, 3)), normal=vec(), radius_mm=float(rng.uniform(1, 50)),
                                    acceptance_half_angle_deg=float(rng.uniform(5, 60)) if rng.random() < 0.5 else None) for j in range(1, 3)]
    obs = tuple(obs)
    return ProjectSnapshot(components=tuple(comps), observations=obs)


def test_seeded_fuzz_never_raises_and_never_leaks_non_finite_numbers():
    rng = np.random.default_rng(20260920)  # test-only RNG, fixed seed
    statuses = set()
    for _ in range(150):
        proj = _random_project(rng)
        res = validate(ValidationRequest(project=proj, configuration=EngineConfiguration(run_layout=False, run_control=False)))
        statuses.add(res.status.value)
        text = res.model_dump_json()
        assert "NaN" not in text and "Infinity" not in text
        assert res.manifest is not None and len(res.manifest.result_hash) == 64
        again = validate(ValidationRequest(project=proj, configuration=EngineConfiguration(run_layout=False, run_control=False)))
        assert again.manifest.result_hash == res.manifest.result_hash  # deterministic on random scenes too
    assert statuses <= {"PASS", "WARNING", "FAIL", "UNKNOWN", "NOT_APPLICABLE", "OUT_OF_MODEL_SCOPE"}
    assert len(statuses) >= 2  # the fuzz actually exercises different outcomes
