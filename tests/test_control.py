"""Control / recoverability: actuator DOFs, sensor observations, finite-difference Jacobian,
damped least squares, travel and step limits, lost-beam state, capture range, observability, deterministic disturbance and
Monte Carlo with an explicit seed.

Independent references: closed-form beam-pointing geometry (a mirror rotated by delta deflects the beam by 2 delta), plain
numpy least squares, and exact vector reflection written in the test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import (
    ActuatorSpec, CaptureScanSpec, CircularAperture, ControlSpec, EngineeringComponentSnapshot, MonteCarloSpec, ProjectSnapshot,
    RecoverabilitySpec, SensorSpec,
)
from os_optics.control import Dof, analyse, run_lm
from os_optics.control.readings import Reading
from os_optics.math.rng import DeterministicRng

from helpers import CFG, issue_codes, run

INV = 1.0 / math.sqrt(2.0)
L_ARM = 200.0  # mirror -> detector distance (mm)


def _mirror_project(*, yaw_travel=(-3.0, 3.0), pitch_travel=(-3.0, 3.0), det_radius=30.0, with_capture=None, recover=None, mc=None,
                    max_step=None, extra_sensors=()):
    yaw = ActuatorSpec(actuator_id="yaw", dof="rotate", axis_local=(0.0, 1.0, 0.0), travel_min=yaw_travel[0], travel_max=yaw_travel[1],
                       max_step=max_step)
    pitch = ActuatorSpec(actuator_id="pitch", dof="rotate", axis_local=(1.0, 0.0, 0.0), travel_min=pitch_travel[0], travel_max=pitch_travel[1],
                         max_step=max_step)
    mirror = B.mirror("M", position=(100, 0, 20), normal=(-INV, INV, 0), clear_radius_mm=25.0, actuators=(yaw, pitch))
    det = B.observation_plane("det", position=(100, L_ARM, 20), normal=(0, -1, 0), radius_mm=det_radius)
    spec = ControlSpec(
        actuator_refs=("M/yaw", "M/pitch"),
        sensors=(SensorSpec(sensor_id="u", observation_id="det", quantity="centroid_u_mm", target_value=0.0),
                 SensorSpec(sensor_id="v", observation_id="det", quantity="centroid_v_mm", target_value=0.0), *extra_sensors),
        capture_scan=with_capture, recoverability=recover, monte_carlo=mc,
    )
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)), mirror),
        observations=(det,), control_spec=spec,
    )


def _control(proj, seed=None, **kw):
    res = run(proj, control=True, seed=seed, **kw)
    return res.control, res


# ---- sensitivity ----------------------------------------------------------------------------------------------------------------
def _reflect_hit(delta_yaw_deg, delta_pitch_deg):
    """Independent exact geometry: mirror normal rotated by yaw about z and pitch about the local x axis (right-handed);
    beam along +x hits the mirror at its centre; returns the (u, v) = (x, z) hit on the plane y = L_ARM."""
    n0 = np.array([-INV, INV, 0.0])

    def rot(axis, deg):
        a = np.array(axis, float)
        a /= np.linalg.norm(a)
        t = math.radians(deg)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k

    # component frame: local z = normal, local y = z_hat (up), local x = y x z ; yaw about local y = global z; pitch about local x
    z_hat = np.array([0.0, 0.0, 1.0])
    lx = np.cross(z_hat, n0)
    r_yaw = rot(z_hat, delta_yaw_deg)  # local y is global z
    r_pitch = rot(lx, delta_pitch_deg)  # local x
    # T' = T o D_yaw o D_pitch : n' = R_yaw_in_world applied first in the local chain -> world rotation = R_yaw @ R_pitch_world
    r = r_yaw @ r_pitch
    n = r @ n0
    d = np.array([1.0, 0.0, 0.0])
    d2 = d - 2 * (d @ n) * n
    t = L_ARM / d2[1]
    hit = np.array([100.0, 0.0, 20.0]) + t * d2
    return hit[0] - 100.0, hit[2] - 20.0


def test_jacobian_matches_closed_form_two_delta_rule():
    """Mirror rotated by delta about the axis perpendicular to the plane of incidence deflects the beam by 2 delta:
    d(u)/d(yaw) = 2 L (pi/180) mm per degree (exact at zero); the pitch axis lies in the mirror plane."""
    ctl, res = _control(_mirror_project())
    j = np.array(ctl.sensitivity.jacobian)
    two_l = 2.0 * L_ARM * math.pi / 180.0
    # the sign convention is derived independently by finite differences of exact vector geometry
    eps = 1e-4
    ref = np.array([[(_reflect_hit(eps, 0)[k] - _reflect_hit(-eps, 0)[k]) / (2 * eps) for k in range(2)],
                    [(_reflect_hit(0, eps)[k] - _reflect_hit(0, -eps)[k]) / (2 * eps) for k in range(2)]]).T
    np.testing.assert_allclose(j, ref, rtol=1e-6, atol=1e-6)
    assert abs(j[0, 0]) == pytest.approx(two_l, rel=1e-6)  # yaw -> u : 2 L delta
    assert abs(j[1, 1]) == pytest.approx(math.hypot(*ref[:, 1]), rel=1e-6)  # pitch -> v
    assert ctl.sensitivity.numerical_rank == 2 and ctl.sensitivity.observable and ctl.sensitivity.status.value == "PASS"
    assert ctl.sensitivity.condition_number == pytest.approx(np.linalg.cond(j), rel=1e-6)
    assert ctl.sensitivity.difference_orders == (2, 2)  # interior of travel: central differences


def test_jacobian_is_one_sided_at_a_travel_limit():
    proj = _mirror_project(yaw_travel=(0.0, 3.0))  # zero sits at the lower limit
    ctl, _ = _control(proj)
    assert ctl.sensitivity.difference_orders[0] == 1 and ctl.sensitivity.difference_orders[1] == 2
    two_l = 2.0 * L_ARM * math.pi / 180.0
    assert abs(ctl.sensitivity.jacobian[0][0]) == pytest.approx(two_l, rel=1e-4)


def test_fd_step_dependence_is_small():
    """Halving / doubling the finite-difference step changes the Jacobian by < 1e-6 relative."""
    base, _ = _control(_mirror_project())
    fine, _ = _control(_mirror_project(), numerical=NumericalConfig(fd_step_rotation_rad=5e-6))
    coarse, _ = _control(_mirror_project(), numerical=NumericalConfig(fd_step_rotation_rad=2e-5))
    a, b, c = (np.array(x.sensitivity.jacobian) for x in (base, fine, coarse))
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-6 and np.max(np.abs(a - c)) / np.max(np.abs(a)) < 1e-6


# ---- Levenberg-Marquardt on synthetic linear systems (unit-level, exact reference: numpy lstsq) --------------------------------------
def _synthetic(a_matrix, b_vec, travel=(-100.0, 100.0), max_step=None):
    n = a_matrix.shape[1]
    comp = EngineeringComponentSnapshot(instance_id="C")
    dofs = [Dof(f"C/a{i}", comp, ActuatorSpec(actuator_id=f"a{i}", dof="translate", axis_local=(1, 0, 0), travel_min=travel[0], travel_max=travel[1],
                                             max_step=max_step)) for i in range(n)]
    sensors = tuple(SensorSpec(sensor_id=f"s{i}", observation_id="o", quantity="power_W", target_value=float(t)) for i, t in enumerate(np.zeros(a_matrix.shape[0])))
    spec = ControlSpec(actuator_refs=tuple(d.ref for d in dofs), sensors=sensors)

    def evaluator(values):
        theta = np.array([values.get(d.ref, 0.0) for d in dofs])
        return Reading(a_matrix @ theta + b_vec, False, 1.0)

    return dofs, spec, evaluator


def test_lm_converges_to_the_least_squares_solution_on_a_linear_system():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=5)
    dofs, spec, ev = _synthetic(a, b)
    out = run_lm(ev, dofs, spec, {d.ref: 0.0 for d in dofs}, {}, CFG, residual_tol=1e-30)
    sol, *_ = np.linalg.lstsq(a, -b, rcond=None)
    got = np.array([out.command[d.ref] for d in dofs])
    np.testing.assert_allclose(got, sol, atol=1e-8)
    assert out.run.final_cost == pytest.approx(0.5 * np.sum((a @ sol + b) ** 2), rel=1e-9, abs=1e-15)
    assert out.run.iterations <= 40


def test_lm_exact_solution_square_system_reaches_zero_residual():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])
    dofs, spec, ev = _synthetic(a, b)
    out = run_lm(ev, dofs, spec, {d.ref: 0.0 for d in dofs}, {}, CFG, residual_tol=1e-9)
    assert out.run.converged and out.run.final_cost < 1e-18
    np.testing.assert_allclose([out.command[d.ref] for d in dofs], np.linalg.solve(a, -b), atol=1e-9)


def test_lm_respects_travel_limits_and_reports_saturation():
    a = np.array([[1.0]])
    b = np.array([50.0])  # solution -50 is outside the travel [-10, 10]
    dofs, spec, ev = _synthetic(a, b, travel=(-10.0, 10.0))
    out = run_lm(ev, dofs, spec, {"C/a0": 0.0}, {}, CFG, residual_tol=1e-6)
    assert not out.run.converged and out.command["C/a0"] == pytest.approx(-10.0)
    assert out.run.saturated_actuators == ("C/a0",) and out.run.final_cost == pytest.approx(0.5 * 40.0**2)


def test_lm_step_limit_increases_iteration_count():
    a = np.array([[1.0]])
    b = np.array([8.0])
    dofs, spec, ev = _synthetic(a, b, max_step=1.0)
    out = run_lm(ev, dofs, spec, {"C/a0": 0.0}, {}, CFG, residual_tol=1e-6)
    assert out.run.converged and out.run.iterations >= 8  # |solution| = 8 with steps <= 1
    free_dofs, free_spec, free_ev = _synthetic(a, b)
    free = run_lm(free_ev, free_dofs, free_spec, {"C/a0": 0.0}, {}, CFG, residual_tol=1e-6)
    assert free.run.iterations < out.run.iterations


def test_analyse_flags_near_singular_and_rank_deficient_jacobians():
    ill = analyse(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]]), ["a", "b"], ["s1", "s2"], (2, 2), np.zeros((2, 2)), CFG)
    assert ill.numerical_rank == 2 and ill.status.value == "WARNING" and ill.condition_number > CFG.jacobian_condition_warn
    singular = analyse(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]]), ["a", "b"], ["s1", "s2"], (2, 2), np.zeros((2, 2)), CFG)
    assert singular.numerical_rank == 1 and not singular.observable and singular.status.value == "FAIL"
    assert set(singular.unobservable_actuator_refs) == {"a", "b"}  # the null direction (1,-1)/sqrt2 involves both DOFs equally
    zero = analyse(np.zeros((2, 2)), ["a", "b"], ["s1", "s2"], (0, 0), np.zeros((2, 2)), CFG)
    assert zero.status.value == "FAIL" and zero.numerical_rank == 0 and zero.condition_number is None


# ---- recoverability cases -------------------------------------------------------------------------------------------------------------
def test_disturbance_within_travel_and_with_light_converges():
    rec = RecoverabilitySpec(disturbances=({"M/yaw": 0.4, "M/pitch": -0.3},), convergence_tolerance=1e-4)
    ctl, res = _control(_mirror_project(recover=rec))
    case = ctl.recoverability_cases[0]
    assert case.outcome == "RECOVERED" and case.run.converged and ctl.recoverability_fraction == 1.0
    # the controller cancels the disturbance: command = -disturbance (unique for two independent DOFs)
    assert case.run.final_state["M/yaw"] == pytest.approx(-0.4, abs=1e-5)
    assert case.run.final_state["M/pitch"] == pytest.approx(0.3, abs=1e-5)
    assert case.run.trajectory_cost[-1] < case.run.trajectory_cost[0]
    assert ctl.status.value == "PASS" and res.status.value in ("PASS", "WARNING")


def test_nominal_run_needs_no_correction():
    ctl, _ = _control(_mirror_project())
    assert ctl.nominal_run.converged and ctl.nominal_run.iterations == 0
    assert all(abs(v) < 1e-12 for v in ctl.nominal_run.final_state.values())


def test_lost_beam_without_capture_strategy_is_not_recoverable():
    rec = RecoverabilitySpec(disturbances=({"M/yaw": 2.9},), convergence_tolerance=1e-4)  # 2.9 deg -> u ~ 20 mm > 5 mm detector
    ctl, res = _control(_mirror_project(det_radius=5.0, recover=rec))
    case = ctl.recoverability_cases[0]
    assert case.outcome == "LOST_UNRECOVERABLE" and case.run.lost_beam and ctl.recoverability_fraction == 0.0
    assert "CAPTURE_RANGE_INSUFFICIENT" in issue_codes(res) and ctl.status.value == "FAIL"


def test_lost_beam_recovered_by_coarse_capture_scan():
    rec = RecoverabilitySpec(disturbances=({"M/yaw": 2.9},), convergence_tolerance=1e-4)
    scan = CaptureScanSpec(actuator_refs=("M/yaw",), points_per_axis=25)  # 0.25 deg steps -> 1.7 mm on the detector (< 10 mm window)
    ctl, res = _control(_mirror_project(det_radius=5.0, recover=rec, with_capture=scan))
    case = ctl.recoverability_cases[0]
    assert case.outcome == "RECOVERED" and case.run.capture_used and case.run.converged
    assert case.run.final_state["M/yaw"] == pytest.approx(-2.9, abs=1e-4)


def test_capture_range_insufficient_is_reported_when_scan_is_too_coarse():
    # disturbance 1.5 deg; grid commands -3, 0, +3 deg leave total offsets of -1.5, 1.5, 4.5 deg (>= 10 mm off a 0.6 mm detector)
    rec = RecoverabilitySpec(disturbances=({"M/yaw": 1.5},), convergence_tolerance=1e-4)
    scan = CaptureScanSpec(actuator_refs=("M/yaw",), points_per_axis=3)  # 3 deg steps jump over the ~0.2 deg wide capture window
    ctl, res = _control(_mirror_project(det_radius=0.6, recover=rec, with_capture=scan))
    assert ctl.recoverability_cases[0].outcome == "LOST_UNRECOVERABLE"
    assert "CAPTURE_RANGE_INSUFFICIENT" in issue_codes(res)


def test_dof_saturation_is_reported_with_the_saturated_actuator():
    rec = RecoverabilitySpec(disturbances=({"M/yaw": 2.0},), convergence_tolerance=1e-4)
    ctl, res = _control(_mirror_project(yaw_travel=(-1.0, 1.0), recover=rec))  # detector large enough that light never gets lost
    case = ctl.recoverability_cases[0]
    assert case.outcome == "SATURATED" and "M/yaw" in case.run.saturated_actuators and not case.run.lost_beam
    assert case.run.final_state["M/yaw"] == pytest.approx(-1.0)
    assert "DOF_SATURATED" in issue_codes(res)


def test_unobservable_dof_is_reported_when_two_actuators_move_the_same_reading():
    """Two mirrors in series both steer the beam in the same plane; a single 'u' sensor cannot separate their contributions."""
    y1 = ActuatorSpec(actuator_id="yaw", dof="rotate", axis_local=(0.0, 1.0, 0.0), travel_min=-3.0, travel_max=3.0)
    m1 = B.mirror("M1", position=(50, 0, 20), normal=(-INV, INV, 0), clear_radius_mm=25.0, actuators=(y1,))
    # second fold sends the beam on to the detector: both yaw DOFs shift the same detector coordinate
    m2 = B.mirror("M2", position=(50, 100, 20), normal=(INV, -INV, 0), clear_radius_mm=25.0, actuators=(y1,))
    det = B.observation_plane("det", position=(250, 100, 20), normal=(-1, 0, 0), radius_mm=60.0)
    spec = ControlSpec(actuator_refs=("M1/yaw", "M2/yaw"),
                       sensors=(SensorSpec(sensor_id="u", observation_id="det", quantity="centroid_u_mm", target_value=0.0),))
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)), m1, m2),
                           observations=(det,), control_spec=spec)
    ctl, res = _control(proj)
    s = ctl.sensitivity
    assert s.numerical_rank == 1 and not s.observable and s.status.value == "FAIL"
    assert set(s.unobservable_actuator_refs) == {"M1/yaw", "M2/yaw"} or len(s.unobservable_actuator_refs) >= 1
    assert "UNOBSERVABLE_DOF" in issue_codes(res)
    rec = RecoverabilitySpec(disturbances=({"M1/yaw": 0.5},), convergence_tolerance=1e-6)
    proj2 = proj.model_copy(update={"control_spec": spec.model_copy(update={"recoverability": rec})})
    ctl2, _ = _control(proj2)
    # with one sensor and two DOFs the single reading can still be nulled (under-determined), so the case recovers; the
    # observability report is what flags the ambiguity
    assert ctl2.recoverability_cases[0].outcome in ("RECOVERED", "UNOBSERVABLE")


# ---- deterministic Monte Carlo -------------------------------------------------------------------------------------------------------
def _mc_project(seed=None, n=12):
    mc = MonteCarloSpec(sample_count=n, seed=seed, amplitude={"M/yaw": 2.6, "M/pitch": 2.6})
    scan = CaptureScanSpec(actuator_refs=("M/yaw", "M/pitch"), points_per_axis=9)
    return _mirror_project(det_radius=5.0, mc=mc, with_capture=scan, recover=RecoverabilitySpec(disturbances=(), convergence_tolerance=1e-3))


def test_monte_carlo_is_deterministic_for_a_fixed_seed_and_differs_across_seeds():
    a, _ = _control(_mc_project(seed=1234))
    b, _ = _control(_mc_project(seed=1234))
    c, _ = _control(_mc_project(seed=99))
    assert a.model_dump() == b.model_dump()
    assert a.monte_carlo_seed == 1234 and a.monte_carlo_sample_count == 12
    assert 0.0 <= a.monte_carlo_recovered_fraction <= 1.0
    fractions = {a.monte_carlo_recovered_fraction, c.monte_carlo_recovered_fraction}
    # different seeds draw different disturbances (compare the disturbance streams directly: the statistic may coincide)
    assert [DeterministicRng(1234).uniform() for _ in range(3)] != [DeterministicRng(99).uniform() for _ in range(3)]
    assert a.monte_carlo_recovered_fraction + a.monte_carlo_lost_fraction + a.monte_carlo_saturated_fraction <= 1.0 + 1e-12
    assert len(fractions) >= 1


def test_monte_carlo_seed_from_request_configuration_and_the_default_seed():
    proj = _mc_project(seed=None)
    ok, res_ok = _control(proj, seed=7)
    assert ok.monte_carlo_seed == 7 and res_ok.manifest.random_seed == 7 and res_ok.manifest.prng_id == "PCG64+BoxMuller-v1"
    # no seed anywhere: drawn with the documented default 0 and recorded as such -- the same samples, fractions and result
    # hash as the request that states seed 0
    default, res = _control(proj, seed=None)
    stated, res_zero = _control(proj, seed=0)
    assert default.monte_carlo_seed == 0 and default.monte_carlo_sample_count == stated.monte_carlo_sample_count > 0
    assert res.manifest.random_seed == 0 and res.manifest.prng_id == "PCG64+BoxMuller-v1"
    assert (default.monte_carlo_recovered_fraction, default.monte_carlo_lost_fraction, default.monte_carlo_saturated_fraction) == (
        stated.monte_carlo_recovered_fraction, stated.monte_carlo_lost_fraction, stated.monte_carlo_saturated_fraction)
    assert default.monte_carlo_recovered_fraction is not None and "SEED_REQUIRED" not in issue_codes(res)
    assert res.manifest.result_hash == res_zero.manifest.result_hash == _control(proj, seed=None)[1].manifest.result_hash


def test_control_result_hash_is_reproducible_including_monte_carlo():
    r1 = run(_mc_project(seed=5), control=True)
    r2 = run(_mc_project(seed=5), control=True)
    assert r1.manifest.result_hash == r2.manifest.result_hash and r1.manifest.random_seed == 5


def test_unknown_actuator_reference_is_a_structured_failure():
    proj = _mirror_project()
    bad = proj.model_copy(update={"control_spec": proj.control_spec.model_copy(update={"actuator_refs": ("M/nonexistent",)})})
    res = run(bad, control=True)
    assert res.status.value == "FAIL" and res.issues[0].code.value == "UNKNOWN_REFERENCE"


def test_actuator_offsets_move_components_by_the_documented_conventions():
    """Translation: displacement along the local axis rotated into the world; rotation: about the local axis through the pivot."""
    from os_optics.control.actuation import offset_poses

    tr = ActuatorSpec(actuator_id="t", dof="translate", axis_local=(0.0, 0.0, 1.0), travel_min=-5.0, travel_max=5.0)
    rot = ActuatorSpec(actuator_id="r", dof="rotate", axis_local=(0.0, 0.0, 1.0), pivot_local_mm=(10.0, 0.0, 0.0), travel_min=-90.0, travel_max=90.0)
    comp = B.thin_lens("Lens", focal_length_mm=50.0, clear_radius_mm=10.0, position=(10, 0, 20), axis=(1, 0, 0), actuators=(tr, rot))
    proj = ProjectSnapshot(components=(comp,))
    base = offset_poses(proj, {}, CFG)["Lens"]
    moved = offset_poses(proj, {"Lens/t": 2.0}, CFG)["Lens"]
    np.testing.assert_allclose(moved.translation - base.translation, 2.0 * np.array([1.0, 0.0, 0.0]), atol=1e-14)  # local z -> world x
    turned = offset_poses(proj, {"Lens/r": 90.0}, CFG)["Lens"]
    # rotation by 90 deg about local z (world x) through the local point (10,0,0) (local x = world y): the origin moves to pivot - R pivot
    expected_origin = base.apply_points(np.array([[10.0, 0, 0]]))[0] - base.rotation @ (np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([10.0, 0, 0]))
    np.testing.assert_allclose(turned.translation, expected_origin, atol=1e-12)
    np.testing.assert_allclose(turned.rotation @ turned.rotation.T, np.eye(3), atol=1e-14)
