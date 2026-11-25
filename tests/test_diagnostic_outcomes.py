"""Four outcomes a plane or a target can have, kept apart (G12 / G13 / G14):

* **known zero** -- every launched watt was traced to a known end and none reaches the plane: ``power_W`` is 0 W
  (``NO_TRACED_ARRIVAL``; a maximum passes, a minimum fails with that value), listed per line in ``power_by_wavelength``;
* **empty selection** -- a source, wavelength, path or crossing the run does not have: ``UNKNOWN`` / ``TARGET_SELECTION_EMPTY``;
* **incomplete trace** -- power of unknown fate (trace budget, interaction limit, out of model scope, power floor):
  ``UNKNOWN`` naming the cause, and the spectral power says it is not complete;
* **unsupported physics** -- ``OUT_OF_MODEL_SCOPE`` (see ``test_central_obstruction.py``).

Also: observation planes are virtual and never use the per-path interaction budget (G12), and partial capture or light
outside a detector's acceptance cone is a measured WARNING, with the cone applied to the detected power (G14).
The crossed-polarizer floating-point residual is omitted faint light, so its traced zero is bounded rather than known.

References are independent closed forms: Malus' law, R^k for k mirror reflections, exact leg lengths of a two-mirror cavity,
and the Gaussian power fraction 1 - exp(-2 r^2 / w^2) on a centred disc with w(z) = w0 sqrt(1 + (z / z_R)^2).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import JonesVector, ProjectSnapshot, Target
from os_optics.errors import Status
from os_optics.optics.rays.tracer import REFLECTION_EVENTS

from helpers import arrivals, assert_dark_truncated, issue_codes, one_arrival, rayleigh_range, run

X = (1.0, 0.0, 0.0)
P_W = 1e-3


def _target(target_id: str, observation_id: str, **kw) -> Target:
    return Target(target_id=target_id, metric=kw.pop("metric", "power_W"), observation_id=observation_id, unit=kw.pop("unit", "W"), **kw)


def _by_id(result):
    return {t.target_id: t for t in result.targets}


def _issue(result, code: str, location: str | None = None):
    found = [i for i in result.issues if i.code.value == code and (location is None or i.location == location)]
    assert len(found) == 1, (code, location, [(i.code.value, i.location) for i in result.issues])
    return found[0]


# ---------------------------------------------------------------------------------------------- known zero (G13)
def _crossed_polarizer(*targets: Target, absorbing: bool = False) -> ProjectSnapshot:
    """A 90 deg linear beam through an ideal 0 deg polarizer (t_perp = 0): Malus, P cos^2(90 deg) = 0 reaches the plane."""
    laser = B.laser("L", wavelength_nm=532.0, power_W=P_W, waist_radius_mm=0.3, position=(0, 0, 20), direction=X,
                    polarization=JonesVector.linear(90.0))
    pol = B.polarizer("P", position=(20, 0, 20), axis=X, angle_deg=0.0, t_perp=0.0)
    plane = B.observation_plane("d", position=(60, 0, 20), normal=X, radius_mm=10.0, absorbing=absorbing)
    return ProjectSnapshot(components=(laser, pol), observations=(plane,), targets=targets)


def test_extinguished_beam_keeps_its_rounding_remainder_bound():
    result = run(_crossed_polarizer(_target("leak_max", "d", maximum=1e-9), _target("leak_min", "d", minimum=1e-6)))
    t = _by_id(result)
    # (the light left out as negligible is the stated truncation of the 0 W both targets are graded on)
    upper = assert_dark_truncated(t["leak_max"])
    assert (t["leak_min"].status, t["leak_min"].value) == (Status.FAIL, 0.0)
    assert t["leak_min"].issue_codes == ("NO_TRACED_ARRIVAL", "FAINT_LIGHT_BOUNDED", "TARGET_NOT_MET")
    # Malus: the polarizer rejects all of it (the ~1e-36 W rounding remainder is below the ledger's resolution)
    (line,) = result.power_budget.lines
    assert line.sinks["POLARIZER_REJECTION"] == pytest.approx(P_W, rel=1e-12)
    assert line.sinks.get("BELOW_POWER_FLOOR", 0.0) <= 1e-9 * P_W
    (obs,) = result.observations
    assert obs.status is Status.PASS and obs.total_power_W == 0.0 and obs.power_complete and obs.untraced_power_W == 0.0
    assert [(s.wavelength_nm, s.power_W, s.complete) for s in obs.power_by_wavelength] == [(532.0, 0.0, True)]
    assert obs.quantity_state.value == "KNOWN_ZERO" and 0 < obs.power_upper_bound_W < 1e-30
    assert upper == pytest.approx(obs.power_upper_bound_W, rel=1e-12)
    # the plane's record does not depend on the targets: its issue reads as for a plane nothing asks about, and the
    # target result carries what the target decides
    issue = _issue(result, "OBSERVATION_NOT_REACHED", "observation:d")
    assert issue.status is Status.NOT_APPLICABLE  # (a notice: the measured zero raises no status)
    assert issue.numerical_context["faint_power_upper_bound_W"] == pytest.approx(obs.power_upper_bound_W, rel=1e-12)
    assert result.status is Status.FAIL  # through leak_min only
    only_max = run(_crossed_polarizer(_target("leak_max", "d", maximum=1e-9)))
    assert only_max.targets[0].assessment.value == "MET" and only_max.status is Status.PASS  # (the summary, beside it)


def test_a_side_plane_that_no_direct_light_reaches_passes_its_leak_bound():
    """Excitation along +x into an absorbing dump; a side detector facing it from +y receives no direct light."""
    laser = B.laser("L", wavelength_nm=532.0, power_W=P_W, waist_radius_mm=0.5, position=(-100, 0, 0), direction=X)
    dump = B.observation_plane("dump", position=(100, 0, 0), normal=X, radius_mm=10.0, absorbing=True)
    side = B.observation_plane("side", position=(0, 100, 0), normal=(0, -1, 0), radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(laser,), observations=(dump, side), targets=(
        _target("excitation", "dump", minimum=0.9 * P_W), _target("direct_light_at_side", "side", maximum=1e-12))))
    t = _by_id(result)
    assert t["excitation"].status is Status.PASS and t["excitation"].value == pytest.approx(P_W, rel=1e-12)
    assert (t["direct_light_at_side"].status, t["direct_light_at_side"].value) == (Status.PASS, 0.0)
    # (the dark plane is a measured zero: its notice raises neither the plane's status nor the run's)
    assert result.status is Status.PASS
    side_plane = next(o for o in result.observations if o.observation_id == "side")
    assert side_plane.status is Status.PASS and side_plane.quantity_state.value == "KNOWN_ZERO"
    assert _issue(result, "OBSERVATION_NOT_REACHED", "observation:side").status is Status.NOT_APPLICABLE
    assert result.power_budget.lines[0].sinks == {"DETECTED": pytest.approx(P_W, rel=1e-12)}


def test_faint_rounding_remainder_without_targets_is_a_dark_plane_and_beam_metrics_fail():
    """Nothing asks for the plane: its darkness is a measured zero with a notice, no status of the run, and the rounding
    remainder left out as negligible does not make it unknown. A metric that needs a beam (here its radius) cannot be
    measured on a dark plane and fails there."""
    plain = run(_crossed_polarizer())
    assert plain.status is Status.PASS
    assert _issue(plain, "OBSERVATION_NOT_REACHED", "observation:d").status is Status.NOT_APPLICABLE
    assert plain.observations[0].quantity_state.value == "KNOWN_ZERO"
    radius = run(_crossed_polarizer(_target("radius", "d", metric="beam_radius_mm", unit="mm", maximum=1.0)))
    (t,) = radius.targets
    assert (t.status, t.value, t.issue_codes) == (Status.FAIL, None, ("OBSERVATION_NOT_REACHED",))


# --------------------------------------------------------------------------------------------- empty selections
def _two_lines(*targets: Target) -> ProjectSnapshot:
    laser = B.laser("L", spectrum=((633.0, 0.5), (780.0, 0.5)), power_W=P_W, waist_radius_mm=0.5, position=(0, 0, 0), direction=X)
    det = B.observation_plane("det", position=(100, 0, 0), normal=X, radius_mm=10.0, absorbing=True)
    return ProjectSnapshot(components=(laser,), observations=(det,), targets=targets)


@pytest.mark.parametrize(("selector", "says"), [
    ({"source_id": "GHOST"}, "source_id 'GHOST' is not a source of this run (sources: L)"),
    ({"wavelength_nm": 700.0}, "no source line of the selected source(s) is at 700 nm (launched lines: 633 nm, 780 nm)"),
    ({"parameters": {"path_id": "L#7"}}, "path_id 'L#7' is not a beam path"),
    ({"parameters": {"crossing_index": 1}}, "crossing_index 1 does not occur"),
])
def test_a_selector_that_selects_nothing_is_unknown_not_zero(selector, says):
    result = run(_two_lines(_target("t", "det", maximum=1e-9, **selector)))
    (t,) = result.targets
    assert (t.status, t.value, t.issue_codes) == (Status.UNKNOWN, None, ("TARGET_SELECTION_EMPTY",))
    assert says in t.message
    assert result.status is Status.UNKNOWN


def test_every_launched_line_is_listed_and_valid_selectors_measure():
    result = run(_two_lines(_target("red", "det", wavelength_nm=633.0, target_value=0.5 * P_W, tolerance_rel=1e-12),
                            _target("first", "det", parameters={"path_id": "L#1", "crossing_index": 0},
                                    target_value=0.5 * P_W, tolerance_rel=1e-12)))
    assert [t.status for t in result.targets] == [Status.PASS, Status.PASS]
    (obs,) = result.observations
    assert [(s.wavelength_nm, s.complete) for s in obs.power_by_wavelength] == [(633.0, True), (780.0, True)]


# -------------------------------------------------------------------------------------------- incomplete trace
def test_light_the_path_budget_left_untraced_makes_a_dark_plane_unknown_not_failed():
    """Source B is not launched (max_paths = 1): the plane only B could reach is UNKNOWN (MAX_PATHS), and so is the run."""
    a = B.laser("A", wavelength_nm=780.0, power_W=P_W, waist_radius_mm=0.5, position=(-100, 0, 0), direction=X)
    b = B.laser("B", wavelength_nm=780.0, power_W=0.5 * P_W, waist_radius_mm=0.5, position=(-100, 30, 0), direction=X)
    det_b = B.observation_plane("det_b", position=(50, 30, 0), normal=X, radius_mm=5.0, absorbing=True)
    project = ProjectSnapshot(components=(a, b), observations=(det_b,), targets=(_target("b_max", "det_b", maximum=1e-9),))
    result = run(project, numerical=NumericalConfig(max_paths=1))
    (t,) = result.targets
    assert (t.status, t.value, t.issue_codes) == (Status.UNKNOWN, None, ("MAX_PATHS",))
    (obs,) = result.observations
    assert obs.status is Status.UNKNOWN and not obs.power_complete
    assert obs.untraced_power_W == pytest.approx(0.5 * P_W, rel=1e-12)
    # nothing arrived and the power is not known: null, not 0.0 (a reader of the value alone must not see 0 W)
    assert obs.total_power_W is None
    assert [(s.power_W, s.complete) for s in obs.power_by_wavelength] == [(None, False)]
    issue = _issue(result, "OBSERVATION_NOT_REACHED", "observation:det_b")
    assert issue.status is Status.UNKNOWN and issue.numerical_context["untraced_causes"] == ["MAX_PATHS"]
    assert result.status is Status.UNKNOWN
    full = run(project)  # traced completely, B reaches the plane: the leak bound fails with B's power
    assert (full.targets[0].status, full.targets[0].value) == (Status.FAIL, pytest.approx(0.5 * P_W, rel=1e-12))


def test_light_a_budget_stopped_is_never_left_out_however_faint():
    """The negligible level applies to a line's unknown-fate ledger power except what a budget stopped: 1e-15 of a 1 W line
    at the path or interaction limit keeps its unknown fate, while the same power out of model scope is left out."""
    from os_optics.contracts.results import PowerBudgetLine
    from os_optics.validation.power import line_fate
    from os_optics.validation.untraced import left_out_ledger_W

    cfg = NumericalConfig()
    for kind, kept in (("PATH_LIMIT", True), ("INTERACTION_LIMIT", True), ("TERMINATED_OUT_OF_MODEL", False),
                       ("UNMODELLED_RESPONSE", False), ("BULK_SCATTER", False)):
        line = PowerBudgetLine(source_id="S", spectral_index=0, wavelength_nm=633.0, input_power_W=1.0,
                               sinks={"ESCAPED": 1.0 - 1e-15, kind: 1e-15})
        fate = line_fate([line], cfg)
        assert (kind in fate.unknown_W) is kept, kind
        assert left_out_ledger_W(line.sinks, 1.0, cfg) == (0.0 if kept else 1e-15)
    # above the level (2e-9 of the line) nothing is left out
    line = PowerBudgetLine(source_id="S", spectral_index=0, wavelength_nm=633.0, input_power_W=1.0,
                           sinks={"ESCAPED": 1.0 - 2e-9, "TERMINATED_OUT_OF_MODEL": 2e-9})
    assert line_fate([line], cfg).unknown_W == {"TERMINATED_OUT_OF_MODEL": 2e-9}


def test_a_source_at_the_power_floor_is_not_a_known_zero():
    """ENG-10 stays distinct: a source at or below power_epsilon_W stops at the first element it meets (the power
    floor), so the power of the plane behind it is not known (ZERO_POWER, UNKNOWN), never a forced 0 W."""
    laser = B.laser("L", wavelength_nm=532.0, power_W=1e-16, waist_radius_mm=0.3, position=(0, 0, 0), direction=X)
    stop = B.aperture_stop("S", position=(20, 0, 0), axis=X, radius_mm=5.0)
    plane = B.observation_plane("d", position=(60, 0, 0), normal=X, radius_mm=10.0)
    result = run(ProjectSnapshot(components=(laser, stop), observations=(plane,), targets=(_target("t", "d", maximum=1e-9),)))
    assert result.paths[0].termination == "ZERO_POWER"
    (t,) = result.targets
    assert (t.status, t.value, t.issue_codes) == (Status.UNKNOWN, None, ("ZERO_POWER",))
    issue = _issue(result, "ZERO_POWER", "observation:d")
    assert issue.status is Status.UNKNOWN and "power floor" in issue.message
    assert not result.observations[0].power_complete


# ------------------------------------------------------------------------------------------ G12 interaction budget
R_MIRROR = 0.999


def _cavity(monitors: int, *targets: Target) -> ProjectSnapshot:
    """Flat mirrors at x = 20 (facing +x) and x = 50 (facing -x); the source at x = 30 heads +x; transparent monitors at
    x = 35, 40 inside the cavity; a plane behind M1 that only light transmitted by the mirror could reach (not followed)."""
    laser = B.laser("L", wavelength_nm=532.0, power_W=P_W, waist_radius_mm=0.2, position=(30, 0, 20), direction=X)
    m1 = B.mirror("M1", position=(50, 0, 20), normal=(-1, 0, 0), clear_radius_mm=10.0, reflectance=R_MIRROR)
    m2 = B.mirror("M2", position=(20, 0, 20), normal=X, clear_radius_mm=10.0, reflectance=R_MIRROR)
    planes = tuple(B.observation_plane(f"mon{k}", position=(35 + 5 * k, 0, 20), normal=X, radius_mm=10.0) for k in range(monitors))
    outside = B.observation_plane("outside", position=(80, 0, 20), normal=X, radius_mm=10.0)
    return ProjectSnapshot(components=(laser, m1, m2), observations=(*planes, outside), targets=targets)


@pytest.mark.parametrize("monitors", [0, 1, 2])
def test_transparent_monitors_do_not_shorten_a_cavity_trace(monitors):
    limit = NumericalConfig().max_interactions_per_path
    result = run(_cavity(monitors, _target("leak", "outside", maximum=1e-9)))
    (path,) = result.paths
    reflections = sum(1 for step in path.lineage if step.rsplit(":", 1)[-1] in REFLECTION_EVENTS)
    assert (path.termination, reflections) == ("MAX_INTERACTIONS", limit)
    # Free propagation after the final allowed reflection reaches the next mirror, but does not reflect.
    assert path.geometric_path_length_mm == pytest.approx(20.0 + limit * 30.0, abs=1e-9)
    assert path.final_power_W == pytest.approx(P_W * R_MIRROR ** limit, rel=1e-9)
    assert result.power_budget.lines[0].sinks["INTERACTION_LIMIT"] == pytest.approx(P_W * R_MIRROR ** limit, rel=1e-9)
    # Every leg, including free propagation after the last reflection, crosses each monitor once.
    for k in range(monitors):
        seen = sorted(arrivals(result, f"mon{k}"), key=lambda a: a.crossing_index)
        assert [a.crossing_index for a in seen] == list(range(limit + 1))
        assert [a.beam.power_W for a in seen] == pytest.approx([P_W * R_MIRROR ** j for j in range(limit + 1)], rel=1e-9)
    # the limit leaves the trace incomplete: UNKNOWN, never a failure or "not reached"
    assert _issue(result, "MAX_INTERACTIONS").status is Status.UNKNOWN
    (leak,) = result.targets
    assert (leak.status, leak.value, leak.issue_codes) == (Status.UNKNOWN, None, ("MAX_INTERACTIONS",))
    assert _issue(result, "OBSERVATION_NOT_REACHED", "observation:outside").status is Status.UNKNOWN
    assert result.status is Status.UNKNOWN


def test_the_cavity_trace_is_identical_with_and_without_monitors():
    base = run(_cavity(0)).paths[0]
    for monitors in (1, 2):
        path = run(_cavity(monitors)).paths[0]
        physical = [s for s in path.lineage if not s.endswith(":observe")]
        assert physical == list(base.lineage)
        assert (path.geometric_path_length_mm, path.final_power_W) == (base.geometric_path_length_mm, base.final_power_W)


# --------------------------------------------------------------------------- G14 partial capture and the acceptance cone
LAM, W0, L_DET = 650.0, 0.05, 2.0


def _pinhole(active_radius_mm: float, *targets: Target) -> tuple[ProjectSnapshot, float]:
    laser = B.laser("L", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W0, position=(0, 0, 0), direction=X)
    det = B.observation_plane("pin", position=(L_DET, 0, 0), normal=X, radius_mm=active_radius_mm, absorbing=True)
    w = W0 * math.sqrt(1.0 + (L_DET / rayleigh_range(W0, LAM)) ** 2)
    return ProjectSnapshot(components=(laser,), observations=(det,), targets=targets), w


@pytest.mark.parametrize("radius_over_w", [0.1, 0.4, 0.8])
def test_intentional_partial_capture_is_a_notice_with_the_captured_fraction(radius_over_w):
    _, w = _pinhole(1.0)
    r = radius_over_w * w
    project, _ = _pinhole(r, _target("throughput", "pin", minimum=0.0))
    result = run(project)
    fraction = 1.0 - math.exp(-2.0 * r * r / (w * w))
    arrival = one_arrival(result, "pin")
    assert arrival.active_area_power_fraction == pytest.approx(fraction, rel=1e-6)
    assert result.targets[0].status is Status.PASS
    assert result.targets[0].value == pytest.approx(P_W * fraction, rel=1e-6)
    overfill = _issue(result, "INCOMPLETE_ENVELOPE", "observation:pin")
    assert overfill.status is Status.NOT_APPLICABLE
    assert overfill.numerical_context["active_area_power_fraction"] == pytest.approx(fraction, rel=1e-6)
    # (the in-tool step only: a design change is not an action)
    assert overfill.suggested_action == "state the requirement as an active_area_power_fraction target"
    assert "MONITOR_MISSED" not in issue_codes(result)
    # (a measured fraction raises no status: the plane's and the run's stay PASS, and the target decides)
    assert result.status is Status.PASS and result.observations[0].status is Status.PASS
    assert result.power_budget.lines[0].sinks["OUTSIDE_ACTIVE_AREA"] == pytest.approx(P_W * (1.0 - fraction), rel=1e-6)


def test_a_beam_whose_1_over_e2_edge_lies_inside_the_active_area_is_not_flagged():
    _, w = _pinhole(1.0)
    project, _ = _pinhole(1.2 * w, _target("throughput", "pin", minimum=0.0))
    result = run(project)
    assert one_arrival(result, "pin").active_area_power_fraction == pytest.approx(1.0 - math.exp(-2.0 * 1.44), rel=1e-6)
    assert "INCOMPLETE_ENVELOPE" not in issue_codes(result) and result.status is Status.PASS


def test_a_capture_requirement_is_a_target_and_fails_when_violated():
    _, w = _pinhole(1.0)
    project, _ = _pinhole(0.4 * w, _target("full_capture", "pin", metric="active_area_power_fraction", unit="1", minimum=0.99),
                          _target("margin", "pin", metric="capture_margin_mm", unit="mm", minimum=0.0))
    result = run(project)
    assert [t.status for t in result.targets] == [Status.FAIL, Status.FAIL]
    assert result.targets[0].value == pytest.approx(1.0 - math.exp(-2.0 * 0.16), rel=1e-6)
    assert result.status is Status.FAIL


def _tilted_detector(tilt_deg: float, *, absorbing: bool, behind: bool = False, targets=()) -> ProjectSnapshot:
    t = math.radians(tilt_deg)
    laser = B.laser("L", wavelength_nm=633.0, power_W=P_W, waist_radius_mm=0.4, position=(0, 0, 20), direction=X)
    det = B.observation_plane("det", position=(100, 0, 20), normal=(math.cos(t), math.sin(t), 0), radius_mm=3.0,
                              acceptance_half_angle_deg=10.0, absorbing=absorbing)
    planes = (det,) + ((B.observation_plane("behind", position=(150, 0, 20), normal=X, radius_mm=10.0),) if behind else ())
    return ProjectSnapshot(components=(laser,), observations=planes, targets=targets)


def test_light_outside_the_acceptance_cone_is_not_detected():
    result = run(_tilted_detector(25.0, absorbing=True, targets=(_target("p", "det", minimum=1e-4),)))
    arrival = one_arrival(result, "det")
    assert arrival.within_acceptance is False and arrival.detected_power_W == 0.0
    assert (result.targets[0].status, result.targets[0].value) == (Status.FAIL, 0.0)
    assert result.power_budget.lines[0].sinks == {"OUTSIDE_ACCEPTANCE": pytest.approx(P_W, rel=1e-12)}
    assert result.paths[0].termination == "OUTSIDE_ACCEPTANCE"
    missed = _issue(result, "MONITOR_MISSED", "observation:det")
    assert missed.status is Status.NOT_APPLICABLE and missed.suggested_action is None
    assert result.observations[0].status is Status.PASS  # (the plane's status holds its beams' own findings only)
    inside = run(_tilted_detector(5.0, absorbing=True, targets=(_target("p", "det", minimum=1e-4),)))
    assert one_arrival(inside, "det").detected_power_W == pytest.approx(P_W, rel=1e-9)
    assert inside.status is Status.PASS


def test_a_transparent_plane_receives_nothing_outside_its_cone_and_the_beam_goes_on():
    result = run(_tilted_detector(25.0, absorbing=False, behind=True))
    assert one_arrival(result, "det").within_acceptance is False
    det = next(o for o in result.observations if o.observation_id == "det")
    assert det.total_power_W == 0.0 and det.power_by_wavelength[0].power_W == 0.0
    assert one_arrival(result, "behind").beam.power_W == pytest.approx(P_W, rel=1e-12)
