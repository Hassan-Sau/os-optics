"""The run's trace budget: ``max_paths`` beam paths and ``max_trace_steps`` chief-ray steps per run.

A branching design (partially reflecting surfaces facing each other, many sources x spectral lines) is traced up to the
budget and the rest is left untraced with exactly one ``MAX_PATHS`` issue that names the budget and gives the count and
the power of what was not traced (the ledger's ``PATH_LIMIT``). Below the budget nothing changes.
"""

from __future__ import annotations

import pytest

from os_optics import NumericalConfig
from os_optics.builders import laser, npbs, observation_plane
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status

from helpers import run

X = (1.0, 0.0, 0.0)
MX = (-1.0, 0.0, 0.0)
SOURCE_W = 1e-3


def _plates(count: int, *, reflectance: float = 0.5, lines: int = 1):
    """``count`` partial reflectors at normal incidence in series (alternate ones facing back): every pair is an etalon."""
    spectrum = tuple((780.0 + 0.5 * k, 1.0 / lines) for k in range(lines)) if lines > 1 else ()
    src = laser("SRC", wavelength_nm=None if lines > 1 else 780.0, spectrum=spectrum, power_W=SOURCE_W, waist_radius_mm=0.5,
                position=(-100.0, 0.0, 0.0), direction=X)
    plates = tuple(npbs(f"P{k}", reflectance=reflectance, position=(20.0 * k, 0.0, 0.0), normal=MX if k % 2 == 0 else X)
                   for k in range(count))
    det = observation_plane("det", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    return ProjectSnapshot(components=(src, *plates), observations=(det,))


def _budget_issues(result):
    return [i for i in result.issues if i.code.value == "MAX_PATHS"]


def test_the_default_budget_is_documented_and_bounded():
    cfg = NumericalConfig()
    assert (cfg.max_paths, cfg.max_trace_steps) == (512, 2048)


def test_three_partial_reflectors_stop_at_the_budget_with_one_issue():
    result = run(_plates(3))
    cfg = NumericalConfig()
    (issue,) = _budget_issues(result)
    assert issue.status is Status.WARNING
    ctx = issue.numerical_context
    assert ctx["max_paths"] == cfg.max_paths and ctx["max_trace_steps"] == cfg.max_trace_steps
    assert ctx["untraced_branches"] > 0 and ctx["untraced_source_lines"] == 0
    untraced = result.power_budget.total_sinks["PATH_LIMIT"]
    assert ctx["untraced_power_W"] == pytest.approx(untraced, rel=1e-12)
    assert ctx["untraced_power_fraction"] == pytest.approx(untraced / SOURCE_W, rel=1e-12)
    assert 0.0 < untraced < 0.05 * SOURCE_W  # the strongest branches are traced first
    assert len(result.paths) <= cfg.max_paths
    assert sum(len(p.segments) for p in result.paths) <= cfg.max_trace_steps
    assert result.power_budget.conserved
    assert "beam branches not traced" in issue.message and "% of the launched power" in issue.message


def test_the_step_budget_cuts_a_path_short_and_skips_the_queue():
    cfg = NumericalConfig(max_trace_steps=12)
    result = run(_plates(3), numerical=cfg)
    (issue,) = _budget_issues(result)
    assert issue.numerical_context["budget"] == ["max_trace_steps"]
    assert "1 beam path cut short" in issue.message
    assert issue.numerical_context["trace_steps"] == 12
    assert issue.numerical_context["phase"] == "primary"
    cut = {p.path_id for p in result.paths if p.termination == "TRACE_BUDGET"}
    assert cut <= set(issue.numerical_context["affected_path_ids"])
    assert issue.numerical_context["affected_source_ids"] == ["SRC"]
    assert sum(len(p.segments) for p in result.paths) <= 12
    assert "TRACE_BUDGET" in {p.termination for p in result.paths}
    assert result.power_budget.conserved


def test_source_lines_beyond_the_path_budget_are_not_traced_and_counted():
    cfg = NumericalConfig(max_paths=4)
    result = run(_plates(1, lines=10), numerical=cfg)
    (issue,) = _budget_issues(result)
    assert len(result.paths) == 4
    assert issue.numerical_context["untraced_source_lines"] == 6  # never launched
    assert issue.numerical_context["untraced_branches"] == 4  # the 4 traced lines' reflections
    assert "4 beam branches not traced, 6 source spectral lines not launched" in issue.message
    assert result.power_budget.total_sinks["PATH_LIMIT"] == pytest.approx(issue.numerical_context["untraced_power_W"], rel=1e-12)
    assert result.power_budget.conserved


def test_a_design_inside_the_budget_is_unchanged():
    """The two-plate etalon: every branch down to the power floor is traced, no budget issue."""
    small = run(_plates(2))
    large = run(_plates(2), numerical=NumericalConfig(max_paths=4096, max_trace_steps=1 << 20))
    assert _budget_issues(small) == []
    assert small.paths == large.paths and small.observations == large.observations and small.power_budget == large.power_budget


def test_the_beam_pair_cap_has_its_own_code():
    result = run(_plates(3))
    codes = [i.code.value for i in result.issues]
    assert codes.count("MAX_PATHS") == 1
    assert codes.count("MAX_BEAM_PAIRS") <= len(result.observations)


def test_the_strongest_queued_branches_are_traced_first():
    """When the step budget runs out, every branch left in the queue is no stronger than any branch that was traced: the
    tracer always traces the strongest queued packet next."""

    result = run(_plates(4, reflectance=0.3), numerical=NumericalConfig(max_trace_steps=200))
    assert _budget_issues(result)
    flows = [e for e in result.power_budget.entries if e.kind == "PATH_LIMIT" and e.detail == "branch not traced (max_trace_steps)"]
    traced_branches = [p.segments[0].power_start_W for p in result.paths if p.parent_path_id is not None and p.segments]
    assert flows and traced_branches
    assert max(e.power_W for e in flows) <= min(traced_branches) * (1.0 + 1e-12)


def test_the_path_budget_launches_the_strongest_source_lines():
    """A source with lines of unequal power: the weakest lines are the ones not launched (ties keep the source order)."""

    weights = (0.05, 0.3, 0.1, 0.25, 0.3)
    src = laser("SRC", spectrum=tuple((780.0 + k, w) for k, w in enumerate(weights)), power_W=SOURCE_W, waist_radius_mm=0.5,
                position=(-100.0, 0.0, 0.0), direction=X)
    det = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(src,), observations=(det,)), numerical=NumericalConfig(max_paths=3))
    assert sorted(p.spectral_index for p in result.paths) == [1, 3, 4]
    (issue,) = _budget_issues(result)
    assert issue.numerical_context["untraced_source_lines"] == 2
    assert issue.numerical_context["affected_source_ids"] == ["SRC"]
    assert issue.numerical_context["affected_path_ids"] == []  # these source lines were never launched
    assert issue.numerical_context["untraced_power_W"] == pytest.approx(0.15 * SOURCE_W, rel=1e-12)


def _two_sources(*targets: Target):
    a = laser("A", wavelength_nm=780.0, power_W=SOURCE_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)
    b = laser("B", wavelength_nm=780.0, power_W=0.5 * SOURCE_W, waist_radius_mm=0.5, position=(-100.0, 3.0, 0.0), direction=X)
    det = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    return ProjectSnapshot(components=(a, b), observations=(det,), targets=targets)


def test_a_target_on_an_untraced_source_is_unknown_not_unreached():
    """Source B is not launched (the path budget is 1). Its target is UNKNOWN with MAX_PATHS, not 'no beam
    reaches the plane'; a power summed over every beam is bounded by the traced beam (1 mW) and the traced beam plus B's
    untraced 0.5 mW on this absorbing plane, so a 1 W maximum is met (PASS, MAX_PATHS named); a target on the traced source
    A is evaluated as usual."""

    targets = (Target(target_id="b", metric="power_W", observation_id="det", source_id="B", minimum=1e-6, unit="W"),
               Target(target_id="total", metric="power_W", observation_id="det", maximum=1.0, unit="W"),
               Target(target_id="a", metric="power_W", observation_id="det", source_id="A", minimum=1e-6, unit="W"),
               Target(target_id="b_radius", metric="beam_radius_mm", observation_id="det", source_id="B", maximum=1.0, unit="mm"))
    result = run(_two_sources(*targets), numerical=NumericalConfig(max_paths=1))
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["b"].status is Status.UNKNOWN and by_id["b"].issue_codes == ("MAX_PATHS",)
    assert "trace budget" in by_id["b"].message and "OBSERVATION_NOT_REACHED" not in by_id["b"].issue_codes
    assert by_id["b_radius"].status is Status.UNKNOWN and by_id["b_radius"].issue_codes == ("MAX_PATHS",)
    assert by_id["total"].status is Status.PASS and by_id["total"].issue_codes == ("MAX_PATHS",)
    assert by_id["total"].value == pytest.approx(SOURCE_W, rel=1e-12)
    assert by_id["a"].status is Status.PASS and by_id["a"].issue_codes == ()
    full = run(_two_sources(*targets))
    assert {t.target_id: t.status for t in full.targets} == {"b": Status.PASS, "total": Status.PASS, "a": Status.PASS, "b_radius": Status.PASS}


def test_a_target_whose_source_is_traced_but_misses_the_plane_is_a_known_zero_with_its_truncation():
    """Source B is traced completely and misses the plane, so no light of it arrives: a known 0 W, and the minimum
    fails on it. The light left out as negligible is the stated truncation of that value: the entry holds its bound, which
    does not depend on whether the target states a minimum or a maximum: the run follows the untraced light
    for a power metric whatever bounds the target sets."""
    targets = (Target(target_id="b", metric="power_W", observation_id="det", source_id="B", minimum=1e-6, unit="W"),)
    snapshot = _two_sources(*targets)
    off = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=1.0, absorbing=True)
    result = run(snapshot.model_copy(update={"observations": (off,)}))
    (target,) = result.targets
    assert target.status is Status.FAIL and target.value == 0.0
    assert target.issue_codes == ("NO_TRACED_ARRIVAL", "FAINT_LIGHT_BOUNDED", "TARGET_NOT_MET")
    (omitted,) = target.value_basis["value"].omitted
    assert target.value_basis["value"].kind.value == "MODEL" and omitted.part.value == "BELOW_NEGLIGIBLE_LEVEL"
    assert 0.0 < omitted.upper < 1e-6  # (far below the minimum the target fails on)
