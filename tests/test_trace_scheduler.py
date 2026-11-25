"""The tracer's scheduling and trace-budget decisions (``optics/rays/scheduler.py``), on their own.

The whole-trace behaviour is covered by ``test_trace_budget.py`` and the ``trace_*`` snapshot fixtures; these tests pin
the scheduler's contract that the tracer relies on: strongest first with ties in queue order, each limit refusing exactly
at its count, and every refusal recorded once with the ids it affected.
"""

from __future__ import annotations

from os_optics.errors import Status
from os_optics.optics.rays.scheduler import MAX_PATHS, MAX_TRACE_STEPS, TraceScheduler


def test_the_queue_pops_the_strongest_first_and_ties_in_queue_order():
    schedule: TraceScheduler[str] = TraceScheduler(10, 10, 10)
    for item, power in (("a", 1.0), ("b", 3.0), ("c", 1.0), ("d", 2.0)):
        schedule.push(item, power, f"{item}#0", "S")
    popped = []
    while schedule.pending():
        item, traced = schedule.pop()
        assert traced
        popped.append(item)
    assert popped == ["b", "d", "a", "c"]
    assert schedule.exhausted == ()


def test_the_step_budget_cuts_the_path_then_refuses_the_queue():
    schedule: TraceScheduler[str] = TraceScheduler(10, 2, 10)
    schedule.push("queued", 1.0, "S#0.1", "S")
    assert schedule.take_step("S#0", "S") and schedule.take_step("S#0", "S")
    assert not schedule.take_step("S#0", "S")
    item, traced = schedule.pop()
    assert (item, traced) == ("queued", False)
    assert (schedule.steps, schedule.cut_short, schedule.untraced) == (2, 1, 1)
    assert schedule.exhausted == (MAX_TRACE_STEPS,)
    assert schedule.affected_path_ids == ("S#0", "S#0.1") and schedule.affected_source_ids == ("S",)


def test_the_path_budget_admits_roots_then_refuses_lines_and_branches():
    schedule: TraceScheduler[str] = TraceScheduler(2, 10, 10)
    assert schedule.paths_left()
    schedule.admit_root()
    assert schedule.reserve_branch("A#0", "A")
    assert not schedule.paths_left()
    schedule.refuse_line("B")
    assert not schedule.reserve_branch("A#0", "A")
    assert (schedule.paths, schedule.untraced, schedule.untraced_lines) == (2, 2, 1)
    assert schedule.exhausted == (MAX_PATHS,)
    assert schedule.affected_path_ids == ("A#0",)  # a line never launched has no path
    assert schedule.affected_source_ids == ("A", "B")
    assert schedule.keep_path(1) and not schedule.keep_path(2)


def test_a_probe_is_refused_by_either_limit_and_names_both():
    schedule: TraceScheduler[str] = TraceScheduler(1, 1, 10)
    assert schedule.admit_probe("S#0", "S")
    assert schedule.take_step("S#0", "S")
    assert not schedule.admit_probe("S#0", "S")
    assert schedule.exhausted == (MAX_PATHS, MAX_TRACE_STEPS)
    assert schedule.untraced == 0  # a probe pass counts the branches it does not follow itself


def test_the_report_names_what_the_budget_left_untraced():
    schedule: TraceScheduler[str] = TraceScheduler(1, 10, 10)
    assert schedule.report(0, 0.0, 1.0) is None
    schedule.admit_root()
    schedule.refuse_line("B")
    assert not schedule.reserve_branch("A#0", "A")
    issue = schedule.report(1, 0.25, 1.0)
    assert issue is not None and issue.status is Status.WARNING and issue.code.value == "MAX_PATHS"
    assert "1 beam branch not traced, 1 source spectral line not launched" in issue.message
    ctx = issue.numerical_context
    assert ctx["budget"] == [MAX_PATHS] and ctx["untraced_power_fraction"] == 0.25 and ctx["traced_paths"] == 1


def test_the_interaction_limit():
    schedule: TraceScheduler[str] = TraceScheduler(10, 10, 3)
    assert not schedule.interaction_spent(2) and schedule.interaction_spent(3)
    issue = schedule.interaction_issue("S#0", "S", 1e-3, 12.5)
    assert issue.status is Status.UNKNOWN and issue.location == "path:S#0"
    assert issue.numerical_context["budget"] == ["max_interactions_per_path"] and issue.numerical_context["limit"] == 3
