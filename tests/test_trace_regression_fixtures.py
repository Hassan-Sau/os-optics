"""The trace regression fixtures still exercise what they were recorded for.

``fixture_scenes.py``'s ``trace_*`` scenes are snapshot baselines of the tracer's budget decisions and of its partial-beam
and return-light handling (``test_fixtures_cli_freeze.py`` compares a fresh run with each stored snapshot). A snapshot
only guards a decision the scene actually reaches, so each stored snapshot is checked here for its condition: a change
of scene, defaults or physics that silently stops exercising it fails this test, not just the snapshot comparison.
"""

from __future__ import annotations

import json

from generate_fixtures import SNAP_DIR


def _snapshot(name: str) -> dict:
    return json.loads((SNAP_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _issues(snap: dict, code: str) -> list[dict]:
    return [i for i in snap["issues"] if i["code"] == code]


def _flows(snap: dict, kind: str) -> list[dict]:
    return [e for e in snap["power_budget"]["entries"] if e["kind"] == kind]


def _observation(snap: dict, observation_id: str) -> dict:
    return next(o for o in snap["observations"] if o["observation_id"] == observation_id)


def test_path_budget_refuses_source_lines_and_branches():
    snap = _snapshot("trace_path_budget")
    budgets = snap["outcome"]["execution"]["budgets"]
    assert budgets["exhausted"] == ["max_paths"]
    assert budgets["exhausted_by_phase"] == {"primary": ["max_paths"]}
    (issue,) = _issues(snap, "MAX_PATHS")
    ctx = issue["numerical_context"]
    assert (ctx["phase"], ctx["untraced_source_lines"], ctx["untraced_branches"]) == ("primary", 6, 4)
    details = [e["detail"] for e in _flows(snap, "PATH_LIMIT")]
    assert details.count("source line not traced (max_paths)") == 6  # never launched (``Tracer.run``)
    assert details.count("beam_splitter_reflection") == 4  # branches the path budget refused
    assert len(snap["paths"]) == 4


def test_step_budget_cuts_a_path_between_surfaces_and_skips_the_queue():
    snap = _snapshot("trace_step_budget")
    assert snap["outcome"]["execution"]["budgets"]["exhausted"] == ["max_trace_steps"]
    (issue,) = _issues(snap, "MAX_PATHS")
    ctx = issue["numerical_context"]
    assert (ctx["trace_steps"], ctx["paths_cut_short"], ctx["untraced_branches"]) == (13, 1, 2)
    (cut,) = [p for p in snap["paths"] if p["termination"] == "TRACE_BUDGET"]
    # stopped after the monitor it crossed last: the path ends at that plane, not at a physical surface
    assert cut["segments"][-1]["end_surface_id"] == "observation/m01"
    assert cut["termination_surface_id"] is None
    details = [e["detail"] for e in _flows(snap, "PATH_LIMIT")]
    assert details.count("TRACE_BUDGET") == 1
    assert details.count("branch not traced (max_trace_steps)") == 2


def test_interaction_limit_ends_a_returning_branch():
    snap = _snapshot("trace_interaction_limit")
    budgets = snap["outcome"]["execution"]["budgets"]
    assert budgets["exhausted"] == ["max_interactions_per_path"]
    (limited,) = [p for p in snap["paths"] if p["termination"] == "MAX_INTERACTIONS"]
    assert limited["parent_path_id"] is not None  # a branch: it counts its parent's interactions
    assert _flows(snap, "INTERACTION_LIMIT")
    (issue,) = _issues(snap, "MAX_INTERACTIONS")
    assert issue["numerical_context"]["affected_path_ids"] == [limited["path_id"]]


def test_partial_beams_at_observation_planes():
    snap = _snapshot("trace_partial_beams")
    assert snap["outcome"]["execution"]["budgets"]["exhausted"] == []
    (small,) = _observation(snap, "S")["arrivals"]
    assert 0.0 < small["active_area_power_fraction"] < 1.0
    behind = _observation(snap, "T")
    # no traced beam reaches T; only the part of L#0 crossing beside its chief ray may, under a supported bound: the stated
    # interval [0, power_upper_bound_W] with no value (BOUNDED from 0)
    assert behind["arrivals"] == [] and behind["quantity_state"] == "BOUNDED" and behind["total_power_W"] is None
    assert behind["power_lower_bound_W"] == 0.0 and 0.0 < behind["power_upper_bound_W"] < 1e-3
    assert [(f["kind"], f["path_id"]) for f in behind["followed_light"]] == [("PARTIAL_CROSSING", "L#0")]
    (issue,) = [i for i in _issues(snap, "OBSERVATION_NOT_REACHED") if i["location"] == "observation:T"]
    assert issue["numerical_context"]["untraced_light"] == ["PARTIAL_CROSSING"]


def test_return_chain_is_traced_and_crosses_planes_twice():
    snap = _snapshot("trace_return_chain")
    assert snap["outcome"]["execution"]["budgets"]["exhausted"] == []
    returns = [p for p in snap["paths"] if sum(event.endswith(":split_reflect") for event in p["lineage"]) >= 2]
    assert returns  # branches of branches: light followed through more than one round trip
    assert sum(p["parent_path_id"] is not None for p in snap["paths"]) >= 4
    inside = _observation(snap, "M")["arrivals"]
    assert {a["crossing_index"] for a in inside} == {0, 1}
    assert all(0.0 < a["active_area_power_fraction"] < 1.0 for a in inside)
    outside = _observation(snap, "B")["arrivals"]
    assert any(a["parent_path_id"] is not None for a in outside)  # returned light leaves through the coupler


def test_probe_budget_limits_the_untraced_light_pass_only():
    snap = _snapshot("trace_probe_budget")
    budgets = snap["outcome"]["execution"]["budgets"]
    assert budgets["exhausted_by_phase"] == {"untraced_light": ["max_paths", "max_trace_steps"]}
    (issue,) = _issues(snap, "MAX_PATHS")
    ctx = issue["numerical_context"]
    assert ctx["phase"] == "untraced_light" and ctx["untraced_branches"] > 0
    assert ctx["affected_path_ids"] and ctx["affected_source_ids"] == ["L"]
    assert not _flows(snap, "PATH_LIMIT")  # the trace itself fits its budget


def test_fresnel_branch_budget_refuses_a_window_reflection():
    """The window's traced Fresnel reflections reach ``Tracer._spawn`` after the path budget is held by the splitters'
    branches: refused, their power booked to ``PATH_LIMIT`` at the window's faces."""
    snap = _snapshot("trace_fresnel_branch_budget")
    budgets = snap["outcome"]["execution"]["budgets"]
    assert budgets["exhausted_by_phase"] == {"primary": ["max_paths"]}
    refused = [e for e in _flows(snap, "PATH_LIMIT") if e["detail"] == "fresnel_reflection"]
    assert len(refused) == 2 and all(0.0 < e["power_W"] < 1e-4 for e in refused)
    assert {e["element_id"] for e in refused} == {"WB"}
    assert len(snap["paths"]) == 3
    assert not any(":fresnel_reflection" in event for p in snap["paths"] for event in p["lineage"])
