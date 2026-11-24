"""``ValidationIssue.affected_targets``: the targets an issue leaves undecided.

A target is named by an issue when the target's result is ``UNKNOWN`` or ``OUT_OF_MODEL_SCOPE``, its ``issue_codes`` carry
the reason the issue stands for, and the issue concerns the target's light:

* the run's trace budget (``MAX_PATHS``, no location): the target's source lines lost power to the budget (the ledger's
  ``PATH_LIMIT``);
* the budget for following untraced light (``MAX_PATHS`` at ``untraced_light``): that light may reach the target's plane
  from the target's source lines and was not followed;
* an issue on one beam path (``path:<id>``, or a ``path_id`` in its context: the per-path interaction limit, a path ended
  out of model scope, a Gaussian state out of scope): the path belongs to one of the target's source lines and either
  its unknown-fate power is why the target is undecided (the cause code of that ledger sink), or the target reads that
  path or a beam branched from it;
* an issue on an observation plane (``observation:<id>``): a target on that plane (the issue's code, or the causes it
  names in ``untraced_causes``);
* an issue on a target (``target:<id>``): that target;
* an issue on a surface without a path: a target that reads a beam whose lineage crosses that surface;
* an exposure upper bound withheld on a surface (``EXPOSURE_BOUND_WITHHELD``): a ``power_rating_utilization`` target of its
  element;
* an issue on a source (``source:<id>``): a target of that source (or of every source).

Targets are listed in Project order. Targets decided on a bound (``PASS`` / ``FAIL`` with the cause codes) are not
named: the issue did not leave them undecided.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import NumericalConfig
from ..contracts.project import Target
from ..contracts.results import ObservationResult, PowerBudget, TargetResult
from ..errors import ErrorCode, Status, ValidationIssue, make_issue
from ..math.vectors import correctly_rounded_sum
from ..optics.rays.tracer import SINK_PATH_LIMIT, UNKNOWN_FATE_SINKS
from .exposures import POWER_RATING_UTILIZATION
from .metrics import _scope_arrivals, _target_lines
from .power import UNKNOWN_FATE_CAUSES, bounded_cause
from .untraced import PROBE_BUDGET, PlaneReach, hazards

UNDECIDED = (Status.UNKNOWN, Status.OUT_OF_MODEL_SCOPE)
UNTRACED_LIGHT_LOCATION = "untraced_light"


def _descends(path_id: str, root: str) -> bool:
    return path_id == root or path_id.startswith(root + ".")


def name_affected_targets(issues: Sequence[ValidationIssue], targets: Sequence[Target], results: Sequence[TargetResult],
                          observations: dict[str, ObservationResult], power_budget: PowerBudget | None,
                          reach: PlaneReach | None, cfg: NumericalConfig) -> list[ValidationIssue]:
    """``issues`` with ``affected_targets`` filled (module docstring)."""
    # (a target that is not assessable is undecided by its own input, never by an issue of the run)
    undecided = [(t, r) for t, r in zip(targets, results) if r.status in UNDECIDED and t.not_assessable is None]
    if not undecided:
        return list(issues)
    entries = power_budget.entries if power_budget is not None else ()
    path_line = {e.path_id: (e.source_id, e.spectral_index) for e in entries if e.path_id is not None}
    unknown_on_path: dict[str, set[str]] = {}  # path -> cause codes of its unknown-fate ledger power
    budget_lines: set[tuple[str, int]] = set()  # lines that lost power to the run's trace budget
    for e in entries:
        if e.kind in UNKNOWN_FATE_SINKS and e.path_id is not None:
            unknown_on_path.setdefault(e.path_id, set()).add(UNKNOWN_FATE_CAUSES[e.kind].value)
        if e.kind == SINK_PATH_LIMIT and e.power_W > 0.0:
            budget_lines.add((e.source_id, e.spectral_index))
    lines_of = {t.target_id: {(line.source_id, line.spectral_index) for line in _target_lines(t, power_budget, cfg)}
                for t, _ in undecided}

    def reads(t: Target, check) -> bool:
        obs = observations.get(t.observation_id or "")
        return any(check(a) for a in _scope_arrivals(t, obs, cfg))

    def named(issue: ValidationIssue, t: Target, r: TargetResult) -> bool:
        codes = set(r.issue_codes)
        location = issue.location or ""
        context = issue.numerical_context
        if issue.code is ErrorCode.EXPOSURE_BOUND_WITHHELD:
            # (a surface's notice names its element; the run's notice of a closed return, every element it leaves unbounded)
            elements = {context.get("element_id"), *(context.get("element_ids") or ())}  # type: ignore[misc]
            return t.metric == POWER_RATING_UTILIZATION and t.device_id in elements
        if issue.code is ErrorCode.MAX_PATHS and not location:
            return "MAX_PATHS" in codes and bool(lines_of[t.target_id] & budget_lines)
        if issue.code is ErrorCode.MAX_PATHS and location == UNTRACED_LIGHT_LOCATION:
            per_line = (reach or {}).get(t.observation_id or "", {})
            return "MAX_PATHS" in codes and any(PROBE_BUDGET in per_line.get(line, {}) for line in lines_of[t.target_id])
        if location.startswith("target:"):
            return t.target_id == location.removeprefix("target:")
        if location.startswith("observation:"):
            causes = {issue.code.value} | set(context.get("untraced_causes") or [])  # type: ignore[arg-type]
            return t.observation_id == location.removeprefix("observation:") and bool(codes & causes)
        path = location.removeprefix("path:") if location.startswith("path:") else context.get("path_id")
        if isinstance(path, str):
            if path_line.get(path) not in lines_of[t.target_id]:
                return False
            if codes & unknown_on_path.get(path, set()):
                return True
            return issue.code.value in codes and reads(t, lambda a: _descends(a.beam.path_id, path))
        if issue.code.value not in codes:
            return False
        if location.startswith("surface:"):
            surface = location.removeprefix("surface:")
            return reads(t, lambda a: any(step.rsplit(":", 1)[0] == surface for step in a.lineage))
        if location.startswith("source:"):
            source = location.removeprefix("source:")
            return any(line[0] == source for line in lines_of[t.target_id])
        return not location

    out = []
    for issue in issues:
        affected = tuple(t.target_id for t, r in undecided if named(issue, t, r))
        out.append(issue.model_copy(update={"affected_targets": affected}) if affected else issue)
    return out


def lit_reach_issues(targets: Sequence[Target], results: Sequence[TargetResult], observations: dict[str, ObservationResult],
                     power_budget: PowerBudget | None, reach: PlaneReach | None, cfg: NumericalConfig) -> list[ValidationIssue]:
    """One ``INCOMPLETE_TRACE`` issue (``UNKNOWN``) per power target a limit leaves undecided on a plane a traced beam reaches,
    where light the trace does not follow reaches it too (``validation/untraced.py``, a run that followed every untraced
    branch): the plane's power lies between the traced light and the traced light plus that light. It sits at the target
    (``target:<id>``, the plane in ``observation_id``), since it exists because of the limit: a plane's own issues do not
    depend on what a target asks. It carries the causes (``untraced_causes``)."""
    if not reach:
        return []
    out: list[ValidationIssue] = []
    for oid in sorted(reach):
        obs = observations.get(oid)
        if obs is None or not obs.lit:
            continue  # (an unlit plane is graded by grade_unlit_planes)
        lit = {(a.beam.source_id, a.beam.spectral_index) for a in obs.arrivals} | {
            (line.source_id, line.spectral_index) for line in obs.sampled_light if line.counted_crossing_count}
        kinds = hazards(reach, oid, [line for line in reach[oid] if line in lit])
        if not kinds:
            continue
        causes = tuple(dict.fromkeys(bounded_cause(kind) for kind in kinds))
        undecided = [t for t, r in zip(targets, results) if t.observation_id == oid and r.status is Status.UNKNOWN
                     and set(r.issue_codes) & set(causes)]
        if not undecided:
            continue
        total = correctly_rounded_sum(kinds.values())
        out.extend(make_issue(
            ErrorCode.INCOMPLETE_TRACE, Status.UNKNOWN,
            f"light the trace does not follow reaches this plane beside the traced beams of the same source lines (up to "
            f"{total:.4g} W: {', '.join(sorted(kinds))}): a power requirement whose limit lies between the traced power and "
            "the traced power plus that light is not decided", location=f"target:{t.target_id}", observation_id=oid, untraced_power_W=total,
            untraced_causes=list(causes), untraced_light=sorted(kinds),
            suggested_action="trace the reflection or transmission involved") for t in undecided)
    return out
