"""The plan of an evaluation: which capabilities run for a set of requested ones.

``plan_evaluation`` is a pure function of the requested capability ids, the exclusions, the selected targets and the
request. It expands the presets, adds every capability the requested ones require (``capabilities.CAPABILITIES``: also what one requires only of a request that traces or
that states a power rating, and in a Project evaluation the model-scope findings that come with the plane readings and the
requirements) and, when the requirements are requested, the capabilities each selected target's metric reads
(``validation.metrics.METRIC_REQUIRES``), and returns the executed scope in execution order with what was added and why.
An id the table does not hold, a capability the request's configuration does not enable, a target the Project does not
declare, a capability that is both required and excluded, and a selection of targets that leaves a declared target out of
a capability made of a complete evaluation (the record) are refused with an ``EvaluationPlanError``: nothing excluded is
ever run, and nothing is run in its place.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .capabilities import (
    ANALYSIS_BY_ID,
    CAPABILITIES,
    CAPABILITY_BY_ID,
    COMPLETE,
    CONTROL,
    ENGINE,
    ENGINE_PRESET_MEMBERS,
    LAYOUT,
    EXPORT_BY_ID,
    FLAG_CONTROL,
    FLAG_LAYOUT,
    FLAG_PHYSICS,
    LOCAL_PRESET_MEMBERS,
    PRESETS,
    REQUIREMENTS,
    level_capabilities,
)

UNKNOWN_CAPABILITY = "UNKNOWN_CAPABILITY"
EXCLUDED_CAPABILITY_REQUIRED = "EXCLUDED_CAPABILITY_REQUIRED"
CAPABILITY_NOT_ENABLED = "CAPABILITY_NOT_ENABLED"
UNKNOWN_TARGET = "UNKNOWN_TARGET"
TARGET_SELECTION_INCOMPLETE = "TARGET_SELECTION_INCOMPLETE"
#: Who asked for a capability the caller did not name: the targets selected with ``targets`` ask for the requirements.
TARGET_SELECTION = "targets"
TARGET_PREFIX = "target:"


class EvaluationPlanError(ValueError):
    """An evaluation that cannot be planned as asked. ``code``: ``UNKNOWN_CAPABILITY`` (an id the capability table does not
    hold at this level), ``CAPABILITY_NOT_ENABLED`` (the request's configuration or content does not enable a requested
    capability), ``UNKNOWN_TARGET`` (a selected target the Project does not declare), ``EXCLUDED_CAPABILITY_REQUIRED`` (a
    capability is excluded and something requested requires it) or ``TARGET_SELECTION_INCOMPLETE`` (the selection of
    targets leaves a declared target out while a requested capability is made of a complete evaluation). ``context``: the
    ids involved."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code, self.message, self.context = code, message, context


@dataclass(frozen=True)
class AddedCapability:
    """A capability the plan runs although the caller did not name it. ``needed_by``: the capability that requires it,
    ``target:<id>`` for a target whose metric reads it, or ``targets`` (the requirements, for the targets selected)."""

    capability: str
    needed_by: str


@dataclass(frozen=True)
class TargetRequirement:
    """One declared target the plan evaluates: the capabilities its metric reads (``requires``) and those of them the
    request's configuration or content does not enable (``missing``: the target is assessed without them and reads as in
    ``validate``)."""

    target_id: str
    metric: str
    requires: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationPlan:
    """What an evaluation runs.

    ``requested``: the ids as the caller gave them (presets by name), once each, presets first and then in execution
    order. ``executed``: the capabilities that run, in execution order. ``added``: the executed capabilities the caller
    did not name, each with what requires it. ``complete``: every capability the request enables is executed and every
    declared target is selected. ``excluded``: the caller's exclusions, as the caller wrote them (presets by name).
    ``excluded_capabilities``: the capabilities those exclusions stand for, presets expanded, in execution order: every
    statement of why a capability did not run reads it (``EXCLUDED``). ``complete_scope``: the capabilities the request
    enables at this level. ``targets``: the selected targets with what each requires (empty when the requirements are not
    executed); ``unselected_targets``: the declared targets the plan does not evaluate.
    """

    requested: tuple[str, ...] = ()
    executed: tuple[str, ...] = ()
    added: tuple[AddedCapability, ...] = ()
    complete: bool = False
    excluded: tuple[str, ...] = ()
    complete_scope: tuple[str, ...] = ()
    targets: tuple[TargetRequirement, ...] = ()
    unselected_targets: tuple[str, ...] = ()
    excluded_capabilities: tuple[str, ...] = ()

    def runs(self, capability_id: str) -> bool:
        """Whether the plan executes the capability."""
        return capability_id in self.executed

    @property
    def not_executed(self) -> tuple[str, ...]:
        """The capabilities of the request's complete scope the plan does not run, in execution order."""
        return tuple(c for c in self.complete_scope if c not in self.executed)


def _ids(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _ordered(ids: Iterable[str]) -> tuple[str, ...]:
    wanted = set(ids)
    return tuple(p for p in PRESETS if p in wanted) + tuple(row.id for row in CAPABILITIES if row.id in wanted)


def enabled_capabilities(request: Any = None, level: str = ENGINE) -> tuple[str, ...]:
    """The capabilities ``request`` enables at ``level``, in execution order: its complete scope. The physics capabilities
    need ``run_physics``; the layout checks ``run_layout`` and a component with an envelope; the control analysis
    ``run_control`` and a control specification. Without a request every capability of the level. A Project evaluation
    (``local``) never runs the layout checks or the control analysis (``capabilities.level_capabilities``)."""
    rows = level_capabilities(level)
    if request is None:
        return tuple(row.id for row in rows)
    return tuple(row.id for row in rows if not disabled_reason(row.id, request))


def disabled_reason(capability_id: str, request: Any) -> str:
    """Why ``request`` does not enable the capability (empty: it does)."""
    row = CAPABILITY_BY_ID[capability_id]
    configuration, project = request.configuration, request.project
    if row.request_flag == FLAG_PHYSICS and not configuration.run_physics:
        return "the request's configuration does not trace the design (run_physics is false)"
    if row.request_flag == FLAG_LAYOUT:
        if not configuration.run_layout:
            return "the request's configuration does not run the layout checks (run_layout is false)"
        if not any(c.envelope is not None for c in project.components):
            return "no component of the Project states an envelope, so the layout checks have nothing to check"
    if row.request_flag == FLAG_CONTROL:
        if not configuration.run_control:
            return "the request's configuration does not run the control analysis (run_control is false)"
        if project.control_spec is None:
            return "the Project states no control specification"
    return ""


def _requires(capability_id: str, request: Any, scope: tuple[str, ...], level: str) -> tuple[str, ...]:
    row = CAPABILITY_BY_ID[capability_id]
    needs = list(row.requires)
    if level != ENGINE:
        needs.extend(row.local_requires)
    if row.traced_requires and (request is None or request.configuration.run_physics):
        needs.extend(row.traced_requires)
    if row.rated_requires and (request is None or any(c.power_ratings for c in request.project.components)):
        needs.extend(row.rated_requires)
    if row.complete_scope:
        needs.extend(c for c in scope if CAPABILITY_BY_ID[c].level == ENGINE)
    return tuple(dict.fromkeys(needs))


#: What a Project evaluation answers for the Engine capabilities of an Engine request only (``ENGINE_REQUEST_ONLY``).
_ENGINE_REQUEST_ONLY_TEXT = {
    LAYOUT: "a Project's request runs no layout checks. Evaluate an Engine request whose configuration sets run_layout and "
              "whose components state envelopes: os_optics.evaluate(request, capabilities=['layout'])",
    CONTROL: "a Project's request runs no control analysis. Evaluate an Engine request whose configuration sets run_control "
             "and whose Project states a control specification: os_optics.evaluate(request, capabilities=['control'])",
}


def _analysis_text(name: str) -> str:
    """How an analysis is run, for a caller who named it as a capability."""
    row = ANALYSIS_BY_ID[name]
    results = ""
    if "RunResults" in row.entry_point:  # (an analysis of its own Project or request has no RunResults)
        computed = f" of an evaluation that computed {', '.join(row.requires)}," if row.requires else ""
        results = f" (RunResults: Evaluation.results(){computed} or os_optics.results.load_run(RUN))"
    shell = f"; from a shell: {row.command}" if row.command else ""
    text = f"{name!r} is an analysis, not a capability of the evaluation plan: run it in Python with {row.entry_point}{results}{shell}"
    if "RUN" in results or "RUN" in row.command:
        text += " (RUN: a run directory that 'os-optics project run PROJECT --out-dir RUN' writes)"
    return text


def _check_ids(ids: tuple[str, ...], level: str, what: str) -> None:
    for name in ids:
        if name in PRESETS:
            continue
        row = CAPABILITY_BY_ID.get(name)
        if row is not None and row in level_capabilities(level):
            continue
        if row is not None and level != ENGINE:
            detail = f"{name!r} is a capability of an Engine request, not of a Project evaluation: {_ENGINE_REQUEST_ONLY_TEXT[name]}"
        elif row is not None:
            detail = (f"{name!r} is a capability of the local workflow (os_optics.local), not of an Engine evaluation; the Engine "
                      f"capabilities are {', '.join(c.id for c in level_capabilities(ENGINE))}")
        elif name in ANALYSIS_BY_ID:
            detail = _analysis_text(name)
        elif name in EXPORT_BY_ID:
            detail = (f"{name!r} is the run directory, not an output of an evaluation (an evaluation writes nothing): write the "
                      "run with Evaluation.write_run(output_dir), run_optical_project(project, output_dir=...) or "
                      "'os-optics project run PROJECT --out-dir DIR'")
        else:
            known = ", ".join((*PRESETS, *(c.id for c in level_capabilities(level))))
            detail = f"{name!r} is not a capability id or a preset; the ids are {known}"
        raise EvaluationPlanError(UNKNOWN_CAPABILITY, f"{what}: {detail}", capability=name)


def _expand(ids: tuple[str, ...], scope: tuple[str, ...], level: str) -> tuple[list[str], dict[str, str]]:
    """``ids`` with each preset replaced by its members, once each, and which preset brought each capability the caller
    did not also name by its id (the first preset that includes it)."""
    out: list[str] = []
    via: dict[str, str] = {}
    for name in ids:
        if name not in PRESETS:
            out.append(name)
            continue
        members = [c for c in scope if CAPABILITY_BY_ID[c].level == ENGINE] if name == COMPLETE else []
        members.extend(ENGINE_PRESET_MEMBERS[name])
        if level != ENGINE:
            members.extend(LOCAL_PRESET_MEMBERS[name])
        for member in members:
            out.append(member)
            via.setdefault(member, name)
    for name in ids:
        via.pop(name, None)  # (named by its id as well: the caller wrote it)
    return list(dict.fromkeys(out)), via


def plan_evaluation(requested: Iterable[str] | str, *, exclude: Iterable[str] | str = (), targets: Iterable[str] | None = None,
                    request: Any = None, level: str = ENGINE, what: str = "requested") -> EvaluationPlan:
    """The plan of an evaluation of ``request`` (a ``ValidationRequest``; ``None``: the table alone, every capability
    enabled and no target known) that asks for ``requested`` -- capability ids and presets (``complete``: every capability
    the request enables; ``readings``: the plane readings).

    ``exclude``: capabilities that must not run; the plan is refused (``EXCLUDED_CAPABILITY_REQUIRED``) when something
    requested requires one of them. ``targets``: the ids of the declared targets to evaluate (``None``: every one when the
    requirements are requested); naming targets requests the requirements, and a selection that leaves a declared target
    out is refused (``TARGET_SELECTION_INCOMPLETE``) when the plan runs a capability made of a complete evaluation (the
    record of the local workflow). ``level``: ``engine`` (the capabilities of
    ``os_optics.evaluate``) or ``local`` (those and the local workflow's, with the local members of the presets).
    ``what``: the name of the caller's parameter that holds ``requested`` (``capabilities`` of ``os_optics.evaluate``,
    ``outputs`` of the local workflow): a refusal names the parameter the caller wrote.
    """
    names, excluded_names = _ids(requested), _ids(exclude)
    _check_ids(names, level, what)
    _check_ids(excluded_names, level, "exclude")
    scope = enabled_capabilities(request, level)
    wanted, via = _expand(names, scope, level)
    excluded_members, excluded_via = _expand(excluded_names, scope, level)
    excluded = set(excluded_members)
    if request is not None:
        for name in wanted:
            reason = disabled_reason(name, request)
            if reason:
                raise EvaluationPlanError(CAPABILITY_NOT_ENABLED, f"{name} cannot be evaluated for this request: {reason}",
                                          capability=name)
    # (capability -> what requires it, and what the caller wrote that the chain starts from: the capability id, the preset
    # that includes it, or a target)
    needed_by: dict[str, str] = {}
    origin: dict[str, str] = {name: via.get(name, name) for name in wanted}
    declared = () if request is None else tuple(request.project.targets)
    selected = declared
    if targets is not None:
        chosen = _ids(targets)
        known = {t.target_id for t in declared}
        unknown = [t for t in chosen if t not in known]
        if unknown:
            raise EvaluationPlanError(UNKNOWN_TARGET, f"the Project declares no target {', '.join(repr(t) for t in unknown)}"
                                      + (f"; its targets are {', '.join(sorted(known))}" if known else "; it declares none"),
                                      targets=unknown)
        selected = tuple(t for t in declared if t.target_id in set(chosen))
        if REQUIREMENTS not in wanted:
            wanted.append(REQUIREMENTS)
            needed_by[REQUIREMENTS] = origin[REQUIREMENTS] = TARGET_SELECTION

    def close(start: str) -> None:
        pending = [start]
        while pending:
            current = pending.pop(0)
            for need in _requires(current, request, scope, level):
                if need not in origin:
                    origin[need] = origin[current]
                    needed_by[need] = current
                    wanted.append(need)
                    pending.append(need)

    for name in list(wanted):
        close(name)
    requirements: list[TargetRequirement] = []
    if REQUIREMENTS in origin:
        from ..validation.metrics import METRIC_REQUIRES  # (the metric registry: it imports the capability table)

        for target in selected:
            # (a target that is not assessable reads nothing, so its metric requires no capability)
            requires = METRIC_REQUIRES.get(target.metric, ()) if target.not_assessable is None else ()
            missing = []
            for need in requires:
                if request is not None and disabled_reason(need, request):
                    missing.append(need)
                elif need not in origin:
                    key = f"{TARGET_PREFIX}{target.target_id}"
                    origin[need], needed_by[need] = key, key
                    wanted.append(need)
                    close(need)
            requirements.append(TargetRequirement(target.target_id, target.metric, tuple(requires), tuple(missing)))
    metrics = {t.target_id: t.metric for t in declared}

    def needs(capability_id: str) -> tuple[str, ...]:
        return _requires(capability_id, request, scope, level)

    for name in (row.id for row in CAPABILITIES):
        if name in excluded and name in origin:
            by_preset = {"excluded_by": excluded_via[name]} if name in excluded_via else {}
            # (what else in the plan requires it, directly or through another capability: the capabilities, then the
            # evaluated targets whose metric reads it)
            requirers = tuple(c for c in (row.id for row in CAPABILITIES) if c in origin and c != name and _reaches(c, name, needs))
            if REQUIREMENTS not in requirers:  # (the requirements themselves do not require it: the targets that read it do)
                requirers += tuple(f"{TARGET_PREFIX}{r.target_id}" for r in requirements
                                   if any(need == name or _reaches(need, name, needs) for need in r.requires))
            raise EvaluationPlanError(EXCLUDED_CAPABILITY_REQUIRED,
                                      _excluded_text(name, needed_by, origin, metrics, excluded_via.get(name), what, requirers),
                                      capability=name, needed_by=needed_by.get(name, ""), origin=origin[name], **by_preset)
    executed = tuple(row.id for row in CAPABILITIES if row.id in origin)
    evaluated = {r.target_id for r in requirements}
    unselected = tuple(t.target_id for t in declared if t.target_id not in evaluated)
    if targets is not None and unselected:
        # (a capability made of a complete evaluation needs every declared target: its record states each one's outcome)
        whole = next((row.id for row in CAPABILITIES if row.complete_scope and row.id in origin), None)
        if whole is not None:
            raise EvaluationPlanError(TARGET_SELECTION_INCOMPLETE, _incomplete_text(whole, needed_by, origin, unselected),
                                      capability=whole, needed_by=needed_by.get(whole, ""), origin=origin[whole],
                                      targets=list(unselected))
    return EvaluationPlan(
        requested=_ordered(names), executed=executed,
        added=tuple(AddedCapability(c, needed_by[c]) for c in executed if c in needed_by),
        complete=all(c in origin for c in scope) and not unselected, excluded=_ordered(excluded_names), complete_scope=scope,
        targets=tuple(requirements), unselected_targets=unselected,
        excluded_capabilities=tuple(row.id for row in CAPABILITIES if row.id in excluded))


def _named(key: str, metrics: dict[str, str]) -> str:
    """A requirer in words: a capability id, the target selection or ``target <id> (metric <name>)``."""
    if key == TARGET_SELECTION:
        return "the selection of targets"
    if key.startswith(TARGET_PREFIX):
        target_id = key.removeprefix(TARGET_PREFIX)
        return f"target {target_id!r} (metric {metrics.get(target_id, '')})"
    return key


def _reaches(start: str, name: str, needs: Any) -> bool:
    """Whether the capability ``start`` requires ``name``, directly or through what it requires (``needs(id)``)."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        for need in needs(pending.pop()):
            if need == name:
                return True
            if need not in seen:
                seen.add(need)
                pending.append(need)
    return False


def _listed(names: tuple[str, ...]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _also_requiring(requirers: tuple[str, ...], what: str) -> list[str]:
    """``requirers`` in words, the capabilities and then the targets: ``a and b require <what>``, ``target 't' reads <what>``."""
    capabilities = tuple(c for c in requirers if not c.startswith(TARGET_PREFIX))
    targets = tuple(repr(c.removeprefix(TARGET_PREFIX)) for c in requirers if c.startswith(TARGET_PREFIX))
    parts = []
    if capabilities:
        parts.append(f"{_listed(capabilities)} {'requires' if len(capabilities) == 1 else 'require'} {what}")
    if targets:
        parts.append(f"{'target' if len(targets) == 1 else 'targets'} {_listed(targets)} {'reads' if len(targets) == 1 else 'read'} {what}")
    return parts


def _excluded_text(name: str, needed_by: dict[str, str], origin: dict[str, str], metrics: dict[str, str],
                   excluded_by: str | None = None, what: str = "requested", requirers: tuple[str, ...] = ()) -> str:
    """The message of ``EXCLUDED_CAPABILITY_REQUIRED``: the excluded capability, what requires it and the two ways out, in
    the words the caller wrote -- a preset that brought the capability into the request (``origin``) or into the
    exclusions (``excluded_by``) is named, never a member id the caller did not write. ``requirers``: the capabilities of
    the plan that require the excluded one, directly or through another, and the evaluated targets whose metric reads it
    (``target:<id>``): where a preset brought it, those the chain does not show are named as well, and the second way out
    is then to name capabilities that do not require it (the preset's other capabilities that require it would be refused
    in turn)."""
    root = origin[name]
    subject = f"{name} is excluded" + (f" (by the preset {excluded_by} in exclude)" if excluded_by else "")
    remove = f"Remove {excluded_by or name} from exclude"
    rule = "the evaluation plan runs nothing that is excluded and computes nothing in its place"
    instead = f"name the capabilities you want instead of {root}"
    exact = f"name capabilities that do not require {name} instead of {root}"
    if any(c.startswith(TARGET_PREFIX) for c in requirers):
        exact += f" (the {REQUIREMENTS} with targets that do not read it)"
    if name not in needed_by:
        if root in PRESETS:
            if requirers:
                also = ", and ".join(_also_requiring(requirers, "it"))
                return f"{subject}, and the preset {root} includes it, and {also}: {rule}. {remove}, or {exact}."
            return f"{subject}, and the preset {root} includes it: {rule}. {remove}, or {instead}."
        both = f"{name} is both requested and excluded" + (f" (by the preset {excluded_by} in exclude)" if excluded_by else "")
        requested = "the requested capabilities" if what == "requested" else what
        return f"{both}: the evaluation plan cannot run it and leave it out. {remove}, or remove it from {requested}."
    links = []
    shown = set()  # (the requirers the chain names)
    current = name
    while current in needed_by:
        needer = needed_by[current]
        verb = "reads" if needer.startswith(TARGET_PREFIX) else "asks for" if needer == TARGET_SELECTION else "requires"
        links.append(f"{_named(needer, metrics)} {verb} {current}")
        shown.add(needer)
        current = needer
    if root.startswith(TARGET_PREFIX):
        other = (f"leave target {root.removeprefix(TARGET_PREFIX)!r} out of the evaluation (select the others with targets, or do "
                 f"not request {REQUIREMENTS})")
    elif root == TARGET_SELECTION:
        other = "select no targets"
    elif root in PRESETS:
        others = tuple(c for c in requirers if c not in shown)
        if others:
            links.append(f"the preset {root} includes {current} ({'; '.join(_also_requiring(others, name))} as well)")
            other = exact
        else:
            links.append(f"the preset {root} includes {current}")
            other = instead
    else:
        other = f"do not request {root}"
    return f"{subject}, and {', and '.join(links)}: {rule}. {remove}, or {other}."


def _incomplete_text(name: str, needed_by: dict[str, str], origin: dict[str, str], unselected: tuple[str, ...]) -> str:
    """The message of ``TARGET_SELECTION_INCOMPLETE``: the capability made of a complete evaluation, what asked for it, the
    declared targets the selection leaves out and the two ways out."""
    root = origin[name]
    links = []
    current = name
    while current in needed_by:
        links.append(f"{needed_by[current]} requires {current}")
        current = needed_by[current]
    if root in PRESETS:
        links.append(f"the preset {root} includes {current}")
        other = f"name the capabilities you want instead of {root}"
    else:
        other = f"do not request {root}"
    left_out = ", ".join(repr(t) for t in unselected)
    asked = f" ({', and '.join(links)})" if links else ""
    return (f"{name} is made of a complete evaluation{asked}: it needs every declared target, and the selection of targets "
            f"leaves out {left_out}. Give no selection of targets (every declared target is then evaluated) or name every "
            f"declared target, or {other}.")


__all__ = ["CAPABILITY_NOT_ENABLED", "EXCLUDED_CAPABILITY_REQUIRED", "TARGET_SELECTION_INCOMPLETE", "UNKNOWN_CAPABILITY",
           "UNKNOWN_TARGET", "AddedCapability",
           "EvaluationPlan", "EvaluationPlanError", "TargetRequirement", "disabled_reason", "enabled_capabilities",
           "plan_evaluation"]
