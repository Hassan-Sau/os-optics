"""Selective evaluation of a Project: ``EvaluationSession``, ``Evaluation`` and ``evaluate_project``.

An evaluation computes what its caller asks for -- capability ids and the presets ``complete`` and ``readings``
(``os_optics.evaluation.capability_table()``) -- with what those require, and nothing else::

    from os_optics.local import EvaluationSession, example_project

    session = EvaluationSession()
    ev = session.evaluate(example_project(), outputs=("readings",))
    ev.scope                # what the caller asked for: requested, executed, added (with what required each), not executed
    ev.describe_scope()     # the same as plain JSON, with the declared targets not selected
    ev.result               # the Engine ValidationResult; ev.result.evaluation is the Engine-level scope (below)
    ev.results().monitor("focal_plane")       # the scientific results, read from memory
    ev.results().power_budget()               # ResultsError NOT_COMPUTED: request power_ledger
    full = ev.extend(outputs=("plane_bounds",))   # runs what the readings did not compute, reusing the Engine products
    run = ev.run()          # the complete run (LocalOpticalRun): record and 3D view; ev.write_run(directory)
    session.statistics()    # what the session prepared, compiled, executed and reused

``readings`` computes the plane readings: each beam's arrival at each observation plane with its traced power, and the
model-scope findings that say whether a number applies; it leaves out the probe traces of light the beam trace does
not follow, the power ledger, the plane bounds, the surface exposures, the beam pairs, the requirements (no declared
target is assessed), the optical-solid check, the record and the view. ``complete`` is the run of
``run_optical_project``: ``run_optical_project(project)`` is
``EvaluationSession().evaluate(project, outputs=("complete",)).run()``. In a Project evaluation the plane readings and
the requirements always come with the model-scope findings (``model_scope``, which reads the ray paths): excluding
either is refused while they run.

``ev.result.evaluation`` is the scope of the Engine evaluation the local one asked for: its ``requested`` and
``executed`` list the Engine capabilities of the plan by id, with no preset and no local capability. What the caller
asked for, with the local capabilities and what was added, is ``ev.scope`` / ``ev.describe_scope()`` (and the
``evaluation`` of a JSON reply).

A session keeps, for the evaluations made through it: the catalogs it loaded, the compiled elements of the latest
Project it lowered (by what each compiles from: product, catalog release, model kind, or declared model with the
Project's models, materials and coatings; never its pose), the Engine build id and one ``EvaluationState`` holding the
products of the Engine's capabilities. A candidate that differs physically (a pose, a source, a plane, a budget, a
prescription, a material) invalidates the Engine's products. A moved element, a changed source, plane or budget reuse
the catalog and the compiled elements; a changed prescription, material or coating of a declared model compiles the
declared elements again, and the session then holds those in place of the earlier ones (the compiled elements it keeps
do not grow over a candidate loop). A changed target threshold reuses every product. One evaluation at a time per
session.

A Local Validation record is made only of an evaluation that ran every capability the request enables, the
optical-solid check and the model scope, for every declared target (what ``record`` requires; the view is not part of
it), and only of optical bodies that are valid solids: an ERROR finding of the optical-solid check refuses an evaluation
that makes the record (``LocalRunError`` carrying the report). An evaluation that asks for ``solid_check`` without the
record is not refused for one: ``ev.optical_solids`` is the report (``valid`` false, each finding with its ``zone`` and
measured value, the thickness of every gap) beside the readings. A run directory or a submission is made of the complete run (record and view): ``run()`` /
``write_run(directory)`` first compute whatever is missing, every declared target included. An evaluation made with
``exclude=`` is not completed that way: the plan runs nothing that is excluded, so ``run()`` / ``write_run(directory)``
refuse it (``EvaluationPlanError`` ``EXCLUDED_CAPABILITY_REQUIRED``) before anything runs; evaluate the Project again
without the exclusion, or call ``run_optical_project``. ``record`` requested with an exclusion it needs is refused the
same way, and so is ``record`` with a selection of targets that leaves a declared target out
(``TARGET_SELECTION_INCOMPLETE``). An evaluation writes nothing: the run directory (``run_files``) is written by
``write_run`` alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .._vendor.shared_contracts.optical_design import (
    LocalRuntimeKindV1,
    ProjectDocumentV2,
    nominal_retained_dof_states,
    validated_project_ref,
)
from .._vendor.shared_contracts.optical_design.dof import dof_sort_index
from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1, OpticalViewSettingsV1
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..evaluation.capabilities import (
    ANALYSIS,
    BEAM_TRACE,
    CAPABILITY_BY_ID,
    COMPLETE,
    ENGINE,
    EXPORT,
    EXPORT_BY_ID,
    LOCAL,
    LOCAL_PRESET_MEMBERS,
    MODEL_SCOPE,
    PLANE_BOUNDS,
    PLANE_READINGS,
    PRESETS,
    RAY_PATHS,
    READINGS,
    RECORD,
    RUN_FILES,
    SOLID_CHECK,
    VIEW,
    analyses,
    capability,
    exports,
    level_capabilities,
    not_computed_produces,
)
from ..evaluation.plan import EXCLUDED_CAPABILITY_REQUIRED, EvaluationPlan, EvaluationPlanError, plan_evaluation
from ..evaluation.state import EvaluationState
from ..manifest import engine_build_id
from ..run_control import RunStopped, active_monitor, begin_attempt, end_attempt, monitored
from ..validation.ray_paths import RAY_DATA_FULL, RAY_DATA_LEVELS, RayPathTrace, ray_path_recorder, ray_paths_document
from .artifacts import recorded_status
from .catalog import load_project
from . import run as stages  # (the capabilities of a local run, one function each)
from .run import LocalOpticalRun, completed_write, element_models, outcome_lead, requirements_coverage, stop_run
from .record import scientific_projection, validation_status_of
from .sample_emission import check_excitation
from .validity import ModelScopeIssue


def _ids(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _members(names: tuple[str, ...]) -> set[str]:
    """The capabilities ``names`` stand for at the local level, presets expanded (``complete``: every capability)."""

    out: set[str] = set()
    for name in names:
        if name == COMPLETE:
            out.update(row.id for row in level_capabilities(LOCAL))
        elif name in PRESETS:
            out.update(LOCAL_PRESET_MEMBERS[name])
            out.add(PLANE_READINGS)
        else:
            out.add(name)
    return out


def _same_physics(a: ValidationRequest | None, b: ValidationRequest) -> bool:
    """Whether two requests trace the same: everything but their targets equal (a target threshold traces nothing)."""

    if a is None or a.configuration != b.configuration:
        return False
    return a.project.model_copy(update={"targets": ()}) == b.project.model_copy(update={"targets": ()})


class EvaluationSession:
    """Evaluations of Projects that reuse what they share (module docstring).

    ``catalog``: the public catalog every evaluation uses (default: the release each Project pins when this runtime holds
    it, else the active one -- resolved at each evaluation, so a release installed or made active since is used, and
    loaded once while it stays the one in force). A catalog file (a path) is read on each evaluation and known by its content: the
    file edited on disk is loaded, and its elements compiled, anew at the next evaluation; an unchanged file is reused.
    Another catalog is read on each evaluation by the identity of the object: assigning another catalog to
    ``session.catalog`` takes effect at the next evaluation (its elements are compiled anew), while a catalog document
    edited in place after its first use is not read again -- pass a new object. ``model_kind``: the simulation
    model of catalog products that publish several; ``runtime_kind``: the runtime the Local Validation record of a
    complete run names (default: detected).
    """

    def __init__(self, catalog: Any = None, model_kind: str | None = None, runtime_kind: LocalRuntimeKindV1 | str | None = None) -> None:
        self.catalog = catalog
        self.model_kind = model_kind
        self.runtime_kind = runtime_kind
        self.state = EvaluationState()
        self._catalogs: dict[Any, Any] = {}
        self._compiled: dict[Any, Any] = {}
        self._recorder: Any = None
        self._recorded_request: ValidationRequest | None = None
        self._counts: dict[str, int] = {"evaluations": 0, "prepared": 0, "compiled_elements": 0, "reused_elements": 0}
        self._local_executions: dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------ public
    def evaluate(self, project: Any, outputs: Iterable[str] | str = (READINGS,), exclude: Iterable[str] | str = (),
                 targets: Iterable[str] | None = None, monitor: Any = None, *,
                 view_settings: OpticalViewSettingsV1 | None = None) -> Evaluation:
        """Evaluate ``project`` (a ``ProjectDocumentV2``, its JSON document or text, or a file path) for ``outputs``:
        capability ids and presets (``readings``, the default: the plane readings with their model-scope findings;
        ``complete``: everything a run computes). The plan adds what they require; ``exclude`` names capabilities that
        must not run (refused, ``EvaluationPlanError``, when something requested requires one); ``targets`` the declared
        targets to evaluate (naming targets requests ``requirements``). ``view_settings``: what the view draws, when
        ``view`` is computed. ``monitor``: progress, cancellation and limits as for ``run_optical_project``; a stop raises
        ``RunStopped`` and leaves the session holding what finished before it, so the next evaluation continues.

        Raises ``LocalRunError`` for a Project that cannot be simulated as given and ``EvaluationPlanError`` for a
        selection that cannot be planned."""

        monitor = monitor if monitor is not None else active_monitor()
        facts: dict[str, Any] = {}
        try:
            with monitored(monitor):
                return self.evaluation(project, outputs, exclude, targets, view_settings=view_settings, facts=facts)
        except RunStopped as stopped:
            raise stop_run(stopped, facts.get("document"), monitor, engine_input_hash=facts.get("engine_input_hash")) from None
        except KeyboardInterrupt:
            if monitor is None or not hasattr(monitor, "stopped_by_interrupt"):
                raise
            raise stop_run(monitor.stopped_by_interrupt(), facts.get("document"), monitor,
                           engine_input_hash=facts.get("engine_input_hash")) from None

    def statistics(self) -> dict[str, Any]:
        """What the session did: ``evaluations``; ``prepared`` (Projects lowered); ``compiled_elements`` /
        ``reused_elements`` (elements compiled, and taken from the session instead: the session holds the compiled
        elements of the latest Project it lowered); ``catalogs`` (loaded); ``executions``
        per capability (Engine and local; a capability whose product was reused did not execute); ``invalidations`` of
        the Engine's products by reason; ``retained``: the Engine capabilities whose products the session holds."""

        executions = {**self.state.executions, **self._local_executions}
        return {**self._counts, "catalogs": len(self._catalogs),
                "executions": {c: executions[c] for c in (*CAPABILITY_BY_ID, *EXPORT_BY_ID) if c in executions},
                "invalidations": dict(sorted(self.state.invalidations.items())), "retained": list(self.state.retained),
                "engine_build_id": self.state.build_id}

    @property
    def build_id(self) -> str:
        """The Engine build id of this session's evaluations (its Engine state's ``build_id``), identified once."""

        if self.state.build_id is None:
            self.state.build_id = engine_build_id()
        return self.state.build_id

    def lowering(self, project: Any, *, check_solids: bool = True) -> Any:
        """``project`` lowered to its Engine request with the catalogs and compiled elements this session keeps
        (``lower_project``, optical solids checked; ``check_solids=False`` reports them without refusing), outside any
        phase: a lowering made before the evaluations that use it (``evaluation(..., lowered=...)``) -- every configuration
        of a Project is lowered before any is traced (``local/configurations.py``)."""

        known = set(self._compiled)
        lowered = stages.lowered_project(load_project(project), self.catalog, model_kind=self.model_kind, check_solids=check_solids,
                                         catalogs=self._catalogs, compiled=self._compiled, engine_state=self.state)
        self._count_lowering(known, lowered)
        return lowered

    # ------------------------------------------------------------------------------------------------------ internals
    def _count_lowering(self, known: set[Any], lowered: Any) -> None:
        """Count one lowering: its elements compiled (those the session did not hold: ``known`` are the keys it held
        before) and those taken from the session. By key, since a lowering drops the elements it did not use."""

        compiled = sum(1 for key in self._compiled if key not in known)
        self._counts["prepared"] += 1
        self._counts["compiled_elements"] += compiled
        self._counts["reused_elements"] += len(lowered.elements) - compiled

    def _count(self, capability: str) -> None:
        self._local_executions[capability] = self._local_executions.get(capability, 0) + 1

    def _recorder_for(self, request: ValidationRequest) -> Any:
        """The recorder of the beam trace the next evaluation makes or reads: the one that observed the retained trace when
        the request traces the same with the same Engine build, else a new one (the Engine then traces again: a state the
        Engine binds to another identity drops its trace). Every trace of the session is observed, whether or not the
        evaluation that makes it reads the ray paths, so a later evaluation that reads them does not trace again."""

        state = self.state
        if (self._recorder is not None and state.has(BEAM_TRACE) and state.recorder is self._recorder
                and state.identity is not None and state.identity.engine_build_id == state.build_id
                and _same_physics(self._recorded_request, request)):
            return self._recorder
        self._recorder, self._recorded_request = ray_path_recorder(), request
        return self._recorder

    def evaluation(self, project: Any, outputs: Iterable[str] | str = (READINGS,), exclude: Iterable[str] | str = (),
                   targets: Iterable[str] | None = None, *, view_settings: OpticalViewSettingsV1 | None = None,
                   ray_data: str = RAY_DATA_FULL, writes: bool = False, facts: dict[str, Any] | None = None,
                   prepared: Evaluation | None = None, lowered: Any = None) -> Evaluation:
        """``evaluate`` inside the caller's monitor context, without converting a stop (``run_optical_project`` keeps its
        own stop handling). ``prepared``: an evaluation of the same Project whose lowering is reused (``extend``);
        ``lowered``: the Project's lowering by ``lowering`` (with its optical solids checked), used instead of lowering it
        again."""

        facts = {} if facts is None else facts
        ran: list[str] = []  # (the local capabilities this call runs)

        def did(capability: str) -> None:
            ran.append(capability)
            self._count(capability)

        names, excluded_names = _ids(outputs), _ids(exclude)
        target_ids = None if targets is None else _ids(targets)
        # (whether the optical solids are checked is known from the names alone: it decides how the Project is lowered.
        # An ERROR finding refuses the Project only when the record is made -- the record is made of valid bodies;
        # any other evaluation that asks for the check carries its report, findings included, beside what it computed)
        named = plan_evaluation(names, level=LOCAL, what="outputs")
        solids = SOLID_CHECK in named.executed and SOLID_CHECK not in _members(excluded_names)
        records = RECORD in named.executed
        plan: EvaluationPlan | None = None
        if prepared is not None:
            document, catalog, lowered = prepared.project, prepared.catalog, prepared.lowered
            facts["document"] = document
            # (its request is known: a selection that cannot be planned is refused before anything runs or is counted)
            plan = plan_evaluation(names, exclude=excluded_names, targets=target_ids, request=lowered.request, level=LOCAL,
                                   what="outputs")
            if solids and lowered.solids is None:
                lowered = stages.checked_solids(document, lowered, refuse=records)
                did(SOLID_CHECK)
        else:
            document = load_project(project)
            facts["document"] = document  # (a run stopped from here on states this Project's observations and targets)
            if lowered is not None:
                catalog, lowered = stages.prepare_project(document, self.catalog, model_kind=self.model_kind, solids=solids,
                                                          catalogs=self._catalogs, lowered=lowered)
            else:
                known = set(self._compiled)
                catalog, lowered = stages.prepare_project(document, self.catalog, model_kind=self.model_kind, solids=solids,
                                                          catalogs=self._catalogs, compiled=self._compiled, engine_state=self.state,
                                                          refuse_solids=records)
                self._count_lowering(known, lowered)
            if solids:
                did(SOLID_CHECK)
        if records:
            # (a lowering made earlier -- by an evaluation without the record, or by ``lowering`` -- holds its report unrefused)
            stages.require_valid_solids(lowered)
        facts["engine_input_hash"] = lowered.engine_input_hash
        request = lowered.request
        if plan is None:
            plan = plan_evaluation(names, exclude=excluded_names, targets=target_ids, request=request, level=LOCAL, what="outputs")
        self._counts["evaluations"] += 1
        complete_run = plan.runs(RECORD) and plan.runs(VIEW)
        if complete_run:
            stages.install_preflight(active_monitor(), document, lowered, ray_data=ray_data, view_settings=view_settings, writes=writes)
        engine = tuple(c for c in plan.executed if CAPABILITY_BY_ID[c].level == ENGINE)
        engine_excluded = tuple(c for c in plan.excluded if c in PRESETS or CAPABILITY_BY_ID[c].level == ENGINE)
        recorder = self._recorder_for(request) if plan.runs(BEAM_TRACE) else None
        result, ray_paths = stages.traced_evaluation(request, capabilities=engine, exclude=engine_excluded, targets=target_ids,
                                                     state=self.state, recorder=recorder, ray_paths=plan.runs(RAY_PATHS))
        if ray_paths is not None:
            did(RAY_PATHS)
        stages.require_complete(request, result)
        if plan.runs(PLANE_BOUNDS):
            # (a sample's emission rests on the excitation this run delivers: the check reads the planes' completed power,
            # which the plane bounds give, so it is made only when they run)
            check_excitation(request, result)
        scope_issues: tuple[ModelScopeIssue, ...] | None = None
        if plan.runs(MODEL_SCOPE):
            assert ray_paths is not None
            scope_issues = stages.scope_findings(lowered, result, ray_paths)
            did(MODEL_SCOPE)
        record = None
        if plan.runs(RECORD):
            assert scope_issues is not None
            record = stages.run_record(document, lowered, result, scope_issues, runtime_kind=self.runtime_kind)
            did(RECORD)
        scene = None
        if plan.runs(VIEW):
            assert ray_paths is not None
            scene = stages.run_view(document, catalog, lowered, result, ray_paths, view_settings=view_settings, model_kind=self.model_kind)
            did(VIEW)
        return Evaluation(
            session=self, project=document, catalog=catalog, lowered=lowered, scope=plan, result=result, ray_paths=ray_paths,
            model_scope_issues=scope_issues, local_validation=record, optical_view=scene, targets=target_ids,
            view_settings=view_settings,
            executed=(*self.state.last_executed, *ran), reused=tuple(self.state.last_reused))


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of a Project (``EvaluationSession.evaluate``, ``evaluate_project``).

    ``result``: the Engine ``ValidationResult`` (``result.evaluation`` is the scope of the Engine evaluation, ``None`` when
    complete: its ``requested`` / ``executed`` are the Engine capabilities of the plan by id); ``scope``: the plan of what
    the caller asked for (``requested`` as written, ``executed`` with the local capabilities, ``added`` with what required
    each, ``not_executed``);
    ``model_scope_issues``: the model-scope findings (``None`` when ``model_scope`` was not computed); ``ray_paths``,
    ``local_validation``, ``optical_view``: the products of ``ray_paths``, ``record`` and ``view`` (``None`` when not
    computed); ``executed`` / ``reused``: the capabilities this call ran and those whose retained products it read (facts
    of this call, part of no hash).
    """

    session: EvaluationSession
    project: ProjectDocumentV2
    catalog: Any
    lowered: Any
    scope: EvaluationPlan
    result: ValidationResult
    ray_paths: RayPathTrace | None = None
    model_scope_issues: tuple[ModelScopeIssue, ...] | None = None
    local_validation: Any = None
    optical_view: OpticalSimulationSceneV1 | None = None
    targets: tuple[str, ...] | None = None
    view_settings: OpticalViewSettingsV1 | None = None
    executed: tuple[str, ...] = ()
    reused: tuple[str, ...] = ()

    @property
    def request(self) -> ValidationRequest:
        return self.lowered.request

    @property
    def engine_input_hash(self) -> str:
        return self.lowered.engine_input_hash

    @property
    def catalog_version(self) -> str | None:
        return self.lowered.catalog_version

    @property
    def optical_solids(self) -> Any:
        """The optical-solid report of ``solid_check`` (``os_optics.optical_solids.OpticalSolidsReport``; ``None`` when the
        check was not computed): ``valid`` and every finding with its code, severity, location, ``zone`` and measured
        value, and per element the centre, minimum and edge thickness of every gap. An evaluation without the record
        carries a report with ERROR findings beside its readings; the record, and so the complete run, is refused for it."""
        return self.lowered.solids

    @property
    def complete(self) -> bool:
        """Whether this is a complete run's evaluation: every capability the request enables, every target, the record and
        the view."""
        return self.result.evaluation is None and self.local_validation is not None and self.optical_view is not None

    @property
    def status(self) -> str:
        """The diagnostic summary of what this evaluation computed: the worst status of its targets, planes and issues, at
        least ``OUT_OF_MODEL_SCOPE`` when a model-scope finding exists (the requirements' outcome is ``result.outcome``).
        It is no verdict on a declared target this evaluation did not assess (``requirements()['not_assessed']``): when the
        Project declares targets and it assessed none, a summary that would be ``PASS`` is ``NOT_APPLICABLE``."""
        return validation_status_of(self.result, bool(self.model_scope_issues)).value

    @property
    def identity(self) -> dict[str, Any]:
        """What the numbers belong to: Engine version and build, the Engine input hash, the hash of the design's physical
        inputs (targets excluded), the result and trace hashes, the Project hash and the catalog release."""

        m = self.result.manifest
        return {"engine_version": m.engine_version if m else None, "engine_build_id": m.engine_build_id if m else None,
                "engine_input_hash": self.engine_input_hash,
                "design_projection_hash": m.design_projection_hash if m else None,
                "result_hash": m.result_hash if m else None, "trace_hash": m.trace_hash if m else None,
                "project_hash": self.project.content_hash(), "catalog_version": self.catalog_version}

    def not_computed(self) -> list[dict[str, Any]]:
        """Each capability of the request's complete scope this evaluation did not compute: why (``NOT_REQUESTED`` /
        ``EXCLUDED``) and what it produces."""

        # (the surface exposures read the return-light analysis for a design that states a power rating)
        rated = any(part.power_ratings for part in self.request.project.components)
        return [{"capability": c, "reason": "EXCLUDED" if c in self.scope.excluded_capabilities else "NOT_REQUESTED",
                 "produces": list(not_computed_produces(c, self.scope.executed, rated))} for c in self.scope.not_executed]

    def describe_scope(self) -> dict[str, Any]:
        """The scope of this evaluation as plain JSON: requested, executed, added, not computed, the declared targets not selected
        (``ValidationResult.evaluation``) and whether it is complete."""

        scope = self.result.evaluation
        return {"requested": list(self.scope.requested), "executed": list(self.scope.executed),
                "added": [{"capability": a.capability, "needed_by": a.needed_by} for a in self.scope.added],
                "excluded": list(self.scope.excluded), "not_computed": self.not_computed(),
                "targets_not_evaluated": [] if scope is None else [t.model_dump(mode="json") for t in scope.targets_not_evaluated],
                "complete": self.complete}

    def requirements(self) -> dict[str, Any] | None:
        """The requirements of this evaluation, as its JSON reply leads with them: ``declared``, the targets it assessed by
        assessment (``met``, ``unmet``, ``indeterminate``, ``not_applicable``; the published models' applicability counted)
        and ``not_assessed``: each declared target it did not assess, in Project order, with its reason. The five lists
        name every declared target once, so a target that was not assessed never reads as met. ``None`` for a result that
        states no outcome."""

        summary = scientific_projection(self.result, self.model_scope_issues or ())
        return requirements_coverage(outcome_lead(summary.outcome, self.status)["requirements"], self.result.evaluation)

    def results(self) -> Any:
        """The scientific results of this evaluation (``os_optics.results.RunResults``), built from memory. A query of a
        part this evaluation did not compute raises ``ResultsError`` ``NOT_COMPUTED`` naming the capability to request."""

        from ..results.model import RunResults

        document = self.project
        status: dict[str, Any] = {}
        if self.local_validation is not None and self.model_scope_issues is not None:
            # (an evaluation with its record states what its run directory's manifest states)
            status = recorded_status(self.local_validation, self.model_scope_issues)
        elif self.model_scope_issues is not None:
            status["model_scope_issues"] = [issue.as_dict() for issue in self.model_scope_issues]
        recorded = {"status": status,
                    "project": validated_project_ref(document).model_dump(mode="json", by_alias=True, exclude_none=True),
                    "catalog_version": self.catalog_version, "element_models": [dict(m) for m in element_models(self.lowered)],
                    # (the record's order -- component id, then Tx..Rz -- so the list is the one the run directory's
                    # manifest states)
                    "retained_dofs": [d.model_dump(mode="json", exclude_none=True) for d in sorted(
                        nominal_retained_dof_states(document), key=lambda d: (d.instance_id, dof_sort_index(d.axis)))],
                    "evaluation": self.describe_scope()}
        ray_data = ray_paths_document(self.ray_paths) if self.ray_paths is not None else None
        return RunResults(result=self.result, ray_data=ray_data, numerical=self.request.configuration.numerical,
                          request=self.request.model_dump(mode="json"), project=document.payload(), recorded=recorded,
                          executed=self.scope.executed, engine_build_id=self.session.build_id)

    def extend(self, outputs: Iterable[str] | str = (), *, monitor: Any = None) -> Evaluation:
        """The evaluation of what this one computed and ``outputs``, over the same session: the Engine products this one
        left in the session are reused (the beam trace, the secondary rays and the probe traces are not made again); the
        ray-path record and the model-scope findings are made again from the retained trace for the new result, and are
        listed in ``executed`` and counted in ``statistics()["executions"]`` each time. The result equals a fresh
        evaluation of the same scope. It evaluates the targets this one selected, except when ``outputs`` bring in the
        record (``record``, ``complete``): the record is made of every declared target, so the selection is dropped and
        every declared target is evaluated. The exclusions this evaluation was made with (``exclude=``) still apply:
        ``outputs`` that need an excluded capability are refused (``EvaluationPlanError``
        ``EXCLUDED_CAPABILITY_REQUIRED``) before anything runs. ``monitor``: progress, cancellation and limits as for
        ``run_optical_project``; a stop or Ctrl-C raises ``RunStopped`` with its record."""

        names = (*self.scope.requested, *(n for n in _ids(outputs) if n not in self.scope.requested))
        targets = None if plan_evaluation(names, level=LOCAL, what="outputs").runs(RECORD) else self.targets
        return self._again(names, targets, monitor)

    def run(self, *, ray_data: str = RAY_DATA_FULL, monitor: Any = None) -> LocalOpticalRun:
        """The complete run of this evaluation (``LocalOpticalRun``): what this evaluation did not compute is computed first,
        reusing the Engine products it left in the session; every declared target is evaluated (a selection of targets is
        dropped). An evaluation made with ``exclude=`` is not completed: the complete run computes every capability and the
        plan runs nothing that is excluded, so it is refused (``EvaluationPlanError`` ``EXCLUDED_CAPABILITY_REQUIRED``)
        before anything runs -- evaluate the Project again without the exclusion and call ``run()`` on that evaluation, or
        call ``run_optical_project``. ``ray_data``: what ``ray_data.json`` records. ``monitor``: progress, cancellation
        and limits as for ``run_optical_project``; a stop or Ctrl-C raises ``RunStopped`` with its record."""

        if ray_data not in RAY_DATA_LEVELS:
            raise ValueError(f"ray_data is one of {', '.join(RAY_DATA_LEVELS)}, not {ray_data!r}")
        self._require_completable("run()", "run_optical_project(project)")
        return self._complete_run(ray_data, monitor, writes=False)

    def write_run(self, output_dir: Path | str, *, ray_data: str = RAY_DATA_FULL, monitor: Any = None) -> LocalOpticalRun:
        """``run()`` written to the run directory ``output_dir`` (``run_files``); the run, with ``artifacts``. Every
        declared target is evaluated (a selection of targets is dropped). An evaluation made with ``exclude=`` is refused
        as ``run()`` refuses it (``EvaluationPlanError`` ``EXCLUDED_CAPABILITY_REQUIRED``), before anything runs or
        ``output_dir`` is touched. ``monitor``: progress, cancellation and limits
        as for ``run_optical_project``, the writing of the files included: a run that stops, or fails before the run is
        moved into place, leaves ``output_dir`` as it found it (a stop adds its record, ``run_stopped.json``), so an earlier
        run there still verifies; a failure or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves
        what a process killed there leaves -- no ``manifest.json``, so no run that verifies, and the Engine's records of
        what is there -- and any run then completes there, as ``run_optical_project`` does; a stop or a Ctrl-C once the
        manifest is in place returns the completed run, with no stop record."""

        if ray_data not in RAY_DATA_LEVELS:
            raise ValueError(f"ray_data is one of {', '.join(RAY_DATA_LEVELS)}, not {ray_data!r}")
        self._require_completable("write_run(output_dir)", "run_optical_project(project, output_dir=output_dir)")
        monitor = monitor if monitor is not None else active_monitor()
        held: dict[str, Any] = {}  # (the run being written: what a stop after its move into place answers with)
        attempt = begin_attempt()
        try:
            with monitored(monitor):
                run = self._complete_run(ray_data, monitor, writes=True)
                held["run"] = run
                run = replace(run, artifacts=run.write_artifacts(output_dir))
                self.session._count(RUN_FILES)
                return run
        except RunStopped as stopped:
            if (written := completed_write(held, output_dir)) is not None:
                return written
            raise stop_run(stopped, self.project, monitor, engine_input_hash=self.engine_input_hash,
                           output_dir=output_dir) from None
        except KeyboardInterrupt:
            if (written := completed_write(held, output_dir)) is not None:
                return written
            if monitor is None or not hasattr(monitor, "stopped_by_interrupt"):
                raise
            raise stop_run(monitor.stopped_by_interrupt(), self.project, monitor, engine_input_hash=self.engine_input_hash,
                           output_dir=output_dir) from None
        finally:
            end_attempt(attempt)

    def _require_completable(self, method: str, direct: str) -> None:
        """Refuse ``method`` (``run()`` / ``write_run(output_dir)``) of an evaluation made with an exclusion the complete
        run computes, before anything runs or a directory is touched: in the words of that call, which has neither
        ``outputs`` nor ``exclude`` -- the exclusion was written where the evaluation was made. ``direct``: the call that
        makes the complete run without this evaluation."""

        if self.complete:
            return
        try:
            plan_evaluation((*self.scope.requested, RECORD, VIEW), exclude=self.scope.excluded, targets=None,
                            request=self.request, level=LOCAL, what="outputs")
        except EvaluationPlanError as refused:
            if refused.code != EXCLUDED_CAPABILITY_REQUIRED:
                raise
            raise EvaluationPlanError(
                EXCLUDED_CAPABILITY_REQUIRED,
                f"{method} makes the complete run, which computes {refused.context['capability']}, and this evaluation was "
                f"made with exclude={', '.join(self.scope.excluded)}: the evaluation plan runs nothing that is excluded and "
                f"computes nothing in its place. Evaluate the Project again without exclude and call {method} on that "
                f"evaluation (in the same session it reuses the Engine products this one left), or make the complete run "
                f"with {direct}.", **refused.context) from None

    def _complete_run(self, ray_data: str, monitor: Any, *, writes: bool) -> LocalOpticalRun:
        """The complete run: this evaluation when it is complete, else the evaluation of the rest (every declared target)."""

        ev = self if self.complete else self._again((*self.scope.requested, RECORD, VIEW), None, monitor, ray_data=ray_data,
                                                    writes=writes)
        assert ev.local_validation is not None and ev.optical_view is not None and ev.ray_paths is not None
        assert ev.model_scope_issues is not None
        # (the record, the scene and the result are made by this evaluation from one result object: that they agree is held
        # by tests; the run directory's manifest states the binding checks once, where the run is written)
        return LocalOpticalRun(
            project=ev.project, catalog_version=ev.catalog_version, simulated_parts=ev.lowered.simulated_parts,
            request=ev.request, engine_input_hash=ev.engine_input_hash, result=ev.result, local_validation=ev.local_validation,
            ray_paths=ev.ray_paths, optical_view=ev.optical_view, model_scope_issues=ev.model_scope_issues,
            element_models=element_models(ev.lowered), ray_data=ray_data)

    def _again(self, names: tuple[str, ...], targets: tuple[str, ...] | None, monitor: Any, *, ray_data: str = RAY_DATA_FULL,
               writes: bool = False) -> Evaluation:
        """The evaluation of ``names`` over this one (its lowering reused). ``ray_data`` / ``writes``: what the complete
        run's estimate models (the ray data recorded, whether run files are written)."""

        session = self.session
        monitor = monitor if monitor is not None else active_monitor()
        try:
            with monitored(monitor):
                return session.evaluation(self.project, names, self.scope.excluded, targets, view_settings=self.view_settings,
                                          ray_data=ray_data, writes=writes, prepared=self)
        except RunStopped as stopped:
            raise stop_run(stopped, self.project, monitor, engine_input_hash=self.engine_input_hash) from None
        except KeyboardInterrupt:
            if monitor is None or not hasattr(monitor, "stopped_by_interrupt"):
                raise
            raise stop_run(monitor.stopped_by_interrupt(), self.project, monitor,
                           engine_input_hash=self.engine_input_hash) from None


def evaluate_project(project: Any, outputs: Iterable[str] | str = (READINGS,), exclude: Iterable[str] | str = (),
                     targets: Iterable[str] | None = None, monitor: Any = None, *, catalog: Any = None,
                     model_kind: str | None = None, view_settings: OpticalViewSettingsV1 | None = None,
                     configuration_id: str | None = None) -> Evaluation:
    """One evaluation in a session of its own (``EvaluationSession(catalog, model_kind).evaluate(...)``): its ``extend``
    and ``run`` still reuse the Engine products it computed. ``configuration_id``: the configuration to evaluate, of a
    Project that declares configurations (its ``configured(id)``; ``configurations.configured_project``)."""

    if configuration_id is not None:
        from .configurations import configured_project  # (configurations.py imports this module)

        project = configured_project(project, configuration_id)
    return EvaluationSession(catalog=catalog, model_kind=model_kind).evaluate(project, outputs, exclude, targets, monitor,
                                                                              view_settings=view_settings)


#: What each preset computes, in words (``capability_index``).
PRESET_SUMMARIES = {
    COMPLETE: "Everything a run computes: every capability the Project's request enables, every declared target, the optical-solid "
              "check, the ray paths, the model scope, the Local Validation record and the 3D view (run_optical_project). "
              "The run directory is written by Evaluation.write_run.",
    READINGS: "The plane readings: each beam's arrival at each observation plane with its traced power, and the model-scope "
              "findings that say whether a number applies. Without the probe traces of untraced light, the power ledger, the "
              "plane bounds, the surface exposures, the beam pairs, the requirements (no declared target is assessed), the "
              "optical-solid check, the record and the view.",
}


def _preset_members(preset: str) -> list[str]:
    return list(plan_evaluation((preset,), level=LOCAL).executed)


def capability_index() -> dict[str, Any]:
    """What a Project evaluation selects from, in brief: the capabilities (``id``, ``level``, ``kind``, ``summary``,
    ``requires``, and ``local_requires`` where the model-scope findings come with it), the presets (``summary``,
    ``members``), the analyses run on demand and the exports written by ``Evaluation.write_run`` (``summary``,
    ``requires``, ``entry_point``: the Python call with its arguments, ``command``): what ``capability_descriptor``
    describes in full, and ``preset_descriptor`` for a preset. The layout checks and the control analysis are capabilities
    of an Engine request only (``os_optics.evaluate``) and are not listed."""

    return {"capabilities": [{"id": row.id, "level": row.level, "kind": row.kind, "summary": row.summary,
                              "requires": list(row.requires),
                              **({"local_requires": list(row.local_requires)} if row.local_requires else {})}
                             for row in level_capabilities(LOCAL)],
            "presets": {name: {"summary": PRESET_SUMMARIES[name], "members": _preset_members(name)} for name in PRESETS},
            "analyses": [{"id": row.id, "summary": row.summary, "requires": list(row.requires), "entry_point": row.entry_point,
                          "command": row.command} for row in analyses()],
            "exports": [{"id": row.id, "summary": row.summary, "requires": list(row.requires), "entry_point": row.entry_point,
                         "command": row.command} for row in exports()]}


def preset_descriptor(preset_id: str) -> dict[str, Any]:
    """One preset of a Project evaluation in full: ``id``, ``summary``, ``members`` (the capabilities it runs, in
    execution order) and ``leaves_out`` (the capabilities of a Project evaluation it does not run; each is described by
    ``capability_descriptor``). ``KeyError`` for an id that is not a preset."""

    if not isinstance(preset_id, str) or preset_id not in PRESETS:
        raise KeyError(preset_id)
    members = _preset_members(preset_id)
    return {"id": preset_id, "summary": PRESET_SUMMARIES[preset_id], "members": members,
            "leaves_out": [row.id for row in level_capabilities(LOCAL) if row.id not in members]}


def capability_descriptor(capability_id: str) -> dict[str, Any]:
    """One capability, analysis or export of a Project evaluation in full: ``id``, ``level`` (``engine`` / ``local``),
    ``kind`` (``computation``, ``recording``, ``presentation``, ``analysis`` or ``export``), ``summary``, ``requires``
    (and ``traced_requires``: what it reads as well when the request traces; ``rated_requires``: what it requires as well
    of a Project that states a power rating; ``local_requires``: what a Project evaluation runs with it and does not let
    be excluded; ``requires_complete_scope``: true where it also requires every capability the Project's request enables
    and every declared target -- the record), ``produces`` (the result sections and fields it produces), ``metrics`` (the target metrics that read
    it: ``MetricDescriptor.requires``), ``controls`` (the ``NumericalConfig`` fields that affect it; of them, the
    ``compute_budget`` ones and the ``numerical_settings`` ones a Project sets), ``phases`` (the run phases it enters), ``method``, ``version``, the presets
    that include it, and for an analysis or an export its ``entry_point`` (the Python call with its arguments),
    ``command`` and ``request_schema``. The model-scope capability also states ``quantity_kinds``: the ``kinds`` of
    quantity, per finding code the kinds it touches (``findings``) and per target metric the kinds it reads (``metrics``);
    a finding names a target when the two lists share a kind. ``KeyError`` for an id a Project evaluation does not hold (the layout checks and
    the control analysis are capabilities of an Engine request only) and for a preset (``preset_descriptor``)."""

    from .._vendor.shared_contracts.optical_design.compute_budget import BUDGET_RANGES
    from .._vendor.shared_contracts.optical_design.numerical_settings import SETTING_NAMES
    from ..validation.metrics import METRIC_REQUIRES
    from ..validation.quantity_kinds import METRIC_QUANTITY_KINDS, QuantityKind
    from .validity import MODEL_SCOPE_AFFECTS

    row = capability(capability_id)
    if row.id in CAPABILITY_BY_ID and row not in level_capabilities(LOCAL):
        raise KeyError(capability_id)
    doc: dict[str, Any] = {
        "id": row.id, "level": row.level, "kind": row.kind, "summary": row.summary, "requires": list(row.requires),
        **({"traced_requires": list(row.traced_requires)} if row.traced_requires else {}),
        **({"rated_requires": list(row.rated_requires)} if row.rated_requires else {}),
        **({"local_requires": list(row.local_requires)} if row.local_requires else {}),
        **({"requires_complete_scope": True} if row.complete_scope else {}),
        "produces": list(row.produces),
        "metrics": sorted(m for m, needs in METRIC_REQUIRES.items() if row.id in needs),
        "controls": list(row.controls), "compute_budget": [c for c in row.controls if c in BUDGET_RANGES],
        "numerical_settings": [c for c in row.controls if c in SETTING_NAMES],
        "phases": list(row.phases), "method": row.method, "version": row.version,
    }
    if row.id == MODEL_SCOPE:
        doc["quantity_kinds"] = {
            "kinds": [kind.value for kind in QuantityKind],
            "findings": {code: [kind.value for kind in QuantityKind if kind in kinds] for code, kinds in MODEL_SCOPE_AFFECTS.items()},
            "metrics": {metric: [kind.value for kind in QuantityKind if kind in kinds]
                        for metric, kinds in sorted(METRIC_QUANTITY_KINDS.items())}}
    if row.kind in (ANALYSIS, EXPORT):
        doc["entry_point"] = row.entry_point
        doc["command"] = row.command
        doc["request_schema"] = row.request_schema or None
    else:
        doc["presets"] = [name for name in PRESETS if row.id in _preset_members(name)]
    return doc


__all__ = ["COMPLETE", "PRESET_SUMMARIES", "READINGS", "Evaluation", "EvaluationSession", "capability_descriptor",
           "capability_index", "evaluate_project", "preset_descriptor"]
