"""The comparison of the configurations of one Project (``AnalysisResultV1``, analysis ``CONFIGURATIONS``).

A Project with ``configurations`` runs each configuration as its own Engine execution (``os_optics.local.configurations``).
This analysis reads those runs and sets their numbers side by side; it computes nothing new and changes no run. Every
quantity is scoped to its configuration (``scope.configuration_id``) and states its own ``QuantityState``:

* ``power_W`` per configuration and observation plane: the plane's received power as the run's Local Validation record
  states it -- its state (``KNOWN``, ``BOUNDED`` with the run's interval ``[lower, upper bound]``, PROVEN unless the value rests on pupil samples (then
  SAMPLING, with the sampling resolution as convergence evidence), ``UNKNOWN``, ...) and
  the issue code behind a missing value;
* per configuration, plane and arriving beam (``scope.path_id``, ``source_id``, ``wavelength_nm``; the crossing index is
  part of the quantity id): ``position_u_mm`` / ``position_v_mm`` (chief ray in the plane frame), ``beam_radius_x_mm`` /
  ``beam_radius_y_mm`` (Gaussian 1/e^2 radii, where the beam has a Gaussian state) and ``spot_rms_mm`` (pupil-ray RMS radius,
  where pupil rays arrived) -- the run's own values;
* with a paraxial request (``paraxial``: the components in the order light meets them and the wavelength, as
  ``os_optics.local.paraxial.paraxial_analysis`` takes it), ``efl_mm`` and ``bfl_mm`` of those components in each
  configuration; an afocal system has none (``NOT_APPLICABLE``, reason ``AFOCAL``), and a configuration the paraxial
  analysis refuses states the refusal's code (``OUT_OF_MODEL_SCOPE``).

Quantity ids are ``<configuration>|<observation>|<path>|<crossing>|<name>`` for beams, ``<configuration>|<observation>|power_W``
for planes and ``<configuration>|paraxial|<name>`` for the paraxial rows.

``arrays`` holds the side-by-side tables: ``configurations`` -- per configuration, in the Project's order, its label,
directory, Engine input and result hashes, validation status, execution record (outcome, termination, budgets) and
requirement record, or the problems that stopped it; ``targets`` -- per target, its metric and unit and per configuration
whether it applies there, its value, status and assessment. There is no verdict over the configurations: each
configuration's requirements stay its own.

``execution`` is ``COMPLETED`` when every configuration's run completed, else ``PARTIAL`` with the configurations that did
not (``BUDGET_EXHAUSTED`` when each of them stopped at a budget); what the other configurations computed stands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.analysis import (AnalysisResultV1, Convergence, ConvergenceMethod, Interval, IntervalKind, ModelBasis, Quantity,
                                  QuantityScope)
from ..errors.status import QuantityState, TerminationReason
from ..manifest.canonical import hash_exact
from . import completed, document, partial

CONFIGURATIONS_ANALYSIS = "CONFIGURATIONS"
CONFIGURATIONS_ANALYSIS_VERSION = "1.1"
RUN_MODEL_ID = "configuration_run"
PARAXIAL_MODEL_ID = "paraxial"
#: The beam quantities read from each arrival: (name, unit, how to read it).
BEAM_QUANTITIES = (
    ("position_u_mm", "mm", lambda a: a.uv_mm[0]),
    ("position_v_mm", "mm", lambda a: a.uv_mm[1]),
    # (the comparison reads valid Gaussian states only)
    ("beam_radius_x_mm", "mm", lambda a: None if a.beam.gaussian is None or not a.beam.gaussian.valid else a.beam.gaussian.beam_radius_x_mm),
    ("beam_radius_y_mm", "mm", lambda a: None if a.beam.gaussian is None or not a.beam.gaussian.valid else a.beam.gaussian.beam_radius_y_mm),
    ("spot_rms_mm", "mm", lambda a: a.pupil_rms_radius_mm),
)
COMPLETED_OUTCOME = "COMPLETED"
#: A beam quantity id has five parts; the fourth is the crossing index.
BEAM_ID_PARTS = 5
BEAM_ID_CROSSING = 3


@dataclass(frozen=True)
class ConfigurationEvidence:
    """What one configuration's run left: its record and Engine result (both ``None`` when the run did not complete its
    files), its execution record (JSON), and the problems that stopped a run that produced no result."""

    configuration_id: str
    label: str | None
    directory: str
    engine_input_hash: str | None  # (None: the configuration's Project could not be lowered)
    execution: Mapping[str, Any]
    record: Any = None
    result: Any = None
    project: Any = None
    problems: tuple[Mapping[str, Any], ...] = field(default=())


def _run_basis() -> ModelBasis:
    return ModelBasis(model_id=RUN_MODEL_ID, name="configuration run", version=CONFIGURATIONS_ANALYSIS_VERSION,
                      assumptions=("each value is the configuration's own run result, read without recomputation",))


def _paraxial_basis() -> ModelBasis:
    return ModelBasis(model_id=PARAXIAL_MODEL_ID, name="paraxial ABCD of placed elements", version="1",
                      assumptions=("elements on one optical axis, air between them (os_optics.local.paraxial)",))


def _power(cid: str, quantity: Any) -> Quantity:
    scope = QuantityScope(configuration_id=cid, observation_id=quantity.observation_id)
    qid = f"{cid}|{quantity.observation_id}|power_W"
    state = QuantityState(quantity.quantity_state) if quantity.quantity_state is not None else QuantityState.UNKNOWN
    common = {"quantity_id": qid, "name": "power_W", "unit": "W", "scope": scope, "basis": RUN_MODEL_ID,
              "depends_on": tuple(quantity.issue_codes)}
    # the run's evidence as the run states it: a sampled value keeps its resolution (convergence evidence, never a bound), a
    # bounded plane keeps its own lower and upper bounds, and an interval resting on a sampled value is not proven
    resolution = getattr(quantity, "power_sampling_resolution_W", None)
    convergence = None if resolution is None else Convergence(method=ConvergenceMethod.NONE, parameter="pupil_samples", resolution=resolution)
    if state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO):
        return Quantity(state=state, value=quantity.total_power_W, convergence=convergence, **common)
    if state is QuantityState.BOUNDED:
        lower = getattr(quantity, "power_lower_bound_W", None)
        kind = IntervalKind.PROVEN if resolution is None else IntervalKind.SAMPLING
        return Quantity(state=state, convergence=convergence, **common,
                        # (an upper end nothing bounds is open: the run's power_upper_bound_W is null there)
                        interval=Interval(lower=quantity.total_power_W if lower is None else lower,
                                          upper=quantity.power_upper_bound_W, kind=kind))
    reason = quantity.issue_codes[0] if quantity.issue_codes else (quantity.quantity_state or "QUANTITY_STATE_NOT_RECORDED")
    return Quantity(state=state, reason=reason, **common)


def _beams(cid: str, result: Any) -> list[Quantity]:
    out = []
    for observation in result.observations:
        for arrival in observation.arrivals:
            beam = arrival.beam
            scope = QuantityScope(configuration_id=cid, observation_id=observation.observation_id, source_id=beam.source_id,
                                  path_id=beam.path_id, wavelength_nm=beam.wavelength_nm)
            for name, unit, read in BEAM_QUANTITIES:
                value = read(arrival)
                if value is None:
                    continue
                out.append(Quantity(quantity_id=f"{cid}|{observation.observation_id}|{beam.path_id}|{arrival.crossing_index}|{name}",
                                    name=name, unit=unit, scope=scope, state=QuantityState.KNOWN, value=value, basis=RUN_MODEL_ID))
    return out


def _paraxial(cid: str, project: Any, request: Mapping[str, Any], catalog: Any) -> list[Quantity]:
    from ..local.errors import LocalRunError
    from ..local.paraxial import ParaxialError, paraxial_analysis

    scope = QuantityScope(configuration_id=cid)
    names = ("efl_mm", "bfl_mm")
    try:
        system = paraxial_analysis(project, request, catalog=catalog)["system"]
    except ParaxialError as exc:
        code = str(exc.problems[0].get("code")) if exc.problems else "PARAXIAL_ANALYSIS_INVALID"
        return [Quantity(quantity_id=f"{cid}|paraxial|{n}", name=n, unit="mm", scope=scope, state=QuantityState.OUT_OF_MODEL_SCOPE,
                         reason=code, message=str(exc), basis=PARAXIAL_MODEL_ID) for n in names]
    except LocalRunError as exc:
        code = exc.problems[0].code.value
        return [Quantity(quantity_id=f"{cid}|paraxial|{n}", name=n, unit="mm", scope=scope, state=QuantityState.UNKNOWN,
                         reason=code, message=str(exc), basis=PARAXIAL_MODEL_ID) for n in names]
    out = []
    for name in names:
        value = system[name]
        if value is None:
            out.append(Quantity(quantity_id=f"{cid}|paraxial|{name}", name=name, unit="mm", scope=scope,
                                state=QuantityState.NOT_APPLICABLE, reason="AFOCAL", basis=PARAXIAL_MODEL_ID))
        else:
            out.append(Quantity(quantity_id=f"{cid}|paraxial|{name}", name=name, unit="mm", scope=scope, state=QuantityState.KNOWN,
                                value=value, basis=PARAXIAL_MODEL_ID))
    return out


def _configuration_row(evidence: ConfigurationEvidence) -> dict[str, Any]:
    record = evidence.record
    row: dict[str, Any] = {"configuration_id": evidence.configuration_id, "label": evidence.label, "directory": evidence.directory,
                           "engine_input_hash": evidence.engine_input_hash, "execution": dict(evidence.execution)}
    if record is not None:
        outcome = record.simulation.outcome
        row.update(result_hash=record.result_hash.digest, validation_status=record.validation_status.value,
                   requirements=None if outcome is None else outcome.requirements.model_dump(mode="json"))
    else:
        row.update(result_hash=None, validation_status=None, requirements=None)
    row["problems"] = [dict(p) for p in evidence.problems]
    return row


def _targets(project: Any, evidence: Sequence[ConfigurationEvidence]) -> list[dict[str, Any]]:
    rows = []
    for target in project.targets:
        cells = []
        for item in evidence:
            applies = target.configuration_ids is None or item.configuration_id in target.configuration_ids
            cell: dict[str, Any] = {"configuration_id": item.configuration_id, "applies": applies}
            outcome = None if item.record is None else next(
                (t for t in item.record.simulation.target_outcomes if t.target_id == target.target_id), None)
            if applies and outcome is not None:
                cell.update(value=outcome.value, unit=outcome.unit, status=outcome.status.value, assessment=outcome.assessment)
            cells.append(cell)
        rows.append({"target_id": target.target_id, "metric": target.metric, "unit": target.unit, "configurations": cells})
    return rows


def _execution(evidence: Sequence[ConfigurationEvidence]) -> Any:
    incomplete = [e for e in evidence if e.execution.get("outcome") != COMPLETED_OUTCOME]
    if not incomplete:
        return completed()
    terminations = {str(e.execution.get("termination")) for e in incomplete}
    termination = (TerminationReason.BUDGET_EXHAUSTED if terminations == {TerminationReason.BUDGET_EXHAUSTED.value}
                   else TerminationReason(sorted(terminations - {TerminationReason.BUDGET_EXHAUSTED.value})[0]))
    names = ", ".join(f"{e.configuration_id} ({e.execution.get('outcome')}, {e.execution.get('termination')})" for e in incomplete)
    return partial(termination, f"configurations whose run did not complete: {names}; the other configurations' values stand")


def configurations_comparison(project: Any, evidence: Sequence[ConfigurationEvidence], *, engine_input_hash: str,
                              paraxial: Mapping[str, Any] | None = None, catalog: Any = None,
                              engine_build_id: str | None = None) -> AnalysisResultV1:
    """The ``CONFIGURATIONS`` analysis document of ``project`` (the Project with its configurations) from each run
    configuration's ``evidence``, in the Project's order (module docstring). ``engine_input_hash``: the run's
    configurations input hash (``os_optics.local.configurations.configurations_input_hash``). ``paraxial``: an optional
    paraxial request evaluated on each configuration's Project (``evidence.project``). ``engine_build_id``: the Engine
    build id the caller already holds (the session's that ran the configurations); without one it is identified here."""

    from .._vendor.shared_contracts.optical_design import project_targets_digest

    quantities: list[Quantity] = []
    for item in evidence:
        if item.record is not None:
            quantities.extend(_power(item.configuration_id, q) for q in item.record.simulation.observation_quantities or ())
        if item.result is not None:
            quantities.extend(_beams(item.configuration_id, item.result))
        if paraxial is not None and item.project is not None:
            quantities.extend(_paraxial(item.configuration_id, item.project, paraxial, catalog))
    settings: dict[str, Any] = {"configuration_ids": [e.configuration_id for e in evidence]}
    if paraxial is not None:
        settings["paraxial"] = dict(paraxial)
    models = [_run_basis()] + ([_paraxial_basis()] if paraxial is not None else [])
    result_hashes = [[e.configuration_id, None if e.record is None else e.record.result_hash.digest] for e in evidence]
    return document(
        CONFIGURATIONS_ANALYSIS, CONFIGURATIONS_ANALYSIS_VERSION, settings, _execution(evidence),
        engine_input_hash=engine_input_hash, run_result_hash=hash_exact({"configurations": result_hashes}),
        targets_hash=project_targets_digest(project), models=models, quantities=quantities,
        arrays={"configurations": [_configuration_row(e) for e in evidence], "targets": _targets(project, evidence)},
        engine_build_id=engine_build_id,
    )


def comparison_table(comparison: AnalysisResultV1) -> dict[str, Any]:
    """The side-by-side view of a comparison: one row per quantity name and scope (its configuration left out), one
    column per configuration (``value``, or the ``state`` and ``reason`` of a quantity without one)."""

    columns = [row["configuration_id"] for row in comparison.arrays.get("configurations", [])]  # type: ignore[union-attr]
    rows: dict[tuple, dict[str, Any]] = {}
    for q in comparison.quantities:
        scope = q.scope.model_dump(mode="json", exclude_none=True)
        cid = scope.pop("configuration_id", None)
        parts = q.quantity_id.split("|")
        crossing = parts[BEAM_ID_CROSSING] if len(parts) == BEAM_ID_PARTS else None
        key = (q.name, tuple(sorted(scope.items())), crossing)
        row = rows.setdefault(key, {"name": q.name, "unit": q.unit, **scope,
                                    **({"crossing_index": int(crossing)} if crossing is not None else {}), "values": {}})
        cell: dict[str, Any] = {"state": q.state.value}
        if q.value is not None:
            cell["value"] = q.value
        if q.interval is not None:
            cell["interval"] = [q.interval.lower, q.interval.upper]
        if q.reason:
            cell["reason"] = q.reason
        row["values"][cid] = cell
    return {"configurations": columns, "rows": list(rows.values()), "targets": comparison.arrays.get("targets", [])}


__all__ = ["CONFIGURATIONS_ANALYSIS", "CONFIGURATIONS_ANALYSIS_VERSION", "ConfigurationEvidence", "comparison_table",
           "configurations_comparison"]
