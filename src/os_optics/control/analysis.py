"""Control / recoverability analysis: sensitivity, nominal alignment, disturbance cases, deterministic Monte Carlo."""

from __future__ import annotations

from ..config import NumericalConfig
from ..contracts.project import ProjectSnapshot
from ..contracts.results import ControlResult, ControllerRun, RecoverabilityCase, SensitivityResult
from ..errors import ErrorCode, Status, ValidationIssue, make_issue, worst
from ..math.rng import DeterministicRng
from ..optics.gaussian.truncation import gauss_legendre_rule
from .actuation import all_dofs, resolve_dofs
from .lm import run_lm
from .readings import evaluate
from .sensitivity import analyse, jacobian, total_values


#: The Monte Carlo seed when neither ``control_spec.monte_carlo.seed`` nor ``configuration.random_seed`` states one: a fixed
#: value, so the same request gives the same samples and result hash, and two designs compared without a seed see the same
#: disturbance stream; it is recorded like a stated seed.
DEFAULT_MONTE_CARLO_SEED = 0


def monte_carlo_seed(spec_seed: int | None, request_seed: int | None) -> int:
    """The seed the Monte Carlo samples are drawn with: the spec's, else the request's, else ``DEFAULT_MONTE_CARLO_SEED``."""
    if spec_seed is not None:
        return spec_seed
    return request_seed if request_seed is not None else DEFAULT_MONTE_CARLO_SEED


def _classify(run: ControllerRun, sensitivity: SensitivityResult | None) -> tuple[str, str]:
    if run.lost_beam:
        return "LOST_UNRECOVERABLE", "beam lost and not recaptured (capture range or actuator travel insufficient)"
    if run.converged:
        return "RECOVERED", ""
    if run.saturated_actuators:
        return "SATURATED", f"actuator travel limits reached: {', '.join(run.saturated_actuators)}"
    if sensitivity is not None and not sensitivity.observable:
        return "UNOBSERVABLE", "the sensors cannot observe some actuator DOFs (rank-deficient Jacobian)"
    return "NOT_CONVERGED", "controller did not converge"


def run_control_analysis(project: ProjectSnapshot, cfg: NumericalConfig, request_seed: int | None) -> tuple[ControlResult, list[ValidationIssue]]:
    spec = project.control_spec
    assert spec is not None
    issues: list[ValidationIssue] = []
    dofs = resolve_dofs(project, spec.actuator_refs)
    refs = [d.ref for d in dofs]
    catalogue = all_dofs(project)
    command0 = {d.ref: d.spec.zero for d in dofs}
    tol = spec.recoverability.convergence_tolerance if spec.recoverability is not None else cfg.lm_residual_tol
    sensor_ids = [s.sensor_id for s in spec.sensors]

    # actuator states whose readings are not known, each with the sensors whose readings were not
    unknown_states: list[tuple[dict[str, float], tuple[str, ...]]] = []
    # one Gauss-Legendre rule for every reading of this validation call (read-only: the traces share it)
    quad_rule = gauss_legendre_rule(cfg.aperture_quad_sector_nodes)
    for part in quad_rule:
        part.setflags(write=False)

    def evaluator(values: dict[str, float]):
        reading = evaluate(project, spec, values, cfg, quad_rule=quad_rule)
        if reading.unknown:
            unknown_states.append((values, tuple(s for s, u in zip(sensor_ids, reading.unknown_sensors) if u)))
        return reading

    def undetermined_sensors(since: int = 0) -> list[str]:
        """The sensors (in ``spec.sensors`` order) whose readings were not known in the states read since ``since``."""
        found = {s for _values, sensors in unknown_states[since:] for s in sensors}
        return [s for s in sensor_ids if s in found]

    # ---- sensitivity at the nominal state ------------------------------------------------------------------------------
    sensitivity: SensitivityResult | None = None
    base = evaluator(total_values(command0, {}))
    if base.unknown:
        # no signal at the nominal state, but the light toward the sensors was not all followed: not a lost beam. Alignment,
        # recoverability and Monte Carlo minimise over every sensor's residual together, so one unknown reading stops them.
        unknown_ids = undetermined_sensors()
        named = ", ".join(unknown_ids)
        issues.append(make_issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.UNKNOWN,
                                 f"the readings of sensor{'s' if len(unknown_ids) > 1 else ''} {named} at the nominal state are not "
                                 "known: light toward them was not followed (see the observation planes' untraced causes), so the "
                                 "control analysis is not run",
                                 location="control", detected_fraction=base.detected_fraction, sensor_ids=unknown_ids,
                                 suggested_action="resolve why the light is not followed (for example an annular beam that a later "
                                                  "aperture or the sensor's active area cuts), then analyse control"))
        return ControlResult(status=Status.UNKNOWN, message=f"sensor readings not known at the nominal state: {named}"), issues
    if spec.compute_sensitivity:
        if base.lost:
            issues.append(make_issue(ErrorCode.BEAM_LOST, Status.FAIL, "the nominal design does not deliver signal to the sensors",
                                     location="control", detected_fraction=base.detected_fraction,
                                     suggested_action="fix the nominal optical path before analysing control"))
        else:
            import numpy as np

            jr = jacobian(evaluator, dofs, command0, {}, len(spec.sensors), cfg, base=base)
            weights = np.array([s.weight for s in spec.sensors])
            sensitivity = analyse(weights[:, None] * jr.jacobian, refs, sensor_ids, jr.orders, jr.jacobian, cfg)
            if sensitivity.status is Status.FAIL:
                issues.append(make_issue(ErrorCode.UNOBSERVABLE_DOF, Status.FAIL, sensitivity.message, location="control",
                                         unobservable=list(sensitivity.unobservable_actuator_refs), rank=sensitivity.numerical_rank,
                                         suggested_action="add a sensor that observes the missing DOFs"))
            elif sensitivity.status is Status.WARNING:
                issues.append(make_issue(ErrorCode.JACOBIAN_ILL_CONDITIONED, Status.WARNING, sensitivity.message, location="control",
                                         condition_number=sensitivity.condition_number))

    # ---- nominal alignment ---------------------------------------------------------------------------------------------
    seen = len(unknown_states)
    nominal = run_lm(evaluator, dofs, spec, command0, {}, cfg, tol)
    nominal_unknown = nominal.run.status is Status.FAIL and len(unknown_states) > seen

    # ---- explicit disturbance cases ------------------------------------------------------------------------------------
    cases: list[RecoverabilityCase] = []
    undetermined: list[int] = []  # cases whose failure came with readings that were not known
    if spec.recoverability is not None:
        for dist in spec.recoverability.disturbances:
            _check_refs(dist, catalogue)
            seen = len(unknown_states)  # (before the run: the readings of this disturbance follow)
            out = run_lm(evaluator, dofs, spec, command0, dict(sorted(dist.items())), cfg, tol)
            label, msg = _classify(out.run, sensitivity)
            cases.append(RecoverabilityCase(disturbance=dict(sorted(dist.items())), outcome=label, run=out.run, message=msg))  # type: ignore[arg-type]
            if label != "RECOVERED" and len(unknown_states) > seen:
                # a reading on the way was not known (light not followed): the outcome is not a measured failure
                undetermined.append(len(cases) - 1)
                unknown_ids = undetermined_sensors(seen)
                issues.append(make_issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.UNKNOWN,
                                         f"disturbance not decided: readings of {', '.join(unknown_ids)} were not known ({msg or label})",
                                         location="control", outcome=label, sensor_ids=unknown_ids,
                                         disturbance=[f"{k}={v:g}" for k, v in sorted(dist.items())]))
            elif label != "RECOVERED":
                code = {"LOST_UNRECOVERABLE": ErrorCode.CAPTURE_RANGE_INSUFFICIENT, "SATURATED": ErrorCode.DOF_SATURATED,
                        "UNOBSERVABLE": ErrorCode.UNOBSERVABLE_DOF}.get(label, ErrorCode.NOT_RECOVERABLE)
                issues.append(make_issue(code, Status.FAIL, msg or "disturbance not recovered", location="control",
                                         outcome=label, disturbance=[f"{k}={v:g}" for k, v in sorted(dist.items())]))
    frac = None
    if cases:
        frac = sum(1 for c in cases if c.outcome == "RECOVERED") / len(cases)

    # ---- deterministic Monte Carlo ---------------------------------------------------------------------------------------
    mc_frac = mc_lost = mc_sat = None
    mc_n, mc_seed = 0, None
    if spec.monte_carlo is not None:
        mc = spec.monte_carlo
        # the stated seed, else the documented default: drawn and recorded (ControlResult.monte_carlo_seed, the manifest)
        seed = monte_carlo_seed(mc.seed, request_seed)
        _check_refs(mc.amplitude, catalogue)
        rng = DeterministicRng(seed)
        keys = sorted(mc.amplitude)
        recovered = lost = sat = 0
        for _ in range(mc.sample_count):
            dist = {}
            for k in keys:
                amp = mc.amplitude[k]
                dist[k] = rng.uniform_between(-amp, amp) if mc.distribution == "uniform" else amp * rng.normal()
            out = run_lm(evaluator, dofs, spec, command0, dist, cfg, tol)
            label, _msg = _classify(out.run, sensitivity)
            recovered += label == "RECOVERED"
            lost += label == "LOST_UNRECOVERABLE"
            sat += label == "SATURATED"
        n = mc.sample_count
        mc_frac, mc_lost, mc_sat = recovered / n, lost / n, sat / n
        mc_n, mc_seed = n, seed

    statuses = [Status.UNKNOWN if nominal_unknown else nominal.run.status] + [Status.PASS if c.outcome == "RECOVERED" else Status.UNKNOWN if k in undetermined else Status.FAIL
                                       for k, c in enumerate(cases)]
    if unknown_states and spec.monte_carlo is not None:
        statuses.append(Status.UNKNOWN)
        unknown_ids = undetermined_sensors()
        issues.append(make_issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.UNKNOWN,
                                 f"{len(unknown_states)} sensor readings during the analysis were not known (light not followed; "
                                 f"sensors {', '.join(unknown_ids)}); the Monte Carlo fractions count those states as not recovered",
                                 location="control", sensor_ids=unknown_ids))
    if sensitivity is not None:
        statuses.append(sensitivity.status)
    if base.lost:
        statuses.append(Status.FAIL)
    result = ControlResult(
        status=worst(statuses, empty=Status.UNKNOWN), sensitivity=sensitivity, nominal_run=nominal.run,
        recoverability_cases=tuple(cases), recoverability_fraction=frac, monte_carlo_recovered_fraction=mc_frac,
        monte_carlo_sample_count=mc_n, monte_carlo_seed=mc_seed, monte_carlo_lost_fraction=mc_lost,
        monte_carlo_saturated_fraction=mc_sat,
    )
    return result, issues


def _check_refs(values: dict[str, float], catalogue: dict) -> None:
    from ..errors import EngineInputError

    for ref in values:
        if ref not in catalogue:
            raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"unknown actuator {ref!r} in a disturbance / Monte Carlo spec",
                                   location="control", known=sorted(catalogue))
