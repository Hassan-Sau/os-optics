"""The exit code of a run: ``project run``, ``workspace run``, ``validate`` and ``project evaluate`` under one
``--status-contract``.

A run answers several independent questions (the scientific-result contract, ``docs/engine/RESULTS_AND_EXPORT.md``): did the calculation
complete, and how did the assessed targets come out. The exit code reports one of them, chosen by the status contract.
Under ``requirements`` it reads ``outcome.execution`` and the lists of the requirement record, nothing else:

* ``requirements`` (the default): 0 the calculation completed (execution ``COMPLETED``) and no assessed target is
  ``UNMET``, ``INDETERMINATE`` or ``NOT_APPLICABLE``: every assessed target is met, or the Project declares no targets;
  1 an assessed target is not met (``UNMET``); 2 an assessed target is undetermined
  (``INDETERMINATE``) or does not apply to the light it names (``NOT_APPLICABLE``), none unmet; 3 the calculation did not
  complete (execution ``PARTIAL``: a budget stopped the beam trace; ``FAILED`` or ``CANCELLED``), a result that states
  no execution (no ``outcome``), or a run a memory or run-file limit stopped. A budget an auxiliary analysis ran out of
  (the untraced-light probes, the return-light analysis, the beam pairs) is recorded in ``outcome.execution.budgets`` and
  on the quantities it leaves without a bound, and does not make the execution ``PARTIAL``;
* ``execution``: 0 the calculation completed, 1 anything else, a stopped run included;
* ``aggregate``: the worst-of diagnostic severity ``ValidationResult.status``: 0 PASS / WARNING / NOT_APPLICABLE,
  1 FAIL, 2 UNKNOWN or a stopped run, 3 OUT_OF_MODEL_SCOPE.

Which targets are assessed, and which record and status the code reads:

* ``validate``: every declared target of the request; the Engine result's requirement record; ``ValidationResult.status``;
* ``project run``, ``workspace run``: every declared target; the Local Validation record's requirements, which also count
  the published models' applicability (a target a model-scope finding affects is ``INDETERMINATE``); the run's validation
  status;
* ``project run`` of a Project with configurations (``configurations_exit_code``): every declared target of each
  configuration that ran, per configuration as above, combined -- ``requirements``: 3 when any configuration's calculation
  did not complete, else 1 when a target of any is not met, else 2 when one is undetermined or does not apply, else 0;
  ``execution``: 0 when every configuration completed, else 1; ``aggregate``: the most severe configuration's code in the
  order FAIL (1), OUT_OF_MODEL_SCOPE (3), UNKNOWN (2), the others (0). A configuration without a result (its Engine
  execution failed) counts as not completed (``aggregate``: UNKNOWN);
* ``project evaluate``: the targets named by ``--targets``, every declared target when ``requirements`` is among the
  outputs, none otherwise (the default ``readings``); the reply's requirement record; the status of what was computed.
  Its default contract follows the selection (``selection_contract``): ``requirements`` when it assesses targets,
  ``execution`` when it assesses none. ``requirements`` named by ``--status-contract`` or ``OS_OPTICS_STATUS_CONTRACT``
  for an evaluation that assesses none of the Project's declared targets is refused before anything is computed (64,
  ``REQUIREMENTS_NOT_SELECTED``); a Project that declares no targets exits 0 under it once the calculation completed.
  The declared targets it did not assess are listed in the reply's ``requirements.not_assessed`` (and
  ``evaluation.targets_not_evaluated``) and take no part; a target left undetermined because a capability was outside
  the evaluation (issue code ``CAPABILITY_NOT_COMPUTED``) is ``INDETERMINATE``: 2.

Under every contract: ``validate`` on input that is not a valid request exits 1 (its result is a structured FAIL with
``INVALID_INPUT``); a Project that cannot be simulated exits 65 (``project evaluate``: also a Project that declares
configurations); a usage error exits 64 (``project evaluate``: also an empty ``--outputs`` / ``--targets`` or a selection
that cannot be planned); an internal error of os-optics exits 70 (``INTERNAL_ERROR_EXIT``: the printed JSON names
``INTERNAL_ERROR`` and the exception type, the traceback goes to standard error); Ctrl-C exits 130 (``INTERRUPTED``).

The contract is chosen by ``--status-contract``, else by the environment variable ``OS_OPTICS_STATUS_CONTRACT``, else it
is ``requirements`` (``project evaluate``: the contract its selection answers). The JSON summary of ``project run`` and ``workspace run`` and the reply of ``project evaluate`` name
it as ``exit_contract``. ``aggregate`` is a permanent third contract, a selectable diagnostic summary.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Mapping
from typing import Any

#: The aggregate contract: ``ValidationResult.status`` -> exit code (also the exit code of ``assembly validate``).
EXIT_CODES = {"PASS": 0, "WARNING": 0, "NOT_APPLICABLE": 0, "FAIL": 1, "UNKNOWN": 2, "OUT_OF_MODEL_SCOPE": 3}
STATUS_CONTRACTS = ("aggregate", "execution", "requirements")
DEFAULT_STATUS_CONTRACT = "requirements"
STATUS_CONTRACT_ENV = "OS_OPTICS_STATUS_CONTRACT"
#: ``requirements``: the exit codes, by what they report.
REQUIREMENTS_MET = 0
REQUIREMENT_UNMET = 1
REQUIREMENT_UNDECIDED = 2
EXECUTION_INCOMPLETE = 3
STATUS_CONTRACT_HELP = (
    "what the exit code reports (default: requirements, or the environment variable OS_OPTICS_STATUS_CONTRACT): "
    "requirements -- 0 the calculation completed and every declared target is met (a Project with no targets: the "
    "calculation completed), 1 a declared target is not met, 2 a declared target is undetermined or does not apply to the "
    "light it names (NOT_APPLICABLE, such as a two-beam metric on beams of different wavelengths) and none is unmet, 3 the "
    "calculation did not complete (PARTIAL: a budget stopped the beam trace; FAILED; CANCELLED; a run a memory or run-file "
    "limit stopped; or a result that states no execution; a budget an auxiliary analysis ran out of -- untraced-light "
    "probes, return-light analysis, beam pairs -- is recorded in outcome.execution.budgets and on the quantities it "
    "leaves without a bound, and does not make a run PARTIAL); execution -- 0 the calculation completed, 1 anything "
    "else, a stopped run included; "
    "aggregate -- the worst-of diagnostic severity: 0 PASS / WARNING / NOT_APPLICABLE, 1 FAIL, 2 UNKNOWN or a stopped run, "
    "3 OUT_OF_MODEL_SCOPE. The JSON summary names the contract used as exit_contract. A Project with configurations: the "
    "same contract over every configuration (requirements: 3 if any did not complete, else 1 if a target of any is not "
    "met, else 2 if one is undetermined; execution: 1 if any did not complete; aggregate: the most severe, FAIL before "
    "OUT_OF_MODEL_SCOPE before UNKNOWN)."
)
EXIT_CONTRACT_EPILOG = (
    "Exit codes of 'project run', 'workspace run', 'validate' and 'project evaluate' follow --status-contract (default "
    "requirements): 0 = the calculation completed and every declared target is met (or none is declared), 1 = a declared "
    "target is not met, 2 = a declared target is undetermined or does not apply, 3 = the calculation did not complete "
    "(PARTIAL: a budget stopped the beam trace; FAILED; CANCELLED; or a run a memory or run-file limit stopped); "
    "'project evaluate' reads the targets it "
    "assessed, and with no contract named takes execution when it assesses none; --status-contract execution: 0 = completed, 1 = anything else; "
    "--status-contract aggregate: 0 = PASS / WARNING / NOT_APPLICABLE, 1 = FAIL, 2 = UNKNOWN or a stopped run, 3 = "
    "OUT_OF_MODEL_SCOPE. A Project with configurations: the contract over every configuration that ran (requirements: 3 "
    "before 1 before 2; execution: 1 if any is incomplete; aggregate: the most severe, FAIL before OUT_OF_MODEL_SCOPE "
    "before UNKNOWN)."
)


class StatusContractError(ValueError):
    """``OS_OPTICS_STATUS_CONTRACT`` names no status contract."""


def add_status_contract_argument(parser: argparse.ArgumentParser, help: str = STATUS_CONTRACT_HELP) -> None:
    """Add ``--status-contract``; ``help``: its text where a command's exit codes cover other targets ('project evaluate')."""
    parser.add_argument("--status-contract", choices=list(STATUS_CONTRACTS), default=None, help=help)


def resolve_status_contract(args: argparse.Namespace, environ: Mapping[str, str]) -> str:
    """The contract a command line asks for: ``--status-contract``, else ``OS_OPTICS_STATUS_CONTRACT`` in ``environ`` (the
    process environment, which ``cli.py`` reads and passes), else the default."""

    return chosen_status_contract(args, environ) or DEFAULT_STATUS_CONTRACT


def chosen_status_contract(args: argparse.Namespace, environ: Mapping[str, str]) -> str | None:
    """The contract a command line names: ``--status-contract``, else ``OS_OPTICS_STATUS_CONTRACT`` in ``environ``; ``None``
    when neither names one ('project evaluate' then takes the contract its selection answers: ``selection_contract``)."""

    chosen = getattr(args, "status_contract", None)
    if chosen is not None:
        return chosen
    value = environ.get(STATUS_CONTRACT_ENV)
    if value is None or value.strip() == "":
        return None
    name = value.strip().lower()
    if name not in STATUS_CONTRACTS:
        raise StatusContractError(f"{STATUS_CONTRACT_ENV}={value!r} is not a status contract; use one of {', '.join(STATUS_CONTRACTS)}")
    return name


#: 'project evaluate' asked for the ``requirements`` contract (by ``--status-contract`` or ``OS_OPTICS_STATUS_CONTRACT``)
#: while it assesses none of the declared targets: refused before anything is computed, exit 64.
REQUIREMENTS_NOT_SELECTED = "REQUIREMENTS_NOT_SELECTED"


def selection_contract(assesses_targets: bool) -> str:
    """The contract of 'project evaluate' when the command line names none: ``requirements`` when the evaluation assesses
    targets, ``execution`` when it assesses none (its exit 0 then says that the calculation completed, nothing about a
    target)."""

    return "requirements" if assesses_targets else "execution"


def exit_code(contract: str, *, status: str, outcome: Any, requirements: Any = None) -> int:
    """The exit code of a run under ``contract``.

    ``status`` is the aggregate (``ValidationResult.status`` value), ``outcome`` the run's ``ResultOutcome`` (execution and
    requirements; None when the result carries none) and ``requirements`` an optional ``RequirementRecord`` that replaces
    ``outcome.requirements`` (the Local Validation record's, which also counts the published models' applicability).
    """

    if contract not in STATUS_CONTRACTS:
        raise ValueError(f"unknown status contract {contract!r}")
    if contract == "aggregate":
        return EXIT_CODES[status]
    execution = None if outcome is None else outcome.execution
    if execution is not None and execution.termination.value == "INVALID_INPUT":
        return EXIT_CODES["FAIL"]  # input that is not a valid request keeps its code under every contract
    completed = execution is not None and execution.outcome.value == "COMPLETED"
    if contract == "execution":
        return 0 if completed else 1
    if not completed:
        return EXECUTION_INCOMPLETE
    record = outcome.requirements if requirements is None else requirements
    if not record.declared:
        return REQUIREMENTS_MET
    if record.unmet:
        return REQUIREMENT_UNMET
    if record.indeterminate or record.not_applicable:
        return REQUIREMENT_UNDECIDED
    return REQUIREMENTS_MET


def run_exit_code(contract: str, run: Any) -> int:
    """The exit code of a local run (``LocalOpticalRun``): its Local Validation record's requirements when it has them."""

    requirements = None
    record = getattr(run, "local_validation", None)
    if record is not None and record.simulation.outcome is not None:
        requirements = record.simulation.outcome.requirements
    return exit_code(contract, status=run.status.value, outcome=run.result.outcome, requirements=requirements)


#: The exit code of ``project run`` when Ctrl-C stopped it (the shell's convention for SIGINT, 128 + 2).
INTERRUPTED = 130


def stopped_exit_code(contract: str, *, interrupted: bool = False) -> int:
    """The exit code of a run that stopped before it completed (``StoppedRun``): ``INTERRUPTED`` when Ctrl-C stopped it,
    else what the contract gives a calculation that did not complete (``requirements`` 3, ``execution`` 1, ``aggregate``
    2: no status was produced, the result is unknown)."""

    if contract not in STATUS_CONTRACTS:
        raise ValueError(f"unknown status contract {contract!r}")
    if interrupted:
        return INTERRUPTED
    return {"requirements": EXECUTION_INCOMPLETE, "execution": 1, "aggregate": EXIT_CODES["UNKNOWN"]}[contract]


#: Which per-configuration code decides a run of configurations, most decisive first (``configurations_exit_code``).
CONFIGURATIONS_PRECEDENCE = {
    "requirements": (EXECUTION_INCOMPLETE, REQUIREMENT_UNMET, REQUIREMENT_UNDECIDED, REQUIREMENTS_MET),
    "execution": (1, 0),
    "aggregate": (EXIT_CODES["FAIL"], EXIT_CODES["OUT_OF_MODEL_SCOPE"], EXIT_CODES["UNKNOWN"], EXIT_CODES["PASS"]),
}


def configurations_exit_code(contract: str, run: Any) -> int:
    """The exit code of a run of a Project with configurations (``ConfiguredProjectRun``; module docstring): each
    configuration's ``run_exit_code``, a configuration without a result counted as not completed, combined by
    ``CONFIGURATIONS_PRECEDENCE``."""

    if contract not in STATUS_CONTRACTS:
        raise ValueError(f"unknown status contract {contract!r}")
    failed = {"requirements": EXECUTION_INCOMPLETE, "execution": 1, "aggregate": EXIT_CODES["UNKNOWN"]}[contract]
    codes = {failed if c.run is None else run_exit_code(contract, c.run) for c in run.configurations}
    return next(code for code in CONFIGURATIONS_PRECEDENCE[contract] if code in codes)


#: ``assembly draw`` / ``assembly drawing --diagnostic``: the local assembly check did not pass and the sheet written is the
#: diagnostic sheet, not the delivery sheet (the code of a requirement that is not met; the files are written all the same).
DIAGNOSTIC_SHEET_WRITTEN = 2


#: An internal error of os-optics: an exception no command documents, raised on input that is known to be good. Its code
#: in the printed JSON and its exit code (the ``sysexits`` code of an internal software error), which no result uses.
INTERNAL_ERROR = "INTERNAL_ERROR"
INTERNAL_ERROR_EXIT = 70


def internal_error_document(exc: BaseException) -> dict[str, Any]:
    """What a command prints for an internal error: ``{"status": "ERROR", "error": {"code": "INTERNAL_ERROR", "type",
    "message"}}`` with the exception's class as ``type``; its traceback goes to standard error."""

    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return {"status": "ERROR", "error": {
        "code": INTERNAL_ERROR, "type": type(exc).__name__,
        "message": f"os-optics failed with an internal error ({detail}); the traceback is on standard error; report it "
                   "with the command you ran ('os-optics feedback submit --category PROBLEM')"}}


def internal_error_traceback(exc: BaseException) -> str:
    """The traceback of an internal error, as a command writes it to standard error."""

    return "".join(traceback.format_exception(exc))


__all__ = [
    "CONFIGURATIONS_PRECEDENCE", "DIAGNOSTIC_SHEET_WRITTEN", "INTERNAL_ERROR", "INTERNAL_ERROR_EXIT", "INTERRUPTED", "internal_error_document",
    "internal_error_traceback",
    "DEFAULT_STATUS_CONTRACT", "EXIT_CODES", "EXIT_CONTRACT_EPILOG", "STATUS_CONTRACTS", "STATUS_CONTRACT_ENV",
    "REQUIREMENTS_NOT_SELECTED", "STATUS_CONTRACT_HELP", "StatusContractError", "add_status_contract_argument",
    "chosen_status_contract", "configurations_exit_code", "exit_code", "resolve_status_contract", "run_exit_code",
    "selection_contract", "stopped_exit_code",
]
