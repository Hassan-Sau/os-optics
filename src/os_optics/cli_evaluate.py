"""``os-optics project evaluate PROJECT [--outputs A,B] [--exclude C] [--targets T1,T2] [--scene-out F] [--glb-out F]``: a
selective evaluation of a Project (``json_api.evaluate_project``); ``os-optics project evaluate --describe [ID]``: the
capabilities it is made of (``json_api.evaluation_capabilities``)."""

from __future__ import annotations

import argparse
from pathlib import Path

USAGE_ERROR = 64
LOCAL_INPUT_ERROR = 65
EVALUATE_DESCRIPTION = (
    "Evaluate selected quantities of a Project with the real Engine on this device, computing what they require and "
    "nothing else, and print the reply as JSON. '--outputs' names capabilities and presets (default: readings). 'readings' "
    "computes each beam's arrival at each observation plane -- position, direction, size, waist, phase, polarization and "
    "traced power -- and the model-scope findings that say whether a number applies; it leaves out the probe traces of light "
    "the beam trace does not follow, the power ledger, the plane bounds (so a plane's quantity_state is NOT_COMPUTED, its "
    "untraced_power_W null and its power the traced power), the surface exposures, the beam pairs, the optical-solid check, "
    "the record and the 3D view. 'complete' computes everything 'project run' computes. Name 'requirements' (or "
    "'--targets') to evaluate the declared targets: each target adds the capabilities its metric reads. The plane readings "
    "and the requirements always come with the model-scope findings. '--exclude' names capabilities that must not run; a "
    "selection that needs one is refused with the excluded capability, what requires it and the two ways out. The reply "
    "states the scope ('evaluation': requested, executed, added with what required each, not computed with what each "
    "produces, the declared targets not selected), the requirements (the assessed targets by assessment and the declared "
    "targets not assessed), the status of what was computed, the outcome, the evaluated targets, "
    "each plane's compact readings (per arriving beam: position, beam radius, waist radius and distance, Rayleigh range, "
    "traced and detected power; the plane's traced power and quantity state; the coherent sums where two or more beams "
    "arrive), the issues, the model-scope findings and the identity. An arrival's direction, phase and polarization are not "
    "in the reply: read them in Python from Evaluation.results() (arrivals(), monitor(id)), or with 'results' from a run "
    "directory 'project run --out-dir' writes. 'view' builds the 3D Optical Simulation View of what the evaluation traced, "
    "without the record, the plane bounds or the targets it does not select: the reply carries the scene as optical_scene "
    "(the scene a run directory holds, its engine_run naming this evaluation's result); '--scene-out FILE' and '--glb-out "
    "FILE' compute the view beside the outputs and write the scene JSON and its glTF binary there instead of printing the "
    "scene; '--pupil-rays' sets which pupil rays it draws. Nothing is written but the scene files these options name: a run "
    "directory and a Local Validation record come from 'project run'. A Project that declares configurations is evaluated one configuration at a time: '--configuration ID' "
    "evaluates that configuration as its own Project (without it the Project is refused, CONFIGURATION_NOT_SELECTED: no "
    "configuration is chosen for you); 'project run PROJECT --configuration ID --out-dir DIR' runs one completely. '--describe' lists the "
    "capabilities, presets, analyses (with the command that runs each) and the run "
    "directory; '--describe ID' describes one (what it requires and produces, the target metrics that read it, its numerical "
    "controls, phases and method; for a preset: its summary, the capabilities it runs and those it leaves out)."
)
EVALUATE_EPILOG = (
    "exit codes follow --status-contract over the targets the evaluation assessed: the targets named by --targets, every "
    "declared target when requirements is among the outputs, none otherwise. With no contract named (no --status-contract, "
    "no OS_OPTICS_STATUS_CONTRACT) the contract follows the selection: requirements when the evaluation assesses targets, "
    "execution when it assesses none (the default 'readings'), so its exit 0 says that the calculation completed and "
    "nothing about a target. requirements: 0 every assessed target met (a Project that declares no targets: the "
    "calculation completed); 1 one unmet; 2 one undetermined or not applicable and none unmet, among them a target a "
    "capability outside the evaluation would decide, issue code CAPABILITY_NOT_COMPUTED; 3 the calculation did not "
    "complete (PARTIAL: a budget stopped the beam trace; FAILED; CANCELLED); execution: 0 completed, 1 anything else; "
    "aggregate, of what was computed: 0 PASS / WARNING / NOT_APPLICABLE, 1 FAIL, 2 UNKNOWN, 3 OUT_OF_MODEL_SCOPE. "
    "requirements named for an evaluation that assesses none of the declared targets is refused before anything is "
    "computed (64, REQUIREMENTS_NOT_SELECTED): add requirements to --outputs, name targets with --targets, or use "
    "--status-contract execution. The declared targets the evaluation did not assess are listed in the reply's "
    "requirements.not_assessed (and evaluation.targets_not_evaluated) and do not affect the exit code; when it assessed "
    "none, status is NOT_APPLICABLE where it would read PASS. 64 = unreadable file, an empty --outputs or "
    "--targets, a selection that cannot be planned, REQUIREMENTS_NOT_SELECTED, or a scene file that cannot be "
    "written (nothing else is printed); 65 = a Project that cannot be "
    "simulated, among them one that declares configurations when no --configuration is given; 70 = an internal error of "
    "os-optics (JSON with code INTERNAL_ERROR and the exception type, the traceback on stderr); 130 = Ctrl-C"
)
#: The --status-contract help of 'project evaluate': over the targets the evaluation assessed, not every declared one.
EVALUATE_STATUS_CONTRACT_HELP = (
    "what the exit code reports (default: the environment variable OS_OPTICS_STATUS_CONTRACT, else requirements when the "
    "evaluation assesses targets and execution when it assesses none): requirements -- over the targets this evaluation "
    "assessed: 0 the calculation completed and every assessed target is met (a Project that declares no targets: the "
    "calculation completed), 1 an assessed target is not met, 2 one is undetermined or does not apply to the light it "
    "names and none is unmet, 3 the calculation did not complete (PARTIAL: a budget stopped the beam trace; FAILED; "
    "CANCELLED); refused before anything is computed when the evaluation assesses none of the declared targets (64, "
    "REQUIREMENTS_NOT_SELECTED); declared targets that were not assessed are listed in requirements.not_assessed and "
    "evaluation.targets_not_evaluated and do not affect the exit code; execution -- 0 the calculation completed, 1 "
    "anything else; aggregate -- the worst-of diagnostic severity of what was computed: 0 PASS / WARNING / "
    "NOT_APPLICABLE, 1 FAIL, 2 UNKNOWN, 3 OUT_OF_MODEL_SCOPE. The JSON reply names the contract used as exit_contract."
)

def _ids(text: str | None) -> list[str] | None:
    return None if text is None else [part.strip() for part in text.split(",") if part.strip()]


def register_project(project_commands: argparse._SubParsersAction) -> None:
    """Add ``os-optics project evaluate``."""

    from .cli_exit import add_status_contract_argument

    evaluate = project_commands.add_parser("evaluate", help="evaluate selected quantities of a Project (readings, complete, or capability ids) "
                                                            "and print what was computed and what was not",
                                           description=EVALUATE_DESCRIPTION, epilog=EVALUATE_EPILOG)
    evaluate.add_argument("project", nargs="?", help="path to a Project JSON file")
    evaluate.add_argument("--outputs", metavar="A,B", help="capability ids and presets to compute, comma-separated (default: readings)")
    evaluate.add_argument("--exclude", metavar="C", help="capability ids that must not run, comma-separated")
    evaluate.add_argument("--targets", metavar="T1,T2", help="the declared targets to evaluate, comma-separated (evaluates the requirements)")
    evaluate.add_argument("--describe", nargs="?", const="", metavar="ID",
                          help="print the capabilities, presets, analyses and the run directory export (or one in full) instead of "
                               "evaluating")
    evaluate.add_argument("--configuration", dest="configuration_id", metavar="ID",
                          help="the configuration to evaluate, of a Project that declares configurations")
    evaluate.add_argument("--catalog", help="use this public optical catalog JSON instead of the release the Project pins / the active one")
    evaluate.add_argument("--scene-out", metavar="FILE", help="compute the 3D view beside the outputs and write its scene JSON "
                                                              "(optical_scene.json of a run directory) to FILE instead of printing it")
    evaluate.add_argument("--glb-out", metavar="FILE", help="compute the 3D view beside the outputs and write it as a glTF 2.0 binary "
                                                            "(optical_scene.glb of a run directory) to FILE")
    evaluate.add_argument("--pupil-rays", choices=["OUTER_RING", "ALL", "NONE"], default="OUTER_RING",
                          help="which Engine pupil rays the 3D view draws (display only; default OUTER_RING)")
    add_status_contract_argument(evaluate, EVALUATE_STATUS_CONTRACT_HELP)
    from .cli import PROGRESS_MODES

    evaluate.add_argument("--progress", choices=list(PROGRESS_MODES), default="auto",
                          help="progress lines on stderr while the evaluation goes through the phases it enters (auto: text lines when "
                               "stderr is a terminal; json: one JSON object per line; none)")
    evaluate.set_defaults(handler=_evaluate, contract_by_selection=True)


def _assesses_targets(args: argparse.Namespace, project: bytes, catalog: bytes | None) -> bool | None:
    """Whether the evaluation the command line selects assesses targets: ``--targets``, or the plan of the selection
    against the request the evaluation runs -- the Project lowered as ``evaluate_project`` lowers it, since what a
    capability requires can depend on the Project (the exposures need the untraced light only on a rated element);
    ``None`` when the Project cannot be lowered or the selection cannot be planned (``evaluate_project`` refuses those)."""

    from .evaluation.capabilities import LOCAL, REQUIREMENTS
    from .evaluation.plan import EvaluationPlanError, plan_evaluation
    from .local.configurations import configured_project
    from .local.errors import LocalRunError
    from .local.evaluate import EvaluationSession

    if args.targets is not None:
        return True
    try:
        document = configured_project(project, args.configuration_id)
        lowered = EvaluationSession(catalog=catalog).lowering(document, check_solids=False)
        return plan_evaluation(_ids(args.outputs) or ["readings"], exclude=_ids(args.exclude) or (), request=lowered.request,
                               level=LOCAL, what="outputs").runs(REQUIREMENTS)
    except (LocalRunError, EvaluationPlanError):
        return None


def _declared_targets(project: bytes, configuration_id: str | None) -> list[str]:
    """The target ids the evaluated Project declares; empty when it cannot be read or is a Project with configurations
    and no configuration is given (``evaluate_project`` refuses those with their own codes)."""

    from .local.configurations import configured_project
    from .local.errors import LocalRunError

    try:
        document = configured_project(project, configuration_id)
    except LocalRunError:
        return []
    if configuration_id is None and getattr(document, "configurations", None):
        return []
    return [t.target_id for t in document.targets]


def _requirements_not_selected(declared: list[str], named_by: str) -> dict:
    from .cli_exit import REQUIREMENTS_NOT_SELECTED

    message = (f"{named_by} asks the exit code to report the targets this evaluation assesses, and it assesses none of the "
               f"{len(declared)} the Project declares ({', '.join(declared)}): add requirements to --outputs (all of them) or "
               "name them with --targets, or use --status-contract execution for an exit code of the calculation alone")
    return {"ok": False, "error": {"code": REQUIREMENTS_NOT_SELECTED, "message": message, "problems": [
        {"code": REQUIREMENTS_NOT_SELECTED, "message": message, "context": {"targets": declared, "contract": "requirements"}}]}}


def _evaluate(args: argparse.Namespace) -> int:
    from .cli import _json_out, _named_status_contract, _progress_printer, _read
    from .cli_exit import STATUS_CONTRACT_ENV, exit_code, selection_contract, stopped_exit_code
    from .contracts.outcome import ResultOutcome
    from .local import json_api
    from .run_control import monitored
    from .run_monitor import RunMonitor

    named, chosen = _named_status_contract(args)
    if not named:
        return USAGE_ERROR
    if args.describe is not None:
        reply = json_api.evaluation_capabilities(args.describe or None)
        _json_out({k: v for k, v in reply.items() if k != "ok"} if reply["ok"] else reply)
        return 0 if reply["ok"] else USAGE_ERROR
    if args.project is None:
        _json_out({"ok": False, "error": {"code": "INPUT_INVALID", "message": "give a Project file, or --describe"}})
        return USAGE_ERROR
    for option in ("outputs", "targets"):
        if getattr(args, option) is not None and not _ids(getattr(args, option)):
            _json_out({"ok": False, "error": {"code": "INPUT_INVALID", "message": f"--{option} names no id: give one or more "
                                                                                  f"comma-separated ids, or leave --{option} out"}})
            return USAGE_ERROR
    project = _read(args.project, "project")
    catalog = _read(args.catalog, "catalog") if args.catalog else None
    if project is None or (args.catalog and catalog is None):
        return USAGE_ERROR
    assesses = _assesses_targets(args, project, catalog)
    contract = chosen or selection_contract(assesses is not False)
    if chosen == "requirements" and assesses is False:
        # (a requirements exit code over no assessed target would be a verdict on nothing: refused before computing)
        declared = _declared_targets(project, args.configuration_id)
        if declared:
            named_by = "--status-contract requirements" if args.status_contract else f"{STATUS_CONTRACT_ENV}=requirements"
            _json_out(_requirements_not_selected(declared, named_by))
            return USAGE_ERROR
    outputs = _ids(args.outputs)
    scene_files = [(name, Path(path)) for name, path in (("json", args.scene_out), ("glb", args.glb_out)) if path]
    if scene_files and "view" not in (outputs or ()):
        outputs = [*(outputs or ["readings"]), "view"]  # (the files need the view: it is computed beside the outputs)
    monitor = RunMonitor(_progress_printer(args.progress), preflight=False)
    with monitored(monitor):
        reply = json_api.evaluate_project(project, outputs, _ids(args.exclude), _ids(args.targets),
                                          {"pupil_ray_selection": args.pupil_rays}, catalog=catalog,
                                          configuration_id=args.configuration_id)
    if not reply["ok"]:
        _json_out(reply)
        if "stopped" in reply:
            return stopped_exit_code(contract, interrupted=bool(monitor.cancelled))
        return LOCAL_INPUT_ERROR if reply["error"]["code"] == json_api.LOCAL_RUN_ERROR else USAGE_ERROR
    if scene_files and (problem := _write_scene(reply["optical_scene"], scene_files)) is not None:
        _json_out({"ok": False, "error": {"code": "INPUT_INVALID", "message": problem}})
        return USAGE_ERROR
    if chosen is None and assesses is None:
        # (the selection could not be planned before computing: the contract follows what the evaluation ran)
        contract = selection_contract("requirements" in reply["evaluation"]["executed"])
    document = {k: v for k, v in reply.items() if k != "ok" and not (scene_files and k == "optical_scene")}
    _json_out({**document, "exit_contract": contract})
    outcome = None if reply["outcome"] is None else ResultOutcome.model_validate(reply["outcome"])
    return exit_code(contract, status=reply["status"], outcome=outcome)


def _write_scene(document: dict, files: list[tuple[str, Path]]) -> str | None:
    """Write the scene ``document`` as each named file (``json``: the scene JSON, ``glb``: its glTF binary) -- the bytes a run
    directory holds for that scene; the reason when a file cannot be written, else ``None``."""

    from ._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
    from .optical_view.artifacts import scene_glb_bytes, scene_json_bytes

    scene = OpticalSimulationSceneV1.model_validate(document)
    for kind, path in files:
        try:
            path.write_bytes(scene_json_bytes(scene) if kind == "json" else scene_glb_bytes(scene))
        except OSError as exc:
            return (f"cannot write the scene {'JSON' if kind == 'json' else 'GLB'} file {str(path)!r}: {exc.strerror or exc}; "
                    "give a file in a directory that exists and can be written")
    return None
