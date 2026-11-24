"""The exit-code contract of ``project run``, ``workspace run`` and ``validate`` (``os_optics.cli_exit``): the acceptance
rows (``tests/test_scientific_result_acceptance.py``) x the three status contracts x the three commands.

Each row states its expected code under every contract, written from the row's documented answers (execution, target
partition, aggregate), never computed by the function under test:

* requirements (the default): 0 completed and every declared target met (or none declared), 1 a target unmet, 2 a target
  INDETERMINATE or NOT_APPLICABLE, 3 the calculation did not complete;
* execution: 0 completed, 1 partial or failed;
* aggregate: ``EXIT_CODES`` of ``ValidationResult.status``.

Input that is not a valid request keeps its code under every contract (1 for ``validate``, 65 for the Project commands).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from os_optics.cli import main
from os_optics.cli_exit import (
    DEFAULT_STATUS_CONTRACT,
    EXECUTION_INCOMPLETE,
    REQUIREMENT_UNDECIDED,
    REQUIREMENT_UNMET,
    REQUIREMENTS_MET,
    STATUS_CONTRACT_ENV,
    STATUS_CONTRACTS,
    StatusContractError,
    resolve_status_contract,
)
from os_optics.contracts import EngineConfiguration, ProjectSnapshot, ValidationRequest

from helpers import CFG
from local_helpers import catalog_document, plane, project, source, with_lossless_bulk
from test_closed_returns import faint_ring
from test_scientific_result_acceptance import (
    _clipped,
    _crossing_fixture,
    _laser_beam,
    _off_axis_plane,
    _plane_t,
    _target,
)
from test_tail_dark_planes_and_surface_coincidence import _ghost_scene

CONTRACTS = ("aggregate", "execution", "requirements")


@pytest.fixture(autouse=True)
def _no_contract_from_the_environment(monkeypatch):
    monkeypatch.delenv(STATUS_CONTRACT_ENV, raising=False)


# ------------------------------------------------------------------------------------------- the Engine rows (validate)
def _scene_of_clipped(radius_mm: float) -> ProjectSnapshot:
    """The clipped-capture row's scene (``_clipped`` runs it; here only its request is needed)."""
    from os_optics import builders as B

    stop = B.aperture_stop("A", position=(50.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=radius_mm)
    targets = (_target("captured", minimum=1e-4), _target("radius", metric="beam_radius_x_mm", unit="mm", minimum=0.0))
    return ProjectSnapshot(components=(_laser_beam(), stop), observations=(_plane_t(),), targets=targets)


def _high_na() -> ProjectSnapshot:
    laser = _laser_beam(405.0, 3e-4)
    targets = (_target("p", minimum=1e-4), _target("aoi", metric="aoi_deg", unit="deg", maximum=1.0),
               _target("radius", metric="beam_radius_x_mm", unit="mm", minimum=0.0))
    return ProjectSnapshot(components=(laser,), observations=(_plane_t(),), targets=targets)


def _lit(*targets) -> ProjectSnapshot:
    return ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(),), targets=tuple(targets))


def _dark_plane_row() -> ProjectSnapshot:
    targets = (_target("max", observation_id="D", maximum=1e-6), _target("min", observation_id="D", minimum=1e-6),
               _target("radius", observation_id="D", metric="beam_radius_x_mm", unit="mm", minimum=0.0),
               _target("wavelength", wavelength_nm=532.0, maximum=1.0), _target("path", maximum=1.0, parameters={"path_id": "nope"}))
    return ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(), _off_axis_plane()), targets=targets)


#: row -> (scene, numerical override, {contract: exit code}).
ENGINE_ROWS = {
    "no targets": (lambda: _lit(), None, {"aggregate": 0, "execution": 0, "requirements": 0}),
    "threshold met": (lambda: _lit(_target("p", minimum=5e-4)), None, {"aggregate": 0, "execution": 0, "requirements": 0}),
    "threshold changed: unmet": (lambda: _lit(_target("p", minimum=2e-3)), None, {"aggregate": 1, "execution": 0, "requirements": 1}),
    "mild clipping (WARNING, both met)": (lambda: _scene_of_clipped(0.8), None, {"aggregate": 0, "execution": 0, "requirements": 0}),
    "severe clipping (radius indeterminate)": (lambda: _scene_of_clipped(0.5), None,
                                               {"aggregate": 3, "execution": 0, "requirements": 2}),
    "high NA (radius indeterminate)": (_high_na, None, {"aggregate": 3, "execution": 0, "requirements": 2}),
    "budget exhausted": (lambda: ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(), _off_axis_plane()),
                                                 targets=(_target("lit", maximum=1.0), _target("dark", observation_id="D", maximum=1.0))),
                         {"max_trace_steps": 1}, {"aggregate": 2, "execution": 1, "requirements": 3}),
    "closed-path bound": (lambda: faint_ring(3), None, {"aggregate": 2, "execution": 0, "requirements": 2}),
    "known dark plane / empty selection": (_dark_plane_row, None, {"aggregate": 1, "execution": 0, "requirements": 1}),
    "met target beside a WARNING plane": (lambda: ProjectSnapshot(components=(_laser_beam(),), observations=(_off_axis_plane(),),
                                                                  targets=(_target("max", observation_id="D", maximum=1e-6),)),
                                          None, {"aggregate": 0, "execution": 0, "requirements": 0}),
    # (the plane is BOUNDED from 0 with status PASS; a maximum at or above the followed light's bound, about 8e-5 W, is
    # MET on that bound, a maximum under it is not decided)
    "plane only followed light reaches": (lambda: _ghost_scene((_target("max", maximum=1e-3),)), None,
                                          {"aggregate": 0, "execution": 0, "requirements": 0}),
    "plane only followed light reaches, maximum under its bound": (lambda: _ghost_scene((_target("max", maximum=1e-9),)), None,
                                                                   {"aggregate": 2, "execution": 0, "requirements": 2}),
}


def _request_file(tmp_path: Path, scene: ProjectSnapshot, numerical: dict | None) -> Path:
    cfg = CFG if numerical is None else CFG.model_copy(update=numerical)
    request = ValidationRequest(project=scene, configuration=EngineConfiguration(numerical=cfg))
    path = tmp_path / "request.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    return path


@pytest.mark.parametrize("row", list(ENGINE_ROWS))
def test_validate_exits_by_each_contract_on_every_acceptance_row(row: str, tmp_path, capsys) -> None:
    scene, numerical, expected = ENGINE_ROWS[row]
    path = _request_file(tmp_path, scene(), numerical)
    codes = {}
    for contract in CONTRACTS:
        codes[contract] = main(["validate", str(path), "--status-contract", contract])
        printed = json.loads(capsys.readouterr().out)
        assert "exit_contract" not in printed  # (validate prints the Engine result itself, unchanged)
    assert codes == expected
    assert main(["validate", str(path)]) == expected[DEFAULT_STATUS_CONTRACT]
    capsys.readouterr()


def test_validate_into_an_unwritable_output_is_a_usage_error_not_an_internal_error(tmp_path, capsys) -> None:
    """A mistyped -o path is the caller's: exit 64 with the reason on stderr, no INTERNAL_ERROR document, no traceback."""
    scene, numerical, _expected = ENGINE_ROWS[next(iter(ENGINE_ROWS))]
    path = _request_file(tmp_path, scene(), numerical)
    assert main(["validate", str(path), "-o", str(tmp_path / "missing" / "out.json")]) == 64
    printed = capsys.readouterr()
    assert "INTERNAL_ERROR" not in printed.out and "Traceback" not in printed.err and "cannot write the result" in printed.err


def test_a_target_on_checks_the_request_turns_off_is_undetermined(tmp_path, capsys) -> None:
    """``run_layout`` false and a target on a layout metric: the target has a result that says the layout stage did not run
    (``UNKNOWN``), and the requirement record lists it as undetermined. ``requirements`` exits 2, ``execution`` 0 (the
    calculation completed) and ``aggregate`` 2 (``UNKNOWN``). The result states no evaluation scope: it is a complete
    evaluation of the request."""
    from os_optics import builders as B
    from os_optics.contracts import EngineeringComponentSnapshot, Target

    scene = ProjectSnapshot(
        components=(EngineeringComponentSnapshot(instance_id="A", envelope=B.rect_envelope(10, 10, 20)),
                    EngineeringComponentSnapshot(instance_id="B", pose=B.Pose3D(position_mm=(25.0, 0.0, 0.0)),
                                                 envelope=B.rect_envelope(10, 10, 20))),
        targets=(Target(target_id="clear", metric="layout_min_clearance_mm", minimum=1.0),))
    request = ValidationRequest(project=scene, configuration=EngineConfiguration(run_layout=False, numerical=CFG))
    path = tmp_path / "request.json"
    path.write_text(request.model_dump_json(), encoding="utf-8")
    codes = {}
    for contract in CONTRACTS:
        codes[contract] = main(["validate", str(path), "--status-contract", contract])
        printed = json.loads(capsys.readouterr().out)
    assert codes == {"requirements": 2, "execution": 0, "aggregate": 2}
    assert "evaluation" not in printed and printed["status"] == "UNKNOWN"
    (target,) = printed["targets"]
    assert (target["target_id"], target["status"], target["assessment"], target["issue_codes"], target["message"]) == (
        "clear", "UNKNOWN", "INDETERMINATE", ["INVALID_INPUT"], "layout stage did not run")
    assert printed["outcome"]["requirements"] == {"declared": True, "met": [], "unmet": [], "indeterminate": ["clear"],
                                                  "not_applicable": []}


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_project_evaluate_exits_on_the_requirement_record_alone(tmp_path, capsys, lossless) -> None:
    """A selection that leaves out the plane bounds, and targets on planes the beam misses: each target has an undetermined
    result (``CAPABILITY_NOT_COMPUTED``) listed in ``outcome.requirements.indeterminate``, and the exit code is read from
    that record -- 2 under ``requirements``, 0 under ``execution``, 2 under ``aggregate`` (``UNKNOWN``). The complete
    evaluation decides them when fixture bulk loss is established. Unknown bulk loss keeps the missed planes undecided."""
    from os_optics.local import json_api

    document = json_api.example_project_document()["project"]
    for observation in document["observations"]:
        observation["pose"]["position_mm"][1] += 200.0  # (beside the beam)
    path = tmp_path / "missed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    catalog = catalog_document()
    given = _write(tmp_path / "catalog.json", with_lossless_bulk(catalog) if lossless else catalog)
    declared = [t["target_id"] for t in document["targets"]]

    def codes(*selection: str) -> tuple[dict[str, int], dict]:
        found, reply = {}, {}
        for contract in CONTRACTS:
            found[contract] = main(["project", "evaluate", str(path), "--catalog", str(given), *selection, "--status-contract", contract])
            reply = json.loads(capsys.readouterr().out)
        return found, reply

    narrowed, reply = codes("--outputs", "readings,requirements")
    assert narrowed == {"requirements": 2, "execution": 0, "aggregate": 2}
    assert reply["evaluation"]["targets_not_evaluated"] == [] and "plane_bounds" not in reply["evaluation"]["executed"]
    assert [(t["target_id"], t["status"], t["assessment"], t["issue_codes"]) for t in reply["targets"]] == [
        (target_id, "UNKNOWN", "INDETERMINATE", ["CAPABILITY_NOT_COMPUTED"]) for target_id in declared]
    assert reply["outcome"]["requirements"] == {"declared": True, "met": [], "unmet": [], "indeterminate": declared,
                                                "not_applicable": []}
    assert all(t["message"].endswith("was not computed: evaluate with plane_bounds") for t in reply["targets"])
    decided, reply = codes("--outputs", "readings,requirements,plane_bounds")
    expected = 1 if lossless else 2
    assert decided == {"requirements": expected, "execution": 0, "aggregate": expected}
    requirements = reply["outcome"]["requirements"]
    assert requirements["unmet"] == (declared if lossless else [])
    assert requirements["indeterminate"] == ([] if lossless else declared)


def test_the_clipped_rows_are_the_acceptance_rows() -> None:
    """The request of the clipping row here is the acceptance test's own scene (same answers)."""
    from helpers import run

    for radius in (0.8, 0.5):
        assert run(_scene_of_clipped(radius)).model_dump(mode="json") == _clipped(radius).model_dump(mode="json")


def test_validate_keeps_the_invalid_input_code_under_every_contract(tmp_path, capsys) -> None:
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{not json", encoding="utf-8")
    for contract in CONTRACTS:
        assert main(["validate", str(garbled), "--status-contract", contract]) == 1
        assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "INVALID_INPUT"
    assert main(["validate", str(tmp_path / "missing.json")]) == 64
    capsys.readouterr()


# ------------------------------------------------------------------------------------ the Project rows (project / workspace)
SCREEN = (100.0, 0.0, 0.0)


def _power(target_id: str, **limits) -> dict:
    return {"target_id": target_id, "metric": "power_W", "observation_id": "screen", "unit": "W", **limits}


def _local(*targets: dict, budget: dict | None = None) -> dict:
    document = project([source()], [plane("screen", SCREEN, radius_mm=40.0)], list(targets))
    if budget is not None:
        document["compute_budget"] = budget
    return document


def _not_applicable() -> dict:
    """A two-beam metric on beams of different wavelengths."""
    return project([source("A", wavelength_nm=532.0), source("B", position=(-200.0, 0.3, 0.0), wavelength_nm=1064.0)],
                   [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)],
                   [{"target_id": "overlap", "metric": "spatial_overlap_ratio", "observation_id": "det", "source_id": "A",
                     "source_id_b": "B", "minimum": 0.0}])


#: row -> (Project document, {contract: exit code}).
PROJECT_ROWS = {
    "no targets": (lambda: _local(), {"aggregate": 0, "execution": 0, "requirements": 0}),
    "target met": (lambda: _local(_power("p", minimum=1e-6)), {"aggregate": 0, "execution": 0, "requirements": 0}),
    "target unmet": (lambda: _local(_power("p", minimum=1.0)), {"aggregate": 1, "execution": 0, "requirements": 1}),
    "target out of the model's scope": (lambda: _local({"target_id": "m", "metric": "mtf", "observation_id": "screen", "minimum": 0.5}),
                                        {"aggregate": 3, "execution": 0, "requirements": 2}),
    "target not applicable": (_not_applicable, {"aggregate": 0, "execution": 0, "requirements": 2}),
    "crossing-plane defect": (lambda: _crossing_fixture(("B", "A")), {"aggregate": 1, "execution": 0, "requirements": 1}),
    # (the aggregate reads this incomplete run as a WARNING: exit 0; execution and requirements say it did not complete)
    "budget exhausted": (lambda: _local(_power("p", minimum=1e-6), budget={"max_trace_steps": 1}),
                         {"aggregate": 0, "execution": 1, "requirements": 3}),
}


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _catalog(tmp_path: Path) -> Path:
    return _write(tmp_path / "catalog.json", catalog_document())


@pytest.mark.parametrize("row", list(PROJECT_ROWS))
def test_project_run_exits_by_each_contract_on_every_acceptance_row(row: str, tmp_path, capsys) -> None:
    document, expected = PROJECT_ROWS[row]
    path = _write(tmp_path / "project.json", document())
    catalog = _catalog(tmp_path)
    codes = {}
    for contract in CONTRACTS:
        codes[contract] = main(["project", "run", str(path), "--catalog", str(catalog), "--status-contract", contract])
        assert json.loads(capsys.readouterr().out)["exit_contract"] == contract
    assert codes == expected
    assert main(["project", "run", str(path), "--catalog", str(catalog)]) == expected["requirements"]
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "requirements"


@pytest.mark.parametrize("row", list(PROJECT_ROWS))
def test_workspace_run_exits_by_each_contract_on_every_acceptance_row(row: str, tmp_path, capsys) -> None:
    document, expected = PROJECT_ROWS[row]
    path = _write(tmp_path / "project.json", document())
    codes = {}
    for contract in CONTRACTS:
        out = tmp_path / f"ws-{contract}.json"
        codes[contract] = main(["workspace", "run", str(path), "-o", str(out), "--status-contract", contract])
        reply = json.loads(capsys.readouterr().out)
        assert reply["exit_contract"] == contract and reply["outcome"]["execution"]["outcome"] in ("COMPLETED", "PARTIAL")
    assert codes == expected
    assert main(["workspace", "run", str(path), "-o", str(tmp_path / "ws.json")]) == expected["requirements"]
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "requirements"


def test_a_project_that_cannot_run_keeps_its_code_under_every_contract(tmp_path, capsys) -> None:
    bad = _write(tmp_path / "bad.json", {**_local(), "components": []})
    for contract in CONTRACTS:
        assert main(["project", "run", str(bad), "--status-contract", contract]) == 65
        assert main(["workspace", "run", str(bad), "-o", str(tmp_path / "ws.json"), "--status-contract", contract]) == 65
    capsys.readouterr()


def test_a_project_without_targets_whose_engine_run_fails_keeps_the_input_code_and_the_engine_diagnosis(tmp_path, capsys) -> None:
    # A Project with no targets passes the "every target evaluated" check (0 of 0) even when the Engine rejected its input;
    # the failed execution itself now stops the run with the Engine's own diagnosis (it used to surface later as an
    # OPTICAL_VIEW_BINDING_FAILED of the 3D view). The case: an observation plane whose medium names a declared model
    # material, which the Engine knows only by its Project id (project:CALCITE; UNKNOWN_MATERIAL, termination INVALID_INPUT).
    # (A declared crystal there is refused before the Engine: test_project_crystals.py.)
    import copy

    from test_project_crystals import CALCITE, WAVELENGTH, displacer, scene

    glass = copy.deepcopy(CALCITE)
    del glass["birefringence"]
    model = displacer()
    del model["optic_axes"]
    document = scene(source(wavelength_nm=WAVELENGTH), ("BD", model, 0.0))
    document["model_materials"] = [glass]
    document["observations"][0]["medium"] = "CALCITE"
    assert not document.get("targets")
    path = _write(tmp_path / "project.json", document)
    for contract in CONTRACTS:
        assert main(["project", "run", str(path), "--status-contract", contract]) == 65
        problems = json.loads(capsys.readouterr().out)["problems"]
        assert [p["code"] for p in problems] == ["ENGINE_RUN_INCOMPLETE"], problems
        assert "(UNKNOWN_MATERIAL)" in problems[0]["message"]
        assert main(["workspace", "run", str(path), "-o", str(tmp_path / "ws.json"), "--status-contract", contract]) == 65
        capsys.readouterr()


# ------------------------------------------------------------------------------------------------------ the environment
def test_the_environment_chooses_the_contract_and_the_flag_overrides_it(tmp_path, capsys, monkeypatch) -> None:
    path = _write(tmp_path / "project.json", _local(_power("p", minimum=1.0)))  # (an unmet target: 1 / 0 / 1)
    catalog = _catalog(tmp_path)
    monkeypatch.setenv(STATUS_CONTRACT_ENV, "execution")
    assert main(["project", "run", str(path), "--catalog", str(catalog)]) == 0
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "execution"
    assert main(["project", "run", str(path), "--catalog", str(catalog), "--status-contract", "requirements"]) == 1
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "requirements"
    assert main(["workspace", "run", str(path), "-o", str(tmp_path / "ws.json")]) == 0
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "execution"
    monkeypatch.setenv(STATUS_CONTRACT_ENV, " Aggregate ")
    assert main(["validate", str(_request_file(tmp_path, _lit(_target("p", minimum=2e-3)), None))]) == 1
    capsys.readouterr()


def test_an_environment_naming_no_contract_is_a_usage_error_and_runs_nothing(tmp_path, capsys, monkeypatch) -> None:
    path = _write(tmp_path / "project.json", _local())
    catalog = _catalog(tmp_path)
    out = tmp_path / "run"
    assert main(["project", "run", str(path), "--catalog", str(catalog), "--out-dir", str(out)]) == 0
    assert (out / "manifest.json").is_file()
    monkeypatch.setenv(STATUS_CONTRACT_ENV, "strict")
    earlier = {p.name: p.read_bytes() for p in out.iterdir()}
    assert main(["project", "run", str(path), "--catalog", str(catalog), "--out-dir", str(out)]) == 64
    assert {p.name: p.read_bytes() for p in out.iterdir()} == earlier  # (nothing ran: the earlier run is as it was)
    assert main(["workspace", "run", str(path), "-o", str(tmp_path / "ws.json")]) == 64
    assert not (tmp_path / "ws.json").exists()
    assert main(["validate", str(_request_file(tmp_path, _lit(), None))]) == 64
    captured = capsys.readouterr()
    assert captured.err.count(f"{STATUS_CONTRACT_ENV}='strict' is not a status contract") == 3
    # an explicit flag needs no environment
    assert main(["project", "run", str(path), "--catalog", str(catalog), "--status-contract", "execution"]) == 0
    capsys.readouterr()


def test_the_contract_resolution_order() -> None:
    from types import SimpleNamespace

    assert STATUS_CONTRACTS == CONTRACTS and DEFAULT_STATUS_CONTRACT == "requirements"
    assert resolve_status_contract(SimpleNamespace(status_contract=None), {}) == "requirements"
    assert resolve_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: ""}) == "requirements"
    assert resolve_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: "AGGREGATE"}) == "aggregate"
    assert resolve_status_contract(SimpleNamespace(status_contract="execution"), {STATUS_CONTRACT_ENV: "aggregate"}) == "execution"
    with pytest.raises(StatusContractError):
        resolve_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: "pass"})
    # ('project evaluate' tells a contract the command line names from none: with none, its selection decides)
    from os_optics.cli_exit import chosen_status_contract, selection_contract

    assert chosen_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: " "}) is None
    assert chosen_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: "Requirements"}) == "requirements"
    assert chosen_status_contract(SimpleNamespace(status_contract="aggregate"), {}) == "aggregate"
    with pytest.raises(StatusContractError):
        chosen_status_contract(SimpleNamespace(status_contract=None), {STATUS_CONTRACT_ENV: "pass"})
    assert selection_contract(True) == "requirements" and selection_contract(False) == "execution"


def test_the_help_states_the_default_and_every_contract(capsys) -> None:
    for argv in (["project", "run", "--help"], ["workspace", "run", "--help"], ["validate", "--help"]):
        with pytest.raises(SystemExit):
            main(argv)
        text = " ".join(capsys.readouterr().out.split())
        assert "--status-contract {aggregate,execution,requirements}" in text, argv
        assert "default: requirements" in text and STATUS_CONTRACT_ENV in text, argv
        assert "exit_contract" in text, argv


#: What each ``requirements`` exit code reports, as the help texts word it (the code is read from ``cli_exit``).
REQUIREMENT_PHRASES = {
    r"every declared target (?:is )?met": REQUIREMENTS_MET,
    r"(?:a declared target is not met|one unmet)": REQUIREMENT_UNMET,
    r"(?:a declared target is undetermined|one undetermined)": REQUIREMENT_UNDECIDED,
    r"the calculation did not complete": EXECUTION_INCOMPLETE,
}


def _help_text(argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit):
        main([*argv, "--help"])
    return " ".join(capsys.readouterr().out.split())


def test_every_help_statement_of_the_run_exit_codes_follows_the_status_contract(capsys) -> None:
    # 'project --help' said "The exit code is the run status ... a FAIL run (exit 1)": under the default contract a FAIL
    # run without an unmet target exits 0, and an OUT_OF_MODEL_SCOPE one 2 or 0, not 3. Every sentence of the run commands'
    # help that states an exit code names --status-contract, and every requirements code it states is cli_exit's.
    stated = 0
    for argv in ([], ["project"], ["project", "run"], ["workspace"], ["workspace", "run"], ["validate"]):
        text = _help_text(argv, capsys)
        assert "exit code is the run status" not in text and "(exit 1)" not in text, argv
        for sentence in re.split(r"(?<=\.) (?=[A-Z'])", text):
            if "--status-contract" not in sentence:
                # a sentence stating exit codes without the contract is about another command's own codes
                # ('workspace apply', 'assembly validate', usage errors), never about a run's
                if re.search(r"\bexit code", sentence, re.IGNORECASE):
                    assert not re.search(r"\b(?:project run|workspace run|validate)\b|\brun (?:status|exit)", sentence), (argv, sentence)
                continue
            for phrase, code in REQUIREMENT_PHRASES.items():
                for match in re.finditer(rf"\b(\d) (?:= )?{phrase}", sentence):
                    assert int(match.group(1)) == code, (argv, match.group(0))
                    stated += 1
    assert stated >= 8  # (the 'project' description and the epilog both state all four codes)


def test_the_summary_of_a_run_names_its_contract_in_every_output(tmp_path, capsys) -> None:
    path = _write(tmp_path / "project.json", _local(_power("p", minimum=1e-6)))
    out = tmp_path / "ws.json"
    assert main(["workspace", "run", str(path), "-o", str(out), "--json", "--status-contract", "execution"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == reply["summary"]["exit_contract"] == "execution"
    assert [t["assessment"] for t in reply["targets"]] == ["MET"]
