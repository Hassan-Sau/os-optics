"""``os-optics project evaluate``: a thin wrapper of ``json_api.evaluate_project`` whose exit code follows the status
contract over the targets the evaluation assessed, and ``--describe`` of the capabilities."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from os_optics.cli import main
from os_optics.local import json_api

from local_helpers import ALONG_X, WINDOW, catalog_document, element, plane, project, source, with_lossless_bulk

CONTRACTS = ("requirements", "execution", "aggregate")


def _write(tmp_path: Path, name: str, document: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _focus() -> dict:
    return json_api.example_project_document()["project"]


def _codes(capsys, argv: list[str]) -> tuple[dict[str, int], dict]:
    codes, reply = {}, {}
    for contract in CONTRACTS:
        codes[contract] = main([*argv, "--status-contract", contract])
        reply = json.loads(capsys.readouterr().out)
        assert reply["exit_contract"] == contract
    return codes, reply


def _unmet() -> dict:
    """The example Project with ``spot_radius`` not met (its maximum divided by 1000) and ``focus_position`` met."""
    document = _focus()
    spot = next(t for t in document["targets"] if t["target_id"] == "spot_radius")
    spot["maximum"] = spot["maximum"] / 1000.0
    return document


def _not_assessed(reply: dict) -> list[str]:
    return [row["target_id"] for row in reply["requirements"]["not_assessed"]]


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_exit_codes_follow_the_contract_over_the_evaluated_targets(tmp_path, capsys, monkeypatch, lossless):
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
    source_catalog = catalog_document()
    catalog = _write(tmp_path, "catalog.json", with_lossless_bulk(source_catalog) if lossless else source_catalog)
    given = ["--catalog", str(catalog)]
    aggregate = 0 if lossless else 2
    unassessed = "NOT_APPLICABLE" if lossless else "UNKNOWN"
    present = "PASS" if lossless else "UNKNOWN"
    path = _write(tmp_path, "focus.json", _focus())
    met, reply = _codes(capsys, ["project", "evaluate", str(path), *given, "--outputs", "readings,requirements"])
    assert met == {"requirements": 0, "execution": 0, "aggregate": aggregate} and len(reply["targets"]) == 2
    assert _not_assessed(reply) == []
    unmet_path = _write(tmp_path, "unmet.json", _unmet())
    # (nothing assessed: the default contract is execution -- its 0 says the calculation completed, nothing about a target
    # -- and requirements named for it is refused, never a 0 over an empty set; one target of the two is unmet)
    for contract, expected in (("execution", 0), ("aggregate", aggregate), ("requirements", 64)):
        assert main(["project", "evaluate", str(unmet_path), *given, "--status-contract", contract]) == expected
        reply = json.loads(capsys.readouterr().out)
        if expected == 64:
            assert reply["error"]["code"] == "REQUIREMENTS_NOT_SELECTED"
        else:
            assert reply["exit_contract"] == contract and reply["targets"] == [] and reply["status"] == unassessed
    assert main(["project", "evaluate", str(unmet_path), *given]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "execution" and reply["evaluation"]["requested"] == ["readings"]
    assert reply["status"] == reply["diagnostic_summary"] == unassessed
    assert _not_assessed(reply) == ["focus_position", "spot_radius"]
    assert reply["requirements"]["met"] == reply["requirements"]["unmet"] == []
    # (one target selected: requirements by default, over the selected target; the other is named as not assessed)
    unmet, reply = _codes(capsys, ["project", "evaluate", str(unmet_path), *given, "--targets", "spot_radius"])
    assert unmet == {"requirements": 1, "execution": 0, "aggregate": 1}
    assert [t["target_id"] for t in reply["targets"]] == ["spot_radius"]
    assert [t["target_id"] for t in reply["evaluation"]["targets_not_evaluated"]] == ["focus_position"]
    assert main(["project", "evaluate", str(unmet_path), *given, "--targets", "spot_radius"]) == 1
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "requirements" and _not_assessed(reply) == ["focus_position"]
    # (a partial selection decides on the selected set: the met target alone exits 0, and names the unmet one as not assessed)
    assert main(["project", "evaluate", str(unmet_path), *given, "--targets", "focus_position"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "requirements" and reply["status"] == present and _not_assessed(reply) == ["spot_radius"]
    # (every target assessed, by the outputs: requirements by default; the run of the same Project exits as it does)
    assert main(["project", "evaluate", str(unmet_path), *given, "--outputs", "readings,requirements"]) == 1
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "requirements" and _not_assessed(reply) == []
    assert main(["project", "run", str(unmet_path), *given, "--progress", "none"]) == 1
    capsys.readouterr()


def test_requirements_named_for_an_evaluation_that_assesses_no_target_is_refused_before_computing(tmp_path, capsys,
                                                                                                    monkeypatch):
    def computed(*args, **kwargs):
        raise AssertionError("the refusal comes before anything is computed")

    real = json_api.evaluate_project
    monkeypatch.setattr(json_api, "evaluate_project", computed)
    path = _write(tmp_path, "unmet.json", _unmet())
    for selection in ([], ["--outputs", "readings"], ["--outputs", "plane_bounds,beam_pairs"],
                      ["--outputs", "readings", "--exclude", "requirements"]):
        monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
        assert main(["project", "evaluate", str(path), *selection, "--status-contract", "requirements"]) == 64
        error = json.loads(capsys.readouterr().out)["error"]
        assert error["code"] == "REQUIREMENTS_NOT_SELECTED", selection
        assert error["problems"][0]["context"]["targets"] == ["focus_position", "spot_radius"]
        message = error["message"]
        assert message.startswith("--status-contract requirements") and "focus_position, spot_radius" in message
        assert "add requirements to --outputs" in message and "--targets" in message and "--status-contract execution" in message
        monkeypatch.setenv("OS_OPTICS_STATUS_CONTRACT", " Requirements ")
        assert main(["project", "evaluate", str(path), *selection]) == 64
        error = json.loads(capsys.readouterr().out)["error"]
        assert error["code"] == "REQUIREMENTS_NOT_SELECTED" and error["message"].startswith("OS_OPTICS_STATUS_CONTRACT=requirements")
    # (the flag overrides the environment, and another contract is not refused)
    monkeypatch.setattr(json_api, "evaluate_project", real)
    assert main(["project", "evaluate", str(path), "--status-contract", "execution", "--progress", "none"]) == 0
    assert json.loads(capsys.readouterr().out)["exit_contract"] == "execution"


def _rated_thin_lens() -> dict:
    """The scope fixture's 532 nm beam through an ideal thin lens that states a CW power rating, with a spot target."""
    document = json.loads((Path(__file__).resolve().parent / "fixtures" / "scope" / "thin_lens_band.project.json").read_text(encoding="utf-8"))
    document["optical_models"][0]["power_ratings"] = [{
        "regime": "CW", "value": "1", "unit": "W/cm2", "wavelength_nm": "532", "value_definition": "PEAK", "status": "DECLARED",
        "basis": "test fixture rating, not product data", "beam_diameter_mm": "1", "beam_diameter_definition": "1/E2",
        "polarization_condition": "linear"}]
    document["targets"] = [{"target_id": "spot", "metric": "spot_rms_mm", "observation_id": "image", "maximum": 1e-6, "unit": "mm"}]
    return document


def test_the_contract_follows_the_plan_of_the_projects_own_request(tmp_path, capsys, monkeypatch):
    """Whether a selection assesses targets is planned against the request the evaluation runs: without a power rating the
    exposures need no untraced light, so excluding it leaves a selection that assesses no target -- the default contract is
    execution and requirements named for it is refused; with a rating the same selection cannot be planned."""
    selection = ["--outputs", "surface_exposures", "--exclude", "untraced_light"]
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
    catalog = _write(tmp_path, "catalog.json", with_lossless_bulk(catalog_document()))
    given = ["--catalog", str(catalog)]
    unmet_path = _write(tmp_path, "unmet.json", _unmet())  # (no power rating; spot_radius is not met)
    assert main(["project", "evaluate", str(unmet_path), *given, *selection]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "execution" and reply["status"] == "NOT_APPLICABLE"
    assert "requirements" not in reply["evaluation"]["executed"] and _not_assessed(reply) == ["focus_position", "spot_radius"]
    assert main(["project", "evaluate", str(unmet_path), *given, *selection, "--status-contract", "requirements"]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "REQUIREMENTS_NOT_SELECTED"
    monkeypatch.setenv("OS_OPTICS_STATUS_CONTRACT", "requirements")
    assert main(["project", "evaluate", str(unmet_path), *given, *selection]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "REQUIREMENTS_NOT_SELECTED"
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT")
    rated_path = _write(tmp_path, "rated.json", _rated_thin_lens())
    assert main(["project", "evaluate", str(rated_path), *selection]) == 64
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "EXCLUDED_CAPABILITY_REQUIRED" and error["problems"][0]["context"]["needed_by"] == "surface_exposures"
    # (the rated Project with the untraced light kept: the exposures run, no target is assessed, execution by default)
    assert main(["project", "evaluate", str(rated_path), "--outputs", "surface_exposures"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["exit_contract"] == "execution" and "untraced_light" in reply["evaluation"]["executed"]


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_a_project_without_targets_keeps_its_requirements_exit(tmp_path, capsys, monkeypatch, lossless):
    """A Project that declares no targets has nothing to leave unassessed: under requirements its evaluation exits 0 once
    the calculation completed (as its run does), with an empty requirements block."""
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
    source_catalog = catalog_document()
    catalog = _write(tmp_path, "catalog.json", with_lossless_bulk(source_catalog) if lossless else source_catalog)
    given = ["--catalog", str(catalog)]
    expected = "PASS" if lossless else "UNKNOWN"
    path = _write(tmp_path, "free.json", {**_focus(), "targets": []})
    for argv, contract in (([], "execution"), (["--status-contract", "requirements"], "requirements"),
                           (["--outputs", "readings,requirements"], "requirements")):
        assert main(["project", "evaluate", str(path), *given, *argv]) == 0, argv
        reply = json.loads(capsys.readouterr().out)
        assert reply["exit_contract"] == contract and reply["status"] == expected, argv
        assert reply["requirements"] == {"declared": False, "met": [], "unmet": [], "indeterminate": [], "not_applicable": [],
                                         "not_assessed": []}, argv
    assert main(["project", "run", str(path), *given, "--status-contract", "requirements", "--progress", "none"]) == 0
    capsys.readouterr()


def test_a_target_outside_the_model_scope_is_undecided(tmp_path, capsys):
    edge = 12.7
    document = project([source(waist_mm=3.0), element("W", "window", WINDOW, (0.0, edge + 0.01, 0.0), ALONG_X)],
                       [plane("D", (100.0, 0.0, 0.0), radius_mm=25.0)],
                       [{"target_id": "radius", "metric": "beam_radius_mm", "observation_id": "D", "maximum": 10.0, "unit": "mm"}])
    catalog = _write(tmp_path, "catalog.json", catalog_document())
    codes, reply = _codes(capsys, ["project", "evaluate", str(_write(tmp_path, "w.json", document)), "--catalog", str(catalog),
                                   "--outputs", "readings,requirements"])
    assert codes == {"requirements": 2, "execution": 0, "aggregate": 3}
    assert reply["status"] == "OUT_OF_MODEL_SCOPE" and reply["targets"][0]["assessment"] == "INDETERMINATE"
    # (the plane readings and the requirements come with the model-scope findings, however the selection is spelled: an
    # exclusion that would drop them is refused, never graded MET without them)
    for outputs in ("requirements", "plane_readings,requirements"):
        for excluded in ("ray_paths", "model_scope"):
            assert main(["project", "evaluate", str(tmp_path / "w.json"), "--catalog", str(catalog), "--outputs", outputs,
                         "--exclude", excluded]) == 64
            refusal = json.loads(capsys.readouterr().out)["error"]
            assert refusal["code"] == "EXCLUDED_CAPABILITY_REQUIRED" and "requires model_scope" in refusal["message"]


def test_refusals_and_usage(tmp_path, capsys):
    path = _write(tmp_path, "focus.json", _focus())
    assert main(["project", "evaluate", str(path), "--outputs", "record", "--exclude", "untraced_light"]) == 64
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["error"]["code"] == "EXCLUDED_CAPABILITY_REQUIRED"
    assert main(["project", "evaluate", str(path), "--outputs", "no_such"]) == 64
    capsys.readouterr()
    # (an evaluation writes nothing; the layout checks and the control analysis are an Engine request's)
    for name, where in (("run_files", "project run"), ("layout", "os_optics.evaluate"), ("control", "os_optics.evaluate")):
        assert main(["project", "evaluate", str(path), "--outputs", name]) == 64
        refusal = json.loads(capsys.readouterr().out)["error"]
        assert refusal["code"] == "UNKNOWN_CAPABILITY" and where in refusal["message"]
    # (an empty list -- an unset shell variable -- is a usage error, not an evaluation of nothing)
    for option in ("--outputs", "--targets"):
        for empty in ("", ",", " "):
            assert main(["project", "evaluate", str(path), option, empty]) == 64
            assert json.loads(capsys.readouterr().out)["error"]["code"] == "INPUT_INVALID"
    broken = _focus()
    broken["components"] = [c for c in broken["components"] if c["optical_role"] != "source"]
    assert main(["project", "evaluate", str(_write(tmp_path, "broken.json", broken))]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "LOCAL_RUN_ERROR"
    assert main(["project", "evaluate", str(tmp_path / "missing.json")]) == 64
    capsys.readouterr()
    assert main(["project", "evaluate"]) == 64


def test_the_scene_of_an_evaluation_is_printed_or_written_to_the_files_named(tmp_path, capsys):
    """'--outputs view' prints the scene in the reply (``optical_scene``, as ``evaluate_project`` returns it);
    '--scene-out' / '--glb-out' compute the view beside what is selected and write the scene and its GLB -- the bytes a run
    directory holds for that scene -- instead of printing it; '--pupil-rays' sets what the view draws."""
    from os_optics._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
    from os_optics.optical_view.artifacts import scene_glb_bytes, scene_json_bytes

    path = _write(tmp_path, "focus.json", _focus())
    assert main(["project", "evaluate", str(path), "--outputs", "view"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["optical_scene"] == json_api.evaluate_project(_focus(), ["view"])["optical_scene"]
    scene_file, glb_file = tmp_path / "scene.json", tmp_path / "scene.glb"
    assert main(["project", "evaluate", str(path), "--outputs", "requirements", "--scene-out", str(scene_file),
                 "--glb-out", str(glb_file), "--pupil-rays", "ALL"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert "optical_scene" not in reply and reply["evaluation"]["requested"] == ["requirements", "view"]
    assert len(reply["targets"]) == 2
    want = json_api.evaluate_project(_focus(), ["requirements", "view"], None, None, {"pupil_ray_selection": "ALL"})
    assert want["identity"] == reply["identity"]
    scene = OpticalSimulationSceneV1.model_validate(want["optical_scene"])
    assert scene_file.read_bytes() == scene_json_bytes(scene) and glb_file.read_bytes() == scene_glb_bytes(scene)
    # (the scene file alone; without --outputs the readings and the view)
    alone = tmp_path / "alone.json"
    assert main(["project", "evaluate", str(path), "--scene-out", str(alone)]) == 0
    assert json.loads(capsys.readouterr().out)["evaluation"]["requested"] == ["readings", "view"]
    assert json.loads(alone.read_text(encoding="utf-8")) == json_api.evaluate_project(_focus(), ["readings", "view"])["optical_scene"]
    # (a refused evaluation writes no file; a file that cannot be written is a usage error naming it)
    refused = tmp_path / "refused.json"
    assert main(["project", "evaluate", str(path), "--outputs", "view", "--exclude", "secondary_rays", "--scene-out", str(refused)]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "EXCLUDED_CAPABILITY_REQUIRED" and not refused.exists()
    assert main(["project", "evaluate", str(path), "--scene-out", str(tmp_path / "missing" / "scene.json")]) == 64
    refusal = json.loads(capsys.readouterr().out)["error"]
    assert refusal["code"] == "INPUT_INVALID" and refusal["message"].startswith("cannot write the scene JSON file")


def test_describe_prints_the_index_or_one_capability(capsys):
    assert main(["project", "evaluate", "--describe"]) == 0
    index = json.loads(capsys.readouterr().out)
    assert index == {k: v for k, v in json_api.evaluation_capabilities().items() if k != "ok"}
    assert main(["project", "evaluate", "--describe", "untraced_light"]) == 0
    one = json.loads(capsys.readouterr().out)["capability"]
    # (the return-light analysis is made inside its readers' phases: the plane bounds and the surface exposures)
    assert one["id"] == "untraced_light" and one["phases"] == ["untraced_light"]
    # (a preset is described as the help says: what it selects and what it leaves out)
    for name in ("readings", "complete"):
        assert main(["project", "evaluate", "--describe", name]) == 0
        preset = json.loads(capsys.readouterr().out)
        assert preset == {"preset": json_api.evaluation_capabilities(name)["preset"]}
        assert preset["preset"]["members"] == index["presets"][name]["members"]
    assert main(["project", "evaluate", "--describe", "nope"]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNKNOWN_CAPABILITY"


def test_a_project_with_configurations_is_refused_and_the_help_says_how_one_is_run(tmp_path, capsys, monkeypatch):
    zoom = Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json"
    assert main(["project", "evaluate", str(zoom), "--progress", "none"]) == 65
    problem = json.loads(capsys.readouterr().out)["error"]["problems"][0]
    assert problem["code"] == "CONFIGURATION_NOT_SELECTED" and "'--configuration ID'" in problem["message"]
    monkeypatch.setenv("COLUMNS", "80")  # (argparse wraps the help to the terminal width; a wrap may split '--out-dir')
    with pytest.raises(SystemExit):
        main(["project", "evaluate", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "CONFIGURATION_NOT_SELECTED" in text and "project run PROJECT --configuration ID --out-dir DIR" in text
    assert "evaluates that configuration as its own Project" in text and "no configuration is chosen for you" in text
    # (the option the help and the refusal name evaluates that one configuration)
    assert main(["project", "evaluate", str(zoom), "--configuration", "wide", "--progress", "none"]) == 0
    assert json.loads(capsys.readouterr().out)["evaluation"]["requested"] == ["readings"]
    # (the shell route the help names runs that one configuration)
    assert main(["project", "run", str(zoom), "--configuration", "wide", "--out-dir", str(tmp_path / "run"), "--progress", "none"]) == 0
    capsys.readouterr()
    assert (tmp_path / "run" / "configurations" / "wide" / "manifest.json").is_file()
    assert not (tmp_path / "run" / "configurations" / "tele").exists()


@pytest.mark.parametrize("argv", [["project", "evaluate", "--help"], ["project", "--help"]])
def test_the_help_states_what_the_command_computes(argv, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    text = " ".join(capsys.readouterr().out.split())
    assert "evaluate" in text and "readings" in text
    if argv[1] == "evaluate":
        assert "NOT_COMPUTED" in text and "--status-contract" in text and "Nothing is written" in text
        # (the exit code covers the assessed targets; the reply's readings are compact, and the help says where the rest is)
        assert "every declared target is met" not in text and "targets_not_evaluated" in text
        assert "A Project with configurations" not in text and "Evaluation.results()" in text and "power ledger" in text


def test_progress_lines_name_only_the_phases_entered(tmp_path, capsys):
    path = _write(tmp_path, "focus.json", copy.deepcopy(_focus()))
    assert main(["project", "evaluate", str(path), "--progress", "json"]) == 0
    captured = capsys.readouterr()
    phases = {json.loads(line)["phase"] for line in captured.err.splitlines() if line.startswith("{")}
    assert {"lower", "trace"} <= phases and not phases & {"untraced_light", "bounds", "scene", "record", "write"}
