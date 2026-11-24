"""``os-optics workspace ...``: the composition operations from the shell (file in, file out, exit codes)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from composition_helpers import FOLD_DESCRIPTOR, LENS_DESCRIPTOR, fold_module, laser, lens_module, project_of

from os_optics.cli import main


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _last_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_the_workspace_commands_compose_run_and_report(tmp_path, capsys):
    fold_ws, lens_ws, system = tmp_path / "fold.ws.json", tmp_path / "lens.ws.json", tmp_path / "system.ws.json"
    assert main(["workspace", "init", _write(tmp_path / "fold.json", fold_module()), "--module", json.dumps(FOLD_DESCRIPTOR),
                 "-o", str(fold_ws)]) == 0
    assert main(["workspace", "init", _write(tmp_path / "lens.json", lens_module()), "--module", json.dumps(LENS_DESCRIPTOR),
                 "-o", str(lens_ws)]) == 0
    capsys.readouterr()
    assert main(["workspace", "run", str(fold_ws)]) == 0
    assert _last_json(capsys)["status"] == "PASS"
    assert main(["workspace", "export", str(fold_ws), "--descriptor", json.dumps(FOLD_DESCRIPTOR), "-o", str(tmp_path / "fold.module.json")]) == 0
    assert main(["workspace", "export", str(lens_ws), "--descriptor", json.dumps(LENS_DESCRIPTOR), "-o", str(tmp_path / "lens.module.json")]) == 0
    assert len(_read(tmp_path / "fold.module.json")["validations"]) == 1
    assert main(["workspace", "init", _write(tmp_path / "system.json", project_of([laser("LASER")])), "-o", str(system)]) == 0
    capsys.readouterr()
    assert main(["workspace", "import", str(system), str(tmp_path / "fold.module.json"), "--key", "fold", "--rebind", "SRC=LASER",
                 "--op-id", "imp-fold"]) == 0
    reply = _last_json(capsys)
    assert reply["written"] == str(system) and reply["result"]["instance_key"] == "fold" and "workspace" not in reply
    digest = reply["workspace_digest"]
    # retry: same op id, same request -> replayed, file unchanged
    before = system.read_text(encoding="utf-8")
    assert main(["workspace", "import", str(system), str(tmp_path / "fold.module.json"), "--key", "fold", "--rebind", "SRC=LASER",
                 "--op-id", "imp-fold"]) == 0
    assert _last_json(capsys)["replayed"] is True and _read(system) == json.loads(before)
    half = math.sqrt(0.5)
    assert main(["workspace", "import", str(system), str(tmp_path / "lens.module.json"), "--key", "foc", "--rebind", "SRC=LASER",
                 "--position", "0,120,0", "--quaternion", f"{half},0,0,{half}", "--expect-base", digest]) == 0
    capsys.readouterr()
    # a stale base is refused and nothing is written
    before = system.read_text(encoding="utf-8")
    assert main(["workspace", "transform", str(system), "--instance", "foc", "--translate", "0,0,1", "--expect-base", digest]) == 65
    refused = _last_json(capsys)
    assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "BASE_CHANGED"
    assert system.read_text(encoding="utf-8") == before
    assert main(["workspace", "select", str(system), "--instance", "fold", "--ids", "LASER"]) == 0
    assert _last_json(capsys)["selection"]["components"] == ["LASER", "fold_M1"]
    assert main(["workspace", "group", "create", str(system), "train", "--ids", "fold_M1,foc_L1", "--frame", "component:fold_M1"]) == 0
    assert main(["workspace", "transform", str(system), "--group", "train", "--axis", "0,0,1", "--angle", "0.5",
                 "--pivot", "component:fold_M1", "--op-id", "tilt"]) == 0
    capsys.readouterr()
    assert main(["workspace", "transform", str(system), "--group", "train", "--axis", "0,0,1", "--angle", "-0.5",
                 "--pivot", "component:fold_M1", "--op-id", "untilt"]) == 0
    capsys.readouterr()
    assert main(["workspace", "run", str(system), "--out-dir", str(tmp_path / "run")]) == 0
    run = _last_json(capsys)
    assert run["status"] == "PASS" and (tmp_path / "run" / "local_validation.json").exists()
    assert main(["workspace", "status", str(system)]) == 0
    status = _last_json(capsys)["status"]
    assert status["current_validation"] == 0 and [i["instance_key"] for i in status["instances"]] == ["fold", "foc"]
    assert main(["workspace", "history", str(system), "--op-id", "tilt"]) == 0
    assert _last_json(capsys)["found"] is True
    assert main(["workspace", "record", str(system), str(tmp_path / "run")]) == 0
    assert len(_read(system)["validations"]) == 1  # the same run is recorded once
    capsys.readouterr()
    assert main(["workspace", "group", "dissolve", str(system), "train"]) == 0
    assert main(["workspace", "transform", str(system), "--group", "nope", "--translate", "1,0,0"]) == 65
    assert main(["workspace", "transform", str(system), "--instance", "foc", "--axis", "0,0,1"]) == 64
    assert main(["workspace", "import", str(system), str(tmp_path / "missing.json"), "--key", "x"]) == 64


def test_workspace_help_states_the_retry_rules(capsys):
    try:
        main(["workspace", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    assert "--op-id" in text and "--expect-base" in text
    assert "atomic" in text and "Exit codes" in text


def _system_with_fold(tmp_path, capsys):
    fold_ws, system = tmp_path / "fold.ws.json", tmp_path / "system.ws.json"
    assert main(["workspace", "init", _write(tmp_path / "fold.json", fold_module()), "--module", json.dumps(FOLD_DESCRIPTOR), "-o", str(fold_ws)]) == 0
    assert main(["workspace", "init", _write(tmp_path / "system.json", project_of([laser("LASER")])), "-o", str(system)]) == 0
    assert main(["workspace", "import", str(system), str(fold_ws), "--key", "b", "--rebind", "SRC=LASER", "--position", "0,0,300"]) == 0
    capsys.readouterr()
    return system


def test_commands_without_op_id_are_each_applied(tmp_path, capsys):
    system = _system_with_fold(tmp_path, capsys)
    for dz in ("10", "-10", "10"):
        assert main(["workspace", "transform", str(system), "--instance", "b", "--translate", f"0,0,{dz}"]) == 0
        assert _last_json(capsys)["replayed"] is False
    mirror = next(c for c in _read(system)["project"]["components"] if c["instance_id"] == "b_M1")
    assert mirror["nominal_pose"]["position_mm"] == [0.0, 0.0, 310.0]


def test_an_operation_is_applied_with_its_check_and_refused_with_require_check(tmp_path, capsys):
    """Two folds imported onto the same place overlap: the import is applied and its check lists the problem; with
    --require-check it is refused and nothing is written."""

    system = _system_with_fold(tmp_path, capsys)
    fold_ws = tmp_path / "fold.ws.json"
    before = system.read_text(encoding="utf-8")
    second = ["workspace", "import", str(system), str(fold_ws), "--key", "c", "--rebind", "SRC=LASER", "--position", "0,0,300"]
    assert main([*second, "--require-check"]) == 65
    refused = _last_json(capsys)
    assert [p["code"] for p in refused["error"]["problems"]] == ["CHECK_PROJECT_REFUSED"] and system.read_text(encoding="utf-8") == before
    assert main(second) == 0
    check = _last_json(capsys)["result"]["check"]
    assert check["checked"] is True and check["ok"] is False and check["carried_problems"] == []
    assert check["new_problems"] == refused["error"]["problems"][0]["details"]["problems"]
    assert {p["code"] for p in check["new_problems"]} == {"OPTICAL_BODIES_OVERLAP"}
    assert system.read_text(encoding="utf-8") != before
    # the next operation repairs it: the second fold moved away
    assert main(["workspace", "transform", str(system), "--instance", "c", "--translate", "0,0,100", "--require-check"]) == 0
    assert _last_json(capsys)["result"]["check"] == {"checked": True, "ok": True, "new_problems": [], "carried_problems": []}


def test_update_project_with_ops_follows_the_same_rule(tmp_path, capsys):
    system = _system_with_fold(tmp_path, capsys)
    before = system.read_text(encoding="utf-8")
    ops = _write(tmp_path / "ops.json", [{"op": "remove_component", "instance_id": "LASER"}])
    assert main(["workspace", "update-project", str(system), "--ops", ops, "--require-check"]) == 65
    assert _last_json(capsys)["error"]["code"] == "PROJECT_REJECTED" and system.read_text(encoding="utf-8") == before
    assert main(["workspace", "update-project", str(system), "--ops", ops]) == 0
    check = _last_json(capsys)["result"]["check"]
    assert check["ok"] is False and [p["code"] for p in check["new_problems"]] == ["SOURCE_MISSING"]
    assert "LASER" not in [c["instance_id"] for c in _read(system)["project"]["components"]]


def test_update_project_replaces_the_project_and_prunes_the_organization(tmp_path, capsys):
    system = _system_with_fold(tmp_path, capsys)
    project = _read(system)["project"]
    project["targets"] = [t for t in project["targets"] if t["observation_id"] != "b_MON"]
    project["observations"] = [o for o in project["observations"] if o["observation_id"] != "b_MON"]
    edited = _write(tmp_path / "edited.project.json", project)
    assert main(["workspace", "update-project", str(system), edited, "--op-id", "drop-monitor"]) == 0
    reply = _last_json(capsys)
    assert reply["result"]["kind"] == "update_project" and [w["code"] for w in reply["result"]["warnings"]] == ["INSTANCE_MEMBERS_REMOVED"]
    written = _read(system)
    assert written["project"] == project and {"kind": "OBSERVATION", "id": "b_MON"} not in written["groups"][0]["members"]
    assert main(["workspace", "update-project", str(system), edited, "--op-id", "drop-monitor"]) == 0  # a retry
    assert _last_json(capsys)["replayed"] is True


def test_a_member_deleted_by_update_project_blocks_replace_until_its_edits_are_discarded(tmp_path, capsys):
    """The shell path of a deleted instance member: ``update-project`` drops the imported mirror, ``replace`` then exits 65
    (``INSTANCE_EDITED``, naming the deleted mirror and both ways to discard), and ``replace --discard-edits`` brings the
    mirror back exactly as the module revision has it."""

    system = _system_with_fold(tmp_path, capsys)
    imported = next(c for c in _read(system)["project"]["components"] if c["instance_id"] == "b_M1")
    project = _read(system)["project"]
    project["components"] = [c for c in project["components"] if c["instance_id"] != "b_M1"]
    assert main(["workspace", "update-project", str(system), _write(tmp_path / "no-mirror.project.json", project)]) == 0
    capsys.readouterr()
    assert all(c["instance_id"] != "b_M1" for c in _read(system)["project"]["components"])
    revision = tmp_path / "fold.ws.json"

    before = system.read_text(encoding="utf-8")
    assert main(["workspace", "replace", str(system), "b", str(revision), "--rebind", "SRC=LASER"]) == 65
    (problem,) = _last_json(capsys)["error"]["problems"]
    assert problem["code"] == "INSTANCE_EDITED"
    assert problem["details"]["edited"] == [{"kind": "COMPONENT", "id": "b_M1", "change": "REMOVED"}]
    assert "discard_edits" in problem["message"] and "--discard-edits" in problem["message"]
    assert system.read_text(encoding="utf-8") == before

    assert main(["workspace", "replace", str(system), "b", str(revision), "--rebind", "SRC=LASER", "--discard-edits"]) == 0
    reply = _last_json(capsys)
    assert {"kind": "COMPONENT", "id": "b_M1", "change": "REMOVED"} in reply["result"]["replaced"]["edits_discarded"]
    restored = next(c for c in _read(system)["project"]["components"] if c["instance_id"] == "b_M1")
    assert restored == imported
    assert main(["workspace", "status", str(system)]) == 0
    (instance,) = _last_json(capsys)["status"]["instances"]
    assert instance["instance_key"] == "b" and not instance["edited"]


def test_an_interrupted_write_leaves_the_workspace_file_as_it_was(tmp_path, capsys, monkeypatch):
    system = _system_with_fold(tmp_path, capsys)
    before = system.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", fail)
    assert main(["workspace", "transform", str(system), "--instance", "b", "--translate", "0,0,1"]) == 64
    assert system.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["fold.json", "fold.ws.json", "system.json", "system.ws.json"])
    assert "No space left on device" in capsys.readouterr().err


def test_a_run_directory_holding_a_file_of_the_persons_own_is_a_usage_error_and_nothing_is_written(tmp_path, capsys):
    """``workspace run --out-dir`` into a directory holding the person's own simulation_result.json (no Engine manifest lists
    it): the run is refused before anything is written -- exit 64 with the reason on standard error, as ``project run``
    gives for the same directory, not an internal error --, the file and the workspace file stay as they were."""
    fold_ws = tmp_path / "fold.ws.json"
    assert main(["workspace", "init", _write(tmp_path / "fold.json", fold_module()), "--module", json.dumps(FOLD_DESCRIPTOR),
                 "-o", str(fold_ws)]) == 0
    capsys.readouterr()
    out = tmp_path / "run"
    out.mkdir()
    (out / "simulation_result.json").write_bytes(b'{"mine": true}')
    workspace = fold_ws.read_bytes()
    assert main(["workspace", "run", str(fold_ws), "--out-dir", str(out)]) == 64
    captured = capsys.readouterr()
    assert "cannot write the run directory" in captured.err and "simulation_result.json" in captured.err
    assert "Traceback" not in captured.err and "INTERNAL_ERROR" not in captured.out
    assert sorted(p.name for p in out.iterdir()) == ["simulation_result.json"]
    assert (out / "simulation_result.json").read_bytes() == b'{"mine": true}'
    assert fold_ws.read_bytes() == workspace
