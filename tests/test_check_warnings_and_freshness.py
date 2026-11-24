"""Advisory check warnings, one-call run freshness and the compact target outcomes of a run.

* ``check_project`` (json_api, ``project check``, the workspace status) lists ``TARGET_UNIT_MISMATCH`` and
  ``SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND`` before any run, and what it announces is what the run then reports; no
  Project that was accepted before is refused.
* ``run_freshness`` (``local``, json_api, ``results freshness``) follows the workspace's ``validation_freshness`` rule for
  a run directory, its files, a ``run_project`` reply or a bare Local Validation record.
* ``RunResults.targets()`` (``results targets``, results query ``targets``) and ``results rays --offset/--limit``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from declared_helpers import declared_project
from local_helpers import catalog_document
from os_optics.cli import main
from os_optics.local import example_project, json_api, run_freshness, run_optical_project
from os_optics.results import load_run


def project() -> dict:
    return example_project().payload()


def _source(doc: dict) -> dict:
    return next(c for c in doc["components"] if c.get("source_model"))


def _printed(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# ------------------------------------------------------------------------------------------------ check warnings


def test_a_target_unit_mismatch_is_announced_and_the_run_reports_the_target_unknown():
    doc = project()
    doc["targets"][0]["unit"] = "um"
    checked = json_api.check_project(doc)
    assert checked["ok"] is True
    (warning,) = checked["warnings"]
    target = doc["targets"][0]
    assert warning["code"] == "TARGET_UNIT_MISMATCH" and warning["location"] == f"target {target['target_id']}"
    assert warning["details"] == {"target_id": target["target_id"], "metric": target["metric"], "unit": "um", "metric_unit": "mm"}
    outcome = next(t for t in run_optical_project(doc).result.targets if t.target_id == target["target_id"])
    assert outcome.status.value == "UNKNOWN" and "unit mismatch" in outcome.message


def test_a_source_line_outside_a_band_is_announced_and_reported_in_the_runs_model_scope_issues():
    doc = project()
    _source(doc)["source_model"]["wavelength_nm"] = 1300.0
    checked = json_api.check_project(doc)
    assert checked["ok"] is True
    lens = next(c["instance_id"] for c in doc["components"] if c.get("product"))
    (warning,) = checked["warnings"]
    assert warning["code"] == "SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND" and warning["location"] == f"component {lens}"
    source_id = _source(doc)["instance_id"]
    assert warning["details"] == {"instance_id": lens, "member_id": None, "source_ids": [source_id], "wavelengths_nm": [1300.0],
                                  "lines": [{"source_id": source_id, "wavelengths_nm": [1300.0]}],
                                  "band_nm": {"min_nm": "400", "max_nm": "1100"}, "reported_in": "model_scope_issues"}
    assert warning["message"].startswith("no source line of the Project (1300 nm) lies inside the 400-1100 nm band of the published model")
    assert f"every beam that meets {lens} is outside that model" in warning["message"] and "in model_scope_issues" in warning["message"]
    assert "If that light meets" not in warning["message"]  # the check states what it knows before a trace, nothing conditional
    run = run_optical_project(doc)
    assert any(i.code == "MODEL_WAVELENGTH_OUTSIDE_BAND" and i.instance_id == lens for i in run.model_scope_issues)
    assert run.status.value in ("OUT_OF_MODEL_SCOPE", "FAIL")


def test_an_element_some_source_line_fits_has_no_warning_and_the_run_reports_the_lines_that_reach_it():
    """A source with 633, 1550 and 350 nm through a 400-1100 nm lens: one line fits, so the check knows nothing more before
    a trace; the run reports the two lines that meet the lens outside its band, with the element and the wavelengths."""

    doc = project()
    assert json_api.check_project(doc)["warnings"] == []
    model = _source(doc)["source_model"]
    model.pop("wavelength_nm"), model.pop("power_w")
    model["spectral_lines"] = [{"wavelength_nm": 633.0, "power_w": 0.001}, {"wavelength_nm": 1550.0, "power_w": 0.001}, {"wavelength_nm": 350.0, "power_w": 0.001}]
    assert json_api.check_project(doc)["warnings"] == []
    lens = next(c["instance_id"] for c in doc["components"] if c.get("product"))
    (issue,) = [i for i in run_optical_project(doc).model_scope_issues if i.code == "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    assert issue.instance_id == lens and "350.0 nm, 1550.0 nm" in issue.message and "633" not in issue.message
    # no line fits: one warning for the element, naming every line of every source
    model["spectral_lines"] = [{"wavelength_nm": 1550.0, "power_w": 0.001}, {"wavelength_nm": 350.0, "power_w": 0.001}]
    second = copy.deepcopy(_source(doc))
    second["instance_id"] = "SRC_B"
    second["nominal_pose"]["position_mm"][1] += 2.0
    second["source_model"]["spectral_lines"] = [{"wavelength_nm": 1310.0, "power_w": 0.001}, {"wavelength_nm": 1550.0, "power_w": 0.001}]
    doc["components"].append(second)
    (warning,) = json_api.check_project(doc)["warnings"]
    first = _source(doc)["instance_id"]
    assert warning["details"]["source_ids"] == [first, "SRC_B"] and warning["details"]["wavelengths_nm"] == [350.0, 1310.0, 1550.0]
    assert warning["details"]["lines"] == [{"source_id": first, "wavelengths_nm": [350.0, 1550.0]},
                                           {"source_id": "SRC_B", "wavelengths_nm": [1310.0, 1550.0]}]
    assert "(350 nm, 1310 nm, 1550 nm)" in warning["message"]


def test_a_declared_model_is_checked_against_its_own_band():
    doc = declared_project(project(), catalog_document())
    _source(doc)["source_model"]["wavelength_nm"] = 1200.0
    (warning,) = json_api.check_project(doc)["warnings"]
    assert "declared model" in warning["message"] and warning["details"]["band_nm"] == {"min_nm": "400", "max_nm": "1100"}


def test_the_workspace_status_check_carries_the_warnings():
    doc = project()
    doc["targets"][0]["unit"] = "W"
    status = json_api.workspace_status(doc)["status"]
    assert status["check"]["ok"] is True and [w["code"] for w in status["check"]["warnings"]] == ["TARGET_UNIT_MISMATCH"]


# ------------------------------------------------------------------------------------------------ run freshness


@pytest.fixture
def run_dir(tmp_path) -> Path:
    directory = tmp_path / "run"
    run_optical_project(project(), output_dir=directory)
    return directory


def _target_edit(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    out["targets"][0]["tolerance_abs"] = 0.5
    return out


def test_a_run_is_current_for_its_own_project_in_every_form_of_the_run(run_dir):
    doc = project()
    reply = json_api.run_project(doc)
    record = json.loads((run_dir / "local_validation.json").read_text(encoding="utf-8"))
    for run in (run_dir, str(run_dir), reply, reply["files"], record, run_dir / "local_validation.json"):
        fresh = run_freshness(doc, run)
        assert (fresh["freshness"], fresh["reasons"]) == ("CURRENT", []), run if not isinstance(run, dict) else sorted(run)[:3]
    fresh = run_freshness(doc, run_dir)
    assert fresh["run"]["project_hash"] == fresh["current"]["project_hash"] == example_project().content_hash()
    assert fresh["run"]["engine_input_hash"] == fresh["current"]["engine_input_hash"]
    # freshness reads the record as stored; the files are verified only when asked
    not_verified = {"checked": False, "consistent": None, "problems": []}
    assert fresh["integrity"] == not_verified and run_freshness(doc, record)["integrity"] == not_verified
    verified = run_freshness(doc, run_dir, verify=True)
    assert verified["integrity"] == {"checked": True, "consistent": True, "problems": []}
    assert {k: v for k, v in verified.items() if k != "integrity"} == {k: v for k, v in fresh.items() if k != "integrity"}
    assert run_freshness(doc, record, verify=True)["integrity"] == not_verified  # (a bare record has no files to verify)


def test_the_reasons_tell_targets_physics_and_other_project_changes(run_dir):
    doc = project()
    assert run_freshness(_target_edit(doc), run_dir)["reasons"] == ["TARGETS_CHANGED"]
    moved = copy.deepcopy(doc)
    moved["components"][1]["nominal_pose"]["position_mm"][0] += 1.0
    assert run_freshness(moved, run_dir)["reasons"] == ["PHYSICS_CHANGED"]
    intent = {**doc, "design_intent": "OPTICAL_SYSTEM"}
    assert run_freshness(intent, run_dir)["reasons"] == ["OTHER_PROJECT_CHANGED"]
    # a bare record does not carry its targets: any Project change reads as physics
    record = json.loads((run_dir / "local_validation.json").read_text(encoding="utf-8"))
    assert run_freshness(_target_edit(doc), record)["reasons"] == ["PHYSICS_CHANGED"]
    # a budget edit together with a target edit: the aggregate's reasons name the targets only (the design projection does
    # not hold the numerical configuration); the physics identity does
    both = run_freshness(_target_edit({**doc, "compute_budget": {"max_paths": 100}}), run_dir)
    assert (both["freshness"], both["reasons"]) == ("HISTORICAL", ["TARGETS_CHANGED"])
    assert both["physics"] == {"freshness": "HISTORICAL", "reasons": ["PHYSICS_CHANGED"]}
    assert both["requirements"] == {"freshness": "HISTORICAL"}


CURRENT_PHYSICS = {"freshness": "CURRENT", "reasons": []}


def test_the_physics_and_the_requirements_are_stated_beside_the_aggregate(run_dir):
    """The physics identity is the Engine request without its targets: any change of the targets leaves it, any other
    change of the Engine input moves it."""

    doc = project()
    same = run_freshness(doc, run_dir)
    assert same["physics"] == CURRENT_PHYSICS and same["requirements"] == {"freshness": "CURRENT"}
    assert same["run"]["physics_input_hash"] == same["current"]["physics_input_hash"] and len(same["run"]["physics_input_hash"]) == 64
    assert same["run"]["physics_input_hash"] != same["run"]["engine_input_hash"]
    # the targets alone: bounds edited, a target added, a target removed
    added = copy.deepcopy(doc)
    added["targets"].append({**copy.deepcopy(doc["targets"][0]), "target_id": "one_more"})
    removed = copy.deepcopy(doc)
    removed["targets"] = removed["targets"][1:]
    for changed in (_target_edit(doc), added, removed):
        fresh = run_freshness(changed, run_dir)
        assert (fresh["freshness"], fresh["reasons"]) == ("HISTORICAL", ["TARGETS_CHANGED"])
        assert fresh["physics"] == CURRENT_PHYSICS and fresh["requirements"] == {"freshness": "HISTORICAL"}
        assert fresh["current"]["physics_input_hash"] == same["run"]["physics_input_hash"]
        assert fresh["current"]["engine_input_hash"] != same["run"]["engine_input_hash"]
    # every physical input moves it: a pose, the source power, an observation plane, a budget, a setting
    moved = copy.deepcopy(doc)
    moved["components"][1]["nominal_pose"]["position_mm"][0] += 1.0
    stronger = copy.deepcopy(doc)
    _source(stronger)["source_model"]["power_w"] = 2.0 * _source(doc)["source_model"]["power_w"]
    plane = copy.deepcopy(doc)
    plane["observations"][0]["pose"]["position_mm"][0] += 0.5
    for changed in (moved, stronger, plane, {**doc, "compute_budget": {"max_paths": 100}},
                    {**doc, "numerical_settings": {"power_epsilon_W": 1e-18}}):
        fresh = run_freshness(changed, run_dir)
        assert fresh["freshness"] == "HISTORICAL"
        assert fresh["physics"] == {"freshness": "HISTORICAL", "reasons": ["PHYSICS_CHANGED"]}, fresh["reasons"]
        assert fresh["requirements"] == {"freshness": "HISTORICAL"}
    # another catalog release of the same Project
    changed = catalog_document()
    lens_sku = next(c["product"]["optical_sku"] for c in doc["components"] if c.get("product"))
    product = next(p for p in changed["products"] if p["optical_sku"] == lens_sku)
    surface = next(s for s in product["simulation_models"][0]["surfaces"] if "radius_mm" in s)
    surface["radius_mm"] = str(float(surface["radius_mm"]) * 1.01)
    assert run_freshness(doc, run_dir, catalog=changed)["physics"] == {"freshness": "HISTORICAL", "reasons": ["CATALOG_CHANGED"]}
    # a bare record holds no Engine request: unchanged it is current, changed it cannot tell physics from targets
    record = json.loads((run_dir / "local_validation.json").read_text(encoding="utf-8"))
    assert run_freshness(doc, record)["physics"] == CURRENT_PHYSICS
    bare = run_freshness(_target_edit(doc), record)
    assert bare["physics"] == {"freshness": "UNDETERMINED", "reasons": ["ENGINE_REQUEST_NOT_AVAILABLE"]}
    assert bare["requirements"] == {"freshness": "UNDETERMINED"}
    # a run whose files do not verify: the physics its hashes give, the files' state in integrity (asked with verify)
    (run_dir / "simulation_result.json").write_bytes((run_dir / "simulation_result.json").read_bytes() + b" ")
    broken = run_freshness(doc, run_dir, verify=True)
    assert broken["physics"] == CURRENT_PHYSICS and broken["requirements"] == {"freshness": "CURRENT"}
    assert broken["integrity"]["checked"] is True and broken["integrity"]["consistent"] is False


def test_another_engine_build_and_another_catalog_input_are_reasons_too(run_dir, monkeypatch):
    import os_optics.composition.validation as validation

    doc = project()
    assert run_freshness(doc, run_dir)["engine_build_matches"] is True
    monkeypatch.setattr(validation, "engine_build_id", lambda: "9.9.9+0000000000000000")
    # the rule keeps a run of the same Project and Engine input current (as the workspace does) and says who made it
    same = run_freshness(doc, run_dir)
    assert (same["freshness"], same["reasons"], same["engine_build_matches"]) == ("CURRENT", [], False)
    assert run_freshness(_target_edit(doc), run_dir)["reasons"] == ["TARGETS_CHANGED", "ENGINE_CHANGED"]
    monkeypatch.undo()
    changed = catalog_document()
    lens_sku = next(c["product"]["optical_sku"] for c in doc["components"] if c.get("product"))
    product = next(p for p in changed["products"] if p["optical_sku"] == lens_sku)
    surface = next(s for s in product["simulation_models"][0]["surfaces"] if "radius_mm" in s)
    surface["radius_mm"] = str(float(surface["radius_mm"]) * 1.01)
    fresh = run_freshness(doc, run_dir, catalog=changed)
    assert (fresh["freshness"], fresh["reasons"]) == ("HISTORICAL", ["CATALOG_CHANGED"])


def test_a_project_that_no_longer_checks_is_undetermined(run_dir, monkeypatch):
    import os_optics.local.lowering as lowering
    from os_optics.local import LocalRunError, LocalRunErrorCode, LocalRunProblem

    def refuse(*_args, **_kwargs):
        raise LocalRunError([LocalRunProblem(LocalRunErrorCode.CATALOG_NOT_AVAILABLE, "no catalog")])

    monkeypatch.setattr(lowering, "lower_project", refuse)
    fresh = run_freshness(project(), run_dir)
    assert (fresh["freshness"], fresh["reasons"]) == ("UNDETERMINED", ["PROJECT_NOT_CHECKED"])
    assert fresh["current"]["check"]["ok"] is False


def test_a_run_whose_files_do_not_verify_keeps_the_freshness_its_hashes_give(run_dir):
    """Freshness and integrity are two answers: a changed run file never moves the freshness (the run is still the run
    of this Project, by the hashes its record states); asked to verify, the reply says the files do not verify."""

    from os_optics.local.freshness import FRESHNESS_REASONS, RunNotReadable

    assert "RUN_UNVERIFIED" not in FRESHNESS_REASONS and "RUN_UNVERIFIED" not in json_api.capabilities()["run_freshness"]["reasons"]
    files = json_api.run_project(project())["files"]
    files["simulation_result.json"] = files["simulation_result.json"] + b" "
    fresh = json_api.run_freshness(project(), files)
    assert (fresh["freshness"], fresh["reasons"]) == ("CURRENT", [])
    assert fresh["integrity"] == {"checked": False, "consistent": None, "problems": []}
    asked = json_api.run_freshness(project(), files, True)
    assert (asked["freshness"], asked["reasons"], asked["integrity"]["checked"], asked["integrity"]["consistent"]) == ("CURRENT", [], True, False)
    assert any(problem.startswith("simulation_result.json:") for problem in asked["integrity"]["problems"])
    # the same on the command line: the flag adds the integrity block's answer and never moves the exit code
    glb = run_dir / "optical_scene.glb"
    glb.write_bytes(glb.read_bytes() + b"\x00\x00\x00\x00")
    path = run_dir.parent / "the_project.json"
    path.write_text(json.dumps(project()), encoding="utf-8")
    assert run_freshness(project(), run_dir)["freshness"] == "CURRENT"
    assert run_freshness(project(), run_dir, verify=True)["integrity"]["consistent"] is False
    # a directory without a manifest holds no run to compare
    (run_dir / "manifest.json").unlink()
    with pytest.raises(RunNotReadable, match="no run was completed into it"):
        run_freshness(project(), run_dir)
    assert json_api.run_freshness(project(), str(run_dir))["error"]["code"] == "RUN_NOT_READABLE"


def test_results_freshness_verifies_on_request(run_dir, capsys):
    path = run_dir.parent / "the_project.json"
    path.write_text(json.dumps(project()), encoding="utf-8")
    assert main(["results", "freshness", str(run_dir), "--project", str(path)]) == 0
    assert _printed(capsys)["integrity"] == {"checked": False, "consistent": None, "problems": []}
    assert main(["results", "freshness", str(run_dir), "--project", str(path), "--verify"]) == 0
    assert _printed(capsys)["integrity"] == {"checked": True, "consistent": True, "problems": []}
    glb = run_dir / "optical_scene.glb"
    glb.write_bytes(glb.read_bytes() + b"\x00\x00\x00\x00")
    assert main(["results", "freshness", str(run_dir), "--project", str(path)]) == 0
    assert _printed(capsys)["integrity"]["checked"] is False
    assert main(["results", "freshness", str(run_dir), "--project", str(path), "--verify"]) == 0  # (still CURRENT)
    report = _printed(capsys)
    assert report["freshness"] == "CURRENT" and report["integrity"]["consistent"] is False and report["integrity"]["problems"]


def test_the_results_help_names_the_two_commands_that_read_a_run_as_stored(capsys):
    """'results freshness' checks the files only with --verify (as test_results_freshness_verifies_on_request shows) and
    'results configurations' reads its run as stored: the group's help and the README say so instead of 'each command checks
    the run's files first'."""
    with pytest.raises(SystemExit):
        main(["results", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Each command that reads a run directory checks the run's files first" in text
    assert "'freshness' reads the run's record as stored and checks the files only with --verify" in text
    assert "'configurations' reads the comparison as stored" in text
    with pytest.raises(SystemExit):
        main(["results", "configurations", "--help"])
    assert "The run is read as stored: 'project verify RUN' checks its files." in " ".join(capsys.readouterr().out.split())
    readme = " ".join((Path(__file__).resolve().parents[1] / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    assert "Two read as stored: `results freshness`, which checks the files only with `--verify`" in readme


def test_results_commands_check_the_run_and_read_it_as_stored_on_request(run_dir, capsys):
    """Every results / export command checks the run's files first: a changed file brings the warning naming 'project
    verify' and integrity.checked is true; --no-verify reads them as stored (no warning, integrity.checked false).
    'project verify' itself always checks."""

    glb = run_dir / "optical_scene.glb"
    glb.write_bytes(glb.read_bytes() + b"\x00\x00\x00\x00")
    assert main(["results", "describe", str(run_dir)]) == 0
    printed = capsys.readouterr()
    integrity = json.loads(printed.out)["integrity"]
    assert integrity["checked"] is True and integrity["consistent"] is False and integrity["problems"]
    assert "does not verify" in printed.err and "'os-optics project verify'" in printed.err
    assert main(["results", "describe", str(run_dir), "--no-verify"]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["integrity"] == {"checked": False, "consistent": None, "problems": []} and printed.err == ""
    for command in (["results", "targets"], ["results", "monitors"], ["export", "arrivals"]):
        assert main([*command, str(run_dir)]) == 0
        assert "does not verify" in capsys.readouterr().err, command
        assert main([*command, str(run_dir), "--no-verify"]) == 0
        assert capsys.readouterr().err == "", command
    assert main(["project", "verify", str(run_dir)]) == 66
    assert _printed(capsys)["status"] == "INCONSISTENT"


def test_a_results_command_names_a_tampered_run_by_default(run_dir, capsys):
    """A target value doubled in simulation_result.json: 'results targets' names the run as one that does not verify."""

    path = run_dir / "simulation_result.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    target = next(t for t in doc["targets"] if t.get("value") is not None)
    target["value"] = 2.0 * target["value"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["results", "targets", str(run_dir)]) == 0
    assert "does not verify" in capsys.readouterr().err


def test_refusals_are_envelopes(tmp_path):
    assert json_api.run_freshness({"schema": "x"}, {})["error"]["code"] == "LOCAL_RUN_ERROR"
    for run in ({"schema": "opensci.local_validation"}, 42, str(tmp_path / "missing"), {"local_validation.json": b"{"}):
        reply = json_api.run_freshness(project(), run)
        assert reply["ok"] is False and reply["error"]["code"] == "RUN_NOT_READABLE", run


def test_cli_results_freshness_exit_codes(run_dir, tmp_path, capsys):
    current = tmp_path / "p.json"
    current.write_text(json.dumps(project()), encoding="utf-8")
    assert main(["results", "freshness", str(run_dir), "--project", str(current)]) == 0
    assert _printed(capsys)["freshness"] == "CURRENT"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(_target_edit(project())), encoding="utf-8")
    assert main(["results", "freshness", str(run_dir), "--project", str(edited)]) == 1
    assert _printed(capsys)["reasons"] == ["TARGETS_CHANGED"]
    assert main(["results", "freshness", str(run_dir / "local_validation.json"), "--project", str(edited)]) == 1
    capsys.readouterr()
    assert main(["results", "freshness", str(tmp_path / "nope"), "--project", str(current)]) == 64
    assert _printed(capsys)["error"]["code"] == "RUN_NOT_READABLE"
    assert main(["results", "freshness", str(run_dir), "--project", str(tmp_path / "missing.json")]) == 64
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "opensci.project"}), encoding="utf-8")
    assert main(["results", "freshness", str(run_dir), "--project", str(bad)]) == 65


# ------------------------------------------------------------------------------------------------ target outcomes, rays paging


def test_targets_lists_every_outcome_with_the_model_scope_issues(tmp_path, capsys):
    doc = project()
    _source(doc)["source_model"]["wavelength_nm"] = 1300.0
    doc["targets"].append({"target_id": "delivered", "metric": "power_W", "observation_id": doc["targets"][0]["observation_id"],
                           "minimum": 0.0, "unit": "W"})
    run = run_optical_project(doc, output_dir=tmp_path / "run")
    targets = load_run(tmp_path / "run").targets()
    assert targets["validation_status"] == run.status.value and targets["engine_status"] == run.result.status.value
    assert [t["target_id"] for t in targets["targets"]] == [t.target_id for t in run.result.targets]
    # The lens is used outside its model band (which touches every kind of quantity) and its coatings outside their bands
    # (which touches power and polarization): a target on the Gaussian beam carries the first, a power both.
    scope_codes = {"waist_distance_x_mm": ["MODEL_WAVELENGTH_OUTSIDE_BAND"], "beam_radius_mm": ["MODEL_WAVELENGTH_OUTSIDE_BAND"],
                   "power_W": ["COATING_WAVELENGTH_OUTSIDE_BANDS", "MODEL_WAVELENGTH_OUTSIDE_BAND"]}
    assert sorted(row["metric"] for row in targets["targets"]) == sorted(scope_codes)
    assert sorted({i.code for i in run.model_scope_issues}) == ["COATING_WAVELENGTH_OUTSIDE_BANDS", "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    for row, outcome in zip(targets["targets"], run.result.targets):
        # Raw numbers and legacy status remain available; applicability governs
        # the scientific assessment and is included in the explanatory codes.
        assert (row["status"], row["value"], row["unit"]) == (outcome.status.value, outcome.value, outcome.unit)
        assert row["assessment"] == "INDETERMINATE"
        assert row["applicability_issue_codes"] == scope_codes[row["metric"]]
        assert row["issue_codes"] == sorted(set(outcome.issue_codes) | set(scope_codes[row["metric"]]))
        assert all(row["bounds"][k] == getattr(outcome, k) for k in row["bounds"])
    # (the coating finding names the power target only: the example's own targets read beam geometry)
    coating = next(i for i in run.model_scope_issues if i.code == "COATING_WAVELENGTH_OUTSIDE_BANDS")
    assert coating.affected_target_ids == ("delivered",) and coating.affected_observation_ids
    assert targets["model_scope_issues"] == [i.as_dict() for i in run.model_scope_issues] and targets["model_scope_issues"]
    assert sum(targets["counts"].values()) == len(run.result.targets)
    assert targets["run"]["project_hash"] == run.local_validation.project.project_hash.digest
    assert main(["results", "targets", str(tmp_path / "run")]) == 0
    assert _printed(capsys) == json.loads(json.dumps(targets))
    reply = json_api.run_results(json_api.run_project(doc)["files"], {"kind": "targets"})
    assert reply["ok"] is True and reply["result"]["targets"] == json.loads(json.dumps(targets["targets"]))


def test_results_rays_pages_like_the_json_api(run_dir, capsys):
    assert main(["results", "rays", str(run_dir)]) == 0
    every = _printed(capsys)
    assert isinstance(every, list) and len(every) > 3
    assert main(["results", "rays", str(run_dir), "--offset", "1", "--limit", "2"]) == 0
    page = _printed(capsys)
    files = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    assert page == json_api.run_results(files, {"kind": "rays", "offset": 1, "limit": 2})["result"]
    assert page["total"] == len(every) and page["rows"] == every[1:3]
    assert main(["results", "rays", str(run_dir), "--limit", "0"]) == 0
    assert _printed(capsys)["rows"] == []
    assert main(["results", "rays", str(run_dir), "--offset", "-1"]) == 64


# ------------------------------------------------------------------------------------------------ tabulated thin lens domain


def _ttl(lines, **model_fields):
    from test_tabulated_thin_lens import ttl_model, ttl_project

    return ttl_project(lines, model=ttl_model(**model_fields) if model_fields else None)


def _engine_scope_codes(run) -> list[str]:
    return [i.code.value for i in run.result.issues if i.status.value == "OUT_OF_MODEL_SCOPE"]


def test_a_tabulated_thin_lens_line_outside_its_rows_is_announced_as_an_engine_issue():
    doc = _ttl(((450.0, 0.001), (500.0, 0.001)))  # rows 490..580 nm, linear: 450 nm is outside the span
    assert json_api.check_project(doc)["warnings"] == []  # (500 nm fits the table: the run says which line reaches the lens)
    (warning,) = json_api.check_project(_ttl(((450.0, 0.001),)))["warnings"]
    assert warning["details"]["wavelengths_nm"] == [450.0] and warning["details"]["reported_in"] == "engine_issues"
    assert warning["message"].endswith("the run reports MODEL_WAVELENGTH_OUTSIDE_BAND as an Engine issue of the run (result.issues, "
                                       "status OUT_OF_MODEL_SCOPE) and in model_scope_issues.")
    run = run_optical_project(doc)
    assert _engine_scope_codes(run) == ["MODEL_WAVELENGTH_OUTSIDE_BAND"]
    # the Engine's issue is also listed with what it affects (the plane behind the lens), like any model-scope issue
    assert [(i.instance_id, i.code, i.surface_id, i.affected_observation_ids) for i in run.model_scope_issues] == [
        ("ML", "MODEL_WAVELENGTH_OUTSIDE_BAND", "lens", ("P",))]


def test_a_points_only_table_is_judged_by_its_rows_as_the_engine_does():
    points = {"focal_interpolation": "POINTS_ONLY", "transmittance_interpolation": "POINTS_ONLY"}
    near_row = _ttl(((532.0, 0.001), (581.0, 0.001)), **points)  # 581 nm is within 0.5 % of the 580 nm row
    assert json_api.check_project(near_row)["warnings"] == []
    run = run_optical_project(near_row)
    assert _engine_scope_codes(run) == [] and run.status.value != "OUT_OF_MODEL_SCOPE"
    between = _ttl(((510.0, 0.001),), **points)  # between rows: outside a points-only table
    (warning,) = json_api.check_project(between)["warnings"]
    assert warning["details"]["wavelengths_nm"] == [510.0] and "its rows only" in warning["message"]
    assert _engine_scope_codes(run_optical_project(between)) == ["MODEL_WAVELENGTH_OUTSIDE_BAND"]


def test_a_unit_mismatch_on_a_target_the_run_cannot_evaluate_promises_no_unknown():
    doc = project()
    doc["targets"][0]["unit"] = "um"
    (warning,) = json_api.check_project(doc)["warnings"]
    assert "its status is UNKNOWN (a target the run cannot evaluate at all keeps that outcome)" in warning["message"]


# ------------------------------------------------------------------------------------------------ out of model in one call


def test_targets_report_an_engine_out_of_model_issue_that_a_failing_target_masks(tmp_path):
    from test_tabulated_thin_lens import ttl_project

    failing = [{"target_id": "far", "metric": "waist_distance_x_mm", "observation_id": "P", "wavelength_nm": 500.0, "target_value": 1000.0,
                "tolerance_abs": 0.1, "unit": "mm"}]
    doc = ttl_project(((450.0, 0.001), (500.0, 0.001)), targets=failing)
    run = run_optical_project(doc, output_dir=tmp_path / "run")
    assert run.status.value == "FAIL"  # worst-of masks it
    targets = load_run(tmp_path / "run").targets()
    assert targets["in_model_scope"] is False and targets["model_scope_issues"] == [i.as_dict() for i in run.model_scope_issues]
    engine, scope = targets["out_of_model_issues"]
    assert (engine["source"], engine["code"], engine["location"]) == ("ENGINE_ISSUE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "surface:ML/lens")
    assert (scope["source"], scope["code"], scope["instance_id"], scope["surface_id"]) == ("MODEL_SCOPE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "ML", "lens")
    # the 450 nm light is outside the lens's table; the target selects the 500 nm line, which the table covers: it is decided
    # on that line alone, as in the same design without the 450 nm line
    (row,) = targets["targets"]
    assert (row["status"], row["assessment"], row["applicability_issue_codes"]) == ("FAIL", "UNMET", [])
    alone = load_run(run_optical_project(ttl_project(((500.0, 0.001),), targets=failing), output_dir=tmp_path / "alone")).targets()
    assert alone["model_scope_issues"] == [] and (alone["targets"][0]["assessment"], alone["targets"][0]["value"]) == ("UNMET", row["value"])
    reply = json_api.run_results(json_api.run_project(doc)["files"], {"kind": "targets"})
    assert reply["result"]["in_model_scope"] is False and reply["result"]["out_of_model_issues"] == json.loads(json.dumps(targets["out_of_model_issues"]))


def test_targets_list_an_engine_issue_repeated_at_one_place_once(tmp_path):
    from test_tabulated_thin_lens import ttl_project

    doc = ttl_project(((440.0, 0.001), (450.0, 0.001), (500.0, 0.001)))  # rows 490..580 nm: two lines outside, one lens
    run = run_optical_project(doc, output_dir=tmp_path / "run")
    assert _engine_scope_codes(run) == ["MODEL_WAVELENGTH_OUTSIDE_BAND"] * 2
    issue, scope = load_run(tmp_path / "run").targets()["out_of_model_issues"]
    assert (issue["source"], issue["code"], issue["location"]) == ("ENGINE_ISSUE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "surface:ML/lens")
    assert issue["message"] == run.result.issues[0].message
    # its model-scope entry is one per element, surface and code too, in the Engine's words for each line
    assert (scope["source"], scope["code"], scope["surface_id"]) == ("MODEL_SCOPE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "lens")
    assert scope["message"] == " ".join(i.message for i in run.result.issues if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND")


def test_targets_list_model_scope_issues_among_the_out_of_model_issues_and_an_in_model_run_is_in_scope(tmp_path):
    doc = project()
    run_optical_project(doc, output_dir=tmp_path / "in")
    inside = load_run(tmp_path / "in").targets()
    assert (inside["in_model_scope"], inside["out_of_model_issues"]) == (True, [])
    _source(doc)["source_model"]["wavelength_nm"] = 1300.0
    run = run_optical_project(doc, output_dir=tmp_path / "out")
    outside = load_run(tmp_path / "out").targets()
    scope = [i for i in outside["out_of_model_issues"] if i["source"] == "MODEL_SCOPE"]
    assert outside["in_model_scope"] is False and [{k: v for k, v in i.items() if k != "source"} for i in scope] == [i.as_dict() for i in run.model_scope_issues]


# ------------------------------------------------------------------------------------------------ freshness inputs


def test_run_files_given_as_text_are_read_as_their_utf8_bytes(run_dir):
    files = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    texts = {name: data.decode("utf-8") if name.endswith(".json") else data for name, data in files.items()}
    fresh = json_api.run_freshness(project(), texts, True)
    assert (fresh["freshness"], fresh["integrity"]) == ("CURRENT", {"checked": True, "consistent": True, "problems": []})
    assert json_api.run_freshness(project(), texts)["integrity"]["checked"] is False
    reply = json_api.run_freshness(project(), {"local_validation.json": 42})
    assert reply["error"]["code"] == "RUN_NOT_READABLE" and "not int" in reply["error"]["problems"][0]["message"]


def test_capabilities_say_engine_changed_comes_only_with_another_reason():
    note = json_api.capabilities()["run_freshness"]["engine_build"]
    assert "only with another reason" in note and "engine_build_matches false" in note
