"""Status consumers: every reply that reports a run's targets also says what was decided -- the requirement ``assessment`` of each
target and the run's ``outcome`` -- beside the diagnostic ``status``, and the assessment is the Local Validation record's
(a published model used outside its validity makes a target INDETERMINATE whatever the Engine graded it).

The case: the reference design run at 1300 nm, beyond its products' bands. Every target keeps the Engine's raw grade as its
status while the record makes it INDETERMINATE (``test_check_warnings_and_freshness.py`` checks the codes). The
expectations are the record's own lists (``outcome.requirements``), never another reply under test."""

from __future__ import annotations

import json

from composition_helpers import ok
from os_optics.cli import main
from os_optics.local import example_project, json_api, run_optical_project
from os_optics.results import load_run


def _out_of_band() -> dict:
    doc = example_project().payload()
    next(c for c in doc["components"] if c.get("source_model"))["source_model"]["wavelength_nm"] = 1300.0
    return doc


def _decided(record_outcome: dict) -> dict[str, str]:
    """target id -> assessment, from the record's requirement lists."""
    requirements = record_outcome["requirements"]
    return {target_id: name.upper() for name in ("met", "unmet", "indeterminate", "not_applicable") for target_id in requirements[name]}


def test_results_targets_and_describe_carry_the_records_assessment(tmp_path, capsys):
    run = run_optical_project(_out_of_band(), output_dir=tmp_path / "run")
    outcome = run.local_validation.simulation.outcome.model_dump(mode="json")
    decided = _decided(outcome)
    assert set(decided.values()) == {"INDETERMINATE"} and {t.status.value for t in run.result.targets} != {"UNKNOWN"}
    results = load_run(tmp_path / "run")
    targets = results.targets()
    assert {row["target_id"]: row["assessment"] for row in targets["targets"]} == decided
    assert targets["assessments"] == {"INDETERMINATE": len(decided)}
    assert sum(targets["counts"].values()) == sum(targets["assessments"].values()) == len(run.result.targets)
    assert targets["outcome"] == outcome
    described = results.describe()
    assert {row["target_id"]: row["assessment"] for row in described["status"]["targets"]} == decided
    # the manifest's own block is unchanged on disk: the assessment is added where the run is read
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert all("assessment" not in row for row in manifest["status"]["targets"])
    assert main(["results", "targets", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["assessments"] == {"INDETERMINATE": len(decided)}


def test_the_json_api_workspace_run_says_what_was_decided_as_the_cli_does(tmp_path, capsys):
    ws = ok(json_api.workspace_init(_out_of_band()))["workspace"]
    reply = ok(json_api.workspace_run(ws))
    outcome = reply["outcome"]
    assert outcome == reply["summary"]["outcome"]
    assert outcome["requirements"] == reply["local_validation"]["simulation"]["outcome"]["requirements"]
    assert {t["target_id"]: t["assessment"] for t in reply["targets"]} == _decided(outcome)
    assert [t["target_id"] for t in reply["targets"]] == [t["target_id"] for t in reply["summary"]["targets"]]
    assert all(t["status"] == s["status"] for t, s in zip(reply["targets"], reply["summary"]["targets"]))
    path = tmp_path / "ws.json"
    path.write_text(json.dumps(ws), encoding="utf-8")
    code = main(["workspace", "run", str(path), "-o", str(tmp_path / "out.json")])
    cli = json.loads(capsys.readouterr().out)
    assert code == 2  # (requirements contract: the targets are undecided)
    # (the same rows; the JSON API states the computed numbers at the result hash's quantization, as its summary does, and
    # the target value as the Project states it)
    from os_optics.config import NumericalConfig
    from os_optics.manifest import quantize

    numerical = NumericalConfig()
    assert cli["outcome"] == reply["outcome"]
    assert [{**quantize(row, numerical.hash_float_sig_digits, numerical.hash_float_abs_floor), "target_value": row["target_value"]}
            for row in cli["targets"]] == reply["targets"]
    assert all("target_value" in row and "error" in row for row in reply["targets"])
