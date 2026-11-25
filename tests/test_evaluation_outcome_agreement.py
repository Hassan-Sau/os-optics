"""The coverage of a Project evaluation reads the same through every interface.

Which declared targets an evaluation assessed, how each came out, which it did not assess, the diagnostic summary beside
them and the exit code they give: Python (``os_optics.local.EvaluationSession`` / ``Evaluation``), the JSON API
(``json_api.evaluate_project``), the command line (``os-optics project evaluate``) and the run directory a complete
evaluation writes (``Evaluation.write_run``, read back with ``os-optics results targets`` and compared with the
summary of ``project run``). Two Projects: the example with one target met and one not met, and a window used outside
its published geometry, whose target is undetermined by its model-scope finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from os_optics.cli import main
from os_optics.cli_exit import (
    REQUIREMENT_UNDECIDED,
    REQUIREMENT_UNMET,
    REQUIREMENTS_MET,
    selection_contract,
)
from os_optics.local import EvaluationSession, json_api

from local_helpers import ALONG_X, WINDOW, catalog_document, element, plane, project, source

ASSESSMENTS = ("met", "unmet", "indeterminate", "not_applicable")
#: Each selection: the keyword arguments of an evaluation and the same selection on the command line.
SELECTIONS = {
    "readings": ({}, []),
    "first target": ({"targets": "FIRST"}, ["--targets", "FIRST"]),
    "second target": ({"targets": "SECOND"}, ["--targets", "SECOND"]),
    "requirements": ({"outputs": ["readings", "requirements"]}, ["--outputs", "readings,requirements"]),
}


#: How each target comes out when it is assessed: the example's spot target is not met; the window's model-scope finding
#: leaves both of its targets undetermined.
EXPECTED = {"example": {"focus_position": "MET", "spot_radius": "UNMET"},
            "window": {"radius": "INDETERMINATE", "waist": "INDETERMINATE"}}


def _example() -> tuple[dict, None]:
    """``focus_position`` met, ``spot_radius`` not met (its maximum divided by 1000)."""
    document = json_api.example_project_document()["project"]
    spot = next(t for t in document["targets"] if t["target_id"] == "spot_radius")
    spot["maximum"] = spot["maximum"] / 1000.0
    return document, None


def _window(tmp_path: Path) -> tuple[dict, Path]:
    """A beam that grazes a window's edge: a model-scope finding on the window makes both targets behind it undetermined."""
    edge = 12.7
    document = project([source(waist_mm=3.0), element("W", "window", WINDOW, (0.0, edge + 0.01, 0.0), ALONG_X)],
                       [plane("D", (100.0, 0.0, 0.0), radius_mm=25.0)],
                       [{"target_id": "radius", "metric": "beam_radius_mm", "observation_id": "D", "maximum": 10.0, "unit": "mm"},
                        {"target_id": "waist", "metric": "waist_radius_x_mm", "observation_id": "D", "maximum": 10.0, "unit": "mm"}])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog_document()), encoding="utf-8")
    return document, path


def _selection(name: str, declared: list[str]) -> tuple[dict, list[str], list[str]]:
    """The selection ``name`` on the Project's declared targets: Python keywords, command-line arguments, assessed ids."""
    keywords, argv = SELECTIONS[name]
    keywords, argv = dict(keywords), list(argv)
    if "targets" in keywords:
        chosen = declared[0] if keywords["targets"] == "FIRST" else declared[1]
        keywords["targets"] = [chosen]
        argv = ["--targets", chosen]
        return keywords, argv, [chosen]
    return keywords, argv, (declared if "outputs" in keywords else [])


def _expected_exit(contract: str, requirements: dict) -> int:
    if contract == "execution":
        return 0
    if requirements["unmet"]:
        return REQUIREMENT_UNMET
    if requirements["indeterminate"] or requirements["not_applicable"]:
        return REQUIREMENT_UNDECIDED
    return REQUIREMENTS_MET


@pytest.mark.parametrize("case", ["example", "window"])
@pytest.mark.parametrize("name", list(SELECTIONS))
def test_python_json_and_the_command_line_agree_on_the_coverage(case, name, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
    document, catalog = _example() if case == "example" else _window(tmp_path)
    declared = [t["target_id"] for t in document["targets"]]
    keywords, argv, assessed = _selection(name, declared)

    evaluation = EvaluationSession(catalog=None if catalog is None else str(catalog)).evaluate(document, **keywords)
    reply = json_api.evaluate_project(document, keywords.get("outputs"), None, keywords.get("targets"),
                                      catalog=None if catalog is None else str(catalog))
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    code = main(["project", "evaluate", str(path), "--progress", "none", *argv,
                 *([] if catalog is None else ["--catalog", str(catalog)])])
    printed = json.loads(capsys.readouterr().out)

    # the requirements block: one partition of the declared targets, the same in Python, JSON and on the command line
    requirements = evaluation.requirements()
    assert reply["requirements"] == printed["requirements"] == requirements
    assessed_ids = [i for key in ASSESSMENTS for i in requirements[key]]
    assert sorted(assessed_ids) == sorted(assessed)
    assert [row["target_id"] for row in requirements["not_assessed"]] == [i for i in declared if i not in assessed]
    assert [row["target_id"] for row in requirements["not_assessed"]] == \
        [t["target_id"] for t in evaluation.describe_scope()["targets_not_evaluated"]] == \
        [t["target_id"] for t in reply["evaluation"]["targets_not_evaluated"]]
    # each assessed target's assessment, wherever it is read
    by_list = {i: key.upper() for key in ASSESSMENTS for i in requirements[key]}
    assert by_list == {i: EXPECTED[case][i] for i in assessed}
    assert {t["target_id"]: t["assessment"] for t in reply["targets"]} == by_list
    assert {t["target_id"]: t["assessment"] for t in printed["targets"]} == by_list
    if assessed:
        rows = evaluation.results().targets()
        assert {row["target_id"]: row["assessment"] for row in rows["targets"]} == by_list
        assert [t["target_id"] for t in rows.get("not_evaluated", [])] == [i for i in declared if i not in assessed]
    # the diagnostic summary: the same everywhere, and never PASS over declared targets none of which was assessed
    assert evaluation.status == reply["status"] == reply["diagnostic_summary"] == printed["status"] == printed["diagnostic_summary"]
    if not assessed:
        assert evaluation.status != "PASS"
    # the exit code: the contract the selection answers, over the targets it assessed
    contract = selection_contract(bool(assessed))
    assert printed["exit_contract"] == contract and code == _expected_exit(contract, requirements)


def test_a_complete_evaluation_its_run_directory_and_project_run_agree(tmp_path, capsys, monkeypatch):
    """Every declared target assessed: the reply names none as not assessed, and the run directory the evaluation writes
    (``results targets``) and the summary of ``project run`` state the same assessments, lists and status."""
    monkeypatch.delenv("OS_OPTICS_STATUS_CONTRACT", raising=False)
    document, _ = _example()
    evaluation = EvaluationSession().evaluate(document, outputs=["complete"])
    reply = json_api.evaluate_project(document, ["complete"])
    requirements = evaluation.requirements()
    assert reply["requirements"] == requirements and requirements["not_assessed"] == []
    assert requirements["met"] == ["focus_position"] and requirements["unmet"] == ["spot_radius"]
    run_dir = tmp_path / "run"
    evaluation.write_run(run_dir)
    assert main(["results", "targets", str(run_dir)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert {row["target_id"]: row["assessment"] for row in exported["targets"]} == {"focus_position": "MET", "spot_radius": "UNMET"}
    assert "not_evaluated" not in exported
    assert exported["outcome"]["requirements"] == {k: v for k, v in requirements.items() if k != "not_assessed"}
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--progress", "none"]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["requirements"] == {k: v for k, v in requirements.items() if k != "not_assessed"}
    assert summary["status"] == reply["status"] == evaluation.status
    assert main(["project", "evaluate", str(path), "--progress", "none", "--outputs", "complete"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["requirements"] == requirements and printed["exit_contract"] == "requirements"
