"""The selective-evaluation benchmark (``benchmarks/bench_evaluation.py``) and its fixtures stay usable.

The benchmark measures time and is not run here at its size. These tests run the whole benchmark (``main``) with one
repeat on small inputs -- the branching fixture, two-design candidate sets on the emitter fixture, the configurations and
the ``coherent_field`` analyses -- so its comparisons of fixtures, candidates and configurations and its report writer
run; they run one fresh-process command and check that an unexpected exit code stops the benchmark; they evaluate every
fixture it ships and check its candidate sets and that its comparison catches a difference. The fresh processes of
``project run`` and ``project evaluate`` and the analyses of the double Gauss fixtures, with their comparison of a
follow-up imaging, run only in a measurement.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from os_optics.local import EvaluationSession, run_optical_project
from os_optics.results import load_run

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_evaluation.py"


def _benchmark():
    spec = importlib.util.spec_from_file_location("bench_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _benchmark()

#: Two-design candidate sets on the emitter fixture (a declared biconvex lens ``L`` placed as component ``LENS``).
SMALL_CANDIDATES = {
    "pose": {"base": "emitter", "component": "LENS", "axis": 0, "offsets_mm": [0.0, 0.1], "what": "the lens moved"},
    "prescription": {"base": "emitter", "model": "L", "surface": "S1", "radii_mm": ["60", "61"], "what": "a radius varied"},
}


def test_the_whole_benchmark_runs_on_small_inputs_and_writes_its_report(tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(SMALL_CANDIDATES), encoding="utf-8")
    report_json, report_md = tmp_path / "bench.json", tmp_path / "bench.md"

    code = BENCH.main(["--repeats", "1", "--fixtures", "branching", "--candidates", str(candidates), "--cold-runs", "0",
                       "--analyses", "--analyses-fixtures", "coherent_field", "--json", str(report_json),
                       "--markdown", str(report_md)])

    assert code == 0  # 1 on any MismatchError the benchmark raises
    report = json.loads(report_json.read_text(encoding="utf-8"))
    (row,) = report["fixtures"]
    follow = row["follow_up"]
    assert follow["extension_equals_fresh"] and follow["run_files_equal"]
    assert "plane_bounds" in follow["extension_executed"] and "beam_trace" in follow["extension_reused"]
    assert "untraced_light" not in row["readings"]["executions"]
    assert row["write_s"]["n"] == 1 and sum(row["run_files_bytes"].values()) > 0
    assert 0 < row["readings"]["allocation_peak_bytes"] < row["complete"]["allocation_peak_bytes"]
    assert row["complete"]["execution"] == {"outcome": "COMPLETED", "termination": "COMPLETED", "exhausted": []}

    pose, prescription = report["candidates"]["pose"], report["candidates"]["prescription"]
    for loop in (pose, prescription):
        assert loop["candidates"] == 2 and loop["each_equals_its_run"]
        assert loop["session_evaluations_per_s"] > 0 and loop["session_per_run_optical_project"] > 0
        assert loop["run_optical_project_execution"] == ["COMPLETED"]  # (the runs the session loop is divided by)
    held = pose["compiled_elements_held"][0]
    assert held > 0
    assert pose["compiled_elements_held"] == prescription["compiled_elements_held"] == [held, held]
    assert pose["statistics"]["compiled_elements"] == held and pose["statistics"]["reused_elements"] == held
    assert prescription["statistics"]["compiled_elements"] == 2 * held and prescription["statistics"]["reused_elements"] == 0

    assert report["configurations"]["files_equal"]
    (analyses,) = report["analyses"]
    assert [(a["analysis"], a["complete"]) for a in analyses["analyses"]] == [("field", True), ("field, refinement false", True)]
    # (the outcome of the complete evaluation the analyses are run on, apart from each analysis document's own)
    assert analyses["complete_execution"] == {"outcome": "COMPLETED", "termination": "COMPLETED", "exhausted": []}

    text = report_md.read_text(encoding="utf-8")
    assert "| branching |" in text and "| COMPLETED |" in text
    assert "| pose: the lens moved | 2 |" in text and "| prescription: a radius varied | 2 |" in text
    assert f"| {held} / {held} |" in text
    assert "\n\nConfigurations (" in text and "\n\nAnalyses on the results" in text
    assert "| run_optical_project execution |" in text and "| analysis complete |" in text
    assert "The complete evaluation held: coherent_field COMPLETED." in text


def test_a_fresh_process_that_exits_otherwise_than_its_command_finishes_stops_the_benchmark():
    assert [(label, finished) for label, _arguments, finished in BENCH.COMMANDS] == [
        ("os-optics --version", (0,)),
        ("os-optics project run --out-dir D --pupil-rays NONE --ray-data summary", (0, 3)),  # 3: a PARTIAL run
        ("os-optics project evaluate --outputs readings", (0,)),
    ]
    (row,) = BENCH.fresh_processes("branching", 1, commands=BENCH.COMMANDS[:1])
    assert row["command"] == "os-optics --version" and row["seconds"]["n"] == 1

    expected_partial = ("os-optics --version", ("--version",), (3,))
    with pytest.raises(RuntimeError, match="--version exited 0"):
        BENCH.fresh_processes("branching", 1, commands=(expected_partial,))


def test_the_field_of_a_readings_evaluation_equals_the_complete_run_s():
    project = BENCH.coherent_field_project()
    readings = EvaluationSession().evaluate(project, outputs=("readings",))
    complete = load_run(run_optical_project(project))

    assert readings.result.evaluation is not None
    for request in (None, {"refinement": False}):
        assert BENCH._comparable(readings.results().field("P", request)) == BENCH._comparable(complete.field("P", request))


@pytest.mark.parametrize("name", BENCH.FIXTURES)
def test_every_fixture_evaluates_its_readings_without_the_plane_bounds(name):
    session = EvaluationSession()
    evaluation = session.evaluate(BENCH.document(name), outputs=("readings",))

    executions = session.statistics()["executions"]
    assert "untraced_light" not in executions and "plane_bounds" not in executions
    monitors = evaluation.results().monitors()
    assert monitors and all(m["quantity_state"] in ("NOT_COMPUTED", "OUT_OF_MODEL_SCOPE") for m in monitors)


def test_the_pose_set_moves_one_component_of_its_base():
    spec = json.loads(BENCH.CANDIDATES.read_text(encoding="utf-8"))["pose"]
    base = BENCH.document(spec["base"])
    candidates = BENCH.candidate_documents(spec)

    assert len(candidates) == len(spec["offsets_mm"]) > 1
    position = next(c for c in base["components"] if c["instance_id"] == spec["component"])["nominal_pose"]["position_mm"]
    for offset, candidate in zip(spec["offsets_mm"], candidates, strict=True):
        moved = next(c for c in candidate["components"] if c["instance_id"] == spec["component"])
        assert moved["nominal_pose"]["position_mm"][spec["axis"]] == position[spec["axis"]] + offset
        rest = [c for c in candidate["components"] if c["instance_id"] != spec["component"]]
        assert rest == [c for c in base["components"] if c["instance_id"] != spec["component"]]
        assert candidate["optical_models"] == base["optical_models"]


def test_the_prescription_set_changes_one_radius_of_its_base():
    spec = json.loads(BENCH.CANDIDATES.read_text(encoding="utf-8"))["prescription"]
    base = BENCH.document(spec["base"])
    candidates = BENCH.candidate_documents(spec)

    assert len(candidates) == len(set(spec["radii_mm"])) > 1
    for radius, candidate in zip(spec["radii_mm"], candidates, strict=True):
        assert candidate["components"] == base["components"]
        expected = copy.deepcopy(base)
        model = next(m for m in expected["optical_models"] if m["model_id"] == spec["model"])
        next(s for s in model["model"]["surfaces"] if s["surface_id"] == spec["surface"])["radius_mm"] = radius
        assert candidate == expected
    assert spec["radii_mm"][len(candidates) // 2] == next(
        s["radius_mm"] for m in base["optical_models"] if m["model_id"] == spec["model"]
        for s in m["model"]["surfaces"] if s["surface_id"] == spec["surface"])  # (the base design is the middle candidate)


def test_the_retained_check_compares_a_readings_evaluation_with_a_run_and_catches_a_difference():
    project = BENCH.document("emitter")  # its plane holds sampled light and no arrival
    readings = EvaluationSession().evaluate(project, outputs=("readings",)).results()
    run = load_run(run_optical_project(project))
    BENCH.check_retained(readings, run, "emitter")

    class Altered:
        def monitors(self):
            return run.monitors()

        def monitor(self, plane):
            document = copy.deepcopy(run.monitor(plane))
            document["sampled_light"][0]["power_W"] = 0.0 if document["sampled_light"][0]["power_W"] else 1.0
            return document

    assert readings.monitor("D")["sampled_light"], "the emitter fixture's plane holds sampled light"
    with pytest.raises(BENCH.MismatchError, match="sampled_light at D"):
        BENCH.check_retained(readings, Altered(), "emitter")
