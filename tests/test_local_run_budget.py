"""Local run cost of a branching three-plate design: three 50:50 plate beamsplitters at normal incidence in series
(the fixture catalog models copied into the Project with an explicit ideal bulk-loss assumption; every pair an etalon)
are traced up to the run's trace budget, and the written run directory stays within the size the
budget documents (``NumericalConfig.max_trace_steps``: about 150 MB of run files). The summary alone tells an Agent the run
was truncated (one ``MAX_PATHS`` issue per exhausted phase) and still carries the model-scope findings of the plates."""

from __future__ import annotations

from os_optics.local import LocalValidationStatusV1
from os_optics.local.artifacts import verify_run_artifacts

from declared_helpers import declared_project
from local_helpers import AGAINST_X, ALONG_X, PLATE_BS, catalog_document, element, plane, project, run, scope_codes, source

#: The run-directory bound the trace budget documents (config.py: about 150 MB at the default max_trace_steps).
RUN_FILES_LIMIT_BYTES = 160 * 1024 * 1024


def _plate_stack() -> dict:
    plates = [element(f"P{k}", "splitter", PLATE_BS, (20.0 * k, 0.0, 0.0), ALONG_X if k % 2 == 0 else AGAINST_X) for k in range(3)]
    # Clone the synthetic catalog plate model into this test Project and explicitly assume ideal zero bulk loss.
    # The trace budget test isolates branching, not an unstated catalog glass transmission.
    document = declared_project(project([source(position=(-100.0, 0.0, 0.0), waist_mm=1.0), *plates],
                                        [plane("D", (150.0, 0.0, 0.0))]), catalog_document())
    for material in document.get("model_materials", []):
        material["absorption_coefficient_per_mm"] = "0"
    return document


def test_a_plate_stack_reports_each_exhausted_phase_and_its_run_directory_is_bounded(tmp_path):
    result = run(_plate_stack(), output_dir=tmp_path / "run")
    size = sum(path.stat().st_size for path in (tmp_path / "run").iterdir())
    assert size <= RUN_FILES_LIMIT_BYTES, size
    assert verify_run_artifacts(tmp_path / "run") == ()

    summary = result.summary()
    budget = [issue for issue in summary["issues"] if issue["code"] == "MAX_PATHS"]
    assert len(budget) == 2
    by_phase = {issue["numerical_context"]["phase"]: issue for issue in budget}
    assert set(by_phase) == {"primary", "untraced_light"}
    primary = by_phase["primary"]
    assert "the run's trace budget was reached" in primary["message"] and "% of the launched power" in primary["message"]
    execution = summary["outcome"]["execution"]
    assert execution["outcome"] == "PARTIAL" and not execution["complete"]  # (through the primary phase)
    assert execution["budgets"]["exhausted_by_phase"] == {
        **{phase: issue["numerical_context"]["budget"] for phase, issue in by_phase.items()},
        "interference": ["max_interference_pairs"]}
    assert any(issue["code"] == "MAX_BEAM_PAIRS" for issue in summary["issues"])
    # the plates are rated at 45 deg: the scope verdict still reaches the user
    assert "COATING_AOI_OUTSIDE_RANGE" in {code for _, code, _ in scope_codes(result)}
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
