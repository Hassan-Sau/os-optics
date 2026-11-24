"""Persisted budgets through editing, lowering, CLI, workspace and complete run artifacts."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from os_optics import NumericalConfig, builders as B
from os_optics.cli import main
from os_optics.contracts import ProjectSnapshot
from os_optics.local import json_api as J
from os_optics.local.project_ops import ProjectOpError, apply_project_ops
from os_optics._vendor.shared_contracts.optical_design.compute_budget import ComputeBudgetV1

from composition_helpers import ALONG_X, build, laser, place, project_of, screen
from helpers import run


def _project():
    return project_of([laser("S")], [screen("screen", (100, 0, 0))])


@pytest.mark.parametrize("count", [63, 64, 65, 160])
def test_exact_interaction_allowance_reaches_downstream_monitor(count):
    src = B.laser("S", wavelength_nm=780, power_W=1e-3, waist_radius_mm=0.5,
                  position=(-1, 0, 0), direction=(1, 0, 0))
    lenses = tuple(B.thin_lens(f"L{i}", focal_length_mm=1e9, clear_radius_mm=50,
                               position=(float(i), 0, 0)) for i in range(count))
    obs = B.observation_plane("D", position=(float(count), 0, 0), normal=(1, 0, 0), radius_mm=50)
    project = ProjectSnapshot(components=(src, *lenses), observations=(obs,))
    result = run(project, numerical=NumericalConfig(max_interactions_per_path=count))
    assert result.observations[0].total_power_W == pytest.approx(1e-3)
    assert result.outcome.execution.complete
    assert result.paths[0].termination == "ESCAPED"
    assert sum(entry.endswith(":thin_lens") for entry in result.paths[0].lineage) == count
    assert not any(i.code.value == "MAX_INTERACTIONS" for i in result.issues)
    limited = run(project, numerical=NumericalConfig(max_interactions_per_path=count - 1))
    assert not limited.outcome.execution.complete
    assert limited.paths[0].termination == "MAX_INTERACTIONS"
    assert not limited.observations[0].arrivals


@pytest.mark.parametrize("field", ["max_interactions_per_path", "max_paths", "max_trace_steps", "max_interference_pairs",
                                   "max_return_analysis_work", "pupil_ring_count", "pupil_points_per_ring",
                                   "max_scatter_origins", "max_mesh_patches"])
def test_local_budgets_have_representation_limits_not_arbitrary_capacity_caps(field):
    assert ComputeBudgetV1.model_validate({field: 2**53 - 1}).overrides()[field] == 2**53 - 1
    assert getattr(NumericalConfig(**{field: 2**53 - 1}), field) == 2**53 - 1
    for invalid in (0, -1, True, 1.5, "512", 2**53):
        with pytest.raises(ValidationError):
            ComputeBudgetV1.model_validate({field: invalid})
        with pytest.raises(ValidationError):
            NumericalConfig(**{field: invalid})
        with pytest.raises(ProjectOpError):
            apply_project_ops(_project(), [{"op": "set_compute_budget", "compute_budget": {field: invalid}}])


def test_configure_inspect_reset_and_identity():
    project = _project()
    baseline = J.check_project(project)
    assert baseline["ok"]
    op = {"op": "set_compute_budget", "compute_budget": {"max_paths": 100000, "max_trace_steps": 2000000}}
    edited = J.update_project(project, [op])
    assert edited["ok"], edited
    assert edited["project"]["compute_budget"] == op["compute_budget"]
    assert edited["compute_budget"] == {"max_interactions_per_path": 64, "max_paths": 100000,
                                         "max_trace_steps": 2000000, "max_interference_pairs": 256,
                                         "max_return_analysis_work": NumericalConfig().max_return_analysis_work,
                                         "pupil_ring_count": 8, "pupil_points_per_ring": 12, "max_scatter_origins": 64,
                                         "max_mesh_patches": 1024}
    assert edited["project_hash"] != baseline["project_hash"]
    assert edited["engine_input_hash"] != baseline["engine_input_hash"]
    assert not J.update_project(edited["project"], [op])["changed"]
    reset = J.update_project(edited["project"], [{"op": "set_compute_budget", "compute_budget": None}])
    assert reset["project_hash"] == baseline["project_hash"]
    assert reset["engine_input_hash"] == baseline["engine_input_hash"]
    assert "compute_budget" not in reset["project"]
    caps = J.capabilities()
    assert caps["compute_budget"]["defaults"] == baseline["compute_budget"]
    assert "set_compute_budget" in caps["project_ops"]["ops"]


def test_more_than_32_source_lines_are_limited_by_the_persisted_trace_budget():
    from os_optics.local import run_optical_project

    project = _project()
    source = project["components"][0]["source_model"]
    source.pop("wavelength_nm")
    source.pop("power_w")
    source["spectral_lines"] = [{"wavelength_nm": 700 + i, "power_w": 0.001} for i in range(33)]
    project["compute_budget"] = {"max_paths": 32}
    partial = run_optical_project(project)
    assert len(partial.result.paths) == 32 and not partial.result.outcome.execution.complete
    project["compute_budget"] = {"max_paths": 33}
    complete = run_optical_project(project)
    assert len(complete.result.paths) == 33 and complete.result.outcome.execution.complete
    assert complete.result.observations[0].total_power_W == pytest.approx(0.033)


def test_interaction_limited_run_artifacts_include_the_last_free_propagation(tmp_path):
    from os_optics.local import run_optical_project
    from os_optics.local.artifacts import verify_run_artifacts

    lens = build("lens.ideal_thin", {"focal_length_mm": "1000", "clear_aperture_diameter_mm": "20"})
    project = project_of([laser("S"), place(lens, "L1", (20, 0, 0), ALONG_X),
                          place(lens, "L2", (40, 0, 0), ALONG_X)], [screen("D", (100, 0, 0))], builds=[lens])
    project["compute_budget"] = {"max_interactions_per_path": 1}
    result = run_optical_project(project, output_dir=tmp_path)
    assert verify_run_artifacts(tmp_path) == ()
    path = result.result.paths[0]
    assert path.termination == "MAX_INTERACTIONS"
    assert sum(event.endswith(":thin_lens") for event in path.lineage) == 1
    assert path.segments[-1].end_mm == (40.0, 0.0, 0.0)
    assert result.ray_paths.paths[0].chief_points_mm[-1] == path.segments[-1].end_mm
    assert result.ray_paths.paths[0].chief_opl_mm[-1] == path.optical_path_length_mm
    assert not result.result.observations[0].arrivals
    quantity = json.loads((tmp_path / 'local_validation.json').read_text())['simulation']['observation_quantities'][0]
    assert quantity['quantity_state'] == 'UNKNOWN' and quantity['power_upper_bound_W'] is None


def test_cli_workspace_edit_run_artifacts_and_freshness(tmp_path, capsys):
    project_path, workspace_path = tmp_path / "project.json", tmp_path / "workspace.json"
    ops_path, out = tmp_path / "ops.json", tmp_path / "run"
    project_path.write_text(json.dumps(_project()), encoding="utf-8")
    ops_path.write_text(json.dumps([{"op": "set_compute_budget", "compute_budget": {
        "max_interactions_per_path": 800, "max_paths": 1024, "max_trace_steps": 12000, "max_interference_pairs": 1024}}]), encoding="utf-8")
    assert main(["project", "update", str(project_path), "--ops", str(ops_path), "-o", str(project_path)]) == 0
    edited = json.loads(capsys.readouterr().out)
    assert edited["compute_budget"]["max_paths"] == 1024
    assert main(["project", "check", str(project_path)]) == 0
    assert json.loads(capsys.readouterr().out)["compute_budget"] == edited["compute_budget"]
    assert main(["workspace", "init", str(project_path), "-o", str(workspace_path)]) == 0
    capsys.readouterr()
    assert main(["workspace", "run", str(workspace_path), "--out-dir", str(out)]) == 0
    capsys.readouterr()
    request = json.loads((out / "engine_request.json").read_text())
    result = json.loads((out / "simulation_result.json").read_text())
    from os_optics.errors import BUDGET_NAMES
    for key, val in edited["compute_budget"].items():
        assert request["configuration"]["numerical"][key] == val
        assert result["manifest"]["numerical_tolerances"][key] == val
        if key in BUDGET_NAMES:  # (the pupil sampling is a resolution setting, not a budget a trace exhausts)
            assert result["outcome"]["execution"]["budgets"]["limits"][key] == val
    from os_optics.local.artifacts import verify_run_artifacts
    assert verify_run_artifacts(out) == ()
    assert J.run_freshness(json.loads(project_path.read_text()), str(out))["freshness"] == "CURRENT"
    ops_path.write_text(json.dumps([{"op": "set_compute_budget", "compute_budget": {"max_paths": 2048}}]), encoding="utf-8")
    assert main(["workspace", "update-project", str(workspace_path), "--ops", str(ops_path), "--op-id", "budget"]) == 0
    capsys.readouterr()
    workspace = json.loads(workspace_path.read_text())
    assert workspace["project"]["compute_budget"] == {"max_paths": 2048}
    status = J.workspace_status(workspace)
    assert status["ok"] and status["status"]["current_validation"] is None
    assert main(["workspace", "update-project", str(workspace_path), "--ops", str(ops_path), "--op-id", "budget"]) == 0
    assert json.loads(capsys.readouterr().out)["replayed"]
    # A refused edit preserves both workspace bytes and its previous run evidence.
    before = workspace_path.read_bytes()
    ops_path.write_text('[{"op":"set_compute_budget","compute_budget":{"max_paths":0}}]', encoding="utf-8")
    assert main(["workspace", "update-project", str(workspace_path), "--ops", str(ops_path)]) == 65
    assert workspace_path.read_bytes() == before


def test_every_statement_of_the_budget_set_names_every_engine_budget(capsys) -> None:
    # the budget set grew to five (max_return_analysis_work); the op catalog, its module text and 'project update --help'
    # name every one and no longer count "four"
    from os_optics.errors import BUDGET_NAMES
    from os_optics.local import project_ops

    spec = project_ops.PROJECT_OPS["set_compute_budget"]
    with pytest.raises(SystemExit):
        main(["project", "update", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    for text in (spec["summary"] + " " + spec["fields"]["compute_budget"], project_ops.__doc__, help_text):
        assert all(name in text for name in BUDGET_NAMES), text
        assert "four budgets" not in text
    assert f"all {('one', 'two', 'three', 'four', 'five', 'six')[len(BUDGET_NAMES) - 1]} budgets" in spec["summary"]
