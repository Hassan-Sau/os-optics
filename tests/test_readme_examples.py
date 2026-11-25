"""Every Python code block in README.md is executed (documented examples must be real, runnable fixtures)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from os_optics.config import NumericalConfig

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = re.findall(r"```python\n(.*?)```", (ROOT / "docs" / "engine" / "README.md").read_text(encoding="utf-8"), flags=re.DOTALL)


def test_readme_has_examples():
    assert len(BLOCKS) >= 2


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_python_block_runs(index, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)  # the examples must work from any directory, as after a PyPI install
    if "validate_assembly_layout" in BLOCKS[index]:  # the assembly example reads ./assembly-preview.zip
        reference = ROOT / "tests" / "fixtures" / "assembly_preview" / "reference.assembly-preview.zip"
        (tmp_path / "assembly-preview.zip").write_bytes(reference.read_bytes())
        monkeypatch.chdir(tmp_path)
    if "open_preview_delivery" in BLOCKS[index]:  # the delivery example reads ./preview-delivery.json and ./detail-delivery.json
        from delivery_helpers import sealed_reference_deliveries

        preview_delivery, detail_delivery = sealed_reference_deliveries()
        (tmp_path / "preview-delivery.json").write_text(json.dumps(preview_delivery), encoding="utf-8")
        (tmp_path / "detail-delivery.json").write_text(json.dumps(detail_delivery), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
    server = None
    if "OpenSciClient" in BLOCKS[index]:  # the commercial example runs against a local stand-in of the service
        from commerce_stub import AGENT_SCOPES, StubServer, StubState, credential_for
        from delivery_helpers import sealed_reference_deliveries

        from os_optics.assembly import load_assembly_preview

        reference = ROOT / "tests" / "fixtures" / "assembly_drawing" / "reference.assembly-preview.zip"
        state = StubState(resolution_id=load_assembly_preview(reference).manifest.binding.commercial_resolution_id,
                          deliveries=sealed_reference_deliveries(), confirm_on_report=True)
        state.add_token("agent", AGENT_SCOPES)
        server = StubServer(state).__enter__()
        monkeypatch.setenv("OPENSCI_API_URL", server.url)
        monkeypatch.setenv("OPENSCI_TOKEN", credential_for("agent"))
    namespace: dict = {}
    try:
        exec(compile(BLOCKS[index], f"README.md[{index}]", "exec"), namespace)  # noqa: S102 - trusted repository documentation
    finally:
        if server is not None:
            server.__exit__(None, None, None)
    out = capsys.readouterr().out
    if "OpenSciClient" in BLOCKS[index]:
        assert namespace["order"]["order_number"] and namespace["validation"].status.value in ("PASS", "WARNING")
        kinds = [s["artifact_kind"] for s in state.artifact_sets.values()]
        assert sorted(kinds) == ["ASSEMBLY_DRAWING", "ASSEMBLY_LOCAL_VALIDATION"]
        return
    if "validate_assembly_layout" in BLOCKS[index]:
        result = namespace["result"]
        assert out.splitlines()[0] == "WARNING NO_GROSS_INTERFERENCE_FOUND"
        assert [c.status.value for c in result.checks].count("PASS") == len(result.checks) - 1
    elif "open_preview_delivery" in BLOCKS[index]:
        kinds = [line.split()[0] for line in out.splitlines()]
        assert kinds == ["ASSEMBLY_LOCAL_VALIDATION", "ASSEMBLY_DRAWING"]
        assert [name for name, _ in namespace["fields"]] == ["manifest", "file", "file", "file", "file"]
        assert list(tmp_path.iterdir()) and {p.name for p in tmp_path.iterdir()} == {"preview-delivery.json", "detail-delivery.json"}
    elif "load_run" in BLOCKS[index]:
        lines = out.splitlines()
        assert lines[0] == "['crossing_plane']"
        assert [line.split()[0] for line in lines[1:3]] == ["SRC#0", "SRC#0.0"]
        assert lines[3] == "OK True" and lines[4].startswith("0.001 [") and "'ESCAPED'" in lines[4]
        assert (tmp_path / "arrivals.csv").read_bytes() == namespace["results"].export("arrivals", "csv")
    elif "paraxial_analysis" in BLOCKS[index]:  # the thick-lens afocal spacing of the beam_expander example's lenses
        gap, position = out.splitlines()[0].split(" ", 1)
        solution = namespace["solution"]
        assert float(gap) == pytest.approx(solution["current_vertex_gap_mm"], abs=1e-3)  # the example is spaced afocally (to 1 um)
        assert json.loads(position) == pytest.approx([round(v, 3) for v in solution["position_mm"]])
        assert out.splitlines()[1] == "-2.0"
    elif "optical_models" in BLOCKS[index]:  # declared optical models: no catalog
        run = namespace["run"]
        assert run.status.value == "PASS" and run.model_scope_issues == () and run.binding_problems() == ()
        assert out.splitlines()[0] == "PASS None ['my-lens', 'pinhole']"
    elif "set_compute_budget" in BLOCKS[index]:
        expected = {"max_interactions_per_path": 800, "max_paths": 4096,
                    "max_trace_steps": 12000, "max_interference_pairs": 1024}
        # (the reply lists every effective budget: the return-light analysis budget at its default too)
        assert namespace["edited"]["compute_budget"] == {
            **expected, "max_return_analysis_work": NumericalConfig().max_return_analysis_work,
            "pupil_ring_count": NumericalConfig().pupil_ring_count, "pupil_points_per_ring": NumericalConfig().pupil_points_per_ring,
            "max_scatter_origins": NumericalConfig().max_scatter_origins, "max_mesh_patches": NumericalConfig().max_mesh_patches}
        assert namespace["project"]["compute_budget"] == expected
    elif "set_numerical_settings" in BLOCKS[index]:
        assert namespace["edited"]["numerical_settings"]["power_epsilon_W"] == 1e-20
        assert namespace["edited"]["project"]["numerical_settings"] == {"power_epsilon_W": 1e-20}
        assert out.splitlines()[0] == "1e-20"
    elif "basis_of" in BLOCKS[index]:  # "Read the result": the four answers and a value's basis
        lines = out.splitlines()
        assert lines[0] == "COMPLETED ('expanded_waist',)"
        assert lines[1].startswith("expanded_waist MET ") and lines[1].endswith(" mm None")  # a model value: no entry
        state, power, upper = lines[2].split()
        assert (state, power, upper) == ("UNKNOWN", "None", "None")
        assert lines[3] == "None"
        assert float(lines[4]) == pytest.approx(namespace["plane"].arrivals[0].beam.gaussian.beam_radius_x_mm)
    elif "run_optical_project" in BLOCKS[index]:
        run = namespace["run"]
        assert run.status.value == "UNKNOWN" and run.model_scope_issues == () and run.binding_problems() == ()
        assert "BULK_LOSS_NOT_ESTABLISHED" in {i.code.value for i in run.result.issues}
        assert [(t.target_id, t.status.value) for t in run.result.targets] == [("expanded_waist", "PASS")]
        assert run.result.observations[0].arrivals[0].beam.power_W is None
        lines = out.splitlines()
        assert lines[0].split()[1].isdigit() and lines[0].endswith(" products")  # catalog version, product count
        assert lines[2].startswith("UNKNOWN [('expanded_waist', 'PASS', ") and lines[3].startswith("UNKNOWN ")
    elif "EvaluationSession" in BLOCKS[index]:  # a selective evaluation, its refusal of what it did not compute, its extension
        lines = out.splitlines()
        assert lines[0] == "('scene', 'beam_trace', 'secondary_rays', 'plane_readings', 'requirements', 'ray_paths', 'model_scope')"
        assert lines[2] == "NOT_COMPUTED" and lines[3].startswith("NOT_COMPUTED ") and "'power_ledger'" in lines[3]
        assert lines[4] == "1" and namespace["run"].local_validation is not None and namespace["full"].complete
    else:
        result = namespace["result"]
        assert result.status.value == "PASS"
        assert result.targets[0].value < 0.05  # focused beam radius (mm) at the observation plane
        assert out.startswith("PASS ")
