"""Machine-derived documentation must match the code (tolerance table) and the docs must reference only existing artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "engine"))

import generate_docs  # noqa: E402


def test_tolerance_table_matches_numerical_config():
    text = generate_docs.DOC.read_text(encoding="utf-8")
    assert generate_docs.render(text) == text, "run: python docs/engine/generate_docs.py"
    from os_optics import NumericalConfig

    for name in NumericalConfig.model_fields:
        if name != "config_version":
            assert f"`{name}`" in text


def test_material_finish_table_matches_the_glb_finishes():
    text = generate_docs.MATERIALS_DOC.read_text(encoding="utf-8")
    assert generate_docs.render_materials(text) == text, "run: python docs/engine/generate_docs.py"


def test_the_table_of_metrics_a_run_does_not_evaluate_matches_the_scope_registry():
    text = generate_docs.SCOPE_DOC.read_text(encoding="utf-8")
    assert generate_docs.render_scope(text) == text, "run: python docs/engine/generate_docs.py"


def test_docs_reference_existing_paths_only():
    """Backticked repository paths in the Engine docs (tests/..., docs/..., benchmarks/..., src/...) exist."""
    pattern = re.compile(r"`((?:tests|docs|benchmarks|src)/[A-Za-z0-9_./*-]+)`")
    missing = []
    for doc in sorted((ROOT / "docs" / "engine").rglob("*.md")):
        if doc.name == "OpenSci_Integrated_Development_Plan_v0.5.0.md":
            continue
        for ref in pattern.findall(doc.read_text(encoding="utf-8")):
            if "*" in ref:
                continue
            if not (ROOT / ref.rstrip("/")).exists():
                missing.append((doc.name, ref))
    assert missing == []


def test_readme_command_row_names_every_top_level_command(capsys):
    """The README's Public API row of the command line lists every top-level command of ``os-optics --help``."""
    import pytest

    from os_optics.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    commands = re.search(r"\{([a-z,-]+)\}", capsys.readouterr().out).group(1).split(",")
    row = next(line for line in (ROOT / "docs" / "engine" / "README.md").read_text(encoding="utf-8").splitlines()
               if line.startswith("| `os-optics doctor`"))
    listed = re.findall(r"`(?:os-optics )?([a-z-]+)`", row.split("|")[1])
    assert sorted(listed) == sorted(commands)


def test_readme_states_the_json_api_version_the_runtime_reports():
    """Every ``opensci.local_json_api`` version the docs state is the one ``capabilities()`` reports."""
    from os_optics.local.json_api import JSON_API_VERSION

    stated = []
    for doc in sorted((ROOT / "docs" / "engine").glob("*.md")):
        stated += [(doc.name, v) for v in re.findall(r"opensci\.local_json_api`? (\d+\.\d+)", doc.read_text(encoding="utf-8"))]
    assert stated, "the README states the JSON API version"
    assert [(name, v) for name, v in stated if v != JSON_API_VERSION] == []


def test_each_results_command_has_a_description_of_its_own():
    """``os-optics results CMD --help`` describes CMD: what it computes or reads and what it prints, in a few sentences, and
    how it reads the run; the overview of every command stays on ``results --help`` alone."""
    from os_optics.cli_results import READ_AS_STORED, RESULT_COMMAND_DESCRIPTIONS, RESULTS_DESCRIPTION, _described

    group_opening = RESULTS_DESCRIPTION.split(";")[0]
    for name, text in RESULT_COMMAND_DESCRIPTIONS.items():
        described = _described(name)
        assert group_opening not in described and len(described) < 1100, name
        assert described.endswith("Output is JSON."), name
        assert (READ_AS_STORED in described) is (name not in ("metrics", "columns")), name
