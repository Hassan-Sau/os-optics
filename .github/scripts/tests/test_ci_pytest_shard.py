"""Tests for .github/scripts/ci_pytest_shard.py: shards of a synthetic suite run every selected test exactly once."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_pytest_shard.py"
REPO = SCRIPT.parents[2]
_spec = importlib.util.spec_from_file_location("ci_pytest_shard", SCRIPT)
shard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shard
_spec.loader.exec_module(shard)


def suite(root: Path) -> Path:
    tests = root / "tests"
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
                                         'markers = ["slow: a marker the shards filter on"]\n', encoding="utf-8")
    for index in range(7):
        body = "".join(f"def test_{index}_{case}():\n    pass\n\n" for case in range(index + 1))
        body += "import pytest\n\n@pytest.mark.slow\ndef test_marked():\n    pass\n"
        (tests / f"test_file{index}.py").write_text(body, encoding="utf-8")
    return root


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)


def passed(output: str) -> set[str]:
    return set(re.findall(r"^(tests/\S+::\S+) PASSED", output, re.M))


@pytest.mark.parametrize("total", [1, 2, 3, 5, 6])
def test_shards_run_every_selected_test_exactly_once(tmp_path, total):
    root = suite(tmp_path)
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"tests/test_file6.py": 50.0, "tests/test_file0.py": 0.1}), encoding="utf-8")
    seen: list[str] = []
    for index in range(1, total + 1):
        result = run(root, f"--shard={index}/{total}", f"--weights={weights}", "--", "-v", "-m", "not slow",
                     "-p", "no:cacheprovider")
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"shard {index}/{total}:" in result.stdout
        seen += sorted(passed(result.stdout))
    assert len(seen) == len(set(seen)) == sum(range(1, 8))  # every unmarked test once, no marked one
    assert not any(name.endswith("::test_marked") for name in seen)


def test_a_test_belongs_to_the_most_specific_unit_the_weights_name():
    weights = {"tests/t.py::test_f[2]": 1.0, "tests/t.py::test_g": 1.0, "tests/t.py::K::test_h": 1.0, "tests/t.py": 1.0}
    assert shard.unit_of("tests/t.py::test_f[2]", weights) == "tests/t.py::test_f[2]"
    assert shard.unit_of("tests/t.py::test_f[3]", weights) == "tests/t.py"
    assert shard.unit_of("tests/t.py::test_g[a-b]", weights) == shard.unit_of("tests/t.py::test_g", weights) == "tests/t.py::test_g"
    assert shard.unit_of("tests/t.py::K::test_h[x]", weights) == "tests/t.py::K::test_h"
    assert shard.unit_of("tests/t.py::test_other", weights) == "tests/t.py"
    assert shard.unit_of("tests/u.py::test_f[2]", weights) == "tests/u.py"
    # file weights alone (the engine's) keep every file one unit
    assert shard.unit_of("tests/t.py::test_f[2]", {"tests/t.py": 1.0}) == "tests/t.py"


def test_weights_can_run_a_test_function_or_one_parametrized_test_apart_from_its_file(tmp_path):
    """The engine-pyodide shards: one file whose parts each run apart; every selected test still runs exactly once."""
    root = suite(tmp_path)
    (root / "tests" / "test_parts.py").write_text(
        "import pytest\n\n@pytest.mark.parametrize('part', range(4))\ndef test_part(part):\n    pass\n\n"
        "@pytest.mark.parametrize('n', range(2))\ndef test_whole(n):\n    pass\n\ndef test_rest():\n    pass\n", encoding="utf-8")
    weights = tmp_path / "weights.json"
    heavy = ["tests/test_parts.py::test_part[0]", "tests/test_parts.py::test_part[1]", "tests/test_parts.py::test_whole"]
    weights.write_text(json.dumps({**dict.fromkeys(heavy, 100.0), "tests/test_parts.py": 100.0}), encoding="utf-8")
    shards = []
    for index in range(1, 5):
        result = run(root, f"--shard={index}/4", f"--weights={weights}", "--", "-v", "-m", "not slow", "-p", "no:cacheprovider")
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"shard {index}/4: " in result.stdout and "3 of them tests run apart from their file" in result.stdout
        shards.append(passed(result.stdout))
    seen = [name for names in shards for name in names]
    assert len(seen) == len(set(seen)) == sum(range(1, 8)) + 7  # the synthetic files' tests and all seven of test_parts.py
    parts = [names for names in shards if any(name.startswith("tests/test_parts.py") for name in names)]
    assert len(parts) == 4  # each heavy unit in a shard of its own
    by_shard = [{name for name in names if name.startswith("tests/test_parts.py")} for names in parts]
    assert {"tests/test_parts.py::test_whole[0]", "tests/test_parts.py::test_whole[1]"} in by_shard  # a function stays whole
    assert {"tests/test_parts.py::test_part[2]", "tests/test_parts.py::test_part[3]", "tests/test_parts.py::test_rest"} in by_shard


def test_partition_is_deterministic_and_balanced():
    seconds = {"a": 10.0, "b": 9.0, "c": 5.0, "d": 4.0, "e": 1.0}
    assert shard.partition(seconds, 2) == shard.partition(dict(reversed(seconds.items())), 2)
    loads = [sum(seconds[name] for name in part) for part in shard.partition(seconds, 2)]
    assert sorted(loads) == [14.0, 15.0]


def test_unmeasured_files_are_estimated_from_their_test_count():
    est = shard.estimate({"known.py": 3, "new.py": 4}, {"known.py": 30.0})
    assert est == {"known.py": 30.0, "new.py": 4 * shard.SECONDS_PER_UNMEASURED_TEST}


def test_record_merges_measured_seconds(tmp_path):
    root = suite(tmp_path)
    record = tmp_path / "timings.json"
    record.write_text(json.dumps({"tests/gone.py": 9.0}), encoding="utf-8")
    result = run(root, "--shard=1/1", f"--record={record}", "--", "-q", "-p", "no:cacheprovider")
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = json.loads(record.read_text(encoding="utf-8"))
    assert set(recorded) == {"tests/gone.py", *(f"tests/test_file{i}.py" for i in range(7))}


@pytest.mark.parametrize("args", [["--shard=3/2", "--"], ["--shard=0/2", "--"], ["--shard=1/2"],
                                  ["--shard=1/2", "--bogus=1", "--"], ["--weights=x", "--"], ["--shard=1/2", "x", "--"]])
def test_bad_arguments_are_usage_errors(tmp_path, args):
    assert run(tmp_path, *args).returncode == 4


def test_the_engine_weights_are_seconds_per_engine_test_file_or_heavy_test():
    """Units are test files, or a heavy test function / parametrized case of one run apart from its file. A stale entry only
    skews the balance (a deleted file or test is simply absent from the collection)."""
    weights = json.loads((REPO / ".github" / "ci-shard-weights" / "engine.json").read_text(encoding="utf-8"))
    assert len(weights) > 100
    assert all(name.split("::")[0].startswith("tests/") and name.split("::")[0].endswith(".py") and value > 0
               for name, value in weights.items())
    for name in weights:
        if "::" in name:
            assert name.split("::")[0] in weights, name  # the rest of its file stays a unit of its own


def test_the_pyodide_weights_name_the_rest_of_the_file_and_each_part_of_the_semantic_comparison():
    """engine-pyodide (.github/workflows/ci.yml): its one Pyodide file runs as the rest of the file plus one unit per part
    (tests/test_packaging_and_pyodide.py, JSON_API_PARTS) of the json_api comparison."""
    weights = json.loads((REPO / ".github" / "ci-shard-weights" / "engine-pyodide.json").read_text(encoding="utf-8"))
    module = (REPO / "tests" / "test_packaging_and_pyodide.py").read_text(encoding="utf-8")
    parts = int(re.search(r"^JSON_API_PARTS = (\d+)$", module, re.M)[1])
    test = "tests/test_packaging_and_pyodide.py::test_the_browser_json_api_is_semantically_identical_to_native"
    assert f"def {test.split('::')[1]}(pyodide_run, tmp_path, part)" in module
    assert set(weights) == {"tests/test_packaging_and_pyodide.py",
                            *(f"{test}[part{part}of{parts}]" for part in range(1, parts + 1))}
    assert all(value > 0 for value in weights.values())
