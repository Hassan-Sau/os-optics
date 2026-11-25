"""The supported Python range is declared once (``MIN_PYTHON`` = ``requires-python``), advertised consistently, exercised
by CI on its oldest version, and an unsupported interpreter gets an actionable message instead of an obscure error."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from os_optics._version import MIN_PYTHON

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
CI = ROOT / ".github" / "workflows" / "ci.yml"
NEWEST_TESTED = (3, 14)  # the Pyodide line


def test_requires_python_is_min_python():
    assert PROJECT["requires-python"] == ">={}.{}".format(*MIN_PYTHON)


def test_classifiers_list_exactly_the_supported_minor_versions():
    listed = sorted(tuple(map(int, m.groups())) for c in PROJECT["classifiers"]
                    if (m := re.fullmatch(r"Programming Language :: Python :: (\d+)\.(\d+)", c)))
    assert listed == [(3, minor) for minor in range(MIN_PYTHON[1], NEWEST_TESTED[1] + 1)]


def test_this_interpreter_is_supported():
    assert sys.version_info >= MIN_PYTHON


@pytest.mark.skipif(not CI.is_file(), reason="repository checkout only")
def test_ci_runs_the_engine_suite_on_the_oldest_and_newest_supported_python():
    text = CI.read_text(encoding="utf-8")
    job = text[text.index("\n  engine:\n"):]
    job = job[: job.index("\n\n  ", 1)]
    matrix = re.search(r"python: \[([^\]]*)\]", job).group(1)
    legs = [leg.strip().strip('"') for leg in matrix.split(",")]
    oldest = "{}.{}".format(*MIN_PYTHON)
    assert f"{oldest}-lowest" in legs and "{}.{}".format(*NEWEST_TESTED) in legs
    assert f"startsWith(matrix.python, '{oldest}')" in job


def test_an_unsupported_interpreter_gets_an_actionable_import_error(tmp_path):
    older = (MIN_PYTHON[0], MIN_PYTHON[1] - 1, 0, "final", 0)
    code = f"import sys; sys.version_info = {older!r}; import os_optics"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode != 0
    message = proc.stderr.strip().splitlines()[-1]
    assert message.startswith("ImportError: os-optics needs Python {}.{} or newer".format(*MIN_PYTHON))
    assert "this interpreter is Python {}.{}.0".format(*older[:2]) in message
    assert 'uv tool install "os-optics[protected]"' in message
