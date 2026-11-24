"""The release-time parity script (``scripts/release_parity.py``) checks the recorded retained-DOF / plate / 32-line-source scenarios on an installed
os-optics. Here it runs against these sources, so the script stays true to the Engine it will be
pointed at, and its Projects are the ones the feature tests use (``test_retained_dof_limits.py``,
``test_continuous_spectrum.py``): a release check of other documents would prove nothing about the accepted behaviour."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import test_continuous_spectrum as spectrum_tests
import test_retained_dof_limits as dof_tests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_parity.py"


def _script():
    spec = importlib.util.spec_from_file_location("release_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_checks_the_projects_of_the_feature_tests():
    parity = _script()
    assert parity.retained_dof_project() == dof_tests._retained_dof_project()
    assert parity.retained_dof_project(parity.BARE) == dof_tests._retained_dof_project(dofs=({"axis": "Tx"}, {"axis": "Rx"}, {"axis": "Ry"}))
    assert (parity.TX, parity.RX, parity.RY) == (dof_tests.TX, dof_tests.RX, dof_tests.RY)
    assert parity.free_space(parity.spectral_lines_project()) == spectrum_tests.free_space(spectrum_tests.spectral_lines_project())
    assert parity.DEFAULT_CATALOG.is_file()


def test_every_check_passes_on_these_sources(tmp_path):
    parity = _script()
    evidence = parity.run(parity.DEFAULT_CATALOG, allow_source=True, workdir=tmp_path)
    failed = [row for row in evidence["checks"] if not row["ok"]]
    assert not failed, json.dumps(failed, indent=1)
    scenarios = {row["scenario"] for row in evidence["checks"]}
    assert {"retained_dof", "plate", "spectral_lines"} <= scenarios
    # Projects that use no field of this release keep the hash the published 0.12.0 gives them
    assert evidence["project_hashes"] == parity.HASHES_0_12_0
    assert evidence["counts"]["checks"] >= 45
