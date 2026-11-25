"""The Engine build is identified once per run (repeated preparation): ``engine_build_id()`` hashes the package's
sources (tens of milliseconds) and takes no cache, so a run passes the value along -- its session's Engine state holds
it. A ``FLUORESCENCE`` sample's excitation trace (inside lowering) and the run's own evaluation share it, and so do the
configurations of a Project and its run directory's manifest. Every value is the build's own (no number changes)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import os_optics.analysis as analysis_module
import os_optics.local.evaluate as evaluate_module
import os_optics.manifest.build as build_module
import os_optics.validation.pipeline as pipeline_module
from os_optics.local import EvaluationSession, json_api, run_configured_project, run_optical_project
from os_optics.local.configurations import configurations_run_files

from test_weak_signal_chain import QE, chain_project, fluorescence

ZOOM = Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json"


@pytest.fixture
def identified(monkeypatch) -> Counter:
    """Where ``engine_build_id()`` is called from during the test, by module."""

    calls: Counter = Counter()
    real = build_module.engine_build_id
    for module in (pipeline_module, evaluate_module, analysis_module):
        def counted(module=module):
            calls[module.__name__] += 1
            return real()

        monkeypatch.setattr(module, "engine_build_id", counted)
    return calls


def _sample_project() -> dict:
    # (a fluorescent sample with few emitter rays: the excitation trace in lowering, then the run with the sample's emitter)
    return chain_project(fluorescence(sampling={"sample_count": 64, "seed": 1}), {"detector": {"quantum_efficiency": QE}})


def test_a_sample_run_identifies_the_build_once(identified):
    session = EvaluationSession()
    run = session.evaluate(_sample_project(), outputs=("complete",)).run()
    assert sum(identified.values()) == 1
    record = next(c.sample_emission for c in run.request.project.components if c.sample_emission is not None)
    assert record.excitation_power_W > 0.0
    assert run.result.manifest.engine_build_id == session.build_id == build_module.engine_build_id()
    # (the excitation trace's products are its own: the session's state holds the run's, invalidated by nothing)
    assert session.statistics()["invalidations"] == {} and session.statistics()["executions"]["beam_trace"] == 1


def test_a_sample_runs_files_and_evaluation_identify_the_build_once(identified):
    run_optical_project(_sample_project())
    assert sum(identified.values()) == 1
    identified.clear()
    reply = json_api.evaluate_project(_sample_project())
    assert reply["ok"] is True and sum(identified.values()) == 1


def test_a_run_of_configurations_identifies_the_build_once_for_its_runs_and_manifest(identified):
    run = run_configured_project(ZOOM.read_bytes())
    files = configurations_run_files(run)
    # (the comparison document, os_optics.analysis.document, takes the session's build id: one identification in all)
    assert sum(identified.values()) == 1
    assert run.comparison.identity.physical.engine_build_id == build_module.engine_build_id()
    built = {json.loads(files[f"configurations/{cid}/manifest.json"])["engine_run"]["engine_build_id"] for cid in ("wide", "mid", "tele")}
    assert built == {json.loads(files["manifest.json"])["engine"]["engine_build_id"]} == {build_module.engine_build_id()}


def test_an_analysis_of_an_evaluation_takes_the_build_id_of_its_session(identified):
    from os_optics.local import example_project

    evaluation = EvaluationSession().evaluate(example_project(), outputs=("complete",))
    assert sum(identified.values()) == 1
    ghosts = evaluation.results().ghosts("focal_plane")
    assert sum(identified.values()) == 1 and ghosts.identity.physical.engine_build_id == evaluation.session.build_id
