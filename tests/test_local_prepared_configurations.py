"""A run of configurations prepares once what its configurations share (repeated preparation): every configuration is
lowered once -- before anything is traced, and that lowering is the one its run uses --, each element is compiled once for
every configuration (configurations change poses, sources and budgets, never which models the components use), and the
run's files stay those of each configuration's own run (``test_project_configurations``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from os_optics.local import LocalRunError, run_configured_project
from os_optics.local import lowering as lowering_module
from os_optics.local import run as run_module

ZOOM = Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json"


def zoom() -> dict:
    return json.loads(ZOOM.read_text(encoding="utf-8"))


def _spy(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list:
    calls: list = []
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


def test_each_configuration_is_lowered_once_and_each_element_compiled_once(monkeypatch):
    lowered = _spy(monkeypatch, run_module, "lower_project")
    compiled = _spy(monkeypatch, lowering_module, "compile_declared")
    run = run_configured_project(ZOOM.read_bytes())
    assert [c.configuration_id for c in run.configurations if c.run is not None] == ["wide", "mid", "tele"]
    assert len(lowered) == 3  # (one lowering per configuration: the up-front one, which its run uses)
    assert sorted(args[0] for args in compiled) == ["G1", "G2", "G3"]  # (the three lens groups, compiled for all three)
    for configuration in run.configurations:
        assert configuration.run.engine_input_hash == configuration.engine_input_hash


def _tele_groups_overlap() -> dict:
    """The zoom with G2 moved onto G1 in 'tele': that configuration's Project cannot be lowered (the solid check)."""

    document = zoom()
    g1 = next(c for c in document["components"] if c["instance_id"] == "G1")
    tele = next(c for c in document["configurations"] if c["configuration_id"] == "tele")
    next(p for p in tele["component_poses"] if p["instance_id"] == "G2")["nominal_pose"] = json.loads(json.dumps(g1["nominal_pose"]))
    return document


def test_a_configuration_that_cannot_be_lowered_is_never_traced_and_the_others_are_traced_once(monkeypatch):
    lowered = _spy(monkeypatch, run_module, "lower_project")
    evaluated = _spy(monkeypatch, run_module, "evaluate")
    document = _tele_groups_overlap()
    run = run_configured_project(document)
    tele = run.configuration("tele")
    assert tele.run is None and tele.engine_input_hash is None
    assert (tele.execution["outcome"], tele.execution["termination"]) == ("FAILED", "INVALID_INPUT")
    assert tele.problems and all(p.location.startswith("configuration tele") for p in tele.problems)
    assert [c.configuration_id for c in run.configurations if c.run is not None] == ["wide", "mid"]
    assert len(lowered) == 3 and len(evaluated) == 2  # (each lowered once up front; wide and mid traced once, tele never)
    # a selection of which no configuration can be lowered is refused before any trace
    with pytest.raises(LocalRunError) as caught:
        run_configured_project(document, configuration_ids=["tele"])
    assert len(evaluated) == 2
    assert caught.value.problems and all(p.location.startswith("configuration tele") for p in caught.value.problems)
