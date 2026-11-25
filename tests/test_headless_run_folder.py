"""A headless JSON host can reuse a run directory after stops and refusals.

A failed attempt preserves the previous valid run; a successful new run replaces it.
The host owns its directory and needs no browser worker or special cleanup.
"""

from __future__ import annotations

from pathlib import Path

from composition_helpers import laser, power_target, project_of, screen
from os_optics.local import json_api
from os_optics.local.artifacts import verify_run_artifacts


def design(offset: float, sources: int = 10) -> dict:
    """Ten sources whose run files exceed a 1 MiB artifact limit."""
    return project_of([laser(f"S{i}", (-100.0, -9.0 + 2.0 * i + offset, 0.0)) for i in range(sources)],
                      [screen("P", (100.0, 0.0, 0.0), radius_mm=20.0), screen("Q", (150.0, 0.0, 0.0), radius_mm=20.0)],
                      [power_target("t-P", "P", 1e-4)])


def refused(offset: float) -> dict:
    document = design(offset, sources=3)
    document["targets"][0]["observation_id"] = "NO_SUCH_PLANE"
    return document


def headless_run(run_dir: Path, document: dict, limits: dict | None = None) -> dict:
    reply, files = json_api.call_method("run_project", {"document": document, "output_dir": str(run_dir),
                                                       **({"resource_limits": limits} if limits else {})})
    if reply.get("ok"):
        assert files and "manifest.json" in files and reply["verify"] == []
        assert verify_run_artifacts(run_dir) == ()
    return reply


LIMIT = {"max_artifact_mib": 1, "preflight": False}


def test_headless_run_after_resource_stop_preserves_and_then_replaces_the_earlier_run(tmp_path):
    run_dir = tmp_path / "run"
    first, second = design(0.0), design(0.5)
    assert headless_run(run_dir, first)["ok"]
    earlier = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    stopped = headless_run(run_dir, second, LIMIT)
    assert stopped["error"]["code"] == "RUN_STOPPED" and stopped["stopped"]["stop"]["cause"] == "ARTIFACT_LIMIT"
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != "run_stopped.json"} == earlier
    assert (run_dir / "run_stopped.json").exists() and verify_run_artifacts(run_dir) == ()
    assert headless_run(run_dir, second)["ok"]
    assert (run_dir / "project.json").read_bytes() != earlier["project.json"]
    assert not (run_dir / "run_stopped.json").exists()
    assert headless_run(run_dir, first, LIMIT)["error"]["code"] == "RUN_STOPPED"
    assert headless_run(run_dir, first)["ok"]


def test_headless_run_after_refusal_preserves_and_then_replaces_the_earlier_run(tmp_path):
    run_dir = tmp_path / "run"
    first, second = design(0.0, sources=3), design(0.5, sources=3)
    assert headless_run(run_dir, first)["ok"]
    earlier = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    refusal = headless_run(run_dir, refused(0.5))
    assert refusal["ok"] is False and refusal["error"]["code"] != "RUN_STOPPED"
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == earlier
    assert headless_run(run_dir, second)["ok"]
    assert (run_dir / "project.json").read_bytes() != earlier["project.json"]
    assert headless_run(run_dir, refused(0.0))["ok"] is False
    assert headless_run(run_dir, first)["ok"]
