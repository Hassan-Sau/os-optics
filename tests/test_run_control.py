"""Progress, cancellation and resource limits of a local run: ``os_optics.run_control`` (the Engine's
checkpoints) and ``os_optics.run_monitor.RunMonitor`` (the host's clock, memory and limits).

* A monitored run reports every phase it goes through, with deterministic counts, and its result is the unmonitored
  run's, byte for byte (the monitor never changes a number).
* A run stopped in any phase -- cancelled, interrupted (Ctrl-C), or at a memory or run-file limit -- has no result: it
  raises ``RunStopped`` whose ``StoppedRun`` record states execution CANCELLED, the phase, why, and every observation and
  target as unevaluated; its record is written into the run directory (``run_stopped.json``).
* The run directory rule: a run that completes replaces the run in its directory (its files staged under temporary
  names, then moved into place); an attempt that does not complete leaves the directory as it found it, so an earlier run
  there still verifies and its readers name the later attempt.
* A run whose pre-flight estimate (``local/resources.py``) exceeds a limit is refused before the trace.
* The command line (``project run --progress``, ``--max-memory-mib``, ``--max-artifact-mib``, Ctrl-C) and the JSON API
  (``run_project(resource_limits=...)``, a host's monitor) produce the same record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from os_optics.local import json_api, run_optical_project
from os_optics.local.artifacts import MANIFEST_FILE, STOPPED_FILE, run_artifact_files, verify_run_artifacts
from os_optics.local.configurations import configurations_run_files
from os_optics.local.resources import (OTHER_FILE_BYTES_PER_BEAM, PRE_TRACE_MEASURED_RATIO, RAY_DATA_BYTES_PER_BEAM,
                                            REFERENCES, beam_count, estimate_run_resources)
from os_optics.run_control import RUN_PHASES, RunStopped, monitored
from os_optics.run_monitor import RunMonitor

from composition_helpers import laser, power_target, project_of, screen

PHASES_RUN = ["lower", "preflight", "trace", "preflight", "emitters", "observation_results", "untraced_light", "bounds",
              "exposures", "interference", "targets", "result", "ray_paths", "model_scope", "record", "scene"]


def _project():
    return project_of([laser("A", (-100.0, 0.0, 0.0)), laser("B", (-100.0, 3.0, 0.0)), laser("C", (-100.0, -3.0, 0.0))],
                      [screen("P", (100.0, 0.0, 0.0)), screen("Q", (150.0, 0.0, 0.0), radius_mm=20.0)],
                      [power_target("t-P", "P", 1e-4)])


class _StopIn(RunMonitor):
    """Cancels the run as soon as it is in ``phase`` (on entering it, or at its first checkpoint)."""

    def __init__(self, phase: str, **kw):
        super().__init__(**kw)
        self.target = phase

    def enter(self, name, total, unit):
        if name == self.target:
            self.cancel(f"cancelled in {name}")
        super().enter(name, total, unit)


class _NoPreflight(RunMonitor):
    def preflight(self, estimate):  # (the write-time check alone)
        pass


def test_a_monitored_run_reports_each_phase_and_keeps_every_number(tmp_path):
    plain = run_optical_project(_project())
    events: list[dict] = []
    monitor = RunMonitor(events.append, interval_s=0.0)
    watched = run_optical_project(_project(), monitor=monitor, output_dir=tmp_path / "run")
    assert run_artifact_files(watched) == run_artifact_files(plain)  # (every file, byte for byte)
    assert watched.result.manifest.result_hash == plain.result.manifest.result_hash
    ended = [e for e in events if e["event"] == "phase_ended"]
    top = [e["phase"] for e in ended if len(e["phases"]) == 1]
    assert top == [p for p in PHASES_RUN if p in top] + ["write"] and set(PHASES_RUN) <= set(top)
    assert all(phase in RUN_PHASES for phase in monitor.completed)
    counts = {e["phase"]: (e["done"], e["total"], e["unit"]) for e in ended}
    assert counts["trace"][0] == len(plain.result.paths) and counts["trace"][2] == "beam paths"
    kept = sum(len(o.arrivals) for o in plain.result.observations)
    assert counts["observation_results"][:2] == (kept, kept)
    assert counts["scene"][:2] == (len(plain.ray_paths.paths),) * 2
    assert counts["write"][0] == sum((tmp_path / "run" / f.name).stat().st_size for f in (tmp_path / "run").iterdir())
    estimate = next(e["estimate"] for e in ended if e["phase"] == "preflight")
    assert estimate["beams"] == 3 and estimate["artifact_bytes"] > 0
    # (deterministic: the same checkpoints and counts on a second run)
    again: list[dict] = []
    run_optical_project(_project(), monitor=RunMonitor(again.append, interval_s=0.0))
    assert [(e["phase"], e["done"], e["total"]) for e in again if e["event"] == "phase_ended"] == \
           [(e["phase"], e["done"], e["total"]) for e in ended if e["phase"] != "write"]
    assert verify_run_artifacts(tmp_path / "run") == () and not (tmp_path / "run" / STOPPED_FILE).exists()


@pytest.mark.parametrize("phase", ["lower", "trace", "observation_results", "untraced_light", "bounds", "exposures", "result", "ray_paths",
                                   "interference", "targets", "model_scope", "record", "scene", "write"])
def test_a_run_cancelled_in_any_phase_has_no_result(tmp_path, phase):
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)  # (an earlier run in the directory)
    earlier = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=_StopIn(phase), output_dir=out)
    record = stopped.value.stopped
    assert record.stop.phase == phase and record.stop.cause == "CALLER" and record.stop.message == f"cancelled in {phase}"
    assert record.outcome.execution.outcome.value == "CANCELLED" and record.outcome.execution.termination.value == "CANCELLED"
    assert record.outcome.execution.complete is False
    assert record.outcome.requirements.declared and not record.outcome.requirements.met
    assert record.affected_observation_ids == ("P", "Q") and record.affected_target_ids == ("t-P",)
    assert phase not in record.stop.completed_phases and set(record.stop.completed_phases) < set(RUN_PHASES)
    written = json.loads((out / STOPPED_FILE).read_bytes())
    assert written == record.model_dump(mode="json")
    # the earlier run is as it was and verifies; the stop record beside it is the later attempt, and no temporary file stays
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.name != STOPPED_FILE} == earlier
    assert verify_run_artifacts(out) == ()
    from os_optics.local.artifacts import later_attempt_in

    assert later_attempt_in(out) == {"execution": "CANCELLED", "cause": "CALLER", "phase": phase, "same_project": True,
                                     "project_hash": record.project_hash}
    assert not list(out.glob("*.partial")) and not list(out.glob(".*.partial"))
    # into a directory that held no run: the record alone, which is no run
    alone = tmp_path / "alone"
    with pytest.raises(RunStopped):
        run_optical_project(_project(), monitor=_StopIn(phase), output_dir=alone)
    assert [p.name for p in alone.iterdir()] == [STOPPED_FILE]
    (problem,) = verify_run_artifacts(alone)
    assert "stopped before it completed" in problem and f"in phase {phase}" in problem


def test_a_monitored_run_interrupted_by_ctrl_c_stops_in_its_phase(tmp_path):
    class Interrupt(RunMonitor):
        def step(self, count):
            if self.stack and self.stack[-1][0] == "trace":
                raise KeyboardInterrupt
            super().step(count)

    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=Interrupt(), output_dir=tmp_path / "run")
    record = stopped.value.stopped
    assert record.stop.cause == "CALLER" and record.stop.phase == "trace" and "Ctrl-C" in record.stop.message
    assert (tmp_path / "run" / STOPPED_FILE).exists() and not (tmp_path / "run" / MANIFEST_FILE).exists()


def test_the_memory_limit_stops_the_run_where_it_is_reached():
    def probe():
        return 900 << 20 if monitor.stack and monitor.stack[-1][0] == "scene" else 100 << 20

    monitor = RunMonitor(max_memory_bytes=500 << 20, memory_probe=probe, memory_interval_s=0.0)
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=monitor)
    stop = stopped.value.stopped.stop
    assert (stop.cause, stop.phase, stop.limit_bytes, stop.measured_bytes) == ("MEMORY_LIMIT", "scene", 500 << 20, 900 << 20)
    assert stopped.value.stopped.progress["peak_memory_bytes"] == 900 << 20


def test_the_artifact_limit_stops_the_write_before_the_files_pass_it(tmp_path):
    files = run_artifact_files(run_optical_project(_project()))
    total = sum(len(v) for v in files.values())
    limit = total - len(files[MANIFEST_FILE]) - 1000
    out = tmp_path / "run"
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=_NoPreflight(max_artifact_bytes=limit), output_dir=out)
    stop = stopped.value.stopped.stop
    assert stop.cause == "ARTIFACT_LIMIT" and stop.phase == "write" and stop.limit_bytes == limit
    assert stop.measured_bytes > limit
    assert sorted(p.name for p in out.iterdir()) == [STOPPED_FILE]  # (no file of the stopped write is kept)
    # (at the exact size the run completes)
    run_optical_project(_project(), monitor=_NoPreflight(max_artifact_bytes=total), output_dir=out)
    assert verify_run_artifacts(out) == () and not (out / STOPPED_FILE).exists()


def test_a_run_whose_estimate_exceeds_a_limit_is_refused_before_the_trace(tmp_path):
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=RunMonitor(max_artifact_bytes=100_000), output_dir=tmp_path / "run")
    stop = stopped.value.stopped.stop
    assert stop.phase == "preflight" and stop.cause == "ARTIFACT_LIMIT" and stop.completed_phases == ("lower",)
    assert stop.estimated_bytes == 3 * (OTHER_FILE_BYTES_PER_BEAM + RAY_DATA_BYTES_PER_BEAM["full"])
    assert "2500 independent beams" in stop.message and stopped.value.stopped.engine_input_hash
    assert not any(word in stop.message for word in ("benchmark", "core-", "#"))  # (facts, no internal names)
    # how the estimate was made is stated as fields, in the stop record too; the message is worded from them
    estimate = stopped.value.stopped.progress["estimate"]
    assert estimate == stopped.value.stopped.progress["estimates"][0] and "basis" not in estimate
    assert (estimate["method"], estimate["reference"], estimate["memory_platform"], estimate["lower_bounds"]) == (
        "PER_BEAM_REFERENCE", "INDEPENDENT_BEAMS_2500", "CPYTHON", [])
    assert (estimate["measured_ratio_low"], estimate["measured_ratio_high"]) == PRE_TRACE_MEASURED_RATIO
    assert REFERENCES[estimate["reference"]] in stop.message and "0.57 to 234 times" in stop.message
    assert stop.message.endswith("a higher limit or no pre-flight refusal lets the run start)")
    with pytest.raises(RunStopped) as memory:
        run_optical_project(_project(), monitor=RunMonitor(max_memory_bytes=1 << 20))
    assert memory.value.stopped.stop.cause == "MEMORY_LIMIT"
    # (preflight=False: the estimate is reported, and only the run's own files stop it -- here they fit)
    started = RunMonitor(max_artifact_bytes=1_000_000, preflight=False)
    run_optical_project(_project(), monitor=started, output_dir=tmp_path / "anyway")
    assert started.estimates[0]["artifact_bytes"] > 1_000_000 > started.artifact_bytes_written
    assert verify_run_artifacts(tmp_path / "anyway") == ()


def test_the_estimate_grows_with_the_beams_and_the_recording_level():
    from os_optics.local.lowering import lower_project
    from os_optics.local.catalog import load_project

    request = lower_project(load_project(_project()), None).request
    assert beam_count(request) == 3
    full = estimate_run_resources(request, base_memory_bytes=10)
    summary = estimate_run_resources(request, ray_data="summary", pupil_rays_drawn=False, base_memory_bytes=10,
                                     files_in_memory=True)
    assert full["rays_per_beam"] == 97 and summary["artifact_bytes"] < full["artifact_bytes"] / 4
    assert summary["memory_bytes"] - full["memory_bytes"] == summary["artifact_bytes"]
    from os_optics.local.resources import RUN_MEMORY_PER_BEAM_WASM
    browser = estimate_run_resources(request, base_memory_bytes=10, browser=True)
    assert browser["memory_bytes"] == 10 + 3 * RUN_MEMORY_PER_BEAM_WASM
    assert estimate_run_resources(request)["memory_bytes"] is None
    # how each was made: the reference scene and whose memory costs (none without a memory estimate)
    assert (full["method"], full["reference"], full["memory_platform"]) == ("PER_BEAM_REFERENCE", "INDEPENDENT_BEAMS_2500", "CPYTHON")
    assert (browser["reference"], browser["memory_platform"]) == ("INDEPENDENT_BEAMS_625_BROWSER", "BROWSER_WASM")
    assert estimate_run_resources(request)["memory_platform"] is None
    assert full["lower_bounds"] == [] and "basis" not in full  # (no bound in either direction; no sentence)


def test_the_json_api_reports_a_stopped_run_and_refuses_bad_limits():
    reply = json_api.run_project(_project(), resource_limits={"max_artifact_mib": 1})
    assert reply["ok"] is False and reply["error"]["code"] == "RUN_STOPPED"
    assert reply["stopped"]["stop"]["phase"] == "preflight" and reply["stopped"]["stop"]["cause"] == "ARTIFACT_LIMIT"
    assert reply["stopped"]["outcome"]["execution"]["outcome"] == "CANCELLED" and "files" not in reply
    assert reply["error"]["problems"][0]["code"] == "ARTIFACT_LIMIT"
    # (a host's monitor: cancelled from its progress callback, in the trace)
    monitor = RunMonitor(interval_s=0.0)
    monitor.on_progress = lambda event: monitor.cancel("Cancel pressed") if event["phase"] == "trace" else None
    with monitored(monitor):
        cancelled = json_api.run_project(_project())
    assert cancelled["stopped"]["stop"]["cause"] == "CALLER" and cancelled["stopped"]["stop"]["message"] == "Cancel pressed"
    assert cancelled["stopped"]["affected_observation_ids"] == ["P", "Q"]
    ok = json_api.run_project(_project(), resource_limits={"max_artifact_mib": 64, "max_memory_mib": 1 << 16})
    assert ok["ok"] is True
    assert {"trace", "scene", "write"} <= set(ok["progress"]["phase_seconds"]) and ok["progress"]["artifact_bytes_written"] > 0
    assert "progress" not in json_api.run_project(_project())  # (an unmonitored run's reply is unchanged)
    anyway = json_api.run_project(_project(), resource_limits={"max_artifact_mib": 1, "preflight": False})
    assert anyway["ok"] is True and anyway["progress"]["estimates"][0]["artifact_bytes"] > 1 << 20
    for bad in ({"max_artifact_mib": 0}, {"memory": 5}, {"max_memory_mib": 1.5}, [1], {"preflight": "no"}):
        refused = json_api.run_project(_project(), resource_limits=bad)
        assert refused["ok"] is False and refused["error"]["code"] == "LOCAL_RUN_ERROR"


def test_the_command_line_prints_progress_and_the_stopped_record(tmp_path, capsys, monkeypatch):
    from os_optics import cli

    path = tmp_path / "project.json"
    path.write_text(json.dumps(_project()), encoding="utf-8")
    code = cli.main(["project", "run", str(path), "--out-dir", str(tmp_path / "run"), "--progress", "json"])
    out, err = capsys.readouterr()
    assert code == 0 and json.loads(out)["status"]
    events = [json.loads(line) for line in err.splitlines()]
    assert {"trace", "scene", "write"} <= {e["phase"] for e in events if e["event"] == "phase_ended"}
    code = cli.main(["project", "run", str(path), "--out-dir", str(tmp_path / "run"), "--progress", "text",
                     "--max-artifact-mib", "1"])
    out, err = capsys.readouterr()
    record = json.loads(out)
    assert code == 3 and record["status"] == "CANCELLED" and record["stop"]["phase"] == "preflight"
    assert record["exit_contract"] == "requirements" and "os-optics: lower done" in err and "no result of it is reported" in err
    assert verify_run_artifacts(tmp_path / "run") == () and (tmp_path / "run" / STOPPED_FILE).exists()  # (the earlier run stays)
    code = cli.main(["project", "run", str(path), "--max-artifact-mib", "1", "--status-contract", "execution", "--progress", "none"])
    assert code == 1 and capsys.readouterr().err.count("os-optics:") == 1

    def interrupted(mode):
        def report(event):
            if event["phase"] == "observation_results":
                raise KeyboardInterrupt
        return report

    monkeypatch.setattr(cli, "_progress_printer", interrupted)
    code = cli.main(["project", "run", str(path), "--out-dir", str(tmp_path / "run")])
    record = json.loads(capsys.readouterr().out)
    assert code == 130 and record["stop"]["phase"] == "observation_results" and record["stop"]["cause"] == "CALLER"
    assert verify_run_artifacts(tmp_path / "run") == ()
    assert json.loads((tmp_path / "run" / STOPPED_FILE).read_bytes())["stop"]["phase"] == "observation_results"
    code = cli.main(["project", "run", str(path), "--out-dir", str(tmp_path / "fresh")])
    capsys.readouterr()
    assert code == 130
    assert verify_run_artifacts(tmp_path / "fresh")[0].startswith("not a run directory: the run written here stopped")
    with pytest.raises(SystemExit):
        cli.main(["project", "run", str(path), "--max-memory-mib", "0"])


def test_an_interrupt_between_two_phases_names_the_phase_it_follows():
    monitor = RunMonitor()
    with monitored(monitor):
        from os_optics.run_control import phase
        with phase("trace", unit="beam paths"):
            pass
    stopped = monitor.stopped_by_interrupt()
    assert stopped.phase == "after:trace" and stopped.completed_phases == ("trace",) and stopped.done is None


def test_a_memory_probe_that_answers_no_number_reads_as_unknown_memory():
    class NotANumber:
        pass

    for answer in (None, NotANumber(), "x"):
        monitor = RunMonitor(max_memory_bytes=1, memory_probe=lambda answer=answer: answer)
        assert monitor.memory_bytes is None
        run_optical_project(_project(), monitor=monitor)  # (no reading, no stop)


# ------------------------------------------------------------------ the run directory rule
#
# A run that completes replaces the run in its directory: its files are written to temporary names and moved into place
# once all are written. An attempt that does not complete leaves the directory as it found it.

ZOOM = Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json"
RUN_OUTPUTS = {"engine_request.json", "simulation_result.json", "ray_data.json", "local_validation.json", "optical_scene.json",
               "optical_scene.glb", MANIFEST_FILE}


def _tree(directory) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*"))


def _files(directory) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def _staged(directory) -> list[str]:
    """The temporary files of a write left under ``directory`` (``.<name>.partial``, and the earlier ``<name>.partial``)."""

    return [name for name in _tree(directory) if name.endswith(".partial")]


def _link_directory(link: Path, target: Path, kind: str) -> None:
    """A directory link: a symbolic link (POSIX; on Windows only where the account may create one) or a Windows junction."""

    import os

    if kind == "junction":
        if os.name != "nt":
            pytest.skip("directory junctions exist on Windows only")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:  # (Windows without the symbolic-link privilege)
            pytest.skip(f"no symbolic link can be created here: {exc}")


def _cli_code(argv: list[str]) -> int:
    from os_optics import cli

    try:
        return int(cli.main(argv))
    except SystemExit as exc:  # (a usage error)
        return int(exc.code)


def _design_file(tmp_path: Path, name: str, payload: dict, *, indent: int | None = None) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def _other_design() -> dict:
    payload = json.loads(json.dumps(_project()))
    payload["components"][0]["nominal_pose"]["position_mm"] = [-450.0, 0.0, 0.0]  # (another design: the source moved)
    return payload


def _payload(data: bytes) -> dict:
    from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2

    return ProjectDocumentV2.model_validate(json.loads(data)).payload()


def _attempt(tmp_path: Path, outcome: str, out: Path) -> list[str]:
    """``project run`` into ``out`` that does not complete: ``usage`` (a usage error, 64), ``refused`` (a Project that
    cannot be simulated, 65) or ``stopped`` (refused before the trace by --max-memory-mib, a stop record, 3)."""

    path = tmp_path / f"attempt_{outcome}.json"
    path.write_text("{}" if outcome == "refused" else json.dumps(_other_design()), encoding="utf-8")
    extra = {"usage": ["--max-memory-mib", "abc"], "refused": [], "stopped": ["--max-memory-mib", "1"]}[outcome]
    return ["project", "run", str(path), "--out-dir", str(out), "--progress", "none", *extra]


def _complete(tmp_path: Path, out: Path, design: dict | None = None) -> list[str]:
    """``project run`` of ``design`` (default: the test Project) into ``out``, which completes."""

    return ["project", "run", str(_design_file(tmp_path, "complete.json", design or _project())), "--out-dir", str(out),
            "--progress", "none"]


EXPECTED_CODE = {"usage": 64, "refused": 65, "stopped": 3}
ATTEMPTS = ["usage", "refused", "stopped"]


def _refuse_on(monkeypatch, paths, *, delete: bool = True, rename: bool = False, write: bool = False) -> None:
    """The system refuses to delete (and to rename, and to write) exactly these files, as Windows does a read-only file
    (delete) or a file another program holds open (delete and rename)."""

    blocked = {Path(p).absolute() for p in paths}
    real_unlink, real_replace, real_open = Path.unlink, Path.replace, Path.open

    def refused(path: Path, what: str):
        return PermissionError(13, f"{what} refused", str(path)) if path.absolute() in blocked else None

    def unlink(self, missing_ok=False):
        if delete and (error := refused(self, "delete")):
            raise error
        return real_unlink(self, missing_ok=missing_ok)

    def replace(self, target):
        if rename and (error := refused(self, "rename") or refused(Path(target), "rename")):
            raise error
        return real_replace(self, target)

    def open_(self, mode="r", *args, **kwargs):
        if write and any(c in mode for c in "wa+x") and (error := refused(self, "write")):
            raise error
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "open", open_)


def _later_attempt(directory) -> dict | None:
    from os_optics.local.artifacts import later_attempt_in

    return later_attempt_in(directory)


# ------------------------------------------------------------------ an attempt that does not complete keeps the earlier run


@pytest.mark.parametrize("outcome", ATTEMPTS)
def test_an_attempt_that_does_not_complete_leaves_the_earlier_run_as_it_was(tmp_path, capsys, outcome):
    """A usage error, a Project that is refused and a stopped run -- each of another design than the run in the directory --
    change nothing there but the stop record a stop adds: the earlier run's files are byte for byte the same, it verifies
    and loads, and ``run_freshness`` says it is not the edited Project's run."""

    from os_optics import cli
    from os_optics.local import run_freshness
    from os_optics.results import load_run

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    (out / "notes.txt").write_text("mine", encoding="utf-8")  # (a file the Engine does not write)
    before = _files(out)
    assert _cli_code(_attempt(tmp_path, outcome, out)) == EXPECTED_CODE[outcome]
    err = capsys.readouterr().err
    assert "Traceback" not in err and "remains" not in err
    after = _files(out)
    stop_record = after.pop(STOPPED_FILE, None)
    assert after == before and (stop_record is not None) == (outcome == "stopped") and _staged(out) == []
    assert verify_run_artifacts(out) == ()
    assert cli.main(["project", "verify", str(out)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "CONSISTENT" and verified["problems"] == []
    kept = load_run(out, verify=True)
    assert kept.describe()["integrity"] == {"checked": True, "consistent": True, "problems": []}
    assert kept.describe()["run"]["project_hash"] == run_optical_project(_project()).local_validation.project.project_hash.digest
    # the kept run is the earlier Project's: current for it, historical for the design of the attempt
    assert run_freshness(_project(), out)["freshness"] == "CURRENT"
    edited = run_freshness(_other_design(), out)
    assert edited["freshness"] == "HISTORICAL" and "PHYSICS_CHANGED" in edited["reasons"]
    attempt = {"execution": "CANCELLED", "cause": "MEMORY_LIMIT", "phase": "lower", "same_project": False,
               "project_hash": edited["current"]["project_hash"]}
    assert _later_attempt(out) == (attempt if outcome == "stopped" else None)
    assert verified.get("later_attempt") == (attempt if outcome == "stopped" else None)
    assert kept.describe().get("later_attempt") == (attempt if outcome == "stopped" else None)
    # a run that completes replaces the run and removes the stop record
    assert _cli_code(_complete(tmp_path, out, _other_design())) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and STOPPED_FILE not in _tree(out) and _later_attempt(out) is None
    assert run_freshness(_other_design(), out)["freshness"] == "CURRENT"
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_a_stop_while_the_files_are_written_leaves_no_temporary_file(tmp_path):
    """The earlier pin of a run stopped by the artifact limit while it writes (``ARTIFACT_LIMIT`` in phase ``write``): the
    earlier run and the person's file stay, the stop record is added, and no ``.partial`` file is left."""

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    before = _files(out)
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=_NoPreflight(max_artifact_bytes=1000), output_dir=out)
    assert (stopped.value.stopped.stop.cause, stopped.value.stopped.stop.phase) == ("ARTIFACT_LIMIT", "write")
    assert _tree(out) == sorted([*before, STOPPED_FILE]) and _staged(out) == []
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before
    assert verify_run_artifacts(out) == ()
    assert _later_attempt(out) == {"execution": "CANCELLED", "cause": "ARTIFACT_LIMIT", "phase": "write", "same_project": True,
                                   "project_hash": stopped.value.stopped.project_hash}
    # the same when it stops before anything is written (cancelled in the trace)
    with pytest.raises(RunStopped):
        run_optical_project(_project(), monitor=_StopIn("trace"), output_dir=out)
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and _later_attempt(out)["phase"] == "trace"
    # a completed run removes the stop record
    run_optical_project(_project(), output_dir=out)
    assert verify_run_artifacts(out) == () and STOPPED_FILE not in _tree(out)


def test_a_stopped_run_of_configurations_keeps_the_earlier_run_of_configurations(tmp_path, capsys, monkeypatch):
    from os_optics import cli
    from os_optics.cli_exit import STATUS_CONTRACT_ENV

    monkeypatch.delenv(STATUS_CONTRACT_ENV, raising=False)
    out = tmp_path / "run"
    assert cli.main(["project", "run", str(ZOOM), "--out-dir", str(out), "--progress", "none"]) == 0
    capsys.readouterr()
    assert {"configurations/wide/manifest.json", "configurations/tele/manifest.json", "comparison.json"} <= set(_tree(out))
    (out / "configurations" / "tele" / "my_notes.txt").write_text("mine", encoding="utf-8")
    before = _files(out)
    edited = json.loads(ZOOM.read_text(encoding="utf-8"))
    edited["components"][0]["source_model"]["power_w"] = 0.002  # (a changed Project: the earlier runs are not its runs)
    path = tmp_path / "zoom_edit.json"
    path.write_text(json.dumps(edited), encoding="utf-8")
    code = cli.main(["project", "run", str(path), "--out-dir", str(out), "--max-artifact-mib", "1", "--no-preflight",
                     "--progress", "none"])
    assert code == 3
    assert json.loads(capsys.readouterr().out)["stop"]["cause"] == "ARTIFACT_LIMIT"
    # the earlier run of configurations is as it was, in every directory; the stop record is beside it
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and _staged(out) == []
    for directory in (out, *(out / "configurations" / cid for cid in ("wide", "mid", "tele"))):
        assert verify_run_artifacts(directory) == (), directory
    assert cli.main(["project", "verify", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["later_attempt"]["same_project"] is False


def test_a_run_of_configurations_cancelled_in_the_trace_keeps_the_earlier_run(tmp_path):
    from os_optics.local import run_configured_project

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    before = _files(out)
    with pytest.raises(RunStopped) as stopped, monitored(_StopIn("trace")):
        run_configured_project(ZOOM.read_bytes(), output_dir=out)
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and verify_run_artifacts(out) == ()
    # the stop's record is written beside the earlier run, as a single run's is
    record = stopped.value.stopped
    assert record is not None and json.loads((out / STOPPED_FILE).read_bytes()) == record.model_dump(mode="json")
    assert _later_attempt(out) == {"execution": "CANCELLED", "cause": "CALLER", "phase": "trace", "same_project": True,
                                   "project_hash": record.project_hash}
    assert record.project_hash == json.loads((out / MANIFEST_FILE).read_bytes())["project"]["project_hash"]["digest"]


def _interrupt_after_the_move(monkeypatch, out: Path, error: BaseException) -> dict:
    """Raise ``error`` once, in the clean-up that follows the directory's manifest move (``_remove_quietly`` of the commit's
    copies), while ``out``'s manifest is in place."""

    from os_optics.local import artifacts

    clean_up = artifacts._remove_quietly
    raised = {"count": 0}

    def interrupted(paths):
        if not raised["count"] and (out / MANIFEST_FILE).exists():
            raised["count"] += 1
            raise error
        return clean_up(paths)

    monkeypatch.setattr(artifacts, "_remove_quietly", interrupted)
    return raised


def _completed_here(out: Path, project_hash: str) -> None:
    """``out`` holds this attempt's complete run, which verifies, with no stop record and no later attempt."""

    assert verify_run_artifacts(out) == () and not (out / STOPPED_FILE).exists() and _later_attempt(out) is None
    assert json.loads((out / MANIFEST_FILE).read_bytes())["project"]["project_hash"]["digest"] == project_hash
    assert not list(out.rglob("*.partial"))


@pytest.mark.parametrize("earlier", [False, True])
def test_a_ctrl_c_after_the_manifest_is_moved_into_place_returns_the_completed_run(tmp_path, monkeypatch, earlier):
    """A Ctrl-C that lands once this attempt's manifest is in place (in the move's clean-up) follows a complete run of this
    very attempt: ``run_optical_project`` returns it with its artifacts, writes no stop record (none names a later attempt
    after it) -- also over an earlier run and its stop record."""

    out = tmp_path / "run"
    if earlier:
        run_optical_project(_project(), output_dir=out)
        with pytest.raises(RunStopped):
            run_optical_project(_project(), monitor=_StopIn("trace"), output_dir=out)
        assert (out / STOPPED_FILE).exists()
    raised = _interrupt_after_the_move(monkeypatch, out, KeyboardInterrupt())
    run = run_optical_project(_project(), output_dir=out, monitor=RunMonitor(None))
    assert raised["count"] == 1 and run.artifacts is not None and run.artifacts.directory == out
    _completed_here(out, run.local_validation.project.project_hash.digest)


def test_a_stop_from_the_progress_callback_as_the_write_ends_returns_the_completed_run(tmp_path):
    """A caller's progress callback that stops the run on the end of phase ``write`` -- after the run was moved into
    place -- gets the completed run, and no stop record is written beside it."""

    def stop_when_written(report):
        if report["event"] == "phase_ended" and report["phase"] == "write":
            raise RunStopped("CALLER", "stopped once written", phase="write")

    out = tmp_path / "run"
    run = run_optical_project(_project(), output_dir=out, monitor=RunMonitor(stop_when_written))
    assert run.artifacts is not None and run.artifacts.directory == out
    _completed_here(out, run.local_validation.project.project_hash.digest)


def test_the_command_reports_a_run_completed_before_a_ctrl_c_as_completed(tmp_path, capsys, monkeypatch):
    """``project run``: a Ctrl-C after the run is in place reports the run (its status and exit code), not CANCELLED with
    exit 130 and 'no result of it is reported'."""

    from os_optics import cli
    from os_optics.cli_exit import STATUS_CONTRACT_ENV

    monkeypatch.delenv(STATUS_CONTRACT_ENV, raising=False)
    out = tmp_path / "run"
    raised = _interrupt_after_the_move(monkeypatch, out, KeyboardInterrupt())
    code = cli.main(_complete(tmp_path, out))
    captured = capsys.readouterr()
    assert raised["count"] == 1 and code == 0, captured.err
    assert json.loads(captured.out)["status"] != "CANCELLED" and "no result of it is reported" not in captured.err
    assert verify_run_artifacts(out) == () and not (out / STOPPED_FILE).exists() and _later_attempt(out) is None


@pytest.mark.parametrize("stop", ["ctrl-c", "stopped"])
def test_a_stop_after_a_run_of_configurations_is_moved_into_place_returns_it(tmp_path, monkeypatch, stop):
    """The same for a run of configurations, written by ``run_configured_project`` (``write_configurations_artifacts``) and
    by the JSON API from its files (``write_run_files``), for a Ctrl-C and for a stop (``RunStopped``, as a monitor's
    callback raises) in the move's clean-up: the completed run, no stop record, no later attempt."""

    from os_optics.local import run_configured_project

    def error():
        return KeyboardInterrupt() if stop == "ctrl-c" else RunStopped("CALLER", "stopped once written", phase="write")

    out = tmp_path / "run"
    raised = _interrupt_after_the_move(monkeypatch, out, error())
    configured = run_configured_project(ZOOM.read_bytes(), output_dir=out)
    assert raised["count"] == 1 and configured.artifacts is not None and configured.artifacts.directory == out
    _completed_here(out, json.loads((out / MANIFEST_FILE).read_bytes())["project"]["project_hash"]["digest"])
    for directory in (out / "configurations").iterdir():
        assert verify_run_artifacts(directory) == (), directory
    files_out = tmp_path / "files"
    raised = _interrupt_after_the_move(monkeypatch, files_out, error())
    with monitored(RunMonitor(None)):
        reply = json_api.run_project(json.loads(ZOOM.read_text(encoding="utf-8")), output_dir=str(files_out))
    assert raised["count"] == 1 and reply["ok"] is True and reply["verify"] == [], reply.get("error")
    assert verify_run_artifacts(files_out) == () and not (files_out / STOPPED_FILE).exists() and _later_attempt(files_out) is None


def _monitored_json_api_run(design: dict, out: Path) -> dict:
    """``json_api.run_project`` under a monitor; an interrupt it lets through fails the test (it does not end the session)."""

    try:
        with monitored(RunMonitor(None)):
            return json_api.run_project(design, output_dir=str(out))
    except (KeyboardInterrupt, RunStopped) as exc:
        pytest.fail(f"json_api.run_project let {type(exc).__name__} through instead of answering")


@pytest.mark.parametrize("stop", ["ctrl-c", "stopped"])
def test_the_json_api_answers_a_stop_after_a_single_run_is_moved_into_place_with_the_run(tmp_path, monkeypatch, stop):
    """``json_api.run_project`` on a Project without configurations: a Ctrl-C or a stop in the clean-up that follows the
    move of its manifest answers with the completed run -- ``ok``, ``verify`` empty, its files -- and writes no stop
    record."""

    from os_optics.local import example_project_of_kind

    def error():
        return KeyboardInterrupt() if stop == "ctrl-c" else RunStopped("CALLER", "stopped once written", phase="write")

    out = tmp_path / "run"
    raised = _interrupt_after_the_move(monkeypatch, out, error())
    reply = _monitored_json_api_run(example_project_of_kind("focus").payload(), out)
    assert raised["count"] == 1 and reply["ok"] is True and reply["verify"] == [], reply.get("error")
    assert reply["files"] == _files(out) and "progress" in reply
    _completed_here(out, reply["manifest"]["project"]["project_hash"]["digest"])


@pytest.mark.parametrize("earlier", [False, True])
def test_the_json_api_stops_a_single_run_interrupted_before_its_move_and_leaves_the_directory(tmp_path, monkeypatch, earlier):
    """A monitored ``json_api.run_project`` (no configurations) that a Ctrl-C interrupts before its run is moved into place
    returns the stopped reply, writes its stop record and leaves every other file as it found it (an earlier run there
    still verifies)."""

    from os_optics.local import artifacts, example_project_of_kind

    design = example_project_of_kind("focus").payload()
    out = tmp_path / "run"
    if earlier:
        assert json_api.run_project(design, output_dir=str(out))["verify"] == []
    found = _files(out) if earlier else {}

    def interrupted(write, artifacts=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(artifacts, "_commit_run", interrupted)
    reply = _monitored_json_api_run(design, out)
    assert reply["ok"] is False and reply["error"]["code"] == "RUN_STOPPED" and reply["stopped"] is not None
    assert (out / STOPPED_FILE).exists()
    assert {name: data for name, data in _files(out).items() if name != STOPPED_FILE} == found
    if earlier:
        assert verify_run_artifacts(out) == () and _later_attempt(out) is not None


def test_a_run_into_a_used_directory_keeps_no_file_the_new_run_does_not_write(tmp_path):
    from os_optics.local import run_configured_project

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    # fewer configurations: the others' run directories go
    run_configured_project(ZOOM.read_bytes(), output_dir=out, configuration_ids=["wide"])
    assert sorted(p.name for p in (out / "configurations").iterdir()) == ["wide"]
    assert verify_run_artifacts(out) == ()
    # a single run where a run of configurations was: no comparison, no configurations
    run_optical_project(_project(), output_dir=out)
    assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", "notes.txt"} and verify_run_artifacts(out) == ()
    # and the JSON API's write: a run of configurations where a single run was -- its run files go
    reply = json_api.run_project(ZOOM.read_text(encoding="utf-8"), output_dir=str(out), configuration_ids=["tele"])
    assert reply["ok"] and reply["verify"] == []
    assert sorted(p.name for p in (out / "configurations").iterdir()) == ["tele"]
    assert not (out / "optical_scene.glb").exists() and not (out / "simulation_result.json").exists()
    assert verify_run_artifacts(out) == () and (out / "notes.txt").read_text(encoding="utf-8") == "mine"


# ------------------------------------------------------------------ the readers of a run kept behind a later attempt


def test_a_stop_record_beside_a_manifest_names_a_later_attempt(tmp_path, capsys):
    """``verify`` checks the run that is there. A stop record beside it is the record of a later attempt: every reader of
    the kept run states it; a stop record with no manifest is no run; a file of that name that is no stop record is
    nothing."""

    from os_optics import cli
    from os_optics.local import run_configured_project
    from os_optics.results import load_run

    stop = tmp_path / "stop"
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(_project(), monitor=_StopIn("trace"), output_dir=stop)
    record = (stop / STOPPED_FILE).read_bytes()
    (problem,) = verify_run_artifacts(stop)  # (a stop record alone: no run)
    assert problem.startswith("not a run directory: the run written here stopped before it completed") and "in phase trace" in problem
    assert _later_attempt(stop) == {"execution": "CANCELLED", "cause": "CALLER", "phase": "trace", "same_project": False,
                                    "project_hash": stopped.value.stopped.project_hash}
    assert cli.main(["project", "verify", str(stop)]) != 0
    assert "later_attempt" not in json.loads(capsys.readouterr().out)

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    (out / STOPPED_FILE).write_bytes(record)
    assert verify_run_artifacts(out) == ()
    attempt = {"execution": "CANCELLED", "cause": "CALLER", "phase": "trace", "same_project": True,
               "project_hash": stopped.value.stopped.project_hash}
    assert cli.main(["project", "verify", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"run_dir": str(out), "status": "CONSISTENT", "problems": [], "later_attempt": attempt}
    assert load_run(out).describe()["later_attempt"] == attempt
    files = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert json_api.run_results(files, {"kind": "describe"})["result"]["later_attempt"] == attempt
    assert cli.main(["results", "describe", str(out)]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["later_attempt"] == attempt
    assert printed.err.count("os-optics: note: a later run into") == 1 and "stopped in phase trace (CALLER)" in printed.err
    assert "the same Project" in printed.err

    zoom = tmp_path / "zoom"
    run_configured_project(ZOOM.read_bytes(), output_dir=zoom)
    (zoom / STOPPED_FILE).write_bytes(record)
    assert verify_run_artifacts(zoom) == () and _later_attempt(zoom)["same_project"] is False
    for cid in ("wide", "mid", "tele"):
        assert verify_run_artifacts(zoom / "configurations" / cid) == (), cid
    # a stop record that is not one (a file of that name) states nothing
    (out / STOPPED_FILE).write_text("{\"mine\": true}", encoding="utf-8")
    assert verify_run_artifacts(out) == () and _later_attempt(out) is None
    assert "later_attempt" not in load_run(out).describe()
    assert cli.main(["results", "describe", str(out)]) == 0
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------ files the system will not let go


@pytest.mark.parametrize("outcome", ATTEMPTS)
@pytest.mark.parametrize("obstacle", ["delete", "delete+rename", "delete+rename+write"])
def test_an_attempt_never_touches_a_manifest_the_system_would_not_delete(tmp_path, capsys, monkeypatch, outcome, obstacle):
    """A manifest that cannot be deleted, renamed or written (Windows: read-only, or held open by another program) was
    the hard case of removing an earlier run. An attempt that does not complete removes nothing, so it never meets it: the
    command keeps its own exit code and the earlier run verifies."""

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    before = _files(out)
    _refuse_on(monkeypatch, [out / MANIFEST_FILE], rename="rename" in obstacle, write="write" in obstacle)
    assert _cli_code(_attempt(tmp_path, outcome, out)) == EXPECTED_CODE[outcome]
    err = capsys.readouterr().err
    monkeypatch.undo()
    assert "remains" not in err and "Traceback" not in err
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before
    assert verify_run_artifacts(out) == () and (out / STOPPED_FILE).exists() == (outcome == "stopped")


@pytest.mark.parametrize("held", [MANIFEST_FILE, STOPPED_FILE])
def test_a_completed_run_that_cannot_remove_the_earlier_manifest_or_stop_record_writes_nothing(tmp_path, capsys, monkeypatch, held):
    """The manifest and the stop record there go before any file is moved into place. When the system refuses to delete
    one (the manifest first), the run is not written: exit 64, every file as it was, no temporary file left."""

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    with pytest.raises(RunStopped):
        run_optical_project(_project(), monitor=_StopIn("trace"), output_dir=out)
    before = _files(out)
    _refuse_on(monkeypatch, [out / held], rename=True, write=True)
    assert _cli_code(_complete(tmp_path, out)) == 64
    err = capsys.readouterr().err
    monkeypatch.undo()
    assert "cannot write the run directory" in err and "delete refused" in err and "Traceback" not in err
    # the stop record goes before the manifest, and the manifest before any file is moved: whichever of them the system
    # refuses, the run that was there is as it was and verifies
    assert _files(out) == ({k: v for k, v in before.items() if k != STOPPED_FILE} if held == MANIFEST_FILE else before)
    assert verify_run_artifacts(out) == () and _staged(out) == []
    # once it can be removed, the run completes there
    assert _cli_code(_complete(tmp_path, out)) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and STOPPED_FILE not in _tree(out)


@pytest.mark.parametrize("held", [MANIFEST_FILE, "ray_data.json"])
def test_a_commit_that_fails_answers_64_on_the_oldest_python_the_engine_supports(tmp_path, capsys, monkeypatch, held):
    """The cleanup of a failing commit uses only what Python 3.11 has: with ``Path.exists`` as 3.11 defines it (no
    ``follow_symlinks``), a manifest the system will not delete, or a file it will not let the commit move into place,
    still answers 64 'cannot write the run directory' -- not 70 with the cleanup's own exception."""

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    real_exists = Path.exists

    def exists(self):  # (Python 3.11: Path.exists() takes no keyword)
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", exists)
    _refuse_on(monkeypatch, [out / held], rename=True, write=True)
    code = _cli_code(_complete(tmp_path, out))
    err = capsys.readouterr().err
    monkeypatch.undo()
    assert code == 64 and "cannot write the run directory" in err and "Traceback" not in err, err


@pytest.mark.parametrize("held", ["comparison.json", "configurations/mid/manifest.json", "configurations/mid/ray_data.json"])
def test_a_file_of_an_earlier_run_of_configurations_that_cannot_be_removed(tmp_path, capsys, monkeypatch, held):
    """A single run written where a run of configurations was removes that run's files by name. A manifest it cannot
    delete stops the write before anything is moved (the configuration's manifest goes with the directory's, first); any
    other file it cannot delete stays where it is, and the new run is complete and verifies."""

    from os_optics.local import run_configured_project

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    before = _files(out)
    _refuse_on(monkeypatch, [out / held], rename=True, write=True)
    code = _cli_code(_complete(tmp_path, out))
    err = capsys.readouterr().err
    monkeypatch.undo()
    if held.endswith(MANIFEST_FILE):
        assert code == 64 and "cannot write the run directory" in err
        # (the configurations' manifests go before the directory's own, in the manifest's order -- wide, mid, tele: wide's is
        # gone, mid's refused, and nothing else has changed but the copy of wide's manifest, which stays as a kill there
        # leaves it -- the record of wide's files; the directory's manifest still describes what is there)
        wide = "configurations/wide/manifest.json"
        assert _files(out) == {**{k: v for k, v in before.items() if k != wide},
                               "configurations/wide/.manifest.json.earlier": before[wide]}
        assert verify_run_artifacts(out) == ("configurations/wide/manifest.json: missing",)
        assert verify_run_artifacts(out / "configurations" / "mid") == ()  # (a complete run directory on its own)
        assert _cli_code(_complete(tmp_path, out)) == 0  # (once it can be removed, the run is written)
        capsys.readouterr()
        assert verify_run_artifacts(out) == () and set(_tree(out)) == RUN_OUTPUTS | {"project.json"}
    else:
        assert code == 0 and verify_run_artifacts(out) == ()
        # (the file it could not remove stays, recorded by name beside the run: .manifest.json.leftover)
        assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", ".manifest.json.leftover",
                                                 *(part for part in (held, *Path(held).parents) if str(part) != "."
                                                   for part in [Path(part).as_posix()])}
        assert (out / held).read_bytes() == before[held]
        # once released: the earlier run of configurations completes there, and the single run then leaves its own files
        run_configured_project(ZOOM.read_bytes(), output_dir=out)
        assert verify_run_artifacts(out) == () and ".manifest.json.leftover" not in _tree(out)
        assert _cli_code(_complete(tmp_path, out)) == 0
        capsys.readouterr()
        assert verify_run_artifacts(out) == () and set(_tree(out)) == RUN_OUTPUTS | {"project.json"}


def test_a_name_the_commit_did_remove_is_the_persons_once_it_is_gone(tmp_path, capsys, monkeypatch):
    """A run of configurations replaces a single run while its engine_request.json is held open: that file stays and is
    recorded; ray_data.json, which the commit removed, is not. A file the person then puts at ray_data.json is their own:
    the single run refuses it (exit 64) and leaves it, and accepts the recorded engine_request.json once it is moved away."""

    from os_optics.local import run_configured_project

    out = tmp_path / "run"
    plain = json.loads(ZOOM.read_bytes())
    del plain["configurations"]
    plain["targets"] = [target for target in plain["targets"] if not target.get("configuration_ids")]
    run_optical_project(plain, output_dir=out)
    _refuse_on(monkeypatch, [out / "engine_request.json"])
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    monkeypatch.undo()
    assert verify_run_artifacts(out) == () and (out / "engine_request.json").exists() and not (out / "ray_data.json").exists()
    (out / "ray_data.json").write_text("my own data", encoding="utf-8")
    assert _cli_code(_complete(tmp_path, out, plain)) == 64
    err = capsys.readouterr().err
    assert "ray_data.json is no file of an earlier run there" in err
    assert (out / "ray_data.json").read_text(encoding="utf-8") == "my own data"
    (out / "ray_data.json").rename(tmp_path / "mine.json")
    assert _cli_code(_complete(tmp_path, out, plain)) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and set(_tree(out)) == RUN_OUTPUTS | {"project.json"}


def _windows_only():
    import os

    if os.name != "nt":
        pytest.skip("the read-only attribute and a file held open keep a file from deletion on Windows only")


def _read_only(path: Path) -> None:
    import os
    import stat

    os.chmod(path, stat.S_IREAD)


def _writable(directory: Path) -> None:
    import os
    import stat

    for p in Path(directory).rglob("*"):
        if p.is_file() and not p.is_symlink():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("outcome", ATTEMPTS)
def test_a_read_only_manifest_stays_with_its_run_on_windows(tmp_path, capsys, outcome):
    _windows_only()
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    _read_only(out / MANIFEST_FILE)
    try:
        before = _files(out)
        assert _cli_code(_attempt(tmp_path, outcome, out)) == EXPECTED_CODE[outcome]
        capsys.readouterr()
        assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and verify_run_artifacts(out) == ()
        # a run that completes cannot delete it: nothing is moved, the earlier run stays
        assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
        assert "cannot write the run directory" in capsys.readouterr().err
        assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and _staged(out) == []
    finally:
        _writable(out)


def test_a_manifest_held_open_stays_with_its_run_on_windows(tmp_path, capsys):
    _windows_only()
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    before = _files(out)
    with open(out / MANIFEST_FILE, "rb"):  # (Python opens a file without sharing its deletion: held open)
        assert _cli_code(_attempt(tmp_path, "stopped", out)) == 3
        assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
    capsys.readouterr()
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and verify_run_artifacts(out) == ()


@pytest.mark.parametrize("then", ["the same run", "the earlier design"])
@pytest.mark.parametrize("held", ["optical_scene.glb", "ray_data.json"])
def test_a_run_file_held_open_while_the_run_is_moved_into_place_leaves_a_directory_any_run_completes_on_windows(
        tmp_path, capsys, held, then):
    """A file of the earlier run held open by another program (a glTF viewer, a reader) refuses the move onto it, after
    the manifest is removed: the run exits 64. What it leaves is what a kill there leaves -- the copy of the earlier
    manifest, which lists the earlier run's files, and the staged manifest, which lists the files already moved -- so,
    once the file is closed, the same run or the earlier design completes there, verifies and leaves exactly its files."""

    from os_optics.local.artifacts import EARLIER_MANIFEST_FILE

    _windows_only()
    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    with open(out / held, "rb"):  # (Python opens a file without sharing its deletion: held open)
        assert _cli_code(_complete(tmp_path, out)) == 64
    assert "cannot write the run directory" in capsys.readouterr().err
    assert not (out / MANIFEST_FILE).exists() and (out / EARLIER_MANIFEST_FILE).exists()
    assert _staged(out) == [".manifest.json.partial"]
    assert _cli_code(_complete(tmp_path, out, _project() if then == "the same run" else _other_design())) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and set(_tree(out)) == RUN_OUTPUTS | {"project.json"}


def test_a_configuration_manifest_held_open_leaves_a_run_of_configurations_the_same_run_completes_on_windows(tmp_path, capsys):
    """A run of configurations over an earlier one, with ``configurations/mid/manifest.json`` held open: wide's manifest is
    removed, mid's is refused, exit 64. The copy of wide's manifest stays (the record of wide's files), so once the file is
    closed the same run completes there and verifies."""

    from os_optics.local import run_configured_project

    _windows_only()
    out = tmp_path / "zoom"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    argv = ["project", "run", str(ZOOM), "--out-dir", str(out), "--progress", "none"]
    with open(out / "configurations" / "mid" / MANIFEST_FILE, "rb"):
        assert _cli_code(argv) == 64
    capsys.readouterr()
    assert (out / "configurations" / "wide" / ".manifest.json.earlier").exists() and (out / MANIFEST_FILE).exists()
    assert _cli_code(argv) in (0, 1, 2)
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and _staged(out) == []
    assert not any(name.endswith(".manifest.json.earlier") for name in _tree(out))


def test_a_directory_that_cannot_be_written_keeps_its_run_on_posix(tmp_path, capsys):
    """POSIX: a directory without write permission lets no file in it be deleted, renamed or created. A stop writes no
    record there and a run that completes writes nothing; the earlier run is as it was."""

    import os
    import stat

    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("POSIX directory permissions, which root does not obey")
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    before = _files(out)
    os.chmod(out, stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert _cli_code(_attempt(tmp_path, "stopped", out)) == 3
        assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
        capsys.readouterr()
        assert _files(out) == before and verify_run_artifacts(out) == ()
    finally:
        os.chmod(out, stat.S_IRWXU)


# ------------------------------------------------------------------ a hard link: only its name here is replaced


def _linked_run(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """A complete run kept outside ``--out-dir`` (``outside``), whose ``manifest.json`` is also ``out/manifest.json`` (a
    hard link: one file, two names)."""

    import os

    outside, out = tmp_path / "outside", tmp_path / "run"
    run_optical_project(_project(), output_dir=outside)
    out.mkdir()
    os.link(outside / MANIFEST_FILE, out / MANIFEST_FILE)
    assert (out / MANIFEST_FILE).stat().st_nlink == 2
    return outside, out, (outside / MANIFEST_FILE).read_bytes()


@pytest.mark.parametrize("outcome", ["usage", "stopped", "complete"])
def test_a_hard_linked_manifest_loses_only_its_name_here(tmp_path, capsys, outcome):
    """An attempt that does not complete leaves the name; a run that completes deletes that name (never the file's
    content) and writes its own manifest: the run outside keeps its manifest and verifies."""

    outside, out, before = _linked_run(tmp_path)
    argv = _complete(tmp_path, out, _other_design()) if outcome == "complete" else _attempt(tmp_path, outcome, out)
    assert _cli_code(argv) == (0 if outcome == "complete" else EXPECTED_CODE[outcome])
    capsys.readouterr()
    assert (outside / MANIFEST_FILE).read_bytes() == before and verify_run_artifacts(outside) == ()
    if outcome == "complete":
        assert (out / MANIFEST_FILE).stat().st_nlink == 1 and verify_run_artifacts(out) == ()
    else:
        assert (out / MANIFEST_FILE).read_bytes() == before and (out / MANIFEST_FILE).stat().st_nlink == 2


def test_a_hard_linked_manifest_that_cannot_be_deleted_is_never_written_into(tmp_path, capsys, monkeypatch):
    """When its name here can be neither deleted nor renamed (Windows: another program holds the file open), a run that
    completes is not written (exit 64); the file -- the run outside's manifest too -- keeps its bytes."""

    outside, out, before = _linked_run(tmp_path)
    _refuse_on(monkeypatch, [out / MANIFEST_FILE], rename=True)
    assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
    err = capsys.readouterr().err
    monkeypatch.undo()
    assert "cannot write the run directory" in err
    assert (outside / MANIFEST_FILE).read_bytes() == before and verify_run_artifacts(outside) == ()
    assert _tree(out) == [MANIFEST_FILE]


@pytest.mark.parametrize("name", ["ray_data.json", MANIFEST_FILE])
def test_a_run_never_writes_into_a_temporary_file_that_is_a_hard_link(tmp_path, capsys, monkeypatch, name):
    """A temporary file left in ``--out-dir`` that is a hard link to a file elsewhere: the run deletes that name and
    creates its own file; if the name cannot be deleted it refuses to write (exit 64). The file elsewhere is unchanged."""

    import os

    outside, out = tmp_path / "outside", tmp_path / "run"
    run_optical_project(_project(), output_dir=outside)
    out.mkdir()
    partial = out / f".{name}.partial"
    os.link(outside / name, partial)
    before = (outside / name).read_bytes()
    _refuse_on(monkeypatch, [partial])
    assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
    assert "cannot write the run directory" in capsys.readouterr().err
    monkeypatch.undo()
    assert (outside / name).read_bytes() == before and verify_run_artifacts(outside) == ()
    assert _tree(out) == [partial.name]
    assert _cli_code(_complete(tmp_path, out, _other_design())) == 0
    capsys.readouterr()
    assert (outside / name).read_bytes() == before and verify_run_artifacts(outside) == () and verify_run_artifacts(out) == ()
    assert _staged(out) == []


def test_a_hard_linked_manifest_held_open_keeps_its_bytes_on_windows(tmp_path, capsys):
    """Another program holds the shared file open (delete and rename fail): neither a stop nor a run that completes
    changes it."""

    _windows_only()
    outside, out, before = _linked_run(tmp_path)
    with open(out / MANIFEST_FILE, "rb"):  # (Python opens a file without sharing its deletion: held open)
        assert _cli_code(_attempt(tmp_path, "stopped", out)) == 3
        assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
        capsys.readouterr()
    assert (outside / MANIFEST_FILE).read_bytes() == before and verify_run_artifacts(outside) == ()


def test_a_read_only_hard_linked_manifest_stays_read_only_on_windows(tmp_path, capsys):
    import stat

    _windows_only()
    outside, out, before = _linked_run(tmp_path)
    _read_only(out / MANIFEST_FILE)
    try:
        assert _cli_code(_attempt(tmp_path, "usage", out)) == 64
        assert _cli_code(_complete(tmp_path, out, _other_design())) == 64
        capsys.readouterr()
        assert not (outside / MANIFEST_FILE).stat().st_mode & stat.S_IWRITE  # (still read-only)
        assert (outside / MANIFEST_FILE).read_bytes() == before and verify_run_artifacts(outside) == ()
    finally:
        _writable(out)
        _writable(outside)


# ------------------------------------------------------------------ a completed run replaces the Engine's names


OTHER_FILES = {"notes.txt": b"mine", "results.csv": b"a,b\n1,2\n", "simulation_result.backup.json": b"{}",
               "sub/manifest.json": b"{\"schema\": \"my.notes\"}", "configurations/myrun/notes.txt": b"mine"}


@pytest.mark.parametrize("own", ["project.json", "simulation_result.json", "manifest.json", "ray_data.json", "optical_scene.glb",
                                 "comparison.json"])
def test_a_completed_run_never_replaces_a_file_of_the_persons_own(tmp_path, capsys, own):
    """The review's case (``project run other.json --out-dir .`` with the person's own files there): a plain file at a
    run-file name that no Engine manifest there lists is the person's own. The run refuses to write (exit 64, naming it) and
    every file there stays as it was; there is no run there afterwards."""

    out = tmp_path / "work"
    out.mkdir()
    mine = json.dumps(_other_design()).encode() if own == "project.json" else b'{"mine": true}'
    (out / own).write_bytes(mine)
    for name, data in OTHER_FILES.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    before = _files(out)
    argv = (["project", "run", str(ZOOM), "--out-dir", str(out), "--progress", "none"] if own == "comparison.json"
            else _complete(tmp_path, out))
    assert _cli_code(argv) == 64
    err = capsys.readouterr().err
    assert "cannot write the run directory" in err and str(out / own) in err and "never replaces it" in err
    assert ("another Project" in err) == (own == "project.json")
    assert _files(out) == before and verify_run_artifacts(out)
    if own == "comparison.json":
        return
    # the JSON API's in-memory files are refused the same way
    from os_optics.local.artifacts import RunDirectoryOccupiedError, write_run_files

    with pytest.raises(RunDirectoryOccupiedError):
        write_run_files(run_artifact_files(run_optical_project(_project())), out)
    assert _files(out) == before


@pytest.mark.parametrize("own", [STOPPED_FILE, ".simulation_result.json.partial", ".manifest.json.partial"])
def test_a_completed_run_replaces_the_engine_s_own_names_and_touches_no_other(tmp_path, capsys, own):
    """The stop record and the temporary names are the Engine's own: a file there is removed by a run that completes. Every
    other file there is byte for byte the same."""

    out = tmp_path / "work"
    out.mkdir()
    (out / own).write_bytes(b'{"mine": true}')
    for name, data in OTHER_FILES.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    assert _cli_code(_complete(tmp_path, out)) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == ()
    assert set(_files(out)) == RUN_OUTPUTS | {"project.json", *OTHER_FILES}
    assert {name: _files(out)[name] for name in OTHER_FILES} == OTHER_FILES
    # the JSON API's write of files held in memory follows the same rule
    from os_optics.local.artifacts import write_run_files

    (out / own).write_bytes(b'{"mine": true}')
    write_run_files(run_artifact_files(run_optical_project(_project())), out)
    assert verify_run_artifacts(out) == () and set(_files(out)) == RUN_OUTPUTS | {"project.json", *OTHER_FILES}


def test_an_edited_file_of_an_earlier_run_is_replaced_by_the_next_run(tmp_path, capsys):
    """A file an earlier run wrote -- its manifest lists the name -- and the person re-saved since: the earlier run no
    longer verifies, and the next run that completes replaces it without reading it (no earlier run file is hashed)."""

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    (out / "simulation_result.json").write_text('{"edited": true}', encoding="utf-8")
    assert verify_run_artifacts(out)
    assert _cli_code(_complete(tmp_path, out)) == 0
    capsys.readouterr()
    assert verify_run_artifacts(out) == ()


def test_a_rerun_reads_and_hashes_no_file_of_the_earlier_run(tmp_path, monkeypatch):
    """A rerun reads the earlier manifest (to know which names that run held) and ``project.json`` (the one protected
    name) and nothing else: no run file is opened for reading."""

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    run = run_optical_project(_other_design())
    read: list[str] = []
    real_open, real_read_bytes = Path.open, Path.read_bytes

    def open_(self, mode="r", *args, **kwargs):
        if self.parent == out and "r" in mode and "+" not in mode:
            read.append(self.name)
        return real_open(self, mode, *args, **kwargs)

    def read_bytes(self):
        if self.parent == out:
            read.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "open", open_)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    run.write_artifacts(out)
    monkeypatch.undo()
    assert set(read) <= {MANIFEST_FILE, "project.json"}, read
    assert verify_run_artifacts(out) == ()


def test_a_project_json_of_another_design_that_no_manifest_lists_is_refused_and_unchanged(tmp_path, capsys):
    """``project.json`` is also the name of a design file (``project run other.json --out-dir .`` with the person's own
    design as ``./project.json``): one holding another design that no Engine manifest lists refuses the write (exit 64,
    naming it), and every file there stays as it was."""

    out = tmp_path / "work"
    out.mkdir()
    (out / "project.json").write_bytes(json.dumps(_other_design()).encode())
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    before = _files(out)
    assert _cli_code(_complete(tmp_path, out)) == 64
    err = capsys.readouterr().err
    assert "cannot write the run directory" in err and str(out / "project.json") in err and "never replaces it" in err
    assert "another Project" in err
    assert _files(out) == before and verify_run_artifacts(out)
    # the JSON API's in-memory files are refused the same way
    from os_optics.local.artifacts import RunDirectoryOccupiedError, write_run_files

    with pytest.raises(RunDirectoryOccupiedError):
        write_run_files(run_artifact_files(run_optical_project(_project())), out)
    assert _files(out) == before
    # and so is an attempt: it writes nothing but, when it stops, its record
    assert _cli_code(_attempt(tmp_path, "stopped", out)) == 3
    capsys.readouterr()
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before


@pytest.mark.parametrize("how", ["same_path", "same_design_elsewhere"])
def test_a_project_json_holding_the_project_being_run_is_written_again(tmp_path, capsys, how):
    """``./project.json`` holds the Project being run -- it is the input file itself (``project run project.json --out-dir
    .``) or another file of the same design (the same ``ProjectDocumentV2`` payload, whatever its formatting). The run
    writes it again in the canonical form, and the manifest records that file's SHA-256; the design in it is unchanged."""

    import hashlib

    from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2

    out = tmp_path / "work"
    out.mkdir()
    payload = ProjectDocumentV2.model_validate(_project()).payload()
    kept = _design_file(out, "project.json", payload, indent=4)  # (the person's formatting)
    formatted = kept.read_bytes()
    source = kept if how == "same_path" else _design_file(tmp_path, "design.json", payload, indent=1)
    assert _cli_code(["project", "run", str(source), "--out-dir", str(out), "--progress", "none"]) in (0, 1, 2)
    capsys.readouterr()
    assert verify_run_artifacts(out) == ()
    assert kept.read_bytes() != formatted and ProjectDocumentV2.model_validate(json.loads(kept.read_bytes())).payload() == payload
    listed = {entry["name"]: entry["sha256"] for entry in json.loads((out / MANIFEST_FILE).read_bytes())["files"]}
    assert listed["project.json"] == hashlib.sha256(kept.read_bytes()).hexdigest()
    # a different design at that name is never taken for it
    _design_file(out, "project.json", _other_design())
    (out / MANIFEST_FILE).unlink()  # (no earlier run lists it now)
    assert _cli_code(["project", "run", str(_design_file(tmp_path, "again.json", payload)), "--out-dir", str(out),
                      "--progress", "none"]) == 64
    assert "another Project" in capsys.readouterr().err


def test_a_run_into_its_own_earlier_run_directory_replaces_that_run(tmp_path, capsys):
    """An earlier run's ``project.json`` of another design, which its manifest lists with its bytes, is the Engine's: a
    new run of another design into the same directory replaces it."""

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    assert _cli_code(_complete(tmp_path, out)) in (0, 1, 2)
    capsys.readouterr()
    assert verify_run_artifacts(out) == () and _payload((out / "project.json").read_bytes()) == _payload(json.dumps(_project()).encode())


def test_a_stop_writes_its_record_over_a_file_of_that_name(tmp_path, capsys):
    """``run_stopped.json`` is the Engine's name in the directory the caller named: a stop writes its record there."""

    out = tmp_path / "run"
    out.mkdir()
    (out / STOPPED_FILE).write_text('{"mine": true}', encoding="utf-8")
    assert _cli_code(_attempt(tmp_path, "stopped", out)) == 3
    capsys.readouterr()
    assert json.loads((out / STOPPED_FILE).read_bytes())["kind"] == "STOPPED_RUN" and _staged(out) == []


# ------------------------------------------------------------------ a run killed between two renames, and the run after it


def _killed_after(monkeypatch, moved: int) -> None:
    """The process ends (``SystemExit``) when the write is about to move its ``moved + 1``-th file into place: the files
    before it are moved, the rest are still under their temporary names, and nothing is cleaned up."""

    from os_optics.output_files import StagedFiles

    real_replace = Path.replace
    count = {"moved": 0}

    def replace(self, target):
        if self.name.endswith(".partial"):
            if count["moved"] == moved:
                raise SystemExit("killed")
            count["moved"] += 1
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(StagedFiles, "discard", lambda self: None)  # (a killed process removes nothing)


@pytest.mark.parametrize("moved", [1, 3, 7])
def test_a_run_killed_between_two_renames_leaves_no_manifest_and_the_next_run_completes(tmp_path, monkeypatch, moved):
    """The commit of a run removes the manifest there first and moves its own manifest last, so a process killed between
    two renames -- after the first, in the middle, with only the manifest left -- leaves no manifest: no run. ``project.json``
    is moved first, so it holds the Project that was being run, and the same run started again replaces whatever files and
    temporary files the killed one left, and verifies."""

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    _killed_after(monkeypatch, moved)
    with pytest.raises(SystemExit):
        run_optical_project(_project(), output_dir=out)
    monkeypatch.undo()
    assert not (out / MANIFEST_FILE).exists()
    (problem,) = verify_run_artifacts(out)
    assert problem.startswith("not a run directory: it has no manifest.json")
    assert len(_staged(out)) == len(RUN_OUTPUTS) + 1 - moved  # (the files not yet moved, the manifest among them)
    assert _payload((out / "project.json").read_bytes()) == _payload(json.dumps(_project()).encode())
    from os_optics.results import ResultsError, load_run

    with pytest.raises(ResultsError):
        load_run(out)
    run_optical_project(_project(), output_dir=out)
    assert verify_run_artifacts(out) == () and _staged(out) == []
    assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", "notes.txt"}


def _third_design() -> dict:
    payload = json.loads(json.dumps(_project()))
    payload["components"][0]["nominal_pose"]["position_mm"] = [-300.0, 0.0, 0.0]  # (neither run's design)
    return payload


def test_a_kill_before_the_project_file_is_moved_leaves_the_copy_of_the_manifest_that_lists_it(tmp_path, monkeypatch):
    """Killed after the manifest is removed and before ``project.json`` is moved (two adjacent steps): the directory holds
    the earlier design's ``project.json`` and the copy of the earlier manifest (``.manifest.json.earlier``) that lists it
    with its bytes. The same run started again completes, and so does a run of the earlier design; a design of the
    person's own placed at ``project.json`` is still refused, naming the file. A completed run leaves no copy."""

    from os_optics.local.artifacts import EARLIER_MANIFEST_FILE, RunDirectoryOccupiedError

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    earlier = (out / MANIFEST_FILE).read_bytes()
    _killed_after(monkeypatch, 0)
    with pytest.raises(SystemExit):
        run_optical_project(_project(), output_dir=out)
    monkeypatch.undo()
    assert not (out / MANIFEST_FILE).exists() and (out / EARLIER_MANIFEST_FILE).read_bytes() == earlier
    kept = (out / "project.json").read_bytes()
    assert _payload(kept) == _payload(json.dumps(_other_design()).encode())
    (out / "project.json").write_bytes(json.dumps(_third_design()).encode())
    with pytest.raises(RunDirectoryOccupiedError, match="another Project"):
        run_optical_project(_project(), output_dir=out)
    (out / "project.json").write_bytes(kept)
    run_optical_project(_project(), output_dir=out)
    assert verify_run_artifacts(out) == () and _staged(out) == [] and not (out / EARLIER_MANIFEST_FILE).exists()
    assert _payload((out / "project.json").read_bytes()) == _payload(json.dumps(_project()).encode())
    run_optical_project(_other_design(), output_dir=out)
    assert verify_run_artifacts(out) == () and _staged(out) == [] and not (out / EARLIER_MANIFEST_FILE).exists()


def test_the_project_file_is_moved_first_and_the_manifest_last(tmp_path, monkeypatch):
    steps: list[str] = []
    real_replace, real_unlink = Path.replace, Path.unlink

    def replace(self, target):
        if tmp_path in Path(target).parents:
            steps.append("move " + Path(target).relative_to(tmp_path).as_posix())
        return real_replace(self, target)

    def unlink(self, missing_ok=False):
        if tmp_path in self.parents and not self.name.endswith(".partial"):
            steps.append("remove " + self.relative_to(tmp_path).as_posix())
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    run_optical_project(_project(), output_dir=tmp_path / "run")
    assert steps[:3] == ["remove run/run_stopped.json", "remove run/manifest.json", "move run/project.json"]
    assert steps[-1] == "move run/manifest.json" and len(steps) == 2 + len(RUN_OUTPUTS) + 1
    steps.clear()
    # over an earlier run, the copy of its manifest is written before anything is removed and removed after the manifest
    # is moved; project.json is replaced, never removed
    run_optical_project(_other_design(), output_dir=tmp_path / "run")
    assert steps[:3] == ["remove run/run_stopped.json", "remove run/manifest.json", "move run/project.json"]
    assert steps[-2:] == ["move run/manifest.json", "remove run/.manifest.json.earlier"]
    assert len(steps) == 2 + len(RUN_OUTPUTS) + 2 and (tmp_path / "run" / ".manifest.json.earlier").exists() is False
    steps.clear()
    json_api.run_project(ZOOM.read_text(encoding="utf-8"), output_dir=str(tmp_path / "zoom"))
    removed = [step for step in steps if step.startswith("remove ")]
    assert removed[-1] == "remove zoom/manifest.json" and removed == steps[:len(removed)]
    assert all(step.endswith(STOPPED_FILE) for step in removed[:len(removed) // 2])  # (the stop records, then the manifests)
    moved = steps[len(removed):]
    assert moved[0] == "move zoom/project.json" and moved[-1] == "move zoom/manifest.json"
    assert "move zoom/comparison.json" in moved and "move zoom/configurations/tele/manifest.json" in moved


# ------------------------------------------------------------------ runs kept under configurations/


@pytest.mark.parametrize("outcome", ["complete", "usage", "stopped"])
def test_a_run_kept_under_configurations_is_never_removed_by_another_run(tmp_path, capsys, outcome):
    """A complete run the person keeps at ``out/configurations/myrun`` is no part of the run in ``out`` (its manifest
    names no such configuration). A single run into ``out`` -- completed, refused as a usage error, stopped -- never
    touches it, and it verifies whatever happened above it."""

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    kept = out / "configurations" / "myrun"
    run_optical_project(_other_design(), output_dir=kept)
    before = _files(kept)
    argv = _complete(tmp_path, out) if outcome == "complete" else _attempt(tmp_path, outcome, out)
    code = _cli_code(argv)
    assert code in ((0, 1, 2) if outcome == "complete" else (64,) if outcome == "usage" else (3,))
    capsys.readouterr()
    assert _files(kept) == before and verify_run_artifacts(kept) == () and verify_run_artifacts(out) == ()


def test_a_run_of_configurations_never_replaces_a_run_kept_where_it_writes(tmp_path, capsys):
    """A run of configurations writes ``configurations/<id>``: a run the person keeps at one of those, which the
    directory's own manifest does not name, is refused (exit 64) and stays; other files there and other directories under
    ``configurations/`` stay too. An earlier run of the same configurations is the Engine's: its directories are replaced."""

    from os_optics.local import run_configured_project

    out = tmp_path / "zoom"
    out.mkdir()
    there = out / "configurations" / "wide"
    run_optical_project(_project(), output_dir=there)  # (a single run kept where the run of configurations writes)
    (there / "notes.txt").write_text("mine", encoding="utf-8")
    elsewhere = out / "configurations" / "myrun"
    run_optical_project(_other_design(), output_dir=elsewhere)
    before, kept = _files(elsewhere), _files(there)
    assert _cli_code(["project", "run", str(ZOOM), "--out-dir", str(out), "--progress", "none"]) == 64
    assert "holds a run that is no part of the earlier run there" in capsys.readouterr().err
    assert _files(there) == kept and verify_run_artifacts(there) == ()
    assert _files(elsewhere) == before and verify_run_artifacts(elsewhere) == ()
    # the person's notes in a configuration directory that holds no run stay; an earlier run of the same configurations is
    # replaced, and a later run of fewer configurations removes the others' run files by name
    other = tmp_path / "zoom2"
    (other / "configurations" / "wide").mkdir(parents=True)
    (other / "configurations" / "wide" / "notes.txt").write_text("mine", encoding="utf-8")
    assert _cli_code(["project", "run", str(ZOOM), "--out-dir", str(other), "--progress", "none"]) in (0, 1, 2, 3)
    assert _cli_code(["project", "run", str(ZOOM), "--out-dir", str(other), "--progress", "none"]) in (0, 1, 2, 3)
    capsys.readouterr()
    assert verify_run_artifacts(other) == ()
    run_configured_project(ZOOM.read_bytes(), output_dir=other, configuration_ids=["tele"])
    assert sorted(p.name for p in (other / "configurations").iterdir()) == ["tele", "wide"]
    assert sorted(p.name for p in (other / "configurations" / "wide").iterdir()) == ["notes.txt"]


def test_a_run_of_configurations_killed_between_two_renames_is_completed_by_the_next(tmp_path, monkeypatch):
    """Killed while the commit moves its files: no manifest, some configuration directories with their manifests moved,
    the others with theirs staged. The staged manifests the commit left name every file and directory it was moving, so the
    same run started again completes there and verifies."""

    from os_optics.local import run_configured_project

    out = tmp_path / "zoom"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    _killed_after(monkeypatch, 5)
    with pytest.raises(SystemExit):
        run_configured_project(ZOOM.read_bytes(), output_dir=out)
    monkeypatch.undo()
    assert not (out / MANIFEST_FILE).exists() and (out / ".manifest.json.partial").exists()
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    assert verify_run_artifacts(out) == () and _staged(out) == []


class _KilledAt(BaseException):
    """The process ends here (no handler, and none of the commit's own cleanup runs, as a kill would not run it)."""


def _counted_commit(monkeypatch, kill_at: int | None) -> dict[str, int]:
    """Counts every step a write takes in the run directory -- a name removed, a staged file moved into place, a directory
    removed -- and ends the process right after step ``kill_at`` (``_KilledAt``); nothing is cleaned up."""

    from os_optics.output_files import StagedFiles

    real_replace, real_unlink, real_rmdir = Path.replace, Path.unlink, Path.rmdir
    count = {"steps": 0}

    def step():
        count["steps"] += 1
        if count["steps"] == kill_at:
            raise _KilledAt

    def replace(self, target):
        moved = real_replace(self, target)
        if self.name.endswith(".partial"):
            step()
        return moved

    def unlink(self, missing_ok=False):
        real_unlink(self, missing_ok=missing_ok)
        if not self.name.endswith(".partial"):
            step()

    def rmdir(self):
        real_rmdir(self)
        step()

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    monkeypatch.setattr(StagedFiles, "discard", lambda self: None)  # (a killed process removes nothing)
    return count


def _single_files(design: dict) -> dict[str, bytes]:
    return run_artifact_files(run_optical_project(design))


def _write_files(files: dict[str, bytes], out: Path) -> None:
    from os_optics.local.artifacts import write_run_files
    from os_optics.local.configurations import is_configurations_manifest, write_configurations_files

    if is_configurations_manifest(json.loads(files[MANIFEST_FILE])):
        write_configurations_files(files, out)
    else:
        write_run_files(files, out)


def _zoom_files(configuration_ids: tuple[str, ...] | None = None) -> dict[str, bytes]:
    from os_optics.local import run_configured_project

    return configurations_run_files(run_configured_project(ZOOM.read_bytes(), configuration_ids=configuration_ids))


def _zoom_single_files() -> dict[str, bytes]:
    plain = json.loads(ZOOM.read_bytes())
    del plain["configurations"]
    plain["targets"] = [target for target in plain["targets"] if not target.get("configuration_ids")]
    return _single_files(plain)


_COMMIT_CASES = {
    # (the earlier run in the directory, the run whose commit is killed)
    "another design": lambda: (_single_files(_other_design()), _single_files(_project())),
    "a configuration dropped": lambda: (_zoom_files(), _zoom_files(("wide", "tele"))),
    "configurations over a single run": lambda: (_zoom_single_files(), _zoom_files()),
    "a single run over configurations": lambda: (_zoom_files(), _zoom_single_files()),
}


@pytest.mark.parametrize("case", sorted(_COMMIT_CASES))
def test_a_run_killed_after_any_step_of_its_commit_is_completed_by_the_same_run_and_replaced_by_the_earlier(tmp_path, monkeypatch,
                                                                                                          case):
    """The commit of a run is killed right after each of its steps in turn (every removal and every move, the commit's
    temporary files and copies left as a kill leaves them). After every one: the same run started again completes,
    verifies and leaves exactly its own files there -- nothing the earlier run held and it does not write -- and then the
    earlier run started again completes the same way: no kill leaves a file of the Engine's that a later run refuses. A
    design of the person's own placed at ``project.json`` is refused all the same."""

    from os_optics.local.artifacts import RunDirectoryOccupiedError

    earlier, later = _COMMIT_CASES[case]()
    out = tmp_path / "count"
    _write_files(earlier, out)
    with monkeypatch.context() as m:
        count = _counted_commit(m, None)
        _write_files(later, out)
    steps = count["steps"]
    assert steps > 3 and verify_run_artifacts(out) == () and set(_files(out)) == set(later)
    for kill_at in range(1, steps + 1):
        out = tmp_path / f"killed-{kill_at}"
        _write_files(earlier, out)
        with monkeypatch.context() as m:
            _counted_commit(m, kill_at)
            with pytest.raises(_KilledAt):
                _write_files(later, out)
        kept = (out / "project.json").read_bytes()
        (out / "project.json").write_bytes(json.dumps(_third_design()).encode())
        with pytest.raises(RunDirectoryOccupiedError, match="another Project"):
            _write_files(later, out)
        (out / "project.json").write_bytes(kept)
        _write_files(later, out)
        assert verify_run_artifacts(out) == () and set(_files(out)) == set(later), kill_at
        _write_files(earlier, out)
        assert verify_run_artifacts(out) == () and set(_files(out)) == set(earlier), kill_at
        # the earlier run started right after the kill, instead of the killed one, completes there too and leaves its own
        # files only (beside the temporary files the killed commit left at names it does not write)
        out = tmp_path / f"earlier-after-{kill_at}"
        _write_files(earlier, out)
        with monkeypatch.context() as m:
            _counted_commit(m, kill_at)
            with pytest.raises(_KilledAt):
                _write_files(later, out)
        _write_files(earlier, out)
        assert verify_run_artifacts(out) == (), kill_at
        assert {name for name in _files(out) if not name.endswith(".partial")} == set(earlier), kill_at


def _failing_commit(monkeypatch, fail_at: int) -> None:
    """The steps ``_counted_commit`` counts, where step ``fail_at`` is refused by the system (``PermissionError``, as
    Windows refuses to delete or replace a file another program holds open) instead of done; the commit's own cleanup
    runs."""

    real_replace, real_unlink, real_rmdir = Path.replace, Path.unlink, Path.rmdir
    count = {"steps": 0}

    def step(path):
        count["steps"] += 1
        if count["steps"] == fail_at:
            raise PermissionError(13, "refused", str(path))

    def replace(self, target):
        if self.name.endswith(".partial"):
            step(target)
        return real_replace(self, target)

    def unlink(self, missing_ok=False):
        if not self.name.endswith(".partial"):
            step(self)
        return real_unlink(self, missing_ok=missing_ok)

    def rmdir(self):
        step(self)
        return real_rmdir(self)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rmdir", rmdir)


@pytest.mark.parametrize("case", sorted(_COMMIT_CASES))
def test_a_run_whose_commit_fails_at_any_step_is_completed_by_the_same_run_and_replaced_by_the_earlier(tmp_path, monkeypatch, case):
    """The commit of a run fails at each of its steps in turn (the system refuses that removal or move, the commit cleans
    up after itself). Where the failure stops the commit, it leaves what a kill there leaves: no copy of a manifest beside
    the manifest it copies, and -- once the directory's manifest is gone -- the copies of the manifests removed. After
    every one: a design of the person's own placed at ``project.json`` is refused, the same run completes with exactly
    its files, the earlier run then completes; the earlier run started right after the failure completes too. Where the
    refused step is a name the commit removes after the moves (a file of the earlier run held open), the commit
    completes and records that name (``.manifest.json.leftover``): once it is released, the same run, the earlier run and
    the later one each complete with exactly their files."""

    from os_optics.local.artifacts import EARLIER_MANIFEST_FILE, RunDirectoryOccupiedError, _leftover_names

    earlier, later = _COMMIT_CASES[case]()
    out = tmp_path / "count"
    _write_files(earlier, out)
    with monkeypatch.context() as m:
        count = _counted_commit(m, None)
        _write_files(later, out)
    stopped = 0
    for fail_at in range(1, count["steps"] + 1):
        for then in ("later", "earlier"):
            out = tmp_path / f"failed-{fail_at}-{then}"
            _write_files(earlier, out)
            with monkeypatch.context() as m:
                _failing_commit(m, fail_at)
                try:
                    _write_files(later, out)
                except OSError:
                    failed = True
                else:
                    failed = False  # (a name the commit removes after the moves: it stays, and is recorded)
            if not failed:
                assert verify_run_artifacts(out) == (), fail_at
                # (a file of the earlier run that stayed is recorded; a copy the cleanup could not remove is the Engine's own,
                # beside the manifest, which the next commit replaces)
                left = {name for name in _files(out) if name not in later and not name.endswith(".partial")
                        and Path(name).name not in (EARLIER_MANIFEST_FILE, ".manifest.json.leftover")}
                assert left <= set(_leftover_names(out)), (fail_at, left)
                runs = [("later", later), ("earlier", earlier), ("later", later)] if then == "later" else [("earlier", earlier), ("later", later)]
                for name, files in runs:
                    _write_files(files, out)
                    assert verify_run_artifacts(out) == (), (fail_at, then, name)
                    assert {n for n in _files(out) if not n.endswith(".partial")} == set(files), (fail_at, then, name)
                continue
            stopped += then == "later"
            copies = [Path(name) for name in _tree(out) if Path(name).name == EARLIER_MANIFEST_FILE]
            assert not any((out / copy.parent / MANIFEST_FILE).exists() for copy in copies), fail_at
            if not (out / MANIFEST_FILE).exists():
                assert (out / EARLIER_MANIFEST_FILE).exists() and (out / ".manifest.json.partial").exists(), fail_at
            if then == "earlier":
                _write_files(earlier, out)
                assert verify_run_artifacts(out) == (), fail_at
                assert {name for name in _files(out) if not name.endswith(".partial")} == set(earlier), fail_at
                continue
            kept = (out / "project.json").read_bytes()
            (out / "project.json").write_bytes(json.dumps(_third_design()).encode())
            with pytest.raises(RunDirectoryOccupiedError, match="another Project"):
                _write_files(later, out)
            (out / "project.json").write_bytes(kept)
            _write_files(later, out)
            assert verify_run_artifacts(out) == () and set(_files(out)) == set(later), fail_at
            _write_files(earlier, out)
            assert verify_run_artifacts(out) == () and set(_files(out)) == set(earlier), fail_at
    assert stopped > len(RUN_OUTPUTS)  # (every removal before the moves and every move stops the commit)


@pytest.mark.parametrize("case", sorted(_COMMIT_CASES))
def test_a_run_killed_right_after_its_manifest_is_moved_while_a_file_is_held_leaves_that_file_recorded(tmp_path, monkeypatch,
                                                                                                      case):
    """The kill sweeps hold no file; here a file of the earlier run that the commit removes by name is held open (its delete
    refused) and the process is killed right after the directory's manifest is moved into place, before the copies are
    removed. The record of the held file (``.manifest.json.leftover``) is written before that move, so it lists the file;
    once it is released, the same run, the earlier run and the later one each complete with exactly their files -- no run
    refuses the Engine's own file as the person's."""

    from os_optics.local.artifacts import _leftover_names
    from os_optics.output_files import StagedFiles

    earlier, later = _COMMIT_CASES[case]()
    held = next((name for name in sorted(earlier) if name not in later and Path(name).name in
                 (RUN_OUTPUTS | {"comparison.json"}) - {MANIFEST_FILE}), None)
    out = tmp_path / "run"
    _write_files(earlier, out)
    real = StagedFiles.commit

    def killed_after(self, **kwargs):
        real(self, **kwargs)
        raise _KilledAt("after the manifest's move")

    with monkeypatch.context() as m:
        if held is not None:
            _refuse_on(m, [out / held])
        m.setattr(StagedFiles, "commit", killed_after)
        with pytest.raises(_KilledAt):
            _write_files(later, out)
    assert verify_run_artifacts(out) == ()
    if held is not None:
        assert (out / held).exists() and held in _leftover_names(out), (held, _tree(out))
    for files in (later, earlier, later):
        _write_files(files, out)
        assert verify_run_artifacts(out) == ()
        assert {name for name in _files(out) if not name.endswith(".partial")} == set(files), held


@pytest.mark.parametrize("how", ["killed", "failing"])
@pytest.mark.parametrize("case", sorted(_COMMIT_CASES))
def test_two_interrupted_commits_in_a_row_are_completed_by_either_run(tmp_path, monkeypatch, case, how):
    """The later run's commit is killed after each of its steps in turn; then the earlier run's commit is interrupted too
    -- killed, or failing as a file held open makes it -- at each of its first steps (keeping the first interruption's
    staged manifest under its own name, then the first removals). After every pair, one run and then the other completes
    there, verifies and leaves exactly its own files: the earlier run first, or the later one (the two orders alternate
    over the steps of the second interruption, so each order meets every kill of the first). No two interruptions leave
    an Engine file that a later run refuses as the person's own."""

    earlier, later = _COMMIT_CASES[case]()
    out = tmp_path / "count"
    _write_files(earlier, out)
    with monkeypatch.context() as m:
        count = _counted_commit(m, None)
        _write_files(later, out)
    interrupted = 0
    for kill_at in range(1, count["steps"] + 1):
        for second_at in (1, 2, 3):
            outs = []
            for first in (("earlier",) if (kill_at + second_at) % 2 else ("later",)):
                out = tmp_path / f"{kill_at}-{second_at}-{first}"
                _write_files(earlier, out)
                with monkeypatch.context() as m:
                    _counted_commit(m, kill_at)
                    with pytest.raises(_KilledAt):
                        _write_files(later, out)
                with monkeypatch.context() as m:
                    try:
                        if how == "killed":
                            _counted_commit(m, second_at)
                        else:
                            _failing_commit(m, second_at)
                        _write_files(earlier, out)
                    except (_KilledAt, OSError):
                        outs.append((first, out))
            if not outs:
                continue
            interrupted += 1
            for first, out in outs:
                order = [("earlier", earlier), ("later", later)] if first == "earlier" else [("later", later), ("earlier", earlier)]
                for name, files in order:
                    _write_files(files, out)
                    assert verify_run_artifacts(out) == (), (kill_at, second_at, first, name)
                    assert {n for n in _files(out) if not n.endswith(".partial")} == set(files), (kill_at, second_at, first, name)
    assert interrupted >= count["steps"]  # (every kill of the later run is followed by an interrupted earlier one)


# ------------------------------------------------------------------ links


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("linked", ["configurations/<id>", "configurations"])
def test_a_run_never_follows_a_link_out_of_the_run_directory(tmp_path, capsys, kind, linked):
    """A link under --out-dir to a directory elsewhere -- even one holding a complete run, manifest and all -- is neither
    followed nor removed by a usage error, a stopped run or a completed run."""

    import os
    import shutil

    from os_optics.local.artifacts import _is_link

    out, outside = tmp_path / "run", tmp_path / "outside"
    run_optical_project(_project(), output_dir=out)
    if linked == "configurations":
        shutil.copytree(out, outside / "sub")
        link = out / "configurations"
    else:
        shutil.copytree(out, outside)
        (out / "configurations").mkdir()
        link = out / "configurations" / "elsewhere"
    (outside / "keep.txt").write_text("mine", encoding="utf-8")
    _link_directory(link, outside, kind)
    assert _is_link(link)
    before = _files(outside)
    assert "manifest.json" in before or "sub/manifest.json" in before

    assert _cli_code(_attempt(tmp_path, "usage", out)) == 64
    capsys.readouterr()
    with pytest.raises(RunStopped):
        run_optical_project(_project(), monitor=_NoPreflight(max_artifact_bytes=1000), output_dir=out)
    run_optical_project(_project(), output_dir=out)
    assert _files(outside) == before
    assert os.path.lexists(link) and _is_link(link)
    assert verify_run_artifacts(out) == ()


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("linked", ["configurations/tele", "configurations"])
def test_a_run_never_removes_through_a_link_an_earlier_manifest_names(tmp_path, capsys, kind, linked):
    """The earlier run of configurations names ``configurations/tele``; that name (or ``configurations/``) is now a link
    to a directory elsewhere holding a run. A single run written there removes the earlier run's files by name -- and
    nothing behind the link."""

    import os
    import shutil

    from os_optics.local import run_configured_project
    from os_optics.local.artifacts import _is_link

    out, outside, other = tmp_path / "run", tmp_path / "outside", tmp_path / "other"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    run_optical_project(_project(), output_dir=other)
    shutil.rmtree(out / linked)
    shutil.copytree(other, outside / "tele" if linked == "configurations" else outside)
    (outside / "keep.txt").write_text("mine", encoding="utf-8")
    link = out / linked
    _link_directory(link, outside, kind)
    before = _files(outside)
    assert _cli_code(_complete(tmp_path, out)) == 0
    capsys.readouterr()
    assert _files(outside) == before and os.path.lexists(link) and _is_link(link)
    assert verify_run_artifacts(out) == ()


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("outcome", ["complete", "usage", "stopped"])
def test_an_out_dir_that_is_a_link_is_refused_and_nothing_behind_it_changes(tmp_path, capsys, kind, outcome):
    """``--out-dir`` that is itself a symbolic link or a junction: the run refuses it (exit 64; a usage error keeps its own
    64) and writes nothing through it -- the run kept behind the link is unchanged and still verifies."""

    import os

    from os_optics.local.artifacts import RunDirectoryLinkError, _is_link

    behind = tmp_path / "behind"
    run_optical_project(_project(), output_dir=behind)
    link = tmp_path / "out"
    _link_directory(link, behind, kind)
    before = _files(behind)
    argv = _complete(tmp_path, link) if outcome == "complete" else _attempt(tmp_path, outcome, link)
    assert _cli_code(argv) == 64
    err = capsys.readouterr().err
    if outcome != "usage":
        assert "is a link" in err
    assert _files(behind) == before and verify_run_artifacts(behind) == () and os.path.lexists(link) and _is_link(link)
    with pytest.raises(RunDirectoryLinkError):
        run_optical_project(_project(), output_dir=link)
    reply = json_api.run_project(json.dumps(_project()), output_dir=str(link))  # (the JSON API replies with the refusal)
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "RUN_DIRECTORY_LINK"
    assert _files(behind) == before


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("linked", ["configurations/tele", "configurations"])
def test_a_run_never_writes_through_a_link_under_its_directory(tmp_path, capsys, kind, linked):
    """A link named like a directory the run writes -- configurations/, or configurations/<id> of a configuration it
    runs -- to a directory elsewhere holding a run of another Project: the run refuses to write (exit 64), the link and
    everything behind it stay as they were, and so does the run that was in the directory; the JSON API's write refuses
    the same way."""

    import os
    import shutil

    from os_optics.local.artifacts import _is_link
    from os_optics.local import run_configured_project

    out, outside, other = tmp_path / "run", tmp_path / "outside", tmp_path / "other"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    run_optical_project(_project(), output_dir=other)
    shutil.rmtree(out / linked)
    target = outside / "tele" if linked == "configurations" else outside
    shutil.copytree(other, target)
    (outside / "keep.txt").write_text("mine", encoding="utf-8")
    link = out / linked
    _link_directory(link, outside, kind)
    before, here = _files(outside), {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}

    assert _cli_code(["project", "run", str(ZOOM), "--out-dir", str(out), "--progress", "none"]) == 64
    err = capsys.readouterr().err
    assert "cannot write the run directory" in err and "is a link" in err
    assert _files(outside) == before and os.path.lexists(link) and _is_link(link)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == here and _staged(out) == []  # (nothing was written)
    reply = json_api.run_project(ZOOM.read_text(encoding="utf-8"), output_dir=str(out))
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "RUN_DIRECTORY_LINK"
    assert _files(outside) == before and _is_link(link)
    if linked == "configurations/tele":  # a run of the other configurations writes beside the link, never through it
        assert _cli_code(["project", "run", str(ZOOM), "--out-dir", str(out), "--configuration", "wide",
                          "--progress", "none"]) == 0
        capsys.readouterr()
        assert _files(outside) == before and _is_link(link)
        assert verify_run_artifacts(out / "configurations" / "wide") == ()


@pytest.mark.parametrize("name", ["simulation_result.json", ".simulation_result.json.partial", ".manifest.json.partial",
                                  "manifest.json", STOPPED_FILE])
def test_a_run_never_writes_through_a_file_link(tmp_path, name):
    """A run file name, its temporary name or the stop record's name that is a symbolic link to a file elsewhere: the
    write is refused before anything is written, and the file behind it is unchanged."""

    import os

    from os_optics.local.artifacts import RunDirectoryLinkError, _is_link

    out, outside = tmp_path / "run", tmp_path / "outside.json"
    out.mkdir()
    outside.write_text("{\"mine\": 1}", encoding="utf-8")
    try:
        os.symlink(outside, out / name)
    except (OSError, NotImplementedError) as exc:  # (Windows without the symbolic-link privilege)
        pytest.skip(f"no symbolic link can be created here: {exc}")
    with pytest.raises(RunDirectoryLinkError):
        run_optical_project(_project(), output_dir=out)
    assert outside.read_text(encoding="utf-8") == "{\"mine\": 1}" and _is_link(out / name)
    assert _tree(out) == [name]
    reply = json_api.run_project(json.dumps(_project()), output_dir=str(out))
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "RUN_DIRECTORY_LINK"
    assert outside.read_text(encoding="utf-8") == "{\"mine\": 1}"
    # a stop writes no record through a link either
    with pytest.raises(RunStopped):
        run_optical_project(_project(), monitor=_StopIn("trace"), output_dir=out)
    assert outside.read_text(encoding="utf-8") == "{\"mine\": 1}" and _is_link(out / name)


# ------------------------------------------------------------------ a write stopped part-way changes nothing


def _ten_sources() -> dict:
    """A simple Project of ten sources, whose run files pass 1 MiB."""

    return project_of([laser(f"S{i}", (-100.0, -9.0 + 2.0 * i, 0.0)) for i in range(10)],
                      [screen("P", (100.0, 0.0, 0.0), radius_mm=20.0), screen("Q", (150.0, 0.0, 0.0), radius_mm=20.0)],
                      [power_target("t-P", "P", 1e-4)])


@pytest.mark.parametrize("kind", ["ten_sources", "configurations"])
def test_a_run_stopped_while_it_writes_keeps_the_project_file_it_runs_in_place(tmp_path, capsys, monkeypatch, kind):
    """The acceptance case: ``os-optics project run project.json --out-dir . --max-artifact-mib 1 --no-preflight``
    in the Project's own directory stops at the artifact limit while it writes (exit 3). The person's ``project.json``
    -- the input itself -- stays exactly as it was, beside the stop record; there is no run there. The same command
    without the limit then completes in place, and a re-run of it too."""

    from os_optics.cli_exit import STATUS_CONTRACT_ENV

    monkeypatch.delenv(STATUS_CONTRACT_ENV, raising=False)
    out = tmp_path / "work"
    out.mkdir()
    payload = _ten_sources() if kind == "ten_sources" else json.loads(ZOOM.read_text(encoding="utf-8"))
    mine = _design_file(out, "project.json", payload, indent=4)  # (the person's formatting)
    before = mine.read_bytes()
    monkeypatch.chdir(out)
    argv = ["project", "run", "project.json", "--out-dir", "."]
    assert _cli_code([*argv, "--max-artifact-mib", "1", "--no-preflight", "--progress", "none"]) == 3
    stop = json.loads(capsys.readouterr().out)["stop"]
    assert (stop["cause"], stop["phase"]) == ("ARTIFACT_LIMIT", "write")
    assert _tree(out) == ["project.json", STOPPED_FILE]
    assert mine.read_bytes() == before
    (problem,) = verify_run_artifacts(out)
    assert "stopped before it completed" in problem
    for _ in range(2):  # (the in-place run completes, and so does its re-run)
        assert _cli_code([*argv, "--progress", "none"]) in (0, 1, 2)
        capsys.readouterr()
        assert verify_run_artifacts(out) == () and STOPPED_FILE not in _tree(out)
        assert _payload(mine.read_bytes()) == _payload(before)  # (written again in canonical form: the same design)


class _StopWhileWriting(RunMonitor):
    """Stops the write once more than ``after`` run-file bytes would be kept: cancelled (``cancel``) or Ctrl-C."""

    def __init__(self, after: int, how: str, **kw):
        super().__init__(**kw)
        self.after, self.how = after, how

    def artifact_bytes(self, written):
        if self.artifact_bytes_written + written > self.after:
            if self.how == "ctrl_c":
                raise KeyboardInterrupt
            self.cancel("cancelled while writing")
        super().artifact_bytes(written)


def _fail_writing(monkeypatch, name: str, error: OSError, where: Path | None = None) -> None:
    """Writing the run file ``name`` (in the directory ``where``; None: in any) fails with ``error`` (a full disk, a
    refused write): its temporary file cannot be created."""

    real_open = Path.open

    def open_(self, mode="r", *args, **kwargs):
        if self.name == f".{name}.partial" and "x" in mode and (where is None or self.parent == where):
            raise error
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)


def _sizes(design) -> tuple[int, int]:
    """The bytes of a run's ``project.json`` + ``engine_request.json``, and of all its run files."""

    files = run_artifact_files(run_optical_project(design))
    return len(files["project.json"]) + len(files["engine_request.json"]), sum(len(v) for v in files.values())


WRITE_STOPS = ["artifact_limit_in_the_result", "artifact_limit_at_the_manifest", "cancelled", "ctrl_c", "disk_full",
               "manifest_refused"]


def _stop_the_write(monkeypatch, stop: str, design) -> RunMonitor | None:
    import errno

    head, total = _sizes(design)
    if stop == "artifact_limit_in_the_result":
        return _NoPreflight(max_artifact_bytes=head + 10)
    if stop == "artifact_limit_at_the_manifest":  # (every other file staged; the manifest's bytes pass the limit)
        return _NoPreflight(max_artifact_bytes=total - 1)
    if stop in ("cancelled", "ctrl_c"):
        return _StopWhileWriting(head + 10, stop)
    if stop == "disk_full":
        _fail_writing(monkeypatch, "ray_data.json", OSError(errno.ENOSPC, "No space left on device"))
    else:  # (every run file staged, the manifest's temporary file refused)
        _fail_writing(monkeypatch, MANIFEST_FILE, PermissionError(errno.EACCES, "Permission denied"))
    return None


@pytest.mark.parametrize("there", ["the_input", "the_earlier_run", "nothing"])
@pytest.mark.parametrize("stop", WRITE_STOPS)
def test_a_write_stopped_part_way_leaves_the_directory_as_it_found_it(tmp_path, monkeypatch, there, stop):
    """A write stopped part-way -- an artifact limit, a cancellation, Ctrl-C, a full disk, a manifest that cannot be
    written -- happens while the files are under their temporary names: they are removed and nothing else has changed. The
    Project being run in place (``project run project.json --out-dir .``) stays byte for byte; an earlier run there stays
    whole and verifies; a stop adds its record."""

    out = tmp_path / "work"
    if there == "the_input":
        out.mkdir()
        _design_file(out, "project.json", _project(), indent=4)  # (the person's formatting)
    elif there == "the_earlier_run":
        run_optical_project(_other_design(), output_dir=out)
    before = _files(out) if out.exists() else {}
    monitor = _stop_the_write(monkeypatch, stop, _project())
    with pytest.raises(RunStopped if monitor is not None else OSError):
        run_optical_project(_project(), monitor=monitor, output_dir=out)
    monkeypatch.undo()
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and _staged(out) == []
    assert (out / STOPPED_FILE).exists() == (monitor is not None)
    assert (verify_run_artifacts(out) == ()) == (there == "the_earlier_run")
    run_optical_project(_project(), output_dir=out)  # (then the run completes there, whatever was in the directory)
    assert verify_run_artifacts(out) == () and STOPPED_FILE not in _tree(out)
    assert _payload((out / "project.json").read_bytes()) == _payload(json.dumps(_project()).encode())


def test_a_write_from_memory_stopped_part_way_leaves_the_directory_as_it_found_it(tmp_path, monkeypatch):
    """The JSON API's write of a run's files held in memory: a write error leaves the ``project.json`` that was there as
    it was, and no file of the attempt (a later run then writes there)."""

    import errno

    out = tmp_path / "work"
    out.mkdir()
    mine = _design_file(out, "project.json", _project(), indent=4)
    before = mine.read_bytes()
    for name in ("ray_data.json", MANIFEST_FILE):
        _fail_writing(monkeypatch, name, OSError(errno.ENOSPC, "No space left on device"))
        reply = None
        with pytest.raises(OSError):
            reply = json_api.run_project(json.dumps(_project()), output_dir=str(out))
        monkeypatch.undo()
        assert reply is None and mine.read_bytes() == before and _tree(out) == ["project.json"]
    reply = json_api.run_project(json.dumps(_project()), output_dir=str(out))
    assert reply["ok"] and reply["verify"] == []


def _configured(cid: str) -> bytes:
    from os_optics.local.catalog import load_project

    return json.dumps(load_project(ZOOM.read_bytes()).configured(cid).payload(), indent=4).encode()


@pytest.mark.parametrize("stop", ["artifact_limit", "manifest_refused"])
def test_a_write_of_configurations_stopped_part_way_leaves_the_directory_as_it_found_it(tmp_path, monkeypatch, stop):
    """A run of configurations in place (``project.json`` holds the Project with its configurations), and a
    ``configurations/tele/project.json`` holding that configuration's Project: a write stopped part-way -- at the
    artifact limit, or when its manifest cannot be written, after every configuration was staged -- leaves both as
    they were, and no file or directory of the attempt (``comparison.json``, ``configurations/wide``) stays."""

    import errno

    from os_optics.local import run_configured_project

    out = tmp_path / "zoom"
    (out / "configurations" / "tele").mkdir(parents=True)
    top = _design_file(out, "project.json", json.loads(ZOOM.read_text(encoding="utf-8")), indent=4)
    tele = out / "configurations" / "tele" / "project.json"
    tele.write_bytes(_configured("tele"))
    before, tree = _files(out), _tree(out)
    if stop == "artifact_limit":
        files = configurations_run_files(run_configured_project(ZOOM.read_bytes()))
        # (the configurations' run files are counted: the limit stops the last one, tele, at its manifest)
        context = monitored(_NoPreflight(max_artifact_bytes=sum(len(v) for k, v in files.items() if "/" in k) - 1))
    else:
        # (the top manifest only: every configuration, comparison.json and project.json staged first)
        _fail_writing(monkeypatch, MANIFEST_FILE, PermissionError(errno.EACCES, "Permission denied"), where=out)
        context = monitored(None)
    with pytest.raises(RunStopped if stop == "artifact_limit" else OSError), context:
        run_configured_project(top.read_bytes(), output_dir=out)
    monkeypatch.undo()
    # (a stop adds its record; nothing else is there that was not there before)
    stopped = [STOPPED_FILE] if stop == "artifact_limit" else []
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before and _tree(out) == sorted([*tree, *stopped])
    assert verify_run_artifacts(out)
    run_configured_project(top.read_bytes(), output_dir=out)  # (then the run completes there)
    assert verify_run_artifacts(out) == ()


@pytest.mark.parametrize("stop", WRITE_STOPS)
def test_a_configuration_run_in_place_stays_whole_when_a_write_into_it_stops(tmp_path, monkeypatch, stop):
    """``project run configurations/tele/project.json --out-dir configurations/tele`` in a run of configurations: when
    the write stops part-way, the configuration's run directory is the earlier run's, whole, and the run of
    configurations above it still verifies."""

    from os_optics.local import run_configured_project

    out = tmp_path / "zoom"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    tele = out / "configurations" / "tele"
    before = _files(out)
    design = json.loads((tele / "project.json").read_bytes())
    monitor = _stop_the_write(monkeypatch, stop, design)
    with pytest.raises(RunStopped if monitor is not None else OSError):
        run_optical_project(design, monitor=monitor, output_dir=tele)
    monkeypatch.undo()
    assert {k: v for k, v in _files(out).items() if not k.endswith(STOPPED_FILE)} == before and _staged(out) == []
    assert verify_run_artifacts(tele) == () and verify_run_artifacts(out) == ()


# ------------------------------------------------------------------ a hand-edited manifest

_DEEP = 100_000


@pytest.mark.parametrize("edit", ["unhashable_name", "files_not_a_list", "odd_entries", "nested_too_deeply", "outside_name",
                                  "outside_configuration"])
@pytest.mark.parametrize("outcome", ATTEMPTS)
def test_a_hand_edited_manifest_never_crashes_a_run_and_names_nothing_outside(tmp_path, capsys, edit, outcome):
    """An earlier manifest of any shape: an attempt that does not complete never reads it and changes nothing; a run that
    completes takes from it only well-formed Engine file names and ``configurations/<id>`` directories (a name that
    leaves the directory names nothing), replaces the run and verifies. A manifest that lists nothing (its file list no
    list, too deeply nested to read, or no Engine manifest at all) proves no file there the Engine's: the run refuses to
    replace them (exit 64, naming one) until they are moved away. No traceback, and a file outside is untouched."""

    out, outside = tmp_path / "run", tmp_path / "outside"
    run_optical_project(_project(), output_dir=out)
    outside.mkdir()
    precious = outside / "precious.json"
    precious.write_text("{\"mine\": 1}", encoding="utf-8")
    (outside / "manifest.json").write_text("{\"mine\": 2}", encoding="utf-8")
    manifest = json.loads((out / MANIFEST_FILE).read_bytes())
    import hashlib

    if edit == "unhashable_name":
        manifest["files"].append({"name": ["x"], "sha256": "0" * 64})
    elif edit == "files_not_a_list":
        manifest["files"] = 5
    elif edit == "odd_entries":
        manifest["files"] += [1, "x", None, True, {"name": {"a": 1}}, {"name": "ray_data.json", "sha256": 7}]
    elif edit == "outside_name":
        manifest["files"].append({"name": "../outside/precious.json",
                                  "sha256": hashlib.sha256(precious.read_bytes()).hexdigest()})
    elif edit == "outside_configuration":
        manifest["schema"] = "opensci.configurations_run"
        manifest["configurations"] = [{"directory": "configurations/../../outside"}, {"directory": "../outside"},
                                      {"directory": 7}, None, {"directory": "configurations/.."}]
    text = json.dumps(manifest)
    if edit == "nested_too_deeply":
        text = text[:-1] + ", \"deep\": " + "[" * _DEEP + "]" * _DEEP + "}"
    (out / MANIFEST_FILE).write_text(text, encoding="utf-8")
    before = _files(out)
    assert _cli_code(_attempt(tmp_path, outcome, out)) == EXPECTED_CODE[outcome]
    err = capsys.readouterr().err
    assert "Traceback" not in err and "remains" not in err
    assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before
    assert (out / STOPPED_FILE).exists() == (outcome == "stopped")
    if edit in ("files_not_a_list", "nested_too_deeply", "outside_configuration"):
        assert _cli_code(_complete(tmp_path, out)) == 64
        err = capsys.readouterr().err
        assert "Traceback" not in err and "is no file of an earlier run there" in err
        assert {k: v for k, v in _files(out).items() if k != STOPPED_FILE} == before
        for leftover in out.iterdir():
            if leftover.is_file() and leftover.name != "project.json":
                leftover.unlink()
    # the next run that completes replaces the run
    assert _cli_code(_complete(tmp_path, out)) == 0
    assert "Traceback" not in capsys.readouterr().err
    assert verify_run_artifacts(out) == () and set(_tree(out)) == RUN_OUTPUTS | {"project.json"}
    assert _files(outside) == {"precious.json": b"{\"mine\": 1}", "manifest.json": b"{\"mine\": 2}"}


# ------------------------------------------------------------------ replay of a large run: lowering, the reply, estimates, Ctrl-C


class _StopAtStep(RunMonitor):
    """Cancels the run at the ``at``-th checkpoint of phase ``name``."""

    def __init__(self, name: str, at: int, **kw):
        super().__init__(**kw)
        self.name, self.at = name, at

    def step(self, count):
        if self.stack and self.stack[-1][0] == self.name and self.stack[-1][1] + count >= self.at:
            self.cancel(f"cancelled at step {self.at} of {self.name}")
        super().step(count)


def _catalog_design():
    """A reference design of catalog parts: the catalog is read, each component lowered, each body and pair checked."""
    from os_optics.local.example import example_project_of_kind

    return example_project_of_kind("folded_expander")


def test_lowering_reports_progress_at_each_component_and_body_and_stops_inside_it(tmp_path):
    from os_optics.local.catalog import load_project
    from os_optics.local.lowering import lower_project
    from os_optics.optical_solids import placed_elements

    design = _catalog_design()
    plain = run_optical_project(design)
    events: list[dict] = []
    watched = run_optical_project(design, monitor=RunMonitor(events.append, interval_s=0.0))
    assert run_artifact_files(watched) == run_artifact_files(plain)  # (the checkpoints change no byte)
    lowering = [e["done"] for e in events if e["phase"] == "lower" and e["event"] == "progress"]
    done = next(e["done"] for e in events if e["phase"] == "lower" and e["event"] == "phase_ended")
    # the catalog read, each component, each placed body, each pair of bodies, then the lowering as a whole
    document = load_project(design)
    lowered = lower_project(document)
    bodies = len(placed_elements(lowered.elements, {c.instance_id: c.nominal_pose for c in document.components})[0])
    assert bodies >= 3 and done >= 1 + len(document.components) + bodies + bodies * (bodies - 1) // 2 + 1
    assert lowering == list(range(1, done + 1))  # (a report at every checkpoint with interval 0, in order)
    again: list[dict] = []
    run_optical_project(design, monitor=RunMonitor(again.append, interval_s=0.0))
    assert next(e["done"] for e in again if e["phase"] == "lower" and e["event"] == "phase_ended") == done  # (deterministic)
    # a Stop inside the lowering: in the component loop, and at its last checkpoint inside the solids check
    out = tmp_path / "run"
    for at in (2, done - 1):
        run_optical_project(design, output_dir=out)
        with pytest.raises(RunStopped) as stopped:
            run_optical_project(design, monitor=_StopAtStep("lower", at), output_dir=out)
        stop = stopped.value.stopped.stop
        assert (stop.phase, stop.done, stop.cause, stop.completed_phases) == ("lower", at, "CALLER", ())
        # (the earlier run stays; the stop record beside it names the attempt)
        assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", STOPPED_FILE} and verify_run_artifacts(out) == ()
        assert _later_attempt(out)["phase"] == "lower"


def test_the_json_api_reports_its_reply_as_the_last_phase_and_can_stop_there():
    events: list[dict] = []
    with monitored(RunMonitor(events.append, interval_s=0.0)):
        reply = json_api.run_project(_project())
    assert reply["ok"] is True
    top = [e["phase"] for e in events if e["event"] == "phase_ended" and len(e["phases"]) == 1]
    assert top[-2:] == ["write", "reply"] and "reply" in RUN_PHASES
    assert "reply" in reply["progress"]["phase_seconds"] and "reply" in reply["progress"]["completed_phases"]
    plain = json_api.run_project(_project())
    assert {k: v for k, v in reply.items() if k != "progress"} == plain  # (the reply is the unmonitored one)
    with monitored(_StopIn("reply")):
        stopped = json_api.run_project(_project())
    assert stopped["ok"] is False and stopped["error"]["code"] == "RUN_STOPPED"
    assert stopped["stopped"]["stop"]["phase"] == "reply" and "files" not in stopped


def test_the_estimate_before_the_trace_starts_from_the_memory_once_the_project_is_lowered():
    from os_optics.local.resources import RUN_MEMORY_PER_BEAM

    def probe():  # (the catalog read and the lowering hold 200 MiB more than the process before the run)
        return (300 if monitor is not None and "lower" in monitor.completed else 100) << 20

    monitor = None
    monitor = RunMonitor(memory_probe=probe, memory_interval_s=0.0)
    run_optical_project(_project(), monitor=monitor)
    before = monitor.estimates[0]
    assert before["stage"] == "before_trace" and before["memory_bytes"] == (300 << 20) + 3 * RUN_MEMORY_PER_BEAM


# Measured runs (os-optics 0.13.0, Windows 10, CPython 3.14, OPENBLAS_NUM_THREADS=1, unless marked Linux; run-file bytes are
# the same on every platform): (case, ray_data, pupil rays drawn, beam paths, steps (trace segments), optical elements, rays
# a beam, run-file bytes, memory added after the trace: the monitor's peak above its reading when the trace ended). "tapwin xN": one beam
# through N stages of a 10 % tap and a bare 2 mm window (a composition scale system), "core:N": benchmarks/bench_scale.py core-N, "example:K": project example --kind K.
MEASURED_RUNS = [
    ("tapwin x4", "full", True, 5, 23, 8, 97, 2_171_960, 8_450_048),
    ("tapwin x16", "full", True, 17, 185, 32, 97, 11_315_363, 34_361_344),
    ("tapwin x32", "full", True, 23, 396, 64, 97, 22_170_952, 46_551_040),
    ("tapwin x64", "full", True, 23, 578, 128, 97, 34_443_939, 98_340_864),
    ("tap x16", "full", True, 17, 153, 16, 97, 8_495_941, 9_969_664),
    ("tapwin x16", "summary", True, 17, 185, 32, 97, 4_354_068, 34_467_840),
    ("tapwin x16", "full", False, 17, 185, 32, 97, 10_696_972, 33_095_680),
    ("tapwin x4", "summary", False, 5, 23, 8, 97, 947_881, 8_118_272),
    ("core:625", "full", True, 1254, 3739, 7, 97, 232_642_857, 244_129_792),
    ("core:625", "summary", True, 1254, 3739, 7, 97, 38_437_921, 247_795_712),
    ("core:2500", "summary", True, 5004, 15014, 7, 97, 152_630_112, 869_150_720),
    ("example:focus", "full", True, 1, 4, 1, 97, 391_626, 888_832),
    ("example:folded", "full", True, 1, 5, 2, 97, 487_785, 700_416),
    ("example:beam_expander", "full", True, 1, 6, 2, 97, 590_025, 1_875_968),
    ("example:relay_4f", "full", True, 1, 6, 2, 97, 555_405, 1_486_848),
    ("example:folded_expander", "full", True, 1, 7, 3, 97, 690_896, 2_084_864),
    ("example:dichroic_routing", "full", True, 4, 10, 1, 97, 812_377, 40_960),
    ("example:polarization", "full", True, 1, 4, 2, 97, 368_597, 1_114_112),
    ("example:two_beam_interference", "full", True, 2, 8, 3, 97, 756_746, 4_239_360),
    ("example:two_color_dichroic", "full", True, 4, 6, 3, 97, 777_628, 2_265_088),
    ("example:dichroic_combiner", "full", True, 4, 6, 2, 97, 708_493, 1_961_984),
    # (Linux 7.0, CPython 3.14.7, the resident set: the same run files, more memory added after the trace)
    ("tapwin x4 (Linux)", "full", True, 5, 23, 8, 97, 2_171_960, 19_402_752),
    ("tapwin x64 (Linux)", "full", True, 23, 578, 128, 97, 34_443_939, 97_251_328),
    # (os-optics 0.13.0 after the review of the estimate, Windows 10: the reference designs with summary ray data, with and
    # without pupil rays drawn; thin elements (waveplates) and lenses placed beside the beam, which the beam never meets.
    # "beside": waveplates at x = -80 + 20 (i mod 10), y = 40 + 25 (i div 10); lenses
    # at y = 40 (i + 1))
    ("example:focus", "summary", True, 1, 4, 1, 97, 202_903, 794_624),
    ("example:focus", "summary", False, 1, 4, 1, 97, 173_462, 2_469_888),
    ("example:folded", "summary", True, 1, 5, 2, 97, 267_772, 1_015_808),
    ("example:folded", "summary", False, 1, 5, 2, 97, 231_909, 1_839_104),
    ("example:beam_expander", "summary", True, 1, 6, 2, 97, 337_455, 1_503_232),
    ("example:beam_expander", "summary", False, 1, 6, 2, 97, 295_286, 983_040),
    ("example:relay_4f", "summary", True, 1, 6, 2, 97, 312_962, 3_276_800),
    ("example:relay_4f", "summary", False, 1, 6, 2, 97, 270_729, 1_781_760),
    ("example:folded_expander", "summary", True, 1, 7, 3, 97, 404_527, 3_698_688),
    ("example:folded_expander", "summary", False, 1, 7, 3, 97, 355_928, 2_187_264),
    ("example:dichroic_routing", "summary", True, 4, 10, 1, 97, 281_749, 208_896),
    ("example:dichroic_routing", "summary", False, 4, 10, 1, 97, 201_791, 2_617_344),
    ("example:polarization", "summary", True, 1, 4, 2, 97, 203_288, 2_691_072),
    ("example:polarization", "summary", False, 1, 4, 2, 97, 173_975, 40_960),
    ("example:two_beam_interference", "summary", True, 2, 8, 3, 97, 396_412, 8_769_536),
    ("example:two_beam_interference", "summary", False, 2, 8, 3, 97, 337_445, 5_378_048),
    ("example:two_color_dichroic", "summary", True, 4, 6, 3, 97, 259_480, 2_043_904),
    ("example:two_color_dichroic", "summary", False, 4, 6, 3, 97, 205_314, 1_048_576),
    ("example:dichroic_combiner", "summary", True, 4, 6, 2, 97, 231_465, 1_568_768),
    ("example:dichroic_combiner", "summary", False, 4, 6, 2, 97, 178_608, 176_128),
    ("polarization + 20 waveplates beside", "full", True, 1, 4, 22, 97, 1_304_427, 4_829_184),
    ("polarization + 20 waveplates beside", "summary", True, 1, 4, 22, 97, 1_139_118, 4_841_472),
    ("polarization + 60 waveplates beside", "full", True, 1, 4, 62, 97, 3_177_334, 29_990_912),
    ("polarization + 60 waveplates beside", "summary", True, 1, 4, 62, 97, 3_012_025, 29_917_184),
    ("polarization + 300 waveplates beside", "full", True, 1, 4, 302, 97, 14_426_417, 195_440_640),
    ("polarization + 300 waveplates beside", "summary", True, 1, 4, 302, 97, 14_261_108, 192_937_984),
    ("focus + 10 lenses beside", "full", True, 1, 4, 11, 97, 1_287_401, 2_789_376),
    ("focus + 10 lenses beside", "summary", True, 1, 4, 11, 97, 1_098_678, 5_918_720),
    ("focus + 40 lenses beside", "full", True, 1, 4, 41, 97, 3_978_554, 5_836_800),
    ("focus + 40 lenses beside", "summary", True, 1, 4, 41, 97, 3_789_831, 6_545_408),
]
# The estimate before the trace for one beam (OTHER_FILE_BYTES_PER_BEAM + RAY_DATA_BYTES_PER_BEAM["full"]) against the
# tapwin runs above: the observation of that replay (372 KB estimated, 2.2 to 34 MB written).
ONE_BEAM_BEFORE_TRACE = 372_493


@pytest.mark.parametrize("case", MEASURED_RUNS, ids=lambda c: f"{c[0]}-{c[1]}-{'ring' if c[2] else 'none'}")
def test_every_measured_run_lies_in_the_range_of_its_estimate_after_the_trace(case):
    from os_optics.local.resources import (AFTER_TRACE_FILES_HIGH, AFTER_TRACE_MEASURED_RATIO, AFTER_TRACE_MEMORY_HIGH,
                                                AFTER_TRACE_MEMORY_MARGIN, after_trace_files, after_trace_memory_growth)

    _name, level, drawn, paths, steps, elements, rays, files, growth = case
    estimate = after_trace_files(paths, steps, elements, rays_per_beam=rays, ray_data=level, pupil_rays_drawn=drawn)
    low, high = AFTER_TRACE_MEASURED_RATIO
    # (the stated ratio of the measured runs, and the high end of the range; the low end is no fit: what a run writes for
    # certain, test_the_low_end_after_the_trace_is_what_the_run_writes_for_certain)
    assert low * estimate <= files <= high * estimate <= AFTER_TRACE_FILES_HIGH * estimate
    assert growth <= AFTER_TRACE_MEMORY_HIGH * after_trace_memory_growth(paths, steps, elements) + AFTER_TRACE_MEMORY_MARGIN


def test_the_measured_ratios_are_the_ones_stated():
    from os_optics.local.resources import AFTER_TRACE_MEASURED_RATIO, after_trace_files

    ratios = [files / after_trace_files(paths, steps, elements, rays_per_beam=rays, ray_data=level, pupil_rays_drawn=drawn)
              for _name, level, drawn, paths, steps, elements, rays, files, _growth in MEASURED_RUNS]
    assert (round(min(ratios), 2), round(max(ratios), 2)) == AFTER_TRACE_MEASURED_RATIO


def _measured_beams(name: str) -> int:
    """The beams of a measured run (``MEASURED_RUNS``): the benchmark core's count, two for the two-colour reference
    designs, else one."""
    if name.startswith("core:"):
        return int(name.split(":")[1])
    return 2 if name.startswith("example:") and "dichroic" in name else 1


def test_the_ratio_stated_with_the_estimate_before_the_trace_is_the_measured_one(tmp_path):
    """``measured_ratio_low`` / ``measured_ratio_high`` of the estimate before the trace: the run files of the measured
    runs against the estimate from their beams. Its high end is the largest of ``MEASURED_RUNS``; its low end is this
    file's three-beam Project, whose beams meet no element (run here: it writes less than its estimate)."""
    from os_optics.local.example import example_project_of_kind
    from os_optics.local.catalog import load_project
    from os_optics.local.lowering import lower_project
    from os_optics.local.resources import SCENE_PUPIL_RING_BYTES_PER_BEAM

    for kind in ("focus", "polarization", "two_beam_interference", "dichroic_routing", "two_color_dichroic", "dichroic_combiner"):
        request = lower_project(load_project(example_project_of_kind(kind).payload()), None).request
        assert beam_count(request) == _measured_beams(f"example:{kind}"), kind
    ratios = [files / (_measured_beams(name) * (OTHER_FILE_BYTES_PER_BEAM - (0 if drawn else SCENE_PUPIL_RING_BYTES_PER_BEAM)
                                                + RAY_DATA_BYTES_PER_BEAM[level]))
              for name, level, drawn, _paths, _steps, _elements, _rays, files, _growth in MEASURED_RUNS]
    low, high = PRE_TRACE_MEASURED_RATIO
    assert round(max(ratios)) == high and low < round(min(ratios), 2) == 0.95
    monitor = RunMonitor()
    run_optical_project(_project(), monitor=monitor, output_dir=tmp_path / "run")
    written = sum(p.stat().st_size for p in (tmp_path / "run").iterdir())
    before = monitor.estimates[0]
    assert (before["measured_ratio_low"], before["measured_ratio_high"]) == (low, high)
    assert low <= written / before["artifact_bytes"] < 1.1 * low  # (the lowest measured: the stated low end)


def test_the_estimate_before_the_trace_is_far_below_split_beams_which_the_estimate_after_the_trace_counts():
    assert OTHER_FILE_BYTES_PER_BEAM + RAY_DATA_BYTES_PER_BEAM["full"] == ONE_BEAM_BEFORE_TRACE
    from os_optics.local.resources import AFTER_TRACE_MEASURED_RATIO, after_trace_files

    for name, level, drawn, paths, steps, elements, rays, files, _growth in MEASURED_RUNS:
        if name.startswith("tapwin") and level == "full" and drawn:
            assert files > 5 * ONE_BEAM_BEFORE_TRACE  # (the replay's observation, recorded)
            assert files >= AFTER_TRACE_MEASURED_RATIO[0] * after_trace_files(paths, steps, elements, rays_per_beam=rays)


def test_the_estimate_after_the_trace_counts_the_traced_paths_and_holds_for_the_run(tmp_path):
    from os_optics.local.example import example_project_of_kind

    design = example_project_of_kind("two_beam_interference")
    monitor = RunMonitor()
    run = run_optical_project(design, monitor=monitor, output_dir=tmp_path / "run")
    before, after = monitor.estimates
    assert (before["stage"], after["stage"]) == ("before_trace", "after_trace")
    assert after["paths"] == len(run.result.paths) and after["steps"] == sum(len(p.segments) for p in run.result.paths)
    written = sum(p.stat().st_size for p in (tmp_path / "run").iterdir())
    # (no run-file limit: the certain low end would be checked against nothing and is not counted)
    assert after["artifact_bytes_low"] is None and written <= after["artifact_bytes_high"]
    assert after["memory_bytes_low"] <= monitor.peak_memory_bytes <= after["memory_bytes_high"]
    assert monitor.summary()["estimate"] == after and monitor.summary()["estimates"] == [before, after]
    assert (after["method"], after["reference"], after["memory_platform"], after["lower_bounds"]) == (
        "COUNTED_PATHS_AND_STEPS", "MEASURED_RUNS", "CPYTHON", ["memory_bytes_low"])
    from os_optics.local.resources import AFTER_TRACE_MEASURED_RATIO

    assert (after["measured_ratio_low"], after["measured_ratio_high"]) == AFTER_TRACE_MEASURED_RATIO
    assert after["measured_ratio_low"] * after["artifact_bytes"] <= written <= after["measured_ratio_high"] * after["artifact_bytes"]
    # with a run-file limit the low end is counted and holds
    limited = RunMonitor(max_artifact_bytes=1 << 40)
    run_optical_project(design, monitor=limited, output_dir=tmp_path / "limited")
    counted = limited.estimates[-1]
    assert 0 < counted["artifact_bytes_low"] <= written <= counted["artifact_bytes_high"]
    assert counted["lower_bounds"] == ["artifact_bytes_low", "memory_bytes_low"]
    # no refusal of a run that fits: a limit at the files it writes lets it complete
    fits = RunMonitor(max_artifact_bytes=written)
    run_optical_project(design, monitor=fits, output_dir=tmp_path / "fits")
    assert verify_run_artifacts(tmp_path / "fits") == ()


def _beside(design, count: int):
    """``design`` (a payload) with ``count`` copies of its half-wave plate placed beside the beam (never met)."""

    import copy

    payload = copy.deepcopy(design)
    plate = next(c for c in payload["components"] if c["instance_id"] == "HWP")
    for i in range(count):
        spare = copy.deepcopy(plate)
        spare.update(instance_id=f"SPARE{i}", retained_dofs=[])
        spare["nominal_pose"]["position_mm"] = [-80.0 + 20.0 * (i % 10), 40.0 + 25.0 * (i // 10), 0.0]
        payload["components"].append(spare)
    return payload


#: A run-file limit no run of these tests comes near: the monitor then counts the certain low end after the trace.
NO_LIMIT_IN_REACH = 1 << 40


def test_the_low_end_after_the_trace_is_what_the_run_writes_for_certain(tmp_path, real_catalog_store, monkeypatch):
    """The low end of the range after the trace is no fitted number: the Project and Engine request files exactly, and in
    the result file each beam path and step at no fewer bytes than its shortest record. Every run writes at least that
    (the result's own ``paths`` member alone holds at least its share); a limit there refuses the run when the trace ends,
    with the reason, and nothing is written."""

    from os_optics.local.artifacts import ENGINE_REQUEST_FILE, PROJECT_FILE, SIMULATION_RESULT_FILE
    from os_optics.local.example import example_project_of_kind
    from os_optics.local.resources import shortest_record_bytes
    from os_optics.manifest.streaming import Elements, json_bytes
    from os_optics.optical_view import OpticalViewSettingsV1

    per_path, per_step = shortest_record_bytes()
    none = OpticalViewSettingsV1(pupil_ray_selection="NONE")
    designs = [(kind, example_project_of_kind(kind).payload(), level, view)
               for kind in ("focus", "polarization", "two_color_dichroic", "dichroic_routing", "two_beam_interference")
               for level, view in (("full", None), ("summary", none))]
    designs.append(("polarization + 20 plates", _beside(example_project_of_kind("polarization").payload(), 20), "summary", none))
    for name, design, level, view in designs:
        out = tmp_path / f"{name}-{level}".replace(" ", "_")
        monitor = RunMonitor(max_artifact_bytes=NO_LIMIT_IN_REACH)
        run = run_optical_project(design, monitor=monitor, output_dir=out, ray_data=level, view_settings=view)
        after = monitor.estimates[-1]
        written = sum(p.stat().st_size for p in out.iterdir())
        fixed = (out / PROJECT_FILE).stat().st_size + (out / ENGINE_REQUEST_FILE).stat().st_size
        result = json.loads((out / SIMULATION_RESULT_FILE).read_bytes())
        paths_bytes = len(json_bytes({"paths": Elements(result["paths"])})) - len(json_bytes({"paths": Elements([])}))
        assert after["paths"] == len(run.result.paths) and after["steps"] == sum(len(p.segments) for p in run.result.paths)
        assert paths_bytes >= after["paths"] * per_path + after["steps"] * per_step, name
        assert after["artifact_bytes_low"] == fixed + after["paths"] * per_path + after["steps"] * per_step <= written, name
        assert after["memory_bytes_low"] <= monitor.peak_memory_bytes
    # a run that writes no run files makes no run-file bytes certain
    bare = RunMonitor(max_artifact_bytes=NO_LIMIT_IN_REACH)
    run_optical_project(example_project_of_kind("focus").payload(), monitor=bare)
    assert bare.estimates[-1]["artifact_bytes_low"] == 0
    # the low end is counted only where a limit would be checked against it: without a run-file limit, or with one that
    # does not refuse on an estimate, it is null and the Project and the request are not serialized for it
    import os_optics.local.run as local_run

    serialized = []
    with monkeypatch.context() as patch:
        patch.setattr(local_run, "fixed_run_file_bytes", lambda *a: serialized.append(a) or 0)
        for unread in (RunMonitor(), RunMonitor(max_memory_bytes=1 << 40),
                       RunMonitor(max_artifact_bytes=NO_LIMIT_IN_REACH, preflight=False)):
            run_optical_project(example_project_of_kind("focus").payload(), monitor=unread, output_dir=tmp_path / "unread")
            assert unread.estimates[-1]["artifact_bytes_low"] is None
            assert unread.estimates[-1]["lower_bounds"] == ["memory_bytes_low"]
        assert serialized == []
        # (with a run-file limit that refuses on it, it is serialized once, when the trace ends)
        run_optical_project(example_project_of_kind("focus").payload(), monitor=RunMonitor(max_artifact_bytes=NO_LIMIT_IN_REACH),
                            output_dir=tmp_path / "read")
        assert len(serialized) == 1
    # a limit between the estimate from the beam and the certain bytes: refused when the trace ends, with the reason
    design = _beside(example_project_of_kind("polarization").payload(), 20)
    probe = RunMonitor(max_artifact_bytes=NO_LIMIT_IN_REACH)
    run_optical_project(design, monitor=probe, output_dir=tmp_path / "probe", ray_data="summary", view_settings=none)
    before, after = probe.estimates
    limit = (before["artifact_bytes"] + after["artifact_bytes_low"]) // 2
    assert before["artifact_bytes"] < limit < after["artifact_bytes_low"]
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(design, monitor=RunMonitor(max_artifact_bytes=limit), output_dir=tmp_path / "refused",
                            ray_data="summary", view_settings=none)
    stop = stopped.value.stopped.stop
    assert (stop.cause, stop.phase, stop.completed_phases) == ("ARTIFACT_LIMIT", "preflight", ("lower", "preflight", "trace"))
    assert stop.estimated_bytes == after["artifact_bytes_low"] and "at least" in stop.message and "after the trace" in stop.message
    assert "reference designs" not in stop.message and "window-and-tap" not in stop.message
    assert f"its {after['paths']} beam paths of {after['steps']} steps in all" in stop.message
    assert stop.message.endswith("a higher limit lets the run complete)")
    refused = stopped.value.stopped.progress["estimates"][-1]
    assert (refused["method"], refused["artifact_bytes_low"]) == ("COUNTED_PATHS_AND_STEPS", after["artifact_bytes_low"])
    assert refused["lower_bounds"] == ["artifact_bytes_low", "memory_bytes_low"]
    assert _tree(tmp_path / "refused") == [STOPPED_FILE]
    # preflight=False: both estimates reported, only the run's own files hold the limit
    anyway = RunMonitor(max_artifact_bytes=limit, preflight=False)
    with pytest.raises(RunStopped) as written_stop:
        run_optical_project(design, monitor=anyway, output_dir=tmp_path / "anyway", ray_data="summary", view_settings=none)
    assert written_stop.value.stopped.stop.phase == "write" and len(anyway.estimates) == 2


@pytest.mark.parametrize("case", ["polarization-summary-none", "two_color_dichroic-summary-none", "two_color_dichroic-summary",
                                  "polarization+60-full"])
def test_a_run_that_fits_a_limit_is_never_refused_by_an_estimate_after_the_trace(tmp_path, real_catalog_store, case):
    """The review's cases (the reference designs on the seed catalog): runs that write 0.66 to 0.78 x the estimate after the
    trace. With the limit at the bytes the run writes it completes; one byte less stops it while it writes (the limit
    holds), never when the trace ends."""

    from os_optics.local.example import example_project_of_kind
    from os_optics.optical_view import OpticalViewSettingsV1

    kind, level, *rest = case.split("-")
    design = example_project_of_kind(kind.split("+")[0]).payload()
    if "+" in kind:
        design = _beside(design, int(kind.split("+")[1]))
    view = OpticalViewSettingsV1(pupil_ray_selection="NONE") if rest else None
    free = RunMonitor(preflight=False)
    run_optical_project(design, monitor=free, output_dir=tmp_path / "free", ray_data=level, view_settings=view)
    written = sum(p.stat().st_size for p in (tmp_path / "free").iterdir())
    assert written < 0.8 * free.estimates[-1]["artifact_bytes"]  # (below the low end stated before the review)
    run_optical_project(design, monitor=RunMonitor(max_artifact_bytes=written), output_dir=tmp_path / "fits", ray_data=level,
                        view_settings=view)
    assert verify_run_artifacts(tmp_path / "fits") == ()
    with pytest.raises(RunStopped) as stopped:
        run_optical_project(design, monitor=RunMonitor(max_artifact_bytes=written - 1), output_dir=tmp_path / "tight",
                            ray_data=level, view_settings=view)
    assert (stopped.value.stopped.stop.cause, stopped.value.stopped.stop.phase) == ("ARTIFACT_LIMIT", "write")


def test_the_command_line_prints_the_estimate_after_the_trace(tmp_path, capsys):
    from os_optics import cli
    from os_optics.local.example import example_project_of_kind

    path = tmp_path / "design.json"
    path.write_text(json.dumps(example_project_of_kind("two_beam_interference").payload()), encoding="utf-8")
    assert cli.main(["project", "run", str(path), "--progress", "text"]) in (0, 1, 2, 3)
    err = capsys.readouterr().err
    assert "preflight done: 1 beams, run files estimated at" in err
    line = next(row for row in err.splitlines() if "after the trace" in row)
    # (no run-file limit: the certain low end is not counted, and the line states the estimate and its high end)
    assert "2 beam paths of 8 steps" in line and "at least" not in line and " MiB (up to " in line and "memory at" in line
    assert cli.main(["project", "run", str(path), "--progress", "text", "--max-artifact-mib", "4096"]) in (0, 1, 2, 3)
    line = next(row for row in capsys.readouterr().err.splitlines() if "after the trace" in row)
    assert "2 beam paths of 8 steps" in line and " MiB (at least " in line and ", up to " in line and "memory at" in line


def test_project_run_help_states_what_a_stop_keeps_of_an_earlier_run(capsys):
    from os_optics import cli

    with pytest.raises(SystemExit):
        cli.main(["project", "run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "removes the earlier run" not in text and "holds no run" not in text
    assert ("--out-dir then holds that record, run_stopped.json, beside whatever was there: an earlier run stays as it was, and "
            "'project verify' and 'results' name the later attempt (later_attempt)") in text
    assert ("A run that does not complete -- a usage error, a refused Project, a stop, a failure before the run is moved into "
            "place -- leaves the directory as it found it: the earlier run there still verifies") in text
    assert ("A failure or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves what a process killed "
            "there leaves: past the move's first step, no manifest.json, so no run that verifies, and the Engine's records "
            "of what is there (.manifest.json.earlier, the staged .manifest.json.partial).") in text
    assert ("The first step removes the run files of each configuration that did not run this time, then the stop records, "
            "then the configurations' manifests, then the directory's own. Stopped while those run files are removed (one "
            "held open elsewhere), it leaves the directory's manifest.json and that configuration's, without the run files "
            "removed before: 'project verify' then reports INCONSISTENT (exit 66), naming them as missing.") in text
    assert ("Stopped while the configurations' manifests are removed (a configuration's manifest.json held open elsewhere), "
            "it leaves the directory's manifest.json beside a configuration whose manifest is gone, recorded by its copy "
            "(configurations/<id>/.manifest.json.earlier): 'project verify' then reports INCONSISTENT (exit 66), naming that "
            "manifest. Either way any run, the same one or the earlier design, then completes there") in text
    assert "there leaves: no manifest.json, so no run that verifies" not in text
    assert "a failure, a stop -- leaves the directory as it found it" not in text
    assert "A run that completes replaces the run there" in text
    # (a name the Engine's leftover record lists is accepted: the refusal clause names that record)
    assert ("that no Engine manifest there lists (nor the Engine's record of what a completed run could not remove, "
            ".manifest.json.leftover), or a run kept under configurations/<id>") in text
    assert "that no Engine manifest there lists, or a run kept" not in text


def test_the_documents_state_what_a_failure_while_the_run_is_moved_into_place_leaves(tmp_path, capsys, monkeypatch):
    """A file of the earlier run held open while the run is moved into place: the commit fails after it removed the
    manifest, so the earlier run no longer verifies, and the copy and the staged manifest stay. A configuration's manifest
    held open (a single run written over a run of configurations) fails the move's first step instead: the directory's
    manifest stays beside a configuration whose manifest is gone, and ``project verify`` reports INCONSISTENT (66) naming
    it. The README and RESULTS_AND_EXPORT state both, and no longer promise that every failure leaves the earlier
    run verifying."""

    out = tmp_path / "run"
    run_optical_project(_other_design(), output_dir=out)
    _refuse_on(monkeypatch, [out / "ray_data.json"], rename=True, write=True)
    assert _cli_code(_complete(tmp_path, out)) == 64
    capsys.readouterr()
    monkeypatch.undo()
    assert not (out / MANIFEST_FILE).exists() and verify_run_artifacts(out) != ()
    assert (out / ".manifest.json.earlier").exists() and (out / ".manifest.json.partial").exists()
    from os_optics.local import run_configured_project

    held = tmp_path / "held"
    run_configured_project(ZOOM.read_bytes(), output_dir=held)
    _refuse_on(monkeypatch, [held / "configurations" / "mid" / MANIFEST_FILE], rename=True, write=True)
    assert _cli_code(_complete(tmp_path, held)) == 64
    capsys.readouterr()
    monkeypatch.undo()
    assert (held / MANIFEST_FILE).exists() and (held / "configurations" / "wide" / ".manifest.json.earlier").exists()
    assert verify_run_artifacts(held) == ("configurations/wide/manifest.json: missing",)
    assert _cli_code(["project", "verify", str(held)]) == 66
    assert "INCONSISTENT" in capsys.readouterr().out
    # (a configuration that did not run in the new run -- its Engine execution failed: its run files go first, before any
    # manifest; one of them held open stops the move there, with both manifests in place and no copy left)
    from os_optics.local.artifacts import write_run_files

    absent = tmp_path / "absent"
    run_configured_project(ZOOM.read_bytes(), output_dir=absent)
    _refuse_on(monkeypatch, [absent / "configurations" / "mid" / "simulation_result.json"], rename=True, write=True)
    with pytest.raises(OSError):
        write_run_files(_zoom_files(("wide", "tele")), absent, without_run=("configurations/mid",))
    monkeypatch.undo()
    assert (absent / MANIFEST_FILE).exists() and (absent / "configurations" / "mid" / MANIFEST_FILE).exists()
    assert not list(absent.rglob(".manifest.json.earlier"))
    missing = verify_run_artifacts(absent)
    assert {"configurations/mid/project.json: missing", "configurations/mid/engine_request.json: missing"} == set(missing)
    assert _cli_code(["project", "verify", str(absent)]) == 66
    assert "INCONSISTENT" in capsys.readouterr().out
    write_run_files(_zoom_files(("wide", "tele")), absent, without_run=("configurations/mid",))
    assert verify_run_artifacts(absent) == ()
    repo = Path(__file__).resolve().parents[1]

    def text(name: str) -> str:
        return " ".join((repo / name).read_text(encoding="utf-8").split())

    readme = text("docs/engine/README.md")
    assert ("a stop, `--max-artifact-mib`, a failure or Ctrl-C before the run is moved into place -- leaves the directory as "
            "it found it") in readme
    assert ("A failure or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves what a process killed "
            "there leaves: past the move's first step, no `manifest.json`, so no run that verifies") in readme
    assert "a failure, a stop, `--max-artifact-mib`, Ctrl-C -- leaves the directory as it found it" not in readme
    assert ("The first step removes the run files of each configuration that did not run this time, then the stop records, "
            "then the configurations' manifests, then the directory's own. Stopped while those run files are removed (one "
            "held open elsewhere), it leaves the directory's `manifest.json` and that configuration's, without the run files "
            "removed before: `project verify` then reports `INCONSISTENT` (exit 66) and names them as missing.") in readme
    assert ("Stopped while the configurations' manifests are removed (a configuration's `manifest.json` held open "
            "elsewhere), it leaves the directory's `manifest.json` beside a configuration whose manifest is gone, recorded by "
            "its copy (`configurations/<id>/.manifest.json.earlier`): `project verify` then reports `INCONSISTENT` (exit 66)") in readme
    results = text("docs/engine/RESULTS_AND_EXPORT.md")
    assert ("A run that completes replaces the run in its directory; an attempt that fails or stops before the run is moved "
            "into place leaves the directory as it found it; one that fails while the run is moved into place leaves the "
            "Engine's records, by which any run then completes there.") in results
    assert ("The first step removes, in this order, the run files of each configuration that did not run in the new run, the "
            "stop records, the configurations' manifests and the directory's own. Stopped while those run files are removed "
            "(one held open elsewhere, a kill, Ctrl-C), it leaves the directory's `manifest.json` and that configuration's, "
            "without the run files removed before: `project verify` reports `INCONSISTENT` (exit 66) and names them as "
            "missing.") in results
    assert ("Stopped while the configurations' manifests are removed (a configuration's `manifest.json` held open "
            "elsewhere), it leaves the directory's `manifest.json` beside a configuration whose manifest is gone") in results
    assert ("At the first step it leaves instead the directory's `manifest.json`: while the run files of a configuration that "
            "did not run in the new run are removed (one held open elsewhere), beside that configuration's `manifest.json` "
            "and without the run files removed before, which `project verify` (`INCONSISTENT`, exit 66) names as missing; "
            "while the configurations' manifests are removed (a configuration's `manifest.json` held open elsewhere), beside "
            "a configuration whose manifest is gone") in results
    assert ("A process killed during these steps past the first, a step that fails there (a file held open elsewhere) or "
            "Ctrl-C there leaves no manifest") in results
    assert ("A failure or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves what a process killed "
            "there leaves: past the move's first step, no manifest, which is no run") in results
    assert "the configurations' manifests go before the directory's own), leaves" not in results
    assert "a stop in any phase or while the files are written, Ctrl-C -- removes its own temporary files and changes nothing else" in results
    assert "An attempt that does not complete -- a usage error" not in results
    for doc in (" ".join((json_api.run_project.__doc__ or "").split()), " ".join((run_optical_project.__doc__ or "").split())):
        assert "past the move's first step, no manifest" in doc
        assert "the run files of each configuration that did not run this time, then the stop records" in doc


def test_ctrl_c_before_the_run_starts_gives_the_stop_record(tmp_path, capsys, monkeypatch):
    from os_optics import cli

    path = tmp_path / "project.json"
    path.write_text(json.dumps(_project()), encoding="utf-8")
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)  # (an earlier run in the directory)
    earlier = (out / "project.json").read_bytes()
    read = cli._read

    def interrupted(source, what):
        if what == "project":
            raise KeyboardInterrupt
        return read(source, what)

    monkeypatch.setattr(cli, "_read", interrupted)
    assert cli.main(["project", "run", str(path), "--out-dir", str(out)]) == 130
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["status"] == "CANCELLED" and record["outcome"]["execution"]["outcome"] == "CANCELLED"
    assert record["stop"]["cause"] == "CALLER" and record["stop"]["phase"] == "lower" and record["stop"]["completed_phases"] == []
    assert "Ctrl-C" in record["stop"]["message"] and "Traceback" not in captured.err
    assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", STOPPED_FILE} and (out / "project.json").read_bytes() == earlier
    assert verify_run_artifacts(out) == ()  # (the earlier run stays as it was)
    assert _later_attempt(out) == {"execution": "CANCELLED", "cause": "CALLER", "phase": "lower", "project_hash": "",
                                   "same_project": False}  # (stopped before the Project was read)


def test_ctrl_c_in_any_other_command_ends_it_with_one_line_and_exit_130(tmp_path, capsys, monkeypatch):
    from os_optics import cli
    import os_optics.local as local

    def interrupted(_run_dir):
        raise KeyboardInterrupt

    monkeypatch.setattr(local, "verify_run_artifacts", interrupted)
    assert cli.main(["project", "verify", str(tmp_path)]) == 130
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "os-optics: interrupted (Ctrl-C); the command stopped before it completed\n"


@pytest.mark.parametrize(("argv", "orig_argv", "expected"), [
    (["/venv/bin/os-optics", "project", "run"], [], True),
    ([r"C:\venv\Scripts\os-optics.exe", "project"], [], True),
    ([r"C:\venv\Scripts\os-optics-script.py"], [], True),
    (["/venv/bin/opensci-engine"], [], True),
    (["-m"], ["python", "-m", "os_optics", "project", "run"], True),
    (["-m"], ["python", "-X", "importtime", "-I", "-m", "os_optics.cli"], True),
    (["-m"], ["python", "-Im", "os_optics"], True),
    (["-m"], ["python", "-mos_optics"], True),
    (["-m"], ["python", "-m", "pytest"], False),
    (["-c"], ["python", "-c", "import os_optics"], False),
    (["my_script.py"], ["python", "my_script.py"], False),
    (["/srv/app/main.py"], ["python", "-X", "dev", "/srv/app/main.py"], False),
])
def test_the_command_is_recognised_by_its_name_or_module(argv, orig_argv, expected):
    from os_optics._command import started_as_command

    assert started_as_command(argv, orig_argv) is expected


_INTERRUPT_HOOK = """
import os, signal, sys, time

class _Interrupt:
    def find_spec(self, name, path=None, target=None):
        if name == "os_optics.validation":
            if os.environ.get("OPENSCI_TEST_SIGNAL"):  # (wait for the real SIGINT, as a terminal sends it)
                signal.signal(signal.SIGINT, signal.default_int_handler)
                open(os.environ["OPENSCI_TEST_SIGNAL"], "w").close()
                time.sleep(60)
            raise KeyboardInterrupt
        return None

sys.meta_path.insert(0, _Interrupt())
"""


def _interrupt_env(tmp_path: Path, signal_file: Path | None = None) -> dict[str, str]:
    import os

    hook = tmp_path / "hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(_INTERRUPT_HOOK, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "OPENSCI_TEST_SIGNAL")}
    env["PYTHONPATH"] = str(hook)
    if signal_file is not None:
        env["OPENSCI_TEST_SIGNAL"] = str(signal_file)
    return env


def _command_lines(tmp_path: Path, out: Path) -> dict[str, list[str]]:
    import shutil
    import sys

    design = tmp_path / "design.json"
    design.write_text(json.dumps(_project()), encoding="utf-8")
    args = ["project", "run", str(design), "--out-dir", str(out)]
    lines = {"module": [sys.executable, "-m", "os_optics", *args]}
    script = shutil.which("os-optics", path=str(Path(sys.executable).parent))
    if script is not None:
        lines["console script"] = [script, *args]
    return lines


@pytest.mark.parametrize("entry", ["module", "console script"])
def test_ctrl_c_while_the_command_is_still_importing_ends_it_without_a_traceback(tmp_path, entry):
    import subprocess

    from os_optics._command import INTERRUPTED_WHILE_STARTING

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    before = _files(out)
    line = _command_lines(tmp_path, out).get(entry)
    if line is None:
        pytest.skip("no os-optics console script beside this interpreter")
    done = subprocess.run(line, env=_interrupt_env(tmp_path), capture_output=True, text=True, timeout=300)
    assert done.returncode == 130, done.stderr
    assert done.stderr == INTERRUPTED_WHILE_STARTING + "\n" and done.stdout == ""
    assert _files(out) == before  # (no run started: nothing written or removed)
    # a program importing os_optics gets its KeyboardInterrupt unchanged
    import sys

    library = subprocess.run([sys.executable, "-c", "import os_optics"], env=_interrupt_env(tmp_path), capture_output=True,
                             text=True, timeout=300)
    assert library.returncode != 0 and "KeyboardInterrupt" in library.stderr and INTERRUPTED_WHILE_STARTING not in library.stderr


def test_a_real_ctrl_c_while_the_command_is_still_importing_ends_it_without_a_traceback(tmp_path):
    import os
    import signal
    import subprocess
    import time

    from os_optics._command import INTERRUPTED_WHILE_STARTING

    if os.name == "nt":
        pytest.skip("a console Ctrl-C cannot be sent to one process on Windows (CTRL_C_EVENT reaches the whole console); "
                    "the interrupt itself is covered by the test above")
    out = tmp_path / "run"
    ready = tmp_path / "importing"
    process = subprocess.Popen(_command_lines(tmp_path, out)["module"], env=_interrupt_env(tmp_path, ready),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 120
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), "the command did not reach its imports"
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 130, stderr
    assert stderr == INTERRUPTED_WHILE_STARTING + "\n" and stdout == "" and not out.exists()


def test_ctrl_c_while_the_command_reads_its_arguments_says_no_run_started(tmp_path, capsys, monkeypatch):
    from os_optics import cli
    from os_optics._command import INTERRUPTED_WHILE_STARTING

    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)
    before = _files(out)

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_register_command_modules", interrupted)
    assert cli.main(["project", "run", str(tmp_path / "design.json"), "--out-dir", str(out)]) == 130
    captured = capsys.readouterr()
    assert captured.err == INTERRUPTED_WHILE_STARTING + "\n" and captured.out == ""
    assert _files(out) == before  # (nothing written or removed)


def test_ctrl_c_as_project_run_begins_gives_the_stop_record(tmp_path, capsys, monkeypatch):
    from os_optics import cli

    path = tmp_path / "project.json"
    path.write_text(json.dumps(_project()), encoding="utf-8")
    out = tmp_path / "run"
    run_optical_project(_project(), output_dir=out)

    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_status_contract", interrupted)  # (the first step of project run)
    assert cli.main(["project", "run", str(path), "--out-dir", str(out)]) == 130
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "CANCELLED" and record["exit_contract"] == "requirements"
    assert record["stop"]["phase"] == "lower" and record["stop"]["completed_phases"] == [] and record["project_hash"] == ""
    assert set(_tree(out)) == RUN_OUTPUTS | {"project.json", STOPPED_FILE} and verify_run_artifacts(out) == ()


def test_the_json_api_replies_with_a_structured_refusal_for_a_directory_it_may_not_write(tmp_path):
    """run_project into a directory holding another design's project.json, or the person's own result file: a reply with
    LOCAL_RUN_ERROR and RUN_DIRECTORY_OCCUPIED naming the file, through the function and through call_method, and every
    file as it was."""
    out = tmp_path / "work"
    out.mkdir()
    (out / "project.json").write_bytes(json.dumps(_other_design()).encode())
    before = _files(out)
    reply = json_api.run_project(_project(), output_dir=str(out))
    assert reply["ok"] is False and reply["error"]["code"] == "LOCAL_RUN_ERROR"
    (problem,) = reply["error"]["problems"]
    assert problem["code"] == "RUN_DIRECTORY_OCCUPIED" and "project.json" in problem["message"]
    assert _files(out) == before
    via, _files_out = json_api.call_method("run_project", {"document": _project(), "output_dir": str(out)})
    assert via["ok"] is False and via["error"]["problems"][0]["code"] == "RUN_DIRECTORY_OCCUPIED"
    assert _files(out) == before
    (out / "project.json").unlink()
    (out / "simulation_result.json").write_bytes(b'{"mine": true}')
    before = _files(out)
    reply = json_api.run_project(_project(), output_dir=str(out))
    assert reply["error"]["problems"][0]["code"] == "RUN_DIRECTORY_OCCUPIED" and _files(out) == before
