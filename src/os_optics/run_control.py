"""Progress, cancellation and resource limits of a run, as the Engine sees them.

A run passes through named phases (``RUN_PHASES``): lowering the Project, the trace, the extended emitters' rays, the
observation results, the untraced-light probes, the return-light analysis, the plane bounds, the exposures, the beam
pairs, the targets, the result and its hashes, the rays of each path, the model-scope check, the local validation
record, the 3D scene, writing the run files and, for a JSON API caller, its reply. Inside
the expensive ones the Engine counts its work at checkpoints (components lowered and optical bodies and pairs of bodies
checked, beam paths traced, arrivals evaluated, probe branches followed, analysis work units, pairs analysed, paths
checked, scene paths drawn, bytes written).

This module is the Engine's side of it, and it has no clock, no I/O and no effect on any number: ``phase`` and
``checkpoint`` tell the run's monitor (a ``RunMonitor``, ``os_optics.run_monitor``, installed by ``monitored``) where the
run is; without a monitor they do nothing. The monitor -- the host's side, which reads the clock and the process's
memory -- reports progress, and stops the run by raising ``RunStopped`` from a checkpoint: when the caller cancelled it,
or when a memory or artifact-size limit the caller set is reached. A stopped run returns no result: nothing it computed
is reported, and no file of it is written to its run directory but its record (``local/run.py``).

The checkpoints are deterministic (the same run reaches the same checkpoints in the same order with the same counts, on
every runtime); only where the monitor stops depends on time and memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

#: The phases of a local run, in the order they first start (``ExecutionRecord.stop.phase``). ``preflight`` is the
#: estimate checked before the trace; the return-light analysis runs inside the plane bounds and the exposures, which
#: read it; ``reply`` is the JSON API's reply (the run's summary and documents), after the run files. ``preflight`` runs
#: again when the trace has ended: the estimate from what it counted (``traced``).
RUN_PHASES = (
    "preflight", "lower", "trace", "emitters", "observation_results", "untraced_light", "return_analysis", "bounds",
    "exposures", "interference", "targets", "result", "ray_paths", "model_scope", "record", "scene", "write", "reply",
)
#: Why a run stopped (``ExecutionRecord.stop.cause``): the caller cancelled it, or a resource limit the caller set.
STOP_CALLER = "CALLER"
STOP_MEMORY = "MEMORY_LIMIT"
STOP_ARTIFACTS = "ARTIFACT_LIMIT"
STOP_CAUSES = (STOP_CALLER, STOP_MEMORY, STOP_ARTIFACTS)


class RunStopped(Exception):
    """The run was stopped before it completed: by its caller (``cause`` ``CALLER``) or by a resource limit
    (``MEMORY_LIMIT``, ``ARTIFACT_LIMIT``). ``phase`` is the innermost phase it was in (``phases``: every phase it was
    in, outermost first), ``completed_phases`` the phases it finished, ``done`` / ``total`` / ``unit`` its count in that
    phase; for a resource limit ``limit_bytes``, the ``measured_bytes`` that reached it, or the ``estimated_bytes`` that
    exceeded it before the trace. ``progress`` is the monitor's last report (times and memory; not part of any record
    hash). ``local/run.py`` adds ``stopped`` (``StoppedRun``): the execution record of the run."""

    def __init__(self, cause: str, message: str, *, phase: str, phases: tuple[str, ...] = (),
                 completed_phases: tuple[str, ...] = (), done: int | None = None, total: int | None = None,
                 unit: str | None = None, limit_bytes: int | None = None, measured_bytes: int | None = None,
                 estimated_bytes: int | None = None, progress: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.cause, self.message, self.phase = cause, message, phase
        self.phases = phases or (phase,)
        self.completed_phases = completed_phases
        self.done, self.total, self.unit = done, total, unit
        self.limit_bytes, self.measured_bytes, self.estimated_bytes = limit_bytes, measured_bytes, estimated_bytes
        self.progress = progress or {}
        self.stopped: Any = None


class RunMonitor(Protocol):
    """What the Engine calls while a run is monitored (``os_optics.run_monitor.RunMonitor`` implements it)."""

    def enter(self, name: str, total: int | None, unit: str) -> None: ...

    def leave(self, name: str) -> None: ...

    def unwind(self, name: str) -> None: ...

    def step(self, count: int) -> None: ...


_MONITOR: ContextVar[RunMonitor | None] = ContextVar("os_optics_run_monitor", default=None)


def active_monitor() -> RunMonitor | None:
    """The monitor of the run in progress in this context, if any."""
    return _MONITOR.get()


#: The run directories (resolved) whose commit completed within the current attempt (``begin_attempt``), each with what
#: the write returns (``local/artifacts.py`` ``committed_run``).
_ATTEMPT_COMMITS: ContextVar[dict[str, Any] | None] = ContextVar("run_attempt_commits", default=None)


def begin_attempt() -> Any:
    """Open the record of the commits of one attempt to run (``end_attempt`` closes it with the token returned)."""
    return _ATTEMPT_COMMITS.set({})


def end_attempt(token: Any) -> None:
    _ATTEMPT_COMMITS.reset(token)


def attempt_commits() -> dict[str, Any] | None:
    """The current attempt's record of completed commits (``None`` outside an attempt)."""
    return _ATTEMPT_COMMITS.get()


@contextmanager
def monitored(monitor: RunMonitor | None) -> Iterator[RunMonitor | None]:
    """Run the enclosed code under ``monitor`` (``None``: unmonitored, as without this)."""
    token = _MONITOR.set(monitor)
    try:
        yield monitor
    finally:
        _MONITOR.reset(token)


@contextmanager
def phase(name: str, total: int | None = None, unit: str = "steps") -> Iterator[None]:
    """The enclosed code is the run's phase ``name`` (``RUN_PHASES``), of ``total`` checkpoints in ``unit`` when known."""
    monitor = _MONITOR.get()
    if monitor is None:
        yield
        return
    monitor.enter(name, total, unit)
    try:
        yield
    except BaseException:
        monitor.unwind(name)  # (a phase left by an exception -- a stop, an interrupt, an error -- did not complete)
        raise
    monitor.leave(name)


def checkpoint(count: int = 1) -> None:
    """``count`` more units of the current phase's work are done; the monitor may stop the run here (``RunStopped``)."""
    monitor = _MONITOR.get()
    if monitor is not None:
        monitor.step(count)


def traced(paths: int, steps: int) -> None:
    """The trace has ended with ``paths`` beam paths of ``steps`` steps (segments: one to each surface or plane a path reaches) in all: the monitor may estimate the rest of
    the run from them and stop it there (``RunMonitor.traced``, ``local/resources.py``)."""
    monitor = _MONITOR.get()
    hook = getattr(monitor, "traced", None) if monitor is not None else None
    if hook is not None:
        hook(paths, steps)


def artifact_bytes(count: int) -> None:
    """``count`` more run-file bytes are about to be kept (written, or held as a run file): the monitor may stop the run
    before they would pass its artifact limit (``RunStopped``)."""
    monitor = _MONITOR.get()
    if monitor is not None:
        keep = getattr(monitor, "artifact_bytes", None)
        if keep is not None:
            keep(count)
        else:
            monitor.step(count)


__all__ = ["RUN_PHASES", "STOP_ARTIFACTS", "STOP_CALLER", "STOP_CAUSES", "STOP_MEMORY", "RunMonitor", "RunStopped",
           "active_monitor", "artifact_bytes", "checkpoint", "monitored", "phase", "traced"]
