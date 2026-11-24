"""The host side of a monitored run: progress reports, cancellation, and memory and artifact-size limits.

``RunMonitor`` is what a caller passes to ``run_optical_project(monitor=...)`` (the ``os-optics project run`` command line
and the JSON API install one themselves). The Engine tells it where the run is (``os_optics.run_control``: phases and
checkpoints); the monitor reads the clock and this process's memory, reports progress to ``on_progress`` at most every
``interval_s`` seconds and at every phase change, and stops the run at the next checkpoint (``RunStopped``) when

* ``cancel()`` was called (from another thread, a signal handler or ``on_progress`` itself): cause ``CALLER``;
* the process's memory reaches ``max_memory_bytes`` (cause ``MEMORY_LIMIT``; checked at most every
  ``memory_interval_s``): the resident set on Linux and Windows, the peak resident set elsewhere, and in a browser the
  WebAssembly memory a ``memory_probe`` of the host reports;
* the run files would pass ``max_artifact_bytes`` (cause ``ARTIFACT_LIMIT``): checked against the bytes written so far
  (``artifact_bytes``), so no run directory larger than the limit is written.

Before the trace (``preflight``) it refuses a run whose estimate from its beams (``local/resources.py``) exceeds a limit,
and when the trace has ended (``traced``) one that needs more for certain: run files at least as large as what the paths and
steps the trace counted make certain, or the memory already read; with ``preflight=False`` it reports both estimates and
lets the run go on, the limits then holding during the run only (an estimate can exceed what a run needs).

The monitor never changes a number of the run: it decides only whether the run goes on.

Each progress report is a JSON-safe dict: ``phase`` (``run_control.RUN_PHASES``), ``phases`` (the nested phases, outermost
first), ``done`` / ``total`` / ``unit`` (the phase's count; ``total`` ``None`` when not known in advance), ``elapsed_s``,
``memory_bytes`` and ``peak_memory_bytes`` (``None`` where memory cannot be read), ``completed_phases`` and ``event``
(``phase_started``, ``progress``, ``phase_ended``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .run_control import STOP_ARTIFACTS, STOP_CALLER, STOP_MEMORY, RunStopped

#: Seconds between two progress reports within one phase (a phase change is always reported).
PROGRESS_INTERVAL_S = 0.5
#: Seconds between two readings of the process's memory while a memory limit is set.
MEMORY_INTERVAL_S = 0.1
_STATM = Path("/proc/self/statm")
#: Bytes in a kibibyte: ``ru_maxrss`` counts kibibytes on Linux (bytes on macOS).
KIB = 1024


def _windows_working_set() -> Callable[[], int | None]:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    process = kernel.GetCurrentProcess()

    def read() -> int | None:
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)

    return read


def _posix_resident() -> Callable[[], int | None]:
    import resource

    page = resource.getpagesize()

    def read() -> int | None:
        try:
            return int(_STATM.read_bytes().split()[1]) * page
        except (OSError, IndexError, ValueError):
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # (no /proc: the peak resident set)
            return int(peak if sys.platform == "darwin" else peak * KIB)

    return read


def process_memory_probe() -> Callable[[], int | None]:
    """A function returning this process's memory in bytes now (``None`` where it cannot be read): the working set on
    Windows, the resident set from ``/proc`` on Linux, the peak resident set on other POSIX systems. A browser runtime has
    none (``None``): its host passes its own ``memory_probe``."""
    try:
        if sys.platform == "win32":
            return _windows_working_set()
        if sys.platform != "emscripten":
            return _posix_resident()
    except (ImportError, OSError, AttributeError):
        pass
    return lambda: None


def _statement(estimate: dict[str, Any], *, memory: bool = False) -> str:
    """How the estimate a run is refused on was made, from its fields (``local/resources.py`` ``estimate_statement``)."""
    from .local.resources import estimate_statement

    return estimate_statement(estimate, memory=memory)


class RunMonitor:
    """Progress, cancellation and resource limits of one run (module docstring)."""

    def __init__(self, on_progress: Callable[[dict[str, Any]], None] | None = None, *, max_memory_bytes: int | None = None,
                 max_artifact_bytes: int | None = None, interval_s: float = PROGRESS_INTERVAL_S,
                 memory_probe: Callable[[], int | None] | None = None, clock: Callable[[], float] = time.monotonic,
                 memory_interval_s: float = MEMORY_INTERVAL_S, files_in_memory: bool = False,
                 preflight: bool = True) -> None:
        for name, value in (("max_memory_bytes", max_memory_bytes), ("max_artifact_bytes", max_artifact_bytes)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} is a positive number of bytes, not {value!r}")
        self.on_progress = on_progress
        # (the JSON API holds run files in memory, so the memory estimate counts them too)
        self.files_in_memory = files_in_memory
        # (False: the estimate is reported, never a reason to refuse; the limits hold during the run only)
        self.refuse_on_estimate = preflight
        self.max_memory_bytes, self.max_artifact_bytes = max_memory_bytes, max_artifact_bytes
        self.interval_s, self.memory_interval_s = interval_s, memory_interval_s
        self.memory_probe = memory_probe if memory_probe is not None else process_memory_probe()
        self.clock = clock
        self.started = clock()
        self.stack: list[list[Any]] = []  # [name, done, total, unit, started]
        self.completed: list[str] = []
        self.phase_seconds: dict[str, float] = {}
        self.memory_bytes: int | None = None
        self.peak_memory_bytes: int | None = None
        self.artifact_bytes_written = 0
        self.cancel_reason: str | None = None
        self.estimate: dict[str, Any] | None = None  # the latest estimate (local/resources.py), once made
        self.estimates: list[dict[str, Any]] = []  # every estimate: before the trace, then after it
        # (set by the run: the estimate after the trace from the beam paths and steps it counted and the memory then)
        self.reestimate: Callable[[int, int, int | None], dict[str, Any]] | None = None
        self._unwound: list[list[Any]] | None = None  # the phases a run was in when an exception left them
        self._last_report = self._last_memory = float("-inf")
        self._read_memory()

    # -------------------------------------------------------------------------------------------- the caller's side
    def cancel(self, reason: str = "") -> None:
        """Stop the run at its next checkpoint (cause ``CALLER``)."""
        self.cancel_reason = reason or "the caller cancelled the run"

    @property
    def cancelled(self) -> bool:
        return self.cancel_reason is not None

    def snapshot(self, event: str = "progress") -> dict[str, Any]:
        """Where the run is now (the module docstring's progress report)."""
        top = self.stack[-1] if self.stack else None
        return {"event": event, "phase": None if top is None else top[0], "phases": [entry[0] for entry in self.stack],
                "done": None if top is None else top[1], "total": None if top is None else top[2],
                "unit": None if top is None else top[3], "elapsed_s": round(self.clock() - self.started, 3),
                "memory_bytes": self.memory_bytes, "peak_memory_bytes": self.peak_memory_bytes,
                "completed_phases": list(self.completed)}

    def summary(self) -> dict[str, Any]:
        """The whole run's progress record: seconds spent in each phase (inclusive of the phases nested in it), the elapsed
        time, the peak memory read, the run-file bytes written."""
        return {"elapsed_s": round(self.clock() - self.started, 3), "phase_seconds": {k: round(v, 3) for k, v in self.phase_seconds.items()},
                "estimate": self.estimate, "estimates": [dict(e) for e in self.estimates],
                "completed_phases": list(self.completed), "peak_memory_bytes": self.peak_memory_bytes,
                "artifact_bytes_written": self.artifact_bytes_written,
                "limits": {"max_memory_bytes": self.max_memory_bytes, "max_artifact_bytes": self.max_artifact_bytes}}

    # ------------------------------------------------------------------------------------------- the Engine's side
    def enter(self, name: str, total: int | None, unit: str) -> None:
        self.stack.append([name, 0, total, unit, self.clock()])
        self._check(force=True)
        self._report("phase_started")

    def leave(self, name: str) -> None:
        if not self.stack or self.stack[-1][0] != name:
            return
        entry = self.stack.pop()
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + (self.clock() - entry[4])
        if name not in self.completed:
            self.completed.append(name)
        if self.on_progress is not None:
            report = self.snapshot("phase_ended")
            report.update(phase=name, done=entry[1], total=entry[2], unit=entry[3],
                          phases=[e[0] for e in self.stack] + [name])
            if name == "preflight" and self.estimate is not None:
                report["estimate"] = dict(self.estimate)
            self.on_progress(report)
            self._last_report = self.clock()

    def unwind(self, name: str) -> None:
        if self._unwound is None:
            self._unwound = [list(entry) for entry in self.stack]
        if self.stack and self.stack[-1][0] == name:
            self.stack.pop()

    def step(self, count: int) -> None:
        if self.stack:
            self.stack[-1][1] += count
        self._check()
        if self.on_progress is not None and self.clock() - self._last_report >= self.interval_s:
            self._report("progress")

    def artifact_bytes(self, written: int) -> None:
        """``written`` more run-file bytes are about to be kept: stops the run (``ARTIFACT_LIMIT``) before the total passes
        the limit."""
        total = self.artifact_bytes_written + written
        if self.max_artifact_bytes is not None and total > self.max_artifact_bytes:
            self._stop(STOP_ARTIFACTS, f"the run files would exceed the artifact limit of {self.max_artifact_bytes} bytes "
                                       f"({total} bytes so far)", limit=self.max_artifact_bytes, measured=total)
        self.artifact_bytes_written = total
        self.step(written)

    def memory_now(self) -> int | None:
        """This process's memory read now (``None`` where it cannot be read)."""
        return self._read_memory()

    def preflight(self, estimate: dict[str, Any]) -> None:
        """Refuse the run when its estimated memory or run-file size (``local/resources.py``) exceeds a limit set here:
        before the trace its estimate from its beams; after the trace (``stage`` ``after_trace``) the low end of its range
        (``memory_bytes_low`` / ``artifact_bytes_low``): the memory read when the trace ended, and the run-file bytes the run
        writes for certain (``local/resources.py`` ``certain_run_file_bytes``). Unless made with
        ``preflight=False``: the estimate is then only reported, and the limits are checked during the run. The estimate
        is kept and reported whole, with the fields that state how it was made (``method``, ``reference``,
        ``measured_ratio_low`` / ``measured_ratio_high``, ``memory_platform``, ``lower_bounds``); a refusal's message is
        worded from them."""
        self.estimate = dict(estimate)
        self.estimates.append(dict(self.estimate))
        self.enter("preflight", None, "estimate")
        after = estimate.get("stage") == "after_trace"
        memory = estimate.get("memory_bytes_low" if after else "memory_bytes")
        files = estimate.get("artifact_bytes_low" if after else "artifact_bytes")
        if not self.refuse_on_estimate:
            memory = files = None
        amount = "at least" if after else "about"
        if self.max_memory_bytes is not None and isinstance(memory, int) and memory > self.max_memory_bytes:
            self._stop(STOP_MEMORY, f"the run would need {amount} {memory} bytes of memory, more than the limit of "
                                    f"{self.max_memory_bytes} bytes ({_statement(estimate, memory=True)})",
                       limit=self.max_memory_bytes, estimated=memory)
        if self.max_artifact_bytes is not None and isinstance(files, int) and files > self.max_artifact_bytes:
            self._stop(STOP_ARTIFACTS, f"the run files would be {amount} {files} bytes, more than the limit of "
                                       f"{self.max_artifact_bytes} bytes ({_statement(estimate)})",
                       limit=self.max_artifact_bytes, estimated=files)
        self.leave("preflight")

    def traced(self, paths: int, steps: int) -> None:
        """The trace has ended with ``paths`` beam paths of ``steps`` steps (segments) in all (``run_control.traced``): the
        estimate after the trace (``reestimate``, set by the run) is reported and checked as ``preflight`` does, once: a
        later trace of the same run (control readings) is not the run's."""
        reestimate, self.reestimate = self.reestimate, None
        if reestimate is not None:
            self.preflight(reestimate(paths, steps, self._read_memory()))

    # ---------------------------------------------------------------------------------------------------- internals
    def _read_memory(self) -> int | None:
        try:
            raw = self.memory_probe()
            value = None if raw is None else int(raw)
        except Exception:  # (a host probe that fails, or answers no number, reads as unknown memory, never as a stop)
            value = None
        self._last_memory = self.clock()
        if value is not None:
            self.memory_bytes = value
            self.peak_memory_bytes = max(self.peak_memory_bytes or 0, value)
        return value

    def _check(self, force: bool = False) -> None:
        if self.cancel_reason is not None:
            self._stop(STOP_CALLER, self.cancel_reason)
        if force or self.clock() - self._last_memory >= self.memory_interval_s:
            value = self._read_memory()
            if self.max_memory_bytes is not None and value is not None and value >= self.max_memory_bytes:
                self._stop(STOP_MEMORY, f"the process reached the memory limit of {self.max_memory_bytes} bytes "
                                        f"({value} bytes in use)", limit=self.max_memory_bytes, measured=value)

    def _report(self, event: str) -> None:
        if self.on_progress is not None:
            self.on_progress(self.snapshot(event))
            self._last_report = self.clock()

    def _stop(self, cause: str, message: str, *, limit: int | None = None, measured: int | None = None,
              estimated: int | None = None, stack: list[list[Any]] | None = None) -> None:
        stack = self.stack if stack is None else stack
        # (stopped between two phases: named by the phase it had finished last)
        between = f"after:{self.completed[-1]}" if self.completed else "lower"
        top = stack[-1] if stack else [between, None, None, None, self.clock()]
        raise RunStopped(cause, message, phase=top[0], phases=tuple(entry[0] for entry in stack) or (top[0],),
                         completed_phases=tuple(self.completed), done=top[1], total=top[2], unit=top[3], limit_bytes=limit,
                         measured_bytes=measured, estimated_bytes=estimated, progress=self.snapshot("stopped"))

    def stopped_by_interrupt(self) -> RunStopped:
        """The ``RunStopped`` of a run its process interrupted (Ctrl-C: ``KeyboardInterrupt``), at the phase it was in."""
        if self.cancel_reason is None:
            self.cancel_reason = "the run was interrupted (Ctrl-C)"
        try:
            self._stop(STOP_CALLER, self.cancel_reason, stack=self._unwound if self._unwound is not None else self.stack)
        except RunStopped as stopped:
            return stopped
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["MEMORY_INTERVAL_S", "PROGRESS_INTERVAL_S", "RunMonitor", "process_memory_probe"]
