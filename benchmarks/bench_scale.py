"""Scale benchmark: wall time, peak memory and run-file sizes of large local runs, one case per process.

    python benchmarks/bench_scale.py CASE [--ray-data full|summary] [--out-dir DIR] [--json out.json] [--phases]
        [--pupil-rays OUTER_RING|ALL|NONE] [--max-memory-mib N] [--max-artifact-mib N] [--cancel-in PHASE]

The run is monitored (``os_optics.run_monitor.RunMonitor``): the report's ``monitor`` gives the seconds of each run phase,
the peak memory the monitor read and the run-file bytes; with a limit or ``--cancel-in`` a run that stops reports its
``stopped`` record (exit 3).

Cases (inputs under benchmarks/scale/):

* ``core-N`` -- a shared objective and imaging chain, two finite observation planes and N independent same-wavelength
  852 nm sources on a 4 um pitch at f = 20 mm (``core-1.json`` and the deterministic generator below; ``core-625`` is the
  Project of the 625-beam measurements that motivated this benchmark: equal JSON, and its text with CRLF line ends has
  their recorded SHA-256 ``c2c33692...``). Every pair of array sources is independent: the run counts those pairs per
  plane and analyses none of them.
* ``m8-o1`` -- 138 optical components of eight preparation modules, one source pointing away from them and one distant
  observation plane, every trace budget 1: no light is left out of the trace, so the return-light analysis does not run.
* ``m8-line`` / ``m8-line-small`` -- the same scene with the source on the first module's beam line, so light is left out
  and the return-light analysis runs over the scene's 128 reflecting surfaces: with the default work budget
  (``max_return_analysis_work``), and with 100,000 units (``RETURN_ANALYSIS_BUDGET`` names what it withheld).

Phases: Project load and lowering, the Engine run (trace, analyses, result and ray data), the 3D scene, writing the run
directory, and ``verify_run_artifacts``. With ``--phases`` the named stages inside them are timed as well (``stages`` in the
report: seconds of each, inclusive of the stages nested in it, and how often it was called): the scene lowering, the trace,
the observation results, the untraced-light probes, the return analysis, the plane reach and exposure bounds, the
interference pair enumeration and analysis, the target evaluation, the result and trace hashes, the 3D scene, and in the
write and verify phases each file's serialization / hashing / reading. Peak resident memory is the process's (``GetProcessMemoryInfo`` on Windows,
``ru_maxrss`` elsewhere) after each phase. Times are of one cold run on the machine it runs on: a benchmark of that
machine, not a property of the Engine.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent / "scale"
SMALL_RETURN_WORK = 100_000


def peak_rss_bytes() -> int:
    """The process's peak resident set (working set on Windows)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _quaternion(direction: list[float]) -> list[float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    xx = math.cos(0) * x + math.sin(0) * y
    yy = np.cross(z, xx)
    q = Rotation.from_matrix(np.column_stack([xx, yy, z])).as_quat()
    return [float(q[3]), *map(float, q[:3])]


def core_project(n: int) -> str:
    """The N-beam core as JSON text (the issue's generator, kept bit for bit)."""
    seed = json.loads((HERE / "core-1.json").read_text(encoding="utf-8"))
    p = copy.deepcopy(seed)
    base = next(c for c in p["components"] if c["instance_id"] == "array-0000")
    p["components"] = [c for c in p["components"] if c["instance_id"] != "array-0000"]
    side = math.ceil(math.sqrt(n))
    total = min(0.003 * n, 1.0)
    for i in range(n):
        x = ((i % side) - (side - 1) / 2) * 0.004
        y = ((i // side) - (side - 1) / 2) * 0.004
        c = copy.deepcopy(base)
        c["instance_id"] = f"array-{i:04d}"
        c["source_model"]["power_w"] = total / n
        c["nominal_pose"]["quaternion_wxyz"] = _quaternion([x / 20, y / 20, -1])
        p["components"].append(c)
    p["compute_budget"] = {"max_interactions_per_path": 128, "max_paths": max(2048, 4 * n), "max_trace_steps": max(16384, 20 * n),
                           "max_interference_pairs": min(4194304, max(1024, n * (n - 1) // 2 + 100))}
    return json.dumps(p, indent=2, ensure_ascii=False)


def m8_project(case: str) -> dict[str, Any]:
    project = json.loads((HERE / "m8-o1.json").read_text(encoding="utf-8"))
    if case == "m8-o1":
        return project
    source = next(c for c in project["components"] if c["instance_id"] == "away")
    # (on the beam line of the first module, along +X: local +Z -> design +X)
    half = math.sqrt(0.5)
    source["nominal_pose"] = {"position_mm": [-1700.0, -450.0, 150.0], "quaternion_wxyz": [half, 0.0, half, 0.0]}
    project.pop("compute_budget", None)
    if case == "m8-line-small":
        project["compute_budget"] = {"max_return_analysis_work": SMALL_RETURN_WORK}
    return project


def project_of(case: str) -> Any:
    if case.startswith("core-"):
        return core_project(int(case.removeprefix("core-")))
    if case in ("m8-o1", "m8-line", "m8-line-small"):
        return m8_project(case)
    raise SystemExit(f"unknown case {case!r}")


class StageTimer:
    """Accumulates the seconds and calls of named stages by wrapping the Engine functions that run them (``--phases``).
    The wrappers only time: they call the original with its arguments and return its result unchanged."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, label: str, seconds: float) -> None:
        self.seconds[label] = self.seconds.get(label, 0.0) + seconds
        self.calls[label] = self.calls.get(label, 0) + 1

    def wrap(self, owner: Any, name: str, label: Any) -> None:
        original = getattr(owner, name)
        timer = self

        def timed(*args: Any, **kwargs: Any) -> Any:
            began = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer.add(label(*args, **kwargs) if callable(label) else label, time.perf_counter() - began)

        timed.__name__ = getattr(original, "__name__", name)
        setattr(owner, name, timed)

    def wrap_iterator(self, owner: Any, name: str, label: str) -> None:
        """Time the production of the items of an iterator-returning function (the lazily enumerated pairs)."""
        original = getattr(owner, name)
        timer = self

        def timed(*args: Any, **kwargs: Any) -> Any:
            iterator = iter(original(*args, **kwargs))
            while True:
                began = time.perf_counter()
                try:
                    item = next(iterator)
                except StopIteration:
                    timer.add(label, time.perf_counter() - began)
                    return
                timer.add(label, time.perf_counter() - began)
                yield item

        setattr(owner, name, timed)


def instrument() -> StageTimer:
    from os_optics.local import artifacts, run as local_run
    from os_optics.optics.rays import tracer
    from os_optics.validation import pipeline, resonance

    t = StageTimer()
    t.wrap(local_run, "lower_project", "Project load: lower to the Engine request")
    t.wrap(local_run, "evaluate", "Engine evaluation (all of the stages below up to the hashes)")
    t.wrap(pipeline, "lower_scene", "scene lowering")
    t.wrap(tracer.Tracer, "run", "trace")
    t.wrap(pipeline, "build_observation_results", "observation results")
    t.wrap(pipeline, "follow_untraced", "untraced-light probes")
    original_get = resonance.ReturnAnalysis.__getitem__

    def analysed(self: Any, wavelength: float) -> Any:
        fresh = wavelength in self._wavelengths and wavelength not in self._done
        began = time.perf_counter()
        try:
            return original_get(self, wavelength)
        finally:
            if fresh:
                t.add("return analysis (first read of a wavelength, incl. coating uses)", time.perf_counter() - began)

    resonance.ReturnAnalysis.__getitem__ = analysed  # type: ignore[method-assign]
    t.wrap(pipeline, "plane_reach", "plane reach and bounds (incl. the return analysis it reads)")
    t.wrap(pipeline, "build_exposure_report", "exposure report (incl. the return analysis it reads)")
    t.wrap_iterator(pipeline, "plane_pairs", "interference: pair enumeration")
    t.wrap(pipeline, "analyse_pair", "interference: pair analysis")
    t.wrap(pipeline, "same_wavelength_pair_count", "interference: pair counting")
    t.wrap(pipeline, "component_pair_count", "interference: component pair counting")
    t.wrap(pipeline, "assess_targets", "target evaluation")
    t.wrap(pipeline, "_manifest", "result manifest (result hash, trace hash)")
    t.wrap(local_run, "model_scope_issues", "model scope issues")
    t.wrap(local_run, "local_validation_record", "local validation record")
    t.wrap(local_run, "build_optical_simulation_scene", "3D scene build")
    t.wrap(local_run, "cross_artifact_problems", "cross-artifact check (run)")
    t.wrap(artifacts, "_write_pieces", lambda path, pieces: f"write: {path.name} (serialize + hash + write)")
    t.wrap(artifacts, "optical_view_files", "write: scene JSON + GLB serialization")
    t.wrap(artifacts, "_load_json", lambda read, name, problems: f"verify: read + parse {name}")
    t.wrap(artifacts, "_parse", lambda model, document, name, problems: f"verify: validate {name} as {model.__name__}")
    t.wrap(artifacts, "_sha256", "verify: file SHA-256 (all files)")
    t.wrap(artifacts, "hash_quantized_parts", "verify: result hash re-computation")
    t.wrap(artifacts, "ray_data_problems", "verify: ray data against result")
    t.wrap(artifacts, "cross_artifact_problems", "cross-artifact check (verify)")
    t.wrap(artifacts, "parse_glb", "verify: GLB structure")
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", help="core-N (e.g. core-625), m8-o1, m8-line or m8-line-small")
    parser.add_argument("--ray-data", choices=["full", "summary"], default="full")
    parser.add_argument("--out-dir", help="write the run directory here (default: no files written)")
    parser.add_argument("--json", help="also write the measurements to this file")
    parser.add_argument("--phases", action="store_true", help="also time the stages inside the phases (wraps Engine functions with timers)")
    parser.add_argument("--max-memory-mib", type=int, help="the run's memory limit (RunMonitor), MiB")
    parser.add_argument("--max-artifact-mib", type=int, help="the run's run-file limit (RunMonitor), MiB")
    parser.add_argument("--cancel-in", metavar="PHASE", help="cancel the run when it enters PHASE (run_control.RUN_PHASES)")
    parser.add_argument("--pupil-rays", choices=["OUTER_RING", "ALL", "NONE"], default="OUTER_RING", help="the 3D scene's pupil rays")
    args = parser.parse_args()

    from os_optics import __version__
    from os_optics.local import OpticalViewSettingsV1, PupilRaySelectionV1, run_optical_project
    from os_optics.local.artifacts import verify_run_artifacts
    from os_optics.run_control import RunStopped, monitored
    from os_optics.run_monitor import RunMonitor

    class Monitor(RunMonitor):
        def enter(self, name: str, total: Any, unit: str) -> None:
            if name == args.cancel_in:
                self.cancel(f"cancelled on entering {name} (bench_scale --cancel-in)")
            super().enter(name, total, unit)

    events: list[dict[str, Any]] = []
    monitor = Monitor(events.append, max_memory_bytes=None if args.max_memory_mib is None else args.max_memory_mib << 20,
                      max_artifact_bytes=None if args.max_artifact_mib is None else args.max_artifact_mib << 20)
    settings = OpticalViewSettingsV1(pupil_ray_selection=PupilRaySelectionV1(args.pupil_rays))

    stages = instrument() if args.phases else None
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    def phase(name: str, began: float) -> None:
        rows.append({"phase": name, "seconds": round(time.perf_counter() - began, 3), "peak_rss_bytes": peak_rss_bytes()})

    t = time.perf_counter()
    project = project_of(args.case)
    phase("generate", t)
    t = time.perf_counter()
    try:
        with monitored(monitor):
            run = run_optical_project(project, ray_data=args.ray_data, view_settings=settings)
            phase("run (lower, trace, analyses, scene)", t)
            if args.out_dir:
                t = time.perf_counter()
                artifacts = run.write_artifacts(args.out_dir)
                phase("write run directory", t)
    except RunStopped as stopped:
        from os_optics.local.run import stop_run

        stop_run(stopped, None, monitor, output_dir=args.out_dir)
        phase("stopped", t)
        report = {"case": args.case, "ray_data": args.ray_data, "engine_version": __version__, "python": platform.python_version(),
                  "platform": platform.platform(), "total_seconds": round(time.perf_counter() - start, 3),
                  "peak_rss_bytes": peak_rss_bytes(), "phases": rows, "stopped": stopped.stopped.model_dump(mode="json"),
                  "monitor": monitor.summary(), "events": [e for e in events if e["event"] != "progress"]}
        text = json.dumps(report, indent=1)
        print(text)
        if args.json:
            Path(args.json).write_text(text + "\n", encoding="utf-8")
        return 3
    sizes: dict[str, int] = {}
    problems = None
    if args.out_dir:
        sizes = {entry["name"]: entry["bytes"] for entry in artifacts.manifest["files"]}
        t = time.perf_counter()
        problems = list(verify_run_artifacts(args.out_dir))
        phase("verify run directory", t)
    execution = run.result.outcome.execution if run.result.outcome is not None else None
    report = {
        "case": args.case, "ray_data": args.ray_data, "engine_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "total_seconds": round(time.perf_counter() - start, 3), "peak_rss_bytes": peak_rss_bytes(),
        "phases": rows,
        "stages": None if stages is None else {k: {"seconds": round(v, 3), "calls": stages.calls[k]} for k, v in sorted(stages.seconds.items())},
        "file_bytes": sizes, "file_bytes_total": sum(sizes.values()), "verify_problems": problems,
        "paths": len(run.result.paths), "interference_results": len(run.result.interference),
        "incoherent_pairs": [p.model_dump(mode="json") for p in run.result.incoherent_pairs],
        "issue_count": len(run.result.issues),
        "execution": None if execution is None else {"outcome": execution.outcome.value, "exhausted": list(execution.budgets.exhausted)},
        "withheld": [i.numerical_context for i in run.result.issues if i.code.value == "RETURN_ANALYSIS_BUDGET"],
        "observations": [{"id": o.observation_id, "power_W": o.total_power_W, "upper_bound_W": o.power_upper_bound_W,
                          "state": None if o.quantity_state is None else o.quantity_state.value} for o in run.result.observations],
        "result_hash": None if run.result.manifest is None else run.result.manifest.result_hash,
        "monitor": monitor.summary(), "progress_reports": len(events),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
