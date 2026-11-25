"""Selective evaluation benchmark (non-gating).

    uv run --no-sync python benchmarks/bench_evaluation.py [--repeats N] [--fixtures a,b] [--candidates FILE]
        [--cold-runs N] [--analyses | --analyses-only] [--analyses-fixtures a,b] [--json FILE] [--markdown FILE]

Fixtures (``benchmarks/evaluation/``, catalog-free Project documents):

* ``double_gauss``: the six-element double Gauss objective of ``tests/fixtures/prescriptions/double_gauss_6.csv``, imported
  as one declared compound lens (inline glasses), lit on axis by a Gaussian beam (waist 3 mm, 587.5618 nm, 1 mW), with one
  image plane -- the lens project of ``tests/test_imaging.py`` / ``tests/test_ghosts.py``. It is a rebuilt double Gauss
  with the shape of the profiled evaluation (most of a complete run is the untraced-light probe traces), not the project
  file recorded there (SHA-256 ``b06dc253...9df8c2``), which is not in the repository; compare phase shares and ratios with
  the issue, not absolute times. It declares no compute budget, and with the default one its complete evaluation is
  ``COMPLETED`` with ``max_trace_steps`` (2048) listed in ``outcome.execution.budgets.exhausted_by_phase`` under
  ``untraced_light`` (an auxiliary budget leaves the calculation complete): the probe traces stop there, so its complete
  time, and every ratio against it, still follows that budget.
* ``double_gauss_field``: the same with a second source tilted 5 degrees (an off-axis field); its complete evaluation is
  ``COMPLETED`` with the same budget listed.
* ``branching``: the two-beam interferometer example (a plate beamsplitter and two mirrors, four targets) with declared
  copies of its catalog parts: branch paths and ghost probes.
* ``emitter``: an extended Lambertian emitter (an LED die) imaged by a declared biconvex N-BK7 lens onto an absorbing
  plane: secondary rays and a sampled reading.
* ``candidates.json``: two candidate loops of 11 designs on ``double_gauss``: ``pose``, the lens moved along the axis in
  0.1 mm steps (each candidate reuses the compiled elements), and ``prescription``, the first radius of the first element
  varied in 0.1 mm steps (each candidate compiles the declared elements again).

Per fixture it reports: the single-evaluation latency of ``complete`` (``EvaluationSession.evaluate``, and
``run_optical_project``) with its execution outcome (``COMPLETED``, or ``PARTIAL`` with the budget it reached) and of
``readings`` -- median, min and max over ``--repeats`` evaluations, each in a fresh session --, the phases each enters
and their seconds, the Engine executions per capability, the Python allocation peak (tracemalloc; Python allocations,
not the process's memory) of one evaluation in a session whose Engine build id is already identified -- the
identification (a hash of the package sources, once per session) is reported on its own --, the
sizes of the complete run's files and the time to write them (``LocalOpticalRun.write_artifacts`` into a new directory,
timed apart from the evaluation), and a follow-up analysis: ``readings`` then ``extend`` to ``plane_bounds`` against a
fresh complete evaluation. For each candidate set: the loop in one session (evaluations per second, executions,
invalidations and the compiled elements the session holds after each candidate) against the same loop through
``run_optical_project``, with the execution outcome of those runs. Fresh processes, timed by this process
(``--cold-runs`` times): ``python -m os_optics --version``, ``os-optics project run`` (``--out-dir``, ``--pupil-rays NONE``, ``--ray-data summary``: the issue's
command) and ``os-optics project evaluate`` (``--outputs readings``) on the first fixture (``double_gauss``).
Configurations: a Project with three configurations run in one session (``run_configured_project``) against the same
configurations lowered and run one by one (section at the end).

Section ``analyses`` (``--analyses``; ``--analyses-only`` runs it alone; ``--analyses-fixtures`` names a part of it): the
on-demand analyses of an evaluation held in memory -- ``imaging`` and ``ghosts`` on ``double_gauss`` and
``double_gauss_field``, ``field`` on a sampled Gaussian field written here (``coherent_field``) -- each on the results
of a complete evaluation with the default request and with the selections that leave work out (imaging
``convergence`` / ``diffraction`` false, ghosts ``convergence`` false, field ``refinement`` false), and the default
imaging on the results of a ``readings`` evaluation in one session: the follow-up analysis without a complete
evaluation, whose document must equal the complete one's (``run_result_hash`` aside); the outcome of the complete
evaluation held is reported with it.
``--before FILE`` puts earlier timings of the default requests beside them (``{fixture: {analysis: [seconds, ...]}}``).

It asserts inside itself, and exits 1 on a mismatch: that what a ``readings`` evaluation retains equals the complete
run's -- per plane, ``RunResults.monitor(id)``'s plane frame, every arrival in full, the sampled light, the light outside
the active area, the coherent sums, the power basis and the traced power --, on every fixture and on each candidate of
each loop against its own ``run_optical_project`` run; that ``readings`` extended to ``plane_bounds`` equals a fresh
evaluation of that scope (``result_hash``); that the extension's complete run writes the bytes ``run_optical_project``
writes (every run file); that the imaging of a ``readings`` evaluation equals the complete evaluation's; and that each
configuration run in one session writes the run files of its own run. A fresh-process command that exits with a code
other than those it is expected to (``COMMANDS``) stops the benchmark. ``tests/test_bench_evaluation.py`` runs the whole
benchmark on small fixtures (``main`` with one fixture, two-design candidate sets, the configurations and the
``coherent_field`` analyses, with one repeat) and one fresh process; the imaging comparison and the ``project run`` /
``project evaluate`` processes run only in a measurement. Timings on a shared machine spread: read min and max
beside the median.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from os_optics import __version__
from os_optics.local import EvaluationSession, run_optical_project
from os_optics.local.artifacts import run_artifact_files
from os_optics.manifest import engine_build_id
from os_optics.results import load_run
from os_optics.run_control import monitored
from os_optics.run_monitor import RunMonitor

HERE = Path(__file__).resolve().parent / "evaluation"
FIXTURES = ("double_gauss", "double_gauss_field", "branching", "emitter")
#: What ``check_retained`` compares per plane: the parts of ``RunResults.monitor(id)`` a ``readings`` evaluation computes.
#: The rest (quantity state and issues, untraced power, power upper bound, polarization mixtures, interference, detection,
#: targets) belongs to capabilities ``readings`` leaves out.
RETAINED_MONITOR_KEYS = ("plane", "arrivals", "sampled_light", "outside_active_area", "superpositions", "power_basis",
                         "total_power_W")


class MismatchError(AssertionError):
    """A retained quantity or an extension differs from the complete or fresh evaluation."""


def document(name: str) -> dict:
    return json.loads((HERE / name / "project.json").read_text(encoding="utf-8"))


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def timed(call) -> tuple[float, object, dict]:
    """Seconds, value and the run monitor's summary (phases entered with their seconds, peak memory read) of one call."""

    monitor = RunMonitor(preflight=False)
    t0 = time.perf_counter()
    with monitored(monitor):
        value = call()
    seconds = time.perf_counter() - t0
    return seconds, value, monitor.summary()


def allocation_peak(call) -> int:
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def check_retained(readings, complete, name: str) -> None:
    """``readings`` / ``complete``: the ``RunResults`` of a ``readings`` evaluation and of the complete run of the same
    design. Raises ``MismatchError`` unless every plane's retained parts (``RETAINED_MONITOR_KEYS``) are equal and every
    plane of ``readings`` states its power as not computed (or out of model scope)."""

    mine, theirs = readings.monitors(), complete.monitors()
    if [m["observation_id"] for m in mine] != [m["observation_id"] for m in theirs]:
        raise MismatchError(f"{name}: the planes of the readings differ from the complete run's")
    for row in mine:
        plane = row["observation_id"]
        a, b = readings.monitor(plane), complete.monitor(plane)
        for key in RETAINED_MONITOR_KEYS:
            if a.get(key) != b.get(key):
                raise MismatchError(f"{name}: {key} at {plane} differs from the complete run's")
        if a["quantity_state"] not in ("NOT_COMPUTED", "OUT_OF_MODEL_SCOPE"):
            raise MismatchError(f"{name}: a plane whose bounds were not computed reads {a['quantity_state']}")


def identified_session() -> EvaluationSession:
    """A new session whose Engine build id is already identified (``build_id`` is read once per session)."""

    session = EvaluationSession()
    _ = session.build_id
    return session


def write_seconds(run, repeats: int) -> list[float]:
    """Seconds of ``run.write_artifacts`` into a new directory, ``repeats`` times."""

    seconds = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="bench_evaluation_") as directory:
            t0 = time.perf_counter()
            run.write_artifacts(Path(directory) / "run")
            seconds.append(time.perf_counter() - t0)
    return seconds


def execution(evaluation) -> dict:
    """The execution outcome of an evaluation: outcome, termination and the budgets it exhausted."""

    ex = evaluation.result.outcome.execution
    exhausted = [getattr(b, "value", b) for b in ex.budgets.exhausted] if ex.budgets is not None else []
    return {"outcome": ex.outcome.value, "termination": ex.termination.value, "exhausted": exhausted}


def execution_text(state: dict) -> str:
    """``COMPLETED``, or ``PARTIAL (max_trace_steps)``: the outcome with the budgets it exhausted."""

    return state["outcome"] + (f" ({', '.join(state['exhausted'])})" if state["exhausted"] else "")


def bench_fixture(name: str, repeats: int) -> dict:
    project = document(name)
    out: dict = {"fixture": name}
    for outputs in ("complete", "readings"):
        seconds, phases, executions, last = [], {}, {}, None
        for _ in range(repeats):
            session = EvaluationSession()
            s, last, summary = timed(lambda: session.evaluate(project, outputs=(outputs,)))
            seconds.append(s)
            for phase, value in summary["phase_seconds"].items():
                phases.setdefault(phase, []).append(value)
            executions = session.statistics()["executions"]
        assert last is not None
        session = identified_session()
        out[outputs] = {"seconds": spread(seconds), "phases": {p: spread(v) for p, v in phases.items()},
                        "phases_entered": sorted(phases), "executions": executions, "execution": execution(last),
                        "allocation_peak_bytes": allocation_peak(lambda: session.evaluate(project, outputs=(outputs,)))}
    run_seconds = []
    run = None
    for _ in range(repeats):
        s, run, _summary = timed(lambda: run_optical_project(project))
        run_seconds.append(s)
    assert run is not None
    files = run_artifact_files(run)
    out["run_optical_project"] = {"seconds": spread(run_seconds)}
    out["run_files_bytes"] = {name_: len(data) for name_, data in sorted(files.items())}
    out["write_s"] = spread(write_seconds(run, repeats))
    # retained quantities and follow-up analysis
    session = EvaluationSession()
    t0 = time.perf_counter()
    readings = session.evaluate(project, outputs=("readings",))
    t1 = time.perf_counter()
    extended = readings.extend(outputs=("plane_bounds",))
    t2 = time.perf_counter()
    complete = EvaluationSession().evaluate(project, outputs=("complete",))
    t3 = time.perf_counter()
    check_retained(readings.results(), complete.results(), name)
    fresh = EvaluationSession().evaluate(project, outputs=("readings", "plane_bounds"))
    if extended.result.manifest.result_hash != fresh.result.manifest.result_hash:
        raise MismatchError(f"{name}: readings extended to plane_bounds differs from a fresh evaluation of that scope")
    whole = extended.run()
    if whole.result.manifest.result_hash != run.result.manifest.result_hash:
        raise MismatchError(f"{name}: an extension to the complete run differs from run_optical_project")
    written = run_artifact_files(whole)
    differing = sorted(n for n in files.keys() | written.keys() if files.get(n) != written.get(n))
    if differing:
        raise MismatchError(f"{name}: the run files of the extension's complete run differ from run_optical_project's: "
                            f"{', '.join(differing)}")
    out["follow_up"] = {"readings_s": t1 - t0, "extend_to_plane_bounds_s": t2 - t1, "fresh_complete_s": t3 - t2,
                        "extension_executed": list(extended.executed), "extension_reused": list(extended.reused),
                        "extension_equals_fresh": True, "run_files_equal": True,
                        "executions": session.statistics()["executions"]}
    return out


CANDIDATES = HERE / "candidates.json"


def candidate_documents(spec: dict) -> list[dict]:
    """The designs of one candidate set of ``candidates.json``: its base fixture with one component moved by each offset
    (``offsets_mm``, ``component``, ``axis``), or with one surface radius of a declared model set to each value
    (``radii_mm``, ``model``, ``surface``)."""

    base = document(spec["base"])
    candidates = []
    if "offsets_mm" in spec:
        for offset in spec["offsets_mm"]:
            candidate = copy.deepcopy(base)
            component = next(c for c in candidate["components"] if c["instance_id"] == spec["component"])
            component["nominal_pose"]["position_mm"][spec["axis"]] += offset
            candidates.append(candidate)
        return candidates
    for radius in spec["radii_mm"]:
        candidate = copy.deepcopy(base)
        model = next(m for m in candidate["optical_models"] if m["model_id"] == spec["model"])
        surface = next(s for s in model["model"]["surfaces"] if s["surface_id"] == spec["surface"])
        surface["radius_mm"] = radius
        candidates.append(candidate)
    return candidates


def bench_candidates(name: str, spec: dict, repeats: int) -> dict:
    """The candidate set ``name`` (``spec``: one entry of ``candidates.json``) evaluated with ``readings`` in one session,
    ``repeats`` times, against the same designs through ``run_optical_project``."""

    candidates = candidate_documents(spec)
    loops = []
    session = None
    evaluations: list = []
    held: list[int] = []
    for _ in range(repeats):
        session = EvaluationSession()
        evaluations, held = [], []
        t0 = time.perf_counter()
        for candidate in candidates:
            evaluations.append(session.evaluate(candidate, outputs=("readings",)))
            held.append(len(session._compiled))  # (the compiled elements the session holds; a count, read between candidates)
        loops.append(time.perf_counter() - t0)
    assert session is not None
    runs = []
    t0 = time.perf_counter()
    for candidate in candidates:
        runs.append(run_optical_project(candidate))
    through_runs = time.perf_counter() - t0
    for index, (evaluation, run) in enumerate(zip(evaluations, runs, strict=True)):
        check_retained(evaluation.results(), load_run(run), f"{name} candidate {index} of {spec['base']}")
    outcomes = sorted({execution_text(execution(run)) for run in runs})  # (one value when every run ended the same way)
    per = [loop / len(candidates) for loop in loops]
    return {"set": name, "candidates": len(candidates), "what": spec["what"],
            "session_readings_loop_s": spread(loops), "session_seconds_per_evaluation": spread(per),
            "session_evaluations_per_s": len(candidates) / statistics.median(loops),
            "run_optical_project_loop_s": through_runs, "run_optical_project_execution": outcomes,
            "run_optical_project_evaluations_per_s": len(candidates) / through_runs,
            "session_per_run_optical_project": statistics.median(per) / (through_runs / len(candidates)),
            "compiled_elements_held": held, "each_equals_its_run": True, "statistics": session.statistics()}


#: The analyses measured per fixture: (label, query, plane, request).
ANALYSES = {
    "double_gauss": (("imaging", "imaging", "image", None), ("imaging, convergence false", "imaging", "image", {"convergence": False}),
                     ("imaging, diffraction false", "imaging", "image", {"diffraction": False}),
                     ("imaging, both false", "imaging", "image", {"convergence": False, "diffraction": False}),
                     ("imaging, convergence false, launch_grid 128", "imaging", "image", {"convergence": False, "launch_grid": 128}),
                     ("ghosts", "ghosts", "image", None), ("ghosts, convergence false", "ghosts", "image", {"convergence": False})),
    "coherent_field": (("field", "field", "P", None), ("field, refinement false", "field", "P", {"refinement": False})),
}
ANALYSES["double_gauss_field"] = ANALYSES["double_gauss"]


def coherent_field_project() -> dict:
    """A sampled Gaussian field (w = 0.3 mm, 128 x 128 points of 0.02 mm, 633 nm, 1024 rays) and a plane 5 mm on in free space."""

    count, pitch, w = 128, 0.02, 0.3
    axis = [(i - 0.5 * (count - 1)) * pitch for i in range(count)]
    amplitude = [[math.sqrt(math.exp(-2.0 * (x ** 2 + y ** 2) / w ** 2)) for x in axis] for y in axis]
    source = {"instance_id": "S", "optical_role": "source", "custom_part": {"custom_part_key": "sampled", "description": "Sampled source."},
              "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
              "source_model": {"model_kind": "SAMPLED_FIELD", "wavelength_nm": 633.0, "power_w": 0.001,
                               "grid": {"column_count": count, "row_count": count, "pitch_mm": [pitch, pitch]},
                               "amplitude": amplitude, "phase_rad": [[0.0] * count for _ in range(count)], "amplitude_unit": "RELATIVE",
                               "polarization": {"kind": "LINEAR", "angle_deg": 0.0}, "ray_sampling": {"sample_count": 1024, "seed": 0}}}
    plane = {"observation_id": "P", "pose": {"position_mm": [5.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
             "aperture": {"shape": "circle", "radius_mm": 5.0}}
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": [source], "observations": [plane], "targets": []}


def _analysis(results, query: str, plane: str, request):
    return getattr(results, query)(plane, request)


def _comparable(doc) -> dict:
    dump = doc.model_dump(mode="json", by_alias=True)
    dump["identity"]["physical"].pop("run_result_hash")
    return dump


def bench_analyses(name: str, repeats: int, before: dict) -> dict:
    project = coherent_field_project() if name == "coherent_field" else document(name)
    t0 = time.perf_counter()
    complete = EvaluationSession().evaluate(project, outputs=("complete",))
    complete_s = time.perf_counter() - t0
    out: dict = {"fixture": name, "complete_s": complete_s, "complete_execution": execution(complete), "analyses": []}
    reference = {}
    for label, query, plane, request in ANALYSES[name]:
        seconds = []
        for _ in range(repeats):
            results = complete.results()
            t0 = time.perf_counter()
            doc = _analysis(results, query, plane, request)
            seconds.append(time.perf_counter() - t0)
        reference[label] = doc
        quantities = doc.quantities
        row = {"analysis": label, "request": request, "seconds": spread(seconds), "complete": doc.execution.complete,
               "quantities": len(quantities),
               "not_computed": sum(1 for q in quantities if q.state.value == "NOT_COMPUTED"),
               "convergence_checked": sum(1 for q in quantities if q.convergence is not None and q.convergence.method.value != "NONE")}
        if request is None and before.get(name, {}).get(query):
            row["before"] = spread(before[name][query])
        out["analyses"].append(row)
    if "imaging" in reference:
        session = EvaluationSession()
        t0 = time.perf_counter()
        readings = session.evaluate(project, outputs=("readings",))
        t1 = time.perf_counter()
        doc = readings.results().imaging("image")
        t2 = time.perf_counter()
        if _comparable(doc) != _comparable(reference["imaging"]):
            raise MismatchError(f"{name}: the imaging of a readings evaluation differs from the complete evaluation's")
        out["follow_up"] = {"readings_s": t1 - t0, "imaging_s": t2 - t1, "readings_plus_imaging_s": t2 - t0,
                            "complete_plus_imaging_s": complete_s + reference_seconds(out, "imaging"), "equals_complete": True}
    return out


def reference_seconds(out: dict, label: str) -> float:
    return next(row["seconds"]["median"] for row in out["analyses"] if row["analysis"] == label)


def analyses_markdown(rows: list[dict]) -> list[str]:
    fmt = lambda s: f"{s['median']:.3f} [{s['min']:.3f}..{s['max']:.3f}]"  # noqa: E731
    lines = ["Analyses on the results of a complete evaluation held in memory (s, median [min..max]); `before`: the same default "
             "request measured earlier (`--before`):", "",
             "| fixture | analysis | seconds | / default | before | quantities | not computed | convergence checked | analysis complete |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        default = {}
        for item in row["analyses"]:
            if item["request"] is None:
                default[item["analysis"].split(",")[0]] = item["seconds"]["median"]
            base = default.get(item["analysis"].split(",")[0])
            lines.append(f"| {row['fixture']} | {item['analysis']} | {fmt(item['seconds'])} | "
                         f"{item['seconds']['median'] / base:.2f} | {fmt(item['before']) if 'before' in item else ''} | "
                         f"{item['quantities']} | {item['not_computed']} | {item['convergence_checked']} | {item['complete']} |")
    lines += ["", "The complete evaluation held: "
              + "; ".join(f"{row['fixture']} {execution_text(row['complete_execution'])}" for row in rows)
              + ". A PARTIAL one stopped at the named budget: `complete + imaging s` follows that budget."]
    follow = [row for row in rows if "follow_up" in row]
    if follow:
        lines += ["", "Follow-up imaging without a complete evaluation (one session; the document equals the complete one's):", "",
                  "| fixture | readings s | imaging s | readings + imaging s | complete + imaging s |", "|---|---|---|---|---|"]
        for row in follow:
            f = row["follow_up"]
            lines.append(f"| {row['fixture']} | {f['readings_s']:.3f} | {f['imaging_s']:.3f} | {f['readings_plus_imaging_s']:.3f} | "
                         f"{f['complete_plus_imaging_s']:.3f} |")
    return lines + [""]


#: The fresh-process commands, timed by this process: (label, arguments after ``python -m os_optics``, the exit codes of a
#: finished command); ``{project}`` and ``{out}`` are filled per run. ``project run`` is the command that was profiled
#: reports; it exits 3 when the run is PARTIAL (double_gauss: its untraced-light probes reach ``max_trace_steps``). The
#: fixtures' targets pass, so 1 (also the exit of an uncaught exception) and 2 are failures here.
COMMANDS = (
    ("os-optics --version", ("--version",), (0,)),
    ("os-optics project run --out-dir D --pupil-rays NONE --ray-data summary",
     ("project", "run", "{project}", "--out-dir", "{out}", "--pupil-rays", "NONE", "--ray-data", "summary", "--progress", "none"),
     (0, 3)),
    ("os-optics project evaluate --outputs readings", ("project", "evaluate", "{project}", "--outputs", "readings", "--progress", "none"),
     (0,)),
)


def fresh_processes(name: str, runs: int, commands=COMMANDS) -> list[dict]:
    """Wall seconds of each of ``commands`` in a new Python process, ``runs`` times, on fixture ``name``. Raises
    ``RuntimeError`` when a command exits with a code it does not finish with."""

    rows = []
    for label, arguments, finished in commands:
        seconds = []
        for _ in range(runs):
            with tempfile.TemporaryDirectory(prefix="bench_evaluation_cli_") as directory:
                command = [sys.executable, "-m", "os_optics",
                           *(a.format(project=HERE / name / "project.json", out=Path(directory) / "run") for a in arguments)]
                t0 = time.perf_counter()
                reply = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
                seconds.append(time.perf_counter() - t0)
            if reply.returncode not in finished:
                raise RuntimeError(f"{' '.join(command)} exited {reply.returncode}: {reply.stderr[-2000:]}")
        rows.append({"fixture": name, "command": label, "seconds": spread(seconds)})
    return rows


def markdown(report: dict) -> str:
    lines = [f"# bench_evaluation -- os-optics {report['engine_version']}, Python {report['python']}, {report['platform']}",
             "", f"repeats per measurement: {report['repeats']} (median [min..max] s)", "",
             "| fixture | complete (evaluate) | complete execution | run_optical_project | readings | readings / complete | "
             "extend to plane_bounds | fresh complete | readings phases | complete Python alloc peak MiB | "
             "readings Python alloc peak MiB | run files bytes | write run files s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda s: f"{s['median']:.3f} [{s['min']:.3f}..{s['max']:.3f}]"  # noqa: E731
    for row in report["fixtures"]:
        ratio = row["readings"]["seconds"]["median"] / row["complete"]["seconds"]["median"]
        lines.append(f"| {row['fixture']} | {fmt(row['complete']['seconds'])} | {execution_text(row['complete']['execution'])} | "
                     f"{fmt(row['run_optical_project']['seconds'])} | "
                     f"{fmt(row['readings']['seconds'])} | {ratio:.3f} | {row['follow_up']['extend_to_plane_bounds_s']:.3f} | "
                     f"{row['follow_up']['fresh_complete_s']:.3f} | {', '.join(row['readings']['phases_entered'])} | "
                     f"{row['complete']['allocation_peak_bytes'] / 2**20:.1f} | {row['readings']['allocation_peak_bytes'] / 2**20:.1f} | "
                     f"{sum(row['run_files_bytes'].values())} | {fmt(row['write_s'])} |")
    lines += ["", "A PARTIAL complete evaluation stopped at the named budget: its time, and every ratio against it, follows "
              "that budget.",
              "Allocation peaks are of one evaluation in a session whose build id is already identified; identifying it "
              f"(`engine_build_id()`, once per session): {report['engine_build_id_allocation_peak_bytes'] / 2**20:.1f} MiB.",
              "Every row: the readings equal the complete run's (per plane: frame, arrivals, sampled light, coherent sums, "
              "traced power), readings extended to plane_bounds equals a fresh evaluation, and the extension's complete run "
              "writes the run files run_optical_project writes."]
    lines += ["", "Candidate loops (readings in one session against run_optical_project per design; each candidate's "
              "readings equal its own run_optical_project run; compiled elements held: by the session after the first "
              "candidate and after the last):", "",
              "| set | designs | one session, loop s | evaluations / s | run_optical_project, loop s | "
              "run_optical_project execution | evaluations / s | "
              "session / run_optical_project per design | compiled elements held |", "|---|---|---|---|---|---|---|---|---|"]
    for c in report["candidates"].values():
        lines.append(f"| {c['set']}: {c['what']} | {c['candidates']} | {fmt(c['session_readings_loop_s'])} | "
                     f"{c['session_evaluations_per_s']:.1f} | {c['run_optical_project_loop_s']:.3f} | "
                     f"{', '.join(c['run_optical_project_execution'])} | "
                     f"{c['run_optical_project_evaluations_per_s']:.2f} | {c['session_per_run_optical_project']:.3f} | "
                     f"{c['compiled_elements_held'][0]} / {c['compiled_elements_held'][-1]} |")
    lines.append("")
    for c in report["candidates"].values():
        lines.append(f"Session statistics after the {c['set']} loop: `{json.dumps(c['statistics'], sort_keys=True)}`")
    lines.append("")
    lines += ["Fresh processes (wall seconds measured by the parent: interpreter start, imports and the command):", "",
              "| fixture | command | seconds |", "|---|---|---|"]
    for row in report["fresh_processes"]:
        lines.append(f"| {row['fixture']} | `{row['command']}` | {fmt(row['seconds'])} |")
    lines += ["", "Phases of a complete evaluation (median s):", ""]
    for row in report["fixtures"]:
        phases = row["complete"]["phases"]
        lines.append(f"- {row['fixture']}: " + ", ".join(f"{p} {v['median']:.3f}" for p, v in sorted(phases.items(), key=lambda kv: -kv[1]["median"])))
    lines += ["", "Executions per capability, one complete evaluation: "
              + "; ".join(f"{row['fixture']}: {json.dumps(row['complete']['executions'], sort_keys=True)}" for row in report["fixtures"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fixtures", default=",".join(FIXTURES))
    parser.add_argument("--candidates", default=str(CANDIDATES), help="the candidate sets (default: benchmarks/evaluation/candidates.json)")
    parser.add_argument("--cold-runs", type=int, default=3, help="runs of each fresh-process command, on the first fixture")
    parser.add_argument("--json")
    parser.add_argument("--markdown")
    parser.add_argument("--analyses", action="store_true", help="also run the analyses section")
    parser.add_argument("--analyses-only", action="store_true", help="run the analyses section alone")
    parser.add_argument("--analyses-fixtures", default=",".join(ANALYSES), help="the fixtures of the analyses section")
    parser.add_argument("--before", help="earlier timings of the default analyses: {fixture: {analysis: [seconds, ...]}}")
    args = parser.parse_args(argv)
    fixtures = args.fixtures.split(",")
    report = {"engine_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
              "processor": platform.processor(), "repeats": args.repeats, "fixtures": [], "fresh_processes": []}
    before = json.loads(Path(args.before).read_text(encoding="utf-8")) if args.before else {}
    try:
        if not args.analyses_only:
            report["engine_build_id_allocation_peak_bytes"] = allocation_peak(engine_build_id)
            for name in fixtures:
                report["fixtures"].append(bench_fixture(name, args.repeats))
                print(f"{name}: done", file=sys.stderr, flush=True)
            sets = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
            report["candidates"] = {name: bench_candidates(name, spec, args.repeats) for name, spec in sets.items()}
            report["fresh_processes"] = fresh_processes(fixtures[0], args.cold_runs) if args.cold_runs > 0 else []
            report["configurations"] = bench_configurations(args.repeats)
        if args.analyses or args.analyses_only:
            report["analyses"] = []
            for name in args.analyses_fixtures.split(","):
                report["analyses"].append(bench_analyses(name, args.repeats, before))
                print(f"analyses {name}: done", file=sys.stderr, flush=True)
    except MismatchError as exc:
        print(f"MISMATCH: {exc}", file=sys.stderr)
        return 1
    if args.analyses_only:
        text = "\n".join([f"# bench_evaluation analyses -- os-optics {report['engine_version']}, Python {report['python']}, {report['platform']}", "",
                          f"repeats per measurement: {report['repeats']}", ""]) + "\n"
    else:
        text = markdown(report) + configurations_markdown(report["configurations"])
    if "analyses" in report:
        text += "\n" + "\n".join(analyses_markdown(report["analyses"])) + "\n"
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------------------------------------------ configurations
# A Project with configurations (tests/fixtures/configurations/zoom_35_70_105.json: a three-group zoom of ideal thin lenses,
# configurations wide / mid / tele). ``run_configured_project`` runs them through one evaluation session -- one lowering
# per configuration, before anything is traced, each element compiled once, the Engine build identified once -- against
# the same work done configuration by configuration: every configuration lowered (``lower_project``), then each run with
# ``run_optical_project`` (its own session: lowered again, compiled again, the build identified again). It asserts that
# each configuration's run files are byte-identical either way.

CONFIGURED = HERE.parent.parent / "tests" / "fixtures" / "configurations" / "zoom_35_70_105.json"


def bench_configurations(repeats: int) -> dict:
    from os_optics.local import ProjectDocumentV2, lower_project, run_configured_project
    from os_optics.local.configurations import configurations_run_files

    data = CONFIGURED.read_bytes()
    project = ProjectDocumentV2.model_validate_json(data)
    ids = [c.configuration_id for c in project.configurations]

    def separately() -> dict:
        for cid in ids:  # (every configuration lowered before any is traced)
            lower_project(project.configured(cid))
        return {cid: run_artifact_files(run_optical_project(project.configured(cid))) for cid in ids}

    shared, apart = [], []
    files = alone = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        files = configurations_run_files(run_configured_project(data))
        shared.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        alone = separately()
        apart.append(time.perf_counter() - t0)
    assert files is not None and alone is not None
    for cid in ids:
        prefix = f"configurations/{cid}/"
        mine = {name[len(prefix):]: value for name, value in files.items() if name.startswith(prefix)}
        if mine != alone[cid]:
            raise MismatchError(f"configuration {cid}: the run of configurations wrote other files than its own run")
    return {"fixture": CONFIGURED.name, "configurations": ids, "run_configured_project_s": spread(shared),
            "lowered_then_run_separately_s": spread(apart), "run_files_bytes": sum(len(v) for v in files.values()),
            "files_equal": True}


def configurations_markdown(row: dict) -> str:
    fmt = lambda s: f"{s['median']:.3f} [{s['min']:.3f}..{s['max']:.3f}]"  # noqa: E731
    return "\n".join([
        "", f"Configurations ({row['fixture']}: {', '.join(row['configurations'])}; run files byte-identical either way):", "",
        "| run | seconds (with run files) |", "|---|---|",
        f"| run_configured_project (one session) | {fmt(row['run_configured_project_s'])} |",
        f"| each lowered, then run_optical_project each | {fmt(row['lowered_then_run_separately_s'])} |", ""])


if __name__ == "__main__":
    raise SystemExit(main())
