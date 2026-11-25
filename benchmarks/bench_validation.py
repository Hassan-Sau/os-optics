"""Standard local-validation benchmark (the standard reference workload: 100 components; target 2 s).

    python benchmarks/bench_validation.py [--iterations 20] [--json out.json]

Workload = tests/fixtures/requests/benchmark_standard.json: <=100 component instances, <=20 optical paths, <=500 deterministic pupil
samples, <=30 targets, single process, no GPU. Cold (first call in the process) and warm (median / P95 of the repeated calls) times
are reported separately; Pyodide start-up is measured by tests/pyodide/run_parity.mjs, never mixed into these numbers."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from os_optics import EngineConfiguration, ValidationRequest, __version__, validate

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "requests" / "benchmark_standard.json"


def percentile(sorted_values: list[float], q: float) -> float:
    k = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[k]


def time_calls(request: ValidationRequest, iterations: int) -> list[float]:
    out = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        validate(request)
        out.append(time.perf_counter() - t0)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json")
    args = parser.parse_args()
    request = ValidationRequest.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    first = validate(request)
    cold = time.perf_counter() - t0
    full = time_calls(request, args.iterations)
    physics_only = time_calls(request.model_copy(update={"configuration": EngineConfiguration(run_layout=False, run_control=False)}), args.iterations)
    layout_only = time_calls(request.model_copy(update={"configuration": EngineConfiguration(run_physics=False, run_control=False)}), args.iterations)
    project = request.project
    src = next(c.source for c in project.components if c.source is not None)
    report = {
        "engine_version": __version__,
        "python": platform.python_version(),
        "platform": sys.platform,
        "workload": {
            "components": len(project.components), "optical_paths": first.summary.path_count,
            "pupil_samples_per_path": (src.pupil_ring_count or 0) * (src.pupil_points_per_ring or 0), "targets": len(project.targets),
            "observations": len(project.observations), "result_status": first.status.value,
        },
        "cold_s": cold,
        "warm_full_s": {"median": statistics.median(full), "p95": percentile(full, 0.95), "max": full[-1], "n": len(full)},
        "warm_physics_only_s": {"median": statistics.median(physics_only), "p95": percentile(physics_only, 0.95)},
        "warm_layout_only_s": {"median": statistics.median(layout_only), "p95": percentile(layout_only, 0.95)},
        "target_p95_s": 2.0,
    }
    report["meets_p95_target"] = report["warm_full_s"]["p95"] <= report["target_p95_s"]
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["meets_p95_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
