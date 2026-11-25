"""Run one deterministic shard of a pytest suite, split by test file or by the tests the weights name (CI only).

    python ci_pytest_shard.py --shard=K/N --weights=WEIGHTS.json [--record=TIMINGS.json] -- <pytest arguments>

Every shard collects the whole suite (with the same arguments, so the same marker selection) and computes the same
partition of the selected tests' units, so the N shards together run every selected test exactly once. A unit is a
test file, never split, so its module- and session-scoped fixtures stay in one process; or, when ``WEIGHTS.json``
names it, a test function (``FILE::TEST``: all its parametrized cases) or one parametrized test (``FILE::TEST[ID]``),
which then runs apart from the rest of its file (the rest stays one unit). A test belongs to the most specific of these
the weights name. Units are balanced by their measured seconds (``WEIGHTS.json``: unit -> seconds, measured on the CI
host) with the longest-processing-time-first rule; a unit without a measurement is estimated from its number of tests,
so a new file always runs and only the balance can drift. ``--record`` merges the measured seconds of the units this
shard ran into a JSON file, for refreshing the weights (a session fixture's setup is measured in the first test that
uses it). It lives outside the package's test tree, so that the published package's tests do not change for CI.

The script runs pytest in-process (``pytest.main``), from the current directory and with the suite's own
environment's pytest, after removing its own directory from ``sys.path`` (the ``pytest`` console script would have
the environment's ``bin`` directory there instead; neither is an import root of the suite).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

SECONDS_PER_UNMEASURED_TEST = 1.0
MIN_FILE_SECONDS = 0.5
_SHARD = re.compile(r"(\d{1,3})/(\d{1,3})", re.ASCII)
_HERE = Path(__file__).resolve().parent


def parse_shard(text: str) -> tuple[int, int]:
    match = _SHARD.fullmatch(text.strip())
    if not match or not 1 <= int(match[1]) <= int(match[2]):
        raise ValueError(f"--shard must be K/N with 1 <= K <= N, not {text!r}")
    return int(match[1]), int(match[2])


def file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def unit_of(nodeid: str, weights: Mapping[str, float]) -> str:
    """The test itself or its test function when the weights name it, else its file."""
    for unit in (nodeid, nodeid.split("[", 1)[0]):
        if "::" in unit and unit in weights:
            return unit
    return file_of(nodeid)


def estimate(tests: Mapping[str, int], weights: Mapping[str, float]) -> dict[str, float]:
    return {name: max(float(weights.get(name, count * SECONDS_PER_UNMEASURED_TEST)), MIN_FILE_SECONDS)
            for name, count in tests.items()}


def partition(seconds: Mapping[str, float], total: int) -> list[list[str]]:
    """Longest processing time first; ties broken by name, so every shard computes the same partition."""
    loads = [0.0] * total
    parts: list[list[str]] = [[] for _ in range(total)]
    for name, cost in sorted(seconds.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(total), key=lambda index: (loads[index], index))
        loads[target] += cost
        parts[target].append(name)
    return parts


class ShardPlugin:
    def __init__(self, index: int, total: int, weights: Mapping[str, float], record: Path | None):
        self.index, self.total, self.weights, self.record = index, total, weights, record
        self.measured: dict[str, float] = defaultdict(float)
        self.started = 0.0

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items) -> None:
        # Runs after pytest's own -m/-k deselection hooks, so the partition covers what will run.
        tests = Counter(unit_of(item.nodeid, self.weights) for item in items)
        seconds = estimate(tests, self.weights)
        parts = partition(seconds, self.total)
        mine = set(parts[self.index - 1])
        selected = [item for item in items if unit_of(item.nodeid, self.weights) in mine]
        deselected = [item for item in items if unit_of(item.nodeid, self.weights) not in mine]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        unmeasured = sorted(name for name in mine if name not in self.weights)
        split = sum("::" in name for name in tests)
        kind = f"units ({split} of them tests run apart from their file)" if split else "test files"
        print(f"\nshard {self.index}/{self.total}: {len(mine)} of {len(tests)} {kind}, {len(selected)} of "
              f"{len(selected) + len(deselected)} tests, estimated {sum(seconds[n] for n in mine):.0f} s of "
              f"{sum(seconds.values()):.0f} s" + (f"; {len(unmeasured)} unit(s) without a measurement" if unmeasured else ""))
        self.started = time.monotonic()

    def pytest_runtest_logreport(self, report) -> None:
        self.measured[unit_of(report.nodeid, self.weights)] += report.duration

    def pytest_terminal_summary(self, terminalreporter) -> None:
        terminalreporter.write_line(f"shard {self.index}/{self.total}: measured {time.monotonic() - self.started:.0f} s "
                                    f"in tests")
        if self.record is not None:
            merged = json.loads(self.record.read_text(encoding="utf-8")) if self.record.is_file() else {}
            merged.update({name: round(value, 1) for name, value in self.measured.items()})
            self.record.write_text(json.dumps(dict(sorted(merged.items())), indent=1) + "\n", encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    argv = list(argv)
    if "--" not in argv:
        print("usage: ci_pytest_shard.py --shard=K/N --weights=FILE [--record=FILE] -- <pytest args>", file=sys.stderr)
        return 4
    split = argv.index("--")
    options = dict(arg.split("=", 1) for arg in argv[:split] if "=" in arg)
    if set(options) - {"--shard", "--weights", "--record"} or len(options) != split or "--shard" not in options:
        print(f"unsupported options: {argv[:split]}", file=sys.stderr)
        return 4
    try:
        index, total = parse_shard(options["--shard"])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 4
    weights = json.loads(Path(options["--weights"]).read_text(encoding="utf-8")) if "--weights" in options else {}
    record = Path(options["--record"]).resolve() if "--record" in options else None
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]
    return int(pytest.main(argv[split + 1:], plugins=[ShardPlugin(index, total, weights, record)]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
