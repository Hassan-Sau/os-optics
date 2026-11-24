"""Run the Engine's real-browser tests and say which of them EXECUTED and which were SKIPPED.

The real-browser tests (``BROWSER_TESTS``: the Optical Simulation View page in Chromium -- its camera pose at several window
sizes and device pixel ratios) need ``playwright`` and a Chromium it can start. Neither is an Engine dependency, so a
plain ``pytest`` run skips them where they are absent and still exits 0; a reader of that run cannot tell whether the browser
was exercised. This runner prints one JSON line per run with the counts and the
reason of every skip, from pytest's own JUnit report (``executed`` = passed + failed; ``errors`` are tests or modules that
could not run), and with ``--require`` turns a skip into a failure
(``OPENSCI_BROWSER_TESTS=required``, which the tests read). CI's ``engine-browser`` job runs it with ``--require``.

From the repository root, with the Engine environment (``uv sync --extra test``)::

    uv pip install "playwright==1.63.0" && uv run --no-sync python -m playwright install chromium    # once
    uv run --no-sync python scripts/run_browser_tests.py --require [-- <extra pytest arguments>]

Exit status: pytest's own (non-zero on any failure or error; a run whose every test was skipped exits 0, not pytest's 5); with
``--require`` 1 when any test was skipped or none executed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
#: Every test file of the Engine that drives a real browser.
BROWSER_TESTS = ("tests/test_viewer_browser_pose.py",)
#: pytest's exit status when no test ran (here: the module skipped itself).
NO_TESTS_RAN = 5


def _playwright() -> dict:
    try:
        version = metadata.version("playwright")
    except metadata.PackageNotFoundError:
        return {"playwright": None, "chromium": None}
    chromium = None
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = Path(p.chromium.executable_path)
            chromium = str(path) if path.is_file() else f"missing: {path}"
    except Exception as exc:  # noqa: BLE001 - reported, the tests decide
        chromium = f"unavailable: {type(exc).__name__}: {exc}"
    return {"playwright": version, "chromium": chromium}


def _counts(junit: Path) -> dict:
    root = ET.parse(junit).getroot()
    cases = root.iter("testcase")
    rows = {"passed": [], "failed": [], "errors": [], "skipped": []}
    for case in cases:
        name = f"{case.get('classname')}::{case.get('name')}"
        if case.find("skipped") is not None:
            rows["skipped"].append({"test": name, "reason": (case.find("skipped").get("message") or "").strip()})
        elif case.find("failure") is not None:
            rows["failed"].append(name)
        elif case.find("error") is not None:
            rows["errors"].append(name)
        else:
            rows["passed"].append(name)
    executed = len(rows["passed"]) + len(rows["failed"])
    return {"tests": executed + len(rows["errors"]) + len(rows["skipped"]), "executed": executed, "passed": len(rows["passed"]), "failed": rows["failed"],
            "errors": rows["errors"], "skipped": rows["skipped"]}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--require", action="store_true", help="a skipped test, or no executed test, fails the run")
    args = parser.parse_args(argv)
    env = dict(os.environ)
    if args.require:
        env["OPENSCI_BROWSER_TESTS"] = "required"
    with tempfile.TemporaryDirectory(prefix="browser-tests-") as tmp:
        junit = Path(tmp) / "junit.xml"
        proc = subprocess.run([sys.executable, "-m", "pytest", *BROWSER_TESTS, "-rA", f"--junitxml={junit}", *extra], cwd=ENGINE, env=env)
        counts = _counts(junit) if junit.is_file() else {"tests": 0, "executed": 0, "passed": 0, "failed": [], "errors": [], "skipped": []}
    status = proc.returncode
    if status == NO_TESTS_RAN and counts["skipped"] and not (counts["failed"] or counts["errors"]):
        status = 0  # (everything skipped: pytest reports "no tests ran"; the counts above say so)
    if args.require and (counts["skipped"] or counts["executed"] == 0):
        status = status or 1
    report = {"browser_tests": list(BROWSER_TESTS), "required": args.require, **_playwright(), **counts, "pytest_exit": proc.returncode,
              "exit": status}
    print("BROWSER TESTS " + json.dumps(report), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
