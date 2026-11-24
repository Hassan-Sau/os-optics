"""Worker processes of the catalog store concurrency tests (``test_catalog_store_concurrency``).

Each worker waits on a shared barrier, then runs its store operations against one catalog directory and writes its
outcome to ``<results>/<name>.json`` (``{"ok": true, ...}`` or the error / traceback), so the test can assert on every
process. Top-level functions only: the tests start them with the ``spawn`` method on every platform.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any


def _environment(root: str, url: str | None = None, interval: str = "0") -> dict[str, str]:
    environ = {"OPENSCI_CATALOG_DIR": root, "OPENSCI_CATALOG_CHECK_INTERVAL": interval}
    if url is not None:
        environ["OPENSCI_CATALOG_URL"] = url
    return environ


def _report(results: str, name: str, outcome: dict[str, Any]) -> None:
    Path(results, f"{name}.json").write_text(json.dumps(outcome), encoding="utf-8")


def run(name: str, results: str, barrier: Any, operation: str, root: str, arguments: dict[str, Any]) -> None:
    """Run ``operation`` (``import`` / ``sync`` / ``check`` / ``pinned`` / ``hold``) after the barrier."""

    from os_optics import catalog

    outcome: dict[str, Any] = {"ok": True, "operation": operation, "steps": []}
    try:
        barrier.wait()
        for _ in range(arguments.get("repeat", 1)):
            if operation == "import":
                for path in arguments["files"]:
                    report = catalog.import_release(Path(path), activate=arguments.get("activate", True), environ=_environment(root))
                    outcome["steps"].append(report.active.catalog_version if report.active else None)
            elif operation == "sync":
                report = catalog.sync(environ=_environment(root, arguments["url"]))
                outcome["steps"].append(report.active.catalog_version)
            elif operation == "check":
                report = catalog.check(environ=_environment(root, arguments["url"]))
                outcome["steps"].append(report.reachable)
            elif operation == "pinned":
                release, _ = catalog.release_bytes(arguments["version"], environ=_environment(root, arguments["url"], "3600"))
                outcome["steps"].append(release.catalog_version)
            elif operation == "hold":  # hold the store lock until the parent says so
                with catalog.CatalogStore(root).lock():
                    Path(results, f"{name}.locked").write_text("", encoding="utf-8")
                    arguments_release = Path(results, "release")
                    while not arguments_release.exists():
                        barrier_pause()
            else:
                raise ValueError(operation)
    except catalog.CatalogError as exc:
        outcome = {"ok": False, "operation": operation, "error": exc.as_dict()}
    except Exception:  # noqa: BLE001 -- reported to the test
        outcome = {"ok": False, "operation": operation, "traceback": traceback.format_exc()}
    _report(results, name, outcome)


def barrier_pause() -> None:
    import time

    time.sleep(0.02)
