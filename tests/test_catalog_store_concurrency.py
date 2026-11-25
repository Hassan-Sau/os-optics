"""Concurrent use of one local catalog store by several processes and threads (``os_optics.catalog``).

Real processes (``multiprocessing``, ``spawn``) released together by a barrier run imports, synchronizations, checks
and pinned fetches against one store directory (``catalog_store_workers``). Whatever the interleaving: every verified
release any of them stored is in the index, the active release is one a process chose and its file is intact, the
same release imported at once stays one release, ``state.json`` is always a valid index, and no temporary file is left
behind. A process that holds the store lock longer than the bounded wait makes the others fail with
``CATALOG_STORE_BUSY`` (nothing changed), never wait forever.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from os_optics import catalog as catalog_api
from os_optics.catalog import CatalogError, CatalogErrorCode, CatalogStore

import catalog_store_workers
from catalog_service_stub import CatalogServiceStub
from catalog_store_helpers import FIXTURE_CATALOG

FIXTURE_BYTES = FIXTURE_CATALOG.read_bytes()
FIXTURE_VERSION = json.loads(FIXTURE_BYTES)["catalog_version"]
PROCESS_TIMEOUT_SECONDS = 180.0


def _release(version: str) -> bytes:
    document = json.loads(FIXTURE_BYTES)
    document["catalog_version"] = version
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _files(directory: Path, versions: list[str]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for version in versions:
        path = directory / f"{version}.json"
        path.write_bytes(_release(version))
        paths.append(str(path))
    return paths


def _run_processes(tmp_path: Path, root: Path, jobs: list[tuple[str, dict]]) -> list[dict]:
    """Start one process per ``(operation, arguments)``, release them together, wait, return their outcomes."""

    context = multiprocessing.get_context("spawn")
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    barrier = context.Barrier(len(jobs))
    processes = [context.Process(target=catalog_store_workers.run, args=(f"w{i}", str(results), barrier, operation, str(root), arguments))
                 for i, (operation, arguments) in enumerate(jobs)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    alive = [p for p in processes if p.is_alive()]
    for process in alive:
        process.kill()
    assert not alive, "a worker process did not finish"
    outcomes = [json.loads((results / f"w{i}.json").read_text(encoding="utf-8")) for i in range(len(jobs))]
    assert [p.exitcode for p in processes] == [0] * len(jobs)
    return outcomes


def _assert_coherent(root: Path) -> dict:
    """``state.json`` is a valid index whose every release file is intact; no temporary file is left behind."""

    document = json.loads((root / "state.json").read_text(encoding="utf-8"))
    store = CatalogStore(root)
    state = store.read_state()
    assert sorted(document["releases"]) == sorted(state.releases)
    for release in state.releases.values():
        assert store.file_intact(release), release.catalog_version
    if state.active is not None:
        assert state.active in state.releases
    assert sorted(p.name for p in root.rglob("*.tmp")) == []
    assert sorted(p.name for p in root.iterdir()) == sorted(n for n in ("releases", "state.json", "store.lock") if (root / n).exists())
    return document


@pytest.fixture
def stub():
    with CatalogServiceStub() as service:
        yield service


def test_simultaneous_imports_of_the_same_release_stay_one_release(tmp_path):
    root = tmp_path / "store"
    files = _files(tmp_path / "files", [FIXTURE_VERSION])
    outcomes = _run_processes(tmp_path, root, [("import", {"files": files, "repeat": 5})] * 6)
    assert all(o["ok"] for o in outcomes), outcomes
    assert {step for o in outcomes for step in o["steps"]} == {FIXTURE_VERSION}
    document = _assert_coherent(root)
    assert list(document["releases"]) == [FIXTURE_VERSION] and document["active"] == FIXTURE_VERSION
    assert [p.name for p in (root / "releases").iterdir()] == [f"{FIXTURE_VERSION}.json"]


def test_simultaneous_imports_of_different_releases_keep_every_release(tmp_path):
    root = tmp_path / "store"
    jobs, versions = [], []
    for worker in range(6):
        mine = [f"2026-02-0{worker + 1}.{n}" for n in (1, 9, 10)]
        versions += mine
        jobs.append(("import", {"files": _files(tmp_path / "files", mine), "activate": worker % 2 == 0, "repeat": 2}))
    outcomes = _run_processes(tmp_path, root, jobs)
    assert all(o["ok"] for o in outcomes), outcomes
    document = _assert_coherent(root)
    assert sorted(document["releases"]) == sorted(versions)
    activated = {v for job in jobs if job[1]["activate"] for v in (Path(f).stem for f in job[1]["files"])}
    assert document["active"] in activated


def test_sync_check_import_and_a_pinned_fetch_at_once(tmp_path, stub):
    current, pinned = "2026-03-01.10", "2026-03-01.9"
    stub.state.publish(_release(pinned), current=False, published_at="2026-10-01T00:00:00Z")
    stub.state.publish(_release(current), published_at="2026-10-02T00:00:00Z")
    root = tmp_path / "store"
    imported = [f"2026-03-0{n}.1" for n in (2, 3, 4, 5)]
    jobs = [("sync", {"url": stub.url, "repeat": 3}), ("sync", {"url": stub.url, "repeat": 3}),
            ("check", {"url": stub.url, "repeat": 3}),
            ("pinned", {"url": stub.url, "version": pinned}),
            ("import", {"files": _files(tmp_path / "files", imported[:2]), "activate": False, "repeat": 2}),
            ("import", {"files": _files(tmp_path / "files", imported[2:]), "activate": False, "repeat": 2})]
    outcomes = _run_processes(tmp_path, root, jobs)
    assert all(o["ok"] for o in outcomes), outcomes
    assert outcomes[0]["steps"] == [current] * 3 and outcomes[3]["steps"] == [pinned]
    document = _assert_coherent(root)
    assert sorted(document["releases"]) == sorted([current, pinned, *imported])
    assert document["active"] == current and document["server_current"]["catalog_version"] == current
    # each release was downloaded once at most per process that lacked it; never a different document
    assert {p for p in stub.state.paths() if "/releases/" in p} <= {f"/api/v1/catalog/releases/{v}" for v in (current, pinned)}


def test_threads_of_one_process_keep_every_release(tmp_path):
    root = tmp_path / "store"
    environ = {"OPENSCI_CATALOG_DIR": str(root), "OPENSCI_CATALOG_CHECK_INTERVAL": "0"}
    versions = [f"2026-04-01.{n}" for n in range(1, 13)]
    errors: list[BaseException] = []
    start = threading.Barrier(len(versions))

    def work(version: str) -> None:
        try:
            start.wait()
            catalog_api.import_release(_release(version), activate=False, environ=environ)
            catalog_api.import_release(FIXTURE_BYTES, activate=True, environ=environ)
        except BaseException as exc:  # noqa: BLE001 -- collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(v,)) for v in versions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    document = _assert_coherent(root)
    assert sorted(document["releases"]) == sorted([*versions, FIXTURE_VERSION]) and document["active"] == FIXTURE_VERSION


def test_a_store_locked_longer_than_the_bounded_wait_is_busy_and_unchanged(tmp_path):
    root = tmp_path / "store"
    catalog_api.import_release(FIXTURE_BYTES, environ={"OPENSCI_CATALOG_DIR": str(root), "OPENSCI_CATALOG_CHECK_INTERVAL": "0"})
    before = (root / "state.json").read_bytes()
    context = multiprocessing.get_context("spawn")
    results = tmp_path / "results"
    results.mkdir()
    barrier = context.Barrier(1)
    holder = context.Process(target=catalog_store_workers.run, args=("holder", str(results), barrier, "hold", str(root), {}))
    holder.start()
    try:
        deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
        while not (results / "holder.locked").exists():
            assert time.monotonic() < deadline and holder.is_alive()
            time.sleep(0.05)
        store = CatalogStore(root, lock_timeout=0.3)
        began = time.monotonic()
        with pytest.raises(CatalogError) as exc:
            store.add_release(_release("2026-05-01.1"), source="import")
        assert exc.value.code is CatalogErrorCode.CATALOG_STORE_BUSY and "try again" in exc.value.message
        assert time.monotonic() - began < 5.0
        assert (root / "state.json").read_bytes() == before and not (root / "releases" / "2026-05-01.1.json").exists()
    finally:
        (results / "release").write_text("", encoding="utf-8")
        holder.join(PROCESS_TIMEOUT_SECONDS)
    assert json.loads((results / "holder.json").read_text(encoding="utf-8"))["ok"] is True
    CatalogStore(root, lock_timeout=0.3).add_release(_release("2026-05-01.1"), source="import")  # free again
    assert "2026-05-01.1" in _assert_coherent(root)["releases"]


def test_temporary_names_are_unique_per_writer(tmp_path):
    from os_optics.catalog.locking import temporary_path

    names = {temporary_path(tmp_path / "state.json").name for _ in range(200)}
    assert len(names) == 200 and all(n.startswith(".state.json.") and n.endswith(".tmp") for n in names)
