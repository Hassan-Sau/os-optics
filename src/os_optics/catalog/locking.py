"""Mutual exclusion on one local catalog store, across threads and processes.

Several processes on one device (Agents, the command line, a local run's automatic check) may use the same store at
once. Every read-modify-write of the store index and every write of a release file happens while holding the store's
lock: an exclusive operating-system lock on ``<store>/store.lock`` (``msvcrt.locking`` on Windows, ``fcntl.flock``
elsewhere), taken with a bounded wait. The lock is re-entrant within one thread, and threads of one process queue on an
in-process lock first, so nested store operations never dead-lock themselves. The lock file is never deleted (deleting
a lock file lets two holders exist at once); it is empty and harmless.

The browser runtime (Pyodide, ``sys.platform == "emscripten"``) has no store shared with other processes: there the
lock is the in-process lock only.

Temporary files get names unique per process, thread and call (``temporary_path``), so concurrent writers never write
to the same temporary file.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .errors import CatalogError, CatalogErrorCode

msvcrt: Any = None
fcntl: Any = None
if sys.platform == "win32":
    import msvcrt
elif sys.platform != "emscripten":  # the browser runtime shares no store with other processes: the in-process lock only
    import fcntl

LOCK_FILE = "store.lock"
LOCK_FILE_MODE = 0o666
#: How long an operation waits for another process's store operation before giving up (``CATALOG_STORE_BUSY``). Store
#: operations hold the lock only for local work (never during a network request), so this is generous.
LOCK_TIMEOUT_SECONDS = 30.0
#: Pause between attempts to take the operating-system lock while another process holds it.
LOCK_POLL_SECONDS = 0.02
#: Random bytes in a temporary file name (with the process and thread id: unique across concurrent writers).
TEMPORARY_TOKEN_BYTES = 6

_HOLDS: dict[str, _Hold] = {}
_HOLDS_GUARD = threading.Lock()


class _Hold:
    """The in-process state of one store's lock: the re-entrant thread lock and the operating-system lock it holds."""

    def __init__(self) -> None:
        self.thread_lock = threading.RLock()
        self.depth = 0
        self.descriptor: int | None = None


def _key(root: Path) -> str:
    return os.path.normcase(os.path.abspath(root))


def _busy(root: Path, timeout: float) -> CatalogError:
    return CatalogError(
        CatalogErrorCode.CATALOG_STORE_BUSY,
        f"the catalog store at {root} is in use by another os-optics operation and stayed locked for {timeout:g} s; "
        "try again when it has finished",
        {"store": str(root), "timeout_s": timeout},
    )


def _system_lock(root: Path, deadline: float, timeout: float) -> int | None:
    """The descriptor of ``<root>/store.lock`` with an exclusive lock on it (None where processes share no store)."""

    if msvcrt is None and fcntl is None:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(root / LOCK_FILE, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), LOCK_FILE_MODE)
    except OSError as exc:
        raise CatalogError(CatalogErrorCode.CATALOG_STORE_ERROR, f"cannot open the catalog store lock at {root}: {exc}") from None
    while True:
        try:
            if msvcrt is not None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except OSError:
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise _busy(root, timeout) from None
            time.sleep(LOCK_POLL_SECONDS)


def _system_unlock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        if msvcrt is not None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class StoreLock:
    """``with StoreLock(root):`` -- the exclusive, re-entrant lock of the store at ``root`` (module docstring)."""

    def __init__(self, root: Path | str, timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        self.root = Path(root)
        self.timeout = timeout
        with _HOLDS_GUARD:
            self._hold = _HOLDS.setdefault(_key(self.root), _Hold())

    def __enter__(self) -> StoreLock:
        hold = self._hold
        deadline = time.monotonic() + self.timeout
        if not hold.thread_lock.acquire(timeout=self.timeout):
            raise _busy(self.root, self.timeout)
        try:
            if hold.depth == 0:
                hold.descriptor = _system_lock(self.root, deadline, self.timeout)
            hold.depth += 1
        except BaseException:
            hold.thread_lock.release()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        hold = self._hold
        try:
            hold.depth -= 1
            if hold.depth == 0:
                descriptor, hold.descriptor = hold.descriptor, None
                _system_unlock(descriptor)
        finally:
            hold.thread_lock.release()


def temporary_path(path: Path) -> Path:
    """A temporary sibling of ``path`` no concurrent writer uses: ``.<name>.<pid>.<thread>.<random>.tmp``."""

    token = os.urandom(TEMPORARY_TOKEN_BYTES).hex()
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{token}.tmp")


__all__ = ["LOCK_FILE", "LOCK_TIMEOUT_SECONDS", "StoreLock", "temporary_path"]
