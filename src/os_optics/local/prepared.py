"""Prepared evaluations: ``EvaluationSession`` objects this runtime keeps for a JSON host under opaque handles.

A host that is not a Python program cannot hold a Python object between two calls of the JSON API. ``json_api.prepare_evaluation(document)`` lowers a Project in a new
``EvaluationSession`` and returns a handle naming it; ``json_api.evaluate_prepared(handle, ...)`` evaluates through that
session -- the Project it was prepared with, or a candidate ``document`` -- so a candidate loop reuses what the session
keeps (the catalogs, the compiled elements of the latest Project, the Engine build id, the Engine's products of the
latest physical identity: ``os_optics.local.evaluate``). A reply equals the reply of ``json_api.evaluate_project`` for the
same document and selection.

The handles live in this module's state, in the memory of the process; a restarted host holds none. Their number is bounded: ``max_handles`` (default
``DEFAULT_MAX_HANDLES``, settable from 1 to ``MAX_HANDLES_LIMIT``). Preparing one more than the bound evicts the least
recently used handle that is not evaluating, and the reply names it (notice ``EVALUATION_HANDLE_EVICTED``). A handle is
used when it is prepared or evaluated. A handle this runtime does not hold -- released, evicted, never issued here --
is refused with ``EVALUATION_HANDLE_UNKNOWN`` (``context.reason``: ``RELEASED``, ``EVICTED`` or ``NOT_HELD``): prepare the
Project again. A session evaluates one thing at a time: a second evaluation of a handle while one is in progress is
refused with ``EVALUATION_HANDLE_BUSY``, and a prepare that would need to evict a handle while every handle is evaluating
with ``EVALUATION_HANDLE_LIMIT``.

Each evaluation of a handle runs under a monitor (the host's, else one of its own with no limit): ``cancel(handle)``
stops the evaluation in progress on that handle alone, at its next checkpoint (``RunStopped``, cause ``CALLER``); the
session keeps what finished before the stop, so the handle stays usable. Releasing a handle while it evaluates stops that
evaluation too. Nothing here starts a thread: the registry's lock only orders calls a threaded host makes, and the
a single-threaded host never waits on it.
"""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .._vendor.shared_contracts.optical_design import ProjectDocumentV2
from .evaluate import EvaluationSession

#: How many handles a runtime holds unless its host sets another bound (``PreparedEvaluations.set_limit``).
DEFAULT_MAX_HANDLES = 4
#: The largest bound a host may set.
MAX_HANDLES_LIMIT = 64
#: How many released or evicted handles are remembered, so a later call with one says why it is not held.
RETIRED_REMEMBERED = 256
#: Random bytes of a handle id (hex): a handle issued before the runtime started again is never taken for a new one.
HANDLE_ID_BYTES = 12

#: A handle this runtime does not hold (``context.reason``: ``RELEASED``, ``EVICTED``, ``NOT_HELD``).
EVALUATION_HANDLE_UNKNOWN = "EVALUATION_HANDLE_UNKNOWN"
#: An evaluation of the handle is in progress.
EVALUATION_HANDLE_BUSY = "EVALUATION_HANDLE_BUSY"
#: A prepare needs a handle evicted and every handle held is evaluating.
EVALUATION_HANDLE_LIMIT = "EVALUATION_HANDLE_LIMIT"
#: The notice of a handle evicted as the least recently used one (a prepare, or a lower bound, made room).
EVALUATION_HANDLE_EVICTED = "EVALUATION_HANDLE_EVICTED"
ERROR_CODES = (EVALUATION_HANDLE_UNKNOWN, EVALUATION_HANDLE_BUSY, EVALUATION_HANDLE_LIMIT)
NOTICE_CODES = (EVALUATION_HANDLE_EVICTED,)
RELEASED, EVICTED, NOT_HELD = "RELEASED", "EVICTED", "NOT_HELD"


@dataclass
class PreparedEvaluation:
    """One handle: its session, the Project its next evaluation uses when the call gives none, how many evaluations it
    made, and the monitor of the evaluation in progress (``None`` when idle)."""

    handle: str
    session: EvaluationSession
    project: ProjectDocumentV2
    evaluations: int = 0
    monitor: Any = None
    running: bool = False

    def row(self) -> dict[str, Any]:
        return {"handle": self.handle, "project_hash": self.project.content_hash(), "evaluations": self.evaluations,
                "evaluating": self.running}


def _refusal(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message,
                                   "problems": [{"code": code, "message": message, **({"context": context} if context else {})}]}}


class PreparedEvaluations:
    """The handles of one runtime, least recently used first (module docstring)."""

    def __init__(self, max_handles: int = DEFAULT_MAX_HANDLES) -> None:
        self.max_handles = max_handles
        self._lock = threading.Lock()
        self._held: OrderedDict[str, PreparedEvaluation] = OrderedDict()
        self._retired: OrderedDict[str, str] = OrderedDict()

    # ---------------------------------------------------------------------------------------------------- handles
    def room(self) -> dict[str, Any] | None:
        """``None`` when a new handle can be held (now, or by evicting an idle one); else the ``EVALUATION_HANDLE_LIMIT``
        refusal. Checked before a Project is lowered, and again when its handle is added."""

        with self._lock:
            return self._room()

    def add(self, session: EvaluationSession, project: ProjectDocumentV2) -> tuple[PreparedEvaluation | None, list[dict[str, Any]], dict[str, Any] | None]:
        """Hold ``session`` under a new handle: ``(entry, eviction notices, None)``, or ``(None, [], refusal)`` when every
        handle held is evaluating and the bound leaves no room."""

        with self._lock:
            refused = self._room()
            if refused is not None:
                return None, [], refused
            handle = "evh_" + secrets.token_hex(HANDLE_ID_BYTES)
            entry = PreparedEvaluation(handle, session, project)
            self._held[handle] = entry
            return entry, self._evict(keep=handle), None

    def acquire(self, handle: Any, monitor: Any) -> PreparedEvaluation | dict[str, Any]:
        """The handle's entry, marked evaluating under ``monitor`` and made the most recently used; or the refusal
        (``EVALUATION_HANDLE_UNKNOWN`` / ``EVALUATION_HANDLE_BUSY``)."""

        with self._lock:
            entry = self._entry(handle)
            if isinstance(entry, dict):
                return entry
            if entry.running:
                return _refusal(EVALUATION_HANDLE_BUSY, f"an evaluation of handle {handle} is in progress: wait for its reply, or stop it "
                                                        f"with cancel_evaluation({handle!r})", handle=handle)
            entry.running, entry.monitor = True, monitor
            self._held.move_to_end(handle)
            return entry

    def finish(self, entry: PreparedEvaluation, *, evaluated: bool) -> list[dict[str, Any]]:
        """The evaluation of ``entry`` ended (``evaluated``: it produced a reply of results); the handle is idle again.
        Returns the notices of handles evicted now that it is (a lower bound set while handles were evaluating): the least
        recently used idle handles beyond the bound, this one included."""

        with self._lock:
            entry.running, entry.monitor = False, None
            if evaluated:
                entry.evaluations += 1
            return self._evict(keep=None) if entry.handle in self._held else []

    def cancel(self, handle: Any) -> dict[str, Any]:
        """Stop the evaluation in progress on ``handle``: ``{"ok": True, "handle", "cancelled"}`` (``cancelled`` false when
        none is in progress), or the ``EVALUATION_HANDLE_UNKNOWN`` refusal."""

        with self._lock:
            entry = self._entry(handle)
            if isinstance(entry, dict):
                return entry
            return {"ok": True, "handle": handle, "cancelled": self._stop(entry, "the host cancelled the evaluation of this handle")}

    def release(self, handle: Any) -> dict[str, Any]:
        """Forget ``handle`` (an evaluation in progress on it is stopped): ``{"ok": True, "handle", "released": True,
        "cancelled"}``, or the ``EVALUATION_HANDLE_UNKNOWN`` refusal."""

        with self._lock:
            entry = self._entry(handle)
            if isinstance(entry, dict):
                return entry
            cancelled = self._stop(entry, "the host released this handle")
            del self._held[handle]
            self._retire(handle, RELEASED)
            return {"ok": True, "handle": handle, "released": True, "cancelled": cancelled}

    def set_limit(self, max_handles: int) -> list[dict[str, Any]]:
        """Bound the handles held to ``max_handles``; returns the notices of the idle handles evicted to meet it (handles
        evaluating are evicted when their evaluation ends, if the bound still requires it)."""

        with self._lock:
            self.max_handles = max_handles
            return self._evict(keep=None)

    def listing(self) -> list[dict[str, Any]]:
        """Every handle held, least recently used first: ``handle``, ``project_hash``, ``evaluations``, ``evaluating``."""

        with self._lock:
            return [entry.row() for entry in self._held.values()]

    def clear(self) -> None:
        """Forget every handle (an evaluation in progress keeps running to its reply) and restore the default bound."""

        with self._lock:
            for handle in self._held:
                self._retire(handle, RELEASED)
            self._held.clear()
            self.max_handles = DEFAULT_MAX_HANDLES

    # ---------------------------------------------------------------------------------------------------- internals
    def _entry(self, handle: Any) -> PreparedEvaluation | dict[str, Any]:
        if isinstance(handle, str) and handle in self._held:
            return self._held[handle]
        reason = self._retired.get(handle, NOT_HELD) if isinstance(handle, str) else NOT_HELD
        why = {RELEASED: "it was released",
               EVICTED: f"it was evicted as the least recently used handle (max_handles {self.max_handles})",
               NOT_HELD: "this runtime never issued it, or the runtime started again since"}[reason]
        return _refusal(EVALUATION_HANDLE_UNKNOWN, f"evaluation handle {handle!r} is not held: {why}. Prepare the Project again with "
                                                   f"prepare_evaluation(document) and use the handle it returns", handle=handle, reason=reason)

    def _room(self) -> dict[str, Any] | None:
        if len(self._held) < self.max_handles or any(not entry.running for entry in self._held.values()):
            return None
        return _refusal(EVALUATION_HANDLE_LIMIT, f"this runtime holds {len(self._held)} evaluation handles (max_handles {self.max_handles}) "
                                                 f"and every one is evaluating: wait for one to finish, release one with "
                                                 f"release_evaluation(handle), or raise the bound with evaluation_handles(max_handles)",
                        max_handles=self.max_handles)

    def _evict(self, keep: str | None) -> list[dict[str, Any]]:
        notices = []
        for handle in list(self._held):
            if len(self._held) <= self.max_handles:
                break
            entry = self._held[handle]
            if handle == keep or entry.running:
                continue
            del self._held[handle]
            self._retire(handle, EVICTED)
            notices.append({"code": EVALUATION_HANDLE_EVICTED, "handle": handle, "max_handles": self.max_handles,
                            "message": f"evaluation handle {handle} was evicted as the least recently used of more than "
                                       f"{self.max_handles} handles; prepare its Project again to evaluate it"})
        return notices

    def _retire(self, handle: str, reason: str) -> None:
        self._retired[handle] = reason
        self._retired.move_to_end(handle)
        while len(self._retired) > RETIRED_REMEMBERED:
            self._retired.popitem(last=False)

    @staticmethod
    def _stop(entry: PreparedEvaluation, reason: str) -> bool:
        cancel = getattr(entry.monitor, "cancel", None) if entry.running else None
        if cancel is None:
            return False
        cancel(reason)
        return True


#: The handles of this runtime (``json_api.prepare_evaluation`` and the calls that name a handle).
PREPARED_EVALUATIONS = PreparedEvaluations()

__all__ = ["DEFAULT_MAX_HANDLES", "ERROR_CODES", "EVALUATION_HANDLE_BUSY", "EVALUATION_HANDLE_EVICTED", "EVALUATION_HANDLE_LIMIT",
           "EVALUATION_HANDLE_UNKNOWN", "MAX_HANDLES_LIMIT", "NOTICE_CODES", "PREPARED_EVALUATIONS", "PreparedEvaluation",
           "PreparedEvaluations"]
