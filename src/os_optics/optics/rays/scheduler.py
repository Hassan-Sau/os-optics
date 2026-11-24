"""Trace scheduling and the run's trace budget (``max_paths``, ``max_trace_steps``, ``max_interactions_per_path``).

``TraceScheduler`` decides *what* the tracer traces next and *whether* the budget still allows it; the tracer
(``optics/rays/tracer.py``) decides *how* light travels and books every power flow. One scheduler serves one ``Tracer``
(a run or a probe pass): it owns the queue of packets waiting to be traced (strongest first), the counts the budget
limits (paths admitted, chief-ray steps taken), what the budget left untraced (source lines not launched, branches not
traced, paths cut short), which limits were reached and the path / source ids they affected.

Every decision is a query plus, where the budget refuses, the record of that refusal; the tracer then books the refused
power (the ledger's ``PATH_LIMIT``) and keeps every geometric side effect (a path cut short ends where its chief ray's last
event was). The decisions are:

* ``admit_root`` / ``refuse_line``: launching a source line (strongest lines first, ``Tracer.run``);
* ``reserve_branch``: a splitting interaction's traced branch (a reflection the trace follows);
* ``admit_probe``: one untraced branch followed by a probe pass (``Tracer.trace_probes``);
* ``push`` / ``pop``: the queue, strongest chief power first, powers within the run's ordering resolution
  (``order_power``) in queue order -- ``pop`` refuses a packet once the step budget is spent;
* ``take_step``: one chief-ray step of the path being traced (refused: the path is cut short);
* ``keep_path``: whether a finished path is kept in the result;
* ``interaction_spent``: the per-path interaction limit at a physical surface.

``report`` turns the run's refusals into its one ``MAX_PATHS`` issue (``validation/outcome.py`` reads it into
``outcome.execution.budgets``), ``interaction_issue`` a path ended at the interaction limit into its ``MAX_INTERACTIONS``
issue. ``exhausted`` / ``affected_path_ids`` / ``affected_source_ids`` are what a probe pass reports
(``validation/untraced.py``).
"""

from __future__ import annotations

import heapq
import math
from typing import Generic, TypeVar

from ...errors import ErrorCode, Status, ValidationIssue, make_issue

T = TypeVar("T")

#: The budget names (``NumericalConfig`` fields) as issues and ``outcome.execution.budgets`` name them.
MAX_PATHS = "max_paths"
MAX_TRACE_STEPS = "max_trace_steps"
MAX_INTERACTIONS_PER_PATH = "max_interactions_per_path"
#: ``PowerFlow.detail`` of the power the budget left untraced (the ledger's ``PATH_LIMIT``) where no interaction names it.
LINE_NOT_TRACED = "source line not traced (max_paths)"
BRANCH_NOT_TRACED = "branch not traced (max_trace_steps)"
PERCENT = 100.0


def order_quantum(cfg_resolution: float, launched_W: float) -> float:
    """The resolution (W) at which a run orders beams by power, one grid for every beam of the run: the run's launched power
    times the largest power of two at most ``cfg_resolution`` (``power_ledger_fraction_resolution``; 2^-40 for 1e-12). A
    binary fraction of the launched power keeps the grid's midpoints off the powers a cascade of equal splits produces
    (the launched power times 2^-n is a whole number of steps up to n = 40, where a decimal fraction puts 2^-13 of it on a
    midpoint). 0: the run launches no known power; beams are then ordered by their exact power."""

    if not (math.isfinite(launched_W) and launched_W > 0.0):
        return 0.0
    return math.ldexp(launched_W, math.frexp(cfg_resolution)[1] - 1)


def order_power(power_W: float, quantum_W: float) -> float:
    """The power by which a beam is ordered: ``power_W`` rounded to the run's grid ``quantum_W`` (``order_quantum``). Two
    beams whose powers are equal in exact arithmetic (the halves of a 50:50 split, symmetric ghosts) carry powers a few units
    in the last place apart that follow the rounding of the poses; on the grid they read the same, and the tie goes to queue
    order in every run alike. Only the order reads it: every booked power is the beam's own."""

    if quantum_W > 0.0 and math.isfinite(power_W):
        return float(round(power_W / quantum_W))
    return power_W


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


class TraceScheduler(Generic[T]):
    """The queue and the trace budget of one ``Tracer`` (module docstring). ``T``: the queued packet type."""

    def __init__(self, max_paths: int, max_trace_steps: int, max_interactions_per_path: int, order_quantum_W: float = 0.0) -> None:
        self.max_paths = max_paths
        self.max_trace_steps = max_trace_steps
        self.max_interactions_per_path = max_interactions_per_path
        # the run's power ordering resolution (``order_quantum``)
        self.order_quantum_W = order_quantum_W
        # packets waiting to be traced, strongest first: (-chief power on the ordering grid, queue order, path id, source id,
        # packet)
        self._queue: list[tuple[float, int, str, str, T]] = []
        self._queued = 0
        self.paths = 0  # paths admitted: source lines launched, branches spawned, probe roots followed
        self.steps = 0  # chief-ray steps taken
        self.untraced = 0  # source lines and branches not traced
        self.untraced_lines = 0  # of which source lines never launched
        self.cut_short = 0  # paths stopped where the step budget ran out
        self._limits: set[str] = set()
        self._path_ids: set[str] = set()
        self._source_ids: set[str] = set()

    # ------------------------------------------------------------------------------------------------- what was reached
    @property
    def exhausted(self) -> tuple[str, ...]:
        """The limits the budget reached (sorted names)."""
        return tuple(sorted(self._limits))

    @property
    def affected_path_ids(self) -> tuple[str, ...]:
        """The paths a reached limit refused, cut short or left a branch of untraced (sorted)."""
        return tuple(sorted(self._path_ids))

    @property
    def affected_source_ids(self) -> tuple[str, ...]:
        """The sources whose light a reached limit left untraced (sorted)."""
        return tuple(sorted(self._source_ids))

    def _reached(self, limit: str, path_id: str | None, source_id: str) -> None:
        self._limits.add(limit)
        if path_id is not None:
            self._path_ids.add(path_id)
        self._source_ids.add(source_id)

    # -------------------------------------------------------------------------------------------------------- decisions
    def paths_left(self) -> bool:
        """The path budget still admits a path."""
        return self.paths < self.max_paths

    def admit_root(self) -> None:
        """A source line was launched (``paths_left`` held)."""
        self.paths += 1

    def refuse_line(self, source_id: str) -> None:
        """A source line is not launched: the path budget is spent (the caller books its power)."""
        self.untraced += 1
        self.untraced_lines += 1
        self._reached(MAX_PATHS, None, source_id)

    def reserve_branch(self, path_id: str, source_id: str) -> bool:
        """A traced branch of path ``path_id``: True admits it (one more path); False: the path budget is spent, the branch
        is not traced (the caller books its power)."""
        if self.paths >= self.max_paths:
            self.untraced += 1
            self._reached(MAX_PATHS, path_id, source_id)
            return False
        self.paths += 1
        return True

    def admit_probe(self, path_id: str, source_id: str) -> bool:
        """One untraced branch a probe pass would follow: True admits it (one more path); False: the path or step budget
        is spent, it is not followed (the caller counts it)."""
        paths_spent, steps_spent = self.paths >= self.max_paths, self.steps >= self.max_trace_steps
        if paths_spent or steps_spent:
            if paths_spent:
                self._limits.add(MAX_PATHS)
            if steps_spent:
                self._limits.add(MAX_TRACE_STEPS)
            self._source_ids.add(source_id)
            self._path_ids.add(path_id)
            return False
        self.paths += 1
        return True

    def push(self, item: T, power_W: float, path_id: str, source_id: str) -> None:
        """Queue a packet with chief power ``power_W``: the strongest queued packet is traced next (powers equal on the run's
        ordering grid, ``order_power``, in queue order), so a budget that runs out leaves the weakest branches untraced."""
        heapq.heappush(self._queue, (-order_power(power_W, self.order_quantum_W), self._queued, path_id, source_id, item))
        self._queued += 1

    def pending(self) -> bool:
        """Packets are waiting to be traced."""
        return bool(self._queue)

    def pop(self) -> tuple[T, bool]:
        """The strongest queued packet, and whether it is traced: False once the step budget is spent (the packet is not
        traced; the caller books its power)."""
        _, _, path_id, source_id, item = heapq.heappop(self._queue)
        if self.steps >= self.max_trace_steps:
            self.untraced += 1
            self._reached(MAX_TRACE_STEPS, path_id, source_id)
            return item, False
        return item, True

    def take_step(self, path_id: str, source_id: str) -> bool:
        """One chief-ray step of the path being traced: True takes it; False: the step budget is spent, the path stops
        here (the caller ends it where its chief ray's last event was)."""
        if self.steps >= self.max_trace_steps:
            self.cut_short += 1
            self._reached(MAX_TRACE_STEPS, path_id, source_id)
            return False
        self.steps += 1
        return True

    def keep_path(self, kept: int) -> bool:
        """A finished path is kept in the result when fewer than ``max_paths`` (``kept``) are."""
        return kept < self.max_paths

    def interaction_spent(self, interactions: int) -> bool:
        """A path that met ``interactions`` physical surfaces may not interact with another one."""
        return interactions >= self.max_interactions_per_path

    # ---------------------------------------------------------------------------------------------------------- issues
    def report(self, traced_paths: int, untraced_W: float, launched_W: float, unknown_paths: int = 0) -> ValidationIssue | None:
        """One ``MAX_PATHS`` issue for the whole run when the trace budget left power untraced (None: it did not): which
        limit, how many source lines / branches were not traced or cut short, and the power they carry (``untraced_W``,
        the ledger's ``PATH_LIMIT``; ``launched_W``: the sources' input). ``unknown_paths`` of them carry a power that is not
        known (the ledger holds it where it became unknown): where that is all of them, no number is stated."""
        if not self._limits:
            return None
        fraction = untraced_W / launched_W if launched_W > 0.0 else 0.0
        limits = " and ".join(text for key, text in ((MAX_PATHS, f"{self.max_paths} beam paths"),
                                                      (MAX_TRACE_STEPS, f"{self.max_trace_steps} trace steps")) if key in self._limits)
        parts = []
        branches = self.untraced - self.untraced_lines
        if branches:
            parts.append(f"{_count(branches, 'beam branch', 'beam branches')} not traced")
        if self.untraced_lines:
            parts.append(f"{_count(self.untraced_lines, 'source spectral line', 'source spectral lines')} not launched")
        if self.cut_short:
            parts.append(f"{_count(self.cut_short, 'beam path', 'beam paths')} cut short")
        carried = f"{untraced_W:.4g} W ({PERCENT * fraction:.4g} % of the launched power)"
        if unknown_paths:
            carried = "a power that is not known" if untraced_W <= 0.0 else f"{carried} and, on {unknown_paths} of them, a power that is not known"
        known_power = {} if unknown_paths and untraced_W <= 0.0 else {"untraced_power_W": untraced_W, "untraced_power_fraction": fraction}
        return make_issue(
            ErrorCode.MAX_PATHS, Status.WARNING,
            f"the run's trace budget was reached ({limits}): {', '.join(parts)}; they carry {carried}, which the results do not include",
            budget=sorted(self._limits), max_paths=self.max_paths, max_trace_steps=self.max_trace_steps,
            traced_paths=traced_paths, trace_steps=self.steps, untraced_branches=branches, untraced_source_lines=self.untraced_lines,
            paths_cut_short=self.cut_short,
            phase="primary", affected_path_ids=sorted(self._path_ids),
            affected_source_ids=sorted(self._source_ids),
            **known_power, **({"untraced_unknown_power_paths": unknown_paths} if unknown_paths else {}),
            suggested_action="reduce what multiplies the beam paths: partially reflecting surfaces facing each other (each "
                             "reflection back and forth doubles them), spectral lines per source, or sources")

    def interaction_issue(self, path_id: str, source_id: str, power_W: float, length_mm: float, power_bound_W: float | None = None) -> ValidationIssue:
        """The ``MAX_INTERACTIONS`` issue of a path ended at the per-path interaction limit (the guard against endless cavity
        loops; observation planes do not count): its remaining ``power_W`` (``power_bound_W`` given: not known, at most that) is not
        followed, so the issue is ``UNKNOWN``."""
        limit = self.max_interactions_per_path
        remaining = f"{power_W:.4g} W" if power_bound_W is None else f"power, not known and at most {power_bound_W:.4g} W,"
        amount = {"untraced_power_W": power_W} if power_bound_W is None else {"power_upper_bound_W": power_bound_W}
        return make_issue(
            ErrorCode.MAX_INTERACTIONS, Status.UNKNOWN,
            f"the path met {limit} surfaces (the per-path interaction limit, a guard against endless cavity loops; observation "
            f"planes do not count): its remaining {remaining} is not followed", location=f"path:{path_id}",
            limit=limit, **amount, geometric_path_length_mm=length_mm,
            phase="primary", budget=[MAX_INTERACTIONS_PER_PATH], affected_path_ids=[path_id],
            affected_source_ids=[source_id],
            suggested_action="raise max_interactions_per_path to follow more physical surfaces or cavity round trips")
