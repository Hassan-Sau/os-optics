"""The work budget of the return-light analysis (``validation/resonance.py``, ``validation/solid_folds.py``,
``validation/closed_returns.py``).

The analysis of one wavelength screens every pair of reflecting surfaces (``resonance._pairs``: polynomial in the
surfaces, a few milliseconds a pair), bounds the internal returns of plane-faced solids (``solid_folds``) and then searches
the closed returns of three or more surfaces (``closed_returns``: trails of directed legs, whose number can grow
exponentially with the surfaces of a large scene). All three count their work in one deterministic unit, the cone test of
the closed-return search (about 4 microseconds on a desktop CPU): each test of a reflected cone against a leg -- a
candidate transition, a step of a trail or of its closure check -- is one unit, and each pair the pair screen examines is
``PAIR_SCREEN_WORK`` units (its cost in cone tests, to within a factor of two) plus, where light between the pair passes
surfaces that turn it, ``pair_cone_work`` units for the cones those surfaces pass the light on in: with ``n`` such
surfaces, about ``2 n (n + 1)`` cones tested as arrays, ``PAIR_CONE_TESTS_PER_UNIT`` of them a unit, and each class of
parallel planes whose momenta a bound reads costs ``momentum_work`` units for the legs and images it tests
(``resonance._Scene.momenta``). (A solid of many refracting facets has every other facet of the solid between each pair and
many classes of parallel faces: both grow with its faces, a pair of a 96-faced rod costs about 10 ms, ten times a pair of a
lens.) The fold-solid analysis counts ``fold_cell_work`` units per batch of aperture cell pairs it tests, one per
direction cone it compares a new cone with, one per state it steps from and one per ``FOLD_ENTRIES_PER_UNIT`` entries of
each iteration of its sums (``solid_folds``: its cones, states and rounds have structural limits, but their product is no
limit on its time). Counting work, never time, keeps a run's result the same on every machine and runtime.

The limit (``NumericalConfig.max_return_analysis_work``) is the run's: every wavelength the run analyses draws on it, in
the order the run first reads them (``resonance.ReturnAnalysis``: the observation planes in Project order, each plane's
source lines in the trace's order, then the rated surfaces; the same order on every machine and runtime), each starting
with what the earlier ones left (``ReturnWork.before``: what they spent). So the whole run's return-light analysis does at
most that many units, however many source lines leave light out. An analysis that reaches it stops where it is
(``ReturnWork.exhausted``, ``stage``; a wavelength that finds nothing left stops before its first unit): the factors it did
not finish are not known, and every upper bound that reads them is withheld -- never the value of a screen that was not
finished.

The closed-return screen ends in one of three states (``ReturnGains.closed_return_screen``): ``COMPLETED_NO_RETURN``
(every trail examined, none closes: the open paths' count bounds the left-out light), ``POTENTIAL_RETURN`` (a closed
return the screen cannot rule out: bounded only by an all-orders series where one applies, else by nothing) and
``INCOMPLETE`` (the work limit came first: nothing is known, which bounds nothing). Its trail search holds at most one
frame per directed leg between the screened surfaces (``ClosedReturnSearch.frames_limit``), whatever the number of
trails. A run that is cancelled or fails during the analysis has no result at all (``run_control.RunStopped``): neither
is ever read as a screen that found no return.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..run_control import checkpoint

#: Work units of one pair of reflecting surfaces examined by the pair screen (its cost, in cone tests).
PAIR_SCREEN_WORK = 1000
#: Cones tested as arrays per work unit in the pair screen's light through turning surfaces (measured: about 0.47
#: microseconds a cone with its incidences, against 4 microseconds a unit).
PAIR_CONE_TESTS_PER_UNIT = 8
#: Legs and images tested per work unit in the momenta of a class of parallel planes (measured: about 0.78 microseconds a
#: test, against 4 microseconds a unit).
MOMENTUM_TESTS_PER_UNIT = 5
#: Aperture cell pairs the fold-solid analysis tests as arrays per work unit (``solid_folds._pair_angles``; measured: about
#: 0.36 microseconds a pair, against 4 microseconds a unit).
FOLD_CELL_PAIRS_PER_UNIT = 10
#: Entries of the step matrix per work unit in each iteration of the fold-solid analysis' certified sums
#: (``solid_folds._certified``: a multiply-add each, in plain Python).
FOLD_ENTRIES_PER_UNIT = 32
#: Work units between two checkpoints of the run's monitor (``os_optics/run_control.py``): progress and cancellation.
CHECKPOINT_WORK = 4096
#: The stages of the analysis, in the order they run.
PAIR_STAGE = "pairs"
SOLID_FOLD_STAGE = "solid_folds"
CLOSED_RETURN_STAGE = "closed_returns"
#: The three states of the closed-return screen (``ReturnGains.closed_return_screen``).
COMPLETED_NO_RETURN = "COMPLETED_NO_RETURN"
POTENTIAL_RETURN = "POTENTIAL_RETURN"
INCOMPLETE = "INCOMPLETE"
#: How the closed-return trail search ended (``ClosedReturnSearch.termination``): every trail examined (or none to
#: examine), a closed return found (the search stops at the first, shortest first), the work limit reached, not started
#: (the pair screen or the fold-solid analysis reached the limit first), or not run: the reflecting surfaces are parallel
#: planes (with a curved mirror closing them, possibly), whose closed returns an all-orders series bounds where one applies
#: (``PARALLEL_STACK``: the surfaces named are the stack's, not a trail).
SEARCH_COMPLETED = "COMPLETED"
RETURN_FOUND = "RETURN_FOUND"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
NOT_STARTED = "NOT_STARTED"
PARALLEL_STACK = "PARALLEL_STACK"


@dataclass(frozen=True)
class ReturnWork:
    """The work of one wavelength's return-light analysis: the units spent (``steps``; ``fold_steps`` of them by the
    fold-solid analysis), the limit it ran under (``limit``: what the run's limit left it, the wavelengths analysed before
    it having spent ``before``; ``None``: none), whether it stopped there before it finished (``exhausted``) and in which
    stage (``stage``: ``pairs``, ``solid_folds`` or ``closed_returns``)."""

    steps: int = 0
    limit: int | None = None
    exhausted: bool = False
    stage: str | None = None
    before: int = 0
    fold_steps: int = 0


@dataclass(frozen=True)
class ClosedReturnSearch:
    """What the closed-return screen's trail search did (``validation/closed_returns.py``): the longest trail it examined
    (``depth``: surfaces, its return to the first counted; it examines every trail of each length before a longer one),
    the most frames it held at once (``frames_peak``) against the most any trail of the scene needs (``frames_limit``:
    the directed legs between the surfaces of its largest group, plus one) and how it ended (``termination``:
    ``COMPLETED``, ``RETURN_FOUND``, ``BUDGET_EXHAUSTED`` or ``NOT_STARTED``). ``candidates``: the surfaces of every group
    the search screens (ids, scene order) -- every closed return of the screened kind runs among the surfaces of one
    group, so they name where closed returns can be at all, whichever one the search finds first. ``cycle_gain``: the
    amplitude gain bound of one round of the closed trail found (the product of its surfaces' reflectance bounds over the
    cones the closure check carries around it; ``None``: no trail found)."""

    depth: int = 0
    frames_peak: int = 0
    frames_limit: int = 0
    termination: str = NOT_STARTED
    candidates: tuple[tuple[str, ...], ...] = ()
    cycle_gain: float | None = None


def pair_cone_work(turning: int) -> int:
    """The work units of a pair's light through ``turning`` surfaces that turn it: ``2 n (n + 1)`` cones (each of the two
    surfaces: the light passed on by each turning surface, from the other surface and from every turning one), at
    ``PAIR_CONE_TESTS_PER_UNIT`` a unit."""
    return (2 * turning * (turning + 1)) // PAIR_CONE_TESTS_PER_UNIT


def fold_cell_work(pairs: int) -> int:
    """The work units of one batch of ``pairs`` aperture cell pairs the fold-solid analysis tests (at least 1)."""
    return 1 + pairs // FOLD_CELL_PAIRS_PER_UNIT


def momentum_work(tests: int) -> int:
    """The work units of the momenta of a class of parallel planes that tests ``tests`` legs and images (at least 1)."""
    return max(1, tests // MOMENTUM_TESTS_PER_UNIT)


class OutOfReturnWork(Exception):
    """The analysis reached its work limit."""


class WorkCounter:
    """Counts the units of one wavelength's analysis against its limit (``None``: no limit; ``before``: the units the run's
    earlier wavelengths spent of the run's limit, ``ReturnWork.before``)."""

    def __init__(self, limit: int | None = None, before: int = 0) -> None:
        self.limit, self.steps, self.stage, self.before, self.fold_steps = limit, 0, None, before, 0
        self._reported = 0  # (the units the run's monitor has been told of)

    def spend(self, units: int, stage: str) -> None:
        """Spend ``units`` in ``stage``; raises ``OutOfReturnWork`` (the units not spent) where they would pass the limit."""
        if self.limit is not None and self.steps + units > self.limit:
            self.stage = stage
            raise OutOfReturnWork
        self.steps += units
        if stage == SOLID_FOLD_STAGE:
            self.fold_steps += units
        if self.steps - self._reported >= CHECKPOINT_WORK:
            checkpoint(self.steps - self._reported)
            self._reported = self.steps

    def record(self) -> ReturnWork:
        return ReturnWork(self.steps, self.limit, self.stage is not None, self.stage, self.before, self.fold_steps)


__all__ = ["BUDGET_EXHAUSTED", "CHECKPOINT_WORK", "CLOSED_RETURN_STAGE", "COMPLETED_NO_RETURN", "ClosedReturnSearch",
           "FOLD_CELL_PAIRS_PER_UNIT", "FOLD_ENTRIES_PER_UNIT", "INCOMPLETE", "MOMENTUM_TESTS_PER_UNIT", "NOT_STARTED",
           "OutOfReturnWork", "PAIR_CONE_TESTS_PER_UNIT", "PAIR_SCREEN_WORK", "PAIR_STAGE", "PARALLEL_STACK", "POTENTIAL_RETURN",
           "RETURN_FOUND", "ReturnWork", "SEARCH_COMPLETED", "SOLID_FOLD_STAGE", "WorkCounter", "fold_cell_work", "momentum_work",
           "pair_cone_work"]
