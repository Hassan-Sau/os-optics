"""The public ``PowerBudget``: where the power of every source spectral line went, from the tracer's own ledger.

The tracer records every place where traced power leaves the modelled beams (``optics/rays/tracer.py``: ``PowerFlow``)
and every launched line (``SourceInput``). Here they are summed per source line and per element; nothing is re-traced.
Power counted at observation planes is reported next to the sinks (a non-absorbing monitor observes without consuming).
In a model that conserves energy ``input = sum of sinks``; the difference is reported as ``residual_W`` and a residual
above ``tolerance_W`` raises ``POWER_BALANCE_RESIDUAL`` (WARNING): it means some power left the beams without being
recorded, which the ledger's own tests treat as a defect.

``line_fate`` reads the ledger of the selected source lines to tell a plane no traced beam reaches apart: a *known* dark
plane (every launched watt ended somewhere known: detected elsewhere, absorbed, clipped, escaped, a declared loss) from
one the trace cannot speak for (power with an *unknown fate*: the trace budget, the per-path interaction limit, a line not
launched, a path ended out of model scope, or light that fell below the power floor while still a non-negligible part of
the lines' power). For one plane, the light that left the traced beams by a model approximation and may still reach it
(``validation/untraced.py``: an untraced reflection or coating transmission, light passing an aperture or an element
its chief ray misses, a beam's tail on an active area it is not counted on, part of a beam crossing the plane where its
chief ray does not) is passed in as ``reach`` and counts as unknown too: a dark plane is never a 0 W result while such
light may arrive.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from ..config import NumericalConfig
from ..contracts.basis import Omitted
from ..contracts.results import ElementPowerLine, ObservationResult, PowerBudget, PowerBudgetLine, PowerSinkEntry
from ..errors import ErrorCode, Status, ValidationIssue, make_issue
from ..errors.status import OmittedPart
from ..math.vectors import correctly_rounded_sum
from ..optics.rays.tracer import (
    SINK_APERTURE_TRUNCATION,
    SINK_BELOW_POWER_FLOOR,
    SINK_CLIPPED,
    SINK_DETECTED,
    SINK_ESCAPED,
    SINK_INTERACTION_LIMIT,
    SINK_KINDS,
    SINK_PATH_LIMIT,
    SINK_SOURCE_NOT_TRACED,
    SINK_TERMINATED_OUT_OF_MODEL,
    SINK_UNMODELLED_RESPONSE,
    SINK_UNTRACED_REFLECTION,
    SINK_BULK_SCATTER,
    SINK_BULK_LOSS_UNKNOWN,
    SINK_SURFACE_SCATTER,
    SINK_UNTRACED_TRANSMISSION,
    UNKNOWN_FATE_SINKS,
    TraceOutput,
)
from .superposition import received_power_W
from .untraced import (ACTIVE_AREA_TAIL, DIFFRACTED_BEAM, NEVER_LEFT_OUT_SINKS, PARTIAL_CROSSING, PROBE_BUDGET, PROBE_INTERACTION_LIMIT,
                       PROBE_OUT_OF_MODEL, PROBE_POWER_FLOOR, PROBE_SCATTERED, SAMPLED_BRANCH, left_out_ledger_W)

#: Light that may reach one plane although no traced beam of its line does (``validation/untraced.py``), in report order.
REACH_KINDS = (SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, SINK_CLIPPED, SINK_APERTURE_TRUNCATION, ACTIVE_AREA_TAIL,
               PARTIAL_CROSSING, DIFFRACTED_BEAM, PROBE_OUT_OF_MODEL, PROBE_BUDGET, PROBE_INTERACTION_LIMIT, PROBE_POWER_FLOOR, SAMPLED_BRANCH,
               PROBE_SCATTERED)
#: Why power has an unknown fate: the issue code a result names for each unknown-fate sink (and each kind of light that
#: may reach a plane no traced beam reaches). Light the probes followed to the plane (``BOUNDED_KINDS``) is
#: ``UNTRACED_LIGHT_BOUNDED`` in every result, whatever else is unknown: its code does not depend on whether the probes'
#: trace budget ran out for other light, which keeps its own code (``MAX_PATHS``) and issue.
#: The untraced reflection / transmission / aperture-truncation / partial-crossing kinds map to UNTRACED_LIGHT_BOUNDED
#: because a graded result follows them: validation runs the probes whenever the Project has an observation or a power
#: rating (``pipeline.run_physics``); the callers that skip the probes (control readings, network export) grade nothing.
UNKNOWN_FATE_CAUSES: dict[str, ErrorCode] = {
    SINK_PATH_LIMIT: ErrorCode.MAX_PATHS,
    SINK_INTERACTION_LIMIT: ErrorCode.MAX_INTERACTIONS,
    SINK_SOURCE_NOT_TRACED: ErrorCode.OUT_OF_MODEL_SCOPE,
    SINK_TERMINATED_OUT_OF_MODEL: ErrorCode.OUT_OF_MODEL_SCOPE,
    SINK_UNMODELLED_RESPONSE: ErrorCode.OUT_OF_MODEL_SCOPE,
    SINK_BULK_LOSS_UNKNOWN: ErrorCode.BULK_LOSS_NOT_ESTABLISHED,
    SINK_SURFACE_SCATTER: ErrorCode.SCATTERED_LIGHT_NOT_FOLLOWED,
    SINK_BULK_SCATTER: ErrorCode.SCATTERED_LIGHT_NOT_FOLLOWED,
    SINK_UNTRACED_REFLECTION: ErrorCode.UNTRACED_LIGHT_BOUNDED,
    SINK_UNTRACED_TRANSMISSION: ErrorCode.UNTRACED_LIGHT_BOUNDED,
    SINK_CLIPPED: ErrorCode.CLIPPED,
    SINK_APERTURE_TRUNCATION: ErrorCode.UNTRACED_LIGHT_BOUNDED,
    ACTIVE_AREA_TAIL: ErrorCode.INCOMPLETE_ENVELOPE,
    PARTIAL_CROSSING: ErrorCode.UNTRACED_LIGHT_BOUNDED,
    DIFFRACTED_BEAM: ErrorCode.GAUSSIAN_SEVERE_TRUNCATION,
    PROBE_OUT_OF_MODEL: ErrorCode.OUT_OF_MODEL_SCOPE,
    PROBE_BUDGET: ErrorCode.MAX_PATHS,
    PROBE_INTERACTION_LIMIT: ErrorCode.MAX_INTERACTIONS,
    PROBE_POWER_FLOOR: ErrorCode.ZERO_POWER,
    SAMPLED_BRANCH: ErrorCode.UNTRACED_LIGHT_BOUNDED,
    PROBE_SCATTERED: ErrorCode.SCATTERED_LIGHT_NOT_FOLLOWED,
}
#: What each kind of light a plane's traced power does not count is in a value-basis entry (``left_out_parts``): light a
#: budget stopped, light on a path or behind a response outside the traced model, light the trace does not follow as beams
#: (scatter, the branches a probe followed to the plane, what sampled rays leave behind), a hard edge's diffraction, light at
#: the power floor.
OMITTED_PARTS: dict[str, OmittedPart] = {
    SINK_PATH_LIMIT: OmittedPart.BUDGET,
    SINK_INTERACTION_LIMIT: OmittedPart.BUDGET,
    SINK_SOURCE_NOT_TRACED: OmittedPart.OUT_OF_MODEL,
    SINK_TERMINATED_OUT_OF_MODEL: OmittedPart.OUT_OF_MODEL,
    SINK_UNMODELLED_RESPONSE: OmittedPart.OUT_OF_MODEL,
    SINK_BULK_LOSS_UNKNOWN: OmittedPart.OUT_OF_MODEL,
    SINK_SURFACE_SCATTER: OmittedPart.NOT_FOLLOWED,
    SINK_BULK_SCATTER: OmittedPart.NOT_FOLLOWED,
    SINK_UNTRACED_REFLECTION: OmittedPart.NOT_FOLLOWED,
    SINK_UNTRACED_TRANSMISSION: OmittedPart.NOT_FOLLOWED,
    SINK_CLIPPED: OmittedPart.NOT_FOLLOWED,
    SINK_APERTURE_TRUNCATION: OmittedPart.NOT_FOLLOWED,
    ACTIVE_AREA_TAIL: OmittedPart.NOT_FOLLOWED,
    PARTIAL_CROSSING: OmittedPart.NOT_FOLLOWED,
    DIFFRACTED_BEAM: OmittedPart.EDGE_DIFFRACTION,
    PROBE_OUT_OF_MODEL: OmittedPart.OUT_OF_MODEL,
    PROBE_BUDGET: OmittedPart.BUDGET,
    PROBE_INTERACTION_LIMIT: OmittedPart.BUDGET,
    PROBE_POWER_FLOOR: OmittedPart.BELOW_POWER_FLOOR,
    SAMPLED_BRANCH: OmittedPart.NOT_FOLLOWED,
    PROBE_SCATTERED: OmittedPart.NOT_FOLLOWED,
}
#: The ledger kinds of the light sampled rays leave behind (``SAMPLED_BRANCH``): booked by the rays with no beam path, at the
#: surface where they left it.
SAMPLED_LEFT_BEHIND = (SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION)
#: The most places an omission of a plane's entry names (``Omitted.at``); the ledger (``PowerBudget.entries``) holds them all.
LOCATED_AT_LISTED = 16
_CAUSE_TEXT = {
    SINK_PATH_LIMIT: "the run's trace budget left {w} W untraced (MAX_PATHS)",
    SINK_INTERACTION_LIMIT: "{w} W stopped at the per-path interaction limit (MAX_INTERACTIONS)",
    SINK_SOURCE_NOT_TRACED: "{w} W of source lines was not launched (no medium data at the source)",
    SINK_TERMINATED_OUT_OF_MODEL: "{w} W left the model on paths ended out of model scope",
    SINK_UNMODELLED_RESPONSE: ("{w} W whose fate a response model does not state may reach it (UNMODELLED_RESPONSE: what a tabulated thin "
                               "lens does not deliver into its focused beam, what a diffraction grating's response does not give to the "
                               "orders it follows)"),
    SINK_BULK_LOSS_UNKNOWN: "{w} W traversed a material whose bulk absorption is not established; zero to this upper bound may reach it",
    SINK_SURFACE_SCATTER: "{w} W scattered by surfaces is not followed as rays and may reach it (SURFACE_SCATTER)",
    SINK_BULK_SCATTER: "{w} W scattered in the bulk of media is not followed and may reach it (BULK_SCATTER)",
    SINK_UNTRACED_REFLECTION: "{w} W of surface reflections the trace does not follow may reach it (UNTRACED_REFLECTION)",
    SINK_UNTRACED_TRANSMISSION: "{w} W that a reflector's coating transmits, not followed, may reach it (UNTRACED_TRANSMISSION)",
    SINK_CLIPPED: "{w} W of a beam booked CLIPPED, whose chief ray falls on an aperture rim, passes the opening and may reach it",
    SINK_APERTURE_TRUNCATION: "{w} W of a beam passes beside an element (beyond its support) and may reach it",
    ACTIVE_AREA_TAIL: ("at most {w} W of a beam crossing the plane beside its active area falls on the area (the beam is not "
                       "counted as an arrival: its chief ray is outside the area and less than capture_min_power_fraction of "
                       "it lies on the area where the chief ray crosses)"),
    PARTIAL_CROSSING: ("part of a traced beam reaches it before the surface where that beam's chief ray goes next, or on its way "
                       "out of the scene, while the chief ray does not cross it; the power that reaches it is not computed and "
                       "is at most {w} W"),
    DIFFRACTED_BEAM: ("a hard edge truncated a beam severely: its diffraction is not computed and reaches beyond its rays, so at most "
                      "{w} W of it may reach this plane ahead of that edge"),
    PROBE_OUT_OF_MODEL: ("{w} W of untraced light (reflections, coating transmissions, clipped light followed through the optics) "
                         "reached a path ended out of model scope, so where it goes is not known"),
    PROBE_BUDGET: "{w} W of untraced light was not followed: the trace budget for following it was reached (MAX_PATHS)",
    PROBE_INTERACTION_LIMIT: "{w} W of untraced light stopped at the per-path interaction limit (MAX_INTERACTIONS)",
    PROBE_POWER_FLOOR: "{w} W of untraced light stopped at the power floor (ZERO_POWER)",
    SAMPLED_BRANCH: "{w} W that sampled rays leave behind at surfaces (reflections the trace does not follow) may reach it",
    PROBE_SCATTERED: "{w} W of untraced light was scattered by a surface or a medium and is not followed",
}
_ORDER = UNKNOWN_FATE_SINKS + REACH_KINDS
#: What a finite bound of a plane no traced beam reaches decides (``validation/metrics.py`` ``_no_arrival``, decision B7):
#: a power target only in the direction the bound proves; nothing else.
BOUND_DECIDES = ("a finite bound decides a power target on this plane only in the direction it proves (a maximum at or above "
                 "it is met, a minimum above it is not); other power limits and beam metrics are not decided")
#: What makes a plane only part of a traced beam reaches (``PARTIAL_CROSSING``) readable, and what its bound decides.
PARTIAL_CROSSING_PLACEMENT = "place the plane where the beam's chief ray crosses it, or outside the beam's envelope, to read its power"
PARTIAL_CROSSING_ACTION = f"{PARTIAL_CROSSING_PLACEMENT}; {BOUND_DECIDES}"
#: What makes a plane readable behind a severe hard-edge truncation (``DIFFRACTED_BEAM``), and what its bound decides.
DIFFRACTED_BEAM_ACTION = ("open the aperture until the beam keeps its Gaussian state, or place the plane where this source line lights it; "
                          f"{BOUND_DECIDES}")
#: Kinds of light that may reach a plane whose power there the probe traces measured (every crossing): a finite amount.
FOLLOWED_REACH_KINDS = frozenset({SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, SINK_CLIPPED, SINK_APERTURE_TRUNCATION,
                                  ACTIVE_AREA_TAIL, PARTIAL_CROSSING, DIFFRACTED_BEAM})
#: Followed untraced light that bounds a result by a known amount: ``UNTRACED_LIGHT_BOUNDED`` (not the OUT_OF_MODEL_SCOPE of
#: light whose path left the model). A clipped beam and a tail keep their own codes (CLIPPED, INCOMPLETE_ENVELOPE).
BOUNDED_KINDS = frozenset({SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, SINK_APERTURE_TRUNCATION, PARTIAL_CROSSING})


def bounded_cause(kind: str) -> str:
    """The code a result names for a kind of light the probes followed (``UNTRACED_LIGHT_BOUNDED`` for an untraced
    reflection / transmission / light beside an element / part of a beam crossing a plane its chief ray does not; else
    the kind's unknown-fate cause): ``UNKNOWN_FATE_CAUSES``."""
    return UNKNOWN_FATE_CAUSES[kind].value


@dataclass(frozen=True)
class LineFate:
    """What the ledger of some source lines says about light no traced beam brought to a plane."""

    input_W: float
    unknown_W: dict[str, float]  # unknown-fate sink (or kind of light that may reach the plane) -> power, only with power
    below_floor_W: float  # BELOW_POWER_FLOOR: what paths still carried when they fell to the power floor
    floor_significant: bool  # that power is a non-negligible part of the lines' input (e.g. a source at the floor)
    # sum of sqrt(power) over the separate contributions of followed untraced light reaching the plane (sqrt(W))
    root_sum: float = 0.0
    # the largest field amplitude (sqrt(W)) light the plane's power does not count may add (``untraced.FAINT_ROOT``): the
    # light left out as negligible and the followed contributions at or below the negligible level; only coherent sums
    faint_root: float = 0.0
    # the most power (W) that light may add (``untraced.FAINT_POWER``; ``inf``: unbounded); the power reads it
    faint_W: float = 0.0
    # that light is unbounded because the return-light analysis reached its work limit before it bounded the light's returns
    # (``untraced.WITHHELD_BY_WORK``, ``RETURN_ANALYSIS_BUDGET``), not because surfaces are known to hold it
    withheld_by_work: bool = False
    # else the returns the finished analysis found that leave it unbounded there, each a clause (``untraced.return_causes``:
    # a pair of surfaces facing each other, a closed path, the parallel surfaces of a stack, the internal folds of a solid)
    returns: tuple[str, ...] = ()
    # the lines' negligible level (W): ``negligible_power_fraction`` x their input
    level_W: float = 0.0

    @property
    def faint_accounted(self) -> bool:
        """The light left out as negligible is an accounting omission here: the most it brings to this plane (``faint_W``,
        its crossings and returns included) is within the lines' negligible level, so it is the stated truncation of the
        value and changes no state or status (a target whose limit lies inside its bound is still not decided by the
        truncation). Above the level -- surfaces facing each other, a closed path or
        many crossings may send it across the plane again and again -- or with nothing bounding it, what it may add is an
        uncertainty of the value (``OmittedPart.POSSIBLE_RETURN``): the plane's power is not complete and a target there is
        judged on the interval it leaves."""
        return self.faint_W <= self.level_W

    def unbounded_returns(self, plane: str = "this plane") -> str:
        """Why nothing bounds the light left out as negligible on a plane: the return-light analysis reached its work limit
        (``withheld_by_work``, ``RETURN_ANALYSIS_BUDGET``), or the returns it found (``returns``: a pair of surfaces facing
        each other, a closed path through three or more surfaces, the parallel surfaces of a stack, the folds of a solid)."""
        if self.withheld_by_work:
            return ("the return-light analysis reached its work limit before it bounded how often that light returns across "
                    f"{plane}, RETURN_ANALYSIS_BUDGET")
        if self.returns:
            return f"{' or '.join(self.returns)} and send it across {plane}"
        return f"nothing bounds how often it returns across {plane}"

    @property
    def unknown_total_W(self) -> float:
        return correctly_rounded_sum(self.unknown_W.values())

    @property
    def partial_only(self) -> bool:
        """All the light that may reach the plane is part of traced beams crossing it where their chief rays do not
        (``PARTIAL_CROSSING``): nothing else is unknown."""
        return bool(self.unknown_W) and set(self.unknown_W) <= {PARTIAL_CROSSING} and not self.floor_significant

    @property
    def complete(self) -> bool:
        """Every launched watt of these lines has a known fate: a plane none of their traced beams reaches is dark."""
        return not self.unknown_W and not self.floor_significant

    @property
    def untraced_W(self) -> float:
        """Power whose fate is not followed (unknown-fate sinks and light that may reach the plane, plus a significant
        below-floor remainder)."""
        return self.unknown_total_W + (self.below_floor_W if self.floor_significant else 0.0)

    @property
    def unbounded(self) -> bool:
        """Part of the untraced power has an unknown fate (not only light the probes followed to the plane): it may cross a
        transparent plane any number of times."""
        return self.floor_significant or any(kind not in FOLLOWED_REACH_KINDS for kind in self.unknown_W)

    def faint_code(self, amount: float) -> str:
        """The code of the light left out as negligible that may add ``amount`` (W, or sqrt(W) for a field) to a result:
        ``FAINT_LIGHT_BOUNDED``, except where the amount is not finite because the return-light analysis reached its work
        limit first (``withheld_by_work``): then ``RETURN_ANALYSIS_BUDGET``."""
        if not math.isfinite(amount) and self.withheld_by_work:
            return ErrorCode.RETURN_ANALYSIS_BUDGET.value
        return ErrorCode.FAINT_LIGHT_BOUNDED.value

    def cause_codes(self) -> tuple[str, ...]:
        """The code of each kind of light the result does not count (``UNKNOWN_FATE_CAUSES``; ``ZERO_POWER`` for the power
        floor), in report order: the same code for the same light whether or not other light has an unknown fate."""
        codes = [UNKNOWN_FATE_CAUSES[kind].value for kind in _ORDER if kind in self.unknown_W]
        if self.floor_significant:
            codes.append(ErrorCode.ZERO_POWER.value)
        return tuple(dict.fromkeys(codes))

    def causes(self) -> list[str]:
        """The ledger sinks / kinds of light behind ``cause_codes`` (``ZERO_POWER`` for the power floor)."""
        return [kind for kind in _ORDER if kind in self.unknown_W] + ([ErrorCode.ZERO_POWER.value] if self.floor_significant else [])

    def describe(self) -> str:
        parts = [_CAUSE_TEXT[kind].format(w=f"{self.unknown_W[kind]:.4g}") for kind in _ORDER if kind in self.unknown_W]
        if self.floor_significant:
            parts.append(f"{self.below_floor_W:.4g} W fell below the power floor, where paths stop (ZERO_POWER)")
        return "; ".join(parts)


def line_fate(lines: Iterable[PowerBudgetLine], cfg: NumericalConfig, reach: dict[str, float] | None = None,
              root_sum: float = 0.0, faint_root: float = 0.0, faint_W: float = 0.0, withheld_by_work: bool = False,
              returns: tuple[str, ...] = ()) -> LineFate:
    """Classify the ledger of ``lines`` (and, for one plane, the light ``reach`` that may still reach it although no traced
    beam of these lines does: kind -> power, ``validation/untraced.py``). A line's unknown-fate power other than what a
    budget stopped (a path ended out of model scope, a response model's unstated share, scattered light not followed) counts
    only above the line's negligible level (``untraced.left_out_ledger_W``); what a budget stopped (``PATH_LIMIT``,
    ``INTERACTION_LIMIT``) counts whatever its power. Below-floor power is *significant* when it
    exceeds the negligible level of the lines together (``negligible_power_fraction`` x their input): then the
    floor, not the optics, decided that the light stops (a source at or below ``power_epsilon_W``); below it, the
    remainder is below the level a result leaves out for these lines (the 1e-36 W a crossed polarizer leaves; the 3e-18 W left of a 1e-12 W
    line next to a 1 W line)."""
    unknown: dict[str, float] = {}
    total_in = below = 0.0
    for line in lines:
        total_in += line.input_power_W
        # (a line whose unknown-fate power -- other than what a budget stopped -- is at or below its negligible level is
        # complete: that power is left out as negligible, in the left-out light of ``untraced.surface_reach``)
        left_out = left_out_ledger_W(line.sinks, line.input_power_W, cfg) > 0.0
        for kind in UNKNOWN_FATE_SINKS:
            power = line.sinks.get(kind, 0.0)
            if power > 0.0 and not (left_out and kind not in NEVER_LEFT_OUT_SINKS):
                unknown[kind] = unknown.get(kind, 0.0) + power
        below += line.sinks.get(SINK_BELOW_POWER_FLOOR, 0.0)
    for kind, power in (reach or {}).items():
        if power > 0.0:
            unknown[kind] = unknown.get(kind, 0.0) + power
    level = cfg.negligible_power_fraction * total_in
    return LineFate(total_in, unknown, below, below > level, root_sum, faint_root, faint_W, withheld_by_work, returns, level)


def left_out_parts(fate: LineFate, budget: PowerBudget | None = None, *, absorbing: bool, gap: float | None,
                   located: dict[str, tuple[str, ...]] | None = None,
                   coherent: tuple[float, float, float] | None = None) -> tuple[Omitted, ...]:
    """What a plane's traced power does not count, as the omissions of its value-basis entry: one per kind of light of
    ``fate`` in report order -- its part (``OMITTED_PARTS``), the code a result names for it (``UNKNOWN_FATE_CAUSES``), where
    it left the traced light (``at``: the surfaces, else the paths or sources, of the ledger's entries of that kind, at most
    ``LOCATED_AT_LISTED``; ``located``: kind -> the places of light a probe followed to this plane) and the most it adds
    (``upper``, W) -- then the power floor's remainder where it is significant, and the light left out as negligible.

    ``upper`` is the kind's power where that bounds what it adds: light the probes followed to the plane (counted at every
    crossing) on any plane, light of unknown fate on an absorbing plane (each watt is absorbed once); null for light of
    unknown fate on a transparent plane (it may cross more than once) and for left-out light nothing bounds. No part
    adds more than ``gap`` (the plane's ``power_upper_bound_W`` less its traced power; ``None``: no supported bound).

    ``coherent`` -- ``(low, high, value)``: the traced coherent sum's ends and its value (W) -- states the parts of a
    coherent sum, which adds fields, not powers: a part of field amplitude at most ``a`` (sqrt(W)) moves the value by at
    most ``max((sqrt(high) + a)^2 - value, value - max(0, sqrt(low) - a)^2)``. Light the probes followed to the plane has
    the amplitude of all its separate contributions together (``LineFate.root_sum``: a kind's own contributions add their
    amplitudes, so the square root of its power is no bound), the light left out as negligible ``LineFate.faint_root``.
    Light of unknown fate and a significant power-floor remainder may arrive in any number of beams, on an absorbing plane
    too: no bound. The left-out light is an accounting omission only while its field's bound at the sum,
    ``(sqrt(high) + faint_root)^2 - high``, stays within the lines' negligible level (W1), else its possible return. The
    parts' bounds do not add up to the sum's interval: amplitudes add."""

    def most(power: float, bounded: bool) -> float | None:
        if not bounded or not math.isfinite(power):
            return None
        return power if gap is None else min(power, gap)

    def places(kind: str) -> tuple[str, ...]:
        if located and kind in located:
            return located[kind][:LOCATED_AT_LISTED]
        found: list[str] = []
        for entry in () if budget is None else budget.entries:
            if entry.power_W <= 0.0 or not (entry.kind == kind or kind == SAMPLED_BRANCH and entry.kind in SAMPLED_LEFT_BEHIND
                                            and entry.path_id is None):
                continue
            where = (f"surface:{entry.surface_id}" if entry.surface_id else f"path:{entry.path_id}" if entry.path_id
                     else f"source:{entry.source_id}")
            if where not in found:
                found.append(where)
        return tuple(sorted(found)[:LOCATED_AT_LISTED])

    if coherent is not None:
        return _coherent_parts(fate, coherent, places)
    parts = [Omitted(part=OMITTED_PARTS[kind], at=places(kind), upper=most(fate.unknown_W[kind], absorbing or kind in FOLLOWED_REACH_KINDS),
                     code=UNKNOWN_FATE_CAUSES[kind].value)
             for kind in _ORDER if kind in fate.unknown_W]
    if fate.floor_significant:
        parts.append(Omitted(part=OmittedPart.BELOW_POWER_FLOOR, at=places(SINK_BELOW_POWER_FLOOR), upper=most(fate.below_floor_W, absorbing),
                             setting="power_epsilon_W", code=ErrorCode.ZERO_POWER.value))
    if fate.faint_W > 0.0:
        # (within the level: an accounting omission; above it, what its returns may add is an uncertainty)
        parts.append(Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL if fate.faint_accounted else OmittedPart.POSSIBLE_RETURN,
                             upper=most(fate.faint_W, True), setting="negligible_power_fraction", code=fate.faint_code(fate.faint_W)))
    return tuple(parts)


def _coherent_parts(fate: LineFate, coherent: tuple[float, float, float], places) -> tuple[Omitted, ...]:
    """``left_out_parts`` of a coherent sum (``coherent``: its traced ends and value, W): each part's ``upper`` is the most
    its field moves the value (``left_out_parts``)."""
    low, high, value = coherent

    def moves(root: float) -> float | None:
        # (the most a field of amplitude ``root`` (sqrt(W)) moves the value, up from the upper end or down from the lower)
        if not (math.isfinite(root) and math.isfinite(high)):
            return None
        fall = math.sqrt(max(low, 0.0)) - root
        return max((math.sqrt(high) + root) ** 2 - value, value - (fall * fall if fall > 0.0 else 0.0), 0.0)

    parts = [Omitted(part=OMITTED_PARTS[kind], at=places(kind), upper=moves(fate.root_sum) if kind in FOLLOWED_REACH_KINDS else None,
                     code=UNKNOWN_FATE_CAUSES[kind].value)
             for kind in _ORDER if kind in fate.unknown_W]
    if fate.floor_significant:
        parts.append(Omitted(part=OmittedPart.BELOW_POWER_FLOOR, at=places(SINK_BELOW_POWER_FLOOR), upper=None,
                             setting="power_epsilon_W", code=ErrorCode.ZERO_POWER.value))
    if fate.faint_root > 0.0:
        # (an accounting omission only while its field's bound at the sum is within the level (W1), not only its power)
        at_sum = (math.sqrt(high) + fate.faint_root) ** 2 - high if math.isfinite(high) else math.inf
        accounted = fate.faint_accounted and at_sum <= fate.level_W
        parts.append(Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL if accounted else OmittedPart.POSSIBLE_RETURN,
                             upper=moves(fate.faint_root), setting="negligible_power_fraction", code=fate.faint_code(fate.faint_root)))
    return tuple(parts)


def _ordered(sinks: dict[str, float]) -> dict[str, float]:
    return {kind: sinks[kind] for kind in SINK_KINDS if kind in sinks}


def build_power_budget(trace: TraceOutput, observations: list[ObservationResult], cfg: NumericalConfig) -> tuple[PowerBudget, list[ValidationIssue]]:
    """Aggregate the trace's ledger into a ``PowerBudget`` (and a ``POWER_BALANCE_RESIDUAL`` issue when it does not close)."""

    entries = tuple(
        PowerSinkEntry(kind=f.kind, power_W=f.power_W, source_id=f.source_id, spectral_index=f.spectral_index, wavelength_nm=f.wavelength_nm,
                       path_id=f.path_id, surface_id=f.surface_id, element_id=f.owner_id, detail=f.detail)
        for f in trace.power_flows
    )
    observed: dict[tuple[str, int], dict[str, float]] = {}
    # (a line some of whose light's power is out of model scope on a plane -- ArrivalResult / SampledLightAtPlane
    # .power_out_of_scope_at -- observes no stated power there: the sum would add chief-ray powers, not the light's)
    scoped: dict[tuple[str, int], set[str]] = {}
    for observation in observations:
        for arrival in observation.arrivals:
            key = (arrival.beam.source_id, arrival.beam.spectral_index)
            if arrival.power_out_of_scope_at:
                scoped.setdefault(key, set()).add(observation.observation_id)
            if arrival.beam.power_W is None:  # (a power that is not known observes nothing known)
                continue
            per = observed.setdefault(key, {})
            received = received_power_W(arrival)
            per[observation.observation_id] = per.get(observation.observation_id, 0.0) + received
        for sampled in observation.sampled_light:
            key = (sampled.source_id, sampled.spectral_index)
            if sampled.power_out_of_scope_at:
                scoped.setdefault(key, set()).add(observation.observation_id)
            per = observed.setdefault(key, {})
            per[observation.observation_id] = per.get(observation.observation_id, 0.0) + sampled.power_W
    lines: list[PowerBudgetLine] = []
    issues: list[ValidationIssue] = []
    total_sinks: dict[str, float] = {}
    total_input = 0.0
    for line in trace.inputs:
        sinks: dict[str, float] = {}
        for entry in entries:
            if entry.source_id == line.source_id and entry.spectral_index == line.spectral_index:
                sinks[entry.kind] = sinks.get(entry.kind, 0.0) + entry.power_W
                total_sinks[entry.kind] = total_sinks.get(entry.kind, 0.0) + entry.power_W
        residual = line.power_W - correctly_rounded_sum(sinks.values())
        tolerance = cfg.energy_conservation_tol * line.power_W + cfg.power_epsilon_W
        # a sink below zero means a model created power there: the ledger does not close even when the sums agree
        negative = sorted(kind for kind, power in sinks.items() if power < -tolerance)
        conserved = abs(residual) <= tolerance and not negative
        total_input += line.power_W
        lines.append(PowerBudgetLine(
            source_id=line.source_id, spectral_index=line.spectral_index, wavelength_nm=line.wavelength_nm, input_power_W=line.power_W,
            sinks=_ordered(sinks), observed_W={oid: power for oid, power in sorted(observed.get((line.source_id, line.spectral_index), {}).items())
                                               if oid not in scoped.get((line.source_id, line.spectral_index), ())},
            observed_out_of_scope_at=tuple(sorted(scoped.get((line.source_id, line.spectral_index), ()))),
            residual_W=residual, conserved=conserved, generated_from=line.generated_from,
        ))
        if negative:
            issues.append(make_issue(ErrorCode.POWER_BALANCE_RESIDUAL, Status.WARNING,
                                     f"the power ledger of this source line does not close: {', '.join(negative)} is negative, so "
                                     "an interaction there gave out more power than it received",
                                     location=f"source:{line.source_id}", spectral_index=line.spectral_index, residual_W=residual,
                                     input_power_W=line.power_W, negative_sinks=negative))
        elif not conserved:
            issues.append(make_issue(ErrorCode.POWER_BALANCE_RESIDUAL, Status.WARNING,
                                     "the power ledger of this source line does not close: some power left the beams unrecorded",
                                     location=f"source:{line.source_id}", spectral_index=line.spectral_index, residual_W=residual,
                                     input_power_W=line.power_W))
    by_element: dict[str, dict[str, float]] = {}
    for entry in entries:
        if entry.element_id is None or entry.kind in (SINK_DETECTED, SINK_ESCAPED):
            continue
        per = by_element.setdefault(entry.element_id, {})
        per[entry.kind] = per.get(entry.kind, 0.0) + entry.power_W
    element_lines = tuple(ElementPowerLine(element_id=element, sinks=_ordered(sinks), total_loss_W=correctly_rounded_sum(sinks.values()))
                          for element, sinks in sorted(by_element.items()))
    residual = total_input - correctly_rounded_sum(total_sinks.values())
    tolerance = cfg.energy_conservation_tol * total_input + cfg.power_epsilon_W * max(1, len(trace.inputs))
    budget = PowerBudget(lines=tuple(lines), by_element=element_lines, entries=entries, total_input_W=total_input,
                         total_sinks=_ordered(total_sinks), residual_W=residual, tolerance_W=tolerance,
                         conserved=abs(residual) <= tolerance and not any(i.code is ErrorCode.POWER_BALANCE_RESIDUAL
                                                                          and "negative_sinks" in i.numerical_context for i in issues))
    return budget, issues
