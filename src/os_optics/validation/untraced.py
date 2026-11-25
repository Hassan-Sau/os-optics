"""Light the trace does not follow that may still reach a plane no traced beam reaches.

A plane no traced beam of a source line reaches is a *known zero* for that line only when none of the line's light can
arrive there. Besides the ledger's unknown-fate sinks (``validation/power.py``), two kinds of light leave the traced
beams by a model approximation while going on physically:

* **untraced branches** (``tracer.UntracedBranch``): an interface reflection or a reflector coating's transmission the
  trace does not follow, the part of a beam inside a clear aperture whose chief ray falls on the opaque rim (the whole
  beam is booked ``CLIPPED``), the part of a beam beyond an element's support (it passes beside the element);
* **uncounted crossings**: a beam that crosses the plane beside its active area, whose Gaussian tail on the area is
  below ``capture_min_power_fraction`` (so it is not an arrival) but not negligible -- unless the plane's acceptance cone
  provably rejects every direction that tail travels in (``_arrival_power``), as it rejects a counted beam outside it;
* **partial crossings** (``tracer.PartialCrossing``, ``PARTIAL_CROSSING``): part of a traced beam reaches an observation
  plane (transparent or absorbing) on a straight leg its chief ray does not cross the plane on, before the surface the
  leg ends at (a physical surface, an absorbing plane that stops the beam). The beam's power follows its chief ray, so
  that light is not on the plane's results; the tracer bounds it (``decisions.leg_share``) by the Gaussian beam's share that
  meets the plane after the leg's start surface, before its end surface and on the plane's active area: the larger of the
  rays' picture (parallel to the chief ray) and the paraxial energy flow's, the only one on a leg that leaves the scene.
  Without a valid Gaussian state it is the whole beam when a live pupil ray meets the plane first or the plane comes
  within three envelopes of those rays. A ray of a beam with a valid Gaussian state that passes the plane of a plane end
  surface beyond its support (a hole, past its outline) ends there for this count: that light is the end surface's
  untraced branch, counted once by its probe.
  Behind a severe hard-edge truncation of a traced beam it is the whole beam on any
  plane ahead that its line does not light (``DIFFRACTED_BEAM``): that diffraction is not localized. A plane the chief ray crosses beside its active area reads the energy flow's on the
  area too (``decisions.beside_area_share``, its tail). A probe's partial crossings count as its own light.

Each untraced branch of a line that is dark somewhere is followed by a **probe trace**: the same tracer, from the branch's
own state (chief ray, Jones vector, Gaussian beam, medium), through every optic it meets -- refracted, reflected, split,
absorbed, clipped exactly as a traced beam -- with the pupil rays of two whole rings of the beam it leaves (the outermost
and the middle one, mapped by the branch: reflected with a ghost, on through a coating; a symmetric sample, against which
its Gaussian state is checked where it meets a surface, as a traced beam's is). Where a traced beam's power that passes
is not computed (a beam without a valid Gaussian state whose pupil rays an aperture cuts) a probe goes on with its whole
power -- probe light only ever raises a bound --, and behind a central obstruction a probe's annulus goes on without pupil
rays (its bounds alone decide). Its own untraced light (a ghost of the ghost, ...) is followed the same way, recursively
and strongest first, down to the negligible level of its line (below). The probe reaches a plane when it crosses it as an
arrival (its received power), or beside the plane's active area with a non-negligible Gaussian tail on it; probe light
whose fate the probe does not know (its trace budget, interaction limit, a path ended out of model scope, light at the
power floor, a branch the budget did not reach -- then also a ``MAX_PATHS`` issue) may reach every plane. Light at the
interaction limit may be stored in a cavity: it is never left out as negligible, whatever its power
(``STORED_KINDS``). A branch of light that passes beside an element (beyond its support) does not meet that element's
surfaces on its way past.

"Negligible" is the line's negligible level: at most ``negligible_power_fraction`` x the line's input (an
untraced branch at or below it is not followed, at any order). The
power that may reach a plane (``untraced_power_W``) is the first-order light's power where it arrives there (the probe's
own value, beside an active area its tail) plus the power whose fate is not known; it never adds power to a result. In a
run that follows every untraced branch, the light left out as negligible bounds a plane's power targets too
(``FAINT_POWER``, ``FAINT_ROOT``): a plane none of the followed light reaches is then not a known zero for them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from ..config import NumericalConfig
from ..contracts.results import FollowedLight
from ..optics.materials.library import MaterialLibrary
from ..errors import ErrorCode, Status, ValidationIssue, make_issue
from ..math.vectors import correctly_rounded_sum
from ..optics.rays.tracer import (
    ProbeOutcome,
    UntracedBranch,
    SINK_BELOW_POWER_FLOOR,
    SINK_INTERACTION_LIMIT,
    SINK_PATH_LIMIT,
    SINK_SOURCE_NOT_TRACED,
    SINK_TERMINATED_OUT_OF_MODEL,
    SINK_UNMODELLED_RESPONSE,
    UNKNOWN_FATE_SINKS,
    SINK_BULK_SCATTER,
    SINK_BULK_LOSS_UNKNOWN,
    SINK_SURFACE_SCATTER,
    ArrivalRecord,
    Tracer,
    TraceOutput,
)
from ..optics.rays.hits import SurfaceHit
from ..scene.model import Scene
from .resonance import ReturnGains

#: Hazard kind of a crossing beside the active area whose Gaussian tail on it is not negligible.
ACTIVE_AREA_TAIL = "ACTIVE_AREA_TAIL"
#: Hazard kind of part of a traced beam crossing a plane on a leg its chief ray does not cross it on (``PartialCrossing``).
PARTIAL_CROSSING = "PARTIAL_CROSSING"
#: Hazard kind of a beam a hard edge truncated severely, on a plane ahead of it its line does not light: its diffraction is not localized.
DIFFRACTED_BEAM = "DIFFRACTED_BEAM"
#: Hazard kinds of untraced light whose fate its probe trace does not know, by why (it may reach any plane).
PROBE_OUT_OF_MODEL = "UNTRACED_LIGHT_OUT_OF_MODEL"  # a probe path ended out of model scope (grazing, medium mismatch, ...)
PROBE_BUDGET = "UNTRACED_LIGHT_BUDGET"  # the probes' trace budget did not reach it
PROBE_INTERACTION_LIMIT = "UNTRACED_LIGHT_INTERACTION_LIMIT"  # a probe path met max_interactions_per_path
PROBE_POWER_FLOOR = "UNTRACED_LIGHT_POWER_FLOOR"  # a probe path stopped at the power floor
PROBE_SCATTERED = "UNTRACED_LIGHT_SCATTERED"  # a surface or a medium scattered part of a probe's light (not followed)
_PROBE_FATE_KIND = {SINK_TERMINATED_OUT_OF_MODEL: PROBE_OUT_OF_MODEL, SINK_SOURCE_NOT_TRACED: PROBE_OUT_OF_MODEL,
                    SINK_UNMODELLED_RESPONSE: PROBE_OUT_OF_MODEL, SINK_BULK_LOSS_UNKNOWN: PROBE_OUT_OF_MODEL, SINK_PATH_LIMIT: PROBE_BUDGET, SINK_INTERACTION_LIMIT: PROBE_INTERACTION_LIMIT,
                    SINK_BELOW_POWER_FLOOR: PROBE_POWER_FLOOR, SINK_SURFACE_SCATTER: PROBE_SCATTERED, SINK_BULK_SCATTER: PROBE_SCATTERED}

LineKey = tuple[str, int]  # (source_id, spectral_index)
#: observation id -> source line -> hazard kind -> power (W) that may reach the plane; the entry ``ROOT_SUM`` (not a kind)
#: is the sum over the separate contributions of light the probes followed to the plane (each probe crossing, each tail)
#: of the square root of its power: sqrt(W), the bound of their summed field amplitude (``hazard_root_sum``)
PlaneReach = dict[str, dict[LineKey, dict[str, float]]]
#: The ``PlaneReach`` entry of the summed square roots of the followed contributions (not a hazard kind).
ROOT_SUM = "_root_sum"
#: The ``PlaneReach`` entry (not a hazard kind) of the largest field amplitude (sqrt(W)) that light of a line the plane's
#: power does not count may still add to the line's light there: the followed contributions at or below the negligible
#: level (their square roots) and the light left out as negligible (the sum of the square roots of its pieces' powers --
#: mutually coherent, they may meet in phase -- times the most field its open paths' crossings bring, from either side,
#: ``ReturnGains.crossing_field``, and times the plane's factor where two surfaces facing each other may return it across
#: the plane again and again, ``validation/resonance.py``; ``faint_root_sum``). Only the coherent power reads it.
FAINT_ROOT = "_faint_root"
#: The ``PlaneReach`` entry (not a hazard kind) of the most power (W) that light of a line the plane's power does not count
#: may still bring to it: the followed contributions at or below the negligible level and the light left out as negligible
#: -- on an absorbing plane its power (each watt is absorbed once), on a transparent one its power times the most its open
#: paths' crossings carry across the plane (``ReturnGains.crossings``: once per straight leg after each number of
#: reflections, less after weak ones, at least twice) and times the plane's power factor where two surfaces facing each
#: other may hold it and send it across the plane (``faint_power_sum``; unbounded where nothing bounds their returns:
#: ``inf``). The power reads it.
FAINT_POWER = "_faint_power"
#: The same light counted once, as a detector placed on the plane would receive it, whatever the plane does (``ObservationResult.absorbed_power_upper_bound_W``).
FAINT_ONCE = "_faint_once"
#: The ``PlaneReach`` entry (not a hazard kind) of the largest field amplitude, as the square root of a beam irradiance
#: (sqrt(W/mm^2): power per unit area perpendicular to the beam, the pair metrics' irradiance), that light of a line the
#: trace does not follow may add anywhere on the plane: the sum over the probes' crossings of the plane (and the traced
#: beams' tails beside its active area) of the square root of each one's peak (its Gaussian peak ``2 P / (pi w_x w_y)``,
#: or the diffraction limit of its power where its Gaussian state is not valid there or its rays show aberrations), and
#: the light left out as negligible: the diffraction limit's amplitude of its pieces' amplitudes added, times its open
#: paths' crossing field and the plane's return factor (``irradiance_root_sum``; the pair metrics' irradiances read it,
#: ``validation/metrics.py``).
IRRADIANCE_ROOT = "_irradiance_root"
#: The ``PlaneReach`` entry (not a hazard kind; 1.0 when present) of a line whose left-out light's returns across the plane the
#: return-light analysis did not bound: it reached its work limit (``ReturnGains.screened`` false,
#: ``RETURN_ANALYSIS_BUDGET``), so ``FAINT_POWER`` / ``FAINT_ROOT`` are unbounded for that reason, not for a known pair.
WITHHELD_BY_WORK = "_withheld_by_work"
#: The prefix of the ``PlaneReach`` entries (not hazard kinds; 1.0 each) that name why nothing bounds a line's left-out light
#: across the plane in a finished analysis, one per cause (``ReturnGains.return_causes``: a pair, a closed path, a stack, the
#: folds of a solid): the key is the prefix and the clause.
RETURN_CAUSE = "_return_cause:"
#: The ``PlaneReach`` entries that are no hazard kind (and every key that starts with ``RETURN_CAUSE``).
_NOT_KINDS = frozenset({ROOT_SUM, FAINT_ROOT, FAINT_POWER, FAINT_ONCE, IRRADIANCE_ROOT, WITHHELD_BY_WORK})

#: Probe power that leaves the model's knowledge: it may reach any plane.
_PROBE_UNFOLLOWED = frozenset(UNKNOWN_FATE_SINKS) | {SINK_BELOW_POWER_FLOOR}
#: Probe light at the interaction limit may be light stored in a cavity (a resonator): it is never left out as negligible.
STORED_KINDS = frozenset({PROBE_INTERACTION_LIMIT})
#: The trace keeps light whose path a budget stopped and power whose bulk loss is uncharacterized in the unknown-fate
#: ledger at any power. The other unknown-fate sinks of a line are left out together when their sum is at or below the
#: line's negligible level (``left_out_ledger_W``).
NEVER_LEFT_OUT_SINKS = frozenset({SINK_PATH_LIMIT, SINK_INTERACTION_LIMIT, SINK_BULK_LOSS_UNKNOWN})


def left_out_ledger_W(sinks: Mapping[str, float], input_W: float, cfg: NumericalConfig) -> float:
    """The unknown-fate power of one source line's ledger (``sinks``: kind -> power) that is left out as negligible: the sum
    of its unknown-fate sinks other than ``NEVER_LEFT_OUT_SINKS`` when that sum is at or below the line's negligible level
    (``negligible_power_fraction`` x ``input_W``), else 0.0 (it then has an unknown fate: the line is not complete)."""
    total = correctly_rounded_sum(sinks.get(kind, 0.0) for kind in UNKNOWN_FATE_SINKS if kind not in NEVER_LEFT_OUT_SINKS)
    return total if 0.0 < total <= cfg.negligible_power_fraction * input_W else 0.0


#: Light of an extended emitter's rays that may reach a plane (``validation/sampled_light.py``): what they left behind at an
#: interface whose reflection is not followed.
SAMPLED_BRANCH = "SAMPLED_RAY_BRANCH"


def _arrival_power(rec: ArrivalRecord) -> float:
    """What a crossing puts on a plane: a counted arrival what the plane receives (detected on an absorbing plane, the
    beam on a transparent one, nothing outside the acceptance cone); an uncounted crossing its Gaussian tail on the active
    area (the whole beam when that is not computed), nothing when the acceptance cone provably rejects that tail.

    The acceptance of a tail is not the chief ray's: the tail lies off the chief ray, where a diverging or converging beam
    travels along its tilted wavefront normal with its own angular spread; the tracer bounds those directions
    (``ArrivalRecord.tail_within_acceptance``) and only a tail all of whose light is outside the cone is dropped. A tail the
    bound cannot decide (it straddles the cone's edge, or the beam has no valid Gaussian state) still may arrive."""
    if rec.counted:
        if rec.detected_power_W is not None:
            return rec.detected_power_W
        return 0.0 if rec.within_acceptance is False else rec.power_W
    if rec.tail_within_acceptance is False:
        return 0.0
    return rec.power_W if rec.area_fraction is None else rec.area_fraction * rec.power_W


def _add(target: dict[str, float], kind: str, power: float) -> None:
    if power > 0.0:
        target[kind] = target.get(kind, 0.0) + power


@dataclass(frozen=True)
class Probes:
    """One probe pass: each followed untraced branch with its ``ProbeOutcome``, and the probes' own ``PATH_LIMIT`` flows
    (per source line) -- shared by the observation planes (``plane_reach``) and the surfaces (``surface_reach``).
    ``left_out_W``: per source line, the total power of the branches the probes left out as negligible
    (``TraceOutput.left_out_W``; in a deeper pass also the trace's own left-out branches at or below the pass's level), and
    ``left_out_root`` the sum of their square roots. ``levels``: a deeper pass's per-line levels (``follow_untraced``);
    ``budget_reached``: its budget left a branch unfollowed."""

    results: tuple[tuple[UntracedBranch, ProbeOutcome], ...] = ()
    path_limit_W: dict[LineKey, float] = field(default_factory=dict)
    left_out_W: dict[LineKey, float] = field(default_factory=dict)
    levels: dict[LineKey, float] = field(default_factory=dict)
    budget_reached: bool = False
    left_out_root: dict[LineKey, float] = field(default_factory=dict)
    exhausted: tuple[str, ...] = ()
    affected_path_ids: tuple[str, ...] = ()
    affected_source_ids: tuple[str, ...] = ()


def follow_untraced(trace: TraceOutput, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig,
                    lines: set[LineKey] | None = None, watch: frozenset[str] = frozenset(),
                    levels: dict[LineKey, float] | None = None) -> Probes:
    """Probe traces of the untraced branches of ``lines`` (every line when ``None``), strongest first over all roots;
    ``watch``: surfaces whose near misses the probes record too (``Tracer``). ``levels``: a deeper pass for the exposures
    over the lines it names -- their untraced branches and the branches the trace left out as negligible
    (``TraceOutput.faint_branches``) above the line's level, followed down to it (``Tracer(follow_W=levels)``); the left-out
    branches at or below it stay left out."""
    left: dict[LineKey, float] = {}
    root: dict[LineKey, float] = {}
    if levels is None:
        branches = [b for b in trace.untraced_branches
                    if b.seed is not None and (lines is None or (b.source_id, b.spectral_index) in lines)]
    else:
        branches = []
        for b in (*trace.untraced_branches, *trace.faint_branches):
            key = (b.source_id, b.spectral_index)
            if key not in levels or b.seed is None:
                continue
            if b.power_W > levels[key]:
                branches.append(b)
            else:
                left[key] = left.get(key, 0.0) + b.power_W
                root[key] = root.get(key, 0.0) + math.sqrt(b.power_W)
    if not branches:
        return Probes(left_out_W=left, levels=dict(levels or {}), left_out_root=root)
    probe = Tracer(scene, library, cfg, include_pupil=False, watch=watch, follow_W=levels, bounding=True)
    results = tuple(zip(branches, probe.trace_probes(branches)))
    limit: dict[LineKey, float] = {}
    for flow in probe.out.power_flows:
        if flow.kind == SINK_PATH_LIMIT:
            _add(limit, (flow.source_id, flow.spectral_index), flow.power_W)  # type: ignore[arg-type]
    for key, power in probe.out.left_out_W.items():
        left[key] = left.get(key, 0.0) + power
        root[key] = root.get(key, 0.0) + probe.out.left_out_root.get(key, math.sqrt(power))
    budget = bool(limit) or any(not outcome.traced or outcome.not_followed for _, outcome in results)
    return Probes(results, limit, left, dict(levels or {}), budget, root,
                  probe.schedule.exhausted, probe.schedule.affected_path_ids, probe.schedule.affected_source_ids)


def probe_budget_issues(probes: Probes, cfg: NumericalConfig,
                        lines: set[LineKey] | None = None) -> list[ValidationIssue]:
    """Report each probe pass's exhausted resources, including interaction-limited stored light, as notices
    (``NOT_APPLICABLE``): the probes are an auxiliary analysis, so their budget leaves the execution ``COMPLETED``
    (``validation/outcome.py``) and raises no status; the planes and surfaces whose bound it leaves open carry the fact.

    Auxiliary power is scoped to this pass; it must not be added to primary untraced power because the
    passes can follow the same physical light. A deeper exposure pass reports its own diagnostics too.
    """
    def relevant(line: LineKey) -> bool:
        return lines is None or line in lines

    selected = [(branch, outcome) for branch, outcome in probes.results
                if relevant((branch.source_id, branch.spectral_index))]
    skipped = sum(outcome.not_followed for _, outcome in selected)
    skipped_W = correctly_rounded_sum(outcome.not_followed_W for _, outcome in selected)
    budget_W = correctly_rounded_sum(w for line, w in probes.path_limit_W.items() if relevant(line)) + skipped_W
    issues: list[ValidationIssue] = []
    if budget_W > 0.0:
        # (a notice: an auxiliary analysis ran out of its budget; the planes and surfaces that lost a bound say so)
        issues.append(make_issue(
            ErrorCode.MAX_PATHS, Status.NOT_APPLICABLE,
            f"the trace budget for following untraced light (ghost reflections, coating transmissions, clipped light, "
            f"{cfg.max_paths} paths and {cfg.max_trace_steps} steps) was reached: {skipped} untraced branches and "
            f"{budget_W:.4g} W were not followed, so a plane no traced beam reaches is not known to be dark where that light "
            "could arrive", location="untraced_light", phase="untraced_light", budget=list(probes.exhausted), untraced_branches=skipped,
            affected_path_ids=list(probes.affected_path_ids), affected_source_ids=list(probes.affected_source_ids),
            untraced_power_W=budget_W, max_paths=cfg.max_paths, max_trace_steps=cfg.max_trace_steps,
            suggested_action="raise max_paths / max_trace_steps, or trace the reflections of the surfaces involved"))
    for _, outcome in selected:
        for flow in outcome.flows:
            if flow.kind != SINK_INTERACTION_LIMIT:
                continue
            issues.append(make_issue(
                ErrorCode.MAX_INTERACTIONS, Status.NOT_APPLICABLE,
                f"an auxiliary probe reached the per-path interaction limit ({cfg.max_interactions_per_path}); "
                f"its remaining {flow.power_W:.4g} W is not followed",
                location=f"path:{flow.path_id}" if flow.path_id else "untraced_light",
                phase="untraced_light", budget=["max_interactions_per_path"],
                affected_path_ids=[flow.path_id] if flow.path_id else [], affected_source_ids=[flow.source_id],
                limit=cfg.max_interactions_per_path, untraced_power_W=flow.power_W,
                suggested_action="raise max_interactions_per_path to follow more cavity round trips"))
    return issues


def _peak_root(rec: ArrivalRecord, limit_root: float | None, cfg: NumericalConfig) -> float:
    """The square root of the largest beam irradiance (W/mm^2) a crossing of a plane can have anywhere on it: its Gaussian
    peak ``2 P / (pi w_x w_y)``, or -- without a valid Gaussian state, or where its rays show aberrations the Gaussian beam
    does not describe (``validation/observations.py``) -- the diffraction limit of its power (``limit_root`` x sqrt(P);
    ``inf`` without one)."""
    from .observations import check_gaussian_aberration  # (observations imports this module)

    if rec.power_W <= 0.0:
        return 0.0
    (checked,), _ = check_gaussian_aberration([rec], cfg)
    g = checked.gstate
    if g.valid:
        return math.sqrt(2.0 * rec.power_W / (math.pi * g.radius("x") * g.radius("y")))
    return math.inf if limit_root is None else limit_root * math.sqrt(rec.power_W)


def plane_reach(trace: TraceOutput, scene: Scene, library: MaterialLibrary, counted: Iterable[ArrivalRecord],
                uncounted: Iterable[ArrivalRecord], observation_ids: Iterable[str],
                cfg: NumericalConfig, probes: Probes | None = None,
                gains: Mapping[float, ReturnGains] | None = None,
                limit: Callable[[str, LineKey], float] | None = None,
                followed: dict[str, list[FollowedLight]] | None = None) -> tuple[PlaneReach, list[ValidationIssue]]:
    """For every observation plane and every source line no counted beam of which reaches it: the power that may still
    reach it, per hazard kind (only kinds above the line's negligible level). A line absent from a plane's map is a known
    zero there as far as this light is concerned (the ledger's unknown-fate sinks are checked separately). Also returns a
    ``MAX_PATHS`` issue when the probes' trace budget left untraced light above the negligible level of such a line not
    followed.

    ``probes``: a probe pass over every line (``follow_untraced``; a run with a power maximum or power ratings). The map
    then also holds the lines whose counted beams reach a plane -- the untraced light of a line that may arrive beside
    its traced beams (a strong coating transmission folded back onto it, a ghost): it bounds the plane's power from
    above (``validation/metrics.py``), as does the light left out as negligible (``FAINT_POWER``, ``FAINT_ROOT``; ``gains``:
    the factors of the surfaces facing each other that may return it across a plane, ``validation/resonance.py``), and the
    most field it may add to a beam irradiance anywhere on the plane (``IRRADIANCE_ROOT``; ``limit``: (plane surface id, line) -> the
    square root of the diffraction limit of 1 W of that line there, W/mm^2). Without it only the lines dark somewhere are
    probed. Light of a probe at the interaction limit is never left out (it may be stored in a cavity).

    ``followed`` (a dict to fill): observation id -> the contributions of followed light on that plane, one per probe
    crossing, tail and partial crossing whose kind is above the line's negligible level there
    (``ObservationResult.followed_light``), in the order the reach adds them."""
    observation_ids = list(observation_ids)
    # (plane, line) -> its contributions as (kind, where it left the traced beam, traced path, power, how it meets the plane)
    listed: dict[tuple[str, LineKey], list[tuple[str, str | None, str | None, float, str]]] = {}

    def note(oid: str, line: LineKey, kind: str, surface_id: str | None, path_id: str | None, power: float, how: str) -> None:
        if followed is not None and power > 0.0:
            listed.setdefault((oid, line), []).append((kind, surface_id, path_id, power, how))

    inputs = {(i.source_id, i.spectral_index): i.power_W for i in trace.inputs}
    lit = {(rec.observation_id, rec.source_id, rec.spectral_index) for rec in counted}
    dark = {oid: [line for line in inputs if (oid, *line) not in lit] for oid in observation_ids}
    needed = {line for lines in dark.values() for line in lines}
    every = probes is not None
    relevant = set(inputs) if every else needed
    if probes is None:  # probe traces of the untraced branches of lines that are dark somewhere
        probes = follow_untraced(trace, scene, library, cfg, needed) if needed else Probes()
    arriving: dict[tuple[str, LineKey], dict[str, float]] = {}  # (plane, line) -> kind -> power arriving
    roots: dict[tuple[str, LineKey], dict[str, float]] = {}  # (plane, line) -> kind -> sum of sqrt(power) of its arrivals
    unfollowed: dict[LineKey, dict[str, float]] = {}  # line -> kind -> probe power of unknown fate (any plane)
    # (plane, line) -> the crossings of followed light (probes, traced tails) whose irradiance may add to a pair's there
    crossings: dict[tuple[str, LineKey], list[ArrivalRecord]] = {}
    # (plane, line) -> the powers of the partial crossings there (their irradiance is bounded by the diffraction limit only)
    partial: dict[tuple[str, LineKey], list[float]] = {}
    issues = probe_budget_issues(probes, cfg, relevant)
    if probes.results:
        for branch, outcome in probes.results:
            line = (branch.source_id, branch.spectral_index)
            # untraced light whose fate is not known is named by why (not by the kind of light it started as)
            if not outcome.traced:
                _add(unfollowed.setdefault(line, {}), PROBE_BUDGET, branch.power_W)
                continue
            _add(unfollowed.setdefault(line, {}), PROBE_BUDGET, outcome.not_followed_W)
            for rec in outcome.arrivals:
                _add(arriving.setdefault((rec.observation_id, line), {}), branch.kind, _arrival_power(rec))
                _add(roots.setdefault((rec.observation_id, line), {}), branch.kind, math.sqrt(_arrival_power(rec)))
                crossings.setdefault((rec.observation_id, line), []).append(rec)
                note(rec.observation_id, line, branch.kind, branch.surface_id, branch.path_id, _arrival_power(rec),
                     "ARRIVAL" if rec.counted else ACTIVE_AREA_TAIL)
            for part in outcome.partial_crossings:  # (part of the probe's beam crosses a plane its chief ray does not)
                _add(arriving.setdefault((part.observation_id, line), {}), branch.kind, part.power_W)
                _add(roots.setdefault((part.observation_id, line), {}), branch.kind, math.sqrt(part.power_W))
                partial.setdefault((part.observation_id, line), []).append(part.power_W)
                note(part.observation_id, line, branch.kind, branch.surface_id, branch.path_id, part.power_W, PARTIAL_CROSSING)
            for flow in outcome.flows:
                if flow.kind in _PROBE_UNFOLLOWED:
                    _add(unfollowed.setdefault(line, {}), _PROBE_FATE_KIND[flow.kind], flow.power_W)
    reached_left = surface_reach(trace, probes, cfg) if every else SurfaceReach()
    left, left_power = reached_left.left_out_root, reached_left.left_out_W
    for part in trace.partial_crossings:  # the traced beams' parts that cross a plane their chief rays do not
        key = (part.observation_id, (part.source_id, part.spectral_index))
        if part.diffracted and (part.observation_id, part.source_id, part.spectral_index) in lit:
            continue
        kind = DIFFRACTED_BEAM if part.diffracted else PARTIAL_CROSSING
        _add(arriving.setdefault(key, {}), kind, part.power_W)
        _add(roots.setdefault(key, {}), kind, math.sqrt(part.power_W))
        partial.setdefault(key, []).append(part.power_W)
        note(part.observation_id, key[1], kind, None, part.path_id, part.power_W, PARTIAL_CROSSING)
    for rec in uncounted:  # the traced beams' own tails beside an active area
        note(rec.observation_id, (rec.source_id, rec.spectral_index), ACTIVE_AREA_TAIL, None, rec.path_id, _arrival_power(rec),
             ACTIVE_AREA_TAIL)
        _add(arriving.setdefault((rec.observation_id, (rec.source_id, rec.spectral_index)), {}), ACTIVE_AREA_TAIL,
             _arrival_power(rec))
        _add(roots.setdefault((rec.observation_id, (rec.source_id, rec.spectral_index)), {}), ACTIVE_AREA_TAIL,
             math.sqrt(_arrival_power(rec)))
        crossings.setdefault((rec.observation_id, (rec.source_id, rec.spectral_index)), []).append(rec)
    planes = {s.interaction.observation_id: s for s in scene.surfaces if s.interaction.kind == "observe"}
    wavelengths = {(i.source_id, i.spectral_index): i.wavelength_nm for i in trace.inputs}
    reach: PlaneReach = {}
    for oid in observation_ids:
        per_line: dict[LineKey, dict[str, float]] = {}
        plane = planes.get(oid)
        absorbing = plane is not None and plane.interaction.absorbing
        for line in (list(inputs) if every else dark[oid]):
            negligible = cfg.negligible_power_fraction * inputs[line]
            kinds: dict[str, float] = {}
            for source in (arriving.get((oid, line), {}), unfollowed.get(line, {})):
                for kind, power in source.items():
                    _add(kinds, kind, power)
            kept = {kind: power for kind, power in kinds.items() if power > negligible or kind in STORED_KINDS}
            if followed is not None:
                followed.setdefault(oid, []).extend(
                    FollowedLight(kind=kind, source_id=line[0], spectral_index=line[1], surface_id=surface_id, path_id=path_id,
                                  power_W=power, reach=how)  # type: ignore[arg-type]
                    for kind, surface_id, path_id, power, how in listed.get((oid, line), ()) if kind in kept)
            root = correctly_rounded_sum(r for kind, r in roots.get((oid, line), {}).items() if kind in kept)
            # (the followed light below the negligible level and the left-out light still add fields and power)
            faint = faint_power = faint_once = irradiance = 0.0
            by_work = False
            causes: tuple[str, ...] = ()
            if every:
                left_root = left.get(line, 0.0)
                factor = power_factor = power_crossings = field_crossings = math.inf
                if left_root > 0.0:
                    # (the returns of the light left out at this wavelength: analysed only for a line that left some out)
                    found = gains.get(wavelengths[line]) if gains else None
                    factor = 1.0 if found is None or plane is None else found.factor(plane.surface_id)
                    # (how often its open paths may cross the plane -- once per straight leg after each number of
                    # reflections: not known without the scene's reflectors. An absorbing plane takes each watt once:
                    # its power reads neither, only its field does)
                    own = None if plane is None else plane.surface_id
                    if not absorbing:
                        power_factor = 1.0 if found is None or plane is None else found.power_factor(plane.surface_id)
                        power_crossings = math.inf if found is None else found.crossings(own)
                    field_crossings = math.inf if found is None else found.crossing_field(own)
                    by_work = found is not None and not found.screened
                    causes = () if found is None or plane is None else found.return_causes(plane.surface_id)
                faint = correctly_rounded_sum(r for kind, r in roots.get((oid, line), {}).items() if kind not in kept)
                spare = correctly_rounded_sum(p for kind, p in arriving.get((oid, line), {}).items() if kind not in kept)
                if left_root > 0.0:
                    # (its field: its pieces' amplitudes added, times the most field its open paths' crossings bring from
                    # either side and the factor of the pairs returning it; its power, the pieces' powers added as the
                    # plane's power adds its beams': each watt absorbed once on an absorbing plane, else crossing as often
                    # as its open paths and the pairs' returns let it)
                    faint += field_crossings * left_root * factor
                    left_W = left_power.get(line, 0.0)
                    faint_once = spare + left_W
                    spare += left_W * (1.0 if absorbing else power_crossings * power_factor)
                else:
                    faint_once = spare
                faint_power = spare
                # (the field it may add to a beam irradiance anywhere on the plane: each crossing's peak, and the left-out
                # light's diffraction limit with its returns)
                limit_root = None if limit is None or plane is None else limit(plane.surface_id, line)
                for rec in crossings.get((oid, line), ()):
                    irradiance += _peak_root(rec, limit_root, cfg)
                for power in partial.get((oid, line), ()):  # (no beam state on the plane: its power's diffraction limit)
                    irradiance += math.inf if limit_root is None else limit_root * math.sqrt(power)
                if left_root > 0.0:
                    irradiance += math.inf if limit_root is None else field_crossings * left_root * factor * limit_root
            if kept or faint > 0.0 or faint_power > 0.0 or irradiance > 0.0:
                per_line[line] = {**kept, **({ROOT_SUM: root} if root > 0.0 else {}), **({FAINT_ROOT: faint} if faint > 0.0 else {}),
                                  **({FAINT_POWER: faint_power, FAINT_ONCE: faint_once} if faint_power > 0.0 else {}),
                                  **({IRRADIANCE_ROOT: irradiance} if irradiance > 0.0 else {}),
                                  **({WITHHELD_BY_WORK: 1.0} if by_work else {}),
                                  **{RETURN_CAUSE + cause: 1.0 for cause in causes}}
        if per_line:
            reach[oid] = per_line
    return reach, issues


def partial_surfaces(trace: TraceOutput) -> dict[str, tuple[str | None, ...]]:
    """Per observation plane, the surfaces part of a traced beam reaches it before (``PartialCrossing.ends_at``; ``None``: on
    its way out of the scene), in trace order."""
    found: dict[str, list[str | None]] = {}
    for part in trace.partial_crossings:
        seen = found.setdefault(part.observation_id, [])
        if part.ends_at not in seen:
            seen.append(part.ends_at)
    return {oid: tuple(surfaces) for oid, surfaces in found.items()}


def hazards(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> dict[str, float]:
    """The power per hazard kind that may reach ``observation_id`` from ``lines`` (summed over them)."""
    out: dict[str, float] = {}
    if not reach or observation_id is None or observation_id not in reach:
        return out
    per_line = reach[observation_id]
    for line in lines:
        for kind, power in per_line.get(line, {}).items():
            if kind not in _NOT_KINDS and not kind.startswith(RETURN_CAUSE):
                out[kind] = out.get(kind, 0.0) + power
    return out


def hazard_root_sum(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> float:
    """Sum over the separate contributions of followed untraced light that reach ``observation_id`` from ``lines`` of the
    square root of each one's power (sqrt(W)): the largest summed field amplitude they can bring, however they interfere."""
    if not reach or observation_id is None or observation_id not in reach:
        return 0.0
    return correctly_rounded_sum(reach[observation_id].get(line, {}).get(ROOT_SUM, 0.0) for line in lines)


def faint_root_sum(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> float:
    """The largest field amplitude (sqrt(W)) light of ``lines`` the plane's power does not count may add at
    ``observation_id`` (``FAINT_ROOT``): the coherent power's bound reads it."""
    if not reach or observation_id is None or observation_id not in reach:
        return 0.0
    return correctly_rounded_sum(reach[observation_id].get(line, {}).get(FAINT_ROOT, 0.0) for line in lines)


def irradiance_root_sum(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> float:
    """The largest field amplitude, as the square root of a beam irradiance (sqrt(W/mm^2)), that light of ``lines`` the
    trace does not follow may add anywhere on ``observation_id`` (``IRRADIANCE_ROOT``): the pair metrics' irradiances
    read it."""
    if not reach or observation_id is None or observation_id not in reach:
        return 0.0
    return correctly_rounded_sum(reach[observation_id].get(line, {}).get(IRRADIANCE_ROOT, 0.0) for line in lines)


def faint_once_sum(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> float:
    """The most power (W) light of ``lines`` the plane's power does not count may bring to a detector placed at ``observation_id``, each watt once (``FAINT_ONCE``)."""
    if not reach or observation_id is None or observation_id not in reach:
        return 0.0
    return correctly_rounded_sum(reach[observation_id].get(line, {}).get(FAINT_ONCE, 0.0) for line in lines)


def return_causes(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> tuple[str, ...]:
    """Why nothing bounds the left-out light of ``lines`` across ``observation_id`` (``RETURN_CAUSE``), each clause once, in
    order."""
    if not reach or observation_id is None or observation_id not in reach:
        return ()
    found = {kind[len(RETURN_CAUSE):] for line in lines for kind in reach[observation_id].get(line, {})
             if kind.startswith(RETURN_CAUSE)}
    return tuple(sorted(found))


def withheld_by_work(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> bool:
    """Whether the left-out light of ``lines`` is unbounded at ``observation_id`` because the return-light analysis reached its
    work limit (``WITHHELD_BY_WORK``)."""
    if not reach or observation_id is None or observation_id not in reach:
        return False
    return any(WITHHELD_BY_WORK in reach[observation_id].get(line, {}) for line in lines)


def faint_power_sum(reach: PlaneReach | None, observation_id: str | None, lines: Iterable[LineKey]) -> float:
    """The most power (W) light of ``lines`` the plane's power does not count may add at ``observation_id`` (``FAINT_POWER``;
    ``inf``: unbounded)."""
    if not reach or observation_id is None or observation_id not in reach:
        return 0.0
    return correctly_rounded_sum(reach[observation_id].get(line, {}).get(FAINT_POWER, 0.0) for line in lines)


@dataclass(frozen=True)
class SurfaceReach:
    """Light the trace does not follow, followed to the physical surfaces (the exposures of a run with power ratings).

    ``hits``: (kind of the untraced branch it started as, the probe's arrival at a physical surface), every arrival of
    the probes whatever its power; ``unfollowed``: per source line, untraced light whose fate its probe does not know (it
    may reach any surface), per kind (``PROBE_*``), above the negligible level, and ``unfollowed_root`` the sum of the square
    roots of its pieces' powers per kind. ``left_out_W``: per source line, the total power of the light left out as
    negligible -- the branches the trace and the probes did not follow, light of unknown fate at or below the negligible
    level, the paths ended at the power floor (the trace's and the probes') -- which may still reach any surface
    (``validation/exposures.py`` bounds it by its diffraction limit); ``left_out_root`` the sum of the square roots of
    those pieces' powers (sqrt(W)): mutually coherent, they may meet in phase, so their field is bounded by this sum, not
    by the square root of the total. ``levels_W``: per source line, the power down to which its untraced light was
    followed (the negligible level, ``negligible_power_fraction`` x the line input, or a deeper pass's level)."""

    hits: tuple[tuple[str, SurfaceHit], ...] = ()
    unfollowed: dict[LineKey, dict[str, float]] = field(default_factory=dict)
    left_out_W: dict[LineKey, float] = field(default_factory=dict)
    levels_W: dict[LineKey, float] = field(default_factory=dict)
    left_out_root: dict[LineKey, float] = field(default_factory=dict)
    unfollowed_root: dict[LineKey, dict[str, float]] = field(default_factory=dict)

    def anywhere(self) -> dict[str, float]:
        """Untraced power of unknown fate (any line), per kind: it may reach every surface."""
        out: dict[str, float] = {}
        for kinds in self.unfollowed.values():
            for kind, power in kinds.items():
                _add(out, kind, power)
        return out


def surface_reach(trace: TraceOutput, probes: Probes, cfg: NumericalConfig) -> SurfaceReach:
    """Where the untraced light of a probe pass over every line (``follow_untraced``) meets physical surfaces. Every
    arrival of a probe is kept, whatever its power (a faint beam focused onto a surface may still matter there). Light the
    probes and the trace left out as negligible (below the line's level: ``negligible_power_fraction`` x the line input, or
    a deeper pass's level for the lines it names, ``Probes.levels``) and light that stopped at the power floor are added
    up per line in ``left_out_W``. Light the probes' budget does not reach, and light of unknown fate above the negligible
    level, is unfollowed (``PROBE_*``) and may reach any surface in any amount; light of unknown fate at or below the
    negligible level is left out. No issue is raised: an exposure the probes cannot bound stays unknown in the exposure
    report."""
    inputs = {(i.source_id, i.spectral_index): i.power_W for i in trace.inputs}
    levels = probes.levels
    left_out: dict[LineKey, float] = {}
    left_root: dict[LineKey, float] = {}

    def leave_out(line: LineKey, power: float, root: float | None = None) -> None:
        """``power`` left out as negligible, in pieces whose square roots sum to ``root`` (one piece: its square root)."""
        if power > 0.0:
            left_out[line] = left_out.get(line, 0.0) + power
            left_root[line] = left_root.get(line, 0.0) + (math.sqrt(power) if root is None else root)

    # (a deeper pass's lines: the trace's left-out branches are its roots or among its left-out light)
    for line, power in trace.left_out_W.items():
        if line not in levels:
            leave_out(line, power, trace.left_out_root.get(line))
    for line, power in probes.left_out_W.items():
        leave_out(line, power, probes.left_out_root.get(line))
    for flow in trace.power_flows:  # traced paths ended at the power floor
        if flow.kind == SINK_BELOW_POWER_FLOOR:
            leave_out((flow.source_id, flow.spectral_index), flow.power_W)  # type: ignore[arg-type]
    # the ledger's own unknown-fate power of a line at or below its negligible level (``left_out_ledger_W``): left out too,
    # piece by piece, so every bound that reads the left-out light covers it
    ledger: dict[LineKey, dict[str, float]] = {}
    for flow in trace.power_flows:
        if flow.kind in UNKNOWN_FATE_SINKS and flow.kind not in NEVER_LEFT_OUT_SINKS:
            _add(ledger.setdefault((flow.source_id, flow.spectral_index), {}), flow.kind, flow.power_W)  # type: ignore[arg-type]
    faint_lines = {line for line, sinks in ledger.items() if left_out_ledger_W(sinks, inputs.get(line, 0.0), cfg) > 0.0}
    for flow in trace.power_flows:
        if flow.kind in UNKNOWN_FATE_SINKS and flow.kind not in NEVER_LEFT_OUT_SINKS and (flow.source_id, flow.spectral_index) in faint_lines:
            leave_out((flow.source_id, flow.spectral_index), flow.power_W)  # type: ignore[arg-type]
    levels_W = {line: levels.get(line, cfg.negligible_power_fraction * power) for line, power in inputs.items()}
    if not probes.results:
        return SurfaceReach(left_out_W=left_out, levels_W=levels_W, left_out_root=left_root)
    hits: list[tuple[str, SurfaceHit]] = []
    unfollowed: dict[LineKey, dict[str, float]] = {}
    roots: dict[LineKey, dict[str, float]] = {}  # (the sum of the square roots of each kind's pieces)
    for branch, outcome in probes.results:
        line = (branch.source_id, branch.spectral_index)
        if not outcome.traced:
            _add(unfollowed.setdefault(line, {}), PROBE_BUDGET, branch.power_W)
            _add(roots.setdefault(line, {}), PROBE_BUDGET, math.sqrt(branch.power_W))
            continue
        _add(unfollowed.setdefault(line, {}), PROBE_BUDGET, outcome.not_followed_W)
        # (the branches the budget did not reach, n of them: their square roots sum to at most sqrt(n x their power))
        _add(roots.setdefault(line, {}), PROBE_BUDGET, math.sqrt(outcome.not_followed * outcome.not_followed_W))
        for hit in outcome.surface_hits:
            hits.append((branch.kind, hit))
        for flow in outcome.flows:
            if flow.kind == SINK_BELOW_POWER_FLOOR:  # (its power is known: left out like the trace's own)
                leave_out(line, flow.power_W)
            elif flow.kind in _PROBE_UNFOLLOWED:
                _add(unfollowed.setdefault(line, {}), _PROBE_FATE_KIND[flow.kind], flow.power_W)
                _add(roots.setdefault(line, {}), _PROBE_FATE_KIND[flow.kind], math.sqrt(flow.power_W))
    kept: dict[LineKey, dict[str, float]] = {}
    kept_roots: dict[LineKey, dict[str, float]] = {}
    for line, kinds in unfollowed.items():
        negligible = cfg.negligible_power_fraction * inputs.get(line, 0.0)
        # (light at the interaction limit may be stored in a cavity: never left out, whatever its power)
        above = {kind: power for kind, power in kinds.items() if power > negligible or kind in STORED_KINDS}
        if above:
            kept[line] = above
            kept_roots[line] = {kind: roots.get(line, {}).get(kind, math.sqrt(power)) for kind, power in above.items()}
        for kind, power in kinds.items():
            if kind not in above:
                leave_out(line, power, roots.get(line, {}).get(kind))
    return SurfaceReach(tuple(hits), kept, left_out, levels_W, left_root, kept_roots)


def merge_reach(first: SurfaceReach, deeper: SurfaceReach, lines: set[LineKey]) -> SurfaceReach:
    """``first`` with the light of ``lines`` replaced by a deeper pass's (``follow_untraced(levels=...)``)."""
    def line_of(hit: SurfaceHit) -> LineKey:
        return (hit.source_id, hit.spectral_index)

    hits = tuple(h for h in first.hits if line_of(h[1]) not in lines) + tuple(h for h in deeper.hits if line_of(h[1]) in lines)
    unfollowed = {**{k: v for k, v in first.unfollowed.items() if k not in lines},
                  **{k: v for k, v in deeper.unfollowed.items() if k in lines}}
    left_out = {**{k: v for k, v in first.left_out_W.items() if k not in lines},
                **{k: v for k, v in deeper.left_out_W.items() if k in lines}}
    levels = {**first.levels_W, **{k: v for k, v in deeper.levels_W.items() if k in lines}}
    left_root = {**{k: v for k, v in first.left_out_root.items() if k not in lines},
                 **{k: v for k, v in deeper.left_out_root.items() if k in lines}}
    unfollowed_root = {**{k: v for k, v in first.unfollowed_root.items() if k not in lines},
                       **{k: v for k, v in deeper.unfollowed_root.items() if k in lines}}
    return SurfaceReach(hits, unfollowed, left_out, levels, left_root, unfollowed_root)
