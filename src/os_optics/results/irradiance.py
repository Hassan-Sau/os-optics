"""Irradiance map of the beams arriving at an observation plane, from the paraxial Gaussian superposition model.

``irradiance_map`` samples, on a square grid of plane coordinates ``(u, v)``, the power flux through the plane per unit
area (W/mm^2) of every beam of one wavelength that counts as arriving at the plane: the sum of their individual
irradiances (incoherent) and the irradiance of their coherent sum (``validation.superposition.irradiance``: paraxial
Gaussian envelopes with plane-wave phases, mutual coherence ``exp(-(OPD / L_c)^2)`` within a coherence group). It is the
Gaussian-beam model on the whole plane (not limited to the active area), without wavefront curvature across the plane,
Gouy phase or diffraction: not a diffracted field.

Sampling: the grid is centred on the mean of the beams' chief-ray hit points (``grid.center_uv_mm``). Two mutually coherent
beams crossing at an angle make fringes of in-plane period ``2 pi / |kappa_a - kappa_b|`` (``kappa = k0 n (d.u, d.v)``); the
coherent map is returned only when the grid step resolves the shortest such period with at least
``FRINGE_SAMPLES_PER_PERIOD`` samples. A call that states no ``samples`` takes ``DEFAULT_SAMPLES`` points per axis, or, where
the beams make fringes that count does not resolve and a count up to ``MAX_SAMPLES`` does, the smallest count that resolves
them (``sampling.samples_basis``: ``DEFAULT``, ``FRINGE_PERIOD`` or ``STATED``; state ``samples`` to fix the count). Where no
allowed count resolves the period, or the stated one does not, the coherent map is withheld (``coherent_reason``
``FRINGES_UNDERSAMPLED``, with the period, the largest usable step and the half width that would resolve it at this sample
count): a coarser grid would show moire patterns that do not exist. The incoherent map is always returned.

Several wavelengths. Without ``wavelength_nm`` on a plane several wavelengths reach, the map is their sum on one grid (lines of
different wavelengths are mutually incoherent: their irradiances add): ``wavelength_nm`` and ``scope.wavelength_nm`` are null,
``wavelengths_nm`` lists them, ``per_wavelength`` gives each line's beams, grid powers, ``coherent_reason``, ``map_complete``
and untraced light, and the coherent map is the sum of the lines' coherent maps when every line has one (else null, with the
first line's reason). The grid is the one a map of all the usable light takes; a line no map exists for (its reason is in its
``per_wavelength`` row) leaves the sum incomplete. With ``wavelength_nm`` the map is that line's alone.

Array layout (JSON and NPZ): 2-D arrays are indexed ``[v_index, u_index]`` (one row per ``v`` value, ``grid.v_mm``; columns
follow ``grid.u_mm``).

Sampled rays: where the rays of an extended emitter cross the plane (``ObservationResult.sampled_light``), the map is a
histogram: ``samples`` square bins per axis (``grid.u_edges_mm`` / ``v_edges_mm``, ``grid.u_mm`` / ``v_mm`` their centres), each
bin holding the summed power of the crossings in it divided by the bin's area, in W/mm^2 of the plane, and ``sample_counts`` how
many crossings stand behind it. A crossing counts wherever it lands on the plane, as a beam's map holds the whole plane. Rays are
independent, so the sum is incoherent and no coherent map exists (``coherent_reason`` ``INDEPENDENT_SAMPLES``). Gaussian beams of
the same wavelength on the plane add at the bin centres. ``power_outside_grid_W`` is the crossing power beyond the grid; the grid
power plus it is the power of all the crossings. Default grid: the plane's active area when it has one, else centred on the
crossings' power centroid with half width holding ``DEFAULT_ENCLOSED_FRACTION`` of their power; ``samples`` by default gives
``DEFAULT_SAMPLES_PER_BIN`` crossings per bin of a uniform map. ``center_uv_mm`` places the grid.

Uniformity: ``roi`` names a region of the plane (``{"shape": "rectangle", "center_uv_mm": [u, v], "half_width_mm": [a, b]}`` or
``{"shape": "circle", "center_uv_mm": [u, v], "radius_mm": r}``). ``uniformity`` holds, over the grid cells whose centres lie in it,
``min_over_mean`` (smallest / mean) and ``std_over_mean`` (population standard deviation / mean) of the incoherent map and, for
sampled rays, the same two from the first half of the sequence: the distance between the two pairs shows how much of the spread
is sampling noise (a bin of ``n`` crossings has a noise of at most about ``1 / sqrt(n)``).

What a map holds. ``scope`` states its wavelength and source (null: every source), and ``outside_scope`` the other light on the plane, each with its known power and upper bound.
The maps count the light known to arrive. ``untraced_power_W`` is the most power of the scope that is not in them and may reach the plane (null: nothing bounds it), and
``untraced_fraction`` that power over the known power, and ``map_complete`` says it is zero. It holds:

* beams the map cannot hold (``excluded_beams``, below);
* beams of unknown power (a grating or a deflector with no stated efficiency), which have a place on the plane and no irradiance: ``omitted_beams`` lists each with its ``uv_mm``,
  ``direction_uvn`` and ``power_upper_bound_W``. A beam outside the plane's acceptance cone cannot reach the map and does not count, where its power is not also shared with an order that was not followed;
* the plane's own statement of light the trace does not follow (``power_by_wavelength``): an order not followed, a ghost, light left out as negligible. A map of one source counts that source's
  ledger, and the light no ledger names;
* ``omitted_sampled_power_W``: the power of a sampled line that went on with no ray.

``interference_open``: that light may interfere with a beam in the map (it shares a source or a coherence group), so it may darken the map as well. Above the call's
``max_untraced_fraction`` (default ``MAX_UNTRACED_FRACTION``, ``results.quantity``; ``None``: no limit; the document states the one applied) such a coherent map is withheld
(``coherent_reason`` ``BEAM_POWER_UNKNOWN``, or ``UNTRACED_POWER_TOO_LARGE`` where no beam or order of unknown power stands behind it); at or below it the
map is stated with the size of what may move it. A uniformity states ``untraced_fraction``, ``roi_untraced_fraction`` (that power over the known power in the region) and
``relative_error_bound``, how far the region's power may move: ``f``, or ``(1 + sqrt(f))^2 - 1`` where the light may interfere (the bound holds at any ``f``); above the limit it reads ``null`` with the
``reason``, ``untraced_fraction`` and ``max_untraced_fraction``. A plane reached only by beams of unknown power gives ``BEAM_POWER_UNKNOWN``, with the beams.

A plane whose bounds an evaluation did not compute (its ``quantity_state`` ``NOT_COMPUTED``: ``plane_bounds`` did not run) has the
incoherent map of its known light; ``untraced_power_W`` and ``untraced_fraction`` are null, the coherent map and a uniformity read
``NOT_COMPUTED`` (``coherent_reason`` / ``reason``), and ``needs`` names ``plane_bounds``, the capability that gives them.

Beams a map cannot hold. Three kinds of beam of the map's scope have no profile or power to add, and are left out of it and listed in
``excluded_beams``, one row each with ``path_id``, ``crossing_index``, ``uv_mm``, the ``reason``, the ``surfaces`` behind it and what is known of the power:

* ``POWER_OUT_OF_MODEL_SCOPE``: the beam's power is out of model scope (``ArrivalResult.power_out_of_scope_at``: behind a crystal whose second-harmonic
  generation is out of model scope there, a variable attenuator or a transmission map that varies across the beam, the side of a gradient-index rod, an
  absorbing plane or element part of the beam meets beside its chief ray): it carries the chief ray's power, not the light's (no power stated). Sampled
  light scattered from such a beam counts the same;
* ``POWER_TAKEN_BEFORE_PLANE``: surfaces take part of the beam before the plane (``ArrivalResult.power_taken_before_at``), and where is not known: its power
  lies between ``power_lower_bound_W`` and ``power_upper_bound_W``;
* ``GAUSSIAN_NOT_AVAILABLE``: the beam has a power (``power_W``) and no valid Gaussian state, so no profile.

The map of the other beams and lines stays: ``map_complete`` is false, the most the excluded beams carry adds to ``untraced_power_W`` (null when one of them has
no bound), and one that shares a coherence group with a beam in the map opens ``interference_open``. Where no beam or line of the scope is left, no map
is stated: ``POWER_OUT_OF_MODEL_SCOPE`` (with ``power_out_of_scope_at`` and the beams' ``path_ids``, as the plane's own power is ``OUT_OF_MODEL_SCOPE``),
``POWER_TAKEN_BEFORE_PLANE`` (with ``power_taken_before_at``) or ``GAUSSIAN_NOT_AVAILABLE``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import NumericalConfig
from ..units import NM_TO_MM
from ..contracts.results import ArrivalResult, ObservationPlaneFrame, ValidationResult
from ..errors import Status
from ..optics.rays.tracer import SINK_UNMODELLED_RESPONSE, UNKNOWN_FATE_SINKS
from ..optics.interference.source_coherence import SourceCoherence
from ..validation.superposition import assumed_coherence, irradiance, same_direction
from ..math.vectors import dot
from ..errors.status import QuantityState
from .quantity import (BEAM_POWER_UNKNOWN, MAX_UNTRACED_FRACTION, NOT_COMPUTED, POWER_OUT_OF_MODEL_SCOPE, POWER_TAKEN_BEFORE_PLANE,
                       UNTRACED_POWER_TOO_LARGE,
                       power_scope_message, taken_before_message, taken_before_surfaces,
                       power_scope_surfaces)

# Default half width of the grid: the farthest beam centre plus this many projected 1/e^2 radii.
EXTENT_BEAM_RADII = 2.5
DEFAULT_SAMPLES = 101
MIN_SAMPLES = 2
MAX_SAMPLES = 1001
# How the grid's point count was chosen (``sampling.samples_basis``): the default count, the count that resolves the beams'
# fringes, or the caller's.
SAMPLES_DEFAULT = "DEFAULT"
SAMPLES_FRINGE_PERIOD = "FRINGE_PERIOD"
SAMPLES_STATED = "STATED"
# A beam travelling nearly in the plane has an unbounded footprint: projected radii use at least this cosine.
MIN_PROJECTION_COS = 1e-6

IRRADIANCE_MODEL = (
    "Paraxial Gaussian beams with plane-wave phases on the observation plane (the coherent superposition model of the "
    "observation's superpositions): power flux through the plane per unit area, whole plane, no wavefront curvature across "
    "the plane, no Gouy phase, no diffraction."
)
WAVELENGTH_REQUIRED = "WAVELENGTH_REQUIRED"
NO_ARRIVALS = "NO_ARRIVALS"
GAUSSIAN_NOT_AVAILABLE = "GAUSSIAN_NOT_AVAILABLE"
GRID_INVALID = "GRID_INVALID"
COHERENCE_UNDEFINED = "COHERENCE_UNDEFINED"
INDEPENDENT_SAMPLES = "INDEPENDENT_SAMPLES"
ROI_INVALID = "ROI_INVALID"
LIMIT_INVALID = "LIMIT_INVALID"
FRINGES_UNDERSAMPLED = "FRINGES_UNDERSAMPLED"
# Samples per shortest fringe period a coherent map needs (twice the Nyquist minimum, so the fringe contrast is visible).
FRINGE_SAMPLES_PER_PERIOD = 4


SAMPLED_IRRADIANCE_MODEL = (
    "Histogram of independent sampled rays: power crossing each bin per unit area of the plane, incoherent sum, whole plane; a bin "
    "of n crossings has a sampling noise of at most about 1 / sqrt(n). Gaussian beams on the plane add at the bin centres."
)
# Half width of a sampled map's default grid: holds this share of the crossings' power, measured from their power centroid.
DEFAULT_ENCLOSED_FRACTION = 0.95
# A sampled map's default bins per axis give about this many crossings per bin of a uniform map.
DEFAULT_SAMPLES_PER_BIN = 16
ROI_RECTANGLE = "rectangle"
ROI_CIRCLE = "circle"


def shortest_fringe_period_mm(members: list[ArrivalResult], plane: ObservationPlaneFrame) -> float | None:
    """The shortest in-plane fringe period (mm) between two mutually coherent beams of ``members`` (None: no fringes)."""
    u, v = np.array(plane.u_axis), np.array(plane.v_axis)
    shortest = None
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if a.coherence_group is None or a.coherence_group != b.coherence_group or not same_direction(a, b, np.array(plane.normal)):
                continue
            g = a.beam.gaussian
            n = g.medium_index if g is not None and g.valid and g.medium_index else 1.0
            k0 = 2.0 * math.pi / (a.beam.wavelength_nm * NM_TO_MM)
            dk = k0 * n * np.array([dot(a.beam.direction, u) - dot(b.beam.direction, u),
                                    dot(a.beam.direction, v) - dot(b.beam.direction, v)])
            norm = float(np.sqrt(dot(dk, dk)))
            if norm > 0.0:
                period = 2.0 * math.pi / norm
                shortest = period if shortest is None else min(shortest, period)
    return shortest


def _groups(arrivals: tuple[ArrivalResult, ...], tol_nm: float) -> list[list[ArrivalResult]]:
    groups: list[list[ArrivalResult]] = []
    for arrival in sorted(arrivals, key=lambda a: a.beam.wavelength_nm):
        if groups and abs(groups[-1][0].beam.wavelength_nm - arrival.beam.wavelength_nm) <= tol_nm:
            groups[-1].append(arrival)
        else:
            groups.append([arrival])
    return groups


def _refusal(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"status": "UNAVAILABLE", "reason": code, "message": message, **context}


def _omitted_beams(beams: list[ArrivalResult]) -> list[dict[str, Any]]:
    """Where each beam of unknown power reaches the plane, and the most it can carry."""
    return [{"path_id": a.beam.path_id, "crossing_index": a.crossing_index, "uv_mm": list(a.uv_mm), "direction_uvn": None if a.direction_uvn is None else list(a.direction_uvn),
             "power_upper_bound_W": a.power_upper_bound_W} for a in beams]


def _sets_order_aside(result: ValidationResult, source_id: str, wavelength_nm: float, tol: float) -> bool:
    """A grating sent part of this line's light into a propagating order the run does not follow: where it goes is not known, whatever the followed orders do."""
    return any(row.source_id == source_id and abs(row.wavelength_nm - wavelength_nm) <= tol and row.propagation == "PROPAGATING" and not row.followed
               for row in result.grating_orders)


def _untraced_light(result: ValidationResult, observation: Any, wavelength_nm: float, tol: float, source_id: str | None) -> tuple[float, frozenset[str], bool]:
    """The most power of this wavelength (and source) that may reach the plane and no beam of known power brings, from the plane's own statement and the ledger: ``(power, the sources whose
    ledger holds some of it, some of it belongs to no ledger)``; ``inf``: nothing bounds it."""
    spectral = [s for s in observation.power_by_wavelength if abs(s.wavelength_nm - wavelength_nm) <= tol and not s.complete]
    if not spectral:
        return 0.0, frozenset(), False
    unresolved = frozenset(c.source_id for c in result.nonlinear_conversions if c.status is Status.UNKNOWN
                           and abs(c.harmonic_wavelength_nm - wavelength_nm) <= tol
                           and (source_id is None or c.source_id == source_id))
    if unresolved:
        return math.inf, unresolved, False
    # (light left out as negligible counts once; where other light may go on crossing a plane that does not absorb, no bound exists)
    upper = observation.power_upper_bound_W if observation.power_upper_bound_W is not None else observation.absorbed_power_upper_bound_W
    if upper is None:
        return math.inf, frozenset(a.beam.source_id for a in observation.arrivals if a.beam.power_W is None), True
    stated = math.fsum(s.untraced_power_W for s in spectral if s.untraced_power_W is not None)
    # (what the plane adds to its total beyond the untraced power: light left out as negligible, which it states for all its wavelengths together)
    faint = max(upper - (observation.total_power_W or 0.0) - observation.untraced_power_W, 0.0)
    per_source: dict[str, float] = {}
    ledgered = 0.0
    for line in (result.power_budget.lines if result.power_budget is not None else ()):
        if abs(line.wavelength_nm - wavelength_nm) > tol:
            continue
        held = math.fsum(line.sinks.get(sink, 0.0) for sink in UNKNOWN_FATE_SINKS)
        ledgered += held
        beams = [a for a in observation.arrivals if a.beam.source_id == line.source_id and abs(a.beam.wavelength_nm - wavelength_nm) <= tol and a.beam.power_W is None]
        outside = math.fsum(a.power_upper_bound_W or 0.0 for a in beams if a.within_acceptance is False)
        if outside > 0.0 and not _sets_order_aside(result, line.source_id, wavelength_nm, tol):
            # (a beam outside the cone cannot reach the map, its bound shares the ledger's power with the other beams of that response)
            inside = max((a.power_upper_bound_W or 0.0 for a in beams if a.within_acceptance is not False), default=0.0)
            held = max(held - min(outside, line.sinks.get(SINK_UNMODELLED_RESPONSE, 0.0)), inside)
        if held > 0.0:
            per_source[line.source_id] = per_source.get(line.source_id, 0.0) + held
    unnamed = max(stated - ledgered, 0.0) + faint
    held = {s: w for s, w in per_source.items() if source_id is None or s == source_id}
    return math.fsum(held.values()) + unnamed, frozenset(held), unnamed > 0.0


def _outside_scope(result: ValidationResult, observation: Any, wavelength_nm: float, tol: float, source_id: str | None) -> list[dict[str, Any]]:
    """The other light on the plane: other wavelengths, and other sources of this wavelength for a map of one source."""
    out = []
    for spectral in observation.power_by_wavelength:
        if abs(spectral.wavelength_nm - wavelength_nm) <= tol:
            continue
        open_W = _untraced_light(result, observation, spectral.wavelength_nm, tol, None)[0]
        known = spectral.power_W or 0.0
        if known > 0.0 or open_W > 0.0:
            out.append({"wavelength_nm": spectral.wavelength_nm, "sources": None, "known_power_W": known, "power_upper_bound_W": known + open_W if math.isfinite(open_W) else None})
    if source_id is not None:
        others = {a.beam.source_id for a in observation.arrivals if a.beam.source_id != source_id and abs(a.beam.wavelength_nm - wavelength_nm) <= tol}
        others |= {line.source_id for line in observation.sampled_light
                   if line.crossing_count and line.source_id != source_id and abs(line.wavelength_nm - wavelength_nm) <= tol}
        known = math.fsum(a.beam.power_W for a in observation.arrivals if a.beam.source_id in others and a.beam.power_W is not None and a.within_acceptance is not False)
        known += math.fsum(line.power_W for line in observation.sampled_light if line.source_id in others)
        everything, own = _untraced_light(result, observation, wavelength_nm, tol, None)[0], _untraced_light(result, observation, wavelength_nm, tol, source_id)[0]
        open_W = max(everything - own, 0.0) if math.isfinite(everything) else everything
        if others and (known > 0.0 or open_W > 0.0):
            out.append({"wavelength_nm": wavelength_nm, "sources": sorted(others), "known_power_W": known, "power_upper_bound_W": known + open_W if math.isfinite(open_W) else None})
    return out


def _usable(arrival: ArrivalResult, plane: ObservationPlaneFrame, cfg: NumericalConfig) -> bool:
    """A beam of known power has an irradiance profile on the plane: a valid Gaussian state with both radii."""
    g = arrival.beam.gaussian
    if g is None or not g.valid or not g.beam_radius_x_mm or not g.beam_radius_y_mm:
        return False
    return irradiance([arrival], plane, np.array([[float(arrival.uv_mm[0]), float(arrival.uv_mm[1])]]), cfg) is not None


def _excluded_row(arrival: ArrivalResult, reason: str, surfaces: Any, power_W: float | None, upper_W: float | None, lower_W: float | None) -> dict[str, Any]:
    return {"path_id": arrival.beam.path_id, "crossing_index": arrival.crossing_index, "uv_mm": list(arrival.uv_mm), "reason": reason,
            "surfaces": list(surfaces), "power_W": power_W, "power_upper_bound_W": upper_W, "power_lower_bound_W": lower_W}


def _excluded_beams(scoped: list[ArrivalResult], taken: list[ArrivalResult], shapeless: list[ArrivalResult], scoped_lines: list[Any]) -> list[dict[str, Any]]:
    """One row per beam (and per sampled line) of the scope the map leaves out because no profile or power of it can be stated,
    with the reason, the surfaces and what is known of its power (module docstring)."""
    rows = [_excluded_row(a, POWER_OUT_OF_MODEL_SCOPE, a.power_out_of_scope_at, None, None, None) for a in scoped]
    for a in taken:
        power = a.beam.power_W
        if power is None:
            rows.append(_excluded_row(a, POWER_TAKEN_BEFORE_PLANE, a.power_taken_before_at, None, a.power_upper_bound_W, None))
            continue
        rows.append(_excluded_row(a, POWER_TAKEN_BEFORE_PLANE, a.power_taken_before_at, None, max(power - (a.power_taken_before_lower_W or 0.0), 0.0),
                                  max(power - (a.power_taken_before_upper_W or 0.0), 0.0)))
    rows.extend(_excluded_row(a, GAUSSIAN_NOT_AVAILABLE, (), a.beam.power_W, a.beam.power_W, a.beam.power_W) for a in shapeless)
    rows.extend({"path_id": line.parent_path_id or line.source_id, "crossing_index": None, "uv_mm": None, "reason": POWER_OUT_OF_MODEL_SCOPE,
                 "surfaces": list(line.power_out_of_scope_at), "power_W": None, "power_upper_bound_W": None, "power_lower_bound_W": None}
                for line in scoped_lines)
    return rows


def _openness(result: ValidationResult, observation: Any, chosen: float, tol: float, source_id: str | None, members: list[ArrivalResult], left_out: list[ArrivalResult],
              lines: list[Any], excluded: list[dict[str, Any]] | None = None, excluded_arrivals: list[ArrivalResult] | None = None,
              limit: float | None = MAX_UNTRACED_FRACTION) -> dict[str, Any]:
    """What the map document says of the light it does not hold (module docstring)."""
    excluded, excluded_arrivals = excluded or [], excluded_arrivals or []
    untraced, sources, unnamed = _untraced_light(result, observation, chosen, tol, source_id)
    bounds = math.fsum(a.power_upper_bound_W or 0.0 for a in left_out)
    untraced = max(untraced, bounds)
    # (an excluded beam is counted in the plane's power, not in what the plane says it does not follow: its most adds)
    untraced += math.inf if any(row["power_upper_bound_W"] is None for row in excluded) else math.fsum(row["power_upper_bound_W"] for row in excluded)
    known = math.fsum(m.beam.power_W for m in members) + math.fsum(line.power_W for line in lines)
    fraction = None
    if math.isfinite(untraced) and (known > 0.0 or untraced <= 0.0):
        fraction = untraced / known if known > 0.0 else 0.0
    groups = {m.coherence_group for m in members} - {None}
    interfering = bool(members) and (unnamed or any(m.beam.source_id in sources for m in members)
                                     or any(a.coherence_group in groups for a in (*left_out, *excluded_arrivals)))
    # (the plane bounds did not run: what the trace does not follow is not computed, so nothing is said of its size)
    not_computed = observation.quantity_state is QuantityState.NOT_COMPUTED
    return {"scope": {"wavelength_nm": chosen, "source_id": source_id}, "outside_scope": _outside_scope(result, observation, chosen, tol, source_id),
            "map_complete": untraced <= 0.0 and not left_out and not excluded and not not_computed, "omitted_beams": _omitted_beams(left_out),
            "excluded_beams": excluded,
            "untraced_power_W": untraced if math.isfinite(untraced) and not not_computed else None,
            "untraced_fraction": None if not_computed else fraction, "max_untraced_fraction": limit,
            "interference_open": bool(members) if not_computed else interfering and untraced > 0.0,
            **({"needs": ["plane_bounds"]} if not_computed else {}),
            "_of_unknown_power": bool(left_out or sources), "_not_computed": not_computed}


def _public(omitted: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in omitted.items() if not key.startswith("_")}


def _too_large(omitted: dict[str, Any]) -> bool:
    """The light the map leaves out is more than the limit of the call (``max_untraced_fraction``; ``None``: no limit), or the
    plane bounds that would state it did not run."""
    if omitted["map_complete"]:
        return False
    if omitted.get("_not_computed"):
        return True
    limit = omitted["max_untraced_fraction"]
    if limit is None:
        return False
    return omitted["untraced_fraction"] is None or omitted["untraced_fraction"] > limit


def _withheld_reason(omitted: dict[str, Any]) -> str:
    if omitted.get("_not_computed"):
        return NOT_COMPUTED
    return BEAM_POWER_UNKNOWN if omitted["_of_unknown_power"] else UNTRACED_POWER_TOO_LARGE


def _unmodelled_W(result: ValidationResult, lines: list[Any]) -> float:
    """The power of these sampled lines that went on with no ray, its power not known (the ledger's ``UNMODELLED_RESPONSE``)."""
    keys = {(line.source_id, line.spectral_index) for line in lines}
    return math.fsum(line.sinks.get(SINK_UNMODELLED_RESPONSE, 0.0) for line in (result.power_budget.lines if result.power_budget is not None else ())
                     if (line.source_id, line.spectral_index) in keys)


def _roi_mask(roi: dict[str, Any], uu: np.ndarray, vv: np.ndarray) -> np.ndarray | None:
    """The grid cells inside ``roi`` (module docstring; ``None``: not a valid region)."""
    try:
        center = [float(c) for c in roi["center_uv_mm"]]
        if len(center) != 2 or not all(math.isfinite(c) for c in center):
            return None
        if roi["shape"] == ROI_RECTANGLE:
            half = [float(h) for h in roi["half_width_mm"]]
            if len(half) != 2 or not all(math.isfinite(h) and h > 0.0 for h in half):
                return None
            return (np.abs(uu - center[0]) <= half[0]) & (np.abs(vv - center[1]) <= half[1])
        if roi["shape"] == ROI_CIRCLE:
            radius = float(roi["radius_mm"])
            if not (math.isfinite(radius) and radius > 0.0):
                return None
            return (uu - center[0]) ** 2 + (vv - center[1]) ** 2 <= radius * radius
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _spread(values: np.ndarray) -> dict[str, float | None]:
    mean = float(values.mean())
    if mean <= 0.0:
        return {"min_over_mean": None, "std_over_mean": None}
    return {"min_over_mean": float(values.min()) / mean, "std_over_mean": float(values.std()) / mean}


def _uniformity(roi: dict[str, Any], mask: np.ndarray, incoherent: np.ndarray, half: np.ndarray | None,
                counts: np.ndarray | None) -> dict[str, Any]:
    """The uniformity of the map over the cells in the region (module docstring)."""
    values = incoherent[mask]
    out: dict[str, Any] = {"roi": roi, "cell_count": int(mask.sum()), "mean_W_per_mm2": float(values.mean()),
                           "min_W_per_mm2": float(values.min()), "max_W_per_mm2": float(values.max()), **_spread(values)}
    if half is not None:
        out["half_samples"] = _spread(half[mask])
    if counts is not None:
        out["mean_crossings_per_cell"] = float(counts[mask].mean())
    return out


def _stated_uniformity(uniformity: dict[str, Any], omitted: dict[str, Any], cell_area_mm2: float) -> dict[str, Any]:
    """A uniformity with what the map leaves out: its size against the region, or no figure above the call's ``max_untraced_fraction`` (module docstring)."""
    if omitted["map_complete"]:
        return uniformity
    if _too_large(omitted):
        # (no figure, with the fraction and the limit that withholds it)
        return {"roi": uniformity["roi"], "min_over_mean": None, "std_over_mean": None, "reason": _withheld_reason(omitted),
                "untraced_fraction": omitted["untraced_fraction"], "max_untraced_fraction": omitted["max_untraced_fraction"]}
    in_region = uniformity["mean_W_per_mm2"] * uniformity["cell_count"] * cell_area_mm2
    untraced = omitted["untraced_power_W"]
    # (no share and no bound where nothing bounds the light left out: a call without a limit states the known light's figures)
    share = untraced / in_region if in_region > 0.0 and untraced is not None else None
    bound = None if share is None else ((1.0 + math.sqrt(share)) ** 2 - 1.0 if omitted["interference_open"] else share)
    return {**uniformity, "untraced_fraction": omitted["untraced_fraction"], "roi_untraced_fraction": share, "relative_error_bound": bound}


def _sampled_rows(lines: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, power, in the first half of its line's sequence)`` of every crossing of ``lines``."""
    if not lines:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    u = np.concatenate([np.array(line.crossings.u_mm) for line in lines])
    v = np.concatenate([np.array(line.crossings.v_mm) for line in lines])
    power = np.concatenate([np.array(line.crossings.power_W) for line in lines])
    first = np.concatenate([np.array(line.crossings.sample_index) < line.half_sample_count for line in lines])
    return u, v, power, first


def _default_grid(u: np.ndarray, v: np.ndarray, power: np.ndarray, plane_area: Any, members: list[ArrivalResult],
                  center: tuple[float, float] | None) -> tuple[tuple[float, float], float]:
    """The centre and half width of a sampled map's default grid (module docstring)."""
    total = float(power.sum())
    if center is None and plane_area is not None:
        center = (float(plane_area.center_offset_mm[0]), float(plane_area.center_offset_mm[1]))
    if center is None:
        center = (float((u * power).sum()) / total, float((v * power).sum()) / total) if total > 0.0 else (0.0, 0.0)
    half = 0.0
    if u.size:
        reach = np.maximum(np.abs(u - center[0]), np.abs(v - center[1]))
        order = np.argsort(reach, kind="stable")
        enclosed = np.cumsum(power[order])
        half = float(reach[order][min(int(np.searchsorted(enclosed, DEFAULT_ENCLOSED_FRACTION * total)), reach.size - 1)])
    if plane_area is not None and plane_area.shape == "circle":
        half = plane_area.radius_mm
    elif plane_area is not None and plane_area.shape == "rectangle":
        half = max(plane_area.half_width_mm, plane_area.half_height_mm)
    elif plane_area is not None:  # (the square about its centre that holds the area)
        from ..optics import apertures
        from ..scene.lower import lower_aperture
        region = lower_aperture(plane_area)
        u_lo, u_hi, v_lo, v_hi = apertures.bounds(region)
        half = max(region.offset_u - u_lo, u_hi - region.offset_u, region.offset_v - v_lo, v_hi - region.offset_v)
    for m in members:
        g = m.beam.gaussian
        half = max(half, max(abs(m.uv_mm[0] - center[0]), abs(m.uv_mm[1] - center[1]))
                   + EXTENT_BEAM_RADII * max(g.beam_radius_x_mm, g.beam_radius_y_mm))
    return center, half


def _stated_center(center_uv_mm: Any) -> tuple[float, float] | None:
    """``center_uv_mm`` as two finite numbers, or ``None`` when it is not."""
    try:
        center = (float(center_uv_mm[0]), float(center_uv_mm[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if len(center_uv_mm) != 2 or not all(math.isfinite(c) for c in center):
        return None
    return center


def _sampled_map(observation: Any, lines: list[Any], members: list[ArrivalResult], cfg: NumericalConfig, half_extent_mm: float | None,
                 samples: int | None, center_uv_mm: Any, roi: dict[str, Any] | None, context: dict[str, Any], omitted: dict[str, Any],
                 wavelength_nm: float) -> dict[str, Any]:
    """The histogram map of sampled rays, with the beams of ``members`` at the bin centres (module docstring)."""
    context = {**context, "model": SAMPLED_IRRADIANCE_MODEL}
    u, v, power, first = _sampled_rows(lines)
    samples_basis = SAMPLES_STATED
    if samples is None:
        samples = min(max(MIN_SAMPLES, int(math.isqrt(u.size // DEFAULT_SAMPLES_PER_BIN))), DEFAULT_SAMPLES)
        samples_basis = SAMPLES_DEFAULT
    if isinstance(samples, bool) or not isinstance(samples, int) or not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        return _refusal(GRID_INVALID, f"samples must be an integer in [{MIN_SAMPLES}, {MAX_SAMPLES}]", **context)
    center = None
    if center_uv_mm is not None:
        center = _stated_center(center_uv_mm)
        if center is None:
            return _refusal(GRID_INVALID, "center_uv_mm must be two numbers [u, v] in mm", **context)
    center, default_half = _default_grid(u, v, power, observation.plane.aperture, members, center)
    half = default_half if half_extent_mm is None else half_extent_mm
    if not (isinstance(half, (int, float)) and math.isfinite(half) and half > 0.0):
        return _refusal(GRID_INVALID, "half_extent_mm must be a positive number", **context)
    u_edges = np.linspace(center[0] - half, center[0] + half, samples + 1)
    v_edges = np.linspace(center[1] - half, center[1] + half, samples + 1)
    step = 2.0 * half / samples
    cell = step * step

    def binned(weights: np.ndarray | None, rows: np.ndarray) -> np.ndarray:
        i = np.floor((u[rows] - u_edges[0]) / step).astype(np.int64)
        j = np.floor((v[rows] - v_edges[0]) / step).astype(np.int64)
        inside = (i >= 0) & (i < samples) & (j >= 0) & (j < samples)
        return np.bincount(j[inside] * samples + i[inside], weights=None if weights is None else weights[rows][inside],
                           minlength=samples * samples).reshape(samples, samples).astype(np.float64)

    everything = np.ones(u.shape, dtype=bool)
    in_grid = binned(power, everything)
    counts = binned(None, everything)
    # (the first half of each line's rays carries half of its power: twice their sum estimates the same map)
    half_map = 2.0 * binned(power, first) / cell
    u_axis, v_axis = 0.5 * (u_edges[:-1] + u_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_axis, v_axis)
    incoherent = in_grid / cell
    if members:
        points = np.column_stack([uu.ravel(), vv.ravel()])
        beams = np.zeros(points.shape[0])
        for m in members:
            single = irradiance([m], observation.plane, points, cfg)
            if single is None:
                return _refusal(GAUSSIAN_NOT_AVAILABLE, "a beam here has no valid Gaussian state", **context)
            beams = beams + single
        incoherent = incoherent + beams.reshape(incoherent.shape)
        half_map = half_map + beams.reshape(incoherent.shape)
    doc: dict[str, Any] = {
        "status": "OK", **context, "wavelength_nm": wavelength_nm, "composition": "INCOHERENT_SUM",
        "path_ids": [m.beam.path_id for m in members], "crossing_indices": [m.crossing_index for m in members],
        "sampled_lines": [{"source_id": line.source_id, "spectral_index": line.spectral_index, "sample_count": line.sample_count,
                           "samples_traced": line.samples_traced, "crossing_count": line.crossing_count} for line in lines],
        "grid": {"u_mm": u_axis.tolist(), "v_mm": v_axis.tolist(), "shape": [samples, samples], "order": "row-major: rows are v, columns u",
                 "center_uv_mm": [center[0], center[1]], "half_extent_mm": float(half), "u_edges_mm": u_edges.tolist(), "v_edges_mm": v_edges.tolist(),
                 "bin_area_mm2": cell},
        "sampling": {"grid_step_mm": step, "shortest_fringe_period_mm": None, "samples_basis": samples_basis},
        "incoherent_W_per_mm2": incoherent.tolist(), "coherent_W_per_mm2": None, "coherent_reason": INDEPENDENT_SAMPLES,
        "sample_counts": counts.astype(np.int64).tolist(),
        "grid_power_incoherent_W": float(incoherent.sum()) * cell, "grid_power_coherent_W": None,
        "power_outside_grid_W": max(float(power.sum()) - float(in_grid.sum()), 0.0), **_public(omitted),
    }
    if roi is not None:
        mask = _roi_mask(roi, uu, vv)
        if mask is None:
            return _refusal(ROI_INVALID, "roi must be {shape: rectangle, center_uv_mm: [u, v], half_width_mm: [a, b]} or "
                            "{shape: circle, center_uv_mm: [u, v], radius_mm: r}, with positive sizes", **context)
        if not mask.any():
            return _refusal(ROI_INVALID, "no grid cell centre lies in roi; use a larger region or a finer grid", **context)
        doc["uniformity"] = _stated_uniformity(_uniformity(roi, mask, incoherent, half_map, counts), omitted, cell)
    return doc


def _beam_reach(members: list[ArrivalResult], plane: ObservationPlaneFrame, middle: np.ndarray) -> float:
    """The default half width of a beam map about ``middle``: the farthest beam centre plus ``EXTENT_BEAM_RADII`` projected radii."""
    normal = np.array(plane.normal)
    reach = [float(np.max(np.abs(np.array(m.uv_mm) - middle))) + EXTENT_BEAM_RADII * max(m.beam.gaussian.beam_radius_x_mm, m.beam.gaussian.beam_radius_y_mm)
             / max(abs(float(dot(m.beam.direction, normal))), MIN_PROJECTION_COS) for m in members]
    return max(reach)


def resolving_samples(half_extent_mm: float, period_mm: float | None) -> int | None:
    """The points per axis that resolve fringes of ``period_mm`` on a grid of this half width with ``FRINGE_SAMPLES_PER_PERIOD``
    points per period, when that is more than ``DEFAULT_SAMPLES`` and at most ``MAX_SAMPLES``; ``None`` otherwise (no fringes,
    the default resolves them, or no allowed count does)."""
    if period_mm is None:
        return None
    # (one step more than the quotient where it is a whole number, so rounding never leaves the step above the limit)
    needed = int(math.floor(2.0 * half_extent_mm / (period_mm / FRINGE_SAMPLES_PER_PERIOD))) + 2
    return needed if DEFAULT_SAMPLES < needed <= MAX_SAMPLES else None


class _Selection:
    """The light of one wavelength on the plane, sorted into what a map can hold and what it leaves out (module docstring)."""

    def __init__(self, observation: Any, group: list[ArrivalResult], sampled: list[Any], cfg: NumericalConfig) -> None:
        # (a beam whose power is not known has no irradiance to add: the map says it leaves it out, and one outside the cone cannot reach it)
        known = [a for a in group if a.beam.power_W is not None]
        unknown = [a for a in group if a.beam.power_W is None and a.within_acceptance is not False]
        candidates = [*known, *unknown]
        self.any_light = bool(known or unknown or sampled)
        self.scoped = [a for a in candidates if a.power_out_of_scope_at]
        self.taken = [a for a in candidates if a.power_taken_before_at and not a.power_out_of_scope_at]
        rest = [a for a in candidates if not a.power_out_of_scope_at and not a.power_taken_before_at]
        self.left_out = [a for a in rest if a.beam.power_W is None]
        self.shapeless, self.members = [], []
        for a in rest:
            if a.beam.power_W is not None:
                (self.members if _usable(a, observation.plane, cfg) else self.shapeless).append(a)
        self.lines = [line for line in sampled if not line.power_out_of_scope_at]
        self.scoped_lines = [line for line in sampled if line.power_out_of_scope_at]
        self.excluded_arrivals = [*self.scoped, *self.taken, *self.shapeless]
        self.excluded = _excluded_beams(self.scoped, self.taken, self.shapeless, self.scoped_lines)

    def refusal(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Why no map of this wavelength exists, when no beam or line of it has a profile and a power (``None``: one does)."""
        if not self.any_light:
            return _refusal(NO_ARRIVALS, "the only beams here arrive outside the plane's acceptance cone, with no known power", **context)
        if self.members or self.lines:
            return None
        out_of_scope = sorted({*power_scope_surfaces(self.scoped), *(s for line in self.scoped_lines for s in line.power_out_of_scope_at)})
        if out_of_scope:
            # (a beam's power behind those surfaces is the chief ray's, not the light's: no map of it, as the plane's own power)
            return _refusal(POWER_OUT_OF_MODEL_SCOPE, power_scope_message(out_of_scope, "its irradiance map"), quantity_state="OUT_OF_MODEL_SCOPE",
                            power_out_of_scope_at=out_of_scope, path_ids=sorted({*(a.beam.path_id for a in self.scoped),
                                                                                 *(line.parent_path_id or line.source_id for line in self.scoped_lines)}),
                            **context)
        taken = taken_before_surfaces(self.taken)
        if taken:
            # (the beam's power and profile are the whole beam's, the light crossing only in [P - S, P]: no map of it, as the plane)
            return _refusal(POWER_TAKEN_BEFORE_PLANE, taken_before_message(taken, "its irradiance map"), quantity_state="BOUNDED",
                            power_taken_before_at=taken, path_ids=sorted(a.beam.path_id for a in self.taken), **context)
        if self.shapeless:
            return _refusal(GAUSSIAN_NOT_AVAILABLE, "a beam here has no valid Gaussian state", **context)
        count = len(self.left_out)
        return _refusal(BEAM_POWER_UNKNOWN, f"{count} {'beam reaches' if count == 1 else 'beams reach'} this plane at a known position with no known power: state the element's efficiency",
                        omitted_beams=_omitted_beams(self.left_out), **context)


def _valid_limit(limit: Any) -> bool:
    return limit is None or (isinstance(limit, (int, float)) and not isinstance(limit, bool) and math.isfinite(limit) and limit >= 0.0)


def irradiance_map(result: ValidationResult, observation_id: str, *, cfg: NumericalConfig | None = None, wavelength_nm: float | None = None,
                   half_extent_mm: float | None = None, samples: int | None = None,
                   sources: SourceCoherence | None = None, source_id: str | None = None, center_uv_mm: Any = None,
                   roi: dict[str, Any] | None = None, max_untraced_fraction: float | None = MAX_UNTRACED_FRACTION) -> dict[str, Any]:
    """The irradiance map of one observation plane (module docstring): at ``wavelength_nm``, or, without it, of the plane's only
    wavelength or the sum over all of them. ``source_id``: of that source's light alone. ``samples``: grid points per axis of a
    beam map (when not given: ``DEFAULT_SAMPLES``, or the count that resolves the beams' fringes), bins per axis of a sampled
    one. ``max_untraced_fraction``: the share of the map's light that may be missing from it (``untraced_fraction``) above which
    a coherent map and a uniformity that light can move are withheld; ``None``: no limit.

    Returns ``status`` ``OK`` with ``grid`` (``u_mm`` / ``v_mm`` axes; values are row-major: one row per ``v``),
    ``incoherent_W_per_mm2``, ``coherent_W_per_mm2`` (``None`` with ``coherent_reason`` when a needed coherence length is
    undeclared) and the grid sums times the cell area (``grid_power_*_W``), ``map_complete`` and what it leaves out (module docstring), or
    ``UNAVAILABLE`` with a ``reason``."""

    cfg = cfg or NumericalConfig()
    context: dict[str, Any] = {"observation_id": observation_id, "model": IRRADIANCE_MODEL}
    observation = next((o for o in result.observations if o.observation_id == observation_id), None)
    if observation is None or observation.plane is None:
        return _refusal(NO_ARRIVALS, f"no observation plane {observation_id!r} in this run", **context)
    if not _valid_limit(max_untraced_fraction):
        return _refusal(LIMIT_INVALID, "max_untraced_fraction must be a number of at least 0, or null for no limit", **context)
    reaching = tuple(a for a in observation.arrivals if source_id is None or a.beam.source_id == source_id)
    sampled = [line for line in observation.sampled_light if line.crossing_count and (source_id is None or line.source_id == source_id)]
    tol = cfg.interference_wavelength_tol_nm
    groups = _groups(reaching, tol)
    wavelengths = sorted({g[0].beam.wavelength_nm for g in groups} | {line.wavelength_nm for line in sampled})
    if not wavelengths:
        return _refusal(NO_ARRIVALS, "no beam counts as arriving at this plane", **context)
    lines_of = _Lines(result, observation, groups, sampled, cfg, tol, source_id, sources, max_untraced_fraction, context)
    if wavelength_nm is None and wavelengths[-1] - wavelengths[0] > tol:
        return lines_of.total(_merged(wavelengths, tol), half_extent_mm, samples, center_uv_mm, roi)
    chosen = wavelengths[0] if wavelength_nm is None else wavelength_nm
    if not any(abs(w - chosen) <= tol for w in wavelengths):
        return _refusal(NO_ARRIVALS, f"no beam of {wavelength_nm} nm arrives here", wavelengths_nm=wavelengths, **context)
    return lines_of.one(chosen, half_extent_mm, samples, center_uv_mm, roi)


def _merged(wavelengths: list[float], tol: float) -> list[float]:
    """The distinct wavelengths (ascending; those within ``tol`` of the one before are the same line)."""
    out: list[float] = []
    for w in wavelengths:
        if not out or w - out[-1] > tol:
            out.append(w)
    return out


class _Lines:
    """The maps of a plane's wavelengths: one wavelength's (``one``), and their sum on one grid (``total``)."""

    def __init__(self, result: ValidationResult, observation: Any, groups: list[list[ArrivalResult]], sampled: list[Any], cfg: NumericalConfig,
                 tol: float, source_id: str | None, sources: SourceCoherence | None, limit: float | None, context: dict[str, Any]) -> None:
        self.result, self.observation, self.groups, self.sampled, self.cfg, self.tol = result, observation, groups, sampled, cfg, tol
        self.source_id, self.sources, self.limit, self.context = source_id, sources, limit, context

    def selection(self, chosen: float) -> _Selection:
        group = next((g for g in self.groups if abs(g[0].beam.wavelength_nm - chosen) <= self.tol), [])
        return _Selection(self.observation, group, [line for line in self.sampled if abs(line.wavelength_nm - chosen) <= self.tol], self.cfg)

    def one(self, chosen: float, half_extent_mm: Any, samples: Any, center_uv_mm: Any, roi: dict[str, Any] | None, binned: bool = False,
            selection: _Selection | None = None) -> dict[str, Any]:
        """The map of the wavelength ``chosen``. ``binned``: as bins although no sampled ray of it crosses the plane (the layout
        of a total over wavelengths one of which has sampled rays)."""

        observation, cfg, context = self.observation, self.cfg, self.context
        chosen_light = selection or self.selection(chosen)
        refusal = chosen_light.refusal(context)
        if refusal is not None:
            return refusal
        members, left_out, lines = chosen_light.members, chosen_light.left_out, chosen_light.lines
        omitted = _openness(self.result, observation, chosen, self.tol, self.source_id, members, left_out, lines, chosen_light.excluded,
                            chosen_light.excluded_arrivals, self.limit)
        if lines or binned:
            unfollowed = _unmodelled_W(self.result, lines)
            omitted.update(map_complete=omitted["map_complete"] and unfollowed <= 0.0, omitted_sampled_power_W=unfollowed)
            return _sampled_map(observation, lines, members, cfg, half_extent_mm, samples, center_uv_mm, roi, context, omitted,
                                lines[0].wavelength_nm if lines else members[0].beam.wavelength_nm)
        centers = np.array([m.uv_mm for m in members])
        middle = centers.mean(axis=0)
        if center_uv_mm is not None:
            try:
                middle = np.array([float(center_uv_mm[0]), float(center_uv_mm[1])])
            except (TypeError, ValueError, IndexError):
                return _refusal(GRID_INVALID, "center_uv_mm must be two numbers [u, v] in mm", **context)
        if half_extent_mm is None:
            half_extent_mm = _beam_reach(members, observation.plane, middle)
        if not (isinstance(half_extent_mm, (int, float)) and math.isfinite(half_extent_mm) and half_extent_mm > 0.0):
            return _refusal(GRID_INVALID, "half_extent_mm must be a positive number", **context)
        period = shortest_fringe_period_mm(members, observation.plane)
        samples_basis = SAMPLES_STATED
        if samples is None:
            # (the count that resolves the beams' fringes, where one of the allowed counts does and the coherent map is not
            # withheld for the light the map leaves out; else the default)
            resolving = None if omitted["interference_open"] and _too_large(omitted) else resolving_samples(half_extent_mm, period)
            samples, samples_basis = (DEFAULT_SAMPLES, SAMPLES_DEFAULT) if resolving is None else (resolving, SAMPLES_FRINGE_PERIOD)
        if isinstance(samples, bool) or not isinstance(samples, int) or not MIN_SAMPLES <= samples <= MAX_SAMPLES:
            return _refusal(GRID_INVALID, f"samples must be an integer in [{MIN_SAMPLES}, {MAX_SAMPLES}]", **context)
        u_axis = np.linspace(middle[0] - half_extent_mm, middle[0] + half_extent_mm, samples)
        v_axis = np.linspace(middle[1] - half_extent_mm, middle[1] + half_extent_mm, samples)
        uu, vv = np.meshgrid(u_axis, v_axis)
        points = np.column_stack([uu.ravel(), vv.ravel()])
        incoherent = np.zeros(points.shape[0])
        for m in members:
            single = irradiance([m], observation.plane, points, cfg)
            if single is None:
                return _refusal(GAUSSIAN_NOT_AVAILABLE, "a beam here has no valid Gaussian state", **context)
            incoherent = incoherent + single
        step = float(u_axis[1] - u_axis[0])
        sampling: dict[str, Any] = {"grid_step_mm": step, "shortest_fringe_period_mm": period, "samples_basis": samples_basis}
        coherent_reason = None
        extra: dict[str, Any] = {}
        if omitted["interference_open"] and _too_large(omitted):
            coherent, coherent_reason = None, _withheld_reason(omitted)
        elif period is not None and step > period / FRINGE_SAMPLES_PER_PERIOD:
            coherent, coherent_reason = None, FRINGES_UNDERSAMPLED
            sampling.update(max_step_mm=period / FRINGE_SAMPLES_PER_PERIOD,
                            resolving_half_extent_mm=period / FRINGE_SAMPLES_PER_PERIOD * (samples - 1) / 2.0)
        else:
            coherent = irradiance(members, observation.plane, points, cfg, self.sources)
            if coherent is None:
                coherent_reason = COHERENCE_UNDEFINED
            elif (entry := assumed_coherence(members, self.sources)) is not None:
                # (a line that states no coherence, taken as monochromatic: the coherent map is the stated model's at that stand-in)
                extra["value_basis"] = {"coherent_W_per_mm2": entry.model_dump(mode="json")}
        cell = float((u_axis[1] - u_axis[0]) * (v_axis[1] - v_axis[0]))
        shape = (samples, samples)
        if roi is not None:
            mask = _roi_mask(roi, uu, vv)
            if mask is None:
                return _refusal(ROI_INVALID, "roi must be {shape: rectangle, center_uv_mm: [u, v], half_width_mm: [a, b]} or "
                                "{shape: circle, center_uv_mm: [u, v], radius_mm: r}, with positive sizes", **context)
            if not mask.any():
                return _refusal(ROI_INVALID, "no grid point lies in roi; use a larger region or a finer grid", **context)
            extra["uniformity"] = _stated_uniformity(_uniformity(roi, mask, incoherent.reshape(shape), None, None), omitted, cell)
        return {
            "status": "OK", **context, **extra, "wavelength_nm": members[0].beam.wavelength_nm,
            "path_ids": [m.beam.path_id for m in members], "crossing_indices": [m.crossing_index for m in members],
            "grid": {"u_mm": u_axis.tolist(), "v_mm": v_axis.tolist(), "shape": list(shape), "order": "row-major: rows are v, columns u",
                     "center_uv_mm": [float(middle[0]), float(middle[1])], "half_extent_mm": float(half_extent_mm)},
            "sampling": sampling,
            "incoherent_W_per_mm2": incoherent.reshape(shape).tolist(),
            "coherent_W_per_mm2": None if coherent is None else coherent.reshape(shape).tolist(),
            "coherent_reason": coherent_reason,
            "grid_power_incoherent_W": float(incoherent.sum()) * cell,
            "grid_power_coherent_W": None if coherent is None else float(coherent.sum()) * cell, **_public(omitted),
        }

    def total(self, wavelengths: list[float], half_extent_mm: Any, samples: Any, center_uv_mm: Any, roi: dict[str, Any] | None) -> dict[str, Any]:
        """The sum over the plane's wavelengths on one grid (module docstring): mutually incoherent lines add in irradiance."""

        observation, context = self.observation, self.context
        # (a wavelength whose only beams arrive outside the acceptance cone with no known power cannot reach a map)
        selections = [(w, s) for w, s in ((w, self.selection(w)) for w in wavelengths) if s.any_light]
        if not selections:
            return _refusal(NO_ARRIVALS, "the only beams here arrive outside the plane's acceptance cone, with no known power", wavelengths_nm=wavelengths, **context)
        usable = [(w, s) for w, s in selections if s.refusal(context) is None]
        if not usable:
            # (no wavelength has a map: the first one's reason, with the wavelengths to ask for one by one)
            return {**selections[0][1].refusal(context), "wavelengths_nm": wavelengths}  # type: ignore[dict-item]
        members = [m for _w, s in usable for m in s.members]
        lines = [line for _w, s in usable for line in s.lines]
        binned = bool(lines)
        # the one grid of every line's map: the centre, half width and count of a map of all the usable light
        if center_uv_mm is not None and _stated_center(center_uv_mm) is None:
            return _refusal(GRID_INVALID, "center_uv_mm must be two numbers [u, v] in mm", **context)
        samples_basis = SAMPLES_STATED
        if binned:
            u, v, power, _first = _sampled_rows(lines)
            center, default_half = _default_grid(u, v, power, observation.plane.aperture, members, None if center_uv_mm is None else _stated_center(center_uv_mm))
            if samples is None:
                samples, samples_basis = min(max(MIN_SAMPLES, int(math.isqrt(u.size // DEFAULT_SAMPLES_PER_BIN))), DEFAULT_SAMPLES), SAMPLES_DEFAULT
            half = default_half if half_extent_mm is None else half_extent_mm
        else:
            middle = np.array([m.uv_mm for m in members]).mean(axis=0) if center_uv_mm is None else np.array(_stated_center(center_uv_mm))
            center = (float(middle[0]), float(middle[1]))
            half = _beam_reach(members, observation.plane, middle) if half_extent_mm is None else half_extent_mm
            if samples is None and isinstance(half, (int, float)) and math.isfinite(half) and half > 0.0:
                # (the count that resolves every line's fringes, where one of the allowed counts does; else the default)
                counts = [resolving_samples(half, shortest_fringe_period_mm(s.members, observation.plane)) for _w, s in usable]
                found = [c for c in counts if c is not None]
                samples, samples_basis = (max(found), SAMPLES_FRINGE_PERIOD) if found else (DEFAULT_SAMPLES, SAMPLES_DEFAULT)
        docs = []
        for w, s in usable:
            doc = self.one(w, half, samples, list(center), None, binned=binned, selection=s)
            if doc["status"] != "OK":
                return doc
            docs.append(doc)
        first = docs[0]
        incoherent = np.array(first["incoherent_W_per_mm2"], dtype=np.float64)
        for doc in docs[1:]:
            incoherent = incoherent + np.array(doc["incoherent_W_per_mm2"], dtype=np.float64)
        without = [doc for doc in docs if doc["coherent_W_per_mm2"] is None]
        coherent = None
        if not without:
            coherent = np.array(first["coherent_W_per_mm2"], dtype=np.float64)
            for doc in docs[1:]:
                coherent = coherent + np.array(doc["coherent_W_per_mm2"], dtype=np.float64)
        grid = first["grid"]
        cell = grid["bin_area_mm2"] if binned else float((grid["u_mm"][1] - grid["u_mm"][0]) * (grid["v_mm"][1] - grid["v_mm"][0]))
        # what the sum leaves out: every line's, and the wavelengths that have no map at all
        refused = [(w, s) for w, s in selections if s.refusal(context) is not None]
        untraced = [doc["untraced_power_W"] for doc in docs]
        unbounded = any(value is None for value in untraced) or bool(refused)
        beams_known = math.fsum(m.beam.power_W for m in members) + math.fsum(line.power_W for line in lines)
        total_untraced = None if unbounded else math.fsum(untraced)
        not_computed = any("needs" in doc for doc in docs)
        fraction = None
        if total_untraced is not None and not not_computed and (beams_known > 0.0 or total_untraced <= 0.0):
            fraction = total_untraced / beams_known if beams_known > 0.0 else 0.0
        complete = all(doc["map_complete"] for doc in docs) and not refused
        omitted: dict[str, Any] = {
            "scope": {"wavelength_nm": None, "source_id": self.source_id},
            "outside_scope": [row for doc in docs for row in doc["outside_scope"] if row["sources"] is not None],
            "map_complete": complete,
            "omitted_beams": [row for doc in docs for row in doc["omitted_beams"]] + [row for _w, s in refused for row in _omitted_beams(s.left_out)],
            "excluded_beams": [row for doc in docs for row in doc["excluded_beams"]] + [row for _w, s in refused for row in s.excluded],
            "untraced_power_W": None if not_computed else total_untraced, "untraced_fraction": fraction, "max_untraced_fraction": self.limit,
            "interference_open": any(doc["interference_open"] for doc in docs),
            **({"needs": ["plane_bounds"]} if not_computed else {}),
            "_of_unknown_power": any(doc["omitted_beams"] for doc in docs) or any(s.left_out for _w, s in refused), "_not_computed": not_computed,
        }
        if binned:
            omitted["omitted_sampled_power_W"] = math.fsum(doc.get("omitted_sampled_power_W", 0.0) for doc in docs)
        reasons = [doc["coherent_reason"] for doc in docs]
        per_wavelength = [{"wavelength_nm": doc["wavelength_nm"], "path_ids": doc["path_ids"], "crossing_indices": doc["crossing_indices"],
                           "grid_power_incoherent_W": doc["grid_power_incoherent_W"], "grid_power_coherent_W": doc["grid_power_coherent_W"],
                           "coherent_reason": doc["coherent_reason"], "map_complete": doc["map_complete"],
                           "untraced_power_W": doc["untraced_power_W"], "untraced_fraction": doc["untraced_fraction"]} for doc in docs]
        per_wavelength += [{"wavelength_nm": w, "path_ids": [], "crossing_indices": [], "grid_power_incoherent_W": None, "grid_power_coherent_W": None,
                            "coherent_reason": None, "map_complete": False, "untraced_power_W": None, "untraced_fraction": None,
                            "reason": s.refusal(context)["reason"]} for w, s in refused]  # type: ignore[index]
        total: dict[str, Any] = {
            "status": "OK", **{**context, "model": first["model"]}, "wavelength_nm": None, "wavelengths_nm": wavelengths,
            "composition_over_wavelengths": "INCOHERENT_SUM", **({"composition": "INCOHERENT_SUM"} if binned else {}),
            "path_ids": [p for doc in docs for p in doc["path_ids"]], "crossing_indices": [c for doc in docs for c in doc["crossing_indices"]],
            **({"sampled_lines": [row for doc in docs for row in doc["sampled_lines"]]} if binned else {}),
            "grid": grid,
            "sampling": {"grid_step_mm": first["sampling"]["grid_step_mm"],
                         "shortest_fringe_period_mm": min((doc["sampling"]["shortest_fringe_period_mm"] for doc in docs
                                                           if doc["sampling"]["shortest_fringe_period_mm"] is not None), default=None),
                         "samples_basis": samples_basis},
            "incoherent_W_per_mm2": incoherent.tolist(), "coherent_W_per_mm2": None if coherent is None else coherent.tolist(),
            "coherent_reason": None if coherent is not None else next(reason for reason in reasons if reason is not None),
            **({"sample_counts": np.sum([np.array(doc["sample_counts"]) for doc in docs], axis=0).astype(np.int64).tolist()} if binned else {}),
            "grid_power_incoherent_W": math.fsum(doc["grid_power_incoherent_W"] for doc in docs),
            "grid_power_coherent_W": None if coherent is None else math.fsum(doc["grid_power_coherent_W"] for doc in docs),
            **({"power_outside_grid_W": math.fsum(doc["power_outside_grid_W"] for doc in docs)} if binned else {}),
            "per_wavelength": sorted(per_wavelength, key=lambda row: row["wavelength_nm"]), **_public(omitted),
        }
        if coherent is not None and (entry := assumed_coherence(members, self.sources)) is not None:
            # (a line of the sum that states no coherence, taken as monochromatic, as in its own map)
            total["value_basis"] = {"coherent_W_per_mm2": entry.model_dump(mode="json")}
        if roi is not None:
            uu, vv = np.meshgrid(np.array(grid["u_mm"]), np.array(grid["v_mm"]))
            mask = _roi_mask(roi, uu, vv)
            if mask is None:
                return _refusal(ROI_INVALID, "roi must be {shape: rectangle, center_uv_mm: [u, v], half_width_mm: [a, b]} or "
                                "{shape: circle, center_uv_mm: [u, v], radius_mm: r}, with positive sizes", **context)
            if not mask.any():
                return _refusal(ROI_INVALID, "no grid point lies in roi; use a larger region or a finer grid", **context)
            counts = np.array(total["sample_counts"], dtype=np.float64) if binned else None
            total["uniformity"] = _stated_uniformity(_uniformity(roi, mask, incoherent, None, counts), omitted, cell)
        return total
