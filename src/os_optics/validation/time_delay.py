"""Time-domain readings of the trace: a beam's group delay and phase delay from its source, and two beams' relative delay.

Conventions (``group_delay_ps``, ``phase_delay_ps``, ``time_delay_ps`` target metrics):

* **group delay** of a beam at a plane: the arrival time of a wave packet's envelope, ``t_g = G / c`` with ``G`` the chief
  ray's group optical path from the source (``ArrivalResult.group_path_length_mm``: the sum of ``n_g L`` over the media it
  crossed, ``n_g = n - lambda dn/dlambda`` at the line's vacuum wavelength; ``n_g = n`` in a medium without dispersion, so
  in air of index 1 it is the geometric length). ``t = 0`` is the emission at the source's origin plane. It is the
  dispersive delay, the delay of the envelope, valid for a pulse whose spectrum is narrow beside the media's dispersion: the
  pulse shape, its broadening by the group-delay dispersion (``group_delay_dispersion_mm2 / c^2``) and higher orders are
  not part of it. It holds for a continuous source too (the delay of its modulation).
* **phase delay**: ``t_p = OPL / c`` with ``OPL`` the chief ray's optical path length (``sum n L``): the delay of the carrier's
  phase fronts (coating phases not included). Behind an element that states no optical path (an ideal or tabulated thin
  lens, a Faraday rotator, a diffraction grating) it has no value.
* **relative delay** of two beams a and b on a plane: ``t_a - t_b = (G_a - G_b) / c`` (``InterferenceResult.
  group_path_difference_mm``), positive when a arrives later; the passes both beams make alike cancel, so it is known where
  the two beams crossed the same passes of undetermined group delay, and known within ``group_delay_bound_mm / c`` where
  every pass they do not share carries a coating's group-delay bound. Two beams of different sources have no common time
  origin: no relative delay.

Passes whose group delay the design does not state (``ArrivalResult.group_delay_unknown_at``: an element without a thickness,
a coating without a stated phase). Where each of them carries a coating's group-delay bound, the delay lies within the
traced value plus or minus the sum of those bounds, and a target is graded on that interval. Where one does not, the
value is the delay of the modelled path -- ``group_path_length_mm / c``, the number the arrival holds, without those
passes -- reported with ``GROUP_DELAY_UNDEFINED`` and the entry ``UNSTATED_GROUP_DELAY_ZERO`` at those passes
(``value_basis["value"]``, an ``ESTIMATE``) and not graded: the element's own delay is not in the number and its sign is
not known in general (a lens adds, a chirped coating may subtract), so no interval is claimed. The same holds for two
beams that cross different such passes: ``(G_a - G_b) / c`` of the two arrivals' modelled paths, not graded. A medium
whose group index is not available leaves no value, and so do two beams of different sources. Arithmetic: one division
by ``MM_PER_PS`` (identical bits in every runtime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts.basis import Assumed, ValueBasis
from ..contracts.results import ArrivalResult, InterferenceResult, ObservationResult
from ..errors import ErrorCode, Status
from ..errors.status import AssumedInput, ValueKind
from ..math.vectors import correctly_rounded_sum
from ..optics.interference.coherence import delay_bound_fs

if TYPE_CHECKING:
    from .metrics import MetricValue

#: The speed of light in vacuum in mm/ps (exact: 299 792 458 m/s).
MM_PER_PS = 0.299792458
#: Femtoseconds per picosecond.
FS_PER_PS = 1000.0
#: The group-delay metrics of one beam and of a beam pair.
GROUP_DELAY_METRICS = frozenset({"group_delay_ps"})
PHASE_DELAY_METRICS = frozenset({"phase_delay_ps"})
PAIR_DELAY_METRICS = frozenset({"time_delay_ps"})
UNIT = "ps"


def _undefined(message: str, codes: tuple[str, ...] = ()) -> MetricValue:
    from .metrics import MetricValue

    return MetricValue(None, UNIT, Status.UNKNOWN, tuple(dict.fromkeys((ErrorCode.GROUP_DELAY_UNDEFINED.value, *codes))), message)


def _open_passes(a: ArrivalResult) -> list[str]:
    """The passes of ``a`` whose group delay the design does not state and no coating bound holds (``surface:event`` each)."""

    return [x.split("@")[0] for x in a.group_delay_unknown_at if delay_bound_fs(x) is None]


def _modelled_path(passes: list[str]) -> ValueBasis:
    """The entry of a delay of the modelled path: the passes without a stated group delay add none."""

    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(
        input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=tuple(dict.fromkeys(passes)), code=ErrorCode.GROUP_DELAY_UNDEFINED.value),))


def group_delay(a: ArrivalResult) -> MetricValue:
    """The beam's group delay from its source in ps (module docstring), or why it has none."""
    from .metrics import MetricValue

    if a.group_path_length_mm is None:
        return _undefined("the group index of a medium on the beam's way is not available at its wavelength: its group delay "
                          "has no value; state the medium's dispersion (Sellmeier or tabulated index)")
    bounds = [delay_bound_fs(x) for x in a.group_delay_unknown_at]
    open_passes = _open_passes(a)
    value = a.group_path_length_mm / MM_PER_PS
    if open_passes:
        return MetricValue(value, UNIT, Status.UNKNOWN, (ErrorCode.GROUP_DELAY_UNDEFINED.value,),
                           f"the beam crosses passes whose group delay the design does not state ({', '.join(open_passes[:8])}: an "
                           "element without a thickness, a coating without a stated phase or group-delay bound): the value is the "
                           "delay of the modelled path without those passes and is not graded; model the element with its "
                           "thickness and material, or state the coating's group-delay bound", basis=_modelled_path(open_passes))
    if not bounds:
        return MetricValue(value, UNIT)
    spread = correctly_rounded_sum(b for b in bounds if b is not None) / FS_PER_PS
    return MetricValue(value, UNIT, Status.PASS, (), f"the coatings on the beam's way bound their group delays: the delay lies "
                       f"within {spread:.6g} ps of the traced value", untraced=(value - spread, value + spread),
                       subject="the group delay is")


def phase_delay(a: ArrivalResult) -> MetricValue:
    """The beam's phase delay ``OPL / c`` in ps (the caller has checked that its optical path is defined)."""
    from .metrics import MetricValue

    return MetricValue(a.beam.optical_path_length_mm / MM_PER_PS, UNIT)  # type: ignore[operator]


def _pair_arrivals(r: InterferenceResult, obs: ObservationResult | None) -> tuple[ArrivalResult, ArrivalResult] | None:
    """The two arrivals of the pair ``r`` on its plane, in the pair's order."""

    if obs is None:
        return None
    by_key = {(a.beam.path_id, a.crossing_index): a for a in obs.arrivals}
    a, b = by_key.get((r.path_id_a, r.crossing_index_a)), by_key.get((r.path_id_b, r.crossing_index_b))
    return None if a is None or b is None else (a, b)


def time_delay(r: InterferenceResult, reverse: bool, obs: ObservationResult | None = None) -> MetricValue:
    """The relative group delay ``t_a - t_b`` of a pair in ps (``reverse``: the target's operands are the pair's b, a;
    ``obs``: the pair's plane, whose arrivals hold each beam's modelled group path)."""
    from .metrics import MetricValue

    if ErrorCode.INCOHERENT_SOURCES.value in r.issue_codes and r.source_id_a != r.source_id_b:
        return _undefined("the two beams come from different sources, whose emission times the Project does not relate: their "
                          "relative delay has no value", (ErrorCode.INCOHERENT_SOURCES.value,))
    if r.group_path_difference_mm is None:
        pair = _pair_arrivals(r, obs)
        if pair is not None and pair[0].group_path_length_mm is not None and pair[1].group_path_length_mm is not None:
            a, b = pair
            # (the passes both beams make alike cancel: the ones one beam makes and the other does not are left out)
            shared = set(a.group_delay_unknown_at) & set(b.group_delay_unknown_at)
            passes = [x.split("@")[0] for beam in pair for x in beam.group_delay_unknown_at
                      if x not in shared and delay_bound_fs(x) is None]
            modelled = (a.group_path_length_mm - b.group_path_length_mm) / MM_PER_PS
            return MetricValue(-modelled if reverse else modelled, UNIT, Status.UNKNOWN, (ErrorCode.GROUP_DELAY_UNDEFINED.value,),
                               "the two beams cross different passes whose group delay the design does not state "
                               f"({', '.join(dict.fromkeys(passes[:8]))}: an element without a thickness, a coating without a stated "
                               "phase or group-delay bound): the value is the relative delay of the two modelled paths without "
                               "those passes and is not graded; model the element with its thickness and material, or route "
                               "both beams through the same passes",
                               basis=_modelled_path(passes or [*_open_passes(a), *_open_passes(b)]))
        return _undefined("a medium's group index is not available at the beams' wavelength: their relative delay has no value; "
                          "state the medium's dispersion (Sellmeier or tabulated index)")
    value = r.group_path_difference_mm / MM_PER_PS
    if reverse:
        value = -value
    entry = (r.value_basis or {}).get("group_path_difference_mm")
    if entry is not None:
        # (the pair's group-path difference on the trace's convention at passes whose delay is not stated: the relative
        # delay of the two modelled paths, reported with those passes and not graded)
        passes = [step for assumed in entry.assumed for step in assumed.at]
        return MetricValue(value, UNIT, Status.UNKNOWN, (ErrorCode.GROUP_DELAY_UNDEFINED.value,),
                           "the two beams cross different passes whose group delay the design does not state "
                           f"({', '.join(dict.fromkeys(passes[:8]))}: an element without a thickness, a coating without a stated "
                           "phase or group-delay bound): the value is the relative delay of the two modelled paths without "
                           "those passes and is not graded; model the element with its thickness and material, or route "
                           "both beams through the same passes", basis=entry)
    if not r.group_delay_bound_mm:
        return MetricValue(value, UNIT)
    spread = r.group_delay_bound_mm / MM_PER_PS
    return MetricValue(value, UNIT, Status.PASS, (), f"the coatings the two beams meet differently bound their group delays: the "
                       f"relative delay lies within {spread:.6g} ps of the traced value", untraced=(value - spread, value + spread),
                       subject="the relative delay is")
