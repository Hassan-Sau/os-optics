"""Resonant build-up of light injected into a linear cavity through its first mirror: the steady-state circulating, transmitted,
reflected and lost powers of each transverse eigenmode (Airy functions), the transverse-mode content of a coaxial Gaussian beam in
the cavity's Hermite-Gauss eigenbasis, proven bounds for the light whose modes are not resolved, and the evaluation of targets
on the analysis's quantities. Pure arithmetic, no libm transcendentals (``math.exact_trig``); used by ``os_optics.local.cavity``.

The coupler. The light enters through the input mirror (power reflectance ``R1``, transmittance ``T1``, loss ``A1 = 1 - R1 - T1``),
travels the forward pass (power fraction ``F``: the fold mirrors' reflectances and the stated loss of one pass), meets the output
mirror (``R2``, ``T2``, ``A2``) and returns along the backward pass (``F`` again) to the input mirror. The input mirror reflects the
field with ``-r1`` from outside and ``+r1`` from inside and transmits ``t1`` either way (``r1 = sqrt(R1)``, ``t1 = sqrt(T1)``: the
Stokes relations of a lossless coating, kept for a lossy one, whose scattering matrix is then a scaled unitary: its loss is
``A1`` times each incident power, with no interference term). With ``b = sqrt(R2) F`` (the field returned to the input mirror per
unit field leaving it) and the round-trip gain ``a = r1 b``, a mode whose round-trip phase is ``delta`` from resonance carries, per
unit injected power ``p`` in that mode:

* circulating (the forward field leaving the input mirror): ``P_c = T1 p / (1 + a^2 - 2 a cos delta)``
  (``= T1 p / ((1 - a)^2 + 4 a sin^2(delta / 2))``);
* at the output mirror ``F P_c``; transmitted ``T2 F P_c``; returned to the input mirror ``b^2 P_c``;
* reflected (the prompt reflection and the leakage of the returned field, added in amplitude):
  ``P_r = p (R1 + (1 - A1)^2 b^2 - 2 r1 (1 - A1) b cos delta) / (1 + a^2 - 2 a cos delta)``;
* lost in the input mirror ``A1 (p + b^2 P_c)``, in the output mirror ``A2 F P_c``, inside the cavity (folds, stated loss)
  ``(1 - F) (P_c + R2 F P_c)``.

These add up to ``p`` (every channel is computed from its own formula; the closure is a check). On resonance the build-up is
``T1 / (1 - sqrt(R1 R2) F)^2``; the reflection vanishes where ``r1 = (1 - A1) b`` (impedance matching, below). Each channel is a
ratio of two functions linear in ``sin^2(delta / 2)``, so it is monotonic in it: over every detuning its extremes are on resonance
and half-way between resonances.

Over every detuning (``swept_extremes``). The modes resonate at different detunings (their Gouy shifts), so the sum of their
channels over the detuning is bounded by branch and bound on the TEM00 round-trip phase ``delta`` in ``[0, 2 pi)``: on an
interval of ``delta`` each mode's ``sin^2`` of half its phase ranges from 0 (if its resonance lies in it, else the smaller of the
ends' values) to 1 (if its anti-resonance does, else the larger), and by the monotonicity its channel lies between its values
there; the sum of those per-mode extremes bounds the sum over the interval. Intervals are halved, those whose bound cannot
beat the best value found at an interval's midpoint are dropped, until the bound is within ``SWEEP_REL_TOL`` of that value or
``SWEEP_LEVELS`` halvings (or ``SWEEP_INTERVALS`` intervals at once) are reached; what remains is a proven bound, never looser than
the sum of each mode's own extremes (the whole range as one interval).

Mode content. A Gaussian beam coaxial with the eigenmode, of complex beam parameter ``q_a`` against the mode's ``q_b`` in one
section, couples the power fraction ``eta = 2 sqrt(Im q_a Im q_b) / |q_a - conj(q_b)|`` into ``HG_0`` and nothing into odd orders;
into ``HG_2k`` it couples ``eta C(2k, k) / 4^k (1 - eta^2)^k`` (the overlap integral of ``H_2k`` with a Gaussian, whose generating
function gives the amplitudes ``c_2k = c_0 sqrt((2k)!) / (2^k k!) mu^k``; completeness fixes ``|mu|^2 = 1 - eta^2``). The two
sections are independent: ``HG_mn`` takes the product. A round beam in a round cavity couples into ``LG_p0`` the sum over
``m + n = 2p``, ``eta_2 (1 - eta_2)^p`` with ``eta_2 = eta^2``.

Unresolved light. Light in modes the analysis does not resolve (above the stated order, or outside the paraxial or aperture
validity of the order) is bounded for every round-trip phase and any additional loss ``0 <= a' <= a``, ``0 <= F' <= F``:
circulating between ``T1 / (1 + a)^2`` and ``T1 / (1 - a)^2`` of its power, transmitted at most ``T2 F T1 / (1 - a)^2``, reflected
at most ``min(1, ((r1 + (1 - A1) b) / (1 - a))^2)`` (``os_optics.local.cavity`` caps a transmitted, reflected or lost power at the
power it follows). Its fraction is at most ``1 - sum(listed fractions)`` (``unresolved_fraction``).

Impedance matching. On resonance the reflected field is ``r1 - (1 - A1) b`` per unit incident field: ``r1 > (1 - A1) b`` is
under-coupled (the prompt reflection dominates), ``r1 < (1 - A1) b`` over-coupled, equality critically coupled (no reflection). With
the input mirror's loss ``A1`` held, the reflection vanishes at ``R1 = (1 - A1)^2 R2 F^2``; with a lossless input mirror at ``R1 = R2 F^2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts.basis import Requirement, grade_point
from ..errors.status import QuantityState, RequirementAssessment, assessment_of
from ..math.exact_trig import cis, cos_sin
from ..math.vectors import correctly_rounded_sum

#: The channels of one mode's steady state (W), in the order the power budget lists them.
CHANNELS = ("circulating_W", "at_output_mirror_W", "transmitted_W", "reflected_W", "input_mirror_loss_W", "output_mirror_loss_W", "inside_loss_W")
#: The largest transverse order ``m + n`` a request may resolve.
MAX_MODE_ORDER = 60
#: The most points of a detuning scan.
MAX_SCAN_POINTS = 4001


@dataclass(frozen=True)
class Coupler:
    """The cavity's power coefficients (module docstring)."""

    r1: float  # R1
    t1: float  # T1
    r2: float  # R2
    t2: float  # T2
    passing: float  # F: the power fraction of one pass between the end mirrors

    @property
    def a1(self) -> float:
        return 1.0 - self.r1 - self.t1

    @property
    def a2(self) -> float:
        return 1.0 - self.r2 - self.t2

    @property
    def b(self) -> float:
        """The field returned to the input mirror per unit field leaving it."""
        return math.sqrt(self.r2) * self.passing

    @property
    def gain(self) -> float:
        """The round-trip field gain ``a = r1 b``."""
        return math.sqrt(self.r1) * self.b


def steady_state(c: Coupler, power: float, delta: float) -> dict[str, float]:
    """The channels (``CHANNELS``) of a mode carrying ``power`` (W) whose round-trip phase is ``delta`` (rad) from resonance. The
    denominators are written ``(1 - a)^2 + 4 a sin^2(delta / 2)``, without the cancellation of ``1 + a^2 - 2 a cos delta`` near
    resonance."""

    a, b = c.gain, c.b
    r1 = math.sqrt(c.r1)
    if c.t1 == 0.0:  # (nothing enters: the input mirror reflects R1 and loses A1 of the light; R1 = 1 would leave a = 1)
        zero = 0.0 * power
        return {"circulating_W": zero, "at_output_mirror_W": zero, "transmitted_W": zero, "reflected_W": c.r1 * power,
                "input_mirror_loss_W": c.a1 * power, "output_mirror_loss_W": zero, "inside_loss_W": zero}
    half = cis(0.5 * delta).imag
    sine2 = half * half
    den = (1.0 - a) * (1.0 - a) + 4.0 * a * sine2
    circulating = c.t1 * power / den
    at_output = c.passing * circulating
    keep = 1.0 - c.a1
    near = r1 - keep * b
    reflected = power * (near * near + 4.0 * r1 * keep * b * sine2) / den
    return {
        "circulating_W": circulating,
        "at_output_mirror_W": at_output,
        "transmitted_W": c.t2 * at_output,
        "reflected_W": reflected,
        "input_mirror_loss_W": c.a1 * (power + b * b * circulating),
        "output_mirror_loss_W": c.a2 * at_output,
        "inside_loss_W": (1.0 - c.passing) * (circulating + c.r2 * at_output),
    }


#: The branch and bound over the detuning (module docstring): its relative gap, its halvings, the intervals followed at once,
#: and the intervals it starts from.
SWEEP_REL_TOL = 1e-9
SWEEP_LEVELS = 48
SWEEP_INTERVALS = 256
SWEEP_START = 64
_TWO_PI = 2.0 * math.pi
#: ``sin^2(pi / 2)`` as ``steady_state`` forms it (half-way between resonances).
_SINE2_ANTI = cis(0.25 * _TWO_PI).imag * cis(0.25 * _TWO_PI).imag


def channel_values(c: Coupler, powers: Any, sine2: Any) -> dict[str, Any]:
    """``steady_state``'s circulating, transmitted, reflected and lost (all three losses) powers elementwise, for modes of
    ``powers`` whose ``sin^2(delta / 2)`` is ``sine2`` (arrays; the same operations in the same order)."""

    a, b = c.gain, c.b
    if c.t1 == 0.0:
        zero = 0.0 * powers
        return {"circulating_W": zero, "transmitted_W": zero, "reflected_W": c.r1 * powers + 0.0 * sine2,
                "lost_W": c.a1 * powers + 0.0 * sine2}
    r1 = math.sqrt(c.r1)
    den = (1.0 - a) * (1.0 - a) + 4.0 * a * sine2
    circulating = c.t1 * powers / den
    at_output = c.passing * circulating
    keep = 1.0 - c.a1
    near = r1 - keep * b
    reflected = powers * (near * near + 4.0 * r1 * keep * b * sine2) / den
    lost = c.a1 * (powers + b * b * circulating) + c.a2 * at_output + (1.0 - c.passing) * (circulating + c.r2 * at_output)
    return {"circulating_W": circulating, "transmitted_W": c.t2 * at_output, "reflected_W": reflected, "lost_W": lost}


def swept_extremes(c: Coupler, powers: Sequence[float], shifts: Sequence[float], key: str) -> tuple[float, float]:
    """Proven ``(lower, upper)`` of the sum over modes (power ``powers``, Gouy shift ``shifts`` rad: a mode resonates where the
    TEM00 round-trip phase is its shift) of the channel ``key`` (``channel_values``) over every detuning (module docstring)."""

    p = np.asarray(powers, dtype=np.float64)
    resonance = np.asarray([x - math.floor(x / _TWO_PI) * _TWO_PI for x in shifts], dtype=np.float64)
    anti = np.where(resonance < math.pi, resonance + math.pi, resonance - math.pi)

    def sine2_at(points: Any) -> Any:
        half = cos_sin(0.5 * (points[:, None] - resonance[None, :]))[1]
        return half * half

    def holds(lo: Any, hi: Any, at: Any) -> Any:
        return ((lo[:, None] <= at[None, :]) & (at[None, :] <= hi[:, None])) | (hi[:, None] >= at[None, :] + _TWO_PI)

    def search(sign: float) -> float:
        lo = np.arange(SWEEP_START, dtype=np.float64) * (_TWO_PI / SWEEP_START)
        hi = np.append(lo[1:], _TWO_PI)
        best = -math.inf
        bound = math.inf
        for level in range(SWEEP_LEVELS + 1):
            s_lo, s_hi = sine2_at(lo), sine2_at(hi)
            least = np.where(holds(lo, hi, resonance), 0.0, np.minimum(s_lo, s_hi))
            most = np.where(holds(lo, hi, anti), _SINE2_ANTI, np.maximum(s_lo, s_hi))
            v_least = channel_values(c, p[None, :], least)[key]
            v_most = channel_values(c, p[None, :], most)[key]
            per_mode = np.maximum(v_least, v_most) if sign > 0.0 else np.minimum(v_least, v_most)
            bounds = np.array([sign * correctly_rounded_sum(row) for row in per_mode.tolist()])
            mid = 0.5 * (lo + hi)
            values = channel_values(c, p[None, :], sine2_at(mid))[key]
            best = max(best, max(sign * correctly_rounded_sum(row) for row in values.tolist()))
            keep = bounds > best
            bound = max(best, float(np.max(bounds)))
            if not bool(keep.any()) or bound - best <= SWEEP_REL_TOL * abs(bound) or level == SWEEP_LEVELS:
                break
            lo, hi, mid = lo[keep], hi[keep], mid[keep]
            if 2 * lo.size > SWEEP_INTERVALS:
                break
            lo, hi = np.concatenate((lo, mid)), np.concatenate((mid, hi))
        return sign * bound

    if p.size == 0:
        return 0.0, 0.0
    return search(-1.0), search(1.0)


def unresolved_bounds(c: Coupler, power: float) -> dict[str, tuple[float, float]]:
    """Proven ``(lower, upper)`` of the circulating, transmitted and reflected power of ``power`` (W) in modes the analysis does not
    resolve: any round-trip phase, any additional loss (module docstring)."""

    a, b = c.gain, c.b
    r1 = math.sqrt(c.r1)
    if c.t1 == 0.0:
        return {"circulating_W": (0.0, 0.0), "transmitted_W": (0.0, 0.0), "reflected_W": (0.0, c.r1 * power)}
    reflect = (r1 + (1.0 - c.a1) * b) / (1.0 - a)
    return {
        "circulating_W": (c.t1 * power / ((1.0 + a) * (1.0 + a)), c.t1 * power / ((1.0 - a) * (1.0 - a))),
        "transmitted_W": (0.0, c.t2 * c.passing * c.t1 * power / ((1.0 - a) * (1.0 - a))),
        "reflected_W": (0.0, power * min(1.0, reflect * reflect)),
    }


def one_section_fractions(eta: float, order_max: int) -> list[float]:
    """The power fractions of ``HG_0 .. HG_order_max`` of a coaxial Gaussian beam whose ``HG_0`` fraction is ``eta`` (odd orders 0)."""

    out = [0.0] * (order_max + 1)
    rest = 1.0 - eta * eta
    term = eta
    for n in range(0, order_max + 1, 2):
        if n > 0:
            term = term * (n - 1) / n * rest
        out[n] = term
    return out


#: The tail of a section's series is summed term by term until a term falls below this fraction of the tail so far (the rest, bounded
#: by a geometric series, is added as its bound) or ``TAIL_TERMS`` terms.
TAIL_REL = 1e-17
TAIL_TERMS = 1_000_000
#: The unresolved fraction is summed term by term only below this; above it the complement of the listed fractions (summed exactly)
#: is accurate to about 1e-10 of it, and the series of a beam far from the eigenmode converges too slowly to sum.
DIRECT_TAIL_BELOW = 1e-6
#: Below ``DIRECT_TAIL_BELOW`` the direct sum is kept unless it exceeds the complement by more than this, the rounding of the listed
#: fractions' sum (at most ``MAX_MODE_ORDER + 3`` roundings of 2^-53 each, about 7e-15).
COMPLEMENT_SLACK = 1e-13


def one_section_tail(eta: float, order: int) -> float:
    """The power fraction of a coaxial Gaussian beam (``HG_0`` fraction ``eta``) in the orders above ``order`` of one section, summed
    directly (no ``1 - sum``): the terms ``p_2k`` from the first even order above ``order``, then the geometric bound of the rest (each
    term is at most ``1 - eta^2`` times the one before), so the value is an upper bound within rounding."""

    if order < 0:
        return 1.0
    rest = 1.0 - eta * eta
    if rest <= 0.0:
        return 0.0
    term, n = eta, 0
    while n + 2 <= order:
        n += 2
        term = term * (n - 1) / n * rest
    total = 0.0
    for _ in range(TAIL_TERMS):
        n += 2
        term = term * (n - 1) / n * rest
        total += term
        if term <= TAIL_REL * total or term == 0.0:
            break
    return total + term * rest / (1.0 - rest)


def unresolved_fraction(eta_x: float, eta_y: float, order_max: int) -> float:
    """The power fraction in ``HG_mn`` with ``m + n > order_max`` (``mode_fractions`` lists the others): ``sum_m p_x(m) tail_y(order_max - m)``
    over the listed ``m``, plus ``tail_x(order_max)``, summed directly where it is below ``DIRECT_TAIL_BELOW``; above that, and wherever
    the direct sum exceeds it beyond rounding, the complement ``1 - sum(listed)`` (the listed fractions summed exactly, ``math.fsum``):
    a beam far from the eigenmode has most of its light above the order, and its series converges too slowly to sum. Never above 1."""

    if order_max < 0:
        return 1.0
    complement = max(0.0, 1.0 - math.fsum(f for _m, _n, f in mode_fractions(eta_x, eta_y, order_max)))
    if complement >= DIRECT_TAIL_BELOW:
        return complement
    fx = one_section_fractions(eta_x, order_max)
    parts = [fx[m] * one_section_tail(eta_y, order_max - m) for m in range(0, order_max + 1, 2)]
    parts.append(one_section_tail(eta_x, order_max))
    tail = sum_left(parts)
    return tail if tail <= complement + COMPLEMENT_SLACK else complement


def mode_fractions(eta_x: float, eta_y: float, order_max: int) -> list[tuple[int, int, float]]:
    """``(m, n, fraction)`` of every ``HG_mn`` with ``m + n <= order_max`` and a non-zero fraction, by order then ``m``."""

    fx, fy = one_section_fractions(eta_x, order_max), one_section_fractions(eta_y, order_max)
    found = []
    for order in range(0, order_max + 1, 2):
        for m in range(0, order + 1, 2):
            found.append((m, order - m, fx[m] * fy[order - m]))
    return found


def lg_fractions(eta_2: float, p_max: int) -> list[float]:
    """The power fractions of ``LG_p0``, ``p = 0 .. p_max``, of a round coaxial beam whose ``LG_00`` fraction is ``eta_2``."""

    out, term = [], eta_2
    for p in range(p_max + 1):
        out.append(term)
        term = term * (1.0 - eta_2)
    return out


# ------------------------------------------------------------------------------------------------ targets

TARGET_KEYS = frozenset({"target_id", "quantity_id", "target_value", "tolerance_abs", "tolerance_rel", "minimum", "maximum"})


def _window(target: Mapping[str, Any]) -> tuple[float | None, float | None]:
    low, high = target.get("minimum"), target.get("maximum")
    value = target.get("target_value")
    if value is not None:
        tolerance = max(target.get("tolerance_abs") or 0.0, (target.get("tolerance_rel") or 0.0) * abs(value))
        low = value - tolerance if low is None else max(low, value - tolerance)
        high = value + tolerance if high is None else min(high, value + tolerance)
    return low, high


def judge(target: Mapping[str, Any], quantity: Any) -> dict[str, Any]:
    """One target's assessment against its quantity. A ``KNOWN`` value decides it -- unless it carries a ``PROVEN`` interval that is
    not a point: the light left out as negligible (light in modes not resolved, below ``negligible_power_fraction``) may move it
    there, so the target is decided by that interval as below, and a limit inside it is not decided by the truncation (decision
    W28). A quantity in another state is decided by its ``PROVEN`` interval where that lies on one side of the target's bounds,
    else by the direction its value's ``value_basis`` proves (``contracts.basis.grade_point``: a lower bound meets a minimum, an
    upper bound a maximum, an estimate decides nothing), else it is ``INDETERMINATE`` with the quantity's state and reason. The
    row carries the quantity's value and its ``value_basis`` beside the interval, whatever decided."""

    low, high = _window(target)
    row: dict[str, Any] = {"target_id": target["target_id"], "quantity_id": target["quantity_id"], "minimum": low, "maximum": high}
    if quantity is None:
        return {**row, "assessment": RequirementAssessment.INDETERMINATE.value, "state": None, "reason": "QUANTITY_NOT_REPORTED",
                "message": f"this analysis reports no quantity {target['quantity_id']!r} for this request"}
    row.update(state=quantity.state.value, unit=quantity.unit)
    v, basis = quantity.value, quantity.value_basis
    interval = quantity.interval
    truncated = (interval is not None and interval.kind.value == "PROVEN" and interval.lower is not None and interval.upper is not None
                 and interval.lower < interval.upper)
    if v is not None and quantity.state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO) and not truncated:
        met = (low is None or v >= low) and (high is None or v <= high)
        return {**row, "value": v, "assessment": (RequirementAssessment.MET if met else RequirementAssessment.UNMET).value}
    if v is not None and quantity.state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO):
        # (the model's value, with the interval the light left out as negligible leaves: W28)
        lo, hi = interval.lower, interval.upper  # type: ignore[union-attr]
        inside = (low is None or lo >= low) and (high is None or hi <= high)
        outside = (low is not None and hi < low) or (high is not None and lo > high)
        assessment = RequirementAssessment.MET if inside else RequirementAssessment.UNMET if outside else RequirementAssessment.INDETERMINATE
        return {**row, "value": v, **({} if basis is None else {"value_basis": basis.model_dump(mode="json")}), "interval": [lo, hi],
                "assessment": assessment.value,
                **({} if inside or outside else {
                    "reason": "INTERVAL_STRADDLES_THRESHOLD",
                    "message": f"the value is {v:.6g}; light left out as negligible (negligible_power_fraction) may move it within "
                               f"[{lo:.6g}, {hi:.6g}], which straddles the target's bounds, so the truncation does not decide it"})}
    if v is not None:
        row["value"] = v
        if basis is not None:
            row["value_basis"] = basis.model_dump(mode="json")
    if interval is not None and interval.kind.value == "PROVEN":
        lo, hi = interval.lower, interval.upper
        inside = (low is None or (lo is not None and lo >= low)) and (high is None or (hi is not None and hi <= high))
        outside = (low is not None and hi is not None and hi < low) or (high is not None and lo is not None and lo > high)
        assessment = RequirementAssessment.MET if inside else RequirementAssessment.UNMET if outside else RequirementAssessment.INDETERMINATE
        return {**row, "interval": [lo, hi], "assessment": assessment.value,
                **({} if inside or outside else {"reason": "INTERVAL_STRADDLES_THRESHOLD",
                                                  "message": "the proven interval of the quantity straddles the target's bounds"})}
    if v is not None and basis is not None:
        assessment = assessment_of(grade_point(v, Requirement(minimum=low, maximum=high), basis=basis))
        return {**row, "assessment": assessment.value,
                **({} if assessment is not RequirementAssessment.INDETERMINATE else {"reason": quantity.reason, "message": quantity.message})}
    return {**row, "assessment": RequirementAssessment.INDETERMINATE.value, "reason": quantity.reason, "message": quantity.message}


def sum_left(values: Sequence[float]) -> float:
    """Left-to-right sum (the same order in every runtime)."""

    total = 0.0
    for value in values:
        total += value
    return total


__all__ = ["CHANNELS", "Coupler", "MAX_MODE_ORDER", "judge", "lg_fractions", "mode_fractions", "one_section_fractions", "one_section_tail",
           "steady_state", "unresolved_bounds", "unresolved_fraction"]
