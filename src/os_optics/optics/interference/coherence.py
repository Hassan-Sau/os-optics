"""The degree of coherence of two fields of one source line (the Gaussian-spectrum model of a declared coherence length).

A source of declared coherence length ``L_c`` has the Gaussian power spectrum ``exp(-((k - k0) L_c / 2)^2)`` over the vacuum
wavenumber ``k``, whose degree of coherence at a pure delay ``D`` (a length: path difference) is ``exp(-(D / L_c)^2)``. Two
fields of it that reached a point along different paths differ in their spectral phase by ``D (k - k0) + B (k - k0)^2 / 2 +
...`` with ``D`` the difference of the paths' group optical paths (``sum n_g L``: the delay of the envelopes, not the
phase path difference ``sum n L``) and ``B`` that of their group-delay dispersions (``sum k2 L``, ``k2 = lambda^3 / (2 pi)
d^2n/dlambda^2``). To second order in the dispersion the complex degree of coherence, relative to the fields' phase
difference ``phi`` at the line, is

    gamma = (1 - i b)^(-1/2) exp(-(D / L_c)^2 / (1 - i b)),   b = 2 B / L_c^2,

    |gamma| = (1 + b^2)^(-1/4) exp(-(D / L_c)^2 / (1 + b^2))

(the dispersion lowers its peak and widens it), and the two fields add to the power ``P_a + P_b + 2 sqrt(P_a P_b)
Re[gamma exp(i phi)]`` (``complex_degree_of_coherence``): where the dispersions differ, ``gamma`` has the phase
``psi = atan(b) / 2 - (D / L_c)^2 b / (1 + b^2)``, which shifts the fringes; without a dispersion difference it is the real
``|gamma|``. Higher orders of the dispersion are not included. Across a beam the envelope
delay follows the wavefront where the beam crossed no dispersive medium; through a plate (two parallel planes) it changes
with the rays' angles only (``optics/rays/envelope.py``); behind a lens or a prism it is not followed.

Passes whose group delay the model does not state. A bare interface (Fresnel, total internal reflection included) and an
uncoated (ideal) mirror delay nothing beyond the media's group paths. Every other pass has an undetermined delay: an
element without a thickness in the model -- an ideal or tabulated thin lens, a splitter plate, a polarizer, a retarder, a
Faraday rotator, a diffraction grating, an attenuator -- and a coating whose spectral phase is not stated as constant, also where light meets it beyond the
critical angle (a coating's group delay is the derivative of its phase over frequency: without a stated phase it can be
anything; a tabulated phase's derivative is not formed). Each such pass is recorded as
``'<surface_id>:<event>:<side>@<angle of incidence, deg>'`` (``delay_key``), a thin lens also with the hit point (its
thickness varies across it), and a pass whose delay may differ between polarizations also with the beam's polarization
state there (``polarization_state``): a coating at oblique incidence delays its s and p components differently, a
retarder its fast and slow ones, a Faraday rotator its two circular ones, a polarizer that passes some light along its blocking axis its two axes. Two beams'
relative envelope delay is known only when they crossed the same such passes, at the same angles, points and
polarization states (``delays_agree``: a delay depends on them) -- the same unknown delay then cancels. Otherwise the
coherence metrics are stated on the trace's convention -- no added group delay at a differing pass that neither states
nor bounds its delay (``unstated_delay_passes``) -- with an ``ESTIMATE`` entry naming those passes
(``UNSTATED_GROUP_DELAY_ZERO``); every bound (exposures, a coherent power's range) still takes such a delay as unknown:
1 in every bound.

Bounded coating delays. A coating whose data bound its group delay (``group_delay_bound_fs``: the magnitude of its
reflection and transmission group delay ``d phi / d omega``, s and p, wherever its data hold) adds that bound to the
record of each pass through it (``delay_key``: ``'...<=<bound>fs'``). Where the passes two beams do not share all carry a
bound, their relative group delay is known within the sum ``B`` of those passes' bounds, as a group path length
(``relative_delay_bound``: a pass the two beams share cancels; each pass one beam makes and the other does not -- also the
same coating met at another angle, point or polarization state -- adds its own bound). The degree of coherence is then
bounded without assuming anything else about the coatings' phases: the two fields' relative spectral phase differs from
the traced one by ``psi(k)``, whose derivative over the vacuum wavenumber ``k`` (the relative group delay, a length) stays
within ``[-B, B]``, so ``|exp(i psi(k)) - exp(i psi(k0))| <= B |k - k0|`` and the complex degree of coherence differs from
the traced one (``psi`` constant, the models' phases at the line) by at most ``B`` times the mean ``|k - k0|`` of the
line's spectrum, ``2 / (sqrt(pi) L_c)`` for the Gaussian spectrum of ``L_c`` (``coherence_spread``):

    | gamma - gamma_traced | <= 2 B / (sqrt(pi) L_c) = delta.

The coherence metrics are stated at the traced delay with the interval ``[|gamma_traced| - delta, |gamma_traced| + delta]``
(within [0, 1]) of the degree of coherence and, linearly, of every quantity it scales, whatever ``delta``: the interval is
the statement. The coatings' delays are read at the line's wavelength, as their power coefficients are (the line is
narrow beside a coating's band). A pulse train whose pulses may be mutually coherent keeps the coherence metrics without a
value where a revival is not negligible anywhere within ``B`` of the traced delay (``revival_coherence_within``: the
Gaussian line's coherence at the closest approach ``max(0, d - B)`` to a non-zero period multiple; a continuous spectrum's
coherence need not fall with the delay, so with a bound it is not stated).

The phase at the line. A group-delay bound limits how the relative phase changes across the line, not its value at the
line, ``psi(k0)``: the reflection / transmission phase of a coating whose phases are not stated is open (the trace uses
the default convention there), and so is the phase through an element without a thickness. Where a pass the two beams
do not share is of that kind, bounded or not (``phase_open``), their phase difference and coherent power are stated at the
trace's default interface phases with an ``ESTIMATE`` entry naming the passes (``UNSTATED_PHASE_AS_INTERFACE``), beside the
power's range over every phase; where such a pass also states no optical path (``path_open``) they have no value
(``PHASE_UNDEFINED``). Passes they share cancel, and a coating that states its phase at the line (a tabulated phase) gives
its value. The magnitude of the degree of coherence and every quantity built on it alone -- the
coherence factor and margin, the fringe visibilities (the contrast of the fringes wherever they lie) and the peak
irradiance (their maximum) -- do not depend on it; the power of two beams that overlap on a plane does, through
``2 Re[gamma exp(i phase)]``. A pass whose coating states its phase at the line (a tabulated phase: its value is stated, its
derivative is not formed) records that too (``'...#phase'``, before a bound: ``'...#phase<=<bound>fs'``;
``pass_phase_stated``); where a differing bounded pass does not (``phase_open``), the coherent power is open over
``[-1, 1]`` of that term (``validation/superposition.py``).
An unstated phase also leaves open how an oblique coating changes a beam that meets it in a mixed s and p state (its s and
p phases differ by an unstated amount): there the polarization overlap of the two beams, and with it the visibility, are
stated on the default phases with an entry naming those passes (``mixed_open_passes``), and a coherent power's range
takes every overlap up to 1.

Pulse trains. The pulses of a train may be mutually coherent (a mode-locked laser: each pulse is a copy of the last, with
a fixed phase): two fields of one such train are then coherent again at relative delays near every whole multiple of the
train's period length ``Lambda = c / f_rep``, where one field's pulse overlaps a later pulse of the other. The source's
data do not say whether its pulses are mutually coherent, so a pulsed source counts as if they were: the exposure bounds
take the larger of the coherence at the delay and at its distance to the nearest non-zero multiple of ``Lambda``
(``nearest_delay``), and a coherence metric has no value where such a revival could change it (``revival_coherence`` not
below ``REVIVAL_NEGLIGIBLE``). Arithmetic: ``+ - x /``, ``sqrt`` and the libm-free exponential and ``cis`` (identical bits in every
runtime).
"""

from __future__ import annotations

import math
from collections import Counter

from ...math.exact_trig import cis
from ...math.exact_trig import exp as exact_exp

SQRT_PI = math.sqrt(math.pi)


def degree_of_coherence(group_difference_mm: float, dispersion_difference_mm2: float, length_mm: float) -> float:
    """``|gamma|`` of two fields of a line of coherence length ``length_mm`` whose group optical paths differ by
    ``group_difference_mm`` and group-delay dispersions by ``dispersion_difference_mm2`` (module docstring)."""
    ratio = group_difference_mm / length_mm
    b = 2.0 * dispersion_difference_mm2 / (length_mm * length_mm)
    u = 1.0 + b * b
    return float(exact_exp(-(ratio * ratio) / u)) / math.sqrt(math.sqrt(u))


def complex_degree_of_coherence(group_difference_mm: float, dispersion_difference_mm2: float, length_mm: float) -> complex | float:
    """The complex degree of coherence ``gamma`` (module docstring: ``|gamma| exp(i psi)``) of two fields of a line of
    coherence length ``length_mm`` relative to their phase difference at the line, which they add to the power with as
    ``P_a + P_b + 2 sqrt(P_a P_b) Re[gamma exp(i phi)]``: ``degree_of_coherence`` itself (a real number, the same bits) where
    the dispersion leaves no phase (``b = 0``), else ``|gamma|`` times the unit factor ``conj(sqrt(1 - i b)) / (1 + b^2)^(1/4)``
    -- with ``sqrt(1 - i b) = sqrt((m + 1) / 2) - i sign(b) |b| / sqrt(2 (m + 1))``, ``m = sqrt(1 + b^2)`` -- and
    ``exp(-i r b / (1 + b^2))``, ``r = (D / L_c)^2``: ``+ - x /``, ``sqrt`` and the libm-free exponential and ``cis``."""
    magnitude = degree_of_coherence(group_difference_mm, dispersion_difference_mm2, length_mm)
    b = 2.0 * dispersion_difference_mm2 / (length_mm * length_mm)
    if b == 0.0:
        return magnitude
    u = 1.0 + b * b
    m = math.sqrt(u)
    root = math.sqrt(m)
    real = math.sqrt(0.5 * (m + 1.0)) / root
    imag = abs(b) / math.sqrt(2.0 * (m + 1.0)) / root
    unit = complex(real, imag if b > 0.0 else -imag)
    ratio = group_difference_mm / length_mm
    turn = cis(-(ratio * ratio) * b / u)
    value = complex(unit.real * turn.real - unit.imag * turn.imag, unit.real * turn.imag + unit.imag * turn.real)
    return complex(magnitude * value.real, magnitude * value.imag)


def largest_coherence(ratio: float, dispersion_low: float, dispersion_high: float, length_mm: float) -> float:
    """The largest ``|gamma|`` (module docstring) at the relative group delay ``ratio x length_mm`` over the relative
    group-delay dispersions between ``dispersion_low`` and ``dispersion_high`` (mm^2): ``u^(-1/4) exp(-ratio^2 / u)`` over
    ``u = 1 + b^2`` grows up to ``u = 4 ratio^2`` and falls beyond."""
    b_low = 2.0 * max(dispersion_low, 0.0) / (length_mm * length_mm)
    b_high = 2.0 * max(dispersion_high, 0.0) / (length_mm * length_mm)
    u = min(max(4.0 * ratio * ratio, 1.0 + b_low * b_low), 1.0 + b_high * b_high)
    return float(exact_exp(-(ratio * ratio) / u)) / math.sqrt(math.sqrt(u))


#: The relative margin ``largest_within_length`` adds to its bracketed maximum: above the rounding of the few operations
#: that form it (a guaranteed upper end, not a nearest value).
WITHIN_LENGTH_MARGIN = 1e-12


def largest_within_length(group_difference_mm: float, dispersion_difference_mm2: float, length_mm: float) -> float:
    """An upper end of ``|gamma|`` (``degree_of_coherence``) over every coherence length up to ``length_mm``: the degree of
    coherence of a pair one of whose lines states no coherence beside another of length ``length_mm`` (its own length,
    unknown, may be shorter; ``LineCoherence.assumed``).

    Without a dispersion difference ``exp(-(D / L)^2)`` grows with ``L``: the value at ``length_mm``. With one, ``B``, the
    shorter length can give the larger value: in ``t = L^2 / (2 |B|)`` and ``r = D^2 / (2 |B|)``,
    ``|gamma| = (t^2 / (1 + t^2))^(1/4) exp(-r t / (1 + t^2))``, whose derivative has the sign of
    ``f(t) = 2 r t^3 + t^2 - 2 r t + 1``: positive from ``t = 1`` on, so the only other maximum is the smaller root ``t1`` of
    ``f`` in ``(0, 1)``, where ``f`` dips below 0 (its least value on ``(0, 1)`` is at ``(sqrt(1 + 12 r^2) - 1) / (6 r)``).
    ``t1`` is bracketed by bisection, and the value there bounded above on its bracket ``[lo, hi]`` -- the first factor grows
    with ``t``, the exponent's ``t / (1 + t^2)`` too on ``(0, 1)`` -- by ``(hi^2 / (1 + hi^2))^(1/4) exp(-r lo / (1 + lo^2))``,
    raised by ``WITHIN_LENGTH_MARGIN``. The answer is the larger of that and the value at ``length_mm`` (that value itself
    where it is the larger: the stated length's value is then the upper end), at most 1."""
    value = degree_of_coherence(group_difference_mm, dispersion_difference_mm2, length_mm)
    if dispersion_difference_mm2 == 0.0 or group_difference_mm == 0.0 or not math.isfinite(length_mm):
        return value
    beta2 = 2.0 * abs(dispersion_difference_mm2)
    r = group_difference_mm * group_difference_mm / beta2
    end = length_mm * length_mm / beta2

    def f(t: float) -> float:
        return 2.0 * r * t * t * t + t * t - 2.0 * r * t + 1.0

    two_r = 2.0 * r
    least = (math.sqrt(1.0 + 3.0 * two_r * two_r) - 1.0) / (3.0 * two_r)  # (sqrt(1 + 12 r^2) - 1) / (6 r)
    if not f(least) < 0.0:
        return value  # (|gamma| never falls as the length grows: the stated length's value is the largest)
    lo, hi = 0.0, least
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    if lo >= end:
        return value  # (|gamma| grows up to the stated length)
    hi = min(hi, end)
    x = hi * hi / (1.0 + hi * hi)
    top = math.sqrt(math.sqrt(x)) * float(exact_exp(-(r * lo / (1.0 + lo * lo)) * (1.0 - WITHIN_LENGTH_MARGIN)))
    top *= 1.0 + WITHIN_LENGTH_MARGIN
    return value if top <= value else min(1.0, top)


#: The speed of light in vacuum (mm/s): a pulse train's period length is ``C_MM_PER_S / f_rep``.
C_MM_PER_S = 2.99792458e11
#: The group path length (mm) of one femtosecond of group delay: ``c x 1 fs``.
MM_PER_FS = 2.99792458e-4
#: What a pass's record (``delay_key``) appends for a coating whose data bound its group delay.
BOUND_MARK = "<="
BOUND_UNIT = "fs"
#: What a bounded pass's record carries before its bound where the coating states its phase at the line.
PHASE_MARK = "#phase"
#: How ``delay_key`` writes a pass through an element that states no optical path (between the surface id and the side).
NO_OPTICAL_PATH_EVENTS = (":thin_lens:", ":faraday_rotator:", ":grating_order_")
#: A degree of coherence below this changes no value it enters (below half a unit in the last place of 1).
REVIVAL_NEGLIGIBLE = 2.0 ** -55


def period_length_mm(repetition_rate_hz: float) -> float:
    """A pulse train's period length ``c / f_rep`` (mm): the group delay between two successive pulses."""
    return C_MM_PER_S / repetition_rate_hz


def revival_distance(delay_mm: float, period_mm: float) -> float:
    """The distance (mm) from a relative group delay to the nearest non-zero whole multiple of a pulse train's period
    length, where two of its pulses overlap."""
    d = abs(delay_mm)
    k = math.floor(d / period_mm)
    best = math.inf
    for m in (k, k + 1):
        if m >= 1:
            gap = abs(d - m * period_mm)
            if gap < best:
                best = gap
    return best


def nearest_delay(delay_mm: float, periods_mm: tuple[float, ...]) -> float:
    """The smallest of ``|delay|`` and its distances to the nearest non-zero multiple of each period length: the delay
    the degree of coherence of fields of mutually coherent pulse trains is at most the coherence of (module docstring)."""
    d = abs(delay_mm)
    for period in periods_mm:
        gap = revival_distance(d, period)
        if gap < d:
            d = gap
    return d


def revival_coherence(delay_mm: float, dispersion_difference_mm2: float, length_mm: float,
                      periods_mm: tuple[float, ...], within_length: bool = False) -> float:
    """The largest degree of coherence a revival of a pulse train could give at the relative group delay ``delay_mm``:
    the coherence at its distance to the nearest non-zero multiple of a period length (0 without a pulse train);
    ``within_length``: over every coherence length up to ``length_mm`` (``largest_within_length``)."""
    best = 0.0
    degree = largest_within_length if within_length else degree_of_coherence
    for period in periods_mm:
        value = degree(revival_distance(delay_mm, period), dispersion_difference_mm2, length_mm)
        if value > best:
            best = value
    return best


def polarization_state(first: complex, second: complex) -> str:
    """A beam's polarization state at a pass from its field's components along the pass's own two axes (s and p of its
    plane of incidence; a retarder's fast and slow axes; a polarizer's transmission and blocking axes): its normalized
    Stokes parameters ``(|a|^2 - |b|^2, 2 Re(a* b), 2 Im(a* b)) / (|a|^2 + |b|^2)``, each to 1e-6 -- whatever the field's
    power and overall phase ("0" for no field)."""
    aa = first.real * first.real + first.imag * first.imag
    bb = second.real * second.real + second.imag * second.imag
    total = aa + bb
    if not total > 0.0:
        return "0"
    cross_re = first.real * second.real + first.imag * second.imag
    cross_im = first.real * second.imag - first.imag * second.real
    values = ((aa - bb) / total, 2.0 * cross_re / total, 2.0 * cross_im / total)
    return ",".join(f"{round(v, 6) + 0.0:.6f}" for v in values)


def delay_key(surface_id: str, event: str, from_positive: bool, aoi_deg: float, at: tuple[float, float] | None = None,
              state: str | None = None, bound_fs: float | None = None, phase_stated: bool = False) -> str:
    """The record of a pass whose group delay the model does not state (module docstring); ``at``: the hit point on the
    surface (mm, its own coordinates) where the delay also depends on it (a thin lens: its thickness varies across it);
    ``state``: the beam's polarization state there (``polarization_state``) where the delay may depend on it;
    ``bound_fs``: the coating's group-delay bound, where its data state one (written exactly, ``repr``);
    ``phase_stated``: a coating pass whose coating states its phase at the line (``PHASE_MARK``, before the bound); a
    hit-point coordinate that rounds to 0 reads 0, whatever the sign of its last bits."""
    where = "" if at is None else f"({round(at[0], 6) + 0.0:.6f},{round(at[1], 6) + 0.0:.6f})"
    polarized = "" if state is None else f"[{state}]"
    bounded = (PHASE_MARK if phase_stated else "") + ("" if bound_fs is None else f"{BOUND_MARK}{bound_fs!r}{BOUND_UNIT}")
    return f"{surface_id}:{event}:{'+' if from_positive else '-'}@{aoi_deg:.6f}{where}{polarized}{bounded}"


def pass_phase_stated(key: str) -> bool:
    """Whether a pass's record says its coating states its phase at the line (``delay_key``)."""
    return PHASE_MARK in key.rpartition("@")[2]


def _sp_mixed(key: str) -> bool:
    """Whether a pass's record carries a polarization state that is neither s nor p (``polarization_state``: its first
    Stokes parameter along the pass's own axes is not +-1)."""
    if "[" not in key or "]" not in key:
        return False
    state = key[key.index("[") + 1:key.index("]")]
    try:
        first = float(state.split(",")[0])
    except ValueError:
        return True
    return abs(first) != 1.0


def phase_open(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether a pass the two beams do not share has no stated phase at the line (module docstring): an element without a
    thickness, or a coating that states no phase there (bounded or not). Their relative phase at the line is then open;
    passes they share cancel."""
    return any(not pass_phase_stated(key) for key in differing_passes(a, b))


def path_open(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether a pass the two beams do not share states no optical path: an ideal or tabulated thin lens, a Faraday rotator or a
    grating. Their optical path difference then has no value; passes they share cancel, and any other pass (a coating's open
    phase included) leaves n x length defined."""
    return any(any(event in key for event in NO_OPTICAL_PATH_EVENTS) for key in differing_passes(a, b))


def polarization_kept(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether every bounded pass of open phase the two beams do not share meets its beam in s or p (or at normal
    incidence), where the coating's unstated s and p phases change only the beam's phase, not its polarization (module
    docstring)."""
    return not any(delay_bound_fs(key) is not None and not pass_phase_stated(key) and _sp_mixed(key)
                   for key in differing_passes(a, b))


def pass_id(key: str) -> str:
    """A pass's short id ``'<surface_id>:<event>:<side>'`` (its record without the angle and what follows)."""
    return key.split("@")[0]


def unstated_delay_passes(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """The short ids of the passes the two beams do not share whose group delay is neither stated nor bounded: the passes
    where the trace's convention (no added group delay) stands for the delay (module docstring)."""
    return tuple(sorted({pass_id(key) for key in differing_passes(a, b) if delay_bound_fs(key) is None}))


def mixed_open_passes(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    """The short ids of the passes the two beams do not share that state no phase at the line and meet their beam in a state
    mixing s and p (or a retarder's two axes): there the trace's default interface phases stand for the polarization the
    pass leaves (module docstring)."""
    return tuple(sorted({pass_id(key) for key in differing_passes(a, b) if not pass_phase_stated(key) and _sp_mixed(key)}))


def delay_bound_fs(key: str) -> float | None:
    """The group-delay bound (fs) a pass's record carries (``delay_key``), or None when its delay is not bounded."""
    if not key.endswith(BOUND_UNIT) or BOUND_MARK not in key:
        return None
    try:
        value = float(key[key.rindex(BOUND_MARK) + len(BOUND_MARK):-len(BOUND_UNIT)])
    except ValueError:
        return None
    return value if value > 0.0 and math.isfinite(value) else None


def delays_agree(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether two beams crossed the same passes of undetermined group delay (the same records, in any order): only then is
    their relative envelope delay known."""
    return a == b or sorted(a) == sorted(b)


def differing_passes(a: tuple[str, ...], b: tuple[str, ...]) -> list[str]:
    """The passes one beam made and the other did not (counted with their multiplicity), sorted."""
    left, right = Counter(a), Counter(b)
    return sorted(((left - right) + (right - left)).elements())


def relative_delay_bound(a: tuple[str, ...], b: tuple[str, ...]) -> float | None:
    """The bound (mm of group path) of two beams' relative group delay over the passes of undetermined delay they do not
    share (module docstring): 0.0 when they agree, the sum of those passes' bounds when every one carries a bound, None
    when one does not (their relative delay is not known)."""
    if delays_agree(a, b):
        return 0.0
    total = 0.0
    for key in differing_passes(a, b):
        bound = delay_bound_fs(key)
        if bound is None:
            return None
        total += bound
    return total * MM_PER_FS


def coherence_spread(bound_mm: float, length_mm: float) -> float:
    """The largest change ``delta = 2 B / (sqrt(pi) L_c)`` of the degree of coherence of a line of coherence length
    ``length_mm`` that a relative group delay known within ``bound_mm`` leaves (module docstring)."""
    return 2.0 * bound_mm / (SQRT_PI * length_mm)


def revival_coherence_within(delay_mm: float, bound_mm: float, dispersion_difference_mm2: float, length_mm: float,
                             periods_mm: tuple[float, ...], within_length: bool = False) -> float:
    """The largest degree of coherence a revival of a pulse train could give at any relative group delay within
    ``bound_mm`` of ``delay_mm``: the distance to the nearest non-zero multiple of a period length changes by at most the
    change of the delay, and the Gaussian line's coherence falls with that distance, so its largest value is at the
    distance ``max(0, d - bound_mm)`` (0 without a pulse train); ``within_length``: over every coherence length up to
    ``length_mm`` (``largest_within_length``)."""
    best = 0.0
    degree = largest_within_length if within_length else degree_of_coherence
    for period in periods_mm:
        value = degree(max(0.0, revival_distance(delay_mm, period) - bound_mm), dispersion_difference_mm2, length_mm)
        if value > best:
            best = value
    return best


__all__ = ["C_MM_PER_S", "MM_PER_FS", "REVIVAL_NEGLIGIBLE", "coherence_spread", "complex_degree_of_coherence", "degree_of_coherence",
           "delay_bound_fs", "delay_key", "delays_agree", "differing_passes", "largest_coherence", "largest_within_length",
           "mixed_open_passes", "nearest_delay", "pass_id", "pass_phase_stated", "period_length_mm", "phase_open", "polarization_kept",
           "polarization_state", "relative_delay_bound", "revival_coherence", "revival_coherence_within", "revival_distance",
           "unstated_delay_passes"]
