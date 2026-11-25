"""Detection at an observation plane: photocurrent, noise and signal-to-noise ratio of the light the plane counts, and the
lock-in measurement of its modulated part.

An observation plane that states a detector (``Observation.detection``, ``contracts/detection.py``) is the detector's sensitive
face: the light it reads is the light the run counts on the plane (``ArrivalResult`` received power: on its active area,
within its acceptance; and the counted sampled rays), each source at each wavelength a ``DetectionContribution``. The model,
with SI constants ``q`` (elementary charge), ``h``, ``c`` and ``k`` (Boltzmann), exact since 2019:

* Responsivity ``R(lambda) = QE(lambda) q lambda / (h c)`` A/W from the stated quantum-efficiency table (linear between rows,
  unavailable outside them). Photocurrent ``I = sum R(lambda_k) P_k``.
* Bandwidth: ``FIRST_ORDER_3DB`` ``f_c`` gives the noise-equivalent bandwidth ``B = pi f_c / 2`` (the integral of
  ``1 / (1 + (f / f_c)^2)`` over f >= 0); ``NOISE_EQUIVALENT`` states ``B``.
* Noise current densities (one-sided, A^2/Hz): the light's shot noise ``2 q I``; the dark floor -- ``(NEP R(lambda_NEP))^2``
  from a stated noise-equivalent power, or ``2 q I_d`` from a dark current plus ``4 k T / R_L`` from a load resistance.
  ``detector_snr = I / sqrt((2 q I + floor) B)``: the DC signal of the light as traced (a chopped source in its open state),
  over its noise in the detector's bandwidth. ``noise_equivalent_power_W_per_rtHz``: the floor over the light's responsivity
  ``I / P``.
* Lock-in (``LockIn``): the modulated current ``I_m = sum R P f`` (``f``: the share of the source's traced power that follows the
  modulation) and the steady current ``I_s = I - I_m``. ``SQUARE`` chopping of duty ``D``: the fundamental's peak amplitude is
  ``(2 / pi) sin(pi D) I_m``; ``SINE`` of depth ``m``: ``m I_m``. The lock-in output ``X`` is that fundamental's rms value,
  ``a1 / sqrt(2)``, with the reference in phase with it. Its noise: white current noise of density ``S^2`` passes the output
  filter's noise-equivalent bandwidth ``ENBW = Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n) tau)`` (n RC sections of time constant
  ``tau``: ``1 / (4 tau)`` for one), ``sigma_X = S sqrt(ENBW)``. The shot noise of a modulated current is weighted by the
  reference over a period: ``2 q (I_s + w I_m)`` with ``w = D + sin(2 pi D) / (2 pi)`` for SQUARE (``D`` itself at D = 1/2)
  and ``w = 1`` for SINE; the dark floor is steady. ``lock_in_snr = X / sigma_X``.
* The detector's response at the modulation frequency ``f``: a ``FIRST_ORDER_3DB`` response passes the fundamental with
  ``|H(f)| = 1 / sqrt(1 + (f / f_c)^2)`` (``detector_response_at_modulation``). The lock-in's signal and noise are stated at
  the detector's output in photocurrent units (the output over the DC gain): ``X |H(f)|`` and ``sigma_X |H(f)|``, every
  modelled noise (shot, dark, the load's thermal noise) entering with the photocurrent and passing the same response. A
  ``NOISE_EQUIVALENT`` bandwidth states no shape: ``|H(f)|`` is not known and the signal and noise are those at the
  detector's input.
* Conditions (``validity``):
  - the modulation lies within the detector's bandwidth, ``f <= f_c`` (``FIRST_ORDER_3DB``) or ``f <= 2 B / pi``
    (``NOISE_EQUIVALENT``: the -3 dB frequency of a single-pole response of that noise bandwidth, at or below that of any
    Butterworth response of it). Within it a single-pole, real-pole or Butterworth response loses at most 3 dB (a peaked
    response may lose more: its shape is not stated) and the SNR of the input-referred white noises
    holds; above it the attenuated signal meets noise added after the response (amplifier voltage noise, the lock-in's own
    input noise), which the model does not hold: the lock-in's noise and SNR are not stated
    (``MODULATION_ABOVE_DETECTOR_BANDWIDTH``), nor, without a stated response shape, its signal. A ``SQUARE`` chop above it
    never lets the detector reach the open state the photocurrent is stated for (its output follows the mean, ``I_s + D I_m``,
    with a residual ripple): the shot and total noise currents and ``detector_snr``, stated at the open state, are not stated
    either (``MODULATION_ABOVE_DETECTOR_BANDWIDTH``); the photocurrent (the open state's) and the noise-equivalent power (a
    ratio to it) stand;
  - the output filter averages over many periods -- ``f tau >= MIN_PERIODS_PER_TIME_CONSTANT`` -- or the lock-in's noise and
    SNR are not stated (``LOCK_IN_AVERAGING_TOO_SHORT``);
  - the photocurrent is linear in the power. A stated ``saturation_current_A`` bounds its peak (the open state of a chopped
    source, ``I + m I_m`` for a SINE of depth ``m``): above it every quantity that reads the photocurrent is not stated
    (``PHOTOCURRENT_ABOVE_LINEAR_RANGE``); without one no saturation is checked and the photocurrent's basis says so.
* Sampled light: where the plane's light holds the sampled rays of extended emitters (a sample's emission), each quantity that
  reads them carries ``sampling_relative_sigma``: its relative standard error from each sampled line's
  ``collection_fraction_sigma`` (``sigma_P = collection_fraction_sigma x emitted_W``), to first order,
  ``sum_k |d ln Q / d P_k| sigma_P_k`` (the lines' errors added linearly: the lines of one emitter are the same rays). Like that
  sigma it is an upper guide; targets are graded on the value, as every sampled power is.
* Not modelled: 1/f noise, internal gain and excess noise, ADC quantization, the shape of saturation, the settling of the
  output filter, harmonics other than the fundamental (the reference is a sine).

Every light-dependent quantity is computed from the traced light the plane counts wherever the plane has such a power: a
known one (``QuantityState`` ``KNOWN`` or ``KNOWN_ZERO``), and a lit plane whose power is ``BOUNDED`` or ``UNKNOWN`` (light
the trace does not count may add; part of a beam is taken before the plane; an annular beam counted whole on an active
area that may miss part of it). The number is the same arithmetic on the traced light, and where the plane's power is not
simply the model's value the quantity says what its number is (``DetectionQuantity.value_basis["value"]``,
``contracts/basis.py``): the plane's entry (``results.basis_of(plane, "total_power_W")``) merged with the detector's own
(``merge_bases``).

* ``detected_power_W``, ``photocurrent_A``, the shot and total noise currents, ``detector_snr``, ``modulated_photocurrent_A``,
  ``lock_in_signal_A`` and ``lock_in_noise_A`` rise with every watt on the plane (the responsivity is not negative; ``I /
  sqrt(a I + b)`` rises with ``I``), so they keep the kind of the plane's power: a ``LOWER_BOUND`` where light may add, an
  ``UPPER_BOUND`` where part of a beam is elsewhere. ``Omitted.upper`` (W) stays on ``detected_power_W`` only.
* ``noise_equivalent_power_W_per_rtHz`` (a ratio to the light's responsivity) is the model's value where the run launches
  one wavelength, else an ``ESTIMATE``; ``lock_in_snr`` is an ``ESTIMATE`` (light that adds raises it when modulated and
  lowers it when steady).
* The linear range. With a stated ``saturation_current_A`` and light that may add, the peak photocurrent is at most the
  traced peak plus ``q lambda_max / (h c)`` (a quantum efficiency of 1 at the longest launched wavelength) times the most
  that light adds (``power_upper_bound_W - total_power_W``), times ``1 + m`` for a SINE of depth ``m``. Where that upper
  value exceeds the saturation current, or nothing bounds the light, the quantities that read the photocurrent are
  ``ESTIMATE`` with the check ``peak_photocurrent_A`` (``within`` null: not decided). The same holds for the open-state
  noise and ``detector_snr`` of a SQUARE chop above the detector's bandwidth whose modulated light is not among the traced
  light and may be among the light that adds (check ``modulation_frequency_Hz``, ``within`` false).
* Target intervals (``Detected.bounds``). Where light may add by at most ``D = q lambda_max / (h c) x`` that power and the
  linear range holds for it, ``detector_snr`` lies between its traced value and its value at ``I + D``, and
  ``lock_in_snr`` between the traced signal over the noise with ``D`` taken as steady current and the value with ``D`` taken
  as modulated current (the reference weight ``w <= 1``); a target on either is graded on that interval. On a plane whose
  power is an upper end (a share taken before it, an annular beam counted whole) and whose fate is complete,
  ``detector_snr`` lies between the SNR of the least power the plane's entry allows (0 where that entry omits or assumes
  anything else) and its traced value -- with or without light left out as negligible, which only widens the upper end.

Withheld (``UNAVAILABLE`` with ``PLANE_POWER_NOT_KNOWN``): a plane no traced light of known power reaches while light may
arrive (not lit: ``UNKNOWN``, or ``BOUNDED`` from 0 with no value where only followed light may arrive), a power out
of model scope (``OUT_OF_MODEL_SCOPE``) and plane bounds that did not run (``NOT_COMPUTED``). A parameter the detector
does not state leaves the quantities that need it ``UNAVAILABLE``
(``PARAMETER_NOT_STATED``, with ``missing``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..contracts.basis import Omitted, ValidityCheck, ValueBasis, merge_bases
from ..contracts.detection import (
    Detection,
    DetectionContribution,
    DetectionQuantity,
    DetectionResult,
    DetectionValidity,
)
from ..contracts.results import ObservationResult
from ..errors.status import OmittedPart, QuantityState, ValueKind
from ..math.exact_trig import cis
from .superposition import received_power_W

#: Exact SI constants (2019 redefinition).
ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0
BOLTZMANN_J_PER_K = 1.380649e-23
#: The output filter must average over at least this many modulation periods per time constant for the stationary noise
#: model (``sigma_X = S sqrt(ENBW)``) to hold.
MIN_PERIODS_PER_TIME_CONSTANT = 10.0
#: Metres per nanometre.
M_PER_NM = 1e-9

MODELED, STATED, UNAVAILABLE = "MODELED", "EXTERNALLY_CHARACTERIZED", "UNAVAILABLE"
PARAMETER_NOT_STATED = "PARAMETER_NOT_STATED"
WAVELENGTH_OUTSIDE_QE_TABLE = "WAVELENGTH_OUTSIDE_QE_TABLE"
PLANE_POWER_NOT_KNOWN = "PLANE_POWER_NOT_KNOWN"
LOCK_IN_AVERAGING_TOO_SHORT = "LOCK_IN_AVERAGING_TOO_SHORT"
NO_SIGNAL_NO_NOISE = "NO_SIGNAL_NO_NOISE"
RESPONSE_SHAPE_NOT_STATED = "RESPONSE_SHAPE_NOT_STATED"
#: No light on the plane: a quantity referred to the light's responsivity has none to refer to.
NO_LIGHT_ON_PLANE = "NO_LIGHT_ON_PLANE"
#: The modulation frequency lies above the detector's bandwidth (module docstring).
MODULATION_ABOVE_DETECTOR_BANDWIDTH = "MODULATION_ABOVE_DETECTOR_BANDWIDTH"
#: The peak photocurrent lies above the detector's stated linear range (``electrical.saturation_current_A``).
PHOTOCURRENT_ABOVE_LINEAR_RANGE = "PHOTOCURRENT_ABOVE_LINEAR_RANGE"
#: ``2 / pi``: the -3 dB frequency of a single-pole response over its noise-equivalent bandwidth.
FIRST_ORDER_3DB_PER_NOISE_BANDWIDTH = 2.0 / math.pi

DETECTION_MODEL = ("Photodetector on the observation plane: photocurrent from the stated quantum efficiency (R = QE q lambda / (h c)), "
                   "white shot noise 2qI, a stated dark-noise floor (noise-equivalent power, or dark current shot noise and load-resistance "
                   "thermal noise 4kT/R) in the stated bandwidth; lock-in: rms fundamental of the modulated current over the noise in the "
                   "output filter's noise-equivalent bandwidth, reference in phase, the modulation within the detector's bandwidth; "
                   "the photocurrent linear in the power (up to a stated saturation current). No 1/f noise, gain, excess noise or ADC.")

#: Each detection quantity's unit and what it is computed from.
UNITS = {
    "detector_bandwidth_Hz": "Hz", "noise_equivalent_bandwidth_Hz": "Hz", "detected_power_W": "W", "photocurrent_A": "A",
    "dark_noise_density_A_per_rtHz": "A/sqrt(Hz)", "shot_noise_current_A": "A", "dark_noise_current_A": "A",
    "total_noise_current_A": "A", "detector_snr": "1", "noise_equivalent_power_W_per_rtHz": "W/sqrt(Hz)",
    "modulation_frequency_Hz": "Hz", "lock_in_enbw_Hz": "Hz", "modulated_photocurrent_A": "A", "lock_in_signal_A": "A",
    "lock_in_noise_A": "A", "lock_in_snr": "1", "detector_response_at_modulation": "1",
}
BASIS = {
    "detector_bandwidth_Hz": "the detector's stated bandwidth (electrical.bandwidth_Hz, as bandwidth_definition says)",
    "noise_equivalent_bandwidth_Hz": "pi / 2 x the -3 dB frequency of a first-order response, or the stated noise-equivalent bandwidth",
    "detected_power_W": "the power the plane counts: every arrival's received power and every counted sampled ray, all wavelengths",
    "photocurrent_A": "sum over the light of R(lambda) P, R = QE(lambda) q lambda / (h c)",
    "dark_noise_density_A_per_rtHz": "NEP x R(nep_wavelength_nm), or sqrt(2 q I_dark + 4 k T / R_load)",
    "shot_noise_current_A": "sqrt(2 q I B): the light's own shot noise in the noise-equivalent bandwidth",
    "dark_noise_current_A": "the dark-noise density x sqrt(B)",
    "total_noise_current_A": "sqrt((2 q I + dark density^2) B)",
    "detector_snr": "photocurrent / total noise current (amplitude ratio, not dB)",
    "noise_equivalent_power_W_per_rtHz": "the dark-noise density over the light's responsivity I / P",
    "modulation_frequency_Hz": "the stated modulation frequency (the lock-in's reference)",
    "lock_in_enbw_Hz": "Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n) tau): 1/(4 tau), 1/(8 tau), 3/(32 tau), 5/(64 tau) for n = 1..4",
    "modulated_photocurrent_A": "sum over the light of R P x the share following the modulation (open state of a chopper, mean of a sine)",
    "lock_in_signal_A": "rms of the fundamental: (2/pi) sin(pi D) I_m / sqrt(2) (SQUARE), m I_m / sqrt(2) (SINE)",
    "lock_in_noise_A": "sqrt((2 q (I_steady + w I_m) + dark density^2) ENBW), w = D + sin(2 pi D)/(2 pi) (SQUARE) or 1 (SINE)",
    "lock_in_snr": "lock-in signal / lock-in noise (X channel, reference in phase with the fundamental)",
    "detector_response_at_modulation": "1 / sqrt(1 + (f / f_c)^2): a first-order response at the modulation frequency",
}
#: Where the lock-in's signal and noise are stated (``BASIS`` suffixes).
SIGNAL_AT_OUTPUT = ("; at the detector's output in photocurrent units: times |H(f)| = 1 / sqrt(1 + (f / f_c)^2) of the stated "
                    "first-order response (the signal and the input-referred noises alike)")
SIGNAL_AT_INPUT = ("; at the detector's input: the response's shape at the modulation frequency is not stated (within the bandwidth "
                   "condition a single-pole or Butterworth response passes between 1 / sqrt(2) and 1 of it)")
SIGNAL_NO_RESPONSE = "; at the detector's input: the detector states no electrical response"
#: The photocurrent's basis where no saturation current is stated.
LINEAR_NOT_CHECKED = "; taken as linear in the power: no saturation current is stated, so none is checked"


#: ``Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n))`` for n = 1..4 RC sections, exact binary fractions (no libm: the same bits in every
#: runtime): the noise-equivalent bandwidth times the time constant.
ENBW_TIMES_TAU = {1: 1.0 / 4.0, 2: 1.0 / 8.0, 3: 3.0 / 32.0, 4: 5.0 / 64.0}


def lock_in_enbw_Hz(time_constant_s: float, filter_order: int) -> float:
    """The noise-equivalent bandwidth of ``filter_order`` identical RC sections of time constant ``time_constant_s``:
    ``int_0^inf (1 + (2 pi f tau)^2)^-n df = Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n) tau)`` (``ENBW_TIMES_TAU``)."""

    return ENBW_TIMES_TAU[filter_order] / time_constant_s


def square_fundamental(duty_cycle: float) -> float:
    """The peak amplitude of the fundamental of a 0/1 square wave of duty ``D``: ``(2 / pi) sin(pi D)`` (the sine from
    ``math.exact_trig``: the same bits in every runtime)."""

    return 2.0 / math.pi * cis(math.pi * duty_cycle).imag


def square_noise_weight(duty_cycle: float) -> float:
    """The shot-noise weight of a 0/1 square-wave current read by an in-phase sine reference of mean square 1:
    ``(1 / T) int_open 2 cos^2(omega t) dt = D + sin(2 pi D) / (2 pi)``."""

    return duty_cycle + cis(2.0 * math.pi * duty_cycle).imag / (2.0 * math.pi)


def responsivity_A_per_W(qe: float, wavelength_nm: float) -> float:
    """``QE q lambda / (h c)``."""

    return qe * ELEMENTARY_CHARGE_C * wavelength_nm * M_PER_NM / (PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S)


def noise_bandwidth_Hz(electrical) -> float:
    """The noise-equivalent bandwidth of a stated electrical response: ``pi f_c / 2`` for a first-order -3 dB frequency, else the
    stated value."""

    return math.pi / 2.0 * electrical.bandwidth_Hz if electrical.bandwidth_definition == "FIRST_ORDER_3DB" else electrical.bandwidth_Hz


def dark_floor(detector) -> tuple[float | None, str | None]:
    """``(the dark-noise current density in A/sqrt(Hz), why it is not known)`` of a detector with an electrical response:
    ``NEP R(nep_wavelength_nm)``, or ``sqrt(2 q I_d + 4 k T / R_L)``."""

    electrical = detector.electrical
    if electrical.nep_W_per_rtHz is not None:
        qe = _qe(detector, electrical.nep_wavelength_nm)
        if qe is None:
            return None, PARAMETER_NOT_STATED if detector.quantum_efficiency is None else WAVELENGTH_OUTSIDE_QE_TABLE
        return electrical.nep_W_per_rtHz * responsivity_A_per_W(qe, electrical.nep_wavelength_nm), None
    variance = 2.0 * ELEMENTARY_CHARGE_C * electrical.dark_current_A
    if electrical.load_resistance_ohm is not None:
        variance += 4.0 * BOLTZMANN_J_PER_K * electrical.temperature_K / electrical.load_resistance_ohm
    return math.sqrt(variance), None


def _qe(detector, wavelength_nm: float) -> float | None:
    table = detector.quantum_efficiency
    if table is None:
        return None
    wavelengths = [row.wavelength_nm for row in table.rows]
    if wavelength_nm < wavelengths[0] or wavelength_nm > wavelengths[-1]:
        return None
    if len(wavelengths) == 1:
        return table.rows[0].qe
    for (w0, row0), (w1, row1) in zip(zip(wavelengths, table.rows), zip(wavelengths[1:], table.rows[1:])):
        if w0 <= wavelength_nm <= w1:
            return row0.qe + (row1.qe - row0.qe) * (wavelength_nm - w0) / (w1 - w0)
    return None  # pragma: no cover


def _contributions(obs: ObservationResult) -> list[tuple[str, float, float]]:
    """``(source_id, wavelength_nm, power_W)`` of the light the plane counts, one row per source and wavelength, in a stable order."""

    powers: dict[tuple[str, float], list[float]] = defaultdict(list)
    for arrival in obs.arrivals:
        powers[(arrival.beam.source_id, arrival.beam.wavelength_nm)].append(received_power_W(arrival))
    for line in obs.sampled_light:
        powers[(line.source_id, line.wavelength_nm)].append(line.power_W)
    return [(source, wavelength, math.fsum(values)) for (source, wavelength), values in sorted(powers.items())]


def _quantity(name: str, value: float, *, stated: bool = False, basis: str | None = None,
              sigma: float | None = None, entry: ValueBasis | None = None) -> DetectionQuantity:
    return DetectionQuantity(name=name, status=STATED if stated else MODELED, unit=UNITS[name], value=value,
                             basis=BASIS[name] if basis is None else basis,
                             sampling_relative_sigma=None if sigma is None or value == 0.0 else sigma,
                             value_basis=None if entry is None else {"value": entry})


def _missing(name: str, reason: str, missing: tuple[str, ...] = ()) -> DetectionQuantity:
    return DetectionQuantity(name=name, status=UNAVAILABLE, unit=UNITS[name], basis=BASIS[name], reason=reason, missing=missing)


def _row_sigmas(obs: ObservationResult) -> dict[tuple[str, float], float]:
    """Per (source, wavelength) of sampled light on the plane: the standard error of its counted power,
    ``collection_fraction_sigma x emitted_W`` summed over its lines."""

    sigmas: dict[tuple[str, float], list[float]] = defaultdict(list)
    for line in obs.sampled_light:
        if not line.crossing_count:  # (a sampled zero: no ray crosses the plane, nothing of it is read)
            continue
        sigmas[(line.source_id, line.wavelength_nm)].append(line.collection_fraction_sigma * line.emitted_W)
    return {key: math.fsum(values) for key, values in sigmas.items()}


def _relative_sigma(gradients: list[float], sigmas: list[float]) -> float:
    """``sum_k |d ln Q / d P_k| sigma_k`` (module docstring)."""

    return math.fsum(abs(g) * s for g, s in zip(gradients, sigmas))


#: The plane power states under which a lit plane's detector reads its traced light with a value basis (module docstring).
QUALIFIED_STATES = frozenset({QuantityState.BOUNDED, QuantityState.UNKNOWN})
#: The validity checks of a photocurrent whose light may add (module docstring).
PEAK_PHOTOCURRENT = "peak_photocurrent_A"
MODULATION_FREQUENCY = "modulation_frequency_Hz"
#: The quantities that read the photocurrent, and those of them stated at a chopped source's open state.
PHOTOCURRENT_QUANTITIES = frozenset({"photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr",
                                     "noise_equivalent_power_W_per_rtHz", "modulated_photocurrent_A", "lock_in_signal_A",
                                     "lock_in_noise_A", "lock_in_snr"})
OPEN_STATE_QUANTITIES = frozenset({"shot_noise_current_A", "total_noise_current_A", "detector_snr"})
#: The quantities that read the plane's light (its power, its photocurrent, or both): a model-scope finding on that light
#: qualifies them (``results.basis.scoped_observation``); the bandwidths, the dark noise and the modulation's own numbers do
#: not read it.
LIGHT_QUANTITIES = PHOTOCURRENT_QUANTITIES | {"detected_power_W"}
#: The quantities that are not monotone in the light on the plane: a ratio to its responsivity, and the lock-in's SNR.
NOT_MONOTONE = frozenset({"noise_equivalent_power_W_per_rtHz", "lock_in_snr"})


@dataclass(frozen=True)
class Detected:
    """A plane's detection and, per target metric, the interval the quantity lies in where one is proven (module docstring).
    ``open_checks``: per quantity, the validity conditions of the detector model that the light which may add leaves open
    (the peak photocurrent against the linear range, a chop above the bandwidth); on a plane of the model's value they are
    in no entry (the quantities are the model's values, decision W28), so the targets read them here."""

    result: DetectionResult
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    open_checks: dict[str, tuple[ValidityCheck, ...]] = field(default_factory=dict)
    #: per quantity graded with its entry, on a plane whose power is an upper end with light left out as negligible: how far
    #: that light may move the value either way (``None``: nothing bounds it, or a check it may leave open applies). Not an
    #: interval a target is graded on -- the stated truncation its entry carries.
    reach: dict[str, float | None] = field(default_factory=dict)


def reads_traced_light(obs: ObservationResult, state: QuantityState | None) -> bool:
    """Whether the detector on ``obs`` has a number to read: a known power, or the traced light of known power on a lit plane
    whose power is ``BOUNDED`` or ``UNKNOWN``."""

    if state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO):
        return True
    return state in QUALIFIED_STATES and obs.power_lit and obs.total_power_W is not None


def _annular_W(obs: ObservationResult) -> float:
    """The most the annular beams counted whole on the active area may miss of what they read (W)."""

    return math.fsum(received_power_W(a) - a.detected_power_min_W for a in obs.arrivals
                     if a.detected_power_min_W is not None and a.detected_power_min_W < received_power_W(a))


def power_basis(obs: ObservationResult, state: QuantityState | None) -> ValueBasis | None:
    """What the power the detector reads is: the plane's entry for its ``total_power_W`` in ``state`` merged with the share
    annular beams counted whole on the active area may miss (``None``: the model's value)."""

    from ..results.basis import basis_of  # (the results package reads the validation package)

    plane = basis_of(obs if obs.quantity_state is state else obs.model_copy(update={"quantity_state": state}), "total_power_W")
    annular = _annular_W(obs)
    own = None if annular <= 0.0 else ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(
        part=OmittedPart.BEAM_PART_ELSEWHERE, upper=annular, setting="capture_min_power_fraction"),))
    return merge_bases(plane, own)


def _derived(entry: ValueBasis | None, *, monotone: bool, checks: tuple[ValidityCheck, ...] = ()) -> ValueBasis | None:
    """The entry of a quantity computed from a power whose entry is ``entry``: its parts without their amounts (they are in
    W), its kind where the quantity rises with the power (``monotone``), ``ESTIMATE`` where it does not or where one of
    ``checks`` (validity conditions the light that may add leaves open) applies."""

    if entry is None:
        return None
    parts: list[Omitted] = []
    for o in entry.omitted:
        part = Omitted(part=o.part, at=o.at, setting=o.setting, code=o.code)
        if part not in parts:
            parts.append(part)
    plain = ValueBasis(kind=entry.kind, omitted=tuple(parts), assumed=entry.assumed, validity=entry.validity)
    if checks:
        return ValueBasis(kind=ValueKind.ESTIMATE, omitted=plain.omitted, assumed=plain.assumed, validity=(*plain.validity, *checks))
    return merge_bases(plain, monotone="increasing" if monotone else None)


def detection_result(obs: ObservationResult, spec: Detection, state: QuantityState | None) -> DetectionResult:
    """The detection of ``obs``'s light by ``spec`` (module docstring); ``state`` is the plane's power state."""

    return detect(obs, spec, state).result


def detect(obs: ObservationResult, spec: Detection, state: QuantityState | None, left_out_W: float | None = None) -> Detected:
    """``detection_result`` with the proven intervals of the target metrics (``Detected``). ``left_out_W``: on a plane whose
    power is the model's value (``KNOWN`` / ``KNOWN_ZERO``), the most the light left out as negligible may add to it (the
    stated truncation; ``inf`` where nothing bounds it): the target metrics' intervals count it as light that may add and
    the validity conditions it may leave open are checked for it (decision W28), the quantities themselves stay the
    model's values."""

    detector, lock_in = spec.detector, spec.lock_in
    electrical = detector.electrical
    out: dict[str, DetectionQuantity] = {}
    validity: list[DetectionValidity] = []
    light_names = ("detected_power_W", "photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr",
                   "noise_equivalent_power_W_per_rtHz")
    lock_names = ("modulated_photocurrent_A", "lock_in_signal_A", "lock_in_noise_A", "lock_in_snr")

    # ---- stated parameters and what follows from them alone
    noise_bandwidth = floor_density = None
    if electrical is None:
        for name in ("detector_bandwidth_Hz", "noise_equivalent_bandwidth_Hz", "dark_noise_density_A_per_rtHz", "dark_noise_current_A"):
            out[name] = _missing(name, PARAMETER_NOT_STATED, ("electrical",))
    else:
        out["detector_bandwidth_Hz"] = _quantity("detector_bandwidth_Hz", electrical.bandwidth_Hz, stated=True)
        noise_bandwidth = noise_bandwidth_Hz(electrical)
        out["noise_equivalent_bandwidth_Hz"] = _quantity("noise_equivalent_bandwidth_Hz", noise_bandwidth)
        floor_density, floor_reason = dark_floor(detector)
        if floor_density is None:
            for name in ("dark_noise_density_A_per_rtHz", "dark_noise_current_A"):
                out[name] = _missing(name, floor_reason or PARAMETER_NOT_STATED, ("quantum_efficiency",))
        else:
            out["dark_noise_density_A_per_rtHz"] = _quantity("dark_noise_density_A_per_rtHz", floor_density)
            out["dark_noise_current_A"] = _quantity("dark_noise_current_A", floor_density * math.sqrt(noise_bandwidth))

    # ---- the light
    contributions: list[DetectionContribution] = []
    fractions: dict[str, float] = {}
    if lock_in is not None:
        fractions = {s.source_id: s.fraction for s in lock_in.modulation.sources}
    known = reads_traced_light(obs, state)
    light = power_basis(obs, state) if known else None  # (what the power the detector reads is; None: the model's value)
    # light the plane's power does not count may add to it, by at most ``add_W`` (None: nothing bounds it)
    may_add = light is not None and light.kind in (ValueKind.LOWER_BOUND, ValueKind.ESTIMATE) and not obs.power_complete
    add_W = None if not may_add or obs.power_upper_bound_W is None or obs.total_power_W is None else max(
        obs.power_upper_bound_W - obs.total_power_W, 0.0)
    # (a plane of the model's value whose truncation is stated: that light may add to the targets' intervals only; with no
    # bound, ``add_W`` stays None and the conditions it may leave open are open)
    faint = known and light is None and left_out_W is not None and 0.0 < left_out_W
    # (a plane whose power is an upper end -- a share taken before it, an annular beam -- whose fate is complete: the
    # photocurrent's SNR is graded on the interval from the least power the entry allows to the traced value,
    # ``Detected.bounds``; where its left-out light is stated, that light may add too, up to the most it brings, decision W28)
    upper = known and light is not None and light.kind is ValueKind.UPPER_BOUND and obs.power_complete
    capped = upper and left_out_W is not None and 0.0 < left_out_W
    if faint or capped:
        add_W = left_out_W if math.isfinite(left_out_W) else None  # type: ignore[arg-type]
    launched = [s.wavelength_nm for s in obs.power_by_wavelength]
    # (the photocurrent that light adds at most: a quantum efficiency of 1 at the longest launched wavelength)
    add_A = None if add_W is None or not launched else responsivity_A_per_W(1.0, max(launched)) * add_W
    checks: dict[str, tuple[ValidityCheck, ...]] = {}
    bounds: dict[str, tuple[float, float]] = {}
    reach: dict[str, float | None] = {}

    def entry_of(name: str) -> ValueBasis | None:
        if name == "detected_power_W":
            return light
        if name == "noise_equivalent_power_W_per_rtHz" and len(set(launched)) == 1 and not checks.get(name):
            return None  # (one wavelength: the ratio to its responsivity does not depend on the amount of light)
        return _derived(light, monotone=name not in NOT_MONOTONE, checks=checks.get(name, ()))

    current = modulated = power = None
    light_reason: tuple[str, tuple[str, ...]] | None = None
    row_sigmas = _row_sigmas(obs)
    sampled = bool(row_sigmas)
    responsivities: list[float | None] = []
    sigmas: list[float] = []
    reads: list[bool] = []  # (per row: whether it holds sampled light)
    if not known:
        light_reason = (PLANE_POWER_NOT_KNOWN, ())
    else:
        rows = _contributions(obs)
        power = math.fsum(p for _, _, p in rows)
        for source_id, wavelength, row_power in rows:
            qe = _qe(detector, wavelength)
            r = None if qe is None else responsivity_A_per_W(qe, wavelength)
            responsivities.append(r)
            sigmas.append(row_sigmas.get((source_id, wavelength), 0.0))
            reads.append((source_id, wavelength) in row_sigmas)
            contributions.append(DetectionContribution(source_id=source_id, wavelength_nm=wavelength, power_W=row_power,
                                                       modulated_fraction=fractions.get(source_id, 0.0), responsivity_A_per_W=r))
        if detector.quantum_efficiency is None:
            light_reason = (PARAMETER_NOT_STATED, ("quantum_efficiency",))
        elif any(r is None and c.power_W > 0.0 for r, c in zip(responsivities, contributions)):
            light_reason = (WAVELENGTH_OUTSIDE_QE_TABLE, ("quantum_efficiency",))
        else:
            current = math.fsum((r or 0.0) * c.power_W for r, c in zip(responsivities, contributions))
            modulated = math.fsum((r or 0.0) * c.power_W * c.modulated_fraction for r, c in zip(responsivities, contributions))
    resp = [r or 0.0 for r in responsivities]
    mods = [c.modulated_fraction for c in contributions]

    def sampled_sigma(gradients: list[float] | None) -> float | None:
        """The relative sampling sigma of a quantity of these log-gradients (``None`` where the plane has no sampled light, or
        where the quantity reads none of it: every sampled row's gradient 0)."""
        if not sampled or gradients is None or all(g == 0.0 for g, read in zip(gradients, reads) if read):
            return None
        return _relative_sigma(gradients, sigmas)

    if power is None:
        out["detected_power_W"] = _missing("detected_power_W", PLANE_POWER_NOT_KNOWN)
    else:
        out["detected_power_W"] = _quantity("detected_power_W", power, entry=entry_of("detected_power_W"),
                                            sigma=sampled_sigma([1.0 / power] * len(sigmas)) if power > 0.0 else None)

    # ---- the linear range: the peak photocurrent against a stated saturation current
    saturation = None if electrical is None else electrical.saturation_current_A
    linear = True
    if current is not None and saturation is not None:
        peak = current
        if lock_in is not None and lock_in.modulation.waveform == "SINE":
            peak = current + lock_in.modulation.modulation_depth * modulated  # type: ignore[operator]
        linear = peak <= saturation
        validity.append(DetectionValidity(
            parameter="peak_photocurrent_within_linear_range", value=peak, limit=saturation, comparison="AT_MOST", within=linear,
            meaning="the photocurrent at its peak (a chopped source open, a sine at its crest) stays within the detector's stated "
                    "linear range (electrical.saturation_current_A): the photocurrent, its noise and the lock-in's signal follow "
                    "the power"))
        if linear and (may_add or faint or capped):
            # (the peak with the light that may add: within the linear range too, or the photocurrent's quantities are estimates;
            # on a plane of the model's value, its targets are not decided by the linear model, ``Detected.open_checks``)
            depth = lock_in.modulation.modulation_depth if lock_in is not None and lock_in.modulation.waveform == "SINE" else 0.0
            peak_upper = None if add_A is None else peak + (1.0 + depth) * add_A  # type: ignore[operator]
            if peak_upper is None or peak_upper > saturation:
                open_range = ValidityCheck(parameter=PEAK_PHOTOCURRENT, unit="A", maximum=saturation, observed=peak_upper,
                                           within=None)
                checks.update({name: (open_range,) for name in PHOTOCURRENT_QUANTITIES})
    note = _sampling_note(obs)
    if saturation is None:
        note += LINEAR_NOT_CHECKED
    # a SQUARE chop above the detector's bandwidth (module docstring): the open-state noise and SNR are not the detector's
    chopped_above_band = False
    if lock_in is not None and electrical is not None and lock_in.modulation.waveform == "SQUARE" and modulated is not None and modulated > 0.0:
        first_order = electrical.bandwidth_definition == "FIRST_ORDER_3DB"
        limit = electrical.bandwidth_Hz if first_order else FIRST_ORDER_3DB_PER_NOISE_BANDWIDTH * electrical.bandwidth_Hz
        chopped_above_band = lock_in.modulation.frequency_Hz > limit
    elif (lock_in is not None and electrical is not None and lock_in.modulation.waveform == "SQUARE" and modulated is not None
          and (may_add or faint or capped) and any(fraction > 0.0 for fraction in fractions.values())):
        # (no modulated light among the traced light, and light that may add may be modulated: the open-state values are
        # the detector's only if it is not)
        first_order = electrical.bandwidth_definition == "FIRST_ORDER_3DB"
        limit = electrical.bandwidth_Hz if first_order else FIRST_ORDER_3DB_PER_NOISE_BANDWIDTH * electrical.bandwidth_Hz
        if lock_in.modulation.frequency_Hz > limit:
            above = ValidityCheck(parameter=MODULATION_FREQUENCY, unit="Hz", maximum=limit, observed=lock_in.modulation.frequency_Hz,
                                  within=False)
            checks.update({name: (*checks.get(name, ()), above) for name in OPEN_STATE_QUANTITIES})
    # (a proven interval of a target metric: light adds at most ``add_A``, and no validity condition is left open by it)
    proven = add_A is not None and (faint or (light is not None and light.kind is ValueKind.LOWER_BOUND)) and not checks
    if current is None or not linear:
        reason, missing = light_reason if current is None else (PHOTOCURRENT_ABOVE_LINEAR_RANGE, ())  # type: ignore[misc]
        for name in light_names[1:]:
            out[name] = _missing(name, reason, missing)
    else:
        log_i = [r / current for r in resp] if current > 0.0 else None
        out["photocurrent_A"] = _quantity("photocurrent_A", current, basis=BASIS["photocurrent_A"] + note, sigma=sampled_sigma(log_i),
                                          entry=entry_of("photocurrent_A"))
        if noise_bandwidth is not None and chopped_above_band:
            for name in ("shot_noise_current_A", "total_noise_current_A", "detector_snr"):
                out[name] = _missing(name, MODULATION_ABOVE_DETECTOR_BANDWIDTH)
            if floor_density is None:
                out["noise_equivalent_power_W_per_rtHz"] = _missing("noise_equivalent_power_W_per_rtHz",
                                                                    out["dark_noise_density_A_per_rtHz"].reason or PARAMETER_NOT_STATED,
                                                                    ("quantum_efficiency",))
            else:
                out["noise_equivalent_power_W_per_rtHz"] = (
                    _quantity("noise_equivalent_power_W_per_rtHz", floor_density * power / current,  # type: ignore[operator]
                              sigma=sampled_sigma([1.0 / power - g for g in log_i]),  # type: ignore[operator,union-attr]
                              entry=entry_of("noise_equivalent_power_W_per_rtHz"))
                    if current > 0.0 else _missing("noise_equivalent_power_W_per_rtHz", NO_LIGHT_ON_PLANE))
        elif noise_bandwidth is None:
            for name in light_names[2:]:
                out[name] = _missing(name, PARAMETER_NOT_STATED, ("electrical",))
        else:
            shot = math.sqrt(2.0 * ELEMENTARY_CHARGE_C * current * noise_bandwidth)
            out["shot_noise_current_A"] = _quantity("shot_noise_current_A", shot, entry=entry_of("shot_noise_current_A"),
                                                    sigma=sampled_sigma(None if log_i is None else [g / 2.0 for g in log_i]))
            if floor_density is None:
                for name in light_names[3:]:
                    out[name] = _missing(name, out["dark_noise_density_A_per_rtHz"].reason or PARAMETER_NOT_STATED, ("quantum_efficiency",))
            else:
                density2 = 2.0 * ELEMENTARY_CHARGE_C * current + floor_density * floor_density
                total = math.sqrt(density2 * noise_bandwidth)
                log_total = [ELEMENTARY_CHARGE_C * r / density2 for r in resp] if density2 > 0.0 else None
                out["total_noise_current_A"] = _quantity("total_noise_current_A", total, sigma=sampled_sigma(log_total),
                                                         entry=entry_of("total_noise_current_A"))
                out["detector_snr"] = (
                    _quantity("detector_snr", current / total, entry=entry_of("detector_snr"),
                              sigma=sampled_sigma(None if log_i is None or log_total is None
                                                  else [a - b for a, b in zip(log_i, log_total)]))
                    if total > 0.0 else _missing("detector_snr", NO_SIGNAL_NO_NOISE))
                if proven and total > 0.0:
                    most = current + add_A  # type: ignore[operator]
                    bounds["detector_snr"] = (current / total, most / math.sqrt(
                        (2.0 * ELEMENTARY_CHARGE_C * most + floor_density * floor_density) * noise_bandwidth))
                elif upper and total > 0.0:
                    # (the SNR rises with the photocurrent: from the least current the entry allows -- the shares taken
                    # before the plane and the annular beams' misses removed at the largest responsivity, 0 where the entry
                    # omits anything else -- to the traced value, or, where light was left out as negligible, to the most
                    # that light adds; no upper end where nothing bounds that light or a validity condition it may leave
                    # open applies)
                    deficit = _upper_deficit_W(obs, light)  # type: ignore[arg-type]
                    least = 0.0 if deficit is None else max(0.0, current - max(resp, default=0.0) * deficit)
                    floor2 = floor_density * floor_density
                    low = least / math.sqrt((2.0 * ELEMENTARY_CHARGE_C * least + floor2) * noise_bandwidth) if least > 0.0 else 0.0
                    if not capped:
                        high = current / total
                    elif add_A is None or checks.get("detector_snr"):
                        high = math.inf
                    else:
                        most = current + add_A
                        high = most / math.sqrt((2.0 * ELEMENTARY_CHARGE_C * most + floor2) * noise_bandwidth)
                    bounds["detector_snr"] = (min(low, current / total), max(high, current / total))
                out["noise_equivalent_power_W_per_rtHz"] = (
                    _quantity("noise_equivalent_power_W_per_rtHz", floor_density * power / current,  # type: ignore[operator]
                              sigma=sampled_sigma([1.0 / power - g for g in log_i]),  # type: ignore[operator,union-attr]
                              entry=entry_of("noise_equivalent_power_W_per_rtHz"))
                    if current > 0.0 else _missing("noise_equivalent_power_W_per_rtHz", NO_LIGHT_ON_PLANE))

    # ---- the lock-in
    if lock_in is not None:
        modulation = lock_in.modulation
        f, tau = modulation.frequency_Hz, lock_in.time_constant_s
        enbw = lock_in_enbw_Hz(tau, lock_in.filter_order)
        out["modulation_frequency_Hz"] = _quantity("modulation_frequency_Hz", f, stated=True)
        out["lock_in_enbw_Hz"] = _quantity("lock_in_enbw_Hz", enbw)
        response = None  # (|H(f)| where the response's shape is stated)
        within_band = True
        if electrical is not None:
            first_order = electrical.bandwidth_definition == "FIRST_ORDER_3DB"
            band_limit = electrical.bandwidth_Hz if first_order else FIRST_ORDER_3DB_PER_NOISE_BANDWIDTH * electrical.bandwidth_Hz
            within_band = f <= band_limit
            validity.append(DetectionValidity(
                parameter="modulation_frequency_within_detector_bandwidth", value=f, limit=band_limit, comparison="AT_MOST",
                within=within_band,
                meaning="the modulation lies within the detector's bandwidth (the -3 dB frequency of a first-order response, or "
                        "2/pi of a stated noise-equivalent bandwidth): a single-pole, real-pole or Butterworth response loses at most 3 dB there "
                        "and the input-referred "
                        "white noises set the lock-in's noise; above it, noise added after the response is not modelled"))
            if first_order:
                response = 1.0 / math.sqrt(1.0 + (f / electrical.bandwidth_Hz) * (f / electrical.bandwidth_Hz))
        periods = f * tau
        averaging = periods >= MIN_PERIODS_PER_TIME_CONSTANT
        validity.append(DetectionValidity(
            parameter="modulation_periods_per_time_constant", value=periods, limit=MIN_PERIODS_PER_TIME_CONSTANT,
            comparison="AT_LEAST", within=averaging,
            meaning="the output filter averages over many modulation periods: its noise is the input noise density times the "
                    "square root of its noise-equivalent bandwidth, and the ripple at twice the frequency is filtered out"))
        if response is not None:
            out["detector_response_at_modulation"] = _quantity("detector_response_at_modulation", response)
        else:
            out["detector_response_at_modulation"] = _missing(
                "detector_response_at_modulation", PARAMETER_NOT_STATED if electrical is None else RESPONSE_SHAPE_NOT_STATED,
                ("electrical",) if electrical is None else ("electrical.bandwidth_definition",))
        if modulated is None or current is None or not linear:
            reason, missing = ((PHOTOCURRENT_ABOVE_LINEAR_RANGE, ()) if current is not None and not linear
                               else light_reason or (PARAMETER_NOT_STATED, ()))  # type: ignore[misc]
            for name in lock_names:
                out[name] = _missing(name, reason, missing)
        else:
            if modulation.waveform == "SQUARE":
                amplitude = square_fundamental(modulation.duty_cycle) * modulated  # type: ignore[arg-type]
                weight = square_noise_weight(modulation.duty_cycle)  # type: ignore[arg-type]
            else:
                amplitude = modulation.modulation_depth * modulated  # type: ignore[operator]
                weight = 1.0
            log_m = [m * r / modulated for m, r in zip(mods, resp)] if modulated > 0.0 else None
            out["modulated_photocurrent_A"] = _quantity("modulated_photocurrent_A", modulated, sigma=sampled_sigma(log_m),
                                                        entry=entry_of("modulated_photocurrent_A"))
            signal = amplitude / math.sqrt(2.0)  # (at the detector's input)
            if response is not None:
                out["lock_in_signal_A"] = _quantity("lock_in_signal_A", signal * response, sigma=sampled_sigma(log_m),
                                                    basis=BASIS["lock_in_signal_A"] + SIGNAL_AT_OUTPUT, entry=entry_of("lock_in_signal_A"))
            elif electrical is not None and not within_band:
                out["lock_in_signal_A"] = _missing("lock_in_signal_A", MODULATION_ABOVE_DETECTOR_BANDWIDTH,
                                                   ("electrical.bandwidth_definition",))
            else:
                out["lock_in_signal_A"] = _quantity("lock_in_signal_A", signal, sigma=sampled_sigma(log_m),
                                                    basis=BASIS["lock_in_signal_A"] + (SIGNAL_AT_INPUT if electrical is not None
                                                                                       else SIGNAL_NO_RESPONSE),
                                                    entry=entry_of("lock_in_signal_A"))
            if floor_density is None:
                missing_reason = out["dark_noise_density_A_per_rtHz"].reason or PARAMETER_NOT_STATED
                for name in lock_names[2:]:
                    out[name] = _missing(name, missing_reason, ("electrical",) if electrical is None else ("quantum_efficiency",))
            elif not within_band:
                for name in lock_names[2:]:
                    out[name] = _missing(name, MODULATION_ABOVE_DETECTOR_BANDWIDTH)
            elif not averaging:
                for name in lock_names[2:]:
                    out[name] = _missing(name, LOCK_IN_AVERAGING_TOO_SHORT)
            else:
                steady = current - modulated
                density2 = 2.0 * ELEMENTARY_CHARGE_C * (steady + weight * modulated) + floor_density * floor_density
                noise = math.sqrt(density2 * enbw)  # (at the detector's input)
                log_noise = ([ELEMENTARY_CHARGE_C * r * (1.0 - m + weight * m) / density2 for m, r in zip(mods, resp)]
                             if density2 > 0.0 else None)
                out["lock_in_noise_A"] = _quantity(
                    "lock_in_noise_A", noise if response is None else noise * response, sigma=sampled_sigma(log_noise),
                    basis=BASIS["lock_in_noise_A"] + (SIGNAL_AT_INPUT if response is None else SIGNAL_AT_OUTPUT),
                    entry=entry_of("lock_in_noise_A"))
                # (the ratio of the input-referred values: the response divides out)
                out["lock_in_snr"] = (
                    _quantity("lock_in_snr", signal / noise, entry=entry_of("lock_in_snr"),
                              sigma=sampled_sigma(None if log_m is None or log_noise is None
                                                  else [a - b for a, b in zip(log_m, log_noise)]))
                    if noise > 0.0 else _missing("lock_in_snr", NO_SIGNAL_NO_NOISE))
                if proven and noise > 0.0:
                    # the light that adds as steady current (the most noise, no signal), and as modulated current (w <= 1)
                    per_ampere = (square_fundamental(modulation.duty_cycle) if modulation.waveform == "SQUARE"  # type: ignore[arg-type]
                                  else modulation.modulation_depth) / math.sqrt(2.0)  # type: ignore[operator]
                    floor2 = floor_density * floor_density
                    least = signal / math.sqrt((2.0 * ELEMENTARY_CHARGE_C * (steady + weight * modulated + add_A) + floor2) * enbw)  # type: ignore[operator]
                    most = per_ampere * (modulated + add_A) / math.sqrt(  # type: ignore[operator]
                        (2.0 * ELEMENTARY_CHARGE_C * (steady + weight * (modulated + add_A)) + floor2) * enbw)  # type: ignore[operator]
                    bounds["lock_in_snr"] = (least, most)
                elif capped and noise > 0.0:
                    # (an upper-end plane with light left out as negligible: the target is graded with its entry, which
                    # carries how far that light may move the value -- as steady current down to ``least``, as modulated
                    # current up to ``most``, the closed forms above at the traced currents)
                    if add_A is None or checks.get("lock_in_snr"):
                        reach["lock_in_snr"] = None
                    else:
                        per_ampere = (square_fundamental(modulation.duty_cycle) if modulation.waveform == "SQUARE"  # type: ignore[arg-type]
                                      else modulation.modulation_depth) / math.sqrt(2.0)  # type: ignore[operator]
                        floor2 = floor_density * floor_density
                        least = signal / math.sqrt((2.0 * ELEMENTARY_CHARGE_C * (steady + weight * modulated + add_A) + floor2) * enbw)
                        most = per_ampere * (modulated + add_A) / math.sqrt(
                            (2.0 * ELEMENTARY_CHARGE_C * (steady + weight * (modulated + add_A)) + floor2) * enbw)
                        snr = signal / noise
                        reach["lock_in_snr"] = max(snr - least, most - snr)
    order = [name for name in UNITS if name in out]
    return Detected(DetectionResult(model=DETECTION_MODEL, power_state=state, contributions=tuple(contributions),
                                    quantities=tuple(out[name] for name in order), validity=tuple(validity)), bounds, checks,
                    reach)


def _upper_deficit_W(obs: ObservationResult, light: ValueBasis) -> float | None:
    """The most the power a detector reads on a plane whose power is an upper end (``light``) may lie below it (W): the
    shares taken before the plane (``total_power_W - power_lower_bound_W``) and what annular beams counted whole may miss
    (``_annular_W``). ``None`` where the entry also omits or assumes anything else (nothing bounds the deficit)."""

    if light.assumed or any(o.part is not OmittedPart.BEAM_PART_ELSEWHERE for o in light.omitted):
        return None
    taken = (obs.total_power_W - obs.power_lower_bound_W
             if obs.total_power_W is not None and obs.power_lower_bound_W is not None else 0.0)
    return max(taken, 0.0) + _annular_W(obs)


def _sampling_note(obs: ObservationResult) -> str:
    """What makes the plane's power a sampled value, for the photocurrent's basis."""

    notes = []
    pupil = [a.power_sampling_resolution_W for a in obs.arrivals if a.power_sampling_resolution_W is not None]
    if pupil:
        notes.append(f"a beam's power is sampled by its pupil rays (within {math.fsum(pupil):.4g} W)")
    zero = [line for line in obs.sampled_light if not line.counted_crossing_count]
    if zero:
        notes.append(f"no ray of {len(zero)} sampled line(s) is counted on the plane, a sampled zero within one ray's weight each")
    if any(line.counted_crossing_count for line in obs.sampled_light):
        notes.append("sampled rays of extended emitters: their collection fractions carry the sampling estimates of sampled_light")
    return "" if not notes else "; " + "; ".join(notes)


#: Target metrics read from a plane's detection, with their units.
DETECTION_METRICS = {"detector_bandwidth_Hz": "Hz", "detector_snr": "1", "lock_in_snr": "1"}
