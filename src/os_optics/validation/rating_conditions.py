"""The conditions of a power rating against the light on a surface (``RatingCondition``) and what the rating then says about
the damage threshold of that light (``RatingComparison.applicability`` and the comparison's status).

A rating (laser-induced damage threshold) was measured under conditions: a regime, a wavelength, for pulses a duration, a
repetition rate and a pulse count (read as the number of pulses each tested site received), a test beam of some diameter
on the sample, a polarization. Each condition is recorded once per comparison:

* ``VERIFIED``: the rating states it and the light meets it within its tolerance;
* ``DIFFERS``: the light is known to differ, with the effect on the threshold -- ``THRESHOLD_AT_LEAST_RATED`` (the
  threshold for this light is at least the rated value: an exposure within the rating is within it),
  ``THRESHOLD_AT_MOST_RATED`` (at most the rated value: an exposure above the rating is above it) or
  ``NO_ESTABLISHED_RELATIONSHIP``;
* ``NOT_VERIFIABLE``: it cannot be checked, with the reason;
* ``NOT_STATED``: the condition does not apply to the rating (a pulse duration, rate or count to a CW rating, an area
  definition to a ``W/cm`` value or to a typical value that states none): never counted as verified, never keeps the
  rating from applying. A test condition
  that does apply but that the rating does not state -- its test beam diameter, a pulsed rating's repetition rate or
  pulse count, its polarization, its value's definition -- is not verifiable (``RATING_INFORMATION_MISSING``): the
  threshold depends on it, and taking a stated condition away must not make the rating apply where the stated one would
  have kept it from applying.

The rules:

* regime, wavelength (``rating_wavelength_tol_rel``), pulse duration (``rating_pulse_duration_tol_rel``; a rated duration
  is read as a FWHM, the convention of laser specifications -- a model assumption) and repetition rate
  (``rating_repetition_rate_tol_rel``; a pulsed rating that states none: not verifiable, ``RATING_INFORMATION_MISSING``):
  verified within the tolerance, otherwise no established relationship (thresholds do not scale across wavelengths,
  durations or rates in any general way). On a surface only light left out
  as negligible reaches, its lines are the light (``Light.left_out``): each meets or leaves not verifiable the conditions
  with its own regime, wavelength and pulse train, and never makes one differ.
* Test beam diameter (``beam_diameter_condition``), the beam on the sample (not stated: not verifiable,
  ``RATING_INFORMATION_MISSING``): the test beam's 1/e^2 diameter ``D`` is read as lying in ``[D_lo, D_hi]``
  (``rated_beam_interval``): ``[d, d]`` for a ``1/E2`` diameter, ``[d, d] / 0.5887`` for ``FWHM``, and ``[d, d / 0.5887]``
  when the definition is not stated -- a model assumption: the FWHM diameter of a Gaussian is 0.5887 of its 1/e^2 one,
  its 1/e and ISO effective-area diameters 0.707, its second-moment and 86.5 % power-content diameters 1, so the standard
  widths of a Gaussian beam lie in that range (a width taken at a lower level, e.g. 1/e^4, would not). Each beam on the surface with a known size is compared by its smallest
  1/e^2 diameter across the beam ``a_lo`` and the largest 1/e^2 extent of its footprint on the surface ``a_hi`` (first match):
  verified when the definition is stated and both transverse 1/e^2 diameters are within ``rating_beam_diameter_tol_rel`` of
  ``D`` (with the definition not stated, a beam one reading would verify is not verifiable, ``RATING_INFORMATION_MISSING``); ``a_hi <= D_lo``: at least the rated value (a smaller spot samples fewer defects, and heats less at a given
  fluence or irradiance); ``a_lo > D_hi``: at most the rated value (small test spots overstate the threshold of larger
  beams); a definition not stated: not verifiable (``RATING_INFORMATION_MISSING``, the beam lies between the readings);
  otherwise the beam's axes lie on both sides of the stated diameter: no established relationship. Exceptions: on a
  metallic reflector (``PowerRating.surface_kind``) a pulsed rating keeps a direction only as a single-pulse rating applied
  to a single pulse per site (the multi-pulse threshold of metal mirrors has been found to rise with the spot size) -- with
  the pulses per site not stated, not verifiable (``DESIGN_INPUT_MISSING``), with no count stated, not verifiable
  (``RATING_INFORMATION_MISSING``) -- and otherwise has no established relationship; a reflecting surface of unknown kind is not taken for a dielectric one where that would change the result
  (``RATING_INFORMATION_MISSING``); a ``W/cm`` rating applied to a larger beam has no established relationship (the
  linear power density removes only the first-order thermal dependence on the beam size). The beams combine: one with no
  established relationship, or beams in opposite directions, give no established relationship; else one not verifiable
  makes the condition not verifiable; else a direction holds when every beam is verified or in it. Light whose size is not
  known gives ``BEAM_SIZE_UNKNOWN``: a beam with no known value on the surface always, light bounded by its diffraction
  limit (untraced light of unknown size, light left out as negligible) when it may carry more than the tolerance of the
  most light the surface can receive and the beams of known size verify the diameter or give a threshold at least the
  rated value (a smaller spot needs every part of the light to be small; the lower bound of the exposure, at the traced
  beams' centres, already allows for such light). The weakest beams of known size that carry, with that light, at most the
  same share (a faint ghost of another size) are left out of the comparison (``exposures._minor_beams``); both shares are
  measured in every quantity that gives the size its direction (``exposures._size_quantities``: the rating's own -- peak
  irradiance, ``P / d`` or peak fluence -- and for a CW rating the power and ``P / d``, for a pulsed one the peak fluence
  and the pulse energy; a beam beside the surface with its tail bound in each).
* Pulse count (``pulse_count_condition``) against the pulses one site of the surface receives (``PulsesPerSite``): each
  crossing of a source's light there brings that source's ``PulseTrain.total_pulses`` (``UNLIMITED``: more than any rated
  count; the lines of one source that cross a surface along one path arrive in its pulses together: one crossing), so a
  site receives between ``low`` (the fewest one crossing brings) and ``high`` (every crossing at that site) pulses. The
  rating's count ``S`` is compared per tested site only where that is what it means: a single-pulse rating (``S = 1``:
  one pulse on each tested site -- what a single-pulse damage test is), or a count stated as applied at one fluence
  level to each tested site (``PowerRating.test_protocol`` ``S_ON_1``). Damage is irreversible: a site damaged within
  ``S`` pulses at a fluence is damaged within any more, and a site that survives ``S`` survived its first fewer. Under
  such a test, therefore: ``low == high == S``, verified; ``low >= S``, at most the rated value; ``high <= S``, at least
  the rated value; ``low < S < high``, not verifiable (``PULSES_PER_SITE_UNDETERMINED``); light whose source states no
  total, not verifiable (``DESIGN_INPUT_MISSING``). A rating that states no count, or a count above one without its
  protocol (one level per site, ramped, raster: no catalog rating states one), leaves the condition not verifiable
  (``RATING_INFORMATION_MISSING``) whatever the sources' totals -- no relation of its threshold to another number of
  pulses is established, and a ramped test's value may lie above or below the one-level value. Continuous light carries
  no pulses: under a pulsed rating the count differs with no established relationship, as its regime does -- except
  light left out as negligible alone, of unknown state there, which leaves it not verifiable (``LIGHT_STATE_UNKNOWN``).
* Value definition (``value_definition_condition``), what a ``J/cm2`` or ``W/cm2`` value is: verified when the rating
  states it (``PEAK``, ``TOTAL_PER_1E2_AREA``); not verifiable (``RATING_INFORMATION_MISSING``) when it does not -- the
  value then establishes no threshold for the peak exposure. Not stated for a ``W/cm`` value (``P / d`` has no area
  definition) and a ``TYPICAL`` value (which bounds no threshold).

Applicability (``rating_applicability``, first match): ``NOT_APPLICABLE`` (a condition differs with no established
relationship, or two differ in opposite directions), ``NOT_VERIFIED`` (a condition is not verifiable),
``THRESHOLD_AT_LEAST_RATED``, ``THRESHOLD_AT_MOST_RATED``, ``AS_RATED`` (every stated condition verified). The status
(``comparison_status``): ``AS_RATED`` decides both ways, ``THRESHOLD_AT_LEAST_RATED`` only ``WITHIN_RATING``,
``THRESHOLD_AT_MOST_RATED`` only ``EXCEEDS_RATING``; ``NOT_VERIFIED`` is ``CONDITIONS_NOT_VERIFIED``, ``NOT_APPLICABLE``
``CONDITIONS_MISMATCH``.

The rated value as a threshold (``rated_threshold``): the exposure compared is a peak value (peak fluence, peak
irradiance) or ``P / d``. ``PEAK``: the threshold for the peak exposure is the value. ``TOTAL_PER_1E2_AREA``: the value
is the total energy (or power) over ``pi (d / 2)^2``, ``d`` the test beam's 1/e^2 diameter; a Gaussian beam's peak is
``2 E / (pi w^2)`` with ``w = d / 2``, twice that ratio, so the threshold is twice the value (the mean fluence within the
diameter would be ``1 - e^-2`` of the ratio: that is not this definition). A value whose definition is not stated
establishes no threshold: test reports use several (a peak; the energy over the 1/e^2 area; others, e.g. over a FWHM
area, whose peak threshold is ``R ln 2``) and no set of them is known to be exhaustive. ``P / d`` is its own threshold
(its diameter definition has its own readings, ``rated_beam_interval``). A ``TYPICAL`` value is the typical value of a
product, not a value each part meets: it bounds the threshold of no particular part, and its comparison decides neither
way.

Plain comparisons of IEEE ``+ - x /`` results only: identical in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config import NumericalConfig
from ..contracts.components import PowerRating, PulseTrain
from ..contracts.results import RatingCondition


def _default(name: str) -> float:
    return float(NumericalConfig.model_fields[name].default)


@dataclass(frozen=True)
class RatingTolerances:
    """The run's settings for matching a rating's test conditions (``NumericalConfig.rating_*``; the defaults are the
    Engine's): the relative windows within which the light's wavelength, pulse duration, repetition rate and beam diameter
    match the rating's, the largest share of a beam's power outside the rated polarization state
    (``polarization_fraction``), and the largest share of the most light a surface can receive that light of unknown size,
    place or state -- and followed light of lines, sizes or states the rating is not for -- may carry before it leaves a
    condition not verifiable (``unlocated_light_share``). A stated test beam diameter is verified when both transverse
    1/e^2 diameters of every beam are within ``beam_diameter_rel`` of it: at the default about 10 % in the threshold under
    the (d1 / d2)^2 rule of thumb for small changes, comparable with the pulse-duration window under tau^(1/2) scaling."""

    wavelength_rel: float = _default("rating_wavelength_tol_rel")
    pulse_duration_rel: float = _default("rating_pulse_duration_tol_rel")
    repetition_rate_rel: float = _default("rating_repetition_rate_tol_rel")
    beam_diameter_rel: float = _default("rating_beam_diameter_tol_rel")
    polarization_fraction: float = _default("rating_polarization_fraction_tol")
    unlocated_light_share: float = _default("rating_unlocated_light_share")

    @classmethod
    def of(cls, cfg: NumericalConfig) -> "RatingTolerances":
        return cls(cfg.rating_wavelength_tol_rel, cfg.rating_pulse_duration_tol_rel, cfg.rating_repetition_rate_tol_rel,
                   cfg.rating_beam_diameter_tol_rel, cfg.rating_polarization_fraction_tol, cfg.rating_unlocated_light_share)

    def as_dict(self) -> dict[str, float]:
        """The values in force, as ``ExposureReport.tolerances`` reports them."""
        return {"wavelength_rel": self.wavelength_rel, "pulse_duration_rel": self.pulse_duration_rel,
                "repetition_rate_rel": self.repetition_rate_rel, "beam_diameter_rel": self.beam_diameter_rel,
                "polarization_fraction": self.polarization_fraction, "unlocated_light_share": self.unlocated_light_share}

    def setting_of(self, condition: str) -> tuple[str, float, float] | None:
        """(the setting that holds a condition's window, its value in force, the Engine default); ``None``: no window."""
        found = CONDITION_SETTINGS.get(condition)
        return None if found is None else (found[0], getattr(self, found[1]), _default(found[0]))


#: Per condition whose match has a window: (the ``NumericalConfig`` setting that holds it, the ``RatingTolerances`` field).
CONDITION_SETTINGS = {
    "wavelength": ("rating_wavelength_tol_rel", "wavelength_rel"),
    "pulse_duration": ("rating_pulse_duration_tol_rel", "pulse_duration_rel"),
    "repetition_rate": ("rating_repetition_rate_tol_rel", "repetition_rate_rel"),
    "beam_diameter": ("rating_beam_diameter_tol_rel", "beam_diameter_rel"),
    "polarization": ("rating_polarization_fraction_tol", "polarization_fraction"),
}
#: The Engine's default settings (a call without a run's configuration).
DEFAULT_TOLERANCES = RatingTolerances()
#: A Gaussian beam's peak fluence or irradiance in units of its total energy or power over the area of its 1/e^2
#: diameter: 2 E / (pi w^2) against E / (pi w^2).
PEAK_PER_TOTAL_OVER_E2_AREA = 2.0
#: FWHM diameter of a Gaussian intensity profile in units of its 1/e^2 diameter: sqrt(2 ln 2) / 2.
FWHM_PER_E2_DIAMETER = 0.5887050112577373
PERCENT = 100.0

VERIFIED, DIFFERS, NOT_VERIFIABLE, NOT_STATED = "VERIFIED", "DIFFERS", "NOT_VERIFIABLE", "NOT_STATED"
AT_LEAST, AT_MOST, NO_RELATIONSHIP = "THRESHOLD_AT_LEAST_RATED", "THRESHOLD_AT_MOST_RATED", "NO_ESTABLISHED_RELATIONSHIP"
DESIGN_INPUT_MISSING = "DESIGN_INPUT_MISSING"
RATING_INFORMATION_MISSING = "RATING_INFORMATION_MISSING"
BEAM_SIZE_UNKNOWN = "BEAM_SIZE_UNKNOWN"
LIGHT_STATE_UNKNOWN = "LIGHT_STATE_UNKNOWN"
COHERENT_STATE_UNDETERMINED = "COHERENT_STATE_UNDETERMINED"
AS_RATED, NOT_VERIFIED, NOT_APPLICABLE = "AS_RATED", "NOT_VERIFIED", "NOT_APPLICABLE"
#: Applicabilities whose comparisons decide in at least one direction.
APPLYING = (AS_RATED, AT_LEAST, AT_MOST)
#: The conditions a comparison records, in order.
CONDITION_KINDS = ("regime", "wavelength", "pulse_duration", "repetition_rate", "beam_diameter", "pulse_count", "polarization",
                   "value_definition")
#: Which reason names a condition not verifiable when several apply (the first found in this order).
PULSES_PER_SITE_UNDETERMINED = "PULSES_PER_SITE_UNDETERMINED"
REASON_ORDER = (BEAM_SIZE_UNKNOWN, RATING_INFORMATION_MISSING, DESIGN_INPUT_MISSING, LIGHT_STATE_UNKNOWN, COHERENT_STATE_UNDETERMINED,
                PULSES_PER_SITE_UNDETERMINED)
#: Plain words for each reason (``measure_utilization`` messages).
REASON_TEXT = {DESIGN_INPUT_MISSING: "an input the Project does not state",
               RATING_INFORMATION_MISSING: "information the rating does not state",
               BEAM_SIZE_UNKNOWN: "light of unknown size", LIGHT_STATE_UNKNOWN: "light of unknown state or wavelength",
               COHERENT_STATE_UNDETERMINED: "coherent beams of different states",
               PULSES_PER_SITE_UNDETERMINED: "an undetermined number of pulses per site"}
#: ``PulseTrain.total_pulses`` of a source that keeps pulsing for as long as the optics are in use.
UNLIMITED = "UNLIMITED"
#: ``PowerRating.test_protocol``: each tested site received the rating's pulse count at one fluence level.
S_ON_1 = "S_ON_1"
#: ``PowerRating.value_definition``: the total energy or power over the area of the test beam's 1/e^2 diameter.
TOTAL_PER_E2_AREA = "TOTAL_PER_1E2_AREA"


@dataclass(frozen=True)
class Light:
    """One contribution of light on the surface as the regime, wavelength and pulse conditions see it: its source, vacuum
    wavelength (nm) and pulse train (``None``: continuous). ``left_out``: a line of the light the trace leaves out as
    negligible, the light of a surface no followed light reaches (``scalar_conditions``)."""

    source_id: str
    wavelength_nm: float
    pulse: PulseTrain | None
    left_out: bool = False


@dataclass(frozen=True)
class BeamSize:
    """One beam's 1/e^2 size on the surface: ``transverse_mm`` its two transverse 1/e^2 diameters (2 w_x, 2 w_y),
    ``largest_mm`` the largest 1/e^2 extent of its footprint on the surface (twice the footprint's semi-major axis;
    ``None``: unbounded, a grazing beam), ``name`` the beam for messages."""

    transverse_mm: tuple[float, float]
    largest_mm: float | None
    name: str = ""

    @property
    def smallest_mm(self) -> float:
        return min(self.transverse_mm)


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / reference


def line_differences(rating: PowerRating, wavelength_nm: float, pulse: PulseTrain | None,
                     tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[str, ...]:
    """The conditions of ``rating`` one line's light does not meet: ``regime`` alone when the regime differs, otherwise
    each of ``wavelength``, ``pulse_duration`` and ``repetition_rate`` (a rate the rating states) it is outside the window
    of. The same tests ``scalar_conditions`` applies to the light on a surface as a whole (``DIFFERS``)."""
    if (rating.regime == "PULSED") != (pulse is not None):
        return ("regime",)
    found: list[str] = []
    if _rel(wavelength_nm, rating.wavelength_nm) > tolerances.wavelength_rel:
        found.append("wavelength")
    if pulse is not None and rating.pulse_duration_s is not None:
        if _rel(pulse.pulse_duration_s, rating.pulse_duration_s) > tolerances.pulse_duration_rel:
            found.append("pulse_duration")
        if (rating.repetition_rate_hz is not None
                and _rel(pulse.repetition_rate_hz, rating.repetition_rate_hz) > tolerances.repetition_rate_rel):
            found.append("repetition_rate")
    return tuple(found)


def _record(condition: str, state: str, message: str, **values) -> RatingCondition:
    return RatingCondition(condition=condition, state=state, message=message, **values)  # type: ignore[arg-type]


def _not_stated(condition: str, **values) -> RatingCondition:
    return _record(condition, NOT_STATED, f"{condition.replace('_', ' ')}: does not apply to this rating", **values)


def _span(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


# ------------------------------------------------------------------------------------- regime, wavelength, pulse duration, rate


def scalar_conditions(rating: PowerRating, lights: Sequence[Light], unknown: Mapping[str, str] | None = None,
                      tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingCondition, RatingCondition, RatingCondition, RatingCondition]:
    """The ``regime``, ``wavelength``, ``pulse_duration`` and ``repetition_rate`` conditions of ``rating`` against the
    light on the surface (every contribution). ``unknown``: per condition, a text saying that light which may reach the
    surface leaves it not verifiable (light left out as negligible of another line: ``LIGHT_STATE_UNKNOWN``) unless the
    light found already differs. Lines of left-out light (``Light.left_out``, on a surface no followed light reaches)
    never make a condition differ: a line the rating is not for leaves the conditions it differs in not verifiable
    (``unknown``), the lines that meet a condition verify it -- a pulsed line with its own pulse train --, and a condition
    no light meets is not verifiable."""

    unknown = unknown or {}
    followed = [x for x in lights if not x.left_out]
    left_out = not followed and bool(lights)
    pulsed_rating = rating.regime == "PULSED"
    # regime
    texts: list[str] = []
    for regime_pulsed, label in ((True, "pulsed"), (False, "continuous (CW)")):
        sources = sorted({x.source_id for x in followed if (x.pulse is not None) == regime_pulsed})
        if sources and pulsed_rating != regime_pulsed:
            texts.append(f"regime: the rating is {rating.regime}, the beams of {', '.join(sources)} are {label}")
    meeting = followed or [x for x in lights if (x.pulse is not None) == pulsed_rating]
    regimes = sorted({"PULSED" if x.pulse is not None else "CW" for x in meeting})
    design = " and ".join(regimes) if regimes else None
    if texts:
        regime = _record("regime", DIFFERS, "; ".join(texts), effect=NO_RELATIONSHIP, rated=rating.regime, design=design)
    elif "regime" in unknown:
        regime = _record("regime", NOT_VERIFIABLE, unknown["regime"], reason=LIGHT_STATE_UNKNOWN, rated=rating.regime, design=design)
    elif not meeting:
        regime = _record("regime", NOT_VERIFIABLE, f"regime: {_NO_LIGHT}", reason=LIGHT_STATE_UNKNOWN, rated=rating.regime)
    else:
        regime = _record("regime", VERIFIED, f"regime: {rating.regime} rating, {'pulsed' if pulsed_rating else 'continuous'} light"
                         f"{_LEFT_OUT if left_out else ''}", rated=rating.regime, design=design)
    # wavelength
    if followed:
        wavelengths = sorted({x.wavelength_nm for x in followed})
    else:
        wavelengths = sorted({x.wavelength_nm for x in lights if _rel(x.wavelength_nm, rating.wavelength_nm) <= tolerances.wavelength_rel})
    values = dict(unit="nm", rated=rating.wavelength_nm, tolerance=tolerances.wavelength_rel)
    if wavelengths:
        values["design_low"], values["design_high"] = _span(wavelengths)
    texts = [f"wavelength: {w:g} nm is {PERCENT * _rel(w, rating.wavelength_nm):.3g} % from the rating's {rating.wavelength_nm:g} nm "
             f"(tolerance {PERCENT * tolerances.wavelength_rel:g} %)" for w in wavelengths
             if _rel(w, rating.wavelength_nm) > tolerances.wavelength_rel]
    if texts:
        wavelength = _record("wavelength", DIFFERS, "; ".join(texts), effect=NO_RELATIONSHIP, **values)
    elif "wavelength" in unknown:
        wavelength = _record("wavelength", NOT_VERIFIABLE, unknown["wavelength"], reason=LIGHT_STATE_UNKNOWN, **values)
    elif not wavelengths:
        wavelength = _record("wavelength", NOT_VERIFIABLE, f"wavelength: {_NO_LIGHT}", reason=LIGHT_STATE_UNKNOWN, **values)
    else:
        wavelength = _record("wavelength", VERIFIED, f"wavelength: {_range_text(wavelengths, 'nm', 'g')} within "
                             f"{PERCENT * tolerances.wavelength_rel:g} % of the rating's {rating.wavelength_nm:g} nm"
                             f"{_LEFT_OUT if left_out else ''}", **values)
    duration, rate = _pulse_conditions(rating, followed, [x for x in lights if x.left_out] if left_out else [], unknown, tolerances=tolerances)
    return regime, wavelength, duration, rate


#: A condition no light known to reach the surface meets (only light left out as negligible, of lines the rating is not
#: for, reaches it; ``scalar_conditions``).
_NO_LIGHT = "no light known to reach here is of the rating's line"
#: The verified conditions of a surface only light left out as negligible reaches.
_LEFT_OUT = " (the light the trace leaves out as negligible: no followed light reaches here)"


def _range_text(values: list[float], unit: str, spec: str) -> str:
    if not values:
        return "no light"
    low, high = _span(values)
    return f"{low:{spec}} {unit}" if low == high else f"{low:{spec}} to {high:{spec}} {unit}"


def _pulse_conditions(rating: PowerRating, followed: Sequence[Light], left_out: Sequence[Light], unknown: Mapping[str, str],
                      tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingCondition, RatingCondition]:
    """The ``pulse_duration`` and ``repetition_rate`` conditions against the followed light ``followed``, or, on a surface
    only light left out as negligible reaches, against the pulse trains of its lines ``left_out`` that meet each condition
    (a line that does not is in ``unknown``, or carries no more than the tolerance)."""
    if rating.regime != "PULSED":
        return _not_stated("pulse_duration"), _not_stated("repetition_rate")
    assert rating.pulse_duration_s is not None
    pulses = sorted({(x.source_id, x.pulse) for x in followed if x.pulse is not None}, key=lambda item: item[0])
    trains = sorted({(x.source_id, x.pulse) for x in left_out if x.pulse is not None}, key=lambda item: item[0])
    out: list[RatingCondition] = []
    for name, unit, rated, tolerance, of in (
            ("pulse_duration", "s", rating.pulse_duration_s, tolerances.pulse_duration_rel, lambda p: p.pulse_duration_s),
            ("repetition_rate", "Hz", rating.repetition_rate_hz, tolerances.repetition_rate_rel, lambda p: p.repetition_rate_hz)):
        label = name.replace("_", " ")
        if rated is None:
            # (a pulsed rating states its duration; one that states no rate leaves the rate's relation unknown)
            out.append(_record(name, NOT_VERIFIABLE, f"{label}: the rating states none, so its relation to the light here is "
                               "not established", reason=RATING_INFORMATION_MISSING, unit=unit))
            continue
        values = dict(unit=unit, rated=rated, tolerance=tolerance)
        if not followed:
            # (only left-out light: the pulse trains of its lines that meet the condition verify it)
            found = [of(p) for _, p in trains if _rel(of(p), rated) <= tolerance]
            if found:
                values["design_low"], values["design_high"] = _span(found)
            if name in unknown:
                out.append(_record(name, NOT_VERIFIABLE, unknown[name], reason=LIGHT_STATE_UNKNOWN, **values))
            elif not found:
                out.append(_record(name, NOT_VERIFIABLE, f"{label}: {_NO_LIGHT}", reason=LIGHT_STATE_UNKNOWN, **values))
            else:
                out.append(_record(name, VERIFIED, f"{label}: {_range_text(found, unit, '.4g')} within {PERCENT * tolerance:g} % of "
                                                   f"the rating's {rated:.4g} {unit}{_LEFT_OUT}", **values))
            continue
        found = [of(p) for _, p in pulses]
        if found:
            values["design_low"], values["design_high"] = _span(found)
        if not pulses:
            out.append(_record(name, DIFFERS, f"{label}: the rating is for pulses, the light here is continuous",
                               effect=NO_RELATIONSHIP, **values))
            continue
        if name == "pulse_duration":
            texts = [f"pulse duration: {source} emits {p.pulse_duration_s:.4g} s pulses, the rating is for {rated:.4g} s (tolerance "
                     f"{PERCENT * tolerance:g} %)" for source, p in pulses if _rel(p.pulse_duration_s, rated) > tolerance]
        else:
            texts = [f"repetition rate: {source} pulses at {p.repetition_rate_hz:.4g} Hz, the rating is for {rated:.4g} Hz (tolerance "
                     f"{PERCENT * tolerance:g} %)" for source, p in pulses if _rel(p.repetition_rate_hz, rated) > tolerance]
        if texts:
            out.append(_record(name, DIFFERS, "; ".join(texts), effect=NO_RELATIONSHIP, **values))
        elif name in unknown:
            out.append(_record(name, NOT_VERIFIABLE, unknown[name], reason=LIGHT_STATE_UNKNOWN, **values))
        else:
            out.append(_record(name, VERIFIED, f"{label}: {_range_text(found, unit, '.4g')} within {PERCENT * tolerance:g} % of the "
                                               f"rating's {rated:.4g} {unit}", **values))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------- test beam diameter


def rated_beam_interval(rating: PowerRating) -> tuple[float, float, bool] | None:
    """The rating's test beam as a 1/e^2 diameter range ``(D_lo, D_hi)`` (mm) and whether its definition is stated;
    ``None`` when it states no test beam. ``1/E2``: the stated diameter; ``FWHM``: the stated value / 0.5887; any other
    or no definition: from the stated value to 1/0.5887 times it."""

    d = rating.beam_diameter_mm
    if d is None:
        return None
    if rating.beam_diameter_definition == "1/E2":
        return d, d, True
    if rating.beam_diameter_definition == "FWHM":
        e2 = d / FWHM_PER_E2_DIAMETER
        return e2, e2, True
    return d, d / FWHM_PER_E2_DIAMETER, False


def _beam_direction(size: BeamSize, low: float, high: float, stated: bool, tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[str, str]:
    """One beam against the test beam ``[low, high]`` (1/e^2 diameters): (state, effect or reason). With the definition not
    stated, a beam that one reading of the test beam would verify is not verifiable: the outcome depends on which reading
    holds (a direction there would be weaker than the verified match one reading gives, and could cancel against another
    condition's direction where the match would not)."""
    if stated and all(abs(diameter - low) <= tolerances.beam_diameter_rel * low for diameter in size.transverse_mm):
        return VERIFIED, ""
    if not stated and _a_reading_verifies(size, low, high, tolerances=tolerances):
        return NOT_VERIFIABLE, RATING_INFORMATION_MISSING
    if size.largest_mm is not None and size.largest_mm <= low:
        return DIFFERS, AT_LEAST
    if size.smallest_mm > high:
        return DIFFERS, AT_MOST
    if not stated:
        return NOT_VERIFIABLE, RATING_INFORMATION_MISSING
    return DIFFERS, NO_RELATIONSHIP


def _a_reading_verifies(size: BeamSize, low: float, high: float, tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> bool:
    """Whether a reading ``D`` in ``[low, high]`` verifies the beam: every transverse diameter within the tolerance of ``D``."""
    least = max(max(size.transverse_mm) / (1.0 + tolerances.beam_diameter_rel), low)
    most = min(min(size.transverse_mm) / (1.0 - tolerances.beam_diameter_rel), high)
    return least <= most


def _metallic(rating: PowerRating, effect: str, design_pulses: float | None) -> tuple[str, str]:
    """A direction across beam sizes on a metallic reflector under a pulsed rating: kept for a single-pulse rating applied
    to a single pulse per site (``design_pulses``: the most pulses a site receives), not verifiable while they are not
    stated, otherwise none."""
    if rating.pulse_count == 1 and design_pulses == 1:
        return DIFFERS, effect
    if rating.pulse_count == 1 and design_pulses is None:
        return NOT_VERIFIABLE, DESIGN_INPUT_MISSING
    if rating.pulse_count is None and design_pulses != 0:
        return NOT_VERIFIABLE, RATING_INFORMATION_MISSING  # (it may be a single-pulse rating)
    return DIFFERS, NO_RELATIONSHIP


def _exception(rating: PowerRating, state: str, effect: str, design_pulses: float | None) -> tuple[str, str, str]:
    """The exceptions to a beam's direction (module docstring): (state, effect or reason, why)."""
    if state != DIFFERS or effect not in (AT_LEAST, AT_MOST):
        return state, effect, ""
    if rating.unit == "W/cm" and effect == AT_MOST:
        return DIFFERS, NO_RELATIONSHIP, "wcm"
    if rating.regime != "PULSED":
        return state, effect, ""
    if rating.surface_kind == "METALLIC_REFLECTOR":
        found = _metallic(rating, effect, design_pulses)
        return found[0], found[1], "metal" if found != (state, effect) else ""
    if rating.surface_kind is None and _metallic(rating, effect, design_pulses) != (state, effect):
        return NOT_VERIFIABLE, RATING_INFORMATION_MISSING, "kind"
    return state, effect, ""


def _rated_text(rating: PowerRating, low: float, high: float, stated: bool) -> str:
    d = rating.beam_diameter_mm
    if not stated:
        return f"a {d:.4g} mm test beam (definition not stated: a 1/e^2 diameter of {low:.4g} to {high:.4g} mm)"
    if rating.beam_diameter_definition == "FWHM":
        return f"a {d:.4g} mm FWHM test beam (a 1/e^2 diameter of {low:.4g} mm)"
    return f"a {d:.4g} mm 1/e^2 test beam"


def beam_diameter_condition(rating: PowerRating, sizes: Sequence[BeamSize], *, unknown_size: str | None = None,
                            unsized: str | None = None, design_pulses: float | None = None, tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> RatingCondition:
    """The ``beam_diameter`` condition (module docstring) for the beams of known size ``sizes``. ``unknown_size``: a text
    naming light with no known value on the surface (always not verifiable); ``unsized``: a text naming light bounded by
    its diffraction limit that may carry more than the tolerance of the most light here; ``design_pulses``: the most pulses
    a site receives (``PulsesPerSite.high``; ``None``: not stated)."""
    return beam_diameter_decision(rating, sizes, unknown_size=unknown_size, unsized=unsized, design_pulses=design_pulses, tolerances=tolerances)[0]


def beam_diameter_decision(rating: PowerRating, sizes: Sequence[BeamSize], *, unknown_size: str | None = None,
                           unsized: str | None = None, design_pulses: float | None = None, tolerances: RatingTolerances = DEFAULT_TOLERANCES) -> tuple[RatingCondition, bool]:
    """``beam_diameter_condition`` and whether ``unsized`` alone keeps it from being verified or directed (less light of
    unknown size would decide it)."""

    interval = rated_beam_interval(rating)
    if interval is None:
        return _record("beam_diameter", NOT_VERIFIABLE, "beam diameter: the rating states no test beam diameter, so the beams "
                       "here may be smaller or larger than its test beam", reason=RATING_INFORMATION_MISSING, unit="mm"), False
    low, high, stated = interval
    values: dict = dict(unit="mm", rated=rating.beam_diameter_mm, rated_low=low, rated_high=high, tolerance=tolerances.beam_diameter_rel)
    if sizes:
        values["design_low"] = min(s.smallest_mm for s in sizes)
        largest = [s.largest_mm for s in sizes]
        if all(x is not None for x in largest):
            values["design_high"] = max(largest)  # type: ignore[type-var]
    head = f"beam diameter: rated with {_rated_text(rating, low, high, stated)}"
    found: list[tuple[str, str, str, BeamSize]] = []
    for size in sizes:
        state, extra = _beam_direction(size, low, high, stated, tolerances=tolerances)
        state, extra, why = _exception(rating, state, extra, design_pulses)
        found.append((state, extra, why, size))
    effects = {extra for state, extra, _, _ in found if state == DIFFERS}
    reasons = {extra for state, extra, _, _ in found if state == NOT_VERIFIABLE}
    if unknown_size is not None:
        reasons.add(BEAM_SIZE_UNKNOWN)

    def beams(effect: str) -> str:
        names = [s.name for state, extra, _, s in found if state == DIFFERS and extra == effect and s.name]
        return f" ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})" if names else ""

    if NO_RELATIONSHIP in effects or {AT_LEAST, AT_MOST} <= effects:
        whys = {why for state, extra, why, _ in found if state == DIFFERS and extra == NO_RELATIONSHIP}
        if "metal" in whys:
            text = ("on a metallic reflector a pulsed rating has no established relationship across beam sizes unless it is a "
                    "single-pulse rating applied to a single pulse per site")
        elif "wcm" in whys:
            text = "a W/cm rating has no established relationship for a beam larger than its test beam"
        elif NO_RELATIONSHIP in effects:
            text = "the axes of a beam here lie on both sides of the test beam's diameter" + beams(NO_RELATIONSHIP)
        else:
            text = (f"beams here are smaller{beams(AT_LEAST)} and larger{beams(AT_MOST)} than the test beam: no direction "
                    "holds for the light as a whole")
        return _record("beam_diameter", DIFFERS, f"{head}; {text}", effect=NO_RELATIONSHIP, **values), False
    if reasons:
        reason = next(r for r in REASON_ORDER if r in reasons)
        parts = []
        if unknown_size is not None:
            parts.append(unknown_size)
        whys = {why for state, _, why, _ in found if state == NOT_VERIFIABLE}
        if RATING_INFORMATION_MISSING in reasons:
            unread = [s for state, extra, why, s in found if state == NOT_VERIFIABLE and not why]
            for band, text in (([s for s in unread if _a_reading_verifies(s, low, high, tolerances=tolerances)], "matches one reading"),
                               ([s for s in unread if not _a_reading_verifies(s, low, high, tolerances=tolerances)], "lies between the readings")):
                if not band:
                    continue
                named = [s.name for s in band if s.name]
                who = (f"the beam of {', '.join(named[:3])}{', ...' if len(named) > 3 else ''}" if named else "a beam here")
                parts.append(f"{who} ({min(s.smallest_mm for s in band):.4g} mm across, 1/e^2) {text} of the test beam's "
                             "diameter, whose definition the rating does not state")
            if "kind" in whys:
                parts.append("whether a reflecting surface of this element is metallic is not stated, and on a metallic reflector "
                             "a pulsed rating keeps no direction across beam sizes")
        if DESIGN_INPUT_MISSING in reasons:
            parts.append("on a metallic reflector a single-pulse rating holds across beam sizes for a single pulse per site only, "
                         "and the light's source states no pulse.total_pulses")
        if "metal" in whys and rating.pulse_count is None:
            parts.append("on a metallic reflector a pulsed rating holds across beam sizes only as a single-pulse rating, and the "
                         "rating states no pulse count")
        return _record("beam_diameter", NOT_VERIFIABLE, f"{head}; {'; '.join(parts)}", reason=reason, **values), False
    if not found:
        return _record("beam_diameter", NOT_VERIFIABLE, f"{head}; no beam of known size meets the surface", reason=BEAM_SIZE_UNKNOWN,
                       **values), False
    # (beams that verify the diameter beside beams in one direction: named apart, their own size and footprint not counted)
    matched = any(state == VERIFIED for state, _, _, _ in found)
    beams_here = "the beams here that differ from the test beam are" if matched else "the beams here are"
    others = f", the others match it within {PERCENT * tolerances.beam_diameter_rel:g} % (1/e^2)" if matched else ""
    if AT_MOST in effects:
        least = min(s.smallest_mm for state, extra, _, s in found if state == DIFFERS and extra == AT_MOST)
        return _record("beam_diameter", DIFFERS, f"{head}; {beams_here} at least {least:.4g} mm across (1/e^2), larger than the "
                       f"test beam under every reading{others}: the damage threshold for them is at most the rated value",
                       effect=AT_MOST, **values), False
    if unsized is not None:
        return _record("beam_diameter", NOT_VERIFIABLE, f"{head}; {unsized}", reason=BEAM_SIZE_UNKNOWN, **values), True
    if AT_LEAST in effects:
        # (every beam in this direction has a bounded footprint: a_hi <= D_lo)
        most = max(s.largest_mm for state, extra, _, s in found if state == DIFFERS and extra == AT_LEAST)  # type: ignore[type-var]
        return _record("beam_diameter", DIFFERS, f"{head}; {beams_here} at most {most:.4g} mm across on the surface (1/e^2 "
                       f"footprint), no larger than the test beam under any reading{others}: the damage threshold for them is "
                       "at least the rated value", effect=AT_LEAST, **values), False
    return _record("beam_diameter", VERIFIED, f"{head}; the beams here are {values['design_low']:.4g} to "
                   f"{max(max(s.transverse_mm) for s in sizes):.4g} mm across (1/e^2), within "
                   f"{PERCENT * tolerances.beam_diameter_rel:g} % of it", **values), False


# ------------------------------------------------------------------------------------------------------- pulse count


@dataclass(frozen=True)
class PulsesPerSite:
    """The pulses one site of a surface receives while the optics stay in place: between ``low`` (the fewest one crossing
    of the light brings: its source's total) and ``high`` (every crossing at one site, and light of unknown place;
    ``math.inf``: a crossing keeps pulsing or the crossings are not bounded). ``missing``: the sources whose light there
    states no total (``PulseTrain.total_pulses``); ``continuous``: the light there carries no pulses (continuous light
    only); ``crossings``: the number of crossings counted; ``unbounded``: what leaves ``high`` without a bound besides a
    train that keeps pulsing (light left out as negligible beyond the share, light that may be stored), ``None`` when
    nothing does; ``left_out``: the light there is only light left out as negligible, whose state the trace does not
    know (it never makes a condition differ)."""

    low: float
    high: float
    missing: tuple[str, ...] = ()
    continuous: bool = False
    crossings: int = 1
    unbounded: str | None = None
    left_out: bool = False


def total_pulses(pulse: PulseTrain | None) -> float | None:
    """A pulse train's ``total_pulses`` as a number (``math.inf`` for ``UNLIMITED``); ``None`` when it is not stated."""
    if pulse is None or pulse.total_pulses is None:
        return None
    return math.inf if pulse.total_pulses == UNLIMITED else float(pulse.total_pulses)


def _pulses_text(n: float) -> str:
    if not math.isfinite(n):
        return "a train that keeps pulsing (UNLIMITED)"
    return f"{n:.0f} pulse{'s' if n != 1 else ''}"


def pulse_count_condition(rating: PowerRating, design: PulsesPerSite | float | None = None) -> RatingCondition:
    """The ``pulse_count`` condition (module docstring) of a pulsed rating: its count ``S`` against the pulses one site
    receives (``PulsesPerSite``; a number ``N``: one crossing of a source that sends ``N`` pulses, ``math.inf`` for
    ``UNLIMITED``; ``None``: the light's source states no total), compared only for a single-pulse rating or one whose
    count was applied at one level to each tested site (``test_protocol`` ``S_ON_1``)."""

    count = rating.pulse_count
    if isinstance(design, (int, float)):
        design = PulsesPerSite(float(design), float(design))
    if rating.regime != "PULSED":
        return _not_stated("pulse_count")
    values: dict = dict(unit="pulses")
    if design is not None and design.continuous and design.left_out:
        # (only light left out as negligible reaches the surface: of unknown state, it never makes a condition differ)
        return _record("pulse_count", NOT_VERIFIABLE, "pulse count: only light left out as negligible, of a continuous line, "
                       "reaches the surface; its state here is not known", reason=LIGHT_STATE_UNKNOWN, **values)
    if count is None:
        if design is not None and design.continuous:
            return _record("pulse_count", DIFFERS, "pulse count: the rating states none; the light here is continuous",
                           effect=NO_RELATIONSHIP, **values)
        return _record("pulse_count", NOT_VERIFIABLE, "pulse count: the rating states none, so how many pulses each tested site "
                       "received is not known", reason=RATING_INFORMATION_MISSING, **_design_values(design, values))
    values["rated"] = count
    plural = "s" if count != 1 else ""
    if count == 1:
        head = "pulse count: the rating states a single pulse, read as one pulse on each tested site"
    elif rating.test_protocol == S_ON_1:
        head = f"pulse count: the rating states {count} pulses on each tested site at one fluence level (S-on-1)"
    else:
        head = f"pulse count: the rating states {count} pulse{plural}"
    if design is not None and design.continuous:
        return _record("pulse_count", DIFFERS, f"{head}; the light here is continuous", effect=NO_RELATIONSHIP, **values)
    if count > 1 and rating.test_protocol != S_ON_1:
        return _record("pulse_count", NOT_VERIFIABLE, f"{head}, not how they were applied to each tested site (at one fluence "
                       "level, ramped, raster), so its value's relation to the pulses a site receives here is not established",
                       reason=RATING_INFORMATION_MISSING, **_design_values(design, values))
    if design is None or design.missing:
        if design is None:
            who = "the light's source states"
        else:
            names = design.missing
            who = f"{', '.join(names[:3])}{', ...' if len(names) > 3 else ''} state{'s' if len(names) == 1 else ''}"
        return _record("pulse_count", NOT_VERIFIABLE, f"{head}; {who} no pulse.total_pulses (how many pulses a source sends "
                       "while the optics stay in place)", reason=DESIGN_INPUT_MISSING, **values)
    low, high = design.low, design.high
    if low == high:
        values["design"] = UNLIMITED if not math.isfinite(low) else int(low)
        here = f"{_pulses_text(low)} per site here" if math.isfinite(low) else f"{_pulses_text(low)} here"
    else:
        values["design_low"] = low
        if math.isfinite(high):
            values["design_high"] = high
        upto = f"{high:.0f}" if math.isfinite(high) else "an unbounded number of"
        why = [f"{design.crossings} crossing{'s' if design.crossings != 1 else ''} of pulsed light"] if design.crossings else []
        if design.unbounded is not None:
            why.append(design.unbounded)
        here = f"a site receives from {_pulses_text(low)} (one crossing) to {upto} pulses here ({', and '.join(why)})"
    if low == count and high == count:
        return _record("pulse_count", VERIFIED, f"{head}; {here}, as many as the test", **values)
    if low >= count:
        return _record("pulse_count", DIFFERS, f"{head}; {here}, at least as many as the test at every site: a site damaged "
                       "within the test's pulses at a fluence is damaged within more, so the damage threshold is at most the "
                       "rated value", effect=AT_MOST, **values)
    if high <= count:
        return _record("pulse_count", DIFFERS, f"{head}; {here}, no more than the test at any site: a site that survives the "
                       "test's pulses at a fluence survived its first fewer, so the damage threshold is at least the rated "
                       "value", effect=AT_LEAST, **values)
    return _record("pulse_count", NOT_VERIFIABLE, f"{head}; {here}: whether one site receives fewer or more pulses than the "
                   "test is not determined", reason=PULSES_PER_SITE_UNDETERMINED, **values)


def _design_values(design: PulsesPerSite | None, values: dict) -> dict:
    """``values`` with the pulses a site receives here (``design``, or ``design_low`` / ``design_high``), when known."""
    if design is None or design.missing or design.continuous:
        return values
    out = dict(values)
    if design.low == design.high:
        out["design"] = UNLIMITED if not math.isfinite(design.low) else int(design.low)
    else:
        out["design_low"] = design.low
        if math.isfinite(design.high):
            out["design_high"] = design.high
    return out


def value_definition_condition(rating: PowerRating) -> RatingCondition:
    """The ``value_definition`` condition (module docstring): what a ``J/cm2`` / ``W/cm2`` value is."""

    if rating.unit not in ("J/cm2", "W/cm2"):
        return _record("value_definition", NOT_STATED, "value definition: a W/cm value is a power per beam diameter (P / d)")
    if rating.status == "TYPICAL":
        if rating.value_definition is not None:
            return _record("value_definition", VERIFIED, "value definition: stated; a typical value bounds no damage threshold",
                           rated=rating.value_definition)
        return _record("value_definition", NOT_STATED, "value definition: a typical value bounds no damage threshold")
    if rating.value_definition == "PEAK":
        return _record("value_definition", VERIFIED, "value definition: the test beam's peak value; the damage threshold for "
                       "the peak exposure is the value", rated="PEAK")
    if rating.value_definition == TOTAL_PER_E2_AREA:
        return _record("value_definition", VERIFIED, "value definition: the total over the area of the test beam's 1/e^2 "
                       "diameter, for a Gaussian test beam half its peak; the damage threshold for the peak exposure is twice "
                       "the value", rated=TOTAL_PER_E2_AREA)
    what = "fluence" if rating.unit == "J/cm2" else "irradiance"
    return _record("value_definition", NOT_VERIFIABLE, f"value definition: the rating does not state whether its {what} is the "
                   f"test beam's peak {what} or a total over some area of the test beam, so it establishes no damage "
                   "threshold for the peak exposure", reason=RATING_INFORMATION_MISSING)


# ------------------------------------------------------------------------------------------- applicability and status


def rating_applicability(conditions: Sequence[RatingCondition]) -> str:
    """What a rating says about the threshold of the light (module docstring), from its conditions."""

    effects = {c.effect for c in conditions if c.state == DIFFERS}
    if NO_RELATIONSHIP in effects or {AT_LEAST, AT_MOST} <= effects:
        return NOT_APPLICABLE
    if any(c.state == NOT_VERIFIABLE for c in conditions):
        return NOT_VERIFIED
    if AT_LEAST in effects:
        return AT_LEAST
    if AT_MOST in effects:
        return AT_MOST
    return AS_RATED


def rated_threshold(value: float, unit: str, status: str, definition: str | None = None) -> tuple[float | None, float | None]:
    """The damage threshold a rated ``value`` in ``unit`` (fact ``status``, ``PowerRating.value_definition``
    ``definition``) puts on the exposure compared (module docstring): ``(lowest, highest)``, ``(None, None)`` when it
    establishes none. ``J/cm2`` and ``W/cm2``: the value (``PEAK``), twice it (``TOTAL_PER_1E2_AREA``), none when the
    definition is not stated; ``W/cm``: the value; ``TYPICAL``: none."""

    if status == "TYPICAL":
        return None, None
    if unit not in ("J/cm2", "W/cm2") or definition == "PEAK":
        return value, value
    if definition == TOTAL_PER_E2_AREA:
        return PEAK_PER_TOTAL_OVER_E2_AREA * value, PEAK_PER_TOTAL_OVER_E2_AREA * value
    return None, None


def comparison_status(applicability: str, lower: float | None, upper: float | None, lowest: float | None,
                      highest: float | None = None) -> str:
    """A comparison's status from its applicability and the exposure's bounds (``lower``: ``None`` or 0 for none above 0;
    ``upper``: ``None`` for none) against the threshold the rated value establishes (``rated_threshold``: within up to
    ``lowest``, exceeded above ``highest``; ``highest`` omitted: ``lowest``; ``None``: not bounded on that side)."""

    if applicability == NOT_APPLICABLE:
        return "CONDITIONS_MISMATCH"
    if applicability == NOT_VERIFIED:
        return "CONDITIONS_NOT_VERIFIED"
    highest = lowest if highest is None and lowest is not None else highest
    exceeded = lower is not None and highest is not None and lower > highest
    within = upper is not None and lowest is not None and upper <= lowest
    if applicability == AS_RATED:
        return "EXCEEDS_RATING" if exceeded else "WITHIN_RATING" if within else "UNKNOWN"
    if applicability == AT_LEAST:
        return "WITHIN_RATING" if within else "UNKNOWN"
    return "EXCEEDS_RATING" if exceeded else "UNKNOWN"
