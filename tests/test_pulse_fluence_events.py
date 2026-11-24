"""A peak fluence is the energy of one pulse per area: light of pulses that need not arrive together is bounded from below
by the strongest single pulse, not by the sum (``validation/exposures.py`` ``_lower``, ``_class_range(events=True)``).

Two mutually incoherent pulsed sources on one surface are not synchronized by anything a Project states, so their
pulses may arrive at different times: the surface's per-pulse fluence lies between the larger of the two (separate
pulses) and their sum (pulses that coincide). Time-averaged quantities (irradiance) still add. The reproduction is the
rating audit's: two 10 Hz lasers, each within the rating alone, were reported as an exceedance.

Fixtures: an ideal mirror at 45 degrees (one beam per source: exact values), fixture lasers and fixture ratings written
here (``value_definition`` PEAK; the beams' test beam size, repetition rate and linear polarization), never product
data. Where a decision is asserted, the rating states the evidence its pulse count needs: a single-pulse rating against
single-shot sources (at least one pulse per site: a threshold at most the rated value, or exactly one: as rated), or a
2-on-1 rating (S-on-1) against two single shots (at most two pulses per site: a threshold at least the rated value).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, PulseTrain, Target

from helpers import run

FIXTURE = "test fixture rating, not product data"
PULSE = PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=5e-9)
#: A laser's 1/e^2 diameter at the mirror 100 mm away (w0 = 0.5 mm, Rayleigh range pi w0^2 / lambda); the second laser,
#: 110 mm away, is within 0.2 % of it.
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)
#: The test conditions the fixture ratings state besides their value and pulse train, met by the beams here: a test beam
#: of their size and linear polarization (the lasers').
STATED = dict(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", polarization_condition="linear", polarization="LINEAR")


def _laser(instance_id: str, power_W: float, x: float, *, pulsed: bool = True, total: int | None = None):
    laser = B.laser(instance_id, wavelength_nm=1064.0, power_W=power_W, waist_radius_mm=0.5, position=(x, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0))
    pulse = PULSE.model_copy(update={"total_pulses": total}) if pulsed else None
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})


def _face(lasers, rating: PowerRating):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(*lasers, mirror), targets=(target,)))
    return next(s for s in result.exposures.surfaces if s.surface_id == "M/face"), result.targets[0]


def _own_fluence(hit) -> float:
    """2 E / (pi w_x w_y) of one hit, E = P / f (J/cm2), from the hit's own power and radii."""
    return 2.0 * (hit.incident_power_W / PULSE.repetition_rate_hz) / (math.pi * hit.beam_radius_x_mm * hit.beam_radius_y_mm) * 100.0


def _pulsed(value: float, **extra) -> PowerRating:
    extra = {"value_definition": "PEAK", **STATED, **extra}
    return PowerRating(regime="PULSED", value=value, unit="J/cm2", wavelength_nm=1064.0, pulse_duration_s=5e-9, repetition_rate_hz=10.0,
                       status="DECLARED", basis=FIXTURE, **extra)


#: A single-pulse rating (one pulse on each tested site).
SINGLE_PULSE = dict(pulse_count=1)
#: Two pulses on each tested site at one fluence level.
TWO_ON_ONE = dict(pulse_count=2, test_protocol="S_ON_1")


def test_two_independent_trains_are_bounded_by_the_stronger_pulse_and_the_sum():
    lasers = (_laser("A", 0.04, -100.0), _laser("B", 0.03, -110.0))
    face, _ = _face(lasers, _pulsed(1e30))
    a, b = sorted(face.hits, key=lambda h: -h.incident_power_W)
    f_a, f_b = _own_fluence(a), _own_fluence(b)
    (comparison,) = face.ratings
    assert comparison.exposure_value is None
    assert comparison.exposure_lower_bound == pytest.approx(f_a, rel=1e-12)  # the stronger train's own pulse
    assert comparison.exposure_upper_bound == pytest.approx(f_a + f_b, rel=1e-12)  # the two pulses together


@pytest.mark.parametrize(("value_of", "count", "applicability", "status", "assessment"), [
    # each train within the rating alone, their sum above it: not decided even under a single-pulse rating that two
    # single shots bring at least one pulse per site (a threshold at most the rated value; the audit's case was
    # EXCEEDS / FAIL)
    (lambda fa, fb: 0.5 * (fa + (fa + fb)), SINGLE_PULSE, "THRESHOLD_AT_MOST_RATED", "UNKNOWN", "INDETERMINATE"),
    # the sum within a 2-on-1 rating the two single shots meet with at most two pulses per site
    (lambda fa, fb: 1.01 * (fa + fb), TWO_ON_ONE, "THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING", "MET"),
    # one pulse above a single-pulse rating
    (lambda fa, fb: 0.99 * fa, SINGLE_PULSE, "THRESHOLD_AT_MOST_RATED", "EXCEEDS_RATING", "UNMET"),
])
def test_an_exceedance_needs_one_pulse_above_the_rating(value_of, count, applicability, status, assessment):
    lasers = (_laser("A", 0.04, -100.0, total=1), _laser("B", 0.04, -110.0, total=1))
    probe, _ = _face(lasers, _pulsed(1e30))
    f_a, f_b = (_own_fluence(h) for h in probe.hits)
    face, target = _face(lasers, _pulsed(value_of(f_a, f_b), **count))
    (comparison,) = face.ratings
    assert comparison.applicability == applicability
    assert (comparison.status, target.assessment.value) == (status, assessment)


def _two_line_laser(power_W: float, instance_id: str = "A", x: float = -100.0, total: int | None = None):
    """One source with two equal lines (1063.5 and 1064.5 nm, two coherence classes): both are parts of the same pulses."""
    laser = B.laser(instance_id, wavelength_nm=None, spectrum=((1063.5, 1.0), (1064.5, 1.0)), power_W=power_W, waist_radius_mm=0.5,
                    position=(x, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    pulse = PULSE.model_copy(update={"total_pulses": total})
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})


def test_the_lines_of_one_pulse_add():
    """One source's lines cross the mirror along one path in the same pulses: the fluence of a pulse is their sum, the
    lower bound included (an earlier draft took the larger line, half of it, and lost the exceedance).
    The exceedance is that of a single-pulse rating against a single shot (one pulse per site: as rated)."""
    probe, _ = _face((_two_line_laser(1.0),), _pulsed(1e30))
    total = sum(_own_fluence(h) for h in probe.hits)
    assert len(probe.hits) == 2
    (comparison,) = probe.ratings
    lower = comparison.exposure_value if comparison.exposure_value is not None else comparison.exposure_lower_bound
    assert lower == pytest.approx(total, rel=1e-12)
    face, target = _face((_two_line_laser(1.0, total=1),), _pulsed(0.75 * total, **SINGLE_PULSE))
    assert face.ratings[0].applicability == "AS_RATED"
    assert (face.ratings[0].status, target.assessment.value) == ("EXCEEDS_RATING", "UNMET")


def test_two_sources_with_two_lines_each_are_bounded_by_one_source_s_pulse():
    """Two such sources: a pulse holds the two lines of one source; the other source's pulses need not arrive with it."""
    face, _ = _face((_two_line_laser(0.04), _two_line_laser(0.03, "B", -110.0)), _pulsed(1e30))
    by_source: dict[str, float] = {}
    for h in face.hits:
        by_source[h.source_id] = by_source.get(h.source_id, 0.0) + _own_fluence(h)
    (comparison,) = face.ratings
    assert comparison.exposure_lower_bound == pytest.approx(max(by_source.values()), rel=1e-12)
    assert comparison.exposure_upper_bound == pytest.approx(sum(by_source.values()), rel=1e-12)


def test_ratings_that_bound_each_other_are_compared_by_their_thresholds():
    """A rating with a threshold at most its value (a 10 um test beam, smaller than the beam here) beside one that applies
    as stated: whether the second bounds the first is read from their thresholds, so one threshold stated two ways (a
    peak of 1.2 F, or a total over the 1/e^2 area of 0.6 F) gives one outcome (``test_rating_decisions`` checks the rule
    itself). Both are single-pulse ratings against a single shot, so their pulse count is verified."""
    single = _laser("A", 1.0, -100.0, total=1)
    probe, _ = _face((single,), _pulsed(1e30))
    f = probe.ratings[0].exposure_value
    small = _pulsed(f, **SINGLE_PULSE).model_copy(update={"beam_diameter_mm": 0.01, "beam_diameter_definition": "1/E2",
                                                          "surface_kind": "OTHER"})
    outcomes = []
    for other in (_pulsed(1.2 * f, **SINGLE_PULSE), _pulsed(0.6 * f, value_definition="TOTAL_PER_1E2_AREA", **SINGLE_PULSE)):
        mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(
            update={"power_ratings": (small, other.model_copy(update={"surface_kind": "OTHER"}))})
        target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
        result = run(ProjectSnapshot(components=(single, mirror), targets=(target,)))
        face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
        outcomes.append((face.rating_status, result.targets[0].status.value, result.targets[0].assessment.value))
    assert outcomes[0] == outcomes[1]


def test_irradiance_still_adds_over_independent_sources():
    """The time-averaged irradiance of two incoherent continuous sources is the sum of theirs: exact."""
    lasers = (_laser("A", 0.04, -100.0, pulsed=False), _laser("B", 0.03, -110.0, pulsed=False))
    rating = PowerRating(regime="CW", value=1e30, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                         basis=FIXTURE)
    face, _ = _face(lasers, rating)
    (comparison,) = face.ratings
    own = sum(2.0 * h.incident_power_W / (math.pi * h.beam_radius_x_mm * h.beam_radius_y_mm) * 100.0 for h in face.hits)
    assert comparison.exposure_value == pytest.approx(own, rel=1e-12)



def _member(fluence: float):
    """A traced contribution of ``fluence`` J/cm2 at the point considered (its profile not used: no Gaussian radii)."""
    from os_optics.contracts.results import ExposureHit
    from os_optics.validation.exposures import _HitValues

    hit = ExposureHit(path_id="S#0", source_id="S", spectral_index=0, wavelength_nm=1064.0, incident_power_W=0.1, aoi_deg=0.0,
                      peak_fluence_J_per_cm2=fluence, pulse_energy_J=0.01)
    return _HitValues(hit, True, PULSE, None)


@pytest.mark.parametrize(("gamma", "one_pulse", "any_timing"), [
    # incoherent members: their sum if they arrive together (5), the stronger alone if not (4)
    (0.0, 5.0, 4.0),
    # fully coherent members of unknown phase: (2 - 1)^2 either way
    (1.0, 1.0, 1.0),
    # partly coherent: one pulse 5 - 2 x 0.5 x 2 = 3; any timing the stronger less its cross term, 4 - 2 x 2 x 0.5 = 2
    (0.5, 3.0, 2.0),
])
def test_a_class_lower_bound_for_one_pulse_and_for_any_timing(gamma, one_pulse, any_timing):
    """Two traced members of one class, 4 and 1 J/cm2 at a point (amplitudes 2 and 1), degree of coherence ``gamma``: the
    bound for light arriving in one pulse, ``sum v - 2 gamma sqrt(v_1 v_2)`` or ``(2 - 1)^2``, and the bound for any
    timing (``events``), ``max(v_k - 2 gamma a_k a_j, (a_k - a_j)^2)`` over the members -- the values written out by hand."""
    from os_optics.validation.exposures import _Class, _Pair, _fluence

    members = [_member(4.0), _member(1.0)]
    cls = _Class((0, 1), {(0, 1): _Pair(gamma)}, False)
    assert _class_range_lower(cls, members, events=False) == pytest.approx(one_pulse, abs=1e-12)
    assert _class_range_lower(cls, members, events=True) == pytest.approx(any_timing, abs=1e-12)


def _class_range_lower(cls, members, *, events: bool) -> float:
    from os_optics.validation.exposures import _class_range, _fluence

    found = _class_range(cls, members, _fluence, (0.0, 0.0, 0.0), events=events)
    assert found is not None
    return found[0]
