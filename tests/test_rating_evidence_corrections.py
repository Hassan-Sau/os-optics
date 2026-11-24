"""Power-rating comparisons decide only what the available evidence establishes (review findings A and B).

A -- pulse count. A rating's pulse count is compared with the pulses a site receives only where its per-site meaning is
established: a single-pulse rating (one pulse on each tested site -- what a single-pulse damage test is) or a count stated
as applied at one fluence level to each tested site (``test_protocol`` ``S_ON_1``). Damage is irreversible: a site
damaged within ``S`` pulses at a fluence is damaged within more, and a site that survives ``S`` survived its first fewer;
under such a test more pulses per site therefore give a threshold at most the rated one and fewer at least the rated one.
A rating that states no count, or a count above one without its protocol (one level, ramped, raster), establishes no
relation to any other number of pulses, whatever the source states (omitted, a number, a single shot or UNLIMITED).

B -- rated value. The exposure is a peak value. A value stated as the peak (``PEAK``) is the threshold; one stated as the
total energy over the area of the test beam's 1/e^2 diameter (``TOTAL_PER_1E2_AREA``) is half a Gaussian beam's peak
(2 E / (pi w^2) against E / (pi w^2)), so the threshold is twice it; a value whose definition is not stated establishes no
threshold (test reports use several definitions -- the energy over a FWHM-diameter area, for one, puts the peak threshold
at R ln 2 -- and none is known to be exhaustive).

In every case the exposure itself is still computed and reported. Fixture data: lasers, mirrors and ratings written here
(never product data); the exposure references come from each hit's own power and 1/e^2 radii.
"""

from __future__ import annotations

import itertools
import math

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, PulseTrain, Target, ValidationResult

from helpers import run

FIXTURE = "test fixture rating, not product data"
#: FWHM diameter of a Gaussian in units of its 1/e^2 diameter: sqrt(2 ln 2) / 2.
FWHM_PER_E2 = math.sqrt(2.0 * math.log(2.0)) / 2.0
PULSE = PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=1e-8)
TOTALS = (None, 1, 100, 2**53 - 1, "UNLIMITED")


def _laser(*, total=None, power_W: float = 1.0, pulsed: bool = True):
    laser = B.laser("A", wavelength_nm=1064.0, power_W=power_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0))
    pulse = PULSE.model_copy(update={"total_pulses": total}) if pulsed else None
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})


#: The fixture beam's 1/e^2 diameter at the mirror, from Gaussian propagation: w0 = 0.5 mm at the laser, 100 mm away,
#: Rayleigh range pi w0^2 / lambda.
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)
#: The test conditions an evidenced rating states besides its value: its test beam (the beam here, 1/e^2) and its
#: polarization (linear, as the fixture laser's). A test removes the one it is about by passing None.
STATED = dict(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", polarization_condition="linear", polarization="LINEAR")


def _pulsed(value: float, **extra) -> PowerRating:
    return PowerRating(regime="PULSED", value=value, unit="J/cm2", wavelength_nm=1064.0, pulse_duration_s=1e-8,
                       **{"repetition_rate_hz": 10.0, "status": "DECLARED", "basis": FIXTURE, **STATED, **extra})


def _cw(value: float, **extra) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=1064.0,
                       **{"status": "DECLARED", "basis": FIXTURE, **STATED, **extra})


def _mirror(laser, rating: PowerRating) -> tuple:
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(laser, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    element = next(e for e in result.exposures.elements if e.element_id == "M")
    return face.ratings[0], face, element, result.targets[0], result


def _condition(comparison, kind: str):
    return next(c for c in comparison.conditions if c.condition == kind)


def _state(condition) -> tuple:
    return condition.state, condition.effect or condition.reason


def _own_peak(face, *, fluence: bool) -> float:
    """The peak of the mirror's one beam from its own hit: 2 P / (pi w_x w_y) (W/cm2) or with E = P / f (J/cm2)."""
    (hit,) = face.hits
    amount = hit.incident_power_W / PULSE.repetition_rate_hz if fluence else hit.incident_power_W
    return 2.0 * amount / (math.pi * hit.beam_radius_x_mm * hit.beam_radius_y_mm) * 100.0  # per mm2 -> per cm2


def _fluence() -> float:
    comparison, face, _, _, _ = _mirror(_laser(total="UNLIMITED"), _pulsed(1e30, value_definition="PEAK", pulse_count=1))
    own = _own_peak(face, fluence=True)
    assert comparison.exposure_value == pytest.approx(own, rel=1e-12)
    return own


def _irradiance() -> float:
    comparison, face, _, _, _ = _mirror(_laser(pulsed=False), _cw(1e30, value_definition="PEAK"))
    own = _own_peak(face, fluence=False)
    assert comparison.exposure_value == pytest.approx(own, rel=1e-12)
    return own


# ------------------------------------------------------------------------------------------------ A: pulse count


def test_reproduction_a_count_free_rating_against_a_train_that_keeps_pulsing():
    """The reviewed head reported WITHIN_RATING / MET for count=None and UNKNOWN for counts 1 and 100. Removing the count
    must not make the result more certain: without it nothing relates the rated value to UNLIMITED pulses."""
    f = _fluence()
    found = {}
    for count in (None, 1, 100):
        comparison, _, _, target, _ = _mirror(_laser(total="UNLIMITED"), _pulsed(2 * f, pulse_count=count, value_definition="PEAK"))
        found[count] = (comparison.status, target.assessment.value, _state(_condition(comparison, "pulse_count")))
        assert comparison.exposure_value == pytest.approx(f, rel=1e-12)  # the exposure stays
    assert found[None] == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE", ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"))
    # a single-pulse rating: every site of a train that keeps pulsing receives more than one pulse -> at most the rated value
    assert found[1] == ("UNKNOWN", "INDETERMINATE", ("DIFFERS", "THRESHOLD_AT_MOST_RATED"))
    # 100 pulses without how they were applied: no relation established
    assert found[100] == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE", ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"))


@pytest.mark.parametrize("total", TOTALS)
def test_a_rating_without_a_count_establishes_no_relation_to_any_total(total):
    """Omitted, finite, a single shot or UNLIMITED: the rating does not say how many pulses its test applied, so the count
    condition is the rating's gap (RATING_INFORMATION_MISSING), never a match; the exposure is still exact. Its ratio to
    the rated value (f / value: 2, then 0.5) is reported as an estimate that assumes the count; it decides nothing."""
    f = _fluence()
    for value in (0.5 * f, 2.0 * f):  # an exposure above, then below the value: neither direction is decided
        comparison, _, element, target, _ = _mirror(_laser(total=total), _pulsed(value, value_definition="PEAK"))
        assert _state(_condition(comparison, "pulse_count")) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
        assert (comparison.status, target.assessment.value, target.status.value) == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE", "UNKNOWN")
        assert comparison.exposure_value == pytest.approx(f, rel=1e-12)
        assert comparison.utilization == target.value == element.max_utilization == pytest.approx(f / value, rel=1e-12)
        for entry in (comparison.value_basis["utilization"], element.value_basis["max_utilization"], target.value_basis["value"]):
            assert entry.kind.value == "ESTIMATE" and "pulse_count" in entry.assumed[0].at
        assert element.utilization_lower_bound is None and element.utilization_upper_bound is None


def test_a_missing_rating_count_and_a_missing_operating_total_are_told_apart():
    """The rating's gap is RATING_INFORMATION_MISSING (stating the total would not help); the customer's gap is
    DESIGN_INPUT_MISSING only where the rating's count has a per-site meaning (stating the total would decide it)."""
    f = _fluence()
    cases = {
        "no count, no total": dict(),
        "100 pulses without protocol, no total": dict(pulse_count=100),
        "single pulse, no total": dict(pulse_count=1),
        "100 pulses S-on-1, no total": dict(pulse_count=100, test_protocol="S_ON_1"),
    }
    found = {name: _state(_condition(_mirror(_laser(total=None), _pulsed(2 * f, value_definition="PEAK", **extra))[0], "pulse_count"))
             for name, extra in cases.items()}
    assert found == {
        "no count, no total": ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"),
        "100 pulses without protocol, no total": ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"),
        "single pulse, no total": ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"),
        "100 pulses S-on-1, no total": ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"),
    }


@pytest.mark.parametrize(("total", "without", "with_protocol"), [
    # fewer pulses than the test: a site that survived 100 pulses at a level survived its first one -> at least rated
    (1, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")),
    (99, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")),
    # the test's own count on every site: verified
    (100, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), ("VERIFIED", None)),
    # more: a site damaged within 100 pulses is damaged within more -> at most rated
    (101, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
    ("UNLIMITED", ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
])
def test_a_known_count_needs_its_protocol(total, without, with_protocol):
    """100 pulses with the protocol unstated (no supplier text states one) decide nothing: under a ramped or raster test
    the rated value may lie above or below the one-level value. Stated S-on-1, the irreversibility of damage gives the
    direction."""
    f = _fluence()
    plain, _, _, _, _ = _mirror(_laser(total=total), _pulsed(2 * f, pulse_count=100, value_definition="PEAK"))
    stated, _, _, _, _ = _mirror(_laser(total=total), _pulsed(2 * f, pulse_count=100, test_protocol="S_ON_1", value_definition="PEAK"))
    assert _state(_condition(plain, "pulse_count")) == without and plain.status == "CONDITIONS_NOT_VERIFIED"
    assert _state(_condition(stated, "pulse_count")) == with_protocol
    assert plain.exposure_value == stated.exposure_value == pytest.approx(f, rel=1e-12)


@pytest.mark.parametrize(("total", "factor", "status", "assessment"), [
    # an evidenced direction decides only its own side: at least rated -> within; at most rated -> exceeds
    (1, 2.0, "WITHIN_RATING", "MET"),      # exposure 0.5 x the value, threshold at least the value
    (1, 0.5, "UNKNOWN", "INDETERMINATE"),  # above the value, but the threshold may be higher
    (100, 2.0, "WITHIN_RATING", "MET"),    # as rated
    (100, 0.5, "EXCEEDS_RATING", "UNMET"),
    ("UNLIMITED", 0.5, "EXCEEDS_RATING", "UNMET"),   # twice a threshold that is at most the value
    ("UNLIMITED", 2.0, "UNKNOWN", "INDETERMINATE"),  # below the value, but the threshold may be lower
])
def test_adequately_evidenced_comparisons_stay_decidable(total, factor, status, assessment):
    """A declared rating stating its definition (PEAK), its count and its protocol (S-on-1): the decisions the evidence
    supports are made, and only those."""
    f = _fluence()
    comparison, _, _, target, _ = _mirror(_laser(total=total), _pulsed(factor * f, pulse_count=100, test_protocol="S_ON_1",
                                                                       value_definition="PEAK"))
    assert (comparison.status, target.assessment.value) == (status, assessment)


def test_a_single_shot_under_a_single_pulse_rating_is_the_test_itself():
    """Count 1 means one pulse on each tested site; a single shot crossing the surface once is that, with no protocol."""
    f = _fluence()
    comparison, _, _, target, _ = _mirror(_laser(total=1), _pulsed(1.25 * f, pulse_count=1, value_definition="PEAK"))
    assert _state(_condition(comparison, "pulse_count")) == ("VERIFIED", None)
    assert (comparison.status, target.assessment.value, target.value) == ("WITHIN_RATING", "MET", pytest.approx(0.8, rel=1e-12))


def test_removing_evidence_never_makes_a_result_more_certain():
    """From a fully evidenced rating, take away the protocol, then the count, then the definition: the set of decided
    directions only shrinks, and the exposure never changes."""
    f = _fluence()
    decided = {"WITHIN_RATING": 2, "EXCEEDS_RATING": 2, "UNKNOWN": 1, "CONDITIONS_NOT_VERIFIED": 0}
    for total, factor in ((1, 2.0), (100, 0.5), ("UNLIMITED", 0.5)):
        steps = [dict(pulse_count=100, test_protocol="S_ON_1", value_definition="PEAK"),
                 dict(pulse_count=100, value_definition="PEAK"), dict(value_definition="PEAK"), dict()]
        ranks, exposures = [], set()
        for extra in steps:
            comparison, _, _, _, _ = _mirror(_laser(total=total), _pulsed(factor * f, **extra))
            ranks.append(decided[comparison.status])
            exposures.add(comparison.exposure_value)
        assert ranks == sorted(ranks, reverse=True), (total, ranks)
        assert ranks[-1] == 0 and len(exposures) == 1


# ------------------------------------------------------------------------------------------------ B: rated value


def test_reproduction_b_an_exposure_of_0_8_r_under_a_value_of_unstated_definition():
    """The reviewed head reported a threshold interval [R, 2R] and WITHIN_RATING / MET. Were the value the energy over a
    FWHM-diameter area, the peak threshold would be R ln 2 < 0.8 R: missing metadata cannot prove "within"."""
    peak = _irradiance()
    value = peak / 0.8
    assert value * math.log(2.0) < peak < value  # the exposure lies above the FWHM-area reading's threshold
    comparison, _, element, target, _ = _mirror(_laser(pulsed=False), _cw(value))
    assert comparison.exposure_value == pytest.approx(peak, rel=1e-12)  # the exposure stays
    assert _state(_condition(comparison, "value_definition")) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    assert comparison.rated_threshold_low is None and comparison.rated_threshold_high is None
    # (the ratio to the value as stated, 0.8: an estimate that reads the value as a peak value -- no threshold, no verdict)
    assert comparison.utilization == pytest.approx(0.8, rel=1e-12)
    definition = [a for a in comparison.value_basis["utilization"].assumed if a.at == ("value_definition",)]
    assert comparison.value_basis["utilization"].kind.value == "ESTIMATE" and [a.value for a in definition] == ["PEAK"]
    assert comparison.status == "CONDITIONS_NOT_VERIFIED"
    assert element.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (None, None)
    assert element.max_utilization == pytest.approx(0.8, rel=1e-12)
    assert element.value_basis["max_utilization"].kind.value == "ESTIMATE"
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert target.value == pytest.approx(0.8, rel=1e-12) and target.value_basis["value"].kind.value == "ESTIMATE"
    assert "value definition: information the rating does not state" in target.message


@pytest.mark.parametrize("factor", [0.1, 0.8, 1.43, 2.5, 10.0])
@pytest.mark.parametrize("kind", ["fluence", "irradiance"])
def test_a_value_of_unstated_definition_decides_nothing_at_any_exposure(kind, factor):
    """From a tenth of the value to ten times it: no threshold, never a decision. The ratio to the value as stated
    (``factor``) is reported as an estimate that reads the value as a peak value; no status or assessment reads it."""
    if kind == "fluence":
        exposure = _fluence()
        comparison, _, element, target, _ = _mirror(_laser(total=1), _pulsed(exposure / factor, pulse_count=1))
    else:
        exposure = _irradiance()
        comparison, _, element, target, _ = _mirror(_laser(pulsed=False), _cw(exposure / factor))
    assert comparison.exposure_value == pytest.approx(exposure, rel=1e-12)
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.utilization == pytest.approx(factor, rel=1e-12)
    assert comparison.value_basis["utilization"].kind.value == "ESTIMATE"
    assert comparison.rated_threshold_low is None and comparison.rated_threshold_high is None
    assert element.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert element.utilization_lower_bound is None and element.utilization_upper_bound is None
    assert element.max_utilization == pytest.approx(factor, rel=1e-12) and element.value_basis["max_utilization"].kind.value == "ESTIMATE"
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert target.value == pytest.approx(factor, rel=1e-12) and target.value_basis["value"].kind.value == "ESTIMATE"


def test_the_stated_definitions_and_their_conversion():
    """PEAK: the threshold is the value. TOTAL_PER_1E2_AREA: the beam's own total over the area of its 1/e^2 diameter,
    P / (pi w_x w_y), is half its peak 2 P / (pi w_x w_y); a value of exactly that ratio is a threshold at the exposure."""
    peak = _irradiance()
    _, face, _, _, _ = _mirror(_laser(pulsed=False), _cw(1e30, value_definition="PEAK"))
    (hit,) = face.hits
    total_over_area = hit.incident_power_W / (math.pi * hit.beam_radius_x_mm * hit.beam_radius_y_mm) * 100.0
    assert peak == pytest.approx(2.0 * total_over_area, rel=1e-12)
    # (the mean within the 1/e^2 diameter is (1 - e^-2) of that ratio: another quantity, not this definition)
    assert (1.0 - math.exp(-2.0)) * total_over_area < total_over_area
    for definition, value, threshold in (("PEAK", peak, peak), ("TOTAL_PER_1E2_AREA", total_over_area, 2.0 * total_over_area)):
        comparison, _, element, target, _ = _mirror(_laser(pulsed=False), _cw(value, value_definition=definition))
        assert _state(_condition(comparison, "value_definition")) == ("VERIFIED", None)
        assert _condition(comparison, "value_definition").rated == definition
        assert comparison.rated_threshold_low == comparison.rated_threshold_high == pytest.approx(threshold, rel=1e-12)
        assert comparison.status == "WITHIN_RATING" and element.max_utilization == pytest.approx(1.0, rel=1e-9)
        above, _, _, _, _ = _mirror(_laser(pulsed=False, power_W=1.01), _cw(value, value_definition=definition))
        assert above.status == "EXCEEDS_RATING"


def test_a_power_per_diameter_value_and_a_typical_value_have_no_definition_condition():
    """W/cm (P / d) is its own threshold; a typical value bounds none: neither is read through value_definition."""
    peak = _irradiance()
    linear = PowerRating(regime="CW", value=1e9, unit="W/cm", wavelength_nm=1064.0, status="DECLARED", basis=FIXTURE, **STATED)
    typical = _cw(10.0 * peak, status="TYPICAL")
    for rating in (linear, typical):
        comparison, _, _, _, _ = _mirror(_laser(pulsed=False), rating)
        assert _state(_condition(comparison, "value_definition")) == ("NOT_STATED", None)
    comparison, _, _, _, _ = _mirror(_laser(pulsed=False), typical)
    assert comparison.status == "UNKNOWN" and comparison.rated_threshold_low is None
    # (exposure / typical value, 0.1: an estimate that takes the typical value as the part's own)
    assert comparison.utilization == pytest.approx(0.1, rel=1e-12)
    (assumed,) = comparison.value_basis["utilization"].assumed
    assert comparison.value_basis["utilization"].kind.value == "ESTIMATE" and (assumed.at, assumed.value) == (("rating_status",), "TYPICAL")


def test_an_undefined_rating_holds_back_a_within_but_not_an_exceedance():
    """Beside a stated rating, a rating of unstated definition that applies to the light may be the stricter one: the
    surface is not within; an exceedance of the stated rating stands."""
    peak = _irradiance()
    for stated, surface in ((2.0 * peak, "CONDITIONS_NOT_VERIFIED"), (0.5 * peak, "EXCEEDS_RATING")):
        mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(
            update={"power_ratings": (_cw(stated, value_definition="PEAK"), _cw(4.0 * peak))})
        target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
        result = run(ProjectSnapshot(components=(_laser(pulsed=False), mirror), targets=(target,)))
        face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
        element = next(e for e in result.exposures.elements if e.element_id == "M")
        assert face.rating_status == surface
        assert element.utilization_upper_bound is None  # the undefined rating bounds nothing from above
        if surface == "EXCEEDS_RATING":
            assert element.utilization_lower_bound == pytest.approx(2.0, rel=1e-9)
            assert result.targets[0].assessment.value == "UNMET"
        else:
            assert result.targets[0].assessment.value == "INDETERMINATE"


def test_the_trace_depth_does_not_depend_on_the_rating_count_or_the_totals():
    """The exposure the comparison reports is the same whatever the rating states about its count and protocol and
    whatever the source states as its total."""
    f = _fluence()
    exposures = set()
    for total in TOTALS:
        for extra in (dict(), dict(pulse_count=100), dict(pulse_count=100, test_protocol="S_ON_1"), dict(pulse_count=1)):
            comparison, _, _, _, _ = _mirror(_laser(total=total), _pulsed(2 * f, value_definition="PEAK", **extra))
            exposures.add(comparison.exposure_value)
    assert len(exposures) == 1


# ------------------------------------------------------------------------------------------------ consumers


def test_the_result_round_trips_and_an_older_record_still_reads():
    """A comparison serializes and reads back unchanged; a record written before the value_definition condition existed
    (seven conditions, thresholds present) still validates."""
    _, _, _, _, result = _mirror(_laser(pulsed=False), _cw(_irradiance() * 1.25))
    payload = result.model_dump(mode="json")
    assert ValidationResult.model_validate(payload).model_dump(mode="json") == payload
    surface = next(s for s in payload["exposures"]["surfaces"] if s["surface_id"] == "M/face")
    comparison = surface["ratings"][0]
    assert [c["condition"] for c in comparison["conditions"]][-1] == "value_definition"
    comparison["conditions"] = comparison["conditions"][:-1]
    comparison.update(rated_threshold_low=1.0, rated_threshold_high=2.0, status="WITHIN_RATING", applicability="AS_RATED")
    ValidationResult.model_validate(payload)


def test_the_power_ratings_export_states_the_missing_definition_and_no_threshold():
    from os_optics.results.tables import power_rating_rows

    _, _, _, _, result = _mirror(_laser(pulsed=False), _cw(_irradiance() * 1.25))
    (row,) = power_rating_rows(result, "M")
    assert "value_definition=NOT_VERIFIABLE(RATING_INFORMATION_MISSING)" in row["conditions"]
    assert row["rated_threshold_low"] in (None, "") and row["rated_threshold_high"] in (None, "")
    assert row["utilization"] == pytest.approx(0.8, rel=1e-12) and row["status"] == "CONDITIONS_NOT_VERIFIED"
    assert row["value_basis"]["utilization"]["kind"] == "ESTIMATE"  # (the ratio's column carries what the number is)
    # a typical value establishes no threshold either: an empty cell, not a missing column (the export failed on it)
    _, _, _, _, result = _mirror(_laser(pulsed=False), _cw(_irradiance() * 1.25, status="TYPICAL"))
    (row,) = power_rating_rows(result, "M")
    assert row["rated_threshold_low"] in (None, "") and row["status"] == "UNKNOWN"


def test_a_declared_rating_carries_its_protocol_and_definition_through_the_project():
    """Project -> lowering -> run: a declared model's rating states test_protocol and value_definition, and the comparison
    uses them (capabilities names both features)."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source

    from os_optics.local import json_api

    assert {"rating_test_protocol", "rating_value_definition"} <= set(json_api.capabilities()["project_features"])
    laser = source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, power_w=0.5)
    laser["source_model"]["pulse"] = {"repetition_rate_hz": 10.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM",
                                      "total_pulses": "UNLIMITED"}
    rating = {"regime": "PULSED", "value": "0.1", "unit": "J/cm2", "wavelength_nm": "780", "pulse_duration_s": "0.00000001",
              "repetition_rate_hz": "10", "pulse_count": 100, "test_protocol": "S_ON_1", "value_definition": "PEAK",
              # (the 1 mm waist 200 mm away is 2.0 mm across at the window, 1/e^2; the source is linearly polarized)
              "beam_diameter_mm": "2", "beam_diameter_definition": "1/E2", "polarization_condition": "linear",
              "status": "DECLARED", "basis": FIXTURE}
    document = project([laser, declared_component("W", "window", "win", (0.0, 0.0, 0.0))], [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)])
    document = with_models(document, [{"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]),
                                       "power_ratings": [rating]}], [nbk7()])
    reply = json_api.run_project(document)
    assert reply["ok"], reply
    exposures = json_api.run_results(reply["files"], {"kind": "exposures"})["result"]
    comparisons = [r for s in exposures["surfaces"] for r in s["ratings"]]
    assert comparisons
    for c in comparisons:
        states = {x["condition"]: (x["state"], x.get("effect")) for x in c["conditions"]}
        assert states["pulse_count"] == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
        assert states["value_definition"] == ("VERIFIED", None)
    assert any(c["status"] == "EXCEEDS_RATING" for c in comparisons)  # ~3 J/cm2 per pulse against 0.1 J/cm2


# ------------------------------------------------------------------------------------------------ unstated test conditions


DECIDED = ("WITHIN_RATING", "EXCEEDS_RATING")


@pytest.mark.parametrize(("kind", "stated", "removed"), [
    # a test beam smaller than the beam here, the same, larger
    ("beam", dict(beam_diameter_mm=0.3 * TEST_BEAM_MM), dict(beam_diameter_mm=None, beam_diameter_definition=None)),
    ("beam", dict(), dict(beam_diameter_mm=None, beam_diameter_definition=None)),
    ("beam", dict(beam_diameter_mm=3.0 * TEST_BEAM_MM), dict(beam_diameter_mm=None, beam_diameter_definition=None)),
    # the light's polarization, another one
    ("polarization", dict(), dict(polarization_condition=None, polarization=None)),
    ("polarization", dict(polarization_condition="s-polarized", polarization="S"), dict(polarization_condition=None, polarization=None)),
    # the light's rate, another one
    ("rate", dict(), dict(repetition_rate_hz=None)),
    ("rate", dict(repetition_rate_hz=1000.0), dict(repetition_rate_hz=None)),
])
@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_removing_a_stated_test_condition_never_makes_a_result_more_certain(kind, stated, removed, factor):
    """A test beam, rate or polarization the rating does not state was treated as no
    constraint: removing a stated test beam turned CONDITIONS_MISMATCH into WITHIN_RATING. An unstated test condition the
    threshold depends on leaves the comparison not verifiable (RATING_INFORMATION_MISSING); the exposure stays."""
    f = _fluence()
    rating = dict(pulse_count=1, value_definition="PEAK")
    with_it, _, _, target_with, _ = _mirror(_laser(total=1), _pulsed(factor * f, **{**rating, **stated}))
    without, _, _, target_without, _ = _mirror(_laser(total=1), _pulsed(factor * f, **{**rating, **stated, **removed}))
    condition = {"beam": "beam_diameter", "polarization": "polarization", "rate": "repetition_rate"}[kind]
    assert _state(_condition(without, condition)) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    assert without.status == "CONDITIONS_NOT_VERIFIED" and target_without.assessment.value == "INDETERMINATE"
    assert (without.status in DECIDED) <= (with_it.status in DECIDED)
    assert without.exposure_value == pytest.approx(f, rel=1e-12)
    # (a stated rate the light differs from makes the rating one for other light: no exposure in its quantity)
    assert with_it.exposure_value in (without.exposure_value, None)


def test_a_power_per_diameter_rating_without_its_test_beam_decides_nothing():
    """The review's W/cm case: 0.5 x P/d with a 0.3 mm test beam does not apply to a 1 mm beam (a W/cm value has no
    relation for a larger beam); without the test beam it was an exceedance. Now it is not verifiable."""
    found = {}
    for beam in (0.3, None):
        linear = PowerRating(regime="CW", value=1.0, unit="W/cm", wavelength_nm=1064.0, status="DECLARED", basis=FIXTURE,
                             **{**STATED, "beam_diameter_mm": beam, "beam_diameter_definition": "1/E2" if beam else None})
        comparison, _, _, target, _ = _mirror(_laser(pulsed=False), linear)
        found[beam] = (comparison.status, target.assessment.value)
    assert found == {0.3: ("CONDITIONS_MISMATCH", "INDETERMINATE"), None: ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE")}


def test_a_metallic_rating_without_a_count_holds_back_a_within():
    """The review's metal case: beside a single-pulse rating the light is within, a micro-spot rating at half the
    exposure exceeds when it states its single pulse; without the count it may still be a single-pulse rating, so it is
    not verifiable and the surface is not within (it was dropped as 'no relationship', giving PASS)."""
    f = _fluence()
    metal = dict(beam_diameter_mm=0.1, surface_kind="METALLIC_REFLECTOR", value_definition="PEAK")
    found = {}
    for count in (1, None):
        ratings = (_pulsed(2.0 * f, pulse_count=1, value_definition="PEAK", surface_kind="METALLIC_REFLECTOR"),
                   _pulsed(0.5 * f, pulse_count=count, **metal))
        mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
        target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
        result = run(ProjectSnapshot(components=(_laser(total=1), mirror), targets=(target,)))
        face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
        found[count] = (face.rating_status, result.targets[0].assessment.value,
                        _state(_condition(face.ratings[1], "beam_diameter")))
    assert found[1] == ("EXCEEDS_RATING", "UNMET", ("DIFFERS", "THRESHOLD_AT_MOST_RATED"))
    assert found[None] == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE", ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"))


# ------------------------------------------------------------------------------------------------ surfaces with two ratings


TILT = math.radians(3.0)
NEAR_NORMAL = (-math.cos(TILT), math.sin(TILT), 0.0)


def _surface(ratings, total):
    """A mirror 3 degrees from normal incidence carrying ``ratings``: (surface status, target assessment, applicabilities)."""
    laser = _laser(total=total)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=NEAR_NORMAL).model_copy(update={"power_ratings": tuple(ratings)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(laser, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    return face.rating_status, result.targets[0].assessment.value, tuple(c.applicability for c in face.ratings)


def test_removing_a_beam_definition_does_not_reverse_an_exceedance():
    """The surface case: B (single pulse, 1/e^2 test beam matching the beam) is at most rated under a burst
    of two pulses and exceeded; A (S-on-1, 100 pulses) is within. Without B's definition, one reading of its test beam
    still verifies the beam, so B is not verifiable (it was a direction cancelling the count's, dropping B, and the surface
    became WITHIN / MET)."""
    f = _fluence()
    a = _pulsed(10.0 * f, pulse_count=100, test_protocol="S_ON_1", value_definition="PEAK", surface_kind="OTHER",
                beam_diameter_mm=1.02)
    b = dict(pulse_count=1, value_definition="PEAK", surface_kind="OTHER", beam_diameter_mm=1.02)
    stated = _surface([a, _pulsed(0.5 * f, **b)], 2)
    removed = _surface([a, _pulsed(0.5 * f, **{**b, "beam_diameter_definition": None})], 2)
    assert stated[:2] == ("EXCEEDS_RATING", "UNMET")
    assert removed[:2] == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE") and removed[2][1] == "NOT_VERIFIED"


DECIDED_SURFACE = {"WITHIN_RATING": "W", "EXCEEDS_RATING": "E"}
DECIDED_TARGET = {"MET": "W", "UNMET": "E"}


def test_removing_one_fact_of_a_second_rating_never_decides_a_surface_more():
    """A surface carries A (every fact stated, within on its own) and B (every fact stated); removing any one fact of B, or
    the source's total, never makes the surface or the target decided where it was not, nor reverses it (the final
    review's sweep, reduced: it found removals that turned UNKNOWN into WITHIN, and EXCEEDS into WITHIN)."""
    f = _fluence()
    a = _pulsed(10.0 * f, pulse_count=10000, test_protocol="S_ON_1", value_definition="PEAK", surface_kind="OTHER",
                beam_diameter_mm=1.02)
    removals = {
        "beam definition": ("beam_diameter_definition",), "test beam": ("beam_diameter_mm", "beam_diameter_definition"),
        "rate": ("repetition_rate_hz",), "polarization": ("polarization", "polarization_condition"),
        "value definition": ("value_definition",), "protocol": ("test_protocol",), "count": ("pulse_count", "test_protocol"),
        "surface kind": ("surface_kind",),
    }
    violations = []
    for ratio, definition, (count, protocol), kind, factor, total in itertools.product(
            (0.98, 2.0), ("1/E2", "FWHM"), ((1, None), (100, "S_ON_1")), ("OTHER", "METALLIC_REFLECTOR"), (0.5, 2.0), (1, "UNLIMITED")):
        diameter = ratio * TEST_BEAM_MM * (1.0 if definition == "1/E2" else FWHM_PER_E2)
        full = dict(pulse_count=count, test_protocol=protocol, value_definition="PEAK", surface_kind=kind,
                    beam_diameter_mm=round(diameter, 6), beam_diameter_definition=definition)
        before = _surface([a, _pulsed(factor * f, **full)], total)
        variants = [(name, {**full, **{field: None for field in fields}}, total) for name, fields in removals.items()
                    if not (name == "protocol" and protocol is None)]
        variants.append(("total", full, None))
        for name, fields, tot in variants:
            after = _surface([a, _pulsed(factor * f, **fields)], tot)
            for i, table in ((0, DECIDED_SURFACE), (1, DECIDED_TARGET)):
                if after[i] in table and table.get(after[i]) != table.get(before[i]):
                    violations.append((name, ratio, definition, count, kind, factor, total, before[:2], after[:2]))
    assert violations == []


def test_the_metallic_message_names_the_missing_count_only_when_it_is_missing():
    found = {}
    for count in (1, None):
        rating = _pulsed(1.0, pulse_count=count, value_definition="PEAK", surface_kind="METALLIC_REFLECTOR", beam_diameter_mm=10.0)
        comparison, _, _, _, _ = _mirror(_laser(total=None), rating)
        found[count] = _condition(comparison, "beam_diameter").message
    assert "the rating states no pulse count" not in found[1] and "pulse.total_pulses" in found[1]
    assert "the rating states no pulse count" in found[None]


def _linear_rating(value: float, test_beam_mm: float | None, definition: str | None, polarized: bool) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm", wavelength_nm=1064.0, status="DECLARED", basis=FIXTURE,
                       beam_diameter_mm=test_beam_mm, beam_diameter_definition=definition if test_beam_mm else None,
                       **({"polarization": "LINEAR", "polarization_condition": "linear"} if polarized else {}))


def _linear_surface(ratings):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=NEAR_NORMAL).model_copy(update={"power_ratings": tuple(ratings)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(_laser(pulsed=False), mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    return face.rating_status, result.targets[0].assessment.value


def test_two_w_per_cm_ratings_are_compared_in_one_quantity():
    """The W/cm case: A (20 W/cm, 5 mm test beam, no definition) applies, at least rated, and is within; B
    (25 W/cm, 3 mm FWHM test beam, no polarization) is not verified. Read in P / d of the 1/e^2 diameter, B's value may lie
    below A's lowest threshold, so the surface is undecided -- also when B's definition is removed (the definition label
    was part of the quantity, so removing it made B 'the same quantity' as A and bounded by it: WITHIN / MET)."""
    a = _linear_rating(20.0, 5.0, None, True)
    for definition in ("FWHM", None):
        assert _linear_surface([a, _linear_rating(25.0, 3.0, definition, False)]) == ("CONDITIONS_NOT_VERIFIED", "INDETERMINATE")
    # bounded under every reading: B's lowest 1/e^2 reading of 40 W/cm (40 x 0.5887 = 23.5) above A's 20 -> within
    assert _linear_surface([a, _linear_rating(40.0, 3.0, None, False)]) == ("WITHIN_RATING", "MET")


def test_removing_a_w_per_cm_definition_or_test_beam_never_decides_a_surface_more():
    """The review's W/cm pair sweep (reduced): A applies (polarization stated, a 5 mm test beam), B is not verified (no
    polarization); removing either one's definition or test beam never makes the surface or target decided otherwise."""
    violations = []
    for da, db, va, vb in itertools.product(("1/E2", "FWHM", None), ("1/E2", "FWHM", None), (18.0, 30.0), (12.0, 20.0, 40.0)):
        before = _linear_surface([_linear_rating(va, 5.0, da, True), _linear_rating(vb, 3.0, db, False)])
        for who in ("A definition", "B definition", "A test beam", "B test beam"):
            a = (va, None if who == "A test beam" else 5.0, None if who.startswith("A") else da, True)
            b = (vb, None if who == "B test beam" else 3.0, None if who.startswith("B") else db, False)
            after = _linear_surface([_linear_rating(*a), _linear_rating(*b)])
            for x, y in zip(after, before):
                if x in DECIDED_SURFACE or x in DECIDED_TARGET:
                    if {**DECIDED_SURFACE, **DECIDED_TARGET}[x] != {**DECIDED_SURFACE, **DECIDED_TARGET}.get(y):
                        violations.append((who, da, va, db, vb, before, after))
    assert violations == []
