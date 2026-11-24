"""The rated value as a damage threshold (``validation/rating_conditions.py`` ``rated_threshold``).

The Engine's exposure is a peak value (peak fluence, peak irradiance) or P / d. A J/cm2 or W/cm2 value is a peak value
(``value_definition`` PEAK, the threshold for the peak exposure is the value) or the test beam's total energy or power
over the area of its 1/e^2 diameter (TOTAL_PER_1E2_AREA, half the peak of a Gaussian test beam: the threshold is twice the
value). A value whose definition is not stated establishes no threshold: test reports use several definitions and no set
of them is known to be exhaustive, so its ``value_definition`` condition is not verifiable (RATING_INFORMATION_MISSING)
and the comparison is CONDITIONS_NOT_VERIFIED, its exposure still reported. No catalog rating states a definition. A
TYPICAL value bounds no threshold. A W/cm value (P / d) is its own threshold.

Fixtures: an ideal mirror at 45 degrees with one CW beam (exact peak irradiance), fixture ratings written here; the
catalog cases read real products of the committed release 2026-10-15.1.
"""

from __future__ import annotations

import json
import math

import pytest

from os_optics import builders as B
from os_optics._vendor.shared_contracts.design_catalog import PowerRatingV1, PublicOpticalProductV2
from os_optics.contracts import PowerRating, ProjectSnapshot, Target
from os_optics.local.lowering import catalog_surface_kind, engine_power_ratings
from os_optics.validation.rating_conditions import comparison_status, rated_threshold, value_definition_condition

from catalog_store_helpers import REAL_CATALOG
from helpers import run

FIXTURE = "test fixture rating, not product data"
LASER = B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
#: The laser's 1/e^2 diameter at the mirror 100 mm away (w0 = 0.5 mm, Rayleigh range pi w0^2 / lambda).
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)
#: The test conditions the fixture ratings state besides their value, met by the beam here: a test beam of its size and
#: linear polarization (the laser's), so that the value's reading alone decides.
STATED = dict(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", polarization_condition="linear", polarization="LINEAR")


@pytest.mark.parametrize(("unit", "status", "definition", "expected"), [
    # (a fluence or irradiance whose definition is not stated: a peak, a total over the 1/e^2 area or another
    # definition -- no threshold for the peak exposure follows from it)
    ("J/cm2", "SPECIFIED", None, (None, None)),
    ("W/cm2", "SPECIFIED", None, (None, None)),
    ("J/cm2", "SPECIFIED", "PEAK", (2.0, 2.0)),
    ("J/cm2", "SPECIFIED", "TOTAL_PER_1E2_AREA", (4.0, 4.0)),
    ("W/cm2", "SPECIFIED", "TOTAL_PER_1E2_AREA", (4.0, 4.0)),
    ("W/cm", "SPECIFIED", None, (2.0, 2.0)),
    ("J/cm2", "TYPICAL", None, (None, None)),
    ("J/cm2", "TYPICAL", "PEAK", (None, None)),
])
def test_the_threshold_a_rated_value_establishes(unit, status, definition, expected):
    """PEAK: the value is the threshold for the peak exposure; TOTAL_PER_1E2_AREA: a Gaussian test beam's peak is twice the
    total over its 1/e^2 area, so twice the value; unstated: the rating does not say which, so no threshold."""
    assert rated_threshold(2.0, unit, status, definition) == expected


@pytest.mark.parametrize(("applicability", "lower", "upper", "peak", "total"), [
    # (rated value 2: threshold 2 when stated as a peak, 4 when stated as the total over the 1/e^2 area)
    ("AS_RATED", 2.5, 3.5, "EXCEEDS_RATING", "WITHIN_RATING"),
    ("AS_RATED", 4.1, 4.5, "EXCEEDS_RATING", "EXCEEDS_RATING"),
    ("AS_RATED", 1.0, 2.0, "WITHIN_RATING", "WITHIN_RATING"),
    ("THRESHOLD_AT_MOST_RATED", 3.0, None, "EXCEEDS_RATING", "UNKNOWN"),
    ("THRESHOLD_AT_MOST_RATED", 4.5, None, "EXCEEDS_RATING", "EXCEEDS_RATING"),
    ("THRESHOLD_AT_LEAST_RATED", 0.5, 2.0, "WITHIN_RATING", "WITHIN_RATING"),
    ("THRESHOLD_AT_LEAST_RATED", 0.5, 2.5, "UNKNOWN", "WITHIN_RATING"),
])
def test_statuses_against_the_threshold_of_each_stated_definition(applicability, lower, upper, peak, total):
    """The same exposure bounds against the threshold each stated definition gives the value 2 J/cm2 (2 as a peak, 4 as a
    total over the 1/e^2 area); a value whose definition is not stated gives none, and decides neither way."""
    assert comparison_status(applicability, lower, upper, *rated_threshold(2.0, "J/cm2", "SPECIFIED", "PEAK")) == peak
    assert comparison_status(applicability, lower, upper,
                             *rated_threshold(2.0, "J/cm2", "SPECIFIED", "TOTAL_PER_1E2_AREA")) == total
    assert comparison_status(applicability, lower, upper, *rated_threshold(2.0, "J/cm2", "SPECIFIED")) == "UNKNOWN"


def test_a_typical_value_decides_neither_way():
    lowest, highest = rated_threshold(2.0, "J/cm2", "TYPICAL")
    assert comparison_status("AS_RATED", 100.0, 200.0, lowest, highest) == "UNKNOWN"
    assert comparison_status("AS_RATED", 0.0, 1e-9, lowest, highest) == "UNKNOWN"


# ------------------------------------------------------------------------------------------------ runs


def _cw(value: float, **extra) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=1064.0, status=extra.pop("status", "DECLARED"),
                       basis=FIXTURE, **{**STATED, **extra})


def _mirror(rating: PowerRating):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    element = next(e for e in result.exposures.elements if e.element_id == "M")
    return face.ratings[0], element, result.targets[0]


def _peak() -> float:
    comparison, _, _ = _mirror(_cw(1e30))
    assert comparison.exposure_value is not None
    return comparison.exposure_value


@pytest.mark.parametrize(("factor", "definition", "status", "assessment"), [
    # the exposure 1.43 times the value: above a peak value, within a total over the 1/e^2 area (twice the value)
    (1 / 0.7, "PEAK", "EXCEEDS_RATING", "UNMET"),
    (1 / 0.7, "TOTAL_PER_1E2_AREA", "WITHIN_RATING", "MET"),
    # 2.5 times the value: exceeded under both definitions
    (2.5, "PEAK", "EXCEEDS_RATING", "UNMET"),
    (2.5, "TOTAL_PER_1E2_AREA", "EXCEEDS_RATING", "UNMET"),
    # at most the value: within under both definitions
    (0.8, "PEAK", "WITHIN_RATING", "MET"),
    (0.8, "TOTAL_PER_1E2_AREA", "WITHIN_RATING", "MET"),
])
def test_one_beam_against_each_stated_definition(factor, definition, status, assessment):
    """A stated definition gives the threshold (the value, or twice it): the comparison decides, the per-comparison
    utilization is exposure / rated value and the element's is exposure / threshold."""
    peak = _peak()
    comparison, element, target = _mirror(_cw(peak / factor, value_definition=definition))
    assert comparison.exposure_value == pytest.approx(peak, rel=1e-12)  # (the exposure is the same whatever the definition)
    assert comparison.status == status and target.assessment.value == assessment
    assert comparison.utilization == pytest.approx(factor, rel=1e-12)  # (exposure / rated value)
    expected = factor / 2.0 if definition == "TOTAL_PER_1E2_AREA" else factor
    assert element.max_utilization == pytest.approx(expected, rel=1e-12)
    assert target.value == pytest.approx(expected, rel=1e-12)
    reading = next(c for c in comparison.conditions if c.condition == "value_definition")
    assert (reading.state, reading.rated) == ("VERIFIED", definition)
    threshold = 2.0 * peak / factor if definition == "TOTAL_PER_1E2_AREA" else peak / factor
    assert (comparison.rated_threshold_low, comparison.rated_threshold_high) == pytest.approx((threshold,) * 2, rel=1e-12)


@pytest.mark.parametrize("factor", [0.8, 1 / 0.7, 2.5])
def test_one_beam_against_a_value_whose_definition_is_not_stated(factor):
    """The rating does not state whether its W/cm2 value is the test beam's peak or a total over some area: it establishes
    no threshold for the peak exposure, so the comparison decides neither within (even below the value) nor exceeds (even
    above twice it); the exposure is still reported, and so is its ratio to the value read as the peak, an estimate whose
    entry names that reading; the target stays undecided with that ratio."""
    peak = _peak()
    comparison, element, target = _mirror(_cw(peak / factor))
    assert comparison.exposure_value == pytest.approx(peak, rel=1e-12)
    assert (comparison.applicability, comparison.status) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED")
    reading = next(c for c in comparison.conditions if c.condition == "value_definition")
    assert (reading.state, reading.reason, reading.rated) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING", None)
    assert (comparison.rated_threshold_low, comparison.rated_threshold_high) == (None, None)
    assert not any("twice it" in note for note in comparison.notes)
    assert (element.rating_status, element.utilization_upper_bound) == ("CONDITIONS_NOT_VERIFIED", None)
    # (exposure / rated value = factor: the value read as the test beam's peak, stated in each entry)
    for value, entry in ((comparison.utilization, comparison.value_basis["utilization"]),
                         (element.max_utilization, element.value_basis["max_utilization"]),
                         (target.value, target.value_basis["value"])):
        assert value == pytest.approx(factor, rel=1e-12)
        assert entry.kind.value == "ESTIMATE" and entry.omitted == () and entry.validity == ()
        assert [(a.input.value, a.at, a.value, a.code) for a in entry.assumed] == [
            ("UNSTATED_CONDITION_AS_DESIGN", ("value_definition",), "PEAK", "POWER_RATING_CONDITIONS_NOT_VERIFIED")]
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert "value definition: information the rating does not state" in target.message
    assert "between the value and twice it" not in target.message


@pytest.mark.parametrize("factor", [0.1, 1.5, 10.0])
def test_a_typical_value_leaves_the_comparison_and_the_target_undecided(factor):
    peak = _peak()
    comparison, element, target = _mirror(_cw(peak / factor, status="TYPICAL", value_definition="PEAK"))
    assert comparison.status == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
    assert (comparison.rated_threshold_low, comparison.rated_threshold_high) == (None, None)
    assert any("typical value (status TYPICAL)" in note for note in comparison.notes)
    assert element.utilization_upper_bound is None
    assert "a typical value (status TYPICAL) bounds the damage threshold of no particular part" in target.message


def test_a_power_per_diameter_value_has_no_area_reading():
    comparison, _, _ = _mirror(PowerRating(regime="CW", value=1e30, unit="W/cm", wavelength_nm=1064.0, beam_diameter_definition="1/E2",
                                           status="DECLARED", basis=FIXTURE))
    assert comparison.rated_threshold_low == comparison.rated_threshold_high == 1e30


# ------------------------------------------------------------------------------------------------ contract and catalog


def test_value_definition_is_refused_on_a_power_per_diameter_rating():
    stated = {"regime": "CW", "value": "100", "unit": "W/cm", "wavelength_nm": "1064", "status": "SPECIFIED", "basis": FIXTURE,
              "value_definition": "PEAK"}
    with pytest.raises(ValueError, match="value_definition"):
        PowerRatingV1.model_validate(stated)
    with pytest.raises(ValueError, match="value_definition"):
        PowerRating(regime="CW", value=1.0, unit="W/cm", wavelength_nm=1064.0, status="DECLARED", value_definition="PEAK")
    assert PowerRatingV1.model_validate(dict(stated, unit="W/cm2")).value_definition == "PEAK"


def test_no_catalog_rating_states_a_value_definition_and_their_payload_is_unchanged():
    catalog = json.loads(REAL_CATALOG.read_text(encoding="utf-8"))
    typical = 0
    for document in catalog["products"]:
        published = document["specifications"].get("power_ratings") or []
        if not published:
            continue
        product = PublicOpticalProductV2.model_validate(document)
        for raw, rating in zip(published, product.specifications.power_ratings):
            assert "value_definition" not in raw and rating.value_definition is None
            assert "test_protocol" not in raw and rating.test_protocol is None
            assert rating.model_dump(mode="json", exclude_none=True) == raw
        for rating in engine_power_ratings(product.specifications.power_ratings, "catalog", surface_kind=catalog_surface_kind(product)):
            lowest, highest = rated_threshold(rating.value, rating.unit, rating.status, rating.value_definition)
            reading = value_definition_condition(rating)
            if rating.status == "TYPICAL":
                typical += 1
                assert (lowest, highest) == (None, None)
                assert reading.state == "NOT_STATED"
            elif rating.unit == "W/cm":
                assert lowest == highest == rating.value
                assert reading.state == "NOT_STATED"
            else:
                # (no catalog text states whether a J/cm2 or W/cm2 value is a peak or a total over an area: no threshold)
                assert (lowest, highest) == (None, None)
                assert (reading.state, reading.reason) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    assert typical == 48  # the catalog's typical values ("for reference only"), recounted in the rating audit


def test_a_surface_only_a_typical_value_applies_to_whose_conditions_cannot_be_checked():
    """A typical value bounds nothing; when its conditions cannot be checked either (a polarization stated only as text),
    the surface says so rather than leaving it undecided."""
    comparison, element, _ = _mirror(_cw(0.1 * _peak(), status="TYPICAL", polarization_condition="s-polarized", polarization=None))
    assert (comparison.applicability, comparison.status) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED")
    assert element.rating_status == "CONDITIONS_NOT_VERIFIED"


def test_a_typical_value_beside_a_specified_one_neither_decides_nor_holds_it_back():
    """A specified peak rating the exposure meets and a typical value it exceeds tenfold on the same surface: the
    specified rating decides (within), the typical value adds nothing either way."""
    peak = _peak()
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (
        _cw(2.0 * peak, value_definition="PEAK"), _cw(0.1 * peak, status="TYPICAL"))})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    element = next(e for e in result.exposures.elements if e.element_id == "M")
    specified, typical = face.ratings
    assert (specified.status, typical.status, face.rating_status) == ("WITHIN_RATING", "UNKNOWN", "WITHIN_RATING")
    assert element.max_utilization == pytest.approx(0.5, rel=1e-12)
    assert result.targets[0].assessment.value == "MET"
