"""Each condition of a power rating against the light on a surface (``validation/rating_conditions.py``,
``RatingComparison.conditions``) and the rating's applicability.

Condition by condition: the test beam diameter compared with the beams' 1/e^2 sizes (verified within 5 % when the
definition is stated; a smaller beam has a threshold at least the rated value, a larger one at most; a definition not
stated leaves a band between its readings not verifiable; metallic reflectors under pulsed ratings and W/cm ratings with a
larger beam have no established direction), the pulses per site (``pulse_count_condition`` against the pulses one site
receives, ``PulsesPerSite``, compared only for a single-pulse rating or a count stated as S-on-1: exactly the test's,
more, fewer, crossings that may together bring more; no count, or a count above one without its protocol: not
verifiable), the polarization (unchanged), the value definition (a fixture's J/cm2 or W/cm2 value stated as a peak; no
catalog rating states one), the W/cm exposure of a rating that does not state its diameter definition (a range over the
1/e^2 and FWHM readings) and the record itself.

Ratings marked "fixture" are test data written here. The catalog cases (E1-E9) read the ratings of real products from
the committed catalog release 2026-10-15.1 by product id and lower them with the engine's own lowering; the design
illuminations are written here (a round beam of the stated 1/e^2 diameter on the surface, at normal incidence).
"""

from __future__ import annotations

import functools
import json
import math

import pytest

from os_optics import builders as B
from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalProductV2
from os_optics.contracts import PowerRating, ProjectSnapshot, PulseTrain, RatingCondition, Target
from os_optics.local import lower_project, run_optical_project
from os_optics.local.lowering import catalog_surface_kind, declared_surface_kind, engine_power_ratings
from os_optics.validation.rating_conditions import (
    CONDITION_KINDS,
    FWHM_PER_E2_DIAMETER,
    BeamSize,
    Light,
    PulsesPerSite,
    beam_diameter_condition,
    beam_diameter_decision,
    comparison_status,
    pulse_count_condition,
    rated_beam_interval,
    rating_applicability,
    scalar_conditions,
    value_definition_condition,
)

from catalog_store_helpers import REAL_CATALOG
from helpers import run

FIXTURE = "test fixture rating, not product data"
INF = math.inf
#: The fixture laser's 1/e^2 diameter at the mirror of ``_mirror_run`` 100 mm away (w0 = 0.5 mm, Rayleigh range
#: pi w0^2 / lambda): a test beam its beam verifies.
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)

CUBE_515 = "OS-OPT-BS-CUBE-A10-R47T47-400-700-NBK7-CA8"
SAMPLER_532 = "OS-OPT-BS-SAMPLER-D12P7-R1-400-700-W0P5-AOI45-T3-UVFS-CA11P43"
SAMPLER_800 = "OS-OPT-BS-SAMPLER-D12P7-R1-700-1100-W0P5-AOI45-T3-UVFS-CA11P43"
PBS_532 = "OS-OPT-PBS-CUBE-A25P4-532-ER3000-TP95RS99P5-UVFS-CA20P32"
AG_MIRROR = "OS-OPT-MIR-CONCAVE-D12P7-F100-AG-480-20000-R96-AOI0T10-T3P08-CA10P16-BOROFLOAT33"
HR_400 = "OS-OPT-MIR-FLAT-D12P7-HR-360-400-R99-AOI45-T6-CA10P795-UVFS"
ND_515 = "OS-OPT-FLT-ND-D25P4-OD1-400-650-ABS-T1P9-CA22P86"
UV_AL = "OS-OPT-MIR-FLAT-D12P7-UVAL-250-600-R90-AOI45-T3-CA11P43-BOROFLOAT33"


@functools.lru_cache(maxsize=1)
def _catalog() -> dict:
    return json.loads(REAL_CATALOG.read_text(encoding="utf-8"))


def _product(sku: str) -> PublicOpticalProductV2:
    return PublicOpticalProductV2.model_validate(next(p for p in _catalog()["products"] if p["optical_sku"] == sku))


def _catalog_rating(sku: str, **match) -> PowerRating:
    """One of a product's published ratings, lowered as a run lowers it (``engine_power_ratings``, ``catalog_surface_kind``)."""
    product = _product(sku)
    ratings = engine_power_ratings(product.specifications.power_ratings, f"catalog product {sku}",
                                   surface_kind=catalog_surface_kind(product))
    found = [r for r in ratings if all(getattr(r, k) == v for k, v in match.items())]
    assert len(found) == 1, (sku, match, ratings)
    return found[0]


def _fixture(value: float = 1.0, *, regime: str = "CW", unit: str = "W/cm2", wavelength_nm: float = 1064.0, **extra) -> PowerRating:
    """A fixture rating; a J/cm2 or W/cm2 value is stated as a peak value (``value_definition`` PEAK, the threshold the
    decisions here are read against) unless ``value_definition`` is given, and its test polarization as linear (every
    fixture laser here is linear) unless a polarization is given."""
    if unit != "W/cm":
        extra.setdefault("value_definition", "PEAK")
    if "polarization" not in extra and "polarization_condition" not in extra:
        extra.update(polarization="LINEAR", polarization_condition="linear")
    return PowerRating(regime=regime, value=value, unit=unit, wavelength_nm=wavelength_nm, status="DECLARED", basis=FIXTURE, **extra)


def _classify(rating: PowerRating, wavelength_nm: float, beam_mm: float, pulse: tuple[float, float] | None = None,
              pulses: float | None = None) -> tuple[dict[str, RatingCondition], str]:
    """A design illumination written here: one round beam ``beam_mm`` across (1/e^2) at normal incidence, the given line and
    pulse train (duration, rate), crossing the surface once with ``pulses`` from its source (``None``: the source states no
    ``total_pulses``), the rating's polarization taken as met where it states one (none stated: not verifiable, as a run
    reads it). The engine's own condition functions classify it, the rated value's definition included
    (``value_definition_condition``)."""
    train = None if pulse is None else PulseTrain(pulse_duration_s=pulse[0], repetition_rate_hz=pulse[1])
    scalar = scalar_conditions(rating, [Light("D", wavelength_nm, train)])
    beam = beam_diameter_condition(rating, [BeamSize((beam_mm, beam_mm), beam_mm, "D")], design_pulses=pulses)
    count = pulse_count_condition(rating, pulses)
    polar = (RatingCondition(condition="polarization", state="VERIFIED") if rating.polarization else
             RatingCondition(condition="polarization", state="NOT_VERIFIABLE", reason="RATING_INFORMATION_MISSING"))
    conditions = (*scalar, beam, count, polar, value_definition_condition(rating))
    return {c.condition: c for c in conditions}, rating_applicability(conditions)


def _state(condition: RatingCondition) -> tuple:
    return (condition.state, condition.effect or condition.reason)


# ------------------------------------------------------------------------------------- test beam diameter (fixtures)


def test_the_test_beam_interval_follows_the_stated_definition():
    assert rated_beam_interval(_fixture(beam_diameter_mm=1.0, beam_diameter_definition="1/E2")) == (1.0, 1.0, True)
    assert rated_beam_interval(_fixture(beam_diameter_mm=0.5887, beam_diameter_definition="FWHM")) == (
        0.5887 / FWHM_PER_E2_DIAMETER, 0.5887 / FWHM_PER_E2_DIAMETER, True)
    for definition in ("NOT_STATED", None):  # (and any spelling the contract does not know)
        assert rated_beam_interval(_fixture(beam_diameter_mm=0.8, beam_diameter_definition=definition)) == (
            0.8, 0.8 / FWHM_PER_E2_DIAMETER, False)
    assert rated_beam_interval(_fixture()) is None
    assert FWHM_PER_E2_DIAMETER == pytest.approx(math.sqrt(2.0 * math.log(2.0)) / 2.0, rel=1e-15)


@pytest.mark.parametrize(("case", "diameter", "definition", "beam", "expected", "applicability"), [
    ("F1", 1.0, "1/E2", 1.009, ("VERIFIED", None), "AS_RATED"),
    ("F2", 0.5887, "FWHM", 1.0, ("VERIFIED", None), "AS_RATED"),
    ("F3", 2.0, "1/E2", 1.0, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), "THRESHOLD_AT_LEAST_RATED"),
    ("F4", 0.1, "1/E2", 1.0, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "THRESHOLD_AT_MOST_RATED"),
    ("F5", 0.8, "NOT_STATED", 1.0, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), "NOT_VERIFIED"),
    ("within 5 %", 1.0, "1/E2", 1.049, ("VERIFIED", None), "AS_RATED"),
    ("just above 5 %", 1.0, "1/E2", 1.051, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "THRESHOLD_AT_MOST_RATED"),
    ("just below 5 %", 1.0, "1/E2", 0.949, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), "THRESHOLD_AT_LEAST_RATED"),
    # a definition not stated: below every reading by more than the tolerance -> at least; above them -> at most; a beam a
    # reading would verify (within 5 % of d or of d / 0.5887) -> not verifiable: the outcome depends on the reading
    ("not stated, no larger", 0.8, "NOT_STATED", 0.75, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), "THRESHOLD_AT_LEAST_RATED"),
    ("not stated, above 1.699 d", 0.8, None, 1.45, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "THRESHOLD_AT_MOST_RATED"),
    ("not stated, at d", 0.8, "NOT_STATED", 0.8, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), "NOT_VERIFIED"),
    ("not stated, just below d", 0.8, "NOT_STATED", 0.77, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), "NOT_VERIFIED"),
    ("not stated, at 1.699 d", 0.8, None, 1.36, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING"), "NOT_VERIFIED"),
])
def test_a_fixture_test_beam_against_one_beam(case, diameter, definition, beam, expected, applicability):
    conditions, found = _classify(_fixture(beam_diameter_mm=diameter, beam_diameter_definition=definition), 1064.0, beam)
    assert _state(conditions["beam_diameter"]) == expected, case
    assert found == applicability, case


def test_what_each_applicability_lets_the_exposure_decide():
    """AS_RATED decides both ways; at least rated (F3) only WITHIN -- above the rating it is UNKNOWN, never EXCEEDS; at most
    rated (F4) only EXCEEDS -- within the number it is UNKNOWN, never WITHIN; not verified and not applicable decide
    nothing."""
    assert [comparison_status("AS_RATED", lo, hi, 1.0) for lo, hi in ((0.5, 0.9), (1.1, 1.2), (0.9, 1.1))] == [
        "WITHIN_RATING", "EXCEEDS_RATING", "UNKNOWN"]
    assert [comparison_status("THRESHOLD_AT_LEAST_RATED", lo, hi, 1.0) for lo, hi in ((0.5, 0.9), (1.1, 1.2), (0.9, None))] == [
        "WITHIN_RATING", "UNKNOWN", "UNKNOWN"]
    assert [comparison_status("THRESHOLD_AT_MOST_RATED", lo, hi, 1.0) for lo, hi in ((0.5, 0.9), (1.1, 1.2), (None, 2.0))] == [
        "UNKNOWN", "EXCEEDS_RATING", "UNKNOWN"]
    assert comparison_status("NOT_VERIFIED", 1.1, 1.2, 1.0) == "CONDITIONS_NOT_VERIFIED"
    assert comparison_status("NOT_APPLICABLE", 1.1, 1.2, 1.0) == "CONDITIONS_MISMATCH"


def test_beams_of_several_sizes_combine():
    """Smaller and larger beams on one surface: no direction holds for the light as a whole; a beam whose size is not
    known leaves it not verifiable; a beam whose axes straddle a stated diameter has no established relationship."""
    rating = _fixture(beam_diameter_mm=1.0, beam_diameter_definition="1/E2")
    small, large = BeamSize((0.5, 0.5), 0.5, "a"), BeamSize((3.0, 3.0), 3.0, "b")
    verified = BeamSize((1.0, 1.0), 1.0, "c")
    assert _state(beam_diameter_condition(rating, [small, large])) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert _state(beam_diameter_condition(rating, [small, verified])) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")
    assert _state(beam_diameter_condition(rating, [large, verified])) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
    assert _state(beam_diameter_condition(rating, [large], unknown_size="the size of the beam of x is not known")) == (
        "NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN")
    assert _state(beam_diameter_condition(rating, [])) == ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN")
    elliptic = BeamSize((0.9, 1.2), 1.2, "e")
    assert _state(beam_diameter_condition(rating, [elliptic])) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    tilted = BeamSize((0.9, 0.9), 1.27, "t")  # (a 0.9 mm beam at 45 deg: its footprint is 1.27 mm long)
    assert _state(beam_diameter_condition(rating, [tilted])) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    grazing = BeamSize((0.5, 0.5), None, "g")
    assert _state(beam_diameter_condition(rating, [grazing])) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")


def test_light_of_unknown_size_beyond_the_tolerance_withholds_only_a_verified_or_smaller_beam():
    """Light bounded by its diffraction limit (light left out as negligible, untraced light of unknown size) may be larger
    than the test beam: it keeps a verified or at-least direction from holding when it may carry more than 1 % of the
    light; an at-most direction stands (the exposure's lower bound already allows for it)."""
    rating = _fixture(beam_diameter_mm=1.0, beam_diameter_definition="1/E2")
    unsized = "light of unknown size may reach the surface with up to 5 % of the most light there"
    record, decisive = beam_diameter_decision(rating, [BeamSize((0.5, 0.5), 0.5)], unsized=unsized)
    assert (_state(record), decisive) == (("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN"), True) and unsized in record.message
    record, decisive = beam_diameter_decision(rating, [BeamSize((1.0, 1.0), 1.0)], unsized=unsized)
    assert (_state(record), decisive) == (("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN"), True)
    record, decisive = beam_diameter_decision(rating, [BeamSize((3.0, 3.0), 3.0)], unsized=unsized)
    assert (_state(record), decisive) == (("DIFFERS", "THRESHOLD_AT_MOST_RATED"), False)


def test_a_metallic_reflector_keeps_a_direction_across_beam_sizes_only_for_one_pulse_per_site():
    """Pulsed fixture ratings on a metallic reflector: multi-pulse -- no established relationship whatever the pulses per
    site; single-pulse -- not verifiable while the pulses per site are not stated, the direction kept for one pulse per
    site, none for a continuing train. A CW rating on a metallic reflector keeps its direction."""
    pulse = (1e-8, 10.0)

    def metal(count: int | None, surface_kind: str | None = "METALLIC_REFLECTOR") -> PowerRating:
        return _fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0, beam_diameter_mm=0.01,
                        beam_diameter_definition="1/E2", pulse_count=count, surface_kind=surface_kind)

    for pulses in (None, 1, INF):
        conditions, found = _classify(metal(100), 1064.0, 2.0, pulse, pulses)
        assert _state(conditions["beam_diameter"]) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and found == "NOT_APPLICABLE"
    conditions, found = _classify(metal(1), 1064.0, 0.008, pulse, None)
    assert _state(conditions["beam_diameter"]) == ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING") and found == "NOT_VERIFIED"
    conditions, found = _classify(metal(1), 1064.0, 0.008, pulse, 1)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED") and found == "THRESHOLD_AT_LEAST_RATED"
    conditions, found = _classify(metal(1), 1064.0, 2.0, pulse, INF)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and found == "NOT_APPLICABLE"
    # a reflector of unknown kind is never taken for a dielectric one where that would change the result
    conditions, found = _classify(metal(100, None), 1064.0, 2.0, pulse, None)
    assert _state(conditions["beam_diameter"]) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING") and found == "NOT_VERIFIED"
    conditions, _ = _classify(metal(1, None), 1064.0, 0.008, pulse, 1)  # (single pulse on either kind: the same direction)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")
    conditions, _ = _classify(metal(100, "OTHER"), 1064.0, 2.0, pulse, None)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
    cw = _fixture(beam_diameter_mm=0.01, beam_diameter_definition="1/E2", surface_kind="METALLIC_REFLECTOR")
    assert _state(_classify(cw, 1064.0, 2.0)[0]["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")


def test_a_w_per_cm_rating_has_no_direction_for_a_larger_beam():
    rating = _fixture(unit="W/cm", beam_diameter_mm=0.1, beam_diameter_definition="1/E2")
    assert _state(_classify(rating, 1064.0, 1.0)[0]["beam_diameter"]) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert _state(_classify(rating, 1064.0, 0.05)[0]["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")


# ------------------------------------------------------------------------------------------------- pulses per site


AT_MOST_RATED = ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
AT_LEAST_RATED = ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")
#: A count the rating states without how it was applied to each tested site, or no count at all: not verifiable.
PROTOCOL = ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
#: ``PowerRating.test_protocol``: each tested site received the rating's count at one fluence level.
S_ON_1 = "S_ON_1"


@pytest.mark.parametrize(("rated", "protocol", "design", "expected"), [
    # a count stated as applied at one fluence level to each tested site (S-on-1): compared per site
    (100, S_ON_1, None, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING")),
    (100, S_ON_1, 100, ("VERIFIED", None)),
    # more pulses per site than the test: a site damaged within 100 pulses at a fluence is damaged within more
    (100, S_ON_1, 101, AT_MOST_RATED), (100, S_ON_1, 1000, AT_MOST_RATED), (100, S_ON_1, INF, AT_MOST_RATED),
    # fewer: a site that survived 100 pulses at a fluence survived its first fewer
    (100, S_ON_1, 1, AT_LEAST_RATED), (100, S_ON_1, 99, AT_LEAST_RATED), (10000, S_ON_1, 1, AT_LEAST_RATED),
    # a count above one that does not state how its pulses were applied (one level per site, ramped, raster): no relation
    # of its value to any number of pulses per site is established, whatever the total
    (100, None, None, PROTOCOL), (100, None, 100, PROTOCOL), (100, None, 101, PROTOCOL), (100, None, INF, PROTOCOL),
    (100, None, 99, PROTOCOL), (10000, None, 1, PROTOCOL),
    # a single-pulse rating: one pulse on each tested site is the test itself; more pulses lower the threshold
    (1, None, None, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING")), (1, None, 1, ("VERIFIED", None)), (1, None, 2, AT_MOST_RATED),
    (1, None, INF, AT_MOST_RATED), (1, S_ON_1, 1, ("VERIFIED", None)), (1, S_ON_1, INF, AT_MOST_RATED),
    # a rating that states no count: how many pulses each tested site received is not stated, whatever the total
    (None, None, None, PROTOCOL), (None, None, INF, PROTOCOL), (None, None, 1, PROTOCOL), (None, None, 10**9, PROTOCOL),
])
def test_the_pulses_per_site(rated, protocol, design, expected):
    """A rating's count is compared with the pulses one site receives only where it states pulses per tested site: a
    single pulse, or a count with its S-on-1 protocol. A count above one without the protocol, or no count, gives no
    relation to any number of pulses, so the condition is not verifiable whatever the source sends."""
    extra = {} if protocol is None else dict(test_protocol=protocol)
    rating = _fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, pulse_count=rated, **extra)
    condition = pulse_count_condition(rating, design)
    assert _state(condition) == expected
    assert (condition.rated, condition.unit) == (rated, "pulses")
    compared = rated == 1 or protocol == S_ON_1
    if rated is None:
        assert condition.message == "pulse count: the rating states none, so how many pulses each tested site received is not known"
    elif not compared:
        assert "not how they were applied to each tested site" in condition.message
    head = ("pulse count: the rating states a single pulse, read as one pulse on each tested site" if rated == 1 else
            f"pulse count: the rating states {rated} pulses on each tested site at one fluence level (S-on-1)")
    if design is None and compared:
        assert condition.message == f"{head}; the light's source states no pulse.total_pulses (how many pulses a source " \
                                    "sends while the optics stay in place)"
    if design is not None:  # (the pulses a site receives are reported whether or not the rating relates to them)
        assert condition.design == ("UNLIMITED" if design == INF else design)
        assert condition.design_low is None and condition.design_high is None


@pytest.mark.parametrize(("rated", "design", "expected", "low", "high"), [
    # (the 100-pulse rating states S-on-1, so its count is compared per site)
    # two crossings of 60 pulses each: a site where both meet receives 120, one only one of them 60
    (100, PulsesPerSite(60.0, 120.0, crossings=2), ("NOT_VERIFIABLE", "PULSES_PER_SITE_UNDETERMINED"), 60.0, 120.0),
    # two of 50: no site receives more than the test's 100
    (100, PulsesPerSite(50.0, 100.0, crossings=2), AT_LEAST_RATED, 50.0, 100.0),
    (100, PulsesPerSite(100.0, 200.0, crossings=2), AT_MOST_RATED, 100.0, 200.0),
    (100, PulsesPerSite(10.0, INF, crossings=2), ("NOT_VERIFIABLE", "PULSES_PER_SITE_UNDETERMINED"), 10.0, None),
    # a single shot crossing twice (or with light of unknown place beyond the share): some site may receive two pulses
    (1, PulsesPerSite(1.0, 2.0, crossings=2), AT_MOST_RATED, 1.0, 2.0),
    (1, PulsesPerSite(1.0, INF), AT_MOST_RATED, 1.0, None),
])
def test_pulses_per_site_from_several_crossings(rated, design, expected, low, high):
    extra = dict(test_protocol=S_ON_1) if rated > 1 else {}
    rating = _fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, pulse_count=rated, **extra)
    condition = pulse_count_condition(rating, design)
    assert _state(condition) == expected
    assert (condition.design, condition.design_low, condition.design_high) == (None, low, high)


def test_an_unbounded_count_names_what_leaves_it_unbounded():
    rating = _fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, pulse_count=100, test_protocol=S_ON_1)
    why = "light left out as negligible, which may carry more than 1 % of the most light here, may bring more"
    condition = pulse_count_condition(rating, PulsesPerSite(1.0, INF, unbounded=why))
    assert _state(condition) == ("NOT_VERIFIABLE", "PULSES_PER_SITE_UNDETERMINED")
    assert why in condition.message and "whether one site receives fewer or more pulses than the test is not determined" in \
        condition.message
    assert (condition.design_low, condition.design_high) == (1.0, None)


def test_pulses_per_site_names_the_sources_without_a_total_and_continuous_light():
    rating = _fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, pulse_count=100, test_protocol=S_ON_1)
    missing = pulse_count_condition(rating, PulsesPerSite(INF, INF, missing=("LASER_B",), crossings=2))
    assert _state(missing) == ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING")
    assert "LASER_B states no pulse.total_pulses" in missing.message
    continuous = pulse_count_condition(rating, PulsesPerSite(0.0, 0.0, continuous=True))
    assert _state(continuous) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert continuous.message.endswith("the light here is continuous")


# ------------------------------------------------------------------------------------------- catalog ratings (E1-E9)


FS_515 = (290e-15, 100.0)


@pytest.mark.parametrize(("case", "beam", "pulses", "beam_state"), [
    ("E1a 2 mm, no pulses per site", 2.0, None, ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
    ("E1b 2 mm, continuing train", 2.0, INF, ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
    ("E1c 2 mm, single shot", 2.0, 1, ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
    ("E1d 8 um, single shot", 0.008, 1, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")),
    ("E1e 8 um, continuing train", 0.008, INF, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")),
    ("E1f 15 um (band 10.72-18.2 um), 100 pulses", 0.015, 100, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")),
])
def test_e1_a_cube_rated_at_515_nm_with_a_micro_spot_over_100_pulses(case, beam, pulses, beam_state):
    """The catalog text states "(100 Pulses)" but not how they were applied to each tested site, nor whether 0.36 J/cm2 is
    a peak: the count and the value definition are not verifiable whatever the pulses per site (a continuing train
    included), so the rating applies to none of these illuminations as stated; the test beam's size still gives its
    direction."""
    rating = _catalog_rating(CUBE_515)
    assert (rating.value, rating.beam_diameter_mm, rating.beam_diameter_definition, rating.pulse_count, rating.surface_kind) == (
        0.36, 0.01072, "NOT_STATED", 100, "OTHER")
    assert (rating.test_protocol, rating.value_definition) == (None, None)
    conditions, found = _classify(rating, 515.0, beam, FS_515, pulses)
    assert [c.state for c in conditions.values()][:4] == ["VERIFIED"] * 4, case
    assert (_state(conditions["beam_diameter"]), _state(conditions["pulse_count"]), _state(conditions["value_definition"])) == (
        beam_state, PROTOCOL, PROTOCOL), case
    assert found == "NOT_VERIFIED", case


@pytest.mark.parametrize(("beam", "expected"), [
    (0.3, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")),
    (1.0, ("DIFFERS", "THRESHOLD_AT_MOST_RATED")),
    (0.6, ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")),  # (band 0.491-0.834 mm)
])
def test_e2_a_sampler_rated_at_532_nm_with_a_0_491_mm_test_beam(beam, expected):
    """The pulsed rating states no pulse count and no value definition: both not verifiable, so it applies to none of these
    beams as stated; the test beam's size still gives its direction."""
    rating = _catalog_rating(SAMPLER_532, wavelength_nm=532.0)
    assert (rating.value, rating.beam_diameter_mm, rating.pulse_count) == (7.5, 0.491, None)
    conditions, found = _classify(rating, 532.0, beam, (1e-8, 10.0))
    assert (_state(conditions["beam_diameter"]), _state(conditions["pulse_count"]), found) == (expected, PROTOCOL, "NOT_VERIFIED")
    assert _state(conditions["value_definition"]) == PROTOCOL


def test_e4_a_2_mm_beam_is_larger_than_the_800_nm_sampler_s_test_beam():
    """The 0.166 mm test beam is smaller than a 2 mm beam: a threshold at most the rated value. The rating states neither a
    pulse count nor whether its value is a peak, so it does not apply as stated and decides no exceedance."""
    rating = _catalog_rating(SAMPLER_800, wavelength_nm=800.0)
    assert (rating.value, rating.beam_diameter_mm, rating.pulse_duration_s, rating.repetition_rate_hz) == (0.246, 0.166, 9.9e-14, 1000.0)
    conditions, found = _classify(rating, 800.0, 2.0, (100e-15, 1000.0))
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
    assert (_state(conditions["pulse_count"]), _state(conditions["value_definition"]), found) == (PROTOCOL, PROTOCOL, "NOT_VERIFIED")


def test_e5_the_532_nm_polarizing_cube_cw_rating():
    rating = _catalog_rating(PBS_532, regime="CW")
    assert (rating.value, rating.unit, rating.beam_diameter_mm, rating.surface_kind) == (750.0, "W/cm", 0.0368, "OTHER")
    conditions, found = _classify(rating, 532.0, 0.2)
    assert (_state(conditions["beam_diameter"]), found) == (("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP"), "NOT_APPLICABLE")
    conditions, found = _classify(rating, 532.0, 0.030)
    # (a smaller beam: at least the rated value -- but the rating states no test polarization, so it is not verified)
    assert (_state(conditions["beam_diameter"]), _state(conditions["polarization"]), found) == (
        ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), PROTOCOL, "NOT_VERIFIED")


def test_e6_a_silver_mirror_with_a_multi_pulse_micro_spot_rating_does_not_apply_at_any_pulses_per_site():
    rating = _catalog_rating(AG_MIRROR)
    assert (rating.surface_kind, rating.pulse_count, rating.beam_diameter_mm, rating.wavelength_nm) == (
        "METALLIC_REFLECTOR", 100, 0.01147, 1550.0)
    for beam, pulses in ((2.0, None), (2.0, INF), (0.008, 1)):
        conditions, found = _classify(rating, 1550.0, beam, (190e-15, 100.0), pulses)
        assert (_state(conditions["beam_diameter"]), found) == (("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP"), "NOT_APPLICABLE")


@pytest.mark.parametrize(("count", "pulses", "expected"), [
    (1, None, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING")), (1, 1, ("VERIFIED", None)),
    # (10000 pulses without how they were applied to each tested site: no relation to any pulses per site)
    (10000, INF, PROTOCOL), (10000, 1, PROTOCOL),
])
def test_e7_an_s_polarized_mirror_rated_at_one_and_at_ten_thousand_pulses(count, pulses, expected):
    """The single-pulse rating is compared per site (a single shot verifies it); the 10000-pulse one states no protocol.
    Neither states its test beam, its repetition rate or whether its value is a peak, so neither applies as stated."""
    rating = _catalog_rating(HR_400, pulse_count=count)
    assert (rating.polarization, rating.beam_diameter_mm, rating.repetition_rate_hz) == ("S", None, None)
    conditions, found = _classify(rating, 400.0, 2.0, (55e-15, 1000.0), pulses)
    assert (_state(conditions["beam_diameter"]), _state(conditions["repetition_rate"])) == (PROTOCOL, PROTOCOL)
    assert _state(conditions["pulse_count"]) == expected
    assert (_state(conditions["value_definition"]), found) == (PROTOCOL, "NOT_VERIFIED")


def test_e8_a_neutral_density_filter_rated_for_a_single_pulse():
    """A larger beam over a continuing train: both at most the rated value; a smaller single shot: at least it, its count
    verified. The value's definition is not stated, so neither applies as stated."""
    rating = _catalog_rating(ND_515, pulse_count=1)
    assert (rating.value, rating.beam_diameter_mm, rating.repetition_rate_hz) == (0.38, 0.01207, 20000.0)
    conditions, found = _classify(rating, 515.0, 2.0, (290e-15, 20000.0), INF)
    assert (_state(conditions["beam_diameter"]), _state(conditions["pulse_count"])) == (AT_MOST_RATED, AT_MOST_RATED)
    assert (_state(conditions["value_definition"]), found) == (PROTOCOL, "NOT_VERIFIED")
    conditions, found = _classify(rating, 515.0, 0.010, (290e-15, 20000.0), 1)
    assert (_state(conditions["beam_diameter"]), _state(conditions["pulse_count"])) == (AT_LEAST_RATED, ("VERIFIED", None))
    assert (_state(conditions["value_definition"]), found) == (PROTOCOL, "NOT_VERIFIED")


def test_e9_a_metallic_mirror_rating_without_a_test_beam_keeps_its_pulse_count_rule():
    """A single-pulse rating on a metallic reflector with no test beam: its count is compared per site as any single-pulse
    rating's (not verifiable without the source's total, at most rated for a continuing train); the value's definition
    is not stated, so it does not apply as stated."""
    rating = _catalog_rating(UV_AL)
    assert (rating.surface_kind, rating.beam_diameter_mm, rating.pulse_count) == ("METALLIC_REFLECTOR", None, 1)
    conditions, found = _classify(rating, 355.0, 1.0, (5e-9, 10.0), None)
    assert (_state(conditions["pulse_count"]), found) == (("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"), "NOT_VERIFIED")
    conditions, found = _classify(rating, 355.0, 1.0, (5e-9, 10.0), INF)
    assert (_state(conditions["pulse_count"]), _state(conditions["value_definition"]), found) == (
        AT_MOST_RATED, PROTOCOL, "NOT_VERIFIED")


def test_every_rated_catalog_mirror_has_a_known_surface_kind():
    kinds: dict[tuple[str, str], int] = {}
    for document in _catalog()["products"]:
        if not (document.get("specifications") or {}).get("power_ratings"):
            continue
        product = PublicOpticalProductV2.model_validate(document)
        key = (document["category"], catalog_surface_kind(product))
        kinds[key] = kinds.get(key, 0) + 1
    assert all(kind in ("METALLIC_REFLECTOR", "OTHER") for _, kind in kinds)
    assert kinds[("MIRROR", "METALLIC_REFLECTOR")] == 50 and kinds[("MIRROR", "OTHER")] == 82
    assert sum(kinds.values()) == 849


# ------------------------------------------------------------------------------------- the surface kind of declared models


def _declared(model: dict, coatings: list[dict] | None = None):
    from declared_helpers import declared_component, nbk7, with_models
    from local_helpers import plane, project, source

    rating = {"regime": "PULSED", "value": "1", "unit": "J/cm2", "wavelength_nm": "780", "pulse_duration_s": "0.00000001",
              "status": "DECLARED", "basis": FIXTURE}
    model = {**model, "power_ratings": [rating]}
    document = with_models(project([source(), declared_component("X", "other_optical", model["model_id"])], [plane("p", (100.0, 0.0, 0.0))]),
                           [model], [nbk7()], coatings)
    from os_optics.local.catalog import load_project

    loaded = load_project(document)
    return declared_surface_kind(loaded.optical_model(model["model_id"]), loaded)


def _coating(coating_id: str, kind: str) -> dict:
    return {"coating_id": coating_id, "kind": kind, "display_name": coating_id, "bands": [{"min_nm": "400", "max_nm": "1100",
            "reflectance": "0.9", "transmittance": "0" if kind == "METALLIC_REFLECTOR" else "0.1", "status": "DECLARED"}],
            "valid_aoi_deg": {"min_deg": "0", "max_deg": "45", "status": "DECLARED"}, "basis": FIXTURE}


def test_the_surface_kind_of_a_declared_model_follows_what_the_project_states():
    from declared_helpers import sequence, stop_model, surface, thin_lens_model

    window = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"])}
    assert _declared(window) == "OTHER"
    assert _declared(thin_lens_model("thin")) == "OTHER"
    assert _declared(stop_model("stop", plane_z_mm="0", opening_radius_mm="1", edge_radius_mm="12.7")) is None
    metal = {"model_id": "metal", "model": sequence([surface(1, "0", traced="REFLECTED", coating_id="AG")], [])}
    assert _declared(metal, [_coating("AG", "METALLIC_REFLECTOR")]) == "METALLIC_REFLECTOR"
    dielectric = {"model_id": "hr", "model": sequence([surface(1, "0", traced="REFLECTED", coating_id="HR")], [])}
    assert _declared(dielectric, [_coating("HR", "DIELECTRIC_HIGH_REFLECTOR")]) == "OTHER"
    bare = {"model_id": "bare", "model": sequence([surface(1, "0", traced="REFLECTED")], [])}
    assert _declared(bare) is None  # (a reflector whose coating the Project does not state)


# ------------------------------------------------------------------------------------- full runs: the record and polarization


def _mirror_run(ratings: tuple[PowerRating, ...], **laser):
    source = B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                     direction=(1.0, 0.0, 0.0), **laser)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
    result = run(ProjectSnapshot(components=(source, mirror), targets=(Target(target_id="u", device_id="M",
                                                                            metric="power_rating_utilization", maximum=1.0),)))
    return next(s for s in result.exposures.surfaces if s.surface_id == "M/face"), result


def test_the_record_lists_every_condition_once_and_derives_the_old_views():
    """The record of every comparison. A condition that does not apply to a rating (a CW rating's pulse duration, rate
    and count) is NOT_STATED; one that applies but that the rating does not state (its test beam, its polarization, a
    pulsed rating's rate) is not verifiable -- the threshold depends on it -- and keeps the rating from applying as
    stated."""
    stated = dict(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2")
    ratings = (_fixture(1e9, **stated), _fixture(1e9, beam_diameter_mm=0.8),
               _fixture(1e9, beam_diameter_mm=0.1, beam_diameter_definition="1/E2"), _fixture(1e9, wavelength_nm=532.0, **stated),
               _fixture(1e9, unit="W/cm", **stated), _fixture(1.0, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, pulse_count=10),
               _fixture(1e9, polarization=None, polarization_condition=None))
    face, _ = _mirror_run(ratings)
    for comparison in face.ratings:
        assert tuple(c.condition for c in comparison.conditions) == CONDITION_KINDS
        for c in comparison.conditions:
            assert (c.effect is not None) == (c.state == "DIFFERS") and (c.reason is not None) == (c.state == "NOT_VERIFIABLE")
            assert c.message.startswith(c.condition.replace("_", " ") + ":")
        assert comparison.conditions_match == (comparison.applicability == "AS_RATED")
        assert comparison.mismatches == tuple(c.message for c in comparison.conditions if c.state == "DIFFERS")
        assert comparison.unverified_conditions == tuple(c.message for c in comparison.conditions if c.state == "NOT_VERIFIABLE")
    assert [c.applicability for c in face.ratings] == ["AS_RATED", "NOT_VERIFIED", "THRESHOLD_AT_MOST_RATED", "NOT_APPLICABLE",
                                                       "AS_RATED", "NOT_APPLICABLE", "NOT_VERIFIED"]
    # the first rating states every condition that applies to a CW rating, and the beam meets each; the pulse conditions
    # do not apply to it
    assert [c.state for c in face.ratings[0].conditions] == ["VERIFIED", "VERIFIED", "NOT_STATED", "NOT_STATED", "VERIFIED",
                                                             "NOT_STATED", "VERIFIED", "VERIFIED"]
    assert face.ratings[0].conditions[2].message == "pulse duration: does not apply to this rating"
    # the last states neither its test beam nor its polarization: both not verifiable, never verified, never not stated
    unstated = {c.condition: _state(c) for c in face.ratings[6].conditions if c.state != "VERIFIED"}
    assert unstated == {"pulse_duration": ("NOT_STATED", None), "repetition_rate": ("NOT_STATED", None),
                        "beam_diameter": PROTOCOL, "pulse_count": ("NOT_STATED", None), "polarization": PROTOCOL}
    # a pulsed rating against continuous light: its regime, duration differ, and the rate it does not state is not
    # verifiable; no exposure in its unit
    pulsed = face.ratings[5]
    assert [c.state for c in pulsed.conditions[:4]] == ["DIFFERS", "VERIFIED", "DIFFERS", "NOT_VERIFIABLE"]
    assert pulsed.exposure_value is None and pulsed.exposure_upper_bound is None and pulsed.status == "CONDITIONS_MISMATCH"


def test_a_w_per_cm_rating_without_a_diameter_definition_is_compared_with_the_range_of_both_readings():
    """One beam on an ideal mirror (exact): P / d from the 1/e^2 diameter 2 w to the FWHM diameter w sqrt(2 ln 2) --
    WITHIN only above the FWHM value. The rating states a test beam but not its definition, so the beam is compared with
    every reading of it: a test beam larger than the beam's footprint under every reading gives a threshold at least the
    rated value, and a value below the 1/e^2 exposure is not an exceedance (one would need the rating to apply as
    stated, which needs the definition the range exists for)."""
    # (a test beam 5 % above the beam's 1/e^2 footprint on the 45 deg mirror, 2 w / cos 45 deg, w from Gaussian propagation)
    larger = dict(beam_diameter_mm=1.05 * math.sqrt(2.0) * TEST_BEAM_MM)
    face, result = _mirror_run((_fixture(1e9, unit="W/cm", **larger),))
    # (the interval comes from the two readings alone: the message says so, and no incomplete trace is claimed)
    (target,) = result.targets
    assert target.status.value == "PASS" and target.issue_codes == () and "its 1/e^2 and FWHM readings" in target.message
    (hit,) = face.hits
    w = min(hit.beam_radius_x_mm, hit.beam_radius_y_mm)
    e2, fwhm = hit.incident_power_W / (2.0 * w) * 10.0, hit.incident_power_W / (w * math.sqrt(2.0 * math.log(2.0))) * 10.0
    (comparison,) = face.ratings
    assert comparison.exposure_value is None
    assert comparison.exposure_lower_bound == pytest.approx(e2, rel=1e-12) and comparison.exposure_upper_bound == pytest.approx(fwhm, rel=1e-12)
    assert "the range from the 1/e^2 diameter 2 w (lower end) to the FWHM diameter" in comparison.notes[0]
    assert (comparison.applicability, _state(next(c for c in comparison.conditions if c.condition == "beam_diameter"))) == (
        "THRESHOLD_AT_LEAST_RATED", ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"))
    statuses = [_mirror_run((_fixture(value, unit="W/cm", **larger),))[0].ratings[0].status
                for value in (1.01 * fwhm, 0.5 * (e2 + fwhm), 0.99 * e2)]
    assert statuses == ["WITHIN_RATING", "UNKNOWN", "UNKNOWN"]
    stated = _mirror_run((_fixture(1e9, unit="W/cm", beam_diameter_definition="1/E2"),))[0].ratings[0]
    assert stated.exposure_value == pytest.approx(e2, rel=1e-12)  # (a stated definition: one value)


def test_a_beam_whose_size_is_not_known_leaves_the_test_beam_not_verifiable():
    from test_power_exposure_ratings import _document, _rating, _surface

    run_ = run_optical_project(_document([_rating("CW", 1.0e9, "W/cm2", beam_diameter_mm=0.5, beam_diameter_definition="1/E2")], stop=True))
    s1 = _surface(run_, "W/S1")
    assert s1.hits[0].reason == "GAUSSIAN_SEVERE_TRUNCATION"
    (comparison,) = s1.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(beam) == ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN") and "is not known" in beam.message
    assert comparison.applicability == "NOT_VERIFIED"


def test_the_polarization_states_map_into_the_record():
    from test_rating_polarization import _document, _rating, _surface, _tilted, _two_beams

    def polar(comparison):
        return _state(next(c for c in comparison.conditions if c.condition == "polarization"))

    for angle, expected in ((90.0, ("VERIFIED", None)), (0.0, ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP"))):
        document = _tilted(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="s-polarized")]))
        document["components"][0]["source_model"]["polarization"]["angle_deg"] = angle
        (comparison,) = _surface(run_optical_project(document), "W/S1").ratings
        assert polar(comparison) == expected
    normal = _surface(run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="s-polarized")])), "W/S1").ratings[0]
    assert polar(normal) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    text = _surface(run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="linear, vertical")])),
                    "W/S1").ratings[0]
    assert polar(text) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    crossed = run(_two_beams(90.0, "G", "LINEAR"))
    (comparison,) = next(s for s in crossed.exposures.surfaces if s.surface_id == "W/front").ratings
    assert polar(comparison) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED")


# ------------------------------------------------------------------------------------- shares in the rating's own quantity


def _two_beams_on_a_mirror(rating: PowerRating, beams: tuple[tuple[str, float, float, PulseTrain | None], ...]):
    """Beams (name, power W, waist radius mm at the mirror, pulse train) meeting one face of a dielectric fixture mirror at
    45 deg, each from its own source (mutually incoherent), and a utilization target of 1 on the mirror."""
    sources = []
    for name, power, w0, pulse in beams:
        laser = B.laser(name, wavelength_nm=1064.0, power_W=power, waist_radius_mm=w0, position=(-100.0, 0.0, 0.0),
                        direction=(1.0, 0.0, 0.0), waist_position_z_mm=100.0)
        if pulse is not None:
            laser = laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})
        sources.append(laser)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    result = run(ProjectSnapshot(components=(*sources, mirror),
                                 targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),)))
    face = next(x for x in result.exposures.surfaces if x.surface_id == "M/face")
    return face, result


def test_a_beam_of_another_size_is_weighed_in_the_w_per_cm_ratings_own_quantity():
    """A CW 400 W/cm fixture rating with a 0.1 mm 1/e^2 test beam. A 1 W beam focused to 0.05 mm alone: the threshold for
    it is at least the rated value, within. With a 50 W 4 mm beam on the same face -- under 1 % of the peak irradiance, but
    38 % of the P / d the rating is compared in -- the larger beam takes part in the size comparison: a W/cm rating has no
    established relationship for a beam larger than its test beam, so the rating does not apply and the target is not
    decided."""
    rating = _fixture(400.0, unit="W/cm", beam_diameter_mm=0.1, beam_diameter_definition="1/E2")
    focused, broad = ("A", 1.0, 0.025, None), ("Bbig", 50.0, 2.0, None)
    face, result = _two_beams_on_a_mirror(rating, (focused,))
    assert (face.ratings[0].applicability, face.ratings[0].status, result.targets[0].status.value) == (
        "THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING", "PASS")
    face, result = _two_beams_on_a_mirror(rating, (focused, broad))
    hits = {h.source_id: h for h in face.hits}
    assert hits["Bbig"].peak_irradiance_W_per_cm2 < 0.01 * (hits["A"].peak_irradiance_W_per_cm2 + hits["Bbig"].peak_irradiance_W_per_cm2)
    assert hits["Bbig"].linear_power_density_W_per_cm > 0.3 * (hits["A"].linear_power_density_W_per_cm + hits["Bbig"].linear_power_density_W_per_cm)
    (comparison,) = face.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(beam) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and beam.design_high == pytest.approx(4.0 * math.sqrt(2.0))
    assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
    assert result.targets[0].status.value == "UNKNOWN"


def test_a_micro_spot_is_weighed_in_the_fluence_rating_s_own_quantity():
    """A pulsed 0.1 J/cm2 fixture rating (10 ns at 100 kHz, a 0.1 mm 1/e^2 test beam, a dielectric surface, a single
    pulse per tested site; each source a single shot, so the count is verified and the beam size alone gives the
    direction). A 10 W 2 mm beam at 100 kHz alone: at most rated, 0.06 of the rating -- not decided. A 50 uW 10 Hz
    micro-spot 0.05 mm across on the same face carries under 1 % of the peak irradiance but 99 % of the fluence: it takes
    part in the size comparison, the beams lie on both sides of the test beam, and no direction holds -- not applicable,
    no exceedance and no failure from a threshold that is at least the rated value for the micro-spot."""
    rating = _fixture(0.1, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=1e5, beam_diameter_mm=0.1,
                      beam_diameter_definition="1/E2", surface_kind="OTHER", pulse_count=1)
    broad = ("Abig", 10.0, 1.0, PulseTrain(repetition_rate_hz=1e5, pulse_duration_s=1e-8, total_pulses=1))
    spot = ("Bspot", 5e-5, 0.025, PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=1e-8, total_pulses=1))
    face, result = _two_beams_on_a_mirror(rating, (broad,))
    assert (face.ratings[0].applicability, face.ratings[0].status, result.targets[0].status.value) == (
        "THRESHOLD_AT_MOST_RATED", "UNKNOWN", "UNKNOWN")
    face, result = _two_beams_on_a_mirror(rating, (broad, spot))
    hits = {h.source_id: h for h in face.hits}
    assert hits["Bspot"].peak_irradiance_W_per_cm2 < 0.01 * hits["Abig"].peak_irradiance_W_per_cm2
    assert hits["Bspot"].peak_fluence_J_per_cm2 > 0.9 * (hits["Abig"].peak_fluence_J_per_cm2 + hits["Bspot"].peak_fluence_J_per_cm2)
    (comparison,) = face.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(beam) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and "smaller" in beam.message and "larger" in beam.message
    assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
    assert result.targets[0].status.value == "UNKNOWN"
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in result.issues)


def _faint_class(key: str, wavelength_nm: float, power_w: float, pulse: PulseTrain | None):
    """A class of light left out as negligible alone (no beam here) of one line, in one piece, on an open path that meets no
    reflecting surface (crossing field the least, ``resonance.LEAST_CROSSINGS`` = 2): ``exposures._Class`` / ``_Faint``."""
    from os_optics.validation import exposures as X
    from os_optics.validation.resonance import LEAST_CROSSINGS

    energy = None if pulse is None else power_w / pulse.repetition_rate_hz
    return X._Class((), {}, False, X._Faint(power_w, energy, 1.0, wavelength_nm, (((key, 0), wavelength_nm, power_w, pulse),),
                                            roots=(math.sqrt(power_w),), crossings=LEAST_CROSSINGS))


def test_left_out_light_of_another_line_leaves_every_condition_it_differs_in_not_verifiable():
    """Light left out as negligible of a line the rating is not for, beyond the tolerance: each condition that line differs
    in is not verifiable -- another duration and another rate both, another wavelength and both pulse conditions -- never
    verified from the traced light alone."""
    from os_optics.validation import exposures as X

    rating = _fixture(1.0, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0)
    traced = [Light("S1", 1064.0, PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=10.0))]
    other = PulseTrain(pulse_duration_s=1e-9, repetition_rate_hz=1000.0)
    for wavelength, expected in ((1064.0, {"pulse_duration", "repetition_rate"}),
                                 (532.0, {"wavelength", "pulse_duration", "repetition_rate"})):
        texts, _ = X._faint_mismatch(rating, [], [_faint_class("S2", wavelength, 1e-6, other)], "M/face")
        assert set(texts) == expected
        conditions = {c.condition: c for c in scalar_conditions(rating, traced, texts)}
        for name in ("wavelength", "pulse_duration", "repetition_rate"):
            assert _state(conditions[name]) == (("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN") if name in expected else ("VERIFIED", None))
            assert conditions[name].message.startswith(name.replace("_", " ") + ":")
    cw = X._faint_mismatch(rating, [], [_faint_class("S3", 1064.0, 1e-6, None)], "M/face")[0]
    assert set(cw) == {"regime"}  # (continuous light: the regime differs, its pulse is not compared)


def test_left_out_light_of_another_rate_is_weighed_in_the_fluence_too():
    """Left-out light of the rating's line at 100 kHz (1 W) and of another repetition rate at 10 Hz (5 mW): the latter is
    0.5 % of the peak irradiance the surface can receive, within the tolerance, but 98 % of the peak fluence a J/cm2
    rating is compared in -- the repetition rate is not verifiable."""
    from os_optics.validation import exposures as X

    rating = _fixture(1.0, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=1e5)
    own, slow = PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=1e5), PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=10.0)
    classes = [_faint_class("S1", 1064.0, 1.0, own), _faint_class("S2", 1064.0, 5e-3, slow)]
    peak_share, _ = X._off_share([], classes, {("S2", 0)}, X._peak)
    fluence_share, _ = X._off_share([], classes, {("S2", 0)}, X._fluence)
    assert peak_share < 0.01 < 0.9 < fluence_share
    texts, _ = X._faint_mismatch(rating, [], classes, "M/face")
    assert set(texts) == {"repetition_rate"} and "S2 at 1064 nm (10 Hz)" in texts["repetition_rate"]


# ------------------------------------------------------------------------------------- full runs with catalog products


def _catalog_bench(sku: str, role: str, *, wavelength_nm: float, power_w: float, w0: float, pulse: dict | None, aoi_deg: float,
                   maximum: float = 1.0) -> dict:
    """A laser along +X 100 mm before a catalog product turned so its face normal makes ``aoi_deg`` with the beam (in the
    x-y plane), its waist at the source, and a utilization target on the product."""
    from local_helpers import plane, project, source
    from catalog_store_helpers import catalog_version_of

    laser = source(position=(-100.0, 0.0, 0.0), wavelength_nm=wavelength_nm, waist_mm=w0, power_w=power_w)
    if pulse is not None:
        laser["source_model"]["pulse"] = pulse
    half = math.radians(180.0 - aoi_deg) / 2.0  # the face normal (local +Z) toward -X, turned by aoi about design z
    c, s = math.cos(half), math.sin(half)
    # local +Z -> design +X is (0.5, 0.5, 0.5, 0.5); then a turn by 180 - aoi about design z
    q = (c * 0.5 - s * 0.5, c * 0.5 - s * 0.5, c * 0.5 + s * 0.5, c * 0.5 + s * 0.5)
    part = {"instance_id": "P", "optical_role": role, "product": {"optical_sku": sku, "catalog_version": catalog_version_of(REAL_CATALOG)},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(q)}}
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "P", "maximum": maximum}]
    return project([laser, part], [plane("far", (0.0, -80.0, 0.0), radius_mm=40.0)], targets)


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_the_800_nm_sampler_reports_the_exposure_of_a_2_mm_beam_and_never_passes(real_catalog_store, lossless):
    """E4 as a run: 100 fs pulses at 1 kHz, a 2 mm beam at 45 deg on the sampler rated 0.246 J/cm2 with a 0.166 mm test
    beam. At 10 W the fluence is about 2.6 times the rated value, and the threshold for this beam would be at most the rated
    value; but the rating states neither how many pulses each tested site received, nor whether its value is a peak, nor
    its test polarization, so it establishes no threshold here: not verified, no exceedance, the target UNKNOWN with the
    exposure upper-bounded. The catalog glass has no bulk-loss evidence, so its unknown power must not become a measured
    value. At 1 W: UNKNOWN as well, never a PASS."""
    from local_helpers import with_lossless_bulk

    catalog = with_lossless_bulk(_catalog()) if lossless else REAL_CATALOG
    pulse = {"repetition_rate_hz": 1000.0, "pulse_duration_s": 1e-13, "duration_definition": "FWHM"}
    document = _catalog_bench(SAMPLER_800, "splitter", wavelength_nm=800.0, power_w=10.0, w0=1.0, pulse=pulse, aoi_deg=45.0)
    request = lower_project(document, catalog=catalog).request
    (part,) = [c for c in request.project.components if c.instance_id == "P"]
    assert {r.surface_kind for r in part.power_ratings} == {"OTHER"}
    result = run_optical_project(document, catalog=catalog).result
    fs = [c for s in result.exposures.surfaces for c in s.ratings if c.rating_wavelength_nm == 800.0]
    assert {c.applicability for c in fs} == {"NOT_VERIFIED"} and {c.status for c in fs} == {"CONDITIONS_NOT_VERIFIED"}
    beams = {_state(next(k for k in c.conditions if k.condition == "beam_diameter")) for c in fs}
    unverified = {(k.condition, k.reason) for c in fs for k in c.conditions if k.state == "NOT_VERIFIABLE"}
    missing = {(name, "RATING_INFORMATION_MISSING") for name in ("pulse_count", "value_definition", "polarization")}
    front = next(s for s in result.exposures.surfaces if s.surface_id == "P/S1")
    assert front.hits[0].peak_fluence_J_per_cm2 / fs[0].rating_value > 2.0
    assert all(c.rated_threshold_low is None for c in fs)
    if lossless:
        assert beams == {("DIFFERS", "THRESHOLD_AT_MOST_RATED")} and unverified == missing
        exposures = [c.exposure_value if c.exposure_value is not None else c.exposure_lower_bound for c in fs]
        assert max(e for e in exposures if e is not None) / fs[0].rating_value > 2.0
        reported = [c for c in fs if c.exposure_value is not None]
        assert reported
        for c in reported:
            entry = c.value_basis["utilization"]
            assert c.utilization == c.exposure_value / c.rating_value and entry.kind.value == "ESTIMATE"
            assert {name for a in entry.assumed for name in a.at} == {"pulse_count", "polarization", "value_definition"}
            assert [(v.parameter, v.within) for v in entry.validity] == [("beam_diameter", False)]
    else:
        assert beams == {("DIFFERS", "THRESHOLD_AT_MOST_RATED"), ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN")}
        assert unverified == missing | {("beam_diameter", "BEAM_SIZE_UNKNOWN")}
        entrance = next(c for c in front.ratings if c.rating_wavelength_nm == 800.0)
        (exit_face,) = [c for c in fs if c is not entrance]
        assert entrance.exposure_value == front.hits[0].peak_fluence_J_per_cm2
        assert entrance.utilization == entrance.exposure_value / entrance.rating_value
        entry = entrance.value_basis["utilization"]
        assert entry.kind.value == "ESTIMATE"
        assert "BULK_LOSS_NOT_ESTABLISHED" in {part.code for part in entry.omitted}
        assert exit_face.exposure_value is None and exit_face.utilization is None
        assert all(c.exposure_lower_bound is None and c.exposure_upper_bound is not None for c in fs)
        assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in result.issues)
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
    assert target.value > 2.0 and target.value_basis["value"].kind.value == "ESTIMATE"
    result = run_optical_project(_catalog_bench(SAMPLER_800, "splitter", wavelength_nm=800.0, power_w=1.0, w0=1.0, pulse=pulse,
                                                aoi_deg=45.0), catalog=catalog).result
    (target,) = result.targets
    assert target.status.value == "UNKNOWN"


def test_faint_ghosts_of_another_size_do_not_decide_the_beam_size(real_catalog_store):
    """The 532 nm polarizing cube's CW rating (750 W/cm, a 36.8 um test beam) with a 30 um beam focused on its entrance
    face: the cube's AR faces return a little of the light onto that face, diverged to a much larger size. Carrying, with
    the light left out as negligible, less than 1 % of the most light the face can receive, they do not change the size of
    the light the rating is compared with: the threshold for it is at least the rated value, and the exposure (P / d from
    the 1/e^2 to the FWHM reading) is within the value -- but the catalog rating states no test polarization, so it is not
    verified there. Deeper in the cube the beam has grown past the test beam: the W/cm rating does not apply there. The
    element's target is not decided. Those exact ghost checks use a declared copy with an explicit ideal-loss assumption;
    the real catalog cube has no bulk-loss evidence, so its exposure remains uncertain."""
    from local_helpers import plane, project, source
    from catalog_store_helpers import catalog_version_of

    laser = source(position=(-62.7, 0.0, 0.0), wavelength_nm=532.0, waist_mm=0.015, power_w=1.0, waist_position_mm=50.0)
    cube = {"instance_id": "P", "optical_role": "splitter", "product": {"optical_sku": PBS_532, "catalog_version": catalog_version_of(REAL_CATALOG)},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "P", "maximum": 1}]
    document = project([laser, cube], [plane("far", (60.0, 0.0, 0.0), radius_mm=20.0)], targets)
    published = run_optical_project(document).result
    published_entrance = next(s for s in published.exposures.surfaces if s.surface_id == "P/S1")
    assert published_entrance.untraced_hits == ()  # unknown bulk loss is not a measured zero or a known ghost flux
    assert published_entrance.ratings[0].exposure_lower_bound is None
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in published.issues)
    assert published.targets[0].status.value == "UNKNOWN"

    # Isolate the original beam-size/ghost comparison in a declared copy of the
    # same prescription. Only this synthetic test model assumes ideal bulk loss.
    from declared_helpers import declared_project

    ideal = declared_project(document, REAL_CATALOG)
    ideal["optical_models"][0]["power_ratings"] = [r.model_dump(mode="json", exclude_none=True)
                                                   for r in _product(PBS_532).specifications.power_ratings]
    for material in ideal["model_materials"]:
        if material["material_id"] == "FUSED_SILICA":
            material["absorption_coefficient_per_mm"] = "0"
            material["basis"] += "; ideal zero bulk loss assumed for this test model"
    result = run_optical_project(ideal).result
    entrance = next(s for s in result.exposures.surfaces if s.surface_id == "P/S1")
    assert entrance.untraced_hits and max(2.0 * h.beam_radius_x_mm for h in entrance.untraced_hits) > 0.0625  # (larger ghosts)
    cw = next(c for c in entrance.ratings if c.regime == "CW")
    beam = next(c for c in cw.conditions if c.condition == "beam_diameter")
    assert (beam.state, beam.effect) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED") and beam.design_high == pytest.approx(0.030, rel=1e-6)
    (hit,) = entrance.hits
    e2 = hit.incident_power_W / (2.0 * hit.beam_radius_x_mm) * 10.0
    assert cw.exposure_lower_bound <= e2 and cw.exposure_upper_bound >= e2 * 2.0 / math.sqrt(2.0 * math.log(2.0))
    assert _state(next(c for c in cw.conditions if c.condition == "polarization")) == PROTOCOL
    assert (cw.applicability, cw.status) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED")
    assert cw.exposure_upper_bound <= cw.rating_value  # (within the value, were the rating verified)
    deeper = [c for s in result.exposures.surfaces if s.surface_id != "P/S1" for c in s.ratings if c.regime == "CW" and s.hits]
    assert deeper and {c.applicability for c in deeper} == {"NOT_APPLICABLE"}
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and "polarization: information the rating does not state" in target.message


# ---------------------------------------------- a beam left out only where it is minor in every quantity that sets the direction


def _focused_and_broad(rating: PowerRating, *, broad_share: float, pulse: PulseTrain | None = None, value: float = 1000.0):
    """A 0.05 mm beam (1/e^2) at 0.9 of ``value`` and a 10 mm beam at ``broad_share`` of it, in the rating's own quantity
    (peak irradiance, or peak fluence for a pulse train), each from its own source, on a dielectric fixture mirror."""
    w_a, w_b = 0.025, 5.0

    def power(level: float, w_mm: float) -> float:  # the power (or, times the rate, the pulse energy) of that peak value
        w_cm = w_mm / 10.0
        return level * math.pi * w_cm * w_cm / 2.0 * (1.0 if pulse is None else pulse.repetition_rate_hz)

    beams = [("A", power(0.9 * value, w_a), w_a, pulse)]
    if broad_share:
        beams.append(("B", power(broad_share * value, w_b), w_b, pulse))
    return _two_beams_on_a_mirror(rating, tuple(beams))


def test_a_broad_beam_is_weighed_in_power_and_p_per_d_under_a_cw_irradiance_rating():
    """A CW 1000 W/cm2 fixture rating with a 0.1 mm 1/e^2 test beam. A beam focused to 0.05 mm at 0.9 of the rating: at least
    rated, within, the target PASSes with the ratio 0.9 as its value. Beside it a 10 mm beam at 0.89 % of the rating
    (3.5 W): under 1 % of the peak irradiance, but nearly all of the power and of the P / d (a beam's heating scales with
    P / d, a broad beam heats the element with its power) -- it takes part in the size comparison, the beams lie on both
    sides of the test beam, and the target is not decided. A broad beam of 1e-5 of the power stays left out."""
    rating = _fixture(1000.0, beam_diameter_mm=0.1, beam_diameter_definition="1/E2", surface_kind="OTHER")
    face, result = _focused_and_broad(rating, broad_share=0.0)
    (comparison,) = face.ratings
    assert (comparison.applicability, comparison.status, result.targets[0].status.value) == (
        "THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING", "PASS")
    assert result.targets[0].value == pytest.approx(comparison.exposure_value / 1000.0, rel=1e-12)
    assert result.targets[0].value == pytest.approx(0.9, rel=1e-6)
    face, result = _focused_and_broad(rating, broad_share=0.0089)
    hits = {h.source_id: h for h in face.hits}
    peak = hits["A"].peak_irradiance_W_per_cm2 + hits["B"].peak_irradiance_W_per_cm2
    assert hits["B"].peak_irradiance_W_per_cm2 < 0.01 * peak
    assert hits["B"].incident_power_W > 0.99 * (hits["A"].incident_power_W + hits["B"].incident_power_W)
    assert hits["B"].linear_power_density_W_per_cm > 0.5 * (
        hits["A"].linear_power_density_W_per_cm + hits["B"].linear_power_density_W_per_cm)
    (comparison,) = face.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(beam) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and "smaller" in beam.message and "larger" in beam.message
    assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
    # (no rating applies: the ratio to the rated value is reported with the condition that differs, and decides nothing)
    undecided = result.targets[0]
    assert (undecided.status.value, undecided.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert undecided.value == comparison.utilization and undecided.value_basis["value"].kind.value == "ESTIMATE"
    assert [(v.parameter, v.within, v.code) for v in undecided.value_basis["value"].validity] == [
        ("beam_diameter", False, "POWER_RATING_CONDITIONS_MISMATCH")]
    face, result = _focused_and_broad(rating, broad_share=0.9e-5 * (0.025 / 5.0) ** 2)  # (1e-5 of A's power)
    hits = {h.source_id: h for h in face.hits}
    assert hits["B"].incident_power_W < 2e-5 * hits["A"].incident_power_W
    (comparison,) = face.ratings
    assert (comparison.applicability, comparison.status, result.targets[0].status.value) == (
        "THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING", "PASS")


def test_a_broad_beam_is_weighed_in_pulse_energy_under_a_fluence_rating():
    """A pulsed 1 J/cm2 fixture rating (10 ns, 10 Hz, a 0.1 mm 1/e^2 test beam, a dielectric surface, a single pulse per
    tested site; each source a single shot): a 0.05 mm spot at 0.9 J/cm2 and, at the same rate, a 10 mm beam at 0.89 % of
    the rating's fluence -- under 1 % of the peak fluence, nearly all of the pulse energy (a broad beam of low fluence meets
    more of the surface's defects). It takes part in the size comparison: not applicable, the target not decided, never a
    PASS from the micro-spot's threshold alone."""
    train = PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=10.0, total_pulses=1)
    rating = _fixture(1.0, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0, beam_diameter_mm=0.1,
                      beam_diameter_definition="1/E2", surface_kind="OTHER", pulse_count=1)
    face, result = _focused_and_broad(rating, broad_share=0.0, pulse=train, value=1.0)
    assert (face.ratings[0].applicability, result.targets[0].status.value) == ("THRESHOLD_AT_LEAST_RATED", "PASS")
    face, result = _focused_and_broad(rating, broad_share=0.0089, pulse=train, value=1.0)
    hits = {h.source_id: h for h in face.hits}
    assert hits["B"].peak_fluence_J_per_cm2 < 0.01 * (hits["A"].peak_fluence_J_per_cm2 + hits["B"].peak_fluence_J_per_cm2)
    assert hits["B"].pulse_energy_J > 0.99 * (hits["A"].pulse_energy_J + hits["B"].pulse_energy_J)
    (comparison,) = face.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(beam) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert (comparison.applicability, result.targets[0].status.value) == ("NOT_APPLICABLE", "UNKNOWN")


def test_the_quantities_a_beam_is_weighed_in_for_the_size_comparison():
    """The rating's own quantity first (the order beams are taken in), then for a CW rating the power and P / d, for a
    pulsed one the peak fluence and the pulse energy; both P / d readings for a W/cm rating without a definition."""
    from os_optics.validation import exposures as X

    def kinds(rating: PowerRating) -> list[tuple[str, str | None]]:
        return [(q.kind, q.definition) for q in X._size_quantities(rating)]

    assert kinds(_fixture()) == [("peak", None), ("power", None), ("linear", None)]
    assert kinds(_fixture(unit="W/cm", beam_diameter_definition="1/E2")) == [("linear", None), ("power", None)]
    assert kinds(_fixture(unit="W/cm")) == [("linear", None), ("linear", "FWHM"), ("power", None)]
    assert kinds(_fixture(regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8)) == [("fluence", None), ("energy", None)]
    # (every P / d the size comparison weighs beams in counts a beam beside the surface with its tail bound)
    for rating in (_fixture(), _fixture(unit="W/cm", beam_diameter_definition="1/E2"), _fixture(unit="W/cm")):
        assert all(q.weighed == (q.kind == "linear") for q in X._size_quantities(rating))


def test_a_beam_beside_the_surface_is_weighed_in_p_per_d_by_its_tail_bound():
    """A beam whose chief ray misses the surface: the size comparison weighs it in ``P / d`` with the share of its power its
    tail bound gives, over its own narrower diameter (1/e^2 or FWHM); an exposure's ``P / d`` has no value for it."""
    from dataclasses import replace

    from os_optics.contracts.results import ExposureHit
    from os_optics.validation import exposures as X

    hit = ExposureHit(path_id="B#0", source_id="B", spectral_index=0, wavelength_nm=1064.0, pass_index=0, incident_power_W=2.0,
                      aoi_deg=0.0, beam_radius_x_mm=1.0, beam_radius_y_mm=0.5, peak_irradiance_W_per_cm2=1e-3,
                      peak_irradiance_surface_W_per_cm2=1e-3, reason="INCOMPLETE_ENVELOPE", miss_distance_mm=1.5)
    beside = X._HitValues(hit, False, None, None, light=X.BESIDE, tail=1e-3)
    assert X._linear("1/E2")(beside) is None and X._linear("FWHM")(beside) is None
    assert replace(X._linear("1/E2"), weighed=True)(beside) == 2.0 * 1e-3 / (2.0 * 0.5) * 10.0
    assert replace(X._linear("FWHM"), weighed=True)(beside) == 2.0 * 1e-3 / (X.FWHM_PER_RADIUS * 0.5) * 10.0
    assert X._power(beside) == 2.0 * 1e-3
    unknown = X._HitValues(hit.model_copy(update={"peak_irradiance_W_per_cm2": None, "beam_radius_x_mm": None,
                                                  "beam_radius_y_mm": None}), False, None, None, light=X.BESIDE)
    assert replace(X._linear("1/E2"), weighed=True)(unknown) is None  # (no valid Gaussian state: no value, never left out)


def _spot_and_a_beam_beside(offset_mm: float):
    """A 0.05 mm spot (1/e^2) at 0.9 of a CW 1000 W/cm2 fixture rating with a 0.1 mm 1/e^2 test beam on a dielectric
    fixture mirror of 12.7 mm radius at normal incidence, and a 2 mm beam of 100 times its power whose chief ray passes
    ``offset_mm`` from the mirror's axis (beside the mirror beyond 12.7 mm)."""
    rating = _fixture(1000.0, beam_diameter_mm=0.1, beam_diameter_definition="1/E2", surface_kind="OTHER")
    p_a = 0.9 * 1000.0 * math.pi * 0.0025 ** 2 / 2.0
    lasers = [B.laser(name, wavelength_nm=1064.0, power_W=power, waist_radius_mm=w, position=(-50.0, y, 0.0),
                      direction=(1.0, 0.0, 0.0), waist_position_z_mm=50.0)
              for name, power, w, y in (("A", p_a, 0.025, 0.0), ("B", 100.0 * p_a, 1.0, offset_mm))]
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=12.7).model_copy(
        update={"power_ratings": (rating,)})
    result = run(ProjectSnapshot(components=(*lasers, mirror),
                                 targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),)))
    face = next(x for x in result.exposures.surfaces if x.surface_id == "M/face")
    (comparison,) = face.ratings
    return face, comparison, {c.condition: c for c in comparison.conditions}, result.targets[0]


@pytest.mark.parametrize(("offset_mm", "miss_mm"), [(15.0, 2.3), (17.0, 4.3)])
def test_a_beam_beside_the_surface_whose_tail_is_minor_does_not_decide_the_size(offset_mm, miss_mm):
    """Its tail on the mirror carries at most 0.25 % of the power and far less of the P / d (``exp(-2 d^2 / w^2)`` of 100
    times the spot's power over its 2 mm diameter) and of the peak irradiance: it is left out, the spot alone gives a
    threshold at least the rated value, and the target PASSes with the spot's ratio 0.9."""
    face, comparison, conditions, target = _spot_and_a_beam_beside(offset_mm)
    (beside,) = face.beside_hits
    assert beside.miss_distance_mm == pytest.approx(miss_mm, abs=1e-9)
    assert 100.0 * math.exp(-2.0 * miss_mm ** 2) < 0.0026  # (its tail's power against the spot's)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED")
    assert (comparison.applicability, comparison.status) == ("THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING")
    assert (target.status.value, target.value) == ("PASS", pytest.approx(0.9, rel=1e-6))


def test_a_beam_beside_the_surface_whose_tail_is_not_minor_takes_part_in_the_size_comparison():
    """0.8 mm beside the mirror its tail carries 28 % of its 100-fold power (more than the spot): the beams lie on both
    sides of the test beam, no direction holds, and the target is not decided."""
    face, comparison, conditions, target = _spot_and_a_beam_beside(13.5)
    assert face.beside_hits[0].miss_distance_mm == pytest.approx(0.8, abs=1e-9)
    assert _state(conditions["beam_diameter"]) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert target.value == comparison.utilization and target.value_basis["value"].kind.value == "ESTIMATE"


def test_a_verified_beam_beside_beams_in_one_direction():
    """Beams that verify a stated diameter beside beams in one direction give that direction; the message names the
    direction's beams only (a verified beam's own footprint, unbounded when it grazes the surface, is not counted)."""
    rating = _fixture(beam_diameter_mm=1.0, beam_diameter_definition="1/E2")
    grazing, small, large = BeamSize((1.0, 1.0), None, "g"), BeamSize((0.5, 0.5), 0.5, "s"), BeamSize((2.0, 2.0), 2.0, "l")
    record = beam_diameter_condition(rating, [grazing, small])
    assert _state(record) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED") and record.design_high is None
    assert "differ from the test beam are at most 0.5 mm across" in record.message
    assert "the others match it within 5 %" in record.message
    tilted = BeamSize((1.0, 1.0), 1.414, "t")  # (a verified beam at 45 deg: its footprint is longer than the test beam)
    record = beam_diameter_condition(rating, [tilted, large])
    assert _state(record) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED") and "differ from the test beam are at least 2 mm" in record.message
    record = beam_diameter_condition(rating, [small])
    assert record.message.endswith("the beams here are at most 0.5 mm across on the surface (1/e^2 footprint), no larger than "
                                   "the test beam under any reading: the damage threshold for them is at least the rated value")


# ------------------------------------------------------------------------------------------ a surface only left-out light reaches


def _beside_a_window(rating: PowerRating, pulse: PulseTrain | None):
    """A review scene: a rated mirror beside a beam crossing an uncoated window -- only the window's ghosts left out as
    negligible may reach it -- with the source's pulse train ``pulse``."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    if pulse is not None:
        src = src.model_copy(update={"source": src.source.model_copy(update={"pulse": pulse})})
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    mirror = B.mirror("M", position=(0.0, 200.0, 0.0), normal=(0.0, -1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    result = run(ProjectSnapshot(components=(src, window, mirror),
                                 targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    assert face.hits == () and face.untraced_hits == () and face.faint_light_peak_W_per_cm2
    (comparison,) = face.ratings
    return {c.condition: c for c in comparison.conditions}, comparison, result.targets[0]


def test_the_left_out_light_on_a_surface_no_followed_light_reaches_keeps_its_pulse_train():
    """The left-out light of a 10 ns, 10 Hz line on a mirror rated for that line: its regime, wavelength, pulse duration and
    rate verify the rating (never "the light here is continuous"), and its total too: a single shot against a
    single-pulse rating brings at least one pulse per site (the left-out light's crossings are not counted, so no upper
    end), a threshold at most the rated value. The rating's test beam and polarization cannot be checked against light
    whose size and state are not known (it is bounded by its diffraction limit only): not verified, not decided. A
    rating that states no count is not verifiable: no relation to the pulses here. A rating at another rate leaves the
    rate not verifiable; a pulsed rating with only continuous left-out light leaves the regime and both pulse conditions
    not verifiable, never differing."""
    train = PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=10.0)

    def rated(rate: float, **extra) -> PowerRating:
        return _fixture(1e6, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=rate,
                        beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", **extra)

    single = train.model_copy(update={"total_pulses": 1})
    conditions, comparison, target = _beside_a_window(rated(10.0, pulse_count=1), single)
    for name in ("regime", "wavelength", "pulse_duration", "repetition_rate"):
        assert _state(conditions[name]) == ("VERIFIED", None), conditions[name]
        assert "leaves out as negligible" in conditions[name].message
    assert conditions["pulse_duration"].design_low == 1e-8 and conditions["repetition_rate"].design_low == 10.0
    count = conditions["pulse_count"]
    assert (_state(count), count.design_low, count.design_high) == (AT_MOST_RATED, 1.0, None)
    assert (_state(conditions["beam_diameter"]), _state(conditions["polarization"])) == (
        ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN"), ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN"))
    assert (comparison.applicability, comparison.status, target.status.value) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED", "UNKNOWN")
    assert target.value is None  # (the exposure lies between 0 and the left-out light's bound: no number, never 0)
    conditions, comparison, target = _beside_a_window(rated(10.0), train)
    assert _state(conditions["pulse_count"]) == PROTOCOL
    assert (comparison.applicability, comparison.status, target.status.value) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED", "UNKNOWN")
    conditions, comparison, target = _beside_a_window(rated(100.0), train)
    assert _state(conditions["repetition_rate"]) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN")
    assert _state(conditions["pulse_duration"]) == ("VERIFIED", None)
    assert (comparison.applicability, target.status.value) == ("NOT_VERIFIED", "UNKNOWN")
    conditions, comparison, target = _beside_a_window(rated(10.0), None)
    for name in ("regime", "pulse_duration", "repetition_rate"):
        assert _state(conditions[name]) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN"), conditions[name]
    assert not any(c.state == "DIFFERS" for c in conditions.values())
    assert (comparison.applicability, comparison.status) == ("NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED")
    conditions, comparison, target = _beside_a_window(_fixture(1e6), train)  # (a CW rating, pulsed left-out light)
    assert _state(conditions["regime"]) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN")
    assert (conditions["pulse_duration"].state, conditions["repetition_rate"].state) == ("NOT_STATED", "NOT_STATED")


# ------------------------------------------------------- a surface only left-out light reaches, where light may be stored


def _beside_a_cavity(rating: PowerRating, lines: tuple[tuple[str, float, float, PulseTrain | None], ...]):
    """The window scene of ``_beside_a_window`` with the beams (name, vacuum wavelength nm, power W, pulse train) crossing
    the window between two lossless mirrors facing each other: their light reaches the interaction limit (it may be stored
    there, ``MAX_INTERACTIONS``) and only the left-out light may reach the rated mirror, in a share nothing bounds."""
    sources = []
    for k, (name, wavelength_nm, power, pulse) in enumerate(lines):
        laser = B.laser(name, wavelength_nm=wavelength_nm, power_W=power, waist_radius_mm=0.5, position=(-75.0, float(k), 0.0),
                        direction=(1.0, 0.0, 0.0))
        if pulse is not None:
            laser = laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})
        sources.append(laser)
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(-40.0, 0.0, 0.0))
    cavity = (B.mirror("C1", position=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), reflectance=1.0),
              B.mirror("C2", position=(-100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), reflectance=1.0))
    mirror = B.mirror("M", position=(-40.0, 200.0, 0.0), normal=(0.0, -1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    result = run(ProjectSnapshot(components=(*sources, window, *cavity, mirror),
                                 targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),)))
    assert not result.exposures.trace_complete
    face = next(x for x in result.exposures.surfaces if x.surface_id == "M/face")
    assert face.hits == () and face.untraced_hits == () and face.beside_hits == ()
    (comparison,) = face.ratings
    return {c.condition: c for c in comparison.conditions}, comparison, result.targets[0]


def test_where_light_may_be_stored_a_left_out_line_the_rating_is_not_for_leaves_its_conditions_not_verifiable():
    """In a run whose light may be stored between two lossless mirrors, a mirror only left-out light reaches: a line the
    rating is not for leaves each condition it differs in not verifiable whatever its power, since nothing bounds its share
    there -- 1 W of 1064 nm beside 1 mW of 532 nm under a 532 nm rating (wavelength), 1 W of CW light beside 1 mW of 10 Hz
    pulses under a pulsed rating (regime), 1 W at 100 Hz beside 1 mW at 10 Hz under a 10 Hz rating (repetition rate).
    Never matched, never differing; the matching lines still verify the other conditions. Lines that all meet the rating
    verify those conditions, but the rating's test beam and polarization cannot be checked against left-out light, whose
    size and state are not known: not verified, and not decided (no upper bound)."""
    t10 = PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=10.0)
    t100 = PulseTrain(pulse_duration_s=1e-8, repetition_rate_hz=100.0)
    pulsed = _fixture(1e6, regime="PULSED", unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0)
    for rating, lines, kind in (
            (_fixture(1e6, wavelength_nm=532.0), (("S1064", 1064.0, 1.0, None), ("S532", 532.0, 0.001, None)), "wavelength"),
            (pulsed, (("SCW", 1064.0, 1.0, None), ("SP", 1064.0, 0.001, t10)), "regime"),
            (pulsed, (("S100", 1064.0, 1.0, t100), ("S10", 1064.0, 0.001, t10)), "repetition_rate")):
        conditions, comparison, target = _beside_a_cavity(rating, lines)
        assert _state(conditions[kind]) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN"), conditions[kind]
        assert "nothing bounds its share" in conditions[kind].message and lines[0][0] in conditions[kind].message
        assert not any(c.state == "DIFFERS" for c in conditions.values())
        assert all(c.state == "VERIFIED" for name, c in conditions.items()
                   if name in ("regime", "wavelength", "pulse_duration", "repetition_rate") and name != kind and c.state != "NOT_STATED")
        assert (comparison.applicability, comparison.conditions_match, comparison.status) == (
            "NOT_VERIFIED", False, "CONDITIONS_NOT_VERIFIED")
        assert (target.status.value, target.value) == ("UNKNOWN", None)
    conditions, comparison, target = _beside_a_cavity(_fixture(1e6, wavelength_nm=532.0, beam_diameter_mm=TEST_BEAM_MM,
                                                               beam_diameter_definition="1/E2"), (("S532", 532.0, 0.001, None),))
    assert (_state(conditions["regime"]), _state(conditions["wavelength"])) == (("VERIFIED", None), ("VERIFIED", None))
    assert (_state(conditions["beam_diameter"]), _state(conditions["polarization"])) == (
        ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN"), ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN"))
    assert (comparison.applicability, comparison.status, target.status.value, target.value) == (
        "NOT_VERIFIED", "CONDITIONS_NOT_VERIFIED", "UNKNOWN", None)


def test_a_stored_line_the_rating_is_not_for_leaves_the_conditions_of_a_surface_only_left_out_light_reaches_not_verifiable():
    """The lines whose light may be stored count with the left-out light's lines on such a surface (``_compare``'s
    ``stored``): a stored 1064 nm line leaves the wavelength of a 532 nm rating not verifiable although the only left-out
    line is at 532 nm; a stored 532 nm line changes nothing."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import exposures as X

    rating = _fixture(1e6, wavelength_nm=532.0)
    left_out = (Light("SG", 532.0, None, left_out=True),)
    for stored, state in (((Light("SC", 1064.0, None, left_out=True),), ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN")),
                          ((Light("SC", 532.0, None, left_out=True),), ("VERIFIED", None)), ((), ("VERIFIED", None))):
        comparison, _ = X._compare(0, rating, [], [], True, "M/face", "light of unknown fate", NumericalConfig(), left_out, stored)
        wavelength = next(c for c in comparison.conditions if c.condition == "wavelength")
        assert _state(wavelength) == state, wavelength



# ------------------------------------------------------- followed light of another line: the share rule (review M2)


def _two_lines(reflectance: float | None, beside_mm: float | None = None) -> ProjectSnapshot:
    """A 1 W 1064 nm beam on an uncoated window rated at 1064 nm (fixture) and, from a second laser, 1 W of 532 nm: reflected
    onto the window's axis by a splitter of ``reflectance`` (the review's ``p2_other_line``), or running parallel to it
    ``beside_mm`` off its axis, outside the window's clear aperture, its Gaussian tail reaching the window
    (``p3_beside_line``)."""
    inv = 1.0 / math.sqrt(2.0)
    comps = [B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))]
    if reflectance is not None:
        comps += [B.laser("B", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.5, position=(-50.0, -100.0, 0.0), direction=(0.0, 1.0, 0.0)),
                  B.npbs("S", reflectance=reflectance, position=(-50.0, 0.0, 0.0), normal=(-inv, inv, 0.0)),
                  B.beam_dump("D", position=(-50.0, 60.0, 0.0), axis=(0.0, 1.0, 0.0))]
    if beside_mm is not None:
        comps += [B.laser("B", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, beside_mm, 0.0), direction=(1.0, 0.0, 0.0)),
                  B.beam_dump("D", position=(100.0, beside_mm, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=3.0)]
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    # (the rating's test beam: the 1064 nm beam's 1/e^2 diameter 100 mm from its waist, at the window; linear, as the laser)
    rated = PowerRating(regime="CW", value=25000.0, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                        basis=FIXTURE, beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", polarization="LINEAR",
                        polarization_condition="linear")
    window = window.model_copy(update={"power_ratings": (rated,)})
    return ProjectSnapshot(components=(*comps, window),
                           targets=(Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),))


@pytest.mark.parametrize(("reflectance", "beside_mm", "applies"), [(None, None, True), (1e-12, None, True), (1e-3, None, True),
                                                                   (0.1, None, False), (None, 16.0, True), (None, 17.0, True)],
                         ids=["alone", "R1e-12", "R1e-3", "R0.1", "beside16", "beside17"])
def test_followed_light_of_another_line_below_the_share_counts_in_the_exposure_not_in_the_conditions(reflectance, beside_mm,
                                                                                                    applies: bool):
    """Review M2: 532 nm light that the 1064 nm rating is not for made the wavelength DIFFER whatever its power (1e-12 W
    reflected onto the axis, the 1e-16 W/cm2 tail of a beam beside the window): the rating did not apply and the target was
    UNKNOWN. Now the rule for light left out as negligible and minor beams: at most 1 % of the most light the surface can
    receive, in peak irradiance and every quantity that gives a comparison its direction, it counts in the exposure (the
    upper bound) but not in the conditions -- the wavelength is verified, a note names that light and its share, the target
    PASSes. 10 % of the light (R 0.1) is not negligible: the wavelength differs (no established relationship), as before."""
    result = run(_two_lines(reflectance, beside_mm))
    for surface in (s for s in result.exposures.surfaces if s.element_id == "W"):
        (comparison,) = surface.ratings
        wavelength = next(c for c in comparison.conditions if c.condition == "wavelength")
        if applies:
            assert (comparison.applicability, comparison.status, wavelength.state) == ("AS_RATED", "WITHIN_RATING", "VERIFIED")
            notes = [n for n in comparison.notes if "which the rating is not for" in n]
            assert bool(notes) == (reflectance is not None or beside_mm is not None)
            if notes:
                assert "light of B at 532 nm" in notes[0] and "it counts in the exposure, not in the rating's conditions" in notes[0]
        else:
            assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
            assert (wavelength.state, wavelength.effect) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    (target,) = result.targets
    assert target.status.value == ("PASS" if applies else "UNKNOWN")
    if reflectance == 1e-3:
        # (the 532 nm light still counts in the exposure: lines add, so the upper bound holds both beams' peaks)
        for surface in (s for s in result.exposures.surfaces if s.element_id == "W"):
            peaks = {h.wavelength_nm: h.peak_irradiance_W_per_cm2 for h in surface.hits}
            assert set(peaks) == {532.0, 1064.0}
            assert surface.ratings[0].exposure_upper_bound >= peaks[532.0] + peaks[1064.0]


def _share_contribution(i: int, peak: float | None, source: str = "S", wavelength: float = 1064.0):
    """A contribution of ``peak`` W/cm2 (``None``: its size is not known, no value) for the share helpers."""
    from os_optics.contracts.results import ExposureHit
    from os_optics.validation.exposures import _HitValues

    hit = ExposureHit(path_id=f"{source}#{i}", source_id=source, spectral_index=0, wavelength_nm=wavelength, incident_power_W=0.1,
                      aoi_deg=0.0, peak_irradiance_W_per_cm2=peak, beam_radius_x_mm=None if peak is None else 0.5,
                      beam_radius_y_mm=None if peak is None else 0.5)
    return _HitValues(hit, False, None, None if peak is None else 0.5)


def _left_out(power_W: float, key=("S", 0), wavelength: float = 1064.0):
    from os_optics.validation.exposures import _Faint

    root = math.sqrt(power_W)
    return _Faint(power_W, None, 1.5, wavelength, ((key, wavelength, power_W, None),), 1.0, 1.0, (root,), 2.0)


def test_a_contribution_without_a_value_never_hides_the_light_outside_a_state_or_a_line():
    """The shares of the light outside a rating's polarization state (``_faint_share``) and of left-out light of a line the
    rating is not for (``_off_share``) with a contribution whose size is not known (no peak irradiance). Outside a state:
    in a class without left-out light it is left out (more of a class within the tolerance only draws the share toward
    its own); in a class with left-out light the share is unbounded -- its part outside the state may add in phase to that
    light's. Off the line: it is left out of the most light, so the share is the larger one, never skipped."""
    from os_optics.validation.exposures import _Class, _faint_share, _off_share, _peak

    beams = [_share_contribution(0, 1000.0), _share_contribution(1, None), _share_contribution(2, 800.0, source="T")]
    shares = {0: 0.005, 2: 0.004}
    alone = _Class((0, 1), {(0, 1): None}, False, _left_out(1e-9))
    other = _Class((2,), {}, False)
    known_only = [beams[0], beams[2]]
    unbounded = _faint_share(beams, [alone, other], shares, _peak)
    assert unbounded[0] == 1.0 and math.isnan(unbounded[1])
    without_left_out = _faint_share(beams, [_Class((0, 1), {(0, 1): None}, False), _Class((2,), {}, False, _left_out(1e-9, ("T", 0)))],
                                    shares, _peak)
    reference = _faint_share(known_only, [_Class((0,), {}, False), _Class((1,), {}, False, _left_out(1e-9, ("T", 0)))],
                             {0: 0.005, 1: 0.004}, _peak)
    assert without_left_out == reference and 0.0 < reference[0] < 1.0
    off = _off_share(beams, [_Class((0, 1), {(0, 1): None}, False, _left_out(1e-6, ("U", 0), 532.0)), other], {("U", 0)}, _peak)
    off_known = _off_share(known_only, [_Class((0,), {}, False, _left_out(1e-6, ("U", 0), 532.0)), _Class((1,), {}, False)],
                           {("U", 0)}, _peak)
    assert off is not None and off == off_known


def test_one_line_rule_decides_what_the_surface_conditions_call_differing():
    """``line_differences`` (one line of light against a rating) and ``scalar_conditions`` (the light on a surface as a
    whole) apply the same windows: for a single line on a grid around each window's edge -- default and caller-set
    windows -- the conditions one calls differing are exactly those the other records as ``DIFFERS``."""
    from os_optics.validation.rating_conditions import (
        DEFAULT_TOLERANCES, DIFFERS, Light, RatingTolerances, line_differences, scalar_conditions)

    rating = PowerRating(regime="PULSED", value=1.0, unit="J/cm2", wavelength_nm=1000.0, status="DECLARED",
                         basis=FIXTURE, pulse_duration_s=1e-8, repetition_rate_hz=100.0)
    windows = (DEFAULT_TOLERANCES, RatingTolerances(0.05, 0.2, 0.3, 0.05, 0.01, 0.01))
    checked = 0
    for tol in windows:
        for f in (-1.0001, -0.9999, 0.0, 0.9999, 1.0001):
            for g in (-1.0001, 0.9999, 1.0001):
                for h in (0.9999, -1.0001):
                    pulse = PulseTrain(pulse_duration_s=1e-8 * (1.0 + g * tol.pulse_duration_rel),
                                       repetition_rate_hz=100.0 * (1.0 + h * tol.repetition_rate_rel))
                    light = Light("src", 1000.0 * (1.0 + f * tol.wavelength_rel), pulse)
                    by_line = set(line_differences(rating, light.wavelength_nm, pulse, tol))
                    by_surface = {c.condition for c in scalar_conditions(rating, [light], tolerances=tol)
                                  if c.state == DIFFERS}
                    assert by_line == by_surface, (tol, f, g, h)
                    checked += 1
    # (another regime: the line rule names the regime alone; the surface rule also records the pulse conditions continuous
    # light cannot meet -- the regime decides both)
    assert line_differences(rating, 1000.0, None) == ("regime",)
    assert "regime" in {c.condition for c in scalar_conditions(rating, [Light("cw", 1000.0, None)]) if c.state == DIFFERS}
    assert checked == 2 * 5 * 3 * 2
