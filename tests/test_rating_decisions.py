"""How a surface's and an element's rating outcome follow from its comparisons' applicability (``validation/exposures.py``,
``validation/rating_conditions.py``).

A comparison that applies as stated (``AS_RATED``) bounds the utilization (exposure / rating) on both sides; one whose
threshold is at least the rated value (``THRESHOLD_AT_LEAST_RATED``: a test beam larger than the beam here) only from
above; one whose threshold is at most the rated value (``THRESHOLD_AT_MOST_RATED``: a smaller test beam) only from below;
one not verified may apply as stated, so it joins the upper bound wherever another rating gives one; one that does not
apply (``NOT_APPLICABLE``) counts in neither. A surface's lower bound is the largest of its lower bounds (a decided
exceedance stands), its upper bound exists only with an ``AS_RATED`` or at-least comparison; where its ratings disagree
(the lower bound above the upper one) the lower bound stands and there is no upper bound. An element takes the largest
of its rated surfaces' bounds and has no upper bound while one of them has none. A ``power_rating_utilization`` target
PASSes only when that interval is within its limit and FAILs when its lower end is not; its message names what makes the
utilization an interval.

The exposures are a single beam on an ideal mirror at 45 degrees (exact: one beam, 1.009 mm across (1/e^2), a 1.427 mm
footprint) and an uncoated window; every rating is test data declared in the test project, never product data, and states
its value as a peak value (``value_definition`` ``PEAK``: its threshold is the value; the reading of a value whose
definition is not stated is tested in ``test_rated_value_reading.py``), linear polarization (the laser's) and, where its
test beam is not what a test is about, the beam's own 1/e^2 diameter as its test beam (verified).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, Target

from helpers import run

FIXTURE = "test fixture rating, not product data"
LASER = B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
#: The laser's 1/e^2 diameter at the mirror 100 mm away (w0 = 0.5 mm, Rayleigh range pi w0^2 / lambda).
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)


def _cw(value: float, **extra) -> PowerRating:
    """A CW W/cm2 fixture rating of a peak value and linear polarization; its test beam the beam's own (``TEST_BEAM_MM``,
    1/e^2) unless a test beam is given."""
    extra.setdefault("value_definition", "PEAK")
    if "polarization" not in extra and "polarization_condition" not in extra:
        extra.update(polarization="LINEAR", polarization_condition="linear")
    if "beam_diameter_mm" not in extra:
        extra.update(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2")
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=1064.0, status="DECLARED", basis=FIXTURE, **extra)


def _mirror(ratings: tuple[PowerRating, ...], maximum: float = 1.0):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=maximum)
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    element = next(e for e in result.exposures.elements if e.element_id == "M")
    return face, element, result.targets[0]


def _peak_on_mirror() -> float:
    face, _, _ = _mirror((_cw(1e30),))
    (comparison,) = face.ratings
    assert comparison.exposure_value is not None  # one beam on an ideal reflector: exact
    (hit,) = face.hits
    assert 2.0 * hit.beam_radius_x_mm == pytest.approx(1.009, abs=5e-4) and hit.beam_radius_x_mm == hit.beam_radius_y_mm
    return comparison.exposure_value


def _condition(comparison, name: str):
    return next(c for c in comparison.conditions if c.condition == name)


# ------------------------------------------------------------------------------------------- a verified test beam


def test_a_verified_test_beam_makes_the_stricter_rating_decide_the_exceedance():
    """Rating A (10 I, the beam's own test beam) and rating B (0.5 I, a 1 mm 1/e^2 test beam; the beam here is 1.009 mm, within
    5 %): B applies as rated and is exceeded at utilization 2, so the surface EXCEEDS and a target of 1 FAILs at 2.0."""
    peak = _peak_on_mirror()
    lenient, strict = _cw(10.0 * peak), _cw(0.5 * peak, beam_diameter_mm=1.0, beam_diameter_definition="1/E2")
    face, element, target = _mirror((lenient, strict))
    a, b = face.ratings
    assert (a.status, a.applicability, a.conditions_match, a.utilization) == ("WITHIN_RATING", "AS_RATED", True, 0.1)
    beam = _condition(b, "beam_diameter")
    assert (beam.state, beam.rated, beam.rated_low, beam.rated_high, beam.tolerance) == ("VERIFIED", 1.0, 1.0, 1.0, 0.05)
    assert beam.design_low == pytest.approx(1.009, abs=5e-4) and beam.design_high == pytest.approx(1.009 * math.sqrt(2.0), abs=1e-3)
    assert (b.status, b.applicability, b.conditions_match, b.utilization) == ("EXCEEDS_RATING", "AS_RATED", True, 2.0)
    assert b.mismatches == () and b.unverified_conditions == ()
    assert face.rating_status == "EXCEEDS_RATING" and element.rating_status == "EXCEEDS_RATING"
    assert element.max_utilization == 2.0 and element.reason_codes == ()
    assert target.status.value == "FAIL" and target.value == 2.0 and "POWER_RATING_EXCEEDED" in target.issue_codes


def test_a_rating_not_verified_keeps_the_surface_from_within_even_where_the_exposure_meets_its_value():
    """Rating B at 2 I states a 0.8 mm test beam without its definition: a 1/e^2 diameter of 0.8 to 1.36 mm, and the
    1.009 mm beam matches one reading (not verifiable: the rating does not state what deciding it needs). The
    exposure meets B's value, but B may have a threshold at most its value (as it would with a definition that made the
    beam larger than its test beam), which contradicts A's 10 I as rated: the surface is not within, and nothing bounds
    the utilization from above (removing B's definition once turned that undecided surface into
    WITHIN). A target of 1 is not decided, one of 0.05 FAILs on A. B's comparison states exposure / rated value (0.5) as an
    estimate that takes the unstated test beam as the light's own: a number beside the verdict, which it does not move."""
    peak = _peak_on_mirror()
    ratings = (_cw(10.0 * peak), _cw(2.0 * peak, beam_diameter_mm=0.8, beam_diameter_definition="NOT_STATED"))
    face, element, target = _mirror(ratings)
    b = face.ratings[1]
    beam = _condition(b, "beam_diameter")
    assert (beam.state, beam.reason) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    assert (beam.rated_low, beam.rated_high) == pytest.approx((0.8, 0.8 / 0.5887050112577373), rel=1e-15)
    assert (b.status, b.applicability, b.utilization) == ("CONDITIONS_NOT_VERIFIED", "NOT_VERIFIED", 0.5)  # (not decided)
    ratio = b.value_basis["utilization"]
    assert ratio.kind.value == "ESTIMATE" and [(a.input.value, a.at) for a in ratio.assumed] == [
        ("UNSTATED_CONDITION_AS_DESIGN", ("beam_diameter",))]
    assert b.exposure_value == peak  # (its exposure, half its value, still reported)
    assert b.unverified_conditions == (beam.message,) and "matches one reading" in beam.message
    assert face.rating_status == "CONDITIONS_NOT_VERIFIED" and element.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.1, None)
    assert "POWER_RATING_CONDITIONS_NOT_VERIFIED" in element.reason_codes
    assert target.status.value == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
    assert "beam diameter: information the rating does not state" in target.message
    assert _mirror(ratings, maximum=0.05)[2].status.value == "FAIL"


def test_a_rating_not_verified_that_may_bind_keeps_the_surface_from_within():
    """B at 0.5 I in the same band: it may apply as stated and be exceeded, or with a threshold at most its value, so the
    surface is not within (A alone made it WITHIN_RATING) and the element's utilization runs from A's 0.1 with no upper
    bound."""
    peak = _peak_on_mirror()
    face, element, target = _mirror((_cw(10.0 * peak), _cw(0.5 * peak, beam_diameter_mm=0.8)))
    assert face.ratings[1].status == "CONDITIONS_NOT_VERIFIED" and face.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.1, None)
    assert target.status.value == "UNKNOWN" and target.value == 0.1


# ------------------------------------------------------------------------------------------- one-sided ratings


def test_a_lenient_rating_as_rated_and_a_strict_one_at_most_rated_disagree_and_the_exceedance_stands():
    """A applies as rated (10 I); B (0.5 I) was measured with a 0.1 mm test beam, ten times smaller than the beam here: the
    threshold for this beam is at most 0.5 I, so B decides the exceedance (lower bound 2) while A bounds from above (0.1).
    The ratings disagree: the decided exceedance stands and nothing bounds the utilization from above -- a target of 1
    FAILs at 2, B's exceedance raises the notice, and the message names the disagreement, not the trace."""
    peak = _peak_on_mirror()
    face, element, target = _mirror((_cw(10.0 * peak), _cw(0.5 * peak, beam_diameter_mm=0.1, beam_diameter_definition="1/E2")))
    b = face.ratings[1]
    beam = _condition(b, "beam_diameter")
    assert (beam.state, beam.effect) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
    assert (b.applicability, b.status, b.conditions_match, b.utilization) == ("THRESHOLD_AT_MOST_RATED", "EXCEEDS_RATING", False, 2.0)
    assert b.mismatches == (beam.message,)
    assert face.rating_status == "EXCEEDS_RATING"
    assert (element.max_utilization, element.utilization_lower_bound, element.utilization_upper_bound) == (None, 2.0, None)
    assert element.reason_codes == ()
    assert target.status.value == "FAIL" and target.value == 2.0
    assert target.issue_codes == ("POWER_RATING_EXCEEDED", "TARGET_NOT_MET")
    assert "the ratings of M/face disagree" in target.message and "test beam smaller than the light here" in target.message


def test_ratings_that_disagree_without_an_exceedance_keep_the_lower_bound_and_no_upper_one():
    """An at-least rating puts the utilization at most 0.5 (a 3 mm test beam), an at-most one at least 0.8 (a 0.1 mm test
    beam): they cannot both hold. Neither rating is exceeded; the lower bound stands, the surface is not within, and nothing
    bounds the utilization from above -- a target of 1 is not decided, one of 0.7 FAILs at the lower end."""
    peak = _peak_on_mirror()
    ratings = (_cw(2.0 * peak, beam_diameter_mm=3.0, beam_diameter_definition="1/E2"),
               _cw(1.25 * peak, beam_diameter_mm=0.1, beam_diameter_definition="1/E2"))
    face, element, target = _mirror(ratings)
    assert [(c.applicability, c.status) for c in face.ratings] == [("THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING"),
                                                                    ("THRESHOLD_AT_MOST_RATED", "UNKNOWN")]
    assert face.rating_status == "UNKNOWN" and element.rating_status == "UNKNOWN"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.8, None)
    assert target.status.value == "UNKNOWN" and target.issue_codes == () and "disagree" in target.message
    strict = _mirror(ratings, maximum=0.7)[2]
    assert strict.status.value == "FAIL" and strict.value == 0.8 and strict.issue_codes == ("TARGET_NOT_MET",)


def test_an_interval_from_one_sided_ratings_names_their_test_beams_not_the_trace():
    """One exact beam, a complete trace: a rating as rated (u 0.5) with an at-least one (u 0.8) gives [0.5, 0.8], a rating
    as rated (u 2) with an at-most one (u 1.5) gives [2, 2]. The interval comes from the ratings' test beams, which the
    message names; no incomplete trace is claimed, and no pulses per site (a Project states none). The target's value is
    the largest exposure / rating of the ratings that apply: 0.8."""
    peak = _peak_on_mirror()
    _, element, target = _mirror((_cw(2.0 * peak), _cw(1.25 * peak, beam_diameter_mm=3.0, beam_diameter_definition="1/E2")))
    assert (element.utilization_lower_bound, element.utilization_upper_bound, element.reason_codes) == (0.5, 0.8, ())
    assert target.status.value == "PASS" and target.value == 0.8 and target.issue_codes == ()
    assert "a rating measured with a test beam at least as large as the light here" in target.message
    first = target.message
    _, element, target = _mirror((_cw(0.5 * peak), _cw(peak / 1.5, beam_diameter_mm=0.1, beam_diameter_definition="1/E2")))
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (2.0, 2.0)
    assert target.status.value == "FAIL" and target.issue_codes == ("POWER_RATING_EXCEEDED", "TARGET_NOT_MET")
    assert "a rating measured with a test beam smaller than the light here" in target.message
    for message in (first, target.message):
        assert "INCOMPLETE_TRACE" not in message and "coherent" not in message and "pulses per site" not in message


def test_a_rating_at_most_rated_alone_can_fail_a_target_but_never_pass_it():
    """Only B (a 0.1 mm test beam): exceeded at 2 I, a target FAILs; at 0.5 its exposure is within the number, but the
    threshold for this larger beam may be lower: UNKNOWN, no upper bound."""
    peak = _peak_on_mirror()
    strict = lambda value: _cw(value, beam_diameter_mm=0.1, beam_diameter_definition="1/E2")  # noqa: E731
    face, element, target = _mirror((strict(0.5 * peak),))
    assert face.rating_status == "EXCEEDS_RATING" and target.status.value == "FAIL" and target.value == 2.0
    face, element, target = _mirror((strict(2.0 * peak),))
    (comparison,) = face.ratings
    assert (comparison.status, comparison.utilization) == ("UNKNOWN", 0.5)  # (the ratio is reported; the status decides)
    assert face.rating_status == "UNKNOWN" and element.rating_status == "UNKNOWN"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.5, None)
    assert target.status.value == "UNKNOWN" and target.value == 0.5
    assert "bounds the utilization from below only" in target.message and "pulses per site" not in target.message


def test_a_rating_at_least_rated_alone_can_pass_a_target_but_never_fail_it():
    """Only a rating with a 2 mm test beam, larger than the 1.43 mm footprint here: the threshold for this beam is at least
    the rated value. At 2 I the exposure is within it: WITHIN, a target PASSes (the utilization lies in [0, 0.5]); at 0.5 I
    the exposure is above the number, but the threshold may be higher: UNKNOWN, never EXCEEDS. The exposure is exact, so the
    target's value is its ratio to the rating (0.5, then 2), never 0: the interval it is graded on is in its message."""
    peak = _peak_on_mirror()
    broad = lambda value: _cw(value, beam_diameter_mm=2.0, beam_diameter_definition="1/E2")  # noqa: E731
    face, element, target = _mirror((broad(2.0 * peak),))
    (comparison,) = face.ratings
    assert (_condition(comparison, "beam_diameter").effect, comparison.applicability) == ("THRESHOLD_AT_LEAST_RATED",) * 2
    assert comparison.status == "WITHIN_RATING" and face.rating_status == "WITHIN_RATING"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.0, 0.5)
    assert target.status.value == "PASS" and "bounds the utilization from above only" in target.message
    # (the exposure is exact; it is the utilization, against a threshold known from below only, that has no lower bound)
    assert target.message.startswith("the exposure on M's rated surfaces reaches 0.5 of a rating that applies; the utilization "
                                      "on M's rated surfaces has no lower bound above 0")
    assert target.value == 0.5 and "the utilization is between 0 and 0.5" in target.message
    face, element, target = _mirror((broad(0.5 * peak),))
    (comparison,) = face.ratings
    assert (comparison.status, comparison.utilization) == ("UNKNOWN", 2.0)
    assert face.rating_status == "UNKNOWN" and target.status.value == "UNKNOWN" and target.value == 2.0
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in run_issues((broad(0.5 * peak),)))


def run_issues(ratings: tuple[PowerRating, ...]):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
    return run(ProjectSnapshot(components=(LASER, mirror))).issues


def test_a_rating_not_verified_alone_reports_its_exposure_and_decides_nothing():
    """A 0.5 I rating whose 0.8 mm test beam of unstated definition leaves the beam between its readings: not verified.
    Its exposure is reported, and exposure / rated value (peak / (0.5 peak) = 2) with the entry that says what it is: an
    estimate that takes the unstated beam definition as the light's own. The rating is not established to apply, so the
    ratio is no exceedance: the comparison, the surface, the element and the target stay undecided."""
    peak = _peak_on_mirror()
    face, element, target = _mirror((_cw(0.5 * peak, beam_diameter_mm=0.8),))
    (comparison,) = face.ratings
    assert (comparison.status, comparison.exposure_value, comparison.utilization) == ("CONDITIONS_NOT_VERIFIED", peak, 2.0)
    assert "exposure_value" not in comparison.value_basis  # (one beam on an ideal reflector: the exposure is exact)
    ratio = comparison.value_basis["utilization"]
    assert ratio.kind.value == "ESTIMATE" and ratio.omitted == () and ratio.validity == ()
    (assumed,) = ratio.assumed
    assert (assumed.input.value, assumed.at, assumed.code) == ("UNSTATED_CONDITION_AS_DESIGN", ("beam_diameter",),
                                                               "POWER_RATING_CONDITIONS_NOT_VERIFIED")
    assert face.rating_status == "CONDITIONS_NOT_VERIFIED" and element.utilization_lower_bound is None
    assert element.max_utilization == 2.0 and element.value_basis["max_utilization"] == ratio
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in run_issues((_cw(0.5 * peak, beam_diameter_mm=0.8),)))
    assert (target.status.value, target.assessment.value, target.value) == ("UNKNOWN", "INDETERMINATE", 2.0)
    assert target.value_basis["value"] == ratio and target.message.startswith("the exposure on M's rated surfaces is 2 of")
    assert target.issue_codes == ("POWER_RATING_CONDITIONS_NOT_VERIFIED",)
    assert "beam diameter: information the rating does not state" in target.message


def test_a_rating_that_does_not_apply_is_no_failure():
    """A 0.5 I rating whose test beam straddles the beam here (both a stated 1/e^2 diameter of 1.2 mm and a 1.43 mm
    footprint): no established relationship -- CONDITIONS_MISMATCH, its exposure reported, never an exceedance."""
    peak = _peak_on_mirror()
    ratings = (_cw(0.5 * peak, beam_diameter_mm=1.2, beam_diameter_definition="1/E2"),)
    face, element, target = _mirror(ratings)
    (comparison,) = face.ratings
    beam = _condition(comparison, "beam_diameter")
    assert (beam.state, beam.effect) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert (comparison.applicability, comparison.status, comparison.exposure_value) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH", peak)
    assert face.rating_status == "CONDITIONS_MISMATCH" and element.rating_status == "CONDITIONS_MISMATCH"
    assert target.status.value == "UNKNOWN" and target.issue_codes == ("POWER_RATING_CONDITIONS_MISMATCH",)
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in run_issues(ratings))


# ------------------------------------------------------------------------------------------- an element of two surfaces


def _window(ratings: tuple[PowerRating, ...], maximum: float):
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    window = window.model_copy(update={"power_ratings": ratings})
    target = Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=maximum)
    result = run(ProjectSnapshot(components=(LASER, window), targets=(target,)))
    surfaces = {s.surface_id: s for s in result.exposures.surfaces if s.element_id == "W"}
    element = next(e for e in result.exposures.elements if e.element_id == "W")
    return surfaces, element, result.targets[0]


def test_an_element_target_does_not_pass_while_a_rated_surface_has_no_upper_bound():
    """The front face's rating is exceeded (decided by its lower bound, utilization at least 2), the rear face's states a
    0.8 mm test beam without its definition (the 1.009 mm beam matches one reading: not verified) at a tenth of its
    exposure. The rear face has no rating that bounds it from above, so a target maximum of 3 is not decided, while a
    maximum of 1.5 FAILs on the front face's decided exceedance."""
    probe, _, _ = _window((_cw(1e30),), 1.0)
    front_low = probe["W/front"].ratings[0].exposure_lower_bound
    back_high = probe["W/back"].ratings[0].exposure_upper_bound
    ratings = (_cw(0.5 * front_low, surface_ids=("front",)),
               _cw(0.1 * back_high, surface_ids=("back",), beam_diameter_mm=0.8, beam_diameter_definition="NOT_STATED"))
    surfaces, element, target = _window(ratings, 3.0)
    assert surfaces["W/front"].rating_status == "EXCEEDS_RATING" and surfaces["W/back"].rating_status == "CONDITIONS_NOT_VERIFIED"
    assert element.rating_status == "EXCEEDS_RATING"
    lower = surfaces["W/front"].ratings[0].utilization
    assert math.isclose(lower, 2.0, rel_tol=1e-9) and element.utilization_lower_bound == lower
    assert element.utilization_upper_bound is None
    assert target.status.value == "UNKNOWN" and target.value == lower
    assert {"POWER_RATING_EXCEEDED", "POWER_RATING_CONDITIONS_NOT_VERIFIED"} <= set(target.issue_codes)
    assert _window(ratings, 1.5)[2].status.value == "FAIL"


def test_a_rated_surface_whose_ratings_do_not_match_leaves_the_utilization_open_above():
    """The front face's CW rating matches (within it), the rear face's only rating is for pulsed light: no rating applies
    there, so the element's utilization is the front face's at least, without an upper bound -- not a failure."""
    probe, _, _ = _window((_cw(1e30),), 1.0)
    front_high = probe["W/front"].ratings[0].exposure_upper_bound
    pulsed = PowerRating(regime="PULSED", value=1.0, unit="J/cm2", wavelength_nm=1064.0, pulse_duration_s=1e-8, status="DECLARED",
                         basis=FIXTURE, surface_ids=("back",), value_definition="PEAK")
    ratings = (_cw(2.0 * front_high, surface_ids=("front",)), pulsed)
    surfaces, element, target = _window(ratings, 1.0)
    assert surfaces["W/front"].rating_status == "WITHIN_RATING" and surfaces["W/back"].rating_status == "CONDITIONS_MISMATCH"
    back = surfaces["W/back"].ratings[0]
    assert back.applicability == "NOT_APPLICABLE" and back.exposure_value is None and back.exposure_upper_bound is None
    assert element.rating_status == "CONDITIONS_MISMATCH" and element.utilization_upper_bound is None
    front = surfaces["W/front"].ratings[0]
    assert element.utilization_lower_bound == front.exposure_lower_bound / front.rating_value
    assert "POWER_RATING_CONDITIONS_MISMATCH" in element.reason_codes
    assert target.status.value == "UNKNOWN" and "POWER_RATING_CONDITIONS_MISMATCH" in target.issue_codes
    assert _window(ratings, 0.5 * element.utilization_lower_bound)[2].status.value == "FAIL"


# ------------------------------------------------------------------- an at-most rating against one that applies as stated


def test_an_at_most_rating_whose_value_is_below_the_exposure_keeps_the_surface_from_within():
    """Review M1: A applies as rated (10 I: within) and B (2 I, measured with a 0.1 mm test beam: the threshold for this
    larger beam is at most 2 I) is not exceeded (u 0.5). B's threshold may lie anywhere below 2 I -- below the exposure
    too --, and A says it is 10 I: the two disagree, the surface has no upper bound and is not within, and a target of 1 is
    not decided (its message names the disagreement; was WITHIN / PASS). With A at 1.5 I, no more than B's 2 I, the
    threshold lies between the two and A bounds it: WITHIN, and the target PASSes at A's utilization."""
    peak = _peak_on_mirror()
    strict = _cw(2.0 * peak, beam_diameter_mm=0.1, beam_diameter_definition="1/E2")
    face, element, target = _mirror((_cw(10.0 * peak), strict))
    a, b = face.ratings
    assert (a.applicability, a.status) == ("AS_RATED", "WITHIN_RATING")
    assert (b.applicability, b.status, b.utilization) == ("THRESHOLD_AT_MOST_RATED", "UNKNOWN", 0.5)  # (exact: its ratio)
    assert face.rating_status == "UNKNOWN" and element.rating_status == "UNKNOWN"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (0.5, None)
    assert target.status.value == "UNKNOWN" and target.value == 0.5 and "the ratings of M/face disagree" in target.message
    face, element, target = _mirror((_cw(1.5 * peak), strict))
    assert face.rating_status == "WITHIN_RATING" and element.rating_status == "WITHIN_RATING"
    assert element.max_utilization is None and element.utilization_upper_bound == pytest.approx(1.0 / 1.5, rel=1e-12)
    assert target.status.value == "PASS" and target.value == pytest.approx(1.0 / 1.5, rel=1e-12)


def test_an_at_most_comparison_is_bounded_only_by_one_of_its_own_quantity():
    """``_at_most_open``: an at-most comparison is bounded by an as-rated or at-least comparison of its unit whose lowest
    threshold is no larger than its highest under every reading (``rated_threshold``; W/cm values compared in P / d of the
    1/e^2 diameter: a FWHM value stands 1 / 0.5887 above the same threshold, an unstated definition between); one whose
    lowest may be larger contradicts it (the ratings disagree); comparisons of another unit are criteria of their own and
    bound nothing for it."""
    from types import SimpleNamespace

    from os_optics.validation.exposures import E2_NOTE, FWHM_NOTE
    from os_optics.validation.exposures import _at_most_open as openings

    def _at_most_open(comparisons):
        # (whether the surface is left open -- some comparison opens it -- and whether the ratings disagree; the
        # comparisons that open it are checked below)
        opened, disagree = openings(comparisons)
        return bool(opened), disagree

    def c(applicability: str, value: float, unit: str = "W/cm2", note: str | None = None, high: float | None = None):
        return SimpleNamespace(applicability=applicability, rating_value=value, unit=unit, notes=(note,) if note else (),
                               rated_threshold_low=value, rated_threshold_high=value if high is None else high)

    at_most = c("THRESHOLD_AT_MOST_RATED", 2.0)
    assert _at_most_open((at_most, c("AS_RATED", 1.5))) == (False, False)
    assert _at_most_open((at_most, c("THRESHOLD_AT_LEAST_RATED", 2.0))) == (False, False)
    assert _at_most_open((at_most, c("AS_RATED", 10.0))) == (True, True)
    assert _at_most_open((at_most, c("AS_RATED", 10.0), c("AS_RATED", 1.0))) == (False, False)
    assert _at_most_open((at_most, c("NOT_VERIFIED", 1.0))) == (True, False)
    # a value whose reading is not stated (threshold between 2 and 4) against a stated peak of 3: they bound it together
    assert _at_most_open((c("THRESHOLD_AT_MOST_RATED", 2.0, high=4.0), c("AS_RATED", 3.0))) == (False, False)
    assert _at_most_open((c("THRESHOLD_AT_MOST_RATED", 2.0, "J/cm2"), c("AS_RATED", 1.0))) == (True, False)
    linear = c("THRESHOLD_AT_MOST_RATED", 2.0, "W/cm", E2_NOTE)
    assert _at_most_open((linear, c("AS_RATED", 1.0, "W/cm", E2_NOTE))) == (False, False)
    # a FWHM value of 3.0 is 3.0 x 0.5887 = 1.766 in 1/e^2 terms (no larger than 2.0); one of 4.0 is 2.355 (they disagree)
    assert _at_most_open((linear, c("AS_RATED", 3.0, "W/cm", FWHM_NOTE))) == (False, False)
    assert _at_most_open((linear, c("AS_RATED", 4.0, "W/cm", FWHM_NOTE))) == (True, True)
    # an applying value of unstated definition is at most its value in 1/e^2 terms: 1.5 bounds the at-most 2.0, 2.5 may not
    assert _at_most_open((linear, c("AS_RATED", 1.5, "W/cm"))) == (False, False)
    assert _at_most_open((linear, c("AS_RATED", 2.5, "W/cm"))) == (True, True)
    # an at-most value of unstated definition may be 2.0 x 0.5887 = 1.177 in 1/e^2 terms: 1.5 (1/e^2) may lie above it
    unread = c("THRESHOLD_AT_MOST_RATED", 2.0, "W/cm")
    assert _at_most_open((unread, c("AS_RATED", 1.5, "W/cm", E2_NOTE))) == (True, True)
    assert _at_most_open((unread, c("AS_RATED", 1.0, "W/cm", E2_NOTE))) == (False, False)
    assert _at_most_open((linear, c("AS_RATED", 1.0, "W/cm2"))) == (True, False)  # (another unit)
    # (each comparison that opens the surface is named, with whether it disagrees with a rating of its unit)
    unverified = c("NOT_VERIFIED", 1.0)
    assert openings((at_most, c("AS_RATED", 10.0), unverified)) == (((at_most, True), (unverified, False)), True)


# ------------------------------------------------------------------------------------------- the exceedance notice


def _notice(ratings: tuple[PowerRating, ...], *, target: bool):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
    targets = (Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),) if target else ()
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=targets))
    (notice,) = [i for i in result.issues if i.code.value == "POWER_RATING_EXCEEDED"]
    return notice, result


def test_the_exceedance_notice_names_the_damage_threshold_a_total_per_area_value_sets():
    """A TOTAL_PER_1E2_AREA value sets a damage threshold twice itself for the peak exposure. At 0.4 of the peak the
    exposure is 2.5 x the value but 1.25 x the threshold the comparison was decided against: the notice states the
    threshold and that ratio (its utilization stays exposure / rated value), and says what assesses the rating -- the
    run's status only when no utilization target on the element does."""
    peak = _peak_on_mirror()
    total = _cw(0.4 * peak, value_definition="TOTAL_PER_1E2_AREA")
    notice, result = _notice((total,), target=False)
    context = notice.numerical_context
    assert context["utilization"] == pytest.approx(2.5, rel=1e-9) and context["threshold_ratio"] == pytest.approx(1.25, rel=1e-9)
    assert context["damage_threshold"] == pytest.approx(0.8 * peak, rel=1e-9) and context["utilization_target"] is False
    assert f"1.25 x the damage threshold of {0.8 * peak:.4g} W/cm2 that the rating sets for the peak exposure" in notice.message
    assert "the run's status is unchanged" in notice.message and result.status.value != "FAIL"
    notice, result = _notice((total,), target=True)
    assert result.targets[0].status.value == "FAIL" and result.targets[0].value == pytest.approx(1.25, rel=1e-9)
    assert "status is unchanged" not in notice.message
    assert "the power_rating_utilization target on the element assesses it as a requirement" in notice.message
    # (a PEAK value is its own threshold: the notice reads in units of the rating)
    notice, _ = _notice((_cw(0.5 * peak),), target=False)
    assert "(2 x the rating)" in notice.message and notice.numerical_context["damage_threshold"] == pytest.approx(0.5 * peak)


def test_the_exceedance_notice_reports_the_rating_whose_threshold_is_exceeded_most():
    """Of two exceeded ratings the notice reports the one whose damage threshold the exposure exceeds most: a PEAK value at
    0.5 of the peak (2 x its threshold) over a TOTAL_PER_1E2_AREA value at 0.3 (3.3 x the value, 1.67 x its threshold)."""
    peak = _peak_on_mirror()
    notice, _ = _notice((_cw(0.3 * peak, value_definition="TOTAL_PER_1E2_AREA"), _cw(0.5 * peak)), target=False)
    assert notice.numerical_context["rating_index"] == 1 and notice.numerical_context["threshold_ratio"] == pytest.approx(2.0)
    assert notice.numerical_context["exceeded_ratings"] == 2


def test_a_utilization_target_without_its_element_names_the_project_field():
    """The refusal names the Project target's field (device_instance_id), not only the Engine request's."""
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (_cw(1.0),)})
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=(Target(target_id="u", metric="power_rating_utilization",
                                                                             maximum=1.0),)))
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and target.issue_codes == ("INVALID_INPUT",)
    assert "device_instance_id" in target.message


# ------------------------------------------------------------------- what the element's summary label stands for


def _open_by(element) -> set[tuple]:
    return {(o.surface_id, o.rating_index, o.applicability, o.cause) for o in element.utilization_open_by}


def _counts_of(surfaces) -> dict[str, int]:
    out: dict[str, int] = {}
    for surface in surfaces.values():
        for c in surface.ratings:
            out[c.status] = out.get(c.status, 0) + 1
    return out


def test_an_element_counts_its_comparisons_and_names_what_leaves_its_utilization_open():
    """The element's ``rating_status`` is the most severe label; the comparisons are the results. The element states how
    many comparisons have each status and, where its utilization has no upper bound, which surface and comparison leave
    it open and why -- the same facts the surface rule reads (``_surface_bounds``). Nothing is decided by them: every
    status, bound and assessment stays.

    * The window with an exceeded front face and a rear face whose test beam definition is not stated (not verified): the
      rear face has no rating that bounds it from above, and its unverified rating may put the threshold anywhere below
      its value -- two entries on ``W/back``; the target stays UNKNOWN and its message names them.
    * The mirror with a rating that applies as stated (10 I) and one with a threshold at most 2 I: the two disagree --
      one entry naming the at-most comparison. With the first at 1.5 I the threshold lies between the two: an upper bound,
      no entry."""
    probe, _, _ = _window((_cw(1e30),), 1.0)
    front_low = probe["W/front"].ratings[0].exposure_lower_bound
    back_high = probe["W/back"].ratings[0].exposure_upper_bound
    ratings = (_cw(0.5 * front_low, surface_ids=("front",)),
               _cw(0.1 * back_high, surface_ids=("back",), beam_diameter_mm=0.8, beam_diameter_definition="NOT_STATED"))
    surfaces, element, target = _window(ratings, 3.0)
    assert element.rating_status == "EXCEEDS_RATING" and element.utilization_upper_bound is None
    assert element.rating_status_counts == _counts_of(surfaces) and sum(element.rating_status_counts.values()) == 2
    assert element.rating_status_counts["EXCEEDS_RATING"] == 1
    assert _open_by(element) == {("W/back", None, None, "NO_UPPER_RATING"),
                                 ("W/back", 1, "NOT_VERIFIED", "THRESHOLD_OPEN_BELOW")}
    assert target.status.value == "UNKNOWN" and "no upper bound from W/back: no rating applies" in target.message
    assert "W/back rating 1 (NOT_VERIFIED): the damage threshold may lie anywhere below the rated value" in target.message

    peak = _peak_on_mirror()
    strict = _cw(2.0 * peak, beam_diameter_mm=0.1, beam_diameter_definition="1/E2")
    face, element, target = _mirror((_cw(10.0 * peak), strict))
    assert element.utilization_upper_bound is None and target.status.value == "UNKNOWN"
    assert _open_by(element) == {("M/face", 1, "THRESHOLD_AT_MOST_RATED", "RATINGS_DISAGREE")}
    assert element.rating_status_counts == {"UNKNOWN": 1, "WITHIN_RATING": 1}
    face, element, target = _mirror((_cw(1.5 * peak), strict))
    assert element.utilization_upper_bound is not None and element.utilization_open_by == ()
    dumped = element.model_dump(mode="json")
    assert "utilization_open_by" not in dumped and dumped["rating_status_counts"] == element.rating_status_counts


def test_an_unrated_element_and_a_bounded_one_carry_no_opening():
    """``utilization_open_by`` is empty exactly where the element's utilization has an upper bound (or is exact); an
    element without ratings carries neither new field."""
    probe, _, _ = _window((_cw(1e30),), 1.0)
    front_high = probe["W/front"].ratings[0].exposure_upper_bound
    surfaces, element, _ = _window((_cw(2.0 * front_high),), 1.0)
    assert element.utilization_open_by == () and element.rating_status_counts == _counts_of(surfaces)
    pulsed = PowerRating(regime="PULSED", value=1.0, unit="J/cm2", wavelength_nm=1064.0, pulse_duration_s=1e-8, status="DECLARED",
                         basis=FIXTURE, surface_ids=("back",), value_definition="PEAK")
    surfaces, element, _ = _window((_cw(2.0 * front_high, surface_ids=("front",)), pulsed), 1.0)
    assert element.utilization_upper_bound is None
    assert _open_by(element) == {("W/back", None, None, "NO_UPPER_RATING")}
    assert element.rating_status_counts == {"CONDITIONS_MISMATCH": 1, "WITHIN_RATING": 1}
