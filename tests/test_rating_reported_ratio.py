"""The numbers a rating comparison reports beside its verdict: the followed beams' exposure where the exposure is a
range, and exposure / rated value where the comparison is not decided -- each with the ``value_basis`` entry that says what
the number is. They are reported, never graded: every status, bound and assessment is the one the bounds and the
conditions give without them.

References are independent of the code under test: the beam's Gaussian peak ``2 P / (pi w^2)`` written out, and the
verdicts recomputed from the same comparisons with the reported numbers taken away.
"""

from __future__ import annotations

import math

import pytest

from os_optics.contracts import PowerRating
from os_optics.contracts.basis import ValueBasis
from os_optics.local import run_optical_project
from os_optics.validation import exposures as E

from test_power_exposure_ratings import _document, _element, _independent, _peak_w_per_cm2, _rating, _surface
from test_rating_decisions import _cw, _mirror, _peak_on_mirror

UTIL = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]


def _stripped(comparison):
    """The comparison as it stood before a number was reported beside its bounds: no entry, and no number an entry names."""
    named = comparison.value_basis or {}
    return comparison.model_copy(update={"value_basis": None, **{name: None for name in named}})


def _verdicts(run):
    report = run.result.exposures
    return ([(s.surface_id, s.rating_status, tuple((c.status, c.applicability) for c in s.ratings)) for s in report.surfaces],
            [(e.element_id, e.rating_status, e.utilization_lower_bound, e.utilization_upper_bound) for e in report.elements],
            [(t.target_id, t.status.value, t.assessment.value) for t in run.result.targets])


def test_the_ratio_of_an_unverified_rating_is_the_exposure_over_the_rated_value_as_stated():
    """Closed form: one beam on an ideal reflector has the exact peak ``I``; a rating of ``I / k`` whose test beam
    definition is not stated reports ``utilization = k`` for every ``k`` -- as an estimate that names the unstated
    condition -- while the comparison stays CONDITIONS_NOT_VERIFIED and the target INDETERMINATE (the rating may have a
    threshold on either side of its value, so the ratio is no verdict)."""
    peak = _peak_on_mirror()
    for k in (0.1, 0.5, 2.0, 10.0):
        face, element, target = _mirror((_cw(peak / k, beam_diameter_mm=0.8),))
        (comparison,) = face.ratings
        assert comparison.utilization == pytest.approx(k, rel=1e-12)
        entry = comparison.value_basis["utilization"]
        assert isinstance(entry, ValueBasis) and entry.kind.value == "ESTIMATE"
        assert [(a.input.value, a.at) for a in entry.assumed] == [("UNSTATED_CONDITION_AS_DESIGN", ("beam_diameter",))]
        assert (comparison.status, face.rating_status, element.rating_status) == ("CONDITIONS_NOT_VERIFIED",) * 3
        assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
        assert target.value == pytest.approx(k, rel=1e-12) and target.value_basis["value"].kind.value == "ESTIMATE"
        assert (element.utilization_lower_bound, element.utilization_upper_bound) == (None, None)


def test_a_decided_utilization_carries_no_entry():
    """A rating whose every stated condition is verified, on an exact exposure: the ratio is the utilization (no entry),
    as before."""
    peak = _peak_on_mirror()
    face, element, target = _mirror((_cw(4.0 * peak),))
    (comparison,) = face.ratings
    assert (comparison.status, comparison.utilization, comparison.value_basis) == ("WITHIN_RATING", 0.25, None)
    assert (element.max_utilization, element.value_basis) == (0.25, None)
    assert (target.status.value, target.value, target.value_basis) == ("PASS", 0.25, None)


def test_the_exposure_of_a_range_is_the_followed_beam_s_own_within_the_bounds():
    """A matched, rated window: the window's own reflections and the light left out as negligible make the exposure of
    its front face a range. The comparison reports the traced beam's Gaussian peak ``2 P / (pi w^2)`` (written out from
    the beam's power and radius there) as ``exposure_value`` inside ``[exposure_lower_bound, exposure_upper_bound]``,
    and its ratio to the rated value -- both estimates naming the light not counted; WITHIN_RATING is still decided by the
    upper bound."""
    power, w = _independent()["W/S1"]
    peak = _peak_w_per_cm2(power, w)
    assert peak == pytest.approx(2.0 * power / (math.pi * w * w) * 100.0, rel=1e-12)
    run = run_optical_project(_document([_rating("CW", 4.0 * peak, "W/cm2")], targets=UTIL))
    (comparison,) = _surface(run, "W/S1").ratings
    assert comparison.exposure_value == pytest.approx(peak, rel=1e-9)
    assert comparison.exposure_lower_bound < comparison.exposure_value < comparison.exposure_upper_bound
    assert comparison.utilization == pytest.approx(0.25, rel=1e-9)
    assert comparison.status == "WITHIN_RATING" and comparison.exposure_upper_bound <= comparison.rating_value
    assert E.exact_exposure(comparison) is None  # (not the exact exposure: no decision reads it)
    for name in ("exposure_value", "utilization"):
        entry = comparison.value_basis[name]
        assert entry.kind.value == "ESTIMATE" and entry.assumed == () and entry.validity == ()
        assert {o.part.value for o in entry.omitted} == {"BELOW_NEGLIGIBLE_LEVEL", "NOT_FOLLOWED"}
    element = _element(run, "W")
    assert element.rating_status == "WITHIN_RATING" and element.utilization_upper_bound < 1.0
    (target,) = run.result.targets
    assert (target.status.value, target.assessment.value) == ("PASS", "MET")
    assert target.value == element.utilization_lower_bound and target.value_basis is None  # (graded on the proven interval)


@pytest.mark.parametrize("factor", [0.2, 0.9, 1.1, 5.0])
def test_no_status_bound_or_assessment_reads_a_reported_number(factor):
    """The verdicts recomputed from the comparisons with the reported numbers taken away (``_stripped``) equal the ones
    the run gives, from a rating far above the exposure to one far below it: the surface's status and utilization bounds
    come from the proven bounds and the conditions alone. (With the guard removed -- ``exact_exposure`` returning the
    reported number -- the rating at 1.1 times the traced peak reads WITHIN_RATING.)"""
    power, w = _independent()["W/S1"]
    peak = _peak_w_per_cm2(power, w)
    ratings = [_rating("CW", factor * peak, "W/cm2"), _rating("CW", factor * peak, "W/cm2", beam_diameter_mm=0.8)]
    run = run_optical_project(_document(ratings, targets=UTIL))
    reported = 0
    for surface in run.result.exposures.surfaces:
        if not surface.ratings:
            continue
        reported += sum(1 for c in surface.ratings if c.value_basis)
        bare = tuple(_stripped(c) for c in surface.ratings)
        assert E._surface_status(list(bare)) == surface.rating_status
        assert E._surface_bounds(bare) == E._surface_bounds(surface.ratings)
        assert [E._utilization_bounds(c) for c in bare] == [E._utilization_bounds(c) for c in surface.ratings]
    assert reported  # (the scene does report numbers beside its bounds)
    (target,) = run.result.targets
    element = _element(run, "W")
    low, high = element.utilization_lower_bound, element.utilization_upper_bound
    # (the target is graded on the proven interval alone: FAIL above the maximum, PASS below it, else not decided)
    expected = "FAIL" if low is not None and low > 1.0 else "PASS" if high is not None and high <= 1.0 else "UNKNOWN"
    assert target.status.value == expected
