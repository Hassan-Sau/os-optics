"""The windows within which a rating's test condition counts as matched are the run's settings
(``NumericalConfig.rating_*``, a Project's ``numerical_settings``): each one moves its own condition and nothing else, the
values in force are reported with the result, and a comparison made in a window the caller set says so on the number it
decides. The exposure of light a rating is not for is stated beside the condition that differs.

References: the mirror scene's exact peak (one beam on an ideal reflector) and the ratios written out from it.
"""

from __future__ import annotations

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import PowerRating, ProjectSnapshot, Target
from os_optics.local import run_optical_project
from os_optics.settings_registry import named_settings, unrecorded_settings
from os_optics.validation.rating_conditions import DEFAULT_TOLERANCES, RatingTolerances

from helpers import run
from test_power_exposure_ratings import _document, _rating, _surface
from test_rating_decisions import FIXTURE, LASER, TEST_BEAM_MM, _peak_on_mirror

DEFAULTS = {"wavelength_rel": 0.02, "pulse_duration_rel": 0.10, "repetition_rate_rel": 0.10, "beam_diameter_rel": 0.05,
            "polarization_fraction": 0.01, "unlocated_light_share": 0.01}


def _mirror(rating: PowerRating, **settings):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0)
    numerical = NumericalConfig().model_copy(update=settings) if settings else None
    result = run(ProjectSnapshot(components=(LASER, mirror), targets=(target,)), numerical=numerical)
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    return result, face.ratings[0], result.targets[0]


def _cw(value: float, wavelength_nm: float = 1064.0, test_beam_mm: float = TEST_BEAM_MM) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength_nm, status="DECLARED", basis=FIXTURE,
                       value_definition="PEAK", polarization="LINEAR", polarization_condition="linear",
                       beam_diameter_mm=test_beam_mm, beam_diameter_definition="1/E2")


def _states(comparison) -> dict[str, str]:
    return {c.condition: c.state for c in comparison.conditions}


def test_the_values_in_force_are_the_run_s_settings_and_default_to_the_engine_s():
    result, _, _ = _mirror(_cw(1e30))
    assert result.exposures.tolerances == DEFAULTS == DEFAULT_TOLERANCES.as_dict()
    widened, _, _ = _mirror(_cw(1e30), rating_wavelength_tol_rel=0.05, rating_unlocated_light_share=0.03)
    assert widened.exposures.tolerances == {**DEFAULTS, "wavelength_rel": 0.05, "unlocated_light_share": 0.03}
    assert RatingTolerances.of(NumericalConfig(rating_beam_diameter_tol_rel=0.2)).beam_diameter_rel == 0.2


def test_a_widened_wavelength_window_verifies_a_near_match_and_says_so_on_the_ratio():
    """A 1030 nm rating against 1064 nm light: 3.3 % apart. In the default 2 % window the wavelength differs -- the
    rating is for other light (CONDITIONS_MISMATCH) -- and the exposure is still stated in the rating's unit, with its
    ratio to the rated value (peak / (4 peak) = 0.25) as an estimate whose entry carries the check that failed: the
    light's 1064 nm outside 1030 nm +- 2 %, under the setting that holds the window. With the window set to 5 % the
    wavelength is verified, the comparison decides (WITHIN_RATING, utilization 0.25) and the ratio's entry states the
    match with the window it was made in. No other condition moves."""
    peak = _peak_on_mirror()
    rating = _cw(4.0 * peak, wavelength_nm=1030.0)
    result, comparison, target = _mirror(rating)
    assert _states(comparison)["wavelength"] == "DIFFERS" and comparison.status == "CONDITIONS_MISMATCH"
    assert comparison.exposure_value == peak and comparison.utilization == pytest.approx(0.25, rel=1e-12)
    entry = comparison.value_basis["utilization"]
    (check,) = entry.validity
    assert entry.kind.value == "ESTIMATE" and (check.parameter, check.within, check.setting, check.code) == (
        "wavelength", False, "rating_wavelength_tol_rel", "POWER_RATING_CONDITIONS_MISMATCH")
    assert (check.minimum, check.maximum, check.observed) == pytest.approx((1030.0 * 0.98, 1030.0 * 1.02, 1064.0), rel=1e-12)
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE") and target.value == comparison.utilization
    assert not unrecorded_settings(result) and "rating_wavelength_tol_rel" in named_settings(result)

    wide_result, wide, wide_target = _mirror(rating, rating_wavelength_tol_rel=0.05)
    wavelength = next(c for c in wide.conditions if c.condition == "wavelength")
    assert (wavelength.state, wavelength.tolerance) == ("VERIFIED", 0.05)
    assert {k: v for k, v in _states(wide).items() if k != "wavelength"} == {k: v for k, v in _states(comparison).items() if k != "wavelength"}
    assert (wide.status, wide.applicability, wide.utilization) == ("WITHIN_RATING", "AS_RATED", pytest.approx(0.25, rel=1e-12))
    (match,) = wide.value_basis["utilization"].validity
    assert wide.value_basis["utilization"].kind.value == "MODEL"
    assert (match.parameter, match.within, match.setting, match.maximum) == ("wavelength", True, "rating_wavelength_tol_rel",
                                                                           pytest.approx(1030.0 * 1.05, rel=1e-12))
    assert (wide_target.status.value, wide_target.assessment.value) == ("PASS", "MET")
    assert wide_target.value == pytest.approx(0.25, rel=1e-12)
    assert not unrecorded_settings(wide_result)


def test_the_beam_diameter_window_moves_the_beam_condition_alone():
    """A test beam 8 % larger than the light's: outside the default 5 % window (the beam differs from the test beam:
    the rating does not apply as stated), inside a 10 % one (verified: it does). The other conditions keep their states."""
    peak = _peak_on_mirror()
    rating = _cw(4.0 * peak, test_beam_mm=1.08 * TEST_BEAM_MM)
    _, default, _ = _mirror(rating)
    _, wide, _ = _mirror(rating, rating_beam_diameter_tol_rel=0.10)
    assert _states(default)["beam_diameter"] == "DIFFERS" and _states(wide)["beam_diameter"] == "VERIFIED"
    assert {k: v for k, v in _states(wide).items() if k != "beam_diameter"} == {
        k: v for k, v in _states(default).items() if k != "beam_diameter"}
    assert default.applicability != "AS_RATED" and wide.applicability == "AS_RATED"
    assert next(c for c in wide.conditions if c.condition == "beam_diameter").tolerance == 0.10
    # (a window that only another setting holds moves nothing here)
    _, other, _ = _mirror(rating, rating_wavelength_tol_rel=0.5, rating_polarization_fraction_tol=0.2)
    assert _states(other) == _states(default)


def test_the_share_unlocated_light_may_carry_is_its_own_setting():
    """A rated window: its own reflections and the light left out as negligible reach its front face beside the beam. At
    the default share (1 %) the rating's conditions are verified. With ``rating_unlocated_light_share`` set to a
    vanishing share that light is no longer minor: conditions become not verifiable for the light of unknown size or
    state -- while a vanishing ``rating_polarization_fraction_tol`` alone (the share of a beam's own power outside the
    rated state) leaves the beam-size condition as it was. The two roles of the former single constant are two settings."""
    def conditions(**settings):
        numerical = {"numerical_settings": settings} if settings else {}
        run_ = run_optical_project({**_document([_rating("CW", 1e9, "W/cm2")]), **numerical})
        (comparison,) = _surface(run_, "W/S1").ratings
        return run_, {c.condition: (c.state, c.reason) for c in comparison.conditions}

    _, default = conditions()
    share_run, share = conditions(rating_unlocated_light_share=1e-12)
    _, state = conditions(rating_polarization_fraction_tol=1e-12)
    assert share_run.result.exposures.tolerances["unlocated_light_share"] == 1e-12
    assert share_run.result.exposures.tolerances["polarization_fraction"] == 0.01
    moved = {name for name in default if share[name] != default[name]}
    assert moved and all(share[name][0] == "NOT_VERIFIABLE" for name in moved)
    assert {share[name][1] for name in moved} <= {"BEAM_SIZE_UNKNOWN", "LIGHT_STATE_UNKNOWN", "PULSES_PER_SITE_UNDETERMINED"}
    assert state["beam_diameter"] == default["beam_diameter"] and state["wavelength"] == default["wavelength"]
