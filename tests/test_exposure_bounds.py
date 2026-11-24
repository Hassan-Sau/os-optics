"""Exposure bounds, rating notices and the related scope rules.

Each test reproduces a finding of the independent reviews with the
review's own setup. Power ratings here are fixture data declared in the test Project, never product data; their W/cm2
values are stated as peak values (``value_definition`` PEAK), the threshold the decisions below are read against, with
the beam's own 1/e^2 diameter at the rated element (Gaussian propagation, ``_beam_mm``) as their test beam and linear
polarization (the source's).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.contracts import NeutralDensityInteraction
from os_optics import ValidationRequest, validate
from os_optics.local import LocalRunError, json_api, lower_project, run_optical_project
from os_optics.math.exact_linalg import inverse_2x2, singular_values
from os_optics.math.exact_trig import atan2_deg, exp, log10, pow10
from os_optics.native_library import add_to_project, build_native_model
from os_optics.optics.rays.tracer import PowerFlow, SourceInput, TraceOutput
from os_optics.config import NumericalConfig
from os_optics.scene.model import InteractionG
from os_optics.validation.power import build_power_budget

from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import ALONG_X, NM, plane, project, rayleigh_range, source
from test_coating_tables_and_phases import _quaternion
from test_power_exposure_ratings import (_document, _element, _independent, _n, _peak_w_per_cm2, _rating, _surface, _traced,
                                        faint_amplitude, open_paths)
from test_tabulated_thin_lens import EVIDENCE

FIXTURE = "test fixture rating, not product data"
INV = 1.0 / math.sqrt(2.0)


def _beam_mm(path_mm: float, waist_mm: float = 0.5, wavelength_nm: float = 780.0) -> float:
    """The 1/e^2 diameter of a source's beam (waist at the source) ``path_mm`` from it in air: 2 w0 sqrt(1 + (z / z_R)^2)."""
    return 2.0 * waist_mm * math.sqrt(1.0 + (path_mm / rayleigh_range(waist_mm, wavelength_nm)) ** 2)


def _stated(rating: dict, beam_mm: float) -> dict:
    """``rating`` with the test conditions the light at the rated element meets: a test beam of ``beam_mm`` (1/e^2) and
    linear polarization."""
    return {**rating, "beam_diameter_mm": f"{beam_mm:.10g}", "beam_diameter_definition": "1/E2", "polarization_condition": "linear"}


# ------------------------------------------------------------------------------------------------ no gate


def test_an_exceeded_rating_without_a_utilization_target_keeps_the_run_and_the_local_validation_passing():
    """rv3 probe_rating_gate: an exceedance is reported (exposures + a NOT_APPLICABLE notice), never a commercial gate."""
    power, w = _independent()["W/S1"]
    peak = _peak_w_per_cm2(power, w)
    targets = [{"target_id": "p", "metric": "power_W", "observation_id": "screen", "minimum": 0.1}]
    for factor in (10.0, 0.5):
        run = run_optical_project(_document([_rating("CW", factor * peak, "W/cm2")], targets=targets))
        assert run.result.status.value == "PASS", factor
        record = run.local_validation
        assert (record.validation_status.value, record.simulation.status.value) == ("PASS", "PASS"), factor
        notices = [i for i in run.result.issues if i.code.value == "POWER_RATING_EXCEEDED"]
        assert all(i.status.value == "NOT_APPLICABLE" for i in notices)
        assert bool(notices) == (factor < 1.0)
        assert _element(run, "W").rating_status == ("EXCEEDS_RATING" if factor < 1.0 else "WITHIN_RATING")
        # the manifest's warnings still name it (informational)
        assert any(w.startswith("POWER_RATING_EXCEEDED") for w in run.result.manifest.warnings) == (factor < 1.0)
    # only a customer-declared target turns it into a requirement
    targets.append({"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0})
    run = run_optical_project(_document([_rating("CW", 0.5 * peak, "W/cm2")], targets=targets))
    assert run.result.status.value == "FAIL"
    assert {t.target_id: t.status.value for t in run.result.targets} == {"p": "PASS", "util": "FAIL"}


# ------------------------------------------------------------------------------- B1: untraced light on a rated element


def _mirror_window(reflectance: str, transmittance: str, *, maximum: float = 1.0) -> dict:
    """rv2 t02b: a 1 W 780 nm beam folded by a flat mirror whose coating band is R / T; a rated N-BK7 window W sits 60 mm
    behind the mirror on its *transmitted* line (light the trace does not follow)."""
    built = json_api.build_native_model("mirror.flat", {"reflectance": "0.9", "valid_band_nm": ["400", "1100"], "diameter_mm": "25.4"})
    model, (coat,) = copy.deepcopy(built["optical_model"]), copy.deepcopy(built["model_coatings"])
    coat["bands"][0]["reflectance"] = reflectance
    coat["bands"][0]["transmittance"] = transmittance
    pose = _quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(-INV, INV, 0.0))
    rating = _stated({"regime": "CW", "value": "1", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
                      "status": "DECLARED", "basis": FIXTURE}, _beam_mm(260.0))  # (W 260 mm from the source's waist)
    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]), "power_ratings": [rating]}
    components = [source(power_w=1.0, waist_mm=0.5), declared_component("M", "mirror", model["model_id"], quaternion=pose),
                  declared_component("W", "window", "win", position=(60.0, 0.0, 0.0))]
    observations = [plane("OUT", (0.0, 100.0, 0.0), quaternion=_quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(0.0, 1.0, 0.0)))]
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": maximum}]
    return with_models(project(components, observations, targets), [model, win], [nbk7()], [coat])


def test_a_rated_element_reached_only_by_untraced_light_is_not_a_known_zero():
    run = run_optical_project(_mirror_window("0.5", "0.5"))
    exposures = run.result.exposures
    assert exposures.untraced_light_included is True and exposures.trace_complete is True
    s1 = _surface(run, "W/S1")
    assert s1.hits == () and not s1.complete and "UNTRACED_LIGHT_BOUNDED" in s1.reason_codes  # (the coating transmission)
    (light,) = [h for h in s1.untraced_hits if h.pass_index == 0]
    assert light.incident_power_W == pytest.approx(0.5, rel=1e-9)  # the coating's transmitted half
    upper = _peak_w_per_cm2(light.incident_power_W, math.sqrt(light.beam_radius_x_mm * light.beam_radius_y_mm))
    (comparison,) = s1.ratings
    assert comparison.status == "UNKNOWN" and comparison.exposure_value is None
    assert comparison.exposure_lower_bound is None and comparison.exposure_upper_bound >= upper * (1.0 - 1e-9) and upper > 100.0
    element = _element(run, "W")
    assert element.rating_status == "UNKNOWN" and element.max_utilization is None
    assert element.utilization_lower_bound == 0.0 and element.utilization_upper_bound > 100.0
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN" and "UNTRACED_LIGHT_BOUNDED" in target.issue_codes  # rv2 t02b: was PASS with 0.0
    assert run.result.status.value != "PASS"


def test_a_rated_element_whose_untraced_light_is_bounded_within_the_rating_passes():
    run = run_optical_project(_mirror_window("0.999", "0.001"))
    (comparison,) = _surface(run, "W/S1").ratings
    assert comparison.status == "WITHIN_RATING" and comparison.exposure_value is None
    assert 0.0 < comparison.exposure_upper_bound <= 1.0
    element = _element(run, "W")
    assert element.rating_status == "WITHIN_RATING" and element.max_utilization is None
    assert 0.0 < element.utilization_upper_bound <= 1.0
    (target,) = run.result.targets
    # (only untraced light reaches it: graded on [0, upper] with no value, never 0 -- as a power target on a plane no traced
    # beam reaches)
    assert target.status.value == "PASS" and target.value is None and "the utilization is between" in target.message
    tight = run_optical_project(_mirror_window("0.999", "0.001", maximum=0.5 * element.utilization_upper_bound))
    assert tight.result.targets[0].status.value == "UNKNOWN"  # met at the lower end, not at the upper


def test_a_lit_rated_window_is_bounded_by_its_own_ghosts():
    """The window's own ghost reflections (its rear face's reflection back onto its front face, ...) reach the surfaces its
    traced beam meets: they bound the comparison (no lit-surface exemption). Coherent with the traced beam
    from the other side of S1, they form a standing wave inside the glass: (sqrt I -+ sum sqrt I_g)^2 (``_traced``)."""
    power, w = _independent()["W/S1"]
    run = run_optical_project(_document([_rating("CW", 4.0 * _peak_w_per_cm2(power, w), "W/cm2")]))
    s1 = _surface(run, "W/S1")
    (comparison,) = s1.ratings
    assert comparison.status == "WITHIN_RATING"
    assert _traced(run, comparison, s1) == pytest.approx(_peak_w_per_cm2(power, w), rel=1e-9)
    assert run.result.exposures.untraced_light_included is True and s1.untraced_hits != ()
    (group,) = s1.coherent_groups
    assert group.incidence == "BOTH_SIDES" and group.members[0] == "traced:SRC#0:0"


def _folded_partial_mirror(observation: bool) -> dict:
    """M1 (R 0.01 / T 0.99 at 45 deg) folds 1 % of a 1 W 780 nm beam to M2 (R 1), which folds it back
    along +X onto a window W (CW rating 10 W/cm2, fixture) -- or an observation plane at its place (power_W maximum 0.1)
    -- that the 99 % the coating transmits (light the trace does not follow) reaches too."""
    def mirror(model_id: str, reflectance: str, transmittance: str) -> tuple[dict, dict]:
        built = json_api.build_native_model("mirror.flat", {"reflectance": "0.9", "valid_band_nm": ["400", "1100"], "diameter_mm": "25.4"})
        model, (coat,) = copy.deepcopy(built["optical_model"]), copy.deepcopy(built["model_coatings"])
        coat["coating_id"] = f"C_{model_id.upper()}"
        coat["bands"][0]["reflectance"], coat["bands"][0]["transmittance"] = reflectance, transmittance
        model["model_id"] = model_id
        model["model"]["surfaces"][0]["coating_id"] = coat["coating_id"]
        return model, coat

    m1, c1 = mirror("m1", "0.01", "0.99")
    m2, c2 = mirror("m2", "1", "0")
    components = [source(power_w=1.0, waist_mm=0.5),
                  declared_component("M1", "mirror", "m1", quaternion=_quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(-INV, INV, 0.0))),
                  declared_component("M2", "mirror", "m2", position=(0.0, 50.0, 0.0),
                                     quaternion=_quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(INV, -INV, 0.0)))]
    if observation:
        targets = [{"target_id": "p", "metric": "power_W", "observation_id": "DET", "maximum": 0.1}]
        return with_models(project(components, [plane("DET", (100.0, 25.0, 0.0), radius_mm=40.0)], targets), [m1, m2], None, [c1, c2])
    # (no test beam or polarization stated: stating them makes the run follow the left-out light deeper to check them, and
    # the exposure's bounds this scene's test asserts are those of the run that does not)
    rating = {"regime": "CW", "value": "10", "unit": "W/cm2", "wavelength_nm": "780", "status": "DECLARED", "basis": FIXTURE,
              "value_definition": "PEAK"}
    win = {"model_id": "win", "model": sequence([surface(1, "0", clear="40"), surface(2, "3", clear="40")], ["N-BK7"], edge="45"),
           "power_ratings": [rating]}
    components.append(declared_component("W", "window", "win", position=(100.0, 25.0, 0.0)))
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    return with_models(project(components, [], targets), [m1, m2, win], [nbk7()], [c1, c2])


def test_a_weak_traced_beam_does_not_stand_for_a_strong_untraced_branch_of_its_line_on_a_rated_surface():
    run = run_optical_project(_folded_partial_mirror(observation=False))
    s1 = _surface(run, "W/S1")
    assert s1.incident_power_W == pytest.approx(0.01, rel=1e-9)  # the traced 1 %
    assert max(h.incident_power_W for h in s1.untraced_hits) == pytest.approx(0.99, rel=1e-9)  # the coating's 99 %
    element = _element(run, "W")
    # the traced 1 % (utilization 0.227) less its own reflections inside the window, coherent with it at its centre (the
    # coating's 99 % meets the window 50 mm away: nothing of it is there), and less the light the run leaves out as
    # negligible (``faint_amplitude``: its diffraction limit, which may be anywhere). Its own reflections are the 99 %
    # beam's scaled by 1 / 99 (one flat window, the same Fresnel factors).
    (traced,) = s1.hits
    powers = [h.incident_power_W for h in s1.untraced_hits]
    own_hits = [h for h in s1.untraced_hits if any(abs(99.0 * h.incident_power_W - q) <= 1e-6 * q for q in powers)]
    assert len(own_hits) == 3
    own = sum(math.sqrt(h.peak_irradiance_W_per_cm2) for h in own_hits)
    (comparison,) = s1.ratings
    assert traced.peak_irradiance_W_per_cm2 / comparison.rating_value == pytest.approx(0.227, rel=0.01)
    # (its open paths: the window's two bare faces, and M1 and M2, met once per leg, may each reflect it)
    gap = math.sqrt(traced.peak_irradiance_W_per_cm2) - own - faint_amplitude(run, "W/cm2", surface=s1,
                                                                              paths=open_paths(1.0, 1.0, per_leg=(0.01, 1.0)))
    least = (gap * gap if gap > 0.0 else 0.0) / comparison.rating_value
    # (from below also the window bound: the traced 1 % against a window over its area, which the light left out as
    # negligible, of unknown place, cannot undo -- the bound at the centre alone gives none)
    lower = (comparison.exposure_lower_bound or 0.0) / comparison.rating_value
    assert least == 0.0 < lower < traced.peak_irradiance_W_per_cm2 / comparison.rating_value
    # (the rating states no test beam or polarization: not verified, so the element's utilization has no bounds -- and with
    # them stated, light of unknown size reaching S1 with just over 1 % of the most light would still leave its test beam
    # not verifiable there; the exposure's bounds stand)
    assert element.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert (element.utilization_lower_bound, element.utilization_upper_bound) == (None, None)
    assert comparison.exposure_upper_bound / comparison.rating_value > 20.0
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN"  # was PASS (WITHIN_RATING, 0.227)


def test_a_weak_traced_beam_does_not_stand_for_a_strong_untraced_branch_of_its_line_on_an_observation_plane():
    run = run_optical_project(_folded_partial_mirror(observation=True))
    (target,) = run.result.targets
    assert target.value == pytest.approx(0.01, rel=1e-9)  # the traced light; the minimum side would read it
    assert target.status.value == "UNKNOWN" and "UNTRACED_TRANSMISSION" in target.message  # was PASS
    assert "UNTRACED_LIGHT_BOUNDED" in target.issue_codes and "TARGET_NOT_MET" not in target.issue_codes
    (issue,) = [i for i in run.result.issues if i.code.value == "INCOMPLETE_TRACE"]  # it names the undecided target
    assert (issue.location, issue.status.value, issue.affected_targets) == ("target:p", "UNKNOWN", ("p",))
    assert issue.numerical_context["observation_id"] == "DET"
    assert issue.numerical_context["untraced_power_W"] == pytest.approx(0.99, rel=1e-6)
    relaxed = _folded_partial_mirror(observation=True)
    relaxed["targets"][0]["maximum"] = 1.5  # above traced + untraced: decided
    run = run_optical_project(relaxed)
    assert run.result.targets[0].status.value == "PASS" and not any(i.code.value == "INCOMPLETE_TRACE" for i in run.result.issues)


def test_without_ratings_untraced_light_is_not_followed_to_the_surfaces():
    document = _mirror_window("0.5", "0.5")
    document["optical_models"] = [{k: v for k, v in m.items() if k != "power_ratings"} for m in document["optical_models"]]
    document["targets"] = []
    run = run_optical_project(document)
    assert run.result.exposures.untraced_light_included is False
    assert all(s.untraced_hits == () for s in run.result.exposures.surfaces)


# ------------------------------------------------------------------------------------------- M8: partial ratings


def test_ratings_of_some_surfaces_are_reported_per_surface_and_the_element_aggregates_its_rated_surfaces():
    """A rating restricted to some surfaces (a compound unit's member ratings apply to that member's surfaces): here the
    window's rating names its S1 only (Engine ``PowerRating.surface_ids``)."""
    power, w = _independent()["W/S1"]
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    request = lower_project(_document([_rating("CW", 4.0 * _peak_w_per_cm2(power, w), "W/cm2")], targets=targets)).request
    components = tuple(c.model_copy(update={"power_ratings": tuple(r.model_copy(update={"surface_ids": ("S1",)}) for r in c.power_ratings)})
                       if c.instance_id == "W" else c for c in request.project.components)
    result = validate(ValidationRequest.model_validate_json(
        request.model_copy(update={"project": request.project.model_copy(update={"components": components})}).model_dump_json()))
    surfaces = {s.surface_id: s for s in result.exposures.surfaces}
    assert surfaces["W/S1"].rating_status == "WITHIN_RATING"
    assert surfaces["W/S2"].rating_status == "RATING_MISSING" and surfaces["W/S2"].ratings == ()
    (element,) = [e for e in result.exposures.elements if e.element_id == "W"]
    assert element.rating_status == "WITHIN_RATING" and element.unrated_surface_ids == ("W/S2",)
    # S1 alone decides: the traced beam's utilization is 0.25, bounded by the window's own ghosts on S1
    (comparison,) = surfaces["W/S1"].ratings
    assert _traced(result, comparison, surfaces["W/S1"]) / comparison.rating_value == pytest.approx(0.25, rel=1e-9)
    least = comparison.exposure_lower_bound / comparison.rating_value
    assert element.worst_surface_id == "W/S1" and element.utilization_lower_bound == least
    assert element.utilization_upper_bound == comparison.exposure_upper_bound / comparison.rating_value < 1.0
    (target,) = result.targets
    assert target.status.value == "PASS" and target.value == least  # was UNKNOWN "has no power rating"


# ------------------------------------------------------------------------ B2: tabulated thin lens and the exposure


def _focused_window_mm() -> float:
    """The mean of the 1/e^2 diameters on the two faces of ``_lens_focus_on_window``'s window by Gaussian q propagation:
    200 mm of air, the thin lens (f 100 mm), 98 mm of air to the front face, 3 mm of N-BK7 to the rear face (q' = n q + t).
    The beams on both faces (the front face's own and the rear face's reflection onto it) lie within 5 % of it."""
    q = complex(200.0, rayleigh_range(0.5, 532.0))
    q_front = 1.0 / (1.0 / q - 1.0 / 100.0) + 98.0
    n = _n("N-BK7", 532.0)
    q_rear = n * q_front + 3.0
    front = 2.0 * math.sqrt(532.0 * NM / math.pi * abs(q_front) ** 2 / q_front.imag)
    rear = 2.0 * math.sqrt(532.0 * NM / (math.pi * n) * abs(q_rear) ** 2 / q_rear.imag)
    return 0.5 * (front + rear)


def _lens_focus_on_window(metrics: list[str]) -> dict:
    """rv2 t06b: a tabulated thin lens (f 100 mm, T 1) focuses 1 W at 532 nm; a rated window W sits 2 mm before the focus."""
    params = {"rows": [{"wavelength_nm": "520", "focal_length_mm": "100", "transmittance": "1"},
                       {"wavelength_nm": "540", "focal_length_mm": "100", "transmittance": "1"}],
              "clear_aperture_diameter_mm": "20", "diameter_mm": "25.4", "max_field_angle_deg": "5",
              "supported_metrics": [{"metric": m, "rel_error": "0.05", "basis": "fixture"} for m in metrics],
              "evidence": copy.deepcopy(EVIDENCE)}
    lens = build_native_model("lens.tabulated_thin", params)
    rating = _stated({"regime": "CW", "value": "1000000", "unit": "W/cm2", "wavelength_nm": "532", "value_definition": "PEAK",
                      "status": "DECLARED", "basis": FIXTURE}, _focused_window_mm())
    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]), "power_ratings": [rating]}
    document = project([source(position=(-200.0, 0.0, 0.0), waist_mm=0.5, wavelength_nm=532.0, power_w=1.0),
                        declared_component("W", "window", "win", position=(98.0, 0.0, 0.0))],
                       [plane("det", (150.0, 0.0, 0.0), radius_mm=20.0)],
                       [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0},
                        {"target_id": "pmin", "metric": "power_W", "observation_id": "det", "minimum": 0.5}])
    document = with_models(document, [win], [nbk7()])
    return add_to_project(document, lens, instance_id="ML", position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


def test_a_utilization_behind_a_tabulated_lens_not_validated_for_beam_size_is_out_of_model_scope():
    run = run_optical_project(_lens_focus_on_window(["power_W"]))
    targets = {t.target_id: t for t in run.result.targets}
    util = targets["util"]
    assert (util.status.value, util.value) == ("OUT_OF_MODEL_SCOPE", None)  # rv2 t06b: was PASS 0.0531
    assert "METRIC_NOT_VALIDATED_FOR_ELEMENT" in util.issue_codes and "ML/lens" in util.message
    assert targets["pmin"].status.value == "PASS"  # the power follows its validated transmittance
    (hit,) = _surface(run, "W/S1").hits
    assert hit.status.value == "OUT_OF_MODEL_SCOPE" and hit.reason == "METRIC_NOT_VALIDATED_FOR_ELEMENT"
    assert hit.beam_radius_x_mm is None and hit.peak_irradiance_W_per_cm2 is None and hit.incident_power_W > 0.9


@pytest.mark.parametrize("metrics", [["power_W", "beam_radius_mm"], ["power_W", "beam_radius_x_mm", "beam_radius_y_mm"],
                                     ["power_rating_utilization"]], ids=["round", "both-axes", "utilization"])
def test_a_tabulated_lens_validated_for_the_beam_size_keeps_the_utilization_in_scope(metrics):
    run = run_optical_project(_lens_focus_on_window(metrics))
    util = next(t for t in run.result.targets if t.target_id == "util")
    (hit,) = _surface(run, "W/S1").hits
    assert hit.status.value == "PASS" and hit.beam_radius_x_mm == pytest.approx(0.036, rel=0.02)
    # S2 (traced utilization 0.0531): the window's own double reflection S2 -> S1 -> S2 is coherent with the traced beam,
    # so S2's exposure is a range around its traced value; S1's likewise with the reflections arriving from inside the glass
    # (a standing wave, ``_traced``); the element's lower bound is the larger of the two lower bounds
    (s2_hit,) = _surface(run, "W/S2").hits
    (s2,) = _surface(run, "W/S2").ratings
    traced_s2 = s2_hit.peak_irradiance_W_per_cm2 / s2.rating_value
    assert traced_s2 == pytest.approx(0.0531, rel=0.02)
    assert s2.exposure_lower_bound / s2.rating_value < traced_s2 < s2.exposure_upper_bound / s2.rating_value
    (s1,) = _surface(run, "W/S1").ratings
    assert _traced(run, s1, _surface(run, "W/S1"), reference=False) == hit.peak_irradiance_W_per_cm2
    least = max(s1.exposure_lower_bound / s1.rating_value, s2.exposure_lower_bound / s2.rating_value)
    assert util.status.value == "PASS" and util.value == least


# ------------------------------------------------------------------------- M2: ND variation without a Gaussian state


def _gradient_nd_document(with_axicon: bool) -> dict:
    """rv2 t03b: a gradient ND (OD 0 -> 4 over x -2..2 mm) across a beam of w 1.5 mm, optionally behind a 1 deg axicon."""
    nd = build_native_model("filter.variable_nd", {"law": "LINEAR_GRADIENT", "od_start": "0", "od_end": "4", "x_start_mm": "-2",
                                                   "x_end_mm": "2", "width_mm": "40", "height_mm": "20", "valid_band_nm": [400, 1100],
                                                   "uncoated_transmittance": "1"})
    document = project([source(position=(-100.0, 0.0, 0.0), waist_mm=1.5, power_w=1.0)], [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)],
                       [{"target_id": "p", "metric": "power_W", "observation_id": "det", "maximum": 0.02, "unit": "W"}])
    if with_axicon:
        axicon = build_native_model("lens.axicon", {"cone_angle_deg": "1", "material": "N-BK7", "diameter_mm": "25.4",
                                                    "design_wavelength_nm": "780"})
        document = add_to_project(document, axicon, instance_id="AX", position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)
    return add_to_project(document, nd, instance_id="ND", position_mm=(20.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


@pytest.mark.parametrize("with_axicon", [False, True], ids=["gaussian-footprint", "pupil-rays-behind-an-axicon"])
def test_a_power_target_through_an_attenuator_whose_density_varies_across_the_beam_does_not_pass(with_axicon):
    run = run_optical_project(_gradient_nd_document(with_axicon))
    (issue,) = [i for i in run.result.issues if i.code.value == "ND_DENSITY_VARIES_ACROSS_BEAM"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.numerical_context["od_variation"] > 0.2
    assert ("pupil rays" in issue.message) == with_axicon
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.power_out_of_scope_at == ("ND/attenuator",)
    (target,) = run.result.targets
    assert (target.status.value, target.value) == ("OUT_OF_MODEL_SCOPE", None)  # rv2 t03b: PASS with the chief value
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" in target.issue_codes and "TARGET_NOT_MET" not in target.issue_codes  # (R5)


def test_a_rating_behind_an_attenuator_whose_density_varies_across_the_beam_is_out_of_model_scope():
    """The gradient ND (OD 0 -> 4 over x -2..2 mm) across the 1.5 mm beam, a rated window 20 mm behind it (CW, 0.5 W/cm2). The
    beam's power follows the chief ray's transmittance (OD 2: 0.01 of the light), so its peak irradiance on the window, 0.28
    W/cm2, is within the rating; the light that crosses the thin side of the filter peaks at 0.044 of the unfiltered beam's
    (max over x of 10^-OD(x) exp(-2 x^2 / w^2)), 1.26 W/cm2, above it. The power_rating_utilization target used to PASS on
    the chief ray's power; the window's hits now carry no exposure (OUT_OF_MODEL_SCOPE) and the target is OUT_OF_MODEL_SCOPE,
    naming the attenuator."""

    document = _gradient_nd_document(False)
    rating = _stated({"regime": "CW", "value": "0.5", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
                      "status": "DECLARED", "basis": FIXTURE}, _beam_mm(140.0, 1.5))
    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]), "power_ratings": [rating]}
    document["components"].append(declared_component("W", "window", "win", position=(40.0, 0.0, 0.0)))
    document = with_models(document, [*document.get("optical_models", []), win], [*document.get("model_materials", []), nbk7()],
                           document.get("model_coatings"))
    document["targets"] = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    w = _beam_mm(140.0, 1.5) / 20.0  # (cm: the 1/e^2 radius at the window)
    peak = 2.0 / (math.pi * w * w)
    x = np.linspace(-6.0, 6.0, 120001)
    transmitted = 10.0 ** -np.clip(x + 2.0, 0.0, 4.0) * np.exp(-2.0 * x * x / (10.0 * w) ** 2)
    assert 0.01 * peak < 0.5 < float(transmitted.max()) * peak
    run = run_optical_project(document)
    util = next(t for t in run.result.targets if t.target_id == "util")
    assert (util.status.value, util.value) == ("OUT_OF_MODEL_SCOPE", None)
    assert "OUT_OF_MODEL_SCOPE" in util.issue_codes and "ND/attenuator" in util.message
    for hit in _surface(run, "W/S1").hits:
        assert hit.status.value == "OUT_OF_MODEL_SCOPE" and hit.peak_irradiance_W_per_cm2 is None


# --------------------------------------------------------------------------------- M3: the attenuator's energy check


def test_a_low_level_attenuator_refuses_reflectance_plus_transmittance_above_one():
    with pytest.raises(ValueError, match="reflectance \\+ transmittance cannot exceed 1"):
        NeutralDensityInteraction(law={"kind": "linear_gradient", "od_start": 0.0, "od_end": 1.0, "x_start_mm": -10, "x_end_mm": 10},
                                  reflectance=0.95)
    NeutralDensityInteraction(law={"kind": "fixed", "optical_density": 2.0}, reflectance=0.5)  # 0.5 + 0.01: accepted
    NeutralDensityInteraction(law={"kind": "fixed", "optical_density": 1.0}, reflectance=0.9)  # 0.9 + 0.1 = 1: accepted
    with pytest.raises(ValueError, match="cannot exceed 1"):
        NeutralDensityInteraction(law={"kind": "stepped_rotary", "steps": [{"start_deg": 0, "end_deg": 90, "optical_density": 3}]},
                                  reflectance=0.2, uncoated_transmittance=0.9)


def test_a_negative_sink_breaks_the_ledger_even_when_the_sums_agree():
    trace = TraceOutput(inputs=[SourceInput("S", "S", 0, 780.0, 1.0)],
                        power_flows=[PowerFlow("ESCAPED", 1.128, "S", 0, 780.0, "S#0", None, None, "ESCAPED"),
                                     PowerFlow("ATTENUATOR_ABSORPTION", -0.128, "S", 0, 780.0, "S#0", "ND/attenuator", "ND", "neutral_density")])
    budget, issues = build_power_budget(trace, [], NumericalConfig())
    assert budget.lines[0].conserved is False and budget.conserved is False
    (issue,) = issues
    assert issue.code.value == "POWER_BALANCE_RESIDUAL" and issue.numerical_context["negative_sinks"] == ["ATTENUATOR_ABSORPTION"]


# ------------------------------------------------------------------------------------- M1: one ellipticity sign


@pytest.mark.parametrize("chi", [-45.0, -20.0, 0.0, 20.0, 45.0])
def test_a_source_ellipticity_arrives_unchanged_through_free_space(chi):
    laser = source("SRC", position=(-100.0, 0.0, 0.0), waist_mm=0.5)
    laser["source_model"]["polarization"] = {"kind": "ELLIPTICAL", "orientation_deg": 0.0, "ellipticity_deg": chi}
    targets = [{"target_id": "ell", "metric": "polarization_ellipticity_deg", "observation_id": "det", "target_value": chi,
                "tolerance_abs": 1e-9}]
    run = run_optical_project(project([laser], [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)], targets))
    state = run.result.observations[0].arrivals[0].polarization_state
    assert state.ellipticity_deg == pytest.approx(chi, abs=1e-9)
    assert state.handedness == ("RIGHT" if chi < 0 else "LEFT" if chi > 0 else "LINEAR")
    assert run.result.targets[0].status.value == "PASS"  # rv2 t01: FAILed with a 2 chi error


# --------------------------------------------------------------------------------------- M9: exponent notation


def test_an_exponent_in_a_rating_is_read_as_the_number_it_spells():
    plain = _rating("PULSED", 5.0, "J/cm2", pulse_duration_s=0.00000001)
    rating = copy.deepcopy(plain)
    rating["pulse_duration_s"] = "1e-8"
    checked, expected = json_api.check_project(_document([rating])), json_api.check_project(_document([plain]))
    assert checked["ok"] and checked["project"] == expected["project"] and checked["project_hash"] == expected["project_hash"]
    assert run_optical_project(_document([rating])).result.model_dump(mode="json") == \
        run_optical_project(_document([plain])).result.model_dump(mode="json")


# --------------------------------------------------------------------------------- N8: a shared band boundary


def test_a_shared_band_boundary_takes_the_smaller_transmittance_and_the_larger_leak():
    interaction = InteractionG(kind="polarizer", t_par=0.8, t_perp=0.0008,
                               bands=((400.0, 700.0, 0.8, 0.0008), (700.0, 1100.0, 0.9, 0.009)))
    assert interaction.transmittances(700.0) == (0.8, 0.009)
    assert interaction.transmittances(600.0) == (0.8, 0.0008) and interaction.transmittances(800.0) == (0.9, 0.009)


# ------------------------------------------------------------------------------ N1: libm-free run-file arithmetic


def test_the_libm_free_functions_match_libm_to_a_few_ulp_and_are_exact_where_libm_is():
    rng = np.random.default_rng(7)
    y, x = rng.normal(size=4000), rng.normal(size=4000)
    assert np.max(np.abs(atan2_deg(y, x) - np.degrees(np.arctan2(y, x)))) < 1e-12
    assert list(atan2_deg(np.array([0.0, 1.0, 0.0, -1.0, 1.0]), np.array([1.0, 0.0, -1.0, 0.0, 1.0]))) == [0.0, 90.0, 180.0, -90.0, 45.0]
    od = rng.uniform(0.0, 6.0, 4000)
    assert np.max(np.abs(pow10(-od) / np.power(10.0, -od) - 1.0)) < 4e-15
    assert [float(pow10(-k)) for k in range(6)] == [10.0 ** -k for k in range(6)]  # integer densities exactly
    v = rng.uniform(-30.0, 30.0, 4000)
    assert np.max(np.abs(exp(v) / np.exp(v) - 1.0)) < 4e-15
    assert max(abs(log10(t) - math.log10(t)) for t in rng.uniform(1e-6, 1.0, 400)) < 1e-15


def test_the_lapack_free_singular_values_and_inverse_match_numpy():
    rng = np.random.default_rng(11)
    for shape in ((2, 2), (3, 5), (6, 4), (1, 3)):
        a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ours, theirs = singular_values(a), np.linalg.svd(a, compute_uv=False)
        assert ours.shape == theirs.shape and np.max(np.abs(ours - theirs)) <= 1e-14 * theirs[0]
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    assert np.allclose(inverse_2x2(a), np.linalg.inv(a), atol=1e-14) and inverse_2x2(np.zeros((2, 2))) is None


def test_the_exposure_exports_list_untraced_light_and_the_upper_bound(tmp_path):
    from os_optics.results import load_run

    run_optical_project(_mirror_window("0.5", "0.5"), output_dir=str(tmp_path / "run"))
    results = load_run(tmp_path / "run")
    rows, _ = results.table("exposures", element="W")
    assert rows and {r["light"] for r in rows} == {"untraced"} and rows[0]["incident_power_W"] == pytest.approx(0.5, rel=1e-9)
    ratings, _ = results.table("power-ratings", element="W")
    assert ratings[0]["status"] == "UNKNOWN" and ratings[0]["exposure_upper_bound"] > 100.0


# ------------------------------------------------------------------------------------- further exposure cases


def _window_beside_beam(offset_mm: float) -> dict:
    """1 W, 780 nm, w = 2 mm along +X; a window W (clear radius 9, edge 10, CW rating 1 W/cm2, fixture)
    centred ``offset_mm`` off the beam axis, so its chief ray misses W while the beam's tail falls on it."""
    rating = _stated({"regime": "CW", "value": "1", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
                      "status": "DECLARED", "basis": FIXTURE}, _beam_mm(100.0, waist_mm=2.0))
    win = {"model_id": "win", "model": sequence([surface(1, "0", clear="9"), surface(2, "3", clear="9")], ["N-BK7"], edge="10"),
           "power_ratings": [rating]}
    return with_models(project([source(power_w=1.0, waist_mm=2.0, position=(-100.0, 0.0, 0.0)),
                                declared_component("W", "window", "win", position=(0.0, offset_mm, 0.0))], [],
                               [{"target_id": "u", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]),
                       [win], [nbk7()])


def test_a_beam_tail_on_a_rated_element_its_chief_ray_misses_is_never_a_known_zero():
    """R6: the chief ray passes 0.5 mm beside W's 10 mm outline (1.5 mm beside its clear aperture): the beam's Gaussian tail
    falls on it. The exposure is bounded by the profile at the support's nearest point (0.5 mm: 2 P / (pi w^2)
    exp(-2 0.5^2 / w^2) = 14.04 W/cm2, hand value), never a known 0 -- the rating of 1 W/cm2 is not known to hold."""
    run = run_optical_project(_window_beside_beam(10.5))
    s1 = _surface(run, "W/S1")
    assert s1.hits == () and not s1.complete and "INCOMPLETE_ENVELOPE" in s1.reason_codes
    (beside,) = s1.beside_hits
    w = beside.beam_radius_x_mm
    assert beside.miss_distance_mm == pytest.approx(0.5, abs=1e-9) and w == pytest.approx(2.0, rel=1e-3)
    hand = 2.0 * 1.0 / (math.pi * w * w) * math.exp(-2.0 * 0.5 ** 2 / (w * w)) * 100.0
    assert beside.peak_irradiance_W_per_cm2 == pytest.approx(hand, rel=1e-12) and beside.linear_power_density_W_per_cm is None
    (comparison,) = s1.ratings
    assert comparison.status == "UNKNOWN" and comparison.exposure_upper_bound == pytest.approx(hand, rel=1e-12)
    element = _element(run, "W")
    assert element.rating_status == "UNKNOWN" and element.utilization_lower_bound == 0.0
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN" and "INCOMPLETE_ENVELOPE" in target.issue_codes  # was PASS 0.0
    far = run_optical_project(_window_beside_beam(30.0))  # 20 mm = 10 beam radii beside: the tail is negligible
    assert _surface(far, "W/S1").beside_hits == () if any(s.surface_id == "W/S1" for s in far.result.exposures.surfaces) else True
    assert far.result.targets[0].status.value == "PASS" and far.result.targets[0].value == 0.0


def test_a_pass_bounded_by_the_element_s_own_ghosts_names_no_out_of_model_scope():
    """R7: the window's own ghosts bound its comparison; the PASS names UNTRACED_LIGHT_BOUNDED, not OUT_OF_MODEL_SCOPE."""
    power, w = _independent()["W/S1"]
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document([_rating("CW", 4.0 * _peak_w_per_cm2(power, w), "W/cm2")], targets=targets))
    (target,) = run.result.targets
    # (COHERENT_BEAMS_BOUNDED: on the rear face the window's own double reflection is coherent with the traced beam;
    # FAINT_LIGHT_BOUNDED: the light the run leaves out as negligible, bounded by its diffraction limit;
    # FAINT_LIGHT_RECIRCULATES: and by its returns between the window's two faces)
    assert target.status.value == "PASS" and target.issue_codes == ("UNTRACED_LIGHT_BOUNDED", "COHERENT_BEAMS_BOUNDED",
                                                                    "FAINT_LIGHT_BOUNDED", "FAINT_LIGHT_RECIRCULATES")


def test_a_dark_plane_only_followed_untraced_light_reaches_is_graded_on_the_interval_of_that_light():
    """R8: a plane no traced beam reaches, where only the window's untraced reflections (followed by the probes) may arrive.
    The power has no value (never a 0 W result while light may arrive) and lies between 0 and that light's bound: a maximum
    above all of that light is met, one within it is not decided, a minimum above it is not met (targets only; the plane's issue and state stay UNKNOWN)."""
    from os_optics.validation.pipeline import run_physics

    run = run_optical_project(_mirror_window("0.5", "0.5"))  # (W: reached only by the coating's transmission)
    document = _mirror_window("0.5", "0.5")
    document["optical_models"] = [{k: v for k, v in m.items() if k != "power_ratings"} for m in document["optical_models"]]
    document["observations"].append(plane("BEHIND", (120.0, 0.0, 0.0), radius_mm=20.0))
    document["targets"] = [{"target_id": "probe", "metric": "power_W", "observation_id": "BEHIND", "maximum": 10.0}]
    lowered = lower_project(document)
    reach = run_physics(lowered.request.project, lowered.request.configuration.numerical).reach
    from os_optics.validation.untraced import hazards

    (line,) = reach["BEHIND"]
    untraced = sum(hazards(reach, "BEHIND", [line]).values())  # (the kinds of light, not the amplitude sums beside them)
    assert 0.4 < untraced < 0.5
    graded = (({"maximum": untraced * 1.001}, "PASS"), ({"maximum": untraced * 0.5}, "UNKNOWN"), ({"minimum": untraced * 1.5}, "FAIL"))
    for limits, status in graded:
        document["targets"] = [{"target_id": "p", "metric": "power_W", "observation_id": "BEHIND", **limits}]
        result = run_optical_project(document).result
        (target,) = result.targets
        assert (target.status.value, target.value) == (status, None), limits
        assert "UNTRACED_LIGHT_BOUNDED" in target.issue_codes
        (issue,) = [i for i in result.issues if i.location == "observation:BEHIND"]
        # (a notice: the plane's BOUNDED interval from 0 and its followed_light carry that light)
        assert issue.status.value == "NOT_APPLICABLE"
        behind = next(o for o in result.observations if o.observation_id == "BEHIND")
        assert behind.quantity_state.value == "BOUNDED" and behind.power_lower_bound_W == 0.0 and behind.total_power_W is None
    assert run.result.targets[0].status.value == "UNKNOWN"


@pytest.mark.parametrize("gap_mm", [1.0, 1.4, 1.8])
def test_a_beam_without_a_gaussian_state_passing_near_a_rated_element_leaves_it_unknown(gap_mm):
    """Behind a 0.5 deg axicon the beam (w 1 mm) has no valid Gaussian state; a window rated 0.1 W/cm2
    whose outline is 1.0 / 1.4 / 1.8 mm from the axis gets light from the beam's tail (hand: about 11 / 1.9 / 0.17 W/cm2)
    whether or not a pupil ray lands on it. Its extent beyond the pupil rays is not known: UNKNOWN, never a known 0. Nor is
    its size, so the window's test beam cannot be checked against it: the rating is not verified (the reasons for the
    unknown exposure stay on the element and in the target's message). A window far beyond the pupil rays' reach stays a
    known zero."""
    # (a lossless AR coating on the axicon: no ghost reflections, whose probe traces carry no pupil rays and would make every
    # rated surface they cross UNKNOWN on their own -- the documented rule for a beam without pupil rays)
    axicon = build_native_model("lens.axicon", {"cone_angle_deg": "0.5", "material": "N-BK7", "diameter_mm": "25.4",
                                                "design_wavelength_nm": "780",
                                                "ar_coating": {"reflectance": "0", "valid_aoi_deg": [0, 10]}})

    def document(offset: float) -> dict:
        rating = _stated({"regime": "CW", "value": "0.1", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
                          "status": "DECLARED", "basis": FIXTURE}, _beam_mm(120.0, waist_mm=1.0))
        win = {"model_id": "win", "model": sequence([surface(1, "0", clear="9"), surface(2, "3", clear="9")], ["N-BK7"], edge="10"),
               "power_ratings": [rating]}
        doc = add_to_project(project([source(power_w=1.0, waist_mm=1.0, position=(-100.0, 0.0, 0.0))], [], []), axicon,
                             instance_id="AX", position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)
        # This synthetic axicon/window isolates the undefined Gaussian footprint, not
        # attenuation: assume ideal zero bulk loss here, never for a catalog glass.
        for material in doc["model_materials"]:
            if material["material_id"] == "N-BK7":
                material["absorption_coefficient_per_mm"] = "0"
        doc["optical_models"].append(win)
        doc["components"].append(declared_component("W", "window", "win", position=(20.0, offset, 0.0)))
        doc["targets"] = [{"target_id": "u", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
        return doc

    run = run_optical_project(document(10.0 + gap_mm))
    (target,) = run.result.targets
    assert (target.status.value, target.value) == ("UNKNOWN", None)  # 1.4 / 1.8 were PASS 0.0
    assert target.issue_codes == ("POWER_RATING_CONDITIONS_NOT_VERIFIED",)
    assert "GAUSSIAN_UNDEFINED_AFTER_AXICON" in target.message and "INCOMPLETE_ENVELOPE" in target.message
    element = _element(run, "W")
    assert {"GAUSSIAN_UNDEFINED_AFTER_AXICON", "INCOMPLETE_ENVELOPE"} <= set(element.reason_codes)
    assert element.rating_status == "CONDITIONS_NOT_VERIFIED" and element.utilization_upper_bound is None
    beams = {(c.state, c.reason) for s in run.result.exposures.surfaces if s.element_id == "W" for r in s.ratings for c in r.conditions
             if c.condition == "beam_diameter"}
    assert beams == {("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN")}
    far = run_optical_project(document(10.0 + 20.0))  # 20 mm beside: beyond 6 x the beam radius the pupil rays imply
    assert far.result.targets[0].status.value == "PASS" and far.result.targets[0].value == 0.0
