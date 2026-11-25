"""Where the beam meets an element in a way the single-packet trace does not follow (``os_optics.local.validity``):
``BEAM_PARTIALLY_INTERCEPTED``, ``RAY_LEFT_ELEMENT_THROUGH_EDGE``, ``RAY_CROSSES_ELEMENT_BODY`` -- and the Engine's
``GAUSSIAN_ABERRATION_DOMINATED`` seen through the local runtime. Each reproduces a finding of the v0.14 physics review
with the committed test catalog (reviewer's R1: a beam half over an element its chief ray misses, delivered at full
power; R2: an aberration-dominated focus passing on Gaussian metrics; R3: light leaving a tilted window through its side
wall), next to the geometries that must stay in scope. The Engine result is never changed by the local checks."""

from __future__ import annotations

import math

from os_optics.local import LocalValidationStatusV1, MODEL_SCOPE_CODES
from os_optics.local.validity import BODY_CROSSING_TOLERANCE_MM
from os_optics.validation.ray_paths import NOT_FOLLOWED

from local_helpers import (
    ALONG_X,
    ALONG_Y,
    AGAINST_X,
    PCX_F50,
    POWER_W,
    WINDOW,
    arrivals,
    element,
    plane,
    product_document,
    project,
    quat_multiply,
    run,
    scope_codes,
    source,
)

EDGE_MM = float(product_document(WINDOW)["simulation_models"][0]["edge_radius_mm"])
POWER_TARGET = {"target_id": "power", "metric": "power_W", "observation_id": "D", "minimum": 0.9 * POWER_W, "unit": "W"}


def _offset_window_run(offset_mm: float, waist_mm: float = 3.0):
    """A 780 nm beam along +X, a window (axis along the beam) whose centre sits ``offset_mm`` beside the beam axis."""

    document = project([source(waist_mm=waist_mm), element("W", "window", WINDOW, (0.0, offset_mm, 0.0), ALONG_X)],
                       [plane("D", (100.0, 0.0, 0.0), radius_mm=25.0)], [POWER_TARGET])
    return run(document)


def test_the_new_codes_follow_the_published_model_codes():
    assert MODEL_SCOPE_CODES[6:9] == ("BEAM_PARTIALLY_INTERCEPTED", "RAY_LEFT_ELEMENT_THROUGH_EDGE", "RAY_CROSSES_ELEMENT_BODY")
    assert BODY_CROSSING_TOLERANCE_MM == 1e-6


def test_a_beam_half_over_an_element_its_chief_ray_misses_is_out_of_model_scope():
    """Reviewer's R1: the chief ray passes 0.01 mm beside the window's edge, about half of a 3 mm beam goes through it.
    The Engine delivers the full power along the chief ray; part of the beam meets the window's face before the detector
    on the chief ray's leg, a share bounded by the cells of the beam's profile whose energy-flow lines may meet the face's
    support (0.474 of the beam by direct integration over the disc), and its back face, which the beam's tail may reach on the
    same leg, is counted too (the sum bounds the share from above, more loosely): the detector reads [P - S, P] and the 0.9 P
    minimum is not decided (it was once out of model scope). The local runtime's interception check keeps the run
    OUT_OF_MODEL_SCOPE."""

    result = _offset_window_run(EDGE_MM + 0.01)
    assert result.engine_status.value == "UNKNOWN"
    (arrival,) = arrivals(result.result, "D")
    assert arrival.beam.power_W == POWER_W and arrival.pupil_rays_arrived < arrival.pupil_ray_count
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("W/S1", "W/S2")
    assert 0.474 * POWER_W <= arrival.power_taken_before_upper_W <= POWER_W
    assert [(t.target_id, t.status.value, t.assessment.value) for t in result.result.targets] == [("power", "UNKNOWN", "INDETERMINATE")]
    assert scope_codes(result) == [("W", "BEAM_PARTIALLY_INTERCEPTED", "S1")]
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    (issue,) = result.model_scope_issues
    intercepted = [r for p in result.ray_paths.paths for r in p.pupil_rays if r.end == NOT_FOLLOWED and r.next_surface_id == "W/S1"]
    assert intercepted and all(r.counted_with_chief for r in intercepted)
    assert f"{len(intercepted)} of {arrival.pupil_ray_count} pupil rays of beam SRC#0 (780.0 nm)" in issue.message
    assert len(intercepted) + arrival.pupil_rays_arrived == arrival.pupil_ray_count
    assert 0.3 < len(intercepted) / arrival.pupil_ray_count < 0.6


def test_elements_fully_in_or_clear_of_the_beam_stay_in_scope():
    clear = _offset_window_run(EDGE_MM + 5.0)  # beyond the outer pupil ring (1.18 w = 3.5 mm)
    assert clear.status is LocalValidationStatusV1.PASS and clear.model_scope_issues == ()
    centred = _offset_window_run(0.0)
    # The window still admits the beam without a geometry scope issue. Its old
    # catalog glass does not state a bulk loss, so its delivered power is bounded,
    # not the former implicit lossless PASS for the power requirement.
    assert centred.status is LocalValidationStatusV1.UNKNOWN and centred.model_scope_issues == ()
    (arrival,) = arrivals(centred.result, "D")
    assert arrival.beam.power_W is None and 0 < arrival.power_upper_bound_W <= POWER_W
    assert arrival.beam.direction == (1.0, 0.0, 0.0)
    assert [(t.status.value, t.assessment.value) for t in centred.result.targets] == [("UNKNOWN", "INDETERMINATE")]


def test_pupil_rays_that_miss_the_chiefs_element_are_its_truncation_not_an_interception():
    """A beam overfilling a window's edge: the rays beyond the edge are already counted as lost by the Engine's aperture
    truncation at that window (``counted_with_chief`` False), so meeting a second element afterwards is not reported as
    an interception (the run is out of scope for the truncation itself)."""

    document = project(
        [source(waist_mm=1.5, position=(-200.0, 11.0, 0.0)),
         element("W1", "window", WINDOW, (0.0, 0.0, 0.0), ALONG_X),
         element("W2", "window", WINDOW, (40.0, 11.5 + EDGE_MM, 0.0), ALONG_X)],  # W2 starts 0.07 mm above W1's clear aperture
        [plane("D", (100.0, 11.0, 0.0), radius_mm=25.0)])
    result = run(document)
    bypass = [r for p in result.ray_paths.paths for r in p.pupil_rays if r.end == NOT_FOLLOWED and r.next_surface_id == "W2/S1"]
    assert bypass and not any(r.counted_with_chief for r in bypass)
    assert "BEAM_PARTIALLY_INTERCEPTED" not in [c for _, c, _ in scope_codes(result)]
    assert "GAUSSIAN_SEVERE_TRUNCATION" in {i.code.value for i in result.result.issues}


def _tilted_window_run(entry_face_mm: float):
    """Reviewer's R3 geometry on the test window: turned 70 deg about the vertical, a 633 nm beam entering its first face
    ``entry_face_mm`` from the centre (along the tilted face)."""

    half = math.radians(70.0) / 2.0
    tilted = quat_multiply(ALONG_X, (math.cos(half), 0.0, math.sin(half), 0.0))
    y0 = -entry_face_mm * math.cos(math.radians(70.0))
    document = project([source(wavelength_nm=633.0, waist_mm=0.2, position=(-100.0, y0, 0.0)), element("W", "window", WINDOW, (0.0, 0.0, 0.0), tilted)],
                       [plane("D", (100.0, 0.0, 0.0), radius_mm=200.0)],
                       [{"target_id": "power", "metric": "power_W", "observation_id": "D", "minimum": 0.5 * POWER_W, "unit": "W"}])
    return run(document)


def test_light_leaving_a_window_through_its_side_wall_is_out_of_model_scope():
    """Entering S1 inside its clear aperture, the refracted chief ray reaches the plane of S2 beyond the window's edge: it
    leaves through the side wall. The Engine warns (MEDIUM_MISMATCH at the detector, the beam still 'in' N-BK7), and the
    detector's power is out of model scope there (part of the beam meets S2 before it on the chief ray's leg
    m2); the local validation is OUT_OF_MODEL_SCOPE."""

    result = _tilted_window_run(10.6)
    assert result.engine_status.value == "OUT_OF_MODEL_SCOPE" and "MEDIUM_MISMATCH" in {i.code.value for i in result.result.issues}
    assert {t.target_id: t.status.value for t in result.result.targets} == {"power": "OUT_OF_MODEL_SCOPE"}
    assert arrivals(result.result, "D")[0].power_out_of_scope_at == ("W/S2",)
    codes = scope_codes(result)
    assert ("W", "RAY_LEFT_ELEMENT_THROUGH_EDGE", "S1") in codes
    assert ("W", "BEAM_PARTIALLY_INTERCEPTED", "S2") in codes  # the part of the beam that does reach S2
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    message = next(i.message for i in result.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE")
    assert "beam SRC#0 (633.0 nm) reaches observation/D" in message and "through its edge or a face that is not a traced surface" in message


def test_a_tilted_window_the_beam_crosses_face_to_face_stays_in_scope():
    result = _tilted_window_run(1.0)
    assert result.status is LocalValidationStatusV1.UNKNOWN and result.model_scope_issues == ()
    (arrival,) = arrivals(result.result, "D")
    assert arrival.beam.power_W is None and 0 < arrival.power_upper_bound_W <= POWER_W
    assert arrival.beam.optical_path_length_mm > 0 and arrival.beam.geometric_length_mm > 0
    assert [(t.status.value, t.assessment.value) for t in result.result.targets] == [("UNKNOWN", "INDETERMINATE")]


def test_a_beam_crossing_an_element_body_sideways_is_out_of_model_scope():
    """The window turned so its faces are parallel to the beam, which runs through the middle of its 3 mm body: no surface
    is met, the Engine traces air."""

    thickness = float(product_document(WINDOW)["simulation_models"][0]["surfaces"][1]["vertex_z_mm"])
    through = project([source(waist_mm=0.2), element("W", "window", WINDOW, (0.0, -thickness / 2.0, 0.0), ALONG_Y)], [plane("D", (100.0, 0.0, 0.0))])
    result = run(through)
    assert result.engine_status.value == "PASS"
    assert scope_codes(result) == [("W", "RAY_CROSSES_ELEMENT_BODY", None)]
    assert "through its edge or a face that is not a traced surface" in result.model_scope_issues[0].message
    beside = project([source(waist_mm=0.2), element("W", "window", WINDOW, (0.0, 1.0, 0.0), ALONG_Y)], [plane("D", (100.0, 0.0, 0.0))])
    assert run(beside).model_scope_issues == ()


def test_an_aberration_dominated_focus_is_out_of_model_scope_in_the_local_runtime():
    """Reviewer's R2 on the test catalog: the f = 50 mm plano-convex lens turned flat side first, a 4 mm collimated beam,
    the plane on the Gaussian waist. Gaussian metrics are withheld by the Engine;
    power is bounded because the fixture catalog leaves glass absorption unstated."""

    def document(quaternion, waist_mm, plane_x):
        return project(
            [source(waist_mm=waist_mm), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0), quaternion)],
            [plane("D", (plane_x, 0.0, 0.0), radius_mm=25.0)],
            [{"target_id": "spot", "metric": "beam_radius_mm", "observation_id": "D", "maximum": 0.05, "unit": "mm"},
             {"target_id": "power", "metric": "power_W", "observation_id": "D", "minimum": 0.5 * POWER_W, "unit": "W"}],
        )

    reversed_lens = run(document(AGAINST_X, 4.0, 60.0))
    assert "GAUSSIAN_ABERRATION_DOMINATED" in {i.code.value for i in reversed_lens.result.issues}
    targets = {t.target_id: t.status.value for t in reversed_lens.result.targets}
    assert targets == {"spot": "OUT_OF_MODEL_SCOPE", "power": "UNKNOWN"}
    (arrival,) = arrivals(reversed_lens.result, "D")
    assert arrival.beam.power_W is None and 0 < arrival.power_upper_bound_W <= POWER_W
    assert reversed_lens.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE and reversed_lens.model_scope_issues == ()
    # the lens the right way round with a 1 mm beam, read on its Gaussian waist: in scope
    probe = run(document(ALONG_X, 1.0, 60.0))
    (arrival,) = arrivals(probe.result, "D")
    waist_x = 60.0 - arrival.beam.gaussian.waist_distance_x_mm
    focused = run(document(ALONG_X, 1.0, waist_x))
    assert focused.status is LocalValidationStatusV1.UNKNOWN and focused.model_scope_issues == ()
    assert {t.target_id: t.status.value for t in focused.result.targets} == {"spot": "PASS", "power": "UNKNOWN"}
