"""Regressions of the independent physics review of the scientific result layer: mount outlines of a different shape than
the clear aperture, runtime-stable angles (polarization azimuth, image rotation), operand order of two crossings of one
path, bulk absorption at strong attenuation, partial capture by an absorbing detector, and fringe metrics of
non-overlapping beams."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.builders import laser, mirror_for_reflection, observation_plane, thin_lens, window
from os_optics.contracts import (
    CircularAperture,
    ConstantIndexMaterial,
    EngineeringComponentSnapshot,
    JonesVector,
    MirrorInteraction,
    PlaneSurface,
    Pose3D,
    ProjectSnapshot,
    RectangularAperture,
    Target,
)

from helpers import arrivals, one_arrival, run

X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def _line(result, source_id: str = "L"):
    (line,) = [ln for ln in result.power_budget.lines if ln.source_id == source_id]
    return line


def _plate_mirror(aperture, outline, position=(50.0, 0.0, 0.0)) -> EngineeringComponentSnapshot:
    """A plane mirror facing -X (it sends a +X beam back along -X) with an explicit mount outline around its clear aperture."""
    face = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), aperture=aperture, mount_outline=outline,
                        aperture_u_axis=(0.0, 1.0, 0.0), interaction=MirrorInteraction(reflective_side="positive"))
    return EngineeringComponentSnapshot(instance_id="M", pose=Pose3D(position_mm=position), surfaces=(face,))


def _beam_at(u: float, v: float):
    return laser("L", wavelength_nm=633.0, waist_radius_mm=0.05, position=(0.0, u, v), direction=X)


@pytest.mark.parametrize(("aperture", "outline", "hit", "expected"), [
    (CircularAperture(radius_mm=4.0), RectangularAperture(half_width_mm=5.0, half_height_mm=5.0), (4.5, 4.5), "CLIPPED"),
    (RectangularAperture(half_width_mm=2.0, half_height_mm=2.0), CircularAperture(radius_mm=20.0), (0.0, 10.0), "CLIPPED"),
    (CircularAperture(radius_mm=4.0), RectangularAperture(half_width_mm=10.0, half_height_mm=2.5), (0.0, 10.0), "ESCAPED"),
])
def test_a_mount_outline_of_another_shape_than_the_clear_aperture_bounds_the_surface(aperture, outline, hit, expected):
    behind = observation_plane("BEHIND", position=(100.0, hit[0], hit[1]), normal=X, radius_mm=1.0)
    result = run(ProjectSnapshot(components=(_beam_at(*hit), _plate_mirror(aperture, outline)), observations=(behind,)))
    (path,) = result.paths
    assert path.termination == expected
    assert bool(arrivals(result, "BEHIND")) == (expected == "ESCAPED")
    assert "GAUSSIAN_SEVERE_TRUNCATION" not in {i.code.value for i in result.issues}


def test_the_polarization_azimuth_does_not_depend_on_the_fold_direction():
    azimuths = []
    for turn in (Y, (0.0, -1.0, 0.0)):
        fold = mirror_for_reflection("F", position=(50.0, 0.0, 0.0), d_in=X, d_out=turn)
        plane = observation_plane("P", position=(50.0, 50.0 * turn[1], 0.0), normal=turn)
        a = one_arrival(run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, polarization=JonesVector.linear(90.0)), fold),
                                            observations=(plane,))), "P")
        azimuths.append(a.polarization_state.azimuth_in_plane_deg)
        axis = np.array(a.polarization_state.major_axis)
        assert axis[int(np.argmax(np.abs(axis)))] > 0.0
    assert azimuths[0] == pytest.approx(azimuths[1], abs=1e-9) and 0.0 <= azimuths[0] < 180.0


@pytest.mark.parametrize("roll_deg", [0.0, 17.0, 90.0, 133.0, 211.0, 305.0])
def test_an_inverted_image_is_plus_180_degrees_whatever_the_roll(roll_deg):
    up = (0.0, math.sin(math.radians(roll_deg)), math.cos(math.radians(roll_deg)))
    lens = thin_lens("LENS", focal_length_mm=50.0, clear_radius_mm=10.0, position=(100.0, 0.0, 0.0), up=up)
    image = observation_plane("IMG", position=(200.0, 0.0, 0.0), normal=X, up=up)
    a = one_arrival(run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, waist_radius_mm=0.2, up=up), lens),
                                        observations=(image,))), "IMG")
    assert a.first_order.image_rotation_deg == 180.0


def test_the_opd_of_two_crossings_of_one_path_follows_the_targets_crossing_order():
    m1 = mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=X, d_out=Y, clear_radius_mm=5.0)
    m2 = mirror_for_reflection("M2", position=(100.0, 20.0, 0.0), d_in=Y, d_out=(-1.0, 0.0, 0.0), clear_radius_mm=5.0)
    big = observation_plane("BIG", position=(50.0, 0.0, 0.0), normal=X)
    params = lambda a, b: {"path_id_a": "L#0", "path_id_b": "L#0", "crossing_index_a": float(a), "crossing_index_b": float(b)}  # noqa: E731
    targets = (Target(target_id="late_minus_early", metric="optical_path_difference_mm", observation_id="BIG", parameters=params(1, 0), maximum=1e9),
               Target(target_id="early_minus_late", metric="optical_path_difference_mm", observation_id="BIG", parameters=params(0, 1), maximum=1e9))
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=532.0), m1, m2), observations=(big,), targets=targets))
    values = {t.target_id: t.value for t in result.targets}
    first, second = sorted(arrivals(result, "BIG"), key=lambda a: a.crossing_index)
    expected = second.beam.optical_path_length_mm - first.beam.optical_path_length_mm
    assert expected > 0.0
    assert values["late_minus_early"] == pytest.approx(expected) and values["early_minus_late"] == pytest.approx(-expected)


@pytest.mark.parametrize("alpha_per_mm", [8.0, 20.0, 1e-3])
def test_bulk_absorption_is_booked_exactly_at_any_attenuation(alpha_per_mm):
    glass = ConstantIndexMaterial(material_id="dark", valid_wavelength_nm=(100.0, 20000.0), refractive_index=1.5,
                                  absorption_coefficient_per_mm=alpha_per_mm)
    block = window("W", thickness_mm=100.0, clear_radius_mm=10.0, material="dark", position=(100.0, 0.0, 0.0))
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0), block), materials=(glass,)))
    line = _line(result)
    r = 0.04
    expected = 1e-3 * (1 - r) * -math.expm1(-alpha_per_mm * 100.0)
    assert line.sinks["BULK_ABSORPTION"] == pytest.approx(expected, rel=1e-9) and line.conserved


def test_an_absorbing_detector_detects_only_the_power_on_its_active_area():
    det = observation_plane("DET", position=(100.0, 1.6, 0.0), normal=X, radius_mm=1.0, absorbing=True)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, waist_radius_mm=1.0),), observations=(det,)))
    a = one_arrival(result, "DET")
    frac = a.active_area_power_fraction
    line = _line(result)
    assert 0.0 < frac < 0.5
    assert line.sinks["DETECTED"] == pytest.approx(1e-3 * frac, rel=1e-6)
    assert line.sinks["OUTSIDE_ACTIVE_AREA"] == pytest.approx(1e-3 * (1.0 - frac), rel=1e-6) and line.conserved


def test_fringe_metrics_of_non_overlapping_beams_are_never_passed_on_a_pattern_that_does_not_form():
    """Two 50 um beams 3 mm apart on the plane (the golden 405 nm / 500 nm geometry with one mirror moved along its beam's
    transverse axis): no fringe forms. A target on the fringe period reports the period of the two wavevectors -- 500 nm,
    the geometry's design value, the directions being unchanged -- and is NOT_APPLICABLE, never PASS (the regression: a
    period target once passed here). The light quantities are compared on the values of that overlap: a visibility
    maximum is met, a visibility minimum is not."""
    from test_golden_interference import _interference_project

    from os_optics import NumericalConfig

    proj, *_ = _interference_project(waist_mm=0.05)
    comps = list(proj.components)
    mb = comps[3]
    comps[3] = mb.model_copy(update={"pose": Pose3D(position_mm=(mb.pose.position_mm[0] + 3.0, *mb.pose.position_mm[1:]),
                                                    quaternion_wxyz=mb.pose.quaternion_wxyz)})
    oid = proj.targets[0].observation_id if proj.targets else proj.observations[0].observation_id
    targets = (Target(target_id="period", metric="fringe_period_nm", observation_id=oid, minimum=0.0),
               Target(target_id="period_missed", metric="fringe_period_nm", observation_id=oid, maximum=100.0),
               Target(target_id="spacing", metric="fringe_spacing_3d_nm", observation_id=oid, minimum=0.0),
               Target(target_id="faint", metric="fringe_visibility", observation_id=oid, maximum=0.05),
               Target(target_id="strong", metric="fringe_visibility", observation_id=oid, minimum=0.5))
    result = run(proj.model_copy(update={"components": tuple(comps), "targets": targets}))
    (pair,) = result.interference
    limit = NumericalConfig().interference_min_overlap
    assert pair.spatial_overlap_ratio < limit and "BEAMS_NOT_OVERLAPPING" in pair.issue_codes
    by_id = {t.target_id: t for t in result.targets}
    for name in ("period", "period_missed", "spacing"):  # (whatever the bounds say: met or missed, the pattern is not there)
        outcome = by_id[name]
        assert (outcome.status.value, outcome.assessment.value) == ("NOT_APPLICABLE", "NOT_APPLICABLE"), name
        assert "BEAMS_NOT_OVERLAPPING" in outcome.issue_codes and "TARGET_NOT_MET" not in outcome.issue_codes
        (check,) = outcome.value_basis["value"].validity
        assert outcome.value_basis["value"].kind.value == "ESTIMATE"
        assert (check.parameter, check.minimum, check.observed, check.within, check.setting, check.code) == (
            "spatial_overlap_ratio", limit, pair.spatial_overlap_ratio, False, "interference_min_overlap", "BEAMS_NOT_OVERLAPPING")
    from os_optics.settings_registry import named_settings, unrecorded_settings

    assert "interference_min_overlap" in named_settings(result) and not unrecorded_settings(result)
    assert by_id["period"].value == pytest.approx(500.0, abs=1e-6)
    assert by_id["spacing"].value == pair.fringe_spacing_3d_nm
    assert by_id["faint"].value == pair.fringe_visibility and pair.fringe_visibility < 1e-4
    assert (by_id["faint"].status.value, by_id["faint"].assessment.value) == ("PASS", "MET")
    assert (by_id["strong"].status.value, by_id["strong"].assessment.value) == ("FAIL", "UNMET")
    assert by_id["strong"].value == pair.fringe_visibility and "BEAMS_NOT_OVERLAPPING" in by_id["strong"].issue_codes
