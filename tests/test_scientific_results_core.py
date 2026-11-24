"""Scientific result layer of the Engine core: observation-plane frames and absorbing semantics, crossing indices, the
power budget (conservation and analytic sink values), the first-order ray-transfer matrix (analytic ABCD), image
orientation, the polarization ellipse, and metric operand order. References are closed-form, never the implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.builders import (
    beam_dump,
    laser,
    mirror_for_reflection,
    npbs,
    observation_plane,
    polarizer,
    thin_lens,
    window,
)
from os_optics.contracts import ConstantIndexMaterial, JonesVector, ProjectSnapshot, Target
from os_optics.local import metric_descriptors, target_metrics
from os_optics.validation.descriptors import DESCRIPTOR_BY_NAME
from os_optics.validation.metrics import ALL_SUPPORTED

from helpers import arrivals, issue_codes, one_arrival, run

# Synthetic lossless glass isolates exact Fresnel-sink and ABCD references; real unstated loss remains bounded.
GLASS = ConstantIndexMaterial(material_id="glass15", valid_wavelength_nm=(100.0, 20000.0), refractive_index=1.5,
                              absorption_coefficient_per_mm=0.0)
X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)


def _budget_line(result, source_id: str = "L"):
    (line,) = [ln for ln in result.power_budget.lines if ln.source_id == source_id]
    return line


# ------------------------------------------------------------------------------------------------ power budget


def test_uncoated_window_loses_exactly_its_two_fresnel_reflections():
    plate = window("W", thickness_mm=5.0, clear_radius_mm=10.0, material="glass15", position=(50.0, 0.0, 0.0))
    det = observation_plane("D", position=(100.0, 0.0, 0.0), normal=X, absorbing=True)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, waist_radius_mm=0.3), plate), observations=(det,),
                                 materials=(GLASS,)))
    line = _budget_line(result)
    r = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2  # normal-incidence Fresnel
    assert line.input_power_W == pytest.approx(1e-3, rel=1e-15)
    assert line.sinks["DETECTED"] == pytest.approx(1e-3 * (1 - r) ** 2, rel=1e-9)
    assert line.sinks["UNTRACED_REFLECTION"] == pytest.approx(1e-3 * (r + (1 - r) * r), rel=1e-9)
    assert line.conserved and abs(line.residual_W) < 1e-15 and result.power_budget.conserved
    assert line.observed_W == {"D": pytest.approx(1e-3 * (1 - r) ** 2, rel=1e-9)}
    (element,) = result.power_budget.by_element
    assert element.element_id == "W" and element.sinks == {"UNTRACED_REFLECTION": pytest.approx(line.sinks["UNTRACED_REFLECTION"])}


def test_split_absorb_and_escape_balance_and_report_each_sink():
    split = npbs("BS", reflectance=0.3, position=(50.0, 0.0, 0.0), normal=(-math.sqrt(0.5), math.sqrt(0.5), 0.0))
    dump = beam_dump("DUMP", position=(50.0, 60.0, 0.0), axis=(0.0, -1.0, 0.0))
    pol = polarizer("P", position=(120.0, 0.0, 0.0), axis=X, angle_deg=60.0)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, waist_radius_mm=0.3), split, dump, pol)))
    line = _budget_line(result)
    assert line.sinks["ABSORBED"] == pytest.approx(0.3e-3, rel=1e-12)
    assert line.sinks["POLARIZER_REJECTION"] == pytest.approx(0.7e-3 * (1 - math.cos(math.radians(60.0)) ** 2), rel=1e-9)
    assert line.sinks["ESCAPED"] == pytest.approx(0.7e-3 * math.cos(math.radians(60.0)) ** 2, rel=1e-9)
    assert line.conserved and "POWER_BALANCE_RESIDUAL" not in issue_codes(result)


def test_aperture_truncation_is_a_sink_equal_to_the_gaussian_tail():
    stop_radius, w = 0.5, 0.5
    from os_optics.builders import aperture_stop

    stop = aperture_stop("A", position=(20.0, 0.0, 0.0), axis=X, radius_mm=stop_radius)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=1000.0, waist_radius_mm=w, waist_position_z_mm=20.0), stop)))
    line = _budget_line(result)
    expected_loss = 1e-3 * math.exp(-2.0 * stop_radius**2 / w**2)  # circular aperture centred on a round Gaussian waist
    assert line.sinks["APERTURE_TRUNCATION"] == pytest.approx(expected_loss, rel=1e-6)
    assert line.conserved


# ------------------------------------------------------------------------------------------ observation planes


def test_absorbing_detector_does_not_absorb_beams_outside_its_active_area():
    # a 5 mm detector 40 mm off the beam must not stop the main beam.
    off_axis = observation_plane("DET_OFF", position=(100.0, 40.0, 0.0), normal=X, radius_mm=2.5, absorbing=True)
    main = observation_plane("DET_MAIN", position=(200.0, 0.0, 0.0), normal=X, radius_mm=2.5, absorbing=True)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=1064.0, waist_radius_mm=0.5),), observations=(off_axis, main)))
    assert one_arrival(result, "DET_MAIN").beam.power_W == pytest.approx(1e-3)
    off = next(o for o in result.observations if o.observation_id == "DET_OFF")
    assert off.arrivals == () and off.plane.absorbing_extent == "ACTIVE_AREA"
    (crossing,) = off.outside_active_area
    assert "power_upper_bound_W" not in crossing.model_dump(mode="json")  # (a known power: earlier documents keep their bytes)
    assert crossing.miss_distance_mm == pytest.approx(37.5) and crossing.power_W == pytest.approx(1e-3)
    assert _budget_line(result).sinks == {"DETECTED": pytest.approx(1e-3)}


def test_absorbing_plane_without_aperture_is_unbounded_and_says_so():
    plane = observation_plane("WALL", position=(100.0, 40.0, 0.0), normal=X, absorbing=True)
    behind = observation_plane("BEHIND", position=(200.0, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=1064.0),), observations=(plane, behind)))
    wall = next(o for o in result.observations if o.observation_id == "WALL")
    assert wall.plane.absorbing_extent == "UNBOUNDED_PLANE" and len(wall.arrivals) == 1
    assert arrivals(result, "BEHIND") == []


def test_plane_frame_is_resolved_in_the_design_frame():
    plane = observation_plane("P", position=(100.0, 2.0, 3.0), normal=X)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=532.0),), observations=(plane,)))
    frame = next(o for o in result.observations if o.observation_id == "P").plane
    assert frame.origin_mm == pytest.approx((100.0, 2.0, 3.0)) and frame.normal == pytest.approx(X)
    assert np.cross(frame.u_axis, frame.v_axis) == pytest.approx(frame.normal) and frame.pose_frame == "DESIGN"
    a = one_arrival(result, "P")
    assert a.direction_uvn == pytest.approx((0.0, 0.0, 1.0), abs=1e-15)
    assert a.uv_mm == pytest.approx((-np.dot((0.0, 2.0, 3.0), frame.u_axis), -np.dot((0.0, 2.0, 3.0), frame.v_axis)))


def test_a_folded_path_crossing_one_plane_twice_gets_two_crossing_indices():
    m1 = mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=X, d_out=Y, clear_radius_mm=5.0)
    m2 = mirror_for_reflection("M2", position=(100.0, 20.0, 0.0), d_in=Y, d_out=(-1.0, 0.0, 0.0), clear_radius_mm=5.0)
    big = observation_plane("BIG", position=(50.0, 0.0, 0.0), normal=X)  # crossed going +x, and again by the return leg
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=532.0), m1, m2), observations=(big,)))
    crossings = sorted(a.crossing_index for a in arrivals(result, "BIG"))
    assert crossings == [0, 1]
    second = next(a for a in arrivals(result, "BIG") if a.crossing_index == 1)
    assert second.lineage == ("M1/face:reflect", "M2/face:reflect")


# ------------------------------------------------------------------------------- first-order transfer / image


def _imaging_relay(fold: bool):
    lens = thin_lens("LENS", focal_length_mm=50.0, clear_radius_mm=10.0, position=(100.0, 0.0, 0.0))
    comps = [laser("L", wavelength_nm=633.0, waist_radius_mm=0.2), lens]
    if fold:
        comps.append(mirror_for_reflection("FOLD", position=(150.0, 0.0, 0.0), d_in=X, d_out=Y, clear_radius_mm=10.0))
        image = observation_plane("IMG", position=(150.0, 50.0, 0.0), normal=Y)
    else:
        image = observation_plane("IMG", position=(200.0, 0.0, 0.0), normal=X)
    return run(ProjectSnapshot(components=tuple(comps), observations=(image,)))


def test_first_order_matrix_of_free_space_is_the_analytic_one():
    plane = observation_plane("P", position=(100.0, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0),), observations=(plane,)))
    fo = one_arrival(result, "P").first_order
    expected = np.array([[1, 0, 100, 0], [0, 1, 0, 100], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    assert np.array(fo.matrix) == pytest.approx(expected, abs=1e-6)
    assert fo.handedness_preserved and fo.image_rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_two_f_two_f_relay_inverts_the_image_with_unit_magnification():
    a = one_arrival(_imaging_relay(fold=False), "IMG")
    m = np.array(a.first_order.matrix)
    # P(100) L(50) P(100) = [[-1, 0], [-1/50, -1]] per axis
    assert m[:2, :2] == pytest.approx(-np.eye(2), abs=1e-6) and m[:2, 2:] == pytest.approx(np.zeros((2, 2)), abs=1e-4)
    assert m[2:, :2] == pytest.approx(-np.eye(2) / 50.0, abs=1e-8)
    assert a.first_order.handedness_preserved and abs(abs(a.first_order.image_rotation_deg) - 180.0) < 1e-4
    assert a.first_order.transverse_magnification == pytest.approx(1.0, rel=1e-6)


def test_a_fold_mirror_mirrors_the_image_and_targets_can_require_parity():
    result = _imaging_relay(fold=True)
    a = one_arrival(result, "IMG")
    assert a.first_order.reflection_count == 1 and not a.first_order.handedness_preserved
    assert a.first_order.image_rotation_deg is None and a.first_order.mirror_axis_deg is not None
    a_viewed = np.array(a.first_order.matrix)[:2, :2] * np.array([[1.0], [-1.0 if np.dot(a.beam.direction, Y) < 0 else 1.0]])
    assert np.linalg.det(a_viewed) < 0  # the mirrored map, consistent with the reflection parity
    lens = thin_lens("LENS", focal_length_mm=50.0, clear_radius_mm=10.0, position=(100.0, 0.0, 0.0))
    fold = mirror_for_reflection("FOLD", position=(150.0, 0.0, 0.0), d_in=X, d_out=Y, clear_radius_mm=10.0)
    target = Target(target_id="upright", metric="image_parity", observation_id="IMG", target_value=1.0, tolerance_abs=0.1)
    checked = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, waist_radius_mm=0.2), lens, fold),
                                  observations=(observation_plane("IMG", position=(150.0, 50.0, 0.0), normal=Y),), targets=(target,)))
    assert checked.targets[0].value == -1.0 and checked.targets[0].status.value == "FAIL"


# ------------------------------------------------------------------------------------------------ polarization


@pytest.mark.parametrize(("jones", "s3", "hand"), [(JonesVector.right_circular(), -1.0, "RIGHT"), (JonesVector.left_circular(), 1.0, "LEFT")])
def test_circular_light_has_unit_s3_and_no_major_axis(jones, s3, hand):
    plane = observation_plane("P", position=(50.0, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, polarization=jones),), observations=(plane,)))
    pol = one_arrival(result, "P").polarization_state
    assert pol.s3 == pytest.approx(s3, abs=1e-12) and pol.handedness == hand and pol.major_axis is None
    assert pol.ellipticity_deg == pytest.approx(45.0 * s3, abs=1e-9)


def test_linear_light_azimuth_is_measured_in_the_plane_frame():
    plane = observation_plane("P", position=(50.0, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0, polarization=JonesVector.linear(30.0)),), observations=(plane,)))
    a = one_arrival(result, "P")
    assert a.polarization_state.azimuth_in_plane_deg == pytest.approx(30.0, abs=1e-9)
    assert a.polarization_state.s1 == pytest.approx(math.cos(math.radians(60.0)), abs=1e-12)


# ------------------------------------------------------------------------------------------------- metrics


def test_opd_sign_follows_the_targets_operand_order():
    near = laser("A", wavelength_nm=633.0, position=(0.0, 0.0, 0.0), direction=X)
    far = laser("B", wavelength_nm=633.0, position=(-30.0, 1.0, 0.0), direction=X)
    plane = observation_plane("P", position=(100.0, 0.0, 0.0), normal=X)
    targets = (
        Target(target_id="ab", metric="optical_path_difference_mm", observation_id="P", source_id="A", source_id_b="B", maximum=1e9),
        Target(target_id="ba", metric="optical_path_difference_mm", observation_id="P", source_id="B", source_id_b="A", maximum=1e9),
        Target(target_id="abs", metric="optical_path_difference_abs_mm", observation_id="P", source_id="B", source_id_b="A", maximum=1e9),
    )
    result = run(ProjectSnapshot(components=(near, far), observations=(plane,), targets=targets))
    values = {t.target_id: t.value for t in result.targets}
    assert values["ab"] == pytest.approx(-30.0, abs=1e-9) and values["ba"] == pytest.approx(30.0, abs=1e-9)
    assert values["abs"] == pytest.approx(30.0, abs=1e-9)


def test_every_metric_has_exactly_one_descriptor():
    optical = set(target_metrics())
    described = {d.name for d in metric_descriptors()}
    assert optical == described
    assert ALL_SUPPORTED == set(DESCRIPTOR_BY_NAME)
    for d in metric_descriptors():
        assert d.definition and d.model_basis and d.unit == target_metrics()[d.name]
    assert "minus waist position" in DESCRIPTOR_BY_NAME["waist_distance_x_mm"].sign_convention
    assert "OPL(a) - OPL(b)" in DESCRIPTOR_BY_NAME["optical_path_difference_mm"].definition


def test_polarizer_ledger_separates_rejection_from_transmission_loss():
    pol = polarizer("P", position=(50.0, 0.0, 0.0), axis=X, angle_deg=60.0, t_par=0.9, t_perp=0.002)
    result = run(ProjectSnapshot(components=(laser("L", wavelength_nm=633.0), pol)))
    line = _budget_line(result)
    along, across = 1e-3 * math.cos(math.radians(60.0)) ** 2, 1e-3 * math.sin(math.radians(60.0)) ** 2
    assert line.sinks["POLARIZER_REJECTION"] == pytest.approx(across * (1 - 0.002), rel=1e-9)
    assert line.sinks["ELEMENT_TRANSMISSION_LOSS"] == pytest.approx(along * (1 - 0.9), rel=1e-9)
    assert line.sinks["ESCAPED"] == pytest.approx(along * 0.9 + across * 0.002, rel=1e-9) and line.conserved
