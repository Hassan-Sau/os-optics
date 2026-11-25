"""Catalog products simulated by the Engine's generic physics (``os_optics.local.compile``), checked against
independent closed forms and paraxial ABCD / Gaussian-q computations done here with plain Python / NumPy from the
catalog's published numbers (``local_helpers``: Sellmeier indices, lens matrices).

Every design here is a real multi-surface system: no thin-lens plane, no collimation guard. The Gaussian beam of the
Engine is paraxial ABCD along the chief ray, so on-axis systems agree with the independent q propagation to rounding;
real-ray quantities (chief-ray heights, lateral shifts) are compared with exact closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.local import LocalValidationStatusV1, lower_project

from local_helpers import (
    AGAINST_X,
    ALONG_X,
    ALONG_Y,
    CONCAVE_MIRROR,
    DICHROIC,
    FOLD_X_TO_Y,
    HWP,
    PCX_F50,
    PCX_F100,
    PLATE_BS,
    POLARIZER,
    POWER_W,
    WINDOW,
    arrivals,
    catalog_document,
    element,
    free,
    lens_matrix,
    plane,
    product_document,
    project,
    q_through,
    quat_multiply,
    rayleigh_range,
    run,
    scope_codes,
    sellmeier_index,
    source,
    thickness,
    turn_about_z,
    waist_radius,
)

COS_22_5 = math.sqrt(2.0 + math.sqrt(2.0)) / 2.0
SIN_22_5 = math.sqrt(2.0 - math.sqrt(2.0)) / 2.0
COS_11_25 = math.sqrt((1.0 + COS_22_5) / 2.0)
SIN_11_25 = math.sqrt((1.0 - COS_22_5) / 2.0)
COS_15 = math.sqrt((1.0 + math.sqrt(3.0) / 2.0) / 2.0)
SIN_15 = math.sqrt((1.0 - math.sqrt(3.0) / 2.0) / 2.0)
SQRT_HALF = math.sqrt(0.5)
# Rotation by 45 deg about design +Z: an element axis along +X turned toward +Y.
DESIGN_Z_45 = (COS_22_5, 0.0, 0.0, SIN_22_5)


def _only(result, observation_id: str, source_id: str | None = None):
    (arrival,) = arrivals(result, observation_id, source_id)
    return arrival


def _unknown_bulk_run(run_) -> None:
    assert run_.status is LocalValidationStatusV1.UNKNOWN and run_.model_scope_issues == ()
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run_.result.issues}


def _bounded_power(arrival, no_bulk_loss_upper_W: float) -> None:
    assert arrival.beam.power_W is None  # no real catalog material states zero loss
    assert arrival.power_upper_bound_W == pytest.approx(no_bulk_loss_upper_W, rel=1e-9)


def _efl(matrix) -> float:
    return -1.0 / matrix[1, 0]


# ------------------------------------------------------------------------------------------- lenses


@pytest.mark.parametrize("turned", [False, True], ids=["curved_side_first", "plane_side_first"])
def test_a_collimated_gaussian_focuses_at_the_paraxial_back_focal_distance(turned):
    """A 1 mm, 780 nm beam 200 mm before the plano-convex prescription: the Engine's waist lies where q propagation
    through the exact paraxial prescription puts it, i.e. at the back focal distance plus Self's small Gaussian shift."""

    wavelength, w0, distance = 780.0, 1.0, 200.0
    t = thickness(PCX_F100)
    matrix = lens_matrix(PCX_F100, wavelength, turned=turned)
    q_out = q_through(matrix, complex(distance, rayleigh_range(w0, wavelength)))
    waist_after_exit = -q_out.real
    efl, bfd, ffd = _efl(matrix), -matrix[0, 0] / matrix[1, 0], -matrix[1, 1] / matrix[1, 0]
    zr = rayleigh_range(w0, wavelength)
    self_shift = efl**2 * (distance - ffd) / ((distance - ffd) ** 2 + zr**2)  # Self (1983), measured from the focal point
    assert waist_after_exit == pytest.approx(bfd + self_shift, abs=1e-9)
    assert 0.0 < self_shift < 0.1  # the Gaussian waist sits just behind the paraxial focus here

    screen_x = t + waist_after_exit  # the exit vertex is at x = t in both orientations
    lens = element("L1", "lens", PCX_F100, (t, 0.0, 0.0), AGAINST_X) if turned else element("L1", "lens", PCX_F100)
    result = run(project([source(position=(-distance, 0.0, 0.0)), lens], [plane("focus", (screen_x, 0.0, 0.0))]))
    _unknown_bulk_run(result)
    gaussian = _only(result.result, "focus").beam.gaussian
    assert gaussian.valid
    assert gaussian.waist_distance_x_mm == pytest.approx(0.0, abs=1e-6)
    assert gaussian.waist_radius_x_mm == pytest.approx(waist_radius(q_out, wavelength), rel=1e-9)
    # The effective focal length at the catalog's design wavelength is the published specification within its tolerance.
    spec = product_document(PCX_F100)["specifications"]
    efl_d = _efl(lens_matrix(PCX_F100, float(spec["design_wavelength_nm"]["value"])))
    assert efl_d == pytest.approx(float(spec["focal_length_mm"]["value"]), rel=float(spec["focal_length_tolerance_percent"]["value"]) / 100)


def test_the_two_orientations_of_a_plano_convex_lens_have_the_same_focal_length_but_different_back_focal_distances():
    forward, turned = lens_matrix(PCX_F100, 780.0), lens_matrix(PCX_F100, 780.0, turned=True)
    assert _efl(forward) == pytest.approx(_efl(turned), rel=1e-12)
    # plane side first: the principal plane sits on the curved (exit) vertex, so the focus is t / n further from the exit vertex
    n = sellmeier_index("N-BK7", 780.0)
    back_forward, back_turned = -forward[0, 0] / forward[1, 0], -turned[0, 0] / turned[1, 0]
    assert back_turned - back_forward == pytest.approx(thickness(PCX_F100) / n, rel=1e-9)


def _keplerian_gap(first, second) -> float:
    return -(second[1, 0] * first[0, 0] + second[1, 1] * first[1, 0]) / (second[1, 0] * first[1, 0])


def test_a_two_lens_telescope_expands_the_beam_by_the_focal_length_ratio():
    """L1 (f ~ 50) then L2 (f ~ 100, turned around) at the afocal spacing: the output waist is w0 * f2 / f1 exactly
    (angular magnification of an afocal system). L2 receives a strongly diverging beam -- still in model scope."""

    wavelength, w0 = 780.0, 1.0
    m1, m2 = lens_matrix(PCX_F50, wavelength), lens_matrix(PCX_F100, wavelength, turned=True)
    gap = _keplerian_gap(m1, m2)
    system = m2 @ free(gap) @ m1
    assert abs(system[1, 0]) < 1e-15
    ratio = _efl(m2) / _efl(m1)
    assert ratio == pytest.approx(2.0, rel=0.02)
    t1, t2 = thickness(PCX_F50), thickness(PCX_F100)
    l2_origin = t1 + gap + t2
    document = project(
        [source(), element("L1", "lens", PCX_F50), element("L2", "lens", PCX_F100, (l2_origin, 0.0, 0.0), AGAINST_X)],
        [plane("output", (l2_origin + 150.0, 0.0, 0.0), radius_mm=20.0)],
        [{"target_id": "expanded", "metric": "waist_radius_x_mm", "observation_id": "output", "target_value": w0 * ratio,
          "tolerance_rel": 1e-6, "unit": "mm"}],
    )
    result = run(document)
    _unknown_bulk_run(result)
    assert [(target.target_id, target.status.value) for target in result.result.targets] == [("expanded", "PASS")]
    gaussian = _only(result.result, "output").beam.gaussian
    assert gaussian.waist_radius_x_mm == pytest.approx(w0 * ratio, rel=1e-9)
    assert gaussian.waist_radius_y_mm == pytest.approx(w0 * ratio, rel=1e-9)
    # the full q, not just the size: the output waist position is the afocal image of the input waist
    q_out = q_through(system, complex(200.0, rayleigh_range(w0, wavelength)))
    assert gaussian.waist_distance_x_mm == pytest.approx(150.0 + q_out.real, abs=1e-6)


def _relay_4f(extra_sources=()):
    wavelength = 780.0
    m1, m2 = lens_matrix(PCX_F100, wavelength, turned=True), lens_matrix(PCX_F100, wavelength)
    ffd1, bfd1 = -m1[1, 1] / m1[1, 0], -m1[0, 0] / m1[1, 0]
    ffd2, bfd2 = -m2[1, 1] / m2[1, 0], -m2[0, 0] / m2[1, 0]
    t = thickness(PCX_F100)
    l2_entrance = t + bfd1 + ffd2
    image_x = l2_entrance + t + bfd2
    system = free(bfd2) @ m2 @ free(bfd1 + ffd2) @ m1 @ free(ffd1)
    document = project(
        [source(position=(-ffd1, 0.0, 0.0), waist_mm=0.1), *extra_sources,
         element("L1", "lens", PCX_F100, (t, 0.0, 0.0), AGAINST_X), element("L2", "lens", PCX_F100, (l2_entrance, 0.0, 0.0))],
        [plane("image", (image_x, 0.0, 0.0))],
    )
    return document, system, -ffd1


def test_a_4f_relay_images_the_front_focal_plane_onto_the_back_focal_plane():
    """Two identical lenses: the object-to-image matrix is ((-1, 0), (0, -1)): the 0.1 mm waist (z_R ~ 40 mm, far from
    collimated at L1) is reproduced on the image plane, and an off-axis parallel beam lands inverted at the same height."""

    document, system, source_x = _relay_4f([source("OFF", (0.0, 0.0, 0.0), waist_mm=0.1)])
    height = 1.0
    document["components"][1]["nominal_pose"]["position_mm"] = [source_x, height, 0.0]
    np.testing.assert_allclose(system, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-12)
    result = run(document)
    assert result.model_scope_issues == ()
    on_axis = _only(result.result, "image", "SRC")
    assert on_axis.beam.gaussian.waist_distance_x_mm == pytest.approx(0.0, abs=1e-6)
    assert on_axis.beam.gaussian.waist_radius_x_mm == pytest.approx(0.1, rel=1e-9)
    off_axis = _only(result.result, "image", "OFF")
    # real rays: paraxially -h; the residual is the relay's (tiny) aberration of a 1 mm high chief ray
    assert off_axis.beam.position_mm[1] == pytest.approx(-height, abs=2e-3)
    assert off_axis.beam.position_mm[2] == pytest.approx(0.0, abs=1e-12)


def test_a_4f_relay_uses_one_product_twice_and_one_engine_material():
    document, _, _ = _relay_4f()
    lowered = lower_project(document, catalog_document())
    assert [p.optical_sku for p in lowered.simulated_parts] == [PCX_F100, PCX_F100]
    # (and the environment's air, which every native request states as the Engine material ``air``)
    assert [m.material_id for m in lowered.request.project.materials] == ["air", "catalog:N-BK7"]
    lenses = [c for c in lowered.request.project.components if c.role == "lens"]
    assert lenses[0].surfaces == lenses[1].surfaces and lenses[0].pose != lenses[1].pose


def test_the_chromatic_focal_shift_of_an_n_bk7_lens_is_the_dispersion_prediction():
    """F (486.13 nm) and C (656.27 nm) beams through the same lens: the waist separation is what the Sellmeier indices
    predict through the prescription (about f / V_d ~ 1.5 mm)."""

    wf, wc, distance = 486.1327, 656.2725, 200.0
    document = project(
        [source("F", wavelength_nm=wf), source("C", wavelength_nm=wc), element("L1", "lens", PCX_F100)],
        [plane("screen", (100.0, 0.0, 0.0))],
    )
    result = run(document)
    assert result.model_scope_issues == ()
    t = thickness(PCX_F100)
    positions = {}
    for name, wavelength in (("F", wf), ("C", wc)):
        predicted = t - q_through(lens_matrix(PCX_F100, wavelength), complex(distance, rayleigh_range(1.0, wavelength))).real
        engine = 100.0 - _only(result.result, "screen", name).beam.gaussian.waist_distance_x_mm
        assert engine == pytest.approx(predicted, abs=1e-6)
        positions[name] = engine
    shift = positions["C"] - positions["F"]
    n_d, n_f, n_c = (sellmeier_index("N-BK7", w) for w in (587.5618, wf, wc))
    abbe = (n_d - 1.0) / (n_f - n_c)
    assert abbe == pytest.approx(64.17, abs=0.05)  # N-BK7 catalogue Abbe number
    assert shift == pytest.approx(_efl(lens_matrix(PCX_F100, 587.5618)) / abbe, rel=0.02)


# ------------------------------------------------------------------------------------------- plates and splitters


def _lateral_shift(thickness_mm: float, n: float, aoi_rad: float) -> float:
    s = math.sin(aoi_rad)
    return thickness_mm * s * (1.0 - math.cos(aoi_rad) / math.sqrt(n * n - s * s))


def _fresnel_tp(n: float, aoi_rad: float) -> float:
    ci = math.cos(aoi_rad)
    ct = math.sqrt(1.0 - (math.sin(aoi_rad) / n) ** 2)
    tp = 2.0 * ci / (n * ci + ct)
    return (n * ct / ci) * tp * tp


def test_a_window_at_45_degrees_shifts_the_beam_laterally_and_loses_the_fresnel_power():
    wavelength = 780.0
    n = sellmeier_index("N-BK7", wavelength)
    window = element("W1", "window", WINDOW, (0.0, 0.0, 0.0), quat_multiply(DESIGN_Z_45, ALONG_X))
    result = run(project([source(position=(-100.0, 0.0, 0.0)), window], [plane("after", (50.0, 0.0, 0.0))]))
    assert result.model_scope_issues == ()  # an uncoated window has no angle limit; 780 nm is inside its band
    arrival = _only(result.result, "after")
    assert arrival.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert arrival.beam.position_mm[1] == pytest.approx(_lateral_shift(3.0, n, math.pi / 4), abs=1e-9)
    # linear polarization along design +Y lies in the plane of incidence: p-polarized, two bare Fresnel interfaces
    _unknown_bulk_run(result)
    _bounded_power(arrival, POWER_W * _fresnel_tp(n, math.pi / 4) ** 2)


def _splitter_run(sku: str, wavelength_nm: float):
    document = project(
        [source(position=(-100.0, 0.0, 0.0), wavelength_nm=wavelength_nm), element("BS", "splitter", sku, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
        [plane("reflected", (0.0, 60.0, 0.0), ALONG_Y), plane("transmitted", (60.0, 0.0, 0.0))],
    )
    return run(document)


def test_a_plate_beamsplitter_splits_by_its_coating_and_shifts_the_transmitted_beam():
    wavelength = 633.0
    result = _splitter_run(PLATE_BS, wavelength)
    assert result.model_scope_issues == ()  # the back-face AR (45-45 deg) is judged by its equivalent angle in air
    reflected, transmitted = _only(result.result, "reflected"), _only(result.result, "transmitted")
    assert reflected.beam.power_W == pytest.approx(0.5 * POWER_W, rel=1e-12)
    _unknown_bulk_run(result)
    _bounded_power(transmitted, 0.5 * 0.99 * POWER_W)
    assert reflected.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert transmitted.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    n = sellmeier_index("N-BK7", wavelength)
    assert transmitted.beam.position_mm[1] == pytest.approx(-_lateral_shift(1.0, n, math.pi / 4), abs=1e-9)
    assert reflected.beam.position_mm[0] == pytest.approx(0.0, abs=1e-9)  # reflected at the coated front face


@pytest.mark.parametrize("wavelength_nm, reflected_fraction, transmitted_fraction", [
    (532.0, 0.95, 0.05 * 0.99),  # reflect band 400-600 nm
    (780.0, 0.1, 0.9 * 0.99),  # transmit band 650-1100 nm
])
def test_a_dichroic_mirror_routes_by_wavelength(wavelength_nm, reflected_fraction, transmitted_fraction):
    result = _splitter_run(DICHROIC, wavelength_nm)
    assert result.model_scope_issues == ()
    assert _only(result.result, "reflected").beam.power_W == pytest.approx(reflected_fraction * POWER_W, rel=1e-12)
    _unknown_bulk_run(result)
    _bounded_power(_only(result.result, "transmitted"), transmitted_fraction * POWER_W)


def test_a_wavelength_in_the_dichroic_transition_gap_is_out_of_model_scope():
    result = _splitter_run(DICHROIC, 620.0)
    assert scope_codes(result) == [("BS", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S2")]
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    assert result.engine_status.value == "UNKNOWN"  # unstated N-BK7 loss; the scope violation does not rewrite physics
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.result.issues}
    assert "620.0 nm" in result.model_scope_issues[0].message and "400-600 nm, 650-1100 nm" in result.model_scope_issues[0].message


# ------------------------------------------------------------------------------------------- mirrors


def test_a_concave_mirror_focuses_a_collimated_beam_at_half_its_radius():
    """Normal incidence on the R = 200 mm concave mirror (face toward -X): the reflected waist lies where the mirror
    matrix ((1, 0), (-2/R, 1)) puts it -- within 10 um of R/2 for a 2 mm beam."""

    wavelength, w0, distance, radius = 780.0, 2.0, 300.0, 200.0
    document = project(
        [source(position=(-distance, 0.0, 0.0), waist_mm=w0), element("M1", "mirror", CONCAVE_MIRROR, (0.0, 0.0, 0.0), AGAINST_X)],
        [plane("focus", (-radius / 2, 0.0, 0.0), radius_mm=20.0)],
    )
    # The path crosses the plane twice (on its way to the mirror and after it); the local run draws both crossings.
    local = run(document)
    result = local.result
    assert local.model_scope_issues == ()
    back = [a for a in arrivals(result, "focus") if a.beam.direction[0] < 0]
    (reflected,) = back
    (incoming,) = [a for a in arrivals(result, "focus") if a.beam.direction[0] > 0]
    assert (incoming.crossing_index, reflected.crossing_index) == (0, 1) and incoming.beam.path_id == reflected.beam.path_id
    q_out = q_through(np.array([[1.0, 0.0], [-2.0 / radius, 1.0]]), complex(distance, rayleigh_range(w0, wavelength)))
    waist_after_mirror = -q_out.real
    assert abs(waist_after_mirror - radius / 2) < 0.01
    assert reflected.beam.gaussian.waist_distance_x_mm == pytest.approx(radius / 2 - waist_after_mirror, abs=1e-6)
    assert reflected.beam.power_W == pytest.approx(0.99 * POWER_W, rel=1e-12)
    # The Optical Simulation View draws one footprint per crossing of the plane.
    (focus,) = local.optical_view.observation_planes
    assert [(f.path_id, f.crossing_index) for f in focus.footprints] == [(incoming.beam.path_id, 0), (incoming.beam.path_id, 1)]
    assert [f.power_W for f in focus.footprints] == pytest.approx([incoming.beam.power_W, reflected.beam.power_W], rel=1e-8)


# ------------------------------------------------------------------------------------------- polarization


def _polarization_angle_deg(arrival) -> float:
    """Angle of a linearly polarized arrival from design +Y toward design +Z (the source's local x / y), mod 180."""

    j = arrival.beam.polarization
    basis = arrival.beam.basis
    field = complex(*j.ex) * np.array(basis.e1) + complex(*j.ey) * np.array(basis.e2)
    phase = field[np.argmax(np.abs(field))]
    real = (field * np.conj(phase) / abs(phase)).real
    assert np.allclose((field * np.conj(phase) / abs(phase)).imag, 0.0, atol=1e-12)  # linear
    return math.degrees(math.atan2(real[2], real[1])) % 180.0


@pytest.mark.parametrize("half_turn, rotation_deg", [((COS_11_25, SIN_11_25), 22.5), ((COS_22_5, SIN_22_5), 45.0)])
def test_a_half_wave_plate_turns_linear_polarization_by_twice_its_angle(half_turn, rotation_deg):
    plate = element("HWP", "waveplate", HWP, (0.0, 0.0, 0.0), quat_multiply(ALONG_X, turn_about_z(*half_turn)))
    result = run(project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=633.0), plate], [plane("out", (50.0, 0.0, 0.0))]))
    assert result.model_scope_issues == ()
    arrival = _only(result.result, "out")
    assert _polarization_angle_deg(arrival) == pytest.approx(2.0 * rotation_deg, abs=1e-9)
    assert arrival.beam.power_W == pytest.approx(0.99 * POWER_W, rel=1e-12)


@pytest.mark.parametrize("half_turn, theta_deg", [((COS_15, SIN_15), 30.0), ((SQRT_HALF, SQRT_HALF), 90.0), ((1.0, 0.0), 0.0)])
def test_a_linear_polarizer_follows_malus_law_with_its_extinction_ratio(half_turn, theta_deg):
    model = product_document(POLARIZER)["simulation_models"][0]
    transmittance, extinction = float(model["transmittance"]["value"]), float(model["extinction_ratio"]["value"])
    polarizer = element("POL", "polarizer", POLARIZER, (0.0, 0.0, 0.0), quat_multiply(ALONG_X, turn_about_z(*half_turn)))
    result = run(project([source(position=(-50.0, 0.0, 0.0)), polarizer], [plane("out", (50.0, 0.0, 0.0))]))
    theta = math.radians(theta_deg)
    expected = POWER_W * transmittance * (math.cos(theta) ** 2 + math.sin(theta) ** 2 / extinction)
    assert _only(result.result, "out").beam.power_W == pytest.approx(expected, rel=1e-9)


def test_a_half_wave_plate_then_a_crossed_polarizer_extinguishes_the_beam():
    """HWP at 22.5 deg turns the polarization to 45 deg; a polarizer at 135 deg (axis turned 3 x 45 deg) blocks it up to
    its extinction ratio."""

    cos_67_5, sin_67_5 = SIN_22_5, COS_22_5  # half of 135 deg
    document = project(
        [source(position=(-50.0, 0.0, 0.0), wavelength_nm=633.0),
         element("HWP", "waveplate", HWP, (0.0, 0.0, 0.0), quat_multiply(ALONG_X, turn_about_z(COS_11_25, SIN_11_25))),
         element("POL", "polarizer", POLARIZER, (20.0, 0.0, 0.0), quat_multiply(ALONG_X, turn_about_z(cos_67_5, sin_67_5)))],
        [plane("out", (50.0, 0.0, 0.0))],
    )
    result = run(document)
    assert _only(result.result, "out").beam.power_W == pytest.approx(POWER_W * 0.99 * 0.8 / 1000.0, rel=1e-6)
