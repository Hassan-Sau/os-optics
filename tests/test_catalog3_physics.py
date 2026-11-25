"""Physics of the schema-3.0 catalog constructs through the local runtime, against independent references (Fresnel and
Snell formulas, Malus' law, the thin-prism formula, the erf product of a truncated Gaussian, the paraboloid's focus, the
thick-lens power of a cylinder) computed here with plain ``math`` from the published numbers -- never with the runtime's
own compile or paraxial helpers.

Frames: products are placed with ``ALONG_X`` (local +Z -> design +X, local +X -> design +Y, local +Y -> design +Z) unless
stated; sources emit along design +X with their local x along design +Y (linear polarization angle 0 = design +Y).
"""

from __future__ import annotations

import copy
import math

import pytest

from os_optics.local import LocalRunError, compile_element, load_public_catalog, run_optical_project

from catalog3_products import (
    BANDPASS,
    CATALOG_VERSION,
    CUBE,
    CYLINDER,
    CYLINDER_RADIUS_MM,
    DOVE,
    OAP,
    PARENT_FOCAL_MM,
    PBS_CUBE,
    PENTA,
    RECT_WINDOW,
    RIGHT_ANGLE,
    SAMPLER,
    WEDGE,
    WEDGE_ANGLE_DEG,
    catalog3_document,
)
from declared_helpers import declared_project
from local_helpers import AGAINST_X, ALONG_X, ALONG_Y, FOLD_X_TO_Y, POWER_W, arrivals, plane, project, quat_multiply, scope_codes, source

WAVELENGTH_NM = 633.0
AR_T = 0.995
DOC = catalog3_document()
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def _nbk7(wavelength_nm: float, document: dict = DOC) -> float:
    material = next(m for m in document["materials"] if m["material_id"] == "N-BK7")
    b = [float(v) for v in material["dispersion"]["b"]]
    c = [float(v) for v in material["dispersion"]["c_um2"]]
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def _rotate(q, v):
    w, x, y, z = q
    # v' = q v q* (Hamilton, active)
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def _about(axis: str, degrees: float):
    half = math.radians(degrees) / 2
    q = [math.cos(half), 0.0, 0.0, 0.0]
    q["xyz".index(axis) + 1] = math.sin(half)
    return tuple(q)


def _product(sku: str, document: dict = DOC) -> dict:
    return next(p for p in document["products"] if p["optical_sku"] == sku)


def _port(sku: str, port_id: str, document: dict = DOC) -> tuple[tuple[float, ...], tuple[float, ...]]:
    port = next(p for p in _product(sku, document)["frame"]["ports"] if p["port_id"] == port_id)
    return tuple(float(v) for v in port["point_mm"]), tuple(float(v) for v in port["direction"])


def _element(instance_id: str, role: str, sku: str, position=(0.0, 0.0, 0.0), quaternion=ALONG_X) -> dict:
    return {"instance_id": instance_id, "optical_role": role, "product": {"optical_sku": sku, "catalog_version": CATALOG_VERSION},
            "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)}, "retained_dofs": []}


def _source_at(point, quaternion=ALONG_X, distance: float = 100.0, **kwargs) -> dict:
    """A source ``distance`` mm before ``point`` along its own emission direction."""

    direction = _rotate(quaternion, (0.0, 0.0, 1.0))
    start = tuple(p - distance * d for p, d in zip(point, direction))
    kwargs.setdefault("wavelength_nm", WAVELENGTH_NM)
    kwargs.setdefault("waist_mm", 0.5)
    return source(position=start, quaternion=quaternion, **kwargs)


def _in_point(sku: str, quaternion=ALONG_X, position=(0.0, 0.0, 0.0), document: dict = DOC):
    local, _ = _port(sku, "IN", document)
    return tuple(p + r for p, r in zip(position, _rotate(quaternion, local)))


def _run(components: list[dict], observations: list[dict], document: dict = DOC):
    # Public catalog materials specify index, not bulk loss. For the exact
    # lossless references in this synthetic physics suite, declare copies of
    # the catalog parts with an explicit ideal test-glass assumption instead
    # of adding an unsupported field to a public catalog release.
    optical_project = project(components, observations)
    declared = declared_project(optical_project, document)
    if not declared.get("model_materials"):
        return run_optical_project(optical_project, catalog=document)  # no glass: retain public coating-scope wording
    for material in declared["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    return run_optical_project(declared, catalog=document)


def _only(result, observation_id: str):
    (arrival,) = arrivals(result, observation_id)
    return arrival


def _fresnel_normal(n: float) -> float:
    return ((n - 1.0) / (n + 1.0)) ** 2


# ----------------------------------------------------------------------------------------------- cube splitters


def test_cube_splitter_sends_half_to_each_port_and_the_power_budget_closes():
    run = _run([_source_at(_in_point(CUBE)), _element("C", "splitter", CUBE)], [plane("T", (100.0, 0.0, 0.0)), plane("R", (0.0, 100.0, 0.0), ALONG_Y)])
    assert run.status.value == "PASS" and run.model_scope_issues == ()
    transmitted, reflected = _only(run.result, "T"), _only(run.result, "R")
    expected = POWER_W * 0.5 * AR_T * AR_T
    assert transmitted.beam.power_W == pytest.approx(expected, rel=1e-12) and reflected.beam.power_W == pytest.approx(expected, rel=1e-12)
    assert transmitted.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert reflected.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)  # the OUT_R port: 90 deg to the input
    assert reflected.first_order.reflection_count == 1 and not reflected.first_order.handedness_preserved
    budget = run.result.power_budget
    assert budget.total_input_W == pytest.approx(POWER_W, rel=1e-12)
    assert sum(budget.total_sinks.values()) + budget.residual_W == pytest.approx(POWER_W, rel=1e-12)
    # the AR residual of the entrance face, and of each exit face for its half of the beam, is not followed
    assert budget.total_sinks["UNTRACED_REFLECTION"] == pytest.approx(POWER_W * (1 - AR_T) * (1 + AR_T), rel=1e-12)


@pytest.mark.parametrize("entry", ["fourth_face", "out_r_face"])
def test_a_cube_combines_and_runs_in_reverse_through_its_four_optical_faces(entry: str):
    """All four optical faces are traced: a beam entering the face opposite OUT_R (a combiner's second input) leaves
    through OUT_T (reflected) and OUT_R (transmitted); a beam entering OUT_R backwards leaves through IN (reflected) and the
    fourth face (transmitted). Each output carries half the power times the two AR transmittances, as forward."""

    a = 6.35
    to_minus_y = _about("x", 90.0)  # local +Z -> design -Y
    if entry == "fourth_face":  # local -X face = design -Y side (ALONG_X: local +X -> design +Y); the beam runs along +Y
        point, q = (0.0, -a, 0.0), ALONG_Y
        planes = [plane("T", (0.0, 100.0, 0.0), ALONG_Y), plane("R", (100.0, 0.0, 0.0))]
        expected = {"T": (0.0, 1.0, 0.0), "R": (1.0, 0.0, 0.0)}
    else:
        point, q = (0.0, a, 0.0), to_minus_y
        planes = [plane("T", (0.0, -100.0, 0.0), to_minus_y), plane("R", (-100.0, 0.0, 0.0), AGAINST_X)]
        expected = {"T": (0.0, -1.0, 0.0), "R": (-1.0, 0.0, 0.0)}
    run = _run([_source_at(point, q), _element("C", "splitter", CUBE)], planes)
    assert run.model_scope_issues == () and run.status.value == "PASS"
    for observation_id, direction in expected.items():
        arrival = _only(run.result, observation_id)
        assert arrival.beam.power_W == pytest.approx(POWER_W * 0.5 * AR_T * AR_T, rel=1e-12)
        assert arrival.beam.direction == pytest.approx(direction, abs=1e-12)


def test_an_uncoated_cube_face_transmits_the_bare_fresnel_fraction():
    document = copy.deepcopy(DOC)
    del _product(CUBE, document)["simulation_models"][0]["surfaces"][0]["coating_id"]
    run = _run([_source_at(_in_point(CUBE)), _element("C", "splitter", CUBE)], [plane("T", (100.0, 0.0, 0.0))], document)
    n = _nbk7(WAVELENGTH_NM)
    assert _only(run.result, "T").beam.power_W == pytest.approx(POWER_W * (1 - _fresnel_normal(n)) * 0.5 * AR_T, rel=1e-12)


@pytest.mark.parametrize("polarization_deg", [0.0, 90.0, 30.0, 63.0])
def test_polarizing_cube_splits_p_and_s_and_follows_malus(polarization_deg: float):
    """The coated diagonal transmits T_p = 0.95, T_s = 0.0003 and reflects R_p = 0.05, R_s = 0.995 (published s / p
    values); light polarized at theta to the plane of incidence (design XY) splits as T_p cos^2 + T_s sin^2."""

    run = _run([_source_at(_in_point(PBS_CUBE), polarization_deg=polarization_deg), _element("C", "splitter", PBS_CUBE)],
               [plane("T", (100.0, 0.0, 0.0)), plane("R", (0.0, 100.0, 0.0), ALONG_Y)])
    c2, s2 = math.cos(math.radians(polarization_deg)) ** 2, math.sin(math.radians(polarization_deg)) ** 2
    faces = POWER_W * AR_T * AR_T
    assert _only(run.result, "T").beam.power_W == pytest.approx(faces * (0.95 * c2 + 0.0003 * s2), rel=1e-9, abs=1e-18)
    assert _only(run.result, "R").beam.power_W == pytest.approx(faces * (0.05 * c2 + 0.995 * s2), rel=1e-9, abs=1e-18)
    assert run.status.value == "PASS"


def test_polarizing_cube_extinction_ratio_is_tp_over_ts():
    powers = {}
    for angle in (0.0, 90.0):
        run = _run([_source_at(_in_point(PBS_CUBE), polarization_deg=angle), _element("C", "splitter", PBS_CUBE)], [plane("T", (100.0, 0.0, 0.0))])
        powers[angle] = _only(run.result, "T").beam.power_W
    assert powers[0.0] / powers[90.0] == pytest.approx(0.95 / 0.0003, rel=1e-9)


# ------------------------------------------------------------------------------------------------------ prisms


def test_right_angle_prism_folds_by_total_internal_reflection():
    n = _nbk7(WAVELENGTH_NM)
    assert math.degrees(math.asin(1.0 / n)) < 45.0  # the hypotenuse is met at 45 deg, beyond the critical angle
    run = _run([_source_at(_in_point(RIGHT_ANGLE)), _element("P", "prism", RIGHT_ANGLE)], [plane("OUT", (0.0, 100.0, 0.0), ALONG_Y, radius_mm=50.0)])
    arrival = _only(run.result, "OUT")
    assert "P/S2:tir" in arrival.lineage and arrival.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    # two uncoated faces at normal incidence; total internal reflection loses nothing
    assert arrival.beam.power_W == pytest.approx(POWER_W * (1 - _fresnel_normal(n)) ** 2, rel=1e-12)
    assert run.status.value == "PASS"


def test_right_angle_prism_of_low_index_glass_does_not_fold():
    """With an index of 1.38 (MgF2-like dispersion) the critical angle exceeds 45 deg: the hypotenuse refracts the light out."""

    document = copy.deepcopy(DOC)
    material = next(m for m in document["materials"] if m["material_id"] == "N-BK7")
    material["dispersion"] = {"kind": "SELLMEIER", "b": ["0.48755108", "0.39875031", "2.3120353"], "c_um2": ["0.001882178", "0.008951888", "566.13559"]}
    n = _nbk7(WAVELENGTH_NM, document)
    assert math.degrees(math.asin(1.0 / n)) > 45.0
    run = _run([_source_at(_in_point(RIGHT_ANGLE)), _element("P", "prism", RIGHT_ANGLE)], [plane("OUT", (0.0, 100.0, 0.0), ALONG_Y, radius_mm=50.0)], document)
    assert arrivals(run.result, "OUT") == []
    path = run.result.paths[0]
    assert [s.end_event for s in path.segments if s.end_surface_id == "P/S2"] == ["refract"]


@pytest.mark.parametrize("degrees", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_penta_prism_deviates_by_90_deg_whatever_its_small_rotation(degrees: float):
    q = quat_multiply(_about("z", degrees), ALONG_X)  # about design Z: perpendicular to the plane of the two reflections
    run = _run([_source_at(_in_point(PENTA, q)), _element("P", "prism", PENTA, quaternion=q)], [plane("OUT", (0.0, 100.0, 0.0), ALONG_Y, radius_mm=50.0)])
    arrival = _only(run.result, "OUT")
    assert arrival.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert arrival.first_order.reflection_count == 2 and arrival.first_order.handedness_preserved
    assert run.status.value == "PASS"  # the mirrors stay inside their 15-30 deg range (angle in the glass)


def test_dove_prism_mirrors_the_image_and_two_crossed_dove_prisms_rotate_it_by_180_deg():
    run = _run([_source_at(_in_point(DOVE)), _element("D1", "prism", DOVE)], [plane("OUT", (100.0, 0.0, 0.0), radius_mm=50.0)])
    arrival = _only(run.result, "OUT")
    first = arrival.first_order
    assert arrival.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12) and "D1/S2:tir" in arrival.lineage
    assert first.reflection_count == 1 and not first.handedness_preserved and first.image_rotation_deg is None
    # inverted across the base normal (local X = design Y = plane u), upright along the base (plane v)
    assert [v for row in first.matrix[:2] for v in row[:2]] == pytest.approx([-1.0, 0.0, 0.0, 1.0], abs=1e-6)
    # a second dove prism turned 90 deg about the beam: the image is rotated by 2 x 90 = 180 deg, handedness restored
    turned = quat_multiply(_about("x", 90.0), ALONG_X)
    second_in = _in_point(DOVE, turned)
    position = (60.0, -second_in[1], -second_in[2])
    run = _run([_source_at(_in_point(DOVE)), _element("D1", "prism", DOVE), _element("D2", "prism", DOVE, position, turned)],
               [plane("OUT", (150.0, 0.0, 0.0), radius_mm=50.0)])
    first = _only(run.result, "OUT").first_order
    assert first.reflection_count == 2 and first.handedness_preserved and first.image_rotation_deg == pytest.approx(180.0, abs=1e-6)
    assert [v for row in first.matrix[:2] for v in row[:2]] == pytest.approx([-1.0, 0.0, 0.0, -1.0], abs=1e-6)


def test_wedge_prism_deviation_matches_snell_and_the_thin_prism_formula():
    n, alpha = _nbk7(WAVELENGTH_NM), math.radians(WEDGE_ANGLE_DEG)
    exact = math.asin(n * math.sin(alpha)) - alpha  # normal incidence on the front face, angle alpha on the back face
    run = _run([_source_at(_in_point(WEDGE)), _element("W", "prism", WEDGE)], [plane("OUT", (100.0, 0.0, 0.0), radius_mm=50.0)])
    direction = _only(run.result, "OUT").beam.direction
    deviation = math.atan2(-direction[1], direction[0])  # toward the thick side: local -X = design -Y
    assert direction[2] == pytest.approx(0.0, abs=1e-15) and deviation == pytest.approx(exact, abs=1e-12)
    assert deviation == pytest.approx((n - 1) * alpha, rel=2e-3)  # thin prism: delta = (n - 1) alpha + O(alpha^3)


# --------------------------------------------------------------------------------------------- apertures, OAP


def test_rectangular_aperture_truncation_is_the_erf_product():
    waist = 4.0  # 1/e^2 radius; clear aperture 16 x 8 mm (half widths 8, 4)
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=waist), _element("W", "window", RECT_WINDOW)],
               [plane("T", (100.0, 0.0, 0.0), radius_mm=40.0)])
    truncation = [e.power_W for e in run.result.power_budget.entries if e.kind == "APERTURE_TRUNCATION" and e.surface_id == "W/S1"]
    rayleigh = math.pi * waist * waist / (WAVELENGTH_NM * 1e-6)
    radius = waist * math.sqrt(1.0 + (100.0 / rayleigh) ** 2)  # the beam radius at the window, 100 mm from the waist
    fraction = math.erf(math.sqrt(2.0) * 8.0 / radius) * math.erf(math.sqrt(2.0) * 4.0 / radius)
    assert truncation == [pytest.approx(POWER_W * (1.0 - fraction), rel=1e-6)]


@pytest.mark.parametrize("local_xy, clipped", [((0.0, 5.8), True), ((9.6, 0.0), True), ((0.0, 6.3), False), ((10.4, 0.0), False)])
def test_light_between_a_rectangular_clear_aperture_and_the_outline_is_blocked(local_xy, clipped: bool):
    """The cylindrical lens's clear aperture is 18 x 11 mm inside a 20 x 12 mm outline (rings of 1 and 0.5 mm): light on the
    rim is blocked, light beyond the outline passes by."""

    x, y = local_xy  # local X -> design Y, local Y -> design Z
    run = _run([source(position=(-100.0, x, y), wavelength_nm=WAVELENGTH_NM, waist_mm=0.05), _element("L", "lens", CYLINDER)],
               [plane("T", (100.0, 0.0, 0.0), radius_mm=40.0)])
    assert (run.result.paths[0].termination == "CLIPPED") == clipped
    assert bool(arrivals(run.result, "T")) != clipped


def test_off_axis_parabola_focuses_collimated_light_at_its_reflected_focal_length():
    """IN: collimated light along the parent axis (local -Z -> design +X with AGAINST_X); the paraboloid's focus is its vertex
    plus the parent focal length along its axis, local (-101.6, 0, 0) -> design (0, 101.6, 0): the reflected focal length
    2 f_p / (1 + cos 90 deg) = 101.6 mm from the aperture centre."""

    rfl = 2.0 * PARENT_FOCAL_MM / (1.0 + math.cos(math.radians(90.0)))
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=1064.0, waist_mm=2.0), _element("M", "mirror", OAP, quaternion=AGAINST_X)],
               [plane("F", (0.0, rfl, 0.0), ALONG_Y, radius_mm=50.0), plane("NEAR", (0.0, rfl - 5.0, 0.0), ALONG_Y, radius_mm=50.0)])
    focus, near = _only(run.result, "F"), _only(run.result, "NEAR")
    assert focus.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert focus.uv_mm == pytest.approx((0.0, 0.0), abs=1e-9) and focus.pupil_rms_radius_mm < 1e-9
    assert near.pupil_rms_radius_mm > 0.05  # 5 mm before the focus the converging beam is still wide
    assert focus.beam.power_W == pytest.approx(POWER_W * 0.96, rel=1e-12) and run.status.value == "PASS"


def test_cylindrical_lens_focuses_a_line_along_its_curved_axis():
    n = _nbk7(WAVELENGTH_NM)
    efl = CYLINDER_RADIUS_MM / (n - 1.0)  # plano-convex, curved face first: thick-lens EFL = R / (n - 1)
    bfl = efl - 4.0 / n
    waist = 1.0
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=waist), _element("L", "lens", CYLINDER)],
               [plane("F", (4.0 + bfl, 0.0, 0.0), radius_mm=40.0)])
    gauss = _only(run.result, "F").beam.gaussian
    axes = {"curved": None, "straight": None}
    for e, radius in ((gauss.e1, gauss.beam_radius_x_mm), (gauss.e2, gauss.beam_radius_y_mm)):
        axes["curved" if abs(e[1]) > abs(e[2]) else "straight"] = radius  # curved axis local X = design Y
    focused = WAVELENGTH_NM * 1e-6 * efl / (math.pi * waist)  # far-field waist of a collimated Gaussian at the focus
    assert axes["curved"] == pytest.approx(focused, rel=1e-2)
    assert axes["straight"] == pytest.approx(waist, rel=1e-2)


# ------------------------------------------------------------------------------------------- coatings and scope


@pytest.mark.parametrize("wavelength_nm, transmittance", [(525.0, 0.6275), (550.0, 0.855), (460.0, 0.2404)])
def test_typical_spectrum_coating_interpolates_its_table(wavelength_nm: float, transmittance: float):
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=wavelength_nm), _element("F", "filter", BANDPASS)], [plane("T", (100.0, 0.0, 0.0))])
    assert _only(run.result, "T").beam.power_W == pytest.approx(POWER_W * transmittance, rel=1e-12)
    assert run.status.value == "PASS"


def test_a_wavelength_beyond_a_spectrum_is_out_of_model_scope():
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=720.0), _element("F", "filter", BANDPASS)], [plane("T", (100.0, 0.0, 0.0))])
    codes = scope_codes(run)
    assert ("F", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S1") in codes and ("F", "MODEL_WAVELENGTH_OUTSIDE_BAND", None) in codes
    message = next(i.message for i in run.model_scope_issues if i.code == "COATING_WAVELENGTH_OUTSIDE_BANDS")
    assert "whose spectrum is published for 400-700 nm only" in message
    assert run.status.value == "OUT_OF_MODEL_SCOPE"


def test_coating_band_and_angle_edges_are_out_of_model_scope():
    run = _run([_source_at(_in_point(PBS_CUBE), wavelength_nm=590.0), _element("C", "splitter", PBS_CUBE)], [plane("T", (100.0, 0.0, 0.0))])
    assert ("C", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S2") in scope_codes(run)
    # the diagonal coating is specified at 45 deg inside the glass: nominal use is in scope (its air-equivalent angle,
    # asin(n sin 45 deg), would be far outside); a single design angle of a splitter coating is matched within 2 deg
    # (validity.SINGLE_ANGLE_TOLERANCE_DEG), so a 2 deg turn of the cube (about 1.3 deg inside the glass) stays in scope and
    # a 5 deg turn (about 3.3 deg inside) is not
    n = _nbk7(WAVELENGTH_NM)
    assert n * math.sin(math.radians(45.0)) > 1.0
    nominal = _run([_source_at(_in_point(CUBE)), _element("C", "splitter", CUBE)], [plane("T", (100.0, 0.0, 0.0))])
    assert nominal.model_scope_issues == ()

    def turned(degrees):
        q = quat_multiply(_about("z", degrees), ALONG_X)
        return _run([_source_at(_in_point(CUBE, q)), _element("C", "splitter", CUBE, quaternion=q)], [plane("T", (100.0, 0.0, 0.0), radius_mm=40.0)])

    assert not [i for i in turned(2.0).model_scope_issues if i.code == "COATING_AOI_OUTSIDE_RANGE" and i.surface_id == "S2"]
    issue = next(i for i in turned(5.0).model_scope_issues if i.code == "COATING_AOI_OUTSIDE_RANGE" and i.surface_id == "S2")
    assert "angle of incidence in the medium the light arrives from" in issue.message
    assert "matched within +-2 deg" in issue.message


def test_a_beam_through_a_prism_body_sideways_is_out_of_model_scope():
    """Along design +Z the beam crosses the right-angle prism through its untraced top and bottom faces (local +-Y)."""

    run = _run([source(position=(0.0, 0.0, -100.0), quaternion=IDENTITY_Q, wavelength_nm=WAVELENGTH_NM), _element("P", "prism", RIGHT_ANGLE)],
               [plane("T", (0.0, 0.0, 100.0), IDENTITY_Q)])
    issue = next(i for i in run.model_scope_issues if i.code == "RAY_CROSSES_ELEMENT_BODY")
    assert issue.instance_id == "P" and "through its edge or a face that is not a traced surface" in issue.message
    assert run.status.value == "OUT_OF_MODEL_SCOPE"


def test_light_leaving_a_cube_through_an_untraced_face_is_out_of_model_scope():
    """Entering the cube's input face tilted 40 deg toward local +Y, the refracted beam reaches the untraced top face
    before the diagonal."""

    q = quat_multiply(_about("y", -40.0), ALONG_X)
    hit = _rotate(ALONG_X, (0.0, 4.5, -6.35))
    run = _run([_source_at(hit, q, distance=50.0), _element("C", "splitter", CUBE)], [plane("T", (0.0, 0.0, 200.0), IDENTITY_Q, radius_mm=400.0)])
    issue = next(i for i in run.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE")
    assert "through its edge or a face that is not a traced surface" in issue.message


def test_wedged_sampler_reflects_the_bare_fresnel_fraction():
    """The sampler's uncoated front face is traced TRANSMITTED_AND_REFLECTED: the reflected branch is the bare Fresnel
    reflection, here of s-polarized light at 45 deg."""

    n = _nbk7(WAVELENGTH_NM)
    cos_i = math.cos(math.radians(45.0))
    cos_t = math.sqrt(1.0 - (math.sin(math.radians(45.0)) / n) ** 2)
    r_s = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    run = _run([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, polarization_deg=90.0), _element("S", "splitter", SAMPLER, quaternion=FOLD_X_TO_Y)],
               [plane("R", (0.0, 100.0, 0.0), ALONG_Y)])
    assert _only(run.result, "R").beam.power_W == pytest.approx(POWER_W * r_s, rel=1e-10)


def test_a_splitting_surface_between_the_same_medium_still_needs_a_coating():
    document = copy.deepcopy(DOC)
    del _product(CUBE, document)["simulation_models"][0]["surfaces"][1]["coating_id"]
    document["coatings"] = [c for c in document["coatings"] if c["coating_id"] != "BS_CUBE_400-1100R0.5T0.5_AOI45"]
    catalog = load_public_catalog(document)
    with pytest.raises(LocalRunError) as exc:
        compile_element("C", catalog.get(CUBE), catalog)
    assert [c.value for c in exc.value.codes] == ["SURFACE_COATING_REQUIRED"] and "same medium on both sides" in str(exc.value)


# ------------------------------------------------------------------------------------------ catalog browsing


def test_catalog_list_filters_and_summarizes_the_new_categories_and_specifications(tmp_path, capsys):
    import json

    from os_optics.cli import main

    path = tmp_path / "catalog3.json"
    path.write_text(json.dumps(DOC), encoding="utf-8")
    assert main(["catalog", "--catalog", str(path), "list", "--category", "PRISM"]) == 0
    listed = json.loads(capsys.readouterr().out)
    rows = {row["optical_sku"]: row for row in listed["products"]}
    assert set(rows) == {RIGHT_ANGLE, PENTA, DOVE, WEDGE}
    assert rows[RIGHT_ANGLE]["design_roles"] == ["mirror", "prism"] and rows[RIGHT_ANGLE]["model_kind"] == "SURFACE_SET"
    assert rows[RIGHT_ANGLE]["deviation_angle_deg"] == "90" and len(rows[RIGHT_ANGLE]["outer_dimensions_mm"]) == 3
    assert rows[WEDGE]["wedge_angle_deg"] == "2" and rows[WEDGE]["outer_diameter_mm"] == "25.4"
    assert main(["catalog", "--catalog", str(path), "list", "--form", "OFF_AXIS_PARABOLIC_MIRROR"]) == 0
    (row,) = json.loads(capsys.readouterr().out)["products"]
    assert (row["parent_focal_length_mm"], row["off_axis_angle_deg"], row["focal_length_mm"]) == ("50.8", "90", "101.6")
    assert main(["catalog", "--catalog", str(path), "list", "--form", "POLARIZING_CUBE_BEAMSPLITTER"]) == 0
    (row,) = json.loads(capsys.readouterr().out)["products"]
    assert row["design_roles"] == ["polarizer", "splitter"] and row["extinction_ratio"] == "3166.6"


def test_catalog_list_rows_carry_the_facts_that_separate_sibling_products(tmp_path, capsys):
    """A summary row states a filter's centre and FWHM, the thickness and the clear aperture (Agent ``catalog list``
    and Designer ``queryCatalog({summary: true})`` share ``product_summary``)."""

    import json

    from os_optics.cli import main
    from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
    from os_optics.local.json_api import product_summary

    doc = json.loads(json.dumps(DOC))
    (bandpass,) = [p for p in doc["products"] if p["optical_sku"] == BANDPASS]
    bandpass["specifications"]["bandwidth_fwhm_nm"] = {"value": "10", "status": "SPECIFIED"}
    path = tmp_path / "catalog3.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["catalog", "--catalog", str(path), "list", "--form", "BANDPASS_FILTER"]) == 0
    (row,) = json.loads(capsys.readouterr().out)["products"]
    assert (row["center_wavelength_nm"], row["bandwidth_fwhm_nm"], row["thickness_mm"], row["clear_aperture_diameter_mm"]) == ("550", "10", "5", "22")
    rows = {p.optical_sku: product_summary(p) for p in PublicOpticalCatalogV2.model_validate(doc).products}
    assert rows[RECT_WINDOW]["clear_aperture_dimensions_mm"] == ["16", "8"] and "bandwidth_fwhm_nm" not in rows[RECT_WINDOW]


def test_catalog_list_states_and_filters_the_supply_facts(tmp_path, capsys):
    """A summary row states whether OpenSci supplies the product, whether OpenSci mounts it and its delivered
    form; ``--readiness`` / ``--mounting`` (``catalog_products(readiness=, mounting=)``) filter by them. A release that
    does not state the facts mounts every ORDERABLE product and delivers the optic itself."""

    import json

    from os_optics.cli import main
    from os_optics.local import json_api

    doc = json.loads(json.dumps(DOC))
    doc["schema_version"] = "3.1"
    for product in doc["products"]:
        product["commercial_readiness"] = "ORDERABLE"
        product["opensci_mounting"], product["delivered_form"] = "NOT_AVAILABLE", "OPTIC"
    by_sku = {p["optical_sku"]: p for p in doc["products"]}
    by_sku[RIGHT_ANGLE]["opensci_mounting"] = "AVAILABLE"
    by_sku[PENTA]["delivered_form"] = "HOUSED"
    by_sku[DOVE].update(commercial_readiness="SIMULATION_ONLY")
    path = tmp_path / "catalog31.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["catalog", "--catalog", str(path), "list", "--category", "PRISM"]) == 0
    rows = {row["optical_sku"]: row for row in json.loads(capsys.readouterr().out)["products"]}
    assert (rows[RIGHT_ANGLE]["commercial_readiness"], rows[RIGHT_ANGLE]["opensci_mounting"]) == ("ORDERABLE", "AVAILABLE")
    assert "delivered_form" not in rows[RIGHT_ANGLE]  # the optic itself
    assert (rows[PENTA]["opensci_mounting"], rows[PENTA]["delivered_form"]) == ("NOT_AVAILABLE", "HOUSED")
    assert rows[DOVE]["commercial_readiness"] == "SIMULATION_ONLY" and "opensci_mounting" not in rows[DOVE]
    assert main(["catalog", "--catalog", str(path), "list", "--category", "PRISM", "--mounting", "AVAILABLE"]) == 0
    assert [r["optical_sku"] for r in json.loads(capsys.readouterr().out)["products"]] == [RIGHT_ANGLE]
    assert main(["catalog", "--catalog", str(path), "list", "--category", "PRISM", "--readiness", "ORDERABLE",
                 "--mounting", "NOT_AVAILABLE"]) == 0
    assert {r["optical_sku"] for r in json.loads(capsys.readouterr().out)["products"]} == {PENTA, WEDGE}
    # a 3.0 release: ORDERABLE means mounted
    earlier = load_public_catalog(DOC)
    first = earlier.products[0]
    row = json_api.product_summary(first)
    assert row.get("opensci_mounting") == ("AVAILABLE" if first.commercial_readiness.value == "ORDERABLE" else None)
    assert "delivered_form" not in row


def test_catalog_products_refuses_an_unknown_supply_filter():
    from os_optics.local import json_api

    for kwargs in ({"readiness": "SOMETIMES"}, {"mounting": "MAYBE"}):
        reply = json_api.catalog_products(**kwargs)
        assert reply["ok"] is False
        assert reply["error"]["problems"][0]["code"] in ("CATALOG_QUERY_INVALID", "CATALOG_NOT_AVAILABLE")
