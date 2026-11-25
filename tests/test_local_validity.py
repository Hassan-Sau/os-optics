"""Published-model validity after the run (``os_optics.local.validity``): every code, exact band edges, the
equivalent-angle-in-air rule, aggregation, and that a violation never changes the Engine result."""

from __future__ import annotations

import math

import pytest

from os_optics.local import (
    AOI_TOLERANCE_DEG,
    MODEL_SCOPE_CODES,
    LocalValidationStatusV1,
    check_local_validation_binding,
)
from os_optics.local.validity import equivalent_air_aoi_deg
from os_optics.optical_view import geometry as g

from local_helpers import (
    AGAINST_X,
    ALONG_X,
    ALONG_Y,
    CONCAVE_MIRROR,
    DICHROIC,
    FLAT_MIRROR,
    FOLD_A,
    FOLD_B,
    FOLD_X_TO_Y,
    HWP,
    PCX_F100,
    PLATE_BS,
    POLARIZER,
    SQRT_HALF,
    TEST_CATALOG_VERSION,
    arrivals,
    catalog_document,
    element,
    plane,
    product_document,
    project,
    quat_multiply,
    run,
    scope_codes,
    source,
    with_lossless_bulk,
)

TILT_20_ABOUT_DESIGN_Z = (math.cos(math.radians(10.0)), 0.0, 0.0, math.sin(math.radians(10.0)))  # a tilted test pose (not hashed data)


def _lens_run(wavelength_nm: float, quaternion=ALONG_X, **kwargs):
    document = project([source(wavelength_nm=wavelength_nm), element("L1", "lens", PCX_F100, (0.0, 0.0, 0.0), quaternion)],
                       [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)],
                       [{"target_id": "power", "metric": "power_W", "observation_id": "screen", "minimum": 0.0, "unit": "W"}])
    return run(document, **kwargs)


def test_the_codes_are_stable():
    assert MODEL_SCOPE_CODES == ("MODEL_WAVELENGTH_OUTSIDE_BAND", "COATING_WAVELENGTH_OUTSIDE_BANDS", "COATING_AOI_OUTSIDE_RANGE",
                                 "ELEMENT_AOI_OUTSIDE_RANGE", "MATERIAL_DATA_OUTSIDE_RANGE", "COATING_DATA_OUTSIDE_RANGE",
                                 "BEAM_PARTIALLY_INTERCEPTED", "RAY_LEFT_ELEMENT_THROUGH_EDGE", "RAY_CROSSES_ELEMENT_BODY",
                                 "ELEMENT_MET_FROM_BEHIND", "MODEL_FIELD_ANGLE_EXCEEDED", "MODEL_BEAM_SIZE_EXCEEDED",
                                 "MODEL_POLARIZATION_NOT_VALIDATED")
    assert AOI_TOLERANCE_DEG == 1e-6


@pytest.mark.parametrize("wavelength_nm", [400.0, 780.0, 1100.0])
def test_wavelengths_inside_the_band_including_its_edges_are_in_scope(wavelength_nm):
    result = _lens_run(wavelength_nm)
    assert result.model_scope_issues == () and result.status is LocalValidationStatusV1.UNKNOWN
    assert {target.status.value for target in result.result.targets} == {"PASS"}  # minimum 0 is bounded by nonnegative power
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.result.issues}


@pytest.mark.parametrize("wavelength_nm", [399.9, 1100.0000000000002, 1200.0])
def test_a_wavelength_outside_the_band_is_out_of_model_scope_and_names_the_published_ranges(wavelength_nm):
    result = _lens_run(wavelength_nm)
    assert scope_codes(result) == [
        ("L1", "MODEL_WAVELENGTH_OUTSIDE_BAND", None),
        ("L1", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S1"),
        ("L1", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S2"),
    ]
    # the Engine result is untouched: N-BK7 is published from 300 to 2500 nm, so it traced the beam normally
    assert result.engine_status.value == "UNKNOWN" and result.local_validation.simulation.status.value == "UNKNOWN"
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.result.issues}
    assert {target.status.value for target in result.result.targets} == {"PASS"}
    assert result.status is result.local_validation.validation_status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    message = result.model_scope_issues[0].message
    # the element by its instance id and product name; the SKU stays a structured fact of the run (simulated_parts)
    name = next(p["display_name"] for p in catalog_document()["products"] if p["optical_sku"] == PCX_F100)
    assert message.startswith(f"{wavelength_nm!r} nm light interacts with L1 ({name}), whose published model") and "400-1100 nm" in message
    assert PCX_F100 not in message and "SURFACE_SEQUENCE" not in message
    assert [p.optical_sku for p in result.local_validation.simulated_parts if p.instance_id == "L1"] == [PCX_F100]
    assert check_local_validation_binding(result.local_validation, result.project) == ()
    summary = result.summary()
    assert summary["model_scope_issues"] == [i.as_dict() for i in result.model_scope_issues]
    assert summary["model_scope_issues"][1]["surface_id"] == "S1" and "surface_id" not in summary["model_scope_issues"][0]


def test_light_beyond_a_materials_published_dispersion_is_stopped_by_the_engine_and_reported():
    result = _lens_run(2600.0)  # N-BK7 data ends at 2500 nm
    # MATERIAL_OUT_OF_RANGE: never extrapolated -- a data-coverage limit for the light behind the face (out of model scope)
    assert result.engine_status.value == "OUT_OF_MODEL_SCOPE"
    assert "MATERIAL_OUT_OF_RANGE" in {i.code.value for i in result.result.issues}
    assert ("L1", "MATERIAL_DATA_OUTSIDE_RANGE", "S1") in scope_codes(result)
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE


def test_a_tilted_lens_leaves_its_anti_reflection_coatings_angle_range():
    """The AR coating is characterized for 0-10 deg; the lens turned 20 deg about the vertical sees ~20 deg at S1."""

    result = _lens_run(780.0, quat_multiply(TILT_20_ABOUT_DESIGN_Z, ALONG_X))
    assert [(i, c, s) for i, c, s in scope_codes(result)] == [("L1", "COATING_AOI_OUTSIDE_RANGE", "S1"), ("L1", "COATING_AOI_OUTSIDE_RANGE", "S2")]
    s1 = result.model_scope_issues[0].message
    assert "0-10 deg" in s1 and "the coating of L1 (" in s1 and "surface S1" in s1 and "20 deg" in s1
    assert "ANTI_REFLECTION" not in s1  # the coating is named by its element and surface, not by its catalog id
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE


def test_the_angle_is_judged_as_its_equivalent_in_air():
    """Inside the plate splitter the chief ray meets the AR back face at ~27.7 deg, which is 45 deg in air: in scope."""

    document = project([source(wavelength_nm=633.0), element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("t", (60.0, 0.0, 0.0))])
    result = run(document)
    assert result.model_scope_issues == ()
    back_face = [s for p in result.result.paths for s in p.segments if s.end_surface_id == "BS/S1" and s.end_event == "refract"]
    (segment,) = back_face
    assert segment.refractive_index > 1.5 and 27.0 < segment.angle_of_incidence_deg < 28.5
    assert equivalent_air_aoi_deg(segment.angle_of_incidence_deg, segment.refractive_index) == pytest.approx(45.0, abs=1e-9)
    assert equivalent_air_aoi_deg(80.0, 1.5) == 90.0  # beyond the critical angle: capped at grazing


def _level_quaternion(axis):
    z = g.unit(axis)
    y = (0.0, 0.0, 1.0)
    return list(g.matrix_to_quat((g.cross(y, z), y, z)))


def _mirror_at(aoi_deg: float):
    theta = math.radians(aoi_deg)
    normal = (-math.cos(theta), math.sin(theta), 0.0)
    reflected = (-math.cos(2 * theta), math.sin(2 * theta), 0.0)
    document = project([source(position=(-100.0, 0.0, 0.0)), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), _level_quaternion(normal))],
                       [plane("screen", g.scale(reflected, 100.0), _level_quaternion(reflected))])
    return run(document)


@pytest.mark.parametrize("aoi_deg", [0.5, 30.0, 44.9999999])
def test_a_mirror_met_inside_its_coating_range_is_in_scope(aoi_deg):
    result = _mirror_at(aoi_deg)
    assert result.model_scope_issues == ()
    (reflect,) = [s for p in result.result.paths for s in p.segments if s.end_surface_id == "M1/S1" and s.end_event == "reflect"]
    assert reflect.angle_of_incidence_deg == pytest.approx(aoi_deg, abs=1e-6)


def test_an_exact_45_degree_fold_is_inside_a_0_to_45_degree_range():
    document = project([source(position=(-100.0, 0.0, 0.0)), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("screen", (0.0, 100.0, 0.0), _level_quaternion((0.0, 1.0, 0.0)))])
    result = run(document)
    assert result.model_scope_issues == () and len(arrivals(result.result, "screen")) == 1


@pytest.mark.parametrize("aoi_deg", [45.001, 50.0, 70.0])
def test_a_mirror_met_beyond_its_coating_range_is_out_of_model_scope(aoi_deg):
    result = _mirror_at(aoi_deg)
    assert scope_codes(result) == [("M1", "COATING_AOI_OUTSIDE_RANGE", "S1")]
    assert "0-45 deg" in result.model_scope_issues[0].message
    assert result.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE


def test_the_back_of_a_mirror_blocks_and_is_not_a_model_scope_question():
    document = project([source(position=(100.0, 0.0, 0.0), quaternion=(0.5, -0.5, -0.5, 0.5)),  # emitting along -X onto the back
                        element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("screen", (-50.0, 0.0, 0.0))])
    result = run(document)
    assert "BLOCKED_BY_BACK_SIDE" in {i.code.value for i in result.result.issues}
    assert result.model_scope_issues == ()


REVERSED_FOLD = quat_multiply((0.0, 0.0, 0.0, 1.0), FOLD_X_TO_Y)  # the same 45 deg fold, turned 180 deg about design Z
AGAINST_Y = (SQRT_HALF, SQRT_HALF, 0.0, 0.0)  # local +Z -> design -Y, local +Y -> design +Z
FOLD_Y_TO_X = (FOLD_B, FOLD_B, FOLD_A, FOLD_A)  # face normal -> normalize(X - Y): folds a +Y beam into +X
AR_T = 0.99  # the test catalog's anti-reflection back face (S1) of the dichroic and of the plate splitter
SPLITTER_COATING = "BEAMSPLITTER.400-1100R0.5T0.5.AOI45-45"


def _in_scope_even_with_unknown_loss(result) -> None:
    assert result.model_scope_issues == ()
    if result.status is LocalValidationStatusV1.UNKNOWN:
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.result.issues}
    else:
        assert result.status in (LocalValidationStatusV1.PASS, LocalValidationStatusV1.WARNING)


def _power_target(observation_id: str, source_id: str) -> dict:
    return {"target_id": f"{observation_id}-{source_id}", "metric": "power_W", "observation_id": observation_id, "source_id": source_id,
            "minimum": 0.0, "unit": "W"}


def _power(result, observation_id: str, source_id: str) -> float:
    """Known coating-only power or the geometrically traced upper bound with unstated glass absorption."""
    (arrival,) = arrivals(result.result, observation_id, source_id)
    if arrival.beam.power_W is None:
        assert arrival.power_upper_bound_W is not None
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.result.issues}
        return arrival.power_upper_bound_W
    return arrival.beam.power_W


def _split(sku, reflected_nm: float, transmitted_nm: float):
    """Two lines from -X meet the coated face at 45 deg: one reflected into +Y, one transmitted along +X."""

    return run(project([source("SRC_R", (-100.0, 0.0, 0.0), wavelength_nm=reflected_nm), source("SRC_T", (-100.0, 0.0, 0.0), wavelength_nm=transmitted_nm),
                        element("BS", "splitter", sku, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("r", (0.0, 60.0, 0.0), ALONG_Y), plane("t", (60.0, 0.0, 0.0))],
                       [_power_target("r", "SRC_R"), _power_target("t", "SRC_T")]))


def _combine(sku, reflected_nm: float, transmitted_nm: float, quaternion=FOLD_X_TO_Y, *, catalog=None, catalog_version=TEST_CATALOG_VERSION):
    """The exact time reversal of ``_split``: SRC_R travels -Y onto the element, SRC_T travels -X onto it, and both leave
    co-linear along -X onto one plane. With ``FOLD_X_TO_Y`` SRC_T meets the element from behind its coated face; turned
    180 deg (``REVERSED_FOLD``) SRC_R does."""

    document = project([source("SRC_R", (0.0, 100.0, 0.0), wavelength_nm=reflected_nm, quaternion=AGAINST_Y),
                        source("SRC_T", (100.0, 0.0, 0.0), wavelength_nm=transmitted_nm, quaternion=AGAINST_X),
                        element("BS", "splitter", sku, (0.0, 0.0, 0.0), quaternion, catalog_version=catalog_version)],
                       [plane("out", (-60.0, 0.0, 0.0), AGAINST_X)],
                       [_power_target("out", "SRC_R"), _power_target("out", "SRC_T")])
    return run(document, catalog=catalog) if catalog is not None else run(document)


@pytest.mark.parametrize(("sku", "reflected_nm", "transmitted_nm"), [(DICHROIC, 500.0, 780.0), (PLATE_BS, 780.0, 780.0)])
def test_a_combiner_is_the_time_reversed_splitter_and_in_model_scope(sku, reflected_nm, transmitted_nm):
    """A co-linear combiner always has one beam arriving from behind the coated face. Its transmitted branch is
    reciprocal and its reflected branch is determined by a lossless coating (R + T = 1 from either side), so the
    splitter, its exact time reversal and the same combiner turned around are all in model scope, with the splitter's
    powers (turned around, the reflected line also crosses the AR back face twice)."""

    split = _split(sku, reflected_nm, transmitted_nm)
    _in_scope_even_with_unknown_loss(split)
    combined = _combine(sku, reflected_nm, transmitted_nm)
    _in_scope_even_with_unknown_loss(combined)
    assert _power(combined, "out", "SRC_R") == pytest.approx(_power(split, "r", "SRC_R"), rel=1e-12)
    assert _power(combined, "out", "SRC_T") == pytest.approx(_power(split, "t", "SRC_T"), rel=1e-12)
    flipped = _combine(sku, reflected_nm, transmitted_nm, REVERSED_FOLD)
    _in_scope_even_with_unknown_loss(flipped)
    assert _power(flipped, "out", "SRC_R") == pytest.approx(_power(split, "r", "SRC_R") * AR_T**2, rel=1e-12)
    assert _power(flipped, "out", "SRC_T") == pytest.approx(_power(split, "t", "SRC_T"), rel=1e-12)


def test_a_plate_michelson_is_in_model_scope():
    """The plate splitter gets one arm back onto its coated face and the other through its back face."""

    document = project([source(position=(-100.0, 0.0, 0.0)), element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y),
                        element("M1", "mirror", FLAT_MIRROR, (0.0, 50.0, 0.0), AGAINST_Y),
                        element("M2", "mirror", FLAT_MIRROR, (50.0, 0.0, 0.0), AGAINST_X)],
                       [plane("det", (0.0, -60.0, 0.0), AGAINST_Y)], [_power_target("det", "SRC")])
    result = run(document)
    _in_scope_even_with_unknown_loss(result)
    assert len(arrivals(result.result, "det")) == 2  # both arms reach the detector


def test_a_plate_mach_zehnder_is_in_model_scope():
    """The recombining plate splitter BS2 meets the arm from MA from behind its coated face."""

    document = project([source(position=(-100.0, 0.0, 0.0)), element("BS1", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y),
                        element("MB", "mirror", FLAT_MIRROR, (0.0, 80.0, 0.0), FOLD_Y_TO_X),
                        element("MA", "mirror", FLAT_MIRROR, (80.0, 0.0, 0.0), FOLD_X_TO_Y),
                        element("BS2", "splitter", PLATE_BS, (80.0, 80.0, 0.0), FOLD_X_TO_Y)],
                       [plane("x_out", (140.0, 80.0, 0.0)), plane("y_out", (80.0, 140.0, 0.0), ALONG_Y)],
                       [_power_target("x_out", "SRC"), _power_target("y_out", "SRC")])
    result = run(document)
    _in_scope_even_with_unknown_loss(result)
    assert len(arrivals(result.result, "x_out")) == 2 and len(arrivals(result.result, "y_out")) == 2


@pytest.mark.parametrize("sku", [DICHROIC, PLATE_BS])
def test_a_transmitting_face_normal_element_met_from_behind_with_a_lossless_coating_is_in_scope(sku):
    """Mounted the wrong way round, a lossless dichroic or plate splitter is still in scope: its transmittance is the
    same from either side and its reflectance from behind equals the front one (R + T = 1)."""

    def splitter(quaternion):
        return run(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0), element("BS", "splitter", sku, (0.0, 0.0, 0.0), quaternion)],
                           [plane("far", (60.0, 0.0, 0.0), radius_mm=40.0)]))

    assert splitter(FOLD_X_TO_Y).model_scope_issues == ()
    assert splitter(REVERSED_FOLD).model_scope_issues == ()


def _lossy_splitter_catalog() -> dict:
    """The test catalog with the plate splitter's coating absorbing 5 %: R 0.45, T 0.5 (R + T < 1)."""

    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    coating = next(c for c in document["coatings"] if c["coating_id"] == SPLITTER_COATING)
    coating["bands"][0]["reflectance"] = "0.45"
    return document


def test_a_reflection_from_behind_a_lossy_coating_is_out_of_model_scope_and_names_the_branch():
    """The residual case: where the coating absorbs (R + T < 1) its reflectance for light from behind is not
    determined. The issue sits on the coated face (S2) and names the reflected branch; the right orientation stays in
    scope, and a beam meeting only the back of the element can be turned."""

    catalog = _lossy_splitter_catalog()

    def splitter(quaternion):
        document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0),
                            element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), quaternion, catalog_version=None)],
                           [plane("far", (60.0, 0.0, 0.0), radius_mm=40.0)])
        return run(document, catalog=catalog)

    assert splitter(FOLD_X_TO_Y).model_scope_issues == ()
    reversed_run = splitter(REVERSED_FOLD)
    assert scope_codes(reversed_run) == [("BS", "ELEMENT_MET_FROM_BEHIND", "S2")]
    assert reversed_run.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    (branch,) = [p for p in reversed_run.result.paths if p.parent_path_id is not None and p.branch_surface_id == "BS/S2"]
    message = reversed_run.model_scope_issues[0].message
    assert message.startswith(f"{branch.path_id} (780.0 nm) is reflected by BS (") and "surface S2" in message
    assert "reflectance + transmittance" in message
    assert "Turn the element so this beam meets the coated face first" in message


def test_a_lossy_combiner_is_never_told_to_turn_the_element():
    """Met from both faces, turning the element only moves the reflection from behind to the other beam, so the remedy
    is a lossless product."""

    catalog = _lossy_splitter_catalog()
    for quaternion, behind in ((FOLD_X_TO_Y, "SRC_T"), (REVERSED_FOLD, "SRC_R")):
        result = _combine(PLATE_BS, 780.0, 780.0, quaternion, catalog=catalog, catalog_version=None)
        assert scope_codes(result) == [("BS", "ELEMENT_MET_FROM_BEHIND", "S2")]
        message = result.model_scope_issues[0].message
        assert message.startswith(f"{behind}#") and "Turn the element" not in message and "lossless" in message


def test_a_reflection_at_the_back_face_from_outside_is_not_met_from_behind():
    """The rule judges the coated face only: light from behind the element reaching its back face S1 from outside
    meets S1's own front, whose published reflectance applies, even where S1 reflects and its band absorbs."""

    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    s1 = product_document(PLATE_BS, document)["simulation_models"][0]["surfaces"][0]
    assert s1["surface_id"] == "S1"
    s1["traced"] = "TRANSMITTED_AND_REFLECTED"
    ar = next(c for c in document["coatings"] if c["coating_id"] == s1["coating_id"])
    ar["bands"][0]["transmittance"] = "0.9"  # R 0.01 + T 0.9 < 1: an absorbing back face
    run_ = run(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0),
                        element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), REVERSED_FOLD, catalog_version=None)],
                       [plane("far", (60.0, 0.0, 0.0), radius_mm=40.0)]), catalog=document)
    assert any(p.branch_surface_id == "BS/S1" and p.branch_kind in ("coating_reflection", "beam_splitter_reflection") for p in run_.result.paths)
    assert run_.model_scope_issues == ()


def test_a_single_wavelength_band_matched_within_its_tolerance_is_checked_for_loss():
    """A lossy single-wavelength splitter band accepted through the wavelength tolerance (780 nm, light at 780.5 nm) still
    gets the reflection-from-behind check: its band is looked up with the same tolerance."""

    document = _lossy_splitter_catalog()
    coating = next(c for c in document["coatings"] if c["coating_id"] == SPLITTER_COATING)
    coating["bands"] = [dict(coating["bands"][0], min_nm="780", max_nm="780")]
    product_document(PLATE_BS, document)["simulation_models"][0]["valid_band_nm"] = {"min_nm": "780", "max_nm": "780", "status": "SPECIFIED"}

    def splitter(quaternion):
        return run(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.5),
                            element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), quaternion, catalog_version=None)],
                           [plane("far", (60.0, 0.0, 0.0), radius_mm=40.0)]), catalog=document)

    assert splitter(FOLD_X_TO_Y).model_scope_issues == ()
    assert scope_codes(splitter(REVERSED_FOLD)) == [("BS", "ELEMENT_MET_FROM_BEHIND", "S2")]


def test_a_polarization_element_has_its_own_angle_range():
    turned = quat_multiply(TILT_20_ABOUT_DESIGN_Z, ALONG_X)
    document = project([source(position=(-50.0, 0.0, 0.0)), element("POL", "polarizer", POLARIZER, (0.0, 0.0, 0.0), turned)],
                       [plane("out", (50.0, 0.0, 0.0), radius_mm=30.0)])
    result = run(document)
    assert scope_codes(result) == [("POL", "ELEMENT_AOI_OUTSIDE_RANGE", "element")]
    assert "0-10 deg" in result.model_scope_issues[0].message


def test_a_zero_order_waveplate_outside_its_band():
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=780.0), element("HWP", "waveplate", HWP)],
                       [plane("out", (50.0, 0.0, 0.0))])
    assert scope_codes(run(document)) == [("HWP", "MODEL_WAVELENGTH_OUTSIDE_BAND", None)]


def test_issues_are_aggregated_per_element_surface_and_code():
    """Two out-of-band sources through one lens: one issue per (surface, code), listing both wavelengths."""

    document = project([source("A", wavelength_nm=350.0), source("B", (-200.0, 3.0, 0.0), wavelength_nm=1200.0), element("L1", "lens", PCX_F100)],
                       [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)])
    result = run(document)
    assert [c for _, c, _ in scope_codes(result)] == ["MODEL_WAVELENGTH_OUTSIDE_BAND", "COATING_WAVELENGTH_OUTSIDE_BANDS", "COATING_WAVELENGTH_OUTSIDE_BANDS"]
    assert "350.0 nm, 1200.0 nm" in result.model_scope_issues[0].message


def _single_wavelength_hwp(*, multi_order: bool = False) -> dict:
    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    model = product_document(HWP, document)["simulation_models"][0]
    model["valid_band_nm"] = {"min_nm": "633", "max_nm": "633", "status": "SPECIFIED"}
    model["design_wavelength_nm"] = "633"
    if multi_order:  # a multi-order plate: its order is not published, so no RETARDANCE_INVERSE_WAVELENGTH approximation
        model["approximations"] = [a for a in model["approximations"] if a != "RETARDANCE_INVERSE_WAVELENGTH"]
    return document


def _hwp_scope(document: dict, wavelength_nm: float):
    doc = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=wavelength_nm), element("HWP", "waveplate", HWP, catalog_version=None)],
                  [plane("out", (50.0, 0.0, 0.0))])
    return run(doc, catalog=document)


def test_a_single_design_wavelength_is_matched_within_its_tolerance():
    """A "633 nm" zero-order waveplate: the HeNe line 632.8 nm (and 633.0000000000001 nm) is in scope; 650 nm is not."""
    document = _single_wavelength_hwp()
    for wavelength_nm in (633.0, 633.0000000000001, 632.8, 630.0, 636.0):
        assert scope_codes(_hwp_scope(document, wavelength_nm)) == [], wavelength_nm
    for wavelength_nm in (629.5, 636.5, 650.0):
        result = _hwp_scope(document, wavelength_nm)
        assert scope_codes(result) == [("HWP", "MODEL_WAVELENGTH_OUTSIDE_BAND", None)], wavelength_nm
        assert "633 nm (a single design wavelength, matched within +-0.5 %)" in result.model_scope_issues[0].message


def test_a_multi_order_waveplate_has_the_tighter_tolerance():
    document = _single_wavelength_hwp(multi_order=True)
    assert scope_codes(_hwp_scope(document, 632.8)) == []
    result = _hwp_scope(document, 633.5)
    assert scope_codes(result) == [("HWP", "MODEL_WAVELENGTH_OUTSIDE_BAND", None)]
    assert "matched within +-0.05 %" in result.model_scope_issues[0].message


def test_a_laser_line_coating_is_matched_within_its_tolerance_but_a_filter_is_not():
    """A single-wavelength band on a mirror's high reflector covers 632.8 nm; the same band on the dichroic stays exact."""
    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    for coating in document["coatings"]:
        if coating["kind"] in ("DIELECTRIC_HIGH_REFLECTOR", "DICHROIC"):
            coating["bands"] = [dict(coating["bands"][0], min_nm="633", max_nm="633")]
    for sku in (FLAT_MIRROR, CONCAVE_MIRROR, DICHROIC):  # a model band lies inside its coatings' bands
        product_document(sku, document)["simulation_models"][0]["valid_band_nm"] = {"min_nm": "633", "max_nm": "633", "status": "SPECIFIED"}
    mirror = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8),
                      element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None)],
                     [plane("out", (0.0, 60.0, 0.0), ALONG_Y)])
    assert [c for c in scope_codes(run(mirror, catalog=document)) if c[1] == "COATING_WAVELENGTH_OUTSIDE_BANDS"] == []
    dichroic = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8),
                        element("DM", "filter", DICHROIC, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None)],
                       [plane("t", (60.0, 0.0, 0.0))])
    assert ("DM", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S2") in scope_codes(run(dichroic, catalog=document))


def _turned_fold(degrees: float):
    half = math.radians(degrees / 2.0)
    return quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), FOLD_X_TO_Y)


@pytest.mark.parametrize("degrees, in_scope", [(0.5, True), (1.5, True), (3.0, False)])
def test_a_single_design_angle_of_a_splitter_coating_is_matched_within_2_degrees(degrees, in_scope):
    """The plate splitter's coatings are published for 45-45 deg: turned 0.5 or 1.5 deg it stays in scope, 3 deg it does not."""
    result = run(project([source(position=(-100.0, 0.0, 0.0)), element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), _turned_fold(degrees))],
                         [plane("t", (60.0, 0.0, 0.0), radius_mm=40.0)]))
    aoi_codes = [c for c in scope_codes(result) if c[1] == "COATING_AOI_OUTSIDE_RANGE"]
    assert (aoi_codes == []) is in_scope


def _dichroic_fold(degrees: float, *, catalog=None, wavelength_nm: float = 780.0):
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=wavelength_nm),
                        element("DM", "filter", DICHROIC, (0.0, 0.0, 0.0), _turned_fold(degrees), catalog_version=None if catalog else TEST_CATALOG_VERSION)],
                       [plane("t", (60.0, 0.0, 0.0), radius_mm=40.0)])
    return run(document, catalog=catalog) if catalog is not None else run(document)


def _aoi_codes(result):
    return [c for c in scope_codes(result) if c[1] == "COATING_AOI_OUTSIDE_RANGE"]


@pytest.mark.parametrize("degrees, in_scope", [(0.01, True), (0.5, True), (2.9, True), (-2.9, True), (3.2, False), (-3.2, False)])
def test_a_single_design_angle_of_a_dichroic_is_matched_within_its_edge_margin(degrees, in_scope):
    """The test dichroic (reflects 400-600 nm, transmits 650-1100 nm at 45 deg) leaves a 50 nm transition gap at
    625 nm; a quarter of it is 5.5 deg of tilt at n_eff 1.7, so its 45 deg point is matched within the 3 deg cap. Its
    anti-reflection back face (S1, also 45-45 deg) keeps the anti-reflection 3 deg though the product is a filter."""

    result = _dichroic_fold(degrees)
    if in_scope:
        assert _aoi_codes(result) == [] and result.model_scope_issues == ()
    else:
        assert ("DM", "COATING_AOI_OUTSIDE_RANGE", "S2") in _aoi_codes(result)
        message = next(i.message for i in result.model_scope_issues if i.surface_id == "S2")
        assert "characterized for 45-45 deg (a single design angle, matched within +-3 deg)" in message


def test_a_narrow_dichroic_edge_has_a_narrower_angle_tolerance():
    """The same dichroic with its transmit band from 610 nm: a 10 nm gap at 605 nm moves by a quarter of its width at 1.1
    deg of tilt, so 1 deg stays in scope and 1.5 deg does not."""

    from os_optics.local.validity import interference_angle_tolerance_deg  # noqa: PLC0415
    from os_optics._vendor.shared_contracts.design_catalog import PublicCoatingV1  # noqa: PLC0415

    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    coating = next(c for c in document["coatings"] if c["kind"] == "DICHROIC")
    coating["bands"][1]["min_nm"] = "610"
    assert interference_angle_tolerance_deg(PublicCoatingV1.model_validate(coating), 45.0) == 1.1
    assert _aoi_codes(_dichroic_fold(1.0, catalog=document)) == []
    result = _dichroic_fold(1.5, catalog=document)
    assert _aoi_codes(result) == [("DM", "COATING_AOI_OUTSIDE_RANGE", "S2")]
    assert "matched within +-1.1 deg" in result.model_scope_issues[0].message


def test_a_dichroic_without_a_transition_gap_keeps_its_angle_exactly():
    """With one band only, the published data state no edge whose margin bounds the tilt."""

    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    coating = next(c for c in document["coatings"] if c["kind"] == "DICHROIC")
    coating["bands"] = coating["bands"][1:]
    product_document(DICHROIC, document)["simulation_models"][0]["valid_band_nm"] = {"min_nm": "650", "max_nm": "1100", "status": "SPECIFIED"}
    assert ("DM", "COATING_AOI_OUTSIDE_RANGE", "S2") in _aoi_codes(_dichroic_fold(0.01, catalog=document))


def test_a_tilted_dichroic_combiner_is_in_model_scope():
    """The dichroic combiner turned 0.5 deg out of its exact 45 deg stays in scope."""

    combined = _combine(DICHROIC, 500.0, 780.0, _turned_fold(0.5))
    _in_scope_even_with_unknown_loss(combined)


def test_an_element_the_beam_misses_is_not_judged():
    document = project([source(wavelength_nm=1200.0), element("L1", "lens", PCX_F100, (0.0, 40.0, 0.0))],  # 40 mm beside the beam
                       [plane("screen", (100.0, 0.0, 0.0))])
    assert run(document).model_scope_issues == ()


def test_the_dichroic_gap_is_the_only_issue_of_a_gap_wavelength():
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=625.0), element("DM", "filter", DICHROIC, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("t", (60.0, 0.0, 0.0))])
    result = run(document)
    assert scope_codes(result) == [("DM", "COATING_WAVELENGTH_OUTSIDE_BANDS", "S2")]


# ------------------------------------------------------------- design-point tolerances are caller settings


def _splitter_fold(degrees: float, design_point: dict | None = None) -> dict:
    document = project([source(position=(-100.0, 0.0, 0.0)), element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), _turned_fold(degrees))],
                       [plane("t", (60.0, 0.0, 0.0), radius_mm=40.0), plane("r", (0.0, 60.0, 0.0), ALONG_Y, radius_mm=40.0)],
                       [{"target_id": "through", "metric": "power_W", "observation_id": "t", "minimum": 1e-4, "unit": "W"}])
    if design_point is not None:
        document["numerical_settings"] = {"design_point": design_point}
    return document


def _beams(result):
    return [(o.observation_id, o.total_power_W, [a.model_dump(mode="json") for a in o.arrivals]) for o in result.result.observations]


def test_a_project_stated_angle_tolerance_keeps_the_run_in_scope_with_the_same_beams():
    """The plate splitter's coatings are published for 45 deg; the Engine defaults match them within 2 deg (the splitting
    face) and 3 deg (the anti-reflection back face). Turned 4 deg, the default run is out of model scope at both faces; a
    Project that states 5 deg for both kinds stays in scope. The Engine computes the same beams either way (the published
    coefficients, held): only the findings, the validation status and the Engine input (the declared slack) differ."""

    catalog = with_lossless_bulk(catalog_document())
    default = run(_splitter_fold(4.0), catalog=catalog)
    stated = run(_splitter_fold(4.0, {"single_angle_deg": {"beamsplitter": 5, "anti_reflection": 5}}), catalog=catalog)
    assert sorted(c for c in scope_codes(default) if c[1] == "COATING_AOI_OUTSIDE_RANGE") == [
        ("BS", "COATING_AOI_OUTSIDE_RANGE", "S1"), ("BS", "COATING_AOI_OUTSIDE_RANGE", "S2")]
    assert default.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    assert stated.model_scope_issues == () and stated.status is LocalValidationStatusV1.PASS
    assert _beams(default) == _beams(stated)  # bit for bit
    assert default.local_validation.input_hash != stated.local_validation.input_hash
    tolerances = stated.result.manifest.numerical_tolerances
    assert tolerances["design_point_angle_beamsplitter_deg"] == tolerances["design_point_angle_anti_reflection_deg"] == 5.0
    assert default.result.manifest.numerical_tolerances["design_point_angle_beamsplitter_deg"] == 2.0
    # the target behind the splitter: assessed in the stated run, withheld by the finding in the default one
    assert [t.assessment for t in stated.local_validation.simulation.target_outcomes] == ["MET"]
    assert [t.assessment for t in default.local_validation.simulation.target_outcomes] == ["INDETERMINATE"]


def test_a_finding_at_a_single_design_angle_carries_the_point_the_tolerance_and_who_set_it():
    default = run(_splitter_fold(4.0))
    by_surface = {i.surface_id: i.details["design_point"] for i in default.model_scope_issues if i.code == "COATING_AOI_OUTSIDE_RANGE"}
    split, back = by_surface["S2"], by_surface["S1"]
    assert split == {"quantity": "angle_of_incidence_deg", "design_values": [45.0], "observed": split["observed"], "reading": "AIR_EQUIVALENT",
                     "tolerance_deg": 2.0, "coating_kind": "BEAMSPLITTER", "setting": "design_point_angle_beamsplitter_deg",
                     "source": "ENGINE_DEFAULT", "assumed": "HELD_AT_NEAREST_STATED_POINT"}
    assert len(split["observed"]) == 1 and abs(abs(split["observed"][0] - 45.0) - 4.0) < 1e-6  # the fold turned 4 deg
    assert (back["tolerance_deg"], back["coating_kind"], back["setting"], back["source"]) == (
        3.0, "ANTI_REFLECTION", "design_point_angle_anti_reflection_deg", "ENGINE_DEFAULT")
    assert abs(abs(back["observed"][0] - 45.0) - 4.0) < 1e-6  # read as its equivalent in air, like the front
    # a Project tolerance that still does not reach the angle: the finding names the Project as its source
    stated = run(_splitter_fold(4.0, {"single_angle_deg": {"beamsplitter": 3.5, "anti_reflection": 5}}))
    (issue,) = stated.model_scope_issues
    facts = issue.details["design_point"]
    assert (issue.surface_id, facts["tolerance_deg"], facts["source"]) == ("S2", 3.5, "PROJECT")
    assert "(a single design angle, matched within +-3.5 deg)" in issue.message
    # every setting a finding names is recorded with the run
    for run_ in (default, stated):
        for found in run_.model_scope_issues:
            assert found.details["design_point"]["setting"] in run_.result.manifest.numerical_tolerances
    # the run lists the facts; the Local Validation record holds the finding's applicability fields only
    assert stated.summary()["model_scope_issues"][0]["details"] == issue.details
    recorded = stated.local_validation.simulation.model_scope_issues[0].model_dump(mode="json", exclude_none=True)
    from os_optics.local.record import LISTING_ONLY_FIELDS

    listing_only = {"details", "validity", "affected_arrivals", "affected_sampled_lines", "validity_reach"}
    assert LISTING_ONLY_FIELDS == listing_only and "validity_reach" in issue.as_dict()
    assert not listing_only & recorded.keys() and recorded == {k: v for k, v in issue.as_dict().items() if k not in listing_only}
    # the finding's validity check: the published point as its range, the angle the light met, the setting of its tolerance
    (check,) = issue.validity
    assert (check.parameter, check.minimum, check.maximum, check.within, check.setting, check.code) == (
        "angle_of_incidence_deg", 45.0, 45.0, False, "design_point_angle_beamsplitter_deg", "COATING_AOI_OUTSIDE_RANGE")
    assert check.observed == facts["observed"][0] and check.at[0].startswith("BS")
    # a finding that is not at a single design point has no details (and its row keeps its keys)
    mirror = _mirror_at(50.0)  # the mirror's coating is published for a range, 0-45 deg
    assert [i.details for i in mirror.model_scope_issues] == [None] and "details" not in mirror.model_scope_issues[0].as_dict()


def test_a_run_directory_with_a_design_point_finding_verifies(tmp_path):
    from os_optics.local import verify_run_artifacts, write_run_artifacts  # noqa: PLC0415

    result = run(_splitter_fold(4.0))
    write_run_artifacts(result, tmp_path / "run")
    assert verify_run_artifacts(tmp_path / "run") == ()
    import json  # noqa: PLC0415

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]["model_scope_issues"] == [i.as_dict() for i in result.model_scope_issues]
    assert all("design_point" in row["details"] for row in manifest["status"]["model_scope_issues"])


def test_the_interference_tolerance_reads_the_projects_shift_fraction_and_cap():
    """The test dichroic's 45 deg point: about 5.5 deg of tilt moves its 50 nm gap by a quarter, so the 3 deg cap binds by
    default; a Project cap of 6 deg leaves the shift fraction as the limit, and a smaller fraction narrows it."""

    # the limit from the thin-film angle shift lambda(theta) = lambda0 sqrt(1 - sin^2(theta) / n_eff^2), n_eff = 1.7: the
    # larger angle at which the spectrum has moved down by a quarter of the 600-650 nm gap (relative width 50 / 625)
    ratio, n2 = 1.0 - 0.25 * 50.0 / 625.0, 1.7 ** 2
    limit = math.degrees(math.asin(math.sqrt(n2 * (1.0 - ratio ** 2 * (1.0 - math.sin(math.radians(45.0)) ** 2 / n2))))) - 45.0
    assert 5.4 < limit < 5.5

    default = _dichroic_fold(3.2)
    facts = next(i.details["design_point"] for i in default.model_scope_issues if i.surface_id == "S2")
    assert (facts["tolerance_deg"], facts["setting"], facts["source"], facts["coating_kind"]) == (
        3.0, "design_point_interference_angle_max_deg", "ENGINE_DEFAULT", "DICHROIC")
    assert "assumed" not in facts  # an interference coating beyond its limit: its published value does not stand for the light's

    def fold(degrees, design_point):
        document = project([source(position=(-100.0, 0.0, 0.0)), element("DM", "filter", DICHROIC, (0.0, 0.0, 0.0), _turned_fold(degrees))],
                           [plane("t", (60.0, 0.0, 0.0), radius_mm=40.0)])
        document["numerical_settings"] = {"design_point": design_point}
        return run(document)

    wide = {"interference_angle_max_deg": 6, "single_angle_deg": {"anti_reflection": 6}}
    assert fold(5.0, wide).model_scope_issues == ()
    (issue,) = fold(5.8, wide).model_scope_issues
    facts = issue.details["design_point"]
    assert (issue.surface_id, facts["tolerance_deg"], facts["setting"], facts["source"]) == (
        "S2", 5.4, "design_point_interference_shift_fraction", "ENGINE_DEFAULT")  # the limit, rounded down to 0.1 deg
    narrow = fold(2.0, {"interference_shift_fraction": 0.05})
    facts = next(i.details["design_point"] for i in narrow.model_scope_issues if i.surface_id == "S2")
    assert facts["setting"] == "design_point_interference_shift_fraction" and facts["source"] == "PROJECT"
    assert 0.0 < facts["tolerance_deg"] < 2.0


def test_a_finding_at_a_single_design_wavelength_carries_the_point_and_a_project_tolerance_moves_it():
    document = _single_wavelength_hwp()
    result = _hwp_scope(document, 650.0)
    (issue,) = result.model_scope_issues
    assert issue.details == {"design_point": {
        "quantity": "wavelength_nm", "design_values": [633.0], "observed": [650.0], "tolerance_rel": 0.005,
        "setting": "design_point_wavelength_rel", "source": "ENGINE_DEFAULT", "assumed": "HELD_AT_NEAREST_STATED_POINT"}}
    # a multi-order plate: its own setting, and no held value (its retardance is off by an unknown part of a wave)
    (issue,) = _hwp_scope(_single_wavelength_hwp(multi_order=True), 633.5).model_scope_issues
    assert issue.details == {"design_point": {
        "quantity": "wavelength_nm", "design_values": [633.0], "observed": [633.5], "tolerance_rel": 0.0005,
        "setting": "design_point_retarder_wavelength_rel", "source": "ENGINE_DEFAULT"}}

    def stated(wavelength_nm, design_point, catalog=document):
        doc = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=wavelength_nm), element("HWP", "waveplate", HWP, catalog_version=None)],
                      [plane("out", (50.0, 0.0, 0.0))])
        doc["numerical_settings"] = {"design_point": design_point}
        return run(doc, catalog=catalog)

    wide = stated(650.0, {"single_wavelength_rel": 0.03})  # 650 nm is 2.7 % from 633 nm
    assert wide.model_scope_issues == ()
    assert _beams(wide) == _beams(result)
    (issue,) = stated(660.0, {"single_wavelength_rel": 0.03}).model_scope_issues
    assert issue.details["design_point"]["source"] == "PROJECT" and "matched within +-3 %" in issue.message
    # the multi-order tolerance is its own setting: the zero-order one does not reach it
    assert scope_codes(stated(633.5, {"single_wavelength_rel": 0.03}, _single_wavelength_hwp(multi_order=True))) == [
        ("HWP", "MODEL_WAVELENGTH_OUTSIDE_BAND", None)]
    assert stated(633.5, {"multi_order_retarder_rel": 0.001}, _single_wavelength_hwp(multi_order=True)).model_scope_issues == ()


def test_a_compiled_element_is_never_reused_under_other_tolerances():
    """One cache of compiled elements across two lowerings whose Projects state different tolerances: each request carries
    its own declared slack, and each element the tolerances it was compiled under."""

    from os_optics.local import lower_project  # noqa: PLC0415
    from os_optics.local.design_point import DEFAULT_DESIGN_POINT  # noqa: PLC0415
    from os_optics.local.catalog import load_public_catalog  # noqa: PLC0415

    catalog = load_public_catalog(catalog_document())
    cache: dict = {}

    def slack(lowered):
        component = next(c for c in lowered.request.project.components if c.instance_id == "BS")
        return {s.surface_id: s.interaction.coating.declared_validity.slack_deg for s in component.surfaces}

    first = lower_project(_splitter_fold(4.0), catalog, compiled=cache)
    second = lower_project(_splitter_fold(4.0, {"single_angle_deg": {"beamsplitter": 5}}), catalog, compiled=cache)
    third = lower_project(_splitter_fold(4.0), catalog, compiled=cache)
    assert slack(first) == slack(third) == {"S1": AOI_TOLERANCE_DEG + 3.0, "S2": AOI_TOLERANCE_DEG + 2.0}
    assert slack(second) == {"S1": AOI_TOLERANCE_DEG + 3.0, "S2": AOI_TOLERANCE_DEG + 5.0}
    assert first.engine_input_hash == third.engine_input_hash != second.engine_input_hash
    assert first.elements[0].design_point == DEFAULT_DESIGN_POINT == first.design_point
    assert second.elements[0].design_point == second.design_point and second.design_point.angle_deg(
        second.elements[0].envelopes[1].coating.kind) == 5.0
    assert second.design_point.source("design_point_angle_beamsplitter_deg") == "PROJECT"
    assert len(cache) == 1  # the latest lowering's element only


def test_the_check_before_the_run_reads_the_projects_wavelength_tolerance():
    from os_optics.local import json_api  # noqa: PLC0415

    def warnings(design_point):
        doc = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=650.0), element("HWP", "waveplate", HWP, catalog_version=None)],
                      [plane("out", (50.0, 0.0, 0.0))])
        if design_point is not None:
            doc["numerical_settings"] = {"design_point": design_point}
        reply = json_api.check_project(doc, catalog=_single_wavelength_hwp())
        assert reply["ok"], reply
        return [w["code"] for w in reply["warnings"]]

    assert warnings(None) == ["SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND"]
    assert warnings({"single_wavelength_rel": 0.03}) == []
