"""Native classes built on the native primitives: axicon, wedge prism, hollow roof mirror, hollow corner-cube
retroreflector, variable neutral-density filter and tabulated thin lens. Each class builds a contract-valid model and is
traced through the local Engine path, and every result is checked against a reference computed here: Snell's law with
an independent Sellmeier evaluation (axicon deflection, wedge deviation, thin-prism limit), the reflection matrices of
plane mirrors (roof: diag(-1, -1, 1) in the design frame used here, corner cube: -I), ``10^-OD`` of the stated law at
the crossing point, and Gaussian ABCD propagation for the tabulated lens.

Fixture data: every reflectance, density law, table row and evidence record below is made up for the test.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics._vendor.shared_contracts.design_catalog import ConvexBodyV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.local import json_api, run_optical_project
from os_optics.native_library import NativeComponentError, add_to_project, build_native_model, describe_native_class
from os_optics.native_library import trig

from local_helpers import ALONG_X, plane, project, source
from native_classes_b import CLASSES_B
from test_native_library import N_BK7_COEFFS, sellmeier
from test_tabulated_thin_lens import PLANE_X, ROWS, SOURCE_X, transmittance, waist_after_lens

C45 = math.sqrt(0.5)
#: A unit whose local +Z points along design -X: light from a source at -X travels along the unit's -Z.
FACING_SOURCE = (C45, 0.0, -C45, 0.0)
BAND = [400, 1100]



def nbk7(wavelength_nm: float) -> float:
    return sellmeier(N_BK7_COEFFS, wavelength_nm)


def _place(build, instance_id: str, *, position=(0.0, 0.0, 0.0), quaternion=ALONG_X, src=None, planes=None, targets=None) -> dict:
    # (in the air the native library designs for, 20 C and 101325 Pa: the independent Snell references use ``nbk7``, the
    # index relative to that air)
    document = project([src or source(position=(-200.0, 0.0, 0.0), waist_mm=0.3)], planes or [], targets, environment=None)
    return add_to_project(document, build, instance_id=instance_id, position_mm=position, quaternion_wxyz=quaternion)


def _leaving(run, surface_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(start, unit direction) of the chief-ray segment that leaves ``surface_id``."""

    path = run.result.paths[0]
    for k, segment in enumerate(path.segments):
        if segment.end_surface_id == surface_id:
            after = path.segments[k + 1]
            d = np.subtract(after.end_mm, after.start_mm)
            return np.array(after.start_mm), d / np.linalg.norm(d)
    raise AssertionError(f"the chief ray does not leave {surface_id}")


# ------------------------------------------------------------------------------------------------- every class


@pytest.mark.parametrize("class_id", sorted(CLASSES_B))
def test_each_new_class_builds_describes_itself_and_enters_a_project(class_id):
    build = build_native_model(class_id, CLASSES_B[class_id])
    description = describe_native_class(class_id)
    assert description["model_kind"] == build.optical_model.model.model_kind
    assert description["availability"] == ["LOCAL_SIMULATION", "CUSTOMER_SUPPLIED"] and description["references"]
    ProjectDocumentV2.model_validate(_place(build, "X"))
    reply = json_api.build_native_model(class_id, CLASSES_B[class_id])
    assert reply["ok"] and reply["optical_model"]["model_id"] == build.optical_model.model_id
    assert json_api.build_native_model(class_id, CLASSES_B[class_id]) == reply  # exact decimals: the same bytes again


def test_catalog_equivalents_name_the_published_forms():
    forms = {cid: [f for e in describe_native_class(cid)["catalog_equivalents"] for f in e["forms"]] for cid in CLASSES_B}
    assert forms == {"lens.axicon": ["AXICON"], "prism.wedge": ["WEDGE_PRISM"], "filter.variable_nd": ["NEUTRAL_DENSITY_FILTER"],
                     "mirror.hollow_roof": ["RIGHT_ANGLE_PRISM"], "retroreflector.hollow_corner": [], "lens.tabulated_thin": []}


def test_decimal_trigonometry_matches_libm():
    for angle in ("0.5", "5", "11.3", "30", "44.9", "60", "89"):
        s, c = trig.sin_cos_deg(Decimal(angle))
        assert float(s) == pytest.approx(math.sin(math.radians(float(angle))), abs=1e-15)
        assert float(c) == pytest.approx(math.cos(math.radians(float(angle))), abs=1e-15)
    assert float(trig.asin_deg(Decimal("0.5"))) == pytest.approx(30.0, abs=1e-13)
    assert str(trig.pi())[:20] == "3.141592653589793238"


# ------------------------------------------------------------------------------------------------------- axicon


@pytest.mark.parametrize("height", [2.0, 6.0])
@pytest.mark.parametrize("orientation", ["PLANE_FIRST", "CONE_FIRST"])
def test_the_axicon_deflects_a_parallel_ray_by_snell(height, orientation):
    alpha = math.radians(10.0)
    build = build_native_model("lens.axicon", {**CLASSES_B["lens.axicon"], "orientation": orientation})
    document = _place(build, "AX", src=source(position=(-200.0, 0.0, height), wavelength_nm=780.0, waist_mm=0.3),
                      planes=[plane("P", (60.0, 0.0, 0.0), radius_mm=25.0)])
    start, d = _leaving(run_optical_project(document), "AX/S2")
    n = nbk7(780.0)
    if orientation == "PLANE_FIRST":
        beta = math.asin(n * math.sin(alpha)) - alpha
    else:  # refraction into the cone, then out of the plane face
        beta = math.asin(n * math.sin(alpha - math.asin(math.sin(alpha) / n)))
    assert math.acos(float(d[0])) == pytest.approx(beta, abs=1e-12)
    assert d[2] < 0.0 and abs(d[1]) < 1e-15  # toward the axis, in the meridional plane
    assert float(build.design["deflection_deg"]) == pytest.approx(math.degrees(beta), abs=1e-9)
    if orientation == "PLANE_FIRST":  # the cone's sag at the hit: z = t - tan(alpha) h
        assert start[0] == pytest.approx(float(build.design["center_thickness_mm"]) - math.tan(alpha) * height, abs=1e-9)


def test_a_given_axicon_thickness_needs_a_positive_edge():
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.axicon", {**CLASSES_B["lens.axicon"], "center_thickness_mm": "2"})  # 12.7 tan 10 deg = 2.24 mm
    assert exc.value.problems[0].parameter == "center_thickness_mm"
    total = build_native_model("lens.axicon", {**CLASSES_B["lens.axicon"], "cone_angle_deg": "35", "center_thickness_mm": "12"})
    assert float(total.design["deflection_deg"]) > 0  # n sin 35 deg < 1 for N-BK7: transmitted
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.axicon", {**CLASSES_B["lens.axicon"], "cone_angle_deg": "45", "center_thickness_mm": "20",
                                           "design_wavelength_nm": "780", "material": {"constant_index": "1.5", "valid_band_nm": [400, 900]}})
    assert exc.value.problems[0].parameter == "cone_angle_deg"  # 1.5 sin 45 deg > 1: totally reflected


# -------------------------------------------------------------------------------------------------------- wedge


@pytest.mark.parametrize(("angle", "wavelength"), [("2", 780.0), ("5", 780.0), ("10", 500.0)])
def test_the_wedge_deviates_toward_its_thick_end_by_snell(angle, wavelength):
    alpha = math.radians(float(angle))
    build = build_native_model("prism.wedge", {**CLASSES_B["prism.wedge"], "wedge_angle_deg": angle,
                                               "design_wavelength_nm": f"{wavelength:g}"})
    # local Z along design +X, local X along design +Y: the thick end (local -X) is toward design -Y
    document = _place(build, "W", src=source(position=(-200.0, 0.0, 0.0), wavelength_nm=wavelength, waist_mm=0.3),
                      planes=[plane("P", (60.0, 0.0, 0.0), radius_mm=25.0)])
    _start, d = _leaving(run_optical_project(document), "W/S2")
    n = nbk7(wavelength)
    delta = math.asin(n * math.sin(alpha)) - alpha
    assert math.acos(float(d[0])) == pytest.approx(delta, abs=1e-12)
    assert d[1] < 0.0 and abs(d[2]) < 1e-15
    assert delta == pytest.approx((n - 1.0) * alpha, rel=0.02 * float(angle))  # the thin-prism limit for small angles
    assert float(build.design["deviation_deg"]) == pytest.approx(math.degrees(delta), abs=1e-9)
    out = [float(v) for v in build.optical_model.frame.port("OUT").direction]  # local: (-sin delta, 0, cos delta)
    assert out == pytest.approx([-math.sin(delta), 0.0, math.cos(delta)], abs=1e-12)


def test_the_wedge_frame_origin_is_on_its_axis_at_the_centroid_height():
    physics = build_native_model("prism.wedge", CLASSES_B["prism.wedge"]).optical_model.model
    _volume, (x, y, z) = ConvexBodyV1.model_validate(physics.body.model_dump(mode="json")).volume_and_centroid()
    assert abs(float(z)) < 1e-6 and float(y) == 0.0  # positions are written to 1e-6 mm
    assert float(x) < 0.0  # the centroid itself lies toward the thick end (local -X), off the axis


# ----------------------------------------------------------------------------------------------- retroreflectors


def _retro_document(class_id: str, *, position=(-200.0, 3.0, 2.0), direction=None) -> dict:
    """The unit at the origin facing the source; a non-absorbing plane at x = -100 sees the beam going in and coming back.
    A tilted beam (``direction`` = (yaw about design Z, pitch about design Y)) still crosses x = 0 at ``position``'s (y, z)."""
    quaternion = list(ALONG_X)
    if direction is not None:  # turn the source about design Z (yaw) then Y (pitch)
        yaw, pitch = direction
        qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        quaternion = _qmul(qy, _qmul(qz, ALONG_X))
        d = (math.cos(pitch) * math.cos(yaw), math.sin(yaw), -math.sin(pitch) * math.cos(yaw))
        position = (position[0], position[1] + position[0] * d[1] / d[0], position[2] + position[0] * d[2] / d[0])
    build = build_native_model(class_id, CLASSES_B[class_id])
    src = source(position=position, waist_mm=0.3, quaternion=quaternion)
    return _place(build, "R", quaternion=FACING_SOURCE, src=src, planes=[plane("P", (-100.0, 0.0, 0.0), radius_mm=40.0)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _in_and_out(run):
    """((point, direction) in, (point, direction) out) at the x = -100 plane, in design coordinates (plane u = +Y, v = +Z,
    n = +X)."""
    crossings = sorted(run.result.observations[0].arrivals, key=lambda a: a.crossing_index)
    assert len(crossings) == 2
    out = []
    for arrival in crossings:
        u, v = arrival.uv_mm
        du, dv, dn = arrival.direction_uvn
        out.append((np.array([-100.0, u, v]), np.array([dn, du, dv])))
    return out


def test_the_hollow_roof_returns_the_beam_and_reverses_one_transverse_coordinate():
    run = run_optical_project(_retro_document("mirror.hollow_roof"))
    (p_in, d_in), (p_out, d_out) = _in_and_out(run)
    # two plane reflections about planes through the ridge (design Z axis through the origin): diag(-1, -1, 1)
    flip = np.diag([-1.0, -1.0, 1.0])
    assert d_out == pytest.approx(flip @ d_in, abs=1e-12) and d_out == pytest.approx(-d_in, abs=1e-12)
    assert p_out[1] == pytest.approx(-p_in[1], abs=1e-9) and p_out[2] == pytest.approx(p_in[2], abs=1e-9)
    reflections = [e for e in run.result.paths[0].segments if e.end_surface_id and e.end_surface_id.startswith("R/")]
    assert [s.end_surface_id for s in reflections] == ["R/A-S1", "R/B-S1"]
    assert run.result.paths[0].final_power_W == pytest.approx(0.001, rel=1e-12)
    # a ray tilted out of the plane across the ridge keeps that component: not antiparallel
    tilted = run_optical_project(_retro_document("mirror.hollow_roof", direction=(0.0, math.radians(3.0))))
    (_p, d_in), (_q, d_out) = _in_and_out(tilted)
    assert d_out == pytest.approx(flip @ d_in, abs=1e-12) and abs(d_out[2] - d_in[2]) < 1e-12 and abs(d_in[2]) > 0.05


@pytest.mark.parametrize("direction", [None, (math.radians(4.0), 0.0), (0.0, math.radians(-5.0)), (math.radians(-3.0), math.radians(6.0))])
@pytest.mark.parametrize("position", [(-200.0, 3.0, 2.0), (-200.0, -4.0, 1.0)])
def test_the_hollow_corner_returns_every_ray_antiparallel(direction, position):
    run = run_optical_project(_retro_document("retroreflector.hollow_corner", position=position, direction=direction))
    (p_in, d_in), (p_out, d_out) = _in_and_out(run)
    assert d_out == pytest.approx(-d_in, abs=1e-12)
    # three reflections in planes through the apex (the origin): the returning ray is the incoming one reflected through it
    s = (p_out[0] + p_in[0]) / -d_in[0]
    assert p_out == pytest.approx(-p_in - s * d_in, abs=2e-6)  # member poses are written to 1e-6 mm
    hits = [e.end_surface_id for e in run.result.paths[0].segments if e.end_surface_id and e.end_surface_id.startswith("R/")]
    assert sorted(hits) == ["R/A-S1", "R/B-S1", "R/C-S1"]


# --------------------------------------------------------------------------------------------------- variable ND


def _nd_power(parameters: dict, beam_yz) -> float:
    build = build_native_model("filter.variable_nd", parameters)
    target = [{"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0, "unit": "W"}]
    document = _place(build, "ND", src=source(position=(-100.0, *beam_yz), wavelength_nm=632.8, waist_mm=0.2),
                      planes=[plane("det", (50.0, *beam_yz), radius_mm=20.0)], targets=target)
    (result,) = run_optical_project(document).result.targets
    return result.value


@pytest.mark.parametrize(("radius", "azimuth", "position", "expected"), [
    (15.0, 90.0, "30", 10.0 ** -(0.04 + 1.96 * 60.0 / 270.0)),  # the wheel turned by 30 deg: law azimuth 60 deg
    (15.0, 0.0, "0", 10.0 ** -0.04),
    (15.0, 300.0, "0", 0.92),                                     # beyond the coated arc
    (3.0, 120.0, "0", 0.92),                                      # in the uncoated hub
])
def test_the_variable_nd_wheel_transmits_its_law_at_the_position(radius, azimuth, position, expected):
    beam = (radius * math.cos(math.radians(azimuth)), radius * math.sin(math.radians(azimuth)))  # local (X, Y) = design (y, z)
    parameters = {**CLASSES_B["filter.variable_nd"], "position": position}
    assert _nd_power(parameters, beam) == pytest.approx(0.001 * expected, rel=1e-9)
    build = build_native_model("filter.variable_nd", parameters)
    assert [p.semantic_key for p in build.component_template.selected_parameters] == ["nd_rotation_deg"]


def test_the_variable_nd_gradient_shifts_with_its_position():
    gradient = {"law": "LINEAR_GRADIENT", "od_start": "0.5", "od_end": "2.5", "x_start_mm": "-10", "x_end_mm": "10", "width_mm": "40",
                "height_mm": "20", "valid_band_nm": [400, 700], "position": "2.5"}
    assert _nd_power(gradient, (3.0, 0.0)) == pytest.approx(0.001 * 10.0 ** -(0.5 + 2.0 * (3.0 - 2.5 + 10.0) / 20.0), rel=1e-9)
    build = build_native_model("filter.variable_nd", gradient)
    (selected,) = build.component_template.selected_parameters
    assert (selected.semantic_key, selected.unit_code) == ("nd_offset_mm", "mm")
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("filter.variable_nd", {**gradient, "arc_end_deg": "270"})
    assert exc.value.problems[0].parameter == "arc_end_deg"


# --------------------------------------------------------------------------------------------- tabulated thin lens


def test_the_tabulated_lens_focuses_each_line_with_its_own_focal_length():
    build = build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "model_id": "ml"})
    from test_local_spectral_lines import multi_line_source

    lines = ((500.0, 0.001), (532.0, 0.002), (560.0, 0.0015))
    src = multi_line_source(lines=lines, position=(SOURCE_X, 0.0, 0.0), waist_mm=0.5)
    document = _place(build, "ML", src=src, planes=[plane("P", (PLANE_X, 0.0, 0.0), radius_mm=20.0)])
    document["environment"] = {"pressure_pa": "0"}  # (the independent Gaussian reference propagates in a unit-index medium)
    run = run_optical_project(document)
    arrivals = {a.beam.wavelength_nm: a for a in run.result.observations[0].arrivals}
    for wavelength, power in lines:
        g = arrivals[wavelength].beam.gaussian
        expected_z, expected_w0 = waist_after_lens(wavelength)
        assert PLANE_X - g.waist_distance_x_mm == pytest.approx(expected_z, rel=1e-9)
        assert g.waist_radius_x_mm == pytest.approx(expected_w0, rel=1e-9)
        assert arrivals[wavelength].beam.power_W == pytest.approx(power * transmittance(wavelength), rel=1e-12)
    # the same table given as columns builds the same model
    columns = {k: [r[k] for r in CLASSES_B["lens.tabulated_thin"]["rows"]] for k in ("wavelength_nm", "focal_length_mm", "transmittance")}
    again = build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "model_id": "ml", "rows": columns})
    assert again.as_dict() == build.as_dict()


def test_the_tabulated_lens_builds_from_the_table_alone():
    """Field angle, validated metrics and evidence are stated when the author has them: the model built without them is
    the same model less those statements, an assumed response, and focuses a line exactly as the stated one does."""

    full_parameters = {**CLASSES_B["lens.tabulated_thin"], "model_id": "ml"}
    optional = ("max_field_angle_deg", "supported_metrics", "evidence")
    full = build_native_model("lens.tabulated_thin", full_parameters)
    bare = build_native_model("lens.tabulated_thin", {k: v for k, v in full_parameters.items() if k not in optional})
    physics, full_physics = bare.as_dict()["optical_model"]["model"], full.as_dict()["optical_model"]["model"]
    assert not set(optional) & set(physics) and physics == {k: v for k, v in full_physics.items() if k not in optional}
    assert bare.as_dict()["design"]["supported_metrics"] is None
    described = {p["name"]: p for p in describe_native_class("lens.tabulated_thin")["parameters"]}
    assert [described[name]["required"] for name in ("rows", *optional)] == [True, False, False, False]
    with pytest.raises(NativeComponentError) as exc:  # a stated value keeps its rules
        build_native_model("lens.tabulated_thin", {**full_parameters, "evidence": {"method": "fixture"}})
    assert exc.value.problems[0].parameter == "evidence"
    src = source(wavelength_nm=532.0, power_w=0.001, position=(SOURCE_X, 0.0, 0.0), waist_mm=0.5)
    documents = [_place(build, "ML", src=src, planes=[plane("P", (PLANE_X, 0.0, 0.0), radius_mm=20.0)]) for build in (full, bare)]
    # The independent Gaussian reference propagates in a unit-index medium.
    runs = [run_optical_project({**document, "environment": {"pressure_pa": "0"}}) for document in documents]
    (a,), (b,) = (run.result.observations[0].arrivals for run in runs)
    assert b.model_dump() == a.model_dump()
    assert PLANE_X - b.beam.gaussian.waist_distance_x_mm == pytest.approx(waist_after_lens(532.0)[0], rel=1e-9)
    (row,) = [r for r in runs[1].summary()["element_models"] if r["instance_id"] == "ML"]
    assert row["response"]["data_kind"] == "ASSUMED"


def test_a_tabulated_lens_with_a_bad_row_names_the_parameter():
    rows = copy.deepcopy(CLASSES_B["lens.tabulated_thin"]["rows"])
    rows[1]["transmittance"] = "1.2"
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "rows": rows})
    assert (exc.value.problems[0].parameter, "row 1") == ("rows", exc.value.problems[0].message[:5])
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "validated_polarizations": ["CIRCULAR"]})
    assert exc.value.problems[0].parameter == "validated_polarizations"


# ------------------------------------------------------------------------------------------ angle-resolved AR coating


def _ar_table(t0: str, t60: str) -> dict:
    t = [[t0, t60], [t0, t60]]
    r = [[f"{1 - float(t0):g}", f"{1 - float(t60):g}"]] * 2
    return {"wavelength_nm": ["600", "900"], "aoi_deg": ["0", "60"], "reflectance_s": r, "reflectance_p": r, "transmittance_s": t,
            "transmittance_p": t}


@pytest.mark.parametrize("tilt_deg", [0.0, 30.0])
def test_a_window_with_an_angle_resolved_ar_coating_transmits_the_table_at_each_face(tilt_deg):
    window = build_native_model("plate.window", {"material": "N-BK7", "thickness_mm": "5", "diameter_mm": "25.4",
                                                 "ar_coating": {"angular_spectrum": _ar_table("0.99", "0.9")}})
    (coating,) = window.model_coatings
    assert coating.is_angle_resolved and (coating.valid_aoi_deg.min_deg, coating.valid_aoi_deg.max_deg) == ("0", "60")
    half = math.radians(tilt_deg) / 2  # the window turned about design Z
    quaternion = _qmul((math.cos(half), 0.0, 0.0, math.sin(half)), ALONG_X)
    target = [{"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0, "unit": "W"}]
    document = _place(window, "W", quaternion=quaternion, src=source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0, waist_mm=0.3),
                      planes=[plane("det", (60.0, 0.0, 0.0), radius_mm=25.0)], targets=target)
    # Isolate the published angle-resolved coating table: this test Project explicitly assumes ideal lossless glass.
    # The named native N-BK7 model alone does not state measured or zero bulk absorption.
    for material in document["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    (result,) = run_optical_project(document).result.targets

    def t_at(aoi_deg: float) -> float:  # the table is the same at every wavelength: linear in angle
        return 0.99 + (0.9 - 0.99) * aoi_deg / 60.0

    inside = math.degrees(math.asin(math.sin(math.radians(tilt_deg)) / nbk7(780.0)))  # the exit face, read in the glass
    assert result.value == pytest.approx(0.001 * t_at(tilt_deg) * t_at(inside), rel=1e-9)
