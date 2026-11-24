"""Native component library (``os_optics.native_library``): every class builds a contract-valid Project optical model,
and the physics of the built models is checked against independent references written here (float arithmetic, own
Sellmeier evaluation, own paraxial y-nu trace, own exact ray trace, SciPy's PCHIP) -- never against the library's own
decimal code."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from os_optics._vendor.shared_contracts.optical_design import (
    OpticalComponentInstanceV1,
    ProjectDocumentV2,
    ProjectOpticalModelV1,
    check_optical_models,
)
from os_optics.local import json_api
from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index
from native_classes_b import CLASSES_B
from test_native_crystals import CLASSES_CRYSTAL
from os_optics.native_library import (
    NativeComponentError,
    add_to_project,
    build_native_model,
    describe_native_class,
    native_class_ids,
    native_components,
)

# SCHOTT N-SF5 Sellmeier coefficients (as published in the public catalog) declared inline: the fixture catalog has no flint.
N_SF5 = {"sellmeier": {"b": ["1.52481889", "0.187085527", "1.42729015"], "c_um2": ["0.011254756", "0.0588995392", "129.141675"]},
         "valid_band_nm": ["370", "2500"], "material_id": "N-SF5", "display_name": "N-SF5"}
N_BK7_COEFFS = ((1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653))
N_SF5_COEFFS = ((1.52481889, 0.187085527, 1.42729015), (0.011254756, 0.0588995392, 129.141675))
FDC = (486.1327, 587.5618, 656.2725)


# ----------------------------------------------------------------------------------------- independent references


def sellmeier(coefficients, wavelength_nm: float) -> float:
    """The index relative to the design air (native designs are made for air at 20 C and 101325 Pa, the default
    environment, ``air_index`` of the shared contracts): the absolute Sellmeier index / n_air."""
    b, c = coefficients
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c))) / air_index(wavelength_nm)


def material_index(material: dict, wavelength_nm: float) -> float:
    """Relative to the design air, as ``sellmeier`` (these materials state no index reference: absolute, vacuum wavelength)."""
    dispersion = material["dispersion"]
    if dispersion["kind"] == "CONSTANT_INDEX":
        return float(dispersion["refractive_index"]) / air_index(wavelength_nm)
    if dispersion["kind"] == "SELLMEIER":
        return sellmeier(([float(v) for v in dispersion["b"]], [float(v) for v in dispersion["c_um2"]]), wavelength_nm)
    table = PchipInterpolator([float(v) for v in dispersion["wavelength_nm"]], [float(v) for v in dispersion["refractive_index"]])
    return float(table(wavelength_nm)) / air_index(wavelength_nm)


def lens_matrix(build: dict, wavelength_nm: float, model: dict | None = None) -> tuple[np.ndarray, float, float]:
    """(reduced ABCD from first to last vertex, first z, last z) of a SURFACE_SEQUENCE model by a float y-nu trace."""

    model = model or build["optical_model"]
    physics = model["model"]
    materials = {m["material_id"]: m for m in build["model_materials"]}
    media = ["AIR", *physics["media"], "AIR"]

    def n(medium: str) -> float:
        return 1.0 if medium == "AIR" else material_index(materials[medium], wavelength_nm)

    matrix = np.eye(2)
    previous_z = None
    for k, surface in enumerate(physics["surfaces"]):
        z = float(surface["vertex_z_mm"])
        if previous_z is not None:
            matrix = np.array([[1.0, (z - previous_z) / n(media[k])], [0.0, 1.0]]) @ matrix
        if "radius_mm" in surface:
            power = (n(media[k + 1]) - n(media[k])) / float(surface["radius_mm"])
            matrix = np.array([[1.0, 0.0], [-power, 1.0]]) @ matrix
        previous_z = z
    return matrix, float(physics["surfaces"][0]["vertex_z_mm"]), float(physics["surfaces"][-1]["vertex_z_mm"])


def efl(build: dict, wavelength_nm: float) -> float:
    return -1.0 / lens_matrix(build, wavelength_nm)[0][1, 0]


def edge_thickness(front_radius: float | None, back_radius: float | None, thickness: float, edge_radius: float) -> float:
    def s(radius):
        if radius is None:
            return 0.0
        return radius - math.copysign(math.sqrt(radius * radius - edge_radius * edge_radius), radius)

    return thickness + s(back_radius) - s(front_radius)


def radius(surface: dict) -> float | None:
    return float(surface["radius_mm"]) if "radius_mm" in surface else None


def model_of(class_id: str, parameters: dict) -> dict:
    return build_native_model(class_id, parameters).as_dict()


#: A gradient-index material (test values, not a product's): n0 = 1.6, g = 0.3 / mm.
GRIN_MATERIAL = {"material_id": "GRIN", "display_name": "Test gradient glass", "basis": "test values",
                 "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.6"},
                 "valid_band_nm": {"min_nm": "400", "max_nm": "1700", "status": "SPECIFIED"},
                 "gradient": {"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": "0.3"}}
SINGLET = {"focal_length_mm": "100", "design_wavelength_nm": "632.8", "material": "N-BK7", "diameter_mm": "25.4"}

ALL_CLASSES = {
    "aperture.circular": {"opening_diameter_mm": "0.05"},
    "aperture.slit": {"slit_width_mm": "0.1", "slit_height_mm": "5", "opening_center_mm": ["0.5", "0"]},
    "aperture.annular": {"opening_diameter_mm": "10", "obstruction_diameter_mm": "3"},
    "aperture.central_obstruction": {"obstruction_diameter_mm": "2"},
    "lens.ideal_thin": {"focal_length_mm": "50", "clear_aperture_diameter_mm": "20"},
    "lens.singlet": {**SINGLET, "form": "BEST_FORM", "ar_coating": {"reflectance": "0.005", "valid_aoi_deg": [0, 10]}},
    "lens.achromat": {"focal_length_mm": "100", "crown_material": "N-BK7", "flint_material": N_SF5, "diameter_mm": "25.4"},
    "plate.window": {"material": {"constant_index": "1.33", "valid_band_nm": [400, 900], "absorption_coefficient_per_mm": "0.01"},
                     "thickness_mm": "5", "diameter_mm": "25", "ar_coating": "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10"},
    "plate.layered": {"layers": [{"material": "FUSED_SILICA", "thickness_mm": "1.25"},
                                 {"material": {"constant_index": "1.333", "valid_band_nm": [400, 900], "material_id": "WATER-NOMINAL"},
                                  "thickness_mm": "10"},
                                 {"material": "FUSED_SILICA", "thickness_mm": "1.25"}], "width_mm": "12.5", "height_mm": "45"},
    "mirror.flat": {"diameter_mm": "25.4", "reflectance": "0.98", "valid_band_nm": [400, 700]},
    "mirror.curved": {"diameter_mm": "25.4", "focal_length_mm": "100", "reflectance": "0.98", "valid_band_nm": [400, 700], "conic": "-1"},
    "splitter.plate": {"diameter_mm": "25.4", "reflectance_s": "0.9", "reflectance_p": "0.1", "valid_band_nm": [600, 700]},
    "rotator.faraday": {"rows": [{"wavelength_nm": "1030", "rotation_deg": "48", "transmittance": "0.93"},
                                 {"wavelength_nm": "1090", "rotation_deg": "42", "transmittance": "0.94"}],
                        "diameter_mm": "10", "valid_aoi_deg": ["0", "5"], "operating_conditions": "fixture: 25 C",
                        "evidence": {"method": "fixture", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "ab" * 32,
                                     "fit_model": "the stated rows", "validation_summary": "fixture", "date": "2026-10-30"}},
    "deflector.acousto_optic": {"acoustic_velocity_m_per_s": "650", "rf_tones_mhz": ["80", "90"], "width_mm": "10", "height_mm": "10"},
    "grating.plane": {"mode": "REFLECTION", "line_density_per_mm": "600", "orders": [-1, 0, 1], "width_mm": "25", "height_mm": "25"},
    "polarizer.linear": {"transmittance": "0.9", "extinction_ratio": "10000", "diameter_mm": "12.7", "valid_band_nm": [400, 700]},
    "waveplate.retarder": {"retardance_waves": "0.25", "design_wavelength_nm": "633", "diameter_mm": "12.7", "valid_band_nm": [600, 660]},
    "compound.keplerian_telescope": {
        "lens1": {"class_id": "lens.singlet", "parameters": {**SINGLET, "focal_length_mm": "50", "diameter_mm": "12.7", "form": "PLANO_EXIT"}},
        "lens2": {"class_id": "lens.achromat", "parameters": {"focal_length_mm": "150", "crown_material": "N-BK7", "flint_material": N_SF5,
                                                              "diameter_mm": "25.4"}},
        "design_wavelength_nm": "632.8", "focal_stop_diameter_mm": "0.03"},
    "compound.galilean_telescope": {
        "lens1": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "-25", "clear_aperture_diameter_mm": "10"}},
        "lens2": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "100", "clear_aperture_diameter_mm": "30"}}},
    "compound.relay_4f": {
        "lens1": {"class_id": "lens.singlet", "parameters": {**SINGLET, "form": "PLANO_EXIT"}},
        "lens2": {"class_id": "lens.singlet", "parameters": {**SINGLET, "focal_length_mm": "200", "form": "PLANO_ENTRY"}},
        "fourier_stop_diameter_mm": "2", "fourier_obstruction_diameter_mm": "0.2"},
    **CLASSES_B,  # axicon, wedge prism, hollow roof / corner, variable ND, tabulated thin lens (test_native_library_classes_b.py)
    **CLASSES_CRYSTAL,  # beam displacer, Glan, Glan-Thompson, Wollaston, Rochon (test_native_crystals.py)
    "lens.grin_rod": {"material": GRIN_MATERIAL, "design_wavelength_nm": "633", "diameter_mm": "1.8", "pitch": "0.25"},
}

SOURCE = {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
          "nominal_pose": {"position_mm": [-100.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
          "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 0.5,
                           "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}


def empty_project() -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": [SOURCE]}


# ------------------------------------------------------------------------------------------------ registry / contract


def test_registry_lists_every_class_with_described_parameters():
    ids = native_class_ids()
    assert ids == sorted(ALL_CLASSES)
    for description in native_components():
        assert description["model_kind"] in {"APERTURE_STOP", "IDEAL_THIN_LENS", "SURFACE_SEQUENCE", "POLARIZATION_ELEMENT", "COMPOUND",
                                             "SURFACE_SET", "NEUTRAL_DENSITY", "TABULATED_THIN_LENS", "FARADAY_ROTATOR", "DIFFRACTION_GRATING"}
        assert description["references"] and description["physics"] and description["approximations"]
        names = [p["name"] for p in description["parameters"]]
        assert len(names) == len(set(names)) and {"model_id", "display_name", "instance_id"} <= set(names)
        for parameter in description["parameters"]:
            assert parameter["description"] and parameter["type"]
            assert not (parameter["required"] and ("default" in parameter or "default_rule" in parameter)), parameter
    assert describe_native_class("lens.singlet")["optical_role"] == "lens"


@pytest.mark.parametrize("class_id", sorted(ALL_CLASSES))
def test_every_class_builds_a_contract_valid_model_that_a_project_accepts(class_id):
    build = build_native_model(class_id, ALL_CLASSES[class_id])
    document = build.as_dict()
    models = [ProjectOpticalModelV1.model_validate(m) for m in (*document["member_models"], document["optical_model"])]
    assert models[-1].model.model_kind == describe_native_class(class_id)["model_kind"]
    check_optical_models(tuple(models), build.model_materials, build.model_coatings)
    template = OpticalComponentInstanceV1.model_validate(document["component_template"])
    assert template.optical_model_id == build.optical_model.model_id and template.product is None
    project = ProjectDocumentV2.model_validate(add_to_project(empty_project(), build, instance_id="X1", position_mm=(10, 0, 0)))
    assert project.optical_model(build.optical_model.model_id) is not None
    assert [c.instance_id for c in project.components] == ["SRC", "X1"]
    # exact, canonical decimal strings: rebuilding gives the same bytes
    assert json.dumps(build_native_model(class_id, ALL_CLASSES[class_id]).as_dict(), sort_keys=True) == json.dumps(document, sort_keys=True)


@pytest.mark.parametrize("class_id", ["aperture.circular", "lens.ideal_thin", "mirror.flat", "lens.singlet", "compound.relay_4f"])
def test_the_json_build_copies_into_a_project_verbatim(class_id):
    """The documented recipe: each part of the reply goes into its Project field as it is (a stop has no materials and no
    coatings: null, which the Project contract accepts where an empty list is refused)."""
    built = json_api.build_native_model(class_id, ALL_CLASSES[class_id])
    project = empty_project()
    project["optical_models"] = [*built["member_models"], built["optical_model"]]
    project["model_materials"] = built["model_materials"]
    project["model_coatings"] = built["model_coatings"]
    project["components"].append(built["component_template"])
    ProjectDocumentV2.model_validate(project)
    if class_id == "aperture.circular":
        assert built["model_materials"] is None and built["model_coatings"] is None
    # the same through options["into"]: the Project with everything added and checked
    merged = json_api.build_native_model(class_id, ALL_CLASSES[class_id], {"into": empty_project(), "instance_id": "X1",
                                                                          "position_mm": [10, 0, 0]})
    assert [c["instance_id"] for c in merged["project"]["components"]] == ["SRC", "X1"]
    assert ("model_materials" in merged["project"]) == (built["model_materials"] is not None)
    refused = json_api.build_native_model(class_id, ALL_CLASSES[class_id], {"into": empty_project(), "unknown": 1})
    assert refused["ok"] is False and refused["error"]["problems"][0]["parameter"] == "options"


def test_the_same_component_added_twice_shares_its_model_and_conflicts_are_refused():
    build = build_native_model("lens.singlet", SINGLET)
    project = add_to_project(add_to_project(empty_project(), build, instance_id="L1"), build, instance_id="L2")
    assert [m["model_id"] for m in project["optical_models"]] == [build.optical_model.model_id]
    other = build_native_model("lens.singlet", {**SINGLET, "focal_length_mm": "120", "model_id": build.optical_model.model_id})
    with pytest.raises(NativeComponentError) as exc:
        add_to_project(project, other, instance_id="L3")
    assert exc.value.problems[0].parameter == "model_id"
    with pytest.raises(NativeComponentError) as exc:
        add_to_project(project, build, instance_id="L1")
    assert exc.value.problems[0].parameter == "instance_id"


# --------------------------------------------------------------------------------------------------------- singlets


@pytest.mark.parametrize("variant", [
    {"form": "SYMMETRIC"}, {"form": "PLANO_EXIT"}, {"form": "PLANO_ENTRY"}, {"form": "BEST_FORM"}, {"shape_factor": "2.5"},
    {"focal_length_mm": "-75", "form": "PLANO_EXIT"}, {"focal_length_mm": "-50", "form": "SYMMETRIC"},
    {"focal_length_mm": "30", "diameter_mm": "12.7", "design_wavelength_nm": "1064"},
    {"material": {"constant_index": "1.7", "valid_band_nm": [400, 1100]}, "focal_length_mm": "40"},
    {"center_thickness_mm": "6", "form": "SYMMETRIC"},
])
def test_singlet_focal_length_and_paraxial_block_match_an_independent_trace(variant):
    parameters = {**SINGLET, **variant}
    build = model_of("lens.singlet", parameters)
    wavelength, focal = float(parameters["design_wavelength_nm"]), float(parameters["focal_length_mm"])
    matrix, _first, last = lens_matrix(build, wavelength)
    a, _b, c, d = matrix.ravel()
    assert -1.0 / c == pytest.approx(focal, rel=1e-7)  # radii are written to 1e-6 mm
    paraxial = build["optical_model"]["model"]["paraxial"]
    assert float(paraxial["effective_focal_length_mm"]) == pytest.approx(-1.0 / c, abs=2e-6)
    assert float(paraxial["back_focal_length_mm"]) == pytest.approx(-a / c, abs=2e-6)
    assert float(paraxial["front_focal_length_mm"]) == pytest.approx(-d / c, abs=2e-6)
    assert float(paraxial["rear_principal_plane_z_mm"]) == pytest.approx(last - a / c + 1.0 / c, abs=2e-6)
    surfaces = build["optical_model"]["model"]["surfaces"]
    if variant.get("form") == "PLANO_EXIT":
        assert "radius_mm" not in surfaces[1]
    if variant.get("form") == "PLANO_ENTRY":
        assert "radius_mm" not in surfaces[0]


def test_singlet_shape_factor_and_best_form_follow_their_definitions():
    for variant, expected in (({"form": "SYMMETRIC"}, 0.0), ({"shape_factor": "2.5"}, 2.5)):
        surfaces = model_of("lens.singlet", {**SINGLET, **variant})["optical_model"]["model"]["surfaces"]
        c1, c2 = 1 / float(surfaces[0]["radius_mm"]), 1 / float(surfaces[1]["radius_mm"])
        assert (c1 + c2) / (c1 - c2) == pytest.approx(expected, abs=1e-7)
    n = sellmeier(N_BK7_COEFFS, 632.8)
    surfaces = model_of("lens.singlet", {**SINGLET, "form": "BEST_FORM"})["optical_model"]["model"]["surfaces"]
    c1, c2 = 1 / float(surfaces[0]["radius_mm"]), 1 / float(surfaces[1]["radius_mm"])
    assert (c1 + c2) / (c1 - c2) == pytest.approx(2 * (n * n - 1) / (n + 2), abs=1e-7)


def _thick_curvatures(power: float, n: float, t: float, factor: float) -> tuple[float, float]:
    """Independent solve of the thick-lens equation for the curvatures at a shape factor (float, quadratic formula)."""

    k = n - 1
    a2 = k * k * t * (factor * factor - 1) / (4 * n)
    a = power / k if a2 == 0 else (-k + math.sqrt(k * k + 4 * a2 * power)) / (2 * a2)
    return a * (factor + 1) / 2, a * (factor - 1) / 2


@pytest.mark.parametrize("variant", [{"form": "SYMMETRIC"}, {"form": "PLANO_EXIT"}, {"focal_length_mm": "-60", "form": "SYMMETRIC"}])
def test_thickening_rule_gives_the_thinnest_micrometre_thickness_meeting_both_minimums(variant):
    parameters = {**SINGLET, **variant}
    build = model_of("lens.singlet", parameters)
    physics = build["optical_model"]["model"]
    t = float(physics["surfaces"][1]["vertex_z_mm"])
    diameter = float(parameters["diameter_mm"])
    te_min, tc_min = max(0.5, 0.05 * diameter), max(1.0, 0.08 * diameter)
    r_edge = diameter / 2
    edge = edge_thickness(radius(physics["surfaces"][0]), radius(physics["surfaces"][1]), t, r_edge)
    assert t >= tc_min - 1e-12 and edge >= te_min - 1e-6
    thinner = t - 0.001
    n = sellmeier(N_BK7_COEFFS, 632.8)
    factor = float(build["design"]["shape_factor"])
    c1, c2 = _thick_curvatures(1 / float(parameters["focal_length_mm"]), n, thinner, factor)
    thinner_edge = edge_thickness(1 / c1 if c1 else None, 1 / c2 if c2 else None, thinner, r_edge)
    assert thinner < tc_min or thinner_edge < te_min
    design_geometry = build["optical_model"]["design_geometry"]
    assert float(design_geometry["diameter_mm"]) == diameter


def test_tabulated_catalog_glass_uses_pchip_like_the_engine(real_catalog_store):
    parameters = {**SINGLET, "material": "H-K9L", "design_wavelength_nm": "700.5"}
    build = model_of("lens.singlet", parameters)
    assert build["model_materials"][0]["dispersion"]["kind"] == "TABULATED"
    n_ref = material_index(build["model_materials"][0], 700.5)  # SciPy PchipInterpolator, as the Engine evaluates tables
    assert float(build["design"]["refractive_index"]) == pytest.approx(n_ref, abs=1e-12)
    assert efl(build, 700.5) == pytest.approx(100.0, rel=1e-7)


def test_cylindrical_singlet_has_power_in_its_curved_section_only():
    build = model_of("lens.singlet", {**SINGLET, "cylinder_axis": "X", "form": "PLANO_EXIT"})
    physics = build["optical_model"]["model"]
    assert physics["surfaces"][0]["cylinder_axis"] == "X" and physics["paraxial"]["axis"] == "Y"
    assert build["optical_model"]["frame"]["symmetry"] == "NONE"
    assert efl(build, 632.8) == pytest.approx(100.0, rel=1e-7)


def test_stigmatic_rear_conic_focuses_a_collimated_beam_to_one_point():
    """Exact real-ray trace (own code): rays parallel to the axis at heights up to the clear radius cross the axis at one
    point for the k = -n^2 exit surface, and spread for the plain sphere."""

    def crossings(build: dict) -> list[float]:
        physics = build["optical_model"]["model"]
        n = sellmeier(N_BK7_COEFFS, 632.8)
        rear = physics["surfaces"][1]
        t, c = float(rear["vertex_z_mm"]), 1 / float(rear["radius_mm"])
        k = float(rear.get("conic", "0"))
        out = []
        for h in np.linspace(0.05, float(rear["clear_radius_mm"]), 12):
            root = math.sqrt(1 - (1 + k) * c * c * h * h)
            z = t + c * h * h / (1 + root)
            slope = c * h / root  # dz/dr of the surface
            normal = np.array([-slope, 1.0]) / math.hypot(slope, 1.0)  # (r, z), toward +z
            incident = np.array([0.0, 1.0])
            cos_i = float(normal @ incident)
            eta = n / 1.0
            cos_t = math.sqrt(1 - eta * eta * (1 - cos_i * cos_i))
            refracted = eta * incident + (cos_t - eta * cos_i) * normal
            out.append(z + (-h / refracted[0]) * refracted[1])
        return out

    conic = model_of("lens.singlet", {**SINGLET, "form": "PLANO_ENTRY", "rear_conic": "STIGMATIC_COLLIMATED"})
    sphere = model_of("lens.singlet", {**SINGLET, "form": "PLANO_ENTRY"})
    n = sellmeier(N_BK7_COEFFS, 632.8)
    assert float(conic["optical_model"]["model"]["surfaces"][1]["conic"]) == pytest.approx(-n * n, abs=1e-11)
    focus = crossings(conic)
    paraxial_focus = float(conic["optical_model"]["model"]["surfaces"][1]["vertex_z_mm"]) + float(
        conic["optical_model"]["model"]["paraxial"]["back_focal_length_mm"])
    assert max(focus) - min(focus) < 1e-9
    assert focus[0] == pytest.approx(paraxial_focus, abs=2e-6)
    spread = crossings(sphere)
    assert max(spread) - min(spread) > 1e-3  # spherical aberration of the plain sphere


# --------------------------------------------------------------------------------------------------------- achromat


def bfl(build: dict, wavelength_nm: float) -> float:
    """Back focal length (last vertex to the rear focal point) by the float y-nu trace: -A / C."""

    matrix = lens_matrix(build, wavelength_nm)[0]
    return -matrix[0, 0] / matrix[1, 0]


@pytest.mark.parametrize("shape_factor", ["0", "1", "-2"])
def test_achromat_brings_the_short_and_long_wavelength_foci_together_independent_trace(shape_factor):
    """The design condition is coincident F and C foci of a collimated beam (equal back focal lengths),
    the axial achromat condition of a focusing doublet -- not equal EFLs, which left them 10 / 74 / -128 um apart for
    these three crown shapes. Checked by the float y-nu trace written here; the design report carries the true axial focus
    shift and the remaining EFL difference under their own names."""

    build = model_of("lens.achromat", {**ALL_CLASSES["lens.achromat"], "crown_shape_factor": shape_factor})
    f_short, f_center, f_long = (efl(build, w) for w in FDC)
    b_short, b_long = bfl(build, FDC[0]), bfl(build, FDC[2])
    assert f_center == pytest.approx(100.0, rel=1e-7)
    assert abs(b_long - b_short) < 2e-6  # radii written to 1e-6 mm; the f/4 depth of focus is about +-0.018 mm
    singlet = model_of("lens.singlet", {**SINGLET, "design_wavelength_nm": "587.5618"})
    singlet_shift = bfl(singlet, FDC[2]) - bfl(singlet, FDC[0])
    assert abs(singlet_shift) > 1.0 and abs(b_long - b_short) < 1e-5 * abs(singlet_shift)
    design = build["design"]
    assert "chromatic_focal_shift_long_minus_short_mm" not in design
    assert float(design["axial_focus_shift_long_minus_short_mm"]) == pytest.approx(b_long - b_short, abs=2e-6)
    assert float(design["efl_difference_long_minus_short_mm"]) == pytest.approx(f_long - f_short, abs=2e-6)
    assert (float(design["bfl_short_mm"]), float(design["bfl_long_mm"])) == pytest.approx((b_short, b_long), abs=2e-6)
    assert abs(f_long - f_short) > 1e-3  # a thick doublet with coincident foci keeps a small EFL difference
    edges = design["edge_thicknesses_mm"]
    assert all(float(e) >= max(0.5, 0.05 * 25.4) - 1e-6 for e in edges)


def test_achromat_without_coincident_foci_names_the_ways_out():
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.achromat", {**ALL_CLASSES["lens.achromat"], "focal_length_mm": "40"})
    problem = exc.value.problems[0]
    assert (problem.code.value, problem.parameter) == ("NATIVE_DESIGN_INFEASIBLE", "focal_length_mm")
    assert "foci together" in problem.message and "crown_shape_factor" in problem.message


def test_achromat_thin_lens_limit_matches_the_thin_achromat_equations():
    """With thicknesses tending to zero the built element powers equal the thin-lens achromat split (phi1/V1 + phi2/V2 = 0)
    and dEFL/dlambda vanishes at the design condition (a stationary focal length inside the band)."""

    parameters = {"focal_length_mm": "100", "crown_material": "N-BK7", "flint_material": N_SF5, "diameter_mm": "0.5",
                  "crown_center_thickness_mm": "0.01", "flint_center_thickness_mm": "0.01", "min_edge_thickness_mm": "0.001"}
    build = model_of("lens.achromat", parameters)
    n1 = [sellmeier(N_BK7_COEFFS, w) for w in FDC]
    n2 = [sellmeier(N_SF5_COEFFS, w) for w in FDC]
    v1, v2 = (n1[1] - 1) / (n1[0] - n1[2]), (n2[1] - 1) / (n2[0] - n2[2])
    phi = 1 / 100.0
    phi1, phi2 = phi * v1 / (v1 - v2), -phi * v2 / (v1 - v2)
    r1, r2, r3 = (float(s["radius_mm"]) for s in build["optical_model"]["model"]["surfaces"])
    assert (n1[1] - 1) * (1 / r1 - 1 / r2) == pytest.approx(phi1, rel=1e-3)
    assert (n2[1] - 1) * (1 / r2 - 1 / r3) == pytest.approx(phi2, rel=1e-3)
    step = 0.5

    def slope(document: dict, wavelength: float) -> float:
        return (efl(document, wavelength + step) - efl(document, wavelength - step)) / (2 * step)

    singlet = model_of("lens.singlet", {**SINGLET, "design_wavelength_nm": "587.5618"})
    # coincident end foci; in the thin-lens limit the end focal lengths are then equal too, so the mean dEFL/dlambda over
    # [short, long] vanishes ...
    assert abs(bfl(build, FDC[2]) - bfl(build, FDC[0])) < 1e-5 * abs(bfl(singlet, FDC[2]) - bfl(singlet, FDC[0]))
    assert abs(efl(build, FDC[2]) - efl(build, FDC[0])) < 1e-4 * abs(efl(singlet, FDC[2]) - efl(singlet, FDC[0]))
    # ... so dEFL/dlambda vanishes inside the band (it changes sign between the end wavelengths), and at the centre
    # wavelength only the secondary spectrum is left: an order of magnitude below a singlet's slope
    assert slope(build, FDC[0]) * slope(build, FDC[2]) < 0
    assert abs(slope(build, FDC[1])) < 0.1 * abs(slope(singlet, FDC[1]))


# ---------------------------------------------------------------------------------------------------- compound units


def _member_matrix(document: dict, member_model: dict, wavelength: float) -> tuple[np.ndarray, float, float]:
    physics = member_model["model"]
    if physics["model_kind"] == "IDEAL_THIN_LENS":
        return np.array([[1.0, 0.0], [-1.0 / float(physics["focal_length_mm"]), 1.0]]), 0.0, 0.0
    return lens_matrix(document, wavelength, member_model)


def _unit_system(document: dict, wavelength: float) -> tuple[np.ndarray, list[float], float]:
    """System matrix from the first lens's first vertex to the last lens's last vertex (own composition from the member
    poses), the lens positions, and the last vertex z."""

    by_id = {m["model_id"]: m for m in document["member_models"]}
    lenses = [m for m in document["optical_model"]["model"]["members"] if m["member_id"].startswith("L")]
    system, previous_end, positions = np.eye(2), None, []
    for member in lenses:
        z = member["pose"]["position_mm"][2]
        positions.append(z)
        matrix, first, last = _member_matrix(document, by_id[member["model_id"]], wavelength)
        if previous_end is not None:
            system = np.array([[1.0, z + first - previous_end], [0.0, 1.0]]) @ system
        system = matrix @ system
        previous_end = z + last
    return system, positions, previous_end


def _member_efl(document: dict, member_id: str, wavelength: float) -> float:
    member = next(m for m in document["optical_model"]["model"]["members"] if m["member_id"] == member_id)
    model = next(m for m in document["member_models"] if m["model_id"] == member["model_id"])
    return -1.0 / _member_matrix(document, model, wavelength)[0][1, 0]


@pytest.mark.parametrize("class_id", ["compound.keplerian_telescope", "compound.galilean_telescope"])
def test_telescope_is_afocal_with_magnification_minus_f2_over_f1(class_id):
    document = model_of(class_id, ALL_CLASSES[class_id])
    wavelength = 632.8
    f1, f2 = _member_efl(document, "L1", wavelength), _member_efl(document, "L2", wavelength)
    system, _positions, _end = _unit_system(document, wavelength)
    a, _b, c, d = system.ravel()
    assert abs(c) < 1e-9
    assert a == pytest.approx(-f2 / f1, rel=1e-7) and d == pytest.approx(-f1 / f2, rel=1e-7)
    assert float(document["design"]["beam_magnification"]) == pytest.approx(-f2 / f1, rel=1e-8)
    if class_id == "compound.galilean_telescope":
        assert _positions == [0.0, 75.0]  # thin lenses: separation f1 + f2


def test_keplerian_spatial_filter_pinhole_sits_at_the_internal_focus():
    document = model_of("compound.keplerian_telescope", ALL_CLASSES["compound.keplerian_telescope"])
    members = {m["member_id"]: m for m in document["optical_model"]["model"]["members"]}
    stop_model = next(m for m in document["member_models"] if m["model_id"] == members["STOP"]["model_id"])
    assert stop_model["model"]["opening_radius_mm"] == "0.015"
    lens1 = next(m for m in document["member_models"] if m["model_id"] == members["L1"]["model_id"])
    matrix, _first, last = lens_matrix(document, 632.8, lens1)
    rear_focus = members["L1"]["pose"]["position_mm"][2] + last - matrix[0, 0] / matrix[1, 0]
    assert members["STOP"]["pose"]["position_mm"][2] == pytest.approx(rear_focus, abs=2e-6)
    envelope = document["optical_model"]["model"]["envelope"]
    assert float(envelope["z_min_mm"]) <= 0 and float(envelope["diameter_mm"]) == 25.4


def test_relay_4f_images_its_object_plane_with_magnification_minus_f2_over_f1():
    document = model_of("compound.relay_4f", ALL_CLASSES["compound.relay_4f"])
    wavelength = 632.8
    system, positions, end = _unit_system(document, wavelength)
    ports = {p["port_id"]: float(p["point_mm"][2]) for p in document["optical_model"]["model"]["ports"]}
    full = np.array([[1.0, ports["OUT"] - end], [0.0, 1.0]]) @ system @ np.array([[1.0, positions[0]], [0.0, 1.0]])
    a, b, c, _d = full.ravel()
    f1, f2 = _member_efl(document, "L1", wavelength), _member_efl(document, "L2", wavelength)
    assert ports["IN"] == 0.0 and abs(b) < 1e-6 and abs(c) < 1e-9
    assert a == pytest.approx(-f2 / f1, rel=1e-7)
    # R2 n7: the reported matrix is the placed unit's (members and the OUT port at their 1e-6 mm positions), not the
    # unrounded design's B = 0
    reported_a, reported_b, reported_c = (float(v) for v in document["design"]["object_to_image_matrix_A_B_C"])
    assert reported_b == pytest.approx(b, abs=1e-10) and reported_b != 0.0
    assert reported_a == pytest.approx(a, abs=1e-11) and reported_c == pytest.approx(c, abs=1e-14)
    stop = next(m for m in document["member_models"] if m["model_id"].endswith("-STOP"))["model"]
    assert (stop["opening_radius_mm"], stop["obstruction_radius_mm"]) == ("1", "0.1")


# ------------------------------------------------------------------------------------------------- stops, plates, ...


def _inside(stop: dict, x: float, y: float) -> bool:
    """Independent reading of an APERTURE_STOP: does a ray crossing (x, y) pass?"""

    cx, cy = (float(v) for v in stop.get("opening_center_mm", ["0", "0"]))
    if "opening_radius_mm" in stop:
        r = math.hypot(x - cx, y - cy)
        return r <= float(stop["opening_radius_mm"]) and r >= float(stop.get("obstruction_radius_mm", "0"))
    hx, hy = (float(v) for v in stop["opening_half_widths_mm"])
    return abs(x - cx) <= hx and abs(y - cy) <= hy


def test_stop_geometry_follows_the_parameters():
    circular = model_of("aperture.circular", {"opening_diameter_mm": "0.05"})["optical_model"]
    assert circular["model"]["opening_radius_mm"] == "0.025" and circular["model"]["edge_radius_mm"] == "12.7"
    assert _inside(circular["model"], 0.02, 0.0) and not _inside(circular["model"], 0.03, 0.0)
    slit = model_of("aperture.slit", ALL_CLASSES["aperture.slit"])["optical_model"]["model"]
    assert slit["opening_half_widths_mm"] == ["0.05", "2.5"] and slit["opening_center_mm"] == ["0.5", "0"]
    assert _inside(slit, 0.54, 2.4) and not _inside(slit, 0.56, 0.0) and not _inside(slit, 0.5, 2.6)
    annulus = model_of("aperture.annular", ALL_CLASSES["aperture.annular"])
    assert _inside(annulus["optical_model"]["model"], 3.0, 0.0) and not _inside(annulus["optical_model"]["model"], 1.0, 0.0)
    assert annulus["design"]["open_area_fraction_of_opening"] == "0.91"  # 1 - (3 / 10)^2
    block = model_of("aperture.central_obstruction", {"obstruction_diameter_mm": "2", "carrier_diameter_mm": "30"})["optical_model"]["model"]
    assert block["opening_radius_mm"] == block["edge_radius_mm"] == "15" and block["obstruction_radius_mm"] == "1"
    big = model_of("aperture.circular", {"opening_diameter_mm": "20"})["optical_model"]["model"]
    assert big["edge_radius_mm"] == "20"  # default plate: twice the opening once that exceeds one inch


def test_a_decentred_circular_opening_is_checked_radially():
    """R2 n7: the opening's far edge lies at |centre| + its radius from the plate centre (centre (10, 10), opening 2 mm:
    15.142 mm), so a 25 mm plate cannot hold it -- refused naming plate_diameter_mm; the default plate covers it."""

    far_edge = math.hypot(10.0, 10.0) + 1.0
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("aperture.circular", {"opening_diameter_mm": "2", "opening_center_mm": ["10", "10"], "plate_diameter_mm": "25"})
    problem = exc.value.problems[0]
    assert (problem.code.value, problem.parameter) == ("NATIVE_PARAMETER_INVALID", "plate_diameter_mm")
    assert f"{2 * far_edge:.3f}" in problem.message
    stop = model_of("aperture.circular", {"opening_diameter_mm": "2", "opening_center_mm": ["10", "10"]})["optical_model"]["model"]
    assert float(stop["edge_radius_mm"]) >= far_edge
    assert _inside(stop, 10.0 + 0.999 / math.sqrt(2), 10.0 + 0.999 / math.sqrt(2)) and not _inside(stop, 0.0, 0.0)
    built = model_of("aperture.circular", {"opening_diameter_mm": "2", "opening_center_mm": ["10", "10"], "plate_diameter_mm": "30.3"})
    assert built["optical_model"]["model"]["edge_radius_mm"] == "15.15"


def test_a_given_centre_thickness_needs_only_a_positive_edge():
    """R2 n7, as NATIVE_COMPONENTS.md states: the thickening rule's edge minimum applies to a thickness the library chooses;
    a given centre thickness is kept when its edge is positive, and an explicit min_edge_thickness_mm still applies."""

    build = model_of("lens.singlet", {**SINGLET, "center_thickness_mm": "2"})
    surfaces = build["optical_model"]["model"]["surfaces"]
    edge = edge_thickness(radius(surfaces[0]), radius(surfaces[1]), 2.0, 12.7)
    assert 0 < edge < max(0.5, 0.05 * 25.4) and float(build["design"]["edge_thickness_mm"]) == pytest.approx(edge, abs=2e-6)
    for parameters in ({"center_thickness_mm": "1.5"}, {"center_thickness_mm": "2", "min_edge_thickness_mm": "1.27"}):
        with pytest.raises(NativeComponentError) as exc:
            build_native_model("lens.singlet", {**SINGLET, **parameters})
        problem = exc.value.problems[0]
        assert (problem.code.value, problem.parameter) == ("NATIVE_DESIGN_INFEASIBLE", "center_thickness_mm"), parameters
    doublet = model_of("lens.achromat", {**ALL_CLASSES["lens.achromat"], "flint_center_thickness_mm": "1.5"})
    t1, t2 = (float(t) for t in doublet["design"]["center_thicknesses_mm"])
    assert t2 == 1.5 and min(float(e) for e in doublet["design"]["edge_thicknesses_mm"]) > 0
    assert float(doublet["design"]["edge_thicknesses_mm"][0]) >= max(0.5, 0.05 * 25.4) - 1e-6  # the chosen crown keeps the rule


def test_window_and_cuvette_surfaces_media_and_outline():
    window = model_of("plate.window", ALL_CLASSES["plate.window"])
    physics = window["optical_model"]["model"]
    assert [s["vertex_z_mm"] for s in physics["surfaces"]] == ["0", "5"] and physics["media"] == ["INDEX-1.33"]
    assert window["model_materials"][0]["absorption_coefficient_per_mm"] == "0.01"
    assert all(s["coating_id"] == "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10" for s in physics["surfaces"])
    assert physics["valid_band_nm"]["min_nm"] == "400" and physics["valid_band_nm"]["max_nm"] == "900"
    cell = model_of("plate.layered", ALL_CLASSES["plate.layered"])
    physics = cell["optical_model"]["model"]
    assert [s["vertex_z_mm"] for s in physics["surfaces"]] == ["0", "1.25", "11.25", "12.5"]
    assert physics["media"] == ["FUSED_SILICA", "WATER-NOMINAL", "FUSED_SILICA"]
    assert physics["edge_half_widths_mm"] == ["6.25", "22.5"] and cell["optical_model"]["design_geometry"]["kind"] == "BOX"


def test_mirrors_have_radius_twice_the_focal_length_and_their_declared_coating():
    concave = model_of("mirror.curved", ALL_CLASSES["mirror.curved"])
    surface = concave["optical_model"]["model"]["surfaces"][0]
    assert surface["radius_mm"] == "200" and surface["conic"] == "-1" and surface["traced"] == "REFLECTED"
    assert concave["optical_model"]["frame"]["reference_axis"] == "FACE_NORMAL"
    coating = concave["model_coatings"][0]
    assert coating["bands"][0]["reflectance"] == "0.98" and coating["bands"][0]["transmittance"] == "0"
    # paraboloid sag at the edge: r^2 / (2 R)
    assert float(concave["design"]["edge_sag_mm"]) == pytest.approx(12.7 ** 2 / 400.0, abs=1e-6)
    convex = model_of("mirror.curved", {**ALL_CLASSES["mirror.curved"], "focal_length_mm": "-100", "conic": "0"})
    assert convex["optical_model"]["model"]["surfaces"][0]["radius_mm"] == "-200"
    dielectric = model_of("mirror.flat", {**ALL_CLASSES["mirror.flat"], "coating_kind": "DIELECTRIC_HIGH_REFLECTOR"})
    band = dielectric["model_coatings"][0]["bands"][0]
    assert (band["reflectance"], band["transmittance"]) == ("0.98", "0.02")
    assert "LOSSLESS_COATING" in dielectric["optical_model"]["model"]["approximations"]


def _solids(build) -> dict:
    """``json_api.inspect_optical_solids`` of the built component alone in a Project (with a source far before it)."""
    return json_api.inspect_optical_solids(add_to_project(empty_project(), build, position_mm=(200.0, 0.0, 0.0)))


def _convex_edge_depth(focal_mm: float, diameter_mm: float) -> float:
    """Depth of a convex spherical face behind its vertex at the outer radius: |R| - sqrt(R^2 - r^2), R = 2 |f|."""
    r, rho = 2.0 * abs(focal_mm), diameter_mm / 2.0
    return r - math.sqrt(r * r - rho * rho)


CONVEX_1000NM = {"diameter_mm": "25.4", "reflectance": "1", "valid_band_nm": ["400", "1000"]}


def test_a_convex_mirror_substrate_holds_its_face_or_is_refused():
    """A convex face falls behind its vertex towards the edge: f = -6.5 mm over 25.4 mm reaches 10.2233 mm behind it, more
    than the default 6.4 mm substrate (it used to build and then fail the solid check, the face outside the body). The
    default adds that depth, so the edge keeps 6.4 mm; a given thickness at the vertex that does not exceed the depth (0.1 mm
    against 0.4036 mm at f = -100 mm) is refused, naming the parameter and the least accepted value, which then builds a
    valid solid. A concave mirror's face rises in front of its vertex: any substrate holds it."""
    deep = build_native_model("mirror.curved", {**CONVEX_1000NM, "focal_length_mm": "-6.5"})
    depth = _convex_edge_depth(-6.5, 25.4)
    assert depth == pytest.approx(10.2233113, abs=1e-6)
    assert deep.as_dict()["design"]["drawn_substrate_thickness_mm"] == "16.7"  # ceil(6.4 + 10.2233, 0.1 mm)
    body = deep.optical_model.design_geometry
    assert float(body.z_min_mm) == -16.7 and float(body.z_max_mm) == 0.0
    assert _solids(deep)["status"] == "VALID"
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("mirror.curved", {**CONVEX_1000NM, "focal_length_mm": "-100", "substrate_thickness_mm": "0.1"})
    (problem,) = exc.value.problems
    assert (problem.code.value, problem.parameter) == ("NATIVE_DESIGN_INFEASIBLE", "substrate_thickness_mm")
    assert _convex_edge_depth(-100.0, 25.4) == pytest.approx(0.4036323, abs=1e-6)
    assert "0.403632 mm behind its vertex" in problem.message and "at least 0.403633 mm" in problem.message
    least = build_native_model("mirror.curved", {**CONVEX_1000NM, "focal_length_mm": "-100", "substrate_thickness_mm": "0.403633"})
    assert _solids(least)["status"] == "VALID"
    concave = build_native_model("mirror.curved", {**CONVEX_1000NM, "focal_length_mm": "100", "substrate_thickness_mm": "0.1"})
    assert _solids(concave)["status"] == "VALID"
    flat_default = build_native_model("mirror.curved", {**CONVEX_1000NM, "focal_length_mm": "6.5"})
    assert flat_default.as_dict()["design"]["drawn_substrate_thickness_mm"] == "6.4"  # concave: the default rule unchanged


BK7_DECLARED = {"sellmeier": {"b": ["1.03961212", "0.231792344", "1.01046945"], "c_um2": ["0.00600069867", "0.0200179144", "103.560653"]},
                "valid_band_nm": ["330", "2500"], "material_id": "BK7-DECLARED"}


def _sweep() -> list[tuple[str, dict]]:
    """Parameter sweeps of every native class, including the edges of each builder's own rules (a nearly hemispherical
    convex or concave mirror, thin substrates and plates, short and negative focal lengths, given centre thicknesses)."""
    import itertools

    cases: list[tuple[str, dict]] = []

    def add(class_id: str, **parameters):
        cases.append((class_id, {k: v for k, v in parameters.items() if v is not None}))

    for f, d, t, k in itertools.product(["-6.36", "-6.5", "-25", "-100", "6.36", "25", "100", "-1000"], ["5", "25.4", "50.8"],
                                        [None, "0.1", "3"], ["0", "-1", "-3", "0.8"]):
        add("mirror.curved", **{**CONVEX_1000NM, "diameter_mm": d}, focal_length_mm=f, substrate_thickness_mm=t, conic=k)
    for d, t in itertools.product(["1", "25.4"], [None, "0.01", "3"]):
        add("mirror.flat", diameter_mm=d, reflectance="0.9", valid_band_nm=["400", "1000"], substrate_thickness_mm=t)
        add("splitter.plate", diameter_mm=d, reflectance="0.5", valid_band_nm=["400", "1000"], thickness_mm=t)
        add("polarizer.linear", diameter_mm=d, transmittance="0.9", extinction_ratio="1000", valid_band_nm=["400", "1000"], thickness_mm=t)
        add("rotator.faraday", **{**ALL_CLASSES["rotator.faraday"], "diameter_mm": d, "length_mm": t})
        add("grating.plane", **{**ALL_CLASSES["grating.plane"], "width_mm": d, "height_mm": d, "thickness_mm": t})
        add("deflector.acousto_optic", **{**ALL_CLASSES["deflector.acousto_optic"], "width_mm": d, "height_mm": d, "thickness_mm": t})
        add("waveplate.retarder", diameter_mm=d, retardance_waves="0.5", design_wavelength_nm="633", valid_band_nm=["400", "1000"],
            thickness_mm=t)
        add("lens.ideal_thin", focal_length_mm="50", clear_aperture_diameter_mm=d, thickness_mm=t)
    for f, d, form, tc, te in itertools.product(["-10", "-50", "10", "13", "50", "1000"], ["5", "25.4"],
                                                ["SYMMETRIC", "PLANO_EXIT", "PLANO_ENTRY", "BEST_FORM"], [None, "0.3", "5"], [None, "0.2"]):
        add("lens.singlet", focal_length_mm=f, design_wavelength_nm="632.8", material=BK7_DECLARED, diameter_mm=d, form=form,
            center_thickness_mm=tc, min_edge_thickness_mm=te)
    for f, d in itertools.product(["-30", "15", "50"], ["5", "25.4"]):
        common = {"focal_length_mm": f, "design_wavelength_nm": "632.8", "material": BK7_DECLARED, "diameter_mm": d}
        add("lens.singlet", **common, form="PLANO_ENTRY", rear_conic="STIGMATIC_COLLIMATED")
        add("lens.singlet", **common, shape_factor="3")
        add("lens.singlet", **common, cylinder_axis="X")
    for f, d, factor, t1 in itertools.product(["-100", "30", "100"], ["12.7", "25.4"], ["0", "-0.5", "1"], [None, "1"]):
        add("lens.achromat", focal_length_mm=f, crown_material=BK7_DECLARED, flint_material=N_SF5, diameter_mm=d, crown_shape_factor=factor,
            crown_center_thickness_mm=t1)
    for t, outline in itertools.product(["0.01", "5"], [{"diameter_mm": "25"}, {"width_mm": "10", "height_mm": "40"}]):
        add("plate.window", material={"constant_index": "1.5", "valid_band_nm": [400, 900]}, thickness_mm=t, **outline)
    add("plate.layered", **ALL_CLASSES["plate.layered"])
    for opening, t in itertools.product(["0.05", "20"], [None, "0.01"]):
        add("aperture.circular", opening_diameter_mm=opening, thickness_mm=t)
    for class_id in ("aperture.slit", "aperture.annular", "aperture.central_obstruction", "compound.keplerian_telescope",
                     "compound.galilean_telescope", "compound.relay_4f"):
        add(class_id, **ALL_CLASSES[class_id])
    for angle, orientation, t in itertools.product(["0.5", "10", "30"], ["PLANE_FIRST", "CONE_FIRST"], [None, "8"]):
        add("lens.axicon", **{**ALL_CLASSES["lens.axicon"], "cone_angle_deg": angle, "orientation": orientation, "center_thickness_mm": t})
    for angle, t in itertools.product(["1", "10", "30"], [None, "4"]):
        add("prism.wedge", **{**ALL_CLASSES["prism.wedge"], "wedge_angle_deg": angle, "center_thickness_mm": t})
    for seam in ("0.001", "0.5"):
        add("mirror.hollow_roof", **{**ALL_CLASSES["mirror.hollow_roof"], "seam_mm": seam})
        add("retroreflector.hollow_corner", **{**ALL_CLASSES["retroreflector.hollow_corner"], "seam_mm": seam})
    for class_id in ("filter.variable_nd", "lens.tabulated_thin"):
        add(class_id, **ALL_CLASSES[class_id])
    for axis in ("20", "45", "80"):
        add("splitter.beam_displacer", **{**ALL_CLASSES["splitter.beam_displacer"], "axis_angle_deg": axis})
    for variant, gap in itertools.product(["GLAN_TAYLOR", "GLAN_FOUCAULT"], ["0.005", "0.5"]):
        add("polarizer.glan", **{**ALL_CLASSES["polarizer.glan"], "variant": variant, "air_gap_mm": gap})
    for pitch, diameter in itertools.product(["0.23", "0.25", "0.5", "1.3"], ["1", "1.8"]):
        add("lens.grin_rod", **{**ALL_CLASSES["lens.grin_rod"], "pitch": pitch, "diameter_mm": diameter})
    add("lens.grin_rod", **{**ALL_CLASSES["lens.grin_rod"], "diameter_mm": "4"})  # (g R >= 1: refused)
    for angle, layer in itertools.product(["72", "85"], ["0.002", "0.5"]):
        add("polarizer.glan_thompson", **{**ALL_CLASSES["polarizer.glan_thompson"], "gap_angle_deg": angle, "cement_thickness_mm": layer})
    for class_id, angle, edge in itertools.product(("splitter.wollaston", "splitter.rochon"), ["5", "20", "40"], [None, "0.05"]):
        add(class_id, **{**ALL_CLASSES[class_id], "wedge_angle_deg": angle, "edge_thickness_mm": edge})
    add("splitter.wollaston", **{**ALL_CLASSES["splitter.wollaston"], "cement": {"constant_index": "1.55", "valid_band_nm": [300, 2500]}})
    return cases


def test_every_successful_native_build_is_a_valid_optical_solid():
    """A built native component is usable: over a sweep of every class's parameters, each request either is refused with a
    structured error or builds a model whose solid passes ``inspect_optical_solids`` (no OPTICAL_MODEL_INVALID, no INVALID
    body) -- success never means a model the run's own solid check then rejects."""
    cases = _sweep()
    assert {class_id for class_id, _ in cases} == set(ALL_CLASSES)
    built = 0
    for class_id, parameters in cases:
        try:
            build = build_native_model(class_id, parameters)
        except NativeComponentError:
            continue
        built += 1
        report = _solids(build)
        assert report["ok"] and report["status"] == "VALID", (class_id, parameters, report)
    assert built > len(cases) // 2


def test_plate_splitter_per_polarization_split():
    splitter = model_of("splitter.plate", ALL_CLASSES["splitter.plate"])
    band = splitter["model_coatings"][0]["bands"][0]
    assert (band["reflectance_s"], band["reflectance_p"], band["transmittance_s"], band["transmittance_p"]) == ("0.9", "0.1", "0.1", "0.9")
    assert (band["reflectance"], band["transmittance"]) == ("0.5", "0.5")
    physics = splitter["optical_model"]["model"]
    assert physics["surfaces"][0]["traced"] == "TRANSMITTED_AND_REFLECTED" and physics["media"] == []
    assert "POLARIZATION_INDEPENDENT_COATING" not in physics["approximations"] and "SUBSTRATE_NOT_MODELLED" in physics["approximations"]
    assert splitter["optical_model"]["frame"]["nominal_propagation"] == "SPLIT_ABOUT_+Z"


def test_polarization_elements_carry_their_declared_jones_parameters():
    polarizer = model_of("polarizer.linear", ALL_CLASSES["polarizer.linear"])["optical_model"]["model"]
    assert polarizer["element"] == "LINEAR_POLARIZER" and polarizer["extinction_ratio"]["value"] == "10000"
    assert polarizer["transmittance"]["value"] == "0.9" and polarizer["axis"] == "+X"
    plate = model_of("waveplate.retarder", ALL_CLASSES["waveplate.retarder"])["optical_model"]["model"]
    assert (plate["retardance_waves"], plate["design_wavelength_nm"], plate["retardance_dispersion"]) == ("0.25", "633", "INVERSE_WAVELENGTH")
    achromatic = model_of("waveplate.retarder", {**ALL_CLASSES["waveplate.retarder"], "retardance_dispersion": "achromatic"})["optical_model"]["model"]
    assert achromatic["retardance_dispersion"] == "CONSTANT" and "CONSTANT_RETARDANCE" in achromatic["approximations"]


def test_ideal_thin_lens_fields():
    lens = model_of("lens.ideal_thin", {**ALL_CLASSES["lens.ideal_thin"], "diameter_mm": "25", "transmittance": "0.99"})["optical_model"]
    assert lens["model"]["focal_length_mm"] == "50" and lens["model"]["clear_radius_mm"] == "10" and lens["model"]["edge_radius_mm"] == "12.5"
    assert lens["model"]["transmittance"] == "0.99" and "frame" not in lens  # a fixed frame (plane z = 0, axis +Z)


# ----------------------------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(("class_id", "parameters", "code", "parameter"), [
    ("lens.nothing", {}, "NATIVE_CLASS_UNKNOWN", "class_id"),
    ("aperture.circular", {"opening_diameter_mm": "1", "radius": "1"}, "NATIVE_PARAMETER_UNKNOWN", "radius"),
    ("aperture.circular", {}, "NATIVE_PARAMETER_MISSING", "opening_diameter_mm"),
    ("aperture.circular", {"opening_diameter_mm": "-1"}, "NATIVE_PARAMETER_INVALID", "opening_diameter_mm"),
    ("aperture.circular", {"opening_diameter_mm": "abc"}, "NATIVE_PARAMETER_INVALID", "opening_diameter_mm"),
    ("aperture.circular", {"opening_diameter_mm": "30", "plate_diameter_mm": "20"}, "NATIVE_PARAMETER_INVALID", "plate_diameter_mm"),
    ("aperture.annular", {"opening_diameter_mm": "3", "obstruction_diameter_mm": "3"}, "NATIVE_PARAMETER_INVALID", "obstruction_diameter_mm"),
    ("lens.singlet", {**SINGLET, "focal_length_mm": "0"}, "NATIVE_PARAMETER_INVALID", "focal_length_mm"),
    ("lens.singlet", {**SINGLET, "focal_length_mm": "8"}, "NATIVE_DESIGN_INFEASIBLE", "focal_length_mm"),
    ("lens.singlet", {**SINGLET, "design_wavelength_nm": "200"}, "NATIVE_PARAMETER_INVALID", "design_wavelength_nm"),
    ("lens.singlet", {**SINGLET, "form": "SYMMETRIC", "shape_factor": "0"}, "NATIVE_PARAMETER_INVALID", "shape_factor"),
    ("lens.singlet", {**SINGLET, "rear_conic": "STIGMATIC_COLLIMATED"}, "NATIVE_PARAMETER_INVALID", "rear_conic"),
    ("lens.singlet", {**SINGLET, "center_thickness_mm": "1"}, "NATIVE_DESIGN_INFEASIBLE", "center_thickness_mm"),
    ("lens.singlet", {**SINGLET, "material": "UNOBTAINIUM"}, "NATIVE_CATALOG_DATA_UNAVAILABLE", "material"),
    ("lens.singlet", {**SINGLET, "material": {"constant_index": "1.5"}}, "NATIVE_PARAMETER_MISSING", "material"),
    ("lens.singlet", {**SINGLET, "material": {"constant_index": "0.9", "valid_band_nm": [400, 700]}}, "NATIVE_PARAMETER_INVALID", "material"),
    ("lens.singlet", {**SINGLET, "material": {"constant_index": "1", "valid_band_nm": [400, 700]}}, "NATIVE_DESIGN_INFEASIBLE", "material"),
    ("lens.singlet", {**SINGLET, "material": {"sellmeier": {"b": [1, 0, 0], "c_um2": ["0.4", 0, 0]}, "valid_band_nm": [400, 700]}},
     "NATIVE_PARAMETER_INVALID", "material"),  # a pole at 632.456 nm, inside the band
    ("lens.achromat", {"focal_length_mm": "100", "crown_material": "N-BK7", "flint_material": "N-BK7", "diameter_mm": "25.4"},
     "NATIVE_PARAMETER_INVALID", "flint_material"),
    ("mirror.flat", {"diameter_mm": "25.4", "valid_band_nm": [400, 700]}, "NATIVE_PARAMETER_INVALID", "reflectance"),
    ("mirror.flat", {"diameter_mm": "25.4", "reflectance": "0.6", "transmittance": "0.6", "valid_band_nm": [400, 700]},
     "NATIVE_PARAMETER_INVALID", "transmittance"),
    ("mirror.curved", {"diameter_mm": "25.4", "focal_length_mm": "5", "reflectance": "0.9", "valid_band_nm": [400, 700]},
     "NATIVE_DESIGN_INFEASIBLE", "focal_length_mm"),
    ("mirror.curved", {"diameter_mm": "25.4", "focal_length_mm": "-100", "reflectance": "1", "valid_band_nm": ["400", "1000"],
                       "substrate_thickness_mm": "0.1"}, "NATIVE_DESIGN_INFEASIBLE", "substrate_thickness_mm"),
    ("waveplate.retarder", {**ALL_CLASSES["waveplate.retarder"], "design_wavelength_nm": "700"}, "NATIVE_PARAMETER_INVALID", "design_wavelength_nm"),
    ("polarizer.linear", {**ALL_CLASSES["polarizer.linear"], "extinction_ratio": "0.5"}, "NATIVE_PARAMETER_INVALID", "extinction_ratio"),
    ("plate.window", {"material": "N-BK7", "thickness_mm": "2"}, "NATIVE_PARAMETER_INVALID", "diameter_mm"),
    ("compound.keplerian_telescope", {**ALL_CLASSES["compound.galilean_telescope"]}, "NATIVE_PARAMETER_INVALID", "lens2"),
    ("compound.galilean_telescope", {**ALL_CLASSES["compound.galilean_telescope"], "focal_stop_diameter_mm": "1"},
     "NATIVE_PARAMETER_UNKNOWN", "focal_stop_diameter_mm"),
    ("compound.relay_4f", {"lens1": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "0", "clear_aperture_diameter_mm": "5"}},
                           "lens2": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "50", "clear_aperture_diameter_mm": "5"}}},
     "NATIVE_PARAMETER_INVALID", "lens1.focal_length_mm"),
    ("compound.relay_4f", {"lens1": {"class_id": "aperture.circular", "parameters": {}}, "lens2": {"class_id": "lens.ideal_thin"}},
     "NATIVE_PARAMETER_INVALID", "lens1"),
])
def test_refusals_are_structured_with_the_offending_parameter(class_id, parameters, code, parameter):
    with pytest.raises(NativeComponentError) as exc:
        build_native_model(class_id, parameters)
    assert (exc.value.problems[0].code.value, exc.value.problems[0].parameter) == (code, parameter), exc.value.problems


def test_every_problem_is_reported_together():
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.singlet", {"focal_length_mm": "x", "colour": "red"})
    reported = {(p.code.value, p.parameter) for p in exc.value.problems}
    assert {("NATIVE_PARAMETER_UNKNOWN", "colour"), ("NATIVE_PARAMETER_INVALID", "focal_length_mm"),
            ("NATIVE_PARAMETER_MISSING", "design_wavelength_nm"), ("NATIVE_PARAMETER_MISSING", "material"),
            ("NATIVE_PARAMETER_MISSING", "diameter_mm")} <= reported


def test_catalog_material_without_a_catalog_names_the_declared_alternatives(empty_catalog_store):
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("lens.singlet", SINGLET)
    problem = exc.value.problems[0]
    assert problem.code.value == "NATIVE_CATALOG_DATA_UNAVAILABLE" and "declare" in problem.message
    declared = {**SINGLET, "material": {"sellmeier": {"b": [str(v) for v in N_BK7_COEFFS[0]], "c_um2": [str(v) for v in N_BK7_COEFFS[1]]},
                                        "valid_band_nm": [300, 2500]}}
    assert efl(model_of("lens.singlet", declared), 632.8) == pytest.approx(100.0, rel=1e-7)


# ------------------------------------------------------------------------------------------------- json_api / CLI


def test_json_api_discovery_build_and_refusal_envelopes():
    listed = json_api.native_components()
    assert listed["ok"] is True and [c["class_id"] for c in listed["classes"]] == native_class_ids()
    assert json_api.capabilities()["native_component_classes"] == native_class_ids()
    built = json_api.build_native_model("aperture.slit", {"slit_width_mm": 0.1, "slit_height_mm": 3})
    assert built["ok"] is True and built["optical_model"]["model"]["opening_half_widths_mm"] == ["0.05", "1.5"]
    assert set(built) == {"ok", "class_id", "optical_model", "member_models", "model_materials", "model_coatings", "component_template", "design"}
    refused = json_api.build_native_model("aperture.slit", {"slit_width_mm": "-1"})
    assert refused["ok"] is False and refused["error"]["code"] == "NATIVE_COMPONENT_INVALID"
    assert {p["parameter"] for p in refused["error"]["problems"]} == {"slit_width_mm", "slit_height_mm"}
    assert json.loads(json.dumps(built)) == built  # JSON-safe


def test_cli_native_list_describe_build_and_into(tmp_path, capsys):
    from os_optics.cli import main

    assert main(["native", "list", "--category", "stop"]) == 0
    assert {c["class_id"] for c in json.loads(capsys.readouterr().out)["classes"]} == {
        "aperture.circular", "aperture.slit", "aperture.annular", "aperture.central_obstruction"}
    assert main(["native", "describe", "lens.achromat"]) == 0
    assert json.loads(capsys.readouterr().out)["class_id"] == "lens.achromat"
    assert main(["native", "build", "lens.singlet", "--param", "focal_length_mm=100", "--param", "design_wavelength_nm=632.8",
                 "--param", "material=N-BK7", "--param", "diameter_mm=25.4"]) == 0
    assert json.loads(capsys.readouterr().out)["optical_model"]["model"]["paraxial"]["effective_focal_length_mm"].startswith("100")
    project = tmp_path / "project.json"
    project.write_text(json.dumps(empty_project()), encoding="utf-8")
    assert main(["native", "build", "aperture.circular", "--param", "opening_diameter_mm=1", "--into", str(project),
                 "--instance-id", "IRIS", "--position", "50,0,0", "--quaternion", "0.5,0.5,0.5,0.5"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["component"] == "IRIS"
    saved = ProjectDocumentV2.model_validate(json.loads(project.read_text(encoding="utf-8")))
    iris = next(c for c in saved.components if c.instance_id == "IRIS")
    assert iris.optical_model_id == summary["optical_model_id"] and iris.nominal_pose.position_mm == (50.0, 0.0, 0.0)
    assert main(["native", "build", "aperture.circular", "--param", "opening_diameter_mm=0"]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["problems"][0]["parameter"] == "opening_diameter_mm"
    assert main(["native", "build", "aperture.circular", "--param", "opening_diameter_mm"]) == 64
    assert main(["native", "build", "aperture.circular", "--param", "opening_diameter_mm=1", "--position", "1,2,3"]) == 64


def test_cli_native_list_help_names_every_category_of_the_registry(capsys):
    import re

    from os_optics.cli import main

    with pytest.raises(SystemExit):
        main(["native", "list", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    named = set(re.search(r"only this category \(([A-Z, ]+)\)", help_text).group(1).split(", "))
    assert named == {c["category"] for c in native_components()}


@pytest.mark.parametrize("prec", [3, 30, 60])
def test_a_native_build_does_not_depend_on_the_callers_decimal_context(prec):
    """Every build runs in its own stated decimal context: a caller's thread context (another precision) changes nothing."""
    from decimal import localcontext

    requests = [("mirror.curved", {"diameter_mm": "25.4", "reflectance": "1", "valid_band_nm": ["400", "1000"], "focal_length_mm": "-6.5"}),
                ("aperture.circular", {"opening_diameter_mm": "3", "opening_center_mm": ["0.3", "0.4"]})]
    expected = [json.dumps(json_api.build_native_model(c, p), sort_keys=True) for c, p in requests]
    assert all(json.loads(e)["ok"] for e in expected)
    with localcontext() as ctx:
        ctx.prec = prec
        assert [json.dumps(json_api.build_native_model(c, p), sort_keys=True) for c, p in requests] == expected
    assert json.loads(expected[0])["design"]["edge_sag_mm"] == "-10.2233113246170358610957099"
