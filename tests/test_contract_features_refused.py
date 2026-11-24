"""Contract features of catalog schema 3.1 / the 0.8.0 Project vocabulary: a catalog 3.1 release is read, the model kinds,
fields and sources it adds are simulated (an attenuator's coated annulus included), and a target that uses a field outside its metric is not assessed, with a structured code (``TARGET_INVALID``) and its
location, never a crash.

Fixture data: every physical value below is made up for the test.
"""

from __future__ import annotations

import copy

import pytest

from os_optics.catalog import SUPPORTED_SCHEMA_VERSIONS
from os_optics.catalog.store import check_compatible
from os_optics.local import LocalRunError, LocalRunErrorCode, json_api, load_public_catalog, lower_project

from declared_helpers import declared_component, nbk7, sequence, surface, thin_lens_model, with_models
from local_helpers import catalog_document, plane, project, source

CODE = LocalRunErrorCode.SIMULATION_MODEL_UNSUPPORTED
BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
RATING = {"regime": "CW", "value": "500", "unit": "W/cm", "wavelength_nm": "780", "status": "DECLARED", "basis": "fixture data"}


def _document(models: list[dict], role: str = "lens", *, materials=None, coatings=None, src: dict | None = None, targets=None) -> dict:
    document = project([src or source(), declared_component("E", role, models[0]["model_id"])], [plane("P", (100.0, 0.0, 0.0))], targets)
    return with_models(document, models, materials, coatings)


def _codes(document: dict, catalog=None) -> list[tuple[str, str | None, str]]:
    with pytest.raises(LocalRunError) as caught:
        lower_project(document, catalog)
    return [(p.code.value, p.location, p.message) for p in caught.value.problems]


ND = {"model_kind": "NEUTRAL_DENSITY", "fidelity": "IDEAL_THIN_ELEMENT", "law": {"kind": "FIXED", "optical_density": "1"},
      "clear_radius_mm": "10", "edge_radius_mm": "12.7", "valid_band_nm": BAND, "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}
TTL = {"model_kind": "TABULATED_THIN_LENS", "rows": [{"wavelength_nm": "780", "focal_length_mm": "100", "transmittance": "0.8"}],
       "focal_interpolation": "POINTS_ONLY", "transmittance_interpolation": "POINTS_ONLY", "clear_radius_mm": "10", "edge_radius_mm": "12.7",
       "max_field_angle_deg": "5", "supported_metrics": [{"metric": "power_W", "rel_error": "0.05", "basis": "fixture"}],
       "evidence": {"method": "fixture", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "0" * 64,
                    "fit_model": "fixture", "validation_summary": "fixture", "date": "2026-10-15"}, "status": "DERIVED"}
AXICON = sequence([surface(1, "0", linear_sag_coefficient="0.2"), surface(2, "5")], ["N-BK7"])
POLARIZER = {"model_kind": "POLARIZATION_ELEMENT", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
             "transmittance": {"value": "0.8", "status": "SPECIFIED"}, "extinction_ratio": {"value": "1000", "status": "SPECIFIED"},
             "clear_radius_mm": "10", "edge_radius_mm": "12.7", "valid_band_nm": BAND,
             "valid_aoi_deg": {"min_deg": "0", "max_deg": "10", "status": "SPECIFIED"},
             "transmittance_bands": [{"min_nm": "400", "max_nm": "700", "transmittance": {"value": "0.8", "status": "SPECIFIED"},
                                      "status": "SPECIFIED"},
                                     {"min_nm": "700", "max_nm": "1100", "transmittance": {"value": "0.9", "status": "SPECIFIED"},
                                      "status": "SPECIFIED"}],
             "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}


@pytest.mark.parametrize(("model", "materials"), [(TTL, None), (AXICON, [nbk7()])], ids=["tabulated-thin-lens", "axicon"])
def test_tabulated_thin_lenses_and_axicons_are_simulated(model: dict, materials) -> None:
    document = _document([{"model_id": "own", "model": copy.deepcopy(model)}], "lens", materials=materials)
    assert lower_project(document, None).elements[0].optical_model_id == "own"


WHEEL = {**ND, "law": {"kind": "LINEAR_ROTARY", "od_start": "0.02", "od_end": "2", "arc_start_deg": "0", "arc_end_deg": "270"},
         "uncoated_transmittance": "0.95", "coated_inner_radius_mm": "4.75", "position_parameter": "nd_rotation_deg"}


def test_a_coated_annulus_is_simulated() -> None:
    document = _document([thin_lens_model("thin"), {"model_id": "own", "model": copy.deepcopy(WHEEL)}], "filter")
    document["components"][-1]["optical_model_id"] = "own"
    (nd,) = [c for c in lower_project(document).request.project.components if c.instance_id == "E"]
    assert nd.surfaces[0].interaction.coated_inner_radius_mm == 4.75
    assert json_api.check_project(document)["ok"] is True


def test_declared_power_ratings_and_attenuators_lower() -> None:
    """Power ratings and NEUTRAL_DENSITY models are simulated (validation/exposures.py, optics/attenuation.py)."""
    model = thin_lens_model("thin")
    model["power_ratings"] = [RATING]
    lowered = lower_project(_document([model, {"model_id": "nd", "model": copy.deepcopy(ND)}]))
    (lens,) = [c for c in lowered.request.project.components if c.instance_id == "E"]
    assert [(r.regime, r.value, r.unit, r.origin) for r in lens.power_ratings] == [("CW", 500.0, "W/cm", "model thin")]


def test_band_resolved_polarizers_and_elliptical_pulsed_sources_lower() -> None:
    laser = source()
    laser["source_model"].update({"polarization": {"kind": "ELLIPTICAL", "orientation_deg": 0.0, "ellipticity_deg": 45.0},
                                  "pulse": {"repetition_rate_hz": 1000.0, "pulse_duration_s": 1e-9, "duration_definition": "FWHM"}})
    lowered = lower_project(_document([{"model_id": "pol", "model": copy.deepcopy(POLARIZER)}], "polarizer", src=laser))
    components = {c.instance_id: c for c in lowered.request.project.components}
    assert [(b.min_nm, b.max_nm) for b in components["E"].surfaces[0].interaction.transmittance_bands] == [(400.0, 700.0), (700.0, 1100.0)]
    source_spec = components["SRC"].source
    # ellipticity +45 is left-hand circular (1, +i) / sqrt 2: the sign of the results' polarization_ellipticity_deg (M1)
    assert source_spec.pulse is not None and source_spec.polarization.ex[1] == 0.0 and source_spec.polarization.ey[1] > 0.0


def test_a_second_operand_wavelength_is_refused_outside_geometric_pair_metrics() -> None:
    target = {"target_id": "t", "metric": "power_W", "observation_id": "P", "wavelength_nm": 780.0, "wavelength_nm_b": 780.0, "minimum": 0.0}
    lowered = lower_project(_document([thin_lens_model("thin")], targets=[target]), None)
    problems = [(p.code.value, p.location, p.message) for p in lowered.target_problems]
    assert [(code, location) for code, location, _ in problems] == [(LocalRunErrorCode.TARGET_INVALID.value, "target t")]
    assert "wavelength_nm_b" in problems[0][2]
    (marked,) = lowered.request.project.targets  # (not assessed: it selects no wavelength)
    assert marked.not_assessable.code == "TARGET_INVALID" and marked.wavelength_nm_b is None


def _nd_product() -> dict:
    return {
        "optical_sku": "OS-OPT-FLT-NDW-D25-OD1", "category": "FILTER", "form": "NEUTRAL_DENSITY_FILTER", "display_name": "Fixture ND attenuator",
        "display_name_zh_cn": "测试中性密度衰减片", "summary": "Fixture attenuator.", "commercial_readiness": "SIMULATION_ONLY",
        "frame": {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "ROTATIONAL_ABOUT_LOCAL_Z"},
        "specifications": {"outer_diameter_mm": {"value": "25.4", "status": "SPECIFIED"},
                           "clear_aperture_diameter_mm": {"value": "20", "status": "SPECIFIED"}, "thickness_mm": {"value": "2", "status": "SPECIFIED"},
                           "coating": "NEUTRAL_DENSITY", "working_band_nm": BAND, "optical_density": {"value": "1", "status": "SPECIFIED"}},
        "simulation_models": [dict(copy.deepcopy(ND), edge_radius_mm="12.7")],
        "design_geometry": {"kind": "CYLINDER", "diameter_mm": "25.4", "z_min_mm": "0", "z_max_mm": "2", "status": "NOMINAL"},
    }


def test_a_catalog_3_1_release_is_read_and_its_attenuators_lower() -> None:
    assert SUPPORTED_SCHEMA_VERSIONS == ("3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6")
    check_compatible("3.1", "2026-10-15.1")
    document = catalog_document()
    document["schema_version"] = "3.1"
    document["products"].append(_nd_product())
    document["categories_offered"] = sorted(set(document["categories_offered"]) | {"FILTER"})
    catalog = load_public_catalog(document)
    product = catalog.get("OS-OPT-FLT-NDW-D25-OD1")
    component = {"instance_id": "ND", "optical_role": "filter", "product": {"optical_sku": product.optical_sku},
                 "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}
    lowered = lower_project(project([source(), component], [plane("P", (100.0, 0.0, 0.0))]), document)
    (nd,) = [c for c in lowered.request.project.components if c.instance_id == "ND"]
    assert [s.interaction.kind for s in nd.surfaces] == ["neutral_density"]
