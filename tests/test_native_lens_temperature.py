"""Native singlets use the *saved Project's* 20 C material indices, not a source's reference temperature.

Independent float lensmaker references evaluate the source coefficients directly. CHARMS Ge has a 295 K reference
law, whereas the Project's default environment is 20 C = 293.15 K; the OHARA S-FPL53 source has a 25 C reference
and Schott dn_abs correction after its air-wavelength / relative-index convention.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from os_optics.native_library import add_to_project, build_native_model
from os_optics.optics.materials.physics import material_point
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2


WAVELENGTHS = {"OHARA_S-FPL53": 632.8, "GE_FREY2006": 3500.0}


def air_index(vacuum_nm: float, temperature_c: float = 20.0) -> float:
    """Independent Edlen formula for the test, not the builder's Decimal design_air_index."""
    s = (vacuum_nm / 1000.0) ** 2
    return 1.0 + (6432.8 + 2949810.0 * s / (146.0 * s - 1.0) + 25540.0 * s / (41.0 * s - 1.0)) * 1e-8 / (
        1.0 + 0.0034785 * (temperature_c - 15.0))


def source_law(material: dict, law_nm: float, temperature_k: float) -> float:
    dispersion = material["dispersion"]
    lam_sq = (law_nm / 1000.0) ** 2
    if dispersion["kind"] == "TEMPERATURE_SELLMEIER":
        def polynomial(coefficients):
            return sum(float(c) * temperature_k ** j for j, c in enumerate(coefficients))

        strengths = [polynomial(row) for row in dispersion["s_coefficients"]]
        poles = [polynomial(row) ** 2 for row in dispersion["pole_coefficients_um"]]
    else:
        strengths = [float(v) for v in dispersion["b"]]
        poles = [float(v) for v in dispersion["c_um2"]]
    return math.sqrt(1.0 + sum(s * lam_sq / (lam_sq - p) for s, p in zip(strengths, poles)))


def project_relative_index(material: dict, wavelength_nm: float, temperature_c: float = 20.0) -> float:
    """Independent law -> source reference air -> Schott absolute-index shift -> *operating* air."""
    reference = material.get("index_reference")
    n_ref = (air_index(wavelength_nm, float(reference["reference_air"]["temperature_c"]))
             if reference and (reference["wavelength"] == "AIR" or reference["index"] == "RELATIVE_TO_AIR") else 1.0)
    law_nm = wavelength_nm / n_ref if reference and reference["wavelength"] == "AIR" else wavelength_nm
    law = material["dispersion"]
    kelvin = (temperature_c + 273.15 if law["kind"] == "TEMPERATURE_SELLMEIER"
              else float(law.get("reference_temperature_k", temperature_c + 273.15)))
    n_law = source_law(material, law_nm, kelvin)
    n_abs = n_law * n_ref if reference and reference["index"] == "RELATIVE_TO_AIR" else n_law
    thermo = material.get("thermo_optic")
    if thermo:
        dt = temperature_c - float(thermo["reference_temperature_c"])
        pole = (law_nm / 1000.0) ** 2 - float(thermo["lambda_tk_um"]) ** 2
        delta = (n_law ** 2 - 1.0) / (2.0 * n_law) * (
            float(thermo["d0"]) * dt + float(thermo["d1"]) * dt ** 2 + float(thermo["d2"]) * dt ** 3
            + (float(thermo["e0"]) * dt + float(thermo["e1"]) * dt ** 2) / pole)
        n_abs += delta
    return n_abs / air_index(wavelength_nm, temperature_c)


MATERIALS = Path(__file__).resolve().parent / "fixtures" / "materials" / "redistributable_materials.json"


def _published(material_id: str) -> dict:
    return next(row["public"] for row in json.loads(MATERIALS.read_text(encoding="utf-8"))["records"]
                if row["material_id"] == material_id)


def glass() -> dict:
    """OHARA S-FPL53's published source physics: 25 C reference air, Sellmeier and Schott coefficients."""
    published = _published("OHARA_S-FPL53")
    return {key: published[key] for key in ("material_id", "display_name", "dispersion", "valid_band_nm",
                                            "index_reference", "thermo_optic", "basis", "bulk_loss")}


def charms_ge() -> dict:
    """The NASA CHARMS Table 10 coefficients of Ge (Frey et al. 2006), not a fabricated constant-index substitute."""
    published = _published("GE_FREY2006")
    return {"material_id": "GE_FREY2006", "display_name": "Ge Frey 2006 (CHARMS)",
            "valid_band_nm": {"min_nm": "1900", "max_nm": "5500", "status": "SPECIFIED"},
            "dispersion": published["dispersion"]}


@pytest.mark.parametrize("material_factory", (glass, charms_ge), ids=("S-FPL53-schott", "Ge-CHARMS"))
def test_native_plano_convex_lens_matches_independent_lensmaker_at_project_temperature(material_factory):
    material = material_factory()
    wavelength = WAVELENGTHS[material["material_id"]]
    build = build_native_model("lens.singlet", {"material": material, "design_wavelength_nm": str(wavelength),
                                                 "focal_length_mm": "100", "diameter_mm": "10", "form": "PLANO_EXIT"})
    project = ProjectDocumentV2.model_validate(add_to_project(
        {"schema": "opensci.project", "project_schema_version": "2.0"}, build, instance_id="L1"))
    saved = next(m for m in project.model_materials if m.material_id == material["material_id"])
    source = saved.model_dump(mode="json", exclude_none=True)
    n = project_relative_index(source, wavelength)
    surface = build.optical_model.model.surfaces[0]
    assert build.optical_model.model.surfaces[1].radius_mm is None  # actual plano-convex lens
    radius = float(surface.radius_mm)
    # Independent lensmaker: c2 = 0, Phi = (n_relative - 1) / R, and EFL = 1 / Phi.
    efl_mm = radius / (n - 1.0)
    reported = float(build.optical_model.model.paraxial.effective_focal_length_mm)
    assert efl_mm == pytest.approx(100.0, abs=3e-6)
    assert reported == pytest.approx(efl_mm, abs=1e-6)
    assert float(build.design["refractive_index"]) == pytest.approx(n, abs=5e-12)
    # The Engine's evaluation of the *saved* material at its default Project environment agrees with the independent law.
    point = material_point(saved, wavelength)
    assert point.reason is None
    assert point.n_abs == pytest.approx(n * air_index(wavelength), abs=3e-10)
    # Old designs at 25 C (Schott source reference) or 295 K (CHARMS source reference) miss this EFL measurably.
    if "thermo_optic" in source:
        old_relative_n = project_relative_index({k: v for k, v in source.items() if k != "thermo_optic"}, wavelength)
    else:
        old_relative_n = source_law(source, wavelength, float(source["dispersion"]["reference_temperature_k"])) / air_index(wavelength)
    old_efl = 100.0 * (old_relative_n - 1.0) / (n - 1.0)  # old radius designed at source T, run at 20 C
    previous_efl = 99.990941 if material["material_id"] == "OHARA_S-FPL53" else 100.025223
    assert old_efl == pytest.approx(previous_efl, abs=1e-6)  # independently reproduces the observed wrong 20 C EFL


def test_native_achromat_uses_20c_indices_at_all_three_design_wavelengths():
    crown = glass()
    flint = {"material_id": "N-BK7", "display_name": "N-BK7 stand-in", "valid_band_nm": crown["valid_band_nm"],
             "dispersion": {"kind": "SELLMEIER", "b": ["1.03961212", "0.231792344", "1.01046945"],
                            "c_um2": ["0.00600069867", "0.0200179144", "103.560653"]}}
    build = build_native_model("lens.achromat", {"crown_material": crown, "flint_material": flint,
                                                  "focal_length_mm": "100", "diameter_mm": "10", "crown_shape_factor": "-0.5"})
    saved = ProjectDocumentV2.model_validate(add_to_project(
        {"schema": "opensci.project", "project_schema_version": "2.0"}, build, instance_id="DBL"))
    materials = {m.material_id: m.model_dump(mode="json", exclude_none=True) for m in saved.model_materials}
    surfaces = build.optical_model.model.surfaces
    wavelengths = [float(v) for v in build.design["wavelengths_nm"]]
    focal_positions = []
    for wavelength, reported_crown, reported_flint in zip(
            wavelengths, build.design["crown_indices"], build.design["flint_indices"]):
        n1 = project_relative_index(materials[crown["material_id"]], wavelength)
        n2 = project_relative_index(materials[flint["material_id"]], wavelength)
        assert float(reported_crown) == pytest.approx(n1, abs=5e-12)
        assert float(reported_flint) == pytest.approx(n2, abs=5e-12)
        # Independent paraxial y,nu trace for a ray starting at height 1, slope 0.
        height, nu, previous_z = 1.0, 0.0, 0.0
        media = (1.0, n1, n2, 1.0)
        for i, surface in enumerate(surfaces):
            z = float(surface.vertex_z_mm)
            if i:
                height += (z - previous_z) * nu / media[i]
            curvature = 0.0 if surface.radius_mm is None else 1.0 / float(surface.radius_mm)
            nu -= (media[i + 1] - media[i]) * curvature * height
            previous_z = z
        if wavelength == wavelengths[1]:
            assert -1.0 / nu == pytest.approx(100.0, abs=3e-6)
        focal_positions.append(-height / nu)
    assert focal_positions[0] == pytest.approx(focal_positions[2], abs=3e-5)


def test_native_plain_material_without_thermal_data_keeps_its_source_index():
    material = glass()
    material.pop("thermo_optic")
    wavelength = WAVELENGTHS[material["material_id"]]
    build = build_native_model("lens.singlet", {"material": material, "design_wavelength_nm": str(wavelength),
                                                 "focal_length_mm": "100", "diameter_mm": "10", "form": "PLANO_EXIT"})
    assert float(build.optical_model.model.surfaces[0].radius_mm) / (project_relative_index(material, wavelength) - 1) == pytest.approx(
        100.0, abs=3e-6)
