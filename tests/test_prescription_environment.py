"""The wavelength convention and reference environment of an imported prescription (assignment
section 1, import part).

``csv-surface-table/1`` states how it writes wavelengths (``wavelengths_in``: ``VACUUM``, the default, or ``AIR`` at the
file's ``temperature_c`` / ``pressure_pa`` | ``pressure_atm``: OpticStudio's convention, wavelengths measured in air at
the system temperature and pressure); air wavelengths are converted once to vacuum wavelengths with the shared air model.
The reference here is the Edlen (1953) dispersion formula written in its usual wavenumber form with OpticStudio's
temperature / pressure scaling, solved by its own iteration -- not the contract's code. The import carries the environment
and ``merge_into_project`` sets it (refusing a Project whose environment differs).
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import pytest

import material_catalog36 as M
from os_optics.prescription import PrescriptionImportError, import_prescription, merge_into_project
from test_material_resolution_import import base_project

CATALOG = M.catalog()
SURFACES = "surface,radius,thickness,material,semi_diameter\n1,60,6,N-BK7,10\n2,-60,90,,10\n"


def edlen_vacuum_nm(air_nm: float, temperature_c: float, pressure_pa: float) -> float:
    """The vacuum wavelength of an air wavelength (Edlen 1953, wavenumber form; OpticStudio's T / P scaling)."""

    vacuum = air_nm
    for _ in range(20):
        sigma2 = (1000.0 / vacuum) ** 2  # (1 / um)^2
        n15 = 1.0 + 1e-8 * (6432.8 + 2949810.0 / (146.0 - sigma2) + 25540.0 / (41.0 - sigma2))
        n = 1.0 + (n15 - 1.0) * (pressure_pa / 101325.0) / (1.0 + 0.0034785 * (temperature_c - 15.0))
        vacuum = air_nm * n
    return vacuum


def rx(*directives: str, body: str = SURFACES) -> str:
    return "".join(f"#! {d}\n" for d in directives) + body


def imported(text: str, **kwargs):
    kwargs.setdefault("catalog", CATALOG)
    return import_prescription(text, **kwargs)


def codes(text: str) -> list[str]:
    with pytest.raises(PrescriptionImportError) as exc:
        imported(text)
    return [p.code.value for p in exc.value.problems]


BAND = ("wavelength_min_nm = 500", "wavelength_max_nm = 1550")


def test_absent_keys_mean_vacuum_wavelengths_and_no_stated_environment():
    result = imported(rx(*BAND))
    assert result.environment is None and result.report["environment"] is None and "environment" not in result.as_dict()
    assert result.report["wavelengths"]["written_in"] == "VACUUM" and not result.report["wavelengths"]["converted_to_vacuum"]
    assert (result.report["wavelength_band_nm"]["min_nm"], result.report["wavelength_band_nm"]["max_nm"]) == ("500", "1550")
    assert any("states no environment" in a for a in result.report["assumptions"])
    assert "environment" not in merge_into_project(base_project(), result).payload()


@pytest.mark.parametrize("temperature, pressure_directive, pressure_pa", [
    ("20", "pressure_atm = 1", 101325.0), ("25", "pressure_pa = 90000", 90000.0), ("-10", "pressure_atm = 0.5", 50662.5)])
def test_air_wavelengths_are_converted_once_at_the_stated_conditions(temperature, pressure_directive, pressure_pa):
    result = imported(rx("wavelengths_in = AIR", f"temperature_c = {temperature}", pressure_directive, *BAND))
    band = result.report["wavelength_band_nm"]
    low, high = (edlen_vacuum_nm(v, float(temperature), pressure_pa) for v in (500.0, 1550.0))
    assert Decimal(band["min_nm"]) == Decimal(repr(low)).quantize(Decimal("0.001"), rounding=ROUND_CEILING)
    assert Decimal(band["max_nm"]) == Decimal(repr(high)).quantize(Decimal("0.001"), rounding=ROUND_FLOOR)
    assert 500.0 < low < 500.2 and 1550.0 < high < 1550.7
    assert result.report["wavelengths"]["written_in"] == "AIR" and result.report["wavelengths"]["converted_to_vacuum"]
    assert Decimal(result.environment.temperature_c) == Decimal(temperature)
    assert Decimal(result.environment.pressure_pa) == Decimal(repr(pressure_pa))
    assert "WAVELENGTHS_CONVERTED" in [w["code"] for w in result.report["warnings"]]
    merged = merge_into_project(base_project(), result)
    assert merged.environment == result.environment and merged.payload()["environment"]["temperature_c"] == temperature


def test_air_without_conditions_is_the_default_air_and_zero_pressure_is_vacuum():
    default = imported(rx("wavelengths_in = air", *BAND))
    assert (default.environment.temperature_c, default.environment.pressure_pa) == ("20", "101325")
    assert default.report["environment"]["stated"] == []
    assert Decimal(default.report["wavelength_band_nm"]["min_nm"]) > 500
    assert any("default reference of the file's air wavelengths" in a for a in default.report["assumptions"])
    vacuum = imported(rx("wavelengths_in = AIR", "pressure_pa = 0", *BAND))
    assert vacuum.environment.is_vacuum and not vacuum.report["wavelengths"]["converted_to_vacuum"]
    assert (vacuum.report["wavelength_band_nm"]["min_nm"], vacuum.report["wavelength_band_nm"]["max_nm"]) == ("500", "1550")
    assert "WAVELENGTHS_CONVERTED" not in [w["code"] for w in vacuum.report["warnings"]]


def test_a_stated_temperature_alone_sets_the_environment_without_converting():
    result = imported(rx("temperature_c = 30", *BAND))
    assert (result.environment.temperature_c, result.environment.pressure_pa) == ("30", "101325")
    assert result.report["wavelengths"]["written_in"] == "VACUUM"
    assert result.report["wavelength_band_nm"]["min_nm"] == "500"


def test_coating_wavelengths_are_converted_too():
    coatings = ("[coatings]\nname,wavelength_nm,reflectance,transmittance,kind,min_aoi_deg,max_aoi_deg\n"
                "AR1,450,0.01,0.99,ANTI_REFLECTION,0,30\nAR1,800,0.01,0.99,,,\n")
    body = "surface,radius,thickness,material,semi_diameter,coating\n1,60,6,N-BK7,10,AR1\n2,-60,90,,10,AR1\n" + coatings
    result = imported(rx("wavelengths_in = AIR", "temperature_c = 20", "pressure_atm = 1", body=body))
    (coating,) = result.model_coatings
    nodes = [Decimal(v) for v in coating.spectrum.wavelength_nm]
    expected = [Decimal(repr(edlen_vacuum_nm(v, 20.0, 101325.0))) for v in (450.0, 800.0)]
    assert all(abs(n - e) <= Decimal("0.000002") for n, e in zip(nodes, expected)), (nodes, expected)
    plain = imported(rx(body=body))
    assert [Decimal(v) for v in plain.model_coatings[0].spectrum.wavelength_nm] == [Decimal(450), Decimal(800)]


def test_a_merge_never_changes_the_medium_of_optics_already_in_the_project():
    warm = imported(rx("temperature_c = 25", "pressure_atm = 1"), model_id="warm")
    standard = imported(rx("wavelengths_in = AIR"), model_id="std")
    project = merge_into_project(base_project(), warm, instance_id="W")  # a Project without optics takes it
    assert project.environment.temperature_c == "25"
    with pytest.raises(PrescriptionImportError) as exc:
        merge_into_project(project.payload(), standard, instance_id="S")  # 20 C into a 25 C Project
    assert exc.value.codes[0].value == "PROJECT_INVALID" and "the Project states air at 25 C" in str(exc.value)
    plain = merge_into_project(base_project(), imported(rx(), model_id="plain"), instance_id="P")  # optics in the default air
    assert plain.environment is None
    with pytest.raises(PrescriptionImportError) as again:
        merge_into_project(plain.payload(), warm, instance_id="W")
    assert "states none: its optics operate in the default air at 20 C and 101325 Pa" in str(again.value)
    same = merge_into_project(plain.payload(), standard, instance_id="S")  # the default air, stated: no change of medium
    assert same.environment.temperature_c == "20" and same.environment.pressure_pa == "101325"
    assert merge_into_project(same.payload(), imported(rx(), model_id="later"), instance_id="L").environment == same.environment


def test_environment_directive_problems():
    assert codes(rx("wavelengths_in = VACUO")) == ["DIRECTIVE_INVALID"]
    assert codes(rx("pressure_pa = 101325", "pressure_atm = 1")) == ["DIRECTIVE_INVALID"]
    assert codes(rx("pressure_pa = -1")) == ["DIRECTIVE_INVALID"]
    assert codes(rx("temperature_c = -300")) == ["DIRECTIVE_INVALID"]
    assert codes(rx("wavelengths_in = AIR", "wavelength_min_nm = 150", "wavelength_max_nm = 600")) == ["VALUE_INVALID"]


def test_a_defined_material_states_its_own_law_convention():
    table = "[materials]\nname,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm{column}\nMYGLASS,SELLMEIER,{coefficients},400,1000{value}\n"
    coefficients = ",".join((*M.BK7[0], *M.BK7[1]))
    body = "surface,radius,thickness,material,semi_diameter\n1,60,6,MYGLASS,10\n2,-60,90,,10\n"
    air_law = imported(rx(body=body + table.format(column=",index_reference", coefficients=coefficients, value=",AIR")))
    (material,) = air_law.model_materials
    assert material.index_reference.wavelength == "AIR" and material.index_reference.index == "RELATIVE_TO_AIR"
    assert material.index_reference.source_band_nm == ("400", "1000")
    assert Decimal(material.valid_band_nm.min_nm) > 400 and Decimal(material.valid_band_nm.max_nm) > 1000  # its vacuum band
    vacuum_law = imported(rx(body=body + table.format(column="", coefficients=coefficients, value="")))
    assert vacuum_law.model_materials[0].index_reference is None
    assert (vacuum_law.model_materials[0].valid_band_nm.min_nm, vacuum_law.model_materials[0].valid_band_nm.max_nm) == ("400", "1000")
    blank_in_air_file = imported(rx("wavelengths_in = AIR", body=body + table.format(column="", coefficients=coefficients, value="")))
    assert "MATERIAL_CONVENTION" in [w["code"] for w in blank_in_air_file.report["warnings"]]
    assert codes(rx(body=body + table.format(column=",index_reference", coefficients=coefficients, value=",SOMETIMES"))) == ["MATERIAL_INVALID"]


def test_a_window_with_a_vacuum_gap_imports_and_keeps_its_media_without_a_catalog():
    """A prescription can put a real window between room air and an evacuated gap without assigning a glass to VACUUM.
    A second window closes the gap. The native model must keep VACUUM, not replace it with the environment's AIR."""

    body = ("surface,radius,thickness,material,semi_diameter\n"
            "1,,2,N-BK7,5\n"
            "2,,10,VACUUM,5\n"
            "3,,2,N-BK7,5\n"
            "4,,10,AIR,5\n")
    prescription = imported(rx("wavelength_min_nm = 550", "wavelength_max_nm = 550", body=body))
    assert len(prescription.model_materials) == 1
    assert prescription.model_materials[0].material_id == "N-BK7"
    assert [m.model.media for m in prescription.optical_models if m.model.model_kind == "SURFACE_SEQUENCE"] == [
        ("N-BK7", "VACUUM", "N-BK7")]
    assert prescription.prescription.surfaces[1].material.name == "VACUUM"
    assert prescription.report["counts"]["materials"] == 1


def test_vacuum_between_different_glasses_stays_in_a_native_sequence():
    body = ("surface,radius,thickness,material,semi_diameter\n"
            "1,,2,N-BK7,5\n"
            "2,,10,VACUUM,5\n"
            "3,,2,OHARA_S-FPL53,5\n"
            "4,,10,AIR,5\n")
    imported_model = imported(rx("wavelength_min_nm = 550", "wavelength_max_nm = 550", body=body))
    assert [m for m in imported_model.model_materials if m.material_id in {"N-BK7", "OHARA_S-FPL53"}]
    assert all(material.material_id != "VACUUM" for material in imported_model.model_materials)
    assert imported_model.prescription.surfaces[1].material.name == "VACUUM"


def test_a_bare_surface_cannot_change_room_air_into_vacuum():
    body = ("surface,radius,thickness,material,semi_diameter\n"
            "1,,10,VACUUM,5\n"
            "2,,10,AIR,5\n")
    with pytest.raises(PrescriptionImportError) as exc:
        imported(rx(body=body))
    assert "without a modelled boundary" in str(exc.value)


def test_the_reference_formula_here_is_the_contracts_air_model():
    """Cross-check of the two writings (not of the import): the wavenumber form above and the contract's air model."""

    from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

    for nm in (300.0, 587.5618, 1064.0, 10000.0):
        vacuum = edlen_vacuum_nm(nm, 20.0, 101325.0)
        assert math.isclose(vacuum / nm, air_index(vacuum), rel_tol=0, abs_tol=1e-12)
