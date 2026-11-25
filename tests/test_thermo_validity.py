"""Pinned-source thermo singularities are not a valid index, but the base law survives.

The coefficients are read from the committed redistributable material records
(``tests/fixtures/materials/redistributable_materials.json``), not re-fitted or replaced.
OHARA's 0.311-um and CDGM's 0.312-um TD poles have only three meaningful
places; Frey 2006 Si Table 5 gives 1e-4 average fit residual and its third
moving pole crosses the measured band near 21.38386 K.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

from catalog_store_helpers import FIXTURE_CATALOG
from os_optics._vendor.shared_contracts.design_catalog.optical_catalog import PublicMaterialV1
from os_optics.cli import main
from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.contracts.materials import ConstantIndexMaterial, SchottThermoOptic
from os_optics.local.compile import engine_material
from os_optics.native_library.materials import absolute_index
from os_optics.optics.materials.group import group_dispersion
from os_optics.optics.materials.index import build_evaluator
from os_optics.optics.materials.physics import material_point


_LIBRARY = Path(__file__).resolve().parent / "fixtures" / "materials" / "redistributable_materials.json"


@pytest.fixture(scope="module")
def records() -> dict[str, dict]:
    return {r["material_id"]: r for r in json.loads(_LIBRARY.read_text(encoding="utf-8"))["records"]}


def _index(record: dict, wavelength_nm: float, temperature_c: float):
    # Exercise the actual public contract and its Engine lowering, not a
    # hand-constructed or modified material with a substituted dispersion.
    public = PublicMaterialV1.model_validate(record["public"])
    model = engine_material(public, temperature_c=temperature_c)
    return build_evaluator(model, CFG).index(wavelength_nm), model


@pytest.mark.parametrize("material_id,law_pole_nm,reference_c,warm_c", [
    ("OHARA_L-BBH2", 311.0, 25.0, 30.0),
    ("CDGM_D-ZK79", 312.0, 20.0, 30.0),
    ("CDGM_D-ZK79-25", 312.0, 20.0, 30.0),
])
def test_schott_pole_is_absent_only_from_thermal_correction(records, material_id, law_pole_nm, reference_c, warm_c):
    record = records[material_id]
    public = record["public"]
    assert public["thermo_optic"]["lambda_tk_um"] == str(law_pole_nm / 1000)
    assert public["thermo_optic"]["reference_temperature_c"] == str(int(reference_c))
    reference_air = public["index_reference"]["reference_air"]
    from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

    # Obtain vacuum wavelength of the pinned *air-coordinate* TD pole, without
    # using the material index evaluator under test.
    vacuum_nm = law_pole_nm
    for _ in range(8):
        vacuum_nm = law_pole_nm * air_index(vacuum_nm, float(reference_air["temperature_c"]),
                                            float(reference_air["pressure_pa"]))
    for displacement in (-0.001, 0.0, 0.001):
        (n_base, base_issue), base = _index(record, vacuum_nm + displacement, reference_c)
        assert base_issue is None and n_base is not None and math.isfinite(n_base)
        renamed_base = base.model_copy(update={"material_id": f"user:RENAMED-{material_id}"})
        assert build_evaluator(renamed_base, CFG).index(vacuum_nm + displacement) == (n_base, None)
        (n_warm, warm_issue), warm = _index(record, vacuum_nm + displacement, warm_c)
        assert n_warm is None and warm_issue.code.value == "MATERIAL_INVALID"
        assert "pole neighbourhood" in warm_issue.message
        assert group_dispersion(warm, vacuum_nm + displacement, CFG) is None
        assert group_dispersion(base, vacuum_nm + displacement, CFG) is not None
        # A copied physical law must keep its domain even under a completely
        # different name; both native evaluator and public material_point use it.
        renamed_model = warm.model_copy(update={"material_id": f"user:RENAMED-{material_id}"})
        renamed_n, renamed_issue = build_evaluator(renamed_model, CFG).index(vacuum_nm + displacement)
        assert renamed_n is None and renamed_issue.code.value == "MATERIAL_INVALID"
        renamed_public = PublicMaterialV1.model_validate(public).model_copy(update={"material_id": f"RENAMED-{material_id}"})
        renamed_point = material_point(renamed_public, vacuum_nm + displacement, temperature_c=warm_c)
        assert renamed_point.n_abs is renamed_point.n_group is None and "pole neighbourhood" in renamed_point.reason
    # Half a last stated 0.001-um pole unit, on the law's air wavelength:
    # immediately across that boundary the original correction is available.
    for displacement in (-0.501, 0.501):
        (n, issue), _ = _index(record, vacuum_nm + displacement, warm_c)
        assert issue is None and n is not None and math.isfinite(n)
    # No material-wide disablement: 400 nm and 632.8 nm retain the original
    # formula at both temperatures and its ordinary group derivatives.
    for wavelength in (400.0, 632.8):
        (at_ref, issue), _ = _index(record, wavelength, reference_c)
        (at_warm, issue_warm), model = _index(record, wavelength, warm_c)
        assert issue is issue_warm is None and 1 < at_ref < 4 and 1 < at_warm < 4
        assert abs(at_ref - at_warm) < 0.01
        assert group_dispersion(model, wavelength, CFG) is not None


@pytest.mark.parametrize(("e0", "e1", "temperature_c"), [(0.0, 0.0, 30.0), (2.0**-10, -(2.0**-12), 24.0)])
def test_a_zero_schott_e_numerator_has_no_pole_or_group_singularity(e0, e1, temperature_c):
    """The remaining D term is finite at the pole, including when nonzero E coefficients cancel."""

    thermo = SchottThermoOptic(d0=1e-6, d1=0.0, d2=0.0, e0=e0, e1=e1,
                               lambda_tk_um=0.311, reference_temperature_c=20.0)
    model = ConstantIndexMaterial(material_id="COPIED_D_ONLY", refractive_index=1.6,
                                  valid_wavelength_nm=(300.0, 1000.0), thermo_optic=thermo,
                                  temperature_c=temperature_c)
    expected = 1.6 + ((1.6 * 1.6 - 1.0) / (2.0 * 1.6)) * (1e-6 * (temperature_c - 20.0))
    for wavelength in (310.999, 311.0, 311.1, 311.499, 311.501, 632.8):
        n, issue = build_evaluator(model, CFG).index(wavelength)
        assert issue is None and n == pytest.approx(expected, abs=1e-14)
        group = group_dispersion(model, wavelength, CFG)
        assert group is not None and group.n == pytest.approx(expected, abs=1e-14)
        assert group.n_group == pytest.approx(expected, abs=1e-14) and group.k2 == pytest.approx(0.0, abs=1e-14)
    at_reference, issue = build_evaluator(model.model_copy(update={"temperature_c": 20.0}), CFG).index(311.0)
    assert issue is None and at_reference == 1.6


@pytest.mark.parametrize(("pole_um", "lower_nm", "upper_nm"), [
    (0.311, 310.5, 311.5),
    (0.102, 101.5, 102.5),  # The upper endpoint rounds upward when multiplied by 0.001 in binary.
    (0.103, 102.5, 103.5),  # The float below the lower endpoint rounds to it in that conversion.
])
def test_schott_engine_uses_inclusive_margin_in_the_laws_nanometres(pole_um, lower_nm, upper_nm):
    thermo = SchottThermoOptic(d0=0.0, d1=0.0, d2=0.0, e0=1e-6, e1=0.0,
                               lambda_tk_um=pole_um, reference_temperature_c=20.0)
    model = ConstantIndexMaterial(material_id="POLE_MARGIN", refractive_index=1.6,
                                  valid_wavelength_nm=(100.0, 1000.0), thermo_optic=thermo,
                                  temperature_c=30.0)
    evaluator = build_evaluator(model, CFG)
    for edge, outward in ((lower_nm, -math.inf), (upper_nm, math.inf)):
        for invalid in (edge, math.nextafter(edge, -outward)):
            n, issue = evaluator.index(invalid)
            assert n is None and issue.code.value == "MATERIAL_INVALID"
            assert "pole neighbourhood" in issue.message
            assert group_dispersion(model, invalid, CFG) is None
        outside = math.nextafter(edge, outward)
        n, issue = evaluator.index(outside)
        assert issue is None and math.isfinite(n)
        group = group_dispersion(model, outside, CFG)
        assert group is not None and math.isfinite(group.n_group)


def test_ohara_reference_index_is_unchanged_at_reported_wavelength(records):
    (n, issue), _ = _index(records["OHARA_L-BBH2"], 311.088184522, 25.0)
    assert issue is None and n == pytest.approx(2.634014680, abs=2e-8)
    public = PublicMaterialV1.model_validate(records["OHARA_L-BBH2"]["public"])
    base = material_point(public, 311.088184522, temperature_c=25.0)
    assert base.n_abs == pytest.approx(n)
    law_um = base.law_wavelength_nm / 1000.0
    n_law = base.n_abs / public.index_reference.reference_air.index(base.wavelength_nm)
    assert public.thermo_optic.delta_n(n_law, law_um, 25.0) == 0.0
    for warm in (30.0, 40.0):
        (value, refused), _ = _index(records["OHARA_L-BBH2"], 311.088184522, warm)
        assert value is None and refused.code.value == "MATERIAL_INVALID"
        point = material_point(public, 311.088184522, temperature_c=warm)
        assert point.n_abs is point.n_group is None and "pole neighbourhood" in point.reason
        with pytest.raises(ValueError, match="pole neighbourhood"):
            public.thermo_optic.delta_n(n_law, law_um, warm)
    with pytest.raises(ValueError, match="pole neighbourhood"):
        absolute_index(public, public.dispersion, Decimal("311.088184522"))
    design = absolute_index(public, public.dispersion, Decimal("632.8"))
    assert float(design) == pytest.approx(material_point(public, 632.8, temperature_c=20.0).n_abs, abs=1e-13)
    ordinary = material_point(public, 632.8, temperature_c=40.0)
    assert ordinary.reason is None and ordinary.n_abs is not None and ordinary.n_group is not None


@pytest.fixture
def charms_catalog(tmp_path, records):
    document = json.loads(FIXTURE_CATALOG.read_bytes())
    document["schema_version"] = "3.6"
    document["materials"] += [records[material_id]["public"] for material_id in ("SI_FREY2006", "GE_FREY2006")]
    path = tmp_path / "charms-catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(("material_id", "temperature_c", "expected_n", "expected_group"), [
    ("SI_FREY2006", "-253.15", 3.39839087754813, 3.41538558447826),
    ("SI_FREY2006", "26.85", 3.42671960911619, 3.44456641581698),
    ("GE_FREY2006", "-253.15", 3.94800425703609, 3.99173417932607),
    ("GE_FREY2006", "26.85", 4.02782051075831, 4.07844077286548),
])
def test_charms_catalog_cli_accepts_exact_endpoints_and_refuses_adjacent_outside_values(
    charms_catalog, capsys, material_id, temperature_c, expected_n, expected_group,
):
    args = ["catalog", "--catalog", str(charms_catalog), "material", material_id, "--at-nm", "4000"]
    assert main([*args, f"--temperature-c={temperature_c}"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["resolution"]["resolved"] == material_id
    point = document["ir"]["at"]
    assert point["n"] == pytest.approx(expected_n, abs=5e-10)
    assert point["n_group"] == pytest.approx(expected_group, abs=5e-10)
    assert math.isfinite(point["gdd_mm2_per_mm"])
    assert "reason" not in point

    direction = -math.inf if temperature_c == "-253.15" else math.inf
    outside_c = math.nextafter(float(temperature_c), direction)
    assert main([*args, f"--temperature-c={outside_c}"]) == 0
    refused = json.loads(capsys.readouterr().out)["ir"]["at"]
    assert refused["n"] is None and "n_group" not in refused
    assert refused["reason"].startswith("MATERIAL_OUT_OF_RANGE:")


def test_silicon_moving_pole_rejects_only_local_two_dimensional_neighbourhood(records):
    record = records["SI_FREY2006"]
    assert record["public"]["dispersion"]["reference_temperature_k"] == "295"
    for temperature_c in (-253.15, -251.15, -243.15):  # 20, 22 and 30 K
        (n, issue), model = _index(record, 4000.0, temperature_c)
        assert issue is None and n == pytest.approx(3.398, abs=0.001)
        assert group_dispersion(model, 4000.0, CFG) is not None
        renamed = model.model_copy(update={"material_id": "MY_SILICON_CRYOGENIC"})
        assert build_evaluator(renamed, CFG).index(4000.0) == (n, None)
    for kelvin in (21.3838, 21.38385, 21.38386, 21.38387, 21.3839):
        (n, issue), model = _index(record, 4000.0, kelvin - 273.15)
        assert n is None and issue.code.value == "MATERIAL_INVALID"
        assert group_dispersion(model, 4000.0, CFG) is None
        point = material_point(PublicMaterialV1.model_validate(record["public"]), 4000.0, temperature_c=kelvin - 273.15)
        assert point.n_abs is point.n_group is None and "MATERIAL_INVALID" in point.reason
        # The same source coefficients retain the invalid neighbourhood under
        # a copied ID, through both the native and public evaluation paths.
        renamed_model = model.model_copy(update={"material_id": "MY_SILICON_CRYOGENIC"})
        renamed_n, renamed_issue = build_evaluator(renamed_model, CFG).index(4000.0)
        assert renamed_n is None and renamed_issue.code.value == "MATERIAL_INVALID"
        renamed_public = PublicMaterialV1.model_validate(record["public"]).model_copy(update={"material_id": "MY_SILICON_CRYOGENIC"})
        renamed_point = material_point(renamed_public, 4000.0, temperature_c=kelvin - 273.15)
        assert renamed_point.n_abs is None and "MATERIAL_INVALID" in renamed_point.reason
    # A moving pole does not erase the material or even the whole temperature
    # slice: the shorter wavelength remains within the source fit's residual.
    (n, issue), _ = _index(record, 1100.0, 21.38 - 273.15)
    assert issue is None and 3.4 < n < 3.6
    (n, issue), _ = _index(record, 4000.0, 21.38 - 273.15)
    assert n is None and issue.code.value == "MATERIAL_INVALID"
    (n, issue), _ = _index(record, 4000.0, 21.37 - 273.15)
    assert issue is None and 3.39 < n < 3.41
