"""Bulk-loss coverage for a 1064 nm Boyd–Kleinman BBO doubler.

The same regression can run against an installed os-optics wheel with pytest
and this file alone.
"""

from __future__ import annotations

import math

import pytest

from os_optics import EngineConfiguration, ValidationRequest, builders as B, validate
from os_optics.contracts import (
    CircularAperture,
    ConstantIndexMaterial,
    EngineeringComponentSnapshot,
    JonesVector,
    PlaneSurface,
    ProjectSnapshot,
    RefractInteraction,
    Target,
    UniaxialMaterial,
)
from os_optics.contracts.materials import InternalTransmittanceLoss, SecondHarmonicGeneration, SellmeierIndexLaw
from os_optics.contracts.optics import SurfaceScatter
from os_optics.local.sample_emission import excitation_at
from os_optics.results.irradiance import irradiance_map


def _bbo(**loss):
    # Tamosauskas et al., Opt. Mater. Express 8, 1410 (2018). Cut for type I phase matching at 1064 nm.
    ordinary = SellmeierIndexLaw(b=(0.90291, 0.83155, 0.76536), c_um2=(0.003926, 0.018786, 60.01))
    extraordinary = SellmeierIndexLaw(b=(1.151075, 0.21803, 0.656), c_um2=(0.007142, 0.02259, 263.0))

    def inverse_n_squared(law, wavelength_um):
        l2 = wavelength_um ** 2
        return 1 / (1 + sum(b * l2 / (l2 - c) for b, c in zip(law.b, law.c_um2)))

    no2_inv = inverse_n_squared(ordinary, 0.532)
    sin2 = (inverse_n_squared(ordinary, 1.064) - no2_inv) / (inverse_n_squared(extraordinary, 0.532) - no2_inv)
    return UniaxialMaterial(
        material_id="BBO", valid_wavelength_nm=(400.0, 1100.0), ordinary=ordinary, extraordinary=extraordinary,
        optic_axis=(math.sqrt(1 - sin2), math.sqrt(sin2), 0.0),
        nonlinear=SecondHarmonicGeneration(fundamental_waves="oo", harmonic_wave="e", d_eff_pm_per_V=2.0,
                                           focusing="boyd_kleinman"), **loss,
    )


def _loss(green=None):
    wavelengths, transmission = (900.0, 1100.0), (1.0, 1.0)
    if green is not None:
        wavelengths, transmission = (532.0, *wavelengths), (green, *transmission)
    return InternalTransmittanceLoss(wavelength_nm=wavelengths, internal_transmittance=transmission,
                                     thickness_mm=(2.0,) * len(wavelengths))


def _trace(material, polarization_deg=90.0, *, upstream=None, front_scatter=None, extra_sources=(), extra_targets=()):
    def face(sid, x, nx):
        return PlaneSurface(surface_id=sid, origin_mm=(x, 0.0, 0.0), normal=(nx, 0.0, 0.0),
                            aperture=CircularAperture(radius_mm=5.0), medium_positive="air", medium_negative="BBO",
                            interaction=RefractInteraction(scatter=front_scatter if sid == "front" else None))

    source = B.laser("L", wavelength_nm=1064.0, waist_radius_mm=0.05, position=(-20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                     waist_position_z_mm=20.0, polarization=JonesVector.linear(polarization_deg), power_W=1.0)
    slab = EngineeringComponentSnapshot(instance_id="X", pose=B.Pose3D(), surfaces=(face("front", 0.0, -1.0), face("back", 5.0, 1.0)))
    observation = B.observation_plane("OUT", position=(45.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=100.0)
    materials, components = (material,), (source, slab)
    if upstream is not None:
        materials = (*materials, upstream)
        window = B.window("window", material=upstream.material_id, position=(-10.0, 0.0, 0.0), thickness_mm=2.0, clear_radius_mm=5.0)
        components = (source, window, slab)
    project = ProjectSnapshot(materials=materials, components=(*components, *extra_sources), observations=(observation,), targets=(
        Target(target_id="efficiency", metric="shg_conversion_efficiency", device_id="X", minimum=1e-5),
        Target(target_id="efficiency_alias", metric="nonlinear_conversion_efficiency", device_id="X", minimum=1e-5),
        Target(target_id="exit_power", metric="shg_power_W", device_id="X", minimum=1e-5),
        Target(target_id="green_power", metric="power_W", observation_id="OUT", source_id="L", wavelength_nm=532.0, minimum=1e-5),
        Target(target_id="pump_power", metric="power_W", observation_id="OUT", source_id="L", wavelength_nm=1064.0, minimum=0.8),
        Target(target_id="pump_length", metric="geometric_path_length_mm", observation_id="OUT", wavelength_nm=1064.0,
               parameters={"path_id": "L#0"}, minimum=64.0),
        *extra_targets,
    ))
    return validate(ValidationRequest(project=project, configuration=EngineConfiguration(run_layout=False, run_control=False)))


def _targets(result):
    return {t.target_id: t for t in result.targets}


@pytest.mark.parametrize("loss", [{"bulk_loss": _loss()}, {}], ids=["green-outside-table", "unstated-scalar"])
def test_unknown_bulk_loss_withholds_conversion_and_derived_pump_power(loss):
    result = _trace(_bbo(**loss))
    (row,) = result.nonlinear_conversions
    assert row.status.value == "UNKNOWN"
    assert row.harmonic_power_W is row.conversion_efficiency is row.depletion_estimate is row.harmonic_path_id is None
    assert row.fundamental_power_W == pytest.approx(0.9392261157328622, rel=1e-12)
    assert row.converting_power_W == pytest.approx(row.fundamental_power_W, rel=1e-12)
    assert "532 nm" in row.reason
    assert ("1064 nm" in row.reason) == (not loss)
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" and i.location == "surface:X/front" for i in result.issues)

    targets = _targets(result)
    for key in ("efficiency", "efficiency_alias", "exit_power", "green_power", "pump_power"):
        target = targets[key]
        assert target.status.value == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
        assert target.value is None and "BULK_LOSS_NOT_ESTABLISHED" in target.issue_codes
    assert targets["pump_length"].status.value == "PASS"
    assert targets["pump_length"].value == pytest.approx(65.0)

    (pump,) = result.observations[0].arrivals
    assert pump.beam.wavelength_nm == 1064.0 and pump.beam.power_W is None
    assert pump.beam.optical_path_length_mm is not None
    assert pump.power_upper_bound_W == pytest.approx(row.fundamental_power_W, rel=1e-12)
    harmonic = next(line for line in result.observations[0].power_by_wavelength if line.wavelength_nm == 532.0)
    assert harmonic.power_W is None and not harmonic.complete
    (ledger,) = result.power_budget.lines
    assert "NONLINEAR_CONVERSION" not in ledger.sinks and "ESCAPED" not in ledger.sinks
    assert ledger.sinks["BULK_LOSS_UNKNOWN"] == pytest.approx(row.converting_power_W, rel=1e-12)
    assert ledger.sinks["UNTRACED_REFLECTION"] == pytest.approx(1.0 - row.fundamental_power_W, rel=1e-12)
    assert ledger.conserved and result.power_budget.conserved


@pytest.mark.parametrize(("loss", "power", "efficiency", "status"), [
    pytest.param({"bulk_loss": _loss(0.5)}, 8.070639369859546e-6, 8.592860903960452e-6, "FAIL", id="known-green-absorption"),
    pytest.param({"bulk_loss": _loss(1.0)}, 1.7695219356467773e-5, 1.884021223436754e-5, "PASS", id="measured-unit-table"),
    pytest.param({"absorption_coefficient_per_mm": 0.0}, 1.7695219356467773e-5, 1.884021223436754e-5, "PASS", id="explicit-zero"),
])
def test_established_bulk_loss_keeps_quantitative_conversion(loss, power, efficiency, status):
    result = _trace(_bbo(**loss))
    (row,) = result.nonlinear_conversions
    assert row.status.value == "PASS"
    assert row.harmonic_power_W == pytest.approx(power, rel=1e-9)
    assert row.conversion_efficiency == pytest.approx(efficiency, rel=1e-9)
    assert row.harmonic_power_W / row.fundamental_power_W == pytest.approx(efficiency, rel=1e-9)
    targets = _targets(result)
    assert targets["exit_power"].value == pytest.approx(power, rel=1e-9)
    assert targets["efficiency"].value == targets["efficiency_alias"].value == pytest.approx(efficiency, rel=1e-9)
    assert all(targets[key].status.value == status for key in ("efficiency", "efficiency_alias", "exit_power", "green_power"))
    assert targets["pump_length"].value == pytest.approx(65.0)
    pump, harmonic = result.power_budget.lines
    assert pump.sinks["NONLINEAR_CONVERSION"] == harmonic.input_power_W
    if status == "FAIL":
        assert harmonic.input_power_W == pytest.approx(4.577725705460641e-5, rel=1e-9)
    else:
        assert harmonic.input_power_W == pytest.approx(power, rel=1e-9)
    assert pump.conserved and harmonic.conserved and result.power_budget.conserved


def test_missing_harmonic_loss_does_not_open_the_other_pump_mode():
    # Mixed polarization supplies both eigenwaves. Type I converts the ordinary wave only.
    unknown = _trace(_bbo(bulk_loss=_loss()), polarization_deg=45.0)
    known = _trace(_bbo(absorption_coefficient_per_mm=0.0), polarization_deg=45.0)
    (row,) = unknown.nonlinear_conversions
    assert row.status.value == "UNKNOWN" and 0.0 < row.converting_power_W < row.fundamental_power_W
    pump = [a for a in unknown.observations[0].arrivals if a.beam.wavelength_nm == 1064.0]
    (ordinary,) = [a for a in pump if a.beam.power_W is None]
    (extraordinary,) = [a for a in pump if a.beam.power_W is not None]
    expected = next(a for a in known.observations[0].arrivals if a.lineage == extraordinary.lineage)
    assert ordinary.power_upper_bound_W == pytest.approx(row.converting_power_W, rel=1e-12)
    assert extraordinary.beam.power_W == pytest.approx(expected.beam.power_W, rel=1e-12)
    assert unknown.power_budget.conserved


def test_known_crystal_loss_does_not_make_a_bounded_upstream_pump_exact():
    window = ConstantIndexMaterial(material_id="G", refractive_index=1.5, valid_wavelength_nm=(400.0, 1100.0))
    material = _bbo(absorption_coefficient_per_mm=0.0)
    unknown = _trace(material, upstream=window)
    known = _trace(material, upstream=window.model_copy(update={"absorption_coefficient_per_mm": 0.0}))
    (row,) = unknown.nonlinear_conversions
    (reference,) = known.nonlinear_conversions
    assert row.status.value == "UNKNOWN" and "fundamental power entering the crystal" in row.reason
    assert row.harmonic_power_W is row.conversion_efficiency is row.harmonic_path_id is None
    assert row.fundamental_power_W is row.converting_power_W is None
    assert "fundamental_power_W" not in row.model_dump() and "converting_power_W" not in row.model_dump()
    for key in ("efficiency", "exit_power", "green_power"):
        assert _targets(unknown)[key].status.value == "UNKNOWN"
        assert "BULK_LOSS_NOT_ESTABLISHED" in _targets(unknown)[key].issue_codes
    assert _targets(unknown)["pump_length"].value == _targets(known)["pump_length"].value == pytest.approx(65.0)
    assert reference.status.value == "PASS" and reference.harmonic_power_W > 0.0
    assert not any(line.generated_from is not None for line in unknown.power_budget.lines)
    assert unknown.power_budget.conserved and known.power_budget.conserved


def test_unknown_conversion_keeps_scattered_and_reflected_entrance_watts():
    scatter = SurfaceScatter(fraction=0.05, sample_count=16)
    result = _trace(_bbo(bulk_loss=_loss()), front_scatter=scatter)
    (row,) = result.nonlinear_conversions
    (pump,) = result.power_budget.lines
    assert row.status.value == "UNKNOWN"
    assert row.fundamental_power_W == pytest.approx(0.95 * 0.9392261157328622, rel=1e-12)
    assert pump.sinks["BULK_LOSS_UNKNOWN"] == pytest.approx(row.fundamental_power_W, rel=1e-12)
    assert pump.sinks["UNTRACED_REFLECTION"] == pytest.approx(0.95 - row.fundamental_power_W, rel=1e-12)
    assert pump.sinks["ESCAPED"] == pytest.approx(scatter.fraction, rel=1e-12)
    assert all(power >= 0.0 for power in pump.sinks.values())
    assert pump.conserved and result.power_budget.conserved


def test_unknown_harmonic_does_not_hide_an_independent_green_crossing():
    green = B.laser("green", wavelength_nm=532.0, waist_radius_mm=0.5, power_W=0.01,
                    position=(10.0, 2.0, 0.0), direction=(1.0, 0.0, 0.0))
    common = dict(metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.005)
    targets = (
        Target(target_id="all_green", **common),
        Target(target_id="green_path", parameters={"path_id": "green#0"}, **common),
        Target(target_id="green_crossing", parameters={"path_id": "green#0", "crossing_index": 0}, **common),
    )
    result = _trace(_bbo(bulk_loss=_loss()), extra_sources=(green,), extra_targets=targets)
    by_id = _targets(result)
    assert by_id["all_green"].status.value == "UNKNOWN" and by_id["all_green"].value is None
    assert "BULK_LOSS_NOT_ESTABLISHED" in by_id["all_green"].issue_codes
    for name in ("green_path", "green_crossing"):
        exact = by_id[name]
        assert exact.status.value == "PASS" and exact.value == pytest.approx(0.01)
        assert "BULK_LOSS_NOT_ESTABLISHED" not in exact.issue_codes
    observation = result.observations[0]
    harmonic = next(line for line in observation.power_by_wavelength if line.wavelength_nm == 532.0)
    assert harmonic.power_W is None and not harmonic.complete
    assert excitation_at(observation, 532.0) == (None, {}, "the power of 532.0 nm arriving at the plane is not established")
    whole = irradiance_map(result, "OUT", wavelength_nm=532.0, samples=9)
    assert whole["status"] == "OK" and not whole["map_complete"] and whole["untraced_power_W"] is None
    separate = irradiance_map(result, "OUT", wavelength_nm=532.0, source_id="green", samples=9)
    assert separate["status"] == "OK" and separate["scope"]["source_id"] == "green"
