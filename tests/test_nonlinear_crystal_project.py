"""A Project's declared NONLINEAR_CRYSTAL end to end: the contract (``NonlinearCrystalModelV1``) compiles to the crystal slab
with the interaction on its Engine crystal material (``local/compile.py``), the run generates the second harmonic, the
results (Python, json_api, command line) state each pass, and targets on the conversion are graded.

References: BBO (Tamosauskas et al., Opt. Mater. Express 8, 1410 (2018)) cut at the type I phase-matching angle computed here
from its Sellmeier formulas; the near-field harmonic of the collimated Gaussian beam with the closed-form walk-off factor;
the Fresnel transmittance at normal incidence.
"""

from __future__ import annotations

import json
import math

import pytest

from os_optics.cli import main as cli_main
from os_optics.local import json_api, run_optical_project
from os_optics.results import load_run

from declared_helpers import declared_component, with_models
from local_helpers import plane, project, source

BO = (["0.90291", "0.83155", "0.76536"], ["0.003926", "0.018786", "60.01"])
BE = (["1.151075", "0.21803", "0.656"], ["0.007142", "0.02259", "263"])
LAMBDA = 1064.0
LENGTH, WAIST = 5.0, 0.05


def _n(law, wavelength_nm: float) -> float:
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(float(b) * l2 / (l2 - float(c)) for b, c in zip(*law)))


def _theta() -> float:
    no1, no2, ne2 = _n(BO, LAMBDA), _n(BO, LAMBDA / 2), _n(BE, LAMBDA / 2)
    return math.asin(math.sqrt((no1 ** -2 - no2 ** -2) / (ne2 ** -2 - no2 ** -2)))


def _bbo() -> dict:
    return {"material_id": "BBO", "display_name": "beta-BBO", "dispersion": {"kind": "SELLMEIER", "b": BO[0], "c_um2": BO[1]},
            "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"},
            "basis": "Tamosauskas et al., Opt. Mater. Express 8, 1410 (2018); ideal zero bulk loss assumed for test",
            "absorption_coefficient_per_mm": "0",
            "birefringence": {"kind": "UNIAXIAL", "extraordinary": {"kind": "SELLMEIER", "b": BE[0], "c_um2": BE[1]}}}


def _design(targets=None, power_w: float = 1.0, **crystal) -> dict:
    theta = _theta()
    model = {"model_kind": "NONLINEAR_CRYSTAL", "medium": "BBO", "length_mm": "5", "width_mm": "4", "height_mm": "4",
             "fundamental_waves": "OO", "harmonic_wave": "E", "d_eff_pm_per_V": "2", "nonlinear_basis": "d_eff stated for the test",
             "valid_band_nm": {"min_nm": "500", "max_nm": "1100", "status": "SPECIFIED"}, "status": "DECLARED", **crystal}
    declared = {"model_id": "SHG", "display_name": "BBO type I 1064 nm doubler",
                "optic_axes": [{"material_id": "BBO", "direction": [repr(math.sin(theta)), "0", repr(math.cos(theta))]}], "model": model}
    # the source's waist (in air) on the crystal's entrance face; polarization along global z: the ordinary wave
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA, waist_mm=WAIST, waist_position_mm=100.0,
                               polarization_deg=90.0, power_w=power_w),
                        declared_component("X", "other_optical", "SHG")],
                       [plane("out", (60.0, 0.0, 0.0), radius_mm=25.0)], targets or [])
    return with_models(document, [declared], [_bbo()])


def _expected_harmonic_W(power_w: float = 1.0) -> float:
    """Near field: 2 w^2 d^2 L^2 P^2 / (n_o^2 n_e(theta) eps0 c^3 pi w_m^2) times the closed-form walk-off factor (w_m: the
    radius in the crystal's middle)."""
    theta = _theta()
    no1, no2, ne2 = _n(BO, LAMBDA), _n(BO, LAMBDA / 2), _n(BE, LAMBDA / 2)
    n2 = 1.0 / math.sqrt(math.cos(theta) ** 2 / no2 ** 2 + math.sin(theta) ** 2 / ne2 ** 2)
    s, c = math.sin(theta), math.cos(theta)
    rho = math.atan((no2 ** 2 - ne2 ** 2) * s * c / (ne2 ** 2 * c * c + no2 ** 2 * s * s))
    t = math.tan(rho) * LENGTH / WAIST
    walk = (math.sqrt(math.pi) * t * math.erf(t) - 1.0 + math.exp(-t * t)) / (t * t)
    zr = math.pi * no1 * WAIST ** 2 / (LAMBDA * 1e-6)
    w_mid = WAIST * math.sqrt(1.0 + (LENGTH / 2 / zr) ** 2) * 1e-3
    p = power_w * 4 * no1 / (1 + no1) ** 2
    omega = 2 * math.pi * 299792458.0 / (LAMBDA * 1e-9)
    k = 2 * omega ** 2 * (2e-12) ** 2 / (no1 ** 2 * n2 * 8.8541878128e-12 * 299792458.0 ** 3)
    return k * p * p * (LENGTH * 1e-3) ** 2 / (math.pi * w_mid ** 2) * walk


def test_a_declared_bbo_doubler_generates_its_harmonic_through_the_project_path():
    targets = [{"target_id": "eta", "metric": "shg_conversion_efficiency", "device_instance_id": "X", "minimum": 1e-5},
               {"target_id": "green", "metric": "power_W", "observation_id": "out", "wavelength_nm": 532.0, "minimum": 1e-5}]
    document = _design(targets)
    checked = json_api.check_project(document)
    assert checked["ok"]
    assert checked["declared_models"] == [{"instance_id": "X", "optical_model_id": "SHG", "model_kind": "NONLINEAR_CRYSTAL",
                                          "engine_surfaces": ["S1", "S2"], "engine_materials": ["project:BBO@X"]}]
    run = run_optical_project(document)
    result = run.result
    (row,) = result.nonlinear_conversions
    assert row.status.value == "PASS" and row.element_id == "X" and abs(row.phase_mismatch_rad) < 1e-6
    assert row.harmonic_power_W == pytest.approx(_expected_harmonic_W(), rel=1e-6)
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["eta"].status.value == "PASS" and by_id["eta"].value == pytest.approx(row.conversion_efficiency, rel=1e-15)
    green = next(a for a in result.observations[0].arrivals if a.beam.wavelength_nm == 532.0)
    assert by_id["green"].status.value == "PASS" and by_id["green"].value == pytest.approx(green.beam.power_W, rel=1e-12)
    assert [step.split(":")[1] for step in green.lineage] == ["second_harmonic_e", "refract"]
    assert result.power_budget.conserved
    assert {line.spectral_index: line.generated_from for line in result.power_budget.lines} == {0: None, 1: "SRC#0"}
    assert "nonlinear_crystals" in json_api.capabilities()["project_features"]


def test_the_results_state_each_pass_in_python_json_api_and_the_command_line(tmp_path, capsys):
    run = run_optical_project(_design(), output_dir=tmp_path / "run")
    rows = load_run(tmp_path / "run").nonlinear()
    assert len(rows) == 1 and rows[0]["status"] == "PASS" and rows[0]["harmonic_wavelength_nm"] == 532.0
    assert rows == load_run(tmp_path / "run").nonlinear("X")
    files = json_api.run_project(_design())["files"]
    reply = json_api.run_results(files, {"kind": "nonlinear", "element": "X"})
    assert reply["ok"] and reply["result"][0]["harmonic_power_W"] == pytest.approx(rows[0]["harmonic_power_W"], rel=1e-12)
    assert cli_main(["results", "nonlinear", str(tmp_path / "run")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0]["angular_acceptance_mrad"] == pytest.approx(rows[0]["angular_acceptance_mrad"], rel=1e-12)
    table = load_run(tmp_path / "run").table("nonlinear")[0]
    assert table[0]["conversion_efficiency"] == pytest.approx(run.result.nonlinear_conversions[0].conversion_efficiency, rel=1e-12)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    (model,) = manifest["element_models"]
    assert model["model_kind"] == "NONLINEAR_CRYSTAL" and model["second_harmonic"] == {
        "focusing": "near_field", "fundamental_waves": "oo", "harmonic_wave": "e"}


def test_a_conversion_outside_the_model_grades_its_targets_out_of_scope():
    """A 3 kW beam: the depletion estimate exceeds the tolerance, the pass and the target on it are out of model scope."""
    targets = [{"target_id": "eta", "metric": "shg_conversion_efficiency", "device_instance_id": "X", "minimum": 1e-5}]
    result = run_optical_project(_design(targets, power_w=3000.0)).result
    (row,) = result.nonlinear_conversions
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "pump depletion is not modelled" in row.reason
    (target,) = result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and target.issue_codes == ("NONLINEAR_CONVERSION_OUT_OF_SCOPE",)


def test_temperature_and_quasi_phase_matching_reach_the_engine_material():
    """temperature_c / thermo_optic and the interaction compile onto the crystal's Engine material (the numbers as stated)."""
    from os_optics.local import lower_project

    thermo = {"reference_temperature_c": "20", "rows": [{"wavelength_nm": "500", "dno_dT_per_K": "-0.0000166", "dne_dT_per_K": "-0.0000093"},
                                                        {"wavelength_nm": "1100", "dno_dT_per_K": "-0.0000166", "dne_dT_per_K": "-0.0000093"}]}
    lowered = lower_project(_design(temperature_c="45", thermo_optic=thermo, focusing="BOYD_KLEINMAN"))
    (crystal,) = [m for m in lowered.request.project.materials if m.material_id == "project:BBO@X"]
    assert crystal.nonlinear.fundamental_waves == "oo" and crystal.nonlinear.harmonic_wave == "e"
    assert crystal.nonlinear.focusing == "boyd_kleinman" and crystal.nonlinear.d_eff_pm_per_V == 2.0
    assert crystal.temperature.temperature_c == 45.0 and crystal.temperature.thermo_optic[0].dne_dT_per_K == -9.3e-6
    row = run_optical_project(_design(temperature_c="45", thermo_optic=thermo)).result.nonlinear_conversions[0]
    assert row.temperature_c == 45.0 and row.temperature_acceptance_K is not None and row.phase_mismatch_rad != 0.0


def test_an_extended_emitters_sampled_rays_pass_the_crystal_unconverted():
    """Sampled rays of an extended emitter carry no beam: they cross the crystal as a linear crystal (no conversion row, no
    harmonic line, no nonlinear issue); a Gaussian beam of the same run is converted."""
    document = _design()
    document["components"].append({
        "instance_id": "EM", "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "Extended emitter."},
        "nominal_pose": {"position_mm": [-50.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
        "source_model": {"model_kind": "EXTENDED_EMITTER", "emitting_surface": {"shape": "DISC", "diameter_mm": 0.02},
                         "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 64, "seed": 1}, "wavelength_nm": LAMBDA,
                         "power_w": 1.0}})
    result = run_optical_project(document).result
    assert [row.source_id for row in result.nonlinear_conversions] == ["SRC"]
    assert {line.source_id for line in result.power_budget.lines if line.generated_from is not None} == {"SRC"}
    assert sum(i.code.value == "NONLINEAR_CONVERSION_OUT_OF_SCOPE" for i in result.issues) == 0


def test_a_projects_source_linewidth_and_pulse_reach_the_conversion():
    """The source model's linewidth_nm and pulse (review findings: the first crashed the run, the second was graded PASS with the
    continuous-wave value of the average power): a 0.1 nm line converts as the single frequency (BBO 5 mm: the line is well
    inside the spectral acceptance) and its harmonic's coherence length is half the fundamental's; a 10 ns / 10 Hz pulse train
    is out of model scope with its target."""
    targets = [{"target_id": "eta", "metric": "shg_conversion_efficiency", "device_instance_id": "X", "minimum": 1e-6}]
    plain = run_optical_project(_design(targets)).result.nonlinear_conversions[0]
    lined = _design(targets)
    lined["components"][0]["source_model"]["linewidth_nm"] = 0.1
    result = run_optical_project(lined).result
    (row,) = result.nonlinear_conversions
    assert row.status.value == "PASS" and row.harmonic_power_W == plain.harmonic_power_W
    lengths = {a.beam.wavelength_nm: a.coherence_length_mm for a in result.observations[0].arrivals}
    fundamental = 2.0 * math.sqrt(math.log(2.0)) / math.pi * LAMBDA ** 2 / 0.1 * 1e-6
    assert lengths[LAMBDA] == pytest.approx(fundamental, rel=1e-12) and lengths[LAMBDA / 2] == pytest.approx(fundamental / 2, rel=1e-12)
    pulsed = _design(targets)
    pulsed["components"][0]["source_model"]["pulse"] = {"repetition_rate_hz": 10.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM"}
    result = run_optical_project(pulsed).result
    (row,) = result.nonlinear_conversions
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "pulsed" in row.reason
    (target,) = result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and target.issue_codes == ("NONLINEAR_CONVERSION_OUT_OF_SCOPE",)
