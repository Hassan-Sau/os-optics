"""The weak-signal measurement chain: sample emission, detector bandwidth and noise, chopper modulation and lock-in.

Every reference below is computed here, independently of the Engine code that is tested:

* sample emission: photon numbers from ``P lambda / (h c)``, the absorbed share ``1 - 10^-A``, the quantum yield and the photon
  shares; Raman wavelengths from wavenumbers and ``N 4 pi (d sigma / d Omega) L``; the collected share of an isotropic point
  emitter through a lens of clear radius ``a`` at its focal distance ``f``: ``(1 - f / sqrt(f^2 + a^2)) / 2``, compared within
  four times the stratified sampling's reported sigma;
* detector noise: ``R = QE q lambda / (h c)``; shot-noise-limited ``SNR = sqrt(I / (2 q B))``; a first-order response's
  noise-equivalent bandwidth by numerical quadrature of ``1 / (1 + (f / f_c)^2)``;
* lock-in: the output filter's noise-equivalent bandwidth by numerical quadrature of ``(1 + (2 pi f tau)^2)^-n`` (1/(4 tau) for
  one section); the square wave's fundamental by a discrete Fourier transform of a sampled 0/1 wave (2/pi at duty 1/2); and a
  time-domain simulation of a lock-in (chopped current with white, cyclostationary shot noise, an in-phase reference and an
  RC output filter), whose output mean and standard deviation the closed forms must reproduce within its statistical error;
* the detector's response at the modulation frequency: ``|H(f)|`` of an analog single-pole filter from ``scipy.signal.freqs``;
  the bandwidth condition ``f <= 2 B / pi`` of a noise-equivalent bandwidth from the quadrature of ``|H|^2`` of single-pole and
  Butterworth responses (``scipy.signal.butter``);
* the sampling sigma of detection quantities: finite differences of the closed forms above in each sampled line's power;
* the stratification behind the seed statement: every coordinate of the shifted Sobol points holds one point per ``1/N`` cell.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import integrate, signal

from declared_helpers import declared_component, thin_lens_model
from local_helpers import ALONG_X, ALONG_Y, plane, project, source

from os_optics.cli import main
from os_optics.local import LocalRunError, json_api, run_optical_project
from os_optics.results import load_run
from os_optics.optics.rays.emitter_samples import sobol_points
from os_optics.validation.detection import (
    LOCK_IN_AVERAGING_TOO_SHORT,
    MODULATION_ABOVE_DETECTOR_BANDWIDTH,
    PHOTOCURRENT_ABOVE_LINEAR_RANGE,
    lock_in_enbw_Hz,
    square_fundamental,
    square_noise_weight,
)

Q = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0
K_B = 1.380649e-23
CHAR = {"kind": "DATASHEET", "reference": "photodiode datasheet rev B"}
QE = {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 700.0, "qe": 0.8}], "source": CHAR}
SAMPLE_RAYS = 16384


def responsivity(qe: float, wavelength_nm: float) -> float:
    return qe * Q * wavelength_nm * 1e-9 / (H * C)


def fluorescence(**changes) -> dict:
    model = {"model_kind": "FLUORESCENCE", "excitation": {"observation_id": "sample_plane", "wavelength_nm": 488.0},
             "absorbance": 0.01, "quantum_yield": 0.9,
             "emission_lines": [{"wavelength_nm": 520.0, "photon_fraction": 0.6}, {"wavelength_nm": 540.0, "photon_fraction": 0.4}],
             "emitting_surface": {"shape": "DISC", "diameter_mm": 0.02}, "angular_law": "ISOTROPIC",
             "sampling": {"sample_count": SAMPLE_RAYS, "seed": 1}}
    model.update(changes)
    return model


def sample_component(model: dict) -> dict:
    return {"instance_id": "S", "optical_role": "source", "custom_part": {"custom_part_key": "cuvette", "description": "Dye cuvette."},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}, "source_model": model}


def electrical(**changes) -> dict:
    value = {"bandwidth_Hz": 10000.0, "bandwidth_definition": "FIRST_ORDER_3DB", "dark_current_A": 1e-12, "load_resistance_ohm": 1e8,
             "temperature_K": 295.0, "source": CHAR}
    value.update(changes)
    return value


def lock_in(**changes) -> dict:
    value = {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": 0.5, "source_ids": ["L"]},
             "time_constant_s": 0.1, "filter_order": 2}
    value.update(changes)
    return value


def chain_project(sample: dict | None = None, detection: dict | None = None, targets: list | None = None, *,
                  laser_power_w: float = 0.001) -> dict:
    """Excitation laser along +x through the sample at the origin into a beam dump; a collection lens (f = 50 mm, clear radius
    10 mm) at y = 50 mm with the sample at its focus, and an absorbing detector of radius 20 mm at y = 150 mm."""

    laser = source("L", (-100.0, 0.0, 0.0), wavelength_nm=488.0, waist_mm=0.5, power_w=laser_power_w)
    lens = {"instance_id": "LENS", "optical_role": "lens", "custom_part": {"custom_part_key": "collector", "description": "Collection lens."},
            "optical_model_id": "thin", "nominal_pose": {"position_mm": [0.0, 50.0, 0.0], "quaternion_wxyz": list(ALONG_Y)}}
    detector = {**plane("D", (0.0, 150.0, 0.0), ALONG_Y, radius_mm=20.0), "absorbing": True}
    if detection is not None:
        detector["detection"] = detection
    observations = [plane("sample_plane", (-1.0, 0.0, 0.0), radius_mm=5.0), {**plane("dump", (100.0, 0.0, 0.0), radius_mm=20.0), "absorbing": True},
                    detector]
    document = project([laser, sample_component(sample or fluorescence()), lens], observations, targets)
    document["optical_models"] = [thin_lens_model("thin", "50", clear="10", edge="12.7")]
    document["design_intent"] = "OPTICAL_SYSTEM"
    return document


def first_order_response(f_hz: float, fc_hz: float) -> float:
    """``|H(f)|`` of an analog single-pole low-pass of -3 dB frequency ``fc_hz`` (scipy's frequency response)."""

    _, h = signal.freqs([1.0], [1.0 / (2.0 * math.pi * fc_hz), 1.0], worN=[2.0 * math.pi * f_hz])
    return float(abs(h[0]))


def collected_share(focal_mm: float = 50.0, clear_mm: float = 10.0) -> float:
    return (1.0 - focal_mm / math.sqrt(focal_mm ** 2 + clear_mm ** 2)) / 2.0


def detection_of(run, oid: str = "D") -> dict:
    observation = next(o for o in run.result.observations if o.observation_id == oid)
    return {q.name: q for q in observation.detection.quantities}


def sample_record(run, instance_id: str = "S"):
    return next(c for c in run.request.project.components if c.instance_id == instance_id).sample_emission


# ------------------------------------------------------------------------------------------------- sample emission
def test_fluorescence_line_powers_follow_photon_numbers_and_the_collection_matches_the_solid_angle():
    run = run_optical_project(chain_project(detection={"detector": {"quantum_efficiency": QE}}))
    record = sample_record(run)
    # (independent arithmetic: absorbed photons x quantum yield x photon share, each photon at its own wavelength)
    p_ex, lam_ex = 1e-3, 488.0
    absorbed = 1.0 - 10.0 ** -0.01
    photons_ex = p_ex * lam_ex * 1e-9 / (H * C)
    assert record.excitation_power_W == pytest.approx(p_ex, rel=1e-12)
    assert [s.source_id for s in record.excitation_by_source] == ["L"]
    assert record.absorbed_power_W == pytest.approx(p_ex * absorbed, rel=1e-12)
    for line, (lam, share) in zip(record.lines, ((520.0, 0.6), (540.0, 0.4))):
        photons = photons_ex * absorbed * 0.9 * share
        assert line.photon_rate_per_s == pytest.approx(photons, rel=1e-12)
        assert line.power_W == pytest.approx(photons * H * C / (lam * 1e-9), rel=1e-12)
        assert line.power_W == pytest.approx(p_ex * absorbed * 0.9 * share * lam_ex / lam, rel=1e-12)
    assert record.emitted_photon_rate_per_s == pytest.approx(photons_ex * absorbed * 0.9, rel=1e-12)
    # the emitter traced carries exactly those lines; the detector collects the lens's solid angle
    emitter = next(c for c in run.request.project.components if c.instance_id == "S").emitter
    assert [(s.wavelength_nm, s.weight) for s in emitter.spectrum] == [(line.wavelength_nm, line.power_W) for line in record.lines]
    detector = next(o for o in run.result.observations if o.observation_id == "D")
    assert detector.quantity_state.value == "KNOWN"
    reference = collected_share()
    for line in detector.sampled_light:
        assert abs(line.collection_fraction - reference) <= 4.0 * line.collection_fraction_sigma, (line.wavelength_nm, line.collection_fraction)
    # the excitation plane sees the same 1 mW in the run with the emission (checked by the run itself), the dump the laser
    dump = next(o for o in run.result.observations if o.observation_id == "dump")
    assert next(e.power_W for e in dump.power_by_wavelength if e.wavelength_nm == 488.0) == pytest.approx(p_ex, rel=1e-12)


def test_raman_bands_shift_by_wavenumber_and_scatter_n_sigma_l():
    raman = {"model_kind": "RAMAN", "excitation": {"observation_id": "sample_plane", "wavelength_nm": 488.0},
             "raman_lines": [{"shift_cm1": 1000.0, "differential_cross_section_cm2_per_sr": 3e-30},
                             {"shift_cm1": 2900.0, "differential_cross_section_cm2_per_sr": 1e-29}],
             "number_density_per_cm3": 3.3e22, "path_length_mm": 10.0,
             "emitting_surface": {"shape": "DISC", "diameter_mm": 0.02}, "angular_law": "ISOTROPIC", "sampling": {"sample_count": 1024, "seed": 3}}
    run = run_optical_project(chain_project(raman, laser_power_w=0.05))
    record = sample_record(run)
    assert record.process == "RAMAN" and record.absorbed_fraction is None
    for line, (shift, dsigma) in zip(record.lines, ((1000.0, 3e-30), (2900.0, 1e-29))):
        lam = 1.0 / (1.0 / 488.0 - shift * 1e-7)
        conversion = 3.3e22 * 4.0 * math.pi * dsigma * 1.0  # L = 1 cm
        assert line.wavelength_nm == pytest.approx(lam, rel=1e-12)
        assert line.raman_shift_cm1 == shift
        assert line.power_W == pytest.approx(0.05 * conversion * 488.0 / lam, rel=1e-12)
    assert any(line.counted_crossing_count for line in run.result.observations[2].sampled_light)  # traced to the detector


def test_the_excitation_is_the_named_wavelength_delivered_to_the_plane():
    """A second laser of another wavelength crossing the sample does not excite it; a laser that misses the plane excites nothing."""

    document = chain_project()
    document["components"].insert(1, source("L2", (-100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5, power_w=0.002))
    record = sample_record(run_optical_project(document))
    assert record.excitation_power_W == pytest.approx(1e-3, rel=1e-12)
    assert [s.source_id for s in record.excitation_by_source] == ["L"]
    missed = chain_project(detection={"detector": {"quantum_efficiency": QE, "electrical": electrical()}, "lock_in": lock_in()})
    missed["components"][0]["nominal_pose"]["position_mm"] = [-100.0, 30.0, 0.0]  # 30 mm beside the 5 mm plane
    run = run_optical_project(missed)
    record = sample_record(run)
    assert record.excitation_power_W == 0.0 and record.emitted_power_W == 0.0
    assert next(c for c in run.request.project.components if c.instance_id == "S").emitter is None
    quantities = detection_of(run)
    assert quantities["photocurrent_A"].value == 0.0 and quantities["lock_in_snr"].value == 0.0


def test_an_excitation_that_is_not_known_refuses_the_project():
    document = chain_project(fluorescence(excitation={"observation_id": "sample_plane", "wavelength_nm": 500.0}))
    with pytest.raises(LocalRunError) as error:
        run_optical_project(document)
    (problem,) = error.value.problems
    assert problem.code.value == "SAMPLE_EXCITATION_NOT_KNOWN" and "no source of the Project emits 500.0 nm" in problem.message
    assert problem.details == {"observation_id": "sample_plane", "wavelength_nm": 500.0, "quantity_state": "KNOWN"}
    # the laser passes a transmission grating that states no order efficiency: the power of its zero order at the sample is
    # not known, so neither is the emission
    dark = chain_project()
    dark["components"].append(declared_component("G", "other_optical", "g", (-50.0, 0.0, 0.0), ALONG_X))
    dark["optical_models"].append({"model_id": "g", "model": {
        "model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "300", "orders": [0],
        "clear_half_widths_mm": ["12", "12"], "edge_half_widths_mm": ["12.5", "12.5"], "status": "SPECIFIED"}})
    with pytest.raises(LocalRunError) as error:
        run_optical_project(dark)
    (problem,) = error.value.problems
    assert problem.code.value == "SAMPLE_EXCITATION_NOT_KNOWN" and "not complete" in problem.message


# ------------------------------------------------------------------------------------------------- detector noise
def _direct_project(detection: dict, power_w: float = 1e-9, targets: list | None = None) -> dict:
    """A 1 nW laser at 600 nm straight onto an absorbing detector plane (every watt counted: a known power)."""

    detector = {**plane("D", (100.0, 0.0, 0.0), radius_mm=20.0), "absorbing": True, "detection": detection}
    return project([source("L", (-100.0, 0.0, 0.0), wavelength_nm=600.0, waist_mm=0.5, power_w=power_w)], [detector], targets)


def test_shot_noise_limited_snr_and_a_first_order_noise_bandwidth():
    detection = {"detector": {"quantum_efficiency": QE, "electrical": electrical(dark_current_A=0.0, load_resistance_ohm=None,
                                                                                    temperature_K=None, bandwidth_definition="NOISE_EQUIVALENT",
                                                                                    bandwidth_Hz=1000.0)}}
    quantities = detection_of(run_optical_project(_direct_project(detection)))
    current = responsivity(0.8, 600.0) * 1e-9
    assert quantities["photocurrent_A"].value == pytest.approx(current, rel=1e-12)
    assert quantities["detector_snr"].value == pytest.approx(math.sqrt(current / (2.0 * Q * 1000.0)), rel=1e-12)
    assert quantities["noise_equivalent_bandwidth_Hz"].value == 1000.0
    # first order: the noise bandwidth is the integral of |H|^2, pi/2 f_c
    fc = 1000.0
    numeric, _ = integrate.quad(lambda f: 1.0 / (1.0 + (f / fc) ** 2), 0.0, math.inf)
    first = detection_of(run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE, "electrical": electrical(bandwidth_Hz=fc)}})))
    assert first["noise_equivalent_bandwidth_Hz"].value == pytest.approx(numeric, rel=1e-9)
    floor2 = 2.0 * Q * 1e-12 + 4.0 * K_B * 295.0 / 1e8
    assert first["dark_noise_density_A_per_rtHz"].value == pytest.approx(math.sqrt(floor2), rel=1e-12)
    assert first["detector_snr"].value == pytest.approx(current / math.sqrt((2.0 * Q * current + floor2) * numeric), rel=1e-9)


def test_a_noise_equivalent_power_sets_the_floor_at_its_own_wavelength():
    elec = electrical(dark_current_A=None, load_resistance_ohm=None, temperature_K=None, nep_W_per_rtHz=2e-14, nep_wavelength_nm=500.0,
                      bandwidth_definition="NOISE_EQUIVALENT", bandwidth_Hz=100.0)
    quantities = detection_of(run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE, "electrical": elec}})))
    floor = 2e-14 * responsivity(0.8, 500.0)
    current = responsivity(0.8, 600.0) * 1e-9
    assert quantities["dark_noise_density_A_per_rtHz"].value == pytest.approx(floor, rel=1e-12)
    assert quantities["detector_snr"].value == pytest.approx(current / math.sqrt((2.0 * Q * current + floor ** 2) * 100.0), rel=1e-12)
    # referred to the light at 600 nm, the same floor is a smaller power (higher responsivity)
    assert quantities["noise_equivalent_power_W_per_rtHz"].value == pytest.approx(2e-14 * 500.0 / 600.0, rel=1e-12)


def test_what_the_detector_does_not_state_stays_unavailable():
    run = run_optical_project(_direct_project({"detector": {"electrical": electrical()}},
                                              targets=[{"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 1.0}]))
    quantities = detection_of(run)
    assert quantities["detector_bandwidth_Hz"].status == "EXTERNALLY_CHARACTERIZED"
    assert (quantities["photocurrent_A"].status, quantities["photocurrent_A"].reason, quantities["photocurrent_A"].missing) == (
        "UNAVAILABLE", "PARAMETER_NOT_STATED", ("quantum_efficiency",))
    (target,) = run.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("DETECTION_QUANTITY_UNAVAILABLE",))
    assert "PARAMETER_NOT_STATED" in target.message
    narrow = {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 550.0, "qe": 0.8}], "source": CHAR}
    outside = detection_of(run_optical_project(_direct_project({"detector": {"quantum_efficiency": narrow, "electrical": electrical()}})))
    assert outside["photocurrent_A"].reason == "WAVELENGTH_OUTSIDE_QE_TABLE"
    assert outside["detected_power_W"].value == pytest.approx(1e-9, rel=1e-12)


# ------------------------------------------------------------------------------------------------- lock-in closed forms
@pytest.mark.parametrize(("order", "closed"), [(1, 1.0 / 4.0), (2, 1.0 / 8.0), (3, 3.0 / 32.0), (4, 5.0 / 64.0)])
def test_lock_in_noise_bandwidth_of_n_rc_sections(order: int, closed: float):
    tau = 0.3
    numeric, _ = integrate.quad(lambda f: (1.0 + (2.0 * math.pi * f * tau) ** 2) ** -order, 0.0, math.inf)
    assert lock_in_enbw_Hz(tau, order) == pytest.approx(closed / tau, rel=1e-14)
    assert lock_in_enbw_Hz(tau, order) == pytest.approx(numeric, rel=1e-9)


@pytest.mark.parametrize("duty", [0.5, 0.25, 0.1])
def test_square_wave_fundamental_and_noise_weight_from_sampled_waves(duty: float):
    samples = 1 << 16
    t = (np.arange(samples) + 0.5) / samples  # one period, the open window centred on t = 0
    wave = ((t < duty / 2.0) | (t > 1.0 - duty / 2.0)).astype(float)
    fundamental = 2.0 * abs(np.fft.rfft(wave)[1]) / samples
    assert square_fundamental(duty) == pytest.approx(fundamental, abs=2e-4)
    weight = float(np.mean(wave * 2.0 * np.cos(2.0 * np.pi * t) ** 2))
    assert square_noise_weight(duty) == pytest.approx(weight, abs=2e-4)
    if duty == 0.5:
        assert square_fundamental(duty) == pytest.approx(2.0 / math.pi, rel=1e-14) and square_noise_weight(duty) == pytest.approx(0.5, abs=1e-14)


def test_a_time_domain_lock_in_reproduces_signal_noise_and_snr():
    """A simulated lock-in: a 0/1-chopped current of duty 1/4 with white shot noise of density 2 c I(t) (cyclostationary: it is
    on only while the light is) and a steady floor, an in-phase reference sqrt(2) cos, one RC section. Its output's mean and
    standard deviation match the closed forms within the simulation's own statistical error."""

    rng = np.random.default_rng(495)
    fs, f, tau, duty = 32000.0, 1000.0, 0.02, 0.25
    current, c, floor2 = 1.0, 2.0e-3, 1.0e-4  # (arbitrary units: c plays 2q)
    realizations, duration = 128, 1.2
    t = (np.arange(int(duration * fs)) + 0.5) / fs  # (samples centred in their steps: a symmetric open window)
    phase = (t * f) % 1.0
    chopped = current * ((phase < duty / 2.0) | (phase > 1.0 - duty / 2.0))
    sigma = np.sqrt((c * chopped + floor2) * fs / 2.0)  # one-sided density over the Nyquist band
    noisy = chopped + sigma * rng.standard_normal((realizations, t.size))
    mixed = noisy * math.sqrt(2.0) * np.cos(2.0 * np.pi * f * t)
    a = math.exp(-1.0 / (fs * tau))
    x = signal.lfilter([1.0 - a], [1.0, -a], mixed, axis=1)
    picks = np.arange(int(0.2 * fs), t.size, int(0.2 * fs))  # every 10 time constants after settling
    readings = x[:, picks].ravel()
    signal_rms = square_fundamental(duty) * current / math.sqrt(2.0)
    noise = math.sqrt((c * square_noise_weight(duty) * current + floor2) * lock_in_enbw_Hz(tau, 1))
    standard_error = noise / math.sqrt(readings.size)
    assert abs(readings.mean() - signal_rms) < 4.0 * standard_error + 0.005 * signal_rms  # (+ the filtered 2f ripple)
    assert readings.std(ddof=1) == pytest.approx(noise, rel=4.0 / math.sqrt(2.0 * readings.size))
    # the naive mean-current weight (duty, 0.25) instead of 0.41 would be far outside that error
    naive = math.sqrt((c * duty * current + floor2) * lock_in_enbw_Hz(tau, 1))
    assert abs(readings.std(ddof=1) - naive) > 4.0 * noise / math.sqrt(2.0 * readings.size)


# ------------------------------------------------------------------------------------------------- the connected chain
CHAIN_TARGETS = [{"target_id": "snr", "metric": "lock_in_snr", "observation_id": "D", "minimum": 100.0},
                 {"target_id": "bw", "metric": "detector_bandwidth_Hz", "observation_id": "D", "minimum": 5000.0},
                 {"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0},
                 {"target_id": "resolution", "metric": "spectral_resolution_nm", "observation_id": "D", "maximum": 1.0}]


def chain_detection() -> dict:
    return {"detector": {"quantum_efficiency": QE, "electrical": electrical()}, "lock_in": lock_in()}


def test_the_connected_chain_excitation_fluorescence_collection_detector_and_lock_in(tmp_path, capsys):
    run = run_optical_project(chain_project(fluorescence(absorbance=1e-4), chain_detection(), CHAIN_TARGETS), output_dir=tmp_path / "run")
    record = sample_record(run)
    detector = next(o for o in run.result.observations if o.observation_id == "D")
    quantities = {q.name: q for q in detector.detection.quantities}
    # (1) the traced detection, recomputed from the traced line powers with the closed forms
    lines = {line.wavelength_nm: line.power_W for line in detector.sampled_light}
    current = sum(responsivity(0.8, lam) * p for lam, p in lines.items())
    assert quantities["photocurrent_A"].value == pytest.approx(current, rel=1e-12)
    floor2 = 2.0 * Q * 1e-12 + 4.0 * K_B * 295.0 / 1e8
    enbw = 1.0 / (8.0 * 0.1)
    signal_rms = 2.0 / math.pi * current / math.sqrt(2.0)  # chopped at duty 1/2: every watt of it follows the laser
    noise = math.sqrt((2.0 * Q * 0.5 * current + floor2) * enbw)
    response = first_order_response(1000.0, 10000.0)  # (1 kHz chop through the 10 kHz first-order detector)
    assert quantities["lock_in_enbw_Hz"].value == pytest.approx(enbw, rel=1e-14)
    assert quantities["detector_response_at_modulation"].value == pytest.approx(response, rel=1e-12)
    # signal and noise at the detector's output (both through |H(f)|), their ratio that of the input-referred values
    assert quantities["lock_in_signal_A"].value == pytest.approx(signal_rms * response, rel=1e-12)
    assert quantities["lock_in_noise_A"].value == pytest.approx(noise * response, rel=1e-12)
    assert quantities["lock_in_snr"].value == pytest.approx(signal_rms / noise, rel=1e-12)
    assert [(v.parameter, v.within) for v in detector.detection.validity] == [
        ("modulation_frequency_within_detector_bandwidth", True), ("modulation_periods_per_time_constant", True)]
    # the sampling sigma of each sampled line's power, carried to first order into the quantities (finite differences of
    # the closed forms in each line's power)
    line_sigma = {line.wavelength_nm: line.collection_fraction_sigma * line.emitted_W for line in detector.sampled_light}

    def closed(powers: dict[float, float]) -> dict[str, float]:
        i = sum(responsivity(0.8, lam) * p for lam, p in powers.items())
        x = 2.0 / math.pi * i / math.sqrt(2.0)
        total = math.sqrt((2.0 * Q * i + floor2) * math.pi / 2.0 * 10000.0)
        return {"detected_power_W": sum(powers.values()), "photocurrent_A": i, "detector_snr": i / total,
                "lock_in_signal_A": x, "lock_in_snr": x / math.sqrt((2.0 * Q * 0.5 * i + floor2) * enbw)}

    nominal = closed(lines)
    for name, value in nominal.items():
        step = 1e-4
        spread = sum(abs(closed({**lines, lam: lines[lam] + step * s})[name] - value) / step for lam, s in line_sigma.items())
        assert quantities[name].sampling_relative_sigma == pytest.approx(spread / value, rel=1e-5), name
    assert quantities["detected_power_W"].sampling_relative_sigma == pytest.approx(sum(line_sigma.values()) / sum(lines.values()), rel=1e-12)
    assert quantities["lock_in_enbw_Hz"].sampling_relative_sigma is None  # (reads no light)
    assert [c.modulated_fraction for c in detector.detection.contributions] == [1.0, 1.0]
    # (2) against the analytic chain: excitation x absorbed share x yield x photon share x energy ratio x collected share
    absorbed = 1.0 - 10.0 ** -1e-4
    share = collected_share()
    analytic = {lam: 1e-3 * absorbed * 0.9 * f * 488.0 / lam * share for lam, f in ((520.0, 0.6), (540.0, 0.4))}
    sigma = max(line.collection_fraction_sigma for line in detector.sampled_light)
    for lam, p in lines.items():
        assert abs(p - analytic[lam]) <= 4.0 * sigma * record.emitted_power_W
    analytic_current = sum(responsivity(0.8, lam) * p for lam, p in analytic.items())
    analytic_snr = (2.0 / math.pi * analytic_current / math.sqrt(2.0)) / math.sqrt((Q * analytic_current + floor2) * enbw)
    assert quantities["lock_in_snr"].value == pytest.approx(analytic_snr, rel=4.0 * sigma / share)
    # (3) the targets: computed where the model supports them, a truthful refusal where it does not
    targets = {t.target_id: t for t in run.result.targets}
    assert targets["snr"].status.value == "PASS" and targets["snr"].value == quantities["lock_in_snr"].value
    assert targets["bw"].status.value == "PASS" and targets["bw"].value == 10000.0
    assert targets["dc"].status.value == "PASS"
    assert (targets["resolution"].status.value, targets["resolution"].assessment.value) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE")
    # (4) the same from the run directory: results detection / sample-emission, the CLI and the JSON API
    results = load_run(tmp_path / "run")
    (reading,) = results.detection("D")
    assert reading["observation_id"] == "D" and reading["power_state"] == "KNOWN"
    (emission,) = results.sample_emission()
    assert emission["source_id"] == "S" and emission["traced"] is True and emission["emitted_power_W"] == record.emitted_power_W
    assert {row["observation_id"] for row in emission["collection"]} >= {"D"}
    assert main(["results", "detection", str(tmp_path / "run"), "--monitor", "D"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert {q["name"]: q["value"] for q in printed[0]["quantities"]}["lock_in_snr"] == quantities["lock_in_snr"].value
    assert main(["results", "sample-emission", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)[0]["process"] == "FLUORESCENCE"
    files = {p.name: p.read_bytes() for p in (tmp_path / "run").iterdir() if p.is_file()}
    reply = json_api.run_results(files, {"kind": "detection", "monitor": "D"})
    assert reply["ok"] and reply["result"][0]["quantities"] == printed[0]["quantities"]
    assert json_api.run_results(files, {"kind": "sample_emission"})["result"][0]["source_id"] == "S"


def test_a_lock_in_that_averages_too_few_periods_states_no_snr():
    short = chain_detection()
    short["lock_in"]["time_constant_s"] = 0.005  # 5 periods of 1 kHz
    targets = [{"target_id": "snr", "metric": "lock_in_snr", "observation_id": "D", "minimum": 1.0}]
    run = run_optical_project(chain_project(detection=short, targets=targets))
    quantities = detection_of(run)
    assert quantities["lock_in_signal_A"].status == "MODELED"
    assert (quantities["lock_in_snr"].status, quantities["lock_in_snr"].reason) == ("UNAVAILABLE", LOCK_IN_AVERAGING_TOO_SHORT)
    validity = next(o for o in run.result.observations if o.observation_id == "D").detection.validity
    assert [(v.parameter, v.within) for v in validity] == [("modulation_frequency_within_detector_bandwidth", True),
                                                          ("modulation_periods_per_time_constant", False)]
    assert run.result.targets[0].issue_codes == ("DETECTION_QUANTITY_UNAVAILABLE",)


def test_unmodulated_light_adds_noise_and_no_signal():
    """A second, unchopped laser at the sample's excitation wavelength: the sample's emission carries only the chopped share."""

    document = chain_project(detection=chain_detection())
    document["components"].insert(1, source("L2", (-100.0, 0.0, 0.0), wavelength_nm=488.0, waist_mm=0.4, power_w=0.003))
    run = run_optical_project(document)
    record = sample_record(run)
    assert record.excitation_power_W == pytest.approx(0.004, rel=1e-12)
    detector = next(o for o in run.result.observations if o.observation_id == "D")
    (fraction,) = {c.modulated_fraction for c in detector.detection.contributions}
    assert fraction == pytest.approx(0.25, rel=1e-12)
    quantities = {q.name: q for q in detector.detection.quantities}
    current = quantities["photocurrent_A"].value
    assert quantities["modulated_photocurrent_A"].value == pytest.approx(0.25 * current, rel=1e-12)
    floor2 = 2.0 * Q * 1e-12 + 4.0 * K_B * 295.0 / 1e8
    noise = math.sqrt((2.0 * Q * (0.75 * current + 0.5 * 0.25 * current) + floor2) / (8.0 * 0.1))
    assert quantities["lock_in_noise_A"].value == pytest.approx(noise, rel=1e-12)


@pytest.mark.parametrize(("change", "message"), [
    ({"target_id": "x", "metric": "lock_in_snr", "observation_id": "dump", "minimum": 1.0}, "states detection"),
    ({"target_id": "x", "metric": "detector_snr", "observation_id": "D", "source_id": "L", "minimum": 1.0}, "leave source_id"),
])
def test_detection_targets_are_not_assessed_where_they_cannot_be_read(change: dict, message: str):
    run = run_optical_project(chain_project(detection={"detector": {"quantum_efficiency": QE}}, targets=[change]))
    (target,) = run.result.targets
    assert (target.status.value, target.value, target.issue_codes) == ("UNKNOWN", None, ("TARGET_INVALID",))
    assert message in target.message and run.result.outcome.requirements.indeterminate == ("x",)
    run = run_optical_project(chain_project(detection={"detector": {"quantum_efficiency": QE}},
                                            targets=[{"target_id": "x", "metric": "lock_in_snr", "observation_id": "D", "minimum": 1.0}]))
    (target,) = run.result.targets
    assert target.issue_codes == ("TARGET_INVALID",) and "needs a lock-in" in target.message


def test_per_pixel_electrical_response_of_results_detector():
    run = run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE}}))
    detector = {"quantum_efficiency": QE, "pixel_pitch_mm": [2.0, 2.0], "fill_factor": 1.0, "pixel_count": [3, 3],
                "electrical": electrical(bandwidth_definition="NOISE_EQUIVALENT", bandwidth_Hz=100.0)}
    response = load_run(run).detector_response("D", detector, half_extent_mm=4.0, samples=65)
    q = response["quantities"]
    power = np.array(q["pixel_power_W"]["values"])
    current = np.array(q["photocurrent_A"]["values"])
    assert np.allclose(current, responsivity(0.8, 600.0) * power, rtol=1e-12, atol=0.0)
    floor2 = 2.0 * Q * 1e-12 + 4.0 * K_B * 295.0 / 1e8
    noise = np.sqrt((2.0 * Q * current + floor2) * 100.0)
    assert np.allclose(np.array(q["electrical_noise_current_A"]["values"]), noise, rtol=1e-12, atol=0.0)
    assert np.allclose(np.array(q["electrical_snr"]["values"]), current / noise, rtol=1e-12, atol=0.0)
    assert response["inputs"]["electrical"]["status"] == "EXTERNALLY_CHARACTERIZED"
    # without the electrical response the document is the camera response as before: no electrical quantity or input
    plain = load_run(run).detector_response("D", {k: v for k, v in detector.items() if k != "electrical"}, half_extent_mm=4.0, samples=65)
    assert not {"photocurrent_A", "electrical_noise_current_A", "electrical_snr"} & set(plain["quantities"])
    assert "electrical" not in plain["inputs"]
    for name, quantity in plain["quantities"].items():
        assert quantity == q[name], name


# ------------------------------------------------------------------------------------------------- the detector's bandwidth
def _chopped(electrical_changes: dict, frequency_hz: float, *, waveform: str = "SQUARE", targets: list | None = None,
             power_w: float = 1e-9) -> dict:
    """The 1 nW 600 nm laser chopped at ``frequency_hz`` onto the detector, a lock-in of tau = 0.1 s (f tau >= 10 here)."""

    modulation = ({"waveform": "SQUARE", "frequency_Hz": frequency_hz, "duty_cycle": 0.5, "source_ids": ["L"]} if waveform == "SQUARE"
                  else {"waveform": "SINE", "frequency_Hz": frequency_hz, "modulation_depth": 0.6, "source_ids": ["L"]})
    detection = {"detector": {"quantum_efficiency": QE, "electrical": electrical(**electrical_changes)},
                 "lock_in": lock_in(modulation=modulation, time_constant_s=0.1)}
    return _direct_project(detection, power_w=power_w, targets=targets)


SNR_TARGET = [{"target_id": "snr", "metric": "lock_in_snr", "observation_id": "D", "minimum": 100.0}]
FLOOR2 = 2.0 * Q * 1e-12 + 4.0 * K_B * 295.0 / 1e8  # (electrical(): 1 pA dark current, 100 MOhm at 295 K)


def test_a_chop_above_a_first_order_detector_bandwidth_states_its_attenuated_signal_and_no_snr():
    """1 MHz chopping on a detector of 1 kHz: the fundamental leaves the detector 1000 times weaker (|H| of a single pole), and
    the lock-in's noise and SNR are not stated -- noise added after the response is not modelled -- so a target on the SNR
    cannot pass."""

    current = responsivity(0.8, 600.0) * 1e-9
    signal_in = 2.0 / math.pi * current / math.sqrt(2.0)
    run = run_optical_project(_chopped({"bandwidth_Hz": 1000.0}, 1e6, targets=SNR_TARGET))
    quantities = detection_of(run)
    response = first_order_response(1e6, 1000.0)
    assert response == pytest.approx(1e-3, rel=1e-6)
    assert quantities["detector_response_at_modulation"].value == pytest.approx(response, rel=1e-12)
    assert quantities["lock_in_signal_A"].status == "MODELED"
    assert quantities["lock_in_signal_A"].value == pytest.approx(signal_in * response, rel=1e-12)
    for name in ("lock_in_noise_A", "lock_in_snr"):
        assert (quantities[name].status, quantities[name].reason) == ("UNAVAILABLE", MODULATION_ABOVE_DETECTOR_BANDWIDTH)
    validity = next(o for o in run.result.observations if o.observation_id == "D").detection.validity
    (band,) = [v for v in validity if v.parameter == "modulation_frequency_within_detector_bandwidth"]
    assert (band.value, band.limit, band.comparison, band.within) == (1e6, 1000.0, "AT_MOST", False)
    (target,) = run.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("DETECTION_QUANTITY_UNAVAILABLE",))
    assert MODULATION_ABOVE_DETECTOR_BANDWIDTH in target.message
    # at the -3 dB frequency itself the condition holds: signal and noise 1/sqrt(2) of the input's, the SNR the input's
    edge = detection_of(run_optical_project(_chopped({"bandwidth_Hz": 1000.0}, 1000.0)))
    noise_in = math.sqrt((2.0 * Q * 0.5 * current + FLOOR2) * lock_in_enbw_Hz(0.1, 2))
    assert edge["lock_in_signal_A"].value == pytest.approx(signal_in / math.sqrt(2.0), rel=1e-12)
    assert edge["lock_in_noise_A"].value == pytest.approx(noise_in / math.sqrt(2.0), rel=1e-12)
    assert edge["lock_in_snr"].value == pytest.approx(signal_in / noise_in, rel=1e-12)


def test_a_noise_equivalent_bandwidth_bounds_the_chop_at_its_single_pole_corner():
    """A noise-equivalent bandwidth B states no response shape: within f <= 2 B / pi (the -3 dB frequency of a single pole of
    noise bandwidth B; a Butterworth response of noise bandwidth B is down by less there) the lock-in reads the input-referred
    closed forms; above it nothing of the lock-in's signal, noise or SNR is stated."""

    bandwidth = 10000.0
    limit = 2.0 * bandwidth / math.pi
    # (the reference for that limit: a single pole of -3 dB at 2B/pi has noise bandwidth B; Butterworth responses of order 2..4
    # with noise bandwidth B pass more than 1/sqrt(2) at 2B/pi)
    numeric, _ = integrate.quad(lambda f: 1.0 / (1.0 + (f / limit) ** 2), 0.0, math.inf)
    assert numeric == pytest.approx(bandwidth, rel=1e-9)
    for order in (2, 3, 4):
        b, a = signal.butter(order, 1.0, analog=True)
        enbw_per_corner, _ = integrate.quad(lambda w: abs(signal.freqs(b, a, worN=[w])[1][0]) ** 2, 0.0, math.inf, limit=200)
        corner = bandwidth / enbw_per_corner  # (the corner frequency of that order's response of noise bandwidth B)
        assert abs(signal.freqs(b, a, worN=[limit / corner])[1][0]) > 1.0 / math.sqrt(2.0)
    current = responsivity(0.8, 600.0) * 1e-9
    changes = {"bandwidth_definition": "NOISE_EQUIVALENT", "bandwidth_Hz": bandwidth}
    inside = detection_of(run_optical_project(_chopped(changes, 6000.0)))
    signal_in = 2.0 / math.pi * current / math.sqrt(2.0)
    noise_in = math.sqrt((2.0 * Q * 0.5 * current + FLOOR2) * lock_in_enbw_Hz(0.1, 2))
    assert inside["lock_in_signal_A"].value == pytest.approx(signal_in, rel=1e-12)
    assert inside["lock_in_snr"].value == pytest.approx(signal_in / noise_in, rel=1e-12)
    assert "at the detector's input" in inside["lock_in_signal_A"].basis
    run = run_optical_project(_chopped(changes, 6400.0, targets=SNR_TARGET))
    above = detection_of(run)
    for name in ("lock_in_signal_A", "lock_in_noise_A", "lock_in_snr"):
        assert (above[name].status, above[name].reason) == ("UNAVAILABLE", MODULATION_ABOVE_DETECTOR_BANDWIDTH), name
    assert above["modulated_photocurrent_A"].value == pytest.approx(current, rel=1e-12)  # (the photocurrent itself stays)
    band = next(v for v in next(o for o in run.result.observations if o.observation_id == "D").detection.validity
                if v.parameter == "modulation_frequency_within_detector_bandwidth")
    assert band.limit == pytest.approx(limit, rel=1e-15) and band.within is False
    assert run.result.targets[0].status.value == "UNKNOWN"


def test_a_square_chop_above_the_detector_bandwidth_states_no_open_state_snr():
    """1 MHz square chopping (duty 0.5) of 1 nW on a 1 kHz first-order detector: the detector's output never reaches the open
    state -- it follows the mean D I with a ripple of |H| (2/pi) I / sqrt 2 rms -- so the open-state DC SNR (576.3 here) is not
    what the detector shows (the mean over its noise: 325.4). A target detector_snr >= 400 used to PASS on the open-state value;
    the shot / total noise and detector_snr are now not stated (MODULATION_ABOVE_DETECTOR_BANDWIDTH) and the target is UNKNOWN,
    while the photocurrent (the open state's) and the noise-equivalent power stand. Within the band (and for a SINE, whose traced
    power is its mean) nothing changes."""

    targets = [{"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 400.0}]
    current = responsivity(0.8, 600.0) * 1e-9
    noise_bandwidth = math.pi / 2.0 * 1000.0
    open_snr = current / math.sqrt((2.0 * Q * current + FLOOR2) * noise_bandwidth)
    mean_snr = 0.5 * current / math.sqrt((2.0 * Q * 0.5 * current + FLOOR2) * noise_bandwidth)
    assert open_snr == pytest.approx(576.3057618418966, rel=1e-9) and mean_snr < 400.0 < open_snr
    run = run_optical_project(_chopped({"bandwidth_Hz": 1000.0}, 1e6, targets=targets))
    quantities = detection_of(run)
    for name in ("shot_noise_current_A", "total_noise_current_A", "detector_snr"):
        assert (quantities[name].status, quantities[name].reason) == ("UNAVAILABLE", MODULATION_ABOVE_DETECTOR_BANDWIDTH), name
    assert quantities["photocurrent_A"].value == pytest.approx(current, rel=1e-12)
    floor = math.sqrt(FLOOR2)
    assert quantities["noise_equivalent_power_W_per_rtHz"].value == pytest.approx(floor / responsivity(0.8, 600.0), rel=1e-12)
    (target,) = run.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("DETECTION_QUANTITY_UNAVAILABLE",))
    # within the band, and a SINE above it: the DC SNR is stated as before
    within = detection_of(run_optical_project(_chopped({"bandwidth_Hz": 1000.0}, 500.0)))
    assert within["detector_snr"].value == pytest.approx(open_snr, rel=1e-12)
    sine = detection_of(run_optical_project(_chopped({"bandwidth_Hz": 1000.0}, 1e6, waveform="SINE")))
    assert sine["detector_snr"].value == pytest.approx(open_snr, rel=1e-12)


def test_a_quantity_that_reads_no_sampled_light_carries_no_sampling_sigma():
    """The sample's fluorescence (sampled rays) and a second laser straight onto the detector, only the second one chopped: the
    modulated photocurrent and the lock-in signal read none of the sampled light, so they carry no sampling sigma (it was 0.0,
    which reads as a sampled value of no spread); the photocurrent, which sums both, carries the fluorescence's."""

    detection = {"detector": {"quantum_efficiency": QE, "electrical": electrical()},
                 "lock_in": lock_in(modulation={"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": 0.5, "source_ids": ["L4"]})}
    document = chain_project(detection=detection)
    document["components"].append(source("L4", (0.0, 120.0, 0.0), quaternion=ALONG_Y, wavelength_nm=600.0, waist_mm=0.5, power_w=1e-9))
    quantities = detection_of(run_optical_project(document))
    for name in ("modulated_photocurrent_A", "lock_in_signal_A"):
        assert quantities[name].value > 0.0 and quantities[name].sampling_relative_sigma is None, name
    assert quantities["photocurrent_A"].sampling_relative_sigma > 0.0


# ------------------------------------------------------------------------------------------------- the linear range
def test_a_photocurrent_above_the_stated_saturation_current_is_not_modelled():
    """1 W onto the photodiode (0.387 A at QE 0.8, 600 nm) against a stated 10 mA linear range: the photocurrent and everything
    that reads it are not stated, the optical power is; without a saturation current the photocurrent's basis says none is
    checked."""

    targets = [{"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0}]
    detection = {"detector": {"quantum_efficiency": QE, "electrical": electrical(saturation_current_A=0.01)}}
    run = run_optical_project(_direct_project(detection, power_w=1.0, targets=targets))
    quantities = detection_of(run)
    current = responsivity(0.8, 600.0) * 1.0
    assert current == pytest.approx(0.3871461, rel=1e-6)
    assert quantities["detected_power_W"].value == pytest.approx(1.0, rel=1e-12)
    for name in ("photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr", "noise_equivalent_power_W_per_rtHz"):
        assert (quantities[name].status, quantities[name].reason) == ("UNAVAILABLE", PHOTOCURRENT_ABOVE_LINEAR_RANGE), name
    (linear,) = next(o for o in run.result.observations if o.observation_id == "D").detection.validity
    assert (linear.parameter, linear.limit, linear.within) == ("peak_photocurrent_within_linear_range", 0.01, False)
    assert linear.value == pytest.approx(current, rel=1e-12)
    assert run.result.targets[0].status.value == "UNKNOWN" and PHOTOCURRENT_ABOVE_LINEAR_RANGE in run.result.targets[0].message
    # 1 nW: within the range, the closed forms as without the statement
    dim = detection_of(run_optical_project(_direct_project(detection)))
    plain = detection_of(run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE, "electrical": electrical()}})))
    assert dim["detector_snr"].value == plain["detector_snr"].value
    assert "no saturation current is stated" in plain["photocurrent_A"].basis
    assert "no saturation current is stated" not in dim["photocurrent_A"].basis


def test_a_sine_modulation_checks_the_linear_range_at_its_crest():
    """A SINE of depth 0.6 around the traced (mean) current I peaks at I + 0.6 I: a limit between the two refuses it."""

    current = responsivity(0.8, 600.0) * 1e-9
    run = run_optical_project(_chopped({"saturation_current_A": 1.3 * current}, 1000.0, waveform="SINE"))
    quantities = detection_of(run)
    validity = {v.parameter: v for v in next(o for o in run.result.observations if o.observation_id == "D").detection.validity}
    assert validity["peak_photocurrent_within_linear_range"].value == pytest.approx(1.6 * current, rel=1e-12)
    assert validity["peak_photocurrent_within_linear_range"].within is False
    assert quantities["lock_in_snr"].reason == PHOTOCURRENT_ABOVE_LINEAR_RANGE
    assert detection_of(run_optical_project(_chopped({"saturation_current_A": 1.7 * current}, 1000.0, waveform="SINE")))[
        "lock_in_snr"].status == "MODELED"


def test_per_pixel_electrical_snr_is_null_above_the_linear_range():
    run = run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE}}))
    detector = {"quantum_efficiency": QE, "pixel_pitch_mm": [2.0, 2.0], "fill_factor": 1.0, "pixel_count": [3, 3],
                "electrical": electrical(bandwidth_definition="NOISE_EQUIVALENT", bandwidth_Hz=100.0)}
    plain = load_run(run).detector_response("D", detector, half_extent_mm=4.0, samples=65)["quantities"]
    current = np.array(plain["photocurrent_A"]["values"], dtype=float)
    limit = float(np.median(current))
    detector["electrical"] = {**detector["electrical"], "saturation_current_A": limit}
    limited = load_run(run).detector_response("D", detector, half_extent_mm=4.0, samples=65)["quantities"]
    assert limited["photocurrent_A"] == plain["photocurrent_A"]
    before_rows, after_rows = plain["electrical_snr"]["values"], limited["electrical_snr"]["values"]
    for current_row, before_row, after_row in zip(current.tolist(), before_rows, after_rows):
        for value, before, after in zip(current_row, before_row, after_row):
            assert after == (None if value > limit else before)
    assert any(value > limit for row in current.tolist() for value in row)


# ------------------------------------------------------------------------------------------------- thin sample, seeds, sampling
@pytest.mark.parametrize(("changes", "parameter", "observed", "limit"), [
    ({"absorbance": 0.1}, "absorbance", 0.1, 0.1), ({"absorbance": 6.0}, "absorbance", 6.0, 0.1),
    ({"absorbance": None, "absorbed_fraction": 0.5}, "absorbed_fraction", 0.5, 1.0 - 10.0 ** -0.1)])
def test_a_fluorescence_sample_beyond_the_thin_sample_limit_is_computed_and_its_record_states_the_limit(
        changes: dict, parameter: str, observed: float, limit: float):
    """Emitted power = P_ex x a x QY x sum(f_i lambda_ex / lambda_i) (the stated model, a closed form); inside the limit the
    record has no validity entry (its bytes as before), beyond it one check: the parameter, the limit, the observed value."""

    run = run_optical_project(chain_project(fluorescence(sampling={"sample_count": 64, "seed": 1}, **changes)))
    record = sample_record(run)
    absorbed = 1.0 - 10.0 ** -changes["absorbance"] if changes.get("absorbance") is not None else changes["absorbed_fraction"]
    assert record.absorbed_fraction == pytest.approx(absorbed, rel=1e-12)
    expected = record.excitation_power_W * absorbed * 0.9 * (0.6 * 488.0 / 520.0 + 0.4 * 488.0 / 540.0)
    assert record.emitted_power_W == pytest.approx(expected, rel=1e-12)
    if observed <= limit:
        assert record.validity == () and "validity" not in record.model_dump(mode="json")
        return
    (check,) = record.validity
    assert (check.parameter, check.maximum, check.observed, check.within, check.at) == (parameter, pytest.approx(limit), observed, False,
                                                                                        ("component S",))


def test_a_raman_sample_beyond_the_single_scattering_limit_is_computed_with_the_linear_form_and_states_the_limit():
    raman = {"model_kind": "RAMAN", "excitation": {"observation_id": "sample_plane", "wavelength_nm": 488.0},
             "raman_lines": [{"shift_cm1": 1000.0, "differential_cross_section_cm2_per_sr": 1e-29}], "number_density_per_cm3": 1e26,
             "path_length_mm": 10.0, "emitting_surface": {"shape": "DISC", "diameter_mm": 0.02}, "angular_law": "ISOTROPIC",
             "sampling": {"sample_count": 64, "seed": 1}}
    record = sample_record(run_optical_project(chain_project(raman)))
    fraction = 1e26 * 4.0 * math.pi * 1e-29 * 1.0  # N 4 pi (d sigma / d Omega) L, L = 1 cm: 0.0126, above 0.01
    emitted_nm = 1e7 / (1e7 / 488.0 - 1000.0)
    assert record.emitted_power_W == pytest.approx(record.excitation_power_W * fraction * 488.0 / emitted_nm, rel=1e-12)
    (check,) = record.validity
    assert (check.parameter, check.maximum, check.within) == ("scattered_fraction", 0.01, False)
    assert check.observed == pytest.approx(fraction, rel=1e-15)


def test_photon_fractions_that_do_not_add_to_1_are_normalized_to_their_shares():
    """Three lines written 0.333 each: three equal shares of 1/3 whose powers add to the total of the same sample written
    with shares adding to 1."""

    lines = [{"wavelength_nm": w, "photon_fraction": 0.333} for w in (520.0, 540.0, 560.0)]
    record = sample_record(run_optical_project(chain_project(fluorescence(sampling={"sample_count": 64, "seed": 1}, emission_lines=lines))))
    assert [line.photon_fraction for line in record.lines] == [1 / 3, 1 / 3, 1 / 3]
    absorbed = 1.0 - 10.0 ** -0.01
    expected = record.excitation_power_W * absorbed * 0.9 * sum(488.0 / w / 3.0 for w in (520.0, 540.0, 560.0))
    assert record.emitted_power_W == pytest.approx(expected, rel=1e-12)
    assert math.fsum(line.power_W for line in record.lines) == record.emitted_power_W


@pytest.mark.parametrize("seed", [0, 1, 2, 12345])
def test_every_seed_shifts_one_stratification(seed: int):
    """Each coordinate of the first N shifted points holds one point per 1/N cell, whatever the seed: an estimate that depends on
    one coordinate differs between seeds by at most 1/N (and may be the same), as the contract's seed statement says."""

    count = 1024
    points = sobol_points(count, seed)
    for d in range(points.shape[1]):
        assert sorted(np.floor(points[:, d] * count).astype(int).tolist()) == list(range(count))
    threshold = 0.3141
    assert abs(np.count_nonzero(points[:, 2] < threshold) - threshold * count) < 1.0


def test_light_that_reads_no_sampled_ray_carries_no_sampling_sigma():
    quantities = detection_of(run_optical_project(_direct_project({"detector": {"quantum_efficiency": QE, "electrical": electrical()}})))
    assert all(q.sampling_relative_sigma is None for q in quantities.values())


def test_the_excitation_trace_of_a_lowering_stops_at_its_monitor_and_is_no_lowering_error():
    """Lowering a Project with an emitting sample traces its excitation (``sample_emission.complete_request``). Under a run
    monitor that stops at its 1000th checkpoint -- the server's integrity check monitors its lowering this way -- a
    Project of 8192 aperture quadrature nodes stops with ``RunStopped`` (inside the trace: the Gauss-Legendre rule's
    recurrence and every path reach checkpoints), not with ``LocalRunError``, and long before the trace's own end."""
    from os_optics.local.lowering import lower_project
    from os_optics.run_control import RunStopped, monitored

    class Stop:
        steps = 0

        def enter(self, name, total, unit):
            pass

        def leave(self, name):
            pass

        def unwind(self, name):
            pass

        def step(self, count):
            self.steps += 1
            if self.steps >= 1000:
                raise RunStopped("CALLER", "stopped", phase="lower")

    document = chain_project()
    document["numerical_settings"] = {"aperture_quad_sector_nodes": 8192}
    monitor = Stop()
    with pytest.raises(RunStopped), monitored(monitor):
        lower_project(document)
    assert monitor.steps == 1000
