"""Model scope declaration: what the Engine models and what it explicitly reports as out of model scope."""

from __future__ import annotations

MODEL_SCOPE_SUPPORTED: tuple[str, ...] = (
    "geometric_optics_3d: plane, sphere and sag surfaces (conic, even asphere, cylinder, off-axis sections), reflection, "
    "Snell refraction, TIR, ideal thin lens, apertures, grating order directions (vector grating equation)",
    "gaussian_beam: paraxial x/y-independent q-parameter (ABCD, Coddington tilted surfaces at the local principal "
    "curvatures), untwisted astigmatism",
    "polarization: Jones calculus with 3D transverse basis transport, Fresnel s/p, stated Faraday rotation; a partially "
    "polarized or unpolarized source traced as two mutually incoherent orthogonal Jones components whose Stokes parameters add",
    "birefringence: uniaxial crystals with plane faces (ordinary and extraordinary beams, walk-off of the extraordinary ray, each "
    "beam's index along its ray)",
    "second_harmonic_generation: a uniaxial crystal slab with a stated chi(2) coefficient or d tensor, birefringent (type I / II) or "
    "quasi-phase matching, angle and temperature tuning; undepleted-pump conversion of a TEM00 Gaussian beam (near field, or "
    "Boyd-Kleinman focusing) with walk-off; the harmonic traced as a new spectral line, the fundamental's power reduced by it",
    "interference: analytic two-beam (plane-wave phase, Gaussian envelope) on a plane; no wave propagation",
    "spectrum: discrete mutually incoherent spectral lines; a continuous spectral density traced as midpoint-quadrature lines",
    "scatter: single diffuse surface scatter (Lambertian, cosine-power) traced as incoherent unpolarized rays from the beam's "
    "chief-ray hit point; bulk scattering attenuates the beams and is booked, not followed",
    "complex_transmission: thin sample of stated amplitude and phase in an aperture stop; rays take its optical path and phase "
    "gradient (geometric), the beam its Gaussian-weighted transmittance",
    "materials: constant / Sellmeier / tabulated index with validity bands; constant and tabulated coatings",
    "sample_emission: fluorescence (absorbed share, quantum yield, emission lines) and spontaneous Stokes Raman scattering "
    "(shifts, cross-sections, number density, path length) of a thin sample, derived from the excitation the Project's other "
    "sources deliver to it and traced as an incoherent emitter (photon numbers, each photon at its own wavelength)",
    "detection: photodetector photocurrent from the quantum efficiency, white shot / dark / thermal noise in a stated bandwidth "
    "and its SNR; square-wave or sinusoidal modulation read by a lock-in (fundamental, RC output filter of 1 to 4 sections)",
    "layout: 2.5D rotated footprints, keepout, beam envelope, plate boundary, moving envelope",
    "control: finite-difference Jacobian, Levenberg-Marquardt, capture scan, deterministic Monte Carlo",
)

#: Target metrics of a linear cavity, evaluated by the cavity analysis ('os-optics project cavity' / 'os-optics results cavity',
#: json_api cavity_analysis / run_results kind 'cavity'), not by a run: metric -> the analysis's quantity id.
CAVITY_METRICS: dict[str, str] = {
    "cavity_circulating_power_W": "circulating_power_W",
    "cavity_transmitted_power_W": "transmitted_power_W",
    "cavity_reflected_power_W": "reflected_power_W",
    "cavity_transmission": "transmission",
    "cavity_reflection": "reflection",
    "cavity_power_build_up": "power_build_up",
    "cavity_resonant_power_build_up": "resonant_power_build_up",
    "cavity_free_spectral_range_Hz": "free_spectral_range_Hz",
    "cavity_transverse_mode_spacing_Hz": "transverse_mode_spacing_Hz",
    "cavity_linewidth_Hz": "linewidth_fwhm_Hz",
    "cavity_finesse": "finesse",
    "cavity_mode_matching": "mode_matching_efficiency",
}
#: The unit of each cavity metric (``'1'``: a ratio); a Project target with another ``unit`` is not assessed (TARGET_UNIT_MISMATCH).
CAVITY_METRIC_UNITS: dict[str, str] = {
    "cavity_circulating_power_W": "W",
    "cavity_transmitted_power_W": "W",
    "cavity_reflected_power_W": "W",
    "cavity_transmission": "1",
    "cavity_reflection": "1",
    "cavity_power_build_up": "1",
    "cavity_resonant_power_build_up": "1",
    "cavity_free_spectral_range_Hz": "Hz",
    "cavity_transverse_mode_spacing_Hz": "Hz",
    "cavity_linewidth_Hz": "Hz",
    "cavity_finesse": "1",
    "cavity_mode_matching": "1",
}
_CAVITY_ROUTE = ("a cavity's resonances and resonant build-up are not computed by a run, whose trace follows single passes; the cavity "
                 "analysis evaluates this target ('os-optics project cavity PROJECT --request R' or 'os-optics results cavity RUN --request R', "
                 "json_api cavity_analysis / run_results kind 'cavity'), the quantity {quantity}")

#: Target metrics an analysis evaluates, not a run (metric -> explanation with the route): a run reports them OUT_OF_MODEL_SCOPE like
#: ``OUT_OF_SCOPE_METRICS``. They are kept apart from that table, which every run's manifest lists (``model_scope``).
ANALYSIS_METRIC_ROUTES: dict[str, str] = {metric: _CAVITY_ROUTE.format(quantity=quantity) for metric, quantity in CAVITY_METRICS.items()}

# metric name -> explanation. Any Target with one of these metrics returns OUT_OF_MODEL_SCOPE, never PASS.
OUT_OF_SCOPE_METRICS: dict[str, str] = {
    "fresnel_propagation": "Fresnel (near-field diffraction) propagation is not a target of a run; 'os-optics results field' propagates a "
                           "sampled-field or speckle source to a plane (angular spectrum, or the paraxial transfer of the source axis)",
    "fraunhofer_propagation": "Fraunhofer (far-field diffraction) propagation is not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil",
    "angular_spectrum_propagation": "angular-spectrum propagation is not a target of a run; 'os-optics results field' propagates a "
                                    "sampled-field or speckle source to a parallel plane through free space by its angular spectrum",
    "complex_amplitude_field": "full-field complex-amplitude grids are not a target of a run; 'os-optics results field' gives the field of a "
                               "sampled-field or speckle source at a plane, 'os-optics results pupil-field' the scalar low-NA focal field of a pupil",
    "field_amplitude_map": "full-field complex-amplitude grids are not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil",
    "fdtd": "FDTD is not modelled by this Engine",
    "rcwa": "RCWA is not modelled by this Engine",
    "high_na_vector_diffraction": "a high-NA vector focal field is not a target of a run; 'os-optics results focal-field' computes the vector focal field of an ideal aplanatic objective",
    "doe_response": "arbitrary diffractive optical element response is not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a thin pupil mask",
    "metasurface_response": "metasurface full-wave response is not modelled by this Engine",
    "speckle_contrast": "speckle contrast is not a target of a run; 'os-optics results field' gives the speckle contrast of a SPECKLE "
                        "source at a plane, of one pattern and of the average of its realizations",
    "phase_retrieval": "phase retrieval is not modelled by this Engine",
    "stray_light": "general stray-light Monte Carlo (scattering, every non-sequential path) is not modelled by this Engine; 'os-optics "
                   "results ghosts' gives a run's interface-reflection ghost images with a bound on the ghosts it does not list",
    "rough_surface_scattering": "rough-surface scattering is not a target metric of a run; a surface's diffuse scatter (scatter on "
                                "a refracting, reflecting or absorbing surface, LAMBERTIAN or COSINE_POWER) is traced as rays and adds "
                                "to each plane's power and sampled_light",
    "pinhole_far_field": "exact far-field propagation behind a pinhole is not modelled by this Engine",
    "far_field_pattern": "diffraction far-field patterns are not a target of a run; 'os-optics results pupil-field' computes the scalar low-NA focal field of a pupil",
    "psf": "point spread functions are not a target of a run; 'os-optics results imaging' computes them for a run's image plane, 'os-optics results pupil-field' the scalar low-NA focal field of a pupil",
    "mtf": "MTF is not a target of a run; 'os-optics results imaging' computes the diffraction and geometric MTF of a run's image plane",
    "partial_polarization": "depolarization (an element whose Mueller matrix is not a Jones matrix) is not modelled by this "
                            "Engine; a partially polarized source is (source polarization PARTIAL or UNPOLARIZED, metric "
                            "degree_of_polarization)",
    "birefringent_double_refraction": "double refraction is traced in uniaxial crystals with plane faces (material kind "
                                      "uniaxial): each beam leaving the crystal is an arrival of its own, read with the usual metrics; "
                                      "this name is not a target metric. Biaxial crystals, optical activity and curved crystal faces "
                                      "are outside the model",
    # the weak-signal measurement chain (MODEL_SCOPE_BOUNDARY); detector_bandwidth_Hz, detector_snr and lock_in_snr are
    # computed on a plane that states detection (validation/detection.py), the second-harmonic metrics by
    # validation/nonlinear.py
    "spectral_resolution_nm": "a spectrometer's instrument function (slit, dispersion onto the detector, pixel sampling) is not "
                              "modelled; 'os-optics results grating-orders' gives the direction of every diffraction order of a "
                              "grating, and a sample's emission lines are discrete lines of stated wavelengths",
}


def out_of_scope_metric(metric: str) -> str | None:
    """The ``OUT_OF_SCOPE_METRICS`` or ``ANALYSIS_METRIC_ROUTES`` key a target metric names, compared without regard to case
    (``SPECTRAL_RESOLUTION_NM`` is ``spectral_resolution_nm``); None for any other metric."""

    if metric in OUT_OF_SCOPE_METRICS or metric in ANALYSIS_METRIC_ROUTES:
        return metric
    folded = metric.casefold()
    return next((key for key in (*OUT_OF_SCOPE_METRICS, *ANALYSIS_METRIC_ROUTES) if key.casefold() == folded), None)


def out_of_scope_explanation(key: str) -> str:
    """Why a run does not evaluate the metric ``key`` (an ``out_of_scope_metric`` key), with its route."""

    return OUT_OF_SCOPE_METRICS.get(key) or ANALYSIS_METRIC_ROUTES[key]


#: An item's disposition: in model scope and supported, out of model scope with an external route, or not modelled with a
#: stated substitute the run can trace. A supported item's ``route`` says how it is declared and read, and what it leaves out.
IN_SCOPE_SUPPORTED = "IN_SCOPE_SUPPORTED"
OUT_OF_SCOPE_EXTERNAL = "OUT_OF_SCOPE_EXTERNAL"
NOT_MODELLED = "NOT_MODELLED"
#: The boundary of the weak-signal measurement chain and of nonlinear conversion, item by item: ``disposition`` (one of the
#: three above), ``missing_model`` (what a calculation of it needs that the Engine does not have; empty for a supported
#: item) and ``route`` (what serves the request). Published as ``capabilities()["model_scope_boundary"]``.
MODEL_SCOPE_BOUNDARY: tuple[dict[str, str], ...] = (
    {"item": "extended_emitter", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "source_model EXTENDED_EMITTER (ISOTROPIC or LAMBERTIAN emission, seeded sampled rays); 'os-optics results "
              "sampled-light' gives each line's collection fraction at the observation planes"},
    {"item": "bulk_absorption", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "a declared model material's absorption_coefficient_per_mm (Beer-Lambert along each traced path, with the "
              "Fresnel losses at its faces)"},
    {"item": "camera_response", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "'os-optics results detector' (json_api detector_response): quantum efficiency, exposure, pixels, fill factor, "
              "full well, read noise, dark current, gain and bit depth give photons, electrons, ADU, saturation, noise and SNR "
              "per pixel"},
    {"item": "acousto_optic_static_deflection", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "native class deflector.acousto_optic: RF tones as the orders of one static grating, with each tone's direction "
              "and optical frequency shift; the order powers come from a stated efficiency table"},
    {"item": "fluorescence_emission", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "source_model FLUORESCENCE: the excitation power its observation plane receives at the excitation wavelength, the "
              "absorbed share (absorbance or absorbed_fraction), the quantum yield and the emission lines' photon shares give each "
              "line's power (P_abs QY f lambda_ex / lambda), traced as an ISOTROPIC or LAMBERTIAN incoherent emitter; 'os-optics "
              "results sample-emission' states the derivation. Linear excitation: no saturation, bleaching or reabsorption"},
    {"item": "raman_emission", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "source_model RAMAN: Stokes shifts with differential cross-sections, number density and path length give each "
              "band's wavelength and power (P_ex N 4 pi (d sigma / d Omega) L lambda_ex / lambda), traced as an ISOTROPIC "
              "incoherent emitter; 'os-optics results sample-emission'. Single scattering; no depolarization ratio"},
    {"item": "chopper_modulation", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "an observation plane's detection.lock_in.modulation: SQUARE chopping (duty cycle) or SINE modulation (depth) of "
              "the light of named sources (a sample excited by them follows); 'os-optics results detection' gives the "
              "fundamental it gives the detector current"},
    {"item": "lock_in_detection", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "an observation plane's detection.lock_in: time constant and filter order give the noise-equivalent bandwidth, "
              "the demodulated signal, its noise and lock_in_snr ('os-optics results detection'; target metric lock_in_snr)"},
    {"item": "detector_bandwidth", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "a detector's electrical response (bandwidth, noise-equivalent power or dark current and load resistance) gives "
              "the photocurrent, shot, dark and thermal noise and detector_snr of a plane ('os-optics results detection'; target "
              "metrics detector_bandwidth_Hz, detector_snr) and per pixel ('os-optics results detector'); lag, crosstalk, "
              "nonlinearity, 1/f and ADC quantization noise are not modelled"},
    {"item": "feedback_and_modulator_timing", "disposition": OUT_OF_SCOPE_EXTERNAL,
     "missing_model": "time-dependent control loops, acousto-optic switching times, RF phases and crosstalk",
     "route": "the static acousto-optic deflector (deflector.acousto_optic) gives the directions and frequency shifts; timing is "
              "an external analysis"},
    {"item": "spectrometer_instrument_function", "disposition": OUT_OF_SCOPE_EXTERNAL,
     "missing_model": "the instrument function: slit image, dispersion onto the detector, pixel sampling",
     "route": "'os-optics results grating-orders' gives every diffraction order's direction; 'os-optics results detector' the "
              "detector response to a stated irradiance"},
    {"item": "second_harmonic_generation", "disposition": IN_SCOPE_SUPPORTED, "missing_model": "",
     "route": "declared optical model NONLINEAR_CRYSTAL (a uniaxial crystal slab with its chi(2) coefficient, phase matching and "
              "temperature): the run generates the second harmonic of a TEM00 beam with an undepleted pump and traces it as a new "
              "spectral line; 'os-optics results nonlinear' gives each pass's efficiency, phase mismatch and acceptance bandwidths, "
              "targets shg_conversion_efficiency / shg_power_W; pump depletion, sum-frequency mixing of several lines and other "
              "nonlinear processes are out of model scope ('os-optics interop show nonlinear-ultrafast-communications')"},
)
#: Model kinds the Project format does not have, refused before the Project is read, with the boundary item each names: a
#: chopper and a lock-in are declared in an observation plane's detection, not as optical elements.
OUT_OF_SCOPE_SOURCE_KINDS: dict[str, str] = {}
OUT_OF_SCOPE_OPTICAL_MODEL_KINDS: dict[str, str] = {
    "CHOPPER": "chopper_modulation", "LOCK_IN": "lock_in_detection",
}
BOUNDARY_BY_ITEM: dict[str, dict[str, str]] = {entry["item"]: entry for entry in MODEL_SCOPE_BOUNDARY}
