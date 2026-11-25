"""Device models that act on a field or an irradiance map rather than on traced rays.

A native class (``registry.py``) builds a Project optical model. A field model is a calculation you run on the light a Project delivers
(or on a declared pupil field): it has no optical model and no component of its own. ``native_field_models()`` lists them with what
each takes, computes and leaves out, so ``native_components`` discovers them beside the classes.
"""

from __future__ import annotations

from typing import Any

from ..results.pupil_field import SCALAR_SINE_LIMIT
from .registry import AVAILABILITY, REF_GOODMAN, REF_SALEH_TEICH

REF_DI_LEONARDO = "R. Di Leonardo, F. Ianni and G. Ruocco, Computer generation of optimal holograms for optical trap arrays, Opt. Express 15, 1913-1922 (2007)"
REF_EMVA_1288 = "EMVA Standard 1288, Standard for Characterization of Image Sensors and Cameras"

FIELD_MODELS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "slm.pupil_mask", "category": "MODULATOR", "title": "Spatial light modulator or phase plate (pupil mask)",
        "description": "A sampled or parameterized phase and amplitude mask in a pupil, with its pixel pitch and fill, and the scalar focal "
                       "field it gives behind an ideal lens: steered spots, gratings, spot arrays.",
        "json_api": "pupil_field", "cli": "os-optics results pupil-field --request REQUEST.json", "schema_kind": "pupil-field-request",
        "takes": ["wavelength_nm", "focal_length_mm", "sampling (pixel_count, pitch_mm, fill_factor)", "aperture", "illumination", "mask",
                  "device_efficiency with its source"],
        "computes": ["power at every stage", "focal irradiance", "focal field of a coherent result", "spot powers, efficiency and uniformity"],
        "physics": "Fourier transform by an ideal lens with the pupil in its front focal plane; a pixel holds one value over its active area, "
                   "which adds the pixel envelope and the higher orders of the pixel grid.",
        "approximations": [f"scalar and paraxial up to a marginal-ray sine of {SCALAR_SINE_LIMIT}; above it the focal quantities are unavailable",
                           "thin-element mask; polarization, aberrations, fringing fields, phase calibration and the response in time are not modeled"],
        "references": [REF_GOODMAN, REF_DI_LEONARDO],
        "keywords": ["SLM", "spatial light modulator", "phase mask", "phase plate", "hologram", "Gerchberg-Saxton", "spot array", "tweezer array", "pupil"],
        "availability": list(AVAILABILITY),
    },
    {
        "model_id": "detector.camera_response", "category": "DETECTOR", "title": "Camera or photodetector response",
        "description": "Photons, electrons, ADU, saturation, noise and SNR per pixel from an irradiance map, for the quantum efficiency, exposure, "
                       "pixels, full well, noise, gain and bit depth a detector states.",
        "json_api": "detector_response", "cli": "os-optics results detector --detector DETECTOR.json", "schema_kind": "detector-model",
        "takes": ["quantum_efficiency rows with their source", "exposure_s", "pixel_pitch_mm", "fill_factor", "pixel_count", "full_well_e", "read_noise_e",
                  "dark_current_e_per_s", "gain_e_per_adu", "bit_depth"],
        "computes": ["photons", "signal electrons", "ADU", "saturation masks", "shot, dark and read noise variance", "SNR"],
        "physics": "Linear response over one exposure: the irradiance integrated over each pixel's light-sensitive area, Poisson shot and dark noise, "
                   "stated read noise.",
        "approximations": ["a parameter that is not stated leaves what needs it unavailable; nothing is assumed in its place",
                           "pixel crosstalk, lag, nonlinearity and ADC quantization noise are not modeled"],
        "references": [REF_EMVA_1288, REF_SALEH_TEICH],
        "keywords": ["camera", "detector", "photodiode", "sCMOS", "CCD", "quantum efficiency", "photon count", "SNR", "saturation", "full well"],
        "availability": list(AVAILABILITY),
    },
)


def native_field_models() -> list[dict[str, Any]]:
    """Every field model's description, sorted by id."""

    return sorted((dict(model) for model in FIELD_MODELS), key=lambda model: model["model_id"])
