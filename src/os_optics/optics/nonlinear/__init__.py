"""Second-order nonlinear conversion in uniaxial crystals (``shg.py``)."""

from .shg import (
    FWHM_DK_L,
    SINC2_HALF_POWER_ARGUMENT,
    Overlap,
    collinear_mismatch,
    coupling_per_W,
    d_eff_from_tensor,
    depletion_estimate,
    grating_wavenumber_per_mm,
    harmonic_power_W,
    overlap_factor,
    wave_index,
)

__all__ = [
    "FWHM_DK_L",
    "SINC2_HALF_POWER_ARGUMENT",
    "Overlap",
    "collinear_mismatch",
    "coupling_per_W",
    "d_eff_from_tensor",
    "depletion_estimate",
    "grating_wavenumber_per_mm",
    "harmonic_power_W",
    "overlap_factor",
    "wave_index",
]
