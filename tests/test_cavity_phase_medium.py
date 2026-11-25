"""Mechanical cavity-length offsets change the phase path in the surrounding medium, not in vacuum by default.

Independent references: the Edlen phase index at the Project's temperature and pressure, a central-difference
phase-index slope for the group FSR, and the two-mirror Fabry-Perot Airy transmission.
"""

import math

import pytest

from os_optics.local.cavity import cavity_analysis
from test_cavity_injection import cavity


WAVELENGTH_NM = 1064.0
LENGTH_MM = 300.0
OFFSET_NM = 5320.0  # ten vacuum resonance spacings of 532 nm, but off resonance in air
LIGHT_MM_PER_S = 299792458000.0
REFLECTANCE = 0.99
TRANSMITTANCE = 0.01


def _air_index(wavelength_nm: float, temperature_c: float, pressure_pa: float) -> float:
    """Independent Edlen phase index, with wavelength in micrometres (vacuum)."""

    lambda_um = wavelength_nm / 1000.0
    sigma_sq = 1.0 / (lambda_um * lambda_um)
    refractivity = (6432.8 + 2949810.0 / (146.0 - sigma_sq) + 25540.0 / (41.0 - sigma_sq)) * 1e-8
    return 1.0 + refractivity * (pressure_pa / 101325.0) / (1.0 + 0.0034785 * (temperature_c - 15.0))


def _known(analysis, name: str) -> float:
    quantity = analysis.quantity(name)
    assert quantity.state.value == "KNOWN", (name, quantity.reason, quantity.message)
    return quantity.value


@pytest.mark.parametrize("pressure_pa", [101325, 0], ids=["air-20C-1atm", "vacuum-20C"])
def test_mechanical_offset_uses_medium_phase_index_and_group_fsr(pressure_pa):
    doc = cavity("500", "500", LENGTH_MM)
    doc["environment"] = {"temperature_c": "20", "pressure_pa": str(pressure_pa)}
    request = {"components": ["M1", "M2"], "wavelength_nm": WAVELENGTH_NM,
               "reflectances": {"M1": REFLECTANCE, "M2": REFLECTANCE},
               "transmittances": {"M1": TRANSMITTANCE, "M2": TRANSMITTANCE},
               "injection": {"power_w": 1.0, "beam": "EIGENMODE", "length_offset_nm": OFFSET_NM}}
    analysis = cavity_analysis(doc, request)
    phase_index = _air_index(WAVELENGTH_NM, 20.0, float(pressure_pa))
    waves = 2.0 * phase_index * OFFSET_NM / WAVELENGTH_NM
    delta = 2.0 * math.pi * (waves - round(waves))
    # Differentiate the phase index to obtain n_group = n_phase - lambda dn_phase/dlambda.
    step_nm = 0.01
    slope = (_air_index(WAVELENGTH_NM + step_nm, 20.0, float(pressure_pa)) -
             _air_index(WAVELENGTH_NM - step_nm, 20.0, float(pressure_pa))) / (2.0 * step_nm)
    group_index = phase_index - WAVELENGTH_NM * slope
    expected_fsr = LIGHT_MM_PER_S / (2.0 * LENGTH_MM * group_index)
    expected_transmission = TRANSMITTANCE ** 2 / ((1.0 - REFLECTANCE) ** 2 +
                                                   4.0 * REFLECTANCE * math.sin(delta / 2.0) ** 2)

    assert analysis.arrays["injection"]["detuning_from"] == "LENGTH_OFFSET"
    assert analysis.arrays["injection"]["round_trip_phase_rad"] == pytest.approx(delta, abs=1e-12)
    assert _known(analysis, "free_spectral_range_Hz") == pytest.approx(expected_fsr, rel=2e-9)
    assert _known(analysis, "detuning_Hz") == pytest.approx(delta * expected_fsr / (2.0 * math.pi), rel=2e-9, abs=1e-5)
    assert _known(analysis, "transmission") == pytest.approx(expected_transmission, rel=1e-11)
    assert _known(analysis, "transmitted_power_W") == pytest.approx(expected_transmission, rel=1e-11)
    if pressure_pa:
        assert expected_transmission == pytest.approx(0.2608, abs=0.0002)
        assert _known(analysis, "transmission") < 0.27
    else:
        assert expected_transmission == pytest.approx(1.0, abs=1e-12)
        assert _known(analysis, "transmission") == pytest.approx(1.0, rel=1e-12)
