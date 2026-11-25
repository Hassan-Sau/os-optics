"""A continuous source spectrum (``spectral_density``): its quadrature lines, the power they carry, the coherence derived from the
spectrum and the quadrature's convergence (``optics/interference/spectrum.py``, ``ValidationResult.source_spectra``).

References are independent of the module: the bin powers from Python's ``math.erf`` / ``math.atan`` and a dense trapezoid
integral of the table; the degree of coherence ``|gamma(D)| = |integral S(lambda) exp(2 pi i D / lambda) dlambda| / integral S``
from NumPy's complex exponential on a dense trapezoid grid, its 1/e crossing by SciPy's ``brentq``; the narrow-band limits
``(2 sqrt(ln 2) / pi) lambda^2 / F`` (Gaussian) and the first 1/e crossing of ``sin(x) / x`` (top hat, ``x = pi D F / lambda^2``).
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import brentq

from os_optics.contracts import SourceSpec, SpectralDensity, SpectralLine
from os_optics.local import lower_project, run_optical_project
from os_optics.optics.interference import spectrum as S
from os_optics.scene.lower import coherence_length_from_linewidth_mm

from local_helpers import plane, project, source

POWER = 1e-3


def density(kind: str, band, line_count: int = 16, **shape) -> SpectralDensity:
    return SpectralDensity(shape={"kind": kind, **shape}, band_nm=band, line_count=line_count)


GAUSSIAN = density("gaussian", (770.0, 830.0), center_nm=800.0, fwhm_nm=20.0)
LORENTZIAN = density("lorentzian", (780.0, 820.0), center_nm=800.0, fwhm_nm=2.0)
TOP_HAT = density("top_hat", (500.0, 700.0), center_nm=600.0, fwhm_nm=200.0)
TABLE = density("tabulated", (600.0, 700.0), wavelength_nm=(600.0, 640.0, 660.0, 700.0), density=(0.2, 1.0, 1.0, 0.0))


def continuous_source(spectral_density: dict, **kw) -> dict:
    laser = source(**kw)
    del laser["source_model"]["wavelength_nm"]
    laser["source_model"]["spectral_density"] = spectral_density
    return laser


def free_space(*sources) -> dict:
    # (no environment: the release-parity script checks this document against the hash os-optics 0.12.0 gave it)
    return project(list(sources), [plane("OUT", (100.0, 0.0, 0.0))], environment=None)


def reference_gamma(spectrum: SpectralDensity, opd_mm: float, samples: int = 200001) -> float:
    low, high = spectrum.band_nm
    lam = np.linspace(low, high, samples)
    shape = spectrum.shape
    if shape.kind == "gaussian":
        s = np.exp(-4.0 * math.log(2.0) * ((lam - shape.center_nm) / shape.fwhm_nm) ** 2)
    elif shape.kind == "lorentzian":
        s = 1.0 / (1.0 + (2.0 * (lam - shape.center_nm) / shape.fwhm_nm) ** 2)
    elif shape.kind == "top_hat":
        s = np.ones_like(lam)
    else:
        s = np.interp(lam, shape.wavelength_nm, shape.density)
    field = np.trapezoid(s * np.exp(2j * np.pi * opd_mm * 1e6 * (1.0 / lam - 1.0 / 650.0)), lam)
    return abs(field) / np.trapezoid(s, lam)


# ---------------------------------------------------------------------------------------------------------- lines


def test_the_lines_are_the_bin_centres_with_the_exact_bin_powers() -> None:
    edges = np.linspace(770.0, 830.0, 17)
    a = 2.0 * math.sqrt(math.log(2.0)) / 20.0
    want = [math.erf(a * (e1 - 800.0)) - math.erf(a * (e0 - 800.0)) for e0, e1 in zip(edges, edges[1:])]
    lines = S.quadrature_lines(GAUSSIAN)
    assert [w for w, _ in lines] == pytest.approx([(e0 + e1) / 2 for e0, e1 in zip(edges, edges[1:])], abs=1e-12)
    assert np.array([p for _, p in lines]) / sum(p for _, p in lines) == pytest.approx(np.array(want) / sum(want), rel=1e-13)
    edges = np.linspace(780.0, 820.0, 17)
    want = [math.atan(e1 - 800.0) - math.atan(e0 - 800.0) for e0, e1 in zip(edges, edges[1:])]
    assert [p for _, p in S.quadrature_lines(LORENTZIAN)] == pytest.approx(want, rel=1e-13)
    assert {p for _, p in S.quadrature_lines(TOP_HAT)} == {12.5}
    fine = np.linspace(600.0, 700.0, 160001)
    table = np.interp(fine, (600.0, 640.0, 660.0, 700.0), (0.2, 1.0, 1.0, 0.0))
    want = [np.trapezoid(table[(fine >= e0) & (fine <= e1)], fine[(fine >= e0) & (fine <= e1)]) for e0, e1 in zip(np.linspace(600, 700, 17), np.linspace(600, 700, 17)[1:])]
    assert [p for _, p in S.quadrature_lines(TABLE)] == pytest.approx(want, rel=1e-6)


def test_a_project_source_lowers_to_its_quadrature_lines() -> None:
    document = free_space(continuous_source({"shape": {"kind": "GAUSSIAN", "center_nm": 800.0, "fwhm_nm": 20.0}, "line_count": 16}))
    (spec,) = [c.source for c in lower_project(document, catalog=None).request.project.components if c.source is not None]
    assert spec.spectral_density == GAUSSIAN  # the band of the default rule: centre +- 1.5 FWHM
    assert tuple((line.wavelength_nm, line.weight) for line in spec.spectrum) == S.quadrature_lines(GAUSSIAN)
    assert spec.linewidth_nm == 60.0 / 16 and spec.coherence_length_mm is None and spec.power_W == POWER


def test_an_engine_source_whose_lines_are_not_the_quadrature_is_refused() -> None:
    lines = tuple(SpectralLine(wavelength_nm=w, weight=p) for w, p in S.quadrature_lines(TOP_HAT))
    SourceSpec(source_id="L", spectrum=lines, power_W=1.0, waist_radius_x_mm=1.0, waist_radius_y_mm=1.0, spectral_density=TOP_HAT,
               linewidth_nm=12.5)
    with pytest.raises(ValidationError, match="spectrum must be the quadrature lines"):
        SourceSpec(source_id="L", spectrum=lines[:-1], power_W=1.0, waist_radius_x_mm=1.0, waist_radius_y_mm=1.0, spectral_density=TOP_HAT,
                   linewidth_nm=12.5)
    with pytest.raises(ValidationError, match="the bin width as linewidth_nm"):
        SourceSpec(source_id="L", spectrum=lines, power_W=1.0, waist_radius_x_mm=1.0, waist_radius_y_mm=1.0, spectral_density=TOP_HAT,
                   linewidth_nm=1.0)


def test_a_flat_band_delivers_its_power_in_equal_lines_each_as_wide_as_its_bin() -> None:
    run = run_optical_project(free_space(continuous_source({"shape": {"kind": "TOP_HAT", "center_nm": 600.0, "fwhm_nm": 200.0},
                                                            "line_count": 16})), catalog=None).result
    (obs,) = run.observations
    assert obs.total_power_W == pytest.approx(POWER, rel=1e-12)
    assert [p.wavelength_nm for p in obs.power_by_wavelength] == pytest.approx([500.0 + 12.5 * (k + 0.5) for k in range(16)], abs=1e-12)
    assert all(p.power_W == pytest.approx(POWER / 16, rel=1e-12) for p in obs.power_by_wavelength)
    for arrival in obs.arrivals:  # each line a Gaussian line of the bin's width: L_k = (2 sqrt(ln 2) / pi) lambda_k^2 / w
        assert arrival.coherence_length_mm == pytest.approx(coherence_length_from_linewidth_mm(arrival.beam.wavelength_nm, 12.5), rel=1e-15)
        assert arrival.coherence_length_mm == pytest.approx(2.0 * math.sqrt(math.log(2.0)) / math.pi * arrival.beam.wavelength_nm ** 2 / 12.5e6,
                                                            rel=1e-14)
    (report,) = run.source_spectra
    assert report.basis == "SAMPLED_CONTINUOUS" and report.quadrature == "MIDPOINT_BINS" and report.line_count == 16
    assert report.band_nm == (500.0, 700.0) and report.line_width_nm == 12.5 and report.shape == "TOP_HAT"


# ------------------------------------------------------------------------------------------------------- coherence


@pytest.mark.parametrize("spectrum", [GAUSSIAN, LORENTZIAN, TOP_HAT, TABLE], ids=lambda d: d.shape.kind)
def test_the_degree_of_coherence_of_the_spectrum_matches_an_independent_integral(spectrum: SpectralDensity) -> None:
    summary = S.spectrum_summary("L", spectrum)
    length = summary.coherence_length_mm
    grid = [length * f for f in (0.0, 0.3, 0.7, 1.0, 1.6, 2.5)]
    got = S.coherence(spectrum, grid)
    assert got == pytest.approx([reference_gamma(spectrum, d) for d in grid], abs=2e-8)
    assert got[0] == 1.0
    reference = brentq(lambda d: reference_gamma(spectrum, d) - math.exp(-1.0), 0.5 * length, 1.5 * length, xtol=1e-12)
    assert length == pytest.approx(reference, rel=1e-6)


def test_narrow_spectra_reach_their_analytic_coherence_lengths() -> None:
    """A Gaussian of FWHM F (here 20 nm at 800 nm, its band +- 1.5 F): (2 sqrt(ln 2) / pi) lambda^2 / F to the relative order
    (F / lambda)^2 of the wavelength-to-wavenumber map; a narrow top hat of width F: the first 1/e of sin(x) / x,
    x = pi D F / lambda^2."""

    gaussian = S.spectrum_summary("L", GAUSSIAN).coherence_length_mm
    assert gaussian == pytest.approx(2.0 * math.sqrt(math.log(2.0)) / math.pi * 800.0**2 / 20.0 * 1e-6, rel=2e-3)
    narrow = density("top_hat", (799.0, 801.0), center_nm=800.0, fwhm_nm=2.0)
    x0 = brentq(lambda x: math.sin(x) / x - math.exp(-1.0), 1.0, 3.0)
    assert S.spectrum_summary("L", narrow).coherence_length_mm == pytest.approx(x0 * 800.0**2 / (math.pi * 2.0) * 1e-6, rel=1e-5)


@pytest.mark.parametrize("spectrum", [GAUSSIAN, TOP_HAT], ids=lambda d: d.shape.kind)
def test_doubling_the_lines_converges_on_the_spectrum(spectrum: SpectralDensity) -> None:
    """The lines' |gamma| approaches the spectrum's as the bins narrow (second order in the bin width: a quarter per doubling)."""

    summary = S.spectrum_summary("L", spectrum)
    assert summary.gamma_max_difference_doubled < 0.3 * summary.gamma_max_difference
    exact = summary.coherence_length_mm
    assert abs(summary.coherence_length_doubled_lines_mm - exact) < abs(summary.coherence_length_lines_mm - exact)
    sixty_four = S.spectrum_summary("L", spectrum.model_copy(update={"line_count": 64}))
    assert sixty_four.gamma_max_difference < 2e-3
    assert sixty_four.coherence_length_lines_mm == pytest.approx(exact, rel=2e-3)
    assert summary.convergence_range_mm == pytest.approx(3.0 * exact, abs=2e-9)


def test_a_spectrum_that_stays_coherent_has_no_coherence_length() -> None:
    """Two narrow bumps 10 nm apart, 3 : 1: |gamma| beats between 1 and |3 - 1| / 4 = 0.5 and never falls to 1/e over the scan
    (each bump alone is coherent over tens of mm): no length, and the reason."""

    rows = (599.98, 600.0, 600.02, 609.98, 610.0, 610.02)
    beats = density("tabulated", (599.98, 610.02), line_count=64, wavelength_nm=rows, density=(0.0, 3.0, 0.0, 0.0, 1.0, 0.0))
    summary = S.spectrum_summary("L", beats)
    assert summary.coherence_length_mm is None and summary.reason.startswith("COHERENCE_BEYOND_SCAN")
    assert min(S.coherence(beats, [summary.convergence_range_mm * k / 64 for k in range(65)])) > 0.49


# --------------------------------------------------------------------------------------------- discrete stays discrete


def spectral_lines_project() -> dict:
    """The recorded 32-line case: 500-700 nm, 1 mW / 32 per line."""

    laser = source()
    del laser["source_model"]["wavelength_nm"], laser["source_model"]["power_w"]
    laser["source_model"]["spectral_lines"] = [{"wavelength_nm": 500.0 + 200.0 * k / 31, "power_w": POWER / 32} for k in range(32)]
    return laser


def test_the_32_line_discrete_source_stays_discrete() -> None:
    document = free_space(spectral_lines_project())
    request = lower_project(document, catalog=None).request
    assert "spectral_density" not in json.dumps(request.model_dump(mode="json"))
    run = run_optical_project(document, catalog=None).result
    (obs,) = run.observations
    assert run.status.value == "PASS" and len(obs.power_by_wavelength) == 32
    assert obs.total_power_W == pytest.approx(POWER, rel=1e-12)
    assert run.source_spectra == () and "source_spectra" not in run.model_dump(mode="json")


def test_beside_a_continuous_source_the_discrete_one_is_labelled_discrete() -> None:
    white = continuous_source({"shape": {"kind": "GAUSSIAN", "center_nm": 800.0, "fwhm_nm": 20.0}, "line_count": 8}, instance_id="WHITE",
                              position=(-200.0, 5.0, 0.0))
    run = run_optical_project(free_space(spectral_lines_project(), white), catalog=None).result
    labels = {s.source_id: (s.basis, s.line_count) for s in run.source_spectra}
    assert labels == {"SRC": ("DISCRETE", 32), "WHITE": ("SAMPLED_CONTINUOUS", 8)}
    discrete = next(s for s in run.source_spectra if s.source_id == "SRC").model_dump(mode="json")
    assert set(discrete) == {"source_id", "basis", "line_count"}


def test_the_report_is_the_same_on_a_second_run() -> None:
    document = free_space(continuous_source({"shape": {"kind": "LORENTZIAN", "center_nm": 1550.0, "fwhm_nm": 1.0}, "line_count": 32}))
    first = run_optical_project(copy.deepcopy(document), catalog=None).result.source_spectra
    assert first == run_optical_project(document, catalog=None).result.source_spectra
    assert first[0].band_nm == (1540.0, 1560.0)
