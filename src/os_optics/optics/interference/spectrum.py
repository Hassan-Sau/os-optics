"""Continuous source spectra: a stated spectral density over a band, traced as quadrature lines, and the source's degree of
coherence derived from that spectrum.

A Gaussian-beam source may state its spectrum as a density ``S(lambda)`` over the vacuum wavelength (``SpectralDensity``:
a Gaussian ``exp(-4 ln 2 (lambda - c)^2 / F^2)``, a Lorentzian ``1 / (1 + (2 (lambda - c) / F)^2)``, a top hat of full width
``F``, or a table linear between its rows) over a band ``[a, b]``; the source's power is the power within the band.

Quadrature (``MIDPOINT_BINS``, ``quadrature_lines``): the band is divided into ``n`` equal bins of width
``w = (b - a) / n``; line ``k`` lies at the bin's centre ``a + (k + 1/2) w`` with the weight of the exact integral of
``S`` over the bin (Gaussian: the error function at the edges, ``math.exact_erf.erf_interval``; Lorentzian: the arctangent,
``math.exact_trig.atan2``; top hat: equal; table: the integral of its linear pieces), and each line is a Gaussian line of
full width ``w`` (``SourceSpec.linewidth_nm``): the Engine's own coherence model of a line, ``|gamma_k(D)| =
exp(-(D / L_k)^2)`` with ``L_k = (2 sqrt(ln 2) / pi) lambda_k^2 / w``. Lines of one source are mutually incoherent: a
power the run sums over them (what a plane receives, a power budget) is the midpoint quadrature ``sum_k P_k T(lambda_k)`` of
``integral S T dlambda``, whose error is at most ``w^2 / 24 max|(S T)''| (b - a)`` for a smooth response ``T``.

Degree of coherence of the source with itself at a path difference ``D`` (Wiener-Khinchin; ``S`` per unit wavelength, so
``S dlambda`` is the power in ``dlambda`` whatever the variable):

    gamma(D) = integral S(lambda) exp(2 pi i D / lambda) dlambda / integral S(lambda) dlambda

over the band (``coherence``): composite Simpson on ``max(SIMPSON_MIN_INTERVALS, SIMPSON_INTERVALS_PER_CYCLE x cycles)``
intervals, ``cycles`` the phase range ``D (1/a - 1/b)`` across the band at the largest ``D`` evaluated (a table: on each piece
between its rows, so that its kinks are nodes), normalized by the same rule's integral of ``S`` (``gamma(0) = 1`` exactly). The lines' own ``gamma_lines(D) = sum_k P_k exp(-(D / L_k)^2)
exp(2 pi i D / lambda_k) / sum_k P_k`` (``line_coherence``) is the coherence the traced lines stand for.

Reported per source (``source_spectra``): the coherence length -- the smallest ``D`` with ``|gamma(D)| = 1/e`` (the definition
of a line's ``L_c``), found on a scan of ``COHERENCE_SCAN_STEPS`` steps up to ``COHERENCE_SCAN_RANGE`` times the estimate
``sqrt(2) lambda_c^2 / (2 pi sigma)`` (``sigma`` the RMS spectral width) and refined by bisection; the same of the traced lines
and of twice as many lines (the convergence of the quadrature); and the largest ``| |gamma_lines| - |gamma| |`` over ``D`` in
``[0, CONVERGENCE_RANGE x L_c]`` for ``n`` and ``2 n`` lines. A spectrum whose ``|gamma|`` does not fall to ``1/e`` within the
scan reports no coherence length (``reason``). Reported lengths are rounded to ``COHERENCE_DIGITS`` decimals of a millimetre,
differences to ``GAMMA_DIGITS`` decimals. Every operation is libm-free (``math.exact_trig``, ``math.exact_erf``) with correctly rounded sums
(``math.fsum``): the same values in every runtime.

References: M. Born and E. Wolf, Principles of Optics, 7th ed., sec. 10.3 and 7.5.8 (coherence and the spectrum); L. Mandel
and E. Wolf, Optical Coherence and Quantum Optics (1995), sec. 4.3.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ...math.exact_erf import erf_interval
from ...math.exact_trig import atan2, cos_sin, exp

if TYPE_CHECKING:  # (the contracts import this module lazily)
    from ...contracts.components import EngineeringComponentSnapshot, SpectralDensity
    from ...contracts.results import SourceSpectrum

#: ln 2 (the correctly rounded double).
LN2 = 0.6931471805599453
#: A Gaussian line of full width w (nm) at lambda (nm) has the coherence length (2 sqrt(ln 2) / pi) lambda^2 / w (``scene/lower``).
LINE_COHERENCE_FACTOR = 2.0 * math.sqrt(LN2) / math.pi
NM_TO_MM = 1e-6
TWO_PI = 2.0 * math.pi
#: Simpson's rule for gamma(D): at least this many intervals, and this many per cycle of the phase across the band.
SIMPSON_MIN_INTERVALS = 2048
SIMPSON_INTERVALS_PER_CYCLE = 64
#: The coherence-length scan: this many steps up to this many times the RMS-width estimate.
COHERENCE_SCAN_STEPS = 256
COHERENCE_SCAN_RANGE = 16.0
COHERENCE_BISECTIONS = 60
#: The quadrature's convergence is stated over 0 .. this many coherence lengths.
CONVERGENCE_RANGE = 3.0
#: Reported lengths are rounded to this many decimals of a millimetre (1 pm), differences of degrees of coherence to this many.
COHERENCE_DIGITS = 9
GAMMA_DIGITS = 12
#: 1/e (the correctly rounded double).
INVERSE_E = 0.36787944117144233
#: A two-beam pair's degree of coherence (``pair_coherence``) is integrated on at most this many Simpson intervals.
PAIR_MAX_INTERVALS = 1 << 21
#: ``mean_wavenumber_offset`` is raised by this (relative) above the quadrature's error (Simpson on smooth pieces with at least
#: ``SIMPSON_MIN_INTERVALS`` intervals: below 1e-9 relative).
OFFSET_MARGIN = 1e-6


# ------------------------------------------------------------------------------------------------- quadrature lines


def nodes_nm(band_nm: tuple[float, float], line_count: int) -> tuple[float, ...]:
    """The centres of ``line_count`` equal bins of the band (the contract's own rule, the same floats)."""

    low, high = band_nm
    width = (high - low) / line_count
    return tuple(low + (k + 0.5) * width for k in range(line_count))


def _edges(band_nm: tuple[float, float], line_count: int) -> np.ndarray:
    low, high = band_nm
    width = (high - low) / line_count
    return np.array([low + k * width for k in range(line_count)] + [high], dtype=np.float64)


def _tabulated_cumulative(rows_nm: Sequence[float], density: Sequence[float], x: float) -> float:
    """Integral of the table (linear between rows) from its first row to ``x``, piece by piece (a fixed order of operations)."""

    total = 0.0
    for k in range(len(rows_nm) - 1):
        w0, w1, d0, d1 = rows_nm[k], rows_nm[k + 1], density[k], density[k + 1]
        if x <= w0:
            break
        end = x if x < w1 else w1
        slope = (d1 - d0) / (w1 - w0)
        total += (end - w0) * (d0 + 0.5 * slope * (end - w0))
    return total


def bin_weights(density: SpectralDensity) -> tuple[float, ...]:
    """The exact integral of the density over each bin (relative units; module docstring)."""

    shape = density.shape
    edges = _edges(density.band_nm, density.line_count)
    if shape.kind == "gaussian":
        scale = 2.0 * math.sqrt(LN2) / shape.fwhm_nm
        x = scale * (edges - shape.center_nm)
        return tuple(float(v) for v in erf_interval(x[:-1], x[1:]))
    if shape.kind == "lorentzian":
        angle = atan2(2.0 * (edges - shape.center_nm) / shape.fwhm_nm, np.ones_like(edges))
        return tuple(float(v) for v in angle[1:] - angle[:-1])
    if shape.kind == "top_hat":
        return tuple(float(v) for v in edges[1:] - edges[:-1])
    cumulative = [_tabulated_cumulative(shape.wavelength_nm, shape.density, float(e)) for e in edges]
    return tuple(b - a for a, b in zip(cumulative, cumulative[1:]))


def quadrature_lines(density: SpectralDensity) -> tuple[tuple[float, float], ...]:
    """``(wavelength_nm, weight)`` of every quadrature line (weights relative: the bin integrals)."""

    return tuple(zip(nodes_nm(density.band_nm, density.line_count), bin_weights(density), strict=True))


def bin_width_nm(density: SpectralDensity) -> float:
    """Width of a bin: the full width of each quadrature line."""

    low, high = density.band_nm
    return (high - low) / density.line_count


# ------------------------------------------------------------------------------------------------------- coherence


def density_at(density: SpectralDensity, wavelength_nm: np.ndarray) -> np.ndarray:
    """``S(lambda)`` (relative) at the given wavelengths of the band."""

    shape = density.shape
    x = np.asarray(wavelength_nm, dtype=np.float64)
    if shape.kind == "gaussian":
        u = (x - shape.center_nm) / shape.fwhm_nm
        return exp(-4.0 * LN2 * u * u)
    if shape.kind == "lorentzian":
        u = 2.0 * (x - shape.center_nm) / shape.fwhm_nm
        return 1.0 / (1.0 + u * u)
    if shape.kind == "top_hat":
        return np.ones_like(x)
    return np.interp(x, np.asarray(shape.wavelength_nm, dtype=np.float64), np.asarray(shape.density, dtype=np.float64))


def _simpson_grid(density: SpectralDensity, largest_opd_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(wavelengths, Simpson weight x S) over the band for gamma up to ``largest_opd_mm``: composite Simpson on each piece
    between the band's ends and, for a table, its rows inside the band (its kinks), with an even number of intervals per piece
    and the pieces' intervals in proportion to their lengths."""

    low, high = density.band_nm
    cycles = largest_opd_mm / NM_TO_MM * (1.0 / low - 1.0 / high)
    return _grid(density, max(SIMPSON_MIN_INTERVALS, int(math.ceil(SIMPSON_INTERVALS_PER_CYCLE * cycles))))


def _grid(density: SpectralDensity, intervals: int, extra_breaks: Sequence[float] = ()) -> tuple[np.ndarray, np.ndarray]:
    """(wavelengths, Simpson weight x S) of composite Simpson with about ``intervals`` intervals over the band (``_simpson_grid``),
    with ``extra_breaks`` inside the band as further piece ends (where an integrand has a kink)."""

    low, high = density.band_nm
    breaks = [low]
    if density.shape.kind == "tabulated":
        breaks += [w for w in density.shape.wavelength_nm if low < w < high]
    if extra_breaks:
        breaks = sorted(set(breaks) | {w for w in extra_breaks if low < w < high})
    breaks.append(high)
    points: list[float] = []
    weights: list[float] = []
    for a, b in zip(breaks, breaks[1:]):
        count = max(2, int(math.ceil(intervals * (b - a) / (high - low))))
        count += count % 2
        step = (b - a) / count
        third = step / 3.0
        piece = [a + k * step for k in range(count)] + [b]
        piece_weights = [third * (1.0 if k in (0, count) else (4.0 if k % 2 else 2.0)) for k in range(count + 1)]
        if points:  # the shared end of two pieces
            weights[-1] += piece_weights[0]
            piece, piece_weights = piece[1:], piece_weights[1:]
        points += piece
        weights += piece_weights
    wavelengths = np.array(points, dtype=np.float64)
    return wavelengths, np.array(weights, dtype=np.float64) * density_at(density, wavelengths)


def _magnitudes(wavelengths: np.ndarray, weights: np.ndarray, reference_nm: float, opd_mm: Sequence[float],
                envelopes: Sequence[np.ndarray] | None = None) -> list[float]:
    """``|sum_j weight_j env_j(D) exp(2 pi i D (1/lambda_j - 1/lambda_ref))| / sum_j weight_j`` for each ``D`` (correctly rounded
    sums; the common carrier exp(2 pi i D / lambda_ref) does not change the magnitude)."""

    norm = math.fsum(weights.tolist())
    detune = 1.0 / wavelengths - 1.0 / reference_nm  # 1/nm
    out = []
    for index, d in enumerate(opd_mm):
        c, s = cos_sin(TWO_PI * (d / NM_TO_MM) * detune)
        w = weights if envelopes is None else weights * envelopes[index]
        re, im = math.fsum((w * c).tolist()), math.fsum((w * s).tolist())
        out.append(math.sqrt(re * re + im * im) / norm)
    return out


def coherence(density: SpectralDensity, opd_mm: Sequence[float], largest_opd_mm: float | None = None) -> list[float]:
    """``|gamma(D)|`` of the continuous spectrum over its band at each path difference ``D`` (mm); the Simpson grid resolves
    the phase up to ``largest_opd_mm`` (default: the largest ``D`` asked)."""

    largest = max((abs(d) for d in opd_mm), default=0.0) if largest_opd_mm is None else largest_opd_mm
    wavelengths, weights = _simpson_grid(density, largest)
    return _magnitudes(wavelengths, weights, _centroid(wavelengths, weights), opd_mm)


def line_coherence(lines: Sequence[tuple[float, float]], line_width_nm: float, opd_mm: Sequence[float]) -> list[float]:
    """``|gamma_lines(D)|`` of quadrature lines, each a Gaussian line of full width ``line_width_nm`` (the Engine's model)."""

    wavelengths = np.array([w for w, _ in lines], dtype=np.float64)
    weights = np.array([p for _, p in lines], dtype=np.float64)
    lengths = LINE_COHERENCE_FACTOR * wavelengths * wavelengths / line_width_nm * NM_TO_MM
    envelopes = [exp(-(d / lengths) * (d / lengths)) for d in opd_mm]
    return _magnitudes(wavelengths, weights, _centroid(wavelengths, weights), opd_mm, envelopes)


def _centroid(wavelengths: np.ndarray, weights: np.ndarray) -> float:
    return math.fsum((weights * wavelengths).tolist()) / math.fsum(weights.tolist())


def moments(density: SpectralDensity) -> tuple[float, float]:
    """(power-weighted mean wavelength, RMS width) of the spectrum over its band, nm (Simpson, as ``coherence``)."""

    wavelengths, weights = _simpson_grid(density, 0.0)
    mean = _centroid(wavelengths, weights)
    spread = math.fsum((weights * (wavelengths - mean) * (wavelengths - mean)).tolist()) / math.fsum(weights.tolist())
    return mean, math.sqrt(spread)


def coherence_length_mm(gamma, estimate_mm: float) -> float | None:
    """The smallest ``D > 0`` with ``|gamma(D)| = 1/e``: a scan of ``COHERENCE_SCAN_STEPS`` steps up to ``COHERENCE_SCAN_RANGE``
    times ``estimate_mm``, then bisection; ``None`` when ``|gamma|`` stays above ``1/e`` over the scan."""

    step = COHERENCE_SCAN_RANGE * estimate_mm / COHERENCE_SCAN_STEPS
    grid = [k * step for k in range(COHERENCE_SCAN_STEPS + 1)]
    values = gamma(grid)
    crossing = next((k for k, v in enumerate(values) if v <= INVERSE_E), None)
    if crossing is None:
        return None
    low, high = grid[crossing - 1], grid[crossing]
    for _ in range(COHERENCE_BISECTIONS):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if gamma([middle])[0] > INVERSE_E:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ------------------------------------------------------------------------------------------------- two-beam pairs


def _wavenumber_offset(wavelengths: np.ndarray, reference_nm: float) -> np.ndarray:
    """``k - k_ref`` (1/mm) of the vacuum wavenumbers ``k = 2 pi / lambda``."""

    return TWO_PI / NM_TO_MM * (1.0 / wavelengths - 1.0 / reference_nm)


def pair_coherence(density: SpectralDensity, reference_nm: float, delay_mm: float, dispersion_mm2: float) -> complex | None:
    """The complex degree of coherence of two fields of the continuous spectrum whose relative spectral phase is
    ``phi(k) = phi(k_ref) + D (k - k_ref) + B (k - k_ref)^2 / 2`` (``D = delay_mm`` the difference of their group optical
    paths, ``B = dispersion_mm2`` of their group-delay dispersions, both at the line ``reference_nm``, the expansion of
    ``optics/interference/coherence.py``), relative to the phase at the line:

        gamma = integral S(lambda) exp(i [D (k - k_ref) + B (k - k_ref)^2 / 2]) dlambda / integral S(lambda) dlambda

    over the band, by composite Simpson with ``SIMPSON_INTERVALS_PER_CYCLE`` intervals per cycle of that phase across the band
    (at least ``SIMPSON_MIN_INTERVALS``); ``None`` where that takes more than ``PAIR_MAX_INTERVALS`` intervals (a path difference
    too large for the integral at that resolution). Two fields of the line ``reference_nm`` then add to the power
    ``P_a + P_b + 2 sqrt(P_a P_b) Re[gamma exp(i phi(k_ref))]`` of the whole spectrum, ``|gamma|`` the contrast of their fringes."""

    low, high = density.band_nm
    span = TWO_PI / NM_TO_MM * (1.0 / low - 1.0 / high)
    far = TWO_PI / NM_TO_MM * max(abs(1.0 / low - 1.0 / reference_nm), abs(1.0 / high - 1.0 / reference_nm))
    cycles = (abs(delay_mm) * span + abs(dispersion_mm2) * far * far) / TWO_PI
    intervals = max(SIMPSON_MIN_INTERVALS, int(math.ceil(SIMPSON_INTERVALS_PER_CYCLE * cycles)))
    if intervals > PAIR_MAX_INTERVALS:
        return None
    wavelengths, weights = _grid(density, intervals)
    x = _wavenumber_offset(wavelengths, reference_nm)
    c, s = cos_sin(delay_mm * x + 0.5 * dispersion_mm2 * (x * x))
    norm = math.fsum(weights.tolist())
    return complex(math.fsum((weights * c).tolist()) / norm, math.fsum((weights * s).tolist()) / norm)


def mean_wavenumber_offset(density: SpectralDensity, reference_nm: float) -> float:
    """The power-weighted mean of ``|k - k_ref|`` (1/mm) over the spectrum: a relative group delay known within ``B`` (a length)
    changes the complex degree of coherence by at most ``B`` times it (``optics/interference/coherence.py``). Simpson with the
    line as a piece end (the kink of ``|k - k_ref|``) and ``SIMPSON_MIN_INTERVALS`` intervals, raised by
    ``OFFSET_MARGIN`` (relative) above the quadrature's error."""

    wavelengths, weights = _grid(density, SIMPSON_MIN_INTERVALS, (reference_nm,))
    offset = np.abs(_wavenumber_offset(wavelengths, reference_nm))
    return math.fsum((weights * offset).tolist()) / math.fsum(weights.tolist()) * (1.0 + OFFSET_MARGIN)


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def doubled(density: SpectralDensity) -> SpectralDensity:
    """The same spectrum with twice the quadrature lines."""

    return density.model_copy(update={"line_count": 2 * density.line_count})


def spectrum_summary(source_id: str, density: SpectralDensity) -> SourceSpectrum:
    """The report of one continuous source (module docstring)."""

    from ...contracts.results import SourceSpectrum

    mean, rms = moments(density)
    estimate = math.sqrt(2.0) * mean * mean / (TWO_PI * rms) * NM_TO_MM
    lines, fine = quadrature_lines(density), quadrature_lines(doubled(density))
    width, fine_width = bin_width_nm(density), bin_width_nm(doubled(density))
    reach = COHERENCE_SCAN_RANGE * estimate  # (one Simpson grid for every evaluation of this source)
    length = coherence_length_mm(lambda d: coherence(density, d, reach), estimate)
    lines_length = coherence_length_mm(lambda d: line_coherence(lines, width, d), estimate)
    fine_length = coherence_length_mm(lambda d: line_coherence(fine, fine_width, d), estimate)
    span = CONVERGENCE_RANGE * (length if length is not None else estimate)
    grid = [span * k / COHERENCE_SCAN_STEPS for k in range(COHERENCE_SCAN_STEPS + 1)]
    exact = coherence(density, grid, max(reach, span))
    differences = [max(abs(a - b) for a, b in zip(line_coherence(ls, w, grid), exact, strict=True)) for ls, w in ((lines, width), (fine, fine_width))]
    return SourceSpectrum(
        source_id=source_id, basis="SAMPLED_CONTINUOUS", line_count=density.line_count, shape=density.shape.kind.upper(),
        band_nm=density.band_nm, quadrature="MIDPOINT_BINS", line_width_nm=width, centroid_nm=_rounded(mean, COHERENCE_DIGITS),
        rms_width_nm=_rounded(rms, COHERENCE_DIGITS), coherence_length_mm=_rounded(length, COHERENCE_DIGITS),
        coherence_length_lines_mm=_rounded(lines_length, COHERENCE_DIGITS),
        coherence_length_doubled_lines_mm=_rounded(fine_length, COHERENCE_DIGITS),
        gamma_max_difference=_rounded(differences[0], GAMMA_DIGITS), gamma_max_difference_doubled=_rounded(differences[1], GAMMA_DIGITS),
        convergence_range_mm=_rounded(span, COHERENCE_DIGITS),
        reason=None if length is not None else "COHERENCE_BEYOND_SCAN: |gamma| of the spectrum stays above 1/e over the scanned path differences",
    )


def source_spectra(components: Sequence[EngineeringComponentSnapshot]) -> tuple[SourceSpectrum, ...]:
    """Per Gaussian-beam source, how its spectrum is traced -- when at least one source states a continuous spectrum (else
    nothing): ``SAMPLED_CONTINUOUS`` (``spectrum_summary``) or ``DISCRETE`` (its lines as stated)."""

    from ...contracts.results import SourceSpectrum

    sources = [c.source for c in components if c.source is not None]
    if not any(s.spectral_density is not None for s in sources):
        return ()
    out = []
    for source in sources:
        if source.spectral_density is not None:
            out.append(spectrum_summary(source.source_id, source.spectral_density))
        else:
            out.append(SourceSpectrum(source_id=source.source_id, basis="DISCRETE", line_count=max(1, len(source.spectrum))))
    return tuple(out)


__all__ = ["bin_weights", "bin_width_nm", "coherence", "coherence_length_mm", "line_coherence", "mean_wavenumber_offset", "moments",
           "nodes_nm", "pair_coherence", "quadrature_lines", "source_spectra", "spectrum_summary"]
