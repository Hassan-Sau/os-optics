"""Scalar propagation of a sampled field ``U`` (sqrt(W/mm^2), ``exp(-i w t)``, propagation ``exp(+i k z)``) to a plane.

``angular_spectrum``: free space of index ``n`` over a distance ``z`` to a parallel plane, the exact scalar transfer function
``H = exp(i 2 pi z sqrt((n / lambda0)^2 - fx^2 - fy^2))`` (evanescent frequencies removed) on the grid zero-padded to
``padding`` times its extent, band-limited to ``|f| <= 1 / (lambda sqrt((2 df z)^2 + 1))`` per axis (Matsushima and
Shimobaba, Opt. Express 17, 19662 (2009)), which keeps the circular convolution of the FFT from folding light back into the
window. The output grid is the padded input grid.

``paraxial_operator``: one axis of the Collins integral of a first-order system in canonical coordinates (position, ``n``
times the direction cosine) with matrix ``[[a, b], [c, d]]`` (``ad - bc = 1``)::

    U2(u) = sqrt(1 / (i lambda0 b)) Integral U1(x) exp(i pi (a x^2 - 2 x u + d u^2) / (lambda0 b)) dx

as a matrix acting on the grid. Two exact discretizations, the first whose sampling condition holds:

* ``FACTORED`` (``a != 0``): ``[[a, b], [c, d]] = [[1, 0], [c / a, 1]] [[a, 0], [0, 1 / a]] [[1, b / a], [0, 1]]`` -- free
  propagation over ``b' = b / a`` by the transfer function ``exp(-i pi lambda0 b' f^2)`` on the padded grid, then the
  magnification ``a`` (``U(u) = U'(u / a) / sqrt(a)``: output pitch ``|a| p``), then the thin-lens phase
  ``exp(i pi (c / a) u^2 / lambda0)``. Condition: ``lambda0 |b'| <= N p^2`` (``N`` the padded count): the transfer function
  is sampled and the propagated field stays within the window.
* ``CHIRP`` (``b != 0``): the integral summed directly on the input grid at ``N`` output points of pitch
  ``lambda0 |b| / (N p)``. Condition: the input chirp ``a x^2 / (lambda0 b)`` changes by at most ``pi`` per pitch at the
  grid's edge, ``|a / b| 2 x_max p / lambda0 <= 1``.

An image plane (``b = 0``) is the factored form with ``b' = 0``: a scaled copy of the field. A Fourier plane (``a = 0``) is
the chirp form. The thin-lens phase changes the phase only, not the irradiance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

FACTORED = "FACTORED"
CHIRP = "CHIRP"
#: The share of a field's power whose spatial-frequency radius gives its angular extent.
ANGULAR_EXTENT_SHARE = 0.99


def grid_axis(count: int, pitch: float) -> Array:
    """The centred coordinates of ``count`` points at ``pitch``."""

    return (np.arange(count, dtype=np.float64) - 0.5 * (count - 1)) * pitch


def _padded_axis(count: int, padded: int, pitch: float) -> tuple[int, Array]:
    """Where the input grid starts in the padded one, and the padded grid's coordinates (the input points keep theirs)."""

    offset = (padded - count) // 2
    return offset, (np.arange(padded, dtype=np.float64) - offset - 0.5 * (count - 1)) * pitch


@dataclass(frozen=True)
class AxisOperator:
    """One axis of a paraxial transfer: ``matrix`` (``(n_out, n_in)``) maps the input samples to the output samples at
    ``coordinates`` (mm, ascending), by ``route``; ``condition`` is its sampling condition's value (at most 1)."""

    matrix: CArray
    coordinates: Array
    route: str
    condition: float


def paraxial_operator(count: int, pitch: float, abcd: tuple[float, float, float, float], wavelength_mm: float,
                      padding: int) -> AxisOperator | None:
    """The operator of one axis (module docstring); ``None`` when neither discretization's condition holds."""

    a, b, c, d = abcd
    padded = padding * count
    x = grid_axis(count, pitch)
    x_max = float(np.abs(x).max()) + 0.5 * pitch
    factored = wavelength_mm * abs(b / a) / (padded * pitch * pitch) if a != 0.0 else math.inf
    chirp = abs(a / b) * 2.0 * x_max * pitch / wavelength_mm if b != 0.0 else math.inf
    if factored <= 1.0:
        offset, xp = _padded_axis(count, padded, pitch)
        unit = np.zeros((padded, count), dtype=np.complex128)
        unit[offset + np.arange(count), np.arange(count)] = 1.0
        f = np.fft.fftfreq(padded, pitch)
        transfer = np.exp(-1j * math.pi * wavelength_mm * (b / a) * f * f)
        free = np.fft.ifft(np.fft.fft(unit, axis=0) * transfer[:, None], axis=0)
        u = a * xp
        scale = 1.0 / np.sqrt(complex(a)) * np.exp(1j * math.pi * (c / a) * u * u / wavelength_mm)
        matrix = scale[:, None] * free
        if a < 0.0:
            matrix, u = matrix[::-1], u[::-1]
        return AxisOperator(matrix, u, FACTORED, factored)
    if chirp <= 1.0:
        du = wavelength_mm * abs(b) / (padded * pitch)
        u = (np.arange(padded, dtype=np.float64) - padded // 2) * du
        kernel = np.exp(1j * math.pi * (a * x[None, :] ** 2 - 2.0 * x[None, :] * u[:, None] + d * u[:, None] ** 2) / (wavelength_mm * b))
        matrix = np.sqrt(1.0 / (1j * wavelength_mm * b)) * pitch * kernel
        return AxisOperator(matrix, u, CHIRP, chirp)
    return None


def upsampled_axis(count: int, pitch: float) -> tuple[Array, Array]:
    """The band-limited interpolation of an axis of ``count`` samples (zero beyond them) to half the pitch: ``(the matrix
    (4 count, count), the fine coordinates)``. The samples are zero-padded to twice their extent and interpolated by the
    discrete Fourier series of that period (the Nyquist term split evenly), so the fine grid holds the input samples at its
    even points and the field's band-limited values between them, its tails included."""

    extended = 2 * count
    offset = count // 2
    x0 = (-offset - 0.5 * (count - 1)) * pitch
    unit = np.zeros((extended, count), dtype=np.complex128)
    unit[offset + np.arange(count), np.arange(count)] = 1.0
    spectrum = np.fft.fft(unit, axis=0)
    fine = np.zeros((2 * extended, count), dtype=np.complex128)
    half = extended // 2
    fine[:half] = spectrum[:half]
    fine[half] = 0.5 * spectrum[half]
    fine[2 * extended - half] = 0.5 * spectrum[half]
    fine[2 * extended - half + 1:] = spectrum[half + 1:]
    matrix = 2.0 * np.fft.ifft(fine, axis=0)
    return matrix, x0 + np.arange(2 * extended, dtype=np.float64) * (0.5 * pitch)


def reference_operator(count: int, pitch: float, abcd: tuple[float, float, float, float], wavelength_mm: float, padding: int,
                       main: AxisOperator) -> AxisOperator | None:
    """The same transfer as ``main`` (``paraxial_operator``'s result) on an output window twice as wide, on the same output
    lattice (it holds every point of ``main``'s): the ``FACTORED`` route with twice the padding (its window is the padded
    grid's), the ``CHIRP`` route summed on the input interpolated to half the pitch (``upsampled_axis``; its window is
    ``lambda0 |b|`` over the pitch) at ``main``'s output pitch over twice its extent. Light that leaves ``main``'s window
    and folds back into it (the circular convolution, or the periodic sum of the chirp route) lies outside it here."""

    a, b, _c, d = abcd
    if main.route == FACTORED:
        wider = paraxial_operator(count, pitch, abcd, wavelength_mm, 2 * padding)
        return wider if wider is not None and wider.route == FACTORED else None
    padded = padding * count
    du = wavelength_mm * abs(b) / (padded * pitch)
    u = (np.arange(2 * padded, dtype=np.float64) - padded) * du
    upsample, x = upsampled_axis(count, pitch)
    kernel = np.exp(1j * math.pi * (a * x[None, :] ** 2 - 2.0 * x[None, :] * u[:, None] + d * u[:, None] ** 2) / (wavelength_mm * b))
    matrix = (np.sqrt(1.0 / (1j * wavelength_mm * b)) * (0.5 * pitch) * kernel) @ upsample
    return AxisOperator(matrix, u, CHIRP, main.condition)


def reference_angular_spectrum(field: CArray, px: float, py: float, wavelength_mm: float, n: float, z: float, padding: int
                               ) -> AngularSpectrum:
    """``angular_spectrum`` with twice the padding: twice the window, the same output lattice. Within its band limit the light
    moves at most half its padded window from where it starts (``z`` times the band limit's tangent), so a field that starts
    within the unpadded grid lands inside this window: what leaves the first window is outside it here."""

    return angular_spectrum(field, px, py, wavelength_mm, n, z, 2 * padding)


def paraxial_conditions(count: int, pitch: float, abcd: tuple[float, float, float, float], wavelength_mm: float,
                        padding: int) -> dict[str, float]:
    """Both discretizations' sampling conditions of one axis (module docstring)."""

    a, b, _c, _d = abcd
    x_max = float(np.abs(grid_axis(count, pitch)).max()) + 0.5 * pitch
    return {FACTORED: wavelength_mm * abs(b / a) / (padding * count * pitch * pitch) if a != 0.0 else math.inf,
            CHIRP: abs(a / b) * 2.0 * x_max * pitch / wavelength_mm if b != 0.0 else math.inf}


def paraxial_transfer(field: CArray, px: float, py: float, abcd_u: tuple[float, float, float, float],
                      abcd_v: tuple[float, float, float, float], wavelength_mm: float, padding: int
                      ) -> tuple[CArray, AxisOperator, AxisOperator] | None:
    """The field (``(ny, nx)``, columns along the input axis paired with u) through the separable transfer: ``(field on (v, u),
    the u operator, the v operator)``; ``None`` when an axis has no valid discretization."""

    ny, nx = field.shape
    ku = paraxial_operator(nx, px, abcd_u, wavelength_mm, padding)
    kv = paraxial_operator(ny, py, abcd_v, wavelength_mm, padding)
    if ku is None or kv is None:
        return None
    return kv.matrix @ field @ ku.matrix.T, ku, kv


@dataclass(frozen=True)
class AngularSpectrum:
    """A field after ``angular_spectrum``: on the padded grid ``(y, x)`` of the source frame, with the band-limit sines."""

    field: CArray
    x: Array
    y: Array
    band_sine: tuple[float, float]


def angular_spectrum(field: CArray, px: float, py: float, wavelength_mm: float, n: float, z: float, padding: int) -> AngularSpectrum:
    """Free propagation over ``z`` (mm) in a medium of index ``n`` (module docstring)."""

    ny, nx = field.shape
    nxp, nyp = padding * nx, padding * ny
    ox, x = _padded_axis(nx, nxp, px)
    oy, y = _padded_axis(ny, nyp, py)
    padded = np.zeros((nyp, nxp), dtype=np.complex128)
    padded[oy:oy + ny, ox:ox + nx] = field
    lam = wavelength_mm / n
    fx = np.fft.fftfreq(nxp, px)
    fy = np.fft.fftfreq(nyp, py)
    limit_x = 1.0 / (lam * math.sqrt((2.0 * z / (nxp * px)) ** 2 + 1.0))
    limit_y = 1.0 / (lam * math.sqrt((2.0 * z / (nyp * py)) ** 2 + 1.0))
    argument = 1.0 / (lam * lam) - fy[:, None] ** 2 - fx[None, :] ** 2
    band = (np.abs(fx)[None, :] <= limit_x) & (np.abs(fy)[:, None] <= limit_y) & (argument > 0.0)
    transfer = np.where(band, np.exp(2j * math.pi * z * np.sqrt(np.maximum(argument, 0.0))), 0.0)
    out = np.fft.ifft2(np.fft.fft2(padded) * transfer)
    return AngularSpectrum(out, x, y, (min(1.0, lam * limit_x), min(1.0, lam * limit_y)))


def angular_extent_sine(field: CArray, px: float, py: float, wavelength_mm: float, n: float, share: float = ANGULAR_EXTENT_SHARE) -> float:
    """The sine (in the medium) of the radius of spatial frequencies holding ``share`` of the field's power: how far from its
    axis the field's light travels."""

    ny, nx = field.shape
    spectrum = np.abs(np.fft.fft2(field, s=(2 * ny, 2 * nx))) ** 2
    fx = np.fft.fftfreq(2 * nx, px)
    fy = np.fft.fftfreq(2 * ny, py)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).ravel()
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(spectrum.ravel()[order])
    index = int(np.searchsorted(cumulative, share * cumulative[-1]))
    return float(radius[order][min(index, radius.size - 1)]) * wavelength_mm / n


__all__ = ["CHIRP", "FACTORED", "AngularSpectrum", "AxisOperator", "angular_extent_sine", "angular_spectrum", "grid_axis",
           "paraxial_conditions", "paraxial_operator", "paraxial_transfer"]
