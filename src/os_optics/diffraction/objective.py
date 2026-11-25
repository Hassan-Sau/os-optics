"""The ideal aplanatic objective and the map of a pupil field onto its reference sphere (Richards and Wolf, Proc. R. Soc. A
253, 358 (1959)).

Conventions. The focus is the origin; light travels toward ``+z``. A point ``(x_p, y_p)`` of the entrance pupil, at height
``h`` and azimuth ``phi``, maps to the ray of direction

    s = (-sin(theta) cos(phi), -sin(theta) sin(phi), cos(theta)),    h = F n sin(theta)     (the sine condition)

with ``F`` the effective focal length (``1 / power``) and ``n`` the index of the medium at the focus, so the pupil radius is
``F NA``. The reference sphere has radius ``f' = n F``. A pupil field ``(J_x, J_y)`` splits into its radial part, which
becomes ``e_p = (cos(theta) cos(phi), cos(theta) sin(phi), sin(theta))`` behind the objective, and its azimuthal part, which
stays ``e_s = (-sin(phi), cos(phi), 0)``; ``(e_p, e_s, s)`` is right-handed. Energy conservation through the aplanatic
objective gives the amplitude factor ``sqrt(cos(theta))``.

Units. Fields are in sqrt(W/mm^2), normalized so that a plane wave in its medium has ``|E|^2`` equal to its irradiance; the
magnetic field ``H = s x E`` of each plane wave is in the same units, so the axial Poynting flux density is
``S_z = Re(E_x conj(H_y) - E_y conj(H_x))`` (W/mm^2).

Window. A plane-parallel plate of index ``n_w`` and thickness ``t`` between the objective and the focus, normal to the axis,
in the focal medium on both sides, multiplies each plane-wave component of the focal field by its single-pass transmission
``t_12 t_21`` (Fresnel amplitudes for ``p`` and ``s``, or 1 for ideal anti-reflection surfaces) and the phase
``k_0 t (n_w cos(theta_w) - n cos(theta))`` (Torok et al., J. Opt. Soc. Am. A 12, 325 (1995)). Light reflected inside the
plate forms separate, weaker foci that are not included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..units import NM_TO_MM

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


@dataclass(frozen=True)
class Window:
    """A plane-parallel plate between the objective and the focus (module docstring)."""

    thickness_mm: float
    index: float
    fresnel: bool = True

    def transfer(self, theta: Array, focal_index: float, k0: float) -> tuple[CArray, CArray, Array]:
        """``(t_p, t_s, phase)`` of every plane-wave component at angle ``theta`` in the focal medium."""

        n1, n2 = focal_index, self.index
        cos1 = np.cos(theta)
        sin2 = n1 * np.sin(theta) / n2
        cos2 = np.sqrt(1.0 - sin2 * sin2)
        phase = k0 * self.thickness_mm * (n2 * cos2 - n1 * cos1)
        if not self.fresnel:
            ones = np.ones(theta.shape, dtype=np.complex128)
            return ones, ones.copy(), phase
        # amplitude coefficients into the plate (1 -> 2) and out of it (2 -> 1)
        ts12 = 2.0 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
        tp12 = 2.0 * n1 * cos1 / (n2 * cos1 + n1 * cos2)
        ts21 = 2.0 * n2 * cos2 / (n2 * cos2 + n1 * cos1)
        tp21 = 2.0 * n2 * cos2 / (n1 * cos2 + n2 * cos1)
        return (tp12 * tp21).astype(np.complex128), (ts12 * ts21).astype(np.complex128), phase

    def paraxial_focal_shift_mm(self, focal_index: float) -> float:
        """The paraxial displacement of the focus away from the objective, ``t (1 - n / n_w)``."""

        return self.thickness_mm * (1.0 - focal_index / self.index)


@dataclass(frozen=True)
class Objective:
    """An ideal aplanatic objective: no aberration at any field point used (module docstring)."""

    wavelength_nm: float
    numerical_aperture: float
    focal_length_mm: float
    focal_index: float = 1.0
    window: Window | None = None

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_nm * NM_TO_MM

    @property
    def k0(self) -> float:
        """Vacuum wavenumber, rad/mm."""
        return 2.0 * math.pi / self.wavelength_mm

    @property
    def k(self) -> float:
        """Wavenumber in the focal medium, rad/mm."""
        return self.focal_index * self.k0

    @property
    def sine_max(self) -> float:
        return self.numerical_aperture / self.focal_index

    @property
    def theta_max(self) -> float:
        return math.asin(self.sine_max)

    @property
    def sphere_radius_mm(self) -> float:
        return self.focal_index * self.focal_length_mm

    @property
    def pupil_radius_mm(self) -> float:
        return self.focal_length_mm * self.numerical_aperture

    @property
    def prefactor(self) -> complex:
        """``-i f' / lambda_medium`` of the Debye integral ``E(r) = -i f' / lambda_m Integral a(s) exp(i k s.r) dOmega``."""
        return -1j * self.sphere_radius_mm * self.focal_index / self.wavelength_mm

    @property
    def fresnel_number(self) -> float:
        """``a^2 / (lambda_m f')`` with ``a = f' sin(theta_max)``: equal to ``F NA^2 / lambda_0``."""
        return self.focal_length_mm * self.numerical_aperture * self.numerical_aperture / self.wavelength_mm

    def sine_of_height(self, h: Array) -> Array:
        return h / (self.focal_length_mm * self.focal_index)

    def height_of_sine(self, sine: Array) -> Array:
        return sine * self.focal_length_mm * self.focal_index

    def transfer(self, theta: Array) -> tuple[CArray, CArray, Array]:
        """``(t_p, t_s, phase)`` between the reference sphere and the focus: the window's, or unity."""

        if self.window is None:
            ones = np.ones(np.shape(theta), dtype=np.complex128)
            return ones, ones.copy(), np.zeros(np.shape(theta))
        return self.window.transfer(np.asarray(theta, dtype=np.float64), self.focal_index, self.k0)


def directions(theta: Array, phi: Array) -> Array:
    """Unit propagation directions ``s`` (3, ...) toward the focus."""

    st = np.sin(theta)
    return np.stack((-st * np.cos(phi), -st * np.sin(phi), np.cos(theta)))


def sphere_field(theta: Array, phi: Array, jx: CArray, jy: CArray, tp: CArray, ts: CArray, apodize: bool = True) -> tuple[CArray, CArray]:
    """``(E, H)`` (3, ...) on the reference sphere of the pupil field ``(jx, jy)`` mapped to ``(theta, phi)``, with the
    aplanatic factor ``sqrt(cos(theta))`` (``apodize``; a traced ray bundle carries its own amplitude) and the transmissions
    ``t_p``, ``t_s`` (module docstring)."""

    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    apod = np.sqrt(ct) if apodize else np.ones(np.shape(ct))
    radial = (jx * cp + jy * sp) * apod * tp
    azimuthal = (-jx * sp + jy * cp) * apod * ts
    e_p = (ct * cp, ct * sp, st)
    e_s = (-sp, cp, np.zeros(np.shape(phi)))
    field = np.stack([radial * e_p[i] + azimuthal * e_s[i] for i in range(len(e_p))])
    # s x e_p = e_s and s x e_s = -e_p
    magnetic = np.stack([radial * e_s[i] - azimuthal * e_p[i] for i in range(len(e_p))])
    return field, magnetic


__all__ = ["Objective", "Window", "directions", "sphere_field"]
