"""The focal field of an objective lit by one or more mutually incoherent pupil fields, by three independent evaluations of
the same vector Debye integral:

* ``"bessel"`` -- the azimuthal integral done analytically (``bessel.py``), for a rotationally symmetric pupil;
* ``"direct"`` -- the sphere summed on a polar Gauss-Legendre x uniform rule (``debye.at_points``), for any pupil;
* ``"grid"`` -- the sphere summed on a Cartesian grid of direction sines (``debye.on_grid``), the matrix Fourier transform.

A ``PupilMode`` is one coherent pupil field ``(J_x, J_y)(x_p, y_p)`` in sqrt(W/mm^2) (irradiance units, in air on the pupil);
the modes' irradiances add. ``scale`` doubles every node count (``scale = 2``) for the convergence check.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bessel import symmetric_field
from .debye import at_points, axial_flux, on_grid
from .objective import Objective, sphere_field
from .quadrature import CartesianGrid, cartesian_grid, gauss_legendre, polar_rule

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
PupilFunction = Callable[[Array, Array], tuple[CArray, CArray]]
RadialFunction = Callable[[Array], CArray]

BESSEL = "bessel"
DIRECT = "direct"
GRID = "grid"


@dataclass(frozen=True)
class PupilMode:
    """One coherent pupil field. ``field(x_p, y_p) -> (J_x, J_y)`` with its phase; for a rotationally symmetric mode also
    ``radial(h)`` (the complex scalar amplitude) and its uniform ``jones`` vector, so ``field = radial(h) * jones``."""

    field: PupilFunction
    radial: RadialFunction | None = None
    jones: tuple[complex, complex] | None = None


@dataclass(frozen=True)
class Sampling:
    theta_nodes: int
    phi_nodes: int
    grid_cells: int


class Focus:
    """The focal field of ``modes`` through ``objective`` (module docstring). ``theta_min``: the inner edge of an annular
    pupil (0 for a full one). ``with_phase=False`` drops the window's phase (its transmissions stay): with modes built without
    their pupil phase, the reference of a Strehl ratio."""

    def __init__(self, objective: Objective, modes: tuple[PupilMode, ...], sampling: Sampling, theta_min: float = 0.0,
                 with_phase: bool = True) -> None:
        self.objective, self.modes, self.sampling, self.theta_min = objective, modes, sampling, theta_min
        self.with_phase = with_phase
        self.symmetric = all(m.radial is not None and m.jones is not None for m in modes)
        self._polar: dict[int, tuple[Array, Array, Array, list[tuple[CArray, CArray]]]] = {}
        self._radial: dict[int, tuple[Array, Array, CArray, CArray, list[CArray]]] = {}
        self._grid: dict[int, tuple[CartesianGrid, list[tuple[CArray, CArray]]]] = {}

    # ------------------------------------------------------------------------------------------------ sphere samplings

    def _transfer(self, theta: Array) -> tuple[CArray, CArray, CArray]:
        tp, ts, phase = self.objective.transfer(theta)
        return tp, ts, (np.exp(1j * phase) if self.with_phase else np.ones(np.shape(theta), dtype=np.complex128))

    def _sphere(self, theta: Array, phi: Array, transmit: bool = True) -> list[tuple[CArray, CArray]]:
        h = self.objective.height_of_sine(np.sin(theta))
        xp, yp = h * np.cos(phi), h * np.sin(phi)
        if transmit:
            tp, ts, carrier = self._transfer(theta)
        else:
            tp = ts = carrier = np.ones(np.shape(theta), dtype=np.complex128)
        out = []
        for mode in self.modes:
            jx, jy = mode.field(xp, yp)
            out.append(sphere_field(theta, phi, jx * carrier, jy * carrier, tp, ts))
        return out

    def polar(self, scale: int = 1) -> tuple[Array, Array, Array, list[tuple[CArray, CArray]]]:
        if scale not in self._polar:
            rule = polar_rule(self.theta_min, self.objective.theta_max, self.sampling.theta_nodes * scale, self.sampling.phi_nodes * scale)
            self._polar[scale] = (rule.theta, rule.phi, rule.weight, self._sphere(rule.theta, rule.phi))
        return self._polar[scale]

    def radial(self, scale: int = 1) -> tuple[Array, Array, CArray, CArray, list[CArray]]:
        if scale not in self._radial:
            theta, weight = gauss_legendre(self.sampling.theta_nodes * scale, self.theta_min, self.objective.theta_max)
            tp, ts, carrier = self._transfer(theta)
            h = self.objective.height_of_sine(np.sin(theta))
            amplitudes = [mode.radial(h) * np.sqrt(np.cos(theta)) * carrier for mode in self.modes]  # type: ignore[misc]
            self._radial[scale] = (theta, weight, tp, ts, amplitudes)
        return self._radial[scale]

    def grid(self, scale: int = 1) -> tuple[CartesianGrid, list[tuple[CArray, CArray]]]:
        if scale not in self._grid:
            cells = self.sampling.grid_cells * scale
            grid = cartesian_grid(np.sin(self.theta_min), self.objective.sine_max, cells)
            self._grid[scale] = (grid, self._sphere(grid.theta, grid.phi))
        return self._grid[scale]

    # ------------------------------------------------------------------------------------------------ fields

    def fields(self, points: Array, method: str, scale: int = 1) -> list[tuple[CArray, CArray]]:
        """``[(E, H), ...]`` (3, P) per mode at ``points`` (3, P) in mm."""

        if method == BESSEL:
            theta, weight, tp, ts, amplitudes = self.radial(scale)
            return [symmetric_field(self.objective, theta, weight, amplitude, tp, ts, mode.jones, points)  # type: ignore[arg-type]
                    for mode, amplitude in zip(self.modes, amplitudes)]
        if method == DIRECT:
            theta, phi, weight, spheres = self.polar(scale)
            return [at_points(self.objective, theta, phi, weight, f, m, points) for f, m in spheres]
        grid, spheres = self.grid(scale)
        out = []
        for f, m in spheres:
            e = np.empty((len(points), points.shape[1]), dtype=np.complex128)
            h = np.empty_like(e)
            for z in np.unique(points[2]):
                chosen = points[2] == z
                e_part, h_part = self._grid_points(grid, f, m, points[0][chosen], points[1][chosen], float(z))
                e[:, chosen], h[:, chosen] = e_part, h_part
            out.append((e, h))
        return out

    def _grid_points(self, grid: CartesianGrid, field: CArray, magnetic: CArray, x: Array, y: Array, z: float) -> tuple[CArray, CArray]:
        k = self.objective.k
        axial = np.exp(1j * k * np.cos(grid.theta) * z) * grid.weight
        mx = np.exp(-1j * k * np.outer(x, grid.sines))
        my = np.exp(-1j * k * np.outer(y, grid.sines))
        out = []
        for values in (field, magnetic):
            out.append(np.stack([((my @ (values[c] * axial)) * mx).sum(axis=1) for c in range(len(values))]))
        return self.objective.prefactor * out[0], self.objective.prefactor * out[1]

    def irradiance(self, points: Array, method: str, scale: int = 1) -> tuple[Array, Array]:
        """``(|E|^2, S_z)`` (P,) at ``points``, summed over the modes, in W/mm^2."""

        total_e = np.zeros(points.shape[1])
        total_s = np.zeros(points.shape[1])
        for e, h in self.fields(points, method, scale):
            total_e = total_e + (np.abs(e) ** 2).sum(axis=0)
            total_s = total_s + axial_flux(e, h)
        return total_e, total_s

    def plane(self, x: Array, y: Array, z: float, scale: int = 1) -> list[tuple[CArray, CArray]]:
        """``[(E, H), ...]`` (3, len(y), len(x)) per mode on a rectangular grid of the plane ``z`` (``grid`` method)."""

        grid, spheres = self.grid(scale)
        return [on_grid(self.objective, grid, f, m, x, y, z) for f, m in spheres]

    # ------------------------------------------------------------------------------------------------ powers

    def sphere_power(self, scale: int = 1, transmit: bool = True) -> float:
        """Power through the reference sphere (with ``transmit``: behind the window), by the polar rule."""

        theta, phi, weight, spheres = self.polar(scale)
        if not transmit:
            spheres = self._sphere(theta, phi, transmit=False)
        radius = self.objective.sphere_radius_mm
        return radius * radius * math.fsum(float((np.abs(f) ** 2).sum(axis=0) @ weight) for f, _ in spheres)

    def longitudinal_fraction(self, scale: int = 1) -> float:
        """The share of ``E_z`` in the integral of ``|E|^2`` over any transverse plane, from the sphere by Parseval's theorem:
        ``Integral |E_c|^2 dx dy = f'^2 Integral |a_c|^2 / cos(theta) dOmega`` (independent of the plane)."""

        theta, _phi, weight, spheres = self.polar(scale)
        per_area = weight / np.cos(theta)
        total = math.fsum(float((np.abs(f) ** 2).sum(axis=0) @ per_area) for f, _ in spheres)
        axial = math.fsum(float((np.abs(f[len(f) - 1]) ** 2) @ per_area) for f, _ in spheres)
        return axial / total

    def plane_integrals(self, z: float, oversampling: int, centre: tuple[float, float], scale: int = 1) -> dict[str, float]:
        """Over one period of the grid transform on the plane ``z``: the axial flux, the integrals of ``|E|^2`` and ``|E_z|^2``,
        and the flux the sphere samples carry (equal to the first by Parseval's theorem)."""

        grid, spheres = self.grid(scale)
        count = grid.sines.size * oversampling
        pitch = 2.0 * np.pi / (self.objective.k * grid.step * count)
        x = centre[0] + (np.arange(count) - count // 2) * pitch
        y = centre[1] + (np.arange(count) - count // 2) * pitch
        flux = electric = longitudinal = sampled = 0.0
        radius = self.objective.sphere_radius_mm
        for f, m in spheres:
            e, h = on_grid(self.objective, grid, f, m, x, y, z)
            flux += float(axial_flux(e, h).sum()) * pitch * pitch
            electric += float((np.abs(e) ** 2).sum()) * pitch * pitch
            longitudinal += float((np.abs(e[len(e) - 1]) ** 2).sum()) * pitch * pitch
            sampled += radius * radius * float((axial_flux(f, m) * grid.weight * grid.weight).sum()) / (grid.step * grid.step)
        return {"flux_W": flux, "electric_integral": electric, "longitudinal_integral": longitudinal, "sampled_W": sampled, "pitch_mm": pitch,
                "period_mm": pitch * count}


__all__ = ["BESSEL", "DIRECT", "Focus", "GRID", "PupilMode", "Sampling"]
