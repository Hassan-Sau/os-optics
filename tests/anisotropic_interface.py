"""An exact plane-wave solver for a plane interface between two linear, non-magnetic media (isotropic or uniaxial), written
for the tests independently of the Engine's crystal model (``optics/polarization/uniaxial.py`` splits a field per wave with
isotropic Fresnel coefficients; this solves Maxwell's boundary conditions).

Units: wave vectors ``K`` in units of the vacuum wavenumber, magnetic fields ``H = K x E`` (``Z0 H`` of a plane wave, mu = 1),
so a wave's time-averaged power flux across the interface is proportional to ``Re(E x conj(H)) . N``.

Method (Berreman's first-order form): with the interface normal ``N`` as local ``z`` and the tangential wave vector ``xi``
along local ``x``, a plane wave ``exp(i k0 (xi x + q z))`` obeys ``q psi = D psi`` for its tangential fields
``psi = (E_x, E_y, H_x, H_y)``; ``D`` (derived from ``K x E = H``, ``K x H = -eps E`` by eliminating ``E_z`` and ``H_z``) is

    [ -xi e_zx/e_zz,               -xi e_zy/e_zz,                 0,  1 - xi^2/e_zz ]
    [  0,                           0,                            -1,  0             ]
    [ -e_yx + e_yz e_zx/e_zz,       xi^2 - e_yy + e_yz e_zy/e_zz,  0,  xi e_yz/e_zz  ]
    [  e_xx - e_xz e_zx/e_zz,       e_xy - e_xz e_zy/e_zz,         0, -xi e_xz/e_zz  ]

Its four eigenvalues are the normal wave-vector components of the medium's four waves for that ``xi`` (two leaving along
``+N``: real ``q`` with positive flux, or ``Im q > 0``). At the interface ``psi`` is continuous: the incident wave plus the
two reflected waves equals the two transmitted waves (a 4 x 4 linear solve). Waves of one medium whose ``q`` coincide
(an isotropic medium; a crystal along its optic axis) are one wave: their fields add.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Two eigenvalues closer than this are one (degenerate) wave.
SAME_Q = 1e-9
#: A wave of |Im q| below this propagates.
PROPAGATING = 1e-12


def uniaxial_eps(n_o: float, n_e: float, axis) -> np.ndarray:
    c = np.asarray(axis, dtype=float)
    c = c / np.linalg.norm(c)
    return n_o * n_o * np.eye(3) + (n_e * n_e - n_o * n_o) * np.outer(c, c)


def isotropic_eps(n: float) -> np.ndarray:
    return n * n * np.eye(3)


@dataclass(frozen=True)
class PlaneWave:
    """A plane wave: wave vector ``K`` (real for a propagating wave), complex field ``E`` (3D) and ``H = K x E``."""

    K: np.ndarray
    E: np.ndarray

    @property
    def H(self) -> np.ndarray:
        return np.cross(self.K, self.E)

    def flux(self, normal) -> float:
        """Time-averaged power flux across a plane of unit ``normal`` (common factor dropped)."""

        return float(np.real(np.dot(np.cross(self.E, np.conj(self.H)), np.asarray(normal, dtype=float))))

    def ray(self) -> np.ndarray:
        """The unit Poynting (ray) direction."""

        s = np.real(np.cross(self.E, np.conj(self.H)))
        return s / np.linalg.norm(s)


def _frame(k_in: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Rows: local x (along the tangential wave vector, or any direction across ``normal`` at normal incidence), y, z = normal."""

    kt = k_in - np.dot(k_in, normal) * normal
    if np.linalg.norm(kt) > 1e-14:
        x = kt / np.linalg.norm(kt)
    else:
        trial = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = trial - np.dot(trial, normal) * normal
        x = x / np.linalg.norm(x)
    return np.array([x, np.cross(normal, x), normal])


def _berreman(eps: np.ndarray, xi: float) -> np.ndarray:
    e = eps
    zz = e[2, 2]
    return np.array([
        [-xi * e[2, 0] / zz, -xi * e[2, 1] / zz, 0.0, 1.0 - xi * xi / zz],
        [0.0, 0.0, -1.0, 0.0],
        [-e[1, 0] + e[1, 2] * e[2, 0] / zz, xi * xi - e[1, 1] + e[1, 2] * e[2, 1] / zz, 0.0, xi * e[1, 2] / zz],
        [e[0, 0] - e[0, 2] * e[2, 0] / zz, e[0, 1] - e[0, 2] * e[2, 1] / zz, 0.0, -xi * e[0, 2] / zz],
    ], dtype=complex)


def _full_field(eps: np.ndarray, xi: float, q: complex, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(K, E)`` in local coordinates from the tangential fields ``psi``."""

    ex, ey, _hx, hy = psi
    ez = (-xi * hy - eps[2, 0] * ex - eps[2, 1] * ey) / eps[2, 2]
    return np.array([xi, 0.0, q], dtype=complex), np.array([ex, ey, ez], dtype=complex)


def _psi(k_local: np.ndarray, e_local: np.ndarray) -> np.ndarray:
    h = np.cross(k_local, e_local)
    return np.array([e_local[0], e_local[1], h[0], h[1]], dtype=complex)


def _modes(eps: np.ndarray, xi: float) -> tuple[list[tuple[complex, np.ndarray]], list[tuple[complex, np.ndarray]]]:
    """The medium's waves for ``xi``: ``(forward, backward)``, each a list of ``(q, psi)``."""

    q_all, vectors = np.linalg.eig(_berreman(eps, xi))
    forward, backward = [], []
    for k in range(4):
        q, psi = complex(q_all[k]), vectors[:, k]
        k_loc, e_loc = _full_field(eps, xi, q, psi)
        psi = _psi(k_loc, e_loc)  # (H from K x E: consistent with the flux below)
        if abs(q.imag) < PROPAGATING:
            q = complex(q.real, 0.0)
            k_loc, e_loc = _full_field(eps, xi, q, psi)
            flux = float(np.real(np.cross(e_loc, np.conj(np.cross(k_loc, e_loc)))[2]))
            (forward if flux > 0.0 else backward).append((q, psi))
        else:
            (forward if q.imag > 0.0 else backward).append((q, psi))
    assert len(forward) == 2 and len(backward) == 2, (forward, backward)
    return forward, backward


def _group(waves: list[tuple[complex, np.ndarray, complex]]) -> list[tuple[complex, np.ndarray]]:
    """Waves of (nearly) one ``q`` added into one: ``(q, psi * amplitude)`` summed."""

    out: list[tuple[complex, np.ndarray]] = []
    for q, psi, amplitude in waves:
        for k, (q0, psi0) in enumerate(out):
            if abs(q - q0) < SAME_Q:
                out[k] = (q0, psi0 + amplitude * psi)
                break
        else:
            out.append((q, amplitude * psi))
    return out


@dataclass(frozen=True)
class Interface:
    """The result at one face: the transmitted and reflected propagating waves (global frame) and the fluxes across it."""

    incident: PlaneWave
    transmitted: tuple[PlaneWave, ...]
    reflected: tuple[PlaneWave, ...]
    normal: np.ndarray

    def share(self, wave: PlaneWave) -> float:
        """The wave's power as a share of the incident power (the ratio of the fluxes across the face)."""

        return abs(wave.flux(self.normal)) / self.incident.flux(self.normal)

    def balance(self) -> float:
        """Transmitted plus reflected minus incident power, as a share of the incident power (0: conserved)."""

        return sum(self.share(w) for w in (*self.transmitted, *self.reflected)) - 1.0


def cross(incident: PlaneWave, normal, eps_from: np.ndarray, eps_to: np.ndarray) -> Interface:
    """The plane wave ``incident`` (in the medium of ``eps_from``) meets a plane face of unit ``normal`` (pointing into the
    medium of ``eps_to``): the exact transmitted and reflected waves (evanescent ones dropped; they carry no flux)."""

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    k_in = np.real(incident.K)
    assert np.dot(k_in, n) > 0.0
    rot = _frame(k_in, n)
    xi = float(np.dot(rot[0], k_in))
    e_from, e_to = rot @ eps_from @ rot.T, rot @ eps_to @ rot.T
    _fwd1, back1 = _modes(e_from, xi)
    fwd2, _back2 = _modes(e_to, xi)
    psi_in = _psi(rot @ incident.K, rot @ incident.E)
    a = np.column_stack([-back1[0][1], -back1[1][1], fwd2[0][1], fwd2[1][1]])
    amp = np.linalg.solve(a, psi_in)

    def waves(eps_local, modes, amplitudes) -> tuple[PlaneWave, ...]:
        out = []
        for q, psi in _group([(q, psi, c) for (q, psi), c in zip(modes, amplitudes)]):
            if abs(q.imag) >= PROPAGATING:
                continue
            k_loc, e_loc = _full_field(eps_local, xi, q, psi)
            out.append(PlaneWave(rot.T @ np.real(k_loc), rot.T @ e_loc))
        return tuple(out)

    return Interface(incident, waves(e_to, fwd2, amp[2:]), waves(e_from, back1, amp[:2]), n)


def entering(direction, field, n_from: float) -> PlaneWave:
    """A plane wave of unit ``direction`` in an isotropic medium of index ``n_from`` with complex ``field`` (its part across
    ``direction``)."""

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    e = np.asarray(field, dtype=complex)
    e = e - np.dot(d, e) * d
    return PlaneWave(n_from * d, e)


__all__ = ["Interface", "PlaneWave", "cross", "entering", "isotropic_eps", "uniaxial_eps"]
