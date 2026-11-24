"""Uniaxial birefringent crystals: the ordinary and extraordinary plane waves at a plane face, their ray (energy) directions,
polarizations, power shares, optical and group paths, and the Gaussian beam's diffraction in them.

Conventions. Wave vectors ``K`` are in units of the vacuum wavenumber ``k0``: ``|K|`` is the wave's phase index along its wave
normal ``k = K / |K|``. ``c`` is the crystal's unit optic axis (scene frame), ``A = 1 / n_o^2``, ``B = 1 / n_e^2``.

* Ordinary wave: ``|K| = n_o``; its ray runs along ``k``; its field is along ``c x K``.
* Extraordinary wave: ``K^T M K = 1`` with ``M = B I + (A - B) c c^T`` (``1/n(theta)^2 = cos^2(theta)/n_o^2 +
  sin^2(theta)/n_e^2``, ``theta`` between ``k`` and ``c``); its ray (Poynting vector) runs along ``S = M K / |M K|``, the normal
  of the index surface, walked off ``k`` by ``rho`` in the plane of ``k`` and ``c``; its field is along ``S x (c x K)``.
* At a plane face of normal ``N`` the tangential part ``K_t`` of the incident wave vector is kept: an outgoing wave is
  ``K_t + q N`` with ``q`` from ``|K|^2 = n^2`` (an isotropic medium, the ordinary wave) or from the quadratic
  ``(A - B)(a + q b)^2 + B (|K_t|^2 + q^2) = 1`` (``a = K_t . c``, ``b = N . c``) of the extraordinary wave; of its two roots
  the larger leaves along ``+N`` (``S . N > 0``: the transmitted wave), the smaller along ``-N`` (the reflected wave). No real
  root: that wave is evanescent (total internal reflection of that wave).
* A beam in a crystal travels along its ray direction; its optical path per mm of ray is ``K . S = n(theta) cos(rho)`` (the
  phase index along the ray; every ray of the beam has its own), and its group index along the ray is ``n_g(theta) cos(rho)``
  with ``n_g(theta) = n - lambda dn/dlambda`` at fixed ``theta`` (the ray path is stationary, so the change of ``k`` with
  wavelength does not enter to first order); its group-delay dispersion per mm is ``lambda^3/(2 pi) d^2n/dlambda^2 cos(rho)``
  at the same fixed geometry, as for an isotropic medium (``optics/materials/group.py``).
* Power at a face (the interface's own anisotropic coefficients are not solved): the incident field, taken transverse to its
  wave normal, is split on an orthonormal pair ``(u_o, u_e)`` -- ``u_o`` the ordinary outgoing wave's field projected
  transverse to the incident wave normal -- and each part passes with the isotropic Fresnel transmittance (or the face
  coating's) at the outgoing wave's index, for its own share of s and p. Reflection takes the rest: the face's power is
  conserved exactly. The reflected field is the sum of the parts' Fresnel reflections, scaled to that power, and in a crystal
  is split again on the reflected waves' pair. At normal incidence this is the exact plane-wave result for any orientation of
  the optic axis (the tangential fields of each eigenwave then obey the isotropic boundary conditions at its own index); at
  oblique incidence it is exact where every crystal's optic axis is perpendicular to the plane of incidence (s-polarized light
  then meets only ``n_e``, p-polarized light only ``n_o``: two isotropic interfaces; ``decoupled``), and an approximation
  elsewhere, which ``tracer.py`` reports (``BIREFRINGENT_INTERFACE_APPROXIMATE``).
* Along the optic axis (the wave normal within ``AXIAL_SIN_TOL`` of it) the two waves are one: both have the index ``n_o``,
  the same wave vector and ray, and every field transverse to the axis is an eigenwave (it is perpendicular to the axis).
  The light crosses such a face as at an isotropic medium of index ``n_o`` (from an isotropic medium the exact plane-wave
  result; from a crystal the face rule above) and travels on as one wave carrying its whole field (``AXIAL``; ``RaySegment.crystal_mode`` ``ORDINARY``): it is not split on a
  basis the crystal does not define. Within the band the waves' indices differ by at most ``|n_o^3 (1/n_e^2 - 1/n_o^2)| /
  2 * AXIAL_SIN_TOL^2`` and the extraordinary ray walks off by at most ``|n_o^2 / n_e^2 - 1| * AXIAL_SIN_TOL``, so the
  coherent sum of the two waves the general rule gives there equals the one wave to about ``1e-12`` of the field.
* Gaussian beam: an ordinary beam diffracts as in an isotropic medium of index ``n_o``. An extraordinary beam's transverse
  wave-vector spread ``kappa`` (perpendicular to ``S``) adds ``-kappa^T M kappa / (2 |M K|)`` to its longitudinal wave
  vector: it diffracts like an isotropic medium of index ``1/D`` along each of its two field axes, ``D_1 = B / |M K|``
  across the plane of ``k`` and ``c`` and ``D_2 = (B + (A - B) |c_perp|^2) / |M K|`` in it. The tracer carries one Gaussian
  state of index ``2 / (D_1 + D_2)`` and the error ``|D_1 - D_2| / 2`` per mm of reduced path (``gaussian_index``). A beam
  along the optic axis has parts of both kinds (each plane-wave part of its spread splits across and along its plane through
  the axis): ``D_o = 1 / n_o`` and ``D_e = n_o / n_e^2``, carried the same way.

Every operation is ``+ - * /`` and ``sqrt`` (no libm transcendental, no fused complex products): the same bits in every
runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.vectors import dot, normalize
from ...units import NM_TO_MM

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

#: (1/n^2)'' = -2 n''/n^3 + 6 n'^2/n^4: the coefficient of its second term.
_SECOND_OF_INVERSE_SQUARE = 6.0
#: (f^(-1/2))'' = 3/4 f^(-5/2) f'^2 - 1/2 f^(-3/2) f'': the coefficient of its first term.
_SECOND_OF_INVERSE_ROOT = 0.75

ORDINARY = "o"
EXTRAORDINARY = "e"
#: The one wave of light along the optic axis (module docstring): its field is any field transverse to the axis.
AXIAL = "a"
#: The contract names of the waves (``RaySegment.crystal_mode``): along the axis every field is perpendicular to it, so the one
#: wave there is ordinary.
MODE_NAMES = {ORDINARY: "ORDINARY", EXTRAORDINARY: "EXTRAORDINARY", AXIAL: "ORDINARY"}
#: A crystal wave whose wave normal lies within this sine of the angle of the optic axis travels along it (module docstring).
AXIAL_SIN_TOL = 1e-6


@dataclass(frozen=True)
class Crystal:
    """A uniaxial crystal at one vacuum wavelength: its two indices, unit optic axis (scene frame) and the ``(n_g, k2)`` of each
    index (``None``: not available there)."""

    n_o: float
    n_e: float
    axis: Array
    group_o: tuple[float, float] | None
    group_e: tuple[float, float] | None
    wavelength_nm: float

    @property
    def a(self) -> float:
        return 1.0 / (self.n_o * self.n_o)

    @property
    def b(self) -> float:
        return 1.0 / (self.n_e * self.n_e)


@dataclass(frozen=True)
class Waves:
    """One outgoing plane wave per ray (rows): ``mode`` (``"o"``, ``"e"``, or ``None`` in an isotropic medium), wave vector
    ``K``, unit ray direction ``S``, ``ok`` (propagating), the beam's transverse basis ``(e1, e2)`` (``e1 x e2 = S``; a crystal
    wave's field is along ``e1`` (ordinary) or ``e2`` (extraordinary)) and the phase index along the ray ``K . S``."""

    mode: str | None
    K: Array
    S: Array
    ok: NDArray[np.bool_]
    e1: Array
    e2: Array
    ray_index: Array

    def field_axis(self) -> Array:
        """The unit field direction of a crystal wave (``e1`` ordinary, ``e2`` extraordinary)."""
        return self.e2 if self.mode == EXTRAORDINARY else self.e1


def _rows(v: Array, n: int) -> Array:
    return np.broadcast_to(np.asarray(v, dtype=np.float64), (n, 3))


def _scale(v: Array, f: Array) -> Array:
    return v * f[:, None]


def outgoing(kt: Array, normal: Array, crystal: Crystal | None, n_iso: float | None, mode: str | None, forward: bool,
             fallback: Array, zero_tol: float) -> Waves:
    """The outgoing wave of tangential wave vector ``kt`` (rows) at a face of unit normal ``normal`` (rows, toward the medium
    beyond the face): in an isotropic medium of index ``n_iso`` (``crystal`` None), or the ``mode`` wave of ``crystal``;
    ``forward``: the wave leaving into the medium beyond the face (``S . N > 0``), else back into the medium before it
    (module docstring). ``fallback`` (rows): the field axis where the wave's own is not defined (an isotropic wave; a crystal
    wave along the optic axis, where both waves have index ``n_o``)."""

    n = kt.shape[0]
    kt2 = dot(kt, kt)
    sign = 1.0 if forward else -1.0
    if crystal is None or mode == ORDINARY:
        index = float(n_iso) if crystal is None else crystal.n_o  # type: ignore[arg-type]
        q2 = index * index - kt2
        ok = q2 > 0.0
        q = np.sqrt(np.where(ok, q2, 0.0))
        k_vec = kt + _scale(normal, sign * q)
        s_dir, _ = normalize(k_vec, zero_tol)
    else:
        c = crystal.axis
        a_c, b_c = dot(kt, c), dot(normal, c)
        diff = crystal.a - crystal.b
        qa = diff * b_c * b_c + crystal.b
        qb = 2.0 * diff * a_c * b_c
        qc = diff * a_c * a_c + crystal.b * kt2 - 1.0
        disc = qb * qb - 4.0 * qa * qc
        ok = disc > 0.0
        root = np.sqrt(np.where(ok, disc, 0.0))
        # the two roots without cancellation: q1 = (-qb - sign(qb) root) / (2 qa), q2 = qc / (qa q1)
        lead = -0.5 * (qb + np.where(qb >= 0.0, root, -root))
        safe = np.where(lead != 0.0, lead, 1.0)
        first = lead / qa
        second = np.where(lead != 0.0, qc / safe, -first)
        high, low = np.maximum(first, second), np.minimum(first, second)
        q = high if forward else low
        k_vec = kt + _scale(normal, q)
        mk = _scale(k_vec, np.full(n, crystal.b)) + _scale(_rows(c, n), diff * dot(k_vec, c))
        s_dir, ok_s = normalize(mk, zero_tol)
        ok = ok & ok_s
    if crystal is None:
        e1, ok_f = normalize(fallback - _scale(s_dir, dot(fallback, s_dir)), zero_tol)
    else:
        own, own_ok = normalize(np.cross(_rows(crystal.axis, n), k_vec), zero_tol)
        backup, _ = normalize(fallback - _scale(s_dir, dot(fallback, s_dir)), zero_tol)
        e1 = np.where(own_ok[:, None], own, backup)
        # (the ordinary field is perpendicular to the ray: exactly so after rounding)
        e1, ok_f = normalize(e1 - _scale(s_dir, dot(e1, s_dir)), zero_tol)
    e2 = np.cross(s_dir, e1)
    return Waves(mode if crystal is not None else None, k_vec, s_dir, ok & ok_f, e1, e2, dot(k_vec, s_dir))


def axial(crystal: Crystal, waves: Waves) -> bool:
    """Whether the chief ray's wave of ``waves`` (an outgoing wave of ``crystal``) travels along the optic axis: its wave normal
    within ``AXIAL_SIN_TOL`` of the axis (module docstring)."""

    if not bool(waves.ok[0]):
        return False
    k = waves.K[0]
    c = crystal.axis
    cross = (float(c[1]) * float(k[2]) - float(c[2]) * float(k[1]), float(c[2]) * float(k[0]) - float(c[0]) * float(k[2]),
             float(c[0]) * float(k[1]) - float(c[1]) * float(k[0]))
    size = float(k[0]) * float(k[0]) + float(k[1]) * float(k[1]) + float(k[2]) * float(k[2])
    return cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2] <= AXIAL_SIN_TOL * AXIAL_SIN_TOL * size


def axial_wave(kt: Array, normal: Array, crystal: Crystal, forward: bool, fallback: Array, zero_tol: float) -> Waves:
    """The one outgoing wave of light along ``crystal``'s optic axis (``axial``): the wave of index ``n_o`` of tangential wave
    vector ``kt``, its basis ``(e1, e2)`` built from ``fallback`` as in an isotropic medium (its field is not restricted to
    either axis)."""

    waves = outgoing(kt, normal, None, crystal.n_o, None, forward, fallback, zero_tol)
    return Waves(AXIAL, waves.K, waves.S, waves.ok, waves.e1, waves.e2, waves.ray_index)


def gaussian_index(crystal: Crystal | None, waves: Waves, n_iso: float | None) -> tuple[float, float]:
    """``(index, error)`` of the chief ray's wave for the Gaussian beam (module docstring): the index of the isotropic medium it
    diffracts like (the mean of its two axes' ``D``) and the error ``|D_1 - D_2| / 2`` of the reduced q-parameter per mm of ray
    that one index makes (0 for an isotropic medium and an ordinary wave; along the optic axis ``D_o = 1 / n_o`` and
    ``D_e = n_o / n_e^2``)."""

    if crystal is None:
        return float(n_iso), 0.0  # type: ignore[arg-type]
    if waves.mode == ORDINARY:
        return crystal.n_o, 0.0
    if waves.mode == AXIAL:
        d_o, d_e = 1.0 / crystal.n_o, crystal.n_o * crystal.b
        return 2.0 / (d_o + d_e), abs(d_o - d_e) / 2.0
    k_vec, s_dir, c = waves.K[0], waves.S[0], crystal.axis
    diff = crystal.a - crystal.b
    mk = crystal.b * k_vec + (diff * float(dot(k_vec, c))) * c
    g = math.sqrt(float(dot(mk, mk)))
    along = float(dot(c, s_dir))
    d1 = crystal.b / g
    d2 = (crystal.b + diff * (1.0 - along * along)) / g
    return 2.0 / (d1 + d2), abs(d1 - d2) / 2.0


def ray_group(crystal: Crystal, waves: Waves) -> tuple[float, float] | None:
    """``(n_g, k2)`` along the chief ray of a crystal wave (module docstring): the ordinary index's own, or for an
    extraordinary wave ``n_g(theta) cos(rho)`` and ``lambda^3 / (2 pi) d^2n(theta)/dlambda^2 cos(rho)``; ``None`` when a group
    index of the crystal is not available at its wavelength."""

    if waves.mode in (ORDINARY, AXIAL):  # (along the axis n(theta) = n_o at every wavelength)
        return crystal.group_o
    if crystal.group_o is None or crystal.group_e is None:
        return None
    k_vec, s_dir = waves.K[0], waves.S[0]
    size = math.sqrt(float(dot(k_vec, k_vec)))
    cos_rho = float(dot(k_vec, s_dir)) / size
    along = float(dot(k_vec, crystal.axis)) / size
    cos2 = along * along
    sin2 = 1.0 - cos2
    lam_mm = crystal.wavelength_nm * NM_TO_MM
    # per index: n, lambda n' = n - n_g, lambda^2 n'' = 2 pi k2 / lambda
    terms = []
    for n_x, (ng_x, k2_x) in ((crystal.n_o, crystal.group_o), (crystal.n_e, crystal.group_e)):
        terms.append((n_x, n_x - ng_x, 2.0 * math.pi * k2_x / lam_mm))
    (n_o, p_o, s_o), (n_e, p_e, s_e) = terms
    # f = cos^2 / n_o^2 + sin^2 / n_e^2 and its lambda-derivatives (lambda f', lambda^2 f'') from those of 1/n^2:
    # lambda (1/n^2)' = -2 p / n^3, lambda^2 (1/n^2)'' = -2 s / n^3 + 6 p^2 / n^4
    f = cos2 / (n_o * n_o) + sin2 / (n_e * n_e)
    f1 = cos2 * (-2.0 * p_o / (n_o * n_o * n_o)) + sin2 * (-2.0 * p_e / (n_e * n_e * n_e))
    f2 = (cos2 * (-2.0 * s_o / (n_o * n_o * n_o) + _SECOND_OF_INVERSE_SQUARE * p_o * p_o / (n_o * n_o * n_o * n_o))
          + sin2 * (-2.0 * s_e / (n_e * n_e * n_e) + _SECOND_OF_INVERSE_SQUARE * p_e * p_e / (n_e * n_e * n_e * n_e)))
    n = 1.0 / math.sqrt(f)
    n3 = n * n * n
    # n = f^(-1/2): n' = -1/2 n^3 f', n'' = 3/4 n^5 f'^2 - 1/2 n^3 f''
    first = -0.5 * n3 * f1  # lambda dn/dlambda
    second = _SECOND_OF_INVERSE_ROOT * n3 * n * n * f1 * f1 - 0.5 * n3 * f2  # lambda^2 d^2n/dlambda^2
    return (n - first) * cos_rho, lam_mm * second / (2.0 * math.pi) * cos_rho


def transverse_field(field: CArray, k_hat: Array, s: Array, zero_tol: float) -> tuple[CArray, CArray, Array]:
    """The incident field ``field`` (rows, 3D) as its ``(s, p)`` components transverse to the wave normal ``k_hat``
    (``p = k_hat x s``), scaled to keep its power (an extraordinary beam's field is transverse to its ray, not to its wave
    normal). Returns ``(j_s, j_p, p)``."""

    p = np.cross(k_hat, s)
    j_s = field[:, 0] * s[:, 0] + field[:, 1] * s[:, 1] + field[:, 2] * s[:, 2]
    j_p = field[:, 0] * p[:, 0] + field[:, 1] * p[:, 1] + field[:, 2] * p[:, 2]
    total = (field.real * field.real + field.imag * field.imag)
    power = total[:, 0] + total[:, 1] + total[:, 2]
    kept = (j_s.real * j_s.real + j_s.imag * j_s.imag) + (j_p.real * j_p.real + j_p.imag * j_p.imag)
    ok = kept > zero_tol * power
    scale = np.sqrt(np.where(ok, power, 1.0) / np.where(ok, kept, 1.0))
    return j_s * scale, j_p * scale, p


def split_axes(field_axis: Array, k_hat: Array, fallback: Array, zero_tol: float) -> tuple[Array, Array]:
    """The orthonormal pair ``(u_o, u_e)`` transverse to the wave normal ``k_hat`` (rows) that splits a field between two crystal
    waves: ``u_o`` the ordinary wave's field ``field_axis`` made transverse (``fallback`` where it is along ``k_hat``),
    ``u_e = k_hat x u_o``."""

    u_o, ok = normalize(field_axis - _scale(k_hat, dot(field_axis, k_hat)), zero_tol)
    backup, _ = normalize(fallback - _scale(k_hat, dot(fallback, k_hat)), zero_tol)
    u_o = np.where(ok[:, None], u_o, backup)
    return u_o, np.cross(k_hat, u_o)


def along(j_s: CArray, j_p: CArray, s: Array, p: Array, u: Array) -> tuple[CArray, Array, Array]:
    """The component of the field ``j_s s + j_p p`` along the unit transverse axis ``u`` and ``u``'s s and p parts."""

    c_s, c_p = dot(u, s), dot(u, p)
    return j_s * c_s + j_p * c_p, c_s, c_p


def oriented(waves: Waves, u: Array) -> Waves:
    """``waves`` with its basis turned by half a turn where the wave's field axis points against ``u`` (so a field part along
    ``u`` maps onto the field axis with its own sign)."""

    flip = dot(waves.field_axis(), u) < 0.0
    sign = np.where(flip, -1.0, 1.0)
    return Waves(waves.mode, waves.K, waves.S, waves.ok, _scale(waves.e1, sign), _scale(waves.e2, sign), waves.ray_index)


def mode_jones(waves: Waves, amplitude: CArray) -> CArray:
    """The Jones vectors (rows, in ``(waves.e1, waves.e2)``) of a crystal wave of complex ``amplitude`` along its field axis."""

    zero = np.zeros_like(amplitude)
    return np.stack([zero, amplitude], axis=1) if waves.mode == EXTRAORDINARY else np.stack([amplitude, zero], axis=1)


#: How close to 1 ``|c . s|`` is for an optic axis perpendicular to the plane of incidence (``decoupled``).
AXIS_ALIGNMENT_TOL = 1e-9


def decoupled(axes: list[Array], s: Array) -> bool:
    """Whether the split at a face is exact at any incidence: every crystal's optic axis on it is perpendicular to the plane of
    incidence (along ``s``), so s-polarized light meets only ``n_e`` and p-polarized light only ``n_o``, each an isotropic
    interface."""

    return all(abs(float(dot(axis, s))) >= 1.0 - AXIS_ALIGNMENT_TOL for axis in axes)


__all__ = ["AXIAL", "AXIAL_SIN_TOL", "AXIS_ALIGNMENT_TOL", "Crystal", "EXTRAORDINARY", "MODE_NAMES", "ORDINARY", "Waves", "along", "axial",
           "axial_wave", "decoupled", "gaussian_index", "mode_jones", "oriented", "outgoing", "ray_group", "split_axes", "transverse_field"]
