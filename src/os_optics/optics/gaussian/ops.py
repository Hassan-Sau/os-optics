"""ABCD / q-parameter operations for x/y-independent Gaussian beams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...errors import ErrorCode, Status
from ...math.exact_trig import atan2, cos_sin, incidence_deg
from ...math.vectors import dot, incidence_s_axis, matvec, normalize, rotation_matrices_between
from .state import GState, Limit, check_divergence

Array = NDArray[np.float64]


def propagate(state: GState, length_mm: float) -> GState:
    """Free-space propagation: ``q' = q + L`` (physical length in the local medium)."""
    if not state.valid:
        return state
    return state.with_(qx=state.qx + length_mm, qy=state.qy + length_mm)


def _lens_like(q: complex, power: float) -> complex:
    """``1/q' = 1/q - P``  ->  ``q' = q / (1 - P q)`` (never singular because Im q > 0)."""
    return q / (1.0 - power * q)


def _interface(q: complex, a: float, k: float) -> complex:
    """``1/q' = a/q - K``  ->  ``q' = q / (a - K q)``."""
    return q / (a - k * q)


def _no_state(state: GState, code: ErrorCode, text: str) -> GState:
    """No first-order state exists behind this step: a valid state takes the code; one that is already not valid keeps its
    own reason."""
    return state.invalidated(code, text) if state.valid else state


def _aoi_limit(state: GState, aoi_deg: float, text: str, cfg: NumericalConfig) -> GState:
    """The angle-of-incidence limit of a focusing step (``gaussian_max_aoi_deg``): an accuracy limit."""
    limit = Limit("gaussian_max_aoi_deg", "aoi_deg", "deg", aoi_deg, cfg.gaussian_max_aoi_deg, ErrorCode.GAUSSIAN_LARGE_AOI)
    return state.limited(limit, text) if aoi_deg > cfg.gaussian_max_aoi_deg else state.within(limit, cfg)


def _strength_limit(state: GState, w_max: float, radius_mm: float, exceeded: bool, text: str, cfg: NumericalConfig) -> GState:
    """The beam radius against the surface's radius of curvature (``gaussian_max_w_over_radius``): an accuracy limit."""
    limit = Limit("gaussian_max_w_over_radius", "beam_radius_over_surface_radius", "1", w_max / radius_mm if radius_mm > 0.0 else math.inf,
                  cfg.gaussian_max_w_over_radius, ErrorCode.GAUSSIAN_SURFACE_TOO_STRONG)
    return state.limited(limit, text) if exceeded else state.within(limit, cfg)


def _transport_frame(state: GState, d_in: Array, d_out: Array, cfg: NumericalConfig) -> tuple[Array, Array]:
    rot = rotation_matrices_between(d_in[None, :], d_out[None, :], cfg.antiparallel_tol, state.bx[None, :])[0]
    return matvec(rot, state.bx), matvec(rot, state.by)


def apply_thin_lens(state: GState, d_in: Array, d_out: Array, cos_aoi: float, focal_length_mm: float, cfg: NumericalConfig, *,
                    normal: Array) -> GState:
    """Ideal thin lens: both axes see ``P = cos(theta)/f`` (slope-form lens is stigmatic); frame transported. ``normal`` is
    the lens axis: the angle of incidence the accuracy check states is ``incidence_deg`` of the beam on it (0 at normal
    incidence, not the rounding of its cosine)."""
    if not state.valid:
        return state
    state = _aoi_limit(state, float(incidence_deg(cos_aoi, d_in, normal)), "thin-lens incidence angle exceeds the Gaussian model limit",
                       cfg)
    if not state.valid:
        return state
    power = cos_aoi / focal_length_mm
    bx, by = _transport_frame(state, d_in, d_out, cfg)
    out = state.with_(qx=_lens_like(state.qx, power), qy=_lens_like(state.qy, power), bx=bx, by=by)
    return check_divergence(out, cfg)


def apply_flat(state: GState, d_in: Array, d_out: Array, cfg: NumericalConfig) -> GState:
    """Undeviated or pure-transport element (polarizer, waveplate, splitter transmission, aperture): ``q`` unchanged."""
    if not state.valid:
        return state
    bx, by = _transport_frame(state, d_in, d_out, cfg)
    return state.with_(bx=bx, by=by)


def _plane_of_incidence(d_in: Array, n_or: Array, zero_tol: float) -> tuple[Array | None, Array | None]:
    """``(t, s)`` of the plane of incidence, or ``(None, None)`` within ``NORMAL_INCIDENCE_SIN`` of normal incidence (no
    resolved plane: the normal-incidence rules apply, the same for a design and for that design turned rigidly)."""
    s, resolved = incidence_s_axis(d_in, n_or, np.zeros(3), zero_tol)
    if not bool(resolved[0]):
        return None, None
    s_in = s[0]
    t_in = np.cross(d_in, s_in)
    return t_in, s_in


def _is_round(state: GState, cfg: NumericalConfig) -> bool:
    """Both axes carry the same ``q`` and ``M^2`` (within ``gaussian_round_tol``): the beam has no preferred transverse axes."""
    round_q = abs(state.qx - state.qy) <= cfg.gaussian_round_tol * max(abs(state.qx), abs(state.qy))
    round_m2 = abs(state.m2x - state.m2y) <= cfg.gaussian_round_tol * max(state.m2x, state.m2y)
    return round_q and round_m2


def _align(state: GState, t_in: Array, s_in: Array, cfg: NumericalConfig) -> tuple[bool | None, GState | None]:
    """Decide whether ``qx`` is tangential (True), sagittal (False), or invalidate (twisted astigmatism).

    Returns ``(x_is_t, invalid_state)``; ``invalid_state`` is set when the model is left."""
    c, sn = float(dot(state.bx, t_in)), float(dot(state.bx, s_in))
    if abs(c * sn) <= cfg.gaussian_astig_align_tol:
        return abs(c) >= abs(sn), None
    if _is_round(state, cfg):
        return True, None  # round beam: the frame can be rotated onto (t, s) without changing the state
    return None, _no_state(
        state, ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM,
        "astigmatic beam axes are not aligned with the plane of incidence: twisted astigmatism is outside the x/y-independent model",
    )


@dataclass(frozen=True)
class LocalCurvature:
    """Principal normal curvatures (1/mm) of a surface at the chief-ray hit and their unit tangent directions (global).

    Signed like ``radius_signed_mm`` of ``apply_curved_or_flat_interface``: refraction: positive when the centre of
    curvature lies ahead of the surface along the ray; mirror / TIR: positive when concave (centre on the incident side)."""

    k1: float
    k2: float
    dir1: Array
    dir2: Array
    # the surface has no curvature the model can use (an axicon): the Gaussian state becomes invalid with this code / text
    undefined: tuple[ErrorCode, str] | None = None


def _interface_powers(qt: complex, qs: complex, *, refract: bool, kt: float, ks: float, n1: float, n2: float, cos_i: float,
                      cos_t: float) -> tuple[complex, complex]:
    """Coddington step for the tangential / sagittal ``q`` with normal curvatures ``kt`` / ``ks`` along the tangential /
    sagittal surface directions: refraction ``1/q'_t = (n1 cos^2 i / n2 cos^2 t)/q_t - (n2 cos t - n1 cos i) kt / (n2 cos^2 t)``,
    ``1/q'_s = (n1/n2)/q_s - (n2 cos t - n1 cos i) ks / n2``; mirror / TIR ``P_t = 2 kt / cos i``, ``P_s = 2 ks cos i``."""
    if refract:
        delta = n2 * cos_t - n1 * cos_i
        a_t = (n1 * (cos_i * cos_i)) / (n2 * (cos_t * cos_t))
        return _interface(qt, a_t, delta * kt / (n2 * (cos_t * cos_t))), _interface(qs, n1 / n2, delta * ks / n2)
    return _lens_like(qt, 2.0 * kt / cos_i), _lens_like(qs, 2.0 * ks * cos_i)


def apply_local_curvature(
    state: GState,
    *,
    kind: str,  # "refract" | "mirror"
    d_in: Array,
    d_out: Array,
    n_or: Array,
    cos_i: float,
    n1: float,
    n2: float,
    cos_t: float,
    tir: bool,
    curvature: LocalCurvature,
    cfg: NumericalConfig,
) -> GState:
    """Refraction (Coddington), reflection or TIR at a surface with local principal curvatures ``k1``, ``k2``.

    The x/y-independent beam needs the surface's principal directions on the beam's tangential / sagittal axes: at oblique
    incidence they must coincide with the plane-of-incidence directions on the surface (sagittal ``s`` and tangential
    ``n x s``) within ``gaussian_astig_align_tol``, unless the surface is umbilic there (``|k1 - k2| <=
    gaussian_umbilic_rel_tol max|k|``); at normal incidence the beam axes must lie on the principal directions (a round
    beam is rotated onto them). Otherwise the result is ``GAUSSIAN_TWISTED_ASTIGMATISM``. The beam-axis alignment rule
    with the plane of incidence applies as for spheres; ``GAUSSIAN_LARGE_AOI`` and ``GAUSSIAN_SURFACE_TOO_STRONG`` (beam
    radius against the smallest local radius of curvature ``1 / max|k|``) as well. A surface without usable curvature
    (``curvature.undefined``: an axicon) invalidates the state with its code."""
    if curvature.undefined is not None:
        return _no_state(state, *curvature.undefined)
    k1, k2 = curvature.k1, curvature.k2
    k_max = max(abs(k1), abs(k2))
    if k_max > 0.0:
        aoi_deg = float(incidence_deg(cos_i, d_in, n_or))
        state = _aoi_limit(state, aoi_deg, f"AOI {aoi_deg:.3g} deg on a curved surface exceeds the Gaussian model limit", cfg)
        if not state.valid:
            return state
        w_max = max(state.radius("x"), state.radius("y"))
        state = _strength_limit(
            state, w_max, 1.0 / k_max, not w_max * k_max <= cfg.gaussian_max_w_over_radius,
            f"beam radius {w_max:.4g} mm is not small against the local surface radius of curvature {1.0 / k_max:.4g} mm", cfg)
        if not state.valid:
            return state
    umbilic = abs(k1 - k2) <= cfg.gaussian_umbilic_rel_tol * k_max
    twisted = ("the surface's principal curvature directions are not aligned with the beam's astigmatic axes or the plane of "
               "incidence: twisted astigmatism is outside the x/y-independent model")
    t_in, s_in = _plane_of_incidence(d_in, n_or, cfg.zero_vector_tol)
    if t_in is None or s_in is None:  # normal incidence: the principal directions are the only preferred axes
        if umbilic:
            kx = ky = 0.5 * (k1 + k2)
        else:
            p1, ok = normalize((curvature.dir1 - float(dot(curvature.dir1, d_in)) * d_in)[None, :], cfg.zero_vector_tol)
            if not bool(ok[0]):  # pragma: no cover - a tangent direction is perpendicular to the (normal) incidence
                return _no_state(state, ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, twisted)
            p1v = p1[0]
            c1, c2 = float(dot(state.bx, p1v)), float(dot(state.bx, np.cross(d_in, p1v)))
            if abs(c1 * c2) <= cfg.gaussian_astig_align_tol:
                x_on_1 = abs(c1) >= abs(c2)
            elif _is_round(state, cfg):
                state = state.with_(bx=p1v, by=np.cross(d_in, p1v))  # a round beam has no preferred axes
                x_on_1 = True
            else:
                return _no_state(state, ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, twisted)
            kx, ky = (k1, k2) if x_on_1 else (k2, k1)
        bx, by = _transport_frame(state, d_in, d_out, cfg)
        refract = kind == "refract" and not tir
        if refract and cos_t <= math.cos(math.radians(cfg.grazing_fail_aoi_deg)):  # pragma: no cover - normal incidence
            return _no_state(state, ErrorCode.GAUSSIAN_LARGE_AOI, "refracted ray is at grazing exit: Gaussian tangential magnification diverges")
        qx2, qy2 = _interface_powers(state.qx, state.qy, refract=refract, kt=kx, ks=ky, n1=n1, n2=n2, cos_i=cos_i, cos_t=cos_t)
        return check_divergence(state.with_(qx=qx2, qy=qy2, n=n2 if refract else n1, bx=bx, by=by), cfg)
    if umbilic:
        kt = ks = 0.5 * (k1 + k2)
    else:
        tangential = np.cross(n_or, s_in)
        c1, c2 = float(dot(curvature.dir1, s_in)), float(dot(curvature.dir1, tangential))
        if abs(c1 * c2) > cfg.gaussian_astig_align_tol:
            return _no_state(state, ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, twisted)
        ks, kt = (k1, k2) if abs(c1) >= abs(c2) else (k2, k1)
    x_is_t, invalid = _align(state, t_in, s_in, cfg)
    if invalid is not None:
        return invalid
    s_out, _ = normalize((s_in - float(dot(s_in, d_out)) * d_out)[None, :], cfg.zero_vector_tol)
    s_out_v = s_out[0]
    t_out = np.cross(d_out, s_out_v)
    bx, by = (t_out, np.cross(d_out, t_out)) if x_is_t else (s_out_v, np.cross(d_out, s_out_v))
    qt, qs = (state.qx, state.qy) if x_is_t else (state.qy, state.qx)
    refract = kind == "refract" and not tir
    if refract and cos_t <= math.cos(math.radians(cfg.grazing_fail_aoi_deg)):
        return _no_state(state, ErrorCode.GAUSSIAN_LARGE_AOI, "refracted ray is at grazing exit: Gaussian tangential magnification diverges")
    qt2, qs2 = _interface_powers(qt, qs, refract=refract, kt=kt, ks=ks, n1=n1, n2=n2, cos_i=cos_i, cos_t=cos_t)
    qx2, qy2 = (qt2, qs2) if x_is_t else (qs2, qt2)
    return check_divergence(state.with_(qx=qx2, qy=qy2, n=n2 if refract else n1, bx=bx, by=by), cfg)


def apply_curved_or_flat_interface(
    state: GState,
    *,
    kind: str,  # "refract" | "mirror"
    d_in: Array,
    d_out: Array,
    n_or: Array,
    cos_i: float,
    n1: float,
    n2: float,
    cos_t: float,
    tir: bool,
    radius_signed_mm: float | None,
    cfg: NumericalConfig,
    curvature: LocalCurvature | None = None,
) -> GState:
    """Refraction (Coddington), reflection or TIR at a flat (``radius_signed_mm=None``) or spherical surface, or at a
    general smooth surface described by its local principal curvatures at the chief-ray hit (``curvature``, see
    ``apply_local_curvature``).

    ``radius_signed_mm``: refraction: ``>0`` if the centre of curvature lies ahead of the surface along the ray;
    mirror/TIR: ``>0`` if concave (centre on the incident side). Tangential/sagittal split by the plane of incidence.
    """
    if not state.valid:
        return state
    if curvature is not None:
        return apply_local_curvature(state, kind=kind, d_in=d_in, d_out=d_out, n_or=n_or, cos_i=cos_i, n1=n1, n2=n2, cos_t=cos_t,
                                     tir=tir, curvature=curvature, cfg=cfg)
    curved = radius_signed_mm is not None
    aoi_deg = float(incidence_deg(cos_i, d_in, n_or))
    if curved:
        state = _aoi_limit(state, aoi_deg, f"AOI {aoi_deg:.3g} deg on a curved surface exceeds the Gaussian model limit", cfg)
        if not state.valid:
            return state
        w_max = max(state.radius("x"), state.radius("y"))
        state = _strength_limit(
            state, w_max, abs(radius_signed_mm), w_max > cfg.gaussian_max_w_over_radius * abs(radius_signed_mm),  # type: ignore[arg-type]
            f"beam radius {w_max:.4g} mm is not small against the surface radius {abs(radius_signed_mm):.4g} mm", cfg)  # type: ignore[arg-type]
        if not state.valid:
            return state
    t_in, s_in = _plane_of_incidence(d_in, n_or, cfg.zero_vector_tol)
    if t_in is None or s_in is None:  # normal incidence: no preferred axes; both axes are equivalent
        x_is_t: bool | None = True
        invalid: GState | None = None
        bx, by = _transport_frame(state, d_in, d_out, cfg)
    else:
        x_is_t, invalid = _align(state, t_in, s_in, cfg)
        if invalid is not None:
            return invalid
        s_out, ok = normalize((s_in - float(dot(s_in, d_out)) * d_out)[None, :], cfg.zero_vector_tol)
        s_out_v = s_out[0]
        t_out = np.cross(d_out, s_out_v)
        bx, by = (t_out, np.cross(d_out, t_out)) if x_is_t else (s_out_v, np.cross(d_out, s_out_v))
    qt, qs = (state.qx, state.qy) if x_is_t else (state.qy, state.qx)

    if kind == "refract" and not tir:
        if cos_t <= math.cos(math.radians(cfg.grazing_fail_aoi_deg)):
            return _no_state(state, ErrorCode.GAUSSIAN_LARGE_AOI, "refracted ray is at grazing exit: Gaussian tangential magnification diverges")
        delta = n2 * cos_t - n1 * cos_i
        a_t = (n1 * (cos_i * cos_i)) / (n2 * (cos_t * cos_t))
        a_s = n1 / n2
        if curved:
            r = radius_signed_mm  # type: ignore[assignment]
            k_t, k_s = delta / (n2 * (cos_t * cos_t) * r), delta / (n2 * r)
        else:
            k_t = k_s = 0.0
        qt2, qs2 = _interface(qt, a_t, k_t), _interface(qs, a_s, k_s)
        n_new = n2
    else:  # mirror or total internal reflection: medium unchanged, effective mirror powers
        if curved:
            r_c = radius_signed_mm  # type: ignore[assignment]
            p_t, p_s = 2.0 / (r_c * cos_i), 2.0 * cos_i / r_c
        else:
            p_t = p_s = 0.0
        qt2, qs2 = _lens_like(qt, p_t), _lens_like(qs, p_s)
        n_new = n1
    qx2, qy2 = (qt2, qs2) if x_is_t else (qs2, qt2)
    out = state.with_(qx=qx2, qy=qy2, n=n_new, bx=bx, by=by)
    return check_divergence(out, cfg)


def apply_grating(state: GState, d_in: Array, d_out: Array, normal: Array, n1: float, n2: float, cfg: NumericalConfig) -> GState:
    """One order of a plane grating (``optics/surfaces/grating.py``). The grating adds a constant tangential wavevector, so a
    transverse offset ``x`` maps by ``M``: along ``d_in`` onto the plane, then across ``d_out``; and with the plane's points
    and tangential wavevector offsets preserved, slopes map by ``(n1 / n2) M^-T``: no focusing, ``q' = (n2 / n1) s^2 q`` along
    each singular direction of ``M``, ``s`` its stretch. In the plane of incidence ``s = cos(exit) / cos(incidence)``, and across
    it 1, as for a flat refraction; in a conical mount the singular directions are neither. An astigmatic beam must have its
    axes on them (else ``GAUSSIAN_TWISTED_ASTIGMATISM``); a round beam is rotated onto them."""

    if not state.valid:
        return state
    if abs(float(dot(d_out, normal))) <= math.cos(math.radians(cfg.grazing_fail_aoi_deg)):
        return _no_state(state, ErrorCode.GAUSSIAN_LARGE_AOI, "the diffraction order leaves near grazing: its Gaussian tangential magnification diverges")
    along = float(dot(d_in, normal))

    def image(x: Array) -> Array:
        on_plane = x - (float(dot(x, normal)) / along) * d_in
        return on_plane - float(dot(on_plane, d_out)) * d_out

    bx, by = state.bx, state.by
    mx, my = image(bx), image(by)
    xx, yy, xy = float(dot(mx, mx)), float(dot(my, my)), float(dot(mx, my))
    if abs(xy) > cfg.gaussian_astig_align_tol * math.sqrt(xx * yy):
        if not _is_round(state, cfg):
            return _no_state(
                state, ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM,
                "astigmatic beam axes are not aligned with the principal directions of the grating's anamorphic map: twisted astigmatism "
                "is outside the x/y-independent model")
        c, s = cos_sin(np.array(0.5 * float(atan2(np.array(2.0 * xy), np.array(xx - yy)))))
        c, s = float(c), float(s)
        bx, by = c * bx + s * by, c * by - s * bx
        mx, my = image(bx), image(by)
        xx, yy = float(dot(mx, mx)), float(dot(my, my))
    scale = n2 / n1
    x_out, ok = normalize(mx[None, :], cfg.zero_vector_tol)
    if not bool(ok[0]):  # pragma: no cover - the order does not run along the plane
        return _no_state(state, ErrorCode.GAUSSIAN_LARGE_AOI, "the diffraction order leaves along the plane")
    out = state.with_(qx=state.qx * (scale * xx), qy=state.qy * (scale * yy), n=n2, bx=x_out[0], by=np.cross(d_out, x_out[0]))
    return check_divergence(out, cfg)


def switch_medium_normal(state: GState, n2: float) -> GState:
    """Normal-incidence flat refraction shortcut (used in tests / helpers): ``q' = (n2/n1) q``."""
    if not state.valid:
        return state
    ratio = n2 / state.n
    return state.with_(qx=state.qx * ratio, qy=state.qy * ratio, n=n2)


def mark_truncation(state: GState, loss_fraction: float, cfg: NumericalConfig) -> tuple[GState, ErrorCode | None]:
    """Apply the truncation applicability rule for a power loss fraction at an aperture. Beyond the scope limit the state
    leaves the model's scope with the failed check (``GState.limited``)."""
    if not state.valid:
        return state, None
    limit = Limit("gaussian_clip_scope_fraction", "truncated_power_fraction", "1", loss_fraction, cfg.gaussian_clip_scope_fraction,
                  ErrorCode.GAUSSIAN_SEVERE_TRUNCATION)
    if loss_fraction > cfg.gaussian_clip_scope_fraction:
        return (
            state.limited(limit, f"aperture truncates {loss_fraction:.3g} of the power; the truncated beam is not Gaussian"),
            ErrorCode.GAUSSIAN_SEVERE_TRUNCATION,
        )
    state = state.within(limit, cfg)
    if loss_fraction > cfg.gaussian_clip_warn_fraction:
        new = state.with_(status=Status.WARNING if state.status is Status.PASS else state.status,
                          reason=state.reason or ErrorCode.GAUSSIAN_TRUNCATION,
                          reason_text=state.reason_text or f"aperture truncates {loss_fraction:.3g} of the power")
        return new, ErrorCode.GAUSSIAN_TRUNCATION
    return state, None
