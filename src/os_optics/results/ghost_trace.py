"""A ghost-scoped trace of a run's request (``results.ghosts``): every transmitting interface's reflection followed up to a
declared bounce order, with a ledger of everything that was not followed.

The run's own Engine request (``engine_request.json``) is lowered again. At every refracting interface the design does not
already split (``trace_reflection`` off: an ordinary lens or window face), the reflected part -- the bare Fresnel reflection, or
the coating's stated reflectance -- becomes a traced branch (a ghost bounce) while the branch that meets it has made fewer than
``max_bounce_order`` such bounces. A reflection the design itself follows (a splitter coating, a mirror, total internal
reflection) is the design's light, not a ghost bounce. Nothing else changes: the primary light is traced exactly as in the run,
the untraced-light probes and the interference analysis are not run, and the run's files, result and hashes are untouched.

Every part of a line's light that leaves the traced beams without a known fate is entered in ``GhostTracer.unfollowed`` with
its power and an upper bound on it:

* ``BOUNCE_LIMIT``: a ghost reflection of a branch that already made ``max_bounce_order`` bounces;
* ``BELOW_FLOOR``: a ghost reflection at or below the listing floor (``relative_power_floor`` x the line's input), not traced;
* ``PATH_BUDGET``: light the trace budget (``max_paths``, ``max_trace_steps``, ``max_interactions_per_path``) left untraced,
  a whole source line included;
* ``OUT_OF_MODEL``: a path that ended outside the model (a grazing ray, a coating or material outside its data, a beam whose
  every pupil sample an aperture cut, evanescent light behind a complex-transmission element), or a source line the trace
  could not launch;
* ``SCATTERED``: light a rough surface or a scattering medium scatters (``SURFACE_SCATTER``, ``BULK_SCATTER``): the ghost
  trace does not follow scattered light;
* ``UNMODELLED_RESPONSE``: light an element's response model does not deliver into a traced beam (a tabulated lens's
  ``1 - T``, a grating's orders not followed, a beam whose polarization the response leaves open);
* ``UNTRACED_<kind>``: other light the trace does not follow (light passing beside an element, a coating's transmission, a
  reflection of an element the design does not split).

A passive system never raises a beam's power, so a piece's power where it leaves the traced beams bounds what it, and every
ghost it makes in turn, brings anywhere. Each piece's bound (``Unfollowed.power_upper_W``) also undoes the single-line
approximations on its way that could have lowered its power:

* where a power rests on the geometric pupil-sample basis (an aperture cut the pupil rays of a beam without a valid Gaussian
  state, ``Tracer._geometric_cut``), the light past a cut is at most the light that met it;
* at every ghost surface (a transmitting interface whose reflection the ghost trace follows) the trace takes the chief ray's
  reflectance and transmittance for the whole beam; the bound takes instead an upper bound on the beam's mean power
  reflectance and transmittance there (``face_bounds``). A paraxial Gaussian beam's light, in position across the beam and
  direction together, is a Gaussian: its plane-wave spectrum has the 1/e^2 half-width ``theta_0`` (the far-field half angle,
  the same at every distance) and its share within ``rho`` whitened radii of the centre is ``1 - exp(-2 rho^2) (1 + 2 rho^2)``
  (in direction alone ``1 - exp(-2 rho^2)``). Light within ``rho`` meets a plane within ``rho theta_0`` of the chief ray's
  angle of incidence; on a sphere of radius ``R`` it meets the surface within ``rho_x = rho w + 2 rho theta_0 R`` of the chief
  ray's line (``w`` the beam radius, ``2 R`` the most the hit point lies along that line from the chief ray's), where the
  normal's angle to that line has its sine within ``rho_x / R`` of the chief ray's. The mean over the beam is bounded shell by
  shell (``SHELL_RADII``): each shell's share times the largest coefficient over the angles its outer radius reaches
  (``validation.resonance``: a bare interface's s reflectance at the widest angle, never below the p one, and 1 from the
  critical angle on; for the transmittance 1 less the smallest p reflectance, 0 where Brewster's angle is inside; a coating's
  largest reflectance and transmittance over its table), every polarization, and the light beyond the last shell counted
  whole. A beam without a valid Gaussian state, an annular beam and a sag surface have no stated distribution of angles:
  there the bound reads the mean over the beam's live pupil rays (equal-power samples) of each ray's larger power reflectance
  and transmittance at its own angle of incidence -- a sampled value, not a proof -- and every piece of light on its way
  after that face carries an unproven bound (``Unfollowed.proven``); a crystal face keeps its chief ray's coefficients,
  unproven too.

Every other interaction acts on the whole beam with its chief ray's coefficients, as in the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..config import NumericalConfig
from ..math.exact_trig import exp_float
from ..math.vectors import correctly_rounded_sum, dot
from ..optics.materials.library import MaterialLibrary
from ..optics.rays.tracer import (
    SINK_BULK_SCATTER,
    SINK_INTERACTION_LIMIT,
    SINK_PATH_LIMIT,
    SINK_SOURCE_NOT_TRACED,
    SINK_SURFACE_SCATTER,
    SINK_TERMINATED_OUT_OF_MODEL,
    SINK_UNMODELLED_RESPONSE,
    SINK_UNTRACED_REFLECTION,
    UNKNOWN_FATE_SINKS,
    Tracer,
    _Packet,
)
from ..scene.model import Scene, SurfaceG
from ..validation.resonance import NEGATIVE, POSITIVE, _asin, _coefficients, _cos_sin, _critical

#: The lineage events of a transmitting interface's reflection.
GHOST_EVENTS = frozenset({"fresnel_reflection", "coating_reflection"})
#: Reasons a piece of light was not followed (module docstring).
BOUNCE_LIMIT = "BOUNCE_LIMIT"
BELOW_FLOOR = "BELOW_FLOOR"
PATH_BUDGET = "PATH_BUDGET"
OUT_OF_MODEL = "OUT_OF_MODEL"
SCATTERED = "SCATTERED"
UNMODELLED_RESPONSE = "UNMODELLED_RESPONSE"
UNTRACED_PREFIX = "UNTRACED_"
#: The reason each sink of unknown fate (``UNKNOWN_FATE_SINKS``) is entered with.
SINK_REASONS = {
    SINK_PATH_LIMIT: PATH_BUDGET, SINK_INTERACTION_LIMIT: PATH_BUDGET, SINK_SOURCE_NOT_TRACED: OUT_OF_MODEL,
    SINK_TERMINATED_OUT_OF_MODEL: OUT_OF_MODEL, SINK_UNMODELLED_RESPONSE: UNMODELLED_RESPONSE, SINK_SURFACE_SCATTER: SCATTERED,
    SINK_BULK_SCATTER: SCATTERED,
}
#: The outer radii (whitened beam radii) of the shells a face's bound sums over (module docstring); beyond the last, all the light
#: counts (its share: ``exp(-72) (1 + 72)`` on a sphere).
SHELL_RADII = tuple(0.25 * k for k in range(1, 25))
_HALF_PI = 0.5 * math.pi


def ghost_surfaces(scene: Scene) -> frozenset[str]:
    """The surfaces whose reflection is a ghost bounce: refracting interfaces whose reflection the design does not follow."""

    return frozenset(s.surface_id for s in scene.surfaces if s.interaction.kind == "refract" and not s.interaction.trace_reflection)


def ghost_bounces(lineage: tuple[str, ...] | list[str], surfaces: frozenset[str]) -> tuple[str, ...]:
    """The ghost-bounce surfaces of a lineage, in order."""

    out = []
    for entry in lineage:
        surface_id, _, event = entry.rpartition(":")
        if event in GHOST_EVENTS and surface_id in surfaces:
            out.append(surface_id)
    return tuple(out)


@dataclass(frozen=True)
class Unfollowed:
    """One piece of a line's light that left the traced beams without a known fate (module docstring)."""

    source_id: str
    spectral_index: int
    path_id: str
    bounce_order: int
    reason: str
    surface_id: str | None
    power_W: float
    power_upper_W: float
    detail: str
    #: ``power_upper_W`` is a proven bound (``False``: a face on its way had only a sampled range of angles)
    proven: bool = True


@dataclass(frozen=True)
class FaceBound:
    """A ghost surface met by a packet (``face_bounds``): the power meeting it, upper bounds on the power reflectance and
    transmittance of the whole beam there, the range of angles of incidence (rad) they rest on (``None``: the chief ray's
    coefficients) and whether that range is proven (else: the angles of the beam's live pupil rays)."""

    surface_id: str
    path_id: str
    incident_W: float
    reflectance: float
    transmittance: float
    angles: tuple[float, float] | None
    proven: bool


def _rp_min(n1: float, n2: float, low: float, high: float) -> float:
    """The smallest p power reflectance of a bare interface from ``n1`` into ``n2`` over the angles of incidence [``low``,
    ``high``] below the critical angle: 0 where Brewster's angle (``tan = n2 / n1``) is inside, else the smaller of its ends
    (it falls to Brewster's angle and rises beyond)."""
    critical = _critical(n1, n2)
    if critical is not None:
        if low >= critical:
            return 1.0
        high = min(high, critical)
    c_lo, s_lo = _cos_sin(low)
    c_hi, s_hi = _cos_sin(high)
    if s_lo * n1 <= c_lo * n2 and s_hi * n1 >= c_hi * n2:
        return 0.0

    def rp(cos_i: float, sin_i: float) -> float:
        sin_t = n1 * sin_i / n2
        cos_t = math.sqrt(max(0.0, (1.0 - sin_t) * (1.0 + sin_t)))
        r = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        return r * r

    return min(rp(c_lo, s_lo), rp(c_hi, s_hi))


def _over(surf: SurfaceG, side: int, n1: float, n2: float, coating: object, library: MaterialLibrary, wavelength_nm: float,
          low: float, high: float) -> tuple[float, float]:
    """(largest power reflectance, largest power transmittance) of the interface over the angles [``low``, ``high``], every
    polarization (``validation.resonance``; a bare interface's transmittance: 1 less its smallest p reflectance)."""
    r_amp, t_amp = _coefficients(surf, side, low, high, library, wavelength_nm)
    return r_amp * r_amp, t_amp * t_amp if coating is not None else 1.0 - _rp_min(n1, n2, low, high)


def _enclosed(rho: float, sphere: bool) -> float:
    """The share of a Gaussian beam's light within ``rho`` whitened radii: in direction alone (a plane) or in position and
    direction together (a sphere)."""
    e = exp_float(-2.0 * rho * rho)
    return 1.0 - (e * (1.0 + 2.0 * rho * rho) if sphere else e)


def _sampled(pk: _Packet, surf: SurfaceG, h, side: int, n1: float, n2: float, coating: object, library: MaterialLibrary,  # type: ignore[no-untyped-def]
             ) -> tuple[float, float, tuple[float, float]]:
    """The mean over the beam's live pupil rays (the chief ray where none is live) of each ray's larger power reflectance and
    transmittance at its own angle of incidence, and the range of those angles: a sampled value (module docstring). A ray that
    missed the face, or that is totally reflected where the chief ray is not, does not take the interaction the beam takes."""
    rows = np.flatnonzero(pk.alive[pk.pupil_slice()]) + 1
    cos_i = np.minimum(np.abs(dot(pk.d[rows], h.normal[rows])), 1.0) if rows.size else np.zeros(0)
    cos_i = cos_i[np.isfinite(cos_i)]
    cos_i = cos_i[n1 * n1 * (1.0 - cos_i) * (1.0 + cos_i) < n2 * n2]
    if not cos_i.size:
        cos_i = np.array([min(abs(float(dot(pk.d[0], h.normal[0]))), 1.0)])
    c_hi, c_lo = float(np.max(cos_i)), float(np.min(cos_i))
    angles = (_asin(math.sqrt((1.0 - c_hi) * (1.0 + c_hi))), _asin(math.sqrt((1.0 - c_lo) * (1.0 + c_lo))))
    if coating is None:  # (a bare interface: the Fresnel coefficients of every ray at once)
        sin_t = n1 * np.sqrt((1.0 - cos_i) * (1.0 + cos_i)) / n2
        cos_t = np.sqrt(np.maximum(0.0, (1.0 - sin_t) * (1.0 + sin_t)))
        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        r_ray, t_ray = np.maximum(rs * rs, rp * rp).tolist(), (1.0 - np.minimum(rs * rs, rp * rp)).tolist()
    else:
        r_ray, t_ray = [], []
        for c in cos_i.tolist():
            angle = _asin(math.sqrt((1.0 - c) * (1.0 + c)))
            r_up, t_up = _over(surf, side, n1, n2, coating, library, pk.wavelength_nm, angle, angle)
            r_ray.append(r_up)
            t_ray.append(t_up)
    count = len(r_ray)
    return min(1.0, correctly_rounded_sum(r_ray) / count), min(1.0, correctly_rounded_sum(t_ray) / count), angles


def face_bounds(pk: _Packet, surf: SurfaceG, h, n1: float, n2: float, coating: object, library: MaterialLibrary,  # type: ignore[no-untyped-def]
                from_pos: bool) -> tuple[float, float, tuple[float, float], bool]:
    """Upper bounds on the mean power reflectance and transmittance of the whole beam ``pk`` meeting the transmitting interface
    ``surf`` (hits ``h``) from the medium of index ``n1`` into ``n2``, the range of angles of incidence they reach and whether
    they are proven (module docstring): over the paraxial Gaussian beam's light, or else sampled over its live pupil rays."""
    g = pk.gstate
    side = POSITIVE if from_pos else NEGATIVE
    if not g.valid or pk.chief_obstructed or surf.geometry not in ("plane", "sphere"):
        r, t, angles = _sampled(pk, surf, h, side, n1, n2, coating, library)
        return r, t, angles, False
    theta0 = max(g.far_field_half_angle("x"), g.far_field_half_angle("y"))
    w = max(g.radius("x"), g.radius("y"))
    c = min(abs(float(dot(pk.d[0], h.normal[0]))), 1.0)
    s = math.sqrt((1.0 - c) * (1.0 + c))
    sphere = surf.geometry == "sphere"
    r_total = t_total = 0.0
    r_open = t_open = True  # (once a coefficient reaches 1 at a radius it is 1 beyond: the rest counts whole)
    inner = r_inner = t_inner = 0.0
    low, high = _asin(s), _asin(s)
    for rho in SHELL_RADII:
        share = _enclosed(rho, sphere) - _enclosed(inner, sphere)
        spread = rho * theta0
        if sphere:
            reach = rho * w / abs(surf.radius) + 2.0 * spread
            s_lo, s_hi = max(0.0, s - reach), min(1.0, s + reach)
        else:
            s_lo = s_hi = s
        low, high = max(0.0, _asin(s_lo) - spread), min(_HALF_PI, _asin(s_hi) + spread)
        r_up, t_up = _over(surf, side, n1, n2, coating, library, pk.wavelength_nm, low, high)
        if r_open:
            r_open = r_up < 1.0
            if r_open:
                r_total += share * r_up
                r_inner = rho
        if t_open:
            t_open = t_up < 1.0
            if t_open:
                t_total += share * t_up
                t_inner = rho
        inner = rho
        if not (r_open or t_open):
            break
    return min(1.0, r_total + (1.0 - _enclosed(r_inner, sphere))), min(1.0, t_total + (1.0 - _enclosed(t_inner, sphere))), (low, high), True


class GhostTracer(Tracer):
    """The run's tracer with every transmitting interface's reflection followed up to ``max_order`` bounces (module docstring).

    ``floor_W``: per source line ``(source id, spectral index)``, the power at or below which a ghost reflection is not traced.
    After ``run()``: ``birth_W`` / ``birth_upper_W`` (path id -> its power where it was born, and a proven bound on it),
    ``unfollowed``, ``unknown_power`` (source id, spectral index and path id of the paths whose power is not known, so no bound
    covers their light), ``faces`` (every ghost surface a packet of known power met, with its bounds) and ``birth_sampling`` /
    ``end_sampling`` (path id -> the sampling resolution of its power where it was born / ended, on the geometric basis) and
    ``power_scope`` (path id -> the surfaces behind which its power is out of model scope)."""

    def __init__(self, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, *, max_order: int, floor_W: dict[tuple[str, int], float]) -> None:
        super().__init__(scene, library, cfg, include_pupil=True, keep_faint=True)
        self.surfaces = ghost_surfaces(scene)
        self.max_order = max_order
        self.floor_W = dict(floor_W)
        self.birth_W: dict[str, float] = {}
        self.birth_upper_W: dict[str, float] = {}
        self.unfollowed: list[Unfollowed] = []
        self.unknown_power: list[tuple[str, int, str]] = []
        # path id -> the factor that undoes the geometric pupil-sample cuts and the chief ray's coefficients at the ghost
        # surfaces on its way (the power past them is at most the power that met them, times the bounds of ``face_bounds``)
        self._undo: dict[str, float] = {}
        self._cut_incident_W: float | None = None
        self._face: FaceBound | None = None
        self.faces: list[FaceBound] = []
        # path id -> the sampling resolution (a fraction) of its power where it was born / where it ended, when that power
        # rests on the geometric pupil-sample basis (``_Packet.sampling``)
        self.birth_sampling: dict[str, float] = {}
        self.end_sampling: dict[str, float] = {}
        # the paths whose bound rests on a face with only a sampled range of angles (``FaceBound.proven``), and their branches
        self.unproven: set[str] = set()
        # path id -> the surfaces behind which its power is out of model scope (``_Packet.power_scope``), where it ended or left
        # a piece not followed
        self.power_scope: dict[str, set[str]] = {}

    def order(self, pk: _Packet) -> int:
        return len(ghost_bounces(pk.lineage, self.surfaces))

    def _note(self, pk: _Packet, reason: str, surface_id: str | None, power_W: float, upper_W: float, detail: str,
              reflected: bool = False) -> None:
        """Enter a piece of ``pk``'s light that is not followed; ``reflected``: a ghost reflection of it (one bounce more)."""
        if power_W <= 0.0 and upper_W <= 0.0:
            return
        if pk.power_scope:
            self.power_scope.setdefault(pk.path_id, set()).update(pk.power_scope)
        self.unfollowed.append(Unfollowed(pk.src.source_id, pk.spectral_index, pk.path_id, self.order(pk) + int(reflected), reason,
                                          surface_id, float(power_W), float(upper_W), detail, pk.path_id not in self.unproven))

    def _reflected_upper(self, pk: _Packet, surf: SurfaceG, power_W: float) -> float:
        """A bound on the power a ghost surface reflects off ``pk``'s beam where the trace takes ``power_W`` (the chief ray's
        reflectance): the face's reflectance bound times the power meeting it (``face_bounds``), never below ``power_W``."""
        face = self._face
        if face is None or face.path_id != pk.path_id or face.surface_id != surf.surface_id:
            return power_W
        return max(power_W, face.reflectance * face.incident_W)

    # ------------------------------------------------------------------------------------------------ the tracer's hooks
    def run(self):  # type: ignore[no-untyped-def]
        out = super().run()
        # (a source line the path budget did not launch, or the trace could not launch, is booked whole, outside ``_flow``)
        for flow in out.power_flows:
            if flow.path_id is None and flow.kind in UNKNOWN_FATE_SINKS:
                self.unfollowed.append(Unfollowed(flow.source_id, flow.spectral_index, flow.source_id, 0,
                                                  SINK_REASONS.get(flow.kind, OUT_OF_MODEL), None, float(flow.power_W),
                                                  float(flow.power_W), flow.detail))
        return out

    def _do_refract(self, pk, surf, h, inter):  # type: ignore[no-untyped-def]
        if surf.surface_id not in self.surfaces:
            return super()._do_refract(pk, surf, h, inter)
        if self.order(pk) < self.max_order:
            inter = replace(inter, trace_reflection=True)
        self._face = self._face_bound(pk, surf, h, inter)
        if self._face is not None and not self._face.proven:
            self.unproven.add(pk.path_id)
        try:
            term = super()._do_refract(pk, surf, h, inter)
        finally:
            face, self._face = self._face, None
        if face is not None:
            self.faces.append(face)
            out = pk.chief_power()
            if term is None and pk.lineage and pk.lineage[-1] == f"{surf.surface_id}:refract" and out > 0.0:
                # the light the face transmits is at most its transmittance bound times the light that met it
                factor = max(1.0, face.transmittance * face.incident_W / out)
                self._undo[pk.path_id] = self._undo.get(pk.path_id, 1.0) * factor
        return term

    def _face_bound(self, pk: _Packet, surf: SurfaceG, h, inter) -> FaceBound | None:  # type: ignore[no-untyped-def]
        """The bounds of the ghost surface ``surf`` for ``pk`` (``face_bounds``); ``None`` where the packet's power is not known
        or the trace ends its path at the face (no index on the far side)."""
        if pk.power_bound_W is not None:
            return None
        from_pos = float(dot(pk.d[0], h.normal[0])) < 0.0
        p_in = pk.chief_power()
        if pk.crystal is not None or self._crystal_face(surf):
            return FaceBound(surf.surface_id, pk.path_id, p_in, 0.0, 0.0, None, False)  # (the chief ray's coefficients)
        med_to = surf.medium_neg if from_pos else surf.medium_pos
        n2, issue = self.library.evaluator(med_to).index(pk.wavelength_nm, location=f"surface:{surf.surface_id}")
        if issue is not None or n2 is None:
            return None  # (the trace ends the path here: MATERIAL_RANGE)
        r, t, angles, proven = face_bounds(pk, surf, h, pk.n, float(n2), inter.coating, self.library, from_pos)
        return FaceBound(surf.surface_id, pk.path_id, p_in, r, t, angles, proven)

    def _spawn(self, pk, surf, kind, power_W):  # type: ignore[no-untyped-def]
        undo = self._undo.get(pk.path_id, 1.0)
        reflected = kind in GHOST_EVENTS and surf.surface_id in self.surfaces
        upper = self._reflected_upper(pk, surf, power_W) if reflected else power_W
        if reflected and pk.power_bound_W is None:
            floor = self.floor_W.get((pk.src.source_id, pk.spectral_index), 0.0)
            if power_W <= floor:
                self._note(pk, BELOW_FLOOR, surf.surface_id, power_W, upper * undo, kind, reflected=True)
                return None
        child = super()._spawn(pk, surf, kind, power_W)
        if child is not None:
            self.birth_W[child.path_id] = float(power_W)
            self.birth_upper_W[child.path_id] = float(upper) * undo
            self._undo[child.path_id] = undo * (upper / power_W if power_W > 0.0 else 1.0)
            if pk.sampling is not None:
                self.birth_sampling[child.path_id] = pk.sampling
            if pk.path_id in self.unproven:
                self.unproven.add(child.path_id)
        return child

    def _finish(self, pk, termination, surface_id):  # type: ignore[no-untyped-def]
        if pk.power_scope:
            self.power_scope.setdefault(pk.path_id, set()).update(pk.power_scope)
        if pk.sampling is not None:
            self.end_sampling[pk.path_id] = pk.sampling
        super()._finish(pk, termination, surface_id)

    def _flow(self, pk, kind, power_W, surface, detail):  # type: ignore[no-untyped-def]
        # (a bulk extinction is booked as its absorbed and scattered parts: the scattered part is light of unknown fate)
        for part, part_W in self._bulk_split(pk, kind, power_W):
            if part not in UNKNOWN_FATE_SINKS:
                continue
            if pk.power_bound_W is not None:
                self.unknown_power.append((pk.src.source_id, pk.spectral_index, pk.path_id))
                continue
            reflected = detail in GHOST_EVENTS
            upper = self._reflected_upper(pk, surface, part_W) if reflected and surface is not None else part_W
            self._note(pk, SINK_REASONS.get(part, OUT_OF_MODEL), None if surface is None else surface.surface_id, part_W,
                       upper * self._undo.get(pk.path_id, 1.0), str(detail), reflected=reflected)
        super()._flow(pk, kind, power_W, surface, detail)

    def _branch(self, pk, kind, surf, direction, gstate, power_W, **kwargs):  # type: ignore[no-untyped-def]
        if pk.power_bound_W is not None:
            self.unknown_power.append((pk.src.source_id, pk.spectral_index, pk.path_id))
        else:
            undo = self._undo.get(pk.path_id, 1.0)
            ghost = kind == SINK_UNTRACED_REFLECTION and surf.surface_id in self.surfaces
            if self._cut_incident_W is not None:
                # light cut on the geometric pupil-sample basis is at most the light that met the aperture
                upper = max(power_W, self._cut_incident_W) * undo
            else:
                upper = (self._reflected_upper(pk, surf, power_W) if ghost else power_W) * undo
            self._note(pk, BOUNCE_LIMIT if ghost else UNTRACED_PREFIX + kind, surf.surface_id, power_W, upper, kind,
                       reflected=kind == SINK_UNTRACED_REFLECTION)
        super()._branch(pk, kind, surf, direction, gstate, power_W, **kwargs)

    def _geometric_cut(self, pk, surf, h, cut, p_in):  # type: ignore[no-untyped-def]
        before = pk.chief_power()
        self._cut_incident_W = p_in
        try:
            term = super()._geometric_cut(pk, surf, h, cut, p_in)
        finally:
            self._cut_incident_W = None
        after = pk.chief_power()
        if term is None and after > 0.0:
            self._undo[pk.path_id] = self._undo.get(pk.path_id, 1.0) * (before / after)
        return term

    def _geometric_divergence(self, pk, surf, diverged):  # type: ignore[no-untyped-def]
        # (the samples that part from the beam take their share on the geometric basis: undone in the bounds as a cut)
        before = pk.chief_power()
        self._cut_incident_W = before
        try:
            term = super()._geometric_divergence(pk, surf, diverged)
        finally:
            self._cut_incident_W = None
        after = pk.chief_power()
        if term is None and after > 0.0 and after != before:
            self._undo[pk.path_id] = self._undo.get(pk.path_id, 1.0) * (before / after)
        return term


__all__ = ["BELOW_FLOOR", "BOUNCE_LIMIT", "SHELL_RADII", "GHOST_EVENTS", "FaceBound", "GhostTracer", "OUT_OF_MODEL", "PATH_BUDGET",
           "SCATTERED", "SINK_REASONS", "UNMODELLED_RESPONSE", "UNTRACED_PREFIX", "Unfollowed", "face_bounds", "ghost_bounces",
           "ghost_surfaces"]
