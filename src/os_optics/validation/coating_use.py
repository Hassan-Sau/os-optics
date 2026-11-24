"""How the light left out as negligible meets a coating from its dense side: its bounded angular extent, read per the record.

A coating record states the angle range its values hold for, in the medium ``aoi_reference`` names; ``AIR_EQUIVALENT``: the
angle in air with the same Snell invariant ``n sin(theta)`` (``CoatingModel.declared_validity``). The exposure bound's
reflectances (``validation/resonance.py``) take a coating's value for the light the trace leaves out, so that value has to
hold for that light. On a surface met from a medium denser than air, an air-equivalent record is therefore evaluated at the
left-out light's own bounded angles, per surface, side and wavelength (``CoatingUse``):

* the angles convert by ``asin(n sin(theta))`` and the coating's value is read at the converted angles;
* a use whose converted extent leaves the record's stated range (with the record's allowance: a single design angle the
  coating kind's tolerance, a range ``AOI_TOLERANCE_DEG``) or whose wavelength the record does not state is refused, and so is one reaching ``n sin(theta) >= 1``, above the
  critical angle, where no air-equivalent angle exists and nothing is extrapolated. The surface has no exposure upper bound,
  and neither have the surfaces of a pair the refusal leaves at a gain bound of 1, each refused surface counting (1, 1)
  (``CoatingUse.refused``; ``resonance.ReturnGains.withheld``); a plane's power bound and every other surface keep the
  record's stated value;
* the zero-to-critical range the pair and open-path geometry uses is not the light's extent and decides nothing here.

Air-side incidence of such a record, ``INCIDENT_MEDIUM`` records, uncoated surfaces and low-level coatings without a record
read as ``resonance._coefficients`` reads them.

Which light. Left-out light starts as a branch of a followed hit (``SurfaceHit``, traced or probe, every order above the
line's level): the reflection or the transmission whose power is at or below the level (``P R <= level``), or any branch of a
hit whose line the probes did not finish (``open_lines``: a branch their budget did not reach, or light whose fate they lost,
is one reflection beyond a followed hit too). A followed hit's own beam is not left out: its angles do not enter. A branch runs
on an open path to its next reflection, through refractions, plates and lenses; the extent holds every meeting of that run
with a coated dense side. The level is the line's own, ``negligible_power_fraction`` x its input: a deeper pass
(``validation/pipeline.py``) follows part of the light these uses cover, and changes neither them nor the factors. After its
next reflection left-out light is at most ``sqrt(R)`` of itself in field; the series reads that reflection at the value
established here and does not bound the directions it leaves in (the same class as the light total internal reflection holds,
which the exposure model states it leaves out).

The bound on one branch (angles in the medium the light is in). A Gaussian beam fills an ellipse in the phase space ``(x,
x')`` of each axis: with the 1/e^2 radius ``w``, the wavefront radius ``R_c`` and the far-field half angle ``theta_far``
(``GState``, ``M^2`` included), ``x = rho w cos t``, ``x' = rho (w / R_c cos t + (w_0 theta_far / w) sin t)``, and any linear
function of ``(x, x')`` over the beam's light is Gaussian: its share past ``K`` times its 1/e^2 half width is the same ``0.5
exp(-2 K^2)`` a transverse line has. ``K`` is ``decisions.envelope_sigmas`` of the followed hit: its beam carries at most an
eighth of the level past such a line, so the branch carries ``R`` of that. A ray's incidence on a surface differs from the
chief ray's, to first order, by such a function: ``x' - c x`` where the normal turns by ``c x`` across the footprint. So the
half width of the incidence about the chief ray's is

    D = K max_axis sqrt((w (1 / R_c - c))^2 + (w_0 theta_far / w)^2)

on a plane (``c = 0``) ``K theta_far``, the far-field half angle free propagation keeps; on a sphere of radius ``R`` ``c = +-1 /
R``, positive when its centre lies behind the surface along the ray (a beam diverging from its centre meets it at normal
incidence), and ``1 / (R cos i)`` along the plane of incidence, the larger counting. The extent is ``[asin(sin i - D), asin(sin i +
D)]``, which holds the first-order result. The branch's Gaussian state is carried from the followed hit by the same steps the
trace applies (``gaussian.ops``). It is not a sampled direction: the envelope bounds the beam's rays whichever of them the
pupil samples.

The hull, always carried too and the bound wherever the Gaussian state does not hold (a surface too strong for it, a sag
surface, a state invalidated on the way): half width ``x = K w`` (on a followed hit's clear aperture) and half angle ``a = K
theta_far`` about the chief ray, ``x -> x + t tan(a)`` between surfaces. Across a meeting at incidence ``i`` the normal moves by
``phi``: 0 on a plane; on a sphere ``asin(sin i + x / R) - i`` exactly for rays parallel to the chief ray (``R (sin(i + phi) -
sin i) = x``: points of the sphere ``phi`` apart have a chord ``2 R sin(phi / 2)`` at ``i + phi / 2`` to the plane across the ray),
and at most what the cone of the normals over the clear aperture leaves (``resonance._face``; on a sag surface that alone). The
meeting's half width is ``a + phi``. A reflection: ``a -> a + 2 phi`` about the chief ray's mirror image (the normal's turn,
doubled). A refraction: ``a -> M (a + phi) + phi`` about the chief ray refracted at its own normal, ``M`` the largest angular
magnification of Snell's law over the angles met (``n1 cos i / (n2 cos t)``, ``n1 / n2``), the width scaled by ``cos t / cos i``
where that widens it; every direction where those angles reach grazing incidence or the critical angle. A thin lens adds
``atan(x / |f|)``. Where both bounds apply the narrower holds.

What cannot be bounded so. A hit without a Gaussian state that describes it (``GState`` not valid, a central obstruction, or
aberration dominated: ``gaussian_ray_spread`` beyond ``gaussian_max_ray_spread_ratio``): its rays lie on the surface's clear
aperture and within the leg from the surface before it (``resonance._leg``), whose cone stands for ``a``; a wide leg bounds
nothing (a lens's two faces). The part of a branch in air beside an element its chief ray misses enters that element by a path the
chief ray does not show: every coated dense side of an element its envelope reaches without its chief ray is not bounded. A use
whose extent is not bounded is refused, whatever the chief ray does.

Arithmetic: ``+ - x /`` and ``sqrt`` on floats and elementwise NumPy, the libm-free trigonometry of ``math.exact_trig``, the
surfaces' own intersections and the Gaussian steps of the trace: the same bits in every runtime. A scene without a coated dense
side of an air-equivalent record computes nothing here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..config import NumericalConfig
from ..errors import EngineInputError
from ..math.exact_trig import atan2
from ..optics.gaussian.ops import apply_curved_or_flat_interface, apply_grating, apply_thin_lens, propagate
from ..optics.gaussian.state import GState
from ..optics.materials.library import MaterialLibrary
from ..optics.polarization.fresnel import fresnel
from ..optics.rays.decisions import envelope_sigmas
from ..optics.rays.hits import SurfaceHit
from ..optics.surfaces.geometry import intersect
from ..optics.surfaces.grating import diffracted, propagating_orders
from ..optics.surfaces.interaction import thin_lens_dir
from ..optics.surfaces.tabulated_lens import response_at
from ..scene.model import Scene, SurfaceG
from .observations import gaussian_ray_spread
from .resonance import (_DEGREES, _HALF_PI, INDEX_TOL, NEGATIVE, POSITIVE, _angle, _asin, _cos_sin, _dot, _Face, _face, _leg, _reflected,
                        _refracted)

LineKey = tuple[str, int]

#: Why a use is refused.
OUTSIDE_VALIDITY = "OUTSIDE_DECLARED_VALIDITY"
OUTSIDE_WAVELENGTH = "OUTSIDE_DECLARED_WAVELENGTH"
NO_AIR_EQUIVALENT = "NO_AIR_EQUIVALENT_ANGLE"
NOT_BOUNDED = "EXTENT_NOT_BOUNDED"
#: The interactions a branch goes straight through, with its Gaussian state as it is.
STRAIGHT = frozenset({"beam_splitter", "neutral_density", "polarizer", "retarder", "faraday_rotator", "aperture_stop"})


@dataclass(frozen=True)
class CoatingUse:
    """One coated surface side met by left-out light at one vacuum wavelength: the record, the medium it is met from, the
    light's extent there (rad, in that medium) and what it converts to (deg; ``None``: no real air-equivalent angle, or the
    extent is not bounded). ``refused``: why the coating's value does not hold for that light (``None``: it does)."""

    surface_id: str
    element_id: str
    side: int
    wavelength_nm: float
    record_id: str
    medium: str
    index: float
    valid_aoi_deg: tuple[float, float]
    valid_wavelength_nm: tuple[tuple[float, float], ...]
    low: float
    high: float
    air_deg: tuple[float, float] | None
    refused: str | None

    @property
    def high_deg(self) -> float:
        """The top of the extent, in the medium (deg)."""
        return self.high * _DEGREES

    def context(self) -> dict[str, object]:
        """The use, as an issue's context."""
        return {"coating_record": self.record_id, "incident_medium": self.medium, "incident_index": self.index,
                "valid_aoi_deg": list(self.valid_aoi_deg),
                "extent_in_medium_deg": None if self.refused == NOT_BOUNDED else [self.low * _DEGREES, self.high * _DEGREES],
                "air_equivalent_deg": None if self.air_deg is None else list(self.air_deg), "coating_refusal": self.refused}


#: Per vacuum wavelength: (surface id, side) -> the use.
CoatingUses = dict[float, dict[tuple[str, int], CoatingUse]]


def declared_air_equivalent(s: SurfaceG):
    """The declared air-equivalent validity of the surface's coating (``None``: no coating, no record, another reference)."""
    inter = s.interaction
    if inter.kind not in ("refract", "mirror", "beam_splitter") or inter.coating is None:
        return None
    declared = inter.coating.declared_validity
    return declared if declared is not None and declared.aoi_reference == "AIR_EQUIVALENT" else None


def is_air(n: float) -> bool:
    """Whether a medium of this index reads an air-equivalent angle as it is (air and vacuum hold the same light)."""
    return abs(n - 1.0) <= INDEX_TOL


def _tan(a: float) -> float:
    if not a < _HALF_PI:
        return math.inf
    c, s = _cos_sin(a)
    return s / c


def _atan(y: float, x: float) -> float:
    return float(atan2(np.array([y]), np.array([x]))[0])


def judge(low: float, high: float, n: float, declared, wavelength_nm: float) -> tuple[tuple[float, float] | None, str | None]:
    """(the air-equivalent extent in deg, why it is refused) of the in-medium extent [``low``, ``high``] (rad) at ``wavelength_nm``."""
    if not high < _HALF_PI:
        return None, NOT_BOUNDED
    sin_high = _cos_sin(high)[1]
    if n * sin_high >= 1.0:
        return None, NO_AIR_EQUIVALENT
    air = (_asin(n * _cos_sin(low)[1]) * _DEGREES, _asin(n * sin_high) * _DEGREES)
    bands = declared.wavelength_bands_nm
    if bands and not any(band_low <= wavelength_nm <= band_high for band_low, band_high in bands):
        return air, OUTSIDE_WAVELENGTH
    inside = declared.min_deg - declared.slack_deg <= air[0] and air[1] <= declared.max_deg + declared.slack_deg
    return air, None if inside else OUTSIDE_VALIDITY


@dataclass
class _Branch:
    """A left-out branch: its chief ray, the index of its medium, its Gaussian state with the ``K`` of the followed hit
    (``state`` ``None``: it does not hold), the hull (half width, half angle), and the element it left."""

    o: np.ndarray
    d: np.ndarray
    n: float
    state: GState | None
    k: float
    x: float
    a: float
    owner: str


def _usable(state: GState | None) -> GState | None:
    return state if state is not None and state.valid else None


def _gaussian_width(g: GState, k: float, signed_c: tuple[float, ...]) -> float:
    """``D`` of the module docstring at a surface whose normal turns by each of ``signed_c`` (1/mm) across a ray offset."""
    if k <= 0.0:
        return 0.0
    widest = 0.0
    for axis in ("x", "y"):
        w = g.radius(axis)
        rc = g.curvature_radius(axis)
        tilt = 0.0 if rc is None else 1.0 / rc
        local = g.waist_radius(axis) * g.far_field_half_angle(axis) / w
        for c in signed_c:
            slope = w * (tilt - c)
            widest = max(widest, slope * slope + local * local)
    return k * math.sqrt(widest)


class _Walk:
    """One wavelength's scene as the branches meet it: the coated dense sides, and the extent found on each."""

    def __init__(self, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, wavelength_nm: float, faces: list[_Face],
                 index_of: dict[str, int]) -> None:
        self.scene, self.library, self.cfg, self.wavelength_nm = scene, library, cfg, wavelength_nm
        self.faces, self.index_of = faces, index_of
        self.indices: dict[str, float | None] = {}
        self.low: dict[tuple[int, int], float] = {}
        self.high: dict[tuple[int, int], float] = {}
        self.dense: dict[int, list[tuple[int, float]]] = {}  # surface -> (side, index) of its coated dense sides
        for i, s in enumerate(scene.surfaces):
            if declared_air_equivalent(s) is None:
                continue
            for side, medium in ((POSITIVE, s.medium_pos), (NEGATIVE, s.medium_neg)):
                n = self.index(medium)
                if n is not None and not is_air(n):
                    self.dense.setdefault(i, []).append((side, n))
        self.candidates = sorted(self.dense)

    def index(self, medium: str) -> float | None:
        if medium not in self.indices:
            try:
                n, issue = self.library.evaluator(medium).index(self.wavelength_nm)
            except EngineInputError:
                n, issue = None, True
            # relative to the surrounding air (the request's ``air``: the environment's in a native run, index 1 in the
            # Engine's own convention): an air-equivalent angle is the angle in that air with the same n sin(theta)
            self.indices[medium] = None if issue is not None or n is None else float(n) / self._air()
        return self.indices[medium]

    def _air(self) -> float:
        if "\x00air" not in self.indices:
            n_air, issue = self.library.evaluator("air").index(self.wavelength_nm)
            self.indices["\x00air"] = 1.0 if issue is not None or n_air is None else float(n_air)
        return self.indices["\x00air"]  # type: ignore[return-value]

    def spread(self, i: int, d, sin_i: float, aoi: float, x: float) -> float:
        """``phi``: how far the incidence moves from the chief ray's ``aoi`` over a footprint of half width ``x``."""
        s, face = self.scene.surfaces[i], self.faces[i]
        if s.geometry == "plane":
            return 0.0
        phi = math.inf
        if s.geometry == "sphere" and s.radius > 0.0 and math.isfinite(x):
            top = sin_i + x / s.radius
            phi = _asin(top) - aoi if top < 1.0 else _HALF_PI - aoi
        if face.cone < _HALF_PI:
            angle = _angle(d, face.axis)
            phi = min(phi, max(0.0, min(angle, math.pi - angle) + face.cone - aoi))
        return phi

    def settled(self) -> bool:
        """Whether no further meeting can change a use: every coated dense side's extent is already not bounded."""
        return all(self.high.get((i, side), 0.0) >= _HALF_PI and self.low.get((i, side), _HALF_PI) <= 0.0
                   for i in self.candidates for side, _ in self.dense[i])

    def reached(self, i: int, side: int, low: float, high: float) -> None:
        key = (i, side)
        self.low[key] = min(self.low.get(key, _HALF_PI), low)
        self.high[key] = max(self.high.get(key, 0.0), high)

    def beside(self, branch: _Branch, t_end: float, landed: str | None, hull_x: float) -> None:
        """Every element a branch's envelope reaches on this leg without its chief ray (module docstring). Light in a body
        stays in that body: its next surface is the leg's end."""
        if not is_air(branch.n):
            return
        for i in self.candidates:
            s, face = self.scene.surfaces[i], self.faces[i]
            if s.owner_id in (branch.owner, landed) or not math.isfinite(face.radius):
                continue
            w = (face.centre[0] - branch.o[0], face.centre[1] - branch.o[1], face.centre[2] - branch.o[2])
            along = w[0] * branch.d[0] + w[1] * branch.d[1] + w[2] * branch.d[2]
            if along <= 0.0 or along > t_end:
                continue
            across = math.sqrt(max(_dot(w, w) - along * along, 0.0))
            if across <= face.radius + hull_x + along * _tan(branch.a):
                for side, _ in self.dense[i]:
                    self.reached(i, side, 0.0, _HALF_PI)

    def run(self, branches: list[_Branch]) -> None:
        """Every branch on to its next reflection, all chief rays of a step at once."""
        surfaces = self.scene.surfaces
        rows = [i for i, s in enumerate(surfaces) if s.interaction.kind != "observe"]
        for _ in range(2 * len(rows) + 2):
            if not branches or self.settled():
                return
            o = np.array([b.o for b in branches], dtype=np.float64)
            d = np.array([b.d for b in branches], dtype=np.float64)
            found = [intersect(surfaces[i], o, d, self.cfg) for i in rows]
            times = np.array([h.t for h in found], dtype=np.float64)
            nearest = np.argmin(times, axis=0)
            after: list[_Branch] = []
            for k, branch in enumerate(branches):
                row = int(nearest[k])
                t = float(times[row, k])
                if not math.isfinite(t):
                    self.beside(branch, math.inf, None, branch.x)
                    continue
                i, hits = rows[row], found[row]
                self.beside(branch, t, surfaces[i].owner_id, branch.x)
                onward = self.meet(branch, i, hits.point[k], hits.normal[k], t, bool(hits.inside[k]))
                if onward is not None:
                    after.append(onward)
            branches = after

    def meet(self, branch: _Branch, i: int, point, normal, t: float, inside: bool) -> _Branch | None:
        """Record the branch's meeting of surface ``i``; the branch after it (``None``: it ends here)."""
        surface = self.scene.surfaces[i]
        d = branch.d
        dn = float(d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2])
        side = POSITIVE if dn < 0.0 else NEGATIVE
        c = min(abs(dn), 1.0)
        sin_i = math.sqrt((1.0 - c) * (1.0 + c))
        aoi = _asin(sin_i)
        x = branch.x + t * _tan(branch.a) if math.isfinite(branch.x) else math.inf
        phi = self.spread(i, d, sin_i, aoi, x)
        width = branch.a + phi
        state = _usable(propagate(branch.state, t)) if branch.state is not None else None
        low, high = max(0.0, aoi - width), min(_HALF_PI, aoi + width)
        radius = None
        if surface.geometry == "sphere":
            ahead = float((surface.center[0] - point[0]) * d[0] + (surface.center[1] - point[1]) * d[1] + (surface.center[2] - point[2]) * d[2]) > 0.0
            radius = surface.radius if ahead else -surface.radius
        if state is not None and surface.geometry != "sag":
            # (a ray at offset x meets a normal turned by c x: toward it where the sphere's centre lies ahead)
            turns = (0.0,) if radius is None else (-1.0 / radius, -1.0 / (radius * c) if c > 0.0 else math.inf)
            gaussian = _gaussian_width(state, branch.k, turns)
            low = max(low, _asin(sin_i - gaussian))
            high = min(high, _asin(sin_i + gaussian) if sin_i + gaussian < 1.0 else _HALF_PI)
        if surface.geometry == "sag":
            state = None  # (the hull alone from here on)
        for dense_side, n in self.dense.get(i, ()):
            if dense_side == side and abs(n - branch.n) <= INDEX_TOL:
                self.reached(i, side, low, high)
        if not inside:
            return None  # (the chief ray ends on the mount)
        kind = surface.interaction.kind
        here = np.array(point, dtype=np.float64)
        if kind == "refract":
            n1 = self.index(surface.medium_pos if side == POSITIVE else surface.medium_neg)
            n2 = self.index(surface.medium_neg if side == POSITIVE else surface.medium_pos)
            if n1 is None or n2 is None or abs(n1 - branch.n) > INDEX_TOL:
                return None
            return self.refract(branch, surface, here, normal, state, n1, n2, aoi, c, x, phi, radius)
        if kind == "thin_lens":
            focal = surface.interaction.focal_length
            if surface.interaction.table is not None:
                focal = response_at(surface.interaction.table, self.wavelength_nm)[0]
            if focal == 0.0 or not math.isfinite(focal):
                return _Branch(here, d, branch.n, None, branch.k, x, math.inf, surface.owner_id)
            out, _ = thin_lens_dir(here[None, :], surface.origin, surface.normal, d[None, :], focal, self.cfg.parallel_dot_tol)
            bend = _atan(x, abs(focal)) if math.isfinite(x) else math.inf
            turned = None if state is None else _usable(apply_thin_lens(state, d, out[0], c, focal, self.cfg, normal=surface.normal))
            return _Branch(here, out[0], branch.n, turned, branch.k, x, branch.a + bend, surface.owner_id)
        if kind in STRAIGHT or (kind == "mirror" and surface.interaction.coating is not None):
            return _Branch(here, d, branch.n, state, branch.k, x, branch.a, surface.owner_id)
        return None  # (a mirror without a coating, an absorber, a grating: its next reflection, or the light ends)

    def refract(self, branch: _Branch, surface: SurfaceG, here, normal, state: GState | None, n1: float, n2: float, aoi: float,
                c: float, x: float, phi: float, radius: float | None) -> _Branch | None:
        """The branch through an interface from ``n1`` into ``n2`` (``None``: its next reflection, total internal reflection)."""
        out = _refracted(tuple(float(v) for v in branch.d), tuple(float(v) for v in normal), n1, n2)
        if out is None:
            return None
        direction = np.array(out, dtype=np.float64)
        magnified, widened = _snell_widening(n1, n2, aoi, branch.a + phi)
        turned = None
        if state is not None:
            oriented = np.array(normal, dtype=np.float64) * (-1.0 if _dot(branch.d, normal) > 0.0 else 1.0)
            cos_t = abs(float(direction[0] * oriented[0] + direction[1] * oriented[1] + direction[2] * oriented[2]))
            turned = _usable(apply_curved_or_flat_interface(
                state, kind="refract", d_in=branch.d, d_out=direction, n_or=oriented, cos_i=c, n1=n1, n2=n2, cos_t=cos_t, tir=False,
                radius_signed_mm=radius, cfg=self.cfg))
        return _Branch(here, direction, n2, turned, branch.k, x * widened if math.isfinite(x) and math.isfinite(widened) else math.inf,
                       magnified * (branch.a + phi) + phi if math.isfinite(magnified) else math.inf, surface.owner_id)


def _snell_widening(n1: float, n2: float, aoi: float, width: float) -> tuple[float, float]:
    """(``M``, the factor on the transverse width) of a refraction at incidence ``aoi`` within ``width`` (``inf``: the
    envelope reaches grazing incidence or the critical angle)."""
    top = aoi + width
    if not top < _HALF_PI:
        return math.inf, math.inf
    ratio = n1 / n2

    def values(angle: float) -> tuple[float, float]:
        cos_i, sin_i = _cos_sin(angle)
        k = 1.0 - ratio * ratio * sin_i * sin_i
        if k <= 0.0 or cos_i <= 0.0:
            return math.inf, math.inf
        cos_t = math.sqrt(k)
        return ratio * cos_i / cos_t, cos_t / cos_i

    m_low, w_low = values(max(0.0, aoi - width))
    m_high, w_high = values(top)
    return max(m_low, m_high, ratio), max(1.0, w_low, w_high)


def _powers(hit: SurfaceHit, s: SurfaceG, n1: float, n2: float | None, cos_i: float) -> tuple[float, float]:
    """(the smaller of the s and p power reflectances, of the transmittances) of the chief ray at the hit."""
    coating = s.interaction.coating
    if coating is not None:
        c = coating.coefficients(hit.wavelength_nm, np.array([float(hit.aoi_deg)]), location="left-out light")
        if c.issue is None:
            return min(float(c.rs[0]), float(c.rp[0])), min(float(c.ts[0]), float(c.tp[0]))
    if n2 is None:
        return 0.0, 0.0
    f = fresnel(n1, n2, np.array([max(cos_i, float(np.finfo(float).tiny))]))
    r_s, r_p = float(np.abs(f.r_s[0])), float(np.abs(f.r_p[0]))
    return min(r_s * r_s, r_p * r_p), max(0.0, 1.0 - max(r_s * r_s, r_p * r_p))


def _envelope(hit: SurfaceHit, i: int, walk: _Walk, level: float, cfg: NumericalConfig) -> tuple[GState | None, float, float, float]:
    """(the Gaussian state that describes a followed hit, or ``None``; ``K``; the hull's half width; its half angle)."""
    g = hit.gstate
    valid = g.valid and not hit.chief_obstructed and hit.miss_distance_mm != math.inf
    if valid and hit.miss_distance_mm is None:
        spread = gaussian_ray_spread(hit, cfg)  # type: ignore[arg-type]  # SurfaceHit carries the ArrivalRecord fields it reads
        valid = spread is None or spread.ratio <= cfg.gaussian_max_ray_spread_ratio
    face = walk.faces[i]
    if valid:
        k = envelope_sigmas(level, hit.power_W, False, 1.0)
        if k is not None:
            width = k * max(g.radius("x"), g.radius("y"))
            if math.isfinite(face.radius) and hit.miss_distance_mm is None:
                width = min(width, face.radius)
            return g, k, width, k * max(g.far_field_half_angle("x"), g.far_field_half_angle("y"))
    # no Gaussian state describes it: the leg from the surface before it, on this surface's clear aperture
    before = walk.index_of.get(hit.lineage[-1].rsplit(":", 1)[0]) if hit.lineage else None
    if before is None or before == i:
        return None, 0.0, face.radius, math.inf
    leg = _leg(walk.faces[before], face)
    if leg.wide:
        return None, 0.0, face.radius, math.inf
    return None, 0.0, face.radius, _angle(leg.u, tuple(float(v) for v in hit.direction)) + leg.reach


def _branches(hit: SurfaceHit, i: int, walk: _Walk, level: float, every: bool, cfg: NumericalConfig) -> list[_Branch]:
    """The left-out branches of a followed hit (module docstring)."""
    s = walk.scene.surfaces[i]
    inter = s.interaction
    kind = inter.kind
    if kind in ("observe", "absorber"):
        return []
    d = np.array(hit.direction, dtype=np.float64)
    normal = np.array(hit.normal, dtype=np.float64)
    dn = float(d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2])
    side = POSITIVE if dn < 0.0 else NEGATIVE
    # (the medium the hit arrives in, from the surface: a state invalidated on the way keeps the index it had then)
    n1 = walk.index(s.medium_pos if side == POSITIVE else s.medium_neg)
    if n1 is None:
        n1 = hit.gstate.n
    c = min(abs(dn), 1.0)
    sin_i = math.sqrt((1.0 - c) * (1.0 + c))
    aoi = _asin(sin_i)
    n2 = walk.index(s.medium_neg if side == POSITIVE else s.medium_pos)
    position = np.array(hit.position, dtype=np.float64)
    if kind == "diffraction_grating":
        return _orders(hit, i, walk, level, cfg, position, d, normal, n1, n2)
    if kind == "mirror":
        if s.geometry == "plane" and inter.reflective_side != "both" and (inter.reflective_side == "positive") != (side == POSITIVE):
            return []  # (its back side stops the light)
        r, t = (1.0, 0.0) if inter.coating is None else _powers(hit, s, n1, n2, c)
    elif kind == "neutral_density":
        r, t = (inter.attenuator.reflectance if inter.attenuator is not None else 1.0), 1.0
    elif kind == "refract" and n2 is not None and n1 > n2 and (n1 / n2) * sin_i >= 1.0:
        r, t = 1.0, 0.0  # (total internal reflection: the trace follows it)
    elif kind in ("refract", "beam_splitter"):
        r, t = _powers(hit, s, n1, n2, c)
    else:
        r, t = 0.0, 1.0
    left_reflection = r > 0.0 and (every or hit.power_W * r <= level)
    left_transmission = t > 0.0 and (every or hit.power_W * t <= level)
    if not (left_reflection or left_transmission):
        return []
    g, k, x, a = _envelope(hit, i, walk, level, cfg)
    phi = walk.spread(i, d, sin_i, aoi, x)
    radius = None
    if s.geometry == "sphere":
        ahead = float((s.center[0] - position[0]) * d[0] + (s.center[1] - position[1]) * d[1] + (s.center[2] - position[2]) * d[2]) > 0.0
        radius = s.radius if ahead else -s.radius
    if s.geometry == "sag":
        g = None
    out: list[_Branch] = []
    oriented = normal * (-1.0 if dn > 0.0 else 1.0)
    if left_reflection:
        mirrored = np.array(_reflected(tuple(d), tuple(normal)), dtype=np.float64)
        # (convex from the incident side where the centre lies ahead)
        concave = None if radius is None else (-abs(radius) if radius > 0.0 else abs(radius))
        state = None if g is None else _usable(apply_curved_or_flat_interface(
            g, kind="mirror", d_in=d, d_out=mirrored, n_or=oriented, cos_i=c, n1=n1, n2=n1, cos_t=c, tir=False, radius_signed_mm=concave, cfg=cfg))
        out.append(_Branch(position, mirrored, n1, state, k, x, a + 2.0 * phi, s.owner_id))
    if left_transmission:
        if kind == "refract":
            if n2 is not None:
                carrier = _Branch(position, d, n1, g, k, x, a, s.owner_id)
                passed = walk.refract(carrier, s, position, normal, g, n1, n2, aoi, c, x, phi, radius)
                if passed is not None:
                    out.append(passed)
        elif kind == "thin_lens":
            focal = inter.focal_length if inter.table is None else response_at(inter.table, hit.wavelength_nm)[0]
            if focal != 0.0 and math.isfinite(focal):
                turned, _ = thin_lens_dir(position[None, :], s.origin, s.normal, d[None, :], focal, cfg.parallel_dot_tol)
                state = None if g is None else _usable(apply_thin_lens(g, d, turned[0], c, focal, cfg, normal=s.normal))
                out.append(_Branch(position, turned[0], n1, state, k, x, a + _atan(x, abs(focal)) if math.isfinite(x) else math.inf, s.owner_id))
            else:
                out.append(_Branch(position, d, n1, None, k, x, math.inf, s.owner_id))
        else:
            out.append(_Branch(position, d, n1, g, k, x, a, s.owner_id))
    return out


def _orders(hit: SurfaceHit, i: int, walk: _Walk, level: float, cfg: NumericalConfig, position, d, normal, n1: float,
            n2: float | None) -> list[_Branch]:
    """A grating's left-out branches: every order that propagates at the chief ray (not every efficiency is known), the half
    angle widened by the order's anamorphism ``cos i / cos o`` where that widens it."""
    s = walk.scene.surfaces[i]
    grating = s.interaction.grating
    exit_n = n1 if grating.reflection else n2
    if exit_n is None:
        return []
    g, k, x, a = _envelope(hit, i, walk, level, cfg)
    ratio = hit.wavelength_nm / grating.period_nm
    out = []
    cos_i = abs(float(d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2]))
    along = float(d[0] * grating.k_axis[0] + d[1] * grating.k_axis[1] + d[2] * grating.k_axis[2])
    across = float(d[0] * grating.groove_axis[0] + d[1] * grating.groove_axis[1] + d[2] * grating.groove_axis[2])
    for m in propagating_orders(along, across, ratio, n1, exit_n):
        turned, ok = diffracted(d[None, :], s.normal, grating.k_axis, m, ratio, n1, exit_n, grating.reflection)
        if not bool(ok[0]):
            continue
        cos_o = abs(float(turned[0, 0] * normal[0] + turned[0, 1] * normal[1] + turned[0, 2] * normal[2]))
        state = None if g is None else _usable(apply_grating(g, d, turned[0], s.normal, n1, exit_n, cfg))
        out.append(_Branch(position, turned[0], exit_n, state, k, x, a * max(1.0, cos_i / cos_o) if cos_o > 0.0 else math.inf, s.owner_id))
    return out


def coating_uses(scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, hits: Iterable[SurfaceHit],
                 inputs: dict[LineKey, float], open_lines: frozenset[LineKey] = frozenset()) -> CoatingUses:
    """Per vacuum wavelength the coated dense sides left-out light meets (module docstring). ``hits``: the followed hits, the
    trace's and the probes'; ``inputs``: each line's input power; ``open_lines``: lines part of whose light above their level
    the probes did not finish, every branch of their hits counting as left out."""
    if not any(declared_air_equivalent(s) is not None for s in scene.surfaces):
        return {}
    index_of = {s.surface_id: i for i, s in enumerate(scene.surfaces)}
    faces = [_face(i, s) for i, s in enumerate(scene.surfaces)]
    by_wavelength: dict[float, list[SurfaceHit]] = {}
    for hit in hits:
        by_wavelength.setdefault(hit.wavelength_nm, []).append(hit)
    out: CoatingUses = {}
    for wavelength in sorted(by_wavelength):
        walk = _Walk(scene, library, cfg, wavelength, faces, index_of)
        if not walk.candidates:
            continue
        branches: list[_Branch] = []
        for hit in by_wavelength[wavelength]:
            i = index_of.get(hit.surface_id)
            if i is not None:
                line = (hit.source_id, hit.spectral_index)
                branches += _branches(hit, i, walk, cfg.negligible_power_fraction * inputs.get(line, 0.0), line in open_lines, cfg)
        walk.run(branches)
        uses: dict[tuple[str, int], CoatingUse] = {}
        for (i, side), high in sorted(walk.high.items()):
            s = scene.surfaces[i]
            declared = declared_air_equivalent(s)
            low = walk.low[(i, side)]
            n = next(n for dense_side, n in walk.dense[i] if dense_side == side)
            air, refused = judge(low, high, n, declared, wavelength)
            uses[(s.surface_id, side)] = CoatingUse(
                s.surface_id, s.owner_id, side, wavelength, declared.record_id, s.medium_pos if side == POSITIVE else s.medium_neg, n,
                (declared.min_deg, declared.max_deg), declared.wavelength_bands_nm, low, high, air, refused)
        out[wavelength] = uses
    return out


__all__ = ["CoatingUse", "CoatingUses", "NOT_BOUNDED", "NO_AIR_EQUIVALENT", "OUTSIDE_VALIDITY", "OUTSIDE_WAVELENGTH", "coating_uses",
           "declared_air_equivalent", "is_air", "judge"]
