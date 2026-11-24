"""Internal returns of the light the trace leaves out inside one plane-faced optical solid: folds and total internal reflection.

The open paths (``validation/resonance.py``) count every sequence of reflections at distinct surfaces, and the pairs count the
returns between two surfaces facing each other. Inside a plane-faced solid -- a cube splitter, a penta prism, a right-angle or
a dove prism -- light can also return to a face through two or more others (a fold: entrance face, diagonal, side face,
diagonal, entrance face), and a face met from inside beyond its critical angle reflects all of it. Neither is an open path
nor a pair's return. This module bounds both, per solid, for the light inside it.

The solid. The surfaces of one element (one facet mesh of it: an arrayed mesh member's cells apart), all planes, that take light: its faces, each between the body's medium and another
(a mirror face with the body's medium on its reflecting side counts as a face), and its internal planes (a coated diagonal),
the body's medium on both sides. The body is convex (every aperture lies on the body's side of every face's plane: a polygon
aperture -- a mesh solid's planar patch -- by its corners, exactly; any other by its extent) and its medium has one known index
(no crystal). Any other element keeps the pair and open-path treatment unchanged. A polygon's cells are those of its bounding
box that meet it (``_meets_outline``).

Directions. Inside a homogeneous body light goes straight between two reflections, and a reflection at a plane maps its
direction by the face's mirror image. Light gets into the body only through a face, refracted: its direction lies within the
cone the light arriving at that face takes there (``_Scene.refracted`` of the leg from every surface outside the body that
redirects light, its mirror images or refractions where it is a plane, and of every source's beam; at most the critical
cone ``asin(n_out / n_in)`` about the inward normal, which every refracted direction lies in). So every direction inside is
the image of an entering cone under a sequence of face reflections: the cones of a solid are generated from the entering
cones by the faces' mirror images (a cone keeps its half angle under a plane's reflection), merged where one holds another
or their axes agree within ``MERGE_RAD`` (the merged cone holds both), at most ``CONE_LIMIT`` of them, widened at most ``WIDEN_LIMIT`` times; a solid whose cones
do not close within that limit is not bounded here (a fold solid: no bound; a solid of two faces: unchanged).

Steps. A state is (cone, the face the light last left -- reflected by it or entered through it --, the part of that face's
aperture it can leave from: a box on the face's grid, the whole aperture for light that enters). Light in a state meets
face ``t`` next only on a straight segment from a point of that part to a point of ``t``'s aperture (the trace's own
apertures: light beyond them leaves the model) in a direction of its cone. Both apertures are cut into cells (``CELLS`` x
``CELLS``), each within a ball about its centre; every direction from one ball to the other lies within the leg cone of the
two balls (``resonance._leg``), and a segment reaches ``t``'s plane at an angle whose cosine is at least the near ball's
least distance from the plane over the balls' farthest points apart. A pair of cells is dropped where that leg cone and the
state's cone share no direction or no angle is left; a pair whose every direction lies in the cone is reached whole; the
others are cut again (each cell in 2 x 2), ``LEVELS`` times or until ``PAIR_LIMIT`` pairs. The step exists where a pair is
left, its angles are the pairs', and the part of ``t`` it reaches (their cells, on ``t``'s grid) is the next state's. Light
from outside enters a face only where a segment of its cone joins its source's ball and a cell of the face
(``ENTRY_CELLS`` a side, ``_enters``). Light of a cone meets a face only if the cone holds a direction toward the face's outer
side (``_heads_out``: a straight segment from a point of the convex body to the face's plane goes outward), so no step is
sought where it holds none. The step carries ``t``'s amplitude reflectance bound for light meeting it from the body's side over those angles:

* a face between the body and a medium of lower index: 1 where the angles reach the critical angle (total internal
  reflection), else the bare Fresnel bound or the coating's largest reflectance over the angles;
* an air-equivalent coating record met from the denser body: read at the air-equivalent angles (``coating_use.judge``);
  a step beyond the record's stated angles or wavelengths, or above the critical angle, reads the record's stated value
  there (as the return factors read it) and its face loses its exposure upper bound (``SolidFold.refused``,
  ``ReturnGains.withheld``), as a refused first run of the left-out light does; any other record keeps its published
  values beyond its angles, as everywhere;
* a mirror face from its back: 0 (it stops the light); an uncoated splitter or mirror: 1; an attenuator: its stated
  reflectance.

Transmission through a face or an internal plane is at most 1 in amplitude.

The bound. A walk is a sequence of states joined by steps; its light is at most the product of its steps' bounds (in power,
of their squares). Erase its loops at every face (the parts between two visits of one face): what is left visits distinct
faces -- an open path the open-path count already counts --, and what was erased are closed walks from a face back to it.
Per face ``s``, ``F`` holds the first returns between its states (walks from a state at ``s`` to one at ``s`` that meet
``s`` nowhere between); every closed walk at ``s`` is a run of first returns, so they sum to ``C = F 1 + F C`` from each
state, and ``G_s = 1 + max C`` (no certified sum: no bound). The runs of alternations with a face the pairs already count are
``A_s`` (``_loop``). So the walks with a loop other than those add, over all open paths ``s_1 .. s_j`` with their meeting
counts ``m``, at most ``prod m r (prod G - prod A)``: ``ReturnGains`` adds ``sum_j j! (e_j(m r G) - e_j(m r A))`` to the open
paths' count of a piece's field (``crossing_field``), and with the power sums ``G2``, ``A2`` the same to its power
(``crossings``); the walks without such a loop keep their count. The open paths' own reflectance bound of a face of such a
solid is at least its largest step bound (total internal reflection included). Every sum is certified, not estimated: ``y``
with ``b + W y <= y`` checked row by row in the computed values (with a rounding allowance) holds every partial sum of
``sum_k W^k b`` (``_certified``).

Which solids. A solid whose steps close a walk through three or more faces (a fold) has its ``G``; any other (a window, a
plate, a wedge, a dove prism: light that meets each face once between the returns of one pair) keeps the pair and
open-path treatment unchanged. A fold solid whose returns have no certificate (a lossless or nearly lossless return, a cone
set that does not close) gives no bound: its faces are a closed return without a series
(``ReturnGains.closed_cycles``), and no surface the left-out light reaches has an exposure upper bound. Once the steps found
close a walk of lossless steps (weight 1: total internal reflection, an uncoated mirror) and a fold, no sum is certified
however the rest steps -- the steps found only grow as parts and cones grow --, so the analysis stops there (``HELD``). A
closed mesh solid of glass (a corner cube, a penta or rhomboid prism, a dove prism, a light guide) is such a solid: a state keeps
one part of its face's aperture for every visit, so light that advances along faces met beyond the critical angle or
silvered closes a lossless walk of states although the light itself moves on and leaves (a zigzag between two parallel faces,
a penta's two silvered faces and a third met beyond the critical angle), and a corner cube's cones, imaged again by its partly
reflecting entrance face, do not close within ``CONE_LIMIT``. That is the analysis' limit, not a lossless return of the light:
ray trees inside these meshes over the light the model admits find every circulation passing a partial reflection.

The pairs' share. The walks subtracted as the pairs' (``A``) are those whose loops are runs of alternations with a paired
face; the pair factor ``1 + sum G_p`` counts one run with one partner per walk. A walk whose loops are such runs at two or
more faces of one open path, or one run alternating with two partners at one face, is left to the pairs but counted by
neither: its share is second order in ``A - 1`` (at most ``prod A - 1 - sum (A - 1)`` over the solid's faces; 1.1e-3, 1.5e-3
and 3.7e-3 of ``prod G`` on the catalog cubes and penta prism). The bound rests on that share being within the counts' other
allowances; it is not proven here.

Work. The analysis counts its work against the return-light analysis' limit (``validation/return_work.py``, stage
``solid_folds``): ``fold_cell_work`` units per batch of aperture cell pairs it tests, a unit per direction cone it compares a
new cone with, one per state it steps from, and one per ``FOLD_ENTRIES_PER_UNIT`` entries of each iteration of its sums.
Its cones (``CONE_LIMIT``), cuts (``LEVELS``, ``PAIR_LIMIT``) and rounds (``ROUNDS``) are limited per solid, but their
product bounds no time; the count does. An analysis that reaches the limit raises ``OutOfReturnWork``: the closed returns
of that wavelength are then not known (``INCOMPLETE``), never a solid without a fold.

Arithmetic: ``+ - x /`` and ``sqrt`` on floats and elementwise NumPy, the libm-free trigonometry of ``math.exact_trig``,
comparisons, and maxima: the same bits in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..math.exact_trig import atan2
from .coating_use import is_air, judge
from .return_work import FOLD_ENTRIES_PER_UNIT, SOLID_FOLD_STAGE, WorkCounter, fold_cell_work

if TYPE_CHECKING:
    from .resonance import _Leg, _Scene

#: At most this many direction cones per solid (``CONE_LIMIT``): a solid whose cones do not close within it is not bounded here.
CONE_LIMIT = 192
#: Two cones whose axes agree within this angle (rad) are merged into one that holds both.
MERGE_RAD = 1e-6
#: At most this many such widenings per solid: faces that are nearly but not exactly at the angles of a finite reflection
#: group (a mesh's rounded coordinates) widen the merged cones a little at every image, without end; past the limit the
#: cones do not close (as past ``CONE_LIMIT``).
WIDEN_LIMIT = 1024
#: Every cone's half angle is widened by this (rad) for its tests: the rounding of the chains of mirror images that give its
#: axis stays far inside it (``SLACK_RAD`` of it is kept free).
PAD_RAD = 1e-9
SLACK_RAD = 1e-12
#: Cells per side of an aperture.
CELLS = 4
#: Times a pair of cells some segment of a cone may join is cut again (each cell into 2 x 2), and the most pairs a level
#: may cut into (past it the level's pairs stand as they are: conservative).
LEVELS = 6
PAIR_LIMIT = 300000
#: Rounds of the iterations of the first-return sums and their certificate.
ROUNDS = 2048
#: The relative margin the certificate keeps over its iterate, and the rounding allowance of its row check.
MARGIN = 1e-9
ROUNDING = 1e-12
#: The relative resolution at which the iterations of the first-return sums and of their certificate's direction stop: far
#: above the rounding of their sums (whose last bits follow the poses' last bits), far below ``MARGIN``. The certificate
#: (``MARGIN``, the residual over ``1 - lam``, the row check) holds wherever they stop, and both converge geometrically,
#: so they stop at the same round whatever those bits.
CERTIFY_RESOLUTION = 1e-12
#: Rounds of the partial sum that bounds the alternations' runs from below (fewer rounds: a lower sum, still a bound).
LOWER_ROUNDS = 64
#: The pairs of cells one pair is cut into (each cell in 2 x 2).
CHILD_PAIRS = 16
#: The relative allowance (of the face's offset and size) within which an aperture counts as on a face's body side.
CONVEX_TOL = 1e-9
#: Cells per side of a face's aperture against which light arriving from outside is checked (``_enters``).
ENTRY_CELLS = 32
#: Every cell's ball is widened by this (mm), and every step's angles by ``ANGLE_PAD_RAD``: the rounding of the tests stays inside.
BALL_PAD_MM = 1e-9
ANGLE_PAD_RAD = 1e-12


@dataclass(frozen=True)
class SolidFold:
    """One plane-faced solid's internal returns at one wavelength: per face (``surfaces``, ids) its largest step bound
    (``step``: amplitude, total internal reflection included), the sum of its closed walks in amplitude (``loop``, ``G``) and
    in power (``loop_power``, ``G2``) (inf: not bounded), the part of those runs of alternations with a face it pairs with
    alone (``alternation``, ``A``; ``alternation_power``, ``A2``: the pairs count them), the coating uses of its steps an
    air-equivalent record does not hold for (``refused``: those faces have no exposure upper bound); whether a walk closes through three or more faces (``fold``), whether a
    step reaches a critical angle (``total_reflection``), and the number of direction cones (``cones``)."""

    owner_id: str
    surfaces: tuple[str, ...]
    step: tuple[float, ...]
    loop: tuple[float, ...]
    loop_power: tuple[float, ...]
    fold: bool
    total_reflection: bool
    cones: int
    alternation: tuple[float, ...] = ()
    alternation_power: tuple[float, ...] = ()
    refused: tuple = ()
    #: Why the solid's returns have no certified sum (``None``: they have one, ``bounded``): ``HELD_LOSSLESS_STATE_WALK`` (its
    #: states close a walk that keeps all the light, beside a fold), ``CONES_NOT_CLOSED`` (its direction cones do not close
    #: within ``CONE_LIMIT`` / ``WIDEN_LIMIT``), ``ROUNDS_NOT_SETTLED`` (its cones and steps did not settle within ``ROUNDS``)
    #: or ``SUM_NOT_CERTIFIED`` (a closed walk's sum has no certificate). It says what this analysis could not certify,
    #: not that the light circulates.
    uncertified: str | None = None

    @property
    def bounded(self) -> bool:
        return all(math.isfinite(g) for g in self.loop) and all(math.isfinite(g) for g in self.loop_power)


@dataclass
class _Solid:
    owner_id: str
    members: list[int]  # every reflecting plane of the element (scene indices)
    inner: dict[int, int]  # face -> the side the body lies on (POSITIVE / NEGATIVE); internal planes absent
    body_index: float


Spend = Callable[[int], None]


def _no_count(units: int) -> None:
    """A spend that counts nothing (an analysis without a work counter)."""


def _spender(counter: WorkCounter | None) -> Spend:
    """``units`` of the analysis' work counted against ``counter`` (module docstring, "Work"; ``OutOfReturnWork`` at its
    limit)."""
    if counter is None:
        return _no_count
    return lambda units: counter.spend(units, SOLID_FOLD_STAGE)


def solid_folds(sc: _Scene, paired: frozenset[frozenset[str]] = frozenset()) -> tuple[SolidFold, ...]:
    """The internal returns of every plane-faced solid of the scene that has a fold (module docstring), in the order of its
    first surface. ``paired``: the pairs of surface ids whose returns the pair bounds count (finite). Its work counts against
    ``sc.work`` (``OutOfReturnWork`` at the limit)."""
    found = []
    for solid in _solids(sc):
        fold = _analyse(sc, solid, paired)
        if fold is not None:
            found.append(fold)
    return tuple(found)


# ------------------------------------------------------------------------------------------------ the solids


def _solids(sc: _Scene) -> list[_Solid]:
    from .resonance import NEGATIVE, POSITIVE, REFLECTING, _dot, _extent, _side_resolution

    def _inside(at: float, base: float, side: int) -> bool:
        return (at - base) * (1.0 if side == POSITIVE else -1.0) > -_side_resolution(at, base)

    # (one element's surfaces; a facet mesh's patches apart -- the cells of an arrayed mesh member are separate solids)
    by_owner: dict[tuple[str, str | None], list[int]] = {}
    for i, s in enumerate(sc.surfaces):
        by_owner.setdefault((s.owner_id, None if s.facet is None else s.facet.mesh_id), []).append(i)
    solids = []
    for (owner, _mesh), indices in by_owner.items():
        members = [i for i in indices if sc.kinds[i] in REFLECTING]
        if len(members) < 2 or any(sc.surfaces[i].geometry != "plane" for i in members):
            continue
        if any(sc.kinds[i] == "diffraction_grating" for i in members):
            continue
        if any(not math.isfinite(sc.faces[i].radius) for i in members):
            continue
        # the body's side of each surface: where the other surfaces' aperture middles lie (a polygon's corners' mean: its
        # origin is one of its corners)
        centre = [0.0, 0.0, 0.0]
        for i in members:
            middle = _middle(sc, i)
            for k in range(3):
                centre[k] += middle[k]
        centre = [c / len(members) for c in centre]
        inner: dict[int, int] = {}
        body: set[str] = set()
        internal = []
        ok = True
        for i in members:
            s, face = sc.surfaces[i], sc.faces[i]
            if s.medium_pos == s.medium_neg:
                internal.append(i)
                body.add(s.medium_pos)
                continue
            depth = _dot((centre[0] - face.centre[0], centre[1] - face.centre[1], centre[2] - face.centre[2]), face.axis)
            if depth == 0.0:
                ok = False
                break
            side = POSITIVE if depth > 0.0 else NEGATIVE
            inner[i] = side
            body.add(s.medium_pos if side == POSITIVE else s.medium_neg)
        if not ok or len(body) != 1 or len(inner) < 2:
            continue
        medium = next(iter(body))
        if sc.crystal(medium):
            continue
        n = sc.index(medium)
        if n is None:
            continue
        # convex: every member's aperture lies on the body's side of every face's plane (within a rounding allowance); a
        # polygon's extent is its corners' (exact: the half-space is convex), any other aperture's its bounding extent
        for a, side in inner.items():
            fa = sc.faces[a]
            base = _dot(fa.centre, fa.axis)
            size = CONVEX_TOL * (1.0 + abs(base) + fa.radius)
            for b in members:
                if b == a:
                    continue
                corners = _corners(sc, b)
                if corners is None:
                    low, high = _extent(sc.faces[b], fa.axis)
                else:
                    along = [_dot(c, fa.axis) for c in corners]
                    low, high = min(along), max(along)
                if (side == POSITIVE and low < base - size) or (side == NEGATIVE and high > base + size):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # a source inside the body sends light the entering cones do not hold: such a solid is not analysed here (a source
        # within the side resolution of a face counts as inside it: resonance._side_resolution)
        for src in sc.scene.sources:
            p = tuple(float(x) for x in src.position)
            if all(_inside(_dot(p, sc.faces[a].axis), _dot(sc.faces[a].centre, sc.faces[a].axis), side)
                   for a, side in inner.items()):
                ok = False
                break
        if ok:
            solids.append(_Solid(owner, members, inner, n))
    return solids


def _outline(sc: _Scene, i: int) -> list[tuple[float, float]] | None:
    """The corners of a plane face's polygon aperture in its ``(u, v)`` frame, the offset added (``None``: not a polygon).
    Its holes are openings inside the outline: the outline holds every point of the face."""
    s = sc.surfaces[i]
    ap = s.aperture
    if s.geometry != "plane" or ap is None or ap.shape != "polygon" or not ap.vertices:
        return None
    return [(ap.offset_u + float(u), ap.offset_v + float(v)) for u, v in ap.vertices]


def _corners(sc: _Scene, i: int) -> list[tuple[float, float, float]] | None:
    """The corners of a plane face's polygon aperture in space (``None``: not a polygon)."""
    outline = _outline(sc, i)
    if outline is None:
        return None
    s = sc.surfaces[i]
    o, u, v = s.origin, s.u_axis, s.v_axis
    return [(float(o[0] + x * u[0] + y * v[0]), float(o[1] + x * u[1] + y * v[1]), float(o[2] + x * u[2] + y * v[2]))
            for x, y in outline]


def _middle(sc: _Scene, i: int) -> tuple[float, float, float]:
    """A point of a face's aperture's plane inside the solid's faces: a polygon's corners' mean, any other aperture's centre."""
    corners = _corners(sc, i)
    if corners is None:
        return sc.faces[i].centre
    n = float(len(corners))
    total = [0.0, 0.0, 0.0]
    for c in corners:
        for k in range(3):
            total[k] += c[k]
    return (total[0] / n, total[1] / n, total[2] / n)


# ------------------------------------------------------------------------------------------------ cones


class _Cones:
    """The direction cones of a solid (module docstring): per cone its unit axis, the half angle of the directions it holds
    (``core``, rad) and the half angle every test uses (``half``: ``core + PAD_RAD``). A representative axis is computed
    once, by a chain of at most ``CONE_LIMIT`` mirror images, within a few units of rounding per image of the exact one:
    ``PAD_RAD`` holds that, so an image whose axis agrees with a cone's within ``PAD_RAD - SLACK_RAD`` less the core
    difference is that cone."""

    def __init__(self, spend: Spend = _no_count) -> None:
        self.spend = spend
        self.axes: list[tuple[float, float, float]] = []
        self.core: list[float] = []
        self.half: list[float] = []
        self.full = False
        self.widened = 0

    def add(self, u, core: float) -> int | None:
        """The index of a cone that holds the cone (``u``, ``core``): one already held, else one whose axis agrees within
        ``MERGE_RAD``, widened to hold it, else a new one (``None``: the limit -- ``CONE_LIMIT`` cones, or ``WIDEN_LIMIT``
        widenings)."""
        from .resonance import _PI, _angle

        self.spend(max(1, len(self.axes)))
        core = min(core, _PI)
        for k, (v, a) in enumerate(zip(self.axes, self.half)):
            between = _angle(u, v)
            if between + core <= a - SLACK_RAD:
                return k
            if between <= MERGE_RAD:
                if between + core > self.core[k]:
                    self.widened += 1
                    if self.widened > WIDEN_LIMIT:
                        self.full = True
                        return None
                self.core[k] = min(max(self.core[k], between + core), _PI)
                self.half[k] = min(self.core[k] + PAD_RAD, _PI)
                return k
        if len(self.axes) >= CONE_LIMIT:
            self.full = True
            return None
        self.axes.append((float(u[0]), float(u[1]), float(u[2])))
        self.core.append(core)
        self.half.append(min(core + PAD_RAD, _PI))
        return len(self.axes) - 1


# ------------------------------------------------------------------------------------------------ apertures as cells


@dataclass(frozen=True)
class _Plane:
    """A face's aperture as cells: its centre, in-plane axes, half extents (a disc: its radius both ways, ``disc``; a polygon:
    its bounding box's, about the box's centre), its unit normal and the normal's offset."""

    centre: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    h1: float
    h2: float
    disc: bool
    normal: np.ndarray
    base: float
    #: a polygon's corners about ``centre`` (along ``e1``, ``e2``): its cells are those that meet it (``None``: not a polygon)
    outline: np.ndarray | None = None


def _plane(sc: _Scene, i: int) -> _Plane:
    from .resonance import _dot

    s, face = sc.surfaces[i], sc.faces[i]
    outline = _outline(sc, i)
    if outline is not None:
        # (a polygon: its bounding box about the box's centre, and its corners about that centre -- ``face.centre`` is its
        # origin, one of its corners, and ``face.radius`` the disc about that corner that holds it)
        us, vs = [x for x, _ in outline], [y for _, y in outline]
        mu, mv = (min(us) + max(us)) / 2.0, (min(vs) + max(vs)) / 2.0
        e1 = np.array([float(x) for x in s.u_axis])
        e2 = np.array([float(x) for x in s.v_axis])
        o = s.origin
        centre = np.array([float(o[k] + mu * e1[k] + mv * e2[k]) for k in range(3)])
        corners = np.array([[x - mu, y - mv] for x, y in outline], dtype=np.float64)
        return _Plane(centre, e1, e2, (max(us) - min(us)) / 2.0, (max(vs) - min(vs)) / 2.0, False, np.array(face.axis),
                      _dot(face.centre, face.axis), corners)
    if face.rectangle is not None:
        ru, rv, hu, hv = face.rectangle
        e1, e2, h1, h2, disc = np.array(ru), np.array(rv), float(hu), float(hv), False
    else:
        e1 = np.array([float(x) for x in s.u_axis])
        e2 = np.array([float(x) for x in s.v_axis])
        h1 = h2 = float(face.radius)
        disc = True
    return _Plane(np.array(face.centre), e1, e2, h1, h2, disc, np.array(face.axis), _dot(face.centre, face.axis))


def _split(plane: _Plane, x: np.ndarray, y: np.ndarray, hx: float, hy: float, parts: int):
    """Cells (centres in the face's coordinates) cut ``parts`` x ``parts`` from cells of half extents ``hx``, ``hy``, as
    (x, y, half extents, which meet the aperture), ``parts * parts`` children per cell in order; a disc keeps the cells
    that meet it."""
    steps = (np.arange(parts, dtype=np.float64) * 2.0 + 1.0 - parts) / parts
    cx = (x[:, None, None] + steps[None, :, None] * hx + 0.0 * steps[None, None, :]).reshape(-1)
    cy = (y[:, None, None] + 0.0 * steps[None, :, None] + steps[None, None, :] * hy).reshape(-1)
    hx, hy = hx / parts, hy / parts
    if plane.disc:
        keep = np.sqrt(cx * cx + cy * cy) <= plane.h1 + math.sqrt(hx * hx + hy * hy)
        return cx, cy, hx, hy, keep
    if plane.outline is not None:
        return cx, cy, hx, hy, _meets_outline(plane.outline, cx, cy, math.sqrt(hx * hx + hy * hy) + BALL_PAD_MM)
    return cx, cy, hx, hy, np.ones(cx.shape, dtype=bool)


def _meets_outline(outline: np.ndarray, x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Which balls (centres ``x``, ``y``, radius ``r``) meet the polygon ``outline``: the centre inside it (its crossings
    along +x) or within ``r`` of an edge. A cell that meets the polygon is kept (its ball holds it)."""
    inside = np.zeros(x.shape, dtype=bool)
    near = np.zeros(x.shape, dtype=bool)
    r2 = r * r
    n = outline.shape[0]
    for k in range(n):
        ax, ay = float(outline[k, 0]), float(outline[k, 1])
        bx, by = float(outline[(k + 1) % n, 0]), float(outline[(k + 1) % n, 1])
        ex, ey = bx - ax, by - ay
        if ey != 0.0:
            crosses = (ay > y) != (by > y)
            inside ^= crosses & (x < ax + (y - ay) * (ex / ey))
        length2 = ex * ex + ey * ey
        if length2 > 0.0:
            s = np.minimum(np.maximum(((x - ax) * ex + (y - ay) * ey) / length2, 0.0), 1.0)
            dx, dy = x - ax - s * ex, y - ay - s * ey
        else:
            dx, dy = x - ax, y - ay
        near |= dx * dx + dy * dy <= r2
    return inside | near


def _points(plane: _Plane, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return plane.centre[None, :] + x[:, None] * plane.e1[None, :] + y[:, None] * plane.e2[None, :]


def _angle_of(cosine: float) -> float:
    """The angle (rad, in [0, pi/2]) of a cosine in [0, 1] (libm-free)."""
    c = min(max(cosine, 0.0), 1.0)
    return float(atan2(np.array([math.sqrt((1.0 - c) * (1.0 + c))]), np.array([c]))[0])


def _within(x: np.ndarray, y: np.ndarray, hx: float, hy: float, box) -> np.ndarray:
    """Which cells meet the box (x0, x1, y0, y1) of a face's coordinates."""
    x0, x1, y0, y1 = box
    return (x + hx >= x0) & (x - hx <= x1) & (y + hy >= y0) & (y - hy <= y1)


def _snap(plane: _Plane, x0: float, x1: float, y0: float, y1: float) -> tuple[float, float, float, float]:
    """The box widened to the face's finest grid (``CELLS * 2 ** LEVELS`` cells a side over the aperture) and clipped to it:
    the parts of a face reached take finitely many values."""
    n = CELLS * (1 << LEVELS)
    gx, gy = 2.0 * plane.h1 / n, 2.0 * plane.h2 / n
    lo_x = max(0, math.floor((x0 + plane.h1) / gx) - 1)
    hi_x = min(n, math.ceil((x1 + plane.h1) / gx) + 1)
    lo_y = max(0, math.floor((y0 + plane.h2) / gy) - 1)
    hi_y = min(n, math.ceil((y1 + plane.h2) / gy) + 1)
    return (-plane.h1 + lo_x * gx, -plane.h1 + hi_x * gx, -plane.h2 + lo_y * gy, -plane.h2 + hi_y * gy)


def _whole(plane: _Plane) -> tuple[float, float, float, float]:
    return (-plane.h1, plane.h1, -plane.h2, plane.h2)


def _pair_angles(p: np.ndarray, rp: float, q: np.ndarray, rq: float, normal: np.ndarray, base: float,
                 u, half: float, cos_half: float, sin_half: float):
    """Per pair of balls (centres ``p``, ``q`` row by row, radii ``rp``, ``rq``, each widened by ``BALL_PAD_MM``): whether
    some segment from the first to the second has a direction within ``half`` of ``u`` (and whether every one has:
    ``sure``), and the cosines of the least and the largest angle of incidence on the far plane such a segment can have:
    within the leg cone of the two balls (``resonance._leg``), and at most the angle whose cosine is ``h / D`` (``h`` the
    near ball's least distance from the far plane, ``D`` the balls' farthest points apart: a segment reaches the plane only
    across ``h``, along at most ``D``). Cosines throughout (angles in [0, pi/2], where the cosine decreases)."""
    from .resonance import _HALF_PI

    w = q - p
    dist = np.sqrt(w[:, 0] * w[:, 0] + w[:, 1] * w[:, 1] + w[:, 2] * w[:, 2])
    total = rp + rq + 2.0 * BALL_PAD_MM
    wide = total >= dist
    safe = np.where(wide, 1.0, dist)
    lsin = np.where(wide, 0.0, total / safe)
    lcos = np.sqrt((1.0 - lsin) * (1.0 + lsin))
    v = w / safe[:, None]
    dot = v[:, 0] * u[0] + v[:, 1] * u[1] + v[:, 2] * u[2]
    if half >= _HALF_PI:
        # (the sum of the half angles reaches pi where the leg's reaches pi - half: its sine at least the cone's)
        meets = wide | (lsin >= sin_half) | (dot >= cos_half * lcos - sin_half * lsin)
        sure = np.zeros(dot.shape, dtype=bool) if half < _HALF_PI else ~wide & (dot >= cos_half * lcos + sin_half * lsin)
    else:
        meets = wide | (dot >= cos_half * lcos - sin_half * lsin)
        # (every direction of the pair within the cone: the angle between the axes at most the difference of the half angles)
        sure = ~wide & (lsin <= sin_half) & (dot >= cos_half * lcos + sin_half * lsin)
    along = np.abs(v[:, 0] * normal[0] + v[:, 1] * normal[1] + v[:, 2] * normal[2])
    across = np.sqrt(np.maximum(0.0, (1.0 - along) * (1.0 + along)))
    # least angle: psi - reach (0 where reach >= psi); largest: psi + reach (pi/2 beyond), at most the slab's
    cos_low = np.where(wide | (lsin * along >= lcos * across), 1.0, along * lcos + across * lsin)
    cos_high = np.where(wide, 0.0, np.maximum(along * lcos - across * lsin, 0.0))
    h = np.maximum(np.abs(p[:, 0] * normal[0] + p[:, 1] * normal[1] + p[:, 2] * normal[2] - base) - rp - BALL_PAD_MM, 0.0)
    far = dist + total
    least = np.minimum(np.where(far > 0.0, h / np.where(far > 0.0, far, 1.0), 0.0), 1.0)
    return meets, np.minimum(cos_low, 1.0), np.maximum(cos_high, least), sure


# ------------------------------------------------------------------------------------------------ step bounds


def _coated(coating, n: float, low: float, high: float, wavelength_nm: float) -> tuple[float, str | None]:
    """(the largest amplitude reflectance of a coating met from a medium of index ``n`` at angles [``low``, ``high``] (rad),
    why its record does not hold there): an air-equivalent record met from a denser medium is read at the air-equivalent
    angles (``coating_use.judge``), and a use outside its stated angles or wavelengths, or above the critical angle, is
    refused -- the value then read is the record's stated value over the angles as the return factors read it
    (``resonance._dense_coating``: up to grazing where no air-equivalent angle exists), and the face's exposure upper bound
    is withheld (``SolidFold.refused``); any other record keeps its published values beyond its angles, as everywhere."""
    from .resonance import _DEGREES, _GRAZING_DEG, _asin, _coating, _cos_sin

    declared = coating.declared_validity
    lo, hi = low * _DEGREES, high * _DEGREES
    refused = None
    if declared is not None and declared.aoi_reference == "AIR_EQUIVALENT" and not is_air(n):
        air, refused = judge(low, high, n, declared, wavelength_nm)
        if air is not None:
            lo, hi = air
        else:
            lo = _asin(n * _cos_sin(low)[1]) * _DEGREES
            hi = _GRAZING_DEG
    value, _ = _coating(coating, wavelength_nm, lo, hi)
    return math.sqrt(value), refused


def _step(sc: _Scene, solid: _Solid, t: int, side: int, low: float, high: float) -> tuple[float, bool, str | None]:
    """(amplitude reflectance bound, whether the angles reach a critical angle, why the coating's record does not hold
    there: ``_coated``) of surface ``t`` of the solid met from the body (``side``: the side of an internal plane the light
    arrives from) at angles [``low``, ``high``] (rad)."""
    from .resonance import NEGATIVE, POSITIVE, _critical, _interface

    s = sc.surfaces[t]
    inter = s.interaction
    kind = inter.kind
    n_in = solid.body_index
    from_side = solid.inner.get(t, side)
    if kind == "refract":
        n_out = sc.side_index(t, NEGATIVE if from_side == POSITIVE else POSITIVE)
        if n_out is None:
            return 1.0, False, None
        critical = _critical(n_in, n_out)
        if critical is not None and high >= critical:
            return 1.0, True, None
        if inter.coating is None:
            return _interface(n_in, n_out, low, high), False, None
        value, refused = _coated(inter.coating, n_in, low, high, sc.wavelength_nm)
        return value, False, refused
    if kind == "mirror":
        if (inter.reflective_side != "both" and from_side in (POSITIVE, NEGATIVE)
                and (inter.reflective_side == "positive") != (from_side == POSITIVE)):
            return 0.0, False, None  # (its back stops the light)
        if inter.coating is None:
            return 1.0, False, None
        value, refused = _coated(inter.coating, n_in, low, high, sc.wavelength_nm)
        return value, False, refused
    if kind == "beam_splitter":
        if inter.coating is None:
            return 1.0, False, None
        value, refused = _coated(inter.coating, n_in, low, high, sc.wavelength_nm)
        return value, False, refused
    if kind == "neutral_density":
        return (math.sqrt(inter.attenuator.reflectance) if inter.attenuator is not None else 1.0), False, None
    return 1.0, False, None


# ------------------------------------------------------------------------------------------------ entering light


def _enters(centre, radius: float, plane: _Plane, cone: _Leg, spend: Spend = _no_count) -> bool:
    """Whether light from the ball (``centre``, ``radius``: a surface's bounding ball, a source's beam) in a direction of
    ``cone`` can reach the face's aperture: some cell of the aperture (``ENTRY_CELLS`` a side) and the ball are joined by a
    segment of the cone (``_pair_angles``; an unbounded ball or a wide cone: yes)."""
    if cone.wide or not math.isfinite(radius):
        return True
    zero = np.zeros(1)
    xt, yt, hx, hy, keep = _split(plane, zero, zero, plane.h1, plane.h2, ENTRY_CELLS)
    xt, yt = xt[keep], yt[keep]
    q = _points(plane, xt, yt)
    p = np.repeat(np.array([[float(x) for x in centre]]), q.shape[0], axis=0)
    spend(fold_cell_work(int(q.shape[0])))
    meets, _, _, _ = _pair_angles(p, radius, q, math.sqrt(hx * hx + hy * hy), plane.normal, plane.base, cone.u, cone.reach,
                                  cone.cos, cone.sin)
    return bool(np.any(meets))


def _entering(sc: _Scene, solid: _Solid, a: int, spend: Spend = _no_count) -> list[_Leg]:
    """The cones of the light entering the solid through face ``a``, in the body (module docstring)."""
    from .resonance import (
        EITHER, NEGATIVE, POSITIVE, REFLECTING, _cone, _critical, _HALF_PI, _Leg, _leg, _overlap,
    )

    face = sc.faces[a]
    plane = _plane(sc, a)
    inner = solid.inner[a]
    outer = NEGATIVE if inner == POSITIVE else POSITIVE
    if sc.coefficients(a, outer, 0.0, _HALF_PI)[1] <= 0.0:
        return []  # (it passes nothing in from outside: a mirror's back)
    n_out = sc.side_index(a, outer)
    if n_out is None:
        return [_cone(face.axis if inner == POSITIVE else tuple(-x for x in face.axis), _HALF_PI, inner)]
    inward = face.axis if inner == POSITIVE else (-face.axis[0], -face.axis[1], -face.axis[2])
    critical = _critical(solid.body_index, n_out)
    cap = _cone(inward, _HALF_PI if critical is None else critical, inner)
    arriving: list[_Leg] = []
    own = set(solid.members)
    for e in sc.redirecting:
        if e in own:
            continue
        geometric = sc.leg(e, a)
        if geometric.side not in (outer, EITHER):
            continue
        if sc.arrival_index(sc.emitted(e, a), a, outer) is False:
            continue
        if sc.faces[e].plane and sc.kinds[e] in REFLECTING:
            toward = sc.toward(e, a)
            for image in sc.images(e):
                if toward != EITHER and image.side != EITHER and image.side != toward:
                    continue
                if _overlap(image, geometric):
                    chosen = image if image.reach < geometric.reach else geometric
                    if _enters(sc.faces[e].centre, sc.faces[e].radius, plane, chosen, spend):
                        arriving.append(chosen)
        elif _enters(sc.faces[e].centre, sc.faces[e].radius, plane, geometric, spend):
            arriving.append(geometric)
    for k, (ball, beam) in enumerate(sc.beams):
        geometric = _leg(ball, face)
        if geometric.side not in (outer, EITHER) or not _overlap(beam, geometric):
            continue
        if sc.arrival_index(sc.index(sc.scene.sources[k].medium), a, outer) is False:
            continue
        if _enters(ball.centre, ball.radius, plane, beam, spend):
            arriving.append(beam)
    cones = []
    for leg in arriving:
        if leg.wide:
            cones.append(cap)
            continue
        refracted = sc.refracted(_Leg(leg.u, leg.reach, leg.cos, leg.sin, outer), a, face.axis, outer, inner)
        cones.append(cap if refracted.wide or refracted.reach >= cap.reach else refracted)
    return cones


# ------------------------------------------------------------------------------------------------ the analysis


def _analyse(sc: _Scene, solid: _Solid, paired: frozenset[frozenset[str]] = frozenset()) -> SolidFold | None:
    from .resonance import _HALF_PI, _PI, _cos_sin, _dot, _reflected

    members = solid.members
    spend = _spender(sc.work)
    cones = _Cones(spend)
    starts: list[tuple[int, int]] = []  # (cone, face)
    for a in solid.inner:
        for leg in _entering(sc, solid, a, spend):
            k = cones.add(leg.u, leg.reach)
            if k is None:
                break
            if (k, a) not in starts:
                starts.append((k, a))
    if not starts and not cones.full:
        return None  # (no light enters it)
    planes = {i: _plane(sc, i) for i in members}

    remembered: dict[tuple, tuple | None] = {}

    def step_angles(k: int, s: int, box, t: int):
        """``_step_angles``, remembered per cone (axis and half angle), faces and part (the closure and the steps are found
        again whenever a cone widens)."""
        key = (cones.axes[k], cones.half[k], s, box, t)
        if key not in remembered:
            remembered[key] = _step_angles(k, s, box, t)
        return remembered[key]

    def _step_angles(k: int, s: int, box, t: int):
        """(the angles of incidence at ``t`` of the light of cone ``k`` leaving the part ``box`` of ``s``'s aperture that can
        reach ``t``, the part of ``t``'s aperture it can reach) (``None``: none): pairs of cells of the two apertures (the
        aperture's fixed grid; on ``s`` the cells that meet ``box``); every pair some segment of the cone may join is cut
        again (each cell in four), ``LEVELS`` times; a pair no such segment joins is dropped; past ``PAIR_LIMIT`` pairs a
        level's pairs stand. The part reached: the box of the surviving cells of ``t``, on its grid."""
        u = cones.axes[k]
        a = cones.half[k]
        c, sn = (_cos_sin(a) if a < _PI else (-1.0, 0.0))
        cu = abs(_dot(u, sc.faces[t].axis))
        su = math.sqrt(max(0.0, (1.0 - cu) * (1.0 + cu)))
        if a >= _HALF_PI:
            cone_cos_low, cone_cos_high = 1.0, 0.0
        else:
            cone_cos_low = 1.0 if sn * cu >= c * su else cu * c + su * sn
            cone_cos_high = max(cu * c - su * sn, 0.0)
        ps, pt = planes[s], planes[t]
        zero = np.zeros(1)
        xs, ys, hxs, hys, ks = _split(ps, zero, zero, ps.h1, ps.h2, CELLS)
        ks &= _within(xs, ys, hxs, hys, box)
        xt, yt, hxt, hyt, kt = _split(pt, zero, zero, pt.h1, pt.h2, CELLS)
        xs, ys, xt, yt = xs[ks], ys[ks], xt[kt], yt[kt]
        si = np.repeat(np.arange(xs.size), xt.size)
        ti = np.tile(np.arange(xt.size), xs.size)
        sx, sy, tx, ty = xs[si], ys[si], xt[ti], yt[ti]
        cos_low_all, cos_high_all = -math.inf, math.inf
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for level in range(LEVELS + 1):
            rp, rq = math.sqrt(hxs * hxs + hys * hys), math.sqrt(hxt * hxt + hyt * hyt)
            spend(fold_cell_work(int(sx.size)))
            meets, cos_low, cos_high, sure = _pair_angles(_points(ps, sx, sy), rp, _points(pt, tx, ty), rq, pt.normal, pt.base, u, a,
                                                          c, sn)
            cos_low, cos_high = np.minimum(cos_low, cone_cos_low), np.maximum(cos_high, cone_cos_high)
            ok = meets & (cos_low >= cos_high)
            # (a pair every segment of which lies in the cone is reached whole: it is kept and not cut again; the others
            # that may be reached are cut again, unless this is the last level or their number has reached the limit)
            last = level == LEVELS or int(np.count_nonzero(ok & ~sure)) * CHILD_PAIRS > PAIR_LIMIT
            done = ok if last else ok & sure
            if bool(np.any(done)):
                cos_low_all = max(cos_low_all, float(np.max(cos_low[done])))
                cos_high_all = min(cos_high_all, float(np.min(cos_high[done])))
                x0, x1 = min(x0, float(np.min(tx[done])) - hxt), max(x1, float(np.max(tx[done])) + hxt)
                y0, y1 = min(y0, float(np.min(ty[done])) - hyt), max(y1, float(np.max(ty[done])) + hyt)
            if last:
                break
            more = ok & ~sure
            sx, sy, tx, ty = sx[more], sy[more], tx[more], ty[more]
            if sx.size == 0:
                break
            # (each pair left: both cells cut in four, every child of one with every child of the other)
            pairs = sx.size
            cx, cy, nhxs, nhys, keep_s = _split(ps, sx, sy, hxs, hys, 2)
            keep_s &= _within(cx, cy, nhxs, nhys, box)
            dx, dy, nhxt, nhyt, keep_t = _split(pt, tx, ty, hxt, hyt, 2)
            cx, cy, keep_s = cx.reshape(pairs, 4), cy.reshape(pairs, 4), keep_s.reshape(pairs, 4)
            dx, dy, keep_t = dx.reshape(pairs, 4), dy.reshape(pairs, 4), keep_t.reshape(pairs, 4)
            sx = np.repeat(cx, 4, axis=1).reshape(-1)
            sy = np.repeat(cy, 4, axis=1).reshape(-1)
            tx = np.tile(dx, (1, 4)).reshape(-1)
            ty = np.tile(dy, (1, 4)).reshape(-1)
            keep = (np.repeat(keep_s, 4, axis=1) & np.tile(keep_t, (1, 4))).reshape(-1)
            sx, sy, tx, ty = sx[keep], sy[keep], tx[keep], ty[keep]
            hxs, hys, hxt, hyt = nhxs, nhys, nhxt, nhyt
        if not cos_low_all >= cos_high_all:
            return None
        # (the angles, each widened by ``ANGLE_PAD_RAD`` over the rounding of their cosines)
        low = max(0.0, _angle_of(cos_low_all) - ANGLE_PAD_RAD)
        high = min(_HALF_PI, _angle_of(cos_high_all) + ANGLE_PAD_RAD)
        return low, high, _snap(pt, x0, x1, y0, y1)

    # the cone set: the entering cones and their mirror images at every face (the closure is over directions only); then
    # the states and their steps from the entering states. A step whose image widens a cone or adds one invalidates the
    # steps found from the narrower cones: the closure and the steps are found again (at most ``ROUNDS`` times).
    for _ in range(ROUNDS):
        queue = list(range(len(cones.axes)))
        seen_half: dict[int, float] = {}
        while queue and not cones.full:
            k = queue.pop()
            if seen_half.get(k) == cones.core[k]:
                continue
            seen_half[k] = cones.core[k]
            for t in members:
                image = _reflected(cones.axes[k], sc.faces[t].axis)
                before = len(cones.axes)
                j = cones.add(image, cones.core[k])
                if j is None:
                    break
                if len(cones.axes) > before or seen_half.get(j) != cones.core[j]:
                    queue.append(j)
        if cones.full:
            return _unbounded(sc, solid, len(cones.axes), CONES_NOT_CLOSED) if len(members) >= 3 else None
        settled = (len(cones.axes), tuple(cones.core))
        found = _steps(sc, solid, cones, starts, step_angles, planes)
        if found is None or found is HELD:
            why = HELD_LOSSLESS_STATE_WALK if found is HELD else CONES_NOT_CLOSED
            return _unbounded(sc, solid, len(cones.axes), why) if len(members) >= 3 else None
        if (len(cones.axes), tuple(cones.core)) == settled:
            break
    else:
        return _unbounded(sc, solid, len(cones.axes), ROUNDS_NOT_SETTLED) if len(members) >= 3 else None
    states, out, step_bound, reaches_critical, refusals = found
    surface_of = [s for _, s in states]
    fold = _closes_through_three(out, surface_of)
    if not fold:
        return None  # (a solid whose light meets each face once between returns of one pair: the pairs count it)
    loop, loop_power, alternation, alternation_power = [], [], [], []
    ids = [sc.surfaces[i].surface_id for i in members]
    for s in members:
        partners = frozenset(t for t in members if frozenset((sc.surfaces[s].surface_id, sc.surfaces[t].surface_id)) in paired)
        g, a = _loop(out, surface_of, s, 1, partners, spend)
        g2, a2 = _loop(out, surface_of, s, 2, partners, spend)
        loop.append(g)
        alternation.append(a)
        loop_power.append(g2)
        alternation_power.append(a2)
    certified = all(math.isfinite(g) for g in loop) and all(math.isfinite(g) for g in loop_power)
    return SolidFold(solid.owner_id, tuple(ids), tuple(step_bound[i] for i in members), tuple(loop), tuple(loop_power), fold,
                     reaches_critical, len(cones.axes), tuple(alternation), tuple(alternation_power), _uses(sc, solid, refusals),
                     None if certified else SUM_NOT_CERTIFIED)


def _uses(sc: _Scene, solid: _Solid, refusals: dict[tuple[int, int], tuple[float, float, str]]) -> tuple:
    """The refused coating uses of the solid's steps, as ``coating_use.CoatingUse`` (their extent: the steps' angles)."""
    from .coating_use import CoatingUse
    from .resonance import POSITIVE

    out = []
    for (t, side), (low, high, why) in sorted(refusals.items()):
        s = sc.surfaces[t]
        declared = s.interaction.coating.declared_validity
        air, _ = judge(low, high, solid.body_index, declared, sc.wavelength_nm)
        out.append(CoatingUse(s.surface_id, s.owner_id, side, sc.wavelength_nm, declared.record_id,
                              s.medium_pos if side == POSITIVE else s.medium_neg, solid.body_index,
                              (declared.min_deg, declared.max_deg), declared.wavelength_bands_nm, low, high, air, why))
    return tuple(out)


def _steps(sc: _Scene, solid: _Solid, cones: _Cones, starts: list[tuple[int, int]], step_angles, planes):
    """The states reached from the entering states and their steps: (states as (cone, face), per state its steps as
    (state, amplitude bound, power bound), per face its largest step bound, whether a step reaches a critical angle);
    ``None``: the cone limit. Each state holds the part of its face's aperture its light can leave from (a box on the face's
    grid, the whole aperture for an entering state), grown as steps reach it; a state is stepped from again whenever its
    part grows, its steps replaced."""
    from .resonance import _reflected

    members = solid.members
    index: dict[tuple[int, int], int] = {}
    states: list[tuple[int, int]] = []
    boxes: list[tuple[float, float, float, float]] = []
    out: list[list[tuple[int, float, float]]] = []
    step_bound = {i: 0.0 for i in members}
    reaches_critical = False
    refusals: dict[tuple[int, int], tuple[float, float, str]] = {}

    def state(k: int, s: int, box) -> tuple[int, bool]:
        key = (k, s)
        if key not in index:
            index[key] = len(states)
            states.append(key)
            boxes.append(box)
            out.append([])
            return index[key], True
        x = index[key]
        old = boxes[x]
        grown = (min(old[0], box[0]), max(old[1], box[1]), min(old[2], box[2]), max(old[3], box[3]))
        if grown != old:
            boxes[x] = grown
            return x, True
        return x, False

    # the steps of weight 1 (a lossless step: total internal reflection, an uncoated mirror), and whether they close a walk
    lossless: dict[int, set[int]] = {}
    held = False

    def closes(x: int, y: int) -> bool:
        """Whether the lossless steps lead from ``y`` back to ``x`` (the new lossless step ``x -> y`` closes a walk)."""
        seen, pending = {y}, [y]
        while pending:
            z = pending.pop()
            if z == x:
                return True
            for nxt in lossless.get(z, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    pending.append(nxt)
        return False

    frontier = []
    for k, a in starts:
        x, _ = state(k, a, _whole(planes[a]))
        frontier.append(x)
    while frontier:
        x = frontier.pop()
        cones.spend(1)
        k, s = states[x]
        steps = []
        for t in members:
            if t == s or not _heads_out(sc, solid, cones, k, t):
                continue
            found = step_angles(k, s, boxes[x], t)
            if found is None:
                continue
            low, high, box = found
            side = solid.inner.get(t, sc.toward(t, s))
            w, critical, refused = _step(sc, solid, t, side, low, high)
            if critical:
                reaches_critical = True
            if refused is not None:
                key = (t, side)
                old = refusals.get(key)
                refusals[key] = (low, high, refused) if old is None else (min(old[0], low), max(old[1], high), old[2])
            step_bound[t] = max(step_bound[t], w)
            if w <= 0.0:
                continue
            j = cones.add(_reflected(cones.axes[k], sc.faces[t].axis), cones.core[k])
            if j is None:
                return None
            y, changed = state(j, t, box)
            steps.append((y, w, w * w))
            if w >= 1.0 and y not in lossless.setdefault(x, set()):
                lossless[x].add(y)
                held = held or closes(x, y)
            if changed and y not in frontier:
                frontier.append(y)
        out[x] = steps
        # (a lossless closed walk and a fold already found: the sums have no certificate however the rest steps -- the steps
        # found only grow --, so the solid is held unbounded now, ``HELD``)
        if held:
            cones.spend(1 + (len(states) + sum(len(row) for row in out)) // FOLD_ENTRIES_PER_UNIT)
            if _closes_through_three(out, [face for _, face in states]):
                return HELD
    return states, out, step_bound, reaches_critical, refusals


def _heads_out(sc: _Scene, solid: _Solid, cones: _Cones, k: int, t: int) -> bool:
    """Whether light of cone ``k`` can meet face ``t`` from inside the body: a straight segment from a point of the convex
    body to a point of ``t``'s plane goes toward ``t``'s outer side (``d . n_out >= 0``), so the cone must hold such a
    direction -- its axis within ``pi/2`` plus its half angle of the outward normal. An internal plane is met from either
    side: always."""
    from .resonance import _HALF_PI, POSITIVE, _cos_sin, _dot

    side = solid.inner.get(t)
    if side is None:
        return True
    half = cones.half[k]
    if half >= _HALF_PI:
        return True
    axis = sc.faces[t].axis
    along = _dot(cones.axes[k], axis)
    outward = -along if side == POSITIVE else along
    # (cos of the angle to the outward normal at least cos(pi/2 + half) = -sin(half), with a rounding allowance)
    return outward >= -_cos_sin(half)[1] - SLACK_RAD


#: ``_steps``: a lossless closed walk and a fold are found -- the solid's returns have no certified sum.
HELD = "held"

#: Why a solid's returns have no certified sum (``SolidFold.uncertified``).
HELD_LOSSLESS_STATE_WALK = "HELD_LOSSLESS_STATE_WALK"
CONES_NOT_CLOSED = "CONES_NOT_CLOSED"
ROUNDS_NOT_SETTLED = "ROUNDS_NOT_SETTLED"
SUM_NOT_CERTIFIED = "SUM_NOT_CERTIFIED"


def _unbounded(sc: _Scene, solid: _Solid, cones: int, why: str) -> SolidFold:
    members = solid.members
    inf = tuple(math.inf for _ in members)
    return SolidFold(solid.owner_id, tuple(sc.surfaces[i].surface_id for i in members), tuple(1.0 for _ in members), inf, inf,
                     True, True, cones, inf, inf, uncertified=why)


def _closes_through_three(out: list[list[tuple[int, float, float]]], surface_of: list[int]) -> bool:
    """Whether some strongly connected set of states spans three or more faces (iterative Kosaraju)."""
    n = len(out)
    order: list[int] = []
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, iter(out[start]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(node)
                stack.pop()
            elif not seen[child[0]]:
                seen[child[0]] = True
                stack.append((child[0], iter(out[child[0]])))
    reverse: list[list[int]] = [[] for _ in range(n)]
    for x, row in enumerate(out):
        for y, _, _ in row:
            reverse[y].append(x)
    seen = [False] * n
    for start in reversed(order):
        if seen[start]:
            continue
        seen[start] = True
        stack, faces, size = [start], set(), 0
        while stack:
            node = stack.pop()
            size += 1
            faces.add(surface_of[node])
            for child in reverse[node]:
                if not seen[child]:
                    seen[child] = True
                    stack.append(child)
        if len(faces) >= 3:
            return True
    return False


def _loop(out: list[list[tuple[int, float, float]]], surface_of: list[int], s: int, power: int,
          paired: frozenset[int], spend: Spend = _no_count) -> tuple[float, float]:
    """(``G_s``, ``A_s``) for face ``s`` (``power`` 1: amplitude; 2: power).

    ``F`` holds the first returns between the states at ``s``: ``F[x][z]`` sums the walks from state ``x`` that next meet
    ``s`` in state ``z`` (certified: the walks through the other states, ``_certified``). Every closed walk at ``s`` from ``x``
    is a run of first returns, so they sum to ``C(x)`` with ``C = F 1 + F C`` (certified the same way), and ``G = 1 + max C``.
    ``F_a`` keeps the first returns that go to a face ``s`` pairs with (``paired``) and straight back, the alternations the
    pairs count; their runs sum to ``C_a`` (bounded above, and below by a partial sum, ``C_low``). The runs with at least one
    other return sum to ``C - C_a`` at most ``D = max (C - C_low)``, so ``A = 1 + max C_a`` and ``G = A + D`` hold the
    telescoping of ``ReturnGains._looped``. (inf, inf): a sum not certified."""
    n = len(out)
    at = [surface_of[x] == s for x in range(n)]
    homes = [x for x in range(n) if at[x]]
    if not homes:
        return 1.0, 1.0
    pick = 1 if power == 1 else 2
    others = [x for x in range(n) if not at[x]]
    position = {x: k for k, x in enumerate(others)}
    home = {x: k for k, x in enumerate(homes)}
    rows: list[list[tuple[int, float]]] = []
    for x in others:
        rows.append([(position[y], weights[pick - 1]) for y, *weights in out[x] if not at[y]])
    # the walks from each other state that next meet s in each state at s
    through: list[list[float]] = []
    for z in homes:
        b = [sum_into(out[x], z, pick) for x in others]
        y = _certified(rows, b, spend)
        if y is None:
            return math.inf, math.inf
        through.append(y)
    first: list[list[tuple[int, float]]] = []
    alternate: list[list[tuple[int, float]]] = []
    for x in homes:
        row, alt = [], []
        for zk, z in enumerate(homes):
            total = straight = 0.0
            for y, *weights in out[x]:
                w = weights[pick - 1]
                if y == z:
                    total += w
                elif not at[y]:
                    total += w * through[zk][position[y]]
                    if surface_of[y] in paired:
                        straight += w * sum_into(out[y], z, pick)
            if total > 0.0:
                row.append((zk, total * (1.0 + ROUNDING)))
            if straight > 0.0:
                alt.append((zk, straight))
        first.append(row)
        alternate.append(alt)
    loops = _certified(first, [_row_sum(row) * (1.0 + ROUNDING) for row in first], spend)
    if loops is None:
        return math.inf, math.inf
    alt_up = _certified([[(k, w * (1.0 + ROUNDING)) for k, w in row] for row in alternate],
                        [_row_sum(row) * (1.0 + ROUNDING) for row in alternate], spend)
    if alt_up is None:
        return math.inf, math.inf
    # (a partial sum of the alternations' runs: below their sum, each product rounded down)
    low_rows = [[(k, w * (1.0 - ROUNDING)) for k, w in row] for row in alternate]
    low_b = [_row_sum(row) * (1.0 - ROUNDING) for row in low_rows]
    alt_low = [0.0] * len(homes)
    for _ in range(LOWER_ROUNDS):
        spend(1 + sum(len(row) for row in low_rows) // FOLD_ENTRIES_PER_UNIT)
        alt_low = [(b + _row_sum(row, alt_low)) * (1.0 - ROUNDING) for b, row in zip(low_b, low_rows)]
    extra = 0.0
    for c, c_low in zip(loops, alt_low):
        extra = max(extra, c - c_low)
    paired_sum = 1.0 + max(alt_up)
    return (paired_sum + extra) * (1.0 + ROUNDING), paired_sum


def _row_sum(row: list[tuple[int, float]], x: list[float] | None = None) -> float:
    """``sum w`` (or ``sum w x[k]``) over a row, plain additions in order (``sum`` is compensated on some Pythons only)."""
    total = 0.0
    for k, w in row:
        total += w if x is None else w * x[k]
    return total


def sum_into(row: list[tuple[int, float, float]], z: int, pick: int) -> float:
    """The weights of the steps of ``row`` into state ``z``."""
    total = 0.0
    for y, *weights in row:
        if y == z:
            total += weights[pick - 1]
    return total


def _certified(rows: list[list[tuple[int, float]]], b: list[float], spend: Spend = _no_count) -> list[float] | None:
    """A vector ``y >= 0`` with ``b + W y <= y`` row by row (``W`` the rows), checked in the computed values with a rounding
    allowance, so ``y`` holds every partial sum of ``sum_k W^k b`` (``None``: none found)."""
    n = len(rows)
    if n == 0:
        return []
    cost = 1 + (n + sum(len(row) for row in rows)) // FOLD_ENTRIES_PER_UNIT

    def apply(x: list[float]) -> list[float]:
        spend(cost)
        z = [0.0] * n
        for k in range(n):
            total = 0.0
            for j, w in rows[k]:
                total += w * x[j]
            z[k] = total
        return z

    def settled(nxt: list[float], x: list[float]) -> bool:
        return max(abs(a - c) for a, c in zip(nxt, x)) <= CERTIFY_RESOLUTION * max(abs(a) for a in nxt)

    u = list(b)
    for _ in range(ROUNDS):
        nxt = [a + c for a, c in zip(b, apply(u))]
        if settled(nxt, u):
            u = nxt
            break
        u = nxt

    def holds(y: list[float]) -> bool:
        return all((a + c) * (1.0 + ROUNDING) <= d for a, c, d in zip(b, apply(y), y))

    # else above the iterate along ``q = sum_k W^k 1`` (iterated), for which ``W q = q - 1``: ``W q <= lam q`` with ``lam =
    # max (W q) / q < 1`` wherever the sum settles (no settled sum: no certificate)
    q = [1.0] * n
    for _ in range(ROUNDS):
        nxt = [1.0 + c for c in apply(q)]
        if not all(math.isfinite(c) for c in nxt):
            return None
        if settled(nxt, q):
            q = nxt
            break
        q = nxt
    wq = apply(q)
    lam = 0.0
    for a, c in zip(wq, q):
        if a > 0.0:
            if not c > 0.0:
                return None
            lam = max(lam, a / c)
    if not lam < 1.0:
        return None
    residual = [a + c - d for a, c, d in zip(b, apply(u), u)]
    eps = MARGIN * max(max(u), 0.0)
    for r, c in zip(residual, q):
        if r > 0.0:
            if not c > 0.0:
                return None
            eps = max(eps, r / ((1.0 - lam) * c))
    y = [(1.0 + MARGIN) * a + 2.0 * eps * c for a, c in zip(u, q)]
    return y if holds(y) else None


__all__ = ["CELLS", "CONE_LIMIT", "SolidFold", "solid_folds"]
