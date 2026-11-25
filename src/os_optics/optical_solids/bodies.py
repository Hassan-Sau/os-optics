"""Optical solid bodies of compiled elements, in the element frame (mm), independent of catalog identity.

A body is what light physically travels in or meets, built from the element's optical model only:

* ``Slab``: the medium between two consecutive coaxial surfaces of a surface sequence over the gap's outline (round
  or rectangular; ``SurfaceSequenceModelV1.gap_outline_mm``: the larger of the two surfaces' edges), bounded by the full
  profiles, each held flat beyond its own mechanical edge: ``front(x, y) <= z <= back(x, y)``. Air gaps of a sequence are
  not solids. A single coated surface whose substrate is not modelled gets the slab between it and the far face of its
  physical design body (when one is given), as the optical view draws it (``surface_mesh.substrate_face_z``): behind a
  reflecting surface the body's -Z face (light meets a mirror's coated face from +Z, so the air in front of a concave
  mirror is never glass), behind any other coated surface the body face farther from its vertex;
* ``Convex``: a convex solid (half-spaces and, for a round plate, a side radius): a surface set's published body, or the
  physical design geometry of an element whose model has no solid (a polarization element's plate, an off-axis
  mirror's substrate);
* ``Plate``: an aperture stop with a stated body: that body minus the opening (light passes there), plus the central
  obstruction disc;
* ``Sheet``: a surface with nothing modelled behind it and no design geometry (zero thickness): it has no interior,
  but another body can still cut it.

Every piece offers ``depth(p)``: the signed distance of an element-frame point inside it (positive inside, to first
order along the surface normals), and ``patches()``: its boundary as parametrized patches ``(u, v) in [0, 1]^2 ->
point`` that the overlap search samples and refines.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from .._vendor.shared_contracts.common.sag_shapes import sag_form
from ..optical_view.solids import Solid, face_polygon, solid_from_planes
from ..optical_view.surface_mesh import Circle, Polygon, Profile, held_point, substrate_face_z

Vec3 = tuple[float, float, float]
AIR = "AIR"
#: A declared model's evacuated gap (the Engine's ``vacuum``); like ``AIR`` it is no body's material.
VACUUM = "VACUUM"
SURROUNDINGS = frozenset({AIR, VACUUM})
#: The ``Convex.source`` of a compound unit's envelope.
UNIT_ENVELOPE = "UNIT_ENVELOPE"
NEG_INF = -math.inf
#: Boundary sampling of the overlap search (grid of a patch's unit square): surface faces, polygon faces, round faces, walls.
FACE_GRID = (24, 48)
POLYGON_FACE_GRID = (24, 24)
ROUND_FACE_GRID = (16, 48)
WALL_GRID = (96, 4)
#: A point this close outside an outline / face is on it (mm); normals closer than this to a plane are parallel to it.
ON_EDGE_MM = 1e-12
ON_FACE_MM = 1e-9
PARALLEL_EPS = 1e-12
#: The quick-rejection radius of a body: its exact reach from local Z with its largest sampled |z|, grown by this factor
#: and allowance.
REACH_MARGIN = 1.01
REACH_ALLOWANCE_MM = 1e-6
#: The z extent of a round solid with tilted faces: vertical lines through its centre and RIM_LINES rim points (just
#: inside the rim, by RIM_INSET of its radius), each VERTICAL_REACH_MM long both ways.
RIM_LINES = 64
RIM_INSET = 1e-9
VERTICAL_REACH_MM = 1e6
#: Share of a stop sheet's radial parameter that samples its central obstruction (the rest samples the plate).
OBSTRUCTION_SHARE = 0.25


def _f(value: Any) -> float:
    return float(Decimal(value)) if isinstance(value, str) else float(value)


def material_name(engine_or_model_id: str) -> str:
    """A medium id without its Engine namespace (``catalog:N-BK7`` / ``project:X`` -> ``N-BK7`` / ``X``; ``air`` -> ``AIR``) and,
    for a crystal as one element cuts it, without the cut (``project:CALCITE@X1`` -> ``CALCITE``; a material id has no ``@``)."""

    if engine_or_model_id.lower() == "air":
        return AIR
    if engine_or_model_id.lower() == "vacuum":
        return VACUUM
    name = engine_or_model_id.split(":", 1)[1] if ":" in engine_or_model_id else engine_or_model_id
    return name.split("@", 1)[0]


@dataclass(frozen=True)
class CoaxialSurface:
    """A surface of a surface sequence: vertex on local Z, its profile, clear aperture and, when it states one, its own
    mechanical ``edge`` (half widths; ``round_edge``: a circle of radius ``edge[0]``), beyond which it is held flat at its
    rim height (``held_point``). Without one it reaches the element outline."""

    surface_id: str
    vertex_z: float
    profile: Profile
    aperture: Circle | Polygon
    edge: tuple[float, float] | None = None
    round_edge: bool = True
    #: the exact clear region (an ``optics.apertures`` region, holes included) of a surface with a shaped clear aperture
    #: (ellipse, D shape, polygon) or holes; ``None``: ``aperture`` is exact
    region: Any = None

    def z(self, x: float, y: float) -> float | None:
        sag = self.profile.sag(*held_point(x, y, self.edge, self.round_edge))
        return None if sag is None else self.vertex_z + sag

    def profile_z(self, x: float, y: float) -> float | None:
        """The surface's own profile at ``(x, y)``, not held flat beyond its edge (``None`` outside its domain)."""

        sag = self.profile.sag(x, y)
        return None if sag is None else self.vertex_z + sag

    def in_clear(self, xs, ys, margin: float):
        """Whether the points (arrays) lie in the clear aperture (within ``margin``), outside its holes."""

        import numpy as np

        if self.region is not None:
            from ..optics.apertures import inside

            return inside(self.region, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), margin)
        return np.array([self.aperture.contains(float(x), float(y), margin=margin) for x, y in zip(xs, ys, strict=True)], dtype=bool)

    def cos_slope(self, x: float, y: float) -> float:
        hx, hy = held_point(x, y, self.edge, self.round_edge)
        if (hx, hy) != (x, y):  # the flat flange beyond the edge
            return 1.0
        n = self.profile.normal(x, y)
        return 1.0 if n is None else max(n[2], 0.0)

    @property
    def edge_outline(self) -> Circle | Polygon | None:
        """The surface's own edge as an outline (``None``: it reaches the element outline)."""

        if self.edge is None:
            return None
        return Circle(0.0, 0.0, self.edge[0]) if self.round_edge else Polygon.rectangle(0.0, 0.0, self.edge[0], self.edge[1])


def outline_margin(outline: Circle | Polygon, x: float, y: float) -> float:
    """Signed distance of ``(x, y)`` inside an outline (positive inside)."""

    if isinstance(outline, Circle):
        return outline.r - math.sqrt((x - outline.cx) ** 2 + (y - outline.cy) ** 2)
    worst = math.inf
    pts = outline.points
    for i in range(len(pts)):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % len(pts)]
        nx, ny = by - ay, ax - bx
        length = math.sqrt(nx * nx + ny * ny)
        worst = min(worst, ((nx * ax + ny * ay) - (nx * x + ny * y)) / length)
    return worst


def outline_point(outline: Circle | Polygon, u: float, v: float) -> tuple[float, float]:
    """A point of an outline's region for ``(u, v)`` in the unit square: polar (radius fraction, turn) for a circle, the
    bounding rectangle for a polygon."""

    if isinstance(outline, Circle):
        r, a = outline.r * u, 2.0 * math.pi * v
        return outline.cx + r * math.cos(a), outline.cy + r * math.sin(a)
    xs = [q[0] for q in outline.points]
    ys = [q[1] for q in outline.points]
    return min(xs) + (max(xs) - min(xs)) * u, min(ys) + (max(ys) - min(ys)) * v


@dataclass(frozen=True)
class Patch:
    """A boundary patch of a piece: ``point(u, v)`` for ``(u, v)`` in the unit square (``None`` where it is not on the
    piece), sampled on a ``grid`` x ``grid`` lattice by the overlap search."""

    point: Callable[[float, float], Vec3 | None]
    grid: tuple[int, int]
    where: str  # a surface id or a face description, for messages


@dataclass(frozen=True)
class Slab:
    """The medium between ``front`` and ``back`` (``front.z <= z <= back.z``) inside ``outline``."""

    medium: str
    front: CoaxialSurface
    back: CoaxialSurface
    outline: Circle | Polygon

    @property
    def rotational(self) -> bool:
        return (isinstance(self.outline, Circle) and self.outline.cx == 0.0 and self.outline.cy == 0.0
                and self.front.profile.rotational and self.back.profile.rotational)

    def depth(self, p: Vec3) -> float:
        x, y, z = p
        lo, hi = self.front.z(x, y), self.back.z(x, y)
        if lo is None or hi is None:
            return NEG_INF
        return min(outline_margin(self.outline, x, y), (z - lo) * self.front.cos_slope(x, y), (hi - z) * self.back.cos_slope(x, y))

    def _xy(self, u: float, v: float) -> tuple[float, float]:
        return outline_point(self.outline, u, v)

    def _perimeter(self, u: float) -> tuple[float, float]:
        if isinstance(self.outline, Circle):
            a = 2.0 * math.pi * u
            return self.outline.cx + self.outline.r * math.cos(a), self.outline.cy + self.outline.r * math.sin(a)
        pts = self.outline.points
        t = (u % 1.0) * len(pts)
        i = int(t)
        f = t - i
        (ax, ay), (bx, by) = pts[i % len(pts)], pts[(i + 1) % len(pts)]
        return ax + (bx - ax) * f, ay + (by - ay) * f

    def patches(self) -> list[Patch]:
        def on(surface: CoaxialSurface) -> Callable[[float, float], Vec3 | None]:
            def point(u: float, v: float) -> Vec3 | None:
                x, y = self._xy(u, v)
                if outline_margin(self.outline, x, y) < -ON_EDGE_MM:
                    return None
                z = surface.z(x, y)
                return None if z is None else (x, y, z)
            return point

        def wall(u: float, v: float) -> Vec3 | None:
            x, y = self._perimeter(u)
            lo, hi = self.front.z(x, y), self.back.z(x, y)
            if lo is None or hi is None:
                return None
            return (x, y, lo + (hi - lo) * v)

        return [Patch(on(self.front), FACE_GRID, self.front.surface_id), Patch(on(self.back), FACE_GRID, self.back.surface_id),
                Patch(wall, WALL_GRID, f"edge {self.front.surface_id}-{self.back.surface_id}")]

    def reach_xy(self) -> float:
        return self.outline.reach


@dataclass(frozen=True)
class Convex:
    """A convex solid: ``solid`` (half-spaces, optional side radius) filled with ``medium`` (``None``: a design body)."""

    solid: Solid
    medium: str | None
    source: str  # "MODEL_BODY" or "DESIGN_GEOMETRY"

    @property
    def rotational(self) -> bool:
        """A cylinder about local Z between two planes perpendicular to it."""

        return self.solid.side_radius is not None and all(abs(n[0]) < PARALLEL_EPS and abs(n[1]) < PARALLEL_EPS for n, _ in self.solid.planes)

    def depth(self, p: Vec3) -> float:
        d = min(off - (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) for n, off in self.solid.planes)
        if self.solid.side_radius is not None:
            d = min(d, self.solid.side_radius - math.sqrt(p[0] * p[0] + p[1] * p[1]))
        return d

    def z_range(self) -> tuple[float, float]:
        """The axial extent ``(z_min, z_max)`` of a rotational convex piece: its planes are ``n . x <= off``, so a plane
        with ``n_z < 0`` bounds it from below at ``z >= off / n_z`` and one with ``n_z > 0`` from above at ``z <= off / n_z``."""

        lo = max(off / n[2] for n, off in self.solid.planes if n[2] < 0.0)
        hi = min(off / n[2] for n, off in self.solid.planes if n[2] > 0.0)
        return lo, hi

    def reach_xy(self) -> float:
        """The largest distance of the piece from local Z (its side radius, or its farthest corner)."""

        if self.solid.side_radius is not None:
            return self.solid.side_radius
        return max((math.sqrt(v[0] * v[0] + v[1] * v[1]) for v in self.solid.vertices), default=0.0)

    def patches(self) -> list[Patch]:
        out = []
        solid = self.solid
        if solid.side_radius is not None:
            radius = solid.side_radius
            for index, (n, off) in enumerate(solid.planes):
                if abs(n[2]) < PARALLEL_EPS:
                    continue

                def face(u: float, v: float, n=n, off=off) -> Vec3 | None:
                    r, a = radius * u, 2.0 * math.pi * v
                    x, y = r * math.cos(a), r * math.sin(a)
                    p = (x, y, (off - n[0] * x - n[1] * y) / n[2])
                    return p if self.depth(p) >= -ON_FACE_MM else None
                out.append(Patch(face, ROUND_FACE_GRID, f"face {index + 1}"))

            def side(u: float, v: float) -> Vec3 | None:
                a = 2.0 * math.pi * u
                x, y = radius * math.cos(a), radius * math.sin(a)
                zs = [(off - n[0] * x - n[1] * y) / n[2] for n, off in solid.planes if abs(n[2]) >= PARALLEL_EPS]
                if len(zs) < 2:
                    return None
                lo, hi = min(zs), max(zs)
                return (x, y, lo + (hi - lo) * v)
            out.append(Patch(side, WALL_GRID, "side"))
            return out
        for index, (n, off) in enumerate(solid.planes):
            polygon = face_polygon(solid, index)
            if len(polygon) < 3:
                continue
            origin = polygon[0]
            e1 = _sub(polygon[1], origin)
            l1 = math.sqrt(_dot(e1, e1))
            if l1 == 0.0:
                continue
            e1 = (e1[0] / l1, e1[1] / l1, e1[2] / l1)
            e2 = _cross(n, e1)
            us = [_dot(_sub(q, origin), e1) for q in polygon]
            vs = [_dot(_sub(q, origin), e2) for q in polygon]
            u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)

            def face(u: float, v: float, origin=origin, e1=e1, e2=e2, u0=u0, u1=u1, v0=v0, v1=v1) -> Vec3 | None:
                a, b = u0 + (u1 - u0) * u, v0 + (v1 - v0) * v
                p = (origin[0] + a * e1[0] + b * e2[0], origin[1] + a * e1[1] + b * e2[1], origin[2] + a * e1[2] + b * e2[2])
                return p if self.depth(p) >= -ON_FACE_MM else None
            out.append(Patch(face, POLYGON_FACE_GRID, f"face {index + 1}"))
        return out


@dataclass(frozen=True)
class Plate:
    """An aperture stop's plate with a stated body: the design solid minus its opening (a prism along local Z through
    ``opening``), plus the central ``obstruction`` disc (the design solid inside it). Light passes the opening, so nothing
    in it except the obstruction is part of the stop."""

    solid: Solid
    opening: Circle | Polygon
    obstruction: Circle | None
    source: str = "DESIGN_GEOMETRY"

    @property
    def rotational(self) -> bool:
        centred = isinstance(self.opening, Circle) and self.opening.cx == 0.0 and self.opening.cy == 0.0
        return (Convex(self.solid, None, self.source).rotational and centred
                and (self.obstruction is None or (self.obstruction.cx == 0.0 and self.obstruction.cy == 0.0)))

    def _solid_depth(self, p: Vec3) -> float:
        return Convex(self.solid, None, self.source).depth(p)

    def depth(self, p: Vec3) -> float:
        inside = self._solid_depth(p)
        plate = min(inside, -outline_margin(self.opening, p[0], p[1]))
        if self.obstruction is None:
            return plate
        return max(plate, min(inside, outline_margin(self.obstruction, p[0], p[1])))

    def z_range(self) -> tuple[float, float]:
        return Convex(self.solid, None, self.source).z_range()

    def radial_span(self, rho: float) -> tuple[float, float] | None:
        """The axial span at radius ``rho`` of a rotational plate (``None`` in the open part of the opening)."""

        in_obstruction = self.obstruction is not None and rho <= self.obstruction.r
        if rho < self.opening.r and not in_obstruction:  # type: ignore[union-attr]
            return None
        return self.z_range()

    def _vertical(self, x: float, y: float) -> tuple[float, float] | None:
        """The z interval of the design solid on the line through ``(x, y)`` along local Z."""

        from ..optical_view.solids import clip_segment

        lo, hi = self._z_bounds()
        span = clip_segment(self.solid, (x, y, lo), (x, y, hi))
        if span is None:
            return None
        return lo + (hi - lo) * span[0], lo + (hi - lo) * span[1]

    def _z_bounds(self) -> tuple[float, float]:
        lo, hi = _solid_z_extent(self.solid)
        return lo - 1.0, hi + 1.0

    def _kept(self, x: float, y: float) -> bool:
        """A point of the solid's boundary that is also the plate's (not inside the opening, or on the obstruction)."""

        if outline_margin(self.opening, x, y) <= ON_EDGE_MM:
            return True
        return self.obstruction is not None and outline_margin(self.obstruction, x, y) >= -ON_EDGE_MM

    def patches(self) -> list[Patch]:
        out = []
        for patch in Convex(self.solid, None, self.source).patches():
            def face(u: float, v: float, patch=patch) -> Vec3 | None:
                p = patch.point(u, v)
                return p if p is not None and self._kept(p[0], p[1]) else None
            out.append(Patch(face, patch.grid, patch.where))
        for name, ring in (("opening", self.opening), ("obstruction", self.obstruction)):
            if ring is None:
                continue

            def wall(u: float, v: float, ring=ring) -> Vec3 | None:
                x, y = _perimeter_point(ring, u)
                span = self._vertical(x, y)
                return None if span is None else (x, y, span[0] + (span[1] - span[0]) * v)
            out.append(Patch(wall, WALL_GRID, f"{name} wall"))
        return out

    def reach_xy(self) -> float:
        return Convex(self.solid, None, self.source).reach_xy()


@dataclass(frozen=True)
class Sheet:
    """A surface with no modelled solid behind it: ``point(u, v)`` over its extent (zero thickness, no interior);
    ``xy_reach``: its largest distance from local Z."""

    surface_id: str
    point: Callable[[float, float], Vec3 | None]
    xy_reach: float = 0.0

    def depth(self, p: Vec3) -> float:
        return NEG_INF

    def patches(self) -> list[Patch]:
        return [Patch(self.point, FACE_GRID, self.surface_id)]

    def reach_xy(self) -> float:
        return self.xy_reach


Piece = Slab | Convex | Plate | Sheet


def _perimeter_point(outline: Circle | Polygon, u: float) -> tuple[float, float]:
    """The point of an outline's perimeter at fraction ``u`` of a turn (a circle) or of its corners (a polygon)."""

    if isinstance(outline, Circle):
        a = 2.0 * math.pi * u
        return outline.cx + outline.r * math.cos(a), outline.cy + outline.r * math.sin(a)
    pts = outline.points
    t = (u % 1.0) * len(pts)
    i = int(t)
    f = t - i
    (ax, ay), (bx, by) = pts[i % len(pts)], pts[(i + 1) % len(pts)]
    return ax + (bx - ax) * f, ay + (by - ay) * f


@dataclass(frozen=True)
class OpticalBody:
    """The optical solid of one element (or compound member) in its element frame.

    ``surfaces``: its coaxial surfaces (surface sequence) for the per-element checks; ``outline``: the element outline;
    ``media``: the sequence media; ``pieces``: what fills space; ``radius``: a bound of every point's distance from the
    element origin (for quick rejection)."""

    instance_id: str
    member_id: str | None
    model_kind: str
    pieces: tuple[Piece, ...]
    surfaces: tuple[CoaxialSurface, ...] = ()
    outline: Circle | Polygon | None = None
    media: tuple[str, ...] = ()
    #: per gap of a surface sequence, its outline (``SurfaceSequenceModelV1.gap_outline_mm``)
    gap_outlines: tuple[Circle | Polygon, ...] = ()
    posed: tuple[Any, ...] = ()  # posed published surfaces of a surface set (for the medium and domain checks)
    solid: Solid | None = None
    engine_surfaces: tuple[Any, ...] = ()
    radius: float = 0.0
    #: the element-frame box of the sampled boundary (the sampled search skips points well outside it)
    box: tuple[Vec3, Vec3] | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    #: ``(plane z, clear radius)`` of a thin element light is traced through (an ideal thin lens, a polarization element)
    thin_plane: tuple[float, float] | None = None
    #: ``(plane z, opening, obstruction, outline)`` of an aperture stop (its plane blocks light on the plate and the obstruction)
    stop_plane: tuple[float, Any, Any, Any] | None = None
    #: a catalog product's model (its plausibility limits are calibrated on the published catalog); else a declared model
    catalog: bool = False

    @property
    def label(self) -> str:
        return self.instance_id if self.member_id is None else f"{self.instance_id} member {self.member_id}"

    @property
    def rotational(self) -> bool:
        """Every piece is a solid of revolution about local Z (an exact coaxial overlap applies)."""

        return bool(self.pieces) and all(isinstance(p, (Slab, Convex, Plate)) and p.rotational for p in self.pieces)

    def depth(self, p: Vec3) -> float:
        return max((piece.depth(p) for piece in self.pieces), default=NEG_INF)

    def traced_surface_at(self, p: Vec3, tolerance: float) -> str | None:
        """The id of a surface light is traced through (or reflected at) whose clear aperture holds the element-frame point
        ``p`` within ``tolerance`` of the surface: a surface of the sequence, a posed surface of a surface set, the plane
        of an ideal thin lens or a polarization element inside its clear aperture (``None`` elsewhere -- an opaque rim, an
        unmodelled substrate face, a stop, whose opening is air)."""

        x, y, z = p
        for surface in self.surfaces:
            at = surface.z(x, y)
            if at is not None and abs(z - at) <= tolerance and surface.aperture.contains(x, y, margin=-tolerance):
                return surface.surface_id
        for surface in self.posed:
            origin = tuple(_f(v) for v in surface.origin_mm)
            normal = tuple(_f(v) for v in surface.normal)
            u = tuple(_f(v) for v in surface.u_axis)
            v = _cross(normal, u)  # type: ignore[arg-type]
            d = _sub(p, origin)  # type: ignore[arg-type]
            a, b, h = _dot(d, u), _dot(d, v), _dot(d, normal)  # type: ignore[arg-type]
            sag = _profile(surface).sag(a, b)
            if sag is not None and abs(h - sag) <= tolerance and _aperture(surface).contains(a, b, margin=-tolerance):
                return surface.surface_id
        if self.thin_plane is not None:
            plane_z, clear = self.thin_plane
            if abs(z - plane_z) <= tolerance and x * x + y * y <= (clear + tolerance) ** 2:
                return "plane"
        if self.stop_plane is not None:  # light meets a stop's plane where it is blocked: its plate and its obstruction
            plane_z, opening, obstruction, outline = self.stop_plane
            if abs(z - plane_z) <= tolerance and outline_margin(outline, x, y) >= -tolerance:
                if obstruction is not None and outline_margin(obstruction, x, y) >= -tolerance:
                    return "obstruction"
                if outline_margin(opening, x, y) <= tolerance:
                    return "plate"
        return None

    def axis_crossings(self) -> list[tuple[str, float]]:
        """``(surface id, local z)`` where local Z meets the element's traced surfaces inside their clear apertures (the
        axial points a beam along the element's axis passes)."""

        out = [(s.surface_id, s.vertex_z) for s in self.surfaces if s.aperture.contains(0.0, 0.0, margin=-ON_EDGE_MM)]
        if self.thin_plane is not None:
            out.append(("plane", self.thin_plane[0]))
        if self.stop_plane is not None and self.traced_surface_at((0.0, 0.0, self.stop_plane[0]), ON_EDGE_MM) is not None:
            out.append(("obstruction", self.stop_plane[0]))
        return out


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# ------------------------------------------------------------------------------------------------ construction


def _profile(surface: Any) -> Profile:
    return Profile(curvature=0.0 if surface.radius_mm is None else 1.0 / _f(surface.radius_mm),
                   conic=0.0 if surface.conic is None else _f(surface.conic),
                   coefficients=tuple(_f(a) for a in surface.aspheric_coefficients or ()), cylinder_axis=surface.cylinder_axis,
                   linear=0.0 if getattr(surface, "linear_sag_coefficient", None) is None else _f(surface.linear_sag_coefficient),
                   form=None if getattr(surface, "sag", None) is None else sag_form(surface.sag))


def _clear_region(surface: Any) -> Any:
    """The exact clear region of a surface with a ``clear_shape`` or ``holes`` (``optics.apertures``), else ``None``."""

    shape, holes = getattr(surface, "clear_shape", None), getattr(surface, "holes", None)
    if shape is None and not holes:
        return None
    from ..local.compile import clear_shape_region, hole_region
    from ..scene.model import ApertureG

    cx, cy = (_f(surface.aperture_center_mm[0]), _f(surface.aperture_center_mm[1])) if surface.aperture_center_mm else (0.0, 0.0)
    if shape is not None:
        region = clear_shape_region(shape, (cx, cy), surface.clear_radius_mm, surface.clear_half_widths_mm)
    elif surface.clear_radius_mm is not None:
        region = ApertureG("circle", _f(surface.clear_radius_mm), 0.0, cx, cy)
    else:
        region = ApertureG("rectangle", _f(surface.clear_half_widths_mm[0]), _f(surface.clear_half_widths_mm[1]), cx, cy)
    return replace(region, holes=tuple(hole_region(h) for h in holes or ()))


def _aperture(surface: Any) -> Circle | Polygon:
    cx, cy = (_f(surface.aperture_center_mm[0]), _f(surface.aperture_center_mm[1])) if surface.aperture_center_mm else (0.0, 0.0)
    if surface.clear_radius_mm is not None:
        return Circle(cx, cy, _f(surface.clear_radius_mm))
    hx, hy = (_f(v) for v in surface.clear_half_widths_mm)
    return Polygon.rectangle(cx, cy, hx, hy)


def _hull_planes(points: list[Vec3]) -> tuple[tuple[Vec3, float], ...]:
    """Outward planes of the convex hull of ``points`` (faces through three of them with every point on one side)."""

    planes: list[tuple[Vec3, float]] = []
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                n = _cross(_sub(points[j], points[i]), _sub(points[k], points[i]))
                length = math.sqrt(_dot(n, n))
                if length <= PARALLEL_EPS:
                    continue
                n = (n[0] / length, n[1] / length, n[2] / length)
                off = _dot(n, points[i])
                sides = [_dot(n, q) - off for q in points]
                for sign in (1.0, -1.0):
                    if all(s * sign <= ON_FACE_MM for s in sides) and any(s * sign < -ON_FACE_MM for s in sides):
                        plane = ((n[0] * sign, n[1] * sign, n[2] * sign), off * sign)
                        if not any(max(abs(plane[0][a] - q[0][a]) for a in range(3)) <= ON_FACE_MM and abs(plane[1] - q[1]) <= ON_FACE_MM for q in planes):
                            planes.append(plane)
    return tuple(planes)


def design_solid(geometry: Any) -> Solid | None:
    """A design geometry (``CylinderGeometryV1`` / ``BoxGeometryV1`` / ``ConvexPolyhedronGeometryV1``, or the scene's
    ``OPTIC_BODY`` shape fields) as a float convex solid."""

    if geometry is None:
        return None
    kind = getattr(geometry, "kind", None) or getattr(geometry, "shape", None)
    if kind == "CYLINDER":
        z0, z1, r = _f(geometry.z_min_mm), _f(geometry.z_max_mm), _f(geometry.diameter_mm) / 2.0
        return solid_from_planes("DESIGN", (((0.0, 0.0, -1.0), -z0), ((0.0, 0.0, 1.0), z1)), r)
    if kind == "BOX":
        c = tuple(_f(v) for v in geometry.center_mm)
        h = tuple(_f(v) for v in geometry.half_extents_mm)
        planes = []
        for axis in range(3):
            for sign in (1.0, -1.0):
                n = [0.0, 0.0, 0.0]
                n[axis] = sign
                planes.append((tuple(n), sign * c[axis] + h[axis]))
        return solid_from_planes("DESIGN", tuple(planes), None)  # type: ignore[arg-type]
    if kind == "CONVEX_POLYHEDRON":
        points = [tuple(_f(c) for c in v) for v in geometry.vertices_mm]
        return Solid("DESIGN", _hull_planes(points), None, tuple(sorted(points)))  # type: ignore[arg-type]
    return None


def _design_z_range(geometry: Any) -> tuple[float, float] | None:
    solid = design_solid(geometry)
    if solid is None:
        return None
    if solid.side_radius is None and not solid.vertices:
        return None
    return _solid_z_extent(solid)


def envelope_body(instance_id: str, envelope: Any) -> OpticalBody | None:
    """A compound unit's envelope (a design geometry in the unit frame) as a body: its whole-unit boundary, which only the
    unit's members may occupy (``None`` without a geometry)."""

    solid = design_solid(envelope)
    if solid is None:
        return None
    pieces: tuple[Piece, ...] = (Convex(solid, None, UNIT_ENVELOPE),)
    radius, box = _radius_bound(pieces)
    return OpticalBody(instance_id, None, "COMPOUND", pieces, radius=radius, box=box)


def _radius_bound(pieces: tuple[Piece, ...]) -> tuple[float, tuple[Vec3, Vec3] | None]:
    """A bound of every point's distance from the element origin: the pieces' largest distance from local Z (exact: an
    outline's or solid's farthest corner or rim) combined with the largest |z| of their sampled boundary points; and the
    element-frame box of those points (``None`` without any)."""

    xy, z = 0.0, 0.0
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for piece in pieces:
        xy = max(xy, piece.reach_xy())
        for patch in piece.patches():
            nu, nv = patch.grid
            for i in range(nu + 1):
                for j in range(nv + 1):
                    p = patch.point(i / nu, j / nv)
                    if p is not None:
                        xy = max(xy, math.sqrt(p[0] * p[0] + p[1] * p[1]))
                        z = max(z, abs(p[2]))
                        for k in range(3):
                            lo[k], hi[k] = min(lo[k], p[k]), max(hi[k], p[k])
    box = None if lo[0] > hi[0] else ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))
    return math.sqrt(xy * xy + z * z) * REACH_MARGIN + REACH_ALLOWANCE_MM, box


def _centre_of(polygon: Polygon) -> tuple[float, float]:
    xs = [p[0] for p in polygon.points]
    ys = [p[1] for p in polygon.points]
    return 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))


def _stop_opening(model: Any) -> tuple[Circle | Polygon, Circle | None]:
    """An aperture stop's opening (a circle or a slit, centred on the axis or at ``opening_center_mm``) and its central
    obstruction disc (``None`` without one)."""

    centre = getattr(model, "opening_center_mm", None)
    cx, cy = (_f(centre[0]), _f(centre[1])) if centre is not None else (0.0, 0.0)
    radius, half = model.opening_radius_mm, model.opening_half_widths_mm
    opening: Circle | Polygon = (Circle(cx, cy, _f(radius)) if radius is not None
                                 else Polygon.rectangle(cx, cy, _f(half[0]), _f(half[1])))
    obstruction = getattr(model, "obstruction_radius_mm", None)
    return opening, (Circle(cx, cy, _f(obstruction)) if obstruction is not None else None)


def _solid_z_extent(solid: Solid) -> tuple[float, float]:
    """The smallest and largest z of a convex solid: a polyhedron's corners; a round solid bounded by planes normal to
    local Z: its z range; any other round solid: its extent on vertical lines through the centre and the rim."""

    if solid.side_radius is None:
        zs = [v[2] for v in solid.vertices]
        return (min(zs), max(zs)) if zs else (0.0, 0.0)
    convex = Convex(solid, None, "DESIGN_GEOMETRY")
    if convex.rotational:
        return convex.z_range()
    from ..optical_view.solids import clip_segment

    r = solid.side_radius * (1.0 - RIM_INSET)
    big = VERTICAL_REACH_MM
    lo, hi = math.inf, -math.inf
    for k in range(-1, RIM_LINES):
        a = 2.0 * math.pi * k / RIM_LINES
        x, y = (0.0, 0.0) if k < 0 else (r * math.cos(a), r * math.sin(a))
        span = clip_segment(solid, (x, y, -big), (x, y, big))
        if span is not None:
            lo, hi = min(lo, -big + 2.0 * big * span[0]), max(hi, -big + 2.0 * big * span[1])
    return (lo, hi) if lo <= hi else (0.0, 0.0)


def _plane_sheet(model: Any, outline: Circle | Polygon, plane_z: float) -> Sheet:
    """A thin element of zero thickness in its plane: an aperture stop's plate from its opening to its outline (the opening
    is air, so nothing in it is part of the stop) and its central obstruction; any other thin element's whole outline."""

    opening_radius = getattr(model, "opening_radius_mm", None)
    opening_half = getattr(model, "opening_half_widths_mm", None)
    if opening_radius is None and opening_half is None:
        def whole(u: float, v: float) -> Vec3:
            x, y = outline_point(outline, u, v)
            return (x, y, plane_z)
        kind = getattr(model, "model_kind", "")
        name = {"POLARIZATION_ELEMENT": "element", "NEUTRAL_DENSITY": "attenuator", "FARADAY_ROTATOR": "rotator", "DIFFRACTION_GRATING": "grating"}.get(kind, "lens")
        return Sheet(name, whole, outline.reach)
    opening, obstruction_disc = _stop_opening(model)
    cx, cy = (opening.cx, opening.cy) if isinstance(opening, Circle) else _centre_of(opening)
    r_obs = obstruction_disc.r if obstruction_disc is not None else 0.0

    def plate(u: float, v: float) -> Vec3:
        a = 2.0 * math.pi * v
        dx, dy = math.cos(a), math.sin(a)
        if r_obs > 0.0 and u < OBSTRUCTION_SHARE:  # the first part of the parameter range draws the obstruction disc
            t = r_obs * u / OBSTRUCTION_SHARE
            return (cx + dx * t, cy + dy * t, plane_z)
        w = (u - OBSTRUCTION_SHARE) / (1.0 - OBSTRUCTION_SHARE) if r_obs > 0.0 else u
        lo, hi = opening.exit(cx, cy, dx, dy), outline.exit(cx, cy, dx, dy)
        t = lo + (max(hi, lo) - lo) * w
        return (cx + dx * t, cy + dy * t, plane_z)

    return Sheet("stop", plate, outline.reach)


def body_of(element: Any, *, member_id: str | None = None, design_geometry: Any = None, instance_id: str | None = None) -> OpticalBody:
    """The optical body of a compiled element (``.instance_id``, ``.model``, ``.surfaces``, ``.solid``; nothing about its
    catalog identity is read). ``design_geometry``: the physical body of an element whose model has no solid (the
    substrate of a single coated surface, a polarization element's plate, an off-axis mirror's blank)."""

    model = element.model
    kind = model.model_kind
    name = instance_id or element.instance_id
    engine = tuple(getattr(element, "surfaces", ()) or ())
    if kind == "SURFACE_SEQUENCE":
        outline: Circle | Polygon = (Circle(0.0, 0.0, _f(model.edge_radius_mm)) if model.edge_radius_mm is not None
                                     else Polygon.rectangle(0.0, 0.0, _f(model.edge_half_widths_mm[0]), _f(model.edge_half_widths_mm[1])))
        round_edge = model.edge_radius_mm is not None
        own_edges = any(getattr(s, "has_own_edge", False) for s in model.surfaces)

        def edge_of(index: int, s: Any) -> tuple[float, float] | None:
            if not getattr(s, "has_own_edge", False):
                return None
            hx, hy = model.surface_edge_mm(index)
            return float(hx), float(hy)

        def gap_outline(k: int) -> Circle | Polygon:
            if not own_edges:  # every surface reaches the outline
                return outline
            hx, hy = model.gap_outline_mm(k)
            return Circle(0.0, 0.0, float(hx)) if round_edge else Polygon.rectangle(0.0, 0.0, float(hx), float(hy))

        surfaces = tuple(CoaxialSurface(s.surface_id, _f(s.vertex_z_mm), _profile(s), _aperture(s), edge_of(i, s), round_edge,
                                        _clear_region(s))
                         for i, s in enumerate(model.surfaces))
        gaps = tuple(gap_outline(k) for k in range(len(model.media)))
        pieces: list[Piece] = [Slab(m, surfaces[k], surfaces[k + 1], gaps[k]) for k, m in enumerate(model.media) if m not in SURROUNDINGS]
        if len(surfaces) == 1:
            (surface,) = surfaces
            span = _design_z_range(design_geometry)
            if span is not None:  # the unmodelled substrate: from the surface to the far face of the design body
                reflects = str(getattr(model.surfaces[0].traced, "value", model.surfaces[0].traced)) == "REFLECTED"
                far = substrate_face_z(surface.vertex_z, span[0], span[1], reflects)
                flat = CoaxialSurface("substrate", far, Profile(), surface.aperture)
                below = reflects or far < surface.vertex_z
                pieces.append(Slab("SUBSTRATE", flat, surface, outline) if below else Slab("SUBSTRATE", surface, flat, outline))
            else:
                def sheet(u: float, v: float, s=surface) -> Vec3 | None:
                    x, y = outline_point(outline, u, v)
                    z = s.z(x, y)
                    return None if z is None else (x, y, z)
                pieces.append(Sheet(surface.surface_id, sheet, outline.reach))
        body = OpticalBody(name, member_id, kind, tuple(pieces), surfaces=surfaces, outline=outline, media=tuple(model.media),
                           engine_surfaces=engine, gap_outlines=gaps, catalog=bool(getattr(getattr(element, "identity", None), "is_catalog", False)))
    elif kind == "SURFACE_SET":
        solid = getattr(element, "solid", None)
        pieces = []
        if solid is not None:
            pieces.append(Convex(solid, solid.medium, "MODEL_BODY"))
        else:
            design = design_solid(design_geometry)
            if design is not None:
                pieces.append(Convex(design, None, "DESIGN_GEOMETRY"))
        body = OpticalBody(name, member_id, kind, tuple(pieces), posed=tuple(model.surfaces), solid=solid, engine_surfaces=engine)
    else:  # a thin element (a polarization element's plane, a stop, an ideal thin lens): its design plate (a stop's open
        # where light passes), else its plane
        design = design_solid(design_geometry)
        plane_z = _f(getattr(model, "plane_z_mm", "0") or "0")
        edge = getattr(model, "edge_radius_mm", None)
        half = getattr(model, "edge_half_widths_mm", None)
        outline_shape: Circle | Polygon | None = (Circle(0.0, 0.0, _f(edge)) if edge is not None
                                                  else Polygon.rectangle(0.0, 0.0, _f(half[0]), _f(half[1])) if half is not None else None)
        pieces = []
        z_extent = _solid_z_extent(design) if design is not None else None
        if design is not None and z_extent is not None and z_extent[1] > z_extent[0]:
            if getattr(model, "opening_radius_mm", None) is not None or getattr(model, "opening_half_widths_mm", None) is not None:
                opening, obstruction = _stop_opening(model)  # an aperture stop's plate: open where light passes
                pieces.append(Plate(design, opening, obstruction))
            else:
                pieces.append(Convex(design, None, "DESIGN_GEOMETRY"))
        elif outline_shape is not None:
            pieces.append(_plane_sheet(model, outline_shape, plane_z))
        clear = getattr(model, "clear_radius_mm", None)
        is_stop = getattr(model, "opening_radius_mm", None) is not None or getattr(model, "opening_half_widths_mm", None) is not None
        traced = None if clear is None or is_stop else (plane_z, _f(clear))
        stop_plane = None
        if is_stop and outline_shape is not None:
            opening, obstruction = _stop_opening(model)
            stop_plane = (plane_z, opening, obstruction, outline_shape)
        body = OpticalBody(name, member_id, kind, tuple(pieces), outline=outline_shape, engine_surfaces=engine, thin_plane=traced,
                           stop_plane=stop_plane)
    radius, box = _radius_bound(body.pieces)
    return replace(body, radius=radius, box=box)
