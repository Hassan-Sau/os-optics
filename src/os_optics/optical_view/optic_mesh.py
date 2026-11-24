"""Display meshes of ``OPTIC_BODY`` elements, one group per optical surface (element frame, mm; presentation only).

Every surface is drawn on its exact profile within ``surface_mesh.CHORD_TOLERANCE_MM`` (``surface_mesh``), split the way
the tracer models it:

* its clear aperture in the surface's material (glass, mirror, splitter or filter face);
* a narrow band just outside the clear aperture (``aperture_outline``), so the clear-aperture outline reads in any glTF
  viewer;
* the opaque rim between the clear aperture and the outline (``optic_rim``): the tracer blocks light there
  (``EDGE_AS_OPAQUE_RIM``); beyond the outline light passes by;
* a surface with its own mechanical edge is drawn on its profile out to that edge, then as its flat flange
  (``optic_edge``, held at its rim height) out to the outline of the glass it bounds.

A profile is never extended or flattened beyond its domain: where an outline reaches past it, the vertices are held at
the domain edge and those triangles are drawn in ``profile_undefined`` (and the surface says so in its facts); the
optical-solid check refuses such a model before any run.

Bodies:

* a surface sequence: every surface over the element's round or rectangular outline (a surface with its own edge: over
  its edge and flange, out to its glass outline); the glass between consecutive surfaces closed by its edge wall at the
  gap's outline (an air gap stays open); a single coated surface's unmodelled substrate is the rest
  of the design body (a flat face and a wall);
* a surface set: a surface on a solid face is that face (its clear aperture, outline band and rim), an internal surface
  is the solid's section in its plane, a surface without a solid (an off-axis mirror face) is its clear aperture; faces
  carrying no traced surface and a round plate's side wall in ``optic_edge``;
* a polarization element: its thin design body and a bar along its local +X axis, plus its Jones plane (clear aperture
  and rim);
* a surface with holes through the element or a non-convex clear outline: its regions triangulated (``region_mesh``): the
  clear aperture minus the holes, and the rim out to the outline row the edge walls meet; the holes are open.

Pure Python: the optical view does not import NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .._vendor.shared_contracts.design_catalog import AIR, CoatingKind
from .._vendor.shared_contracts.design_catalog.optical_catalog import RESERVED_MEDIA
from .._vendor.shared_contracts.optical_view import OpticBodyGeometryV1, OpticSurfaceRoleV1, OpticSurfaceV1
from . import geometry as g
from . import region_mesh
from . import surface_mesh as sm
from .solids import ROUND_SEGMENTS, Solid, face_polygon, section_polygon, solid_from_planes

# A surface lies on a solid face when its vertex is this close to the face plane and its normal is parallel to the face's.
ON_FACE_TOLERANCE_MM = 1e-6
PARALLEL_TOLERANCE = 1e-6
PLAIN_COATINGS = frozenset({CoatingKind.ANTI_REFLECTION})
#: Width of the clear-aperture outline band, as a fraction of the outline's reach (at most a quarter of the rim width).
OUTLINE_BAND_FRACTION = 0.012
OUTLINE_BAND_RIM_SHARE = 0.25
#: Directions along which a zone that is not one exact surface of revolution is sampled for its stations.
PROBE_DIRECTIONS = 16
#: A surface checked against its profile states the largest deviation it reached (mm, to this many decimals); refining stops
#: before one surface would exceed MAX_SURFACE_VERTICES -- when doubling would, the rest of that budget goes to directions
#: once -- and a surface still outside the tolerance then states chord_tolerance_met: false with its deviation.
DEVIATION_DIGITS = 9
MAX_SURFACE_VERTICES = 60000
#: Radii at which the starting number of directions of a surface that is not one exact surface of revolution is estimated.
SURFACE_TERM_SAMPLES = 64
#: The local +X axis bar of a waveplate / polarizer (mm).
AXIS_BAR_HALF_WIDTH_MM = 0.4
AXIS_BAR_LIFT_MM = 0.05
#: The rules drawn across a grating's clear aperture (a sign of its groove direction, not its line count), the share of their
#: spacing each covers and how far above the plane they are drawn (mm).
GRATING_RULES = 15
GRATING_RULE_SHARE = 0.3
GRATING_RULE_LIFT_MM = 0.02
GRATING_RULE_HEIGHT_MM = 0.04
#: How far a Faraday rotator's field arrow reaches beyond its body (a fraction of its edge radius), and the blocks of its head.
FIELD_ARROW_REACH = 0.25
FIELD_ARROW_STEPS = 3

RIM_MATERIAL = "optic_rim"
STOP_MATERIAL = "stop_plate"
OUTLINE_MATERIAL = "aperture_outline"

Material = str


#: The GLB material of each optical surface role; ``None``: a refracting surface, glass or a filter face by its coating
#: (``surface_material``).
ROLE_MATERIALS: dict[str, Material | None] = {
    "REFRACT": None, "REFLECT": "mirror_face", "SPLIT": "splitter_face", "POLARIZATION_ELEMENT": "polarization_plane",
    "APERTURE_STOP": STOP_MATERIAL, "THIN_LENS": "thin_lens", "ATTENUATOR": "attenuator_plane", "FARADAY_ROTATOR": "faraday_plane", "GRATING": "grating_face",
}
#: The materials of a refracting surface: plain glass (uncoated or anti-reflection) and a coated filter face.
REFRACT_MATERIALS = ("optic_glass", "filter_face")


def surface_material(surface: OpticSurfaceV1) -> Material:
    material = ROLE_MATERIALS[surface.role.value]
    if material is not None:
        return material
    return REFRACT_MATERIALS[0] if surface.coating_kind is None or surface.coating_kind in PLAIN_COATINGS else REFRACT_MATERIALS[1]


def is_plain(geom: OpticBodyGeometryV1) -> bool:
    """A round coaxial element (a polarization element, or a surface sequence of centred planes / spheres with circular
    apertures) in a cylindrical design body: every surface is a surface of revolution about local Z."""

    if geom.model_kind == "SURFACE_SET" or geom.shape != "CYLINDER" or geom.edge_radius_mm is None or geom.has_surface_edges:
        return False
    return all(s.clear_radius_mm is not None and s.aperture_center_mm is None and (s.is_plane or s.is_spherical) for s in geom.surfaces)


@dataclass
class SurfaceDrawing:
    """One optical surface as drawn: its meshes by material (element frame) and the facts a viewer shows."""

    surface: OpticSurfaceV1
    meshes: list[tuple[Material, g.Mesh]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyDrawing:
    """An ``OPTIC_BODY`` as drawn: one ``SurfaceDrawing`` per optical surface (model order) and the body meshes that are
    not optical surfaces (edge walls, substrate, untraced solid faces, a polarization element's body and axis bar)."""

    surfaces: list[SurfaceDrawing] = field(default_factory=list)
    body: list[tuple[Material, g.Mesh]] = field(default_factory=list)
    tolerance_mm: float = sm.CHORD_TOLERANCE_MM

    def meshes(self) -> list[tuple[Material, g.Mesh]]:
        return [m for s in self.surfaces for m in s.meshes] + list(self.body)


# --------------------------------------------------------------------------------------------- zone construction


def _aperture(surface: OpticSurfaceV1) -> sm.Boundary:
    """The clear aperture as a star-shaped boundary: its outline ring when it states a convex one (an ellipse, a D shape, a
    convex polygon), else its circle or rectangle (a non-convex outline and holes are drawn as regions, ``_draw_regions``)."""

    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    ring = surface.clear_outline_mm
    if ring is not None and _convex(ring):
        return sm.Polygon(tuple((float(x), float(y)) for x, y in ring))
    if surface.clear_radius_mm is not None:
        return sm.Circle(cx, cy, surface.clear_radius_mm)
    hx, hy = surface.clear_half_widths_mm  # type: ignore[misc]
    return sm.Polygon.rectangle(cx, cy, hx, hy)


def _convex(ring) -> bool:
    """Is the counter-clockwise ring convex (no right turn)?"""

    count = len(ring)
    for i in range(count):
        (x0, y0), (x1, y1), (x2, y2) = ring[i - 1], ring[i], ring[(i + 1) % count]
        if (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) < 0.0:
            return False
    return True


def _exits(boundary: sm.Boundary, centre: sm.Vec2, dirs: list[sm.Vec2]) -> tuple[float, ...]:
    return tuple(boundary.exit(centre[0], centre[1], dx, dy) for dx, dy in dirs)


def _concentric(a: sm.Boundary, centre: sm.Vec2) -> bool:
    return isinstance(a, sm.Circle) and a.cx == centre[0] and a.cy == centre[1]


def _zone_stations(profile: sm.Profile, centre: sm.Vec2, dirs: list[sm.Vec2], inner: tuple[float, ...] | None, outer: tuple[float, ...],
                   budget: float) -> list[float]:
    """Station fractions of one zone of a surface that is not one exact surface of revolution: the profile sampled along
    a spread of directions and the longest one so its chords stay within ``budget`` (the deviation check refines
    whatever they miss)."""

    if profile.is_flat:
        return [1.0]
    cx, cy = centre
    lists = []
    step = max(1, len(dirs) // PROBE_DIRECTIONS)
    longest = max(range(len(dirs)), key=lambda j: outer[j] - (0.0 if inner is None else inner[j]))
    for j in sorted(set(range(0, len(dirs), step)) | {longest}):
        dx, dy = dirs[j]
        lo = 0.0 if inner is None else inner[j]
        hi = outer[j]
        if hi - lo <= sm.SAME_POINT_MM:
            continue
        stations = sm.profile_stations(lambda t, dx=dx, dy=dy: profile.sag(cx + dx * t, cy + dy * t), lo, hi, budget)
        lists.append(sm.fractions(stations))
    # the densest direction's stations (a union of every direction's stations would multiply the rows; the deviation check
    # refines what this misses)
    return max(lists, key=len) if lists else [1.0]


def _radial_stations(profile: sm.Profile, rho0: float, rho1: float, budget: float) -> list[float]:
    """Transverse radii ``rho0 .. rho1`` of the rows of a surface of revolution: equal arc angles on a sphere, the
    chord sampler on any other profile, none in between on a plane."""

    if profile.is_flat or rho1 <= rho0:
        return [rho0, rho1]
    if not profile.coefficients and profile.conic == 0.0 and profile.linear == 0.0 and profile.rotational:
        return sm.sphere_stations(1.0 / profile.curvature, rho0, rho1, budget)
    return sm.profile_stations(lambda t: profile.sag_rho_sq(t * t), rho0, rho1, budget)


def _ring_directions(profile: sm.Profile, rho: float, budget: float) -> int:
    """Directions a ring of radius ``rho`` of a surface of revolution needs so the chord between two neighbours stays
    within ``budget`` of the surface: ``rho (1 - cos(pi / n)) sin(slope) <= rho (pi / n)^2 / 2 sin(slope) <= budget``."""

    if profile.is_flat or rho <= 0.0:
        return sm.MIN_DIRECTIONS
    a, b = profile.domain_point(rho, 0.0)
    n = profile.normal(a, b)
    sine = 0.0 if n is None else math.sqrt(n[0] * n[0] + n[1] * n[1])
    if sine <= 0.0:
        return sm.MIN_DIRECTIONS
    return math.ceil(math.pi * math.sqrt(rho * sine / (2.0 * budget)))


def _surface_term(profile: sm.Profile, centre: sm.Vec2, reach: float, budget: float) -> int:
    """Directions a surface that is not one exact surface of revolution needs to start with (the deviation check then
    refines): the ring bound at sampled radii along the local axes."""

    if profile.is_flat:
        return sm.MIN_DIRECTIONS
    worst = 0.0
    samples = SURFACE_TERM_SAMPLES
    for k in range(1, samples + 1):
        rho = reach * k / samples
        for dx, dy in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
            a, b = profile.domain_point(centre[0] + dx * rho, centre[1] + dy * rho)
            n = profile.normal(a, b)
            if n is None:
                continue
            worst = max(worst, rho * math.sqrt(n[0] * n[0] + n[1] * n[1]))
    if worst <= 0.0:
        return sm.MIN_DIRECTIONS
    return math.ceil(math.pi * math.sqrt(worst / (2.0 * budget)))


@dataclass
class _Plan:
    """What one surface needs to be drawn: its profile, placement, centre, boundaries and facing; for an exact surface of
    revolution also its zones as radii (``radial``: per zone, material and row radii) and each row's direction need."""

    surface: OpticSurfaceV1
    profile: sm.Profile
    placement: sm.Placement
    centre: sm.Vec2
    aperture: sm.Boundary
    outline: sm.Boundary | None
    facing: float
    facts: dict[str, Any]
    exact: bool = False  # within the tolerance by construction (a plane, or a surface of revolution in concentric circles)
    rotational: bool = False  # a surface of revolution in a concentric round aperture and outline: drawn by radii
    split: bool = True
    refine: int = 0
    radial: list[tuple[str, list[float]]] = field(default_factory=list)
    needs: list[int] = field(default_factory=list)
    #: rings beyond the outline (a surface's own edge) out to its glass outline, inside out: its flat flange zones
    flanges: list[sm.Boundary] = field(default_factory=list)
    #: rows that use every direction (the edge and flange rings an edge wall meets)
    full_rows: set[int] = field(default_factory=set)


def _band_width(inner: float, outer: float, reach: float) -> float:
    return min(OUTLINE_BAND_FRACTION * reach, OUTLINE_BAND_RIM_SHARE * (outer - inner))


def _plan_radial(plan: _Plan, tolerance: float) -> None:
    """Zones and rows of an exact surface of revolution (concentric round aperture and outline, or a plane in them)."""

    budget_m = tolerance * sm.MERIDIONAL_SHARE
    budget_a = tolerance * (1.0 - sm.MERIDIONAL_SHARE)
    material = surface_material(plan.surface)
    ca = plan.aperture.r if isinstance(plan.aperture, sm.Circle) else 0.0
    edges: list[tuple[str, float, float]] = []
    if plan.outline is None:
        edges.append((material, 0.0, ca))
    else:
        out = plan.outline.r  # type: ignore[union-attr]
        if not plan.split:
            edges.append((material, 0.0, out))
        else:
            ca = min(ca, out)
            edges.append((material, 0.0, ca))
            if out - ca > sm.SAME_POINT_MM:
                band = ca + _band_width(ca, out, out)
                edges += [(OUTLINE_MATERIAL, ca, band), (RIM_MATERIAL, band, out)]
    base = len(edges)
    for flange in plan.flanges:
        edges.append((sm.FLANGE_MATERIAL, edges[-1][2], flange.r))  # type: ignore[union-attr]
    plan.radial, plan.needs, plan.full_rows = [], [], set()
    for index, (name, lo, hi) in enumerate(edges):
        flange = index >= base
        radii = [hi] if flange else _radial_stations(plan.profile, lo, hi, budget_m)[1:]
        plan.radial.append((name, [lo, *radii]))
        for k, rho in enumerate(radii):
            need = g.circle_segments(rho, tolerance) if flange else _ring_directions(plan.profile, rho, budget_a)
            if k == len(radii) - 1:  # a zone boundary is a drawn circle: within the tolerance of it
                need = max(need, g.circle_segments(rho, tolerance))
                if plan.flanges and index >= base - 1:  # the edge and flange rings an edge wall meets
                    plan.full_rows.add(len(plan.needs))
            plan.needs.append(need)


def _irregular(surface: OpticSurfaceV1) -> bool:
    """Whether a surface's drawn regions are not star-shaped about its centre: it has holes, or a non-convex clear outline."""

    return bool(surface.hole_outlines_mm) or (surface.clear_outline_mm is not None and not _convex(surface.clear_outline_mm))


def _draw_regions(plan: _Plan, dirs: list[sm.Vec2], tolerance: float) -> sm.SurfaceMesh | None:
    """A surface with holes or a non-convex clear outline (``region_mesh``): its clear aperture (outline minus holes) in its
    material and, within the glass outline, the rim; the outline row along ``dirs`` as the star mesh draws it (the edge walls
    meet it). ``None`` where the profile is undefined on a region (the star mesh draws it then, holes stated only)."""

    surface, centre = plan.surface, plan.centre
    cx, cy = centre
    if surface.clear_outline_mm is not None:
        clear = [(float(x), float(y)) for x, y in surface.clear_outline_mm]
    else:
        clear = [(cx + dx * t, cy + dy * t) for (dx, dy), t in zip(dirs, _exits(plan.aperture, centre, dirs), strict=True)]
    holes = [[(float(x), float(y)) for x, y in ring] for ring in surface.hole_outlines_mm or ()]
    regions: list = [(surface_material(surface), clear, holes)]
    outer: list[sm.Vec2] = []
    if plan.outline is not None and plan.split:
        outer = [(cx + dx * t, cy + dy * t) for (dx, dy), t in zip(dirs, _exits(plan.outline, centre, dirs), strict=True)]
        regions.append((RIM_MATERIAL, outer, [clear]))
    try:
        drawn = region_mesh.region_meshes(plan.profile, plan.placement, regions, plan.facing, tolerance, MAX_SURFACE_VERTICES)
    except ValueError:  # (rings that touch: the star mesh draws it)
        return None
    if drawn is None:
        return None
    meshes, deviation = drawn
    plan.facts["max_deviation_mm"] = round(deviation, DEVIATION_DIGITS)
    plan.facts["holes_drawn"] = len(holes)
    out = sm.SurfaceMesh(meshes=list(meshes))
    for a, b in outer:
        hit = plan.profile.sag_normal(a, b)
        if hit is None:
            return None
        out.outline.append(plan.placement.point(a, b, hit[0]))
    out.rows.append(out.outline)
    return out


def _with_flanges(plan: _Plan, dirs: list[sm.Vec2], drawn: sm.SurfaceMesh) -> sm.SurfaceMesh:
    """A surface drawn to its regions (``_draw_regions``) out to its own edge, then its flat flange beyond that edge out to its
    glass outline: one star-mesh zone per flange ring along ``dirs``, every vertex at the sag of the edge point along its
    direction (as the star mesh draws the flange of a surface without holes); its first row is the regions' outline row."""

    centre = plan.centre
    hold = _exits(plan.outline, centre, dirs)  # type: ignore[arg-type]
    zones, inner = [], hold
    for flange in plan.flanges:
        outer = _exits(flange, centre, dirs)
        zones.append(sm.Zone(sm.FLANGE_MATERIAL, inner, outer, (1.0,), hold=hold))
        inner = outer
    ring = sm.star_mesh(plan.profile, plan.placement, centre, dirs, zones, plan.facing)
    return sm.SurfaceMesh(meshes=[*drawn.meshes, *ring.meshes], outline=ring.outline, undefined=drawn.undefined or ring.undefined,
                          rows=[*drawn.rows, *ring.rows])


def _draw(plan: _Plan, dirs: list[sm.Vec2], tolerance: float, strided: bool) -> sm.SurfaceMesh:
    if _irregular(plan.surface) and (not plan.flanges or (plan.outline is not None and plan.split)):
        drawn = _draw_regions(plan, dirs, tolerance)
        if drawn is not None:
            return _with_flanges(plan, dirs, drawn) if plan.flanges else drawn
    centre = plan.centre
    count = len(dirs)
    if plan.rotational:
        required = [count if i in plan.full_rows else need for i, need in enumerate(plan.needs[:-1])] + [count]
        strides = sm.row_strides(required, count) if strided else [1] * len(plan.needs)
        zones, k, inner_stride = [], 0, 1
        hold = (plan.outline.r,) * count if plan.flanges else None  # type: ignore[union-attr]
        for name, radii in plan.radial:
            lo, hi = radii[0], radii[-1]
            rows = radii[1:]
            zone_strides = tuple(strides[k:k + len(rows)])
            k += len(rows)
            span = hi - lo
            zones.append(sm.Zone(name, None if lo == 0.0 else (lo,) * count, (hi,) * count,
                                 tuple((r - lo) / span for r in rows[:-1]) + (1.0,), zone_strides, inner_stride,
                                 hold if name == sm.FLANGE_MATERIAL else None))
            inner_stride = zone_strides[-1]
        return sm.star_mesh(plan.profile, plan.placement, centre, dirs, zones, plan.facing)
    budget = tolerance * sm.MERIDIONAL_SHARE
    material = surface_material(plan.surface)
    if plan.outline is None:
        zones_spec: list[tuple[str, tuple[float, ...] | None, tuple[float, ...]]] = [(material, None, _exits(plan.aperture, centre, dirs))]
    else:
        out = _exits(plan.outline, centre, dirs)
        if not plan.split:
            zones_spec = [(material, None, out)]
        else:
            ca = tuple(min(a, o) for a, o in zip(_exits(plan.aperture, centre, dirs), out, strict=True))
            zones_spec = [(material, None, ca)]
            if any(o - a > sm.SAME_POINT_MM for a, o in zip(ca, out, strict=True)):
                band = tuple(a + _band_width(a, o, plan.outline.reach) for a, o in zip(ca, out, strict=True))
                zones_spec += [(OUTLINE_MATERIAL, ca, band), (RIM_MATERIAL, band, out)]
    zones = []
    for name, inner, outer in zones_spec:
        stations = _zone_stations(plan.profile, centre, dirs, inner, outer, budget)
        for _ in range(plan.refine):
            stations = sm.bisected(stations)
        zones.append(sm.Zone(name, inner, outer, tuple(stations)))
    if plan.flanges:  # the flat flange beyond the surface's own edge: one row per ring (flat along every direction)
        hold = _exits(plan.outline, centre, dirs)  # type: ignore[arg-type]
        inner = zones[-1].outer
        for flange in plan.flanges:
            outer = _exits(flange, centre, dirs)
            zones.append(sm.Zone(sm.FLANGE_MATERIAL, inner, outer, (1.0,), hold=hold))
            inner = outer
    return sm.star_mesh(plan.profile, plan.placement, centre, dirs, zones, plan.facing)


def _reach_from(boundary: sm.Boundary, centre: sm.Vec2) -> float:
    """The farthest distance of a boundary from the star centre (the rays' length, not the distance from the surface axis)."""

    if isinstance(boundary, sm.Circle):
        return math.sqrt((boundary.cx - centre[0]) ** 2 + (boundary.cy - centre[1]) ** 2) + boundary.r
    return max(math.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2) for x, y in boundary.points)


def _directions_needed(plans: list[_Plan], tolerance: float) -> int:
    budget = tolerance * (1.0 - sm.MERIDIONAL_SHARE)
    n = sm.MIN_DIRECTIONS
    for plan in plans:
        if plan.rotational:
            n = max(n, *plan.needs)
            continue
        for boundary in (plan.aperture, plan.outline, *plan.flanges):
            if isinstance(boundary, sm.Circle):
                cx, cy = plan.centre
                offset = math.sqrt((boundary.cx - cx) ** 2 + (boundary.cy - cy) ** 2)
                n = max(n, g.circle_segments(offset + boundary.r, tolerance))
        reach = max((_reach_from(b, plan.centre) for b in (plan.aperture, plan.outline) if b is not None), default=0.0)
        n = max(n, _surface_term(plan.profile, plan.centre, reach, budget))
    return sm.base_count(n)


def _largest_count_within(limit: int) -> int:
    """The largest direction count ``c * 2^k`` (``c`` in 8..15, as ``surface_mesh.base_count`` makes them) not above
    ``limit`` (at least the smallest one)."""

    k = 0
    while sm.BASE_MIN * (1 << (k + 1)) <= limit:
        k += 1
    return max(sm.BASE_MIN, min(sm.BASE_MAX, limit >> k)) * (1 << k)


def _draw_all(plans: list[_Plan], corner_sets: list, tolerance: float, centre: sm.Vec2) -> tuple[list[sm.Vec2], list[sm.SurfaceMesh]]:
    """Draw every plan with one shared set of directions (so edge walls meet the surfaces' outlines), refining the
    directions and the failing surfaces' stations until each non-exact surface is within the tolerance."""

    for plan in plans:
        if plan.rotational:
            _plan_radial(plan, tolerance)
    n = _directions_needed(plans, tolerance)
    strided = not any(corners for corners in corner_sets)
    squeezed = False
    for attempt in range(sm.MAX_REFINEMENTS + 1):
        dirs = sm.directions(n, centre, corner_sets)
        drawn = [_draw(plan, dirs, tolerance, strided) for plan in plans]
        failing = []
        for i, (plan, mesh) in enumerate(zip(plans, drawn, strict=True)):
            if plan.exact:
                continue
            deviation = sm.max_deviation(mesh.meshes, plan.profile, plan.placement)
            plan.facts["max_deviation_mm"] = round(deviation, DEVIATION_DIGITS)
            if deviation > tolerance:
                failing.append(i)
        size = max((sum(len(m.positions) for _, m in drawn[i].meshes) for i in failing), default=0)
        if failing and attempt < sm.MAX_REFINEMENTS and 2 * size > MAX_SURFACE_VERTICES and not squeezed:
            # doubling again would exceed the vertex budget: spend what is left of it on directions once (the vertex count
            # grows in proportion to them), the largest c * 2^k that fits
            grown = _largest_count_within(n * MAX_SURFACE_VERTICES // max(size, 1))
            squeezed = True
            if grown > n:
                n = grown
                continue
        if not failing or attempt == sm.MAX_REFINEMENTS or 2 * size > MAX_SURFACE_VERTICES:
            for i in failing:  # the vertex budget of one surface is spent: it states the deviation it reached
                plans[i].facts["chord_tolerance_met"] = False
            return dirs, drawn
        # refine one way at a time (twice the directions, then every failing surface's stations halved, and so on):
        # doubling both at once overshoots the tolerance about four times over
        if attempt % 2 == 0:
            n *= 2
        else:
            for i in failing:
                plans[i].refine += 1
    raise AssertionError("unreachable")


# ------------------------------------------------------------------------------------------------ surface sequence


def _media(geom: OpticBodyGeometryV1, k: int) -> tuple[str, str]:
    last = len(geom.surfaces) - 1
    return (AIR if k == 0 else geom.media[k - 1]), (AIR if k == last else geom.media[k])


def _facing(before: str, after: str) -> float:
    """The drawn front of a coaxial surface: out of the glass it bounds (-Z where glass follows air), else +Z."""

    return -1.0 if before in RESERVED_MEDIA and after not in RESERVED_MEDIA else 1.0


def _design_extent_z(geom: OpticBodyGeometryV1) -> tuple[float, float]:
    if geom.shape == "CYLINDER":
        return geom.z_min_mm, geom.z_max_mm  # type: ignore[return-value]
    if geom.shape == "BOX":
        return geom.center_mm[2] - geom.half_extents_mm[2], geom.center_mm[2] + geom.half_extents_mm[2]  # type: ignore[index]
    zs = [v[2] for v in geom.vertices_mm or ()]
    return min(zs), max(zs)


def _rect_band(surface: OpticSurfaceV1, hx: float, hy: float, reach: float) -> tuple[float, float, float, float, float, float, float, float]:
    """``(x0, x1, y0, y1)`` of a rectangular clear aperture and of its outline band (clipped to the outline)."""

    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    ax, ay = surface.clear_half_widths_mm  # type: ignore[misc]
    x0, x1, y0, y1 = max(cx - ax, -hx), min(cx + ax, hx), max(cy - ay, -hy), min(cy + ay, hy)
    width = OUTLINE_BAND_FRACTION * reach
    return (x0, x1, y0, y1, x0 - min(width, OUTLINE_BAND_RIM_SHARE * (x0 + hx)), x1 + min(width, OUTLINE_BAND_RIM_SHARE * (hx - x1)),
            y0 - min(width, OUTLINE_BAND_RIM_SHARE * (y0 + hy)), y1 + min(width, OUTLINE_BAND_RIM_SHARE * (hy - y1)))


def _unique_sorted(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > sm.SAME_POINT_MM:
            out.append(v)
    return out


def _axis_stations(breaks: list[float], f, budget: float) -> list[float]:
    """Breakpoints plus the chord sampler's stations of ``f`` between each pair of them."""

    out: list[float] = []
    for a, b in zip(breaks, breaks[1:]):
        out += sm.profile_stations(f, a, b, budget)
    return out


def _sequence_grid(geom: OpticBodyGeometryV1, tolerance: float) -> tuple[list[sm.SurfaceMesh], list[dict[str, Any]], list[sm.Profile]]:
    """A rectangular element whose apertures are rectangles: every surface on one shared grid (so edge walls meet), the
    clear aperture, outline band and rim as cells. A plane or a cylinder (a ruled surface: its triangles are exact
    chords across the curved coordinate) is within the tolerance by construction; any other profile is checked and the
    grid refined."""

    hx, hy = geom.edge_half_widths_mm  # type: ignore[misc]
    reach = math.sqrt(hx * hx + hy * hy)
    xb, yb = [-hx, hx], [-hy, hy]
    bands = []
    for surface in geom.surfaces:
        band = _rect_band(surface, hx, hy, reach)
        bands.append(band)
        xb += [band[0], band[1], band[4], band[5]]
        yb += [band[2], band[3], band[6], band[7]]
    xb, yb = _unique_sorted(xb), _unique_sorted(yb)
    profiles = [sm.Profile.of(s) for s in geom.surfaces]
    xs, ys, exact = list(xb), list(yb), True
    if any(p.linear != 0.0 for p in profiles):  # an axicon's apex (or roof ridge) is a kink: a grid line runs through it
        xs += [0.0] if xb[0] < 0.0 < xb[-1] else []
        ys += [0.0] if yb[0] < 0.0 < yb[-1] else []
    for profile in profiles:
        if profile.is_flat:
            continue
        if profile.cylinder_axis == "Y":  # curved along x
            xs += _axis_stations(xb, lambda t, p=profile: p.sag(t, 0.0), tolerance)
        elif profile.cylinder_axis == "X":  # curved along y
            ys += _axis_stations(yb, lambda t, p=profile: p.sag(0.0, t), tolerance)
        else:
            exact = False
            budget = tolerance * sm.MERIDIONAL_SHARE
            for y in (0.0, hy):
                xs += _axis_stations(xb, lambda t, p=profile, y=y: p.sag(t, y), budget)
            for x in (0.0, hx):
                ys += _axis_stations(yb, lambda t, p=profile, x=x: p.sag(x, t), budget)
    xs, ys = _unique_sorted(xs), _unique_sorted(ys)
    facts_list = []
    for attempt in range(sm.MAX_REFINEMENTS + 1):
        drawn, facts_list = [], []
        for k, (surface, profile, band) in enumerate(zip(geom.surfaces, profiles, bands, strict=True)):
            before, after = _media(geom, k)
            x0, x1, y0, y1, bx0, bx1, by0, by1 = band
            name = surface_material(surface)

            def material_of(x: float, y: float, name=name, x0=x0, x1=x1, y0=y0, y1=y1, bx0=bx0, bx1=bx1, by0=by0, by1=by1) -> str:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return name
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    return OUTLINE_MATERIAL
                return RIM_MATERIAL

            drawn.append(sm.grid_mesh(profile, sm.Placement.coaxial(surface.vertex_z_mm or 0.0), xs, ys, material_of, _facing(before, after)))
            facts_list.append({"placement": "COAXIAL", "medium_before": before, "medium_after": after, "clear_aperture_drawn": True})
        if exact:
            break
        failing = [k for k, (mesh, profile, surface) in enumerate(zip(drawn, profiles, geom.surfaces, strict=True))
                   if not profile.is_flat and profile.cylinder_axis is None
                   and sm.max_deviation(mesh.meshes, profile, sm.Placement.coaxial(surface.vertex_z_mm or 0.0)) > tolerance]
        if not failing:
            break
        if attempt == sm.MAX_REFINEMENTS:
            for k in failing:
                facts_list[k]["chord_tolerance_met"] = False
            break
        xs, ys = sm.bisected_values(xs), sm.bisected_values(ys)
    return drawn, facts_list, profiles


def _rings(geom: OpticBodyGeometryV1, k: int) -> list[tuple[float, float]]:
    """Half widths of surface ``k``'s rings, inside out: its own edge, then the outlines of the glass gaps it bounds and its
    glass outline where they reach beyond it (the flange rings an edge wall meets)."""

    own = geom.surface_edge(k)
    rings = {geom.gap_outline(gap) for gap in (k - 1, k) if 0 <= gap < len(geom.media) and geom.media[gap] not in RESERVED_MEDIA}
    rings.add(geom.glass_outline(k))
    beyond = sorted(r for r in rings if r != own and r[0] >= own[0] and r[1] >= own[1])
    nested = [r for i, r in enumerate(beyond) if all(r[0] >= q[0] and r[1] >= q[1] for q in beyond[:i])]
    return [own, *nested]


def _sequence(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    centre = (0.0, 0.0)
    outline: sm.Boundary = (sm.Circle(0.0, 0.0, geom.edge_radius_mm) if geom.edge_radius_mm is not None
                            else sm.Polygon.rectangle(0.0, 0.0, *geom.edge_half_widths_mm))  # type: ignore[misc]
    drawing = BodyDrawing(tolerance_mm=tolerance)
    round_outline = isinstance(outline, sm.Circle)
    edged = geom.has_surface_edges

    def boundary(half: tuple[float, float]) -> sm.Boundary:
        return sm.Circle(0.0, 0.0, half[0]) if round_outline else sm.Polygon.rectangle(0.0, 0.0, half[0], half[1])

    rings = [_rings(geom, k) for k in range(len(geom.surfaces))] if edged else []
    if not round_outline and not edged and all(s.clear_half_widths_mm is not None and not _irregular(s) for s in geom.surfaces):
        drawn, facts_list, profiles = _sequence_grid(geom, tolerance)
        dirs: list[sm.Vec2] = []
    else:
        plans = []
        corner_sets = [outline.corners]
        for k, surface in enumerate(geom.surfaces):
            before, after = _media(geom, k)
            aperture = _aperture(surface)
            profile = sm.Profile.of(surface)
            split = aperture.contains(0.0, 0.0, margin=sm.SAME_POINT_MM)
            own = boundary(rings[k][0]) if edged else outline
            flanges = [boundary(r) for r in rings[k][1:]] if edged else []
            rotational = profile.rotational and _concentric(aperture, centre) and _concentric(own, centre)
            facts: dict[str, Any] = {"placement": "COAXIAL", "medium_before": before, "medium_after": after, "clear_aperture_drawn": split}
            plans.append(_Plan(surface, profile, sm.Placement.coaxial(surface.vertex_z_mm or 0.0), centre, aperture, own,
                               _facing(before, after), facts, exact=rotational or profile.is_flat, rotational=rotational, split=split,
                               flanges=flanges))
            corner_sets += [b.corners for b in (own, *flanges)] if edged else []
        corner_sets += [p.aperture.corners for p in plans]
        dirs, drawn = _draw_all(plans, corner_sets, tolerance, centre)
        facts_list, profiles = [p.facts for p in plans], [p.profile for p in plans]
    for surface, mesh, facts, profile in zip(geom.surfaces, drawn, facts_list, profiles, strict=True):
        if mesh.undefined:
            facts["profile_domain_limit_mm"] = profile.domain_rho
        drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, facts))
    if len(geom.surfaces) == 1:  # one coated surface; its substrate (not modelled) is the rest of the design body
        (surface,), (mesh,) = geom.surfaces, drawn
        z_min, z_max = _design_extent_z(geom)
        vertex = surface.vertex_z_mm or 0.0
        reflector = surface.role is OpticSurfaceRoleV1.REFLECT
        far = sm.substrate_face_z(vertex, z_min, z_max, reflector)  # behind a mirror: its -Z side, never the air in front
        below = reflector or far < vertex
        loop = [(p[0], p[1], far) for p in mesh.outline]
        face = _polygon_mesh(loop[::-1] if below else loop, (0.0, 0.0, -1.0 if below else 1.0))
        face.extend(sm.wall(loop, mesh.outline, dirs, round_outline) if below else sm.wall(mesh.outline, loop, dirs, round_outline))
        drawing.body.append(("mirror_substrate" if reflector else "optic_substrate", face))
        return drawing
    def ring_row(k: int, half: tuple[float, float]) -> list[g.Vec3]:
        """Surface ``k``'s drawn row on the ring of half widths ``half`` (its edge, or a flange ring; the edges of two surfaces
        bounding one glass nest, OpticBodyGeometryV1, so a glass gap's outline is a ring of both)."""

        first = len(drawn[k].rows) - len(rings[k])
        return drawn[k].rows[first + rings[k].index(half)]

    walls = g.Mesh()
    for k, medium in enumerate(geom.media):
        if medium in RESERVED_MEDIA:
            continue
        if edged:  # the glass's wall at the gap's outline, between the two surfaces' rows on that ring
            gap = geom.gap_outline(k)
            walls.extend(sm.wall(ring_row(k, gap), ring_row(k + 1, gap), dirs, round_outline))
        else:
            walls.extend(sm.wall(drawn[k].outline, drawn[k + 1].outline, dirs, round_outline))
    if not walls.empty:
        drawing.body.append(("optic_edge", walls))
    return drawing


# ------------------------------------------------------------------------------------------------- surface sets


def scene_solid(geom: OpticBodyGeometryV1) -> Solid | None:
    """The float solid of a surface-set body (its corners found from its faces)."""

    if geom.solid is None:
        return None
    planes = tuple((tuple(f.outward_normal), g.dot(tuple(f.outward_normal), tuple(f.point_mm))) for f in geom.solid.faces)
    return solid_from_planes(geom.solid.medium, planes, geom.solid.side_radius_mm)  # type: ignore[arg-type]


def _polygon_mesh(polygon: list[g.Vec3], normal: g.Vec3) -> g.Mesh:
    """A convex polygon as a fan, normal ``normal`` (corners counter-clockwise about it)."""

    m = g.Mesh()
    m.positions = list(polygon)
    m.normals = [normal] * len(polygon)
    for k in range(1, len(polygon) - 1):
        m.indices.extend((0, k, k + 1))
    return m


def _frame(surface: OpticSurfaceV1) -> tuple[g.Vec3, g.Vec3, g.Vec3, g.Vec3]:
    origin, normal, u = tuple(surface.origin_mm), g.unit(tuple(surface.normal)), g.unit(tuple(surface.u_axis))  # type: ignore[arg-type]
    return origin, normal, u, g.cross(normal, u)  # type: ignore[return-value]


def _on_face(surface: OpticSurfaceV1, solid: Solid) -> int | None:
    origin, normal, _, _ = _frame(surface)
    for index, (n, offset) in enumerate(solid.planes):
        if abs(g.dot(n, origin) - offset) <= ON_FACE_TOLERANCE_MM and abs(abs(g.dot(n, normal)) - 1.0) <= PARALLEL_TOLERANCE:
            return index
    return None


def _local_polygon(points: list[g.Vec3], origin: g.Vec3, u: g.Vec3, v: g.Vec3) -> sm.Polygon | None:
    """A 3D polygon in a surface's plane as a counter-clockwise 2D polygon in the surface's (u, v) coordinates."""

    flat = [(g.dot(g.sub(p, origin), u), g.dot(g.sub(p, origin), v)) for p in points]
    if len(flat) < 3:
        return None
    area = sum(flat[i][0] * flat[(i + 1) % len(flat)][1] - flat[(i + 1) % len(flat)][0] * flat[i][1] for i in range(len(flat)))
    if area < 0.0:
        flat.reverse()
    return sm.Polygon(tuple(flat))


def _set(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    drawing = BodyDrawing(tolerance_mm=tolerance)
    solid = scene_solid(geom)
    segments = g.circle_segments(solid.side_radius, tolerance) if solid is not None and solid.side_radius is not None else 0
    on_faces: set[int] = set()
    for surface in geom.surfaces:
        origin, normal, u, v = _frame(surface)
        placement = sm.Placement(origin, u, v, normal)
        aperture = _aperture(surface)
        centre = surface.aperture_center_mm or (0.0, 0.0)
        index = _on_face(surface, solid) if solid is not None else None
        polygon: list[g.Vec3] | None = None
        if index is not None:
            on_faces.add(index)
            polygon = face_polygon(solid, index, segments=segments or ROUND_SEGMENTS)  # type: ignore[arg-type]
            placement_kind = "BODY_FACE"
        elif solid is not None:
            polygon = section_polygon(solid, origin, normal, segments=segments or ROUND_SEGMENTS)
            placement_kind = "BODY_SECTION"
        else:
            placement_kind = "FREE"
        outline = _local_polygon(polygon, origin, u, v) if polygon is not None else None
        facts: dict[str, Any] = {"placement": placement_kind}
        if solid is not None:
            facts["solid_medium"] = solid.medium
        split = outline is None or outline.contains(centre[0], centre[1], margin=sm.SAME_POINT_MM)
        facts["clear_aperture_drawn"] = split
        if not split:  # the aperture centre is not inside the face: draw the face, centred on its own centroid
            centre = (sum(p[0] for p in outline.points) / len(outline.points), sum(p[1] for p in outline.points) / len(outline.points))  # type: ignore[union-attr]
        profile = sm.Profile.of(surface)
        plan = _Plan(surface, profile, placement, centre, aperture, outline, 1.0, facts, exact=profile.is_flat, split=split)
        corner_sets = [aperture.corners] + ([outline.corners] if outline is not None else [])
        _, (mesh,) = _draw_all([plan], corner_sets, tolerance, centre)
        if mesh.undefined:
            facts["profile_domain_limit_mm"] = profile.domain_rho
        drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, facts))
    if solid is None:
        return drawing
    for index, (normal, _) in enumerate(solid.planes):
        if index in on_faces:
            continue
        polygon = face_polygon(solid, index, segments=segments or ROUND_SEGMENTS)
        if len(polygon) >= 3:
            drawing.body.append(("optic_edge", _polygon_mesh(polygon, normal)))
    if solid.side_radius is not None:
        front, back = face_polygon(solid, 0, segments=segments), face_polygon(solid, 1, segments=segments)
        front_by_angle = sorted(front, key=lambda p: g.pseudo_angle(p[0], p[1]))
        back_by_angle = sorted(back, key=lambda p: g.pseudo_angle(p[0], p[1]))
        side = g.Mesh()
        for k in range(segments):
            a, b = front_by_angle[k], back_by_angle[k]
            n = g.unit((a[0], a[1], 0.0))
            side.positions.extend((b, a))
            side.normals.extend((n, n))
        for k in range(segments):
            b0, t0, b1, t1 = 2 * k, 2 * k + 1, 2 * ((k + 1) % segments), 2 * ((k + 1) % segments) + 1
            side.indices.extend((b0, b1, t1, b0, t1, t0))
        drawing.body.append(("optic_edge", side))
    return drawing


# ---------------------------------------------------------------------------------------- polarization element


def _polarization(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    drawing = BodyDrawing(tolerance_mm=tolerance)
    (surface,) = geom.surfaces
    edge = geom.edge_radius_mm
    plane_z = surface.vertex_z_mm or 0.0
    segments = g.circle_segments(edge, tolerance)  # type: ignore[arg-type]
    body = g.side_wall(edge, geom.z_min_mm, geom.z_max_mm, segments=segments)  # type: ignore[arg-type]
    for z, facing in ((geom.z_min_mm, -1.0), (geom.z_max_mm, 1.0)):
        if abs(z - plane_z) > sm.SAME_POINT_MM:  # type: ignore[operator]
            body.extend(g.disc(edge, z, facing=facing, segments=segments))  # type: ignore[arg-type]
    name = "polarizer_body" if geom.polarization.element.value in ("LINEAR_POLARIZER", "CIRCULAR_POLARIZER") else "waveplate_body"  # type: ignore[union-attr]
    clear = surface.clear_radius_mm
    bar = g.box(-clear, clear, -AXIS_BAR_HALF_WIDTH_MM, AXIS_BAR_HALF_WIDTH_MM, geom.z_min_mm - AXIS_BAR_LIFT_MM, geom.z_max_mm + AXIS_BAR_LIFT_MM)  # type: ignore[operator]
    drawing.body += [(name, body), ("polarization_axis", bar)]
    centre = (0.0, 0.0)
    plan = _Plan(surface, sm.Profile(), sm.Placement.coaxial(plane_z), centre, _aperture(surface), sm.Circle(0.0, 0.0, edge), 1.0,  # type: ignore[arg-type]
                 {"placement": "COAXIAL", "medium_before": AIR, "medium_after": AIR, "clear_aperture_drawn": True}, exact=True,
                 rotational=_concentric(_aperture(surface), centre))
    _, (mesh,) = _draw_all([plan], [], tolerance, centre)
    drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, plan.facts))
    return drawing


# ------------------------------------------------------------------------------------------- declared single planes


def _outline_of(geom: OpticBodyGeometryV1) -> sm.Boundary:
    if geom.edge_radius_mm is not None:
        return sm.Circle(0.0, 0.0, geom.edge_radius_mm)
    return sm.Polygon.rectangle(0.0, 0.0, *geom.edge_half_widths_mm)  # type: ignore[misc]


def _stop(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    """A declared aperture stop: its opaque plate from the opening to the outline (the opening itself is empty: light
    passes there) and its central obstruction disc, in one plane (zero thickness), as the tracer models it."""

    drawing = BodyDrawing(tolerance_mm=tolerance)
    (surface,) = geom.surfaces
    outline = _outline_of(geom)
    aperture = _aperture(surface)
    centre = surface.aperture_center_mm or (0.0, 0.0)
    placement = sm.Placement.coaxial(surface.vertex_z_mm or 0.0)
    n = sm.MIN_DIRECTIONS
    for boundary in (aperture, outline):
        if isinstance(boundary, sm.Circle):
            n = max(n, g.circle_segments(math.sqrt((boundary.cx - centre[0]) ** 2 + (boundary.cy - centre[1]) ** 2) + boundary.r, tolerance))
    dirs = sm.directions(sm.base_count(n), centre, [aperture.corners, outline.corners])
    inner = tuple(min(a, o) for a, o in zip(_exits(aperture, centre, dirs), _exits(outline, centre, dirs), strict=True))
    outer = _exits(outline, centre, dirs)
    plate = sm.star_mesh(sm.Profile(), placement, centre, dirs, [sm.Zone(STOP_MATERIAL, inner, outer, (1.0,))], 1.0)
    meshes = list(plate.meshes)
    facts: dict[str, Any] = {"placement": "COAXIAL", "medium_before": AIR, "medium_after": AIR, "clear_aperture_drawn": True}
    if surface.obstruction_radius_mm is not None:
        obstruction = sm.Circle(centre[0], centre[1], surface.obstruction_radius_mm)
        ring = sm.directions(sm.base_count(g.circle_segments(surface.obstruction_radius_mm, tolerance)), centre, [])
        disc = sm.star_mesh(sm.Profile(), placement, centre, ring, [sm.Zone(STOP_MATERIAL, None, _exits(obstruction, centre, ring), (1.0,))], 1.0)
        meshes += disc.meshes
    drawing.surfaces.append(SurfaceDrawing(surface, meshes, facts))
    return drawing


def _thin_lens(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    """A declared ideal or tabulated thin lens: one plane (zero thickness) -- its clear aperture, the outline band and the
    opaque ring out to its edge."""

    drawing = BodyDrawing(tolerance_mm=tolerance)
    (surface,) = geom.surfaces
    centre = (0.0, 0.0)
    aperture, outline = _aperture(surface), _outline_of(geom)
    plan = _Plan(surface, sm.Profile(), sm.Placement.coaxial(surface.vertex_z_mm or 0.0), centre, aperture, outline, 1.0,
                 {"placement": "COAXIAL", "medium_before": AIR, "medium_after": AIR, "clear_aperture_drawn": True}, exact=True,
                 rotational=_concentric(aperture, centre) and _concentric(outline, centre))
    _, (mesh,) = _draw_all([plan], [aperture.corners, outline.corners], tolerance, centre)
    drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, plan.facts))
    return drawing


def _faraday_rotator(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    """A declared Faraday rotator: its rod or plate between the z limits of its stated body (a body of zero thickness: only
    the plane), its plane with the clear aperture, and an arrow along the outside of the body pointing along its field
    axis, local +Z."""

    drawing = BodyDrawing(tolerance_mm=tolerance)
    (surface,) = geom.surfaces
    edge, clear = geom.edge_radius_mm, surface.clear_radius_mm
    plane_z = surface.vertex_z_mm or 0.0
    low, high = geom._z_range()
    low, high = plane_z if low is None else low, plane_z if high is None else high
    segments = g.circle_segments(edge, tolerance)  # type: ignore[arg-type]
    body = g.Mesh()
    if high > low and geom.shape == "BOX":
        c, h = geom.center_mm, geom.half_extents_mm
        body.extend(g.box(c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], low, high))  # type: ignore[index]
    elif high > low:
        body.extend(g.side_wall(edge, low, high, segments=segments))  # type: ignore[arg-type]
        for z, facing in ((low, -1.0), (high, 1.0)):
            body.extend(g.disc(edge, z, facing=facing, segments=segments))  # type: ignore[arg-type]
    if not body.empty:
        drawing.body.append(("faraday_body", body))
    reach = FIELD_ARROW_REACH * edge  # type: ignore[operator]
    shaft = g.box(edge, edge + 2.0 * AXIS_BAR_HALF_WIDTH_MM, -AXIS_BAR_HALF_WIDTH_MM, AXIS_BAR_HALF_WIDTH_MM, low - reach, high + reach)  # type: ignore[operator]
    for step in range(FIELD_ARROW_STEPS):  # the arrowhead: stacked blocks narrowing toward +Z
        half = AXIS_BAR_HALF_WIDTH_MM * (FIELD_ARROW_STEPS + 1 - step)
        z_start = high + reach + step * reach / FIELD_ARROW_STEPS
        shaft.extend(g.box(edge + AXIS_BAR_HALF_WIDTH_MM - half, edge + AXIS_BAR_HALF_WIDTH_MM + half, -half, half, z_start,  # type: ignore[operator]
                           z_start + reach / FIELD_ARROW_STEPS))
    drawing.body.append(("polarization_axis", shaft))
    centre = (0.0, 0.0)
    plan = _Plan(surface, sm.Profile(), sm.Placement.coaxial(plane_z), centre, _aperture(surface), sm.Circle(0.0, 0.0, edge), 1.0,  # type: ignore[arg-type]
                 {"placement": "COAXIAL", "medium_before": AIR, "medium_after": AIR, "clear_aperture_drawn": True}, exact=True,
                 rotational=_concentric(_aperture(surface), centre))
    _, (mesh,) = _draw_all([plan], [], tolerance, centre)
    drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, plan.facts))
    return drawing


def _chord(surface: OpticSurfaceV1, kx: float, ky: float, offset: float) -> tuple[float, float] | None:
    """The stretch of the rule at ``offset`` along the unit vector ``(kx, ky)`` inside the clear aperture, in distance along
    the rule (its direction ``(-ky, kx)``): ``None`` when the rule misses the aperture."""

    if surface.clear_radius_mm is not None:
        square = surface.clear_radius_mm * surface.clear_radius_mm - offset * offset
        if square <= 0.0:
            return None
        half = math.sqrt(square)
        return -half, half
    hx, hy = surface.clear_half_widths_mm  # type: ignore[misc]
    low, high = -math.inf, math.inf
    for origin, along, half in ((offset * kx, -ky, hx), (offset * ky, kx, hy)):
        if abs(along) <= sm.SAME_POINT_MM:
            if abs(origin) > half:
                return None
            continue
        ends = sorted(((-half - origin) / along, (half - origin) / along))
        low, high = max(low, ends[0]), min(high, ends[1])
    return (low, high) if low < high else None


def _grating_rules(surface: OpticSurfaceV1, kx: float, ky: float, z: float) -> g.Mesh:
    """Parallel bars on the ruled plane across the clear aperture, along the grooves ``(-ky, kx)``."""

    if surface.clear_radius_mm is not None:
        reach = surface.clear_radius_mm
    else:
        hx, hy = surface.clear_half_widths_mm  # type: ignore[misc]
        reach = hx * abs(kx) + hy * abs(ky)
    spacing = 2.0 * reach / (GRATING_RULES + 1)
    half_width = 0.5 * GRATING_RULE_SHARE * spacing
    rules = g.Mesh()
    for k in range(GRATING_RULES):
        offset = -reach + (k + 1) * spacing
        chord = _chord(surface, kx, ky, offset)
        if chord is None:
            continue
        start, end = chord
        low, high = z + GRATING_RULE_LIFT_MM, z + GRATING_RULE_LIFT_MM + GRATING_RULE_HEIGHT_MM
        corners = [(offset + a) * kx - b * ky for a in (-half_width, half_width) for b in (start, end)], [(offset + a) * ky + b * kx for a in (-half_width, half_width) for b in (start, end)]
        (x0, x1, x2, x3), (y0, y1, y2, y3) = corners
        for top in (low, high):  # both faces of the bar, seen from either side of the plate
            base = len(rules.positions)
            rules.positions.extend([(x0, y0, top), (x1, y1, top), (x3, y3, top), (x2, y2, top)])
            rules.normals.extend([(0.0, 0.0, 1.0 if top == high else -1.0)] * 4)
            rules.indices.extend((base, base + 1, base + 2, base, base + 2, base + 3) if top == high else (base, base + 2, base + 1, base, base + 3, base + 2))
    return rules


def _diffraction_grating(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    """A declared diffraction grating: its plate between the z limits of its stated body (a body of zero thickness: only the
    plane), the ruled plane with its clear aperture, and rules along the grooves over it."""

    drawing = BodyDrawing(tolerance_mm=tolerance)
    (surface,) = geom.surfaces
    plane_z = surface.vertex_z_mm or 0.0
    low, high = geom._z_range()
    low, high = plane_z if low is None else low, plane_z if high is None else high
    plate = g.Mesh()
    if high > low and geom.shape == "BOX":
        c, h = geom.center_mm, geom.half_extents_mm
        plate.extend(g.box(c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], low, high))  # type: ignore[index]
    elif high > low and geom.edge_radius_mm is not None:
        segments = g.circle_segments(geom.edge_radius_mm, tolerance)
        plate.extend(g.side_wall(geom.edge_radius_mm, low, high, segments=segments))
        for z, facing in ((low, -1.0), (high, 1.0)):
            plate.extend(g.disc(geom.edge_radius_mm, z, facing=facing, segments=segments))
    if not plate.empty:
        # (a reflection grating's plate stops light; a transmission grating's is glass the light crosses)
        drawing.body.append(("grating_body" if geom.grating.mode == "REFLECTION" else "optic_substrate", plate))  # type: ignore[union-attr]
    kx, ky = geom.grating.grating_vector_local  # type: ignore[union-attr]
    drawing.body.append(("grating_rule", _grating_rules(surface, kx, ky, plane_z)))
    centre = (0.0, 0.0)
    aperture, outline = _aperture(surface), _outline_of(geom)
    plan = _Plan(surface, sm.Profile(), sm.Placement.coaxial(plane_z), centre, aperture, outline, 1.0,
                 {"placement": "COAXIAL", "medium_before": AIR, "medium_after": AIR, "clear_aperture_drawn": True}, exact=True,
                 rotational=_concentric(aperture, centre) and _concentric(outline, centre))
    _, (mesh,) = _draw_all([plan], [aperture.corners, outline.corners], tolerance, centre)
    drawing.surfaces.append(SurfaceDrawing(surface, mesh.meshes, plan.facts))
    return drawing


#: The GLB material of each facet group role of a ``FACET_SOLID`` body.
FACET_ROLE_MATERIALS: dict[str, Material] = {"REFLECT": "mirror_face", "ABSORB": STOP_MATERIAL}


def _facet_solid(geom: OpticBodyGeometryV1, tolerance: float) -> BodyDrawing:
    """A declared facet solid: its flat triangular facets as they are (each triangle with its own normal, outward), glass or a
    filter face by the default coating, the mirrored and blackened facet groups in their own materials. The facets are the
    body; nothing else is drawn."""

    facets = geom.facets
    assert facets is not None
    drawing = BodyDrawing(tolerance_mm=tolerance)
    group_of: dict[int, int] = {}
    for g_index, group in enumerate(facets.groups):
        for k in group.triangles:
            group_of[k] = g_index

    def material(k: int) -> Material:
        g_index = group_of.get(k)
        if g_index is None:
            kind = facets.coating_kind
        else:
            group = facets.groups[g_index]
            if group.role in FACET_ROLE_MATERIALS:
                return FACET_ROLE_MATERIALS[group.role]
            kind = group.coating_kind
        return REFRACT_MATERIALS[0] if kind is None or kind in PLAIN_COATINGS else REFRACT_MATERIALS[1]

    meshes: dict[Material, g.Mesh] = {}
    for k, (a, b, c) in enumerate(facets.triangles):
        pa, pb, pc = (tuple(facets.vertices_mm[i]) for i in (a, b, c))
        normal = g.unit(g.cross(g.sub(pb, pa), g.sub(pc, pa)))  # type: ignore[arg-type]
        mesh = meshes.setdefault(material(k), g.Mesh())
        base = len(mesh.positions)
        mesh.positions.extend([pa, pb, pc])  # type: ignore[list-item]
        mesh.normals.extend([normal] * 3)
        mesh.indices.extend((base, base + 1, base + 2))
    drawing.body.extend(sorted(meshes.items()))
    return drawing


# ----------------------------------------------------------------------------------------------------- entry points


#: How each optic-body model kind is drawn (a thin lens, a tabulated thin lens and an attenuator: one plane, its clear
#: aperture inside its outline).
BODY_DRAWINGS = {
    "SURFACE_SEQUENCE": _sequence, "SURFACE_SET": _set, "POLARIZATION_ELEMENT": _polarization, "APERTURE_STOP": _stop,
    "IDEAL_THIN_LENS": _thin_lens, "TABULATED_THIN_LENS": _thin_lens, "NEUTRAL_DENSITY": _thin_lens, "FARADAY_ROTATOR": _faraday_rotator,
    "DIFFRACTION_GRATING": _diffraction_grating, "FACET_SOLID": _facet_solid,
}


def draw_body(geom: OpticBodyGeometryV1, tolerance: float = sm.CHORD_TOLERANCE_MM) -> BodyDrawing:
    """The element-frame drawing of an ``OPTIC_BODY``: one group per optical surface and the remaining body meshes."""

    return BODY_DRAWINGS[geom.model_kind](geom, tolerance)


def body_meshes(geom: OpticBodyGeometryV1) -> list[tuple[Material, g.Mesh]]:
    """Element-frame display meshes (material name, mesh) of an ``OPTIC_BODY``: every surface group, then the body."""

    return draw_body(geom).meshes()


def extent_points(geom: OpticBodyGeometryV1) -> list[g.Vec3]:
    """Element-frame points bounding what ``body_meshes`` draws together with the design body."""

    points: list[g.Vec3] = []
    if geom.shape == "CYLINDER":
        r = geom.diameter_mm / 2  # type: ignore[operator]
        points += [(sx * r, sy * r, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (geom.z_min_mm, geom.z_max_mm)]  # type: ignore[misc]
    elif geom.shape == "BOX":
        c, h = geom.center_mm, geom.half_extents_mm
        points += [(c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]  # type: ignore[index]
    else:
        points += [tuple(v) for v in geom.vertices_mm or ()]  # type: ignore[misc]
    for _, mesh in body_meshes(geom):
        points += mesh.positions
    return points
