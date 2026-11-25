"""Optical-solid validity of placed optical bodies: per element (profile domain, thickness, medium chain), per pair
(shared volume) and per compound unit (members inside the envelope). Pure Python; no ray is traced.

Numbers are element-frame or design-frame millimetres. ``TOLERANCE_MM`` separates touching from overlapping and a knife
edge from crossing surfaces.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..optical_view.surface_mesh import Circle, Polygon
from .bodies import UNIT_ENVELOPE, envelope_body
from .bodies import AIR, SURROUNDINGS, Convex, OpticalBody, Patch, Plate, Slab, Vec3, design_solid, material_name, outline_point
from .report import ERROR, INFO, WARNING, ZONE_CLEAR_APERTURE, ZONE_OUTSIDE_CLEAR_APERTURE, ZONE_UNIT_ENVELOPE, SolidIssue

#: Two bodies closer than this (mm) touch; deeper they overlap. Two surfaces of one element closer than this meet.
TOLERANCE_MM = 1e-6
#: Rational points per quarter turn of a round outline where a two-variable profile must be defined.
SHAPE_RIM_STEPS = 64
RADIAL_SAMPLES = 1024
GRID_SAMPLES = 96
PERIMETER_SAMPLES = 1024
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
REFINE_ITERATIONS = 80
CANDIDATES = 6
PARALLEL_TOLERANCE = 1e-9
COAXIAL_TOLERANCE_MM = 1e-9
#: The sampled search skips a boundary point farther than this share of the other body's box diagonal (plus
#: BOX_ALLOWANCE_MM) outside that box: it lies outside the other body, deeper candidates come from nearer points.
BOX_MARGIN = 0.1
BOX_ALLOWANCE_MM = 1e-3
#: The pattern search stops below this step (fraction of a patch); a value this much (relative) beyond a limit exceeds it.
MIN_STEP = 1e-9
RELATIVE_EPS = 1e-12
#: A surface lies on a face this close to its plane (mm, and in the cosine of their normals); a point is on an outline or in
#: an aperture this close to it (mm).
ON_FACE_MM = 1e-6
ON_EDGE_MM = 1e-12
ON_APERTURE_MM = 1e-9
#: Plausibility of an edged element (warnings, never refusals): a glass gap whose edge thickness exceeds a multiple of the
#: clear diameter it serves (``EdgeThicknessBand``), and a surface whose glass outline reaches beyond this multiple of its
#: clear radius and by more than EDGE_MARGIN_MIN_MM. ``EDGE_THICKNESS_RATIO_LIMIT`` and ``EDGE_TO_CLEAR_RATIO_LIMIT`` are
#: set above every product of the published catalog (its extremes are molded aspheres with wide flanges: an edge
#: thickness of 1.30 clear diameters, a glass outline of 2.73 clear radii), so a manufactured element does not trigger
#: them; an outline taken from the largest surface for every surface does. A declared model (a Project's own lens, an
#: imported prescription) is reviewed against ``EDGE_THICKNESS_REVIEW_RATIO``: of the published catalog's 1872 glass
#: gaps, the median is 0.12 and 93 % are at most 0.35.
EDGE_THICKNESS_RATIO_LIMIT = 1.5
EDGE_THICKNESS_REVIEW_RATIO = 0.35
EDGE_TO_CLEAR_RATIO_LIMIT = 3.0
EDGE_MARGIN_MIN_MM = 2.0


@dataclass(frozen=True)
class EdgeThicknessBand:
    """When a glass gap's edge thickness over its clear diameter is reported (``EDGE_THICKNESS_IMPLAUSIBLE``): a warning
    above ``catalog_ratio`` (beyond every published product) for any model; for a declared model an INFO finding between
    ``review_ratio`` and that. A review threshold of the check, not a fact of the design: it changes no Project, hash or run."""

    review_ratio: float = EDGE_THICKNESS_REVIEW_RATIO
    catalog_ratio: float = EDGE_THICKNESS_RATIO_LIMIT

    def limit(self, catalog: bool) -> float:
        return self.catalog_ratio if catalog else self.review_ratio


DEFAULT_EDGE_BAND = EdgeThicknessBand()


def _r(value: float, digits: int = 6) -> float:
    return round(value, digits) + 0.0


def _rv(values, digits: int = 6) -> tuple[float, ...]:
    return tuple(_r(v, digits) for v in values)


# ------------------------------------------------------------------------------------------------ placement


def _matrix(q: tuple[float, float, float, float]) -> tuple[Vec3, Vec3, Vec3]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


@dataclass(frozen=True)
class PlacedBody:
    """An optical body at its design-frame pose (``x_world = R(q) x_local + position``)."""

    body: OpticalBody
    position_mm: Vec3
    quaternion_wxyz: tuple[float, float, float, float]

    #: per-placement results computed once (the rotation matrix, the sampled boundary): not part of the placement
    memo: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def rotation(self) -> tuple[Vec3, Vec3, Vec3]:
        if "rotation" not in self.memo:
            self.memo["rotation"] = _matrix(self.quaternion_wxyz)
        return self.memo["rotation"]

    @property
    def samples(self) -> list[tuple[Patch, float, float, Vec3]]:
        """Every sampled boundary point of the body (patch, u, v, design-frame point), computed once per placement."""

        if "samples" not in self.memo:
            self.memo["samples"] = _samples(self)
        return self.memo["samples"]

    def near_box(self, local: Vec3) -> bool:
        """Whether an element-frame point lies within the body's sampled box grown by ``BOX_MARGIN`` of its diagonal."""

        box = self.body.box
        if box is None:
            return True
        lo, hi = box
        grow = BOX_MARGIN * math.sqrt(sum((h - l) ** 2 for l, h in zip(lo, hi))) + BOX_ALLOWANCE_MM
        return all(lo[k] - grow <= local[k] <= hi[k] + grow for k in range(3))

    def to_world(self, p: Vec3) -> Vec3:
        m, t = self.rotation, self.position_mm
        return (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + t[0], m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + t[1],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + t[2])

    def to_local(self, p: Vec3) -> Vec3:
        m, t = self.rotation, self.position_mm
        d = (p[0] - t[0], p[1] - t[1], p[2] - t[2])
        return (m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2], m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
                m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2])

    @property
    def axis(self) -> Vec3:
        m = self.rotation
        return (m[0][2], m[1][2], m[2][2])


# ------------------------------------------------------------------------------------------------ 1D / 2D search


def _maximize_1d(f: Callable[[float], float | None], lo: float, hi: float, samples: int = RADIAL_SAMPLES) -> tuple[float, float] | None:
    """``(t, f(t))`` at the largest value of ``f`` on ``[lo, hi]``: dense samples, then golden-section refinement on the
    bracket around the best sample (``None`` where ``f`` is undefined everywhere)."""

    best: tuple[float, float] | None = None
    best_k = 0
    for k in range(samples + 1):
        t = lo + (hi - lo) * k / samples
        v = f(t)
        if v is not None and (best is None or v > best[1]):
            best, best_k = (t, v), k
    if best is None or hi <= lo:
        return best
    a = lo + (hi - lo) * max(best_k - 1, 0) / samples
    b = lo + (hi - lo) * min(best_k + 1, samples) / samples
    for _ in range(REFINE_ITERATIONS):
        c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
        fc, fd = f(c), f(d)
        if fc is None and fd is None:
            break
        if fd is None or (fc is not None and fc >= fd):
            b = d
        else:
            a = c
    t = 0.5 * (a + b)
    v = f(t)
    if v is not None and v > best[1]:
        best = (t, v)
    return best


def _climb(f: Callable[[float, float], float | None], u: float, v: float, step: float, *, wrap_v: bool = False) -> tuple[float, float, float]:
    """Pattern search for the largest ``f`` on the unit square from ``(u, v)``: ``(u, v, f)``."""

    best = f(u, v)
    if best is None:
        return u, v, -math.inf
    while step > MIN_STEP:
        moved = False
        for du, dv in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step), (step, step), (-step, -step), (step, -step), (-step, step)):
            nu = min(1.0, max(0.0, u + du))
            nv = (v + dv) % 1.0 if wrap_v else min(1.0, max(0.0, v + dv))
            value = f(nu, nv)
            if value is not None and value > best:
                u, v, best, moved = nu, nv, value, True
                break
        if not moved:
            step *= 0.5
    return u, v, best


# ------------------------------------------------------------------------------------------------ per element


def _reach(outline: Circle | Polygon, cylinder_axis: str | None) -> tuple[float, tuple[float, float]]:
    """The largest profile coordinate over an outline and the outline point where it is reached."""

    if isinstance(outline, Circle):
        if cylinder_axis == "X":
            return abs(outline.cy) + outline.r, (outline.cx, outline.cy + math.copysign(outline.r, outline.cy or 1.0))
        if cylinder_axis == "Y":
            return abs(outline.cx) + outline.r, (outline.cx + math.copysign(outline.r, outline.cx or 1.0), outline.cy)
        d = math.sqrt(outline.cx * outline.cx + outline.cy * outline.cy)
        ux, uy = (outline.cx / d, outline.cy / d) if d > 0.0 else (1.0, 0.0)
        return d + outline.r, (outline.cx + ux * outline.r, outline.cy + uy * outline.r)
    best = (-1.0, (0.0, 0.0))
    for x, y in outline.points:
        value = abs(y) if cylinder_axis == "X" else abs(x) if cylinder_axis == "Y" else math.sqrt(x * x + y * y)
        if value > best[0]:
            best = (value, (x, y))
    return best


def _outline_samples(outline: Circle | Polygon) -> list[tuple[float, float]]:
    """Points of an outline's boundary (a circle: the rational points of ``SHAPE_RIM_STEPS`` per quarter turn; a polygon: its
    corners and edge midpoints)."""

    if isinstance(outline, Circle):
        out = []
        for k in range(SHAPE_RIM_STEPS + 1):
            t = k / SHAPE_RIM_STEPS
            c, s = (1.0 - t * t) / (1.0 + t * t), 2.0 * t / (1.0 + t * t)
            out += [(outline.cx + sx * c * outline.r, outline.cy + sy * s * outline.r) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        return out
    points = list(outline.points)
    return points + [((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in zip(points, points[1:] + points[:1])]


def _zone(in_clear: bool) -> str:
    """The zone of a domain or thickness finding (``report.ZONES``): inside a clear aperture, or beyond them."""

    return ZONE_CLEAR_APERTURE if in_clear else ZONE_OUTSIDE_CLEAR_APERTURE


def _shape_domain_issue(body: OpticalBody, surface_id: str, profile, outline: Circle | Polygon, what: str,
                        aperture: Circle | Polygon | None) -> SolidIssue | None:
    """A two-variable profile that is not defined somewhere on the boundary it must reach. ``aperture``: the surface's
    clear aperture (the finding's zone is ``CLEAR_APERTURE`` when the profile also ends at one of its boundary points)."""

    for x, y in _outline_samples(outline):
        if profile.sag(x, y) is None:
            in_clear = aperture is None or any(profile.sag(a, b) is None for a, b in _outline_samples(aperture))
            return SolidIssue("SAG_DOMAIN_INSIDE_OUTLINE", ERROR, f"surface {surface_id}'s profile is not defined at ({_r(x, 4):g}, {_r(y, 4):g}) mm on "
                              f"{what}: the surface does not exist over its whole outline", body.instance_id, body.member_id, surface_id,
                              zone=_zone(in_clear))
    return None


def _domain_issues(body: OpticalBody) -> list[SolidIssue]:
    issues = []
    for surface in body.surfaces:
        profile = surface.profile
        if profile.form is not None:
            outline = surface.edge_outline or body.outline
            found = None if outline is None else _shape_domain_issue(body, surface.surface_id, profile, outline,
                                                                     "its edge" if surface.edge is not None else "the element outline",
                                                                     surface.aperture)
            if found is not None:
                issues.append(found)
            continue
        limit = profile.domain_rho
        # a surface with its own edge exists out to that edge (held flat beyond it); else out to the element outline
        outline = surface.edge_outline or body.outline
        if math.isinf(limit) or outline is None:
            continue
        reach, (x, y) = _reach(outline, profile.cylinder_axis)
        if reach > limit * (1.0 + RELATIVE_EPS) + TOLERANCE_MM:
            what = "its edge" if surface.edge is not None else "the element outline"
            issues.append(SolidIssue(
                "SAG_DOMAIN_INSIDE_OUTLINE", ERROR,
                f"surface {surface.surface_id}'s profile is defined out to {_r(limit, 4):g} mm from its axis, but {what} "
                f"reaches {_r(reach, 4):g} mm: the surface does not exist over its whole outline",
                body.instance_id, body.member_id, surface.surface_id, local_position_mm=_rv((x, y, surface.vertex_z)), value_mm=_r(limit),
                limit_mm=_r(reach), zone=_zone(_ends_inside(limit, surface.aperture, profile.cylinder_axis))))
    for surface in body.posed:
        if surface.radius_mm is None and surface.sag is None:
            continue
        from .bodies import _aperture, _profile  # the posed surface's profile (same float conversion as the coaxial ones)

        profile = _profile(surface)
        limit = profile.domain_rho
        outline = _posed_outline(body, surface)
        if profile.form is not None:
            found = None if outline is None else _shape_domain_issue(body, surface.surface_id, profile, outline, "the face it lies on",
                                                                     _aperture(surface))
            if found is not None:
                issues.append(found)
            continue
        if math.isinf(limit) or outline is None:
            continue
        reach, (x, y) = _reach(outline, profile.cylinder_axis)
        if reach > limit * (1.0 + RELATIVE_EPS) + TOLERANCE_MM:
            issues.append(SolidIssue(
                "SAG_DOMAIN_INSIDE_OUTLINE", ERROR,
                f"surface {surface.surface_id}'s profile is defined out to {_r(limit, 4):g} mm from its axis, but the face it lies on "
                f"reaches {_r(reach, 4):g} mm: the surface does not exist over its whole face",
                body.instance_id, body.member_id, surface.surface_id, value_mm=_r(limit), limit_mm=_r(reach),
                zone=_zone(_ends_inside(limit, _aperture(surface), profile.cylinder_axis))))
    return issues


def _ends_inside(limit: float, aperture: Circle | Polygon, cylinder_axis: str | None) -> bool:
    """Whether a profile defined out to ``limit`` from its axis ends inside the clear ``aperture``."""

    return _reach(aperture, cylinder_axis)[0] > limit * (1.0 + RELATIVE_EPS) + TOLERANCE_MM


def _posed_outline(body: OpticalBody, surface: Any) -> Polygon | None:
    """The body face (or section) a posed surface lies on, in the surface's (u, v) coordinates."""

    from ..optical_view.solids import face_polygon, section_polygon
    from .bodies import _f

    if body.solid is None:
        return None
    origin = tuple(_f(v) for v in surface.origin_mm)
    normal = tuple(_f(v) for v in surface.normal)
    u = tuple(_f(v) for v in surface.u_axis)
    v = (normal[1] * u[2] - normal[2] * u[1], normal[2] * u[0] - normal[0] * u[2], normal[0] * u[1] - normal[1] * u[0])
    on_face = [i for i, (n, d) in enumerate(body.solid.planes)
               if abs(sum(a * b for a, b in zip(n, origin)) - d) <= ON_FACE_MM and abs(abs(sum(a * b for a, b in zip(n, normal))) - 1.0) <= ON_FACE_MM]
    polygon = face_polygon(body.solid, on_face[0]) if on_face else section_polygon(body.solid, origin, normal)
    if len(polygon) < 3:
        return None
    flat = [(sum((p[k] - origin[k]) * u[k] for k in range(3)), sum((p[k] - origin[k]) * v[k] for k in range(3))) for p in polygon]
    area = sum(flat[i][0] * flat[(i + 1) % len(flat)][1] - flat[(i + 1) % len(flat)][0] * flat[i][1] for i in range(len(flat)))
    return Polygon(tuple(flat if area >= 0.0 else flat[::-1]))


def _gap_thickness(slab: Slab) -> dict[str, Any]:
    """Centre, minimum (and where) and edge thickness of the medium between two consecutive surfaces over the outline."""

    front, back, outline = slab.front, slab.back, slab.outline

    def t(x: float, y: float) -> float | None:
        a, b = front.z(x, y), back.z(x, y)
        return None if a is None or b is None else b - a

    centre = t(0.0, 0.0)
    if slab.rotational:
        r = outline.r  # type: ignore[union-attr]
        low = _maximize_1d(lambda rho: None if t(rho, 0.0) is None else -t(rho, 0.0), 0.0, r)  # type: ignore[operator]
        at, minimum = (low[0], 0.0), -low[1]  # type: ignore[index]
        edge = t(r, 0.0)
        return {"centre": centre, "minimum": minimum, "at": at, "edge": edge}
    best: tuple[float, float, float] | None = None
    for i in range(GRID_SAMPLES + 1):
        for j in range(GRID_SAMPLES + 1):
            x, y = outline_point(outline, i / GRID_SAMPLES, j / GRID_SAMPLES)
            if _outside(outline, x, y):
                continue
            value = t(x, y)
            if value is not None and (best is None or value < best[0]):
                best = (value, i / GRID_SAMPLES, j / GRID_SAMPLES)
    if best is None:
        return {"centre": centre, "minimum": None, "at": None, "edge": None}

    def neg(u: float, v: float) -> float | None:
        x, y = outline_point(outline, u, v)
        if _outside(outline, x, y):
            return None
        value = t(x, y)
        return None if value is None else -value

    u, v, value = _climb(neg, best[1], best[2], 1.0 / GRID_SAMPLES, wrap_v=isinstance(outline, Circle))
    at = outline_point(outline, u, v)
    edge = None
    for k in range(PERIMETER_SAMPLES):
        x, y = _perimeter(outline, k / PERIMETER_SAMPLES)
        value_e = t(x, y)
        if value_e is not None and (edge is None or value_e < edge):
            edge = value_e
    return {"centre": centre, "minimum": -value, "at": at, "edge": edge}


def _outside(outline: Circle | Polygon, x: float, y: float) -> bool:
    from .bodies import outline_margin

    return outline_margin(outline, x, y) < -ON_EDGE_MM


def _perimeter(outline: Circle | Polygon, u: float) -> tuple[float, float]:
    if isinstance(outline, Circle):
        a = 2.0 * math.pi * u
        return outline.cx + outline.r * math.cos(a), outline.cy + outline.r * math.sin(a)
    pts = outline.points
    s = (u % 1.0) * len(pts)
    i = int(s)
    f = s - i
    (ax, ay), (bx, by) = pts[i % len(pts)], pts[(i + 1) % len(pts)]
    return ax + (bx - ax) * f, ay + (by - ay) * f


def _in_aperture(aperture: Circle | Polygon, x: float, y: float) -> bool:
    return aperture.contains(x, y, margin=-ON_APERTURE_MM)


def _clear_minimum(t: Callable[[float, float], float | None], surface: Any, rotational: bool) -> tuple[float, tuple[float, float]] | None:
    """The smallest ``t(x, y)`` over ``surface``'s clear aperture (holes left out) and where: on the radius for a centred
    round aperture between rotational profiles (``rotational``), else on a grid of the aperture's box refined by a
    pattern search that stays in the aperture."""

    aperture = surface.aperture
    if rotational:
        low = _maximize_1d(lambda rho: None if t(rho, 0.0) is None else -t(rho, 0.0), 0.0, aperture.r)  # type: ignore[operator]
        return None if low is None else (-low[1], (low[0], 0.0))
    if isinstance(aperture, Circle):
        x0, x1, y0, y1 = aperture.cx - aperture.r, aperture.cx + aperture.r, aperture.cy - aperture.r, aperture.cy + aperture.r
    else:
        xs, ys = [p[0] for p in aperture.points], [p[1] for p in aperture.points]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    def at(u: float, v: float) -> tuple[float, float]:
        return x0 + (x1 - x0) * u, y0 + (y1 - y0) * v

    grid = [at(i / GRID_SAMPLES, j / GRID_SAMPLES) for i in range(GRID_SAMPLES + 1) for j in range(GRID_SAMPLES + 1)]
    inside = surface.in_clear([p[0] for p in grid], [p[1] for p in grid], -ON_APERTURE_MM)
    best: tuple[float, float, float] | None = None
    for k, (x, y) in enumerate(grid):
        if not inside[k]:
            continue
        value = t(x, y)
        if value is not None and (best is None or value < best[0]):
            best = (value, (k // (GRID_SAMPLES + 1)) / GRID_SAMPLES, (k % (GRID_SAMPLES + 1)) / GRID_SAMPLES)
    if best is None:
        return None

    def neg(u: float, v: float) -> float | None:
        x, y = at(u, v)
        if not surface.in_clear([x], [y], -ON_APERTURE_MM)[0]:
            return None
        value = t(x, y)
        return None if value is None else -value

    u, v, value = _climb(neg, best[1], best[2], 1.0 / GRID_SAMPLES)
    return -value, at(u, v)


def _profile_crossing(front: Any, back: Any) -> tuple[float, tuple[float, float], Any] | None:
    """The smallest distance from ``front``'s own profile to ``back``'s (neither held flat beyond its edge) over the clear
    aperture of either surface, where it is, and the surface whose clear aperture holds that point: the thickness light
    entering the clear aperture there meets."""

    def t(x: float, y: float) -> float | None:
        a, b = front.profile_z(x, y), back.profile_z(x, y)
        return None if a is None or b is None else b - a

    best: tuple[float, tuple[float, float], Any] | None = None
    for surface in (front, back):
        aperture = surface.aperture
        rotational = (surface.region is None and isinstance(aperture, Circle) and aperture.cx == 0.0 and aperture.cy == 0.0
                      and front.profile.rotational and back.profile.rotational)
        found = _clear_minimum(t, surface, rotational)
        if found is not None and (best is None or found[0] < best[0]):
            best = (found[0], found[1], surface)
    return best


def _thickness_issues(body: OpticalBody, band: EdgeThicknessBand = DEFAULT_EDGE_BAND) -> tuple[list[SolidIssue], list[dict[str, Any]]]:
    issues: list[SolidIssue] = []
    facts: list[dict[str, Any]] = []
    surfaces = body.surfaces
    if body.model_kind != "SURFACE_SEQUENCE" or len(surfaces) < 2 or body.outline is None:
        return issues, facts
    for k, medium in enumerate(body.media):
        slab = Slab(medium, surfaces[k], surfaces[k + 1], body.gap_outlines[k] if body.gap_outlines else body.outline)
        info = _gap_thickness(slab)
        if info["minimum"] is None:
            continue
        x, y = info["at"]
        glass = medium not in SURROUNDINGS
        crossing = _profile_crossing(surfaces[k], surfaces[k + 1]) if glass else None
        fact = {"from_surface": surfaces[k].surface_id, "to_surface": surfaces[k + 1].surface_id, "medium": medium,
                "centre_thickness_mm": None if info["centre"] is None else _r(info["centre"]),
                "min_thickness_mm": _r(info["minimum"]), "min_at_mm": list(_rv((x, y))),
                "edge_thickness_mm": None if info["edge"] is None else _r(info["edge"])}
        if crossing is not None:
            fact["clear_aperture_min_thickness_mm"] = _r(crossing[0])
        facts.append(fact)
        zone = "inside the clear aperture" if _in_aperture(surfaces[k].aperture, x, y) and _in_aperture(surfaces[k + 1].aperture, x, y) \
            else "between the clear aperture and the outline"
        z = surfaces[k].z(x, y)
        where = _rv((x, y, z if z is not None else surfaces[k].vertex_z))
        what = f"the {medium} between {surfaces[k].surface_id} and {surfaces[k + 1].surface_id}" if glass \
            else f"the air gap between {surfaces[k].surface_id} and {surfaces[k + 1].surface_id}"
        if info["minimum"] < -TOLERANCE_MM:
            # the finding concerns a clear aperture when the body is negative at a point of one, or the two profiles cross
            # inside one (the minimum reported may then lie farther out)
            own = crossing if glass else _profile_crossing(surfaces[k], surfaces[k + 1])
            in_clear = (_in_aperture(surfaces[k].aperture, x, y) or _in_aperture(surfaces[k + 1].aperture, x, y)
                        or (own is not None and own[0] < -TOLERANCE_MM))
            issues.append(SolidIssue(
                "SURFACES_INTERSECT", ERROR,
                f"surfaces {surfaces[k].surface_id} and {surfaces[k + 1].surface_id} cross: {what} is {_r(info['minimum'], 4):g} mm thick "
                f"at ({_r(x, 4):g}, {_r(y, 4):g}) mm, {zone}",
                body.instance_id, body.member_id, surfaces[k].surface_id, other_surface_id=surfaces[k + 1].surface_id,
                local_position_mm=where, value_mm=_r(info["minimum"]), zone=_zone(in_clear)))
        elif crossing is not None and crossing[0] < -TOLERANCE_MM:
            issues.append(_crossing_issue(body, surfaces[k], surfaces[k + 1], crossing, info["minimum"], what))
        elif glass and info["minimum"] <= TOLERANCE_MM:
            issues.append(SolidIssue(
                "KNIFE_EDGE", WARNING,
                f"{what} thins to zero at ({_r(x, 4):g}, {_r(y, 4):g}) mm: the surfaces meet in a knife edge",
                body.instance_id, body.member_id, surfaces[k].surface_id, other_surface_id=surfaces[k + 1].surface_id,
                local_position_mm=where, value_mm=_r(info["minimum"])))
        elif glass and info["edge"] is not None:
            diameter = 2.0 * min(_clear_half(surfaces[k].aperture), _clear_half(surfaces[k + 1].aperture))
            limit = band.limit(body.catalog)
            if diameter > 0.0 and info["edge"] > band.catalog_ratio * diameter:
                # beyond every published product: the sign of an outline that is not the element's edge
                issues.append(SolidIssue(
                    "EDGE_THICKNESS_IMPLAUSIBLE", WARNING,
                    f"{_element(body)}: {what} is {_r(info['edge'], 4):g} mm thick at its edge, {info['edge'] / diameter:.0%} of its clear "
                    f"diameter {_r(diameter, 4):g} mm (more than {band.catalog_ratio:.0%}); a surface edged smaller "
                    f"than the outline states its own {'edge_radius_mm' if isinstance(body.outline, Circle) else 'edge_half_widths_mm'}",
                    body.instance_id, body.member_id, surfaces[k].surface_id, other_surface_id=surfaces[k + 1].surface_id,
                    value_mm=_r(info["edge"]), limit_mm=_r(band.catalog_ratio * diameter)))
            elif diameter > 0.0 and info["edge"] > limit * diameter:
                # a declared model above the review ratio: a fact for review, not a warning
                issues.append(SolidIssue(
                    "EDGE_THICKNESS_IMPLAUSIBLE", INFO,
                    f"{_element(body)}: {what} is {_r(info['edge'], 4):g} mm thick at its edge, {info['edge'] / diameter:.0%} of its clear "
                    f"diameter {_r(diameter, 4):g} mm, above the review ratio {limit:.0%}",
                    body.instance_id, body.member_id, surfaces[k].surface_id, other_surface_id=surfaces[k + 1].surface_id,
                    value_mm=_r(info["edge"]), limit_mm=_r(limit * diameter)))
    issues += _edge_margin_issues(body)
    return issues, facts


def _element(body: OpticalBody) -> str:
    return f"element {body.instance_id}" if body.member_id is None else f"element {body.instance_id} member {body.member_id}"


def _crossing_issue(body: OpticalBody, front: Any, back: Any, crossing: tuple[float, tuple[float, float], Any], drawn: float,
                    what: str) -> SolidIssue:
    """``SURFACES_INTERSECT`` inside a clear aperture: the two profiles cross there, and the drawn body stays positive
    only because a surface is held flat beyond its own edge."""

    value, (x, y), holder = crossing
    held = [s for s in (front, back) if s.edge is not None and not s.edge_outline.contains(x, y, margin=-ON_EDGE_MM)]
    because = (f"; the drawn body keeps {_r(drawn, 4):g} mm only because {' and '.join(s.surface_id for s in held)} "
               f"{'is' if len(held) == 1 else 'are'} held flat beyond {'its own edge' if len(held) == 1 else 'their own edges'} "
               f"({', '.join(f'{_r(s.edge[0], 4):g} mm' for s in held)})") if held else ""
    z = front.profile_z(x, y)
    return SolidIssue(
        "SURFACES_INTERSECT", ERROR,
        f"surfaces {front.surface_id} and {back.surface_id} cross inside the clear aperture of {holder.surface_id}: {what} is "
        f"{_r(value, 4):g} mm thick at ({_r(x, 4):g}, {_r(y, 4):g}) mm between their profiles{because}. Light entering the clear "
        f"aperture there has no glass of that thickness: thicken the element, or make the clear apertures fit inside the "
        f"surfaces' edges", body.instance_id, body.member_id, front.surface_id, other_surface_id=back.surface_id,
        local_position_mm=_rv((x, y, z if z is not None else front.vertex_z)), value_mm=_r(value), zone=ZONE_CLEAR_APERTURE)


def _clear_half(aperture: Circle | Polygon) -> float:
    """The smaller half width of a clear aperture (a circle: its radius)."""

    if isinstance(aperture, Circle):
        return aperture.r
    xs, ys = [p[0] for p in aperture.points], [p[1] for p in aperture.points]
    return min(max(xs) - min(xs), max(ys) - min(ys)) / 2.0


def _edge_margin_issues(body: OpticalBody) -> list[SolidIssue]:
    """``EDGE_FAR_BEYOND_CLEAR_APERTURE`` (warning): a surface bounding glass whose glass outline reaches more than
    ``EDGE_TO_CLEAR_RATIO_LIMIT`` times its clear radius out, and by more than ``EDGE_MARGIN_MIN_MM`` (round elements with
    centred circular apertures)."""

    issues: list[SolidIssue] = []
    if not isinstance(body.outline, Circle):
        return issues
    last = len(body.surfaces) - 1
    for k, surface in enumerate(body.surfaces):
        glass = [g for g in ((k - 1) if k > 0 else None, k if k < last else None) if g is not None and body.media[g] not in SURROUNDINGS]
        aperture = surface.aperture
        if not glass or not isinstance(aperture, Circle) or aperture.cx != 0.0 or aperture.cy != 0.0:
            continue
        outlines = [body.gap_outlines[g] if body.gap_outlines else body.outline for g in glass]
        reach = max(o.r for o in outlines if isinstance(o, Circle))
        if reach > EDGE_TO_CLEAR_RATIO_LIMIT * aperture.r and reach - aperture.r > EDGE_MARGIN_MIN_MM:
            issues.append(SolidIssue(
                "EDGE_FAR_BEYOND_CLEAR_APERTURE", WARNING,
                f"surface {surface.surface_id}'s glass reaches {_r(reach, 4):g} mm from the axis, {reach / aperture.r:.2f} times its clear "
                f"radius {_r(aperture.r, 4):g} mm (more than {EDGE_TO_CLEAR_RATIO_LIMIT:g} times); a surface edged smaller than the "
                "outline states its own edge_radius_mm",
                body.instance_id, body.member_id, surface.surface_id, value_mm=_r(reach), limit_mm=_r(EDGE_TO_CLEAR_RATIO_LIMIT * aperture.r)))
    return issues


def _coaxial_sides(surface: Any) -> tuple[str, str] | None:
    """(medium on the -Z side, medium on the +Z side) of a compiled coaxial Engine surface (``None`` when it is not one)."""

    positive, negative = material_name(surface.medium_positive), material_name(surface.medium_negative)
    if getattr(surface, "geometry", None) == "sphere":
        axis = surface.cap_axis
        if abs(axis[0]) > PARALLEL_TOLERANCE or abs(axis[1]) > PARALLEL_TOLERANCE:
            return None
        # positive side = outside the sphere; near the vertex that is the side away from the centre
        return (positive, negative) if axis[2] < 0.0 else (negative, positive)
    normal = getattr(surface, "normal", None)
    if normal is None or abs(normal[0]) > PARALLEL_TOLERANCE or abs(normal[1]) > PARALLEL_TOLERANCE:
        return None
    return (negative, positive) if normal[2] > 0.0 else (positive, negative)


def _medium_issues(body: OpticalBody) -> list[SolidIssue]:
    issues: list[SolidIssue] = []
    if body.model_kind == "SURFACE_SEQUENCE" and body.engine_surfaces:
        sides = [_coaxial_sides(s) for s in body.engine_surfaces]
        last = len(sides) - 1
        expected = [((AIR if k == 0 else body.media[k - 1]), (AIR if k == last else body.media[k])) for k in range(len(sides))]
        for k, (side, surface) in enumerate(zip(sides, body.engine_surfaces, strict=True)):
            if side is None:
                continue
            if side != tuple(expected[k]):
                issues.append(SolidIssue(
                    "MEDIUM_BOUNDARY_INCONSISTENT", ERROR,
                    f"surface {surface.surface_id} separates {side[0]} (-Z side) from {side[1]} (+Z side), but the element's media "
                    f"put {expected[k][0]} before it and {expected[k][1]} after it",
                    body.instance_id, body.member_id, surface.surface_id))
            elif k < last and sides[k + 1] is not None and side[1] != sides[k + 1][0]:  # type: ignore[index]
                issues.append(SolidIssue(
                    "MEDIUM_BOUNDARY_INCONSISTENT", ERROR,
                    f"surface {surface.surface_id} has {side[1]} after it but surface {body.engine_surfaces[k + 1].surface_id} has "
                    f"{sides[k + 1][0]} before it", body.instance_id, body.member_id, surface.surface_id))  # type: ignore[index]
    if body.model_kind == "SURFACE_SET":
        solid = body.solid
        for surface in body.engine_surfaces:
            front, back = material_name(surface.medium_positive), material_name(surface.medium_negative)
            if solid is None:
                if front not in SURROUNDINGS or back not in SURROUNDINGS:
                    issues.append(SolidIssue(
                        "MEDIUM_BOUNDARY_INCONSISTENT", ERROR,
                        f"surface {surface.surface_id} has {front} in front and {back} behind but the element has no solid body to hold them",
                        body.instance_id, body.member_id, surface.surface_id))
                continue
            normal, origin = surface.normal, surface.origin_mm
            on = [(n, d) for n, d in solid.planes if abs(sum(a * b for a, b in zip(n, origin)) - d) <= ON_FACE_MM
                  and abs(abs(sum(a * b for a, b in zip(n, normal))) - 1.0) <= ON_FACE_MM]
            medium = material_name(solid.medium)
            if on:
                outward = sum(a * b for a, b in zip(on[0][0], normal)) > 0.0
                want = (AIR, medium) if outward else (medium, AIR)
            else:
                want = (medium, medium)
            if (front, back) != want:
                issues.append(SolidIssue(
                    "MEDIUM_BOUNDARY_INCONSISTENT", ERROR,
                    f"surface {surface.surface_id} has {front} in front and {back} behind, but it {'lies on a face' if on else 'lies inside'} of "
                    f"the {medium} body (expected {want[0]} in front and {want[1]} behind)", body.instance_id, body.member_id, surface.surface_id))
    return issues


def element_issues(body: OpticalBody, edge_band: EdgeThicknessBand = DEFAULT_EDGE_BAND) -> tuple[list[SolidIssue], list[dict[str, Any]]]:
    """The checks of one body on its own: profile domain over the outline, thickness between consecutive surfaces over the
    whole outline (glass and air) and between their own profiles over each clear aperture, and the medium chain of its
    compiled surfaces. Returns the issues and the thickness facts of every gap."""

    thickness, facts = _thickness_issues(body, edge_band)
    return _domain_issues(body) + thickness + _medium_issues(body), facts


# ------------------------------------------------------------------------------------------------ pairs


@dataclass(frozen=True)
class Contact:
    """The deepest shared point of two bodies: ``depth`` (mm, positive: overlap), its design-frame ``position`` and
    what of each body meets there."""

    depth: float
    position: Vec3
    where_a: str
    where_b: str


def _coaxial(a: PlacedBody, b: PlacedBody) -> int:
    """+1 / -1 when the two bodies' local Z axes lie on one line (same / opposite direction), else 0."""

    xa, xb = a.axis, b.axis
    cross = (xa[1] * xb[2] - xa[2] * xb[1], xa[2] * xb[0] - xa[0] * xb[2], xa[0] * xb[1] - xa[1] * xb[0])
    if math.sqrt(sum(c * c for c in cross)) > PARALLEL_TOLERANCE:
        return 0
    d = tuple(pb - pa for pa, pb in zip(a.position_mm, b.position_mm))
    along = sum(c * x for c, x in zip(d, xa))
    off = tuple(c - along * x for c, x in zip(d, xa))
    if math.sqrt(sum(c * c for c in off)) > COAXIAL_TOLERANCE_MM:
        return 0
    return 1 if sum(p * q for p, q in zip(xa, xb)) > 0.0 else -1


def _radial(piece) -> tuple[float, Callable[[float], tuple[float, float] | None], str]:
    """A rotational piece as (outer radius, rho -> (z_low, z_high), description)."""

    if isinstance(piece, Slab):
        def span(rho: float, piece=piece) -> tuple[float, float] | None:
            lo, hi = piece.front.z(rho, 0.0), piece.back.z(rho, 0.0)
            return None if lo is None or hi is None else (lo, hi)
        return piece.outline.r, span, f"{piece.front.surface_id}-{piece.back.surface_id} {piece.medium}"  # type: ignore[union-attr]
    if isinstance(piece, Plate):
        return piece.solid.side_radius, piece.radial_span, "stop plate"  # type: ignore[return-value]
    lo, hi = piece.z_range()
    return piece.solid.side_radius, (lambda rho, lo=lo, hi=hi: (lo, hi)), "body"


def _coaxial_contact(a: PlacedBody, b: PlacedBody, sign: int) -> Contact | None:
    """Exact for two solids of revolution on one axis: at every radius both reach, the axial length their solids share
    (the sum over their pieces' overlapping spans; the largest separation when none overlap), maximized over the radius
    (dense samples, golden-section refined)."""

    s0 = sum((pb - pa) * x for pa, pb, x in zip(a.position_mm, b.position_mm, a.axis))
    pieces_a = [_radial(p) for p in a.body.pieces]
    pieces_b = [_radial(p) for p in b.body.pieces]
    limit = min(max(r for r, _, _ in pieces_a), max(r for r, _, _ in pieces_b))

    def pairs(rho: float):
        for ra, fa, wa in pieces_a:
            if ra < rho:
                continue
            sa = fa(rho)
            if sa is None:
                continue
            for rb, fb, wb in pieces_b:
                if rb < rho:
                    continue
                sb = fb(rho)
                if sb is None:
                    continue
                lo_b, hi_b = (s0 + sb[0], s0 + sb[1]) if sign > 0 else (s0 - sb[1], s0 - sb[0])
                yield min(sa[1], hi_b) - max(sa[0], lo_b), max(sa[0], lo_b), min(sa[1], hi_b), wa, wb

    def shared(rho: float) -> float | None:
        values = [v for v, *_ in pairs(rho)]
        if not values:
            return None
        positive = [v for v in values if v > 0.0]
        return sum(positive) if positive else max(values)

    found = _maximize_1d(shared, 0.0, limit)
    if found is None:
        return None
    rho, depth = found
    value, lo, hi, wa, wb = max(pairs(rho), key=lambda item: item[0])
    return Contact(depth, a.to_world((rho, 0.0, 0.5 * (lo + hi))), wa, wb)


def _samples(placed: PlacedBody) -> list[tuple[Patch, float, float, Vec3]]:
    out = []
    for piece in placed.body.pieces:
        for patch in piece.patches():
            nu, nv = patch.grid
            for i in range(nu + 1):
                for j in range(nv + 1):
                    p = patch.point(i / nu, j / nv)
                    if p is not None:
                        out.append((patch, i / nu, j / nv, placed.to_world(p)))
    return out


def _sampled_contact(a: PlacedBody, b: PlacedBody) -> Contact | None:
    """Two bodies in any placement: every sampled boundary point of each is tested inside the other, and the deepest
    candidates are refined on their boundary patch (pattern search) until the step is below 1e-9 of the patch."""

    best: Contact | None = None
    for src, dst in ((a, b), (b, a)):
        heap: list[tuple[float, int, Patch, float, float]] = []
        for n, (patch, u, v, w) in enumerate(src.samples):
            local = dst.to_local(w)
            if not dst.near_box(local):
                continue
            d = dst.body.depth(local)
            if d == -math.inf:
                continue
            item = (d, n, patch, u, v)
            if len(heap) < CANDIDATES:
                heapq.heappush(heap, item)
            elif d > heap[0][0]:
                heapq.heapreplace(heap, item)
        for _, _, patch, u, v in heap:
            def depth(uu: float, vv: float, patch=patch) -> float | None:
                p = patch.point(uu, vv)
                if p is None:
                    return None
                return dst.body.depth(dst.to_local(src.to_world(p)))

            uu, vv, d = _climb(depth, u, v, 1.0 / max(patch.grid))
            if best is None or d > best.depth:
                p = patch.point(uu, vv)
                point = src.to_world(p)  # type: ignore[arg-type]
                where_dst = _deepest_piece(dst, point)
                best = Contact(d, point, patch.where if src is a else where_dst, where_dst if src is a else patch.where)
    return best


def _deepest_piece(placed: PlacedBody, world: Vec3) -> str:
    local = placed.to_local(world)
    pieces = sorted(placed.body.pieces, key=lambda p: -p.depth(local))
    if not pieces:
        return "body"
    piece = pieces[0]
    if isinstance(piece, Slab):
        return f"{piece.front.surface_id}-{piece.back.surface_id} {piece.medium}"
    if isinstance(piece, Convex):
        return {"MODEL_BODY": "body", UNIT_ENVELOPE: "envelope"}.get(piece.source, "design body")
    if isinstance(piece, Plate):
        return "stop plate"
    return piece.surface_id


def contact(a: PlacedBody, b: PlacedBody) -> Contact | None:
    """The deepest shared point of two placed bodies (``None`` when they are clearly apart)."""

    gap = math.sqrt(sum((p - q) ** 2 for p, q in zip(a.position_mm, b.position_mm)))
    if gap > a.body.radius + b.body.radius + TOLERANCE_MM:
        return None
    sign = _coaxial(a, b)
    if sign and a.body.rotational and b.body.rotational:
        return _coaxial_contact(a, b, sign)
    return _sampled_contact(a, b)


def _crossing_contact(a: PlacedBody, b: PlacedBody, found: Contact | None) -> tuple[Vec3, str, str] | None:
    """Where two touching bodies meet on a traced surface of each inside its clear aperture -- light would cross there
    from one body into the other with no air between them: ``(design-frame point, surface of a, surface of b)``, else
    ``None``. Checked at the deepest contact point and, for two bodies on one axis, at the axial points of their traced
    surfaces (a beam along the axis passes there)."""

    candidates = [] if found is None else [found.position]
    if _coaxial(a, b):
        for _, za in a.body.axis_crossings():
            pa = a.to_world((0.0, 0.0, za))
            for _, zb in b.body.axis_crossings():
                pb = b.to_world((0.0, 0.0, zb))
                if math.sqrt(sum((p - q) ** 2 for p, q in zip(pa, pb))) <= TOLERANCE_MM:
                    candidates.append(pa)
    for point in candidates:
        sa = a.body.traced_surface_at(a.to_local(point), TOLERANCE_MM)
        sb = b.body.traced_surface_at(b.to_local(point), TOLERANCE_MM)
        if sa is not None and sb is not None:
            return point, sa, sb
    return None


def pair_issues(a: PlacedBody, b: PlacedBody) -> list[SolidIssue]:
    found = contact(a, b)
    if found is None or found.depth <= TOLERANCE_MM:
        # not overlapping: is it a contact where light would cross from one body into the other?
        crossing = _crossing_contact(a, b, found if found is not None and found.depth >= -TOLERANCE_MM else None)
        if crossing is not None:
            point, sa, sb = crossing
            where = _rv(point)
            return [SolidIssue(
                "OPTICAL_BODIES_TOUCH", ERROR,
                f"{a.body.label} and {b.body.label} touch at ({', '.join(f'{v:g}' for v in _rv(where, 4))}) mm where light crosses "
                f"from one into the other ({a.body.label}: surface {sa}; {b.body.label}: surface {sb}) with no air gap: the trace "
                f"cannot pass two surfaces at one point. Merge them into one SURFACE_SEQUENCE with a shared (cemented) surface, "
                f"or separate them by more than {TOLERANCE_MM:g} mm", a.body.instance_id, a.body.member_id, surface_id=sa,
                other_instance_id=b.body.instance_id, other_member_id=b.body.member_id, other_surface_id=sb, position_mm=where,
                value_mm=0.0)]
    if found is None or found.depth < -TOLERANCE_MM:
        return []
    position = _rv(found.position)
    if found.depth > TOLERANCE_MM:
        return [SolidIssue(
            "OPTICAL_BODIES_OVERLAP", ERROR,
            f"{a.body.label} and {b.body.label} overlap by {_r(found.depth, 4):g} mm at ({', '.join(f'{v:g}' for v in _rv(position, 4))}) mm "
            f"({a.body.label}: {found.where_a}; {b.body.label}: {found.where_b}): two optical bodies cannot share volume",
            a.body.instance_id, a.body.member_id, other_instance_id=b.body.instance_id, other_member_id=b.body.member_id,
            position_mm=position, value_mm=_r(found.depth))]
    return [SolidIssue(
        "OPTICAL_BODIES_TOUCH", INFO,
        f"{a.body.label} and {b.body.label} touch at ({', '.join(f'{v:g}' for v in _rv(position, 4))}) mm",
        a.body.instance_id, a.body.member_id, other_instance_id=b.body.instance_id, other_member_id=b.body.member_id,
        position_mm=position, value_mm=_r(found.depth))]


# ------------------------------------------------------------------------------------------------ envelopes


def intrusion_issues(instance_id: str, envelope: Any, position_mm: Vec3, quaternion_wxyz: tuple[float, float, float, float],
                     others: Sequence[PlacedBody]) -> list[SolidIssue]:
    """No optical body other than the unit's own members lies inside a compound unit's ``envelope`` (its whole-unit
    boundary, a design geometry in the unit frame): the same shared-volume measure as ``pair_issues`` (exact on one axis,
    sampled and refined otherwise); touching the envelope is allowed. Reported as ``OPTICAL_BODIES_OVERLAP`` on the
    intruding component with ``other_instance_id`` the unit and ``other_region`` ``UNIT_ENVELOPE``."""

    body = envelope_body(instance_id, envelope)
    if body is None:
        return []
    hull = PlacedBody(body, position_mm, quaternion_wxyz)
    issues = []
    for other in others:
        found = contact(other, hull)
        if found is None or found.depth <= TOLERANCE_MM:
            continue
        position = _rv(found.position)
        issues.append(SolidIssue(
            "OPTICAL_BODIES_OVERLAP", ERROR,
            f"{other.body.label} lies {_r(found.depth, 4):g} mm inside the envelope of compound unit {instance_id} at "
            f"({', '.join(f'{v:g}' for v in _rv(position, 4))}) mm ({other.body.label}: {found.where_a}): the envelope is the unit's "
            f"whole boundary and only its members may occupy it -- place {other.body.instance_id} outside the unit (before its entrance "
            f"or past its exit along its axis) or make it a member", other.body.instance_id, other.body.member_id,
            other_instance_id=instance_id, other_region=UNIT_ENVELOPE, position_mm=position, value_mm=_r(found.depth),
            zone=ZONE_UNIT_ENVELOPE))
    return issues


def envelope_issues(instance_id: str, envelope: Any, position_mm: Vec3, quaternion_wxyz: tuple[float, float, float, float],
                    members: Sequence[PlacedBody]) -> list[SolidIssue]:
    """Every member body of a compound unit lies inside the unit's ``envelope`` (a design geometry in the unit frame, the
    unit placed at ``position_mm`` / ``quaternion_wxyz``): the largest distance of a member's boundary outside it, found
    by sampling and refinement."""

    solid = design_solid(envelope)
    if solid is None:
        return []
    hull = Convex(solid, None, "DESIGN_GEOMETRY")
    unit = PlacedBody(OpticalBody(instance_id, None, "COMPOUND", ()), position_mm, quaternion_wxyz)
    issues = []
    for member in members:
        worst: tuple[float, Vec3] | None = None
        for patch, u, v, w in _samples(member):
            outside = -hull.depth(unit.to_local(w))
            if worst is None or outside > worst[0]:
                worst = (outside, w)
                best_patch, best_u, best_v = patch, u, v
        if worst is None:
            continue

        def out(uu: float, vv: float, patch=best_patch) -> float | None:  # noqa: B023 (bound at definition)
            p = patch.point(uu, vv)
            return None if p is None else -hull.depth(unit.to_local(member.to_world(p)))

        uu, vv, value = _climb(out, best_u, best_v, 1.0 / max(best_patch.grid))
        if value > TOLERANCE_MM:
            p = best_patch.point(uu, vv)
            position = _rv(member.to_world(p))  # type: ignore[arg-type]
            issues.append(SolidIssue(
                "MEMBER_OUTSIDE_ENVELOPE", ERROR,
                f"member {member.body.member_id} reaches {_r(value, 4):g} mm outside the envelope of {instance_id} at "
                f"({', '.join(f'{c:g}' for c in _rv(position, 4))}) mm", instance_id, member.body.member_id, position_mm=position,
                value_mm=_r(value), zone=ZONE_UNIT_ENVELOPE))
    return issues
