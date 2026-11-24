"""The exact meridional section of optical elements: numbers, not a rendering of the mesh.

``optical_section(elements, reference_id, plane)`` cuts the element ``reference_id`` with its local ``XZ`` plane (y = 0)
or ``YZ`` plane (x = 0) and draws in that plane:

* every surface's profile ``z = vertex + sag`` along the transverse coordinate ``t`` over the element outline, sampled
  so that its chords stay within ``CHORD_TOLERANCE_MM`` (exact sag at every point), split into its clear-aperture spans
  and its opaque-rim spans, with the clear-aperture limits as ticks; a surface with its own mechanical edge is drawn to
  its edge, then flat at its rim height out to the outline of the glass it bounds (its flange, part of its rim);
* the media between consecutive surfaces as closed polygons (front profile, edge, back profile, edge) over the gap's
  outline;
* a surface set's solid as its exact section polygon and the traces of its surfaces; a thin (polarization) element as
  its plane and plate;
* every other element on the same axis (a lens train, a compound objective), placed along the axis exactly where the
  design puts it (``axial_offset_mm``), turned around when it faces the other way; an element off the axis, tilted,
  or turned about the axis so that the plane does not cut it along one of its own section planes is listed in
  ``excluded`` with the reason;
* the Engine's chief and pupil rays of a run projected orthogonally onto the plane (``max_out_of_plane_mm`` tells how
  far each one leaves it: 0 for a ray in the plane);
* every observation plane square to the axis and centred in the section plane (the image plane), as a line at its
  place along the axis over its aperture; any other is listed in ``excluded``.

``system_sections(elements, plane)`` is the same section of a whole system: one per axis, referred to the element
the light reaches first on it. ``section_layout`` frames a drawing on the optics, the whole scene or an explicit box
of the design frame; the image takes the box's aspect within the size asked for, at one scale on both axes.

Coordinates: ``z`` along the reference element's local +Z (its origin at 0), ``t`` along its local +X (``XZ``) or +Y
(``YZ``); millimetres; equal scale on both axes in ``section_svg``. Pure Python (no NumPy), deterministic.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._vendor.shared_contracts.optical_view import OpticBodyGeometryV1, OpticSurfaceV1
from . import geometry as g
from .solids import section_polygon
from .surface_mesh import CHORD_TOLERANCE_MM, Profile, held_point, profile_stations, substrate_face_z

SECTION_SCHEMA = "opensci.optical_section"
SECTION_VERSION = "1.0"
PLANES = ("XZ", "YZ")
DIGITS = 6
AXIS_TOLERANCE = 1e-9
OFFSET_TOLERANCE_MM = 1e-6
QUARTER_TOLERANCE = 1e-9
TRACE_SAMPLES = 256
BISECTIONS = 60
ON_EDGE_MM = 1e-12
RELATIVE_EPS = 1e-12
ALL_ELEMENTS = "ALL"  # the id that asks for the section of the whole system
# A far observation plane (e.g. a 1 m far-field plane) stays out of a frame on the optics when it lies farther than this
# many optics extents from them.
FAR_PLANE_FACTOR = 2.5
THIN_LENS_SYMBOL_FRACTION = 0.08  # the arrow heads of a thin lens: this fraction of its outline radius
#: A grating's rules drawn on a section across its grooves, how tall against their spacing, and the least cosine between the
#: section line and the grating vector for the section to count as across them.
GRATING_SECTION_RULES = 21
GRATING_SECTION_TICK_FRACTION = 0.8
GRATING_SECTION_MIN_COSINE = 0.5
ALONG_AXIS_COS = 0.5  # a ray segment closer than 60 deg to the axis tells which way the light runs along it


class SectionError(ValueError):
    """The section cannot be drawn as asked (unknown element, no optical body, bad plane)."""


@dataclass(frozen=True)
class SectionElement:
    """An element to cut: its id, label, design-frame pose and ``OPTIC_BODY`` geometry (``None``: not an optical body).

    A ``COMPOUND`` unit is one frame-only entry (``unit_frame``: its id and pose, the reference of a section through the
    whole unit) plus one entry per member, ``<unit>/<member>``, at the member's design-frame pose (unit pose then member
    pose) with the member's body."""

    element_id: str
    label: str
    position_mm: g.Vec3
    quaternion_wxyz: g.Quat
    geometry: OpticBodyGeometryV1 | None
    unit_frame: bool = False
    unit_id: str | None = None
    member_id: str | None = None
    emitting_area: Any = None  # an extended emitter's ``EmittingAreaV1``: no body, drawn as its outline


def section_elements_of(element_id: str, label: str, position_mm: g.Vec3, quaternion_wxyz: g.Quat, geometry: Any) -> list[SectionElement]:
    """The entries of one scene element (``OPTIC_BODY`` / ``OPTICAL_UNIT`` / anything else)."""

    if getattr(geometry, "kind", None) == "OPTICAL_UNIT":
        out = [SectionElement(element_id, label, position_mm, quaternion_wxyz, None, unit_frame=True)]
        for member in geometry.members:
            position, q = g.compose_pose(position_mm, quaternion_wxyz, tuple(member.pose.position_mm), tuple(member.pose.quaternion_wxyz))  # type: ignore[arg-type]
            out.append(SectionElement(f"{element_id}/{member.member_id}", member.member_id, position, q, member.body,
                                      unit_id=element_id, member_id=member.member_id))
        return out
    body = geometry if isinstance(geometry, OpticBodyGeometryV1) else None
    return [SectionElement(element_id, label, position_mm, quaternion_wxyz, body, emitting_area=getattr(geometry, "emitting_area", None))]


def _r(v: float) -> float:
    return round(v, DIGITS) + 0.0


def _pt(t: float, z: float) -> list[float]:
    return [_r(t), _r(z)]


# ----------------------------------------------------------------------------------------------- one element


def _extent(geom: OpticBodyGeometryV1, axis: str) -> float:
    if geom.edge_radius_mm is not None:
        return geom.edge_radius_mm
    hx, hy = geom.edge_half_widths_mm  # type: ignore[misc]
    return hx if axis == "X" else hy


def _xy(axis: str, t: float) -> tuple[float, float]:
    return (t, 0.0) if axis == "X" else (0.0, t)


def _aperture_span(surface: OpticSurfaceV1, axis: str) -> tuple[float, float] | None:
    """The clear aperture's interval along the section line (``None``: the line misses it)."""

    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    along, across = (cx, cy) if axis == "X" else (cy, cx)
    if surface.clear_radius_mm is not None:
        r = surface.clear_radius_mm
        if abs(across) > r:
            return None
        half = math.sqrt(r * r - across * across)
        return along - half, along + half
    hx, hy = surface.clear_half_widths_mm  # type: ignore[misc]
    h_along, h_across = (hx, hy) if axis == "X" else (hy, hx)
    if abs(across) > h_across:
        return None
    return along - h_along, along + h_along


def _stations(profiles: Sequence[tuple[Profile, float]], axis: str, lo: float, hi: float, breaks: Sequence[float], tolerance: float) -> list[float]:
    """Common transverse stations of several coaxial profiles on ``[lo, hi]``: every break, and the chord sampler's
    stations of each profile between consecutive breaks."""

    cuts = sorted({lo, hi, *(b for b in breaks if lo < b < hi)})
    out: set[float] = set(cuts)
    for profile, _ in profiles:
        if profile.is_flat:
            continue
        for a, b in zip(cuts, cuts[1:]):
            out.update(profile_stations(lambda t, p=profile: p.sag(*_xy(axis, t)), a, b, tolerance))
    return sorted(out)


def _surface_curve(surface: OpticSurfaceV1, profile: Profile, axis: str, ts: Sequence[float],
                   edge: tuple[float, float] | None = None, round_edge: bool = True) -> tuple[list[tuple[float, float | None]], bool]:
    z0 = surface.vertex_z_mm or 0.0
    out = []
    undefined = False
    for t in ts:
        sag = profile.sag(*held_point(*_xy(axis, t), edge, round_edge))
        undefined = undefined or sag is None
        out.append((t, None if sag is None else z0 + sag))
    return out, undefined


def _segments(curve: list[tuple[float, float | None]], span: tuple[float, float] | None) -> list[dict[str, Any]]:
    """The curve split into clear-aperture and rim spans (points where the profile is undefined end a span)."""

    segments: list[dict[str, Any]] = []
    for (t0, z0), (t1, z1) in zip(curve, curve[1:]):
        if z0 is None or z1 is None:
            continue
        mid = 0.5 * (t0 + t1)
        zone = "CLEAR_APERTURE" if span is not None and span[0] - ON_EDGE_MM <= mid <= span[1] + ON_EDGE_MM else "RIM"
        if segments and segments[-1]["zone"] == zone and segments[-1]["points"][-1] == _pt(t0, z0):
            segments[-1]["points"].append(_pt(t1, z1))
        else:
            segments.append({"zone": zone, "points": [_pt(t0, z0), _pt(t1, z1)]})
    return segments


def _sequence_section(geom: OpticBodyGeometryV1, axis: str, tolerance: float) -> dict[str, Any]:
    e = _extent(geom, axis)
    k_axis = 0 if axis == "X" else 1
    edged = geom.has_surface_edges
    round_edge = geom.edge_radius_mm is not None
    profiles = [(Profile.of(s), s.vertex_z_mm or 0.0) for s in geom.surfaces]
    spans = [_aperture_span(s, axis) for s in geom.surfaces]
    breaks = [0.0] + [v for span in spans if span is not None for v in span]
    edges = [geom.surface_edge(i) if s.edge_radius_mm is not None or s.edge_half_widths_mm is not None else None
             for i, s in enumerate(geom.surfaces)] if edged else [None] * len(geom.surfaces)
    reach = [geom.glass_outline(i)[k_axis] for i in range(len(geom.surfaces))] if edged else [e] * len(geom.surfaces)
    gaps = [geom.gap_outline(k)[k_axis] for k in range(len(geom.media))] if edged else [e] * len(geom.media)
    if edged:
        breaks += [v for edge in edges if edge is not None for v in (-edge[k_axis], edge[k_axis])]
        breaks += [v for r in (*reach, *gaps) for v in (-r, r)]
    ts = _stations(profiles, axis, -e, e, breaks, tolerance)
    surfaces, curves = [], []
    for index, (surface, (profile, _), span) in enumerate(zip(geom.surfaces, profiles, spans, strict=True)):
        curve, undefined = _surface_curve(surface, profile, axis, ts, edges[index], round_edge)
        curves.append(curve)
        drawn = [(t, z) for t, z in curve if abs(t) <= reach[index] + ON_EDGE_MM]  # out to the outline of its glass
        entry: dict[str, Any] = {"surface_id": surface.surface_id, "role": surface.role.value, "vertex_z_mm": _r(surface.vertex_z_mm or 0.0),
                                 "segments": _segments(drawn, span)}
        for name in ("radius_mm", "conic", "linear_sag_coefficient", "cylinder_axis"):
            if getattr(surface, name) is not None:
                entry[name] = getattr(surface, name)
        if surface.sag is not None:
            entry["sag"] = surface.sag.model_dump(mode="json", exclude_none=True)
        if span is not None:
            ticks = [(t, z) for t in span for t, z in [(t, (surface.vertex_z_mm or 0.0) + (profile.sag(*_xy(axis, t)) or 0.0))]]
            entry["clear_aperture_mm"] = [_r(span[0]), _r(span[1])]
            entry["clear_aperture_ticks"] = [_pt(t, z) for t, z in ticks]
        if edges[index] is not None:
            entry["edge_mm"] = [_r(-edges[index][k_axis]), _r(edges[index][k_axis])]  # type: ignore[index]
        if undefined:
            entry["profile_domain_limit_mm"] = _r(profile.domain_rho)
        surfaces.append(entry)
    media = []
    for k, medium in enumerate(geom.media):
        if medium in ("AIR", "VACUUM"):
            continue
        inside = [abs(t) <= gaps[k] + ON_EDGE_MM for t, _ in curves[k]]
        front = [(t, z) for (t, z), keep in zip(curves[k], inside) if keep and z is not None]
        back = [(t, z) for (t, z), keep in zip(curves[k + 1], inside) if keep and z is not None]
        if len(front) < 2 or len(back) < 2:
            continue
        polygon = [_pt(t, z) for t, z in front] + [_pt(t, z) for t, z in reversed(back)]
        thickness = [(zb - zf) for (_, zf), (_, zb), keep in zip(curves[k], curves[k + 1], inside) if keep and zf is not None and zb is not None]
        media.append({"medium": medium, "from_surface": geom.surfaces[k].surface_id, "to_surface": geom.surfaces[k + 1].surface_id,
                      "polygon": polygon, "centre_thickness_mm": _r(geom.surfaces[k + 1].vertex_z_mm - geom.surfaces[k].vertex_z_mm),  # type: ignore[operator]
                      "edge_thickness_mm": [_r(thickness[0]), _r(thickness[-1])] if thickness else None})
    out: dict[str, Any] = {"surfaces": surfaces, "media": media, "outline_mm": [_r(-e), _r(e)]}
    if len(geom.surfaces) == 1:  # the unmodelled substrate (as the 3D view draws it): to the far face of the design body
        from .optic_mesh import _design_extent_z

        z_lo, z_hi = _design_extent_z(geom)
        curve = [(t, z) for t, z in curves[0] if z is not None]
        if z_hi > z_lo and len(curve) >= 2:
            surface = geom.surfaces[0]
            far = substrate_face_z(surface.vertex_z_mm or 0.0, z_lo, z_hi, surface.role.value == "REFLECT")
            out["substrate"] = {"polygon": [_pt(t, z) for t, z in curve] + [_pt(curve[-1][0], far), _pt(curve[0][0], far)]}
    return out


def _set_section(geom: OpticBodyGeometryV1, axis: str, tolerance: float) -> dict[str, Any]:
    from .optic_mesh import scene_solid

    normal = (0.0, 1.0, 0.0) if axis == "X" else (1.0, 0.0, 0.0)
    out: dict[str, Any] = {"surfaces": [], "media": []}
    solid = scene_solid(geom)
    if solid is not None:
        polygon = section_polygon(solid, (0.0, 0.0, 0.0), normal, segments=g.circle_segments(solid.side_radius or 1.0, tolerance))
        if len(polygon) >= 3:
            out["media"].append({"medium": solid.medium, "polygon": [_pt(p[0] if axis == "X" else p[1], p[2]) for p in polygon]})
    for surface in geom.surfaces:
        trace = _posed_trace(surface, normal, axis, solid)
        entry: dict[str, Any] = {"surface_id": surface.surface_id, "role": surface.role.value, "segments": trace}
        out["surfaces"].append(entry)
    return out


def _posed_trace(surface: OpticSurfaceV1, plane_normal: g.Vec3, axis: str, solid) -> list[dict[str, Any]]:
    """Where a posed surface crosses the section plane, over its clear aperture: for every sampled ``a`` across the
    aperture, the ``b`` on the surface where it meets the plane (bisection), as one polyline."""

    origin, n, u = tuple(surface.origin_mm), g.unit(tuple(surface.normal)), g.unit(tuple(surface.u_axis))  # type: ignore[arg-type]
    v = g.cross(n, u)
    profile = Profile.of(surface)
    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    if surface.clear_radius_mm is not None:
        ha = hb = surface.clear_radius_mm
    else:
        ha, hb = surface.clear_half_widths_mm  # type: ignore[misc]

    def point(a: float, b: float) -> g.Vec3 | None:
        sag = profile.sag(a, b)
        if sag is None:
            return None
        return g.add(origin, g.add(g.add(g.scale(u, a), g.scale(v, b)), g.scale(n, sag)))

    def inside(a: float, b: float) -> bool:
        if surface.clear_radius_mm is not None:
            return (a - cx) ** 2 + (b - cy) ** 2 <= surface.clear_radius_mm ** 2 * (1.0 + RELATIVE_EPS)
        return abs(a - cx) <= ha and abs(b - cy) <= hb

    points: list[list[float]] = []
    for swap in (False, True):  # cut along b at every a, else along a at every b (a plane parallel to one of them)
        found = []
        for k in range(TRACE_SAMPLES + 1):
            s = -1.0 + 2.0 * k / TRACE_SAMPLES
            fixed = (cx + ha * s) if not swap else (cy + hb * s)
            lo, hi = (cy - hb, cy + hb) if not swap else (cx - ha, cx + ha)

            def side(w: float, fixed=fixed, swap=swap) -> float | None:
                p = point(w, fixed) if swap else point(fixed, w)
                return None if p is None else g.dot(p, plane_normal)

            f_lo, f_hi = side(lo), side(hi)
            if f_lo is None or f_hi is None or f_lo * f_hi > 0.0:
                continue
            a, b = lo, hi
            for _ in range(BISECTIONS):
                m = 0.5 * (a + b)
                f_m = side(m)
                if f_m is None:
                    break
                if (f_m <= 0.0) == (f_lo <= 0.0):
                    a, f_lo = m, f_m
                else:
                    b = m
            w = 0.5 * (a + b)
            aa, bb = (w, fixed) if swap else (fixed, w)
            if inside(aa, bb):
                p = point(aa, bb)
                if p is not None:
                    found.append(_pt(p[0] if axis == "X" else p[1], p[2]))
        if len(found) >= 2:
            points = found
            break
    return [{"zone": "CLEAR_APERTURE", "points": points}] if points else []


def _obstruction_span(surface: OpticSurfaceV1, axis: str) -> tuple[float, float] | None:
    """The central obstruction's chord along the section line (a disc of ``obstruction_radius_mm`` about the opening
    centre; ``None`` when the line misses it)."""

    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    along, across = (cx, cy) if axis == "X" else (cy, cx)
    r = surface.obstruction_radius_mm or 0.0
    if abs(across) >= r:
        return None
    half = math.sqrt(r * r - across * across)
    return along - half, along + half


def _plate_extent(geom: OpticBodyGeometryV1, axis: str) -> tuple[float, float, float, float] | None:
    """``(t_min, t_max, z_min, z_max)`` of a thin element's stated body along the section line (``None``: no thickness)."""

    if geom.shape == "CYLINDER" and geom.z_max_mm > geom.z_min_mm:  # type: ignore[operator]
        r = geom.diameter_mm / 2  # type: ignore[operator]
        return -r, r, geom.z_min_mm, geom.z_max_mm  # type: ignore[return-value]
    if geom.shape == "BOX" and geom.half_extents_mm[2] > 0.0:  # type: ignore[index]
        k = 0 if axis == "X" else 1
        c, h = geom.center_mm, geom.half_extents_mm  # type: ignore[assignment]
        return c[k] - h[k], c[k] + h[k], c[2] - h[2], c[2] + h[2]  # type: ignore[index]
    return None


def _thin_lens_symbol(edge: float, z: float, focal_length_mm: float | None) -> list[list[list[float]]]:
    """The heads at both ends of a thin lens's line, as polylines: pointing out for a positive focal length, in for a
    negative one, a flat bar for a tabulated lens (its focal length depends on the wavelength)."""

    a = THIN_LENS_SYMBOL_FRACTION * edge
    if focal_length_mm is None:
        return [[_pt(sign * edge, z - a), _pt(sign * edge, z + a)] for sign in (1.0, -1.0)]
    tip, foot = (edge, edge - a) if focal_length_mm > 0.0 else (edge - a, edge)
    return [[_pt(sign * foot, z - a), _pt(sign * tip, z), _pt(sign * foot, z + a)] for sign in (1.0, -1.0)]


def _grating_rules(geom: OpticBodyGeometryV1, axis: str, span: tuple[float, float], z: float) -> list[list[list[float]]]:
    """The rules a section across the grooves cuts, as short ticks on the ruled plane over the clear aperture (a sign of the
    grooves, not their count); a section along the grooves cuts none."""

    kx, ky = geom.grating.grating_vector_local  # type: ignore[union-attr]
    if abs(kx if axis == "X" else ky) < GRATING_SECTION_MIN_COSINE:
        return []
    start, end = span
    step = (end - start) / GRATING_SECTION_RULES
    height = GRATING_SECTION_TICK_FRACTION * step
    return [[_pt(start + (k + 0.5) * step, z), _pt(start + (k + 0.5) * step, z + height)] for k in range(GRATING_SECTION_RULES)]


def _thin_section(geom: OpticBodyGeometryV1, axis: str) -> dict[str, Any]:
    (surface,) = geom.surfaces
    e = _extent(geom, axis)
    z = surface.vertex_z_mm or 0.0
    span = _aperture_span(surface, axis)
    curve = [(-e, z)] + ([(span[0], z), (span[1], z)] if span is not None else []) + [(e, z)]
    entry = {"surface_id": surface.surface_id, "role": surface.role.value, "vertex_z_mm": _r(z), "segments": _segments(curve, span)}
    if span is not None:
        entry["clear_aperture_mm"] = [_r(span[0]), _r(span[1])]
        entry["clear_aperture_ticks"] = [_pt(span[0], z), _pt(span[1], z)]
    out: dict[str, Any] = {"surfaces": [entry], "media": []}
    stop = surface.role.value == "APERTURE_STOP"
    blocked = None
    if stop:  # light passes the opening: only the plate (and obstruction) is drawn
        entry["segments"] = [seg for seg in entry["segments"] if seg["zone"] == "RIM"]
        blocked = _obstruction_span(surface, axis) if surface.obstruction_radius_mm is not None else None
        if blocked is not None:
            entry["segments"].append({"zone": "OBSTRUCTION", "points": [_pt(blocked[0], z), _pt(blocked[1], z)]})
            entry["obstruction_mm"] = [_r(blocked[0]), _r(blocked[1])]
    if surface.role.value == "THIN_LENS":  # no surfaces to draw: the thin-lens symbol, never a plate
        out["symbol"] = _thin_lens_symbol(e, z, surface.focal_length_mm)
        return out
    if surface.role.value == "GRATING" and span is not None:
        out["symbol"] = _grating_rules(geom, axis, span, z)
    extent = _plate_extent(geom, axis)
    if extent is not None:
        t0, t1, z0, z1 = extent
        pieces = [(t0, t1)]
        if stop and span is not None:  # the plate is open where light passes: on both sides of the opening, and the obstruction
            pieces = [(a, b) for a, b in ((t0, max(t0, span[0])), (min(t1, span[1]), t1)) if b > a]
            if blocked is not None:
                pieces.append(blocked)
        polygons = [[_pt(a, z0), _pt(b, z0), _pt(b, z1), _pt(a, z1)] for a, b in sorted(pieces)]
        if polygons:
            out["plate"] = {"polygon": polygons[0], "polygons": polygons}
    return out


def _facet_section(geom: OpticBodyGeometryV1, axis: str) -> dict[str, Any]:
    """Where the facets of a ``FACET_SOLID`` body cross the section plane: one straight piece per facet the plane cuts, in
    facet order, grouped by the role of the facets (the default ``REFRACT`` first, then each facet group's)."""

    facets = geom.facets
    assert facets is not None
    k_axis = 1 if axis == "X" else 0
    roles = {k: group.role for group in facets.groups for k in group.triangles}
    pieces: dict[str, list[dict[str, Any]]] = {}
    for k, triangle in enumerate(facets.triangles):
        corners = [tuple(facets.vertices_mm[i]) for i in triangle]
        crossing: list[list[float]] = []
        for a, b in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
            da, db = a[k_axis], b[k_axis]
            if (da < 0.0) == (db < 0.0) or da == db:
                if da == 0.0:
                    crossing.append(_pt(a[0] if axis == "X" else a[1], a[2]))
                continue
            t = da / (da - db)
            p = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
            crossing.append(_pt(p[0] if axis == "X" else p[1], p[2]))
        unique = [q for i, q in enumerate(crossing) if q not in crossing[:i]]
        if len(unique) == 2:
            pieces.setdefault(roles.get(k, "REFRACT"), []).append({"zone": "CLEAR_APERTURE", "points": unique})
    surfaces = [{"surface_id": "facets", "role": role, "segments": segments} for role, segments in pieces.items()]
    loops = _section_loops([piece["points"] for segments in pieces.values() for piece in segments])
    return {"surfaces": surfaces, "media": [{"medium": facets.medium, "polygon": loop} for loop in loops]}


def _section_loops(pieces: list[list[list[float]]]) -> list[list[list[float]]]:
    """The closed loops the section pieces of a solid's facets form, each the outline of one area of the solid's medium;
    none where the pieces do not close into separate loops (an end met by other than two pieces: the plane through an edge
    or a face) or where one loop lies inside another (a cavity: the drawing keeps the outlines only)."""

    ends: dict[tuple[float, float], list[int]] = {}
    for k, (a, b) in enumerate(pieces):
        ends.setdefault((a[0], a[1]), []).append(k)
        ends.setdefault((b[0], b[1]), []).append(k)
    if not pieces or any(len(at) != 2 for at in ends.values()):
        return []
    used = [False] * len(pieces)
    loops: list[list[list[float]]] = []
    for start in range(len(pieces)):
        if used[start]:
            continue
        used[start] = True
        loop = [pieces[start][0]]
        point = pieces[start][1]
        while (point[0], point[1]) != (loop[0][0], loop[0][1]):
            loop.append(point)
            nxt = next((k for k in ends[(point[0], point[1])] if not used[k]), None)
            if nxt is None:  # pragma: no cover - every end is met by two pieces: the walk returns to its start
                return []
            used[nxt] = True
            a, b = pieces[nxt]
            point = b if (a[0], a[1]) == (point[0], point[1]) else a
        if len(loop) < 3:
            return []
        loops.append(loop)

    def inside(p: list[float], polygon: list[list[float]]) -> bool:
        hit = False
        for i in range(len(polygon)):
            (x1, y1), (x2, y2) = polygon[i - 1], polygon[i]
            if (y1 > p[1]) != (y2 > p[1]) and p[0] < x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
        return hit

    if any(inside(a[0], b) for i, a in enumerate(loops) for j, b in enumerate(loops) if i != j):
        return []
    return loops


#: How each optic-body model kind's section is drawn: a surface sequence, a surface set, one plane, or facets.
SECTION_DRAWINGS = {
    "SURFACE_SEQUENCE": "SEQUENCE", "SURFACE_SET": "SET", "POLARIZATION_ELEMENT": "PLANE", "APERTURE_STOP": "PLANE",
    "IDEAL_THIN_LENS": "PLANE", "TABULATED_THIN_LENS": "PLANE", "NEUTRAL_DENSITY": "PLANE", "FARADAY_ROTATOR": "PLANE",
    "DIFFRACTION_GRATING": "PLANE", "FACET_SOLID": "FACETS",
}


def element_section(geom: OpticBodyGeometryV1, plane: str = "XZ", *, tolerance: float = CHORD_TOLERANCE_MM) -> dict[str, Any]:
    """The section of one ``OPTIC_BODY`` in its own frame (``t`` along local X for ``XZ``, Y for ``YZ``; ``z`` local)."""

    if plane not in PLANES:
        raise SectionError(f"plane must be one of {list(PLANES)}, got {plane!r}")
    axis = plane[0]
    drawing = SECTION_DRAWINGS[geom.model_kind]
    if drawing == "SET":
        return _set_section(geom, axis, tolerance)
    if drawing == "PLANE":
        return _thin_section(geom, axis)
    if drawing == "FACETS":
        return _facet_section(geom, axis)
    return _sequence_section(geom, axis, tolerance)


# ----------------------------------------------------------------------------------------------- the system


def section_polygons(entry: dict[str, Any], key: str) -> list[list[list[float]]]:
    """The filled polygons of an element entry's ``substrate`` / ``plate`` (a stop's plate is open where light passes: one
    polygon per side of the opening and its obstruction, ``polygons``; anything else one ``polygon``)."""

    part = entry.get(key)
    if not part:
        return []
    return list(part.get("polygons") or [part["polygon"]])


def _rotationally_symmetric(geom: OpticBodyGeometryV1) -> bool:
    return (geom.model_kind != "SURFACE_SET" and geom.edge_radius_mm is not None
            and all(s.cylinder_axis is None and s.sag is None and s.clear_radius_mm is not None and s.aperture_center_mm is None for s in geom.surfaces))


def _map(transform: tuple[float, float, float], curve: list[list[float]]) -> list[list[float]]:
    t_sign, z_sign, offset = transform
    return [_pt(t_sign * t, offset + z_sign * z) for t, z in curve]


def _mapped(section: dict[str, Any], transform: tuple[float, float, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"surfaces": [], "media": []}
    for surface in section["surfaces"]:
        entry = dict(surface)
        entry["segments"] = [{"zone": s["zone"], "points": _map(transform, s["points"])} for s in surface["segments"]]
        if "clear_aperture_ticks" in surface:
            entry["clear_aperture_ticks"] = _map(transform, surface["clear_aperture_ticks"])
        if "clear_aperture_mm" in surface:
            a, b = (transform[0] * v for v in surface["clear_aperture_mm"])
            entry["clear_aperture_mm"] = [_r(min(a, b)), _r(max(a, b))]
        out["surfaces"].append(entry)
    for medium in section["media"]:
        out["media"].append({**medium, "polygon": _map(transform, medium["polygon"])})
    for key in ("substrate", "plate"):
        if key in section:
            out[key] = {"polygon": _map(transform, section[key]["polygon"])}
            if "polygons" in section[key]:
                out[key]["polygons"] = [_map(transform, polygon) for polygon in section[key]["polygons"]]
    if "outline_mm" in section:
        a, b = (transform[0] * v for v in section["outline_mm"])
        out["outline_mm"] = [_r(min(a, b)), _r(max(a, b))]
    if "symbol" in section:
        out["symbol"] = [_map(transform, line) for line in section["symbol"]]
    return out


def _plane_half_extent(plane: Any, t_dir: g.Vec3) -> tuple[float, float]:
    """``(centre offset, half extent)`` of an observation plane's aperture along the section direction (its stated half
    size when it has no aperture)."""

    aperture = plane.aperture
    if aperture is None:
        return 0.0, plane.display_half_extent_mm
    m = g.quat_to_matrix(tuple(plane.pose.quaternion_wxyz))
    cu, cv = g.dot((m[0][0], m[1][0], m[2][0]), t_dir), g.dot((m[0][1], m[1][1], m[2][1]), t_dir)
    centre = aperture.center_offset_mm[0] * cu + aperture.center_offset_mm[1] * cv
    if aperture.shape == "circle":
        return centre, aperture.radius_mm
    if aperture.shape == "rectangle":
        return centre, abs(cu) * aperture.half_width_mm + abs(cv) * aperture.half_height_mm
    from ..optics.apertures import spread  # (any other shape: its exact extent along the section direction)
    from ..scene.lower import lower_aperture
    region = lower_aperture(aperture)
    high, low = spread(region, cu, cv), -spread(region, -cu, -cv)
    return centre + 0.5 * (high + low), 0.5 * (high - low)


def _section_planes(planes: Sequence[Any], origin: g.Vec3, axis0: g.Vec3, t_dir: g.Vec3, normal: g.Vec3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The observation planes square to the axis and centred in the section plane, and the others (``excluded``)."""

    out, excluded = [], []
    for plane in planes:
        m = g.quat_to_matrix(tuple(plane.pose.quaternion_wxyz))
        d = g.sub(tuple(plane.pose.position_mm), origin)
        if abs(abs(g.dot((m[0][2], m[1][2], m[2][2]), axis0)) - 1.0) > AXIS_TOLERANCE or abs(g.dot(d, normal)) > OFFSET_TOLERANCE_MM:
            excluded.append({"element_id": plane.observation_id, "reason": "NOT_ON_AXIS"})
            continue
        centre, half = _plane_half_extent(plane, t_dir)
        t = g.dot(d, t_dir) + centre
        out.append({"observation_id": plane.observation_id, "kind": plane.kind.value, "z_mm": _r(g.dot(d, axis0)), "t_mm": [_r(t - half), _r(t + half)]})
    return out, excluded


def optical_section(elements: Sequence[SectionElement], reference_id: str, plane: str = "XZ", *, beam_paths: Sequence[Any] = (),
                    observation_planes: Sequence[Any] = (), tolerance: float = CHORD_TOLERANCE_MM) -> dict[str, Any]:
    """The section of ``reference_id`` (and of every element on its axis) in its local ``plane``; ``beam_paths``: the
    run's ``BeamPathV1`` list, whose chief and pupil rays are projected onto the plane; ``observation_planes``: the
    run's ``ObservationPlaneV1`` list, drawn where they are square to the axis."""

    if plane not in PLANES:
        raise SectionError(f"plane must be one of {list(PLANES)}, got {plane!r}")
    by_id = {e.element_id: e for e in elements}
    reference = by_id.get(reference_id)
    if reference is None:
        raise SectionError(f"no element {reference_id!r}; elements: {sorted(by_id)}")
    if reference.geometry is None and not reference.unit_frame and reference.emitting_area is None:
        raise SectionError(f"element {reference_id!r} has no optical body to cut (a source, a detector or a symbolic marker)")
    q0 = reference.quaternion_wxyz
    m0 = g.quat_to_matrix(q0)
    axis0 = (m0[0][2], m0[1][2], m0[2][2])
    t_dir = (m0[0][0], m0[1][0], m0[2][0]) if plane == "XZ" else (m0[0][1], m0[1][1], m0[2][1])
    normal = g.cross(axis0, t_dir)  # out of the section plane
    origin = reference.position_mm
    out_elements, excluded = [], []
    emitters = []
    for element in elements:
        if element.emitting_area is not None:
            emitters.append(_emitter_outline(element, origin, t_dir, axis0))
        if element.geometry is None:
            continue
        m = g.quat_to_matrix(element.quaternion_wxyz)
        axis = (m[0][2], m[1][2], m[2][2])
        d = g.sub(element.position_mm, origin)
        along = g.dot(d, axis0)
        off = g.norm(g.sub(d, g.scale(axis0, along)))
        z_sign = g.dot(axis, axis0)
        if abs(abs(z_sign) - 1.0) > AXIS_TOLERANCE or off > OFFSET_TOLERANCE_MM:
            if element.element_id != reference_id:
                excluded.append({"element_id": element.element_id, "reason": "NOT_ON_AXIS"})
            continue
        z_sign = 1.0 if z_sign > 0.0 else -1.0
        local_x, local_y = (m[0][0], m[1][0], m[2][0]), (m[0][1], m[1][1], m[2][1])
        cx, cy = g.dot(local_x, t_dir), g.dot(local_y, t_dir)  # the section direction in the element's own axes
        geom = element.geometry
        if abs(abs(cx) - 1.0) <= QUARTER_TOLERANCE:
            own_plane, t_sign = "XZ", (1.0 if cx > 0.0 else -1.0)
        elif abs(abs(cy) - 1.0) <= QUARTER_TOLERANCE:
            own_plane, t_sign = "YZ", (1.0 if cy > 0.0 else -1.0)
        elif _rotationally_symmetric(geom):
            own_plane, t_sign = "XZ", 1.0
        else:
            excluded.append({"element_id": element.element_id, "reason": "TURNED_ABOUT_THE_AXIS"})
            continue
        section = _mapped(element_section(geom, own_plane, tolerance=tolerance), (t_sign, z_sign, along))
        entry = {"element_id": element.element_id, "label": element.label, "model_kind": geom.model_kind,
                 "axial_offset_mm": _r(along), "flipped": z_sign < 0.0, "own_plane": own_plane, **section}
        if element.member_id is not None:
            entry.update({"unit_id": element.unit_id, "member_id": element.member_id})
        out_elements.append(entry)
    rays = []
    for path in beam_paths:
        for ray in (path.chief_ray, *path.pupil_rays):
            pts, worst = [], 0.0
            for p in ray.points_mm:
                d = g.sub(tuple(p), origin)
                worst = max(worst, abs(g.dot(d, normal)))
                pts.append(_pt(g.dot(d, t_dir), g.dot(d, axis0)))
            entry = {"path_id": path.path_id, "ray_index": ray.ray_index, "role": "CHIEF" if ray.ray_index == 0 else "PUPIL",
                     "display_rgb": list(path.display_rgb), "wavelength_nm": path.wavelength_nm, "points": pts,
                     "max_out_of_plane_mm": _r(worst)}
            source_id = getattr(path, "source_element_id", None)
            if source_id is not None:
                entry["source_element_id"] = source_id
                entry["from_source"] = getattr(path, "parent_path_id", None) is None
            display_role = getattr(path, "display_role", None)
            if display_role is not None:
                entry["display_role"] = display_role.value
            reference_from = getattr(path, "chief_reference_from", None)
            if ray.ray_index == 0 and reference_from is not None:
                entry["reference_from"] = reference_from  # behind a central obstruction: the beam's axis, not light
            rays.append(entry)
    planes, plane_excluded = _section_planes(observation_planes, origin, axis0, t_dir, normal)
    bounds = _bounds(out_elements, rays, planes, emitters)
    extra = {"emitters": emitters} if emitters else {}
    return {**extra, "schema": SECTION_SCHEMA, "schema_version": SECTION_VERSION, "reference_element_id": reference_id, "plane": plane,
            "length_unit": "mm", "chord_tolerance_mm": tolerance,
            "axes": {"z": f"along local +Z of {reference_id}, from its origin", "t": f"along local +{plane[0]} of {reference_id}"},
            "basis": {"origin_mm": [_r(v) for v in origin], "z_axis": list(axis0), "t_axis": list(t_dir)},
            "elements": out_elements, "excluded": excluded + plane_excluded, "observation_planes": planes, "rays": rays, "bounds": bounds}


def _emitter_outline(element: SectionElement, origin: g.Vec3, t_dir: g.Vec3, axis0: g.Vec3) -> dict[str, Any]:
    """An extended emitter in the section: its area's four extreme points (along its local x and y) projected on the plane;
    an emitter facing along the section axis is a line of its true width there."""
    area = element.emitting_area
    half_x, half_y = area.size_x_mm / 2.0, area.size_y_mm / 2.0
    corners = ((-half_x, 0.0), (0.0, half_y), (half_x, 0.0), (0.0, -half_y)) if area.shape == "DISC" else (
        (-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    points = []
    for x, y in corners:
        d = g.sub(g.apply_pose(element.position_mm, element.quaternion_wxyz, (x, y, 0.0)), origin)
        points.append(_pt(g.dot(d, t_dir), g.dot(d, axis0)))
    return {"element_id": element.element_id, "label": element.label, "shape": area.shape, "size_x_mm": area.size_x_mm,
            "size_y_mm": area.size_y_mm, "outline": points}


def _light_direction(section: dict[str, Any]) -> float:
    """``+1`` when the light runs along +z of this section, ``-1`` when against it (the first chief-ray stretch along the
    axis; ``+1`` without one)."""

    for ray in section["rays"]:
        if ray["role"] != "CHIEF":
            continue
        for (t0, z0), (t1, z1) in zip(ray["points"], ray["points"][1:]):
            length = math.hypot(t1 - t0, z1 - z0)
            if length > 0.0 and abs(z1 - z0) >= ALONG_AXIS_COS * length:
                return 1.0 if z1 > z0 else -1.0
    return 1.0


def _first_reached(section: dict[str, Any]) -> str:
    """The element of a section that the light reaches first among those that face along the light (all of them when
    none does): a whole-system section then runs left to right from it."""

    sign = _light_direction(section)
    entries = section["elements"]
    along = [e for e in entries if (e["flipped"] and sign < 0.0) or (not e["flipped"] and sign > 0.0)] or entries
    first = min(along, key=lambda e: sign * e["axial_offset_mm"])
    return first.get("unit_id") or first["element_id"]


def system_sections(elements: Sequence[SectionElement], plane: str = "XZ", *, beam_paths: Sequence[Any] = (),
                    observation_planes: Sequence[Any] = (), tolerance: float = CHORD_TOLERANCE_MM) -> list[dict[str, Any]]:
    """The section of the whole system: one section per axis (a folded system has several), each referred to the element
    the light reaches first on it (``scope``: ``"ALL"``). Elements that no axis can cut stay listed in ``excluded``."""

    remaining = [e.element_id for e in elements if e.unit_id is None and (e.geometry is not None or e.unit_frame)]
    # (an extended emitter is drawn in every section; alone, with no body to cut, it is the section's reference)
    remaining = remaining or [e.element_id for e in elements if e.emitting_area is not None]
    if not remaining:
        raise SectionError("the design has no optical body to cut")
    sections = []

    def cut(reference: str) -> dict[str, Any]:
        return optical_section(elements, reference, plane, beam_paths=beam_paths, observation_planes=observation_planes, tolerance=tolerance)

    while remaining:
        section = cut(remaining[0])
        covered = {e.get("unit_id") or e["element_id"] for e in section["elements"]} | {remaining[0]}
        if section["elements"]:
            first = _first_reached(section)
            if first != remaining[0]:
                section = cut(first)
        section["scope"] = ALL_ELEMENTS
        sections.append(section)
        remaining = [r for r in remaining if r not in covered]
    return sections


def _plane_ray_heights(rays: list[dict[str, Any]], plane: dict[str, Any]) -> list[float]:
    """Where the rays of main beams cross an observation plane inside its extent (the image height of a field)."""

    z, (lo, hi) = plane["z_mm"], plane["t_mm"]
    out = []
    for ray in rays:
        if ray.get("display_role", "PRIMARY") != "PRIMARY":
            continue
        for (t0, z0), (t1, z1) in zip(ray["points"], ray["points"][1:]):
            if z1 != z0 and min(z0, z1) <= z <= max(z0, z1):
                t = t0 + (t1 - t0) * (z - z0) / (z1 - z0)
                if lo <= t <= hi:
                    out.append(t)
                break
    return out


def _bounds(elements: list[dict[str, Any]], rays: list[dict[str, Any]], planes: Sequence[dict[str, Any]] = (),
            emitters: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """``optics_*``: the elements and extended emitters; ``frame_*``: those and the observation planes near them, with the
    image heights on those planes (a frame on the optics); ``beyond_frame``: the planes far from them; ``z_mm`` / ``t_mm``:
    everything."""

    ts, zs = [], []
    for emitter in emitters:
        for t, z in emitter["outline"]:
            ts.append(t)
            zs.append(z)
    for e in elements:
        for s in e["surfaces"]:
            for seg in s["segments"]:
                for t, z in seg["points"]:
                    ts.append(t)
                    zs.append(z)
        for key in ("media",):
            for m in e[key]:
                for t, z in m["polygon"]:
                    ts.append(t)
                    zs.append(z)
        for key in ("substrate", "plate"):
            for polygon in section_polygons(e, key):
                for t, z in polygon:
                    ts.append(t)
                    zs.append(z)
    optics = (min(ts), max(ts), min(zs), max(zs)) if ts else (0.0, 0.0, 0.0, 0.0)
    for r in rays:
        for t, z in r["points"]:
            ts.append(t)
            zs.append(z)
    for p in planes:
        ts.extend(p["t_mm"])
        zs.append(p["z_mm"])
    if not ts:
        return {"t_mm": [0.0, 0.0], "z_mm": [0.0, 0.0], "optics_t_mm": [0.0, 0.0], "optics_z_mm": [0.0, 0.0],
                "frame_t_mm": [0.0, 0.0], "frame_z_mm": [0.0, 0.0], "beyond_frame": []}
    extent = max(optics[1] - optics[0], optics[3] - optics[2], MIN_SPAN_MM)
    frame_ts, frame_zs, beyond = [optics[0], optics[1]], [optics[2], optics[3]], []
    for p in planes:
        gap = max(optics[2] - p["z_mm"], p["z_mm"] - optics[3], 0.0)
        if gap <= FAR_PLANE_FACTOR * extent:
            frame_zs.append(p["z_mm"])
            frame_ts.extend(_plane_ray_heights(rays, p))
        else:
            beyond.append({"observation_id": p["observation_id"], "distance_mm": _r(gap)})
    return {"t_mm": [_r(min(ts)), _r(max(ts))], "z_mm": [_r(min(zs)), _r(max(zs))],
            "optics_t_mm": [_r(optics[0]), _r(optics[1])], "optics_z_mm": [_r(optics[2]), _r(optics[3])],
            "frame_t_mm": [_r(min(frame_ts)), _r(max(frame_ts))], "frame_z_mm": [_r(min(frame_zs)), _r(max(frame_zs))],
            "beyond_frame": beyond}


def scene_section_elements(scene: Any) -> list[SectionElement]:
    """The elements of an ``OpticalSimulationSceneV1`` to cut (a compound unit's members each on their own)."""

    return [entry for e in scene.elements
            for entry in section_elements_of(e.element_id, e.label, tuple(e.pose.position_mm), tuple(e.pose.quaternion_wxyz), e.geometry)]


# ------------------------------------------------------------------------------------------------------------ SVG

SVG_WIDTH_PX = 1400
SVG_HEIGHT_PX = 760
MIN_WIDTH_PX = 960  # the header lines need this much; a narrower box leaves white beside it
MIN_HEIGHT_PX = 360
SVG_MARGIN_PX = 48
SVG_TOP_PX = 70
SVG_BOTTOM_PX = 70
GLASS_FILL = "#bcd7ef"
SUBSTRATE_FILL = "#dbe6f0"
CLEAR_STROKE = "#1f4f8f"
RIM_STROKE = "#7d8794"
TICK_STROKE = "#0b2a52"
AXIS_STROKE = "#9aa3ad"
PLANE_STROKE = "#c2570c"
TEXT_FILL = "#1e2329"
EMITTER_RGB = (0.9, 0.36, 0.1)
SCALE_STEPS = (1.0, 2.0, 5.0)
CHANNEL_MAX = 255
DECADE = 10.0
MIN_SPAN_MM = 1e-3
CHIEF_WIDTH_PX = 1.6
NEAREST_WIDTH_PX = 1.1
PUPIL_WIDTH_PX = 0.8
NEAREST_OPACITY = 0.9  # the rays of a path nearest the plane ...
PUPIL_OPACITY = 0.35  # ... the others, and every ray of a weak branch
CLEAR_WIDTH_PX = 1.4
RIM_WIDTH_PX = 2.4
EDGE_WIDTH_PX = 1.0
PLANE_WIDTH_PX = 2.0
TICK_WIDTH_PX = 1.6
BAR_WIDTH_PX = 2.0
BAR_TICK_PX = 5.0
LABEL_LIFT_PX = 10
SCALE_BAR_FRACTION = 6.0  # the scale bar is about this fraction of the drawing's width
FRAME_PAD = 0.03  # a named frame shows this fraction of its longer side beyond what it fits
LABEL_CHAR_PX = 12.0  # the width of one label character (staggering labels that would overlap)
LABEL_ROW_PX = 17
LABEL_BAND_ROWS = 2  # room above the drawing for labels (more when they stagger higher)
SCALE_BAR_FROM_BOTTOM_PX = 30
KEY_FROM_BOTTOM_PX = 48
NOTE_FROM_BOTTOM_PX = 26
SECOND_NOTE_FROM_BOTTOM_PX = 10
HEADER_CHAR_PX = 6.2  # the mean width of one 12 px header character in the SVG
HEADER_FONT_PX = 12.0
LUMINANCE = (0.2126, 0.7152, 0.0722)
MAX_RAY_LUMINANCE = 0.42  # a ray's display colour is darkened to this on the white drawing (yellow would vanish)
# Rays are coloured by source when several sources share a wavelength (the fields of a lens): in this order.
SOURCE_COLOURS = ((0.0, 0.27, 0.8), (0.0, 0.55, 0.2), (0.8, 0.1, 0.1), (0.55, 0.2, 0.7), (0.8, 0.42, 0.0), (0.0, 0.55, 0.6))


def _hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(CHANNEL_MAX, round(c * CHANNEL_MAX))):02x}" for c in rgb)


def _nice(mm: float) -> float:
    if mm <= 0.0:
        return 1.0
    power = DECADE ** math.floor(math.log10(mm))
    best = power
    for step in SCALE_STEPS:
        if step * power <= mm:
            best = step * power
    return best


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


@dataclass(frozen=True)
class SectionLayout:
    """One drawing's frame: the image size, the common scale (pixels per mm, equal on both axes), the box it shows
    (``z0, z1, t0, t1``), the mapping ``to_px(z, t)`` (the optical axis runs left to right, ``t`` upwards) and the scale
    bar."""

    width: int
    height: int
    scale: float
    box: tuple[float, float, float, float]
    x_off: float
    y_mid: float
    frame: Any

    def to_px(self, z: float, t: float) -> tuple[float, float]:
        return self.x_off + z * self.scale, self.y_mid - (t - 0.5 * (self.box[2] + self.box[3])) * self.scale

    @property
    def bar_mm(self) -> float:
        return _nice((self.width - 2 * SVG_MARGIN_PX) / self.scale / SCALE_BAR_FRACTION)


def section_box(section: dict[str, Any], frame: Any = "optics") -> tuple[float, float, float, float]:
    """``(z0, z1, t0, t1)`` a frame shows: ``"optics"`` the elements and the observation planes near them, ``"scene"`` those
    with the rays and far planes (both about the axis, with a margin of ``FRAME_PAD``), or the box
    ``(x0, y0, z0, x1, y1, z1)`` of the design frame, projected onto the section."""

    b = section["bounds"]
    if isinstance(frame, str):
        if frame not in ("optics", "scene"):
            raise SectionError(f"frame must be optics, scene or a box, got {frame!r}")
        z0, z1 = b["frame_z_mm"] if frame == "optics" else b["z_mm"]
        t0, t1 = b["frame_t_mm"] if frame == "optics" else b["t_mm"]
        t_half = max(abs(t0), abs(t1), MIN_SPAN_MM)
        pad = FRAME_PAD * max(z1 - z0, 2.0 * t_half, MIN_SPAN_MM)
        return z0 - pad, z1 + pad, -t_half - pad, t_half + pad
    basis = section["basis"]
    corners = [g.sub(corner, tuple(basis["origin_mm"])) for corner in itertools.product(*zip(frame[:3], frame[3:]))]
    zs = [g.dot(c, tuple(basis["z_axis"])) for c in corners]
    ts = [g.dot(c, tuple(basis["t_axis"])) for c in corners]
    if max(zs) - min(zs) < MIN_SPAN_MM or max(ts) - min(ts) < MIN_SPAN_MM:
        raise SectionError(f"the frame box has no extent in this section (plane {section['plane']} of {section['reference_element_id']})")
    return min(zs), max(zs), min(ts), max(ts)


def _fit(box: tuple[float, float, float, float], width: int, height: int, top: int, frame: Any) -> SectionLayout:
    z0, z1, t0, t1 = box
    span_z, span_t = z1 - z0, t1 - t0
    scale = min((width - 2 * SVG_MARGIN_PX) / span_z, (height - top - SVG_BOTTOM_PX) / span_t)
    out_w = max(min(MIN_WIDTH_PX, width), math.ceil(span_z * scale) + 2 * SVG_MARGIN_PX)
    out_h = max(min(MIN_HEIGHT_PX, height), math.ceil(span_t * scale) + top + SVG_BOTTOM_PX)
    return SectionLayout(out_w, out_h, scale, box, (out_w - span_z * scale) / 2.0 - z0 * scale, top + (out_h - top - SVG_BOTTOM_PX) / 2.0, frame)


def section_layout(section: dict[str, Any], width: int = SVG_WIDTH_PX, height: int = SVG_HEIGHT_PX, frame: Any = "optics") -> SectionLayout:
    """The frame of a section drawing within ``width`` x ``height``: the image takes the aspect of the box (a long train
    gives a low image, one lens a narrow one), with room above for the labels."""

    box = section_box(section, frame)
    top = SVG_TOP_PX + LABEL_BAND_ROWS * LABEL_ROW_PX
    layout = _fit(box, width, height, top, frame)
    highest = min((y for _, _, y in label_places(section, layout)), default=float(top))
    short = SVG_TOP_PX + LABEL_ROW_PX - highest
    if short > 0.0:
        layout = _fit(box, width, height, top + math.ceil(short), frame)
    return layout


def label_places(section: dict[str, Any], layout: SectionLayout) -> list[tuple[str, float, float]]:
    """``(text, x, y)`` of every label in the frame, above what it names, raised by a row where it would overlap the label
    before it: each element (a compound member: its member id), each observation plane, each source of the rays."""

    z0, z1, t0, t1 = layout.box
    items = []
    for e in section["elements"]:
        tops = [t for s in e["surfaces"] for seg in s["segments"] for t, _ in seg["points"]]
        zs = [z for s in e["surfaces"] for seg in s["segments"] for _, z in seg["points"]]
        if tops and min(zs) <= z1 and max(zs) >= z0 and max(tops) >= t0:
            x, y = layout.to_px(0.5 * (max(min(zs), z0) + min(max(zs), z1)), min(max(tops), t1))
            items.append((e.get("member_id") or e["element_id"], x, y - LABEL_LIFT_PX))
    for emitter in section.get("emitters", ()):
        ts, zs = [t for t, _ in emitter["outline"]], [z for _, z in emitter["outline"]]
        if min(zs) <= z1 and max(zs) >= z0 and max(ts) >= t0:
            x, y = layout.to_px(0.5 * (max(min(zs), z0) + min(max(zs), z1)), min(max(ts), t1))
            items.append((emitter["element_id"], x, y - LABEL_LIFT_PX))
    for plane in section.get("observation_planes", ()):
        if z0 <= plane["z_mm"] <= z1 and plane["t_mm"][1] >= t0:
            x, y = layout.to_px(plane["z_mm"], min(plane["t_mm"][1], t1))
            items.append((plane["observation_id"], x, y - LABEL_LIFT_PX))
    sources = set()
    for ray in section["rays"]:
        source_id = ray.get("source_element_id")
        t, z = ray["points"][0]
        if ray["role"] == "CHIEF" and ray.get("from_source") and source_id not in sources and z0 <= z <= z1 and t0 <= t <= t1:
            sources.add(source_id)
            x, y = layout.to_px(z, t)
            items.append((source_id, x, y - LABEL_LIFT_PX))
    items.sort(key=lambda item: item[1])
    placed: list[tuple[str, float, float]] = []
    for text, x, y in items:
        half = len(text) * LABEL_CHAR_PX / 2.0
        while any(abs(x - px) < half + len(pt) * LABEL_CHAR_PX / 2.0 and abs(y - py) < LABEL_ROW_PX for pt, px, py in placed):
            y -= LABEL_ROW_PX
        placed.append((text, x, y))
    return placed


@dataclass(frozen=True)
class RayPaint:
    rgb: tuple[float, float, float]
    width_px: float
    opacity: float


def _legible(rgb: Sequence[float]) -> tuple[float, float, float]:
    """A ray's display colour on the white drawing: darkened, its hue kept, when it is too light to read."""

    luminance = sum(k * c for k, c in zip(LUMINANCE, rgb))
    factor = min(1.0, MAX_RAY_LUMINANCE / luminance) if luminance > 0.0 else 1.0
    return (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor)


def _sources_by_colour(section: dict[str, Any]) -> list[str]:
    """The sources of the rays when several of them share a wavelength (their rays are then coloured by source), else
    nothing (coloured by wavelength)."""

    lines = {(r.get("source_element_id"), r["wavelength_nm"]) for r in section["rays"]}
    wavelengths = [w for _, w in lines]
    if len(set(wavelengths)) == len(wavelengths):
        return []
    return sorted({s for s, _ in lines if s is not None})


def ray_paints(section: dict[str, Any]) -> list[RayPaint]:
    """How each ray of ``section["rays"]`` is drawn: chief rays strong, a path's pupil rays nearest the plane clear, its
    others pale, every ray of a weak branch pale."""

    sources = _sources_by_colour(section)
    nearest: dict[str, float] = {}
    for r in section["rays"]:
        if r["role"] == "PUPIL":
            nearest[r["path_id"]] = min(nearest.get(r["path_id"], math.inf), r["max_out_of_plane_mm"])
    out = []
    for r in section["rays"]:
        if sources and r.get("source_element_id") in sources:
            rgb = SOURCE_COLOURS[sources.index(r["source_element_id"]) % len(SOURCE_COLOURS)]
        else:
            rgb = _legible(r["display_rgb"])
        if r.get("display_role", "PRIMARY") != "PRIMARY":
            out.append(RayPaint(rgb, PUPIL_WIDTH_PX, PUPIL_OPACITY))
        elif r["role"] == "CHIEF":
            out.append(RayPaint(rgb, CHIEF_WIDTH_PX, 1.0))
        elif r["max_out_of_plane_mm"] <= nearest[r["path_id"]] + OFFSET_TOLERANCE_MM:
            out.append(RayPaint(rgb, NEAREST_WIDTH_PX, NEAREST_OPACITY))
        else:
            out.append(RayPaint(rgb, PUPIL_WIDTH_PX, PUPIL_OPACITY))
    return out


def ray_key(section: dict[str, Any]) -> list[tuple[str, tuple[float, float, float]]]:
    """``(source id, colour)`` of each source when rays are coloured by source (nothing otherwise)."""

    sources = _sources_by_colour(section)
    return [(s, SOURCE_COLOURS[k % len(SOURCE_COLOURS)]) for k, s in enumerate(sources)]


def section_subject(section: dict[str, Any]) -> str:
    """What the drawing's heading says it cuts."""

    reference = section["reference_element_id"]
    return f"the system on the axis of {reference}" if section.get("scope") == ALL_ELEMENTS else reference


def section_notes(section: dict[str, Any], layout: SectionLayout) -> list[str]:
    """The lines under a drawing: what is not in the section, and what lies beyond a frame on the optics."""

    notes = []
    if section["excluded"]:
        notes.append("not in this section: " + ", ".join(f'{x["element_id"]} ({x["reason"].lower().replace("_", " ")})' for x in section["excluded"]))
    if layout.frame == "optics" and section["bounds"].get("beyond_frame"):
        notes.append("beyond this frame: " + ", ".join(f'{p["observation_id"]} ({p["distance_mm"]:g} mm from the optics)' for p in section["bounds"]["beyond_frame"]))
    return notes


def section_svg(section: dict[str, Any], *, width: int = SVG_WIDTH_PX, height: int = SVG_HEIGHT_PX, frame: Any = "optics") -> str:
    """The section as a standalone SVG: exact profiles at one scale on both axes (nothing stretched), glass filled, clear
    apertures dark, opaque rims grey, clear-aperture limits ticked, observation planes, rays projected onto the plane,
    element and plane labels, a scale bar. ``frame``: ``section_box``; the image takes its aspect within the size."""

    layout = section_layout(section, width, height, frame)
    width, height, scale, to_px = layout.width, layout.height, layout.scale, layout.to_px
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<clipPath id="drawing"><rect x="0" y="{SVG_TOP_PX}" width="{width}" height="{height - SVG_TOP_PX - SVG_BOTTOM_PX}"/></clipPath>',
           '<g clip-path="url(#drawing)">']

    def path(points: Sequence[Sequence[float]], close: bool = False) -> str:
        pts = [to_px(z, t) for t, z in points]
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
        return d + (" Z" if close else "")

    ax0, ay = to_px(layout.box[0], 0.0)
    ax1, _ = to_px(layout.box[1], 0.0)
    out.append(f'<line x1="{ax0:.2f}" y1="{ay:.2f}" x2="{ax1:.2f}" y2="{ay:.2f}" stroke="{AXIS_STROKE}" stroke-width="0.8" stroke-dasharray="8 4 2 4"/>')
    for e in section["elements"]:
        for key, fill in (("substrate", SUBSTRATE_FILL), ("plate", SUBSTRATE_FILL)):
            for polygon in section_polygons(e, key):
                out.append(f'<path d="{path(polygon, True)}" fill="{fill}" fill-opacity="0.8" stroke="none"/>')
        for m in e["media"]:
            out.append(f'<path d="{path(m["polygon"], True)}" fill="{GLASS_FILL}" fill-opacity="0.75" stroke="{RIM_STROKE}" stroke-width="{EDGE_WIDTH_PX}">'
                       f'<title>{_escape(e["element_id"] + ": " + m["medium"])}</title></path>')
    for r, paint in zip(section["rays"], ray_paints(section), strict=True):
        opacity = f' stroke-opacity="{paint.opacity}"' if paint.opacity < 1.0 else ""
        lit = r["points"][:r["reference_from"] + 1] if "reference_from" in r else r["points"]
        if len(lit) >= 2:
            out.append(f'<path d="{path(lit)}" fill="none" stroke="{_hex(paint.rgb)}" stroke-width="{paint.width_px}"{opacity}/>')
        if "reference_from" in r:
            out.append(f'<path d="{path(r["points"][r["reference_from"]:])}" fill="none" stroke="{AXIS_STROKE}" stroke-width="1" '
                       'stroke-dasharray="4 3"><title>beam axis behind a central obstruction (no light)</title></path>')
    for e in section["elements"]:
        for s in e["surfaces"]:
            for seg in s["segments"]:
                stroke, w = (CLEAR_STROKE, CLEAR_WIDTH_PX) if seg["zone"] == "CLEAR_APERTURE" else (RIM_STROKE, RIM_WIDTH_PX)
                out.append(f'<path d="{path(seg["points"])}" fill="none" stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round">'
                           f'<title>{_escape(e["element_id"] + " " + s["surface_id"] + " " + seg["zone"])}</title></path>')
            for t, z in s.get("clear_aperture_ticks", []):
                x, y = to_px(z, t)
                out.append(f'<line x1="{x - 5:.2f}" y1="{y:.2f}" x2="{x + 5:.2f}" y2="{y:.2f}" stroke="{TICK_STROKE}" stroke-width="1.6"/>')
        for line in e.get("symbol", ()):
            out.append(f'<path d="{path(line)}" fill="none" stroke="{CLEAR_STROKE}" stroke-width="{CLEAR_WIDTH_PX}"/>')
    for plane in section.get("observation_planes", ()):
        x0, y0 = to_px(plane["z_mm"], plane["t_mm"][0])
        x1, y1 = to_px(plane["z_mm"], plane["t_mm"][1])
        out.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{PLANE_STROKE}" stroke-width="{PLANE_WIDTH_PX}">'
                   f'<title>{_escape(plane["observation_id"])}</title></line>')
    for emitter in section.get("emitters", ()):
        out.append(f'<path d="{path(emitter["outline"], True)}" fill="{_hex(EMITTER_RGB)}" fill-opacity="0.35" stroke="{_hex(EMITTER_RGB)}" '
                   f'stroke-width="{CLEAR_WIDTH_PX}"><title>{_escape(emitter["element_id"] + ": emitting surface")}</title></path>')
    for text, x, y in label_places(section, layout):
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="14" font-weight="600" text-anchor="middle" fill="{TEXT_FILL}">{_escape(text)}</text>')
    out.append("</g>")
    title = f'Section {section["plane"]} of {section_subject(section)}: exact profiles, true scale (1:1 on both axes)'
    out.append(f'<text x="{SVG_MARGIN_PX}" y="26" font-size="16" font-weight="600" fill="{TEXT_FILL}">{_escape(title)}</text>')
    sub = (f'z: {section["axes"]["z"]} (mm) | t: {section["axes"]["t"]} (mm) | profiles within {section["chord_tolerance_mm"]:g} mm | '
           'blue: clear aperture, grey: opaque rim, ticks: clear-aperture limits, orange: observation planes, rays: projected onto the plane, pale farther from it')
    font_px = min(HEADER_FONT_PX, HEADER_FONT_PX * (width - 2 * SVG_MARGIN_PX) / (len(sub) * HEADER_CHAR_PX))
    out.append(f'<text x="{SVG_MARGIN_PX}" y="48" font-size="{font_px:.2f}" fill="#4a5360">{_escape(sub)}</text>')
    bar = layout.bar_mm
    x1 = width - SVG_MARGIN_PX
    x0 = x1 - bar * scale
    y = height - SCALE_BAR_FROM_BOTTOM_PX
    out.append(f'<path d="M {x0:.2f} {y - BAR_TICK_PX} V {y + BAR_TICK_PX} M {x0:.2f} {y} H {x1:.2f} M {x1:.2f} {y - BAR_TICK_PX} V {y + BAR_TICK_PX}" '
               f'fill="none" stroke="{TEXT_FILL}" stroke-width="{BAR_WIDTH_PX}"/>')
    out.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y - 8}" font-size="12" text-anchor="middle" fill="{TEXT_FILL}">{bar:g} mm</text>')
    kx = float(SVG_MARGIN_PX)
    for source_id, rgb in ray_key(section):
        out.append(f'<text x="{kx:.2f}" y="{height - KEY_FROM_BOTTOM_PX}" font-size="12" font-weight="600" fill="{_hex(rgb)}">{_escape(source_id)}</text>')
        kx += (len(source_id) + 2) * LABEL_CHAR_PX
    for note, from_bottom in zip(section_notes(section, layout), (NOTE_FROM_BOTTOM_PX, SECOND_NOTE_FROM_BOTTOM_PX)):
        out.append(f'<text x="{SVG_MARGIN_PX}" y="{height - from_bottom}" font-size="12" fill="#4a5360">{_escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
