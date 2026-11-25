"""The viewer's scene model: exact part triangles in one world frame (mm, +Z up), beams, markers and camera presets.

Both views are reduced to this one model before anything is drawn, so the browser view, the PNG snapshots and
``scene_summary`` show the same geometry:

* Optical Simulation View: the run's ``optical_scene.glb`` element / plane meshes placed at their design-frame poses,
  and the Engine's beam polylines from ``optical_scene.json`` (frame ``PROJECT_DESIGN_FRAME``);
* Experimental Lab Assembly: every part GLB of the Assembly Preview Package (metres, part frame) scaled to mm and
  placed at ``pose_in_world``, the beam reference axes and the design anchors (frame ``ASSEMBLY_WORLD``).

Nothing is moved, snapped or re-traced here: coordinates are the package's / run's own numbers.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .outline import outline_groups
from .palette import BEAM_STYLES, IDENTITY_MATERIALS, class_table
from .shading import OPTICAL_VIEW, beam_needs_casing, shading_document

ASSEMBLY_VIEW = "EXPERIMENTAL_LAB_ASSEMBLY"
VIEW_DOCUMENT_SCHEMA = "opensci.local_viewer.scene"
VIEW_DOCUMENT_VERSION = "1.0"
SUMMARY_SCHEMA = "opensci.local_viewer.scene_summary"
SUMMARY_VERSION = "1.0"
LENGTH_UNIT = "mm"
UP_AXIS = "+Z"
COORD_DECIMALS = 4
# Camera presets: where the eye sits relative to the target (unit direction) and the screen-up vector.
ISO_ELEVATION_DEG = 32.0
ISO_AZIMUTH_OFFSET_DEG = 28.0  # the iso eye swings this far from square-on towards the start of the beam
# A lens layout (one straight main axis): nearly square-on and nearly level, where a lens shows its curved faces. Seen less
# than about 40 degrees from its axis any lens is a disc. Together these put the eye about 76 degrees off the axis.
LAYOUT_ELEVATION_DEG = 12.0
LAYOUT_AZIMUTH_OFFSET_DEG = 14.0
STRAIGHT_AXIS_DEG = 15.0  # a main beam stretch within this angle of the axis lies on it
STRAIGHT_AXIS_SHARE = 0.9  # of the main beam's length: more than this on the axis makes a straight layout
STEEP_AXIS_COS = 0.9  # a main axis this close to +Z has +Y for its screen up
PRESET_ORDER = ("iso", "path_horizontal", "top", "front", "side", "beam_axis", "plane_xy", "plane_zx", "plane_yz")
ORTHOGRAPHIC_PRESETS = ("path_horizontal", "plane_xy", "plane_zx", "plane_yz")
PERSPECTIVE_FOV_DEG = 14.0  # a long lens: a surface seen from the side stays side-on across the picture, also close up
FIT_MARGIN = 1.15
MIN_EXTENT_MM = 1.0
ANGLE_DECIMALS = 6  # recorded camera angles, degrees
PARALLEL_EPS = 1e-9  # a line of sight this close to the axis (sine of the angle) looks along it
DEFAULT_ROUGHNESS = 0.35


@dataclass(frozen=True)
class ViewMesh:
    positions: np.ndarray  # (n, 3) float64, world mm
    triangles: np.ndarray  # (m, 3) int64
    rgb: tuple[float, float, float]
    alpha: float
    material: str
    surface_id: str | None = None  # the optical surface this mesh draws (Optical Simulation View); None: the body
    normals: np.ndarray | None = None  # (n, 3) float64 unit vectors, world frame; None: smoothed from the triangles when needed
    metallic: float = 0.0
    roughness: float = DEFAULT_ROUGHNESS
    transmission: float = 0.0  # 1: glass, light passes (viewer.shading)
    ior: float | None = None  # the glass's refractive index, where the run states it
    thickness_mm: float | None = None  # the glass's thickness, where the run states it
    tint: float = 0.0  # what a coated or modelled face absorbs, whatever its thickness (0: the glass's path decides)

    @property
    def studio_lit(self) -> bool:
        """Glass and metal are lit by the studio (viewer.shading); a mesh of another finish by the plain head light."""

        return self.transmission > 0.0 or self.metallic > 0.0


@dataclass(frozen=True)
class ViewPart:
    part_id: str
    part_class: str
    label: str
    sku: str | None
    meshes: tuple[ViewMesh, ...]
    facts: dict[str, Any] = field(default_factory=dict)
    surfaces: tuple[dict[str, Any], ...] = ()  # per optical surface: id, role, profile, clear aperture, media (pick / hide)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points = np.concatenate([m.positions for m in self.meshes]) if self.meshes else np.zeros((1, 3))
        return points.min(axis=0), points.max(axis=0)


@dataclass(frozen=True)
class ViewBeam:
    beam_id: str
    rgb: tuple[float, float, float]
    wavelength_nm: float | None
    # (role, points) runs: role is PRIMARY / SECONDARY / PUPIL / DESIGN_AXIS; points (n, 3) world mm
    runs: tuple[tuple[str, np.ndarray], ...]
    extensions: tuple[tuple[str, np.ndarray], ...] = ()  # display-only stubs past the last Engine point
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewMarker:
    marker_id: str
    marker_class: str
    label: str
    position: np.ndarray
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewerScene:
    """One view, ready to draw. ``focus_min`` / ``focus_max`` bound the default framing (the optics and their planes,
    or the whole assembly); ``bounds_min`` / ``bounds_max`` bound everything."""

    view_kind: str
    frame: str
    title: str
    parts: tuple[ViewPart, ...]
    beams: tuple[ViewBeam, ...]
    markers: tuple[ViewMarker, ...]
    focus_min: np.ndarray
    focus_max: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    identity: dict[str, Any] = field(default_factory=dict)
    notes: tuple[dict[str, Any], ...] = ()  # {code, text, ...}: English text plus the fields a localized UI needs
    source_scene: Any = None  # the OpticalSimulationSceneV1 an optical view was read from (sections are cut from it)

    @property
    def classes(self) -> list[str]:
        present = {p.part_class for p in self.parts} | {m.marker_class for m in self.markers}
        return [key for key in class_table(self.view_kind) if key in present]

    def class_swatches(self, key: str) -> list[tuple[float, float, float]]:
        """The colours a class is drawn in: its palette colour, except a compound unit, drawn by its members -- the
        distinct identity materials of its meshes (lens glass, stop plate, mirror face, ...), in drawing order."""

        if key != "OPTICAL_UNIT":
            return [class_table(self.view_kind)[key].rgb]
        out: list[tuple[float, float, float]] = []
        for part in self.parts:
            if part.part_class != key:
                continue
            for mesh in part.meshes:
                rgb = tuple(mesh.rgb)
                if mesh.material in IDENTITY_MATERIALS and rgb not in out:
                    out.append(rgb)  # type: ignore[arg-type]
        return out or [class_table(self.view_kind)[key].rgb]


# ----------------------------------------------------------------------------------------------- geometry helpers


def quat_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(c) for c in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def place(points: np.ndarray, position, q_wxyz) -> np.ndarray:
    return points @ quat_matrix(q_wxyz).T + np.asarray(position, dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _r(values) -> list[float]:
    return [round(float(v), COORD_DECIMALS) + 0.0 for v in values]


def scene_points(scene: ViewerScene, *, focus_only: bool) -> np.ndarray:
    """Every drawn point (part vertices, beam and marker points), optionally only those inside the focus box."""

    chunks = [m.positions for p in scene.parts for m in p.meshes]
    chunks += [pts for b in scene.beams for _, pts in b.runs]
    chunks += [m.position.reshape(1, 3) for m in scene.markers]
    points = np.concatenate(chunks) if chunks else np.zeros((1, 3))
    if focus_only:
        pad = max(float(np.linalg.norm(scene.focus_max - scene.focus_min)), MIN_EXTENT_MM) * (FIT_MARGIN - 1.0)
        inside = np.all((points >= scene.focus_min - pad) & (points <= scene.focus_max + pad), axis=1)
        points = points[inside] if inside.any() else points
    return points


@dataclass(frozen=True)
class PartsFrame:
    """A framing on named parts (elements, observation planes; ``--focus``): the view fits their exact geometry."""

    part_ids: tuple[str, ...]
    label: str = ""

    def describe(self) -> str:
        return self.label or "element:" + "+".join(self.part_ids)


FOCUS_KINDS = ("element", "axis")


def focus_frame(scene: ViewerScene, text: str) -> PartsFrame:
    """``--focus``: ``element:<id>[+<id>...]`` frames those parts (elements, observation planes); ``axis:<id>`` frames the
    element and every element and observation plane on its axis (the parts of its section, ``view --section``)."""

    kind, _, ids = text.partition(":")
    names = tuple(v.strip() for v in ids.split("+") if v.strip())
    if kind not in FOCUS_KINDS or not names or (kind == "axis" and len(names) != 1):
        raise ValueError(f"focus must be element:<id>[+<id>...] or axis:<id>, got {text!r}")
    if kind == "element":
        frame = PartsFrame(names, text)
        frame_points(scene, frame)  # every id names a part
        return frame
    source = scene.source_scene
    if source is None:
        raise ValueError("axis:<id> needs a run directory's Optical Simulation View")
    from ..optical_view.section import SectionError, optical_section, scene_section_elements

    try:
        section = optical_section(scene_section_elements(source), names[0], "XZ", beam_paths=source.beam_paths,
                                  observation_planes=source.observation_planes)
    except SectionError as exc:
        raise ValueError(str(exc)) from exc
    parts = {p.part_id for p in scene.parts}
    ids_on_axis = [e["element_id"] for e in section["elements"]] + [p["observation_id"] for p in section["observation_planes"]]
    return PartsFrame(tuple(dict.fromkeys(i for i in ids_on_axis if i in parts)), text)


def frame_points(scene: ViewerScene, frame=None) -> np.ndarray:
    """The points a framing of a view fits: ``None`` the default framing (the setup, far planes left out); ``"scene"``
    every part, beam and far plane; ``"optics"`` the optical elements and the observation planes near them (no source);
    a ``PartsFrame`` the vertices of those parts; or the eight corners of the box ``(x0, y0, z0, x1, y1, z1)`` of the
    view's frame."""

    if frame is None:
        return scene_points(scene, focus_only=True)
    if isinstance(frame, PartsFrame):
        by_id = {p.part_id: p for p in scene.parts}
        unknown = [i for i in frame.part_ids if i not in by_id]
        if unknown:
            raise ValueError(f"no part {', '.join(unknown)} in this view; its parts are {', '.join(sorted(by_id))}")
        chunks = [m.positions for i in frame.part_ids for m in by_id[i].meshes]
        if not chunks:
            raise ValueError(f"{frame.describe()} has no geometry to frame")
        return np.concatenate(chunks)
    if frame == "scene":
        return scene_points(scene, focus_only=False)
    if frame == "optics":
        far = {n["id"] for n in scene.notes if n.get("code") == "FAR_PLANE"}
        chunks = [m.positions for p in scene.parts if p.part_class != "SOURCE" and p.part_id not in far for m in p.meshes]
        return np.concatenate(chunks) if chunks else scene_points(scene, focus_only=True)
    if isinstance(frame, str):
        raise ValueError(f"frame must be optics, scene or a box, got {frame!r}")
    x0, y0, z0, x1, y1, z1 = (float(v) for v in frame)
    return np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def main_beam_direction(scene: ViewerScene) -> np.ndarray | None:
    """Length-weighted mean direction of the main beam stretches in the XY plane (sign kept from the first stretch)."""

    total = np.zeros(3)
    reference = None
    for beam in scene.beams:
        for role, pts in beam.runs:
            if role not in ("PRIMARY", "DESIGN_AXIS") or len(pts) < 2:
                continue
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                d = np.array([b[0] - a[0], b[1] - a[1], 0.0])
                if np.linalg.norm(d) == 0.0:
                    continue
                if reference is None:
                    reference = _unit(d)
                # fold opposite directions together (a folded path still has one dominant axis)
                total += d if float(np.dot(d, reference)) >= 0.0 else -d
    return None if reference is None or np.linalg.norm(total) == 0.0 else _unit(total)


def main_beam_axis(scene: ViewerScene) -> tuple[np.ndarray, bool] | None:
    """The main beam's axis in 3D (length-weighted mean direction, the sign of its first stretch, opposite stretches folded
    together) and whether nearly all of the beam runs along it: a straight layout such as a lens."""

    stretches = []
    for beam in scene.beams:
        for role, pts in beam.runs:
            if role in ("PRIMARY", "DESIGN_AXIS") and len(pts) >= 2:
                stretches.append(np.asarray(pts[1:], dtype=float) - np.asarray(pts[:-1], dtype=float))
    d = np.concatenate(stretches) if stretches else np.zeros((0, 3))
    lengths = np.linalg.norm(d, axis=1)
    if not lengths.any():
        return None
    reference = d[np.argmax(lengths > 0.0)]
    axis = _unit((d * np.where(d @ reference >= 0.0, 1.0, -1.0)[:, None]).sum(axis=0))
    along = np.abs(d @ axis) >= lengths * math.cos(math.radians(STRAIGHT_AXIS_DEG))
    return axis, float(lengths[along].sum()) > STRAIGHT_AXIS_SHARE * float(lengths.sum())


def layout_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(side, up)`` of a layout along ``axis``: seen from ``side`` with screen up ``up``, the axis runs exactly left to right."""

    z = np.array([0.0, 0.0, 1.0])
    wanted = np.array([0.0, 1.0, 0.0]) if abs(float(axis @ z)) > STEEP_AXIS_COS else z
    up = _unit(wanted - axis * float(axis @ wanted))
    return np.cross(axis, up), up


def axis_frame(scene: ViewerScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(axis, side, lift)``: the frame a camera pose is stated in. ``axis`` is the main beam's axis in the direction the
    light travels (``main_beam_axis``; +X in a view without a beam), ``side`` and ``lift`` its ``layout_frame``: seen from
    ``side`` with ``lift`` up the screen, the axis runs left to right (``lift`` is +Z unless the axis is nearly vertical)."""

    main = main_beam_axis(scene)
    axis = main[0] if main is not None else np.array([1.0, 0.0, 0.0])
    side, lift = layout_frame(axis)
    return axis, side, lift


def camera_from_angles(scene: ViewerScene, angle_to_axis_deg: float, elevation_deg: float = 0.0,
                       roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(view_direction, up)`` of a camera stated by angles in ``axis_frame``.

    ``angle_to_axis_deg``: the angle between the line of sight and the axis (0: looking along the beam from behind its start,
    90: square-on, 180: looking back at the light); ``elevation_deg``: how far the eye is turned about the axis from the
    ``side`` towards ``lift`` (90: from above for a horizontal axis); ``roll_deg``: the camera turned about its line of
    sight, positive turning the picture clockwise. At roll 0 the axis runs level across the screen (exactly left to right
    for an angle of 90 degrees)."""

    axis, side, lift = axis_frame(scene)
    a, e, r = (math.radians(float(v)) for v in (angle_to_axis_deg, elevation_deg, roll_deg))
    across = side * math.cos(e) + lift * math.sin(e)
    direction = _unit(-axis * math.cos(a) + across * math.sin(a))
    level_up = lift * math.cos(e) - side * math.sin(e)  # at right angles to the axis and to the line of sight
    up = _unit(level_up * math.cos(r) + np.cross(direction, level_up) * math.sin(r))
    return direction, up


def camera_angles(scene: ViewerScene, view_direction, up) -> dict[str, Any]:
    """The angles of ``camera_from_angles`` measured back from a camera's vectors (``view_direction`` from the target to
    the eye, ``up`` up the screen), and the axis they are measured against. Looking exactly along the axis, the elevation
    is 0 and the roll carries the whole turn of the picture."""

    axis, side, lift = axis_frame(scene)
    v = _unit(np.asarray(view_direction, dtype=float))
    u = np.asarray(up, dtype=float)
    u = _unit(u - v * float(u @ v))
    along = float(-v @ axis)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, along))))
    across = v + axis * along
    e = math.atan2(float(v @ lift), float(v @ side)) if float(np.linalg.norm(across)) > PARALLEL_EPS else 0.0
    level_up = lift * math.cos(e) - side * math.sin(e)
    roll = math.atan2(float(np.cross(v, level_up) @ u), float(level_up @ u))
    return {"angle_to_axis_deg": _angle(angle), "elevation_deg": _angle(math.degrees(e)), "roll_deg": _angle(math.degrees(roll)),
            "axis": _r(axis)}


def _angle(degrees: float) -> float:
    return round(degrees, ANGLE_DECIMALS) + 0.0


def first_beam_segment(scene: ViewerScene) -> tuple[np.ndarray, np.ndarray] | None:
    """Start point and unit direction of the first main beam stretch."""

    for beam in scene.beams:
        for role, pts in beam.runs:
            if role in ("PRIMARY", "DESIGN_AXIS") and len(pts) >= 2:
                start = np.asarray(pts[0], dtype=float)
                d = _unit(np.asarray(pts[1], dtype=float) - start)
                if np.linalg.norm(d) > 0.0:
                    return start, d
    return None


def camera_presets(scene: ViewerScene) -> dict[str, dict[str, Any]]:
    """Named views: ``view_direction`` points from the target to the eye, ``up`` is the screen-up vector.

    iso: 3/4 view from above, square-on to the main beam so it runs left to right; of a straight optical layout, a near-side view
    at a small elevation, at least 70 degrees off the axis; path_horizontal: square-on to the main beam, which runs exactly left to
    right whatever its direction; top: looking down -Z (X right, Y up the screen); front: looking along +Y (X right, Z up);
    side: looking along -X (Y right, Z up); beam_axis: looking down the first main beam from behind its start; plane_xy /
    plane_zx / plane_yz: square-on to a coordinate plane, its first letter to the right and its second up the screen.
    ``projection``: what the page draws the view with (snapshots are all orthographic).
    """

    z = np.array([0.0, 0.0, 1.0])
    beam = main_beam_direction(scene)
    if beam is None:
        beam = np.array([1.0, 0.0, 0.0])
    perp = np.array([beam[1], -beam[0], 0.0])  # beam x perp = +Z: seen from +perp, the beam runs to the right
    if float(np.dot(np.cross(-perp, z), beam)) < 0.0:
        perp = -perp
    az = math.radians(ISO_AZIMUTH_OFFSET_DEG)
    el = math.radians(ISO_ELEVATION_DEG)
    horizontal = _unit(perp * math.cos(az) - beam * math.sin(az))
    iso = _unit(horizontal * math.cos(el) + z * math.sin(el))
    iso_up = z
    main = main_beam_axis(scene)
    axis = main[0] if main is not None else np.array([1.0, 0.0, 0.0])
    side, side_up = layout_frame(axis)
    if main is not None and main[1] and scene.view_kind == OPTICAL_VIEW:
        az, el = math.radians(LAYOUT_AZIMUTH_OFFSET_DEG), math.radians(LAYOUT_ELEVATION_DEG)
        iso = _unit((side * math.cos(az) - axis * math.sin(az)) * math.cos(el) + side_up * math.sin(el))
        iso_up = _unit(np.cross(iso, axis))  # rolled so that the axis stays level on the screen
    segment = first_beam_segment(scene)
    along = -segment[1] if segment is not None else -beam
    along_up = z if abs(float(np.dot(_unit(along), z))) < math.cos(math.radians(ISO_AZIMUTH_OFFSET_DEG)) else np.array([0.0, 1.0, 0.0])
    presets = {
        "iso": (iso, iso_up),
        "path_horizontal": (side, side_up),
        "top": (z, np.array([0.0, 1.0, 0.0])),
        "front": (np.array([0.0, -1.0, 0.0]), z),
        "side": (np.array([1.0, 0.0, 0.0]), z),
        "beam_axis": (_unit(along), along_up),
        "plane_xy": (z, np.array([0.0, 1.0, 0.0])),
        "plane_zx": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        "plane_yz": (np.array([1.0, 0.0, 0.0]), z),
    }
    target, distance = fit_distance(scene_points(scene, focus_only=True))
    out = {}
    for name in PRESET_ORDER:
        direction, up = presets[name]
        aim = target
        if name == "beam_axis" and segment is not None:  # aim at the beam line itself, so the beam is seen end-on
            aim = segment[0] + segment[1] * float(np.dot(target - segment[0], segment[1]))
        out[name] = {"view_direction": _r(direction), "up": _r(up), "target_mm": _r(aim), "eye_mm": _r(aim + direction * distance),
                     "projection": "orthographic" if name in ORTHOGRAPHIC_PRESETS else "perspective"}
    return out


def fit_distance(points: np.ndarray) -> tuple[np.ndarray, float]:
    """``(target, distance)``: the centre of the points' box and how far from it the page's perspective camera
    (``PERSPECTIVE_FOV_DEG``) sees them all, with ``FIT_MARGIN``."""

    target = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.linalg.norm(points - target, axis=1).max()), MIN_EXTENT_MM)
    return target, radius * FIT_MARGIN / math.sin(math.radians(PERSPECTIVE_FOV_DEG) / 2.0)


# ------------------------------------------------------------------------------------------------ JSON documents


def _b64(array: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _bounds_doc(lo: np.ndarray, hi: np.ndarray) -> dict[str, list[float]]:
    return {"min_mm": _r(lo), "max_mm": _r(hi), "size_mm": _r(hi - lo)}


def _outlines_doc(part: ViewPart) -> list[dict[str, Any]]:
    """The see-through surfaces' outlines: end points and side normals of each edge (float32) and whether it is always drawn."""

    return [{**({"surface": o.surface_id} if o.surface_id else {}), "points": _b64(o.points, "<f4"), "sides": _b64(o.sides, "<f4"),
             "feature": _b64(o.feature, "u1"), "faint": o.faint} for o in outline_groups(part, merge_surfaces=True) if o.see_through and o.studio_lit]


def view_document(scene: ViewerScene) -> dict[str, Any]:
    """The JSON the browser view draws: little-endian float32 positions and normals (world mm) and uint32 triangle indices."""

    classes = class_table(scene.view_kind)
    main = main_beam_axis(scene)
    return {
        "schema": VIEW_DOCUMENT_SCHEMA,
        "schema_version": VIEW_DOCUMENT_VERSION,
        "view_kind": scene.view_kind,
        "title": scene.title,
        "frame": scene.frame,
        "length_unit": LENGTH_UNIT,
        "up_axis": UP_AXIS,
        "identity": scene.identity,
        "classes": [{"key": k, "rgb": list(classes[k].rgb), "alpha": classes[k].alpha, "en": classes[k].en, "zh": classes[k].zh,
                     **({"swatches": [list(c) for c in scene.class_swatches(k)]} if k == "OPTICAL_UNIT" else {})}
                    for k in scene.classes],
        "beam_styles": BEAM_STYLES,
        "parts": [{
            "id": p.part_id, "class": p.part_class, "label": p.label, "sku": p.sku, "facts": p.facts,
            "bounds": _bounds_doc(*p.bounds),
            "meshes": [{"positions": _b64(m.positions, "<f4"), "indices": _b64(m.triangles, "<u4"), "rgb": list(m.rgb),
                        "alpha": m.alpha, "material": m.material, **({"surface": m.surface_id} if m.surface_id else {}),
                        **({"normals": _b64(m.normals, "<f4")} if m.normals is not None else {}),
                        "finish": {"studio_lit": m.studio_lit, "metallic": m.metallic, "roughness": m.roughness, "transmission": m.transmission,
                                   "ior": m.ior, "thickness_mm": m.thickness_mm, "tint": m.tint}} for m in p.meshes],
            **({"surfaces": [dict(f) for f in p.surfaces]} if p.surfaces else {}),
            **({"outlines": _outlines_doc(p)} if scene.view_kind == OPTICAL_VIEW else {}),
        } for p in scene.parts],
        "beams": [{
            "id": b.beam_id, "rgb": list(b.rgb), "wavelength_nm": b.wavelength_nm, "facts": b.facts,
            **({"casing": True} if beam_needs_casing(b.rgb, scene.view_kind) else {}),
            "runs": [{"role": role, "points": [_r(pt) for pt in pts]} for role, pts in b.runs],
            "extensions": [{"role": role, "points": [_r(pt) for pt in pts]} for role, pts in b.extensions],
        } for b in scene.beams],
        "markers": [{"id": m.marker_id, "class": m.marker_class, "label": m.label, "position_mm": _r(m.position), "facts": m.facts}
                    for m in scene.markers],
        "bounds": _bounds_doc(scene.bounds_min, scene.bounds_max),
        "focus_bounds": _bounds_doc(scene.focus_min, scene.focus_max),
        "camera_presets": camera_presets(scene),
        "notes": [dict(n) for n in scene.notes],
        "shading": shading_document(scene.view_kind),
        "camera_fov_deg": PERSPECTIVE_FOV_DEG,
        "main_axis": None if main is None else _r(main[0]),
        "camera_frame": dict(zip(("axis", "side", "lift"), (_r(v) for v in axis_frame(scene)), strict=True)),
    }


def scene_summary(scene: ViewerScene) -> dict[str, Any]:
    """What is in the view, for an Agent without a browser: parts by class (ids, names, bounding boxes in mm), beam
    paths (role runs as polylines, mm), markers, camera presets, units and axes."""

    classes = class_table(scene.view_kind)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for p in scene.parts:
        lo, hi = p.bounds
        entry = {"id": p.part_id, "label": p.label, "sku": p.sku, **p.facts, "bounds": _bounds_doc(lo, hi), "center_mm": _r((lo + hi) / 2.0)}
        if p.surfaces:
            entry["surfaces"] = [dict(f) for f in p.surfaces]
        by_class.setdefault(p.part_class, []).append(entry)
    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_VERSION,
        "view_kind": scene.view_kind,
        "title": scene.title,
        "frame": scene.frame,
        "units": {"length": LENGTH_UNIT},
        "axes": {"up": UP_AXIS, "handedness": "right-handed", "x": "+X", "y": "+Y", "z": "+Z (up)"},
        "identity": scene.identity,
        "classes": [{"key": k, "label": classes[k].en, "rgb": list(classes[k].rgb), "count": len(by_class.get(k, ()))
                     + sum(1 for m in scene.markers if m.marker_class == k)} for k in scene.classes],
        "parts_by_class": {k: by_class[k] for k in classes if k in by_class},
        "beams": [{"id": b.beam_id, "wavelength_nm": b.wavelength_nm, "display_rgb": list(b.rgb), **b.facts,
                   "runs": [{"role": role, "points_mm": [_r(pt) for pt in pts]} for role, pts in b.runs]} for b in scene.beams],
        "markers": [{"id": m.marker_id, "class": m.marker_class, "label": m.label, "position_mm": _r(m.position), **m.facts}
                    for m in scene.markers],
        "bounds": _bounds_doc(scene.bounds_min, scene.bounds_max),
        "default_framing_bounds": _bounds_doc(scene.focus_min, scene.focus_max),
        "camera_presets": camera_presets(scene),
        "notes": [n["text"] for n in scene.notes],
    }
