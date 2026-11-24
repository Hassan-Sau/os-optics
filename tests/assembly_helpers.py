"""Test helpers for the local assembly layout checker (``os_optics.assembly``).

Three small toolkits, all pure standard library plus the vendored package contract:

* **Pose math** (``Pose`` = ``(position_mm, quaternion_wxyz)``): compose / invert / rotate, written independently of
  ``os_optics.assembly.transforms`` so tests can predict world-space results without trusting the code under test.
* **Meshes and GLBs**: ``TriMesh`` (part-frame millimetres) with ``box``, ``cylinder`` (solid or tube, along +Z),
  ``combine`` (several closed shells in one part), ``inverted`` and ``without_last_triangles`` (bad geometry on
  purpose), and ``glb_bytes`` -- a GLB that passes the package contract's GLB rules (one node + mesh named after the
  part, the generator block, allow-listed strings only, positions in metres as float32).
* **Packages**: ``reference_package()`` (the reference fixture), ``rebuild`` / ``mutate`` (edit a scene payload
  dict, optionally swap part geometry, and re-sign a valid package with the reference binding), and
  ``synthetic_package`` (a minimal valid scene: a breadboard plus a few box/cylinder parts, support tree, mates whose
  ``b_in_a`` is derived from the poses unless given).

Every built package goes through ``build_assembly_preview_package``, i.e. the canonical verifier, so a helper can
never hand the checker something the contract would refuse.
"""

from __future__ import annotations

import copy
import json
import math
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from os_optics._vendor.shared_contracts.assembly_preview import (
    GLB_GENERATOR,
    AssemblyPreviewPackageV1,
    AssemblyPreviewSceneV1,
    build_assembly_preview_package,
    load_assembly_preview_package,
    read_package_archive,
)

FIXTURE_ZIP = Path(__file__).resolve().parent / "fixtures" / "assembly_preview" / "reference.assembly-preview.zip"
REFERENCE_PARTS = ("b01-optic", "b02-optic-mount", "b03-post", "b04-post-holder-base", "b05-breadboard")
STACK = REFERENCE_PARTS[:4]  # every OpenSci part the breadboard carries (optic -> mount -> post -> post holder/base)
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Pose = tuple[Vec3, Quat]


# ------------------------------------------------------------------------------------------------ pose math


def normalized(q: Iterable[float]) -> Quat:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def axis_angle(axis: Vec3, degrees: float) -> Quat:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    half = math.radians(degrees) / 2.0
    s = math.sin(half) / n
    return (math.cos(half), ax * s, ay * s, az * s)


def q_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def q_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def rotate(q: Quat, v: Vec3) -> Vec3:
    """Active rotation of ``v`` by the unit quaternion ``q`` (q v q*)."""

    w, x, y, z = q_mul(q_mul(q, (0.0, v[0], v[1], v[2])), q_conj(q))
    return (x, y, z)


def compose(outer: Pose, inner: Pose) -> Pose:
    """``outer o inner`` (apply ``inner`` first)."""

    (po, qo), (pi, qi) = outer, inner
    r = rotate(qo, pi)
    return ((po[0] + r[0], po[1] + r[1], po[2] + r[2]), q_mul(qo, qi))


def invert(pose: Pose) -> Pose:
    p, q = pose
    qc = q_conj(q)
    r = rotate(qc, p)
    return ((-r[0], -r[1], -r[2]), qc)


def apply(pose: Pose, v: Vec3) -> Vec3:
    p, q = pose
    r = rotate(q, v)
    return (p[0] + r[0], p[1] + r[1], p[2] + r[2])


def rotation_about(point: Vec3, q: Quat) -> Pose:
    """The rigid motion rotating by ``q`` about ``point`` (``point`` stays fixed)."""

    r = rotate(q, point)
    return ((point[0] - r[0], point[1] - r[1], point[2] - r[2]), q)


def pose_dict(position: Vec3 = (0.0, 0.0, 0.0), quaternion: Quat = IDENTITY_Q) -> dict:
    return {"position_mm": [float(c) for c in position], "quaternion_wxyz": [float(c) for c in quaternion]}


def pose_of(d: Mapping) -> Pose:
    return tuple(d["position_mm"]), tuple(d["quaternion_wxyz"])  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------ meshes / GLBs


@dataclass(frozen=True)
class TriMesh:
    """Triangle mesh in the part frame (mm); ``triangles`` index ``vertices`` counter-clockwise seen from outside."""

    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    @property
    def bounds(self) -> tuple[Vec3, Vec3]:
        """Bounds of the used, finite coordinates."""

        used = {i for t in self.triangles for i in t}
        pts = [self.vertices[i] for i in sorted(used)]
        return (tuple(_finite_extreme((p[k] for p in pts), min) for k in range(3)),  # type: ignore[return-value]
                tuple(_finite_extreme((p[k] for p in pts), max) for k in range(3)))

    def with_vertex(self, old: Vec3, new: Vec3) -> TriMesh:
        """The same connectivity with one vertex moved (e.g. to a non-finite coordinate)."""

        return TriMesh(tuple(new if v == old else v for v in self.vertices), self.triangles)


class _Builder:
    def __init__(self) -> None:
        self.index: dict[Vec3, int] = {}
        self.vertices: list[Vec3] = []
        self.triangles: list[tuple[int, int, int]] = []

    def vid(self, p: Vec3) -> int:
        p = (float(p[0]), float(p[1]), float(p[2]))
        if p not in self.index:
            self.index[p] = len(self.vertices)
            self.vertices.append(p)
        return self.index[p]

    def tri(self, a: Vec3, b: Vec3, c: Vec3) -> None:
        self.triangles.append((self.vid(a), self.vid(b), self.vid(c)))

    def quad(self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> None:
        self.tri(a, b, c)
        self.tri(a, c, d)

    def mesh(self) -> TriMesh:
        return TriMesh(tuple(self.vertices), tuple(self.triangles))


def box(lo: Vec3, hi: Vec3) -> TriMesh:
    """Closed, outward-oriented box (12 triangles)."""

    (x0, y0, z0), (x1, y1, z1) = lo, hi
    b = _Builder()
    b.quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))  # -z
    b.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))  # +z
    b.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))  # -y
    b.quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))  # +y
    b.quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))  # -x
    b.quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))  # +x
    return b.mesh()


def cylinder(radius: float, z0: float, z1: float, *, center: tuple[float, float] = (0.0, 0.0), segments: int = 32,
             inner_radius: float = 0.0) -> TriMesh:
    """Closed, outward-oriented prism along +Z: a solid ``segments``-gon cylinder, or a tube if ``inner_radius > 0``."""

    cx, cy = center
    ring = [(math.cos(2.0 * math.pi * k / segments), math.sin(2.0 * math.pi * k / segments)) for k in range(segments)]
    b = _Builder()
    for k in range(segments):
        (c0, s0), (c1, s1) = ring[k], ring[(k + 1) % segments]
        o = [(cx + radius * c0, cy + radius * s0), (cx + radius * c1, cy + radius * s1)]
        b.quad((*o[0], z0), (*o[1], z0), (*o[1], z1), (*o[0], z1))
        if inner_radius > 0.0:
            i = [(cx + inner_radius * c0, cy + inner_radius * s0), (cx + inner_radius * c1, cy + inner_radius * s1)]
            b.quad((*i[0], z0), (*i[0], z1), (*i[1], z1), (*i[1], z0))
            b.quad((*o[0], z1), (*o[1], z1), (*i[1], z1), (*i[0], z1))  # top ring, +z
            b.quad((*i[0], z0), (*i[1], z0), (*o[1], z0), (*o[0], z0))  # bottom ring, -z
        else:
            b.tri((cx, cy, z1), (*o[0], z1), (*o[1], z1))
            b.tri((cx, cy, z0), (*o[1], z0), (*o[0], z0))
    return b.mesh()


def combine(*meshes: TriMesh) -> TriMesh:
    """Several closed shells as one part mesh (overlapping shells unite under the winding rule)."""

    vertices: list[Vec3] = []
    triangles: list[tuple[int, int, int]] = []
    for m in meshes:
        base = len(vertices)
        vertices.extend(m.vertices)
        triangles.extend((a + base, b + base, c + base) for a, b, c in m.triangles)
    return TriMesh(tuple(vertices), tuple(triangles))


def transformed(mesh: TriMesh, pose: Pose) -> TriMesh:
    return TriMesh(tuple(apply(pose, v) for v in mesh.vertices), mesh.triangles)


def inverted(mesh: TriMesh) -> TriMesh:
    """Same surface, every triangle reversed (inward-facing: closed but negative volume)."""

    return TriMesh(mesh.vertices, tuple((a, c, b) for a, b, c in mesh.triangles))


def without_last_triangles(mesh: TriMesh, count: int = 2) -> TriMesh:
    """An OPEN surface (drops the last ``count`` triangles, e.g. one box face)."""

    return TriMesh(mesh.vertices, mesh.triangles[:-count])


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _finite_extreme(values: Iterable[float], pick: Callable) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return pick(finite) if finite else 0.0


def glb_bytes(part_ref: str, mesh: TriMesh, finish: str = "BODY") -> bytes:
    """A contract-valid part GLB: positions in metres (float32), uint32 indices, one material named ``finish``.

    The accessor ``min``/``max`` (what the package verifier compares with the scene bounds) cover the FINITE
    coordinates only, so a mesh with a non-finite vertex still produces a GLB whose JSON verifies."""

    used = sorted({i for t in mesh.triangles for i in t})
    remap = {old: new for new, old in enumerate(used)}
    metres = [tuple(_f32(c / 1000.0) for c in mesh.vertices[i]) for i in used]
    positions = b"".join(struct.pack("<3f", *p) for p in metres)
    indices = b"".join(struct.pack("<I", remap[i]) for t in mesh.triangles for i in t)
    document = {
        "asset": {"version": "2.0", "generator": GLB_GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": part_ref}],
        "meshes": [{"name": part_ref, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0, "mode": 4}]}],
        "materials": [{"name": finish.lower(), "alphaMode": "OPAQUE"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(metres), "type": "VEC3",
             "min": [_finite_extreme((p[k] for p in metres), min) for k in range(3)],
             "max": [_finite_extreme((p[k] for p in metres), max) for k in range(3)]},
            {"bufferView": 1, "componentType": 5125, "count": 3 * len(mesh.triangles), "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
                        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)}],
        "buffers": [{"byteLength": len(positions) + len(indices)}],
    }
    binary = positions + indices
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


# ------------------------------------------------------------------------------------------------ packages


def reference_files() -> dict[str, bytes]:
    return read_package_archive(FIXTURE_ZIP.read_bytes())


def reference_package() -> AssemblyPreviewPackageV1:
    return load_assembly_preview_package(reference_files())


def scene_payload(package: AssemblyPreviewPackageV1) -> dict:
    """A deep, freely editable copy of the package's scene JSON payload."""

    return copy.deepcopy(package.scene.payload())


def geometry_files(package: AssemblyPreviewPackageV1) -> dict[str, bytes]:
    return {path: bytes(data) for path, data in package.files.items() if path.startswith("geometry/")}


def rebuild(scene: Mapping, geometry: Mapping[str, bytes], *, like: AssemblyPreviewPackageV1 | None = None) -> AssemblyPreviewPackageV1:
    """Re-sign a package from a scene payload and geometry files, with the binding of ``like`` (reference)."""

    base = like or reference_package()
    return build_assembly_preview_package(scene=AssemblyPreviewSceneV1.model_validate(scene), geometry=geometry,
                                          binding=base.manifest.binding)


def mutate(edit: Callable[[dict, dict[str, bytes]], None], base: AssemblyPreviewPackageV1 | None = None) -> AssemblyPreviewPackageV1:
    """Apply ``edit(scene_payload, geometry_files)`` in place to copies of ``base`` (reference) and re-sign."""

    base = base or reference_package()
    scene, geometry = scene_payload(base), geometry_files(base)
    edit(scene, geometry)
    return rebuild(scene, geometry, like=base)


def part(scene: Mapping, ref: str) -> dict:
    return next(p for p in scene["parts"] if p["part_ref"] == ref)


def mate(scene: Mapping, ref: str) -> dict:
    return next(m for m in scene["mates"] if m["mate_ref"] == ref)


def move_parts(scene: Mapping, refs: Iterable[str], pose: Pose) -> None:
    """Apply the world rigid motion ``pose`` to the given parts' world poses."""

    for ref in refs:
        p = part(scene, ref)
        p["pose_in_world"] = pose_dict(*compose(pose, pose_of(p["pose_in_world"])))


def translate_parts(scene: Mapping, refs: Iterable[str], delta: Vec3) -> None:
    move_parts(scene, refs, (delta, IDENTITY_Q))


def move_world(scene: Mapping, pose: Pose) -> None:
    """Rigidly move EVERYTHING placed in the world by ``pose``: every part (breadboard included), the beam axes, the
    design anchors, the design frame and the DOF directions. Mates are relative and stay valid. The breadboard
    outline and hole grid are world-XY data and do not move (callers remap fixations if they need them)."""

    move_parts(scene, [p["part_ref"] for p in scene["parts"]], pose)
    beam = scene["beam_reference"]
    beam["design_frame_in_world"] = pose_dict(*compose(pose, pose_of(beam["design_frame_in_world"])))
    for axis in beam.get("beam_axes", []):
        axis["points_mm"] = [list(apply(pose, tuple(p))) for p in axis["points_mm"]]
    for anchor in beam.get("design_anchors", []):
        anchor["position_mm"] = list(apply(pose, tuple(anchor["position_mm"])))
    for dof in scene.get("dof_implementations", []):
        direction = rotate(pose[1], tuple(dof["world_direction"]))
        n = math.sqrt(sum(c * c for c in direction))
        dof["world_direction"] = [c / n for c in direction]


def datum_world(scene: Mapping, part_ref: str, datum_key: str) -> Pose:
    p = part(scene, part_ref)
    datum = next(d for d in p["datums"] if d["datum_key"] == datum_key)
    return compose(pose_of(p["pose_in_world"]), pose_of(datum["pose_in_part"]))


def consistent_b_in_a(scene: Mapping, mate_d: Mapping) -> dict:
    """``b_in_a`` that makes the mate close exactly for the scene's current poses."""

    a = datum_world(scene, mate_d["part_a_ref"], mate_d["datum_a"])
    b = datum_world(scene, mate_d["part_b_ref"], mate_d["datum_b"])
    return pose_dict(*compose(invert(a), b))


def resync_mate(scene: Mapping, mate_ref: str) -> None:
    m = mate(scene, mate_ref)
    m["b_in_a"] = consistent_b_in_a(scene, m)


def set_geometry(scene: Mapping, geometry: dict[str, bytes], part_ref: str, mesh: TriMesh, *, finish: str | None = None,
                 declared_triangles: int | None = None) -> None:
    """Replace a part's GLB and keep the scene's ``PartGeometryV1`` (bounds, triangle count) consistent with it."""

    p = part(scene, part_ref)
    finish = finish or p["appearance"]["finish"]
    lo, hi = mesh.bounds
    p["geometry"]["bounds_min_mm"] = list(lo)
    p["geometry"]["bounds_max_mm"] = list(hi)
    p["geometry"]["triangle_count"] = declared_triangles or len(mesh.triangles)
    geometry[p["geometry"]["asset_path"]] = glb_bytes(part_ref, mesh, finish)


# ------------------------------------------------------------------------------------------ synthetic scenes


@dataclass
class SynthPart:
    """One part of a synthetic scene. ``parent``/``mate_ref``/``relation`` are ``None`` only for the breadboard."""

    ref: str
    mesh: TriMesh
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quat = IDENTITY_Q
    category: str = "OTHER_SUPPORT"
    parent: str | None = None
    mate_ref: str | None = None
    relation: str | None = None
    datums: dict[str, Pose] = field(default_factory=lambda: {"origin": ((0.0, 0.0, 0.0), IDENTITY_Q)})
    finish: str = "BODY"


@dataclass
class SynthMate:
    """A mate; ``b_in_a=None`` derives the exact closing pose from the part poses and datums."""

    ref: str
    part_a: str
    datum_a: str
    part_b: str
    datum_b: str
    designation: str | None = None
    b_in_a: Pose | None = None


BOARD_REF = "p0-board"


def board(outline_max: tuple[float, float] = (500.0, 500.0), thickness: float = 12.7) -> SynthPart:
    """A breadboard part: top face at world z = 0 over ``0..outline_max``; datum ``top`` at the world origin."""

    return SynthPart(BOARD_REF, box((0.0, 0.0, -thickness), (outline_max[0], outline_max[1], 0.0)), category="BREADBOARD",
                     datums={"top": ((0.0, 0.0, 0.0), IDENTITY_Q)})


def standing(ref: str, mesh: TriMesh, position: Vec3, quaternion: Quat = IDENTITY_Q, *, mate_ref: str,
             category: str = "BASE") -> tuple[SynthPart, SynthMate]:
    """A part STANDS_ON the board through its ``origin`` datum (mate closed exactly for the given pose)."""

    p = SynthPart(ref, mesh, position, quaternion, category=category, parent=BOARD_REF, mate_ref=mate_ref, relation="STANDS_ON")
    return p, SynthMate(mate_ref, ref, "origin", BOARD_REF, "top")


def _identity(ref: str, category: str, index: int) -> dict:
    token = "".join(c for c in ref.upper() if c.isalnum())
    if category == "OPTICAL_ELEMENT":
        return {"opensci_sku": f"OS-OPT-TEST-{token}", "identity_kind": "PUBLIC_OPTICAL_SKU", "display_name": f"Test optic {index}"}
    return {"opensci_sku": f"OS-MECH-TEST-{token}", "identity_kind": "COMMERCIAL_SKU", "display_name": f"Test part {index}"}


def synthetic_scene(parts: list[SynthPart], mates: list[SynthMate], *, outline_max: tuple[float, float] = (500.0, 500.0),
                    hole_grid: dict | None = None, fixations: list[dict] | None = None,
                    beam_axes: list[list[Vec3]] | None = None) -> tuple[dict, dict[str, bytes]]:
    """A minimal valid scene payload (+ geometry files). ``parts`` must list parents before children."""

    part_dicts, geometry = [], {}
    for order, sp in enumerate(parts, start=1):
        d = {
            "part_ref": sp.ref,
            "category": sp.category,
            "identity": _identity(sp.ref, sp.category, order),
            "commercial_line_ref": f"B{order:02d}",
            "pose_in_world": pose_dict(sp.position, sp.quaternion),
            "assembly_order": order,
            "datums": [{"datum_key": k, "pose_in_part": pose_dict(*v)} for k, v in sorted(sp.datums.items())],
            "geometry": {"asset_path": f"geometry/{sp.ref}.glb", "fidelity": "NOMINAL_SHAPE", "bounds_min_mm": [0.0, 0.0, 0.0],
                         "bounds_max_mm": [0.0, 0.0, 0.0], "triangle_count": 1},
            "appearance": {"base_color_rgb": [0.5, 0.5, 0.5], "finish": sp.finish},
        }
        if sp.parent is not None:
            d["support"] = {"parent_ref": sp.parent, "mate_ref": sp.mate_ref, "relation": sp.relation}
        if sp.category == "OPTICAL_ELEMENT":
            d["design_binding"] = {"instance_id": f"L{order}", "optical_sku": d["identity"]["opensci_sku"], "catalog_version": "2026-09-27.1"}
        part_dicts.append(d)
    scene: dict = {
        "resolution": {"passed_checks": []},
        "parts": part_dicts,
        "mates": [],
        "fixations": fixations or [],
        "breadboard": {"part_ref": BOARD_REF, "top_z_mm": 0.0, "thickness_mm": 12.7, "outline_min_xy_mm": [0.0, 0.0],
                       "outline_max_xy_mm": list(outline_max), **({"hole_grid": hole_grid} if hole_grid else {})},
        "beam_reference": {
            "design_frame_in_world": pose_dict(),
            "beam_axes": [{"path_key": f"axis-{k}", "basis": "DESIGN_NOMINAL_AXIS", "points_mm": [list(p) for p in pts]}
                          for k, pts in enumerate(beam_axes or [])],
        },
        "bounds": {"min_mm": [0.0, 0.0, -12.7], "max_mm": [outline_max[0], outline_max[1], 100.0]},
    }
    for sm in mates:
        m = {"mate_ref": sm.ref, "part_a_ref": sm.part_a, "datum_a": sm.datum_a, "part_b_ref": sm.part_b, "datum_b": sm.datum_b,
             "b_in_a": pose_dict()}
        if sm.designation is not None:
            m["interface_designation"] = sm.designation
        m["b_in_a"] = pose_dict(*sm.b_in_a) if sm.b_in_a is not None else consistent_b_in_a(scene, m)
        scene["mates"].append(m)
    for sp in parts:
        set_geometry(scene, geometry, sp.ref, sp.mesh, finish=sp.finish)
    return scene, geometry


def synthetic_package(parts: list[SynthPart], mates: list[SynthMate], **options) -> AssemblyPreviewPackageV1:
    scene, geometry = synthetic_scene(parts, mates, **options)
    return rebuild(scene, geometry)


# ------------------------------------------------------------------------------------------ result queries


def check_status(result, check: str) -> str:
    return next(c.status.value for c in result.checks if c.check.value == check)


def statuses(result) -> dict[str, str]:
    return {c.check.value: c.status.value for c in result.checks}


def issues(result, code: str | None = None) -> list:
    return [i for i in result.issues if code is None or i.code.value == code]


def issue_keys(result) -> list[tuple[str, tuple[str, ...], str | None]]:
    return [(i.code.value, tuple(i.part_refs), i.mate_ref) for i in result.issues]


def metric(issue, name: str) -> float:
    return next(m.value for m in issue.metrics if m.name.value == name)


def part_status(result, ref: str) -> str:
    return next(p.status.value for p in result.parts if p.part_ref == ref)
