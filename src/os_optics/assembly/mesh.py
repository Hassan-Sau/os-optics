"""Part geometry of a verified Assembly Preview Package as closed triangle meshes (part frame, millimetres).

A part GLB (``PartGeometryV1``: one node / mesh, triangle lists, positions in metres, part frame, +Z up, no node
transform) has already passed the canonical package verifier. Here its triangles are decoded, vertices are welded
by exact position, and the mesh is accepted for volume-based checks only if

* every directed edge is matched by its reverse (a closed, consistently oriented surface, possibly made of several
  overlapping shells such as a tube and a box: the checks use the winding number, so overlapping shells unite), and
* every connected shell encloses a positive volume (outward orientation).

Anything else raises ``MeshError`` with a stable code (``require_solid=False``: an open or inward-facing mesh is
returned instead, with its ``defect`` and no volume, for the checks that need only its vertices; an unreadable one
still raises).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import StrEnum

from .._vendor.shared_contracts.assembly_preview import parse_glb
from .transforms import Vec3

MM_PER_M = 1000.0
FLOAT_COMPONENT = 5126
INDEX_COMPONENTS = {5121: "B", 5123: "H", 5125: "I"}
TRIANGLES_MODE = 4
VEC3_SIZE = 3
MIN_SHELL_VOLUME_MM3 = 1e-9
TETRAHEDRON_FACTOR = 6.0  # signed volume = sum of det(p, q, r) / 6 over the triangles


class MeshErrorCode(StrEnum):
    UNREADABLE = "UNREADABLE"
    NOT_CLOSED = "NOT_CLOSED"
    DEGENERATE = "DEGENERATE"


class MeshError(ValueError):
    def __init__(self, code: MeshErrorCode, detail: str):
        super().__init__(f"{code.value}: {detail}")
        self.code = code


@dataclass(frozen=True)
class PartMesh:
    """A triangle mesh in the part frame (mm); ``triangles`` index ``vertices``. A closed, outward-oriented mesh has its
    ``volume_mm3``; a mesh read with ``require_solid=False`` that is not one has ``defect`` (``NOT_CLOSED`` /
    ``DEGENERATE``) and ``volume_mm3`` None: only its vertices and bounds may be used."""

    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    volume_mm3: float | None
    bounds_min: Vec3
    bounds_max: Vec3
    defect: MeshErrorCode | None = None

    @property
    def solid(self) -> bool:
        return self.defect is None

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


def _accessor_floats(document: dict, binary: bytes, index: int) -> list[Vec3]:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    if accessor["componentType"] != FLOAT_COMPONENT or accessor["type"] != "VEC3" or view.get("byteStride", VEC3_SIZE * 4) != VEC3_SIZE * 4:
        raise MeshError(MeshErrorCode.UNREADABLE, "POSITION must be tightly packed float32 VEC3")
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if offset + count * VEC3_SIZE * 4 > view.get("byteOffset", 0) + view["byteLength"]:
        raise MeshError(MeshErrorCode.UNREADABLE, "POSITION accessor overruns its buffer view")
    flat = struct.unpack_from(f"<{count * VEC3_SIZE}f", binary, offset)
    if not all(math.isfinite(c) for c in flat):
        raise MeshError(MeshErrorCode.UNREADABLE, "a vertex coordinate is not finite")
    return [(flat[i] * MM_PER_M, flat[i + 1] * MM_PER_M, flat[i + 2] * MM_PER_M) for i in range(0, len(flat), VEC3_SIZE)]


def _accessor_indices(document: dict, binary: bytes, index: int) -> tuple[int, ...]:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    fmt = INDEX_COMPONENTS.get(accessor["componentType"])
    if fmt is None or accessor["type"] != "SCALAR" or "byteStride" in view:
        raise MeshError(MeshErrorCode.UNREADABLE, "indices must be tightly packed unsigned integers")
    count = accessor["count"]
    size = struct.calcsize(fmt)
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if offset + count * size > view.get("byteOffset", 0) + view["byteLength"]:
        raise MeshError(MeshErrorCode.UNREADABLE, "index accessor overruns its buffer view")
    return struct.unpack_from(f"<{count}{fmt}", binary, offset)


def _raw_triangles(data: bytes) -> list[tuple[Vec3, Vec3, Vec3]]:
    try:
        document, binary = parse_glb(data)
        primitives = document["meshes"][0]["primitives"]
        triangles: list[tuple[Vec3, Vec3, Vec3]] = []
        for primitive in primitives:
            if primitive.get("mode", TRIANGLES_MODE) != TRIANGLES_MODE:
                raise MeshError(MeshErrorCode.UNREADABLE, "only triangle lists are supported")
            positions = _accessor_floats(document, binary, primitive["attributes"]["POSITION"])
            if "indices" in primitive:
                indices = _accessor_indices(document, binary, primitive["indices"])
            else:
                indices = tuple(range(len(positions)))
            if len(indices) % VEC3_SIZE or any(i >= len(positions) for i in indices):
                raise MeshError(MeshErrorCode.UNREADABLE, "index list is not a valid triangle list")
            triangles.extend((positions[indices[k]], positions[indices[k + 1]], positions[indices[k + 2]])
                             for k in range(0, len(indices), VEC3_SIZE))
    except MeshError:
        raise
    except (ValueError, KeyError, IndexError, TypeError, struct.error, UnicodeDecodeError) as exc:
        raise MeshError(MeshErrorCode.UNREADABLE, f"malformed GLB mesh ({type(exc).__name__})") from exc
    if not triangles:
        raise MeshError(MeshErrorCode.UNREADABLE, "the mesh has no triangles")
    return triangles


def _signed_volume(vertices: tuple[Vec3, ...], triangles: list[tuple[int, int, int]]) -> float:
    total = 0.0
    for a, b, c in triangles:
        p, q, r = vertices[a], vertices[b], vertices[c]
        total += (p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0]))
    return total / TETRAHEDRON_FACTOR


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def read_part_mesh(data: bytes, *, require_solid: bool = True) -> PartMesh:
    """Decode, weld and validate one part GLB. Raises ``MeshError``; with ``require_solid=False`` only for an unreadable
    mesh (a mesh that is not a closed, outward-oriented solid is returned with its ``defect``)."""

    welded: dict[Vec3, int] = {}
    vertices: list[Vec3] = []
    triangles: list[tuple[int, int, int]] = []
    for tri in _raw_triangles(data):
        ids = []
        for p in tri:
            index = welded.get(p)
            if index is None:
                index = welded[p] = len(vertices)
                vertices.append(p)
            ids.append(index)
        a, b, c = ids
        if a != b and b != c and a != c:
            triangles.append((a, b, c))
    frozen = tuple(vertices)
    defect, volume = _solid_volume(frozen, triangles)
    if defect is not None and require_solid:
        raise MeshError(*defect)
    used = sorted({i for tri in triangles for i in tri}) or list(range(len(frozen)))  # no triangle left: every vertex
    lo = tuple(min(frozen[i][k] for i in used) for k in range(VEC3_SIZE))
    hi = tuple(max(frozen[i][k] for i in used) for k in range(VEC3_SIZE))
    return PartMesh(vertices=frozen, triangles=tuple(triangles), volume_mm3=volume,
                    bounds_min=lo, bounds_max=hi, defect=defect[0] if defect else None)  # type: ignore[arg-type]


def _solid_volume(frozen: tuple[Vec3, ...], triangles: list[tuple[int, int, int]]
                  ) -> tuple[tuple[MeshErrorCode, str] | None, float | None]:
    """``(None, volume)`` of a closed, outward-oriented welded mesh, else ``((code, why), None)``."""

    if not triangles:
        return (MeshErrorCode.DEGENERATE, "every triangle collapses when vertices are welded"), None
    balance: dict[tuple[int, int], int] = {}
    for a, b, c in triangles:
        for edge in ((a, b), (b, c), (c, a)):
            balance[edge] = balance.get(edge, 0) + 1
    if any(balance.get((b, a), 0) != count for (a, b), count in balance.items()):
        return (MeshErrorCode.NOT_CLOSED, "an edge is not matched by an opposite edge"), None
    volumes = _shell_volumes(frozen, triangles)
    if not all(v > MIN_SHELL_VOLUME_MM3 for v in volumes):  # written so that NaN cannot pass
        return (MeshErrorCode.DEGENERATE, "a closed shell encloses no positive volume (flat or inward-facing)"), None
    return None, sum(volumes)


def _shell_volumes(frozen: tuple[Vec3, ...], triangles: list[tuple[int, int, int]]) -> list[float]:
    parent = list(range(len(frozen)))
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c)):
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    shells: dict[int, list[tuple[int, int, int]]] = {}
    for tri in triangles:
        shells.setdefault(_find(parent, tri[0]), []).append(tri)
    return [_signed_volume(frozen, shells[root]) for root in sorted(shells)]
