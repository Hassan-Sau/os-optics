"""Posed part geometry and view projections of the drawing (pure Python floats, deterministic).

Pose authority is the verified Assembly Preview Package only: a part GLB (part frame, metres) is decoded and welded by
``os_optics.assembly.mesh.read_part_mesh`` (scaled to millimetres) and placed with its EXACT ``pose_in_world``
(``x_world = R(q / |q|) x_local + position_mm``, the quaternion normalized like the detail verifier and the viewers).
Nothing here moves, sizes or re-poses a part.

Only ``+ - * /`` and ``sqrt`` (correctly rounded by IEEE 754) are used, so CPython on every platform and Pyodide give
bit-identical coordinates. Circles use ``UNIT_CIRCLE``: eight literal octant cosines expanded by exact symmetry (the
same table the Assembly Preview geometry writer uses); view directions are fixed literal fractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..assembly.mesh import MeshError, read_part_mesh
from ..assembly.transforms import Rigid, Vec3

V2 = tuple[float, float]

# cos(k * pi / 16), k = 0..8 (exact literals; k = 8 is exactly 0).
OCTANT_COS = (
    1.0,
    0.9807852804032304,
    0.9238795325112867,
    0.8314696123025452,
    0.7071067811865476,
    0.5555702330196023,
    0.38268343236508984,
    0.19509032201612833,
    0.0,
)
OCTANT = 8
QUADRANTS = 4
CIRCLE_SEGMENTS = OCTANT * QUADRANTS
#: Creases: adjacent faces whose normals differ by more than this angle (cos) are a drawn feature edge. The 32-gon
#: mesh cylinders (11.25 deg between facets) stay smooth; box and cap edges (90 deg) are features.
FEATURE_COS = 0.8660254037844386  # cos 30 deg


def _unit_circle() -> tuple[V2, ...]:
    points = []
    for k in range(CIRCLE_SEGMENTS):
        quadrant, r = divmod(k, OCTANT)
        c, s = OCTANT_COS[r], OCTANT_COS[OCTANT - r]
        points.append(((c, s), (-s, c), (-c, -s), (s, -c))[quadrant])
    return tuple(points)


UNIT_CIRCLE = _unit_circle()


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def dist2(a: V2, b: V2) -> float:
    return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))


def placement(position_mm: Vec3, quaternion_wxyz: tuple[float, float, float, float]) -> Rigid:
    """The rigid placement of a pose, the quaternion normalized (``Rigid.from_pose``)."""

    return Rigid.from_pose(position_mm, quaternion_wxyz)


# ------------------------------------------------------------------------------------------------ views


@dataclass(frozen=True)
class View:
    """An orthographic view: ``forward`` points from the viewer into the scene; paper u = p.right, v = p.up."""

    key: str
    forward: Vec3
    right: Vec3
    up: Vec3

    def uv(self, p: Vec3) -> V2:
        return (dot(p, self.right), dot(p, self.up))

    def depth(self, p: Vec3) -> float:
        return dot(p, self.forward)


FRONT = View("FRONT", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # viewed along +Y
SIDE = View("SIDE", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # right side view, viewed along -X
PLAN = View("PLAN", (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # viewed along -Z
# Pictorial: from the front-left, above. forward = (9, 12, -8) / 17 (a Pythagorean quadruple: exact unit length),
# right = (12, -9, 0) / 15, up = right x forward = (72, 96, 225) / 255 -- all fixed literal fractions.
PICTORIAL = View("PICTORIAL", (9.0 / 17.0, 12.0 / 17.0, -8.0 / 17.0), (12.0 / 15.0, -9.0 / 15.0, 0.0),
                 (72.0 / 255.0, 96.0 / 255.0, 225.0 / 255.0))
VIEWS = (FRONT, SIDE, PLAN, PICTORIAL)


# ------------------------------------------------------------------------------------------------ parts


@dataclass(frozen=True)
class PosedMesh:
    """One part's mesh in world millimetres.

    ``shell_of`` gives each triangle's edge-connected shell (overlapping shells of one part unite under the winding
    rule); ``edges`` maps each welded edge ``(i, j)`` (``i < j``) to the triangles that share it."""

    part_ref: str
    vertices: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    normals: tuple[Vec3, ...]  # unnormalized, outward (right-hand winding)
    shell_of: tuple[int, ...]
    edges: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    glass: bool = False

    def bounds(self) -> tuple[Vec3, Vec3]:
        used = sorted({i for t in self.triangles for i in t})
        lo = tuple(min(self.vertices[i][k] for i in used) for k in range(3))
        hi = tuple(max(self.vertices[i][k] for i in used) for k in range(3))
        return lo, hi  # type: ignore[return-value]


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def posed_mesh(part_ref: str, vertices: tuple[Vec3, ...], triangles: tuple[tuple[int, int, int], ...], *,
               glass: bool = False) -> PosedMesh:
    """Topology of a welded world-space triangle mesh: normals, shells, edge -> triangles."""

    normals = tuple(cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a])) for a, b, c in triangles)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for ti, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(ti)
    parent = list(range(len(triangles)))
    for faces in edge_faces.values():
        for other in faces[1:]:
            ra, rb = _find(parent, faces[0]), _find(parent, other)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots: dict[int, int] = {}
    shell_of = []
    for ti in range(len(triangles)):
        root = _find(parent, ti)
        shell_of.append(roots.setdefault(root, len(roots)))
    edges = tuple((key, tuple(faces)) for key, faces in sorted(edge_faces.items()))
    return PosedMesh(part_ref, vertices, triangles, normals, tuple(shell_of), edges, glass)


def load_posed_mesh(part_ref: str, glb: bytes, pose: Rigid, *, glass: bool = False) -> PosedMesh:
    """Decode a part GLB (verified by the package contract), weld it and place it with its pose. Raises ``MeshError``."""

    mesh = read_part_mesh(glb)
    return posed_mesh(part_ref, pose.apply_all(mesh.vertices), mesh.triangles, glass=glass)


def box_mesh(part_ref: str, lo: Vec3, hi: Vec3) -> PosedMesh:
    """A closed, outward-oriented world-axis box (the semantic breadboard slab of one view)."""

    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5))
    triangles = tuple(t for a, b, c, d in quads for t in ((a, b, c), (a, c, d)))
    return posed_mesh(part_ref, corners, triangles)  # type: ignore[arg-type]


def circle_points(centre: Vec3, e1: Vec3, e2: Vec3, radius: float) -> list[Vec3]:
    """The 32-gon of ``UNIT_CIRCLE`` in the plane spanned by unit vectors ``e1``, ``e2`` about ``centre``."""

    return [add(centre, add(scale(e1, radius * c), scale(e2, radius * s))) for c, s in UNIT_CIRCLE]


__all__ = [
    "CIRCLE_SEGMENTS",
    "FEATURE_COS",
    "FRONT",
    "MeshError",
    "PICTORIAL",
    "PLAN",
    "SIDE",
    "UNIT_CIRCLE",
    "VIEWS",
    "PosedMesh",
    "View",
    "box_mesh",
    "circle_points",
    "load_posed_mesh",
    "placement",
    "posed_mesh",
]
