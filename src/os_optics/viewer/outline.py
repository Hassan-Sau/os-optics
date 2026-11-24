"""The drawn edges of a mesh: creases and boundaries, and the edges that can be a silhouette from some direction.

The surface of a lens is drawn in zones (clear aperture, outline band, rim), one mesh each. Their shared seams are not edges of the
object, so the see-through meshes of one optical surface are welded together first (``outline_groups``): what remains is its real
outer edge and, per view, its silhouette, the curved outline of a face against what lies behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CREASE_DEG = 32.0
WELD_DECIMALS = 5
NEAR_ZERO = 1e-12


@dataclass(frozen=True)
class OutlineGroup:
    """Welded edges of the meshes drawn as one surface: end points (k, 2, 3), the face normals on each side (k, 2, 3), whether an
    edge is always drawn (boundary or crease: ``feature``, k) or only where it is a silhouette."""

    surface_id: str | None
    see_through: bool
    studio_lit: bool
    faint: bool  # a cemented interface: glass on both sides
    rgb: tuple[float, float, float]
    points: np.ndarray
    sides: np.ndarray
    feature: np.ndarray


def weld(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.round(positions, WELD_DECIMALS), axis=0, return_inverse=True)
    return inverse.reshape(-1)[triangles]


def face_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > NEAR_ZERO, n / np.maximum(length, NEAR_ZERO), 0.0)


def feature_edges(positions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welded edges of a mesh: ``(edges (k, 2) welded vertex ids, face normals of each side (k, 2, 3), is_feature (k,))``.

    A feature edge is a boundary edge or a crease sharper than ``CREASE_DEG``; the others can still be silhouettes."""

    welded = weld(positions, triangles)
    normals = face_normals(positions, triangles)
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    faces = np.tile(np.arange(len(triangles)), 3)
    keys = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = faces[order[starts]]
    second = faces[order[np.minimum(starts + 1, len(order) - 1)]]
    second = np.where(counts >= 2, second, first)
    sides = np.stack([normals[first], normals[second]], axis=1)
    cos = np.einsum("ij,ij->i", sides[:, 0], sides[:, 1])
    feature = (counts == 1) | (counts > 2) | (cos < math.cos(math.radians(CREASE_DEG)))
    return unique, sides, feature


def welded_positions(positions: np.ndarray) -> np.ndarray:
    return np.unique(np.round(positions, WELD_DECIMALS), axis=0)


def _group(meshes: list[Any], surface_id: str | None) -> OutlineGroup:
    positions = np.concatenate([m.positions for m in meshes])
    offsets = np.cumsum([0] + [len(m.positions) for m in meshes[:-1]])
    triangles = np.concatenate([m.triangles + offset for m, offset in zip(meshes, offsets, strict=True)])
    edges, sides, feature = feature_edges(positions, triangles)
    points = welded_positions(positions)[edges]
    return OutlineGroup(surface_id=surface_id, see_through=meshes[0].alpha < 1.0 or meshes[0].transmission > 0.0, studio_lit=meshes[0].studio_lit,
                        faint=all(m.thickness_mm == 0.0 for m in meshes), rgb=tuple(meshes[0].rgb), points=points, sides=sides, feature=feature)


def outline_groups(part: Any, *, merge_surfaces: bool) -> list[OutlineGroup]:
    """A part's outlines. ``merge_surfaces``: its see-through studio-lit meshes, per optical surface, as one surface each (the wall of
    a body is its own group); every other mesh by itself."""

    groups: list[OutlineGroup] = []
    merged: dict[str | None, list[Any]] = {}
    for mesh in part.meshes:
        if merge_surfaces and (mesh.alpha < 1.0 or mesh.transmission > 0.0) and mesh.studio_lit:
            merged.setdefault(mesh.surface_id, []).append(mesh)
        else:
            groups.append(_group([mesh], mesh.surface_id))
    groups += [_group(meshes, surface_id) for surface_id, meshes in merged.items()]
    return groups
