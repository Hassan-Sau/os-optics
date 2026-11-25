"""Triangle meshes of polyhedral parts for the facet-tracing tests: convex solids from their corners and faces, oriented
outward by construction (each face is fanned from its first corner and turned so its normal points away from the solid's
centroid), and their analytic references."""

from __future__ import annotations

import math

import numpy as np

from os_optics import EngineConfiguration, ValidationRequest
from os_optics.contracts import (EngineeringComponentSnapshot, FacetApproximation, MeshFacetGroup, MeshSurface, ProjectSnapshot)
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.contracts.optics import MirrorInteraction, RefractInteraction

EXACT = FacetApproximation(kind="FACETED_EXACT", source="polyhedral part")
GLASS = "GLASS15"
N_GLASS = 1.5


def glass(index: float = N_GLASS, material_id: str = GLASS) -> ConstantIndexMaterial:
    return ConstantIndexMaterial(material_id=material_id, refractive_index=index, valid_wavelength_nm=(400.0, 1100.0))


def convex_solid(vertices, faces) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...], list[list[int]]]:
    """``(vertices, triangles, face_triangles)`` of a convex solid: every face (a loop of vertex indices, either sense) fanned
    into triangles, counter-clockwise seen from outside; ``face_triangles[f]``: the triangle indices of face ``f``."""
    pts = np.array(vertices, dtype=float)
    centroid = pts.mean(axis=0)
    triangles: list[tuple[int, int, int]] = []
    face_triangles: list[list[int]] = []
    for face in faces:
        loop = list(face)
        normal = np.cross(pts[loop[1]] - pts[loop[0]], pts[loop[2]] - pts[loop[0]])
        if float(np.dot(normal, pts[loop].mean(axis=0) - centroid)) < 0.0:
            loop.reverse()
        mine = []
        for k in range(1, len(loop) - 1):
            mine.append(len(triangles))
            triangles.append((loop[0], loop[k], loop[k + 1]))
        face_triangles.append(mine)
    return tuple(tuple(float(x) for x in v) for v in vertices), tuple(triangles), face_triangles


def extruded(polygon_xz, half_length_y: float):
    """A prism: the polygon (x, z) corners extruded along y over ``[-half_length_y, half_length_y]``; faces: the two caps, then
    one side face per polygon edge (edge ``i`` from corner ``i`` to ``i + 1``)."""
    n = len(polygon_xz)
    vertices = [(x, -half_length_y, z) for x, z in polygon_xz] + [(x, half_length_y, z) for x, z in polygon_xz]
    faces = [list(range(n)), list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return convex_solid(vertices, faces)


def solid(surface_id: str, vertices, triangles, *, medium: str = GLASS, groups=None, approximation=EXACT) -> MeshSurface:
    return MeshSurface(surface_id=surface_id, vertices_mm=vertices, triangles=triangles, approximation=approximation, closed=True,
                       medium_positive="air", medium_negative=medium, interaction=RefractInteraction(), facet_groups=groups)


def silvered(triangles) -> MeshFacetGroup:
    """Facets mirrored on the inside (light in the solid, the negative side, reflects)."""
    return MeshFacetGroup(triangles=tuple(triangles), interaction=MirrorInteraction(reflective_side="negative"))


def part(instance_id: str, *surfaces) -> EngineeringComponentSnapshot:
    return EngineeringComponentSnapshot(instance_id=instance_id, role="prism", surfaces=tuple(surfaces))


def request(*components, observations=(), materials=None) -> ValidationRequest:
    return ValidationRequest(project=ProjectSnapshot(name="facets", components=tuple(components), observations=tuple(observations),
                                                     materials=tuple(materials if materials is not None else (glass(),))),
                             configuration=EngineConfiguration(run_layout=False, run_control=False))


def unit(v) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def snell(d, n, n1: float, n2: float) -> np.ndarray:
    """Vector Snell refraction of the unit direction ``d`` at a plane of unit normal ``n`` (either orientation)."""
    d, n = unit(d), unit(n)
    if np.dot(d, n) > 0:
        n = -n
    cos_i = -float(np.dot(d, n))
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    return eta * d + (eta * cos_i - math.sqrt(k)) * n


def fresnel_t(cos_i: float, n1: float, n2: float) -> tuple[float, float]:
    """Power transmittances (s, p) of a bare interface."""
    sin_t = n1 / n2 * math.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return 1.0 - rs * rs, 1.0 - rp * rp
