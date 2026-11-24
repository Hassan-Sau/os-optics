"""Convex bodies of ``SURFACE_SET`` models for the local runtime: ``solid_of`` builds the float solid of a published
``ConvexBodyV1`` from its exact corners (the contract's decimal geometry); the polygon, section and clipping geometry is
``optical_view.solids``, shared with the optical view."""

from __future__ import annotations

from .._vendor.shared_contracts.design_catalog import ConvexBodyV1
from ..optical_view.solids import Solid, clip_segment, face_polygon, section_polygon

__all__ = ["Solid", "clip_segment", "face_polygon", "section_polygon", "solid_of"]


def _float3(values) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def solid_of(body: ConvexBodyV1) -> Solid:
    """The float solid of a published body: its half-spaces, side radius and (for a polyhedron) its exact corners."""

    planes = tuple((_float3(face.outward_normal), _dot(_float3(face.outward_normal), _float3(face.point_mm))) for face in body.faces)
    side = float(body.side_radius_mm) if body.side_radius_mm is not None else None
    return Solid(body.medium, planes, side, tuple(_float3(v) for v in body.vertices_mm()))
