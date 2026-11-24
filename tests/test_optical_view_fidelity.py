"""Physics and picture agree on a representative compound objective: every Engine ray point (chief and pupil rays of
the run's own trace, as recorded in the scene) that lies on a lens surface lies on the exact published profile and on
the triangles that the GLB draws for that very surface (its ``surface:<element>/<surface>`` node), within the stated
chord tolerance -- the drawing is the traced geometry at true scale."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from catalog_store_helpers import REAL_CATALOG
from compound_objective import objective_project
from os_optics.local import load_public_catalog, run_optical_project
from os_optics.optical_view import geometry as g
from os_optics.optical_view import optical_view_files, parse_glb, read_accessor
from os_optics.optical_view.surface_mesh import CHORD_TOLERANCE_MM

FLOAT32_MM = 2e-5  # float32 storage of element-frame coordinates (< 30 mm) and the scene's 1e-6 mm rounding
ON_SURFACE_MM = 1e-4  # an Engine point this close to a surface's exact profile is a hit on that surface


@pytest.fixture(scope="module")
def run():
    catalog = load_public_catalog(REAL_CATALOG)
    return run_optical_project(objective_project(catalog), catalog=REAL_CATALOG), catalog


def _surface_triangles(gltf, binary, element_id: str) -> dict[str, np.ndarray]:
    (node,) = [n for n in gltf["nodes"] if n.get("name") == f"element:{element_id}"]
    out = {}
    for c in node["children"]:
        child = gltf["nodes"][c]
        tris = []
        for prim in gltf["meshes"][child["mesh"]]["primitives"]:
            pos = np.asarray(read_accessor(gltf, binary, prim["attributes"]["POSITION"]), dtype=float)
            idx = np.asarray(read_accessor(gltf, binary, prim["indices"]), dtype=np.int64).reshape(-1, 3)
            tris.append(pos[idx])
        out[child["extras"]["surface_id"]] = np.concatenate(tris)
    return out


def _distance_to_mesh(tris: np.ndarray, p: tuple[float, float, float]) -> float:
    """Normal distance from ``p`` to the plane of the triangle whose (x, y) projection contains it (a coaxial surface is a
    height field over its outline)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    v2 = np.asarray(p[:2]) - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-15
    u = np.where(ok, (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(ok, den, 1.0), -1.0)
    v = np.where(ok, (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(ok, den, 1.0), -1.0)
    inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
    assert inside.any(), p
    k = int(np.flatnonzero(inside)[0])
    n = np.cross(b[k] - a[k], c[k] - a[k])
    n /= np.linalg.norm(n)
    return float(abs(np.dot(np.asarray(p) - a[k], n)))


def test_engine_points_on_every_lens_surface_lie_on_the_drawn_surface(run):
    run, catalog = run
    scene = run.optical_view
    gltf, binary = parse_glb(optical_view_files(scene)["optical_scene.glb"])
    hits: dict[tuple[str, str], int] = {}
    worst_profile = worst_mesh = 0.0
    for element in scene.elements:
        if element.optical_sku is None:
            continue
        model = catalog.get(element.optical_sku).simulation_models[0]
        triangles = _surface_triangles(gltf, binary, element.element_id)
        q = element.pose.quaternion_wxyz
        inverse = (q[0], -q[1], -q[2], -q[3])
        for path in scene.beam_paths:
            for ray in (path.chief_ray, *path.pupil_rays):
                for point in ray.points_mm:
                    local = g.rotate(inverse, g.sub(tuple(point), tuple(element.pose.position_mm)))
                    rho = math.hypot(local[0], local[1])
                    for surface in model.surfaces:
                        radius = None if surface.radius_mm is None else float(Decimal(surface.radius_mm))
                        vertex = float(Decimal(surface.vertex_z_mm))
                        exact = vertex if radius is None else vertex + radius - math.copysign(math.sqrt(radius * radius - rho * rho), radius)
                        if abs(local[2] - exact) > ON_SURFACE_MM or rho > float(Decimal(surface.clear_radius_mm)) + 1e-6:
                            continue
                        worst_profile = max(worst_profile, abs(local[2] - exact))
                        worst_mesh = max(worst_mesh, _distance_to_mesh(triangles[surface.surface_id], local))
                        hits[(element.element_id, surface.surface_id)] = hits.get((element.element_id, surface.surface_id), 0) + 1
    assert set(hits) == {(e, s) for e, n in (("L1", 3), ("L2", 3), ("L3", 2)) for s in (f"S{i}" for i in range(1, n + 1))}
    assert sum(hits.values()) >= 50
    assert worst_profile <= 1e-5  # the Engine's own intersections lie on the published profile
    assert worst_mesh <= CHORD_TOLERANCE_MM + FLOAT32_MM  # ... and on the drawn mesh of that surface
