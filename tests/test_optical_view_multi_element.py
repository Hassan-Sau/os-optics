"""Multi-element designs are ordinary views. A folded path (a Keplerian telescope of two catalog lenses, a 45 deg fold
mirror, a window, a 45 deg plate beamsplitter with both branches and a thin transmissive filter) and a polarization path
(a waveplate and a linear polarizer), every element found in the catalog by query and drawn as one generic
``OPTIC_BODY`` bound to its compiled public model. The Engine's rays meet every element on its published surfaces, the
split branch is drawn as a child path, and the polarization elements carry their local +X axis."""

from __future__ import annotations

import math

import pytest
from optical_view_reference import build_scene, folded_parts, folded_project, polarization_parts, polarization_project, to_local

from os_optics import trace_ray_paths
from os_optics._vendor.shared_contracts.optical_view import OpticBodyGeometryV1, OpticSurfaceRoleV1
from os_optics.local import load_public_catalog

from catalog_store_helpers import REAL_CATALOG
from os_optics.optical_view import optical_view_files, parse_glb, read_accessor
from os_optics.optical_view import geometry as g

FOLDED_ROLES = {"L1": "lens", "L2": "lens", "M1": "mirror", "W1": "window", "BS": "splitter", "F1": "filter"}
POLARIZATION_ROLES = {"WP": "waveplate", "POL": "polarizer"}


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


@pytest.fixture(scope="module")
def folded(catalog):
    return build_scene(folded_project(catalog), catalog)


@pytest.fixture(scope="module")
def polarization(catalog):
    return build_scene(polarization_project(catalog), catalog)


def _check_bodies(scene, parts, roles):
    elements = {e.element_id: e for e in scene.elements}
    assert list(elements) == ["SRC", *roles]
    for element_id, role in roles.items():
        element, product = elements[element_id], parts[element_id]
        geom = element.geometry
        assert isinstance(geom, OpticBodyGeometryV1)
        assert (element.optical_role.value, element.optical_sku, geom.category, geom.form) == (role, product.optical_sku, product.category, product.form)
        assert (geom.reference_axis is not None) == product.frame.is_reflector
    return elements


def test_every_category_is_one_optic_body_of_its_product(folded, polarization, catalog):
    elements = _check_bodies(folded[0], folded_parts(catalog), FOLDED_ROLES)
    assert _coated_face(elements["M1"].geometry).role is OpticSurfaceRoleV1.REFLECT
    assert _coated_face(elements["BS"].geometry).role is OpticSurfaceRoleV1.SPLIT
    assert {s.role for s in elements["F1"].geometry.surfaces} <= {OpticSurfaceRoleV1.REFRACT, OpticSurfaceRoleV1.SPLIT}
    _, parts = polarization_parts(catalog)
    elements = _check_bodies(polarization[0], parts, POLARIZATION_ROLES)
    for element_id, kind in (("POL", "LINEAR_POLARIZER"), ("WP", "RETARDER")):
        geom = elements[element_id].geometry
        assert geom.model_kind == "POLARIZATION_ELEMENT" and geom.polarization.element.value == kind and geom.polarization.axis_local == "+X"
        assert [s.surface_id for s in geom.surfaces] == ["element"]


def _coated_face(geom):
    """The coated face of a face-normal body: the surface at the frame origin with local +Z its outward normal (a surface
    sequence's last surface, at z = 0; a surface set's posed surface there)."""
    assert geom.reference_axis is not None and geom.reference_axis.value == "FACE_NORMAL"
    (face,) = [s for s in geom.surfaces if (tuple(s.origin_mm) == (0.0, 0.0, 0.0) and g.unit(tuple(s.normal)) == (0.0, 0.0, 1.0)
                                            if s.is_posed else s.vertex_z_mm == 0.0)]
    return face


def _surface_coordinates(surface, point):
    """A point of the element frame in the surface's own frame (a, b along its local X / Y, c along its axis from the vertex)."""
    if not surface.is_posed:
        return point[0], point[1], point[2] - surface.vertex_z_mm
    n, u = g.unit(tuple(surface.normal)), g.unit(tuple(surface.u_axis))
    d = g.sub(point, tuple(surface.origin_mm))
    return g.dot(d, u), g.dot(d, g.cross(n, u)), g.dot(d, n)


def _inside_clear_aperture(surface, a, b, tol):
    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    if surface.clear_radius_mm is not None:
        return math.hypot(a - cx, b - cy) <= surface.clear_radius_mm + tol
    hx, hy = surface.clear_half_widths_mm
    return abs(a - cx) <= hx + tol and abs(b - cy) <= hy + tol


def test_the_split_branch_is_a_drawn_child_path(folded):
    scene, _ = folded
    main = [p for p in scene.beam_paths if p.parent_path_id is None]
    branches = [p for p in scene.beam_paths if p.parent_path_id is not None]
    assert len(main) == 1 and branches, "the plate beamsplitter must give a reflected branch"
    assert all(b.parent_path_id == main[0].path_id and b.path_id.startswith(f"{main[0].path_id}.") for b in branches)
    planes = {p.observation_id: p for p in scene.observation_planes}
    assert [f.path_id for f in planes["screen"].footprints] == [main[0].path_id]
    assert planes["screen_r"].footprints and {f.path_id for f in planes["screen_r"].footprints} <= {b.path_id for b in branches}


@pytest.mark.parametrize(("design", "roles"), [("folded", FOLDED_ROLES), ("polarization", POLARIZATION_ROLES)])
def test_the_engine_rays_meet_every_element_on_its_published_surfaces(design, roles, request):
    scene, engine_request = request.getfixturevalue(design)
    elements = {e.element_id: e for e in scene.elements}
    traced = {p.path_id: p for p in trace_ray_paths(engine_request).paths}
    met = set()
    for path in scene.beam_paths:
        for k, surface_key in enumerate(traced[path.path_id].chief_surface_ids):
            owner, _, surface_id = (surface_key or "").partition("/")
            element = elements.get(owner)
            if element is None or not isinstance(element.geometry, OpticBodyGeometryV1):
                continue
            (surface,) = [s for s in element.geometry.surfaces if s.surface_id == surface_id]
            for ray in (path.chief_ray, *path.pupil_rays):
                if k >= len(ray.points_mm):
                    continue
                a, b, c = _surface_coordinates(surface, to_local(element, ray.points_mm[k]))
                assert _inside_clear_aperture(surface, a, b, 1e-6), (owner, surface_id)
                assert c == pytest.approx(surface.sag_at(a, b), abs=1e-5), (owner, surface_id)
            met.add(owner)
    assert met == set(roles)


def _materials(gltf, nodes, element_id):
    owners = [nodes[f"element:{element_id}"]] + [gltf["nodes"][c] for c in nodes[f"element:{element_id}"].get("children", ())]
    return {gltf["materials"][p["material"]]["name"] for o in owners if "mesh" in o for p in gltf["meshes"][o["mesh"]]["primitives"]}


def test_the_glb_draws_each_category_with_its_coated_faces(folded):
    scene, _ = folded
    gltf, _ = parse_glb(optical_view_files(scene)["optical_scene.glb"])
    nodes = {n["name"]: n for n in gltf["nodes"]}
    assert {"optic_glass", "optic_edge"} <= _materials(gltf, nodes, "L1") and "optic_edge" in _materials(gltf, nodes, "L2")
    assert {"mirror_face", "mirror_substrate"} <= _materials(gltf, nodes, "M1") and "optic_glass" not in _materials(gltf, nodes, "M1")
    assert "splitter_face" in _materials(gltf, nodes, "BS")
    assert _materials(gltf, nodes, "F1") & {"filter_face", "optic_glass", "optic_substrate"}
    assert nodes["element:M1"]["extras"]["surfaces"][-1].endswith(":REFLECT")


def test_the_glb_draws_the_polarization_axis_along_local_x(polarization):
    scene, _ = polarization
    gltf, binary = parse_glb(optical_view_files(scene)["optical_scene.glb"])
    nodes = {n["name"]: n for n in gltf["nodes"]}
    jones_plane = {"polarization_plane", "aperture_outline", "optic_rim"}  # the Jones plane: clear aperture, outline band, rim
    assert _materials(gltf, nodes, "POL") == {"polarizer_body", "polarization_axis", *jones_plane}
    assert _materials(gltf, nodes, "WP") == {"waveplate_body", "polarization_axis", *jones_plane}
    for element_id in ("POL", "WP"):
        node = nodes[f"element:{element_id}"]
        assert node["extras"]["polarization_axis_local"] == "+X" and node["extras"]["model_kind"] == "POLARIZATION_ELEMENT"
        (bar,) = [p for p in gltf["meshes"][node["mesh"]]["primitives"] if gltf["materials"][p["material"]]["name"] == "polarization_axis"]
        points = read_accessor(gltf, binary, bar["attributes"]["POSITION"])
        span_x = max(p[0] for p in points) - min(p[0] for p in points)
        span_y = max(p[1] for p in points) - min(p[1] for p in points)
        assert span_x > 5 * span_y  # a bar along local +X
        element = next(e for e in scene.elements if e.element_id == element_id)
        assert g.rotate(element.pose.quaternion_wxyz, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)  # design -Z here
