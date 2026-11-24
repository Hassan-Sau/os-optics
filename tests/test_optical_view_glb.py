"""The drawing of every public catalog product (no Engine run needed): each ``OPTIC_BODY`` is drawn from its published
surfaces at true scale, within the stated chord tolerance, split the way the tracer models it, deterministically.

* every optical surface is one group (one GLB child node ``surface:<element>/<surface>`` with its identity, published
  profile, clear aperture and media in its extras) whose vertices lie exactly on the published profile; its clear
  aperture is drawn in the surface's material, a band just outside it marks the clear-aperture outline and the opaque
  rim reaches the outline in its own material; nothing lies outside the published design body;
* the triangles stay within ``CHORD_TOLERANCE_MM`` of the profile: checked here against the published decimals in
  40-digit arithmetic at every triangle's centroid and edge midpoints (the exact sphere distance for spheres), on the
  steepest surface of every kind of body and on the near-hemispherical lens that the fixed 12 x 48 tessellation drew
  0.43 mm off its sphere;
* a shallow lens keeps its true sag and thickness (no curvature is magnified);
* the zones of a surface are watertight (every inner edge shared by two triangles: no cracks between rows of different
  density), and every triangle winds counter-clockwise about its own drawn normal;
* the bytes are identical in process and across interpreter processes with different hash seeds.

The products are the seed catalog release's own, whatever release it is; nothing here names a SKU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from decimal import Decimal, localcontext

import pytest

from os_optics._vendor.shared_contracts.common.hashes import HashProfile, HashRef
from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from os_optics._vendor.shared_contracts.optical_design import ValidatedProjectRefV1
from os_optics._vendor.shared_contracts.optical_view import (
    CameraHintV1,
    EngineRunRefV1,
    OpticalSimulationSceneV1,
    OpticalViewSettingsV1,
    OpticBodyGeometryV1,
    OpticSurfaceV1,
    SceneBoundsV1,
    SceneElementV1,
    ViewBuilderRefV1,
)
from os_optics.optical_view import geometry as g
from catalog_store_helpers import REAL_CATALOG
from os_optics.optical_view import parse_glb, read_accessor, scene_glb_bytes
from os_optics.optical_view.builder import optic_body
from os_optics.optical_view.glb import AXIS_BAR_LIFT_MM
from os_optics.optical_view.optic_mesh import draw_body, surface_material
from os_optics.optical_view.surface_mesh import CHORD_TOLERANCE_MM, UNDEFINED_MATERIAL

D = "0" * 64
EPS = 1e-4  # float32 storage of mm coordinates
FLOAT_EPS = 1e-9  # float64 drawing coordinates against the float profile
RIM_MATERIALS = {"aperture_outline", "optic_rim"}
REFERENCE_DIGITS = 40


def bundled_catalog() -> PublicOpticalCatalogV2:
    """The repository's seed catalog release: every real product is drawn (the Engine bundles no catalog)."""
    return PublicOpticalCatalogV2.model_validate(json.loads(REAL_CATALOG.read_bytes()))


def single_element_scene(catalog_version: str, element: SceneElementV1) -> OpticalSimulationSceneV1:
    return OpticalSimulationSceneV1(
        schema="opensci.optical_simulation_scene",
        project=ValidatedProjectRefV1(project_hash=HashRef(profile=HashProfile.OSC1, digest=D, subject="project_document")),
        catalog_version=catalog_version,
        engine_run=EngineRunRefV1(engine_version="0.0.0", input_hash=HashRef(profile=HashProfile.ENGINE_EXACT_PY1, digest=D, subject="engine_input"),
                                  trace_hash=D, ray_trace_version="1.0"),
        builder=ViewBuilderRefV1(builder_package="os-optics", builder_version="0.0.0", settings=OpticalViewSettingsV1()),
        elements=(element,),
        bounds=SceneBoundsV1(min_mm=(-50.0, -50.0, -50.0), max_mm=(50.0, 50.0, 50.0)),
        camera_hint=CameraHintV1(target_mm=(0.0, 0.0, 0.0), eye_mm=(-100.0, -200.0, 100.0)),
    )


def product_scenes(catalog: PublicOpticalCatalogV2) -> list[tuple[str, OpticalSimulationSceneV1]]:
    """One single-element scene per (product, published model): the element at the origin, playing its first design role."""
    scenes = []
    for product in catalog.products:
        for model in product.simulation_models:
            element = SceneElementV1(element_id="E", optical_role=sorted(product.design_roles)[0], optical_sku=product.optical_sku,
                                     label=product.display_name, pose={"position_mm": (0.0, 0.0, 0.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0)},
                                     geometry=optic_body(product, model, catalog))
            scenes.append((f"{product.optical_sku}:{model.model_kind}", single_element_scene(catalog.catalog_version, element)))
    return scenes


@pytest.fixture(scope="module")
def catalog():
    return bundled_catalog()


@pytest.fixture(scope="module")
def drawings(catalog):
    """``(name, published model, geometry, drawing)`` of every product and model, drawn once."""
    out = []
    for product in catalog.products:
        for model in product.simulation_models:
            geom = optic_body(product, model, catalog)
            out.append((f"{product.optical_sku}:{model.model_kind}", model, geom, draw_body(geom)))
    return out


# ------------------------------------------------------------------------------------------------ independent profile
# z = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2)) + A4 r^4 + A6 r^6 ... (r along the curved axis of a cylinder), written here
# independently of the drawing code.


def _profile_z(surface: OpticSurfaceV1, x: float, y: float) -> float | None:
    rho_sq = y * y if surface.cylinder_axis == "X" else x * x if surface.cylinder_axis == "Y" else x * x + y * y
    z = 0.0
    if surface.radius_mm is not None:
        c, k = 1.0 / surface.radius_mm, surface.conic or 0.0
        arg = 1.0 - (1.0 + k) * c * c * rho_sq
        if arg < 0.0:
            return None
        z = c * rho_sq / (1.0 + math.sqrt(arg))
    for n, a in enumerate(surface.aspheric_coefficients or (), start=2):
        z += a * rho_sq ** n
    if surface.linear_sag_coefficient:  # an axicon's cone term a * rho
        z += surface.linear_sag_coefficient * math.sqrt(rho_sq)
    return z


def _surface_frame(surface: OpticSurfaceV1):
    """(origin, normal, u, v) of a posed surface (unit axes), or the coaxial frame at its vertex."""
    if not surface.is_posed:
        return (0.0, 0.0, surface.vertex_z_mm), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    n, u = g.unit(tuple(surface.normal)), g.unit(tuple(surface.u_axis))
    return tuple(surface.origin_mm), n, u, g.cross(n, u)


def _in_surface(surface: OpticSurfaceV1, p):
    origin, n, u, v = _surface_frame(surface)
    d = g.sub(p, origin)
    return g.dot(d, u), g.dot(d, v), g.dot(d, n)


def _inside_design_body(geom, point, lift: float = 0.0) -> bool:
    x, y, z = point
    if geom.shape == "CYLINDER":
        return math.hypot(x, y) <= geom.diameter_mm / 2 + EPS and geom.z_min_mm - lift - EPS <= z <= geom.z_max_mm + lift + EPS
    if geom.shape == "BOX":
        return all(abs(p - c) <= h + EPS for p, c, h in zip(point, geom.center_mm, geom.half_extents_mm))
    lows = [min(v[k] for v in geom.vertices_mm) for k in range(3)]
    highs = [max(v[k] for v in geom.vertices_mm) for k in range(3)]
    return all(lo - EPS <= p <= hi + EPS for p, lo, hi in zip(point, lows, highs))


# --------------------------------------------------------------------------------------------------- every product


def test_every_catalog_product_is_drawn_within_its_published_body(drawings, catalog):
    kinds = set()
    for name, _, geom, drawing in drawings:
        kinds.add(geom.model_kind)
        assert [s.surface.surface_id for s in drawing.surfaces] == [s.surface_id for s in geom.surfaces], name
        for material, mesh in drawing.meshes():
            lift = AXIS_BAR_LIFT_MM if material == "polarization_axis" else 0.0  # the axis bar stands just out of both faces
            assert all(_inside_design_body(geom, p, lift) for p in mesh.positions), (name, material)
        for surface in drawing.surfaces:
            materials = {m for m, _ in surface.meshes}
            assert surface_material(surface.surface) in materials, (name, surface.surface.surface_id)
            assert UNDEFINED_MATERIAL not in materials and "profile_domain_limit_mm" not in surface.facts, name  # the catalog is clean
            assert surface.facts.get("chord_tolerance_met", True), (name, surface.surface.surface_id)
        if geom.model_kind == "SURFACE_SEQUENCE":
            every = [p for _, m in drawing.meshes() for p in m.positions]
            if geom.edge_radius_mm is not None:  # inside the published outline and out to it
                assert max(math.hypot(x, y) for x, y, _ in every) == pytest.approx(geom.edge_radius_mm, abs=FLOAT_EPS), name
            else:
                hx, hy = geom.edge_half_widths_mm
                assert (max(abs(x) for x, _, _ in every), max(abs(y) for _, y, _ in every)) == pytest.approx((hx, hy), abs=FLOAT_EPS), name
            body = {m for m, _ in drawing.body}
            assert ("optic_edge" in body) == (len(geom.surfaces) > 1 and any(m != "AIR" for m in geom.media)), name
            assert bool(body & {"mirror_substrate", "optic_substrate"}) == (len(geom.surfaces) == 1), name
    assert kinds == {m.model_kind for p in catalog.products for m in p.simulation_models}  # every model kind drawn


def test_every_drawn_vertex_lies_on_its_published_profile(drawings):
    """Every vertex of a surface's group lies on that surface's profile (float64, independently evaluated here)."""
    checked = Counter()
    for name, _, geom, drawing in drawings:
        for surface in drawing.surfaces:
            s = surface.surface
            for material, mesh in surface.meshes:
                for p in mesh.positions:
                    a, b, c = _in_surface(s, p)
                    z = _profile_z(s, a, b)
                    assert z is not None and abs(c - z) <= FLOAT_EPS, (name, s.surface_id, material, p)
            checked[geom.model_kind] += 1
    assert set(checked) == {"SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY"}


def _sequence_zone(surface: OpticSurfaceV1, geom: OpticBodyGeometryV1, material: str, x: float, y: float) -> bool:
    """Where a coaxial surface's zone may lie: its clear aperture, then the outline band, then the rim to the outline."""
    cx, cy = surface.aperture_center_mm or (0.0, 0.0)
    if surface.clear_radius_mm is not None:
        reach, limit = math.hypot(x - cx, y - cy), surface.clear_radius_mm
    else:
        hx, hy = surface.clear_half_widths_mm
        reach, limit = max(abs(x - cx) / hx, abs(y - cy) / hy), 1.0
    if material in ("aperture_outline", "optic_rim"):
        return reach >= limit * (1.0 - 1e-12) - FLOAT_EPS
    return reach <= limit + FLOAT_EPS


def test_the_clear_aperture_and_the_opaque_rim_are_drawn_where_the_tracer_puts_them(drawings):
    """A coaxial surface: its own material only inside its clear aperture; the outline band and the rim only outside it,
    present exactly when the clear aperture is smaller than the outline (the tracer blocks light there)."""
    for name, _, geom, drawing in drawings:
        if geom.model_kind == "SURFACE_SET":
            continue
        for surface in drawing.surfaces:
            s = surface.surface
            for material, mesh in surface.meshes:
                for x, y, _ in mesh.positions:
                    assert _sequence_zone(s, geom, material, x, y), (name, s.surface_id, material, (x, y))
            has_rim = {m for m, _ in surface.meshes} >= RIM_MATERIALS
            if geom.edge_radius_mm is not None and s.clear_radius_mm is not None:
                assert has_rim == (s.clear_radius_mm < geom.edge_radius_mm - 1e-9), (name, s.surface_id)


# ------------------------------------------------------------------------------------------------ chord tolerance


def _decimal(value) -> Decimal:
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(value)


def _published_surface(model, surface_id: str):
    if hasattr(model, "surfaces"):
        return next(s for s in model.surfaces if s.surface_id == surface_id)
    return None


def _steepness(surface: OpticSurfaceV1, reach: float) -> float:
    """sin of the profile slope at ``reach`` along local x (a cylinder: along its curved axis)."""
    if surface.is_plane:
        return 0.0
    x, y = (0.0, reach) if surface.cylinder_axis == "X" else (reach, 0.0)
    h = 1e-6 * max(reach, 1.0)
    z0, z1 = _profile_z(surface, x - (h if x else 0.0), y - (h if y else 0.0)), _profile_z(surface, x, y)
    if z0 is None or z1 is None:
        return 1.0
    slope = abs(z1 - z0) / h
    return slope / math.sqrt(1.0 + slope * slope)


def _reach(surface: OpticSurfaceV1, geom: OpticBodyGeometryV1) -> float:
    if geom.edge_radius_mm is not None:
        return geom.edge_radius_mm
    if geom.edge_half_widths_mm is not None:
        return math.hypot(*geom.edge_half_widths_mm)
    return surface.aperture_extent_mm()


def _kind(geom: OpticBodyGeometryV1, surface: OpticSurfaceV1) -> tuple:
    profile = "plane" if surface.is_plane else "sphere" if surface.is_spherical else "cylinder" if surface.cylinder_axis else "conic/asphere"
    outline = "round" if geom.edge_radius_mm is not None else "rect" if geom.edge_half_widths_mm is not None else "solid" if geom.solid else "free"
    aperture = ("circle" if surface.clear_radius_mm is not None else "rect") + ("-decentred" if surface.aperture_center_mm else "")
    return geom.model_kind, outline, aperture, profile


def _deviation(model, geom, surface_drawing) -> float:
    """The largest distance (mm) between the drawn triangles and the published surface, at every triangle's centroid and
    edge midpoints, in 40-digit arithmetic: a sphere's exact distance to its centre minus its radius, any other profile
    the axial gap times the cosine of its slope."""
    s = surface_drawing.surface
    published = _published_surface(model, s.surface_id)
    worst = 0.0
    with localcontext() as ctx:
        ctx.prec = REFERENCE_DIGITS
        radius = _decimal(published.radius_mm) if published is not None and published.radius_mm is not None else None
        for material, mesh in surface_drawing.meshes:
            local = [tuple(_decimal(c) for c in _in_surface(s, p)) for p in mesh.positions]
            idx = mesh.indices
            for t in range(0, len(idx), 3):
                a, b, c = local[idx[t]], local[idx[t + 1]], local[idx[t + 2]]
                three = Decimal(3)
                points = (tuple((a[k] + b[k] + c[k]) / three for k in range(3)),
                          tuple((a[k] + b[k]) / 2 for k in range(3)), tuple((b[k] + c[k]) / 2 for k in range(3)), tuple((c[k] + a[k]) / 2 for k in range(3)))
                for x, y, z in points:
                    if published is not None and s.is_spherical:
                        d = abs((x * x + y * y + (z - radius) ** 2).sqrt() - abs(radius))
                    else:
                        rho_sq = y * y if s.cylinder_axis == "X" else x * x if s.cylinder_axis == "Y" else x * x + y * y
                        exact = published.profile_sag(rho_sq) if published is not None else Decimal(0)
                        fx, fy = float(x), float(y)
                        h = 1e-7
                        zx = (_profile_z(s, fx + h, fy) - _profile_z(s, fx - h, fy)) / (2 * h)
                        zy = (_profile_z(s, fx, fy + h) - _profile_z(s, fx, fy - h)) / (2 * h)
                        d = abs(z - exact) / Decimal(repr(math.sqrt(1.0 + zx * zx + zy * zy)))
                    worst = max(worst, float(d))
    return worst


def test_the_steepest_surface_of_every_kind_is_within_the_chord_tolerance(drawings):
    """Per kind of surface (model, outline, aperture, profile): the steepest one in the catalog, against its published
    decimals."""
    steepest: dict[tuple, tuple] = {}
    for name, model, geom, drawing in drawings:
        if geom.model_kind == "POLARIZATION_ELEMENT":
            continue
        for surface in drawing.surfaces:
            if surface.surface.is_plane and geom.model_kind == "SURFACE_SET":
                continue  # flat polygons of a solid: exact by construction
            key = _kind(geom, surface.surface)
            score = (_steepness(surface.surface, _reach(surface.surface, geom)), name, surface.surface.surface_id)
            if key not in steepest or score > steepest[key][0]:
                steepest[key] = (score, model, geom, surface)
    assert {k[3] for k in steepest} >= {"plane", "sphere", "cylinder", "conic/asphere"}
    for key, (score, model, geom, surface) in sorted(steepest.items()):
        deviation = _deviation(model, geom, surface)
        assert deviation <= CHORD_TOLERANCE_MM * (1.0 + 1e-6), (key, score[1:], deviation)


def test_a_near_hemispherical_lens_is_drawn_within_the_tolerance(drawings):
    """The catalog's steepest sphere (the case the fixed 12 x 48 tessellation drew 0.43 mm off its sphere): every
    triangle within the tolerance of the exact sphere."""
    best = max(((abs(s.surface.radius_mm) / geom.edge_radius_mm, name, model, geom, s) for name, model, geom, d in drawings
                if geom.model_kind == "SURFACE_SEQUENCE" and geom.edge_radius_mm is not None for s in d.surfaces if s.surface.is_spherical),
               key=lambda item: (-item[0], item[1]))
    ratio, name, model, geom, surface = best
    assert ratio < 1.05, name  # it is steep: the sphere's edge slope exceeds 70 degrees
    assert _deviation(model, geom, surface) <= CHORD_TOLERANCE_MM * (1.0 + 1e-6), name


# ---------------------------------------------------------------------------------------- true scale, watertight


def _shallow_lens(radius_mm: float) -> OpticBodyGeometryV1:
    return OpticBodyGeometryV1(diameter_mm=25.4, z_min_mm=-0.2, z_max_mm=3.2, category="LENS", form="BICONVEX_LENS", model_kind="SURFACE_SEQUENCE",
                               fidelity="SURFACE_PRESCRIPTION", edge_radius_mm=12.7,
                               surfaces=(OpticSurfaceV1(surface_id="S1", role="REFRACT", vertex_z_mm=0.0, radius_mm=radius_mm, clear_radius_mm=11.43),
                                         OpticSurfaceV1(surface_id="S2", role="REFRACT", vertex_z_mm=3.0, radius_mm=-radius_mm, clear_radius_mm=11.43)),
                               media=("N-BK7",))


def test_a_shallow_lens_is_drawn_at_true_scale(catalog):
    """R = 1000 mm over a 12.7 mm edge radius: a sag of 80.7 um. The GLB (float32, what every viewer draws) holds exactly
    that sag at the rim and the published centre and edge thicknesses -- nothing is magnified."""
    radius = 1000.0
    geom = _shallow_lens(radius)
    drawing = draw_body(geom)
    sag = radius - math.sqrt(radius * radius - 12.7 * 12.7)
    s1, s2 = drawing.surfaces
    rim1 = [p for _, m in s1.meshes for p in m.positions if math.hypot(p[0], p[1]) > 12.7 - 1e-9]
    rim2 = [p for _, m in s2.meshes for p in m.positions if math.hypot(p[0], p[1]) > 12.7 - 1e-9]
    assert rim1 and rim2
    assert all(p[2] == pytest.approx(sag, abs=1e-12) for p in rim1)
    assert all(p[2] == pytest.approx(3.0 - sag, abs=1e-12) for p in rim2)
    centre1 = [p for _, m in s1.meshes for p in m.positions if p[:2] == (0.0, 0.0)]
    centre2 = [p for _, m in s2.meshes for p in m.positions if p[:2] == (0.0, 0.0)]
    assert centre2[0][2] - centre1[0][2] == 3.0  # the published centre thickness
    assert rim2[0][2] - rim1[0][2] == pytest.approx(3.0 - 2 * sag, abs=1e-12)  # edge thickness = CT - 2 sag
    # through the GLB bytes
    scene = single_element_scene(catalog.catalog_version, SceneElementV1(
        element_id="E", optical_role="lens", optical_sku=next(p.optical_sku for p in catalog.products if p.category.value == "LENS"),
        label="lens", pose={"position_mm": (0.0, 0.0, 0.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0)}, geometry=geom))
    gltf, binary = parse_glb(scene_glb_bytes(scene))
    (s1_node,) = [n for n in gltf["nodes"] if n.get("name") == "surface:E/S1"]
    points = [p for prim in gltf["meshes"][s1_node["mesh"]]["primitives"] for p in read_accessor(gltf, binary, prim["attributes"]["POSITION"])]
    drawn_sag = max(p[2] for p in points)
    assert drawn_sag == pytest.approx(sag, rel=1e-6)  # float32 storage, no scale factor
    assert max(math.hypot(p[0], p[1]) for p in points) == pytest.approx(12.7, rel=1e-6)


def _edge_uses(mesh) -> Counter:
    uses: Counter = Counter()
    for t in range(0, len(mesh.indices), 3):
        tri = [mesh.positions[i] for i in mesh.indices[t:t + 3]]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            uses[(a, b) if a <= b else (b, a)] += 1
    return uses


def test_surface_zones_are_watertight_and_wind_about_their_normals(drawings):
    """Per zone (one material of one surface): every edge is used by at most two triangles, the edges used once form the
    zone's boundary (its inner and outer loops: no crack where a row of fewer directions meets a denser one), and every
    triangle winds counter-clockwise about the drawn normals of its vertices."""
    seen = set()
    for name, _, geom, drawing in drawings[:: max(1, len(drawings) // 400)]:
        for surface in drawing.surfaces:
            for material, mesh in surface.meshes:
                uses = _edge_uses(mesh)
                assert max(uses.values()) <= 2, (name, material)
                boundary = [e for e, n in uses.items() if n == 1]
                degree = Counter(p for e in boundary for p in e)
                assert all(n == 2 for n in degree.values()), (name, surface.surface.surface_id, material)  # closed loops only
                for t in range(0, len(mesh.indices), 3):
                    i, j, k = mesh.indices[t:t + 3]
                    a, b, c = mesh.positions[i], mesh.positions[j], mesh.positions[k]
                    n = g.cross(g.sub(b, a), g.sub(c, a))
                    assert g.dot(n, g.add(g.add(mesh.normals[i], mesh.normals[j]), mesh.normals[k])) > 0.0, (name, material)
                seen.add(geom.model_kind)
    assert seen == {"SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY"}


def test_circle_points_use_no_platform_trigonometry():
    for n in (6, 8, 12, 48, 184, 192, 977):
        points = g.unit_circle(n)
        for k, (c, s) in enumerate(points):
            assert c == pytest.approx(math.cos(2 * math.pi * k / n), abs=2e-15) and s == pytest.approx(math.sin(2 * math.pi * k / n), abs=2e-15)
        if n % 4 == 0:  # quarter turns are exact and the circle keeps its symmetry bit for bit
            assert points[0] == (1.0, 0.0) and points[n // 4] == (0.0, 1.0) and points[n // 2] == (-1.0, 0.0) and points[3 * n // 4] == (0.0, -1.0)
            for k in range(n // 4):
                c, s = points[k]
                assert points[k + n // 4] == (-s, c)
    assert g.circle_segments(12.7, CHORD_TOLERANCE_MM) % 8 == 0
    n = g.circle_segments(12.7, CHORD_TOLERANCE_MM)
    assert 12.7 * (1.0 - math.cos(math.pi / n)) <= CHORD_TOLERANCE_MM


# --------------------------------------------------------------------------------------------------------- GLB


def _representatives(catalog) -> list[tuple[str, OpticalSimulationSceneV1]]:
    """One product per (category, form, model kind, outline): every way a body is drawn, in a bounded test time."""
    chosen: dict[tuple, tuple[str, OpticalSimulationSceneV1]] = {}
    for name, scene in product_scenes(catalog):
        geom = scene.elements[0].geometry
        key = (geom.category.value, geom.form.value, geom.model_kind, geom.edge_radius_mm is None, len(geom.surfaces))
        chosen.setdefault(key, (name, scene))
    return [chosen[k] for k in sorted(chosen)]


@pytest.fixture(scope="module")
def representatives(catalog):
    return _representatives(catalog)


def test_every_surface_is_a_named_glb_node_with_its_published_facts(representatives):
    for name, scene in representatives:
        (element,) = scene.elements
        geom = element.geometry
        gltf, _ = parse_glb(scene_glb_bytes(scene))
        root = gltf["nodes"][gltf["scenes"][0]["nodes"][0]]
        assert root["extras"]["mesh_chord_tolerance_mm"] == CHORD_TOLERANCE_MM
        (node,) = [n for n in gltf["nodes"] if n.get("name") == "element:E"]
        children = [gltf["nodes"][c] for c in node["children"]]
        assert [c["name"] for c in children] == [f"surface:E/{s.surface_id}" for s in geom.surfaces], name
        for child, surface in zip(children, geom.surfaces, strict=True):
            extras = child["extras"]
            assert extras["kind"] == "OPTICAL_SURFACE" and extras["element_id"] == "E" and extras["surface_id"] == surface.surface_id
            assert extras["role"] == surface.role.value and extras["chord_tolerance_mm"] == CHORD_TOLERANCE_MM
            assert extras.get("radius_mm") == surface.radius_mm and extras.get("conic") == surface.conic
            assert extras.get("clear_radius_mm") == surface.clear_radius_mm
            if geom.model_kind == "SURFACE_SEQUENCE":
                k = int(surface.surface_id[1:]) - 1
                assert extras["medium_before"] == ("AIR" if k == 0 else geom.media[k - 1])
                assert extras["medium_after"] == ("AIR" if k == len(geom.surfaces) - 1 else geom.media[k])
            assert "mesh" in child, name


def _all_bytes_digest(scenes) -> str:
    h = hashlib.sha256()
    for name, scene in scenes:
        h.update(name.encode())
        h.update(scene_glb_bytes(scene))
    return h.hexdigest()


_CHILD = """
import sys
sys.path.insert(0, {tests!r})
from test_optical_view_glb import _all_bytes_digest, _representatives, bundled_catalog
print(_all_bytes_digest(_representatives(bundled_catalog())))
"""


def test_glbs_are_byte_identical_in_process_and_across_processes(representatives):
    """One product of every drawing kind (the whole catalog's drawing is checked above; drawing is pure arithmetic, so a
    representative of every code path is what a second interpreter can disagree on)."""
    from pathlib import Path

    expected = _all_bytes_digest(representatives)
    assert _all_bytes_digest(representatives) == expected
    for seed in ("0", "4242"):
        out = subprocess.run([sys.executable, "-c", _CHILD.format(tests=str(Path(__file__).resolve().parent))], capture_output=True, text=True,
                             env={**os.environ, "PYTHONHASHSEED": seed}, timeout=600, check=True)
        assert out.stdout.strip() == expected


def test_profile_geometry_helpers():
    # a convex sphere (centre on +Z) sags toward +Z, a concave one toward -Z; a plane has no sag
    assert g.sag(50.0, 10.0) == pytest.approx(50.0 - math.sqrt(2400.0)) and g.sag(-50.0, 10.0) == pytest.approx(-g.sag(50.0, 10.0))
    assert g.sag(None, 10.0) == 0.0
    assert g.surface_rim(0.0, 50.0, 10.0, 12.7) == (12.7, g.sag(50.0, 12.7))
    assert g.side_wall(12.7, 3.0, 1.0).empty
    box = g.box(-1.0, 1.0, -0.5, 0.5, 0.0, 2.0)
    assert len(box.positions) == 24 and len(box.indices) == 36
    for k in range(0, 36, 3):  # every face winds counter-clockwise around its outward normal
        a, b, c = (box.positions[i] for i in box.indices[k:k + 3])
        assert g.dot(g.cross(g.sub(b, a), g.sub(c, a)), box.normals[box.indices[k]]) > 0.0
