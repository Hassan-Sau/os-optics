"""The Optical Simulation View draws every schema-3.0 construct: cube splitters, prisms, a wedge prism and a wedged
sampler (convex solids, a round plate's side wall), a rectangular window and a cylindrical lens (profiles over a
rectangular outline), an off-axis paraboloid (a decentred sag patch) and a typical-spectrum filter; every drawn point lies
inside the product's public design geometry, and the scene round-trips through its contract."""

from __future__ import annotations

import math

import pytest

from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from os_optics._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from optical_view_reference import element_primitives
from os_optics.optical_view import parse_glb, read_accessor, scene_glb_bytes

from catalog3_products import BANDPASS, CUBE, CYLINDER, DOVE, OAP, PENTA, RECT_WINDOW, RIGHT_ANGLE, SAMPLER, WEDGE, catalog3_document
from test_optical_view_glb import product_scenes

EPS = 1e-4  # float32 storage of mm coordinates


@pytest.fixture(scope="module")
def scenes():
    catalog = PublicOpticalCatalogV2.model_validate(catalog3_document())
    return {name.split(":")[0]: scene for name, scene in product_scenes(catalog)}


def _primitives(scene) -> dict[str, list[tuple[float, float, float]]]:
    gltf, binary = parse_glb(scene_glb_bytes(scene))
    return element_primitives(gltf, binary, "E")


RIM_MATERIALS = {"aperture_outline", "optic_rim"}


def _inside_design_body(geom, point) -> bool:
    x, y, z = point
    if geom.shape == "CYLINDER":
        return math.hypot(x, y) <= geom.diameter_mm / 2 + EPS and geom.z_min_mm - EPS <= z <= geom.z_max_mm + EPS
    if geom.shape == "BOX":
        return all(abs(p - c) <= h + EPS for p, c, h in zip(point, geom.center_mm, geom.half_extents_mm))
    lows = [min(v[k] for v in geom.vertices_mm) for k in range(3)]
    highs = [max(v[k] for v in geom.vertices_mm) for k in range(3)]
    return all(lo - EPS <= p <= hi + EPS for p, lo, hi in zip(point, lows, highs))


def test_every_schema3_product_is_drawn_inside_its_design_body(scenes):
    assert set(scenes) == {CUBE, RIGHT_ANGLE, PENTA, DOVE, WEDGE, SAMPLER, RECT_WINDOW, OAP, CYLINDER, BANDPASS, "OS-OPT-TEST-PBS-CUBE-A12P7"}
    for sku, scene in scenes.items():
        assert OpticalSimulationSceneV1.model_validate(scene.payload()) == scene, sku
        (element,) = scene.elements
        prims = _primitives(scene)
        assert prims, sku
        for material, points in prims.items():
            assert all(_inside_design_body(element.geometry, p) for p in points), (sku, material)


@pytest.mark.parametrize("sku, materials", [
    (CUBE, {"optic_glass", "splitter_face", "optic_edge"}),  # AR faces, the coated diagonal, the untraced faces
    (PENTA, {"optic_glass", "mirror_face", "optic_edge"}),
    (RIGHT_ANGLE, {"optic_glass", "optic_edge"}),
    (WEDGE, {"optic_glass", "optic_edge"}),  # the two faces and the round side wall
    (SAMPLER, {"splitter_face", "optic_glass", "optic_edge"}),
    (RECT_WINDOW, {"optic_glass", "optic_edge"}),
    (CYLINDER, {"optic_glass", "optic_edge"}),
    (OAP, {"mirror_face"}),
    (BANDPASS, {"filter_face", "optic_substrate"}),
])
def test_bodies_are_drawn_with_the_materials_of_their_surfaces(scenes, sku: str, materials: set[str]):
    # besides the surfaces' own materials: the clear-aperture outline band and the opaque rim of every surface whose clear
    # aperture is smaller than its outline (a free-standing off-axis mirror face is only its clear aperture)
    drawn = set(_primitives(scenes[sku]))
    assert drawn - RIM_MATERIALS == materials
    assert (drawn & RIM_MATERIALS == RIM_MATERIALS) == (sku != OAP)


def test_the_cylindrical_lens_face_follows_its_cylinder_and_the_window_its_rectangle(scenes):
    from catalog3_products import CYLINDER_RADIUS_MM

    points = _primitives(scenes[CYLINDER])["optic_glass"]
    front = [p for p in points if p[2] < 2.0]
    r = CYLINDER_RADIUS_MM
    for x, y, z in front:  # curved along local X, straight along Y
        assert z == pytest.approx(r - math.sqrt(r * r - x * x), abs=EPS)
    window = [p for name, points in _primitives(scenes[RECT_WINDOW]).items() if name in {"optic_glass", *RIM_MATERIALS} for p in points]
    assert max(abs(p[0]) for p in window) == pytest.approx(10.0, abs=EPS) and max(abs(p[1]) for p in window) == pytest.approx(6.0, abs=EPS)


def test_the_paraboloid_patch_is_its_decentred_clear_aperture(scenes):
    from catalog3_products import PARENT_FOCAL_MM

    points = _primitives(scenes[OAP])["mirror_face"]
    fp = PARENT_FOCAL_MM
    for x, y, z in points:  # on the paraboloid z = (a^2 + b^2) / (4 fp) about the vertex (-2 fp, 0, -fp)
        a, b = x + 2 * fp, y
        assert z + fp == pytest.approx((a * a + b * b) / (4 * fp), abs=1e-3)
        assert math.hypot(a - 2 * fp, b) <= 11.43 + EPS
