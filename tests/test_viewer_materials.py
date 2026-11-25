"""The viewer's material model (viewer/shading.py, docs/VIEWER_MATERIALS.md) and its camera presets.

Independent references: the Fresnel reflectance at normal incidence and the Beer-Lambert transmission of a glass are written out
here from their closed forms; the refractive index in the GLB is evaluated here from the Sellmeier coefficients of the prescription
fixture and of SCHOTT F2 as published; the thickness from the fixture's radii, thicknesses and semi-diameters. The retained
local browser page's camera is checked against the Python snapshot pose in ``test_viewer_browser_pose.py``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from catalog_store_helpers import REAL_CATALOG
from os_optics.local import load_public_catalog, run_optical_project
from os_optics.optical_view import parse_glb, scene_glb_bytes
from os_optics.optical_view.colors import wavelength_display_rgb
from os_optics.optical_view.glb import MATERIAL_FINISHES, STATIC_MATERIALS
from os_optics.prescription import import_prescription, merge_into_project
from os_optics.viewer import optical_view_scene, view_document
from os_optics.viewer.model import PRESET_ORDER, ViewBeam, ViewerScene, camera_presets, main_beam_axis
from os_optics.viewer.outline import outline_groups
from os_optics.viewer.palette import IDENTITY_MATERIALS
from os_optics.viewer.raster import _camera_normals, make_camera, parts_silhouette, render_view
from os_optics.viewer.shading import (
    BEAM_CASING_RGB,
    BEHIND_DIM,
    ENV_FLOOR,
    LIGHTS,
    OPTICAL_BACKDROP,
    OPTICAL_VIEW,
    beam_needs_casing,
    contrast,
    environment,
    fresnel_f0,
    schlick,
    shade,
    light_frames,
    shading_document,
    srgb_to_linear,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prescriptions"
SEED = load_public_catalog(REAL_CATALOG.read_bytes())
IDENTITY = [1.0, 0.0, 0.0, 0.0]
FACING = np.array([[0.0, 0.0, 1.0]])
SELLMEIER = {  # the fixture's XSK16 and SCHOTT F2 as published in the seed catalog
    "XSK16": ((1.34317774, 0.241144399, 0.994317969), (0.00704687339, 0.0229005, 92.7508526)),
    "F2": ((1.34533359, 0.209073176, 0.937357162), (0.00997743871, 0.0470450767, 111.886764)),
}


def sellmeier(name: str, wavelength_nm: float) -> float:
    b, c = SELLMEIER[name]
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def cooke_run(tmp_path_factory, direction="+Z"):
    base = {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Laser."},
        "nominal_pose": {"position_mm": [0.0, 0.0, -30.0], "quaternion_wxyz": IDENTITY},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 587.56, "power_w": 0.001, "waist_radius_mm": 3.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "image", "pose": {"position_mm": [0.0, 0.0, 62.2], "quaternion_wxyz": IDENTITY},
                          "aperture": {"shape": "circle", "radius_mm": 10.0}}]}
    imported = import_prescription((FIXTURES / "cooke_triplet.csv").read_text(encoding="utf-8"), catalog=SEED, model_id="cooke")
    project = merge_into_project(base, imported, instance_id="OBJ", pose={"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": IDENTITY})
    directory = tmp_path_factory.mktemp("cooke")
    run_optical_project(project, catalog=REAL_CATALOG, output_dir=directory)
    return directory


@pytest.fixture(scope="module")
def cooke(tmp_path_factory) -> Path:
    return cooke_run(tmp_path_factory)


# ---------------------------------------------------------------------------------------------------- the equations


def test_fresnel_at_normal_incidence_is_the_closed_form_and_goes_to_one_at_grazing():
    n = 1.5168
    assert fresnel_f0(np.array(n)) == pytest.approx(0.04216, abs=1e-4)  # (0.5168 / 2.5168) ** 2 by hand
    assert schlick(fresnel_f0(np.array(n)), np.array(1.0)) == pytest.approx(((n - 1) / (n + 1)) ** 2, rel=1e-12)
    assert schlick(fresnel_f0(np.array(n)), np.array(0.0)) == pytest.approx(1.0, rel=1e-12)
    # a denser glass reflects more
    assert fresnel_f0(np.array(1.8)) > fresnel_f0(np.array(1.5))


def test_the_environment_has_no_edge_and_lights_a_dome_from_above_and_below_differently():
    # along an arc through every light, in 1 degree steps, the radiance never jumps: a reflection rolls, it does not draw a shape
    for box in light_frames():
        centre = box["centre"]
        side = np.cross(centre, [0.0, 1.0, 0.0])
        side = side / np.linalg.norm(side)
        angles = np.radians(np.arange(-90.0, 91.0))
        directions = np.outer(np.cos(angles), centre) + np.outer(np.sin(angles), side)
        radiance = environment(directions, np.zeros(len(directions)))[:, 0]
        assert np.abs(np.diff(radiance)).max() < 0.1 * radiance.max()  # an edge would be the whole step within a degree
        assert radiance[90] >= box["intensity"]  # its middle: the full light
    # the sky above the horizon, the floor below it, on the viewer's side of the scene
    sky = environment(np.array([[0.0, 0.8, 0.6]]), np.zeros(1))[0]
    floor = environment(np.array([[0.0, -0.8, 0.6]]), np.zeros(1))[0]
    assert sky[0] > 10.0 * floor[0] and floor == pytest.approx(np.asarray(ENV_FLOOR) * (BEHIND_DIM + (1.0 - BEHIND_DIM) * 0.8), abs=1e-3)
    # and behind the scene the sky is dim: a surface seen edge-on is a dark mirror
    high = environment(np.array([[0.0, 0.9, 0.436]]), np.zeros(1))[0][0]
    assert environment(np.array([[0.0, 0.9, -0.436]]), np.zeros(1))[0][0] < 0.6 * high
    # a rough surface spreads a light: its peak falls
    key = light_frames()[0]["centre"][None, :]
    assert environment(key, np.array([0.5]))[0][0] < environment(key, np.zeros(1))[0][0]


def test_glass_transmits_the_fresnel_and_beer_lambert_share_of_what_lies_behind():
    """Facing the viewer: ``a = 1 - (1 - F0) * exp(-sigma * t)``, ``sigma`` from the material's alpha over the reference thickness."""

    from os_optics.viewer.shading import GLASS_ABSORPTION, REFERENCE_THICKNESS_MM

    n, alpha, t = 1.62, 0.35, 8.0
    src, a = shade(FACING, (0.62, 0.8, 0.95), alpha=alpha, metallic=0.0, roughness=0.04, transmission=1.0, ior=n, thickness_mm=t)
    f0 = ((n - 1) / (n + 1)) ** 2
    through = (1.0 - f0) * (1.0 - alpha * GLASS_ABSORPTION) ** (t / REFERENCE_THICKNESS_MM)
    assert a[0] == pytest.approx(1.0 - through * (1.0 - src[0].max()), rel=1e-12)  # what it shows covers too
    # thicker glass absorbs more; a higher index reflects more; both leave less of what lies behind
    _, thicker = shade(FACING, (0.62, 0.8, 0.95), alpha=alpha, metallic=0.0, roughness=0.04, transmission=1.0, ior=n, thickness_mm=2 * t)
    _, denser = shade(FACING, (0.62, 0.8, 0.95), alpha=alpha, metallic=0.0, roughness=0.04, transmission=1.0, ior=1.9, thickness_mm=t)
    assert thicker[0] > a[0] and denser[0] > a[0]


def test_a_curved_surface_shows_a_highlight_where_a_flat_one_shows_one_value():
    """A spherical cap's normals sweep across the studio; a plane has one normal. This is the cue that a Lambert shade lacks."""

    grid = np.linspace(-0.6, 0.6, 81)
    x, y = np.meshgrid(grid, grid)
    keep = x * x + y * y < 0.36
    normals = np.stack([x[keep], y[keep], np.sqrt(1.0 - x[keep] ** 2 - y[keep] ** 2)], axis=-1)
    finish = dict(alpha=0.35, metallic=0.0, roughness=0.04, transmission=1.0, ior=1.5168, thickness_mm=4.0)
    curved, _ = shade(normals, (0.62, 0.8, 0.95), **finish)
    flat, _ = shade(np.repeat(FACING, 4, axis=0), (0.62, 0.8, 0.95), **finish)
    assert np.ptp(flat, axis=0).max() == 0.0
    assert float(np.percentile(curved.mean(axis=1), 99.5) - np.percentile(curved.mean(axis=1), 50)) > 0.15


def test_a_coated_face_shows_its_identity_colour_whatever_its_thickness(cooke):
    finish = dict(alpha=0.4, metallic=0.0, roughness=0.04, transmission=1.0, ior=None, thickness_mm=None)
    green = (0.45, 0.85, 0.55)
    clear, _ = shade(FACING, green, **finish)
    coated, _ = shade(FACING, green, **finish, tint=0.22)
    assert coated[0][1] > clear[0][1] + 0.1 and coated[0][1] > coated[0][0] and coated[0][1] > coated[0][2]  # it reads green
    thick, _ = shade(FACING, green, **{**finish, "thickness_mm": 20.0}, tint=0.22)
    assert thick == pytest.approx(coated)


def test_a_metal_reflects_in_its_own_colour_and_either_side_of_a_surface_shades_alike():
    src, a = shade(FACING, (0.74, 0.79, 0.88), alpha=1.0, metallic=1.0, roughness=0.05, transmission=0.0, ior=None, thickness_mm=None)
    assert a[0] == 1.0 and src[0][2] >= src[0][0]  # bluish like its colour, opaque
    tilted = np.array([[0.3, 0.2, math.sqrt(1 - 0.13)]])
    front, _ = shade(tilted, (0.6, 0.8, 0.95), alpha=0.35, metallic=0.0, roughness=0.04, transmission=1.0, ior=1.5, thickness_mm=2.0)
    back, _ = shade(-tilted, (0.6, 0.8, 0.95), alpha=0.35, metallic=0.0, roughness=0.04, transmission=1.0, ior=1.5, thickness_mm=2.0)
    assert front == pytest.approx(back)


# ------------------------------------------------------------------------------------- the GLB carries the finish


def test_every_static_material_shown_as_an_optic_has_a_finish():
    assert set(IDENTITY_MATERIALS) - {"stop_plate", "mirror_face"} <= {k for k, f in MATERIAL_FINISHES.items() if f.transmission > 0.0}
    assert all(MATERIAL_FINISHES[k].transmission > 0.0 for k in ("optic_rim", "optic_edge", "aperture_outline"))  # no opaque rings on a lens
    assert MATERIAL_FINISHES["mirror_face"].metallic == 1.0
    assert set(MATERIAL_FINISHES) <= set(STATIC_MATERIALS)


def test_the_glb_states_the_index_and_thickness_of_the_models_own_glass(cooke):
    gltf, _ = parse_glb((cooke / "optical_scene.glb").read_bytes())
    assert {"KHR_materials_ior", "KHR_materials_transmission", "KHR_materials_volume"} == set(gltf["extensionsUsed"])
    glass = [m for m in gltf["materials"] if m["name"] == "optic_glass"]
    indices = sorted({m["extensions"]["KHR_materials_ior"]["ior"] for m in glass})
    assert indices == pytest.approx(sorted((sellmeier("XSK16", 587.56), sellmeier("F2", 587.56))), abs=1e-4)
    assert all(m["extensions"]["KHR_materials_transmission"]["transmissionFactor"] == 1.0 for m in glass)
    # L2, the flint: vertices 0.99997 mm apart, R -22.21328 and +20.29192, semi-diameter 7.5 mm: edge to edge, from the fixture's numbers
    sag = lambda r, s=7.5: abs(r) - math.sqrt(r * r - s * s)  # noqa: E731
    flint = [m for m in glass if m["extensions"]["KHR_materials_ior"]["ior"] == pytest.approx(sellmeier("F2", 587.56), abs=1e-4)]
    assert [m["extensions"]["KHR_materials_volume"]["thicknessFactor"] for m in flint] == pytest.approx([0.99997 + sag(22.21328) + sag(20.29192)] * len(flint), abs=2e-3)


def test_the_scene_states_its_media_and_a_glb_of_another_glass_differs_only_there(cooke):
    scene = optical_view_scene(cooke).source_scene
    media = {m.medium_id: m for m in scene.display_media}
    assert media["project:F2"].refractive_index == pytest.approx(sellmeier("F2", 587.56), abs=1e-4)
    assert media["project:XSK16"].at_wavelength_nm == 587.56
    gltf, _ = parse_glb(scene_glb_bytes(scene.model_copy(update={"display_media": ()})))
    unknown = [m for m in gltf["materials"] if m["name"] == "optic_glass"]
    assert unknown and all("KHR_materials_ior" not in m["extensions"] for m in unknown)  # glTF's default then applies


def test_the_view_document_hands_the_page_the_glb_finish_and_the_models_numbers(cooke):
    scene = optical_view_scene(cooke)
    doc = view_document(scene)
    gltf, _ = parse_glb((cooke / "optical_scene.glb").read_bytes())
    assert doc["shading"] == shading_document(OPTICAL_VIEW) and len(doc["shading"]["lights"]) == len(LIGHTS)
    meshes = [m for part in doc["parts"] for m in part["meshes"]]
    glass = [m for m in meshes if m["material"] == "optic_glass"]
    assert glass and all(m["finish"]["transmission"] == 1.0 and m["finish"]["studio_lit"] and m["normals"] for m in glass)
    finished = {(m["pbrMetallicRoughness"]["metallicFactor"], m["pbrMetallicRoughness"]["roughnessFactor"]) for m in gltf["materials"] if m["name"] == "optic_rim"}
    assert {(m["finish"]["metallic"], m["finish"]["roughness"]) for m in meshes if m["material"] == "optic_rim"} == finished
    plane = next(m for m in meshes if m["material"] == "plane_observation")
    assert not plane["finish"]["studio_lit"]


def test_normals_are_carried_rotated_with_their_mesh(cooke):
    scene = optical_view_scene(cooke)
    mesh = next(m for p in scene.parts for m in p.meshes if m.material == "optic_glass")
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5) and len(mesh.normals) == len(mesh.positions)
    # a surface of a lens along +Z: normals around the axis, tilted by at most the surface's edge slope
    assert np.abs(mesh.normals[:, 2]).min() > 0.8


# --------------------------------------------------------------------------------------------------------- pictures


def test_a_lens_surface_in_the_iso_view_shows_far_more_tone_than_a_lambert_shade_of_it(cooke):
    """Over the vertices of the first lens surface, seen as `iso` sees it: the spread of the displayed value, against the head-light
    Lambert shade (0.58 + 0.42 |n.l|, alpha 0.35 over the backdrop) that this model replaced."""

    scene = optical_view_scene(cooke)
    mesh = next(m for p in scene.parts for m in p.meshes if m.material == "optic_glass")
    cam = make_camera(scene, "iso", 960, 720)
    normals = _camera_normals(mesh.normals, cam)
    backdrop = np.mean(OPTICAL_BACKDROP, axis=0)
    src, a = shade(normals, mesh.rgb, alpha=mesh.alpha, metallic=mesh.metallic, roughness=mesh.roughness, transmission=mesh.transmission,
                   ior=mesh.ior, thickness_mm=mesh.thickness_mm)
    lit = (src + backdrop[None, :] * (1.0 - a)[:, None]).mean(axis=1)
    light = np.array([-0.3, 0.55, 1.0]) / np.linalg.norm([-0.3, 0.55, 1.0])
    lambert = ((np.asarray(mesh.rgb) * 0.35 + backdrop * 0.65)[None, :] * (0.58 + 0.42 * np.abs(normals @ light))[:, None]).mean(axis=1)
    spread, old = (float(np.percentile(v, 99) - np.percentile(v, 1)) for v in (lit, lambert))
    assert spread > 3.0 * old and spread > 0.2


def test_a_surfaces_zones_have_one_outline_its_real_edge_and_its_silhouette(cooke):
    """The seams between clear aperture, outline band and rim are not edges of the glass: a lens surface's always-drawn edges lie on
    its outer edge only. Seen from the side, its silhouette is drawn too."""

    scene = optical_view_scene(cooke)
    lens = next(p for p in scene.parts if p.part_id == "OBJ")
    group = next(g for g in outline_groups(lens, merge_surfaces=True) if g.surface_id == "L1-S1")
    radii = np.hypot(group.points[group.feature][..., 0], group.points[group.feature][..., 1])
    assert radii.min() > 0.99 * radii.max() and radii.max() == pytest.approx(10.0, abs=1e-3)  # mech semi-diameter 10, clear 9
    facing = np.einsum("ijk,k->ij", group.sides, np.array([0.0, 1.0, 0.0]))
    silhouette = np.sign(facing[:, 0]) != np.sign(facing[:, 1])
    assert silhouette.sum() > 10  # a dome seen from the side has its curved outline
    doc = view_document(scene)
    part = next(p for p in doc["parts"] if p["id"] == "OBJ")
    assert {o.get("surface") for o in part["outlines"]} >= {"L1-S1", None} and doc["camera_fov_deg"] < 20.0


def test_a_cemented_interface_shows_hardly_at_all_and_reflects_by_the_two_glasses_ratio(tmp_path):
    text = (FIXTURES / "lens_cemented_doublet.csv").read_text(encoding="utf-8")
    imported = import_prescription(text, catalog=SEED, model_id="d")
    base = {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Laser."},
        "nominal_pose": {"position_mm": [0.0, 0.0, -30.0], "quaternion_wxyz": IDENTITY},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 587.56, "power_w": 0.001, "waist_radius_mm": 3.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "image", "pose": {"position_mm": [0.0, 0.0, 90.0], "quaternion_wxyz": IDENTITY},
                          "aperture": {"shape": "circle", "radius_mm": 10.0}}]}
    project = merge_into_project(base, imported, instance_id="OBJ", pose={"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": IDENTITY})
    run_optical_project(project, catalog=REAL_CATALOG, output_dir=tmp_path)
    scene = optical_view_scene(tmp_path)
    media = {m.medium_id: m.refractive_index for m in scene.source_scene.display_media}
    lens = next(p for p in scene.parts if p.part_id == "OBJ")
    between = [s["surface_id"] for s in lens.surfaces if s["medium_before"] != "AIR" and s["medium_after"] != "AIR"]
    assert len(between) == 1
    meshes = [m for m in lens.meshes if m.surface_id == between[0]]
    (n1, n2) = sorted(media.values())
    for mesh in meshes:  # the whole interface, its rim too: no band inside the glass
        assert mesh.transmission == 1.0 and mesh.thickness_mm == 0.0 and mesh.alpha < 0.05 and mesh.ior == pytest.approx(n2 / n1, abs=1e-3)
    normals = np.repeat(FACING, 1, axis=0)
    src, a = shade(normals, meshes[0].rgb, alpha=meshes[0].alpha, metallic=0.0, roughness=meshes[0].roughness, transmission=1.0, ior=meshes[0].ior, thickness_mm=0.0)
    assert ((n2 - n1) / (n2 + n1)) ** 2 < 0.003 and a[0] < 0.05 and src.max() < 0.05  # N-BK7 / flint, by hand: (0.156 / 3.19) ** 2 = 0.0024


def test_every_view_of_a_lens_draws_the_same_pixels_every_time(cooke):
    scene = optical_view_scene(cooke)
    first, _ = render_view(scene, "iso", 480, 360, labels=False)
    second, _ = render_view(scene, "iso", 480, 360, labels=False)
    assert np.array_equal(first, second)


def test_the_parts_silhouette_is_the_projected_outline_of_an_ideal_disc(cooke):
    """The image plane seen square-on in the XY plane view is a disc of its radius (10 mm), within a pixel."""

    scene = optical_view_scene(cooke)
    alone = replace(scene, parts=tuple(p for p in scene.parts if p.part_id == "image"))
    mask, cam = parts_silhouette(alone, "plane_xy", 640, 480)
    assert float(mask.sum()) == pytest.approx(math.pi * (10.0 * cam.scale / 2.0) ** 2, rel=0.02)


def test_every_beam_colour_stands_out_on_the_dark_backdrop():
    """Directly (contrast 3 : 1 against both ends of the gradient), or by the light casing the renderers draw under it."""

    for wavelength in range(380, 1600, 5):
        rgb = wavelength_display_rgb(float(wavelength))
        lowest = min(contrast(rgb, OPTICAL_BACKDROP[0]), contrast(rgb, OPTICAL_BACKDROP[1]))
        if lowest < 3.0:
            assert beam_needs_casing(rgb, OPTICAL_VIEW), wavelength
        assert not beam_needs_casing(rgb, "EXPERIMENTAL_LAB_ASSEMBLY")
    assert contrast(BEAM_CASING_RGB, OPTICAL_BACKDROP[0]) > 3.0


def test_linear_and_srgb_are_inverse_where_the_glb_uses_them():
    from os_optics.optical_view.glb import _srgb_to_linear

    for c in (0.0, 0.02, 0.2, 0.62, 1.0):
        assert srgb_to_linear(np.array(c)) == pytest.approx(_srgb_to_linear(c), abs=1e-5)


# ----------------------------------------------------------------------------------------------------- camera presets


def _off_axis_deg(direction, axis) -> float:
    return math.degrees(math.acos(abs(float(np.dot(_u(direction), _u(axis))))))


def _u(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def _straight(scene: ViewerScene, axis) -> ViewerScene:
    points = np.outer(np.linspace(-20.0, 20.0, 5), _u(axis))
    beam = ViewBeam(beam_id="b", rgb=(1.0, 0.0, 0.0), wavelength_nm=632.8, runs=(("PRIMARY", points),))
    return replace(scene, beams=(beam,))


@pytest.mark.parametrize("axis", [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0.2, 0.3, 1), (1, -2, 3)])
def test_a_lens_layout_is_seen_from_the_near_side_with_its_path_level_for_a_beam_along_any_axis(cooke, axis):
    scene = _straight(optical_view_scene(cooke), axis)
    presets = camera_presets(scene)
    for name, least in (("iso", 70.0), ("path_horizontal", 89.99)):
        p = presets[name]
        assert _off_axis_deg(p["view_direction"], axis) >= least, name
        # the axis runs along the screen's horizontal, in the direction of the light
        direction, up = _u(p["view_direction"]), _u(p["up"])
        assert abs(float(np.dot(up, _u(axis)))) < 1e-3, name
        right = np.cross(-direction, up)
        assert float(np.dot(right, _u(axis))) > 0.9, name
    assert presets["path_horizontal"]["projection"] == "orthographic" and presets["iso"]["projection"] == "perspective"


def test_the_near_side_view_has_a_small_elevation(cooke):
    scene = _straight(optical_view_scene(cooke), (1, 0, 0))
    direction = _u(camera_presets(scene)["iso"]["view_direction"])
    assert 0.0 < math.degrees(math.asin(direction[2])) <= 15.0
    assert _off_axis_deg(direction, (1, 0, 0)) == pytest.approx(math.degrees(math.acos(math.sin(math.radians(14)) * math.cos(math.radians(12)))), abs=0.01)


def test_a_folded_layout_keeps_its_three_quarter_view(cooke):
    scene = optical_view_scene(cooke)
    legs = ViewBeam(beam_id="b", rgb=(1.0, 0.0, 0.0), wavelength_nm=632.8,
                    runs=(("PRIMARY", np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 50.0, 0.0]])),))
    folded = replace(scene, beams=(legs,))
    assert main_beam_axis(folded)[1] is False
    direction = _u(camera_presets(folded)["iso"]["view_direction"])
    assert math.degrees(math.asin(direction[2])) == pytest.approx(32.0, abs=0.01)


@pytest.mark.parametrize("name, direction, up", [("plane_xy", (0, 0, 1), (0, 1, 0)), ("plane_zx", (0, 1, 0), (1, 0, 0)), ("plane_yz", (1, 0, 0), (0, 0, 1))])
def test_a_plane_view_is_square_on_with_its_first_letter_right_and_its_second_up(cooke, name, direction, up):
    preset = camera_presets(optical_view_scene(cooke))[name]
    assert tuple(preset["view_direction"]) == tuple(float(c) for c in direction) and tuple(preset["up"]) == tuple(float(c) for c in up)
    assert preset["projection"] == "orthographic"
    # the plane's first letter to the right: right x up points at the viewer
    right = np.cross(-np.asarray(direction, float), np.asarray(up, float))
    assert np.allclose(np.cross(right, up), direction)


def test_every_preset_renders_and_names_its_view(cooke):
    scene = optical_view_scene(cooke)
    assert set(camera_presets(scene)) == set(PRESET_ORDER)
    _, drawn = render_view(scene, "plane_zx", 320, 240, labels=False)
    assert drawn["screen_right"] == [0.0, 0.0, 1.0] and drawn["screen_up"] == [1.0, 0.0, 0.0]
