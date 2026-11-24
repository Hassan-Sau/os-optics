"""The Optical Simulation View of a run directory as a ``ViewerScene``.

Element and plane meshes come from the run's own ``optical_scene.glb`` (the Engine's GLB: element-frame millimetres
under a design-frame -> glTF root node), placed with each node's design-frame pose; beams come from
``optical_scene.json`` (the Engine's chief and pupil polylines with their PRIMARY / SECONDARY stretches). The GLB must
be the rendering of that same scene (``scene_content_hash``), so the view never mixes two runs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .._vendor.shared_contracts.common.sag_shapes import sag_shape_text
from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from ..optical_view import GLB_FILE, SCENE_FILE, parse_glb, read_accessor
from ..optical_view.glb import EXTENSION_IOR, EXTENSION_TRANSMISSION, EXTENSION_VOLUME, element_part_class
from ..optical_view.section import FAR_PLANE_FACTOR
from .model import OPTICAL_VIEW, ViewBeam, ViewerScene, ViewMesh, ViewPart, place
from .shading import COATED_FACE_TINT

LINEAR_KNEE = 0.0031308
LINEAR_SLOPE = 12.92
SRGB_SCALE = 1.055
SRGB_OFFSET = 0.055
SRGB_EXPONENT = 1.0 / 2.4
MIN_EXTENT_MM = 1.0
NO_TRANSLATION = (0.0, 0.0, 0.0)
#: Faces whose colour says what they are, as the legend shows it: a coating or a modelled plane, not a tint of the glass.
COATED_FACES = frozenset({"filter_face", "splitter_face", "polarization_plane", "polarizer_body", "waveplate_body", "thin_lens", "attenuator_plane",
                          "faraday_plane", "faraday_body", "grating_face"})


class ViewerInputError(ValueError):
    """The files cannot be shown: missing, unreadable, or not one consistent view."""


def _srgb(linear: float) -> float:
    c = max(0.0, min(1.0, float(linear)))
    return c * LINEAR_SLOPE if c <= LINEAR_KNEE else SRGB_SCALE * c ** SRGB_EXPONENT - SRGB_OFFSET


def _finish(material: dict[str, Any]) -> dict[str, Any]:
    """A ViewMesh's finish, read from the glTF material; only a coated face's tint goes by the material's name."""

    pbr = material["pbrMetallicRoughness"]
    extensions = material.get("extensions", {})
    return {"metallic": float(pbr["metallicFactor"]), "roughness": float(pbr["roughnessFactor"]),
            "transmission": float(extensions.get(EXTENSION_TRANSMISSION, {}).get("transmissionFactor", 0.0)),
            "ior": extensions.get(EXTENSION_IOR, {}).get("ior"), "thickness_mm": extensions.get(EXTENSION_VOLUME, {}).get("thicknessFactor"),
            "tint": COATED_FACE_TINT * float(pbr["baseColorFactor"][3]) if material["name"] in COATED_FACES else 0.0}


def _node_meshes(gltf: dict[str, Any], binary: bytes, node: dict[str, Any]) -> list[ViewMesh]:
    """The node's meshes placed with its pose, and its child nodes' meshes placed with their own pose and then the node's
    (an element's optical-surface nodes, a compound unit's members and their surface nodes); a mesh drawn by an
    optical-surface node carries that surface's id (``<member>-<surface>`` inside a compound unit)."""

    position = node.get("translation", (0.0, 0.0, 0.0))
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    meshes = [replace(m, positions=place(m.positions, position, (w, x, y, z)), normals=place(m.normals, NO_TRANSLATION, (w, x, y, z)))
              for child in node.get("children", ()) for m in _node_meshes(gltf, binary, gltf["nodes"][child])]
    if "mesh" not in node:
        return meshes
    extras = node.get("extras", {})
    surface_id = _surface_key(extras) if extras.get("kind") == "OPTICAL_SURFACE" else None
    for prim in gltf["meshes"][node["mesh"]]["primitives"]:
        local = np.asarray(read_accessor(gltf, binary, prim["attributes"]["POSITION"]), dtype=float)
        indices = np.asarray(read_accessor(gltf, binary, prim["indices"]), dtype=np.int64).reshape(-1, 3)
        material = gltf["materials"][prim["material"]]
        factor = material["pbrMetallicRoughness"]["baseColorFactor"]
        rgb = (round(_srgb(factor[0]), 4), round(_srgb(factor[1]), 4), round(_srgb(factor[2]), 4))
        normals = np.asarray(read_accessor(gltf, binary, prim["attributes"]["NORMAL"]), dtype=float)
        meshes.append(ViewMesh(positions=place(local, position, (w, x, y, z)), triangles=indices, rgb=rgb, alpha=float(factor[3]),
                               material=material["name"], surface_id=surface_id, normals=place(normals, NO_TRANSLATION, (w, x, y, z)),
                               **_finish(material)))
    return meshes


#: Surface node extras a viewer shows for a picked surface (in this order).
SURFACE_FACTS = ("surface_id", "member_id", "role", "radius_mm", "conic", "aspheric_coefficients", "linear_sag_coefficient", "cylinder_axis", "sag",
                 "vertex_z_mm", "origin_mm", "normal",
                 "clear_radius_mm", "clear_half_widths_mm", "aperture_center_mm", "edge_radius_mm", "edge_half_widths_mm", "coating_kind",
                 "medium_before", "medium_after",
                 "solid_medium", "placement", "obstruction_radius_mm", "focal_length_mm", "profile_domain_limit_mm", "chord_tolerance_mm", "max_deviation_mm",
                 "chord_tolerance_met")


def _surface_key(extras: dict[str, Any]) -> str:
    """A surface's id within its part: ``<member>-<surface>`` inside a compound unit, else the surface id."""

    return extras.get("engine_surface_id") or extras["surface_id"]


def _surface_facts(gltf: dict[str, Any], node: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """The facts of every optical-surface node below ``node`` (a compound unit's members' surfaces too), in node order."""

    out = []
    for c in node.get("children", ()):
        extras = gltf["nodes"][c].get("extras", {})
        if extras.get("kind") == "OPTICAL_SURFACE":
            facts = {k: sag_shape_text(extras[k]) if k == "sag" else extras[k] for k in SURFACE_FACTS if k in extras}
            facts["surface_id"] = _surface_key(extras)
            out.append(facts)
        out.extend(_surface_facts(gltf, gltf["nodes"][c]))
    return tuple(out)


def _runs(points, roles, fallback: str) -> list[tuple[str, np.ndarray]]:
    pts = np.asarray(points, dtype=float)
    runs: list[tuple[str, list[int]]] = []
    for k in range(len(pts) - 1):
        role = roles[min(k, len(roles) - 1)] if roles else fallback
        if runs and runs[-1][0] == role:
            runs[-1][1].append(k + 1)
        else:
            runs.append((role, [k, k + 1]))
    return [(role, pts[idx]) for role, idx in runs]


def optical_view_scene(run_dir: str | Path) -> ViewerScene:
    """Read ``optical_scene.json`` + ``optical_scene.glb`` of a run directory (``os-optics project run --out-dir``)."""

    directory = Path(run_dir)
    try:
        scene_bytes = (directory / SCENE_FILE).read_bytes()
        glb_bytes = (directory / GLB_FILE).read_bytes()
    except OSError as exc:
        from ..local.configurations import run_reader

        if run_reader(directory) is not None:  # the run of a Project with configurations: one view per configuration
            raise ViewerInputError(f"{directory.as_posix()} is the run of a Project with configurations: view one configuration, "
                                   f"{(directory / 'configurations' / '<id>').as_posix()}") from None
        raise ViewerInputError(f"{directory.as_posix()} is not a run directory with {SCENE_FILE} and {GLB_FILE} ({exc.strerror})") from None
    try:
        scene = OpticalSimulationSceneV1.model_validate(json.loads(scene_bytes.decode("utf-8")))
        gltf, binary = parse_glb(glb_bytes)
    except ValueError as exc:
        raise ViewerInputError(f"the run's optical view cannot be read: {str(exc).splitlines()[0]}") from None
    if gltf["asset"].get("extras", {}).get("scene_content_hash") != scene.content_hash():
        raise ViewerInputError(f"{GLB_FILE} is not the rendering of {SCENE_FILE} (scene content hashes differ): re-run the project")
    return optical_scene_from(scene, gltf, binary)


def view_part_class(element: Any) -> str:
    """The legend class of a scene element, named by what is drawn: a compound unit is ``OPTICAL_UNIT`` (drawn by its
    members, whatever its role); otherwise the GLB's part class (catalog category, else optical role), with what that
    calls "other" named too -- an aperture stop (``APERTURE``: its plate), an ideal thin lens (``LENS``), an attenuator
    (``FILTER``)."""

    geom = element.geometry
    if getattr(geom, "kind", None) == "OPTICAL_UNIT":  # whatever its role: it is drawn by its members
        return "OPTICAL_UNIT"
    cls = element_part_class(element)
    if cls != "OTHER_OPTICAL":
        return cls
    kind = getattr(geom, "model_kind", None)
    if kind == "APERTURE_STOP" or element.optical_role.value == "aperture":
        return "APERTURE"
    if kind in ("IDEAL_THIN_LENS", "TABULATED_THIN_LENS"):
        return "LENS"
    if kind == "NEUTRAL_DENSITY":
        return "FILTER"
    return cls


def optical_scene_from(scene: OpticalSimulationSceneV1, gltf: dict[str, Any], binary: bytes) -> ViewerScene:
    elements = {e.element_id: e for e in scene.elements}
    planes = {p.observation_id: p for p in scene.observation_planes}
    parts: list[ViewPart] = []
    for node in gltf["nodes"]:
        extras = node.get("extras", {})
        name = node.get("name", "")
        if name.startswith("element:") and extras.get("element_id") in elements:
            element = elements[extras["element_id"]]
            facts = {"optical_role": element.optical_role.value, "position_mm": list(element.pose.position_mm)}
            geom = element.geometry
            if element.optical_model_id is not None:
                facts["optical_model_id"] = element.optical_model_id
            if getattr(geom, "form", None) is not None:
                facts.update({"category": geom.category.value, "form": geom.form.value})
                if getattr(geom, "diameter_mm", None) is not None:
                    facts["diameter_mm"] = geom.diameter_mm
                elif getattr(geom, "half_extents_mm", None) is not None:
                    facts["size_mm"] = [2.0 * h for h in geom.half_extents_mm]
            parts.append(ViewPart(part_id=element.element_id, part_class=view_part_class(element), label=element.label,
                                  sku=element.optical_sku, meshes=tuple(_node_meshes(gltf, binary, node)), facts=facts,
                                  surfaces=_surface_facts(gltf, node)))
        elif name.startswith("observation:") and extras.get("observation_id") in planes:
            plane = planes[extras["observation_id"]]
            target = plane.kind.value == "TARGET"
            facts = {"plane_kind": plane.kind.value, "absorbing_extent": plane.absorbing_extent.value, "target_ids": list(plane.target_ids),
                     "position_mm": list(plane.pose.position_mm),
                     "footprints": [{"path_id": f.path_id, "crossing_index": f.crossing_index, "power_W": f.power_W,
                                     "beam_radius_u_mm": f.beam_radius_u_mm, "beam_radius_v_mm": f.beam_radius_v_mm} for f in plane.footprints]}
            parts.append(ViewPart(part_id=plane.observation_id, part_class="TARGET_PLANE" if target else "OBSERVATION_PLANE",
                                  label=("Target plane " if target else "Observation plane ") + plane.observation_id, sku=None,
                                  meshes=tuple(_node_meshes(gltf, binary, node)), facts=facts))
    beams = []
    for path in scene.beam_paths:
        roles = [r.value for r in path.segment_roles]
        if path.chief_reference_from is not None:  # behind a central obstruction: the reference axis, not light
            k = path.chief_reference_from
            runs = _runs(path.chief_ray.points_mm[:k + 1], roles, path.display_role.value) if k > 0 else []
            runs += [("REFERENCE", np.asarray(path.chief_ray.points_mm[k:], dtype=float))]
        else:
            runs = _runs(path.chief_ray.points_mm, roles, path.display_role.value)
        for ray in path.pupil_rays:
            runs += [("PUPIL" if role == "PRIMARY" else "SECONDARY_PUPIL", pts) for role, pts in _runs(ray.points_mm, roles, path.display_role.value)]
        extensions = []
        for ray in (path.chief_ray, *path.pupil_rays):
            if ray.display_extension_to_mm is not None:
                role = path.display_role.value if ray.ray_index == 0 else ("PUPIL" if path.display_role.value == "PRIMARY" else "SECONDARY_PUPIL")
                if ray.ray_index == 0 and path.chief_reference_from is not None:
                    role = "REFERENCE"
                extensions.append((role, np.asarray([ray.points_mm[-1], ray.display_extension_to_mm], dtype=float)))
        beams.append(ViewBeam(beam_id=path.path_id, rgb=tuple(path.display_rgb), wavelength_nm=path.wavelength_nm,
                              runs=tuple(runs), extensions=tuple(extensions),
                              facts={"source_element_id": path.source_element_id, **({} if path.parent_path_id is None else {"parent_path_id": path.parent_path_id}),
                                     "termination": path.termination, "final_power_W": path.final_power_W,
                                     "display_role": path.display_role.value}))
    element_points = [m.positions for p in parts if p.part_class not in ("OBSERVATION_PLANE", "TARGET_PLANE") for m in p.meshes]
    lo = np.concatenate(element_points).min(axis=0) if element_points else np.asarray(scene.bounds.min_mm, dtype=float)
    hi = np.concatenate(element_points).max(axis=0) if element_points else np.asarray(scene.bounds.max_mm, dtype=float)
    extent = max(float(np.linalg.norm(hi - lo)), MIN_EXTENT_MM)
    notes = []
    for p in parts:
        if p.part_class in ("OBSERVATION_PLANE", "TARGET_PLANE"):
            plo, phi = p.bounds
            gap = np.maximum(np.maximum(lo - phi, plo - hi), 0.0)
            if float(np.linalg.norm(gap)) <= FAR_PLANE_FACTOR * extent:
                lo, hi = np.minimum(lo, plo), np.maximum(hi, phi)
            else:
                distance = round(float(np.linalg.norm(gap)), 1)
                notes.append({"code": "FAR_PLANE", "id": p.part_id, "distance_mm": distance,
                              "text": f"{p.part_id} lies {distance:g} mm from the optics: the default framing shows the optics, "
                                      "'Fit all' shows everything"})
    return ViewerScene(
        view_kind=OPTICAL_VIEW, frame=scene.frame, title="Optical Simulation View", parts=tuple(parts), beams=tuple(beams), markers=(),
        focus_min=lo, focus_max=hi, bounds_min=np.asarray(scene.bounds.min_mm, dtype=float), bounds_max=np.asarray(scene.bounds.max_mm, dtype=float),
        identity={"scene_content_hash": scene.content_hash(), "project_hash": scene.project.project_hash.digest,
                  "engine_input_hash": scene.engine_run.input_hash.digest, "catalog_version": scene.catalog_version},
        notes=tuple(notes), source_scene=scene,
    )
