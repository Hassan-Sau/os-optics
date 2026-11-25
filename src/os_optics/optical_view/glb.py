"""Deterministic glTF 2.0 binary (GLB) export of an ``OpticalSimulationSceneV1``. Stdlib only (``struct`` + ``json``).

Layout: one root node maps the OpenSci design frame (mm, +Z up) to glTF (m, +Y up) with a -90 deg rotation about X and a
0.001 scale. Under it:

* one node per optical element, at its nominal pose. An ``OPTIC_BODY`` -- a catalog product's or a declared model's
  (``optic_mesh.draw_body``) -- has one child node per optical surface (``surface:<element_id>/<surface_id>``, extras: the
  surface's id, role, profile, placement, clear aperture, coating kind, an aperture stop's obstruction or a thin lens's
  focal length, and the media on both sides), each drawn on its exact profile within ``CHORD_TOLERANCE_MM`` and split as
  the tracer models it: the clear aperture in the surface's material (glass, mirror, splitter, filter, Jones plane, thin
  lens), a narrow band marking the clear-aperture outline, and the rim, where the tracer blocks light, out to the outline in its own material (an
  aperture stop: its opaque plate around the opening and its central obstruction); the element node's own mesh holds the
  rest of the body (the edge walls of the glass between surfaces -- an air gap stays open --, an unmodelled substrate from
  the design geometry, a surface set's untraced faces, a polarization element's thin body and the bar along its local +X
  axis). A ``COMPOUND`` unit (``OPTICAL_UNIT``) is a node without a mesh whose child nodes are its members
  (``member:<element_id>:<member_id>``, ``OPTICAL_UNIT_MEMBER``, each at its pose in the unit frame with its body mesh and
  its own surface nodes ``surface:<element_id>/<member_id>-<surface_id>``). A source is an emitter, any other element a
  symbolic marker;
* one node per observation plane, at its pose: the plane (an absorbing detector plane in its own, more opaque
  material), plus the Engine beam footprints on it: a path's first crossing of the plane is a filled disc, each later
  crossing of the same path (the beam coming back through the plane after a fold, a double pass or a retro-reflection)
  an opaque ring of the same radius drawn a little further above the plane, so a returning beam that lands on its own
  first footprint stays visible;
* one node per beam path, holding the Engine's global polylines as thin tubes: the chief ray, the pupil rays, and the
  display-only stubs of free rays (their own translucent material). A PRIMARY stretch of a path is a thick opaque
  tube; a SECONDARY stretch (a weak branch such as a coating leak or a ghost reflection, ``BeamPathV1.segment_roles``)
  is a thin translucent tube in its own ``beam_secondary_*`` material, so it never hides a main beam on the same line;
* one node for the optical-axis markers.

The asset, the root and every node carry ``extras`` naming the view kind and each element's ``part_class`` (LENS,
MIRROR, BEAMSPLITTER, FILTER, WINDOW, WAVEPLATE, POLARIZER, SOURCE, DETECTOR, OTHER_OPTICAL), so any glTF viewer can
colour, filter or pick by class, element and surface. The root node's ``extras`` also state the frame mapping and the
mesh chord tolerance (``mesh_chord_tolerance_mm``). Output bytes depend only on the scene: fixed insertion order, sorted JSON keys, compact separators and
little-endian float32 / uint32 buffers.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any

from ..run_control import checkpoint
from .._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, RectangularApertureV1
from .._vendor.shared_contracts.optical_view import (
    BeamPathV1,
    CameraHintV1,
    ElementMarkerGeometryV1,
    OpticalUnitGeometryV1,
    ObservationAbsorptionV1,
    ObservationPlaneKindV1,
    ObservationPlaneV1,
    OpticalSimulationSceneV1,
    OpticBodyGeometryV1,
    SceneElementV1,
    SourceEmitterGeometryV1,
)

from .._version import ENGINE_VERSION
from . import geometry as g
from . import optic_mesh
from .identity import BUILDER_PACKAGE
from .optic_mesh import AXIS_BAR_HALF_WIDTH_MM, AXIS_BAR_LIFT_MM  # noqa: F401  (re-exported: the axis bar's display sizes)
from .surface_mesh import CHORD_TOLERANCE_MM

GLB_MAGIC = 0x46546C67  # "glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
GLB_HEADER_BYTES = 12
CHUNK_HEADER_BYTES = 8
WORD_BYTES = 4
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
TRIANGLES = 4

# OpenSci design frame (z up, mm) -> glTF (y up, m): rotate -90 deg about X, scale mm -> m.
ROOT_ROTATION_XYZW = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
ROOT_SCALE = (0.001, 0.001, 0.001)

# Display sizes (mm). Presentation only.
CHIEF_TUBE_RADIUS_MM = 0.3
PUPIL_TUBE_RADIUS_MM = 0.12
SECONDARY_CHIEF_TUBE_RADIUS_MM = 0.12  # a SECONDARY stretch: thinner than a primary chief ray, drawn inside it when collinear
SECONDARY_CHIEF_ALPHA = 0.5
SECONDARY_PUPIL_ALPHA = 0.18
AXIS_TUBE_RADIUS_MM = 0.08
# Behind a central obstruction the chief ray is drawn as a thin grey reference axis (it carries no light).
REFERENCE_AXIS_TUBE_RADIUS_MM = 0.06
REFERENCE_AXIS_ALPHA = 0.6
REFERENCE_AXIS_RGB = (0.45, 0.47, 0.52)
MIN_FOOTPRINT_DISPLAY_RADIUS_MM = 0.3
FOOTPRINT_LIFT_MM = 0.02  # above the plane, to avoid z-fighting; crossing k of a path is lifted (k + 1) times this
REPEATED_CROSSING_RING_WIDTH = 0.35  # ring width of a later crossing, as a fraction of its footprint radius
CAMERA_YFOV_RAD = 0.6
CAMERA_ZNEAR_FACTOR = 1e-3  # of the eye-target distance
CAMERA_ZFAR_FACTOR = 10.0

# Materials: sRGB display colours -> glTF linear factors (IEC 61966-2-1 transfer curve).
SRGB_LINEAR_KNEE = 0.04045
SRGB_LINEAR_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_SCALE = 1.055
SRGB_GAMMA = 2.4
MATERIAL_DECIMALS = 6
MATERIAL_ROUGHNESS = 0.35
FOOTPRINT_ALPHA = 0.85
FOOTPRINT_EMISSIVE = 0.8
REPEATED_FOOTPRINT_ALPHA = 1.0
REPEATED_FOOTPRINT_EMISSIVE = 0.95
PUPIL_RAY_ALPHA = 0.6
PUPIL_RAY_EMISSIVE = 0.7
DISPLAY_STUB_ALPHA = 0.25
DISPLAY_STUB_EMISSIVE = 0.4
HEX_CHANNEL_MAX = 255
THICKNESS_DECIMALS = 3
RATIO_DECIMALS = 4
AIR_MEDIUM = "AIR"
#: The media that are no body's glass (the environment's air and a declared model's evacuated gap).
SURROUNDING_MEDIA = frozenset({AIR_MEDIUM, "VACUUM"})
EXTENSION_IOR = "KHR_materials_ior"
EXTENSION_TRANSMISSION = "KHR_materials_transmission"
EXTENSION_VOLUME = "KHR_materials_volume"
CEMENTED_ALPHA = 0.02  # a cemented interface, wherever it reaches: a real doublet shows no band there
RAY_SOURCE_NOTE = f"{BUILDER_PACKAGE} trace (os_optics.validate_with_ray_paths)"
FRAME_MAPPING_NOTE = ("the root node maps the design frame (mm, +Z up) to glTF (m, +Y up): rotation -90 deg about X, scale 0.001; "
                      "node translations below the root are millimetres in the design frame")
# The display class of an element without a catalog category (a Project custom part or declared model), by optical role.
PART_CLASS_BY_ROLE = {"source": "SOURCE", "detector": "DETECTOR", "lens": "LENS", "mirror": "MIRROR", "splitter": "BEAMSPLITTER",
                      "filter": "FILTER", "window": "WINDOW", "polarizer": "POLARIZER", "waveplate": "WAVEPLATE"}
OTHER_PART_CLASS = "OTHER_OPTICAL"


def element_part_class(element: SceneElementV1) -> str:
    """The display class of a scene element: its catalog category, else a class from its optical role."""

    geom = element.geometry
    if isinstance(geom, OpticBodyGeometryV1) and geom.category is not None:
        return geom.category.value
    return PART_CLASS_BY_ROLE.get(element.optical_role.value, OTHER_PART_CLASS)


def _srgb_to_linear(c: float) -> float:
    return c / SRGB_LINEAR_SLOPE if c <= SRGB_LINEAR_KNEE else ((c + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA


@dataclass(frozen=True)
class MaterialFinish:
    """What a material's surface is made of, besides its colour: the source of the GLB's PBR factors, which every renderer of
    the view reads (``viewer.shading``, ``docs/VIEWER_MATERIALS.md``). ``transmission`` 1 is glass: light passes."""

    metallic: float = 0.0
    roughness: float = MATERIAL_ROUGHNESS
    transmission: float = 0.0


POLISHED_GLASS = MaterialFinish(roughness=0.04, transmission=1.0)
GROUND_GLASS = MaterialFinish(roughness=0.55, transmission=1.0)  # the edge of a lens, wall and rim: frosted, still see-through
BLACKENED_METAL = MaterialFinish(metallic=0.3, roughness=0.55)
#: Materials not named here (beams, footprints, markers, planes) keep the default finish.
MATERIAL_FINISHES = {
    "optic_glass": POLISHED_GLASS, "optic_substrate": POLISHED_GLASS, "polarization_plane": POLISHED_GLASS, "filter_face": POLISHED_GLASS,
    "splitter_face": POLISHED_GLASS, "polarizer_body": POLISHED_GLASS, "waveplate_body": POLISHED_GLASS, "thin_lens": POLISHED_GLASS,
    "attenuator_plane": POLISHED_GLASS, "faraday_plane": POLISHED_GLASS, "faraday_body": POLISHED_GLASS, "grating_face": POLISHED_GLASS,
    "grating_body": MaterialFinish(metallic=0.6, roughness=0.45), "grating_rule": BLACKENED_METAL,
    "optic_edge": GROUND_GLASS, "optic_rim": GROUND_GLASS, "aperture_outline": POLISHED_GLASS, "stop_plate": BLACKENED_METAL,
    "mirror_face": MaterialFinish(metallic=1.0, roughness=0.05), "mirror_substrate": MaterialFinish(metallic=0.6, roughness=0.45),
}


def _material(name: str, rgb: tuple[float, float, float], alpha: float, *, emissive: float = 0.0, double_sided: bool = False,
              ior: float | None = None, thickness_mm: float | None = None, cemented: bool = False) -> dict[str, Any]:
    lin = [round(_srgb_to_linear(c), MATERIAL_DECIMALS) for c in rgb]
    finish = POLISHED_GLASS if cemented else MATERIAL_FINISHES.get(name, MaterialFinish())
    alpha = CEMENTED_ALPHA if cemented else alpha
    mat: dict[str, Any] = {
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": [*lin, alpha], "metallicFactor": finish.metallic, "roughnessFactor": finish.roughness},
        "doubleSided": double_sided,
    }
    extensions: dict[str, Any] = {}
    if finish.transmission > 0.0:
        extensions[EXTENSION_TRANSMISSION] = {"transmissionFactor": finish.transmission}
        if thickness_mm is not None:  # mesh units (mm below the root node)
            extensions[EXTENSION_VOLUME] = {"thicknessFactor": thickness_mm}
    if ior is not None:
        extensions[EXTENSION_IOR] = {"ior": ior}
    if extensions:
        mat["extensions"] = extensions
    if alpha < 1.0:
        mat["alphaMode"] = "BLEND"
    if emissive > 0.0:
        mat["emissiveFactor"] = [round(c * emissive, MATERIAL_DECIMALS) for c in lin]
    return mat


STATIC_MATERIALS: dict[str, tuple[tuple[float, float, float], float, float, bool]] = {
    "optic_glass": ((0.62, 0.80, 0.95), 0.35, 0.0, True),  # a refracting surface (uncoated or anti-reflection)
    "optic_edge": ((0.55, 0.72, 0.88), 0.08, 0.0, True),  # the edge wall of the glass between two surfaces
    "optic_substrate": ((0.70, 0.82, 0.92), 0.3, 0.0, True),  # an unmodelled transmissive substrate (design geometry)
    "optic_rim": ((0.50, 0.55, 0.62), 0.1, 0.0, True),  # between the clear aperture and the outline: the tracer blocks light there
    "aperture_outline": ((0.10, 0.27, 0.52), 0.12, 0.0, True),  # the band that marks the clear-aperture outline (a viewer may mark it)
    "profile_undefined": ((0.86, 0.16, 0.16), 1.0, 0.3, True),  # where an outline reaches beyond the profile's domain
    "polarization_plane": ((0.60, 0.52, 0.86), 0.35, 0.0, True),  # the clear aperture of a waveplate / polarizer's Jones plane
    "filter_face": ((0.45, 0.85, 0.55), 0.55, 0.0, True),  # a refracting surface with a filtering / partial coating
    "splitter_face": ((0.96, 0.78, 0.35), 0.6, 0.0, True),  # a splitting coated face
    "mirror_face": ((0.74, 0.79, 0.88), 1.0, 0.0, True),  # a reflecting coated face
    "mirror_substrate": ((0.55, 0.57, 0.62), 0.9, 0.0, False),  # an unmodelled reflector substrate (design geometry)
    "polarizer_body": ((0.42, 0.42, 0.48), 0.55, 0.0, True),
    "waveplate_body": ((0.78, 0.68, 0.95), 0.45, 0.0, True),
    "polarization_axis": ((0.98, 0.97, 0.90), 1.0, 0.35, False),  # the local +X axis of a waveplate / polarizer, a Faraday rotator's field axis
    "source_emitter": ((0.9, 0.36, 0.1), 1.0, 0.25, False),  # a light source: deep orange, distinct from every optic and plane
    "element_marker": ((0.60, 0.60, 0.66), 0.6, 0.0, True),
    "stop_plate": ((0.22, 0.22, 0.25), 1.0, 0.0, True),  # the opaque plate (and central obstruction) of an aperture stop
    "thin_lens": ((0.55, 0.85, 0.80), 0.4, 0.0, True),  # the clear aperture of a declared ideal thin lens (one plane)
    "attenuator_plane": ((0.35, 0.35, 0.38), 0.55, 0.0, True),  # the clear aperture of a neutral-density attenuator (one plane)
    "faraday_plane": ((0.86, 0.45, 0.62), 0.45, 0.0, True),  # the clear aperture of a Faraday rotator's plane
    "faraday_body": ((0.72, 0.36, 0.50), 0.5, 0.0, True),  # the rod or plate of a Faraday rotator
    "grating_face": ((0.62, 0.72, 0.86), 0.75, 0.0, True),  # the clear aperture of a diffraction grating's ruled plane
    "grating_body": ((0.36, 0.40, 0.48), 0.95, 0.0, False),  # the plate of a diffraction grating
    "grating_rule": ((0.13, 0.16, 0.22), 1.0, 0.0, False),  # the rules along a diffraction grating's grooves
    "plane_observation": ((0.75, 0.78, 0.82), 0.22, 0.0, True),  # a transparent monitor plane
    "plane_target": ((0.98, 0.72, 0.25), 0.5, 0.0, True),
    "plane_observation_absorbing": ((0.30, 0.34, 0.42), 0.55, 0.0, True),  # a detector plane: absorbs what reaches it
    "plane_target_absorbing": ((0.85, 0.50, 0.12), 0.55, 0.0, True),
    "axis_marker": ((0.35, 0.35, 0.38), 1.0, 0.0, False),
}


def _hex(rgb: tuple[float, float, float]) -> str:
    return "".join(f"{round(c * HEX_CHANNEL_MAX):02x}" for c in rgb)


@dataclass(frozen=True)
class _Glass:
    """The glass a surface bounds, where the run states it: its index and its thickness along the element's local z (mm)."""

    ior: float | None = None
    thickness_mm: float | None = None
    cemented: bool = False  # between two glasses: ``ior`` is their ratio and nothing lies in it


class _GltfBuilder:
    def __init__(self, media: dict[str, float] | None = None) -> None:
        self.media = media or {}
        self.bin = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self.material_index: dict[str, int] = {}
        self.nodes: list[dict[str, Any]] = []

    def material(self, name: str, rgb=None, alpha: float = 1.0, *, emissive: float = 0.0, double_sided: bool = False, glass: _Glass | None = None) -> int:
        """The index of a material; one name has one glTF material per glass it is drawn on."""

        finish = MATERIAL_FINISHES.get(name, MaterialFinish())
        if finish.transmission == 0.0:
            glass = None  # not on glass
        elif glass is not None and not glass.cemented and finish.roughness != POLISHED_GLASS.roughness:
            glass = _Glass(glass.ior)  # a ground edge has the glass's index, but its depth is not what it shows
        key = name if glass is None else (name, glass)
        if key not in self.material_index:
            if rgb is None:
                rgb, alpha, emissive, double_sided = STATIC_MATERIALS[name]
            self.material_index[key] = len(self.materials)
            self.materials.append(_material(name, rgb, alpha, emissive=emissive, double_sided=double_sided, ior=glass.ior if glass else None,
                                            thickness_mm=glass.thickness_mm if glass else None, cemented=bool(glass and glass.cemented)))
        return self.material_index[key]

    def _view(self, data: bytes, target: int) -> int:
        while len(self.bin) % WORD_BYTES:
            self.bin.append(0)
        self.buffer_views.append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data), "target": target})
        self.bin.extend(data)
        return len(self.buffer_views) - 1

    def _vec3_accessor(self, values: list[g.Vec3], *, with_bounds: bool) -> int:
        flat = [float(c) for v in values for c in v]
        data = struct.pack(f"<{len(flat)}f", *flat)
        accessor: dict[str, Any] = {"bufferView": self._view(data, ARRAY_BUFFER), "componentType": FLOAT, "count": len(values), "type": "VEC3"}
        if with_bounds:  # bounds of the float32 values actually stored
            stored = struct.unpack(f"<{len(flat)}f", data)
            accessor["min"] = [min(stored[i::3]) for i in range(3)]
            accessor["max"] = [max(stored[i::3]) for i in range(3)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def primitive(self, mesh: g.Mesh, material: int) -> dict[str, Any]:
        positions = self._vec3_accessor(mesh.positions, with_bounds=True)
        normals = self._vec3_accessor([_safe_unit(n) for n in mesh.normals], with_bounds=False)
        data = struct.pack(f"<{len(mesh.indices)}I", *mesh.indices)
        self.accessors.append({"bufferView": self._view(data, ELEMENT_ARRAY_BUFFER), "componentType": UNSIGNED_INT,
                               "count": len(mesh.indices), "type": "SCALAR"})
        return {"attributes": {"NORMAL": normals, "POSITION": positions}, "indices": len(self.accessors) - 1,
                "material": material, "mode": TRIANGLES}

    def mesh(self, name: str, primitives: list[dict[str, Any]]) -> int:
        self.meshes.append({"name": name, "primitives": primitives})
        return len(self.meshes) - 1

    def node(self, node: dict[str, Any]) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1


def _safe_unit(v: g.Vec3) -> g.Vec3:
    n = g.norm(v)
    return (0.0, 0.0, 1.0) if n == 0.0 else g.scale(v, 1.0 / n)


def _xyzw(q) -> list[float]:
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [x / n, y / n, z / n, w / n]


def _pose_fields(pose) -> dict[str, Any]:
    return {"translation": [float(c) for c in pose.position_mm], "rotation": _xyzw(pose.quaternion_wxyz)}


def _merged_primitives(b: _GltfBuilder, meshes: list[tuple[str, g.Mesh]], glass: _Glass | None = None) -> list[dict[str, Any]]:
    merged: dict[str, g.Mesh] = {}
    for name, mesh in meshes:
        merged.setdefault(name, g.Mesh()).extend(mesh)
    return [b.primitive(mesh, b.material(name, glass=glass)) for name, mesh in merged.items() if not mesh.empty]


def _z_span(surfaces) -> float | None:
    """The extent along the element's local z of what the drawn ``surfaces`` hold (the glass's edge-to-edge thickness)."""

    zs = [p[2] for surface in surfaces for _, mesh in surface.meshes for p in mesh.positions]
    return round(max(zs) - min(zs), THICKNESS_DECIMALS) if zs and max(zs) > min(zs) else None


def _surface_glasses(b: _GltfBuilder, geom: OpticBodyGeometryV1, drawing: optic_mesh.BodyDrawing) -> list[_Glass]:
    """Per drawn surface, the glass behind it: the medium after it, else before it (a surface between two glasses draws the one
    after it), or the solid of a surface set. A surface in air on both sides has none."""

    namespace = "project:" if geom.is_declared else "catalog:"
    index = lambda medium: b.media.get(namespace + medium)  # noqa: E731
    out = []
    for k, surface in enumerate(drawing.surfaces):
        facts = surface.facts
        if "solid_medium" in facts:
            out.append(_Glass(b.media.get(namespace + facts["solid_medium"]), _z_span(drawing.surfaces)))
            continue
        after, before = facts.get("medium_after", AIR_MEDIUM), facts.get("medium_before", AIR_MEDIUM)
        if after in SURROUNDING_MEDIA and before in SURROUNDING_MEDIA:
            out.append(_Glass())
            continue
        if after not in SURROUNDING_MEDIA and before not in SURROUNDING_MEDIA:  # a cemented interface reflects by the two glasses' ratio
            n1, n2 = index(before), index(after)
            ratio = None if n1 is None or n2 is None else round(max(n1 / n2, n2 / n1), RATIO_DECIMALS)
            out.append(_Glass(ratio, 0.0, True))
            continue
        pair = drawing.surfaces[k:k + 2] if after not in SURROUNDING_MEDIA else drawing.surfaces[max(k - 1, 0):k + 1]
        out.append(_Glass(b.media.get(namespace + (after if after not in SURROUNDING_MEDIA else before)), _z_span(pair)))
    return out


def _surface_extras(element: SceneElementV1, drawing: optic_mesh.SurfaceDrawing, tolerance: float) -> dict[str, Any]:
    """What a viewer shows when a surface is picked: its identity, the published profile, placement and clear aperture
    (the numbers the run's surface was compiled from) and the media on both sides."""

    surface = drawing.surface
    extras: dict[str, Any] = {"opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "OPTICAL_SURFACE", "element_id": element.element_id,
                              "surface_id": surface.surface_id, "role": surface.role.value, "length_unit": "mm",
                              "chord_tolerance_mm": tolerance}
    for name in SURFACE_FACT_FIELDS:
        value = getattr(surface, name)
        if value is not None:
            if hasattr(value, "model_dump"):
                extras[name] = value.model_dump(mode="json", exclude_none=True)
            else:
                extras[name] = value.value if hasattr(value, "value") else (list(value) if isinstance(value, tuple) else value)
    extras.update(drawing.facts)
    return extras


#: The published surface fields a surface node repeats in its extras (when present).
SURFACE_FACT_FIELDS = ("vertex_z_mm", "origin_mm", "normal", "u_axis", "radius_mm", "conic", "aspheric_coefficients", "linear_sag_coefficient",
                       "cylinder_axis", "sag",
                       "clear_radius_mm", "clear_half_widths_mm", "aperture_center_mm", "edge_radius_mm", "edge_half_widths_mm",
                       "coating_kind", "obstruction_radius_mm", "focal_length_mm")


def _body_content(b: _GltfBuilder, element: SceneElementV1, geom: OpticBodyGeometryV1, member_id: str | None = None
                  ) -> tuple[list[dict[str, Any]], list[int]]:
    """An ``OPTIC_BODY``'s body primitives and its optical-surface child nodes (a compound member's surfaces are named
    ``<member_id>-<surface_id>``, as the Engine names them)."""

    drawing = optic_mesh.draw_body(geom)
    glasses = _surface_glasses(b, geom, drawing)
    children = []
    for surface, glass in zip(drawing.surfaces, glasses, strict=True):
        key = surface.surface.surface_id if member_id is None else f"{member_id}-{surface.surface.surface_id}"
        name = f"surface:{element.element_id}/{key}"
        extras = _surface_extras(element, surface, drawing.tolerance_mm)
        if member_id is not None:
            extras.update({"member_id": member_id, "engine_surface_id": key})
        node: dict[str, Any] = {"name": name, "extras": extras}
        surface_prims = _merged_primitives(b, surface.meshes, glass)
        if surface_prims:
            node["mesh"] = b.mesh(name, surface_prims)
        children.append(b.node(node))
    known = [glass for glass in glasses if glass.ior is not None and not glass.cemented]
    return _merged_primitives(b, drawing.body, known[0] if known else None), children


def _body_extras(geom: OpticBodyGeometryV1) -> dict[str, Any]:
    extras: dict[str, Any] = {"model_kind": geom.model_kind, "surfaces": [f"{s.surface_id}:{s.role.value}" for s in geom.surfaces]}
    if geom.category is not None and geom.form is not None:
        extras.update({"category": geom.category.value, "form": geom.form.value})
    if geom.polarization is not None:
        extras["polarization_axis_local"] = geom.polarization.axis_local
    return extras


def _unit_node(b: _GltfBuilder, element: SceneElementV1, geom: OpticalUnitGeometryV1, extras: dict[str, Any]) -> int:
    """A ``COMPOUND`` unit: a node at the unit's pose whose children are its members, each at its pose in the unit frame."""

    children = []
    for member in geom.members:
        prims, surfaces = _body_content(b, element, member.body, member.member_id)
        # (a body of a catalog product names no declared model)
        named = {} if member.optical_model_id is None else {"optical_model_id": member.optical_model_id}
        member_extras = {"opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "OPTICAL_UNIT_MEMBER", "element_id": element.element_id,
                         "member_id": member.member_id, **named, "length_unit": "mm", **_body_extras(member.body)}
        node: dict[str, Any] = {"name": f"member:{element.element_id}:{member.member_id}", **_pose_fields(member.pose), "extras": member_extras}
        if prims:
            node["mesh"] = b.mesh(f"element:{element.element_id}:{member.member_id}", prims)
        if surfaces:
            node["children"] = surfaces
        children.append(b.node(node))
    extras["members"] = [m.member_id for m in geom.members]
    return b.node({"name": f"element:{element.element_id}", "children": children, **_pose_fields(element.pose), "extras": extras})


def _emitter_plate(half_x: float, half_y: float, depth: float) -> g.Mesh:
    """A rectangular emitting surface at ``z = 0`` facing +z, on a plate of ``depth`` behind it."""
    m = g.Mesh()
    faces = (
        ((-half_x, -half_y, 0.0), (half_x, -half_y, 0.0), (half_x, half_y, 0.0), (-half_x, half_y, 0.0), (0.0, 0.0, 1.0)),
        ((-half_x, half_y, -depth), (half_x, half_y, -depth), (half_x, -half_y, -depth), (-half_x, -half_y, -depth), (0.0, 0.0, -1.0)),
        ((-half_x, -half_y, -depth), (half_x, -half_y, -depth), (half_x, -half_y, 0.0), (-half_x, -half_y, 0.0), (0.0, -1.0, 0.0)),
        ((half_x, -half_y, -depth), (half_x, half_y, -depth), (half_x, half_y, 0.0), (half_x, -half_y, 0.0), (1.0, 0.0, 0.0)),
        ((half_x, half_y, -depth), (-half_x, half_y, -depth), (-half_x, half_y, 0.0), (half_x, half_y, 0.0), (0.0, 1.0, 0.0)),
        ((-half_x, half_y, -depth), (-half_x, -half_y, -depth), (-half_x, -half_y, 0.0), (-half_x, half_y, 0.0), (-1.0, 0.0, 0.0)),
    )
    for a, b, c, d, normal in faces:
        first = len(m.positions)
        m.positions.extend((a, b, c, d))
        m.normals.extend((normal,) * 4)
        m.indices.extend((first, first + 1, first + 2, first, first + 2, first + 3))
    return m


def _element_node(b: _GltfBuilder, element: SceneElementV1) -> int:
    geom = element.geometry
    prims = []
    children: list[int] = []
    if isinstance(geom, OpticalUnitGeometryV1):
        extras_unit: dict[str, Any] = {
            "opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "OPTICAL_ELEMENT", "element_id": element.element_id,
            "optical_role": element.optical_role.value, "part_class": element_part_class(element), "label": element.label,
            "geometry_kind": geom.kind, "length_unit": "mm", "geometry_status": geom.status.value, "geometry_basis": geom.basis.value,
            "optical_model_id": element.optical_model_id, "chord_tolerance_mm": CHORD_TOLERANCE_MM,
        }
        return _unit_node(b, element, geom, extras_unit)
    if isinstance(geom, OpticBodyGeometryV1):
        prims, children = _body_content(b, element, geom)
    elif isinstance(geom, SourceEmitterGeometryV1):
        if geom.emitting_area is not None and geom.emitting_area.shape == "RECTANGLE":
            prims.append(b.primitive(_emitter_plate(geom.emitting_area.size_x_mm / 2.0, geom.emitting_area.size_y_mm / 2.0, geom.marker_length_mm),
                                     b.material("source_emitter")))
        else:
            prims.append(b.primitive(g.cylinder(geom.emitter_radius_mm, -geom.marker_length_mm, 0.0), b.material("source_emitter")))
    else:
        assert isinstance(geom, ElementMarkerGeometryV1)
        prims.append(b.primitive(g.cylinder(geom.radius_mm, -geom.thickness_mm / 2, geom.thickness_mm / 2), b.material("element_marker")))
    extras: dict[str, Any] = {
        "opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "OPTICAL_ELEMENT", "element_id": element.element_id,
        "optical_role": element.optical_role.value, "part_class": element_part_class(element), "label": element.label,
        "geometry_kind": geom.kind, "length_unit": "mm",
        "geometry_status": geom.status.value, "geometry_basis": geom.basis.value,
    }
    if element.optical_sku is not None:
        extras["optical_sku"] = element.optical_sku
    if element.optical_model_id is not None:
        extras["optical_model_id"] = element.optical_model_id
    if isinstance(geom, OpticBodyGeometryV1):
        extras.update(_body_extras(geom))
        extras["chord_tolerance_mm"] = CHORD_TOLERANCE_MM
    node = {"name": f"element:{element.element_id}", **_pose_fields(element.pose), "extras": extras}
    if prims:
        node["mesh"] = b.mesh(f"element:{element.element_id}", prims)
    if children:
        node["children"] = children
    return b.node(node)


def _plane_node(b: _GltfBuilder, plane: ObservationPlaneV1, colours: dict[str, tuple[float, float, float]]) -> int:
    aperture = plane.aperture
    if isinstance(aperture, CircularApertureV1):
        shape = g.disc(aperture.radius_mm, 0.0, segments=g.circle_segments(aperture.radius_mm, CHORD_TOLERANCE_MM))
        shape.positions = [(p[0] + aperture.center_offset_mm[0], p[1] + aperture.center_offset_mm[1], p[2]) for p in shape.positions]
    elif isinstance(aperture, RectangularApertureV1):
        shape = g.rectangle(aperture.half_width_mm, aperture.half_height_mm, center=aperture.center_offset_mm)
    else:
        shape = g.rectangle(plane.display_half_extent_mm, plane.display_half_extent_mm)
    material = "plane_target" if plane.kind is ObservationPlaneKindV1.TARGET else "plane_observation"
    if plane.absorbing_extent is not ObservationAbsorptionV1.NONE:
        material += "_absorbing"
    prims = [b.primitive(shape, b.material(material))]
    for footprint in plane.footprints:
        rgb = colours[footprint.path_id]
        radius = max(footprint.beam_radius_u_mm or 0.0, footprint.beam_radius_v_mm or 0.0, MIN_FOOTPRINT_DISPLAY_RADIUS_MM)
        lift = FOOTPRINT_LIFT_MM * (footprint.crossing_index + 1)
        if footprint.crossing_index == 0:
            spot = g.disc(radius, lift)
            spot_material = b.material(f"footprint_{_hex(rgb)}", rgb, FOOTPRINT_ALPHA, emissive=FOOTPRINT_EMISSIVE, double_sided=True)
        else:
            spot = g.annulus(radius * (1.0 - REPEATED_CROSSING_RING_WIDTH), radius, lift)
            spot_material = b.material(f"footprint_repeat_{_hex(rgb)}", rgb, REPEATED_FOOTPRINT_ALPHA, emissive=REPEATED_FOOTPRINT_EMISSIVE,
                                       double_sided=True)
        spot.positions = [(p[0] + footprint.chief_uv_mm[0], p[1] + footprint.chief_uv_mm[1], p[2]) for p in spot.positions]
        prims.append(b.primitive(spot, spot_material))
    mesh = b.mesh(f"observation:{plane.observation_id}", prims)
    extras = {"opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": f"{plane.kind.value}_PLANE", "observation_id": plane.observation_id,
              "target_ids": list(plane.target_ids), "absorbing_extent": plane.absorbing_extent.value,
              "footprints": [{"path_id": f.path_id, "crossing_index": f.crossing_index, "beam_radius_u_mm": f.beam_radius_u_mm,
                              "beam_radius_v_mm": f.beam_radius_v_mm} for f in plane.footprints]}
    return b.node({"name": f"observation:{plane.observation_id}", "mesh": mesh, **_pose_fields(plane.pose), "extras": extras})


def _role_runs(points: list, roles: tuple) -> list[tuple[str, list]]:
    """Consecutive polyline segments of one display role, as ``(role, points)`` runs."""

    runs: list[tuple[str, list]] = []
    for k in range(len(points) - 1):
        role = roles[min(k, len(roles) - 1)].value if roles else "PRIMARY"
        if runs and runs[-1][0] == role:
            runs[-1][1].append(points[k + 1])
        else:
            runs.append((role, [points[k], points[k + 1]]))
    return runs


def _beam_node(b: _GltfBuilder, path: BeamPathV1) -> int:
    rgb = tuple(path.display_rgb)
    key = _hex(rgb)
    meshes = {name: g.Mesh() for name in ("chief", "pupil", "secondary_chief", "secondary_pupil", "stub", "reference")}
    reference_from = path.chief_reference_from
    for ray in (path.chief_ray, *path.pupil_rays):
        chief = ray.ray_index == 0
        points = list(ray.points_mm)
        if chief and reference_from is not None:  # behind a central obstruction the chief ray is the reference axis, not light
            meshes["reference"].extend(g.tube(points[reference_from:], REFERENCE_AXIS_TUBE_RADIUS_MM))
            points = points[:reference_from + 1]
        for role, run in _role_runs(points, path.segment_roles):
            secondary = role == "SECONDARY"
            radius = (SECONDARY_CHIEF_TUBE_RADIUS_MM if secondary else CHIEF_TUBE_RADIUS_MM) if chief else PUPIL_TUBE_RADIUS_MM
            meshes[("secondary_" if secondary else "") + ("chief" if chief else "pupil")].extend(g.tube(run, radius))
        if ray.display_extension_to_mm is not None and chief and reference_from is not None:
            meshes["reference"].extend(g.tube([ray.points_mm[-1], ray.display_extension_to_mm], REFERENCE_AXIS_TUBE_RADIUS_MM))
        elif ray.display_extension_to_mm is not None:
            secondary = path.display_role.value == "SECONDARY"
            radius = (SECONDARY_CHIEF_TUBE_RADIUS_MM if secondary else CHIEF_TUBE_RADIUS_MM) if chief else PUPIL_TUBE_RADIUS_MM
            meshes["stub"].extend(g.tube([ray.points_mm[-1], ray.display_extension_to_mm], radius))
    styles = {
        "chief": (f"beam_chief_{key}", 1.0, 1.0),
        "pupil": (f"beam_pupil_{key}", PUPIL_RAY_ALPHA, PUPIL_RAY_EMISSIVE),
        "secondary_chief": (f"beam_secondary_chief_{key}", SECONDARY_CHIEF_ALPHA, PUPIL_RAY_EMISSIVE),
        "secondary_pupil": (f"beam_secondary_pupil_{key}", SECONDARY_PUPIL_ALPHA, DISPLAY_STUB_EMISSIVE),
        "stub": (f"beam_display_extension_{key}", DISPLAY_STUB_ALPHA, DISPLAY_STUB_EMISSIVE),
        "reference": ("beam_reference_axis", REFERENCE_AXIS_ALPHA, 0.0),
    }
    prims = [b.primitive(mesh, b.material(styles[name][0], REFERENCE_AXIS_RGB if name == "reference" else rgb, styles[name][1],
                                          emissive=styles[name][2])) for name, mesh in meshes.items() if not mesh.empty]
    extras = {"opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "BEAM_PATH", "path_id": path.path_id,
              "source_element_id": path.source_element_id, "wavelength_nm": path.wavelength_nm, "termination": path.termination,
              "display_role": path.display_role.value, "segment_roles": [r.value for r in path.segment_roles],
              "final_power_W": path.final_power_W, "pupil_rays_shown": len(path.pupil_rays),
              "pupil_rays_traced": path.pupil_ray_count_traced, "ray_source": RAY_SOURCE_NOTE, "length_unit": "mm"}
    if reference_from is not None:
        extras["chief_reference_from"] = reference_from
    node: dict[str, Any] = {"name": f"beam:{path.path_id}", "extras": extras}
    if prims:
        node["mesh"] = b.mesh(f"beam:{path.path_id}", prims)
    return b.node(node)


def _to_gltf(p: g.Vec3) -> list[float]:
    """Design frame (mm, +Z up) -> glTF frame (m, +Y up): the root node's transform, applied to a point."""
    return [p[0] * ROOT_SCALE[0], p[2] * ROOT_SCALE[0], -p[1] * ROOT_SCALE[0]]


def _camera_node(b: _GltfBuilder, hint: CameraHintV1, cameras: list[dict[str, Any]], view_kind: str = "OPTICAL_SIMULATION_VIEW") -> int:
    """A top-level perspective camera at the scene's camera hint (glTF cameras look along local -Z, up +Y)."""
    eye, target = _to_gltf(hint.eye_mm), _to_gltf(hint.target_mm)
    up = _to_gltf(hint.up)
    forward = g.unit(g.sub(tuple(target), tuple(eye)))
    right = g.unit(g.cross(forward, g.unit(tuple(up))))
    true_up = g.cross(right, forward)
    w, x, y, z = g.matrix_to_quat((right, true_up, g.scale(forward, -1.0)))
    distance = g.norm(g.sub(tuple(target), tuple(eye)))
    cameras.append({"name": "optical_view_camera", "type": "perspective",
                    "perspective": {"yfov": CAMERA_YFOV_RAD, "znear": distance * CAMERA_ZNEAR_FACTOR, "zfar": distance * CAMERA_ZFAR_FACTOR}})
    return b.node({"name": "optical_view_camera", "camera": len(cameras) - 1, "translation": eye, "rotation": [x, y, z, w],
                   "extras": {"opensci_view": view_kind, "kind": "CAMERA_HINT"}})


def _checked(node: int) -> int:
    checkpoint(0)
    return node


def scene_glb_bytes(scene: OpticalSimulationSceneV1) -> bytes:
    b = _GltfBuilder({m.medium_id: m.refractive_index for m in scene.display_media})
    colours = {p.path_id: tuple(p.display_rgb) for p in scene.beam_paths}
    # (each node is a checkpoint of the run's monitor: progress and cancellation while a large scene is drawn,
    # os_optics/run_control.py; nothing here depends on it)
    children = [_checked(_element_node(b, e)) for e in scene.elements]
    children += [_checked(_plane_node(b, p, colours)) for p in scene.observation_planes]
    children += [_checked(_beam_node(b, p)) for p in scene.beam_paths]
    if scene.axis_markers:
        children.append(_axis_marker_node(b, scene.axis_markers))
    view_extras = {
        "opensci_view": scene.view_kind, "view_notice": scene.view_notice, "scene_schema": scene.schema_id,
        "scene_schema_version": scene.schema_version, "scene_content_hash": scene.content_hash(),
        "project_hash": scene.project.project_hash.digest, "engine_input_hash": scene.engine_run.input_hash.digest,
        "engine_result_hash": None if scene.engine_run.result_hash is None else scene.engine_run.result_hash.digest,
        "engine_trace_hash": scene.engine_run.trace_hash, "length_unit_in_scene": "mm", "design_frame_up_axis": "+Z",
        "frame_mapping": FRAME_MAPPING_NOTE, "mesh_chord_tolerance_mm": CHORD_TOLERANCE_MM,
    }
    return _view_glb(b, children, view_extras, scene.camera_hint, root_name="optical_simulation_view", scene_name="Optical Simulation View")


def _axis_marker_node(b: _GltfBuilder, markers: Any) -> int:
    axes = g.Mesh()
    for marker in markers:
        axes.extend(g.tube([marker.start_mm, marker.end_mm], AXIS_TUBE_RADIUS_MM))
    mesh = b.mesh("optical_axes", [b.primitive(axes, b.material("axis_marker"))])
    return b.node({"name": "optical_axes", "mesh": mesh, "extras": {"opensci_view": "OPTICAL_SIMULATION_VIEW", "kind": "OPTICAL_AXIS_MARKERS"}})


def _view_glb(b: _GltfBuilder, children: list[int], view_extras: dict[str, Any], camera_hint: CameraHintV1, *, root_name: str,
              scene_name: str) -> bytes:
    """The GLB of a view: the root node (design frame -> glTF) over ``children``, and the camera of ``camera_hint``."""

    root = b.node({"name": root_name, "children": children, "rotation": list(ROOT_ROTATION_XYZW),
                   "scale": list(ROOT_SCALE), "extras": view_extras})
    cameras: list[dict[str, Any]] = []
    camera = _camera_node(b, camera_hint, cameras, view_extras["opensci_view"])
    while len(b.bin) % WORD_BYTES:
        b.bin.append(0)
    extensions = sorted({name for material in b.materials for name in material.get("extensions", ())})
    gltf = {
        **({"extensionsUsed": extensions} if extensions else {}),
        "asset": {"version": "2.0", "generator": f"{BUILDER_PACKAGE} {ENGINE_VERSION}", "extras": view_extras},
        "scene": 0,
        "scenes": [{"name": scene_name, "nodes": [root, camera], "extras": view_extras}],
        "cameras": cameras,
        "nodes": b.nodes,
        "meshes": b.meshes,
        "materials": b.materials,
        "accessors": b.accessors,
        "bufferViews": b.buffer_views,
        "buffers": [{"byteLength": len(b.bin)}],
    }
    checkpoint(0)
    # (the buffer is joined into the file as it is: no copy of it is held beside the file's bytes)
    return _pack_glb(gltf, b.bin)


#: The view kind of a layout (``os_optics.optical_view.layout``): the design without a run.
LAYOUT_VIEW_KIND = "OPTICAL_LAYOUT_VIEW"
#: Display size (mm), colour and opacity of a layout's nominal beam axes: thin grey lines, never a traced beam's colour.
NOMINAL_AXIS_TUBE_RADIUS_MM = 0.15
NOMINAL_AXIS_RGB = (0.55, 0.57, 0.62)
NOMINAL_AXIS_ALPHA = 0.8


def layout_glb_bytes(*, elements: Any, planes: Any, axis_markers: Any, nominal_axes: Any, display_media: Any, camera_hint: CameraHintV1,
                     extras: dict[str, Any]) -> bytes:
    """The GLB of a layout (``os_optics.optical_view.layout``): the elements and observation planes drawn exactly as a run's
    scene draws them (without footprints), the optical-axis markers, and each source's nominal beam axis as one thin grey
    polyline node (``nominal_axis:<source id>``, kind ``NOMINAL_BEAM_AXIS``). No beam path, footprint or power: nothing is
    traced. Every node says ``opensci_view: OPTICAL_LAYOUT_VIEW``; the root and the asset carry ``extras``."""

    b = _GltfBuilder({m.medium_id: m.refractive_index for m in display_media})
    children = [_checked(_element_node(b, e)) for e in elements]
    children += [_checked(_plane_node(b, p, {})) for p in planes]
    if axis_markers:
        children.append(_axis_marker_node(b, axis_markers))
    for axis in nominal_axes:
        points = [tuple(p) for p in axis["points_mm"]]
        if axis.get("display_extension_to_mm") is not None:
            points.append(tuple(axis["display_extension_to_mm"]))
        node: dict[str, Any] = {"name": f"nominal_axis:{axis['source_id']}",
                                "extras": {"kind": "NOMINAL_BEAM_AXIS", "source_element_id": axis["source_id"], "end": axis["end"],
                                           "length_unit": "mm"}}
        if len(points) >= 2:
            line = g.Mesh()
            line.extend(g.tube(points, NOMINAL_AXIS_TUBE_RADIUS_MM))
            node["mesh"] = b.mesh(f"nominal_axis:{axis['source_id']}",
                                  [b.primitive(line, b.material("nominal_beam_axis", NOMINAL_AXIS_RGB, NOMINAL_AXIS_ALPHA))])
        children.append(_checked(b.node(node)))
    for built in b.nodes:  # (the element and plane nodes are drawn as a run's scene draws them: they name this view)
        built.setdefault("extras", {})["opensci_view"] = LAYOUT_VIEW_KIND
    view_extras = {**extras, "opensci_view": LAYOUT_VIEW_KIND, "length_unit_in_scene": "mm", "design_frame_up_axis": "+Z",
                   "frame_mapping": FRAME_MAPPING_NOTE, "mesh_chord_tolerance_mm": CHORD_TOLERANCE_MM}
    return _view_glb(b, children, view_extras, camera_hint, root_name="optical_layout_view", scene_name="Optical Layout View")


def _pack_glb(gltf: dict[str, Any], binary: bytes | bytearray) -> bytes:
    text = json.dumps(gltf, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    text += b" " * (-len(text) % WORD_BYTES)
    checkpoint(0)
    pad = b"\x00" * (-len(binary) % WORD_BYTES)
    total = GLB_HEADER_BYTES + CHUNK_HEADER_BYTES + len(text) + CHUNK_HEADER_BYTES + len(binary) + len(pad)
    return b"".join((struct.pack("<III", GLB_MAGIC, GLB_VERSION, total), struct.pack("<II", len(text), CHUNK_JSON), text,
                     struct.pack("<II", len(binary) + len(pad), CHUNK_BIN), binary, pad))


class GlbFormatError(ValueError):
    """The bytes are not a structurally valid GLB of the kind this module writes."""


def parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Parse and structurally validate a GLB produced here (header, chunks, buffer/accessor bounds, references)."""
    first_chunk = GLB_HEADER_BYTES + CHUNK_HEADER_BYTES
    if len(data) < first_chunk:
        raise GlbFormatError("too short")
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or total != len(data):
        raise GlbFormatError("bad GLB header")
    json_len, json_type = struct.unpack_from("<II", data, GLB_HEADER_BYTES)
    if json_type != CHUNK_JSON or json_len % WORD_BYTES:
        raise GlbFormatError("first chunk must be 4-byte aligned JSON")
    gltf = json.loads(data[first_chunk : first_chunk + json_len].decode("utf-8"))
    offset = first_chunk + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, offset)
    if bin_type != CHUNK_BIN or offset + CHUNK_HEADER_BYTES + bin_len != len(data):
        raise GlbFormatError("second chunk must be the BIN chunk and end the file")
    binary = data[offset + CHUNK_HEADER_BYTES :]
    if gltf["asset"]["version"] != "2.0" or gltf["buffers"][0]["byteLength"] > len(binary):
        raise GlbFormatError("bad asset version or buffer length")
    for view in gltf["bufferViews"]:
        if view["byteOffset"] % WORD_BYTES or view["byteOffset"] + view["byteLength"] > gltf["buffers"][0]["byteLength"]:
            raise GlbFormatError("buffer view out of bounds or misaligned")
    sizes = {"SCALAR": 1, "VEC3": 3}
    for accessor in gltf["accessors"]:
        view = gltf["bufferViews"][accessor["bufferView"]]
        if accessor["count"] * sizes[accessor["type"]] * WORD_BYTES > view["byteLength"]:
            raise GlbFormatError("accessor exceeds its buffer view")
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            count = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
            if gltf["accessors"][prim["attributes"]["NORMAL"]]["count"] != count:
                raise GlbFormatError("NORMAL / POSITION count mismatch")
            if max(read_accessor(gltf, binary, prim["indices"]), default=-1) >= count:
                raise GlbFormatError("index out of range")
            if not 0 <= prim["material"] < len(gltf["materials"]):
                raise GlbFormatError("bad material reference")
    for node in gltf["nodes"]:
        if "mesh" in node and not 0 <= node["mesh"] < len(gltf["meshes"]):
            raise GlbFormatError("bad mesh reference")
        if any(not 0 <= c < len(gltf["nodes"]) for c in node.get("children", ())):
            raise GlbFormatError("bad child reference")
    return gltf, binary


def read_accessor(gltf: dict[str, Any], binary: bytes, index: int) -> list:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    n = accessor["count"] * {"SCALAR": 1, "VEC3": 3}[accessor["type"]]
    code = "f" if accessor["componentType"] == FLOAT else "I"
    values = struct.unpack_from(f"<{n}{code}", binary, view["byteOffset"])
    if accessor["type"] == "VEC3":
        return [tuple(values[i : i + 3]) for i in range(0, n, 3)]
    return list(values)
