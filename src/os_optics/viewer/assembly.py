"""The Experimental Lab Assembly of a verified Assembly Preview Package as a ``ViewerScene`` (in memory only).

Every part is its package GLB (part frame, metres) scaled to millimetres and placed exactly at ``pose_in_world``;
the beam is the package's beam reference axis; the customer's own hardware and the design observation planes are
markers at their design anchors. Nothing is re-posed, snapped or spaced here, and nothing is written anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._vendor.shared_contracts.assembly_preview import AssemblyPreviewPackageV1, parse_glb
from ..optical_view.colors import wavelength_display_rgb
from ..optical_view.glb import POLISHED_GLASS
from .model import ASSEMBLY_VIEW, ViewBeam, ViewerScene, ViewMarker, ViewMesh, ViewPart, place
from .palette import ASSEMBLY_CLASS_BY_CATEGORY, class_table

MM_PER_M = 1000.0
FLOAT32 = 5126
INDEX_TYPES = {5121: "<u1", 5123: "<u2", 5125: "<u4"}
TRIANGLES = 4
#: A design axis without one wavelength (a multi-line source): neutral grey, unlike every wavelength's display colour.
DEFAULT_BEAM_RGB = (0.55, 0.58, 0.62)


class AssemblyViewError(ValueError):
    """The package cannot be shown (it is verified first; a part geometry that cannot be decoded is refused)."""


def _axis_runs(axis: Any) -> tuple[tuple[str, np.ndarray], ...]:
    """A beam axis as runs of one display role: ``DESIGN_AXIS`` for its main-beam stretches, ``SECONDARY`` for the
    stretches only weak light reaches (``segment_roles``), in path order."""

    points = np.asarray(axis.points_mm, dtype=float)
    roles = [r.value for r in axis.segment_roles] if axis.segment_roles else None
    if not roles:
        return (("DESIGN_AXIS", points),)
    runs, start = [], 0
    for k in range(1, len(roles) + 1):
        if k == len(roles) or roles[k] != roles[start]:
            runs.append(("SECONDARY" if roles[start] == "SECONDARY" else "DESIGN_AXIS", points[start:k + 1]))
            start = k
    return tuple(runs)


def _accessor(document: dict[str, Any], binary: bytes, index: int, dtype: str, width: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"] * width
    return np.frombuffer(binary, dtype=dtype, count=count, offset=offset).astype(np.float64 if dtype == "<f4" else np.int64)


def part_meshes(data: bytes) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(positions mm, triangles)`` of every triangle primitive of one part GLB (part frame)."""

    document, binary = parse_glb(data)
    out = []
    for mesh in document.get("meshes", ()):
        for prim in mesh["primitives"]:
            if prim.get("mode", TRIANGLES) != TRIANGLES:
                raise AssemblyViewError("a part geometry holds a non-triangle primitive")
            positions = _accessor(document, binary, prim["attributes"]["POSITION"], "<f4", 3).reshape(-1, 3) * MM_PER_M
            if "indices" in prim:
                accessor = document["accessors"][prim["indices"]]
                dtype = INDEX_TYPES.get(accessor["componentType"])
                if dtype is None:
                    raise AssemblyViewError("a part geometry uses an unsupported index type")
                triangles = _accessor(document, binary, prim["indices"], dtype, 1).reshape(-1, 3)
            else:
                triangles = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
            if len(triangles) and (triangles.max() >= len(positions) or triangles.min() < 0):
                raise AssemblyViewError("a part geometry indexes past its vertices")
            out.append((positions, triangles))
    return out


def _chain_root(scene, part) -> str | None:
    """The part directly on the breadboard that (indirectly) carries ``part``: its support chain."""

    parts = {p.part_ref: p for p in scene.parts}
    ref, board = part.part_ref, scene.breadboard.part_ref
    while ref in parts and parts[ref].support is not None and parts[ref].support.parent_ref != board:
        ref = parts[ref].support.parent_ref
    return None if ref == board else ref


def assembly_view_scene(package: AssemblyPreviewPackageV1) -> ViewerScene:
    """The view of a verified package (``os_optics.assembly.load_assembly_preview`` or the client helper's
    in-memory preview). Reads only ``package.scene`` and ``package.files``."""

    if not isinstance(package, AssemblyPreviewPackageV1):
        raise AssemblyViewError("view_assembly needs a verified AssemblyPreviewPackageV1 (os_optics.assembly.load_assembly_preview)")
    scene = package.scene
    classes = class_table(ASSEMBLY_VIEW)
    parts = []
    for part in sorted(scene.parts, key=lambda p: (p.assembly_order, p.part_ref)):
        part_class = ASSEMBLY_CLASS_BY_CATEGORY.get(part.category.value, "OTHER_SUPPORT")
        style = classes[part_class]
        try:
            raw = part_meshes(package.files[part.geometry.asset_path])
        except (KeyError, ValueError) as exc:
            raise AssemblyViewError(f"part {part.part_ref}: its geometry cannot be read ({type(exc).__name__})") from None
        pose = part.pose_in_world
        finish = {"transmission": POLISHED_GLASS.transmission, "roughness": POLISHED_GLASS.roughness} if style.alpha < 1.0 else {}
        meshes = tuple(ViewMesh(positions=place(pos, pose.position_mm, pose.quaternion_wxyz), triangles=tri, rgb=style.rgb, alpha=style.alpha,
                                material=part_class, **finish) for pos, tri in raw)
        facts: dict[str, Any] = {"category": part.category.value, "assembly_order": part.assembly_order,
                                 "position_mm": list(pose.position_mm), "quaternion_wxyz": list(pose.quaternion_wxyz)}
        line = getattr(part, "commercial_line_ref", None)
        if line is not None:
            facts["line_ref"] = line
        if part.part_ref == scene.breadboard.part_ref and getattr(scene.breadboard, "supplied_by", None) is not None:
            facts["supplied_by"] = scene.breadboard.supplied_by  # the customer's own optical table: not supplied
        if part.design_binding is not None:
            facts["design_instance_id"] = part.design_binding.instance_id
        if part.support is not None:
            facts["carried_by"] = part.support.parent_ref
            facts["support_relation"] = part.support.relation.value
        chain = _chain_root(scene, part)
        if chain is not None:
            facts["chain"] = chain
        parts.append(ViewPart(part_id=part.part_ref, part_class=part_class, label=part.identity.display_name,
                              sku=part.identity.opensci_sku, meshes=meshes, facts=facts))
    reference = scene.beam_reference
    beams = []
    for axis in reference.beam_axes:
        rgb = wavelength_display_rgb(axis.wavelength_nm) if axis.wavelength_nm is not None else DEFAULT_BEAM_RGB
        facts: dict[str, Any] = {"basis": axis.basis.value}
        if axis.notes:
            facts["notes"] = [n.value for n in axis.notes]
        beams.append(ViewBeam(beam_id=axis.path_key, rgb=rgb, wavelength_nm=axis.wavelength_nm, runs=_axis_runs(axis), facts=facts))
    markers = []
    for anchor in reference.design_anchors:
        if getattr(anchor, "customer_supplied", False):
            markers.append(ViewMarker(marker_id=anchor.instance_id, marker_class="CUSTOMER_SUPPLIED",
                                      label=f"{anchor.instance_id} (your own hardware)", position=np.asarray(anchor.position_mm, dtype=float),
                                      facts={"design_instance_id": anchor.instance_id, "anchor_kind": anchor.anchor_kind.value,
                                             **_supply_facts(anchor)}))
        elif anchor.anchor_kind.value == "OBSERVATION":
            markers.append(ViewMarker(marker_id=anchor.instance_id, marker_class="OBSERVATION", label=f"{anchor.instance_id} (observation plane)",
                                      position=np.asarray(anchor.position_mm, dtype=float),
                                      facts={"design_instance_id": anchor.instance_id, "anchor_kind": anchor.anchor_kind.value}))
    lo = np.asarray(scene.bounds.min_mm, dtype=float)
    hi = np.asarray(scene.bounds.max_mm, dtype=float)
    board = scene.breadboard
    return ViewerScene(
        view_kind=ASSEMBLY_VIEW, frame=scene.frame, title="Experimental Lab Assembly", parts=tuple(parts), beams=tuple(beams),
        markers=tuple(markers), focus_min=lo, focus_max=hi, bounds_min=lo, bounds_max=hi,
        identity={"package_content_hash": package.content_hash, "breadboard": board.part_ref,
                  "breadboard_outline_mm": [list(board.outline_min_xy_mm), list(board.outline_max_xy_mm)],
                  "breadboard_top_z_mm": board.top_z_mm,
                  "optical_axis_height_mm": reference.optical_axis_height_mm},
    )


def _supply_facts(anchor) -> dict[str, str]:  # noqa: ANN001 - the vendored DesignAnchorV1
    """What the customer supplies at a customer-supplied anchor: role, reason, OpenSci product."""

    supply = getattr(anchor, "customer_supply", None)
    if supply is None:
        return {}
    facts = {"optical_role": supply.optical_role, "reason": supply.reason.value}
    if supply.opensci_sku:
        facts["opensci_sku"] = supply.opensci_sku
    if supply.reserved_envelope is not None:
        facts["reserved_envelope"] = "CYLINDER"
    return facts
