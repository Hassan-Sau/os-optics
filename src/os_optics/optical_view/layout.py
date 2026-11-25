"""The layout of a Project without a run: its optical elements, sources and observation planes at their design poses, and the
nominal beam axis of each source -- for a viewer that shows the design before (or instead of) a calculation.

Nothing is traced. The elements and planes are drawn exactly as a run's Optical Simulation View draws them (the same
public catalog geometry and declared models, ``builder``), from the Project and the Engine request its lowering builds
(``os_optics.local.lowering``); observation planes carry no footprint, and there is no beam path, power, status or
target: a layout is never a result.

The nominal beam axes are computed outside the view, on the lowered surfaces (``os_optics.local.nominal_axes``): the view
only draws the points it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._vendor.shared_contracts.optical_design import ProjectDocumentV2, validated_project_ref
from .._vendor.shared_contracts.optical_view import OpticalViewSettingsV1
from . import geometry as g
from .builder import (
    _bounds_and_camera,
    _display_media,
    _element_extent_points,
    _elements,
    _lowered_models,
    _planes,
    _plane_extent_points,
    axis_markers,
)
from .glb import LAYOUT_VIEW_KIND, layout_glb_bytes

LAYOUT_SCHEMA = "opensci.optical_layout_view"
LAYOUT_SCHEMA_VERSION = "1.0"
LAYOUT_NOTICE = ("Layout of the design: optical elements, sources and observation planes at their design poses, and each source's "
                 "nominal beam axis. Nothing is traced: no beam, power or target result.")
#: The GLB file of a layout reply (``json_api.project_layout``).
LAYOUT_GLB_FILE = "optical_layout.glb"


@dataclass(frozen=True)
class ProjectLayout:
    """A layout: its JSON document and its GLB."""

    document: dict[str, Any]
    glb: bytes


def build_project_layout(*, project: ProjectDocumentV2, catalog: Any, request: Any, compiled: dict[str, Any], nominal_axes: list[dict[str, Any]],
                         nominal_axes_unavailable: str | None = None, settings: OpticalViewSettingsV1 | None = None,
                         model_kind: str | None = None) -> ProjectLayout:
    """The layout of ``project`` (module docstring) from the Engine request its lowering built and its compiled elements
    (``LoweredProject.request`` / ``.elements`` by instance id), and the sources' nominal beam axes
    (``os_optics.local.nominal_axes``); ``catalog`` the release its products pin (None without catalog products).
    Deterministic: the same inputs give the same document and GLB bytes."""

    settings = settings or OpticalViewSettingsV1()
    d = settings.coordinate_decimals
    models = _lowered_models(project, catalog, model_kind)
    elements, used_catalog = _elements(project, catalog, models, request, d, compiled)
    cfg = request.configuration.numerical
    planes = _planes(project, None, d, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor)
    markers = axis_markers(elements, settings, d)
    axes = nominal_axes
    media = _display_media(request)
    points: list[g.Vec3] = []
    for element in elements:
        points.extend(_element_extent_points(element))
    for plane in planes:
        points.extend(_plane_extent_points(plane))
    for axis in axes:
        points.extend(tuple(p) for p in axis["points_mm"])
        if axis["display_extension_to_mm"] is not None:
            points.append(tuple(axis["display_extension_to_mm"]))
    bounds, camera = _bounds_and_camera(points or [(0.0, 0.0, 0.0)], d)
    project_ref = validated_project_ref(project)

    def dumped(model: Any) -> dict[str, Any]:
        return model.model_dump(mode="json", by_alias=True, exclude_none=True)

    document: dict[str, Any] = {
        "schema": LAYOUT_SCHEMA,
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "view_kind": LAYOUT_VIEW_KIND,
        "view_notice": LAYOUT_NOTICE,
        "frame": "PROJECT_DESIGN_FRAME",
        "length_unit": "mm",
        "up_axis": "+Z",
        "project": dumped(project_ref),
        "catalog_version": catalog.catalog_version if used_catalog and catalog is not None else None,
        "settings": dumped(settings),
        "elements": [dumped(e) for e in elements],
        "observation_planes": [dumped(p) for p in planes],
        "axis_markers": [dumped(m) for m in markers],
        "display_media": [dumped(m) for m in media],
        "nominal_axes": axes,
        "nominal_axes_unavailable": nominal_axes_unavailable,
        "bounds": dumped(bounds),
        "camera_hint": dumped(camera),
    }
    glb = layout_glb_bytes(elements=elements, planes=planes, axis_markers=markers, nominal_axes=axes, display_media=media, camera_hint=camera,
                           extras={"view_notice": LAYOUT_NOTICE, "layout_schema": LAYOUT_SCHEMA, "layout_schema_version": LAYOUT_SCHEMA_VERSION,
                                   "project_hash": project_ref.project_hash.digest})
    return ProjectLayout(document=document, glb=glb)


__all__ = ["LAYOUT_GLB_FILE", "LAYOUT_SCHEMA", "LAYOUT_SCHEMA_VERSION", "ProjectLayout", "build_project_layout"]
