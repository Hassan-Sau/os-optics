"""Build the renderer-independent ``OpticalSimulationSceneV1`` (scene schema 3.0) from a pure-optical Project, the public
optical catalog (when the Project uses catalog products) and an Engine run of that Project.

Inputs and authority:

* ``ProjectDocumentV2``: which optical elements exist, their roles, public identities and nominal poses, and the
  observation / target planes.
* ``PublicOpticalCatalogV2``: the public design geometry of each catalog product and the simulation model the run
  lowered (its surfaces, media and coatings, or its polarization element) as optical annotations of one generic
  ``OPTIC_BODY``. Nothing else about a product is read.
* the Engine ``ValidationRequest`` that the local runtime built from that Project (the Project -> Engine lowering is not
  done here), its ``ValidationResult``, and the ray paths the Engine recorded while tracing it
  (``os_optics.validate_with_ray_paths`` / ``trace_ray_paths``): every ray point and every beam footprint comes
  from the Engine. The view never traces, refracts or propagates anything itself.

Fail-closed bindings (``OpticalViewBindingError``):

* the recorded trace is the result's trace (equal input and trace hashes) and the result is unaltered (its quantized
  hash is recomputed and compared with its manifest);
* the Engine request holds exactly the Project's components and observation planes (same ids, roles, poses,
  attachments and absorbing flags), so a view cannot draw rays shaped by an element the Project does not show;
* each catalog product's Engine element is exactly the Engine's compilation of the public model the run lowered
  (``os_optics.local.compile.compile_element``, the single source of truth of the public model -> Engine element
  lowering): the same surfaces in the element frame (ids ``S1..Sn`` or ``element``, geometry, apertures, interactions,
  coatings, media) and, for every catalog material it refracts through, the same Engine material in the request. The
  selected model is the one the run lowered (``resolve_simulated_parts`` with the run's ``model_kind``), never an
  assumed single model;
* each component with a declared model (``optical_model_id``) is, the same way, exactly the Engine's compilation of that
  model (``compile.compile_declared``) with the Project's materials (``project:<MATERIAL_ID>``); a ``COMPOUND`` unit's
  Engine component holds exactly its members' compiled surfaces at their poses;
* each source component carries one source whose ``source_id`` is the component ``instance_id`` and no surfaces; only
  sources emit; every beam starts at a Project source and every path id derives from Project ids.

Observation planes: every Engine arrival is one footprint, keyed by its path and its ``crossing_index`` (a beam path that
comes back through a plane after a fold, a double pass or a retro-reflection has one footprint per crossing), and each
plane carries the Engine's absorbing semantics (``absorbing_extent``).

Numbers: coordinates are rounded to ``settings.coordinate_decimals``; a footprint's power is quantized exactly as the
Engine quantizes its result for the result hash (the run's ``hash_float_sig_digits`` / ``hash_float_abs_floor``), so the
last-bit floating-point noise two runtimes may show on a product of many coating / Fresnel factors never changes the
scene bytes while their result hashes agree.

Declared models are drawn like catalog products (``OPTIC_BODY``, ``basis = PROJECT_DECLARED_MODEL``, labelled by their
model id); a ``COMPOUND`` unit is one ``OPTICAL_UNIT`` element holding a body per member at its pose in the unit frame.

Only ids, enums, public catalog fields and numbers reach the scene. Free text of the Project (custom part descriptions,
model display names, parameter values) is never copied.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from ..run_control import checkpoint
from .._vendor.shared_contracts.common.geometry import PoseV1
from .._vendor.shared_contracts.common.hashes import HashProfile, HashRef
from .._vendor.shared_contracts.design_catalog import (
    BoxGeometryV1,
    CylinderGeometryV1,
    DesignGeometryV1,
    JoinedBodiesModelV1,
    ModelFidelity,
    PolarizationElementModelV1,
    PublicOpticalCatalogV2,
    PublicOpticalProductV2,
    SurfaceSequenceModelV1,
    SurfaceSetModelV1,
    TracedBranches,
)
from .._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, RectangularApertureV1
from .._vendor.shared_contracts.optical_design import OpticalRoleV1, ProjectDocumentV2, resolve_simulated_parts, validated_project_ref
from .._vendor.shared_contracts.optical_view import (
    DisplayMediumV1,
    APERTURE_STOP_SURFACE_ID,
    BLOCKED_POWER_SHARE,
    ATTENUATOR_SURFACE_ID,
    FARADAY_ROTATOR_SURFACE_ID,
    GRATING_SURFACE_ID,
    OpticFacetGroupV1,
    OpticFacetsV1,
    OpticGratingV1,
    POLARIZATION_ELEMENT_SURFACE_ID,
    THIN_LENS_SURFACE_ID,
    GeometryBasisV1,
    OpticalUnitGeometryV1,
    UnitMemberV1,
    UnitPortV1,
    SECONDARY_BRANCH_POWER_SHARE,
    AxisMarkerV1,
    BeamDisplayRoleV1,
    BeamFootprintV1,
    BeamPathV1,
    CameraHintV1,
    ElementMarkerGeometryV1,
    EngineRunRefV1,
    ObservationAbsorptionV1,
    ObservationPlaneKindV1,
    ObservationPlaneV1,
    OpticalSimulationSceneV1,
    OpticalViewSettingsV1,
    OpticBodyGeometryV1,
    OpticSurfaceRoleV1,
    OpticSurfaceV1,
    PolarizationAxisV1,
    PupilRaySelectionV1,
    RayEndV1,
    SolidBodyV1,
    SolidFaceV1,
    RayPolylineV1,
    SceneBoundsV1,
    SceneElementV1,
    EmittingAreaV1,
    SourceEmitterGeometryV1,
    ViewBuilderRefV1,
    PlaneArrivalV1,
    coating_peak,
    designed_arrivals,
)

from ..contracts import EngineeringComponentSnapshot
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..manifest import hash_quantized, quantize_float
from ..optics.apertures import reach as region_reach, ring as region_ring
from ..optics.materials.index import build_evaluator
from ..scene.model import ApertureG
from ..validation.ray_paths import BRANCH_SUFFIX_PATTERN, RayPathTrace, trace_ray_paths, validate_with_ray_paths
from . import geometry as g
from . import optic_mesh
from .surface_mesh import CHORD_TOLERANCE_MM
from .colors import wavelength_display_rgb
from .identity import BUILDER_PACKAGE, BUILDER_VERSION

Progress = Callable[[str], None]
PublicModel = SurfaceSequenceModelV1 | SurfaceSetModelV1 | PolarizationElementModelV1 | JoinedBodiesModelV1

# Binding tolerances between the Project nominal pose and the Engine component pose (same numbers, re-serialized).
POSITION_TOLERANCE_MM = 1e-6
ROTATION_TOLERANCE = 1e-9
# Symbolic marker sizes of sources and custom parts (display only).
SOURCE_MARKER_LENGTH_MM = 10.0
D_LINE_NM = 587.56
DISPLAY_INDEX_DECIMALS = 4
DISPLAY_MEDIUM_PREFIXES = ("catalog:", "project:")
DEFAULT_MARKER_RADIUS_MM = 12.7
MARKER_THICKNESS_MM = 1.0
UNBOUNDED_PLANE_HALF_EXTENT_MM = 12.7
CAMERA_VIEW_DIRECTION = (-0.55, -1.0, 0.65)  # eye offset direction from the scene centre, design frame
CAMERA_DISTANCE_FACTOR = 1.3
# What each traced-branch declaration of a published surface does, as a scene surface role.
SURFACE_ROLE_BY_TRACED = {
    TracedBranches.TRANSMITTED: OpticSurfaceRoleV1.REFRACT,
    TracedBranches.REFLECTED: OpticSurfaceRoleV1.REFLECT,
    TracedBranches.TRANSMITTED_AND_REFLECTED: OpticSurfaceRoleV1.SPLIT,
}


class OpticalViewBindingError(ValueError):
    """The Project, the public catalog and the Engine run do not describe the same optical system (no scene is drawn)."""


def _round(value: float, decimals: int) -> float:
    r = round(float(value), decimals)
    return r + 0.0  # fold -0.0


def _rv(v, decimals: int) -> tuple[float, float, float]:
    return (_round(v[0], decimals), _round(v[1], decimals), _round(v[2], decimals))


def _quat(pose) -> g.Quat:
    return tuple(float(c) for c in pose.quaternion_wxyz)  # type: ignore[return-value]


def _vec(v) -> g.Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _same_pose(project_pose: PoseV1, engine_pose, where: str) -> None:
    if g.norm(g.sub(_vec(project_pose.position_mm), _vec(engine_pose.position_mm))) > POSITION_TOLERANCE_MM or not g.same_rotation(
        _quat(project_pose), _quat(engine_pose), ROTATION_TOLERANCE
    ):
        raise OpticalViewBindingError(f"{where}: the Engine pose differs from the Project nominal pose")


def _decimal_string(value: float) -> str:
    text = format(Decimal(repr(float(value))), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


# ------------------------------------------------------------------------------------------ catalog binding


def request_design_point(request: ValidationRequest) -> Any:
    """The design-point tolerances the elements of ``request`` were compiled under: its numerical configuration's
    (``os_optics.local.design_point.DesignPointTolerances``)."""

    from ..local.design_point import DesignPointTolerances  # lazy: os_optics.local imports this package

    return DesignPointTolerances.of(request.configuration.numerical)


def compile_declared_element(instance_id: str, project: ProjectDocumentV2, design_point: Any = None) -> Any:
    """The Engine element the local runtime lowers the component ``instance_id``'s declared model to:
    ``os_optics.local.compile.compile_declared`` with the Project's own materials and coatings, under ``design_point``
    (the tolerances of the run's request; the Engine defaults when not given)."""

    from ..local.compile import ProjectResolver, compile_declared, positioned_for  # lazy: os_optics.local imports this package
    from ..local.design_point import DEFAULT_DESIGN_POINT

    (component,) = [c for c in project.components if c.instance_id == instance_id]
    declared = project.optical_model(component.optical_model_id)  # type: ignore[arg-type]
    resolver = ProjectResolver(project, design_point or DEFAULT_DESIGN_POINT)
    return positioned_for(compile_declared(instance_id, declared, resolver), component)  # type: ignore[arg-type]


def compile_catalog_element(instance_id: str, product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, model: PublicModel,
                            component: Any = None, design_point: Any = None, *, environment: Any = None) -> Any:
    """The Engine element the local runtime lowers ``product`` to: ``os_optics.local.compile.compile_element``, the
    single source of truth of the public model -> Engine element compilation (``surfaces`` in the element frame and the
    Engine ``materials`` they refract through), for the published ``model`` the run selected (a product publishes one
    model per kind, so its ``model_kind`` selects it), under ``design_point`` (the tolerances of the run's request; the
    Engine defaults when not given)."""

    from ..local.compile import compile_element, positioned_for  # lazy: os_optics.local imports this package
    from ..local.design_point import DEFAULT_DESIGN_POINT

    compiled = compile_element(instance_id, product, catalog, model_kind=model.model_kind, environment=environment,
                               design_point=design_point or DEFAULT_DESIGN_POINT)
    return compiled if component is None else positioned_for(compiled, component)


_DIFFERENCE_EXCERPT_CHARS = 80  # characters of each value quoted in a binding refusal


def _first_difference(got: Any, want: Any, path: str = "") -> str:
    """A short JSON path to the first place two JSON values differ, with both values."""

    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            if got.get(key) != want.get(key):
                return _first_difference(got.get(key), want.get(key), f"{path}.{key}")
    elif isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        for k, (a, b) in enumerate(zip(got, want, strict=True)):
            if a != b:
                return _first_difference(a, b, f"{path}[{k}]")
    text = repr(got)[:_DIFFERENCE_EXCERPT_CHARS], repr(want)[:_DIFFERENCE_EXCERPT_CHARS]
    return f"{path.lstrip('.') or '<value>'}: Engine {text[0]}, public model {text[1]}"


def _bind_catalog_element(instance_id: str, engine: EngineeringComponentSnapshot, product: PublicOpticalProductV2, model: PublicModel,
                          catalog: PublicOpticalCatalogV2, engine_materials: dict[str, Any], component: Any = None,
                          design_point: Any = None, *, environment: Any = None) -> Any:
    """The Engine element of a catalog product is exactly the Engine compilation of its public model (surfaces and the
    catalog materials they use), so the view never draws rays shaped by anything but the published product. Returns the
    compiled element."""

    def refuse(why: str) -> OpticalViewBindingError:
        return OpticalViewBindingError(f"component {instance_id!r} ({product.optical_sku}): the Engine element is not its public model: {why}")

    try:
        compiled = compile_catalog_element(instance_id, product, catalog, model, component, design_point, environment=environment)
    except ValueError as exc:
        raise refuse(f"the public model does not compile ({exc})") from None
    compiled_model = getattr(compiled, "model", None)
    if compiled_model is not None and compiled_model != model:
        raise refuse(f"the Engine compiled its {compiled_model.model_kind} model, the run lowered its {model.model_kind} model")
    _same_compilation(compiled, engine, engine_materials, refuse, model.model_kind)
    return compiled


def _bind_declared_element(instance_id: str, engine: EngineeringComponentSnapshot, project: ProjectDocumentV2,
                           engine_materials: dict[str, Any], design_point: Any = None) -> Any:
    """The Engine element of a declared model is exactly the Engine compilation of that model (surfaces and the Project's
    materials they use). Returns the compiled element."""

    model_id = next(c.optical_model_id for c in project.components if c.instance_id == instance_id)

    def refuse(why: str) -> OpticalViewBindingError:
        return OpticalViewBindingError(f"component {instance_id!r} (model {model_id}): the Engine element is not its declared model: {why}")

    try:
        compiled = compile_declared_element(instance_id, project, design_point)
    except ValueError as exc:
        raise refuse(f"the declared model does not compile ({exc})") from None
    _same_compilation(compiled, engine, engine_materials, refuse, compiled.model_kind)
    return compiled


def _same_compilation(compiled: Any, engine: EngineeringComponentSnapshot, engine_materials: dict[str, Any],
                      refuse: Callable[[str], OpticalViewBindingError], model_kind: str) -> None:
    want = [surface.model_dump(mode="json") for surface in compiled.surfaces]
    got = [surface.model_dump(mode="json") for surface in engine.surfaces]
    want_ids, got_ids = [s["surface_id"] for s in want], [s["surface_id"] for s in got]
    if got_ids != want_ids:
        raise refuse(f"surfaces {got_ids} instead of the compiled {model_kind} surfaces {want_ids}")
    for got_surface, want_surface in zip(got, want, strict=True):
        if got_surface != want_surface:
            raise refuse(f"surface {got_surface['surface_id']} differs ({_first_difference(got_surface, want_surface)})")
    for material in getattr(compiled, "materials", ()):
        have = engine_materials.get(material.material_id)
        if have is None:
            raise refuse(f"material {material.material_id!r} is not defined in the Engine request")
        if have.model_dump(mode="json") != material.model_dump(mode="json"):
            raise refuse(f"material {material.material_id!r} differs ({_first_difference(have.model_dump(mode='json'), material.model_dump(mode='json'))})")


def _lowered_models(project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2 | None, model_kind: str | None) -> dict[str, tuple[PublicOpticalProductV2, PublicModel]]:
    """Each catalog component's product and the published model the run lowered (``resolve_simulated_parts``: the same
    selection the local runtime records in ``simulated_parts``), by component ``instance_id``."""

    if catalog is None:
        if any(c.product is not None for c in project.components):
            raise OpticalViewBindingError("the Project uses catalog products but no public optical catalog was given")
        return {}
    try:
        parts = resolve_simulated_parts(project, catalog, model_kind=model_kind)
    except ValueError as exc:
        raise OpticalViewBindingError(str(exc)) from exc
    selected: dict[str, tuple[PublicOpticalProductV2, PublicModel]] = {}
    for part in parts:
        product = catalog.get(part.optical_sku)
        assert product is not None  # resolve_simulated_parts already failed closed
        ref = part.simulation_model
        (model,) = [m for m in product.simulation_models if (m.model_kind, m.model_version) == (ref.model_kind, ref.model_version)]
        selected[part.instance_id] = (product, model)
    return selected


def _bind(project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2 | None, models: dict[str, tuple[PublicOpticalProductV2, PublicModel]],
          request: ValidationRequest, result: ValidationResult, trace: RayPathTrace, same_evaluation: bool = False) -> dict[str, Any]:
    """Every binding of the module docstring; returns the compiled element of each catalog and declared component.
    ``same_evaluation``: ``result`` is the object the caller's own evaluation produced, so it is not serialized and hashed
    again against its manifest (a result brought from elsewhere is)."""

    manifest = result.manifest
    if manifest is None or not manifest.input_hash:
        raise OpticalViewBindingError("the Engine result has no manifest")
    if not same_evaluation:
        cfg = request.configuration.numerical
        body = result.model_copy(update={"manifest": None}).model_dump(mode="json")
        if hash_quantized(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor) != manifest.result_hash:
            raise OpticalViewBindingError("the Engine result does not match its own manifest result_hash")
    if trace.status != "TRACED" or trace.trace_hash is None:
        raise OpticalViewBindingError(f"the Engine could not trace the request: {trace.message}")
    if trace.input_hash != manifest.input_hash:
        raise OpticalViewBindingError("the ray trace and the Engine result are for different Engine inputs")
    if trace.trace_hash != manifest.trace_hash:
        raise OpticalViewBindingError("the ray trace does not reproduce the Engine result's trace hash")
    engine_components = {c.instance_id: c for c in request.project.components}
    project_components = {c.instance_id: c for c in project.components}
    if set(engine_components) != set(project_components):
        raise OpticalViewBindingError(
            "the Engine request and the Project place different components "
            f"(Engine only: {sorted(set(engine_components) - set(project_components))}, "
            f"Project only: {sorted(set(project_components) - set(engine_components))})"
        )
    for instance_id, component in project_components.items():
        engine = engine_components[instance_id]
        if engine.role != component.optical_role.value:
            raise OpticalViewBindingError(f"component {instance_id!r}: Engine role {engine.role!r} != Project role {component.optical_role.value!r}")
        _same_pose(component.nominal_pose, engine.pose, f"component {instance_id!r}")
    engine_observations = {o.observation_id: o for o in request.project.observations}
    project_observations = {o.observation_id: o for o in project.observations}
    if set(engine_observations) != set(project_observations):
        raise OpticalViewBindingError("the Engine request and the Project declare different observation planes")
    for observation_id, observation in project_observations.items():
        engine_obs = engine_observations[observation_id]
        if engine_obs.device_id != observation.device_instance_id:
            raise OpticalViewBindingError(f"observation {observation_id!r}: attached to a different component in the Engine request")
        _same_pose(observation.pose, engine_obs.pose, f"observation {observation_id!r}")
        if engine_obs.absorbing != observation.absorbing:
            raise OpticalViewBindingError(f"observation {observation_id!r}: the Engine plane's absorbing flag differs from the Project")
    engine_materials = {m.material_id: m for m in request.project.materials}
    # the medium every ray outside the elements travels in: the Project's environment, as the native lowering states it
    from ..local.compile import ENGINE_AIR, environment_air  # lazy: os_optics.local imports this package

    air = environment_air(project.effective_environment)
    if ENGINE_AIR not in engine_materials:
        raise OpticalViewBindingError(f"material {ENGINE_AIR!r} is not defined in the Engine request: the Project's environment states it")
    if engine_materials[ENGINE_AIR] != air:
        raise OpticalViewBindingError(f"material {ENGINE_AIR!r} differs from the air of the Project's environment: "
                                      f"{_first_difference(engine_materials[ENGINE_AIR].model_dump(mode='json'), air.model_dump(mode='json'))}")
    compiled: dict[str, Any] = {}
    design_point = request_design_point(request)  # (the binding compares compilations under the request's own tolerances)
    for instance_id, component in project_components.items():
        engine = engine_components[instance_id]
        if component.optical_role is OpticalRoleV1.SOURCE:
            emission = engine.source if engine.source is not None else engine.emitter
            silent_sample = emission is None and engine.sample_emission is not None  # (a sample its excitation leaves dark)
            if (emission is None and not silent_sample) or (emission is not None and emission.source_id != instance_id) or engine.surfaces:
                raise OpticalViewBindingError(
                    f"source {instance_id!r}: the Engine component must carry one source with source_id {instance_id!r} and no surfaces"
                )
        elif engine.source is not None or engine.emitter is not None:
            raise OpticalViewBindingError(f"component {instance_id!r}: only a Project source may emit light in the Engine request")
        if component.product is not None:
            product, model = models[instance_id]
            compiled[instance_id] = _bind_catalog_element(instance_id, engine, product, model, catalog, engine_materials, component,  # type: ignore[arg-type]
                                                          design_point, environment=project.effective_environment)
        elif component.optical_model_id is not None:
            compiled[instance_id] = _bind_declared_element(instance_id, engine, project, engine_materials, design_point)
    sources = {c.instance_id for c in project.components if c.optical_role is OpticalRoleV1.SOURCE}
    for path in trace.paths:
        if path.source_component_id not in sources:
            raise OpticalViewBindingError(f"path {path.path_id!r} starts at {path.source_component_id!r}, which is not a Project source")
        if not re.fullmatch(re.escape(path.source_component_id) + r"#\d+" + BRANCH_SUFFIX_PATTERN, path.path_id):
            raise OpticalViewBindingError(f"path {path.path_id!r} is not derived from its Project source {path.source_component_id!r}")
    return compiled


# -------------------------------------------------------------------------------------------------- elements


def _floats(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _design_shape(product: PublicOpticalProductV2) -> dict[str, Any]:
    """The public design geometry as ``OPTIC_BODY`` extent fields (published decimals as floats)."""

    return _shape_fields(product.design_geometry)


def _shape_fields(body: DesignGeometryV1) -> dict[str, Any]:
    """A design geometry (or a unit's envelope) as ``OPTIC_BODY`` / ``OPTICAL_UNIT`` extent fields (decimals as floats)."""

    if isinstance(body, CylinderGeometryV1):
        return {"shape": "CYLINDER", "diameter_mm": float(body.diameter_mm), "z_min_mm": float(body.z_min_mm), "z_max_mm": float(body.z_max_mm)}
    if isinstance(body, BoxGeometryV1):
        return {"shape": "BOX", "center_mm": _floats(body.center_mm), "half_extents_mm": _floats(body.half_extents_mm)}
    return {"shape": "CONVEX_POLYHEDRON", "vertices_mm": tuple(_floats(v) for v in body.vertices_mm)}


def _profile_fields(surface: Any) -> dict[str, Any]:
    """Profile and clear-aperture fields of a published surface (either surface model kind)."""

    fields: dict[str, Any] = {
        "radius_mm": None if surface.radius_mm is None else float(surface.radius_mm),
        "conic": None if surface.conic is None else float(surface.conic),
        "aspheric_coefficients": None if surface.aspheric_coefficients is None else _floats(surface.aspheric_coefficients),
        "linear_sag_coefficient": None if surface.linear_sag_coefficient is None else float(surface.linear_sag_coefficient),
        "cylinder_axis": surface.cylinder_axis,
        "sag": surface.sag,
        "clear_radius_mm": None if surface.clear_radius_mm is None else float(surface.clear_radius_mm),
        "clear_half_widths_mm": None if surface.clear_half_widths_mm is None else _floats(surface.clear_half_widths_mm),
        "aperture_center_mm": None if surface.aperture_center_mm is None else _floats(surface.aperture_center_mm),
    }
    from ..local.compile import clear_shape_region  # (the local runtime imports the view builder)

    centre = _floats(surface.aperture_center_mm) if surface.aperture_center_mm is not None else (0.0, 0.0)
    if getattr(surface, "clear_shape", None) is not None:
        fields["clear_outline_mm"] = _outline_ring(clear_shape_region(surface.clear_shape, centre, surface.clear_radius_mm,
                                                                      surface.clear_half_widths_mm))
    if getattr(surface, "holes", None):
        fields["hole_outlines_mm"] = tuple(_outline_ring(_hole_region(hole)) for hole in surface.holes)
    return fields


def _hole_region(hole: Any) -> ApertureG:
    """A surface's hole as a region in its local X / Y."""

    from ..local.compile import hole_region

    return hole_region(hole)


#: Decimal places of a drawn outline ring's points (mm).
OUTLINE_RING_DECIMALS = 9


def _outline_ring(region: ApertureG) -> tuple[tuple[float, float], ...]:
    """A region's outer boundary as a counter-clockwise ring: corners exactly, round parts by chords within the view's
    drawing tolerance."""

    points = region_ring(region, g.circle_segments(region_reach(region), CHORD_TOLERANCE_MM))
    return tuple((round(u, OUTLINE_RING_DECIMALS), round(v, OUTLINE_RING_DECIMALS)) for u, v in points)


def _edge_fields(surface: Any) -> dict[str, Any]:
    """A surface-sequence surface's own mechanical edge (none: it reaches the outline)."""

    if surface.edge_radius_mm is not None:
        return {"edge_radius_mm": float(surface.edge_radius_mm)}
    if surface.edge_half_widths_mm is not None:
        return {"edge_half_widths_mm": _floats(surface.edge_half_widths_mm)}
    return {}


def _thin_shape(model: Any) -> dict[str, Any]:
    """The body of a declared element of zero thickness without a stated body: its outline at its plane."""

    z = float(model.plane_z_mm) if hasattr(model, "plane_z_mm") else float(model.surfaces[0].vertex_z_mm)
    if getattr(model, "edge_half_widths_mm", None) is not None:
        hx, hy = _floats(model.edge_half_widths_mm)
        return {"shape": "BOX", "center_mm": (0.0, 0.0, z), "half_extents_mm": (hx, hy, 0.0)}
    return {"shape": "CYLINDER", "diameter_mm": 2.0 * float(model.edge_radius_mm), "z_min_mm": z, "z_max_mm": z}


def optic_body(product: PublicOpticalProductV2, model: PublicModel, catalog: PublicOpticalCatalogV2) -> OpticBodyGeometryV1 | OpticalUnitGeometryV1:
    """The ``OPTIC_BODY`` of a catalog product drawn with its published ``model``: the public design geometry plus the
    model's surfaces (or polarization element plane) and, for a surface set, its solid body, as optical annotations.
    Numbers are the published decimals as floats; nothing is derived. A polarizing prism (a ``JOINED_BODIES`` model) is the
    ``OPTICAL_UNIT`` of its bodies (``catalog_unit``)."""

    def coating_kind(surface: Any):
        coating = catalog.coating(surface.coating_id) if surface.coating_id is not None else None
        return None if coating is None else coating.kind

    if isinstance(model, JoinedBodiesModelV1):
        return catalog_unit(product, model, coating_kind)
    common: dict[str, Any] = dict(
        **_design_shape(product), category=product.category, form=product.form, reference_axis=product.frame.reference_axis,
        model_kind=model.model_kind, fidelity=model.fidelity,
    )
    return _body(common, model, coating_kind)


#: A joined body is stated in its element's frame: its pose in the unit is the identity.
_IDENTITY_POSE = PoseV1()
#: The longest label of a unit member (the scene contract's limit).
MEMBER_LABEL_MAX_CHARS = 200


def catalog_unit(product: PublicOpticalProductV2, model: JoinedBodiesModelV1, coating_kind: Callable[[Any], Any]) -> OpticalUnitGeometryV1:
    """A catalog polarizing prism as an ``OPTICAL_UNIT`` (basis ``PUBLIC_CATALOG_DESIGN_GEOMETRY``): its public design geometry
    as the envelope and each body (a ``SURFACE_SET`` with its solid, drawn with the product's category and form) at the identity
    pose (the bodies are stated in the element frame)."""

    members = []
    for body in model.members:
        common: dict[str, Any] = dict(**_design_shape(product), category=product.category, form=product.form,
                                      reference_axis=product.frame.reference_axis, model_kind=body.model.model_kind, fidelity=body.model.fidelity)
        members.append(UnitMemberV1(member_id=body.member_id, label=f"{product.display_name} body {body.member_id}"[:MEMBER_LABEL_MAX_CHARS], pose=_IDENTITY_POSE,
                                    body=_body(common, body.model, coating_kind)))
    return OpticalUnitGeometryV1(**_design_shape(product), members=tuple(members), basis=GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY,
                                 category=product.category, form=product.form)


def declared_body(element: Any, project: ProjectDocumentV2) -> OpticBodyGeometryV1:
    """The ``OPTIC_BODY`` of a compiled declared model (not a compound unit): its stated or derived design geometry (a body
    of zero thickness at its plane when it has none) plus the model's surfaces, as for a catalog product, with ``basis``
    ``PROJECT_DECLARED_MODEL``."""

    identity, model = element.identity, element.model
    shape = _shape_fields(identity.design_geometry) if identity.design_geometry is not None else _thin_shape(model)
    frame = identity.frame
    fidelity = getattr(model, "fidelity", None)
    if fidelity is None and model.model_kind in ("IDEAL_THIN_LENS", "TABULATED_THIN_LENS", "FARADAY_ROTATOR", "DIFFRACTION_GRATING"):
        fidelity = ModelFidelity.IDEAL_THIN_ELEMENT
    common: dict[str, Any] = dict(**shape, reference_axis=frame.reference_axis if frame is not None else None, model_kind=model.model_kind,
                                  fidelity=fidelity, basis=GeometryBasisV1.PROJECT_DECLARED_MODEL)
    coatings = {c.coating_id: c for c in project.model_coatings or ()}

    def coating_kind(surface: Any):
        coating = coatings.get(surface.coating_id) if surface.coating_id is not None else None
        return None if coating is None else coating.kind

    if model.model_kind == "APERTURE_STOP":
        opening = ({"clear_radius_mm": float(model.opening_radius_mm)} if model.opening_radius_mm is not None
                   else {"clear_half_widths_mm": _floats(model.opening_half_widths_mm)})
        if model.opening_shape is not None:
            from ..local.compile import clear_shape_region

            opening["clear_outline_mm"] = _outline_ring(clear_shape_region(
                model.opening_shape, _floats(model.opening_center_mm) if model.opening_center_mm is not None else (0.0, 0.0),
                model.opening_radius_mm, model.opening_half_widths_mm))
        plane = OpticSurfaceV1(surface_id=APERTURE_STOP_SURFACE_ID, role=OpticSurfaceRoleV1.APERTURE_STOP, vertex_z_mm=float(model.plane_z_mm),
                               aperture_center_mm=None if model.opening_center_mm is None else _floats(model.opening_center_mm),
                               obstruction_radius_mm=None if model.obstruction_radius_mm is None else float(model.obstruction_radius_mm),
                               **opening)
        outline = ({"edge_radius_mm": float(model.edge_radius_mm)} if model.edge_radius_mm is not None
                   else {"edge_half_widths_mm": _floats(model.edge_half_widths_mm)})
        return OpticBodyGeometryV1(**common, **outline, surfaces=(plane,))
    if model.model_kind == "IDEAL_THIN_LENS":
        plane = OpticSurfaceV1(surface_id=THIN_LENS_SURFACE_ID, role=OpticSurfaceRoleV1.THIN_LENS, vertex_z_mm=float(model.plane_z_mm),
                               clear_radius_mm=float(model.clear_radius_mm), focal_length_mm=float(model.focal_length_mm))
        return OpticBodyGeometryV1(**common, edge_radius_mm=float(model.edge_radius_mm), surfaces=(plane,))
    if model.model_kind == "TABULATED_THIN_LENS":  # its focal length depends on the wavelength: the plane states none
        plane = OpticSurfaceV1(surface_id=THIN_LENS_SURFACE_ID, role=OpticSurfaceRoleV1.THIN_LENS, vertex_z_mm=float(model.plane_z_mm),
                               clear_radius_mm=float(model.clear_radius_mm))
        return OpticBodyGeometryV1(**common, edge_radius_mm=float(model.edge_radius_mm), surfaces=(plane,))
    if model.model_kind == "FARADAY_ROTATOR":
        plane = OpticSurfaceV1(surface_id=FARADAY_ROTATOR_SURFACE_ID, role=OpticSurfaceRoleV1.FARADAY_ROTATOR, vertex_z_mm=float(model.plane_z_mm),
                               clear_radius_mm=float(model.clear_radius_mm))
        return OpticBodyGeometryV1(**common, edge_radius_mm=float(model.edge_radius_mm), surfaces=(plane,))
    if model.model_kind == "FACET_SOLID":
        return OpticBodyGeometryV1(**{**common, "fidelity": ModelFidelity.SURFACE_PRESCRIPTION}, facets=_facets(model, coatings))
    if model.model_kind == "DIFFRACTION_GRATING":
        opening = ({"clear_radius_mm": float(model.clear_radius_mm)} if model.clear_radius_mm is not None
                   else {"clear_half_widths_mm": _floats(model.clear_half_widths_mm)})
        outline = ({"edge_radius_mm": float(model.edge_radius_mm)} if model.edge_radius_mm is not None
                   else {"edge_half_widths_mm": _floats(model.edge_half_widths_mm)})
        plane = OpticSurfaceV1(surface_id=GRATING_SURFACE_ID, role=OpticSurfaceRoleV1.GRATING, vertex_z_mm=float(model.plane_z_mm), **opening)
        vector = element.surfaces[0].interaction.grating_vector_local
        return OpticBodyGeometryV1(**common, **outline, surfaces=(plane,), grating=OpticGratingV1(
            mode=model.mode, period_nm=float(model.period), grating_vector_local=(vector[0], vector[1])))
    return _body(common, model, coating_kind)


#: The view role of each facet group of a ``FACET_SOLID`` (``FacetGroupV1.traced``).
FACET_GROUP_ROLES = {"TRANSMITTED": "REFRACT", "REFLECTED": "REFLECT", "ABSORBED": "ABSORB"}


def _facets(model: Any, coatings: dict[str, Any]) -> OpticFacetsV1:
    """The facets of a declared ``FACET_SOLID`` as the view draws them (the run traces the same facets)."""

    def kind(coating_id: str | None):
        coating = coatings.get(coating_id) if coating_id is not None else None
        return None if coating is None else coating.kind

    return OpticFacetsV1(
        vertices_mm=model.vertex_floats(), triangles=tuple(tuple(t) for t in model.triangles), approximation=model.approximation,
        max_chord_deviation_mm=None if model.max_chord_deviation_mm is None else float(model.max_chord_deviation_mm), medium=model.medium,
        coating_kind=kind(model.coating_id),
        groups=tuple(OpticFacetGroupV1(triangles=tuple(group.triangles), role=FACET_GROUP_ROLES[group.traced], coating_kind=kind(group.coating_id))
                     for group in model.facet_groups or ()))


def _attenuator_body(common: dict[str, Any], model: Any) -> OpticBodyGeometryV1:
    """A ``NEUTRAL_DENSITY`` attenuator: its ``attenuator`` plane (clear aperture) inside its outline."""

    opening = ({"clear_radius_mm": float(model.clear_radius_mm)} if model.clear_radius_mm is not None
               else {"clear_half_widths_mm": _floats(model.clear_half_widths_mm)})
    plane = OpticSurfaceV1(surface_id=ATTENUATOR_SURFACE_ID, role=OpticSurfaceRoleV1.ATTENUATOR, vertex_z_mm=float(model.plane_z_mm), **opening)
    outline = ({"edge_radius_mm": float(model.edge_radius_mm)} if model.edge_radius_mm is not None
               else {"edge_half_widths_mm": _floats(model.edge_half_widths_mm)})
    return OpticBodyGeometryV1(**common, **outline, surfaces=(plane,))


def _body(common: dict[str, Any], model: PublicModel, coating_kind: Callable[[Any], Any]) -> OpticBodyGeometryV1:

    if isinstance(model, SurfaceSequenceModelV1):
        surfaces = [OpticSurfaceV1(surface_id=s.surface_id, role=SURFACE_ROLE_BY_TRACED[s.traced], vertex_z_mm=float(s.vertex_z_mm),
                                   coating_kind=coating_kind(s), **_profile_fields(s), **_edge_fields(s)) for s in model.surfaces]
        outline = ({"edge_radius_mm": float(model.edge_radius_mm)} if model.edge_radius_mm is not None
                   else {"edge_half_widths_mm": _floats(model.edge_half_widths_mm)})  # type: ignore[arg-type]
        return OpticBodyGeometryV1(**common, **outline, surfaces=tuple(surfaces), media=model.media)
    if isinstance(model, SurfaceSetModelV1):
        surfaces = [OpticSurfaceV1(surface_id=s.surface_id, role=SURFACE_ROLE_BY_TRACED[s.traced], origin_mm=_floats(s.origin_mm),
                                   normal=_floats(s.normal), u_axis=_floats(s.u_axis), coating_kind=coating_kind(s), **_profile_fields(s))
                    for s in model.surfaces]
        solid = None
        if model.body is not None:
            solid = SolidBodyV1(medium=model.body.medium,
                                faces=tuple(SolidFaceV1(point_mm=_floats(f.point_mm), outward_normal=_floats(f.outward_normal)) for f in model.body.faces),
                                side_radius_mm=None if model.body.side_radius_mm is None else float(model.body.side_radius_mm))
        return OpticBodyGeometryV1(**common, surfaces=tuple(surfaces), solid=solid)
    if model.model_kind == "NEUTRAL_DENSITY":
        return _attenuator_body(common, model)
    if model.model_kind != "POLARIZATION_ELEMENT":
        raise KeyError(f"no OPTIC_BODY drawing of a {model.model_kind} model")
    plane = OpticSurfaceV1(surface_id=POLARIZATION_ELEMENT_SURFACE_ID, role=OpticSurfaceRoleV1.POLARIZATION_ELEMENT,
                           vertex_z_mm=float(model.plane_z_mm), clear_radius_mm=float(model.clear_radius_mm))
    return OpticBodyGeometryV1(**common, edge_radius_mm=float(model.edge_radius_mm), surfaces=(plane,),
                               polarization=PolarizationAxisV1(element=model.element))


def _source_radius(engine: EngineeringComponentSnapshot) -> float:
    if engine.emitter is not None:  # (the circle enclosing the emitting area)
        size = engine.emitter
        return size.size_x_mm / 2.0 if size.shape == "DISC" else math.hypot(size.size_x_mm, size.size_y_mm) / 2.0
    if engine.source is None:
        return 1.0
    return max(engine.source.waist_radius_x_mm, engine.source.waist_radius_y_mm)


def _marker_radius(engine: EngineeringComponentSnapshot) -> float:
    radii = [s.aperture.radius_mm for s in engine.surfaces if getattr(s, "aperture", None) is not None and s.aperture.shape == "circle"]
    radii += [s.aperture_radius_mm for s in engine.surfaces if s.geometry == "sphere"]
    return max(radii) if radii else DEFAULT_MARKER_RADIUS_MM


def unit_geometry(element: Any, project: ProjectDocumentV2, d: int) -> OpticalUnitGeometryV1:
    """The ``OPTICAL_UNIT`` of a compiled ``COMPOUND`` unit: its envelope and ports, and each member's declared body at its
    pose in the unit frame."""

    model = element.model
    members = tuple(UnitMemberV1(member_id=m.member_id, optical_model_id=m.optical_model_id, label=m.optical_model_id,
                                 pose=PoseV1(position_mm=_rv(m.member_pose.position_mm, d), quaternion_wxyz=m.member_pose.quaternion_wxyz),
                                 body=declared_body(m, project)) for m in element.members)
    ports = None if model.ports is None else tuple(
        UnitPortV1(port_id=port.port_id, role=port.role, point_mm=_floats(port.point_mm), direction=_floats(port.direction))
        for port in model.ports)
    return OpticalUnitGeometryV1(**_shape_fields(model.envelope), ports=ports, members=members)


def _elements(project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2 | None, models: dict[str, tuple[PublicOpticalProductV2, PublicModel]],
              request: ValidationRequest, d: int, compiled: dict[str, Any] | None = None) -> tuple[list[SceneElementV1], bool]:
    engine_components = {c.instance_id: c for c in request.project.components}
    elements: list[SceneElementV1] = []
    used_catalog = False
    for component in project.components:
        pose = PoseV1(position_mm=_rv(component.nominal_pose.position_mm, d), quaternion_wxyz=component.nominal_pose.quaternion_wxyz)
        engine = engine_components[component.instance_id]
        sku = model_id = None
        try:
            if component.product is not None:
                product, model = models[component.instance_id]
                geometry = optic_body(product, model, catalog)  # type: ignore[arg-type]
                sku, label, used_catalog = product.optical_sku, product.display_name, True
            elif component.optical_model_id is not None:
                element = (compiled or {}).get(component.instance_id) or compile_declared_element(
                    component.instance_id, project, request_design_point(request))
                geometry = unit_geometry(element, project, d) if element.is_compound else declared_body(element, project)
                model_id = label = component.optical_model_id
            elif component.optical_role is OpticalRoleV1.SOURCE:
                emitting = None if engine.emitter is None else EmittingAreaV1(shape=engine.emitter.shape, size_x_mm=engine.emitter.size_x_mm,
                                                                               size_y_mm=engine.emitter.size_y_mm)
                geometry = SourceEmitterGeometryV1(emitter_radius_mm=_round(_source_radius(engine), d), marker_length_mm=SOURCE_MARKER_LENGTH_MM,
                                                   emitting_area=emitting)
                label = "Source"
            else:
                geometry = ElementMarkerGeometryV1(radius_mm=_round(_marker_radius(engine), d), thickness_mm=MARKER_THICKNESS_MM)
                label = component.optical_role.value.replace("_", " ").capitalize()
            elements.append(SceneElementV1(element_id=component.instance_id, optical_role=component.optical_role, optical_sku=sku,
                                           label=label, pose=pose, geometry=geometry, optical_model_id=model_id))
        except ValidationError as exc:
            what = (f" ({component.product.optical_sku})" if component.product is not None
                    else f" (model {component.optical_model_id})" if component.optical_model_id is not None else "")
            raise OpticalViewBindingError(f"component {component.instance_id!r}{what} cannot be drawn: {exc.errors(include_url=False)[0]['msg']}") from None
    return elements, used_catalog


def _observation_pose(project: ProjectDocumentV2, observation) -> tuple[g.Vec3, g.Quat]:
    position, q = _vec(observation.pose.position_mm), _quat(observation.pose)
    if observation.device_instance_id is None:
        return position, q
    (parent,) = [c for c in project.components if c.instance_id == observation.device_instance_id]
    return g.compose_pose(_vec(parent.nominal_pose.position_mm), _quat(parent.nominal_pose), position, q)


def _absorbing_extent(observation, engine_plane) -> ObservationAbsorptionV1:
    """The plane's absorbing semantics: the Project's ``absorbing`` flag on its active area (``aperture``) or, without
    one, on the whole plane. When the Engine reports the resolved plane it must say the same."""

    if not observation.absorbing:
        extent = ObservationAbsorptionV1.NONE
    else:
        extent = ObservationAbsorptionV1.ACTIVE_AREA if observation.aperture is not None else ObservationAbsorptionV1.UNBOUNDED_PLANE
    if engine_plane is not None and engine_plane.absorbing_extent != extent.value:
        raise OpticalViewBindingError(
            f"observation {observation.observation_id!r}: the Engine plane absorbs on {engine_plane.absorbing_extent}, the Project on {extent.value}"
        )
    return extent


def _planes(project: ProjectDocumentV2, result: ValidationResult | None, d: int, sig_digits: int, abs_floor: float) -> list[ObservationPlaneV1]:
    """The observation planes of the Project, with the footprints of ``result``'s arrivals (none without a result: a layout)."""

    engine_observations = {o.observation_id: o for o in result.observations} if result is not None else {}
    planes = []
    for observation in project.observations:
        position, q = _observation_pose(project, observation)
        target_ids = tuple(t.target_id for t in project.targets if t.observation_id == observation.observation_id)
        engine_observation = engine_observations.get(observation.observation_id)
        footprints = []
        for arrival in engine_observation.arrivals if engine_observation is not None else ():
            gauss = arrival.beam.gaussian
            valid = gauss is not None and gauss.valid
            footprints.append(BeamFootprintV1(
                path_id=arrival.beam.path_id, crossing_index=arrival.crossing_index,
                chief_uv_mm=(_round(arrival.uv_mm[0], d), _round(arrival.uv_mm[1], d)),
                beam_radius_u_mm=_round(gauss.beam_radius_x_mm, d) if valid and gauss.beam_radius_x_mm else None,
                beam_radius_v_mm=_round(gauss.beam_radius_y_mm, d) if valid and gauss.beam_radius_y_mm else None,
                pupil_rms_radius_mm=None if arrival.pupil_rms_radius_mm is None else _round(arrival.pupil_rms_radius_mm, d),
                # (not known: a grating order without a stated response, or a power out of model scope -- the chief ray's, not the light's)
                power_W=None if arrival.beam.power_W is None or arrival.power_out_of_scope_at
                else quantize_float(float(arrival.beam.power_W), sig_digits, abs_floor),
            ))
        aperture = observation.aperture
        if isinstance(aperture, CircularApertureV1):
            extent = aperture.radius_mm
        elif isinstance(aperture, RectangularApertureV1):
            extent = max(aperture.half_width_mm, aperture.half_height_mm)
        else:
            extent = UNBOUNDED_PLANE_HALF_EXTENT_MM
        planes.append(ObservationPlaneV1(
            observation_id=observation.observation_id,
            kind=ObservationPlaneKindV1.TARGET if target_ids else ObservationPlaneKindV1.OBSERVATION,
            pose=PoseV1(position_mm=_rv(position, d), quaternion_wxyz=q), aperture=aperture, display_half_extent_mm=extent,
            absorbing_extent=_absorbing_extent(observation, None if engine_observation is None else engine_observation.plane),
            target_ids=target_ids, footprints=tuple(sorted(footprints, key=lambda f: (f.path_id, f.crossing_index))),
        ))
    return planes


def _extension(points, direction, length: float, d: int):
    if direction is None:
        return None
    return _rv(g.add(_vec(points[-1]), g.scale(g.unit(_vec(direction)), length)), d)


# Two points of one Engine path are the same split point when they agree to this distance (mm).
SPLIT_POINT_TOLERANCE_MM = 1e-6
PathRoles = dict[str, tuple[float | None, tuple[BeamDisplayRoleV1, ...], BeamDisplayRoleV1]]


def _arriving_power(segment, sig_digits: int, abs_floor: float) -> float:
    power = segment.power_end_W if segment.power_end_W is not None else segment.power_start_W
    return 0.0 if power is None else quantize_float(float(power), sig_digits, abs_floor)  # (a stretch of unknown power: no share)


#: Internal display states of a path's segments: a main beam, a weak branch, light its element rejected.
_MAIN, _WEAK, _BLOCKED = 2, 1, 0
#: ``(surface "<instance>/<surface>", reflected) -> the best share that surface's published coating passes`` (None: none).
SurfacePeak = Callable[[str, bool, float | None, float | None], "float | None"]


def surface_peaks(models: dict[str, tuple[PublicOpticalProductV2, PublicModel]], catalog: PublicOpticalCatalogV2 | None,
                  compiled: dict[str, Any] | None = None) -> SurfacePeak:
    """The best reflectance / transmittance (``coating_peak``) of each coated surface of the run's catalog elements, by the
    Engine's surface id (``<instance id>/<surface id>``), and of its declared elements (``compiled``: their compiled
    elements, whose surface envelopes carry their coatings; a compound unit's ``<member>-<surface>`` ids)."""

    coatings: dict[str, Any] = {}
    for instance_id, (_, model) in models.items():
        for surface in getattr(model, "surfaces", ()):
            coating = catalog.coating(surface.coating_id) if getattr(surface, "coating_id", None) and catalog is not None else None
            if coating is not None:
                coatings[f"{instance_id}/{surface.surface_id}"] = coating
    for instance_id, element in (compiled or {}).items():
        if instance_id in models:
            continue
        for engine_id, envelope in element.envelope_by_engine_id().items():
            if envelope.coating is not None:
                coatings[engine_id] = envelope.coating

    def peak(surface_id: str, reflected: bool, wavelength_nm: float | None = None,
             aoi_deg: float | None = None) -> float | None:
        coating = coatings.get(surface_id)
        if coating is None:
            return None
        if wavelength_nm is None:
            return coating_peak(coating, reflected=reflected)
        # When the incident bulk loss is unknown, an Engine segment has no numeric power. A published coating can
        # still guarantee that *its own* outgoing fraction is tiny at this wavelength, independently of that loss.
        band_at = getattr(coating, "band_at", None)
        band = (band_at(Decimal(repr(wavelength_nm)), None if aoi_deg is None else Decimal(repr(aoi_deg)))
                if callable(band_at) else None)
        if band is None:
            return None
        field = "reflectance" if reflected else "transmittance"
        values = [float(getattr(band, field))]
        for pol in ("s", "p"):
            if (value := getattr(band, f"{field}_{pol}", None)) is not None:
                values.append(float(value))
        return max(values)  # the largest polarization share: a conservative guarantee of coating rejection

    return peak


def _path_roles(project: ProjectDocumentV2, trace: RayPathTrace, result: ValidationResult, sig_digits: int, abs_floor: float,
                tolerance_nm: float, peak: SurfacePeak | None = None) -> PathRoles:
    """Each traced path's end power (quantized like the result hash) and the display role of each of its Engine segments.

    A stretch whose power is not known keeps the state its light arrives in unless a published coating *alone* guarantees
    a blocked share at that line; absent bulk-loss data never becomes a measured zero. A branch that receives less than
    ``SECONDARY_BRANCH_POWER_SHARE`` of the power arriving at a split is weak from there
    on (a leak, a ghost reflection), and a line that a surface passes at less than ``BLOCKED_POWER_SHARE`` of the best
    share its published coating passes is blocked there (a filter's stop band; a neutral density filter passes every
    line at its best share and never blocks; a surface without a published coating is judged by its absolute share).
    Weak and blocked stretches are SECONDARY, and so is everything downstream of them. A path that reaches an observation
    plane whose Project target asks for it (``designed_arrivals``) is the designed signal: its weak stretches are PRIMARY
    (a weak pick-off), never a blocked one. Powers are quantized before comparing, so both runtimes
    decide identically.
    """

    engine = {p.path_id: p for p in result.paths}
    missing = sorted(p.path_id for p in trace.paths if p.path_id not in engine)
    if missing:
        raise OpticalViewBindingError(f"the Engine result has no path record for traced path(s) {missing}")
    q = lambda value: quantize_float(float(value), sig_digits, abs_floor)  # noqa: E731
    children: dict[str, list[Any]] = {}
    for path in result.paths:
        if path.parent_path_id is not None:
            children.setdefault(path.parent_path_id, []).append(path)

    def splits_at(path, k: int) -> bool:
        start = path.segments[k].start_mm
        return any(c.segments and c.branch_surface_id == path.segments[k - 1].end_surface_id  # a branch below the power floor has none
                   and g.norm(g.sub(_vec(c.segments[0].start_mm), _vec(start))) <= SPLIT_POINT_TOLERANCE_MM
                   for c in children.get(path.path_id, ()))

    def share(power_after: float, arriving: float) -> float:
        return 1.0 if arriving <= 0.0 else power_after / arriving

    def rejected(kept: float | None, event: str | None, wavelength_nm: float, aoi_deg: float | None) -> bool:
        """A known retained share, or a guaranteed small coating share when upstream bulk power is unknown.

        No unknown beam power is substituted with zero: only the coating's published *fraction* can mark a line blocked.
        An angular/polarization coating uses its largest polarized share so rejection is guaranteed for any input field.
        """

        best = None
        if event is not None and peak is not None:
            surface, _, interaction = event.rpartition(":")
            reflected = "reflect" in interaction
            best = peak(surface, reflected, None, None)
            if kept is None:
                kept = peak(surface, reflected, wavelength_nm, aoi_deg)
        return kept is not None and kept < BLOCKED_POWER_SHARE * (best if best is not None and best > 0.0 else 1.0)

    def states(path, start: int | None, lift_to: int) -> list[int]:
        """The path's segment states, from the state its parent's light arrives at its split in (None: a root path);
        the weak segments before ``lift_to`` (a designed path: all of them) are lifted, never a blocked one."""

        own: list[int] = []
        if not path.segments:
            return own
        state = _MAIN if start is None else start
        offset = 1 if path.parent_path_id is not None else 0  # a branch's lineage starts with the split that made it
        lineage = list(path.lineage)
        for k in range(len(path.segments)):
            power = path.segments[k].power_start_W
            if k == 0 and path.parent_path_id is not None:
                state = branch_state(path, state)
            elif k > 0:
                kept = (share(q(power), _arriving_power(path.segments[k - 1], sig_digits, abs_floor))
                        if power is not None and (path.segments[k - 1].power_end_W is not None or
                                                  path.segments[k - 1].power_start_W is not None) else None)
                event = lineage[k - 1 + offset] if k - 1 + offset < len(lineage) else None
                if rejected(kept, event, path.wavelength_nm, path.segments[k - 1].angle_of_incidence_deg):
                    state = _BLOCKED
                elif kept is not None and kept < SECONDARY_BRANCH_POWER_SHARE and splits_at(path, k):
                    state = min(state, _WEAK)
            own.append(state)
        for k in range(min(lift_to, len(own))):
            if own[k] == _WEAK:
                own[k] = _MAIN
        return own

    def branch_state(path, arriving: int) -> int:
        parent = engine.get(path.parent_path_id)
        if parent is None:
            return arriving
        start = _vec(path.segments[0].start_mm)
        at = [k for k in range(1, len(parent.segments)) if parent.segments[k - 1].end_surface_id == path.branch_surface_id
              and g.norm(g.sub(_vec(parent.segments[k].start_mm), start)) <= SPLIT_POINT_TOLERANCE_MM]
        if not at:
            return arriving
        before = parent.segments[at[0] - 1]
        power = path.segments[0].power_start_W
        kept = (share(q(power), _arriving_power(before, sig_digits, abs_floor))
                if power is not None and (before.power_end_W is not None or before.power_start_W is not None) else None)
        if rejected(kept, path.lineage[0] if path.lineage else None, path.wavelength_nm, before.angle_of_incidence_deg):
            return _BLOCKED
        return min(arriving, _WEAK) if kept is not None and kept < SECONDARY_BRANCH_POWER_SHARE else arriving

    def parent_state(path, roles: dict[str, list[int]]) -> int | None:
        """The state the parent's light arrives in at the split that made ``path``."""

        parent = engine.get(path.parent_path_id) if path.parent_path_id is not None else None
        if parent is None:
            return None
        start = _vec(path.segments[0].start_mm) if path.segments else None
        at = [k for k in range(1, len(parent.segments)) if start is not None and parent.segments[k - 1].end_surface_id == path.branch_surface_id
              and g.norm(g.sub(_vec(parent.segments[k].start_mm), start)) <= SPLIT_POINT_TOLERANCE_MM]
        before = roles.get(parent.path_id, [])
        return before[at[0] - 1] if at and before else None

    def crossing(path, observation_id: str) -> int | None:
        """The index of the segment that arrives at an observation plane (None: not found)."""

        end = f"observation/{observation_id}"
        return next((k for k, s in enumerate(path.segments) if s.end_surface_id == end), None)

    depth: dict[str, int] = {}

    def branch_depth(path) -> int:
        """Splits between the path and its source line (parents first; a deep branch's id is compacted, so its dots do not
        count its depth: ``BRANCH_SUFFIX_PATTERN``)."""

        chain = []
        while path is not None and path.path_id not in depth:
            chain.append(path)
            path = engine.get(path.parent_path_id) if path.parent_path_id is not None else None
        level = -1 if path is None else depth[path.path_id]
        for item in reversed(chain):
            level += 1
            depth[item.path_id] = level
        return depth[chain[0].path_id] if chain else level

    order = sorted(result.paths, key=lambda p: (branch_depth(p), p.path_id))

    def run(lift: dict[str, int]) -> dict[str, list[int]]:
        roles: dict[str, list[int]] = {}
        for path in order:
            roles[path.path_id] = states(path, parent_state(path, roles), lift.get(path.path_id, 0))
        return roles

    first = run({})
    arrivals: dict[str, list[PlaneArrivalV1]] = {}
    for observation in result.observations:
        for arrival in observation.arrivals:
            path = engine.get(arrival.beam.path_id)
            if path is None:
                continue
            k = crossing(path, observation.observation_id)
            own = first[path.path_id]
            state = own[k] if k is not None and k < len(own) else (own[-1] if own else _MAIN)
            arrivals.setdefault(observation.observation_id, []).append(PlaneArrivalV1(
                key=path.path_id, source_id=path.source_id, wavelength_nm=path.wavelength_nm, power=0.0 if arrival.beam.power_W is None else q(arrival.beam.power_W),
                blocked=state == _BLOCKED, path_id=path.path_id))
    # a designed path: every weak stretch of it is the designed signal (its footprints and display extension included)
    lift = {path_id: len(engine[path_id].segments) for _, path_id in designed_arrivals(project.targets, arrivals, tolerance_nm=tolerance_nm)}
    states_by_path = run(lift) if lift else first
    roles = {pid: [BeamDisplayRoleV1.PRIMARY if s == _MAIN else BeamDisplayRoleV1.SECONDARY for s in own]
             for pid, own in states_by_path.items()}

    out: PathRoles = {}
    for path in trace.paths:
        own = roles[path.path_id] or [BeamDisplayRoleV1.PRIMARY]
        count = max(len(path.chief_points_mm) - 1, 0)
        segment_roles = tuple(own[min(k, len(own) - 1)] for k in range(count))
        final = engine[path.path_id].final_power_W
        if engine[path.path_id].power_out_of_scope_at:  # (the chief ray's power, not the light's: not known)
            final = None
        out[path.path_id] = (None if final is None else q(final), segment_roles, own[-1])
    return out


#: The Engine issue that marks a beam whose chief ray fell on a central obstruction (its light is the annulus around it).
CENTRAL_OBSTRUCTION_CODE = "GAUSSIAN_CENTRAL_OBSTRUCTION"


def chief_reference_segments(trace: RayPathTrace, result: ValidationResult) -> dict[str, int]:
    """Per path, the first chief-ray segment from which its chief ray is only the beam's reference axis: from the surface
    where it fell on a central obstruction (the Engine's ``GAUSSIAN_CENTRAL_OBSTRUCTION`` issue of that path), and from the
    start of every branch spawned behind it."""

    surfaces = {}
    for issue in result.issues:
        path_id = issue.numerical_context.get("path_id") if issue.numerical_context else None
        if issue.code.value == CENTRAL_OBSTRUCTION_CODE and path_id is not None and (issue.location or "").startswith("surface:"):
            surfaces.setdefault(str(path_id), issue.location.split(":", 1)[1])  # type: ignore[union-attr]
    by_id = {p.path_id: p for p in trace.paths}
    out: dict[str, int] = {}

    def resolve(path) -> int | None:
        if path.path_id in out:
            return out[path.path_id]
        found = None
        surface = surfaces.get(path.path_id)
        if surface is not None and surface in path.chief_surface_ids:
            found = path.chief_surface_ids.index(surface)
        elif path.parent_path_id in by_id:
            parent = by_id[path.parent_path_id]
            start = resolve(parent)
            if start is not None and path.chief_points_mm:
                spawn = [k for k, q in enumerate(parent.chief_points_mm) if q == path.chief_points_mm[0]]
                if spawn and spawn[0] >= start:
                    found = 0
        if found is not None and found < len(path.chief_points_mm) - 1:
            out[path.path_id] = found
        return out.get(path.path_id)

    for path in trace.paths:
        resolve(path)
    return out


def _beam_paths(trace: RayPathTrace, roles: PathRoles, settings: OpticalViewSettingsV1, progress: Progress | None,
                references: dict[str, int] | None = None) -> list[BeamPathV1]:
    d = settings.coordinate_decimals
    paths = []
    for k, path in enumerate(trace.paths, start=1):
        checkpoint()  # (one more beam path drawn: the run's monitor may stop the run here, os_optics/run_control.py)
        if progress is not None:
            progress(f"phase 3/4 beam paths {k} / {len(trace.paths)}")
        # a path that ends at a surface (clipped, absorbed, detected, out of model range, ...) stops there
        stopped = path.termination != "ESCAPED" and path.termination_surface_id is not None
        chief_end = RayEndV1.STOPPED_AT_SURFACE if stopped else RayEndV1.PATH_END
        chief = RayPolylineV1(
            ray_index=0, points_mm=tuple(_rv(p, d) for p in path.chief_points_mm), end=chief_end,
            display_extension_to_mm=_extension(path.chief_points_mm, path.chief_exit_direction, settings.free_ray_display_length_mm, d),
        )
        outer = max((r.ring_index for r in path.pupil_rays), default=-1)
        if settings.pupil_ray_selection is PupilRaySelectionV1.ALL:
            chosen = list(path.pupil_rays)
        elif settings.pupil_ray_selection is PupilRaySelectionV1.OUTER_RING:
            chosen = [r for r in path.pupil_rays if r.ring_index == outer]
        else:
            chosen = []
        pupil = tuple(
            RayPolylineV1(
                ray_index=r.ray_index, ring_index=r.ring_index, points_mm=tuple(_rv(p, d) for p in r.points_mm), end=RayEndV1(r.end),
                # a free ray gets a display stub; a NOT_FOLLOWED ray ends where the Engine stopped following it
                display_extension_to_mm=_extension(r.points_mm, r.exit_direction, settings.free_ray_display_length_mm, d)
                if r.end == "PATH_END" else None,
            )
            for r in chosen
        )
        paths.append(BeamPathV1(
            path_id=path.path_id, source_element_id=path.source_component_id, wavelength_nm=path.wavelength_nm,
            spectral_index=path.spectral_index, parent_path_id=path.parent_path_id, termination=path.termination,
            display_rgb=wavelength_display_rgb(path.wavelength_nm), chief_ray=chief, pupil_rays=pupil,
            pupil_ray_count_traced=path.pupil_ray_count, final_power_W=roles[path.path_id][0], segment_roles=roles[path.path_id][1],
            display_role=roles[path.path_id][2], chief_reference_from=(references or {}).get(path.path_id),
        ))
    return paths


def _unit_extent_points(geom: OpticalUnitGeometryV1) -> list[g.Vec3]:
    """Unit-frame points bounding a unit: its envelope's corners and every member body's points (member pose applied)."""

    if geom.shape == "CYLINDER":
        r = geom.diameter_mm / 2  # type: ignore[operator]
        points = [(sx * r, sy * r, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (geom.z_min_mm, geom.z_max_mm)]
    elif geom.shape == "BOX":
        c, h = geom.center_mm, geom.half_extents_mm
        points = [(c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]  # type: ignore[index]
    else:
        points = [tuple(v) for v in geom.vertices_mm or ()]
    for member in geom.members:
        p, q = _vec(member.pose.position_mm), _quat(member.pose)
        points += [g.apply_pose(p, q, point) for point in _body_extent_points(member.body)]
    return points


def _body_extent_points(geom: OpticBodyGeometryV1) -> list[g.Vec3]:
    """Element-frame points bounding an ``OPTIC_BODY``."""

    if geom.model_kind == "APERTURE_STOP":  # a plate in one plane: its outline there
        z = geom.surfaces[0].vertex_z_mm or 0.0
        hx, hy = (geom.edge_radius_mm, geom.edge_radius_mm) if geom.edge_radius_mm is not None else geom.edge_half_widths_mm  # type: ignore[misc]
        return [(sx * hx, sy * hy, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    if not optic_mesh.is_plain(geom):
        return optic_mesh.extent_points(geom)
    r, z0, z1 = geom.diameter_mm / 2, geom.z_min_mm, geom.z_max_mm  # type: ignore[operator]
    rims = [g.surface_rim(s.vertex_z_mm, s.radius_mm, s.clear_radius_mm, geom.edge_radius_mm)[1] for s in geom.surfaces]
    z0, z1 = min(z0, *rims), max(z1, *rims)  # type: ignore[type-var]
    return [(sx * r, sy * r, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (z0, z1)]  # type: ignore[misc]


def _element_extent_points(element: SceneElementV1) -> list[g.Vec3]:
    geom = element.geometry
    local: list[g.Vec3]
    if isinstance(geom, OpticalUnitGeometryV1):
        p, q = _vec(element.pose.position_mm), _quat(element.pose)
        return [g.apply_pose(p, q, point) for point in _unit_extent_points(geom)]
    if isinstance(geom, OpticBodyGeometryV1) and (geom.is_declared or not optic_mesh.is_plain(geom)):
        p, q = _vec(element.pose.position_mm), _quat(element.pose)
        return [g.apply_pose(p, q, point) for point in _body_extent_points(geom)]
    if isinstance(geom, OpticBodyGeometryV1):
        r, z0, z1 = geom.diameter_mm / 2, geom.z_min_mm, geom.z_max_mm
        # a curved surface's rim may stand proud of the bounding body (e.g. a concave mirror face)
        rims = [g.surface_rim(s.vertex_z_mm, s.radius_mm, s.clear_radius_mm, geom.edge_radius_mm)[1] for s in geom.surfaces]
        z0, z1 = min(z0, *rims), max(z1, *rims)
    elif isinstance(geom, SourceEmitterGeometryV1):
        r, z0, z1 = geom.emitter_radius_mm, -geom.marker_length_mm, 0.0
    else:
        r, z0, z1 = geom.radius_mm, -geom.thickness_mm / 2, geom.thickness_mm / 2
    local = [(sx * r, sy * r, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in (z0, z1)]
    p, q = _vec(element.pose.position_mm), _quat(element.pose)
    return [g.apply_pose(p, q, point) for point in local]


def _plane_extent_points(plane: ObservationPlaneV1) -> list[g.Vec3]:
    h = plane.display_half_extent_mm
    p, q = _vec(plane.pose.position_mm), _quat(plane.pose)
    return [g.apply_pose(p, q, (sx * h, sy * h, 0.0)) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def _display_media(request: ValidationRequest) -> tuple[DisplayMediumV1, ...]:
    """The Engine materials of the run with their index at the d line, or at the middle of a band that does not reach it."""

    out = []
    for material in sorted(request.project.materials, key=lambda m: m.material_id):
        if not material.material_id.startswith(DISPLAY_MEDIUM_PREFIXES):
            continue
        low, high = material.valid_wavelength_nm
        wavelength = D_LINE_NM if low <= D_LINE_NM <= high else (low + high) / 2.0
        index, _ = build_evaluator(material, request.configuration.numerical).index(wavelength)
        if index is not None:
            out.append(DisplayMediumV1(medium_id=material.material_id, refractive_index=round(index, DISPLAY_INDEX_DECIMALS), at_wavelength_nm=wavelength))
    return tuple(out)


def axis_markers(elements: list[SceneElementV1], settings: OpticalViewSettingsV1, d: int) -> list[AxisMarkerV1]:
    """The optical-axis marker of every element but the sources: its local Z axis through its origin, both ways."""

    markers = []
    for element in elements:
        if element.optical_role is OpticalRoleV1.SOURCE:
            continue
        p, q = _vec(element.pose.position_mm), _quat(element.pose)
        h = settings.axis_marker_half_length_mm
        markers.append(AxisMarkerV1(element_id=element.element_id, start_mm=_rv(g.apply_pose(p, q, (0.0, 0.0, -h)), d),
                                    end_mm=_rv(g.apply_pose(p, q, (0.0, 0.0, h)), d)))
    return markers


def _bounds_and_camera(points: list[g.Vec3], d: int) -> tuple[SceneBoundsV1, CameraHintV1]:
    lo = tuple(min(p[i] for p in points) for i in range(3))
    hi = tuple(max(p[i] for p in points) for i in range(3))
    center = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
    diagonal = max(g.norm(g.sub(hi, lo)), 1.0)
    eye = g.add(center, g.scale(g.unit(CAMERA_VIEW_DIRECTION), CAMERA_DISTANCE_FACTOR * diagonal))
    return SceneBoundsV1(min_mm=_rv(lo, d), max_mm=_rv(hi, d)), CameraHintV1(target_mm=_rv(center, d), eye_mm=_rv(eye, d))


def build_optical_simulation_scene(
    *,
    project: ProjectDocumentV2,
    catalog: PublicOpticalCatalogV2 | None,
    engine_request: ValidationRequest,
    engine_result: ValidationResult | None = None,
    ray_paths: RayPathTrace | None = None,
    settings: OpticalViewSettingsV1 | None = None,
    progress: Progress | None = None,
    model_kind: str | None = None,
    same_evaluation: bool = False,
) -> OpticalSimulationSceneV1:
    """Scene of one local Engine run of ``project``.

    Normal use needs none of this: ``os_optics.local.run_optical_project`` builds the view of its own run
    (``run.optical_view``). Here, advanced callers pass the pieces of one run:

    * ``engine_result`` and ``ray_paths`` both given: the pair returned by ``validate_with_ray_paths(engine_request)``,
      one Engine execution;
    * only ``engine_result``: the ray paths are recorded by ``trace_ray_paths(engine_request)`` and must reproduce the
      result's input and trace hashes;
    * neither: the request is run here, once, with ``validate_with_ray_paths``.

    ``model_kind`` is the run's model selection (``run_optical_project(model_kind=...)``); by default every catalog
    product must publish exactly one simulation model. Every case is bound by hash (input, trace and quantized result
    hash), every catalog element by its compiled public model and every declared element by its compiled declared model;
    a mismatch raises ``OpticalViewBindingError`` and no scene is returned. ``catalog`` may be ``None`` for a Project
    without catalog products. ``same_evaluation`` (``run_optical_project`` passes true): ``engine_result`` and
    ``ray_paths`` are the objects one evaluation in this process just produced, so the result is not serialized and
    hashed again against its own manifest; the input and trace hashes and every element binding are still compared.
    """
    settings = settings or OpticalViewSettingsV1()
    d = settings.coordinate_decimals
    if progress is not None:
        progress("phase 1/4 engine trace")
    models = _lowered_models(project, catalog, model_kind)
    if engine_result is None and ray_paths is not None:
        raise OpticalViewBindingError("ray_paths were given without the Engine result of the same run")
    if engine_result is None:
        result, trace = validate_with_ray_paths(engine_request)
    else:
        result, trace = engine_result, ray_paths if ray_paths is not None else trace_ray_paths(engine_request)
    compiled = _bind(project, catalog, models, engine_request, result, trace, same_evaluation and engine_result is not None)

    if progress is not None:
        progress(f"phase 2/4 elements {len(project.components)} / {len(project.components)}")
    elements, used_catalog = _elements(project, catalog, models, engine_request, d, compiled)
    cfg = engine_request.configuration.numerical
    planes = _planes(project, result, d, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor)
    roles = _path_roles(project, trace, result, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor, cfg.interference_wavelength_tol_nm,
                        surface_peaks(models, catalog, compiled))
    beam_paths = _beam_paths(trace, roles, settings, progress, chief_reference_segments(trace, result))

    markers = axis_markers(elements, settings, d)

    points: list[g.Vec3] = []
    for element in elements:
        points.extend(_element_extent_points(element))
    for plane in planes:
        points.extend(_plane_extent_points(plane))
    for path in beam_paths:
        for ray in (path.chief_ray, *path.pupil_rays):
            points.extend(ray.points_mm)
            if ray.display_extension_to_mm is not None:
                points.append(ray.display_extension_to_mm)
    bounds, camera = _bounds_and_camera(points, d)

    manifest = result.manifest
    assert manifest is not None and manifest.trace_hash is not None
    engine_run = EngineRunRefV1(
        engine_package=BUILDER_PACKAGE,
        engine_version=manifest.engine_version,
        input_hash=HashRef(profile=HashProfile.ENGINE_EXACT_PY1, digest=manifest.input_hash, subject="engine_input"),
        result_hash=HashRef(profile=HashProfile.ENGINE_QUANTIZED_PY1, digest=manifest.result_hash, subject="engine_result",
                            params={"sig_digits": str(cfg.hash_float_sig_digits), "abs_floor": _decimal_string(cfg.hash_float_abs_floor)}),
        trace_hash=manifest.trace_hash,
        ray_trace_version=trace.ray_paths_version,
    )
    return OpticalSimulationSceneV1(
        project=validated_project_ref(project),
        catalog_version=catalog.catalog_version if used_catalog and catalog is not None else None,
        engine_run=engine_run,
        builder=ViewBuilderRefV1(builder_package=BUILDER_PACKAGE, builder_version=BUILDER_VERSION, settings=settings),
        elements=tuple(elements),
        observation_planes=tuple(planes),
        beam_paths=tuple(beam_paths),
        axis_markers=tuple(markers),
        display_media=_display_media(engine_request),
        bounds=bounds,
        camera_hint=camera,
    )
