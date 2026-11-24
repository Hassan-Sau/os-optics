"""Field-level headless edit operations on a Project, in Python.

``apply_project_ops(document, ops)`` applies a list of operations to a copy of a ``ProjectDocumentV2`` JSON document. An
operation only rearranges JSON fields (set a pose, merge source-model fields, replace the retained DOFs, add or remove
an item, ...); it computes no optics and decides no validity. ``update_project(document, ops)`` applies them, requires
the result to be a Project (the ``ProjectDocumentV2`` contract) and reports the check a run performs
(``json_api.check_project``: the lowering, the catalog binding, the optical solids) with the edited Project. The edit
and the check are two results: an edit that leaves a Project is accepted whatever the check says, so a design is built
and repaired one step at a time (a lens before its source, a position passed through on the way to another), and
``check.ok`` says whether a run would accept it. ``require_check=True`` makes the check a condition of the edit: all
operations and the check pass, or nothing changes.

Operations (``PROJECT_OPS``; each is ``{"op": NAME, ...}``):

* ``set_component_pose`` ``{instance_id, position_mm?, quaternion_wxyz?}`` (at least one), ``set_observation_pose``
  ``{observation_id, position_mm?, quaternion_wxyz?}``: absolute poses (an attached plane's pose is in its device's frame);
* ``set_source_model`` ``{instance_id, changes}``, ``set_observation`` ``{observation_id, changes}``, ``set_target``
  ``{target_id, changes}``, ``update_optical_model`` ``{model_id, changes}`` (never ``model_id``), ``update_model_material``
  ``{material_id, changes}`` (never ``material_id``): ``changes`` is merged into the object, a ``null`` value removes that
  field. A source declares one line (``wavelength_nm`` + ``power_w``) or several (``spectral_lines`` with ``wavelength_nm``
  and ``power_w`` set to ``null``), and at most one of ``linewidth_nm`` / ``coherence_length_mm``: the check refuses
  anything else;
* ``set_retained_dofs`` ``{instance_id, retained_dofs: [{axis, control}]}`` (sorted Tx..Rz; ``[]`` fixes every DOF),
  ``set_retained_dof`` ``{instance_id, axis, control: MANUAL | NUMERICAL | null}`` (``null`` releases the axis; an axis
  already retained keeps its stated limits);
* ``set_component_product`` ``{instance_id, optical_sku, catalog_version?}`` (``null`` removes the pin; a component using
  a declared model becomes a catalog component again: its ``custom_part`` and ``optical_model_id`` are removed, the model
  stays declared), ``set_component_model`` ``{instance_id, optical_model_id, custom_part_key?, description?}`` (the
  component becomes the customer's own optic: ``product`` removed, ``custom_part`` set, default key ``own-<instance_id>``);
* ``set_design_intent`` ``{design_intent: LAB_ASSEMBLY | OPTICAL_SYSTEM | null}``;
* ``set_compute_budget`` ``{compute_budget: {max_interactions_per_path?, max_paths?, max_trace_steps?, max_interference_pairs?, max_return_analysis_work?,
  pupil_ring_count?, pupil_points_per_ring?, max_scatter_origins?, max_mesh_patches?} | null}``: replace persisted
  resource settings (the budgets, the pupil sampling and the scatter-origin and mesh-patch limits); omitted fields use
  defaults, ``null`` resets the block;
* ``set_numerical_settings`` ``{numerical_settings: {power_epsilon_W?, negligible_power_fraction?, gaussian_*?, ..., rating_*?,
  design_point?: {...}} | null}``: replace the persisted validity limits, levels and match windows
  (``NumericalSettingsV1``); omitted fields use the Engine defaults, ``null`` (or an empty object) resets the block;
* ``add_component`` ``{component}``, ``remove_component`` ``{instance_id}``, ``add_observation`` ``{observation}``,
  ``remove_observation`` ``{observation_id}``, ``add_target`` ``{target}``, ``remove_target`` ``{target_id}``;
* ``add_optical_model`` ``{optical_model, model_materials?, model_coatings?}`` (a material / coating identical to one
  already declared is shared, a different one under the same id is refused), ``remove_optical_model`` ``{model_id}``
  (refused ``IN_USE`` while a component or a COMPOUND unit uses it; materials and coatings no remaining model uses go with
  it), ``add_model_material`` ``{material}``, ``remove_model_material`` ``{material_id}``, ``add_model_coating``
  ``{coating}``, ``remove_model_coating`` ``{coating_id}`` (refused ``IN_USE`` while a model uses them);
* ``transform`` ``{selector, translation_mm?, rotation?, frame?, pivot?}``: a rigid move of the selected components and
  observation planes, exactly as ``os_optics.composition.transform`` computes it on the bare Project (selector
  ``{ids, kinds}``; frame ``"design"`` or ``{component}``; pivot ``"frame_origin"``, ``"selection_centroid"``,
  ``{point_mm}`` or ``{component}``). Groups and instances belong to a workspace: use the workspace calls for them.

Refusals (``ProjectOpError``) carry the failing operation's index and one of ``UNKNOWN_OP``, ``NOT_FOUND``,
``DUPLICATE_ID``, ``BAD_ARGUMENT``, ``IN_USE`` and, for ``transform``, ``TRANSFORM_REFUSED`` (``problems``: the
composition codes such as ``SELECTION_INVALID``, ``ROTATION_INVALID``, ``FRAME_INVALID``). A malformed ``ops`` list has
index -1.

Retries: the set / merge / remove-by-value operations are absolute, so applying the same list again to its result
changes nothing (``changed`` false). An ``add_*`` of an item identical to the one already present under its id (compared
in the contract's normalized form) adds nothing and is listed in ``already_present``; different content under an
existing id is ``DUPLICATE_ID``. ``remove_*`` of an absent id is ``NOT_FOUND``, and ``transform`` is relative. A list
that adds an item and then changes or removes it (``[add_target X, set_target X ...]``) is not repeatable either: applied
again to its own result, the add meets the changed item (or the removed id is absent) and the list is refused
(``DUPLICATE_ID`` / ``NOT_FOUND``; nothing changes). So retry a list after a lost response with
``expected_project_hash`` (the hash the list was written against): a Project that has changed since is refused
``PROJECT_CHANGED`` with its current hash, which equals the ``project_hash`` the lost reply would have carried when the
first application took effect.

The operations are pure Python on JSON values, usable from CPython or a local Pyodide host.
A document that does not follow the Project contract where an operation reads it is refused (``BAD_ARGUMENT`` at that
operation, or ``PROJECT_REJECTED`` when the edited document is no Project), never with an exception.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

#: The operations and their fields (also advertised by ``capabilities().project_ops``).
PROJECT_OPS: dict[str, dict[str, Any]] = {
    "set_component_pose": {
        "summary": "Move / rotate a component (nominal pose in the design frame).",
        "fields": {"instance_id": "component id", "position_mm": "[x, y, z] mm (optional)",
                   "quaternion_wxyz": "[w, x, y, z] unit quaternion (optional)"},
    },
    "set_source_model": {
        "summary": "Change fields of a source's source_model. A GAUSSIAN_BEAM source: wavelength_nm, power_w, waist_radius_mm, waist_position_mm, m2, "
                   "polarization, waist_radius_y_mm, m2_y, and its coherence: linewidth_nm or coherence_length_mm, at most one -- set the "
                   "other to null. An EXTENDED_EMITTER: wavelength_nm, power_w, emitting_surface, angular_law, sampling. Several spectral "
                   "lines: spectral_lines [{wavelength_nm, power_w}, ...] (two or more, distinct "
                   "wavelengths) with wavelength_nm and power_w set to null; back to one line: wavelength_nm + power_w with "
                   "spectral_lines null. null removes an optional field.",
        "fields": {"instance_id": "source component id", "changes": "object of source_model fields"},
    },
    "set_retained_dofs": {
        "summary": "Replace a component's retained DOF intent. Axes Tx..Rz in the component's local frame; control MANUAL or "
                   "NUMERICAL. [] = every DOF fixed.",
        "fields": {"instance_id": "component id", "retained_dofs": "[{axis, control}]"},
    },
    "set_retained_dof": {
        "summary": "Retain one DOF axis with a control intent, or release it with control: null. An axis already retained "
                   "keeps its stated travel, hard_limits, repeatability, drift and unit; set_retained_dofs replaces them.",
        "fields": {"instance_id": "component id", "axis": "Tx|Ty|Tz|Rx|Ry|Rz", "control": "MANUAL|NUMERICAL|null"},
    },
    "set_component_product": {
        "summary": "Point a component at a public optical SKU (see catalog_products). A component using a declared optical model "
                   "becomes a catalog component again (its custom_part and optical_model_id are removed; the model stays in "
                   "optical_models).",
        "fields": {"instance_id": "component id", "optical_sku": "OS-OPT-...", "catalog_version": "catalog release (optional; null removes)"},
    },
    "set_component_model": {
        "summary": "Make a component the customer's own optic simulated with a declared optical model: product removed, custom_part "
                   "{custom_part_key, description} set (kept when present), optical_model_id set. Not for a source, detector or dump.",
        "fields": {"instance_id": "component id", "optical_model_id": "a model_id of optical_models",
                   "custom_part_key": "optional (default own-<instance_id>)", "description": "optional free text (default: the model id)"},
    },
    "set_design_intent": {
        "summary": "What the design is for: LAB_ASSEMBLY (optics on a breadboard with OpenSci mounts; the meaning when absent) or "
                   "OPTICAL_SYSTEM (an optical design simulated and inspected without lab mechanics; ordered as optics only or with "
                   "OpenSci mounts at one beam height, no platform). null removes the field (LAB_ASSEMBLY).",
        "fields": {"design_intent": "LAB_ASSEMBLY | OPTICAL_SYSTEM | null"},
    },
    "set_compute_budget": {
        "summary": "Replace the Project's computation budget: all five budgets, the pupil sampling (rings x points per ring "
                   "of pupil rays per beam; more rays narrow the sampling resolution of a cut power and cost time, memory and "
                   "ray data in proportion) and the scatter-origin and mesh-patch limits. Omitted fields use the Engine "
                   "defaults; null resets them all. The settings persist with the Project and apply to its next run.",
        "fields": {"compute_budget": "{max_interactions_per_path?, max_paths?, max_trace_steps?, max_interference_pairs?, max_return_analysis_work?, "
                                     "pupil_ring_count?, pupil_points_per_ring?, max_scatter_origins?, max_mesh_patches?} | null"},
    },
    "add_optical_model": {
        "summary": "Append a ProjectOpticalModelV1 to optical_models, with the materials / coatings it needs (an entry identical to one "
                   "already declared is shared; a different one with the same id is refused). Kinds: SURFACE_SEQUENCE, SURFACE_SET, "
                   "POLARIZATION_ELEMENT, NEUTRAL_DENSITY, APERTURE_STOP, IDEAL_THIN_LENS, TABULATED_THIN_LENS, FARADAY_ROTATOR, DIFFRACTION_GRATING, FACET_SOLID, NONLINEAR_CRYSTAL, COMPOUND (members: "
                   "other models of the Project). "
                   "catalog_model / import_prescription / build_native_model return ready documents.",
        "fields": {"optical_model": "{model_id, display_name?, model, frame?, design_geometry?}",
                   "model_materials": "[ProjectModelMaterialV1] (optional)", "model_coatings": "[PublicCoatingV1] (optional)"},
    },
    "update_optical_model": {
        "summary": "Merge fields into a declared model: display_name, model (the whole physics document, replaced), frame, "
                   "design_geometry; null removes (frame / design_geometry are then derived by os-optics).",
        "fields": {"model_id": "model id", "changes": "object; null removes"},
    },
    "remove_optical_model": {
        "summary": "Remove a declared model no component or COMPOUND unit uses (refused with IN_USE otherwise); materials and coatings "
                   "no remaining model uses go with it.",
        "fields": {"model_id": "model id"},
    },
    "add_model_material": {
        "summary": "Append a ProjectModelMaterialV1 (dispersion SELLMEIER {b, c_um2} | TABULATED | CONSTANT_INDEX {refractive_index}; "
                   "valid_band_nm; optional absorption_coefficient_per_mm). Material ids are uppercase; AIR is reserved.",
        "fields": {"material": "the material document"},
    },
    "update_model_material": {"summary": "Merge fields into a declared material (null removes).",
                              "fields": {"material_id": "material id", "changes": "object; null removes"}},
    "remove_model_material": {"summary": "Remove a declared material no model uses (IN_USE otherwise).", "fields": {"material_id": "material id"}},
    "add_model_coating": {"summary": "Append a coating document (PublicCoatingV1, e.g. a copy of a catalog coating).",
                          "fields": {"coating": "the coating document"}},
    "remove_model_coating": {"summary": "Remove a declared coating no model uses (IN_USE otherwise).", "fields": {"coating_id": "coating id"}},
    "add_component": {"summary": "Append an OpticalComponentInstanceV1.", "fields": {"component": "the component document"}},
    "remove_component": {"summary": "Remove a component.", "fields": {"instance_id": "component id"}},
    "set_observation_pose": {
        "summary": "Move / rotate an observation plane (its local +Z is the plane normal). A plane attached to a component "
                   "(device_instance_id) is posed in that component's local frame.",
        "fields": {"observation_id": "observation id", "position_mm": "[x, y, z] mm (optional)", "quaternion_wxyz": "[w, x, y, z] (optional)"},
    },
    "set_observation": {"summary": "Merge fields into an observation (e.g. aperture).",
                        "fields": {"observation_id": "observation id", "changes": "object; null removes"}},
    "add_observation": {"summary": "Append an ObservationV1.", "fields": {"observation": "the observation document"}},
    "remove_observation": {"summary": "Remove an observation (targets that use it must be removed first).",
                           "fields": {"observation_id": "observation id"}},
    "set_target": {
        "summary": "Merge fields into a target (minimum, maximum, target_value, tolerance_abs, tolerance_rel, metric, observation_id, "
                   "unit). null removes a field.",
        "fields": {"target_id": "target id", "changes": "object; null removes"},
    },
    "add_target": {"summary": "Append a TargetV1 (metric from capabilities().target_metrics).", "fields": {"target": "the target document"}},
    "remove_target": {"summary": "Remove a target.", "fields": {"target_id": "target id"}},
    "transform": {
        "summary": "Move the selected components and observation planes as one rigid body: translation_mm and / or rotation "
                   "({axis, angle_deg} or {quaternion_wxyz}) in the axes of frame (\"design\" or {component}), rotating about pivot "
                   "(\"frame_origin\", \"selection_centroid\", {point_mm} or {component}); computed by os_optics.composition. "
                   "Relative: a retry moves again (guard it with expected_project_hash).",
        "fields": {"selector": "{ids: [...], kinds: [...]}", "translation_mm": "[x, y, z] mm (optional)",
                   "rotation": "{axis, angle_deg} | {quaternion_wxyz} (optional)", "frame": "\"design\" | {component: id}",
                   "pivot": "\"frame_origin\" | \"selection_centroid\" | {point_mm} | {component}"},
    },
    "set_numerical_settings": {
        "summary": "Replace the Project's numerical settings: the validity limits, levels and match windows its results are "
                   "computed under. Omitted fields use the Engine defaults; null resets them all. The settings persist with the "
                   "Project and apply to its next run; capabilities.numerical_settings lists every field.",
        "fields": {"numerical_settings": "{power_epsilon_W?, negligible_power_fraction?, gaussian_*?, interference_*?, nonlinear_*?, "
                                         "rating_*?, design_point?: {single_wavelength_rel?, multi_order_retarder_rel?, "
                                         "single_angle_deg?: {<coating kind>?}, interference_shift_fraction?, "
                                         "interference_angle_max_deg?}} | null"},
    },
}

#: The refusal codes of an operation (``ProjectOpError.code``).
OP_ERROR_CODES = ("UNKNOWN_OP", "NOT_FOUND", "DUPLICATE_ID", "BAD_ARGUMENT", "IN_USE", "TRANSFORM_REFUSED")
DOF_AXES = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")
DESIGN_INTENTS = ("LAB_ASSEMBLY", "OPTICAL_SYSTEM")
AIR = "AIR"
VACUUM = "VACUUM"
LIBRARY_KEYS = ("optical_models", "model_materials", "model_coatings")
#: The longest custom_part_key a default ``own-<instance_id>`` key keeps.
CUSTOM_PART_KEY_MAX = 64
_ABSENT = object()


class ProjectOpError(ValueError):
    """An operation that cannot be applied: ``code`` (``OP_ERROR_CODES``), ``message``, ``op_index`` (-1: the list itself)
    and, for ``transform``, the composition ``problems``."""

    def __init__(self, code: str, message: str, op_index: int, problems: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.op_index = op_index
        self.problems = problems

    def as_dict(self) -> dict[str, Any]:
        problems = self.problems or [{"code": self.code, "message": self.message}]
        return {"code": "PROJECT_OP_INVALID", "message": self.message, "op_index": self.op_index, "op_code": self.code,
                "problems": problems}


def _q(value: Any) -> str:
    """A value quoted using compact JSON for stable operation errors."""

    if value is _ABSENT:
        return "undefined"
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


# ------------------------------------------------------------------------------------------------ declared-model usage


def materials_of(model: Any) -> list[str]:
    """The materials a model's physics names (media of a sequence; media and body of a surface set), AIR excluded."""

    m = _record(model)
    if m is None:
        return []
    out: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in out:
            out.append(value)

    if m.get("model_kind") == "SURFACE_SEQUENCE" and isinstance(m.get("media"), list):
        for medium in m["media"]:
            add(medium)
    if m.get("model_kind") == "SURFACE_SET":
        for surface in m.get("surfaces") if isinstance(m.get("surfaces"), list) else []:
            s = _record(surface) or {}
            add(s.get("medium_front"))
            add(s.get("medium_back"))
        add((_record(m.get("body")) or {}).get("medium"))
    return [x for x in out if x not in (AIR, VACUUM)]


def coatings_of(model: Any) -> list[str]:
    """The coatings a model's surfaces name."""

    m = _record(model)
    if m is None or not isinstance(m.get("surfaces"), list):
        return []
    out: list[str] = []
    for surface in m["surfaces"]:
        coating = (_record(surface) or {}).get("coating_id")
        if isinstance(coating, str) and coating not in out:
            out.append(coating)
    return out


def member_model_ids(model: Any) -> list[str]:
    """The member model ids of a COMPOUND unit (empty for any other kind)."""

    m = _record(model)
    if m is None or m.get("model_kind") != "COMPOUND" or not isinstance(m.get("members"), list):
        return []
    return [x["model_id"] for x in m["members"] if isinstance(x, dict) and isinstance(x.get("model_id"), str)]


def model_users(doc: Mapping[str, Any], model_id: str) -> dict[str, list[str]]:
    """Who uses a model: components naming it and COMPOUND units listing it as a member."""

    return {"components": [c.get("instance_id") for c in doc.get("components") or [] if isinstance(c, dict) and c.get("optical_model_id") == model_id],
            "units": [m.get("model_id") for m in doc.get("optical_models") or [] if isinstance(m, dict) and model_id in member_model_ids(m.get("model"))]}


def material_users(doc: Mapping[str, Any], material_id: str) -> list[str]:
    return [m.get("model_id") for m in doc.get("optical_models") or [] if isinstance(m, dict) and material_id in materials_of(m.get("model"))]


def coating_users(doc: Mapping[str, Any], coating_id: str) -> list[str]:
    return [m.get("model_id") for m in doc.get("optical_models") or [] if isinstance(m, dict) and coating_id in coatings_of(m.get("model"))]


# ------------------------------------------------------------------------------------------------ identical items


def _item_model(kind: str) -> Any:
    from .._vendor.shared_contracts.design_catalog import PublicCoatingV1
    from .._vendor.shared_contracts.optical_design import OpticalComponentInstanceV1
    from .._vendor.shared_contracts.optical_design.design_content import ObservationV1, TargetV1
    from .._vendor.shared_contracts.optical_design.optical_model import ProjectModelMaterialV1, ProjectOpticalModelV1

    return {"component": OpticalComponentInstanceV1, "observation": ObservationV1, "target": TargetV1, "optical_model": ProjectOpticalModelV1,
            "material": ProjectModelMaterialV1, "coating": PublicCoatingV1}[kind]


def _normalized(kind: str, item: Any) -> Any:
    """The item as the Project's payload writes it (``None`` when the contract does not accept it on its own)."""

    from pydantic import ValidationError

    try:
        return _item_model(kind).model_validate(item).model_dump(mode="json", by_alias=True, exclude_none=True)
    except (ValidationError, ValueError, TypeError, RecursionError):
        return None


def identical(kind: str, a: Any, b: Any) -> bool:
    """Whether two items of one kind are the same document: equal as JSON, or equal in the contract's normalized form."""

    if a == b:
        return True
    left = _normalized(kind, a)
    return left is not None and left == _normalized(kind, b)


def _merge_by_id(current: Any, incoming: Any, key: str, kind: str, index: int) -> tuple[list[Any], list[str]]:
    """A declared material / coating bundle merged into the Project's list: identical entries are shared, a different one
    under the same id is a conflict (returned, never overwritten)."""

    if incoming is None or incoming is _ABSENT:
        incoming = []
    if not isinstance(incoming, list) or not all(isinstance(item, dict) for item in incoming):
        raise ProjectOpError("BAD_ARGUMENT", f"{'model_materials' if kind == 'material' else 'model_coatings'} must be an array of objects", index)
    merged = list(current or [])
    conflicts: list[str] = []
    for item in incoming:
        existing = next((m for m in merged if isinstance(m, dict) and m.get(key) == item.get(key)), None)
        if existing is None:
            merged.append(copy.deepcopy(item))
        elif not identical(kind, existing, item):
            conflicts.append(str(item.get(key)))
    return merged, conflicts


# ------------------------------------------------------------------------------------------------ operation helpers


def _merge(target: Mapping[str, Any], changes: Any, index: int) -> dict[str, Any]:
    if not isinstance(changes, dict):
        raise ProjectOpError("BAD_ARGUMENT", "changes must be an object", index)
    out = dict(target)
    for key, value in changes.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _vector(value: Any, length: int, name: str, index: int) -> list[Any]:
    if not isinstance(value, list) or len(value) != length or not all(_number(v) for v in value):
        raise ProjectOpError("BAD_ARGUMENT", f"{name} must be an array of {length} finite numbers", index)
    return list(value)


def _find(items: Any, key: str, wanted: Any, what: str, index: int) -> int:
    # An op without an id finds an item without one (contract-invalid input).
    for at, item in enumerate(items or []):
        if isinstance(item, dict) and item.get(key, _ABSENT) == wanted:
            return at
    raise ProjectOpError("NOT_FOUND", f"{what} does not exist", index)


def _js_falsy(value: Any) -> bool:
    """JavaScript's ``!value`` for a JSON value: ``null`` / absent, ``false``, ``0``, ``""`` (an empty object or array is
    truthy)."""

    if value is None or value is _ABSENT or value is False or value == "":
        return True
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (value == 0 or value != value)


def _set_pose(holder: Mapping[str, Any], key: str, op: Mapping[str, Any], index: int) -> dict[str, Any]:
    position, quaternion = op.get("position_mm", _ABSENT), op.get("quaternion_wxyz", _ABSENT)
    if position is _ABSENT and quaternion is _ABSENT:
        raise ProjectOpError("BAD_ARGUMENT", "give position_mm and/or quaternion_wxyz", index)
    pose = dict(holder.get(key) or {})
    if position is not _ABSENT:
        pose["position_mm"] = _vector(position, 3, "position_mm", index)
    if quaternion is not _ABSENT:
        pose["quaternion_wxyz"] = _vector(quaternion, 4, "quaternion_wxyz", index)
    return {**holder, key: pose}


def _check_axis(axis: Any, index: int) -> str:
    if axis not in DOF_AXES:
        raise ProjectOpError("BAD_ARGUMENT", f"axis must be one of {', '.join(DOF_AXES)}", index)
    return axis


def _sorted_dofs(dofs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(dofs, key=lambda d: DOF_AXES.index(d.get("axis")) if isinstance(d, dict) and d.get("axis") in DOF_AXES else -1)


def _with_library(doc: dict[str, Any]) -> dict[str, Any]:
    """The library lists as the contract keeps them: an empty list is absent (never ``[]``, so a Project without declared
    models keeps its exact bytes and hashes)."""

    out = dict(doc)
    for key in LIBRARY_KEYS:
        if isinstance(out.get(key), list) and not out[key]:
            del out[key]
    return out


@dataclass
class _Applied:
    """The state of one batch: the document and the adds answered by an identical item already present."""

    doc: dict[str, Any]
    already_present: list[dict[str, Any]] = field(default_factory=list)


def _add_item(state: _Applied, list_key: str, id_key: str, kind: str, item: Any, index: int, op_name: str, label: str) -> None:
    doc = state.doc
    item_id = item.get(id_key) if isinstance(item, dict) else None
    if not isinstance(item_id, str):
        raise ProjectOpError("BAD_ARGUMENT", f"{label}.{id_key} is required", index)
    items = list(doc.get(list_key) or [])
    existing = next((x for x in items if isinstance(x, dict) and x.get(id_key) == item_id), None)
    if existing is not None:
        if identical(kind, existing, item):
            state.already_present.append({"op_index": index, "op": op_name, "kind": kind.upper(), "id": item_id})
            return
        raise ProjectOpError("DUPLICATE_ID", f"{_what(kind)} {_q(item_id)} already exists", index)
    items.append(copy.deepcopy(item))
    state.doc = {**doc, list_key: items}


def _what(kind: str) -> str:
    return {"optical_model": "optical model"}.get(kind, kind)


def _transform(doc: dict[str, Any], op: Mapping[str, Any], index: int) -> dict[str, Any]:
    """``composition.transform`` on the bare Project (no journal, no groups or instances)."""

    from ..composition.document import load_workspace
    from ..composition.errors import CompositionError
    from ..composition.operations import _MUTATORS, _request

    request = {k: v for k, v in op.items() if k != "op"}
    if "op_id" in request or "expected_base" in request:
        raise ProjectOpError("BAD_ARGUMENT", "a transform in a Project update takes no op_id / expected_base (guard the update with "
                             "expected_project_hash)", index)
    try:
        workspace = load_workspace(doc)
        payload, _summary, _warnings = _MUTATORS["transform"](workspace, _request(request, "transform"), None, None)
    except CompositionError as exc:
        problems = exc.as_dict()["problems"]
        message = "; ".join(f"{p['code']}: {p['message']}" for p in problems[:5])
        raise ProjectOpError("TRANSFORM_REFUSED", message, index, problems) from None
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProjectOpError("TRANSFORM_REFUSED", f"the transform could not be applied: {exc}", index) from None
    return payload["project"]


def _apply_one(state: _Applied, op: dict[str, Any], index: int) -> None:  # noqa: C901 - one branch per operation
    doc = state.doc
    components = list(doc.get("components") or [])
    observations = list(doc.get("observations") or [])
    targets = list(doc.get("targets") or [])
    name = op.get("op", _ABSENT)

    def component(instance_id: Any) -> int:
        return _find(components, "instance_id", instance_id, f"component {_q(instance_id)}", index)

    def observation(observation_id: Any) -> int:
        return _find(observations, "observation_id", observation_id, f"observation {_q(observation_id)}", index)

    def target(target_id: Any) -> int:
        return _find(targets, "target_id", target_id, f"target {_q(target_id)}", index)

    def with_component(at: int, item: dict[str, Any]) -> None:
        state.doc = {**doc, "components": [item if i == at else c for i, c in enumerate(components)]}

    def with_observation(at: int, item: dict[str, Any]) -> None:
        state.doc = {**doc, "observations": [item if i == at else o for i, o in enumerate(observations)]}

    get = op.get
    if name == "set_component_pose":
        at = component(get("instance_id", _ABSENT))
        with_component(at, _set_pose(components[at], "nominal_pose", op, index))
    elif name == "set_source_model":
        at = component(get("instance_id", _ABSENT))
        current = components[at].get("source_model")
        if _js_falsy(current):
            raise ProjectOpError("BAD_ARGUMENT", f"component {_q(get('instance_id'))} has no source_model", index)
        with_component(at, {**components[at], "source_model": _merge(current, get("changes"), index)})
    elif name == "set_retained_dofs":
        at = component(get("instance_id", _ABSENT))
        dofs = get("retained_dofs")
        if not isinstance(dofs, list):
            raise ProjectOpError("BAD_ARGUMENT", "retained_dofs must be an array", index)
        checked = []
        for dof in dofs:
            if not isinstance(dof, dict):
                raise ProjectOpError("BAD_ARGUMENT", f"axis must be one of {', '.join(DOF_AXES)}", index)
            checked.append({**copy.deepcopy(dof), "axis": _check_axis(dof.get("axis"), index)})
        with_component(at, {**components[at], "retained_dofs": _sorted_dofs(checked)})
    elif name == "set_retained_dof":
        at = component(get("instance_id", _ABSENT))
        axis = _check_axis(get("axis"), index)
        current = components[at].get("retained_dofs") or []
        others = [d for d in current if not (isinstance(d, dict) and d.get("axis") == axis)]
        control = get("control", _ABSENT)
        if control is not None and control not in ("MANUAL", "NUMERICAL"):
            raise ProjectOpError("BAD_ARGUMENT", "control must be MANUAL, NUMERICAL or null", index)
        # an axis that stays retained keeps what was stated about it (frame, travel, hard limits, repeatability, drift, unit)
        stated = next((d for d in current if isinstance(d, dict) and d.get("axis") == axis), None)
        kept = {**copy.deepcopy(stated), "axis": axis, "control": control} if stated is not None else {"axis": axis, "control": control}
        dofs = others if control is None else [*others, kept]
        with_component(at, {**components[at], "retained_dofs": _sorted_dofs(dofs)})
    elif name == "set_component_product":
        at = component(get("instance_id", _ABSENT))
        product = dict(components[at].get("product") or {})
        sku = get("optical_sku", _ABSENT)
        if sku is _ABSENT:
            product.pop("optical_sku", None)
        else:
            product["optical_sku"] = sku
        version = get("catalog_version", _ABSENT)
        if version is None:
            product.pop("catalog_version", None)
        elif version is not _ABSENT:
            product["catalog_version"] = version
        nxt = {**components[at], "product": product}
        if "optical_model_id" in nxt:  # a declared optic becomes a catalog component again (its model stays declared)
            nxt.pop("optical_model_id")
            nxt.pop("custom_part", None)
        with_component(at, nxt)
    elif name == "set_component_model":
        at = component(get("instance_id", _ABSENT))
        model_id = get("optical_model_id", _ABSENT)
        if not isinstance(model_id, str) or not any(isinstance(m, dict) and m.get("model_id") == model_id for m in doc.get("optical_models") or []):
            raise ProjectOpError("NOT_FOUND", f"optical model {_q(model_id)} is not declared in optical_models", index)
        current = components[at]
        nxt = {**current, "optical_model_id": model_id}
        nxt.pop("product", None)
        part = current.get("custom_part")
        if part is None:
            part = {"custom_part_key": f"own-{current.get('instance_id')}"[:CUSTOM_PART_KEY_MAX], "description": model_id}
        part = dict(part)
        if "custom_part_key" in op:
            part["custom_part_key"] = op["custom_part_key"]
        if "description" in op:
            part["description"] = op["description"]
        nxt["custom_part"] = part
        with_component(at, nxt)
    elif name == "set_compute_budget":
        from .._vendor.shared_contracts.optical_design.compute_budget import BUDGET_RANGES

        budget = get("compute_budget", _ABSENT)
        if budget is not None:
            if not isinstance(budget, dict) or any(k not in BUDGET_RANGES for k in budget):
                raise ProjectOpError("BAD_ARGUMENT", "compute_budget must be an object of budget fields or null", index)
            for field, value in budget.items():
                low, high = BUDGET_RANGES[field]
                if value is not None and (type(value) is not int or not low <= value <= high):
                    raise ProjectOpError("BAD_ARGUMENT", f"{field} must be an integer from {low} to {high} or null", index)
        nxt = dict(doc)
        if budget is None:
            nxt.pop("compute_budget", None)
        else:
            nxt["compute_budget"] = {k: v for k, v in budget.items() if v is not None}
        state.doc = nxt
    elif name == "set_numerical_settings":
        from pydantic import ValidationError as _Invalid

        from .._vendor.shared_contracts.optical_design.numerical_settings import NumericalSettingsV1

        settings = get("numerical_settings", _ABSENT)
        stated: dict[str, Any] = {}
        if settings is not None:
            if not isinstance(settings, dict):
                raise ProjectOpError("BAD_ARGUMENT", "numerical_settings must be an object of setting fields or null", index)
            try:
                stated = NumericalSettingsV1.model_validate(settings).model_dump(mode="json", exclude_none=True)
            except _Invalid as exc:
                first = exc.errors()[0]
                where = ".".join(str(part) for part in first["loc"])
                raise ProjectOpError("BAD_ARGUMENT", f"numerical_settings.{where}: {first['msg']}", index) from None
        nxt = dict(doc)
        if stated:
            nxt["numerical_settings"] = stated
        else:
            nxt.pop("numerical_settings", None)
        state.doc = nxt
    elif name == "set_design_intent":
        intent = get("design_intent", _ABSENT)
        if intent is not None and intent not in DESIGN_INTENTS:
            raise ProjectOpError("BAD_ARGUMENT", f"design_intent must be one of {', '.join(DESIGN_INTENTS)} or null", index)
        nxt = dict(doc)
        if intent is None:
            nxt.pop("design_intent", None)
        else:
            nxt["design_intent"] = intent
        state.doc = nxt
    elif name == "add_optical_model":
        model = get("optical_model")
        model_id = model.get("model_id") if isinstance(model, dict) else None
        physics = model.get("model") if isinstance(model, dict) else None
        if not isinstance(model_id, str) or not isinstance(physics, (dict, list)):
            raise ProjectOpError("BAD_ARGUMENT", "optical_model {model_id, model} is required", index)
        models = list(doc.get("optical_models") or [])
        existing = next((m for m in models if isinstance(m, dict) and m.get("model_id") == model_id), None)
        if existing is not None and not identical("optical_model", existing, model):
            raise ProjectOpError("DUPLICATE_ID", f"optical model {_q(model_id)} already exists", index)
        materials, material_conflicts = _merge_by_id(doc.get("model_materials"), get("model_materials"), "material_id", "material", index)
        coatings, coating_conflicts = _merge_by_id(doc.get("model_coatings"), get("model_coatings"), "coating_id", "coating", index)
        conflicts = [f"material {m}" for m in material_conflicts] + [f"coating {c}" for c in coating_conflicts]
        if conflicts:
            raise ProjectOpError("DUPLICATE_ID", f"{', '.join(conflicts)} already declared with other data", index)
        if existing is not None:
            state.already_present.append({"op_index": index, "op": name, "kind": "OPTICAL_MODEL", "id": model_id})
            models_out = models
        else:
            models_out = [*models, copy.deepcopy(model)]
        state.doc = _with_library({**doc, "optical_models": models_out, "model_materials": materials, "model_coatings": coatings})
    elif name == "update_optical_model":
        models = list(doc.get("optical_models") or [])
        at = _find(models, "model_id", get("model_id", _ABSENT), f"optical model {_q(get('model_id', _ABSENT))}", index)
        changes = get("changes")
        if isinstance(changes, dict) and "model_id" in changes:
            raise ProjectOpError("BAD_ARGUMENT", "a model keeps its model_id (add a new model to use another id)", index)
        nxt = _merge(models[at], changes, index)
        state.doc = {**doc, "optical_models": [nxt if i == at else m for i, m in enumerate(models)]}
    elif name == "remove_optical_model":
        models = list(doc.get("optical_models") or [])
        model_id = get("model_id", _ABSENT)
        _find(models, "model_id", model_id, f"optical model {_q(model_id)}", index)
        users = model_users(doc, model_id)
        if users["components"] or users["units"]:
            named = ", ".join([f"component {c}" for c in users["components"]] + [f"unit {u}" for u in users["units"]])
            raise ProjectOpError("IN_USE", f"optical model {_q(model_id)} is used by {named}", index)
        rest = {**doc, "optical_models": [m for m in models if not (isinstance(m, dict) and m.get("model_id") == model_id)]}
        # materials and coatings belong to models: those no remaining model uses go with it
        state.doc = _with_library({
            **rest,
            "model_materials": [m for m in doc.get("model_materials") or [] if not isinstance(m, dict)
                                or material_users(rest, m.get("material_id")) or model_id not in material_users(doc, m.get("material_id"))],
            "model_coatings": [c for c in doc.get("model_coatings") or [] if not isinstance(c, dict)
                               or coating_users(rest, c.get("coating_id")) or model_id not in coating_users(doc, c.get("coating_id"))],
        })
    elif name == "add_model_material":
        material = get("material")
        if not isinstance(material, dict) or not isinstance(material.get("material_id"), str):
            raise ProjectOpError("BAD_ARGUMENT", "material.material_id is required", index)
        _add_item(state, "model_materials", "material_id", "material", material, index, name, "material")
        state.doc = _with_library(state.doc)
    elif name == "update_model_material":
        materials = list(doc.get("model_materials") or [])
        at = _find(materials, "material_id", get("material_id", _ABSENT), f"material {_q(get('material_id', _ABSENT))}", index)
        changes = get("changes")
        if isinstance(changes, dict) and "material_id" in changes:
            raise ProjectOpError("BAD_ARGUMENT", "a material keeps its material_id", index)
        nxt = _merge(materials[at], changes, index)
        state.doc = {**doc, "model_materials": [nxt if i == at else m for i, m in enumerate(materials)]}
    elif name == "remove_model_material":
        material_id = get("material_id", _ABSENT)
        _find(doc.get("model_materials"), "material_id", material_id, f"material {_q(material_id)}", index)
        users = material_users(doc, material_id)
        if users:
            raise ProjectOpError("IN_USE", f"material {_q(material_id)} is used by model {', '.join(users)}", index)
        state.doc = _with_library({**doc, "model_materials": [m for m in doc.get("model_materials") or []
                                                              if not isinstance(m, dict) or m.get("material_id") != material_id]})
    elif name == "add_model_coating":
        coating = get("coating")
        if not isinstance(coating, dict) or not isinstance(coating.get("coating_id"), str):
            raise ProjectOpError("BAD_ARGUMENT", "coating.coating_id is required", index)
        _add_item(state, "model_coatings", "coating_id", "coating", coating, index, name, "coating")
        state.doc = _with_library(state.doc)
    elif name == "remove_model_coating":
        coating_id = get("coating_id", _ABSENT)
        _find(doc.get("model_coatings"), "coating_id", coating_id, f"coating {_q(coating_id)}", index)
        users = coating_users(doc, coating_id)
        if users:
            raise ProjectOpError("IN_USE", f"coating {_q(coating_id)} is used by model {', '.join(users)}", index)
        state.doc = _with_library({**doc, "model_coatings": [c for c in doc.get("model_coatings") or []
                                                             if not isinstance(c, dict) or c.get("coating_id") != coating_id]})
    elif name == "add_component":
        _add_item(state, "components", "instance_id", "component", get("component"), index, name, "component")
    elif name == "remove_component":
        at = component(get("instance_id", _ABSENT))
        state.doc = {**doc, "components": [c for i, c in enumerate(components) if i != at]}
    elif name == "set_observation_pose":
        at = observation(get("observation_id", _ABSENT))
        with_observation(at, _set_pose(observations[at], "pose", op, index))
    elif name == "set_observation":
        at = observation(get("observation_id", _ABSENT))
        with_observation(at, _merge(observations[at], get("changes"), index))
    elif name == "add_observation":
        _add_item(state, "observations", "observation_id", "observation", get("observation"), index, name, "observation")
    elif name == "remove_observation":
        at = observation(get("observation_id", _ABSENT))
        state.doc = {**doc, "observations": [o for i, o in enumerate(observations) if i != at]}
    elif name == "set_target":
        at = target(get("target_id", _ABSENT))
        state.doc = {**doc, "targets": [_merge(t, get("changes"), index) if i == at else t for i, t in enumerate(targets)]}
    elif name == "add_target":
        _add_item(state, "targets", "target_id", "target", get("target"), index, name, "target")
    elif name == "remove_target":
        at = target(get("target_id", _ABSENT))
        state.doc = {**doc, "targets": [t for i, t in enumerate(targets) if i != at]}
    elif name == "transform":
        state.doc = _transform(doc, op, index)
    else:
        raise _unknown_op(name, index)


def _unknown_op(name: Any, index: int) -> ProjectOpError:
    """UNKNOWN_OP: an op without an ``op`` name, or the unknown name quoted."""

    known = ", ".join(PROJECT_OPS)
    if name is _ABSENT:
        return ProjectOpError("UNKNOWN_OP", f'an op has no "op" name; known: {known}', index)
    return ProjectOpError("UNKNOWN_OP", f"unknown op {_q(name)}; known: {known}", index)


def apply_ops(document: Mapping[str, Any], ops: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``apply_project_ops`` plus the adds answered by an identical item already present (``already_present``)."""

    if not isinstance(ops, list) or not ops:
        raise ProjectOpError("BAD_ARGUMENT", "ops must be a non-empty array", -1)
    state = _Applied(copy.deepcopy(dict(document)))
    for index, op in enumerate(ops):
        if isinstance(op, list):  # an array names no operation
            raise _unknown_op(_ABSENT, index)
        if not isinstance(op, dict):
            raise ProjectOpError("BAD_ARGUMENT", "each op must be an object", index)
        try:
            _apply_one(state, op, index)
        except ProjectOpError:
            raise
        except (AttributeError, TypeError, KeyError, IndexError, ValueError):
            # a document that does not follow the Project contract where this op reads it (a list that is not a list of
            # objects, ...): a refusal of this op, never an exception
            raise ProjectOpError("BAD_ARGUMENT", f"op {index} ({_q(op.get('op'))}) cannot be applied: a field it reads does "
                                                 "not follow the Project contract; check the Project for the problems",
                                 index) from None
    return state.doc, state.already_present


def apply_project_ops(document: Mapping[str, Any], ops: Any) -> dict[str, Any]:
    """Apply ``ops`` to a copy of the Project JSON ``document`` (module docstring) and return it; the input is never
    changed. Raises ``ProjectOpError`` (with the failing op's index) on the first operation that cannot be applied."""

    return apply_ops(document, ops)[0]


# ------------------------------------------------------------------------------------------------ what changed


#: The Project's lists of named items: (list key, id key, affected kind).
ITEM_LISTS = (("components", "instance_id", "COMPONENT"), ("observations", "observation_id", "OBSERVATION"), ("targets", "target_id", "TARGET"),
              ("optical_models", "model_id", "OPTICAL_MODEL"), ("model_materials", "material_id", "MODEL_MATERIAL"),
              ("model_coatings", "coating_id", "MODEL_COATING"))


def affected_objects(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What differs between two Project documents: per named item (component, observation, target, declared model,
    material, coating) ``{kind, id, change: ADDED | REMOVED | MODIFIED, fields}`` with the item's top-level fields that
    differ, and ``{kind: PROJECT, id: None, change: MODIFIED, fields}`` for other Project fields (e.g. ``design_intent``)."""

    def named(items: Any, id_key: str) -> dict[Any, dict[str, Any]]:
        # the named objects of a list (a document that is not a valid Project may hold anything there)
        return {x.get(id_key): x for x in (items if isinstance(items, list) else ()) if isinstance(x, dict)
                and isinstance(x.get(id_key), (str, int, float, bool, type(None)))}

    out: list[dict[str, Any]] = []
    listed = {key for key, _, _ in ITEM_LISTS}
    for key, id_key, kind in ITEM_LISTS:
        old = named(before.get(key), id_key)
        new = named(after.get(key), id_key)
        for item_id, item in new.items():
            if item_id not in old:
                out.append({"kind": kind, "id": item_id, "change": "ADDED", "fields": sorted(item)})
            elif item != old[item_id]:
                fields = sorted(k for k in set(item) | set(old[item_id]) if item.get(k, _ABSENT) != old[item_id].get(k, _ABSENT))
                out.append({"kind": kind, "id": item_id, "change": "MODIFIED", "fields": fields})
        for item_id, item in old.items():
            if item_id not in new:
                out.append({"kind": kind, "id": item_id, "change": "REMOVED", "fields": sorted(item)})
    fields = sorted(k for k in (set(before) | set(after)) - listed if before.get(k, _ABSENT) != after.get(k, _ABSENT))
    if fields:
        out.append({"kind": "PROJECT", "id": None, "change": "MODIFIED", "fields": fields})
    return out


# ------------------------------------------------------------------------------------------------ validated update


#: Refusal codes of ``update_project`` (``ProjectUpdateError.code``).
PROJECT_OP_INVALID = "PROJECT_OP_INVALID"
PROJECT_REJECTED = "PROJECT_REJECTED"
PROJECT_CHANGED = "PROJECT_CHANGED"


class ProjectUpdateError(ValueError):
    """A refused update; nothing changed. ``code``: ``PROJECT_OP_INVALID`` (``op_index``, ``op_code``),
    ``PROJECT_REJECTED`` (``problems``: ``LocalRunErrorCode`` problems -- the edited document is no Project, or, with
    ``require_check``, the check refused it; ``optical_solids`` when the solid check refused) or ``PROJECT_CHANGED``
    (``project_hash``: the Project's current hash)."""

    def __init__(self, code: str, message: str, *, problems: list[dict[str, Any]] | None = None, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.problems = problems or []
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details, "problems": list(self.problems)}


@dataclass(frozen=True)
class ProjectUpdate:
    """An applied update: the edited Project (``project``, its normalized document, and ``project_hash``), the check of
    it (``check``: ``json_api.check_project``'s reply, accepted or refused; ``check_ok``, ``check_problems``), the hash of
    the Project it was applied to (``base_project_hash``; ``None`` when that document was not a valid Project), whether
    the Project changed, what changed (``affected``), the adds answered by an identical item already present and the
    check's advisory ``warnings``. What only an accepted check states (``engine_input_hash``, ``catalog_version``,
    ``compute_budget``, ``numerical_settings``, ``warnings``) is null or empty when the check did not pass;
    ``optical_solids`` holds the solid report whenever the solid check ran."""

    project: dict[str, Any]
    project_hash: str
    design_intent: str
    check: dict[str, Any]
    base_project_hash: str | None
    affected: tuple[dict[str, Any], ...]
    already_present: tuple[dict[str, Any], ...]

    @property
    def check_ok(self) -> bool:
        return bool(self.check["ok"])

    @property
    def check_problems(self) -> list[dict[str, Any]]:
        return [] if self.check_ok else list(self.check["error"]["problems"])

    @property
    def changed(self) -> bool:
        return self.project_hash != self.base_project_hash

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return list(self.check.get("warnings") or [])

    def as_dict(self) -> dict[str, Any]:
        return {"project": self.project, "project_hash": self.project_hash, "engine_input_hash": self.check.get("engine_input_hash"),
                "base_project_hash": self.base_project_hash, "changed": self.changed, "affected": list(self.affected),
                "already_present": list(self.already_present), "check": {"ok": self.check_ok, "problems": self.check_problems},
                "warnings": self.warnings,
                "catalog_version": self.check.get("catalog_version"), "design_intent": self.design_intent,
                "optical_solids": self.check.get("optical_solids"), "compute_budget": self.check.get("compute_budget"),
                "numerical_settings": self.check.get("numerical_settings")}


def _base_document(document: Any) -> tuple[dict[str, Any], str | None]:
    """The document the operations apply to: the normalized payload of a valid Project (and its hash), else the JSON
    object as given (hash ``None``: a Project the operations may repair)."""

    from .._vendor.shared_contracts.optical_design import ProjectDocumentV2
    from .catalog import _document, load_project
    from .errors import LocalRunError, LocalRunErrorCode

    if isinstance(document, ProjectDocumentV2):
        return document.payload(), document.content_hash()
    try:
        raw = _document(document, "project", LocalRunErrorCode.PROJECT_INVALID)
    except LocalRunError as exc:
        raise ProjectUpdateError(PROJECT_REJECTED, "the document is not a Project", problems=[p.as_dict() for p in exc.problems]) from None
    if not isinstance(raw, dict):
        raise ProjectUpdateError(PROJECT_REJECTED, "the document is not a Project",
                                 problems=[{"code": LocalRunErrorCode.PROJECT_INVALID.value, "message": "a Project is a JSON object"}])
    try:
        project = load_project(raw)
    except LocalRunError:
        return copy.deepcopy(raw), None
    return project.payload(), project.content_hash()


def update_project(document: Any, ops: Any, *, expected_project_hash: str | None = None, catalog: Any = None,
                   require_check: bool = False) -> ProjectUpdate:
    """Apply ``ops`` (module docstring) to a Project and report the check of the result (``json_api.check_project``) --
    atomically: a refusal raises ``ProjectUpdateError`` and nothing changes. The edited document must be a Project
    (``PROJECT_REJECTED`` otherwise); the check is reported in the update (``check_ok``, ``check_problems``) and, with
    ``require_check``, required (``PROJECT_REJECTED`` with the check's problems). ``document``: a Project document (dict,
    JSON text or bytes, a file path) or a ``ProjectDocumentV2``; ``expected_project_hash``: the osc1 hash the operations
    were written against (``PROJECT_CHANGED`` when the Project's hash differs); ``catalog``: a catalog release to check
    against instead of the one the Project pins."""

    from .catalog import load_project
    from .errors import LocalRunError
    from .json_api import check_project

    if not isinstance(require_check, bool):
        raise ProjectUpdateError(PROJECT_OP_INVALID, "require_check is true or false", op_index=-1, op_code="BAD_ARGUMENT",
                                 problems=[{"code": "BAD_ARGUMENT", "message": "require_check is true or false"}])

    if expected_project_hash is not None and not isinstance(expected_project_hash, str):
        raise ProjectUpdateError(PROJECT_OP_INVALID, "expected_project_hash is a Project hash (a string)", op_index=-1, op_code="BAD_ARGUMENT",
                                 problems=[{"code": "BAD_ARGUMENT", "message": "expected_project_hash is a Project hash (a string)"}])
    base, base_hash = _base_document(document)
    if expected_project_hash is not None and expected_project_hash != base_hash:
        message = ("the Project changed since the operations were written (its hash is not expected_project_hash); read it again"
                   if base_hash is not None else "the document is not a valid Project, so it has no hash to compare with expected_project_hash")
        raise ProjectUpdateError(PROJECT_CHANGED, message, project_hash=base_hash, expected_project_hash=expected_project_hash,
                                 problems=[{"code": PROJECT_CHANGED, "message": message}])
    try:
        edited, already = apply_ops(base, ops)
    except ProjectOpError as exc:
        details = exc.as_dict()
        raise ProjectUpdateError(PROJECT_OP_INVALID, exc.message, problems=details.pop("problems"), op_index=exc.op_index,
                                 op_code=exc.code) from None
    refused = "os-optics refused the edited Project; the Project is unchanged"
    try:
        project = load_project(edited)
    except LocalRunError as exc:  # (no Project: there is nothing to keep, with or without the check)
        raise ProjectUpdateError(PROJECT_REJECTED, refused, problems=[p.as_dict() for p in exc.problems]) from None
    check = check_project(project, catalog=catalog)
    if require_check and not check["ok"]:
        extra = {"optical_solids": check["optical_solids"]} if check.get("optical_solids") is not None else {}
        raise ProjectUpdateError(PROJECT_REJECTED, refused, problems=check["error"]["problems"], **extra)
    payload = project.payload()
    return ProjectUpdate(project=payload, project_hash=project.content_hash(), design_intent=project.effective_design_intent.value,
                         check=check, base_project_hash=base_hash, affected=tuple(affected_objects(base, payload)),
                         already_present=tuple(already))


__all__ = ["DESIGN_INTENTS", "DOF_AXES", "OP_ERROR_CODES", "PROJECT_CHANGED", "PROJECT_OPS", "PROJECT_OP_INVALID", "PROJECT_REJECTED",
           "ProjectOpError", "ProjectUpdate", "ProjectUpdateError", "affected_objects", "apply_ops", "apply_project_ops", "identical",
           "update_project"]
