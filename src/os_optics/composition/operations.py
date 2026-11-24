"""The composition operations: atomic, retry-safe edits of an ``OpticalWorkspaceV1``.

Every operation is a pure function: it takes a workspace (and a request) and returns an ``OperationResult`` holding a new
workspace, or raises ``CompositionError`` -- the given workspace is never modified.

Retry and concurrency (every operation that changes the workspace):

* ``op_id`` (1..128 printable ASCII characters, chosen by the caller) names the operation. The journal
  (``OpticalWorkspaceV1.operations``, the last 256) records each applied operation with the digest of its request. The
  same ``op_id`` with the same request again returns the recorded result and changes nothing (``replayed``); the same
  ``op_id`` with another request is refused (``OP_ID_REUSED``). Use one ``op_id`` per intended operation and reuse it
  when retrying after a lost response. An operation without ``op_id`` is always applied (never answered from the
  journal) and recorded under a generated id that is unique in the journal (``auto-...``, ``result.op_id``): only an
  ``op_id`` the caller chose makes a retry safe.
* ``expected_base`` (a ``workspace_digest``) states which workspace the request was written for; a new operation on
  another workspace is refused (``BASE_CHANGED``, with the current digest).
* An import names its instance with ``instance_key``: importing the same module with the same placement under an
  existing key is answered with the existing instance (nothing is added); a second instance needs a new key.

Physical check: the local check (``json_api.check_project``) of the resulting Project is reported with the operation,
not required by it. ``result.check`` holds ``ok`` (a run accepts the result), ``new_problems`` (the problems the
workspace did not have before: two optical bodies overlapping, a reference that no longer resolves) and
``carried_problems`` (the ones it already had), so a design is composed and repaired one operation at a time. A
request with ``require_check`` true makes the check a condition: a result with a new problem is refused
(``CHECK_PROJECT_REFUSED``). ``require_check`` is an option of the call, not part of the operation: it does not enter
the request digest, and a replayed ``op_id`` returns the recorded result whatever it says. In a ``batch`` the final
state is checked, and the batch carries the option.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .._vendor.shared_contracts.common.hashes import osc1_digest
from .._vendor.shared_contracts.common.osc1 import Osc1Error, canonicalize_osc1_text
from .._vendor.shared_contracts.optical_design import ModuleDescriptorV1, OpticalWorkspaceV1, ProjectDocumentV2
from .._vendor.shared_contracts.optical_design.workspace import MAX_OPERATIONS, MAX_SUMMARY_CHARS, OperationId
from ..local.catalog import _validation_message
from ..local.errors import LocalRunError
from . import rotation as rot
from .document import load_module, load_workspace, validate_payload
from .errors import CompositionError, CompositionErrorCode, CompositionProblem, CompositionWarningCode, refuse, warning
from .importing import (
    CATALOG_BINDINGS,
    LACKED_MEMBER_DIGEST,
    LIBRARY,
    OBJECTS,
    SOURCED,
    digest_refs,
    import_module,
    member_digests,
    parse_fixture_policy,
    pose_dict,
    pose_tuple,
    valid_id,
)
from .importing import without_pin as _without_pin
from .selection import Selection, instance_members, resolve_selection

_E = CompositionErrorCode
_OP_ID = TypeAdapter(OperationId)
#: Request keys every changing operation takes besides its own.
COMMON_KEYS = ("op_id", "expected_base", "require_check")
REQUEST_KEYS = {
    "import": ("instance_key", "placement", "label", "fixtures", "catalog_binding", "group_id"),
    "transform": ("selector", "translation_mm", "rotation", "frame", "pivot"),
    "group_create": ("group_id", "label", "members", "frame"),
    "group_dissolve": ("group_id",),
    "group_set_members": ("group_id", "members"),
    "group_add": ("group_id", "members"),
    "group_remove": ("group_id", "members"),
    "replace_instance": ("instance_key", "discard_edits", "fixtures", "catalog_binding", "label"),
    "update_project": ("project",),
    "batch": ("operations",),
}
MODULE_KINDS = frozenset({"import", "replace_instance"})
BATCH_KINDS = tuple(k for k in REQUEST_KEYS if k != "batch")
#: Characters of a workspace digest (SHA-256, hexadecimal).
DIGEST_LENGTH = 64
#: A generated op_id is ``auto-<AUTO_ID_DIGITS hexadecimal digits>``.
AUTO_PREFIX = "auto-"
AUTO_ID_DIGITS = 32
#: The longest module description (``PublicText``).
DESCRIPTION_MAX = 1000
ELLIPSIS = "..."
#: The moved-edit digest earlier builds gave every edited or deleted component or observation plane alike, whatever its
#: digest before the move: it says nothing about which import content the member had, so it stays as it is through
#: further moves and never shows that two records hold the same content.
UNDERIVED_MOVED_DIGEST = osc1_digest({"edited_member_moved_with_its_instance": True})


def moved_edit_digest(digest: str) -> str:
    """The digest an edited or deleted component or observation plane gets when its instance moves as a whole
    (``MemberDigestV1``), from its digest before the move: its import content after the move is not known, and no
    content has this digest, so the member stays an edit of the instance (also when the move brings it back onto its old
    digest, or when it is added back). Records that held the same import content and moved together keep agreeing (the
    same digest), so a nested replacement that re-creates the member from that content's module object gives them its
    content (``_follow_recreated``); a record that moved on its own no longer agrees with the others. The digest earlier
    builds wrote (``UNDERIVED_MOVED_DIGEST``) stays."""

    if digest == UNDERIVED_MOVED_DIGEST:
        return digest
    return osc1_digest({"edited_member_moved_with_its_instance": digest})


#: The digest with which an enclosing instance records a member that one of its nested instances gained when that nested
#: instance was replaced on its own (``MemberDigestV1``): the enclosing instance's import never had it, so it is an
#: edit of the enclosing instance (``ADDED``) whatever its content, also through whole-instance moves.
ADDED_EDIT_DIGEST = osc1_digest({"member_added_by_a_nested_replacement": True})
#: The member digest field naming the object and placement a nested instance had for a member of an enclosing import
#: that a replacement of it changed or left out (``MemberDigestV1``, ``nested_revision_digest``).
_NESTED_REVISION = "nested_revision_digest"
#: Digests that stand for no content (``_follow_digests`` leaves them as they are).
_SENTINEL_DIGESTS = frozenset({ADDED_EDIT_DIGEST, LACKED_MEMBER_DIGEST})
#: The field holding the pose a placement gives a component / an observation plane.
_POSE_FIELD = {"COMPONENT": "nominal_pose", "OBSERVATION": "pose"}
#: The kinds of definitions (optical models, materials, coatings).
_DEFINITION_KINDS = frozenset(kind for kind, _, _ in LIBRARY)


@dataclass(frozen=True)
class OperationResult:
    """An applied (or replayed) operation: the new ``workspace``, the ``result`` an Agent reads (affected ids, poses,
    mappings, warnings) and whether it was ``replayed`` from the journal (nothing changed)."""

    workspace: OpticalWorkspaceV1
    result: dict[str, Any]
    replayed: bool = False

    @property
    def workspace_digest(self) -> str:
        return self.workspace.workspace_digest()

    def as_dict(self) -> dict[str, Any]:
        return {"workspace": self.workspace.payload(), "workspace_digest": self.workspace_digest, "replayed": self.replayed,
                "result": self.result}


class _Unchanged(Exception):
    """An import under an existing key of the same module and placement: the existing instance answers it."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("unchanged")
        self.result = result


# ------------------------------------------------------------------------------------------------ helpers


def _request(request: Any, kind: str) -> dict[str, Any]:
    if request is None:
        request = {}
    if not isinstance(request, Mapping):
        raise refuse(_E.REQUEST_INVALID, f"a {kind} request is a JSON object")
    request = copy.deepcopy(dict(request))
    unknown = sorted(set(request) - set(REQUEST_KEYS[kind]) - set(COMMON_KEYS))
    if unknown:
        raise refuse(_E.REQUEST_INVALID, f"a {kind} request takes {sorted(REQUEST_KEYS[kind] + COMMON_KEYS)}, not {unknown}")
    return request


def _vector(value: Any, name: str, code: CompositionErrorCode = _E.REQUEST_INVALID) -> tuple[float, float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise refuse(code, f"{name} is a list of three numbers", name)
    vector = tuple(float(v) for v in value)
    if not all(v == v and abs(v) != float("inf") for v in vector):
        raise refuse(code, f"{name} holds finite numbers", name)
    return vector  # type: ignore[return-value]


def _pose_value(value: Any, name: str) -> tuple[rot.Vector, rot.Quaternion]:
    if not isinstance(value, Mapping) or set(value) - {"position_mm", "quaternion_wxyz"}:
        raise refuse(_E.REQUEST_INVALID, f"{name} is a pose {{position_mm, quaternion_wxyz}}", name)
    position = _vector(value.get("position_mm", [0.0, 0.0, 0.0]), f"{name}.position_mm")
    q = value.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    if not isinstance(q, (list, tuple)) or len(q) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in q):
        raise refuse(_E.REQUEST_INVALID, f"{name}.quaternion_wxyz is a list of four numbers", name)
    try:
        return position, rot.stated_quaternion(q)
    except ValueError:
        raise refuse(_E.REQUEST_INVALID, f"{name}.quaternion_wxyz is a non-zero quaternion (its length does not matter)", name) from None


def _project_problems(project: ProjectDocumentV2, catalog: Any) -> list[Any]:
    from ..local.lowering import lower_project

    try:
        lower_project(project, catalog)
    except LocalRunError as exc:
        return list(exc.problems)
    return []


def _problem_key(problem: Any) -> tuple[Any, ...]:
    details = problem.details or {}
    return (problem.code.value, problem.location, details.get("other_instance_id"), details.get("other_member_id"))


def _gate(base: ProjectDocumentV2, result: ProjectDocumentV2, catalog: Any, require_check: bool = False) -> dict[str, Any]:
    """The local check of ``result`` (module docstring): whether it passes, the problems ``base`` did not have and the
    ones it had. With ``require_check`` a new problem refuses the operation. A result that is the base's Project is not
    checked."""

    if base.content_hash() == result.content_hash():
        return {"checked": False, "carried_problems": []}
    after = _project_problems(result, catalog)
    before = {_problem_key(p) for p in _project_problems(base, catalog)} if after else set()
    new = [p for p in after if _problem_key(p) not in before]
    if new and require_check:
        raise refuse(_E.CHECK_PROJECT_REFUSED, "the result has problems the local check refuses: "
                     + "; ".join(f"{p.code.value}{f' at {p.location}' if p.location else ''}: {p.message}" for p in new[:5])
                     + (f" ... and {len(new) - 5} more" if len(new) > 5 else ""), problems=[p.as_dict() for p in new])
    return {"checked": True, "ok": not after, "new_problems": [p.as_dict() for p in new],
            "carried_problems": [p.as_dict() for p in after if _problem_key(p) in before]}


def _bounded(summary: dict[str, Any]) -> dict[str, Any]:
    """``summary`` when it fits the journal (``MAX_SUMMARY_CHARS``), else its compact form."""

    if len(canonicalize_osc1_text(summary)) <= MAX_SUMMARY_CHARS:
        return summary
    compact = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
    compact["truncated"] = True
    compact["counts"] = {k: len(v) for k, v in summary.items() if isinstance(v, (list, dict))}
    return compact


def _journal(workspace: dict[str, Any], record: dict[str, Any]) -> None:
    operations = list(workspace.get("operations") or [])
    operations.append(record)
    workspace["operations"] = operations[-MAX_OPERATIONS:]


def _object_pose(project: ProjectDocumentV2, kind: str, object_id: str) -> dict[str, Any]:
    if kind == "COMPONENT":
        component = next(c for c in project.components if c.instance_id == object_id)
        return component.nominal_pose.model_dump(mode="json")
    observation = next(o for o in project.observations if o.observation_id == object_id)
    return project.observation_design_pose(observation).model_dump(mode="json")


# ------------------------------------------------------------------------------------------------ the operation frame


def apply_operation(kind: str, workspace: Any, request: Any, *, module: Any = None, catalog: Any = None) -> OperationResult:
    """Apply one operation (``kind``: a key of ``REQUEST_KEYS``) with the journal, base and physical-check rules of the
    module docstring."""

    if kind not in REQUEST_KEYS:
        raise refuse(_E.REQUEST_INVALID, f"unknown operation {kind!r}; operations: {sorted(REQUEST_KEYS)}")
    ws = load_workspace(workspace)
    base = ws.workspace_digest()
    try:
        request = _request(request, kind)
        op_id = request.pop("op_id", None)
        expected = request.pop("expected_base", None)
        require_check = request.pop("require_check", False)
        if not isinstance(require_check, bool):
            raise refuse(_E.REQUEST_INVALID, "require_check is true or false", "require_check")
        if op_id is not None:
            try:
                _OP_ID.validate_python(op_id)
            except ValidationError:
                raise refuse(_E.REQUEST_INVALID, "op_id is 1 to 128 printable ASCII characters without spaces", "op_id") from None
        if expected is not None and not (isinstance(expected, str) and len(expected) == DIGEST_LENGTH):
            raise refuse(_E.REQUEST_INVALID, "expected_base is a workspace digest (64 hexadecimal characters)", "expected_base")
        module_ws = None
        if kind in MODULE_KINDS:
            module_ws = load_module(module)
        elif module is not None:
            raise refuse(_E.REQUEST_INVALID, f"a {kind} operation takes no module")
        try:
            request_digest = osc1_digest({"kind": kind, "request": request,
                                          "module_digest": module_ws.module_digest() if module_ws is not None else None})
        except Osc1Error as exc:
            raise refuse(_E.REQUEST_INVALID, f"the request holds a value that is not plain JSON ({exc})") from None
        if op_id is None:
            op_id = _generated_op_id(ws, request_digest, base)
        else:
            recorded = ws.operation(op_id)
            if recorded is not None:
                if recorded.kind == kind and recorded.request_digest == request_digest:
                    return OperationResult(ws, _replayed(ws, recorded.summary), replayed=True)
                raise refuse(_E.OP_ID_REUSED, f"op_id {op_id!r} was already used for another request ({recorded.kind}); use a new op_id", "op_id",
                             op_id=op_id, recorded_kind=recorded.kind)
        if expected is not None and expected != base:
            raise refuse(_E.BASE_CHANGED, "the workspace changed since the request was written; read it again", "expected_base",
                         current_digest=base, expected_base=expected)
        try:
            payload, summary, warnings = _MUTATORS[kind](ws, request, module_ws, catalog)
        except _Unchanged as unchanged:
            result = {"op_id": op_id, "kind": kind, **unchanged.result}
            out = ws.payload()
            _journal(out, {"op_id": op_id, "kind": kind, "request_digest": request_digest, "base_digest": base, "result_digest": base,
                           "summary": _bounded(result)})
            return OperationResult(validate_payload(out), result, replayed=True)
        new_ws = validate_payload(payload)
        check = _gate(ws.project, new_ws.project, catalog, require_check)
        result = {"op_id": op_id, "kind": kind, **summary, "warnings": [w.as_dict() for w in warnings], "check": check}
        out = new_ws.payload()
        _journal(out, {"op_id": op_id, "kind": kind, "request_digest": request_digest, "base_digest": base,
                       "result_digest": new_ws.workspace_digest(), "summary": _bounded(result)})
        return OperationResult(validate_payload(out), result)
    except CompositionError as exc:
        exc.workspace_digest = base
        raise
    except (ValueError, TypeError, OverflowError) as exc:  # an input outside the JSON / number domain the checks above missed
        raise CompositionError([CompositionProblem(_E.REQUEST_INVALID, f"the request could not be applied: {exc}")], workspace_digest=base) from None


def _generated_op_id(ws: OpticalWorkspaceV1, request_digest: str, base: str) -> str:
    """The op_id of an operation the caller did not name: unique in the journal (it follows the journal's last entry and
    length), so such an operation is always applied, never answered from the journal. The same in every runtime."""

    previous = ws.operations[-1].op_id if ws.operations else None
    salt = 0
    while True:
        op_id = AUTO_PREFIX + osc1_digest({"previous": previous, "count": len(ws.operations), "request_digest": request_digest,
                                           "base": base, "salt": salt})[:AUTO_ID_DIGITS]
        if ws.operation(op_id) is None:
            return op_id
        salt += 1


def _replayed(ws: OpticalWorkspaceV1, summary: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(summary)
    if result.get("truncated") and result.get("kind") in ("import", "replace_instance"):
        instance = ws.instance(result.get("instance_key", ""))
        if instance is not None:
            result["id_map"] = [e.model_dump(mode="json") for e in instance.id_map]
            result["fixtures"] = [f.model_dump(mode="json", exclude_none=True) for f in instance.fixtures]
    return result


# ------------------------------------------------------------------------------------------------ import


def _catalog_binding(project: dict[str, Any], mode: str, catalog: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``project`` with every catalog component on one release: unchanged when the pins agree; otherwise (``repin``)
    moved to the active release when every product is published there unchanged, else refused."""

    pins = sorted({c["product"]["catalog_version"] for c in project.get("components", [])
                   if c.get("product") and c["product"].get("catalog_version") is not None})
    if len(pins) <= 1:
        return project, []
    rows = [{"instance_id": c["instance_id"], "optical_sku": c["product"]["optical_sku"], "catalog_version": c["product"].get("catalog_version")}
            for c in project.get("components", []) if c.get("product")]
    if mode == "refuse":
        raise refuse(_E.CATALOG_BINDING_CONFLICT, f"the catalog components would pin different catalog releases ({', '.join(pins)})",
                     "catalog_binding", releases=pins, components=rows)
    from ..local.catalog_update import CURRENT, UPDATED, update_project_catalog

    try:
        update = update_project_catalog(project, catalog)
    except LocalRunError as exc:
        raise refuse(_E.CATALOG_BINDING_CONFLICT, f"the catalog components pin different releases ({', '.join(pins)}) and the active release "
                     f"could not be used to align them: {'; '.join(p.message for p in exc.problems)}", "catalog_binding",
                     releases=pins, components=rows) from None
    if update.status not in (CURRENT, UPDATED):
        raise refuse(_E.CATALOG_BINDING_CONFLICT, f"the catalog components pin different releases ({', '.join(pins)}); moving them to "
                     f"{update.catalog_version} would change products: {update.summary}", "catalog_binding", releases=pins, components=rows,
                     active_release=update.catalog_version, changed=list(update.changed), withdrawn=list(update.withdrawn),
                     unverified=list(update.unverified))
    return update.project.payload(), list(update.repinned)


def _existing_instance(ws: OpticalWorkspaceV1, module_ws: OpticalWorkspaceV1, request: dict[str, Any]) -> None:
    key = request.get("instance_key")
    instance = ws.instance(key) if isinstance(key, str) else None
    if instance is None:
        return
    placement = request.get("placement")
    same_place = placement is None and instance.placement.model_dump(mode="json") == {"position_mm": [0.0, 0.0, 0.0],
                                                                                         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    if placement is not None and isinstance(placement, Mapping):
        try:
            same_place = pose_dict(*_pose_value(placement, "placement")) == instance.placement.model_dump(mode="json")
        except CompositionError:
            same_place = False
    if instance.source.module_digest == module_ws.module_digest() and same_place:
        raise _Unchanged({"instance_key": key, "module_id": instance.source.module_id, "module_digest": instance.source.module_digest,
                          "placement": instance.placement.model_dump(mode="json"), "already_imported": True,
                          "id_map": [e.model_dump(mode="json") for e in instance.id_map],
                          "fixtures": [f.model_dump(mode="json", exclude_none=True) for f in instance.fixtures]})
    raise refuse(_E.INSTANCE_KEY_EXISTS, f"instance {key!r} already holds {'another module' if instance.source.module_digest != module_ws.module_digest() else 'this module at another placement'}; "
                 "a second instance needs a new instance_key", "instance_key", instance_key=key,
                 existing_module_digest=instance.source.module_digest)


def _deleted_ids(ws: OpticalWorkspaceV1, released: Callable[[str, Any], bool] | None = None) -> dict[str, dict[str, str]]:
    """kind -> {id: instance key} of the members instances record as deleted (``released(instance key, id map entry)``:
    left out). An import never takes such an id: an object with it is that instance's member again."""

    reserved: dict[str, dict[str, str]] = {}
    for instance in ws.instances:
        for entry in ws.deleted_members(instance.instance_key):
            if released is None or not released(instance.instance_key, entry):
                reserved.setdefault(entry.kind, {}).setdefault(entry.parent_id, instance.instance_key)
    return reserved


def _do_import(ws: OpticalWorkspaceV1, request: dict[str, Any], module_ws: OpticalWorkspaceV1 | None, catalog: Any,
               *, parent_project: dict[str, Any] | None = None, group_id: str | None = None,
               reserved: dict[str, dict[str, str]] | None = None, adopt: AbstractSet[tuple[str, str]] = frozenset()
               ) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    assert module_ws is not None
    if "instance_key" not in request:
        raise refuse(_E.REQUEST_INVALID, "an import names its instance_key", "instance_key")
    if parent_project is None:
        _existing_instance(ws, module_ws, request)
    mode = request.get("catalog_binding", "repin")
    if mode not in CATALOG_BINDINGS:
        raise refuse(_E.REQUEST_INVALID, f"catalog_binding is one of {list(CATALOG_BINDINGS)}", "catalog_binding")
    label = request.get("label")
    if label is not None and not isinstance(label, str):
        raise refuse(_E.REQUEST_INVALID, "label is a string", "label")
    gid = request.get("group_id", group_id)
    if gid is not None and (not isinstance(gid, str) or ws.group(gid) is not None and parent_project is None):
        raise refuse(_E.GROUP_EXISTS if isinstance(gid, str) else _E.REQUEST_INVALID, f"the workspace already has a group {gid!r}; give another "
                     "group_id", "group_id")
    if gid is None and parent_project is None and ws.group(request["instance_key"]) is not None:
        raise refuse(_E.GROUP_EXISTS, f"the workspace already has a group {request['instance_key']!r} (the default group id of this "
                     "import); give another group_id", "group_id")
    placement = None
    if request.get("placement") is not None:
        placement = pose_dict(*_pose_value(request["placement"], "placement"))
    policy = parse_fixture_policy(request.get("fixtures"), module_ws)
    outcome = import_module(ws, module_ws, instance_key=request["instance_key"], placement=placement, label=label, fixtures=policy,
                            group_id=gid, parent_project=parent_project, reserved=_deleted_ids(ws) if reserved is None else reserved,
                            adopt=adopt)
    project, repinned = _catalog_binding(outcome.project, mode, catalog)
    member_digests(project, outcome.instances)
    payload = ws.payload()
    if repinned:  # a re-pinned member of an earlier instance is not an edit of it
        for instance in payload.get("instances", []):
            _follow_digests(instance, outcome.project, project)
    payload["project"] = project
    payload["instances"] = [*payload.get("instances", []), *outcome.instances]
    payload["groups"] = [*payload.get("groups", []), *outcome.groups]
    report = {**outcome.report, "catalog": {"repinned": repinned}}
    return payload, report, outcome.warnings


# ------------------------------------------------------------------------------------------------ transform


def _frame(ws: OpticalWorkspaceV1, value: Any) -> tuple[dict[str, Any], tuple[rot.Vector, rot.Quaternion]]:
    if value is None or value == "design":
        return {"design": True}, ((0.0, 0.0, 0.0), rot.IDENTITY)
    if isinstance(value, Mapping) and len(value) == 1:
        (name, ref), = value.items()
        if name == "group":
            group = ws.group(ref) if isinstance(ref, str) else None
            if group is None:
                raise refuse(_E.FRAME_INVALID, f"the workspace has no group {ref!r}", "frame")
            if group.frame is None:
                raise refuse(_E.FRAME_INVALID, f"group {ref!r} has no reference frame; give it one (group_create frame) or use another frame", "frame")
            return {"group": ref}, pose_tuple(group.frame.model_dump(mode="json"))
        if name == "component":
            component = next((c for c in ws.project.components if c.instance_id == ref), None)
            if component is None:
                raise refuse(_E.FRAME_INVALID, f"the Project has no component {ref!r}", "frame")
            return {"component": ref}, pose_tuple(component.nominal_pose.model_dump(mode="json"))
        if name == "instance":
            instance = ws.instance(ref) if isinstance(ref, str) else None
            if instance is None:
                raise refuse(_E.FRAME_INVALID, f"the workspace has no instance {ref!r}", "frame")
            return {"instance": ref}, pose_tuple(instance.placement.model_dump(mode="json"))
    raise refuse(_E.FRAME_INVALID, "frame is 'design', {group: id}, {component: id} or {instance: key}", "frame")


def _rotation(value: Any) -> rot.Quaternion | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise refuse(_E.ROTATION_INVALID, "rotation is {quaternion_wxyz} or {axis, angle_deg}", "rotation")
    if set(value) == {"quaternion_wxyz"}:
        q = value["quaternion_wxyz"]
        if not isinstance(q, (list, tuple)) or len(q) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in q):
            raise refuse(_E.ROTATION_INVALID, "rotation.quaternion_wxyz is a list of four numbers", "rotation")
        try:
            return rot.normalize_quaternion(q)
        except ValueError:
            raise refuse(_E.ROTATION_INVALID, "rotation.quaternion_wxyz is a non-zero quaternion (its length does not matter)",
                         "rotation") from None
    if set(value) == {"axis", "angle_deg"}:
        axis = _vector(value["axis"], "rotation.axis", _E.ROTATION_INVALID)
        angle = value["angle_deg"]
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or angle != angle or abs(angle) == float("inf"):
            raise refuse(_E.ROTATION_INVALID, "rotation.angle_deg is a finite number", "rotation")
        if axis == (0.0, 0.0, 0.0):
            raise refuse(_E.ROTATION_INVALID, "rotation.axis must not be the zero vector", "rotation")
        return rot.axis_angle_quaternion(axis, float(angle))
    raise refuse(_E.ROTATION_INVALID, "rotation is {quaternion_wxyz} or {axis, angle_deg}", "rotation")


class _Rigid:
    """``x -> R(q) (x - pivot) + pivot + d`` (a pure translation when ``q`` is None)."""

    def __init__(self, q: rot.Quaternion | None, pivot: rot.Vector, d: rot.Vector) -> None:
        self.q, self.pivot, self.d = q, pivot, d

    def point(self, p: rot.Vector) -> rot.Vector:
        if self.q is None:
            return rot.add(p, self.d)
        return rot.add(rot.add(rot.rotate(self.q, rot.sub(p, self.pivot)), self.pivot), self.d)

    def pose(self, position: rot.Vector, q: rot.Quaternion) -> tuple[rot.Vector, rot.Quaternion]:
        if self.q is None:
            return rot.add(position, self.d), q
        return self.point(position), rot.normalize_quaternion(rot.multiply(self.q, q))


def _do_transform(ws: OpticalWorkspaceV1, request: dict[str, Any], _module: Any, _catalog: Any
                  ) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    if "selector" not in request:
        raise refuse(_E.REQUEST_INVALID, "a transform names its selector", "selector")
    selection = resolve_selection(ws, request["selector"], allow_empty=False)
    translation = None if request.get("translation_mm") is None else _vector(request["translation_mm"], "translation_mm", _E.TRANSLATION_INVALID)
    q_local = _rotation(request.get("rotation"))
    if translation is None and q_local is None:
        raise refuse(_E.REQUEST_INVALID, "a transform gives translation_mm and / or rotation")
    frame_name, (frame_p, frame_q) = _frame(ws, request.get("frame"))
    framed = not rot.is_identity(frame_q)
    q_design = None
    if q_local is not None:
        q_design = rot.normalize_quaternion(rot.multiply(rot.multiply(frame_q, q_local), rot.conjugate(frame_q))) if framed else q_local
    d = (0.0, 0.0, 0.0) if translation is None else (rot.rotate(frame_q, translation) if framed else translation)
    pivot = _pivot(ws, request.get("pivot", "frame_origin"), selection, frame_p, frame_q)
    rigid = _Rigid(q_design, pivot, d)
    project = ws.project
    payload = ws.payload()
    components = {c["instance_id"]: c for c in payload["project"].get("components", [])}
    observations = {o["observation_id"]: o for o in payload["project"].get("observations", [])}
    affected = []
    moved_components = set(selection.components)
    for cid in selection.components:
        before = components[cid]["nominal_pose"]
        position, q = rigid.pose(*pose_tuple(before))
        components[cid]["nominal_pose"] = pose_dict(position, q)
        affected.append({"kind": "COMPONENT", "id": cid, "before": copy.deepcopy(before), "after": components[cid]["nominal_pose"]})
    followers = []
    for oid in selection.observations:
        obs = observations[oid]
        device = obs.get("device_instance_id")
        if device is None:
            before = obs.get("pose", {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
            position, q = rigid.pose(*pose_tuple(before))
            obs["pose"] = pose_dict(position, q)
            affected.append({"kind": "OBSERVATION", "id": oid, "before": copy.deepcopy(before), "after": obs["pose"]})
        elif device in moved_components:
            followers.append({"observation_id": oid, "device_instance_id": device, "selected": True})
        else:
            dev_p, dev_q = pose_tuple(components[device]["nominal_pose"])
            design_before = _object_pose(project, "OBSERVATION", oid)
            world = rigid.pose(*pose_tuple(design_before))
            inv_p, inv_q = rot.inverse_pose(dev_p, dev_q)
            new_local = rot.compose_pose(inv_p, inv_q, *world)
            obs["pose"] = pose_dict(*new_local)
            affected.append({"kind": "OBSERVATION", "id": oid, "before": design_before, "after": pose_dict(*world),
                             "attached_to": device, "local_pose": obs["pose"]})
    # a plane attached to a moved device moves with it (it is stated in the device's frame) even when not selected
    selected_observations = set(selection.observations)
    followers += [{"observation_id": oid, "device_instance_id": obs["device_instance_id"], "selected": False}
                  for oid, obs in observations.items()
                  if obs.get("device_instance_id") in moved_components and oid not in selected_observations]
    moved = {("COMPONENT", c) for c in selection.components} | {("OBSERVATION", o) for o in selection.observations}
    moved |= {("OBSERVATION", f["observation_id"]) for f in followers}
    placements, frames = [], []
    members_of = {}
    for instance in payload.get("instances", []):
        members_c, members_o = instance_members(ws, instance["instance_key"])
        members_of[instance["instance_key"]] = {("COMPONENT", c) for c in members_c} | {("OBSERVATION", o) for o in members_o}
    whole = {key for key, members in members_of.items() if members and members <= moved}
    # an instance nested in one that moves as a whole moves with it also when none of its components and observation
    # planes is left in the Project: its placement stays where the enclosing instance's module puts it
    whole |= {inner for key in sorted(whole) for inner in nested_instances(ws, key) if not members_of.get(inner)}
    for instance in payload.get("instances", []):
        if instance["instance_key"] in whole:
            instance["placement"] = pose_dict(*rigid.pose(*pose_tuple(instance["placement"])))
            placements.append(instance["instance_key"])
            _follow_digests(instance, ws.project.payload(), payload["project"], whole_move=True)
    for group in payload.get("groups", []):
        members = {(m["kind"], m["id"]) for m in group.get("members", [])}
        if group.get("frame") is None:
            continue
        if (members and members <= moved) or (not members and group.get("origin") == "IMPORT" and group.get("instance_key") in whole):
            group["frame"] = pose_dict(*rigid.pose(*pose_tuple(group["frame"])))
            frames.append(group["group_id"])
    summary = {"frame": {**frame_name, "pose": pose_dict(frame_p, frame_q)}, "pivot_mm": list(pivot),
               "rotation_quaternion_wxyz": list(q_design) if q_design is not None else None, "translation_mm": list(d),
               "selection": selection.as_dict(), "affected": affected, "followers": followers,
               "placements_updated": placements, "group_frames_updated": frames}
    return payload, summary, []


def _follow_digests(instance: dict[str, Any], before: dict[str, Any], after: dict[str, Any], *, only_repin: bool = False,
                    whole_move: bool = False) -> None:
    """A whole instance moved as one body (or a catalog component moved to another catalog release unchanged) is not an
    edit: each member that was unedited before gets the digest of its new content. ``only_repin``: follow only members
    whose sole change is ``product.catalog_version``.

    ``whole_move`` (a transform of the whole instance): a component or observation plane that is not the import's
    content -- edited earlier, with content the move changed, or deleted from the Project -- would have its import
    content moved too, which is not known; its digest becomes ``moved_edit_digest`` of it, so it stays an edit (a later move
    can never bring it back onto its old digest, and a deleted member added back is an edit). An edited member the move
    did not change (a plane stated in its device's frame) keeps its digest; a move changes no target or definition. A
    member a nested replacement added (``ADDED_EDIT_DIGEST``) stays added, and one the instance's revision lacks
    (``LACKED_MEMBER_DIGEST``) stays lacked. A nested revision digest (``_follow_recreated``) goes: it names the
    placement the nested instance had before the move."""

    old = {kind: {item[ident]: item for item in before.get(name) or []} for kind, name, ident in OBJECTS}
    new = {kind: {item[ident]: item for item in after.get(name) or []} for kind, name, ident in OBJECTS}
    for (kind, parent_id), member in digest_refs(instance):
        if whole_move:  # the placement a nested revision digest names moved too
            member.pop(_NESTED_REVISION, None)
        if kind not in old or member["digest"] in _SENTINEL_DIGESTS:
            continue
        previous, moved = old[kind].get(parent_id), new[kind].get(parent_id)
        if whole_move and kind in ("COMPONENT", "OBSERVATION") and (
                moved is None or (previous is not None and previous != moved and osc1_digest(previous) != member["digest"])):
            member["digest"] = moved_edit_digest(member["digest"])
            continue
        if previous is None or moved is None or osc1_digest(previous) != member["digest"]:
            continue
        if only_repin and _without_pin(previous) != _without_pin(moved):
            continue
        member["digest"] = osc1_digest(moved)


def _pivot(ws: OpticalWorkspaceV1, value: Any, selection: Selection, frame_p: rot.Vector, frame_q: rot.Quaternion) -> rot.Vector:
    if value is None or value == "frame_origin":
        return frame_p
    if value == "selection_centroid":
        points = [tuple(c.nominal_pose.position_mm) for c in ws.project.components if c.instance_id in selection.components]
        attached_followers = {oid for oid, _, follows in selection.attached if follows}
        points += [tuple(ws.project.observation_design_pose(o).position_mm) for o in ws.project.observations
                   if o.observation_id in selection.observations and o.observation_id not in attached_followers]
        n = float(len(points))
        return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, sum(p[2] for p in points) / n)
    if isinstance(value, Mapping) and set(value) == {"point_mm"}:
        point = _vector(value["point_mm"], "pivot.point_mm", _E.PIVOT_INVALID)
        return rot.add(rot.rotate(frame_q, point), frame_p) if not rot.is_identity(frame_q) else rot.add(point, frame_p)
    if isinstance(value, Mapping) and set(value) == {"component"}:
        component = next((c for c in ws.project.components if c.instance_id == value["component"]), None)
        if component is None:
            raise refuse(_E.PIVOT_INVALID, f"the Project has no component {value['component']!r}", "pivot")
        return tuple(component.nominal_pose.position_mm)  # type: ignore[return-value]
    raise refuse(_E.PIVOT_INVALID, "pivot is 'frame_origin', 'selection_centroid', {point_mm: [x, y, z]} (in the frame) or {component: id}", "pivot")


# ------------------------------------------------------------------------------------------------ groups


def _group_frame(ws: OpticalWorkspaceV1, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping) and set(value) <= {"position_mm", "quaternion_wxyz"}:
        return pose_dict(*_pose_value(value, "frame"))
    if isinstance(value, Mapping) and len(value) == 1 and next(iter(value)) in ("component", "instance", "group"):
        _, pose = _frame(ws, value)
        return pose_dict(*pose)
    raise refuse(_E.FRAME_INVALID, "a group frame is a pose {position_mm, quaternion_wxyz}, {component: id}, {instance: key} or {group: id}", "frame")


def _group_id(request: dict[str, Any]) -> str:
    gid = request.get("group_id")
    if not isinstance(gid, str) or not gid:
        raise refuse(_E.REQUEST_INVALID, "group_id is a non-empty string", "group_id")
    return gid


def _members(ws: OpticalWorkspaceV1, request: dict[str, Any]) -> list[dict[str, str]]:
    if "members" not in request:
        raise refuse(_E.REQUEST_INVALID, "members is a selector {ids, groups, instances, kinds}", "members")
    return resolve_selection(ws, request["members"]).refs()


def _do_group_create(ws: OpticalWorkspaceV1, request: dict[str, Any], _m: Any, _c: Any) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    gid = _group_id(request)
    if ws.group(gid) is not None:
        raise refuse(_E.GROUP_EXISTS, f"the workspace already has a group {gid!r}", "group_id", group_id=gid)
    group: dict[str, Any] = {"group_id": gid, "members": _members(ws, request) if "members" in request else [], "origin": "USER"}
    if request.get("label") is not None:
        group["label"] = request["label"]
    frame = _group_frame(ws, request.get("frame"))
    if frame is not None:
        group["frame"] = frame
    payload = ws.payload()
    payload["groups"] = [*payload.get("groups", []), group]
    return payload, {"group": group}, []


def _existing_group(ws: OpticalWorkspaceV1, request: dict[str, Any]) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    gid = _group_id(request)
    payload = ws.payload()
    for index, group in enumerate(payload.get("groups", [])):
        if group["group_id"] == gid:
            return gid, index, group, payload
    raise refuse(_E.GROUP_UNKNOWN, f"the workspace has no group {gid!r}; groups: {sorted(g.group_id for g in ws.groups)}", "group_id", group_id=gid)


def _do_group_dissolve(ws: OpticalWorkspaceV1, request: dict[str, Any], _m: Any, _c: Any) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    _gid, index, group, payload = _existing_group(ws, request)
    del payload["groups"][index]
    return payload, {"group": group}, []


def _membership(kind: str) -> Callable[..., tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]]:
    def mutate(ws: OpticalWorkspaceV1, request: dict[str, Any], _m: Any, _c: Any) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
        _gid, index, group, payload = _existing_group(ws, request)
        current = [(m["kind"], m["id"]) for m in group.get("members", [])]
        given = [(m["kind"], m["id"]) for m in _members(ws, request)]
        if kind == "set":
            new = given
        elif kind == "add":
            new = current + [m for m in given if m not in current]
        else:
            new = [m for m in current if m not in given]
        group["members"] = [{"kind": k, "id": i} for k, i in new]
        payload["groups"][index] = group
        added = sorted({m for m in new if m not in current})
        removed = sorted({m for m in current if m not in new})
        return payload, {"group": group, "added": [{"kind": k, "id": i} for k, i in added],
                         "removed": [{"kind": k, "id": i} for k, i in removed]}, []
    return mutate


# ------------------------------------------------------------------------------------------------ update_project


def _project_ids(project: Mapping[str, Any]) -> dict[str, set[str]]:
    return {kind: {item[ident] for item in project.get(name) or []} for kind, name, ident in (*OBJECTS, *LIBRARY)}


def _copied(instance: Mapping[str, Any]) -> set[tuple[str, str]]:
    """``(kind, parent_id)`` of the members an instance record copied (it holds their digests)."""

    own = {(e["kind"], e["parent_id"]) for e in instance.get("id_map", []) if not e.get("shared")}
    return {ref for ref, _ in digest_refs(instance) if ref in own}


def prune_to_project(payload: dict[str, Any], before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Drop from the organization (groups, instance records, module descriptor) every reference to an object the
    Project no longer has, and report what was dropped. A member an instance copied stays in its record when it is
    deleted (id map entry, digest, fixture outcome): the deletion is an edit of the instance (``edited_members``).
    ``removed_instance_members`` lists the instance members the Project lost: with ``before`` (the Project before the
    change) only those this change deleted, and not one the instance's revision lacks (``LACKED_MEMBER_DIGEST``: no
    edit of it)."""

    project = payload["project"]
    ids = _project_ids(project)
    previous = _project_ids(before) if before is not None else None
    dropped_members, dropped_instance = [], []
    for group in payload.get("groups", []):
        keep = [m for m in group.get("members", []) if m["id"] in ids[m["kind"]]]
        dropped_members += [{"group_id": group["group_id"], **m} for m in group.get("members", []) if m not in keep]
        group["members"] = keep
    for instance in payload.get("instances", []):
        removed_fixtures = {(f["object_kind"], f["source_id"]) for f in instance.get("fixtures", []) if f["action"] != "RETAINED"}
        copied = _copied(instance)
        digests = digest_refs(instance)  # read with the id map they refer to
        lacked = {ref for ref, member in digests if member["digest"] == LACKED_MEMBER_DIGEST}
        keep = []
        for entry in instance.get("id_map", []):
            kind, parent = entry["kind"], entry["parent_id"]
            if parent in ids[kind] or (kind, entry["source_id"]) in removed_fixtures:
                keep.append(entry)
                continue
            if (kind, parent) in copied:  # deleted from the Project: an edit the instance keeps
                keep.append(entry)
                if (previous is not None and parent not in previous[kind]) or (kind, parent) in lacked:
                    continue  # deleted earlier, reported then; or one the instance's revision lacks
            dropped_instance.append({"instance_key": instance["instance_key"], **entry})
        instance["id_map"] = keep
        mapped = {(e["kind"], e["parent_id"]) for e in keep}
        instance["member_digests"] = [d for ref, d in digests if ref in mapped]
        for outcome in instance.get("fixtures", []):
            kind = outcome["object_kind"]
            for name in ("parent_id", "rebound_to"):
                value = outcome.get(name)
                if value is not None and value not in ids[kind] and (kind, value) not in copied:
                    outcome.pop("parent_id", None)
                    outcome.pop("rebound_to", None)
                    outcome["action"] = "REMOVED"
            outcome["affected_targets"] = [t for t in outcome.get("affected_targets", []) if t in ids["TARGET"] or ("TARGET", t) in copied]
    module = payload.get("module")
    if module is not None:
        module["fixtures"] = [f for f in module.get("fixtures", []) if f["object_id"] in ids[f["object_kind"]]]
        module["requirements"] = [r for r in module.get("requirements", []) if r["target_id"] in ids["TARGET"]]
        sources = {c["instance_id"] for c in project.get("components", []) if c.get("optical_role") == "source"}
        for port in module.get("interfaces", []):
            if port.get("beam_reference") is not None and port["beam_reference"] not in sources:
                port.pop("beam_reference")
    return {"removed_group_members": dropped_members, "removed_instance_members": dropped_instance}


def restore_returned_members(payload: dict[str, Any], before: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The deleted members of instances that are in the Project again (an object of the same kind and id, absent from
    ``before``): their records never left, so each is the instance's member again (edited or not by its digest); a
    component or observation plane also rejoins its instance's import group, at its import position. A member the
    instance's revision lacks (``LACKED_MEMBER_DIGEST``) is its member too (an added one) but is not reported."""

    ids, previous = _project_ids(payload["project"]), _project_ids(before)
    import_groups = {g["instance_key"]: g for g in payload.get("groups", []) if g.get("origin") == "IMPORT"}
    restored = []
    for instance in payload.get("instances", []):
        copied = _copied(instance)
        lacked = {ref for ref, member in digest_refs(instance) if member["digest"] == LACKED_MEMBER_DIGEST}
        order = {(e["kind"], e["parent_id"]): n for n, e in enumerate(instance.get("id_map", []))}
        for entry in instance.get("id_map", []):
            ref = (entry["kind"], entry["parent_id"])
            if ref not in copied or entry["parent_id"] not in ids[entry["kind"]] or entry["parent_id"] in previous[entry["kind"]]:
                continue
            if ref not in lacked:
                restored.append({"instance_key": instance["instance_key"], **entry})
            group = import_groups.get(instance["instance_key"])
            if group is None or entry["kind"] not in ("COMPONENT", "OBSERVATION"):
                continue
            members = group.setdefault("members", [])
            if {"kind": entry["kind"], "id": entry["parent_id"]} not in members:
                at = next((n for n, m in enumerate(members) if order.get((m["kind"], m["id"]), -1) > order[ref]), len(members))
                members.insert(at, {"kind": entry["kind"], "id": entry["parent_id"]})
    return restored


def _do_update_project(ws: OpticalWorkspaceV1, request: dict[str, Any], _m: Any, _c: Any) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    if "project" not in request:
        raise refuse(_E.REQUEST_INVALID, "update_project gives the new project", "project")
    from .document import _validated_workspace

    project = _validated_workspace(request["project"], "project", _E.REQUEST_INVALID).project
    before = ws.project.payload()
    payload = ws.payload()
    payload["project"] = project.payload()
    for instance in payload.get("instances", []):  # a catalog re-pin of an unedited member is not an edit of it
        _follow_digests(instance, before, payload["project"], only_repin=True)
    report = prune_to_project(payload, before)
    restored = restore_returned_members(payload, before)
    warnings = []
    if report["removed_instance_members"]:
        warnings.append(warning(CompositionWarningCode.INSTANCE_MEMBERS_REMOVED, "objects of imported instances were deleted from the Project: "
                                + ", ".join(sorted({f"{e['parent_id']} ({e['instance_key']})" for e in report["removed_instance_members"]}))
                                + "; their instances record the deletions as edits"))
    return payload, {"project_hash": project.content_hash(), **report, "restored_instance_members": restored}, warnings


# ------------------------------------------------------------------------------------------------ replace_instance


def _record_edits(instance: Any, index: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    """The edits one instance record shows (``member_digests``): ``EDITED``, ``REMOVED``, ``ADDED`` (a member a nested
    replacement added; one the instance's revision lacks that the Project has)."""

    edited = []
    for member in instance.member_digests:
        kind = instance.member_kind(member)
        item = index[kind].get(member.parent_id)
        if member.digest == LACKED_MEMBER_DIGEST:
            if item is None:
                continue  # the revision lacks it and so does the Project: as imported
            change = "ADDED"
        elif item is None:
            change = "REMOVED"
        elif member.digest == ADDED_EDIT_DIGEST:
            change = "ADDED"
        elif osc1_digest(item) != member.digest:
            change = "EDITED"
        else:
            continue
        edited.append({"kind": kind, "id": member.parent_id, "change": change})
    return edited


def edited_members(ws: OpticalWorkspaceV1, instance_key: str) -> list[dict[str, str]]:
    """The local edits of an instance -- exactly what replacing it would discard -- as ``{kind, id, change}``, each member
    once: members whose content differs from the import's (``EDITED``), members deleted from the Project since
    (``REMOVED``), and members added (``ADDED``: gained by a nested instance replaced on its own, or objects in the
    Project that a nested instance records as lacked by its revision), of the instance and of every instance nested in
    it (``nested_instances``), the instance's own record first. A definition of these instances that objects outside
    them use is not listed: a replacement keeps it as it is (``kept_definitions``). Nor is a definition a nested
    instance's revision lacks that the nested instance keeps for objects outside it, when the instance (or one between)
    holds it with its import content: replacing the nested instance by a revision that changes nothing leaves the
    enclosing instances unedited."""

    instance = ws.instance(instance_key)
    if instance is None:
        return []
    payload = ws.project.payload()
    index = {kind: {item[ident]: item for item in payload.get(name) or []} for kind, name, ident in (*OBJECTS, *LIBRARY)}
    rows = _record_edits(instance, index)
    nested = nested_instances(ws, instance_key)
    recorded = {(e.kind, e.parent_id) for e in instance.id_map}
    for inner in ws.instances:
        if inner.instance_key not in nested:
            continue
        between = [instance, *(i for i in ws.instances if i.instance_key in nested and inner.instance_key in nested_instances(ws, i.instance_key))]
        imported = _kept_as_imported(ws, inner, between, index)
        rows += [row for row in _record_edits(inner, index) if (row["kind"], row["id"]) not in imported]
        # a member of the nested instance the enclosing record does not hold: added by a replacement of the nested
        # instance made by an earlier build (this one records it in the enclosing instance)
        copied = inner.copied_members()
        rows += [{"kind": e.kind, "id": e.parent_id, "change": "ADDED"} for e in inner.id_map
                 if (e.kind, e.parent_id) in copied and (e.kind, e.parent_id) not in recorded and e.parent_id in index[e.kind]]
    kept = _kept_refs(ws, instance_key)[1]
    edited, seen = [], set()
    for row in rows:
        if (row["kind"], row["id"]) not in seen and (row["kind"], row["id"]) not in kept:
            seen.add((row["kind"], row["id"]))
            edited.append(row)
    return edited


def _kept_as_imported(ws: OpticalWorkspaceV1, inner: Any, enclosing: list[Any], index: Mapping[str, Mapping[str, Any]]) -> set[tuple[str, str]]:
    """The definitions a nested instance ``inner`` holds as lacked by its revision (``LACKED_MEMBER_DIGEST``) and keeps
    because objects outside it use them (``kept_definitions``) that one of the instances ``enclosing`` it (the instance
    whose edits are listed and those between) holds with its present content: as that instance imported it, so the
    nested record's row is no edit of the instances enclosing it (their records decide, as for any member they hold)."""

    lacked = {ref for ref in ((inner.member_kind(m), m.parent_id) for m in inner.member_digests if m.digest == LACKED_MEMBER_DIGEST)
              if ref[0] in _DEFINITION_KINDS and ref[1] in index[ref[0]]}
    if not lacked:
        return set()
    lacked &= _kept_refs(ws, inner.instance_key)[1]
    return {ref for record in enclosing for ref, digest in record.member_digest_map().items()
            if ref in lacked and digest == osc1_digest(index[ref[0]][ref[1]])}


def _scope_ids(ws: OpticalWorkspaceV1, instance_key: str) -> set[tuple[str, str]]:
    """``(kind, id)`` of every member the instance and the instances nested in it hold (not a definition another
    instance owns): what replacing it removes from the Project."""

    keys = {instance_key, *nested_instances(ws, instance_key)}
    return {(e.kind, e.parent_id) for i in ws.instances if i.instance_key in keys for e in i.id_map if not e.shared}


def _remove_scope(project: dict[str, Any], old_ids: AbstractSet[tuple[str, str]]) -> set[tuple[str, str]]:
    """Remove from the Project payload ``project`` (its lists are replaced, not changed) the components, observation
    planes and targets ``old_ids`` names and the definitions among them that nothing left uses; returns the definitions
    among them that objects left in the Project use (they stay)."""

    for kind, name, ident in OBJECTS:
        project[name] = [item for item in project.get(name) or [] if (kind, item[ident]) not in old_ids]
    used = _referenced_library(project)
    for kind, name, ident in LIBRARY:
        items = [item for item in project.get(name) or [] if (kind, item[ident]) not in old_ids or item[ident] in used[kind]]
        if items:
            project[name] = items
        else:
            project.pop(name, None)
    # material / coating ids still used by kept models are recomputed after models are pruned
    used = _referenced_library(project)
    for kind, name, ident in LIBRARY[1:]:
        items = [item for item in project.get(name) or [] if (kind, item[ident]) not in old_ids or item[ident] in used[kind]]
        if items:
            project[name] = items
        else:
            project.pop(name, None)
    return {(kind, item[ident]) for kind, name, ident in LIBRARY for item in project.get(name) or [] if (kind, item[ident]) in old_ids}


def kept_definitions(ws: OpticalWorkspaceV1, instance_key: str) -> list[dict[str, Any]]:
    """The definitions (optical models, materials, coatings) of an instance and of the instances nested in it that
    objects outside them use -- a component of the system or of another instance, a definition of the system's own or
    of another instance --, each with those objects (``used_by``: ``{kind, id}``, also the ones that use it through
    another of these definitions). Replacing the instance never removes or changes such a definition: it stays with its
    current content, so it is never an edit the replacement discards, and the instance keeps holding it when its new
    revision lacks it. A new revision that defines the same id differently is refused (``ID_CONFLICT``, ``used_by``):
    one id names one definition."""

    project, kept = _kept_refs(ws, instance_key)
    return _kept_rows(project, kept)


def _kept_rows(project: Mapping[str, Any], kept: AbstractSet[tuple[str, str]]) -> list[dict[str, Any]]:
    """``{kind, id, used_by}`` of each kept definition (``kept``) of the Project payload ``project`` (without the
    replaced instances' components, observation planes and targets), in the Project's order."""

    return [{"kind": kind, "id": item[ident], "used_by": _outside_users(project, kind, item[ident], kept)}
            for kind, name, ident in LIBRARY for item in project.get(name) or [] if (kind, item[ident]) in kept]


def _outside_users(project: Mapping[str, Any], kind: str, definition_id: str, kept: AbstractSet[tuple[str, str]]) -> list[dict[str, str]]:
    """The objects of the Project payload ``project`` that use a kept definition, directly or through other kept
    definitions (``kept``: the definitions of the replaced instances that stay; their own users are named instead)."""

    out: list[dict[str, str]] = []
    seen = {(kind, definition_id)}
    pending = [(kind, definition_id)]
    while pending:
        current = pending.pop(0)
        for user in _users(project, *current):
            ref = (user["kind"], user["id"])
            if ref in seen:
                continue
            seen.add(ref)
            if ref in kept:
                pending.append(ref)
            else:
                out.append(user)
    return out


def _kept_refs(ws: OpticalWorkspaceV1, instance_key: str) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """The Project payload without the instance's scope (``_remove_scope``) and the definitions of the scope it keeps."""

    scope = _scope_ids(ws, instance_key)
    project = ws.project.payload()
    if not any(kind in ("MODEL", "MATERIAL", "COATING") for kind, _ in scope):
        return project, set()
    return project, _remove_scope(project, scope)


def _users(project: Mapping[str, Any], kind: str, definition_id: str) -> list[dict[str, str]]:
    """The objects of the Project payload that use a definition: components and compound models for an optical model,
    models for a material or a coating (``{kind, id}``)."""

    from .document import _referenced

    models = project.get("optical_models") or []
    if kind == "MODEL":
        users = [{"kind": "COMPONENT", "id": c["instance_id"]} for c in project.get("components") or [] if c.get("optical_model_id") == definition_id]
        return users + [{"kind": "MODEL", "id": m["model_id"]} for m in models
                        if any(x.get("model_id") == definition_id for x in (m.get("model") or {}).get("members") or [])]
    field = "material" if kind == "MATERIAL" else "coating"
    return [{"kind": "MODEL", "id": m["model_id"]} for m in models if definition_id in _referenced(m.get("model"), field)]


def _edit_counts(edited: list[dict[str, str]]) -> str:
    """``"1 member changed, 2 members added and 1 member deleted"``."""

    counts = [f"{n} member{'s' if n != 1 else ''} {what}" for n, what in (
        (sum(1 for e in edited if e["change"] == "EDITED"), "changed"), (sum(1 for e in edited if e["change"] == "ADDED"), "added"),
        (sum(1 for e in edited if e["change"] == "REMOVED"), "deleted")) if n]
    return counts[0] if len(counts) == 1 else ", ".join(counts[:-1]) + " and " + counts[-1]


def _referenced_library(project: dict[str, Any]) -> dict[str, set[str]]:
    from .document import _referenced

    models = {m["model_id"]: m for m in project.get("optical_models") or []}
    used_models: set[str] = set()
    pending = [c.get("optical_model_id") for c in project.get("components", []) if c.get("optical_model_id")]
    while pending:
        model_id = pending.pop()
        if model_id in used_models or model_id not in models:
            continue
        used_models.add(model_id)
        pending += [m.get("model_id") for m in (models[model_id].get("model") or {}).get("members") or []]
    materials: set[str] = set()
    coatings: set[str] = set()
    for model in models.values():
        materials |= _referenced(model.get("model"), "material")
        coatings |= _referenced(model.get("model"), "coating")
    return {"MODEL": used_models, "MATERIAL": materials, "COATING": coatings}


def _do_replace_instance(ws: OpticalWorkspaceV1, request: dict[str, Any], module_ws: OpticalWorkspaceV1 | None, catalog: Any
                         ) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    key = request.get("instance_key")
    instance = ws.instance(key) if isinstance(key, str) else None
    if instance is None:
        raise refuse(_E.INSTANCE_UNKNOWN, f"the workspace has no instance {key!r}", "instance_key", instance_key=key)
    edited = edited_members(ws, key)
    discard = request.get("discard_edits", False)
    if not isinstance(discard, bool):
        raise refuse(_E.REQUEST_INVALID, "discard_edits is true or false", "discard_edits")
    if edited and not discard:
        raise refuse(_E.INSTANCE_EDITED, f"instance {key!r} was edited after its import ({_edit_counts(edited)}); to drop those "
                     "edits, replace it with discard_edits set to true (on the command line: --discard-edits)", "instance_key",
                     edited=edited)
    nested_keys = nested_instances(ws, key)
    # the members of the instance and of every instance nested in it (also one its enclosing record does not hold: a
    # member a nested replacement made by an earlier build added) leave with it
    scope = [instance, *(i for i in ws.instances if i.instance_key in nested_keys)]
    old_ids = {(e.kind, e.parent_id) for i in scope for e in i.id_map if not e.shared}
    # the instances enclosing this one, innermost first (each encloses the instances the previous one encloses)
    ancestors = sorted((i.instance_key for i in ws.instances if i.instance_key != key and key in nested_instances(ws, i.instance_key)),
                       key=lambda k: len(nested_instances(ws, k)))
    project = ws.project.payload()
    # a list in which every member of the instance was deleted (also one its revision lacks, which an enclosing instance
    # records as deleted): its new members go before the objects of the instances that follow this one, where importing
    # them in this order put them
    deleted_kinds = {e.kind for i in scope for e in ws.deleted_members(i.instance_key)}
    deleted_kinds |= {e.kind for k in ancestors for e in ws.deleted_members(k) if (e.kind, e.parent_id) in old_ids}
    keys = [i.instance_key for i in ws.instances]
    later = {(e.kind, e.parent_id) for i in ws.instances[keys.index(key) + 1:] if i.instance_key not in nested_keys
             for e in i.id_map if not e.shared}
    first_index: dict[str, int] = {}
    for kind, name, ident in (*OBJECTS, *LIBRARY):
        items = project.get(name) or []
        positions = [n for n, item in enumerate(items) if (kind, item[ident]) in old_ids]
        if not positions and kind in deleted_kinds:
            positions = [n for n, item in enumerate(items) if (kind, item[ident]) in later]
        if positions:
            first_index[name] = positions[0]
    # definitions of the instance that objects outside it use stay, with their content (kept_definitions)
    kept_in_use = _remove_scope(project, old_ids)
    kept_definitions_ = _kept_rows(project, kept_in_use)
    before_counts = {name: len(project.get(name) or []) for _, name, _ in (*OBJECTS, *LIBRARY)}
    old_group = next((g for g in ws.groups if g.origin == "IMPORT" and g.instance_key == key), None)
    sub = {"instance_key": key, "placement": instance.placement.model_dump(mode="json"), "fixtures": request.get("fixtures"),
           "catalog_binding": request.get("catalog_binding", "repin"), "label": request.get("label", instance.source.label)}
    sub = {k: v for k, v in sub.items() if v is not None}
    shell = ws.payload()
    shell["instances"] = [i for i in shell.get("instances", []) if i["instance_key"] != key and i["instance_key"] not in nested_keys]
    shell["groups"] = []
    shell_ws = validate_payload(shell)
    # the ids of members other instances record as deleted stay theirs, except the members of this instance's part of
    # the instances enclosing it: those this instance or an instance nested in it holds (deleted ones, and ones its
    # revision lacks), which the new revision may bring back to the enclosing instances
    reserved = _deleted_ids(shell_ws, lambda owner, entry: owner in ancestors and (entry.kind, entry.parent_id) in old_ids)
    try:
        payload, report, warnings = _do_import(shell_ws, sub, module_ws, catalog, parent_project=project,
                                               group_id=old_group.group_id if old_group is not None else None, reserved=reserved,
                                               adopt=kept_in_use)
    except CompositionError as exc:
        raise _restated(exc, ws, key, kept_in_use, project) from None
    new_project = payload["project"]
    new_keys = {key, *report["nested_instances"]}
    # a kept definition the new revision has unchanged was copied over, among the revision's definitions
    adopted = kept_in_use & {(e["kind"], e["parent_id"]) for i in payload["instances"] if i["instance_key"] in new_keys for e in i["id_map"]}
    # put the new members where the old ones were (author order is part of the Engine input)
    for list_kind, name, _ in (*OBJECTS, *LIBRARY):
        if name in first_index and new_project.get(name):
            items = new_project[name]
            count = before_counts[name] - sum(1 for kind, _ in adopted if kind == list_kind)
            kept, added = items[:count], items[count:]
            at = min(first_index[name], len(kept))
            new_project[name] = kept[:at] + added + kept[at:]
    _keep_reproduced(ws, instance, scope, payload, new_keys, edited,
                     same_revision=module_ws is not None and instance.source.module_digest == module_ws.module_digest())
    member_digests(new_project, [i for i in payload["instances"] if i["instance_key"] in new_keys])
    new_instance = next(i for i in payload["instances"] if i["instance_key"] == key)
    if instance.parent_instance is not None:  # a nested instance replaced on its own keeps its provenance
        new_instance["parent_instance"] = instance.parent_instance
    # group memberships follow the source ids
    old_sources = {(e.kind, e.parent_id): e.source_id for e in instance.id_map if not e.shared}
    new_by_source = {(e["kind"], e["source_id"]): e["parent_id"] for e in new_instance["id_map"]}
    new_to_source = {(e["kind"], e["parent_id"]): e["source_id"] for e in new_instance["id_map"]}
    # a member the old record holds from its import and the group lost (deleted) rejoins it at its import position
    # (restore_returned_members); a member recorded as added by a nested replacement had no import position
    added_refs = {ref for ref, member in digest_refs(instance.model_dump(mode="json")) if member["digest"] == ADDED_EDIT_DIGEST}
    known_sources = {(kind, source) for (kind, parent), source in old_sources.items() if (kind, parent) not in added_refs}
    new_groups = payload["groups"]
    main_new = next(g for g in new_groups if g.get("instance_key") == key)
    nested_new = [g for g in new_groups if g is not main_new]
    replaced_ids = {g["group_id"] for g in nested_new}
    dropped, groups = [], []
    for group in ws.payload().get("groups", []):
        if group.get("instance_key") in nested_keys or group["group_id"] in replaced_ids:
            continue
        mapped = []
        for member in group.get("members", []):
            ref = (member["kind"], member["id"])
            if ref in old_sources:
                new_id = new_by_source.get((member["kind"], old_sources[ref]))
                if new_id is None:
                    dropped.append({"group_id": group["group_id"], **member})
                    continue
                mapped.append({"kind": member["kind"], "id": new_id})
            else:
                mapped.append(member)
        if old_group is not None and group["group_id"] == old_group.group_id:
            present = {(m["kind"], m["id"]) for m in mapped}
            mapped += [m for m in main_new["members"] if (m["kind"], m["id"]) not in present
                       and (m["kind"], new_to_source.get((m["kind"], m["id"]))) not in known_sources]
        groups.append({**group, "members": mapped})
    payload["groups"] = groups + nested_new
    # the instance keeps its place in the list
    old_index = [i.instance_key for i in ws.instances].index(key)
    others = [i for i in payload["instances"] if i["instance_key"] not in new_keys]
    payload["instances"] = others[:old_index] + [i for i in payload["instances"] if i["instance_key"] in new_keys] + others[old_index:]
    # members deleted before the replacement are back: they rejoin the import groups of this instance (and of an
    # enclosing one); an enclosing instance records members the new revision does not have as deleted
    before = ws.project.payload()
    pruned = prune_to_project(payload, before)
    restore_returned_members(payload, before)
    _record_lacked(payload, key, ancestors, scope, new_keys, kept_in_use)
    forgotten = _record_in_enclosing(payload, ws, ancestors, old_ids, nested_keys, new_keys)
    _follow_recreated(payload, instance, ancestors)
    for record in payload["instances"]:  # a catalog re-pin is no edit of the enclosing instances either
        if record["instance_key"] in ancestors:
            _follow_digests(record, before, payload["project"], only_repin=True)
    removed = [e for e in pruned["removed_instance_members"] if (e["instance_key"], e["kind"], e["parent_id"]) not in forgotten]
    report = {**report, "replaced": {"previous_module_digest": instance.source.module_digest, "edits_discarded": edited if discard else [],
                                     "dropped_group_members": dropped + pruned["removed_group_members"],
                                     "removed_nested_instances": sorted(nested_keys),
                                     "removed_instance_members": removed, "kept_definitions": kept_definitions_}}
    return payload, report, warnings


def _same_but_pose(kind: str, old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """Whether two versions of a component or observation plane are equal apart from the pose a placement gives it and
    the catalog release pin (``product.catalog_version``, which the catalog binding sets for the whole Project)."""

    pose = _POSE_FIELD[kind]
    return (pose in old) == (pose in new) and (_without_pin({k: v for k, v in old.items() if k != pose})
                                               == _without_pin({k: v for k, v in new.items() if k != pose}))


def _kept_content(previous: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """``previous`` (a member keeping its exact content, ``_same_but_pose``) with the catalog release pin of its new
    version ``new``."""

    kept = copy.deepcopy(dict(previous))
    if "product" in new:
        kept["product"] = copy.deepcopy(new["product"])
    return kept


def _keep_reproduced(ws: OpticalWorkspaceV1, instance: Any, scope: list[Any], payload: dict[str, Any], new_keys: AbstractSet[str],
                     edited: list[dict[str, str]], *, same_revision: bool) -> None:
    """A replacement re-creates the members its revision shares with the old one at the same placement. After a
    whole-instance move or rotation a fresh placement of a member differs from its moved content by floating-point
    rounding alone. A component or observation plane keeps its exact content when that is provably the case, by digests
    only (no numeric tolerance):

    * it is no edit of the replaced instance (not in ``edited``) and its content is the one the replaced instance's
      record holds (its digest), so it is the module object that record names placed at the instance's placement and
      moved with the instance since;
    * the new revision's object for it is the same: the old record and the new one name the same source id and the
      same source digest (``MemberDigestV1.source_digest``; a record without one proves nothing);
    * the re-created content equals it apart from its pose and its catalog release pin; the pin is the re-created one's
      (the catalog binding sets one release for the whole Project), so after a whole-instance rotation and a catalog
      re-pin the member differs from the content its enclosing instances record by its pin alone, which they follow.

    Every other member gets the new revision's content: an edited one (also one a sub-rounding move changed), one the
    revision changed however little, one whose source the old record does not name. With the same revision an instance
    nested in the replaced one keeps its placement when all its components and observation planes kept their content.
    So replacing an unedited instance by the revision it has changes nothing, and members a new revision has unchanged
    keep their bytes, also inside the instances that enclose the replaced one."""

    key = instance.instance_key
    record = next(r for r in payload["instances"] if r["instance_key"] == key)
    old_digests, old_sources = instance.member_digest_map(), instance.source_digest_map()
    old_source_ids = {(e.kind, e.parent_id): e.source_id for e in instance.id_map if not e.shared}
    new_sources = {ref: member.get("source_digest") for ref, member in digest_refs(record)}
    listed = {(row["kind"], row["id"]) for row in edited}
    old_project, new_project = ws.project.payload(), payload["project"]
    posed = [(kind, name, ident) for kind, name, ident in OBJECTS if kind in SOURCED]
    old_index = {kind: {item[ident]: item for item in old_project.get(name) or []} for kind, name, ident in posed}
    at = {kind: {item[ident]: (name, n) for n, item in enumerate(new_project.get(name) or [])} for kind, name, ident in posed}
    kept: set[tuple[str, str]] = set()
    for entry in record["id_map"]:
        ref = (entry["kind"], entry["parent_id"])
        if entry["shared"] or ref[0] not in SOURCED or ref in listed or old_source_ids.get(ref) != entry["source_id"]:
            continue
        previous, where = old_index[ref[0]].get(ref[1]), at[ref[0]].get(ref[1])
        source = old_sources.get(ref)
        if previous is None or where is None or source is None or source != new_sources.get(ref) or old_digests.get(ref) != osc1_digest(previous):
            continue
        name, n = where
        if _same_but_pose(ref[0], previous, new_project[name][n]):
            new_project[name][n] = _kept_content(previous, new_project[name][n])
            kept.add(ref)
    if not same_revision:
        return
    old_records = {i.instance_key: i for i in scope}
    for nested in payload["instances"]:
        old = old_records.get(nested["instance_key"])
        if nested["instance_key"] == key or nested["instance_key"] not in new_keys or old is None:
            continue
        members = {(e["kind"], e["parent_id"]) for e in nested["id_map"]
                   if not e["shared"] and e["kind"] in SOURCED and e["parent_id"] in at[e["kind"]]}
        if members and members <= kept:
            nested["placement"] = old.placement.model_dump(mode="json")


def _record_lacked(payload: dict[str, Any], key: str, ancestors: list[str], scope: list[Any], new_keys: AbstractSet[str],
                   kept: AbstractSet[tuple[str, str]] = frozenset()) -> None:
    """A nested instance replaced on its own by a revision that leaves out members of its part of an enclosing import
    (members an enclosing instance holds with their import digest: it records them as deleted) keeps them as their
    part, with ``LACKED_MEMBER_DIGEST``: recorded by the innermost instance whose revision lacks each -- the deepest old
    record holding it whose instance the new revision has (else the nearest one enclosing it) -- and by every instance
    between that one and the replaced one, so the records of enclosing instances hold what nested ones hold. A later
    replacement of any of them may bring the member back, and an object added back with its id is a member added to
    them; it is no edit of them while the Project lacks it. ``scope``: the replaced instance's records and those of the
    instances nested in it, before the replacement.

    The same holds for a definition of the replaced instances that objects outside them use (``kept``: it stays in the
    Project with its content) and that the new revision lacks: those instances keep it as theirs, so it is never left
    to no instance. It is no edit of them while such objects use it (``kept_definitions``); once nothing outside uses it,
    it is a member added to them, which replacing them with ``discard_edits`` removes. A new revision with the same
    definition (same id, same content) holds it as imported."""

    if not ancestors and not kept:
        return
    records = {r["instance_key"]: r for r in payload["instances"]}
    ids = _project_ids(payload["project"])
    held = {(e["kind"], e["parent_id"]) for k in new_keys for e in records[k]["id_map"] if not e.get("shared")}
    imported = {ref for k in ancestors for ref, member in digest_refs(records[k]) if member["digest"] not in _SENTINEL_DIGESTS}
    old_records = {i.instance_key: i for i in scope}

    def depth(k: str) -> int:  # nesting depth below the replaced instance, before the replacement
        n = 0
        while k != key and k in old_records and n <= len(old_records):
            k, n = old_records[k].parent_instance or key, n + 1
        return n

    holders: dict[tuple[str, str], list[str]] = {}
    for old in scope:
        for entry in old.id_map:
            if not entry.shared:
                holders.setdefault((entry.kind, entry.parent_id), []).append(old.instance_key)
    sources = {k: {(e["kind"], e["source_id"]) for e in records[k]["id_map"]} for k in new_keys}
    lacked: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for old in scope:  # the replaced instance's own record first
        for entry in old.id_map:
            ref = (entry.kind, entry.parent_id)
            if entry.shared or ref in seen or ref in held:
                continue
            if ref not in kept and (ref not in imported or entry.parent_id in ids[entry.kind]):
                continue
            seen.add(ref)
            innermost = max(holders[ref], key=depth)
            while innermost not in new_keys:
                innermost = old_records[innermost].parent_instance or key
            chain = [innermost]
            while chain[-1] != key:
                parent = records[chain[-1]].get("parent_instance")
                chain.append(parent if parent in new_keys else key)
            for k in reversed(chain):  # the replaced instance first
                own = old_records.get(k)
                source = next((e.source_id for e in own.id_map if (e.kind, e.parent_id) == ref and not e.shared), None) if own else None
                source = source if source is not None else _enclosed_source_id(k, *ref)
                if source is None or (ref[0], source) in sources[k]:
                    continue
                # where the old record of this instance had it (its import order: the place it takes back in groups)
                id_map = records[k]["id_map"]
                refs = [(e.kind, e.parent_id) for e in own.id_map] if own is not None else []
                at = len(id_map)
                if ref in refs:
                    before = set(refs[:refs.index(ref)])
                    at = max((n + 1 for n, e in enumerate(id_map) if (e["kind"], e["parent_id"]) in before), default=0)
                id_map.insert(at, {"kind": ref[0], "source_id": source, "parent_id": ref[1], "shared": False})
                sources[k].add((ref[0], source))
                lacked.setdefault(k, []).append(ref)
    for k, refs in lacked.items():
        rows = digest_refs(records[k])
        rows += [(ref, {"parent_id": ref[1], "digest": LACKED_MEMBER_DIGEST}) for ref in refs]
        # in id map order, as an import writes them (an import of a module holding the same record writes the same bytes)
        order = {(e["kind"], e["parent_id"]): n for n, e in reversed(list(enumerate(records[k]["id_map"])))}
        rows.sort(key=lambda row: order.get(row[0], len(order)))
        _write_digests(records[k], rows, {parent_id for _, parent_id in refs})


def _write_digests(record: dict[str, Any], rows: list[tuple[tuple[str, str], dict[str, Any]]], touched: AbstractSet[str]) -> None:
    """Set ``record``'s member digests to ``rows`` (``digest_refs`` pairs); a digest of an id in ``touched`` names its
    member's kind where members of different kinds share the id (``MemberDigestV1``)."""

    kinds: dict[str, set[str]] = {}
    for entry in record["id_map"]:
        if not entry.get("shared"):
            kinds.setdefault(entry["parent_id"], set()).add(entry["kind"])
    record["member_digests"] = [{"parent_id": ref[1], **({"kind": ref[0]} if len(kinds.get(ref[1], ())) > 1 else {}), "digest": member["digest"],
                                 **{name: member[name] for name in ("source_digest", _NESTED_REVISION) if member.get(name)}}
                                if ref[1] in touched else member for ref, member in rows]


def _enclosed_source_id(enclosing_key: str, kind: str, parent_id: str) -> str | None:
    """The id a member of a nested instance has in the module of an instance enclosing it: its Project id without the
    enclosing instance's prefix (``None`` when the id does not carry that prefix or the rest is not a valid id)."""

    prefix = (enclosing_key.upper() if kind in ("MATERIAL", "COATING") else enclosing_key) + "_"
    source = parent_id[len(prefix):] if parent_id.startswith(prefix) else ""
    return source if source and valid_id(kind, source) else None


def _record_in_enclosing(payload: dict[str, Any], ws: OpticalWorkspaceV1, enclosing: list[str], before: AbstractSet[tuple[str, str]],
                         old_nested: AbstractSet[str], new_keys: AbstractSet[str]) -> set[tuple[str, str, str]]:
    """A nested instance replaced on its own changes the content of the instances enclosing it (``enclosing``, innermost
    first); their records say how. A member the new revision adds is recorded as added: an id map entry (source id: its
    id without the enclosing key's prefix) with ``ADDED_EDIT_DIGEST`` (an edit of the enclosing instance) and a place in
    the enclosing import group. An added member the new revision no longer has is forgotten (the enclosing import never
    had it) unless an instance between holds it (an enclosing instance replaced on its own brought it in its import). A
    member of the enclosing import the revision lacks stays recorded as deleted, one it changes keeps its import digest
    (``prune_to_project``, ``edited_members``): edits of the enclosing instance as well.

    A definition an enclosing record holds as lacked by its revision (``LACKED_MEMBER_DIGEST``: kept for objects outside
    a replaced instance, ``_record_lacked``) is released the same way once the Project no longer has it and no instance
    nested in that one holds it, unless an instance enclosing that one holds it with an import digest (then it is part
    of that import, recorded as deleted there). Such a row is no edit (the revision lacks it and so does the Project), so
    releasing it changes no status; it no longer reserves the id, so a later revision of the nested instance may define
    it again (recorded as added, like any member a nested revision adds).

    ``before``: ``(kind, id)`` of the members of the replaced instance and of its nested instances before the
    replacement; ``old_nested``: the instances nested in it before; ``new_keys``: the replaced instance and the instances
    nested in its new revision. Returns ``(instance key, kind, id)`` of every forgotten entry."""

    forgotten: set[tuple[str, str, str]] = set()
    if not enclosing:
        return forgotten
    ids = _project_ids(payload["project"])
    records = {r["instance_key"]: r for r in payload["instances"]}
    new_members: list[tuple[str, str]] = []
    for record in payload["instances"]:
        if record["instance_key"] in new_keys:
            for entry in record["id_map"]:
                ref = (entry["kind"], entry["parent_id"])
                if not entry.get("shared") and ref not in new_members and entry["parent_id"] in ids[entry["kind"]]:
                    new_members.append(ref)
    import_groups = {g["instance_key"]: g for g in payload.get("groups", []) if g.get("origin") == "IMPORT"}
    for n, key in enumerate(enclosing):
        record = records[key]
        inside = (nested_instances(ws, key) - old_nested) | new_keys
        held = {(e["kind"], e["parent_id"]) for k in inside if k in records for e in records[k]["id_map"] if not e.get("shared")}
        # held with an import digest by an instance enclosing this one (``enclosing`` is innermost first)
        claimed = {ref for k in enclosing[n + 1:] for ref, member in digest_refs(records[k]) if member["digest"] not in _SENTINEL_DIGESTS}
        rows = digest_refs(record)
        gone = {ref for ref, member in rows if ref[1] not in ids[ref[0]] and ref not in held and (
            (member["digest"] == ADDED_EDIT_DIGEST and ref in before)
            or (member["digest"] == LACKED_MEMBER_DIGEST and ref[0] in _DEFINITION_KINDS and ref not in claimed))}
        id_map = [e for e in record["id_map"] if e.get("shared") or (e["kind"], e["parent_id"]) not in gone]
        rows = [(ref, member) for ref, member in rows if ref not in gone]
        mapped = {(e["kind"], e["parent_id"]) for e in id_map}
        sources = {(e["kind"], e["source_id"]) for e in id_map}
        added = []
        for kind, parent_id in new_members:
            source = _enclosed_source_id(key, kind, parent_id)
            if (kind, parent_id) in mapped or source is None or (kind, source) in sources:
                continue
            id_map.append({"kind": kind, "source_id": source, "parent_id": parent_id, "shared": False})
            rows.append(((kind, parent_id), {"parent_id": parent_id, "digest": ADDED_EDIT_DIGEST}))
            mapped.add((kind, parent_id))
            sources.add((kind, source))
            added.append((kind, parent_id))
        if not gone and not added:
            continue
        forgotten |= {(key, kind, parent_id) for kind, parent_id in gone}
        record["id_map"] = id_map
        _write_digests(record, rows, {parent_id for _, parent_id in (*added, *gone)})
        group = import_groups.get(key)
        if group is not None:
            members = group.setdefault("members", [])
            members += [{"kind": kind, "id": parent_id} for kind, parent_id in added
                        if kind in ("COMPONENT", "OBSERVATION") and {"kind": kind, "id": parent_id} not in members]
    return forgotten


def nested_revision_digest(source_digest: str, placement: Mapping[str, Any]) -> str:
    """``MemberDigestV1.nested_revision_digest``: osc1 of ``{"source_digest": ..., "placement": ...}``, the source digest
    a replaced instance's record had for a member and the instance's placement. An enclosing record keeps it on a
    component or observation plane of its import that the replacement changed or left out: a later replacement of that
    instance that re-creates the same module object at the same placement gives the member its import content back."""

    return osc1_digest({"source_digest": source_digest, "placement": dict(placement)})


def _follow_recreated(payload: dict[str, Any], instance: Any, ancestors: list[str]) -> None:
    """The records of the instances enclosing a replaced instance (``ancestors``) follow what the replacement did to the
    components and observation planes they hold as it imported them, by exact identity only.

    An enclosing record holds a member as the replaced instance imported it when its digest is the replaced instance's
    old one (the same content) and the replaced instance's record names the module object the member was copied from
    (``source_digest``): that content is that object placed at the instance's placement, moved with it since. Then:

    * the new revision has the same object (the same source id and source digest): the replacement re-created that
      content at the unchanged placement, an edit it discarded included, and the enclosing record takes the re-created
      content's digest (after a whole-instance rotation a fresh placement differs from the rotated content by
      floating-point rounding alone);
    * the new revision changes or lacks it: the enclosing record keeps its digest (an edit, or a deletion while the
      Project lacks the member) and names the old object and the placement (``nested_revision_digest``);
    * otherwise an enclosing record naming the new revision's object at the instance's placement (an earlier
      replacement changed or removed the member, this one goes back) takes the re-created content's digest;
    * an enclosing record whose digest is the re-created content's needs no name any more.

    The digests compared are exact: an edited or deleted member moved with its instance (``moved_edit_digest``) is held
    as the replaced instance imported it only when both records moved together since they agreed. A member added by a
    nested replacement or one a revision lacks never matches, nor does the moved-edit digest earlier builds wrote
    (``UNDERIVED_MOVED_DIGEST``); a record without source digests (written by an earlier build) proves nothing."""

    if not ancestors:
        return
    key = instance.instance_key
    records = {r["instance_key"]: r for r in payload["instances"]}
    new_record = records[key]
    old_digests, old_sources = instance.member_digest_map(), instance.source_digest_map()
    old_ids = {(e.kind, e.parent_id): e.source_id for e in instance.id_map if not e.shared}
    new_ids = {(e["kind"], e["parent_id"]): e["source_id"] for e in new_record["id_map"] if not e.get("shared")}
    new_sources = {ref: member.get("source_digest") for ref, member in digest_refs(new_record)}
    placement = instance.placement.model_dump(mode="json")
    index = {kind: {item[ident]: item for item in payload["project"].get(name) or []} for kind, name, ident in OBJECTS if kind in SOURCED}
    for ancestor in ancestors:
        for ref, member in digest_refs(records[ancestor]):
            digest = member["digest"]
            if ref[0] not in SOURCED or digest in _SENTINEL_DIGESTS or digest == UNDERIVED_MOVED_DIGEST:
                continue
            item = index[ref[0]].get(ref[1])
            source = new_sources.get(ref) if item is not None and ref in new_ids else None
            if source is not None and ref in old_ids and old_ids[ref] != new_ids[ref]:
                source = None
            if digest == old_digests.get(ref) and old_sources.get(ref) is not None:
                if source == old_sources[ref]:
                    member["digest"] = osc1_digest(item)
                    member.pop(_NESTED_REVISION, None)
                else:
                    member[_NESTED_REVISION] = nested_revision_digest(old_sources[ref], placement)
            elif source is not None and member.get(_NESTED_REVISION) == nested_revision_digest(source, placement):
                member["digest"] = osc1_digest(item)
                member.pop(_NESTED_REVISION)
            elif source is not None and member["digest"] == osc1_digest(item):
                member.pop(_NESTED_REVISION, None)  # re-created exactly as the enclosing instance imported it


def nested_instances(ws: OpticalWorkspaceV1, key: str) -> set[str]:
    """The instances the import of instance ``key`` created as instances of a composed module, at any depth
    (``parent_instance``). A workspace written before instances recorded their provenance: the ``<key>_`` instances
    whose (non-empty) components, observation planes and targets all belong to ``key``."""

    children: dict[str, list[str]] = {}
    for item in ws.instances:
        if item.parent_instance is not None:
            children.setdefault(item.parent_instance, []).append(item.instance_key)
    found: set[str] = set()
    pending = [key]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    instance = ws.instance(key)
    owned = {(e.kind, e.parent_id) for e in instance.id_map if e.kind in ("COMPONENT", "OBSERVATION", "TARGET")} if instance else set()
    for item in ws.instances:
        if item.parent_instance is None and item.instance_key != key and item.instance_key.startswith(f"{key}_"):
            members = {(e.kind, e.parent_id) for e in item.id_map if e.kind in ("COMPONENT", "OBSERVATION", "TARGET")}
            if members and members <= owned:
                found.add(item.instance_key)
    return found


#: How a replacement's refusals name the kinds of the ids they are about.
_KIND_WORDS = {"COMPONENT": "component", "OBSERVATION": "observation plane", "TARGET": "target", "MODEL": "optical model",
               "MATERIAL": "material", "COATING": "coating"}


def _restated(exc: CompositionError, ws: OpticalWorkspaceV1, key: str, kept: AbstractSet[tuple[str, str]], project: dict[str, Any]) -> CompositionError:
    """The ``ID_CONFLICT`` problems of a replacement's re-import, stated for the replacement (its instance key stays):
    a definition of the replaced instance that objects outside it use and that the new revision defines differently
    (with those objects, ``used_by``: the definition stays as it is for them), an id another instance records as a
    deleted member, or an object the Project already has (with the instance holding it, ``held_by``)."""

    problems = []
    for problem in exc.problems:
        details = problem.details or {}
        ref = (details.get("kind"), details.get("parent_id"))
        if problem.code is _E.ID_CONFLICT and ref in kept:
            users = _outside_users(project, *ref, kept)  # type: ignore[misc]
            names = ", ".join(f"{_KIND_WORDS[user['kind']]} {user['id']}" for user in users)
            problem = CompositionProblem(_E.ID_CONFLICT, f"{_KIND_WORDS[ref[0]]} {ref[1]!r} of instance {key!r} is used by {names}, outside the instance, "
                                         "so it stays as it is, and the new revision defines it differently: point those objects at another "
                                         "definition first", problem.location, {**details, "used_by": users})
        elif problem.code is _E.ID_CONFLICT and ref[0] in _KIND_WORDS and isinstance(ref[1], str):
            what = f"{_KIND_WORDS[ref[0]]} {ref[1]!r}"
            owner = details.get("deleted_from")
            if owner is not None:
                message = (f"{what} is a member that instance {owner!r} records as deleted (an object with this id would be its member "
                           f"again), so the new revision of instance {key!r} cannot take its id")
            else:
                holder = next((i.instance_key for i in ws.instances if any((e.kind, e.parent_id) == ref for e in i.id_map)), None)
                details = {**details, "held_by": holder} if holder is not None else details
                message = (f"the Project already has {what}{f' (a member of instance {holder!r})' if holder is not None else ''}, so the new "
                           f"revision of instance {key!r} cannot take its id")
            problem = CompositionProblem(_E.ID_CONFLICT, message, problem.location, details)
        problems.append(problem)
    return CompositionError(problems)


# ------------------------------------------------------------------------------------------------ batch


def _do_batch(ws: OpticalWorkspaceV1, request: dict[str, Any], _m: Any, catalog: Any) -> tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]:
    operations = request.get("operations")
    if not isinstance(operations, list) or not operations:
        raise refuse(_E.BATCH_INVALID, "operations is a non-empty list of {kind, request, module?}", "operations")
    current = ws
    results, warnings = [], []
    for index, op in enumerate(operations):
        where = f"operations[{index}]"
        if not isinstance(op, Mapping) or set(op) - {"kind", "request", "module"} or op.get("kind") not in BATCH_KINDS:
            raise refuse(_E.BATCH_INVALID, f"{where} is {{kind, request, module?}} with kind one of {list(BATCH_KINDS)}", where)
        kind = op["kind"]
        sub = _request(op.get("request"), kind)
        if set(sub) & set(COMMON_KEYS):
            raise refuse(_E.BATCH_INVALID, f"{where}: the batch carries op_id, expected_base and require_check, not its operations", where)
        module_ws = load_module(op.get("module")) if kind in MODULE_KINDS else None
        try:
            payload, summary, sub_warnings = _MUTATORS[kind](current, sub, module_ws, catalog)
        except _Unchanged as unchanged:
            results.append({"kind": kind, **unchanged.result})
            continue
        except CompositionError as exc:
            raise CompositionError([CompositionProblem(p.code, f"{where} ({kind}): {p.message}", p.location, p.details) for p in exc.problems]) from None
        current = validate_payload(payload)
        results.append({"kind": kind, **summary, "warnings": [w.as_dict() for w in sub_warnings]})
        warnings += sub_warnings
    return current.payload(), {"results": results}, warnings


_MUTATORS: dict[str, Callable[..., tuple[dict[str, Any], dict[str, Any], list[CompositionProblem]]]] = {
    "import": _do_import,
    "transform": _do_transform,
    "group_create": _do_group_create,
    "group_dissolve": _do_group_dissolve,
    "group_set_members": _membership("set"),
    "group_add": _membership("add"),
    "group_remove": _membership("remove"),
    "replace_instance": _do_replace_instance,
    "update_project": _do_update_project,
    "batch": _do_batch,
}


# ------------------------------------------------------------------------------------------------ public functions


def import_layout(workspace: Any, module: Any, request: Any, *, catalog: Any = None) -> OperationResult:
    """Import ``module`` (a module workspace or a Project) into ``workspace`` as instance ``request["instance_key"]``
    (``importing`` module docstring). Request: ``instance_key``, ``placement`` (the module's design frame in the
    workspace's design frame; default identity), ``label``, ``group_id`` (of the import's group; default the instance
    key), ``fixtures`` (``{policy, overrides, rebind, unbound_targets}``), ``catalog_binding`` (``repin``: move catalog
    components to the active release when their pins differ and every product is unchanged there; ``refuse``),
    ``op_id``, ``expected_base``. ``result``: the id map, fixture outcomes, carried requirements, the module's interfaces
    in the workspace frame and warnings."""

    return apply_operation("import", workspace, request, module=module, catalog=catalog)


def transform(workspace: Any, request: Any, *, catalog: Any = None) -> OperationResult:
    """Move the selected objects as one rigid body (``selector``) by ``translation_mm`` and / or ``rotation``
    (``{quaternion_wxyz}`` or ``{axis, angle_deg}``), both stated in the axes of ``frame`` (``"design"``, ``{group}``,
    ``{component}``, ``{instance}``), rotating about ``pivot`` (``"frame_origin"`` (default), ``"selection_centroid"``,
    ``{point_mm}`` in the frame, ``{component}``): ``x -> R (x - pivot) + pivot + t``. Each selected object moves once;
    an observation plane attached to a moved device moves with it (``result.followers``, ``selected``: whether the
    selector named it); nothing else moves. An instance's placement and a
    group's frame move with it when all their members moved. ``result.affected`` lists every moved object's pose before
    and after (design frame)."""

    return apply_operation("transform", workspace, request, catalog=catalog)


def group_create(workspace: Any, request: Any) -> OperationResult:
    """A new editing group ``group_id`` (``label``, ``members``: a selector, ``frame``: a pose or ``{component}`` /
    ``{instance}`` / ``{group}``, whose current pose it copies). Never changes the Project."""

    return apply_operation("group_create", workspace, request)


def group_dissolve(workspace: Any, request: Any) -> OperationResult:
    """Remove group ``group_id`` (its members stay where they are)."""

    return apply_operation("group_dissolve", workspace, request)


def group_set_members(workspace: Any, request: Any) -> OperationResult:
    """Replace group ``group_id``'s members by the ``members`` selector's objects."""

    return apply_operation("group_set_members", workspace, request)


def group_add(workspace: Any, request: Any) -> OperationResult:
    """Add the ``members`` selector's objects to group ``group_id``."""

    return apply_operation("group_add", workspace, request)


def group_remove(workspace: Any, request: Any) -> OperationResult:
    """Remove the ``members`` selector's objects from group ``group_id``."""

    return apply_operation("group_remove", workspace, request)


def replace_instance(workspace: Any, module: Any, request: Any, *, catalog: Any = None) -> OperationResult:
    """Replace instance ``instance_key`` with ``module`` (normally a new revision of its module) at the same placement:
    the instance's members are removed and the module imported under the same key, at the same place in the Project's
    lists; group memberships follow the source ids. The instances its import created from a composed module
    (``parent_instance``) are replaced with it, with every member they hold. Refused (``INSTANCE_EDITED``, listing them)
    when the instance or an instance nested in it has local edits (``edited_members``: members edited or deleted since
    the import, members added), unless ``discard_edits`` is true (``result.replaced.edits_discarded`` then lists them;
    the new revision brings deleted members back). A nested instance replaced on its own updates the records of the
    instances enclosing it: a member its new revision adds is recorded there as added (an edit of each enclosing
    instance), and the enclosing instances hold it (selection, whole-instance moves, import group); a member of its part
    of the enclosing import that the new revision lacks stays recorded as deleted there and as lacked by the innermost
    instance whose revision lacks it and every instance between, so a later replacement of any of them may bring the
    member back. Only that part: an id an enclosing instance records as a deleted member of its own
    stays reserved (``ID_CONFLICT``). A member the new revision re-creates from the module object the enclosing
    instances imported (an edited one it discards included) is their import content again; for one it changes or
    leaves out they name that object and the placement (``nested_revision_digest``), so the nested instance going back
    to it leaves them unedited.

    Every member gets the new revision's content, decided by exact identity (no numeric tolerance): only an unedited
    component or observation plane whose module object the new revision has with the same source digest keeps its
    exact content (a whole-instance move or rotation is no edit). A definition of the replaced instances that objects
    outside them use stays with its content as the workspace's own (``result.replaced.kept_definitions``,
    ``kept_definitions``)."""

    return apply_operation("replace_instance", workspace, request, module=module, catalog=catalog)


def update_project(workspace: Any, request: Any, *, catalog: Any = None) -> OperationResult:
    """Replace the workspace's Project by ``request["project"]`` (a revision edited outside this workspace):
    groups and the module descriptor drop the objects that are no longer there (reported). An instance member that is no
    longer there stays in its instance's record as a deleted member, an edit of the instance (``removed_instance_members``
    and warning ``INSTANCE_MEMBERS_REMOVED``); an object of the same kind and id added back is that member again (an
    edit when its content differs from the import's; a component or observation plane always, once the whole instance
    moved in between) and rejoins the instance's import group (``restored_instance_members``). A catalog component whose
    only change is its catalog release pin (``product.catalog_version``) is not an edit of its instance."""

    return apply_operation("update_project", workspace, request, catalog=catalog)


def batch(workspace: Any, request: Any, *, catalog: Any = None) -> OperationResult:
    """Apply ``request["operations"]`` (``[{kind, request, module?}]``, kinds: ``BATCH_KINDS``) in order as one
    operation: all or nothing, one journal entry, the physical check on the final state only."""

    return apply_operation("batch", workspace, request, catalog=catalog)


def select(workspace: Any, selector: Any) -> dict[str, Any]:
    """The objects ``selector`` selects (``selection`` module docstring)."""

    ws = load_workspace(workspace)
    try:
        return resolve_selection(ws, selector).as_dict()
    except CompositionError as exc:
        exc.workspace_digest = ws.workspace_digest()
        raise


def export_module(workspace: Any, request: Any) -> OpticalWorkspaceV1:
    """The workspace as a reusable module: ``request`` gives ``module_id``, ``title`` and optionally ``revision``,
    ``description``, ``fixtures``, ``interfaces`` and ``requirements`` (``ModuleDescriptorV1``). The module keeps the
    Project, its instances and groups (so a composed layout keeps its provenance and organization), and the system
    validations of exactly this Project; the journal is not exported. The workspace itself is unchanged."""

    ws = load_workspace(workspace)
    try:
        if not isinstance(request, Mapping):
            raise refuse(_E.REQUEST_INVALID, "an export request is {module_id, title, revision?, description?, fixtures?, interfaces?, requirements?}")
        descriptor = {**(ws.module.model_dump(mode="json", exclude_none=True) if ws.module is not None else {}), **dict(request)}
        if ws.instances:
            parts = ", ".join(f"{i.source.title} ({i.source.module_id}) as {i.instance_key}" for i in ws.instances)
            summary = f"Composed of {len(ws.instances)} imported layout{'s' if len(ws.instances) != 1 else ''}: {parts}."
            current = (descriptor.get("description") or "").split(" Composed of ")[0]
            current = "" if current.startswith("Composed of ") else current
            text = f"{current} {summary}" if current else summary
            descriptor["description"] = text if len(text) <= DESCRIPTION_MAX else text[:DESCRIPTION_MAX - len(ELLIPSIS)] + ELLIPSIS
        try:
            module = ModuleDescriptorV1.model_validate(descriptor)
        except ValidationError as exc:
            raise refuse(_E.REQUEST_INVALID, f"the module descriptor is not valid: {_validation_message(exc)}") from None
        project_hash = ws.project.content_hash()
        payload = {"schema": "opensci.optical_workspace", "schema_version": "1.0", "project": ws.project.payload(),
                   "module": module.model_dump(mode="json", exclude_none=True),
                   "instances": [i.model_dump(mode="json", exclude_none=True) for i in ws.instances],
                   "groups": [g.model_dump(mode="json", exclude_none=True) for g in ws.groups],
                   "validations": [v.model_dump(mode="json", exclude_none=True) for v in ws.validations
                                   if v.scope == "SYSTEM" and v.project_hash == project_hash]}
        try:
            return OpticalWorkspaceV1.model_validate(payload)
        except ValidationError as exc:
            raise refuse(_E.REQUEST_INVALID, f"the module is not valid: {_validation_message(exc)}") from None
    except CompositionError as exc:
        exc.workspace_digest = ws.workspace_digest()
        raise


def history(workspace: Any, op_id: str | None = None, limit: int | None = None) -> dict[str, Any]:
    """The journal (oldest first; ``limit``: the last ``limit`` entries), or, with ``op_id``, whether that operation was
    applied (``found``) and its record -- how a caller tells whether an interrupted call completed."""

    ws = load_workspace(workspace)
    if op_id is not None:
        record = ws.operation(op_id)
        return {"found": record is not None, "operation": record.model_dump(mode="json") if record is not None else None,
                "workspace_digest": ws.workspace_digest()}
    operations = [o.model_dump(mode="json") for o in ws.operations]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise refuse(_E.REQUEST_INVALID, "limit is a non-negative integer", "limit", workspace_digest=ws.workspace_digest())
        operations = operations[len(operations) - limit:] if limit else []
    return {"operations": operations, "workspace_digest": ws.workspace_digest()}


__all__ = [
    "BATCH_KINDS",
    "REQUEST_KEYS",
    "OperationResult",
    "apply_operation",
    "batch",
    "edited_members",
    "export_module",
    "group_add",
    "group_create",
    "group_dissolve",
    "group_remove",
    "group_set_members",
    "history",
    "import_layout",
    "kept_definitions",
    "prune_to_project",
    "replace_instance",
    "select",
    "transform",
    "update_project",
]
