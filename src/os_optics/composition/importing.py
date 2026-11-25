"""Copy-on-import of a module into a workspace (``import_layout``) and the removal half of ``replace_instance``.

An import copies the module's Project into the workspace's Project:

* components, observation planes and targets get the id ``<instance_key>_<id>``;
* optical models, materials and coatings are copied as ``<instance_key>_<id>`` (``<INSTANCE_KEY>_<ID>`` for materials
  and coatings, whose ids are upper case), also when the Project already has an identical definition: every instance
  owns its definitions, so editing one instance never changes another instance, the system's own objects or the
  source module;
* poses in the module's design frame are composed with ``placement`` (an observation plane attached to a component keeps
  its pose in that component's frame);
* fixtures are retained, removed or rebound (``FixturePolicy``), targets are carried by their requirement scope;
* the module's own validations travel with the instance (scope ``MODULE``); instances and groups of a composed module
  become instances (``<instance_key>_<key>``) and groups (``<instance_key>_<group_id>``) of the workspace.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .._vendor.shared_contracts.common.geometry import PoseV1
from .._vendor.shared_contracts.common.hashes import osc1_digest
from .._vendor.shared_contracts.common.identifiers import OpenSciId
from .._vendor.shared_contracts.design_catalog.optical_catalog import CoatingId, MaterialId
from .._vendor.shared_contracts.optical_design import OpticalWorkspaceV1
from .._vendor.shared_contracts.optical_design.component import ModelId
from .._vendor.shared_contracts.optical_design.workspace import InstanceKeyV1
from . import rotation as rot
from .document import MATERIAL_KEYS
from .errors import CompositionError, CompositionErrorCode, CompositionProblem, CompositionWarningCode, refuse, warning

#: Roles of a terminal instrument (a detector, a beam dump): an absorbing plane attached to one (``device_instance_id``) is its
#: sensing face and is imported without ``ABSORBING_PLANE_IMPORTED``.
TERMINAL_DEVICE_ROLES = frozenset({"detector", "dump"})
_E = CompositionErrorCode
_W = CompositionWarningCode
POLICIES = ("module_defaults", "retain_all", "remove_all")
UNBOUND_TARGETS = ("refuse", "remove")
CATALOG_BINDINGS = ("repin", "refuse")
_GRAMMAR = {"COMPONENT": TypeAdapter(OpenSciId), "OBSERVATION": TypeAdapter(OpenSciId), "TARGET": TypeAdapter(OpenSciId),
            "MODEL": TypeAdapter(ModelId), "MATERIAL": TypeAdapter(MaterialId), "COATING": TypeAdapter(CoatingId)}
_LIMIT = {"COMPONENT": 128, "OBSERVATION": 128, "TARGET": 128, "MODEL": 64, "MATERIAL": 64, "COATING": 96}
_INSTANCE_KEY = TypeAdapter(InstanceKeyV1)
LIBRARY = (("MODEL", "optical_models", "model_id"), ("MATERIAL", "model_materials", "material_id"), ("COATING", "model_coatings", "coating_id"))
OBJECTS = (("COMPONENT", "components", "instance_id"), ("OBSERVATION", "observations", "observation_id"), ("TARGET", "targets", "target_id"))
#: The members whose digest names the module object they were copied from (``MemberDigestV1.source_digest``): the ones an
#: import places, whose content a whole-instance move changes.
SOURCED = ("COMPONENT", "OBSERVATION")
#: Key under which an import hands ``member_digests`` the source digests of a new instance record (removed there).
_SOURCES = "_source_digests"
#: The digest with which a nested instance records a member of its part of an enclosing import that its current revision
#: lacks (``MemberDigestV1``): the enclosing instance records the member as deleted, and this record says the member
#: belongs to this instance's part (replacing this instance may bring it back). Not an edit of this instance while the
#: Project lacks it too; an object with its id in the Project is a member added to this instance (``ADDED``), except a
#: definition objects outside the instance use (kept for them).
LACKED_MEMBER_DIGEST = osc1_digest({"member_its_revision_lacks": True})
#: Key under which ``_nested`` hands the import the definitions a nested instance of the module holds as lacked by its
#: revision (removed by ``_keep_lacked``).
_LACKED = "_lacked_definitions"
_DEFINITIONS = frozenset(kind for kind, _, _ in LIBRARY)
PATH_KEYS = ("path_id", "path_id_a", "path_id_b")
#: An unnamed module (a bare Project) gets the id ``layout-<first digits of its Project hash>``.
LAYOUT_ID_DIGITS = 12
#: The longest instance key (``InstanceKeyV1``).
INSTANCE_KEY_MAX = 24


def namespaced(key: str, kind: str, source_id: str) -> str:
    """``<key>_<id>`` (upper case for materials and coatings), refused as ``ID_TOO_LONG`` when it exceeds the id's
    grammar."""

    prefix = key.upper() if kind in ("MATERIAL", "COATING") else key
    new_id = f"{prefix}_{source_id}"
    if len(new_id) > _LIMIT[kind]:
        raise refuse(_E.ID_TOO_LONG, f"{kind.lower()} id {new_id!r} would be {len(new_id)} characters (at most {_LIMIT[kind]}); use a shorter "
                                     f"instance_key or rename {source_id!r} in the module", f"{kind.lower()} {source_id}",
                     kind=kind, source_id=source_id, parent_id=new_id, limit=_LIMIT[kind])
    try:
        _GRAMMAR[kind].validate_python(new_id)
    except ValidationError:
        raise refuse(_E.REQUEST_INVALID, f"{kind.lower()} id {new_id!r} is not a valid id", f"{kind.lower()} {source_id}") from None
    return new_id


def valid_id(kind: str, value: str) -> bool:
    """Whether ``value`` is a valid id of ``kind`` (an id map kind)."""

    try:
        _GRAMMAR[kind].validate_python(value)
    except ValidationError:
        return False
    return len(value) <= _LIMIT[kind]


def pose_tuple(pose: Mapping[str, Any] | None) -> tuple[rot.Vector, rot.Quaternion]:
    pose = pose or {}
    position = tuple(float(v) for v in pose.get("position_mm", (0.0, 0.0, 0.0)))
    quaternion = tuple(float(v) for v in pose.get("quaternion_wxyz", rot.IDENTITY))
    return position, quaternion  # type: ignore[return-value]


def pose_dict(position: Any, quaternion: Any) -> dict[str, Any]:
    return {"position_mm": [float(v) for v in position], "quaternion_wxyz": [float(v) for v in quaternion]}


def port_rows(ports: Any, placement: tuple[rot.Vector, rot.Quaternion], beam_reference: Any) -> list[dict[str, Any]]:
    """Interface ports (``InterfacePortV1``, in a module's design frame) in the workspace's design frame through
    ``placement``: ``point_mm`` = R p + t, ``direction`` = R d. ``beam_reference(source_id)`` names the Project component
    a port's ``beam_reference`` became (``None``: the import did not copy or rebind it)."""

    rows = []
    for port in ports:
        if rot.is_identity(placement[1]):
            point, direction = rot.add(port.point_mm, placement[0]), tuple(float(c) for c in port.direction)
        else:
            point, direction = rot.add(rot.rotate(placement[1], port.point_mm), placement[0]), rot.rotate(placement[1], port.direction)
        row: dict[str, Any] = {"port_id": port.port_id, "role": port.role, "point_mm": list(point), "direction": list(direction)}
        if port.wavelength_band_nm is not None:
            row["wavelength_band_nm"] = list(port.wavelength_band_nm)
        if port.beam_reference is not None:
            row["beam_reference"] = beam_reference(port.beam_reference)
        if port.aperture_radius_mm is not None:
            row["aperture_radius_mm"] = port.aperture_radius_mm
        rows.append(row)
    return rows


def instance_ports(instance: Any, components: AbstractSet[str] | None = None) -> list[dict[str, Any]]:
    """A workspace instance's interface ports in the Project's design frame at its current ``placement`` (moved with the
    instance by every whole-instance transform); a port's ``beam_reference`` is the component its source became (the
    copied source, or the system component a fixture was rebound to; ``None`` when the import removed it or, with
    ``components`` (the Project's component ids), when the Project no longer has it)."""

    copied = {e.source_id: e.parent_id for e in instance.id_map if e.kind == "COMPONENT"}
    rebound = {f.source_id: f.rebound_to for f in instance.fixtures if f.object_kind == "COMPONENT" and f.rebound_to is not None}
    placement = pose_tuple(instance.placement.model_dump(mode="json"))

    def reference(source: str) -> str | None:
        component = rebound.get(source, copied.get(source))
        return component if components is None or component in components else None

    return port_rows(instance.interfaces, placement, reference)


def without_pin(item: Mapping[str, Any]) -> Any:
    """``item`` without its catalog release pin (``product.catalog_version``)."""

    product = item.get("product")
    if not isinstance(product, Mapping) or "catalog_version" not in product:
        return item
    return {**item, "product": {k: v for k, v in product.items() if k != "catalog_version"}}


def source_digest(item: Mapping[str, Any]) -> str:
    """The source digest of a module object (``MemberDigestV1.source_digest``): osc1 of the object without its catalog
    release pin, which the catalog binding sets for the whole Project (a revision whose object differs only by its pin
    has the same object)."""

    return osc1_digest(without_pin(item))


def placed(placement: tuple[rot.Vector, rot.Quaternion], pose: Mapping[str, Any] | None) -> dict[str, Any]:
    """``pose`` (in the module's design frame) in the workspace's design frame."""

    position, quaternion = pose_tuple(pose)
    new_position, new_quaternion = rot.compose_pose(placement[0], placement[1], position, quaternion)
    return pose_dict(new_position, new_quaternion)


def rename_model_references(model: Any, materials: Mapping[str, str], coatings: Mapping[str, str], models: Mapping[str, str]) -> Any:
    """A copy of a model payload whose material, coating and compound-member references are renamed."""

    def walk(node: Any, parent_key: str | None = None) -> Any:
        if isinstance(node, Mapping):
            out = {}
            for key, value in node.items():
                if key in MATERIAL_KEYS and isinstance(value, str):
                    out[key] = materials.get(value, value)
                elif key == "media" and isinstance(value, list):
                    out[key] = [materials.get(v, v) if isinstance(v, str) else v for v in value]
                elif key == "coating_id" and isinstance(value, str):
                    out[key] = coatings.get(value, value)
                elif key == "members" and isinstance(value, list):
                    out[key] = [{**m, "model_id": models.get(m.get("model_id"), m.get("model_id"))} if isinstance(m, Mapping) else m for m in value]
                else:
                    out[key] = walk(value, key)
            return out
        if isinstance(node, list):
            return [walk(item, parent_key) for item in node]
        return node

    return walk(model)


@dataclass
class FixturePolicy:
    """What an import does with the module's fixtures: ``policy`` (``module_defaults``: each fixture's
    ``default_action``; ``retain_all``; ``remove_all``), per-object ``overrides`` (``RETAIN`` / ``REMOVE``), ``rebind``
    (a removed fixture -> the Project object that takes its place in the targets that used it) and ``unbound_targets``
    (``refuse``: refuse the import while a carried target uses a removed fixture that is not rebound; ``remove``: leave
    such targets out)."""

    policy: str = "module_defaults"
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    rebind: dict[tuple[str, str], str] = field(default_factory=dict)
    unbound_targets: str = "refuse"


def parse_fixture_policy(value: Any, module: OpticalWorkspaceV1) -> FixturePolicy:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise refuse(_E.REQUEST_INVALID, "fixtures is an object {policy, overrides, rebind, unbound_targets}", "fixtures")
    unknown = sorted(set(value) - {"policy", "overrides", "rebind", "unbound_targets"})
    if unknown:
        raise refuse(_E.REQUEST_INVALID, f"fixtures takes policy, overrides, rebind and unbound_targets, not {unknown}", "fixtures")
    policy = value.get("policy", "module_defaults")
    if policy not in POLICIES:
        raise refuse(_E.REQUEST_INVALID, f"fixtures.policy is one of {list(POLICIES)}, not {policy!r}", "fixtures.policy")
    unbound = value.get("unbound_targets", "refuse")
    if unbound not in UNBOUND_TARGETS:
        raise refuse(_E.REQUEST_INVALID, f"fixtures.unbound_targets is one of {list(UNBOUND_TARGETS)}, not {unbound!r}", "fixtures.unbound_targets")
    fixtures = {(f.object_kind, f.object_id): f for f in (module.module.fixtures if module.module is not None else ())}
    problems: list[CompositionProblem] = []

    def key_of(object_id: Any, kind: Any, where: str) -> tuple[str, str] | None:
        if not isinstance(object_id, str):
            problems.append(CompositionProblem(_E.REQUEST_INVALID, f"{where}: an object id is a string", where))
            return None
        matches = [k for k in fixtures if k[1] == object_id and (kind is None or k[0] == kind)]
        if not matches:
            problems.append(CompositionProblem(_E.FIXTURE_INVALID, f"{where}: {object_id!r} is not a fixture of module "
                                               f"{module.module.module_id if module.module else '(no module descriptor)'}; fixtures: "
                                               f"{sorted(k[1] for k in fixtures)}", where, {"object_id": object_id}))
            return None
        if len(matches) > 1:
            problems.append(CompositionProblem(_E.FIXTURE_INVALID, f"{where}: {object_id!r} is a fixture of several kinds; give "
                                               "[{object_kind, object_id, ...}] entries", where, {"object_id": object_id}))
            return None
        return matches[0]

    def entries(raw: Any, name: str, value_key: str) -> list[tuple[str | None, Any, Any]]:
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            return [(None, k, v) for k, v in raw.items()]
        if isinstance(raw, list) and all(isinstance(e, Mapping) for e in raw):
            return [(e.get("object_kind"), e.get("object_id"), e.get(value_key)) for e in raw]
        problems.append(CompositionProblem(_E.REQUEST_INVALID, f"fixtures.{name} is an object {{object_id: value}} or a list of "
                                           f"{{object_kind, object_id, {value_key}}}", f"fixtures.{name}"))
        return []

    overrides: dict[tuple[str, str], str] = {}
    for kind, object_id, action in entries(value.get("overrides"), "overrides", "action"):
        if action not in ("RETAIN", "REMOVE"):
            problems.append(CompositionProblem(_E.REQUEST_INVALID, f"fixtures.overrides[{object_id!r}] is RETAIN or REMOVE, not {action!r}",
                                               "fixtures.overrides"))
            continue
        key = key_of(object_id, kind, f"fixtures.overrides[{object_id!r}]")
        if key is not None:
            overrides[key] = action
    rebind: dict[tuple[str, str], str] = {}
    for kind, object_id, target in entries(value.get("rebind"), "rebind", "rebind_to"):
        key = key_of(object_id, kind, f"fixtures.rebind[{object_id!r}]")
        if key is None:
            continue
        if not isinstance(target, str):
            problems.append(CompositionProblem(_E.REQUEST_INVALID, f"fixtures.rebind[{object_id!r}] names a Project object id", "fixtures.rebind"))
            continue
        if fixtures[key].role == "TEST_TARGET":
            problems.append(CompositionProblem(_E.FIXTURE_INVALID, f"fixtures.rebind[{object_id!r}]: a test target is removed or retained, not "
                                               "rebound", "fixtures.rebind"))
            continue
        if overrides.get(key) == "RETAIN":
            problems.append(CompositionProblem(_E.FIXTURE_INVALID, f"fixture {object_id!r} is both retained and rebound", "fixtures.rebind"))
            continue
        rebind[key] = target
    if problems:
        raise CompositionError(problems)
    return FixturePolicy(policy, overrides, rebind, unbound)


@dataclass
class ImportOutcome:
    """The Project, instance record(s) and group(s) an import produced, and its report."""

    project: dict[str, Any]
    instances: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    report: dict[str, Any]
    warnings: list[CompositionProblem]
    maps: dict[str, dict[str, str]]


def _fixture_action(policy: FixturePolicy, fixture: Any) -> str:
    key = (fixture.object_kind, fixture.object_id)
    if key in policy.rebind:
        return "REBOUND"
    if key in policy.overrides:
        return "RETAINED" if policy.overrides[key] == "RETAIN" else "REMOVED"
    if policy.policy == "retain_all":
        return "RETAINED"
    if policy.policy == "remove_all":
        return "REMOVED"
    return "RETAINED" if fixture.default_action == "RETAIN" else "REMOVED"


def _used_models(components: list[Mapping[str, Any]], models: Mapping[str, Mapping[str, Any]]) -> set[str]:
    used: set[str] = set()
    pending = [c.get("optical_model_id") for c in components if c.get("optical_model_id") is not None]
    while pending:
        model_id = pending.pop()
        if model_id in used or model_id not in models:
            continue
        used.add(model_id)
        for member in (models[model_id].get("model") or {}).get("members") or []:
            pending.append(member.get("model_id"))
    return used


def import_module(parent: OpticalWorkspaceV1, module: OpticalWorkspaceV1, *, instance_key: str, placement: Any = None,
                  label: str | None = None, fixtures: FixturePolicy | None = None, group_id: str | None = None,
                  parent_project: dict[str, Any] | None = None, keep_group_ids: bool = False,
                  reserved: Mapping[str, Mapping[str, str]] | None = None, adopt: AbstractSet[tuple[str, str]] = frozenset()) -> ImportOutcome:
    """Copy ``module`` into ``parent``'s Project (module docstring). ``parent_project``: the Project payload to import
    into instead of ``parent.project`` (``replace_instance`` passes the Project with the old instance removed).
    ``reserved``: kind -> {id: instance key} of members instances record as deleted; an id among them is refused like an
    id the Project has (``ID_CONFLICT``). ``adopt``: ``(kind, id)`` of definitions of the Project (the definitions of a
    replaced instance that other objects use) that the import copies over where the module's definition with that id is
    identical (same osc1 digest: nothing changes for the objects using it); a different one is refused like any id the
    Project has. The caller validates the result as a workspace."""

    try:
        _INSTANCE_KEY.validate_python(instance_key)
    except ValidationError:
        raise refuse(_E.REQUEST_INVALID, f"instance_key {instance_key!r} is a lowercase letter, then lowercase letters, digits or '_' (at most 24)",
                     "instance_key") from None
    try:
        place = PoseV1.model_validate(placement) if placement is not None else PoseV1()
    except ValidationError as exc:
        raise refuse(_E.REQUEST_INVALID, f"placement is a pose {{position_mm, quaternion_wxyz}}: {exc.errors(include_url=False)[0]['msg']}",
                     "placement") from None
    try:
        quaternion = rot.stated_quaternion(place.quaternion_wxyz)
    except ValueError:
        raise refuse(_E.REQUEST_INVALID, "placement.quaternion_wxyz is a non-zero quaternion (its length does not matter)",
                     "placement") from None
    placement_t = (tuple(place.position_mm), quaternion)
    policy = fixtures or FixturePolicy()
    key = instance_key
    project = copy.deepcopy(parent_project if parent_project is not None else parent.project.payload())
    source = module.project.payload()
    descriptor = module.module
    problems: list[CompositionProblem] = []
    warnings: list[CompositionProblem] = []
    reserved = reserved or {}

    def taken(kind: str, new_id: str, where: str) -> CompositionProblem | None:
        owner = reserved.get(kind, {}).get(new_id)
        if owner is None:
            return None
        return CompositionProblem(_E.ID_CONFLICT, f"{kind.lower()} id {new_id!r} belongs to instance {owner!r}, which records that member as "
                                  "deleted (an object with this id becomes its member again); import with another instance_key", where,
                                  {"kind": kind, "parent_id": new_id, "deleted_from": owner})

    # ----------------------------------------------------------------- fixtures and requirements
    fixture_roles = {(f.object_kind, f.object_id): f for f in (descriptor.fixtures if descriptor is not None else ())}
    actions = {k: _fixture_action(policy, f) for k, f in fixture_roles.items()}
    removed = {k for k, a in actions.items() if a != "RETAINED"}
    removed_components = {i for (kind, i) in removed if kind == "COMPONENT"}
    removed_observations = {i for (kind, i) in removed if kind == "OBSERVATION"}
    removed_targets = {i for (kind, i) in removed if kind == "TARGET"}
    scopes = {r.target_id: r.scope for r in (descriptor.requirements if descriptor is not None else ())}
    module_only = sorted(t["target_id"] for t in source.get("targets", []) if scopes.get(t["target_id"]) == "MODULE_ONLY")
    for observation in source.get("observations", []):
        if observation["observation_id"] in removed_observations:
            continue
        device = observation.get("device_instance_id")
        if device in removed_components:
            problems.append(CompositionProblem(_E.FIXTURE_DEPENDENCY, f"observation {observation['observation_id']!r} is attached to fixture {device!r}, "
                                               "which this import removes: mark the observation as a fixture too or retain the component",
                                               f"observation {observation['observation_id']}", {"observation_id": observation["observation_id"],
                                                                                                "device_instance_id": device}))

    # ----------------------------------------------------------------- object ids
    maps: dict[str, dict[str, str]] = {"COMPONENT": {}, "OBSERVATION": {}, "TARGET": {}, "MODEL": {}, "MATERIAL": {}, "COATING": {}}
    existing = {kind: {item[ident] for item in project.get(field_name, [])} for kind, field_name, ident in OBJECTS}
    for kind, field_name, ident in OBJECTS:
        drop = {"COMPONENT": removed_components, "OBSERVATION": removed_observations, "TARGET": removed_targets | set(module_only)}[kind]
        for item in source.get(field_name, []):
            if item[ident] in drop:
                continue
            try:
                new_id = namespaced(key, kind, item[ident])
            except CompositionError as exc:
                problems.extend(exc.problems)
                continue
            if new_id in existing[kind]:
                problems.append(CompositionProblem(_E.ID_CONFLICT, f"the Project already has {kind.lower()} {new_id!r}; import with another instance_key",
                                                   f"{kind.lower()} {item[ident]}", {"kind": kind, "parent_id": new_id}))
            elif (conflict := taken(kind, new_id, f"{kind.lower()} {item[ident]}")) is not None:
                problems.append(conflict)
            maps[kind][item[ident]] = new_id
    # rebind targets (in the resulting Project: an existing object, or one this import copies)
    source_components = {c["instance_id"]: c for c in source.get("components", [])}
    parent_components = {c["instance_id"]: c for c in project.get("components", [])}
    parent_observations = {o["observation_id"] for o in project.get("observations", [])}
    rebound_to: dict[tuple[str, str], str] = {}
    for fkey, target in policy.rebind.items():
        kind, object_id = fkey
        role = fixture_roles[fkey].role
        new_components = {maps["COMPONENT"][i]: source_components[i] for i in maps["COMPONENT"]}
        if kind == "COMPONENT":
            candidate = parent_components.get(target) or new_components.get(target)
            ok = candidate is not None and (role != "TEST_SOURCE" or candidate.get("optical_role") == "source")
            wanted = "a source component" if role == "TEST_SOURCE" else "a component"
        else:
            ok = target in parent_observations or target in maps["OBSERVATION"].values()
            wanted = "an observation plane"
        if not ok:
            problems.append(CompositionProblem(_E.FIXTURE_INVALID, f"fixtures.rebind[{object_id!r}]: {target!r} is not {wanted} of the Project",
                                               "fixtures.rebind", {"object_id": object_id, "rebind_to": target}))
        rebound_to[fkey] = target

    # ----------------------------------------------------------------- targets that use removed fixtures
    affected: dict[tuple[str, str], list[str]] = {}
    unbound: list[dict[str, Any]] = []
    dropped_targets: list[str] = []
    removed_sources = {i for i in removed_components if source_components.get(i, {}).get("optical_role") == "source"}
    for target in source.get("targets", []):
        tid = target["target_id"]
        if tid not in maps["TARGET"]:
            continue
        uses = []
        for name, kind_key in (("device_instance_id", "COMPONENT"), ("observation_id", "OBSERVATION"), ("source_id", "COMPONENT"),
                               ("source_id_b", "COMPONENT")):
            value = target.get(name)
            if value is not None and (kind_key, value) in removed:
                uses.append((kind_key, value))
        for p in target.get("parameters", []):
            if p.get("key") in PATH_KEYS and isinstance(p.get("value"), str):
                head = p["value"].split("#", 1)[0]
                if head in removed_sources:
                    uses.append(("COMPONENT", head))
        missing = sorted({u for u in uses if u not in rebound_to})
        if missing:
            if policy.unbound_targets == "remove":
                dropped_targets.append(tid)
                del maps["TARGET"][tid]
            else:
                unbound.append({"target_id": tid, "fixtures": [m[1] for m in missing]})
            continue
        for u in set(uses):
            affected.setdefault(u, []).append(maps["TARGET"][tid])
    if unbound:
        problems.append(CompositionProblem(
            _E.FIXTURE_REBIND_REQUIRED,
            "targets use fixtures this import removes: rebind each fixture to an object of the system (fixtures.rebind), retain it, "
            "or leave the targets out (fixtures.unbound_targets = 'remove', or the module's MODULE_ONLY requirement scope)",
            "fixtures", {"targets": unbound}))

    # ----------------------------------------------------------------- library: materials, coatings, models
    kept_components = [c for c in source.get("components", []) if c["instance_id"] in maps["COMPONENT"]]
    source_models = {m["model_id"]: m for m in source.get("optical_models", [])}
    used_by_any = _used_models(source.get("components", []), source_models)
    wanted_models = _used_models(kept_components, source_models) | (set(source_models) - used_by_any)
    wanted_materials: set[str] = set()
    wanted_coatings: set[str] = set()
    from .document import _referenced

    for model_id in wanted_models:
        wanted_materials |= _referenced(source_models[model_id].get("model"), "material")
        wanted_coatings |= _referenced(source_models[model_id].get("model"), "coating")
    library_out: dict[str, list[dict[str, Any]]] = {name: list(project.get(name) or []) for _, name, _ in LIBRARY}
    for kind, field_name, ident in (("MATERIAL", "model_materials", "material_id"), ("COATING", "model_coatings", "coating_id")):
        wanted = wanted_materials if kind == "MATERIAL" else wanted_coatings
        current = library_out[field_name]
        for item in source.get(field_name, []):
            sid = item[ident]
            if sid not in wanted:
                continue
            try:
                new_id = namespaced(key, kind, sid)
            except CompositionError as exc:
                problems.extend(exc.problems)
                continue
            existing = next((c for c in current if c[ident] == new_id), None)
            if existing is not None and (kind, new_id) in adopt and osc1_digest(existing) == osc1_digest({**item, ident: new_id}):
                current.remove(existing)  # the same definition: copied in its place among the module's
                existing = None
            if existing is not None:
                problems.append(CompositionProblem(_E.ID_CONFLICT, f"the Project already declares {kind.lower()} {new_id!r}; import with another instance_key",
                                                   f"{kind.lower()} {sid}", {"kind": kind, "parent_id": new_id}))
                maps[kind][sid] = new_id  # the models using it are compared with their new ids
                continue
            if (conflict := taken(kind, new_id, f"{kind.lower()} {sid}")) is not None:
                problems.append(conflict)
                continue
            maps[kind][sid] = new_id
            current.append({**copy.deepcopy(item), ident: new_id})
    ordered_models = sorted(wanted_models, key=lambda m: (bool((source_models[m].get("model") or {}).get("members")),
                                                          list(source_models).index(m)))
    current_models = library_out["optical_models"]
    for model_id in ordered_models:
        item = copy.deepcopy(source_models[model_id])
        item["model"] = rename_model_references(item.get("model"), maps["MATERIAL"], maps["COATING"], maps["MODEL"])
        try:
            new_id = namespaced(key, "MODEL", model_id)
        except CompositionError as exc:
            problems.extend(exc.problems)
            continue
        existing = next((m for m in current_models if m["model_id"] == new_id), None)
        if existing is not None and ("MODEL", new_id) in adopt and osc1_digest(existing) == osc1_digest({**item, "model_id": new_id}):
            current_models.remove(existing)  # the same definition: copied in its place among the module's
            existing = None
        if existing is not None:
            problems.append(CompositionProblem(_E.ID_CONFLICT, f"the Project already declares optical model {new_id!r}; import with another instance_key",
                                               f"optical model {model_id}", {"kind": "MODEL", "parent_id": new_id}))
            maps["MODEL"][model_id] = new_id  # the compound models using it are compared with their new ids
            continue
        if (conflict := taken("MODEL", new_id, f"optical model {model_id}")) is not None:
            problems.append(conflict)
            continue
        maps["MODEL"][model_id] = new_id
        current_models.append({**item, "model_id": new_id})
    if problems:
        raise CompositionError(problems)

    # ----------------------------------------------------------------- objects
    def ref(kind: str, value: str | None, fkey_kind: str | None = None) -> str | None:
        if value is None:
            return None
        rebind = rebound_to.get((fkey_kind or kind, value))
        return rebind if rebind is not None else maps[kind].get(value, value)

    components = list(project.get("components") or [])
    for component in kept_components:
        new = copy.deepcopy(component)
        new["instance_id"] = maps["COMPONENT"][component["instance_id"]]
        if component.get("optical_model_id") is not None:
            new["optical_model_id"] = maps["MODEL"][component["optical_model_id"]]
        new["nominal_pose"] = placed(placement_t, component.get("nominal_pose"))
        components.append(new)
    observations = list(project.get("observations") or [])
    for observation in source.get("observations", []):
        if observation["observation_id"] not in maps["OBSERVATION"]:
            continue
        new = copy.deepcopy(observation)
        new["observation_id"] = maps["OBSERVATION"][observation["observation_id"]]
        if observation.get("device_instance_id") is not None:
            new["device_instance_id"] = maps["COMPONENT"][observation["device_instance_id"]]
        else:
            new["pose"] = placed(placement_t, observation.get("pose"))
        observations.append(new)
    targets = list(project.get("targets") or [])
    path_renamed: list[str] = []
    rebound_targets: dict[str, str] = {}
    for target in source.get("targets", []):
        if target["target_id"] not in maps["TARGET"]:
            continue
        new = copy.deepcopy(target)
        new["target_id"] = maps["TARGET"][target["target_id"]]
        for name, kind in (("device_instance_id", "COMPONENT"), ("observation_id", "OBSERVATION"), ("source_id", "COMPONENT"),
                           ("source_id_b", "COMPONENT")):
            if target.get(name) is not None:
                new[name] = ref(kind, target[name])
                if (kind, target[name]) in rebound_to:
                    rebound_targets[new["target_id"]] = rebound_to[(kind, target[name])]
        parameters = []
        for p in target.get("parameters", []):
            if p.get("key") in PATH_KEYS and isinstance(p.get("value"), str) and "#" in p["value"]:
                head, rest = p["value"].split("#", 1)
                if head in source_components:
                    p = {**p, "value": f"{ref('COMPONENT', head)}#{rest}"}
                    path_renamed.append(new["target_id"])
            parameters.append(p)
        if parameters:
            new["parameters"] = parameters
        targets.append(new)
    project["components"] = components
    project["observations"] = observations
    project["targets"] = targets
    for _, field_name, _ in LIBRARY:
        if library_out[field_name]:
            project[field_name] = library_out[field_name]
    module_intent = source.get("design_intent")
    parent_intent = project.get("design_intent")
    if module_intent is not None and module_intent != parent_intent:
        if not (parent_project if parent_project is not None else parent.project.payload()).get("components") and parent_intent is None:
            project["design_intent"] = module_intent
        elif (module_intent or "LAB_ASSEMBLY") != (parent_intent or "LAB_ASSEMBLY"):
            warnings.append(warning(_W.DESIGN_INTENT_DIFFERS, f"the module is a {module_intent} design; the Project keeps {parent_intent or 'LAB_ASSEMBLY'}"))

    # ----------------------------------------------------------------- warnings
    for component in kept_components:
        if component.get("optical_role") == "source" and ("COMPONENT", component["instance_id"]) not in fixture_roles:
            warnings.append(warning(_W.UNMARKED_SOURCE_IMPORTED, f"source {component['instance_id']!r} of the module is not marked as a test source; "
                                    f"it is imported as {maps['COMPONENT'][component['instance_id']]!r} and illuminates the system",
                                    f"component {maps['COMPONENT'][component['instance_id']]}"))
    roles = {c.get("instance_id"): c.get("optical_role") for c in source.get("components", []) if isinstance(c, dict)}
    for observation in source.get("observations", []):
        oid = observation["observation_id"]
        device = observation.get("device_instance_id")
        if device in maps["COMPONENT"] and roles.get(device) in TERMINAL_DEVICE_ROLES:
            continue  # the sensing face of an imported detector or beam dump: a terminal plane by design
        if oid in maps["OBSERVATION"] and observation.get("absorbing") and ("OBSERVATION", oid) not in fixture_roles:
            warnings.append(warning(_W.ABSORBING_PLANE_IMPORTED, f"absorbing observation plane {oid!r} of the module is not marked as a test "
                                    f"observation; it is imported as {maps['OBSERVATION'][oid]!r} and stops the beams that reach it",
                                    f"observation {maps['OBSERVATION'][oid]}"))
    for target_id, source_id in sorted(rebound_targets.items()):
        warnings.append(warning(_W.TARGET_REBOUND, f"target {target_id!r} now reads {source_id!r}: its wavelength, power and coherence group are "
                                "that object's, not the module's test fixture's", f"target {target_id}"))
    for target_id in sorted(set(path_renamed)):
        warnings.append(warning(_W.PATH_SELECTOR_RENAMED, f"target {target_id!r} selects a beam path by id; its source part was renamed and its "
                                "branch part counts the interactions of the composed system: check it against a run", f"target {target_id}"))

    # ----------------------------------------------------------------- instance record
    id_map = []
    for kind in ("COMPONENT", "OBSERVATION", "TARGET", "MODEL", "MATERIAL", "COATING"):
        for sid, pid in maps[kind].items():
            id_map.append({"kind": kind, "source_id": sid, "parent_id": pid, "shared": False})
    outcomes = []
    for fkey, fixture in fixture_roles.items():
        action = actions[fkey]
        if action == "RETAINED" and fixture.object_id not in maps[fixture.object_kind]:
            action = "REMOVED"  # a retained test target that stays with the module (MODULE_ONLY) or was left out
        entry: dict[str, Any] = {"object_kind": fixture.object_kind, "source_id": fixture.object_id, "action": action}
        if action == "RETAINED":
            entry["parent_id"] = maps[fixture.object_kind][fixture.object_id]
        if action == "REBOUND":
            entry["rebound_to"] = rebound_to[fkey]
        entry["affected_targets"] = sorted(affected.get(fkey, []))
        outcomes.append(entry)
    module_digest = module.module_digest()
    module_hash = module.project.content_hash()
    module_validations = []
    for v in module.validations:
        if v.scope == "SYSTEM" and v.project_hash == module_hash:
            module_validations.append({**v.model_dump(mode="json", exclude_none=True), "scope": "MODULE", "instance_key": key,
                                       "module_digest": module_digest})
    title = descriptor.title if descriptor is not None else "Imported layout"
    instance: dict[str, Any] = {
        "instance_key": key,
        "source": {"module_id": descriptor.module_id if descriptor is not None else f"layout-{module_hash[:LAYOUT_ID_DIGITS]}",
                   "module_digest": module_digest, "title": title,
                   **({"revision": descriptor.revision} if descriptor is not None and descriptor.revision is not None else {}),
                   **({"label": label} if label is not None else {})},
        "placement": pose_dict(*placement_t),
        "id_map": id_map,
        "fixtures": outcomes,
        "member_digests": [],
        "interfaces": [port.model_dump(mode="json", exclude_none=True) for port in (descriptor.interfaces if descriptor is not None else ())],
        "module_validations": module_validations,
        # the module objects this import placed (written by member_digests as source digests)
        _SOURCES: {(kind, parent_id): source_digest(item) for kind, field_name, ident in OBJECTS if kind in SOURCED
                   for item in source.get(field_name, []) if (parent_id := maps[kind].get(item[ident])) is not None},
    }
    groups = []
    members = [{"kind": "COMPONENT", "id": maps["COMPONENT"][c["instance_id"]]} for c in kept_components]
    members += [{"kind": "OBSERVATION", "id": maps["OBSERVATION"][o["observation_id"]]} for o in source.get("observations", [])
                if o["observation_id"] in maps["OBSERVATION"]]
    groups.append({"group_id": group_id or key, "label": label or title, "members": members, "origin": "IMPORT", "instance_key": key,
                   "frame": pose_dict(*placement_t)})

    # ----------------------------------------------------------------- a composed module: its instances and groups
    nested_instances, nested_groups = _nested(module, key, placement_t, maps)
    _keep_lacked(project, nested_instances)
    instances = [instance, *nested_instances]
    groups += nested_groups
    member_digests(project, instances)

    interfaces = port_rows(descriptor.interfaces if descriptor is not None else (), placement_t,
                           lambda source: ref("COMPONENT", source) if source in maps["COMPONENT"] or ("COMPONENT", source) in rebound_to
                           else None)
    carried = sorted(maps["TARGET"][t] for t in maps["TARGET"])
    report = {
        "instance_key": key,
        "module_id": instance["source"]["module_id"],
        "module_digest": module_digest,
        "placement": instance["placement"],
        "group_id": groups[0]["group_id"],
        "id_map": id_map,
        "fixtures": outcomes,
        "requirements": {"carried": carried, "module_only": module_only, "removed": sorted(dropped_targets)},
        "interfaces": interfaces,
        "nested_instances": [i["instance_key"] for i in nested_instances],
    }
    return ImportOutcome(project, instances, groups, report, warnings, maps)


def member_digests(project: Mapping[str, Any], instances: list[dict[str, Any]]) -> None:
    """Fill each instance's ``member_digests``: osc1 of every member it copied (not a definition another instance owns,
    ``shared`` in workspaces written by earlier builds). A digest names its member's ``kind`` when members of different
    kinds have the same id, and the digest of the module object a component or observation plane was copied from when
    the import placed it (``MemberDigestV1``): given by the import (``import_module``), else kept from the record's
    current digests. A definition the record holds as lacked by its revision (``LACKED_MEMBER_DIGEST``, ``_keep_lacked``)
    stays so."""

    index = {}
    for kind, field_name, ident in (*OBJECTS, *LIBRARY):
        index[kind] = {item[ident]: item for item in project.get(field_name) or []}
    for instance in instances:
        sources = instance.pop(_SOURCES, None)
        if sources is None:
            sources = {ref: member["source_digest"] for ref, member in digest_refs(instance) if member.get("source_digest")}
        lacked = {ref for ref, member in digest_refs(instance) if member["digest"] == LACKED_MEMBER_DIGEST and ref[0] in _DEFINITIONS}
        own = [entry for entry in instance["id_map"] if not entry["shared"]]
        kinds: dict[str, set[str]] = {}
        for entry in own:
            kinds.setdefault(entry["parent_id"], set()).add(entry["kind"])
        digests, seen = [], set()
        for entry in own:
            ref = (entry["kind"], entry["parent_id"])
            item = index[entry["kind"]].get(entry["parent_id"])
            if item is None or ref in seen:
                continue
            seen.add(ref)
            row = {"parent_id": entry["parent_id"], "digest": LACKED_MEMBER_DIGEST if ref in lacked else osc1_digest(item)}
            if len(kinds[entry["parent_id"]]) > 1:
                row = {"parent_id": entry["parent_id"], "kind": entry["kind"], "digest": row["digest"]}
            if entry["kind"] in SOURCED and sources.get(ref) is not None:
                row["source_digest"] = sources[ref]
            digests.append(row)
        instance["member_digests"] = digests


def digest_refs(instance: Mapping[str, Any]) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """Each member digest of an instance payload with the ``(kind, parent_id)`` of the member it is of: its ``kind``,
    else the first id map entry with its id that is not ``shared`` (``LayoutInstanceV1.member_kind``)."""

    first: dict[str, str] = {}
    for entry in instance.get("id_map", []):
        if not entry.get("shared"):
            first.setdefault(entry["parent_id"], entry["kind"])
    for entry in instance.get("id_map", []):
        first.setdefault(entry["parent_id"], entry["kind"])
    return [((member.get("kind") or first.get(member["parent_id"], ""), member["parent_id"]), member)
            for member in instance.get("member_digests", [])]


def _nested(module: OpticalWorkspaceV1, key: str, placement_t: tuple[rot.Vector, rot.Quaternion],
            maps: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The instances and groups of a composed module, as instances / groups of the importing workspace. A nested
    instance carries the source digest of each component and observation plane that was unedited in the module (its
    content there is still the module object it was copied from, placed at the instance's placement)."""

    instances: list[dict[str, Any]] = []
    keys: dict[str, str] = {}
    problems: list[CompositionProblem] = []
    module_project = module.project.payload()
    module_index = {kind: {item[ident]: item for item in module_project.get(field_name) or []} for kind, field_name, ident in OBJECTS}
    for inner in module.instances:
        new_key = f"{key}_{inner.instance_key}"
        try:
            _INSTANCE_KEY.validate_python(new_key)
        except ValidationError:
            problems.append(CompositionProblem(_E.ID_TOO_LONG, f"the module's instance {inner.instance_key!r} would become {new_key!r} (instance keys "
                                               "have at most 24 characters); use a shorter instance_key", f"instance {inner.instance_key}",
                                               {"kind": "INSTANCE", "source_id": inner.instance_key, "parent_id": new_key, "limit": INSTANCE_KEY_MAX}))
            continue
        keys[inner.instance_key] = new_key
        id_map = []
        for entry in inner.id_map:
            pid = maps[entry.kind].get(entry.parent_id)
            if pid is not None:
                id_map.append({"kind": entry.kind, "source_id": entry.source_id, "parent_id": pid, "shared": entry.shared})
        fixtures = []
        for outcome in inner.fixtures:
            row = outcome.model_dump(mode="json", exclude_none=True)
            kind = outcome.object_kind
            if outcome.parent_id is not None:
                mapped = maps[kind].get(outcome.parent_id)
                if mapped is None:
                    row = {"object_kind": kind, "source_id": outcome.source_id, "action": "REMOVED"}
                else:
                    row["parent_id"] = mapped
            if outcome.rebound_to is not None:
                mapped = maps[kind].get(outcome.rebound_to)
                if mapped is None:
                    row = {"object_kind": kind, "source_id": outcome.source_id, "action": "REMOVED"}
                else:
                    row["rebound_to"] = mapped
            row["affected_targets"] = sorted(maps["TARGET"][t] for t in outcome.affected_targets if t in maps["TARGET"])
            fixtures.append(row)
        position, quaternion = pose_tuple(inner.placement.model_dump(mode="json"))
        new_position, new_quaternion = rot.compose_pose(placement_t[0], placement_t[1], position, quaternion)
        validations = [{**v.model_dump(mode="json", exclude_none=True), "instance_key": new_key} for v in inner.module_validations]
        parent = f"{key}_{inner.parent_instance}" if inner.parent_instance is not None else key
        digests = inner.member_digest_map()
        carried = {(kind, maps[kind][parent_id]): source for (kind, parent_id), source in inner.source_digest_map().items()
                   if parent_id in maps[kind] and parent_id in module_index[kind]
                   and osc1_digest(module_index[kind][parent_id]) == digests.get((kind, parent_id))}
        lacked = [(kind, maps[kind][parent_id]) for (kind, parent_id), digest in digests.items()
                  if digest == LACKED_MEMBER_DIGEST and kind in _DEFINITIONS and parent_id in maps[kind]]
        instances.append({"instance_key": new_key, "source": inner.source.model_dump(mode="json", exclude_none=True), "parent_instance": parent,
                          "placement": pose_dict(new_position, new_quaternion), "id_map": id_map, "fixtures": fixtures,
                          "member_digests": [], "interfaces": [p.model_dump(mode="json", exclude_none=True) for p in inner.interfaces],
                          "module_validations": validations, _SOURCES: carried, _LACKED: lacked})
    groups = []
    for group in module.groups:
        members = []
        for member in group.members:
            mapped = maps[member.kind].get(member.id)
            if mapped is not None:
                members.append({"kind": member.kind, "id": mapped})
        row: dict[str, Any] = {"group_id": f"{key}_{group.group_id}", "members": members}
        if group.label is not None:
            row["label"] = group.label
        if group.origin == "IMPORT" and group.instance_key in keys:
            row.update(origin="IMPORT", instance_key=keys[group.instance_key])
        else:
            row["origin"] = "USER"
        if group.frame is not None:
            position, quaternion = pose_tuple(group.frame.model_dump(mode="json"))
            row["frame"] = pose_dict(*rot.compose_pose(placement_t[0], placement_t[1], position, quaternion))
        try:
            TypeAdapter(OpenSciId).validate_python(row["group_id"])
        except ValidationError:
            problems.append(CompositionProblem(_E.ID_TOO_LONG, f"group id {row['group_id']!r} is longer than 128 characters", f"group {group.group_id}"))
            continue
        groups.append(row)
    if problems:
        raise CompositionError(problems)
    return instances, groups


def _keep_lacked(project: dict[str, Any], nested: list[dict[str, Any]]) -> None:
    """A definition that a nested instance of the module holds as lacked by its revision (``LACKED_MEMBER_DIGEST``: kept
    there for objects outside it) and that objects outside it use in the Project stays recorded so: the instance keeps
    it for them as in the module, so replacing it by its own revision changes nothing, and once nothing else uses it it
    is a member added to it. Any other definition of a nested instance is part of its import, like the other content
    the module's author left in it (``project``: the Project with the module's objects copied in)."""

    from .operations import _remove_scope

    children: dict[str, list[str]] = {}
    for record in nested:
        children.setdefault(record["parent_instance"], []).append(record["instance_key"])
    records = {record["instance_key"]: record for record in nested}
    for record in nested:
        lacked = record.pop(_LACKED, [])
        if not lacked:
            continue
        keys, pending = {record["instance_key"]}, [record["instance_key"]]
        while pending:
            for child in children.get(pending.pop(), []):
                if child not in keys:
                    keys.add(child)
                    pending.append(child)
        scope = {(entry["kind"], entry["parent_id"]) for k in keys for entry in records[k]["id_map"] if not entry["shared"]}
        kept = _remove_scope(dict(project), scope)
        record["member_digests"] = [{"parent_id": parent_id, "kind": kind, "digest": LACKED_MEMBER_DIGEST} for kind, parent_id in lacked
                                    if (kind, parent_id) in kept]


__all__ = ["FixturePolicy", "ImportOutcome", "digest_refs", "import_module", "instance_ports", "member_digests", "namespaced", "parse_fixture_policy",
           "placed", "port_rows", "source_digest", "valid_id", "without_pin"]
