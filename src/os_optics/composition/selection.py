"""Selectors: which components and observation planes an operation acts on.

A selector is a JSON object combining any of:

* ``ids``: object ids -- a string (a component or an observation plane; when an id names both, give its kind) or
  ``{"kind": "COMPONENT" | "OBSERVATION", "id": ...}``;
* ``groups``: editing group ids (their current members);
* ``instances``: instance keys (every component and observation plane the import copied that is still in the Project);
* ``kinds``: keep only ``COMPONENT`` and / or ``OBSERVATION`` objects.

The result is the union, each object once, sorted by id. An observation plane attached to a component
(``device_instance_id``) follows that component when the component is selected too (``follows_device``); selected
without it, it is moved on its own (its pose in the component's frame changes).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._vendor.shared_contracts.optical_design import OpticalWorkspaceV1
from .errors import CompositionError, CompositionErrorCode, CompositionProblem, refuse

_E = CompositionErrorCode
SELECTOR_KEYS = ("ids", "groups", "instances", "kinds")
KINDS = ("COMPONENT", "OBSERVATION")


@dataclass(frozen=True)
class Selection:
    """The resolved selection: component ids, observation ids (free and attached), and for each attached observation its
    device and whether that device is selected too."""

    components: tuple[str, ...]
    observations: tuple[str, ...]
    attached: tuple[tuple[str, str, bool], ...]

    @property
    def count(self) -> int:
        return len(self.components) + len(self.observations)

    def refs(self) -> list[dict[str, str]]:
        return [{"kind": "COMPONENT", "id": i} for i in self.components] + [{"kind": "OBSERVATION", "id": i} for i in self.observations]

    def as_dict(self) -> dict[str, Any]:
        attached = {oid for oid, _, _ in self.attached}
        return {
            "components": list(self.components),
            "observations": [o for o in self.observations if o not in attached],
            "attached_observations": [{"observation_id": oid, "device_instance_id": device, "follows_device": follows}
                                      for oid, device, follows in self.attached],
            "count": self.count,
        }


def instance_members(workspace: OpticalWorkspaceV1, instance_key: str) -> tuple[set[str], set[str]]:
    """(component ids, observation ids) an instance copied that are still in the Project."""

    instance = workspace.instance(instance_key)
    components = {c.instance_id for c in workspace.project.components}
    observations = {o.observation_id for o in workspace.project.observations}
    if instance is None:
        return set(), set()
    return ({e.parent_id for e in instance.id_map if e.kind == "COMPONENT" and e.parent_id in components},
            {e.parent_id for e in instance.id_map if e.kind == "OBSERVATION" and e.parent_id in observations})


def resolve_selection(workspace: OpticalWorkspaceV1, selector: Any, *, allow_empty: bool = True) -> Selection:
    """Resolve ``selector`` (module docstring) against ``workspace``; unknown ids, groups or instances are refused
    (``SELECTION_INVALID``, every one listed), an id naming both a component and an observation is refused
    (``SELECTION_AMBIGUOUS``) and, unless ``allow_empty``, an empty result is refused (``SELECTION_EMPTY``)."""

    if not isinstance(selector, Mapping):
        raise refuse(_E.REQUEST_INVALID, "a selector is an object {ids, groups, instances, kinds}", "selector")
    unknown = sorted(set(selector) - set(SELECTOR_KEYS))
    if unknown:
        raise refuse(_E.REQUEST_INVALID, f"a selector takes {list(SELECTOR_KEYS)}, not {unknown}", "selector")
    problems: list[CompositionProblem] = []
    components = {c.instance_id: c for c in workspace.project.components}
    observations = {o.observation_id: o for o in workspace.project.observations}
    chosen_c: set[str] = set()
    chosen_o: set[str] = set()

    def as_list(name: str) -> list[Any]:
        value = selector.get(name)
        if value is None:
            return []
        if not isinstance(value, list):
            problems.append(CompositionProblem(_E.REQUEST_INVALID, f"selector.{name} is a list", f"selector.{name}"))
            return []
        return value

    for item in as_list("ids"):
        if isinstance(item, Mapping):
            kind, object_id = item.get("kind"), item.get("id")
            if kind not in KINDS or not isinstance(object_id, str):
                problems.append(CompositionProblem(_E.REQUEST_INVALID, f"an id entry is a string or {{kind: {list(KINDS)}, id}}, not {dict(item)!r}",
                                                   "selector.ids"))
                continue
            known = components if kind == "COMPONENT" else observations
            if object_id not in known:
                problems.append(CompositionProblem(_E.SELECTION_INVALID, f"the Project has no {kind.lower()} {object_id!r}", "selector.ids",
                                                   {"kind": kind, "id": object_id}))
                continue
            (chosen_c if kind == "COMPONENT" else chosen_o).add(object_id)
        elif isinstance(item, str):
            in_c, in_o = item in components, item in observations
            if in_c and in_o:
                problems.append(CompositionProblem(_E.SELECTION_AMBIGUOUS, f"{item!r} is both a component and an observation plane; give "
                                                   "{kind, id}", "selector.ids", {"id": item}))
            elif in_c:
                chosen_c.add(item)
            elif in_o:
                chosen_o.add(item)
            else:
                problems.append(CompositionProblem(_E.SELECTION_INVALID, f"the Project has no component or observation plane {item!r}",
                                                   "selector.ids", {"id": item}))
        else:
            problems.append(CompositionProblem(_E.REQUEST_INVALID, "an id entry is a string or {kind, id}", "selector.ids"))
    for group_id in as_list("groups"):
        group = workspace.group(group_id) if isinstance(group_id, str) else None
        if group is None:
            problems.append(CompositionProblem(_E.SELECTION_INVALID, f"the workspace has no group {group_id!r}; groups: "
                                               f"{sorted(g.group_id for g in workspace.groups)}", "selector.groups", {"group_id": group_id}))
            continue
        for member in group.members:
            (chosen_c if member.kind == "COMPONENT" else chosen_o).add(member.id)
    for key in as_list("instances"):
        if not isinstance(key, str) or workspace.instance(key) is None:
            problems.append(CompositionProblem(_E.SELECTION_INVALID, f"the workspace has no instance {key!r}; instances: "
                                               f"{sorted(i.instance_key for i in workspace.instances)}", "selector.instances", {"instance_key": key}))
            continue
        members_c, members_o = instance_members(workspace, key)
        chosen_c |= members_c
        chosen_o |= members_o
    kinds = selector.get("kinds")
    if kinds is not None:
        if not isinstance(kinds, list) or not kinds or any(k not in KINDS for k in kinds):
            problems.append(CompositionProblem(_E.REQUEST_INVALID, f"selector.kinds is a non-empty list of {list(KINDS)}", "selector.kinds"))
        else:
            if "COMPONENT" not in kinds:
                chosen_c = set()
            if "OBSERVATION" not in kinds:
                chosen_o = set()
    if problems:
        raise CompositionError(problems)
    attached = tuple(sorted((oid, observations[oid].device_instance_id, observations[oid].device_instance_id in chosen_c)
                            for oid in chosen_o if observations[oid].device_instance_id is not None))
    selection = Selection(tuple(sorted(chosen_c)), tuple(sorted(chosen_o)), attached)  # type: ignore[arg-type]
    if not allow_empty and selection.count == 0:
        raise refuse(_E.SELECTION_EMPTY, "the selector selects no component or observation plane", "selector")
    return selection


__all__ = ["Selection", "instance_members", "resolve_selection"]
