"""Composition (review): a member deleted from the parent Project is an edit of its instance.

The deletion stays in the instance's record (id map entry, member digest, fixture outcome), so the status reports it,
``replace_instance`` refuses it unless ``discard_edits`` (then ``edits_discarded`` lists it and the new revision brings
the member back), an object of the same kind and id added back is the member again, and no import takes its id. An
instance reported unedited is exactly what replacing it with the same revision gives (no silent overwrite), also after
whole-instance moves."""

from __future__ import annotations

import copy
import json
import random

import pytest
from composition_helpers import FOLD_DESCRIPTOR, fold_module, ok, screen, system_with_laser
from test_composition_isolation_and_retries import REBIND, codes, component, fold_mod, group, two_instances

from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def without(project: dict, kind: str, *ids: str) -> dict:
    """``project`` without the objects ``ids`` of ``kind`` (``components`` / ``observations`` / ``targets``)."""

    ident = {"components": "instance_id", "observations": "observation_id", "targets": "target_id"}[kind]
    out = copy.deepcopy(project)
    out[kind] = [item for item in out[kind] if item[ident] not in ids]
    return out


def instance(ws: dict, key: str) -> dict:
    return next(i for i in ws["instances"] if i["instance_key"] == key)


def changes(ws: dict) -> dict[str, list[tuple[str, str, str]]]:
    return {i["instance_key"]: [(e["kind"], e["id"], e["change"]) for e in i["edited"]] for i in ok(J.workspace_status(ws))["status"]["instances"]}


def ids(ws: dict, kind: str) -> list[str]:
    ident = {"components": "instance_id", "observations": "observation_id", "targets": "target_id"}[kind]
    return [item[ident] for item in ws["project"][kind]]


# ------------------------------------------------------------------------------------------------ the review's case


def test_a_deleted_member_is_an_edit_and_replacing_its_instance_needs_discard_edits():
    """review360_repro.py, case deleted_member: two fold instances, m1_M1 deleted from the parent Project, the workspace
    updated, m1 replaced with revision r2 without discard_edits."""

    ws = two_instances()
    reply = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))
    changed = reply["workspace"]
    # the update reports the deletion; the instance keeps the member's record
    assert [(e["instance_key"], e["kind"], e["parent_id"]) for e in reply["result"]["removed_instance_members"]] == [("m1", "COMPONENT", "m1_M1")]
    assert [w["code"] for w in reply["result"]["warnings"]] == ["INSTANCE_MEMBERS_REMOVED"]
    assert reply["result"]["restored_instance_members"] == []
    assert {(g["group_id"], g["id"]) for g in reply["result"]["removed_group_members"]} == {("m1", "m1_M1")}
    assert instance(changed, "m1")["id_map"] == instance(ws, "m1")["id_map"]
    assert instance(changed, "m1")["member_digests"] == instance(ws, "m1")["member_digests"]
    assert instance(changed, "m2") == instance(ws, "m2")
    # status: the deletion is m1's edit, m2 is unedited
    assert changes(changed) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")], "m2": []}
    status = {i["instance_key"]: i for i in ok(J.workspace_status(changed))["status"]["instances"]}
    assert status["m1"]["components"] == [] and status["m2"]["components"] == ["m2_M1"]
    # replacing m1 without discard_edits: the same structured refusal as any other local edit; nothing changes
    revised = fold_mod("0.95", "r2")
    refused = J.workspace_replace_instance(changed, revised, {"instance_key": "m1", "fixtures": REBIND})
    assert codes(refused) == ["INSTANCE_EDITED"]
    problem = refused["error"]["problems"][0]
    assert problem["details"]["edited"] == [{"kind": "COMPONENT", "id": "m1_M1", "change": "REMOVED"}]
    assert "1 member deleted" in problem["message"]
    assert refused["workspace_digest"] == reply["workspace_digest"]
    # with discard_edits the deletion is discarded: the member is back, and the result is what replacing the
    # undeleted instance gives (nothing else changes)
    discarded = ok(J.workspace_replace_instance(changed, revised, {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))
    assert discarded["result"]["replaced"]["edits_discarded"] == [{"kind": "COMPONENT", "id": "m1_M1", "change": "REMOVED"}]
    reference = ok(J.workspace_replace_instance(ws, revised, {"instance_key": "m1", "fixtures": REBIND}))
    for name in ("project", "instances", "groups"):
        assert discarded["workspace"][name] == reference["workspace"][name], name
    assert discarded["workspace_digest"] == reference["workspace_digest"]
    assert "m1_M1" in ids(discarded["workspace"], "components") and changes(discarded["workspace"]) == {"m1": [], "m2": []}
    assert {"kind": "COMPONENT", "id": "m1_M1"} in group(discarded["workspace"], "m1")["members"]


def test_a_deletion_stays_an_edit_through_later_operations_and_is_reported_once():
    ws = two_instances()
    changed = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))["workspace"]
    # another Project edit: the earlier deletion is not reported again
    project = copy.deepcopy(changed["project"])
    component({"project": project}, "m2_M1")["nominal_pose"]["position_mm"][2] = 201.0
    later = ok(J.workspace_update_project(changed, {"project": project}))
    assert later["result"]["removed_instance_members"] == [] and later["result"]["warnings"] == []
    # moving what is left of m1 as a whole is not an edit; the deletion still is
    moved = ok(J.workspace_transform(later["workspace"], {"selector": {"instances": ["m1"]}, "translation_mm": [0.0, 0.0, -30.0]}))
    assert moved["result"]["placements_updated"] == ["m1"]
    assert changes(moved["workspace"]) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")], "m2": [("COMPONENT", "m2_M1", "EDITED")]}
    # a batch that deletes and replaces in one step is refused the same way
    batch = J.workspace_batch(ws, {"operations": [
        {"kind": "update_project", "request": {"project": without(ws["project"], "components", "m1_M1")}},
        {"kind": "replace_instance", "request": {"instance_key": "m1", "fixtures": REBIND}, "module": fold_mod("0.95", "r2")}]})
    assert codes(batch) == ["INSTANCE_EDITED"]
    # saved and reopened: the record is part of the workspace
    reopened = ok(J.workspace_init(json.loads(json.dumps(changed))))
    assert reopened["workspace_digest"] == ok(J.workspace_status(changed))["status"]["workspace_digest"]
    assert changes(reopened["workspace"])["m1"] == [("COMPONENT", "m1_M1", "REMOVED")]


# ------------------------------------------------------------------------------------------------ edge cases


def attached_module() -> dict:
    """The fold module with an observation plane attached to its mirror (ATT, 5 mm along the mirror's local +Z)."""

    project = fold_module()
    project["observations"].append(screen("ATT", (0.0, 0.0, 5.0), (1.0, 0.0, 0.0, 0.0), device="M1"))
    module = ok(J.workspace_init(project, {"module": FOLD_DESCRIPTOR}))["workspace"]
    return ok(J.workspace_export_module(module, {}))["module"]


def test_deleting_an_attached_observation_or_a_target_of_an_instance_is_an_edit():
    module = attached_module()
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    assert {"kind": "OBSERVATION", "id": "a_ATT"} in group(ws, "a")["members"]
    # the attached plane alone
    plane = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "observations", "a_ATT")}))["workspace"]
    assert changes(plane) == {"a": [("OBSERVATION", "a_ATT", "REMOVED")]}
    assert {"kind": "OBSERVATION", "id": "a_ATT"} not in group(plane, "a")["members"]
    # the carried target alone: no longer listed among the instance's targets, reported as deleted
    target = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "targets", "a_T_MON_CENTER")}))["workspace"]
    assert changes(target) == {"a": [("TARGET", "a_T_MON_CENTER", "REMOVED")]}
    status = ok(J.workspace_status(target))["status"]["instances"][0]
    assert status["targets"] == [] and status["observations"] == ["a_ATT", "a_MON"]
    # the device with its attached plane
    both = ok(J.workspace_update_project(ws, {"project": without(without(ws["project"], "observations", "a_ATT"), "components", "a_M1")}))["workspace"]
    assert changes(both) == {"a": [("COMPONENT", "a_M1", "REMOVED"), ("OBSERVATION", "a_ATT", "REMOVED")]}
    for edited_ws, expected in ((plane, ["a_ATT"]), (target, ["a_T_MON_CENTER"]), (both, ["a_M1", "a_ATT"])):
        assert codes(J.workspace_replace_instance(edited_ws, module, {"instance_key": "a", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
        reply = ok(J.workspace_replace_instance(edited_ws, module, {"instance_key": "a", "fixtures": REBIND, "discard_edits": True}))
        assert [e["id"] for e in reply["result"]["replaced"]["edits_discarded"]] == expected
        for name in ("project", "instances", "groups"):  # the instance is exactly as imported again
            assert reply["workspace"][name] == ws[name], name


def test_a_deleted_retained_fixture_keeps_its_fixture_outcome():
    retain = {"policy": "retain_all"}
    placement = {"position_mm": [0.0, 0.0, 50.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "r", "fixtures": retain, "placement": placement}))["workspace"]
    outcomes = instance(ws, "r")["fixtures"]
    assert {(f["source_id"], f["action"], f.get("parent_id")) for f in outcomes} == {("SRC", "RETAINED", "r_SRC"), ("OUT", "RETAINED", "r_OUT"),
                                                                                       ("T_OUT", "RETAINED", "r_T_OUT")}
    deleted = ok(J.workspace_update_project(ws, {"project": without(without(ws["project"], "targets", "r_T_OUT"), "observations", "r_OUT")}))["workspace"]
    assert instance(deleted, "r")["fixtures"] == outcomes  # what the import did with the fixtures stays recorded
    assert changes(deleted) == {"r": [("OBSERVATION", "r_OUT", "REMOVED"), ("TARGET", "r_T_OUT", "REMOVED")]}
    assert codes(J.workspace_replace_instance(deleted, fold_mod(), {"instance_key": "r", "fixtures": retain})) == ["INSTANCE_EDITED"]
    reply = ok(J.workspace_replace_instance(deleted, fold_mod(), {"instance_key": "r", "fixtures": retain, "discard_edits": True}))
    for name in ("project", "instances", "groups"):
        assert reply["workspace"][name] == ws[name], name


def test_deleting_then_re_adding_an_object_with_the_same_id():
    ws = two_instances()
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))["workspace"]
    # added back unchanged (an undo): the member again, unedited, back in its import group at its place
    back = ok(J.workspace_update_project(deleted, {"project": ws["project"]}))
    assert [(e["instance_key"], e["parent_id"]) for e in back["result"]["restored_instance_members"]] == [("m1", "m1_M1")]
    assert back["result"]["removed_instance_members"] == [] and back["result"]["warnings"] == []
    assert changes(back["workspace"]) == {"m1": [], "m2": []}
    assert back["workspace_digest"] == ok(J.workspace_status(ws))["status"]["workspace_digest"]
    for name in ("project", "instances", "groups"):
        assert back["workspace"][name] == ws[name], name
    assert ok(J.workspace_replace_instance(back["workspace"], fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND}))
    # added back with other content: the member again, edited
    project = copy.deepcopy(ws["project"])
    component({"project": project}, "m1_M1")["nominal_pose"]["position_mm"][2] = 0.5
    other = ok(J.workspace_update_project(deleted, {"project": project}))["workspace"]
    assert changes(other) == {"m1": [("COMPONENT", "m1_M1", "EDITED")], "m2": []}
    assert codes(J.workspace_replace_instance(other, fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    replaced = ok(J.workspace_replace_instance(other, fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))
    assert replaced["result"]["replaced"]["edits_discarded"] == [{"kind": "COMPONENT", "id": "m1_M1", "change": "EDITED"}]
    assert component(replaced["workspace"], "m1_M1")["nominal_pose"] == component(ws, "m1_M1")["nominal_pose"]
    # an object of another kind under the same id is not the member
    plane = without(ws["project"], "components", "m1_M1")
    plane["observations"].append(screen("m1_M1", (0.0, -40.0, 0.0)))
    unrelated = ok(J.workspace_update_project(deleted, {"project": plane}))
    assert unrelated["result"]["restored_instance_members"] == []
    assert changes(unrelated["workspace"]) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")], "m2": []}


def test_deleting_a_member_that_is_in_a_user_group():
    ws = ok(J.workspace_group_create(two_instances(), {"group_id": "mirrors", "members": {"ids": ["m1_M1", "m2_M1"]}}))["workspace"]
    reply = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))
    deleted = reply["workspace"]
    assert {(g["group_id"], g["id"]) for g in reply["result"]["removed_group_members"]} == {("m1", "m1_M1"), ("mirrors", "m1_M1")}
    assert group(deleted, "mirrors")["members"] == [{"kind": "COMPONENT", "id": "m2_M1"}]
    assert changes(deleted) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")], "m2": []}
    assert codes(J.workspace_replace_instance(deleted, fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    # discarded: the member is back in its instance's import group; the user group stays as the user left it
    replaced = ok(J.workspace_replace_instance(deleted, fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))
    assert group(replaced["workspace"], "mirrors")["members"] == [{"kind": "COMPONENT", "id": "m2_M1"}]
    assert group(replaced["workspace"], "m1")["members"] == group(ws, "m1")["members"]
    assert replaced["result"]["replaced"]["dropped_group_members"] == []
    # added back: likewise
    back = ok(J.workspace_update_project(deleted, {"project": ws["project"]}))["workspace"]
    assert group(back, "m1")["members"] == group(ws, "m1")["members"]
    assert group(back, "mirrors")["members"] == [{"kind": "COMPONENT", "id": "m2_M1"}]


def test_a_catalog_re_pin_is_not_an_edit_next_to_a_deletion():
    from composition_helpers import laser, project_of
    from local_helpers import FLAT_MIRROR, FOLD_X_TO_Y, catalog_document, element

    newer = catalog_document()
    newer["catalog_version"] = "2026-02-01.1"
    assert J.install_catalog(json.dumps(catalog_document()).encode("utf-8"))["ok"] is True
    assert J.install_catalog(json.dumps(newer).encode("utf-8"))["ok"] is True
    module = project_of([element("M", "mirror", FLAT_MIRROR, (0, 0, 0), FOLD_X_TO_Y, catalog_version="2026-02-01.1")], [], [], [])
    parent = ok(J.workspace_init(project_of([laser("LASER")], [], [], [])))["workspace"]
    ws = ok(J.workspace_import(parent, module, {"instance_key": "k"}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "m1", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 0.0, 200.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    pinned = copy.deepcopy(ws["project"])
    component({"project": pinned}, "k_M")["product"]["catalog_version"] = "2026-01-01.1"  # what the Designer's re-pin writes
    only_pin = ok(J.workspace_update_project(ws, {"project": pinned}))
    assert changes(only_pin["workspace"]) == {"k": [], "m1": []}
    assert only_pin["result"]["removed_instance_members"] == [] and only_pin["result"]["restored_instance_members"] == []
    with_deletion = ok(J.workspace_update_project(ws, {"project": without(pinned, "components", "m1_M1")}))["workspace"]
    assert changes(with_deletion) == {"k": [], "m1": [("COMPONENT", "m1_M1", "REMOVED")]}


def test_a_deleted_member_s_id_is_kept_for_its_instance():
    # instance "a" of a module whose mirror is called "b_M1" becomes a_b_M1, the id a second instance "a_b" of the
    # plain fold module would give its mirror
    renamed = fold_module()
    component({"project": renamed}, "M1")["instance_id"] = "b_M1"
    module = ok(J.workspace_export_module(ok(J.workspace_init(renamed, {"module": FOLD_DESCRIPTOR}))["workspace"], {}))["module"]
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "a_b_M1")}))["workspace"]
    reply = J.workspace_import(deleted, fold_mod(), {"instance_key": "a_b", "fixtures": REBIND,
                                                     "placement": {"position_mm": [0.0, 0.0, 300.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}})
    assert codes(reply) == ["ID_CONFLICT"]
    assert reply["error"]["problems"][0]["details"] == {"kind": "COMPONENT", "parent_id": "a_b_M1", "deleted_from": "a"}
    assert reply["workspace_digest"] == ok(J.workspace_status(deleted))["status"]["workspace_digest"]


def test_replacing_a_nested_instance_restores_the_member_its_enclosing_instance_records_as_deleted():
    from test_composition_isolation_and_retries import composed_module

    module = composed_module()
    host = system_with_laser((-100.0, 0.0, 300.0))
    ws = ok(J.workspace_import(host, module, {"instance_key": "c", "fixtures": {"rebind": {"LASER": "LASER"}},
                                              "placement": {"position_mm": [0.0, 0.0, 300.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "c_a_M1")}))["workspace"]
    assert changes(deleted) == {"c": [("COMPONENT", "c_a_M1", "REMOVED")], "c_a": [("COMPONENT", "c_a_M1", "REMOVED")], "c_b": []}
    assert codes(J.workspace_replace_instance(deleted, fold_mod(), {"instance_key": "c_a", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    alone = ok(J.workspace_replace_instance(deleted, fold_mod(), {"instance_key": "c_a", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    assert changes(alone) == {"c": [], "c_a": [], "c_b": []}
    assert {"kind": "COMPONENT", "id": "c_a_M1"} in group(alone, "c")["members"] and {"kind": "COMPONENT", "id": "c_a_M1"} in group(alone, "c_a")["members"]
    assert alone["project"] == ws["project"]
    # the enclosing instance replaced as a whole: everything as imported
    whole = ok(J.workspace_replace_instance(deleted, module, {"instance_key": "c", "fixtures": {"rebind": {"LASER": "LASER"}}, "discard_edits": True}))
    assert [e["id"] for e in whole["result"]["replaced"]["edits_discarded"]] == ["c_a_M1"]
    for name in ("project", "instances", "groups"):
        assert whole["workspace"][name] == ws[name], name


def test_a_composed_module_exported_after_a_deletion_imports_without_it():
    ws = two_instances()
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))["workspace"]
    composed = ok(J.workspace_export_module(deleted, {"module_id": "folds", "title": "Folds", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    assert [(e["kind"], e["parent_id"]) for e in instance(composed, "m1")["id_map"] if e["parent_id"] == "m1_M1"] == [("COMPONENT", "m1_M1")]
    host = system_with_laser((-100.0, 0.0, 600.0))
    imported = ok(J.workspace_import(host, composed, {"instance_key": "s", "fixtures": {"rebind": {"LASER": "LASER"}},
                                                      "placement": {"position_mm": [0.0, 0.0, 600.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))
    # the module is what it is: its import holds no mirror for m1 and no deletion to discard
    assert "s_m1_M1" not in ids(imported["workspace"], "components")
    assert changes(imported["workspace"]) == {"s": [], "s_m1": [], "s_m2": []}


def test_a_member_added_back_after_its_instance_moved_as_a_whole_is_an_edit():
    """Deleted, the rest of the instance moved, then added back: its content at import is known only before the move,
    so the member added back is an edit whatever its pose; replace_instance never moves it silently."""

    ws = two_instances()
    mirror = copy.deepcopy(component(ws, "m1_M1"))
    target = copy.deepcopy(next(t for t in ws["project"]["targets"] if t["target_id"] == "m1_T_MON_CENTER"))
    deleted = ok(J.workspace_update_project(ws, {"project": without(without(ws["project"], "components", "m1_M1"), "targets", "m1_T_MON_CENTER")}))
    deleted = deleted["workspace"]
    moved = ok(J.workspace_transform(deleted, {"selector": {"instances": ["m1"]}, "translation_mm": [0.0, 0.0, -30.0]}))
    assert moved["result"]["placements_updated"] == ["m1"]
    moved = moved["workspace"]
    before = {d["parent_id"]: d["digest"] for d in instance(deleted, "m1")["member_digests"]}
    after = {d["parent_id"]: d["digest"] for d in instance(moved, "m1")["member_digests"]}
    assert after["m1_M1"] != before["m1_M1"] and after.keys() == before.keys()
    assert after["m1_T_MON_CENTER"] == before["m1_T_MON_CENTER"]  # a move changes no target
    assert changes(moved) == {"m1": [("COMPONENT", "m1_M1", "REMOVED"), ("TARGET", "m1_T_MON_CENTER", "REMOVED")], "m2": []}
    for z in (0.0, -30.0):  # its pose at the import, and where the move would have put it
        project = copy.deepcopy(moved["project"])
        back = copy.deepcopy(mirror)
        back["nominal_pose"]["position_mm"][2] = z
        project["components"].insert(1, back)
        project["targets"].append(copy.deepcopy(target))
        reply = ok(J.workspace_update_project(moved, {"project": project}))
        assert sorted((e["kind"], e["parent_id"]) for e in reply["result"]["restored_instance_members"]) == [("COMPONENT", "m1_M1"),
                                                                                                            ("TARGET", "m1_T_MON_CENTER")]
        assert changes(reply["workspace"]) == {"m1": [("COMPONENT", "m1_M1", "EDITED")], "m2": []}
        refused = J.workspace_replace_instance(reply["workspace"], fold_mod(), {"instance_key": "m1", "fixtures": REBIND})
        assert codes(refused) == ["INSTANCE_EDITED"]
        assert refused["error"]["problems"][0]["details"]["edited"] == [{"kind": "COMPONENT", "id": "m1_M1", "change": "EDITED"}]
        discarded = ok(J.workspace_replace_instance(reply["workspace"], fold_mod(), {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))
        assert component(discarded["workspace"], "m1_M1")["nominal_pose"]["position_mm"] == [0.0, 0.0, -30.0]
        assert changes(discarded["workspace"]) == {"m1": [], "m2": []}
    # deleted again after being added back: a deletion again
    project = copy.deepcopy(moved["project"])
    project["components"].insert(1, copy.deepcopy(mirror))
    readded = ok(J.workspace_update_project(moved, {"project": project}))["workspace"]
    again = ok(J.workspace_update_project(readded, {"project": moved["project"]}))["workspace"]
    assert changes(again) == {"m1": [("COMPONENT", "m1_M1", "REMOVED"), ("TARGET", "m1_T_MON_CENTER", "REMOVED")], "m2": []}
    # an instance moved with nothing deleted keeps following its members' content (not an edit)
    plain = ok(J.workspace_transform(ws, {"selector": {"instances": ["m1"]}, "translation_mm": [0.0, 0.0, -30.0]}))["workspace"]
    assert changes(plain) == {"m1": [], "m2": []}


def test_the_status_names_no_deleted_component_as_a_port_s_beam_reference():
    retain = {"policy": "retain_all"}
    placement = {"position_mm": [0.0, 0.0, 50.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "r", "fixtures": retain, "placement": placement}))["workspace"]

    def references(workspace: dict) -> dict[str, str | None]:
        ports = ok(J.workspace_status(workspace))["status"]["instances"][0]["interfaces"]
        return {p["port_id"]: p["beam_reference"] for p in ports if "beam_reference" in p}

    assert references(ws) == {"IN": "r_SRC"}
    deleted = ok(J.workspace_update_project(ws, {"project": without(without(ws["project"], "targets", "r_T_OUT"), "components", "r_SRC")}))["workspace"]
    assert references(deleted) == {"IN": None}
    assert changes(deleted) == {"r": [("COMPONENT", "r_SRC", "REMOVED"), ("TARGET", "r_T_OUT", "REMOVED")]}
    # added back, or restored by replacing the instance: named again
    assert references(ok(J.workspace_update_project(deleted, {"project": ws["project"]}))["workspace"]) == {"IN": "r_SRC"}
    replaced = ok(J.workspace_replace_instance(deleted, fold_mod(), {"instance_key": "r", "fixtures": retain, "discard_edits": True}))["workspace"]
    assert references(replaced) == {"IN": "r_SRC"}
    # a source rebound to the system's own laser: named while the laser is there
    from composition_helpers import laser

    rebound = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "b", "fixtures": REBIND}))["workspace"]
    assert references(rebound) == {"IN": "LASER"}
    other_laser = without(rebound["project"], "components", "LASER")
    other_laser["components"].insert(0, laser("LASER_2"))
    assert references(ok(J.workspace_update_project(rebound, {"project": other_laser}))["workspace"]) == {"IN": None}


def same_id_module() -> dict:
    """The fold module with an observation plane that has its mirror's id (M1)."""

    project = fold_module()
    project["observations"].append(screen("M1", (0.0, 30.0, 0.0), radius_mm=10.0))
    return ok(J.workspace_export_module(ok(J.workspace_init(project, {"module": FOLD_DESCRIPTOR}))["workspace"], {}))["module"]


def test_members_of_different_kinds_with_the_same_id_are_told_apart():
    module = same_id_module()
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    digests = instance(ws, "m1")["member_digests"]
    # a digest names its kind only where two members share the id
    assert [(d["parent_id"], d.get("kind")) for d in digests if d["parent_id"] == "m1_M1"] == [("m1_M1", "COMPONENT"), ("m1_M1", "OBSERVATION")]
    assert all("kind" not in d for d in digests if d["parent_id"] != "m1_M1")
    assert all("kind" not in d for i in two_instances()["instances"] for d in i["member_digests"])
    assert changes(ws) == {"m1": []}  # unedited right after the import
    # each one deleted on its own is reported with its kind
    no_mirror = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "components", "m1_M1")}))["workspace"]
    assert changes(no_mirror) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")]}
    no_plane = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "observations", "m1_M1")}))["workspace"]
    assert changes(no_plane) == {"m1": [("OBSERVATION", "m1_M1", "REMOVED")]}
    assert {"kind": "OBSERVATION", "id": "m1_M1"} not in group(no_plane, "m1")["members"]
    assert {"kind": "COMPONENT", "id": "m1_M1"} in group(no_plane, "m1")["members"]
    # an edit of the plane is the plane's
    project = copy.deepcopy(ws["project"])
    next(o for o in project["observations"] if o["observation_id"] == "m1_M1")["aperture"]["radius_mm"] = 11.0
    assert changes(ok(J.workspace_update_project(ws, {"project": project}))["workspace"]) == {"m1": [("OBSERVATION", "m1_M1", "EDITED")]}
    # refused, then discarded: exactly as imported
    for edited_ws, expected in ((no_mirror, ("COMPONENT", "REMOVED")), (no_plane, ("OBSERVATION", "REMOVED"))):
        refused = J.workspace_replace_instance(edited_ws, module, {"instance_key": "m1", "fixtures": REBIND})
        assert codes(refused) == ["INSTANCE_EDITED"]
        assert [(e["kind"], e["change"]) for e in refused["error"]["problems"][0]["details"]["edited"]] == [expected]
        reply = ok(J.workspace_replace_instance(edited_ws, module, {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))
        for name in ("project", "instances", "groups"):
            assert reply["workspace"][name] == ws[name], name
    # the plane added back unchanged: the exact workspace again
    back = ok(J.workspace_update_project(no_plane, {"project": ws["project"]}))
    assert back["workspace_digest"] == ok(J.workspace_status(ws))["status"]["workspace_digest"]
    assert [(e["kind"], e["parent_id"]) for e in back["result"]["restored_instance_members"]] == [("OBSERVATION", "m1_M1")]
    # the whole instance moved: both members follow
    moved = ok(J.workspace_transform(ws, {"selector": {"instances": ["m1"]}, "translation_mm": [0.0, 5.0, 0.0]}))["workspace"]
    assert changes(moved) == {"m1": []}


def test_a_workspace_written_before_digests_named_their_kind_still_opens():
    """An earlier build kept one digest per id (the first member's): the workspace opens and that member is tracked."""

    ws = ok(J.workspace_import(system_with_laser(), same_id_module(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    legacy = copy.deepcopy(ws)
    rows = instance(legacy, "m1")["member_digests"]
    first = next(d for d in rows if d["parent_id"] == "m1_M1")
    instance(legacy, "m1")["member_digests"] = [d for d in rows if d["parent_id"] != "m1_M1" or d is first]
    first.pop("kind")
    opened = ok(J.workspace_init(json.loads(json.dumps(legacy))))["workspace"]
    assert changes(opened) == {"m1": []}
    deleted = ok(J.workspace_update_project(opened, {"project": without(opened["project"], "components", "m1_M1")}))["workspace"]
    assert changes(deleted) == {"m1": [("COMPONENT", "m1_M1", "REMOVED")]}


def test_an_edited_member_stays_an_edit_when_its_instance_moves_as_a_whole():
    """A move of the whole instance can bring an edited member back onto its import digest (edited 10 mm down, the
    instance moved 10 mm up): it stays an edit, and replace_instance does not move it without discard_edits."""

    ws = two_instances()
    edited = ok(J.workspace_transform(ws, {"selector": {"ids": ["m2_M1"]}, "translation_mm": [0.0, 0.0, -10.0]}))["workspace"]
    moved = ok(J.workspace_transform(edited, {"selector": {"instances": ["m2"]}, "translation_mm": [0.0, 0.0, 10.0]}))["workspace"]
    assert component(moved, "m2_M1") == component(ws, "m2_M1")  # back where the import put it, but the instance moved
    assert changes(moved) == {"m1": [], "m2": [("COMPONENT", "m2_M1", "EDITED")]}
    assert codes(J.workspace_replace_instance(moved, fold_mod(), {"instance_key": "m2", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    replaced = ok(J.workspace_replace_instance(moved, fold_mod(), {"instance_key": "m2", "fixtures": REBIND, "discard_edits": True}))
    assert component(replaced["workspace"], "m2_M1")["nominal_pose"]["position_mm"] == [0.0, 0.0, 210.0]
    # a plane stated in its device's frame does not change when the instance moves: its edit set back is no edit
    module = attached_module()
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    plane = next(o for o in ws["project"]["observations"] if o["observation_id"] == "a_ATT")
    project = copy.deepcopy(ws["project"])
    next(o for o in project["observations"] if o["observation_id"] == "a_ATT")["aperture"]["radius_mm"] = 6.0
    wider = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    moved = ok(J.workspace_transform(wider, {"selector": {"instances": ["a"]}, "translation_mm": [0.0, 0.0, 20.0]}))["workspace"]
    assert changes(moved) == {"a": [("OBSERVATION", "a_ATT", "EDITED")]}
    project = copy.deepcopy(moved["project"])
    project["observations"] = [copy.deepcopy(plane) if o["observation_id"] == "a_ATT" else o for o in project["observations"]]
    reverted = ok(J.workspace_update_project(moved, {"project": project}))["workspace"]
    assert changes(reverted) == {"a": []}
    again = ok(J.workspace_replace_instance(reverted, module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    assert again["project"] == reverted["project"]


@pytest.mark.parametrize("seed", [3, 7])
def test_replacing_an_instance_reported_unedited_never_changes_the_project(seed):
    """Random deletions, re-additions (as imported or moved), whole-instance moves, single-member edits and discarding
    replacements of two fold instances. After every step, each instance the status reports unedited is replaced with the
    same revision without discard_edits: the Project stays exactly as it was. Integer moves keep the pose arithmetic
    exact."""

    rng = random.Random(seed)
    module = fold_mod()
    ws = two_instances()
    ident = {"components": "instance_id", "observations": "observation_id", "targets": "target_id"}
    original = {(name, item[key]): copy.deepcopy(item) for name, key in ident.items() for item in ws["project"][name]}
    members = {ref for ref in original if ref[1].startswith(("m1_", "m2_"))}
    checked = 0
    for _ in range(40):
        present = {(name, item[key]) for name, key in ident.items() for item in ws["project"][name]}
        op = rng.choice(["delete", "readd", "readd_moved", "move", "edit", "replace"])
        if op == "delete" and present & members:
            name, oid = rng.choice(sorted(present & members))
            reply = J.workspace_update_project(ws, {"project": without(ws["project"], name, oid)})
        elif op in ("readd", "readd_moved") and members - present:
            name, oid = rng.choice(sorted(members - present))
            item = copy.deepcopy(original[(name, oid)])
            if op == "readd_moved" and name == "components":
                item["nominal_pose"]["position_mm"][2] += rng.choice([-20.0, 10.0])
            project = copy.deepcopy(ws["project"])
            project[name].append(item)
            reply = J.workspace_update_project(ws, {"project": project})
        elif op == "move":
            reply = J.workspace_transform(ws, {"selector": {"instances": [rng.choice(["m1", "m2"])]},
                                               "translation_mm": [0.0, 0.0, float(rng.choice([-10, 5, 20]))]})
        elif op == "edit" and any(name == "components" for name, _ in present & members):
            target = rng.choice(sorted(oid for name, oid in present & members if name == "components"))
            reply = J.workspace_transform(ws, {"selector": {"ids": [target]}, "translation_mm": [0.0, 0.0, 1.0]})
        elif op == "replace":
            reply = J.workspace_replace_instance(ws, module, {"instance_key": rng.choice(["m1", "m2"]), "fixtures": REBIND,
                                                              "discard_edits": True})
        else:
            continue
        if not reply["ok"]:  # e.g. a target left without its plane: the local check refuses, nothing changes
            continue
        ws = json.loads(json.dumps(reply["workspace"]))
        for status in ok(J.workspace_status(ws))["status"]["instances"]:
            if status["edited"]:
                continue
            replaced = J.workspace_replace_instance(ws, module, {"instance_key": status["instance_key"], "fixtures": REBIND})
            assert replaced["ok"], replaced["error"]
            assert replaced["workspace"]["project"] == ws["project"], (op, status["instance_key"])
            checked += 1
    assert checked >= 5  # the walk reached unedited states
