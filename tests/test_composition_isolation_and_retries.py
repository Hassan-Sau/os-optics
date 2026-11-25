"""Composition: operations without an op_id, per-instance definitions, replacement of nested and sibling
instances, implicit followers, catalog re-pins, out-of-domain inputs, the design-intent freshness reason and quaternion
signs. Each test reproduces a case of the independent review of the composition workspace (RV1)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from composition_helpers import FOLD_DESCRIPTOR, fold_module, fold_workspace, ok, screen, system_with_laser
from local_helpers import FOLD_X_TO_Y

from os_optics import composition as comp
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
REBIND = {"rebind": {"SRC": "LASER"}}


def codes(reply: dict) -> list[str]:
    assert reply["ok"] is False, reply
    return [p["code"] for p in reply["error"]["problems"]]


def component(ws: dict, instance_id: str) -> dict:
    return next(c for c in ws["project"]["components"] if c["instance_id"] == instance_id)


def group(ws: dict, group_id: str) -> dict | None:
    return next((g for g in ws["groups"] if g["group_id"] == group_id), None)


def edited(ws: dict) -> dict[str, list[str]]:
    return {i["instance_key"]: [e["id"] for e in i["edited"]] for i in ok(J.workspace_status(ws))["status"]["instances"]}


def fold_mod(reflectance: str | None = None, revision: str = "r1") -> dict:
    module = ok(J.workspace_init(fold_module(reflectance=reflectance), {"module": {**FOLD_DESCRIPTOR, "revision": revision}}))["workspace"]
    return ok(J.workspace_export_module(module, {}))["module"]


def two_instances() -> dict:
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    return ok(J.workspace_import(ws, fold_mod(), {"instance_key": "m2", "fixtures": REBIND,
                                                  "placement": {"position_mm": [0.0, 0.0, 200.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]


# ------------------------------------------------------------------------------------------------ no op_id


def test_an_operation_without_op_id_is_always_applied_even_when_the_workspace_returns_to_an_earlier_state():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    replies = []
    for dx in (10.0, -10.0, 10.0):
        reply = ok(J.workspace_transform(ws, {"selector": {"instances": ["m1"]}, "translation_mm": [dx, 0.0, 0.0]}))
        replies.append(reply)
        ws = reply["workspace"]
    third = replies[2]
    assert third["replayed"] is False
    assert component(ws, "m1_M1")["nominal_pose"]["position_mm"] == [10.0, 0.0, 0.0]
    assert [a["after"]["position_mm"] for a in third["result"]["affected"] if a["id"] == "m1_M1"] == [[10.0, 0.0, 0.0]]
    op_ids = [r["result"]["op_id"] for r in replies]
    assert len(set(op_ids)) == 3 and all(o.startswith("auto-") and len(o) == len("auto-") + 32 for o in op_ids)
    assert [o["op_id"] for o in ws["operations"]][-3:] == op_ids
    # the generated ids depend only on the journal, the request and the base: the same steps give the same ids
    again = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    for dx in (10.0, -10.0, 10.0):
        again = ok(J.workspace_transform(again, {"selector": {"instances": ["m1"]}, "translation_mm": [dx, 0.0, 0.0]}))["workspace"]
    assert again["operations"] == ws["operations"]


def test_groups_dissolved_and_recreated_or_members_added_back_without_op_id_are_applied():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    g = ok(J.workspace_group_create(ws, {"group_id": "G", "members": {"ids": ["LASER"]}}))["workspace"]
    dissolved = ok(J.workspace_group_dissolve(g, {"group_id": "G"}))["workspace"]
    recreated = ok(J.workspace_group_create(dissolved, {"group_id": "G", "members": {"ids": ["LASER"]}}))
    assert recreated["replayed"] is False and group(recreated["workspace"], "G") is not None
    assert ok(J.workspace_group_dissolve(recreated["workspace"], {"group_id": "G"}))["replayed"] is False
    added = ok(J.workspace_group_add(g, {"group_id": "G", "members": {"ids": ["a_M1"]}}))["workspace"]
    removed = ok(J.workspace_group_remove(added, {"group_id": "G", "members": {"ids": ["a_M1"]}}))["workspace"]
    again = ok(J.workspace_group_add(removed, {"group_id": "G", "members": {"ids": ["a_M1"]}}))
    assert again["replayed"] is False and [m["id"] for m in group(again["workspace"], "G")["members"]] == ["LASER", "a_M1"]


def test_a_project_edit_set_back_and_made_again_without_op_id_is_applied():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    p0 = copy.deepcopy(ws["project"])
    p1 = copy.deepcopy(p0)
    component({"project": p1}, "LASER")["nominal_pose"]["position_mm"][1] = 5.0
    r1 = ok(J.workspace_update_project(ws, {"project": p1}))
    r2 = ok(J.workspace_update_project(r1["workspace"], {"project": p0}))
    r3 = ok(J.workspace_update_project(r2["workspace"], {"project": p1}))
    assert r3["replayed"] is False and component(r3["workspace"], "LASER")["nominal_pose"]["position_mm"][1] == 5.0
    assert r3["workspace_digest"] == r1["workspace_digest"] and r3["workspace_digest"] == comp.load_workspace(r3["workspace"]).workspace_digest()


def test_an_explicit_op_id_still_replays_its_recorded_result():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    up = {"selector": {"instances": ["a"]}, "translation_mm": [0.0, 0.0, 10.0], "op_id": "up-1"}
    first = ok(J.workspace_transform(ws, up))
    down = ok(J.workspace_transform(first["workspace"], {"selector": {"instances": ["a"]}, "translation_mm": [0.0, 0.0, -10.0], "op_id": "down-1"}))
    retry = ok(J.workspace_transform(down["workspace"], up))  # a lost response of up-1, retried: nothing is applied again
    assert retry["replayed"] is True and retry["workspace"] == down["workspace"] and retry["result"] == first["result"]
    assert codes(J.workspace_transform(down["workspace"], {**up, "translation_mm": [0.0, 0.0, 11.0]})) == ["OP_ID_REUSED"]


def test_generated_op_ids_stay_unique_when_the_journal_is_full():
    ws = ok(J.workspace_group_create(system_with_laser(), {"group_id": "G"}))["workspace"]
    for n in range(270):
        name = J.workspace_group_add if n % 2 == 0 else J.workspace_group_remove
        ws = ok(name(ws, {"group_id": "G", "members": {"ids": ["LASER"]}}))["workspace"]
    ids = [o["op_id"] for o in ws["operations"]]
    assert len(ids) == 256 and len(set(ids)) == 256
    assert group(ws, "G")["members"] == []  # 135 adds and 135 removes, all applied


# ------------------------------------------------------------------------------------------------ definitions per instance


def test_editing_one_instance_s_model_leaves_the_other_instance_and_the_source_unchanged():
    module = fold_mod()
    pristine = copy.deepcopy(module)
    ws = two_instances()
    assert component(ws, "m1_M1")["optical_model_id"] == "m1_fold-mirror" and component(ws, "m2_M1")["optical_model_id"] == "m2_fold-mirror"
    project = copy.deepcopy(ws["project"])
    model = next(m for m in project["optical_models"] if m["model_id"] == "m1_fold-mirror")
    model["model"]["surfaces"][0]["clear_radius_mm"] = "5"
    changed = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    models = {m["model_id"]: m for m in changed["project"]["optical_models"]}
    assert models["m2_fold-mirror"] == next(m for m in ws["project"]["optical_models"] if m["model_id"] == "m2_fold-mirror")
    assert edited(changed) == {"m1": ["m1_fold-mirror"], "m2": []}
    assert module == pristine


def test_replacing_one_of_two_instances_of_a_module_with_a_revision():
    ws = two_instances()
    revised = fold_mod("0.95", "r2")
    for key, other in (("m1", "m2"), ("m2", "m1")):
        reply = ok(J.workspace_replace_instance(ws, revised, {"instance_key": key, "fixtures": REBIND}))
        new = reply["workspace"]
        models = {m["model_id"]: m for m in new["project"]["optical_models"]}
        assert set(models) == {"m1_fold-mirror", "m2_fold-mirror"}
        coating = {c["coating_id"] for c in new["project"]["model_coatings"]}
        assert any(c.startswith(f"{key.upper()}_") and "R0.95" in c for c in coating)
        assert models[f"{other}_fold-mirror"] == next(m for m in ws["project"]["optical_models"] if m["model_id"] == f"{other}_fold-mirror")
        assert edited(new) == {"m1": [], "m2": []}
        assert next(i for i in new["instances"] if i["instance_key"] == key)["source"]["revision"] == "r2"


def test_a_definition_still_used_elsewhere_is_named_when_the_revision_cannot_take_its_id():
    ws = two_instances()
    project = copy.deepcopy(ws["project"])
    component({"project": project}, "m2_M1")["optical_model_id"] = "m1_fold-mirror"  # the user points m2's mirror at m1's model
    ws = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    reply = J.workspace_replace_instance(ws, fold_mod("0.95", "r2"), {"instance_key": "m1", "fixtures": REBIND})
    assert codes(reply) == ["ID_CONFLICT"]
    problem = reply["error"]["problems"][0]
    assert problem["details"]["parent_id"] == "m1_fold-mirror" and problem["details"]["used_by"] == [{"kind": "COMPONENT", "id": "m2_M1"}]


def composed_module() -> dict:
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "b", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    return ok(J.workspace_export_module(ws, {"module_id": "two-folds", "title": "Two folds", "revision": "c1", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]


def test_every_instance_of_a_composed_module_can_be_replaced_and_keeps_its_own_definitions():
    module = composed_module()
    host = system_with_laser((-100.0, 0.0, 300.0))
    rebind = {"rebind": {"LASER": "LASER"}}
    ws = ok(J.workspace_import(host, module, {"instance_key": "c", "fixtures": rebind,
                                              "placement": {"position_mm": [0.0, 0.0, 300.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    ws = ok(J.workspace_import(ws, module, {"instance_key": "d", "fixtures": rebind,
                                            "placement": {"position_mm": [0.0, 0.0, 600.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    keys = [(i["instance_key"], i.get("parent_instance")) for i in ws["instances"]]
    assert keys == [("c", None), ("c_a", "c"), ("c_b", "c"), ("d", None), ("d_a", "d"), ("d_b", "d")]
    digests = {i["instance_key"]: {m["parent_id"] for m in i["member_digests"]} for i in ws["instances"]}
    assert all(p.lower().startswith("d_") for p in digests["d_a"]) and "d_a_fold-mirror" in digests["d_a"]
    for key in ("c", "d"):
        reply = ok(J.workspace_replace_instance(ws, module, {"instance_key": key, "fixtures": rebind}))
        assert reply["result"]["replaced"]["removed_nested_instances"] == [f"{key}_a", f"{key}_b"]
        assert [(i["instance_key"], i.get("parent_instance")) for i in reply["workspace"]["instances"]] == keys
        assert reply["workspace"]["project"] == ws["project"] and edited(reply["workspace"]) == {k: [] for k, _ in keys}
    # editing c_a's model marks c and c_a edited, never d or d_a
    project = copy.deepcopy(ws["project"])
    next(m for m in project["optical_models"] if m["model_id"] == "c_a_fold-mirror")["model"]["surfaces"][0]["clear_radius_mm"] = "5"
    changed = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    assert edited(changed) == {"c": ["c_a_fold-mirror"], "c_a": ["c_a_fold-mirror"], "c_b": [], "d": [], "d_a": [], "d_b": []}
    # a nested instance replaced on its own keeps its provenance
    alone = ok(J.workspace_replace_instance(ws, fold_mod(), {"instance_key": "d_a", "fixtures": REBIND}))["workspace"]
    assert next(i for i in alone["instances"] if i["instance_key"] == "d_a")["parent_instance"] == "d"


def test_replacing_an_instance_keeps_an_unrelated_instance_whose_members_were_all_deleted():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "a_b", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 0.0, 200.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    project = copy.deepcopy(ws["project"])
    for name, ident in (("components", "instance_id"), ("observations", "observation_id"), ("targets", "target_id")):
        project[name] = [item for item in project[name] if not item[ident].startswith("a_b_")]
    ws = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    reply = ok(J.workspace_replace_instance(ws, fold_mod(), {"instance_key": "a", "fixtures": REBIND}))
    assert reply["result"]["replaced"]["removed_nested_instances"] == []
    assert [i["instance_key"] for i in reply["workspace"]["instances"]] == ["a", "a_b"]
    assert group(reply["workspace"], "a_b") is not None


# ------------------------------------------------------------------------------------------------ implicit followers


def test_a_plane_attached_to_a_moved_device_is_reported_as_a_follower():
    project = fold_module()
    project["observations"].append(screen("ATT", (0.0, 0.0, 5.0), (1.0, 0.0, 0.0, 0.0), device="M1"))
    module = ok(J.workspace_init(project, {"module": FOLD_DESCRIPTOR}))["workspace"]
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    reply = ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "translation_mm": [0.0, 0.0, 7.0]}))
    assert [(a["kind"], a["id"]) for a in reply["result"]["affected"]] == [("COMPONENT", "a_M1")]
    assert reply["result"]["followers"] == [{"observation_id": "a_ATT", "device_instance_id": "a_M1", "selected": False}]
    selected = ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1", "a_ATT"]}, "translation_mm": [0.0, 0.0, 7.0]}))
    assert selected["result"]["followers"] == [{"observation_id": "a_ATT", "device_instance_id": "a_M1", "selected": True}]
    assert selected["workspace"]["project"] == reply["workspace"]["project"]


# ------------------------------------------------------------------------------------------------ catalog re-pin


def test_a_catalog_re_pin_through_update_project_is_not_an_edit():
    import json

    from local_helpers import FLAT_MIRROR, catalog_document, element

    from composition_helpers import laser, project_of

    newer = catalog_document()
    newer["catalog_version"] = "2026-02-01.1"
    assert J.install_catalog(json.dumps(catalog_document()).encode("utf-8"))["ok"] is True
    assert J.install_catalog(json.dumps(newer).encode("utf-8"))["ok"] is True
    module = project_of([element("M", "mirror", FLAT_MIRROR, (0, 0, 0), FOLD_X_TO_Y, catalog_version="2026-02-01.1")], [], [], [])
    parent = ok(J.workspace_init(project_of([laser("LASER")], [], [], [])))["workspace"]
    ws = ok(J.workspace_import(parent, module, {"instance_key": "k"}))["workspace"]
    pinned = copy.deepcopy(ws["project"])
    component({"project": pinned}, "k_M")["product"]["catalog_version"] = "2026-01-01.1"  # what the Designer's re-pin writes
    repinned = ok(J.workspace_update_project(ws, {"project": pinned}))["workspace"]
    assert edited(repinned) == {"k": []}
    moved = copy.deepcopy(pinned)
    component({"project": moved}, "k_M")["nominal_pose"]["position_mm"][2] = 1.0
    assert edited(ok(J.workspace_update_project(repinned, {"project": moved}))["workspace"]) == {"k": ["k_M"]}
    # a re-pin together with another change of the same member is an edit
    both = copy.deepcopy(ws["project"])
    component({"project": both}, "k_M")["product"]["catalog_version"] = "2026-01-01.1"
    component({"project": both}, "k_M")["nominal_pose"]["position_mm"][2] = 1.0
    assert edited(ok(J.workspace_update_project(ws, {"project": both}))["workspace"]) == {"k": ["k_M"]}


# ------------------------------------------------------------------------------------------------ input domain


def test_a_very_long_or_very_short_rotation_axis_is_scaled_and_non_json_values_are_refused():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    plain = ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "rotation": {"axis": [1.0, 1.0, 0.0], "angle_deg": 30.0}}))
    huge = ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "rotation": {"axis": [1e200, 1e200, 0.0], "angle_deg": 30.0}}))
    tiny = ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "rotation": {"axis": [1e-300, 1e-300, 0.0], "angle_deg": 30.0}}))
    assert huge["workspace"]["project"] == plain["workspace"]["project"] == tiny["workspace"]["project"]
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "rotation": {"axis": [0.0, 0.0, 0.0], "angle_deg": 30.0}})) == ["ROTATION_INVALID"]
    assert ok(J.workspace_transform(ws, {"selector": {"ids": ["a_M1"]}, "rotation": {"axis": [1e308, 1e308, 0.0], "angle_deg": 1e308}}))["ok"]
    for request in ({"selector": {"ids": ["a_M1"]}, "translation_mm": [0.0, np.float64(1.0), 0.0]},
                    {"selector": {"ids": [object()]}, "translation_mm": [0.0, 1.0, 0.0]},
                    {"selector": {"ids": ["a_M1"]}, "translation_mm": [0.0, 1.0, 0.0], "pivot": {"point_mm": [0.0, 0.0, np.float32(1.0)]}}):
        reply = J.workspace_transform(ws, request)
        assert codes(reply) == ["REQUEST_INVALID"] and reply["workspace_digest"] == comp.load_workspace(ws).workspace_digest()
    assert codes(J.workspace_select(ws, {"ids": [object()]})) == ["REQUEST_INVALID"]


# ------------------------------------------------------------------------------------------------ design intent


def test_a_design_intent_change_is_reported_as_another_project_change():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_run(ws))["workspace"]
    project = copy.deepcopy(ws["project"])
    assert project["design_intent"] == "OPTICAL_SYSTEM"
    project["design_intent"] = "LAB_ASSEMBLY"
    changed = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    rows = ok(J.workspace_status(changed))["status"]["validations"]
    assert [(r["freshness"], r["reasons"]) for r in rows] == [("HISTORICAL", ["OTHER_PROJECT_CHANGED"])]
    moved = ok(J.workspace_transform(changed, {"selector": {"ids": ["a_M1"]}, "translation_mm": [0.0, 0.0, 0.5]}))["workspace"]
    assert ok(J.workspace_status(moved))["status"]["validations"][0]["reasons"] == ["PHYSICS_CHANGED"]


# ------------------------------------------------------------------------------------------------ quaternion sign


def test_rotated_poses_are_written_with_a_non_negative_w():
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    before = component(ws, "a_M1")["nominal_pose"]["quaternion_wxyz"]
    reply = ok(J.workspace_transform(ws, {"selector": {"instances": ["a"]}, "rotation": {"axis": [0.0, 0.0, 1.0], "angle_deg": 270.0}}))
    assert reply["result"]["rotation_quaternion_wxyz"] == [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]
    after = component(reply["workspace"], "a_M1")["nominal_pose"]["quaternion_wxyz"]
    assert after[0] >= 0.0
    w, x, y, z = before  # reference: (-90 degrees about +Z) * before, by an independent product, either sign
    h = math.sqrt(0.5)
    expected = np.array([h * w + h * z, h * x + h * y, h * y - h * x, h * z - h * w])
    assert np.allclose(np.array(after), expected, atol=1e-15) or np.allclose(np.array(after), -expected, atol=1e-15)
    for c in reply["workspace"]["project"]["components"]:
        assert c["nominal_pose"]["quaternion_wxyz"][0] >= 0.0
    for instance in reply["workspace"]["instances"]:
        assert instance["placement"]["quaternion_wxyz"][0] >= 0.0
