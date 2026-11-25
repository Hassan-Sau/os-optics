"""Composition (review): the local edits of the instances nested in an instance are edits of it.

An instance of a composed module encloses the instances its import created (``parent_instance``). Replacing it replaces
them, so their edits and deletions, and the members a nested instance gained when it was replaced on its own, are edits
of the enclosing instance: ``replace_instance`` refuses them (``INSTANCE_EDITED``, listing each) unless
``discard_edits``, and then lists each in ``edits_discarded`` and leaves no object that no instance holds. A nested
instance replaced on its own updates the records of the instances enclosing it: a member its new revision adds is
recorded there as added (``ADDED``), so the enclosing instance holds it (selection, whole-instance moves, its import
group) and replacing it accounts for it."""

from __future__ import annotations

import copy
import functools
import json
import random

import pytest
from composition_helpers import FOLD_DESCRIPTOR, fold_module, ok, screen, system_with_laser
from test_composition_deletions import changes, instance, without
from test_composition_isolation_and_retries import REBIND, codes, composed_module, fold_mod, group

from os_optics.composition.document import load_workspace
from os_optics.composition.operations import ADDED_EDIT_DIGEST, nested_instances
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

RB = {"rebind": {"LASER": "LASER"}}
PLACE = {"position_mm": [0.0, 0.0, 300.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
NAMES = {"components": ("COMPONENT", "instance_id"), "observations": ("OBSERVATION", "observation_id"), "targets": ("TARGET", "target_id"),
         "optical_models": ("MODEL", "model_id"), "model_materials": ("MATERIAL", "material_id"), "model_coatings": ("COATING", "coating_id")}
ALL = {"project": True, "instances": True, "groups": True}
EXTRA_ADDED = {"kind": "OBSERVATION", "id": "c_a_EXTRA", "change": "ADDED"}
EXTRA_REMOVED = {"kind": "OBSERVATION", "id": "c_a_EXTRA", "change": "REMOVED"}


def rt(value):
    return json.loads(json.dumps(value))


def fold_revision(revision: str, planes=(("EXTRA", (0.0, 70.0, 0.0)),), *, carried_target: bool = False, without_monitor: bool = False) -> dict:
    """The fold module with more observation planes (a revision that adds members), optionally a carried target on
    EXTRA, or without its monitor plane and target (a revision that lacks members)."""

    project = fold_module()
    descriptor = {**FOLD_DESCRIPTOR, "revision": revision}
    for oid, at in planes:
        project["observations"].append(screen(oid, at, (0.7071067811865476, -0.7071067811865476, 0.0, 0.0), radius_mm=15.0))
    if carried_target:
        project["targets"].append({"target_id": "T_EXTRA", "metric": "spot_centroid_u_mm", "observation_id": "EXTRA", "target_value": 0.0,
                                   "tolerance_abs": 0.5})
        descriptor["requirements"] = [*FOLD_DESCRIPTOR["requirements"], {"target_id": "T_EXTRA", "scope": "CARRY_TO_PARENT"}]
    if without_monitor:
        project = without(without(project, "targets", "T_MON_CENTER"), "observations", "MON")
        descriptor["requirements"] = []
    return ok(J.workspace_export_module(ok(J.workspace_init(project, {"module": descriptor}))["workspace"], {}))["module"]


def composed(a_module: dict, revision: str) -> dict:
    """The two-fold composed module with its fold ``a`` at another revision."""

    ws = ok(J.workspace_import(system_with_laser(), a_module, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "b", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    return ok(J.workspace_export_module(ws, {"module_id": "two-folds", "title": "Two folds", "revision": revision, "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]


@functools.cache
def modules() -> dict:
    r3 = fold_revision("r3")
    return {"c1": composed_module(), "r1": fold_mod(), "r3": r3, "c2": composed(r3, "c2")}


def host() -> dict:
    return system_with_laser((-100.0, 0.0, 300.0))


@functools.cache
def _imported() -> str:
    return json.dumps(ok(J.workspace_import(host(), modules()["c1"], {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"])


def imported() -> dict:
    """The composed module c1 imported as c (nested c_a, c_b), unedited."""

    return json.loads(_imported())


def replaced(ws: dict, key: str, module: str, *, discard: bool = False) -> dict:
    return J.workspace_replace_instance(ws, modules()[module], {"instance_key": key, "fixtures": RB if key == "c" else REBIND,
                                                                "discard_edits": discard})


def same(a: dict, b: dict, *nested_replaced: str) -> dict[str, bool]:
    """Project, instances and groups equal (apart from the fixture outcomes of nested instances replaced on their own:
    the replacement's fixture policy is its request's)."""

    def records(ws):
        return [{k: v for k, v in i.items() if not (k == "fixtures" and i["instance_key"] in nested_replaced)} for i in rt(ws["instances"])]

    return {"project": rt(a["project"]) == rt(b["project"]), "instances": records(a) == records(b), "groups": rt(a["groups"]) == rt(b["groups"])}


def orphans(ws: dict) -> list[tuple[str, str]]:
    """Project objects no instance holds, except the system's own laser."""

    present = {(kind, item[ident]) for name, (kind, ident) in NAMES.items() for item in ws["project"].get(name) or []}
    held = {(e["kind"], e["parent_id"]) for i in ws["instances"] for e in i["id_map"]}
    return sorted(present - held - {("COMPONENT", "LASER")})


def edited_details(reply: dict) -> list[dict]:
    assert codes(reply) == ["INSTANCE_EDITED"]
    return reply["error"]["problems"][0]["details"]["edited"]


def with_added_plane() -> dict:
    """c_a replaced on its own by r3, which adds the plane EXTRA."""

    return ok(replaced(imported(), "c_a", "r3"))["workspace"]


# ------------------------------------------------------------------------------------------------ the record


def test_a_member_a_nested_replacement_adds_is_recorded_by_the_enclosing_instance():
    ws, added = imported(), with_added_plane()
    assert changes(added) == {"c": [("OBSERVATION", "c_a_EXTRA", "ADDED")], "c_a": [], "c_b": []}
    assert [e for e in instance(added, "c")["id_map"] if e["parent_id"] == "c_a_EXTRA"] == [
        {"kind": "OBSERVATION", "source_id": "a_EXTRA", "parent_id": "c_a_EXTRA", "shared": False}]
    assert [d["digest"] for d in instance(added, "c")["member_digests"] if d["parent_id"] == "c_a_EXTRA"] == [ADDED_EDIT_DIGEST]
    assert group(added, "c")["members"] == [*group(ws, "c")["members"], {"kind": "OBSERVATION", "id": "c_a_EXTRA"}]
    assert "c_a_EXTRA" in ok(J.workspace_select(added, {"instances": ["c"]}))["selection"]["observations"]
    assert orphans(added) == []
    assert ok(J.workspace_init(rt(added)))["workspace_digest"] == ok(J.workspace_status(added))["status"]["workspace_digest"]
    # the whole enclosing instance moves with the added member; the nested records follow the move
    moved = ok(J.workspace_transform(added, {"selector": {"instances": ["c"]}, "translation_mm": [0.0, 0.0, 10.0]}))
    assert moved["result"]["placements_updated"] == ["c", "c_a", "c_b"]
    assert changes(moved["workspace"]) == changes(added)
    # the nested instance back at its first revision: the enclosing record is as imported
    back = ok(replaced(added, "c_a", "r1"))["workspace"]
    assert same(back, ws, "c_a") == ALL and changes(back) == {"c": [], "c_a": [], "c_b": []}


def test_replacing_the_enclosing_instance_after_a_nested_replacement_needs_discard_edits():
    ws, added = imported(), with_added_plane()
    for module in ("c2", "c1"):
        refused = replaced(added, "c", module)
        assert edited_details(refused) == [EXTRA_ADDED]
        assert "(1 member added)" in refused["error"]["problems"][0]["message"]
    fresh = ok(replaced(ws, "c", "c2"))["workspace"]  # c2's a is r3: the replacement of the unedited instance
    for module, expected in (("c2", fresh), ("c1", ws)):
        reply = ok(replaced(added, "c", module, discard=True))
        assert reply["result"]["replaced"]["edits_discarded"] == [EXTRA_ADDED]
        assert same(reply["workspace"], expected) == ALL and orphans(reply["workspace"]) == []


# ------------------------------------------------------------------------------------------------ the review's cases


@pytest.mark.parametrize("module", ["c2", "c1"])
def test_a_deleted_member_of_a_nested_instance_is_an_edit_of_the_enclosing_instance(module):
    """Review cases (a) and (b): c_a replaced alone by r3 (adds EXTRA), c_a_EXTRA deleted, then c replaced by a composed
    revision that has the member (c2) or lacks it (c1): refused, never restored or dropped silently."""

    ws, added = imported(), with_added_plane()
    update = ok(J.workspace_update_project(added, {"project": without(added["project"], "observations", "c_a_EXTRA")}))
    deleted = update["workspace"]
    assert [(e["instance_key"], e["parent_id"]) for e in update["result"]["removed_instance_members"]] == [("c", "c_a_EXTRA"), ("c_a", "c_a_EXTRA")]
    assert changes(deleted) == {"c": [("OBSERVATION", "c_a_EXTRA", "REMOVED")], "c_a": [("OBSERVATION", "c_a_EXTRA", "REMOVED")], "c_b": []}
    refused = replaced(deleted, "c", module)
    assert edited_details(refused) == [EXTRA_REMOVED]
    assert "(1 member deleted)" in refused["error"]["problems"][0]["message"]
    assert codes(replaced(deleted, "c_a", "r3")) == ["INSTANCE_EDITED"]
    reply = ok(replaced(deleted, "c", module, discard=True))
    assert reply["result"]["replaced"]["edits_discarded"] == [EXTRA_REMOVED]
    expected = ok(replaced(ws, "c", module))["workspace"]
    assert same(reply["workspace"], expected) == ALL and orphans(reply["workspace"]) == []
    assert changes(reply["workspace"]) == {"c": [], "c_a": [], "c_b": []}
    # the nested instance discarded back to r3 is the state before the deletion; the member added back too
    assert same(ok(replaced(deleted, "c_a", "r3", discard=True))["workspace"], added, "c_a") == ALL
    readd = ok(J.workspace_update_project(deleted, {"project": added["project"]}))
    assert sorted((e["instance_key"], e["parent_id"]) for e in readd["result"]["restored_instance_members"]) == [("c", "c_a_EXTRA"), ("c_a", "c_a_EXTRA")]
    assert same(readd["workspace"], added) == ALL


@pytest.mark.parametrize("module", ["c2", "c1"])
def test_an_edited_member_of_a_nested_instance_is_an_edit_of_the_enclosing_instance(module):
    """Review case (c): the added member edited instead: INSTANCE_EDITED (not ID_CONFLICT); with discard_edits no object
    stays behind that no instance holds."""

    ws, added = imported(), with_added_plane()
    project = copy.deepcopy(added["project"])
    next(o for o in project["observations"] if o["observation_id"] == "c_a_EXTRA")["aperture"]["radius_mm"] = 9.0
    edited = ok(J.workspace_update_project(added, {"project": project}))["workspace"]
    assert changes(edited) == {"c": [("OBSERVATION", "c_a_EXTRA", "ADDED")], "c_a": [("OBSERVATION", "c_a_EXTRA", "EDITED")], "c_b": []}
    assert edited_details(replaced(edited, "c", module)) == [EXTRA_ADDED]
    reply = ok(replaced(edited, "c", module, discard=True))
    assert reply["result"]["replaced"]["edits_discarded"] == [EXTRA_ADDED]
    assert same(reply["workspace"], ok(replaced(ws, "c", module))["workspace"]) == ALL and orphans(reply["workspace"]) == []


def test_an_edit_only_the_nested_record_shows_is_an_edit_of_the_enclosing_instance():
    """c_a replaced on its own by r2 (another mirror coating), then its mirror model and coating set back to the first
    revision's by hand: the enclosing record sees its imported model and coating again, the nested record (r2) an edited
    model and an added coating (r2 lacks it). Replacing the enclosing instance lists both records' edits, each member
    once."""

    ws = imported()
    revised = ok(J.workspace_replace_instance(ws, fold_mod("0.95", "r2"), {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    coatings = {c["coating_id"] for c in ws["project"]["model_coatings"] if c["coating_id"].startswith("C_A_")}
    new_coatings = {c["coating_id"] for c in revised["project"]["model_coatings"] if c["coating_id"].startswith("C_A_")}
    (old,), (new,) = coatings - new_coatings, new_coatings - coatings
    assert changes(revised) == {"c": [("MODEL", "c_a_fold-mirror", "EDITED"), ("COATING", old, "REMOVED"), ("COATING", new, "ADDED")],
                                "c_a": [], "c_b": []}
    project = copy.deepcopy(revised["project"])
    project["optical_models"] = [copy.deepcopy(m) if m["model_id"] != "c_a_fold-mirror" else
                                 copy.deepcopy(next(x for x in ws["project"]["optical_models"] if x["model_id"] == "c_a_fold-mirror"))
                                 for m in project["optical_models"]]
    project["model_coatings"].append(copy.deepcopy(next(c for c in ws["project"]["model_coatings"] if c["coating_id"] == old)))
    by_hand = ok(J.workspace_update_project(revised, {"project": project}))["workspace"]
    assert changes(by_hand) == {"c": [("COATING", new, "ADDED"), ("MODEL", "c_a_fold-mirror", "EDITED"), ("COATING", old, "ADDED")],
                                "c_a": [("MODEL", "c_a_fold-mirror", "EDITED"), ("COATING", old, "ADDED")], "c_b": []}
    assert edited_details(replaced(by_hand, "c", "c1")) == [{"kind": "COATING", "id": new, "change": "ADDED"},
                                                             {"kind": "MODEL", "id": "c_a_fold-mirror", "change": "EDITED"},
                                                             {"kind": "COATING", "id": old, "change": "ADDED"}]
    reply = ok(replaced(by_hand, "c", "c1", discard=True))
    assert same(reply["workspace"], ws) == ALL and orphans(reply["workspace"]) == []


def test_a_workspace_written_before_enclosing_records_held_added_members():
    """An earlier build left the member a nested replacement added out of the enclosing record: it is still an edit of
    the enclosing instance, and replacing that one removes it (no orphan, no ID_CONFLICT)."""

    ws, added = imported(), with_added_plane()
    legacy = rt(added)
    record = instance(legacy, "c")
    record["id_map"] = [e for e in record["id_map"] if e["parent_id"] != "c_a_EXTRA"]
    record["member_digests"] = [d for d in record["member_digests"] if d["parent_id"] != "c_a_EXTRA"]
    group(legacy, "c")["members"] = [m for m in group(legacy, "c")["members"] if m["id"] != "c_a_EXTRA"]
    legacy = ok(J.workspace_init(legacy))["workspace"]
    assert changes(legacy) == {"c": [("OBSERVATION", "c_a_EXTRA", "ADDED")], "c_a": [], "c_b": []}
    assert edited_details(replaced(legacy, "c", "c1")) == [EXTRA_ADDED]
    reply = ok(replaced(legacy, "c", "c1", discard=True))
    assert same(reply["workspace"], ws) == ALL and orphans(reply["workspace"]) == []
    project = copy.deepcopy(legacy["project"])
    next(o for o in project["observations"] if o["observation_id"] == "c_a_EXTRA")["aperture"]["radius_mm"] = 9.0
    edited = ok(J.workspace_update_project(legacy, {"project": project}))["workspace"]
    assert codes(replaced(edited, "c", "c2")) == ["INSTANCE_EDITED"]
    assert same(ok(replaced(edited, "c", "c2", discard=True))["workspace"], ok(replaced(ws, "c", "c2"))["workspace"]) == ALL


# ------------------------------------------------------------------------------------------------ more shapes


def test_two_levels_of_nesting():
    inner = ok(J.workspace_import(system_with_laser(), modules()["c1"], {"instance_key": "c", "fixtures": RB}))["workspace"]
    nest = ok(J.workspace_export_module(inner, {"module_id": "nest", "title": "Nest", "revision": "n1", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    ws = ok(J.workspace_import(host(), nest, {"instance_key": "d", "fixtures": RB, "placement": PLACE}))["workspace"]
    assert [(i["instance_key"], i.get("parent_instance")) for i in ws["instances"]] == [("d", None), ("d_c", "d"), ("d_c_a", "d_c"), ("d_c_b", "d_c")]
    added = ok(J.workspace_replace_instance(ws, modules()["r3"], {"instance_key": "d_c_a", "fixtures": REBIND}))["workspace"]
    assert changes(added) == {"d": [("OBSERVATION", "d_c_a_EXTRA", "ADDED")], "d_c": [("OBSERVATION", "d_c_a_EXTRA", "ADDED")], "d_c_a": [], "d_c_b": []}
    assert [e["source_id"] for key in ("d", "d_c") for e in instance(added, key)["id_map"] if e["parent_id"] == "d_c_a_EXTRA"] == ["c_a_EXTRA", "a_EXTRA"]
    deleted = ok(J.workspace_update_project(added, {"project": without(added["project"], "observations", "d_c_a_EXTRA")}))["workspace"]
    for key, module, fixtures in (("d", nest, RB), ("d_c", modules()["c1"], RB)):
        assert codes(J.workspace_replace_instance(deleted, module, {"instance_key": key, "fixtures": fixtures})) == ["INSTANCE_EDITED"]
        reply = ok(J.workspace_replace_instance(deleted, module, {"instance_key": key, "fixtures": fixtures, "discard_edits": True}))
        assert reply["result"]["replaced"]["edits_discarded"] == [{"kind": "OBSERVATION", "id": "d_c_a_EXTRA", "change": "REMOVED"}]
        assert same(reply["workspace"], ws, "d_c") == ALL and orphans(reply["workspace"]) == []
    back = ok(J.workspace_replace_instance(added, modules()["r1"], {"instance_key": "d_c_a", "fixtures": REBIND}))["workspace"]
    assert same(back, ws, "d_c_a") == ALL
    moved = ok(J.workspace_transform(added, {"selector": {"instances": ["d"]}, "translation_mm": [0.0, 5.0, 0.0]}))
    assert moved["result"]["placements_updated"] == ["d", "d_c", "d_c_a", "d_c_b"] and changes(moved["workspace"]) == changes(added)


def test_an_added_target_deleted_in_a_batch_and_after_reopening():
    ws = imported()
    r4 = fold_revision("r4", carried_target=True)
    c4 = composed(r4, "c4")
    added = ok(J.workspace_replace_instance(ws, r4, {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    assert changes(added)["c"] == [("OBSERVATION", "c_a_EXTRA", "ADDED"), ("TARGET", "c_a_T_EXTRA", "ADDED")]
    no_target = without(added["project"], "targets", "c_a_T_EXTRA")
    deleted = ok(J.workspace_init(rt(ok(J.workspace_update_project(added, {"project": no_target}))["workspace"])))["workspace"]
    assert [(e["id"], e["change"]) for e in edited_details(J.workspace_replace_instance(deleted, c4, {"instance_key": "c", "fixtures": RB}))] == [
        ("c_a_EXTRA", "ADDED"), ("c_a_T_EXTRA", "REMOVED")]
    fresh = ok(J.workspace_replace_instance(ws, c4, {"instance_key": "c", "fixtures": RB}))["workspace"]
    steps = [{"kind": "replace_instance", "request": {"instance_key": "c_a", "fixtures": REBIND}, "module": r4},
             {"kind": "update_project", "request": {"project": no_target}},
             {"kind": "replace_instance", "request": {"instance_key": "c", "fixtures": RB}, "module": c4}]
    assert codes(J.workspace_batch(ws, {"operations": steps})) == ["INSTANCE_EDITED"]
    steps[2]["request"]["discard_edits"] = True
    assert same(ok(J.workspace_batch(ws, {"operations": steps}))["workspace"], fresh) == ALL


def test_user_groups_follow_the_source_id_of_an_added_member():
    added = with_added_plane()
    grouped = ok(J.workspace_group_create(added, {"group_id": "ug", "members": {"ids": ["c_a_EXTRA", "c_b_M1"]}}))["workspace"]
    kept = ok(replaced(grouped, "c", "c2", discard=True))
    assert group(kept["workspace"], "ug")["members"] == group(grouped, "ug")["members"]
    dropped = ok(replaced(grouped, "c", "c1", discard=True))
    assert group(dropped["workspace"], "ug")["members"] == [{"kind": "COMPONENT", "id": "c_b_M1"}]
    assert {"group_id": "ug", "kind": "OBSERVATION", "id": "c_a_EXTRA"} in dropped["result"]["replaced"]["dropped_group_members"]


def test_sibling_nested_replacements_are_recorded_and_forgotten_separately():
    added = with_added_plane()
    both = ok(J.workspace_replace_instance(added, fold_revision("r3b"), {"instance_key": "c_b", "fixtures": REBIND}))["workspace"]
    assert changes(both)["c"] == [("OBSERVATION", "c_a_EXTRA", "ADDED"), ("OBSERVATION", "c_b_EXTRA", "ADDED")]
    back = ok(replaced(both, "c_b", "r1"))["workspace"]
    assert same(back, added, "c_b") == ALL


def test_an_added_member_sharing_its_id_with_a_member_of_another_kind():
    ws = imported()
    same_id = fold_revision("rs", (("M1", (0.0, 30.0, 0.0)),))
    added = ok(J.workspace_replace_instance(ws, same_id, {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    assert changes(added) == {"c": [("OBSERVATION", "c_a_M1", "ADDED")], "c_a": [], "c_b": []}
    digests = sorted((d.get("kind"), d["digest"] == ADDED_EDIT_DIGEST) for d in instance(added, "c")["member_digests"] if d["parent_id"] == "c_a_M1")
    assert digests == [("COMPONENT", False), ("OBSERVATION", True)]
    assert same(ok(replaced(added, "c_a", "r1"))["workspace"], ws, "c_a") == ALL
    assert same(ok(replaced(added, "c", "c1", discard=True))["workspace"], ws) == ALL


def test_a_nested_revision_lacking_a_member_and_the_first_revision_again():
    """The enclosing instance records the member the nested revision lacks as deleted; the first revision restores it
    (its id is not reserved against the nested instance), at its place in the Project's lists."""

    ws = imported()
    lacking = ok(J.workspace_replace_instance(ws, fold_revision("rm", (), without_monitor=True), {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    assert changes(lacking) == {"c": [("OBSERVATION", "c_a_MON", "REMOVED"), ("TARGET", "c_a_T_MON_CENTER", "REMOVED")], "c_a": [], "c_b": []}
    assert codes(replaced(lacking, "c", "c1")) == ["INSTANCE_EDITED"]
    assert same(ok(replaced(lacking, "c_a", "r1"))["workspace"], ws, "c_a") == ALL


def test_an_id_an_enclosing_instance_records_as_deleted_stays_reserved_for_the_nested_instance_owning_it():
    """Only ids in the replaced nested instance's own namespace are released: c_a may not take c_a_b_MON, which c records
    as deleted for its nested instance c_a_b (whose revision dropped it)."""

    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "a_b", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    module = ok(J.workspace_export_module(ws, {"module_id": "folds", "title": "Folds", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    ws = ok(J.workspace_import(host(), module, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    lacking = ok(J.workspace_replace_instance(ws, fold_revision("rm", (), without_monitor=True), {"instance_key": "c_a_b", "fixtures": REBIND}))["workspace"]
    assert ("OBSERVATION", "c_a_b_MON", "REMOVED") in changes(lacking)["c"]
    reply = J.workspace_replace_instance(lacking, fold_revision("rb", (("b_MON", (0.0, 70.0, 0.0)),)), {"instance_key": "c_a", "fixtures": REBIND})
    assert codes(reply) == ["ID_CONFLICT"]
    assert reply["error"]["problems"][0]["details"] == {"kind": "OBSERVATION", "parent_id": "c_a_b_MON", "deleted_from": "c"}
    assert same(ok(J.workspace_replace_instance(lacking, fold_mod(), {"instance_key": "c_a_b", "fixtures": REBIND}))["workspace"], ws, "c_a_b") == ALL


def test_a_module_exported_after_a_nested_replacement_is_what_it_is():
    added = with_added_plane()
    module = ok(J.workspace_export_module(added, {"module_id": "x", "title": "X", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    ws = ok(J.workspace_import(system_with_laser((-100.0, 0.0, 600.0)), module, {
        "instance_key": "s", "fixtures": RB, "placement": {"position_mm": [0.0, 0.0, 600.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    assert changes(ws) == {"s": [], "s_c": [], "s_c_a": [], "s_c_b": []}
    assert [e["source_id"] for e in instance(ws, "s_c")["id_map"] if e["parent_id"] == "s_c_a_EXTRA"] == ["a_EXTRA"]
    assert ADDED_EDIT_DIGEST not in {d["digest"] for i in ws["instances"] for d in i["member_digests"]}


# ------------------------------------------------------------------------------------------------ random walk


def _objects(ws: dict) -> dict:
    return {(kind, item[ident]): item for name, (kind, ident) in NAMES.items() for item in ws["project"].get(name) or []}


@pytest.mark.parametrize("seed", [4, 11])
def test_replacing_an_instance_never_loses_a_local_edit_of_its_nested_instances(seed):
    """Random delete / re-add / edit / whole-instance move / replace (revisions adding, lacking or changing members,
    with and without discard_edits) on a composed instance c (c_a, c_b) and an instance m. After every step: a replacement
    without discard_edits brings back no object an instance (other than one enclosing the replaced one) reported deleted
    and changes no object reported edited or added in its scope; a replacement leaves no object that no instance holds
    and was held before; every enclosing record holds its nested instances' members; an instance reported unedited,
    replaced by its own module, keeps the Project's objects."""

    r3 = modules()["r3"]
    folds = [modules()["r1"], r3, fold_revision("rm", (), without_monitor=True), fold_mod("0.95", "r2")]
    composed_family = [modules()["c1"], modules()["c2"]]
    by_digest = {ok(J.workspace_init(m))["module_digest"]: m for m in folds + composed_family}
    ws = ok(J.workspace_import(imported(), folds[0], {"instance_key": "m", "fixtures": REBIND,
                                                      "placement": {"position_mm": [0.0, 0.0, 700.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    originals = copy.deepcopy(_objects(ws))
    rng = random.Random(seed)
    for step in range(30):
        op = rng.choice(["delete", "delete", "readd", "edit", "move", "replace", "replace", "replace", "discard", "same"])
        objects = _objects(ws)
        members = sorted(r for r in objects if r != ("COMPONENT", "LASER") and r[0] in ("COMPONENT", "OBSERVATION", "TARGET"))
        name = {"COMPONENT": "components", "OBSERVATION": "observations", "TARGET": "targets"}
        reply = None
        if op == "delete" and members:
            kind, oid = rng.choice(members)
            reply = J.workspace_update_project(ws, {"project": without(ws["project"], name[kind], oid)})
        elif op == "readd" and (gone := sorted(r for r in originals if r not in objects and r[0] in name)):
            kind, oid = rng.choice(gone)
            project = copy.deepcopy(ws["project"])
            project[name[kind]].append(copy.deepcopy(originals[(kind, oid)]))
            reply = J.workspace_update_project(ws, {"project": project})
        elif op == "edit" and (planes := [r for r in members if r[0] == "OBSERVATION"]):
            project = copy.deepcopy(ws["project"])
            oid = rng.choice(planes)[1]
            next(o for o in project["observations"] if o["observation_id"] == oid)["aperture"]["radius_mm"] = rng.choice([9.0, 11.0, 20.0])
            reply = J.workspace_update_project(ws, {"project": project})
        elif op == "move":
            key = rng.choice([i["instance_key"] for i in ws["instances"]])
            reply = J.workspace_transform(ws, {"selector": {"instances": [key]}, "translation_mm": [0.0, 0.0, float(rng.choice([-10, 10]))]})
        elif op in ("replace", "discard", "same"):
            key = rng.choice([i["instance_key"] for i in ws["instances"]])
            source = by_digest[instance(ws, key)["source"]["module_digest"]]
            module = source if op == "same" else rng.choice(composed_family if key == "c" else folds)
            model = load_workspace(ws)
            enclosing = {i.instance_key for i in model.instances if i.instance_key != key and key in nested_instances(model, i.instance_key)}
            scope = {key, *nested_instances(model, key)}
            rows = {i["instance_key"]: i["edited"] for i in ok(J.workspace_status(ws))["status"]["instances"]}
            reply = J.workspace_replace_instance(ws, module, {"instance_key": key, "fixtures": RB if key == "c" else REBIND, "discard_edits": op == "discard"})
            after = _objects(reply["workspace"]) if reply["ok"] else None
            if reply["ok"] and op != "discard":
                removed = {(e["kind"], e["id"]) for k, r in rows.items() if k not in enclosing for e in r if e["change"] == "REMOVED"}
                assert not removed & set(after), (seed, step, key)
                touched = {(e["kind"], e["id"]) for k in scope for e in rows.get(k, []) if e["change"] != "REMOVED"}
                assert all(after.get(r) == objects.get(r) for r in touched), (seed, step, key)
            if op == "same" and not rows[key]:
                assert reply["ok"] and after == objects, (seed, step, key)
            if reply["ok"]:
                assert set(orphans(reply["workspace"])) <= set(orphans(ws)), (seed, step, key)
        if reply is None or not reply["ok"]:
            continue
        ws = rt(reply["workspace"])
        model = load_workspace(ws)
        for record in ws["instances"]:
            held = {(e["kind"], e["parent_id"]) for e in record["id_map"]}
            for key in nested_instances(model, record["instance_key"]):
                assert {(e["kind"], e["parent_id"]) for e in instance(ws, key)["id_map"] if not e["shared"]} <= held, (seed, step, key)
