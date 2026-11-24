"""Composition (review): a nested instance's part of the instances enclosing it.

The import of a composed module copies the members of its nested instances into the enclosing instance's record too.
A nested instance replaced on its own by a revision that lacks some of those members leaves them recorded as deleted in
the enclosing records, and records them itself with ``LACKED_MEMBER_DIGEST``: they stay its part. So replacing it again
brings them back (their ids are released for it, and only for it), an object added back with such an id is a member
added to it (``ADDED``, an edit that replacing it discards), and an enclosing instance between keeps the outer records
consistent. A member of an enclosing module whose id merely lies in a nested instance's namespace is not that
instance's part. A replacement by the revision an unedited instance already has changes nothing, also after a rotation
and after a catalog re-pin."""

from __future__ import annotations

import copy
import json

import pytest
from composition_helpers import FOLD_DESCRIPTOR, fold_module, laser, ok, project_of, screen, system_with_laser
from test_composition_deletions import changes, instance, without
from test_composition_isolation_and_retries import REBIND, codes, composed_module, fold_mod, group
from test_composition_nested_edits import ALL, PLACE, RB, edited_details, fold_revision, host, imported, modules, orphans, replaced, rt, same

from os_optics.composition.document import load_workspace
from os_optics.composition.operations import ADDED_EDIT_DIGEST, LACKED_MEMBER_DIGEST, nested_instances
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

Q = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
ALONG_Y = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
FIXTURES = [{"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]


def digest_of(ws: dict, key: str, parent_id: str) -> list[str]:
    return [d["digest"] for d in instance(ws, key)["member_digests"] if d["parent_id"] == parent_id]


def holders(ws: dict, parent_id: str) -> list[str]:
    return [i["instance_key"] for i in ws["instances"] if any(e["parent_id"] == parent_id for e in i["id_map"])]


def enclosing_records_hold_nested_members(ws: dict) -> bool:
    model = load_workspace(ws)
    for record in ws["instances"]:
        held = {(e["kind"], e["parent_id"]) for e in record["id_map"]}
        for key in nested_instances(model, record["instance_key"]):
            if not {(e["kind"], e["parent_id"]) for e in instance(ws, key)["id_map"] if not e["shared"]} <= held:
                return False
    return True


def lacking() -> dict:
    """c_a replaced on its own by rm (lacks the monitor plane MON and its target)."""

    return ok(J.workspace_replace_instance(imported(), fold_revision("rm", (), without_monitor=True), {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]


def objects(ws: dict) -> dict:
    """Every Project object by (list name, id): its content, whatever its place in the list."""

    idents = {"components": "instance_id", "observations": "observation_id", "targets": "target_id", "optical_models": "model_id",
              "model_materials": "material_id", "model_coatings": "coating_id"}
    return {(name, item[ident]): item for name, ident in idents.items() for item in ws["project"].get(name) or []}


def readded(ws: dict, original: dict, *refs: tuple[str, str]) -> dict:
    """``ws`` with the objects ``refs`` (list name, id) of ``original`` appended again."""

    idents = {"observations": "observation_id", "targets": "target_id", "components": "instance_id"}
    project = copy.deepcopy(ws["project"])
    for name, oid in refs:
        project[name].append(copy.deepcopy(next(x for x in original["project"][name] if x[idents[name]] == oid)))
    return project


# ------------------------------------------------------------------------------------------------ a revision lacking members


def test_a_nested_instance_records_the_members_its_revision_lacks():
    ws, lack = imported(), lacking()
    assert changes(lack) == {"c": [("OBSERVATION", "c_a_MON", "REMOVED"), ("TARGET", "c_a_T_MON_CENTER", "REMOVED")], "c_a": [], "c_b": []}
    assert [(e["kind"], e["source_id"], e["parent_id"]) for e in instance(lack, "c_a")["id_map"] if e["parent_id"] in ("c_a_MON", "c_a_T_MON_CENTER")] == [
        ("OBSERVATION", "MON", "c_a_MON"), ("TARGET", "T_MON_CENTER", "c_a_T_MON_CENTER")]
    # at the places the import had them (their place in the import group when they come back)
    assert instance(lack, "c_a")["id_map"] == instance(ws, "c_a")["id_map"]
    assert digest_of(lack, "c_a", "c_a_MON") == [LACKED_MEMBER_DIGEST] and digest_of(lack, "c", "c_a_MON") == digest_of(ws, "c", "c_a_MON")
    assert enclosing_records_hold_nested_members(lack) and orphans(lack) == []
    # replaced by the same revision again: nothing changes; by the first revision: as imported
    assert same(ok(replaced(lack, "c_a", "r1"))["workspace"], ws, "c_a") == ALL
    again = ok(J.workspace_replace_instance(lack, fold_revision("rm", (), without_monitor=True), {"instance_key": "c_a", "fixtures": REBIND}))
    assert same(again["workspace"], lack) == ALL and again["result"]["replaced"]["removed_instance_members"] == []
    # moving the enclosing instance as a whole keeps the nested record's lacked members as they are
    moved = ok(J.workspace_transform(lack, {"selector": {"instances": ["c"]}, "translation_mm": [0.0, 0.0, 10.0]}))["workspace"]
    assert digest_of(moved, "c_a", "c_a_MON") == [LACKED_MEMBER_DIGEST] and changes(moved)["c_a"] == []
    reopened = ok(J.workspace_init(rt(lack)))
    assert reopened["workspace_digest"] == ok(J.workspace_status(lack))["status"]["workspace_digest"]


def test_an_object_added_back_that_the_nested_revision_lacks_is_an_edit_of_the_nested_instance():
    """The monitor plane and its target added back after c_a was replaced by rm are c's members as
    imported, and members added to c_a (whose revision lacks them). Replacing c_a by r1 is refused (not ID_CONFLICT),
    and with discard_edits it replaces them with r1's."""

    ws, lack = imported(), lacking()
    back = ok(J.workspace_update_project(lack, {"project": readded(lack, ws, ("observations", "c_a_MON"), ("targets", "c_a_T_MON_CENTER"))}))
    assert [(e["instance_key"], e["parent_id"]) for e in back["result"]["restored_instance_members"]] == [("c", "c_a_MON"), ("c", "c_a_T_MON_CENTER")]
    added = [("OBSERVATION", "c_a_MON", "ADDED"), ("TARGET", "c_a_T_MON_CENTER", "ADDED")]
    assert changes(back["workspace"]) == {"c": added, "c_a": added, "c_b": []}
    assert {"kind": "OBSERVATION", "id": "c_a_MON"} in group(back["workspace"], "c_a")["members"]
    assert "c_a_MON" in ok(J.workspace_select(back["workspace"], {"instances": ["c_a"]}))["selection"]["observations"]
    rows = [{"kind": k, "id": i, "change": c} for k, i, c in added]
    assert edited_details(replaced(back["workspace"], "c_a", "r1")) == rows
    assert edited_details(replaced(back["workspace"], "c", "c1")) == rows
    first = ok(replaced(back["workspace"], "c_a", "r1", discard=True))
    assert first["result"]["replaced"]["edits_discarded"] == rows
    # as imported (the objects added back at the end of their lists stay there)
    assert objects(first["workspace"]) == objects(ws) and same(first["workspace"], ws, "c_a") == {**ALL, "project": False}
    # discarded back to the revision lacking them: as before they were added back
    assert same(ok(J.workspace_replace_instance(back["workspace"], fold_revision("rm", (), without_monitor=True),
                                                {"instance_key": "c_a", "fixtures": REBIND, "discard_edits": True}))["workspace"], lack) == ALL
    # deleted again: the enclosing instance's deletion, nothing of the nested one
    again = ok(J.workspace_update_project(back["workspace"], {"project": lack["project"]}))
    assert [(e["instance_key"], e["parent_id"]) for e in again["result"]["removed_instance_members"]] == [("c", "c_a_MON"), ("c", "c_a_T_MON_CENTER")]
    assert same(again["workspace"], lack) == ALL
    # added back edited: an edit of c as well
    project = readded(lack, ws, ("observations", "c_a_MON"), ("targets", "c_a_T_MON_CENTER"))
    next(o for o in project["observations"] if o["observation_id"] == "c_a_MON")["aperture"]["radius_mm"] = 7.0
    edited = ok(J.workspace_update_project(lack, {"project": project}))["workspace"]
    assert changes(edited)["c"] == [("OBSERVATION", "c_a_MON", "EDITED"), ("TARGET", "c_a_T_MON_CENTER", "ADDED")]
    reply = ok(replaced(edited, "c", "c1", discard=True))
    assert same(reply["workspace"], ws) == ALL and orphans(reply["workspace"]) == []


def test_going_back_to_the_first_revision_reports_no_member_the_enclosing_record_forgets():
    """Case: c_a replaced by r3 (adds EXTRA) and back by r1: c's added member leaves its record without a
    deletion, so the result does not report it."""

    back = ok(replaced(ok(replaced(imported(), "c_a", "r3"))["workspace"], "c_a", "r1"))
    assert back["result"]["replaced"]["removed_instance_members"] == [] and back["result"]["warnings"] == []
    assert changes(back["workspace"]) == {"c": [], "c_a": [], "c_b": []}


# ------------------------------------------------------------------------------------------------ whose part an id is


def _own_plane_module(plane: tuple) -> dict:
    """The two-fold composed module with a plane of its own whose id lies in its fold a's namespace."""

    oid, at, q, radius = plane
    inner = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    inner = ok(J.workspace_import(inner, fold_mod(), {"instance_key": "b", "fixtures": REBIND,
                                                      "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    project = copy.deepcopy(inner["project"])
    project["observations"].append(screen(oid, at, q, radius_mm=radius))
    inner = ok(J.workspace_update_project(inner, {"project": project}))["workspace"]
    return ok(J.workspace_export_module(inner, {"module_id": "two-folds-own", "title": "Two folds and a plane", "fixtures": FIXTURES}))["module"]


@pytest.mark.parametrize("case", ["added_plane", "retained_fixture"])
def test_a_member_of_the_enclosing_module_in_a_nested_namespace_is_not_the_nested_instance_s(case):
    """The composed module's own plane a_Z (or a_OUT), imported as c_a_Z, held by c only, deleted
    by the user. Replacing c_a by a revision that would create c_a_Z (or retaining fold a's own test plane OUT) is
    refused with ID_CONFLICT, with or without discard_edits: c's deletion is not c_a's to undo."""

    if case == "added_plane":
        module = _own_plane_module(("a_Z", (0.0, 70.0, 0.0), Q, 15.0))
        oid, revision, fixtures = "c_a_Z", fold_revision("rz", (("Z", (0.0, 70.0, 0.0)),)), REBIND
    else:
        module = _own_plane_module(("a_OUT", (0.0, 90.0, 0.0), ALONG_Y, 10.0))
        oid, revision, fixtures = "c_a_OUT", fold_mod(), {"rebind": {"SRC": "LASER"}, "overrides": {"OUT": "RETAIN"}}
    ws = ok(J.workspace_import(host(), module, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    assert holders(ws, oid) == ["c"]
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "observations", oid)}))["workspace"]
    assert changes(deleted) == {"c": [("OBSERVATION", oid, "REMOVED")], "c_a": [], "c_b": []}
    for discard in (False, True):
        reply = J.workspace_replace_instance(deleted, revision, {"instance_key": "c_a", "fixtures": fixtures, "discard_edits": discard})
        assert codes(reply) == ["ID_CONFLICT"]
        problem = reply["error"]["problems"][0]
        assert problem["details"] == {"kind": "OBSERVATION", "parent_id": oid, "deleted_from": "c"}
        assert "the new revision of instance 'c_a' cannot take its id" in problem["message"] and "another instance_key" not in problem["message"]
        assert reply["workspace_digest"] == ok(J.workspace_status(deleted))["status"]["workspace_digest"]
    # not deleted: the Project has it, and c holds it
    reply = J.workspace_replace_instance(ws, revision, {"instance_key": "c_a", "fixtures": fixtures})
    assert codes(reply) == ["ID_CONFLICT"] and reply["error"]["problems"][0]["details"] == {"kind": "OBSERVATION", "parent_id": oid, "held_by": "c"}
    assert "(a member of instance 'c')" in reply["error"]["problems"][0]["message"]
    # c itself replaced with discard_edits brings its plane back
    assert same(ok(J.workspace_replace_instance(deleted, module, {"instance_key": "c", "fixtures": RB, "discard_edits": True}))["workspace"], ws) == ALL


def test_a_nested_instance_s_own_member_in_a_sibling_s_namespace_comes_back_with_discard_edits():
    """Nested instances a and a_b; a's own plane b_X is c_a_b_X. Deleted, then c_a replaced with
    discard_edits: the plane comes back (it is c_a's part, whatever namespace its id lies in)."""

    rbx = fold_revision("rbx", (("b_X", (0.0, 70.0, 0.0)),))
    inner = ok(J.workspace_import(system_with_laser(), rbx, {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    inner = ok(J.workspace_import(inner, fold_mod(), {"instance_key": "a_b", "fixtures": REBIND,
                                                      "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    module = ok(J.workspace_export_module(inner, {"module_id": "folds", "title": "Folds", "fixtures": FIXTURES}))["module"]
    ws = ok(J.workspace_import(host(), module, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    assert holders(ws, "c_a_b_X") == ["c", "c_a"]
    deleted = ok(J.workspace_update_project(ws, {"project": without(ws["project"], "observations", "c_a_b_X")}))["workspace"]
    removed = [{"kind": "OBSERVATION", "id": "c_a_b_X", "change": "REMOVED"}]
    assert edited_details(J.workspace_replace_instance(deleted, rbx, {"instance_key": "c_a", "fixtures": REBIND})) == removed
    reply = ok(J.workspace_replace_instance(deleted, rbx, {"instance_key": "c_a", "fixtures": REBIND, "discard_edits": True}))
    assert reply["result"]["replaced"]["edits_discarded"] == removed
    assert same(reply["workspace"], ws, "c_a") == ALL and orphans(reply["workspace"]) == []


# ------------------------------------------------------------------------------------------------ three levels


@pytest.fixture(scope="module")
def nest() -> dict:
    inner = ok(J.workspace_import(system_with_laser(), composed_module(), {"instance_key": "c", "fixtures": RB}))["workspace"]
    return ok(J.workspace_export_module(inner, {"module_id": "nest", "title": "Nest", "revision": "n1", "fixtures": FIXTURES}))["module"]


def test_an_instance_between_keeps_the_outer_record_of_a_member_its_import_has(nest):
    """Case: d > d_c > d_c_a. d_c replaced on its own by c2 (its fold a has EXTRA): d records
    d_c_a_EXTRA as added, d_c holds it from its import. Then d_c_a replaced on its own by r1 (no EXTRA): d_c records the
    plane as deleted, d_c_a as lacked by its revision, and d keeps its record of it. Added back, the plane is selected
    and moved with d."""

    ws = ok(J.workspace_import(host(), nest, {"instance_key": "d", "fixtures": RB, "placement": PLACE}))["workspace"]
    w1 = ok(J.workspace_replace_instance(ws, modules()["c2"], {"instance_key": "d_c", "fixtures": RB}))["workspace"]
    assert holders(w1, "d_c_a_EXTRA") == ["d", "d_c", "d_c_a"] and digest_of(w1, "d", "d_c_a_EXTRA") == [ADDED_EDIT_DIGEST]
    w2 = ok(replaced(w1, "d_c_a", "r1"))
    assert [(e["instance_key"], e["source_id"], e["parent_id"]) for e in w2["result"]["replaced"]["removed_instance_members"]] == [
        ("d", "c_a_EXTRA", "d_c_a_EXTRA"), ("d_c", "a_EXTRA", "d_c_a_EXTRA")]
    w2 = w2["workspace"]
    assert holders(w2, "d_c_a_EXTRA") == ["d", "d_c", "d_c_a"] and enclosing_records_hold_nested_members(w2)
    assert digest_of(w2, "d_c_a", "d_c_a_EXTRA") == [LACKED_MEMBER_DIGEST] and digest_of(w2, "d", "d_c_a_EXTRA") == [ADDED_EDIT_DIGEST]
    removed = ("OBSERVATION", "d_c_a_EXTRA", "REMOVED")
    assert changes(w2) == {"d": [removed], "d_c": [removed], "d_c_a": [], "d_c_b": []}
    # d_c_a back to r3 (c2's revision of a): as after d_c's replacement (the plane rejoins d_c's group at its import
    # position, where d_c's replacement had appended it)
    back = ok(replaced(w2, "d_c_a", "r3"))["workspace"]
    assert same(back, w1, "d_c_a") == {**ALL, "groups": False}
    assert {g["group_id"]: sorted(m["id"] for m in g["members"]) for g in back["groups"]} == {
        g["group_id"]: sorted(m["id"] for m in g["members"]) for g in w1["groups"]}
    # the plane added back is held by every instance it belongs to: selected and moved with d
    w3 = ok(J.workspace_update_project(w2, {"project": w1["project"]}))["workspace"]
    added = ("OBSERVATION", "d_c_a_EXTRA", "ADDED")
    assert changes(w3) == {"d": [added], "d_c": [added], "d_c_a": [added], "d_c_b": []}
    assert "d_c_a_EXTRA" in ok(J.workspace_select(w3, {"instances": ["d"]}))["selection"]["observations"]
    moved = ok(J.workspace_transform(w3, {"selector": {"instances": ["d"]}, "translation_mm": [0.0, 0.0, 10.0]}))
    assert moved["result"]["placements_updated"] == ["d", "d_c", "d_c_a", "d_c_b"]
    assert any(a["id"] == "d_c_a_EXTRA" for a in moved["result"]["affected"]) and changes(moved["workspace"]) == changes(w3)
    # replacing the outer instances accounts for it
    for key, module in (("d", nest), ("d_c", modules()["c1"])):
        assert edited_details(J.workspace_replace_instance(w3, module, {"instance_key": key, "fixtures": RB})) == [
            {"kind": "OBSERVATION", "id": "d_c_a_EXTRA", "change": "ADDED"}]
        reply = ok(J.workspace_replace_instance(w3, module, {"instance_key": key, "fixtures": RB, "discard_edits": True}))
        assert same(reply["workspace"], ws, "d_c") == ALL and orphans(reply["workspace"]) == []


# ------------------------------------------------------------------------------------------------ the same revision again


def test_replacing_an_unedited_instance_by_its_revision_after_a_rotation_changes_nothing():
    """Case: c rotated as a whole, then an unedited nested instance replaced by its own revision: its
    members keep their exact content (the re-import differs only by rounding), so nothing becomes an edit."""

    ws = imported()
    rotated = ok(J.workspace_transform(ws, {"selector": {"instances": ["c"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0},
                                            "frame": {"instance": "c"}}))["workspace"]
    for key, module in (("c_b", "r1"), ("c_a", "r1"), ("c", "c1")):
        reply = ok(replaced(rotated, key, module))
        assert reply["workspace"]["project"] == rotated["project"], key
        assert same(reply["workspace"], rotated, "c_a", "c_b") == ALL, key
        assert changes(reply["workspace"]) == {"c": [], "c_a": [], "c_b": []}


@pytest.mark.parametrize("revision", ["r3", "r2"])
def test_a_nested_instance_back_at_its_first_revision_after_a_rotation_leaves_the_enclosing_instance_unedited(revision):
    """Case: c_b replaced on its own (by r3, adding a plane, or r2, another coating), c rotated as a whole,
    c_b back at r1: the members r1 shares with the other revision keep their exact rotated content, so c is as its
    import rotated, unedited."""

    ws = imported()
    module = modules()["r3"] if revision == "r3" else fold_mod("0.95", "r2")
    turn = {"selector": {"instances": ["c"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0}, "frame": {"instance": "c"}}
    w = ok(J.workspace_replace_instance(ws, module, {"instance_key": "c_b", "fixtures": REBIND}))["workspace"]
    back = ok(replaced(ok(J.workspace_transform(w, turn))["workspace"], "c_b", "r1"))["workspace"]
    assert changes(back) == {"c": [], "c_a": [], "c_b": []}
    assert objects(back) == objects(ok(J.workspace_transform(ws, turn))["workspace"])


def catalog_composition(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[dict, dict, dict]:
    """Two catalog releases installed (2026-01-01.1 and the active 2026-02-01.1); a module k (a catalog mirror at
    ``position`` and a plane) in revisions k1 and k2 (another mirror added); the composed module ck (k1 as a and b) and
    a workspace with ck imported as c. Returns (k1, k2, ck) as modules and the workspace under ``"ws"`` of the third."""

    from local_helpers import FLAT_MIRROR, FOLD_X_TO_Y, catalog_document, element

    newer = catalog_document()
    newer["catalog_version"] = "2026-02-01.1"
    assert J.install_catalog(json.dumps(catalog_document()).encode("utf-8"))["ok"] is True
    assert J.install_catalog(json.dumps(newer).encode("utf-8"))["ok"] is True

    def kmod(revision: str, extra: bool = False) -> dict:
        comps = [element("M", "mirror", FLAT_MIRROR, position, FOLD_X_TO_Y, catalog_version="2026-02-01.1")]
        if extra:
            comps.append(element("M2", "mirror", FLAT_MIRROR, (0, 0, 40.0), FOLD_X_TO_Y, catalog_version="2026-02-01.1"))
        k = ok(J.workspace_init(project_of(comps, [screen("P", (0.0, 50.0, 0.0), ALONG_Y)], [], []),
                                {"module": {"module_id": "k", "title": "K", "revision": revision}}))["workspace"]
        return ok(J.workspace_export_module(k, {}))["module"]

    k1, k2 = kmod("k1"), kmod("k2", extra=True)
    inner = ok(J.workspace_import(ok(J.workspace_init(project_of([laser("LASER")], [], [], [])))["workspace"], k1, {"instance_key": "a"}))["workspace"]
    inner = ok(J.workspace_import(inner, k1, {"instance_key": "b", "placement": {"position_mm": [0.0, 0.0, 100.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    ck = ok(J.workspace_export_module(inner, {"module_id": "ck", "title": "CK", "fixtures": FIXTURES}))["module"]
    ws = ok(J.workspace_import(ok(J.workspace_init(project_of([laser("LASER")], [], [], [])))["workspace"], ck,
                               {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    return k1, k2, {"module": ck, "ws": ws}


def re_pinned(ws: dict, version: str = "2026-01-01.1") -> dict:
    """``ws`` with every catalog component pinned to ``version`` (a re-pin: no edit)."""

    project = copy.deepcopy(ws["project"])
    for c in project["components"]:
        if c.get("product"):
            c["product"]["catalog_version"] = version
    return ok(J.workspace_update_project(ws, {"project": project}))["workspace"]


def test_a_catalog_re_pin_is_no_edit_of_the_enclosing_instance_through_a_nested_replacement():
    """Every catalog component re-pinned to an older release (not an edit), then a nested instance
    replaced by its own revision, or by another one and back: the enclosing instance stays unedited."""

    k1, k2, composed = catalog_composition()
    pinned = re_pinned(composed["ws"])
    assert changes(pinned) == {"c": [], "c_a": [], "c_b": []}
    unedited = {"c": [], "c_a": [], "c_b": []}
    assert changes(ok(J.workspace_replace_instance(pinned, k1, {"instance_key": "c_a"}))["workspace"]) == unedited
    w = ok(J.workspace_replace_instance(pinned, k2, {"instance_key": "c_a"}))["workspace"]
    assert changes(w) == {"c": [("COMPONENT", "c_a_M2", "ADDED")], "c_a": [], "c_b": []}
    assert changes(ok(J.workspace_replace_instance(w, k1, {"instance_key": "c_a"}))["workspace"]) == unedited


def test_a_catalog_re_pin_after_a_whole_rotation_is_no_edit_through_nested_or_enclosing_replacements():
    """Case: c rotated as a whole, every catalog component re-pinned to the older release, then a nested
    instance replaced by its own revision. Its members keep their exact rotated content and get the pins the catalog
    binding sets (the active release, as before the re-pin), so the enclosing instance stays unedited, and replacing it
    by its own revision is accepted and moves nothing. Replacing c by its own revision right after the re-pin likewise."""

    k1, _k2, composed = catalog_composition((7.3, 11.1, 3.7))
    turn = {"selector": {"instances": ["c"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0}, "frame": {"instance": "c"}}
    turned = ok(J.workspace_transform(composed["ws"], turn))["workspace"]
    pinned = re_pinned(turned)
    unedited = {"c": [], "c_a": [], "c_b": []}
    assert changes(pinned) == unedited
    for key in ("c_a", "c_b"):
        w = ok(J.workspace_replace_instance(pinned, k1, {"instance_key": key}))["workspace"]
        assert w["project"] == turned["project"], key
        assert changes(w) == unedited, key
        back = ok(J.workspace_replace_instance(w, composed["module"], {"instance_key": "c", "fixtures": RB}))["workspace"]
        assert back["project"] == turned["project"] and changes(back) == unedited, key
    direct = ok(J.workspace_replace_instance(pinned, composed["module"], {"instance_key": "c", "fixtures": RB}))["workspace"]
    assert direct["project"] == turned["project"] and changes(direct) == unedited


def test_a_whole_move_between_keeps_a_lacked_member_an_edit_of_the_enclosing_instance():
    """The enclosing instance moved as a whole while its nested instance's revision lacked a member: the member's moved
    import content is not known, so with the first revision back it stays an edit of the enclosing instance (listed,
    and discarded by replacing it, which changes nothing else). The same holds for a member the revision changed."""

    ws = imported()
    w = ok(J.workspace_replace_instance(ws, fold_revision("rm", (), without_monitor=True), {"instance_key": "c_b", "fixtures": REBIND}))["workspace"]
    w = ok(J.workspace_transform(w, {"selector": {"instances": ["c"]}, "translation_mm": [0.0, 0.0, 10.0]}))["workspace"]
    w = ok(replaced(w, "c_b", "r1"))["workspace"]
    assert changes(w) == {"c": [("OBSERVATION", "c_b_MON", "EDITED")], "c_a": [], "c_b": []}
    reply = ok(replaced(w, "c", "c1", discard=True))
    assert reply["result"]["replaced"]["edits_discarded"] == [{"kind": "OBSERVATION", "id": "c_b_MON", "change": "EDITED"}]
    assert reply["workspace"]["project"] == w["project"] and changes(reply["workspace"]) == {"c": [], "c_a": [], "c_b": []}


# ------------------------------------------------------------------------------------------------ a nested instance with no member left


MOVES = {"translate": {"translation_mm": [0.0, 0.0, 10.0]},
         "turn": {"rotation": {"axis": [0, 0, 1], "angle_deg": 90.0}, "frame": {"instance": "c"}}}


def emptied(ws: dict, key: str) -> dict:
    """``ws`` with every component, observation plane and target of instance ``key`` deleted."""

    ids = {e["parent_id"] for e in instance(ws, key)["id_map"] if e["kind"] in ("COMPONENT", "OBSERVATION", "TARGET")}
    project = copy.deepcopy(ws["project"])
    for name, ident in (("components", "instance_id"), ("observations", "observation_id"), ("targets", "target_id")):
        project[name] = [x for x in project[name] if x[ident] not in ids]
    return ok(J.workspace_update_project(ws, {"project": project}))["workspace"]


def import_frame(ws: dict, key: str) -> dict:
    return next(g["frame"] for g in ws["groups"] if g.get("origin") == "IMPORT" and g["instance_key"] == key)


@pytest.mark.parametrize("move", sorted(MOVES))
def test_a_nested_instance_with_no_member_left_moves_with_its_enclosing_instance(move):
    """Every member of c_a deleted, then c moved as a whole (also in a batch, and after
    reopening): c_a's placement and import group frame move with c as when its members are there, so replacing c_a by
    its revision with discard_edits puts its members where c's move puts them. c's record moved with c_a's (both hold
    the deleted members with the same digest), so the members c_a's revision re-creates are c's import content again:
    c is unedited, and replacing it with discard_edits changes no object."""

    ws = imported()
    request = {"selector": {"instances": ["c"]}, **MOVES[move]}
    reference = ok(J.workspace_transform(ws, request))
    empty = emptied(ws, "c_a")
    moved = ok(J.workspace_transform(empty, request))
    assert moved["result"]["placements_updated"] == reference["result"]["placements_updated"] == ["c", "c_a", "c_b"]
    batch = rt(ok(J.workspace_init(rt(ok(J.workspace_batch(empty, {"operations": [{"kind": "transform", "request": request}]}))["workspace"])))["workspace"])
    for w in (moved["workspace"], batch):
        assert [i["placement"] for i in w["instances"]] == [i["placement"] for i in reference["workspace"]["instances"]]
        assert import_frame(w, "c_a") == import_frame(reference["workspace"], "c_a")
    members = [ref for ref in objects(ws) if ref[1].startswith("c_a_")]
    for w in (moved["workspace"], batch):
        back = ok(replaced(w, "c_a", "r1", discard=True))["workspace"]
        assert all(objects(back)[ref] == objects(reference["workspace"])[ref] for ref in members)
        assert changes(back) == {"c": [], "c_a": [], "c_b": []}
        whole = ok(replaced(back, "c", "c1", discard=True))["workspace"]
        assert objects(whole) == objects(back) and changes(whole) == {"c": [], "c_a": [], "c_b": []}


def test_a_nested_instance_emptied_after_its_own_replacement_moves_with_its_enclosing_instance():
    """Case: c_a replaced on its own by rx, whose members all have other ids (c
    records them as added), then rx's members deleted, c moved +10 mm along z, and c_a replaced by rx with
    discard_edits: rx's members are where moving c without the deletion puts them, and every status is the same."""

    project = fold_module()
    names = {"M1": "N1", "MON": "NMON", "T_MON_CENTER": "T_NMON"}
    for item, ident in [(c, "instance_id") for c in project["components"]] + [(o, "observation_id") for o in project["observations"]]:
        item[ident] = names.get(item[ident], item[ident])
    for target in project["targets"]:
        for name in ("target_id", "observation_id"):
            target[name] = names.get(target[name], target[name])
    descriptor = {**FOLD_DESCRIPTOR, "revision": "rx", "requirements": [{"target_id": "T_NMON", "scope": "CARRY_TO_PARENT"}]}
    rx = ok(J.workspace_export_module(ok(J.workspace_init(project, {"module": descriptor}))["workspace"], {}))["module"]
    x = ok(J.workspace_replace_instance(imported(), rx, {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    move = {"selector": {"instances": ["c"]}, "translation_mm": [0.0, 0.0, 10.0]}
    path_a = ok(J.workspace_transform(x, move))["workspace"]
    path_b = ok(J.workspace_transform(emptied(x, "c_a"), move))["workspace"]
    path_b = ok(J.workspace_replace_instance(path_b, rx, {"instance_key": "c_a", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    for ref in (("components", "c_a_N1"), ("observations", "c_a_NMON"), ("targets", "c_a_T_NMON")):
        assert objects(path_b)[ref] == objects(path_a)[ref], ref
    assert instance(path_b, "c_a")["placement"] == instance(path_a, "c_a")["placement"]
    assert changes(path_b) == changes(path_a)


def test_an_emptied_instance_two_levels_down_turns_with_the_outermost_instance(nest):
    """Case: d > d_c > d_c_a. Every member of d_c_a deleted, d turned 90 degrees about z in its own
    frame: d_c_a turns with d, and d_c_a back at its revision with discard_edits is where the turn puts its members."""

    ws = ok(J.workspace_import(host(), nest, {"instance_key": "d", "fixtures": RB, "placement": PLACE}))["workspace"]
    turn = {"selector": {"instances": ["d"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 90.0}, "frame": {"instance": "d"}}
    reference = ok(J.workspace_transform(ws, turn))["workspace"]
    turned = ok(J.workspace_transform(emptied(ws, "d_c_a"), turn))
    assert turned["result"]["placements_updated"] == ["d", "d_c", "d_c_a", "d_c_b"]
    assert [i["placement"] for i in turned["workspace"]["instances"]] == [i["placement"] for i in reference["instances"]]
    back = ok(J.workspace_replace_instance(turned["workspace"], fold_mod(), {"instance_key": "d_c_a", "fixtures": REBIND, "discard_edits": True}))
    members = [ref for ref in objects(ws) if ref[1].startswith("d_c_a_")]
    assert all(objects(back["workspace"])[ref] == objects(reference)[ref] for ref in members)


def test_an_instance_with_no_member_left_stays_where_it_is_when_nothing_encloses_it():
    """A top-level instance whose members were all deleted is in no selection: moving the other instances leaves its
    placement as it is."""

    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "m2", "fixtures": REBIND, "placement": PLACE}))["workspace"]
    empty = emptied(ws, "m1")
    reply = ok(J.workspace_transform(empty, {"selector": {"instances": ["m2"]}, "translation_mm": [0.0, 0.0, 10.0]}))
    assert reply["result"]["placements_updated"] == ["m2"] and instance(reply["workspace"], "m1") == instance(empty, "m1")
