"""Composition (review): the records of enclosing instances follow nested replacements by exact
identity.

A replacement of a nested instance re-creates each member its revision has from that revision's module object at the
instance's unchanged placement. An enclosing record that holds the member as the nested instance imported it (the same
digest; the nested record names the object it was copied from) takes the re-created content's digest when the object is
the same, also for an edited member the replacement discards. When the revision changes the object or lacks it, the
enclosing record keeps its digest and names the old object and the placement (``nested_revision_digest``), so the
nested instance going back to that object at that placement gives the enclosing instance its import content back, also
after a whole-instance rotation made before (a fresh placement differs from the rotated content by rounding alone).
Nothing else counts: the nested instance moved on its own, the enclosing instance moved as a whole in between, or an
enclosing record holding another import content leaves the enclosing instance edited. Source digests leave the catalog
release pin out, so a revision whose objects differ only by their pin keeps every member exactly."""

from __future__ import annotations

import copy
import functools
import json

import pytest
from composition_helpers import MIRROR, build, laser, ok, place, project_of, screen
from local_helpers import ALONG_Y, FLAT_MIRROR, FOLD_X_TO_Y, element
from test_composition_deletions import changes, instance
from test_composition_exact_replacement import objects
from test_composition_isolation_and_retries import codes
from test_composition_nested_parts import catalog_composition

from os_optics._vendor.shared_contracts.common.hashes import osc1_digest
from os_optics.composition.importing import source_digest
from os_optics.composition.operations import UNDERIVED_MOVED_DIGEST, nested_revision_digest
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LASER_FIX = [{"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]
RB = {"rebind": {"LASER": "LASER"}}
TURN = {"rotation": {"axis": [1, 1, 0], "angle_deg": 37.0}}
AT = (7.3, 11.1, 3.7)  # a general position: a fresh placement after a rotation differs from the rotated one by rounding


def pose(z: float) -> dict:
    return {"position_mm": [0.0, 0.0, z], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}


def host() -> dict:
    return ok(J.workspace_init(project_of([laser("LASER")], [], [], [])))["workspace"]


def kmod(revision: str, at=AT, *, with_mirror: bool = True) -> dict:
    """Module k: a native flat mirror M (optional) and a plane P."""

    mirror = build("mirror.flat", MIRROR, model_id="m-mirror")
    comps = [place(mirror, "M", at, FOLD_X_TO_Y)] if with_mirror else []
    ws = ok(J.workspace_init(project_of(comps, [screen("P", (0.0, 50.0, 0.0), ALONG_Y)], [], [mirror] if with_mirror else []),
                             {"module": {"module_id": "k", "title": "K", "revision": revision}}))["workspace"]
    return ok(J.workspace_export_module(ws, {}))["module"]


def composed(module_id: str, revision: str, parts: list[tuple[str, dict, float, bool]]) -> dict:
    ws = host()
    for key, module, z, rebind in parts:
        ws = ok(J.workspace_import(ws, module, {"instance_key": key, "placement": pose(z), **({"fixtures": RB} if rebind else {})}))["workspace"]
    return ok(J.workspace_export_module(ws, {"module_id": module_id, "title": module_id.upper(), "revision": revision, "fixtures": LASER_FIX}))["module"]


@functools.cache
def _modules() -> str:
    k1, k1x, k1lack = kmod("k1"), kmod("k1x", (7.3, 11.1, 4.7)), kmod("k1lack", with_mirror=False)
    ck1 = composed("ck", "ck1", [("a", k1, 0.0, False), ("b", k1, 100.0, False)])
    ck2 = composed("ck", "ck2", [("a", k1x, 0.0, False), ("b", k1, 100.0, False)])
    ck3 = composed("ck", "ck3", [("a", k1lack, 0.0, False), ("b", k1, 100.0, False)])
    dk1 = composed("dk", "dk1", [("c", ck1, 0.0, True)])
    return json.dumps({"k1": k1, "k1x": k1x, "k1lack": k1lack, "ck1": ck1, "ck2": ck2, "ck3": ck3, "dk1": dk1})


def modules() -> dict:
    return json.loads(_modules())


def put(ws: dict, key: str, name: str, *, discard: bool = False) -> dict:
    return J.workspace_replace_instance(ws, modules()[name], {"instance_key": key, **({"fixtures": RB} if name[0] in "cd" else {}),
                                                             "discard_edits": discard})


def turned(ws: dict, key: str) -> dict:
    return ok(J.workspace_transform(ws, {"selector": {"instances": [key]}, **TURN, "frame": {"instance": key}}))["workspace"]


def c_ws(turn: bool = True) -> dict:
    ws = ok(J.workspace_import(host(), modules()["ck1"], {"instance_key": "c", "fixtures": RB, "placement": pose(300.0)}))["workspace"]
    return turned(ws, "c") if turn else ws


def d_ws(turn: bool = True) -> dict:
    ws = ok(J.workspace_import(host(), modules()["dk1"], {"instance_key": "d", "fixtures": RB, "placement": pose(500.0)}))["workspace"]
    return turned(ws, "d") if turn else ws


def unedited(ws: dict) -> bool:
    return all(not rows for rows in changes(ws).values())


def named(ws: dict) -> list[tuple[str, str]]:
    """(instance, member) of every member digest naming a nested revision."""

    return sorted((i["instance_key"], m["parent_id"]) for i in ws["instances"] for m in i["member_digests"] if m.get("nested_revision_digest"))


def differing(a: dict, b: dict) -> list[tuple[str, str]]:
    x, y = objects(a), objects(b)
    return sorted(ref for ref in set(x) | set(y) if x.get(ref) != y.get(ref))


def edited(ws: dict, member: str, by: float = 1.0) -> dict:
    project = copy.deepcopy(ws["project"])
    next(c for c in project["components"] if c["instance_id"] == member)["nominal_pose"]["position_mm"][2] += by
    return ok(J.workspace_update_project(ws, {"project": project}))["workspace"]


# ------------------------------------------------------------------------------------------------ discarded at depth


@pytest.mark.parametrize("turn", [False, True])
def test_a_nested_discard_leaves_every_enclosing_instance_unedited(turn):
    """Case: c (two instances of k) turned as a whole; c_a_M moved by 1 mm (an edit of c_a and c), then c_a replaced
    by its revision with discard_edits: c is unedited, and replacing c by its own revision is accepted and changes
    nothing. Three levels (d > d_c > d_c_a): the edit discarded at d_c_a or at d_c leaves d unedited. A deleted member
    discarded at the nested instance likewise."""

    ws = c_ws(turn)
    e = edited(ws, "c_a_M")
    assert changes(e) == {"c": [("COMPONENT", "c_a_M", "EDITED")], "c_a": [("COMPONENT", "c_a_M", "EDITED")], "c_b": []}
    assert codes(put(e, "c_a", "k1")) == ["INSTANCE_EDITED"]
    back = ok(put(e, "c_a", "k1", discard=True))
    assert back["result"]["replaced"]["edits_discarded"] == [{"kind": "COMPONENT", "id": "c_a_M", "change": "EDITED"}]
    w = back["workspace"]
    assert unedited(w) and differing(ws, w) == ([("COMPONENT", "c_a_M")] if turn else [])
    again = ok(put(w, "c", "ck1"))["workspace"]
    assert again["project"] == w["project"] and unedited(again)
    gone = ok(J.workspace_update_project(ws, {"project": {**ws["project"], "components": [
        c for c in ws["project"]["components"] if c["instance_id"] != "c_a_M"]}}))["workspace"]
    assert ("COMPONENT", "c_a_M", "REMOVED") in changes(gone)["c"]
    assert unedited(ok(put(gone, "c_a", "k1", discard=True))["workspace"])

    deep = d_ws(turn)
    e = edited(deep, "d_c_a_M")
    for key, name in (("d_c_a", "k1"), ("d_c", "ck1")):
        w = ok(put(e, key, name, discard=True))["workspace"]
        assert unedited(w), key
        again = ok(put(w, "d", "dk1"))["workspace"]
        assert again["project"] == w["project"] and unedited(again), key


def test_a_nested_discard_after_the_enclosing_instance_moved_with_the_edit_leaves_it_unedited():
    """c_a_M moved by 1 mm, then c turned as a whole (c and c_a record the edited member with the same moved-edit
    digest): c_a replaced by its revision with discard_edits re-creates it, and c is unedited. c_a turned on its own
    first (an edit of c), then c turned: c_a's discard leaves c edited, since c_a no longer sits where c's import put it."""

    e = turned(edited(c_ws(False), "c_a_M"), "c")
    assert changes(e)["c"] == changes(e)["c_a"] == [("COMPONENT", "c_a_M", "EDITED")]
    w = ok(put(e, "c_a", "k1", discard=True))["workspace"]
    assert unedited(w)
    again = ok(put(w, "c", "ck1"))["workspace"]
    assert again["project"] == w["project"] and unedited(again)
    alone = turned(turned(c_ws(False), "c_a"), "c")
    w = ok(put(alone, "c_a", "k1", discard=True))["workspace"]
    assert ("COMPONENT", "c_a_M", "EDITED") in changes(w)["c"] and codes(put(w, "c", "ck1")) == ["INSTANCE_EDITED"]


# ------------------------------------------------------------------------------------------------ back at the revision


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("revision", ["k1x", "k1lack"])
def test_a_nested_instance_back_at_its_revision_leaves_the_enclosing_instance_unedited(turn, revision):
    """Case: c turned as a whole; c_a replaced by a revision whose mirror sits 1 mm further (k1x), or that lacks it
    (k1lack): c is edited, and its record names the object and the placement c_a had. c_a back at k1: nothing is
    edited or named, only c_a_M is placed afresh (and only after the rotation), and replacing c by its own revision
    changes nothing. The same in one batch and across a save and reopen."""

    ws = c_ws(turn)
    old_source = next(m["source_digest"] for m in instance(ws, "c_a")["member_digests"] if m["parent_id"] == "c_a_M")
    x = ok(put(ws, "c_a", revision))["workspace"]
    assert changes(x)["c"][0] == ("COMPONENT", "c_a_M", "EDITED" if revision == "k1x" else "REMOVED")
    assert named(x) == [("c", "c_a_M")]
    row = next(m for m in instance(x, "c")["member_digests"] if m["parent_id"] == "c_a_M")
    assert row["nested_revision_digest"] == nested_revision_digest(old_source, instance(ws, "c_a")["placement"])
    assert row["digest"] == next(m["digest"] for m in instance(ws, "c")["member_digests"] if m["parent_id"] == "c_a_M")
    for w in (ok(put(x, "c_a", "k1"))["workspace"], ok(put(ok(J.workspace_init(json.loads(json.dumps(x))))["workspace"], "c_a", "k1"))["workspace"],
              ok(J.workspace_batch(ws, {"operations": [
                  {"kind": "replace_instance", "module": modules()[revision], "request": {"instance_key": "c_a"}},
                  {"kind": "replace_instance", "module": modules()["k1"], "request": {"instance_key": "c_a"}}]}))["workspace"]):
        assert unedited(w) and named(w) == []
        assert differing(ws, w) == ([("COMPONENT", "c_a_M")] if turn else [])
        again = ok(put(w, "c", "ck1"))["workspace"]
        assert again["project"] == w["project"] and unedited(again)


@pytest.mark.parametrize("turn", [False, True])
def test_a_middle_instance_back_at_its_revision_leaves_the_outer_instance_unedited(turn):
    """d turned as a whole; d_c replaced by ck2 (its a at k1x): d is edited by d_c_a_M; d_c back at ck1: nothing is
    edited, and d replaced by its own revision changes nothing. d_c_a replaced by k1x and back likewise."""

    ws = d_ws(turn)
    x = ok(put(ws, "d_c", "ck2"))["workspace"]
    assert changes(x)["d"] == [("COMPONENT", "d_c_a_M", "EDITED")] and named(x) == [("d", "d_c_a_M")]
    for w in (ok(put(x, "d_c", "ck1"))["workspace"], ok(put(ok(put(ws, "d_c_a", "k1x"))["workspace"], "d_c_a", "k1"))["workspace"]):
        assert unedited(w) and named(w) == []
        again = ok(put(w, "d", "dk1"))["workspace"]
        assert again["project"] == w["project"] and unedited(again)


# ------------------------------------------------------------------------------------------------ no false success


@pytest.mark.parametrize("turn", [False, True])
def test_the_enclosing_instance_stays_edited_while_its_import_content_is_not_back(turn):
    """The enclosing record follows only what it can prove: c_a moved on its own after k1x, then back at k1 (c stays
    edited and its replacement is refused); c turned as a whole after k1x (the moved import content is not known: c
    stays edited, the name goes); an edit discarded at k1x (c stays edited, the name stays); k1x, k1lack, k1x (c
    edited). Three levels: d_c at ck2, d_c_a lacking the mirror and then at k1x (ck2's revision): d_c is unedited, d
    is not (its import has k1); d_c back at ck1 clears it."""

    ws = c_ws(turn)
    x = ok(put(ws, "c_a", "k1x"))["workspace"]
    alone = ok(J.workspace_transform(x, {"selector": {"instances": ["c_a"]}, "translation_mm": [0.0, 0.0, 5.0]}))["workspace"]
    w = ok(put(alone, "c_a", "k1"))["workspace"]
    assert ("COMPONENT", "c_a_M", "EDITED") in changes(w)["c"] and codes(put(w, "c", "ck1")) == ["INSTANCE_EDITED"]
    whole = turned(x, "c")
    assert named(whole) == []
    back = ok(put(whole, "c_a", "k1"))["workspace"]
    assert changes(back)["c"] == [("COMPONENT", "c_a_M", "EDITED")] and codes(put(back, "c", "ck1")) == ["INSTANCE_EDITED"]
    assert unedited(ok(put(back, "c", "ck1", discard=True))["workspace"])  # documented: replacing c with discard_edits clears it
    w = ok(put(edited(x, "c_a_M", 2.0), "c_a", "k1x", discard=True))["workspace"]
    assert changes(w)["c"] == [("COMPONENT", "c_a_M", "EDITED")] and named(w) == named(x)
    w = ok(put(ok(put(x, "c_a", "k1lack"))["workspace"], "c_a", "k1x"))["workspace"]
    assert changes(w)["c"] == [("COMPONENT", "c_a_M", "EDITED")]

    deep = ok(put(d_ws(turn), "d_c", "ck2"))["workspace"]
    w = ok(put(ok(put(deep, "d_c_a", "k1lack"))["workspace"], "d_c_a", "k1x"))["workspace"]
    assert changes(w)["d_c"] == [] and changes(w)["d"] == [("COMPONENT", "d_c_a_M", "EDITED")]
    assert codes(put(w, "d", "dk1")) == ["INSTANCE_EDITED"]
    w = ok(put(w, "d_c", "ck1"))["workspace"]
    assert unedited(w) and named(w) == []


def test_the_moved_edit_digest_of_an_earlier_build_proves_nothing():
    """A workspace an earlier build wrote gave every edited member moved with its instance one constant digest: c at
    ck1, c_a replaced by k1x, c_a_M edited, c turned (c and c_a then both hold that digest, for different import
    contents). Replacing c_a by k1x with discard_edits leaves c edited (its import has k1), and further moves keep the
    digest."""

    w = turned(edited(ok(put(c_ws(False), "c_a", "k1x"))["workspace"], "c_a_M"), "c")
    older = copy.deepcopy(w)
    for key in ("c", "c_a"):
        next(m for m in instance(older, key)["member_digests"] if m["parent_id"] == "c_a_M")["digest"] = UNDERIVED_MOVED_DIGEST
    older = ok(J.workspace_init(older))["workspace"]
    back = ok(put(older, "c_a", "k1x", discard=True))["workspace"]
    assert changes(back)["c"] == [("COMPONENT", "c_a_M", "EDITED")] and codes(put(back, "c", "ck1")) == ["INSTANCE_EDITED"]
    again = turned(older, "c")
    assert next(m for m in instance(again, "c")["member_digests"] if m["parent_id"] == "c_a_M")["digest"] == UNDERIVED_MOVED_DIGEST
    # this build's digests differ there: c's record holds k1's content, c_a's k1x's
    rows = {key: next(m for m in instance(w, key)["member_digests"] if m["parent_id"] == "c_a_M")["digest"] for key in ("c", "c_a")}
    assert rows["c"] != rows["c_a"]
    assert changes(ok(put(w, "c_a", "k1x", discard=True))["workspace"])["c"] == [("COMPONENT", "c_a_M", "EDITED")]


def test_a_member_re_created_as_the_enclosing_instance_imported_it_needs_no_name():
    """d_c at ck3 (its a lacks the mirror: d names what d_c had), then d_c_a replaced by k1 on its own: the mirror is
    re-created exactly as d imported it, so d's record names nothing any more, and replacing the unedited d_c_a by its
    own revision changes no record."""

    w = ok(put(ok(put(d_ws(False), "d_c", "ck3"))["workspace"], "d_c_a", "k1"))["workspace"]
    assert named(w) == [] and changes(w)["d_c_a"] == []
    again = ok(put(w, "d_c_a", "k1"))["workspace"]
    assert again["project"] == w["project"] and again["instances"] == w["instances"]


@pytest.mark.parametrize("turn", [False, True])
def test_an_object_added_back_while_the_nested_revision_lacks_it_is_added(turn):
    """c_a at k1lack; the mirror, its model and coating added back as c imported them: c and c_a list them ADDED (c's
    record keeps its import digests), and c_a back at k1 with discard_edits leaves nothing edited."""

    ws = c_ws(turn)
    lack = ok(put(ws, "c_a", "k1lack"))["workspace"]
    project = copy.deepcopy(lack["project"])
    project["components"].append(copy.deepcopy(objects(ws)[("COMPONENT", "c_a_M")]))
    for name, ident in (("optical_models", "model_id"), ("model_coatings", "coating_id")):
        project[name] = [*(project.get(name) or []), *(x for x in ws["project"][name] if x[ident].lower().startswith("c_a_"))]
    back = ok(J.workspace_update_project(lack, {"project": project}))["workspace"]
    rows = changes(back)
    assert rows["c"] == rows["c_a"] and rows["c"][0] == ("COMPONENT", "c_a_M", "ADDED") and {r[2] for r in rows["c"]} == {"ADDED"}
    assert codes(put(back, "c_a", "k1")) == ["INSTANCE_EDITED"]
    assert unedited(ok(put(back, "c_a", "k1", discard=True))["workspace"])


# ------------------------------------------------------------------------------------------------ catalog pins


def test_a_revision_differing_only_by_its_catalog_pin_keeps_every_member_exactly():
    """Case: c rotated as a whole; c_a replaced by k1 exported against the older catalog release (its objects
    differ only by their pin): the catalog binding moves the pins to the active release, every member keeps its exact
    content and nothing is edited. Source digests leave the pin out."""

    k1, _k2, composed_ = catalog_composition(AT)
    ws = turned(composed_["ws"], "c")
    older = copy.deepcopy(k1)
    for component in older["project"]["components"]:
        component["product"]["catalog_version"] = "2026-01-01.1"
    k1_older = ok(J.workspace_export_module(ok(J.workspace_init(older["project"], {"module": older["module"]}))["workspace"], {}))["module"]
    mirror = next(c for c in k1["project"]["components"] if c["instance_id"] == "M")
    older_mirror = next(c for c in k1_older["project"]["components"] if c["instance_id"] == "M")
    assert mirror != older_mirror and source_digest(mirror) == source_digest(older_mirror)
    assert next(m["source_digest"] for m in instance(ws, "c_a")["member_digests"] if m["parent_id"] == "c_a_M") == source_digest(mirror)
    w = ok(J.workspace_replace_instance(ws, k1_older, {"instance_key": "c_a"}))["workspace"]
    assert w["project"] == ws["project"] and unedited(w)
    again = ok(J.workspace_replace_instance(w, composed_["module"], {"instance_key": "c", "fixtures": RB}))["workspace"]
    assert again["project"] == ws["project"] and unedited(again)


def test_a_source_digest_is_the_module_object_without_its_pin():
    item = element("M", "mirror", FLAT_MIRROR, AT, FOLD_X_TO_Y, catalog_version="2026-02-01.1")
    other = copy.deepcopy(item)
    other["product"]["catalog_version"] = "2026-01-01.1"
    plane = screen("P", (0.0, 50.0, 0.0), ALONG_Y)
    assert item != other and source_digest(item) == source_digest(other) == osc1_digest({**item, "product": {"optical_sku": FLAT_MIRROR}})
    assert source_digest(plane) == osc1_digest(plane)
