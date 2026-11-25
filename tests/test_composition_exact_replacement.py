"""Composition (review): what a replacement writes, decided by exact identity.

A replacement writes the new revision's content, except for an unedited component or observation plane whose module
object the new revision has with the same digest (``MemberDigestV1.source_digest``): that member keeps its exact
content (after a whole-instance move or rotation a fresh placement differs from it by floating-point rounding alone).
No numeric tolerance decides it, so a revision that changes a value however little reaches the Project, and discarding
edits restores the new revision's content exactly, also after a move smaller than any rounding threshold.

A member of a nested instance's part of an enclosing import that a replacement leaves out is recorded by the innermost
instance whose revision lacks it and by every instance between, at any depth, so replacing that instance by its
original revision brings it back. A definition of the replaced instances that objects outside them use stays, with
its content: never an edit the replacement discards, always reported with those objects, and held by the instances
whose revision lacks it (never left to no instance). A revision with the same definition holds it again as imported;
one defining its id differently is refused."""

from __future__ import annotations

import copy
import functools
import json
import random

import pytest
from composition_helpers import FOLD_DESCRIPTOR, fold_module, ok, project_of, screen, system_with_laser
from local_helpers import ALONG_X, source
from test_composition_deletions import changes, instance, without
from test_composition_isolation_and_retries import REBIND, codes, fold_mod
from test_composition_nested_edits import ALL, PLACE, RB, composed, edited_details, fold_revision, host, imported, modules, orphans, replaced, rt, same

from os_optics.cli import main
from os_optics.composition.document import load_workspace
from os_optics.composition.operations import LACKED_MEMBER_DIGEST, nested_instances
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

FIXTURES = [{"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]
TURN = {"rotation": {"axis": [1, 1, 0], "angle_deg": 37.0}}
NAMES = {"components": ("COMPONENT", "instance_id"), "observations": ("OBSERVATION", "observation_id"), "targets": ("TARGET", "target_id"),
         "optical_models": ("MODEL", "model_id"), "model_materials": ("MATERIAL", "material_id"), "model_coatings": ("COATING", "coating_id")}


def pose(z: float) -> dict:
    return {"position_mm": [0.0, 0.0, z], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}


def objects(ws: dict) -> dict:
    """Every Project object by (kind, id)."""

    return {(kind, item[ident]): item for name, (kind, ident) in NAMES.items() for item in ws["project"].get(name) or []}


def turned(ws: dict, key: str) -> dict:
    return ok(J.workspace_transform(ws, {"selector": {"instances": [key]}, **TURN, "frame": {"instance": key}}))["workspace"]


def holders(ws: dict, parent_id: str) -> list[tuple[str, str, bool]]:
    """(instance key, source id, whether its revision lacks the member) of every record holding ``parent_id``."""

    out = []
    for record in ws["instances"]:
        for entry in record["id_map"]:
            if entry["parent_id"] == parent_id:
                digest = next(d["digest"] for d in record["member_digests"] if d["parent_id"] == parent_id)
                out.append((record["instance_key"], entry["source_id"], digest == LACKED_MEMBER_DIGEST))
    return out


def enclosing_records_hold_nested_members(ws: dict) -> bool:
    model = load_workspace(ws)
    for record in ws["instances"]:
        held = {(e["kind"], e["parent_id"]) for e in record["id_map"]}
        for key in nested_instances(model, record["instance_key"]):
            if not {(e["kind"], e["parent_id"]) for e in instance(ws, key)["id_map"] if not e["shared"]} <= held:
                return False
    return True


# ------------------------------------------------------------------------------------------------ small values


def pulsed(revision: str, duration_s: float, power_w: float, minimum_w: float) -> dict:
    """A pulsed source, a monitor plane and a power target: revisions differ in values far below 1."""

    src = source("SRC", (-50.0, 0.0, 0.0), wavelength_nm=1064.0, waist_mm=0.5, power_w=power_w)
    src["source_model"]["pulse"] = {"repetition_rate_hz": 80000000.0, "pulse_duration_s": duration_s, "duration_definition": "FWHM"}
    project = project_of([src], [screen("MON", (50.0, 0.0, 0.0), ALONG_X, radius_mm=10.0)],
                         [{"target_id": "T_P", "metric": "power_W", "observation_id": "MON", "minimum": minimum_w, "source_id": "SRC"}])
    descriptor = {"module_id": "pulsed", "title": "Pulsed source", "revision": revision}
    return ok(J.workspace_export_module(ok(J.workspace_init(project, {"module": descriptor}))["workspace"], {}))["module"]


@functools.cache
def _pulsed() -> str:
    return json.dumps({"r1": pulsed("r1", 1e-13, 1e-10, 1e-11), "r2": pulsed("r2", 1e-12, 9e-10, 5e-10)})


def pulse_revisions() -> dict:
    return json.loads(_pulsed())


def values(ws: dict, key: str) -> tuple[float, float, float]:
    src = objects(ws)[("COMPONENT", f"{key}_SRC")]["source_model"]
    return src["pulse"]["pulse_duration_s"], src["power_w"], objects(ws)[("TARGET", f"{key}_T_P")]["minimum"]


R2_VALUES = (1e-12, 9e-10, 5e-10)


def test_a_revision_that_changes_small_values_reaches_the_project():
    """A 100 fs -> 1 ps pulse, 100 pW -> 900 pW power and 10 pW -> 500 pW target minimum (all
    below any tolerance relative to 1) are the new revision's content after the replacement, with or without
    discard_edits; the Project equals a fresh import of the new revision."""

    r = pulse_revisions()
    ws = ok(J.workspace_import(system_with_laser(), r["r1"], {"instance_key": "p", "placement": pose(300.0)}))["workspace"]
    fresh = ok(J.workspace_import(system_with_laser(), r["r2"], {"instance_key": "p", "placement": pose(300.0)}))["workspace"]
    for discard in (False, True):
        reply = ok(J.workspace_replace_instance(ws, r["r2"], {"instance_key": "p", "discard_edits": discard}))
        assert reply["workspace"]["project"] == fresh["project"] and values(reply["workspace"], "p") == R2_VALUES
        assert instance(reply["workspace"], "p") == instance(fresh, "p") and changes(reply["workspace"]) == {"p": []}
    # after a rotation: the changed source and target are r2's, exactly as a fresh import at the turned placement writes
    # them; the plane r2 has unchanged keeps its exact turned content
    turn = turned(ws, "p")
    reply = ok(J.workspace_replace_instance(turn, r["r2"], {"instance_key": "p"}))["workspace"]
    fresh_turned = ok(J.workspace_import(system_with_laser(), r["r2"], {"instance_key": "p", "placement": instance(turn, "p")["placement"]}))["workspace"]
    assert values(reply, "p") == R2_VALUES
    for ref in (("COMPONENT", "p_SRC"), ("TARGET", "p_T_P")):
        assert objects(reply)[ref] == objects(fresh_turned)[ref]
    assert objects(reply)[("OBSERVATION", "p_MON")] == objects(turn)[("OBSERVATION", "p_MON")]
    assert changes(reply) == {"p": []}


def bench(module: dict, revision: str) -> dict:
    """A composed bench: the pulsed module as nested ``a``, a fold as ``b``."""

    inner = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "a", "placement": {"position_mm": [0.0, 0.0, 40.0],
                                                                                                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    inner = ok(J.workspace_import(inner, fold_mod(), {"instance_key": "b", "fixtures": REBIND, "placement": pose(100.0)}))["workspace"]
    return ok(J.workspace_export_module(inner, {"module_id": "pulsed-bench", "title": "Pulsed bench", "revision": revision,
                                                "fixtures": FIXTURES}))["module"]


def test_a_small_value_change_reaches_the_project_through_nested_and_enclosing_replacements(tmp_path):
    """The nested instance replaced on its own (its enclosing instance records the change), the
    enclosing instance replaced in a batch and reopened, and the CLI."""

    r = pulse_revisions()
    b1, b2 = bench(r["r1"], "c1"), bench(r["r2"], "c2")
    ws = ok(J.workspace_import(host(), b1, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    nested = ok(J.workspace_replace_instance(ws, r["r2"], {"instance_key": "c_a"}))["workspace"]
    assert values(nested, "c_a") == R2_VALUES
    assert changes(nested) == {"c": [("COMPONENT", "c_a_SRC", "EDITED"), ("TARGET", "c_a_T_P", "EDITED")], "c_a": [], "c_b": []}
    batch = ok(J.workspace_batch(ws, {"operations": [{"kind": "replace_instance", "request": {"instance_key": "c", "fixtures": RB}, "module": b2}]}))
    reopened = rt(ok(J.workspace_init(rt(batch["workspace"])))["workspace"])
    fresh = ok(J.workspace_import(host(), b2, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    assert values(reopened, "c_a") == R2_VALUES and reopened["project"] == fresh["project"]
    assert changes(reopened) == {"c": [], "c_a": [], "c_b": []}
    # the CLI: the file holds r2's values under revision r2
    files = {"ws.json": ok(J.workspace_import(system_with_laser(), r["r1"], {"instance_key": "p", "placement": pose(300.0)}))["workspace"],
             "r2.json": r["r2"]}
    for name, document in files.items():
        (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")
    assert main(["workspace", "replace", str(tmp_path / "ws.json"), "p", str(tmp_path / "r2.json")]) == 0
    saved = json.loads((tmp_path / "ws.json").read_text(encoding="utf-8"))
    assert values(saved, "p") == R2_VALUES and instance(saved, "p")["source"]["revision"] == "r2"


# ------------------------------------------------------------------------------------------------ discarding edits


@pytest.mark.parametrize("step", [2e-7, 1e-12])
def test_discarding_edits_restores_the_new_revision_exactly_after_a_tiny_move(step):
    """The nested instance moved on its own by a sub-micrometre step (an edit of the enclosing
    instance); replacing the enclosing instance with discard_edits lists the moved members and restores them, and the
    nested placement, exactly as imported."""

    ws = imported()
    moved = ok(J.workspace_transform(ws, {"selector": {"instances": ["c_a"]}, "translation_mm": [0.0, 0.0, step]}))["workspace"]
    listed = [{"kind": "COMPONENT", "id": "c_a_M1", "change": "EDITED"}, {"kind": "OBSERVATION", "id": "c_a_MON", "change": "EDITED"}]
    assert edited_details(replaced(moved, "c", "c1")) == listed
    reply = ok(replaced(moved, "c", "c1", discard=True))
    assert reply["result"]["replaced"]["edits_discarded"] == listed
    assert same(reply["workspace"], ws) == ALL and changes(reply["workspace"]) == {"c": [], "c_a": [], "c_b": []}


def test_an_edited_member_never_keeps_its_content_after_a_rotation():
    """An edit as small as a last-digit change of a turned member is discarded too."""

    ws = turned(imported(), "c")
    project = copy.deepcopy(ws["project"])
    mirror = next(c for c in project["components"] if c["instance_id"] == "c_b_M1")
    position = mirror["nominal_pose"]["position_mm"]
    position[0] = position[0] + abs(position[0]) * 2e-16 + 1e-300
    edited = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    assert ("COMPONENT", "c_b_M1", "EDITED") in changes(edited)["c"]
    reply = ok(replaced(edited, "c", "c1", discard=True))["workspace"]
    assert objects(reply) == objects(ws) and changes(reply) == {"c": [], "c_a": [], "c_b": []}


# ------------------------------------------------------------------------------------------------ exact identity


def test_unedited_members_keep_their_bytes_only_through_identical_source_objects():
    """After a rotation: replacing by the same revision changes nothing (also for a nested instance); a revision that
    changes the mirror's coating keeps the mirror (its component object is the same) and writes the new coating."""

    ws = turned(imported(), "c")
    for key, module in (("c", "c1"), ("c_a", "r1"), ("c_b", "r1")):
        reply = ok(replaced(ws, key, module))["workspace"]
        assert reply["project"] == ws["project"], key
    other = ok(J.workspace_replace_instance(ws, fold_mod("0.95", "r2"), {"instance_key": "c_b", "fixtures": REBIND}))["workspace"]
    assert objects(other)[("COMPONENT", "c_b_M1")] == objects(ws)[("COMPONENT", "c_b_M1")]
    coatings = [c for c in other["project"]["model_coatings"] if c["coating_id"].startswith("C_B_")]
    assert len(coatings) == 1 and "R0.95" in coatings[0]["coating_id"]


def test_a_member_edited_in_the_module_is_written_as_its_revision_has_it():
    """A composed module exported after its author moved a nested member: that member's content is not its module
    object placed at the nested placement, so its record names no source digest, and replacing the nested instance by
    its own revision writes the revision's content (an edit of the enclosing instance); the unedited plane keeps its
    turned bytes."""

    inner = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    project = copy.deepcopy(inner["project"])
    next(c for c in project["components"] if c["instance_id"] == "a_M1")["nominal_pose"]["position_mm"][2] = 1.5
    inner = ok(J.workspace_update_project(inner, {"project": project}))["workspace"]
    module = ok(J.workspace_export_module(inner, {"module_id": "baked", "title": "Baked", "fixtures": FIXTURES}))["module"]
    ws = ok(J.workspace_import(host(), module, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    sources = {d["parent_id"]: d.get("source_digest") for d in instance(ws, "c_a")["member_digests"]}
    assert sources["c_a_M1"] is None and sources["c_a_MON"] is not None
    turn = turned(ws, "c")
    reply = ok(J.workspace_replace_instance(turn, fold_mod(), {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    fresh = ok(J.workspace_import(host(), fold_mod(), {"instance_key": "c_a", "fixtures": REBIND, "placement": instance(turn, "c_a")["placement"]}))
    assert objects(reply)[("COMPONENT", "c_a_M1")] == objects(fresh["workspace"])[("COMPONENT", "c_a_M1")]
    assert objects(reply)[("OBSERVATION", "c_a_MON")] == objects(turn)[("OBSERVATION", "c_a_MON")]
    assert ("COMPONENT", "c_a_M1", "EDITED") in changes(reply)["c"] and changes(reply)["c_a"] == []


def test_a_workspace_without_source_digests_gets_the_new_revision_s_content():
    """A workspace written before records named source digests: nothing proves an unedited member's source, so the
    replacement writes the revision's content (as a fresh import at the instance's placement) and records the source
    digests from then on."""

    ws = turned(imported(), "c")
    plain = copy.deepcopy(ws)
    for record in plain["instances"]:
        for digest in record["member_digests"]:
            digest.pop("source_digest", None)
    plain = rt(ok(J.workspace_init(plain))["workspace"])
    reply = ok(replaced(plain, "c", "c1"))["workspace"]
    fresh = ok(J.workspace_import(host(), modules()["c1"], {"instance_key": "c", "fixtures": RB, "placement": instance(ws, "c")["placement"]}))
    assert reply["project"] == fresh["workspace"]["project"]
    assert all(d.get("source_digest") for d in instance(reply, "c")["member_digests"] if d["parent_id"] in ("c_a_M1", "c_b_MON"))
    again = ok(replaced(reply, "c", "c1"))["workspace"]
    assert again["project"] == reply["project"]


# ------------------------------------------------------------------------------------------------ lacked members at depth


@functools.cache
def _family() -> str:
    rm = fold_revision("rm", (), without_monitor=True)
    inner = ok(J.workspace_import(system_with_laser(), modules()["c1"], {"instance_key": "c", "fixtures": RB}))["workspace"]
    n1 = ok(J.workspace_export_module(inner, {"module_id": "nest", "title": "Nest", "revision": "n1", "fixtures": FIXTURES}))["module"]
    return json.dumps({"n1": n1, "c1": modules()["c1"], "c2": modules()["c2"], "c3": composed(rm, "c3"), "r1": modules()["r1"],
                       "r3": modules()["r3"], "rm": rm})


def family() -> dict:
    return json.loads(_family())


def nested_import() -> dict:
    return ok(J.workspace_import(host(), family()["n1"], {"instance_key": "d", "fixtures": RB, "placement": PLACE}))["workspace"]


def put(ws: dict, key: str, name: str, *, discard: bool = False) -> dict:
    return J.workspace_replace_instance(ws, family()[name], {"instance_key": key, "fixtures": RB if name[0] in "cn" else REBIND,
                                                            "discard_edits": discard})


MONITOR = ("d_c_a_MON", "d_c_a_T_MON_CENTER")


def test_a_member_a_middle_revision_lacks_is_recorded_down_to_the_instance_whose_revision_lacks_it():
    """Case: d > d_c > d_c_a. d_c replaced on its own by c3, whose fold a lacks the monitor plane and its
    target: d records them as deleted; d_c and d_c_a (whose revision lacks them) hold them as lacked. Replacing the
    unedited d_c_a by its original revision r1 brings them back (no ID_CONFLICT), with or without discard_edits."""

    ws = nested_import()
    w1 = ok(put(ws, "d_c", "c3"))["workspace"]
    assert holders(w1, "d_c_a_MON") == [("d", "c_a_MON", False), ("d_c", "a_MON", True), ("d_c_a", "MON", True)]
    assert holders(w1, "d_c_a_T_MON_CENTER") == [("d", "c_a_T_MON_CENTER", False), ("d_c", "a_T_MON_CENTER", True), ("d_c_a", "T_MON_CENTER", True)]
    removed = [("OBSERVATION", "d_c_a_MON", "REMOVED"), ("TARGET", "d_c_a_T_MON_CENTER", "REMOVED")]
    assert changes(w1) == {"d": removed, "d_c": [], "d_c_a": [], "d_c_b": []}
    assert enclosing_records_hold_nested_members(w1) and orphans(w1) == []
    added = [("OBSERVATION", "d_c_a_MON", "ADDED"), ("TARGET", "d_c_a_T_MON_CENTER", "ADDED")]
    for discard in (False, True):
        back = ok(put(w1, "d_c_a", "r1", discard=discard))
        assert back["result"]["replaced"]["edits_discarded"] == []
        w2 = back["workspace"]
        assert all(objects(w2)[ref] == objects(ws)[ref] for ref in (("OBSERVATION", MONITOR[0]), ("TARGET", MONITOR[1])))
        # d_c's revision c3 lacks them: added to d_c (and so to d)
        assert changes(w2) == {"d": added, "d_c": added, "d_c_a": [], "d_c_b": []}
        assert enclosing_records_hold_nested_members(w2) and orphans(w2) == []
    # the same replacement again changes nothing; d_c back at c1: as imported
    assert same(ok(put(w1, "d_c", "c3"))["workspace"], w1, "d_c") == ALL
    assert same(ok(put(w1, "d_c", "c1"))["workspace"], ws, "d_c") == ALL
    # added back by hand: a member added to d_c_a and d_c, selected with them; replacing d_c_a is an edit to discard
    project = copy.deepcopy(w1["project"])
    project["observations"].append(copy.deepcopy(objects(ws)[("OBSERVATION", MONITOR[0])]))
    w3 = ok(J.workspace_update_project(w1, {"project": project}))["workspace"]
    plane = [("OBSERVATION", "d_c_a_MON", "ADDED")]
    assert changes(w3) == {"d": [removed[1], *plane], "d_c": plane, "d_c_a": plane, "d_c_b": []}
    for key in ("d_c_a", "d_c", "d"):
        assert "d_c_a_MON" in ok(J.workspace_select(w3, {"instances": [key]}))["selection"]["observations"], key
    assert codes(put(w3, "d_c_a", "r1")) == ["INSTANCE_EDITED"]
    w4 = ok(put(w3, "d_c_a", "r1", discard=True))
    assert w4["result"]["replaced"]["edits_discarded"] == [{"kind": "OBSERVATION", "id": "d_c_a_MON", "change": "ADDED"}]
    assert changes(w4["workspace"]) == {"d": added, "d_c": added, "d_c_a": [], "d_c_b": []}


def test_lacked_records_follow_nested_and_enclosing_replacements():
    """Case: d_c_a replaced by rm first, then d_c by c3 with discard_edits: d_c_a (and d_c) still hold
    the monitor plane as lacked, so d_c_a back at r1 succeeds; d_c back at c1 gives the import exactly."""

    ws = nested_import()
    w1 = ok(put(ws, "d_c_a", "rm"))["workspace"]
    assert holders(w1, "d_c_a_MON") == [("d", "c_a_MON", False), ("d_c", "a_MON", False), ("d_c_a", "MON", True)]
    assert codes(put(w1, "d_c", "c3")) == ["INSTANCE_EDITED"]
    w2 = ok(put(w1, "d_c", "c3", discard=True))["workspace"]
    assert holders(w2, "d_c_a_MON") == [("d", "c_a_MON", False), ("d_c", "a_MON", True), ("d_c_a", "MON", True)]
    w3 = ok(put(w2, "d_c_a", "r1"))["workspace"]
    assert objects(w3)[("OBSERVATION", MONITOR[0])] == objects(ws)[("OBSERVATION", MONITOR[0])]
    for start in (w1, w2, w3):
        back = ok(put(start, "d_c", "c1", discard=True))["workspace"]
        assert same(back, ws, "d_c") == ALL
        assert same(ok(put(start, "d", "n1", discard=True))["workspace"], ws) == ALL


@pytest.mark.parametrize("seed", [5, 23])
def test_replacements_at_every_depth_never_lose_a_lacked_member(seed):
    """A random walk of nested and enclosing replacements (c1 / c2 / c3 for d_c; r1 / r3 / rm for d_c_a and d_c_b),
    deletions and rotations: every replacement with discard_edits succeeds (no ID_CONFLICT), enclosing records hold what
    nested ones hold, no object is left that no instance holds, and d replaced with discard_edits is the import."""

    rng = random.Random(seed)
    ws = nested_import()
    options = {"d_c": ("c1", "c2", "c3"), "d_c_a": ("r1", "r3", "rm"), "d_c_b": ("r1", "r3", "rm")}
    for step in range(14):
        action = rng.choice(["replace", "replace", "delete", "turn"])
        if action == "replace":
            key = rng.choice(sorted(options))
            if key not in {i["instance_key"] for i in ws["instances"]}:
                continue
            reply = put(ws, key, rng.choice(options[key]), discard=True)
            assert reply["ok"], (step, key, reply["error"]["problems"])
            ws = reply["workspace"]
        elif action == "delete":
            planes = sorted(o["observation_id"] for o in ws["project"]["observations"] if o["observation_id"].startswith("d_"))
            if planes:
                plane = rng.choice(planes)
                project = without(ws["project"], "observations", plane)
                project = without(project, "targets", *(t["target_id"] for t in project["targets"] if t.get("observation_id") == plane))
                ws = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
        else:
            ws = turned(ws, rng.choice(["d", "d_c"]))
        assert enclosing_records_hold_nested_members(ws) and orphans(ws) == [], step
        for key in options:
            if key in {i["instance_key"] for i in ws["instances"]}:
                assert put(ws, key, options[key][0], discard=True)["ok"], (step, key)
    placement = instance(ws, "d")["placement"]
    reset = ok(put(ws, "d", "n1", discard=True))["workspace"]
    fresh = ok(J.workspace_import(host(), family()["n1"], {"instance_key": "d", "fixtures": RB, "placement": placement}))["workspace"]
    assert sorted(objects(reset)) == sorted(objects(fresh)) and changes(reset) == {k: [] for k in ("d", "d_c", "d_c_a", "d_c_b")}


# ------------------------------------------------------------------------------------------------ definitions others use


def with_user_mirror(ws: dict, copied: str, z: float) -> dict:
    """``ws`` with the system's own mirror U1, a copy of ``copied`` at ``z``: it uses that member's optical model."""

    project = copy.deepcopy(ws["project"])
    user = copy.deepcopy(next(c for c in project["components"] if c["instance_id"] == copied))
    user["instance_id"] = "U1"
    user["nominal_pose"]["position_mm"] = [0.0, 0.0, z]
    project["components"].append(user)
    return ok(J.workspace_update_project(ws, {"project": project}))["workspace"]


def shared_definitions() -> tuple[dict, str]:
    """The fold imported as m; the system's own mirror U1 uses m's mirror model; the user edits m's coating."""

    ws = with_user_mirror(ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m", "fixtures": REBIND}))["workspace"], "m_M1", 500.0)
    project = copy.deepcopy(ws["project"])
    coating = project["model_coatings"][0]
    coating["bands"][0]["reflectance"] = "0.9"
    return ok(J.workspace_update_project(ws, {"project": project}))["workspace"], coating["coating_id"]


@functools.cache
def _rb() -> str:
    return json.dumps(ok(J.workspace_export_module(ok(J.workspace_init(fold_module(mirror_model_id="fold-mirror-b", reflectance="0.95"),
                                                                       {"module": {**FOLD_DESCRIPTOR, "revision": "rb"}}))["workspace"], {}))["module"])


def rb() -> dict:
    """A fold revision with another mirror model (another model id) and coating."""

    return json.loads(_rb())


USER = [{"kind": "COMPONENT", "id": "U1"}]


def test_a_definition_other_objects_use_stays_with_the_instance_and_is_reported():
    """One rule: the model and the edited coating that the user's mirror uses are no edit
    of m. Replacing m by a revision with another model keeps them as they are: m still holds them (its revision lacks
    them), so no object is left that no instance holds, and the result and the status name the objects that use them
    (the coating through the model). Never listed as discarded. Once nothing outside m uses them they are additions to
    m, which replacing it with discard_edits removes."""

    ws, coating_id = shared_definitions()
    kept = [{"kind": "MODEL", "id": "m_fold-mirror", "used_by": USER}, {"kind": "COATING", "id": coating_id, "used_by": USER}]
    status = ok(J.workspace_status(ws))["status"]["instances"][0]
    assert status["edited"] == [] and status["kept_definitions"] == kept
    for discard in (False, True):
        reply = ok(J.workspace_replace_instance(ws, rb(), {"instance_key": "m", "fixtures": REBIND, "discard_edits": discard}))
        result, new = reply["result"]["replaced"], reply["workspace"]
        assert result["kept_definitions"] == kept and result["edits_discarded"] == []
        for ref in (("COATING", coating_id), ("MODEL", "m_fold-mirror"), ("COMPONENT", "U1")):
            assert objects(new)[ref] == objects(ws)[ref], ref
        assert holders(new, "m_fold-mirror") == [("m", "fold-mirror", True)] and orphans(new) == [("COMPONENT", "U1")]
        assert changes(new) == {"m": []} and ok(J.workspace_status(new))["status"]["instances"][0]["kept_definitions"] == kept
    # the user's mirror deleted: they are additions to m, gone with discard_edits (as if m had been replaced without U1)
    alone = ok(J.workspace_update_project(new, {"project": without(new["project"], "components", "U1")}))["workspace"]
    assert changes(alone) == {"m": [("MODEL", "m_fold-mirror", "ADDED"), ("COATING", coating_id, "ADDED")]}
    assert codes(J.workspace_replace_instance(alone, rb(), {"instance_key": "m", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    clean = ok(J.workspace_replace_instance(alone, rb(), {"instance_key": "m", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    fresh = ok(J.workspace_import(system_with_laser(), rb(), {"instance_key": "m", "fixtures": REBIND}))["workspace"]
    assert objects(clean) == objects(fresh) and changes(clean) == {"m": []} and orphans(clean) == []


def test_a_revision_defining_a_kept_definition_differently_is_refused():
    """One id names one definition: a new revision whose definition with the id of one that objects outside the
    instance use differs from it (r2: another coating for the same model id; r1 while the coating is edited) is refused,
    with or without discard_edits, naming those objects."""

    ws, coating_id = shared_definitions()
    for module, ref in ((fold_mod("0.95", "r2"), ("MODEL", "m_fold-mirror")), (fold_mod(), ("COATING", coating_id))):
        for discard in (False, True):
            reply = J.workspace_replace_instance(ws, module, {"instance_key": "m", "fixtures": REBIND, "discard_edits": discard})
            assert codes(reply) == ["ID_CONFLICT"]
            problem = reply["error"]["problems"][0]
            assert (problem["details"]["kind"], problem["details"]["parent_id"]) == ref and problem["details"]["used_by"] == USER
            assert "component U1" in problem["message"] and "defines it differently" in problem["message"]


def test_replacing_an_instance_by_its_revision_while_other_objects_use_its_definitions_changes_nothing():
    """The user's mirror uses m's model, nothing of m is edited. Replacing m by its own revision
    (with or without discard_edits, also after a rotation) changes nothing: the revision's model and coating are the
    ones the Project has, so m holds them again as imported. m -> rb -> r1 comes back to the start."""

    start = with_user_mirror(ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m", "fixtures": REBIND}))["workspace"], "m_M1", 500.0)
    coating_id = start["project"]["model_coatings"][0]["coating_id"]
    kept = [{"kind": "MODEL", "id": "m_fold-mirror", "used_by": USER}, {"kind": "COATING", "id": coating_id, "used_by": USER}]
    for ws in (start, turned(start, "m")):
        for discard in (False, True):
            reply = ok(J.workspace_replace_instance(ws, fold_mod(), {"instance_key": "m", "fixtures": REBIND, "discard_edits": discard}))
            assert reply["result"]["replaced"]["kept_definitions"] == kept and reply["result"]["replaced"]["edits_discarded"] == []
            assert same(reply["workspace"], ws) == ALL and changes(reply["workspace"]) == {"m": []}
    there = ok(J.workspace_replace_instance(start, rb(), {"instance_key": "m", "fixtures": REBIND}))["workspace"]
    back = ok(J.workspace_replace_instance(there, fold_mod(), {"instance_key": "m", "fixtures": REBIND}))["workspace"]
    assert same(back, start) == ALL and holders(back, "m_fold-mirror") == [("m", "fold-mirror", False)]


def test_a_kept_definition_nothing_uses_any_more_is_never_left_to_no_instance():
    """Case: m2's mirror (an edit of m2) uses m1's model; m1 replaced by rb keeps that model and its
    coating (m1 holds them, reported); m2 replaced with discard_edits then leaves them used by nothing: they are
    additions to m1, and replacing m1 with discard_edits gives what the other order of steps gives."""

    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "m2", "fixtures": REBIND, "placement": pose(200.0)}))["workspace"]
    project = copy.deepcopy(ws["project"])
    next(c for c in project["components"] if c["instance_id"] == "m2_M1")["optical_model_id"] = "m1_fold-mirror"
    ws = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    coating_id = next(c["coating_id"] for c in ws["project"]["model_coatings"] if c["coating_id"].startswith("M1_"))
    first = ok(J.workspace_replace_instance(ws, rb(), {"instance_key": "m1", "fixtures": REBIND}))
    m2_user = [{"kind": "COMPONENT", "id": "m2_M1"}]
    assert first["result"]["replaced"]["kept_definitions"] == [{"kind": "MODEL", "id": "m1_fold-mirror", "used_by": m2_user},
                                                                {"kind": "COATING", "id": coating_id, "used_by": m2_user}]
    second = ok(J.workspace_replace_instance(first["workspace"], fold_mod(), {"instance_key": "m2", "fixtures": REBIND, "discard_edits": True}))
    w = second["workspace"]
    assert orphans(w) == [] and changes(w) == {"m1": [("MODEL", "m1_fold-mirror", "ADDED"), ("COATING", coating_id, "ADDED")], "m2": []}
    w = ok(J.workspace_replace_instance(w, rb(), {"instance_key": "m1", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    other = ok(J.workspace_replace_instance(ws, fold_mod(), {"instance_key": "m2", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    other = ok(J.workspace_replace_instance(other, rb(), {"instance_key": "m1", "fixtures": REBIND}))["workspace"]
    assert objects(w) == objects(other) and changes(w) == changes(other) == {"m1": [], "m2": []} and orphans(w) == []


def test_a_nested_definition_another_object_uses_is_kept_by_nested_and_enclosing_replacements():
    """The system's own mirror uses the model of the nested fold c_a. Replacing c_a on its own, or c by a revision whose
    fold a has another model, keeps the model and its coating, held by the instances whose revision lacks them (c_a,
    and c too in the second case), and reports them. The revisions they came from bring them back as imported: c by c1,
    c_a by r1. A revision defining the model differently is refused."""

    ws = with_user_mirror(imported(), "c_a_M1", 900.0)
    model = next(c for c in ws["project"]["components"] if c["instance_id"] == "U1")["optical_model_id"]
    coating = next(c["coating_id"] for c in ws["project"]["model_coatings"] if c["coating_id"].startswith("C_A_"))
    kept = [{"kind": "MODEL", "id": model, "used_by": USER}, {"kind": "COATING", "id": coating, "used_by": USER}]
    for key in ("c", "c_a"):
        assert next(i for i in ok(J.workspace_status(ws))["status"]["instances"] if i["instance_key"] == key)["kept_definitions"] == kept
    assert next(i for i in ok(J.workspace_status(ws))["status"]["instances"] if i["instance_key"] == "c_b")["kept_definitions"] == []
    alone = ok(J.workspace_replace_instance(ws, rb(), {"instance_key": "c_a", "fixtures": REBIND}))
    assert alone["result"]["replaced"]["kept_definitions"] == kept and orphans(alone["workspace"]) == [("COMPONENT", "U1")]
    assert holders(alone["workspace"], model) == [("c", "a_fold-mirror", False), ("c_a", "fold-mirror", True)]
    whole = ok(J.workspace_replace_instance(ws, composed(rb(), "cb"), {"instance_key": "c", "fixtures": RB}))
    assert whole["result"]["replaced"]["kept_definitions"] == kept and orphans(whole["workspace"]) == [("COMPONENT", "U1")]
    assert holders(whole["workspace"], model) == [("c", "a_fold-mirror", True), ("c_a", "fold-mirror", True)]
    assert objects(whole["workspace"])[("MODEL", model)] == objects(ws)[("MODEL", model)]
    assert enclosing_records_hold_nested_members(alone["workspace"]) and enclosing_records_hold_nested_members(whole["workspace"])
    # back at the revisions they came from: as imported
    assert same(ok(replaced(ws, "c", "c1"))["workspace"], ws) == ALL
    assert same(ok(replaced(alone["workspace"], "c_a", "r1"))["workspace"], ws, "c_a") == ALL
    assert same(ok(replaced(whole["workspace"], "c", "c1"))["workspace"], ws) == ALL
    # a revision whose fold a defines the model differently cannot take its id while U1 uses it
    reply = J.workspace_replace_instance(ws, composed(fold_mod("0.95", "r2"), "c4"), {"instance_key": "c", "fixtures": RB, "discard_edits": True})
    assert codes(reply) == ["ID_CONFLICT"] and reply["error"]["problems"][0]["details"]["used_by"] == USER


@functools.cache
def _cb() -> str:
    return json.dumps(composed(rb(), "cb"))


def cb() -> dict:
    """The two-fold module with its fold a at rb (another mirror model and coating)."""

    return json.loads(_cb())


def put_back(ws: dict, key: str, module: dict, *, discard: bool = False) -> dict:
    return J.workspace_replace_instance(ws, module, {"instance_key": key, "fixtures": RB if key in ("c", "d", "d_c") else REBIND,
                                                     "discard_edits": discard})


def edit_sets(ws: dict) -> dict:
    return {key: sorted(rows) for key, rows in changes(ws).items()}


def freed(ws: dict, key: str, module: dict, copied: str, z: float) -> dict:
    """``ws`` with the user's mirror U1 (a copy of ``copied``), ``key`` replaced by ``module`` (which lacks the model U1
    uses: kept, held as lacked), then U1 deleted: the model and its coating are additions to the instances holding them."""

    w = ok(put_back(with_user_mirror(ws, copied, z), key, module))["workspace"]
    return ok(J.workspace_update_project(w, {"project": without(w["project"], "components", "U1")}))["workspace"]


@pytest.mark.parametrize("revision", ["rb", "r2"])
def test_a_kept_definition_the_project_no_longer_has_leaves_the_enclosing_records(revision):
    """Case: c -> cb keeps c_a's model and coating for the user's U1 (c and c_a hold them as lacked by
    their revisions); U1 deleted; c_a replaced with discard_edits by a revision without that coating (rb: another model
    too; r2: the same model id, another coating). The coating leaves the Project, and the enclosing record releases it
    (no record holds it), so c_a -> r1 is accepted, with or without discard_edits, and gives the objects and edits that
    c_a -> r1 gives without the intermediate step; c -> c1 with discard_edits then gives the fresh import."""

    ws = freed(imported(), "c", cb(), "c_a_M1", 900.0)
    model = "c_a_fold-mirror"
    coating = next(c["coating_id"] for c in ws["project"]["model_coatings"] if c["coating_id"].startswith("C_A_") and "R0.98" in c["coating_id"])
    assert holders(ws, coating) == [("c", "A_NATIVE-METAL_400-700R0.98T0_AOI0-90", True), ("c_a", "NATIVE-METAL_400-700R0.98T0_AOI0-90", True)]
    direct = ok(put_back(ws, "c_a", fold_mod(), discard=True))["workspace"]
    step = ok(put_back(ws, "c_a", rb() if revision == "rb" else fold_mod("0.95", "r2"), discard=True))["workspace"]
    assert ("COATING", coating) not in objects(step) and holders(step, coating) == [] and orphans(step) == []
    if revision == "rb":
        assert holders(step, model) == [] and changes(step) == {"c": [], "c_a": [], "c_b": []}
    else:  # r2's own model with that id: an addition to c (cb's fold lacks it)
        assert holders(step, model) == [("c", "a_fold-mirror", True), ("c_a", "fold-mirror", False)]
    assert enclosing_records_hold_nested_members(step)
    for discard in (False, True):
        back = ok(put_back(step, "c_a", fold_mod(), discard=discard))["workspace"]
        assert objects(back) == objects(direct) and edit_sets(back) == edit_sets(direct)
        fresh = ok(put_back(back, "c", modules()["c1"], discard=True))["workspace"]
        assert objects(fresh) == objects(imported()) and changes(fresh) == {"c": [], "c_a": [], "c_b": []} and orphans(fresh) == []


@functools.cache
def _bench_modules() -> str:
    """dk1 (c at c1) and dkb (c at cb): a module enclosing the two-fold module."""

    out = {}
    for name, module in (("dk1", modules()["c1"]), ("dkb", cb())):
        ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "c", "fixtures": RB}))["workspace"]
        out[name] = ok(J.workspace_export_module(ws, {"module_id": "dk", "title": "DK", "revision": name, "fixtures": FIXTURES}))["module"]
    return json.dumps(out)


def test_kept_definitions_leave_the_records_at_every_level():
    """At depth 3: d -> dkb keeps d_c_a's model for U1 at every level (d, d_c and d_c_a hold it as
    lacked). U1 deleted, then d_c_a -> rb or d_c -> cb with discard_edits: the model and coating leave the Project and
    every enclosing record, nothing is edited, and d_c_a -> r1 is accepted as without the step. When an enclosing
    import has the definition (d at dk1, d_c -> cb), it stays recorded there as deleted (REMOVED for d) and the rows
    below it stay. update_project keeps the rows (the same workspace digest after deleting and re-adding the model);
    the next replacement in the tree releases them."""

    bench = json.loads(_bench_modules())
    start = ok(J.workspace_import(system_with_laser(), bench["dk1"], {"instance_key": "d", "fixtures": RB, "placement": pose(900.0)}))["workspace"]
    ws = freed(start, "d", bench["dkb"], "d_c_a_M1", 2900.0)
    model = "d_c_a_fold-mirror"
    coating = next(c["coating_id"] for c in ws["project"]["model_coatings"] if c["coating_id"].startswith("D_C_A_") and "R0.98" in c["coating_id"])
    assert [h[0] for h in holders(ws, model) if h[2]] == ["d", "d_c", "d_c_a"]
    direct = ok(put_back(ws, "d_c_a", fold_mod(), discard=True))["workspace"]
    for key, module in (("d_c_a", rb()), ("d_c", cb())):
        step = ok(put_back(ws, key, module, discard=True))["workspace"]
        assert holders(step, model) == holders(step, coating) == [] and orphans(step) == []
        assert all(rows == [] for rows in changes(step).values())
        for discard in (False, True):
            back = ok(put_back(step, "d_c_a", fold_mod(), discard=discard))["workspace"]
            assert objects(back) == objects(direct) and edit_sets(back) == edit_sets(direct)
    # d's import has the definition: recorded there as deleted, the rows below stay, d_c_a -> r1 brings it back
    claimed = ok(put_back(freed(start, "d_c", cb(), "d_c_a_M1", 2900.0), "d_c_a", rb(), discard=True))["workspace"]
    assert holders(claimed, model) == [("d", "c_a_fold-mirror", False), ("d_c", "a_fold-mirror", True), ("d_c_a", "fold-mirror", True)]
    assert ("MODEL", model, "REMOVED") in changes(claimed)["d"]
    assert ("MODEL", model) in objects(ok(put_back(claimed, "d_c_a", fold_mod()))["workspace"])
    # deleted by update_project: the records stay (deleting and re-adding gives the same workspace); a replacement releases them
    project = copy.deepcopy(ws["project"])
    project["optical_models"] = [m for m in project["optical_models"] if m["model_id"] != model]
    project["model_coatings"] = [c for c in project["model_coatings"] if c["coating_id"] != coating]
    deleted = ok(J.workspace_update_project(ws, {"project": project}))
    assert [h[0] for h in holders(deleted["workspace"], model)] == ["d", "d_c", "d_c_a"] and deleted["result"]["removed_instance_members"] == []
    again = ok(J.workspace_update_project(deleted["workspace"], {"project": ws["project"]}))
    assert again["workspace_digest"] == ok(J.workspace_status(ws))["status"]["workspace_digest"]
    step = ok(put_back(deleted["workspace"], "d_c_a", rb()))["workspace"]
    assert holders(step, model) == [] and ok(put_back(step, "d_c_a", fold_mod()))["ok"]


# ------------------------------------------------------------------------------------------------ definitions kept inside an enclosing instance


def records(ws: dict) -> list[dict]:
    """Instance records without their fixture outcomes (a nested instance replaced on its own records its request's)."""

    return [{k: v for k, v in i.items() if k != "fixtures"} for i in rt(ws["instances"])]


def spare_fold(revision: str, *, spare: bool = True) -> dict:
    """The fold with a second mirror model ``spare-mirror`` (reflectance 0.9) that nothing in it uses (``spare=False``:
    the fold alone, under another revision)."""

    from composition_helpers import MIRROR, build

    project = fold_module()
    if spare:
        extra = build("mirror.flat", {**MIRROR, "reflectance": "0.9"}, model_id="spare-mirror")
        project["optical_models"].append(extra["optical_model"])
        project["model_coatings"] += [c for c in extra["model_coatings"] or () if c not in project["model_coatings"]]
    return ok(J.workspace_export_module(ok(J.workspace_init(project, {"module": {**FOLD_DESCRIPTOR, "revision": revision}}))["workspace"], {}))["module"]


def kept_bench() -> dict:
    """A composed module exported while its fold m (at rb) keeps the first revision's model and coating for the
    module's own mirror U1, which uses them: m holds them as lacked by its revision."""

    ws = with_user_mirror(ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "m", "fixtures": REBIND}))["workspace"], "m_M1", 500.0)
    ws = ok(J.workspace_replace_instance(ws, rb(), {"instance_key": "m", "fixtures": REBIND}))["workspace"]
    return ok(J.workspace_export_module(ws, {"module_id": "bench", "title": "Bench", "revision": "x1", "fixtures": FIXTURES}))["module"]


def test_a_nested_instance_replaced_by_its_revision_while_it_keeps_a_definition_for_its_enclosing_module():
    """The composed module's fold k_m keeps a definition its revision lacks for the module's
    own mirror k_U1, which lies inside the enclosing instance k. The import records it as the module does (lacked by
    k_m's revision, kept for k_U1), and k holds it as imported: replacing k_m by the revision it has changes nothing
    (objects and records), k stays unedited, and replacing k by its own revision is accepted and changes nothing. Once
    k_U1 no longer uses it, it is a member added to k_m: replacing k_m needs discard_edits, which removes it."""

    x = kept_bench()
    ws = ok(J.workspace_import(system_with_laser(), x, {"instance_key": "k", "fixtures": RB, "placement": pose(300.0)}))["workspace"]
    model, coating = "k_m_fold-mirror", next(c["coating_id"] for c in ws["project"]["model_coatings"] if "R0.98" in c["coating_id"])
    unedited = {"k": [], "k_m": []}
    assert changes(ws) == unedited
    assert holders(ws, model) == [("k", "m_fold-mirror", False), ("k_m", "fold-mirror", True)]
    kept = [{"kind": "MODEL", "id": model, "used_by": [{"kind": "COMPONENT", "id": "k_U1"}]},
            {"kind": "COATING", "id": coating, "used_by": [{"kind": "COMPONENT", "id": "k_U1"}]}]
    assert next(i for i in ok(J.workspace_status(ws))["status"]["instances"] if i["instance_key"] == "k_m")["kept_definitions"] == kept
    for discard in (False, True):
        reply = ok(J.workspace_replace_instance(ws, rb(), {"instance_key": "k_m", "fixtures": REBIND, "discard_edits": discard}))
        assert reply["result"]["replaced"]["kept_definitions"] == kept and reply["result"]["replaced"]["edits_discarded"] == []
        w = reply["workspace"]
        assert w["project"] == ws["project"] and records(w) == records(ws) and rt(w["groups"]) == rt(ws["groups"])
        assert changes(w) == unedited
        again = ok(J.workspace_replace_instance(w, x, {"instance_key": "k", "fixtures": RB}))["workspace"]
        assert again["project"] == ws["project"] and records(again) == records(ws) and changes(again) == unedited
    # k_U1 pointed at the other model: nothing outside k_m uses the kept one, which is then an addition to k_m
    project = copy.deepcopy(ws["project"])
    next(c for c in project["components"] if c["instance_id"] == "k_U1")["optical_model_id"] = "k_m_fold-mirror-b"
    alone = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    assert changes(alone)["k_m"] == [("MODEL", model, "ADDED"), ("COATING", coating, "ADDED")]
    assert codes(J.workspace_replace_instance(alone, rb(), {"instance_key": "k_m", "fixtures": REBIND})) == ["INSTANCE_EDITED"]
    clean = ok(J.workspace_replace_instance(alone, rb(), {"instance_key": "k_m", "fixtures": REBIND, "discard_edits": True}))["workspace"]
    assert ("MODEL", model) not in objects(clean) and changes(clean)["k_m"] == []


def test_a_nested_replacement_that_changes_nothing_while_the_enclosing_module_uses_a_nested_definition():
    """An own mirror of the composed module uses its fold a's spare model, which nothing in a
    uses. Replacing c_a by a revision without that model changes no object (c_a keeps the model for c_OWN): c stays
    unedited, and replacing c by its own revision is accepted (with discard_edits too, discarding nothing) and changes
    no object. An edit of the kept coating is an edit of c, which replacing c with discard_edits restores."""

    ws = ok(J.workspace_import(system_with_laser(), spare_fold("s1"), {"instance_key": "a", "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "b", "fixtures": REBIND, "placement": pose(100.0)}))["workspace"]
    project = copy.deepcopy(ws["project"])
    own = copy.deepcopy(next(c for c in project["components"] if c["instance_id"] == "a_M1"))
    own.update(instance_id="OWN", optical_model_id="a_spare-mirror")
    own["nominal_pose"]["position_mm"] = [0.0, 0.0, 50.0]
    project["components"].append(own)
    ws = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    co1 = ok(J.workspace_export_module(ws, {"module_id": "two-folds", "title": "Two folds", "revision": "co1", "fixtures": FIXTURES}))["module"]
    c = ok(J.workspace_import(system_with_laser(), co1, {"instance_key": "c", "fixtures": RB, "placement": PLACE}))["workspace"]
    unedited = {"c": [], "c_a": [], "c_b": []}
    w = ok(J.workspace_replace_instance(c, spare_fold("s2", spare=False), {"instance_key": "c_a", "fixtures": REBIND}))["workspace"]
    assert w["project"] == c["project"] and changes(w) == unedited
    kept = next(i for i in ok(J.workspace_status(w))["status"]["instances"] if i["instance_key"] == "c_a")["kept_definitions"]
    assert [(k["kind"], k["id"]) for k in kept][0] == ("MODEL", "c_a_spare-mirror")
    coating = next(k["id"] for k in kept if k["kind"] == "COATING")
    for discard in (False, True):
        reply = ok(J.workspace_replace_instance(w, co1, {"instance_key": "c", "fixtures": RB, "discard_edits": discard}))
        assert reply["result"]["replaced"]["edits_discarded"] == [] and reply["workspace"]["project"] == c["project"]
    project = copy.deepcopy(w["project"])
    next(x for x in project["model_coatings"] if x["coating_id"] == coating)["bands"][0]["reflectance"] = "0.8"
    edited = ok(J.workspace_update_project(w, {"project": project}))["workspace"]
    assert changes(edited) == {"c": [("COATING", coating, "EDITED")], "c_a": [], "c_b": []}
    assert edited_details(J.workspace_replace_instance(edited, co1, {"instance_key": "c", "fixtures": RB})) == [
        {"kind": "COATING", "id": coating, "change": "EDITED"}]
    back = ok(J.workspace_replace_instance(edited, co1, {"instance_key": "c", "fixtures": RB, "discard_edits": True}))["workspace"]
    assert back["project"] == c["project"] and changes(back) == unedited
