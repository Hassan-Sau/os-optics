"""Composition of reusable optical layouts: ``os_optics.composition`` and its ``json_api`` calls.

Modules are small native-library layouts (``composition_helpers``; fixture values). Poses are checked against an
independent NumPy computation (rotation matrices from ``math.cos`` / ``math.sin``), never against the composition's own
arithmetic.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
from composition_helpers import (
    FOLD_DESCRIPTOR,
    LENS_DESCRIPTOR,
    build,
    fold_module,
    fold_workspace,
    laser,
    lens_module,
    lens_workspace,
    ok,
    place,
    project_of,
    screen,
    system_with_laser,
    turn_z,
)
from local_helpers import ALONG_X, ALONG_Y, FOLD_X_TO_Y

from os_optics import composition as comp
from os_optics.composition import rotation as rot
from os_optics.local import json_api as J

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------------------------------------------ helpers


def fold_mod() -> dict:
    return ok(J.workspace_export_module(fold_workspace(), FOLD_DESCRIPTOR))["module"]


def lens_mod(focal_length_mm: str = "50") -> dict:
    return ok(J.workspace_export_module(lens_workspace(focal_length_mm), LENS_DESCRIPTOR))["module"]


REBIND = {"rebind": {"SRC": "LASER"}}


def codes(reply: dict) -> list[str]:
    assert reply["ok"] is False, reply
    assert reply["error"]["code"] == "WORKSPACE_OPERATION_REFUSED"
    return [p["code"] for p in reply["error"]["problems"]]


def matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle_matrix(axis, degrees: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(degrees)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def rt(workspace: dict, kind: str, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of an object's design pose (independent of the composition arithmetic)."""

    project = workspace["project"]
    if kind == "COMPONENT":
        pose = next(c for c in project["components"] if c["instance_id"] == object_id)["nominal_pose"]
        return matrix(pose["quaternion_wxyz"]), np.array(pose["position_mm"], dtype=float)
    obs = next(o for o in project["observations"] if o["observation_id"] == object_id)
    r, t = matrix(obs["pose"]["quaternion_wxyz"]), np.array(obs["pose"]["position_mm"], dtype=float)
    if obs.get("device_instance_id"):
        dr, dt = rt(workspace, "COMPONENT", obs["device_instance_id"])
        return dr @ r, dr @ t + dt
    return r, t


def two_fold_system() -> dict:
    ws = system_with_laser()
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "fa", "fixtures": REBIND, "op_id": "imp-fa"}))["workspace"]
    placement = {"position_mm": [0.0, 200.0, 0.0], "quaternion_wxyz": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]}
    return ok(J.workspace_import(ws, fold_mod(), {"instance_key": "fb", "placement": placement, "fixtures": REBIND, "op_id": "imp-fb"}))["workspace"]


# ------------------------------------------------------------------------------------------------ rotation arithmetic


def test_trig_is_exact_at_multiples_of_45_degrees_and_accurate_elsewhere_without_libm(monkeypatch):
    half = math.sqrt(0.5)
    assert rot.cos_sin_deg(0) == (1.0, 0.0) and rot.cos_sin_deg(90) == (0.0, 1.0) and rot.cos_sin_deg(-90) == (0.0, -1.0)
    assert rot.cos_sin_deg(45) == (half, half) and rot.cos_sin_deg(180) == (-1.0, 0.0) and rot.cos_sin_deg(720 + 135) == (-half, half)
    reference = [(a, math.cos(math.radians(a)), math.sin(math.radians(a))) for a in np.linspace(-400.0, 400.0, 3203)]
    monkeypatch.setattr(math, "sin", lambda *_: pytest.fail("math.sin called"))
    monkeypatch.setattr(math, "cos", lambda *_: pytest.fail("math.cos called"))
    worst = max(max(abs(c - rc), abs(s - rs)) for a, rc, rs in reference for c, s in [rot.cos_sin_deg(float(a))])
    assert worst < 2e-15
    q = rot.axis_angle_quaternion((0.0, 0.0, 2.0), 90.0)
    assert q == (half, 0.0, 0.0, half) or abs(sum(c * c for c in q) - 1.0) < 1e-15
    assert rot.signed_permutation(q) == ((1, -1), (0, 1), (2, 1))
    assert rot.rotate(q, (3.0, 5.0, 7.0)) == (-5.0, 3.0, 7.0)  # an exact quarter turn
    with pytest.raises(ValueError):
        rot.axis_angle_quaternion((0.0, 0.0, 0.0), 10.0)


def test_the_composition_arithmetic_uses_no_libm_function():
    """Hashed numbers are made with + - * / and sqrt only: no module of the package calls a libm transcendental."""

    import ast
    from pathlib import Path

    forbidden = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log", "pow", "radians", "degrees", "hypot", "fsum"}
    package = Path(comp.__file__).parent
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                 and node.value.id in ("math", "np", "numpy", "cmath") and node.attr in forbidden}
        assert not calls, (path.name, calls)
    assert "**" not in (package / "rotation.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ init / load


def test_a_project_opens_as_a_workspace_and_round_trips_through_json(tmp_path):
    ws = ok(J.workspace_init(fold_module()))
    assert ws["workspace"]["project"] == comp.load_workspace(fold_module()).project.payload()
    assert set(ws["workspace"]) == {"schema", "schema_version", "project", "instances", "groups", "operations", "validations"}
    composed = two_fold_system()
    path = tmp_path / "system.workspace.json"
    path.write_text(json.dumps(composed), encoding="utf-8")
    reopened = comp.load_workspace(path)
    assert reopened.payload() == composed and reopened.workspace_digest() == comp.load_workspace(composed).workspace_digest()
    assert comp.load_workspace(json.dumps(composed)).workspace_digest() == reopened.workspace_digest()


def test_a_module_descriptor_is_checked_and_missing_models_are_listed():
    bad = copy.deepcopy(FOLD_DESCRIPTOR)
    bad["fixtures"][0]["object_id"] = "NOPE"
    assert codes(J.workspace_init(fold_module(), {"module": bad})) == ["REQUEST_INVALID"]
    broken = fold_module()
    broken["components"][1]["optical_model_id"] = "missing-model"
    broken["observations"].append(screen("ATT", (0, 0, 0), device="GHOST"))
    reply = J.workspace_init(broken)
    assert sorted(codes(reply)) == ["MODEL_MISSING", "REFERENCE_INVALID"]
    missing = next(p for p in reply["error"]["problems"] if p["code"] == "MODEL_MISSING")
    assert missing["details"] == {"model_id": "missing-model"} and missing["location"] == "component M1"
    assert codes(J.workspace_import(system_with_laser(), broken, {"instance_key": "x"})) == ["MODEL_MISSING", "REFERENCE_INVALID"]


# ------------------------------------------------------------------------------------------------ import


def test_import_into_a_non_empty_project_namespaces_ids_and_keeps_existing_content():
    parent = system_with_laser()
    reply = ok(J.workspace_import(parent, fold_mod(), {"instance_key": "fa", "fixtures": REBIND,
                                                        "placement": {"position_mm": [5.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))
    ws, result = reply["workspace"], reply["result"]
    project = ws["project"]
    assert project["components"][0] == parent["project"]["components"][0]  # earlier content untouched, first
    assert [c["instance_id"] for c in project["components"]] == ["LASER", "fa_M1"]
    assert [o["observation_id"] for o in project["observations"]] == ["fa_MON"]
    assert [t["target_id"] for t in project["targets"]] == ["fa_T_MON_CENTER"]
    assert project["targets"][0]["observation_id"] == "fa_MON"
    assert project["components"][1]["optical_model_id"] == "fa_fold-mirror"
    assert [m["model_id"] for m in project["optical_models"]] == ["fa_fold-mirror"]
    assert [c["coating_id"] for c in project["model_coatings"]] == ["FA_NATIVE-METAL_400-700R0.98T0_AOI0-90"]
    assert project["components"][1]["nominal_pose"]["position_mm"] == [5.0, 0.0, 0.0]
    assert {(e["kind"], e["source_id"], e["parent_id"], e["shared"]) for e in result["id_map"]} == {
        ("COMPONENT", "M1", "fa_M1", False), ("OBSERVATION", "MON", "fa_MON", False), ("TARGET", "T_MON_CENTER", "fa_T_MON_CENTER", False),
        ("MODEL", "fold-mirror", "fa_fold-mirror", False), ("COATING", "NATIVE-METAL_400-700R0.98T0_AOI0-90", "FA_NATIVE-METAL_400-700R0.98T0_AOI0-90", False)}
    assert result["fixtures"] == [
        {"object_kind": "COMPONENT", "source_id": "SRC", "action": "REBOUND", "rebound_to": "LASER", "affected_targets": []},
        {"object_kind": "OBSERVATION", "source_id": "OUT", "action": "REMOVED", "affected_targets": []},
        {"object_kind": "TARGET", "source_id": "T_OUT", "action": "REMOVED", "affected_targets": []}]
    assert result["requirements"] == {"carried": ["fa_T_MON_CENTER"], "module_only": [], "removed": []}
    assert ws["groups"] == [{"group_id": "fa", "label": "90 degree fold", "members": [{"kind": "COMPONENT", "id": "fa_M1"},
                                                                                       {"kind": "OBSERVATION", "id": "fa_MON"}],
                             "origin": "IMPORT", "instance_key": "fa", "frame": {"position_mm": [5.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}]
    instance = ws["instances"][0]
    assert instance["source"] == {"module_id": "fold-90", "module_digest": comp.load_module(fold_mod()).module_digest(), "title": "90 degree fold",
                                  "revision": "r1"}
    assert len(instance["member_digests"]) == 5
    assert result["interfaces"][0] == {"port_id": "IN", "role": "INPUT", "point_mm": [-15.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0],
                                       "beam_reference": "LASER"}
    assert result["check"]["checked"] is True and result["check"]["carried_problems"] == []


def test_import_composes_poses_with_a_rotated_placement():
    placement_q = (math.cos(math.radians(15)), 0.0, 0.0, math.sin(math.radians(15)))  # 30 degrees about +Z
    placement = {"position_mm": [10.0, -20.0, 3.0], "quaternion_wxyz": list(placement_q)}
    module = fold_mod()
    ws = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "placement": placement, "fixtures": REBIND}))["workspace"]
    pr, pt = matrix(placement_q), np.array(placement["position_mm"])
    source = comp.load_module(module).project.payload()
    for kind, sid in (("COMPONENT", "M1"), ("OBSERVATION", "MON")):
        sr, st = rt({"project": source}, kind, sid)
        r, t = rt(ws, kind, f"k_{sid}")
        assert np.allclose(r, pr @ sr, atol=1e-12) and np.allclose(t, pr @ st + pt, atol=1e-12)


def test_a_second_instance_owns_its_own_definitions_and_is_independent():
    ws = two_fold_system()
    project = ws["project"]
    assert [c["instance_id"] for c in project["components"]] == ["LASER", "fa_M1", "fb_M1"]
    assert [m["model_id"] for m in project["optical_models"]] == ["fa_fold-mirror", "fb_fold-mirror"]  # identical, still copied
    assert [c["coating_id"] for c in project["model_coatings"]] == ["FA_NATIVE-METAL_400-700R0.98T0_AOI0-90",
                                                                    "FB_NATIVE-METAL_400-700R0.98T0_AOI0-90"]
    fb = next(i for i in ws["instances"] if i["instance_key"] == "fb")
    assert not any(e["shared"] for i in ws["instances"] for e in i["id_map"])
    assert {d["parent_id"] for d in fb["member_digests"]} == {"fb_M1", "fb_MON", "fb_T_MON_CENTER", "fb_fold-mirror",
                                                              "FB_NATIVE-METAL_400-700R0.98T0_AOI0-90"}
    # a different model under the same source id gets its own namespaced copy
    other = fold_module()
    other["optical_models"][0]["model"]["surfaces"][0]["clear_radius_mm"] = "10"
    module = ok(J.workspace_init(other, {"module": FOLD_DESCRIPTOR}))["workspace"]
    third = ok(J.workspace_import(ws, module, {"instance_key": "fc", "fixtures": REBIND,
                                               "placement": {"position_mm": [0, -200, 0], "quaternion_wxyz": [1, 0, 0, 0]}}))
    assert [m["model_id"] for m in third["workspace"]["project"]["optical_models"]] == ["fa_fold-mirror", "fb_fold-mirror", "fc_fold-mirror"]


def test_materials_and_models_are_copied_per_instance_even_when_identical():
    window = {"material": {"constant_index": "1.45", "valid_band_nm": [400, 900], "material_id": "GLASS-A"}, "thickness_mm": "3",
              "diameter_mm": "20"}
    a = build("plate.window", window, model_id="win")
    module = project_of([place(a, "W", (0, 0, 0), ALONG_X)], [], [], [a])
    parent = ok(J.workspace_init(project_of([laser("LASER"), place(a, "W0", (-50, 0, 0), ALONG_X)], [], [], [a])))["workspace"]
    reply = ok(J.workspace_import(parent, module, {"instance_key": "p"}))
    materials = reply["workspace"]["project"]["model_materials"]
    assert [m["material_id"] for m in materials] == ["GLASS-A", "P_GLASS-A"]  # the system's own glass is never the instance's
    assert {(e["kind"], e["parent_id"], e["shared"]) for e in reply["result"]["id_map"] if e["kind"] in ("MATERIAL", "MODEL")} == {
        ("MATERIAL", "P_GLASS-A", False), ("MODEL", "p_win", False)}
    denser = build("plate.window", {**window, "material": {**window["material"], "constant_index": "1.6"}}, model_id="win")
    module2 = project_of([place(denser, "W", (0, 0, 0), ALONG_X)], [], [], [denser])
    reply2 = ok(J.workspace_import(reply["workspace"], module2, {"instance_key": "q", "placement": {"position_mm": [50, 0, 0],
                                                                                                    "quaternion_wxyz": [1, 0, 0, 0]}}))
    project = reply2["workspace"]["project"]
    assert [m["material_id"] for m in project["model_materials"]] == ["GLASS-A", "P_GLASS-A", "Q_GLASS-A"]
    q_model = next(m for m in project["optical_models"] if m["model_id"] == "q_win")
    assert q_model["model"]["media"] == ["Q_GLASS-A"]


def test_namespaced_ids_that_overflow_or_collide_are_refused_with_the_ids():
    long_id = "C" * 126
    module = fold_module()
    module["components"][1]["instance_id"] = long_id
    module["targets"] = []
    reply = J.workspace_import(system_with_laser(), module, {"instance_key": "ab", "fixtures": {"policy": "retain_all"}})
    assert codes(reply) == ["ID_TOO_LONG"]
    problem = reply["error"]["problems"][0]
    assert problem["details"]["source_id"] == long_id and problem["details"]["limit"] == 128
    long_model = fold_module(mirror_model_id="m" * 62)
    assert codes(J.workspace_import(system_with_laser(), long_model, {"instance_key": "ab"})) == ["ID_TOO_LONG"]
    parent = system_with_laser()
    parent["project"]["components"].append({**copy.deepcopy(laser("x_M1")), "nominal_pose": {"position_mm": [0, -300, 0],
                                                                                              "quaternion_wxyz": list(ALONG_X)}})
    reply = J.workspace_import(parent, fold_mod(), {"instance_key": "x", "fixtures": REBIND})
    assert codes(reply) == ["ID_CONFLICT"] and reply["error"]["problems"][0]["details"] == {"kind": "COMPONENT", "parent_id": "x_M1"}
    assert reply["workspace_digest"] == comp.load_workspace(parent).workspace_digest()


def test_fixture_policies_overrides_and_outcomes():
    module = fold_mod()
    keep = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {"policy": "retain_all"}}))
    ids = [c["instance_id"] for c in keep["workspace"]["project"]["components"]]
    assert ids == ["LASER", "k_SRC", "k_M1"]
    assert [f["action"] for f in keep["result"]["fixtures"]] == ["RETAINED"] * 3
    assert keep["result"]["fixtures"][0]["parent_id"] == "k_SRC"
    assert keep["result"]["warnings"] == []  # marked fixtures retained on purpose are not reported as unmarked
    drop = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {
        "policy": "remove_all", "overrides": {"OUT": "RETAIN"}, "unbound_targets": "remove"}}))
    project = drop["workspace"]["project"]
    assert [c["instance_id"] for c in project["components"]] == ["LASER", "k_M1"]
    assert [o["observation_id"] for o in project["observations"]] == ["k_OUT", "k_MON"]
    assert drop["result"]["requirements"]["removed"] == []
    assert codes(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {"overrides": {"M1": "REMOVE"}}})) == ["FIXTURE_INVALID"]
    assert codes(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {"rebind": {"SRC": "NOPE"}}})) == ["FIXTURE_INVALID"]
    assert codes(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {"policy": "everything"}})) == ["REQUEST_INVALID"]


def test_targets_of_a_removed_test_source_must_be_rebound_or_left_out():
    project = fold_module()
    project["targets"].append({"target_id": "T_MON_POWER", "metric": "power_W", "observation_id": "MON", "source_id": "SRC", "minimum": 1e-4,
                               "parameters": [{"key": "path_id", "value": "SRC#0.0"}]})
    module = ok(J.workspace_init(project, {"module": FOLD_DESCRIPTOR}))["workspace"]
    refused = J.workspace_import(system_with_laser(), module, {"instance_key": "k"})
    assert codes(refused) == ["FIXTURE_REBIND_REQUIRED"]
    assert refused["error"]["problems"][0]["details"] == {"targets": [{"target_id": "T_MON_POWER", "fixtures": ["SRC"]}]}
    removed = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": {"unbound_targets": "remove"}}))
    assert removed["result"]["requirements"]["removed"] == ["T_MON_POWER"]
    assert [t["target_id"] for t in removed["workspace"]["project"]["targets"]] == ["k_T_MON_CENTER"]
    rebound = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": REBIND}))
    target = next(t for t in rebound["workspace"]["project"]["targets"] if t["target_id"] == "k_T_MON_POWER")
    assert target["source_id"] == "LASER" and target["parameters"] == [{"key": "path_id", "value": "LASER#0.0"}]
    assert rebound["result"]["fixtures"][0] == {"object_kind": "COMPONENT", "source_id": "SRC", "action": "REBOUND", "rebound_to": "LASER",
                                                "affected_targets": ["k_T_MON_POWER"]}
    assert {w["code"] for w in rebound["result"]["warnings"]} == {"TARGET_REBOUND", "PATH_SELECTOR_RENAMED"}


def test_an_observation_attached_to_a_removed_fixture_is_refused():
    project = fold_module()
    project["observations"].append(screen("CAM", (0, 0, 5.0), device="SRC"))
    module = ok(J.workspace_init(project, {"module": FOLD_DESCRIPTOR}))["workspace"]
    reply = J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": REBIND})
    assert codes(reply) == ["FIXTURE_DEPENDENCY"]
    assert reply["error"]["problems"][0]["details"] == {"observation_id": "CAM", "device_instance_id": "SRC"}


def test_an_unmarked_module_reports_its_source_and_absorbing_plane():
    reply = ok(J.workspace_import(system_with_laser(), fold_module(), {"instance_key": "raw",
                                                                       "placement": {"position_mm": [0, 0, 40], "quaternion_wxyz": [1, 0, 0, 0]}}))
    assert sorted(w["code"] for w in reply["result"]["warnings"]) == ["ABSORBING_PLANE_IMPORTED", "UNMARKED_SOURCE_IMPORTED"]
    assert reply["workspace"]["instances"][0]["source"]["module_id"].startswith("layout-")


def test_an_absorbing_plane_of_an_imported_detector_or_dump_is_not_reported():
    for role in ("detector", "dump"):
        project = fold_module()
        out = next(o for o in project["observations"] if o["observation_id"] == "OUT")
        project["components"].append({"instance_id": "CAM", "optical_role": role, "nominal_pose": copy.deepcopy(out["pose"]),
                                      "custom_part": {"custom_part_key": "camera", "description": "The module's own camera."}})
        out["device_instance_id"] = "CAM"
        out["pose"] = {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
        reply = ok(J.workspace_import(system_with_laser(), project, {"instance_key": "raw"}))
        assert [w["code"] for w in reply["result"]["warnings"]] == ["UNMARKED_SOURCE_IMPORTED"], role
    # attached to an element that is not a terminal instrument, an absorbing plane is still reported
    project = fold_module()
    next(o for o in project["observations"] if o["observation_id"] == "OUT")["device_instance_id"] = "M1"
    reply = ok(J.workspace_import(system_with_laser(), project, {"instance_key": "raw"}))
    assert "ABSORBING_PLANE_IMPORTED" in [w["code"] for w in reply["result"]["warnings"]]


def test_module_only_requirements_stay_with_the_module_and_module_validations_travel_as_module_scope():
    descriptor = copy.deepcopy(FOLD_DESCRIPTOR)
    descriptor["requirements"] = [{"target_id": "T_MON_CENTER", "scope": "MODULE_ONLY"}]
    module_ws = ok(J.workspace_init(fold_module(), {"module": descriptor}))["workspace"]
    module_ws = ok(J.workspace_run(module_ws))["workspace"]
    assert module_ws["validations"][0]["validation_status"] == "PASS"
    module = ok(J.workspace_export_module(module_ws, descriptor))["module"]
    assert len(module["validations"]) == 1
    reply = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": REBIND}))
    assert reply["result"]["requirements"] == {"carried": [], "module_only": ["T_MON_CENTER"], "removed": []}
    copied = reply["workspace"]["instances"][0]["module_validations"]
    assert len(copied) == 1 and copied[0]["scope"] == "MODULE" and copied[0]["instance_key"] == "k"
    assert copied[0]["module_digest"] == comp.load_module(module).module_digest()
    status = ok(J.workspace_status(reply["workspace"]))["status"]
    assert status["instances"][0]["module_validations"][0]["reasons"] == ["MODULE_SCOPE"] and status["current_validation"] is None


def test_catalog_pins_are_aligned_on_the_active_release_or_refused_with_the_skus():
    from local_helpers import FLAT_MIRROR, catalog_document, element

    def module_with(version):
        return project_of([element("M", "mirror", FLAT_MIRROR, (0, 0, 0), FOLD_X_TO_Y, catalog_version=version)], [], [], [])

    parent = ok(J.workspace_init(project_of([laser("LASER"), element("P", "mirror", FLAT_MIRROR, (0, 0, 60), FOLD_X_TO_Y)], [], [], [])))["workspace"]
    refused = J.workspace_import(parent, module_with("2025-01-01.1"), {"instance_key": "k"})
    assert codes(refused) == ["CATALOG_BINDING_CONFLICT"]
    details = refused["error"]["problems"][0]["details"]
    assert details["releases"] == ["2025-01-01.1", "2026-01-01.1"]
    assert {(c["instance_id"], c["optical_sku"]) for c in details["components"]} == {("P", FLAT_MIRROR), ("k_M", FLAT_MIRROR)}
    assert codes(J.workspace_import(parent, module_with("2025-01-01.1"), {"instance_key": "k", "catalog_binding": "refuse"})) == ["CATALOG_BINDING_CONFLICT"]
    # a newer release publishing the same products unchanged: both pins move to it
    newer = catalog_document()
    newer["catalog_version"] = "2026-02-01.1"
    assert J.install_catalog(json.dumps(catalog_document()).encode("utf-8"))["ok"] is True  # the release both were designed with
    assert J.install_catalog(json.dumps(newer).encode("utf-8"))["ok"] is True
    moved = ok(J.workspace_import(parent, module_with("2026-02-01.1"), {"instance_key": "k"}))
    assert {c["product"]["catalog_version"] for c in moved["workspace"]["project"]["components"] if c.get("product")} == {"2026-02-01.1"}
    assert [r["instance_id"] for r in moved["result"]["catalog"]["repinned"]] == ["P"]


# ------------------------------------------------------------------------------------------------ retry / atomicity


def test_retries_replay_the_recorded_result_and_conflicts_are_refused():
    parent = system_with_laser()
    request = {"instance_key": "fa", "fixtures": REBIND, "op_id": "import-1", "expected_base": comp.load_workspace(parent).workspace_digest()}
    first = ok(J.workspace_import(parent, fold_mod(), request))
    again = ok(J.workspace_import(first["workspace"], fold_mod(), request))
    assert again["replayed"] is True and again["workspace"] == first["workspace"]
    assert again["result"]["id_map"] == first["result"]["id_map"]
    assert codes(J.workspace_import(first["workspace"], fold_mod(), {**request, "instance_key": "fz"})) == ["OP_ID_REUSED"]
    stale = J.workspace_import(first["workspace"], fold_mod(), {**request, "op_id": "import-2", "instance_key": "fb"})
    assert codes(stale) == ["BASE_CHANGED"]
    assert stale["error"]["problems"][0]["details"]["current_digest"] == first["workspace_digest"] == stale["workspace_digest"]
    # the same module and placement under the same key, under a new op_id: the existing instance answers, nothing is added
    same = ok(J.workspace_import(first["workspace"], fold_mod(), {"instance_key": "fa", "fixtures": REBIND, "op_id": "import-3"}))
    assert same["replayed"] is True and same["result"]["already_imported"] is True
    assert same["workspace"]["project"] == first["workspace"]["project"] and len(same["workspace"]["instances"]) == 1
    assert codes(J.workspace_import(first["workspace"], lens_mod(), {"instance_key": "fa"})) == ["INSTANCE_KEY_EXISTS"]
    second = ok(J.workspace_import(first["workspace"], fold_mod(), {"instance_key": "fb", "fixtures": REBIND, "op_id": "import-4",
                                                                    "placement": {"position_mm": [0, 0, 50], "quaternion_wxyz": [1, 0, 0, 0]}}))
    assert len(second["workspace"]["instances"]) == 2
    assert ok(J.workspace_history(second["workspace"], {"op_id": "import-1"}))["found"] is True
    assert ok(J.workspace_history(second["workspace"], {"op_id": "never"}))["found"] is False


def test_an_interrupted_response_is_retried_safely():
    """The Agent sends an operation, the host applies and saves it, the response is lost. The Agent retries with the same
    op_id and expected_base against the saved workspace: nothing is applied twice."""

    ws = two_fold_system()
    base = comp.load_workspace(ws).workspace_digest()
    request = {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, 0.0, 12.5], "op_id": "lift-fb", "expected_base": base}
    saved = ok(J.workspace_transform(ws, request))["workspace"]  # applied and saved; the reply never reached the Agent
    retry = ok(J.workspace_transform(saved, request))
    assert retry["replayed"] is True and retry["workspace"] == saved
    assert pose_tuple(saved, "fb_M1")[2] == pose_tuple(ws, "fb_M1")[2] + 12.5  # applied once
    imported = ok(J.workspace_import(saved, fold_mod(), {"instance_key": "fc", "fixtures": REBIND, "op_id": "imp-fc",
                                                         "placement": {"position_mm": [0, 0, -40], "quaternion_wxyz": [1, 0, 0, 0]}}))
    retried = ok(J.workspace_import(imported["workspace"], fold_mod(), {"instance_key": "fc", "fixtures": REBIND, "op_id": "imp-fc",
                                                                        "placement": {"position_mm": [0, 0, -40], "quaternion_wxyz": [1, 0, 0, 0]}}))
    assert retried["replayed"] and len(retried["workspace"]["instances"]) == 3


def pose_tuple(ws: dict, instance_id: str) -> list[float]:
    return next(c for c in ws["project"]["components"] if c["instance_id"] == instance_id)["nominal_pose"]["position_mm"]


def test_a_refused_operation_changes_nothing():
    ws = two_fold_system()
    before = copy.deepcopy(ws)
    module = fold_mod()
    module_before = copy.deepcopy(module)
    reply = J.workspace_transform(ws, {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, -200.0, 0.0],  # onto fa
                                       "require_check": True})
    assert codes(reply) == ["CHECK_PROJECT_REFUSED"]
    assert {p["code"] for p in reply["error"]["problems"][0]["details"]["problems"]} == {"OPTICAL_BODIES_OVERLAP"}
    assert reply["workspace_digest"] == comp.load_workspace(before).workspace_digest()
    assert codes(J.workspace_import(ws, module, {"instance_key": "fc", "fixtures": REBIND, "require_check": True})) == [
        "CHECK_PROJECT_REFUSED"]  # on top of fa
    assert ws == before and module == module_before


def test_an_operation_is_applied_and_the_check_of_its_result_reported():
    """The edit and the check are two results: fb moved onto fa is applied, and the reply says which problems the move
    introduced; the next operation, on a workspace that has them, lists them as carried; moving fa away clears them."""

    ws = two_fold_system()
    onto = ok(J.workspace_transform(ws, {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, -200.0, 0.0], "op_id": "onto"}))
    check = onto["result"]["check"]
    assert check["checked"] is True and check["ok"] is False and check["carried_problems"] == []
    assert {p["code"] for p in check["new_problems"]} == {"OPTICAL_BODIES_OVERLAP"}
    # the problems are the ones a run of the same Project reports
    refused = J.run_project(onto["workspace"]["project"])
    assert refused["ok"] is False and [p["code"] for p in refused["error"]["problems"]] == [p["code"] for p in check["new_problems"]]
    overlapping = onto["workspace"]
    nudged = ok(J.workspace_transform(overlapping, {"selector": {"ids": ["LASER"]}, "translation_mm": [-1.0, 0.0, 0.0]}))
    assert nudged["result"]["check"]["ok"] is False and nudged["result"]["check"]["new_problems"] == []
    assert {p["code"] for p in nudged["result"]["check"]["carried_problems"]} == {"OPTICAL_BODIES_OVERLAP"}
    # require_check refuses a new problem only: an operation on a workspace that already has one is applied
    required = ok(J.workspace_transform(overlapping, {"selector": {"ids": ["LASER"]}, "translation_mm": [-1.0, 0.0, 0.0],
                                                      "require_check": True}))
    assert required["result"]["check"] == nudged["result"]["check"]
    repaired = ok(J.workspace_transform(overlapping, {"selector": {"instances": ["fa"]}, "translation_mm": [0.0, 0.0, 60.0]}))
    assert repaired["result"]["check"] == {"checked": True, "ok": True, "new_problems": [], "carried_problems": []}
    unchanged = ok(J.workspace_group_create(ws, {"group_id": "mirrors", "members": {"ids": ["fa_M1", "fb_M1"]}}))
    assert unchanged["result"]["check"] == {"checked": False, "carried_problems": []}


def test_require_check_is_an_option_of_the_call_not_part_of_the_operation():
    ws = two_fold_system()
    request = {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, -200.0, 0.0], "op_id": "onto"}
    applied = ok(J.workspace_transform(ws, request))
    # the journal's request digest does not hold the option: a retry that adds it is the same operation, answered from the journal
    again = ok(J.workspace_transform(applied["workspace"], {**request, "require_check": True}))
    assert again["replayed"] is True and again["workspace_digest"] == applied["workspace_digest"]
    assert again["result"]["check"]["ok"] is False
    harmless = {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, 0.0, 5.0], "op_id": "lift"}
    digests = [comp.load_workspace(ok(J.workspace_transform(two_fold_system(), {**harmless, **option}))["workspace"]).operation("lift").request_digest
               for option in ({}, {"require_check": False}, {"require_check": True})]
    assert len(set(digests)) == 1
    assert codes(J.workspace_transform(ws, {**request, "require_check": "yes"})) == ["REQUEST_INVALID"]
    assert codes(J.workspace_batch(ws, {"operations": [{"kind": "transform", "request": {
        "selector": {"ids": ["fa_M1"]}, "translation_mm": [1, 0, 0], "require_check": True}}]})) == ["BATCH_INVALID"]


# ------------------------------------------------------------------------------------------------ selection and groups


def test_selection_unions_ids_groups_and_instances_once_each():
    ws = two_fold_system()
    ws = ok(J.workspace_group_create(ws, {"group_id": "mirrors", "members": {"ids": ["fa_M1", "fb_M1"]}}))["workspace"]
    selection = ok(J.workspace_select(ws, {"groups": ["fa", "mirrors"], "instances": ["fb"], "ids": ["LASER", "fa_M1"]}))["selection"]
    assert selection == {"components": ["LASER", "fa_M1", "fb_M1"], "observations": ["fa_MON", "fb_MON"], "attached_observations": [],
                         "count": 5}
    assert ok(J.workspace_select(ws, {"instances": ["fa", "fb"], "kinds": ["OBSERVATION"]}))["selection"]["components"] == []
    assert codes(J.workspace_select(ws, {"ids": ["nope"], "groups": ["none"]})) == ["SELECTION_INVALID", "SELECTION_INVALID"]
    ws["project"]["observations"].append(screen("LASER", (0, -500, 0), ALONG_X))
    assert codes(J.workspace_select(ws, {"ids": ["LASER"]})) == ["SELECTION_AMBIGUOUS"]
    assert ok(J.workspace_select(ws, {"ids": [{"kind": "OBSERVATION", "id": "LASER"}]}))["selection"]["observations"] == ["LASER"]


def test_group_operations_never_change_the_project_and_later_transforms_follow_current_members():
    ws = two_fold_system()
    project = ws["project"]
    steps = [
        ("workspace_group_create", {"group_id": "stage", "label": "Fold stage", "members": {"ids": ["fa_M1", "LASER"]},
                                    "frame": {"component": "fa_M1"}}),
        ("workspace_group_add", {"group_id": "stage", "members": {"instances": ["fb"]}}),
        ("workspace_group_remove", {"group_id": "stage", "members": {"ids": ["LASER"]}}),
        ("workspace_group_dissolve", {"group_id": "fa"}),
        ("workspace_group_set_members", {"group_id": "fb", "members": {"ids": ["fb_M1"]}}),
    ]
    for name, request in steps:
        reply = ok(getattr(J, name)(ws, request))
        assert reply["workspace"]["project"] == project and reply["result"]["check"] == {"checked": False, "carried_problems": []}
        ws = reply["workspace"]
    stage = next(g for g in ws["groups"] if g["group_id"] == "stage")
    assert stage["members"] == [{"kind": "COMPONENT", "id": "fa_M1"}, {"kind": "COMPONENT", "id": "fb_M1"}, {"kind": "OBSERVATION", "id": "fb_MON"}]
    assert [g["group_id"] for g in ws["groups"]] == ["fb", "stage"]
    assert codes(J.workspace_group_create(ws, {"group_id": "stage", "members": {}})) == ["GROUP_EXISTS"]
    assert codes(J.workspace_group_add(ws, {"group_id": "nope", "members": {}})) == ["GROUP_UNKNOWN"]
    moved = ok(J.workspace_transform(ws, {"selector": {"groups": ["stage"]}, "translation_mm": [0, 0, 7.0]}))
    assert sorted(a["id"] for a in moved["result"]["affected"]) == ["fa_M1", "fb_M1", "fb_MON"]
    assert pose_tuple(moved["workspace"], "LASER") == pose_tuple(ws, "LASER")
    assert next(o for o in moved["workspace"]["project"]["observations"] if o["observation_id"] == "fa_MON") == \
        next(o for o in ws["project"]["observations"] if o["observation_id"] == "fa_MON")


def test_regrouping_keeps_hashes_and_optical_results_identical():
    ws = two_fold_system()
    before = ok(J.workspace_run(ws))
    regrouped = ok(J.workspace_group_create(before["workspace"], {"group_id": "both", "members": {"instances": ["fa", "fb"]}}))["workspace"]
    regrouped = ok(J.workspace_group_dissolve(regrouped, {"group_id": "fa"}))["workspace"]
    status = ok(J.workspace_status(regrouped))["status"]
    assert [(v["freshness"], v["reasons"]) for v in status["validations"]] == [("CURRENT", [])] and status["current_validation"] == 0
    after = ok(J.workspace_run(regrouped))
    assert after["summary"]["project_hash"] == before["summary"]["project_hash"]
    assert after["summary"]["engine_input_hash"] == before["summary"]["engine_input_hash"]
    assert after["summary"]["engine_result_hash"] == before["summary"]["engine_result_hash"]
    assert after["summary"]["observations"] == before["summary"]["observations"]
    assert len(after["workspace"]["validations"]) == 1  # the same run is recorded once


# ------------------------------------------------------------------------------------------------ transforms


def test_rotating_a_rotated_layout_keeps_its_internal_geometry_and_leaves_the_rest_in_place():
    ws = two_fold_system()  # fb: placed at (0, 200, 0) turned 90 degrees about +Z
    ws = ok(J.workspace_group_create(ws, {"group_id": "g", "members": {"instances": ["fb"]}, "frame": {"instance": "fb"}}))["workspace"]
    members = [("COMPONENT", "fb_M1"), ("OBSERVATION", "fb_MON")]
    outside = [("COMPONENT", "LASER"), ("COMPONENT", "fa_M1"), ("OBSERVATION", "fa_MON")]
    pivot_local = np.array([3.0, -4.0, 1.5])
    request = {"selector": {"groups": ["g"]}, "rotation": {"axis": [0.2, 0.3, 1.0], "angle_deg": 37.0}, "translation_mm": [1.0, 2.0, -0.5],
               "frame": {"group": "g"}, "pivot": {"point_mm": pivot_local.tolist()}}
    reply = ok(J.workspace_transform(ws, request))
    new = reply["workspace"]
    fr, ft = matrix([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]), np.array([0.0, 200.0, 0.0])
    rd = fr @ axis_angle_matrix([0.2, 0.3, 1.0], 37.0) @ fr.T
    pivot = fr @ pivot_local + ft
    d = fr @ np.array([1.0, 2.0, -0.5])
    for kind, oid in members:
        r0, t0 = rt(ws, kind, oid)
        r1, t1 = rt(new, kind, oid)
        assert np.allclose(r1, rd @ r0, atol=1e-12) and np.allclose(t1, rd @ (t0 - pivot) + pivot + d, atol=1e-9)
    for kind, oid in outside:
        assert rt(new, kind, oid)[1].tolist() == rt(ws, kind, oid)[1].tolist()
        assert np.array_equal(rt(new, kind, oid)[0], rt(ws, kind, oid)[0])
    # internal relative geometry
    (ra0, ta0), (rb0, tb0) = rt(ws, *members[0]), rt(ws, *members[1])
    (ra1, ta1), (rb1, tb1) = rt(new, *members[0]), rt(new, *members[1])
    assert np.allclose(ra0.T @ rb0, ra1.T @ rb1, atol=1e-12) and np.allclose(ra0.T @ (tb0 - ta0), ra1.T @ (tb1 - ta1), atol=1e-9)
    # the instance placement and the group frame moved with it; the target still names the moved plane
    fb = next(i for i in new["instances"] if i["instance_key"] == "fb")
    pr = matrix(fb["placement"]["quaternion_wxyz"])
    assert np.allclose(pr, rd @ fr, atol=1e-12) and reply["result"]["placements_updated"] == ["fb"]
    assert sorted(reply["result"]["group_frames_updated"]) == ["fb", "g"]
    assert next(t for t in new["project"]["targets"] if t["target_id"] == "fb_T_MON_CENTER")["observation_id"] == "fb_MON"
    assert all(abs(sum(c * c for c in c_["nominal_pose"]["quaternion_wxyz"]) - 1.0) < 1e-15 for c_ in new["project"]["components"])
    # a whole-instance move is not an edit of the instance
    assert ok(J.workspace_status(new))["status"]["instances"][1]["edited"] == []


def test_overlapping_selections_move_each_object_once():
    ws = two_fold_system()
    ws = ok(J.workspace_group_create(ws, {"group_id": "both", "members": {"instances": ["fb"]}}))["workspace"]
    once = ok(J.workspace_transform(ws, {"selector": {"instances": ["fb"]}, "translation_mm": [0, 0, 3.0]}))["workspace"]
    overlapping = ok(J.workspace_transform(ws, {"selector": {"instances": ["fb"], "groups": ["fb", "both"], "ids": ["fb_M1", "fb_M1"]},
                                                "translation_mm": [0, 0, 3.0]}))
    assert overlapping["workspace"]["project"] == once["project"]
    assert sorted(a["id"] for a in overlapping["result"]["affected"]) == ["fb_M1", "fb_MON"]


def test_frames_pivots_and_attached_observations():
    ws = two_fold_system()
    ws["project"]["observations"].append(screen("CAM", (0.0, 0.0, -2.0), ALONG_X, device="fa_M1"))
    ws = comp.load_workspace(ws).payload()
    # translation along the component's own local +Z (its face normal)
    reply = ok(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "translation_mm": [0, 0, 2.0], "frame": {"component": "fa_M1"}}))
    r, t = rt(ws, "COMPONENT", "fa_M1")
    assert np.allclose(rt(reply["workspace"], "COMPONENT", "fa_M1")[1], t + r @ np.array([0, 0, 2.0]), atol=1e-12)
    # CAM was not selected: it stays attached and so follows its device anyway (reported as a follower)
    assert reply["result"]["followers"] == [{"observation_id": "CAM", "device_instance_id": "fa_M1", "selected": False}]
    follow = ok(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1", "CAM"]}, "translation_mm": [0, 0, 2.0]}))
    assert follow["result"]["followers"] == [{"observation_id": "CAM", "device_instance_id": "fa_M1", "selected": True}]
    assert [a["id"] for a in follow["result"]["affected"]] == ["fa_M1"]
    assert follow["result"]["selection"]["attached_observations"] == [
        {"observation_id": "CAM", "device_instance_id": "fa_M1", "follows_device": True}]
    # an attached plane selected on its own keeps its attachment and moves in the design frame
    cam0 = rt(ws, "OBSERVATION", "CAM")
    alone = ok(J.workspace_transform(ws, {"selector": {"ids": ["CAM"]}, "translation_mm": [1.0, 0, 0]}))
    cam1 = rt(alone["workspace"], "OBSERVATION", "CAM")
    assert np.allclose(cam1[1], cam0[1] + np.array([1.0, 0, 0]), atol=1e-12) and np.allclose(cam1[0], cam0[0], atol=1e-12)
    assert next(o for o in alone["workspace"]["project"]["observations"] if o["observation_id"] == "CAM")["device_instance_id"] == "fa_M1"
    # quarter turn about the selection centroid
    turn = ok(J.workspace_transform(ws, {"selector": {"instances": ["fb"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 180},
                                         "pivot": "selection_centroid"}))
    assert turn["result"]["pivot_mm"] == [-25.0, 200.0, 0.0]
    m1 = rt(turn["workspace"], "COMPONENT", "fb_M1")[1]
    assert m1.tolist() == [-50.0, 200.0, 0.0]
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 5},
                                            "frame": {"group": "nope"}})) == ["FRAME_INVALID"]
    ws2 = ok(J.workspace_group_create(ws, {"group_id": "frameless", "members": {"ids": ["fa_M1"]}}))["workspace"]
    assert codes(J.workspace_transform(ws2, {"selector": {"ids": ["fa_M1"]}, "translation_mm": [1, 0, 0], "frame": {"group": "frameless"}})) == ["FRAME_INVALID"]
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "rotation": {"axis": [0, 0, 0], "angle_deg": 5}})) == ["ROTATION_INVALID"]
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "rotation": {"quaternion_wxyz": [0, 0, 0, 0]}})) == ["ROTATION_INVALID"]
    # a quaternion of any non-zero length is the rotation of its direction
    by_q = ok(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "rotation": {"quaternion_wxyz": [2, 0, 0, 2]}}))
    by_axis = ok(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 90}}))
    for a, b in zip(rt(by_q["workspace"], "COMPONENT", "fa_M1"), rt(by_axis["workspace"], "COMPONENT", "fa_M1")):
        assert np.allclose(a, b, rtol=0, atol=1e-15)
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "translation_mm": [1, 0, 0], "pivot": {"component": "x"}})) == ["PIVOT_INVALID"]
    assert codes(J.workspace_transform(ws, {"selector": {}, "translation_mm": [1, 0, 0]})) == ["SELECTION_EMPTY"]
    assert codes(J.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}})) == ["REQUEST_INVALID"]


def test_a_batch_may_pass_through_overlapping_states():
    ws = two_fold_system()
    # with the check required, moving fb onto fa alone is refused; moving fb onto fa and then fa away in one batch is accepted
    assert codes(J.workspace_transform(ws, {"selector": {"instances": ["fb"]}, "translation_mm": [0, -200, 0],
                                            "require_check": True})) == ["CHECK_PROJECT_REFUSED"]
    reply = ok(J.workspace_batch(ws, {"op_id": "swap", "require_check": True, "operations": [
        {"kind": "transform", "request": {"selector": {"instances": ["fb"]}, "translation_mm": [0, -200, 0]}},
        {"kind": "transform", "request": {"selector": {"instances": ["fa"]}, "translation_mm": [0, 0, 60]}},
    ]}))
    assert [r["kind"] for r in reply["result"]["results"]] == ["transform", "transform"]
    assert reply["result"]["check"] == {"checked": True, "ok": True, "new_problems": [], "carried_problems": []}
    assert codes(J.workspace_batch(ws, {"require_check": True, "operations": [
        {"kind": "transform", "request": {"selector": {"instances": ["fb"]}, "translation_mm": [0, -200, 0]}}]})) == ["CHECK_PROJECT_REFUSED"]
    assert [o["op_id"] for o in reply["workspace"]["operations"]][-1] == "swap"
    assert codes(J.workspace_batch(ws, {"operations": [{"kind": "transform", "request": {"selector": {"ids": ["fa_M1"]},
                                                                                         "translation_mm": [1, 0, 0], "op_id": "x"}}]})) == ["BATCH_INVALID"]
    failing = J.workspace_batch(ws, {"operations": [
        {"kind": "transform", "request": {"selector": {"instances": ["fb"]}, "translation_mm": [0, 0, 1]}},
        {"kind": "group_dissolve", "request": {"group_id": "nope"}}]})
    assert codes(failing) == ["GROUP_UNKNOWN"] and "operations[1]" in failing["error"]["problems"][0]["message"]


# ------------------------------------------------------------------------------------------------ freshness


def test_validation_freshness_follows_physics_and_targets_not_organization():
    ws = two_fold_system()
    ws = ok(J.workspace_run(ws))["workspace"]
    moved = ok(J.workspace_transform(ws, {"selector": {"ids": ["fb_MON"]}, "translation_mm": [0, 0, 0.5]}))["workspace"]
    status = ok(J.workspace_status(moved))["status"]
    assert [(v["freshness"], v["reasons"]) for v in status["validations"]] == [("HISTORICAL", ["PHYSICS_CHANGED"])]
    assert status["current_validation"] is None
    project = copy.deepcopy(ws["project"])
    project["targets"][0]["tolerance_abs"] = 0.25
    retargeted = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    assert [(v["freshness"], v["reasons"]) for v in ok(J.workspace_status(retargeted))["status"]["validations"]] == [
        ("HISTORICAL", ["TARGETS_CHANGED"])]
    rerun = ok(J.workspace_run(moved))["workspace"]
    rows = ok(J.workspace_status(rerun))["status"]["validations"]
    assert [r["freshness"] for r in rows] == ["HISTORICAL", "CURRENT"]


def test_a_run_made_elsewhere_is_recorded_against_this_project_only():
    ws = two_fold_system()
    run = J.run_project(ws["project"])
    recorded = ok(J.workspace_record_validation(ws, {"local_validation": run["local_validation"]}))
    assert recorded["freshness"] == {"freshness": "CURRENT", "reasons": []}
    assert recorded["validation"]["engine_input_hash"] == run["summary"]["engine_input_hash"]
    assert recorded["validation"]["design_projection_hash"] == run["manifest"].get("design_projection_hash", recorded["validation"]["design_projection_hash"])
    other = ok(J.workspace_transform(ws, {"selector": {"ids": ["fb_M1"]}, "translation_mm": [0, 0, 1]}))["workspace"]
    assert codes(J.workspace_record_validation(other, {"local_validation": run["local_validation"]})) == ["VALIDATION_PROJECT_MISMATCH"]


# ------------------------------------------------------------------------------------------------ replace / update / isolation


def test_replace_instance_keeps_placement_order_and_group_membership():
    ws = system_with_laser()
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "fold", "fixtures": REBIND}))["workspace"]
    placement = {"position_mm": [0.0, 120.0, 0.0], "quaternion_wxyz": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]}
    ws = ok(J.workspace_import(ws, lens_mod("50"), {"instance_key": "foc", "placement": placement, "fixtures": REBIND}))["workspace"]
    ws = ok(J.workspace_group_create(ws, {"group_id": "train", "members": {"ids": ["fold_M1", "foc_L1"]}}))["workspace"]
    before_order = [c["instance_id"] for c in ws["project"]["components"]]
    reply = ok(J.workspace_replace_instance(ws, lens_mod("60"), {"instance_key": "foc", "fixtures": REBIND}))
    new = reply["workspace"]
    assert [c["instance_id"] for c in new["project"]["components"]] == before_order
    assert [i["instance_key"] for i in new["instances"]] == ["fold", "foc"]
    lens_model = next(m for m in new["project"]["optical_models"] if m["model_id"] == "foc_focus-lens")
    assert lens_model["model"]["focal_length_mm"] == "60"
    assert next(i for i in new["instances"] if i["instance_key"] == "foc")["placement"] == placement
    assert next(g for g in new["groups"] if g["group_id"] == "train")["members"] == [{"kind": "COMPONENT", "id": "foc_L1"},
                                                                                     {"kind": "COMPONENT", "id": "fold_M1"}]
    assert next(g for g in new["groups"] if g["group_id"] == "foc")["members"] == next(g for g in ws["groups"] if g["group_id"] == "foc")["members"]
    assert reply["result"]["replaced"]["previous_module_digest"] == comp.load_module(lens_mod("50")).module_digest()
    # an edited instance is refused unless its edits are discarded
    edited = ok(J.workspace_transform(new, {"selector": {"ids": ["foc_FOCUS"]}, "translation_mm": [0, 0, 0.1]}))["workspace"]
    refused = J.workspace_replace_instance(edited, lens_mod("50"), {"instance_key": "foc", "fixtures": REBIND})
    assert codes(refused) == ["INSTANCE_EDITED"]
    assert refused["error"]["problems"][0]["details"]["edited"] == [{"kind": "OBSERVATION", "id": "foc_FOCUS", "change": "EDITED"}]
    discarded = ok(J.workspace_replace_instance(edited, lens_mod("50"), {"instance_key": "foc", "fixtures": REBIND, "discard_edits": True}))
    assert discarded["workspace"]["project"] == ws["project"]
    assert codes(J.workspace_replace_instance(ws, lens_mod(), {"instance_key": "none"})) == ["INSTANCE_UNKNOWN"]


def test_source_and_instances_are_isolated():
    module = fold_mod()
    pristine = copy.deepcopy(module)
    ws = two_fold_system()
    project = copy.deepcopy(ws["project"])
    fa = next(c for c in project["components"] if c["instance_id"] == "fa_M1")
    fa["nominal_pose"]["position_mm"] = [0.0, 0.0, 1.0]
    edited = ok(J.workspace_update_project(ws, {"project": project}))["workspace"]
    status = ok(J.workspace_status(edited))["status"]
    edits = {i["instance_key"]: i["edited"] for i in status["instances"]}
    assert edits == {"fa": [{"kind": "COMPONENT", "id": "fa_M1", "change": "EDITED"}], "fb": []}
    assert next(c for c in edited["project"]["components"] if c["instance_id"] == "fb_M1") == \
        next(c for c in ws["project"]["components"] if c["instance_id"] == "fb_M1")
    assert module == pristine  # the source module is never touched
    # removing an instance member drops it from groups and the instance record
    project = copy.deepcopy(ws["project"])
    project["targets"] = [t for t in project["targets"] if t["observation_id"] != "fb_MON"]
    project["observations"] = [o for o in project["observations"] if o["observation_id"] != "fb_MON"]
    pruned = ok(J.workspace_update_project(ws, {"project": project}))
    assert {(m["group_id"], m["id"]) for m in pruned["result"]["removed_group_members"]} == {("fb", "fb_MON")}
    assert [w["code"] for w in pruned["result"]["warnings"]] == ["INSTANCE_MEMBERS_REMOVED"]


# ------------------------------------------------------------------------------------------------ export / re-import


def test_a_composed_workspace_exports_and_reimports_with_its_organization_and_provenance():
    ws = two_fold_system()
    ws = ok(J.workspace_group_create(ws, {"group_id": "mirrors", "label": "Both mirrors", "members": {"ids": ["fa_M1", "fb_M1"]},
                                          "frame": {"component": "fa_M1"}}))["workspace"]
    ws = ok(J.workspace_run(ws))["workspace"]
    exported = ok(J.workspace_export_module(ws, {"module_id": "periscope", "title": "Two folds", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))
    module = exported["module"]
    assert module["instances"] == ws["instances"] and module["groups"] == ws["groups"] and "operations" not in module or module["operations"] == []
    assert len(module["validations"]) == 1 and "Composed of 2 imported layouts" in module["module"]["description"]
    host = ok(J.workspace_init(project_of([laser("SYS_LASER", (-100.0, 0.0, 300.0))], [], [], [])))["workspace"]
    reply = ok(J.workspace_import(host, module, {"instance_key": "p", "fixtures": {"rebind": {"LASER": "SYS_LASER"}},
                                                 "placement": {"position_mm": [0.0, 0.0, 300.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))
    new = reply["workspace"]
    keys = [i["instance_key"] for i in new["instances"]]
    assert keys == ["p", "p_fa", "p_fb"] and reply["result"]["nested_instances"] == ["p_fa", "p_fb"]
    nested = {i["instance_key"]: i for i in new["instances"]}
    assert nested["p_fa"]["source"] == ws["instances"][0]["source"]  # provenance of the inner import is kept
    assert {(e["kind"], e["source_id"], e["parent_id"]) for e in nested["p_fa"]["id_map"] if e["kind"] == "COMPONENT"} == {("COMPONENT", "M1", "p_fa_M1")}
    assert nested["p_fa"]["placement"]["position_mm"] == [0.0, 0.0, 300.0]
    groups = {g["group_id"]: g for g in new["groups"]}
    assert set(groups) == {"p", "p_fa", "p_fb", "p_mirrors"}
    assert groups["p_mirrors"]["members"] == [{"kind": "COMPONENT", "id": "p_fa_M1"}, {"kind": "COMPONENT", "id": "p_fb_M1"}]
    assert groups["p_fa"]["origin"] == "IMPORT" and groups["p_fa"]["instance_key"] == "p_fa" and groups["p_mirrors"]["origin"] == "USER"
    targets = {t["target_id"]: t for t in new["project"]["targets"]}
    assert targets["p_fa_T_MON_CENTER"]["observation_id"] == "p_fa_MON"
    # the nested instances stay independently editable: moving p_fb leaves p_fa unedited
    moved = ok(J.workspace_transform(new, {"selector": {"instances": ["p_fb"]}, "translation_mm": [0, 0, 5.0]}))["workspace"]
    status = {i["instance_key"]: i for i in ok(J.workspace_status(moved))["status"]["instances"]}
    assert status["p_fa"]["edited"] == [] and status["p_fb"]["edited"] == []  # a whole-instance move
    assert status["p"]["edited"] != []  # the outer instance's members moved relative to it
    # save / reopen keeps everything
    reopened = comp.load_workspace(json.loads(json.dumps(moved)))
    assert reopened.payload() == moved


# ------------------------------------------------------------------------------------------------ physics of the composition


def test_a_composition_gives_the_optical_results_of_the_same_directly_authored_project():
    """Fold module + focusing module composed vs the same physics authored directly (different ids): every mapped
    observation receives the same beams."""

    ws = system_with_laser()
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "fold", "fixtures": REBIND}))["workspace"]
    turn = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]  # the lens module's +X axis along +Y
    ws = ok(J.workspace_import(ws, lens_mod(), {"instance_key": "foc", "fixtures": REBIND,
                                                "placement": {"position_mm": [0.0, 120.0, 0.0], "quaternion_wxyz": turn}}))["workspace"]
    composed = ok(J.workspace_run(ws))
    mirror = build("mirror.flat", {"diameter_mm": "25.4", "reflectance": "0.98", "valid_band_nm": [400, 700]}, model_id="m")
    lens = build("lens.ideal_thin", {"focal_length_mm": "50", "clear_aperture_diameter_mm": "20"}, model_id="l")
    direct = project_of([laser("LASER"), place(mirror, "MIRROR", (0, 0, 0), FOLD_X_TO_Y),
                         place(lens, "LENS", (0, 120.0, 0), ALONG_Y)],
                        [screen("MONITOR", (0.0, 50.0, 0.0), ALONG_Y, radius_mm=20.0), screen("SPOT", (0.0, 170.0, 0.0), ALONG_Y, absorbing=True)],
                        [{"target_id": "C", "metric": "spot_centroid_u_mm", "observation_id": "MONITOR", "target_value": 0.0, "tolerance_abs": 0.5},
                         {"target_id": "S", "metric": "beam_radius_mm", "observation_id": "SPOT", "maximum": 0.1}], [mirror, lens])
    reference = J.run_project(direct)
    assert reference["ok"] and composed["status"] == reference["status"] == "PASS"
    mapping = {"fold_MON": "MONITOR", "foc_FOCUS": "SPOT"}
    rows = {o["observation_id"]: o for o in composed["summary"]["observations"]}
    ref_rows = {o["observation_id"]: o for o in reference["summary"]["observations"]}
    for ours, theirs in mapping.items():
        a, b = rows[ours], ref_rows[theirs]
        assert a["total_power_W"] == pytest.approx(b["total_power_W"], rel=1e-9)
        assert len(a["arrivals"]) == len(b["arrivals"])
        for x, y in zip(a["arrivals"], b["arrivals"]):
            for key in ("power_W", "beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm"):
                assert x[key] == pytest.approx(y[key], rel=1e-6, abs=1e-12), key
            assert np.allclose(x["position_mm"], y["position_mm"], atol=1e-9)
    values = {t["target_id"]: t["value"] for t in composed["summary"]["targets"]}
    ref_values = {t["target_id"]: t["value"] for t in reference["summary"]["targets"]}
    assert values["fold_T_MON_CENTER"] == pytest.approx(ref_values["C"], abs=1e-9)
    assert values["foc_T_SPOT"] == pytest.approx(ref_values["S"], rel=1e-6)


def test_modules_that_pass_alone_can_fail_together():
    lens_ws = ok(J.workspace_run(lens_workspace()))
    assert lens_ws["status"] == "PASS"
    ws = system_with_laser()
    ws = ok(J.workspace_import(ws, fold_mod(), {"instance_key": "fold", "fixtures": REBIND}))["workspace"]
    # the focusing stage placed with its axis along +X instead of along the folded beam (+Y): the beam misses it
    module = ok(J.workspace_export_module(lens_ws["workspace"], LENS_DESCRIPTOR))["module"]
    ws = ok(J.workspace_import(ws, module, {"instance_key": "foc", "fixtures": REBIND,
                                            "placement": {"position_mm": [0.0, 120.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}}))["workspace"]
    system = ok(J.workspace_run(ws))
    assert system["status"] in ("FAIL", "UNKNOWN")
    assert {t["target_id"]: t["status"] for t in system["summary"]["targets"]}["foc_T_SPOT"] != "PASS"
    instance = ok(J.workspace_status(system["workspace"]))["status"]["instances"][1]
    assert [v["validation_status"] for v in instance["module_validations"]] == ["PASS"]  # the module's own PASS stays its own


def test_capabilities_describe_the_composition_calls():
    caps = J.capabilities()["composition"]
    assert caps["workspace_schema"] == {"schema": "opensci.optical_workspace", "schema_version": "1.0"}
    assert caps["functions"] == sorted(name for name in dir(J) if name.startswith("workspace_"))
    assert "BASE_CHANGED" in caps["error_codes"] and "UNMARKED_SOURCE_IMPORTED" in caps["warning_codes"]
    assert J.json_schema("optical-workspace")["schema"]["title"] == "OpticalWorkspaceV1"


def test_the_worked_example_of_the_guide_runs(capsys):
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "engine" / "COMPOSITION.md").read_text(encoding="utf-8")
    (block,) = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    exec(compile(block, "COMPOSITION.md", "exec"), {})  # noqa: S102 - trusted repository documentation
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "FAIL" and lines[2].startswith("PASS") and lines[3] == "[('CURRENT', [])]"


def test_a_retained_test_target_that_stays_with_the_module_is_reported_removed():
    descriptor = copy.deepcopy(FOLD_DESCRIPTOR)
    descriptor["fixtures"][2]["default_action"] = "RETAIN"
    descriptor["requirements"] = [{"target_id": "T_OUT", "scope": "MODULE_ONLY"}]
    module = ok(J.workspace_init(fold_module(), {"module": descriptor}))["workspace"]
    reply = ok(J.workspace_import(system_with_laser(), module, {"instance_key": "k", "fixtures": REBIND}))
    assert reply["result"]["fixtures"][2] == {"object_kind": "TARGET", "source_id": "T_OUT", "action": "REMOVED", "affected_targets": []}
    assert reply["result"]["requirements"]["module_only"] == ["T_OUT"]


def test_export_completes_the_workspace_module_descriptor_and_summarizes_its_instances_once():
    ws = two_fold_system()
    first = ok(J.workspace_export_module(ws, {"module_id": "pair", "title": "Pair", "description": "Two folds."}))["module"]
    again = ok(J.workspace_export_module(first, {"revision": "r2"}))["module"]
    assert again["module"]["module_id"] == "pair" and again["module"]["revision"] == "r2"
    assert again["module"]["description"] == first["module"]["description"]
    assert first["module"]["description"].startswith("Two folds. Composed of 2 imported layouts: 90 degree fold (fold-90) as fa")


def test_status_reports_each_instance_s_ports_at_its_current_placement(tmp_path, capsys):
    """The fold's ports (IN at (-20,0,0) +X, OUT at (0,20,0) +Y in the module frame) follow the instance through a turn and a
    shift; a composed module's nested instance reports them through both placements. References: own rotation matrices."""
    ws = ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "fa", "fixtures": REBIND,
                                                                  "placement": {"position_mm": [5.0, 0.0, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}}))["workspace"]
    axis, angle, pivot, shift = (0.2, -0.3, 1.0), 33.0, np.array([10.0, -4.0, 2.0]), np.array([1.5, 2.5, -3.0])
    ws = ok(J.workspace_transform(ws, {"selector": {"instances": ["fa"]}, "rotation": {"axis": list(axis), "angle_deg": angle},
                                       "pivot": {"point_mm": pivot.tolist()}, "translation_mm": shift.tolist()}))["workspace"]
    r = axis_angle_matrix(axis, angle)
    origin = r @ (np.array([5.0, 0.0, 0.0]) - pivot) + pivot + shift  # the module frame's origin, moved
    expected = {"IN": (r @ np.array([-20.0, 0.0, 0.0]) + origin, r @ np.array([1.0, 0.0, 0.0])),
                "OUT": (r @ np.array([0.0, 20.0, 0.0]) + origin, r @ np.array([0.0, 1.0, 0.0]))}
    (row,) = ok(J.workspace_status(ws))["status"]["instances"]
    ports = {p["port_id"]: p for p in row["interfaces"]}
    assert set(ports) == {"IN", "OUT"} and ports["IN"]["beam_reference"] == "LASER" and ports["IN"]["role"] == "INPUT"
    for port_id, (point, direction) in expected.items():
        assert np.max(np.abs(np.asarray(ports[port_id]["point_mm"]) - point)) < 1e-9
        assert np.max(np.abs(np.asarray(ports[port_id]["direction"]) - direction)) < 1e-12
    path = tmp_path / "turned.ws.json"
    path.write_text(json.dumps(ws), encoding="utf-8")
    from os_optics.cli import main

    capsys.readouterr()
    assert main(["workspace", "status", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"]["instances"][0]["interfaces"] == row["interfaces"]
    # the composed system as a module, imported turned by 90 degrees about +Z at (100, 0, 0): its nested fold instance
    composed = ok(J.workspace_export_module(ws, {"module_id": "folded-system", "title": "Folded system"}))["module"]
    outer = ok(J.workspace_import(system_with_laser((-300.0, 0.0, 0.0)), composed,
                                  {"instance_key": "sys", "placement": {"position_mm": [100.0, 0.0, 0.0], "quaternion_wxyz": list(turn_z(90.0))}}))
    rows = {i["instance_key"]: i for i in ok(J.workspace_status(outer["workspace"]))["status"]["instances"]}
    q90 = axis_angle_matrix((0.0, 0.0, 1.0), 90.0)
    nested = {p["port_id"]: p for p in rows["sys_fa"]["interfaces"]}
    assert rows["sys"]["interfaces"] == []  # the composed module declares no ports of its own
    for port_id, (point, direction) in expected.items():
        assert np.max(np.abs(np.asarray(nested[port_id]["point_mm"]) - (q90 @ point + np.array([100.0, 0.0, 0.0])))) < 1e-9
        assert np.max(np.abs(np.asarray(nested[port_id]["direction"]) - q90 @ direction)) < 1e-12
    assert nested["IN"]["beam_reference"] == "sys_LASER"
