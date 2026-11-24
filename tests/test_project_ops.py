"""The headless Project edit operations (``os_optics.local.project_ops``).

The fixed Project/operation regression corpus checks exact documents, error codes,
operation indexes and messages, including idempotent retries. Additional Python
cases cover the supported operation vocabulary, declared models and invalid input.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from os_optics.local.project_ops import (
    PROJECT_OPS,
    ProjectOpError,
    affected_objects,
    apply_ops,
    apply_project_ops,
    model_users,
    materials_of,
    member_model_ids,
)

REGRESSIONS = json.loads((Path(__file__).resolve().parent / "fixtures" / "project_ops_regressions.json").read_text(encoding="utf-8"))
Q = [0.7071067811865476, 0, 0.7071067811865476, 0]
Q2 = [0.5, 0.5, 0.5, 0.5]
BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
PROJECT_EDITOR_OPS = ["add_component", "add_observation", "add_target", "remove_component", "remove_observation", "remove_target", "set_component_pose",
                "set_component_product", "set_observation", "set_observation_pose", "set_retained_dof", "set_retained_dofs", "set_source_model",
                "set_target", "set_design_intent", "set_compute_budget", "add_optical_model", "update_optical_model", "remove_optical_model", "add_model_material",
                "update_model_material", "remove_model_material", "add_model_coating", "remove_model_coating", "set_component_model"]


# ------------------------------------------------------------------------------------------------ retained headless Project operation regressions


@pytest.mark.parametrize("case", REGRESSIONS["cases"], ids=[c["name"] for c in REGRESSIONS["cases"]])
def test_the_project_operations_preserve_the_regression_result(case):
    document, ops = case["document"], case["ops"]
    before = copy.deepcopy(document)
    expected = case.get("expected")
    if expected is None:  # retrying the same addition is an idempotent success
        out, already = apply_ops(document, ops)
        assert out == document and [a["op_index"] for a in already] == [0]
    elif expected["ok"]:
        out = apply_project_ops(document, ops)
        assert out == expected["project"]
        assert json.dumps(out) == json.dumps(expected["project"])  # stable key order
    else:
        with pytest.raises(ProjectOpError) as raised:
            apply_project_ops(document, ops)
        assert (raised.value.code, raised.value.op_index) == (expected["code"], expected["op_index"])
        if expected["code"] == "UNKNOWN_OP":  # native vocabulary also includes transform and numerical settings
            assert raised.value.message == expected["message"] + ", transform, set_numerical_settings"
        else:
            assert raised.value.message == expected["message"]
    assert document == before  # the input is never changed


def test_the_regression_corpus_covers_every_project_editor_operation_and_error_code():
    assert REGRESSIONS["schema"] == "opensci.project_ops_regressions" and REGRESSIONS["schema_version"] == "1.0"
    used = {op["op"] for case in REGRESSIONS["cases"] if isinstance(case["ops"], list) for op in case["ops"] if isinstance(op, dict) and "op" in op}
    assert set(PROJECT_EDITOR_OPS) <= used
    codes = {case["expected"]["code"] for case in REGRESSIONS["cases"] if case.get("expected") and not case["expected"]["ok"]}
    assert codes == {"UNKNOWN_OP", "NOT_FOUND", "DUPLICATE_ID", "BAD_ARGUMENT", "IN_USE"}
    assert any("expected" not in case for case in REGRESSIONS["cases"])  # idempotent retry has its own assertion


def test_the_vocabulary_includes_project_edits_transform_and_numerical_settings():
    assert sorted(PROJECT_OPS) == sorted([*PROJECT_EDITOR_OPS, "transform", "set_numerical_settings"])
    for spec in PROJECT_OPS.values():
        assert len(spec["summary"]) > 10 and spec["fields"]


# ------------------------------------------------------------------------------------------------ Project edit cases


def reference() -> dict:
    """A representative reference Project for operation tests."""

    return {
        "schema": "opensci.project", "project_schema_version": "2.0",
        "components": [
            {"instance_id": "SRC", "optical_role": "source",
             "custom_part": {"custom_part_key": "user-collimated-laser", "description": "User-supplied collimated laser source."},
             "nominal_pose": {"position_mm": [-200, 0, 0], "quaternion_wxyz": list(Q)}, "selected_parameters": [], "retained_dofs": [],
             "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 780, "power_w": 0.001, "waist_radius_mm": 1, "waist_position_mm": 0,
                              "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}}},
            {"instance_id": "L1", "optical_role": "lens", "product": {"optical_sku": "OS-OPT-ACHROMAT-D25-F75-NIR", "catalog_version": "2026-09-27.1"},
             "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": list(Q)}, "selected_parameters": [],
             "retained_dofs": [{"axis": "Tx", "frame": "COMPONENT_LOCAL", "control": "MANUAL"}, {"axis": "Ty", "frame": "COMPONENT_LOCAL", "control": "MANUAL"},
                               {"axis": "Tz", "frame": "COMPONENT_LOCAL", "control": "NUMERICAL"}]},
        ],
        "observations": [{"observation_id": "focal_plane", "pose": {"position_mm": [76.473, 0, 0], "quaternion_wxyz": list(Q)},
                          "aperture": {"shape": "circle", "radius_mm": 12.5, "center_offset_mm": [0, 0], "obstruction_radius_mm": 0},
                          "absorbing": False, "medium": "air"}],
        "targets": [
            {"target_id": "focus_position", "metric": "waist_distance_x_mm", "observation_id": "focal_plane", "target_value": 0, "tolerance_abs": 0.1,
             "unit": "mm", "parameters": []},
            {"target_id": "spot_radius", "metric": "beam_radius_mm", "observation_id": "focal_plane", "maximum": 0.05, "unit": "mm", "parameters": []},
        ],
        "materials": [],
    }


def component(doc: dict, instance_id: str) -> dict:
    return next(c for c in doc["components"] if c["instance_id"] == instance_id)


def refused(doc, ops, code: str, op_index: int = 0, message: str | None = None) -> ProjectOpError:
    with pytest.raises(ProjectOpError) as raised:
        apply_project_ops(doc, ops)
    assert (raised.value.code, raised.value.op_index) == (code, op_index)
    if message is not None:
        assert re.search(message, raised.value.message)
    return raised.value


def test_never_mutates_its_input_and_returns_a_fresh_document():
    doc = reference()
    frozen = json.dumps(doc)
    out = apply_project_ops(doc, [{"op": "set_component_pose", "instance_id": "L1", "position_mm": [5, 0, 0]}])
    assert json.dumps(doc) == frozen
    assert component(out, "L1")["nominal_pose"] == {"position_mm": [5, 0, 0], "quaternion_wxyz": Q}
    assert out["observations"] == doc["observations"] and out["observations"] is not doc["observations"]


def test_set_component_pose_sets_position_and_or_quaternion_and_validates_vectors():
    doc = reference()
    out = apply_project_ops(doc, [{"op": "set_component_pose", "instance_id": "SRC", "quaternion_wxyz": [1, 0, 0, 0]}])
    assert component(out, "SRC")["nominal_pose"] == {"position_mm": [-200, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    refused(doc, [{"op": "set_component_pose", "instance_id": "SRC"}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_component_pose", "instance_id": "SRC", "position_mm": [1, 2]}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_component_pose", "instance_id": "SRC", "position_mm": [1, float("nan"), 0]}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_component_pose", "instance_id": "SRC", "position_mm": [1, "2", 0]}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_component_pose", "instance_id": "NOPE", "position_mm": [0, 0, 0]}], "NOT_FOUND", message='component "NOPE" does not exist')


def test_set_source_model_merges_null_removes_and_non_sources_are_refused():
    doc = reference()
    out = apply_project_ops(doc, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"wavelength_nm": 850, "waist_radius_y_mm": 0.8}},
                                  {"op": "set_source_model", "instance_id": "SRC", "changes": {"waist_radius_y_mm": None}}])
    assert component(out, "SRC")["source_model"] == {**component(doc, "SRC")["source_model"], "wavelength_nm": 850}
    refused(doc, [{"op": "set_source_model", "instance_id": "L1", "changes": {"m2": 1.1}}], "BAD_ARGUMENT", message="has no source_model")
    refused(doc, [{"op": "set_source_model", "instance_id": "SRC", "changes": []}], "BAD_ARGUMENT", message="changes must be an object")


def test_set_source_model_declares_the_coherence_once():
    doc = reference()
    by_linewidth = apply_project_ops(doc, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"linewidth_nm": 0.001, "coherence_length_mm": None}}])
    assert component(by_linewidth, "SRC")["source_model"] == {**component(doc, "SRC")["source_model"], "linewidth_nm": 0.001}
    by_length = apply_project_ops(by_linewidth, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"coherence_length_mm": 300, "linewidth_nm": None}}])
    assert component(by_length, "SRC")["source_model"] == {**component(doc, "SRC")["source_model"], "coherence_length_mm": 300}
    cleared = apply_project_ops(by_length, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"coherence_length_mm": None}}])
    assert component(cleared, "SRC")["source_model"] == component(doc, "SRC")["source_model"]


def test_set_retained_dof_retains_releases_one_axis_and_keeps_tx_to_rz_order():
    doc = reference()
    out = apply_project_ops(doc, [{"op": "set_retained_dof", "instance_id": "L1", "axis": "Rz", "control": "MANUAL"},
                                  {"op": "set_retained_dof", "instance_id": "L1", "axis": "Tz", "control": None},
                                  {"op": "set_retained_dof", "instance_id": "L1", "axis": "Tx", "control": "NUMERICAL"}])
    # (Tx stays retained: it keeps its stated frame and changes only its control intent)
    assert component(out, "L1")["retained_dofs"] == [{"axis": "Tx", "frame": "COMPONENT_LOCAL", "control": "NUMERICAL"},
                                                     {"axis": "Ty", "frame": "COMPONENT_LOCAL", "control": "MANUAL"},
                                                     {"axis": "Rz", "control": "MANUAL"}]
    refused(doc, [{"op": "set_retained_dof", "instance_id": "L1", "axis": "Qx", "control": "MANUAL"}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_retained_dof", "instance_id": "L1", "axis": "Tx", "control": "AUTO"}], "BAD_ARGUMENT")


def test_set_retained_dof_keeps_an_axis_stated_stage_limits_when_only_the_control_changes():
    # a retained axis that only changes its control intent keeps its stated travel, hard limits, repeatability, drift and
    # unit (they were dropped, and every dof_* target became INDETERMINATE with MECHANICAL_LIMIT_NOT_DECLARED)
    tz = {"axis": "Tz", "frame": "COMPONENT_LOCAL", "control": "NUMERICAL", "travel": {"min": -1.5, "max": 2.5},
          "hard_limits": {"min": -2.0, "max": 3.0}, "repeatability": 0.003, "drift": {"per_kelvin": 0.0011}, "unit": "mm"}
    rx = {"axis": "Rx", "control": "MANUAL", "travel": {"min": -0.5, "max": 0.5}, "unit": "deg"}
    doc = apply_project_ops(reference(), [{"op": "set_retained_dofs", "instance_id": "L1", "retained_dofs": [tz, rx]}])
    out = apply_project_ops(doc, [{"op": "set_retained_dof", "instance_id": "L1", "axis": "Tz", "control": "MANUAL"},
                                  {"op": "set_retained_dof", "instance_id": "L1", "axis": "Rx", "control": None},
                                  {"op": "set_retained_dof", "instance_id": "L1", "axis": "Ty", "control": "NUMERICAL"}])
    assert component(out, "L1")["retained_dofs"] == [{"axis": "Ty", "control": "NUMERICAL"}, {**tz, "control": "MANUAL"}]
    assert component(doc, "L1")["retained_dofs"][0] == tz  # the input is not changed
    # a released and re-retained axis states nothing (releasing removes the axis with its limits)
    again = apply_project_ops(out, [{"op": "set_retained_dof", "instance_id": "L1", "axis": "Tz", "control": None},
                                    {"op": "set_retained_dof", "instance_id": "L1", "axis": "Tz", "control": "NUMERICAL"}])
    assert component(again, "L1")["retained_dofs"] == [{"axis": "Ty", "control": "NUMERICAL"}, {"axis": "Tz", "control": "NUMERICAL"}]


def test_set_retained_dof_keeps_limits_through_the_json_api_and_the_project_still_validates():
    # the reviewer's case: the Engine example's lens with a fully stated Tz stage, control changed to MANUAL
    from os_optics.local import example_project, json_api

    doc = example_project().payload()
    lens = next(c for c in doc["components"] if c.get("optical_role") != "source" and c.get("instance_id"))
    stated = {"axis": "Tz", "control": "NUMERICAL", "travel": {"min": -1.5, "max": 2.5}, "hard_limits": {"min": -2.0, "max": 3.0},
              "repeatability": 0.003, "drift": {"per_kelvin": 0.0011}, "unit": "mm"}
    lens["retained_dofs"] = [stated]
    assert json_api.check_project(doc)["ok"] is True
    reply = json_api.update_project(doc, [{"op": "set_retained_dof", "instance_id": lens["instance_id"], "axis": "Tz", "control": "MANUAL"}])
    assert reply["ok"] is True
    after = next(c for c in reply["project"]["components"] if c.get("instance_id") == lens["instance_id"])["retained_dofs"]
    assert [{k: v for k, v in d.items() if k != "frame"} for d in after] == [{**stated, "control": "MANUAL"}]


def test_set_retained_dofs_replaces_the_set_sorted_and_empty_fixes_every_dof():
    doc = reference()
    out = apply_project_ops(doc, [{"op": "set_retained_dofs", "instance_id": "L1",
                                   "retained_dofs": [{"axis": "Rx", "control": "MANUAL"}, {"axis": "Ty", "control": "NUMERICAL"}]}])
    assert component(out, "L1")["retained_dofs"] == [{"axis": "Ty", "control": "NUMERICAL"}, {"axis": "Rx", "control": "MANUAL"}]
    assert component(apply_project_ops(doc, [{"op": "set_retained_dofs", "instance_id": "L1", "retained_dofs": []}]), "L1")["retained_dofs"] == []
    refused(doc, [{"op": "set_retained_dofs", "instance_id": "L1", "retained_dofs": [{"axis": "Tw"}]}], "BAD_ARGUMENT")
    refused(doc, [{"op": "set_retained_dofs", "instance_id": "L1", "retained_dofs": "Tx"}], "BAD_ARGUMENT")


def test_set_component_product_null_removes_the_pin_and_absent_keeps_it():
    doc = reference()
    kept = apply_project_ops(doc, [{"op": "set_component_product", "instance_id": "L1", "optical_sku": "OS-OPT-X"}])
    assert component(kept, "L1")["product"] == {"optical_sku": "OS-OPT-X", "catalog_version": "2026-09-27.1"}
    removed = apply_project_ops(doc, [{"op": "set_component_product", "instance_id": "L1", "optical_sku": "OS-OPT-X", "catalog_version": None}])
    assert component(removed, "L1")["product"] == {"optical_sku": "OS-OPT-X"}


def test_add_and_remove_components_observations_and_targets():
    doc = reference()
    lens2 = {**copy.deepcopy(component(doc, "L1")), "instance_id": "L2"}
    plane = {"observation_id": "mid", "pose": {"position_mm": [-100, 0, 0], "quaternion_wxyz": list(Q)}}
    target = {"target_id": "power", "metric": "power_W", "observation_id": "mid", "minimum": 0.0009, "unit": "W"}
    out = apply_project_ops(doc, [{"op": "add_component", "component": lens2}, {"op": "add_observation", "observation": plane},
                                  {"op": "add_target", "target": target}, {"op": "remove_target", "target_id": "focus_position"},
                                  {"op": "remove_component", "instance_id": "L1"}])
    assert [c["instance_id"] for c in out["components"]] == ["SRC", "L2"]
    assert [o["observation_id"] for o in out["observations"]] == ["focal_plane", "mid"]
    assert [t["target_id"] for t in out["targets"]] == ["spot_radius", "power"]
    assert out["components"][1] is not lens2  # the added objects are copies
    refused(doc, [{"op": "add_component", "component": {}}], "BAD_ARGUMENT", message=r"component\.instance_id is required")
    refused(doc, [{"op": "remove_observation", "observation_id": "nope"}], "NOT_FOUND")
    refused(doc, [{"op": "remove_target", "target_id": "nope"}], "NOT_FOUND")


def test_an_identical_add_is_already_present_and_different_content_under_the_id_is_duplicate():
    doc = reference()
    for op, kind, item_id in (({"op": "add_component", "component": component(doc, "L1")}, "COMPONENT", "L1"),
                              ({"op": "add_observation", "observation": doc["observations"][0]}, "OBSERVATION", "focal_plane"),
                              ({"op": "add_target", "target": doc["targets"][0]}, "TARGET", "focus_position")):
        out, already = apply_ops(doc, [op])
        assert out == doc and already == [{"op_index": 0, "op": op["op"], "kind": kind, "id": item_id}]
    refused(doc, [{"op": "add_component", "component": {**component(doc, "L1"), "optical_role": "window"}}], "DUPLICATE_ID",
            message='component "L1" already exists')
    refused(doc, [{"op": "add_target", "target": {**doc["targets"][0], "maximum": 1}}], "DUPLICATE_ID")


def test_an_add_identical_in_the_contracts_normalized_form_is_already_present():
    """A retried add whose item omits defaults the stored (normalized) Project writes, or writes 1 for 1.0, is the same item."""

    doc = reference()
    stored = {**copy.deepcopy(component(doc, "L1")), "instance_id": "L2"}
    doc["components"].append(stored)
    sparse = {k: v for k, v in stored.items() if k not in ("selected_parameters",)}
    sparse["nominal_pose"] = {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(Q)}
    out, already = apply_ops(doc, [{"op": "add_component", "component": sparse}])
    assert out == doc and already[0]["id"] == "L2"


def test_set_observation_pose_set_observation_and_set_target_merge_fields():
    doc = reference()
    out = apply_project_ops(doc, [
        {"op": "set_observation_pose", "observation_id": "focal_plane", "position_mm": [80, 0, 0]},
        {"op": "set_observation", "observation_id": "focal_plane", "changes": {"aperture": {"shape": "circle", "radius_mm": 50}, "medium": None}},
        {"op": "set_target", "target_id": "spot_radius", "changes": {"maximum": 0.02, "minimum": 0.001}},
        {"op": "set_target", "target_id": "focus_position", "changes": {"tolerance_abs": None, "tolerance_rel": 0.01}},
    ])
    plane = out["observations"][0]
    assert plane["pose"]["position_mm"] == [80, 0, 0] and plane["aperture"] == {"shape": "circle", "radius_mm": 50} and "medium" not in plane
    assert out["targets"][1]["maximum"] == 0.02 and out["targets"][1]["minimum"] == 0.001
    assert "tolerance_abs" not in out["targets"][0] and out["targets"][0]["tolerance_rel"] == 0.01


def test_a_batch_is_atomic_and_a_bad_op_names_its_index():
    doc = reference()
    before = json.dumps(doc)
    ops = [{"op": "set_component_pose", "instance_id": "L1", "position_mm": [5, 0, 0]},
           {"op": "set_target", "target_id": "spot_radius", "changes": {"maximum": 0.01}}, {"op": "frobnicate"}]
    error = refused(doc, ops, "UNKNOWN_OP", 2)
    assert "set_component_pose" in error.message and "transform" in error.message
    assert json.dumps(doc) == before
    refused(doc, [ops[0], None], "BAD_ARGUMENT", 1)
    refused(doc, [], "BAD_ARGUMENT", -1)
    refused(doc, "set", "BAD_ARGUMENT", -1)


def test_later_ops_in_a_batch_see_the_result_of_earlier_ones():
    out = apply_project_ops(reference(), [
        {"op": "add_observation", "observation": {"observation_id": "p2", "pose": {"position_mm": [10, 0, 0], "quaternion_wxyz": list(Q)}}},
        {"op": "set_observation_pose", "observation_id": "p2", "position_mm": [20, 0, 0]},
        {"op": "remove_observation", "observation_id": "p2"},
    ])
    assert [o["observation_id"] for o in out["observations"]] == ["focal_plane"]


def test_fields_the_operations_do_not_know_are_kept():
    doc = reference()
    doc["x_future"] = {"keep": True}
    doc["components"][1]["x_extra"] = [1, 2]
    out = apply_project_ops(doc, [{"op": "set_component_pose", "instance_id": "L1", "position_mm": [1, 0, 0]}])
    assert out["x_future"] == {"keep": True} and out["components"][1]["x_extra"] == [1, 2]


# ------------------------------------------------------------------------------------------------ declaredModels.test.ts, ported


def nbk7() -> dict:
    return {"material_id": "N-BK7", "display_name": "N-BK7", "valid_band_nm": dict(BAND),
            "dispersion": {"kind": "SELLMEIER", "b": ["1.03961212", "0.231792344", "1.01046945"], "c_um2": ["0.00600069867", "0.0200179144", "103.560653"]}}


def pcx(model_id: str = "pcx") -> dict:
    return {"model_id": model_id, "model": {
        "model_kind": "SURFACE_SEQUENCE", "model_version": "2", "fidelity": "SURFACE_PRESCRIPTION",
        "surfaces": [{"surface_id": "S1", "vertex_z_mm": "0", "radius_mm": "52", "clear_radius_mm": "9", "traced": "TRANSMITTED", "status": "SPECIFIED"},
                     {"surface_id": "S2", "vertex_z_mm": "4.5", "clear_radius_mm": "9", "traced": "TRANSMITTED", "status": "SPECIFIED"}],
        "media": ["N-BK7"], "edge_radius_mm": "10", "valid_band_nm": dict(BAND), "approximations": ["IDEAL_SURFACE_FIGURE", "NOMINAL_PRESCRIPTION"],
        "status": "SPECIFIED"}}


def stop(model_id: str = "stop") -> dict:
    return {"model_id": model_id, "model": {"model_kind": "APERTURE_STOP", "model_version": "1", "opening_radius_mm": "6", "edge_radius_mm": "10",
                                            "status": "SPECIFIED"}}


def unit(model_id: str = "unit") -> dict:
    return {"model_id": model_id, "model": {
        "model_kind": "COMPOUND", "model_version": "1",
        "members": [{"member_id": m, "model_id": i, "pose": {"position_mm": [0, 0, z], "quaternion_wxyz": [1, 0, 0, 0]}} for m, i, z in (("L1", "pcx", 0), ("STOP", "stop", 14))],
        "envelope": {"kind": "CYLINDER", "diameter_mm": "22", "z_min_mm": "-1", "z_max_mm": "30", "status": "SPECIFIED"}}}


def base() -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0",
            "components": [{"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
                            "nominal_pose": {"position_mm": [-100, 0, 0], "quaternion_wxyz": list(Q2)}},
                           {"instance_id": "L1", "optical_role": "lens", "product": {"optical_sku": "OS-OPT-LENS-X"},
                            "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": list(Q2)}}],
            "observations": [], "targets": []}


def with_unit() -> dict:
    return apply_project_ops(base(), [
        {"op": "add_optical_model", "optical_model": pcx(), "model_materials": [nbk7()]},
        {"op": "add_optical_model", "optical_model": stop()},
        {"op": "add_optical_model", "optical_model": unit()},
        {"op": "add_component", "component": {"instance_id": "OBJ", "optical_role": "lens", "custom_part": {"custom_part_key": "own-obj", "description": "objective"},
                                              "optical_model_id": "unit", "nominal_pose": {"position_mm": [50, 0, 0], "quaternion_wxyz": list(Q2)}}},
    ])


def test_set_design_intent_sets_optical_system_and_null_restores_the_exact_document():
    doc = base()
    system = apply_project_ops(doc, [{"op": "set_design_intent", "design_intent": "OPTICAL_SYSTEM"}])
    assert system["design_intent"] == "OPTICAL_SYSTEM"
    back = apply_project_ops(system, [{"op": "set_design_intent", "design_intent": None}])
    assert "design_intent" not in back and json.dumps(back) == json.dumps(doc)
    refused(doc, [{"op": "set_design_intent", "design_intent": "BENCH"}], "BAD_ARGUMENT")


def test_add_optical_model_shares_identical_materials_and_refuses_different_ones():
    doc = with_unit()
    assert [m["model_id"] for m in doc["optical_models"]] == ["pcx", "stop", "unit"] and [m["material_id"] for m in doc["model_materials"]] == ["N-BK7"]
    again = apply_project_ops(doc, [{"op": "add_optical_model", "optical_model": pcx("pcx2"), "model_materials": [nbk7()]}])
    assert len(again["model_materials"]) == 1
    other = {**nbk7(), "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.5"}}
    refused(doc, [{"op": "add_optical_model", "optical_model": pcx("pcx3"), "model_materials": [other]}], "DUPLICATE_ID", message="material N-BK7 already declared")
    refused(doc, [{"op": "add_optical_model", "optical_model": {**pcx(), "display_name": "another"}}], "DUPLICATE_ID")
    refused(doc, [{"op": "add_optical_model", "optical_model": {"model_id": "x"}}], "BAD_ARGUMENT")
    out, already = apply_ops(doc, [{"op": "add_optical_model", "optical_model": pcx(), "model_materials": [nbk7()]}])
    assert out == doc and already[0]["kind"] == "OPTICAL_MODEL"


def test_a_project_without_declared_models_keeps_no_empty_lists():
    doc = apply_project_ops(base(), [{"op": "add_optical_model", "optical_model": stop()}])
    assert len(doc["optical_models"]) == 1 and "model_materials" not in doc and "model_coatings" not in doc
    empty = apply_project_ops(doc, [{"op": "remove_optical_model", "model_id": "stop"}])
    assert "optical_models" not in empty and json.dumps(empty) == json.dumps(base())


def test_update_optical_model_replaces_fields_never_the_id():
    doc = with_unit()
    edited = apply_project_ops(doc, [{"op": "update_optical_model", "model_id": "stop",
                                      "changes": {"model": {**stop()["model"], "opening_radius_mm": "4"}, "display_name": "Iris"}}])
    model = next(m for m in edited["optical_models"] if m["model_id"] == "stop")
    assert model["model"]["opening_radius_mm"] == "4" and model["display_name"] == "Iris"
    cleared = apply_project_ops(edited, [{"op": "update_optical_model", "model_id": "stop", "changes": {"display_name": None}}])
    assert "display_name" not in next(m for m in cleared["optical_models"] if m["model_id"] == "stop")
    refused(doc, [{"op": "update_optical_model", "model_id": "stop", "changes": {"model_id": "other"}}], "BAD_ARGUMENT")
    refused(doc, [{"op": "update_optical_model", "model_id": "nope", "changes": {}}], "NOT_FOUND")


def test_remove_optical_model_is_refused_while_used_and_takes_its_own_materials():
    doc = with_unit()
    refused(doc, [{"op": "remove_optical_model", "model_id": "unit"}], "IN_USE", message="component OBJ")
    refused(doc, [{"op": "remove_optical_model", "model_id": "pcx"}], "IN_USE", message="unit unit")
    gone = apply_project_ops(doc, [{"op": "remove_component", "instance_id": "OBJ"}, {"op": "remove_optical_model", "model_id": "unit"},
                                   {"op": "remove_optical_model", "model_id": "pcx"}])
    assert [m["model_id"] for m in gone["optical_models"]] == ["stop"] and "model_materials" not in gone


def test_material_and_coating_operations():
    doc = with_unit()
    flint = {"material_id": "FLINT", "display_name": "Flint", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.7174"}, "valid_band_nm": dict(BAND)}
    added = apply_project_ops(doc, [{"op": "add_model_material", "material": flint}])
    assert [m["material_id"] for m in added["model_materials"]] == ["N-BK7", "FLINT"]
    refused(added, [{"op": "add_model_material", "material": {**flint, "display_name": "other"}}], "DUPLICATE_ID")
    updated = apply_project_ops(added, [{"op": "update_model_material", "material_id": "FLINT", "changes": {"absorption_coefficient_per_mm": "0.01"}}])
    assert updated["model_materials"][1]["absorption_coefficient_per_mm"] == "0.01"
    refused(added, [{"op": "remove_model_material", "material_id": "N-BK7"}], "IN_USE", message="model pcx")
    assert len(apply_project_ops(added, [{"op": "remove_model_material", "material_id": "FLINT"}])["model_materials"]) == 1
    coating = {"coating_id": "AR", "kind": "ANTI_REFLECTION"}
    coated = apply_project_ops(doc, [{"op": "add_model_coating", "coating": coating}])
    assert coated["model_coatings"] == [coating]
    assert "model_coatings" not in apply_project_ops(coated, [{"op": "remove_model_coating", "coating_id": "AR"}])


def test_set_component_model_and_set_component_product_turn_a_component_into_and_out_of_a_declared_optic():
    doc = apply_project_ops(base(), [{"op": "add_optical_model", "optical_model": pcx(), "model_materials": [nbk7()]}])
    own = apply_project_ops(doc, [{"op": "set_component_model", "instance_id": "L1", "optical_model_id": "pcx", "description": "Declared copy of OS-OPT-LENS-X"}])
    l1 = own["components"][1]
    assert "product" not in l1 and l1["optical_model_id"] == "pcx"
    assert l1["custom_part"] == {"custom_part_key": "own-L1", "description": "Declared copy of OS-OPT-LENS-X"}
    assert l1["nominal_pose"] == doc["components"][1]["nominal_pose"]
    refused(doc, [{"op": "set_component_model", "instance_id": "L1", "optical_model_id": "nope"}], "NOT_FOUND")
    back = apply_project_ops(own, [{"op": "set_component_product", "instance_id": "L1", "optical_sku": "OS-OPT-LENS-X"}])
    assert "optical_model_id" not in back["components"][1] and "custom_part" not in back["components"][1]
    assert back["components"][1]["product"] == {"optical_sku": "OS-OPT-LENS-X"} and len(back["optical_models"]) == 1


def test_usage_helpers():
    doc = with_unit()
    assert materials_of(pcx()["model"]) == ["N-BK7"]
    assert materials_of({"model_kind": "SURFACE_SEQUENCE", "media": ["AIR", "N-BK7"]}) == ["N-BK7"]
    assert member_model_ids(unit()["model"]) == ["pcx", "stop"]
    assert model_users(doc, "unit") == {"components": ["OBJ"], "units": []}
    assert model_users(doc, "pcx") == {"components": [], "units": ["unit"]}


def test_affected_objects_name_what_changed():
    doc = reference()
    out = apply_project_ops(doc, [{"op": "set_component_pose", "instance_id": "L1", "position_mm": [5, 0, 0]},
                                  {"op": "remove_target", "target_id": "spot_radius"}, {"op": "set_design_intent", "design_intent": "OPTICAL_SYSTEM"},
                                  {"op": "add_observation", "observation": {"observation_id": "p2", "pose": {"position_mm": [1, 0, 0], "quaternion_wxyz": list(Q)}}}])
    assert affected_objects(doc, out) == [
        {"kind": "COMPONENT", "id": "L1", "change": "MODIFIED", "fields": ["nominal_pose"]},
        {"kind": "OBSERVATION", "id": "p2", "change": "ADDED", "fields": ["observation_id", "pose"]},
        {"kind": "TARGET", "id": "spot_radius", "change": "REMOVED", "fields": ["maximum", "metric", "observation_id", "parameters", "target_id", "unit"]},
        {"kind": "PROJECT", "id": None, "change": "MODIFIED", "fields": ["design_intent"]},
    ]
    assert affected_objects(doc, doc) == []


# ------------------------------------------------------------------------------------------------ contract-invalid input


def test_set_source_model_merges_into_an_empty_source_model():
    # An empty source model is merged into; the later Project check decides validity.
    doc = {"components": [{"instance_id": "SRC", "source_model": {}}]}
    out = apply_project_ops(doc, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"model_kind": "GAUSSIAN_BEAM"}}])
    assert out["components"][0]["source_model"] == {"model_kind": "GAUSSIAN_BEAM"}
    for missing in (None, 0, "", False):
        with pytest.raises(ProjectOpError) as raised:
            apply_project_ops({"components": [{"instance_id": "SRC", "source_model": missing}]},
                              [{"op": "set_source_model", "instance_id": "SRC", "changes": {}}])
        assert (raised.value.code, raised.value.message) == ("BAD_ARGUMENT", 'component "SRC" has no source_model')


def test_an_op_without_its_id_finds_an_item_without_one_as_the_designer_does():
    # TypeScript ``c.instance_id === op.instance_id`` with both undefined matches
    out = apply_project_ops({"components": [{"optical_role": "lens"}]}, [{"op": "set_component_pose", "position_mm": [1, 2, 3]}])
    assert out["components"][0]["nominal_pose"] == {"position_mm": [1, 2, 3]}
    with pytest.raises(ProjectOpError) as raised:  # a JSON null id is not an absent one
        apply_project_ops({"components": [{"optical_role": "lens"}]}, [{"op": "set_component_pose", "instance_id": None, "position_mm": [1, 2, 3]}])
    assert raised.value.code == "NOT_FOUND"
