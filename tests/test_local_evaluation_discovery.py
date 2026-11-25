"""Selective evaluation through the JSON API and its discovery: ``evaluate_project`` / ``evaluation_capabilities`` round
trips, refusals with stable codes, and a bounded candidate search written only against ``capabilities()["evaluation"]``,
``evaluation_capabilities()`` and the public JSON entries (``os_optics.local.json_api``; the run monitor of the stop test is
the host's own)."""

from __future__ import annotations

import copy
import json

import pytest

from os_optics.local import json_api
from os_optics.run_control import monitored
from os_optics.run_monitor import RunMonitor


def _focus() -> dict:
    return json_api.example_project_document()["project"]


def test_the_default_evaluation_is_the_readings_and_states_its_scope():
    reply = json_api.evaluate_project(_focus())
    assert reply["ok"] and json.loads(json.dumps(reply)) == reply  # (plain JSON)
    scope = reply["evaluation"]
    assert scope["requested"] == ["readings"] and not scope["complete"]
    assert "untraced_light" not in scope["executed"] and "plane_bounds" in [c["capability"] for c in scope["not_computed"]]
    assert {t["target_id"] for t in scope["targets_not_evaluated"]} == {"focus_position", "spot_radius"}
    assert reply["targets"] == [] and reply["outcome"]["requirements"]["declared"] is True
    (plane,) = reply["monitors"]
    assert plane["quantity_state"] == "NOT_COMPUTED" and plane["power_complete"] is False and plane["arrivals"]
    # (the plane bounds did not run: the untraced power is not computed, never a number)
    assert plane["untraced_power_W"] is None and "power_upper_bound_W" not in plane
    assert reply["model_scope_issues"] == [] and reply["identity"]["engine_input_hash"]


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_no_summary_reads_pass_when_declared_targets_exist_and_none_was_assessed(lossless):
    """An unassessed evaluation has no verdict. The diagnostic summary also preserves known bulk-loss uncertainty."""
    from local_helpers import catalog_document, with_lossless_bulk
    from os_optics.local import EvaluationSession

    catalog = catalog_document()
    if lossless:
        catalog = with_lossless_bulk(catalog)
    document = _focus()
    session = EvaluationSession(catalog=catalog)
    none = session.evaluate(document)
    assert none.result.targets == () and len(none.result.evaluation.targets_not_evaluated) == 2
    unassessed = "NOT_APPLICABLE" if lossless else "UNKNOWN"
    assert none.result.status.value == none.status == unassessed
    reply = json_api.evaluate_project(document, catalog=catalog)
    assert reply["status"] == reply["diagnostic_summary"] == unassessed
    assert none.results().describe()["status"]["engine_status"] == unassessed
    if not lossless:
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in none.result.issues}
    # The readings agree when an evaluation also assesses a geometric target.
    one = session.evaluate(document, targets=["focus_position"])
    expected = "PASS" if lossless else "UNKNOWN"
    assert one.status == expected and one.result.outcome.requirements.met == ("focus_position",)
    one_reply = json_api.evaluate_project(document, targets=["focus_position"], catalog=catalog)
    assert one_reply["status"] == expected and one_reply["requirements"]["met"] == ["focus_position"]
    assert [o.arrivals for o in one.result.observations] == [o.arrivals for o in none.result.observations]
    free = json_api.evaluate_project({**document, "targets": []}, catalog=catalog)
    assert free["status"] == expected and not free["requirements"]["declared"]


#: Selections of the example Project (``focus_position``, ``spot_radius``) and the targets each assesses.
COVERAGE_SELECTIONS = {
    "readings": ({}, []),
    "one target": ({"targets": ["spot_radius"]}, ["spot_radius"]),
    "the other": ({"targets": ["focus_position"]}, ["focus_position"]),
    "requirements": ({"outputs": ["readings", "requirements"]}, ["focus_position", "spot_radius"]),
    "complete": ({"outputs": ["complete"]}, ["focus_position", "spot_radius"]),
}


def test_the_lead_requirements_name_every_declared_target_once():
    """The reply's lead ``requirements`` partitions the declared targets: the assessed ones by assessment, the others in
    ``not_assessed`` with the reason, in Project order -- so an evaluation that assessed nothing never reads as one whose
    targets are all met. ``Evaluation.requirements()`` is the same block; ``outcome.requirements`` keeps the record's shape."""
    from os_optics.local import EvaluationSession

    declared = [t["target_id"] for t in _focus()["targets"]]
    session = EvaluationSession()
    for name, (selection, assessed) in COVERAGE_SELECTIONS.items():
        reply = json_api.evaluate_project(_focus(), **selection)
        lead = reply["requirements"]
        lists = [lead[key] for key in ("met", "unmet", "indeterminate", "not_applicable")]
        named = [i for ids in lists for i in ids] + [row["target_id"] for row in lead["not_assessed"]]
        assert sorted(named) == sorted(declared) and len(named) == len(set(named)), name
        assert [row["target_id"] for row in lead["not_assessed"]] == [i for i in declared if i not in assessed], name
        assert {row["reason"] for row in lead["not_assessed"]} <= {"NOT_REQUESTED"}, name
        assert lead["not_assessed"] == [{"target_id": t["target_id"], "reason": t["reason"]}
                                        for t in reply["evaluation"]["targets_not_evaluated"]], name
        assert sorted(i for ids in lists for i in ids) == sorted(assessed), name
        assert "not_assessed" not in reply["outcome"]["requirements"], name  # (the record's mirror is untouched)
        evaluation = session.evaluate(_focus(), **selection)
        assert evaluation.requirements() == lead, name


def test_a_result_that_does_not_name_each_declared_target_once_is_refused():
    """``require_complete`` checks the partition by id, not by count: a result whose evaluated and not-evaluated targets
    have the right number but not the declared ids is no evaluation of that Project."""
    import pytest

    from os_optics.local.errors import LocalRunError
    from os_optics.local.run import require_complete, targets_accounted

    from os_optics.local import EvaluationSession

    evaluation = EvaluationSession().evaluate(_focus(), targets=["focus_position"])
    request, result = evaluation.request, evaluation.result
    assert targets_accounted(request, result)
    require_complete(request, result)
    (skipped,) = result.evaluation.targets_not_evaluated
    renamed = result.model_copy(update={"evaluation": result.evaluation.model_copy(update={
        "targets_not_evaluated": (skipped.model_copy(update={"target_id": "focus_position"}),)})})
    assert not targets_accounted(request, renamed)
    with pytest.raises(LocalRunError):
        require_complete(request, renamed)


def test_a_complete_evaluation_agrees_with_run_project():
    document = _focus()
    reply = json_api.evaluate_project(document, ["complete"])
    run = json_api.run_project(document)
    # (a complete evaluation computed everything an evaluation computes: the run directory is write_run's)
    assert reply["evaluation"]["complete"] and reply["evaluation"]["not_computed"] == []
    summary = run["summary"]
    assert [(t["target_id"], t["status"], t["assessment"], t["value"]) for t in reply["targets"]] == [
        (t["target_id"], t["status"], t["assessment"], t["value"]) for t in summary["targets"]]
    assert reply["monitors"] == summary["observations"] and reply["status"] == run["status"]
    assert reply["identity"]["result_hash"] == summary["engine_result_hash"]
    assert "files" not in reply
    # (a complete evaluation computed the view: its scene is the run's)
    assert reply["optical_scene"] == run["optical_scene"]


def test_an_evaluation_that_computes_the_view_returns_its_scene():
    """``view`` among the outputs: the reply carries the scene (the run's scene contract, ``OpticalSimulationSceneV1``)
    bound to this evaluation's own result, beside the scope that says what else was and was not computed; it draws what
    the complete run draws. Without ``view`` the reply has no scene."""
    from os_optics._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1

    document = _focus()
    reply = json_api.evaluate_project(document, ["view"])
    assert reply["ok"] and json.loads(json.dumps(reply)) == reply  # (plain JSON)
    scene = reply["optical_scene"]
    assert OpticalSimulationSceneV1.model_validate(scene).payload() == scene
    scope = reply["evaluation"]
    assert not scope["complete"] and "view" in scope["executed"] and reply["targets"] == []
    assert {"plane_bounds", "requirements", "record"} <= {c["capability"] for c in scope["not_computed"]}
    assert scene["engine_run"]["result_hash"]["digest"] == reply["identity"]["result_hash"]
    run = json_api.run_project(document)["optical_scene"]
    assert scene["engine_run"]["result_hash"] != run["engine_run"]["result_hash"]  # (it names the narrower result)
    assert scene["engine_run"]["trace_hash"] == run["engine_run"]["trace_hash"]
    assert {k: v for k, v in scene.items() if k != "engine_run"} == {k: v for k, v in run.items() if k != "engine_run"}
    # (what the view draws, as for run_project; display only)
    every_ray = json_api.evaluate_project(document, ["view"], None, None, {"pupil_ray_selection": "ALL"})
    assert every_ray["identity"] == reply["identity"] and every_ray["optical_scene"] != scene
    assert json_api.run_project(document, view_settings={"pupil_ray_selection": "ALL"})["optical_scene"]["beam_paths"] == \
        every_ray["optical_scene"]["beam_paths"]
    refused = json_api.evaluate_project(document, ["view"], None, None, {"pupil_ray_selection": "SOME"})
    assert refused == json_api.run_project(document, {"pupil_ray_selection": "SOME"})  # (refused as run_project refuses it)
    assert refused["error"]["code"] == "LOCAL_RUN_ERROR" and "invalid view settings" in refused["error"]["problems"][0]["message"]
    reply, files = json_api.call_method("evaluate_project", {"document": document, "outputs": ["view"],
                                                             "view_settings": {"pupil_ray_selection": "ALL"}})
    assert files is None and reply["optical_scene"] == every_ray["optical_scene"]
    assert "optical_scene" not in json_api.evaluate_project(document, ["readings", "requirements"])


def test_the_json_host_path_and_the_refusals():
    document = _focus()
    reply, files = json_api.call_method("evaluate_project", {"document": document, "outputs": ["readings", "requirements"]})
    assert reply["ok"] and files is None and {t["target_id"] for t in reply["targets"]} == {"focus_position", "spot_radius"}
    excluded = json_api.evaluate_project(document, ["record"], ["untraced_light"])
    assert excluded["error"]["code"] == "EXCLUDED_CAPABILITY_REQUIRED"
    assert "Remove untraced_light from exclude, or do not request record" in excluded["error"]["message"]
    assert json_api.evaluate_project(document, ["no_such"])["error"]["code"] == "UNKNOWN_CAPABILITY"
    for name in ("layout", "control"):  # (capabilities of an Engine request: a Project's request runs neither)
        refused = json_api.evaluate_project(document, [name])["error"]
        assert refused["code"] == "UNKNOWN_CAPABILITY" and f"os_optics.evaluate(request, capabilities=['{name}'])" in refused["message"]
        assert f"{name} is false" not in refused["message"]
    files = json_api.evaluate_project(document, ["run_files"])["error"]  # (an evaluation writes nothing)
    assert files["code"] == "UNKNOWN_CAPABILITY" and "Evaluation.write_run" in files["message"] and "project run" in files["message"]
    # (the plane readings come with the models' applicability: neither it nor the ray paths it reads can be excluded)
    for excluded in ("ray_paths", "model_scope"):
        refused = json_api.evaluate_project(document, ["plane_readings", "requirements"], [excluded])["error"]
        assert refused["code"] == "EXCLUDED_CAPABILITY_REQUIRED" and "requires model_scope" in refused["message"]
    assert json_api.evaluate_project(document, None, None, ["nope"])["error"]["code"] == "UNKNOWN_TARGET"
    assert json_api.evaluate_project(document, 5)["error"]["code"] == "INPUT_INVALID"
    broken = copy.deepcopy(document)
    broken["components"] = [c for c in broken["components"] if c["optical_role"] != "source"]
    assert json_api.evaluate_project(broken)["error"]["code"] == "LOCAL_RUN_ERROR"


def test_a_stopped_evaluation_replies_with_its_stop_record():
    monitor = RunMonitor(lambda event: monitor.cancel("stop") if event.get("phase") == "trace" else None)
    with monitored(monitor):
        reply = json_api.evaluate_project(_focus())
    assert reply["ok"] is False and reply["error"]["code"] == "RUN_STOPPED" and reply["stopped"]["stop"]["phase"] == "trace"
    # (stopped after the lowering: the record names the Engine input, as a stopped run_project's does)
    monitor = RunMonitor(lambda event: monitor.cancel("stop") if event.get("phase") == "trace" else None)
    with monitored(monitor):
        run = json_api.run_project(_focus())
    assert reply["stopped"]["engine_input_hash"] == run["stopped"]["engine_input_hash"] is not None


def test_the_capability_index_and_descriptors():
    discovery = json_api.capabilities()["evaluation"]
    assert discovery["evaluate_project"] == json_api.API_METHODS["evaluate_project"].signature()
    assert discovery["evaluation_capabilities"] == json_api.API_METHODS["evaluation_capabilities"].signature()
    index = json_api.evaluation_capabilities()
    assert index["ok"] and [row["id"] for row in index["capabilities"]] == discovery["capabilities"]
    assert {name: row["members"] for name, row in index["presets"].items()} == discovery["presets"]
    assert [row["id"] for row in index["analyses"]] == discovery["analyses"]
    ids = [row["id"] for row in index["capabilities"]]
    assert ids[0] == "solid_check" and ids[-1] == "view" and "imaging" in [row["id"] for row in index["analyses"]]
    # (only what a Project evaluation selects: the layout checks and the control analysis are an Engine request's)
    assert not {"layout", "control", "run_files"} & set(ids) and discovery["exports"] == ["run_files"]
    assert [row["id"] for row in index["exports"]] == ["run_files"] and "write_run" in index["exports"][0]["entry_point"]
    # (the refusal codes a Project evaluation can return: CAPABILITY_NOT_ENABLED is an Engine request's, whose flags a
    # Project does not set)
    assert discovery["error_codes"] == ["UNKNOWN_CAPABILITY", "UNKNOWN_TARGET", "EXCLUDED_CAPABILITY_REQUIRED",
                                        "TARGET_SELECTION_INCOMPLETE"]
    assert index["presets"]["readings"]["members"] == ["scene", "beam_trace", "secondary_rays", "plane_readings", "ray_paths", "model_scope"]
    assert "untraced_light" in index["presets"]["complete"]["members"] and "run_files" not in index["presets"]["complete"]["members"]
    assert not {"layout", "control"} & set(index["presets"]["complete"]["members"])
    assert {row["id"]: row.get("local_requires") for row in index["capabilities"] if row.get("local_requires")} == {
        "plane_readings": ["model_scope"], "requirements": ["model_scope"]}
    bounds = json_api.evaluation_capabilities("plane_bounds")["capability"]
    assert bounds["requires"] == ["plane_readings", "power_ledger", "untraced_light"] and "power_W" in bounds["metrics"]
    # (the return-light analysis is made inside the phases of its readers, the plane bounds first)
    assert bounds["phases"] == ["bounds", "return_analysis"] and "max_return_analysis_work" in bounds["compute_budget"]
    assert bounds["presets"] == ["complete"]
    readings = json_api.evaluation_capabilities("plane_readings")["capability"]
    assert "beam_radius_mm" in readings["metrics"] and readings["presets"] == ["complete", "readings"]
    assert readings["local_requires"] == ["model_scope"]
    # (a Project sets no Engine request flag: the descriptor names none; the record states that it needs everything)
    assert "request_flag" not in readings
    record = json_api.evaluation_capabilities("record")["capability"]
    assert record["requires_complete_scope"] is True and record["requires"] == ["solid_check", "model_scope"]
    imaging = json_api.evaluation_capabilities("imaging")["capability"]
    assert imaging["kind"] == "analysis" and imaging["request_schema"] == "imaging-request"
    assert imaging["entry_point"] == "os_optics.results.RunResults.imaging(monitor, request=None)"
    assert imaging["command"] == "os-optics results imaging RUN --monitor ID [--request FILE]"
    run_files = json_api.evaluation_capabilities("run_files")["capability"]
    assert run_files["kind"] == "export" and run_files["command"] == "os-optics project run PROJECT --out-dir DIR"
    assert json_api.evaluation_capabilities("solid_check")["capability"]["produces"] == ["optical_solids"]
    unknown = json_api.evaluation_capabilities("nope")["error"]
    assert unknown["code"] == "UNKNOWN_CAPABILITY" and "the presets are complete, readings" in unknown["message"]
    assert json_api.evaluation_capabilities(["readings"])["error"]["code"] == "UNKNOWN_CAPABILITY"
    layout = json_api.evaluation_capabilities("layout")["error"]
    assert layout["code"] == "UNKNOWN_CAPABILITY" and "os_optics.evaluate(request, capabilities=['layout'])" in layout["message"]


def test_a_preset_is_described_by_what_it_selects_and_what_it_leaves_out():
    index = json_api.evaluation_capabilities()
    every = [row["id"] for row in index["capabilities"]]
    for name in ("readings", "complete"):
        reply = json_api.evaluation_capabilities(name)
        assert reply["ok"] and set(reply) == {"ok", "preset"}
        preset = reply["preset"]
        assert preset["id"] == name and preset["summary"] == index["presets"][name]["summary"]
        assert preset["members"] == index["presets"][name]["members"]
        assert preset["leaves_out"] == [c for c in every if c not in preset["members"]]
    readings = json_api.evaluation_capabilities("readings")["preset"]
    assert readings["leaves_out"] == ["solid_check", "power_ledger", "untraced_light", "plane_bounds", "surface_exposures",
                                      "beam_pairs", "requirements", "record", "view"]
    assert json_api.evaluation_capabilities("complete")["preset"]["leaves_out"] == []
    # (what the preset leaves out is what its default reply lists as not computed, and its summary names the requirements)
    reply = json_api.evaluate_project(_focus(), ["readings"])
    assert [row["capability"] for row in reply["evaluation"]["not_computed"]] == readings["leaves_out"]
    assert "the requirements (no declared target is assessed)" in readings["summary"]


def test_a_project_with_configurations_is_refused_with_the_ways_to_one_configuration():
    from pathlib import Path

    from os_optics.local import load_project

    zoom = (Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json").read_text(encoding="utf-8")
    refused = json_api.evaluate_project(json.loads(zoom))["error"]
    assert refused["code"] == "LOCAL_RUN_ERROR" and refused["problems"][0]["code"] == "CONFIGURATION_NOT_SELECTED"
    message = refused["problems"][0]["message"]
    assert "'--configuration ID'" in message and "ProjectDocumentV2.configured(id)" in message
    # (the way out the message and the docstring name: one configuration's Project is evaluated as any Project)
    one = json_api.evaluate_project(load_project(zoom).configured("wide").payload())
    assert one["ok"] and one["evaluation"]["requested"] == ["readings"]
    assert "configuration_id= of evaluate_project" in message and "request field configuration_id" in message
    named = json_api.evaluate_project(json.loads(zoom), configuration_id="wide")
    assert named["ok"] and named["monitors"] == one["monitors"] and named["identity"] == one["identity"]
    assert "CONFIGURATION_NOT_SELECTED" in json_api.evaluate_project.__doc__


def test_a_candidate_loop_written_from_discovery_finds_the_best_lens_position():
    """An agent's bounded search: from the published discovery alone it picks the preset that computes the plane readings
    without the untraced-light probes, learns that the spot radius target reads them, then moves the lens along the beam and
    keeps the position of the smallest spot -- every candidate evaluated by ``evaluate_project``."""

    discovery = json_api.capabilities()["evaluation"]
    preset = next(name for name, members in discovery["presets"].items()
                  if "plane_readings" in members and "untraced_light" not in members)
    document = _focus()
    target = next(t for t in document["targets"]
                  if t["metric"] in json_api.evaluation_capabilities("plane_readings")["capability"]["metrics"]
                  and t["metric"] == "beam_radius_mm")
    lens = next(c for c in document["components"] if c["optical_role"] == "lens")
    nominal = lens["nominal_pose"]["position_mm"][0]

    def spot(offset_mm: float) -> float:
        candidate = copy.deepcopy(document)
        next(c for c in candidate["components"] if c["instance_id"] == lens["instance_id"])["nominal_pose"]["position_mm"][0] = nominal + offset_mm
        reply = json_api.evaluate_project(candidate, [preset], None, [target["target_id"]])
        assert reply["ok"] and "untraced_light" not in reply["evaluation"]["executed"]
        (row,) = reply["targets"]
        return row["value"]

    low, high = -6.0, 6.0  # (mm, the travel of the search)
    golden = (5 ** 0.5 - 1) / 2
    a, b = high - golden * (high - low), low + golden * (high - low)
    fa, fb = spot(a), spot(b)
    for _ in range(14):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - golden * (high - low)
            fa = spot(a)
        else:
            low, a, fa = a, b, fb
            b = low + golden * (high - low)
            fb = spot(b)
    best = (low + high) / 2
    radius = spot(best)
    assert -6.0 < best < 6.0 and radius < spot(-6.0) and radius < spot(6.0)
    assert radius <= min(spot(best - 0.5), spot(best + 0.5))
