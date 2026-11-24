"""Target parameters never silently drop out.

A target's ``parameters`` select the beam(s) its metric reads (``path_id`` / ``crossing_index``; two-beam ``path_id_a`` /
``path_id_b`` / ``crossing_index_a`` / ``crossing_index_b``) or carry a metric-specific number (``dose_mj_per_cm2``).
Formerly a selector of the wrong JSON type, a fractional crossing index or an unknown / misspelled key was ignored
(or truncated) and the target then passed on another beam. Such a target is never assessed on another beam: it is not
assessed at all. ``lower_project`` (``project run``, ``check_project`` / ``run_project``, the Designer, the server's
re-lowering) carries it as a requirement that cannot be read (``LoweredProject.target_problems``: ``TARGET_PARAMETER_INVALID``
naming the target and the parameter; a check warning), the run reports it ``UNKNOWN`` / ``INDETERMINATE`` with that code
(CLI exit 2) and evaluates everything else as before. The Engine ``Target`` contract still refuses such parameters in an
Engine request; valid selectors behave exactly as before.
"""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from os_optics.cli import main
from os_optics.contracts.project import Target
from os_optics.local import EXAMPLE_KINDS, LocalRunError, LocalRunErrorCode, example_project_of_kind, json_api, lower_project, run_optical_project
from os_optics.validation import validate_json
from os_optics.validation.descriptors import DESCRIPTORS, DOF_DESCRIPTORS, STAGE_DESCRIPTORS
from os_optics.validation.mechanical import KIND_PARAMETERS, MECHANICAL_METRICS
from os_optics.validation.alignment import ALIGNMENT_METRICS
from os_optics.validation.exposures import ELEMENT_METRICS
from os_optics.validation.metrics import OBSERVATION_METRICS, PAIR_METRICS
from os_optics.validation.scope import OUT_OF_SCOPE_METRICS
from os_optics.validation.target_parameters import (PARAMETER_TYPES, REQUIRED_PARAMETERS, accepted_parameters,
                                                         parameter_problems, requires_observation)

from declared_helpers import declared_project
from local_helpers import catalog_document


def _selector_fixture(kind):
    """Test fixture optics are ideal lossless so this suite isolates parameter selection."""
    project = declared_project(example_project_of_kind(kind).payload(), catalog_document())
    for material in project.get("model_materials", []):
        material["absorption_coefficient_per_mm"] = "0"
    return project


# the focus example's spot radius on its first (only) crossing of the focal plane (in air at 20 C and 101325 Pa, n_air ~
# 1.00027, the absolute-index N-BK7 lens is weaker relative to the air, f ~ +8e-4, and the beam's wavelength there is lambda / n_air,
# -2.7e-4: 0.025005956 in unit index before)
SPOT_RADIUS_MM = 0.0250192099


def _focus(parameters):
    project = _selector_fixture("focus")
    next(t for t in project["targets"] if t["target_id"] == "spot_radius")["parameters"] = parameters
    return project


def _pair(parameters, target_id="fringe_period"):
    project = _selector_fixture("two_beam_interference")
    next(t for t in project["targets"] if t["target_id"] == target_id)["parameters"] = parameters
    return project


def _spot(run):
    return next(t for t in run.result.targets if t.target_id == "spot_radius")


def _not_assessed(project, target_id="spot_radius"):
    """The problems of the requirements that cannot be read; the target is in the request with only its id, metric and
    marker (it selects nothing)."""

    lowered = lower_project(project)
    (marked,) = [t for t in lowered.request.project.targets if t.target_id == target_id]
    assert marked.not_assessable is not None and marked.parameters == {} and marked.observation_id is None
    assert marked.not_assessable.code == lowered.target_problems[0].code.value
    return lowered.target_problems


# ------------------------------------------------------------------------------------- the issue's table (focus example)
@pytest.mark.parametrize(("value", "key"), [
    ("1", "crossing_index"),       # a string is not a crossing index (was: ignored -> PASS on crossing 0)
    (0.9, "crossing_index"),       # never truncated (was: int(0.9) == 0 -> PASS)
    (-0.9, "crossing_index"),      # never truncated (was: int(-0.9) == 0 -> PASS)
    (1.5, "crossing_index"),
    (123, "path_id"),              # a number is not a path id (was: ignored -> PASS on SRC#0)
    (0.0, "path_id"),
])
def test_a_selector_of_the_wrong_type_leaves_the_target_unassessed(value, key):
    (problem,) = _not_assessed(_focus([{"key": key, "value": value}]))
    assert problem.code is LocalRunErrorCode.TARGET_PARAMETER_INVALID
    assert problem.location == f"target spot_radius parameters.{key}"
    assert repr(value) in problem.message or str(value) in problem.message


@pytest.mark.parametrize("key", ["does_not_exist", "crossin_index", "pathid", "path_id_a", "crossing_index_b", "dose_mj_per_cm2"])
def test_a_key_the_metric_does_not_take_leaves_the_target_unassessed(key):
    """Unknown, misspelled and other metrics' keys (a pair selector or the exposure dose on a single-beam metric)."""

    (problem,) = _not_assessed(_focus([{"key": key, "value": "SRC#0" if key.startswith("path") else 0}]))
    assert problem.code is LocalRunErrorCode.TARGET_PARAMETER_INVALID
    assert problem.location == f"target spot_radius parameters.{key}"
    assert "'beam_radius_mm' takes the parameters ['crossing_index', 'path_id']" in problem.message


def test_a_boolean_is_never_a_crossing_index():
    """``true`` used to be read as 1.0 by the contract; the Project contract refuses a boolean parameter value."""

    with pytest.raises(LocalRunError) as refused:
        lower_project(_focus([{"key": "crossing_index", "value": True}]))
    (problem,) = refused.value.problems
    assert problem.code is LocalRunErrorCode.PROJECT_INVALID and problem.location == "target spot_radius"
    assert "'crossing_index'" in problem.message and "boolean" in problem.message


def test_every_problem_is_listed_at_once():
    problems = _not_assessed(_focus([{"key": "crossing_index", "value": "1"}, {"key": "path_id", "value": 7}, {"key": "x", "value": "y"}]))
    assert [p.location for p in problems] == ["target spot_radius parameters.crossing_index", "target spot_radius parameters.path_id",
                                              "target spot_radius parameters.x"]
    assert {p.code for p in problems} == {LocalRunErrorCode.TARGET_PARAMETER_INVALID}


def test_valid_selectors_behave_as_before():
    passed = _spot(run_optical_project(_focus([{"key": "path_id", "value": "SRC#0"}, {"key": "crossing_index", "value": 0}])))
    assert passed.status.value == "PASS" and passed.value == pytest.approx(SPOT_RADIUS_MM, rel=1e-6)
    # A crossing or a path that does not occur selects no beam: UNKNOWN with TARGET_SELECTION_EMPTY naming what the run has
    #, never a value and never "not reached" (which would read as a measured result).
    for parameters, says in (([{"key": "crossing_index", "value": 1}], "crossing_index 1 does not occur"),   # no second crossing
                             ([{"key": "crossing_index", "value": 1.0}], "crossing_index 1 does not occur"),  # an integer
                             ([{"key": "crossing_index", "value": -1}], "crossing_index -1 does not occur"),
                             ([{"key": "path_id", "value": "123"}], "path_id '123' is not a beam path")):     # no such path
        empty = _spot(run_optical_project(_focus(parameters)))
        assert (empty.status.value, empty.value, empty.issue_codes) == ("UNKNOWN", None, ("TARGET_SELECTION_EMPTY",)), parameters
        assert says in empty.message, empty.message


# ------------------------------------------------------------------------------------------------- CLI and json_api
def _write(tmp_path, project, name="project.json"):
    path = tmp_path / name
    path.write_text(json.dumps(project), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(("parameters", "code", "exit_code"), [
    ([{"key": "crossing_index", "value": "1"}], "TARGET_PARAMETER_INVALID", 2),
    ([{"key": "crossing_index", "value": 0.9}], "TARGET_PARAMETER_INVALID", 2),
    ([{"key": "crossing_index", "value": -0.9}], "TARGET_PARAMETER_INVALID", 2),
    ([{"key": "crossing_index", "value": True}], "PROJECT_INVALID", 65),
    ([{"key": "path_id", "value": 123}], "TARGET_PARAMETER_INVALID", 2),
    ([{"key": "does_not_exist", "value": "must-match"}], "TARGET_PARAMETER_INVALID", 2),
    ([{"key": "crossing_index", "value": 1}], "TARGET_SELECTION_EMPTY", 2),
    ([{"key": "path_id", "value": "SRC#0"}], None, 0),
])
def test_cli_project_run_exit_codes(tmp_path, capsys, parameters, code, exit_code):
    out = tmp_path / "run"
    assert main(["project", "run", _write(tmp_path, _focus(parameters)), "--out-dir", str(out)]) == exit_code
    printed = json.loads(capsys.readouterr().out)
    if code == "PROJECT_INVALID":  # (not a valid Project document: refused, nothing written)
        assert printed["status"] == "ERROR" and [p["code"] for p in printed["problems"]] == [code]
        assert printed["problems"][0]["location"].startswith("target spot_radius")
        assert not out.exists()
        return
    spot = next(t for t in printed["targets"] if t["target_id"] == "spot_radius")
    assert spot["status"] == {0: "PASS", 2: "UNKNOWN"}[exit_code]
    assert spot["issue_codes"] == ([] if code is None else [code])
    assert (out / "manifest.json").exists()  # the run is written, the requirement counted as undetermined


def test_json_api_check_warns_and_run_leaves_a_malformed_selector_unassessed():
    project = _focus([{"key": "crossing_index", "value": "1"}])
    check, run = json_api.check_project(project), json_api.run_project(project)
    json.dumps(check, allow_nan=False)
    assert check["ok"] is True and run["ok"] is True
    (warning,) = [w for w in check["warnings"] if w["code"] == "TARGET_PARAMETER_INVALID"]
    assert warning["location"] == "target spot_radius parameters.crossing_index" and warning["details"] == {"target_id": "spot_radius"}
    assert warning["message"].startswith("parameter 'crossing_index' must be an integer (a JSON number without a fractional part), not '1' (a string)")
    spot = next(t for t in run["summary"]["targets"] if t["target_id"] == "spot_radius")
    assert (spot["status"], spot["value"], spot["issue_codes"]) == ("UNKNOWN", None, ["TARGET_PARAMETER_INVALID"])
    assert "spot_radius" in run["local_validation"]["simulation"]["outcome"]["requirements"]["indeterminate"]
    reply = json_api.check_project(_focus([{"key": "crossing_index", "value": True}]))
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "PROJECT_INVALID"
    assert json_api.check_project(_focus([{"key": "crossing_index", "value": 0}]))["ok"] is True


# ------------------------------------------------------------------------------------------------- two-beam metrics
@pytest.mark.parametrize(("parameters", "key"), [
    ([{"key": "path_id_a", "value": 123}, {"key": "path_id_b", "value": "SRC#0.0"}], "path_id_a"),
    ([{"key": "path_id_a", "value": "SRC#0"}, {"key": "path_id_b", "value": "SRC#0.0"},
      {"key": "crossing_index_a", "value": 0}, {"key": "crossing_index_b", "value": "0"}], "crossing_index_b"),
    ([{"key": "path_id_a", "value": "SRC#0"}, {"key": "path_id_b", "value": "SRC#0.0"},
      {"key": "crossing_index_a", "value": 0.5}, {"key": "crossing_index_b", "value": 0}], "crossing_index_a"),
    ([{"key": "path_id", "value": "SRC#0"}], "path_id"),                # a single-beam selector on a pair metric
    ([{"key": "crossing_index", "value": 0}], "crossing_index"),
    ([{"key": "path_id_aa", "value": "SRC#0"}], "path_id_aa"),          # misspelled
    ([{"key": "crossing_index_a", "value": 0}], "crossing_index_a"),    # crossings number the two named paths only
    ([{"key": "path_id_a", "value": "SRC#0"}, {"key": "crossing_index_a", "value": 0}, {"key": "crossing_index_b", "value": 0}],
     "crossing_index_a"),
])
def test_pair_selectors_are_checked_the_same_way(parameters, key):
    problems = _not_assessed(_pair(parameters), "fringe_period")
    assert {p.code for p in problems} == {LocalRunErrorCode.TARGET_PARAMETER_INVALID}
    assert f"target fringe_period parameters.{key}" in [p.location for p in problems]


def test_valid_pair_selectors_behave_as_before():
    def fringe(parameters):
        return next(t for t in run_optical_project(_pair(parameters)).result.targets if t.target_id == "fringe_period")

    full = [{"key": "path_id_a", "value": "SRC#0"}, {"key": "path_id_b", "value": "SRC#0.0"},
            {"key": "crossing_index_a", "value": 0}, {"key": "crossing_index_b", "value": 0}]
    assert fringe(full).status.value == "PASS"
    assert fringe(full[:2]).status.value == "PASS"
    assert fringe([{"key": "path_id_a", "value": "SRC#0.0"}]).status.value == "PASS"  # one path names operand a
    late = [*full[:3], {"key": "crossing_index_b", "value": 1}]                         # SRC#0.0 crosses once
    assert fringe(late).status.value == "UNKNOWN" and fringe(late).issue_codes == ("TARGET_SELECTION_EMPTY",)
    # A single named path is applied (it used to be ignored for the selection): a path that does not exist selects no pair.
    missing = fringe([{"key": "path_id_a", "value": "SRC#9"}])
    assert missing.status.value == "UNKNOWN" and missing.issue_codes == ("TARGET_SELECTION_EMPTY",)


def test_the_exposure_dose_is_a_number():
    (problem,) = _not_assessed(_pair([{"key": "dose_mj_per_cm2", "value": "100"}], target_id="fringe_period"), "fringe_period")
    assert "'fringe_period_nm' takes the parameters" in problem.message  # not a parameter of fringe_period_nm
    assert parameter_problems("estimated_exposure_time_s", {"dose_mj_per_cm2": "100"}) == [
        ("dose_mj_per_cm2", "parameter 'dose_mj_per_cm2' must be a number, not '100' (a string)")]
    assert parameter_problems("estimated_exposure_time_s", {"dose_mj_per_cm2": 100}) == []


def test_the_exposure_dose_is_required_and_positive():
    """A target on estimated_exposure_time_s without its dose, or with a dose that is not greater than 0, is refused as input
    (it ran to a target without a value before) by the parameter check and by the Target contract; the local lowering
    carries it as a requirement that cannot be read (not assessed, with the problem)."""
    assert parameter_problems("estimated_exposure_time_s", {}) == [
        ("dose_mj_per_cm2", "metric 'estimated_exposure_time_s' needs parameter 'dose_mj_per_cm2': add it to the target's parameters")]
    for dose in (0, -1.5):
        assert parameter_problems("estimated_exposure_time_s", {"dose_mj_per_cm2": dose}) == [
            ("dose_mj_per_cm2", f"parameter 'dose_mj_per_cm2' must be greater than 0, not {dose}")]
    with pytest.raises(ValidationError, match="needs parameter 'dose_mj_per_cm2'"):
        Target(target_id="t", metric="estimated_exposure_time_s", observation_id="o", maximum=1.0)
    project = _pair([], target_id="fringe_period")
    exposure = next(t for t in project["targets"] if t["target_id"] == "fringe_period")
    exposure.update(metric="estimated_exposure_time_s", unit="s")
    for parameters, words in (([], "needs parameter"), ([{"key": "dose_mj_per_cm2", "value": 0}], "greater than 0")):
        exposure["parameters"] = parameters
        (problem,) = _not_assessed(project, target_id="fringe_period")
        assert problem.code is LocalRunErrorCode.TARGET_PARAMETER_INVALID and words in problem.message
        assert problem.location == "target fringe_period parameters.dose_mj_per_cm2"
    exposure["parameters"] = [{"key": "dose_mj_per_cm2", "value": 100}]
    assert lower_project(project).request.project.targets


def test_a_metric_that_reads_a_plane_needs_its_observation_id():
    """The metrics whose descriptor lists the selector observation_id -- read from the descriptors, not from a second list --
    refuse a target without one; every supported metric still builds a target that states what it needs."""
    from os_optics.validation.descriptors import DESCRIPTOR_BY_NAME
    from os_optics.validation.detection import DETECTION_METRICS
    from os_optics.validation.metrics import ALL_SUPPORTED

    plane_metrics = {m for m in ALL_SUPPORTED if requires_observation(m)}
    assert plane_metrics == set(OBSERVATION_METRICS) | set(PAIR_METRICS) | set(ALIGNMENT_METRICS) | set(DETECTION_METRICS)
    for metric in sorted(ALL_SUPPORTED):
        parameters = {key: 1.0 for key in REQUIRED_PARAMETERS.get(metric, ())}
        fields = dict(target_id="t", metric=metric, maximum=1.0, parameters=parameters)
        assert Target(observation_id="o", device_id="d", **fields).metric == metric
        if metric in plane_metrics:
            assert "observation_id" in DESCRIPTOR_BY_NAME[metric].selectors
            with pytest.raises(ValidationError, match="reads an observation plane: set observation_id"):
                Target(device_id="d", **fields)
        else:
            assert Target(device_id="d", **fields).observation_id is None
    # the local lowering names the target and carries it as not assessed
    project = _focus([])
    next(t for t in project["targets"] if t["target_id"] == "spot_radius")["observation_id"] = None
    (problem,) = _not_assessed(project)
    assert problem.code is LocalRunErrorCode.TARGET_INVALID and problem.location == "target spot_radius"
    assert "set observation_id" in problem.message


# ------------------------------------------------------------------------------------------------ Engine requests
def test_the_engine_target_contract_refuses_the_same_parameters():
    def target(metric="beam_radius_mm", **parameters):
        return Target(target_id="t", metric=metric, observation_id="o", maximum=1.0, parameters=parameters)

    assert target(crossing_index=1).parameters == {"crossing_index": 1.0}
    assert target(crossing_index=2.0, path_id="L#0").parameters == {"crossing_index": 2.0, "path_id": "L#0"}
    for bad in ({"crossing_index": "1"}, {"crossing_index": 0.9}, {"crossing_index": True}, {"path_id": 123}, {"crossin_index": 0},
                {"path_id_a": "L#0"}):
        with pytest.raises(ValidationError, match="parameter"):
            target(**bad)
    with pytest.raises(ValidationError, match="takes no parameters"):
        target(metric="fdtd", path_id="L#0")
    assert target(metric="not_a_metric", anything="x").parameters == {"anything": "x"}  # reported as an unknown metric instead


def test_a_target_that_is_not_assessable_states_only_its_id_metric_and_reason():
    marker = {"code": "TARGET_PARAMETER_INVALID", "message": "parameter 'x' is not taken"}
    marked = Target(target_id="t", metric="beam_radius_mm", not_assessable=marker)
    assert marked.model_dump(mode="json", exclude_none=True)["not_assessable"] == marker
    for stated in ({"observation_id": "o"}, {"maximum": 1.0}, {"parameters": {"crossing_index": 0}}, {"wavelength_nm": 500.0}):
        with pytest.raises(ValidationError, match="states only target_id, metric and not_assessable"):
            Target(target_id="t", metric="beam_radius_mm", not_assessable=marker, **stated)
    # without the marker a target keeps its bytes (the field is omitted when absent: request hashes do not move)
    plain = Target(target_id="t", metric="beam_radius_mm", observation_id="o", maximum=1.0)
    assert "not_assessable" not in plain.model_dump(mode="json")


def test_an_engine_request_with_a_marked_target_runs_the_rest_and_reports_it_unknown():
    lowered = lower_project(_focus([{"key": "crossing_index", "value": "1"}]))
    result = json.loads(validate_json(json.dumps(lowered.request.model_dump(mode="json"))))
    spot = next(t for t in result["targets"] if t["target_id"] == "spot_radius")
    assert (spot["status"], spot.get("value"), spot["issue_codes"]) == ("UNKNOWN", None, ["TARGET_PARAMETER_INVALID"])
    assert result["outcome"]["requirements"]["indeterminate"] == ["spot_radius"]
    focus = next(t for t in result["targets"] if t["target_id"] == "focus_position")
    assert focus["status"] == "PASS"
    # the physics is that of the Project without the target: the same request hash as the Project that drops it
    without = _selector_fixture("focus")
    without["targets"] = [t for t in without["targets"] if t["target_id"] != "spot_radius"]
    dropped = lower_project(without).request.model_dump(mode="json")
    kept = lowered.request.model_dump(mode="json")
    kept["project"]["targets"] = [t for t in kept["project"]["targets"] if t["target_id"] != "spot_radius"]
    assert kept == dropped


def test_an_engine_request_with_a_malformed_selector_is_invalid_input():
    request = json.loads(json.dumps(lower_project(_selector_fixture("focus")).request.model_dump(mode="json")))
    good = copy.deepcopy(request)
    next(t for t in good["project"]["targets"] if t["target_id"] == "spot_radius")["parameters"] = {"crossing_index": 0}
    assert json.loads(validate_json(json.dumps(good)))["status"] == "PASS"
    next(t for t in request["project"]["targets"] if t["target_id"] == "spot_radius")["parameters"] = {"crossing_index": "1"}
    result = json.loads(validate_json(json.dumps(request)))
    assert result["status"] == "FAIL" and [i["code"] for i in result["issues"]] == ["INVALID_INPUT"]
    assert "crossing_index" in json.dumps(result["issues"][0]) and not result["targets"]


# ------------------------------------------------------------------------------------------------- the key inventory
def test_the_accepted_keys_follow_the_metric_descriptors():
    single = ("crossing_index", "path_id")
    pair = ("crossing_index_a", "crossing_index_b", "path_id_a", "path_id_b")
    for descriptor in DESCRIPTORS:
        expected = single if descriptor.name in OBSERVATION_METRICS else () if descriptor.name in ELEMENT_METRICS else pair
        if descriptor.name == "estimated_exposure_time_s":
            expected = tuple(sorted((*pair, "dose_mj_per_cm2")))
        assert accepted_parameters(descriptor.name) == expected, descriptor.name
        assert any(descriptor.name in registry for registry in (OBSERVATION_METRICS, PAIR_METRICS, ALIGNMENT_METRICS, ELEMENT_METRICS))
    for descriptor in STAGE_DESCRIPTORS:
        assert accepted_parameters(descriptor.name) == ()
    for metric in OUT_OF_SCOPE_METRICS:
        assert accepted_parameters(metric) == ()
    assert accepted_parameters("not_a_metric") is None
    # a retained DOF's limit metrics: the DOF's axis and the metric's own numbers (validation/mechanical.py)
    for descriptor in DOF_DESCRIPTORS:
        _unit, kind, _motion = MECHANICAL_METRICS[descriptor.name]
        assert accepted_parameters(descriptor.name) == tuple(sorted(("axis", *KIND_PARAMETERS[kind]))), descriptor.name
    used = {key for d in DESCRIPTORS + DOF_DESCRIPTORS for key in accepted_parameters(d.name)}
    assert used == set(PARAMETER_TYPES)


@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_every_example_still_lowers(kind):
    assert lower_project(example_project_of_kind(kind)).request.project.targets


# ------------------------------------------------------------------- malformed parameter entries (review P2-1)
#: Entries that are not ``{key, value}`` objects, or keys that cannot be ordered: the contract's canonical sort used to raise a
#: raw KeyError / TypeError (CLI traceback, json_api exception, Designer ENGINE_FAILURE, server 500).
MALFORMED_ENTRIES = [
    [{"value": 1}],
    [1],
    [None],
    [{"key": 1, "value": 1}, {"key": "a", "value": 1}],
    [{"key": None, "value": 1}, {"key": "path_id", "value": "SRC#0"}],
    [{"key": "path_id", "value": "SRC#0"}, "crossing_index"],
]


@pytest.mark.parametrize("parameters", MALFORMED_ENTRIES, ids=repr)
def test_a_malformed_parameter_entry_is_a_structured_refusal(tmp_path, capsys, parameters):
    project = _focus(parameters)
    for reply in (json_api.check_project(project), json_api.run_project(project),
                  json_api.update_project_catalog(project)):
        json.dumps(reply, allow_nan=False)
        assert reply["ok"] is False and [p["code"] for p in reply["error"]["problems"]] == ["PROJECT_INVALID"], reply
        assert reply["error"]["problems"][0]["location"] == "target spot_radius"
        assert "Traceback" not in json.dumps(reply)
    out = tmp_path / "run"
    assert main(["project", "run", _write(tmp_path, project), "--out-dir", str(out)]) == 65
    captured = capsys.readouterr()
    assert json.loads(captured.out)["problems"][0]["code"] == "PROJECT_INVALID"
    assert "Traceback" not in captured.err and not out.exists()


@pytest.mark.parametrize("entries", [[1], [{"value": "x"}], [{"semantic_key": 1}, {"semantic_key": "a"}]], ids=repr)
def test_every_canonically_sorted_project_list_refuses_malformed_items(entries):
    """The same helper orders every map-like Project list (here a component's ``selected_parameters``)."""

    project = _selector_fixture("focus")
    project["components"][1]["selected_parameters"] = entries
    reply = json_api.check_project(project)
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "PROJECT_INVALID"
