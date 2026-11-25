"""Stated limits of a retained DOF: travel, hard limits, repeatability and drift on a Project's retained DOF, checked by
the Project contract, recorded per DOF in the run, lowered into the Engine actuator and assessed by the ``dof_*`` targets
(``validation/mechanical.py``).

The retained-DOF scenario: a catalog flat mirror folding a beam, with retained ``Tx``, ``Rx`` and ``Ry``. Every expected number is
the closed form of the metric applied to the stated limits here (``min(travel_max - required_max, required_min -
travel_min)``; ``drift_per_kelvin * dT + drift_per_hour * dt``), never read from the Engine under test.
"""

from __future__ import annotations

import copy
import json

import pytest

from os_optics.cli import main
from os_optics.contracts import ActuatorSpec
from os_optics.local import LocalRunError, json_api, load_project, lower_project, run_optical_project

from local_helpers import ALONG_Y, FLAT_MIRROR, FOLD_X_TO_Y, catalog_document, element, plane, project, source

TX = {"axis": "Tx", "travel": {"min": -2.0, "max": 3.0}, "hard_limits": {"min": -2.5, "max": 3.5}, "repeatability": 0.002,
      "drift": {"per_kelvin": 0.0015, "per_hour": 0.0004}, "unit": "mm"}
RX = {"axis": "Rx", "travel": {"min": -4.0, "max": 4.0}, "repeatability": 0.01, "unit": "deg"}
RY = {"axis": "Ry"}  # retained, no limits stated


def _retained_dof_project(*, dofs=(TX, RX, RY), targets=()) -> dict:
    mirror = element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)
    mirror["retained_dofs"] = [copy.deepcopy(d) for d in dofs]
    # (no environment: the release-parity script checks this document against the hash os-optics 0.12.0 gave it)
    return project([source(), mirror], [plane("OUT", (0.0, 100.0, 0.0), ALONG_Y, radius_mm=20.0)], list(targets), environment=None)


def _target(target_id: str, metric: str, axis: str, *, device: str | None = "M1", **comparison) -> dict:
    parameters = comparison.pop("parameters", {})
    named = {} if device is None else {"device_instance_id": device}
    return {"target_id": target_id, "metric": metric, **named, **comparison,
            "parameters": [{"key": key, "value": value} for key, value in {"axis": axis, **parameters}.items()]}


def _run(document: dict):
    return run_optical_project(document, catalog=catalog_document())


# ----------------------------------------------------------------------------------------------------------- contract
def test_a_project_without_limits_keeps_its_hash_and_engine_input():
    """The retained-DOF control: [Tx, Rx, Ry] retained without limits reads exactly as before (no actuator, same payload)."""
    bare = _retained_dof_project(dofs=({"axis": "Tx"}, {"axis": "Rx"}, {"axis": "Ry"}))
    loaded = load_project(bare)
    assert [d.model_dump(mode="json", exclude_none=True) for d in loaded.components[1].retained_dofs] == [
        {"axis": axis, "frame": "COMPONENT_LOCAL", "control": "MANUAL"} for axis in ("Tx", "Rx", "Ry")]
    assert "travel" not in json.dumps(loaded.payload())
    request = lower_project(bare, catalog=catalog_document()).request.model_dump(mode="json")
    assert all(c["actuators"] == [] for c in request["project"]["components"])


def test_stated_limits_are_accepted_and_lowered_into_the_engine_actuator():
    lowered = lower_project(_retained_dof_project(), catalog=catalog_document())
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M1"]
    tx, rx = mirror.actuators  # (Ry states no travel: no actuator)
    assert tx == ActuatorSpec(actuator_id="Tx", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-2.0, travel_max=3.0,
                              hard_limit_min=-2.5, hard_limit_max=3.5, repeatability=0.002, drift_per_kelvin=0.0015, drift_per_hour=0.0004)
    assert (rx.actuator_id, rx.dof, rx.axis_local, rx.travel_min, rx.travel_max, rx.repeatability) == ("Rx", "rotate", (1.0, 0.0, 0.0), -4.0, 4.0, 0.01)
    assert rx.drift_per_kelvin is None and "drift_per_kelvin" not in rx.model_dump(mode="json")  # (absent fields stay omitted)
    # the limits change the Engine input, never the optics: the stated numbers are inputs of the dof_* targets only
    bare = _run(_retained_dof_project(dofs=({"axis": "Tx"}, {"axis": "Rx"}, {"axis": "Ry"})))
    stated = _run(_retained_dof_project())
    assert stated.result.manifest.input_hash != bare.result.manifest.input_hash
    assert stated.result.observations == bare.result.observations and stated.result.paths == bare.result.paths


@pytest.mark.parametrize(("dof", "message"), [
    ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "unit": "deg"}, "is in mm, not deg"),
    ({"axis": "Rx", "travel": {"min": 0.5, "max": 1.0}}, "travel.min"),  # the nominal pose lies inside the travel
    ({"axis": "Rx", "travel": {"min": -1.0, "max": -0.5}}, "travel.max"),
    ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "hard_limits": {"min": -0.5, "max": 2.0}}, "must contain the travel"),
    ({"axis": "Tx", "drift": {"per_kelvin": 0.001}}, "state travel with them"),
    ({"axis": "Tx", "unit": "mm"}, "state travel with them"),
    ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "repeatability": 0.0}, "repeatability"),
    ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "drift": {}}, "per_kelvin and / or per_hour"),
    ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "drift": {"per_hour": -1.0}}, "per_hour"),
    ({"axis": "Tx", "travel_min_mm": -1.0}, "travel_min_mm"),  # a field of no such name stays refused
])
def test_check_project_refuses_contradictory_limits(dof: dict, message: str):
    reply = json_api.check_project(_retained_dof_project(dofs=(dof,)))
    assert reply["ok"] is False
    (problem,) = reply["error"]["problems"]
    assert problem["code"] == "PROJECT_INVALID" and message in problem["message"], problem


# --------------------------------------------------------------------------------------------- recorded per DOF
def test_the_run_records_the_stated_limits_per_retained_dof(tmp_path):
    produced = run_optical_project(_retained_dof_project(), catalog=catalog_document(), output_dir=tmp_path / "run")
    expected = [{"instance_id": "M1", "control": "MANUAL", **{k: v for k, v in d.items()}} for d in (TX, RX, RY)]
    records = [d.model_dump(mode="json", exclude_none=True) for d in produced.local_validation.retained_dofs]
    assert records == expected
    assert produced.summary()["retained_dofs"] == expected
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["retained_dofs"] == expected
    assert main(["results", "describe", str(tmp_path / "run")]) == 0


def test_results_describe_lists_the_retained_dofs(tmp_path, capsys):
    run_optical_project(_retained_dof_project(), catalog=catalog_document(), output_dir=tmp_path / "run")
    assert main(["results", "describe", str(tmp_path / "run")]) == 0
    described = json.loads(capsys.readouterr().out)
    assert [d["axis"] for d in described["retained_dofs"]] == ["Tx", "Rx", "Ry"]
    assert described["retained_dofs"][0]["travel"] == {"min": -2.0, "max": 3.0}


def test_a_record_whose_limits_differ_from_the_project_does_not_bind():
    from os_optics._vendor.shared_contracts.optical_design import LocalValidationRecordV1, check_local_validation_binding

    produced = _run(_retained_dof_project())
    project_ = load_project(_retained_dof_project())
    assert check_local_validation_binding(produced.local_validation, project_) == ()
    edited = produced.local_validation.payload()
    edited["retained_dofs"][0]["travel"] = {"min": -2.0, "max": 3.25}
    forged = LocalValidationRecordV1.model_validate(edited)
    assert check_local_validation_binding(forged, project_) == ("retained DOF M1/Tx travel differ from the project's",)


# ------------------------------------------------------------------------------------------------------ assessment
def test_travel_drift_and_repeatability_targets_are_met_unmet_or_indeterminate():
    targets = [
        _target("travel_inside", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_min": -0.5, "required_max": 1.0}),
        _target("travel_beyond", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_max": 3.25}),
        _target("tilt_inside", "dof_travel_margin_deg", "Rx", minimum=1.0, parameters={"required_min": -2.0, "required_max": 2.0}),
        _target("drift", "dof_drift_mm", "Tx", maximum=0.02, parameters={"temperature_span_K": 10.0, "duration_h": 8.0}),
        _target("drift_tight", "dof_drift_mm", "Tx", maximum=0.01, parameters={"temperature_span_K": 10.0}),
        _target("repeat", "dof_repeatability_deg", "Rx", maximum=0.005),
        _target("no_travel", "dof_travel_margin_deg", "Ry", minimum=0.0, parameters={"required_max": 0.1}),
        _target("no_drift", "dof_drift_deg", "Rx", maximum=1.0, parameters={"duration_h": 1.0}),
    ]
    result = _run(_retained_dof_project(targets=targets)).result
    by_id = {t.target_id: t for t in result.targets}
    expected = {
        # min(3 - 1, -0.5 - (-2)) = 1.5 mm left beyond the needed adjustment
        "travel_inside": ("MET", 1.5, "mm"),
        # 3 - 3.25: the needed adjustment exceeds the travel by 0.25 mm
        "travel_beyond": ("UNMET", -0.25, "mm"),
        # min(4 - 2, -2 - (-4)) = 2 deg
        "tilt_inside": ("MET", 2.0, "deg"),
        # 0.0015 * 10 + 0.0004 * 8 = 0.0182 mm
        "drift": ("MET", 0.0015 * 10.0 + 0.0004 * 8.0, "mm"),
        "drift_tight": ("UNMET", 0.0015 * 10.0, "mm"),
        "repeat": ("UNMET", 0.01, "deg"),
        "no_travel": ("INDETERMINATE", None, "deg"),
        "no_drift": ("INDETERMINATE", None, "deg"),
    }
    for target_id, (assessment, value, unit) in expected.items():
        target = by_id[target_id]
        assert (target.assessment.value, target.unit) == (assessment, unit), (target_id, target)
        assert target.value == (None if value is None else pytest.approx(value, abs=1e-15)), target_id
    for undecided in ("no_travel", "no_drift"):
        assert by_id[undecided].issue_codes == ("MECHANICAL_LIMIT_NOT_DECLARED",)
        assert "not verified" in by_id[undecided].message
    requirements = result.outcome.requirements
    assert set(requirements.met) == {"travel_inside", "tilt_inside", "drift"}
    assert set(requirements.indeterminate) == {"no_travel", "no_drift"}
    # the optics are not judged by any of this: the plane's quantity is the same as without targets
    assert result.observations[0].quantity_state.value == "KNOWN"


def test_the_requirements_exit_code_reads_the_dof_targets(tmp_path, capsys):
    path = tmp_path / "retained_dof_project.json"
    met = _target("travel", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_min": -1.0, "required_max": 1.0})
    path.write_text(json.dumps(_retained_dof_project(targets=[met])), encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(catalog_document()), encoding="utf-8")
    assert main(["project", "run", str(path), "--catalog", str(catalog)]) == 0
    undecided = _target("tilt", "dof_travel_margin_deg", "Ry", minimum=0.0, parameters={"required_max": 0.5})
    path.write_text(json.dumps(_retained_dof_project(targets=[met, undecided])), encoding="utf-8")
    assert main(["project", "run", str(path), "--catalog", str(catalog)]) == 2  # (missing limits keep the adjustment unverified)
    capsys.readouterr()


@pytest.mark.parametrize(("target", "message"), [
    (_target("t", "dof_travel_margin_mm", "Tz", minimum=0.0, parameters={"required_max": 1.0}), "retained: Tx, Rx, Ry"),
    (_target("t", "dof_travel_margin_mm", "Rx", minimum=0.0, parameters={"required_max": 1.0}), "use the metric dof_travel_margin_deg"),
    (_target("t", "dof_travel_margin_mm", "Tx", device="SRC", minimum=0.0, parameters={"required_max": 1.0}), "retained: none"),
    (_target("t", "dof_travel_margin_mm", "Tx", device=None, minimum=0.0, parameters={"required_max": 1.0}), "device_instance_id"),
    (_target("t", "dof_travel_margin_mm", "Tx", minimum=0.0), "required_min"),
    (_target("t", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_min": 1.0, "required_max": -1.0}), "must not exceed"),
    (_target("t", "dof_drift_mm", "Tx", maximum=0.1), "temperature_span_K"),
    (_target("t", "dof_drift_mm", "Tx", maximum=0.1, parameters={"duration_h": 0.0}), "is positive"),
])
def test_a_dof_target_that_names_no_readable_dof_is_not_assessed(target: dict, message: str):
    lowered = lower_project(_retained_dof_project(targets=[target]), catalog=catalog_document())
    (problem,) = lowered.target_problems
    assert problem.code.value == "TARGET_INVALID" and message in problem.message, problem
    (marked,) = lowered.request.project.targets
    assert marked.not_assessable.code == "TARGET_INVALID" and marked.device_id is None and marked.parameters == {}


def test_a_dof_target_takes_only_its_own_parameters():
    wrong = _target("t", "dof_repeatability_mm", "Tx", maximum=0.1, parameters={"required_max": 1.0})
    lowered = lower_project(_retained_dof_project(targets=[wrong]), catalog=catalog_document())
    assert [p.code.value for p in lowered.target_problems] == ["TARGET_PARAMETER_INVALID"]


def test_the_metrics_are_published_with_their_meaning():
    from os_optics.local import metric_descriptors

    descriptors = {d.name: d.model_dump(mode="json") for d in metric_descriptors()}
    for name, unit in (("dof_travel_margin_mm", "mm"), ("dof_travel_margin_deg", "deg"), ("dof_drift_mm", "mm"),
                       ("dof_drift_deg", "deg"), ("dof_repeatability_mm", "mm"), ("dof_repeatability_deg", "deg")):
        descriptor = descriptors[name]
        assert descriptor["unit"] == unit and descriptor["aggregation"] == "declared_limit" and descriptor["kind"] == "element"
        assert "MECHANICAL_LIMIT_NOT_DECLARED" in descriptor["validity"]
    capabilities = json_api.capabilities()
    assert "retained_dof_limits" in capabilities["project_features"]
    assert capabilities["target_metrics"]["dof_drift_deg"] == "deg"


# ------------------------------------------------------------------------------------------------------- low level
def test_the_engine_actuator_refuses_hard_limits_inside_the_travel():
    with pytest.raises(ValueError, match="contain the travel"):
        ActuatorSpec(actuator_id="Tx", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-1.0, travel_max=1.0,
                     hard_limit_min=-0.5, hard_limit_max=2.0)
    with pytest.raises(ValueError, match="come together"):
        ActuatorSpec(actuator_id="Tx", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-1.0, travel_max=1.0, hard_limit_min=-2.0)
    plain = ActuatorSpec(actuator_id="a", dof="rotate", axis_local=(0.0, 0.0, 1.0), travel_min=-1.0, travel_max=1.0)
    assert set(plain.model_dump(mode="json")) == {"actuator_id", "dof", "axis_local", "pivot_local_mm", "travel_min", "travel_max",
                                                  "max_step", "resolution", "zero"}
