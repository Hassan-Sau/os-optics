"""A control sensor's reading where a beam of unknown power reaches it: not known, rather than the reading of the known beams alone."""

from __future__ import annotations

import math

import pytest

from os_optics.config import NumericalConfig
from os_optics.contracts import ControlSpec, SensorSpec
from os_optics.control.readings import evaluate
from os_optics.local.lowering import lower_project

from declared_helpers import declared_component, with_models
from local_helpers import ALONG_X, plane, project, source


def _grating(orders) -> dict:
    return {"model_id": "g", "model": {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "300", "orders": list(orders),
                                       "clear_half_widths_mm": ["10", "10"], "edge_half_widths_mm": ["11", "11"], "status": "SPECIFIED"}}


def _snapshot(*observations, orders=(0,)):
    lasers = [source("A", (-100.0, 0.0, 0.0), wavelength_nm=550.0), source("B", (50.0, 3.0, 0.0), wavelength_nm=550.0)]
    document = with_models(project([*lasers, declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], list(observations)), [_grating(orders)], design_intent="OPTICAL_SYSTEM")
    return lower_project(document, None, check_solids=False).request.project


def _read(snapshot, **sensor):
    spec = ControlSpec(actuator_refs=("A/x",), sensors=(SensorSpec(sensor_id="s", observation_id="SCREEN", target_value=0.0, **sensor),))
    return evaluate(snapshot, spec, {}, NumericalConfig())


@pytest.mark.parametrize("quantity", ["centroid_u_mm", "centroid_v_mm", "power_W"])
def test_a_sensor_with_a_beam_of_unknown_power_on_it_has_no_reading(quantity):
    """One known beam and one of unknown power: a centroid of the known one alone, or its power alone, is not what the sensor reads."""

    reading = _read(_snapshot(plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)), quantity=quantity)
    assert reading.unknown and reading.lost and math.isnan(reading.values[0])


@pytest.mark.parametrize("quantity", ["centroid_u_mm", "power_W"])
def test_a_sensor_of_the_known_source_reads_it(quantity):
    reading = _read(_snapshot(plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)), quantity=quantity, source_ids=("B",))
    assert not reading.unknown and not reading.lost and math.isfinite(reading.values[0])


def test_the_sensor_whose_reading_is_not_known_is_named():
    """Two sensors on the screen: ``all`` sees the beam of unknown power, ``b_only`` reads the known source B alone. The
    reading flags ``all`` only, and the control analysis names it on its finding and its message (and only it)."""

    from os_optics.contracts.components import ActuatorSpec
    from os_optics.control.analysis import run_control_analysis

    snapshot = _snapshot(plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0))
    stage = ActuatorSpec(actuator_id="x", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-1.0, travel_max=1.0)
    snapshot = snapshot.model_copy(update={"components": tuple(
        c.model_copy(update={"actuators": (stage,)}) if c.instance_id == "A" else c for c in snapshot.components)})
    spec = ControlSpec(actuator_refs=("A/x",), sensors=(
        SensorSpec(sensor_id="all", observation_id="SCREEN", target_value=0.0, quantity="power_W"),
        SensorSpec(sensor_id="b_only", observation_id="SCREEN", target_value=0.0, quantity="power_W", source_ids=("B",))))
    reading = evaluate(snapshot, spec, {}, NumericalConfig())
    assert reading.unknown_sensors == (True, False) and reading.unknown and reading.lost
    assert math.isnan(reading.values[0]) and reading.values[1] == pytest.approx(1e-3, rel=1e-9)  # B's 1 mW reaches the screen whole
    result, issues = run_control_analysis(snapshot.model_copy(update={"control_spec": spec}), NumericalConfig(), None)
    (issue,) = issues
    assert issue.code.value == "OUT_OF_MODEL_SCOPE" and issue.location == "control"
    assert issue.numerical_context["sensor_ids"] == ["all"] and "sensor all " in issue.message and "b_only" not in issue.message
    assert result.status.value == "UNKNOWN" and result.message == "sensor readings not known at the nominal state: all"


def test_a_disturbance_whose_readings_are_not_known_is_undecided_and_names_the_sensor(monkeypatch):
    """The analysis logic alone (the readings stubbed): every reading is known at the nominal state; at the disturbed
    state sensor ``b_only``'s is not. The disturbance is not a measured failure: UNKNOWN, naming ``b_only`` only."""

    import numpy as np

    from os_optics.contracts.components import ActuatorSpec
    from os_optics.contracts.project import RecoverabilitySpec
    from os_optics.control import analysis
    from os_optics.control.readings import Reading

    def fake(_project, _spec, values, _cfg, quad_rule=None):
        if values.get("A/x", 0.0) == 0.0:
            return Reading(np.array([1e-3, 1e-3]), False, 1.0, False, (False, False))
        return Reading(np.array([np.nan, np.nan]), True, 0.0, True, (False, True))

    monkeypatch.setattr(analysis, "evaluate", fake)
    snapshot = _snapshot(plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0))
    stage = ActuatorSpec(actuator_id="x", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-1.0, travel_max=1.0)
    snapshot = snapshot.model_copy(update={"components": tuple(
        c.model_copy(update={"actuators": (stage,)}) if c.instance_id == "A" else c for c in snapshot.components)})
    spec = ControlSpec(actuator_refs=("A/x",), compute_sensitivity=False, sensors=(
        SensorSpec(sensor_id="all", observation_id="SCREEN", target_value=1e-3, quantity="power_W"),
        SensorSpec(sensor_id="b_only", observation_id="SCREEN", target_value=1e-3, quantity="power_W", source_ids=("B",))),
        recoverability=RecoverabilitySpec(disturbances=({"A/x": 0.5},), convergence_tolerance=1e-9))
    result, issues = analysis.run_control_analysis(snapshot.model_copy(update={"control_spec": spec}), NumericalConfig(), None)
    (issue,) = issues
    assert (issue.code.value, issue.status.value) == ("OUT_OF_MODEL_SCOPE", "UNKNOWN")
    assert issue.numerical_context["sensor_ids"] == ["b_only"] and "readings of b_only were not known" in issue.message
    assert result.status.value == "UNKNOWN" and result.recoverability_cases[0].outcome != "RECOVERED"


def test_a_beam_of_unknown_power_outside_the_acceptance_cone_does_not_touch_the_reading():
    """Order 1 of 300 lines/mm at 550 nm leaves at 9.5 deg, outside a 1 deg cone: the sensor reads nothing of it, whatever its power, and the known beam reads as it is."""

    screen = {**plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0), "acceptance_half_angle_deg": 1.0}
    reading = _read(_snapshot(screen, orders=(1,)), quantity="power_W")
    assert not reading.unknown and not reading.lost and reading.values[0] == pytest.approx(1e-3, rel=1e-9)
