"""The weak-signal measurement chain and nonlinear conversion boundary (``docs/PHYSICS_SCOPE.md`` P9b).

Second-harmonic generation (``NONLINEAR_CRYSTAL``, ``test_second_harmonic.py``) and sample emission (FLUORESCENCE, RAMAN),
detector bandwidth and noise, chopper modulation and lock-in detection (``tests/test_weak_signal_chain.py``) are in the model:
their items are supported, their metrics are evaluated and their kinds are read.

Every request on an item outside the model gets a structured, discoverable answer naming what is missing and the route:
an out-of-scope target metric is accepted and reported ``OUT_OF_MODEL_SCOPE`` (the target ``INDETERMINATE``, never met), and
a model kind the Project format does not have is refused before the Project is read as ``SIMULATION_MODEL_UNSUPPORTED`` with
the boundary in ``details``. Nothing is estimated.
"""

from __future__ import annotations

import json

import pytest

from os_optics.cli import main
from os_optics.local import LocalRunError, example_project, json_api, load_project, lower_project, run_optical_project
from os_optics.validation.scope import (
    BOUNDARY_BY_ITEM,
    IN_SCOPE_SUPPORTED,
    MODEL_SCOPE_BOUNDARY,
    NOT_MODELLED,
    OUT_OF_SCOPE_EXTERNAL,
    OUT_OF_SCOPE_METRICS,
    OUT_OF_SCOPE_OPTICAL_MODEL_KINDS,
    OUT_OF_SCOPE_SOURCE_KINDS,
    out_of_scope_metric,
)

from local_helpers import catalog_document, plane, project, source

#: The metrics of the recorded G15 / G16 probes, as the probes spelled them, that stay outside the model (the G16 SHG metric is
#: supported now: ``test_the_g16_metric_spelling_is_refused_with_the_supported_name``).
PROBED_METRICS = ("spectral_resolution_nm",)
#: The probed metrics a plane that states a detector computes.
COMPUTED_PROBED_METRICS = ("detector_bandwidth_Hz", "lock_in_snr")


def _screen_project(*targets: dict) -> dict:
    return project([source()], [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)], list(targets))


# ------------------------------------------------------------------------------------------------------ the table
def test_the_boundary_table_states_every_item_with_its_route():
    dispositions = {entry["item"]: entry["disposition"] for entry in MODEL_SCOPE_BOUNDARY}
    assert {item for item, d in dispositions.items() if d == IN_SCOPE_SUPPORTED} == {
        "extended_emitter", "bulk_absorption", "camera_response", "acousto_optic_static_deflection", "fluorescence_emission",
        "raman_emission", "chopper_modulation", "lock_in_detection", "detector_bandwidth", "second_harmonic_generation"}
    assert {item for item, d in dispositions.items() if d == NOT_MODELLED} == set()
    assert {item for item, d in dispositions.items() if d == OUT_OF_SCOPE_EXTERNAL} == {
        "feedback_and_modulator_timing", "spectrometer_instrument_function"}
    for entry in MODEL_SCOPE_BOUNDARY:
        assert set(entry) == {"item", "disposition", "missing_model", "route"} and entry["route"]
        assert bool(entry["missing_model"]) is (entry["disposition"] != IN_SCOPE_SUPPORTED), entry
    # a chopper and a lock-in are declared in an observation plane's detection, not as optical models (a nonlinear crystal is
    # a Project model kind: test_a_nonlinear_crystal_is_read_by_the_schema)
    assert OUT_OF_SCOPE_SOURCE_KINDS == {}
    assert {kind: BOUNDARY_BY_ITEM[item]["disposition"] for kind, item in OUT_OF_SCOPE_OPTICAL_MODEL_KINDS.items()} == {
        "CHOPPER": IN_SCOPE_SUPPORTED, "LOCK_IN": IN_SCOPE_SUPPORTED}


def test_the_table_is_discoverable_in_capabilities():
    boundary = json_api.capabilities()["model_scope_boundary"]
    assert boundary == {"items": {entry["item"]: entry["disposition"] for entry in MODEL_SCOPE_BOUNDARY},
                        "refused_source_kinds": [],
                        "refused_optical_model_kinds": ["CHOPPER", "LOCK_IN"]}
    # (the missing model and the route travel with each refusal and each out-of-scope target: tests below)


def test_every_route_names_a_command_or_entry_that_exists():
    """The routes point at commands and interop entries this runtime has."""
    from os_optics.interop.entries import ENTRIES
    from os_optics.native_library.registry import NATIVE_CLASSES

    text = " ".join(entry["route"] for entry in MODEL_SCOPE_BOUNDARY) + " ".join(OUT_OF_SCOPE_METRICS.values())
    assert "nonlinear-ultrafast-communications" in {entry.entry_id for entry in ENTRIES}
    assert "deflector.acousto_optic" in NATIVE_CLASSES
    for command in ("detector", "detection", "sample-emission", "grating-orders", "sampled-light", "nonlinear"):
        assert f"os-optics results {command}" in text or f"'os-optics results {command}'" in text
        with pytest.raises(SystemExit) as shown:
            main(["results", command, "--help"])
        assert shown.value.code == 0


# --------------------------------------------------------------------------------------------- out-of-scope metrics
@pytest.mark.parametrize("metric", PROBED_METRICS)
def test_an_out_of_scope_metric_is_indeterminate_with_its_reason_and_route(metric: str):
    target = {"target_id": "t", "metric": metric, "observation_id": "screen", "minimum": 0.1}
    produced = run_optical_project(_screen_project(target), catalog=catalog_document())
    (result,) = produced.result.targets
    assert (result.status.value, result.assessment.value, result.value) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE", None)
    assert result.issue_codes == ("OUT_OF_MODEL_SCOPE",)
    assert result.message == OUT_OF_SCOPE_METRICS[out_of_scope_metric(metric)]
    assert "'os-optics " in result.message  # (the route)
    assert produced.result.outcome.requirements.indeterminate == ("t",)
    # the light itself is reported as before
    assert produced.result.observations[0].quantity_state.value == "KNOWN"


@pytest.mark.parametrize("metric", COMPUTED_PROBED_METRICS)
def test_a_probed_detection_metric_on_a_plane_without_a_detector_is_not_assessed_with_the_route(metric: str):
    target = {"target_id": "t", "metric": metric, "observation_id": "screen", "minimum": 0.1}
    assert out_of_scope_metric(metric) is None
    produced = run_optical_project(_screen_project(target), catalog=catalog_document())
    (result,) = produced.result.targets
    assert (result.status.value, result.value, result.issue_codes) == ("UNKNOWN", None, ("TARGET_INVALID",))
    assert "states detection" in result.message
    assert produced.result.outcome.requirements.indeterminate == ("t",)
    assert produced.result.observations[0].quantity_state.value == "KNOWN"  # (the light itself is reported)


def test_out_of_scope_metrics_match_without_regard_to_case_and_nothing_else_does():
    assert out_of_scope_metric("SPECTRAL_RESOLUTION_NM") == "spectral_resolution_nm"
    assert out_of_scope_metric("DETECTOR_BANDWIDTH_HZ") is None
    assert out_of_scope_metric("shg_conversion_efficiency") is None and out_of_scope_metric("SHG_conversion_efficiency") is None
    assert out_of_scope_metric("power_W") is None and out_of_scope_metric("shg") is None
    unknown = _screen_project({"target_id": "t", "metric": "shg_efficiency", "observation_id": "screen", "minimum": 0.1})
    assert [p.code.value for p in lower_project(unknown, catalog_document()).target_problems] == ["TARGET_INVALID"]


def test_the_g16_metric_spelling_is_not_assessed_and_names_the_supported_name():
    """The G16 probe spelled the metric SHG_conversion_efficiency: it is not a metric name (they are case-sensitive): the
    target is not assessed, and its message lists the supported shg_conversion_efficiency (``test_second_harmonic.py``
    evaluates it)."""
    unknown = _screen_project({"target_id": "t", "metric": "SHG_conversion_efficiency", "device_instance_id": "SRC", "minimum": 0.1})
    (result,) = run_optical_project(unknown, catalog=catalog_document()).result.targets
    assert result.status.value == "UNKNOWN" and result.issue_codes == ("TARGET_INVALID",)
    assert "shg_conversion_efficiency" in result.message


# ---------------------------------------------------------------------------------------------- refused model kinds
def _with_source_kind(kind: str) -> dict:
    document = _screen_project()
    document["components"][0]["source_model"]["model_kind"] = kind
    return document


def _with_optical_model_kind(kind: str) -> dict:
    document = _screen_project()
    document["optical_models"] = [{"model_id": "crystal", "model": {"model_kind": kind, "length_mm": 10}}]
    document["components"].append({
        "instance_id": "X1", "optical_role": "custom", "custom_part": {"custom_part_key": "x", "description": "Element."},
        "optical_model_id": "crystal", "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}})
    return document


@pytest.mark.parametrize(("build", "kind", "item", "location"), [
    (_with_optical_model_kind, "CHOPPER", "chopper_modulation", "optical model crystal"),
    (_with_optical_model_kind, "LOCK_IN", "lock_in_detection", "optical model crystal"),
])
def test_an_out_of_scope_model_kind_is_refused_with_the_missing_model_and_the_route(build, kind, item, location, tmp_path, capsys):
    document = build(kind)
    with pytest.raises(LocalRunError) as error:
        load_project(document)
    (problem,) = error.value.problems
    boundary = BOUNDARY_BY_ITEM[item]
    assert (problem.code.value, problem.location) == ("SIMULATION_MODEL_UNSUPPORTED", location)
    assert problem.details == {"model_kind": kind, "boundary_item": item, "disposition": boundary["disposition"],
                               "missing_model": boundary["missing_model"], "route": boundary["route"]}
    assert boundary["missing_model"] in problem.message and boundary["route"] in problem.message
    # the same refusal from check_project (json_api) and from the command line (exit 65, the problems on stdout)
    reply = json_api.check_project(document)
    assert reply["ok"] is False and reply["error"]["problems"][0]["details"]["boundary_item"] == item
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path)]) == 65
    printed = json.loads(capsys.readouterr().out)
    assert printed["problems"][0]["code"] == "SIMULATION_MODEL_UNSUPPORTED"
    assert printed["problems"][0]["details"]["route"] == boundary["route"]


def test_any_other_unknown_kind_keeps_the_schemas_own_refusal():
    with pytest.raises(LocalRunError) as error:
        load_project(_with_source_kind("PLASMA"))
    assert [p.code.value for p in error.value.problems] == ["PROJECT_INVALID"]


def test_a_nonlinear_crystal_is_read_by_the_schema():
    """NONLINEAR_CRYSTAL is a Project model kind: an incomplete one is refused by the schema (PROJECT_INVALID, naming its
    missing fields), not as physics outside the model."""
    with pytest.raises(LocalRunError) as error:
        load_project(_with_optical_model_kind("NONLINEAR_CRYSTAL"))
    (problem,) = error.value.problems
    assert problem.code.value == "PROJECT_INVALID" and "medium" in problem.message


# -------------------------------------------------------------------------------------- supported slices unchanged
def test_a_two_line_source_standing_for_a_harmonic_traces_both_lines_and_states_no_conversion():
    """The fundamental-plus-harmonic stand-in: both lines arrive with the powers stated; the run reports no conversion."""
    document = _screen_project()
    model = document["components"][0]["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": 1064.0, "power_w": 0.0009}, {"wavelength_nm": 532.0, "power_w": 0.0001}]
    result = run_optical_project(document, catalog=catalog_document()).result
    arrivals = sorted((a.beam.wavelength_nm, a.beam.power_W) for a in result.observations[0].arrivals)
    assert arrivals == [(532.0, pytest.approx(0.0001, rel=1e-12)), (1064.0, pytest.approx(0.0009, rel=1e-12))]
    body = {key: value for key, value in result.model_dump(mode="json").items() if key != "manifest"}  # (its model-scope list names them)
    assert "conversion" not in json.dumps(body) and "harmonic" not in json.dumps(body)


def test_the_boundary_moves_the_manifests_model_scope_list_only():
    """The new out-of-scope metrics are listed in every manifest's model scope; the result hash covers the result body, not
    the manifest, so a run's result hash does not move with the list."""
    from os_optics.manifest import hash_quantized

    produced = run_optical_project(example_project().payload())
    manifest = produced.result.manifest
    assert "unsupported:spectral_resolution_nm" in manifest.model_scope
    computed = ("detector_bandwidth_Hz", "lock_in_snr", "shg_conversion_efficiency")
    assert not {f"unsupported:{name}" for name in computed} & set(manifest.model_scope)
    body = produced.result.model_copy(update={"manifest": None}).model_dump(mode="json")
    cfg = produced.request.configuration.numerical
    assert manifest.result_hash == hash_quantized(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor)
    assert produced.result.outcome.requirements.met == tuple(t.target_id for t in produced.result.targets)
