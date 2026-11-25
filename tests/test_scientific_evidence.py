"""Scientific outcomes survive supported result, artifact and record consumers."""
import copy

import pytest
from pydantic import ValidationError

from os_optics._vendor.shared_contracts.optical_design import LocalValidationRecordV1
from os_optics.local import run_optical_project
from os_optics.local.artifacts import cross_artifact_problems
from os_optics.local.example import example_project
from os_optics.results import load_run


@pytest.fixture(scope="module")
def local_run():
    return run_optical_project(example_project())


def test_result_consumers_keep_execution_quantities_and_requirements(local_run):
    result = local_run.result
    record = local_run.local_validation
    assert record.simulation.outcome.model_dump(mode="json") == result.outcome.model_dump(mode="json")
    assert record.simulation.observation_quantities
    assert all(t.assessment is not None for t in record.simulation.target_outcomes)
    reader = load_run(local_run)
    assert reader.describe()["outcome"]["execution"]["outcome"] == "COMPLETED"
    assert all("assessment" in t for t in reader.targets()["targets"])
    assert all("quantity_state" in o for o in reader.monitors())


@pytest.mark.parametrize("edit", [
    lambda r: r["simulation"]["outcome"]["execution"].update(complete=False),
    lambda r: r["simulation"]["outcome"]["requirements"].update(declared=False),
    lambda r: r["simulation"]["target_outcomes"][0].update(assessment="UNMET"),
    lambda r: r["simulation"]["observation_quantities"][0].update(quantity_state="KNOWN_ZERO", total_power_W=1),
])
def test_record_rejects_contradictory_scientific_claims(local_run, edit):
    body = copy.deepcopy(local_run.local_validation.payload())
    edit(body)
    with pytest.raises(ValidationError):
        LocalValidationRecordV1.model_validate(body)


def test_legacy_record_keeps_absent_fields_and_original_hash(local_run):
    from os_optics._vendor.shared_contracts.common.hashes import osc1_digest
    body = local_run.local_validation.payload()
    body["simulation"].pop("outcome")
    body["simulation"].pop("observation_quantities")
    for target in body["simulation"]["target_outcomes"]:
        target.pop("assessment")
    record = LocalValidationRecordV1.model_validate(body)
    assert record.payload() == body
    assert record.record_hash() == osc1_digest(body)
    assert record.simulation.outcome is None


def test_hash_string_cannot_hide_changed_compact_values(local_run):
    body = local_run.local_validation.payload()
    body["simulation"]["target_outcomes"][0]["value"] = 1000.0
    record = LocalValidationRecordV1.model_validate(body)
    assert any("target" in p for p in cross_artifact_problems(record, local_run.optical_view, local_run.result))


def test_optional_targets_do_not_change_missing_light_evidence():
    from os_optics import builders as B
    from os_optics.contracts.project import ProjectSnapshot
    from helpers import run
    from os_optics.config import NumericalConfig
    CFG = NumericalConfig()
    source = B.laser("S", wavelength_nm=780, power_W=1e-3, waist_radius_mm=0.5,
                     position=(-60, 0, 0), direction=(1, 0, 0))
    plane = B.observation_plane("far", position=(150, 0, 0), normal=(1, 0, 0), radius_mm=12.7)
    project = ProjectSnapshot(components=(source,), observations=(plane,), targets=())
    from os_optics.contracts import Target
    targeted = project.model_copy(update={"targets": (Target(target_id="power", observation_id="far", metric="power_W",
                                                              maximum=1, unit="W"),)})
    cfg = CFG.model_copy(update={"max_trace_steps": 1})
    a, b = run(project, numerical=cfg), run(targeted, numerical=cfg)
    for x, y in zip(a.observations, b.observations):
        for field in ("total_power_W", "power_complete", "untraced_power_W", "quantity_state"):
            assert getattr(x, field) == getattr(y, field)
    assert a.outcome.execution == b.outcome.execution
    assert not a.outcome.requirements.declared and b.outcome.requirements.declared


@pytest.mark.parametrize("with_target", [False, True])
@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_local_model_applicability_reaches_compact_evidence_and_consumers(tmp_path, with_target, lossless):
    from local_helpers import ALONG_X, PCX_F100, catalog_document, element, plane, project, run, source, with_lossless_bulk
    from os_optics.local.json_api import run_summary
    from os_optics.local.artifacts import verify_run_artifacts
    targets = [{"target_id": "power", "metric": "power_W", "observation_id": "screen", "minimum": 0, "unit": "W"}] if with_target else []
    document = project([source(wavelength_nm=1200), source("unrelated", position=(-200, 100, 0)),
                        element("L1", "lens", PCX_F100, quaternion=ALONG_X)],
                       [plane("upstream", (-100, 0, 0)), plane("screen", (100, 0, 0), radius_mm=40),
                        plane("independent", (100, 100, 0))], targets)
    catalog = catalog_document()
    local = run(document, catalog=with_lossless_bulk(catalog) if lossless else catalog, output_dir=tmp_path)
    raw = local.result.model_dump(mode="json")
    simulation = local.local_validation.simulation
    quantities = {q.observation_id: q for q in simulation.observation_quantities}
    if lossless:
        upstream, independent = quantities["upstream"], quantities["independent"]
        assert upstream.quantity_state == "BOUNDED" and upstream.issue_codes == ()
        assert upstream.power_upper_bound_W > upstream.total_power_W
        assert independent.quantity_state == "KNOWN"
        assert independent.power_upper_bound_W > independent.total_power_W
    else:
        for name in ("upstream", "independent"):
            quantity = quantities[name]
            assert quantity.quantity_state == "BOUNDED" and quantity.power_upper_bound_W is None
            assert quantity.total_power_W == 0.001
            observation = next(o for o in local.result.observations if o.observation_id == name)
            entry = observation.value_basis["total_power_W"]
            assert entry.kind.value == "LOWER_BOUND"
            (bulk,) = [part for part in entry.omitted if part.code == "BULK_LOSS_NOT_ESTABLISHED"]
            assert bulk.upper is None and bulk.at == ("path:SRC#0",)
    assert quantities["screen"].quantity_state == "OUT_OF_MODEL_SCOPE"
    assert set(quantities["screen"].issue_codes) == {"MODEL_WAVELENGTH_OUTSIDE_BAND", "COATING_WAVELENGTH_OUTSIDE_BANDS"}
    assert all(i.affected_observation_ids == ("screen",) for i in simulation.model_scope_issues)
    assert simulation.outcome.execution.model_dump(mode="json") == raw["outcome"]["execution"]
    assert simulation.outcome.requirements.met == ()
    assert simulation.outcome.requirements.indeterminate == (("power",) if with_target else ())
    assert simulation.outcome.requirements.declared == with_target
    if with_target:
        assert simulation.target_outcomes[0].status.value == "PASS"
        assert simulation.target_outcomes[0].assessment == "INDETERMINATE"
        assert raw["targets"][0]["assessment"] == "MET"
    screen = next(o for o in raw["observations"] if o["observation_id"] == "screen")
    assert screen["quantity_state"] == ("BOUNDED" if lossless else "UNKNOWN")
    assert not cross_artifact_problems(local.local_validation, local.optical_view, local.result)
    assert not verify_run_artifacts(tmp_path)
    for reader in (load_run(local), load_run(tmp_path)):
        assert next(o for o in reader.monitors() if o["observation_id"] == "screen")["quantity_state"] == "OUT_OF_MODEL_SCOPE"
        assert reader.monitor("screen")["quantity_state"] == "OUT_OF_MODEL_SCOPE"
        assert reader.describe()["outcome"]["requirements"]["met"] == []
        if with_target:
            assert reader.targets()["targets"][0]["assessment"] == "INDETERMINATE"
            assert reader.monitor("screen")["targets"][0]["assessment"] == "INDETERMINATE"
    summary = run_summary(local)
    assert next(o for o in summary["observations"] if o["observation_id"] == "screen")["quantity_state"] == "OUT_OF_MODEL_SCOPE"
    assert summary["outcome"]["requirements"]["met"] == []
    assert local.result.model_dump(mode="json") == raw


def test_applicability_override_cannot_be_unbound_or_disagree():
    from test_local_validity import _lens_run
    local = _lens_run(1200)
    body = local.local_validation.payload()
    body["simulation"]["model_scope_issues"] = []
    with pytest.raises(ValidationError, match="applicability"):
        LocalValidationRecordV1.model_validate(body)
    body = local.local_validation.payload()
    body["validation_status"] = "PASS"
    with pytest.raises(ValidationError, match="applicability"):
        LocalValidationRecordV1.model_validate(body)
    body = local.local_validation.payload()
    body["simulation"]["observation_quantities"][0]["quantity_state"] = "UNKNOWN"
    with pytest.raises(ValidationError, match="applicability"):
        LocalValidationRecordV1.model_validate(body)


def test_device_target_on_unsupported_element_is_explicitly_indeterminate():
    from local_helpers import PCX_F100, element, plane, project, run, source
    document = project([source(wavelength_nm=1200), element("L1", "lens", PCX_F100)],
                       [plane("screen", (100, 0, 0), radius_mm=40)],
                       [{"target_id": "rating", "metric": "power_rating_utilization", "device_instance_id": "L1", "maximum": 1}])
    local = run(document)
    target = local.local_validation.simulation.target_outcomes[0]
    assert target.assessment == "INDETERMINATE"
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in target.applicability_issue_codes
    assert all(i.affected_target_ids == ("rating",) for i in local.local_validation.simulation.model_scope_issues)
    body = local.local_validation.payload()
    body["simulation"]["target_outcomes"][0].pop("assessment")
    with pytest.raises(ValidationError, match="applicability"):
        LocalValidationRecordV1.model_validate(body)


def test_failed_bare_reader_retains_declared_but_unevaluated_requirements():
    from fixture_scenes import invalid_zero_normal
    from os_optics import validate
    from os_optics.contracts import Target
    from os_optics.results import results_from_engine
    request = invalid_zero_normal()
    request = request.model_copy(update={"project": request.project.model_copy(update={
        "targets": (Target(target_id="power", device_id="L", metric="power_rating_utilization", maximum=1),)})})
    result = validate(request)
    assert result.outcome.execution.outcome.value == "FAILED"
    assert result.outcome.requirements.declared
    assert result.targets == ()
    reader = results_from_engine(result)
    assert reader.describe()["outcome"]["requirements"]["declared"] is True
    assert reader.targets()["outcome"]["requirements"]["declared"] is True


def test_geometry_scope_does_not_reach_an_upstream_measurement():
    from local_helpers import ALONG_Y, WINDOW, element, plane, project, run, source, product_document
    model = product_document(WINDOW)["simulation_models"][0]
    thickness = float(model["surfaces"][-1]["vertex_z_mm"])
    document = project([source(waist_mm=0.2), element("W", "window", WINDOW, (0, -thickness / 2, 0), ALONG_Y)],
                       [plane("up", (-100, 0, 0)), plane("D", (100, 0, 0))])
    local = run(document)
    issue = next(i for i in local.model_scope_issues if i.code == "RAY_CROSSES_ELEMENT_BODY")
    assert issue.affected_observation_ids == ("D",)
    quantities = {q.observation_id: q for q in local.local_validation.simulation.observation_quantities}
    assert quantities["up"].quantity_state == "KNOWN"
    assert quantities["D"].quantity_state == "OUT_OF_MODEL_SCOPE"


def test_upstream_model_scope_reaches_a_downstream_device_target():
    from test_power_exposure_ratings import _document, _rating
    document = _document([_rating("CW", 1e9, "W/cm2")], targets=[{
        "target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1}])
    document["optical_models"][0]["model"]["valid_band_nm"] = {"min_nm": "400", "max_nm": "800", "status": "DECLARED"}
    local = run_optical_project(document)
    # This declared test window assumes zero bulk loss: the raw Engine's rating target
    # is met, but the upstream model's wavelength scope makes local acceptance undecided.
    assert local.result.targets[0].assessment.value == "MET"
    assert local.local_validation.simulation.target_outcomes[0].assessment == "INDETERMINATE"
    assert local.local_validation.simulation.outcome.requirements.met == ()
    assert any(i.instance_id == "L" and i.affected_target_ids == ("util",) for i in local.model_scope_issues)


def test_exposure_applicability_does_not_require_optional_targets():
    from test_power_exposure_ratings import _document, _rating
    document = _document([_rating("CW", 1e9, "W/cm2")], targets=[])
    document["optical_models"][0]["model"]["valid_band_nm"] = {"min_nm": "400", "max_nm": "800", "status": "DECLARED"}
    local = run_optical_project(document)
    assert not local.local_validation.simulation.outcome.requirements.declared
    assert "W" in local.model_scope_issues[0].affected_element_ids
    before = local.result.model_dump(mode="json")
    exposure = load_run(local).exposures("W")
    assert exposure["local_model_applicability"]["state"] == "OUT_OF_MODEL_SCOPE"
    assert all(s["local_model_applicability"]["state"] == "OUT_OF_MODEL_SCOPE" for s in exposure["surfaces"])
    assert all(r["local_model_applicability"]["state"] == "OUT_OF_MODEL_SCOPE" for s in exposure["surfaces"] for r in s["ratings"])
    assert local.result.model_dump(mode="json") == before


def test_edge_scope_retains_samples_along_the_affected_medium_run():
    from test_local_beam_scope import _tilted_window_run
    from local_helpers import run
    document = _tilted_window_run(10.6).project.payload()
    document["components"][0]["source_model"]["waist_radius_mm"] = "0.01"
    local = run(document)
    edge = next(i for i in local.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE")
    assert edge.affected_observation_ids == ("D",)
    assert local.local_validation.simulation.observation_quantities[0].quantity_state == "OUT_OF_MODEL_SCOPE"


def test_scope_element_references_bind_to_project():
    from test_local_validity import _lens_run
    from os_optics.local import check_local_validation_binding
    local = _lens_run(1200)
    body = local.local_validation.payload()
    body["simulation"]["model_scope_issues"][0]["affected_element_ids"].append("missing")
    record = LocalValidationRecordV1.model_validate(body)
    assert any("component missing absent" in p for p in check_local_validation_binding(record, local.project))


@pytest.mark.parametrize("scope", [None, [{"instance_id": "W", "code": "MODEL_WAVELENGTH_OUTSIDE_BAND", "message": "old", "affected_element_ids": None}]])
def test_exposure_reader_does_not_invent_absent_local_applicability(scope):
    from test_power_exposure_ratings import _document, _rating
    local = run_optical_project(_document([_rating("CW", 1e9, "W/cm2")], targets=[]))
    reader = load_run(local)
    reader.manifest["status"]["model_scope_issues"] = scope
    assert reader.exposures("W")["local_model_applicability"]["state"] == "UNRECORDED"
    reader.manifest["status"].pop("model_scope_issues")
    assert reader.exposures("W")["local_model_applicability"]["state"] == "UNRECORDED"
