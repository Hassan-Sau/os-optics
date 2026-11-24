"""Local pure-optical workflow (``os_optics.local``): public catalog + ProjectDocumentV2 -> Engine -> LocalValidationRecordV1.

The reference design is built in-test against the committed test catalog (``local_helpers``; release ``2026-01-01.1``):
a collimated 780 nm Gaussian source 200 mm before the plano-convex N-BK7 prescription ``OS-OPT-TEST-LENS-PCX-F100``
(curved side first) and a screen at the Gaussian waist predicted by independent paraxial q propagation.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from decimal import Context, getcontext, localcontext
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

import os_optics
from os_optics import Status, _version
from os_optics.manifest import hash_exact
from os_optics.local import (
    LocalRunError,
    LocalRuntimeKindV1,
    LocalValidationRecordV1,
    LocalValidationStatusV1,
    ProjectDocumentV2,
    check_local_validation_binding,
    compile_element,
    example_project,
    load_project,
    load_public_catalog,
    local_validation_record,
    lower_project,
    run_optical_project,
    target_metrics,
)

from helpers import issue_codes, one_arrival
from identity_mask import IDENTITY_MASK, mask_request_identity, masked_input_hash
from local_helpers import (
    FIXTURE_CATALOG,
    PCX_F100,
    POWER_W,
    TEST_CATALOG_VERSION,
    catalog_document,
    element,
    lens_matrix,
    plane,
    project,
    q_through,
    rayleigh_range,
    source,
    thickness,
    with_lossless_bulk,
)

SOURCE_X, W0_MM, WAVELENGTH_NM = -200.0, 1.0, 780.0
# Lowering drift guard: the exact Engine input the reference design lowers to, with the schema version the Engine request
# stamps masked (identity_mask.py). Any change to the compile rules, the test catalog's lens model or the Engine request
# contract moves it; update it only with a reviewed reason. A schema or version bump does not move it: with 1.9.0 stamped,
# unmasked, this input is 544914b3...053e, which moved at every schema bump (and by nothing else: 657e75be...4229 at 1.6.0,
# 7b0f005d...ac38 at 1.7.0). Last change: the request's numerical configuration states the caller settings of config
# 1.7.0 (`negligible_power_fraction`, `max_scatter_origins`, `max_mesh_patches`, the six `rating_*` windows and the eight
# `design_point_*` tolerances, every one at today's value; without them and with config_version 1.6.0 the input is
# 586c8b0f...cabcc). Before that: the request's numerical configuration states the second-harmonic tolerances
# `nonlinear_depletion_rel_tol`, `nonlinear_focusing_rel_tol`, `nonlinear_quadrature_rel_tol`, `nonlinear_harmonic_gaussian_rel_tol`,
# `nonlinear_beam_quality_tol` and `nonlinear_linewidth_rel_tol` (config 1.6.0; without them the input is f0b83f87...1f23).
# Before that: `max_return_analysis_work` and the crystal tolerances `birefringent_interface_warn_aoi_deg` and
# `birefringent_gaussian_rel_tol` (f19d0966...aeb4 before); before that `laboratory_reach_mm`, config 1.5.0, 1bcd7fb8...f1c6d.
# tests/test_declared_models.py checks that every existing Project otherwise lowers to the same bytes.
# Bulk-loss provenance leaves catalog N-BK7's absorption unstated. The pin combines that change with the config 1.7.0
# settings above and excludes the reference's explicitly lossless unit-index vacuum air.
REFERENCE_ENGINE_INPUT_MASKED_HASH = "b1a145d04bf512c36dc4c427e09267b8f89c58315585aa8a0366581a380f8264"


def _screen_x(wavelength_nm: float = WAVELENGTH_NM) -> float:
    q_out = q_through(lens_matrix(PCX_F100, wavelength_nm), complex(-SOURCE_X, rayleigh_range(W0_MM, wavelength_nm)))
    return round(thickness(PCX_F100) - q_out.real, 3)


SCREEN_X = _screen_x()


def _reference(*, wavelength_nm: float = WAVELENGTH_NM, screen_x: float = SCREEN_X, catalog_version: str | None = TEST_CATALOG_VERSION) -> dict:
    document = project(
        [source(position=(SOURCE_X, 0.0, 0.0), wavelength_nm=wavelength_nm),
         element("L1", "lens", PCX_F100, catalog_version=catalog_version, retained=(("Tx", "MANUAL"), ("Ty", "MANUAL"), ("Tz", "NUMERICAL")))],
        [plane("focal_plane", (screen_x, 0.0, 0.0))],
        [{"target_id": "focus_position", "metric": "waist_distance_x_mm", "observation_id": "focal_plane", "target_value": 0.0,
          "tolerance_abs": 0.1, "unit": "mm"},
         {"target_id": "spot_radius", "metric": "beam_radius_mm", "observation_id": "focal_plane", "maximum": 0.05, "unit": "mm"}],
    )
    document["materials"] = []
    return document


def _run(document, **kwargs):
    kwargs.setdefault("catalog", catalog_document())
    return run_optical_project(document, **kwargs)


def _lower(document, catalog=None, **kwargs):
    return lower_project(document, catalog if catalog is not None else catalog_document(), **kwargs)


def _target(run, target_id: str):
    return next(t for t in run.result.targets if t.target_id == target_id)


def _outcome(record: LocalValidationRecordV1, target_id: str):
    return next(o for o in record.simulation.target_outcomes if o.target_id == target_id)


@pytest.fixture(scope="module")
def reference_run():
    return _run(ProjectDocumentV2.model_validate(_reference()))


# ---- A. reference loop --------------------------------------------------------------------------------------------------------------
def test_the_summary_leads_with_execution_and_requirements_and_names_the_aggregate(reference_run):
    """What decides comes first: did the calculation complete, how did the declared targets come out. The aggregate
    diagnostic severity follows under its own name, with the same value as the unchanged ``status`` key."""
    summary = reference_run.summary()
    assert list(summary)[:5] == ["execution", "requirements", "diagnostic_summary", "status", "engine_status"]
    assert summary["execution"] == {"outcome": "COMPLETED", "termination": "COMPLETED", "complete": True}
    assert summary["requirements"] == summary["outcome"]["requirements"]
    assert summary["requirements"]["declared"] and summary["requirements"]["met"] == [t.target_id for t in reference_run.result.targets]
    assert summary["diagnostic_summary"] == summary["status"] == reference_run.status.value


def test_the_reference_design_preserves_its_focus_but_does_not_assume_lossless_catalog_glass(reference_run):
    assert reference_run.catalog_version == TEST_CATALOG_VERSION
    assert reference_run.status is LocalValidationStatusV1.UNKNOWN and reference_run.engine_status is Status.UNKNOWN
    assert [t.status.value for t in reference_run.result.targets] == ["PASS", "PASS"]
    assert abs(_target(reference_run, "focus_position").value) < 0.001
    assert reference_run.binding_problems() == ()


#: The environment's air of the reference design (its Project operates in vacuum, ``local_helpers.UNIT_INDEX_ENVIRONMENT``):
#: every native request states it as its first material; it is the Engine's unit-index built-in ``air`` itself.
#: Its explicit zero absorption is physically justified; unstated loss on a catalog solid is not.
VACUUM_AIR = {"absorption_coefficient_per_mm": 0.0, "kind": "constant", "material_id": "air", "refractive_index": 1.0,
              "valid_wavelength_nm": [100.0, 100000.0]}


def _without_environment_air(request) -> dict:
    """The masked request with the stated air (``VACUUM_AIR``) taken out: the pin predates it, and nothing else changed."""
    document = mask_request_identity(request)
    assert document["project"]["materials"][0] == VACUUM_AIR
    document["project"]["materials"] = document["project"]["materials"][1:]
    return document


def test_engine_input_is_pinned_and_its_hash_binds_it(reference_run):
    assert hash_exact(_without_environment_air(reference_run.request)) == REFERENCE_ENGINE_INPUT_MASKED_HASH, \
        "the reference design lowers to a different Engine input"
    lowered = _lower(reference_run.project)
    assert lowered.request == reference_run.request
    # the binding: recomputed from its parts, the same twice, and it moves with the schema stamp the pin masks
    assert lowered.engine_input_hash == reference_run.engine_input_hash == hash_exact(lowered.request.model_dump(mode="json"))
    assert lowered.request.project.project_schema_version == _version.SCHEMA_VERSION
    restamped = lowered.request.model_dump(mode="json")
    restamped["project"]["project_schema_version"] = IDENTITY_MASK
    assert hash_exact(restamped) == masked_input_hash(lowered.request) != lowered.engine_input_hash
    assert hash_exact(_without_environment_air(lowered.request)) == REFERENCE_ENGINE_INPUT_MASKED_HASH


def test_record_hashes_are_the_engine_manifest_hashes(reference_run):
    record, manifest = reference_run.local_validation, reference_run.result.manifest
    assert record.trust_basis == "LOCAL_RUNTIME"
    expected_input = hash_exact(reference_run.request.model_dump(mode="json"))
    assert record.input_hash.digest == reference_run.engine_input_hash == manifest.input_hash == expected_input
    assert (record.input_hash.profile.value, record.input_hash.subject) == ("eng-exact-py1", "engine_input")
    assert record.result_hash.digest == manifest.result_hash
    assert (record.result_hash.profile.value, record.result_hash.subject) == ("eng-quant-py1", "engine_result")
    assert dict(record.result_hash.params) == {"sig_digits": "9", "abs_floor": "0.000000000001"}
    assert record.project.project_hash.digest == reference_run.project.content_hash()
    assert record.project.project_hash.profile.value == "osc1"


def test_record_runtime_identity_and_reproducibility(reference_run):
    record, manifest = reference_run.local_validation, reference_run.result.manifest
    runtime = record.runtime
    assert runtime.runtime_kind is LocalRuntimeKindV1.PYTHON_LOCAL
    assert runtime.engine_package == _version.DISTRIBUTION_NAME == "os-optics"
    assert runtime.engine_version == _version.ENGINE_VERSION == manifest.engine_version
    assert runtime.engine_build_id == manifest.engine_build_id
    assert runtime.engine_build_id.startswith(_version.ENGINE_VERSION + "+")
    assert runtime.engine_schema_version == _version.SCHEMA_VERSION == manifest.schema_version
    assert runtime.numerical_config_version == _version.NUMERICAL_CONFIG_VERSION == manifest.numerical_config_version
    repro = record.reproducibility
    assert repro.deterministic and repro.random_seed is None and repro.prng_id is None
    assert {e.key: e.version for e in repro.model_versions} == manifest.model_versions == _version.MODEL_VERSIONS
    assert record.simulation.path_count == 1 and record.simulation.observation_count == 1
    assert record.validation_status is LocalValidationStatusV1.UNKNOWN


def test_run_status_is_the_validation_status(reference_run):
    record = reference_run.local_validation
    assert reference_run.status is record.validation_status is LocalValidationStatusV1.UNKNOWN
    assert record.simulation.status.value == reference_run.engine_status.value
    summary = reference_run.summary()
    assert (summary["status"], summary["engine_status"], summary["model_scope_issues"]) == ("UNKNOWN", "UNKNOWN", [])
    assert summary["local_validation"] == record.payload()


def test_record_simulated_parts_and_retained_dofs(reference_run):
    record = reference_run.local_validation
    assert [(p.instance_id, p.optical_sku, p.catalog_version, p.simulation_model.model_kind, p.simulation_model.model_version)
            for p in record.simulated_parts] == [("L1", PCX_F100, TEST_CATALOG_VERSION, "SURFACE_SEQUENCE", "2")]
    assert all(set(p.payload() if hasattr(p, "payload") else p.model_dump()) == {"instance_id", "optical_sku", "catalog_version", "simulation_model"} for p in record.simulated_parts)
    assert [(d.instance_id, d.axis.value, d.control.value) for d in record.retained_dofs] == [
        ("L1", "Tx", "MANUAL"), ("L1", "Ty", "MANUAL"), ("L1", "Tz", "NUMERICAL"),
    ]
    assert [o.value for o in record.simulation.target_outcomes] == [t.value for t in reference_run.result.targets]


def test_runtime_kind_override_changes_identity_not_hashes(reference_run):
    browser = _run(reference_run.project, runtime_kind="BROWSER_PYTHON")
    assert browser.local_validation.runtime.runtime_kind is LocalRuntimeKindV1.BROWSER_PYTHON
    assert browser.local_validation.input_hash == reference_run.local_validation.input_hash
    assert browser.local_validation.result_hash == reference_run.local_validation.result_hash


def test_record_round_trips_through_its_payload(reference_run):
    record = reference_run.local_validation
    again = LocalValidationRecordV1.model_validate(json.loads(json.dumps(record.payload())))
    assert again == record and again.record_hash() == record.record_hash()


# ---- B. determinism and inputs -----------------------------------------------------------------------------------------------------
def test_repeated_runs_are_identical(reference_run):
    again = _run(ProjectDocumentV2.model_validate(_reference()))
    assert again.local_validation.payload() == reference_run.local_validation.payload()
    assert again.result.manifest.result_hash == reference_run.result.manifest.result_hash
    assert again.summary() == reference_run.summary()


def test_component_order_is_engine_input_but_not_physics(reference_run):
    document = _reference()
    document["components"].reverse()
    swapped = _run(document)
    assert [c.instance_id for c in swapped.request.project.components] == ["L1", "SRC"]
    assert swapped.engine_input_hash != reference_run.engine_input_hash
    assert [t.value for t in swapped.result.targets] == pytest.approx([t.value for t in reference_run.result.targets], rel=1e-12)


def test_every_project_input_form_gives_the_same_record(reference_run, tmp_path):
    document = _reference()
    text = json.dumps(document)
    path = tmp_path / "project.json"
    path.write_text(text, encoding="utf-8")
    expected = reference_run.local_validation.payload()
    forms = {
        "model": ProjectDocumentV2.model_validate(document), "dict": document, "json_text": text, "indented_json_text": "\n  " + text,
        "bytes": text.encode("utf-8"), "path": path, "str_path": str(path), "any_pydantic_model": _LooseModel.model_validate(document),
    }
    for name, form in forms.items():
        assert _run(form).local_validation.payload() == expected, name
        assert load_project(form).content_hash() == reference_run.project.content_hash(), name


class _LooseModel(BaseModel):
    """Stands in for another pydantic class carrying the same document (no ``payload()``)."""

    model_config = ConfigDict(extra="allow")


def test_str_that_is_not_json_is_a_path(tmp_path):
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(str(tmp_path / "missing.json"))
    assert [c.value for c in exc.value.codes] == ["PROJECT_INVALID"]
    assert "cannot read the project file" in str(exc.value)
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog(str(tmp_path / "missing-catalog.json"))
    assert [c.value for c in exc.value.codes] == ["CATALOG_INVALID"]


def test_catalog_input_forms_are_equivalent(reference_run, tmp_path):
    text = FIXTURE_CATALOG.read_text(encoding="utf-8")
    path = tmp_path / "catalog.json"
    path.write_text(text, encoding="utf-8")
    expected = reference_run.local_validation.payload()
    parsed = load_public_catalog(catalog_document())
    for catalog in (parsed, catalog_document(), text, text.encode("utf-8"), path, str(path)):
        assert run_optical_project(reference_run.project, catalog=catalog).local_validation.payload() == expected
        assert lower_project(reference_run.project, catalog).engine_input_hash == reference_run.engine_input_hash


def test_catalog_parsing_is_memoized_by_content():
    assert load_public_catalog() is load_public_catalog()
    assert load_public_catalog(FIXTURE_CATALOG.read_bytes()) is load_public_catalog()  # the suite's active stored release
    assert load_public_catalog(catalog_document()) is load_public_catalog(catalog_document())
    reformatted = json.dumps(catalog_document(), indent=4)
    assert load_public_catalog(reformatted) == load_public_catalog(catalog_document())


def test_caller_decimal_context_does_not_change_the_engine_input(reference_run):
    with localcontext(Context(prec=3)):
        narrow = _lower(_reference())
        narrow_run = _run(_reference())
        assert getcontext().prec == 3
    assert narrow.engine_input_hash == reference_run.engine_input_hash
    assert narrow_run.local_validation.payload() == reference_run.local_validation.payload()


# ---- C. physics from public data ----------------------------------------------------------------------------------------------------
def test_the_lens_is_lowered_as_its_compiled_prescription(reference_run):
    catalog = load_public_catalog(catalog_document())
    compiled = compile_element("L1", catalog.get(PCX_F100), catalog)
    lens = next(c for c in reference_run.request.project.components if c.instance_id == "L1")
    assert lens.surfaces == compiled.surfaces and len(lens.surfaces) == 2
    assert lens.role == "lens" and lens.envelope is None and lens.actuators == ()
    # (after the environment's air, which the request states first)
    assert reference_run.request.project.materials[0].model_dump(mode="json") == VACUUM_AIR
    assert reference_run.request.project.materials[1:] == compiled.materials
    source_spec = next(c for c in reference_run.request.project.components if c.instance_id == "SRC").source
    assert (source_spec.source_id, source_spec.wavelength_nm, source_spec.power_W) == ("SRC", WAVELENGTH_NM, POWER_W)
    assert source_spec.polarization.ex == (1.0, 0.0) and source_spec.polarization.ey == (0.0, 0.0)
    config = reference_run.request.configuration
    assert (config.run_physics, config.run_layout, config.run_control, config.random_seed) == (True, False, False, None)


def test_the_focus_and_spot_are_the_independent_gaussian_prediction(reference_run):
    q_out = q_through(lens_matrix(PCX_F100, WAVELENGTH_NM), complex(-SOURCE_X, rayleigh_range(W0_MM, WAVELENGTH_NM)))
    gaussian = one_arrival(reference_run.result, "focal_plane").beam.gaussian
    assert gaussian.waist_distance_x_mm == pytest.approx(SCREEN_X - (thickness(PCX_F100) - q_out.real), abs=1e-6)
    assert _target(reference_run, "spot_radius").value == pytest.approx(gaussian.beam_radius_x_mm, rel=1e-12)


def test_the_coated_lens_bounds_power_by_its_two_ar_surfaces_without_bulk_loss_data(reference_run):
    arrival = one_arrival(reference_run.result, "focal_plane")
    assert arrival.beam.power_W is None  # catalog N-BK7 gives an index but no measured or assumed bulk transmission
    assert arrival.power_upper_bound_W == pytest.approx(POWER_W * 0.995 * 0.995, rel=1e-12)
    plane = reference_run.result.observations[0]
    assert plane.total_power_W is None and plane.power_upper_bound_W is None  # transparent: unknown light may cross repeatedly
    assert plane.quantity_state.value == "UNKNOWN"
    assert "BULK_LOSS_NOT_ESTABLISHED" in [i.code.value for i in reference_run.result.issues]

    lossless = _run(_reference(), catalog=with_lossless_bulk(catalog_document()))
    transmitted = one_arrival(lossless.result, "focal_plane").beam.power_W
    assert transmitted == pytest.approx(POWER_W * 0.995 * 0.995, rel=1e-12)
    assert "BULK_LOSS_NOT_ESTABLISHED" not in issue_codes(lossless.result)


def test_several_wavelengths_share_one_lens():
    document = _reference()
    second = source("SRC2", (SOURCE_X, 0.0, 0.0), wavelength_nm=532.0)
    document["components"].insert(1, second)
    document["targets"] = []
    run = _run(document)
    assert run.model_scope_issues == ()
    waists = {a.beam.source_id: a.beam.gaussian.waist_distance_x_mm for a in run.result.observations[0].arrivals}
    assert waists["SRC2"] != pytest.approx(waists["SRC"], abs=0.1)  # dispersion moves the 532 nm focus


# ---- D. honest statuses -------------------------------------------------------------------------------------------------------------
def _assert_honest_record(run, status: str) -> None:
    record = run.local_validation
    assert run.status.value == status == run.engine_status.value and run.model_scope_issues == ()
    assert record.validation_status.value == status == record.simulation.status.value
    assert check_local_validation_binding(record, run.project) == ()
    assert record.input_hash.digest == run.result.manifest.input_hash
    assert record.result_hash.digest == run.result.manifest.result_hash


def test_out_of_scope_metric_is_reported_never_passed():
    document = _reference()
    document["targets"].append({"target_id": "psf_width", "metric": "psf", "observation_id": "focal_plane", "maximum": 0.05})
    run = _run(document)
    _assert_honest_record(run, "OUT_OF_MODEL_SCOPE")
    assert _outcome(run.local_validation, "psf_width").status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    assert _outcome(run.local_validation, "spot_radius").status is LocalValidationStatusV1.PASS


def test_missed_target_is_fail():
    document = _reference()
    document["targets"][1]["maximum"] = 0.01  # the focal spot is ~25 um
    run = _run(document)
    _assert_honest_record(run, "FAIL")
    assert _outcome(run.local_validation, "spot_radius").status is LocalValidationStatusV1.FAIL


def test_a_plane_behind_the_source_that_the_ghosts_reach_is_undecided():
    document = _reference(screen_x=SOURCE_X - 100.0)
    run = _run(document)
    _assert_honest_record(run, "UNKNOWN")
    assert "OBSERVATION_NOT_REACHED" in issue_codes(run.result)
    (plane,) = run.result.observations
    assert plane.quantity_state.value == "UNKNOWN" and plane.total_power_W is None and plane.power_upper_bound_W is None
    assert "BULK_LOSS_NOT_ESTABLISHED" in issue_codes(run.result)

    lossless = _run(document, catalog=with_lossless_bulk(catalog_document()))
    (ghosts,) = lossless.result.observations
    assert ghosts.quantity_state.value == "BOUNDED" and ghosts.total_power_W is None
    assert 0.0 < ghosts.untraced_power_W <= ghosts.power_upper_bound_W and ghosts.power_lower_bound_W == 0.0


def test_a_model_scope_issue_takes_precedence_over_unknown_catalog_glass_loss():
    run = _run(_reference(wavelength_nm=1200.0, screen_x=_screen_x(1200.0)))
    assert run.engine_status is Status.UNKNOWN and run.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    assert run.local_validation.simulation.status is LocalValidationStatusV1.UNKNOWN
    assert [t.status.value for t in run.result.targets] == ["PASS", "PASS"]  # geometry remains decidable
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in [i.code for i in run.model_scope_issues]


def test_engine_input_rejection_is_engine_run_incomplete():
    document = _reference()
    document["observations"][0]["medium"] = "water"  # a valid Project id, but not an Engine material
    assert _lower(document).request.project.observations[0].medium == "water"
    with pytest.raises(LocalRunError) as exc:
        _run(document)
    assert [c.value for c in exc.value.codes] == ["ENGINE_RUN_INCOMPLETE"]
    (problem,) = exc.value.problems
    assert "UNKNOWN_MATERIAL" in problem.message and "water" in problem.message


def test_target_metrics_are_the_optical_metrics():
    metrics = target_metrics()
    assert {name: metrics[name] for name in ("beam_radius_mm", "waist_distance_x_mm", "power_W")} == {
        "beam_radius_mm": "mm", "waist_distance_x_mm": "mm", "power_W": "W",
    }
    for mechanical in ("layout_violation_count", "layout_min_clearance_mm", "recoverability_fraction", "jacobian_condition_number",
                       "monte_carlo_recovered_fraction", "numerical_rank"):
        assert mechanical not in metrics
    assert "psf" not in metrics and list(metrics) == sorted(metrics)


# ---- E. fail-closed lowering --------------------------------------------------------------------------------------------------------
def _unpublished_sku(d):
    d["components"][1]["product"]["optical_sku"] = "OS-OPT-TEST-LENS-PCX-F999"


def _catalog_pin_mismatch(d):
    d["components"][1]["product"]["catalog_version"] = "2025-01-01.1"


def _source_without_model(d):
    del d["components"][0]["source_model"]


def _source_as_catalog_product(d):
    src = d["components"][0]
    del src["custom_part"]
    src["product"] = {"optical_sku": PCX_F100}


def _custom_mirror(d):
    d["components"].append({"instance_id": "M1", "optical_role": "mirror", "custom_part": {"custom_part_key": "fold", "description": "A user fold mirror."},
                            "nominal_pose": {"position_mm": [150.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}})


def _customer_instrument(role: str, *, attach: bool = True, absorbing: bool = True):
    """The user's own terminal instrument at the focal plane: a ``custom_part`` ``detector`` / ``dump`` with
    the screen attached to it (``device_instance_id``, the plane at the instrument's origin)."""

    def mutate(d):
        [screen] = d["observations"]
        d["components"].append({"instance_id": "CAM", "optical_role": role,
                                "custom_part": {"custom_part_key": "user-instrument", "description": "The user's own instrument."},
                                "nominal_pose": copy.deepcopy(screen["pose"])})
        if attach:
            screen["device_instance_id"] = "CAM"
            screen["pose"] = {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
        screen["absorbing"] = absorbing
    return mutate


def _selected_parameters(d):
    d["components"][1]["selected_parameters"] = [{"semantic_key": "focal_length_mm", "value_kind": "DECIMAL", "value": "100"}]


def _materials(d):
    d["materials"] = [{"material_id": "my_glass", "kind": "constant"}]


def _position_parameter_on_the_lens(d):
    d["components"][1]["selected_parameters"] = [{"semantic_key": "nd_rotation_deg", "value_kind": "DECIMAL", "value": "10"}]


def _position_parameter_on_the_source(d):
    d["components"][0]["selected_parameters"] = [{"semantic_key": "nd_offset_mm", "value_kind": "DECIMAL", "value": "1"}]


def _no_source(d):
    d["components"] = [d["components"][1]]


def _target_without_tolerance(d):
    del d["targets"][0]["tolerance_abs"]


def _target_names_a_lens_as_source(d):
    d["targets"][0]["source_id"] = "L1"


def _role(role):
    def mutate(d):
        d["components"][1]["optical_role"] = role
    return mutate


def _metric(name):
    def mutate(d):
        d["targets"][1]["metric"] = name
    return mutate


LOWERING_ERRORS = [
    ("unpublished_sku", _unpublished_sku, ["OPTICAL_SKU_NOT_PUBLISHED"]),
    ("catalog_pin_mismatch", _catalog_pin_mismatch, ["CATALOG_VERSION_MISMATCH"]),
    ("source_without_model", _source_without_model, ["SOURCE_MODEL_MISSING"]),
    ("source_as_catalog_product", _source_as_catalog_product, ["SOURCE_PRODUCT_UNSUPPORTED"]),
    ("custom_non_source_part", _custom_mirror, ["CUSTOM_PART_UNSUPPORTED"]),
    ("custom_detector_without_absorbing_plane", _customer_instrument("detector", absorbing=False), ["CUSTOM_PART_UNSUPPORTED"]),
    ("custom_dump_without_attached_plane", _customer_instrument("dump", attach=False), ["CUSTOM_PART_UNSUPPORTED"]),
    ("custom_aperture", _customer_instrument("aperture"), ["CUSTOM_PART_UNSUPPORTED"]),
    ("position_parameter_on_a_lens", _position_parameter_on_the_lens, ["SELECTED_PARAMETERS_UNSUPPORTED"]),
    ("position_parameter_on_the_source", _position_parameter_on_the_source, ["SELECTED_PARAMETERS_UNSUPPORTED"]),
    ("no_source", _no_source, ["SOURCE_MISSING"]),
    ("lens_as_mirror", _role("mirror"), ["PRODUCT_ROLE_MISMATCH"]),
    ("lens_as_window", _role("window"), ["PRODUCT_ROLE_MISMATCH"]),
]

#: Requirements that cannot be read as written: (case, mutation, the target it spoils, its code).
TARGET_PROBLEMS = [
    ("target_value_without_tolerance", _target_without_tolerance, "focus_position", "TARGET_INVALID"),
    ("target_source_is_not_a_source", _target_names_a_lens_as_source, "focus_position", "UNKNOWN_SOURCE_REFERENCE"),
    ("unknown_metric", _metric("spot_radius_mm"), "spot_radius", "TARGET_INVALID"),
    ("layout_metric", _metric("layout_min_clearance_mm"), "spot_radius", "TARGET_INVALID"),
    ("control_metric", _metric("recoverability_fraction"), "spot_radius", "TARGET_INVALID"),
]


@pytest.mark.parametrize("mutate, target_id, code", [pytest.param(m, t, c, id=i) for i, m, t, c in TARGET_PROBLEMS])
def test_a_requirement_that_cannot_be_read_is_undetermined_and_the_rest_runs(mutate, target_id, code, reference_run):
    """A malformed requirement is carried, not refused: the run's physics is the reference design's (the same planes, the
    other target's value), the spoilt target is UNKNOWN / INDETERMINATE with its input code and no value, and the
    requirements outcome counts it."""

    document = _reference()
    mutate(document)
    run = _run(ProjectDocumentV2.model_validate(document))
    spoilt = _target(run, target_id)
    assert (spoilt.status, spoilt.assessment, spoilt.value, spoilt.issue_codes) == (Status.UNKNOWN, "INDETERMINATE", None, (code,))
    assert spoilt.message
    assert target_id in run.result.outcome.requirements.indeterminate
    assert [o.model_dump(mode="json") for o in run.result.observations] == [
        o.model_dump(mode="json") for o in reference_run.result.observations]
    (other,) = [t for t in run.result.targets if t.target_id != target_id]
    assert other.model_dump(mode="json") == _target(reference_run, other.target_id).model_dump(mode="json")
    lowered = _lower(ProjectDocumentV2.model_validate(document))
    assert [(p.code.value, p.location) for p in lowered.target_problems] == [(code, f"target {target_id}")]
    (marked,) = [t for t in lowered.request.project.targets if t.target_id == target_id]
    assert (marked.not_assessable.code, marked.observation_id, marked.parameters) == (code, None, {})


@pytest.mark.parametrize("mutate, codes", [pytest.param(m, c, id=i) for i, m, c in LOWERING_ERRORS])
def test_unsupported_input_fails_closed(mutate, codes):
    document = _reference()
    mutate(document)
    project_ = ProjectDocumentV2.model_validate(document)  # a valid Project the local runtime cannot simulate faithfully
    with pytest.raises(LocalRunError) as exc:
        _run(project_)
    assert [c.value for c in exc.value.codes] == codes
    assert exc.value.as_dict()["status"] == "ERROR"
    assert all(p.message for p in exc.value.problems)
    with pytest.raises(LocalRunError) as lowered:
        _lower(project_)
    assert lowered.value.codes == exc.value.codes


@pytest.mark.parametrize("role", ["detector", "dump"])
def test_a_user_instrument_is_a_device_its_absorbing_plane_measures(role, reference_run):
    """The user's own camera / spectrometer / power meter (``detector``) or beam dump (``dump``) is a
    user-defined terminal instrument. It has no public optical model: it is an Engine device without surfaces, and the
    absorbing observation plane attached to it is its face, where the beam is evaluated and stops. The optics, and so
    every target, are the reference design's; the instrument is not a simulated catalog part."""

    document = _reference()
    _customer_instrument(role)(document)
    run = _run(ProjectDocumentV2.model_validate(document))
    device = next(c for c in run.request.project.components if c.instance_id == "CAM")
    assert (device.role, device.surfaces, device.source) == (role, (), None)
    assert device.pose.position_mm == (SCREEN_X, 0.0, 0.0)
    [screen] = run.request.project.observations
    assert (screen.device_id, screen.absorbing) == ("CAM", True)
    assert run.status == reference_run.status
    for target_id in ("focus_position", "spot_radius"):
        assert _target(run, target_id).value == pytest.approx(_target(reference_run, target_id).value, abs=1e-9)
    assert [p.instance_id for p in run.simulated_parts] == ["L1"]
    assert run.optical_view is not None


def test_an_attached_plane_is_posed_in_its_instrument_frame():
    """An attached plane's pose is in the instrument's local frame. A design-frame pose copied onto it lands
    elsewhere; the miss names the frame rule. The scene and the Optical View place the plane at the composed pose."""

    document = _reference()
    _customer_instrument("detector")(document)
    [screen] = document["observations"]
    screen["pose"] = {"position_mm": [SCREEN_X, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}  # a design-frame pose
    run = _run(ProjectDocumentV2.model_validate(document))
    [missed] = [i for i in run.result.issues if i.code == "OBSERVATION_NOT_REACHED"]
    assert "attached to 'CAM'" in missed.suggested_action and "local frame" in missed.suggested_action

    screen["pose"] = {"position_mm": [0.0, 0.0, -2.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}  # 2 mm in front of the camera
    project = ProjectDocumentV2.model_validate(document)
    run = _run(project)
    [plane] = run.optical_view.observation_planes
    composed = project.observation_design_pose(project.observations[0])
    assert plane.pose.position_mm == pytest.approx(composed.position_mm, abs=1e-9)
    assert composed.position_mm != pytest.approx((SCREEN_X, 0.0, 0.0), abs=1.0)


def test_a_role_refusal_names_the_roles_the_product_may_play():
    document = _reference()
    _role("mirror")(document)
    with pytest.raises(LocalRunError) as exc:
        _run(document)
    (problem,) = exc.value.problems
    assert problem.location == "component L1"
    assert PCX_F100 in problem.message and "LENS" in problem.message and "may play optical_role lens, not mirror" in problem.message


def test_a_dichroic_may_be_a_filter_or_a_splitter():
    for role in ("filter", "splitter"):
        document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=532.0),
                            element("DM", role, "OS-OPT-TEST-FLT-DICHROIC-SP600", (0.0, 0.0, 0.0), (0.27059805007309845, 0.27059805007309845, -0.6532814824381883, -0.6532814824381883))],
                           [plane("r", (0.0, 50.0, 0.0), (0.0, 0.0, 0.7071067811865476, 0.7071067811865476))])
        assert _run(document).model_scope_issues == ()


def test_all_problems_are_reported_together():
    document = _reference()
    for mutate in (_unpublished_sku, _materials, _custom_mirror, _selected_parameters, _target_names_a_lens_as_source):
        mutate(document)
    document["targets"][1]["source_id"] = "NOPE"
    document["targets"].append({"target_id": "no_tolerance", "metric": "beam_radius_mm", "observation_id": "focal_plane", "target_value": 0.02})
    with pytest.raises(LocalRunError) as exc:
        _run(document)
    # (the materials list and the lens's selected parameter are not simulation input: passed over, never a problem; a
    # requirement that cannot be read is carried as undetermined, never a problem)
    assert sorted(c.value for c in exc.value.codes) == sorted(["OPTICAL_SKU_NOT_PUBLISHED", "CUSTOM_PART_UNSUPPORTED"])
    assert {p.location for p in exc.value.problems} == {"component L1", "component M1"}
    # (the requirements alone: every one that cannot be read is listed at once, in Project order)
    targets = _reference()
    _target_names_a_lens_as_source(targets)
    targets["targets"][1]["source_id"] = "NOPE"
    targets["targets"].append({"target_id": "no_tolerance", "metric": "beam_radius_mm", "observation_id": "focal_plane", "target_value": 0.02})
    assert [(p.code.value, p.location) for p in _lower(ProjectDocumentV2.model_validate(targets)).target_problems] == [
        ("UNKNOWN_SOURCE_REFERENCE", "target focus_position"), ("UNKNOWN_SOURCE_REFERENCE", "target spot_radius"),
        ("TARGET_INVALID", "target no_tolerance")]


def test_inputs_the_simulation_does_not_read_are_passed_over_and_listed(reference_run, tmp_path):
    """A selected parameter that is not an attenuator position and the Project's materials list are not simulation input:
    the Project runs with the result of the Project without them, and the check, the run summary and the manifest name
    each entry."""

    import json as _json  # noqa: PLC0415

    from os_optics.local import json_api, verify_run_artifacts, write_run_artifacts  # noqa: PLC0415

    document = _reference()
    _selected_parameters(document)
    _materials(document)
    listed = [{"location": "materials", "field": "materials", "key": "my_glass"},
              {"location": "component L1", "field": "selected_parameters", "key": "focal_length_mm"}]
    lowered = _lower(document)
    assert list(lowered.inputs_not_simulated) == listed
    assert lowered.engine_input_hash == reference_run.engine_input_hash  # neither enters the Engine request
    run = _run(document)
    assert run.result.manifest.result_hash == reference_run.result.manifest.result_hash
    assert run.summary()["inputs_not_simulated"] == listed and "inputs_not_simulated" not in reference_run.summary()
    reply = json_api.check_project(document, catalog=catalog_document())
    assert reply["ok"] is True
    warnings = [w for w in reply["warnings"] if w["code"] == "INPUT_NOT_SIMULATED"]
    assert [(w["location"], w["details"]) for w in warnings] == [
        ("materials", {"field": "materials", "key": "my_glass"}),
        ("component L1", {"field": "selected_parameters", "key": "focal_length_mm"})]
    assert "'focal_length_mm' of component L1 is not simulation input" in warnings[1]["message"]
    assert "INPUT_NOT_SIMULATED" in json_api.capabilities()["check_warnings"]
    write_run_artifacts(run, tmp_path / "run")
    assert verify_run_artifacts(tmp_path / "run") == ()
    manifest = _json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs_not_simulated"] == listed
    write_run_artifacts(reference_run, tmp_path / "reference")
    assert "inputs_not_simulated" not in _json.loads((tmp_path / "reference" / "manifest.json").read_text(encoding="utf-8"))


def test_several_sources_without_models_report_each():
    document = _reference()
    document["components"].append(source("SRC2", (SOURCE_X, 5.0, 0.0)))
    for component in document["components"]:
        component.pop("source_model", None)
    with pytest.raises(LocalRunError) as exc:
        _run(document)
    assert [c.value for c in exc.value.codes] == ["SOURCE_MODEL_MISSING", "SOURCE_MODEL_MISSING"]


def test_model_kind_selects_the_published_model(reference_run):
    with pytest.raises(LocalRunError) as exc:
        _run(_reference(), model_kind="THIN_LENS")
    assert [(p.code.value, p.location) for p in exc.value.problems] == [("SIMULATION_MODEL_UNSUPPORTED", "component L1")]
    assert _run(_reference(), model_kind="SURFACE_SEQUENCE").engine_input_hash == reference_run.engine_input_hash


def test_a_duplicate_simulation_model_cannot_be_published():
    document = catalog_document()
    models = document["products"][0]["simulation_models"]
    models.append(copy.deepcopy(models[0]))
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog(document)
    assert [c.value for c in exc.value.codes] == ["CATALOG_INVALID"] and "duplicate simulation model" in str(exc.value)


def test_an_alternative_catalog_release_is_bound_in_the_record():
    document = catalog_document() | {"catalog_version": "2026-10-01.1"}
    lens = next(p for p in document["products"] if p["optical_sku"] == PCX_F100)
    lens["simulation_models"][0]["surfaces"][0]["radius_mm"] = "52"
    with pytest.raises(LocalRunError) as exc:  # the Project pins the test release
        _run(_reference(), catalog=document)
    assert [c.value for c in exc.value.codes] == ["CATALOG_VERSION_MISMATCH"]
    run = _run(_reference(catalog_version=None), catalog=document)
    assert run.catalog_version == "2026-10-01.1"
    assert [p.catalog_version for p in run.local_validation.simulated_parts] == ["2026-10-01.1"]
    surface = next(c for c in run.request.project.components if c.instance_id == "L1").surfaces[0]
    assert surface.radius_mm == 52.0
    assert check_local_validation_binding(run.local_validation, run.project) == ()


def test_an_explicit_catalog_is_used_as_given_and_the_store_keeps_one_document_per_version(tmp_path):
    """No bundled release to compare against: an explicit catalog is the caller's choice. The local store (and an
    in-memory install) never accepts a second, different document under a catalog_version it holds."""

    from os_optics.catalog import CatalogError, CatalogErrorCode, import_release
    from os_optics.local import install_catalog

    edited = catalog_document()
    edited["products"][0]["display_name"] = edited["products"][0]["display_name"] + " (edited)"
    assert load_public_catalog(edited).catalog_version == TEST_CATALOG_VERSION
    with pytest.raises(CatalogError) as exc:
        import_release(json.dumps(edited).encode("utf-8"))
    assert exc.value.code is CatalogErrorCode.CATALOG_RELEASE_CONFLICT
    install_catalog(FIXTURE_CATALOG.read_bytes())
    with pytest.raises(LocalRunError) as refused:
        install_catalog(json.dumps(edited).encode("utf-8"))
    assert [c.value for c in refused.value.codes] == ["CATALOG_INVALID"]


@pytest.mark.parametrize("source_", [
    {"schema": "opensci.project", "project_schema_version": "1.0"},
    {**project([], []), "baseplate": {"sku": "anything"}},
    "{not json",
    b"\xff\xfe",
    42,
])
def test_malformed_project_is_project_invalid(source_):
    for call in (load_project, run_optical_project):
        with pytest.raises(LocalRunError) as exc:
            call(source_)
        assert [c.value for c in exc.value.codes] == ["PROJECT_INVALID"]


def test_missing_project_file_is_project_invalid(tmp_path):
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(tmp_path / "missing.json")
    assert [c.value for c in exc.value.codes] == ["PROJECT_INVALID"]


@pytest.mark.parametrize("catalog", [{"schema": "opensci.design_catalog.public_optical_catalog"}, "[]", "{oops"])
def test_invalid_catalog_is_catalog_invalid(catalog):
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(_reference(), catalog=catalog)
    assert [c.value for c in exc.value.codes] == ["CATALOG_INVALID"]


def test_record_refuses_a_result_of_another_request(reference_run):
    other = _run(_reference(screen_x=80.0))
    lowered = _lower(reference_run.project)
    with pytest.raises(LocalRunError) as exc:
        local_validation_record(reference_run.project, lowered, other.result)
    assert [c.value for c in exc.value.codes] == ["BINDING_INCONSISTENT"]


# ---- F. no mechanical / supplier leakage --------------------------------------------------------------------------------------------
FORBIDDEN_TERMS = ("supplier", "mount", "post", "holder", "breadboard", "baseplate", "design_part")


def test_no_mechanical_or_supplier_data_in_the_outputs(reference_run):
    for name, doc in (("summary", reference_run.summary()), ("record", reference_run.local_validation.payload())):
        text = json.dumps(doc, sort_keys=True).lower()
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"{term!r} in the {name}"
    request = reference_run.request.model_dump(mode="json")
    # Two Engine request field names match on purpose: *.mount_margin_mm is the opaque rim from the published clear radius
    # to the published edge radius (no mount geometry), and ProjectSnapshot.baseplate is the Engine's empty plate slot.
    assert request["project"]["baseplate"] is None
    text = json.dumps(request, sort_keys=True).lower()
    assert text.count("mount") == text.count('"mount_margin_mm"') == 2
    assert text.count("baseplate") == text.count('"baseplate": null') == 1
    for term in FORBIDDEN_TERMS:
        if term not in ("mount", "baseplate"):
            assert term not in text, f"{term!r} in the Engine request"


# ---- G. low-level Engine API --------------------------------------------------------------------------------------------------------
def test_low_level_engine_api_is_unchanged():
    from os_optics import ValidationRequest, builders as B, validate
    from os_optics.contracts import ProjectSnapshot

    assert os_optics.validate is validate
    project_ = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=780.0, waist_radius_mm=1.0, position=(0, 0, 0), direction=(1, 0, 0)),
                    B.thin_lens("Lens", focal_length_mm=75.0, clear_radius_mm=11.43, position=(10, 0, 0), axis=(1, 0, 0))),
        observations=(B.observation_plane("focal", position=(85, 0, 0), normal=(1, 0, 0), radius_mm=5.0),),
    )
    result = validate(ValidationRequest(project=project_))
    assert result.status.value in ("PASS", "WARNING") and result.manifest is not None
    assert one_arrival(result, "focal").beam.gaussian.valid


# ---- H. CLI -------------------------------------------------------------------------------------------------------------------------
def _cli(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # catalog rows carry Chinese names
    return subprocess.run([sys.executable, "-m", "os_optics", *args], capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=120)


def _write(path: Path, document: object) -> str:
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_cli_catalog_list_and_show(tmp_path):
    listed = _cli("catalog", "--catalog", str(FIXTURE_CATALOG), "list", "--category", "LENS", "--wavelength-nm", "780")
    assert listed.returncode == 0, listed.stderr
    doc = json.loads(listed.stdout)
    assert doc["catalog_version"] == TEST_CATALOG_VERSION and doc["total"] == 3 and doc["offset"] == 0 and doc["limit"] is None
    assert [p["optical_sku"] for p in doc["products"]] == ["OS-OPT-TEST-LENS-PCV-FN50", PCX_F100, "OS-OPT-TEST-LENS-PCX-F50"]
    row = doc["products"][1]
    assert (row["model_kind"], row["fidelity"], row["focal_length_mm"], row["design_roles"]) == ("SURFACE_SEQUENCE", "SURFACE_PRESCRIPTION", "100", ["lens"])
    assert row["working_band_nm"] == {"min_nm": "400", "max_nm": "1100"}
    paged = json.loads(_cli("catalog", "--catalog", str(FIXTURE_CATALOG), "list", "--form", "PLANO_CONVEX_LENS", "--offset", "1", "--limit", "5").stdout)
    assert (paged["total"], [p["optical_sku"] for p in paged["products"]]) == (2, ["OS-OPT-TEST-LENS-PCX-F50"])
    assert json.loads(_cli("catalog", "--catalog", str(FIXTURE_CATALOG), "list", "--wavelength-nm", "250").stdout)["products"] == []
    bundled = json.loads(_cli("catalog", "list", "--limit", "2").stdout)
    assert bundled["catalog_version"] == load_public_catalog().catalog_version and len(bundled["products"]) == 2
    assert bundled["total"] == len(load_public_catalog().products)
    shown = _cli("catalog", "--catalog", str(FIXTURE_CATALOG), "show", PCX_F100)
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["simulation_models"][0]["surfaces"][0]["radius_mm"] == "51.5"
    missing = _cli("catalog", "show", "OS-OPT-NOPE")
    assert missing.returncode == 65
    assert [p["code"] for p in json.loads(missing.stdout)["problems"]] == ["OPTICAL_SKU_NOT_PUBLISHED"]
    no_file = _cli("catalog", "--catalog", str(tmp_path / "missing.json"), "list")
    assert no_file.returncode == 64 and "cannot read the catalog" in no_file.stderr
    assert _cli("catalog", "list", "--category", "SPOON").returncode == 64


def test_cli_project_example_and_schemas():
    example = _cli("project", "example")
    assert example.returncode == 0, example.stderr
    assert ProjectDocumentV2.model_validate(json.loads(example.stdout)).payload() == example_project().payload()
    for which, title in (("project", "ProjectDocumentV2"), ("local-validation", "LocalValidationRecordV1"), ("catalog", "PublicOpticalCatalogV2")):
        out = _cli("schema", which)
        assert out.returncode == 0, (which, out.stderr)
        assert json.loads(out.stdout)["title"] == title


def test_cli_project_run_writes_a_valid_record(tmp_path, reference_run):
    project_path, record_path = _write(tmp_path / "project.json", _reference()), tmp_path / "record.json"
    out = _cli("project", "run", project_path, "--catalog", str(FIXTURE_CATALOG), "-o", str(record_path))
    assert out.returncode == 0, out.stderr
    summary = json.loads(out.stdout)
    assert (summary["status"], summary["engine_status"], summary["model_scope_issues"]) == ("UNKNOWN", "UNKNOWN", [])
    assert [target["status"] for target in summary["targets"]] == ["PASS", "PASS"]
    record = LocalValidationRecordV1.model_validate_json(record_path.read_text(encoding="utf-8"))
    assert check_local_validation_binding(record, load_project(Path(project_path))) == ()
    assert record.input_hash.digest == reference_run.engine_input_hash
    assert record.payload() == summary["local_validation"] == reference_run.local_validation.payload()
    # The summary is printed before the record is written: an unwritable -o still shows the result, then exits 64.
    unwritable = _cli("project", "run", project_path, "--catalog", str(FIXTURE_CATALOG), "-o", str(tmp_path))
    assert unwritable.returncode == 64 and "cannot write the record" in unwritable.stderr


def test_cli_project_run_exit_codes(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "opensci.project"}', encoding="utf-8")
    out = _cli("project", "run", str(bad))
    assert out.returncode == 65 and [p["code"] for p in json.loads(out.stdout)["problems"]] == ["PROJECT_INVALID"]
    assert _cli("project", "run", str(tmp_path / "missing.json")).returncode == 64
    catalog = ("--catalog", str(FIXTURE_CATALOG))
    assert _cli("project", "run", _write(tmp_path / "p.json", _reference()), "--catalog", str(tmp_path / "missing-catalog.json")).returncode == 64

    role = _reference()
    _role("mirror")(role)
    out = _cli("project", "run", _write(tmp_path / "role.json", role), *catalog)
    assert out.returncode == 65 and [p["code"] for p in json.loads(out.stdout)["problems"]] == ["PRODUCT_ROLE_MISMATCH"]

    water = _reference()
    water["observations"][0]["medium"] = "water"
    out = _cli("project", "run", _write(tmp_path / "water.json", water), *catalog)
    assert out.returncode == 65 and [p["code"] for p in json.loads(out.stdout)["problems"]] == ["ENGINE_RUN_INCOMPLETE"]

    band = _write(tmp_path / "band.json", _reference(wavelength_nm=1200.0, screen_x=_screen_x(1200.0)))
    out = _cli("project", "run", band, *catalog)
    assert out.returncode == 2  # (the default requirements contract: the targets on the out-of-band light are undetermined)
    summary = json.loads(out.stdout)
    assert summary["exit_contract"] == "requirements" and summary["outcome"]["requirements"]["met"] == []
    assert _cli("project", "run", band, *catalog, "--status-contract", "aggregate").returncode == 3
    assert (summary["status"], summary["engine_status"]) == ("OUT_OF_MODEL_SCOPE", "UNKNOWN")
    assert [i["code"] for i in summary["model_scope_issues"]] == ["MODEL_WAVELENGTH_OUTSIDE_BAND", "COATING_WAVELENGTH_OUTSIDE_BANDS",
                                                                   "COATING_WAVELENGTH_OUTSIDE_BANDS"]

    missed = _reference()
    missed["targets"][1]["maximum"] = 0.01
    assert _cli("project", "run", _write(tmp_path / "missed.json", missed), *catalog).returncode == 1
