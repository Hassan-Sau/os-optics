"""Caller settings as Project state: ``numerical_settings`` through the contract, lowering, editing, discovery and the run
record, and the registry that names every setting's home, default, direction and meaning.

Recording rule (tested here): every setting a result names (``Omitted.setting``, ``ValidityCheck.setting``) is a key of its
run's ``manifest.numerical_tolerances``, where the value in force is recorded.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import pytest
from annotated_types import Ge, Gt, Le, Lt
from pydantic import ValidationError

from os_optics import NumericalConfig
from os_optics._vendor.shared_contracts.design_catalog.enums import CoatingKind
from os_optics._vendor.shared_contracts.optical_design.compute_budget import BUDGET_RANGES, ComputeBudgetV1
from os_optics._vendor.shared_contracts.optical_design.numerical_settings import (
    DESIGN_POINT_SETTINGS,
    SETTING_NAMES,
    DesignPointTolerancesV1,
    NumericalSettingsV1,
    SingleAngleTolerancesV1,
)
from os_optics.config import DEFAULT_NUMERICAL_CONFIG
from os_optics.contracts.basis import ValidityCheck, ValueBasis
from os_optics.errors.status import ValueKind
from os_optics.local import json_api as J
from os_optics.local import lower_project, run_optical_project
from os_optics.local.design_point import DEFAULT_DESIGN_POINT, ENGINE_DEFAULT, PROJECT
from os_optics.local.evaluate import capability_descriptor
from os_optics.local.lowering import pure_optical_engine_configuration
from os_optics.local.project_ops import ProjectOpError, apply_project_ops
from os_optics.local.resources import estimate_run_resources
from os_optics.run_control import STOP_ARTIFACTS, RunStopped
from os_optics.run_monitor import RunMonitor
from os_optics.settings_registry import (
    NOT_SETTABLE,
    SETTINGS,
    SettingDirection,
    SettingHome,
    named_settings,
    unrecorded_settings,
)

from composition_helpers import laser, project_of, screen


def _project(power_w: float = 0.001) -> dict:
    return project_of([laser("S", power_w=power_w)], [screen("screen", (100, 0, 0), absorbing=True)])


def _bounds(model, name: str) -> dict[str, float]:
    found = {}
    for item in model.model_fields[name].metadata:
        for kind, attr in ((Gt, "gt"), (Ge, "ge"), (Lt, "lt"), (Le, "le")):
            if isinstance(item, kind):
                found[attr] = getattr(item, attr)
    return found


def _contract_field(name: str) -> tuple[type, str]:
    """The contract model and field that set NumericalConfig field ``name``."""

    if name in BUDGET_RANGES:
        return ComputeBudgetV1, name
    for path, engine in DESIGN_POINT_SETTINGS.items():
        if engine == name:
            *parent, leaf = path.split(".")
            return (SingleAngleTolerancesV1 if parent else DesignPointTolerancesV1), leaf
    return NumericalSettingsV1, name


# ------------------------------------------------------------------------------------------------ contract = Engine


@pytest.mark.parametrize("name", [*BUDGET_RANGES, *SETTING_NAMES])
def test_every_settable_field_accepts_exactly_the_engines_range(name):
    # a contract range wider than NumericalConfig's would accept a value the Engine refuses at lowering
    model, field = _contract_field(name)
    engine = _bounds(NumericalConfig, name)
    contract = _bounds(model, field)
    assert contract == engine, (name, contract, engine)
    assert getattr(DEFAULT_NUMERICAL_CONFIG, name) is not None


def _faint_leak() -> dict:
    import json as _json

    return _json.loads((Path(__file__).parent / "fixtures" / "regressions" / "faint_leak_project.json").read_text(encoding="utf-8"))


def _fluorescence_sample() -> dict:
    from test_weak_signal_chain import CHAIN_TARGETS, chain_detection, chain_project, fluorescence

    return chain_project(fluorescence(absorbance=1e-4), chain_detection(), CHAIN_TARGETS)


@pytest.mark.parametrize("design", [_faint_leak, _fluorescence_sample])
@pytest.mark.parametrize("nodes", [2**53 - 1, 2**53 + 1])
def test_a_node_count_the_machine_cannot_hold_is_a_named_refusal_not_an_internal_error(nodes, design, tmp_path, capsys):
    """A Project setting more Gauss-Legendre nodes than JSON holds exactly (2^53 + 1: the contract), or a count whose rule no
    machine can allocate (2^53 - 1: 64 PiB per array), used to end the run in a MemoryError (exit 70): it is refused by
    name (exit 65), no traceback, no internal error -- also for a Project with a fluorescence sample, whose excitation trace
    runs while the Project is lowered, and from the JSON API (a PROJECT_INVALID reply, nothing raised)."""
    import json as _json

    from os_optics.cli import main

    document = design()
    document["numerical_settings"] = {"aperture_quad_sector_nodes": nodes}
    path = tmp_path / "p.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(tmp_path / "run"), "--progress", "none"]) == 65
    printed = capsys.readouterr()
    assert "aperture_quad_sector_nodes" in printed.out + printed.err and "Traceback" not in printed.err
    assert "INTERNAL_ERROR" not in printed.out
    reply = J.run_project(document)
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "PROJECT_INVALID", reply
    assert "aperture_quad_sector_nodes" in reply["error"]["problems"][0]["message"]


def test_an_engine_request_whose_rule_the_machine_cannot_hold_is_a_named_invalid_input(tmp_path, capsys):
    """The Engine request of a run with ``aperture_quad_sector_nodes`` 2^53 - 1 (within the contract, no machine holds its
    rule): ``os-optics validate`` answers a FAIL result with an INVALID_INPUT issue naming the setting (exit 1), no
    traceback, no internal error; ``validate`` and ``validate_json`` return that result and raise nothing."""
    import json as _json

    from os_optics import ValidationRequest, validate
    from os_optics.cli import main
    from os_optics.validation.pipeline import QUADRATURE_SETTING, validate_json

    out = tmp_path / "run"
    main(["project", "run", str(_write(tmp_path, _faint_leak())), "--out-dir", str(out), "--progress", "none"])  # (its targets' exit)
    capsys.readouterr()
    request = _json.loads((out / "engine_request.json").read_text(encoding="utf-8"))
    request["configuration"]["numerical"]["aperture_quad_sector_nodes"] = 2**53 - 1
    path = tmp_path / "request.json"
    path.write_text(_json.dumps(request), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    printed = capsys.readouterr()
    assert "Traceback" not in printed.err and "INTERNAL_ERROR" not in printed.out
    for result in (_json.loads(printed.out), _json.loads(validate_json(_json.dumps(request))),
                   validate(ValidationRequest.model_validate(request)).model_dump(mode="json")):
        assert result["status"] == "FAIL"
        (issue,) = result["issues"]
        assert issue["code"] == "INVALID_INPUT" and issue["location"] == QUADRATURE_SETTING
        assert issue["message"].startswith(f"{QUADRATURE_SETTING} is {2**53 - 1}: ")
        assert result["outcome"]["execution"]["outcome"] == "FAILED"


def _write(tmp_path: Path, document: dict) -> Path:
    import json as _json

    path = tmp_path / "project.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    return path


def test_a_large_node_count_is_the_callers_and_its_rule_is_stopped_by_the_runs_monitor(tmp_path, capsys):
    """No capacity cap on the node count: 1025 nodes run like 1024 (the rule's sum of weights stays 2, every node inside
    (-1, 1)), and the build of a rule of 10 million nodes -- months of work -- is stopped by the run's monitor at its
    first checkpoint (cancel, a memory limit), not left running."""
    import json as _json

    from os_optics.cli import main
    from os_optics.optics.gaussian.truncation import gauss_legendre_rule
    from os_optics.run_control import RunStopped, monitored

    nodes, weights = gauss_legendre_rule(1025)
    assert abs(float(weights.sum()) - 2.0) < 1e-12 and float(abs(nodes).max()) < 1.0
    document = _json.loads((Path(__file__).parent / "fixtures" / "regressions" / "faint_leak_project.json").read_text(encoding="utf-8"))
    path = tmp_path / "p.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    default = main(["project", "run", str(path), "--progress", "none"])
    capsys.readouterr()
    document["numerical_settings"] = {"aperture_quad_sector_nodes": 1025}
    path.write_text(_json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--progress", "none"]) == default
    assert _json.loads(capsys.readouterr().out)["status"]

    class Stop:
        steps = 0

        def enter(self, name, total, unit):
            pass

        def leave(self, name):
            pass

        def unwind(self, name):
            pass

        def step(self, count):
            self.steps += 1
            raise RunStopped("CALLER", "stopped", phase="trace")

    monitor = Stop()
    with monitored(monitor), pytest.raises(RunStopped):
        gauss_legendre_rule(10**7)
    assert monitor.steps == 1


def test_the_registry_names_every_settable_field_once_with_its_home_default_and_meaning():
    assert list(SETTINGS) == [*BUDGET_RANGES, *SETTING_NAMES]
    for name, entry in SETTINGS.items():
        assert name in NumericalConfig.model_fields
        assert entry.default == getattr(NumericalConfig(), name)
        assert entry.home is (SettingHome.COMPUTE_BUDGET if name in BUDGET_RANGES else SettingHome.NUMERICAL_SETTINGS)
        assert entry.path.startswith(entry.home.value + ".")
        assert isinstance(entry.direction, SettingDirection)
        assert 10 < len(entry.meaning) < 140 and "." not in entry.meaning.rstrip(".")[-1:]
    for name in NOT_SETTABLE:
        assert name in NumericalConfig.model_fields and name not in SETTINGS
    assert {"laboratory_reach_mm", "hash_float_sig_digits", "hash_float_abs_floor", "cross_runtime_rel_tol",
            "cross_runtime_abs_tol", "config_version"} <= set(NOT_SETTABLE)
    assert SETTINGS["design_point_angle_dielectric_high_reflector_deg"].path == \
        "numerical_settings.design_point.single_angle_deg.dielectric_high_reflector"
    assert SETTINGS["power_epsilon_W"].direction is SettingDirection.LOWER_ACCEPTS_MORE
    assert SETTINGS["gaussian_scope_divergence_rad"].direction is SettingDirection.RAISE_ACCEPTS_MORE


def test_the_defaults_are_todays_constants():
    from os_optics.validation import exposures as X, rating_conditions as RC

    cfg = DEFAULT_NUMERICAL_CONFIG
    assert cfg.negligible_power_fraction == cfg.energy_conservation_tol == 1e-9
    # (the two work budgets are read from the run's config alone: no module constant holds them any more)
    assert (cfg.max_scatter_origins, cfg.max_mesh_patches) == (64, 1024)
    # (the rating windows have no module constant any more: the comparison reads the run's settings, and a call without
    # a configuration reads these defaults -- 2 %, 10 %, 10 %, 5 %, 1 %, 1 %, the constants they replaced)
    assert RC.RatingTolerances.of(cfg) == RC.DEFAULT_TOLERANCES == RC.RatingTolerances(0.02, 0.10, 0.10, 0.05, 0.01, 0.01)
    assert not [name for name in ("WAVELENGTH_TOLERANCE_REL", "PULSE_DURATION_TOLERANCE_REL", "REPETITION_RATE_TOLERANCE_REL",
                                  "BEAM_DIAMETER_TOLERANCE_REL") if hasattr(RC, name)]
    assert not hasattr(X, "POLARIZATION_FRACTION_TOLERANCE") and not hasattr(X, "TOLERANCES")
    point = DEFAULT_DESIGN_POINT
    assert point.wavelength_rel == Decimal("0.005")
    assert point.retarder_wavelength_rel == Decimal("0.0005")
    assert dict(point.angles_deg) == {CoatingKind.METALLIC_REFLECTOR: 5.0, CoatingKind.DIELECTRIC_HIGH_REFLECTOR: 3.0,
                                      CoatingKind.ANTI_REFLECTION: 3.0, CoatingKind.BEAMSPLITTER: 2.0}
    assert point.angle_deg(CoatingKind.DICHROIC) == 0.0
    assert point.interference_shift_fraction == 0.25
    assert point.interference_angle_max_deg == 3.0


# ------------------------------------------------------------------------------------------------ identity and lowering


def test_a_project_without_settings_lowers_as_with_an_empty_block():
    plain = _project()
    stated_empty = {**plain, "numerical_settings": {}}
    a, b = lower_project(plain), lower_project(stated_empty)
    assert a.engine_input_hash == b.engine_input_hash
    assert a.request.configuration.numerical == NumericalConfig()
    assert a.design_point == DEFAULT_DESIGN_POINT and a.stated_settings == ()
    assert pure_optical_engine_configuration(None, None) == pure_optical_engine_configuration()


def test_the_settings_are_lowered_into_the_engine_input_and_the_design_point_resolved_once():
    settings = {"gaussian_max_aoi_deg": 70, "rating_wavelength_tol_rel": 0.06,
                "design_point": {"single_wavelength_rel": 0.01, "single_angle_deg": {"dielectric_high_reflector": 15}}}
    lowered = lower_project({**_project(), "numerical_settings": settings})
    cfg = lowered.request.configuration.numerical
    assert (cfg.gaussian_max_aoi_deg, cfg.rating_wavelength_tol_rel) == (70.0, 0.06)
    assert cfg.design_point_angle_dielectric_high_reflector_deg == 15.0 and cfg.design_point_wavelength_rel == 0.01
    assert cfg.design_point_angle_beamsplitter_deg == 2.0  # (not stated: the default)
    point = lowered.design_point
    assert point.angle_deg(CoatingKind.DIELECTRIC_HIGH_REFLECTOR) == 15.0 and point.wavelength_rel == Decimal("0.01")
    assert point.source("design_point_angle_dielectric_high_reflector_deg") == PROJECT
    assert point.source("design_point_angle_beamsplitter_deg") == ENGINE_DEFAULT
    assert set(lowered.stated_settings) == {"gaussian_max_aoi_deg", "rating_wavelength_tol_rel", "design_point_wavelength_rel",
                                            "design_point_angle_dielectric_high_reflector_deg"}
    assert hash(point)  # (hashable: it can key the compiled-element cache)
    assert lowered.engine_input_hash != lower_project(_project()).engine_input_hash


def test_configure_inspect_reset_and_identity():
    project = _project()
    baseline = J.check_project(project)
    assert baseline["numerical_settings"] == {name: getattr(NumericalConfig(), name) for name in SETTING_NAMES}
    op = {"op": "set_numerical_settings", "numerical_settings": {"power_epsilon_W": 1e-18,
                                                                 "design_point": {"single_angle_deg": {"beamsplitter": 4}}}}
    edited = J.update_project(project, [op])
    assert edited["ok"], edited
    assert edited["project"]["numerical_settings"] == {"power_epsilon_W": 1e-18, "design_point": {"single_angle_deg": {"beamsplitter": 4.0}}}
    assert edited["numerical_settings"]["power_epsilon_W"] == 1e-18
    assert edited["numerical_settings"]["design_point_angle_beamsplitter_deg"] == 4.0
    assert edited["compute_budget"] == baseline["compute_budget"]
    assert edited["project_hash"] != baseline["project_hash"] and edited["engine_input_hash"] != baseline["engine_input_hash"]
    assert not J.update_project(edited["project"], [op])["changed"]
    for reset_op in ({"op": "set_numerical_settings", "numerical_settings": None},
                     {"op": "set_numerical_settings", "numerical_settings": {}}):
        reset = J.update_project(edited["project"], [reset_op])
        assert (reset["project_hash"], reset["engine_input_hash"]) == (baseline["project_hash"], baseline["engine_input_hash"])
        assert "numerical_settings" not in reset["project"]


@pytest.mark.parametrize(("settings", "message"), [
    ([1], "must be an object"),
    ({"laboratory_reach_mm": 2000}, "numerical_settings.laboratory_reach_mm"),
    ({"gaussian_max_aoi_deg": 95}, "numerical_settings.gaussian_max_aoi_deg"),
    ({"design_point": {"single_angle_deg": {"beamsplitter": -1}}}, "numerical_settings.design_point.single_angle_deg.beamsplitter"),
    ({"power_epsilon_W": True}, "numerical_settings.power_epsilon_W"),
])
def test_a_setting_the_engine_would_refuse_is_refused_at_the_edit(settings, message):
    with pytest.raises(ProjectOpError) as refused:
        apply_project_ops(_project(), [{"op": "set_numerical_settings", "numerical_settings": settings}])
    assert refused.value.code == "BAD_ARGUMENT" and message in refused.value.message


def test_each_configuration_lowers_with_its_merged_settings():
    project = {**_project(), "numerical_settings": {"gaussian_max_aoi_deg": 70},
               "configurations": [{"configuration_id": "a"},
                                  {"configuration_id": "b", "numerical_settings": {"capture_min_power_fraction": 1e-4}}]}
    reply = J.check_project(project)
    assert reply["ok"], reply
    assert reply["numerical_settings"] is None
    rows = {row["configuration_id"]: row["numerical_settings"] for row in reply["configurations"]}
    assert rows["a"]["gaussian_max_aoi_deg"] == rows["b"]["gaussian_max_aoi_deg"] == 70.0
    assert rows["a"]["capture_min_power_fraction"] == 1e-3 and rows["b"]["capture_min_power_fraction"] == 1e-4


# ------------------------------------------------------------------------------------------------ discovery


def test_capabilities_list_every_setting_with_its_range_default_and_home():
    caps = J.capabilities()
    block = caps["numerical_settings"]
    assert block["operation"] == "set_numerical_settings" and block["persistence"] == "Project.numerical_settings"
    assert list(block["fields"]) == list(SETTING_NAMES)
    assert {name: f["default"] for name, f in block["fields"].items()} == J.check_project(_project())["numerical_settings"]
    assert block["fields"]["power_epsilon_W"] == {"default": 1e-15, "unit": "W", "direction": "LOWER_ACCEPTS_MORE"}
    assert block["fields"]["design_point_angle_beamsplitter_deg"]["path"] ==         "numerical_settings.design_point.single_angle_deg.beamsplitter"
    assert "set_numerical_settings" in caps["project_ops"]["ops"] and "numerical_settings" in caps["project_features"]
    assert caps["compute_budget"]["fields"]["pupil_ring_count"] == {"minimum": 1, "maximum": 2**53 - 1, "default": 8}
    assert caps["compute_budget"]["fields"]["max_scatter_origins"] == {"minimum": 1, "maximum": 2**53 - 1, "default": 64}


def test_a_capability_descriptor_names_the_controls_a_project_sets():
    from os_optics.evaluation.capabilities import CAPABILITIES

    settable = set(BUDGET_RANGES) | set(SETTING_NAMES)
    for capability in CAPABILITIES:
        try:
            row = capability_descriptor(capability.id)
        except KeyError:
            continue
        assert row["compute_budget"] == [c for c in row["controls"] if c in BUDGET_RANGES]
        assert row["numerical_settings"] == [c for c in row["controls"] if c in SETTING_NAMES]
        assert set(row["compute_budget"]) | set(row["numerical_settings"]) == {c for c in row["controls"] if c in settable}
    trace = capability_descriptor("beam_trace")
    assert {"power_epsilon_W", "gaussian_scope_divergence_rad"} <= set(trace["numerical_settings"])
    pairs = capability_descriptor("beam_pairs")
    assert {"interference_wavelength_tol_nm", "interference_min_overlap"} <= set(pairs["numerical_settings"])
    assert pairs["compute_budget"] == ["max_interference_pairs"]
    # (the two work budgets the lowering and the scatter rays read)
    assert capability_descriptor("scene")["compute_budget"] == ["max_mesh_patches"]
    assert capability_descriptor("secondary_rays")["compute_budget"] == ["max_scatter_origins"]
    assert "laboratory_reach_mm" in capability_descriptor("plane_bounds")["controls"]
    assert "laboratory_reach_mm" not in capability_descriptor("plane_bounds")["numerical_settings"]
    # (each setting the trace or an analysis reads is named by what reads it: the nonlinear limits by the trace, the Gaussian
    # scope limit by the focal field, the interference wavelength tolerance by the irradiance map and the detector response)
    assert {"nonlinear_depletion_rel_tol", "nonlinear_focusing_rel_tol", "nonlinear_quadrature_rel_tol",
            "nonlinear_harmonic_gaussian_rel_tol", "nonlinear_beam_quality_tol", "nonlinear_linewidth_rel_tol"} <= set(trace["numerical_settings"])
    assert capability_descriptor("focal_field")["numerical_settings"] == ["gaussian_scope_divergence_rad"]
    assert capability_descriptor("irradiance_map")["numerical_settings"] == ["interference_wavelength_tol_nm"]
    assert "interference_wavelength_tol_nm" in capability_descriptor("detector_response")["numerical_settings"]
    named = {c for capability in CAPABILITIES for c in capability.controls}
    assert set(SETTING_NAMES) <= named  # (every Project setting is read by something the descriptors name)
    # the analyses that lower and trace the Project again read the lowering's and the trace's settings, less those their
    # own request or launch grid sets: ghosts replace the path and step budgets, imaging the pupil sampling
    from os_optics.evaluation.capabilities import ANALYSES

    for analysis in ANALYSES:
        row = capability_descriptor(analysis.id)
        assert row["compute_budget"] == [c for c in row["controls"] if c in BUDGET_RANGES]
        assert row["numerical_settings"] == [c for c in row["controls"] if c in SETTING_NAMES]
    ghosts, field, imaging = (capability_descriptor(name) for name in ("ghosts", "field", "imaging"))
    for row in (ghosts, field, imaging):
        assert {"max_interactions_per_path", "negligible_power_fraction", "max_mesh_patches"} <= set(row["compute_budget"]) | set(
            row["numerical_settings"]), row["id"]
        assert "gaussian_scope_divergence_rad" in row["numerical_settings"], row["id"]
    # (every tracer builds the aperture rule -- the power a beam passes at an obstruction or a clipping aperture -- and
    # decides a beam's capture on an active area: the beam trace and each analysis that traces again read both settings)
    for row in (trace, ghosts, field, imaging):
        assert {"aperture_quad_sector_nodes", "capture_min_power_fraction"} <= set(row["numerical_settings"]), row["id"]
    assert {"pupil_ring_count", "pupil_points_per_ring", "max_paths", "max_trace_steps"} <= set(field["compute_budget"])
    assert {"pupil_ring_count", "pupil_points_per_ring"} <= set(ghosts["compute_budget"])
    assert not {"max_paths", "max_trace_steps"} & set(ghosts["controls"])
    assert {"max_paths", "max_trace_steps"} <= set(imaging["compute_budget"])
    assert not {"pupil_ring_count", "pupil_points_per_ring"} & set(imaging["controls"])


# ------------------------------------------------------------------------------------------------ a setting in force


def test_a_lowered_power_floor_traces_a_source_below_the_default_floor():
    from local_helpers import PCX_F100, catalog_document, element, plane, project, run, source, with_lossless_bulk

    def lens_project(power_w: float) -> dict:
        return project([source(power_w=power_w), element("L1", "lens", PCX_F100)], [plane("focal_plane", (101.5, 0.0, 0.0))])

    catalog = with_lossless_bulk(catalog_document())
    weak = run(lens_project(1e-16), catalog=catalog).result
    assert weak.observations[0].total_power_W is None and [i.code.value for i in weak.issues] == ["ZERO_POWER"]
    stated = run({**lens_project(1e-16), "numerical_settings": {"power_epsilon_W": 1e-20}}, catalog=catalog).result
    bright = run(lens_project(1e-3), catalog=catalog).result
    # the trace is linear in power: the weak source's plane power is the bright one's scaled by 1e-13
    assert stated.observations[0].total_power_W == pytest.approx(bright.observations[0].total_power_W * 1e-13, rel=1e-9)
    assert "ZERO_POWER" not in [i.code.value for i in stated.issues]
    assert stated.manifest.numerical_tolerances["power_epsilon_W"] == 1e-20
    assert not unrecorded_settings(stated)


# ------------------------------------------------------------------------------------------------ the recording rule (C4)


def test_every_setting_is_recorded_with_the_run():
    run = run_optical_project(_project())
    recorded = run.result.manifest.numerical_tolerances
    assert set(SETTINGS) <= set(recorded)
    assert all(recorded[name] == float(entry.default) for name, entry in SETTINGS.items())
    assert not unrecorded_settings(run.result)


def test_every_setting_a_result_names_is_a_key_of_its_run_level_record():
    run = run_optical_project(_project())
    (plane,) = run.result.observations
    entry = ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(
        parameter="far_field_half_angle_rad", unit="rad", maximum=0.4, observed=0.35, within=True,
        setting="gaussian_scope_divergence_rad", at=("screen",)),))
    named = plane.model_copy(update={"value_basis": {"total_power_W": entry}})
    result = run.result.model_copy(update={"observations": (named,)})
    assert named_settings(result) == {"gaussian_scope_divergence_rad"}
    assert unrecorded_settings(result) == ()
    stray = ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(
        parameter="x", unit="1", maximum=1.0, observed=0.5, within=True, setting="not_a_setting"),))
    wrong = run.result.model_copy(update={"observations": (plane.model_copy(update={"value_basis": {"total_power_W": stray}}),)})
    assert unrecorded_settings(wrong) == ("not_a_setting",)
    assert unrecorded_settings(wrong.model_copy(update={"manifest": None})) == ("not_a_setting",)


@pytest.mark.parametrize("fixture", ["keplerian_expander", "trace_partial_beams", "trace_return_chain", "hwp_pbs"])
def test_the_fixture_scenes_name_only_recorded_settings(fixture):
    from fixture_scenes import FIXTURES

    from os_optics import validate

    result = validate(FIXTURES[fixture]())
    assert not unrecorded_settings(result)


def test_the_entries_of_planes_and_targets_name_recorded_settings():
    """The first results that store entries with a setting: a known zero whose truncation a target states
    (``negligible_power_fraction``) and a lit plane with an open upper end. Every name is a key of the run's record."""
    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot, Target
    from os_optics.settings_registry import named_settings

    from helpers import run

    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, power_W=1e-3, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lenses = [B.singlet_lens(f"L{i}", r1_mm=60.0, r2_mm=float("inf"), thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7",
                             position=(20.0 + 25.0 * i, 0.0, 0.0)) for i in range(2)]
    planes = (B.observation_plane("ON", position=(120.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0),
              B.observation_plane("SIDE", position=(30.0, 40.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=2.0, absorbing=True))
    target = Target(target_id="side", metric="power_W", observation_id="SIDE", maximum=1e-6)
    result = run(ProjectSnapshot(components=(laser, *lenses), observations=planes, targets=(target,)))
    assert "negligible_power_fraction" in named_settings(result)
    assert not unrecorded_settings(result)


# ------------------------------------------------------------------------------------------------ pupil-sampling ceiling


def test_a_large_pupil_sampling_is_accepted_and_bounded_by_the_run_monitor():
    big = {**_project(), "compute_budget": {"pupil_ring_count": 128, "pupil_points_per_ring": 512}}
    lowered, default = lower_project(big), lower_project(_project())
    small_estimate = estimate_run_resources(default.request)
    big_estimate = estimate_run_resources(lowered.request)
    assert big_estimate["rays_per_beam"] == 128 * 512 + 1
    # the run files grow with the rays per beam (the ray data in proportion, the rest per beam)
    assert big_estimate["artifact_bytes"] > small_estimate["artifact_bytes"] * 100
    monitor = RunMonitor(max_artifact_bytes=small_estimate["artifact_bytes"] * 10)
    monitor.preflight(small_estimate)
    with pytest.raises(RunStopped) as stopped:
        monitor.preflight(big_estimate)
    assert stopped.value.cause == STOP_ARTIFACTS
    assert math.isfinite(big_estimate["artifact_bytes"])
