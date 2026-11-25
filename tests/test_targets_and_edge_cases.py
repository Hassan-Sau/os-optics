"""Target evaluation semantics (tolerance rules, selection ambiguity, units, layout / control metrics) and tracer edge cases
(medium mismatch, coating override, Fresnel branches, path limits) plus in-process CLI calls."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout

import pytest
from pydantic import ValidationError

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, validate
from os_optics import builders as B
from os_optics.cli import main as cli_main
from os_optics.contracts import (
    ActuatorSpec, CircularAperture, ConstantCoating, ConstantIndexMaterial, ControlSpec, EngineeringComponentSnapshot, JonesVector,
    PlaneSurface, ProjectSnapshot, RefractInteraction, SensorSpec, Target,
)

from helpers import issue_codes, one_arrival, run
from generate_fixtures import REQ_DIR

INV = 1.0 / math.sqrt(2.0)


def _line_project(targets=(), power=2e-3):
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=power),),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
        targets=targets,
    )


def _status(target: Target, project=None) -> str:
    proj = (project or _line_project()).model_copy(update={"targets": (target,)})
    return run(proj).targets[0].status.value


# ---- comparison rules -------------------------------------------------------------------------------------------------------------
def test_tolerance_is_the_larger_of_absolute_and_relative():
    p = 2e-3
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", target_value=p * 1.05, tolerance_abs=1e-9, tolerance_rel=0.1)) == "PASS"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", target_value=p * 1.05, tolerance_abs=1e-9, tolerance_rel=0.01)) == "FAIL"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", target_value=p * 1.05, tolerance_abs=1e-3)) == "PASS"


def test_target_result_carries_value_error_tolerance_unit_and_message():
    res = run(_line_project((Target(target_id="t", metric="power_W", observation_id="d", device_id="L", target_value=1.5e-3, tolerance_abs=1e-4,
                                    source_revision="rev-7", unit="W"),)))
    t = res.targets[0]
    assert (t.device_id, t.observation_id, t.metric, t.unit, t.source_revision) == ("L", "d", "power_W", "W", "rev-7")
    assert t.value == pytest.approx(2e-3) and t.error == pytest.approx(5e-4) and t.tolerance == 1e-4 and t.target_value == 1.5e-3
    assert t.status.value == "FAIL" and t.message == "|error| 0.0005 W exceeds the tolerance 0.0001 W" and "TARGET_NOT_MET" in t.issue_codes


def test_min_max_bounds_and_combined_with_target_value():
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", minimum=1e-3, maximum=3e-3)) == "PASS"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", minimum=2.5e-3)) == "FAIL"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", maximum=1e-3)) == "FAIL"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", target_value=2e-3, tolerance_abs=1e-6, maximum=1e-3)) == "FAIL"


def test_unit_mismatch_is_unknown_not_a_silent_conversion():
    """A target that states its bounds in another unit than its metric's: the measured value is reported in the metric's
    unit (the same number as with the right unit), the bounds are echoed as stated and not compared, nothing is
    converted -- a 5 mW minimum would otherwise read as 5 W, or pass against a value in W."""
    res = run(_line_project((Target(target_id="t", metric="power_W", observation_id="d", minimum=5.0, unit="mW"),
                             Target(target_id="w", metric="power_W", observation_id="d", minimum=0.0, unit="W"))))
    t, same = res.targets
    assert t.status.value == "UNKNOWN" and t.assessment.value == "INDETERMINATE" and "unit mismatch" in t.message
    assert t.issue_codes == ("INVALID_INPUT",) and t.error is None
    assert same.status.value == "PASS" and same.value > 0.0
    assert (t.value, t.unit, t.minimum) == (same.value, "W", 5.0)
    assert "'W'" in t.message and "'mW'" in t.message


def test_warning_level_measurement_is_not_upgraded_to_pass():
    """A valid comparison on a beam whose Gaussian state carries a WARNING stays WARNING."""
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=0.7e-3, position=(0, 0, 20), direction=(1, 0, 0)),),
        observations=(B.observation_plane("d", position=(1, 0, 20), normal=(1, 0, 0), radius_mm=5.0),),
        targets=(Target(target_id="t", metric="beam_radius_x_mm", observation_id="d", minimum=0.0),),
    )
    assert run(proj).targets[0].status.value == "WARNING"


def _near_high_na(targets) -> ProjectSnapshot:
    """A 0.7 um waist at 405 nm (far-field half angle 0.18 rad, between the warning and the scope limits of the Gaussian
    model: its state is valid with a WARNING)."""
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=0.7e-3, position=(0, 0, 20), direction=(1, 0, 0)),),
        observations=(B.observation_plane("d", position=(1, 0, 20), normal=(1, 0, 0), radius_mm=5.0, acceptance_half_angle_deg=30.0),),
        targets=tuple(targets))


def test_a_warning_of_the_gaussian_state_marks_the_metrics_that_read_it_and_only_them():
    """Every single-beam metric on a beam whose Gaussian state is valid with a WARNING: a metric computed from that state
    (the registry's needs-Gaussian flag) is WARNING with the state's code; a metric read from the chief ray, the pupil rays,
    the polarization or the path lengths is PASS without it."""
    from os_optics.validation.metrics import OBSERVATION_METRICS, POWER_SUM_METRICS

    metrics = [m for m in OBSERVATION_METRICS if m not in POWER_SUM_METRICS]
    result = run(_near_high_na(Target(target_id=m, metric=m, observation_id="d", minimum=-1e9) for m in metrics))
    state = result.observations[0].arrivals[0].beam.gaussian
    assert (state.valid, state.status.value) == (True, "WARNING")
    assert 0.10 < state.far_field_half_angle_x_rad < 0.30
    codes = (state.reason,) if state.reason else ()
    by_id = {t.target_id: t for t in result.targets}
    read, not_read = [], []
    for metric in metrics:
        target = by_id[metric]
        if target.value is None:  # (a first-order image metric a bare beam does not have)
            assert not OBSERVATION_METRICS[metric][2], metric
            continue
        if OBSERVATION_METRICS[metric][2]:
            read.append(metric)
            assert (target.status.value, target.issue_codes) == ("WARNING", codes), metric
        else:
            not_read.append(metric)
            assert (target.status.value, target.issue_codes) == ("PASS", ()), metric
    assert {"beam_radius_x_mm", "waist_radius_x_mm", "divergence_half_angle_x_rad", "capture_margin_mm", "active_area_power_fraction"} <= set(read)
    assert {"aoi_deg", "position_u_mm", "geometric_path_length_mm", "optical_path_length_mm", "phase_rad", "wavelength_nm", "spot_rms_mm",
            "direction_cosine_u", "stokes_s1", "acceptance_margin_deg", "group_delay_ps"} <= set(not_read)


# ---- selection of beams ----------------------------------------------------------------------------------------------------------
def _two_source_project(targets):
    return ProjectSnapshot(
        components=(B.laser("A", wavelength_nm=405.0, waist_radius_mm=0.5, position=(0, 5, 20), direction=(1, 0, 0), power_W=1e-3),
                    B.laser("B", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, -5, 20), direction=(1, 0, 0), power_W=3e-3)),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=50.0),), targets=targets)


def test_ambiguous_selection_is_unknown_and_explicit_selectors_resolve_it():
    assert _status(Target(target_id="t", metric="beam_radius_mm", observation_id="d", minimum=0.0), _two_source_project(())) == "UNKNOWN"
    ok = run(_two_source_project((
        Target(target_id="a", metric="power_W", observation_id="d", source_id="A", target_value=1e-3, tolerance_rel=1e-9),
        Target(target_id="b", metric="power_W", observation_id="d", wavelength_nm=532.0, target_value=3e-3, tolerance_rel=1e-9),
        Target(target_id="sum", metric="power_W", observation_id="d", target_value=4e-3, tolerance_rel=1e-9),  # power_W sums beams
        Target(target_id="pid", metric="wavelength_nm", observation_id="d", parameters={"path_id": "B#0"}, target_value=532.0, tolerance_abs=1e-9),
    )))
    assert [t.status.value for t in ok.targets] == ["PASS"] * 4


def test_target_references_are_validated_structurally():
    assert _status(Target(target_id="t", metric="power_W", observation_id="nope", minimum=0.0)) == "UNKNOWN"
    # a metric that reads a plane names it: the Target contract refuses one that does not, before any run
    for metric in ("power_W", "fringe_period_nm", "centroid_separation_mm", "detector_snr"):
        with pytest.raises(ValidationError, match="set observation_id"):
            Target(target_id="t", metric=metric, minimum=0.0)
    # a source the run does not have selects nothing: not a value, and not "not reached"
    assert _status(Target(target_id="t", metric="power_W", observation_id="d", source_id="ghost", minimum=0.0)) == "UNKNOWN"


def test_pair_metric_selection_and_exposure_parameter_validation():
    from test_golden_interference import _interference_project

    def status(**kw):
        proj, *_ = _interference_project(targets=(Target(target_id="t", observation_id="sub", minimum=0.0, **kw),))
        return run(proj).targets[0]

    assert status(metric="fringe_period_nm", source_id="L", source_id_b="L").status.value == "PASS"
    assert status(metric="fringe_period_nm", parameters={"path_id_a": "L#0", "path_id_b": "L#0.0"}).status.value == "PASS"
    # a path, sources or a wavelength the run does not have select no pair: UNKNOWN, TARGET_SELECTION_EMPTY
    for selected in ({"parameters": {"path_id_a": "L#0", "path_id_b": "zzz"}}, {"source_id": "X", "source_id_b": "Y"},
                     {"wavelength_nm": 500.0}):
        result = status(metric="fringe_period_nm", **selected)
        assert (result.status.value, result.value, result.issue_codes) == ("UNKNOWN", None, ("TARGET_SELECTION_EMPTY",)), selected
    # the exposure time needs its dose, greater than 0: the Target contract refuses the target without it
    with pytest.raises(ValidationError, match="needs parameter 'dose_mj_per_cm2'"):
        status(metric="estimated_exposure_time_s")
    for dose in (-1.0, 0.0):
        with pytest.raises(ValidationError, match="must be greater than 0"):
            status(metric="estimated_exposure_time_s", parameters={"dose_mj_per_cm2": dose})
    assert status(metric="estimated_exposure_time_s", parameters={"dose_mj_per_cm2": 100.0}).value > 0.0


# ---- layout / control metrics -----------------------------------------------------------------------------------------------------
def test_layout_metrics_and_stage_not_run_is_unknown():
    a = EngineeringComponentSnapshot(instance_id="A", envelope=B.rect_envelope(10, 10, 20))
    b = EngineeringComponentSnapshot(instance_id="B", pose=B.Pose3D(position_mm=(25.0, 0.0, 0.0)), envelope=B.rect_envelope(10, 10, 20))
    proj = ProjectSnapshot(components=(a, b), targets=(
        Target(target_id="c", metric="layout_min_clearance_mm", target_value=15.0, tolerance_abs=1e-9),
        Target(target_id="v", metric="layout_violation_count", target_value=0.0, tolerance_abs=0.0)))
    res = run(proj, layout=True)
    assert [t.status.value for t in res.targets] == ["PASS", "PASS"] and res.targets[0].unit == "mm"
    off = run(proj, layout=False)
    assert [t.status.value for t in off.targets] == ["UNKNOWN", "UNKNOWN"] and "did not run" in off.targets[0].message


def test_control_metrics_via_targets():
    from test_control import _mirror_project

    proj = _mirror_project()
    targets = (Target(target_id="rank", metric="numerical_rank", target_value=2.0, tolerance_abs=0.0),
               Target(target_id="cond", metric="jacobian_condition_number", maximum=100.0),
               Target(target_id="rec", metric="recoverability_fraction", minimum=1.0),
               Target(target_id="mc", metric="monte_carlo_recovered_fraction", minimum=0.5))
    res = run(proj.model_copy(update={"targets": targets}), control=True)
    by = {t.target_id: t.status.value for t in res.targets}
    assert by == {"rank": "PASS", "cond": "PASS", "rec": "UNKNOWN", "mc": "UNKNOWN"}  # no recoverability / Monte Carlo configured
    no_control = run(proj.model_copy(update={"targets": targets[:1]}), control=False)
    assert no_control.targets[0].status.value == "UNKNOWN"


# ---- tracer edge cases ---------------------------------------------------------------------------------------------------------------
def test_medium_mismatch_from_inconsistent_solids_is_reported():
    """Two windows whose facing media do not match: the ray leaves glass A into 'air' but the next surface expects 'N-BK7'."""
    a = B.window("A", thickness_mm=2.0, clear_radius_mm=10.0, material="N-BK7", position=(20, 0, 20))
    bad = EngineeringComponentSnapshot(
        instance_id="Bad", pose=B.Pose3D.from_axis_direction((40, 0, 20), (1, 0, 0)),
        surfaces=(PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=(0, 0, -1), aperture=CircularAperture(radius_mm=10.0),
                               medium_positive="fused_silica", medium_negative="N-BK7", interaction=RefractInteraction()),))
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)), a, bad),
                           observations=(B.observation_plane("d", position=(80, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    res = run(proj)
    ctx = next(i for i in res.issues if i.code.value == "MEDIUM_MISMATCH").numerical_context
    assert ctx["tracked_medium"] == "air" and ctx["expected_medium"] == "fused_silica" and res.paths[0].termination == "MEDIUM_MISMATCH"


def test_observation_medium_mismatch_is_a_warning():
    proj = _line_project().model_copy(update={"observations": (B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=20.0,
                                                                                   medium="fused_silica"),)})
    res = run(proj)
    assert "MEDIUM_MISMATCH" in issue_codes(res) and res.status.value == "WARNING" and res.observations[0].arrivals


def test_refraction_coating_replaces_bare_fresnel_power():
    """An AR-coated window (T = 0.995 per surface) instead of the bare-glass Fresnel loss."""
    coat = ConstantCoating(reflectance_s=0.005, reflectance_p=0.005, transmittance_s=0.995, transmittance_p=0.995)
    # Ideal test-only substrate makes the *coating* transmission the sole power loss in this exact comparison.
    glass = ConstantIndexMaterial(material_id="glass", valid_wavelength_nm=(300.0, 900.0), refractive_index=1.5,
                                  absorption_coefficient_per_mm=0.0)
    def surf(name, z, nrm, pos_medium, neg_medium):
        return PlaneSurface(surface_id=name, origin_mm=(0, 0, z), normal=nrm, aperture=CircularAperture(radius_mm=10.0),
                            medium_positive=pos_medium, medium_negative=neg_medium, interaction=RefractInteraction(coating=coat))
    win = EngineeringComponentSnapshot(instance_id="W", pose=B.Pose3D.from_axis_direction((30, 0, 20), (1, 0, 0)),
                                       surfaces=(surf("f", -1.0, (0, 0, -1), "air", "glass"), surf("b", 1.0, (0, 0, 1), "air", "glass")))
    proj = ProjectSnapshot(materials=(glass,), components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3), win),
                           observations=(B.observation_plane("d", position=(80, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    assert one_arrival(run(proj), "d").beam.power_W == pytest.approx(1e-3 * 0.995**2, rel=1e-12)


def test_coating_out_of_range_on_refraction_and_max_paths_limit():
    coat = ConstantCoating(reflectance_s=0.005, reflectance_p=0.005, transmittance_s=0.995, transmittance_p=0.995, valid_wavelength_nm=(400.0, 500.0))
    win = EngineeringComponentSnapshot(instance_id="W", pose=B.Pose3D.from_axis_direction((30, 0, 20), (1, 0, 0)), surfaces=(
        PlaneSurface(surface_id="f", origin_mm=(0, 0, 0), normal=(0, 0, -1), aperture=CircularAperture(radius_mm=10.0), medium_positive="air",
                     medium_negative="N-BK7", interaction=RefractInteraction(coating=coat)),))
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)), win))
    assert "COATING_OUT_OF_RANGE" in issue_codes(run(proj))
    # branching guard: a comb of 5 splitters with max_paths = 3 stops creating children and says so
    comps = [B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(-10, 0, 20), direction=(1, 0, 0))]
    comps += [B.npbs(f"S{i}", position=(20.0 * i, 0, 20), normal=(-INV, INV, 0)) for i in range(5)]
    res = run(ProjectSnapshot(components=tuple(comps)), numerical=NumericalConfig(max_paths=3))
    assert "MAX_PATHS" in issue_codes(res) and len(res.paths) <= 3


def test_fresnel_reflection_branch_of_a_curved_surface_is_traced():
    glass_lens = B.singlet_lens("Lens", r1_mm=80.0, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=8.0, material="N-BK7", position=(30, 0, 20))
    lens = glass_lens.model_copy(update={"surfaces": tuple(
        s.model_copy(update={"interaction": RefractInteraction(trace_fresnel_reflection=True)}) for s in glass_lens.surfaces)})
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3), lens),
                           observations=(B.observation_plane("R", position=(-20, 0, 20), normal=(1, 0, 0), radius_mm=30.0),
                                         B.observation_plane("T", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=30.0)))
    res = run(proj)
    kinds = {p.branch_kind for p in res.paths if p.branch_kind}
    assert kinds == {"fresnel_reflection"}
    reflected = [a for a in res.observations[0].arrivals if a.beam.direction[0] < 0]
    assert reflected and reflected[0].beam.power_W == pytest.approx(1e-3 * 0.0426, rel=0.05)  # ~4.3 % bare-BK7 reflection at 532 nm
    assert reflected[0].beam.gaussian.valid  # a diverging beam from the convex front surface (reflected Gaussian)


def test_absorbing_detector_and_beam_dump_terminate_paths():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.beam_dump("Dump", position=(50, 0, 20), axis=(1, 0, 0))),
        observations=(B.observation_plane("beyond", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    res = run(proj)
    assert res.paths[0].termination == "ABSORBED" and res.observations[0].arrivals == ()
    det = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),),
                          observations=(B.observation_plane("det", position=(50, 0, 20), normal=(1, 0, 0), radius_mm=10.0, absorbing=True),
                                        B.observation_plane("behind", position=(80, 0, 20), normal=(1, 0, 0), radius_mm=10.0)))
    r2 = run(det)
    assert r2.paths[0].termination == "DETECTED" and r2.observations[1].arrivals == ()


# ---- in-process CLI (coverage of the same code the subprocess tests execute) ---------------------------------------------------------
def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli_main(list(argv))
        except SystemExit as exc:  # argparse --version
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def test_cli_in_process(tmp_path):
    code, out, _ = _cli("validate", str(REQ_DIR / "hwp_pbs.json"))
    assert code == 0 and json.loads(out)["status"] == "PASS"
    code, out, err = _cli("validate", str(tmp_path / "missing.json"))
    assert code == 64 and "cannot read" in err
    target = tmp_path / "o.json"
    code, _, _ = _cli("validate", str(REQ_DIR / "layout_example.json"), "-o", str(target), "--pretty", "--status-contract", "aggregate")
    # its layout finding is the layout's own outcome (no layout target is declared): the aggregate passes
    written = target.read_text(encoding="utf-8")
    assert code == 0 and "\n " in written and json.loads(written)["layout"]["status"] == "FAIL"
    assert _cli("validate", str(REQ_DIR / "out_of_scope_targets.json"), "--status-contract", "aggregate")[0] == 3
    assert _cli("validate", str(REQ_DIR / "out_of_scope_targets.json"))[0] == 2  # (the default: the targets are undetermined)
    code, out, _ = _cli("schema", "request")
    assert code == 0 and json.loads(out)["title"] == "ValidationRequest"
    left = tmp_path / "l.json"
    left.write_text(_cli("validate", str(REQ_DIR / "hwp_pbs.json"))[1], encoding="utf-8")
    code, out, _ = _cli("compare", str(left), str(left))
    assert code == 0 and "equivalent" in out
    code, _, err = _cli("compare", str(left), str(tmp_path / "nope.json"))
    assert code == 64
    assert _cli("--version")[0] == 0
