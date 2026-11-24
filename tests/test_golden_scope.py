"""Golden case 16 (+ related scope tests): unsupported wave-optics / high-NA requests must return OUT_OF_MODEL_SCOPE or UNKNOWN,
never PASS; Gaussian applicability limits invalidate the state instead of returning pseudo-exact numbers."""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.validation.scope import OUT_OF_SCOPE_METRICS

from helpers import issue_codes, one_arrival, run

pytestmark = pytest.mark.golden


def _basic_project(targets=(), **laser_kw):
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=laser_kw.pop("waist_radius_mm", 1.0), position=(0, 0, 20),
                            direction=(1, 0, 0), **laser_kw),),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
        targets=targets,
    )


@pytest.mark.parametrize("metric", sorted(OUT_OF_SCOPE_METRICS))
def test_unsupported_physics_never_passes(metric):
    """Every entry of the out-of-scope registry returns OUT_OF_MODEL_SCOPE with an explanation; even a target that a naive
    implementation would satisfy trivially (minimum = -inf-like) must not turn into PASS."""
    t = Target(target_id="t", metric=metric, observation_id="d", minimum=-1e30)
    res = run(_basic_project(targets=(t,)))
    tr = res.targets[0]
    assert tr.status.value == "OUT_OF_MODEL_SCOPE"
    assert tr.value is None and tr.message == OUT_OF_SCOPE_METRICS[metric]
    assert "OUT_OF_MODEL_SCOPE" in tr.issue_codes
    assert res.status.value == "OUT_OF_MODEL_SCOPE"


def test_unknown_metric_is_unknown_not_pass():
    res = run(_basic_project(targets=(Target(target_id="t", metric="brightness_of_the_moon", observation_id="d", minimum=0.0),)))
    assert res.targets[0].status.value == "UNKNOWN" and "UNSUPPORTED_METRIC" in res.targets[0].issue_codes
    assert res.status.value == "UNKNOWN"


def test_result_with_only_supported_pass_is_pass_but_scope_targets_dominate():
    good = Target(target_id="ok", metric="power_W", observation_id="d", target_value=1e-3, tolerance_rel=1e-9)
    bad = Target(target_id="bad", metric="fresnel_propagation", observation_id="d", minimum=0.0)
    res = run(_basic_project(targets=(good, bad)))
    by = {t.target_id: t.status.value for t in res.targets}
    assert by == {"ok": "PASS", "bad": "OUT_OF_MODEL_SCOPE"}
    assert res.status.value == "OUT_OF_MODEL_SCOPE"


def test_manifest_declares_model_scope_and_unsupported_list():
    res = run(_basic_project())
    scope = res.manifest.model_scope
    assert any(s.startswith("gaussian_beam") for s in scope)
    for metric in OUT_OF_SCOPE_METRICS:
        assert f"unsupported:{metric}" in scope


@pytest.mark.parametrize("w0_um,expected_status", [(3.0, "PASS"), (0.7, "WARNING"), (0.3, "OUT_OF_MODEL_SCOPE")])
def test_high_na_gaussian_source(w0_um, expected_status):
    """theta = lambda/(pi w0): 3 um -> 0.043 rad (ok), 0.7 um -> 0.184 rad (warn > 0.10), 0.3 um -> 0.43 rad (> 0.30: refused)."""
    w0 = w0_um * 1e-3
    res = run(_basic_project(waist_radius_mm=w0, targets=(Target(target_id="t", metric="beam_radius_x_mm", observation_id="d", minimum=0.0),)))
    theta = 405e-6 / (math.pi * w0)
    g = one_arrival(res, "d").beam.gaussian
    assert g.status.value == expected_status
    if expected_status == "OUT_OF_MODEL_SCOPE":
        assert theta > 0.30 and not g.valid and g.reason == "GAUSSIAN_HIGH_NA"
        assert res.targets[0].status.value == "OUT_OF_MODEL_SCOPE" and res.targets[0].value is None
        assert "GAUSSIAN_HIGH_NA" in issue_codes(res)
        # (the reason is structured: the failed check, on the state and on the target; no number goes with it)
        (check,) = g.validity
        assert (check.code, check.parameter, check.setting, check.within, check.at) == (
            "GAUSSIAN_HIGH_NA", "far_field_half_angle_rad", "gaussian_scope_divergence_rad", False, ("L",))
        assert check.observed == pytest.approx(theta, rel=1e-9) and check.maximum == 0.30
        assert res.targets[0].validity == (check,) and res.targets[0].value_basis is None
        assert g.beam_radius_x_mm is None and g.q_x_mm is None and g.value_basis is None
    else:
        assert g.valid
        assert res.targets[0].status.value == ("PASS" if expected_status == "PASS" else "WARNING")


def test_strong_focusing_lens_leaves_paraxial_scope():
    """w = 2 mm focused by f = 5 mm -> w0' = lambda f/(pi w) = 0.32 um, divergence w/f = 0.4 rad: high-NA focus -> out of scope."""
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=2.0, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.thin_lens("Lens", focal_length_mm=5.0, clear_radius_mm=6.0, position=(10, 0, 20), axis=(1, 0, 0))),
        observations=(B.observation_plane("d", position=(15, 0, 20), normal=(1, 0, 0), radius_mm=5.0),),
        targets=(Target(target_id="w", metric="waist_radius_x_mm", observation_id="d", target_value=3.2e-4, tolerance_rel=0.1),),
    )
    res = run(proj)
    assert res.targets[0].status.value == "OUT_OF_MODEL_SCOPE" and res.targets[0].value is None
    assert one_arrival(res, "d").beam.gaussian.reason == "GAUSSIAN_HIGH_NA"
    (check,) = res.targets[0].validity
    assert (check.code, check.setting, check.within, check.at) == ("GAUSSIAN_HIGH_NA", "gaussian_scope_divergence_rad", False, ("Lens/lens",))
    assert check.observed > 0.30


def test_invalid_gaussian_state_is_never_resurrected_downstream():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=2.0, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.thin_lens("L1", focal_length_mm=5.0, clear_radius_mm=6.0, position=(10, 0, 20), axis=(1, 0, 0)),
                    B.thin_lens("L2", focal_length_mm=5.0, clear_radius_mm=6.0, position=(20, 0, 20), axis=(1, 0, 0))),
        observations=(B.observation_plane("d", position=(200, 0, 20), normal=(1, 0, 0), radius_mm=100.0),),
    )
    g = one_arrival(run(proj), "d").beam.gaussian
    assert not g.valid and g.reason == "GAUSSIAN_HIGH_NA" and g.q_x_mm is None
    # (the second lens would collimate the beam again: the state still names the first lens's failed check, and no number)
    (check,) = g.validity
    assert (check.code, check.at, check.within) == ("GAUSSIAN_HIGH_NA", ("L1/lens",), False)
    assert g.beam_radius_x_mm is None and g.far_field_half_angle_x_rad is None and g.value_basis is None


def test_surface_too_strong_for_beam_size():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=3.0, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.spherical_mirror("M", position=(50, 0, 20), normal=(-1, 0, 0), radius_of_curvature_mm=10.0, clear_radius_mm=9.0)),
        observations=(B.observation_plane("d", position=(20, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
    )
    res = run(proj)
    assert "GAUSSIAN_SURFACE_TOO_STRONG" in issue_codes(res) and res.status.value == "OUT_OF_MODEL_SCOPE"


def test_large_aoi_on_curved_surface_is_out_of_scope():
    a = math.radians(65.0)
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.spherical_mirror("M", position=(50, 0, 20), normal=(-math.cos(a), math.sin(a), 0), radius_of_curvature_mm=300.0,
                                       clear_radius_mm=10.0)),
        observations=(B.observation_plane("d", position=(50, 60, 20), normal=(0, -1, 0), radius_mm=1000.0),),
    )
    res = run(proj)
    assert "GAUSSIAN_LARGE_AOI" in issue_codes(res)
    # the geometric ray result is still delivered (only the Gaussian state is refused)
    assert res.observations[0].arrivals
