"""Wrapped-phase (circular angular) comparison policy for the cross-runtime comparator.

The two normatively-wrapped result fields — ``observations[*].arrivals[*].beam.phase_rad`` and
``interference[*].phase_difference_rad`` — are angles modulo 2π wrapped to [-π, π). Comparing them with the generic
``|a-b| <= abs + rel*max(|a|,|b|)`` rule is wrong near the ±π seam: ``+π-ε`` and ``-π+ε`` are the same physical angle
but differ by ~2π. These fields are compared with the shortest circular distance instead, reusing the SAME tolerance
values on the natural 2π angular scale (``abs_tol + rel_tol * 2π``). No new NumericalConfig field is introduced.
"""

from __future__ import annotations

import math

import pytest

from os_optics.compare import _angular_distance_rad, _is_wrapped_phase_path, compare_results
from os_optics.config import NumericalConfig

CFG = NumericalConfig()  # cross_runtime_abs_tol=1e-12, cross_runtime_rel_tol=1e-9


def _beam_phase(v: float) -> dict:
    return {"observations": [{"arrivals": [{"beam": {"phase_rad": v}}]}]}


def _interf_phase(v: float) -> dict:
    return {"interference": [{"phase_difference_rad": v}]}


# ---- 1. ordinary floats keep generic tolerance -----------------------------------------------------------
def test_normal_float_below_generic_tolerance_passes():
    # just below abs + rel*max(|a|,|b|) = 1e-12 + 1e-9*1.0 ≈ 1.0000e-9
    assert compare_results({"wavelength_nm": 1.0}, {"wavelength_nm": 1.0 + 5e-10}, CFG) == []


def test_normal_float_above_generic_tolerance_fails():
    # just above the generic tolerance → still a difference
    assert compare_results({"wavelength_nm": 1.0}, {"wavelength_nm": 1.0 + 1.8e-9}, CFG) != []


# ---- 2. observed ~1.8e-9 rad drift passes ----------------------------------------------------------------
def test_phase_drift_around_1p8e9_rad_passes():
    assert compare_results(_beam_phase(0.5), _beam_phase(0.5 + 1.8e-9), CFG) == []
    assert compare_results(_interf_phase(0.5), _interf_phase(0.5 + 1.8e-9), CFG) == []


# ---- 3. real ~1e-7 rad discrepancy still fails -----------------------------------------------------------
def test_phase_discrepancy_around_1e7_rad_fails():
    assert compare_results(_beam_phase(0.5), _beam_phase(0.5 + 1e-7), CFG) != []
    assert compare_results(_interf_phase(0.5), _interf_phase(0.5 + 1e-7), CFG) != []


# ---- 4. wrap boundary is compared as shortest circular distance -------------------------------------------
def test_angular_distance_is_shortest_circular_distance():
    eps = 1e-10
    d = _angular_distance_rad(math.pi - eps, -math.pi + eps)
    assert d == pytest.approx(2 * eps, abs=1e-16)
    # the naive |a-b| would be ~2π (this is exactly the failure the policy fixes)
    assert abs((math.pi - eps) - (-math.pi + eps)) == pytest.approx(2 * math.pi, abs=1e-9)


def test_wrap_seam_same_physical_angle_passes():
    eps = 1e-10
    assert compare_results(_beam_phase(math.pi - eps), _beam_phase(-math.pi + eps), CFG) == []
    assert compare_results(_interf_phase(math.pi - eps), _interf_phase(-math.pi + eps), CFG) == []


# ---- 5. non-phase fields do NOT inherit angular tolerance -------------------------------------------------
def test_non_phase_field_does_not_inherit_angular_tolerance():
    # same wrap-seam numeric values (~2π apart) must FAIL for a plain unbounded float
    assert compare_results({"aoi_deg": math.pi - 1e-10}, {"aoi_deg": -math.pi + 1e-10}, CFG) != []
    # same 1.8e-9 drift that a phase field tolerates must still FAIL for a plain float
    assert compare_results({"aoi_deg": 0.5}, {"aoi_deg": 0.5 + 1.8e-9}, CFG) != []


# ---- classifier: only the two wrapped result paths ---------------------------------------------------------
def test_classifier_matches_only_wrapped_phase_paths():
    assert _is_wrapped_phase_path("observations[0].arrivals[0].beam.phase_rad")
    assert _is_wrapped_phase_path("observations[3].arrivals[1].beam.phase_rad")
    assert _is_wrapped_phase_path("interference[0].phase_difference_rad")
    # every other *_rad / angle field must NOT be classified as wrapped
    assert not _is_wrapped_phase_path("phase_difference_rad")  # no dotted context (top-level key)
    assert not _is_wrapped_phase_path("observations[0].arrivals[0].aoi_deg")
    assert not _is_wrapped_phase_path("interference[0].beam_crossing_angle_deg")
    assert not _is_wrapped_phase_path("observations[0].arrivals[0].beam.gaussian.far_field_half_angle_x_rad")
    assert not _is_wrapped_phase_path("manifest.numerical_tolerances.fd_step_rotation_rad")
    assert not _is_wrapped_phase_path("observations[0].arrivals[0].beam.phase_rad_extra")  # not an exact suffix
