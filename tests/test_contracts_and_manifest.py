"""Phase 2: public contracts, error model, canonical serialization and hashing."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

import os_optics.contracts as c
from os_optics.errors import ErrorCode, Status, make_issue, worst
from os_optics.manifest import canonical_json, engine_build_id, hash_exact, hash_quantized, quantize_float


# ---------------------------------------------------------------------------------------- contracts
def test_pose_rejects_nan_inf_and_zero_quaternion():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            c.Pose3D(position_mm=(bad, 0.0, 0.0))
    with pytest.raises(ValidationError):
        c.Pose3D(quaternion_wxyz=(0.0, 0.0, 0.0, 0.0))


def test_pose_lowering_takes_a_quaternion_by_its_direction():
    import numpy as np

    ok = c.Pose3D(quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)).to_pose()
    assert ok.rotation.shape == (3, 3)
    assert np.array_equal(c.Pose3D(quaternion_wxyz=(2.0, 0.0, 0.0, 0.0)).to_pose().rotation, np.eye(3))
    half = math.sqrt(0.5)
    np.testing.assert_allclose(c.Pose3D(quaternion_wxyz=(3.0, 0.0, 0.0, 3.0)).to_pose().rotation,
                               c.Pose3D(quaternion_wxyz=(half, 0.0, 0.0, half)).to_pose().rotation, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        c.Pose3D.model_construct(position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=(0.0, 0.0, 0.0, 0.0)).to_pose()


def test_a_stated_axis_is_stored_as_a_unit_vector_and_kept_bit_for_bit_within_its_tolerance():
    from os_optics.contracts.base import unit_direction
    from os_optics.contracts.optics import PhaseStep
    from os_optics.contracts.polarization import PolarizationBasis

    half = math.sqrt(0.5)
    step = PhaseStep(phase_rad=1.0, edge_normal=(0.7071, 0.7071))
    assert step.edge_normal == pytest.approx((half, half), rel=1e-15)
    assert PhaseStep(phase_rad=1.0, edge_normal=(1.0, 1e-7)).edge_normal == (1.0, 1e-7)  # within 1e-6: as given
    basis = PolarizationBasis(direction=(0.0, 0.0, 2.0), e1=(0.7071, 0.7071, 0.0), e2=(-0.7071, 0.7071, 0.0))
    assert basis.direction == (0.0, 0.0, 1.0) and basis.e1 == pytest.approx((half, half, 0.0), rel=1e-15)
    for zero in ((0.0, 0.0), (0.0, 0.0, 0.0)):
        with pytest.raises(ValueError, match="non-zero"):
            unit_direction(zero, 1e-6, "axis")
    with pytest.raises(ValidationError, match="non-zero"):
        PhaseStep(phase_rad=1.0, edge_normal=(0.0, 0.0))


def test_contracts_are_frozen_and_forbid_extra_fields():
    p = c.Pose3D()
    with pytest.raises(ValidationError):
        p.position_mm = (1.0, 0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        c.Pose3D(position_mm=(0, 0, 0), bogus=1)  # type: ignore[call-arg]


def test_euler_and_yaw_helpers_are_right_handed():
    import numpy as np

    pose = c.Pose3D.from_yaw_deg((0, 0, 0), 90.0).to_pose()
    np.testing.assert_allclose(pose.rotation @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    p2 = c.Pose3D.from_euler_zyx_deg((0, 0, 0), yaw_deg=90.0, pitch_deg=90.0).to_pose()
    # R = Rz(90) Ry(90): x -> (Ry) -> -z -> (Rz) -> -z ; y -> y -> x_hat rotated: Rz(90) y = -x
    np.testing.assert_allclose(p2.rotation @ [1, 0, 0], [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(p2.rotation @ [0, 1, 0], [-1, 0, 0], atol=1e-15)


def test_jones_helpers_normalized_and_handedness():
    for jv in (c.JonesVector.linear(37.0), c.JonesVector.right_circular(), c.JonesVector.left_circular(), c.JonesVector.elliptical(20.0, 15.0)):
        assert jv.norm == pytest.approx(1.0, abs=1e-15)
    rc = c.JonesVector.right_circular()
    assert rc.ey[0] == 0.0 and rc.ey[1] == pytest.approx(-1 / math.sqrt(2))
    # elliptical chi=-45 deg equals right circular up to global phase magnitude structure
    e = c.JonesVector.elliptical(0.0, -45.0)
    assert e.ex[0] == pytest.approx(1 / math.sqrt(2)) and e.ey[1] == pytest.approx(-1 / math.sqrt(2))


def test_polarization_basis_validation():
    c.PolarizationBasis(direction=(0, 0, 1), e1=(1, 0, 0), e2=(0, 1, 0))
    with pytest.raises(ValidationError):
        c.PolarizationBasis(direction=(0, 0, 1), e1=(1, 0, 0), e2=(1, 0, 0))
    assert c.PolarizationBasis(direction=(0, 0, 2), e1=(1, 0, 0), e2=(0, 1, 0)).direction == (0.0, 0.0, 1.0)  # normalized
    with pytest.raises(ValidationError):
        c.PolarizationBasis(direction=(0, 0, 0), e1=(1, 0, 0), e2=(0, 1, 0))


def test_material_and_coating_validation():
    with pytest.raises(ValidationError):
        c.ConstantIndexMaterial(material_id="x", valid_wavelength_nm=(500.0, 400.0), refractive_index=1.5)
    with pytest.raises(ValidationError):
        c.TabulatedIndexMaterial(material_id="x", valid_wavelength_nm=(400.0, 700.0), wavelength_nm=(500.0, 600.0), refractive_index=(1.5, 1.4))
    with pytest.raises(ValidationError):
        c.TabulatedIndexMaterial(material_id="x", valid_wavelength_nm=(500.0, 600.0), wavelength_nm=(600.0, 500.0), refractive_index=(1.5, 1.4))
    with pytest.raises(ValidationError):
        c.ConstantCoating(reflectance_s=0.7, transmittance_s=0.7)  # violates energy conservation
    with pytest.raises(ValidationError):
        c.TabulatedCoating(
            wavelength_nm=(400.0, 500.0), aoi_deg=(0.0, 45.0),
            reflectance_s=((0, 0), (0, 0)), reflectance_p=((0, 0), (0, 0)),
            transmittance_s=((1, 1), (1, 1)), transmittance_p=((1, 1), (1, 2)),  # 2 > 1
        )


def test_surface_contract_geometry_specific_rules():
    with pytest.raises(ValidationError):
        c.SphereSurface(surface_id="s", center_mm=(0, 0, 0), radius_mm=5.0, cap_axis=(0, 0, 1), aperture_radius_mm=6.0,
                        interaction=c.RefractInteraction())
    with pytest.raises(ValidationError):  # thin lens is only a plane interaction
        c.SphereSurface(surface_id="s", center_mm=(0, 0, 0), radius_mm=5.0, cap_axis=(0, 0, 1), aperture_radius_mm=2.0,
                        interaction=c.ThinLensInteraction(focal_length_mm=10.0))
    with pytest.raises(ValidationError):
        c.ThinLensInteraction(focal_length_mm=0.0)
    with pytest.raises(ValidationError):
        c.CircularAperture(radius_mm=1.0, obstruction_radius_mm=1.0)


def test_source_requires_exactly_one_wavelength_definition():
    base = dict(source_id="L", power_W=1e-3, waist_radius_x_mm=0.5, waist_radius_y_mm=0.5)
    c.SourceSpec(wavelength_nm=405.0, **base)
    c.SourceSpec(spectrum=(c.SpectralLine(wavelength_nm=405.0, weight=1.0),), **base)
    with pytest.raises(ValidationError):
        c.SourceSpec(**base)
    with pytest.raises(ValidationError):
        c.SourceSpec(wavelength_nm=405.0, spectrum=(c.SpectralLine(wavelength_nm=405.0, weight=1.0),), **base)


def test_target_needs_comparison_and_tolerance():
    with pytest.raises(ValidationError, match="needs target_value"):
        c.Target(target_id="t", metric="fringe_period_nm", observation_id="o")
    with pytest.raises(ValidationError, match="requires tolerance_abs"):
        c.Target(target_id="t", metric="fringe_period_nm", observation_id="o", target_value=500.0)
    c.Target(target_id="t", metric="fringe_period_nm", observation_id="o", target_value=500.0, tolerance_abs=1.0)
    c.Target(target_id="t", metric="fringe_visibility", observation_id="o", minimum=0.8)


def test_project_rejects_duplicate_ids():
    comp = c.EngineeringComponentSnapshot(instance_id="A")
    with pytest.raises(ValidationError):
        c.ProjectSnapshot(components=(comp, comp))


def test_actuator_travel_rules():
    with pytest.raises(ValidationError):
        c.ActuatorSpec(actuator_id="a", dof="rotate", axis_local=(0, 0, 1), travel_min=1.0, travel_max=-1.0)
    with pytest.raises(ValidationError):
        c.ActuatorSpec(actuator_id="a", dof="rotate", axis_local=(0, 0, 1), travel_min=1.0, travel_max=2.0, zero=0.0)


def test_request_json_roundtrip():
    req = c.ValidationRequest(project=c.ProjectSnapshot(name="empty"))
    text = req.model_dump_json()
    again = c.ValidationRequest.model_validate_json(text)
    assert again == req
    assert json.loads(text)["configuration"]["numerical"]["config_version"]


# ------------------------------------------------------------------------------------------ errors
def test_status_aggregation_never_reads_empty_as_pass():
    assert worst([]) is Status.NOT_APPLICABLE
    assert worst([Status.NOT_APPLICABLE]) is Status.NOT_APPLICABLE
    assert worst([Status.PASS, Status.NOT_APPLICABLE]) is Status.PASS
    assert worst([Status.PASS, Status.WARNING]) is Status.WARNING
    assert worst([Status.WARNING, Status.UNKNOWN]) is Status.UNKNOWN
    assert worst([Status.UNKNOWN, Status.OUT_OF_MODEL_SCOPE]) is Status.OUT_OF_MODEL_SCOPE
    assert worst([Status.OUT_OF_MODEL_SCOPE, Status.FAIL, Status.PASS]) is Status.FAIL


def test_issue_context_is_json_safe_even_for_nan():
    import numpy as np

    issue = make_issue(
        ErrorCode.NUMERICAL_FAILURE, Status.FAIL, "boom", location="component:X/surface:s",
        affected_targets=("t1",), suggested_action="fix", value=np.float64("nan"), arr=np.array([1.0, 2.0]), ok=np.True_,
    )
    dumped = json.loads(issue.model_dump_json())
    assert dumped["numerical_context"]["value"] == "nan"
    assert dumped["numerical_context"]["arr"] == [1.0, 2.0]
    assert dumped["code"] == "NUMERICAL_FAILURE" and dumped["affected_targets"] == ["t1"]


# ------------------------------------------------------------------------------ canonical + hashing
def test_canonical_json_sorted_compact_and_rejects_nan():
    assert canonical_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"é"],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_hash_exact_key_order_independent_and_neg_zero_normalized():
    assert hash_exact({"a": 1.0, "b": [1, 2]}) == hash_exact({"b": [1, 2], "a": 1.0})
    assert hash_exact({"a": -0.0}) == hash_exact({"a": 0.0})
    assert hash_exact({"a": 1.0}) != hash_exact({"a": 1.0000000001})


def test_quantization_absorbs_last_bits_but_not_real_differences():
    assert quantize_float(0.1 + 0.2, 9, 1e-12) == quantize_float(0.3, 9, 1e-12)
    assert quantize_float(1e-15, 9, 1e-12) == 0.0
    assert quantize_float(-1e-15, 9, 1e-12) == 0.0
    assert hash_quantized({"v": 500.00000000000006}, 9, 1e-12) == hash_quantized({"v": 500.0}, 9, 1e-12)
    assert hash_quantized({"v": 500.001}, 9, 1e-12) != hash_quantized({"v": 500.0}, 9, 1e-12)
    with pytest.raises(ValueError):
        quantize_float(float("inf"), 9, 1e-12)


def test_engine_build_id_is_stable_and_versioned():
    a, b = engine_build_id(), engine_build_id()
    from os_optics._version import ENGINE_VERSION

    assert a == b and a.startswith(ENGINE_VERSION + "+") and len(a.split("+")[1]) == 16


def test_no_stdlib_shadowing_by_package_math_module():
    """os_optics.math is a subpackage; stdlib ``math`` must still be the real one inside the package."""
    import os_optics.math.rng as rng_module

    assert rng_module.math.__name__ == "math" and hasattr(rng_module.math, "log")
