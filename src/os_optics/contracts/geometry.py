"""Math-layer contracts: Pose3D (position + quaternion) and helpers."""

from __future__ import annotations

import math

import numpy as np
from pydantic import field_validator

from ..config import DEFAULT_NUMERICAL_CONFIG
from ..math.pose import Pose
from ..math.rotation import quat_from_axis_angle, quat_multiply
from .base import Quaternion, StrictModel, Vector3


# Envelope frame for optical elements (local z = beam axis, local y = up): envelope x = local z, y = local x, z = local y, i.e.
# the rotation matrix [[0,1,0],[0,0,1],[1,0,0]] whose columns are the envelope axes in component coordinates.
_PLAN_FRAME_OPTICAL_Z_Q = (0.5, -0.5, -0.5, -0.5)  # Hamilton (w, x, y, z)


class Pose3D(StrictModel):
    """Rigid pose ``x_global = R(q) x_local + position_mm`` (mm; ``q`` = ``quaternion_wxyz``, Hamilton, active rotation;
    a quaternion of any non-zero length states the rotation of ``q / |q|``)."""

    position_mm: Vector3 = (0.0, 0.0, 0.0)
    quaternion_wxyz: Quaternion = (1.0, 0.0, 0.0, 0.0)

    @field_validator("quaternion_wxyz")
    @classmethod
    def _nonzero(cls, q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        # (by its components: a length computed from squares underflows to 0 for a small non-zero quaternion, and
        # quat_normalize takes any non-zero finite one)
        if not all(math.isfinite(c) for c in q):
            raise ValueError("quaternion must be finite")
        if not any(c != 0.0 for c in q):
            raise ValueError("quaternion must be non-zero")
        return q

    def to_pose(self) -> Pose:
        """Lower to an internal ``Pose``, the quaternion normalized (``math.rotation.quat_to_matrix``); raises
        ``ValueError`` for a quaternion that is zero or not finite."""
        q = np.array(self.quaternion_wxyz, dtype=np.float64)
        return Pose.from_quat(np.array(self.position_mm, dtype=np.float64), q)

    @classmethod
    def from_pose(cls, pose: Pose) -> Pose3D:
        q = pose.quaternion()
        return cls(
            position_mm=tuple(float(v) for v in pose.translation),  # type: ignore[arg-type]
            quaternion_wxyz=tuple(float(v) for v in q),  # type: ignore[arg-type]
        )

    @classmethod
    def from_axis_direction(
        cls,
        position_mm: Vector3,
        axis: Vector3,
        up: Vector3 = (0.0, 0.0, 1.0),
    ) -> Pose3D:
        """Pose whose local ``+z`` is ``axis`` and local ``+y`` is ``up`` orthogonalized (the compute_pose helper)."""
        pose = Pose.from_axis_direction(
            np.array(position_mm, dtype=np.float64),
            np.array(axis, dtype=np.float64),
            np.array(up, dtype=np.float64),
            DEFAULT_NUMERICAL_CONFIG.zero_vector_tol,
        )
        return cls.from_pose(pose)

    @classmethod
    def plan_frame_for_optical_axis(cls) -> Pose3D:
        """Envelope frame for elements whose local ``+z`` is the optical (beam) axis and whose local ``+y`` is 'up':
        envelope ``x`` = optical axis, envelope ``y`` = transverse horizontal, envelope ``z`` = up."""
        return cls(position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=_PLAN_FRAME_OPTICAL_Z_Q)

    @classmethod
    def from_yaw_deg(cls, position_mm: Vector3, yaw_deg: float) -> Pose3D:
        """Rotation about global ``+z`` (plan-view rotation) by ``yaw_deg`` (right-handed)."""
        q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg))
        return cls(position_mm=position_mm, quaternion_wxyz=tuple(float(v) for v in q))  # type: ignore[arg-type]

    @classmethod
    def from_euler_zyx_deg(cls, position_mm: Vector3, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> Pose3D:
        """Intrinsic z-y'-x'' Tait-Bryan: ``R = Rz(yaw) Ry(pitch) Rx(roll)`` (angles right-handed)."""
        qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg))
        qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(pitch_deg))
        qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(roll_deg))
        q = quat_multiply(quat_multiply(qz, qy), qx)
        return cls(position_mm=position_mm, quaternion_wxyz=tuple(float(v) for v in q))  # type: ignore[arg-type]
