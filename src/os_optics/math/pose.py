"""Rigid transform ``x_g = R x_l + t`` (NumPy-backed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_trig import atan2
from .rotation import matrix_to_quat, quat_to_matrix
from .vectors import Array, dot, matmul, matvec, normalize


@dataclass(frozen=True)
class Pose:
    rotation: Array  # (3,3), proper rotation
    translation: Array  # (3,)

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_quat(position: Array, quat_wxyz: Array) -> Pose:
        return Pose(quat_to_matrix(quat_wxyz), np.asarray(position, dtype=np.float64))

    @staticmethod
    def from_axis_direction(position: Array, axis: Array, up: Array, zero_tol: float) -> Pose:
        """Local ``z := axis``, local ``y := up`` orthogonalized against ``axis``, local ``x := y x z``.

        Raises ``ValueError`` if ``axis`` is zero or ``up`` is (anti)parallel to ``axis``.
        """
        z, ok_z = normalize(np.asarray(axis, dtype=np.float64), zero_tol)
        if not ok_z:
            raise ValueError("axis must be non-zero")
        u = np.asarray(up, dtype=np.float64)
        y, ok_y = normalize(u - float(dot(u, z)) * z, zero_tol)
        if not ok_y:
            raise ValueError("up vector must not be parallel to the axis")
        x = np.cross(y, z)
        return Pose(np.column_stack([x, y, z]), np.asarray(position, dtype=np.float64))

    def apply_points(self, p: Array) -> Array:
        return matvec(self.rotation, np.asarray(p, dtype=np.float64)) + self.translation

    def apply_dirs(self, d: Array) -> Array:
        return matvec(self.rotation, np.asarray(d, dtype=np.float64))

    def compose(self, child: Pose) -> Pose:
        """``self ∘ child`` (child expressed in self's frame)."""
        return Pose(matmul(self.rotation, child.rotation), matvec(self.rotation, child.translation) + self.translation)

    def inverse(self) -> Pose:
        rt = self.rotation.T
        return Pose(rt, -matvec(rt, self.translation))

    def quaternion(self) -> Array:
        return matrix_to_quat(self.rotation)

    def yaw_rad(self) -> float:
        """Rotation of the local x-axis about global z (plan-view rotation)."""
        return float(atan2(self.rotation[1, 0], self.rotation[0, 0]))  # (libm-free)
