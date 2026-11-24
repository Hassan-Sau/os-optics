"""Wire geometry shared by the OpenSci contracts."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field, field_validator

from ..base import StrictContractModel


Vector3 = Annotated[
    tuple[float, float, float],
    Field(description="Cartesian vector (x, y, z)."),
]
Quaternion = Annotated[
    tuple[float, float, float, float],
    Field(description="Quaternion (w, x, y, z), Hamilton, active rotation; non-zero, taken by its direction (normalized)."),
]


class PoseV1(StrictContractModel):
    """Rigid pose ``x_global = R(q) x_local + position_mm`` (mm; ``q`` = ``quaternion_wxyz``, Hamilton, active rotation;
    a quaternion of any non-zero length states the rotation of ``q / |q|``)."""

    position_mm: Vector3 = (0.0, 0.0, 0.0)
    quaternion_wxyz: Quaternion = (1.0, 0.0, 0.0, 0.0)

    @field_validator("quaternion_wxyz")
    @classmethod
    def _nonzero(cls, q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        # (by its components: a length computed from squares underflows to 0 for a small non-zero quaternion)
        if not any(c != 0.0 for c in q):
            raise ValueError("quaternion must be non-zero")
        return q

    def unit_quaternion(self) -> tuple[float, float, float, float]:
        """The rotation this pose states, ``q / |q|`` (the components as given when their length is exactly 1; the sign is
        kept). A stored pose is never rewritten: every rotation computed from it uses this."""

        return _unit(self.quaternion_wxyz)

    def compose(self, child: PoseV1) -> PoseV1:
        """``child`` (a pose in this pose's local frame) in this pose's parent frame:
        ``position = R(q) child.position + position``, ``q = q * child.q`` (Hamilton product), from the rotations both
        poses state (``unit_quaternion``): the result does not depend on the lengths of the stated quaternions, and its
        quaternion is unit."""

        w1, x1, y1, z1 = self.unit_quaternion()
        w2, x2, y2, z2 = child.unit_quaternion()
        quaternion = _unit((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))
        # v' = v + 2 u x (u x v + w v), u = (x1, y1, z1): the active rotation of the child's position
        vx, vy, vz = child.position_mm
        cx, cy, cz = y1 * vz - z1 * vy + w1 * vx, z1 * vx - x1 * vz + w1 * vy, x1 * vy - y1 * vx + w1 * vz
        rotated = (vx + 2.0 * (y1 * cz - z1 * cy), vy + 2.0 * (z1 * cx - x1 * cz), vz + 2.0 * (x1 * cy - y1 * cx))
        position = tuple(r + p for r, p in zip(rotated, self.position_mm, strict=True))
        return PoseV1(position_mm=position, quaternion_wxyz=quaternion)


#: The least length whose squared components keep their bits (below it, and where the squares overflow, ``_unit`` divides by
#: the largest component first).
_LEAST_PLAIN_LENGTH = 1e-150


def _unit(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(sum(c * c for c in q))
    if n == 1.0:
        return q
    if not (math.isfinite(n) and n >= _LEAST_PLAIN_LENGTH):
        # (a length that overflows or underflows: q / max|c| has length in [1, 2], then its own direction)
        largest = max(abs(c) for c in q)
        q = (q[0] / largest, q[1] / largest, q[2] / largest, q[3] / largest)
        n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
