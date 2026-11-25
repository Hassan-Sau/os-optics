"""Actuator DOF resolution and pose offsets. Unit is fixed by ``dof``: translate -> mm, rotate -> degrees."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import NumericalConfig
from ..contracts.components import ActuatorSpec, EngineeringComponentSnapshot
from ..contracts.project import ProjectSnapshot
from ..errors import EngineInputError, ErrorCode
from ..math.pose import Pose
from ..math.rotation import rotation_matrix_axis_angle
from ..scene.lower import _lower_pose


@dataclass(frozen=True)
class Dof:
    ref: str  # "<instance_id>/<actuator_id>"
    component: EngineeringComponentSnapshot
    spec: ActuatorSpec

    @property
    def unit(self) -> str:
        return "mm" if self.spec.dof == "translate" else "deg"


def resolve_dofs(project: ProjectSnapshot, refs: tuple[str, ...]) -> list[Dof]:
    table: dict[str, Dof] = {}
    for comp in project.components:
        for act in comp.actuators:
            ref = f"{comp.instance_id}/{act.actuator_id}"
            table[ref] = Dof(ref, comp, act)
    out = []
    for ref in refs:
        if ref not in table:
            raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"unknown actuator {ref!r}", location=f"control:{ref}", known=sorted(table))
        out.append(table[ref])
    if len({d.ref for d in out}) != len(out):
        raise EngineInputError(ErrorCode.DUPLICATE_ID, "duplicate actuator reference in control spec")
    return out


def all_dofs(project: ProjectSnapshot) -> dict[str, Dof]:
    return {f"{c.instance_id}/{a.actuator_id}": Dof(f"{c.instance_id}/{a.actuator_id}", c, a) for c in project.components for a in c.actuators}


def offset_poses(project: ProjectSnapshot, values: dict[str, float], cfg: NumericalConfig) -> dict[str, Pose]:
    """Component poses after applying DOF values (``value - zero`` relative to the as-designed pose).

    Multiple DOFs of one component compose in declaration order: ``T' = T ∘ D_1 ∘ D_2 ...`` (local-frame displacements)."""
    poses: dict[str, Pose] = {}
    dofs = all_dofs(project)
    for comp in project.components:
        pose = _lower_pose(comp.pose, cfg, f"component:{comp.instance_id}")
        for act in comp.actuators:
            ref = f"{comp.instance_id}/{act.actuator_id}"
            if ref not in values:
                continue
            delta = values[ref] - dofs[ref].spec.zero
            axis = np.array(act.axis_local, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            if act.dof == "translate":
                local = Pose(np.eye(3), axis * delta)
            else:
                rot = rotation_matrix_axis_angle(axis, math.radians(delta))
                pivot = np.array(act.pivot_local_mm, dtype=np.float64)
                local = Pose(rot, pivot - rot @ pivot)
            pose = pose.compose(local)
        poses[comp.instance_id] = pose
    return poses
