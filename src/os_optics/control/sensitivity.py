"""Finite-difference Jacobian and observability analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.results import SensitivityResult
from ..errors import Status
from .actuation import Dof
from .readings import Reading

Array = NDArray[np.float64]
Evaluator = Callable[[dict[str, float]], Reading]


@dataclass(frozen=True)
class JacobianResult:
    jacobian: Array  # (m, n)
    orders: tuple[int, ...]  # 2 central, 1 one-sided, 0 unavailable (beam lost)
    base: Reading


def total_values(command: dict[str, float], disturbance: dict[str, float]) -> dict[str, float]:
    """Physical DOF values = command + disturbance (uncontrolled DOFs appear only through the disturbance)."""
    keys = sorted(set(command) | set(disturbance))
    return {k: command.get(k, 0.0) + disturbance.get(k, 0.0) for k in keys}


def step_size(dof: Dof, cfg: NumericalConfig) -> float:
    return cfg.fd_step_translation_mm if dof.spec.dof == "translate" else math.degrees(cfg.fd_step_rotation_rad)


def jacobian(
    evaluator: Evaluator, dofs: list[Dof], command: dict[str, float], disturbance: dict[str, float], n_sensors: int,
    cfg: NumericalConfig, base: Reading | None = None,
) -> JacobianResult:
    """Central differences with the configured step; one-sided at travel limits; column unavailable if signal is lost."""
    if base is None:
        base = evaluator(total_values(command, disturbance))
    jac = np.zeros((n_sensors, len(dofs)))
    orders: list[int] = []
    for j, dof in enumerate(dofs):
        h = step_size(dof, cfg)
        lo, hi = dof.spec.travel_min, dof.spec.travel_max
        c0 = command[dof.ref]
        can_plus, can_minus = c0 + h <= hi, c0 - h >= lo
        readings: dict[str, Reading | None] = {"plus": None, "minus": None}
        for name, ok, sign in (("plus", can_plus, +1.0), ("minus", can_minus, -1.0)):
            if ok:
                probe = dict(command)
                probe[dof.ref] = c0 + sign * h
                r = evaluator(total_values(probe, disturbance))
                readings[name] = None if r.lost else r
        plus, minus = readings["plus"], readings["minus"]
        if plus is not None and minus is not None:
            jac[:, j] = (plus.values - minus.values) / (2.0 * h)
            orders.append(2)
        elif plus is not None and not base.lost:
            jac[:, j] = (plus.values - base.values) / h
            orders.append(1)
        elif minus is not None and not base.lost:
            jac[:, j] = (base.values - minus.values) / h
            orders.append(1)
        else:
            orders.append(0)
    return JacobianResult(jac, tuple(orders), base)


def analyse(jac_weighted: Array, refs: list[str], sensor_ids: list[str], orders: tuple[int, ...], jac_public: Array, cfg: NumericalConfig) -> SensitivityResult:
    """SVD-based observability: numerical rank, condition number, unobservable DOFs (null-space participation >= observability_participation_min)."""
    m, n = jac_weighted.shape
    if not np.any(jac_weighted):
        return SensitivityResult(
            actuator_refs=tuple(refs), sensor_ids=tuple(sensor_ids), jacobian=_rows(jac_public), difference_orders=orders,
            singular_values=(), numerical_rank=0, condition_number=None, unobservable_actuator_refs=tuple(refs), observable=False,
            status=Status.FAIL, message="no actuator moves any sensor reading (signal lost or DOFs inactive)",
        )
    u, sv, vt = np.linalg.svd(jac_weighted, full_matrices=True)
    smax = float(sv[0])
    rank = int(np.sum(sv > cfg.jacobian_rcond * smax))
    null = vt[rank:].T  # (n, n-rank)
    unobs: list[str] = []
    if null.shape[1] > 0:
        participation = np.sum(null**2, axis=1)  # diagonal of the null-space projector
        unobs = [refs[i] for i in range(n) if participation[i] >= cfg.observability_participation_min]
    cond = float(sv[0] / sv[rank - 1]) if rank == n and rank > 0 else None
    observable = rank == n
    if not observable:
        status, msg = Status.FAIL, f"Jacobian rank {rank} < {n} DOFs: some DOFs are unobservable from the sensors"
    elif cond is not None and cond >= cfg.jacobian_condition_warn:
        status, msg = Status.WARNING, f"Jacobian is ill-conditioned (condition number {cond:.3g})"
    else:
        status, msg = Status.PASS, ""
    return SensitivityResult(
        actuator_refs=tuple(refs), sensor_ids=tuple(sensor_ids), jacobian=_rows(jac_public), difference_orders=orders,
        singular_values=tuple(float(x) for x in sv), numerical_rank=rank, condition_number=cond,
        unobservable_actuator_refs=tuple(unobs), observable=observable, status=status, message=msg,
    )


def _rows(a: Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in a)
