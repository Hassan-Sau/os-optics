"""Damped least squares (Levenberg-Marquardt, Marquardt scaling) reference controller with limits and coarse capture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.project import CaptureScanSpec, ControlSpec
from ..contracts.results import ControllerRun
from ..errors import Status
from .actuation import Dof
from .readings import Reading
from .sensitivity import Evaluator, jacobian, total_values

Array = NDArray[np.float64]


@dataclass
class LMOutcome:
    run: ControllerRun
    command: dict[str, float]
    jacobian_rank: int | None = None
    orders: list[tuple[int, ...]] = field(default_factory=list)


def _targets(spec: ControlSpec) -> tuple[Array, Array]:
    target = np.array([s.target_value for s in spec.sensors], dtype=np.float64)
    weight = np.array([s.weight for s in spec.sensors], dtype=np.float64)
    return target, weight


def capture_scan(
    evaluator: Evaluator, dofs: list[Dof], scan: CaptureScanSpec, command: dict[str, float], disturbance: dict[str, float],
    cfg: NumericalConfig,
) -> dict[str, float] | None:
    """Deterministic serpentine raster over the travel of the scan DOFs; returns the best-signal command, or ``None``
    if no grid point sees the beam (capture range insufficient)."""
    by_ref = {d.ref: d for d in dofs}
    axes = []
    for ref in scan.actuator_refs:
        d = by_ref[ref]
        axes.append((ref, np.linspace(d.spec.travel_min, d.spec.travel_max, scan.points_per_axis)))
    points = scan.points_per_axis ** len(axes)
    if points > cfg.capture_scan_max_points:
        keep = max(2, int(cfg.capture_scan_max_points ** (1.0 / len(axes))))
        axes = [(ref, np.linspace(by_ref[ref].spec.travel_min, by_ref[ref].spec.travel_max, keep)) for ref, _ in axes]
    grid: list[tuple[float, ...]] = []
    if len(axes) == 1:
        grid = [(float(v),) for v in axes[0][1]]
    else:
        for i, a in enumerate(axes[0][1]):
            seq = axes[1][1] if i % 2 == 0 else axes[1][1][::-1]
            grid.extend((float(a), float(b)) for b in seq)
    best, best_fraction = None, -1.0
    for point in grid:
        probe = dict(command)
        for (ref, _), v in zip(axes, point):
            probe[ref] = v
        r = evaluator(total_values(probe, disturbance))
        if r.detected_fraction > best_fraction:
            best, best_fraction = probe, r.detected_fraction
    if best is None or best_fraction < cfg.capture_min_power_fraction:
        return None
    return best


def run_lm(
    evaluator: Evaluator,
    dofs: list[Dof],
    spec: ControlSpec,
    command0: dict[str, float],
    disturbance: dict[str, float],
    cfg: NumericalConfig,
    residual_tol: float,
) -> LMOutcome:
    target, weight = _targets(spec)
    refs = [d.ref for d in dofs]
    lo = np.array([d.spec.travel_min for d in dofs])
    hi = np.array([d.spec.travel_max for d in dofs])
    max_step = np.array([d.spec.max_step if d.spec.max_step is not None else math.inf for d in dofs])
    command = dict(command0)
    capture_used = False
    reading = evaluator(total_values(command, disturbance))
    if reading.lost:
        cmd = None
        if spec.capture_scan is not None:
            cmd = capture_scan(evaluator, dofs, spec.capture_scan, command, disturbance, cfg)
        if cmd is None:
            run = ControllerRun(converged=False, iterations=0, initial_cost=0.0, final_cost=0.0,
                                final_state=command, lost_beam=True, capture_used=spec.capture_scan is not None,
                                status=Status.FAIL, message="beam is lost and no capture strategy recovers it")
            return LMOutcome(run, command)
        command = cmd
        capture_used = True
        reading = evaluator(total_values(command, disturbance))

    def cost_of(r: Reading) -> tuple[Array, float]:
        res = weight * (r.values - target)
        return res, 0.5 * float(res @ res)

    res, cost = cost_of(reading)
    initial_cost = cost
    trajectory = [cost]
    lam = cfg.lm_initial_damping
    iterations = 0
    converged = math.sqrt(2.0 * cost) <= residual_tol
    rank: int | None = None
    orders: list[tuple[int, ...]] = []
    while not converged and iterations < cfg.lm_max_iterations:
        iterations += 1
        jr = jacobian(evaluator, dofs, command, disturbance, len(spec.sensors), cfg, base=reading)
        orders.append(jr.orders)
        a = weight[:, None] * jr.jacobian
        diag = np.maximum(np.einsum("ij,ij->j", a, a), cfg.lm_scaling_floor)
        accepted = False
        while lam <= cfg.lm_max_damping:
            aug = np.vstack([a, math.sqrt(lam) * np.diag(np.sqrt(diag))])
            rhs = np.concatenate([-res, np.zeros(len(dofs))])
            delta, *_ = np.linalg.lstsq(aug, rhs, rcond=cfg.jacobian_rcond)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(delta) > 0.0, max_step / np.abs(delta), math.inf)
            scale = min(1.0, float(np.min(ratio))) if len(delta) else 1.0
            delta = delta * scale
            c_vec = np.array([command[r] for r in refs])
            new_vec = np.clip(c_vec + delta, lo, hi)
            step_norm = float(np.linalg.norm(new_vec - c_vec))
            if step_norm <= cfg.lm_step_tol:
                lam = math.inf  # no admissible movement: give up
                break
            trial = dict(command)
            for r, v in zip(refs, new_vec):
                trial[r] = float(v)
            r_new = evaluator(total_values(trial, disturbance))
            if r_new.lost:
                lam *= cfg.lm_damping_up
                continue
            res_new, cost_new = cost_of(r_new)
            predicted = 0.5 * float(res @ res) - 0.5 * float(np.sum((res + a @ (new_vec - c_vec)) ** 2))
            if cost_new < cost and (cost - cost_new) >= cfg.lm_gain_accept * max(predicted, 0.0):
                command, reading, res, cost = trial, r_new, res_new, cost_new
                lam = max(lam * cfg.lm_damping_down, np.finfo(float).tiny)
                accepted = True
                break
            lam *= cfg.lm_damping_up
        trajectory.append(cost)
        if not accepted:
            break
        if math.sqrt(2.0 * cost) <= residual_tol or cost <= cfg.lm_cost_tol:
            converged = True
    converged = converged or math.sqrt(2.0 * cost) <= residual_tol
    final_vec = np.array([command[r] for r in refs])
    at_limit = [r for r, v, l_, h_ in zip(refs, final_vec, lo, hi)
                if v <= l_ + cfg.lm_step_tol or v >= h_ - cfg.lm_step_tol]
    status = Status.PASS if converged else Status.FAIL
    run = ControllerRun(
        converged=converged, iterations=iterations, initial_cost=initial_cost, final_cost=cost,
        final_state={k: float(command[k]) for k in sorted(command)}, saturated_actuators=tuple(at_limit),
        lost_beam=False, capture_used=capture_used, trajectory_cost=tuple(trajectory), status=status,
        message="" if converged else "Levenberg-Marquardt did not reach the residual tolerance",
    )
    return LMOutcome(run, command, rank, orders)
