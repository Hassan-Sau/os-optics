"""Sensor readings: deterministic engine outputs at observation planes for a given actuator state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.project import ControlSpec, ProjectSnapshot, SensorSpec
from ..contracts.results import ArrivalResult
from ..validation.observations import plane_fate
from ..validation.pipeline import SENSOR_READINGS, run_physics
from .actuation import offset_poses

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Reading:
    values: Array  # one value per sensor; NaN entries only when ``lost`` (never exposed publicly)
    lost: bool
    detected_fraction: float
    # a sensor without signal whose light was not all followed (an annular beam not followed, the trace budget, a path
    # ended out of model scope, untraced light that may reach it), or with a beam of unknown power on it: its reading is
    # not known, not a measured loss (``lost`` is then also True, so the controller does not use it)
    unknown: bool = False
    # per sensor (``spec.sensors`` order): whether that sensor's reading is not known; ``unknown`` is their any()
    unknown_sensors: tuple[bool, ...] = ()


def _available_power(project: ProjectSnapshot, sensor: SensorSpec) -> float:
    total = 0.0
    for c in project.components:
        if c.source is not None and (not sensor.source_ids or c.source.source_id in sensor.source_ids):
            total += c.source.power_W
    return total


def sensed_power_W(arrival: ArrivalResult) -> float:
    """The power a sensor reads from one arrival: what an absorbing plane detects (its active area's share, nothing outside
    its acceptance cone); on a transparent plane the beam's share on the active area (the whole beam without one, or
    when the share is not computed), nothing outside the acceptance cone."""
    if arrival.detected_power_W is not None:
        return arrival.detected_power_W
    if arrival.within_acceptance is False:
        return 0.0
    fraction = arrival.active_area_power_fraction
    power = arrival.beam.power_W
    return 0.0 if power is None else power * (fraction if fraction is not None else 1.0)


def evaluate(project: ProjectSnapshot, spec: ControlSpec, values: dict[str, float], cfg: NumericalConfig,
             quad_rule: tuple[Array, Array] | None = None) -> Reading:
    """Trace the scene with DOF ``values`` applied and read every sensor (``sensed_power_W``).

    A sensor is *lost* when the detected power (inside the active area, if any, and the acceptance cone) is below
    ``capture_min_power_fraction`` of the power its sources emit -- and its reading is *not known* (``unknown``) instead
    when light of its sources that was not followed (``validation/observations.plane_fate``: an annular beam not followed,
    the trace budget, a path ended out of model scope, untraced light that may reach the plane) could make up that signal. A beam of unknown power
    inside the sensor's acceptance leaves its reading not known too, whatever else arrives: it weighs in a power and a centroid by an amount no one states; so does a
    beam whose power is out of model scope there (``ArrivalResult.power_out_of_scope_at``), or light scattered out of one. The trace runs without
    pupil rays; behind a central obstruction the annulus is then carried by its envelope bounds alone. ``quad_rule``: the
    analysis' Gauss-Legendre rule, built once for all its readings."""
    poses = offset_poses(project, values, cfg)
    outcome = run_physics(project, cfg, plan=SENSOR_READINGS, component_poses=poses, quad_rule=quad_rule)
    obs = {o.observation_id: o for o in outcome.observations}
    out = np.full(len(spec.sensors), np.nan)
    worst_fraction = 1.0
    lost = False
    flags = [False] * len(spec.sensors)
    for k, sensor in enumerate(spec.sensors):
        o = obs.get(sensor.observation_id)
        arrivals = [] if o is None else [a for a in o.arrivals if not sensor.source_ids or a.beam.source_id in sensor.source_ids]
        powers = np.array([sensed_power_W(a) for a in arrivals])
        avail = _available_power(project, sensor)
        detected = float(powers.sum()) / avail if avail > 0 else 0.0
        worst_fraction = min(worst_fraction, detected)
        if any(a.beam.power_W is None and a.within_acceptance is not False for a in arrivals):
            lost = flags[k] = True
            continue
        if any(a.power_out_of_scope_at and a.within_acceptance is not False for a in arrivals) or (
                o is not None and any(line.power_out_of_scope_at for line in o.sampled_light
                                      if not sensor.source_ids or line.source_id in sensor.source_ids)):
            # (a beam whose power is out of model scope there: its power is its chief ray's, not the light's)
            lost = flags[k] = True
            continue
        if detected < cfg.capture_min_power_fraction:
            lost = True
            if o is not None and outcome.power_budget is not None:
                lines = [line for line in outcome.power_budget.lines if not sensor.source_ids or line.source_id in sensor.source_ids]
                # not known only when the light not followed could lift the reading to a signal (a beam's known tail
                # beside the active area, or a ghost too weak to matter, leaves it lost)
                fate = plane_fate(o, lines, cfg, outcome.reach)
                flags[k] = not fate.complete and float(powers.sum()) + fate.untraced_W >= cfg.capture_min_power_fraction * avail
            continue
        if sensor.quantity == "power_W":
            out[k] = float(powers.sum())
        else:
            idx = 0 if sensor.quantity == "centroid_u_mm" else 1
            us = np.array([a.uv_mm[idx] for a in arrivals])
            out[k] = float((us * powers).sum() / powers.sum())
    return Reading(out, lost, worst_fraction, any(flags), tuple(flags))
