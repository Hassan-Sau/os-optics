"""The results of the light the run's surfaces and media scatter (``optics/rays/scatter.py``, ``optics/rays/scatter_trace.py``).

Surface scatter that the run traces is reported like an extended emitter's light: on each plane its rays cross, one
``SampledLightAtPlane`` per traced origin (``scatter_surface_id`` and ``parent_path_id`` name the surface and the beam), with
the collection fraction of the scattered power, its first-half estimate and the crossings' columns; its counted power is part
of the plane's power. ``sampled_sources`` lists each traced origin (a point source of the scattered power, its angular law and
its sampling). The rays' fates replace the origin's ``SURFACE_SCATTER`` entry in its line's ledger.

Accounting on a plane, as for an emitter: what the scatter rays leave behind at an interface whose reflection is not followed
is light that may reach it (``SAMPLED_BRANCH``), and a plane none of an origin's rays is counted on reads that origin as a
sampled zero (``scattered_at_plane``) whose resolution is one ray's weight;
scattered light the run does not trace (``SURFACE_SCATTER`` still in the ledger) and bulk scattering (``BULK_SCATTER``) are
light of unknown fate of their line (``tracer.UNKNOWN_FATE_SINKS``): above the line's negligible level no plane that line
lights reads a complete power, and one notice per surface or medium (``SCATTERED_LIGHT_NOT_FOLLOWED``) states the power not
followed; at or below it the power is left out as negligible (``untraced.left_out_ledger_W``).
"""

from __future__ import annotations

import math

from ..config import NumericalConfig
from ..contracts.results import PI_SR, SampledLightAtPlane, SampledSourceLine
from ..errors import ErrorCode, ValidationIssue, notice
from ..math.vectors import correctly_rounded_sum
from ..optics.materials.library import MaterialLibrary
from ..optics.rays.emitter_samples import SEQUENCE_ID
from ..optics.rays.scatter import SINK_BULK_SCATTER, SINK_SURFACE_SCATTER
from ..optics.rays.scatter_trace import ScatterTrace, trace_scatter
from ..optics.rays.tracer import SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, TraceOutput
from ..scene.model import Scene
from .sampled_light import _plane_line
from .untraced import SAMPLED_BRANCH, PlaneReach

_LEFT_BEHIND = (SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION)
_TWO_PI_SR = 2.0 * PI_SR


def trace_scattered_light(scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, trace: TraceOutput, quad_rule) -> ScatterTrace:
    """Trace the rays of ``trace.scatter_origins`` and put what they booked into ``trace``: their ledger entries in place of
    the traced origins' ``SURFACE_SCATTER`` entries, their issues, their surface hits (one per origin and surface) and one
    trace-hash event per origin and plane."""

    scattered, tracer = trace_scatter(scene, library, cfg, trace.scatter_origins, quad_rule)
    traced = [(line.origin.path_id, line.origin.surface_id, line.origin.power_W) for line in scattered.lines]
    kept = []
    for flow in trace.power_flows:
        key = (flow.path_id, flow.surface_id, flow.power_W)
        if flow.kind == SINK_SURFACE_SCATTER and key in traced:
            traced.remove(key)
            continue
        kept.append(flow)
    trace.power_flows = kept + tracer.out.power_flows
    scattered.flows = list(tracer.out.power_flows)
    trace.issues.extend(tracer.out.issues)
    for line in scattered.lines:
        trace.surface_hits.extend(line.hits)
        origin = line.origin
        trace.trace_events.append({"path": origin.path_id, "scatter": origin.surface_id, "launched": line.line.launched,
                                   "seed": line.line.seed, "power": origin.power_W})
        for oid, rows in sorted(line.rows.items()):
            trace.trace_events.append({"path": origin.path_id, "scatter": origin.surface_id, "observe": oid, "crossings": len(rows.power_W),
                                       "power": correctly_rounded_sum(rows.power_W), "u": correctly_rounded_sum(rows.u_mm),
                                       "v": correctly_rounded_sum(rows.v_mm)})
    return scattered


def scattered_at_plane(scattered: ScatterTrace | None, observation_id: str) -> tuple[SampledLightAtPlane, ...]:
    """The scattered light of every traced origin at the plane, an origin none of whose rays crosses it as the sampled zero."""

    if scattered is None:
        return ()
    return tuple(_plane_line(line.line, line.rows.get(observation_id)).model_copy(update={
        "scatter_surface_id": line.origin.surface_id, "parent_path_id": line.origin.path_id, "power_out_of_scope_at": line.origin.power_scope})
        for line in scattered.lines)


def scattered_sources(scattered: ScatterTrace | None) -> tuple[SampledSourceLine, ...]:
    """What each traced origin is (a point source of the scattered power) and how it was sampled."""

    if scattered is None:
        return ()
    out = []
    for item in scattered.lines:
        origin, line = item.origin, item.line
        n = origin.exponent if origin.law == "COSINE_POWER" else 1.0
        out.append(SampledSourceLine(
            source_id=line.source_id, spectral_index=line.spectral_index, wavelength_nm=line.wavelength_nm, emitted_W=line.emitted_W,
            shape="POINT", area_mm2=0.0, angular_law=origin.law, solid_angle_measure="SOLID_ANGLE", domain_sr=_TWO_PI_SR,
            radiance_W_per_m2_sr=None, intensity_W_per_sr=line.emitted_W * (n + 1.0) / _TWO_PI_SR, sample_count=line.sample_count,
            samples_traced=line.launched, seed=line.seed, sequence=SEQUENCE_ID, ray_weight_W=line.weight_W,
            scatter_surface_id=origin.surface_id, parent_path_id=origin.path_id, scatter_side=origin.side.upper(),
            scattered_fraction=origin.fraction, incident_power_W=origin.incident_power_W,
            exponent=origin.exponent if origin.law == "COSINE_POWER" else None, beam_radius_mm=origin.beam_radius_mm,
            power_out_of_scope_at=origin.power_scope))
    return tuple(out)


def scattered_reach(reach: PlaneReach, scattered: ScatterTrace | None, observation_ids: list[str], cfg: NumericalConfig) -> PlaneReach:
    """``reach`` with the scattered light that may reach each plane (module docstring): per traced origin, what its rays left
    behind (``ScatterTrace.flows``: the scatter rays' ledger entries)."""

    if scattered is None or not scattered.lines:
        return reach
    out = {oid: {line: dict(kinds) for line, kinds in lines.items()} for oid, lines in reach.items()}
    left_behind: dict[tuple[str, int], float] = {}
    for flow in scattered.flows:
        if flow.kind in _LEFT_BEHIND:
            key = (flow.source_id, flow.spectral_index)
            left_behind[key] = left_behind.get(key, 0.0) + flow.power_W
    for oid in observation_ids:
        for item in scattered.lines:
            key = (item.line.source_id, item.line.spectral_index)
            kinds = out.setdefault(oid, {}).setdefault(key, {})
            if left_behind.get(key, 0.0) > cfg.negligible_power_fraction * item.line.emitted_W:
                kinds[SAMPLED_BRANCH] = max(kinds.get(SAMPLED_BRANCH, 0.0), left_behind[key])
            if not kinds:
                del out[oid][key]
        if oid in out and not out[oid]:
            del out[oid]
    return out


def scatter_issues(trace: TraceOutput, library: MaterialLibrary, scattered: ScatterTrace | None,
                   cfg: NumericalConfig) -> list[ValidationIssue]:
    """One ``SCATTERED_LIGHT_NOT_FOLLOWED`` notice (``NOT_APPLICABLE``) per surface whose scattered light stays booked and per
    medium that scatters a traced beam, where that power exceeds the negligible level of the source lines it comes from
    (``negligible_power_fraction`` x their input, the level ``validation/power.line_fate`` applies to the same light): the
    power not followed, which every plane its source lines light may receive. The planes carry the consequence in their
    own state, ``untraced_power_W`` and entry; the ledger keeps every scattered watt whatever its size. A surface's notice
    states the scatter-origin budget in force (``max_scatter_origins``) and how many of its scatter events lie beyond it
    (``origins_beyond_budget``, with their power)."""

    level = cfg.negligible_power_fraction
    inputs = {(line.source_id, line.spectral_index): line.power_W for line in trace.inputs}
    surfaces: dict[str, float] = {}
    media: dict[str, float] = {}
    lines: dict[str, set] = {}  # (surface / medium -> the source lines whose light it scattered)
    for flow in trace.power_flows:
        if flow.kind == SINK_SURFACE_SCATTER and flow.surface_id is not None:
            surfaces[flow.surface_id] = surfaces.get(flow.surface_id, 0.0) + flow.power_W
            lines.setdefault(f"surface:{flow.surface_id}", set()).add((flow.source_id, flow.spectral_index))
        elif flow.kind == SINK_BULK_SCATTER:
            media[flow.detail] = media.get(flow.detail, 0.0) + flow.power_W
            lines.setdefault(f"material:{flow.detail}", set()).add((flow.source_id, flow.spectral_index))

    def above(location: str, power: float) -> bool:
        return power > level * math.fsum(inputs.get(line, 0.0) for line in sorted(lines[location]))

    issues = []
    beyond_budget = [] if scattered is None else scattered.beyond_budget
    for surface_id, power in sorted(surfaces.items()):
        if not above(f"surface:{surface_id}", power):
            continue
        beyond = [o for o in beyond_budget if o.surface_id == surface_id]
        action = "read power_upper_bound_W and untraced_power_W of the planes"
        why = "light a scattered or sampled ray scatters again, or a calculation of the beams alone"
        context: dict[str, object] = {}
        if scattered is not None:
            context["max_scatter_origins"] = scattered.max_origins
            context["origins_beyond_budget"] = len(beyond)
        if beyond:
            context["power_beyond_budget_W"] = correctly_rounded_sum([o.power_W for o in beyond])
            why = (f"{len(beyond)} of its scatter events lie beyond the {scattered.max_origins} strongest of the run, which "
                   f"compute_budget.max_scatter_origins lets it trace; else {why}")
            action = (f"raise compute_budget.max_scatter_origins (set_compute_budget) to trace more than {scattered.max_origins} "
                      f"scatter events; {action}")
        issues.append(notice(
            ErrorCode.SCATTERED_LIGHT_NOT_FOLLOWED,
            f"{power:.4g} W scattered by this surface is not followed as rays ({why}): it may "
            "reach any plane, whose power is then not complete", location=f"surface:{surface_id}", scattered_power_W=power,
            rays_traced=scattered is not None, suggested_action=action, **context))
    for medium, power in sorted(media.items()):
        if not above(f"material:{medium}", power):
            continue
        ev = library.evaluator(medium)
        issues.append(notice(
            ErrorCode.SCATTERED_LIGHT_NOT_FOLLOWED,
            f"{power:.4g} W is scattered out of the traced light in the bulk of {medium} (scattering coefficient "
            f"{ev.scattering_per_mm:.6g} /mm) and not followed: it may reach any plane, whose power is then not complete",
            location=f"material:{medium}", scattered_power_W=power, scattering_coefficient_per_mm=ev.scattering_per_mm,
            anisotropy_g=ev.anisotropy_g,
            suggested_action="read power_upper_bound_W and untraced_power_W of the planes; the transmitted beams carry "
                             "exp(-(absorption + scattering) x path) of their power"))
    return issues


__all__ = ["scatter_issues", "scattered_at_plane", "scattered_reach", "scattered_sources", "trace_scattered_light"]
