"""The results of an extended emitter's sampled rays, from ``optics/rays/sampled.py``.

Each source line's rays have equal weight ``emitted / sample_count``. On a plane, a crossing counts when the ray lands on the
active area (the whole plane without one) inside the acceptance cone; ``power_W`` is the sum of the counted crossings' powers
and ``collection_fraction = power_W / emitted_W``. The estimate from the first half of the sequence (a stratified sample by
itself) stands next to it; their absolute difference is the sampling-error indication the results report.

Accounting (``sampled_reach``): the ledger holds every emitted watt (``emitted = sum of its sinks``). Light a ray leaves
behind at an interface whose reflection is not followed may still reach a plane: it is stated per plane as light that may
reach it, so the plane's power is not complete. A plane that no ray of a line is counted on reads that line as a sampled zero
(``sampled_at_plane``: power 0.0, no counted crossing), whose resolution is one ray's weight: the plane's power is ``KNOWN``
with ``power_sampling_resolution_W`` beside it (never ``KNOWN_ZERO``: a finite number of rays does not prove a plane dark).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import NumericalConfig
from ..contracts.results import PI_SR, SampledCrossings, SampledLightAtPlane, SampledSourceLine
from ..optics.materials.library import MaterialLibrary
from ..optics.rays.emitter_samples import SEQUENCE_ID
from ..optics.rays.sampled import SampledRows, SampledTrace, SampleTracer
from ..optics.rays.tracer import SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, TraceObserver, TraceOutput
from ..scene.model import EmitterG, Scene
from .untraced import SAMPLED_BRANCH, PlaneReach

#: The ledger's sinks that hold light a ray left behind which goes on physically and is not followed.
_LEFT_BEHIND = (SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION)
_FULL_SPHERE_SR = 4.0 * PI_SR
_MM2_PER_M2 = 1e6


def trace_sampled_light(scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, trace: TraceOutput, quad_rule,
                        recorder: TraceObserver | None = None) -> SampledTrace:
    """Trace the emitters' samples and add what they booked to ``trace``: inputs, ledger, issues, kept paths and surface hits,
    and one trace-hash event per line and plane (their rows are not events)."""
    tracer = SampleTracer(scene, library, cfg, recorder=recorder, quad_rule=quad_rule)
    sampled = tracer.run_samples()
    trace.inputs.extend(tracer.out.inputs)
    trace.power_flows.extend(tracer.out.power_flows)
    trace.issues.extend(tracer.out.issues)
    trace.paths.extend(tracer.out.paths)
    trace.surface_hits.extend(tracer.out.surface_hits)
    for line in sampled.lines:
        trace.trace_events.append({"path": f"{line.source_id}#{line.spectral_index}", "launched": line.launched,
                                   "seed": line.seed, "steps": sampled.ray_steps})
    for (oid, source_id, j), rows in sorted(sampled.rows.items()):
        trace.trace_events.append({"path": f"{source_id}#{j}", "observe": oid, "crossings": len(rows.power_W),
                                   "power": float(np.sum(rows.power_W)), "u": float(np.sum(rows.u_mm)), "v": float(np.sum(rows.v_mm))})
    if recorder is not None:
        recorder.sampled_finished(sampled)
    return sampled


def _plane_line(line, rows: SampledRows | None) -> SampledLightAtPlane:
    """One line's sampled light at a plane; ``rows`` ``None`` (no ray of it crosses the plane): the sampled zero."""
    rows = SampledRows() if rows is None else rows
    power = np.array(rows.power_W, dtype=float)
    counted = np.array(rows.counted, dtype=bool)
    index = np.array(rows.sample_index, dtype=int)
    half_count = line.sample_count // 2
    received = float(power[counted].sum())
    first_half = float(power[counted & (index < half_count)].sum())
    fraction = received / line.emitted_W
    half_fraction = first_half / (line.emitted_W / 2.0)
    # (each ray's own contribution, its branches together: the spread of those over the rays traced)
    traced = max(line.launched, 1)
    contributions = np.bincount(index[counted], weights=power[counted], minlength=traced) / line.weight_W
    sigma = float(contributions.std()) / np.sqrt(traced)
    return SampledLightAtPlane(
        source_id=line.source_id, spectral_index=line.spectral_index, wavelength_nm=line.wavelength_nm, emitted_W=line.emitted_W,
        sample_count=line.sample_count, samples_traced=line.launched, power_W=received, collection_fraction=fraction,
        half_sample_count=half_count, collection_fraction_half_samples=half_fraction,
        collection_fraction_difference=abs(fraction - half_fraction), collection_fraction_sigma=float(sigma), crossing_count=int(power.size),
        counted_crossing_count=int(counted.sum()),
        crossings=SampledCrossings(sample_index=tuple(rows.sample_index), u_mm=tuple(rows.u_mm), v_mm=tuple(rows.v_mm),
                                   direction_u=tuple(rows.direction_u), direction_v=tuple(rows.direction_v),
                                   direction_n=tuple(rows.direction_n), power_W=tuple(rows.power_W), counted=tuple(rows.counted)))


def sampled_at_plane(sampled: SampledTrace | None, observation_id: str) -> tuple[SampledLightAtPlane, ...]:
    """The sampled light of every line at the plane, a line none of whose rays crosses it as the sampled zero."""
    if sampled is None:
        return ()
    return tuple(_plane_line(line, sampled.rows.get((observation_id, line.source_id, line.spectral_index))) for line in sampled.lines)


#: A sampled pattern's name in the results.
_PATTERNS = {"map": "INTENSITY_MAP", "field": "SAMPLED_FIELD", "speckle": "SPECKLE"}


def _source_line(emitter: EmitterG, line) -> SampledSourceLine:
    if emitter.shape == "disc":
        area = PI_SR * emitter.half_x * emitter.half_x
    else:
        area = 4.0 * emitter.half_x * emitter.half_y
    pattern = emitter.pattern
    law = "FIELD_WAVEVECTOR" if pattern is not None and pattern.coherent else emitter.law
    cone_sr = PI_SR * emitter.cone_sine * emitter.cone_sine
    domain = {"LAMBERTIAN": PI_SR, "ISOTROPIC": _FULL_SPHERE_SR, "COSINE_POWER": 2.0 * PI_SR, "UNIFORM_CONE": cone_sr,
              "COLLIMATED": 0.0, "FIELD_WAVEVECTOR": None}[law]
    radiance = None
    if pattern is None and law in ("LAMBERTIAN", "UNIFORM_CONE"):
        radiance = line.emitted_W / (domain * area / _MM2_PER_M2)
    intensity = None
    if law == "ISOTROPIC":
        intensity = line.emitted_W / _FULL_SPHERE_SR
    elif law == "COSINE_POWER":
        intensity = line.emitted_W * (emitter.exponent + 1.0) / (2.0 * PI_SR)
    extra: dict = {}
    if pattern is not None:
        mass = float(pattern.weights.sum()) * pattern.grid.cell_area
        if pattern.interpolation == "BILINEAR":
            w = pattern.weights
            mass = float(0.25 * (w[:-1, :-1] + w[:-1, 1:] + w[1:, :-1] + w[1:, 1:]).sum()) * pattern.grid.cell_area
        extra = {"pattern": _PATTERNS[pattern.kind], "exitance_mean_W_per_mm2": line.emitted_W / area,
                 "exitance_peak_W_per_mm2": line.emitted_W * float(pattern.weights.max()) / mass}
        if pattern.kind == "speckle":
            extra["realizations"] = pattern.realizations
    if law == "UNIFORM_CONE":
        extra["cone_half_angle_deg"] = emitter.cone_half_angle_deg
    if law == "COSINE_POWER":
        extra["exponent"] = emitter.exponent
    return SampledSourceLine(
        source_id=line.source_id, spectral_index=line.spectral_index, wavelength_nm=line.wavelength_nm, emitted_W=line.emitted_W,
        shape="DISC" if emitter.shape == "disc" else "RECTANGLE", area_mm2=area, angular_law=law,
        solid_angle_measure="PROJECTED_SOLID_ANGLE" if law in ("LAMBERTIAN", "UNIFORM_CONE", "COLLIMATED") else "SOLID_ANGLE",
        domain_sr=domain, radiance_W_per_m2_sr=radiance, intensity_W_per_sr=intensity,
        sample_count=line.sample_count, samples_traced=line.launched, seed=line.seed, sequence=SEQUENCE_ID,
        ray_weight_W=line.weight_W, **extra)


def sampled_sources(scene: Scene, sampled: SampledTrace | None) -> tuple[SampledSourceLine, ...]:
    """What each emitter line is and how it was sampled."""
    if sampled is None:
        return ()
    emitters = {e.source_id: e for e in scene.emitters}
    return tuple(_source_line(emitters[line.source_id], line) for line in sampled.lines)


def sampled_reach(reach: PlaneReach, sampled: SampledTrace | None, trace: TraceOutput, observation_ids: list[str],
                  cfg: NumericalConfig) -> PlaneReach:
    """``reach`` with the light of the sampled lines that may reach each plane (module docstring)."""
    if sampled is None:
        return reach
    left_behind: dict[tuple[str, int], float] = {}
    for flow in trace.power_flows:
        if flow.kind in _LEFT_BEHIND:
            key = (flow.source_id, flow.spectral_index)
            left_behind[key] = left_behind.get(key, 0.0) + flow.power_W
    out = {oid: dict(lines) for oid, lines in reach.items()}
    for oid in observation_ids:
        for line in sampled.lines:
            key = (line.source_id, line.spectral_index)
            kinds = dict(out.get(oid, {}).get(key, {}))
            if left_behind.get(key, 0.0) > cfg.negligible_power_fraction * line.emitted_W:
                kinds[SAMPLED_BRANCH] = left_behind[key]
            if kinds:
                out.setdefault(oid, {})[key] = kinds
    return out


def combine_surface_hits(hits: list, sampled: SampledTrace | None) -> list:
    """The trace's surface hits with each sampled line's one hit per surface holding the power of all its rays there."""
    if sampled is None or not sampled.hit_W:
        return hits
    return [replace(hit, power_W=sampled.hit_W[key]) if (key := (hit.surface_id, hit.source_id, hit.spectral_index)) in sampled.hit_W
            and hit.path_id in sampled.hit_paths else hit for hit in hits]

