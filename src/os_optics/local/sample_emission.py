"""Sample emission and detection in the Engine request of a Project.

A ``FLUORESCENCE`` or ``RAMAN`` source (``shared_contracts.optical_design.source``) emits light whose power follows from the
excitation the Project's other sources deliver to the sample. ``complete_request`` gives the Engine request its sample
emitters in two steps:

1. the request without the samples' emission and without targets is traced (the complete evaluation,
   ``validation.validate``): the excitation trace;
2. at each sample's excitation plane the power ``P_ex`` of the excitation wavelength must be a known quantity (that
   wavelength's ``power_by_wavelength`` entry complete, and none of its beams there out of model scope or bounded only, whatever
   other wavelengths do); it gives the emitted lines
   (fluorescence: ``P_i = P_ex a QY f_i lambda_ex / lambda_i``, ``f_i`` the line's photon share; Raman:
   ``P_j = P_ex N sigma_j L lambda_ex / lambda_j``), recorded as the component's ``SampleEmission`` and traced as its
   ``ExtendedEmitter`` (none when it emits nothing). A power that is not known refuses the Project
   (``SAMPLE_EXCITATION_NOT_KNOWN``): the emission is never guessed. A sample outside its model's validity limit (absorbance
   above 0.1; a Raman scattered share above 0.01) is computed by the same formulas and its record carries the exceeded
   condition (``SampleEmission.validity``).

The trace of the completed request delivers the same excitation (its other sources are traced the same way, and the emission
lines are longer than the excitation's); ``check_excitation`` verifies that on the run and refuses a run where it differs.

Detection (``ObservationV1.detection``) lowers to the Engine's ``Detection``: a lock-in's modulated sources carry the share of
their traced power that follows the modulation -- 1 for a named source, and for a sample not named the share of its excitation
that named sources deliver (its emission follows its excitation instantly).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .._vendor.shared_contracts.optical_design import FluorescenceSourceV1, RamanSourceV1
from .._vendor.shared_contracts.optical_design.source import MAX_RAMAN_SCATTERED_FRACTION, photon_energy_j
from ..contracts.basis import ValidityCheck
from ..contracts.components import EngineeringComponentSnapshot, ExtendedEmitter, SpectralLine
from ..contracts.detection import (
    Detection,
    LockIn,
    ModulatedSource,
    Modulation,
    SampleEmission,
    SampleEmissionLine,
    SampleExcitationShare,
)
from ..contracts.request import ValidationRequest
from ..contracts.results import ObservationResult, ValidationResult
from ..errors import ExecutionOutcome, QuantityState
from ..manifest import hash_exact
from ..math.exact_trig import _LN10 as LN10, expm1
from ..validation.superposition import received_power_W
from .errors import LocalRunError, LocalRunErrorCode, LocalRunProblem

#: Two wavelengths this close (nm) are the same excitation wavelength.
WAVELENGTH_MATCH_NM = 1e-9
#: The excitation the completed trace delivers may differ from the excitation trace's by this relative amount (rounding).
EXCITATION_REPEAT_RTOL = 1e-12
FLUORESCENCE_MODEL = ("Fluorescence of a thin, optically dilute sample (absorbance at most 0.1): absorbed share of the arriving excitation power (1 - 10^-A, or as stated), "
                      "quantum yield photons per absorbed photon, emission lines by photon share, each photon carrying the energy of "
                      "its own wavelength; linear excitation, no saturation, bleaching or reabsorption, emission following the "
                      "excitation instantly, unpolarized, incoherent; the excitation beam is not attenuated by the sample.")
RAMAN_MODEL = ("Spontaneous Stokes Raman scattering of a thin sample: N 4 pi (d sigma / d Omega) L of the arriving excitation photons "
               "per band, each scattered photon carrying the energy of its own wavelength; single scattering of an undepleted "
               "excitation, isotropic and unpolarized, incoherent; the excitation beam is not attenuated by the scattering.")


def is_sample(model: Any) -> bool:
    """Whether a source model is a sample emission (``FLUORESCENCE`` / ``RAMAN``)."""

    return isinstance(model, (FluorescenceSourceV1, RamanSourceV1))


def bare_component(component: Any, pose: Any) -> EngineeringComponentSnapshot:
    """The Engine component of a sample before its emission is known: placed, no emitter."""

    return EngineeringComponentSnapshot(instance_id=component.instance_id, role="source", pose=pose)


def _match(a: float, b: float) -> bool:
    return abs(a - b) <= WAVELENGTH_MATCH_NM


def excitation_at(obs: ObservationResult, wavelength_nm: float) -> tuple[float | None, dict[str, float], str | None]:
    """``(P_ex, power per source, why it is not known)`` of ``wavelength_nm`` on ``obs``."""

    entry = next((e for e in obs.power_by_wavelength if _match(e.wavelength_nm, wavelength_nm)), None)
    if entry is None:
        return None, {}, f"no source of the Project emits {wavelength_nm} nm, the excitation wavelength"
    if entry.untraced_power_W is None:
        if obs.quantity_state is QuantityState.NOT_COMPUTED:
            return None, {}, (f"whether the power of {wavelength_nm} nm arriving at the plane is complete was not computed: the "
                              "evaluation did not run the plane bounds")
        return None, {}, f"the power of {wavelength_nm} nm arriving at the plane is not established"
    if not entry.complete or entry.power_W is None:
        return None, {}, (f"the power of {wavelength_nm} nm arriving at the plane is not complete: {entry.untraced_power_W:.4g} W of it "
                          "is not followed, or light the trace does not follow may reach the plane")
    arrivals = [a for a in obs.arrivals if _match(a.beam.wavelength_nm, wavelength_nm)]
    if any(a.power_out_of_scope_at for a in arrivals):
        return None, {}, "the power of an excitation beam at the plane is out of model scope (power_out_of_scope_at)"
    if any(line.power_out_of_scope_at for line in obs.sampled_light if _match(line.wavelength_nm, wavelength_nm)):
        return None, {}, ("light scattered out of a beam whose power is out of model scope reaches the plane at the excitation wavelength "
                          "(power_out_of_scope_at)")
    if any(a.power_taken_before_upper_W is not None for a in arrivals):
        return None, {}, "part of an excitation beam is taken before the plane by a bounded share: its power there is bounded, not known"
    if any(a.detected_power_min_W is not None and a.detected_power_min_W < received_power_W(a) for a in arrivals):
        return None, {}, "part of an excitation beam may miss the plane's active area: its power there is bounded, not known"
    shares: dict[str, list[float]] = defaultdict(list)
    for arrival in obs.arrivals:
        if _match(arrival.beam.wavelength_nm, wavelength_nm):
            shares[arrival.beam.source_id].append(received_power_W(arrival))
    for line in obs.sampled_light:
        if line.crossing_count and _match(line.wavelength_nm, wavelength_nm):  # (a sampled zero adds no share)
            shares[line.source_id].append(line.power_W)
    return entry.power_W, {source: math.fsum(values) for source, values in sorted(shares.items())}, None


def absorbed_share(model: Any) -> float:
    """A fluorescence sample's absorbed share: ``absorbed_fraction``, or ``1 - 10^-A = -expm1(-A ln 10)`` from
    ``math.exact_trig`` (the line powers are Engine input: the same bits in every runtime)."""

    if model.absorbed_fraction is not None:
        return float(model.absorbed_fraction)
    return -float(expm1(np.array(-model.absorbance * LN10)))


def model_validity(model: Any, instance_id: str) -> tuple[ValidityCheck, ...]:
    """The sample model's validity condition where ``model`` is outside it (``SampleEmission.validity``); empty inside it
    (the limit is a constant of the model: nothing to state there)."""

    if isinstance(model, FluorescenceSourceV1):
        parameter, limit, observed = model.thin_sample_check
    else:
        parameter, limit, observed = "scattered_fraction", MAX_RAMAN_SCATTERED_FRACTION, math.fsum(model.scattered_fractions)
    if observed <= limit:
        return ()
    return (ValidityCheck(parameter=parameter, maximum=limit, observed=observed, within=False, at=(f"component {instance_id}",)),)


def emission_record(model: Any, excitation_W: float, shares: dict[str, float], input_hash: str,
                    resolution_W: float | None, instance_id: str = "") -> SampleEmission:
    """The ``SampleEmission`` of ``model`` excited by ``excitation_W`` (module docstring)."""

    wavelength = model.excitation.wavelength_nm
    rate = excitation_W / photon_energy_j(wavelength)
    lines = []
    if isinstance(model, FluorescenceSourceV1):
        absorbed = absorbed_share(model)
        for line, share in zip(model.emission_lines, model.photon_shares, strict=True):
            conversion = absorbed * model.quantum_yield * share
            ratio = wavelength / line.wavelength_nm
            lines.append(SampleEmissionLine(wavelength_nm=line.wavelength_nm, photon_fraction=share,
                                            photons_per_excitation_photon=conversion, energy_ratio=ratio,
                                            photon_rate_per_s=rate * conversion, power_W=excitation_W * conversion * ratio))
        extra = {"absorbed_fraction": absorbed, "quantum_yield": model.quantum_yield, "absorbed_power_W": excitation_W * absorbed}
        process, text = "FLUORESCENCE", FLUORESCENCE_MODEL
    else:
        for line, emitted, conversion in zip(model.raman_lines, model.wavelengths_nm, model.scattered_fractions):
            ratio = wavelength / emitted
            lines.append(SampleEmissionLine(wavelength_nm=emitted, raman_shift_cm1=line.shift_cm1, photons_per_excitation_photon=conversion,
                                            energy_ratio=ratio, photon_rate_per_s=rate * conversion, power_W=excitation_W * conversion * ratio))
        extra = {}
        process, text = "RAMAN", RAMAN_MODEL
    return SampleEmission(
        process=process, model=text, excitation_observation_id=model.excitation.observation_id, excitation_wavelength_nm=wavelength,
        excitation_power_W=excitation_W, excitation_power_resolution_W=resolution_W, excitation_photon_rate_per_s=rate,
        excitation_by_source=tuple(SampleExcitationShare(source_id=s, power_W=p) for s, p in shares.items()),
        excitation_input_hash=input_hash, lines=tuple(lines), emitted_power_W=math.fsum(line.power_W for line in lines),
        emitted_photon_rate_per_s=math.fsum(line.photon_rate_per_s for line in lines), validity=model_validity(model, instance_id),
        **extra)


def emitter_of(component: EngineeringComponentSnapshot, model: Any, record: SampleEmission) -> EngineeringComponentSnapshot:
    """The sample's component with its emission record and, when it emits, the emitter of exactly those lines."""

    emitted = [line for line in record.lines if line.power_W > 0.0]
    if not emitted:
        return component.model_copy(update={"sample_emission": record})
    surface = model.emitting_surface
    size = (surface.width_mm, surface.height_mm) if surface.shape == "RECTANGLE" else (surface.diameter_mm, surface.diameter_mm)
    emitter = ExtendedEmitter(
        source_id=component.instance_id, spectrum=tuple(SpectralLine(wavelength_nm=line.wavelength_nm, weight=line.power_W) for line in emitted),
        power_W=record.emitted_power_W, shape=surface.shape, size_x_mm=size[0], size_y_mm=size[1], angular_law=model.angular_law,
        sample_count=model.sampling.sample_count, seed=model.sampling.seed)
    return EngineeringComponentSnapshot.model_validate({**component.model_dump(), "emitter": emitter.model_dump(), "sample_emission": record.model_dump()})


def _refusal(instance_id: str, model: Any, message: str, state: Any = None) -> LocalRunProblem:
    return LocalRunProblem(
        LocalRunErrorCode.SAMPLE_EXCITATION_NOT_KNOWN,
        f"the excitation of {model.model_kind.lower()} sample {instance_id} at observation {model.excitation.observation_id} "
        f"({model.excitation.wavelength_nm} nm) is not known: {message}; its emission cannot be derived",
        f"component {instance_id}",
        {"observation_id": model.excitation.observation_id, "wavelength_nm": model.excitation.wavelength_nm,
         "quantity_state": None if state is None else getattr(state, "value", state)})


def _lock_in_shares(lock_in: Any, records: dict[str, SampleEmission]) -> tuple[ModulatedSource, ...]:
    named = set(lock_in.modulation.source_ids)
    sources = [ModulatedSource(source_id=s, fraction=1.0) for s in lock_in.modulation.source_ids]
    for instance_id, record in records.items():
        if instance_id in named or record.excitation_power_W <= 0.0:
            continue
        share = math.fsum(s.power_W for s in record.excitation_by_source if s.source_id in named) / record.excitation_power_W
        if share > 0.0:
            sources.append(ModulatedSource(source_id=instance_id, fraction=min(share, 1.0)))
    return tuple(sources)


def engine_detection(detection: Any, records: dict[str, SampleEmission]) -> Detection:
    """The Engine ``Detection`` of a Project observation's ``DetectionV1``."""

    lock_in = detection.lock_in
    engine_lock_in = None
    if lock_in is not None:
        modulation = lock_in.modulation
        engine_lock_in = LockIn(
            modulation=Modulation(waveform=modulation.waveform, frequency_Hz=modulation.frequency_Hz, duty_cycle=modulation.duty_cycle,
                                  modulation_depth=modulation.modulation_depth, sources=_lock_in_shares(lock_in, records)),
            time_constant_s=lock_in.time_constant_s, filter_order=lock_in.filter_order)
    return Detection(detector=detection.detector, lock_in=engine_lock_in)


def complete_request(project: Any, request: ValidationRequest, *, engine_state: Any = None) -> ValidationRequest:
    """``request`` (the Project's Engine request, its samples without emitters and its planes without detection) with the
    samples' emission derived from the excitation trace and every plane's detection (module docstring). Unchanged for a
    Project without samples or detection. ``engine_state``: the ``EvaluationState`` of the evaluations that follow (an
    ``EvaluationSession``'s): the excitation trace identifies the Engine build through it, so a run identifies it once.
    The excitation trace's own products are not kept: its request is another than the run's (no emitter on the samples,
    no detection on the planes, no targets)."""

    samples = {c.instance_id: c.source_model for c in project.components if is_sample(c.source_model)}
    detections = {o.observation_id: o.detection for o in project.observations if o.detection is not None}
    if not samples and not detections:
        return request
    records: dict[str, SampleEmission] = {}
    components = list(request.project.components)
    if samples:
        from ..evaluation.state import EvaluationState
        from ..optics.gaussian.truncation import QuadratureRuleTooLarge
        from ..validation.pipeline import evaluate, quadrature_refusal

        excitation_request = request.model_copy(update={"project": request.project.model_copy(update={"targets": ()})})
        input_hash = hash_exact(excitation_request.model_dump(mode="json"))
        # (the complete evaluation, as validate(); the Engine build of the caller's state, identified here when it is not yet)
        excitation_state = EvaluationState(build_id=None if engine_state is None else engine_state.build_id)
        result = evaluate(excitation_request, state=excitation_state)
        if engine_state is not None and engine_state.build_id is None:
            engine_state.build_id = excitation_state.build_id
        nodes = quadrature_refusal(result)
        if nodes is not None:  # (the Project's setting, refused by name as the run of the Project refuses it)
            raise LocalRunError([LocalRunProblem(LocalRunErrorCode.PROJECT_INVALID, str(QuadratureRuleTooLarge(nodes)))])
        problems = []
        failed = result.outcome is not None and result.outcome.execution.outcome is ExecutionOutcome.FAILED
        observations = {o.observation_id: o for o in result.observations}
        for instance_id, model in samples.items():
            obs = observations.get(model.excitation.observation_id)
            if failed or obs is None:
                problems.append(_refusal(instance_id, model, "the trace of the Project's other sources did not complete"))
                continue
            power, shares, why = excitation_at(obs, model.excitation.wavelength_nm)
            if why is not None:
                problems.append(_refusal(instance_id, model, why, obs.quantity_state))
                continue
            records[instance_id] = emission_record(model, power, shares, input_hash, obs.power_sampling_resolution_W,  # type: ignore[arg-type]
                                                   instance_id)
        if problems:
            raise LocalRunError(problems)
        components = [emitter_of(c, samples[c.instance_id], records[c.instance_id]) if c.instance_id in records else c
                      for c in components]
    observations = tuple(o.model_copy(update={"detection": engine_detection(detections[o.observation_id], records)})
                         if o.observation_id in detections else o for o in request.project.observations)
    # (validated construction: the emitter and its record are checked against each other, contracts/components.py)
    snapshot = type(request.project).model_validate({**request.project.model_dump(), "components": [c.model_dump() for c in components],
                                                     "observations": [o.model_dump() for o in observations]})
    return request.model_copy(update={"project": snapshot})


def check_excitation(request: ValidationRequest, result: ValidationResult) -> None:
    """Refuse a run whose trace delivers to a sample another excitation than the excitation trace did (module docstring)."""

    observations = {o.observation_id: o for o in result.observations}
    problems = []
    for component in request.project.components:
        record = component.sample_emission
        if record is None:
            continue
        obs = observations.get(record.excitation_observation_id)
        power = None if obs is None else excitation_at(obs, record.excitation_wavelength_nm)[0]
        expected = record.excitation_power_W
        if power is None or abs(power - expected) > EXCITATION_REPEAT_RTOL * max(abs(expected), abs(power)):
            problems.append(LocalRunProblem(
                LocalRunErrorCode.SAMPLE_EXCITATION_NOT_KNOWN,
                f"the run with the samples' emission delivers {power!r} W of {record.excitation_wavelength_nm} nm to observation "
                f"{record.excitation_observation_id}, the excitation trace {expected!r} W (the trace budget or the emission changed "
                f"the excitation's trace): the emission of sample {component.instance_id} is not known",
                f"component {component.instance_id}",
                {"observation_id": record.excitation_observation_id, "wavelength_nm": record.excitation_wavelength_nm,
                 "quantity_state": None if obs is None or obs.quantity_state is None else obs.quantity_state.value}))
    if problems:
        raise LocalRunError(problems)
