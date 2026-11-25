"""Detection contracts: a stated detector on an observation plane, a lock-in measurement of modulated light, the record of a
sample's emission derived from the light that excites it, and their results.

Input (``Observation.detection``, ``Detection``): the detector (``DetectorModelV1``: quantum efficiency, ``electrical``
bandwidth and dark-noise floor) and, optionally, a lock-in (``LockIn``) reading the light of the modulated sources
(``Modulation.sources``: each with the share of its traced power that is modulated -- 1 for a chopped source, the modulated
share of the excitation for a sample's emission). The Project form (``shared_contracts.optical_design.detection``) names
sources; the local runtime resolves the shares.

Sample emission (``SampleEmission``, on the sample's ``EngineeringComponentSnapshot``): the derivation of a fluorescence or
Raman emitter's line powers from the excitation power the Project's other sources deliver to the sample's plane (a first trace
of the Project without its samples). The emitter (``ExtendedEmitter``) carries exactly those line powers.

Result (``ObservationResult.detection``, ``DetectionResult``): every quantity of the detection with its status --
``MODELED`` (computed by the stated model), ``EXTERNALLY_CHARACTERIZED`` (a stated parameter, used as stated) or
``UNAVAILABLE`` (with ``reason`` and ``missing`` parameters) -- the light it reads (``contributions``) and the conditions the
model needs (``validity``). ``validation/detection.py`` states the model.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from .._vendor.shared_contracts.optical_design.detector import DetectorModelV1
from ..errors.status import QuantityState
from .base import Id, OmitAbsentFields, StrictModel
from .basis import BasisCarrier, ValidityCheck, ValueBasisField

# ------------------------------------------------------------------------------------------------- input


class ModulatedSource(StrictModel):
    """A source whose light is modulated, and the share of its traced power that is."""

    source_id: Id
    fraction: float = Field(ge=0.0, le=1.0, description="Share of the source's traced power that follows the modulation.")


class Modulation(StrictModel):
    """The modulation of some sources' light (``shared_contracts.optical_design.detection.ModulationV1`` states the waveforms)."""

    waveform: Literal["SQUARE", "SINE"]
    frequency_Hz: float = Field(gt=0)
    duty_cycle: float | None = Field(None, gt=0, lt=1)
    modulation_depth: float | None = Field(None, gt=0, le=1)
    sources: tuple[ModulatedSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _waveform_parameters(self) -> Modulation:
        if self.waveform == "SQUARE" and (self.duty_cycle is None or self.modulation_depth is not None):
            raise ValueError("a SQUARE modulation states duty_cycle and no modulation_depth")
        if self.waveform == "SINE" and (self.modulation_depth is None or self.duty_cycle is not None):
            raise ValueError("a SINE modulation states modulation_depth and no duty_cycle")
        ids = [s.source_id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("modulated sources must be distinct")
        return self


class LockIn(StrictModel):
    """A lock-in amplifier demodulating the detector current at the modulation frequency."""

    modulation: Modulation
    time_constant_s: float = Field(gt=0)
    filter_order: int = Field(strict=True, ge=1, le=4)


class Detection(OmitAbsentFields, StrictModel):
    """A stated detector on an observation plane (its sensitive face) and, optionally, a lock-in measurement."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("lock_in",)

    detector: DetectorModelV1
    lock_in: LockIn | None = None


class SampleEmissionLine(OmitAbsentFields, StrictModel):
    """One emitted line of a sample: its wavelength, the photons it carries per arriving excitation photon and its power."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("photon_fraction", "raman_shift_cm1")

    wavelength_nm: float = Field(gt=0)
    photon_fraction: float | None = Field(None, description="Fluorescence: the line's share of the emitted photons.")
    raman_shift_cm1: float | None = Field(None, description="Raman: the line's Stokes shift.")
    photons_per_excitation_photon: float = Field(
        ge=0, description="Emitted photons of this line per excitation photon arriving at the plane: absorbed share x quantum yield "
        "x photon fraction (fluorescence), N 4 pi (d sigma / d Omega) L (Raman).")
    energy_ratio: float = Field(gt=0, description="lambda_ex / lambda: the energy of an emitted photon over an excitation photon's.")
    photon_rate_per_s: float = Field(ge=0, description="Emitted photons per second in this line.")
    power_W: float = Field(ge=0, description="Emitted power of this line: excitation power x photons per excitation photon x energy ratio.")


class SampleExcitationShare(StrictModel):
    """The excitation power one source delivers to the sample's plane."""

    source_id: Id
    power_W: float = Field(ge=0)


class SampleEmission(OmitAbsentFields, StrictModel):
    """How a sample's emission follows from its excitation (module docstring). ``validity``: the sample model's validity
    condition where the sample is outside it (the thin, optically dilute fluorescence sample: ``absorbance`` or
    ``absorbed_fraction``; the single-scattering Raman sample: ``scattered_fraction``), with the observed value and
    ``within`` false; every number of the record is then the stated model's value outside its validity limit."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("absorbed_fraction", "quantum_yield", "absorbed_power_W", "excitation_power_resolution_W",
                                                 "validity")

    process: Literal["FLUORESCENCE", "RAMAN"]
    model: str = Field(description="The emission model in words, with its conditions.")
    excitation_observation_id: Id
    excitation_wavelength_nm: float = Field(gt=0)
    excitation_power_W: float = Field(ge=0, description="Power of the excitation wavelength arriving at the plane (KNOWN).")
    excitation_power_resolution_W: float | None = Field(
        None, ge=0, description="Where that power is a sampled value: its sampling resolution (convergence evidence, not a bound).")
    excitation_photon_rate_per_s: float = Field(ge=0)
    excitation_by_source: tuple[SampleExcitationShare, ...] = ()
    excitation_input_hash: str = Field(description="Hash of the Engine input of the trace without samples that gave the excitation.")
    absorbed_fraction: float | None = Field(None, ge=0, le=1)
    quantum_yield: float | None = Field(None, ge=0, le=1)
    absorbed_power_W: float | None = Field(None, ge=0)
    lines: tuple[SampleEmissionLine, ...] = Field(min_length=1)
    emitted_power_W: float = Field(ge=0, description="Total emitted power: the sum of the line powers.")
    emitted_photon_rate_per_s: float = Field(ge=0)
    validity: tuple[ValidityCheck, ...] = ()


# ------------------------------------------------------------------------------------------------- result

#: A quantity's status (``os_optics.results.quantity``).
QuantityStatus = Literal["MODELED", "EXTERNALLY_CHARACTERIZED", "UNAVAILABLE"]


class DetectionQuantity(BasisCarrier, OmitAbsentFields, StrictModel):
    """One quantity of a detection: its value with unit and basis, or why it is not available. ``basis`` is the definition in
    words; ``value_basis["value"]`` says what the number is where the plane's power is not simply the model's value."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("value", "reason", "missing", "sampling_relative_sigma", "value_basis")

    name: str
    status: QuantityStatus
    unit: str | None = None
    value: float | None = None
    basis: str = ""
    reason: str | None = None
    missing: tuple[str, ...] = ()
    sampling_relative_sigma: float | None = Field(
        None, ge=0, description="Where the value reads the sampled rays of extended emitters (a sample's emission): its standard "
        "error relative to the value, propagated to first order from each sampled line's collection_fraction_sigma (an upper "
        "guide, as that sigma is), the lines' errors added linearly. Absent where the value reads no sampled ray.")
    value_basis: ValueBasisField = None


class DetectionContribution(OmitAbsentFields, StrictModel):
    """The light of one source at one wavelength that the detector reads, and its responsivity there."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("responsivity_A_per_W",)

    source_id: str
    wavelength_nm: float
    power_W: float = Field(ge=0, description="Traced power on the plane (the open state of a chopped source).")
    modulated_fraction: float = Field(0.0, ge=0, le=1, description="Share of it that follows the lock-in's modulation (0 without one).")
    responsivity_A_per_W: float | None = Field(None, ge=0, description="QE q lambda / (h c); absent outside the QE table.")


class DetectionValidity(StrictModel):
    """A condition of the model: the parameter, its value, the limit and whether it holds."""

    parameter: str
    value: float
    limit: float
    comparison: Literal["AT_LEAST", "AT_MOST"]
    within: bool
    meaning: str


class DetectionResult(StrictModel):
    """The detection of an observation plane's light: what it reads, every quantity and the model's conditions."""

    model: str
    power_state: QuantityState | None = Field(
        description="The plane's power state. KNOWN, KNOWN_ZERO: the light-dependent quantities are the model's values. BOUNDED, "
        "or UNKNOWN on a plane traced light of known power reaches: they are computed from that traced light, and each "
        "says what its number is in value_basis['value']. Any other state: they are UNAVAILABLE (PLANE_POWER_NOT_KNOWN).")
    contributions: tuple[DetectionContribution, ...] = ()
    quantities: tuple[DetectionQuantity, ...] = ()
    validity: tuple[DetectionValidity, ...] = ()

    def quantity(self, name: str) -> DetectionQuantity | None:
        return next((q for q in self.quantities if q.name == name), None)
