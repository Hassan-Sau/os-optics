"""Detection at an observation plane: a stated detector and, optionally, a modulated measurement read by a lock-in amplifier.

An observation plane that states ``detection`` (``ObservationV1.detection``) is the detector's sensitive face: every watt the
run counts on the plane (on its active area, within its acceptance) reaches the detector, at its own wavelength.

* ``detector`` (``DetectorModelV1``): its quantum efficiency gives the responsivity of each wavelength, ``electrical``
  (``DetectorElectricalV1``) its bandwidth and dark-noise floor. What the detector does not state leaves the quantities that
  need it unavailable.
* ``lock_in`` (``LockInV1``): the light of ``modulation.source_ids`` is modulated (a chopper in their beams, or a modulated
  source) and a lock-in amplifier demodulates the detector's current at the modulation frequency, with the time constant and
  filter order stated. The light of a sample whose excitation is modulated (``FLUORESCENCE`` / ``RAMAN`` sources) carries
  the modulated share of its excitation. All other light on the plane is steady and adds only noise.

Modulation (``ModulationV1``): ``SQUARE``, an on/off chopper of ``duty_cycle`` ``D`` -- the traced power of a modulated
source is its open-state power, its mean ``D`` times that, the peak amplitude of its fundamental ``(2 / pi) sin(pi D)``
times that; ``SINE``, a sinusoidally modulated source of ``modulation_depth`` ``m`` -- the traced power is its mean power, the
peak amplitude of its fundamental ``m`` times that.

Lock-in (``LockInV1``): output filter of ``filter_order`` ``n`` identical RC sections of time constant ``tau`` (6, 12, 18,
24 dB/octave for n = 1..4), whose noise-equivalent bandwidth is ``Gamma(n - 1/2) / (4 sqrt(pi) Gamma(n) tau)``: ``1 / (4 tau)``,
``1 / (8 tau)``, ``3 / (32 tau)``, ``5 / (64 tau)``. The reference is in phase with the signal's fundamental; the output is
the root-mean-square value of that fundamental (``X``).

Hash evolution: ``ObservationV1.detection`` defaults to ``None`` and is serialized only when given (``_omit_when_none``):
every document that does not use it keeps its exact payload and hash -- a ``ProjectDocumentV2``, and also a
``ProjectDocumentV1`` (``content_hash``) and a ``VerificationInputV1`` (``input_fingerprint``), whose payloads keep their
other ``null`` fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..base import StrictContractModel
from ..common.identifiers import InstanceKey
from .detector import DetectorModelV1

#: The orders of a lock-in's output filter: 6, 12, 18 and 24 dB/octave.
MAX_LOCK_IN_FILTER_ORDER = 4
#: How many sources one modulation may name.
MAX_MODULATED_SOURCES = 64


class ModulationV1(StrictContractModel):
    """The modulation of the light of some of the Project's sources (module docstring)."""

    waveform: Literal["SQUARE", "SINE"] = Field(
        description="SQUARE: on/off chopping (the traced power is the open-state power). SINE: sinusoidal modulation (the traced "
        "power is the mean power).")
    frequency_Hz: float = Field(gt=0, description="Modulation (and lock-in reference) frequency in Hz.")
    duty_cycle: float | None = Field(None, gt=0, lt=1, description="SQUARE: the open share of each period, between 0 and 1.")
    modulation_depth: float | None = Field(
        None, gt=0, le=1, description="SINE: the depth m, the power swinging between (1 - m) and (1 + m) times its mean.")
    source_ids: tuple[InstanceKey, ...] = Field(
        min_length=1, max_length=MAX_MODULATED_SOURCES,
        description="The source components whose light is modulated (a sample excited by them carries the modulated share of "
        "its excitation).")

    @model_validator(mode="after")
    def _waveform_parameters(self) -> ModulationV1:
        if self.waveform == "SQUARE" and (self.duty_cycle is None or self.modulation_depth is not None):
            raise ValueError("a SQUARE modulation states duty_cycle and no modulation_depth")
        if self.waveform == "SINE" and (self.modulation_depth is None or self.duty_cycle is not None):
            raise ValueError("a SINE modulation states modulation_depth and no duty_cycle")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("modulation source_ids must be distinct")
        return self


class LockInV1(StrictContractModel):
    """A lock-in amplifier demodulating the detector's current at the modulation frequency (module docstring)."""

    modulation: ModulationV1 = Field(description="The modulated light and its reference frequency.")
    time_constant_s: float = Field(gt=0, description="Time constant tau of each RC section of the output filter, in s.")
    filter_order: int = Field(strict=True, ge=1, le=MAX_LOCK_IN_FILTER_ORDER,
                              description="Number of RC sections of the output filter: 1 to 4 (6 to 24 dB/octave).")


class DetectionV1(StrictContractModel):
    """A stated detector on an observation plane and, optionally, a lock-in measurement of modulated light (module docstring)."""

    detector: DetectorModelV1 = Field(description="The detector: quantum efficiency and electrical response.")
    lock_in: LockInV1 | None = Field(None, description="A lock-in measurement of the modulated light on the plane.")
