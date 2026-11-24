"""What a detector states about itself, for the detector response of an irradiance map (``detector_response``).

Every parameter is optional: what a detector does not state leaves the quantities that need it unavailable and every other
quantity as it is. Nothing is assumed in its place; state ``0`` for a dark current or a read noise you take as negligible.
Where a stated value comes from (``source``, ``characterization``) is optional too: a value stated without it is used as
written, and the result reports its source as not stated.

Frame and units: pixel coordinates are the plane coordinates ``(u, v)`` of the irradiance map, in mm. ``pixel_count`` and
``pixel_pitch_mm`` are ``[along u, along v]``. A pixel's light-sensitive area is a rectangle centred in its cell, of the
cell's aspect ratio, whose area is ``fill_factor`` of the cell. Electrons, ADU and noise follow the statements of
``read_noise_e``, ``dark_current_e_per_s``, ``full_well_e``, ``gain_e_per_adu`` and ``bit_depth``.

``electrical`` (``DetectorElectricalV1``) states a photodetector's continuous electrical response: its bandwidth and its
dark-noise floor (a noise-equivalent power, or a dark current with an optional load resistance). With the quantum
efficiency it gives the photocurrent, the shot, dark and thermal noise currents in that bandwidth and the signal-to-noise
ratio of the detected power, and the noise of a lock-in measurement (``detection.DetectionV1``). Its optional
``saturation_current_A`` states where the photocurrent stops being linear in the power.

Hash evolution: ``electrical`` defaults to ``None``; a detector document that does not state it keeps its exact payload.
``electrical.saturation_current_A`` and the two ``source`` fields default to ``None`` the same way.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from ..base import OmitAbsentFields, StrictContractModel

#: How many rows a quantum-efficiency table may hold.
MAX_QE_ROWS = 4096
#: The largest ADC a detector may state.
MAX_BIT_DEPTH = 32
#: The most pixels per axis of a detector.
MAX_DETECTOR_PIXELS = 4096


class CharacterizationSourceV1(StrictContractModel):
    """Where a stated device value comes from."""

    kind: Literal["DATASHEET", "MEASUREMENT", "CUSTOMER_STATEMENT"] = Field(
        description="DATASHEET: a manufacturer's published value. MEASUREMENT: measured on this device or its type. "
        "CUSTOMER_STATEMENT: your own stated value.")
    reference: str = Field(
        min_length=1, max_length=500,
        description="What identifies the source: the document and its revision, the measurement record, the person or the project.")


class QuantumEfficiencyRowV1(StrictContractModel):
    """The quantum efficiency at one vacuum wavelength."""

    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength in nm.")
    qe: float = Field(ge=0, le=1, description="Electrons per incident photon on the light-sensitive area, 0 to 1.")


class QuantumEfficiencyTableV1(OmitAbsentFields, StrictContractModel):
    """Quantum efficiency over wavelength: linear between rows, unavailable outside them (it is never extrapolated)."""

    # Serialized only when given: a table stated without it has one payload, with or without exclude_none.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("source",)

    rows: tuple[QuantumEfficiencyRowV1, ...] = Field(
        min_length=1, max_length=MAX_QE_ROWS, description="Rows in strictly increasing wavelength. One row holds at its wavelength only.")
    source: CharacterizationSourceV1 | None = Field(None, description="Where the table comes from. Omitted: not stated.")

    @model_validator(mode="after")
    def _increasing(self) -> QuantumEfficiencyTableV1:
        wavelengths = [row.wavelength_nm for row in self.rows]
        if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError("quantum_efficiency rows must be in strictly increasing wavelength")
        return self


#: How a stated bandwidth is defined.
BANDWIDTH_DEFINITIONS = ("FIRST_ORDER_3DB", "NOISE_EQUIVALENT")


class DetectorElectricalV1(OmitAbsentFields, StrictContractModel):
    """The continuous electrical response of a photodetector (a photodiode with its amplifier): bandwidth and dark-noise floor.

    * ``bandwidth_Hz`` with ``bandwidth_definition``: ``FIRST_ORDER_3DB``, the -3 dB frequency ``f_c`` of a first-order
      (single-pole) response ``|H(f)|^2 = 1 / (1 + (f / f_c)^2)``, whose noise-equivalent bandwidth is ``pi f_c / 2``; or
      ``NOISE_EQUIVALENT``, the noise-equivalent bandwidth ``B`` itself (the response's shape is then not stated).
    * The dark-noise floor, stated one way: ``nep_W_per_rtHz`` (the noise-equivalent power of the detector in the dark, in
      W/sqrt(Hz), at ``nep_wavelength_nm``: it holds every noise but the signal's own shot noise, and its current noise
      density is ``NEP R(nep_wavelength_nm)``); or ``dark_current_A`` (its shot noise ``2 q I_d``), with
      ``load_resistance_ohm`` and ``temperature_K`` for the thermal (Johnson) noise ``4 k T / R`` of a load or feedback
      resistance. State 0 for a dark current taken as negligible; leave the resistance out where no resistor sets the noise.
    * The responsivity is the quantum efficiency's: ``R(lambda) = QE(lambda) q lambda / (h c)`` A/W.
    * Linear range: the photocurrent is taken as linear in the power. ``saturation_current_A`` (optional) states the upper end
      of that range (the photodiode's or its amplifier's, whichever is lower): a photocurrent above it is not modelled, and
      the quantities that read it are not stated.

    Not modelled: internal gain and its excess noise (APD, PMT), 1/f noise, amplifier voltage noise, the shape of the
    nonlinearity and of saturation.
    """

    # Serialized only when given: an electrical response stated without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("saturation_current_A", "source")

    bandwidth_Hz: float = Field(gt=0, description="The detector's electrical bandwidth in Hz, as bandwidth_definition says.")
    bandwidth_definition: Literal["FIRST_ORDER_3DB", "NOISE_EQUIVALENT"] = Field(
        description="FIRST_ORDER_3DB: the -3 dB frequency of a single-pole response (noise bandwidth pi/2 times it). "
        "NOISE_EQUIVALENT: the noise-equivalent bandwidth itself.")
    nep_W_per_rtHz: float | None = Field(
        None, ge=0, description="Noise-equivalent power in the dark, W/sqrt(Hz), at nep_wavelength_nm. Give this or dark_current_A.")
    nep_wavelength_nm: float | None = Field(None, gt=0, description="The vacuum wavelength the noise-equivalent power is stated at, nm.")
    dark_current_A: float | None = Field(None, ge=0, description="Dark current in A (0 when negligible). Give this or nep_W_per_rtHz.")
    load_resistance_ohm: float | None = Field(
        None, gt=0, description="Load or feedback resistance whose thermal noise 4kT/R adds (with temperature_K), ohm.")
    temperature_K: float | None = Field(None, gt=0, description="Temperature of that resistance, K.")
    saturation_current_A: float | None = Field(
        None, gt=0, description="Upper end of the linear range of the photocurrent, A (the photodiode's or its amplifier's, "
        "whichever is lower). Omitted: the photocurrent is taken as linear at any power.")
    source: CharacterizationSourceV1 | None = Field(
        None, description="Where the bandwidth, the noise floor and the linear range come from. Omitted: not stated.")

    @model_validator(mode="after")
    def _one_floor(self) -> DetectorElectricalV1:
        if (self.nep_W_per_rtHz is None) == (self.dark_current_A is None):
            raise ValueError("state the dark-noise floor once: nep_W_per_rtHz (with nep_wavelength_nm) or dark_current_A")
        if (self.nep_W_per_rtHz is None) != (self.nep_wavelength_nm is None):
            raise ValueError("nep_W_per_rtHz and nep_wavelength_nm come together: a noise-equivalent power holds at one wavelength")
        if (self.load_resistance_ohm is None) != (self.temperature_K is None):
            raise ValueError("load_resistance_ohm and temperature_K come together: they give the thermal noise 4kT/R")
        if self.load_resistance_ohm is not None and self.nep_W_per_rtHz is not None:
            raise ValueError("a noise-equivalent power already holds the thermal noise: state load_resistance_ohm with dark_current_A only")
        return self


class DetectorModelV1(OmitAbsentFields, StrictContractModel):
    """A detector as stated (module docstring): any parameter may be left out."""

    # Serialized only when given: a document without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("electrical",)

    quantum_efficiency: QuantumEfficiencyTableV1 | None = Field(None, description="Quantum efficiency over wavelength.")
    exposure_s: float | None = Field(None, gt=0, description="Exposure time in s.")
    pixel_pitch_mm: tuple[float, float] | None = Field(None, description="Pixel pitch [along u, along v] in mm.")
    fill_factor: float | None = Field(None, gt=0, le=1, description="Light-sensitive share of a pixel's cell area (with any microlens gain included), 0 to 1.")
    pixel_count: tuple[int, int] | None = Field(
        None, description="Pixels [along u, along v]. Omitted: as many whole pixels as fit in the irradiance grid.")
    center_uv_mm: tuple[float, float] | None = Field(
        None, description="Centre of the pixel array on the plane, [u, v] in mm. Omitted: the centre of the irradiance grid.")
    full_well_e: float | None = Field(None, gt=0, description="Full-well capacity in electrons.")
    read_noise_e: float | None = Field(None, ge=0, description="Read noise in electrons RMS per pixel (0 when negligible).")
    dark_current_e_per_s: float | None = Field(None, ge=0, description="Dark current in electrons per second per pixel (0 when negligible).")
    gain_e_per_adu: float | None = Field(None, gt=0, description="Conversion gain in electrons per ADU.")
    bit_depth: int | None = Field(None, strict=True, ge=1, le=MAX_BIT_DEPTH, description="ADC bits; the largest count is 2^bit_depth - 1.")
    characterization: CharacterizationSourceV1 | None = Field(
        None, description="Where the full well, read noise, dark current and gain come from. Omitted: not stated.")
    electrical: DetectorElectricalV1 | None = Field(
        None, description="The continuous electrical response: bandwidth and dark-noise floor (photocurrent, noise currents, SNR).")

    @model_validator(mode="after")
    def _consistent(self) -> DetectorModelV1:
        if self.pixel_pitch_mm is not None and not all(p > 0 for p in self.pixel_pitch_mm):
            raise ValueError("pixel_pitch_mm must be two positive numbers")
        if self.pixel_count is not None and not all(1 <= n <= MAX_DETECTOR_PIXELS for n in self.pixel_count):
            raise ValueError(f"pixel_count must be two whole numbers from 1 to {MAX_DETECTOR_PIXELS}")
        return self
