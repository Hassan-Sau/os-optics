"""A pupil-plane field and the mask that modulates it, for the scalar low-NA focal field (``pupil_field``).

Frame and signs: the pupil plane has coordinates ``(x, y)`` in mm, its grid centred on ``(0, 0)``; arrays are ``[row][column]``,
one row per ``y``, columns along ``x``. The focal plane has coordinates ``(u, v)`` parallel to ``(x, y)``. A field is
``exp(-i w t)`` with propagation ``exp(+i k z)``, so a phase ``phi(x) = 2 pi x sin(theta) / lambda`` tilts the light toward
``+x`` and focuses it at ``u = +f sin(theta)``. A mask multiplies the field by ``amplitude exp(+i phase_rad)``.

Pixels: ``pitch_mm`` is the grid step. Without ``fill_factor`` the grid samples a continuous pupil. With ``fill_factor`` each
grid cell is a pixel that holds one value across a rectangle centred in the cell, of the cell's aspect ratio and ``fill_factor``
of its area: the far field gains the pixel envelope and the higher diffraction orders of the pixel grid.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..base import StrictContractModel
from .detector import CharacterizationSourceV1

#: The most pixels per axis of a pupil grid and points per axis of its transform.
MAX_PUPIL_SAMPLES = 2048
MAX_FFT_SAMPLES = 4096
#: The most values per axis of a mask written in the request itself; a larger one is a FILE mask.
MAX_INLINE_MASK_SAMPLES = 256
MAX_SPOTS = 1024
MAX_BEAMS = 1024
MAX_MASK_ITERATIONS = 500
DEFAULT_MASK_ITERATIONS = 30
SHA256_PATTERN = r"^[0-9a-f]{64}$"
FILE_NAME_PATTERN = r"^[^/\\:]{1,200}$"


class PupilSamplingV1(StrictContractModel):
    """The pupil grid (module docstring)."""

    pixel_count: tuple[int, int] = Field(description="Grid points [along x, along y].")
    pitch_mm: tuple[float, float] = Field(description="Grid step [along x, along y] in mm: the pixel pitch of a pixelated modulator.")
    fill_factor: float | None = Field(
        None, gt=0, le=1, description="Share of a pixel's cell area that modulates the light. Omitted: the grid samples a continuous pupil, with no pixel structure.")
    fft_size: tuple[int, int] | None = Field(
        None, description="Transform points [along x, along y], at least twice the illuminated span so the focal irradiance is fully sampled. "
        "Omitted: the next power of two at or above twice pixel_count.")

    @model_validator(mode="after")
    def _within_limits(self) -> PupilSamplingV1:
        if not all(1 <= n <= MAX_PUPIL_SAMPLES for n in self.pixel_count):
            raise ValueError(f"pixel_count must be two whole numbers from 1 to {MAX_PUPIL_SAMPLES}")
        if not all(p > 0 for p in self.pitch_mm):
            raise ValueError("pitch_mm must be two positive numbers")
        if self.fft_size is not None and not all(n <= m <= MAX_FFT_SAMPLES for n, m in zip(self.pixel_count, self.fft_size)):
            raise ValueError(f"fft_size must be at least pixel_count and at most {MAX_FFT_SAMPLES} on each axis")
        return self


class PupilCircleV1(StrictContractModel):
    """A round opening centred on the pupil grid; a grid point counts as inside when its centre is."""

    shape: Literal["CIRCLE"]
    diameter_mm: float = Field(gt=0, description="Diameter of the opening in mm.")


class PupilRectangleV1(StrictContractModel):
    """A rectangular opening centred on the pupil grid; a grid point counts as inside when its centre is."""

    shape: Literal["RECTANGLE"]
    width_mm: float = Field(gt=0, description="Full width along x in mm.")
    height_mm: float = Field(gt=0, description="Full height along y in mm.")


PupilApertureV1 = Annotated[PupilCircleV1 | PupilRectangleV1, Field(discriminator="shape")]


class UniformIlluminationV1(StrictContractModel):
    """Uniform irradiance over the opening; ``power_w`` is the power inside it."""

    kind: Literal["UNIFORM"]
    power_w: float = Field(gt=0, description="Power inside the opening in W.")


class GaussianIlluminationV1(StrictContractModel):
    """A collimated Gaussian beam along the axis; ``power_w`` is its whole power, of which the opening passes a part."""

    kind: Literal["GAUSSIAN"]
    power_w: float = Field(gt=0, description="Whole power of the beam in W.")
    radius_mm: tuple[float, float] = Field(description="1/e^2 intensity radii [along x, along y] in mm.")
    center_mm: tuple[float, float] = Field((0.0, 0.0), description="Beam centre [x, y] on the pupil plane in mm.")

    @model_validator(mode="after")
    def _positive(self) -> GaussianIlluminationV1:
        if not all(w > 0 for w in self.radius_mm):
            raise ValueError("radius_mm must be two positive numbers")
        return self


class PupilBeamV1(StrictContractModel):
    """One TEM00 beam crossing the pupil plane: the field of ``results handoff`` (``FIELD_CONVENTION``), its radii along x and y."""

    label: str = Field(min_length=1, max_length=200, description="Name of the beam in the results (a tone, an order, a path id).")
    power_w: float = Field(gt=0, description="Whole power of the beam in W.")
    radius_mm: tuple[float, float] = Field(description="1/e^2 intensity radii [along x, along y] on the pupil plane in mm.")
    center_mm: tuple[float, float] = Field((0.0, 0.0), description="Where the beam's chief ray crosses the pupil plane, [x, y] in mm.")
    direction_cosines: tuple[float, float] = Field((0.0, 0.0), description="Direction of the chief ray along [x, y]: it focuses at u = f l_x, v = f l_y.")
    curvature_radius_mm: tuple[float | None, float | None] = Field(
        (None, None), description="Wavefront curvature radii [along x, along y] in mm, positive when the beam diverges; null is a flat wavefront.")
    phase_rad: float | None = Field(None, description="Field phase at the chief ray in rad; needed when beams add coherently.")

    @model_validator(mode="after")
    def _positive(self) -> PupilBeamV1:
        if not all(w > 0 for w in self.radius_mm):
            raise ValueError("radius_mm must be two positive numbers")
        return self


class BeamsIlluminationV1(StrictContractModel):
    """Several tilted beams on the pupil plane."""

    kind: Literal["BEAMS"]
    beams: tuple[PupilBeamV1, ...] = Field(min_length=1, max_length=MAX_BEAMS, description="The beams.")
    composition: Literal["INCOHERENT", "COHERENT"] = Field(
        description="INCOHERENT: irradiances add, as for beams of different optical frequencies seen over an exposure much longer "
        "than the inverse of their frequency difference (the tones of an acousto-optic deflector). COHERENT: fields add with their stated phases.")


class HandoffBeamsIlluminationV1(StrictContractModel):
    """The beams a run delivers to the pupil plane, read from the ``handoff`` document given with the request."""

    kind: Literal["HANDOFF_BEAMS"]
    composition: Literal["INCOHERENT", "COHERENT"] = Field(
        description="INCOHERENT: irradiances add. COHERENT: fields add with the phases of the handoff, which every beam must define.")
    path_ids: tuple[str, ...] | None = Field(
        None, min_length=1, description="Only these beams of the handoff, by path id (an AOD's tones without its zero order, say). Omitted: every beam of this wavelength.")


class FieldIlluminationV1(StrictContractModel):
    """A sampled complex field on the pupil grid, in a file given with the request: an ``.npz`` archive of ``field_re`` and
    ``field_im`` (``[row][column]``, one row per y, the shape of ``sampling.pixel_count``), the field ``U`` in sqrt(W/mm^2) at
    each grid point (``os-optics results field`` writes the field a run's sampled-field or speckle source delivers to a plane
    in this form). The power is the sum of ``|U|^2`` times the cell area; the opening applies to it."""

    kind: Literal["FIELD"]
    file_name: str = Field(pattern=FILE_NAME_PATTERN, description="Name of the .npz file given with the request.")
    sha256: str = Field(pattern=SHA256_PATTERN, description="SHA-256 of the file's bytes; a file with another hash is refused.")


PupilIlluminationV1 = Annotated[UniformIlluminationV1 | GaussianIlluminationV1 | BeamsIlluminationV1 | HandoffBeamsIlluminationV1
                                | FieldIlluminationV1, Field(discriminator="kind")]


class BlazedRampMaskV1(StrictContractModel):
    """A linear phase ramp ``phi = 2 pi (s_x x + s_y y) / lambda``, sampled at the pixel centres; it steers to ``u = f s_x, v = f s_y``."""

    kind: Literal["BLAZED_RAMP"]
    sin_theta: tuple[float, float] = Field(description="Sine of the deflection [along x, along y].")


class BinaryGratingMaskV1(StrictContractModel):
    """A binary phase grating: ``phase_depth_rad`` over the first ``duty_cycle`` of each period, 0 over the rest, read at the pixel centres."""

    kind: Literal["BINARY_GRATING"]
    period_mm: float = Field(gt=0, description="Grating period in mm.")
    axis: Literal["X", "Y"] = Field(description="Axis the phase varies along.")
    duty_cycle: float = Field(0.5, gt=0, lt=1, description="Share of the period at phase_depth_rad.")
    phase_depth_rad: float = Field(math.pi, description="Phase of the raised part in rad.")
    offset_mm: float = Field(0.0, description="Where a period starts on the axis, in mm.")


class SampledMaskV1(StrictContractModel):
    """A mask written in the request, one value per grid point."""

    kind: Literal["SAMPLED"]
    phase_rad: tuple[tuple[float, ...], ...] = Field(description="Phase per grid point in rad: one row per y.")
    amplitude: tuple[tuple[float, ...], ...] | None = Field(None, description="Amplitude transmission per grid point, 0 to 1. Omitted: 1 everywhere.")
    source: CharacterizationSourceV1 = Field(description="Where the mask comes from.")

    @model_validator(mode="after")
    def _small(self) -> SampledMaskV1:
        if len(self.phase_rad) > MAX_INLINE_MASK_SAMPLES or any(len(row) > MAX_INLINE_MASK_SAMPLES for row in self.phase_rad):
            raise ValueError(f"a mask written in the request holds at most {MAX_INLINE_MASK_SAMPLES} values per axis; a larger one is a FILE mask")
        return self


class FileMaskV1(StrictContractModel):
    """A mask in a file given with the request: an ``.npy`` array of the phase in rad, or an ``.npz`` archive of ``phase_rad`` and optionally ``amplitude``."""

    kind: Literal["FILE"]
    file_name: str = Field(pattern=FILE_NAME_PATTERN, description="Name of the file given with the request.")
    sha256: str = Field(pattern=SHA256_PATTERN, description="SHA-256 of the file's bytes; a file with another hash is refused.")
    source: CharacterizationSourceV1 = Field(description="Where the mask comes from.")


class TargetSpotV1(StrictContractModel):
    """One spot of an array on the focal plane."""

    u_mm: float = Field(description="Position along u in mm.")
    v_mm: float = Field(description="Position along v in mm.")
    weight: float = Field(1.0, gt=0, description="Relative power of the spot.")


class SpotArrayMaskV1(StrictContractModel):
    """A phase mask generated for an array of spots by weighted Gerchberg-Saxton iterations (deterministic: the same request gives the same mask)."""

    kind: Literal["SPOT_ARRAY"]
    spots: tuple[TargetSpotV1, ...] = Field(min_length=1, max_length=MAX_SPOTS, description="The spots.")
    iterations: int = Field(DEFAULT_MASK_ITERATIONS, strict=True, ge=1, le=MAX_MASK_ITERATIONS, description="Weighted Gerchberg-Saxton iterations.")


PupilMaskV1 = Annotated[BlazedRampMaskV1 | BinaryGratingMaskV1 | SampledMaskV1 | FileMaskV1 | SpotArrayMaskV1, Field(discriminator="kind")]


class DeviceEfficiencyV1(StrictContractModel):
    """The share of the modulated light a device delivers into the mask's pattern, as characterized."""

    value: float = Field(gt=0, le=1, description="Delivered share of the light after the mask, 0 to 1.")
    source: CharacterizationSourceV1 = Field(description="Where the value comes from.")


class PupilFieldOutputV1(StrictContractModel):
    """What the reply holds."""

    half_extent_mm: tuple[float, float] | None = Field(
        None, description="Half width of the focal grid [along u, along v] in mm. Omitted: the centred box holding 99 % of the focal power and every target spot.")
    center_uv_mm: tuple[float, float] = Field((0.0, 0.0), description="Centre of the focal grid [u, v] in mm.")
    include_field: bool = Field(False, description="Also return the complex focal field of a coherent result.")
    include_mask: bool = Field(False, description="Also return the mask's phase per grid point.")


class PupilFieldRequestV1(StrictContractModel):
    """A pupil-plane field and its mask (module docstring); the pupil lies in the front focal plane of an ideal lens of ``focal_length_mm``."""

    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength in nm.")
    focal_length_mm: float = Field(gt=0, description="Focal length of the ideal lens forming the focal plane, in mm.")
    sampling: PupilSamplingV1 = Field(description="The pupil grid.")
    aperture: PupilApertureV1 | None = Field(None, description="The opening. Omitted: the whole grid.")
    illumination: PupilIlluminationV1 = Field(description="The light on the pupil plane.")
    mask: PupilMaskV1 | None = Field(None, description="The modulation. Omitted: no mask.")
    device_efficiency: DeviceEfficiencyV1 | None = Field(None, description="A device's characterized efficiency. Omitted: not stated, and the focal power is that of an ideal device.")
    output: PupilFieldOutputV1 = Field(PupilFieldOutputV1(), description="What the reply holds.")

    @model_validator(mode="after")
    def _mask_fits(self) -> PupilFieldRequestV1:
        nx, ny = self.sampling.pixel_count
        if isinstance(self.mask, SampledMaskV1):
            for name, rows in (("phase_rad", self.mask.phase_rad), ("amplitude", self.mask.amplitude)):
                if rows is not None and (len(rows) != ny or any(len(row) != nx for row in rows)):
                    raise ValueError(f"mask.{name} must hold {ny} rows of {nx} values, as sampling.pixel_count says")
        if isinstance(self.mask, SpotArrayMaskV1) and not isinstance(self.illumination, (UniformIlluminationV1, GaussianIlluminationV1)):
            raise ValueError("a SPOT_ARRAY mask is generated for one UNIFORM or GAUSSIAN illumination")
        return self
