"""Pure-optical source models of a user-defined source component (``OpticalComponentInstanceV1.source_model``).

``SourceModelV1`` is one of seven kinds, told apart by ``model_kind``: ``GAUSSIAN_BEAM`` (``GaussianBeamSourceV1``, below),
``EXTENDED_EMITTER`` (``ExtendedEmitterSourceV1``, a finite planar emitter; its class states its frame, units and power
normalization), ``SAMPLED_INTENSITY`` (``SampledIntensitySourceV1``, an arbitrary intensity pattern: an incoherent emitter
whose exitance is a sampled map), ``SAMPLED_FIELD`` (``SampledFieldSourceV1``, a coherent complex field sampled on a grid),
``SPECKLE`` (``SpeckleSourceV1``, coherent light through a seeded random phase screen), and the two sample emissions whose
power derives from the Project's own light arriving at the sample: ``FLUORESCENCE`` (``FluorescenceSourceV1``: absorbed
share, quantum yield, emission spectrum) and ``RAMAN`` (``RamanSourceV1``: Stokes shifts, cross-sections, number density,
path length). The sampled kinds state their grid (``SourceGridV1``), normalization, interpolation, coherence and validity in
their classes; each sample-emission class states its photon-number and energy semantics.

A light source is a user-defined component (for example the user's own laser). The Engine's Gaussian source needs the
wavelength, power, waist size and position, beam quality and polarization; ``GaussianBeamSourceV1`` states every one
of them explicitly, so a simulation never depends on an assumed value. The beam is described in the component's own
local frame (the frame placed by ``nominal_pose``):

* the source emits along local ``+Z`` from the local origin;
* emission: either one spectral line -- ``wavelength_nm`` (vacuum wavelength) and ``power_w`` -- or several lines,
  ``spectral_lines`` (two or more lines, each with its own vacuum ``wavelength_nm`` and ``power_w``, at
  distinct wavelengths), for example a 532 nm + 1064 nm laser, a multi-line ion laser or a combined RGB source.
  Exactly one of the two forms is declared: with ``spectral_lines``, ``wavelength_nm`` and ``power_w`` are absent and
  the source's total power is the sum of the line powers (``total_power_w``). The lines keep the author's order; the
  Engine traces one beam per line (its ``spectral_index`` is the line's position in that order), and every result
  carries the wavelength of its line. A one-line source uses the single-line form, so each source has one written
  form;
* ``waist_position_mm`` is the signed distance from the local origin to the beam waist along local ``+Z``
  (positive: the waist lies downstream; ``0``: the waist, and a flat wavefront, is at the origin);
* ``waist_radius_mm`` is the 1/e^2 intensity radius (not the diameter) along local ``+X``;
  ``waist_radius_y_mm`` is the one along local ``+Y`` and, when omitted, equals ``waist_radius_mm``
  (a round beam). ``m2_y`` likewise defaults to ``m2``. Those two omissions are stated semantics, not
  runtime-chosen values. Waist, ``M^2`` and polarization apply to every spectral line;
* ``polarization`` is linear at ``angle_deg`` from local ``+X`` toward local ``+Y`` (``LinearPolarizationV1``) or
  elliptical (``EllipticalPolarizationV1``: the major axis at ``orientation_deg`` from local ``+X`` toward local ``+Y``
  and the ellipticity angle ``ellipticity_deg``, -45 right-hand and +45 left-hand circular, the sign of the results'
  ``polarization_ellipticity_deg``), or partially polarized: ``PartialPolarizationV1`` (kind ``PARTIAL``: a fully polarized
  part ``polarized``, linear or elliptical, and the degree of polarization ``degree_of_polarization`` ``P`` in ``[0, 1]``)
  or ``UnpolarizedV1`` (kind ``UNPOLARIZED``, ``P = 0``). Partially polarized light is the incoherent sum of two
  orthogonal fully polarized components: the polarized state with ``(1 + P) / 2`` of the power and its orthogonal state
  with ``(1 - P) / 2`` (unpolarized light: local ``+X`` and local ``+Y``, half the power each). The two components are
  mutually incoherent; each is traced as its own beam, and the results add their powers and Stokes parameters;
* ``pulse`` (optional) describes a pulse train (``PulseTrainV1``: repetition rate and FWHM pulse duration).
  ``power_w`` and the line powers are then **average** powers; the energy of one pulse is the average power divided by
  the repetition rate (``pulse_energy_j``). Nothing else in the beam's propagation changes. ``pulse.total_pulses``
  (optional) is how many pulses the source sends into the optical system while the optics stay in place -- a whole
  number, or ``UNLIMITED`` when it keeps pulsing for as long as they are in use; the Engine compares the pulses this
  gives each rated surface with a power rating's pulse count;
* continuous spectrum (``spectral_density``, ``SpectralDensityV1``): instead of lines, a stated spectral density over a
  band -- a Gaussian, Lorentzian or top-hat shape of centre and full width at half maximum, or a tabulated density -- with
  ``power_w`` the power within the band. It is traced as ``line_count`` quadrature lines (the midpoints of equal bins of
  the band, each with the power of its bin, each a Gaussian line as wide as its bin, so that the lines together cover the
  spectrum without gaps); the source's coherence follows from the spectrum (``linewidth_nm`` and
  ``coherence_length_mm`` are absent) and the run reports it with the quadrature basis;
* coherence (optional; at most one of the two): ``linewidth_nm`` is the full width at half maximum of the source's
  (Gaussian) spectrum, ``coherence_length_mm`` the optical path difference at which the degree of coherence of the
  source with itself falls to 1/e (``|gamma| = exp(-(OPD / L_c)^2)``). A linewidth gives
  ``L_c = (2 sqrt(ln 2) / pi) * wavelength^2 / linewidth``. Beams that come from one source component are mutually
  coherent within that length (one laser split into two arms); beams of different source components are mutually
  incoherent. A source without either is taken as monochromatic: the two-beam quantities that depend on the degree of
  coherence (``fringe_visibility``, ``fringe_visibility_peak``, ``peak_irradiance_W_per_mm2``, ``coherence_factor``, the
  coherent power) are stated at that stand-in with a ``value_basis`` entry naming it (assumed ``FULL_COHERENCE``), and
  ``coherence_length_mm`` and ``coherence_margin`` are not reported. Where neither source of the pair states its
  coherence -- in a Project the only such pair, its beams coming from one source component -- the values are at a degree
  of coherence of 1: ``fringe_visibility``, ``fringe_visibility_peak``, ``peak_irradiance_W_per_mm2`` and
  ``coherence_factor`` are upper ends (``UPPER_BOUND``), also behind coatings of bounded group delay, which give such a
  pair no ``coherence_factor_min`` or ``coherence_factor_max``, and a target on one of them is judged from 0 up to that
  value. At the other source's coherence length where that source states it -- a pair of two sources, which only the
  low-level Engine API forms through a shared ``coherence_group`` (in a Project, beams of different source components
  are incoherent) -- the values are at that length, and a target is judged from 0 up to that value; where the two paths'
  group-delay dispersions differ, or behind coatings of bounded group delay, they are ``ESTIMATE`` values at the traced
  delay, the pair states ``coherence_factor_max`` (the largest degree of coherence any shorter coherence length of the
  unstated source gives; 1 behind the bounded coatings), and a target is judged from 0 up to the value at the pair's
  ``coherence_factor_max``. The coherent power of the beams of one wavelength on a plane (``coherent_power_W``,
  ``CoherentSuperposition.total_power_W``) at a stand-in degree of coherence of 1 is an ``ESTIMATE`` (assumed
  ``FULL_COHERENCE``); a target on it is judged on ``total_power_min_W`` .. ``total_power_max_W``, the range every degree
  of coherence from 0 to 1 gives -- each cross term between 0 and its value, so the range lies between the beams'
  incoherent power sum and the value --, widened by light the trace does not follow. For a two-beam pair at the other
  source's stated coherence length (a shared ``coherence_group`` only), the collinear combined power
  (``combined_power_collinear_W``) is an ``ESTIMATE`` judged from 0 up to the larger of the value and the beams'
  incoherent power sum. A pair with a pulse train among its sources where
  neither source states its coherence leaves those quantities not computed (its pulses' overlap needs a stated
  coherence). Metrics that do not depend on coherence (fringe period, path
  difference, overlap, ...) are unaffected. With ``spectral_lines``, the declared coherence describes every line: ``linewidth_nm`` is the width of
  each line (each line's coherence length follows from its own wavelength), ``coherence_length_mm`` the coherence
  length of each line. Lines of one source have different frequencies and are mutually incoherent: only beams of the
  same line interfere.

Hash evolution: the component field that carries this model (``OpticalComponentInstanceV1.source_model``)
defaults to ``None``, so every Project written before it existed keeps its exact payload and osc1 hash; the same holds
for the optional coherence fields, ``spectral_lines``, ``spectral_density``, ``pulse`` and ``pulse.total_pulses``, which
default to ``None``
(a single-line source keeps its exact payload and hash, a pulse train without ``total_pulses`` its exact payload). A linear polarization written without ``kind`` still reads as ``LINEAR``. The polarization kinds ``PARTIAL`` and ``UNPOLARIZED`` are new members of the polarization union: a Project that does not use them keeps its exact payload and hash. The source kinds ``SAMPLED_INTENSITY``, ``SAMPLED_FIELD``, ``SPECKLE``, ``FLUORESCENCE`` and ``RAMAN`` are new members of the source union, with the same rule.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag, model_validator

from ..base import StrictContractModel
from ..common.identifiers import OpenSciId
from .compute_budget import MAX_EXACT_JSON_INTEGER

#: The smallest positive normal float64: a field's peak irradiance |U|^2 must reach it (below it, squares lose precision and
#: underflow to 0).
MIN_NORMAL_FLOAT = 2.2250738585072014e-308


class LinearPolarizationV1(StrictContractModel):
    """Linear polarization at ``angle_deg`` from the component's local +X toward local +Y."""

    kind: Literal["LINEAR"] = "LINEAR"
    angle_deg: float = Field(ge=-180.0, le=180.0)


class EllipticalPolarizationV1(StrictContractModel):
    """Elliptical polarization (circular at ``ellipticity_deg = +-45``).

    ``orientation_deg``: the angle of the ellipse's major axis from the component's local +X toward local +Y, in
    ``(-90, 90]``. ``ellipticity_deg``: the ellipticity angle ``chi`` in ``[-45, 45]``, ``tan(|chi|)`` = minor / major
    axis (0: linear, +-45: circular), with the sign of the normalized Stokes parameter ``s3 = sin(2 chi)``, as the
    Engine reports an arriving beam's ``polarization_ellipticity_deg``: a source of ellipticity ``chi`` arrives with
    ``chi`` through free space. Handedness: negative ``chi`` is right-handed and positive ``chi`` left-handed for an
    observer looking toward the source (against the propagation direction), with the time dependence
    ``exp(-i omega t)`` of the field; ``-45`` is right-hand circular, ``+45`` left-hand circular. In the source frame
    (propagation along local +Z) the normalized Jones vector is ``(cos(theta) cos(chi) - i sin(theta) sin(chi),
    sin(theta) cos(chi) + i cos(theta) sin(chi))`` with ``theta = orientation_deg``: right-hand circular (``chi = -45``)
    is ``(1, -i) / sqrt(2)``, whose field turns from local +X toward local -Y.
    """

    kind: Literal["ELLIPTICAL"]
    orientation_deg: float = Field(gt=-90.0, le=90.0)
    ellipticity_deg: float = Field(ge=-45.0, le=45.0)


def _polarization_kind(value: Any) -> str:
    """The discriminator of a source polarization; a linear polarization written without ``kind`` reads as LINEAR."""

    kind = value.get("kind", "LINEAR") if isinstance(value, dict) else getattr(value, "kind", "LINEAR")
    return kind if kind in SOURCE_POLARIZATION_TAGS else "LINEAR"


def _polarized_kind(value: Any) -> str:
    """The discriminator of the fully polarized part of a partial polarization (LINEAR when ``kind`` is omitted)."""

    kind = value.get("kind", "LINEAR") if isinstance(value, dict) else getattr(value, "kind", "LINEAR")
    return kind if kind in ("LINEAR", "ELLIPTICAL") else "LINEAR"


#: A fully polarized state: linear or elliptical.
FullPolarizationV1 = Annotated[
    Annotated[LinearPolarizationV1, Tag("LINEAR")] | Annotated[EllipticalPolarizationV1, Tag("ELLIPTICAL")],
    Discriminator(_polarized_kind),
]


class PartialPolarizationV1(StrictContractModel):
    """Partially polarized light of degree of polarization ``degree_of_polarization`` ``P`` (``0``: unpolarized, ``1``:
    fully polarized in the state ``polarized``).

    Its Stokes vector is ``S0 (1, P s1, P s2, P s3)``, with ``(s1, s2, s3)`` the normalized Stokes parameters of the fully
    polarized state ``polarized`` (linear or elliptical, as a fully polarized source states it). It is traced as the
    incoherent sum of two orthogonal fully polarized components: ``polarized`` with ``(1 + P) / 2`` of the power and the
    orthogonal state (the opposite point of the Poincare sphere) with ``(1 - P) / 2``. The two components are mutually
    incoherent: no interference term exists between them. The decomposition is exact for every element the Engine models
    with a Jones matrix (no element depolarizes light)."""

    kind: Literal["PARTIAL"]
    polarized: FullPolarizationV1 = Field(
        description="The fully polarized part: linear (kind LINEAR, the meaning when kind is omitted) or elliptical (kind "
        "ELLIPTICAL), in the source's local frame.")
    degree_of_polarization: float = Field(
        ge=0.0, le=1.0, description="Degree of polarization P in [0, 1]: the polarized power over the total power. 1 is the "
        "fully polarized state, 0 unpolarized light.")


class UnpolarizedV1(StrictContractModel):
    """Unpolarized light (degree of polarization 0): the incoherent sum of equal powers linearly polarized along the
    source's local +X and local +Y. Its Stokes vector is ``S0 (1, 0, 0, 0)`` whatever pair of orthogonal states is chosen."""

    kind: Literal["UNPOLARIZED"]


#: The kinds of a source polarization (``SourcePolarizationV1``).
SOURCE_POLARIZATION_TAGS = ("LINEAR", "ELLIPTICAL", "PARTIAL", "UNPOLARIZED")

SourcePolarizationV1 = Annotated[
    Annotated[LinearPolarizationV1, Tag("LINEAR")] | Annotated[EllipticalPolarizationV1, Tag("ELLIPTICAL")]
    | Annotated[PartialPolarizationV1, Tag("PARTIAL")] | Annotated[UnpolarizedV1, Tag("UNPOLARIZED")],
    Discriminator(_polarization_kind),
]


class PulseTrainV1(StrictContractModel):
    """A pulsed source: pulses of full width at half maximum ``pulse_duration_s`` (``duration_definition = FWHM``) at
    ``repetition_rate_hz``. The source's ``power_w`` / line powers are average powers; the pulse energy is the average
    power divided by the repetition rate. The pulse duration is shorter than the pulse period.

    ``total_pulses`` (optional): how many pulses the source sends into the optical system while the optics stay in
    place -- a whole number from 1 (a single shot), or ``UNLIMITED`` when it keeps pulsing for as long as they are in
    use. A surface receives them once for every crossing of this source's light; a power rating that states a pulse
    count (pulses per tested site) is compared with that. Exposures (pulse energy, peak fluence) do not depend on it."""

    repetition_rate_hz: float = Field(gt=0, description="Pulse repetition rate in Hz.")
    pulse_duration_s: float = Field(gt=0, description="Pulse duration in s (full width at half maximum of the intensity).")
    duration_definition: Literal["FWHM"] = Field(
        description="How pulse_duration_s is measured: FWHM, the full width at half maximum of the pulse's intensity.")
    total_pulses: Annotated[int, Field(strict=True, ge=1, le=MAX_EXACT_JSON_INTEGER)] | Literal["UNLIMITED"] | None = Field(
        None,
        description="How many pulses this source sends into the optical system while the optics stay in place: a whole "
        "number (1 for a single shot), or UNLIMITED when it keeps pulsing for as long as they are in use. A surface receives "
        "them once for every crossing of this source's light. It is compared with the pulse count of a power rating (pulses "
        "per tested site); omitted, that count cannot be checked against this source's light. Exposures do not depend on it.",
    )

    @model_validator(mode="after")
    def _shorter_than_period(self) -> PulseTrainV1:
        if self.pulse_duration_s * self.repetition_rate_hz >= 1.0:
            raise ValueError("pulse_duration_s must be shorter than the pulse period 1 / repetition_rate_hz")
        return self


#: Kept for import compatibility; None means no fixed source-line count ceiling. Project.compute_budget
#: bounds traced paths and steps, independently of how many lines the source declares.
MAX_SPECTRAL_LINES: int | None = None


class SpectralLineV1(StrictContractModel):
    """One emission line of a multi-line source: its vacuum wavelength and its optical power."""

    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength of the line, in nm.")
    power_w: float = Field(gt=0, description="Optical power of the line, in W.")


#: The most quadrature lines one continuous spectrum is traced with.
MAX_DENSITY_LINES = 256
#: The most rows of a tabulated spectral density.
MAX_DENSITY_ROWS = 4096
#: The band a Gaussian / Lorentzian density is traced over when none is stated: the centre +- this many FWHM.
DEFAULT_BAND_HALF_WIDTH_FWHM = {"GAUSSIAN": 1.5, "LORENTZIAN": 10.0}


class GaussianSpectrumV1(StrictContractModel):
    """A Gaussian spectral density over the vacuum wavelength: ``exp(-4 ln 2 (lambda - center)^2 / fwhm^2)``."""

    kind: Literal["GAUSSIAN"]
    center_nm: float = Field(gt=0, description="Centre vacuum wavelength, in nm.")
    fwhm_nm: float = Field(gt=0, description="Full width at half maximum of the density, in nm.")


class LorentzianSpectrumV1(StrictContractModel):
    """A Lorentzian spectral density over the vacuum wavelength: ``1 / (1 + (2 (lambda - center) / fwhm)^2)``."""

    kind: Literal["LORENTZIAN"]
    center_nm: float = Field(gt=0, description="Centre vacuum wavelength, in nm.")
    fwhm_nm: float = Field(gt=0, description="Full width at half maximum of the density, in nm.")


class TopHatSpectrumV1(StrictContractModel):
    """A flat spectral density from ``center - width / 2`` to ``center + width / 2`` (zero outside)."""

    kind: Literal["TOP_HAT"]
    center_nm: float = Field(gt=0, description="Centre vacuum wavelength, in nm.")
    fwhm_nm: float = Field(gt=0, description="Full width of the flat density, in nm.")


class TabulatedSpectrumV1(StrictContractModel):
    """A tabulated spectral density (power per nm, in any unit: ``power_w`` sets the scale), linear in wavelength between
    its rows and zero outside them."""

    kind: Literal["TABULATED"]
    wavelength_nm: tuple[float, ...] = Field(min_length=2, max_length=MAX_DENSITY_ROWS,
                                             description="Vacuum wavelengths of the rows, strictly increasing, in nm.")
    density: tuple[float, ...] = Field(min_length=2, max_length=MAX_DENSITY_ROWS,
                                       description="Relative spectral density at each row (>= 0; linear between rows).")

    @model_validator(mode="after")
    def _rows(self) -> TabulatedSpectrumV1:
        if len(self.wavelength_nm) != len(self.density):
            raise ValueError("wavelength_nm and density must have the same number of rows")
        if self.wavelength_nm[0] <= 0 or any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
            raise ValueError("wavelength_nm must be positive and strictly increasing")
        if any(not (math.isfinite(d) and d >= 0) for d in self.density) or not any(d > 0 for d in self.density):
            raise ValueError("density must be finite, at least 0 and positive somewhere")
        return self


SpectralShapeV1 = Annotated[GaussianSpectrumV1 | LorentzianSpectrumV1 | TopHatSpectrumV1 | TabulatedSpectrumV1,
                            Field(discriminator="kind")]


class SpectralDensityV1(StrictContractModel):
    """A continuous spectrum: the density ``shape`` over the vacuum wavelength, traced over ``band_nm`` (the source's
    ``power_w`` is the power within it) with ``line_count`` quadrature lines.

    ``quadrature`` ``MIDPOINT_BINS``: the band is divided into ``line_count`` equal bins; each line lies at its bin's centre
    with the power of its bin (the exact integral of the density over the bin, scaled so that the lines sum to
    ``power_w``) and the linewidth of its bin's width (a Gaussian line: together the lines cover the spectrum without the
    revivals of a comb of sharp lines). Every bin must carry power. ``band_nm`` absent: a top-hat its own width, a
    tabulated density its rows' range, a Gaussian the centre +- 1.5 FWHM, a Lorentzian the centre +- 10 FWHM. A power a
    run sums over the lines (a plane's power, a power budget) is the midpoint quadrature of the band's integral."""

    shape: SpectralShapeV1
    band_nm: tuple[float, float] | None = Field(
        None, description="[min, max] vacuum wavelengths the spectrum is traced over, in nm; power_w is the power in it.")
    line_count: int = Field(ge=2, le=MAX_DENSITY_LINES, description="Number of quadrature lines (equal bins of the band).")
    quadrature: Literal["MIDPOINT_BINS"] = Field(
        "MIDPOINT_BINS", description="MIDPOINT_BINS: one line at the centre of each of line_count equal bins, with its bin's power.")

    @model_validator(mode="after")
    def _band(self) -> SpectralDensityV1:
        low, high = self.band()
        if not 0 < low < high:
            raise ValueError("band_nm must be [min, max] with 0 < min < max")
        shape = self.shape
        if isinstance(shape, TopHatSpectrumV1):
            if low < shape.center_nm - shape.fwhm_nm / 2 or high > shape.center_nm + shape.fwhm_nm / 2:
                raise ValueError("a top-hat spectrum's band lies inside the top hat (every bin carries power)")
        if isinstance(shape, TabulatedSpectrumV1):
            if low < shape.wavelength_nm[0] or high > shape.wavelength_nm[-1]:
                raise ValueError("a tabulated spectrum's band lies inside its rows")
            if any(w <= 0 for w in self.bin_weights()):
                raise ValueError("every quadrature bin must carry power: the tabulated density is zero over a whole bin")
        return self

    def band(self) -> tuple[float, float]:
        """The traced band ``[min, max]`` in nm (stated, or by the default rule)."""

        if self.band_nm is not None:
            return self.band_nm
        shape = self.shape
        if isinstance(shape, TabulatedSpectrumV1):
            return shape.wavelength_nm[0], shape.wavelength_nm[-1]
        half = shape.fwhm_nm / 2 if isinstance(shape, TopHatSpectrumV1) else DEFAULT_BAND_HALF_WIDTH_FWHM[shape.kind] * shape.fwhm_nm
        return shape.center_nm - half, shape.center_nm + half

    def bin_width_nm(self) -> float:
        low, high = self.band()
        return (high - low) / self.line_count

    def nodes_nm(self) -> tuple[float, ...]:
        """The quadrature lines' wavelengths: the centres of ``line_count`` equal bins of the band."""

        low, high = self.band()
        width = (high - low) / self.line_count
        return tuple(low + (k + 0.5) * width for k in range(self.line_count))

    def bin_weights(self) -> tuple[float, ...]:
        """The integral of the density over each bin (relative; the run's own lines use the Engine's evaluation)."""

        low, high = self.band()
        width = (high - low) / self.line_count
        edges = [low + k * width for k in range(self.line_count)] + [high]
        shape = self.shape
        if isinstance(shape, GaussianSpectrumV1):
            scale = 2.0 * math.sqrt(math.log(2.0)) / shape.fwhm_nm
            values = [math.erf(scale * (e - shape.center_nm)) for e in edges]
        elif isinstance(shape, LorentzianSpectrumV1):
            values = [math.atan(2.0 * (e - shape.center_nm) / shape.fwhm_nm) for e in edges]
        elif isinstance(shape, TopHatSpectrumV1):
            values = edges
        else:
            values = [_tabulated_cumulative(shape, e) for e in edges]
        return tuple(b - a for a, b in zip(values, values[1:]))


def _tabulated_cumulative(shape: TabulatedSpectrumV1, wavelength_nm: float) -> float:
    """The integral of a tabulated density (linear between rows) from its first row to ``wavelength_nm``."""

    total = 0.0
    rows = list(zip(shape.wavelength_nm, shape.density))
    for (w0, d0), (w1, d1) in zip(rows, rows[1:]):
        if wavelength_nm <= w0:
            break
        end = min(wavelength_nm, w1)
        slope = (d1 - d0) / (w1 - w0)
        total += (end - w0) * (d0 + 0.5 * slope * (end - w0))
    return total


class GaussianBeamSourceV1(StrictContractModel):
    """A Gaussian beam of one or several spectral lines emitted along the component's local +Z from its local origin.

    Emission is declared once: ``wavelength_nm`` + ``power_w`` (one line) or ``spectral_lines`` (two or more
    lines at distinct wavelengths; then ``wavelength_nm`` and ``power_w`` are absent and the
    total power is the sum of the line powers). Every other physics field except ``waist_radius_y_mm``, ``m2_y`` and the
    coherence fields is required (nothing is defaulted); ``waist_radius_y_mm`` and ``m2_y``, when omitted, equal
    ``waist_radius_mm`` and ``m2`` (a round beam). Waist, beam quality, polarization and coherence apply to every line.
    Waist radii are 1/e^2 intensity radii along local +X / +Y; ``waist_position_mm`` is the signed distance from the
    local origin to the waist along local +Z (0: the waist and a flat wavefront are at the origin).

    Coherence: declare at most one of ``linewidth_nm`` or ``coherence_length_mm`` (module docstring). Omitted, the
    source's coherence is undefined and coherence-dependent two-beam metrics of its beams are not computed.

    Continuous spectrum: ``spectral_density`` (``SpectralDensityV1``) with ``power_w`` the power within its band, instead of
    ``wavelength_nm`` / ``spectral_lines``; its coherence follows from the spectrum (no ``linewidth_nm`` /
    ``coherence_length_mm``).

    ``pulse``: a pulse train (``PulseTrainV1``); the powers are then average powers. Absent: a continuous-wave source."""

    model_kind: Literal["GAUSSIAN_BEAM"]
    wavelength_nm: float | None = Field(
        None, gt=0, description="Vacuum wavelength in nm of a single-line source; absent when spectral_lines is given."
    )
    power_w: float | None = Field(
        None, gt=0, description="Optical power in W of a single-line source; absent when spectral_lines is given (the total "
        "power is then the sum of the line powers)."
    )
    spectral_lines: tuple[SpectralLineV1, ...] | None = Field(
        None, min_length=2,
        description="Emission lines of a multi-line source (two or more, distinct wavelengths, author's order), instead of "
        "wavelength_nm and power_w. Lines of one source are mutually incoherent; the coherence fields apply to every line.",
    )
    spectral_density: SpectralDensityV1 | None = Field(
        None, description="A continuous spectrum (a density shape over a band, traced as quadrature lines), instead of "
        "wavelength_nm or spectral_lines; power_w is then the power within its band and the coherence follows from it.")
    waist_radius_mm: float = Field(gt=0, description="1/e^2 intensity radius along local +X, in mm (not the diameter).")
    waist_radius_y_mm: float | None = Field(None, gt=0, description="1/e^2 radius along local +Y; omitted means equal to waist_radius_mm.")
    waist_position_mm: float = Field(description="Signed waist position along local +Z from the local origin, in mm (positive = downstream).")
    m2: float = Field(ge=1.0, description="Beam quality factor M^2 along local +X.")
    m2_y: float | None = Field(None, ge=1.0, description="M^2 along local +Y; omitted means equal to m2.")
    polarization: SourcePolarizationV1 = Field(
        description="Linear (kind LINEAR, the meaning when kind is omitted) or elliptical (kind ELLIPTICAL) polarization in the "
        "source's local frame; or partially polarized light (kind PARTIAL: a linear or elliptical polarized part and the "
        "degree of polarization) or unpolarized light (kind UNPOLARIZED).")
    linewidth_nm: float | None = Field(
        None, gt=0,
        description="Spectral linewidth: full width at half maximum of the (Gaussian) source spectrum, in nm (of every line "
        "of a multi-line source). Sets the coherence length L_c = (2 sqrt(ln 2) / pi) * wavelength^2 / linewidth (per line, "
        "from the line's own wavelength). Mutually exclusive with coherence_length_mm; "
        "omitted with it, the source coherence is undefined and coherence-dependent two-beam metrics are not computed.",
    )
    coherence_length_mm: float | None = Field(
        None, gt=0,
        description="Coherence length in mm: the optical path difference at which the degree of coherence |gamma| falls to "
        "1/e (|gamma| = exp(-(OPD / L_c)^2)); of every line of a multi-line source. Mutually exclusive with linewidth_nm; "
        "omitted with it, the source coherence is "
        "undefined and coherence-dependent two-beam metrics are not computed.",
    )
    pulse: PulseTrainV1 | None = Field(
        None, description="Pulse train of a pulsed source (repetition rate, FWHM duration and, optionally, the total number "
        "of pulses); power_w and line powers are then average powers. Absent: a continuous-wave source.")

    @model_validator(mode="after")
    def _one_coherence_measure(self) -> GaussianBeamSourceV1:
        if self.linewidth_nm is not None and self.coherence_length_mm is not None:
            raise ValueError("declare the source coherence once: linewidth_nm or coherence_length_mm, not both")
        return self

    @model_validator(mode="after")
    def _one_emission_form(self) -> GaussianBeamSourceV1:
        if self.spectral_density is not None:
            if self.wavelength_nm is not None or self.spectral_lines is not None or self.power_w is None:
                raise ValueError("a continuous spectrum states spectral_density and power_w (the power in its band), without "
                                 "wavelength_nm or spectral_lines")
            if self.linewidth_nm is not None or self.coherence_length_mm is not None:
                raise ValueError("a continuous spectrum's coherence follows from the spectrum: omit linewidth_nm and "
                                 "coherence_length_mm")
            return self
        if self.spectral_lines is not None:
            if self.wavelength_nm is not None or self.power_w is not None:
                raise ValueError("declare the emission once: wavelength_nm and power_w (one line) or spectral_lines (several "
                                 "lines), not both; with spectral_lines the total power is the sum of the line powers")
            wavelengths = [line.wavelength_nm for line in self.spectral_lines]
            if len(set(wavelengths)) != len(wavelengths):
                raise ValueError("spectral_lines must have distinct wavelengths")
        elif self.wavelength_nm is None or self.power_w is None:
            raise ValueError("a source needs wavelength_nm and power_w (one line) or spectral_lines (several lines)")
        return self

    @property
    def emission_lines(self) -> tuple[SpectralLineV1, ...]:
        """The source's spectral lines in the author's order: ``spectral_lines``, the one line of a single-line source, or
        the quadrature lines of a continuous spectrum (``power_w`` shared in proportion to the bins' densities; the run's
        own lines are the Engine's evaluation of the same rule)."""

        if self.spectral_density is not None:
            weights = self.spectral_density.bin_weights()
            total = math.fsum(weights)
            return tuple(SpectralLineV1(wavelength_nm=w, power_w=self.power_w * weight / total)  # type: ignore[operator]
                         for w, weight in zip(self.spectral_density.nodes_nm(), weights))
        if self.spectral_lines is not None:
            return self.spectral_lines
        return (SpectralLineV1(wavelength_nm=self.wavelength_nm, power_w=self.power_w),)  # type: ignore[arg-type]

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        """The vacuum wavelength of every spectral line, in the author's order."""

        if self.spectral_density is not None:
            return self.spectral_density.nodes_nm()
        return tuple(line.wavelength_nm for line in self.emission_lines)

    @property
    def total_power_w(self) -> float:
        """The source's total optical power: ``power_w`` (of a continuous spectrum: in its band), or the correctly rounded
        sum of the line powers."""

        if self.spectral_lines is None:
            return self.power_w  # type: ignore[return-value]
        return math.fsum(line.power_w for line in self.spectral_lines)

    @property
    def pulse_energy_j(self) -> float | None:
        """Energy of one pulse of a pulsed source (``total_power_w / repetition_rate_hz``), or None for a CW source."""

        return None if self.pulse is None else self.total_power_w / self.pulse.repetition_rate_hz

    @property
    def coherence_declared(self) -> bool:
        """Whether the source's coherence is known: declared (``linewidth_nm`` or ``coherence_length_mm``) or following
        from a continuous spectrum (``spectral_density``)."""

        return self.linewidth_nm is not None or self.coherence_length_mm is not None or self.spectral_density is not None


#: The sample counts one spectral line of an extended emitter may declare: powers of two from this low to this high.
EMITTER_SAMPLE_COUNT_MIN = 16
EMITTER_SAMPLE_COUNT_MAX = 65536


class EmitterRectangleV1(StrictContractModel):
    """A rectangular emitting surface centred on the local origin in the local XY plane."""

    shape: Literal["RECTANGLE"]
    width_mm: float = Field(gt=0, description="Full width along local +X, in mm.")
    height_mm: float = Field(gt=0, description="Full height along local +Y, in mm.")

    @property
    def area_mm2(self) -> float:
        """The emitting area in mm^2."""

        return self.width_mm * self.height_mm


class EmitterDiscV1(StrictContractModel):
    """A circular emitting surface centred on the local origin in the local XY plane."""

    shape: Literal["DISC"]
    diameter_mm: float = Field(gt=0, description="Full diameter in mm (not the radius).")

    @property
    def area_mm2(self) -> float:
        """The emitting area in mm^2."""

        return math.pi * self.diameter_mm * self.diameter_mm / 4.0


EmittingSurfaceV1 = Annotated[EmitterRectangleV1 | EmitterDiscV1, Field(discriminator="shape")]


class EmitterSamplingV1(StrictContractModel):
    """How an extended emitter is sampled by rays: ``sample_count`` rays of equal weight per spectral line, from a
    deterministic low-discrepancy sequence that ``seed`` shifts. The same Project gives the same rays on every platform."""

    sample_count: int = Field(
        strict=True, ge=EMITTER_SAMPLE_COUNT_MIN, le=EMITTER_SAMPLE_COUNT_MAX,
        description="Rays per spectral line: a power of two from 16 to 65536. Each carries 1 / sample_count of the line's "
        "power. Results also report the estimate from the first half of the rays, which shows the sampling error.",
    )
    seed: int = Field(
        strict=True, ge=0, le=MAX_EXACT_JSON_INTEGER,
        description="Shift of the sampling sequence (a digital shift of every coordinate of one stratified point set). Each "
        "seed gives an unbiased estimate of the same quantity; two seeds' estimates differ by the stratified sampling error, "
        "typically a few times 1 / sample_count: comparable to collection_fraction_sigma at a few thousand samples (at times "
        "above it), smaller than it at larger counts, and may coincide "
        "where the result depends on one or two sampled coordinates.",
    )

    @model_validator(mode="after")
    def _power_of_two(self) -> EmitterSamplingV1:
        if self.sample_count & (self.sample_count - 1):
            raise ValueError("sample_count must be a power of two (16, 32, ..., 65536)")
        return self


class ExtendedEmitterSourceV1(StrictContractModel):
    """A finite planar emitter of incoherent, unpolarized light in the component's local frame (placed by ``nominal_pose``).

    * Surface: ``emitting_surface`` is a rectangle or a disc centred on the local origin in the local XY plane (``z = 0``).
      Its emission is uniform over that area. It is an emitter only: it does not stop, reflect or absorb light.
    * Spectrum: ``wavelength_nm`` + ``power_w`` (one line) or ``spectral_lines`` (two or more lines at distinct vacuum
      wavelengths), as on a Gaussian source. Each line is sampled on its own and lines add incoherently.
    * Power normalization: the authoritative quantity of each line is its total radiant power in W, emitted into the
      law's whole domain below. The radiance or intensity is derived and reported with the results.
    * ``LAMBERTIAN``: uniform radiance ``L`` over the hemisphere around local +Z, ``d^2 P = L cos(theta) dOmega dA``
      (``theta`` from local +Z, ``dOmega`` in steradians). The projected solid angle of the hemisphere is ``pi`` sr, so
      ``L = P / (pi A)``, reported in W/(m^2 sr).
    * ``ISOTROPIC``: uniform intensity over the whole sphere (both sides of the surface, ``4 pi`` sr of solid angle
      ``dOmega``), ``d^2 P = P / (4 pi A) dOmega dA``, so ``I = P / (4 pi)``, reported in W/sr. It is a different law from
      ``LAMBERTIAN``: the two give different collected power from one geometry.
    * Composition: rays are independent samples. Their powers add on every plane; no phase relation exists between them,
      so results of this source hold no interference or coherent sum, and no Gaussian beam quantity.
    """

    model_kind: Literal["EXTENDED_EMITTER"]
    wavelength_nm: float | None = Field(
        None, gt=0, description="Vacuum wavelength in nm of a single-line emitter; absent when spectral_lines is given.")
    power_w: float | None = Field(
        None, gt=0, description="Total radiant power in W of a single-line emitter, into the angular law's whole domain "
        "(hemisphere for LAMBERTIAN, sphere for ISOTROPIC); absent when spectral_lines is given.")
    spectral_lines: tuple[SpectralLineV1, ...] | None = Field(
        None, min_length=2,
        description="Emission lines (two or more, distinct wavelengths, author's order), instead of wavelength_nm and "
        "power_w. Each power_w is that line's total radiant power into the angular law's whole domain.",
    )
    emitting_surface: EmittingSurfaceV1 = Field(
        description="Emitting area in the local XY plane, centred on the local origin: shape RECTANGLE (width_mm, height_mm) "
        "or DISC (diameter_mm).")
    angular_law: Literal["LAMBERTIAN", "ISOTROPIC"] = Field(
        description="LAMBERTIAN: uniform radiance over the hemisphere around local +Z (power per solid angle falls with "
        "cos(theta)). ISOTROPIC: uniform intensity over the whole sphere, both sides of the surface.")
    sampling: EmitterSamplingV1 = Field(description="Rays per spectral line and the shift of their sequence.")

    @model_validator(mode="after")
    def _one_emission_form(self) -> ExtendedEmitterSourceV1:
        if self.spectral_lines is not None:
            if self.wavelength_nm is not None or self.power_w is not None:
                raise ValueError("declare the emission once: wavelength_nm and power_w (one line) or spectral_lines (several "
                                 "lines), not both; with spectral_lines the total power is the sum of the line powers")
            wavelengths = [line.wavelength_nm for line in self.spectral_lines]
            if len(set(wavelengths)) != len(wavelengths):
                raise ValueError("spectral_lines must have distinct wavelengths")
        elif self.wavelength_nm is None or self.power_w is None:
            raise ValueError("an emitter needs wavelength_nm and power_w (one line) or spectral_lines (several lines)")
        return self

    @property
    def emission_lines(self) -> tuple[SpectralLineV1, ...]:
        """The emitter's spectral lines in the author's order."""

        if self.spectral_lines is not None:
            return self.spectral_lines
        return (SpectralLineV1(wavelength_nm=self.wavelength_nm, power_w=self.power_w),)  # type: ignore[arg-type]

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        """The vacuum wavelength of every spectral line, in the author's order."""

        return tuple(line.wavelength_nm for line in self.emission_lines)

    @property
    def total_power_w(self) -> float:
        """The emitter's total radiant power: ``power_w``, or the correctly rounded sum of the line powers."""

        if self.spectral_lines is None:
            return self.power_w  # type: ignore[return-value]
        return math.fsum(line.power_w for line in self.spectral_lines)

    @property
    def coherence_declared(self) -> bool:
        """An emitter's rays have no phase relation: there is no coherence to declare."""

        return False


#: The most grid points per axis of a map or field written in the Project (an intensity map, a sampled field).
MAX_SOURCE_GRID_SAMPLES = 256
#: The most grid points per axis of a generated speckle field.
MAX_SPECKLE_GRID_SAMPLES = 1024
#: The most independent realizations of a speckle source.
MAX_SPECKLE_REALIZATIONS = 256


class SourceGridV1(StrictContractModel):
    """A regular grid in the source's local XY plane, centred on the local origin.

    ``column_count`` points along local +X at ``pitch_mm[0]`` and ``row_count`` points along local +Y at ``pitch_mm[1]``: the
    point of row ``r`` and column ``c`` lies at ``x = (c - (column_count - 1) / 2) pitch_mm[0]``,
    ``y = (r - (row_count - 1) / 2) pitch_mm[1]``. Arrays on the grid are ``[row][column]``: one row per ``y``, row 0 at the
    most negative ``y``, columns along ``+x``. Each point's cell is the pitch rectangle centred on it; the cells together
    cover ``column_count pitch_mm[0]`` by ``row_count pitch_mm[1]``."""

    column_count: int = Field(strict=True, ge=1, le=MAX_SOURCE_GRID_SAMPLES, description="Grid points along local +X.")
    row_count: int = Field(strict=True, ge=1, le=MAX_SOURCE_GRID_SAMPLES, description="Grid points along local +Y.")
    pitch_mm: tuple[float, float] = Field(description="Grid step [along local X, along local Y] in mm.")

    @model_validator(mode="after")
    def _pitch(self) -> SourceGridV1:
        if not all(math.isfinite(p) and p > 0 for p in self.pitch_mm):
            raise ValueError("pitch_mm must be two positive numbers")
        return self

    @property
    def cell_area_mm2(self) -> float:
        """The area of one grid cell, mm^2."""

        return self.pitch_mm[0] * self.pitch_mm[1]


class SpeckleGridV1(SourceGridV1):
    """The grid a speckle field is generated on (``SourceGridV1``, up to ``MAX_SPECKLE_GRID_SAMPLES`` points per axis: the
    field is generated, not written in the Project)."""

    column_count: int = Field(strict=True, ge=2, le=MAX_SPECKLE_GRID_SAMPLES, description="Grid points along local +X.")
    row_count: int = Field(strict=True, ge=2, le=MAX_SPECKLE_GRID_SAMPLES, description="Grid points along local +Y.")


def _check_rows(rows: tuple[tuple[float, ...], ...], grid: SourceGridV1, name: str, *, nonnegative: bool) -> None:
    """``rows`` holds ``grid.row_count`` rows of ``grid.column_count`` finite values (at least 0 when ``nonnegative``)."""

    if len(rows) != grid.row_count or any(len(row) != grid.column_count for row in rows):
        raise ValueError(f"{name} must hold {grid.row_count} rows of {grid.column_count} values, as the grid says")
    for row in rows:
        for value in row:
            if not math.isfinite(value) or (nonnegative and value < 0):
                raise ValueError(f"{name} must be finite numbers" + (" of at least 0" if nonnegative else ""))
    if nonnegative and not any(value > 0 for row in rows for value in row):
        raise ValueError(f"{name} must be positive somewhere")


def _finite_sum(values, name: str) -> float:  # type: ignore[no-untyped-def]
    """The correctly rounded sum of nonnegative ``values``; a sum beyond the largest finite number is refused."""

    try:
        total = math.fsum(values)
    except OverflowError:
        total = math.inf
    if not math.isfinite(total):
        raise ValueError(f"{name}: the sum over the grid exceeds the largest finite number; state smaller values (a RELATIVE "
                         "map or amplitude keeps its shape at any scale)")
    return total


class IntensityMapV1(StrictContractModel):
    """The emitted power per area (radiant exitance) of a sampled intensity source, on a grid of its emitting plane.

    ``interpolation`` ``PIECEWISE_CONSTANT``: each grid point's value holds over its cell; the emitting area is the grid's
    whole extent. ``BILINEAR``: the values are the exitance at the grid points, bilinear between them; the emitting area is
    the rectangle through the outer grid points (``(column_count - 1) pitch_mm[0]`` by ``(row_count - 1) pitch_mm[1]``, at
    least two points per axis). Outside the emitting area nothing is emitted.

    ``unit`` ``RELATIVE``: the values give the shape only; the source's line powers set the scale. ``W_PER_MM2``: the values
    are the exitance in W/mm^2; the source's power is their integral over the emitting area (``integral``)."""

    grid: SourceGridV1
    values: tuple[tuple[float, ...], ...] = Field(
        description="Exitance at each grid point: one row per y, row 0 at the most negative y; finite, at least 0, positive somewhere.")
    unit: Literal["RELATIVE", "W_PER_MM2"] = Field(
        description="RELATIVE: the shape only, the source's power_w or line powers set the scale. W_PER_MM2: exitance in W/mm^2, "
        "whose integral is the source's power.")
    interpolation: Literal["PIECEWISE_CONSTANT", "BILINEAR"] = Field(
        description="PIECEWISE_CONSTANT: each value holds over its grid cell. BILINEAR: bilinear between the grid points, over the "
        "rectangle through the outer points.")

    @model_validator(mode="after")
    def _values(self) -> IntensityMapV1:
        _check_rows(self.values, self.grid, "values", nonnegative=True)
        if not math.isfinite(_finite_sum((v for row in self.values for v in row), "values") * self.grid.cell_area_mm2):
            raise ValueError("values: the map's integral (the sum times the cell area) exceeds the largest finite number")
        if self.interpolation == "BILINEAR":
            if self.grid.column_count < 2 or self.grid.row_count < 2:
                raise ValueError("a BILINEAR map needs at least two grid points along each axis")
            if self.integral() <= 0:
                raise ValueError("a BILINEAR map must be positive between its grid points somewhere")
        return self

    def integral(self) -> float:
        """The integral of the map over its emitting area, in the values' unit times mm^2 (``W_PER_MM2``: the power in W).
        ``PIECEWISE_CONSTANT``: the sum of the values times the cell area; ``BILINEAR``: the trapezoidal rule, exact for the
        bilinear surface."""

        cell = self.grid.cell_area_mm2
        if self.interpolation == "PIECEWISE_CONSTANT":
            return math.fsum(v for row in self.values for v in row) * cell
        rows, cols = self.grid.row_count, self.grid.column_count
        total = math.fsum(self.values[r][c] * (0.5 if r in (0, rows - 1) else 1.0) * (0.5 if c in (0, cols - 1) else 1.0)
                          for r in range(rows) for c in range(cols))
        return total * cell

    def emitting_size_mm(self) -> tuple[float, float]:
        """Width and height of the emitting area (module docstring)."""

        nx, ny = self.grid.column_count, self.grid.row_count
        if self.interpolation == "BILINEAR":
            nx, ny = nx - 1, ny - 1
        return nx * self.grid.pitch_mm[0], ny * self.grid.pitch_mm[1]


#: The angular laws of a sampled intensity source.
INTENSITY_ANGULAR_LAWS = ("LAMBERTIAN", "ISOTROPIC", "COSINE_POWER", "UNIFORM_CONE", "COLLIMATED")


class SampledIntensitySourceV1(StrictContractModel):
    """An arbitrary intensity pattern: a planar emitter of incoherent, unpolarized light whose exitance follows a sampled map
    (``IntensityMapV1``), in the component's local frame (placed by ``nominal_pose``).

    * Surface: the map's emitting area in the local XY plane (``z = 0``), centred on the local origin. It is an emitter only.
    * Spectrum and power: ``wavelength_nm`` + ``power_w`` or ``spectral_lines`` as on an extended emitter, each line's power
      the radiant power it emits into the angular law's domain, distributed over the area as the map; with a ``W_PER_MM2``
      map the power is the map's integral (one line: ``wavelength_nm`` without ``power_w``).
    * ``angular_law`` (the same at every point): ``LAMBERTIAN`` uniform radiance over the hemisphere around local +Z;
      ``ISOTROPIC`` uniform intensity over the whole sphere; ``COSINE_POWER`` intensity ``cos(theta)^n`` over the hemisphere
      (``n`` = ``cosine_power_exponent``; ``n = 1`` is Lambertian); ``UNIFORM_CONE`` uniform radiance within
      ``cone_half_angle_deg`` of local +Z (uniform in projected solid angle, none beyond it); ``COLLIMATED`` every ray along
      local +Z (a geometric idealization: zero divergence, no diffraction spread).
    * Coherence: none. Grid points and rays are mutually incoherent samples: their powers add on every plane; results of this
      source hold no interference and no Gaussian beam quantity.
    * Sampling: ``sampling.sample_count`` rays of equal power per line, from the extended emitter's low-discrepancy sequence
      shifted by ``sampling.seed``: positions by the inverse of the map's cumulative distribution (rows, then columns in the
      row, then the interpolation within the cell), directions by the angular law. Results report the estimate from the
      first half of the rays beside the full one: their difference shows the sampling error.
    """

    model_kind: Literal["SAMPLED_INTENSITY"]
    wavelength_nm: float | None = Field(
        None, gt=0, description="Vacuum wavelength in nm of a single-line source; absent when spectral_lines is given.")
    power_w: float | None = Field(
        None, gt=0, description="Total radiant power in W of a single-line source with a RELATIVE map, into the angular law's "
        "whole domain; absent with spectral_lines or a W_PER_MM2 map (whose integral is the power).")
    spectral_lines: tuple[SpectralLineV1, ...] | None = Field(
        None, min_length=2,
        description="Emission lines (two or more, distinct wavelengths, author's order) of a RELATIVE map, instead of "
        "wavelength_nm and power_w. Every line has the map's shape.")
    intensity_map: IntensityMapV1 = Field(description="The exitance over the emitting plane: grid, values, unit, interpolation.")
    angular_law: Literal["LAMBERTIAN", "ISOTROPIC", "COSINE_POWER", "UNIFORM_CONE", "COLLIMATED"] = Field(
        description="LAMBERTIAN: uniform radiance over the hemisphere around local +Z. ISOTROPIC: uniform intensity over the "
        "whole sphere. COSINE_POWER: intensity cos(theta)^n over the hemisphere. UNIFORM_CONE: uniform radiance within "
        "cone_half_angle_deg of local +Z. COLLIMATED: every ray along local +Z.")
    cosine_power_exponent: float | None = Field(
        None, gt=0, le=10000, description="The exponent n of a COSINE_POWER law (only with it).")
    cone_half_angle_deg: float | None = Field(
        None, gt=0, le=90, description="Half angle of a UNIFORM_CONE law in degrees (only with it).")
    sampling: EmitterSamplingV1 = Field(description="Rays per spectral line and the shift of their sequence.")

    @model_validator(mode="after")
    def _one_emission_form(self) -> SampledIntensitySourceV1:
        if self.intensity_map.unit == "W_PER_MM2":
            if self.wavelength_nm is None or self.power_w is not None or self.spectral_lines is not None:
                raise ValueError("a W_PER_MM2 map states the power (its integral): give wavelength_nm only, without power_w or "
                                 "spectral_lines")
        elif self.spectral_lines is not None:
            if self.wavelength_nm is not None or self.power_w is not None:
                raise ValueError("declare the emission once: wavelength_nm and power_w (one line) or spectral_lines (several "
                                 "lines), not both")
            wavelengths = [line.wavelength_nm for line in self.spectral_lines]
            if len(set(wavelengths)) != len(wavelengths):
                raise ValueError("spectral_lines must have distinct wavelengths")
        elif self.wavelength_nm is None or self.power_w is None:
            raise ValueError("a RELATIVE map needs wavelength_nm and power_w (one line) or spectral_lines (several lines)")
        if (self.cosine_power_exponent is not None) != (self.angular_law == "COSINE_POWER"):
            raise ValueError("cosine_power_exponent is given with a COSINE_POWER law, and only with it")
        if (self.cone_half_angle_deg is not None) != (self.angular_law == "UNIFORM_CONE"):
            raise ValueError("cone_half_angle_deg is given with a UNIFORM_CONE law, and only with it")
        return self

    @property
    def emission_lines(self) -> tuple[SpectralLineV1, ...]:
        """The source's spectral lines in the author's order (a ``W_PER_MM2`` map: its one line with the map's integral)."""

        if self.spectral_lines is not None:
            return self.spectral_lines
        power = self.intensity_map.integral() if self.intensity_map.unit == "W_PER_MM2" else self.power_w
        return (SpectralLineV1(wavelength_nm=self.wavelength_nm, power_w=power),)  # type: ignore[arg-type]

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        """The vacuum wavelength of every spectral line, in the author's order."""

        return tuple(line.wavelength_nm for line in self.emission_lines)

    @property
    def total_power_w(self) -> float:
        """The source's total radiant power."""

        return math.fsum(line.power_w for line in self.emission_lines)

    @property
    def coherence_declared(self) -> bool:
        """The rays of a sampled intensity source have no phase relation: there is no coherence to declare."""

        return False


class SampledFieldSourceV1(StrictContractModel):
    """A sampled complex field: the coherent, monochromatic scalar field ``U = amplitude exp(i phase_rad)`` on a grid of the
    source's local XY plane (``z = 0``), propagating toward local +Z, with one polarization over the whole field.

    Conventions: time dependence ``exp(-i w t)``, propagation ``exp(+i k z)``; a phase ``2 pi x sin(theta) / lambda`` (in
    the source's medium, ``lambda`` = ``wavelength_nm`` / n) tilts the light toward local +X by ``theta``. ``|U|^2`` is the
    irradiance in W/mm^2 (``amplitude_unit`` ``SQRT_W_PER_MM2``) or in proportion to it (``RELATIVE``, scaled so that the
    field's power over the grid is ``power_w``); the power is the sum of ``|U|^2`` times the cell area.

    Interpolation and validity: the grid points sample a band-limited field (the field's own diffraction analyses take
    them as such, ``os-optics results field``). Between neighbouring lit grid points (``|U|^2`` at least ``1e-6`` of its
    peak, and the brightest points that hold all but ``1e-3`` of the power) the phase, read as the difference wrapped into
    ``(-pi, pi]``, must change by at most ``pi / 2`` (four or more grid points per ``2 pi`` of phase: the grid resolves the
    field's local wavevector); with a coarser phase (``SOURCE_FIELD_UNDERSAMPLED``) a run does not trace the field: an
    ``OUT_OF_MODEL_SCOPE`` issue at the source, its line booked as not traced, the rest of the scene evaluated and the run
    completed; ``os-optics results field`` refuses such a field. The peak of ``|U|^2`` must be a normal float64 (at least
    about ``2.2e-308``) and the field's power over the grid finite.

    How a run traces it (geometric optics): ``ray_sampling.sample_count`` rays, positions drawn in proportion to ``|U|^2``
    (constant over each cell), each leaving along the field's local wavevector: its transverse direction in the medium is
    ``grad(phase) / (n k0)`` (the eikonal; ``k0 = 2 pi / wavelength_nm``), its optical path starting at ``phase / k0``, its
    polarization ``polarization``. The rays add their powers on every plane: the run gives the field's geometric
    transport (power, collection, sampled irradiance), not its interference or diffraction, which ``os-optics results
    field`` computes from the same field. A phase gradient at or beyond the medium's wavenumber at any grid point a ray takes
    its direction from (the points of every cell with a nonzero amplitude and their neighbours) is evanescent: the field is
    not traced, as with a coarser phase (``SOURCE_FIELD_EVANESCENT``)."""

    model_kind: Literal["SAMPLED_FIELD"]
    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength in nm (one line: the field is monochromatic).")
    power_w: float | None = Field(
        None, gt=0, description="Power of the field over the grid in W, with a RELATIVE amplitude; absent with SQRT_W_PER_MM2 "
        "(whose squares give it).")
    grid: SourceGridV1 = Field(description="The grid in the local XY plane, centred on the local origin.")
    amplitude: tuple[tuple[float, ...], ...] = Field(
        description="|U| at each grid point: one row per y, row 0 at the most negative y; finite, at least 0, positive somewhere.")
    phase_rad: tuple[tuple[float, ...], ...] = Field(description="Phase of U at each grid point in rad: one row per y.")
    amplitude_unit: Literal["RELATIVE", "SQRT_W_PER_MM2"] = Field(
        description="RELATIVE: the shape only, power_w sets the scale. SQRT_W_PER_MM2: |U| in sqrt(W/mm^2).")
    polarization: FullPolarizationV1 = Field(
        description="The field's polarization in the source's local frame: linear (kind LINEAR, the meaning when kind is "
        "omitted) or elliptical (kind ELLIPTICAL). A coherent field is fully polarized.")
    ray_sampling: EmitterSamplingV1 = Field(description="Rays of the run's geometric trace of the field and the shift of their sequence.")

    @model_validator(mode="after")
    def _field(self) -> SampledFieldSourceV1:
        _check_rows(self.amplitude, self.grid, "amplitude", nonnegative=True)
        _check_rows(self.phase_rad, self.grid, "phase_rad", nonnegative=False)
        peak = max(a for row in self.amplitude for a in row)
        if not MIN_NORMAL_FLOAT <= peak * peak < math.inf:
            raise ValueError(f"amplitude: the square of its peak ({peak:.6g}) must be a finite number of at least "
                             f"{MIN_NORMAL_FLOAT:.6g} (|U|^2 is the irradiance); state the amplitude in that range (a RELATIVE "
                             "amplitude keeps its shape at any scale)")
        if not math.isfinite(_finite_sum((a * a for row in self.amplitude for a in row), "amplitude^2") * self.grid.cell_area_mm2):
            raise ValueError("amplitude: the field's power (the sum of |U|^2 times the cell area) exceeds the largest finite number")
        if (self.power_w is None) != (self.amplitude_unit == "SQRT_W_PER_MM2"):
            raise ValueError("a RELATIVE amplitude needs power_w; an SQRT_W_PER_MM2 amplitude states the power itself (omit power_w)")
        return self

    def grid_power_w(self) -> float:
        """The sum of the stated ``|U|^2`` times the cell area (in W for ``SQRT_W_PER_MM2``)."""

        return math.fsum(a * a for row in self.amplitude for a in row) * self.grid.cell_area_mm2

    @property
    def emission_lines(self) -> tuple[SpectralLineV1, ...]:
        """The field's one line with its power."""

        power = self.power_w if self.power_w is not None else self.grid_power_w()
        return (SpectralLineV1(wavelength_nm=self.wavelength_nm, power_w=power),)

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        return (self.wavelength_nm,)

    @property
    def total_power_w(self) -> float:
        return self.emission_lines[0].power_w

    @property
    def coherence_declared(self) -> bool:
        """The run's rays of the field are independent samples with no phase relation for the run's two-beam metrics; the
        field's own coherence is used by ``os-optics results field``."""

        return False


class SpeckleUniformDiscV1(StrictContractModel):
    """Uniform irradiance over a disc centred on the local origin."""

    shape: Literal["UNIFORM_DISC"]
    diameter_mm: float = Field(gt=0, description="Diameter of the illuminated disc in mm.")


class SpeckleUniformRectangleV1(StrictContractModel):
    """Uniform irradiance over a rectangle centred on the local origin."""

    shape: Literal["UNIFORM_RECTANGLE"]
    width_mm: float = Field(gt=0, description="Full width along local +X in mm.")
    height_mm: float = Field(gt=0, description="Full height along local +Y in mm.")


class SpeckleGaussianV1(StrictContractModel):
    """A round Gaussian irradiance ``exp(-2 r^2 / w^2)`` centred on the local origin."""

    shape: Literal["GAUSSIAN"]
    radius_mm: float = Field(gt=0, description="1/e^2 intensity radius w in mm.")


SpeckleIlluminationV1 = Annotated[SpeckleUniformDiscV1 | SpeckleUniformRectangleV1 | SpeckleGaussianV1, Field(discriminator="shape")]


class PhaseScreenV1(StrictContractModel):
    """A random phase screen (a diffuser) of Gaussian statistics and Gaussian correlation.

    ``GAUSSIAN_CORRELATED``: a zero-mean Gaussian random phase whose ensemble autocorrelation is
    ``phase_rms_rad^2 exp(-r^2 / correlation_length_mm^2)``. Generation (integer and ``+ - x / sqrt`` arithmetic: the same
    screen on every platform): on the grid extended by ``ceil(3 correlation_length_mm / pitch)`` points on each side, one
    deviate of unit variance per point (the centred sum of four uniform deviates from consecutive SplitMix64 words of the
    realization's seed, scaled), convolved along each axis with ``exp(-2 t^2 / correlation_length_mm^2)`` truncated at
    ``|t| <= 3 correlation_length_mm`` and scaled by ``phase_rms_rad`` over the kernels' root sum of squares. Each screen
    value is a weighted sum of dozens of independent deviates (at least two grid pitches per correlation length), so it is
    Gaussian to the central-limit approximation. Realization ``m`` (from 0) uses the seed ``seed + m``."""

    statistics: Literal["GAUSSIAN_CORRELATED"]
    phase_rms_rad: float = Field(gt=0, le=1000, description="Standard deviation of the phase in rad.")
    correlation_length_mm: float = Field(gt=0, description="Distance at which the phase autocorrelation falls to 1/e, in mm.")
    seed: int = Field(strict=True, ge=0, le=MAX_EXACT_JSON_INTEGER - MAX_SPECKLE_REALIZATIONS,
                      description="Seed of realization 0; realization m uses seed + m.")


class SpeckleSourceV1(StrictContractModel):
    """A speckle source: coherent light of one wavelength through a random phase screen, as a reproducible complex field on a
    grid of the source's local XY plane, propagating toward local +Z.

    Field: ``U = sqrt(I(x, y)) exp(i phi_m(x, y))`` on ``grid``, with ``I`` the illumination's irradiance (scaled so that the
    power over the grid is ``power_w``) and ``phi_m`` realization ``m`` of ``phase_screen``. Each realization is one
    coherent field (one speckle pattern); different realizations are mutually incoherent (a moving diffuser over an exposure,
    for example): ``os-optics results field`` gives each pattern's irradiance, their mean, the speckle contrast of one
    pattern (fully developed speckle: 1) and of the average of the ``realizations`` patterns, and its convergence.

    Validity: the screen must be sampled by the grid: ``correlation_length_mm`` at least two grid pitches, and the phase
    step between neighbouring lit grid points at most ``pi / 2`` (as for a sampled field; otherwise the field is not traced,
    ``SOURCE_FIELD_UNDERSAMPLED``). The rms phase step is about ``sqrt(2) phase_rms_rad pitch / correlation_length_mm``, so
    a strong diffuser needs a correlation length of many pitches. The illumination beyond the grid is not part of the field.

    How a run traces it (geometric optics): ``ray_sampling.sample_count`` rays drawn in proportion to the illumination, ray
    ``k`` through realization ``k mod realizations``, each leaving along its realization's local wavevector (its transverse
    direction ``grad(phi_m) / (n k0)``): the run's sampled irradiance is the geometric transport of the diffused light,
    without the speckle (as a sampled field's rays, ``SampledFieldSourceV1``)."""

    model_kind: Literal["SPECKLE"]
    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength in nm.")
    power_w: float = Field(gt=0, description="Power of the illumination over the grid in W (the same for every realization).")
    polarization: FullPolarizationV1 = Field(
        description="The polarization of the light in the source's local frame (linear or elliptical), kept by the screen.")
    illumination: SpeckleIlluminationV1 = Field(
        description="The irradiance on the screen, centred on the local origin, with a flat phase: UNIFORM_DISC (diameter_mm), "
        "UNIFORM_RECTANGLE (width_mm, height_mm) or GAUSSIAN (radius_mm, the 1/e^2 radius).")
    phase_screen: PhaseScreenV1 = Field(description="The random phase screen: statistics, rms phase, correlation length and seed.")
    grid: SpeckleGridV1 = Field(description="The grid the field is generated on, centred on the local origin.")
    realizations: int = Field(1, strict=True, ge=1, le=MAX_SPECKLE_REALIZATIONS,
                              description="Independent realizations of the screen (seed, seed + 1, ...): mutually incoherent patterns.")
    ray_sampling: EmitterSamplingV1 = Field(description="Rays of the run's geometric trace and the shift of their sequence.")

    @model_validator(mode="after")
    def _screen_sampled(self) -> SpeckleSourceV1:
        if self.phase_screen.correlation_length_mm < 2.0 * max(self.grid.pitch_mm):
            raise ValueError("the phase screen's correlation_length_mm must be at least two grid pitches (the grid samples the screen)")
        return self

    @property
    def emission_lines(self) -> tuple[SpectralLineV1, ...]:
        return (SpectralLineV1(wavelength_nm=self.wavelength_nm, power_w=self.power_w),)

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        return (self.wavelength_nm,)

    @property
    def total_power_w(self) -> float:
        return self.power_w

    @property
    def coherence_declared(self) -> bool:
        """As a sampled field's (``SampledFieldSourceV1.coherence_declared``)."""

        return False


#: Planck's constant (J s) and the speed of light (m/s), exact SI values: the photon energy ``h c / lambda`` of a line.
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0
#: Photon fractions of a fluorescence spectrum within this of 1 in sum are used as written; others are read as relative shares.
PHOTON_FRACTION_SUM_TOLERANCE = 1e-9
#: The validity limit of the Raman model: the single-scattering, undepleted-excitation model (``RamanSourceV1``) holds for a
#: thin scatterer, ``N sigma L`` summed over the lines small against 1. Above it the linear form overstates the scattered
#: power by about half the scattered fraction; the run records the exceeded limit with the observed fraction.
MAX_RAMAN_SCATTERED_FRACTION = 0.01
#: The validity limit of the fluorescence model in decadic absorbance at the excitation wavelength: the thin, optically
#: dilute sample of ``FluorescenceSourceV1`` (the usual limit of quantitative fluorescence, absorbance below 0.1, IUPAC
#: Technical Report, Brouwer, Pure Appl. Chem. 83 (2011) 2213). Across such a sample the excitation falls by at most 21 %,
#: so the whole sample emits; a denser sample emits from a thin layer at its entrance and reabsorbs its own emission
#: (inner-filter effects), which the model does not follow: the run records the exceeded limit with the observed value.
MAX_FLUORESCENCE_ABSORBANCE = 0.1
#: The same limit as an absorbed share, ``1 - 10^-0.1``.
MAX_FLUORESCENCE_ABSORBED_FRACTION = 1.0 - 10.0 ** -MAX_FLUORESCENCE_ABSORBANCE
#: Raman lines one sample may declare, and fluorescence emission lines.
MAX_SAMPLE_LINES = 64
#: nm per cm: a wavenumber in cm^-1 is ``1e7 / wavelength_nm``.
NM_PER_CM = 1e7


def photon_energy_j(wavelength_nm: float) -> float:
    """The energy of one photon of vacuum wavelength ``wavelength_nm``: ``h c / lambda``, in J."""

    return PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S / (wavelength_nm * 1e-9)


class SampleExcitationV1(StrictContractModel):
    """The light that excites a sample: the power of one vacuum wavelength arriving at an observation plane of the Project
    (place the plane at the sample, across the excitation beam). It is read from the trace of every other source."""

    observation_id: OpenSciId = Field(description="The observation plane at the sample whose arriving power excites it.")
    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength of the excitation light, in nm: the power of this "
                                 "wavelength arriving at the plane (from every source that emits it) excites the sample.")


class FluorescenceLineV1(StrictContractModel):
    """One line of a fluorescence emission spectrum: its vacuum wavelength and the share of the emitted photons in it."""

    wavelength_nm: float = Field(gt=0, description="Vacuum wavelength of the emission line, in nm (longer than the excitation's).")
    photon_fraction: float = Field(gt=0, le=1, description="Share of the emitted photons in this line; the shares add to 1.")


class FluorescenceSourceV1(StrictContractModel):
    """Fluorescence of a thin sample excited by the Project's own light, emitted as an incoherent, unpolarized emitter.

    * Excitation: ``excitation`` names the observation plane at the sample and the excitation wavelength. The power ``P_ex``
      of that wavelength arriving at the plane is read from a trace of the Project's other sources (a sample's emission does
      not excite another sample).
    * Absorption: ``absorbance`` ``A`` (decadic, the sample's optical density at the excitation wavelength) gives the
      absorbed share ``a = 1 - 10^-A``; or ``absorbed_fraction`` states ``a`` itself. ``P_abs = a P_ex``. The excitation beam
      is not attenuated by it: past a transparent plane it goes on with its full power (make the plane absorbing, or model
      the sample's body with an absorption coefficient, where the absorbed share matters downstream). The model holds for a
      thin, optically dilute sample: ``A`` at most ``MAX_FLUORESCENCE_ABSORBANCE`` (0.1), ``a`` at most
      ``MAX_FLUORESCENCE_ABSORBED_FRACTION`` (``1 - 10^-0.1``, 0.2057). A denser sample is computed the same way and its
      emission record states the exceeded limit (a validity check with the observed value).
    * Photon numbers: the excitation photon rate is ``P_ex lambda_ex / (h c)``; the sample emits ``quantum_yield`` photons
      per absorbed photon, line ``i`` the share ``photon_fraction_i`` of them (fractions that do not add to 1 are read as
      relative shares: each divided by their sum, ``photon_shares``). Energy: line ``i`` carries
      ``P_i = P_abs quantum_yield photon_fraction_i (lambda_ex / lambda_i)``: each emitted photon has the energy of its own
      wavelength (the Stokes shift's energy stays in the sample). Every emission line is longer than the excitation's.
    * Emission: an incoherent, unpolarized planar emitter on ``emitting_surface`` (centred on the local origin, in the local
      XY plane), uniform over that area, with ``angular_law`` ``ISOTROPIC`` (randomly oriented emitters, the whole sphere) or
      ``LAMBERTIAN`` (a scattering surface, the hemisphere around local +Z), sampled by ``sampling`` as an extended emitter.
    * Conditions: linear excitation (no saturation, no photobleaching), no reabsorption of the emission (the sample's
      absorbance at the emission lines is taken as negligible), emission following the excitation instantly (a lifetime
      short against any modulation period), no polarization memory. ``quantum_yield`` is above 0: a sample that does not
      fluoresce emits nothing and is left out of the Project (or modelled by its body's absorption), as an absorbance of 0
      absorbs nothing.
    """

    model_kind: Literal["FLUORESCENCE"]
    excitation: SampleExcitationV1 = Field(description="The observation plane at the sample and the excitation wavelength.")
    absorbance: float | None = Field(
        None, ge=0, description="Decadic absorbance (optical density) of the sample at the excitation wavelength: the absorbed "
        "share is 1 - 10^-absorbance. The model holds up to 0.1 (a thin, optically dilute sample); above it the run states the "
        "exceeded limit. Give this or absorbed_fraction.")
    absorbed_fraction: float | None = Field(
        None, gt=0, le=1, description="The share of the arriving excitation power the sample absorbs. The model holds up to "
        "1 - 10^-0.1 (0.2057: a thin, optically dilute sample); above it the run states the exceeded limit. Give this or "
        "absorbance.")
    quantum_yield: float = Field(gt=0, le=1, description="Emitted photons per absorbed photon, above 0 and at most 1 (a sample that "
                                 "does not fluoresce is left out of the Project).")
    emission_lines: tuple[FluorescenceLineV1, ...] = Field(
        min_length=1, max_length=MAX_SAMPLE_LINES,
        description="The emission spectrum as lines of distinct wavelengths, each with its share of the emitted photons (shares "
        "that do not add to 1 are read as relative shares); every line is longer than the excitation wavelength.")
    emitting_surface: EmittingSurfaceV1 = Field(
        description="Emitting area in the local XY plane, centred on the local origin: shape RECTANGLE or DISC.")
    angular_law: Literal["ISOTROPIC", "LAMBERTIAN"] = Field(
        description="ISOTROPIC: uniform intensity over the whole sphere. LAMBERTIAN: uniform radiance over the hemisphere "
        "around local +Z.")
    sampling: EmitterSamplingV1 = Field(description="Rays per emission line and the shift of their sequence.")

    @model_validator(mode="after")
    def _consistent(self) -> FluorescenceSourceV1:
        if (self.absorbance is None) == (self.absorbed_fraction is None):
            raise ValueError("state the sample's absorption once: absorbance (decadic) or absorbed_fraction")
        if self.absorbance is not None and self.absorbance == 0.0:
            raise ValueError("absorbance 0 absorbs nothing: a sample that emits absorbs some of the excitation")
        wavelengths = [line.wavelength_nm for line in self.emission_lines]
        if len(set(wavelengths)) != len(wavelengths):
            raise ValueError("emission_lines must have distinct wavelengths")
        if any(w <= self.excitation.wavelength_nm for w in wavelengths):
            raise ValueError(f"every emission line must be longer than the excitation wavelength {self.excitation.wavelength_nm} nm "
                             "(Stokes-shifted fluorescence)")
        return self

    @property
    def photon_shares(self) -> tuple[float, ...]:
        """Per emission line, its share of the emitted photons: ``photon_fraction`` as written when the fractions add to 1
        (within ``PHOTON_FRACTION_SUM_TOLERANCE``), else each divided by their sum."""

        fractions = tuple(line.photon_fraction for line in self.emission_lines)
        total = math.fsum(fractions)
        if abs(total - 1.0) <= PHOTON_FRACTION_SUM_TOLERANCE:
            return fractions
        return tuple(f / total for f in fractions)

    @property
    def thin_sample_check(self) -> tuple[str, float, float]:
        """The thin-sample condition as ``(parameter, limit, observed)``: the stated ``absorbance`` against
        ``MAX_FLUORESCENCE_ABSORBANCE``, or the stated ``absorbed_fraction`` against ``MAX_FLUORESCENCE_ABSORBED_FRACTION``."""

        if self.absorbance is not None:
            return "absorbance", MAX_FLUORESCENCE_ABSORBANCE, self.absorbance
        return "absorbed_fraction", MAX_FLUORESCENCE_ABSORBED_FRACTION, self.absorbed_fraction  # type: ignore[return-value]

    @property
    def absorbed_share(self) -> float:
        """The share ``a`` of the arriving excitation power the sample absorbs."""

        if self.absorbed_fraction is not None:
            return self.absorbed_fraction
        return -math.expm1(-self.absorbance * math.log(10.0))  # type: ignore[operator]

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        """The vacuum wavelength of every emission line, in the author's order."""

        return tuple(line.wavelength_nm for line in self.emission_lines)

    @property
    def coherence_declared(self) -> bool:
        """An emitter's rays have no phase relation: there is no coherence to declare."""

        return False


class RamanLineV1(StrictContractModel):
    """One Raman band: its Stokes shift and the scatterers' differential cross-section for it."""

    shift_cm1: float = Field(gt=0, description="Stokes Raman shift in cm^-1: the line's wavenumber is the excitation's less this.")
    differential_cross_section_cm2_per_sr: float = Field(
        gt=0, description="Differential Raman cross-section d sigma / d Omega of the band per scatterer at the excitation "
        "wavelength, in cm^2/sr, photon-counting: scattered photons per second into a unit solid angle over incident photons "
        "per second per cm^2 (not an energy-weighted cross-section), taken as the same in every direction (isotropic): "
        "the band scatters 4 pi d sigma / d Omega per scatterer in all.")


class RamanSourceV1(StrictContractModel):
    """Spontaneous Stokes Raman scattering of a thin sample excited by the Project's own light, emitted as an incoherent,
    unpolarized emitter.

    * Excitation: as for fluorescence (``SampleExcitationV1``): the power ``P_ex`` of the excitation wavelength arriving at the
      named plane, from a trace of the Project's other sources.
    * Lines: band ``j`` of shift ``s_j`` (cm^-1) scatters at the wavenumber ``1e7 / lambda_ex - s_j``, so
      ``lambda_j = 1e7 / (1e7 / lambda_ex - s_j)`` nm.
    * Photon numbers: with ``N`` scatterers per cm^3 along ``path_length_mm`` (``L``, in cm) and the total cross-section
      ``sigma_j = 4 pi (d sigma / d Omega)_j`` (the stated differential cross-section, the same in every direction), band ``j``
      scatters ``N sigma_j L`` of the excitation photons. Energy: ``P_j = P_ex N sigma_j L (lambda_ex / lambda_j)``.
    * Domain: single scattering of an undepleted excitation, ``sum_j N sigma_j L`` at most
      ``MAX_RAMAN_SCATTERED_FRACTION`` (above it the same linear form is computed and the emission record states the exceeded
      limit; a sum of 1 or more, more photons scattered than arrive, is refused); Stokes lines only; isotropic, unpolarized scattering (no depolarization ratio, no
      polarization dependence of the cross-section); the excitation beam is not attenuated by the scattering.
    * Emission: an incoherent, unpolarized ``ISOTROPIC`` planar emitter on ``emitting_surface``, sampled by ``sampling``.
    """

    model_kind: Literal["RAMAN"]
    excitation: SampleExcitationV1 = Field(description="The observation plane at the sample and the excitation wavelength.")
    raman_lines: tuple[RamanLineV1, ...] = Field(
        min_length=1, max_length=MAX_SAMPLE_LINES, description="The Raman bands, each a Stokes shift with its cross-section; distinct shifts.")
    number_density_per_cm3: float = Field(gt=0, description="Scatterers per cm^3 in the excited volume.")
    path_length_mm: float = Field(gt=0, description="Length of the excitation path through the sample, in mm.")
    emitting_surface: EmittingSurfaceV1 = Field(
        description="Emitting area in the local XY plane, centred on the local origin: shape RECTANGLE or DISC.")
    angular_law: Literal["ISOTROPIC"] = Field(description="ISOTROPIC: uniform intensity over the whole sphere.")
    sampling: EmitterSamplingV1 = Field(description="Rays per Raman line and the shift of their sequence.")

    @model_validator(mode="after")
    def _consistent(self) -> RamanSourceV1:
        shifts = [line.shift_cm1 for line in self.raman_lines]
        if len(set(shifts)) != len(shifts):
            raise ValueError("raman_lines must have distinct shifts")
        excitation_cm1 = NM_PER_CM / self.excitation.wavelength_nm
        if any(shift >= excitation_cm1 for shift in shifts):
            raise ValueError(f"a Stokes shift must be smaller than the excitation's wavenumber {excitation_cm1:.6g} cm^-1")
        total = math.fsum(self.scattered_fractions)
        if total >= 1.0:
            raise ValueError(f"the sample scatters {total:.4g} of the excitation photons (N sigma L summed over the lines): a share "
                             "of 1 or more scatters more photons than arrive; state the number density, path length or cross-sections "
                             "of a sample that scatters less")
        return self

    @property
    def scattered_fractions(self) -> tuple[float, ...]:
        """Per line, the share of the excitation photons it scatters: ``N 4 pi (d sigma / d Omega) L``."""

        length_cm = self.path_length_mm / 10.0
        return tuple(self.number_density_per_cm3 * 4.0 * math.pi * line.differential_cross_section_cm2_per_sr * length_cm
                     for line in self.raman_lines)

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        """The vacuum wavelength of every Raman line, in the author's order: ``1e7 / (1e7 / lambda_ex - shift)``."""

        excitation_cm1 = NM_PER_CM / self.excitation.wavelength_nm
        return tuple(NM_PER_CM / (excitation_cm1 - line.shift_cm1) for line in self.raman_lines)

    @property
    def coherence_declared(self) -> bool:
        """An emitter's rays have no phase relation: there is no coherence to declare."""

        return False


#: The source models whose emission is derived from the Project's own light at a sample.
SampleEmissionSourceV1 = FluorescenceSourceV1 | RamanSourceV1
SAMPLE_EMISSION_KINDS = ("FLUORESCENCE", "RAMAN")

#: The kinds of a source model (``SourceModelV1``).
SOURCE_MODEL_KINDS = ("GAUSSIAN_BEAM", "EXTENDED_EMITTER", "SAMPLED_INTENSITY", "SAMPLED_FIELD", "SPECKLE", *SAMPLE_EMISSION_KINDS)

#: The emission model a source component carries.
SourceModelV1 = Annotated[GaussianBeamSourceV1 | ExtendedEmitterSourceV1 | SampledIntensitySourceV1 | SampledFieldSourceV1
                          | SpeckleSourceV1 | FluorescenceSourceV1 | RamanSourceV1, Field(discriminator="model_kind")]
