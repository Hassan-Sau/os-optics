"""Engine input snapshots of sources, components, envelopes and actuators.

Component data reaches the Engine only as an ``EngineeringComponentSnapshot``: each field is something the Engine
computes with.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal

from pydantic import Field, model_validator

from .base import Id, OmitAbsentFields, StrictModel, Vector2, Vector3
from .detection import SampleEmission
from .geometry import Pose3D
from .optics import OpticalPort, OpticalSurface
from .polarization import JonesVector


class SpectralLine(StrictModel):
    wavelength_nm: float = Field(gt=0)
    weight: float = Field(gt=0, description="Relative power weight; normalized over the spectrum.")


class PulseTrain(OmitAbsentFields, StrictModel):
    """A pulsed source: pulses of ``pulse_duration_s`` (full width at half maximum of the intensity) at
    ``repetition_rate_hz``. The source's ``power_W`` is the average power; the pulse energy of a line is its average power
    divided by the repetition rate. The per-surface exposure (peak fluence) and the power-rating comparison use it, and a
    crystal's second-harmonic generation reads it (the conversion of a pulse train is out of model scope): the trace itself
    is the same as for a continuous source. ``total_pulses``: how many pulses the source sends into the
    optical system while the optics stay in place (``UNLIMITED``: it keeps pulsing for as long as they are in use;
    ``None``: not stated), which only the pulse-count condition of a rating comparison reads."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("total_pulses",)

    repetition_rate_hz: float = Field(gt=0)
    pulse_duration_s: float = Field(gt=0)
    duration_definition: Literal["FWHM"] = "FWHM"
    total_pulses: Annotated[int, Field(strict=True, ge=1, le=2**53 - 1)] | Literal["UNLIMITED"] | None = None

    @model_validator(mode="after")
    def _shorter_than_period(self) -> PulseTrain:
        if self.pulse_duration_s * self.repetition_rate_hz >= 1.0:
            raise ValueError("pulse_duration_s must be shorter than the pulse period 1 / repetition_rate_hz")
        return self


class GaussianSpectralShape(StrictModel):
    """``exp(-4 ln 2 (lambda - center)^2 / fwhm^2)`` over the vacuum wavelength."""

    kind: Literal["gaussian"] = "gaussian"
    center_nm: float = Field(gt=0)
    fwhm_nm: float = Field(gt=0)


class LorentzianSpectralShape(StrictModel):
    """``1 / (1 + (2 (lambda - center) / fwhm)^2)`` over the vacuum wavelength."""

    kind: Literal["lorentzian"] = "lorentzian"
    center_nm: float = Field(gt=0)
    fwhm_nm: float = Field(gt=0)


class TopHatSpectralShape(StrictModel):
    """Flat from ``center - fwhm / 2`` to ``center + fwhm / 2``."""

    kind: Literal["top_hat"] = "top_hat"
    center_nm: float = Field(gt=0)
    fwhm_nm: float = Field(gt=0)


class TabulatedSpectralShape(StrictModel):
    """A relative density at rows of the vacuum wavelength, linear between them."""

    kind: Literal["tabulated"] = "tabulated"
    wavelength_nm: tuple[float, ...] = Field(min_length=2)
    density: tuple[float, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _rows(self) -> TabulatedSpectralShape:
        if len(self.wavelength_nm) != len(self.density) or any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
            raise ValueError("wavelength_nm (strictly increasing) and density need one value per row")
        if any(d < 0 for d in self.density):
            raise ValueError("density must not be negative")
        return self


SpectralShape = Annotated[GaussianSpectralShape | LorentzianSpectralShape | TopHatSpectralShape | TabulatedSpectralShape,
                          Field(discriminator="kind")]
#: The most quadrature lines of one continuous spectrum.
MAX_DENSITY_LINES = 256


class SpectralDensity(StrictModel):
    """A continuous spectrum traced as quadrature lines (``optics/interference/spectrum.py``): the density ``shape`` over the
    vacuum wavelength within ``band_nm``, ``line_count`` lines at the centres of equal bins, each with its bin's integral of the
    density as weight and the bin's width as its Gaussian linewidth (``quadrature`` ``midpoint_bins``)."""

    shape: SpectralShape
    band_nm: tuple[float, float] = Field(description="[min, max] vacuum wavelengths the spectrum is traced over (nm).")
    line_count: int = Field(ge=2, le=MAX_DENSITY_LINES)
    quadrature: Literal["midpoint_bins"] = "midpoint_bins"

    @model_validator(mode="after")
    def _band(self) -> SpectralDensity:
        if not 0 < self.band_nm[0] < self.band_nm[1]:
            raise ValueError("band_nm must satisfy 0 < min < max")
        return self


class SourceSpec(OmitAbsentFields, StrictModel):
    """Coherent Gaussian source. Emits along the source frame ``+z``; polarization Jones in the frame ``(x, y)``.

    ``degree_of_polarization`` (optional): partially polarized light of degree ``P`` whose polarized part is
    ``polarization``. Each line is then traced as two mutually incoherent fully polarized components, ``polarization``
    with ``(1 + P) / 2`` of the line's power and the orthogonal state ``(-conj(E_y), conj(E_x))`` with ``(1 - P) / 2``
    (``optics/polarization/partial.py``). Omitted (or 1): fully polarized."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("pulse", "degree_of_polarization", "spectral_density")

    source_id: Id
    local_pose: Pose3D = Pose3D()
    wavelength_nm: float | None = Field(None, gt=0, description="Single wavelength (mutually exclusive with spectrum).")
    spectrum: tuple[SpectralLine, ...] = ()
    power_W: float = Field(gt=0)
    waist_radius_x_mm: float = Field(gt=0, description="1/e^2 intensity radius, not diameter.")
    waist_radius_y_mm: float = Field(gt=0)
    waist_position_z_mm: float = Field(0.0, description="Waist position along source +z relative to the source origin.")
    m2_x: float = Field(1.0, ge=1.0)
    m2_y: float = Field(1.0, ge=1.0)
    polarization: JonesVector = JonesVector.linear(0.0)
    medium: Id = "air"
    coherence_length_mm: float | None = Field(
        None, gt=0, description="OPD at which |gamma| = 1/e (Gaussian-decay model), of every spectral line."
    )
    linewidth_nm: float | None = Field(
        None, gt=0, description="Gaussian spectral FWHM of every spectral line; converted to a coherence length per line (from "
        "the line's own wavelength) if no length is given.",
    )
    coherence_group: str | None = Field(
        None, description="Sources sharing a group id are mutually coherent; default: each source is its own group."
    )
    pupil_ring_count: int | None = Field(None, ge=1)
    pupil_points_per_ring: int | None = Field(None, ge=1)
    pulse: PulseTrain | None = Field(None, description="Pulse train of a pulsed source (power_W is then the average power).")
    degree_of_polarization: float | None = Field(
        None, ge=0.0, le=1.0, description="Degree of polarization of partially polarized light whose polarized part is "
        "polarization; omitted: fully polarized. Below 1 every line is traced as two mutually incoherent orthogonal "
        "polarization components.")
    spectral_density: SpectralDensity | None = Field(
        None, description="A continuous spectrum the lines sample: spectrum is then exactly its quadrature lines and "
        "linewidth_nm its bin width (both checked), power_W the power within its band, and the run reports the spectrum's "
        "coherence (ValidationResult.source_spectra). Omitted: the lines are the source's discrete spectrum.")

    @model_validator(mode="after")
    def _spectrum(self) -> SourceSpec:
        if (self.wavelength_nm is None) == (len(self.spectrum) == 0):
            raise ValueError("exactly one of wavelength_nm or spectrum must be given")
        if self.spectral_density is not None:
            from ..optics.interference.spectrum import bin_width_nm, quadrature_lines

            lines = tuple((line.wavelength_nm, line.weight) for line in self.spectrum)
            if lines != quadrature_lines(self.spectral_density):
                raise ValueError("spectrum must be the quadrature lines of spectral_density (their wavelengths and bin weights)")
            if self.linewidth_nm != bin_width_nm(self.spectral_density) or self.coherence_length_mm is not None:
                raise ValueError("a continuous spectrum's lines have the bin width as linewidth_nm and no coherence_length_mm")
        return self


class EmitterGrid(StrictModel):
    """A grid in the emitter's local XY plane, centred on its origin: ``columns`` points along x at ``pitch_x_mm``, ``rows``
    along y at ``pitch_y_mm`` (point ``(r, c)`` at ``x = (c - (columns - 1) / 2) pitch_x_mm``, ``y = (r - (rows - 1) / 2)
    pitch_y_mm``). Values on it are flattened row-major (row 0 at the most negative y)."""

    columns: int = Field(ge=1)
    rows: int = Field(ge=1)
    pitch_x_mm: float = Field(gt=0)
    pitch_y_mm: float = Field(gt=0)

    @property
    def size(self) -> int:
        return self.columns * self.rows


def _flat(values: tuple[float, ...], grid: EmitterGrid, name: str, nonnegative: bool) -> None:
    if len(values) != grid.size:
        raise ValueError(f"{name} must hold rows x columns = {grid.size} values")
    if any(not math.isfinite(v) or (nonnegative and v < 0) for v in values):
        raise ValueError(f"{name} must be finite" + (" and at least 0" if nonnegative else ""))
    if nonnegative and not any(v > 0 for v in values):
        raise ValueError(f"{name} must be positive somewhere")


class EmitterIntensityMap(StrictModel):
    """The relative exitance of an emitter over its plane (the Project's ``IntensityMapV1``): ``PIECEWISE_CONSTANT`` (each value
    over its cell; the emitting area is the grid's extent) or ``BILINEAR`` (between the grid points, over the rectangle through
    the outer points). The emitter's ``shape`` is the ``RECTANGLE`` of that area."""

    grid: EmitterGrid
    values: tuple[float, ...]
    interpolation: Literal["PIECEWISE_CONSTANT", "BILINEAR"]

    @model_validator(mode="after")
    def _values(self) -> EmitterIntensityMap:
        _flat(self.values, self.grid, "values", True)
        if self.interpolation == "BILINEAR" and (self.grid.columns < 2 or self.grid.rows < 2):
            raise ValueError("a BILINEAR map needs at least two grid points along each axis")
        return self


class EmitterField(StrictModel):
    """A coherent scalar field ``amplitude exp(i phase_rad)`` on a grid of the emitter's plane (the Project's
    ``SampledFieldSourceV1``), fully polarized (``jones`` in the emitter frame). Its rays start in proportion to
    ``amplitude^2`` (constant over each cell) along the field's local wavevector."""

    grid: EmitterGrid
    amplitude: tuple[float, ...]
    phase_rad: tuple[float, ...]
    jones: JonesVector

    @model_validator(mode="after")
    def _values(self) -> EmitterField:
        _flat(self.amplitude, self.grid, "amplitude", True)
        _flat(self.phase_rad, self.grid, "phase_rad", False)
        return self


class EmitterSpeckle(StrictModel):
    """A speckle field (the Project's ``SpeckleSourceV1``): an illumination of flat phase through ``realizations`` seeded
    Gaussian-correlated random phase screens (``optics/fields/speckle.py``), fully polarized (``jones``).
    ``illumination_size_mm``: a disc's diameter twice, a rectangle's width and height, a Gaussian's 1/e^2 radius twice."""

    grid: EmitterGrid
    illumination: Literal["UNIFORM_DISC", "UNIFORM_RECTANGLE", "GAUSSIAN"]
    illumination_size_mm: tuple[float, float]
    phase_rms_rad: float = Field(gt=0)
    correlation_length_mm: float = Field(gt=0)
    seed: int = Field(ge=0)
    realizations: int = Field(ge=1)
    jones: JonesVector

    @model_validator(mode="after")
    def _sizes(self) -> EmitterSpeckle:
        if not all(s > 0 for s in self.illumination_size_mm):
            raise ValueError("illumination_size_mm must be positive")
        if self.grid.columns < 2 or self.grid.rows < 2:
            raise ValueError("a speckle grid needs at least two points along each axis")
        return self


class ExtendedEmitter(OmitAbsentFields, StrictModel):
    """A finite planar emitter sampled by rays (the Project's ``ExtendedEmitterSourceV1`` states its frame and normalization).
    ``power_W``: total radiant power into the law's domain; ``spectrum``: the lines' relative powers. A ``DISC`` has
    ``size_x_mm == size_y_mm``, its diameter.

    Laws: ``LAMBERTIAN``, ``ISOTROPIC``; ``COSINE_POWER`` (``angular_parameter`` = the exponent ``n``), ``UNIFORM_CONE``
    (``angular_parameter`` = the half angle in degrees), ``COLLIMATED`` (along +z). Optional, at most one: ``intensity_map``
    (a sampled exitance; unpolarized, incoherent rays), ``field`` or ``speckle`` (a coherent field of one line, ``COLLIMATED``:
    each ray along the field's local wavevector with the field's polarization; the run adds the rays' powers, the field's
    coherent propagation is ``results.field``). Absent, the emission is uniform over the shape, unpolarized and incoherent."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("angular_parameter", "intensity_map", "field", "speckle")

    source_id: Id
    local_pose: Pose3D = Pose3D()
    spectrum: tuple[SpectralLine, ...] = Field(min_length=1)
    power_W: float = Field(gt=0)
    shape: Literal["RECTANGLE", "DISC"]
    size_x_mm: float = Field(gt=0)
    size_y_mm: float = Field(gt=0)
    angular_law: Literal["LAMBERTIAN", "ISOTROPIC", "COSINE_POWER", "UNIFORM_CONE", "COLLIMATED"]
    sample_count: int = Field(ge=1)
    seed: int = Field(ge=0)
    medium: Id = "air"
    angular_parameter: float | None = Field(None, gt=0, description="COSINE_POWER: the exponent; UNIFORM_CONE: the half angle, deg.")
    intensity_map: EmitterIntensityMap | None = None
    field: EmitterField | None = None
    speckle: EmitterSpeckle | None = None

    @model_validator(mode="after")
    def _pattern(self) -> ExtendedEmitter:
        if (self.angular_parameter is None) != (self.angular_law not in ("COSINE_POWER", "UNIFORM_CONE")):
            raise ValueError("angular_parameter is given with a COSINE_POWER or UNIFORM_CONE law, and only with it")
        if self.angular_law == "UNIFORM_CONE" and self.angular_parameter > 90.0:  # type: ignore[operator]
            raise ValueError("a UNIFORM_CONE half angle is at most 90 degrees")
        if sum(x is not None for x in (self.intensity_map, self.field, self.speckle)) > 1:
            raise ValueError("an emitter states at most one of intensity_map, field and speckle")
        if (self.field is not None or self.speckle is not None) and (self.angular_law != "COLLIMATED" or len(self.spectrum) != 1):
            raise ValueError("a field or speckle emitter is COLLIMATED (its rays follow the field's wavevector) and of one line")
        if any(x is not None for x in (self.intensity_map, self.field, self.speckle)) and self.shape != "RECTANGLE":
            raise ValueError("a sampled emitter's shape is the RECTANGLE of its grid")
        return self


class PolygonRegion(StrictModel):
    """Named 2.5D region: polygon in the component local XY plus a local z interval."""

    region_id: Id
    polygon_mm: tuple[Vector2, ...] = Field(min_length=3)
    z_range_mm: tuple[float, float]

    @model_validator(mode="after")
    def _z(self) -> PolygonRegion:
        if self.z_range_mm[1] < self.z_range_mm[0]:
            raise ValueError("z_range_mm must satisfy min <= max")
        return self


class EngineeringEnvelopeSnapshot(StrictModel):
    """Layout-relevant envelope of a component."""

    frame_pose: Pose3D = Field(
        Pose3D(), description="Pose of the envelope frame relative to the component frame. The footprint polygon, z range, keepout, "
        "tool and cable regions live in this frame's XY plane / Z axis, which must be parallel to the baseplate for plan-view "
        "checks (for optical elements whose local z is the beam axis use Pose3D.plan_frame_for_optical_axis())."
    )
    footprint_mm: tuple[Vector2, ...] = Field(min_length=3, description="Component footprint polygon in the envelope-frame XY plane.")
    z_range_mm: tuple[float, float] = Field((0.0, 0.0), description="Local z extent of the body.")
    keepout_margin_mm: float = Field(0.0, ge=0, description="Isotropic keepout dilation of the footprint.")
    keepout_polygons: tuple[PolygonRegion, ...] = ()
    required_clearance_mm: float = Field(0.0, ge=0, description="Clearance required to any other component.")
    optical_center_local_mm: Vector3 | None = Field(
        None, description="Nominal optical axis point (local); its global z is the nominal optical height."
    )
    optical_height_tolerance_mm: float = Field(0.5, gt=0)
    tool_access: tuple[PolygonRegion, ...] = ()
    cable_keepout: tuple[PolygonRegion, ...] = ()
    participates_in_beam: bool = Field(True, description="False for elements that never intersect beams (e.g., bases).")

    @model_validator(mode="after")
    def _z(self) -> EngineeringEnvelopeSnapshot:
        if self.z_range_mm[1] < self.z_range_mm[0]:
            raise ValueError("z_range_mm must satisfy min <= max")
        return self


class ActuatorSpec(OmitAbsentFields, StrictModel):
    """One controllable degree of freedom of a component.

    Unit is fixed by ``dof``: ``translate`` -> mm, ``rotate`` -> degrees. The optional stated limits of the stage that
    provides it (a Project's retained DOF lowers into one: ``local/lowering.py``): ``hard_limit_min`` / ``hard_limit_max``
    (the mechanical end stops, containing the travel), ``repeatability`` (the +/- spread of returning to a commanded
    position) and the drift magnitudes ``drift_per_kelvin`` (per kelvin of temperature change) and ``drift_per_hour``, each
    in the DOF's unit. Targets on ``dof_travel_margin_*``, ``dof_drift_*`` and ``dof_repeatability_*`` compare them
    (``validation/mechanical.py``); the trace does not read them.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("hard_limit_min", "hard_limit_max", "repeatability", "drift_per_kelvin",
                                                  "drift_per_hour")

    actuator_id: Id
    dof: Literal["translate", "rotate"]
    axis_local: Vector3
    pivot_local_mm: Vector3 = (0.0, 0.0, 0.0)
    travel_min: float
    travel_max: float
    max_step: float | None = Field(None, gt=0, description="Largest command change per control iteration.")
    resolution: float | None = Field(None, gt=0)
    zero: float = 0.0
    hard_limit_min: float | None = Field(None, description="Mechanical end stop below the travel (the DOF's unit).")
    hard_limit_max: float | None = Field(None, description="Mechanical end stop above the travel (the DOF's unit).")
    repeatability: float | None = Field(None, gt=0, description="+/- spread of returning to a commanded position (the DOF's unit).")
    drift_per_kelvin: float | None = Field(None, gt=0, description="Drift magnitude per kelvin of temperature change (the DOF's unit / K).")
    drift_per_hour: float | None = Field(None, gt=0, description="Drift magnitude per hour (the DOF's unit / h).")

    @model_validator(mode="after")
    def _travel(self) -> ActuatorSpec:
        if self.travel_max < self.travel_min:
            raise ValueError("travel_max must be >= travel_min")
        if not (self.travel_min <= self.zero <= self.travel_max):
            raise ValueError("zero must lie within the travel range")
        if (self.hard_limit_min is None) != (self.hard_limit_max is None):
            raise ValueError("hard_limit_min and hard_limit_max come together")
        if self.hard_limit_min is not None and not (self.hard_limit_min <= self.travel_min and self.travel_max <= self.hard_limit_max):
            raise ValueError("the hard limits must contain the travel")
        return self


#: A rating's test polarization in machine-readable form: ``LINEAR`` (any orientation), ``CIRCULAR`` (either handedness),
#: ``S`` / ``P`` (linear, perpendicular / parallel to the plane of incidence at the rated surface), ``UNPOLARIZED``.
RatingPolarization = Literal["LINEAR", "CIRCULAR", "S", "P", "UNPOLARIZED"]

#: Plain statements of a rating's test polarization (lower case, single spaces, without a final full stop) -> the
#: ``RatingPolarization`` they state (``rating_polarization``). The catalog's canonical forms of a published S-Pol / P-Pol
#: are ``s-polarized`` / ``p-polarized``; ``TE`` / ``TM`` are s / p at the rated surface.
RATING_POLARIZATION_TEXT: dict[str, str] = {
    **{text: "S" for text in ("s", "s-polarized", "s-polarised", "s polarized", "s polarised", "s-pol", "s pol",
                              "s-polarization", "s polarization", "s-polarized light", "s-polarised light", "te",
                              "te-polarized", "te-polarised", "te polarized", "te polarised")},
    **{text: "P" for text in ("p", "p-polarized", "p-polarised", "p polarized", "p polarised", "p-pol", "p pol",
                              "p-polarization", "p polarization", "p-polarized light", "p-polarised light", "tm",
                              "tm-polarized", "tm-polarised", "tm polarized", "tm polarised")},
    **{text: "LINEAR" for text in ("linear", "linearly polarized", "linearly polarised", "linear polarization",
                                   "linear polarisation", "linearly polarized light", "linearly polarised light")},
    **{text: "CIRCULAR" for text in ("circular", "circularly polarized", "circularly polarised", "circular polarization",
                                     "circular polarisation", "circularly polarized light", "circularly polarised light")},
    **{text: "UNPOLARIZED" for text in ("unpolarized", "unpolarised", "randomly polarized", "randomly polarised",
                                        "random polarization", "random polarisation", "unpolarized light", "unpolarised light",
                                        "random")},
}

#: Words of a polarization statement and the ``RatingPolarization`` each names (``rating_polarization_terms``): a word for
#: a linear orientation names ``LINEAR``, a handedness ``CIRCULAR``.
RATING_POLARIZATION_WORDS: dict[str, str] = {
    **{word: "S" for word in ("s", "te")},
    **{word: "P" for word in ("p", "tm")},
    **{word: "LINEAR" for word in ("linear", "linearly", "vertical", "vertically", "horizontal", "horizontally")},
    **{word: "CIRCULAR" for word in ("circular", "circularly", "rhc", "lhc", "rcp", "lcp", "helical")},
    **{word: "UNPOLARIZED" for word in ("unpolarized", "unpolarised", "random", "randomly", "depolarized", "depolarised")},
}


def _plain(text: str) -> str:
    return " ".join(text.strip().lower().split()).rstrip(".")


def rating_polarization(text: str | None) -> str | None:
    """The machine-readable test polarization a rating's ``polarization_condition`` states (``RATING_POLARIZATION_TEXT``;
    case, spacing and a final full stop ignored), ``None`` for any other text."""
    if not text:
        return None
    return RATING_POLARIZATION_TEXT.get(_plain(text))


def rating_polarization_terms(text: str | None) -> frozenset[str]:
    """The ``RatingPolarization`` values the words of a polarization statement name (``RATING_POLARIZATION_WORDS``; any
    text, e.g. ``TE (s-polarized)`` names ``S``, ``linear, vertical`` names ``LINEAR``)."""
    if not text:
        return frozenset()
    words = "".join(ch if ch.isalpha() else " " for ch in text.lower()).split()
    return frozenset(RATING_POLARIZATION_WORDS[w] for w in words if w in RATING_POLARIZATION_WORDS)


def rating_polarization_conflict(field: str | None, text: str | None) -> str | None:
    """The condition a rating's ``polarization_condition`` names that its ``polarization`` field contradicts (``None``:
    none): any other condition it names -- except ``LINEAR`` beside an ``S`` / ``P`` field, of which it is the general
    form (``linear, s-polarized``)."""
    if field is None or not text:
        return None
    stated = rating_polarization(text)
    if stated is not None:
        return None if stated == field else stated
    for named in sorted(rating_polarization_terms(text)):
        if named != field and not (named == "LINEAR" and field in ("S", "P")):
            return named
    return None


class PowerRating(OmitAbsentFields, StrictModel):
    """A power-handling rating (laser-induced damage threshold) of an element and the conditions it was established under.

    ``regime`` ``CW``: ``value`` in ``W/cm`` (power per beam diameter) or ``W/cm2`` (irradiance); ``PULSED``: a fluence in
    ``J/cm2`` for pulses of ``pulse_duration_s`` (at ``repetition_rate_hz`` when stated). ``pulse_count`` is the number of
    pulses the rating states (1 for a single-pulse rating, read as one pulse on each tested site); ``test_protocol`` how they were
    applied when the rating states it (``S_ON_1``: each tested site received ``pulse_count`` pulses at one fluence level).
    ``value_definition`` (``J/cm2`` / ``W/cm2``): ``PEAK`` (the test beam's peak value) or ``TOTAL_PER_1E2_AREA`` (the
    total energy or power over pi (d/2)^2, d the test beam's 1/e^2 diameter: half the peak of a Gaussian test beam);
    ``None`` when the rating does not state it (the value then establishes no threshold for a peak exposure). ``wavelength_nm`` is the test wavelength,
    ``beam_diameter_mm`` the diameter of the test beam on the sample and ``beam_diameter_definition`` how that diameter is
    measured (``1/E2``, ``FWHM``; ``NOT_STATED`` or absent when the rating does not say), ``polarization_condition`` the stated test polarization as published,
    ``polarization`` the same condition in machine-readable form (``RatingPolarization``; compared with the polarization
    of every beam at the rated surface), ``status`` the fact status and ``basis`` where the rating comes from; ``origin``
    names the product or model that states it. ``surface_kind`` is the kind of surface the rating is for:
    ``METALLIC_REFLECTOR`` (a metal-coated reflector), ``OTHER`` (a dielectric coating, an uncoated or transmissive
    surface), absent when the element's surfaces include a reflector of unknown kind; on a metallic reflector a pulsed
    rating gives no direction across beam sizes unless it is a single-pulse rating (``RatingComparison``). A
    ``polarization_condition`` without ``polarization`` is a condition the exposure comparison cannot check (not
    verifiable: the rating does not state it in a form the comparison reads). A ``polarization_condition`` must not name
    another condition than ``polarization`` when both are given (``rating_polarization_conflict``: a rating whose text says
    ``s-polarized``, ``TM`` or ``circular (RHC)`` and whose field says ``LINEAR`` is refused); a text that is not a plain
    form of the field's condition (``linear, vertical``) states more than the field, and the exposure comparison reports
    that part of it as a condition it cannot check. ``surface_ids`` limits it to some of the component's surfaces (a member
    of a compound unit); empty = every surface of the component."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("polarization", "surface_kind", "value_definition", "test_protocol")

    regime: Literal["CW", "PULSED"]
    value: float = Field(gt=0)
    unit: Literal["W/cm", "W/cm2", "J/cm2"]
    wavelength_nm: float = Field(gt=0)
    pulse_duration_s: float | None = Field(None, gt=0)
    repetition_rate_hz: float | None = Field(None, gt=0)
    beam_diameter_mm: float | None = Field(None, gt=0)
    beam_diameter_definition: Literal["1/E2", "FWHM", "NOT_STATED"] | None = None
    pulse_count: int | None = Field(None, ge=1)
    test_protocol: Literal["S_ON_1"] | None = None
    value_definition: Literal["PEAK", "TOTAL_PER_1E2_AREA"] | None = None
    polarization_condition: str | None = None
    polarization: RatingPolarization | None = Field(
        None, description="The test polarization in machine-readable form (LINEAR, CIRCULAR, S, P, UNPOLARIZED), compared with "
                          "the beams' polarization at the rated surface; omitted when the rating states none or states it only "
                          "as text.")
    status: str = Field(min_length=1)
    basis: str = ""
    origin: str = ""
    surface_ids: tuple[str, ...] = ()
    surface_kind: Literal["METALLIC_REFLECTOR", "OTHER"] | None = Field(
        None, description="The kind of surface the rating is for: METALLIC_REFLECTOR (a metal-coated reflector) or OTHER (a "
                          "dielectric coating, an uncoated or transmissive surface); omitted when a reflecting surface of the "
                          "element is of unknown kind.")

    @model_validator(mode="after")
    def _regime(self) -> PowerRating:
        if self.regime == "PULSED":
            if self.unit != "J/cm2" or self.pulse_duration_s is None:
                raise ValueError("a PULSED rating is a fluence in J/cm2 and states pulse_duration_s")
        elif self.unit == "J/cm2" or self.pulse_duration_s is not None or self.repetition_rate_hz is not None or self.pulse_count is not None:
            raise ValueError("a CW rating is in W/cm or W/cm2 and has no pulse fields")
        if self.value_definition is not None and self.unit == "W/cm":
            raise ValueError("value_definition applies to a J/cm2 or W/cm2 value; a W/cm value is a power per beam diameter")
        if self.test_protocol is not None and self.pulse_count is None:
            raise ValueError("test_protocol says how a pulsed rating's pulse_count pulses were applied: state pulse_count")
        stated = rating_polarization_conflict(self.polarization, self.polarization_condition)
        if stated is not None:
            raise ValueError(f"polarization {self.polarization!r} disagrees with polarization_condition "
                             f"{self.polarization_condition!r}, which states {stated!r}")
        return self


class EngineeringComponentSnapshot(OmitAbsentFields, StrictModel):
    """Engine input for one placed component: its pose, surfaces, sources, envelope, actuators and power ratings."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("power_ratings", "emitter", "sample_emission")

    instance_id: Id
    role: str | None = None
    pose: Pose3D = Pose3D()
    surfaces: tuple[OpticalSurface, ...] = ()
    source: SourceSpec | None = None
    emitter: ExtendedEmitter | None = None
    sample_emission: SampleEmission | None = Field(
        None, description="A fluorescence or Raman sample: how its emitter's line powers follow from the excitation delivered to "
        "it (contracts/detection.py). The emitter carries exactly those lines; none when the sample emits nothing.")
    ports: tuple[OpticalPort, ...] = ()
    envelope: EngineeringEnvelopeSnapshot | None = None
    actuators: tuple[ActuatorSpec, ...] = ()
    power_ratings: tuple[PowerRating, ...] = Field(
        (), description="Power-handling ratings of the element, compared with the per-surface exposure of the run (ValidationResult.exposures).")

    @model_validator(mode="after")
    def _unique(self) -> EngineeringComponentSnapshot:
        for label, ids in (
            ("surface", [s.surface_id for s in self.surfaces]),
            ("port", [p.port_id for p in self.ports]),
            ("actuator", [a.actuator_id for a in self.actuators]),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} ids in component {self.instance_id!r}")
        if self.sample_emission is not None:
            emitted = tuple((line.wavelength_nm, line.power_W) for line in self.sample_emission.lines if line.power_W > 0.0)
            if self.emitter is None:
                if emitted:
                    raise ValueError(f"sample {self.instance_id!r} emits {self.sample_emission.emitted_power_W!r} W but has no emitter")
            elif (tuple((line.wavelength_nm, line.weight) for line in self.emitter.spectrum) != emitted
                  or self.emitter.power_W != self.sample_emission.emitted_power_W or self.emitter.source_id != self.instance_id):
                raise ValueError(f"the emitter of sample {self.instance_id!r} must carry exactly the derived emission lines and power")
        known = {s.surface_id for s in self.surfaces}
        for rating in self.power_ratings:
            unknown = sorted(set(rating.surface_ids) - known)
            if unknown:
                raise ValueError(f"power rating of component {self.instance_id!r} names unknown surfaces {unknown}")
        return self
