"""Optics contracts: apertures, interactions, OpticalSurface, OpticalPort, Ray, RaySegment,
OpticalPath, BeamState, GaussianBeamState.

Surface geometry is given in the *owning frame* (component local frame for components; global for
observation planes). Normals point toward the "positive side".
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator

from ..errors.status import Status
from .base import Complex, Id, OmitAbsentFields, StrictModel, Vector2, Vector3, unit_direction
from .basis import BasisCarrier, ValidityCheck, ValueBasisField
from .geometry import Pose3D
from .materials import CoatingModel
from .._vendor.shared_contracts.common.polygons import MAX_POLYGON_VERTICES, polygon_problem
from .polarization import JonesVector, PolarizationBasis

# --------------------------------------------------------------------------------------------- apertures


class CircularAperture(StrictModel):
    shape: Literal["circle"] = "circle"
    radius_mm: float = Field(gt=0, description="Clear aperture radius (not diameter).")
    center_offset_mm: Vector2 = (0.0, 0.0)
    obstruction_radius_mm: float = Field(0.0, ge=0, description="Central circular obstruction radius.")

    @model_validator(mode="after")
    def _obstruction(self) -> CircularAperture:
        if self.obstruction_radius_mm >= self.radius_mm:
            raise ValueError("obstruction_radius_mm must be smaller than radius_mm")
        return self


class RectangularAperture(StrictModel):
    shape: Literal["rectangle"] = "rectangle"
    half_width_mm: float = Field(gt=0, description="Half extent along the surface u axis.")
    half_height_mm: float = Field(gt=0, description="Half extent along the surface v axis.")
    center_offset_mm: Vector2 = (0.0, 0.0)


class EllipticalAperture(StrictModel):
    """An ellipse with semi-axes ``semi_axes_mm`` (along its first axis, then across it); the first axis is turned
    ``rotation_deg`` from the surface u axis toward v."""

    shape: Literal["ellipse"] = "ellipse"
    semi_axes_mm: tuple[float, float] = Field(description="Semi-axes (not diameters) along the first axis and across it.")
    rotation_deg: float = Field(0.0, ge=-180.0, le=180.0, description="Angle of the first axis from the u axis, toward v.")
    center_offset_mm: Vector2 = (0.0, 0.0)

    @model_validator(mode="after")
    def _positive(self) -> EllipticalAperture:
        if not all(value > 0.0 for value in self.semi_axes_mm):
            raise ValueError("semi_axes_mm must be positive")
        return self


class DShapedAperture(StrictModel):
    """A disc of radius ``radius_mm`` cut by a straight flat (a D shape): the points whose coordinate along the flat's
    outward normal (turned ``flat_angle_deg`` from the u axis toward v) is at most ``flat_distance_mm`` from the centre."""

    shape: Literal["d_shape"] = "d_shape"
    radius_mm: float = Field(gt=0, description="Radius of the round part (not diameter).")
    flat_distance_mm: float = Field(description="Signed distance of the flat from the centre along its outward normal (|d| < radius).")
    flat_angle_deg: float = Field(0.0, ge=-180.0, le=180.0, description="Direction of the flat's outward normal from the u axis, toward v.")
    center_offset_mm: Vector2 = (0.0, 0.0)

    @model_validator(mode="after")
    def _flat(self) -> DShapedAperture:
        if not abs(self.flat_distance_mm) < self.radius_mm:
            raise ValueError("|flat_distance_mm| must be smaller than radius_mm (the flat cuts the disc)")
        return self


class PolygonAperture(StrictModel):
    """A simple polygon (convex or not; a triangle has 3 corners): corners ``vertices_mm`` relative to
    ``center_offset_mm``, counter-clockwise in (u, v), no two edges crossing or touching."""

    shape: Literal["polygon"] = "polygon"
    vertices_mm: tuple[Vector2, ...] = Field(min_length=3, max_length=MAX_POLYGON_VERTICES,
                                             description="Corners (u, v) relative to center_offset_mm, counter-clockwise.")
    center_offset_mm: Vector2 = (0.0, 0.0)

    @model_validator(mode="after")
    def _simple(self) -> PolygonAperture:
        problem = polygon_problem(self.vertices_mm)
        if problem is not None:
            raise ValueError(problem)
        return self


Aperture = Annotated[CircularAperture | RectangularAperture | EllipticalAperture | DShapedAperture | PolygonAperture,
                     Field(discriminator="shape")]

HOLES_DESCRIPTION = (
    "Openings through the element inside its clear aperture (holed mirrors, perforated plates), in the same in-surface "
    "(u, v) coordinates: there is no surface there and light passes straight through. Each hole lies inside the clear "
    "aperture and the holes lie apart (their enclosing circles about their centres do not overlap); a circular hole has no "
    "obstruction. Omitted: no hole."
)

# ------------------------------------------------------------------------------------------ interactions


class _OmitAbsent:
    """Serializes the optional fields named in ``_omit_when_none`` only when they are given, so a contract without them
    keeps its exact earlier form (and every Engine input hash written before they existed)."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for name in self._omit_when_none:
                if name in data and data[name] is None:
                    del data[name]
        return data


_PHASE_PAIRS = (("reflection_s_rad", "reflection_p_rad"), ("transmission_s_rad", "transmission_p_rad"))


class InterfacePhases(StrictModel):
    """Stated phases (rad) of an interface's s and p amplitude coefficients, replacing the interaction's own phases.

    Convention: amplitudes ``r = sqrt(R) exp(i phase)`` and ``t = sqrt(T) exp(i phase)`` per polarization in the s / p
    triad of the Engine (``s = d x n`` normalized, ``p = d x s``; ``d`` the propagation direction before (``s``) and after
    (``p = d_out x s``) the interface, ``n`` the surface normal), field ``exp[i(k.r - w t)]``. ``R`` and ``T`` are the
    interaction's power coefficients (its coating, or 1 for an uncoated mirror).

    ``reflection_s_rad`` / ``reflection_p_rad`` and ``transmission_s_rad`` / ``transmission_p_rad`` are stated in s / p
    pairs (at least one pair). Without ``wavelength_nm`` each stated phase holds one value, the same at every
    wavelength; with ``wavelength_nm`` (strictly increasing vacuum wavelengths) one value per wavelength, linear between
    neighbouring wavelengths and never extrapolated (at a wavelength outside the table the interface states no phase:
    its own phases apply and ``COATING_OUT_OF_RANGE`` is reported). Each interaction says which phases it uses.
    """

    wavelength_nm: tuple[float, ...] | None = Field(None, min_length=2, description="Vacuum wavelengths of a tabulated phase (nm).")
    reflection_s_rad: tuple[float, ...] | None = Field(None, min_length=1, description="Phase of r_s (rad).")
    reflection_p_rad: tuple[float, ...] | None = Field(None, min_length=1, description="Phase of r_p (rad).")
    transmission_s_rad: tuple[float, ...] | None = Field(None, min_length=1, description="Phase of t_s (rad).")
    transmission_p_rad: tuple[float, ...] | None = Field(None, min_length=1, description="Phase of t_p (rad).")

    @model_validator(mode="after")
    def _table(self) -> InterfacePhases:
        stated = 0
        for s_name, p_name in _PHASE_PAIRS:
            s_value, p_value = getattr(self, s_name), getattr(self, p_name)
            if (s_value is None) != (p_value is None):
                raise ValueError(f"{s_name} and {p_name} are stated together")
            stated += s_value is not None
        if not stated:
            raise ValueError("state reflection_s_rad / reflection_p_rad, transmission_s_rad / transmission_p_rad or both")
        count = 1 if self.wavelength_nm is None else len(self.wavelength_nm)
        if self.wavelength_nm is not None:
            if self.wavelength_nm[0] <= 0 or any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
                raise ValueError("wavelength_nm must be positive and strictly increasing")
        for s_name, p_name in _PHASE_PAIRS:
            for name in (s_name, p_name):
                values = getattr(self, name)
                if values is not None and len(values) != count:
                    raise ValueError(f"{name} holds {count} value(s): one per wavelength_nm node, or one without wavelength_nm")
        return self

    @property
    def states_reflection(self) -> bool:
        return self.reflection_s_rad is not None

    @property
    def states_transmission(self) -> bool:
        return self.transmission_s_rad is not None


#: The most rays one scattering surface sends out per beam that meets it.
SCATTER_SAMPLE_COUNT_MAX = 65536
#: The largest exponent of a COSINE_POWER scatter lobe.
SCATTER_EXPONENT_MAX = 1000.0


class SurfaceScatter(StrictModel):
    """Diffuse scatter of a rough surface: ``fraction`` of the power that meets the surface leaves it as incoherent,
    unpolarized light on ``side`` (``reflect``: back into the medium the light comes from; ``transmit``: into the medium
    behind the surface), with the angular law ``bsdf`` about the surface normal on that side; the rest meets the surface's
    own interaction (refraction, reflection, absorption) unchanged.

    ``LAMBERTIAN``: radiance independent of direction, intensity ``I(theta) = P cos(theta) / pi``. ``COSINE_POWER``: intensity
    ``I(theta) = P (n + 1) cos^n(theta) / (2 pi)`` with ``n = exponent`` (``n = 1`` is Lambertian; a larger ``n`` a narrower
    lobe about the normal). The scattered light of each beam that meets the surface is traced as ``sample_count`` rays from the
    beam's hit point (``seed``: the sequence's digital shift), whose powers add (no phase, no speckle, no diffraction). Light
    those rays scatter again at another scattering surface is booked, not followed."""

    bsdf: Literal["LAMBERTIAN", "COSINE_POWER"] = Field(
        "LAMBERTIAN", description="Angular law of the scattered light about the surface normal on its side.")
    exponent: float | None = Field(
        None, gt=0, le=SCATTER_EXPONENT_MAX, description="The exponent n of a COSINE_POWER lobe (intensity proportional to cos^n); "
        "only with COSINE_POWER.")
    fraction: float = Field(gt=0, le=1, description="Share of the power meeting the surface that is scattered.")
    side: Literal["reflect", "transmit"] = Field(
        "reflect", description="reflect: back into the medium the light arrives from; transmit: into the medium behind the surface.")
    sample_count: int = Field(1024, ge=2, le=SCATTER_SAMPLE_COUNT_MAX,
                              description="Rays traced for the scattered light of each beam meeting the surface.")
    seed: int = Field(0, ge=0, description="Digital shift of the rays' sampling sequence.")

    @model_validator(mode="after")
    def _law(self) -> SurfaceScatter:
        if (self.bsdf == "COSINE_POWER") != (self.exponent is not None):
            raise ValueError("exponent is stated with bsdf COSINE_POWER and only with it")
        return self


SCATTER_DESCRIPTION = ("Diffuse scatter of the surface (SurfaceScatter): a share of the light meeting it leaves as incoherent "
                       "light with a stated angular law; omitted: a smooth surface.")


class RefractInteraction(_OmitAbsent, StrictModel):
    """Dielectric interface: Snell refraction / TIR with Fresnel power (or a coating override).

    ``phases`` (only with a ``coating``): its transmission phases replace the bare Fresnel phases of the transmitted
    light, its reflection phases those of the reflected light; under total internal reflection the coating and its
    phases are not used (the Fresnel TIR amplitudes apply)."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("phases", "scatter")

    kind: Literal["refract"] = "refract"
    coating: CoatingModel | None = Field(None, description="If set, its T/R replace the bare Fresnel power.")
    trace_fresnel_reflection: bool = Field(False, description="Also trace the reflected branch as a child path.")
    phases: InterfacePhases | None = Field(None, description="Coating phases replacing the bare Fresnel phases (needs a coating).")
    scatter: SurfaceScatter | None = Field(None, description=SCATTER_DESCRIPTION)

    @model_validator(mode="after")
    def _phases_need_coating(self) -> RefractInteraction:
        if self.phases is not None and self.coating is None:
            raise ValueError("phases replace a coating's phases: a refracting interface with phases needs a coating")
        return self


class MirrorInteraction(_OmitAbsent, StrictModel):
    """Reflector. ``phases``: its reflection phases replace ``phase_s_rad`` / ``phase_p_rad`` (leave those at their
    defaults); its transmission phases apply to the light the coating transmits (recorded, not traced)."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("phases", "scatter")

    kind: Literal["mirror"] = "mirror"
    reflective_side: Literal["positive", "negative", "both"] = "positive"
    coating: CoatingModel | None = Field(None, description="R_s/R_p (ideal R=1 when omitted).")
    phase_s_rad: float = Field(math.pi, description="Reflection phase of the s component (ideal metal: pi).")
    phase_p_rad: float = Field(0.0, description="Reflection phase of the p component in the (s, d x s) triad (ideal metal: 0).")
    phases: InterfacePhases | None = Field(None, description="Stated (constant or wavelength-tabulated) phases; reflection phases "
                                           "replace phase_s_rad / phase_p_rad.")
    scatter: SurfaceScatter | None = Field(None, description=SCATTER_DESCRIPTION + " A reflector scatters on its reflecting side.")

    @model_validator(mode="after")
    def _one_reflection_phase(self) -> MirrorInteraction:
        if self.phases is not None and self.phases.states_reflection and (self.phase_s_rad != math.pi or self.phase_p_rad != 0.0):
            raise ValueError("phases.reflection_* replace phase_s_rad / phase_p_rad: state the reflection phases once")
        if self.scatter is not None and self.scatter.side != "reflect":
            raise ValueError("a reflector scatters on its reflecting side: scatter.side is reflect")
        return self


class ThinLensInteraction(StrictModel):
    """Ideal aberration-free thin lens in slope form (``s' = s - h/f``, exact at every angle)."""

    kind: Literal["thin_lens"] = "thin_lens"
    focal_length_mm: float = Field(description="Signed focal length (>0 converging). Must be non-zero.")
    transmittance: float = Field(1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _f(self) -> ThinLensInteraction:
        if self.focal_length_mm == 0.0:
            raise ValueError("focal_length_mm must be non-zero")
        return self


#: Relative wavelength tolerance a ``points_only`` row (and a one-row table) is matched within (0.5 %).
THIN_LENS_POINT_TOLERANCE = 0.005
#: Largest number of rows of a tabulated thin lens.
MAX_TABULATED_THIN_LENS_ROWS = 4096
#: Largest power fraction in the linear polarization orthogonal to a validated one that still counts as that
#: polarization: sin^2(1 deg).
POLARIZATION_MATCH_FRACTION = 3.046e-4


class TabulatedThinLensInteraction(StrictModel):
    """Thin lens whose focal length and transmittance are tabulated per vacuum wavelength (a response model, for example
    one fitted to an external simulation or measurement of a metalens).

    Each traced line takes, at its vacuum wavelength ``lambda``, a focal length ``f(lambda)`` and a transmittance
    ``T(lambda)`` from the rows (``wavelength_nm`` strictly increasing):

    * ``focal_interpolation = "linear_in_optical_power"``: the optical power ``1 / f`` is linear in ``lambda`` between
      the two rows around it; ``"points_only"``: the row within ``point_tolerance`` (relative) of ``lambda``;
    * ``transmittance_interpolation = "linear"``: ``T`` is linear in ``lambda`` between rows; ``"points_only"``: as above.

    The table's domain is the rows' span (a one-row table: its wavelength within ``point_tolerance``), and with
    ``points_only`` only the rows themselves. A line outside it is traced with the values of the nearest row and the
    Engine reports ``MODEL_WAVELENGTH_OUTSIDE_BAND`` (``OUT_OF_MODEL_SCOPE``); nothing is extrapolated.

    Rays are deflected in the slope form of ``ThinLensInteraction`` with ``f(lambda)`` (``s' = s - h / f``, exact at every
    angle), the Gaussian beam of each line takes the ideal thin-lens ABCD step of power ``cos(theta) / f(lambda)``, and the
    line's power is multiplied by ``T(lambda)``: the remaining ``1 - T(lambda)`` is the element's unmodelled response (power
    ledger sink ``UNMODELLED_RESPONSE``: light the response model does not deliver into its focused beam, such as other
    diffraction orders, reflection, scatter and absorption). The optical path length through the element is not modelled
    (as for an ideal thin lens), so phases, wavefronts and network exports of beams through it are refused.

    Validity domain (each violation is an ``OUT_OF_MODEL_SCOPE`` issue at this surface; the trace itself is unchanged; a
    computed angle or radius is beyond its maximum only when it exceeds it by more than 1e-9 of the maximum plus 1e-12 deg /
    1e-12 mm, so floating-point rounding of a value at the limit never takes it out of the domain):

    * ``max_field_angle_deg``: the chief ray's angle to the element axis (the surface normal) at the element,
      ``MODEL_FIELD_ANGLE_EXCEEDED`` beyond it;
    * ``max_beam_radius_mm``: the larger 1/e^2 radius of the arriving Gaussian beam, ``MODEL_BEAM_SIZE_EXCEEDED`` beyond
      it (also when the beam has no valid Gaussian state there, so its size is not known);
    * ``validated_polarizations``: ``"linear_u"`` / ``"linear_v"`` = light linearly polarized along the plane's in-plane
      ``u`` / ``v`` axis (``PlaneSurface.aperture_u_axis`` and ``normal x u``, projected onto the ray's transverse plane);
      ``MODEL_POLARIZATION_NOT_VALIDATED`` when the arriving chief-ray polarization is none of them (the power in the
      orthogonal linear component above ``POLARIZATION_MATCH_FRACTION``, sin^2 of 1 deg); ``None``: every polarization;
    * ``supported_metrics``: the target metrics the response was validated for (``None``: every metric). A target whose
      beams crossed this surface and whose metric is not listed is ``OUT_OF_MODEL_SCOPE`` with code
      ``METRIC_NOT_VALIDATED_FOR_ELEMENT``; its value is the ideal thin-lens step's, reported as an estimate and not graded.
    """

    kind: Literal["tabulated_thin_lens"] = "tabulated_thin_lens"
    wavelength_nm: tuple[float, ...] = Field(min_length=1, max_length=MAX_TABULATED_THIN_LENS_ROWS,
                                             description="Vacuum wavelengths of the rows (nm), strictly increasing.")
    focal_length_mm: tuple[float, ...] = Field(description="Signed focal length per row (>0 converging), non-zero.")
    transmittance: tuple[float, ...] = Field(description="Power fraction delivered into the focused beam per row, in (0, 1].")
    focal_interpolation: Literal["linear_in_optical_power", "points_only"]
    transmittance_interpolation: Literal["linear", "points_only"]
    point_tolerance: float = Field(THIN_LENS_POINT_TOLERANCE, gt=0, lt=1,
                                   description="Relative wavelength tolerance of points_only rows and of a one-row table.")
    max_field_angle_deg: float | None = Field(None, gt=0, lt=90, description="Largest chief-ray angle to the element axis.")
    max_beam_radius_mm: float | None = Field(None, gt=0, description="Largest 1/e^2 radius of the arriving Gaussian beam.")
    validated_polarizations: tuple[Literal["linear_u", "linear_v"], ...] | None = Field(None, min_length=1, max_length=2)
    supported_metrics: tuple[str, ...] | None = Field(None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _rows(self) -> TabulatedThinLensInteraction:
        n = len(self.wavelength_nm)
        if len(self.focal_length_mm) != n or len(self.transmittance) != n:
            raise ValueError("wavelength_nm, focal_length_mm and transmittance must have the same length")
        if any(w <= 0.0 for w in self.wavelength_nm) or any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
            raise ValueError("wavelength_nm must be positive and strictly increasing")
        if any(f == 0.0 for f in self.focal_length_mm):
            raise ValueError("focal_length_mm must be non-zero")
        if any(not 0.0 < t <= 1.0 for t in self.transmittance):
            raise ValueError("transmittance must lie in (0, 1]")
        if self.focal_interpolation == "linear_in_optical_power" and len({f > 0.0 for f in self.focal_length_mm}) > 1:
            raise ValueError("linear_in_optical_power needs focal lengths of one sign")
        if self.validated_polarizations is not None and len(set(self.validated_polarizations)) != len(self.validated_polarizations):
            raise ValueError("duplicate validated polarization")
        return self


#: Largest number of rows of a Faraday rotator's response.
MAX_FARADAY_ROTATOR_ROWS = 4096


class FaradayRotatorInteraction(StrictModel):
    """Thin nonreciprocal polarization rotator whose response is tabulated per vacuum wavelength.

    The field vector turns by ``theta(lambda)`` right-handed about the surface normal, whichever way the light travels: in
    a ray's right-handed transverse basis ``(e1, e2)``, ``e1 x e2 = d``, the Jones matrix is
    ``sqrt(T) [[cos b, -sin b], [sin b, cos b]]`` with ``b = sign(d . normal) theta``. A ray that returns through the plane
    after a normal-incidence mirror has turned by ``2 theta``. ``theta`` applies unchanged at every angle of incidence.

    ``wavelength_nm`` is strictly increasing. ``interpolation = "linear"``: ``theta`` and ``T`` are linear in ``lambda``
    between rows over the rows' span; ``"points_only"`` (and a one-row table): the rows' wavelengths exactly. A line outside
    that domain stops at the surface (``MODEL_WAVELENGTH_OUTSIDE_BAND``, ``OUT_OF_MODEL_SCOPE``, path termination
    ``FARADAY_RESPONSE_UNAVAILABLE``): nothing is extrapolated. The power ``1 - T`` is booked as
    ``ELEMENT_TRANSMISSION_LOSS``. The element has no thickness in the trace and states no optical path: per-ray optical
    path lengths are undefined behind it (as behind a thin lens), and network exports of beams through it are refused.
    """

    kind: Literal["faraday_rotator"] = "faraday_rotator"
    wavelength_nm: tuple[float, ...] = Field(min_length=1, max_length=MAX_FARADAY_ROTATOR_ROWS,
                                             description="Vacuum wavelengths of the rows (nm), strictly increasing.")
    rotation_rad: tuple[float, ...] = Field(description="Rotation per row, right-handed about the surface normal.")
    transmittance: tuple[float, ...] = Field(description="Power transmittance per row, in (0, 1].")
    interpolation: Literal["linear", "points_only"]

    @model_validator(mode="after")
    def _rows(self) -> FaradayRotatorInteraction:
        n = len(self.wavelength_nm)
        if len(self.rotation_rad) != n or len(self.transmittance) != n:
            raise ValueError("wavelength_nm, rotation_rad and transmittance must have the same length")
        if any(w <= 0.0 for w in self.wavelength_nm) or any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
            raise ValueError("wavelength_nm must be positive and strictly increasing")
        if any(not 0.0 < t <= 1.0 for t in self.transmittance):
            raise ValueError("transmittance must lie in (0, 1]")
        return self


#: The most orders a diffraction grating follows, and the most values of its response table.
MAX_GRATING_ORDERS = 65
MAX_GRATING_TABLE_VALUES = 8192


class GratingOrderTable(StrictModel):
    """A grating's stated order response on a grid of vacuum wavelengths and signed angles of incidence (deg, in the incident
    medium, positive toward the grating vector: ``sin(aoi)`` is the incident direction's component along it).

    ``efficiency_te`` / ``efficiency_tm`` hold ``orders x wavelength_nm x aoi_deg`` values, last index fastest (TE: the
    electric field along the grooves, TM: across them; an ``unpolarized`` table holds its mean in both), ``absorption_te`` /
    ``absorption_tm`` (``None``: not stated) ``wavelength_nm x aoi_deg`` values. ``interpolation = "linear"``: bilinear over
    the grid's span; ``"points_only"``: the grid's points. ``normalization = "absolute"``: fractions of the incident power,
    summing with the absorption to at most 1 per grid point and polarization; ``"relative"``: not fractions of it."""

    normalization: Literal["absolute", "relative"]
    polarization_basis: Literal["te_tm", "unpolarized"]
    interpolation: Literal["linear", "points_only"]
    data_kind: Literal["measured", "independently_computed", "assumed"]
    wavelength_nm: tuple[float, ...] = Field(min_length=1, max_length=MAX_GRATING_TABLE_VALUES)
    aoi_deg: tuple[float, ...] = Field(min_length=1, max_length=MAX_GRATING_TABLE_VALUES)
    orders: tuple[int, ...] = Field(min_length=1, max_length=MAX_GRATING_TABLE_VALUES)
    efficiency_te: tuple[float, ...]
    efficiency_tm: tuple[float, ...]
    absorption_te: tuple[float, ...] | None = None
    absorption_tm: tuple[float, ...] | None = None
    max_conical_angle_deg: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _grid(self) -> GratingOrderTable:
        for name, axis in (("wavelength_nm", self.wavelength_nm), ("aoi_deg", self.aoi_deg)):
            if any(b <= a for a, b in zip(axis, axis[1:])):
                raise ValueError(f"{name} must be strictly increasing")
        points = len(self.wavelength_nm) * len(self.aoi_deg)
        if len(self.efficiency_te) != points * len(self.orders) or len(self.efficiency_tm) != points * len(self.orders):
            raise ValueError("efficiency_te and efficiency_tm hold orders x wavelength_nm x aoi_deg values")
        if (self.absorption_te is None) != (self.absorption_tm is None):
            raise ValueError("state absorption_te and absorption_tm together")
        if self.absorption_te is not None and (len(self.absorption_te) != points or len(self.absorption_tm) != points):  # type: ignore[arg-type]
            raise ValueError("absorption_te and absorption_tm hold wavelength_nm x aoi_deg values")
        if any(v < 0.0 for v in (*self.efficiency_te, *self.efficiency_tm, *(self.absorption_te or ()), *(self.absorption_tm or ()))):
            raise ValueError("efficiencies and absorptions cannot be negative")
        return self


class DiffractionGratingInteraction(StrictModel):
    """Thin periodic plane: each order leaves by the vector grating equation ``n2 t2 = n1 t1 + m (lambda / L) K`` (``t``: a
    direction's part in the plane, ``K``: the unit grating vector, ``L``: the period, ``lambda``: the vacuum wavelength), on the
    incident side (``reflection``; light from the -normal side is blocked) or the far side (``transmission``, into the other
    medium). ``orders``: the orders followed; ``all_propagating``: they are a range, each followed where it propagates.
    Each order is a branch path.

    ``response`` states the order powers. Where it does not hold (none stated, outside its domain, a relative or unpolarized
    table, a conical mount beyond its limit, a transmission table met against +normal) the order powers are not known:
    the incident power goes to ``UNMODELLED_RESPONSE`` once and every order is followed with no power (``BeamState.power_W`` is
    ``None``), bounded from above. The element states no optical path: phase is not defined behind it and network exports
    through it are refused.
    """

    kind: Literal["diffraction_grating"] = "diffraction_grating"
    mode: Literal["reflection", "transmission"]
    period_nm: float = Field(gt=0.0)
    grating_vector_local: Vector3
    orders: tuple[int, ...] = Field(min_length=1, max_length=MAX_GRATING_ORDERS)
    all_propagating: bool = False
    response: GratingOrderTable | None = None


class BeamSplitterInteraction(_OmitAbsent, StrictModel):
    """Thin ideal splitter: reflected + transmitted branches; PBS/NPBS differ only by coating.

    Default phases: ``r_s = sqrt(R_s) exp(i reflection_phase_rad)``, ``r_p = -sqrt(R_p) exp(i reflection_phase_rad)``
    (a physical reflection: ``r_p = -r_s`` at normal incidence), ``t = sqrt(T) exp(i transmission_phase_rad)`` for s and
    p. ``phases``: its reflection phases replace those of ``r_s`` and ``r_p`` (leave ``reflection_phase_rad`` at its
    default), its transmission phases those of ``t_s`` and ``t_p`` (leave ``transmission_phase_rad`` at 0)."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("phases",)

    kind: Literal["beam_splitter"] = "beam_splitter"
    coating: CoatingModel
    reflection_phase_rad: float = Field(math.pi / 2, description="Phase of r relative to t (lossless symmetric splitter: pi/2).")
    transmission_phase_rad: float = 0.0
    phases: InterfacePhases | None = Field(None, description="Stated (constant or wavelength-tabulated) s / p phases of r and t.")

    @model_validator(mode="after")
    def _one_phase(self) -> BeamSplitterInteraction:
        if self.phases is not None:
            if self.phases.states_reflection and self.reflection_phase_rad != math.pi / 2:
                raise ValueError("phases.reflection_* replace reflection_phase_rad: state the reflection phases once")
            if self.phases.states_transmission and self.transmission_phase_rad != 0.0:
                raise ValueError("phases.transmission_* replace transmission_phase_rad: state the transmission phases once")
        return self


def _check_bands(bands: tuple[Any, ...] | None) -> None:
    for band in bands or ():
        if not band.min_nm <= band.max_nm:
            raise ValueError("a transmittance band needs min_nm <= max_nm")
    for a, b in zip(bands or (), (bands or ())[1:]):
        if b.min_nm < a.max_nm:
            raise ValueError("transmittance_bands are sorted by wavelength and do not overlap (neighbours may share a boundary)")


class PolarizerTransmittanceBand(StrictModel):
    """A linear polarizer's power transmittances for vacuum wavelengths in ``[min_nm, max_nm]`` (inclusive)."""

    min_nm: float = Field(gt=0)
    max_nm: float = Field(gt=0)
    transmittance_parallel: float = Field(ge=0, le=1)
    transmittance_perpendicular: float = Field(ge=0, le=1)


class RetarderTransmittanceBand(StrictModel):
    """A retarder's power transmittance for vacuum wavelengths in ``[min_nm, max_nm]`` (inclusive)."""

    min_nm: float = Field(gt=0)
    max_nm: float = Field(gt=0)
    transmittance: float = Field(ge=0, le=1)


TRANSMITTANCE_BANDS_DESCRIPTION = (
    "Wavelength-resolved transmittance: sorted bands (neighbours may share a boundary); a wavelength inside a band uses its "
    "values (on a shared boundary the conservative pair of the two bands: the smaller transmittance -- of a polarizer, along "
    "its axis -- and, for a polarizer, the larger perpendicular transmittance, the larger leak), a wavelength in no band the "
    "scalar values."
)


class PolarizerInteraction(_OmitAbsent, StrictModel):
    _omit_when_none: ClassVar[tuple[str, ...]] = ("transmittance_bands",)

    kind: Literal["polarizer"] = "polarizer"
    axis_local: Vector3 = Field(description="Transmission axis (element local frame), projected onto the ray's transverse plane.")
    transmittance_parallel: float = Field(1.0, ge=0, le=1)
    transmittance_perpendicular: float = Field(0.0, ge=0, le=1)
    transmittance_bands: tuple[PolarizerTransmittanceBand, ...] | None = Field(None, min_length=1, max_length=64,
                                                                               description=TRANSMITTANCE_BANDS_DESCRIPTION)

    @model_validator(mode="after")
    def _bands(self) -> PolarizerInteraction:
        _check_bands(self.transmittance_bands)
        return self


class RetarderInteraction(_OmitAbsent, StrictModel):
    _omit_when_none: ClassVar[tuple[str, ...]] = ("transmittance_bands", "retardance_table_nm", "retardance_table_rad",
                                                  "retardance_interpolation")

    kind: Literal["retarder"] = "retarder"
    fast_axis_local: Vector3
    retardance_rad: float = Field(description="Retardance at design_wavelength_nm (pi = HWP, pi/2 = QWP).")
    design_wavelength_nm: float = Field(gt=0)
    achromatic: bool = Field(False, description="If false, retardance scales as delta0 * lambda0 / lambda (zero order).")
    transmittance: float = Field(1.0, ge=0, le=1)
    transmittance_bands: tuple[RetarderTransmittanceBand, ...] | None = Field(None, min_length=1, max_length=64,
                                                                              description=TRANSMITTANCE_BANDS_DESCRIPTION)
    retardance_table_nm: tuple[float, ...] | None = Field(
        None, min_length=1, max_length=64, description="A retardance stated per vacuum wavelength (a dual-wavelength plate): the "
        "rows' wavelengths (nm), strictly increasing. It replaces retardance_rad / achromatic: linear in wavelength between rows over "
        "their span (retardance_interpolation 'linear') or at the rows' wavelengths only ('points_only'); a line elsewhere stops at "
        "the plate (MODEL_WAVELENGTH_OUTSIDE_BAND, OUT_OF_MODEL_SCOPE).")
    retardance_table_rad: tuple[float, ...] | None = Field(None, description="The retardance (rad) of each row of retardance_table_nm.")
    retardance_interpolation: Literal["linear", "points_only"] | None = None

    @model_validator(mode="after")
    def _bands(self) -> RetarderInteraction:
        _check_bands(self.transmittance_bands)
        table = (self.retardance_table_nm, self.retardance_table_rad, self.retardance_interpolation)
        if any(part is not None for part in table):
            if any(part is None for part in table):
                raise ValueError("retardance_table_nm, retardance_table_rad and retardance_interpolation are given together")
            rows = self.retardance_table_nm
            if len(self.retardance_table_rad) != len(rows):  # type: ignore[arg-type]
                raise ValueError("retardance_table_nm and retardance_table_rad have the same length")
            if any(w <= 0.0 for w in rows) or any(b <= a for a, b in zip(rows, rows[1:])):  # type: ignore[union-attr, arg-type]
                raise ValueError("retardance_table_nm is positive and strictly increasing")
        return self


class AbsorberInteraction(_OmitAbsent, StrictModel):
    """Absorbs the light that meets it. ``scatter`` (side ``reflect``): a diffusely reflecting absorber (a beam dump, a painted
    or anodized wall) sends that share back as scattered light and absorbs the rest."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("scatter",)

    kind: Literal["absorber"] = "absorber"
    scatter: SurfaceScatter | None = Field(None, description=SCATTER_DESCRIPTION + " An absorber scatters back only (side reflect).")

    @model_validator(mode="after")
    def _reflect_only(self) -> AbsorberInteraction:
        if self.scatter is not None and self.scatter.side != "reflect":
            raise ValueError("an absorber scatters back only: scatter.side is reflect")
        return self


class ApertureStopInteraction(StrictModel):
    """Transmits inside the aperture (optionally attenuated), blocks outside."""

    kind: Literal["aperture_stop"] = "aperture_stop"
    transmittance: float = Field(1.0, ge=0, le=1)


class FixedDensity(StrictModel):
    """One optical density everywhere on the plane."""

    kind: Literal["fixed"] = "fixed"
    optical_density: float = Field(ge=0)


class LinearRotaryDensity(StrictModel):
    """Optical density linear in the wheel azimuth: ``od_start`` at ``arc_start_deg`` to ``od_end`` at ``arc_end_deg``;
    outside the arc the plane is uncoated."""

    kind: Literal["linear_rotary"] = "linear_rotary"
    od_start: float = Field(ge=0)
    od_end: float = Field(ge=0)
    arc_start_deg: float = Field(ge=0, le=360)
    arc_end_deg: float = Field(ge=0, le=360)

    @model_validator(mode="after")
    def _arc(self) -> LinearRotaryDensity:
        if not self.arc_start_deg < self.arc_end_deg:
            raise ValueError("arc_start_deg must be smaller than arc_end_deg")
        return self


class DensityStep(StrictModel):
    """One sector ``[start_deg, end_deg]`` of wheel azimuth with optical density ``optical_density``."""

    start_deg: float = Field(ge=0, le=360)
    end_deg: float = Field(ge=0, le=360)
    optical_density: float = Field(ge=0)

    @model_validator(mode="after")
    def _arc(self) -> DensityStep:
        if not self.start_deg < self.end_deg:
            raise ValueError("start_deg must be smaller than end_deg")
        return self


class SteppedRotaryDensity(StrictModel):
    """Sectors of constant optical density (sorted, not overlapping); between them the plane is uncoated."""

    kind: Literal["stepped_rotary"] = "stepped_rotary"
    steps: tuple[DensityStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted(self) -> SteppedRotaryDensity:
        for a, b in zip(self.steps, self.steps[1:]):
            if b.start_deg < a.end_deg:
                raise ValueError("density steps must be sorted by start_deg and must not overlap")
        return self


class LinearGradientDensity(StrictModel):
    """Optical density linear along the plane's x axis: ``od_start`` at ``x_start_mm`` to ``od_end`` at ``x_end_mm``;
    beyond that interval the plane is uncoated."""

    kind: Literal["linear_gradient"] = "linear_gradient"
    od_start: float = Field(ge=0)
    od_end: float = Field(ge=0)
    x_start_mm: float
    x_end_mm: float

    @model_validator(mode="after")
    def _interval(self) -> LinearGradientDensity:
        if not self.x_start_mm < self.x_end_mm:
            raise ValueError("x_start_mm must be smaller than x_end_mm")
        return self


DensityLaw = Annotated[FixedDensity | LinearRotaryDensity | SteppedRotaryDensity | LinearGradientDensity, Field(discriminator="kind")]


class NeutralDensityInteraction(_OmitAbsent, StrictModel):
    """Thin attenuator plane: power transmittance ``10^-OD`` with the optical density ``OD`` of ``law`` evaluated where each
    ray crosses the plane (so it varies across a beam's footprint), a constant power ``reflectance`` (the reflected beam is
    traced as a branch when it is not 0) and the rest absorbed.

    Plane coordinates: origin at the surface origin, ``x`` along ``x_axis_local`` (projected into the plane), ``y =
    normal x x``; the azimuth of a point is measured from ``+x`` toward ``+y`` in [0, 360). ``position`` turns a wheel
    (rotary laws, degrees from ``+x`` toward ``+y``: the law's azimuth is the point's azimuth minus ``position``) or shifts
    a gradient along ``+x`` (mm: the law's x is the point's x minus ``position``). ``coated_inner_radius_mm`` (a law that
    depends on position): the coating is an annulus around an uncoated hub, so a point closer than this radius to the law's
    centre (the surface origin; for a gradient, shifted by ``position`` along ``+x`` with the filter) is uncoated. Where the
    law leaves the plane uncoated the transmittance is ``uncoated_transmittance``; without it light there is out of model
    scope.

    The packet's power and Gaussian beam use the value at the chief ray; the run reports how much the density varies
    across the beam (``ND_DENSITY_VARIES_ACROSS_BEAM`` beyond ``ND_FOOTPRINT_OD_TOLERANCE``: across the Gaussian 1/e^2
    footprint, or across the live pupil rays when the beam has no valid Gaussian state), and the power of such a beam is
    out of model scope from there on (``ArrivalResult.power_out_of_scope_at``). ``reflectance`` plus the highest
    transmittance (the lowest optical density's, or ``uncoated_transmittance``) cannot exceed 1. The reflected branch takes
    the phase convention of an ideal reflector (``pi`` for s, ``0`` for p in the ``(s, d x s)`` triad).
    """

    kind: Literal["neutral_density"] = "neutral_density"
    law: DensityLaw
    x_axis_local: Vector3 = (1.0, 0.0, 0.0)
    position: float = 0.0
    uncoated_transmittance: float | None = Field(None, ge=0, le=1)
    reflectance: float = Field(0.0, ge=0, le=1)
    coated_inner_radius_mm: float | None = Field(None, gt=0)

    _omit_when_none: ClassVar[tuple[str, ...]] = ("coated_inner_radius_mm",)

    @model_validator(mode="after")
    def _annulus(self) -> NeutralDensityInteraction:
        if self.coated_inner_radius_mm is not None and isinstance(self.law, FixedDensity):
            raise ValueError("coated_inner_radius_mm bounds the coated annulus of a position-dependent law (not a fixed density)")
        highest = DECADE_BASE ** -_lowest_density(self.law)
        if self.uncoated_transmittance is not None:
            highest = max(highest, self.uncoated_transmittance)
        if self.reflectance + highest > 1.0 + ND_ENERGY_TOLERANCE:
            raise ValueError(f"reflectance + transmittance cannot exceed 1: reflectance {self.reflectance:.9g} with the highest "
                             f"transmittance {highest:.9g} (lowest optical density or uncoated transmittance) would create energy")
        return self


#: The most nodes of a sampled complex-transmission map (256 x 256).
TRANSMISSION_MAP_MAX_NODES = 65536
#: The most phase terms of one complex-transmission element, and the most Zernike terms of one term.
MAX_PHASE_TERMS = 16
MAX_ZERNIKE_PHASE_TERMS = 231


class LinearPhaseRamp(StrictModel):
    """A phase growing linearly across the plane: ``phi = g_u u + g_v v`` (rad; ``u``, ``v`` in mm from the surface origin
    along its in-plane axes). It steers light as a prism or a grating's first order does: ``n sin`` of the direction's
    tangential part grows by ``lambda g / (2 pi)``."""

    kind: Literal["linear_ramp"] = "linear_ramp"
    gradient_rad_per_mm: Vector2 = Field(description="(d phi / d u, d phi / d v) in rad/mm.")


class PhaseStep(StrictModel):
    """A phase step: ``phi = phase_rad`` where ``edge_normal . (u, v) >= edge_offset_mm``, 0 elsewhere (a step across a
    straight edge; ``edge_normal`` a unit (u, v) direction). Light crossing either side keeps its direction; a beam across
    the edge is not a Gaussian beam behind it."""

    kind: Literal["step"] = "step"
    phase_rad: float
    edge_normal: Vector2 = (1.0, 0.0)
    edge_offset_mm: float = 0.0

    @field_validator("edge_normal")
    @classmethod
    def _unit(cls, v: Vector2) -> Vector2:
        return unit_direction(v, UNIT_DIRECTION_TOL, "edge_normal")  # type: ignore[return-value]


class ZernikePhaseTerm(StrictModel):
    index: int = Field(ge=1, description="Zernike index in the stated indexing (1: piston).")
    value_rad: float = Field(description="Coefficient, in rad of phase.")


class ZernikePhase(StrictModel):
    """A Zernike phase ``phi = sum value_rad Z_index(rho, theta)``, ``rho = sqrt(u^2 + v^2) / normalization_radius_mm``:
    ``NOLL`` with ``RMS_ORTHONORMAL`` (unit RMS over the unit disc), ``FRINGE`` with ``PEAK_UNIT`` (unit peak), as for a
    Zernike sag. The element's clear aperture lies within the normalization radius."""

    kind: Literal["zernike"] = "zernike"
    indexing: Literal["NOLL", "FRINGE"]
    normalization: Literal["RMS_ORTHONORMAL", "PEAK_UNIT"]
    normalization_radius_mm: float = Field(gt=0)
    terms: tuple[ZernikePhaseTerm, ...] = Field(min_length=1, max_length=MAX_ZERNIKE_PHASE_TERMS)

    @model_validator(mode="after")
    def _pairing(self) -> ZernikePhase:
        if (self.indexing == "NOLL") != (self.normalization == "RMS_ORTHONORMAL"):
            raise ValueError("NOLL indexing pairs with RMS_ORTHONORMAL, FRINGE with PEAK_UNIT")
        if self.indexing == "FRINGE" and any(t.index > FRINGE_MAX_INDEX for t in self.terms):
            raise ValueError(f"FRINGE indices run to {FRINGE_MAX_INDEX}")
        return self


PhaseTerm = Annotated[LinearPhaseRamp | PhaseStep | ZernikePhase, Field(discriminator="kind")]
#: Slack of a unit direction written with a few decimals: within it the components are used unchanged, beyond it normalized.
UNIT_DIRECTION_TOL = 1e-6
#: The last FRINGE Zernike index.
FRINGE_MAX_INDEX = 37


class SampledTransmissionMap(StrictModel):
    """A complex transmission sampled on a regular grid: ``nodes_u`` x ``nodes_v`` nodes, ``pitch_mm`` apart along u and v,
    centred on ``center_mm`` (node ``(i, j)`` at ``u = center_u + (i - (nodes_u - 1) / 2) pitch_u``, likewise v), values
    listed row by row (``k = j nodes_u + i``). ``amplitude`` (0 .. 1, a passive element; omitted: 1 everywhere) and
    ``phase_rad`` (the unwrapped phase; omitted: 0) are interpolated bilinearly between nodes (the phase gradient is that of
    the bilinear patch). The map covers the element's clear aperture; structure finer than ``pitch_mm`` is not represented."""

    nodes_u: int = Field(ge=2, le=TRANSMISSION_MAP_MAX_NODES)
    nodes_v: int = Field(ge=2, le=TRANSMISSION_MAP_MAX_NODES)
    pitch_mm: Vector2
    center_mm: Vector2 = (0.0, 0.0)
    amplitude: tuple[float, ...] | None = None
    phase_rad: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _grid(self) -> SampledTransmissionMap:
        count = self.nodes_u * self.nodes_v
        if count > TRANSMISSION_MAP_MAX_NODES:
            raise ValueError(f"a map holds at most {TRANSMISSION_MAP_MAX_NODES} nodes")
        if self.pitch_mm[0] <= 0.0 or self.pitch_mm[1] <= 0.0:
            raise ValueError("pitch_mm must be positive")
        if self.amplitude is None and self.phase_rad is None:
            raise ValueError("a map states amplitude, phase_rad or both")
        for name in ("amplitude", "phase_rad"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValueError(f"{name} holds nodes_u x nodes_v = {count} values")
            if values is not None and not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} values must be finite")
        if self.amplitude is not None and any(a < 0.0 or a > 1.0 for a in self.amplitude):
            raise ValueError("amplitude lies in [0, 1] (a passive element)")
        return self


class ComplexTransmissionInteraction(_OmitAbsent, StrictModel):
    """A thin sample with a complex transmission ``t(u, v) = A(u, v) exp(i phi(u, v))`` over its plane, the same medium on both
    sides: ``A = amplitude x`` the map's amplitude, ``phi = phase_rad +`` the phase terms ``+`` the map's phase.

    Model (thin element, scalar, geometric): each ray keeps its polarization and gains the optical path ``phi / k0`` (``k0 = 2 pi /
    lambda``, vacuum) and the tangential wavevector ``grad phi`` (the local grating equation); a ray whose tangential direction
    would exceed 1 does not propagate (evanescent: retired; the chief ray: out of model scope). The beam's power is multiplied
    by the Gaussian-weighted mean of ``A^2`` over its footprint (Gauss-Legendre in the encircled power x a uniform azimuth rule,
    checked against a rule of half the nodes); a beam without a valid Gaussian state takes the chief ray's ``A^2`` and its
    power is out of model scope where ``A`` varies across its rays. A beam whose phase, less its best-fitting plane over the
    footprint, varies by more than a tolerance is not a Gaussian beam behind the element (its Gaussian quantities are
    unavailable; its rays and power go on). Diffraction by the structure (light outside the rays' directions) is not computed
    here: the imaging and focal-field analyses read the traced rays' optical paths."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("sampled_map",)

    kind: Literal["complex_transmission"] = "complex_transmission"
    amplitude: float = Field(1.0, ge=0, le=1, description="Uniform amplitude transmission factor (power factor amplitude^2).")
    phase_rad: float = Field(0.0, description="Uniform phase, in rad (an optical path phase_rad / k0).")
    phase_terms: tuple[PhaseTerm, ...] = Field((), max_length=MAX_PHASE_TERMS, description="Analytic phases added to phase_rad.")
    sampled_map: SampledTransmissionMap | None = Field(None, description="A sampled amplitude / phase map (omitted: none).")


#: Optical density is a base-10 logarithm: T = 10^-OD.
DECADE_BASE = 10.0
#: Slack of the reflectance + transmittance <= 1 check (decimal values lowered to floats).
ND_ENERGY_TOLERANCE = 1e-9


def _lowest_density(law: FixedDensity | LinearRotaryDensity | SteppedRotaryDensity | LinearGradientDensity) -> float:
    if isinstance(law, FixedDensity):
        return law.optical_density
    if isinstance(law, SteppedRotaryDensity):
        return min(step.optical_density for step in law.steps)
    return min(law.od_start, law.od_end)


Interaction = Annotated[
    RefractInteraction
    | MirrorInteraction
    | ThinLensInteraction
    | TabulatedThinLensInteraction
    | BeamSplitterInteraction
    | PolarizerInteraction
    | RetarderInteraction
    | AbsorberInteraction
    | ApertureStopInteraction
    | NeutralDensityInteraction
    | FaradayRotatorInteraction
    | DiffractionGratingInteraction
    | ComplexTransmissionInteraction,
    Field(discriminator="kind"),
]

# ------------------------------------------------------------------------------------------- surfaces

MOUNT_OUTLINE_DESCRIPTION = (
    "Outer boundary of the opaque mount around the clear aperture, in the same in-surface (u, v) coordinates as the "
    "aperture, which it must contain; when given it replaces the ring of mount_margin_mm (an element whose outline is not "
    "the clear aperture grown evenly, e.g. a rectangular plate or a decentred aperture). Rays inside it but outside the "
    "clear aperture are blocked; rays beyond it pass by. Omitted (null): the clear aperture grown by mount_margin_mm."
)


#: The most holes one surface has.
MAX_HOLES = 16


def _check_holes(aperture: Any, holes: tuple[Any, ...] | None) -> None:
    """The contract's own part of the hole rules (the geometry -- inside the clear aperture, apart -- is checked when the
    surface is lowered)."""
    if holes is None:
        return
    if aperture is None:
        raise ValueError("holes need a clear aperture")
    if isinstance(aperture, CircularAperture) and aperture.obstruction_radius_mm > 0.0:
        raise ValueError("a clear aperture with a central obstruction has no holes")
    if any(isinstance(hole, CircularAperture) and hole.obstruction_radius_mm > 0.0 for hole in holes):
        raise ValueError("a circular hole has no obstruction_radius_mm")


class _OptionalMountOutline:
    """Serializes ``mount_outline`` (and a sag surface's ``linear_sag_coefficient`` and ``sag_shape``) only when it is given, so
    a surface without one keeps its exact earlier form."""

    @model_serializer(mode="wrap")
    def _omit_absent_outline(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for name in ("mount_outline", "linear_sag_coefficient", "sag_shape", "holes"):
                if name in data and data[name] is None:
                    data.pop(name)
        return data


class PlaneSurface(_OptionalMountOutline, StrictModel):
    surface_id: Id
    geometry: Literal["plane"] = "plane"
    origin_mm: Vector3
    normal: Vector3
    aperture: Aperture | None = Field(None, description="None = unbounded plane.")
    aperture_u_axis: Vector3 | None = Field(None, description="In-plane u axis for rectangular apertures (default: deterministic basis).")
    mount_margin_mm: float = Field(
        0.0, ge=0, description="Width of the opaque mount ring around the clear aperture: rays hitting the ring are blocked "
        "(CLIPPED); rays beyond it (or beyond the aperture when 0) miss the element and pass by."
    )
    mount_outline: Aperture | None = Field(None, description=MOUNT_OUTLINE_DESCRIPTION)
    holes: tuple[Aperture, ...] | None = Field(None, min_length=1, max_length=MAX_HOLES, description=HOLES_DESCRIPTION)
    medium_positive: Id = "air"
    medium_negative: Id = "air"
    interaction: Interaction

    @model_validator(mode="after")
    def _outline(self) -> PlaneSurface:
        if self.mount_outline is not None and (self.aperture is None or self.mount_margin_mm != 0.0):
            raise ValueError("mount_outline needs a clear aperture and replaces mount_margin_mm (leave it 0)")
        _check_holes(self.aperture, self.holes)
        return self


class SphereSurface(StrictModel):
    """Spherical cap: ``|x - center| = radius``, points with ``(x-center).cap_axis > 0`` within ``aperture_radius_mm``
    (measured transversely from the cap axis). Normal is the outward radial vector; positive side = outside."""

    surface_id: Id
    geometry: Literal["sphere"] = "sphere"
    center_mm: Vector3
    radius_mm: float = Field(gt=0)
    cap_axis: Vector3 = Field(description="Unit vector from the centre toward the cap vertex.")
    aperture_radius_mm: float = Field(gt=0)
    mount_margin_mm: float = Field(0.0, ge=0, description="Opaque mount ring width around the clear aperture (see PlaneSurface).")
    medium_positive: Id = "air"
    medium_negative: Id = "air"
    interaction: RefractInteraction | MirrorInteraction | AbsorberInteraction

    @model_validator(mode="after")
    def _cap(self) -> SphereSurface:
        if self.aperture_radius_mm > self.radius_mm:
            raise ValueError("aperture_radius_mm must not exceed radius_mm")
        return self


class ZernikeSagTerm(StrictModel):
    index: int = Field(strict=True, ge=1, description="Term number in the shape's indexing.")
    value_mm: float = Field(description="Coefficient of the term (mm).")


class ZernikeSag(StrictModel):
    """Zernike sag added to the surface's radius, conic and aspheric terms: ``z += sum value_mm Z_index(rho, theta)``,
    ``rho = sqrt(a^2 + b^2) / normalization_radius_mm``, ``a = rho cos(theta) normalization_radius_mm``. ``NOLL`` pairs with
    ``RMS_ORTHONORMAL`` (unit RMS over the unit disc, index 4 .. 231), ``FRINGE`` with ``PEAK_UNIT`` (unit peak, index 4 .. 37).
    The terms are not referenced to the vertex. Piston and tilt are refused (``SAG_TERM_NOT_SUPPORTED``). The polynomial is
    defined at every radius: an aperture beyond the normalization radius evaluates the stated terms there (``rho > 1``; the
    reach is stated in ``SurfaceProfileEvidence.aperture_reach_mm``)."""

    kind: Literal["ZERNIKE"] = "ZERNIKE"
    indexing: Literal["NOLL", "FRINGE"]
    normalization: Literal["RMS_ORTHONORMAL", "PEAK_UNIT"]
    normalization_radius_mm: float = Field(gt=0)
    terms: tuple[ZernikeSagTerm, ...] = Field(min_length=1, max_length=231)


class XySagTerm(StrictModel):
    x_power: int = Field(strict=True, ge=0)
    y_power: int = Field(strict=True, ge=0)
    value_mm: float = Field(description="Coefficient of (a / R)^x_power (b / R)^y_power (mm).")


class XyPolynomialSag(StrictModel):
    """XY polynomial sag added to the surface's radius, conic and aspheric terms: ``z += sum value_mm (a / R)^x_power
    (b / R)^y_power``, ``R = normalization_radius_mm``, ``2 <= x_power + y_power <= 20``."""

    kind: Literal["XY_POLYNOMIAL"] = "XY_POLYNOMIAL"
    normalization_radius_mm: float = Field(gt=0)
    terms: tuple[XySagTerm, ...] = Field(min_length=1, max_length=231)


class BiconicSag(StrictModel):
    """``z = (c_a a^2 + c_b b^2) / (1 + sqrt(1 - (1 + k_a) c_a^2 a^2 - (1 + k_b) c_b^2 b^2))``; ``radius_x_mm`` and ``conic_x`` state
    the section along ``u``, ``radius_y_mm`` and ``conic_y`` the one along ``v`` (no radius: flat along that axis). The surface's
    ``curvature_per_mm``, ``conic`` and ``aspheric_coefficients`` stay 0 / empty."""

    kind: Literal["BICONIC"] = "BICONIC"
    radius_x_mm: float | None = None
    radius_y_mm: float | None = None
    conic_x: float | None = None
    conic_y: float | None = None


class ToroidalSag(StrictModel):
    """The curve ``z_b(b) = c b^2 / (1 + sqrt(1 - (1 + k) c^2 b^2)) + sum_i y_terms[i-1] b^(2i)`` over the ``v`` axis (``c = 1 /
    radius_y_mm``) rotated about the axis parallel to ``v`` through ``z = radius_of_rotation_mm`` (no radius of rotation: the curve
    extruded along ``u``). The surface's ``curvature_per_mm``, ``conic`` and ``aspheric_coefficients`` stay 0 / empty."""

    kind: Literal["TOROIDAL"] = "TOROIDAL"
    radius_y_mm: float | None = None
    conic: float | None = None
    y_terms: tuple[float, ...] = Field((), max_length=8)
    radius_of_rotation_mm: float | None = None


SagShape = Annotated[ZernikeSag | XyPolynomialSag | BiconicSag | ToroidalSag, Field(discriminator="kind")]


class SagSurface(_OptionalMountOutline, StrictModel):
    """A conic / even-asphere surface, rotationally symmetric or cylindrical, or a surface with a two-variable sag shape, in the
    owning frame.

    Local surface frame: vertex ``origin_mm``, axis ``normal`` (unit), ``u_axis`` (unit, perpendicular to ``normal``) and
    ``v = normal x u_axis``. A point at transverse coordinates ``(a, b)`` (along u, v) lies at
    ``origin + a u + b v + sag(rho) normal`` with ``rho = sqrt(a^2 + b^2)`` (rotational) or ``rho = |a|`` (``cylindrical``:
    curved along u, straight along v) and

        sag(rho) = c rho^2 / (1 + sqrt(1 - (1 + k) c^2 rho^2)) + sum_i A_(2i+4) rho^(2i+4) + L rho

    (``c = curvature_per_mm`` = 1 / vertex radius, positive when the centre of curvature lies on the +normal side;
    ``k = conic``: 0 sphere, -1 paraboloid, < -1 hyperboloid; ``aspheric_coefficients`` = (A4, A6, A8, ...) in
    mm^(1-2i-4); ``L = linear_sag_coefficient``, the axicon term: a cone of slope ``dz / d rho = L`` with its apex at the
    vertex, or on a cylinder a roof of two planes meeting along the straight axis; ``None`` = 0). The apex of an axicon
    term is a single point where the profile has no tangent plane: a ray meeting the surface within
    ``AXICON_APEX_RADIUS_MM`` of the apex (``rho`` below it) takes the surface axis as its normal, so a ray along the
    axis passes the apex undeviated, as it does by symmetry. The Gaussian beam model does not hold behind an axicon term
    (a cone turns a Gaussian beam into a Bessel-like zone): the beam's Gaussian state becomes invalid there
    (``GAUSSIAN_UNDEFINED_AFTER_AXICON``), while rays, powers and polarization are traced as on any surface. ``medium_positive`` is the medium on the +normal side of the surface (``z > sag``), ``medium_negative``
    the other one. The clear aperture is measured on ``(a, b)`` (circle / rectangle, optionally decentred: an off-axis
    section such as an off-axis paraboloid); ``mount_margin_mm`` is the opaque ring around it, as for planes. A
    ``MirrorInteraction`` reflects on ``reflective_side`` with ``positive`` = the +normal side.

    ``sag_shape`` makes the profile depend on both ``a`` and ``b``: a Zernike or XY polynomial shape adds to the terms above, a
    biconic or a toroid replaces them. It does not combine with ``cylindrical`` or an axicon term
    (``SAG_SHAPE_COMBINATION_NOT_SUPPORTED``), and its profile covers the clear aperture (``SAG_OUTSIDE_DOMAIN``).
    """

    surface_id: Id
    geometry: Literal["sag"] = "sag"
    origin_mm: Vector3
    normal: Vector3
    u_axis: Vector3
    curvature_per_mm: float = Field(description="Vertex curvature 1/R (signed, see the class docstring); 0 = no base curvature.")
    conic: float = 0.0
    aspheric_coefficients: tuple[float, ...] = Field((), max_length=8, description="A4, A6, ... (even polynomial terms).")
    cylindrical: bool = Field(False, description="Curved along u only (a cylinder / acylinder), straight along v.")
    linear_sag_coefficient: float | None = Field(
        None, description="Axicon term L (dz / d rho, dimensionless) added to the sag; None = no axicon term (0 is read as None). "
        "Serialized only when given.")
    sag_shape: SagShape | None = Field(None, description="Two-variable sag shape (Zernike, XY polynomial, biconic, toroid). Serialized only when given.")
    aperture: Aperture
    mount_margin_mm: float = Field(0.0, ge=0)
    mount_outline: Aperture | None = Field(None, description=MOUNT_OUTLINE_DESCRIPTION)
    holes: tuple[Aperture, ...] | None = Field(None, min_length=1, max_length=MAX_HOLES, description=HOLES_DESCRIPTION)
    medium_positive: Id = "air"
    medium_negative: Id = "air"
    interaction: RefractInteraction | MirrorInteraction | AbsorberInteraction | BeamSplitterInteraction

    @model_validator(mode="after")
    def _outline(self) -> SagSurface:
        if self.mount_outline is not None and self.mount_margin_mm != 0.0:
            raise ValueError("mount_outline replaces mount_margin_mm (leave it 0)")
        _check_holes(self.aperture, self.holes)
        return self

    @field_validator("linear_sag_coefficient")
    @classmethod
    def _no_axicon_term(cls, value: float | None) -> float | None:
        """An axicon term of zero is no axicon term: one form (``None``) for one surface."""

        return None if value == 0.0 else value


#: A ray meeting an axicon (``SagSurface.linear_sag_coefficient``) closer than this to the apex takes the surface axis as
#: its normal (mm): the apex is a single point without a tangent plane.
AXICON_APEX_RADIUS_MM = 1e-9

#: The most vertices and triangles of one facet surface.
MESH_MAX_VERTICES = 20000
MESH_MAX_TRIANGLES = 40000
#: A facet whose doubled area is at most this fraction of the square of its longest edge is degenerate (a sliver or a point).
MESH_DEGENERATE_AREA_FRACTION = 1e-12


class FacetApproximation(StrictModel):
    """What a facet surface stands for: ``FACETED_EXACT`` (the surface is these flat facets: a prism-like or polyhedral part)
    or ``TESSELLATED_FROM_CAD`` (flat facets approximating a curved surface, whose largest distance from it is
    ``max_chord_deviation_mm``); ``source``: where the facets come from (a CAD export, a measurement)."""

    kind: Literal["FACETED_EXACT", "TESSELLATED_FROM_CAD"]
    max_chord_deviation_mm: float | None = Field(None, ge=0, description="Largest distance of the facets from the surface they "
                                                 "approximate (TESSELLATED_FROM_CAD).")
    source: str | None = Field(None, max_length=512, description="Where the facets come from.")

    @model_validator(mode="after")
    def _deviation(self) -> FacetApproximation:
        if (self.kind == "TESSELLATED_FROM_CAD") != (self.max_chord_deviation_mm is not None):
            raise ValueError("max_chord_deviation_mm is stated for TESSELLATED_FROM_CAD facets and only for them")
        return self


def mesh_problem(vertices: tuple[tuple[float, float, float], ...], triangles: tuple[tuple[int, int, int], ...], closed: bool) -> str | None:
    """Why a triangle mesh is not a valid facet surface, or ``None``: every index names a vertex, no facet is degenerate, every
    edge belongs to at most two facets (a manifold sheet) that run it in opposite directions (one orientation), and a
    ``closed`` mesh (one bounding a solid) has every edge on exactly two facets and its facets facing outward (a positive
    enclosed volume)."""

    count = len(vertices)
    directed: dict[tuple[int, int], int] = {}
    volume = 0.0
    for k, (a, b, c) in enumerate(triangles):
        if not (0 <= a < count and 0 <= b < count and 0 <= c < count):
            return f"triangle {k} names a vertex outside 0 .. {count - 1}"
        if a == b or b == c or a == c:
            return f"triangle {k} repeats a vertex"
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        u = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        v = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        w = (pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2])
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        area2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
        longest = max(u[0] * u[0] + u[1] * u[1] + u[2] * u[2], v[0] * v[0] + v[1] * v[1] + v[2] * v[2], w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if area2 <= (MESH_DEGENERATE_AREA_FRACTION * longest) ** 2:
            return f"triangle {k} is degenerate (no area)"
        for edge in ((a, b), (b, c), (c, a)):
            if edge in directed:
                return (f"edge {edge[0]}-{edge[1]} is run in the same direction by triangles {directed[edge]} and {k}: the facets are not "
                        "oriented consistently, or more than two facets share the edge")
            directed[edge] = k
        volume += pa[0] * cross[0] + pa[1] * cross[1] + pa[2] * cross[2]
    for (a, b) in directed:
        if closed and (b, a) not in directed:
            return f"edge {a}-{b} belongs to one facet only: a closed facet surface (closed true) has no open edge"
    if closed and not volume > 0.0:
        return "the facets of a closed surface face inward (the enclosed volume is not positive): list each triangle counter-clockwise seen from outside"
    return None


#: The most facet groups of one facet surface.
MESH_MAX_FACET_GROUPS = 64


class MeshFacetGroup(StrictModel):
    """Facets of a facet surface with their own interaction (a coated, mirrored or blackened face of a faceted part):
    ``triangles`` (indices into the surface's ``triangles``) take ``interaction`` instead of the surface's."""

    triangles: tuple[int, ...] = Field(min_length=1, max_length=MESH_MAX_TRIANGLES)
    interaction: RefractInteraction | MirrorInteraction | AbsorberInteraction


class MeshSurface(_OmitAbsent, StrictModel):
    """A surface made of flat triangular facets in the owning frame: ``vertices_mm`` and ``triangles`` (vertex index triples,
    counter-clockwise seen from the positive side, which ``medium_positive`` fills), with the statement of what the facets
    stand for (``approximation``). ``closed``: the facets bound a solid (watertight, facing outward: ``medium_positive``
    outside, ``medium_negative`` inside). ``interaction`` acts on every facet but those of a ``facet_groups`` entry, which
    take that group's interaction.

    Facet surfaces are validated (indices, degenerate facets, a manifold sheet with one orientation, watertight when closed,
    ``MESH_MAX_VERTICES`` / ``MESH_MAX_TRIANGLES``) and traced facet by facet: the edge-connected facets of one group that
    lie in one plane form a planar patch, traced as a plane whose clear aperture is the patch's outline (exact for a
    ``FACETED_EXACT`` part: a prism, a corner cube, a polyhedral window). A ``TESSELLATED_FROM_CAD`` surface is traced as its
    facets (rays meet and leave the facets, not the curved surface they approximate within ``max_chord_deviation_mm``); its
    flat facets carry no curvature, so a beam's Gaussian state is not defined behind one
    (``GAUSSIAN_UNDEFINED_AFTER_FACETS``). The part of a beam that falls beyond a patch's edge onto a neighbouring facet is
    not followed by that beam (``FACET_EDGE_DIVIDES_BEAM``: its power's further fate is not known). Refused
    (``FACET_SURFACE_NOT_TRACED``): an open sheet that refracts between two different media (a solid needs closed facets),
    a uniaxial crystal on a side of the facets, more patches than the run's ``max_mesh_patches`` (compute budget; 1024
    unless the Project states another).

    ``placement``: where the facets sit in the owning frame, as a rigid pose (a vertex ``v`` is at ``R(q) v + position_mm``);
    omitted, the vertices are in the owning frame. The faces are found from the vertices as written (which facets lie in one
    plane is judged against the coordinates' rounding), so a part placed this way has the faces it has unplaced; the
    placement moves those faces."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ("facet_groups", "placement")

    surface_id: Id
    geometry: Literal["mesh"] = "mesh"
    vertices_mm: tuple[Vector3, ...] = Field(min_length=3, max_length=MESH_MAX_VERTICES)
    triangles: tuple[tuple[int, int, int], ...] = Field(min_length=1, max_length=MESH_MAX_TRIANGLES)
    approximation: FacetApproximation
    closed: bool = False
    medium_positive: Id = "air"
    medium_negative: Id = "air"
    interaction: RefractInteraction | MirrorInteraction | AbsorberInteraction
    facet_groups: tuple[MeshFacetGroup, ...] | None = Field(
        None, min_length=1, max_length=MESH_MAX_FACET_GROUPS,
        description="Facets with their own interaction (each triangle in at most one group); omitted: every facet takes interaction.")
    placement: Pose3D | None = Field(
        None, description="The rigid pose of the facets in the owning frame (a vertex v is at R(q) v + position_mm); omitted: the "
        "vertices are in the owning frame. The faces are found from the vertices as written and then placed.")

    @model_validator(mode="after")
    def _mesh(self) -> MeshSurface:
        problem = mesh_problem(self.vertices_mm, self.triangles, self.closed)
        if problem is not None:
            raise ValueError(problem)
        seen: dict[int, int] = {}
        for g, group in enumerate(self.facet_groups or ()):
            for k in group.triangles:
                if not 0 <= k < len(self.triangles):
                    raise ValueError(f"facet group {g} names triangle {k}, outside 0 .. {len(self.triangles) - 1}")
                if k in seen:
                    raise ValueError(f"triangle {k} is in facet groups {seen[k]} and {g}: a facet has one interaction")
                seen[k] = g
        return self

    def facet_group_of(self) -> tuple[int, ...]:
        """The facet group of each triangle (``0``: the surface's own ``interaction``; ``g + 1``: ``facet_groups[g]``)."""
        group = [0] * len(self.triangles)
        for g, entry in enumerate(self.facet_groups or ()):
            for k in entry.triangles:
                group[k] = g + 1
        return tuple(group)


OpticalSurface = Annotated[PlaneSurface | SphereSurface | SagSurface | MeshSurface, Field(discriminator="geometry")]


class OpticalPort(StrictModel):
    """Named entry/exit reference of a component (local frame); axis points along nominal propagation."""

    port_id: Id
    position_mm: Vector3
    axis: Vector3
    kind: Literal["input", "output", "bidirectional"] = "bidirectional"


# --------------------------------------------------------------------------------------------- results


class Ray(StrictModel):
    origin_mm: Vector3
    direction: Vector3
    wavelength_nm: float = Field(gt=0)
    power_W: float = Field(ge=0)
    polarization: JonesVector
    basis: PolarizationBasis


class GaussianBeamState(BasisCarrier, OmitAbsentFields, StrictModel):
    """Gaussian beam at a plane, per transverse axis of the transported beam frame ``(e1, e2)``.

    ``q = z + i z_R`` (mm, in the local medium); ``z`` = signed distance from the waist to this plane
    (positive after the waist). ``valid=False`` means the model left its applicability domain (``reason`` gives the code);
    its numeric fields are then ``None``, and ``validity`` states the accuracy check that failed where a limit of the
    first-order model was exceeded.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("value_basis", "validity")

    valid: bool
    status: Status
    reason: str | None = None
    e1: Vector3 | None = None
    e2: Vector3 | None = None
    wavelength_nm: float | None = None
    medium_index: float | None = None
    q_x_mm: Complex | None = None
    q_y_mm: Complex | None = None
    m2_x: float | None = None
    m2_y: float | None = None
    waist_radius_x_mm: float | None = None
    waist_radius_y_mm: float | None = None
    waist_distance_x_mm: float | None = Field(None, description="Signed distance waist -> plane along the chief ray (x axis).")
    waist_distance_y_mm: float | None = None
    rayleigh_range_x_mm: float | None = None
    rayleigh_range_y_mm: float | None = None
    beam_radius_x_mm: float | None = Field(None, description="1/e^2 intensity radius at the plane along e1.")
    beam_radius_y_mm: float | None = None
    far_field_half_angle_x_rad: float | None = None
    far_field_half_angle_y_rad: float | None = None
    local_divergence_x_rad: float | None = Field(None, description="dw/dz at this plane (x).")
    local_divergence_y_rad: float | None = None
    curvature_radius_x_mm: float | None = Field(None, description="Wavefront radius of curvature (null when the wavefront is flat).")
    curvature_radius_y_mm: float | None = None
    value_basis: ValueBasisField = None
    validity: tuple[ValidityCheck, ...] = Field(
        (), description="With valid=false: the accuracy limit of the first-order model the beam exceeded (the quantity, the "
        "value observed, the limit and the setting that holds it, where it was first exceeded, the code). The state stays not "
        "valid for the rest of the path. Omitted when empty.")


class BeamState(StrictModel):
    """State of one beam (chief ray + polarization + Gaussian) at a point of a path."""

    source_id: str
    path_id: str
    spectral_index: int
    wavelength_nm: float
    position_mm: Vector3
    direction: Vector3
    power_W: float | None = Field(
        description="The beam's power (W). null: not known, where it is an order of a diffraction grating whose response does "
        "not give its power (ArrivalResult.power_upper_bound_W bounds it); null is not 0 W.")
    geometric_length_mm: float
    optical_path_length_mm: float | None = Field(description="Sum of n x length from the source. None where ArrivalResult.ray_opl_defined is false: "
                                                 "an element on the way states no optical path.")
    phase_rad: float | None = Field(None, description="Only when a phase reference (single coherent source) exists; None where ArrivalResult.ray_opl_defined is false.")
    polarization: JonesVector | None = Field(
        description="The Jones vector in the basis. null: the polarization is not known (an order of a diffraction grating "
        "whose response leaves it open).")
    basis: PolarizationBasis
    gaussian: GaussianBeamState | None = None


class RaySegment(OmitAbsentFields, StrictModel):
    absent_omitted: ClassVar[tuple[str, ...]] = ("crystal_mode",)

    start_mm: Vector3
    end_mm: Vector3 | None = Field(None, description="None for a segment that leaves the scene (escapes to infinity).")
    length_mm: float | None = None
    medium_id: str
    refractive_index: float
    optical_path_length_mm: float | None = None
    power_start_W: float | None = Field(description="null: the power of this stretch is not known (a grating order without a stated response).")
    power_end_W: float | None = None
    end_surface_id: str | None = None
    end_event: str | None = None
    angle_of_incidence_deg: float | None = None
    beam_radius_start_mm: float | None = Field(None, description="Max(x, y) 1/e^2 beam radius at the segment start (None if the Gaussian model is not valid).")
    beam_radius_end_mm: float | None = None
    crystal_mode: Literal["ORDINARY", "EXTRAORDINARY"] | None = Field(
        None, description="In a uniaxial crystal: the wave this stretch travels as (its ray runs along the segment; an "
        "extraordinary wave's ray walks off its wave normal) and refractive_index is its phase index along the ray, n(theta) "
        "cos(rho), so optical_path_length_mm is the optical path along it. Absent outside crystals.")


class OpticalPath(OmitAbsentFields, StrictModel):
    """One chief-ray path from a source (or from a branch point) to its termination.

    Branching: ``parent_path_id`` and ``branch_surface_id`` locate the branch point; cumulative lengths and
    OPL include all ancestor segments (``*_from_source``). Recombination is represented by observations that
    receive arrivals from several paths (see InterferenceResult), not by merging paths.
    """

    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    parent_path_id: str | None = None
    branch_surface_id: str | None = None
    branch_kind: str | None = None
    lineage: tuple[str, ...] = Field((), description="Ordered '<surface_id>:<event>' interaction history from the source.")
    segments: tuple[RaySegment, ...]
    termination: str
    termination_surface_id: str | None = None
    absent_omitted: ClassVar[tuple[str, ...]] = ("power_out_of_scope_at",)

    final_power_W: float | None = Field(description="null: the path's power is not known (a grating order without a stated response).")
    power_out_of_scope_at: tuple[str, ...] = Field(
        (), description="Surfaces behind which the path's power is out of model scope (as ArrivalResult.power_out_of_scope_at): "
        "final_power_W and the segments' powers from there on are its chief ray's, not the light's. Omitted when there is none.")
    geometric_path_length_mm: float = Field(description="Cumulative from the source, this path's segments plus ancestors.")
    optical_path_length_mm: float | None = Field(description="Cumulative from the source, this path's segments plus ancestors. None behind an ideal or "
                                                 "tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical path "
                                                 "(a segment's own optical_path_length_mm is still n x its length).")
