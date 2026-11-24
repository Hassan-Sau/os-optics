"""Imaging analysis of a run's image plane: point spread function, MTF and encircled energy per field (``imaging``,
``ImagingRequestV1``), as an analysis document (``AnalysisResultV1``, analysis ``IMAGING``) that changes no run.

Rays. Each field (a source's spectral line) is traced again (``results.imaging_trace``) with a square grid of rays over a disc
that overfills the system's aperture stop: the apertures cut it to the vignetted pupil, launched uniformly (a point object, at
infinity for a collimated source). The disc is the request's (``launch_radius_beam_radii``), or it is found: grown from
``LAUNCH_START_RADII`` beam radii by ``LAUNCH_GROWTH`` until an aperture cuts it, up to the request's
``max_launch_radius_beam_radii``, then fitted to the light that passes (``rays.pupil`` ``APERTURE``). Where no aperture cuts it
within that reach the system's pupil is not bounded, and the field is analysed on the disc of ``LAUNCH_START_RADII`` beam
radii, exactly as a request that states that radius (``rays.pupil`` ``LAUNCH_DISC``, ``rays.pupil_search_radii``): every
number of the field is then that of a uniformly lit disc of that size, not of the system's pupil and not of the beam's own
Gaussian profile, and says so -- state ``UNKNOWN`` with reason ``PUPIL_NOT_BOUNDED`` beside the value, and a ``value_basis``
of kind ``ESTIMATE`` with the assumption ``UNIFORM_PUPIL`` (the radius used). Each ray then carries the irradiance its field keeps to the plane (a sample's amplitude,
the interfaces' and coatings' transmissions). The two mutually incoherent components of a partially polarized or unpolarized
source each give their own field on the same rays and optical paths (the trace carries no coating or interface phase), and
their point spread functions add in irradiance. The rays that reach the image plane define, per field:

* geometric quantities -- the image-space numerical aperture, the spot (centroid and RMS radius on the plane normal to the
  image-space axis), the geometric MTF ``|sum_j p_j exp(-i 2 pi f . (r_j - r_c))|`` (``p_j`` each ray's share of the irradiance) (basis: geometric optics, valid where the
  wavefront error is large, ``GEOMETRIC_MIN_RMS_WAVES``). The curve is stated whenever the rays form one spot. Where the
  wavefront error is below that (``GEOMETRIC_REGIME_NOT_MET``, ``OUT_OF_MODEL_SCOPE``: diffraction sets the MTF, and the
  geometric curve lies above it) or is not defined on the path (``GEOMETRIC_REGIME_UNKNOWN``, ``UNKNOWN``) the quantity keeps
  that state and its curve carries a ``value_basis``: an ``ESTIMATE`` with the regime check (``opd_rms_waves``, its limit, the
  observed value, ``within`` false or null). Such a curve decides no comparison (``Quantity.decides``);
* the wavefront -- each ray's optical path from the source's phase front less ``n`` times the projection of its offset from the
  reference point (the rays' centroid on the plane, or the source axis's ray) on its direction: the wavefront on the reference
  sphere at infinity about that point (direction space), the phase the Debye integral sums; piston removed, RMS and
  peak-to-valley in waves. The same on the finite sphere through the last surface before the plane is evidence beside it
  (``rays.opd_rms_waves_on_last_surface_sphere``): it agrees where that radius is large against the spot and the focus offset;
* diffraction quantities -- the point spread function on the plane normal to the image-space axis through the reference point,
  by the Debye integral over the traced rays (``diffraction.debye.scattered_on_grid``): each ray stands for the solid angle its
  tube fills (from its grid neighbours) and carries the phase of its optical path difference. Up to an image-space sine of
  ``SCALAR_SINE_LIMIT`` the field is scalar; above it the vector form of the same kernel is used, for unpolarized light (two
  orthogonal pupil polarizations added in irradiance, each carried as by an aplanatic lens). From it: the Strehl ratio (the peak
  against that of the same rays without their phase), the MTF (the Fourier transform of the point spread function, tangential
  along the projection of the plane's normal and sagittal across it), the diffraction-limited MTF of the same pupil, and the
  radii holding 50 % and 80 % of the light.

Validity and evidence. The diffraction quantities need the rays' optical paths (not after an ideal or tabulated thin lens, a
Faraday rotator or a grating: ``RAY_OPL_UNDEFINED``), neighbouring rays within half a wave of each other and a map window inside
the period the ray spacing allows (``PUPIL_UNDERSAMPLED``), and a window that holds the light (``PSF_WINDOW_TOO_SMALL``);
outside them they are ``OUT_OF_MODEL_SCOPE`` and the geometric quantities stay. Every quantity is computed again with twice the
rays per axis (``convergence``). The grid is deterministic: no random sampling, no seed.

Model scope. A field reads the light of its source's line at the plane. Where the run found that light outside an element's
published model (a model-scope finding whose arrivals on this plane include the line: ``local.validity``), every number of
the field that reads what the finding touches is the model's number outside its validity: state ``OUT_OF_MODEL_SCOPE``
(reason the finding's code), the number kept, and a ``value_basis`` of kind ``ESTIMATE`` with the finding's validity check
(the parameter, the published limits, the value the light met, the setting of a single point's tolerance, the element,
the code) beside what qualified it already. A finding about a coating's coefficients leaves the image-space cone, the
spot and the wavefront as they are (``GEOMETRY_QUANTITIES``). Another line on the same plane keeps its quantities.

Selection. With ``convergence`` false the analysis runs once, on ``launch_grid`` rays: the doubled grid is not traced and each
value's convergence is ``NONE`` (not checked). With ``diffraction`` false the point spread function is not computed: the
diffraction quantities are ``NOT_COMPUTED`` (reason ``NOT_REQUESTED``) and the geometric and wavefront quantities are those of
the full analysis, with their states and reasons where a field lacks its wavefront or its geometric quantities (a thin
ideal lens, a source's two polarization components that part). A request that keeps both at their defaults is the full
analysis, and its settings do not name them.

Execution. The analysis has no resource budget of its own: its execution is ``COMPLETED``, and what is known of each quantity
is its own state and reason (a field that is not a beam source: ``NOT_APPLICABLE``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from ..analysis import completed, document
from ..contracts.analysis import AnalysisResultV1, Convergence, ConvergenceMethod, ModelBasis, Quantity, QuantityScope, ValidityCheck
from ..contracts.basis import Assumed, ValueBasis, merge_bases
from ..contracts.imaging import LAUNCH_SEARCH_DEFAULT_RADII, MAX_LAUNCH_GRID, MAX_PSF_SAMPLES, ImagingRequestV1
from ..diffraction.debye import scattered_at_points, scattered_on_grid
from ..diffraction.metrics import maximize, radius_for_fraction
from ..diffraction.objective import sphere_field
from ..diffraction.quadrature import gauss_legendre
from ..errors.status import AssumedInput, QuantityState, ValueKind
from ..math.vectors import dot
from ..optics.rays.path_ids import root_path_id
from ..units import NM_TO_MM
from .imaging_trace import FieldRays, launch_grid, trace_fields
from .model import ResultsError
from .pupil_field import SCALAR_SINE_LIMIT

IMAGING_ANALYSIS = "IMAGING"
IMAGING_ANALYSIS_VERSION = "1.2"
IMAGING_INVALID = "IMAGING_INVALID"
GEOMETRIC_MODEL_ID = "geometric_ray_bundle"
MM_TO_UM = 1000.0
SIGNIFICANT_DIGITS = 9
ROUNDING_SHARE = 10.0 ** (1 - SIGNIFICANT_DIGITS)
MAP_ROUNDING = 1e-9
#: The launch disc starts at this many beam radii and grows by ``LAUNCH_GROWTH`` until an aperture cuts it, up to the request's
#: ``max_launch_radius_beam_radii``; where none does, the disc of ``LAUNCH_START_RADII`` is the pupil the field is analysed on.
LAUNCH_START_RADII = 2.0
LAUNCH_GROWTH = 1.5
#: How the pupil of a field is defined (``arrays["<line>rays"]["pupil"]``): the system's apertures cut the launch disc, or the
#: launch disc itself is the pupil (the request's, or the stand-in of an unbounded system).
PUPIL_APERTURE = "APERTURE"
PUPIL_LAUNCH_DISC = "LAUNCH_DISC"
PUPIL_NOT_BOUNDED = "PUPIL_NOT_BOUNDED"
#: The fitted disc reaches this many grid steps beyond the outermost ray that passes.
LAUNCH_FIT_STEPS = 2.0
#: The geometric MTF stands for the MTF where the wavefront's RMS error is at least this (waves).
GEOMETRIC_MIN_RMS_WAVES = 1.0
#: Neighbouring rays may differ by at most half a wave of optical path.
MAX_NEIGHBOUR_PHASE_RAD = math.pi
#: The map window may reach at most this share of the period the ray spacing gives the summed field: the replica of a spot
#: at most half the window wide then stays a window's width away from it.
ALIAS_SHARE = 0.5
#: The window holds at least this share of the light for the MTF and encircled energy to be stated.
MIN_WINDOW_SHARE = 0.95
#: Default half width of the map: this many Airy radii, or ``WINDOW_SPOT_FACTOR`` times the geometric spot's largest radius.
WINDOW_AIRY_RADII = 16.0
WINDOW_SPOT_FACTOR = 2.0
AIRY_FACTOR = 0.61
#: Default MTF frequencies: this many steps from 0 to the cut-off.
MTF_STEPS = 32
#: Convergence tolerances between the two ray grids (the vignetted rim converges as 1 / launch_grid): relative for spot, wavefront,
#: Strehl and radii; absolute for MTF values.
RAY_TOLERANCE = 1e-2
MTF_TOLERANCE = 3e-2
#: Encircled energy: annuli out to the map's half width, Gauss-Legendre nodes per annulus and azimuths.
ENCIRCLED_ANNULI = 32
RADIAL_NODES = 4
AZIMUTH_NODES = 32
ENCIRCLED_FRACTIONS = (0.5, 0.8)
PERCENT = 100.0
#: Passes of the peak search on the map.
PEAK_PASSES = 2
#: A ray tube's direction differences below this count as a collimated bundle (no image).
MIN_IMAGE_SINE = 1e-6
#: Directions closer than this to the plane's normal leave the tangential direction to the plane's u axis.
AXIS_TOLERANCE = 1e-9
#: The diffraction quantities of a field (``diffraction``): computed from the point spread function, or not at all.
DIFFRACTION_QUANTITIES = ("strehl_ratio", "mtf_tangential", "mtf_sagittal", "diffraction_limited_mtf_tangential",
                          "diffraction_limited_mtf_sagittal", "encircled_radius_50", "encircled_radius_80",
                          "diffraction_limited_encircled_radius_80")
#: The reason of a quantity the request leaves out.
NOT_REQUESTED = "NOT_REQUESTED"
#: The convergence of a value whose discretization was not checked (``convergence`` false).
NOT_CHECKED = Convergence(method=ConvergenceMethod.NONE)
#: The request's selections at their defaults (the full analysis): the settings name a selection only when it narrows the
#: analysis, so the full analysis' settings and their hash do not depend on the selections existing.
SELECTION_DEFAULTS = {"convergence": True, "diffraction": True, "max_launch_radius_beam_radii": LAUNCH_SEARCH_DEFAULT_RADII}

#: Why a source's two polarization components (a partially polarized or unpolarized source) cannot be read together on the plane.
COMPONENT_MESSAGES = {
    "POLARIZATION_COMPONENT_ROUTES_DIFFER": "the source's two polarization components cross the plane at different points (a "
                                            "birefringent or polarization-dependent split on their way): they do not form one image",
    "POLARIZATION_COMPONENT_NOT_RETRACED": "the run delivers a polarization component to the plane that the imaging rays do not "
                                           "reach: its share of the image is not known",
}
ASSUMPTIONS_DIFFRACTION = (
    "a point object lights the pupil uniformly; each ray carries the irradiance its field keeps to the plane (a sample's "
    "amplitude, the interfaces' and coatings' transmissions); a partially polarized or unpolarized source's two components are "
    "mutually incoherent: each component's pupil field is propagated to the image apart and their point spread functions add in "
    "irradiance, each by its share of the light that reaches the plane",
    "each ray stands for the solid angle its tube fills in image space and carries the phase of its optical path difference",
    "Debye integral over the rays' directions about the reference point (Fresnel number of the converging cone much above 1)",
    "monochromatic: one spectral line per field",
)


class ImagingError(ResultsError):
    """A request that is not valid: no analysis is made."""


def parse_request(request: Any) -> ImagingRequestV1:
    if isinstance(request, ImagingRequestV1):
        return request
    try:
        return ImagingRequestV1.model_validate({} if request is None else request)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise ImagingError(IMAGING_INVALID, problems) from None


def settings_request(parsed: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """The request as an analysis' settings record it: without the selections that keep their defaults."""

    dump = parsed.model_dump(mode="json")
    return {k: v for k, v in dump.items() if k not in defaults or v != defaults[k]}


def _sig(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _list(values: np.ndarray, largest: float | None = None) -> list[Any]:
    largest = float(np.abs(values).max()) if largest is None else largest
    if largest <= 0.0:
        return values.tolist()
    step = largest * MAP_ROUNDING
    return (np.round(values / step) * step + 0.0).tolist()


def _convergence(coarse: float | None, fine: float | None, parameter: str, tolerance: float, relative: bool = True) -> Convergence:
    if coarse is None or fine is None:
        return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, converged=False)
    difference = abs(fine - coarse)
    size = max(abs(fine), abs(coarse)) if relative else 1.0
    return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, coarse=_sig(coarse), fine=_sig(fine), difference=_sig(difference),
                       resolution=_sig(max(difference, abs(fine) * ROUNDING_SHARE)), converged=difference <= tolerance * size)


# ---------------------------------------------------------------------------------------------- one field at one ray grid


class _Level:
    """Everything one ray grid gives for one field (module docstring)."""

    def __init__(self, rays: FieldRays, request: ImagingRequestV1, plane: Any) -> None:
        self.rays = rays
        self.problems: dict[str, str] = {}
        wavelength_mm = rays.wavelength_nm * NM_TO_MM
        self.wavelength_mm = wavelength_mm
        self.k0 = 2.0 * math.pi / wavelength_mm
        self.n = rays.medium_index
        arrived = rays.arrived
        self.count = int(arrived.sum())
        points = rays.points[arrived]
        directions = rays.directions[arrived]
        directions = directions / np.sqrt(dot(directions, directions))[:, None]
        self.reference = points.mean(axis=0) if request.reference == "CENTROID" else rays.chief_point
        axis = directions.mean(axis=0)
        axis = axis / math.sqrt(float(dot(axis, axis)))
        normal = np.array(plane.normal, dtype=np.float64)
        tangential = normal - float(dot(normal, axis)) * axis
        if math.sqrt(float(dot(tangential, tangential))) < AXIS_TOLERANCE:
            tangential = np.array(plane.u_axis, dtype=np.float64) - float(dot(np.array(plane.u_axis, dtype=np.float64), axis)) * axis
        tangential = tangential / math.sqrt(float(dot(tangential, tangential)))
        sagittal = np.cross(axis, tangential)
        self.axis, self.tangential, self.sagittal = axis, tangential, sagittal
        u, v, c = dot(directions, tangential), dot(directions, sagittal), dot(directions, axis)
        self.u, self.v, self.c = u, v, c
        # each ray's irradiance on the pupil relative to its launch (|J|^2 of its field over that of the source's Jones vector): a
        # sample's amplitude map and the coatings along the way weight the pupil; 1 for every ray of a path that changes no ray's
        # field differently
        self.intensity = np.ones(self.count) if rays.irradiance is None else rays.irradiance[arrived]
        # the mutually incoherent polarization components' pupils, each propagated to the image apart (their sum: intensity)
        self.parts = [part[arrived] for part in rays.component_irradiance] or [self.intensity]
        if rays.problem is not None:  # (the source's polarization components cannot be added on this route)
            self.problems["wavefront"] = self.problems["diffraction"] = rays.problem
        if not float(self.intensity.sum()) > 0.0:  # (no ray keeps any field: nothing weights the pupil)
            self.problems["diffraction"] = "NO_LIGHT_ON_PUPIL"
            self.intensity = np.ones(self.count)
            self.parts = [self.intensity]
        self.image_sine = float(np.sqrt(u * u + v * v).max())
        self.na = self.n * self.image_sine
        # the geometric spot on the plane normal to the axis through the reference point
        rel = points - self.reference
        tau = -dot(rel, axis) / c
        spot = rel + tau[:, None] * directions
        self.spot_a, self.spot_b = dot(spot, tangential), dot(spot, sagittal)
        ca, cb = float(self.spot_a.mean()), float(self.spot_b.mean())
        self.spot_rms = math.sqrt(float(((self.spot_a - ca) ** 2 + (self.spot_b - cb) ** 2).mean()))
        self.spot_reach = float(np.sqrt(self.spot_a * self.spot_a + self.spot_b * self.spot_b).max())
        self.spot_centroid = (ca, cb)
        if self.image_sine < MIN_IMAGE_SINE or rays.previous_point is None:
            self.problems["diffraction"] = "EXIT_PUPIL_UNDEFINED"
            self.radius = None
        else:
            self.radius = float(dot(self.reference - rays.previous_point, axis))
            if self.radius <= 0.0:
                self.problems["diffraction"] = "EXIT_PUPIL_UNDEFINED"
        self.opd = None
        # the wavefront: each ray's optical path less n times the projection of its offset from the reference point on its
        # direction -- the direction-space wavefront about the reference point (the reference sphere at infinity), the phase the
        # Debye integral sums, which no ray can miss. Beside it, as evidence, the wavefront on the finite sphere through the last
        # surface before the plane (None where a ray's line passes farther from the reference point than that radius): the two
        # agree where that radius is large against the spot and the focus offset, and part where it is not.
        self.opd_rms_last: float | None = None
        if not rays.opl_defined or rays.opl is None:
            self.problems.setdefault("wavefront", "RAY_OPL_UNDEFINED")
            self.problems.setdefault("diffraction", "RAY_OPL_UNDEFINED")
        elif self.radius is not None and "diffraction" not in self.problems:
            opl = rays.opl[arrived]
            path = opl - self.n * dot(directions, points - self.reference)
            self.opd = path - path.mean()
            last = self._opd(points, directions, opl, self.radius)
            self.opd_rms_last = None if last is None else math.sqrt(float((last * last).mean())) / wavelength_mm
        else:
            self.problems.setdefault("wavefront", "EXIT_PUPIL_UNDEFINED")
        self.opd_waves = None if self.opd is None else self.opd / wavelength_mm
        # each ray's solid angle from its grid neighbours
        self._tubes(rays, arrived)

    def _opd(self, points: np.ndarray, directions: np.ndarray, opl: np.ndarray, radius: float) -> np.ndarray | None:
        rel = points - self.reference
        b = dot(directions, rel)
        disc = b * b - (dot(rel, rel) - radius * radius)
        if bool((disc < 0.0).any()):
            return None
        back = -b - np.sqrt(disc)  # the root behind the plane, toward the exit pupil
        path = opl + self.n * back
        return path - path.mean()

    def _tubes(self, rays: FieldRays, arrived: np.ndarray) -> None:
        grid = rays.grid
        size = grid.count
        u_map = np.full((size, size), np.nan)
        v_map = np.full((size, size), np.nan)
        columns, rows = grid.index[arrived, 0], grid.index[arrived, 1]
        u_map[rows, columns], v_map[rows, columns] = self.u, self.v
        derivatives = []
        for values in (u_map, v_map):
            pair = []
            for axis in (1, 0):
                forward = np.roll(values, -1, axis=axis)
                backward = np.roll(values, 1, axis=axis)
                edge = np.zeros((size, size), dtype=bool)
                if axis == 1:
                    edge[:, -1] = True
                    forward[edge] = np.nan
                    edge[:] = False
                    edge[:, 0] = True
                    backward[edge] = np.nan
                else:
                    edge[-1, :] = True
                    forward[edge] = np.nan
                    edge[:] = False
                    edge[0, :] = True
                    backward[edge] = np.nan
                central = 0.5 * (forward - backward)
                one_sided = np.where(np.isnan(forward), values - backward, forward - values)
                pair.append(np.where(np.isnan(central), one_sided, central))
            derivatives.append(pair)
        jacobian = np.abs(derivatives[0][0] * derivatives[1][1] - derivatives[0][1] * derivatives[1][0])
        tube = jacobian[rows, columns]
        usable = np.isfinite(tube) & (tube > 0.0)
        self.isolated = int((~usable).sum())
        self.usable = usable
        self.solid_angle = np.where(usable, tube, 0.0) / self.c
        # the largest spacing of neighbouring rays in direction sines, and the largest phase step between them
        spacing = 0.0
        step = 0.0
        opd_map = np.full((size, size), np.nan)
        if self.opd is not None:
            opd_map[rows, columns] = self.opd
        for axis in (0, 1):
            for values in (u_map, v_map):
                d = np.abs(np.diff(values, axis=axis))
                if np.isfinite(d).any():
                    spacing = max(spacing, float(np.nanmax(d)))
            if self.opd is not None:
                d = np.abs(np.diff(opd_map, axis=axis))
                if np.isfinite(d).any():
                    step = max(step, float(np.nanmax(d)) * self.k0)
        self.direction_spacing = spacing
        self.neighbour_phase = step

    # ------------------------------------------------------------------------------------------------ the point spread function

    def amplitudes(self, with_phase: bool) -> tuple[np.ndarray, list[np.ndarray]]:
        """``(directions (2, J), [field amplitudes (C, J) per polarization])`` of the usable rays."""

        keep = self.usable
        total = float(self.intensity[keep].sum())
        phase = np.exp(1j * self.k0 * self.opd[keep]) if with_phase and self.opd is not None else 1.0
        directions = np.stack((self.u[keep], self.v[keep]))
        # one field per mutually incoherent polarization component (each with its own pupil irradiance; their shares of
        # the light sum to 1), whose irradiances add on the plane
        bases = [(np.sqrt(part[keep] / total * self.solid_angle[keep]) / self.wavelength_mm * self.n) * phase + 0j for part in self.parts]
        if not self.vector:
            return directions, [base[None, :] for base in bases]
        sine = np.sqrt(self.u[keep] ** 2 + self.v[keep] ** 2)
        theta = np.arcsin(np.minimum(sine, 1.0))
        phi = np.arctan2(-self.v[keep], -self.u[keep])
        ones = np.ones(theta.shape, dtype=np.complex128)
        modes = []
        for jx, jy in ((1.0, 0.0), (0.0, 1.0)):
            field, _ = sphere_field(theta, phi, jx * ones, jy * ones, ones, ones, apodize=False)
            modes.extend(field * base[None, :] * math.sqrt(0.5) for base in bases)
        return directions, modes

    @property
    def vector(self) -> bool:
        return self.na > SCALAR_SINE_LIMIT

    @property
    def total(self) -> float:
        """The integral of the summed ``|E|^2`` over the plane (Parseval over the rays): the map's normalization."""

        keep = self.usable
        return float((self.intensity[keep] / self.c[keep]).sum()) / float(self.intensity[keep].sum())

    def psf_map(self, a: np.ndarray, with_phase: bool = True) -> np.ndarray:
        directions, modes = self.amplitudes(with_phase)
        k = self.k0 * self.n
        out = np.zeros((a.size, a.size))
        for amplitudes in modes:
            field = scattered_on_grid(k, directions, amplitudes, a, a)
            out = out + (np.abs(field) ** 2).sum(axis=0)
        return out / self.total

    def psf_at(self, a: np.ndarray, b: np.ndarray, with_phase: bool = True) -> np.ndarray:
        directions, modes = self.amplitudes(with_phase)
        k = self.k0 * self.n
        out = np.zeros(a.size)
        for amplitudes in modes:
            out = out + (np.abs(scattered_at_points(k, directions, amplitudes, a, b)) ** 2).sum(axis=0)
        return out / self.total


def _field_result(level: _Level, request: ImagingRequestV1, half: float, samples: int, frequencies: np.ndarray) -> dict[str, Any]:
    """The diffraction and geometric quantities of one field at one ray grid, on a map of half width ``half`` (mm)."""

    out: dict[str, Any] = {"spot_rms": level.spot_rms, "na": level.na, "count": level.count}
    if level.opd_waves is not None:
        out["opd_rms"] = math.sqrt(float((level.opd_waves * level.opd_waves).mean()))
        out["opd_pv"] = float(level.opd_waves.max() - level.opd_waves.min())
        out["opd_rms_last"] = level.opd_rms_last
    # geometric MTF (each ray weighted by its irradiance on the pupil)
    a0, b0 = level.spot_centroid
    share = level.intensity / float(level.intensity.sum())
    geo = []
    for along, centre in ((level.spot_a, a0), (level.spot_b, b0)):
        phase = -2.0 * math.pi * np.outer(frequencies, along - centre)
        geo.append(np.abs(np.exp(1j * phase) @ share))
    out["geometric_mtf"] = geo
    if "diffraction" in level.problems or not request.diffraction:
        return out
    pitch = 2.0 * half / samples
    a = (np.arange(samples) - samples // 2) * pitch
    psf = level.psf_map(a)
    window = float(psf.sum()) * pitch * pitch
    out["window_share"] = window
    peak_index = np.unravel_index(int(np.argmax(psf)), psf.shape)
    pa, pb = float(a[peak_index[1]]), float(a[peak_index[0]])
    peak = float(psf.max())
    for _ in range(PEAK_PASSES):
        pa, peak = maximize(lambda x, pb=pb: level.psf_at(x, np.full(x.shape, pb)), pa, pitch)
        pb, peak = maximize(lambda y, pa=pa: level.psf_at(np.full(y.shape, pa), y), pb, pitch)
    reference_peak = float(level.psf_at(np.zeros(1), np.zeros(1), with_phase=False)[0])
    out["strehl"] = peak / reference_peak
    out["peak_position"] = (pa, pb)
    free = level.psf_map(a, with_phase=False)
    out["psf"] = psf
    out["axis"] = a
    # the MTF: the Fourier transform of the map's line-spread functions over all the light (the map is normalized to unit
    # integral over the whole plane), so the light beyond the map is missing only from frequencies below 1 / (2 half)
    for name, image in (("mtf", psf), ("reference_mtf", free)):
        curves = []
        for line in (image.sum(axis=0), image.sum(axis=1)):
            phase = -2.0 * math.pi * np.outer(frequencies, a)
            curves.append(np.abs(np.exp(1j * phase) @ line) * pitch * pitch)
        out[name] = curves
    # encircled energy around the peak (as a share of all the light), from the maps by cubic-spline interpolation
    for name, image, centre in (("encircled", psf, (pa, pb)), ("reference_encircled", free, (0.0, 0.0))):
        out[name] = _encircled(image, a, centre, half)
    return out


def _encircled(image: np.ndarray, a: np.ndarray, centre: tuple[float, float], reach: float) -> dict[str, Any]:
    """The share of the light within radius ``r`` of ``centre`` on a normalized map (Gauss-Legendre annuli over its cubic
    spline), out to ``reach``."""

    from scipy.ndimage import map_coordinates

    pitch = float(a[1] - a[0])
    edges = np.linspace(0.0, reach, ENCIRCLED_ANNULI + 1)
    phi = 2.0 * math.pi * (np.arange(AZIMUTH_NODES) + 0.5) / AZIMUTH_NODES

    def ring(low: float, high: float) -> float:
        radii, weights = gauss_legendre(RADIAL_NODES, low, high)
        rr, pp = np.meshgrid(radii, phi, indexing="ij")
        columns = (centre[0] + rr * np.cos(pp) - float(a[0])) / pitch
        rows = (centre[1] + rr * np.sin(pp) - float(a[0])) / pitch
        values = map_coordinates(image, np.stack((rows.ravel(), columns.ravel())), order=len("xyz"), mode="constant", cval=0.0).reshape(rr.shape)
        return float((values.sum(axis=1) * radii * weights).sum()) * 2.0 * math.pi / AZIMUTH_NODES

    cumulative = [0.0]
    for low, high in zip(edges[:-1], edges[1:]):
        cumulative.append(cumulative[-1] + ring(float(low), float(high)))

    def within(radius: float) -> float:
        index = min(int(radius / (reach / ENCIRCLED_ANNULI)), ENCIRCLED_ANNULI - 1)
        return cumulative[index] + ring(float(edges[index]), radius)

    radii = {}
    for fraction in ENCIRCLED_FRACTIONS:
        found = None
        for index in range(1, len(cumulative)):
            if cumulative[index] >= fraction:
                found = radius_for_fraction(within, fraction, float(edges[index - 1]), float(edges[index]))
                break
        radii[fraction] = found
    return {"edges": edges, "cumulative": cumulative, "radii": radii}


# ---------------------------------------------------------------------------------------------- the analysis


def _fields(results: Any, monitor: str, request: ImagingRequestV1) -> tuple[list[tuple[str, float, int]], dict[tuple[str, float], str], list[str]]:
    """The (source, wavelength, spectral index) lines to analyse, the run's strongest path of each at the plane, and the sources
    asked for that are not beam sources."""

    observation = results._observation(monitor)
    preferred: dict[tuple[str, float], str] = {}
    strongest: dict[tuple[str, float], float] = {}
    for arrival in observation.arrivals:
        beam = arrival.beam
        key = (beam.source_id, beam.wavelength_nm)
        power = beam.power_W or 0.0
        if key not in strongest or power > strongest[key]:
            strongest[key], preferred[key] = power, beam.path_id
    beam_sources: dict[str, dict[float, int]] = {}
    for path in results.result.paths:  # the run's beam paths: its Gaussian sources' spectral lines (an extended emitter has none)
        beam_sources.setdefault(path.source_id, {}).setdefault(path.wavelength_nm, path.spectral_index)
    wanted = list(request.fields) if request.fields is not None else sorted({source for source, _ in preferred})
    lines = []
    refused = []
    for source in wanted:
        if source not in beam_sources:
            refused.append(source)
            continue
        for wavelength, index in beam_sources[source].items():
            if request.wavelengths_nm is None or any(abs(wavelength - w) <= WAVELENGTH_MATCH_NM for w in request.wavelengths_nm):
                lines.append((source, wavelength, index))
    return lines, preferred, refused


#: A requested wavelength matches a spectral line within this, in nm.
WAVELENGTH_MATCH_NM = 1e-6
#: The quantities of a field fixed by the traced geometry alone (the image-space cone; the spot, an unweighted mean over the
#: rays that arrive; the wavefront, a sum of indices times lengths): a finding about a coefficient -- a coating's power or
#: polarization data (``local.validity.MODEL_SCOPE_AFFECTS``) -- leaves them as the trace computed them. Every other
#: quantity of a field reads every kind (the MTF and the point spread function weigh each ray by the irradiance the
#: coatings keep).
GEOMETRY_QUANTITIES = frozenset({"image_space_na", "geometric_spot_rms_radius", "opd_rms_waves", "opd_pv_waves"})


Arrival = tuple[str, str, int]


def _arrivals(rows: Any) -> set[Arrival]:
    """``(observation_id, path_id, crossing_index)`` of listed arrivals."""

    return {(a["observation_id"], a["path_id"], a["crossing_index"]) for a in rows or ()
            if isinstance(a, dict) and {"observation_id", "path_id", "crossing_index"} <= a.keys()}


def _scope_findings(results: Any, monitor: str) -> list[tuple[dict[str, Any], set[Arrival]]]:
    """The run's model-scope findings whose light reaches the plane ``monitor``, each with the arrivals it reaches there
    (its ``affected_arrivals``: path and crossing); empty where the model scope was not computed (a bare Engine result)."""

    found = []
    for finding in results._scope_findings() or ():
        arrivals = {a for a in _arrivals(finding.get("affected_arrivals")) if a[0] == monitor}
        if arrivals:
            found.append((finding, arrivals))
    return found


def _field_light(results: Any, monitor: str, source: str, index: int, preferred: str | None, crossing: int | None) -> set[Arrival]:
    """The arrivals at ``monitor`` the field of the line (``source``, spectral ``index``) analyses: the route the trace
    follows (``imaging_trace.trace_fields``: the run's strongest path of the line at the plane, ``preferred``, in both of the
    line's polarization components) at the crossing asked for, else the last one the run records on that route."""

    if preferred is None:
        return set()
    second = root_path_id(source, index, 1)
    branch = preferred[len(second):] if preferred.startswith(second) else preferred[len(root_path_id(source, index)):]
    ids = {root_path_id(source, index, component) + branch for component in (0, 1)}
    crossings = [a.crossing_index for a in results._observation(monitor).arrivals if a.beam.path_id in ids]
    if not crossings:
        return set()
    at = crossing if crossing is not None else max(crossings)
    return {(monitor, path_id, at) for path_id in ids}


def _field_marks(findings: list[tuple[dict[str, Any], set[Arrival]]], light: set[Arrival]
                 ) -> list[tuple[str, frozenset[Any], tuple[ValidityCheck, ...]]]:
    """The findings that reach the light a field analyses (``_field_light``): ``(code, the kinds of quantity it touches, the
    validity checks it states of that light)`` -- per check, the light met at its value (``validity_reach``); a finding
    that does not say per check, or whose checks name none of this light, gives all of its checks."""

    from ..local.validity import finding_quantity_kinds

    marks = []
    for finding, arrivals in findings:
        if arrivals.isdisjoint(light):
            continue
        code = finding["code"]
        checks = [ValidityCheck.model_validate(c) for c in finding.get("validity") or ()]
        reach = finding.get("validity_reach") or ()
        if len(reach) == len(checks):
            checks = [c for c, met in zip(checks, reach, strict=True)
                      if not light.isdisjoint(_arrivals(met.get("affected_arrivals")))] or checks
        if not checks:  # (a finding a run of an earlier Engine listed without its checks)
            checks = [ValidityCheck(parameter=code.lower(), within=False, at=(finding["instance_id"],), code=code)]
        marks.append((code, finding_quantity_kinds(code), tuple(checks)))
    return marks


def outside_model(quantity: Quantity, marks: list[tuple[str, frozenset[Any], tuple[ValidityCheck, ...]]],
                  geometry: frozenset[str] = GEOMETRY_QUANTITIES) -> Quantity:
    """``quantity`` with the findings of ``marks`` that touch what it reads (``geometry``: the quantity ids that read the light's
    geometry only; every other one reads every kind): a number of the field is the model's number outside its validity --
    state ``OUT_OF_MODEL_SCOPE`` (reason the finding's code, or its own where it was out of scope already), the number kept,
    its ``value_basis`` an ``ESTIMATE`` with the findings' checks beside what qualified it. A quantity without a number stays
    as it is."""

    from ..validation.quantity_kinds import ALL_QUANTITY_KINDS, QuantityKind

    if quantity.value is None and quantity.values_ref is None:
        return quantity
    reads = frozenset({QuantityKind.GEOMETRY}) if quantity.quantity_id.rsplit(".", 1)[-1] in geometry else ALL_QUANTITY_KINDS
    applying = [(code, checks) for code, kinds, checks in marks if kinds & reads]
    if not applying:
        return quantity
    scope = ValueBasis(kind=ValueKind.ESTIMATE, validity=tuple(check for _, checks in applying for check in checks))
    out_of_scope = quantity.state is QuantityState.OUT_OF_MODEL_SCOPE
    return Quantity.model_validate({**quantity.model_dump(mode="python"), "state": QuantityState.OUT_OF_MODEL_SCOPE,
                                    "reason": quantity.reason if out_of_scope else applying[0][0],
                                    "value_basis": merge_bases(quantity.value_basis, scope, monotone=None)})


def _trace(results: Any, monitor: str, source: str, preferred: dict[tuple[str, float], str], count: int, radius: float,
           crossing: int | None) -> dict[float, FieldRays]:
    arriving = {arrival.beam.path_id for arrival in results._observation(monitor).arrivals}
    traced = trace_fields(results.request, monitor, {source}, preferred, launch_grid(count, radius), crossing, arriving)
    return {wavelength: rays for (src, wavelength), rays in traced.items() if src == source}


def imaging(results: Any, monitor: str, request: Any) -> AnalysisResultV1:
    """The imaging analysis of the run ``results`` at the observation plane ``monitor`` (module docstring)."""

    parsed = parse_request(request)
    if results.request is None:
        raise ResultsError("REQUEST_NOT_AVAILABLE", "this run carries no Engine request (engine_request.json): its light cannot be traced again")
    plane = results._observation(monitor).plane
    lines, preferred, refused = _fields(results, monitor, parsed)
    settings = {"request": settings_request(parsed, SELECTION_DEFAULTS), "monitor": monitor, "random_seed": None,
                "sampling": "a square grid of rays over the launch disc, deterministic"}
    models: list[ModelBasis] = [ModelBasis(model_id=GEOMETRIC_MODEL_ID, name="geometric ray bundle of the traced rays", version=IMAGING_ANALYSIS_VERSION,
                                           assumptions=("the traced rays of a uniformly lit pupil; the geometric MTF stands for the MTF where the wavefront error is large",))]
    quantities: list[Quantity] = []
    arrays: dict[str, Any] = {"conventions": {
        "plane": "the point spread function, the spot and the MTF lie on the plane normal to the image-space axis (the rays' mean direction) through the reference point",
        "axes": "tangential: the projection of the image plane's normal on that plane (the image plane's u axis on axis); sagittal: axis x tangential",
        "units": "um for positions, cycles/mm for frequencies, waves at the field's wavelength for the wavefront, 1/um^2 for the normalized point spread function",
    }}
    for source in refused:
        quantities.append(Quantity(quantity_id=f"{source}.field", name="imaging field", scope=QuantityScope(source_id=source, observation_id=monitor),
                                   state=QuantityState.NOT_APPLICABLE, reason="FIELD_NOT_A_BEAM_SOURCE", basis=GEOMETRIC_MODEL_ID,
                                   message=f"{source} is not a beam source of this run (an extended emitter, or no such source)"))
    findings = _scope_findings(results, monitor)
    for source, wavelength, index in lines:
        start = len(quantities)
        _analyse_field(results, monitor, parsed, plane, source, wavelength, index, preferred, models, quantities, arrays)
        marks = _field_marks(findings, _field_light(results, monitor, source, index, preferred.get((source, wavelength)),
                                                    parsed.crossing_index)) if findings else []
        if marks:
            quantities[start:] = [outside_model(q, marks) for q in quantities[start:]]
    identity = results.identity()
    return document(IMAGING_ANALYSIS, IMAGING_ANALYSIS_VERSION, settings, completed(), engine_input_hash=identity.get("engine_input_hash"),
                    run_result_hash=identity.get("result_hash"), models=models, quantities=quantities, arrays=arrays,
                    engine_build_id=results.engine_build_id)


def _analyse_field(results: Any, monitor: str, request: ImagingRequestV1, plane: Any, source: str, wavelength: float, index: int,
                   preferred: dict[tuple[str, float], str], models: list[ModelBasis], quantities: list[Quantity], arrays: dict[str, Any]) -> None:
    # a line's ids carry its spectral index, the run's identity of the line (as in its path ids, <source>#<index>); the wavelength
    # is in the scope at full precision: two lines closer than any display format stay distinct
    prefix = f"{source}#{index}."
    scope = QuantityScope(source_id=source, observation_id=monitor, wavelength_nm=wavelength)
    diffraction_id = f"{prefix}diffraction"

    def add(name: str, label: str, unit: str | None, value: float | None, *, basis: str, convergence: Convergence | None = None,
            reason: str | None = None, message: str = "", state: QuantityState | None = None, values_ref: str | None = None,
            value_basis: ValueBasis | None = None, depends_on: tuple[str, ...] = ()) -> None:
        """A quantity of this field: without a number, its state and reason; with one, ``KNOWN`` -- or, with ``value_basis``
        (what the number is where it is not simply the model's value), the stated state and reason beside the number."""

        if value is None and values_ref is None:
            quantities.append(Quantity(quantity_id=prefix + name, name=label, unit=unit, scope=scope, state=state or QuantityState.OUT_OF_MODEL_SCOPE,
                                       reason=reason or "NOT_DETERMINED", message=message, basis=basis))
            return
        # (the stand-in pupil of an unbounded system qualifies every number of the field, beside what else qualifies it)
        entry = value_basis if stand_in is None else merge_bases(stand_in, value_basis, monotone=None)
        if entry is not None:
            quantities.append(Quantity(quantity_id=prefix + name, name=label, unit=unit, scope=scope, state=state or QuantityState.UNKNOWN,
                                       value=None if value is None else _sig(value), values_ref=values_ref, basis=basis, convergence=convergence,
                                       reason=reason or PUPIL_NOT_BOUNDED, message=message, value_basis=entry, depends_on=depends_on))
            return
        quantities.append(Quantity(quantity_id=prefix + name, name=label, unit=unit, scope=scope, state=QuantityState.KNOWN,
                                   value=None if value is None else _sig(value), values_ref=values_ref, basis=basis, convergence=convergence))

    # the launch disc: the request's, or grown until an aperture cuts it and then fitted to the light that passes
    count = request.launch_grid
    stated = request.launch_radius_beam_radii is not None
    radius = request.launch_radius_beam_radii or LAUNCH_START_RADII
    stand_in: ValueBasis | None = None
    traced = _trace(results, monitor, source, preferred, count, radius, request.crossing_index).get(wavelength)
    while not stated and traced is not None and bool(traced.arrived.all()) and radius * LAUNCH_GROWTH <= request.max_launch_radius_beam_radii:
        radius *= LAUNCH_GROWTH
        traced = _trace(results, monitor, source, preferred, count, radius, request.crossing_index).get(wavelength)
    if traced is None or not bool(traced.arrived.any()):
        add("field", "imaging field", None, None, basis=GEOMETRIC_MODEL_ID, state=QuantityState.UNKNOWN, reason="NO_ARRIVAL",
            message="no traced ray of this field reaches the image plane: its path ends before it (a beam an aperture truncates is not followed), "
                    "or it reaches another plane")
        return
    searched = radius
    pupil = PUPIL_LAUNCH_DISC if bool(traced.arrived.all()) else PUPIL_APERTURE
    if not stated and pupil == PUPIL_LAUNCH_DISC:
        # no aperture cuts the disc within the search: the system's pupil is not bounded. The field is analysed on the disc the
        # search starts from, exactly as a request that states it, and every number says it stands on that stand-in pupil
        radius = LAUNCH_START_RADII
        stand_in = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.UNIFORM_PUPIL, value=LAUNCH_START_RADII, unit="beam_radii"),))
        traced = _trace(results, monitor, source, preferred, count, radius, request.crossing_index).get(wavelength)
    elif not stated:
        launched = traced.grid.offsets[traced.arrived]
        fitted = float(np.sqrt((launched * launched).sum(axis=1)).max()) + LAUNCH_FIT_STEPS * traced.grid.step
        if fitted < radius:
            radius = fitted
            traced = _trace(results, monitor, source, preferred, count, radius, request.crossing_index).get(wavelength)
    # (the doubled grid, traced only for the convergence evidence)
    check = request.convergence
    finer = _trace(results, monitor, source, preferred, min(2 * count, 2 * MAX_LAUNCH_GRID), radius, request.crossing_index).get(wavelength) \
        if check else traced
    if traced is None or finer is None or not bool(traced.arrived.any()) or not bool(finer.arrived.any()):
        add("field", "imaging field", None, None, basis=GEOMETRIC_MODEL_ID, state=QuantityState.UNKNOWN, reason="NO_ARRIVAL",
            message="the fitted launch disc leaves no ray on the image plane")
        return
    levels = [_Level(traced, request, plane), _Level(finer, request, plane)] if check else [_Level(traced, request, plane)]
    coarse_level, fine_level = levels[0], levels[-1]
    refine = f"launch_grid {count} doubled"

    def evidence(coarse_value: float | None, fine_value: float | None, parameter: str, tolerance: float,
                 relative: bool = True) -> Convergence:
        return _convergence(coarse_value, fine_value, parameter, tolerance, relative) if check else NOT_CHECKED

    def refined(difference: float, tolerance: float) -> Convergence:
        if not check:
            return NOT_CHECKED
        return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=refine, difference=_sig(difference), converged=difference <= tolerance)

    arrays[f"{prefix}rays"] = {"launch_radius_beam_radii": _sig(radius), "pupil": pupil, "pupil_search_radii": None if stated else _sig(searched),
                               "launch_grid": [count, 2 * count] if check else [count],
                               "rays_arrived": [lv.count for lv in levels],
                               "rays_without_neighbours": [lv.isolated for lv in levels], "path_id": fine_level.rays.path_id,
                               "crossing_index": fine_level.rays.crossing_index, "reference_point_mm": [_sig(float(c)) for c in fine_level.reference],
                               "image_axis": [_sig(float(c)) for c in fine_level.axis], "tangential_axis": [_sig(float(c)) for c in fine_level.tangential],
                               "reference_sphere_radius_mm": None if fine_level.radius is None else _sig(fine_level.radius)}
    # (the source's two polarization components cannot be read as one beam here, at either sampling)
    components = fine_level.rays.problem or coarse_level.rays.problem
    if components is None:
        add("image_space_na", "numerical aperture of the traced image-space cone", None, fine_level.na, basis=GEOMETRIC_MODEL_ID,
            convergence=evidence(coarse_level.na, fine_level.na, refine, RAY_TOLERANCE))
        add("geometric_spot_rms_radius", "RMS radius of the ray spot about its centroid", "um", fine_level.spot_rms * MM_TO_UM, basis=GEOMETRIC_MODEL_ID,
            convergence=evidence(coarse_level.spot_rms * MM_TO_UM, fine_level.spot_rms * MM_TO_UM, refine, RAY_TOLERANCE))
    else:
        for name, label, unit in (("image_space_na", "numerical aperture of the traced image-space cone", None),
                                  ("geometric_spot_rms_radius", "RMS radius of the ray spot about its centroid", "um")):
            add(name, label, unit, None, basis=GEOMETRIC_MODEL_ID, reason=components, message=COMPONENT_MESSAGES[components])

    # the map window and the frequencies
    lam = fine_level.wavelength_mm
    cutoff = 2.0 * fine_level.na / lam
    airy = AIRY_FACTOR * lam / max(fine_level.na, MIN_IMAGE_SINE)
    half = request.psf_half_width_um / MM_TO_UM if request.psf_half_width_um is not None else max(WINDOW_AIRY_RADII * airy, WINDOW_SPOT_FACTOR * fine_level.spot_reach)
    samples = request.psf_samples
    while samples / (4.0 * half) < cutoff and samples * 2 <= MAX_PSF_SAMPLES:
        samples *= 2
    nyquist = samples / (4.0 * half)
    frequencies = np.array(request.frequencies_cyc_per_mm, dtype=np.float64) if request.frequencies_cyc_per_mm is not None else \
        np.linspace(0.0, cutoff, MTF_STEPS + 1)
    resolved = frequencies <= nyquist
    results_by_level = [_field_result(lv, request, half, samples, frequencies) for lv in levels]
    coarse, fine = results_by_level[0], results_by_level[-1]

    # the wavefront
    if fine_level.opd_waves is not None and coarse_level.opd_waves is not None:
        add("opd_rms_waves", "RMS optical path difference on the reference sphere", "waves", fine["opd_rms"], basis=GEOMETRIC_MODEL_ID,
            convergence=evidence(coarse["opd_rms"], fine["opd_rms"], refine, RAY_TOLERANCE, relative=False))
        add("opd_pv_waves", "peak-to-valley optical path difference on the reference sphere", "waves", fine["opd_pv"], basis=GEOMETRIC_MODEL_ID,
            convergence=evidence(coarse["opd_pv"], fine["opd_pv"], refine, RAY_TOLERANCE, relative=False))
        arrays[f"{prefix}rays"]["reference_sphere"] = "INFINITY"
        # (the same wavefront on the finite sphere through the last surface: null where a ray misses it)
        arrays[f"{prefix}rays"]["last_surface_sphere_radius_mm"] = _sig(fine_level.radius)
        arrays[f"{prefix}rays"]["opd_rms_waves_on_last_surface_sphere"] = None if fine.get("opd_rms_last") is None else _sig(fine["opd_rms_last"])
    else:
        reason = fine_level.problems.get("wavefront", "RAY_OPL_UNDEFINED")
        for name in ("opd_rms_waves", "opd_pv_waves"):
            add(name, name.replace("_", " "), "waves", None, basis=GEOMETRIC_MODEL_ID, reason=reason,
                message=COMPONENT_MESSAGES[reason] if reason in COMPONENT_MESSAGES
                else "the rays' optical paths are not defined here (after an ideal or tabulated thin lens, a Faraday rotator or a grating), or no exit pupil exists")

    # the geometric MTF, in its regime
    rms = fine.get("opd_rms")
    geometric_ok = rms is not None and rms >= GEOMETRIC_MIN_RMS_WAVES
    geometric_check = ValidityCheck(parameter="opd_rms_waves", minimum=GEOMETRIC_MIN_RMS_WAVES, observed=None if rms is None else _sig(rms),
                                    within=None if rms is None else geometric_ok)
    models.append(ModelBasis(model_id=f"{prefix}geometric_mtf", name="geometric MTF of the ray spot", version=IMAGING_ANALYSIS_VERSION,
                             assumptions=("|sum over the rays of p_j exp(-i 2 pi f (r_j - r_c))|, p_j each ray's share of the irradiance on the pupil",), validity=(geometric_check,)))
    arrays[f"{prefix}frequencies_cyc_per_mm"] = [_sig(float(f)) for f in frequencies]
    # (outside the regime the curve is the stated model's number beyond its validity limit: an estimate, with the check)
    outside_regime = None if geometric_ok else ValueBasis(kind=ValueKind.ESTIMATE, validity=(geometric_check,))
    for index, name in enumerate(("tangential", "sagittal")):
        if components is not None:  # (the source's two components do not form one spot: there is no curve)
            add(f"geometric_mtf_{name}", f"geometric MTF, {name}", None, None, basis=f"{prefix}geometric_mtf", reason=components,
                message=COMPONENT_MESSAGES[components])
            continue
        key = f"{prefix}geometric_mtf_{name}"
        arrays[key] = [_sig(float(v)) for v in fine["geometric_mtf"][index]]
        difference = float(np.abs(fine["geometric_mtf"][index] - coarse["geometric_mtf"][index]).max())
        if geometric_ok:
            add(f"geometric_mtf_{name}", f"geometric MTF, {name}", None, None, basis=f"{prefix}geometric_mtf", values_ref=key,
                convergence=refined(difference, MTF_TOLERANCE))
        elif rms is None:
            add(f"geometric_mtf_{name}", f"geometric MTF, {name}", None, None, basis=f"{prefix}geometric_mtf", values_ref=key,
                convergence=refined(difference, MTF_TOLERANCE), reason="GEOMETRIC_REGIME_UNKNOWN", state=QuantityState.UNKNOWN,
                value_basis=outside_regime, depends_on=("opd_rms_waves",),
                message=f"the wavefront error is not determined here ({fine_level.problems.get('wavefront', 'no wavefront')})")
        else:
            add(f"geometric_mtf_{name}", f"geometric MTF, {name}", None, None, basis=f"{prefix}geometric_mtf", values_ref=key,
                convergence=refined(difference, MTF_TOLERANCE), reason="GEOMETRIC_REGIME_NOT_MET", state=QuantityState.OUT_OF_MODEL_SCOPE,
                value_basis=outside_regime, depends_on=("opd_rms_waves",),
                message=f"diffraction sets the MTF here: read mtf_{name}")

    # what leaves the field without a wavefront (and so without a point spread function)
    problem = fine_level.problems.get("diffraction") or components
    # the diffraction quantities, in their domain -- when the request asks for them
    if not request.diffraction:
        for name in DIFFRACTION_QUANTITIES:
            quantities.append(Quantity(quantity_id=prefix + name, name=name.replace("_", " "), scope=scope, state=QuantityState.NOT_COMPUTED,
                                       reason=NOT_REQUESTED, message="the request leaves out the diffraction quantities (diffraction false)"))
        return
    checks = [ValidityCheck(parameter="neighbour_phase_step_rad", maximum=MAX_NEIGHBOUR_PHASE_RAD, observed=_sig(fine_level.neighbour_phase),
                            within=fine_level.neighbour_phase <= MAX_NEIGHBOUR_PHASE_RAD)]
    period = lam / (fine_level.n * fine_level.direction_spacing) if fine_level.direction_spacing > 0.0 else math.inf
    checks.append(ValidityCheck(parameter="window_half_width_over_ray_period", maximum=ALIAS_SHARE, observed=_sig(half / period) if math.isfinite(period) else 0.0,
                                within=half <= ALIAS_SHARE * period))
    window = fine.get("window_share")
    checks.append(ValidityCheck(parameter="window_light_share", minimum=MIN_WINDOW_SHARE, observed=None if window is None else _sig(window),
                                within=None if window is None else window >= MIN_WINDOW_SHARE))
    checks.append(ValidityCheck(parameter="image_space_na", maximum=None if fine_level.vector else SCALAR_SINE_LIMIT, observed=_sig(fine_level.na), within=True))
    vector = fine_level.vector
    models.append(ModelBasis(model_id=diffraction_id, name=("vector" if vector else "scalar") + " Debye integral over the traced rays",
                             version=IMAGING_ANALYSIS_VERSION,
                             assumptions=ASSUMPTIONS_DIFFRACTION + (("unpolarized light: two orthogonal pupil polarizations, each carried as by an aplanatic lens, added in irradiance",)
                                                                    if vector else ("scalar field: image-space sine at most the scalar limit",)),
                             validity=tuple(checks)))
    if problem is None and not checks[0].within:
        problem = "PUPIL_UNDERSAMPLED"
    if problem is None and not checks[1].within:
        problem = "PUPIL_UNDERSAMPLED"
    window_problem = problem is None and window is not None and window < MIN_WINDOW_SHARE
    if problem is not None:
        message = {"RAY_OPL_UNDEFINED": "the rays' optical paths are not defined on this path: no wavefront, no diffraction",
                   "EXIT_PUPIL_UNDEFINED": "the light is collimated at this plane or no surface precedes it: there is no image to diffract into",
                   "RAY_MISSES_REFERENCE": "a ray does not meet the reference sphere",
                   **COMPONENT_MESSAGES,
                   "NO_LIGHT_ON_PUPIL": "no ray keeps any light to the plane (a sample or the coatings stop it): there is no pupil field",
                   "PUPIL_UNDERSAMPLED": f"the rays sample the pupil too coarsely for this aberration or window (phase step {fine_level.neighbour_phase:.3g} rad): "
                                         "raise launch_grid or narrow psf_half_width_um"}[problem]
        for name in DIFFRACTION_QUANTITIES:
            add(name, name.replace("_", " "), None, None, basis=diffraction_id, reason=problem, message=message)
        arrays[f"{prefix}geometric_reason"] = problem
        return
    add("strehl_ratio", "peak of the point spread function over that of the same rays without their phase", None, fine["strehl"], basis=diffraction_id,
        convergence=evidence(coarse["strehl"], fine["strehl"], refine, RAY_TOLERANCE))
    arrays[f"{prefix}psf"] = {"window_light_share": [_sig(res["window_share"]) for res in results_by_level], "half_width_um": _sig(half * MM_TO_UM),
                              "samples": samples, "peak_um": [_sig(fine["peak_position"][0] * MM_TO_UM), _sig(fine["peak_position"][1] * MM_TO_UM)],
                              "mtf_nyquist_cyc_per_mm": _sig(nyquist), "cutoff_cyc_per_mm": _sig(cutoff), "kernel": "vector" if vector else "scalar"}
    if request.include_psf_map:
        arrays[f"{prefix}psf"]["map_per_um2"] = _list(fine["psf"] / (MM_TO_UM * MM_TO_UM))
        arrays[f"{prefix}psf"]["axis_um"] = [_sig(float(v) * MM_TO_UM) for v in fine["axis"]]
    for name, key in (("mtf", "mtf"), ("diffraction_limited_mtf", "reference_mtf")):
        for index, direction in enumerate(("tangential", "sagittal")):
            array_key = f"{prefix}{name}_{direction}"
            values = fine[key][index]
            arrays[array_key] = [_sig(float(v)) if ok else None for v, ok in zip(values, resolved)]
            difference = float(np.abs(values - coarse[key][index])[resolved].max()) if resolved.any() else 0.0
            if window_problem:
                add(f"{name}_{direction}", f"{name.replace('_', ' ')}, {direction}", None, None, basis=diffraction_id, reason="PSF_WINDOW_TOO_SMALL",
                    message=f"the map holds {window:.3g} of the light: widen psf_half_width_um")
                continue
            add(f"{name}_{direction}", f"{name.replace('_', ' ')}, {direction} (null beyond the map's Nyquist frequency)", None, None, basis=diffraction_id,
                values_ref=array_key, convergence=refined(difference, MTF_TOLERANCE))
    for fraction in ENCIRCLED_FRACTIONS:
        percent = int(round(fraction * PERCENT))
        values = [res["encircled"]["radii"][fraction] for res in results_by_level]
        label = f"radius around the peak holding {percent} % of the light"
        if window_problem or values[-1] is None:
            add(f"encircled_radius_{percent}", label, "um", None, basis=diffraction_id, reason="PSF_WINDOW_TOO_SMALL",
                message=f"{percent} % of the light is not within the map's half width")
            continue
        add(f"encircled_radius_{percent}", label, "um", values[-1] * MM_TO_UM, basis=diffraction_id,
            convergence=evidence(None if values[0] is None else values[0] * MM_TO_UM, values[-1] * MM_TO_UM, refine, RAY_TOLERANCE))
    reference = [res["reference_encircled"]["radii"][ENCIRCLED_FRACTIONS[-1]] for res in results_by_level]
    if reference[-1] is not None:
        add("diffraction_limited_encircled_radius_80", "radius holding 80 % of the light of the same rays without their phase", "um", reference[-1] * MM_TO_UM,
            basis=diffraction_id, convergence=evidence(None if reference[0] is None else reference[0] * MM_TO_UM, reference[-1] * MM_TO_UM, refine, RAY_TOLERANCE))
    else:
        add("diffraction_limited_encircled_radius_80", "diffraction-limited 80 % radius", "um", None, basis=diffraction_id, reason="PSF_WINDOW_TOO_SMALL",
            message="80 % of the light is not within the map's half width")
    edges = fine["encircled"]["edges"]
    arrays[f"{prefix}encircled_energy"] = {"radius_um": [_sig(float(r) * MM_TO_UM) for r in edges],
                                          "fraction": [_sig(v) for v in fine["encircled"]["cumulative"]],
                                          "diffraction_limited_fraction": [_sig(v) for v in fine["reference_encircled"]["cumulative"]]}


__all__ = ["IMAGING_ANALYSIS", "IMAGING_INVALID", "ImagingError", "imaging", "parse_request"]
