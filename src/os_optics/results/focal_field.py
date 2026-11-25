"""The vector focal field of an ideal aplanatic objective near its focus (``focal_field(request)``, ``FocalFieldRequestV1``).

The light on the entrance pupil -- a stated uniform, annular or Gaussian illumination, a set of TEM00 beams, or the beams a run
delivers to the pupil plane (its ``handoff``) -- with its polarization is mapped onto the objective's reference sphere under
the sine condition and integrated to the focal region by the vector Debye integral (``os_optics.diffraction``). The reply is an
analysis document (``contracts.analysis.AnalysisResultV1``, analysis ``FOCAL_FIELD``) that changes no run.

Quantities. The power ledger (stated, inside the pupil, clipped by it, behind the window, through the focal plane) and its
energy balance; per focal spot the peak electric irradiance ``|E|^2`` and its position, the full widths at half maximum along
``x``, ``y`` and ``z``, the half widths at ``1/e^2`` of the peak along ``x`` and ``y``, the radii holding 50 % and 80 % of the
transmitted power, the power within twice the Airy radius and the Strehl ratio; the share of the focal-plane electric energy
in ``E_z``; and the maps ``|E|^2`` and ``S_z`` of transverse planes and of the ``xz`` and ``yz`` sections through the peak (with
``include_components`` the complex ``E_x``, ``E_y``, ``E_z``). Fields are normalized so that a plane wave has ``|E|^2`` equal to
its irradiance: ``|E|^2`` and ``S_z`` are in W/um^2, positions in um.

The focus is never described as a Gaussian beam unless it is one: the Gaussian waist ``lambda F / (pi w)`` is reported only
for a Gaussian illumination the pupil does not clip and whose far-field angle lies within the paraxial Gaussian scope
(``NumericalConfig.gaussian_scope_divergence_rad``); otherwise it is ``OUT_OF_MODEL_SCOPE``. The widths and radii above hold
for any spot.

Evidence. Every value is computed twice, with the stated sampling and with every node count doubled (``convergence``), and
checked against an independent evaluation of the same integral (``arrays.reference``): a rotationally symmetric pupil is
computed by the Bessel reduction and checked by the direct sphere sum; any other pupil is computed by the direct sphere sum and
checked by the Cartesian matrix Fourier transform. The energy balance compares the flux through the focal plane (the matrix
transform over one full period, where Parseval's theorem holds exactly) with the power the pupil sends through the window.

Validity. The Debye integral needs a Fresnel number ``F NA^2 / lambda`` far above 1 (``MIN_FRESNEL_NUMBER``) and a focal region
small against the reference sphere (``MAX_FOCAL_EXTENT_RATIO``); outside them the focal quantities are ``OUT_OF_MODEL_SCOPE``
and the pupil ledger stays. The objective is ideal: it is aberration-free and aplanatic at every field point the light uses, so
an off-axis input focuses at ``x = F l_x`` with the on-axis spot's shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import ValidationError

from ..analysis import completed, document
from ..config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig
from ..contracts.analysis import AnalysisResultV1, Convergence, ConvergenceMethod, ModelBasis, Quantity, QuantityScope, ValidityCheck
from ..contracts.basis import ValueBasis
from ..contracts.focal_field import (
    AnnularPupilV1,
    BeamsPupilV1,
    CircularPolarizationV1,
    FocalFieldRequestV1,
    GaussianPupilV1,
    HandoffBeamsPupilV1,
    JonesPolarizationV1,
    LinearPolarizationV1,
    UniformPupilV1,
    UnpolarizedV1,
)
from ..diffraction.debye import axial_flux
from ..diffraction.focus import BESSEL, DIRECT, GRID, Focus, PupilMode, Sampling
from ..diffraction.metrics import maximize, radius_for_fraction, widths, zernike
from ..diffraction.objective import Objective, Window
from ..diffraction.quadrature import gauss_legendre
from ..errors.status import QuantityState, ValueKind
from .handoff import handoff_scope, scope_marks
from .pupil_field import _beam_field, _Unavailable, beams_of_handoff, check_phases

FOCAL_FIELD_ANALYSIS = "FOCAL_FIELD"
FOCAL_FIELD_ANALYSIS_VERSION = "1.0"
FOCAL_FIELD_INVALID = "FOCAL_FIELD_INVALID"
DEBYE_MODEL_ID = "vector_debye_ideal_aplanatic"
GAUSSIAN_MODEL_ID = "paraxial_gaussian_focus"
#: The setting that holds the Gaussian description's far-field limit (``NumericalConfig``; a Project's ``numerical_settings``).
GAUSSIAN_LIMIT_SETTING = "gaussian_scope_divergence_rad"
#: The Debye integral holds for Fresnel numbers ``F NA^2 / lambda`` far above 1 (Li and Wolf, Opt. Commun. 39, 211 (1981)).
MIN_FRESNEL_NUMBER = 100.0
#: Largest distance of an evaluated point from the focal point, as a share of the reference sphere's radius.
MAX_FOCAL_EXTENT_RATIO = 0.1
MM_TO_UM = 1000.0
MM2_TO_UM2 = 1e-6
#: Every reported number is rounded to this many significant digits (the transforms' rounding lies far below).
SIGNIFICANT_DIGITS = 9
#: The relative step of that rounding, and a percentage.
ROUNDING_SHARE = 10.0 ** (1 - SIGNIFICANT_DIGITS)
PERCENT = 100.0
#: The maps' centre is the peak rounded to this share of the lateral and axial scales.
MAP_CENTRE_RESOLUTION = 1e-3
#: Differences, residuals and relative checks are reported to this resolution (relative to their scale): below it is rounding.
RELATIVE_RESOLUTION = 1e-12
#: Maps are rounded to this share of their largest value.
MAP_ROUNDING = 1e-9
#: A value has converged when doubling every node count moves it by at most this share of its value (quadratures) or of its
#: value or length scale (the grid transform, whose cells cut the pupil's rim, converges as 1 / cells).
QUADRATURE_TOLERANCE = 1e-6
GRID_TOLERANCE = 2e-3
#: Peak positions are reported to this share of the lateral / axial scale: a flat-topped peak fixes its position to about the
#: square root of the arithmetic's rounding, far below it.
POSITION_RESOLUTION_SHARE = 1e-4
#: The default maps reach this many lambda / NA laterally and this many axial scales lambda / (n (cos(theta_min) - cos(theta_max))).
MAP_LATERAL_SCALES = 3.0
MAP_AXIAL_SCALES = 3.0
#: The peak search spans this many lateral / axial scales around a spot's nominal focus (the window's focal shifts added).
SEARCH_LATERAL_SCALES = 0.5
SEARCH_AXIAL_SCALES = 2.0
#: Passes of the axial and lateral peak search; each narrows the spans by ``SEARCH_NARROWING``.
SEARCH_PASSES = 2
SEARCH_NARROWING = 4.0
#: Steps per scale of the outward walk that brackets a width.
WALK_STEPS_PER_SCALE = 40
#: Encircled energy: the curve reaches this many lambda / NA in ``ENCIRCLED_ANNULI`` annuli of ``RADIAL_NODES`` Gauss-Legendre
#: nodes and ``AZIMUTH_NODES`` azimuths.
ENCIRCLED_REACH_SCALES = 3.0
ENCIRCLED_ANNULI = 24
RADIAL_NODES = 4
AZIMUTH_NODES = 32
#: The map the grid transform's encircled energy is interpolated in: points per half width, and points beyond it.
ENCIRCLED_MAP_HALF = 64
SPLINE_MARGIN = 4
#: A rotationally symmetric pupil of one polarization has S_z harmonics up to cos(4 phi): eight azimuths integrate it exactly.
SYMMETRIC_AZIMUTH_NODES = 8
ENCIRCLED_FRACTIONS = (0.5, 0.8)
#: The spot power is the power within this many Airy radii (1.22 lambda / (2 NA)) of a spot's peak.
SPOT_DISC_AIRY_RADII = 2.0
AIRY_RADIUS_FACTOR = 0.61
#: Focal points per pupil-grid cell and axis of the energy-balance transform.
ENERGY_OVERSAMPLING = 1
#: Sampling demanded by an evaluation region far from a spot's own focus: nodes per radian of the pupil phase across it.
NODES_PER_RADIAN = 1.0
EXTRA_NODES = 16
GRID_PERIOD_MARGIN = 2.5
#: A Gaussian illumination counts as unclipped by the pupil when the pupil radius is at least this many beam radii.
UNCLIPPED_BEAM_RADII = 2.0
#: 1/e^2 of the peak.
E_SQUARED_LEVEL = math.exp(-2.0)

ASSUMPTIONS = (
    "ideal aplanatic objective obeying the sine condition h = F n sin(theta): no aberration at any field point used",
    "vector Debye integral over the reference sphere of radius n F (Richards and Wolf)",
    "energy-conserving apodization sqrt(cos(theta)); the pupil is lit in air",
    "a stated window is a plane-parallel plate normal to the axis, single pass, its reflections not included",
    "mutually incoherent pupil fields add in irradiance",
)


class FocalFieldError(Exception):
    """A request that is not valid, or an illumination no field can be built from: no analysis is made."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def parse_request(request: Any) -> FocalFieldRequestV1:
    if isinstance(request, FocalFieldRequestV1):
        return request
    try:
        return FocalFieldRequestV1.model_validate(request)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise FocalFieldError(FOCAL_FIELD_INVALID, problems) from None


def _sig(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _resolved(value: float, resolution: float) -> float:
    """``value`` rounded to the absolute ``resolution`` (a difference or a residual at the level of the arithmetic's rounding
    reports 0, the same in every runtime)."""

    if resolution <= 0.0:
        return _sig(value)
    return _sig(round(value / resolution) * resolution) + 0.0


def _map(values: np.ndarray) -> list[Any]:
    largest = float(np.abs(values).max())
    if largest <= 0.0:
        return values.tolist()
    step = largest * MAP_ROUNDING
    return (np.round(values / step) * step).tolist()


# ---------------------------------------------------------------------------------------------- the pupil


def _jones_list(polarization: Any) -> list[tuple[tuple[complex, complex], float]]:
    """``[(Jones vector, share of the power)]``: one entry, or two orthogonal halves for unpolarized light."""

    if isinstance(polarization, UnpolarizedV1):
        return [((1.0 + 0j, 0j), 0.5), ((0j, 1.0 + 0j), 0.5)]
    if isinstance(polarization, LinearPolarizationV1):
        angle = math.radians(polarization.angle_deg)
        return [((complex(math.cos(angle)), complex(math.sin(angle))), 1.0)]
    if isinstance(polarization, CircularPolarizationV1):
        half = math.sqrt(0.5)
        return [((complex(half), complex(0.0, -half if polarization.handedness == "RIGHT" else half)), 1.0)]
    assert isinstance(polarization, JonesPolarizationV1)
    jx, jy = complex(*polarization.jones[0]), complex(*polarization.jones[1])
    norm = math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)
    return [((jx / norm, jy / norm), 1.0)]


def _phase_functions(request: FocalFieldRequestV1, pupil_radius: float) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
                                                                                    Callable[[np.ndarray], np.ndarray] | None]:
    """``(phase(x, y), radial_phase(h) or None)`` in rad of the stated Zernike pupil phase, or ``(None, None)``."""

    if request.pupil_phase is None:
        return None, None
    terms = request.pupil_phase.terms

    def phase(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rho = np.sqrt(x * x + y * y) / pupil_radius
        phi = np.arctan2(y, x)
        total = np.zeros(np.shape(x))
        for t in terms:
            total = total + t.coefficient_waves * zernike(t.n, t.m, rho, phi)
        return 2.0 * math.pi * total

    if any(t.m != 0 for t in terms):
        return phase, None
    return phase, lambda h: phase(h, np.zeros(np.shape(h)))


def _carrier(phase: Callable[[np.ndarray, np.ndarray], np.ndarray] | None, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ones(np.shape(x), dtype=np.complex128) if phase is None else np.exp(1j * phase(x, y))


def _stated_modes(illumination: Any, jones_list: list[tuple[tuple[complex, complex], float]], objective: Objective,
                  phase: Any, radial_phase: Any) -> tuple[tuple[PupilMode, ...], float, float]:
    """``(modes, inner sine of an annulus, incident power)`` of a UNIFORM, ANNULAR or GAUSSIAN illumination."""

    a = objective.pupil_radius_mm
    k0 = objective.k0
    lx, ly = illumination.direction_cosines
    inner = illumination.inner_radius_fraction if isinstance(illumination, AnnularPupilV1) else 0.0
    if isinstance(illumination, GaussianPupilV1):
        w, (cx, cy) = illumination.radius_mm, illumination.center_mm
        peak = math.sqrt(2.0 * illumination.power_w / (math.pi * w * w))

        def amplitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            dx, dy = x - cx, y - cy
            return peak * np.exp(-(dx * dx + dy * dy) / (w * w))

        centred = cx == 0.0 and cy == 0.0
    else:
        level = math.sqrt(illumination.power_w / (math.pi * a * a * (1.0 - inner * inner)))

        def amplitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.full(np.shape(x), level)

        centred = True
    symmetric = centred and lx == 0.0 and ly == 0.0 and (phase is None or radial_phase is not None)
    modes = []
    for jones, share in jones_list:
        scale = math.sqrt(share)

        def field(x: np.ndarray, y: np.ndarray, jones: tuple[complex, complex] = jones, scale: float = scale) -> tuple[np.ndarray, np.ndarray]:
            u = scale * amplitude(x, y) * np.exp(1j * k0 * (lx * x + ly * y)) * _carrier(phase, x, y)
            return jones[0] * u, jones[1] * u

        def radial(h: np.ndarray, scale: float = scale) -> np.ndarray:
            u = scale * amplitude(h, np.zeros(np.shape(h))).astype(np.complex128)
            return u if radial_phase is None else u * np.exp(1j * radial_phase(h))

        modes.append(PupilMode(field, radial if symmetric else None, jones if symmetric else None))
    return tuple(modes), inner * objective.sine_max, illumination.power_w


def _beam_modes(beams: tuple[Any, ...], composition: str, jones_of: list[list[tuple[tuple[complex, complex], float]]],
                objective: Objective, phase: Any) -> tuple[PupilMode, ...]:
    """Modes of TEM00 beams: coherent beams form one field per polarization share, incoherent beams one field each."""

    k0 = objective.k0
    coherent = composition == "COHERENT"

    def beam_field(beam: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _beam_field(beam, k0, x, y, coherent) * _carrier(phase, x, y)

    if coherent:
        shares = len(jones_of[0])

        def make(index: int) -> PupilMode:
            def field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                jx = np.zeros(np.shape(x), dtype=np.complex128)
                jy = np.zeros(np.shape(x), dtype=np.complex128)
                for beam, jones in zip(beams, jones_of):
                    (ex, ey), share = jones[index]
                    u = math.sqrt(share) * beam_field(beam, x, y)
                    jx, jy = jx + ex * u, jy + ey * u
                return jx, jy
            return PupilMode(field)

        return tuple(make(index) for index in range(shares))
    modes = []
    for beam, jones in zip(beams, jones_of):
        for (ex, ey), share in jones:
            def field(x: np.ndarray, y: np.ndarray, beam: Any = beam, ex: complex = ex, ey: complex = ey, share: float = share) -> tuple[np.ndarray, np.ndarray]:
                u = math.sqrt(share) * beam_field(beam, x, y)
                return ex * u, ey * u
            modes.append(PupilMode(field))
    return tuple(modes)


def _handoff_jones(handoff: Any, label: str) -> tuple[complex, complex]:
    for beam in handoff["beams"]:
        if str(beam["path_id"]) != label:
            continue
        polarization = beam.get("polarization") or {}
        jones = polarization.get("jones")
        if jones is None:
            raise _Unavailable("POLARIZATION_NOT_DEFINED", f"beam {label} has no Jones vector at this plane: its vector focal field is not defined")
        e1, e2 = polarization["basis_e1_uvn"], polarization["basis_e2_uvn"]
        j1, j2 = complex(*jones[0]), complex(*jones[1])
        jx, jy = j1 * float(e1[0]) + j2 * float(e2[0]), j1 * float(e1[1]) + j2 * float(e2[1])
        norm = math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)
        if norm <= 0.0:
            raise _Unavailable("POLARIZATION_NOT_DEFINED", f"beam {label} has no transverse field on the pupil plane")
        return jx / norm, jy / norm
    raise _Unavailable("POLARIZATION_NOT_DEFINED", f"beam {label} is not in the handoff")


# ---------------------------------------------------------------------------------------------- the analysis


class _Setup:
    """Everything a request determines before any integral is evaluated."""

    def __init__(self, request: FocalFieldRequestV1, handoff: Any, phase_free: bool = False) -> None:
        o = request.objective
        window = None if o.window is None else Window(o.window.thickness_mm, o.window.refractive_index, o.window.surfaces == "UNCOATED")
        self.objective = Objective(request.wavelength_nm, o.numerical_aperture, o.focal_length_mm, o.focal_medium_index, window)
        phase, radial_phase = (None, None) if phase_free else _phase_functions(request, self.objective.pupil_radius_mm)
        self.has_phase = request.pupil_phase is not None or window is not None
        illumination = request.illumination
        f = self.objective.focal_length_mm
        self.gaussian_radius: float | None = None
        #: The handoff beams an incoherent sum leaves out, and the phase reference of a single beam without a stated phase.
        self.excluded: list[dict[str, Any]] = []
        #: The model-scope findings of the handoff beams the field is built from (``handoff.handoff_scope``).
        self.scope: list[dict[str, Any]] = []
        self.phase_reference: str | None = None
        if isinstance(illumination, (UniformPupilV1, AnnularPupilV1, GaussianPupilV1)):
            self.modes, inner_sine, self.incident = _stated_modes(illumination, _jones_list(request.polarization), self.objective, phase, radial_phase)
            self.spots = [("focus", f * illumination.direction_cosines[0], f * illumination.direction_cosines[1], self.incident)]
            if isinstance(illumination, GaussianPupilV1):
                self.gaussian_radius = illumination.radius_mm if illumination.center_mm == (0.0, 0.0) else None
        else:
            if isinstance(illumination, BeamsPupilV1):
                beams: tuple[Any, ...] = illumination.beams
                jones_of = [_jones_list(request.polarization) for _ in beams]
            else:
                assert isinstance(illumination, HandoffBeamsPupilV1)
                beams, self.excluded = beams_of_handoff(handoff, request.wavelength_nm, illumination.path_ids, f, illumination.composition)
                jones_of = [[(_handoff_jones(handoff, beam.label), 1.0)] for beam in beams]
                self.scope = handoff_scope(handoff, [beam.label for beam in beams])
            self.phase_reference = check_phases(beams, illumination.composition == "COHERENT")
            self.modes = _beam_modes(beams, illumination.composition, jones_of, self.objective, phase)
            inner_sine = 0.0
            self.incident = math.fsum(beam.power_w for beam in beams)
            self.spots = [(beam.label, f * beam.direction_cosines[0], f * beam.direction_cosines[1], beam.power_w) for beam in beams]
        self.theta_min = math.asin(inner_sine)
        n = self.objective.focal_index
        self.lateral_scale = self.objective.wavelength_mm / self.objective.numerical_aperture
        self.axial_scale = self.objective.wavelength_mm / (n * (math.cos(self.theta_min) - math.cos(self.objective.theta_max)))
        self.z_centre, self.z_reach = 0.0, 0.0
        if window is not None:
            paraxial = window.paraxial_focal_shift_mm(n)
            theta = self.objective.theta_max
            sin_w = n * math.sin(theta) / window.index
            marginal = window.thickness_mm * (1.0 - math.tan(math.asin(sin_w)) / math.tan(theta))
            self.z_centre, self.z_reach = 0.5 * (paraxial + marginal), 0.5 * abs(marginal - paraxial)
        out = request.output
        if out.half_extent_um is not None:
            self.half = tuple(v / MM_TO_UM for v in out.half_extent_um)
        else:
            xs, ys = [s[1] for s in self.spots], [s[2] for s in self.spots]
            spread = (0.5 * (max(xs) - min(xs)), 0.5 * (max(ys) - min(ys)))
            self.half = (spread[0] + MAP_LATERAL_SCALES * self.lateral_scale, spread[1] + MAP_LATERAL_SCALES * self.lateral_scale,
                         MAP_AXIAL_SCALES * self.axial_scale + self.z_reach)
        # the evaluated region's reach from any spot: it sets the sampling of a pupil whose phase is tilted
        xs, ys = [s[1] for s in self.spots], [s[2] for s in self.spots]
        self.map_centre = (0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys)))
        reach = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + math.hypot(self.half[0], self.half[1]) + self.lateral_scale
        self.extent = max(math.hypot(abs(self.map_centre[0]) + self.half[0], abs(self.map_centre[1]) + self.half[1]),
                          abs(self.z_centre) + self.half[2])
        beta = self.objective.k * self.objective.sine_max * reach
        needed = int(math.ceil(NODES_PER_RADIAN * beta)) + EXTRA_NODES
        s = request.sampling
        cells = int(math.ceil(GRID_PERIOD_MARGIN * 2.0 * reach * 2.0 * self.objective.numerical_aperture / self.objective.wavelength_mm))
        self.sampling = Sampling(max(s.theta_nodes, needed), max(s.phi_nodes, needed), max(s.grid_cells, cells + cells % 2))


def _profile(focus: Focus, method: str, scale: int, axis: int, fixed: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:
    def values(positions: np.ndarray) -> np.ndarray:
        points = np.tile(np.array(fixed, dtype=np.float64)[:, None], (1, positions.size))
        points[axis] = positions
        return focus.irradiance(points, method, scale)[0]
    return values


def _spot(focus: Focus, setup: _Setup, method: str, scale: int, centre: tuple[float, float]) -> dict[str, Any]:
    """Peak, widths and encircled energy of the spot near ``centre`` (mm)."""

    found, _ = _find_peak(focus, setup, method, scale, centre)
    # the peak at its reported resolution: widths and encircled energy are measured through it, so the search's last digits do
    # not reach them
    scales = (setup.lateral_scale, setup.lateral_scale, setup.axial_scale)
    position = tuple(_position(c, s) / MM_TO_UM for c, s in zip(found, scales))
    peak = float(focus.irradiance(np.array(position, dtype=np.float64)[:, None], method, scale)[0][0])
    step = (setup.lateral_scale / WALK_STEPS_PER_SCALE, setup.lateral_scale / WALK_STEPS_PER_SCALE, setup.axial_scale / WALK_STEPS_PER_SCALE)
    out: dict[str, Any] = {"position": position, "peak": peak, "fwhm": [], "half_e2": [], "crossings": []}
    for axis in range(len(position)):
        profile = _profile(focus, method, scale, axis, position)
        half = widths(profile, position[axis], peak, step[axis], 0.5)
        out["fwhm"].append(None if half is None else half[1] - half[0])
        out["crossings"].append((axis, 0.5, half))
        if axis < len("xy"):
            e2 = widths(profile, position[axis], peak, step[axis], E_SQUARED_LEVEL)
            out["half_e2"].append(None if e2 is None else 0.5 * (e2[1] - e2[0]))
            out["crossings"].append((axis, E_SQUARED_LEVEL, e2))
    out["encircled"] = _encircled(focus, setup, method, scale, position)
    return out


def _flux_in_ring(flux_at: Callable[[np.ndarray], np.ndarray], azimuths: int, position: tuple[float, float, float], low: float, high: float) -> float:
    """The axial flux through the annulus ``low <= r <= high`` around ``position``: Gauss-Legendre in ``r``, uniform in ``phi``."""

    radii, weights = gauss_legendre(RADIAL_NODES, low, high)
    phi = 2.0 * math.pi * (np.arange(azimuths) + 0.5) / azimuths
    rr, pp = np.meshgrid(radii, phi, indexing="ij")
    points = np.stack((position[0] + (rr * np.cos(pp)).ravel(), position[1] + (rr * np.sin(pp)).ravel(), np.full(rr.size, position[2])))
    flux = flux_at(points).reshape(rr.shape)
    return float((flux.sum(axis=1) * radii * weights).sum()) * 2.0 * math.pi / azimuths


def _map_flux(focus: Focus, setup: _Setup, scale: int, position: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:
    """``S_z`` at any point near ``position`` by cubic-spline interpolation of a grid-transform map around it (the transform
    evaluates a whole map at the cost of a few points)."""

    from scipy.ndimage import map_coordinates

    reach = ENCIRCLED_REACH_SCALES * setup.lateral_scale
    pitch = reach / ENCIRCLED_MAP_HALF
    count = 2 * ENCIRCLED_MAP_HALF + 1 + 2 * SPLINE_MARGIN
    first = -(ENCIRCLED_MAP_HALF + SPLINE_MARGIN) * pitch
    offsets = first + np.arange(count) * pitch
    flux = np.zeros((count, count))
    for e, h in focus.plane(position[0] + offsets, position[1] + offsets, position[2], scale):
        flux = flux + axial_flux(e, h)

    def flux_at(points: np.ndarray) -> np.ndarray:
        rows = (points[1] - position[1] - first) / pitch
        columns = (points[0] - position[0] - first) / pitch
        return map_coordinates(flux, np.stack((rows, columns)), order=len("xyz"), mode="nearest")

    return flux_at


def _encircled(focus: Focus, setup: _Setup, method: str, scale: int, position: tuple[float, float, float]) -> dict[str, Any]:
    """The axial flux within radius ``r`` of the peak on its plane, as a function and on the annuli's edges."""

    if method == GRID:
        flux_at, azimuths = _map_flux(focus, setup, scale, position), AZIMUTH_NODES
    else:
        def flux_at(points: np.ndarray) -> np.ndarray:
            return focus.irradiance(points, method, scale)[1]
        azimuths = SYMMETRIC_AZIMUTH_NODES if method == BESSEL else AZIMUTH_NODES
    reach = ENCIRCLED_REACH_SCALES * setup.lateral_scale
    edges = np.linspace(0.0, reach, ENCIRCLED_ANNULI + 1)
    rings = [_flux_in_ring(flux_at, azimuths, position, float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]
    cumulative = [0.0]
    for ring in rings:
        cumulative.append(cumulative[-1] + ring)

    def within(radius: float) -> float:
        index = min(int(radius / (reach / ENCIRCLED_ANNULI)), ENCIRCLED_ANNULI)
        if index >= ENCIRCLED_ANNULI:
            return cumulative[-1] + _flux_in_ring(flux_at, azimuths, position, reach, radius) if radius > reach else cumulative[-1]
        return cumulative[index] + _flux_in_ring(flux_at, azimuths, position, float(edges[index]), radius)

    return {"edges": edges, "cumulative": cumulative, "within": within, "reach": reach}


def _encircled_radius(encircled: dict[str, Any], power: float) -> float | None:
    """The radius holding ``power``: the annulus from the cumulative table, the radius inside it by bisection."""

    cumulative, edges = encircled["cumulative"], encircled["edges"]
    for index in range(1, len(cumulative)):
        if cumulative[index] >= power:
            return radius_for_fraction(encircled["within"], power, float(edges[index - 1]), float(edges[index]))
    return None


def _convergence(coarse: float | None, fine: float | None, parameter: str, tolerance: float = QUADRATURE_TOLERANCE,
                 scale: float = 0.0) -> Convergence:
    """Refinement evidence: converged when the doubled sampling moves the value by at most ``tolerance`` of the larger of its
    magnitude and ``scale``."""

    if coarse is None or fine is None:
        return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, converged=False)
    difference = abs(fine - coarse)
    size = max(abs(fine), abs(coarse), scale)
    resolution = max(difference, abs(fine) * ROUNDING_SHARE)
    return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, coarse=_sig(coarse), fine=_sig(fine),
                       difference=_resolved(difference, size * RELATIVE_RESOLUTION), resolution=_sig(resolution), converged=difference <= tolerance * size)


def _position(value_mm: float, scale_mm: float) -> float:
    """A peak coordinate in um, rounded to ``POSITION_RESOLUTION_SHARE`` of its scale."""

    step = POSITION_RESOLUTION_SHARE * scale_mm
    return round(value_mm / step) * step * MM_TO_UM + 0.0


def focal_field(request: Any, handoff: Any = None, *, run_identity: dict[str, Any] | None = None,
                config: NumericalConfig | None = None, engine_build_id: str | None = None) -> AnalysisResultV1:
    """The focal-field analysis of ``request`` (module docstring). ``handoff``: the ``results handoff`` document of the plane
    whose beams light the pupil (``HANDOFF_BEAMS``); ``run_identity``: that run's identity (``RunResults.identity()``);
    ``engine_build_id``: the Engine build id the caller holds (``os_optics.analysis.document``).
    ``FocalFieldError`` when the request is invalid or no field can be built from its illumination."""

    parsed = parse_request(request)
    try:
        setup = _Setup(parsed, handoff)
    except _Unavailable as exc:
        raise FocalFieldError(exc.code, exc.message) from None
    return _analysis(parsed, handoff, setup, run_identity or {}, config or _handoff_config(handoff), engine_build_id)


def _handoff_config(handoff: Any) -> NumericalConfig:
    """The numerical configuration of a call that names none: the Engine defaults, with the limits the handoff document
    states for its run (``numerical_settings``), so a limit a Project sets holds for a field built from its handoff."""

    stated = handoff.get("numerical_settings") if isinstance(handoff, dict) else None
    if not isinstance(stated, dict):
        return NumericalConfig()
    try:
        return NumericalConfig(**{name: value for name, value in stated.items() if name in NumericalConfig.model_fields})
    except ValidationError as exc:
        problems = "; ".join(f"numerical_settings.{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False))
        raise FocalFieldError(FOCAL_FIELD_INVALID, problems) from None


def _analysis(request: FocalFieldRequestV1, handoff: Any, setup: _Setup, identity: dict[str, Any], cfg: NumericalConfig,
              engine_build_id: str | None = None) -> AnalysisResultV1:
    objective = setup.objective
    focus = Focus(objective, setup.modes, setup.sampling, setup.theta_min)
    primary, reference = (BESSEL, DIRECT) if focus.symmetric else (GRID, DIRECT)
    sampling = setup.sampling
    refine = f"grid_cells {sampling.grid_cells} doubled" if primary == GRID else f"theta_nodes {sampling.theta_nodes} doubled"
    metric_tolerance = GRID_TOLERANCE if primary == GRID else QUADRATURE_TOLERANCE
    settings = {"request": request.model_dump(mode="json"), "effective_sampling": {"theta_nodes": sampling.theta_nodes, "phi_nodes": sampling.phi_nodes,
                                                                                    "grid_cells": sampling.grid_cells},
                "method": primary, "reference_method": reference}
    if setup.gaussian_radius is not None and cfg.gaussian_scope_divergence_rad != DEFAULT_NUMERICAL_CONFIG.gaussian_scope_divergence_rad:
        # (the Gaussian description's limit in force, where a Project set it: the settings record of this analysis)
        settings["numerical_settings"] = {GAUSSIAN_LIMIT_SETTING: cfg.gaussian_scope_divergence_rad}
    if handoff is not None and isinstance(request.illumination, HandoffBeamsPupilV1):
        settings["handoff_sha256"] = hashlib.sha256(json.dumps(handoff.get("beams"), sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    fresnel = objective.fresnel_number
    extent_ratio = setup.extent / objective.sphere_radius_mm
    checks = (ValidityCheck(parameter="fresnel_number", minimum=MIN_FRESNEL_NUMBER, observed=_sig(fresnel), within=fresnel >= MIN_FRESNEL_NUMBER),
              ValidityCheck(parameter="focal_extent_over_sphere_radius", maximum=MAX_FOCAL_EXTENT_RATIO, observed=_sig(extent_ratio),
                            within=extent_ratio <= MAX_FOCAL_EXTENT_RATIO),
              ValidityCheck(parameter="marginal_sine", maximum=1.0, observed=_sig(objective.sine_max), within=True))
    in_domain = all(c.within for c in checks)
    models = [ModelBasis(model_id=DEBYE_MODEL_ID, name="vector Debye integral of an ideal aplanatic objective", version=FOCAL_FIELD_ANALYSIS_VERSION,
                         assumptions=ASSUMPTIONS, validity=checks)]
    quantities: list[Quantity] = []
    arrays: dict[str, Any] = {"conventions": {
        "frame": "x, y on the pupil and the focal plane are parallel; z along the propagation, from the objective's focal point",
        "units": "um for focal positions, W/um^2 for |E|^2 and S_z, W for powers",
        "field": "a plane wave in the focal medium has |E|^2 equal to its irradiance; S_z = Re(E_x conj(H_y) - E_y conj(H_x)), H = s x E per plane wave",
        "time_dependence": "exp(-i w t); propagation exp(+i k s.r)",
    }}
    if setup.phase_reference is not None:
        # (one beam whose phase the handoff does not define: its constant phase is taken as 0; no |E|^2 or flux depends on it)
        arrays["phase_reference"] = setup.phase_reference
    if setup.excluded:
        # (an incoherent sum that leaves handoff beams out: each spot is right, the powers are those of the beams it is built from)
        arrays["illumination"] = {"complete": False, "excluded_beams": setup.excluded}

    def add(quantity_id: str, name: str, unit: str | None, value: float | None, *, coarse: float | None = None, scope: QuantityScope | None = None,
            reason: str | None = None, message: str = "", basis: str = DEBYE_MODEL_ID, convergence: Convergence | None = None,
            state: QuantityState | None = None, values_ref: str | None = None, tolerance: float | None = None, size: float = 0.0,
            value_basis: ValueBasis | None = None) -> None:
        if value_basis is not None and value is not None:
            # (the stated model's number outside its validity: reported with what makes it an estimate; it decides nothing)
            quantities.append(Quantity(quantity_id=quantity_id, name=name, unit=unit, scope=scope or QuantityScope(),
                                       state=state or QuantityState.OUT_OF_MODEL_SCOPE, value=_sig(value), reason=reason or "NOT_DETERMINED",
                                       message=message, basis=basis, convergence=convergence, value_basis=value_basis))
            return
        if value is None and values_ref is None:
            quantities.append(Quantity(quantity_id=quantity_id, name=name, unit=unit, scope=scope or QuantityScope(),
                                       state=state or QuantityState.UNKNOWN, reason=reason or "NOT_DETERMINED", message=message, basis=basis))
            return
        conv = convergence if convergence is not None else (
            _convergence(coarse, value, refine, tolerance or metric_tolerance, size) if coarse is not None else None)
        quantities.append(Quantity(quantity_id=quantity_id, name=name, unit=unit, scope=scope or QuantityScope(), state=QuantityState.KNOWN,
                                   value=None if value is None else _sig(value), values_ref=values_ref, basis=basis, convergence=conv,
                                   message=message))

    # the power ledger: the pupil integrals by the polar rule, at the stated and the doubled node counts
    pupil = (focus.sphere_power(1, transmit=False), focus.sphere_power(2, transmit=False))
    transmitted = (focus.sphere_power(1), focus.sphere_power(2))
    add("incident_power_W", "power of the stated illumination" if not setup.excluded else "power of the beams the field is built from", "W", setup.incident,
        convergence=Convergence(method=ConvergenceMethod.ANALYTIC),
        message="" if not setup.excluded else "the handoff beams left out are listed in arrays.illumination.excluded_beams")
    polar_refine = f"theta_nodes {sampling.theta_nodes} and phi_nodes {sampling.phi_nodes} doubled"
    add("pupil_power_W", "power inside the pupil", "W", pupil[1], convergence=_convergence(pupil[0], pupil[1], polar_refine))
    clipped = [max(setup.incident - p, 0.0) for p in pupil]
    add("clipped_power_W", "power the pupil's rim clips", "W", clipped[1], convergence=_convergence(clipped[0], clipped[1], polar_refine, scale=setup.incident))
    add("transmitted_power_W", "power sent toward the focus (behind the window)", "W", transmitted[1],
        convergence=_convergence(transmitted[0], transmitted[1], polar_refine))

    if not in_domain:
        message = ("the Debye integral does not hold here: " + "; ".join(f"{c.parameter} {c.observed} outside its limit" for c in checks if not c.within)
                   + ". Use a longer focal length or a larger aperture (Fresnel number), or a smaller focal region")
        for quantity_id, unit in (("peak_electric_irradiance", "W/um^2"), ("fwhm_x", "um"), ("fwhm_y", "um"), ("fwhm_z", "um"),
                                  ("focal_plane_flux_W", "W"), ("electric_irradiance_maps", "W/um^2")):
            add(quantity_id, quantity_id.replace("_", " "), unit, None, state=QuantityState.OUT_OF_MODEL_SCOPE, reason="DEBYE_APPROXIMATION_INVALID", message=message)
        # (the analysis ran to its end: the pupil ledger is stated, the focal quantities say they are outside the model)
        return document(FOCAL_FIELD_ANALYSIS, FOCAL_FIELD_ANALYSIS_VERSION, settings, completed(),
                        engine_input_hash=identity.get("engine_input_hash"), run_result_hash=identity.get("result_hash"),
                        models=models, quantities=_scoped(quantities, setup), arrays=arrays, engine_build_id=engine_build_id)

    # the energy balance: the flux through the focal plane over one period of the grid transform
    plane = [focus.plane_integrals(setup.z_centre, ENERGY_OVERSAMPLING, setup.map_centre, scale) for scale in (1, 2)]
    flux = [p["flux_W"] for p in plane]
    grid_refine = f"grid_cells {sampling.grid_cells} doubled"
    add("focal_plane_flux_W", "axial flux through the focal plane over one period of the grid transform", "W", flux[1],
        convergence=_convergence(flux[0], flux[1], grid_refine, GRID_TOLERANCE))
    residual = [(f - t) / t for f, t in zip(flux, transmitted)]
    add("energy_relative_residual", "(focal-plane flux - transmitted power) / transmitted power", None, residual[1],
        convergence=_convergence(residual[0], residual[1], grid_refine, GRID_TOLERANCE, 1.0),
        message="the grid's cells cut the pupil's rim, so the balance closes as 1 / grid_cells; Parseval's identity of the transform itself is in arrays.energy_balance")
    longitudinal = (focus.longitudinal_fraction(1), focus.longitudinal_fraction(2))
    add("longitudinal_energy_fraction", "share of E_z in the integral of |E|^2 over a transverse plane", None, longitudinal[1],
        convergence=_convergence(longitudinal[0], longitudinal[1], polar_refine))
    arrays["energy_balance"] = {"stated_W": _sig(setup.incident), "pupil_W": _sig(pupil[1]), "transmitted_W": _sig(transmitted[1]),
                                "focal_plane_flux_W": [_sig(v) for v in flux],
                                "parseval_relative_residuals": [_resolved(abs(p["flux_W"] - p["sampled_W"]) / p["sampled_W"], RELATIVE_RESOLUTION) for p in plane],
                                "grid_cells": [sampling.grid_cells, 2 * sampling.grid_cells], "period_um": [_sig(p["period_mm"] * MM_TO_UM) for p in plane],
                                "longitudinal_energy_fraction_grid": [_sig(p["longitudinal_integral"] / p["electric_integral"]) for p in plane]}

    # the spots: peak, widths, encircled energy, by the primary method at both samplings, checked by the reference method
    spots = setup.spots
    reference_rows: list[dict[str, Any]] = []
    strehl_reference = None
    if setup.has_phase and len(spots) == 1 and len(setup.modes) <= len("xy"):
        free = _Setup(request, handoff, phase_free=True)
        strehl_reference = Focus(objective, free.modes, free.sampling, free.theta_min, with_phase=False)
    first_peak: tuple[float, float, float] | None = None
    for index, (label, cx, cy, spot_power) in enumerate(spots):
        prefix = "" if len(spots) == 1 else f"{label.replace('/', '_')}."
        scope = QuantityScope() if len(spots) == 1 else QuantityScope(path_id=label)
        results = [_spot(focus, setup, primary, scale, (cx, cy)) for scale in (1, 2)]
        coarse, fine = results
        if first_peak is None:
            first_peak = fine["position"]
        for axis, name in enumerate("xyz"):
            scale = setup.axial_scale if name == "z" else setup.lateral_scale
            add(f"{prefix}peak_position_{name}", f"position of the peak along {name}", "um", _position(fine["position"][axis], scale),
                coarse=_position(coarse["position"][axis], scale), scope=scope, size=scale * MM_TO_UM)
        add(f"{prefix}peak_electric_irradiance", "peak |E|^2", "W/um^2", fine["peak"] * MM2_TO_UM2, coarse=coarse["peak"] * MM2_TO_UM2, scope=scope)
        for axis, name in enumerate("xyz"):
            value, before = fine["fwhm"][axis], coarse["fwhm"][axis]
            add(f"{prefix}fwhm_{name}", f"full width at half maximum of |E|^2 along {name} through the peak", "um",
                None if value is None else value * MM_TO_UM, coarse=None if before is None else before * MM_TO_UM, scope=scope,
                reason="WIDTH_NOT_BRACKETED", message="|E|^2 does not fall to half its peak within the searched range")
        for axis, name in enumerate("xy"):
            value, before = fine["half_e2"][axis], coarse["half_e2"][axis]
            add(f"{prefix}radius_1e2_{name}", f"half width at 1/e^2 of the peak |E|^2 along {name}", "um",
                None if value is None else value * MM_TO_UM, coarse=None if before is None else before * MM_TO_UM, scope=scope,
                reason="WIDTH_NOT_BRACKETED", message="|E|^2 does not fall to 1/e^2 of its peak within the searched range")
        # encircled energy as a share of the transmitted power
        for fraction in ENCIRCLED_FRACTIONS:
            radii = [_encircled_radius(res["encircled"], fraction * transmitted[i]) for i, res in enumerate(results)]
            percent = int(round(fraction * PERCENT))
            add(f"{prefix}encircled_radius_{percent}", f"radius around the peak holding {percent} % of the transmitted power on the peak's plane", "um",
                None if radii[1] is None else radii[1] * MM_TO_UM, coarse=None if radii[0] is None else radii[0] * MM_TO_UM, scope=scope,
                reason="BEYOND_ENCIRCLED_REACH", message=f"less than {percent} % of the power lies within {ENCIRCLED_REACH_SCALES} lambda / NA of the peak")
        disc = SPOT_DISC_AIRY_RADII * AIRY_RADIUS_FACTOR * setup.lateral_scale
        add(f"{prefix}power_within_disc_W", f"axial flux within {SPOT_DISC_AIRY_RADII} Airy radii ({_sig(disc * MM_TO_UM)} um) of the peak", "W",
            fine["encircled"]["within"](disc), coarse=coarse["encircled"]["within"](disc), scope=scope)
        if index == 0:
            edges = fine["encircled"]["edges"]
            arrays["encircled_energy"] = {"radius_um": [_sig(float(r) * MM_TO_UM) for r in edges],
                                          "fraction_of_transmitted": [_sig(v / transmitted[1]) for v in fine["encircled"]["cumulative"]],
                                          "spot": label}
            if request.output.encircled_energy_radii_um:
                arrays["encircled_energy"]["requested"] = [
                    {"radius_um": r, "fraction_of_transmitted": _sig(fine["encircled"]["within"](r / MM_TO_UM) / transmitted[1])}
                    for r in request.output.encircled_energy_radii_um]
        # the independent reference: the peak and |E|^2 at every reported crossing, relative to the level
        peak_ref = float(focus.irradiance(np.array(fine["position"], dtype=np.float64)[:, None], reference, 1)[0][0])
        reference_rows.append({"spot": label, "quantity": "peak_electric_irradiance", "method": reference,
                               "relative_difference": _resolved((peak_ref - fine["peak"]) / fine["peak"], RELATIVE_RESOLUTION)})
        for axis, level, pair in fine["crossings"]:
            if pair is None:
                continue
            for edge in pair:
                point = list(fine["position"])
                point[axis] = edge
                value = float(focus.irradiance(np.array(point, dtype=np.float64)[:, None], reference, 1)[0][0])
                reference_rows.append({"spot": label, "quantity": f"level {_sig(level)} along {'xyz'[axis]}", "method": reference,
                                       "relative_difference": _resolved((value / fine["peak"] - level) / level, RELATIVE_RESOLUTION)})
        # Strehl ratio: the peak against the same pupil without its phase (the window's transmissions kept)
        if len(spots) > 1:
            add(f"{prefix}strehl_ratio", "Strehl ratio", None, None, scope=scope, state=QuantityState.NOT_APPLICABLE, reason="SEVERAL_SPOTS",
                message="a Strehl ratio needs one spot of one pupil field")
        elif not setup.has_phase:
            add("strehl_ratio", "Strehl ratio", None, 1.0, convergence=Convergence(method=ConvergenceMethod.ANALYTIC),
                message="the ideal objective without a stated phase: 1 by definition")
        elif strehl_reference is not None:
            free_method = BESSEL if strehl_reference.symmetric else DIRECT
            ratios = []
            for scale, res in zip((1, 2), results):
                ratios.append(res["peak"] / _find_peak(strehl_reference, setup, free_method, scale, (cx, cy), with_window=False)[1])
            add("strehl_ratio", "peak |E|^2 over that of the same pupil without its phase (window and Zernike terms)", None, ratios[1], coarse=ratios[0])
        else:
            add("strehl_ratio", "Strehl ratio", None, None, state=QuantityState.NOT_APPLICABLE, reason="SEVERAL_FIELDS",
                message="a Strehl ratio needs one coherent pupil field")
    arrays["reference"] = reference_rows

    # the Gaussian description, only where it holds
    _gaussian_waist(add, setup, cfg, models)

    # the maps, by the primary method (rotationally symmetric pupils) or the grid transform, at the stated sampling
    assert first_peak is not None
    _maps(focus, setup, request, primary, first_peak, arrays)
    add("electric_irradiance_maps", "|E|^2 and S_z on the transverse planes and the xz / yz sections", "W/um^2", None, values_ref="maps")
    return document(FOCAL_FIELD_ANALYSIS, FOCAL_FIELD_ANALYSIS_VERSION, settings, completed(),
                    engine_input_hash=identity.get("engine_input_hash"), run_result_hash=identity.get("result_hash"),
                    models=models, quantities=_scoped(quantities, setup), arrays=arrays, engine_build_id=engine_build_id)


def _scoped(quantities: list[Quantity], setup: _Setup) -> list[Quantity]:
    """The quantities of a field built from handoff beams a model-scope finding reaches (``setup.scope``): each with a number
    is ``OUT_OF_MODEL_SCOPE`` with the finding's ``ESTIMATE`` (``imaging.outside_model``; every focal quantity reads every
    kind of the beams' light); the others as they are."""

    if not setup.scope:
        return quantities
    from .imaging import outside_model

    marks = scope_marks(setup.scope)
    return [outside_model(q, marks, geometry=frozenset()) for q in quantities]


def _find_peak(focus: Focus, setup: _Setup, method: str, scale: int, centre: tuple[float, float],
               with_window: bool = True) -> tuple[tuple[float, float, float], float]:
    """``(position, |E|^2)`` of the peak near the nominal focus ``centre`` (with ``with_window``, displaced by the window):
    alternating axial and lateral searches."""

    z_centre, z_reach = (setup.z_centre, setup.z_reach) if with_window else (0.0, 0.0)
    x, y, z = centre[0], centre[1], z_centre
    lateral, axial = SEARCH_LATERAL_SCALES * setup.lateral_scale, SEARCH_AXIAL_SCALES * setup.axial_scale + z_reach
    peak = 0.0
    for _ in range(SEARCH_PASSES):
        z, peak = maximize(_profile(focus, method, scale, len("xy"), (x, y, z)), z, axial)
        x, peak = maximize(_profile(focus, method, scale, 0, (x, y, z)), x, lateral)
        y, peak = maximize(_profile(focus, method, scale, 1, (x, y, z)), y, lateral)
        lateral, axial = lateral / SEARCH_NARROWING, axial / SEARCH_NARROWING
    return (x, y, z), peak


def _gaussian_waist(add: Callable[..., None], setup: _Setup, cfg: NumericalConfig, models: list[ModelBasis]) -> None:
    w = setup.gaussian_radius
    objective = setup.objective
    if w is None:
        return
    divergence = w / objective.sphere_radius_mm
    unclipped = objective.pupil_radius_mm >= UNCLIPPED_BEAM_RADII * w
    limit = cfg.gaussian_scope_divergence_rad
    # (the check names the setting that holds its limit where the case is outside it, or the limit is not the Engine default)
    named = {"setting": GAUSSIAN_LIMIT_SETTING} if divergence > limit or limit != DEFAULT_NUMERICAL_CONFIG.gaussian_scope_divergence_rad else {}
    checks = (ValidityCheck(parameter="far_field_half_angle_rad", maximum=limit, observed=_sig(divergence), within=divergence <= limit, **named),
              ValidityCheck(parameter="pupil_radius_over_beam_radius", minimum=UNCLIPPED_BEAM_RADII, observed=_sig(objective.pupil_radius_mm / w),
                            within=unclipped))
    models.append(ModelBasis(model_id=GAUSSIAN_MODEL_ID, name="paraxial Gaussian focus", version=FOCAL_FIELD_ANALYSIS_VERSION,
                             assumptions=("TEM00 beam, paraxial (small far-field angle), not clipped by the pupil",), validity=checks))
    waist = objective.wavelength_mm * objective.focal_length_mm / (math.pi * w)
    if all(c.within for c in checks):
        add("gaussian_waist_radius", "1/e^2 waist radius of the paraxial Gaussian focus, lambda F / (pi w)", "um", waist * MM_TO_UM,
            basis=GAUSSIAN_MODEL_ID, convergence=Convergence(method=ConvergenceMethod.ANALYTIC))
        return
    # Outside the Gaussian model the formula's number is still reported, as the estimate it is: the entry holds the checks
    # that failed (a beam the pupil clips focuses wider than lambda F / (pi w); a wide cone is not paraxial).
    failed = tuple(c.model_copy(update={"code": "NON_GAUSSIAN_FOCUS"}) for c in checks if not c.within)
    add("gaussian_waist_radius", "1/e^2 waist radius of the paraxial Gaussian focus, lambda F / (pi w)", "um", waist * MM_TO_UM,
        state=QuantityState.OUT_OF_MODEL_SCOPE, basis=GAUSSIAN_MODEL_ID, reason="NON_GAUSSIAN_FOCUS",
        convergence=Convergence(method=ConvergenceMethod.ANALYTIC), value_basis=ValueBasis(kind=ValueKind.ESTIMATE, validity=failed),
        message=("the focus is not a paraxial Gaussian beam (" + ", ".join(c.parameter for c in failed) + " outside the Gaussian "
                 "model): the value is the Gaussian formula's; the widths and encircled-energy radii describe this spot"))


def _maps(focus: Focus, setup: _Setup, request: FocalFieldRequestV1, primary: str, peak: tuple[float, float, float], arrays: dict[str, Any]) -> None:
    samples = request.output.samples
    # the maps are centred on the peak to a stated resolution (a share of the spot's scales), so the axes do not carry the
    # search's last digits
    lateral, axial = MAP_CENTRE_RESOLUTION * setup.lateral_scale, MAP_CENTRE_RESOLUTION * setup.axial_scale
    peak = (round(peak[0] / lateral) * lateral + 0.0, round(peak[1] / lateral) * lateral + 0.0, round(peak[2] / axial) * axial + 0.0)
    cx, cy = setup.map_centre if len(setup.spots) > 1 else (peak[0], peak[1])
    x = cx + np.linspace(-setup.half[0], setup.half[0], samples)
    y = cy + np.linspace(-setup.half[1], setup.half[1], samples)
    z_axis = peak[2] + np.linspace(-setup.half[2], setup.half[2], samples)
    planes = [peak[2]] if request.output.z_planes_um is None else [z / MM_TO_UM for z in request.output.z_planes_um]
    components = request.output.include_components and len(setup.modes) == 1
    maps: dict[str, Any] = {"x_um": [_sig(v * MM_TO_UM) for v in x], "y_um": [_sig(v * MM_TO_UM) for v in y], "z_um": [_sig(v * MM_TO_UM) for v in z_axis],
                            "order": "row-major: rows are y (transverse planes) or z (sections), columns x (or y for the yz section)",
                            "method": primary if focus.symmetric else GRID, "planes": []}
    for z in planes:
        if focus.symmetric:
            xx, yy = np.meshgrid(x, y)
            points = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
            fields = [(e.reshape(len("xyz"), samples, samples), h.reshape(len("xyz"), samples, samples)) for e, h in focus.fields(points, primary)]
        else:
            fields = focus.plane(x, y, z)
        electric = np.zeros((samples, samples))
        flux = np.zeros((samples, samples))
        for e, h in fields:
            electric = electric + (np.abs(e) ** 2).sum(axis=0)
            flux = flux + axial_flux(e, h)
        entry: dict[str, Any] = {"z_um": _sig(z * MM_TO_UM), "electric_irradiance_W_per_um2": _map(electric * MM2_TO_UM2),
                                 "axial_flux_W_per_um2": _map(flux * MM2_TO_UM2)}
        if components:
            e = fields[0][0] * math.sqrt(MM2_TO_UM2)
            scale = float(np.abs(e).max())
            entry["field_sqrt_W_per_um2"] = {name: {"re": _scaled(e[i].real, scale), "im": _scaled(e[i].imag, scale)} for i, name in enumerate(("Ex", "Ey", "Ez"))}
        maps["planes"].append(entry)
    for name, axis_values, fixed_axis in (("xz", x, 1), ("yz", y, 0)):
        rows = []
        for z in z_axis:
            line = np.tile(np.array([peak[0], peak[1], z])[:, None], (1, samples))
            line[1 - fixed_axis] = axis_values
            if focus.symmetric:
                rows.append(focus.irradiance(line, primary)[0])
            else:
                rows.append(focus.irradiance(line, GRID)[0])
        maps[f"section_{name}_electric_irradiance_W_per_um2"] = _map(np.array(rows) * MM2_TO_UM2)
    arrays["maps"] = maps


def _scaled(values: np.ndarray, largest: float) -> list[Any]:
    if largest <= 0.0:
        return values.tolist()
    step = largest * MAP_ROUNDING
    return (np.round(values / step) * step).tolist()


__all__ = ["FOCAL_FIELD_ANALYSIS", "FOCAL_FIELD_INVALID", "FocalFieldError", "focal_field", "parse_request"]
