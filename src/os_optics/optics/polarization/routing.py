"""Polarization-resolved routing of a beam line along a known path, without tracing rays.

An assembly presentation of a design (the Experimental Lab Assembly View, its drawing, the Optical Layout) follows each
spectral line of each light source along the design's nominal path -- straight segments between its elements -- and
needs, at every element, the share of the line's power the element sends on along that path and the polarization the
line leaves with. ``PolarizationRouter`` answers both from the Engine's own description of the elements (the lowered
``ProjectSnapshot``) with the Engine's own polarization functions: Fresnel amplitudes (``fresnel``), coating s / p power
coefficients (``CoatingEvaluator``) with a coating's stated phases and the mirror and splitter phase conventions of the
tracer (``phases``: the same amplitude functions the tracer uses), polarizer ``t_par`` / ``t_perp`` and retarder Jones
matrices about their posed axes (``jones``) with the transmittance band holding the wavelength, a Faraday rotator's Jones
matrix about its plane's normal with its response at the wavelength, and a tabulated thin lens's transmittance at the
wavelength. No ray is traced and nothing outside the
element is looked at: the line meets the element's own surfaces in order along the nominal direction, each where the
nominal line crosses its plane (a curved surface: its vertex plane), refracting and reflecting as Snell's law and the law
of reflection say; a thin element (polarizer, retarder, Faraday rotator, thin lens, stop) acts where the line crosses its plane and does
not turn it.

Conventions (``jones.py``): a state is a unit Jones vector ``(j1, j2)`` in a right-handed transverse basis ``(e1, e2)`` of
its direction (``e1 x e2 = d``), field ``E = j1 e1 + j2 e2``, time dependence ``exp(-i omega t)``, global frame. Shares are
fractions of the power arriving at the element.

A step yields the element's 2 x 2 Jones matrix ``M`` (arriving basis -> leaving basis) and from it

* with a known arriving state ``j``: the share ``|M j|^2`` and the leaving state ``M j / |M j|``;
* in every case: the unpolarized share ``tr(M^H M) / 2`` and the range of shares over every arriving polarization (the
  two eigenvalues of ``M^H M``; equal for an element that does not depend on polarization);
* with no known arriving state, the leaving state is still known behind an element that polarizes (it passes one
  polarization at no more than ``POLARIZING_RATIO`` of the other): its dominant output polarization;
* ``best``: the product over the surfaces met of each coated surface's best unpolarized share at any wavelength and
  angle of its table, and of an uncoated interface's own share; ``None`` when a polarizer, retarder, Faraday rotator, thin lens, stop or
  absorber is met (it has no published best share: its share is judged on its own).

``step`` returns ``None`` when the element cannot be followed this way: it has no surfaces, the line meets none of them,
a medium's index, a coating's table or its stated phases do not cover the wavelength, a reflected splitter branch is
asked of an element without a splitting surface, the element is an attenuator (its density depends on where the beam
crosses it), a diffraction grating, or a Faraday rotator whose response does not cover the wavelength, or the line leaves in a direction other than the nominal next one. A coating is evaluated at the nominal
angle of incidence, held inside the angles its table covers.

``StepPassage.notes`` say where the step's share is not backed by the element's characterized data (the Engine's run
reports the same cases out of model scope; the share is the one the run traces with):

* ``COATING_AOI_OUTSIDE_RANGE`` / ``ELEMENT_AOI_OUTSIDE_RANGE``: a coating met beyond the angles its table covers (held
  at the nearest), or -- with the design's characterized ranges (``PolarizationRouter.for_lowered``, ``aoi_checks``) --
  a coated surface or polarization element met outside its ``valid_aoi_deg``, by the rule of the run's validity;
* ``MODEL_WAVELENGTH_OUTSIDE_BAND``: a tabulated thin lens used outside its response table's wavelengths (a Faraday rotator
  answers ``None`` there instead: it states no response to use);
* ``MODEL_POLARIZATION_NOT_VALIDATED``: a tabulated thin lens with validated polarizations reached by another
  polarization, or by a polarization that is not known.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig
from ...contracts.materials import ConstantCoating, TabulatedCoating
from ...contracts.optics import POLARIZATION_MATCH_FRACTION
from ...contracts.project import ProjectSnapshot
from ...errors import EngineInputError
from ...math.exact_trig import acos_deg
from ...math.vectors import dot, normalize, parallel_transport_basis, transverse_basis
from ...contracts.optics import MeshSurface
from ...scene.lower import facet_patches, lower_scene
from ...scene.model import SurfaceG
from .. import apertures
from ..materials.coatings import CoatingEvaluator
from ..materials.library import MaterialLibrary
from ..surfaces.interaction import reflect_dir, refract
from ..surfaces.tabulated_lens import orthogonal_fractions, response_at
from .fresnel import fresnel
from .jones import apply_interface, apply_thin_transmission, faraday_rotation, polarizer, project_axis, retarder, s_axis
from .phases import PhaseValues, coated_interface, mirror_reflection, splitter_amplitudes

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
Vec3 = tuple[float, float, float]

#: Follow the element as the light meets it (transmit at a splitting surface, reflect at a mirror, refract or totally
#: reflect at an interface).
ROUTE_PASS = "PASS"
#: As ``ROUTE_PASS``, but take the reflected branch at the first splitting surface met (a plate beamsplitter's or a
#: dichroic's coated face, a cube's diagonal).
ROUTE_SPLIT_REFLECT = "SPLIT_REFLECT"
ROUTES = (ROUTE_PASS, ROUTE_SPLIT_REFLECT)
#: An element that passes one polarization at no more than this fraction of the other emits a known polarization (its
#: dominant output polarization, to within this fraction of the power) whatever arrives: a polarizer, a polarizing
#: splitter's transmitted port (extinction ratios of 100:1 and more).
POLARIZING_RATIO = 0.01
#: The leaving direction agrees with the nominal next direction when their cosine is at least this (about 2.6 deg).
DIRECTION_COSINE = 0.999
#: Interactions ``_interact`` applies to a line, and those whose element it cannot follow (a neutral-density attenuator:
#: its density depends on where the beam crosses it; a diffraction grating: a line leaves it in several directions; a
#: complex-transmission sample: its phase turns the line by where it crosses). Every lowered kind but ``observe`` (planes are not routed through)
#: is in exactly one of them; a design with another kind is refused (``KeyError``).
FOLLOWED_INTERACTIONS = frozenset({"refract", "mirror", "beam_splitter", "polarizer", "retarder", "faraday_rotator", "thin_lens", "aperture_stop", "absorber"})
NOT_FOLLOWED_INTERACTIONS = frozenset({"neutral_density", "diffraction_grating", "complex_transmission"})
#: Interactions of a thin element: no published best share, the line keeps its direction.
_THIN = frozenset({"polarizer", "retarder", "faraday_rotator", "thin_lens", "aperture_stop", "absorber"})
#: How far beyond a surface's clear aperture and mount (mm) the nominal line may still be taken to cross it.
_MISS_MM = 1.0
#: The line runs along a surface's plane (does not cross it) when its direction cosine with the normal is below this.
_ALONG_PLANE_COSINE = 1e-9
#: A surface this close behind the current point (mm) is not ahead of the line.
_AHEAD_MM = 1e-9
#: Shares over two polarizations differing by less than this fraction do not depend on the polarization.
_SAME_SHARE_REL = 1e-12
#: A coating's own angle range (``max_aoi_deg``, a table's angles) is exceeded beyond this (deg).
_AOI_RANGE_TOLERANCE_DEG = 1e-9
#: Step notes (module docstring); the codes of the run's own out-of-scope issues.
COATING_AOI_OUTSIDE_RANGE = "COATING_AOI_OUTSIDE_RANGE"
ELEMENT_AOI_OUTSIDE_RANGE = "ELEMENT_AOI_OUTSIDE_RANGE"
MODEL_WAVELENGTH_OUTSIDE_BAND = "MODEL_WAVELENGTH_OUTSIDE_BAND"
MODEL_POLARIZATION_NOT_VALIDATED = "MODEL_POLARIZATION_NOT_VALIDATED"
#: ``check(aoi_deg, refractive_index) -> out-of-range code | None`` of one surface (``local.validity.surface_aoi_checks``).
AoiCheck = Callable[[float, float], "str | None"]


def _scene_interactions(owner: str, surface) -> tuple[tuple[str, object], ...]:
    """The scene id and contract interaction of each surface a declared surface lowers to (a facet surface: its patches)."""
    if isinstance(surface, MeshSurface):
        return facet_patches(owner, surface)
    return ((f"{owner}/{surface.surface_id}", surface.interaction),)


@dataclass(frozen=True)
class LineState:
    """The polarization of a beam line at a point of its path: the unit Jones vector ``jones`` in the right-handed
    transverse basis ``(e1, e2)`` of ``direction`` (global frame, ``e1 x e2 = direction``)."""

    direction: Vec3
    e1: Vec3
    e2: Vec3
    jones: tuple[complex, complex]

    def field(self) -> tuple[complex, complex, complex]:
        """The complex field vector ``j1 e1 + j2 e2``."""

        j1, j2 = self.jones
        return tuple(j1 * a + j2 * b for a, b in zip(self.e1, self.e2, strict=True))  # type: ignore[return-value]


@dataclass(frozen=True)
class StepPassage:
    """What one element does to a beam line along one step of its path (module docstring).

    ``share``: the share of the arriving power sent on, for the known arriving state (else the unpolarized share);
    ``unpolarized_share``, ``share_range`` (min, max over every arriving polarization), ``best``; ``state``: the leaving
    polarization (``None``: not known); ``matrix``: the Jones matrix, rows of the leaving basis, columns of the arriving
    one; ``surfaces``: the Engine surface ids met, in order; ``notes``: where the share is not backed by the element's
    characterized data (module docstring), sorted."""

    share: float
    unpolarized_share: float
    share_range: tuple[float, float]
    best: float | None
    state: LineState | None
    arriving_known: bool
    matrix: tuple[tuple[complex, complex], tuple[complex, complex]]
    surfaces: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether the share is a polarization-resolved value inside the element's characterized data (no notes)."""

        return not self.notes

    @property
    def polarization_dependent(self) -> bool:
        """Whether the share depends on the arriving polarization (the two extremes differ)."""

        lo, hi = self.share_range
        return hi - lo > _SAME_SHARE_REL * max(hi, 1.0)


def _unit(v: Sequence[float]) -> Array | None:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.sqrt(dot(a, a)))
    return None if n == 0.0 or not math.isfinite(n) else a / n


def _tuple(v: Array) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _coating_peak(model, reflected: bool) -> float:
    """The best unpolarized power reflectance / transmittance of an Engine coating at any node of its data."""

    if isinstance(model, ConstantCoating):
        s, p = (model.reflectance_s, model.reflectance_p) if reflected else (model.transmittance_s, model.transmittance_p)
        return 0.5 * (s + p)
    assert isinstance(model, TabulatedCoating)
    s_rows, p_rows = (model.reflectance_s, model.reflectance_p) if reflected else (model.transmittance_s, model.transmittance_p)
    return max(0.5 * (s + p) for s_row, p_row in zip(s_rows, p_rows, strict=True) for s, p in zip(s_row, p_row, strict=True))


def _reach(surface: SurfaceG) -> float:
    """How far from the surface's reference point (in its plane) the nominal line may cross it."""

    if surface.geometry == "sphere":
        return surface.aperture_radius + surface.mount_margin
    ap = surface.outline if surface.outline is not None else surface.aperture
    if not apertures.bounded(ap):
        return math.inf
    if surface.cylindrical:
        reach = math.inf  # straight along v: no bound across the cylinder axis
    else:
        reach = apertures.farthest(ap, ap.offset_u, ap.offset_v)  # (from the reference point, the frame's origin)
    return reach + surface.mount_margin


def _plane(surface: SurfaceG) -> tuple[Array, Array]:
    """A point and the unit normal of the plane the nominal line meets the surface in (a sphere: its vertex plane)."""

    if surface.geometry == "sphere":
        return surface.center + surface.radius * surface.cap_axis, surface.cap_axis
    return surface.origin, surface.normal


def _start_distance(surfaces: list[SurfaceG], here: Array, d: Array) -> float:
    """How far before ``here`` (along ``d``) the walk starts: beyond every surface's reference point and every point where
    the nominal line crosses a surface's plane (a tilted plate's far face can cross the line well before its centre)."""

    back = 0.0
    for surface in surfaces:
        origin, normal = _plane(surface)
        back = max(back, math.sqrt(float(dot(origin - here, origin - here))))
        dn = float(dot(d, normal))
        if abs(dn) >= _ALONG_PLANE_COSINE:
            back = max(back, -float(dot((origin - here), normal)) / dn)
    return back + 1.0


class _Walk:
    """The two basis states of one step, carried through the element together (rows)."""

    def __init__(self, d: Array, e1: Array, e2: Array, arriving: tuple[complex, complex] | None = None) -> None:
        self.arriving = arriving  # the arriving Jones vector in the step's first basis (None: not known)
        self.notes: set[str] = set()
        self.d = np.tile(d, (2, 1))
        self.e1 = np.tile(e1, (2, 1))
        self.e2 = np.tile(e2, (2, 1))
        self.jones: CArray = np.eye(2, dtype=np.complex128)
        self.best: float | None = 1.0
        self.reflected = False
        self.medium: str | None = None
        self.met: list[str] = []

    def factor(self, value: float | None) -> None:
        self.best = None if value is None or self.best is None else self.best * value


class PolarizationRouter:
    """Routes beam lines through the elements of one lowered design (``ProjectSnapshot``: the Engine request's project,
    as ``os_optics.local.lower_project`` builds it). Deterministic and side-effect free; build once per design."""

    def __init__(self, project: ProjectSnapshot, config: NumericalConfig = DEFAULT_NUMERICAL_CONFIG, *,
                 aoi_checks: Mapping[str, AoiCheck] | None = None) -> None:
        self._cfg = config
        self._aoi_checks = dict(aoi_checks or {})
        self._library = MaterialLibrary(project.materials, config)
        scene, _ = lower_scene(project, config, self._library)
        self._surfaces: dict[str, list[SurfaceG]] = {}
        for surface in scene.surfaces:
            if surface.interaction.kind != "observe":
                if surface.interaction.kind not in FOLLOWED_INTERACTIONS | NOT_FOLLOWED_INTERACTIONS:
                    raise KeyError(f"interaction kind {surface.interaction.kind!r} at {surface.surface_id} is neither followed nor not followed")
                self._surfaces.setdefault(surface.owner_id, []).append(surface)
        self._sources = {source.owner_id: source for source in scene.sources}
        self._coatings = {sid: inter.coating for component in project.components for surface in component.surfaces
                          for sid, inter in _scene_interactions(component.instance_id, surface)
                          if getattr(inter, "coating", None) is not None}

    @classmethod
    def for_lowered(cls, lowered, config: NumericalConfig = DEFAULT_NUMERICAL_CONFIG) -> PolarizationRouter:
        """The router of a lowered Project (``os_optics.local.lower_project``), with the angle ranges its coated
        surfaces and polarization elements are characterized for (``aoi_checks``)."""

        from ...local.validity import surface_aoi_checks

        return cls(lowered.request.project, config, aoi_checks=surface_aoi_checks(lowered.elements))

    def source_state(self, source_id: str) -> LineState | None:
        """The polarization a light source emits (its Jones vector about its emission direction), or ``None`` (no such
        source, or partially polarized light: two incoherent states, not one)."""

        source = self._sources.get(source_id)
        if source is None or source.degree_of_polarization is not None:
            # (partially polarized light is no single Jones state: the unpolarized share and the share range decide)
            return None
        jones = np.asarray(source.jones, dtype=np.complex128)
        norm = float(np.sqrt(np.real(np.vdot(jones, jones))))
        if norm == 0.0:
            return None
        return LineState(_tuple(source.direction), _tuple(source.e1), _tuple(source.e2), (complex(jones[0] / norm), complex(jones[1] / norm)))

    def elements(self) -> tuple[str, ...]:
        """The ids of the elements with surfaces, sorted."""

        return tuple(sorted(self._surfaces))

    # ------------------------------------------------------------------------------------------------------- step
    def step(self, element_id: str, route: str, wavelength_nm: float, point: Sequence[float], direction_in: Sequence[float],
             direction_out: Sequence[float] | None = None, state: LineState | None = None) -> StepPassage | None:
        """What ``element_id`` does to a line of ``wavelength_nm`` arriving along ``direction_in`` on the nominal line
        through ``point`` (global frame, mm), leaving toward ``direction_out`` (``None``: the path ends here), with the
        arriving polarization ``state`` (``None``: not known). ``route``: ``ROUTE_PASS`` or ``ROUTE_SPLIT_REFLECT``."""

        if route not in ROUTES:
            raise ValueError(f"unknown route {route!r}; known: {', '.join(ROUTES)}")
        surfaces = self._surfaces.get(element_id)
        d = _unit(direction_in)
        if not surfaces or d is None:
            return None
        tol = self._cfg.zero_vector_tol
        if state is not None:
            s_d, s_e1, s_e2 = (np.asarray(v, dtype=np.float64)[None, :] for v in (state.direction, state.e1, state.e2))
            e1, e2 = parallel_transport_basis(s_e1, s_e2, s_d, d[None, :], self._cfg.antiparallel_tol)
            e1, e2 = e1[0], e2[0]
        else:
            b1, b2 = transverse_basis(d[None, :], tol)
            e1, e2 = b1[0], b2[0]
        walk = _Walk(d, e1, e2, None if state is None else (complex(state.jones[0]), complex(state.jones[1])))
        here = np.asarray(point, dtype=np.float64)
        position = here - _start_distance(surfaces, here, d) * d
        used: set[str] = set()
        for _ in range(2 * len(surfaces) + 2):
            hit = self._next(surfaces, used, position, walk)
            if hit is None:
                break
            surface, at = hit
            used.add(surface.surface_id)
            walk.met.append(surface.surface_id)
            if not self._interact(surface, walk, wavelength_nm, route):
                return None
            position = at
            if not np.any(walk.jones):
                break  # absorbed (an absorber, a mirror's non-reflective side)
        if not walk.met or (route == ROUTE_SPLIT_REFLECT and not walk.reflected):
            return None
        d_last, e1_out, e2_out = walk.d[:1], walk.e1[:1], walk.e2[:1]
        if direction_out is not None:
            d_out = _unit(direction_out)
            if d_out is None or float(dot(d_last[0], d_out)) < DIRECTION_COSINE:
                return None
            e1_out, e2_out = parallel_transport_basis(e1_out, e2_out, d_last, d_out[None, :], self._cfg.antiparallel_tol)
            d_last = d_out[None, :]
        return self._passage(walk, state, d_last[0], e1_out[0], e2_out[0])

    def _next(self, surfaces: list[SurfaceG], used: set[str], position: Array, walk: _Walk) -> tuple[SurfaceG, Array] | None:
        d = walk.d[0]
        best: tuple[float, int, SurfaceG, Array] | None = None
        for order, surface in enumerate(surfaces):
            if surface.surface_id in used:
                continue
            origin, normal = _plane(surface)
            dn = float(dot(d, normal))
            if abs(dn) < _ALONG_PLANE_COSINE:
                continue  # the line runs along the surface's plane: it does not cross it
            t = float(dot((origin - position), normal)) / dn
            if t <= _AHEAD_MM:
                continue
            at = position + t * d
            offset = at - origin
            offset = offset - float(dot(offset, normal)) * normal
            if float(np.sqrt(dot(offset, offset))) > _reach(surface) + _MISS_MM:
                continue
            kind = surface.interaction.kind
            if kind in ("refract", "beam_splitter") and walk.medium is not None:
                arriving = surface.medium_pos if dn < 0.0 else surface.medium_neg
                if arriving != walk.medium:
                    continue  # not reachable from the medium the line is in
            if best is None or (t, order) < (best[0], best[1]):
                best = (t, order, surface, at)
        return None if best is None else (best[2], best[3])

    def _coefficients(self, surface: SurfaceG, wavelength_nm: float, cos_i: Array, walk: _Walk,
                      refractive_index: float | None) -> tuple[Array, Array, Array, Array] | None:
        """The coating's (rs, rp, ts, tp) at the nominal angle, held inside the angles its table covers (noted, as is an
        angle outside the surface's characterized range); ``None`` outside its wavelength range."""

        model = self._coatings.get(surface.surface_id)
        evaluator: CoatingEvaluator = surface.interaction.coating  # type: ignore[assignment]
        aoi = acos_deg(cos_i)
        nominal = float(aoi[0])
        self._check_aoi(surface, nominal, refractive_index, walk, wavelength_nm)
        if isinstance(model, ConstantCoating):
            if nominal > model.max_aoi_deg + _AOI_RANGE_TOLERANCE_DEG:
                walk.notes.add(COATING_AOI_OUTSIDE_RANGE)
            aoi = np.minimum(aoi, model.max_aoi_deg)
        elif isinstance(model, TabulatedCoating):
            if not model.aoi_deg[0] - _AOI_RANGE_TOLERANCE_DEG <= nominal <= model.aoi_deg[-1] + _AOI_RANGE_TOLERANCE_DEG:
                walk.notes.add(COATING_AOI_OUTSIDE_RANGE)
            aoi = np.clip(aoi, model.aoi_deg[0], model.aoi_deg[-1])
        if not bool(np.all(evaluator.in_range(wavelength_nm, aoi))):
            return None
        cc = evaluator.coefficients(wavelength_nm, aoi, location=f"surface:{surface.surface_id}")
        return cc.rs, cc.rp, cc.ts, cc.tp

    def _check_aoi(self, surface: SurfaceG, aoi_deg: float, refractive_index: float | None, walk: _Walk,
                   wavelength_nm: float | None = None) -> None:
        """Note an angle outside the surface's characterized range (``aoi_checks``; the incident medium's absolute index at
        the vacuum wavelength for a range stated as an equivalent angle in the environment's air; no index known: that air)."""

        check = self._aoi_checks.get(surface.surface_id)
        if check is None:
            return
        code = check(aoi_deg, 1.0, None) if refractive_index is None else check(aoi_deg, refractive_index, wavelength_nm)
        if code is not None:
            walk.notes.add(code)

    def _index(self, medium: str | None, wavelength_nm: float) -> float | None:
        if medium is None:
            return None
        try:
            n, _ = self._library.evaluator(medium).index(wavelength_nm)
        except EngineInputError:
            return None
        return n

    def _phases(self, surface: SurfaceG, wavelength_nm: float) -> PhaseValues | None | bool:
        """The surface's stated phases at ``wavelength_nm`` (``None``: it states none; ``False``: outside their table)."""

        evaluator = surface.interaction.phases
        if evaluator is None:
            return None
        values, issue = evaluator.at(wavelength_nm, location=f"surface:{surface.surface_id}")
        return False if issue is not None else values

    def _peak(self, surface: SurfaceG, reflected: bool) -> float | None:
        model = self._coatings.get(surface.surface_id)
        return None if model is None else _coating_peak(model, reflected)

    def _interact(self, surface: SurfaceG, walk: _Walk, wavelength_nm: float, route: str) -> bool:
        """Apply one surface to both rows; ``False``: the element cannot be followed (module docstring)."""

        cfg = self._cfg
        inter = surface.interaction
        kind = inter.kind
        normal = np.tile(surface.normal if surface.geometry != "sphere" else surface.cap_axis, (2, 1))
        d = walk.d
        if kind == "refract":
            from_pos = float(dot(d[0], normal[0])) < 0.0
            med_from, med_to = (surface.medium_pos, surface.medium_neg) if from_pos else (surface.medium_neg, surface.medium_pos)
            try:
                n1, _ = self._library.evaluator(med_from).index(wavelength_nm)
                n2, _ = self._library.evaluator(med_to).index(wavelength_nm)
            except EngineInputError:
                return False  # a medium the design's materials do not define
            if self._library.uniaxial(med_from) is not None or self._library.uniaxial(med_to) is not None:
                return False  # a crystal face splits the line into two waves of their own directions: not one line
            if n1 is None or n2 is None:
                return False
            res = refract(d, normal, n1, n2, cfg.zero_vector_tol)
            tir = bool(res.tir[0])
            fr = fresnel(n1, n2, np.maximum(res.cos_i, np.finfo(float).tiny))
            tau_s, tau_p, r_s, r_p = fr.tau_s, fr.tau_p, fr.r_s, fr.r_p
            if inter.coating is not None and not tir:
                cc = self._coefficients(surface, wavelength_nm, res.cos_i, walk, n1)
                if cc is None:
                    return False
                ph = self._phases(surface, wavelength_nm)
                if ph is False:
                    return False
                rs, rp, ts, tp = cc
                tau_s, tau_p, r_s, r_p = coated_interface(rs, rp, ts, tp, tau_s, tau_p, r_s, r_p, ph)
            s = s_axis(d, res.n_oriented, walk.e1, cfg.zero_vector_tol)
            reflect_here = not tir and route == ROUTE_SPLIT_REFLECT and inter.trace_reflection and not walk.reflected
            if tir or reflect_here:
                d_out, _, _ = reflect_dir(d, normal, cfg.zero_vector_tol)
                coef_s, coef_p = r_s, r_p
                walk.reflected = walk.reflected or reflect_here
                peak = 1.0 if tir else (self._peak(surface, True) if inter.coating is not None else
                                        0.5 * float(abs(r_s[0]) ** 2 + abs(r_p[0]) ** 2))
            else:
                d_out = res.d_out
                coef_s, coef_p = tau_s, tau_p
                walk.medium = med_to
                peak = self._peak(surface, False) if inter.coating is not None else 0.5 * float(abs(tau_s[0]) ** 2 + abs(tau_p[0]) ** 2)
            if walk.medium is None:
                walk.medium = med_from
            walk.jones, walk.e1, walk.e2 = apply_interface(walk.jones, walk.e1, walk.e2, d, s, coef_s, coef_p, d_out)
            walk.d = d_out
            walk.factor(peak)
            return True
        if kind == "mirror":
            from_pos = float(dot(d[0], normal[0])) < 0.0
            if not (inter.reflective_side == "both" or (inter.reflective_side == "positive") == from_pos):
                walk.jones = np.zeros_like(walk.jones)  # the non-reflective side: blocked
                walk.factor(None)
                return True
            d_out, n_or, cos_i = reflect_dir(d, normal, cfg.zero_vector_tol)
            if inter.coating is not None:
                medium = surface.medium_pos if from_pos else surface.medium_neg
                cc = self._coefficients(surface, wavelength_nm, cos_i, walk, self._index(medium, wavelength_nm))
                if cc is None:
                    return False
                rs, rp = np.sqrt(cc[0]), np.sqrt(cc[1])
                peak = self._peak(surface, True)
            else:
                rs = rp = np.ones(2)
                peak = 1.0
            ph = self._phases(surface, wavelength_nm)
            if ph is False:
                return False
            coef_s, coef_p = mirror_reflection(rs, rp, inter.phase_s, inter.phase_p, ph)
            s = s_axis(d, n_or, walk.e1, cfg.zero_vector_tol)
            walk.jones, walk.e1, walk.e2 = apply_interface(walk.jones, walk.e1, walk.e2, d, s, coef_s, coef_p, d_out)
            walk.d = d_out
            walk.reflected = True
            walk.factor(peak)
            return True
        if kind == "beam_splitter":
            d_ref, n_or, cos_i = reflect_dir(d, normal, cfg.zero_vector_tol)
            medium = surface.medium_pos if float(dot(d[0], normal[0])) < 0.0 else surface.medium_neg
            cc = self._coefficients(surface, wavelength_nm, cos_i, walk, self._index(medium, wavelength_nm))
            if cc is None:
                return False
            ph = self._phases(surface, wavelength_nm)
            if ph is False:
                return False
            # the tracer's splitter: r_p = -r_s in the (s, d_out x s) triad, reflection_phase relative to t, or the stated
            # phases
            t_s, t_p, r_s, r_p = splitter_amplitudes(*cc, inter.reflection_phase, inter.transmission_phase, ph)
            s = s_axis(d, n_or, walk.e1, cfg.zero_vector_tol)
            if route == ROUTE_SPLIT_REFLECT and not walk.reflected:
                walk.jones, walk.e1, walk.e2 = apply_interface(walk.jones, walk.e1, walk.e2, d, s, r_s, r_p, d_ref)
                walk.d = d_ref
                walk.reflected = True
                walk.factor(self._peak(surface, True))
            else:
                walk.jones = apply_thin_transmission(walk.jones, walk.e1, walk.e2, d, s, t_s, t_p)
                walk.factor(self._peak(surface, False))
            return True
        if kind in ("polarizer", "retarder"):
            assert inter.axis is not None
            a1, a2, ok = project_axis(inter.axis, d, walk.e1, walk.e2, cfg.zero_vector_tol)
            if not bool(ok[0]):
                return False
            if surface.surface_id in self._aoi_checks:
                cos_axis = min(1.0, abs(float(dot(d[0], normal[0]))))
                self._check_aoi(surface, math.degrees(math.acos(cos_axis)), self._index(walk.medium, wavelength_nm), walk, wavelength_nm)
            t, t_perp = inter.transmittances(wavelength_nm)  # the band holding the wavelength, else the scalar values
            if kind == "polarizer":
                walk.jones = polarizer(walk.jones, a1, a2, t, t_perp)
            else:
                delta = inter.retardance if inter.achromatic else inter.retardance * inter.design_wavelength_nm / wavelength_nm
                walk.jones = retarder(walk.jones, a1, a2, delta) * math.sqrt(t)
            walk.factor(None)
            return True
        if kind == "faraday_rotator":
            assert inter.faraday is not None
            response = inter.faraday.at(wavelength_nm)
            if response is None:
                return False
            if surface.surface_id in self._aoi_checks:
                cos_axis = min(1.0, abs(float(dot(d[0], normal[0]))))
                self._check_aoi(surface, math.degrees(math.acos(cos_axis)), self._index(walk.medium, wavelength_nm), walk, wavelength_nm)
            walk.jones = faraday_rotation(walk.jones, dot(d, normal), response[0]) * math.sqrt(response[1])
            walk.factor(None)
            return True
        if kind in ("thin_lens", "aperture_stop"):
            # a tabulated thin lens: its transmittance at this wavelength (the tracer's response_at)
            transmittance = inter.transmittance
            if inter.table is not None:
                _focal, transmittance, in_domain = response_at(inter.table, wavelength_nm)
                if not in_domain:
                    walk.notes.add(MODEL_WAVELENGTH_OUTSIDE_BAND)
                if not self._validated_polarization(inter.table, walk):
                    walk.notes.add(MODEL_POLARIZATION_NOT_VALIDATED)
            walk.jones = walk.jones * math.sqrt(transmittance)
            walk.factor(None)
            return True
        if kind == "absorber":
            walk.jones = np.zeros_like(walk.jones)
            walk.factor(None)
            return True
        if kind in NOT_FOLLOWED_INTERACTIONS:
            return False
        raise KeyError(f"interaction kind {kind!r} has no routing rule")

    @staticmethod
    def _validated_polarization(table, walk: _Walk) -> bool:
        """Whether the line reaches a tabulated thin lens in a polarization its response is validated for (the tracer's
        rule: the power in the orthogonal linear polarization at most sin^2 of 1 deg); an unknown polarization is not."""

        if table.validated_polarizations is None or table.u_axis is None or table.v_axis is None:
            return True
        if walk.arriving is None:
            return False
        jones = walk.arriving[0] * walk.jones[0] + walk.arriving[1] * walk.jones[1]
        fractions = orthogonal_fractions(jones, walk.e1[0], walk.e2[0], walk.d[0], table.u_axis, table.v_axis)
        return any(fractions[name] <= POLARIZATION_MATCH_FRACTION for name in table.validated_polarizations)

    def _passage(self, walk: _Walk, state: LineState | None, d: Array, e1: Array, e2: Array) -> StepPassage:
        m = walk.jones.T.copy()  # column k: the leaving Jones vector of arriving basis state k
        h = m.conj().T @ m
        a, c = float(np.real(h[0, 0])), float(np.real(h[1, 1]))
        b = complex(h[0, 1])
        mean = 0.5 * (a + c)
        spread = math.sqrt(max(0.25 * (a - c) ** 2 + abs(b) ** 2, 0.0))
        lo, hi = max(mean - spread, 0.0), mean + spread
        leaving: CArray | None = None
        share = mean
        if state is not None:
            out = m @ np.asarray(state.jones, dtype=np.complex128)
            share = float(np.real(np.vdot(out, out)))
            leaving = out if share > 0.0 else None
        elif hi > 0.0 and lo <= POLARIZING_RATIO * hi:
            # an element that polarizes: the arriving polarization it passes best, and what it becomes
            v = np.array([b, hi - a], dtype=np.complex128) if abs(b) > 0.0 else (
                np.array([1.0, 0.0], dtype=np.complex128) if a >= c else np.array([0.0, 1.0], dtype=np.complex128))
            leaving = m @ (v / np.sqrt(np.real(np.vdot(v, v))))
        out_state = None
        if leaving is not None:
            norm = float(np.sqrt(np.real(np.vdot(leaving, leaving))))
            if norm > 0.0:
                j = leaving / norm
                out_state = LineState(_tuple(d), _tuple(e1), _tuple(e2), (complex(j[0]), complex(j[1])))
        matrix = ((complex(m[0, 0]), complex(m[0, 1])), (complex(m[1, 0]), complex(m[1, 1])))
        return StepPassage(share=share, unpolarized_share=mean, share_range=(lo, hi), best=walk.best, state=out_state,
                           arriving_known=state is not None, matrix=matrix, surfaces=tuple(walk.met), notes=tuple(sorted(walk.notes)))


__all__ = [
    "COATING_AOI_OUTSIDE_RANGE",
    "DIRECTION_COSINE",
    "ELEMENT_AOI_OUTSIDE_RANGE",
    "MODEL_POLARIZATION_NOT_VALIDATED",
    "MODEL_WAVELENGTH_OUTSIDE_BAND",
    "POLARIZING_RATIO",
    "ROUTES",
    "ROUTE_PASS",
    "ROUTE_SPLIT_REFLECT",
    "LineState",
    "PolarizationRouter",
    "StepPassage",
]
