"""The partially polarized light on each route of an observation plane (``ObservationResult.polarization_mixtures``).

A partially polarized source's line is traced as two mutually incoherent fully polarized components
(``optics/polarization/partial.py``). On a plane, the arrivals of one line that came along the same interactions
(``ArrivalResult.lineage``) at the same crossing are one route's light; their Stokes parameters add (no interference term
between the components):

    S = sum_k P_k (1, (|E_x|^2 - |E_y|^2) / n_k, 2 Re(conj(E_x) E_y) / n_k, 2 Im(conj(E_x) E_y) / n_k),  n_k = |E_x|^2 + |E_y|^2

in the transverse basis of the route's first arrival (an arrival in another basis is re-expressed in it through its 3D
field; both components cross the same surfaces, so their directions agree), with ``P_k`` the arrival's power. The degree of
polarization is ``sqrt(S1^2 + S2^2 + S3^2) / S0`` and the polarized part's ellipse follows from the unit Stokes vector
``(S1, S2, S3) / |.|`` (a Jones vector built from it with square roots only, then the beam ellipse of
``observations.jones_summary``).

A component missing on a route either did not reach the plane there (an element stopped it, or what was left of it fell
below the trace's negligible level, which bounds its contribution by that level) or was not traced
because the path budget ran out: a route whose line booked power to ``PATH_LIMIT`` or ``SOURCE_NOT_TRACED`` and lacks a
component is ``UNKNOWN`` (``INCOMPLETE_TRACE``). A component whose power or polarization is not known (an order of a
diffraction grating without a stated response) makes the route ``UNKNOWN`` too; one whose power is out of model scope
(``ArrivalResult.power_out_of_scope_at``: its power is its chief ray's, not the light's) makes it ``OUT_OF_MODEL_SCOPE``, with
no Stokes value. Arithmetic: ``+ - x /`` and ``sqrt`` on
Python floats (separately rounded products; identical bits in every runtime).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from ..config import NumericalConfig
from ..contracts.results import ArrivalResult, ObservationResult, PolarizationMixture, PowerBudget
from ..errors import ErrorCode
from ..errors.status import QuantityState
from ..optics.rays.tracer import SINK_PATH_LIMIT, SINK_SOURCE_NOT_TRACED
from .observations import jones_summary

if TYPE_CHECKING:
    from .metrics import MetricValue

#: Ledger sinks of a line that mean part of its light was not launched or not followed by the path budget.
UNTRACED_COMPONENT_SINKS = (SINK_PATH_LIMIT, SINK_SOURCE_NOT_TRACED)

Key = tuple[str, int, int, tuple[str, ...]]
Stokes = tuple[float, float, float, float]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _jones_in(a: ArrivalResult, first: ArrivalResult) -> tuple[complex, complex]:
    """The arrival's Jones vector re-expressed in the transverse basis of ``first`` (the same basis: unchanged)."""
    pol = a.beam.polarization
    assert pol is not None
    ex, ey = complex(*pol.ex), complex(*pol.ey)
    b, f = a.beam.basis, first.beam.basis
    if b.e1 == f.e1 and b.e2 == f.e2:
        return ex, ey
    # field = ex e1 + ey e2 (real basis vectors): its components along the first arrival's e1, e2
    c11, c21 = _dot(b.e1, f.e1), _dot(b.e2, f.e1)
    c12, c22 = _dot(b.e1, f.e2), _dot(b.e2, f.e2)
    return (complex(ex.real * c11 + ey.real * c21, ex.imag * c11 + ey.imag * c21),
            complex(ex.real * c12 + ey.real * c22, ex.imag * c12 + ey.imag * c22))


def _stokes(power_W: float, ex: complex, ey: complex) -> Stokes:
    """``P (1, s1, s2, s3)`` of a beam of power ``power_W`` and Jones vector ``(ex, ey)`` (any norm)."""
    ix, iy = ex.real * ex.real + ex.imag * ex.imag, ey.real * ey.real + ey.imag * ey.imag
    n = ix + iy
    if power_W == 0.0 or n == 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    re = ex.real * ey.real + ex.imag * ey.imag  # Re(conj(ex) ey)
    im = ex.real * ey.imag - ex.imag * ey.real  # Im(conj(ex) ey)
    return (power_W, power_W * ((ix - iy) / n), power_W * (2.0 * re / n), power_W * (2.0 * im / n))


def polarized_jones(p1: float, p2: float, p3: float) -> tuple[complex, complex]:
    """A unit Jones vector of the normalized Stokes vector ``(p1, p2, p3)`` (``p1^2 + p2^2 + p3^2 = 1``): square roots only."""
    if p1 >= 0.0:
        a = math.sqrt((1.0 + p1) / 2.0)
        return complex(a, 0.0), complex(p2 / (2.0 * a), p3 / (2.0 * a))
    b = math.sqrt((1.0 - p1) / 2.0)
    return complex(p2 / (2.0 * b), -p3 / (2.0 * b)), complex(b, 0.0)


def _same_direction(a: ArrivalResult, b: ArrivalResult, cfg: NumericalConfig) -> bool:
    da, db = a.beam.direction, b.beam.direction
    return abs(_dot(da, db) - 1.0) <= cfg.basis_orthonormality_tol


def _mixture(arrivals: list[ArrivalResult], observation: ObservationResult, untraced_lines: set[tuple[str, int]],
             cfg: NumericalConfig) -> PolarizationMixture:
    first = arrivals[0]
    base = {"source_id": first.beam.source_id, "spectral_index": first.beam.spectral_index,
            "wavelength_nm": first.beam.wavelength_nm, "crossing_index": first.crossing_index,
            "path_ids": tuple(a.beam.path_id for a in arrivals)}
    scope = sorted({at for a in arrivals for at in (*a.power_out_of_scope_at, *a.power_taken_before_at)})
    if scope:
        # (a component's power is its chief ray's, not the light's: no Stokes sum of it -- ArrivalResult.power_out_of_scope_at,
        # or a surface took a share of it before the plane, power_taken_before_at: the shares the components lost are not known)
        return PolarizationMixture(
            **base, quantity_state=QuantityState.OUT_OF_MODEL_SCOPE, issue_codes=(ErrorCode.OUT_OF_MODEL_SCOPE.value,),
            power_out_of_scope_at=tuple(scope),
            message=f"the power of a polarization component on this route is out of model scope behind {', '.join(scope)} "
            "(power_out_of_scope_at): it is its chief ray's power, not the light's, so the Stokes parameters and the degree of "
            "polarization of the light are not stated")
    if any(a.beam.power_W is None or a.beam.polarization is None for a in arrivals):
        return PolarizationMixture(
            **base, quantity_state=QuantityState.UNKNOWN, issue_codes=(ErrorCode.GRATING_RESPONSE_UNAVAILABLE.value,),
            message="a polarization component on this route left a diffraction grating with a power or polarization that "
            "is not known: the Stokes parameters of the light are not known")
    if not all(_same_direction(a, first, cfg) for a in arrivals[1:]):
        return PolarizationMixture(
            **base, quantity_state=QuantityState.UNKNOWN, issue_codes=(ErrorCode.NUMERICAL_FAILURE.value,),
            message="the polarization components on this route arrive in different directions: their Stokes parameters do "
            "not refer to one beam")
    components = {a.polarization_component for a in arrivals}
    if len(components) < 2 and (first.beam.source_id, first.beam.spectral_index) in untraced_lines:
        return PolarizationMixture(
            **base, quantity_state=QuantityState.UNKNOWN, issue_codes=(ErrorCode.INCOMPLETE_TRACE.value,),
            message="only one polarization component of this line arrives on this route, and the path budget left part of "
            "the line's light untraced: the other component may arrive here too; raise compute_budget.max_paths")
    total = [0.0, 0.0, 0.0, 0.0]
    for a in arrivals:
        ex, ey = _jones_in(a, first)
        s = _stokes(float(a.beam.power_W), ex, ey)  # type: ignore[arg-type]
        total = [total[0] + s[0], total[1] + s[1], total[2] + s[2], total[3] + s[3]]
    s0, s1, s2, s3 = total
    if s0 <= 0.0:
        return PolarizationMixture(**base, stokes_W=(0.0, 0.0, 0.0, 0.0), quantity_state=QuantityState.KNOWN_ZERO)
    polarized = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    dop = min(polarized / s0, 1.0)
    state = None
    if polarized > cfg.energy_conservation_tol * s0:
        ex, ey = polarized_jones(s1 / polarized, s2 / polarized, s3 / polarized)
        assert observation.plane is not None
        ellipse = jones_summary(ex, ey, first.beam.basis.e1, first.beam.basis.e2, observation.plane.u_axis, observation.plane.v_axis)
        if ellipse is not None:
            state = ellipse.model_copy(update={"s1": s1 / s0, "s2": s2 / s0, "s3": s3 / s0, "degree_of_polarization": dop})
    return PolarizationMixture(**base, stokes_W=(s0, s1, s2, s3), degree_of_polarization=dop, polarization_state=state,
                               quantity_state=QuantityState.KNOWN)


def _key(a: ArrivalResult) -> Key:
    return (a.beam.source_id, a.beam.spectral_index, a.crossing_index, a.lineage)


def polarization_mixtures(observation: ObservationResult, untraced_lines: set[tuple[str, int]],
                          cfg: NumericalConfig) -> tuple[PolarizationMixture, ...]:
    """The plane's ``PolarizationMixture`` per route of a partially polarized source's line, in the order of their first
    arrival on the plane; ``untraced_lines``: the ``(source_id, spectral_index)`` lines the path budget left partly
    untraced."""
    routes: dict[Key, list[ArrivalResult]] = {}
    for a in observation.arrivals:
        if a.polarization_component is not None:
            routes.setdefault(_key(a), []).append(a)
    return tuple(_mixture(sorted(group, key=lambda a: a.polarization_component or 0), observation, untraced_lines, cfg)
                 for group in routes.values())


def untraced_lines(power_budget: PowerBudget | None) -> set[tuple[str, int]]:
    """The lines part of whose light the path budget did not launch or follow (``UNTRACED_COMPONENT_SINKS``)."""
    lines: Iterable = power_budget.lines if power_budget is not None else ()
    return {(line.source_id, line.spectral_index) for line in lines
            if any(line.sinks.get(kind, 0.0) > 0.0 for kind in UNTRACED_COMPONENT_SINKS)}


def with_polarization_mixtures(observations: list[ObservationResult], power_budget: PowerBudget | None,
                               cfg: NumericalConfig) -> list[ObservationResult]:
    """``observations`` with their ``polarization_mixtures`` (unchanged where no partially polarized light arrives)."""
    if not any(a.polarization_component is not None for o in observations for a in o.arrivals):
        return observations
    lines = untraced_lines(power_budget)
    out = []
    for o in observations:
        mixtures = polarization_mixtures(o, lines, cfg)
        out.append(o.model_copy(update={"polarization_mixtures": mixtures}) if mixtures else o)
    return out


#: The single-beam polarization metrics: on a beam of a partially polarized source they describe the light on its route
#: (its ``PolarizationMixture``), not one component.
POLARIZATION_METRICS = frozenset({"stokes_s1", "stokes_s2", "stokes_s3", "polarization_ellipticity_deg", "polarization_azimuth_deg",
                                  "degree_of_polarization"})


def mixture_metric(metric: str, unit: str, arrivals: Sequence[ArrivalResult], observation: ObservationResult) -> MetricValue:
    """A polarization metric (``POLARIZATION_METRICS``) of selected beams of which some are components of a partially
    polarized source: the value of the one route they lie on."""
    from ..errors import Status
    from .metrics import MetricValue

    def on(mix: PolarizationMixture, a: ArrivalResult) -> bool:
        return a.beam.path_id in mix.path_ids and a.crossing_index == mix.crossing_index

    routes = [mix for mix in observation.polarization_mixtures if any(on(mix, a) for a in arrivals)]
    fully = [a for a in arrivals if a.polarization_component is None]
    if fully or len(routes) != 1:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                           f"{len(routes) + len(fully)} routes of light match; select one with source_id / wavelength_nm / "
                           "parameters['path_id'] (either polarization component's path id selects its route)")
    mix = routes[0]
    if mix.quantity_state is QuantityState.UNKNOWN:
        return MetricValue(None, unit, Status.UNKNOWN, mix.issue_codes, mix.message or "")
    if mix.stokes_W is None or mix.quantity_state is QuantityState.KNOWN_ZERO or mix.degree_of_polarization is None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.ZERO_POWER.value,),
                           "no power arrives on this route: its polarization is not defined")
    s0, s1, s2, s3 = mix.stokes_W
    if metric == "degree_of_polarization":
        return MetricValue(mix.degree_of_polarization, unit)
    if metric in ("stokes_s1", "stokes_s2", "stokes_s3"):
        return MetricValue({"stokes_s1": s1, "stokes_s2": s2, "stokes_s3": s3}[metric] / s0, unit)
    state = mix.polarization_state
    if state is None:
        return MetricValue(None, unit, Status.NOT_APPLICABLE, (ErrorCode.PARTIAL_POLARIZATION_COMPONENTS.value,),
                           "the light on this route is unpolarized (degree of polarization numerically 0): it has no "
                           "polarization ellipse")
    value = state.ellipticity_deg if metric == "polarization_ellipticity_deg" else state.azimuth_in_plane_deg
    if value is None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.NUMERICAL_FAILURE.value,), "metric is undefined for this beam")
    return MetricValue(value, unit)


def component_twins(arrivals: Sequence[ArrivalResult]) -> bool:
    """Whether the selected arrivals are the two polarization components of one route (the same light)."""
    return (len(arrivals) == 2 and all(a.polarization_component is not None for a in arrivals)
            and arrivals[0].polarization_component != arrivals[1].polarization_component and _key(arrivals[0]) == _key(arrivals[1]))


def route_arrivals(arrivals: Sequence[ArrivalResult], observation: ObservationResult) -> list[ArrivalResult]:
    """``arrivals`` with every arrival of ``observation`` on the route of a polarization component among them (the same
    source line, crossing and lineage: the light ``polarization_mixtures`` sums), in the plane's order."""
    routes = {_key(a) for a in arrivals if a.polarization_component is not None}
    if not routes:
        return list(arrivals)
    chosen = {id(a) for a in arrivals}
    return [a for a in observation.arrivals if id(a) in chosen or (a.polarization_component is not None and _key(a) in routes)]
