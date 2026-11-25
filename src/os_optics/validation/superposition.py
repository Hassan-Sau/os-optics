"""Coherent superposition of every beam that reaches one observation plane (any number of beams).

Model (the same basis as the two-beam analysis, ``optics/interference/two_beam.py``, extended to N beams): on the plane,
beam ``j`` is a paraxial Gaussian envelope with a plane-wave phase,

    E_j(p) = sqrt(A_j) exp(-(p - c_j)^T S_j (p - c_j)) exp(i [k0 OPL_j + kappa_j . (p - c_j)]) e_j

with ``p`` the plane coordinates (u, v), ``c_j`` the chief-ray hit point, ``S_j = sum_k b_k b_k^T / w_k^2`` the envelope
matrix of its two Gaussian axes projected on the plane, ``A_j = 2 P_j / (pi w_x w_y)``, ``kappa_j = k0 n (d_j . u,
d_j . v)`` and ``e_j`` its unit complex polarization vector (Jones vector in its 3D transverse basis). The power through
the plane is

    P = sum_j P_j + sum_{j<k} 2 Re[ gamma_jk (e_j* . e_k) Integral E-envelopes and phases ] (cos a_j + cos a_k) / 2

where the cross term is formed only between beams that cross the plane in the same direction (the same sign of
``d . n``): for two beams crossing it in opposite directions the time-averaged Poynting flux through the plane has no
interference term (their standing wave modulates the energy density, not the power through the plane). The integral over the whole plane is closed form (a Gaussian with a complex linear term) and ``gamma_jk`` is the
mutual coherence: 0 for beams of different coherence groups (mutually incoherent sources) or wavelengths, else the
Gaussian-spectrum model's complex degree of coherence after the two paths' relative group delay and group-delay dispersion
(``optics/interference/coherence.py``, ``L_c`` the smaller declared coherence length; the real ``exp(-(OPD / L_c)^2)`` in
media without dispersion, and for an arrival without a group path -- a result of an earlier Engine version; a dispersion
difference gives it a phase that shifts the fringes). Two beams of a source of continuous spectrum have the spectrum's
complex degree of coherence instead. A complex ``gamma`` enters the cross term as ``2 Re[conj(gamma) (e_j* . e_k)
Integral]`` (for a continuous spectrum the line's share of the whole spectrum's); a partially polarized source is coherent
with another source of its group through its polarized part only (``optics/interference/source_coherence.py``). The Gouy phase and
wavefront curvature across the plane are not included (as for the two-beam analysis), and the integral runs over the
whole plane, not the active area.

Where two beams meet coatings differently whose data bound their group delay, their complex degree of coherence is known
within ``delta`` of its value at the traced delays (``optics/interference/coherence.py``), its phase at the line only where
those coatings state their phases. Where they do, ``total_power_W`` is the sum at the traced delays, and the bounds leave
each pair's cross term within ``2 delta_jk |(e_j* . e_k) Integral| (cos a_j + cos a_k) / 2`` of its traced value and within
the most a degree of coherence of 1 gives (``total_power_min_W`` / ``total_power_max_W``). Where a differing pass states no phase (``phase_open``), the cross term of that pair may take any
phase: it lies within ``+- 2 min(1, |gamma| + delta) |(e_j* . e_k) Integral| (cos a_j + cos a_k) / 2`` (the overlap taken as 1
where an open-phase pass meets a beam mixing s and p); the sum is stated at the trace's default interface phases
(``total_power_W``, ``interference_power_W``, with an ``ESTIMATE`` entry naming the passes) and ``total_power_min_W`` /
``total_power_max_W`` give the power's range over every phase (at least 0). A differing pass that bounds no delay counts at
no added delay in the value (the entry names it) and with every degree of coherence up to the line's in the range. Behind
an element that states no optical path the open phase has no value (status UNKNOWN, code ``PHASE_UNDEFINED``,
``phase_statement``).

A source line that states no coherence (no ``coherence_length_mm``, ``linewidth_nm`` or spectrum) is taken as monochromatic:
the degree of coherence of its pairs is 1 (``LineCoherence.assumed``). ``total_power_W`` is then the sum at that stand-in,
with a ``value_basis`` entry (``ESTIMATE``, assumed ``FULL_COHERENCE`` at the sources), and ``total_power_min_W`` /
``total_power_max_W`` the range every degree of coherence from 0 to 1 gives (each such cross term between 0 and its value).
Where the other line of such a pair states its coherence length, the term is at that length, and the range takes every
degree of coherence from 0 to the largest any shorter length of the unstated line gives (``LineCoherence.top``: more than
the value where the paths' group-delay dispersions differ; the line's whole degree behind coatings of bounded group delay).
Where those dispersions differ the phase of the degree of coherence also moves with that length, so the range takes the
cross term at every phase up to that end: within ``+- 2 top |(e_j* . e_k) Integral| (cos a_j + cos a_k) / 2``.

Where the beams cancel (a dark port), the sum is a residue of a few units in the last place of the powers it adds, of
either sign: a sum or an end of its range within ``CANCELLATION_REL_TOL`` (1e-12) of the beams' powers and the
magnitudes of their cross terms reads 0 W, and ``interference_power_W`` is then ``-incoherent_power_W``.

On an absorbing plane the power is the power the plane detects: ``P_j`` is each beam's detected power
(``ArrivalResult.detected_power_W``: its power on the active area), and each beam's field enters the cross term scaled by
``sqrt(detected / incident)``. A single beam's coherent power is then exactly its detected power (``power_W``); with part of a
beam outside the active area the cross term is still the whole-plane integral of the scaled fields (flagged
``SUPERPOSITION_WHOLE_PLANE``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.basis import Assumed, ValueBasis
from ..contracts.results import ArrivalResult, CoherentSuperposition, ObservationPlaneFrame
from ..errors import ErrorCode, Status, worst
from ..errors.status import AssumedInput, ValueKind
from ..math.exact_linalg import abs2, cmul, cvdot, det2, solve2
from ..math.exact_trig import cis
from ..math.exact_trig import exp as exact_exp
from ..math.vectors import correctly_rounded_sum, dot, matvec
from ..optics.interference.coherence import (REVIVAL_NEGLIGIBLE, delay_bound_fs, delays_agree, differing_passes,
                                              mixed_open_passes, pass_id, phase_open, relative_delay_bound,
                                              unstated_delay_passes)
from ..optics.interference.source_coherence import (LineCoherence, SourceCoherence, SpectrumCoherence, abs_complex,
                                                     pair_model)
from ..optics.interference.two_beam import listed_passes, open_pass_ids
from ..run_control import checkpoint
from ..units import NM_TO_MM

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

# A beam whose active-area fraction is below this has part of its power outside the active area, which the whole-plane
# superposition integral still counts: the result is then flagged.
FULL_CAPTURE_FRACTION = 1.0 - 1e-6
#: A coherent sum (or an end of its range) within this fraction of the powers it adds -- the beams' powers and the
#: magnitudes of their cross terms -- is 0: the cancellation residue of a dark port is a few units in the last place of
#: those powers, of either sign, and follows the last bits of the poses. Below it the sum reads 0 W, never a negative power.
CANCELLATION_REL_TOL = 1e-12


def _cancelled(power_W: float, scale_W: float) -> float:
    """``power_W``, or 0 where it lies within the cancellation residue of a sum whose added powers total ``scale_W``
    (``CANCELLATION_REL_TOL``) or below 0."""

    return 0.0 if power_W <= CANCELLATION_REL_TOL * scale_W else power_W


def received_power_W(arrival: ArrivalResult) -> float:
    """The power the plane receives from ``arrival``: its detected power on an absorbing plane, else the crossing beam's;
    nothing from a beam outside the plane's acceptance cone (``within_acceptance`` False)."""
    if arrival.detected_power_W is not None:
        return arrival.detected_power_W
    power = arrival.beam.power_W
    return 0.0 if arrival.within_acceptance is False or power is None else power


def _envelope(arrival: ArrivalResult, u: Array, v: Array, power_W: float | None = None) -> tuple[Array, float] | None:
    """(S, A): the envelope matrix in plane coordinates and the peak transverse intensity (W/mm^2) of a beam of
    ``power_W`` (default: the incident beam's); None without a valid Gaussian state."""
    g = arrival.beam.gaussian
    if g is None or not g.valid or g.e1 is None or g.e2 is None or not g.beam_radius_x_mm or not g.beam_radius_y_mm:
        return None
    wx, wy = g.beam_radius_x_mm, g.beam_radius_y_mm
    bx = np.array([dot(g.e1, u), dot(g.e1, v)])
    by = np.array([dot(g.e2, u), dot(g.e2, v)])
    s = np.outer(bx, bx) / (wx * wx) + np.outer(by, by) / (wy * wy)
    power = arrival.beam.power_W if power_W is None else power_W
    return s, 2.0 * power / (math.pi * wx * wy)


def _polarization(arrival: ArrivalResult) -> CArray:
    j = arrival.beam.polarization
    basis = arrival.beam.basis
    field = complex(*j.ex) * np.array(basis.e1) + complex(*j.ey) * np.array(basis.e2)
    norm = math.sqrt(cvdot(field, field).real)
    return field / norm if norm > 0.0 else field


def _cos(arrival: ArrivalResult, normal: Array) -> float:
    return abs(float(dot(arrival.beam.direction, normal)))


def _fringe_path(arrival: ArrivalResult) -> float:
    """The length whose difference between two beams gives their fringes: the optical path, or behind an element that states none
    its stand-in, which means something only against a beam that crossed the same passes (``phase_open`` decides that)."""
    length = arrival.beam.optical_path_length_mm
    return arrival.relative_path_mm if length is None else length  # type: ignore[return-value]


def same_direction(a: ArrivalResult, b: ArrivalResult, normal: Array) -> bool:
    """Whether two arrivals cross the plane in the same direction (``d . n`` of the same sign): only such beams have an
    interference term in the power through the plane."""
    return float(dot(a.beam.direction, normal)) * float(dot(b.beam.direction, normal)) > 0.0


def cross_integral(a: ArrivalResult, b: ArrivalResult, u: Array, v: Array, n_medium: float, *,
                   received: bool = False) -> complex | None:
    """``Integral sqrt(I_a I_b) exp(i (psi_b - psi_a)) dA`` over the plane (W per (mm^2 / mm^2): power units), closed form.
    ``received``: each beam at the power the plane receives from it (``received_power_W``) instead of its incident power."""
    ea = _envelope(a, u, v, received_power_W(a) if received else None)
    eb = _envelope(b, u, v, received_power_W(b) if received else None)
    if ea is None or eb is None:
        return None
    (sa, aa), (sb, ab) = ea, eb
    k0 = 2.0 * math.pi / (a.beam.wavelength_nm * NM_TO_MM)
    ka = k0 * n_medium * np.array([dot(a.beam.direction, u), dot(a.beam.direction, v)])
    kb = k0 * n_medium * np.array([dot(b.beam.direction, u), dot(b.beam.direction, v)])
    ca, cb = np.array(a.uv_mm), np.array(b.uv_mm)
    m = sa + sb
    det = det2(m)
    if det <= 0.0:
        return None
    beta = 2.0 * (matvec(sa, ca) + matvec(sb, cb)) + 1j * (kb - ka)
    const = float(dot(ca, matvec(sa, ca)) + dot(cb, matvec(sb, cb)))
    x = solve2(m, beta)
    exponent = complex(cmul(beta[0], x[0]) + cmul(beta[1], x[1])) / 4.0 - const + 1j * float(dot(ka, ca) - dot(kb, cb))
    exponent += 1j * k0 * (_fringe_path(b) - _fringe_path(a))
    # exp(exponent) = e^Re cis(Im), libm-free (the coherent powers a run reports)
    return math.sqrt(aa * ab) * math.pi / math.sqrt(det) * (float(exact_exp(exponent.real)) * cis(exponent.imag))


def _model(a: ArrivalResult, b: ArrivalResult, sources: SourceCoherence | None):
    """The coherence model of two arrivals of one group (``optics/interference/source_coherence.py``: ``None`` where a line's
    coherence length is not declared, ``MIXED_SPECTRA`` where the sources do not state their mutual coherence)."""
    lengths = [x for x in (a.coherence_length_mm, b.coherence_length_mm) if x is not None]
    return pair_model(sources, a.beam.source_id, b.beam.source_id, a.beam.wavelength_nm, lengths)


def _share(a: ArrivalResult, b: ArrivalResult, sources: SourceCoherence | None) -> float:
    """The factor of a pair's degree of coherence for the partially polarized sources among them (two sources of one group:
    only a polarized part is coherent with another source, ``optics/interference/source_coherence.py``)."""
    if sources is None:
        return 1.0
    return sources.polarized_share(a.beam.source_id, a.polarization_component, b.beam.source_id, b.polarization_component)


def _scaled(gamma: complex | float, share: float) -> complex | float:
    if share == 1.0:
        return gamma
    if isinstance(gamma, complex):
        return complex(gamma.real * share, gamma.imag * share)
    return gamma * share


def _gamma(a: ArrivalResult, b: ArrivalResult, cfg: NumericalConfig,
           sources: SourceCoherence | None = None) -> tuple[complex | float | None, str | None]:
    """Mutual coherence of two arrivals (None + issue code when it cannot be stated: also where a pulse train's revival could
    change it, ``sources`` what the sources state beyond their lines, ``optics/interference/source_coherence.py``): a line's
    or a continuous spectrum's ``gamma`` relative to the phase at the line -- complex, or a real number for a line whose
    paths' group-delay dispersions do not differ."""
    if a.coherence_group != b.coherence_group:
        return 0.0, None
    model = _model(a, b, sources)
    if not isinstance(model, (LineCoherence, SpectrumCoherence)):
        return None, ErrorCode.COHERENCE_UNDEFINED.value
    # (their relative envelope delay is known only along the same passes of undetermined group delay, and only where every
    # medium's group index is known: optics/interference/coherence.py)
    if (a.group_path_length_mm is None or b.group_path_length_mm is None
            or not delays_agree(a.group_delay_unknown_at, b.group_delay_unknown_at)):
        return None, ErrorCode.COHERENCE_UNDEFINED.value
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = (a.group_delay_dispersion_mm2 or 0.0) - (b.group_delay_dispersion_mm2 or 0.0)
    trains = sources.trains(a.beam.source_id, b.beam.source_id) if sources is not None else ()
    if trains:
        revival = model.revival(delay, dispersion, trains)
        if revival is None or revival >= REVIVAL_NEGLIGIBLE:
            return None, ErrorCode.COHERENCE_UNDEFINED.value
    gamma = model.gamma(delay, dispersion)
    if gamma is None:
        return None, ErrorCode.COHERENCE_UNDEFINED.value
    return _scaled(gamma, _share(a, b, sources)), None


def _bounded_gamma(a: ArrivalResult, b: ArrivalResult, cfg: NumericalConfig, sources: SourceCoherence | None = None
                   ) -> tuple[complex | float | None, float, bool, str | None, tuple[Assumed, ...]]:
    """``(gamma, delta, open_phase, code, assumed)``: the mutual coherence of two arrivals at their traced group delays
    (``_gamma``), the most the passes they meet differently may change it (``delta``: the coatings' group-delay bounds, 0.0
    where the two beams crossed the same passes, and the whole range up to the line's coherence where a pass bounds no
    delay), whether a differing pass states no phase at the line (``open_phase``: the pair's cross term may take any phase)
    and the conventions the value at the traced delays and phases stands on (``assumed``: no added delay at a pass that
    states none, the default interface phases where a pass of unstated phase meets a beam mixing s and p); ``None`` + an
    issue code where it cannot be stated (``_gamma``; also where a pulse train's revival is not negligible anywhere the
    bounded delays allow, ``optics/interference/coherence.py``)."""
    if a.coherence_group != b.coherence_group:
        return 0.0, 0.0, False, None, ()
    model = _model(a, b, sources)
    if not isinstance(model, (LineCoherence, SpectrumCoherence)):
        return None, 0.0, False, ErrorCode.COHERENCE_UNDEFINED.value, ()
    passes_a, passes_b = a.group_delay_unknown_at, b.group_delay_unknown_at
    bound = relative_delay_bound(passes_a, passes_b)
    if a.group_path_length_mm is None or b.group_path_length_mm is None:
        return None, 0.0, False, ErrorCode.COHERENCE_UNDEFINED.value, ()
    if bound == 0.0:
        gamma, code = _gamma(a, b, cfg, sources)
        return gamma, 0.0, False, code, ()
    unstated = unstated_delay_passes(passes_a, passes_b) if bound is None else ()
    mixed = mixed_open_passes(passes_a, passes_b)
    assumed = ((Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=unstated, value=0.0, unit="fs"),) if unstated else ()) + (
        (Assumed(input=AssumedInput.UNSTATED_PHASE_AS_INTERFACE, at=mixed),) if mixed else ())
    trains = sources.trains(a.beam.source_id, b.beam.source_id) if sources is not None else ()
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = (a.group_delay_dispersion_mm2 or 0.0) - (b.group_delay_dispersion_mm2 or 0.0)
    if trains:
        revival = model.revival_within(delay, bound or 0.0, dispersion, trains)
        if revival is None or revival >= REVIVAL_NEGLIGIBLE:
            return None, 0.0, False, ErrorCode.COHERENCE_UNDEFINED.value, ()
    gamma = model.gamma(delay, dispersion)
    if gamma is None:
        return None, 0.0, False, ErrorCode.COHERENCE_UNDEFINED.value, ()
    share = _share(a, b, sources)
    # (an unbounded delay leaves any degree of coherence up to the line's, ``share``: the range covers it; so do bounded
    # delays where one line states no coherence length -- their change 2 B / (sqrt(pi) L) grows without limit as its own
    # length L shrinks, ``LineCoherence.open_length``)
    open_length = isinstance(model, LineCoherence) and model.open_length
    spread = share if bound is None or open_length else model.spread(bound) * share
    return _scaled(gamma, share), spread, phase_open(passes_a, passes_b), None, assumed


def _stand_in_top(a: ArrivalResult, b: ArrivalResult, model: LineCoherence, sources: SourceCoherence | None,
                  gamma: complex | float) -> complex | float:
    """The upper end of the degree of coherence of a pair whose line states no coherence (``LineCoherence.assumed``) at its
    traced delay: ``gamma`` itself where neither line states a length (1 at every delay), else the largest ``|gamma|`` over
    every length of the unstated line up to the stated one (``LineCoherence.top``: more than ``|gamma|`` where the paths'
    group-delay dispersions differ) -- ``gamma`` itself where that is its own real value."""
    if not model.open_length or a.group_path_length_mm is None or b.group_path_length_mm is None:
        return gamma
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = (a.group_delay_dispersion_mm2 or 0.0) - (b.group_delay_dispersion_mm2 or 0.0)
    top = model.top(delay, dispersion)
    if not isinstance(gamma, complex) and top == model.magnitude(delay, dispersion):
        return gamma
    return float(_scaled(top, _share(a, b, sources)))


def _line_magnitude(a: ArrivalResult, b: ArrivalResult, model: object, sources: SourceCoherence | None,
                    gamma: complex | float) -> float:
    """``|gamma|`` of a pair: a line's ``degree_of_coherence`` (the same bits whatever its phase), else ``abs_complex``."""
    if (not isinstance(gamma, complex) or not isinstance(model, LineCoherence) or a.group_path_length_mm is None
            or b.group_path_length_mm is None):
        return abs_complex(gamma)
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = (a.group_delay_dispersion_mm2 or 0.0) - (b.group_delay_dispersion_mm2 or 0.0)
    return float(_scaled(model.magnitude(delay, dispersion), _share(a, b, sources)))


def _merged(assumed: list[Assumed]) -> tuple[Assumed, ...]:
    """One entry per stand-in input, its passes the union of the pairs'."""
    out: dict[AssumedInput, Assumed] = {}
    for x in assumed:
        prev = out.get(x.input)
        out[x.input] = x if prev is None else prev.model_copy(update={"at": tuple(sorted(set(prev.at) | set(x.at)))})
    return tuple(out.values())


def _phase_statement(ids: set[str], path_known: bool) -> str:
    """``CoherentSuperposition.phase_statement``: which passes leave the phase open (coatings: an element without a thickness has
    no bound, so no range), how to read the range, what gives the value."""
    return "".join((
        f"The phase between beams on this plane is open: the passes {listed_passes(sorted(ids))} state no phase at this wavelength. ",
        "total_power_min_W and total_power_max_W give the coherent power's range over every phase. ",
        "The change of a beam's optical_path_length_mm between two positions of an arm gives the change of the phase. " if path_known else "",
        "To obtain the value, state the coating's phases."))


def superpose(members: list[ArrivalResult], plane: ObservationPlaneFrame, cfg: NumericalConfig,
              sources: SourceCoherence | None = None) -> CoherentSuperposition:
    """The coherent superposition of ``members`` (arrivals of one wavelength on ``plane``), in the power the plane receives
    (on an absorbing plane each beam's detected power); ``sources``: what the sources state beyond their lines (``_gamma``).
    ``OUT_OF_MODEL_SCOPE`` with no total where a member's power is out of model scope (``ArrivalResult.power_out_of_scope_at``)."""
    u, v, normal = np.array(plane.u_axis), np.array(plane.v_axis), np.array(plane.normal)
    incoherent = float(correctly_rounded_sum(received_power_W(m) for m in members))
    base = dict(wavelength_nm=members[0].beam.wavelength_nm, path_ids=tuple(m.beam.path_id for m in members),
                crossing_indices=tuple(m.crossing_index for m in members), incoherent_power_W=incoherent)
    codes: set[str] = set()
    statuses: list[Status] = []
    if any(m.power_out_of_scope_at for m in members):
        # (a beam's power is its chief ray's, not the light's: no coherent sum of it -- ArrivalResult.power_out_of_scope_at)
        return CoherentSuperposition(status=Status.OUT_OF_MODEL_SCOPE, issue_codes=(ErrorCode.OUT_OF_MODEL_SCOPE.value,), **base)
    if any(m.active_area_power_fraction is not None and m.active_area_power_fraction < FULL_CAPTURE_FRACTION for m in members):
        codes.add(ErrorCode.SUPERPOSITION_WHOLE_PLANE.value)
        statuses.append(Status.WARNING)
    interference = 0.0
    below = above = 0.0  # how far the coatings' group-delay bounds may move the sum down / up (module docstring)
    open_terms = 0.0  # the largest magnitude of the cross terms whose phase is open (module docstring)
    open_passes: set[str] = set()  # the passes that leave those phases open
    at_convention = 0.0  # those cross terms at the trace's default interface phases
    added = incoherent  # the powers the sum adds: the beams' and the magnitudes of the cross terms (``_cancelled``)
    assumed: list[Assumed] = []  # the conventions the sum stands on (value_basis)
    delayed = False  # a pair's degree of coherence is at its traced delay within a range (bounded or unbounded passes)
    bounded_at: set[str] = set()  # the passes of bounded group delay the pairs meet differently (the traced delay stands there)
    stand_in: set[str] = set()  # the sources of the pairs whose line states no coherence (taken as monochromatic)
    # (two beams of different coherence groups have no cross term -- _bounded_gamma: gamma 0, delta 0 --, so only the pairs
    # within one group are formed, in the same (i, j) order as over every pair: the sums add the same terms in the same order)
    groups: dict[str, list[ArrivalResult]] = {}
    position: list[int] = []
    for member in members:
        group = groups.setdefault(member.coherence_group, [])
        position.append(len(group))
        group.append(member)
    for i, a in enumerate(members):
        checkpoint(0)  # (the run's monitor may stop a long sum: os_optics/run_control.py)
        for b in groups[a.coherence_group][position[i] + 1:]:
            if not same_direction(a, b, normal):
                continue  # counter-propagating: no cross term in the power through the plane
            gamma, delta, open_phase, code, pair_assumed = _bounded_gamma(a, b, cfg, sources)
            if code is not None:
                return CoherentSuperposition(status=Status.UNKNOWN, issue_codes=tuple(sorted(codes | {code})), **base)
            assumed.extend(x for x in pair_assumed if x not in assumed)
            mixed = any(x.input is AssumedInput.UNSTATED_PHASE_AS_INTERFACE for x in pair_assumed)
            delayed = delayed or delta > 0.0
            if delta > 0.0 and relative_delay_bound(a.group_delay_unknown_at, b.group_delay_unknown_at):
                # (every differing pass bounds its delay: the value is at the traced delay there, as the pair states)
                bounded_at.update(pass_id(x) for x in differing_passes(a.group_delay_unknown_at, b.group_delay_unknown_at)
                                  if delay_bound_fs(x) is not None)
            model = _model(a, b, sources)
            full = isinstance(model, LineCoherence) and model.assumed
            top = _stand_in_top(a, b, model, sources, gamma) if full and delta == 0.0 else gamma  # type: ignore[arg-type]
            if gamma == 0.0 and delta == 0.0 and top == 0.0:
                continue
            if full:  # (the sources taken as monochromatic: those whose line states no coherence)
                stand_in.update(x.beam.source_id for x in (a, b) if x.coherence_length_mm is None)
            magnitude = _line_magnitude(a, b, model, sources, gamma)  # type: ignore[arg-type]
            # (the index of a valid Gaussian state: a state outside the model publishes none)
            na = a.beam.gaussian.medium_index if a.beam.gaussian is not None and a.beam.gaussian.valid else None
            nb = b.beam.gaussian.medium_index if b.beam.gaussian is not None and b.beam.gaussian.valid else None
            if na is None or nb is None:
                return CoherentSuperposition(status=Status.OUT_OF_MODEL_SCOPE,
                                             issue_codes=tuple(sorted(codes | {ErrorCode.OUT_OF_MODEL_SCOPE.value})), **base)
            if abs(na - nb) > cfg.energy_conservation_tol:
                return CoherentSuperposition(status=Status.OUT_OF_MODEL_SCOPE, issue_codes=tuple(sorted(codes | {ErrorCode.MEDIUM_MISMATCH.value})), **base)
            integral = cross_integral(a, b, u, v, na, received=True)
            if integral is None:
                return CoherentSuperposition(status=Status.OUT_OF_MODEL_SCOPE,
                                             issue_codes=tuple(sorted(codes | {ErrorCode.OUT_OF_MODEL_SCOPE.value})), **base)
            overlap = cvdot(_polarization(a), _polarization(b))
            flux = 0.5 * (_cos(a, normal) + _cos(b, normal))
            cross = overlap * integral
            # (a real gamma; a complex one -- a continuous spectrum's, a line's through different group-delay dispersions --:
            # Re[conj(gamma) cross], optics/interference/source_coherence.py)
            term = (2.0 * gamma * cross.real * flux if not isinstance(gamma, complex)
                    else 2.0 * (gamma.real * cross.real + gamma.imag * cross.imag) * flux)
            added += abs(term)
            if open_phase or mixed:
                # the term at the trace's default phases (the value, ``assumed``) and its range over every phase and over
                # every polarization overlap where a pass mixes s and p
                at_convention += term
                open_terms += 2.0 * min(1.0, magnitude + delta) * abs((1.0 if mixed else overlap) * integral) * flux
                open_passes.update(open_pass_ids(a.group_delay_unknown_at, b.group_delay_unknown_at))
                continue
            interference += term
            if full and isinstance(gamma, complex) and delta == 0.0:
                # (the other line's stated length through different group-delay dispersions: a shorter length of the
                # unstated line moves the phase of the degree of coherence as well as its magnitude, up to ``top``, so the
                # term takes every phase up to that end)
                reach = 2.0 * abs_complex(top) * abs(cross) * flux  # type: ignore[arg-type]
                below += term + reach
                above += reach - term
            elif full:
                # (a degree of coherence taken as 1 for a line that states none, or the other line's stated length: every
                # degree from 0 to its upper end ``top`` -- the value itself, or more where a shorter length of the
                # unstated line gives more, ``_stand_in_top`` -- scales this term between 0 and its value at that end, so
                # the sum lies in the range those give)
                end = term if top == gamma else 2.0 * top * cross.real * flux  # type: ignore[operator]
                below += term - min(end, 0.0)
                above += max(end, 0.0) - term
            if delta > 0.0:
                # (within 2 delta |c| of the traced term, and never beyond the most a degree of coherence of 1 gives)
                most = 2.0 * abs(overlap * integral) * flux
                below += min(2.0 * delta * abs(overlap * integral) * flux, most + term)
                above += min(2.0 * delta * abs(overlap * integral) * flux, most - term)
    total = incoherent + interference
    # (the ends of the range: a cancellation residue of the powers they add, the bounds' included, reads 0)
    ranged = added + below + above + open_terms
    least, most = _cancelled(total - below - open_terms, ranged), _cancelled(total + above + open_terms, ranged)
    path_known = all(m.beam.optical_path_length_mm is not None for m in members)
    if open_terms > 0.0 and not path_known:
        # (a cross term of open phase behind an element that states no optical path: no phase is formed, so no value; the
        # power's range over every phase)
        return CoherentSuperposition(status=Status.UNKNOWN, issue_codes=tuple(sorted(codes | {ErrorCode.PHASE_UNDEFINED.value})),
                                     total_power_min_W=least, total_power_max_W=most, **base,
                                     phase_statement=_phase_statement(open_passes, False))
    bounds = dict(total_power_min_W=least, total_power_max_W=most) if below > 0.0 or above > 0.0 or open_terms > 0.0 else {}
    if open_terms > 0.0:  # (the cross terms of open phase at the trace's default interface phases)
        assumed.append(Assumed(input=AssumedInput.UNSTATED_PHASE_AS_INTERFACE, at=tuple(sorted(open_passes))))
    if delayed and (assumed or stand_in):
        # (delays known within bounds, or not at all: their traced values, the range beside -- also beside a monochromatic
        # stand-in, whose entry would otherwise not say so; the bounded passes named, as the pairs name them)
        assumed.append(Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=tuple(sorted(bounded_at))))
    if stand_in:
        # (a cross term at an assumed full coherence: the sum's value is the stated model's at that stand-in, which may lie
        # on either side of the source's own -- an estimate)
        assumed.append(Assumed(input=AssumedInput.FULL_COHERENCE, at=tuple(f"source:{x}" for x in sorted(stand_in)), value=1.0))
    entry = ValueBasis(kind=ValueKind.ESTIMATE, assumed=_merged(assumed)) if assumed else None
    # (the sum at its conventions; within the cancellation residue it reads 0 and its cross terms the beams' powers back)
    value = total + at_convention
    cancelled = _cancelled(value, added)
    interference_W = interference + at_convention if cancelled == value else cancelled - incoherent
    return CoherentSuperposition(status=worst(statuses, empty=Status.PASS), total_power_W=cancelled,
                                 interference_power_W=interference_W, issue_codes=tuple(sorted(codes)), **bounds, **base,
                                 value_basis=None if entry is None else {"total_power_W": entry, "interference_power_W": entry})


def superpositions(arrivals: tuple[ArrivalResult, ...], plane: ObservationPlaneFrame, cfg: NumericalConfig,
                   sources: SourceCoherence | None = None) -> tuple[CoherentSuperposition, ...]:
    """One ``CoherentSuperposition`` per wavelength with at least two arrivals on the plane, in wavelength order
    (``sources``: what the sources state beyond their lines, ``superpose``). A wavelength with a beam whose power or polarization is
    not known (a grating's order) has none: its field is not known."""
    groups: list[list[ArrivalResult]] = []
    for arrival in sorted(arrivals, key=lambda a: a.beam.wavelength_nm):
        if groups and abs(groups[-1][0].beam.wavelength_nm - arrival.beam.wavelength_nm) <= cfg.interference_wavelength_tol_nm:
            groups[-1].append(arrival)
        else:
            groups.append([arrival])
    return tuple(superpose(group, plane, cfg, sources) for group in groups
                 if len(group) > 1 and all(a.beam.power_W is not None and a.beam.polarization is not None for a in group))


def assumed_coherence(members: list[ArrivalResult], sources: SourceCoherence | None) -> ValueBasis | None:
    """The entry of a coherent quantity of ``members`` where a pair of one coherence group has a line that states no
    coherence, taken as monochromatic (``LineCoherence.assumed``): ``ESTIMATE``, assumed ``FULL_COHERENCE`` at those sources;
    ``None`` where every such pair states its coherence."""
    found: set[str] = set()
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            model = _model(a, b, sources) if a.coherence_group == b.coherence_group else None
            if isinstance(model, LineCoherence) and model.assumed:  # (only the sources whose line states no coherence)
                found.update(x.beam.source_id for x in (a, b) if x.coherence_length_mm is None)
    if not found:
        return None
    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=(
        Assumed(input=AssumedInput.FULL_COHERENCE, at=tuple(f"source:{x}" for x in sorted(found)), value=1.0),))


def irradiance(members: list[ArrivalResult], plane: ObservationPlaneFrame, points_uv: Array, cfg: NumericalConfig,
               sources: SourceCoherence | None = None) -> Array | None:
    """Power flux through the plane per unit plane area (W/mm^2) at ``points_uv`` (shape (N, 2)) of the coherent superposition
    of ``members`` (one wavelength); None when a member has no valid Gaussian state or a needed coherence is undefined."""
    u, v, normal = np.array(plane.u_axis), np.array(plane.v_axis), np.array(plane.normal)
    fields: list[tuple[CArray, CArray, float, ArrivalResult]] = []
    for m in members:
        env = _envelope(m, u, v)
        g = m.beam.gaussian
        if env is None or g is None or g.medium_index is None:
            return None
        s, a = env
        k0 = 2.0 * math.pi / (m.beam.wavelength_nm * NM_TO_MM)
        kappa = k0 * g.medium_index * np.array([dot(m.beam.direction, u), dot(m.beam.direction, v)])
        rel = points_uv - np.array(m.uv_mm)
        amp = np.sqrt(a) * np.exp(-dot(rel, matvec(s, rel)))
        phase = np.exp(1j * (k0 * _fringe_path(m) + dot(rel, kappa)))
        fields.append((amp * phase, _polarization(m), _cos(m, normal), m))
    total = np.zeros(points_uv.shape[0])
    # (only beams of one coherence group have a cross term -- _gamma: 0 across groups --; the terms keep their (i, j) order)
    groups: dict[str, list[tuple[CArray, CArray, float, ArrivalResult]]] = {}
    position: list[int] = []
    for field_ in fields:
        group = groups.setdefault(field_[3].coherence_group, [])
        position.append(len(group))
        group.append(field_)
    for i, (fa, pa, cos_a, ma) in enumerate(fields):
        total = total + abs2(fa) * cos_a
        for fb, pb, cos_b, mb in groups[ma.coherence_group][position[i] + 1:]:
            if not same_direction(ma, mb, normal):
                continue
            gamma, code = _gamma(ma, mb, cfg, sources)
            if code is not None:
                return None
            if isinstance(gamma, complex):
                cross = cmul(cmul(np.conj(fa), fb), cvdot(pa, pb))
                total = total + 2.0 * (gamma.real * cross.real + gamma.imag * cross.imag) * 0.5 * (cos_a + cos_b)
            elif gamma:
                total = total + 2.0 * gamma * cmul(cmul(np.conj(fa), fb), cvdot(pa, pb)).real * 0.5 * (cos_a + cos_b)
    return total
