"""Analytic two-beam interference on an observation plane.

Uses the actual 3D chief-ray wave vectors, traced OPLs (the fringe phase), group optical paths and group-delay dispersions
(the degree of coherence: ``optics/interference/coherence.py``), Jones fields and the Gaussian envelopes at the plane.
It is a two-plane-wave / paraxial-Gaussian-envelope model, not wave-optics propagation. The degree of coherence ``gamma`` is
relative to the phase at the beams' line: complex for a line whose paths' group-delay dispersions differ (its phase shifts the
fringes) and for a source of continuous spectrum (the spectrum's coherence), and the combined power is
``P_a + P_b + 2 sqrt(P_a P_b) eta_pol eta_sp Re[gamma exp(i phi)]``; a partially polarized source is coherent with another
source of its group through its polarized part only (``optics/interference/source_coherence.py``).

Where a pass the two beams do not share states no phase at the line (``phase_open``) their phase difference and combined power
are stated at the trace's default interface phases, ``value_basis`` naming the passes (where such a pass also states no
optical path they have no value, and ``phase_statement`` says which passes leave the phase open); where a differing pass
neither states nor bounds its group delay, the coherence quantities are stated at no added delay there, ``value_basis``
naming it. Two nearly collinear beams then also have the range of their combined power over every phase,

    Pa + Pb -+ 2 sqrt(Pa Pb) eta g eta_sp,   at least 0,

with ``g`` the largest degree of coherence the group-delay bounds, or a line's unstated coherence length, allow
(``coherence_factor_max``; the line's where a pass bounds no delay) and ``eta`` the polarization overlap (1 where an open-phase pass meets a beam mixing s and p). ``eta_sp`` is the envelopes'
overlap, without the fields' tilt across the plane: for beams at an angle within the collinear limit this range holds the plane's
(``validation/superposition.py``, the whole-plane integral with the fringes), whose cross term is smaller by up to 1.2 % at that limit.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...contracts.basis import Assumed, Omitted, ValidityCheck, ValueBasis, merge_bases
from ...contracts.results import InterferenceResult
from ...errors import ErrorCode, Status, ValidationIssue, make_issue, worst
from ...errors.status import AssumedInput, OmittedPart, ValueKind
from ...math.exact_linalg import cvdot, det2, solve2
from ...math.exact_trig import atan2, cis
from ...math.exact_trig import exp as exact_exp
from ...math.phase import reported_phase
from ...math.vectors import dot, matvec, normalize
from ...units import NM_TO_MM
from ..gaussian.state import GState
from ..polarization import field_3d, polarization_overlap
from ..rays.tracer import ArrivalRecord
from .coherence import (REVIVAL_NEGLIGIBLE, delay_bound_fs, differing_passes, mixed_open_passes, pass_id, pass_phase_stated,
                        path_open, phase_open, relative_delay_bound, unstated_delay_passes)
from .source_coherence import (MIXED_SPECTRA, LineCoherence, SourceCoherence, SpectrumCoherence, abs_complex, in_phase,
                               pair_model)
from .spectrum import PAIR_MAX_INTERVALS

Array = NDArray[np.float64]


def _s_matrix_plane(g: GState, u: Array, v: Array) -> Array:
    wx, wy = g.radius("x"), g.radius("y")
    bxu = np.array([dot(g.bx, u), dot(g.bx, v)])
    byu = np.array([dot(g.by, u), dot(g.by, v)])
    return np.outer(bxu, bxu) / (wx * wx) + np.outer(byu, byu) / (wy * wy)


OPEN_PATH_NOTE = ", nor their optical path difference (an element on the way states no optical path)"
#: How many pass ids a phase statement or an issue's context lists.
OPEN_PASSES_LISTED = 8


def open_pass_ids(a: tuple[str, ...], b: tuple[str, ...]) -> list[str]:
    """The short ids (``'<surface_id>:<event>:<side>'``) of the passes two beams do not share that state no phase at the line."""
    return sorted({x.split("@")[0] for x in differing_passes(a, b) if not pass_phase_stated(x)})


def listed_passes(ids: list[str]) -> str:
    """``ids`` for a sentence: the first ``OPEN_PASSES_LISTED`` of them."""
    more = len(ids) - OPEN_PASSES_LISTED
    return ", ".join(ids[:OPEN_PASSES_LISTED]) + (f" and {more} more" if more > 0 else "")


def _phase_statement(ids: list[str], *, ranged: bool, path_known: bool) -> str:
    """``InterferenceResult.phase_statement``: which passes leave the phase open, what still reads, and what gives the value."""
    return "".join((
        f"The phase between these beams is open: the passes {listed_passes(ids)} state no phase at this wavelength. ",
        "combined_power_collinear_min_W and combined_power_collinear_max_W give the combined power's range over every phase. " if ranged else "",
        "optical_path_difference_mm gives the change of the phase between two positions of an arm. " if path_known else "",
        "To obtain the value, state the coating's phases or model the element with its thickness and material." if path_known
        else "To obtain the value, model the element with its thickness and material."))


def _carried(bases: dict[str, ValueBasis], fields: dict) -> dict[str, ValueBasis] | None:
    """The entries of the numbers the pair states (``InterferenceResult.value_basis``): only for a field holding a number."""
    kept = {name: entry for name, entry in bases.items() if isinstance(fields.get(name), (int, float))}
    return kept or None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _length(model: LineCoherence | SpectrumCoherence) -> dict[str, float]:
    """``coherence_length_mm`` of a pair: the line's, or the continuous spectrum's (none where it has none, or where the line
    states none and is taken as monochromatic)."""
    return {} if model.length is None or getattr(model, "assumed", False) else {"coherence_length_mm": model.length}


#: The quantities of a pair that grow with its degree of coherence: with an assumed full coherence (a line that states none)
#: each is an upper end (``ValueKind.UPPER_BOUND``); the combined power, which may also fall with it, is an estimate.
COHERENCE_UPPER_ENDS = ("coherence_factor", "fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2")


def assumed_coherence_basis(fields: dict, source_ids: list[str], bounded_at: tuple[str, ...] = ()) -> dict[str, ValueBasis]:
    """The value-basis entries of the quantities a pair's assumed full coherence gives (``LineCoherence.assumed``).
    Where the pair states the upper end its degree of coherence may reach (``coherence_factor_max``: one line states its
    length and a shorter length of the other, or bounded coating delays, may give a larger value) the values are no upper
    end -- each an ``ESTIMATE``, judged on the interval up to that end; ``bounded_at``: the passes of bounded group delay
    the beams meet differently, where the values are at the traced delay (no added delay at those passes)."""
    stated = Assumed(input=AssumedInput.FULL_COHERENCE, at=tuple(f"source:{s}" for s in source_ids), value=1.0)
    if fields.get("coherence_factor_max") is not None:
        delay = (Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=bounded_at),) if bounded_at else ()
        traced = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(stated, *delay))
        out = {name: traced for name in COHERENCE_UPPER_ENDS if fields.get(name) is not None}
        if fields.get("combined_power_collinear_W") is not None and fields.get("counter_propagating") is not True:
            out["combined_power_collinear_W"] = traced
        return out
    out = {name: ValueBasis(kind=ValueKind.UPPER_BOUND, assumed=(stated,)) for name in COHERENCE_UPPER_ENDS if fields.get(name) is not None}
    if fields.get("combined_power_collinear_W") is not None and fields.get("counter_propagating") is not True:
        out["combined_power_collinear_W"] = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(stated,))
    return out


#: The two-beam quantities read from the beams' powers: none is stated where a beam's power is out of model scope.
POWER_QUANTITIES = ("intensity_balance", "fringe_visibility", "fringe_visibility_peak", "mean_irradiance_W_per_mm2",
                    "peak_irradiance_W_per_mm2", "peak_position_uv_mm", "incoherent_power_sum_W", "combined_power_collinear_W",
                    "combined_power_collinear_min_W", "combined_power_collinear_max_W")


def analyse_pair(a: ArrivalRecord, b: ArrivalRecord, cfg: NumericalConfig,
                 sources: SourceCoherence | None = None) -> tuple[InterferenceResult, list[ValidationIssue]]:
    """``_analyse_pair``; where the power of either beam is out of model scope (``ArrivalRecord.power_scope``: its power is its chief
    ray's, not the light's) the quantities read from the powers (``POWER_QUANTITIES``) are withheld and the pair is
    ``OUT_OF_MODEL_SCOPE``; the geometry, the path difference and the coherence stay."""
    result, issues = _analyse_pair(a, b, cfg, sources)
    scope = tuple(dict.fromkeys((*a.power_scope, *b.power_scope)))
    taken = tuple(dict.fromkeys((*a.taken_at, *b.taken_at)))
    if not scope and taken:
        # (a surface took a bounded share of a beam before the plane: its power is its chief ray's within that share, and every
        # pair quantity read from the powers is stated with the entry that says so -- not monotone in them: an estimate)
        upper = (a.taken_upper_W or 0.0) + (b.taken_upper_W or 0.0)
        entry = ValueBasis(kind=ValueKind.ESTIMATE, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=taken, upper=upper,
                                                                     code=ErrorCode.POWER_TAKEN_BEFORE_PLANE.value),))
        bases = dict(result.value_basis or {})
        for name in POWER_QUANTITIES:
            if isinstance(getattr(result, name), (int, float)):
                bases[name] = merge_bases(bases.get(name), entry, monotone=None)  # type: ignore[assignment]
        return result.model_copy(update={"value_basis": bases or None}), issues
    if not scope:
        return result, issues
    pair = f"{a.path_id}@{a.crossing_index}|{b.path_id}@{b.crossing_index}"
    issues = [*issues, make_issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                                  f"the power of a beam of this pair is out of model scope behind {', '.join(scope)}: it is its chief ray's "
                                  "power, not the light's, so the pair's intensity balance, fringe visibilities, irradiances and powers are "
                                  "not stated", location=f"observation:{a.observation_id}", pair=pair, surface_ids=list(scope))]
    withheld = {name: None for name in POWER_QUANTITIES}
    # (an entry of a withheld quantity goes with it: the carrier names only fields that hold a number)
    kept = {k: v for k, v in (result.value_basis or {}).items() if k not in withheld}
    return result.model_copy(update={**withheld, "value_basis": kept or None, "status": worst([result.status, Status.OUT_OF_MODEL_SCOPE]),
                                     "issue_codes": tuple(sorted({*result.issue_codes, ErrorCode.OUT_OF_MODEL_SCOPE.value}))}), issues


def _analyse_pair(a: ArrivalRecord, b: ArrivalRecord, cfg: NumericalConfig,
                  sources: SourceCoherence | None = None) -> tuple[InterferenceResult, list[ValidationIssue]]:
    """Compute all two-beam quantities for arrivals ``a`` and ``b`` on the same plane. ``sources``: what the beams' sources
    state beyond their lines (``optics/interference/source_coherence.py``): a pulse train's period length ``c / f_rep`` (its
    pulses may be mutually coherent, so where a revival could change the degree of coherence it is not stated, nor the
    coherence margin, which says whether the beams interfere: ``optics/interference/coherence.py``), a continuous spectrum
    (the pair interferes with the spectrum's coherence, not its line's) and a partially polarized source sharing a coherence
    group with another source (only its polarized part is coherent with it)."""
    issues: list[ValidationIssue] = []
    loc = f"observation:{a.observation_id}"
    pair = f"{a.path_id}@{a.crossing_index}|{b.path_id}@{b.crossing_index}"

    def issue(code: ErrorCode, status: Status, msg: str, **ctx: object) -> None:
        issues.append(make_issue(code, status, msg, location=loc, pair=pair, **ctx))

    base: dict = dict(observation_id=a.observation_id, source_id_a=a.source_id, source_id_b=b.source_id, path_id_a=a.path_id,
                      path_id_b=b.path_id, crossing_index_a=a.crossing_index, crossing_index_b=b.crossing_index, wavelength_nm=a.wavelength_nm)
    if abs(a.wavelength_nm - b.wavelength_nm) > cfg.interference_wavelength_tol_nm:
        issue(ErrorCode.WAVELENGTH_MISMATCH, Status.NOT_APPLICABLE, "beams with different wavelengths do not interfere",
              wavelength_a_nm=a.wavelength_nm, wavelength_b_nm=b.wavelength_nm)
        return InterferenceResult(status=Status.NOT_APPLICABLE, issue_codes=tuple(i.code.value for i in issues), **base), issues
    if abs(a.n - b.n) > cfg.energy_conservation_tol:
        issue(ErrorCode.MEDIUM_MISMATCH, Status.OUT_OF_MODEL_SCOPE, "the two beams arrive in media of different refractive index",
              n_a=a.n, n_b=b.n)
        return InterferenceResult(status=Status.OUT_OF_MODEL_SCOPE, issue_codes=tuple(i.code.value for i in issues), **base), issues

    lam_nm, n = a.wavelength_nm, a.n
    d1, d2 = a.direction, b.direction
    cross = np.cross(d1, d2)
    # crossing angle (stable for small angles); libm-free like every reported pair quantity
    alpha = float(atan2(math.sqrt(float(dot(cross, cross))), float(dot(d1, d2))))
    alpha_deg = math.degrees(alpha)
    fields: dict = dict(beam_crossing_angle_deg=alpha_deg, half_angle_deg=alpha_deg / 2.0)

    # ------------------------------------------------------------------------------ fringe geometry
    dd = d1 - d2
    nu = a.plane_normal
    # beams crossing the plane in opposite directions (a beam and its retro-reflection): the time-averaged Poynting flux
    # through the plane has no cross term between them
    counter_propagating = float(dot(d1, nu)) * float(dot(d2, nu)) < 0.0
    if counter_propagating:
        fields["counter_propagating"] = True  # (its combined power reads no degree of coherence: results/basis.py _coherence)
    dd_par = dd - float(dot(dd, nu)) * nu
    dd_par_norm = math.sqrt(float(dot(dd_par, dd_par)))
    if alpha_deg > cfg.interference_min_crossing_deg:
        fields["fringe_spacing_3d_nm"] = lam_nm / (2.0 * n * cis(alpha / 2.0).imag)
    if alpha_deg > cfg.interference_min_crossing_deg and dd_par_norm > cfg.zero_vector_tol:
        fields["fringe_period_nm"] = lam_nm / (n * dd_par_norm)
        direction, _ = normalize(dd_par[None, :], cfg.zero_vector_tol)
        fields["fringe_direction"] = tuple(float(x) for x in direction[0])
    elif counter_propagating:
        issue(ErrorCode.NO_FRINGES, Status.NOT_APPLICABLE, "the beams cross the plane in opposite directions along (nearly) the "
              "same line: their standing wave does not modulate the plane and the power through it has no interference term",
              crossing_angle_deg=alpha_deg)
    else:
        issue(ErrorCode.NO_FRINGES, Status.NOT_APPLICABLE, "the beams are (nearly) collinear or their fringes do not modulate the plane: "
              "a fringe period does not exist (a detector sees a power modulation instead; see combined_power_collinear_W)",
              crossing_angle_deg=alpha_deg)

    # --------------------------------------------------------------------------------------- OPL / OPD
    # (the trace's lengths, used below for the fringes only: behind an element that states no optical path they are no path length)
    fields["optical_path_length_mm_a"] = a.opl_mm if a.ray_opl_defined else None
    fields["optical_path_length_mm_b"] = b.opl_mm if b.ray_opl_defined else None
    opd = a.opl_mm - b.opl_mm
    opd_open = path_open(a.delays, b.delays)
    if not opd_open:
        fields["optical_path_difference_mm"] = opd
    fields["incoherent_power_sum_W"] = a.power_W + b.power_W

    # ---------------------------------------------------------------------------------- coherence
    same_group = a.coherence_group == b.coherence_group
    gamma: float | None = None
    if not same_group:
        issue(ErrorCode.INCOHERENT_SOURCES, Status.NOT_APPLICABLE,
              "the beams come from mutually incoherent sources: no fringes, intensities add",
              coherence_group_a=a.coherence_group, coherence_group_b=b.coherence_group)
        fields["polarization_overlap"] = polarization_overlap(field_3d(a.jones[None, :], a.e1[None, :], a.e2[None, :])[0],
                                                              field_3d(b.jones[None, :], b.e1[None, :], b.e2[None, :])[0])
        return InterferenceResult(status=Status.NOT_APPLICABLE, issue_codes=tuple(sorted({i.code.value for i in issues})),
                                  **base, **fields), issues
    # the envelopes' relative delay: the difference of the chief rays' group optical paths, known when the two beams crossed
    # the same passes of undetermined group delay, and known within the sum of the coatings' group-delay bounds when every
    # pass they do not share carries one (optics/interference/coherence.py)
    group_defined = a.group_opl_mm is not None and b.group_opl_mm is not None
    bound = relative_delay_bound(a.delays, b.delays)  # 0.0: the same passes; None: an unbounded pass differs
    # the traced delay: at a pass whose delay the design neither states nor bounds, the trace's convention (no added group
    # delay) stands for it, and every number read from it says so (``bases``: value_basis)
    delay = a.group_opl_mm - b.group_opl_mm if group_defined else None  # type: ignore[operator]
    unstated = unstated_delay_passes(a.delays, b.delays) if bound is None else ()
    # passes of unstated phase meeting a beam in a state mixing s and p: the default phases stand for the polarization they
    # leave, so the polarization overlap and every visibility read from it are on that convention
    mixed = mixed_open_passes(a.delays, b.delays)
    assumed: list[Assumed] = []
    if unstated:
        assumed.append(Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=unstated, value=0.0, unit="fs"))
    bases: dict[str, ValueBasis] = {}
    if delay is not None:
        fields["group_path_difference_mm"] = delay
        if bound:
            fields["group_delay_bound_mm"] = bound
        if unstated:
            bases["group_path_difference_mm"] = ValueBasis(kind=ValueKind.ESTIMATE, assumed=tuple(assumed))
    lc_candidates = [x for x in (a.coherence_length_mm, b.coherence_length_mm) if x is not None]
    model = pair_model(sources, a.source_id, b.source_id, lam_nm, lc_candidates)
    complex_gamma: complex | None = None  # (a continuous spectrum's degree of coherence, relative to the phase at the line)
    share = 1.0 if sources is None else sources.polarized_share(a.source_id, a.polarization_component, b.source_id,
                                                                b.polarization_component)
    if model == MIXED_SPECTRA:
        issue(ErrorCode.COHERENCE_UNDEFINED, Status.NOT_APPLICABLE,
              "the sources of this pair share a coherence group but state different spectra, one of them continuous, so the "
              "pair's coherence quantities are not computed", source_ids=sorted({a.source_id, b.source_id}),
              suggested_action="give the two sources the same spectral density, or put them in different coherence groups")
    elif isinstance(model, (LineCoherence, SpectrumCoherence)) and delay is not None:
        trains = sources.trains(a.source_id, b.source_id) if sources is not None else ()
        # (a revival anywhere the bounded delays allow; at the traced delay where no bound applies)
        revival = model.revival_within(delay, bound or 0.0, a.gdd_mm2 - b.gdd_mm2, trains) if trains else 0.0
        spread = model.spread(bound) if bound else 0.0
        if mixed:
            assumed.append(Assumed(input=AssumedInput.UNSTATED_PHASE_AS_INTERFACE, at=mixed))
        if revival is None or revival >= REVIVAL_NEGLIGIBLE:
            # (the margin 1 - |D| / L_c says whether the beams interfere; near a revival they may, whatever its value)
            fields.update(_length(model))
            issue(ErrorCode.COHERENCE_UNDEFINED, Status.NOT_APPLICABLE,
                  f"a revival of the source's pulse train (period length {min(trains):.6g} mm) lies within reach of the pair's "
                  f"delay of {abs(delay):.6g} mm, so the pair's coherence quantities are not computed",
                  source_ids=sorted({a.source_id, b.source_id}), period_length_mm=min(trains), group_path_difference_mm=delay,
                  delay_mm=abs(delay), **({"group_delay_bound_mm": bound} if bound else {}))
        else:
            value = model.gamma(delay, a.gdd_mm2 - b.gdd_mm2)
            if value is None:
                fields.update(_length(model))
                issue(ErrorCode.COHERENCE_UNDEFINED, Status.NOT_APPLICABLE,
                      f"the continuous spectrum's coherence at the pair's delay of {abs(delay):.6g} mm needs more than "
                      f"{PAIR_MAX_INTERVALS} intervals of its integral, so the pair's coherence quantities are not computed",
                      source_ids=sorted({a.source_id, b.source_id}), group_path_difference_mm=delay, delay_mm=abs(delay),
                      max_intervals=PAIR_MAX_INTERVALS)
            else:
                if isinstance(value, complex):
                    complex_gamma = complex(value.real * share, value.imag * share)
                    # (a line's magnitude from degree_of_coherence: the same bits whatever its phase)
                    gamma = (model.magnitude(delay, a.gdd_mm2 - b.gdd_mm2) * share if isinstance(model, LineCoherence)
                             else abs_complex(complex_gamma))
                else:
                    gamma = value * share
                spread = spread * share
                fields.update(_length(model), coherence_factor=gamma)
                if model.length is not None and not getattr(model, "assumed", False):
                    fields["coherence_margin"] = 1.0 - abs(delay) / model.length
                if bound and not getattr(model, "assumed", False):
                    # (the degree of coherence the coatings' group-delay bounds allow: within ``spread`` of the traced one)
                    fields.update(coherence_factor_min=max(0.0, gamma - spread), coherence_factor_max=min(1.0, gamma + spread))
                elif isinstance(model, LineCoherence) and model.open_length and bound is not None:
                    # (one line states its coherence length, the other none, whose own length may be any shorter one: the
                    # upper end is the largest degree of coherence over those lengths -- the stated length's value where no
                    # dispersion difference lets a shorter length give more, ``LineCoherence.top``; behind bounded coating
                    # delays the line's whole degree, since their change 2 B / (sqrt(pi) L) grows without limit as L
                    # shrinks. Where that end is above the value the pair states it; nothing proves a lower end above 0,
                    # so no coherence_factor_min: judged from 0, ``validation/metrics.py`` ``_assumed_coherence``. A line of
                    # infinite length has a degree of coherence of 1 at every delay.)
                    top = share if bound else model.top(delay, a.gdd_mm2 - b.gdd_mm2) * share
                    if top > gamma:
                        fields["coherence_factor_max"] = min(1.0, top)
                if assumed:
                    # (on the trace's convention: no added delay where none is stated, the default phases where a pass mixes
                    # s and p; a bounded pass's delay lies within the interval beside the value)
                    if bound:
                        assumed.append(Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO, at=tuple(sorted(
                            {pass_id(x) for x in differing_passes(a.delays, b.delays) if delay_bound_fs(x) is not None}))))
                    coherence_basis = ValueBasis(kind=ValueKind.ESTIMATE, assumed=tuple(assumed))
                    bases.update(dict.fromkeys(("coherence_factor", "coherence_margin") if "coherence_margin" in fields
                                               else ("coherence_factor",), coherence_basis))
    elif isinstance(model, (LineCoherence, SpectrumCoherence)):
        fields.update(_length(model))
        issue(ErrorCode.COHERENCE_UNDEFINED, Status.NOT_APPLICABLE,
              f"a medium on a beam's path states no group index at {lam_nm:g} nm, so the pair's coherence quantities are "
              "not computed", source_ids=sorted({a.source_id, b.source_id}), wavelength_nm=lam_nm)
    else:
        # Informational for the overall status (NOT_APPLICABLE): only the coherence-dependent quantities are withheld, and
        # a target on one of them is UNKNOWN by itself (validation/metrics.py); every other two-beam quantity stands.
        issue(ErrorCode.COHERENCE_UNDEFINED, Status.NOT_APPLICABLE,
              "the source coherence is not declared (coherence_length_mm or linewidth_nm): fringe visibility, peak "
              "irradiance and the coherence metrics are not computed for this pair; other two-beam quantities are unaffected",
              source_ids=sorted({a.source_id, b.source_id}),
              suggested_action="declare the source's coherence_length_mm or linewidth_nm (from the laser's data sheet)")

    def stand_in(values: dict) -> dict[str, ValueBasis]:
        """The entries where the line states no coherence and is taken as monochromatic (``LineCoherence.assumed``): every
        quantity read from that full coherence says so; else none."""
        if not (isinstance(model, LineCoherence) and model.assumed and gamma is not None):
            return {}
        unstated = sorted({x.source_id for x in (a, b) if x.coherence_length_mm is None})  # (only the sources taken as monochromatic)
        bounded_at = tuple(sorted({pass_id(x) for x in differing_passes(a.delays, b.delays)
                                   if delay_bound_fs(x) is not None})) if bound else ()
        return assumed_coherence_basis({**values, "counter_propagating": counter_propagating}, unstated, bounded_at)

    def carried(values: dict) -> dict[str, ValueBasis] | None:
        """``InterferenceResult.value_basis``: the entries of the trace's conventions (``bases``) merged with those of the
        monochromatic stand-in (``stand_in``), only for a field holding a number (``_carried``)."""
        merged = dict(bases)
        for name, entry in stand_in(values).items():
            merged[name] = merge_bases(merged.get(name), entry)  # type: ignore[assignment]
        return _carried(merged, values)

    # ---------------------------------------------------------------------------------- polarization
    ea = field_3d(a.jones[None, :], a.e1[None, :], a.e2[None, :])[0]
    eb = field_3d(b.jones[None, :], b.e1[None, :], b.e2[None, :])[0]
    eta_pol = polarization_overlap(ea, eb)
    fields["polarization_overlap"] = eta_pol
    if mixed:
        bases["polarization_overlap"] = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(
            Assumed(input=AssumedInput.UNSTATED_PHASE_AS_INTERFACE, at=mixed),))
    if eta_pol <= cfg.energy_conservation_tol and share > 0.0:  # (a pair its sources keep incoherent forms no fringes anyway)
        issue(ErrorCode.ORTHOGONAL_POLARIZATION, Status.WARNING, "the two polarizations are orthogonal: no fringes form",
              polarization_overlap=eta_pol)
    # phase difference phi_a - phi_b of the two fields (propagation k0*OPL plus interface phases), common phase reference.
    # With E_j = J_j exp(i k0 OPL_j): phi_a - phi_b = k0 (OPL_a - OPL_b) - arg(<E_a, E_b>) where <E_a, E_b> = conj(E_a).E_b.
    k0 = 2.0 * math.pi / (lam_nm * NM_TO_MM)
    overlap = cvdot(ea, eb)  # (no BLAS / fused multiply-add: identical bits in every runtime)
    # (orthogonal polarizations -- the test of ORTHOGONAL_POLARIZATION, whatever the sources' coherence --: the overlap is
    # rounding noise and its argument the direction of that noise, no phase of the fields; the phase difference is stated on
    # the convention arg(<E_a, E_b>) = 0, k0 (OPL_a - OPL_b) wrapped, an ESTIMATE naming the overlap below its limit)
    orthogonal = eta_pol <= cfg.energy_conservation_tol
    overlap_phase = float(atan2(overlap.imag, overlap.real)) if overlap != 0 and not orthogonal else 0.0
    phase_difference = _wrap(k0 * opd - overlap_phase)
    orthogonal_basis = ValueBasis(kind=ValueKind.ESTIMATE, validity=(ValidityCheck(
        parameter="polarization_overlap", unit="1", minimum=cfg.energy_conservation_tol, observed=eta_pol, within=False,
        setting="energy_conservation_tol",
        code=ErrorCode.ORTHOGONAL_POLARIZATION.value if share > 0.0 else None),)) if orthogonal else None
    open_ids: list[str] = []
    phase_basis: ValueBasis | None = None
    if not phase_open(a.delays, b.delays):
        fields["phase_difference_rad"] = reported_phase(phase_difference)  # reported in steps of 1e-6 rad (math.phase)
        if orthogonal_basis is not None:
            bases["phase_difference_rad"] = orthogonal_basis
    elif not opd_open:
        # (a pass one beam makes and the other does not has no stated phase at the line: the value is on the trace's default
        # interface phases there, the passes named in its basis; optics/interference/coherence.py)
        open_ids = open_pass_ids(a.delays, b.delays)
        phase_basis = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(
            Assumed(input=AssumedInput.UNSTATED_PHASE_AS_INTERFACE, at=tuple(open_ids)),))
        fields["phase_difference_rad"] = reported_phase(phase_difference)
        bases["phase_difference_rad"] = merge_bases(phase_basis, orthogonal_basis, monotone=None)  # type: ignore[assignment]
    else:
        # (and an element on the way states no optical path: the traced length is no path length, so no phase is formed)
        open_ids = open_pass_ids(a.delays, b.delays)
        issue(ErrorCode.PHASE_UNDEFINED, Status.NOT_APPLICABLE,
              "a pass one beam makes and the other does not states neither its phase nor its optical path, so the "
              "pair's phase difference, optical path difference and combined power have no value",
              source_ids=sorted({a.source_id, b.source_id}), passes=open_ids[:OPEN_PASSES_LISTED],
              suggested_action="model the element with its thickness and material")
        # (two nearly collinear beams of known coherence also get the sentence on their combined power's range, below)
        fields["phase_statement"] = _phase_statement(open_ids, ranged=False, path_known=False)

    # ----------------------------------------------------------------------- spatial envelopes / irradiance
    ga, gb = a.gstate, b.gstate
    if not (ga.valid and gb.valid):
        why = ga if not ga.valid else gb
        issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
              "a Gaussian beam state is outside the model scope: spatial overlap, visibility and irradiance are not available",
              reason=(why.reason.value if why.reason else None))
        status = worst([i.status for i in issues], empty=Status.PASS)
        return InterferenceResult(status=status, issue_codes=tuple(sorted({i.code.value for i in issues})), **base, **fields, value_basis=carried(fields)), issues
    u, v = a.plane_u, a.plane_v
    for g, arr in ((ga, a), (gb, b)):
        cos_a = abs(float(dot(arr.direction, nu)))
        tilt_dz = 2.0 * max(g.radius("x"), g.radius("y")) * math.sqrt(max(0.0, 1.0 - cos_a * cos_a)) / max(cos_a, cfg.zero_vector_tol)
        z_r = min(g.rayleigh("x"), g.rayleigh("y"))
        if tilt_dz > cfg.gaussian_plane_axial_spread_fraction * z_r:
            issue(ErrorCode.GAUSSIAN_PLANE_AXIAL_SPREAD, Status.WARNING,
                  "the beam footprint spans a significant part of the Rayleigh range along its axis: constant-w evaluation on the plane is approximate",
                  axial_spread_mm=tilt_dz, rayleigh_range_mm=z_r, path_id=arr.path_id)
    cos_a1, cos_a2 = abs(float(dot(d1, nu))), abs(float(dot(d2, nu)))
    if min(cos_a1, cos_a2) <= cfg.zero_vector_tol:
        issue(ErrorCode.GRAZING_INCIDENCE, Status.OUT_OF_MODEL_SCOPE, "a beam is parallel to the observation plane")
        return InterferenceResult(status=Status.OUT_OF_MODEL_SCOPE, issue_codes=tuple(sorted({i.code.value for i in issues})), **base, **fields, value_basis=carried(fields)), issues
    s1, s2 = _s_matrix_plane(ga, u, v), _s_matrix_plane(gb, u, v)
    c1, c2 = np.array(a.uv), np.array(b.uv)
    wx1, wy1 = ga.radius("x"), ga.radius("y")
    wx2, wy2 = gb.radius("x"), gb.radius("y")
    amp1 = 2.0 * a.power_W / (math.pi * wx1 * wy1)  # transverse-flux intensity amplitude (W/mm^2)
    amp2 = 2.0 * b.power_W / (math.pi * wx2 * wy2)
    m = s1 + s2
    det_m = det2(m)
    if det_m <= cfg.interference_det_tol:
        issue(ErrorCode.NUMERICAL_FAILURE, Status.FAIL, "the summed envelope matrix is singular", det=det_m)
        return InterferenceResult(status=Status.FAIL, issue_codes=tuple(sorted({i.code.value for i in issues})), **base, **fields, value_basis=carried(fields)), issues
    bvec = matvec(s1, c1) + matvec(s2, c2)
    x_star = solve2(m, bvec)
    const = float(dot(c1, matvec(s1, c1)) + dot(c2, matvec(s2, c2)) - dot(bvec, x_star))
    integral = math.sqrt(amp1 * amp2) * math.pi / math.sqrt(det_m) * float(exact_exp(-const))
    p1_plane = amp1 * math.pi / (2.0 * math.sqrt(det2(s1)))  # integral of I1 over plane area = P1 / cos(alpha1)
    p2_plane = amp2 * math.pi / (2.0 * math.sqrt(det2(s2)))
    eta_sp = float(min(1.0, max(0.0, integral / math.sqrt(p1_plane * p2_plane))))
    fields["spatial_overlap_ratio"] = eta_sp
    fields["overlap_effective_area_mm2"] = 2.0 * math.pi / math.sqrt(det_m)
    i1 = amp1 * float(exact_exp(-2.0 * float(dot(x_star - c1, matvec(s1, x_star - c1)))))
    i2 = amp2 * float(exact_exp(-2.0 * float(dot(x_star - c2, matvec(s2, x_star - c2)))))
    fields["peak_position_uv_mm"] = (float(x_star[0]), float(x_star[1]))
    fields["mean_irradiance_W_per_mm2"] = i1 + i2
    fields["intensity_balance"] = min(i1, i2) / max(i1, i2) if max(i1, i2) > 0.0 else 0.0
    if eta_sp <= cfg.interference_min_overlap:
        # informational: a plane may receive spatially separate beams (two spots, a folded path's out and back crossings);
        # overlap-dependent targets judge the numbers themselves (a visibility minimum fails on its own)
        issue(ErrorCode.BEAMS_NOT_OVERLAPPING, Status.NOT_APPLICABLE, "the two beams do not overlap on the observation plane",
              spatial_overlap_ratio=eta_sp)
    if gamma is not None:
        v_local = (2.0 * math.sqrt(i1 * i2) / (i1 + i2)) * eta_pol * gamma if (i1 + i2) > 0 else 0.0
        balance_int = 2.0 * math.sqrt(p1_plane * p2_plane) / (p1_plane + p2_plane)
        fields["fringe_visibility_peak"] = v_local
        fields["fringe_visibility"] = balance_int * eta_sp * eta_pol * gamma
        fields["peak_irradiance_W_per_mm2"] = i1 + i2 + 2.0 * math.sqrt(i1 * i2) * eta_pol * gamma
        coherence_basis = bases.get("coherence_factor")
        if coherence_basis is not None:  # (each is the degree of coherence on the convention, scaled)
            bases.update(dict.fromkeys(("fringe_visibility_peak", "fringe_visibility", "peak_irradiance_W_per_mm2"), coherence_basis))
        period = fields.get("fringe_period_nm")
        beam_diam_nm = 2.0 * max(wx1, wy1, wx2, wy2) / NM_TO_MM
        if period is None or period > cfg.interference_collinear_period_ratio * beam_diam_nm:
            # the power the plane receives: on an absorbing plane each beam's detected power (its power on the active area)
            # the power the plane receives from each beam (none from a beam outside its acceptance cone)
            pa = a.detected_power_W if a.detected_power_W is not None else (0.0 if a.within_acceptance is False else a.power_W)
            pb = b.detected_power_W if b.detected_power_W is not None else (0.0 if b.within_acceptance is False else b.power_W)
            if counter_propagating:
                fields["combined_power_collinear_W"] = pa + pb
            elif not opd_open or not open_ids:
                # (at the traced phase: the default interface phases where a differing pass states none, the basis says so)
                if complex_gamma is not None:
                    carrier = cis(phase_difference)
                    fields["combined_power_collinear_W"] = pa + pb + (
                        2.0 * math.sqrt(pa * pb) * eta_pol * eta_sp * in_phase(complex_gamma, carrier.real, carrier.imag))
                else:
                    fields["combined_power_collinear_W"] = pa + pb + (
                        2.0 * math.sqrt(pa * pb) * eta_pol * gamma * eta_sp * cis(phase_difference).real)
                # (its degree of coherence's entry: the stored one, or -- behind bounded coating delays, where the value is
                # at the traced delay -- the derived ESTIMATE, ``results/basis.py`` ``_coherence``)
                coherence_entry = bases.get("coherence_factor")
                if coherence_entry is None and fields.get("coherence_factor_min") is not None and fields.get("coherence_factor_max") is not None:
                    coherence_entry = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO),))
                power_basis = merge_bases(coherence_entry, phase_basis, monotone=None)
                if power_basis is not None:
                    bases["combined_power_collinear_W"] = power_basis
            if open_ids and not counter_propagating:
                # the phase is open: the power's range over every phase, cos(phi) in [-1, 1], at the largest degree of
                # coherence the coatings' group-delay bounds allow (1 where a differing pass bounds no delay) and the largest
                # polarization overlap the unstated phases allow (1 where a pass mixes s and p)
                g_max = fields.get("coherence_factor_max", share if unstated else gamma)
                largest = 2.0 * math.sqrt(pa * pb) * (1.0 if mixed else eta_pol) * g_max * eta_sp
                fields.update(combined_power_collinear_min_W=max(0.0, pa + pb - largest), combined_power_collinear_max_W=pa + pb + largest)
                if opd_open:
                    fields["phase_statement"] = _phase_statement(open_ids, ranged=True, path_known=False)
            elif (not counter_propagating and isinstance(model, LineCoherence) and model.open_length
                  and (complex_gamma is not None or bound)):
                # (one line states its coherence length, the other none, and the paths' group-delay dispersions differ or
                # coatings of bounded group delay differ: a shorter length of the unstated line, or a delay those bounds
                # allow, moves the phase of the degree of coherence as well as its magnitude, so the power takes every
                # phase of the cross term up to the largest degree of coherence, ``coherence_factor_max``; the polarized
                # share where a differing pass bounds no delay)
                g_max = fields.get("coherence_factor_max", share if unstated else gamma)
                largest = 2.0 * math.sqrt(pa * pb) * eta_pol * g_max * eta_sp
                fields.update(combined_power_collinear_min_W=max(0.0, pa + pb - largest), combined_power_collinear_max_W=pa + pb + largest)
    elif counter_propagating:
        # (the power through the plane of beams crossing it in opposite directions has no cross term, whatever their
        # coherence: their powers add)
        pa = a.detected_power_W if a.detected_power_W is not None else (0.0 if a.within_acceptance is False else a.power_W)
        pb = b.detected_power_W if b.detected_power_W is not None else (0.0 if b.within_acceptance is False else b.power_W)
        fields["combined_power_collinear_W"] = pa + pb
    status = worst([i.status for i in issues], empty=Status.PASS)
    if status is Status.PASS and any(i.code is ErrorCode.BEAMS_NOT_OVERLAPPING for i in issues):
        status = Status.NOT_APPLICABLE  # two separate spots: there is no interference to pass
    return InterferenceResult(status=status, issue_codes=tuple(sorted({i.code.value for i in issues})), **base, **fields, value_basis=carried(fields)), issues
