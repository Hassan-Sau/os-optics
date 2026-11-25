"""Per-surface beam exposure and power-rating comparison (``validation/exposures.py``).

The reference optic is a 3 mm N-BK7 window (S1 at 60 mm behind an ideal f = 100 mm thin lens, a 1064 nm Gaussian beam of
2 W). The Gaussian radius at each window surface is computed here independently with ABCD q propagation (free space, the
thin lens, the flat air-glass interface q' = n q), and from it the peak irradiance 2P / (pi w^2), the linear power density
P / d and the pulsed peak fluence 2E / (pi w^2) the Engine must report. Every power rating below is fixture data declared
in the test Project (status DECLARED), never catalog data.
"""

from __future__ import annotations

import copy
import math

from decimal import Decimal

import pytest

from os_optics import ValidationRequest, validate
from os_optics._vendor.shared_contracts.common.decimal_string import decimal_to_string
from os_optics.local import LocalRunError, lower_project, run_optical_project
from os_optics.results import results_from_engine

from declared_helpers import declared_component, nbk7, sequence, stop_model, surface, thin_lens_model, with_models
from local_helpers import NM, plane, project, rayleigh_range, source
from test_declared_physics import _n

WAVELENGTH_NM = 1064.0
POWER_W = 2.0
W0_MM = 0.5
SOURCE_X, LENS_X, WINDOW_X, THICKNESS = -200.0, 0.0, 60.0, 3.0
FOCAL_MM = 100.0
FIXTURE = "test fixture rating, not product data"


def _dec(value: float) -> str:
    """A plain decimal string of ``value`` (10 significant digits)."""
    return decimal_to_string(Decimal(f"{value:.10g}"))


def _rating(regime: str, value: float, unit: str, wavelength_nm: float = WAVELENGTH_NM, **extra) -> dict:
    """A declared fixture rating; a ``J/cm2`` / ``W/cm2`` value is stated as a peak value (``value_definition`` ``PEAK``)
    with the window's beams' size as its test beam (``window_test_beam_mm``, 1/e^2) unless ``value_definition`` or
    ``beam_diameter_mm`` is given, and its test polarization as linear (the source's) unless ``polarization_condition``
    is given (``None``: not stated)."""
    rating = {"regime": regime, "value": _dec(value), "unit": unit, "wavelength_nm": _dec(wavelength_nm), "status": "DECLARED",
              "basis": FIXTURE}
    if unit != "W/cm":
        extra.setdefault("value_definition", "PEAK")
        if "beam_diameter_mm" not in extra:
            extra.update(beam_diameter_mm=window_test_beam_mm(wavelength_nm), beam_diameter_definition="1/E2")
    extra.setdefault("polarization_condition", "linear")
    rating.update({k: (_dec(v) if isinstance(v, float) else v) for k, v in extra.items() if v is not None})
    return rating


def window_test_beam_mm(wavelength_nm: float = WAVELENGTH_NM) -> float:
    """A test beam the window's beams match: the 1/e^2 diameter on its rear face S2 (``_independent``); the beams on the
    front face S1 -- the incident one and the rear face's reflection -- lie within 5 % of it."""
    return 2.0 * _independent(wavelength_nm)["W/S2"][1]


def _window(ratings: list[dict] | None) -> dict:
    model = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, f"{THICKNESS:g}")], ["N-BK7"])}
    if ratings:
        model["power_ratings"] = ratings
    return model


def _document(ratings: list[dict] | None, *, pulse: dict | None = None, wavelength_nm: float = WAVELENGTH_NM, stop: bool = False,
              targets: list[dict] | None = None) -> dict:
    laser = source(position=(SOURCE_X, 0.0, 0.0), wavelength_nm=wavelength_nm, waist_mm=W0_MM, power_w=POWER_W)
    if pulse is not None:
        laser["source_model"]["pulse"] = pulse
    components = [laser, declared_component("L", "lens", "thin", (LENS_X, 0.0, 0.0)), declared_component("W", "window", "win", (WINDOW_X, 0.0, 0.0))]
    models = [thin_lens_model("thin", f"{FOCAL_MM:g}"), _window(ratings)]
    if stop:
        components.insert(1, declared_component("STOP", "aperture", "stop", (-20.0, 0.0, 0.0)))
        models.append(stop_model("stop", plane_z_mm="0", opening_radius_mm="0.3", edge_radius_mm="12.7"))
    document = project(components, [plane("screen", (WINDOW_X + 20.0, 0.0, 0.0), radius_mm=20.0)], targets)
    return with_models(document, models, [nbk7()])


def _independent(wavelength_nm: float = WAVELENGTH_NM) -> dict[str, tuple[float, float]]:
    """(incident power W, 1/e^2 radius mm) at the window's S1 and S2 by ABCD q propagation, written out here."""
    lam = wavelength_nm * NM
    q = complex(LENS_X - SOURCE_X, rayleigh_range(W0_MM, wavelength_nm))  # waist at the source
    q = 1.0 / (1.0 / q - 1.0 / FOCAL_MM)  # ideal thin lens
    q_s1 = q + (WINDOW_X - LENS_X)
    n = _n("N-BK7", wavelength_nm)
    q_s2 = n * q_s1 + THICKNESS  # flat interface into glass: q' = (n2 / n1) q
    w_s1 = math.sqrt(lam / math.pi * abs(q_s1) ** 2 / q_s1.imag)
    w_s2 = math.sqrt(lam / (math.pi * n) * abs(q_s2) ** 2 / q_s2.imag)
    t_fresnel = 1.0 - ((n - 1.0) / (n + 1.0)) ** 2  # normal incidence, uncoated
    return {"W/S1": (POWER_W, w_s1), "W/S2": (POWER_W * t_fresnel, w_s2)}


def _independent_q(wavelength_nm: float = WAVELENGTH_NM) -> dict[str, tuple[complex, float]]:
    """(q in mm, the medium's index) at the window's S1 (in air) and S2 (in the glass), as ``_independent``."""
    q = complex(LENS_X - SOURCE_X, rayleigh_range(W0_MM, wavelength_nm))
    q = 1.0 / (1.0 / q - 1.0 / FOCAL_MM)
    q_s1 = q + (WINDOW_X - LENS_X)
    n = _n("N-BK7", wavelength_nm)
    return {"W/S1": (q_s1, 1.0), "W/S2": (n * q_s1 + THICKNESS, n)}


def _peak_w_per_cm2(power: float, w: float) -> float:
    return 2.0 * power / (math.pi * w * w) * 100.0


def window_written_out(value: float, w: float, q: complex, n_beam: float, n_max: float, ghosts: list[tuple[float, float]],
                       unknown: float, wavelength_nm: float = WAVELENGTH_NM) -> float:
    """The window bound (``validation/window_bound.py``) of a round traced beam at normal incidence on a plane, written out:
    peak ``value`` (W/cm2 or J/cm2), 1/e^2 radius ``w`` (mm), complex parameter ``q`` (mm) in its medium of index
    ``n_beam``; ``ghosts``: coaxial beams of its line (peak value, radius); ``unknown``: the light of unknown place,
    ``sum sqrt(P_i) + c f sqrt(P*)`` (sqrt W, or sqrt J). For each window ``chi = exp(-s rho^2)``: the beam's mean against
    it, ``sqrt(V) pi w^2 / (1 + s)`` over ``pi w^2 / s``, less each ghost's ``sqrt(V_g) pi / (s / w^2 + 1 / w_g^2)``,
    less ``10 H unknown`` with ``H^2`` the window's overlap with each side's plane waves weighted by ``1 / cos``: inside
    ``|K| <= sigma sqrt(Lambda)`` at most ``integral chi^2 / c_A`` (Parseval), outside the Gaussian tail over the disk;
    ``Lambda = s / w^2 + beta^2 w^2 / s``, ``beta = k |Re 1/q|`` the beam's wavefront phase."""
    lam = wavelength_nm * NM
    k_beam = 2.0 * math.pi * n_beam / lam
    k_max = 2.0 * math.pi * max(n_max, n_beam) / lam
    sides = ((k_beam, k_beam), (k_max, k_max)) if n_beam < max(n_max, n_beam) * (1.0 - 1e-9) else (
        (k_beam, k_beam), (2.0 * math.pi / lam, k_max))
    beta = k_beam * abs(q.real) / (q.real * q.real + q.imag * q.imag)
    scales = [0.0625]
    while scales[-1] < 1.5e6:
        scales.append(scales[-1] * math.sqrt(2.0))
    best = 0.0
    for s in scales:
        a = s / (w * w)
        lam_big = a + beta * beta / a
        squared = math.pi * w * w / (2.0 * s)
        h = 0.0
        for k_low, k_high in sides:
            side = math.inf
            for sigma in (6.0, 8.0, 10.0, 12.0):
                kappa = sigma * math.sqrt(lam_big)
                tail = math.pi * k_high * k_high / (2.0 * a * a) * math.exp(-kappa * kappa / (2.0 * lam_big))
                if kappa < k_low:
                    side = min(side, squared / math.sqrt(1.0 - (kappa / k_low) ** 2) + tail)
            h = max(h, math.sqrt(side))
        taken = sum(math.sqrt(v) * min(math.pi / (a + 1.0 / (wg * wg)), math.pi / a) for v, wg in ghosts)
        mean = (math.sqrt(value) * math.pi * w * w / (1.0 + s) - taken - 10.0 * h * unknown) / (math.pi / a)
        if mean > 0.0:
            best = max(best, mean * mean)
    return best


def linear_window_written_out(value: float, w: float, ghosts: list[tuple[float, float]], unknown: float, waist: float,
                              kappa: float) -> float:
    """The ``P / d`` window bound (``window_bound.linear_lower``) of a round traced beam at normal incidence on a plane,
    written out: ``P / d`` ``value`` (W/cm), 1/e^2 radius ``w`` (mm), coaxial ghosts (``P / d``, radius); ``unknown``:
    ``sum sqrt(P_i) + c f sqrt(P*)`` (sqrt W) of light of unknown place whose diffraction-limited waist is ``waist`` (mm);
    ``kappa``: the ``P / d`` of 1 W over a 1 mm radius (5 for the 1/e^2 diameter). For each window ``exp(-s rho^2)``
    (``a = s / w^2``, ``integral chi = pi / a``): a beam of that light of narrower radius ``u`` crossing the plane from one
    side covers at most a strip ``sqrt(pi) u sqrt(2 u^2 + waist^2) / waist`` times ``sqrt(pi / a)`` (a line across
    ``chi``) of the window; it takes away at most ``sqrt(kappa / u)`` times the smaller of that and ``pi / a``, largest where
    the two meet -- ``u^2 = (sqrt(waist^4 + 8 c^2) - waist^2) / 4`` with ``c = waist / sqrt(a)`` -- or at ``u = waist``."""
    scales = [0.0625]
    while scales[-1] < 1.5e6:
        scales.append(scales[-1] * math.sqrt(2.0))
    best = 0.0
    for s in scales:
        a = s / (w * w)
        window = math.pi / a
        c = waist / math.sqrt(a)
        u = max(waist, math.sqrt((math.sqrt(waist ** 4 + 8.0 * c * c) - waist * waist) / 4.0))
        strip = math.sqrt(math.pi) * u * math.sqrt(2.0 * u * u + waist * waist) / waist * math.sqrt(math.pi / a)
        taken = sum(math.sqrt(v) * min(math.pi / (a + 1.0 / (wg * wg)), window) for v, wg in ghosts)
        mean = (math.sqrt(value) * math.pi * w * w / (1.0 + s) - taken - unknown * math.sqrt(kappa / u) * min(strip, window)) / window
        if mean > 0.0:
            best = max(best, mean * mean)
    return best


def _surface(run, surface_id: str):
    return next(s for s in run.result.exposures.surfaces if s.surface_id == surface_id)


def _element(run, element_id: str):
    return next(e for e in run.result.exposures.elements if e.element_id == element_id)


def _peak_of(hit) -> float:
    return hit.peak_irradiance_W_per_cm2


def _open_counts(reflectances, per_leg, turning, plane):
    """(m, N, R) of ``open_paths``."""
    legs = (len(reflectances) if turning is None else turning) + 1
    return (legs if plane else 1), len(reflectances) + len(per_leg), sum(reflectances) + legs * sum(per_leg)


def _ordered_sums(weights: list[float]) -> list[float]:
    """``j! e_j(weights)`` for j = 0 .. len(weights): the sum over ordered sequences of ``j`` distinct entries of their
    weights' product, from the coefficients of ``prod (1 + w t)`` (multiplied out here)."""
    coefficients = [1.0]
    for w in weights:
        coefficients = [a + w * b for a, b in zip(coefficients + [0.0], [0.0] + coefficients)]
    return [math.factorial(j) * c for j, c in enumerate(coefficients)]


def _open_fields(reflectances, per_leg, turning):
    """(amplitude, pieces, power) of ``open_paths`` after each number of reflections: the ordered sums over sequences of
    distinct reflecting surfaces of ``m_s sqrt(R_s)``, ``m_s`` and ``m_s R_s``."""
    legs = (len(reflectances) if turning is None else turning) + 1
    surfaces = [(r, 1) for r in reflectances] + [(r, legs) for r in per_leg]
    return (_ordered_sums([m * math.sqrt(r) for r, m in surfaces]), _ordered_sums([float(m) for _, m in surfaces]),
            _ordered_sums([m * r for r, m in surfaces]))


def open_paths(*reflectances: float, per_leg: tuple[float, ...] = (), turning: int | None = None, plane: bool = False) -> float:
    """The most field one piece of left-out light brings to a point over its open paths, as a multiple of one crossing of
    its whole power, written out. Between two reflections it meets each lens and each refracting surface between media of
    different index at most once and travels in at most ``L = T + 1`` straight legs (``T`` those lenses and refracting
    surfaces, ``turning``; by default one per surface of ``reflectances``), each crossing a plane at most once. So it meets
    a lens or a refracting surface between different media at most ``m = 1`` time, a plane surface or an observation plane
    at most ``m = L`` times (``plane``: the field there). ``reflectances``: the power reflectance bounds of the refracting
    surfaces between different media that may reflect it, over every angle and side (a bare face 1, the reflectance just
    below its critical angle); ``per_leg``: those of the plane mirrors, splitters and attenuators (an uncoated mirror 1). Each reflection takes at most its surface's bound of
    each meeting: after ``j`` reflections the light carries at most ``min(1, R^j)`` of its power, ``R`` the sum of the
    bounds, each times its surface's ``m``; its pieces, from different sequences of distinct surfaces (no surface reflects
    it twice on an open path), are mutually coherent, so their fields add: at most the sum over those sequences of the
    product of ``m_s sqrt(R_s)``, and at most the square root of the number of pieces (the product of ``m_s``) times
    their power (the product of ``m_s R_s``, at most 1); split between the two sides each time: ``sqrt(2) m sum_{j=0}^{N}
    a_j`` with ``a_j`` the smaller, at least 2 (the piece once from either side)."""
    m, n, total = _open_counts(reflectances, per_leg, turning, plane)
    amplitude, pieces, power = _open_fields(reflectances, per_leg, turning)
    terms = [min(amplitude[j], math.sqrt(pieces[j] * min(1.0, power[j]))) for j in range(1, n + 1)]
    return max(2.0, math.sqrt(2.0) * m * (1.0 + sum(terms)))


def open_power(*reflectances: float, per_leg: tuple[float, ...] = (), turning: int | None = None, plane: bool = False) -> float:
    """The most power one piece of left-out light brings across a transparent plane or onto a surface over its open paths,
    as a multiple of its power, written out (``open_paths``): ``m sum_{j=0}^{N} min(1, R^j)``, at least 2."""
    m, n, total = _open_counts(reflectances, per_leg, turning, plane)
    return max(2.0, m * sum(min(1.0, total ** j) for j in range(n + 1)))


#: The reference window's two bare faces (``open_paths``).
WINDOW_PATHS = open_paths(1.0, 1.0)


def faint_amplitude(run, unit: str, *, fwhm: bool = False, repetition_rate_hz: float | None = None,
                    material: str | None = "N-BK7", returns: float | None = None, surface=None,
                    paths: float = WINDOW_PATHS) -> float:
    """The bound on the field of the light the run left out as negligible (``exposures.faint_light``), written out: the power
    its pieces could carry together in phase, P = (sum of the square roots of their powers)^2 (``in_phase_power_W``: they
    are mutually coherent), focused to the smallest waist a beam of its wavelength can have, w = lambda / (pi n) (n: ``material``'s
    index -- the window's faces meet its glass -- or 1), its value there in the rating's ``unit`` (2P / (pi w^2);
    P / (2 w), or P / (w sqrt(2 ln 2)) for an FWHM diameter; 2E / (pi w^2) with E = P / f_rep), and its square root times
    the most its open paths' crossings bring (``paths``, ``open_paths`` of the scene's reflecting surfaces: by default the
    reference window's two bare faces). The diffraction limit 2 pi n^2 P / lambda^2 is the most any field of power P
    crossing a plane from one side reaches (the Cauchy-Schwarz bound over its plane-wave spectrum). Light that bounces
    between two surfaces facing each other (the two faces of an uncoated window) returns to them again and again and is
    sent on to every surface: that field times the factor (``returns``; by default the one ``surface`` -- else every
    surface of the run, all alike -- reports, which ``test_group_delay_and_returns.window_returns`` writes out for a
    window; 1 without a pair)."""
    exposures = run.result.exposures if hasattr(run, "result") else run.exposures
    (line,) = exposures.faint_light
    n = 1.0 if material is None else _n(material, line.wavelength_nm)
    if returns is None:
        found = {surface.faint_light_return_factor} if surface is not None else {
            x.faint_light_return_factor for x in exposures.surfaces if x.faint_light_return_factor is not None}
        assert len(found) <= 1, found
        returns = 1.0 if not found or None in found else found.pop()
    w = line.wavelength_nm * NM / (math.pi * n)
    power = line.in_phase_power_W
    if unit == "W/cm2":
        value = 2.0 * power / (math.pi * w * w) * 100.0
    elif unit == "J/cm2":
        assert repetition_rate_hz is not None
        value = 2.0 * (power / repetition_rate_hz) / (math.pi * w * w) * 100.0
    elif fwhm:
        value = power / (w * math.sqrt(2.0 * math.log(2.0))) * 10.0
    else:
        value = power / (2.0 * w) * 10.0
    return paths * math.sqrt(value) * returns


def _unknown_root(run, surface, unit: str, *, repetition_rate_hz: float | None = None, returns: float | None = None,
                  paths: float = WINDOW_PATHS, **_) -> float:
    """The light of unknown place on ``surface`` for the window bound, written out: the left-out light's open paths'
    crossings times its returns times the square root of its in-phase power (``W/cm2``) or pulse energy (``J/cm2``):
    ``c f sqrt(P*)`` (``faint_amplitude``'s factors)."""
    exposures = run.result.exposures if hasattr(run, "result") else run.exposures
    (line,) = exposures.faint_light
    if returns is None:
        returns = surface.faint_light_return_factor or 1.0
    amount = line.in_phase_power_W if unit in ("W/cm2", "W/cm") else line.in_phase_power_W / repetition_rate_hz
    return paths * returns * math.sqrt(amount)


def _traced(run, comparison, surface, value=_peak_of, reference: bool = True, **faint) -> float:
    """The traced beam's own exposure on the uncoated window's front face S1, checked against the comparison's bounds: the
    window's own reflections (its rear face's, arriving from inside the glass, and their returns) meet S1 on the beam's
    axis at normal incidence. Light of the beam's line, they are coherent with it whichever side they arrive from (a
    standing wave inside the glass), and so is the light the run left out as negligible (``faint_amplitude``, ``A``): with
    the phases open the exposure lies between (sqrt v_t - sum sqrt v_g - A)^2 and (sqrt v_t + sum sqrt v_g + A)^2, v the
    values of the traced hit and the untraced hits in the rating's quantity (``value``). (Before the verifier's findings
    the returns were added as flux from the other side, and the left-out light was not counted.) The comparison reports the
    traced beam's own exposure as ``exposure_value`` beside those bounds, with the entry that names what it does not count:
    the window's reflections (light not followed) and the light left out as negligible -- an estimate, since both are
    coherent with the beam and may raise or lower the exposure."""
    (hit,) = surface.hits
    traced = value(hit)
    roots = sum(math.sqrt(value(g)) for g in surface.untraced_hits) + faint_amplitude(run, comparison.unit, surface=surface, **faint)
    assert surface.untraced_hits and "FAINT_LIGHT_BOUNDED" in surface.reason_codes
    assert comparison.exposure_value == pytest.approx(traced, rel=1e-12)
    entry = comparison.value_basis["exposure_value"]
    assert entry.kind.value == "ESTIMATE" and {(o.part.value, o.code) for o in entry.omitted} >= {
        ("BELOW_NEGLIGIBLE_LEVEL", "FAINT_LIGHT_BOUNDED"), ("NOT_FOLLOWED", "UNTRACED_LIGHT_BOUNDED")}
    point = (math.sqrt(traced) - roots) ** 2 if math.sqrt(traced) > roots else 0.0
    if comparison.unit == "W/cm" and reference:
        # (P / d: and its window bound, the light of unknown place as beams crossing the face from one side)
        unknown = _unknown_root(run, surface, comparison.unit, **faint)
        (line,) = (run.result.exposures if hasattr(run, "result") else run.exposures).faint_light
        waist = line.wavelength_nm * NM / (math.pi * _n("N-BK7", line.wavelength_nm))
        fwhm = faint.get("fwhm", False)
        window = linear_window_written_out(traced, hit.beam_radius_x_mm, [(value(g), g.beam_radius_x_mm) for g in surface.untraced_hits],
                                           unknown, waist, 10.0 / (math.sqrt(2.0 * math.log(2.0)) if fwhm else 2.0))
        assert comparison.exposure_lower_bound == pytest.approx(max(point, window), rel=1e-6)
    elif comparison.unit == "W/cm" or not reference:
        # (another beam than the reference window's: at least the bound at its centre, below the beam's own value)
        assert point * (1.0 - 1e-9) <= comparison.exposure_lower_bound < traced
    else:
        # (and the window bound: the beam against a window over its area, which light of unknown place cannot undo)
        q, n_beam = _independent_q()[surface.surface_id]
        unknown = _unknown_root(run, surface, comparison.unit, **faint)
        window = window_written_out(traced, hit.beam_radius_x_mm, q, n_beam, _n("N-BK7", WAVELENGTH_NM),
                                    [(value(g), g.beam_radius_x_mm) for g in surface.untraced_hits], unknown)
        assert comparison.exposure_lower_bound == pytest.approx(max(point, window), rel=1e-6)
        assert window > point
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(traced) + roots) ** 2, rel=1e-9)
    return traced


# ------------------------------------------------------------------------------------------- exposure records


def test_exposure_records_match_independent_abcd_peak_irradiance_and_linear_density():
    run = run_optical_project(_document(None))
    expected = _independent()
    for surface_id, (power, w) in expected.items():
        exposure = _surface(run, surface_id)
        (hit,) = exposure.hits
        assert hit.incident_power_W == pytest.approx(power, rel=1e-12)
        assert hit.beam_radius_x_mm == pytest.approx(w, rel=1e-9) and hit.beam_radius_y_mm == pytest.approx(w, rel=1e-9)
        assert hit.aoi_deg == pytest.approx(0.0, abs=1e-9)
        assert hit.peak_irradiance_W_per_cm2 == pytest.approx(_peak_w_per_cm2(power, w), rel=1e-9)
        assert hit.peak_irradiance_surface_W_per_cm2 == pytest.approx(_peak_w_per_cm2(power, w), rel=1e-9)  # normal incidence
        assert hit.linear_power_density_W_per_cm == pytest.approx(power / (2.0 * w) * 10.0, rel=1e-9)
        assert hit.footprint_semi_axes_mm == pytest.approx((w, w), rel=1e-9)
        assert hit.pulse_energy_J is None and hit.peak_fluence_J_per_cm2 is None and hit.status.value == "PASS"
        assert exposure.complete and exposure.rating_status == "RATING_MISSING" and exposure.ratings == ()
    report = run.result.exposures
    assert report.trace_complete and report.untraced_light_included is False
    assert report.tolerances == {"wavelength_rel": 0.02, "pulse_duration_rel": 0.10, "repetition_rate_rel": 0.10,
                                 "beam_diameter_rel": 0.05, "polarization_fraction": 0.01, "unlocated_light_share": 0.01}
    assert _element(run, "W").rating_status == "RATING_MISSING"
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in run.result.issues)


def test_a_tilted_surface_spreads_the_footprint_and_lowers_the_surface_peak_by_cos_aoi():
    tilt = math.radians(30.0)
    document = _document(None)
    (window,) = [c for c in document["components"] if c["instance_id"] == "W"]
    # turn the window by 30 deg about the design z axis (local +Z = design +X -> (cos 30, sin 30, 0))
    c, s = math.cos(tilt / 2.0), math.sin(tilt / 2.0)
    window["nominal_pose"]["quaternion_wxyz"] = [0.5 * (c - s), 0.5 * (c - s), 0.5 * (c + s), 0.5 * (c + s)]
    run = run_optical_project(document)
    (hit,) = _surface(run, "W/S1").hits
    _, w = _independent()["W/S1"]
    assert hit.aoi_deg == pytest.approx(30.0, abs=1e-9)
    assert hit.peak_irradiance_surface_W_per_cm2 == pytest.approx(hit.peak_irradiance_W_per_cm2 * math.cos(tilt), rel=1e-12)
    major, minor = hit.footprint_semi_axes_mm
    assert major == pytest.approx(w / math.cos(tilt), rel=1e-6) and minor == pytest.approx(w, rel=1e-6)


# ------------------------------------------------------------------------------------------- CW ratings


def test_a_cw_rated_window_with_matched_illumination_is_within_its_ratings():
    expected = _independent()
    peak_s1 = _peak_w_per_cm2(*expected["W/S1"])
    peak_s2 = _peak_w_per_cm2(*expected["W/S2"])
    density_s1 = expected["W/S1"][0] / (2.0 * expected["W/S1"][1]) * 10.0
    ratings = [_rating("CW", 4.0 * peak_s2, "W/cm2"),
               _rating("CW", 4.0 * density_s1, "W/cm", beam_diameter_mm=window_test_beam_mm(), beam_diameter_definition="1/E2")]
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document(ratings, targets=targets))
    s1 = _surface(run, "W/S1")
    irradiance, linear = s1.ratings
    assert (irradiance.status, irradiance.conditions_match, irradiance.mismatches, irradiance.unverified_conditions) == (
        "WITHIN_RATING", True, (), ())
    assert _traced(run, irradiance, s1) == pytest.approx(peak_s1, rel=1e-9)
    # (bounded: WITHIN because the upper bound is; the traced beam's ratio to the rated value is reported beside it)
    assert irradiance.utilization == pytest.approx(peak_s1 / irradiance.rating_value, rel=1e-9)
    assert irradiance.value_basis["utilization"].kind.value == "ESTIMATE" and irradiance.value_basis["utilization"].assumed == ()
    assert _traced(run, linear, s1, lambda h: h.linear_power_density_W_per_cm) == pytest.approx(density_s1, rel=1e-9)
    assert linear.status == "WITHIN_RATING"
    assert any("1/e^2 diameter 2 w" in note for note in linear.notes)
    assert irradiance.origin == "model win" and irradiance.rating_status == "DECLARED" and irradiance.basis == FIXTURE
    element = _element(run, "W")
    comparisons = [c for s in run.result.exposures.surfaces if s.element_id == "W" for c in s.ratings]
    # (not exact: the element's number is the largest reported exposure / threshold, with its entry -- beside the bounds)
    assert element.rating_status == "WITHIN_RATING"
    assert element.max_utilization == max(c.exposure_value / c.rating_value for c in comparisons)
    assert element.value_basis["max_utilization"].kind.value == "ESTIMATE"
    # the element's utilization lies between the largest lower and the largest upper bound of its comparisons
    assert element.utilization_lower_bound == max(c.exposure_lower_bound / c.rating_value for c in comparisons)
    assert element.utilization_upper_bound == max(c.exposure_upper_bound / c.rating_value for c in comparisons) < 1.0
    (target,) = run.result.targets
    assert target.status.value == "PASS" and target.value == pytest.approx(element.utilization_lower_bound, rel=1e-12)
    assert not any(i.code.value == "POWER_RATING_EXCEEDED" for i in run.result.issues)


def test_a_stated_test_beam_diameter_is_compared_with_the_beams_on_the_surface():
    """A CW W/cm rating established with a 0.5 mm 1/e^2 test beam: the beam on both faces (0.443 and 0.424 mm across,
    1/e^2, at normal incidence) is no larger than the test beam, so the threshold for it is at least the rated value and
    an exposure within the rating is within it (WITHIN_RATING, a utilization target PASSes; it could never FAIL on this
    rating). With a 0.2 mm test beam the beam here is larger: a W/cm rating has no established relationship for a larger
    beam (the linear power density removes only the first-order thermal dependence on the size), so the rating does not
    apply -- CONDITIONS_MISMATCH, its exposure still reported, never an exceedance."""
    expected = _independent()
    density_s1 = expected["W/S1"][0] / (2.0 * expected["W/S1"][1]) * 10.0
    rating = _rating("CW", 4.0 * density_s1, "W/cm", beam_diameter_mm=0.5, beam_diameter_definition="1/E2")
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document([rating], targets=targets))
    s1 = _surface(run, "W/S1")
    (comparison,) = s1.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert (beam.state, beam.effect, beam.rated, beam.rated_low, beam.rated_high) == (
        "DIFFERS", "THRESHOLD_AT_LEAST_RATED", 0.5, 0.5, 0.5)
    # (the smallest and largest 1/e^2 size of every beam on S1: the traced beam and the window's own reflections)
    assert beam.design_low <= 2.0 * expected["W/S1"][1] <= beam.design_high <= 0.5
    assert (comparison.applicability, comparison.status, comparison.conditions_match) == ("THRESHOLD_AT_LEAST_RATED", "WITHIN_RATING", False)
    assert comparison.mismatches == (beam.message,) and comparison.unverified_conditions == ()
    assert _traced(run, comparison, s1, lambda h: h.linear_power_density_W_per_cm) == pytest.approx(density_s1, rel=1e-9)
    assert _element(run, "W").rating_status == "WITHIN_RATING" and _element(run, "W").utilization_lower_bound == 0.0
    (target,) = run.result.targets
    assert target.status.value == "PASS" and "bounds the utilization from above only" in target.message
    small = _rating("CW", 4.0 * density_s1, "W/cm", beam_diameter_mm=0.2, beam_diameter_definition="1/E2")
    run = run_optical_project(_document([small], targets=targets))
    (comparison,) = _surface(run, "W/S1").ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert (beam.state, beam.effect) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP") and "W/cm rating" in beam.message
    assert (comparison.applicability, comparison.status) == ("NOT_APPLICABLE", "CONDITIONS_MISMATCH")
    assert comparison.exposure_lower_bound is not None and comparison.exposure_upper_bound is not None
    (target,) = run.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("POWER_RATING_CONDITIONS_MISMATCH",))
    # (no rating applies: the largest ratio to the rated value is reported with the condition that differs, never graded)
    assert target.assessment.value == "INDETERMINATE" and target.value == _element(run, "W").max_utilization
    assert target.value_basis["value"].kind.value == "ESTIMATE"
    assert ("beam_diameter", False, "POWER_RATING_CONDITIONS_MISMATCH") in {
        (v.parameter, v.within, v.code) for v in target.value_basis["value"].validity}


def test_an_fwhm_linear_density_rating_uses_the_fwhm_diameter():
    power, w = _independent()["W/S1"]
    run = run_optical_project(_document([_rating("CW", 1.0e6, "W/cm", beam_diameter_mm=1.0, beam_diameter_definition="FWHM")]))
    s1 = _surface(run, "W/S1")
    (comparison,) = s1.ratings

    def fwhm(hit) -> float:  # P / d with d = w sqrt(2 ln 2), the FWHM diameter of the narrower axis
        return hit.incident_power_W / (min(hit.beam_radius_x_mm, hit.beam_radius_y_mm) * math.sqrt(2.0 * math.log(2.0))) * 10.0

    assert _traced(run, comparison, s1, fwhm, fwhm=True) == pytest.approx(power / (w * math.sqrt(2.0 * math.log(2.0))) * 10.0,
                                                                  rel=1e-9)
    assert any("FWHM diameter" in note for note in comparison.notes)


def test_an_exceeded_cw_rating_is_a_structured_warning_and_fails_a_utilization_target():
    power, w = _independent()["W/S1"]
    peak = _peak_w_per_cm2(power, w)
    rating_value = 0.5 * peak
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document([_rating("CW", rating_value, "W/cm2")], targets=targets))
    s1 = _surface(run, "W/S1")
    comparison = s1.ratings[0]
    assert _traced(run, comparison, s1) == pytest.approx(peak, rel=1e-9)
    # decided by the lower bound: the window's own reflections cancelling the beam at their worst still exceed the rating
    least = comparison.exposure_lower_bound
    assert comparison.status == "EXCEEDS_RATING" and least > rating_value
    assert comparison.utilization == least / comparison.rating_value
    issues = [i for i in run.result.issues if i.code.value == "POWER_RATING_EXCEEDED"]
    assert {i.location for i in issues} == {"surface:W/S1", "surface:W/S2"}
    s1 = next(i for i in issues if i.location == "surface:W/S1")
    assert s1.status.value == "NOT_APPLICABLE"  # a notice: it never changes the run's status
    ctx = s1.numerical_context
    assert (ctx["element_id"], ctx["surface_id"], ctx["regime"], ctx["unit"]) == ("W", "W/S1", "CW", "W/cm2")
    assert ctx["exposure_value"] == least and ctx["rating_value"] == pytest.approx(rating_value, rel=1e-9)
    assert ctx["exposure_is_lower_bound"] is True and ctx["rating_status"] == "DECLARED"
    assert _element(run, "W").rating_status == "EXCEEDS_RATING"
    (target,) = run.result.targets
    assert target.status.value == "FAIL" and target.value >= least / rating_value and "POWER_RATING_EXCEEDED" in target.issue_codes
    assert run.result.status.value == "FAIL"


# ------------------------------------------------------------------------------------------- condition matching


def test_a_pulsed_only_rating_does_not_apply_to_a_cw_source():
    pulsed = _rating("PULSED", 10.0, "J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0)
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document([pulsed], targets=targets))
    (comparison,) = _surface(run, "W/S1").ratings
    assert comparison.status == "CONDITIONS_MISMATCH" and not comparison.conditions_match
    assert comparison.exposure_value is None and comparison.utilization is None
    assert any(m.startswith("regime: the rating is PULSED") for m in comparison.mismatches)
    assert _element(run, "W").rating_status == "CONDITIONS_MISMATCH"
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN" and target.issue_codes == ("POWER_RATING_CONDITIONS_MISMATCH",)


def test_a_matched_pulsed_rating_compares_the_independent_peak_fluence():
    """A single shot (``total_pulses`` 1) against a 2-on-1 rating (two pulses on each tested site at one level) whose
    other conditions it meets: S1 receives the forward pass and the back-reflection from S2, one or two pulses per site,
    no more than the test, so the threshold is at least the rated value and the fluence below it is within. A 1000-pulse
    rating that does not state how its pulses were applied to each tested site is reported, not decided; one stated as
    1000-on-1 is not verifiable where the source states no total."""
    pulse = {"repetition_rate_hz": 1000.0, "pulse_duration_s": 1.0e-8, "duration_definition": "FWHM", "total_pulses": 1}
    power, w = _independent()["W/S1"]
    fluence = 2.0 * (power / 1000.0) / (math.pi * w * w) * 100.0
    ratings = [_rating("PULSED", 3.0 * fluence, "J/cm2", pulse_duration_s=1.05e-8, repetition_rate_hz=1050.0, pulse_count=2,
                       test_protocol="S_ON_1"),
               _rating("PULSED", 3.0 * fluence, "J/cm2", pulse_duration_s=2.0e-8),
               _rating("CW", 1.0e9, "W/cm2"),
               _rating("PULSED", 3.0 * fluence, "J/cm2", pulse_duration_s=1.05e-8, repetition_rate_hz=1050.0, pulse_count=1000)]
    run = run_optical_project(_document(ratings, pulse=pulse))
    (hit,) = _surface(run, "W/S1").hits
    assert hit.pulse_energy_J == pytest.approx(power / 1000.0, rel=1e-12)
    assert hit.peak_fluence_J_per_cm2 == pytest.approx(fluence, rel=1e-9)
    matched, long_pulse, cw, counted = _surface(run, "W/S1").ratings
    assert matched.status == "WITHIN_RATING" and matched.applicability == "THRESHOLD_AT_LEAST_RATED"
    count = next(c for c in matched.conditions if c.condition == "pulse_count")
    assert (count.state, count.effect, count.design_low, count.design_high) == ("DIFFERS", "THRESHOLD_AT_LEAST_RATED", 1.0, 2.0)
    assert _traced(run, matched, _surface(run, "W/S1"), lambda h: h.peak_fluence_J_per_cm2,
                   repetition_rate_hz=1000.0) == pytest.approx(fluence, rel=1e-9)
    assert long_pulse.status == "CONDITIONS_MISMATCH" and any(m.startswith("pulse duration") for m in long_pulse.mismatches)
    assert cw.status == "CONDITIONS_MISMATCH" and any("pulsed" in m for m in cw.mismatches)
    # a rating over 1000 pulses that does not state how they were applied to each tested site: no relation to one pulse
    assert counted.status == "CONDITIONS_NOT_VERIFIED" and not counted.conditions_match and counted.applicability == "NOT_VERIFIED"
    assert counted.unverified_conditions == ("pulse count: the rating states 1000 pulses, not how they were applied to each tested "
                                             "site (at one fluence level, ramped, raster), so its value's relation to the pulses a "
                                             "site receives here is not established",)
    count = next(c for c in counted.conditions if c.condition == "pulse_count")
    assert (count.state, count.reason, count.rated, count.unit) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING", 1000, "pulses")
    # (reported, not decided: the ratio to the rated value as an estimate that assumes the count's protocol)
    assert counted.utilization == pytest.approx(fluence / counted.rating_value, rel=1e-9)
    assert counted.value_basis["utilization"].kind.value == "ESTIMATE" and counted.exposure_upper_bound < counted.rating_value
    assert _surface(run, "W/S1").rating_status == "WITHIN_RATING"  # (decided by the rating whose conditions are all checked)
    # the same count stated as 1000-on-1, the source stating no pulse.total_pulses
    no_total = {k: v for k, v in pulse.items() if k != "total_pulses"}
    s_on_1 = _rating("PULSED", 3.0 * fluence, "J/cm2", pulse_duration_s=1.05e-8, repetition_rate_hz=1050.0, pulse_count=1000,
                     test_protocol="S_ON_1")
    (counted,) = _surface(run_optical_project(_document([s_on_1], pulse=no_total)), "W/S1").ratings
    assert counted.status == "CONDITIONS_NOT_VERIFIED" and counted.applicability == "NOT_VERIFIED"
    assert counted.unverified_conditions == ("pulse count: the rating states 1000 pulses on each tested site at one fluence level "
                                             "(S-on-1); SRC states no pulse.total_pulses (how many pulses a source sends while "
                                             "the optics stay in place)",)
    count = next(c for c in counted.conditions if c.condition == "pulse_count")
    assert (count.state, count.reason, count.rated, count.unit) == ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING", 1000, "pulses")
    # (reported, not decided: the count cannot be checked against a source that states no total -- a check without a result)
    assert counted.utilization == pytest.approx(fluence / counted.rating_value, rel=1e-9)
    ratio = counted.value_basis["utilization"]
    assert ratio.kind.value == "ESTIMATE" and ("pulse_count", None) in {(v.parameter, v.within) for v in ratio.validity}
    assert counted.exposure_upper_bound < counted.rating_value


@pytest.mark.parametrize(("source_nm", "match"), [(1064.0 * 1.019, True), (1064.0 * 1.021, False)], ids=["inside-2pct", "outside-2pct"])
def test_the_wavelength_condition_is_two_percent_of_the_rating_wavelength(source_nm: float, match: bool):
    run = run_optical_project(_document([_rating("CW", 1.0e9, "W/cm2", wavelength_nm=1064.0)], wavelength_nm=source_nm))
    (comparison,) = _surface(run, "W/S1").ratings
    assert comparison.conditions_match is match
    assert comparison.status == ("WITHIN_RATING" if match else "CONDITIONS_MISMATCH")


def test_a_missing_rating_is_distinct_from_an_exceedance():
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document(None, targets=targets))
    assert _element(run, "W").rating_status == "RATING_MISSING" and _element(run, "W").max_utilization is None
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN" and target.issue_codes == ("POWER_RATING_MISSING",) and target.value is None


def test_an_invalid_gaussian_state_makes_the_exposure_unknown_never_zero():
    """A severely truncated beam has no valid Gaussian state: its exposure is unknown (never 0), and its size is not known
    either, so the rating's test beam cannot be checked against it: not verified, not decided."""
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    run = run_optical_project(_document([_rating("CW", 1.0e9, "W/cm2")], stop=True, targets=targets))
    s1 = _surface(run, "W/S1")
    (hit,) = s1.hits
    assert hit.status.value == "UNKNOWN" and hit.reason == "GAUSSIAN_SEVERE_TRUNCATION"
    assert hit.incident_power_W > 0.0 and hit.peak_irradiance_W_per_cm2 is None and hit.linear_power_density_W_per_cm is None
    assert s1.peak_irradiance_W_per_cm2 is None and not s1.complete and "GAUSSIAN_SEVERE_TRUNCATION" in s1.reason_codes
    (comparison,) = s1.ratings
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert (beam.state, beam.reason) == ("NOT_VERIFIABLE", "BEAM_SIZE_UNKNOWN")
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.exposure_value is None and comparison.exposure_lower_bound is None
    (target,) = run.result.targets
    assert target.status.value == "UNKNOWN" and target.value is None


def test_lines_of_one_source_are_summed_per_surface_and_each_line_must_match_the_rating():
    document = _document([_rating("CW", 1.0e9, "W/cm2", wavelength_nm=1064.0)])
    model = document["components"][0]["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": 1064.0, "power_w": 1.5}, {"wavelength_nm": 532.0, "power_w": 0.5}]
    run = run_optical_project(document)
    s1 = _surface(run, "W/S1")
    assert sorted(h.wavelength_nm for h in s1.hits) == [532.0, 1064.0]
    assert s1.incident_power_W == pytest.approx(2.0, rel=1e-12)
    assert s1.peak_irradiance_W_per_cm2 == pytest.approx(sum(h.peak_irradiance_W_per_cm2 for h in s1.hits), rel=1e-12)
    (comparison,) = s1.ratings
    assert comparison.status == "CONDITIONS_MISMATCH" and any(m.startswith("wavelength: 532 nm") for m in comparison.mismatches)


def test_an_incomplete_trace_leaves_the_exposure_of_its_line_unbounded(tmp_path):
    """Power of unknown fate (the per-path interaction limit) may reach any surface: no upper bound. Being light of the same
    source line it is also coherent with the traced beam there and may cancel it: no lower bound above 0 either (before
    the coherent combination the traced value was reported as a lower bound). Its polarization is not known either, so a
    rating stated for linear light is not verified there."""
    lowered = lower_project(_document([_rating("CW", 1.0e9, "W/cm2")]))
    cfg = lowered.request.configuration.numerical.model_copy(update={"max_interactions_per_path": 2})
    request = lowered.request.model_copy(update={"configuration": lowered.request.configuration.model_copy(update={"numerical": cfg})})
    result = validate(ValidationRequest.model_validate_json(request.model_dump_json()))
    report = result.exposures
    assert not report.trace_complete and report.unknown_fate_power_W > 0.0
    s1 = next(s for s in report.surfaces if s.surface_id == "W/S1")
    assert not s1.complete and "INCOMPLETE_TRACE" in s1.reason_codes and "MAX_INTERACTIONS" in s1.reason_codes
    (comparison,) = s1.ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN")
    assert comparison.status == "CONDITIONS_NOT_VERIFIED"
    assert comparison.exposure_lower_bound is None and comparison.exposure_upper_bound is None
    # (the traced beam's own exposure is reported with what it does not count: light at the interaction limit)
    (hit,) = s1.hits
    entry = comparison.value_basis["exposure_value"]
    assert comparison.exposure_value == hit.peak_irradiance_W_per_cm2 and entry.kind.value == "ESTIMATE"
    assert ("BUDGET", "MAX_INTERACTIONS") in {(o.part.value, o.code) for o in entry.omitted}
    results = results_from_engine(result)
    assert results.exposures("W")["available"] is True


def test_declared_ratings_of_a_compound_member_apply_to_its_own_surfaces():
    """A rating declared on member A of a compound unit applies to A's surfaces only (B's have none). The light the two
    members' parallel faces may return between them, left out as negligible, is of unknown size and state and may carry
    nearly all of the light there, so A's rating -- although it states the beam's own size and polarization -- is not
    verified."""
    from declared_helpers import compound_model, cylinder

    # (the member's test beam: the source beam's 1/e^2 diameter at the unit, WINDOW_X - SOURCE_X from its waist, no lens)
    beam_mm = 2.0 * W0_MM * math.sqrt(1.0 + ((WINDOW_X - SOURCE_X) / rayleigh_range(W0_MM, WAVELENGTH_NM)) ** 2)
    window = _window([_rating("CW", 1.0e9, "W/cm2", beam_diameter_mm=beam_mm, beam_diameter_definition="1/E2")])
    unit = compound_model("unit", [("A", "win", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), ("B", "plain", (0.0, 0.0, 10.0), (1.0, 0.0, 0.0, 0.0))],
                          cylinder("26", "0", "14"))
    plain = {"model_id": "plain", "model": sequence([surface(1, "0"), surface(2, f"{THICKNESS:g}")], ["N-BK7"])}
    laser = source(position=(SOURCE_X, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=W0_MM, power_w=POWER_W)
    document = with_models(project([laser, declared_component("U", "other_optical", "unit", (WINDOW_X, 0.0, 0.0))],
                                   [plane("screen", (WINDOW_X + 40.0, 0.0, 0.0), radius_mm=20.0)]),
                           [unit, window, plain], [nbk7()])
    run = run_optical_project(document)
    statuses = {s.surface_id: s.rating_status for s in run.result.exposures.surfaces}
    assert statuses == {"U/A-S1": "CONDITIONS_NOT_VERIFIED", "U/A-S2": "CONDITIONS_NOT_VERIFIED", "U/B-S1": "RATING_MISSING",
                        "U/B-S2": "RATING_MISSING"}
    unverified = {(k.condition, k.reason) for s in run.result.exposures.surfaces for c in s.ratings for k in c.conditions
                  if k.state == "NOT_VERIFIABLE"}
    assert unverified == {("beam_diameter", "BEAM_SIZE_UNKNOWN"), ("polarization", "LIGHT_STATE_UNKNOWN")}
    (component,) = [c for c in lowered_components(document) if c.instance_id == "U"]
    assert component.power_ratings[0].surface_ids == ("A-S1", "A-S2") and component.power_ratings[0].origin == "model win (member A)"


def lowered_components(document):
    return lower_project(document).request.project.components


# ------------------------------------------------------------------------------------------- engine input and results


def test_projects_without_ratings_or_pulses_keep_their_engine_input():
    document = _document(None)
    request = lower_project(document).request.model_dump(mode="json")
    assert all("power_ratings" not in c for c in request["project"]["components"])
    assert all("pulse" not in c["source"] for c in request["project"]["components"] if c["source"] is not None)


def test_results_queries_and_exports_carry_the_exposures(tmp_path):
    from os_optics.local import json_api
    from os_optics.results import load_run

    run_optical_project(_document([_rating("CW", 1.0e-3, "W/cm2")]), output_dir=str(tmp_path / "run"))
    results = load_run(tmp_path / "run")
    doc = results.exposures()
    assert doc["available"] and {s["surface_id"] for s in doc["surfaces"]} == {"L/lens", "W/S1", "W/S2"}
    assert results.describe()["available"]["exposures"] == 3
    rows, _ = results.table("exposures", element="W")
    traced = [r for r in rows if r["light"] == "traced"]
    assert [r["surface_id"] for r in traced] == ["W/S1", "W/S2"] and traced[0]["rating_status"] == "EXCEEDS_RATING"
    assert {r["light"] for r in rows} == {"traced", "untraced"}  # the window's own ghosts on its faces
    csv = results.export("power-ratings", "csv").decode()
    assert csv.splitlines()[0].startswith("surface_id,element_id,rating_index") and "EXCEEDS_RATING" in csv
    files = {p.name: p.read_bytes() for p in (tmp_path / "run").iterdir() if p.is_file()}
    answer = json_api.run_results(files, {"kind": "exposures", "element": "W"})
    assert answer["ok"] and [s["surface_id"] for s in answer["result"]["surfaces"]] == ["W/S1", "W/S2"]
    missing = json_api.run_results(files, {"kind": "exposures", "element": "nope"})
    assert missing["ok"] is False and missing["error"]["code"] == "ELEMENT_NOT_FOUND"


def test_cli_results_exposures(tmp_path):
    import json
    import subprocess
    import sys

    run_optical_project(_document([_rating("CW", 1.0e9, "W/cm2")]), output_dir=str(tmp_path / "run"))
    proc = subprocess.run([sys.executable, "-m", "os_optics", "results", "exposures", str(tmp_path / "run"), "--element", "W"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert [e["element_id"] for e in doc["elements"]] == ["W"] and doc["elements"][0]["rating_status"] == "WITHIN_RATING"
    export = subprocess.run([sys.executable, "-m", "os_optics", "export", "exposures", str(tmp_path / "run"), "--format", "jsonl"],
                            capture_output=True, text=True, timeout=120)
    rows = [json.loads(line) for line in export.stdout.splitlines()]
    assert export.returncode == 0 and len([r for r in rows if r["light"] == "traced"]) == 3


def test_a_rating_on_an_unknown_surface_is_refused_by_the_engine_contract():
    from os_optics.contracts import EngineeringComponentSnapshot, PowerRating

    with pytest.raises(ValueError, match="unknown surfaces"):
        EngineeringComponentSnapshot(instance_id="X", power_ratings=(PowerRating(regime="CW", value=1.0, unit="W/cm2", wavelength_nm=1064.0,
                                                                                status="DECLARED", surface_ids=("S9",)),))
    with pytest.raises(ValueError, match="PULSED"):
        PowerRating(regime="PULSED", value=1.0, unit="W/cm2", wavelength_nm=1064.0, status="DECLARED")


def test_a_pulse_longer_than_its_period_is_refused():
    document = _document(None, pulse={"repetition_rate_hz": 1.0e9, "pulse_duration_s": 2.0e-9, "duration_definition": "FWHM"})
    with pytest.raises(LocalRunError):
        lower_project(copy.deepcopy(document))
