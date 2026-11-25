"""Mutually coherent beams on one surface in the per-surface exposure (review, finding P1-a;
``validation/exposures.py``).

Two branches of one coherent source recombined on a surface add their fields, not their irradiances: two equal branches
in phase reach 4 I, in antiphase 0. The exposure leaves the relative phases open and bounds the combination: at most
``sum v + 2 sum gamma sqrt(v_i v_j)`` (``(sum sqrt v)^2`` for full coherence), at least ``max(0, sqrt v_k - sum sqrt
v_j)^2``; light that cannot interfere (other lines, other coherence groups) adds. Light of one line arriving from the two
sides of a surface is coherent too (a standing wave: verifier finding P2), while the incident power, a flux, has no cross
term between the sides. Light the trace leaves out as negligible still adds its field: its whole power, focused to the
diffraction limit and arriving from either side, bounds it (``faint_amplitude``); where that
alone keeps a comparison from being decided, the run follows the light deeper. The hand values below come from the
Gaussian beam written out here (q propagation, ``2 P / (pi w^2)``) and from the Fresnel coefficients; the Engine's own
two-beam interference analysis at an observation plane in front of the face confirms which port is bright. Every power
rating here is fixture data declared in the test project, never product data: a peak value with linear polarization (the
sources') and the beam's own 1/e^2 diameter at the rated surface as its test beam (``_beam_mm``, Gaussian propagation).
"""

from __future__ import annotations

import math

import pytest

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, builders as B, validate
from os_optics.contracts import PowerRating, ProjectSnapshot, Target
from os_optics.local import run_optical_project

from helpers import rayleigh_range, run
from test_golden_interference import _mzi_project, bk7_index
from test_power_exposure_ratings import (_document, _independent_q, _n, _rating, _surface, _unknown_root, faint_amplitude,
                                         open_paths, window_written_out)

FIXTURE = "test fixture rating, not product data"
MZI_W0_MM, MZI_LAMBDA_NM, MZI_POWER_W = 0.8, 532.0, 1e-3
#: The rated interferometer's surfaces that may reflect its left-out light (``open_paths``): the rated window's two bare
#: faces, and the two splitters and the two uncoated fold mirrors, met once per leg (their reflectance bounds add up to more
#: than 1), on the window's faces.
MZI_PATHS = open_paths(1.0, 1.0, per_leg=(0.5, 1.0, 1.0, 0.5))
#: The attenuated window's: the window's two bare faces and the pick-off splitter (reflectance 1 - T), on the window's faces.
PICKOFF_PATHS = open_paths(1.0, 1.0, per_leg=(1.0,))


def _beam_mm(waist_mm: float, path_mm: float, wavelength_nm: float) -> float:
    """A source's 1/e^2 diameter ``path_mm`` of air from its waist: 2 w0 sqrt(1 + (z / z_R)^2)."""
    return 2.0 * waist_mm * math.sqrt(1.0 + (path_mm / rayleigh_range(waist_mm, wavelength_nm)) ** 2)


def _cw_rating(value: float, beam_mm: float, wavelength_nm: float = MZI_LAMBDA_NM) -> PowerRating:
    """A fixture rating whose value is a peak irradiance (``value_definition`` ``PEAK``), for linear light, measured with a
    test beam of ``beam_mm`` (1/e^2)."""
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength_nm, status="DECLARED", basis=FIXTURE,
                       value_definition="PEAK", beam_diameter_mm=beam_mm, beam_diameter_definition="1/E2", polarization="LINEAR",
                       polarization_condition="linear")


def _rated_mzi(rating_W_per_cm2: float, *, port: str = "x", window_thickness_mm: float = 0.0,
               coherence_length_mm: float | None = None) -> ProjectSnapshot:
    """The review's reproducer (review360_coherent.py): the balanced Mach-Zehnder fixture of ``test_golden_interference``
    (1 mW, 532 nm, w0 0.8 mm, coherence length 10 m, two NPBS and two fold mirrors, arms of 2 x 100 mm), a 1 mm N-BK7
    window RATED (CW rating ``rating_W_per_cm2``) 39.5 mm behind the second splitter on port ``x`` (the bright one) or
    ``y`` (the dark one), an observation plane 0.001 mm in front of its face, and a power_rating_utilization target
    (maximum 1). ``window_thickness_mm``: the fixture's window WB in arm B; ``coherence_length_mm``: the source's."""
    project = _mzi_project(window_thickness_mm)
    components = list(project.components)
    if coherence_length_mm is not None:
        source = components[0]
        components[0] = source.model_copy(update={"source": source.source.model_copy(update={"coherence_length_mm": coherence_length_mm})})
    if port == "x":
        window = B.window("RATED", thickness_mm=1, clear_radius_mm=12.7, material="N-BK7", position=(140.0, 100.0, 20.0))
        pre = B.observation_plane("pre", position=(139.499, 100.0, 20.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    else:
        window = B.window("RATED", thickness_mm=1, clear_radius_mm=12.7, material="N-BK7", position=(100.0, 140.0, 20.0),
                          axis=(0.0, 1.0, 0.0))
        pre = B.observation_plane("pre", position=(100.0, 139.499, 20.0), normal=(0.0, 1.0, 0.0), radius_mm=12.7)
    # (the test beam: each branch's size at the window, 50 + 2 x 100 + 39.5 mm from the waist, as ``_hand_branch_peak``)
    branch_mm = _beam_mm(MZI_W0_MM, 50.0 + 100.0 + 100.0 + 39.5, MZI_LAMBDA_NM)
    window = window.model_copy(update={"power_ratings": (_cw_rating(rating_W_per_cm2, branch_mm),)})
    target = Target(target_id="rating", device_id="RATED", metric="power_rating_utilization", maximum=1.0)
    return project.model_copy(update={"components": (*components, window), "observations": (*project.observations, pre),
                                      "targets": (target,)})


def _front(result):
    return next(s for s in result.exposures.surfaces if s.surface_id == "RATED/front")


#: The conditions light of unknown size and state leaves not verifiable where it may carry more than 1 % of the light: the
#: rating's stated test beam and polarization.
UNKNOWN_SIZE_AND_STATE = {("beam_diameter", "BEAM_SIZE_UNKNOWN"), ("polarization", "LIGHT_STATE_UNKNOWN")}


def _unverified(comparison) -> set:
    return {(c.condition, c.reason) for c in comparison.conditions if c.state == "NOT_VERIFIABLE"}


def _hand_branch_peak() -> float:
    """One branch at the window's face: P / 4 (each splitter halves it), w(z) of the 0.8 mm waist after z = 50 + 2 x 100 +
    39.5 mm, peak 2 P / (pi w^2) in W/cm^2."""
    z = 50.0 + 100.0 + 100.0 + 39.5
    w = MZI_W0_MM * math.sqrt(1.0 + (z / rayleigh_range(MZI_W0_MM, MZI_LAMBDA_NM)) ** 2)
    return 2.0 * (MZI_POWER_W / 4.0) / (math.pi * w * w) * 100.0


# ------------------------------------------------------------------------------------------------ the review case


def test_the_review_mzi_bright_port_is_not_within_a_rating_below_its_coherent_peak():
    """review360_coherent.py: rating 0.07 W/cm2. The summed irradiance of the two branches (0.0494) was reported as the
    exposure (upper bound 0.0515 with the ghosts) and the target PASSed; the bright port's peak is 4 x 0.0247 = 0.0989.
    The light left out as negligible near the window, of unknown size and state, may carry most of the peak irradiance
    there: the rating's stated test beam and polarization cannot be verified for it, so the rating is not verified -- never
    within -- and its exposure bounds stand."""
    result = run(_rated_mzi(0.07))
    surface = _front(result)
    single = _hand_branch_peak()
    coherent = 4.0 * single  # (sqrt I + sqrt I)^2: the two equal fields in phase
    assert [h.peak_irradiance_W_per_cm2 for h in surface.hits] == pytest.approx([single, single], rel=1e-9)
    # the Engine's own interference analysis 0.001 mm in front of the face: the two branches in phase (all the power)
    (pair,) = [i for i in result.interference if i.observation_id == "pre"]
    assert pair.combined_power_collinear_W == pytest.approx(MZI_POWER_W, rel=1e-12)
    assert pair.peak_irradiance_W_per_mm2 * 100.0 == pytest.approx(coherent, rel=1e-6)
    # the traced beams' values on the surface are their in-phase maximum (incident power: the collinear superposition's)
    assert surface.peak_irradiance_W_per_cm2 == pytest.approx(coherent, rel=1e-9)
    assert surface.incident_power_W == pytest.approx(MZI_POWER_W, rel=1e-12)
    # one coherent group: the two branches and the window's own reflections arriving from inside the glass
    (group,) = surface.coherent_groups
    assert group.members[:2] == ("traced:L0#0.0:0", "traced:L0#0.1:0") and group.incidence == "BOTH_SIDES"
    assert all(m.startswith("untraced:") for m in group.members[2:]) and len(group.members) == 2 + len(surface.untraced_hits)
    assert group.coherence_group == "source:L0" and group.wavelength_nm == MZI_LAMBDA_NM
    assert not surface.complete and "COHERENT_BEAMS_BOUNDED" in surface.reason_codes and "FAINT_LIGHT_BOUNDED" in surface.reason_codes
    (comparison,) = surface.ratings
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
    assert not comparison.conditions_match and comparison.status == "CONDITIONS_NOT_VERIFIED"  # review: WITHIN_RATING
    assert comparison.exposure_value is None and comparison.exposure_lower_bound is None  # (antiphase would cancel)
    # every member in phase (the 10 m coherence length: gamma = 1 - O(1e-7) over the window's millimetres), and the light
    # the run left out as negligible at its diffraction limit, in phase with them (the window's faces meet its glass)
    roots = sum(math.sqrt(h.peak_irradiance_W_per_cm2) for h in (*surface.hits, *surface.untraced_hits))
    faint = faint_amplitude(result, "W/cm2", surface=surface, paths=MZI_PATHS)
    assert surface.faint_light_peak_W_per_cm2 == pytest.approx(faint * faint, rel=1e-12)
    assert comparison.exposure_upper_bound == pytest.approx((roots + faint) ** 2, rel=1e-9)
    assert comparison.exposure_upper_bound >= coherent > 0.07
    assert surface.rating_status == "CONDITIONS_NOT_VERIFIED"
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and "COHERENT_BEAMS_BOUNDED" in target.message  # review: PASS
    assert result.status.value != "PASS"


def test_a_rating_above_the_coherent_bound_is_within_it():
    """A rating above both faces' upper bounds would be within it; the light left out as negligible, of unknown size and
    state, keeps the rating's stated test beam and polarization from being verified on these faces, so it is not
    verified and the target not decided -- with the exposure still within the value."""
    probe = run(_rated_mzi(1e9))
    uppers = [s.ratings[0].exposure_upper_bound for s in probe.exposures.surfaces if s.element_id == "RATED"]
    assert min(uppers) >= 4.0 * _hand_branch_peak()  # (both faces: the two branches in phase, and the window's ghosts)
    rating = 1.001 * max(uppers)
    result = run(_rated_mzi(rating))
    for surface in (s for s in result.exposures.surfaces if s.element_id == "RATED"):
        (comparison,) = surface.ratings
        assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
        assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.exposure_upper_bound <= rating
    (target,) = result.targets
    assert target.status.value == "UNKNOWN"


def test_the_dark_port_is_bounded_by_the_same_interval_its_phases_decide_within():
    """The dark port: the same two branches in antiphase (the Engine's interference analysis: no light). With the phases
    left open the exposure lies between 0 and 4 I, so a rating inside that interval is not decided either way (nor
    verified: light of unknown size and state, left out as negligible, may reach the face)."""
    result = run(_rated_mzi(0.07, port="y"))
    surface = _front(result)
    single = _hand_branch_peak()
    (pair,) = [i for i in result.interference if i.observation_id == "pre"]
    assert pair.combined_power_collinear_W < 1e-12 * MZI_POWER_W  # dark: the fields cancel
    (comparison,) = surface.ratings
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.exposure_lower_bound is None
    assert surface.peak_irradiance_W_per_cm2 == pytest.approx(4.0 * single, rel=1e-9)  # the traced branches' in-phase maximum
    assert comparison.exposure_upper_bound > 4.0 * single  # (+ the window's own ghosts)


# ----------------------------------------------------------------------------------- degree of coherence and groups


def test_branches_beyond_the_coherence_length_add_their_irradiances():
    """The fixture's window WB (2 mm N-BK7) in arm B delays that arm's envelope by the group path difference (n_g - 1) t =
    1.10 mm (the phase path difference (n - 1) t is 1.04 mm; a finite bandwidth decorrelates by the envelope delay). With a
    source coherence length of 0.01 mm the branches are incoherent: collinear and co-propagating, their relative delay
    changes across the face by far less than 1 mm (the window is a plate: its envelope delay changes with the rays' angles
    only), so their degree of coherence is 0 anywhere -- their irradiances add, exactly (their peaks coincide). With L_c =
    the group path difference the degree of coherence is that of the dispersed Gaussian spectrum at one coherence length
    -- (1 + b^2)^(-1/4) exp(-1 / (1 + b^2)), b = 2 x 2 mm x k2 / L_c^2, about 1/e -- at the chief rays and hardly more anywhere
    on the face: the partially coherent value."""
    n = bk7_index(MZI_LAMBDA_NM)
    lam, step = MZI_LAMBDA_NM * 1e-3, 1e-5
    n_g = n - lam * (bk7_index((lam + step) * 1e3) - bk7_index((lam - step) * 1e3)) / (2.0 * step)
    coarse = 1e-3
    curvature = (bk7_index((lam + coarse) * 1e3) - 2.0 * n + bk7_index((lam - coarse) * 1e3)) / (coarse * coarse)
    k2 = lam ** 3 * curvature * 1e-3 / (2.0 * math.pi)  # mm^2 per mm of N-BK7
    opd = (n_g - 1.0) * 2.0
    budget = NumericalConfig(max_paths=4096, max_trace_steps=16384)  # (every ghost of the two windows followed)
    incoherent = run(_rated_mzi(0.07, window_thickness_mm=2.0, coherence_length_mm=0.01), numerical=budget)
    surface = _front(incoherent)
    a, b = (h.peak_irradiance_W_per_cm2 for h in surface.hits)
    assert "COHERENT_BEAMS_BOUNDED" not in surface.reason_codes  # (no traced pair interferes)
    assert surface.peak_irradiance_W_per_cm2 == a + b
    (comparison,) = surface.ratings
    assert comparison.exposure_lower_bound is None or comparison.exposure_lower_bound <= a + b <= comparison.exposure_upper_bound
    partial = run(_rated_mzi(0.07, window_thickness_mm=2.0, coherence_length_mm=opd), numerical=budget)
    surface = _front(partial)
    a, b = (h.peak_irradiance_W_per_cm2 for h in surface.hits)
    assert "COHERENT_BEAMS_BOUNDED" in surface.reason_codes
    gamma = (surface.peak_irradiance_W_per_cm2 - a - b) / (2.0 * math.sqrt(a * b))
    spread = 2.0 * (2.0 * k2) / (opd * opd)
    chief = (1.0 + spread * spread) ** -0.25 * math.exp(-1.0 / (1.0 + spread * spread))
    assert chief * (1.0 - 1e-6) <= gamma <= chief * (1.0 + 1e-3)


def _two_lasers(group_b: str | None) -> ProjectSnapshot:
    """Two 1 mW 532 nm lasers whose chief rays meet at the centre of a window's front face, 3 deg apart; laser B in its own
    coherence group (the default: mutually incoherent sources) or in ``group_b``, A's group ``G``."""
    tilt = math.radians(3.0)
    a = B.laser("A", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                coherence_group="G")
    b = B.laser("B", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5,
                position=(-100.0 * math.cos(tilt), -100.0 * math.sin(tilt), 0.0), direction=(math.cos(tilt), math.sin(tilt), 0.0),
                coherence_group=group_b)
    window = B.window("W", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=(0.5, 0.0, 0.0))
    window = window.model_copy(update={"power_ratings": (_cw_rating(1e9, _beam_mm(0.5, 100.0, 532.0)),)})
    return ProjectSnapshot(components=(a, b, window))


def test_mutually_incoherent_sources_add_and_one_coherence_group_combines_their_fields():
    separate = next(s for s in run(_two_lasers(None)).exposures.surfaces if s.surface_id == "W/front")
    a, b = (h.peak_irradiance_W_per_cm2 for h in separate.hits)
    assert separate.peak_irradiance_W_per_cm2 == a + b and separate.incident_power_W == pytest.approx(2e-3, rel=1e-12)
    (comparison,) = separate.ratings
    # both peaks at the face centre: the sum is reached there, less what each laser's own reflections inside the window
    # and its own light left out as negligible (coherent with it alone) may cancel: between
    # sum_c (sqrt t_c - sum sqrt g_c - A_c)^2 and a + b
    ghosts = {source: sum(math.sqrt(h.peak_irradiance_W_per_cm2) for h in separate.untraced_hits if h.source_id == source)
              for source in ("A", "B")}
    result = run(_two_lasers(None))
    n = bk7_index(532.0)
    faint = {}
    returns = separate.faint_light_return_factor  # (its returns between the window's faces: window_returns)
    for line in result.exposures.faint_light:  # (each laser's left-out light at its diffraction limit, the face meets glass)
        w = line.wavelength_nm * 1e-6 / (math.pi * n)
        faint[line.source_id] = open_paths(1.0, 1.0) * math.sqrt(2.0 * line.power_W / (math.pi * w * w) * 100.0) * returns
    least = (math.sqrt(a) - ghosts["A"] - faint["A"]) ** 2 + (math.sqrt(b) - ghosts["B"] - faint["B"]) ** 2
    assert least * (1.0 - 1e-12) <= comparison.exposure_lower_bound <= a + b
    assert {g.coherence_group for g in separate.coherent_groups} == {"G", "source:B"}
    shared = next(s for s in run(_two_lasers("G")).exposures.surfaces if s.surface_id == "W/front")
    (traced,) = shared.coherent_groups  # both lasers and their reflections inside the window, from both sides
    assert traced.coherence_group == "G" and traced.members[:2] == ("traced:A#0:0", "traced:B#0:0")
    assert len(traced.members) == 2 + len(shared.untraced_hits) and traced.incidence == "BOTH_SIDES"
    assert shared.peak_irradiance_W_per_cm2 == pytest.approx((math.sqrt(a) + math.sqrt(b)) ** 2, rel=1e-12)
    (comparison,) = shared.ratings
    assert comparison.exposure_lower_bound is None  # equal amplitudes: (sqrt a - sqrt b)^2 = 0
    # the power through the face: cross term of beams at 0 and 3 deg up to 2 sqrt(P_a P_b) (cos 0 + cos 3) / (2 sqrt(cos 3))
    c = math.cos(math.radians(3.0))
    assert shared.incident_power_W == pytest.approx(2e-3 + 2e-3 * (1.0 + c) / (2.0 * math.sqrt(c)), rel=1e-9)


# --------------------------------------------------------------------------------------- the window's own etalon


def test_the_rear_face_of_a_window_is_bounded_by_its_own_etalon():
    """An uncoated window's rear face S2 sees the traced beam and the double reflection S2 -> S1 -> S2 (and its returns),
    both arriving from inside the glass: coherent with it, they bound the exposure by (sqrt I_t -+ sum sqrt I_g)^2 -- from
    below also by the window bound, the beam against a window over its area (``window_written_out``), which the light the
    run leaves out as negligible, of unknown place, cannot undo. The first ghost's power is the traced beam's at S2 times
    R^2 (Fresnel at normal incidence, written out here)."""
    run_ = run_optical_project(_document([_rating("CW", 1.0e9, "W/cm2")]))
    s2 = _surface(run_, "W/S2")
    (traced,) = s2.hits
    n = _n("N-BK7", 1064.0)
    reflectance = ((n - 1.0) / (n + 1.0)) ** 2
    first = min(s2.untraced_hits, key=lambda h: h.pass_index)
    assert first.incident_power_W == pytest.approx(traced.incident_power_W * reflectance ** 2, rel=1e-9)
    # (and the light the run left out as negligible, at its diffraction limit: ``faint_amplitude``)
    roots = sum(math.sqrt(h.peak_irradiance_W_per_cm2) for h in s2.untraced_hits) + faint_amplitude(run_, "W/cm2", surface=s2)
    (comparison,) = s2.ratings
    point = (math.sqrt(traced.peak_irradiance_W_per_cm2) - roots) ** 2
    q, n_beam = _independent_q()["W/S2"]
    window = window_written_out(traced.peak_irradiance_W_per_cm2, traced.beam_radius_x_mm, q, n_beam, _n("N-BK7", 1064.0),
                                [(h.peak_irradiance_W_per_cm2, h.beam_radius_x_mm) for h in s2.untraced_hits],
                                _unknown_root(run_, s2, "W/cm2"))
    assert window > point and comparison.exposure_lower_bound == pytest.approx(window, rel=1e-6)
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(traced.peak_irradiance_W_per_cm2) + roots) ** 2, rel=1e-9)
    (group,) = s2.coherent_groups
    assert group.members[0] == "traced:SRC#0:0" and group.incidence == "ALONG_NORMAL" and "COHERENT_BEAMS_BOUNDED" in s2.reason_codes
    # the front face: its ghosts arrive from inside the glass (the other side), counter-propagating to the transmitted beam
    # on one axis: a standing wave, bounded alike (verifier finding P2; before, they only added to the upper bound)
    s1 = _surface(run_, "W/S1")
    (incident,) = s1.hits
    (front,) = s1.ratings
    back = sum(math.sqrt(h.peak_irradiance_W_per_cm2) for h in s1.untraced_hits) + faint_amplitude(run_, "W/cm2", surface=s1)
    q, n_beam = _independent_q()["W/S1"]
    window = window_written_out(incident.peak_irradiance_W_per_cm2, incident.beam_radius_x_mm, q, n_beam, _n("N-BK7", 1064.0),
                                [(h.peak_irradiance_W_per_cm2, h.beam_radius_x_mm) for h in s1.untraced_hits],
                                _unknown_root(run_, s1, "W/cm2"))
    assert window > (math.sqrt(incident.peak_irradiance_W_per_cm2) - back) ** 2
    assert front.exposure_lower_bound == pytest.approx(window, rel=1e-6)
    assert front.exposure_upper_bound == pytest.approx((math.sqrt(incident.peak_irradiance_W_per_cm2) + back) ** 2, rel=1e-9)
    (group,) = s1.coherent_groups
    assert group.incidence == "BOTH_SIDES" and "COHERENT_BEAMS_BOUNDED" in s1.reason_codes
    # the incident power is a flux: the beams crossing S1 in opposite directions add their powers, without a cross term
    assert s1.incident_power_W == incident.incident_power_W


# ---------------------------------------------------------------------------------------- light of unknown fate


def test_unknown_fate_of_another_source_leaves_the_lower_bound_of_an_incoherent_beam():
    """Laser A meets a rated ideal mirror M (no ghosts); laser B, an independent source elsewhere, runs through three
    windows into the per-path interaction limit: part of B's light has an unknown fate and may reach M, so M has no upper
    bound; being incoherent with A it can only add, so A's peak stays a lower bound -- above the rating. The rating states
    a test beam half A's 1/e^2 diameter (a threshold at most the rated value, whatever light of unknown size adds), but
    light of unknown fate may reach M in a state the trace does not know: its polarization is not verifiable, so it is not
    verified and decides no exceedance."""
    w0, power = 0.5, 1e-3
    a = B.laser("A", wavelength_nm=532.0, power_W=power, waist_radius_mm=w0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    peak = 2.0 * power / (math.pi * (w0 * math.sqrt(1.0 + (100.0 / rayleigh_range(w0, 532.0)) ** 2)) ** 2) * 100.0
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(
        update={"power_ratings": (_cw_rating(0.5 * peak, 0.5 * _beam_mm(w0, 100.0, 532.0)),)})
    b = B.laser("B", wavelength_nm=532.0, power_W=power, waist_radius_mm=w0, position=(-100.0, 100.0, 0.0), direction=(1.0, 0.0, 0.0))
    windows = tuple(B.window(f"WB{k}", thickness_mm=2.0, clear_radius_mm=10.0, material="N-BK7", position=(-60.0 + 20.0 * k, 100.0, 0.0))
                    for k in range(3))
    project = ProjectSnapshot(components=(a, mirror, b, *windows),
                              targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),))
    cfg = NumericalConfig(max_interactions_per_path=3)
    result = validate(ValidationRequest(project=project, configuration=EngineConfiguration(numerical=cfg)))
    assert not result.exposures.trace_complete
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    (comparison,) = face.ratings
    assert comparison.exposure_upper_bound is None and comparison.exposure_lower_bound == pytest.approx(peak, rel=1e-9)
    beam = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert (beam.state, beam.effect) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED")
    assert _unverified(comparison) == {("polarization", "LIGHT_STATE_UNKNOWN")}
    assert comparison.status == "CONDITIONS_NOT_VERIFIED"
    (target,) = result.targets
    assert target.status.value == "UNKNOWN"


# ------------------------------------------------------------------------------------------- a standing wave


def test_a_retro_reflected_beam_and_its_source_form_a_standing_wave_on_the_coating():
    """Verifier finding P2: laser -> 45 deg NPBS (R 1 %) -> flat mirror 50 mm behind it at normal incidence -> back onto the
    coating from its other side. The coating sees the forward beam I_f and the returning beam I_r of one source and line,
    counter-propagating: their fields add (a standing wave), up to I_f + I_r + 2 gamma sqrt(I_f I_r). At the chief rays
    gamma = exp(-(OPD / L_c)^2) = 0.9999 for the round trip OPD = 2 x 50 mm and L_c = 10 m (written out here): at least
    0.999, the pair counts as fully coherent (an upper bound of the degree of coherence anywhere on the coating). The flux
    sum I_f + I_r was reported before and a rating of 3 I_f was WITHIN with the target PASS. (No light is left out here:
    the coating and the mirror split and reflect without untraced branches.)"""
    inv = 1.0 / math.sqrt(2.0)
    laser = B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0), coherence_length_mm=1e4)

    def project(rating: float | None) -> ProjectSnapshot:
        splitter = B.npbs("BS", reflectance=0.01, position=(0.0, 0.0, 0.0), normal=(-inv, inv, 0.0))
        if rating is not None:
            # (the test beam: between the forward beam 100 mm and the returning one 200 mm from the waist, within 2 % of both)
            test_mm = 0.5 * (_beam_mm(0.5, 100.0, 1064.0) + _beam_mm(0.5, 200.0, 1064.0))
            splitter = splitter.model_copy(update={"power_ratings": (_cw_rating(rating, test_mm, 1064.0),)})
        targets = (Target(target_id="u", device_id="BS", metric="power_rating_utilization", maximum=1.0),) if rating else ()
        mirror = B.mirror("R", position=(50.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
        return ProjectSnapshot(components=(laser, splitter, mirror), targets=targets)

    coating = next(s for s in run(project(1e9)).exposures.surfaces if s.surface_id.startswith("BS/") and len(s.hits) == 2)
    forward, back = coating.hits
    i_f, i_r = forward.peak_irradiance_W_per_cm2, back.peak_irradiance_W_per_cm2
    assert math.exp(-(100.0 / 1e4) ** 2) >= 0.999
    (group,) = coating.coherent_groups
    assert group.incidence == "BOTH_SIDES" and group.largest_coherence_factor == 1.0
    (comparison,) = coating.ratings
    assert coating.faint_light_peak_W_per_cm2 is None and "FAINT_LIGHT_BOUNDED" not in coating.reason_codes
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(i_f) + math.sqrt(i_r)) ** 2, rel=1e-12)
    assert comparison.exposure_lower_bound == pytest.approx((math.sqrt(i_f) - math.sqrt(i_r)) ** 2, rel=1e-9)
    assert coating.incident_power_W == forward.incident_power_W + back.incident_power_W  # a flux: no cross term
    result = run(project(3.0 * i_f))
    coating = next(s for s in result.exposures.surfaces if s.surface_id == coating.surface_id)
    (comparison,) = coating.ratings
    assert comparison.status == "UNKNOWN" and result.targets[0].status.value == "UNKNOWN"  # verifier: WITHIN, PASS


# ------------------------------------------------------------------------------------------- faint light left out


def _attenuated_window(pickoff_t: float | None, rating_value: float | None, *, stated: bool = True) -> ProjectSnapshot:
    """Verifier finding P1: a 1 W 1064 nm beam, optionally behind a pick-off transmitting ``pickoff_t``, meets an uncoated
    3 mm N-BK7 window at normal incidence; a CW rating (fixture) on its rear face only, with a utilization target.
    ``stated`` False: the rating states no test beam or polarization (a run's depth is then that of e344aab7)."""
    components = [B.laser("SRC", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                          direction=(1.0, 0.0, 0.0))]
    if pickoff_t is not None:
        inv = 1.0 / math.sqrt(2.0)
        components.append(B.npbs("PK", reflectance=1.0 - pickoff_t, position=(-50.0, 0.0, 0.0), normal=(-inv, inv, 0.0)))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(60.0, 0.0, 0.0))
    targets: tuple[Target, ...] = ()
    if rating_value is not None:
        # (the test beam: the beam on the rear face, 158.5 mm of air and 3 mm of N-BK7 from the waist, q' = n q + t)
        n = bk7_index(1064.0)
        q = n * complex(158.5, rayleigh_range(0.5, 1064.0)) + 3.0
        rear_mm = 2.0 * math.sqrt(1064.0e-6 / (math.pi * n) * abs(q) ** 2 / q.imag)
        rating = _cw_rating(rating_value, rear_mm, 1064.0).model_copy(update={"surface_ids": ("back",)})
        if not stated:
            rating = rating.model_copy(update={"beam_diameter_mm": None, "beam_diameter_definition": None, "polarization": None,
                                               "polarization_condition": None})
        window = window.model_copy(update={"power_ratings": (rating,)})
        targets = (Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),)
    return ProjectSnapshot(components=(*components, window), targets=targets)


@pytest.mark.parametrize(("pickoff_t", "decided"), [(1e-7, 200.0), (1e-5, 1.7)], ids=["T1e-7", "T1e-5"])
def test_faint_light_behind_a_pick_off_is_bounded_by_its_diffraction_limit_and_followed_deeper(pickoff_t: float, decided: float):
    """Behind the pick-off the window's own double reflection (amplitude R = ((n - 1) / (n + 1))^2 of the beam's, so the
    rear face's peak lies in [(1 - R)^2, (1 + R)^2] x I_t to first order, written out here) falls below the line's
    negligible level (1e-9 x 1 W): the verifier saw an exact I_t, WITHIN a rating of 1.04 I_t and the target PASS at
    T = 1e-7. Left out, that light is bounded by its diffraction limit (``faint_amplitude``): far above the faint beam
    itself. A rating of 1.04 I_t is never decided; for a rating above the etalon the run follows the light deeper (down to
    the power floor at most) until the probes' known spots replace the bound: then (sqrt I_t -+ (sum sqrt I_g + A))^2
    with what is still left out, and WITHIN / PASS at ``decided`` x I_t (at T = 1e-7 the pieces left at the power floor,
    mutually coherent, add their amplitudes over their open paths -- the pieces of each number of reflections too, from
    every sequence of the window's bare faces and the pick-off: 196 I_t at most, so 200 x). Light left out as negligible,
    of unknown size and state, still reaches the rear face beyond 1 % of its light: the rating's stated test beam and
    polarization cannot be verified, so neither rating is verified or decided, the exposure bounds as before."""
    n = bk7_index(1064.0)
    reflectance = ((n - 1.0) / (n + 1.0)) ** 2
    probe = run(_attenuated_window(pickoff_t, 1e30))
    back = next(s for s in probe.exposures.surfaces if s.surface_id == "W/back")
    (hit,) = back.hits
    i_t = hit.peak_irradiance_W_per_cm2
    assert hit.incident_power_W < 1e-4 and "FAINT_LIGHT_BOUNDED" in back.reason_codes
    (comparison,) = back.ratings
    roots = sum(math.sqrt(g.peak_irradiance_W_per_cm2) for g in back.untraced_hits) + faint_amplitude(probe, "W/cm2",
                                                                                                      paths=PICKOFF_PATHS)
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(i_t) + roots) ** 2, rel=1e-9)
    assert comparison.exposure_upper_bound >= (1.0 + reflectance) ** 2 * i_t
    result = run(_attenuated_window(pickoff_t, 1.04 * i_t))
    (comparison,) = next(s for s in result.exposures.surfaces if s.surface_id == "W/back").ratings
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and result.targets[0].status.value == "UNKNOWN"  # verifier: WITHIN, PASS
    above = run(_attenuated_window(pickoff_t, decided * i_t))
    back = next(s for s in above.exposures.surfaces if s.surface_id == "W/back")
    (comparison,) = back.ratings
    (line,) = above.exposures.faint_light
    assert line.followed_to_W < 1e-9 * 1.0 and back.untraced_hits  # (followed deeper: the double reflection is known)
    roots = sum(math.sqrt(g.peak_irradiance_W_per_cm2) for g in back.untraced_hits) + faint_amplitude(above, "W/cm2",
                                                                                                      paths=PICKOFF_PATHS)
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(i_t) + roots) ** 2, rel=1e-9)
    assert (1.0 + reflectance) ** 2 * i_t * (1.0 - 1e-6) <= comparison.exposure_upper_bound <= decided * i_t
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and above.targets[0].status.value == "UNKNOWN"


def test_light_below_the_power_floor_is_bounded_by_its_diffraction_limit():
    """Behind a pick-off of 1e-14 the beam on the window carries about 1e-14 W; its double reflection (R^2 of it) is below
    the power floor (power_epsilon_W = 1e-15 W), where no path is followed. Its power is still known: its diffraction
    limit bounds the rear face (FAINT_LIGHT_BOUNDED), never unboundedly -- far above the faint beam itself, so a rating of
    1.04 I_t is not decided, and only a rating above that bound would be within it; that light, of unknown size and
    state, keeps the rating's stated test beam and polarization from being verified, so neither is decided."""
    probe = run(_attenuated_window(1e-14, 1e30))
    back = next(s for s in probe.exposures.surfaces if s.surface_id == "W/back")
    (hit,) = back.hits
    assert 1e-15 < hit.incident_power_W < 1e-13
    assert "FAINT_LIGHT_BOUNDED" in back.reason_codes and not back.complete
    (comparison,) = back.ratings
    upper = (math.sqrt(hit.peak_irradiance_W_per_cm2) + faint_amplitude(probe, "W/cm2", paths=PICKOFF_PATHS)) ** 2
    assert comparison.exposure_upper_bound == pytest.approx(upper, rel=1e-9) and comparison.exposure_lower_bound is None
    result = run(_attenuated_window(1e-14, 1.04 * hit.peak_irradiance_W_per_cm2))
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and "FAINT_LIGHT_BOUNDED" in target.message
    above = run(_attenuated_window(1e-14, 1.01 * upper))
    (comparison,) = next(s for s in above.exposures.surfaces if s.surface_id == "W/back").ratings
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE and comparison.exposure_upper_bound <= 1.01 * upper
    assert above.targets[0].status.value == "UNKNOWN"


def test_at_full_power_the_etalon_and_the_light_left_out_bound_the_rear_face():
    """Without the pick-off the double reflections down to the negligible level are followed: the rear face is bounded by
    its etalon, which contains the first-order range [(1 - R)^2, (1 + R)^2] x I_t (R the Fresnel reflectance, written out),
    and by the light left out below that level (about 2e-10 W of the 1 W line), at its diffraction limit:
    (sqrt I_t -+ (sum sqrt I_g + A))^2. A rating of 1.04 I_t, inside the etalon, is never decided."""
    n = bk7_index(1064.0)
    reflectance = ((n - 1.0) / (n + 1.0)) ** 2
    probe = run(_attenuated_window(None, 1e30))
    back = next(s for s in probe.exposures.surfaces if s.surface_id == "W/back")
    (hit,) = back.hits
    i_t = hit.peak_irradiance_W_per_cm2
    (comparison,) = back.ratings
    assert "FAINT_LIGHT_BOUNDED" in back.reason_codes
    roots = sum(math.sqrt(g.peak_irradiance_W_per_cm2) for g in back.untraced_hits) + faint_amplitude(probe, "W/cm2")
    assert comparison.exposure_upper_bound == pytest.approx((math.sqrt(i_t) + roots) ** 2, rel=1e-9)
    # (from below: the window bound, the beam against a window over its area, ``window_written_out``)
    q_front = complex(160.0, rayleigh_range(0.5, 1.0640e3))
    window = window_written_out(i_t, hit.beam_radius_x_mm, n * q_front + 3.0, n, n,
                                [(g.peak_irradiance_W_per_cm2, g.beam_radius_x_mm) for g in back.untraced_hits],
                                _unknown_root(probe, back, "W/cm2"))
    assert window > (math.sqrt(i_t) - roots) ** 2
    assert comparison.exposure_lower_bound == pytest.approx(window, rel=1e-6)
    assert comparison.exposure_upper_bound >= (1.0 + reflectance) ** 2 * i_t * (1.0 - 1e-6)
    assert comparison.exposure_lower_bound <= (1.0 - reflectance) ** 2 * i_t * (1.0 + 1e-6)
    result = run(_attenuated_window(None, 1.04 * i_t))
    assert next(s for s in result.exposures.surfaces if s.surface_id == "W/back").ratings[0].status == "UNKNOWN"
    assert result.targets[0].status.value == "UNKNOWN"


def test_a_deeper_pass_that_runs_out_of_its_budget_leaves_the_first_pass_bound():
    """With a trace budget of 5 paths the deeper pass behind the 1e-7 pick-off cannot finish: the first pass stands, the
    rear face bounded by the diffraction limit of the light left out there (finite: FAINT_LIGHT_BOUNDED) and the target
    UNKNOWN at 1.04 I_t, never PASS (the rating not verified: that light, of unknown size and state, keeps its stated test
    beam and polarization from being checked)."""
    cfg = NumericalConfig(max_paths=5)
    probe = run(_attenuated_window(1e-7, 1e30), numerical=cfg)
    (hit,) = next(s for s in probe.exposures.surfaces if s.surface_id == "W/back").hits
    result = run(_attenuated_window(1e-7, 1.04 * hit.peak_irradiance_W_per_cm2), numerical=cfg)
    back = next(s for s in result.exposures.surfaces if s.surface_id == "W/back")
    (comparison,) = back.ratings
    (line,) = result.exposures.faint_light
    assert result.outcome.execution.outcome.value == "COMPLETED"  # (the probes' budget is an auxiliary analysis's)
    assert result.outcome.execution.budgets.exhausted_by_phase == {"untraced_light": ("max_paths",)}
    assert result.outcome.execution.budgets.untraced_power_W is None
    assert line.followed_to_W == 1e-9  # (the first pass's level: the deeper one ran out of its budget)
    upper = (math.sqrt(hit.peak_irradiance_W_per_cm2) + faint_amplitude(result, "W/cm2", paths=PICKOFF_PATHS)) ** 2
    assert "FAINT_LIGHT_BOUNDED" in back.reason_codes and comparison.exposure_upper_bound == pytest.approx(upper, rel=1e-9)
    assert _unverified(comparison) == UNKNOWN_SIZE_AND_STATE
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and result.targets[0].status.value == "UNKNOWN"


def test_deeper_exposure_probe_interaction_limit_is_recorded_per_phase():
    """(The rating states no test beam or polarization: a stated one is checked against the light left out as negligible,
    and the run then follows that light deeper even under a rating of 1e30, which is not what this test is about.)"""
    cfg = NumericalConfig(max_interactions_per_path=5)
    initial = run(_attenuated_window(1e-7, 1e30, stated=False), numerical=cfg)
    assert initial.outcome.execution.outcome.value == "COMPLETED"
    (hit,) = next(s for s in initial.exposures.surfaces if s.surface_id == "W/back").hits
    # This tighter rating triggers a deeper pass; the initial pass itself did not exhaust anything.
    refined = run(_attenuated_window(1e-7, 1.04 * hit.peak_irradiance_W_per_cm2, stated=False), numerical=cfg)
    execution = refined.outcome.execution
    assert execution.outcome.value == "COMPLETED" and execution.budgets.exhausted == ("max_interactions_per_path",)
    assert execution.budgets.exhausted_by_phase == {"untraced_light": ("max_interactions_per_path",)}
    assert execution.budgets.affected_source_ids == ("SRC",)
    assert execution.budgets.affected_path_ids
    assert execution.budgets.untraced_power_W is None
    assert refined.targets[0].assessment.value == "INDETERMINATE"
