"""Group delay in the degree of coherence, left-out light that returns between surfaces facing each other, the power of
a plane, probe aberration and rating text.

* A source of declared coherence length ``L_c`` has a finite bandwidth: two of its fields are coherent according to the
  delay of their envelopes -- the difference of their group optical paths ``sum n_g L``, not of their phase paths ``sum n
  L`` -- and their group-delay dispersion (``optics/interference/coherence.py``). Light that crossed dispersive glass is
  taken as fully coherent with its class on a surface (its envelope delay across the beam is not followed).
* Light the trace leaves out as negligible may return to a surface between two surfaces facing each other (an etalon, a
  cavity): its bound counts every return (``validation/resonance.py``; the pairs and the model's scope:
  ``test_return_pairs.py``), and light a probe follows into a cavity meets the interaction limit: its fate is then not
  known.
* The power of a plane (``power_W``) includes the left-out light's power; probe arrivals are checked for aberrations with
  pupil rays as traced beams are; a rating's polarization text that names another condition than its field is refused,
  and text that says more than the field is a condition the comparison cannot check.

Every power rating is fixture data declared in the test project (its value stated as a peak irradiance), never product
data; the hand values are written out (Sellmeier N-BK7 by finite differences, Fresnel reflectance, Gaussian q, the
diffraction limit).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from os_optics import builders as B
from os_optics.contracts import JonesVector, PowerRating, ProjectSnapshot, PulseTrain, Target
from os_optics.contracts.materials import ConstantCoating
from os_optics.contracts.optics import InterfacePhases

from helpers import run
from test_power_exposure_ratings import open_paths, open_power

FIX = "verifier fixture rating, not product data"
INV = 1.0 / math.sqrt(2.0)
# N-BK7 (Schott catalogue Sellmeier coefficients, lambda in um)
BK7_B = (1.03961212, 0.231792344, 1.01046945)
BK7_C = (0.00600069867, 0.0200179144, 103.560653)


def _bk7(lam_um: float) -> float:
    return math.sqrt(1.0 + sum(b * lam_um * lam_um / (lam_um * lam_um - c) for b, c in zip(BK7_B, BK7_C)))


def _bk7_group(wavelength_nm: float) -> tuple[float, float, float]:
    """(n, n_g = n - lambda dn/dlambda, k2 = lambda^3 n'' / (2 pi) in mm) of N-BK7 by central differences."""
    lam = wavelength_nm * 1e-3
    n0 = _bk7(lam)
    fine, coarse = 1e-5, 1e-3  # (um: the first derivative's truncation and the second's rounding both far below 1e-9)
    slope = (_bk7(lam + fine) - _bk7(lam - fine)) / (2.0 * fine)
    curvature = (_bk7(lam + coarse) - 2.0 * n0 + _bk7(lam - coarse)) / (coarse * coarse)  # 1/um^2
    return n0, n0 - lam * slope, lam ** 3 * curvature * 1e-3 / (2.0 * math.pi)


def _rating(value: float, wavelength_nm: float, **kw) -> PowerRating:
    kw.setdefault("value_definition", "PEAK")  # (a peak irradiance: the threshold the decisions here are read against)
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength_nm, status="DECLARED", basis=FIX, **kw)


def _rated(component, rating: PowerRating):
    return component.model_copy(update={"power_ratings": (rating,)})


def _coat(component, r: float, both: bool = False):
    s = component.surfaces[0]
    c = ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=1.0 - r, transmittance_p=1.0 - r)
    update = {"coating": c, **({"reflective_side": "both"} if both else {})}
    return component.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update=update)}),)})


def _utilization(device: str) -> tuple[Target, ...]:
    return (Target(target_id="u", device_id=device, metric="power_rating_utilization", maximum=1.0),)


def _surface(result, surface_id: str):
    return next(s for s in result.exposures.surfaces if s.surface_id == surface_id)


# --------------------------------------------------------------------------------------- group delay


def _stated(component):
    """A splitter whose phases are stated constant over wavelength (no group delay): those of the default lossless
    splitter, r_s = i sqrt(R), r_p = -i sqrt(R), t = sqrt(T)."""
    s = component.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2.0,), reflection_p_rad=(3.0 * math.pi / 2.0,),
                             transmission_s_rad=(0.0,), transmission_p_rad=(0.0,))
    return component.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})}),)})


def _folding(component):
    """A fold mirror of 99 % reflectance (fixture coating) that states the default reflection phases of a mirror (pi for s,
    0 for p), constant over wavelength: its group delay is known (none)."""
    coated = _coat(component, 0.99)
    s = coated.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi,), reflection_p_rad=(0.0,))
    return coated.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})}),)})


def _balanced_interferometer(rating: float | None, *, length: float = 0.1, stated: bool = True):
    """A Mach-Zehnder interferometer at 532 nm (coherence length 0.1 mm) with 20 mm of N-BK7 in
    arm B and, in arm A, the extra air path that equals the window's group-delay excess (n_g - 1) 20 mm: the arms' group
    delays agree (the white-light fringe) while their phase paths differ by (n - n_g) 20 mm = -5.9 L_c. Its splitters state
    their phases (``stated``): their group delay is then known (none), else not. The fold mirrors reflect 99 % and state a
    mirror's default phases (``_folding``): MA is turned off the square, so the two face each other along the diagonal
    through the window, and two mirrors that reflect everything there would bound the left-out light nowhere."""
    n, ng, _ = _bk7_group(532.0)
    g = (ng - 1.0) * 20.0
    delta = (2.0 * 100.0 * g + g * g) / (2.0 * (100.0 + g))
    norm = math.hypot(delta, 100.0)
    u = (-delta / norm, 100.0 / norm, 0.0)
    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.8, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(90.0), coherence_length_mm=length)
    bs1 = B.npbs("BS1", position=(0.0, 0.0, 0.0), normal=(-INV, INV, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    mb = _folding(B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7))
    ma = _folding(B.mirror_for_reflection("MA", position=(100.0 + delta, 0.0, 0.0), d_in=(1, 0, 0), d_out=u, clear_radius_mm=12.7))
    win = B.window("WB", thickness_mm=20.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 100.0, 0.0), axis=(1.0, 0.0, 0.0))
    bisector = (1.0 - u[0], -u[1], 0.0)
    length_b = math.sqrt(sum(x * x for x in bisector))
    bs2 = B.npbs("BS2", position=(100.0, 100.0, 0.0), normal=tuple(x / length_b for x in bisector), reflectance=0.5, clear_radius_mm=12.7)
    if stated:
        bs1, bs2 = _stated(bs1), _stated(bs2)
    dump = B.beam_dump("D", position=(100.0 + 40.0 * u[0], 100.0 + 40.0 * u[1], 0.0), axis=u, radius_mm=12.7)
    targets: tuple[Target, ...] = ()
    if rating is not None:
        # (the test beam: either arm's 1/e^2 diameter at the dump, 50 + 100 + 100 + 40 mm from the 0.8 mm waist; s-polarized
        # linear light, as the source's through the 45 deg splitters and folds)
        beam_mm = 2.0 * 0.8 * math.sqrt(1.0 + (290.0 / (math.pi * 0.64 / (532.0 * 1e-6))) ** 2)
        dump = _rated(dump, _rating(rating, 532.0, beam_diameter_mm=beam_mm, beam_diameter_definition="1/E2", polarization="LINEAR",
                                    polarization_condition="linear"))
        targets = _utilization("D")
    plane = B.observation_plane("pre", position=(100.0 + 39.0 * u[0], 100.0 + 39.0 * u[1], 0.0), normal=u, radius_mm=12.7)
    return ProjectSnapshot(components=(src, bs1, mb, ma, win, bs2, dump), observations=(plane,), targets=targets), (n, ng)


def test_a_group_delay_balanced_interferometer_is_coherent_at_its_white_light_fringe():
    """The plane analysis and the exposure took the phase path difference: degree of coherence 9e-16, the dump bounded
    at [27.3, 69.0] W/cm2 -- a rating of 71.05 WITHIN / PASS, one of 10 EXCEEDS / FAIL -- while at the white-light fringe the
    arms are coherent (0.9998, the window's group-delay dispersion) and the dump reaches 94.7 W/cm2 or nearly 0. With its
    splitters' phases stated the plane reports the group path difference (about 0) and the dispersed Gaussian spectrum's
    coherence written out here; the dump's arms, one through glass, are fully coherent in its bounds: neither rating is
    decided. (Light left out as negligible, of unknown size and state, may also reach the dump with most of its peak
    irradiance: the ratings' stated test beam and polarization cannot be verified there, so they are not verified.)"""
    project, (n, ng) = _balanced_interferometer(None)
    result = run(project)
    (pair,) = [r for r in result.interference if r.observation_id == "pre"]
    assert pair.optical_path_difference_mm == pytest.approx(-(n - ng) * 20.0, rel=1e-6)  # (the phase paths: 5.9 L_c apart)
    assert abs(pair.group_path_difference_mm) < 1e-8
    k2 = _bk7_group(532.0)[2]
    b = 2.0 * (20.0 * k2) / (0.1 * 0.1)
    hand = (1.0 + b * b) ** -0.25 * math.exp(-((pair.group_path_difference_mm / 0.1) ** 2) / (1.0 + b * b))
    assert pair.coherence_factor == pytest.approx(hand, rel=1e-7) and 0.9997 < pair.coherence_factor < 0.9999
    assert pair.coherence_margin == pytest.approx(1.0 - abs(pair.group_path_difference_mm) / 0.1, rel=1e-12)
    a, c = (arr for arr in result.observations[0].arrivals)
    assert a.group_path_length_mm is not None and c.group_path_length_mm is not None
    dump = _surface(run(_balanced_interferometer(1e9)[0]), "D/dump")
    i_a, i_b = sorted((h.peak_irradiance_W_per_cm2 for h in dump.hits), reverse=True)
    (group,) = dump.coherent_groups
    assert group.largest_coherence_factor == 1.0
    (comparison,) = dump.ratings
    peak = (math.sqrt(i_a) + math.sqrt(i_b)) ** 2
    assert comparison.exposure_upper_bound >= peak
    assert (comparison.exposure_lower_bound or 0.0) <= (math.sqrt(i_a) - math.sqrt(i_b)) ** 2
    for rating in (71.05, 10.0):  # (the verifier's false WITHIN / PASS and false EXCEEDS / FAIL)
        verdict = run(_balanced_interferometer(rating)[0])
        (rated,) = _surface(verdict, "D/dump").ratings
        assert {(c.condition, c.reason) for c in rated.conditions if c.state == "NOT_VERIFIABLE"} == {
            ("beam_diameter", "BEAM_SIZE_UNKNOWN"), ("polarization", "LIGHT_STATE_UNKNOWN")}
        assert rated.status == "CONDITIONS_NOT_VERIFIED" and verdict.targets[0].status.value == "UNKNOWN"


def test_splitters_whose_phases_are_not_stated_leave_the_interferometer_s_coherence_unknown():
    """The same interferometer with plain (zero-thickness) splitters: a real splitter delays its reflected and transmitted
    light differently (its coating, its substrate), and the arms take them the other way round -- their relative envelope
    delay, and their degree of coherence, are not known: the group path difference and the coherence metrics are stated
    at no added delay at those passes (an ESTIMATE entry naming them), the coherent sum on the plane likewise with its
    range over every delay and phase, the dump's bounds at full coherence."""
    project, _ = _balanced_interferometer(None, stated=False)
    result = run(project)
    (pair,) = [r for r in result.interference if r.observation_id == "pre"]
    passes = {"BS1/coating:split_reflect:+", "BS1/coating:split_transmit:+", "BS2/coating:split_reflect:-", "BS2/coating:split_transmit:+"}
    assert pair.coherence_length_mm == 0.1 and "COHERENCE_UNDEFINED" not in pair.issue_codes
    # (the balanced arms: no traced group path difference; the degree of coherence on the convention near 1, lowered only by
    # the arms' group-delay dispersion)
    assert abs(pair.group_path_difference_mm) < 1e-6 and 0.999 < pair.coherence_factor <= 1.0
    (assumed,) = pair.value_basis["coherence_factor"].assumed
    assert assumed.input.value == "UNSTATED_GROUP_DELAY_ZERO" and set(assumed.at) == passes
    a, c = result.observations[0].arrivals
    assert {x.split("@")[0] for x in a.group_delay_unknown_at} ^ {x.split("@")[0] for x in c.group_delay_unknown_at} == passes
    (total,) = result.observations[0].superpositions
    assert total.total_power_W is not None and total.value_basis["total_power_W"].kind.value == "ESTIMATE"
    assert total.total_power_min_W <= total.total_power_W <= total.total_power_max_W
    dump = _surface(run(_balanced_interferometer(1e9, stated=False)[0]), "D/dump")
    i_a, i_b = sorted((h.peak_irradiance_W_per_cm2 for h in dump.hits), reverse=True)
    (comparison,) = dump.ratings
    assert comparison.exposure_upper_bound >= (math.sqrt(i_a) + math.sqrt(i_b)) ** 2


# ------------------------------------------------------------------------------- returns between surfaces


def window_returns(n: float, wavelength_nm: float, waist_mm: float) -> float:
    """The factor of an uncoated window's faces (``faint_light_return_factor``), written out: light left out inside the
    window meets each face from inside again and again -- each round trip keeps at most R^2 of its power, R the face's
    reflectance from inside at the largest angle the light can have there: the source's beam reaches the window within 8
    far-field divergences, theta = 8 lambda / (pi w0), of its normal, and the glass keeps its momentum along the faces,
    sin t = sin(theta) / n (Fresnel s from the glass at t, the larger) -- so all its arrivals there bring at most
    1 / (1 - R) times one arrival's field; what each return sends out of the window is at most that return's field, so
    the factor holds on either side of every surface: 1 / (1 - R)."""
    theta = 8.0 * wavelength_nm * 1e-6 / (math.pi * waist_mm)
    sin_t = math.sin(theta) / n
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    rs = (n * cos_t - math.cos(theta)) / (n * cos_t + math.cos(theta))
    return 1.0 / (1.0 - rs * rs)


def test_a_window_s_two_faces_return_left_out_light_and_its_bound_counts_every_return():
    """The light left out inside an uncoated window bounces between its faces: its bound counts every return (the factor
    each face reports, ``window_returns``; ``faint_light_return_factor``, ``FAINT_LIGHT_RECIRCULATES``)."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = _rated(B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0)),
                    _rating(1e9, 1064.0))
    result = run(ProjectSnapshot(components=(src, window), targets=_utilization("W")))
    for face in ("W/front", "W/back"):
        surface = _surface(result, face)
        assert surface.faint_light_return_factor == pytest.approx(window_returns(_bk7(1.064), 1064.0, 0.5), rel=1e-12)
        assert "FAINT_LIGHT_RECIRCULATES" in surface.reason_codes


def _cavity(rating: float, coupler: float = 1e-5):
    """M1 leaks 5e-10 of a 1 W, 1064 nm beam (left out: below 1e-9 W); a thin lens mode-matches the
    leak into a hemispherical cavity (flat C1, concave C2 of R 10.1 mm, 10 mm apart; both mirrors transmit ``coupler``),
    which stores eta P / T on resonance -- 9.3 W/cm2 on C1 from 5e-10 W."""
    lam = 1064e-6
    w_mode = math.sqrt(lam / math.pi * math.sqrt(10.0 * (10.1 - 10.0)))
    w_lens = 0.5 * math.sqrt(1.0 + (150.0 / (math.pi * 0.25 / lam)) ** 2)
    f = w_mode * math.pi * w_lens / lam
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    m1 = _coat(B.mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7), 1.0 - 5e-10)
    dump = B.beam_dump("D", position=(100.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    lens = B.thin_lens("L", focal_length_mm=f, clear_radius_mm=12.7, position=(150.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    c1 = _rated(_coat(B.mirror("C1", position=(150.0 + f, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=5.0), 1.0 - coupler, both=True),
                _rating(rating, 1064.0))
    c2 = _rated(_coat(B.spherical_mirror("C2", position=(160.0 + f, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=10.1,
                                         clear_radius_mm=3.0), 1.0 - coupler, both=True), _rating(rating, 1064.0))
    return ProjectSnapshot(components=(src, m1, dump, lens, c1, c2),
                           targets=(Target(target_id="c1", device_id="C1", metric="power_rating_utilization", maximum=1.0),
                                    Target(target_id="c2", device_id="C2", metric="power_rating_utilization", maximum=1.0)))


@pytest.mark.parametrize("rating", [2.22, 0.555, 0.0555, 2.22e5])
def test_a_cavity_only_left_out_light_enters_is_never_within_a_rating(rating: float):
    """The cavity's mirrors were bounded by the left-out leak's diffraction limit ([0, 1.11] W/cm2) as if it crossed them
    once: a rating of 2.22 W/cm2 PASSed, and with ratings inside that bound the deeper pass left out the 5e-15 W entering
    the cavity again (0.0555 W/cm2 PASSed, 168 x below the stored light). Now the two mirrors facing each other return
    the left-out light with a round-trip gain of (1 - 1e-5)^2: its bound is 1e5 times one arrival's field, and light a
    deeper pass follows into the cavity meets the interaction limit -- no rating there is decided below the stored peak."""
    result = run(_cavity(rating))
    for face in ("C1/face", "C2/face"):
        surface = _surface(result, face)
        assert "FAINT_LIGHT_RECIRCULATES" in surface.reason_codes
        (comparison,) = surface.ratings
        assert comparison.status != "WITHIN_RATING"
    assert all(t.status.value != "PASS" for t in result.targets)


def test_two_fold_mirrors_facing_each_other_across_an_interferometer_return_no_light():
    """The fold mirrors of a Mach-Zehnder interferometer face each other along its diagonal, but no light of it travels
    that way (every plane of the scene turns its axis directions into axis directions): coated mirrors that transmit 1 %
    there are not taken as a cavity."""
    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.5, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    bs1 = B.npbs("BS1", position=(0.0, 0.0, 0.0), normal=(-INV, INV, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    ma = _rated(_coat(B.mirror_for_reflection("MA", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7), 0.99),
                _rating(1e9, 532.0))
    mb = _coat(B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7), 0.99)
    bs2 = B.npbs("BS2", position=(100.0, 100.0, 0.0), normal=(-INV, INV, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    d1 = B.beam_dump("D1", position=(200.0, 100.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7)
    d2 = B.beam_dump("D2", position=(100.0, 200.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    result = run(ProjectSnapshot(components=(src, bs1, ma, mb, bs2, d1, d2), targets=_utilization("MA")))
    surface = _surface(result, "MA/face")
    assert surface.faint_light_return_factor is None and "FAINT_LIGHT_RECIRCULATES" not in surface.reason_codes


# ------------------------------------------------------------------------------- the power of a plane


@pytest.mark.parametrize("absorbing", [False, True])
def test_the_power_of_a_dark_plane_includes_the_left_out_light(absorbing: bool):
    """A plane no traced or followed light reaches was a known 0 W for power_W -- a maximum of
    1e-14 W PASSed -- while the light left out as negligible (1.8e-10 W) may reach it. Its power now lies between 0 and that
    light's power on an absorbing plane (each watt absorbed once), on a transparent one as often as its open paths cross
    it -- once per straight leg, three between two reflections (the window's two faces may turn it), before and after
    each reflection at the window's two faces: nine times -- times the square of the window's factor (its returns may cross
    the plane again)."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    win = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 0.0, 0.0))
    dump = _rated(B.beam_dump("D0", position=(150.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7), _rating(1e9, 1064.0))
    plane = B.observation_plane("DARK", position=(50.0, 80.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=12.7)
    if absorbing:
        plane = plane.model_copy(update={"absorbing": True})
    targets = (Target(target_id="p", observation_id="DARK", metric="power_W", maximum=1e-14),
               Target(target_id="q", observation_id="DARK", metric="power_W", maximum=1.0))
    result = run(ProjectSnapshot(components=(src, win, dump), observations=(plane,), targets=targets))
    (line,) = result.exposures.faint_light
    small, large = result.targets
    factor = _surface(result, "W/front").faint_light_return_factor
    assert factor == pytest.approx(window_returns(_bk7(1.064), 1064.0, 0.5), rel=1e-12)
    crossings = open_power(1.0, 1.0, plane=True)
    assert crossings == 9.0
    most = (1.0 if absorbing else crossings * factor * factor) * line.power_W
    if absorbing:
        # (each watt absorbed once: 1.8e-10 W, within the 1 W line's negligible level -- the stated truncation of the 0 W,
        # decision B4; the entry holds the bound. The maximum of 1e-14 W lies inside [0, bound], so the truncation does not
        # decide it: INDETERMINATE with the value 0.0 (W28); the maximum of 1 W above the bound is met)
        assert most <= 1e-9
        assert small.status.value == "UNKNOWN" and small.assessment.value == "INDETERMINATE"
        assert small.value == 0.0 and "FAINT_LIGHT_BOUNDED" in small.issue_codes
        assert f"between 0 W and {most:.6g} W" in small.message
        (omitted,) = small.value_basis["value"].omitted
        assert omitted.part.value == "BELOW_NEGLIGIBLE_LEVEL" and omitted.upper == pytest.approx(most, rel=1e-12)
    else:
        # (crossed nine times and returned by the window: above the negligible level, so the target is not decided)
        assert most > 1e-9
        assert small.status.value == "UNKNOWN" and small.value is None and "FAINT_LIGHT_BOUNDED" in small.issue_codes
        assert f"{most:.4g} W" in small.message
    assert large.status.value == "PASS"


# --------------------------------------------------------------------------------- probe aberration


def _leaky_lens_scene(leak: float, rating: float):
    """A 45 deg mirror whose coating leaks ``leak`` of a 1 W, 1064 nm beam (w0 1.9 mm);
    the leak crosses an AR-coated plano-convex N-BK7 lens flat side first (R2 -10 mm: strong spherical aberration) onto a
    rated dump 221 mm from the source (true on-axis peak 3.4e5 W/cm2 per W of leak)."""
    ar = ConstantCoating(reflectance_s=0.0, reflectance_p=0.0, transmittance_s=1.0, transmittance_p=1.0)
    lens = B.singlet_lens("L", r1_mm=math.inf, r2_mm=-10.0, thickness_mm=4.0, clear_radius_mm=9.0, material="N-BK7", position=(200.0, 0.0, 0.0))
    lens = lens.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": ar})})
                                                     for s in lens.surfaces)})
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=1.9, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    m = _coat(B.mirror_for_reflection("M", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7), 1.0 - leak)
    d2 = B.beam_dump("D2", position=(100.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    dump = _rated(B.beam_dump("D", position=(221.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7), _rating(rating, 1064.0))
    return ProjectSnapshot(components=(src, m, d2, lens, dump), targets=_utilization("D"))


@pytest.mark.parametrize(("leak", "rating"), [(1e-2, 376.7), (5e-10, 3e-5)])
def test_a_probe_through_an_aberrating_lens_is_not_a_paraxial_spot(leak: float, rating: float):
    """The probes ran without pupil rays: the leak's arrival counted as its paraxial Gaussian spot (125.6 W/cm2 for 1e-2 W,
    6.3e-6 W/cm2 for 5e-10 W after a deeper pass) and ratings below the true aberrated peak PASSed. The probes now carry the
    pupil rays of the beam they leave, and their arrival is checked like a traced one: aberration dominated, bounded by the
    diffraction limit of its power, never a known spot."""
    result = run(_leaky_lens_scene(leak, rating))
    dump = _surface(result, "D/dump")
    (spot,) = [h for h in dump.untraced_hits if h.incident_power_W == pytest.approx(leak, rel=1e-6)]
    assert spot.status.value == "UNKNOWN" and spot.reason == "GAUSSIAN_ABERRATION_DOMINATED" and spot.peak_irradiance_W_per_cm2 is None
    assert dump.ratings[0].status != "WITHIN_RATING" and result.targets[0].status.value != "PASS"


# ------------------------------------------------------------------------------------ rating text


@pytest.mark.parametrize(("field", "text"), [("S", "p"), ("S", "TM"), ("S", "p-polarized light"), ("S", "P-pol."),
                                             ("P", "TE (s-polarized)"), ("LINEAR", "circular (RHC)"), ("CIRCULAR", "linear 45 deg")])
def test_a_rating_text_naming_another_polarization_than_its_field_is_refused(field: str, text: str):
    """A field with a text the mapping did not read was compared on the field alone."""
    with pytest.raises(ValidationError, match="disagrees with polarization_condition"):
        _rating(1e9, 1064.0, polarization=field, polarization_condition=text)


def test_a_rating_text_that_says_more_than_its_field_is_a_condition_not_verified():
    """``LINEAR`` with ``linear, vertical (E || z)`` on a horizontally polarized beam: conditions_match was true, WITHIN,
    PASS. The text states an orientation the comparison does not check: not verified, never matched."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(0.0))
    m = _rated(B.mirror_for_reflection("M", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7),
               _rating(1e9, 1064.0, polarization="LINEAR", polarization_condition="linear, vertical (E || z)"))
    dump = B.beam_dump("D", position=(100.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    result = run(ProjectSnapshot(components=(src, m, dump), targets=_utilization("M")))
    (comparison,) = _surface(result, "M/face").ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.conditions_match is False
    assert any("linear, vertical" in text for text in comparison.unverified_conditions)
    assert result.targets[0].status.value != "PASS"


# ------------------------------------------------------------------------------ the diffraction limit's spectrum


def test_the_diffraction_limit_of_a_short_pulse_counts_its_spectrum():
    """A 10 fs pulse at 800 nm has a transform-limited spectrum whose mean (lambda0 / lambda)^2 is 1 + ln2 lambda^2 / (2 pi^2
    c^2 tau^2) = 1.0025: the diffraction limit of its left-out light is that much larger (on a mirror only that light
    reaches: A^2 = c^2 f^2 x 2 pi P / lambda^2 x 1.0025, f the factor of the window's returns, c = 12 sqrt(2) the most
    its open paths' crossings bring: the window's two bare faces and the uncoated mirror may each reflect it once, and
    between two reflections it may meet the plane mirror once per straight leg, three of them -- the window's faces may
    turn it)."""
    pulse = PulseTrain(repetition_rate_hz=1e6, pulse_duration_s=1e-14)
    src = B.laser("S", wavelength_nm=800.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    src = src.model_copy(update={"source": src.source.model_copy(update={"pulse": pulse})})
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    mirror = _rated(B.mirror("M", position=(0.0, 200.0, 0.0), normal=(0.0, -1.0, 0.0)), _rating(1e12, 800.0))
    result = run(ProjectSnapshot(components=(src, window, mirror), targets=_utilization("M")))
    (line,) = result.exposures.faint_light
    lam_mm, tau_mm = 800e-6, 2.99792458e11 * 1e-14
    spectral = 1.0 + math.log(2.0) * lam_mm * lam_mm / (2.0 * math.pi * math.pi * tau_mm * tau_mm)
    assert spectral == pytest.approx(1.0025, abs=1e-4)
    limit = 2.0 * math.pi * line.power_W / (800e-7) ** 2 * spectral  # W/cm2, lambda in cm
    surface = _surface(result, "M/face")
    factor = surface.faint_light_return_factor
    assert factor is not None and factor == _surface(result, "W/front").faint_light_return_factor
    paths = open_paths(1.0, 1.0, per_leg=(1.0,), plane=True)
    # (the window's faces once and the mirror once per leg, three legs: the field's sequences of distinct surfaces)
    assert paths == pytest.approx(3.0 * math.sqrt(2.0) * (1.0 + math.sqrt(5.0) + math.sqrt(14.0) + math.sqrt(18.0)), rel=1e-12)
    assert surface.faint_light_peak_W_per_cm2 == pytest.approx(paths * paths * factor * factor * limit, rel=1e-9)
