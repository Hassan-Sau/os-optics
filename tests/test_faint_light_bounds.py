"""Light left out as negligible, the degree of coherence across a surface and rating polarization text
(``validation/exposures.py``).

The trace and the probes leave out branches of light at or below ``negligible_power_fraction`` x the line input, and paths
stop at the power floor. Such light is faint in power, but focused it can reach an irradiance far above its share of the
power: a 1e-9 W branch focused to a 1.4 um spot gives 0.03 W/cm2 next to a 6 mm beam's 1.8 W/cm2, and its field adds
(coherent, one source line) up to (sqrt 1.77 + sqrt 0.029)^2 = 2.25 W/cm2. The exposure bounds that light by its
diffraction limit -- light of power P at vacuum wavelength lambda in a medium of index n reaches at most 2 pi n^2 P /
lambda^2 when it crosses a plane from one side, the peak of a Gaussian beam of waist lambda / (pi n) -- times the most
its open paths' crossings bring (at least 2: all of it once from either side; more where it may meet the surface after
reflections, or once per straight leg on a plane surface), and follows it deeper where that alone keeps a comparison
from being decided. Every scene below is a review scene; the hand values are written out here (Gaussian q
propagation, 2 P / (pi w^2)). Every power rating is fixture data declared in the test project, never product data; a
rating's value is stated as a peak (``value_definition`` PEAK), and the pulsed one's count as two pulses on each tested
site at one level (S-on-1) against a single shot. Where a decision is asserted the rating states the main beam's own
1/e^2 diameter at the dump as its test beam and linear polarization (``STATED``): the focused leak is far smaller than the
test beam, a threshold at least the rated value, so the bound alone keeps a rating from being within.
"""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from os_optics import builders as B, validate_json
from os_optics.contracts import JonesVector, PowerRating, ProjectSnapshot, Target
from os_optics.contracts.components import RATING_POLARIZATION_TEXT, PulseTrain, rating_polarization
from os_optics.contracts.materials import ConstantCoating
from os_optics.contracts.request import EngineConfiguration, ValidationRequest
from os_optics.validation.pipeline import run_physics
from os_optics.validation.untraced import FAINT_ROOT

from helpers import rayleigh_range, run
from test_power_exposure_ratings import faint_amplitude, open_paths, open_power

FIX = "verifier fixture rating, not product data"
LAM, P0, W0 = 532.0, 1.0, 6.0
INV = 1.0 / math.sqrt(2.0)
#: The main beam's 1/e^2 diameter at the dump of ``_focus``, 1000 mm from its waist (Gaussian propagation), and the test
#: conditions a rating there states: that test beam and linear polarization (the source's).
MAIN_BEAM_MM = 2.0 * W0 * math.sqrt(1.0 + (1000.0 / rayleigh_range(W0, LAM)) ** 2)
STATED = dict(beam_diameter_mm=MAIN_BEAM_MM, beam_diameter_definition="1/E2", polarization="LINEAR", polarization_condition="linear")


def _leaky(mirror, t: float):
    s = mirror.surfaces[0]
    coat = ConstantCoating(reflectance_s=1.0 - t, reflectance_p=1.0 - t, transmittance_s=t, transmittance_p=t)
    return mirror.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": coat})}),)})


def _focus(t_leak: float, ratings, *, main_to_dump: bool = True, f_mm: float = 50.0, hwp: bool = False, pulse=None,
           splitter: float | None = None) -> ProjectSnapshot:
    """A review scene: a 1 W, 532 nm beam of waist 6 mm folded by M1, M2, M3 onto a beam dump D
    1000 mm from the source along its path (100 + 300 + 300 + 300 mm; or sent away from it); M1's coating leaks
    ``t_leak``, which a thin lens of focal length
    ``f_mm`` focuses onto D. ``hwp``: a half-wave plate at 45 deg in the leak turns it to p; ``splitter``: an NPBS of
    that reflectance in the leak."""
    src = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=W0, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(90.0))
    if pulse is not None:
        src = src.model_copy(update={"source": src.source.model_copy(update={"pulse": pulse})})
    m1 = _leaky(B.mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=50.0), t_leak)
    m2 = B.mirror_for_reflection("M2", position=(100.0, 300.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=50.0)
    m3 = B.mirror_for_reflection("M3", position=(400.0, 300.0, 0.0), d_in=(1, 0, 0), d_out=(0, -1, 0) if main_to_dump else (0, 1, 0),
                                 clear_radius_mm=50.0)
    lens = B.thin_lens("L", focal_length_mm=f_mm, clear_radius_mm=50.0, position=(400.0 - f_mm, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    dump = B.beam_dump("D", position=(400.0, 0.0, 0.0), axis=(INV, -INV, 0.0), radius_mm=50.0).model_copy(
        update={"power_ratings": tuple(ratings)})
    parts = [src, m1, m2, m3, lens, dump]
    if hwp:
        parts.append(B.half_waveplate("HWP", position=(200.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), fast_axis_angle_deg=45.0,
                                      design_wavelength_nm=LAM, clear_radius_mm=50.0))
    if splitter is not None:
        parts.append(B.npbs("BS", position=(200.0, 0.0, 0.0), normal=(-INV, -INV, 0.0), reflectance=splitter, clear_radius_mm=50.0))
    return ProjectSnapshot(components=tuple(parts), targets=(Target(target_id="u", device_id="D", metric="power_rating_utilization", maximum=1.0),))


def _cw(value: float, **kw) -> PowerRating:
    kw.setdefault("value_definition", "PEAK")
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=LAM, status="DECLARED", basis=FIX, **kw)


def _radius(q: complex) -> float:
    """The 1/e^2 radius (mm) of a Gaussian beam of complex parameter ``q`` (mm) at 532 nm in air."""
    return math.sqrt(-LAM * 1e-6 / (math.pi * (1.0 / q).imag))


def _hand(t_leak: float, f_mm: float = 50.0) -> tuple[float, float]:
    """(main beam's peak at the dump, the focused leak's peak there), from the Gaussian q of the source waist: the main beam
    (1 - T of the source: M1 transmits the leak) after 1000 mm of free space (flat mirrors), the leak after 400 - f mm, the
    thin lens (1/q' = 1/q - 1/f) and f mm."""
    q0 = complex(0.0, rayleigh_range(W0, LAM))
    w_main = _radius(q0 + 1000.0)
    q_lens = q0 + (400.0 - f_mm)
    w_spot = _radius(1.0 / (1.0 / q_lens - 1.0 / f_mm) + f_mm)
    return 2.0 * P0 * (1.0 - t_leak) / (math.pi * w_main ** 2) * 100.0, 2.0 * (P0 * t_leak) / (math.pi * w_spot ** 2) * 100.0


def _dump(result):
    return next(s for s in result.exposures.surfaces if s.surface_id == "D/dump")


# ------------------------------------------------------------------------------------- a faint branch focused


@pytest.mark.parametrize("t_leak", [5e-9, 9e-10], ids=["followed", "left-out"])
def test_a_faint_branch_focused_onto_a_broad_beam_is_never_within_a_rating_below_their_coherent_peak(t_leak: float):
    """At T = 9e-10 the leak is left out (below 1e-9 W); the dump was an exact 1.768 W/cm2,
    WITHIN a rating of 1.1 x that, target PASS -- while the leak's field adds up to (sqrt I_m + sqrt I_f)^2 = 1.27 x. Now
    the left-out leak's diffraction limit keeps the rating undecided, the run follows it deeper (its focused spot becomes a
    known contribution) and the comparison stays UNKNOWN, its upper bound at least the coherent peak."""
    i_m, i_f = _hand(t_leak)
    coherent = (math.sqrt(i_m) + math.sqrt(i_f)) ** 2
    result = run(_focus(t_leak, [_cw(1.1 * i_m, **STATED)]))
    dump = _dump(result)
    (traced,) = dump.hits
    assert traced.peak_irradiance_W_per_cm2 == pytest.approx(i_m, rel=1e-9)
    (spot,) = [h for h in dump.untraced_hits if h.incident_power_W == pytest.approx(t_leak, rel=1e-9)]
    assert spot.peak_irradiance_W_per_cm2 == pytest.approx(i_f, rel=1e-9)
    (comparison,) = dump.ratings
    assert comparison.status == "UNKNOWN" and comparison.exposure_upper_bound >= coherent * (1.0 - 1e-4)
    assert result.targets[0].status.value == "UNKNOWN"  # verifier at 9e-10: WITHIN_RATING, PASS
    assert result.exposures.faint_light == () or all(line.followed_to_W < 9e-10 for line in result.exposures.faint_light)


def test_a_rated_surface_only_faint_light_reaches_is_never_a_known_zero():
    """The main beam sent away, only the left-out leak (9e-10 W) reaches the dump,
    focused to 0.029 W/cm2. The dump had no exposure record at all (a known zero) and a rating of half the spot's peak
    PASSed. Now it is bounded: first by the left-out light's diffraction limit, then -- followed deeper -- by the spot
    itself, never 0, and that rating is not decided. (The rating states no test beam or polarization: not verified. Stated,
    they could not be checked against light the trace leaves out either, of unknown size and state, and the run would not
    follow it to the spot this test is about.)"""
    _, i_f = _hand(9e-10)
    result = run(_focus(9e-10, [_cw(0.5 * i_f)], main_to_dump=False))
    dump = _dump(result)
    assert dump.hits == ()
    (spot,) = dump.untraced_hits
    assert spot.peak_irradiance_W_per_cm2 == pytest.approx(i_f, rel=1e-9)
    (comparison,) = dump.ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.exposure_upper_bound >= i_f * (1.0 - 1e-4)
    (target,) = result.targets
    assert target.status.value == "UNKNOWN"  # verifier: PASS, value 0


def test_a_rated_surface_no_light_follows_to_is_bounded_by_the_left_out_light():
    """A rated mirror beside a beam that crosses an uncoated window: no traced or followed light reaches the mirror, but
    the window's ghosts below the negligible level are left out and may go anywhere. The mirror is listed with no hits and
    its comparison lies between 0 and the left-out light's bound A^2 (written out: the total left-out power P focused to
    w = lambda / pi in air, 2P / (pi w^2), its amplitude times the most its open paths' crossings bring -- the window's two
    faces and the mirror may each reflect it once, and between two reflections it may meet the plane mirror once per
    straight leg, three of them (the window's two faces may turn it) -- and times the factor of the window's returns,
    which may reach the mirror too): never a known zero. A rating above that bound would be within it, but the rating's
    test beam and polarization cannot be checked against light of unknown size and state: not verified, not decided."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    mirror = B.mirror("M", position=(0.0, 200.0, 0.0), normal=(0.0, -1.0, 0.0)).model_copy(update={"power_ratings": (
        PowerRating(regime="CW", value=1e6, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                    basis=FIX, beam_diameter_mm=2.0 * 0.5 * math.sqrt(1.0 + (100.0 / rayleigh_range(0.5, 1064.0)) ** 2),
                    beam_diameter_definition="1/E2", polarization="LINEAR", polarization_condition="linear"),)})
    result = run(ProjectSnapshot(components=(src, window, mirror),
                                 targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    assert face.hits == () and face.untraced_hits == () and "FAINT_LIGHT_BOUNDED" in face.reason_codes and not face.complete
    # (its open paths: the window's two bare faces and the uncoated mirror, each 1; the mirror met once per leg)
    bound = faint_amplitude(result, "W/cm2", material=None, paths=open_paths(1.0, 1.0, per_leg=(1.0,), plane=True)) ** 2
    assert face.faint_light_peak_W_per_cm2 == pytest.approx(bound, rel=1e-12)
    (comparison,) = face.ratings
    assert comparison.exposure_lower_bound is None and comparison.exposure_upper_bound == pytest.approx(bound, rel=1e-12)
    assert {(c.condition, c.reason) for c in comparison.conditions if c.state == "NOT_VERIFIABLE"} == {
        ("beam_diameter", "BEAM_SIZE_UNKNOWN"), ("polarization", "LIGHT_STATE_UNKNOWN")}
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.exposure_upper_bound < 1e6  # (within the value)
    element = next(e for e in result.exposures.elements if e.element_id == "M")
    assert (element.max_utilization, element.utilization_upper_bound) == (None, None)
    assert result.targets[0].status.value == "UNKNOWN"


def test_many_faint_reflections_add_their_amplitudes():
    """25 thin windows whose faces reflect 9.9e-10 each (left out: below 1e-9 W) send 50 faint
    reflections back onto a rated plate in front of them. The largest single branch was used (visibility 6e-5) and the
    plate was exact, WITHIN a rating between its traced value and the 50 reflections' standing-wave maximum. Now their
    total power (written out: R (1 - R)^(2k) and R (1 - R)^(2k+1) over the faces) bounds the plate as one field."""
    reflectance = 9.9e-10
    coat = ConstantCoating(reflectance_s=reflectance, reflectance_p=reflectance, transmittance_s=1.0 - reflectance,
                           transmittance_p=1.0 - reflectance)

    def project(rating: float) -> ProjectSnapshot:
        src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
        plate = B.thin_lens("PL", focal_length_mm=1e12, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0)).model_copy(update={
            "power_ratings": (PowerRating(regime="CW", value=rating, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK",
                                          status="DECLARED", basis=FIX),)})
        windows = []
        for k in range(25):
            w = B.window(f"W{k}", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=(20.0 + 5.0 * k, 0.0, 0.0))
            windows.append(w.model_copy(update={"surfaces": tuple(
                s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": coat})}) for s in w.surfaces)}))
        return ProjectSnapshot(components=(src, plate, *windows),
                               targets=(Target(target_id="u", device_id="PL", metric="power_rating_utilization", maximum=1.0),))

    probe = run(project(1e9))
    (line,) = probe.exposures.faint_light
    reflections = sum(reflectance * (1.0 - reflectance) ** k for k in range(50))
    assert line.power_W >= reflections * (1.0 - 1e-6)
    plate = next(s for s in probe.exposures.surfaces if s.surface_id == "PL/lens")
    (hit,) = plate.hits
    (comparison,) = plate.ratings
    # (its open paths: the 50 coated faces, each reflecting 9.9e-10; the plate is a thin lens, which reflects nothing and
    # the light meets at most once between two reflections: the least count, 2)
    paths = open_paths(*[reflectance] * 50, turning=51)
    assert paths == 2.0
    upper = (math.sqrt(hit.peak_irradiance_W_per_cm2) + faint_amplitude(probe, "W/cm2", material=None, paths=paths)) ** 2
    assert comparison.exposure_upper_bound == pytest.approx(upper, rel=1e-9)
    # the hand standing-wave maximum of the 50 reflections (each one's Gaussian at the plate) lies inside the bounds
    zr = rayleigh_range(0.5, 1064.0)
    amplitudes = 0.0
    for k in range(25):
        for face, extra in ((20.0 + 5.0 * k - 0.5, 0), (20.0 + 5.0 * k + 0.5, 1)):
            w = 0.5 * math.sqrt(1.0 + ((100.0 + 2.0 * face) / zr) ** 2)
            amplitudes += math.sqrt(2.0 * reflectance * (1.0 - reflectance) ** (2 * k + extra) / (math.pi * w * w) * 100.0)
    true_max = (math.sqrt(hit.peak_irradiance_W_per_cm2) + amplitudes) ** 2
    assert comparison.exposure_upper_bound >= true_max
    result = run(project(0.5 * (hit.peak_irradiance_W_per_cm2 + true_max)))
    assert next(s for s in result.exposures.surfaces if s.surface_id == "PL/lens").ratings[0].status != "WITHIN_RATING"
    assert result.targets[0].status.value != "PASS"  # verifier: WITHIN_RATING, PASS


@pytest.mark.parametrize(("f_mm", "t_leak"), [(50.0, 9e-10), (25.0, 9e-10), (50.0, 5e-9)])
def test_a_faint_branch_of_another_polarization_does_not_hold_an_s_condition(f_mm: float, t_leak: float):
    """The leak turned to p by a half-wave plate and focused onto the dump where the main beam is
    s: at the spot 1.6 % (f 50 mm) or 6.1 % (f 25 mm) of the light is p at T = 9e-10. Left out, it was not seen and the S
    condition matched; now its share of the light (bounded by its diffraction limit, then followed) is not within 1 %:
    never matched."""
    i_m, i_f = _hand(t_leak, f_mm)
    assert i_f / (i_m + i_f) > 0.01
    result = run(_focus(t_leak, [_cw(1e3, polarization="S")], f_mm=f_mm, hwp=True))
    (comparison,) = _dump(result).ratings
    assert comparison.conditions_match is False and comparison.status in ("CONDITIONS_NOT_VERIFIED", "CONDITIONS_MISMATCH")
    assert result.targets[0].status.value != "PASS"


def test_a_probe_arrival_below_the_negligible_level_with_a_known_spot_is_kept():
    """A 1.5e-9 W leak (followed: above 1e-9 W) halved by a 50 % splitter before it
    focuses onto the dump arrives with 7.5e-10 W -- below the level -- and was dropped although its spot is known. It is
    kept as a known contribution; a rating between the main beam and their coherent peak is not decided."""
    i_m, i_f_full = _hand(1.5e-9)
    i_f = 0.5 * i_f_full
    coherent = (math.sqrt(i_m) + math.sqrt(i_f)) ** 2
    result = run(_focus(1.5e-9, [_cw(0.5 * (i_m + coherent), **STATED)], splitter=0.5))
    dump = _dump(result)
    spots = [h for h in dump.untraced_hits if h.incident_power_W == pytest.approx(7.5e-10, rel=1e-9)]
    assert len(spots) == 1 and spots[0].peak_irradiance_W_per_cm2 == pytest.approx(i_f, rel=1e-9)
    (comparison,) = dump.ratings
    assert comparison.status == "UNKNOWN" and comparison.exposure_upper_bound >= coherent * (1.0 - 1e-4)
    assert result.targets[0].status.value == "UNKNOWN"  # verifier: WITHIN_RATING, PASS


def test_a_faint_branch_of_a_pulsed_source_bounds_the_fluence_too():
    """The left-out leak at T = 0.9e-9 with 10 ns pulses at 1 kHz; a J/cm2 rating of 1.1 x the
    main beam's fluence (below their coherent peak, 1.27 x) was WITHIN / PASS. Here a single shot against a 2-on-1
    rating: a site receives the main beam's pulse and the leak's (one or two pulses, no more than the test), a threshold
    at least the rated value -- WITHIN would stand but for the bound."""
    pulse = PulseTrain(repetition_rate_hz=1e3, pulse_duration_s=1e-8, total_pulses=1)
    i_m, i_f = _hand(0.9e-9)
    f_m = i_m / 1e3
    rating = PowerRating(regime="PULSED", value=1.1 * f_m, unit="J/cm2", wavelength_nm=LAM, pulse_duration_s=1e-8,
                         repetition_rate_hz=1e3, pulse_count=2, test_protocol="S_ON_1", value_definition="PEAK", status="DECLARED",
                         basis=FIX, surface_kind="OTHER", **STATED)  # (a dielectric surface: a pulsed rating's beam-size direction)
    result = run(_focus(0.9e-9, [rating], pulse=pulse))
    (comparison,) = _dump(result).ratings
    assert comparison.applicability == "THRESHOLD_AT_LEAST_RATED"
    assert comparison.status == "UNKNOWN" and comparison.exposure_upper_bound >= (math.sqrt(f_m) + math.sqrt(i_f / 1e3)) ** 2 * (1 - 1e-4)
    assert result.targets[0].status.value == "UNKNOWN"


# ------------------------------------------------------------------------------- the deeper pass decides


def test_a_rating_just_above_a_window_s_etalon_is_decided_by_following_its_ghosts_deeper():
    """A 1 W window's rear face (its etalon range about [0.917, 1.087] x I_t): the left-out light's diffraction limit
    first widens the bounds to about [0.80, 1.22] x I_t; for a rating of 1.1 x I_t the run follows the line's light deeper
    until the left-out light is small enough, and the comparison is WITHIN, the target PASS -- as without it. (The rating
    states the beam's 1/e^2 diameter 100 mm from its waist as its test beam, and linear polarization.)"""
    beam_mm = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / rayleigh_range(0.5, 1064.0)) ** 2)
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

    def project(rating: float) -> ProjectSnapshot:
        window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0)).model_copy(update={
            "power_ratings": (PowerRating(regime="CW", value=rating, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK",
                                          status="DECLARED", basis=FIX, surface_ids=("back",), beam_diameter_mm=beam_mm,
                                          beam_diameter_definition="1/E2", polarization="LINEAR", polarization_condition="linear"),)})
        return ProjectSnapshot(components=(src, window), targets=(Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),))

    probe = run(project(1e9))
    back = next(s for s in probe.exposures.surfaces if s.surface_id == "W/back")
    i_t = back.hits[0].peak_irradiance_W_per_cm2
    assert back.ratings[0].exposure_upper_bound > 1.1 * i_t  # (the first pass alone would leave 1.1 x I_t open)
    result = run(project(1.1 * i_t))
    back = next(s for s in result.exposures.surfaces if s.surface_id == "W/back")
    (line,) = result.exposures.faint_light
    assert line.followed_to_W < 1e-9 and back.ratings[0].exposure_upper_bound <= 1.1 * i_t
    assert back.ratings[0].status == "WITHIN_RATING" and result.targets[0].status.value == "PASS"


# ------------------------------------------------------------------- the degree of coherence across a surface


def test_beams_crossing_at_an_angle_are_fully_coherent_somewhere_in_their_overlap():
    """Two arms of one source (coherence length 0.01 mm) cross on a dump at about
    10 deg with a chief-ray optical path difference of 2 L_c. The exposure took gamma = e^-4 everywhere (upper bound
    2 I (1 + e^-4)), but the arms' relative delay changes by 0.18 mm per mm across the overlap: about 0.1 mm from the
    centre it vanishes and the fringes have full contrast (hand maximum 3.88 I). The degree of coherence is now the
    largest the overlap allows: 1 here, the upper bound 4 I."""
    lc = 0.01
    dump_at = (600.0, 600.0, 0.0)

    def unit(v):
        n = math.sqrt(sum(x * x for x in v))
        return tuple(x / n for x in v)

    def geometry(delta: float):
        ma, mb = (100.0, 0.0, 0.0), (0.0, 100.0 + delta, 0.0)
        da = unit(tuple(d - m for d, m in zip(dump_at, ma)))
        db = unit(tuple(d - m for d, m in zip(dump_at, mb)))
        opl_a = 300.0 + math.dist(dump_at, ma)
        opl_b = 300.0 + delta + math.dist(dump_at, mb)
        return ma, mb, da, db, opl_b - opl_a

    lo, hi = 0.0, 1.0
    for _ in range(200):  # the offset of MB that makes the chief rays' path difference 2 L_c
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if geometry(mid)[4] < 2.0 * lc else (lo, mid)
    ma, mb, da, db, _ = geometry(0.5 * (lo + hi))

    def project(rating: float) -> ProjectSnapshot:
        src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-200.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                      polarization=JonesVector.linear(90.0), coherence_length_mm=lc)
        bs = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=(-INV, INV, 0.0), reflectance=0.5, clear_radius_mm=12.7)
        m_a = B.mirror_for_reflection("MA", position=ma, d_in=(1, 0, 0), d_out=da, clear_radius_mm=12.7)
        m_b = B.mirror_for_reflection("MB", position=mb, d_in=(0, 1, 0), d_out=db, clear_radius_mm=12.7)
        dump = B.beam_dump("DUMP", position=dump_at, axis=(INV, INV, 0.0), radius_mm=25.0).model_copy(update={"power_ratings": (
            PowerRating(regime="CW", value=rating, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                        basis=FIX),)})
        return ProjectSnapshot(components=(src, bs, m_a, m_b, dump),
                               targets=(Target(target_id="u", device_id="DUMP", metric="power_rating_utilization", maximum=1.0),))

    probe = run(project(1e12))
    dump = next(s for s in probe.exposures.surfaces if s.surface_id == "DUMP/dump")
    a, b = (h.peak_irradiance_W_per_cm2 for h in dump.hits)
    (group,) = dump.coherent_groups
    assert group.largest_coherence_factor == 1.0
    assert dump.ratings[0].exposure_upper_bound == pytest.approx((math.sqrt(a) + math.sqrt(b)) ** 2, rel=1e-12)
    result = run(project(3.88 * 0.5 * (a + b)))  # (the verifier's local-delay maximum: 3.88 I)
    assert result.targets[0].status.value != "PASS"  # verifier: WITHIN_RATING, PASS


# --------------------------------------------------------------------------------- rating polarization text


def test_a_rating_whose_polarization_text_states_another_condition_is_refused():
    """A rating stating 's-polarized' with the machine-readable field LINEAR was compared on
    the field alone (a p beam matched). Text that states a condition (``RATING_POLARIZATION_TEXT``) must agree with the
    field: such a rating is refused as input; text the mapping does not read may refine the field."""
    with pytest.raises(ValidationError, match="disagrees with polarization_condition"):
        _cw(1e9, polarization_condition="s-polarized", polarization="LINEAR")
    with pytest.raises(ValidationError):
        _cw(1e9, polarization_condition="P-Pol", polarization="S")
    assert _cw(1e9, polarization_condition="s-polarized", polarization="S").polarization == "S"
    assert _cw(1e9, polarization_condition="linear, vertical", polarization="LINEAR").polarization == "LINEAR"
    assert _cw(1e9, polarization_condition="s-polarized").polarization is None  # (the text alone: not verified, as before)
    # through the JSON entry point: a structured refusal
    project = _focus(1e-3, [])
    raw = json.loads(ValidationRequest(project=project, configuration=EngineConfiguration()).model_dump_json())
    rated = next(c for c in raw["project"]["components"] if c["instance_id"] == "D")
    rated["power_ratings"] = [{"regime": "CW", "value": 1e9, "unit": "W/cm2", "wavelength_nm": LAM, "status": "DECLARED",
                               "polarization_condition": "s-polarized", "polarization": "LINEAR"}]
    reply = json.loads(validate_json(json.dumps(raw)))
    assert reply["status"] == "FAIL" and reply["issues"][0]["code"] == "INVALID_INPUT"
    assert any("disagrees with polarization_condition" in e for e in reply["issues"][0]["numerical_context"]["errors"])
    # every plain form the catalog's ratings are lowered with is consistent with the field it maps to
    for text, condition in RATING_POLARIZATION_TEXT.items():
        assert rating_polarization(text.upper()) == condition
        assert _cw(1e9, polarization_condition=text, polarization=condition).polarization == condition


# --------------------------------------------------------------------------- observation planes: coherent power


def test_a_coherent_power_on_a_plane_includes_the_left_out_light():
    """A plane beside a beam that crosses an uncoated window, with a rating on the window (so that the run follows every
    untraced branch and reports the left-out light): no traced or followed light reaches the plane, so both powers are the
    known 0 W the targets are graded on, and the light left out as negligible is their stated truncation. The entries hold
    the most it may add: to the coherent power (c f sqrt P)^2, P the light left out as negligible (its field: c the most
    its open paths' crossings bring -- once per straight leg, three between two reflections (the window's two faces may
    turn it), before and after each reflection at the window's two faces, from either side --, times the factor f of the
    window's returns, which may cross the plane too); to its power (power_W) 9 f^2 P (the plane is transparent: crossed in
    each of the three legs before and after each of the two faces' reflections, and again by the returns)."""
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0)).model_copy(update={
        "power_ratings": (PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=1064.0, status="DECLARED", basis=FIX),)})
    plane = B.observation_plane("side", position=(0.0, 100.0, 0.0), normal=(0.0, -1.0, 0.0), radius_mm=10.0)
    targets = (Target(target_id="c", metric="coherent_power_W", observation_id="side", maximum=1.0),
               Target(target_id="p", metric="power_W", observation_id="side", maximum=1.0))
    project = ProjectSnapshot(components=(src, window), observations=(plane,), targets=targets)
    result = run(project)
    (line,) = result.exposures.faint_light
    factor = next(s for s in result.exposures.surfaces if s.surface_id == "W/front").faint_light_return_factor
    assert factor is not None and factor > 1.0  # (the window's returns may cross the plane too)
    outcome = run_physics(project, EngineConfiguration().numerical)
    # (its open paths: the window's two bare faces, each 1; the plane crossed once per leg)
    field = open_paths(1.0, 1.0, plane=True) * math.sqrt(line.power_W) * factor
    assert open_power(1.0, 1.0, plane=True) == 9.0
    assert outcome.reach["side"][("S", 0)][FAINT_ROOT] == pytest.approx(field, rel=1e-12)
    coherent, power = result.targets
    # (9 f^2 P is above the 1 W line's negligible level here: the light left out may add more than that level, so the
    # targets are judged on the interval from 0 -- no value -- and the plane is BOUNDED, not a known zero)
    most = open_power(1.0, 1.0, plane=True) * line.power_W * factor * factor
    assert most > 1e-9
    assert coherent.status.value == "PASS" and coherent.value is None and "FAINT_LIGHT_BOUNDED" in coherent.issue_codes
    assert f"sqrt({field * field:.4g} W)" in coherent.message
    assert power.status.value == "PASS" and power.value is None and "FAINT_LIGHT_BOUNDED" in power.issue_codes
    assert f"{most:.4g} W" in power.message
    (plane_result,) = result.observations
    assert plane_result.quantity_state.value == "BOUNDED" and plane_result.total_power_W == 0.0 and not plane_result.power_complete
    from os_optics.results import basis_of

    assert basis_of(plane_result, "total_power_W").omitted[0].part.value == "POSSIBLE_RETURN"
    assert plane_result.power_upper_bound_W == pytest.approx(open_power(1.0, 1.0, plane=True) * line.power_W * factor * factor, rel=1e-12)
