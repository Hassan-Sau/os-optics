"""A power rating's test polarization is checked against the beams before a match is claimed (review, finding
P1-b; ``validation/exposures.py``, ``local/lowering.py``).

A rating states its test polarization as text (``polarization_condition``; the catalog publishes its suppliers' S-Pol /
P-Pol as ``s-polarized`` / ``p-polarized``). A plain statement (linear, circular, s, p, unpolarized) becomes the Engine
rating's ``polarization`` and is compared with the light on the surface as a whole, from every beam's Jones state there:
it holds when every beam has at most 1 % of its power outside one common state (s / p in its plane of incidence, one
linear axis for ``LINEAR``, one circular state for ``CIRCULAR``), so that any incoherent mixture of them is within it
too (verifier finding: two mutually incoherent beams of crossed linear states were each linear, yet their mixture is
partly unpolarized); mutually coherent beams of different states hold it only as a sum, for every relative phase
(``test_coherent_polarization.py``). It does not hold (``CONDITIONS_MISMATCH``) only where the light at a traced beam's centre violates it for
certain, whatever the phases and the light of unknown size there; anything else -- other text, s / p at normal
incidence, beams of different states whose mixture is not decided -- is a condition the comparison cannot check:
``CONDITIONS_NOT_VERIFIED``, never WITHIN or EXCEEDS, and a utilization target is not PASS. The expected shares are
written out here from the source's Jones vector and the Fresnel coefficients. Every power rating here is fixture data
declared in the test project (its value stated as a peak irradiance), never product data.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics._vendor.shared_contracts.design_catalog import PowerRatingV1
from os_optics.contracts import JonesVector, PowerRating, ProjectSnapshot, Target
from os_optics.local import lower_project, run_optical_project
from os_optics.local.lowering import engine_power_ratings, rating_polarization

from helpers import run
from test_power_exposure_ratings import _document, _element, _n, _rating, _surface, faint_amplitude

UTIL = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]


def _beam_mm(wavelength_nm: float, path_mm: float = 100.0, waist_mm: float = 0.5) -> float:
    """A fixture laser's 1/e^2 diameter ``path_mm`` from its waist: 2 w0 sqrt(1 + (z / z_R)^2), z_R = pi w0^2 / lambda."""
    return 2.0 * waist_mm * math.sqrt(1.0 + (path_mm / (math.pi * waist_mm ** 2 / (wavelength_nm * 1e-6))) ** 2)


def _with_source_polarization(document: dict, polarization: dict) -> dict:
    document["components"][0]["source_model"]["polarization"] = polarization
    return document


def _least_share(run_, own: float) -> float:
    """The least share of the light at the centre of the beam on the window's front face S1 that is outside a rating's
    state, written out: the beam carries ``own`` of its power outside it; the window's own reflections (one state with the
    beam, arriving on its axis) leave at least e = sqrt(I_t) - sum sqrt(I_g) of its field there; the light the run leaves
    out as negligible, of unknown state, adds at most A (``faint_amplitude``) in any state:
    ((sqrt(own) e - A) / (e + A))^2."""
    s1 = _surface(run_, "W/S1")
    (hit,) = s1.hits
    e = math.sqrt(hit.peak_irradiance_W_per_cm2) - sum(math.sqrt(g.peak_irradiance_W_per_cm2) for g in s1.untraced_hits)
    a = faint_amplitude(run_, "W/cm2", surface=s1)
    gap = math.sqrt(own) * e - a
    return (gap / (e + a)) ** 2 if gap > 0.0 else 0.0


def _tilted(document: dict, degrees: float = 30.0) -> dict:
    """The window turned by ``degrees`` about the design z axis (as ``test_a_tilted_surface_spreads_the_footprint...``): the
    plane of incidence is the design x-y plane, s along design z."""
    (window,) = [c for c in document["components"] if c["instance_id"] == "W"]
    c, s = math.cos(math.radians(degrees) / 2.0), math.sin(math.radians(degrees) / 2.0)
    window["nominal_pose"]["quaternion_wxyz"] = [0.5 * (c - s), 0.5 * (c - s), 0.5 * (c + s), 0.5 * (c + s)]
    return document


# ---------------------------------------------------------------------------------------------------- the review


def test_the_review_case_a_linear_rating_does_not_match_a_circular_source():
    """review360_repro.py rating_polarization: rating 'linear', source ELLIPTICAL orientation 0 ellipticity 45 (circular).
    The review saw conditions_match true, WITHIN_RATING and the target PASS beside a note 'not compared'."""
    document = _document([_rating("CW", 1e9, "W/cm2", polarization_condition="linear")], targets=UTIL)
    _with_source_polarization(document, {"kind": "ELLIPTICAL", "orientation_deg": 0, "ellipticity_deg": 45})
    run_ = run_optical_project(document)
    (comparison,) = _surface(run_, "W/S1").ratings
    assert comparison.conditions_match is False and comparison.status == "CONDITIONS_MISMATCH"
    # circular light: half its power is on the minor axis of any linear state, sin^2(45 deg) = 0.5 -- the window's own
    # reflections at normal incidence keep that share; the light left out as negligible, of unknown state, may lower it at
    # most to ``_least_share`` (about 29 %), still far above the tolerance
    (mismatch,) = comparison.mismatches
    least = _least_share(run_, 0.5)
    assert 0.25 < least < 0.5
    assert mismatch == (f"polarization: at the centre of the beam of SRC (traced SRC#0, pass 0) at least {100.0 * least:.3g} % of "
                        "the light's power is in the state orthogonal to the rating's linear polarization (tolerance 1 %)")
    assert not any("not compared" in note for note in comparison.notes)
    (target,) = run_.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("POWER_RATING_CONDITIONS_MISMATCH",))


def test_a_linear_rating_matches_a_linear_source_and_says_what_it_checked():
    document = _document([_rating("CW", 1e9, "W/cm2", polarization_condition="Linearly polarized")], targets=UTIL)
    run_ = run_optical_project(document)
    for surface_id in ("W/S1", "W/S2"):
        (comparison,) = _surface(run_, surface_id).ratings
        assert comparison.conditions_match and comparison.status == "WITHIN_RATING" and comparison.unverified_conditions == ()
        assert comparison.notes == ("polarization: linear along one common axis at every beam here (largest share outside it "
                                    "0 %)",)
    (target,) = run_.result.targets
    assert target.status.value == "PASS"


@pytest.mark.parametrize(("chi_deg", "match"), [(5.0, True), (6.5, False)])
def test_an_elliptical_beam_matches_a_linear_rating_within_one_percent_on_its_minor_axis(chi_deg: float, match: bool):
    """Jones (cos chi, i sin chi): sin^2 chi of the power on the minor axis -- 0.76 % at 5 deg, 1.28 % at 6.5 deg (the
    window's reflections at normal incidence keep the state). At 6.5 deg the beam is outside the tolerance, but the light
    left out as negligible, of unknown state, may bring the share at the beam's centre down to ``_least_share`` -- below
    1 %: the condition is not verified (not matched, not decided either)."""
    document = _document([_rating("CW", 1e9, "W/cm2", polarization_condition="linear")], targets=UTIL)
    _with_source_polarization(document, {"kind": "ELLIPTICAL", "orientation_deg": 0, "ellipticity_deg": chi_deg})
    run_ = run_optical_project(document)
    (comparison,) = _surface(run_, "W/S1").ratings
    share = math.sin(math.radians(chi_deg)) ** 2
    assert comparison.conditions_match is match
    if match:
        assert f"largest share outside it {100.0 * share:.3g} %" in comparison.notes[0]
    else:
        least = _least_share(run_, share)
        assert least < 0.01 and comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.mismatches == ()
        assert comparison.unverified_conditions == (
            f"polarization: the beams here are not all within 1 % of one linear state (up to {100.0 * share:.3g} % outside it: "
            "the beam of SRC (traced SRC#0, pass 0)), and the state of their mixture where they overlap is not determined",)
    assert run_.result.targets[0].status.value == ("PASS" if match else "UNKNOWN")


def test_a_circular_rating_matches_circular_light_and_not_linear_light():
    circular = _with_source_polarization(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="circular")], targets=UTIL),
                                         {"kind": "ELLIPTICAL", "orientation_deg": 0, "ellipticity_deg": -45})
    assert _surface(run_optical_project(circular), "W/S1").ratings[0].status == "WITHIN_RATING"
    linear = run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="circular")], targets=UTIL))
    (comparison,) = _surface(linear, "W/S1").ratings
    least = _least_share(linear, 0.5)  # (1 - |s3|) / 2 of the beam's own power, less what the left-out light may take
    assert comparison.status == "CONDITIONS_MISMATCH" and f"at least {100.0 * least:.3g} %" in comparison.mismatches[0]


# ------------------------------------------------------------------------------------------- s and p at a surface


@pytest.mark.parametrize(("angle_deg", "s_match", "p_match"), [(90.0, True, False), (0.0, False, True)])
def test_s_and_p_ratings_follow_the_plane_of_incidence(angle_deg: float, s_match: bool, p_match: bool):
    """The window turned 30 deg about design z: the plane of incidence is the design x-y plane. The source's local +Y is
    design +Z (the source faces +X with ALONG_X): 90 deg is s at the window, 0 deg is p. (On the tilted window the beams
    are elliptical -- the front face's reflection from the rear face 10 % longer across one axis than the other -- so no
    one test beam is within 5 % of them all: the polarization condition itself is what follows the plane of incidence.)"""
    ratings = [_rating("CW", 1e9, "W/cm2", polarization_condition="s-polarized"),
               _rating("CW", 1e9, "W/cm2", polarization_condition="P-Pol")]
    document = _tilted(_document(ratings))
    document["components"][0]["source_model"]["polarization"]["angle_deg"] = angle_deg
    run_ = run_optical_project(document)
    for surface_id in ("W/S1", "W/S2"):
        s_rating, p_rating = _surface(run_, surface_id).ratings
        states = tuple(next(c for c in r.conditions if c.condition == "polarization").state for r in (s_rating, p_rating))
        assert states == tuple("VERIFIED" if match else "DIFFERS" for match in (s_match, p_match)), surface_id


def test_light_at_45_deg_to_the_plane_of_incidence_is_neither_s_nor_p():
    """Source at 45 deg: at the front face half the power is p. Inside the glass the Fresnel amplitude transmissions t_s,
    t_p (30 deg, N-BK7 at 1064 nm, written out here) leave t_p^2 / (t_s^2 + t_p^2) of the power in p at the rear face."""
    document = _tilted(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="s-polarized")]))
    document["components"][0]["source_model"]["polarization"]["angle_deg"] = 45.0
    run_ = run_optical_project(document)
    n = _n("N-BK7", 1064.0)
    cos_i = math.cos(math.radians(30.0))
    cos_t = math.sqrt(1.0 - (math.sin(math.radians(30.0)) / n) ** 2)
    t_s, t_p = 2.0 * cos_i / (cos_i + n * cos_t), 2.0 * cos_i / (n * cos_i + cos_t)
    p_share = t_p ** 2 / (t_s ** 2 + t_p ** 2)
    # the traced beam alone carries 50 % (front) and p_share (rear) in p; the window's own reflections, coherent with it,
    # can cancel part of that at its centre, so the decided share is a lower bound below the beam's own and above 1 %
    for surface_id, own in (("W/S1", 0.5), ("W/S2", p_share)):
        (comparison,) = _surface(run_, surface_id).ratings
        assert comparison.status == "CONDITIONS_MISMATCH", surface_id
        text = next(m for m in comparison.mismatches if m.startswith("polarization:"))
        assert text.startswith("polarization: at the centre of the beam of SRC (traced SRC#0, pass 0) at least ")
        least = float(text.split("at least ")[1].split(" %")[0]) / 100.0
        assert 0.01 < least <= own * (1.0 + 1e-3), (surface_id, least, own)


def test_light_at_45_deg_to_the_plane_of_incidence_of_a_mirror_carries_half_its_power_in_p():
    """One beam on a fold mirror at 30 deg (no other light reaches its face): the decided share is the beam's own, 50 %."""
    source = B.laser("SRC", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                     direction=(1.0, 0.0, 0.0), polarization=JonesVector.linear(45.0))
    tilt = math.radians(30.0)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-math.cos(tilt), math.sin(tilt), 0.0))
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                         basis="fixture", polarization="S")
    result = run(ProjectSnapshot(components=(source, mirror.model_copy(update={"power_ratings": (rating,)}))))
    (comparison,) = next(s for s in result.exposures.surfaces if s.surface_id == "M/face").ratings
    assert comparison.status == "CONDITIONS_MISMATCH"
    assert comparison.mismatches == ("polarization: at the centre of the beam of SRC (traced SRC#0, pass 0) at least 50 % of the "
                                     "light's power is in the state orthogonal to the rating's s-polarized polarization "
                                     "(tolerance 1 %)",)


def test_s_or_p_at_normal_incidence_is_not_verified():
    run_ = run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="s-polarized")], targets=UTIL))
    (comparison,) = _surface(run_, "W/S1").ratings
    assert (comparison.status, comparison.conditions_match, comparison.mismatches) == ("CONDITIONS_NOT_VERIFIED", False, ())
    assert any("at normal incidence, where s and p are not defined" in c for c in comparison.unverified_conditions)
    assert comparison.exposure_lower_bound is not None  # (the exposure itself is still reported)
    assert _element(run_, "W").rating_status == "CONDITIONS_NOT_VERIFIED"
    (target,) = run_.result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("POWER_RATING_CONDITIONS_NOT_VERIFIED",))
    # (the ratio to the rated value, reported as an estimate that takes the unstated test angle as the light's own)
    assert comparison.utilization == comparison.exposure_value / comparison.rating_value
    assert target.value == max(c.utilization for s in run_.result.exposures.surfaces if s.element_id == "W" for c in s.ratings)
    assert target.value_basis["value"].kind.value == "ESTIMATE"
    assert "polarization" in {name for a in target.value_basis["value"].assumed for name in a.at}
    assert "polarization" in target.message


def test_a_polarization_stated_only_as_text_is_not_verified():
    text = "linear, 45 deg to the plane of incidence"
    run_ = run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition=text)], targets=UTIL))
    (comparison,) = _surface(run_, "W/S1").ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and comparison.conditions_match is False
    assert comparison.unverified_conditions == (f"polarization: rated for {text!r}, a polarization stated as text only (the "
                                                "comparison reads linear, circular, s, p or unpolarized)",)
    assert run_.result.targets[0].status.value == "UNKNOWN"


def test_an_unpolarized_rating_does_not_match_a_polarized_beam():
    run_ = run_optical_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition="unpolarized")], targets=UTIL))
    (comparison,) = _surface(run_, "W/S1").ratings
    assert comparison.status == "CONDITIONS_MISMATCH"
    # one coherence class (the beam and the window's reflections, in one state) carries all the light at the beam's centre
    # but what the light left out as negligible may add in another state: at least (e - A)^2 in the beam's state against
    # at most A^2 in another
    s1 = _surface(run_, "W/S1")
    (hit,) = s1.hits
    e = math.sqrt(hit.peak_irradiance_W_per_cm2) - sum(math.sqrt(g.peak_irradiance_W_per_cm2) for g in s1.untraced_hits)
    a = faint_amplitude(run_, "W/cm2", surface=s1)
    share = (e - a) ** 2 / ((e - a) ** 2 + a * a)
    assert comparison.mismatches == ("polarization: the rating is for unpolarized light; at the centre of the beam of SRC (traced "
                                     f"SRC#0, pass 0) at least {100.0 * share:.3g} % of the light is in one polarization state "
                                     "(every traced beam is fully polarized)",)


# -------------------------------------------------------------------------------- coherent beams of two states


def _two_beams(pol_b_deg: float, group_b: str | None, rating_polarization: str, pol_a_deg: float = 0.0) -> ProjectSnapshot:
    """Two 1 mW 532 nm beams crossing at 3 deg in the design x-y plane on a window at normal incidence: A along +x, B
    tilted toward +y; 0 deg is in that plane (A's field along y), 90 deg along z."""
    tilt = math.radians(3.0)
    a = B.laser("A", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                coherence_group="G", polarization=JonesVector.linear(pol_a_deg))
    b = B.laser("B", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5,
                position=(-100.0 * math.cos(tilt), -100.0 * math.sin(tilt), 0.0), direction=(math.cos(tilt), math.sin(tilt), 0.0),
                coherence_group=group_b, polarization=JonesVector.linear(pol_b_deg))
    window = B.window("W", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=(0.5, 0.0, 0.0))
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=532.0, value_definition="PEAK", status="DECLARED",
                         basis="fixture", polarization=rating_polarization, beam_diameter_mm=_beam_mm(532.0),
                         beam_diameter_definition="1/E2")
    return ProjectSnapshot(components=(a, b, window.model_copy(update={"power_ratings": (rating,)})),
                           targets=(Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),))


def test_coherent_beams_of_different_linear_states_leave_a_linear_condition_unverified():
    """Two linear beams of one coherence group, crossed: their sum is elliptical or linear depending on their phase."""
    crossed = run(_two_beams(90.0, "G", "LINEAR"))
    (comparison,) = next(s for s in crossed.exposures.surfaces if s.surface_id == "W/front").ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED"
    assert any("different polarization states" in c for c in comparison.unverified_conditions)
    assert crossed.targets[0].status.value == "UNKNOWN"
    # both polarized along z, across the plane in which they cross: one state, their sum is that state whatever the phase
    # (the polarization condition itself: on the front face light of unknown size may carry just over 1 % of the most
    # light, which leaves the rating's test beam not verifiable there)
    parallel = run(_two_beams(90.0, "G", "LINEAR", pol_a_deg=90.0))
    front = next(s for s in parallel.exposures.surfaces if s.surface_id == "W/front").ratings[0]
    assert next(c for c in front.conditions if c.condition == "polarization").state == "VERIFIED"
    # both polarized in that plane, each across its own direction: their fields differ by the 3 deg between the beams, and
    # where their phases cancel the rest -- 2 sin(1.5 deg) of one field, along x -- is all the light (review finding 1: a
    # field within 1 % of the other's is not enough)
    in_plane = run(_two_beams(0.0, "G", "LINEAR"))
    (comparison,) = next(s for s in in_plane.exposures.surfaces if s.surface_id == "W/front").ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED")
    assert in_plane.targets[0].status.value == "UNKNOWN"


def test_incoherent_beams_of_crossed_states_do_not_hold_a_linear_or_circular_condition():
    """Verifier finding (P3): two mutually incoherent sources with orthogonal linear states were 'linear at every beam',
    WITHIN, PASS -- yet where they overlap their mixture is partly unpolarized. Each beam alone is linear, so whether the
    light at a point holds the condition depends on how much of each is there: not verified, never matched."""
    independent = run(_two_beams(90.0, None, "LINEAR"))
    (comparison,) = next(s for s in independent.exposures.surfaces if s.surface_id == "W/front").ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and not comparison.conditions_match and comparison.mismatches == ()
    assert any("not all within 1 % of one linear state" in c for c in comparison.unverified_conditions)
    assert independent.targets[0].status.value == "UNKNOWN"
    def at_normal_incidence(project: ProjectSnapshot) -> ProjectSnapshot:
        # (B parallel to A, 1 mm beside it: at normal incidence each beam's ghost inside the window keeps its state,
        # so only the two sources' states matter -- at 3 deg B's ghost inside the glass is in another state than B
        # and displaced from it on the face, and their combined state varies across it)
        a, b, window = project.components
        b = B.laser("B", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 1.0, 0.0),
                    direction=(1.0, 0.0, 0.0), coherence_group=b.source.coherence_group, polarization=b.source.polarization)
        return project.model_copy(update={"components": (a, b, window)})

    def polarization_of(result) -> tuple[str, str | None]:
        (comparison,) = next(s for s in result.exposures.surfaces if s.surface_id == "W/front").ratings
        condition = next(c for c in comparison.conditions if c.condition == "polarization")
        return condition.state, condition.reason

    # the same two sources in one linear state: any mixture of them is that state
    assert polarization_of(run(at_normal_incidence(_two_beams(0.0, None, "LINEAR")))) == ("VERIFIED", None)
    assert polarization_of(run(_two_beams(0.0, None, "LINEAR"))) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED")
    # opposite circular states (right and left about the direction of travel): their mixture is not circular
    a, b, window = _two_beams(0.0, None, "CIRCULAR").components
    a = a.model_copy(update={"source": a.source.model_copy(update={"polarization": JonesVector.right_circular()})})
    b = b.model_copy(update={"source": b.source.model_copy(update={"polarization": JonesVector.left_circular()})})
    opposite = run(ProjectSnapshot(components=(a, b, window), targets=_two_beams(0.0, None, "CIRCULAR").targets))
    (comparison,) = next(s for s in opposite.exposures.surfaces if s.surface_id == "W/front").ratings
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and opposite.targets[0].status.value == "UNKNOWN"
    b_same = b.model_copy(update={"source": b.source.model_copy(update={"polarization": JonesVector.right_circular()})})
    same = ProjectSnapshot(components=(a, b_same, window), targets=_two_beams(0.0, None, "CIRCULAR").targets)
    assert polarization_of(run(at_normal_incidence(same))) == ("VERIFIED", None)


# ------------------------------------------------------------------------------------- reading the stated text


def test_plain_polarization_statements_are_read_and_anything_else_is_left_as_text():
    """Plain statements become the machine-readable condition (``TE`` / ``TM`` are s / p at the rated surface); a statement
    that says more (an orientation, a handedness) stays text the comparison cannot check."""
    assert {text: rating_polarization(text) for text in ("s-polarized", "S-Pol", " p-polarized ", "Linear", "linearly  polarized",
                                                          "Circularly polarized", "unpolarized", "randomly polarized", "s", "TE",
                                                          "P-pol.", "TM")} == {
        "s-polarized": "S", "S-Pol": "S", " p-polarized ": "P", "Linear": "LINEAR", "linearly  polarized": "LINEAR",
        "Circularly polarized": "CIRCULAR", "unpolarized": "UNPOLARIZED", "randomly polarized": "UNPOLARIZED", "s": "S",
        "TE": "S", "P-pol.": "P", "TM": "P"}
    for text in ("linear, 45 deg to the plane of incidence", "circular (RHC)", "TE (s-polarized)", "", None):
        assert rating_polarization(text) is None


def test_the_catalog_s_pol_statement_becomes_a_machine_readable_s_condition_and_other_ratings_keep_their_input():
    published = PowerRatingV1.model_validate({"regime": "PULSED", "value": "0.081", "unit": "J/cm2", "wavelength_nm": "400",
                                              "pulse_duration_s": "0.000000000000055", "pulse_count": 1,
                                              "polarization_condition": "s-polarized", "status": "SPECIFIED",
                                              "basis": "Manufacturer-stated laser-induced damage threshold"})
    (rating,) = engine_power_ratings((published,), "catalog product X")
    assert rating.polarization == "S" and rating.polarization_condition == "s-polarized"
    assert rating.model_dump(mode="json")["polarization"] == "S"
    # a rating without a polarization keeps its Engine input exactly (the field is omitted, the request hash unchanged)
    request = lower_project(_document([_rating("CW", 1e9, "W/cm2", polarization_condition=None)])).request.model_dump(mode="json")
    (window,) = [c for c in request["project"]["components"] if c["instance_id"] == "W"]
    assert "polarization" not in window["power_ratings"][0]



# -------------------------------------------------- followed light of another state: the share rule (review M2)


def _beside_state(offset_mm: float | None, crossed_W: float | None = None) -> ProjectSnapshot:
    """The review's ``p4_beside_polarization``: a linearly polarized (x) 1 W 1064 nm beam on a window rated for linear light
    (fixture) and a second laser polarized along y, running ``offset_mm`` off the axis outside the clear aperture (its tail
    reaching the window), or ``crossed_W`` of it combined onto the axis by a splitter."""
    inv = 1.0 / math.sqrt(2.0)
    comps = [B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))]
    if offset_mm is not None:
        comps += [B.laser("B", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, offset_mm, 0.0),
                          direction=(1.0, 0.0, 0.0), polarization=JonesVector.linear(90.0)),
                  B.beam_dump("D", position=(100.0, offset_mm, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=3.0)]
    if crossed_W is not None:
        comps += [B.laser("B", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-50.0, -100.0, 0.0),
                          direction=(0.0, 1.0, 0.0), polarization=JonesVector.linear(90.0)),  # (across A's after the fold)
                  B.npbs("S", reflectance=crossed_W, position=(-50.0, 0.0, 0.0), normal=(-inv, inv, 0.0)),
                  B.beam_dump("D", position=(-50.0, 60.0, 0.0), axis=(0.0, 1.0, 0.0))]
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    rating = PowerRating(regime="CW", value=25000.0, unit="W/cm2", wavelength_nm=1064.0, polarization_condition="linear",
                         polarization="LINEAR", value_definition="PEAK", status="DECLARED", basis="test fixture rating, not product data",
                         beam_diameter_mm=_beam_mm(1064.0), beam_diameter_definition="1/E2")
    window = window.model_copy(update={"power_ratings": (rating,)})
    return ProjectSnapshot(components=(*comps, window),
                           targets=(Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),))


@pytest.mark.parametrize(("offset_mm", "crossed_W", "held"), [(17.0, None, True), (None, 1e-4, True), (None, 0.1, False)],
                         ids=["beside", "crossed1e-4", "crossed0.1"])
def test_followed_light_of_another_state_below_the_share_counts_in_the_exposure_not_in_the_polarization(offset_mm, crossed_W,
                                                                                                       held: bool):
    """Review M2: an independent beam polarized across the rating's axis, 17 mm beside the window (its tail there about
    2e-16 W/cm2), left the polarization not verifiable. Now, as light left out as negligible and minor beams: at most 1 %
    of the most light the surface can receive, it counts in the exposure but not in the state of the light as a whole -- the
    polarization is verified, with a note naming that beam and the share outside the state; so is 1e-4 W of crossed light
    combined onto the axis. 10 % of crossed light is not negligible: the polarization is not verifiable, as before (the
    beams are not all within 1 % of one linear state; the mixture of two independent sources is not decided)."""
    result = run(_beside_state(offset_mm, crossed_W))
    for surface in (s for s in result.exposures.surfaces if s.element_id == "W"):
        (comparison,) = surface.ratings
        polarization = next(c for c in comparison.conditions if c.condition == "polarization")
        if held:
            assert (polarization.state, comparison.applicability, comparison.status) == ("VERIFIED", "AS_RATED", "WITHIN_RATING")
            assert "at every beam here but the beam of B" in polarization.message and "counts in the exposure" in polarization.message
        else:
            assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "LIGHT_STATE_UNKNOWN")
            assert comparison.applicability == "NOT_VERIFIED"
    assert result.targets[0].status.value == ("PASS" if held else "UNKNOWN")


def _tilted_elliptical(chi_deg: float, other_W: float) -> ProjectSnapshot:
    """A window tilted 30 deg, a P rating on its front face (fixture): an elliptical 1064 nm beam of ellipticity
    ``chi_deg`` with its major axis in the plane of incidence, its own ghosts coherent with it in slightly different states,
    and ``other_W`` of an independent s-polarized beam crossing it at 2 deg."""
    tilt, cross = math.radians(30.0), math.radians(2.0)
    comps = [B.laser("A", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                     direction=(1.0, 0.0, 0.0), polarization=JonesVector.elliptical(0.0, chi_deg))]
    if other_W:
        comps.append(B.laser("B", wavelength_nm=1064.0, power_W=other_W, waist_radius_mm=0.5,
                             position=(-100.0 * math.cos(cross), -100.0 * math.sin(cross), 0.0),
                             direction=(math.cos(cross), math.sin(cross), 0.0), polarization=JonesVector.linear(90.0)))
    window = B.window("W", thickness_mm=2.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0),
                      axis=(math.cos(tilt), math.sin(tilt), 0.0))
    # (the test beam: between A's beam on the front face, 2 w 100 mm from the waist, and the rear face's reflection there,
    # wider in the plane of incidence by cos(theta_t) / cos(theta_i) (Snell, 30 deg into N-BK7): their geometric mean,
    # within 5 % of both axes of each)
    sin_t = math.sin(tilt) / _n("N-BK7", 1064.0)
    widening = math.sqrt(1.0 - sin_t * sin_t) / math.cos(tilt)
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=1064.0, value_definition="PEAK", status="DECLARED",
                         basis="test fixture rating, not product data", polarization="P", surface_ids=("front",),
                         beam_diameter_mm=_beam_mm(1064.0) * math.sqrt(widening), beam_diameter_definition="1/E2")
    window = window.model_copy(update={"power_ratings": (rating,)})
    return ProjectSnapshot(components=(*comps, window),
                           targets=(Target(target_id="u", device_id="W", metric="power_rating_utilization", maximum=1.0),))


@pytest.mark.parametrize(("chi_deg", "other_W", "expected"),
                         [(0.0, 0.002, "VERIFIED"), (0.0, 0.0084, "VERIFIED"), (0.0, 0.02, "DIFFERS"),
                          (3.8, 0.0, "COHERENT_STATE_UNDETERMINED"), (2.0, 0.002, "COHERENT_STATE_UNDETERMINED")])
def test_beams_of_another_state_are_left_out_only_when_the_light_outside_it_stays_within_the_share(chi_deg: float, other_W: float,
                                                                                                     expected: str):
    """Beams of another state count in the exposure, not in the rating's polarization, only when the light outside the
    state as a whole stays within 1 % of the most light. p light (chi 0): A's beam and its ghosts inside the window are
    all p, so their sum is p wherever they overlap; 0.2 % or 0.84 % of independent s light (B) counts in the exposure,
    with a note naming the beams left out; with 2 % of s light at least 1.02 % of the light at B's centre is s whatever
    the phases: the condition differs. Elliptical light (chi 2 or 3.8 deg): A's ghosts inside the window are coherent
    with it, in other states, and displaced from it on the face, so the state of their sum varies across the face and is
    not verified, whatever the main beam's own share outside p."""
    result = run(_tilted_elliptical(chi_deg, other_W))
    surface = next(s for s in result.exposures.surfaces if s.surface_id == "W/front")
    (comparison,) = surface.ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    target = result.targets[0].status.value
    if expected == "VERIFIED":
        assert (polarization.state, comparison.status, target) == ("VERIFIED", "WITHIN_RATING", "PASS")
        assert "counts in the exposure, not in the rating's polarization" in polarization.message
    elif expected == "DIFFERS":
        assert (polarization.state, comparison.status, target) == ("DIFFERS", "CONDITIONS_MISMATCH", "UNKNOWN")
        assert "the beam of B" in polarization.message
    else:
        assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED")
        assert "with different footprints" in polarization.message
        assert (comparison.status, target) == ("CONDITIONS_NOT_VERIFIED", "UNKNOWN")
