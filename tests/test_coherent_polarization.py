"""Mutually coherent beams hold a rating's test polarization only as a sum (review, finding 1;
``validation/exposures.py``: ``_coherent_bound``).

Two equal, collinear, mutually coherent beams linear at +2 deg and -2 deg are each within 0.49 % of one linear axis, yet
with a relative Jones phase of pi - 4 deg their sum is circular (|S3| = 1): a LINEAR rating was VERIFIED, WITHIN_RATING,
and its utilization target PASS. The model leaves the relative phase of coherent beams open, so a coherent group's
condition is now checked on its sum for every phase: in the rating's basis the part outside the state is at most
``U = sum sqrt(f_i) a_i`` and the part in it at least ``W = sqrt(1 - f_j) a_j - sum_{i != j} sqrt(1 - f_i) a_i``, so at
most ``U^2 / (U^2 + W^2)`` of the sum is outside it; in the state of one beam ``j`` the sum keeps at least ``B = a_j -
sum a_i |e_j^* . e_i|`` and the rest is at most ``D = sum a_i sqrt(1 - |e_j^* . e_i|^2)`` -- at most ``(B sqrt(f_j) +
D)^2 / (B^2 + D^2)`` outside (``a`` field amplitudes, square roots of the peak irradiances; ``f`` a beam's own share
outside the state). The smaller bound stands; above the tolerance the condition is not verifiable
(``COHERENT_STATE_UNDETERMINED``), never verified. For two beams, one of them in the state, both bounds equal the largest
share over the relative phase written out below, so the reported share is the exact worst case. Incoherent mixtures are
unchanged (their powers add). Every rating here is fixture data declared in the test project (its value stated as a
peak irradiance), never product data.
"""

from __future__ import annotations

import cmath
import math
import random

import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, PowerRating, ProjectSnapshot, Target

from helpers import run

THETA = math.radians(2.0)


def _beam_mm(waist_mm: float, path_mm: float = 100.0, wavelength_nm: float = 532.0) -> float:
    """A fixture laser's 1/e^2 diameter ``path_mm`` from its waist: 2 w0 sqrt(1 + (z / z_R)^2), z_R = pi w0^2 / lambda (the
    ratings' test beam, so that their polarization alone decides whether they apply)."""
    return 2.0 * waist_mm * math.sqrt(1.0 + (path_mm / (math.pi * waist_mm ** 2 / (wavelength_nm * 1e-6))) ** 2)


def _jones(angle: float, phase: float = 0.0) -> JonesVector:
    ex, ey = math.cos(angle) * cmath.exp(1j * phase), math.sin(angle) * cmath.exp(1j * phase)
    return JonesVector(ex=(ex.real, ex.imag), ey=(ey.real, ey.imag))


def _mirror_project(pol_a: JonesVector, pol_b: JonesVector, condition: str, *, power_b_W: float = 1e-3,
                    group: str | None = "G") -> ProjectSnapshot:
    """The review's scene: two lasers on one ray (532 nm, waist 0.5 mm, from (-100, 0, 0) along +x) onto a fold mirror at 45
    deg (plane of incidence x-y: s along z, the sources' e2; their e1 along y, p at the mirror), a rating of ``condition``
    on its face (fixture) and a utilization target. ``group``: both sources in that coherence group (None: independent)."""
    sources = [B.laser(name, wavelength_nm=532.0, power_W=power, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                       direction=(1.0, 0.0, 0.0), coherence_group=group, polarization=pol)
               for name, power, pol in (("A", 1e-3, pol_a), ("B", power_b_W, pol_b))]
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=532.0, value_definition="PEAK", status="DECLARED",
                         basis="test fixture rating, not product data", polarization=condition, beam_diameter_mm=_beam_mm(0.5),
                         beam_diameter_definition="1/E2")
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0))
    return ProjectSnapshot(components=(*sources, mirror.model_copy(update={"power_ratings": (rating,)})),
                           targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),))


def _compared(result):
    (surface,) = [s for s in result.exposures.surfaces if s.surface_id == "M/face"]
    (comparison,) = surface.ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    return surface, comparison, polarization


def _worst_over_phase(a: float, b: float, f: float) -> float:
    """The largest share outside a state of ``a e_A + b exp(i phi) e_B`` over phi, ``e_A`` in the state and ``e_B`` with
    ``f`` of its power outside it: ``|P_out E|^2 = b^2 f`` whatever phi, and ``|E|^2`` is least, ``(a - b sqrt(1 - f))^2 +
    b^2 f``, where the parts in the state cancel."""
    return b * b * f / ((a - b * math.sqrt(1.0 - f)) ** 2 + b * b * f)


# ---------------------------------------------------------------------------------------------------------- the review


def test_the_review_case_two_coherent_beams_at_plus_and_minus_2_deg_do_not_verify_a_linear_rating():
    """The review reproduction: A linear +2 deg, B linear -2 deg with Jones phase pi - 4 deg, one coherence group:
    their sum is circular. It was conditions_match true, WITHIN_RATING, target PASS; now the polarization is not verifiable
    (the comparison CONDITIONS_NOT_VERIFIED, the target UNKNOWN), and the message gives the most of their light that may be
    outside the linear axis: sin^2(4 deg) / ((1 - cos 4 deg)^2 + sin^2(4 deg)) = 99.9 % (at phase pi their sum is along
    the axis 90 deg from A's)."""
    pol_a, pol_b = _jones(THETA), _jones(-THETA, math.pi - 2.0 * THETA)
    ex = math.cos(THETA) * (1.0 + cmath.exp(1j * (math.pi - 2.0 * THETA)))
    ey = math.sin(THETA) * (1.0 - cmath.exp(1j * (math.pi - 2.0 * THETA)))
    assert abs(2.0 * (ex.conjugate() * ey).imag) / (abs(ex) ** 2 + abs(ey) ** 2) == pytest.approx(1.0, abs=1e-12)  # circular
    result = run(_mirror_project(pol_a, pol_b, "LINEAR"))
    surface, comparison, polarization = _compared(result)
    assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED")
    assert (comparison.conditions_match, comparison.applicability, comparison.status) == (False, "NOT_VERIFIED",
                                                                                         "CONDITIONS_NOT_VERIFIED")
    a = math.sqrt(surface.hits[0].peak_irradiance_W_per_cm2)
    worst = _worst_over_phase(a, a, math.sin(2.0 * THETA) ** 2)
    assert worst == pytest.approx(0.99878, abs=1e-5)
    assert polarization.message == ("polarization: mutually coherent beams of A, B in different polarization states meet "
                                    "M/face; their combined state depends on their relative phase, and up to "
                                    f"{100.0 * worst:.3g} % of their light may be outside the rating's linear polarization "
                                    "(tolerance 1 %)")
    assert comparison.exposure_upper_bound is not None  # (the exposure itself is still reported)
    (target,) = result.targets
    assert (target.status.value, target.issue_codes) == ("UNKNOWN", ("POWER_RATING_CONDITIONS_NOT_VERIFIED",))


def test_the_relative_jones_phase_does_not_matter_the_model_leaves_it_open():
    """The same beams with no relative Jones phase (their sum linear for that phase) give the same result: the trace does
    not fix the relative phase of mutually coherent beams, so every phase counts."""
    with_phase = _compared(run(_mirror_project(_jones(THETA), _jones(-THETA, math.pi - 2.0 * THETA), "LINEAR")))[2]
    without = _compared(run(_mirror_project(_jones(THETA), _jones(-THETA), "LINEAR")))[2]
    assert with_phase == without


def test_the_same_beams_mutually_incoherent_still_verify_a_linear_rating():
    """Without a common coherence group their powers add: the mixture's share outside A's axis is the power-weighted mean
    of the beams' own (0 and sin^2(4 deg) = 0.487 %), so the polarization is verified as before."""
    result = run(_mirror_project(_jones(THETA), _jones(-THETA, math.pi - 2.0 * THETA), "LINEAR", group=None))
    _, comparison, polarization = _compared(result)
    assert polarization.state == "VERIFIED" and comparison.status == "WITHIN_RATING"
    assert polarization.message == ("polarization: linear along one common axis at every beam here (largest share outside it "
                                    f"{100.0 * math.sin(2.0 * THETA) ** 2:.3g} %)")
    assert result.targets[0].status.value == "PASS"


# ------------------------------------------------------------------------------ every condition: S / P / LINEAR / CIRCULAR

#: (condition, the state of A, a state of B at the tolerance's edge or beyond, B's own share outside the state)
_STATES = {
    "LINEAR": (JonesVector.linear(0.0), JonesVector.linear(4.0), math.sin(math.radians(4.0)) ** 2),
    "S": (JonesVector.linear(90.0), JonesVector.linear(86.0), math.sin(math.radians(4.0)) ** 2),
    "P": (JonesVector.linear(0.0), JonesVector.linear(4.0), math.sin(math.radians(4.0)) ** 2),
    "CIRCULAR": (JonesVector.elliptical(0.0, -45.0), JonesVector.elliptical(0.0, -41.0),
                 0.5 * (1.0 - math.sin(math.radians(82.0)))),
}
#: (condition, a weak ghost's state far from the rating's, its own share outside it)
_GHOSTS = {
    "LINEAR": (JonesVector.linear(10.0), math.sin(math.radians(10.0)) ** 2),
    "S": (JonesVector.linear(80.0), math.sin(math.radians(10.0)) ** 2),
    "P": (JonesVector.linear(10.0), math.sin(math.radians(10.0)) ** 2),
    "CIRCULAR": (JonesVector.elliptical(0.0, -30.0), 0.5 * (1.0 - math.sin(math.radians(60.0)))),
}
_TEXT = {"LINEAR": "linear along one common axis", "S": "s-polarized", "P": "p-polarized", "CIRCULAR": "circular"}
_WANTED = {"LINEAR": "linear", "S": "s-polarized", "P": "p-polarized", "CIRCULAR": "circular"}


@pytest.mark.parametrize("condition", ["LINEAR", "S", "P", "CIRCULAR"])
def test_equal_coherent_beams_in_slightly_different_states_are_not_verifiable(condition: str):
    """Two equal coherent beams, A in the state and B 0.487 % outside it (4 deg off; for CIRCULAR an ellipticity of 41
    deg): where their parts in the state cancel, their sum is B's part outside it -- up to 99.9 %. Not verifiable, never
    verified, the target UNKNOWN."""
    pol_a, pol_b, f = _STATES[condition]
    result = run(_mirror_project(pol_a, pol_b, condition))
    surface, comparison, polarization = _compared(result)
    assert (polarization.state, polarization.reason) == ("NOT_VERIFIABLE", "COHERENT_STATE_UNDETERMINED"), polarization.message
    a = math.sqrt(surface.hits[0].peak_irradiance_W_per_cm2)
    worst = _worst_over_phase(a, a, f)
    assert f"up to {100.0 * worst:.3g} % of their light may be outside the rating's {_WANTED[condition]} polarization" in (
        polarization.message)
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and result.targets[0].status.value == "UNKNOWN"


@pytest.mark.parametrize("condition", ["LINEAR", "S", "P", "CIRCULAR"])
def test_a_dominant_beam_with_a_weak_coherent_ghost_of_another_state_is_verified_by_the_bound(condition: str):
    """A 1 mW beam in the state and a coherent 1 uW ghost far from it (10 deg off: 3 % of its own power outside the state;
    for CIRCULAR an ellipticity of 30 deg, 6.7 %): whatever their phase the ghost can take at most its amplitude from the
    sum's part in the state, so at most ``_worst_over_phase`` = 3e-5 of the sum is outside it -- verified, WITHIN, PASS,
    with the bound in the note (the ghost's own share alone would have left it not verifiable)."""
    pol_a = _STATES[condition][0]
    pol_b, f = _GHOSTS[condition]
    assert f > 0.01
    result = run(_mirror_project(pol_a, pol_b, condition, power_b_W=1e-6))
    surface, comparison, polarization = _compared(result)
    assert polarization.state == "VERIFIED", polarization.message
    a, b = (math.sqrt(h.peak_irradiance_W_per_cm2) for h in sorted(surface.hits, key=lambda h: -h.incident_power_W))
    worst = _worst_over_phase(a, b, f)
    assert worst < 1e-4
    assert polarization.message == (f"polarization: {_TEXT[condition]} here: the mutually coherent beams of A, B, in different "
                                    f"states, have at most {100.0 * worst:.3g} % of their light outside it whatever their "
                                    "relative phase")
    assert (comparison.conditions_match, comparison.status, result.targets[0].status.value) == (True, "WITHIN_RATING", "PASS")


@pytest.mark.parametrize("condition", ["LINEAR", "S", "P", "CIRCULAR"])
def test_equal_coherent_beams_of_one_state_are_verified(condition: str):
    """Two equal coherent beams in one state: their sum is that state whatever their phase (it may cancel, it cannot turn)."""
    pol_a = _STATES[condition][0]
    result = run(_mirror_project(pol_a, pol_a, condition))
    _, comparison, polarization = _compared(result)
    assert polarization.state == "VERIFIED" and polarization.message.startswith(f"polarization: {_TEXT[condition]} at every beam")
    assert (comparison.status, result.targets[0].status.value) == ("WITHIN_RATING", "PASS")


# ------------------------------------------------------------------------------------------ the bound against the phases


def _field(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """A random transverse field (real parts, imaginary parts) in the x-y plane."""
    x = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    y = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    return (x.real, y.real, 0.0), (x.imag, y.imag, 0.0)


@pytest.mark.parametrize("basis", ["LINEAR", "CIRCULAR"])
def test_the_bound_is_never_below_the_share_any_relative_phase_gives(basis: str):
    """``_bound_in`` against brute force: 300 random groups of two to four coherent beams (random transverse states,
    amplitudes from comparable to one dominant), the largest share outside the state (along x; the circular state (x - i
    y) / sqrt 2) over 400 random phase sets and, for two beams, the exact worst case. The bound is never below it."""
    from os_optics.validation.exposures import _State, _bound_in

    rng = random.Random(360)
    circ = (1.0 / math.sqrt(2.0), -1j / math.sqrt(2.0))
    tight = 0
    for _ in range(300):
        n = rng.randint(2, 4)
        fields = [_field(rng) for _ in range(n)]
        states = []
        f: dict[int, float] = {}
        amp: dict[int, float] = {}
        vectors = []
        for i, (re, im) in enumerate(fields):
            norm2 = sum(v * v for v in re) + sum(v * v for v in im)
            states.append(_State(norm2, None, 0.0, None, (re, im)))  # type: ignore[arg-type]
            e = (complex(re[0], im[0]) / math.sqrt(norm2), complex(re[1], im[1]) / math.sqrt(norm2))
            vectors.append(e)
            inside = abs(e[0]) ** 2 if basis == "LINEAR" else abs(circ[0].conjugate() * e[0] + circ[1].conjugate() * e[1]) ** 2
            f[i] = min(1.0, max(0.0, 1.0 - inside))
            amp[i] = rng.choice((1.0, rng.uniform(0.0, 1.0), rng.uniform(0.0, 0.05)))
        group = tuple(range(n))
        bound = _bound_in(group, f, amp, states, list(group), list(group))

        def outside(phases) -> float:
            x = sum(amp[i] * vectors[i][0] * cmath.exp(1j * phases[i]) for i in group)
            y = sum(amp[i] * vectors[i][1] * cmath.exp(1j * phases[i]) for i in group)
            total = abs(x) ** 2 + abs(y) ** 2
            if total <= 1e-300:
                return 0.0
            inside = abs(x) ** 2 if basis == "LINEAR" else abs(circ[0].conjugate() * x + circ[1].conjugate() * y) ** 2
            return 1.0 - inside / total

        worst = max(outside([0.0] + [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n - 1)]) for _ in range(400))
        if n == 2:
            worst = max(worst, max(outside([0.0, 2.0 * math.pi * k / 2000]) for k in range(2000)))
        assert bound >= worst - 1e-9, (n, amp, f, bound, worst)
        tight += bound < 0.5 and bound <= 2.0 * worst + 1e-6
    assert tight > 20  # (the bound is near the worst case where one beam dominates, not trivially 1)


def _shifted_or_wide(group: str | None, rating_polarization: str, *, wide: bool) -> ProjectSnapshot:
    """Verifier fixture of the coherent bound: A 1 mW s-polarized beam of waist 0.3 mm and B 0.09 mW 13 deg off s on a
    45 deg mirror, B either 0.6 mm beside A (two radii, ``wide`` False) or on A's axis with a waist of 1 mm (``wide``)."""
    a = B.laser("A", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.3, position=(-100.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_group=group, polarization=JonesVector.linear(90.0))
    b = B.laser("B", wavelength_nm=532.0, power_W=0.09e-3, waist_radius_mm=1.0 if wide else 0.3,
                position=(-100.0, 0.0, 0.0 if wide else 0.6), direction=(1.0, 0.0, 0.0), coherence_group=group,
                polarization=JonesVector.linear(77.0))
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=532.0, value_definition="PEAK", status="DECLARED",
                         basis="test fixture rating, not product data", polarization=rating_polarization)
    m = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    return ProjectSnapshot(components=(a, b, m),
                           targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),))


@pytest.mark.parametrize("rating_polarization", ["S", "LINEAR", "CIRCULAR"])
@pytest.mark.parametrize("wide", [False, True])
def test_coherent_beams_of_different_profiles_never_verify_a_polarization_their_sum_may_not_have(wide: bool,
                                                                                               rating_polarization: str):
    """Verifier finding on the coherent bound: it held the beams' peak amplitudes at one point, so a coherent beam of
    another state displaced from the main beam (or wider than it) passed -- although at B's centre at least 4.4 % of the
    light is p whatever the phase (S: a certain mismatch, as for the same beams mutually incoherent). Beams of different
    profiles change their amplitude ratio across the surface: their combined state is never verified from one point."""
    result = run(_shifted_or_wide("G", rating_polarization, wide=wide))
    surface = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    (comparison,) = surface.ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    assert polarization.state != "VERIFIED" and not comparison.conditions_match
    assert comparison.status in ("CONDITIONS_MISMATCH", "CONDITIONS_NOT_VERIFIED")
    assert result.targets[0].status.value == "UNKNOWN"
    if rating_polarization == "S" and not wide:
        assert polarization.state == "DIFFERS" and "at the centre of the beam of B" in polarization.message
    elif polarization.state == "NOT_VERIFIABLE" and polarization.reason == "COHERENT_STATE_UNDETERMINED":
        assert "with different footprints" in polarization.message


@pytest.mark.parametrize("rating_polarization", ["LINEAR", "S", "P"])
def test_coherent_beams_from_different_directions_have_different_footprints(rating_polarization: str):
    """Verifier finding on the one-profile rule: two coherent beams meeting a mirror at one point with one transverse
    radius but from different directions (A at normal incidence, B at 73.7 deg) passed as one profile, although their
    footprints on the surface differ (0.50 mm and 1.79 mm); 0.5 mm from the centre for some relative phase 6.8 % of the
    light is outside the linear axis. The profile is compared on the surface, so their combined state is not verified."""
    a = B.laser("A", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_group="G", polarization=JonesVector.linear(90.0))
    b = B.laser("B", wavelength_nm=532.0, power_W=1.44e-5, waist_radius_mm=0.5, position=(-28.0, 96.0, 0.0),
                direction=(0.28, -0.96, 0.0), coherence_group="G", polarization=JonesVector.linear(45.0))
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=532.0, value_definition="PEAK", status="DECLARED",
                         basis="test fixture rating, not product data", polarization=rating_polarization)
    m = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)).model_copy(update={"power_ratings": (rating,)})
    result = run(ProjectSnapshot(components=(a, b, m), targets=(Target(target_id="u", device_id="M",
                                                                       metric="power_rating_utilization", maximum=1.0),)))
    (comparison,) = next(s for s in result.exposures.surfaces if s.surface_id == "M/face").ratings
    polarization = next(c for c in comparison.conditions if c.condition == "polarization")
    assert polarization.state != "VERIFIED" and not comparison.conditions_match
    assert result.targets[0].status.value == "UNKNOWN"


def _curved(ratio: float, *, reflected: bool) -> ProjectSnapshot:
    """Verifier fixture on a curved surface: A 1 mW s-polarized and B ``ratio`` mW p-polarized, coherent, both of waist 2
    mm, aimed at the vertex of a concave mirror (R = 20 mm) at 45 deg -- B along A's mirror image (``reflected``: a beam
    and its reflection) or along A itself."""
    c, s = math.cos(math.radians(45.0)), math.sin(math.radians(45.0))
    b_from, b_along = ((-100.0 * c, -100.0 * s, 0.0), (c, s, 0.0)) if reflected else ((-100.0 * c, 100.0 * s, 0.0), (c, -s, 0.0))
    a = B.laser("A", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=2.0, position=(-100.0 * c, 100.0 * s, 0.0),
                direction=(c, -s, 0.0), coherence_group="G", polarization=JonesVector.linear(90.0))
    b = B.laser("B", wavelength_nm=532.0, power_W=ratio * 1e-3, waist_radius_mm=2.0, position=b_from, direction=b_along,
                coherence_group="G", polarization=JonesVector.linear(0.0))
    rating = PowerRating(regime="CW", value=1e9, unit="W/cm2", wavelength_nm=532.0, value_definition="PEAK", status="DECLARED",
                         basis="test fixture rating, not product data", polarization="S", beam_diameter_mm=_beam_mm(2.0),
                         beam_diameter_definition="1/E2")
    m = B.spherical_mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=20.0,
                           clear_radius_mm=12.7).model_copy(update={"power_ratings": (rating,)})
    return ProjectSnapshot(components=(a, b, m),
                           targets=(Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=1.0),))


def test_on_a_curved_surface_a_beam_and_its_reflection_are_not_one_profile():
    """Verifier finding on the one-profile rule: on a concave mirror (R = 20 mm) a coherent beam and its mirror image at
    45 deg have one footprint in the tangent plane at the vertex, so the bound at the centre (0.99 % outside s) VERIFIED an
    S rating -- but the sphere leaves that plane, their amplitude ratio changes across it, and 2.5 mm from the vertex
    some relative phase puts up to 3.3 % of the light outside s. On a curved surface one profile needs the same profile in
    space (direction, radii, axes)."""
    surface, comparison, polarization = _compared(run(_curved(0.0100, reflected=True)))
    assert polarization.state == "NOT_VERIFIABLE" and polarization.reason == "COHERENT_STATE_UNDETERMINED"
    assert "with different footprints" in polarization.message and not comparison.conditions_match


def test_on_a_curved_surface_coherent_beams_identical_in_space_are_verified_by_the_bound():
    """Control: the same weak coherent p beam along A itself is one profile with it everywhere, so the bound holds."""
    result = run(_curved(0.002, reflected=False))
    surface, comparison, polarization = _compared(result)
    assert polarization.state == "VERIFIED" and comparison.conditions_match
    assert result.targets[0].status.value == "PASS"
