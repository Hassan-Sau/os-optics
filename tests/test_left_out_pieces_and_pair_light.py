"""Light the trace leaves out comes in pieces that add their amplitudes, and it reaches a pair's fringes too.

* The light of a line left out as negligible is left out in pieces -- each branch not followed, each path stopped at the
  power floor --, all mutually coherent. Where several meet, their fields add in amplitude: the bound per piece (its
  diffraction limit times the return factor of the surface) adds over the pieces as the sum of their square roots,
  ``in_phase_power_W = (sum sqrt p)^2``, never the square root of their total power. The verifier's 1-D check: two mirrors
  of R = 0.9 fed at two places reach 1.45 times what one place of the same total power gives.
* A pair's irradiance at its peak position, its local contrast there and its power-weighted visibility over the plane
  are light on the plane, which light of their wavelength that the trace does not follow (untraced ghosts, the left-out
  light) adds to: their targets are graded on the range it leaves, and a run with such a target follows that light. The
  verifier's rated interferometer PASSed a maximum at 1.0001 times the pair's peak although the left-out light the run
  itself bounds may raise it by a third, and a visibility minimum at 0.9999 times the pair's.

The 1-D model, the Minkowski ranges and the diffraction limit are written out. Every power rating is fixture data
declared in the test project, never product data.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import InterferenceResult, ProjectSnapshot, Status, Target
from os_optics.contracts.materials import ConstantCoating
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import SourceInput, TraceOutput
from os_optics.scene.lower import lower_scene
from os_optics.validation import resonance
from os_optics.validation.metrics import _visibility_light
from os_optics.validation.power import LineFate
from os_optics.validation.untraced import Probes, surface_reach

from helpers import run
from test_coherent_exposure import _rated_mzi

LAM = 1064.0


# ----------------------------------------------------------------------------------------------- pieces add amplitudes


def test_the_left_out_light_counts_with_the_sum_of_its_pieces_amplitudes():
    """Two pieces of 1e-10 W and 4e-10 W of one line: 5e-10 W in all, but together in phase (1e-5 + 2e-5)^2 = 9e-10 W --
    the amplitude a field of both can reach."""
    line = ("S", 0)
    trace = TraceOutput(inputs=[SourceInput("S", "S", 0, LAM, 1.0)], left_out_W={line: 5e-10},
                        left_out_root={line: math.sqrt(1e-10) + math.sqrt(4e-10)})
    reach = surface_reach(trace, Probes(), NumericalConfig())
    assert reach.left_out_W[line] == pytest.approx(5e-10, rel=1e-15)
    assert reach.left_out_root[line] ** 2 == pytest.approx(9e-10, rel=1e-12)


def _two_mirrors(r: float):
    """Two plane mirrors of reflectance ``r`` (both sides, lossless: T = 1 - R) facing each other 20 mm apart, a source in
    front: the screen's factors of the mirrors' sides."""
    def mirror(name: str, x: float):
        component = B.mirror(name, position=(x, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=12.7)
        s = component.surfaces[0]
        c = ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=1.0 - r, transmittance_p=1.0 - r)
        return component.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(
            update={"coating": c, "reflective_side": "both"})}),)})

    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    project = ProjectSnapshot(components=(src, mirror("M0", 100.0), mirror("M1", 120.0)))
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    return resonance.return_gains(scene, library, [LAM])[LAM]


def _cavity_responses(r: float, phase: float) -> np.ndarray:
    """The 1-D two-mirror cavity (amplitudes r from the left, -r from the right, t = sqrt(1 - r^2)) at one round-trip
    phase: the field of the wave arriving at M1 from inside per unit wave injected at each of the four places (left of M0
    toward it, inside toward M1, inside toward M0, right of M1 toward it) -- a row of (I - A)^-1."""
    t = math.sqrt(1.0 - r * r)
    e = cmath.exp(1j * phase)
    # waves: 0 = right-going inside (arrives at M1), 1 = left-going inside (arrives at M0)
    a = np.zeros((2, 2), dtype=complex)
    a[0, 1] = e * r      # left-going inside, reflected at M0 (+r from its inside), travels to M1
    a[1, 0] = e * -r     # right-going inside, reflected at M1 (-r from its inside), travels to M0
    m = np.linalg.inv(np.eye(2) - a)
    inject = np.array([[e * t, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, e * t]], dtype=complex)  # (into waves 0 / 1)
    return np.abs(np.array([m[0] @ col for col in inject]))


@pytest.mark.parametrize("r", [math.sqrt(0.9), math.sqrt(0.5)])
def test_pieces_fed_at_several_places_stay_within_the_sum_of_their_bounds(r: float):
    """The 1-D check: the field arriving at the far mirror from inside, from pieces of
    unit total power injected at four places, at every round-trip phase. One piece at its worst place stays within the
    screen's factor of that side (the one-place worst case, the largest response); pieces at several places reach the
    responses' Euclidean norm (1.45 times for R = 0.9) -- within the factor times the sum of the pieces' amplitudes, which
    is what the bound uses."""
    gains = _two_mirrors(r * r)
    factor = gains.sides("M1/face")[0]  # (M1's positive side faces the source: the inside of the cavity)
    worst_one = worst_all = 0.0
    for phase in np.linspace(0.0, 2.0 * math.pi, 257):
        responses = _cavity_responses(r, phase)
        worst_one = max(worst_one, float(np.max(responses)))
        worst_all = max(worst_all, float(np.linalg.norm(responses)))
        # pieces with powers matched to the responses (the worst split of unit power): the in-phase sum of their fields
        powers = responses ** 2 / float(np.sum(responses ** 2))
        assert float(np.sum(responses * np.sqrt(powers))) <= factor * float(np.sum(np.sqrt(powers))) * (1.0 + 1e-9)
    assert worst_one <= factor * (1.0 + 1e-9)
    if r * r > 0.8:
        assert worst_all > 1.3 * worst_one  # (the square root of the total power would not hold them)


def test_the_bounds_use_the_in_phase_power_of_the_pieces():
    """The review's rated interferometer leaves its light out in two pieces: the left-out field on the window's face is
    the diffraction-limit amplitude of their in-phase power (2 pi n^2 P / lambda^2, n the glass's) times the most their
    open paths' crossings bring (``open_paths``: the window's two bare faces, and the two splitters and the two uncoated
    fold mirrors, met once per leg) and the face's return factor -- ``faint_light_peak_W_per_cm2`` is its square."""
    result = run(_rated_mzi(1e9))
    (line,) = result.exposures.faint_light
    assert line.in_phase_power_W >= line.power_W * (1.0 - 1e-12)
    front = next(s for s in result.exposures.surfaces if s.surface_id == "RATED/front")
    from test_power_exposure_ratings import _n

    n = _n("N-BK7", 532.0)
    lam_cm = 532.0e-7
    spectral = 1.0 + (532.0e-6) ** 2 / (2.0 * math.pi ** 2 * 10000.0 ** 2)  # (the source's coherence length, 10 m)
    limit = 2.0 * math.pi * n * n * line.in_phase_power_W * spectral / (lam_cm * lam_cm)
    from test_power_exposure_ratings import open_paths

    amplitude = open_paths(1.0, 1.0, per_leg=(0.5, 1.0, 1.0, 0.5)) * math.sqrt(limit) * front.faint_light_return_factor
    assert front.faint_light_peak_W_per_cm2 == pytest.approx(amplitude * amplitude, rel=1e-9)


# --------------------------------------------------------------------------------------------- a pair's light on its plane


def _pre_targets(*targets: Target):
    project = _rated_mzi(1e9)
    rating = Target(target_id="rating", device_id="RATED", metric="power_rating_utilization", maximum=1.0)
    return project.model_copy(update={"targets": (rating, *targets)})


def _pair_on_pre(result):
    (pair,) = [p for p in result.interference if p.observation_id == "pre"]
    return pair


def test_a_pair_s_irradiance_counts_the_light_the_trace_does_not_follow():
    """On the plane 0.5 um before the rated window, a peak or mean irradiance
    maximum 1.0001 times the pair's value, and a local-contrast minimum 0.9999 times it, are not decided -- light of the
    pair's wavelength the trace does not follow (the window's own ghosts, the left-out light, which the window's faces
    may return and send out across the plane) may reach the plane with a field of at most F there: the irradiance lies in
    [(sqrt v - F)^2, (sqrt v + F)^2]. Maxima above (sqrt v + F)^2 PASS."""
    probe = run(_pre_targets())
    pair = _pair_on_pre(probe)
    peak, mean, contrast = pair.peak_irradiance_W_per_mm2, pair.mean_irradiance_W_per_mm2, pair.fringe_visibility_peak
    tight = run(_pre_targets(
        Target(target_id="peak", observation_id="pre", metric="peak_irradiance_W_per_mm2", maximum=1.0001 * peak),
        Target(target_id="mean", observation_id="pre", metric="mean_irradiance_W_per_mm2", maximum=1.0001 * mean),
        Target(target_id="contrast", observation_id="pre", metric="fringe_visibility_peak", minimum=0.9999 * contrast)))
    by_id = {t.target_id: t for t in tight.targets}
    for target_id in ("peak", "mean", "contrast"):
        target = by_id[target_id]
        assert target.status.value == "UNKNOWN", target_id
        assert "FAINT_LIGHT_BOUNDED" in target.issue_codes and target.value is not None
    # the range: from the message's interval, (sqrt v + F)^2 for the peak
    # (F^2 is about 2e5 x v here: the left-out light's pieces of each reflection count, from different sequences of the
    # window's faces, the mirrors and the splitters, add in phase)
    loose = run(_pre_targets(
        Target(target_id="peak", observation_id="pre", metric="peak_irradiance_W_per_mm2", maximum=1e6 * peak),
        Target(target_id="mean", observation_id="pre", metric="mean_irradiance_W_per_mm2", maximum=1e6 * mean)))
    assert all(t.status.value == "PASS" for t in loose.targets if t.target_id != "rating")


def test_a_pair_irradiance_target_follows_the_untraced_light_without_ratings():
    """Without power ratings a peak-irradiance target alone makes the run follow the light the trace does not follow
    (as a power maximum does): the same tight maximum is not decided."""
    project = _rated_mzi(1e9)
    window = next(c for c in project.components if c.instance_id == "RATED")
    unrated = window.model_copy(update={"power_ratings": ()})
    project = project.model_copy(update={"components": tuple(unrated if c.instance_id == "RATED" else c for c in project.components)})
    probe = run(project.model_copy(update={"targets": ()}))
    peak = _pair_on_pre(probe).peak_irradiance_W_per_mm2
    target = Target(target_id="peak", observation_id="pre", metric="peak_irradiance_W_per_mm2", maximum=1.0001 * peak)
    (outcome,) = run(project.model_copy(update={"targets": (target,)})).targets
    assert outcome.status.value == "UNKNOWN" and outcome.value == pytest.approx(peak, rel=1e-12)


def _visibility_range(visibility: float, power_W: float, root: float) -> tuple[float, float]:
    """The engine's range of the two beams' visibility ``visibility`` (beams of ``power_W`` together on the plane) with
    light the trace does not follow of summed field amplitude ``root`` (sqrt(W), all of it left out as negligible); the
    value itself where no such light reaches the plane (``_visibility_light`` returns ``None``: the value is exact)."""
    pair = InterferenceResult.model_construct(observation_id="pre", wavelength_nm=LAM, status=Status.PASS, issue_codes=(),
                                              incoherent_power_sum_W=power_W, fringe_visibility=visibility)
    fate = LineFate(input_W=1.0, unknown_W={}, below_floor_W=0.0, floor_significant=False, root_sum=0.0, faint_root=root)
    got = _visibility_light(pair, visibility, "1", Status.PASS, fate)
    if got is None:
        assert root == 0.0
        return visibility, visibility
    assert got.untraced is not None
    return got.untraced


@pytest.mark.parametrize("rho", [0.0, 0.01, 0.2, 0.7, 1.5])
def test_the_visibility_over_the_plane_holds_any_field_of_the_untraced_light(rho: float):
    """Two beams' fields on a plane (two polarization components on a grid; partly overlapping, partly polarized) and a
    third field of the same wavelength of any shape, phase and polarization with ``||E3|| = rho sqrt(P_a + P_b)``: the
    fringe's contrast over the plane as beam a's phase varies -- amplitude ``2 sum |E_a . (E_b + E3)*|`` over mean
    ``sum |E_a|^2 + |E_b + E3|^2`` -- stays within the engine's range for the two beams' visibility ``2 sum |E_a . E_b*| /
    (P_a + P_b)`` -- fields against beam b and along beam a among them."""
    rng = np.random.default_rng(7)
    x = np.linspace(-3.0, 3.0, 41)
    xx, yy = np.meshgrid(x, x)
    env_a = np.exp(-(xx ** 2 + yy ** 2))
    env_b = 0.7 * np.exp(-((xx - 0.6) ** 2 + yy ** 2) / 1.4)
    e_a = np.stack([env_a, 0.2j * env_a], axis=-1)
    e_b = np.stack([env_b * np.exp(1j * 2.0 * xx), 0.5 * env_b], axis=-1)
    p = float(np.sum(np.abs(e_a) ** 2) + np.sum(np.abs(e_b) ** 2))
    two = 2.0 * float(np.sum(np.abs(np.sum(e_a * np.conj(e_b), axis=-1)))) / p
    low, high = _visibility_range(two, p, rho * math.sqrt(p))
    assert 0.0 <= low <= two <= high <= 1.0
    if rho == 0.0:
        assert low == pytest.approx(two, rel=1e-12) and high == pytest.approx(two, rel=1e-12)
    for trial in range(200):
        if trial == 0:  # against beam b: lowers the fringe and the mean
            e_3 = -e_b
        elif trial == 1:  # along beam a, in phase with b: raises both
            e_3 = e_a * np.exp(1j * np.angle(np.sum(e_a * np.conj(e_b))))
        else:
            e_3 = rng.normal(size=e_a.shape) + 1j * rng.normal(size=e_a.shape)
            e_3 = e_3 * (env_a[..., None] if trial % 2 else 1.0)
        e_3 = e_3 * (rho * math.sqrt(p) / math.sqrt(float(np.sum(np.abs(e_3) ** 2))))
        both = e_b + e_3
        amplitude = 2.0 * float(np.sum(np.abs(np.sum(e_a * np.conj(both), axis=-1))))
        mean = float(np.sum(np.abs(e_a) ** 2) + np.sum(np.abs(both) ** 2))
        contrast = amplitude / mean
        assert low - 1e-12 <= contrast <= high + 1e-12, (trial, contrast, low, high)


def test_a_visibility_target_counts_the_light_the_trace_does_not_follow():
    """A ``fringe_visibility`` minimum 0.9999 times the pair's value on the
    plane before the rated window is not decided -- the window's ghosts and the left-out light reach the plane --, a
    minimum of 0 PASSes; without power ratings the target alone makes the run follow that light."""
    pair = _pair_on_pre(run(_pre_targets()))
    vis = pair.fringe_visibility
    tight = run(_pre_targets(Target(target_id="vis", observation_id="pre", metric="fringe_visibility", minimum=0.9999 * vis),
                             Target(target_id="any", observation_id="pre", metric="fringe_visibility", minimum=0.0)))
    by_id = {t.target_id: t for t in tight.targets}
    assert by_id["vis"].status.value == "UNKNOWN" and by_id["vis"].value == pytest.approx(vis, rel=1e-12)
    assert "FAINT_LIGHT_BOUNDED" in by_id["vis"].issue_codes
    assert by_id["any"].status.value == "PASS"
    project = _rated_mzi(1e9)
    window = next(c for c in project.components if c.instance_id == "RATED")
    unrated = window.model_copy(update={"power_ratings": ()})
    project = project.model_copy(update={"components": tuple(unrated if c.instance_id == "RATED" else c for c in project.components)})
    target = Target(target_id="vis", observation_id="pre", metric="fringe_visibility", minimum=0.9999 * vis)
    (outcome,) = run(project.model_copy(update={"targets": (target,)})).targets
    assert outcome.status.value == "UNKNOWN"
