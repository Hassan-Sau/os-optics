"""How often the light the trace leaves out as negligible may cross a plane or meet a surface between the returns of
surfaces facing each other (``validation/resonance.py``, ``ReturnGains.crossings`` / ``crossing_field``).

On an open path no surface reflects the light twice, and between two reflections it meets each surface that turns it -- a
lens, a refracting surface between media of different index -- at most once (light turned back onto one it already passed
before it is reflected again is outside the model). Between two reflections it travels in straight legs, a new one only
where such a surface turns it: at most ``L = T + 1`` legs (``T`` those surfaces), each meeting any other surface at most
``K`` times, as often as a straight line can (a plane once, a sphere or a conic twice, an asphere by the degree of its
sag). So one piece of left-out light meets a lens or a refracting surface between different media at most once after each
number ``j`` of reflections, ``j`` up to ``N``, the number of surfaces that may reflect it, and any other surface or an
observation plane at most ``K L`` times (``m``). Each reflection takes at most its surface's reflectance bound ``R_s`` of each meeting,
so after ``j`` reflections the light carries at most ``min(1, R^j)`` of its power, ``R = sum m_s R_s``: its crossings
carry at most ``m sum_{j=0}^{N} min(1, R^j)`` times its power. Its field at a point, the light split between the two
sides each time, is at most ``sqrt(2) m sum_{j=0}^{N} a_j`` times one crossing's: after ``j`` reflections its pieces from
different sequences of ``j`` distinct reflecting surfaces are mutually coherent, their fields add, so ``a_j`` is the
smaller of the ordered sum of the products of ``m_s sqrt(R_s)`` and the square root of their number times the power they
share. Both at least 2 (the piece once from either side). Hand values are written out.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import EngineeringComponentSnapshot, ProjectSnapshot, Target
from os_optics.contracts.components import PowerRating
from os_optics.contracts.materials import ConstantCoating
from os_optics.contracts.optics import (BeamSplitterInteraction, CircularAperture, MirrorInteraction, PlaneSurface,
                                             RefractInteraction, SagSurface, TabulatedThinLensInteraction)
from os_optics.contracts.request import EngineConfiguration
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.surfaces.sag import sag, transverse_s
from os_optics.scene.lower import lower_scene
from os_optics.scene.model import SurfaceG
from os_optics.validation import resonance
from os_optics.validation.pipeline import run_physics
from os_optics.validation.untraced import FAINT_POWER, FAINT_ROOT

from helpers import run
from test_power_exposure_ratings import open_paths, open_power

LAM = 1064.0
LEAK = 5e-10  # (a reflection or transmission of the 1 W beam at or below the 1e-9 x input level: left out)
R_FOLD = 0.999
OBS = "observation/OBS"


def _coat(r: float) -> ConstantCoating:
    return ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=1.0 - r, transmittance_p=1.0 - r)


def _gains(project: ProjectSnapshot) -> resonance.ReturnGains:
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    return resonance.return_gains(scene, library, [LAM])[LAM]


def _status(project: ProjectSnapshot) -> list[str]:
    return [t.status.value for t in run(project).targets]


def _open_path(maximum: float) -> ProjectSnapshot:
    """A 1 W, 1064 nm beam along +z onto the fold mirror M0, which sends it to -x
    and transmits ``LEAK`` W on along +z (left out); F1 at (0, 0, 20) folds +z into +x, F2 at (20, 0, 20) folds +x into
    (1, 0, 1) / sqrt(2) (both R 0.999; their normals are 22.5 deg from antiparallel: they do not face each other). The
    transparent plane OBS, -x / 2 + z = 15 (radius 25 mm about (12, 0, 21)), is crossed by that path three times: at (0,
    0, 15) going +, at (10, 0, 20) going -, at (30, 0, 30) going + -- no surface met twice, no pair. Power and coherent
    power maxima ``maximum`` on it."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    m0 = B.mirror_for_reflection("M0", position=(0.0, 0.0, 0.0), d_in=(0.0, 0.0, 1.0), d_out=(-1.0, 0.0, 0.0), clear_radius_mm=12.7)
    face = m0.surfaces[0]
    coating = ConstantCoating(reflectance_s=1.0 - LEAK, reflectance_p=1.0 - LEAK, transmittance_s=LEAK, transmittance_p=LEAK)
    m0 = m0.model_copy(update={"surfaces": (face.model_copy(update={"interaction": face.interaction.model_copy(update={"coating": coating})}),)})
    f1 = B.mirror_for_reflection("F1", position=(0.0, 0.0, 20.0), d_in=(0.0, 0.0, 1.0), d_out=(1.0, 0.0, 0.0), clear_radius_mm=5.0,
                                 reflectance=R_FOLD)
    f2 = B.mirror_for_reflection("F2", position=(20.0, 0.0, 20.0), d_in=(1.0, 0.0, 0.0), d_out=B.unit((1.0, 0.0, 1.0)),
                                 clear_radius_mm=5.0, reflectance=R_FOLD)
    s5 = math.sqrt(5.0)
    obs = B.observation_plane("OBS", position=(12.0, 0.0, 21.0), normal=(-1.0 / s5, 0.0, 2.0 / s5), radius_mm=25.0)
    targets = (Target(target_id="p", observation_id="OBS", metric="power_W", maximum=maximum),
               Target(target_id="c", observation_id="OBS", metric="coherent_power_W", maximum=maximum))
    return ProjectSnapshot(components=(src, m0, f1, f2), observations=(obs,), targets=targets)


def test_left_out_light_crossing_a_plane_three_times_on_an_open_path_is_bounded_by_its_crossings():
    """An earlier version bounded the left-out light on a transparent plane by twice its power (once in either direction) and its
    field by twice its amplitude: the plane PASSed a power maximum of 2.6 x the leak while the leak's three crossings carry
    (1 + R + R^2) = 2.997 x it. The three plane mirrors may each reflect it once (R summing to more than 1) and nothing
    turns it between two reflections (one leg): its crossings carry at most 4 x its power, its field at most sqrt(2) (1 +
    sqrt(3) + 2 sqrt(6)) x one crossing's -- the pieces of each reflection count, from different sequences of the three
    mirrors, add in phase: at most 3, 6 and 6 pieces sharing at most the whole power after one, two and three
    reflections -- above the three crossings in phase, (1 + sqrt(R) + R)^2 = 9.0 x."""
    gains = _gains(_open_path(1.0))
    assert gains.pairs == () and gains.reflectors == 3 and gains.legs == 1 and OBS not in gains.once
    assert gains.reflectance == pytest.approx(1.0 - LEAK + 2.0 * R_FOLD, rel=1e-12)
    mirrors = (1.0 - LEAK, R_FOLD, R_FOLD)
    power_crossings = open_power(per_leg=mirrors, plane=True)
    field_crossings = open_paths(per_leg=mirrors, plane=True)
    assert (gains.crossings(OBS), gains.crossing_field(OBS)) == pytest.approx((power_crossings, field_crossings), rel=1e-12)
    assert power_crossings == 4.0
    assert field_crossings == pytest.approx(math.sqrt(2.0) * (1.0 + math.sqrt(3.0) + 2.0 * math.sqrt(6.0)), rel=1e-12)
    outcome = run_physics(_open_path(1.0), EngineConfiguration().numerical)
    reach = outcome.reach["OBS"][("S", 0)]
    assert reach[FAINT_POWER] == pytest.approx(power_crossings * LEAK, rel=1e-6)
    assert reach[FAINT_ROOT] == pytest.approx(field_crossings * math.sqrt(LEAK), rel=1e-6)
    hand_power = (1.0 + R_FOLD + R_FOLD * R_FOLD) * LEAK
    hand_field = (1.0 + math.sqrt(R_FOLD) + R_FOLD) ** 2 * LEAK
    assert reach[FAINT_POWER] >= hand_power and reach[FAINT_ROOT] ** 2 >= hand_field
    for maximum in (2.6 * LEAK, 3.9 * LEAK):  # (the verifier's 2.6 x PASSed)
        assert _status(_open_path(maximum)) == ["UNKNOWN", "UNKNOWN"]
    assert _status(_open_path(4.01 * LEAK)) == ["PASS", "UNKNOWN"]
    assert _status(_open_path(32.1 * LEAK)) == ["PASS", "UNKNOWN"]  # (the field's sequences in phase: 116.5 x at most)
    assert _status(_open_path(1.01 * field_crossings ** 2 * LEAK)) == ["PASS", "PASS"]


# ------------------------------------------------------------------ refraction or a lens turning the light back


N_BK7 = MaterialLibrary([], NumericalConfig()).evaluator("N-BK7").index(LAM)[0]
AR = _coat(0.0025)
APEX = math.radians(35.0)
SLOPE = 0.25  # (the plane OBS: z = SLOPE (x - 10))


def _refract(d, normal, n1: float, n2: float):
    """Snell's law for the unit direction ``d`` through a face of unit normal ``normal``, from index ``n1`` into ``n2``."""
    c = sum(a * b for a, b in zip(d, normal))
    if c < 0.0:
        normal, c = tuple(-x for x in normal), -c
    r = n1 / n2
    k = 1.0 - r * r * (1.0 - c * c)
    return tuple(r * a + (math.sqrt(k) - r * c) * b for a, b in zip(d, normal))


def _prism(iid: str, start, d, sense: int):
    """A 2 mm N-BK7 prism (faces coated R 0.25 %): its front face normal to ``d`` at ``start``, its back face 2 mm on with
    its outward normal turned by ``-sense`` x 35 deg from ``d`` (in x-z). Returns the prism, its back face's point on the
    ray and the direction the ray leaves in (Snell's law)."""
    phi = math.atan2(d[2], d[0])
    back = tuple(s + 2.0 * x for s, x in zip(start, d))
    normal = (math.cos(phi - sense * APEX), 0.0, math.sin(phi - sense * APEX))
    out = _refract(d, normal, N_BK7, 1.0)
    surfaces = tuple(PlaneSurface(surface_id=sid, origin_mm=o, normal=nrm, aperture=CircularAperture(radius_mm=3.0),
                                  mount_margin_mm=0.5, medium_positive="air", medium_negative="N-BK7",
                                  interaction=RefractInteraction(coating=AR))
                     for sid, o, nrm in (("front", start, tuple(-x for x in d)), ("back", back, normal)))
    element = EngineeringComponentSnapshot(instance_id=iid, role="window", surfaces=surfaces,
                                           pose=B._pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    return element, back, out


def _on_obs(p, d):
    """Where the ray ``p + t d`` crosses the plane OBS."""
    t = (SLOPE * (p[0] - 10.0) - p[2]) / (d[2] - SLOPE * d[0])
    return tuple(a + t * b for a, b in zip(p, d))


def _ghost_window(r0: float, traced: bool):
    """A 1 mm N-BK7 window at the origin (front face normal (1, 0, -1) / sqrt(2)): its front face reflects ``r0`` of the
    1 W beam along +z toward +x (``r0`` = ``LEAK``: left out; a traced control otherwise), its back face nothing."""
    axis = B.unit((-1.0, 0.0, 1.0))
    window = B.window("W", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=tuple(0.5 * a for a in axis), axis=axis)
    front, back = window.surfaces
    return window.model_copy(update={"surfaces": (
        front.model_copy(update={"interaction": front.interaction.model_copy(update={"coating": _coat(r0), "trace_fresnel_reflection": traced})}),
        back.model_copy(update={"interaction": back.interaction.model_copy(update={"coating": _coat(0.0)})}))})


def _prisms(count: int):
    """The window's ghost leaves along +x and crosses OBS at (10, 0, 0); P1 at x = 40
    turns it up by asin(n sin 35 deg) - 35 deg = 24.8 deg across OBS again at (79.8, 0, 17.4); with two prisms P2 on the
    light at x = 110, above the plane, turns it back to +x across OBS a third time at (139.0, 0, 32.2). No surface reflects
    it on the way (the prisms' faces only refract the traced control); the prisms' own faces face each other."""
    p1, b1, d1 = _prism("P1", (40.0, 0.0, 0.0), (1.0, 0.0, 0.0), +1)
    crossings, elements = [_on_obs((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), _on_obs(b1, d1)], [p1]
    if count == 2:
        start = tuple(a + (110.0 - b1[0]) / d1[0] * b for a, b in zip(b1, d1))
        p2, b2, d2 = _prism("P2", start, d1, -1)
        elements.append(p2)
        crossings.append(_on_obs(b2, d2))
    return elements, crossings


def _prism_project(count: int, r0: float, maximum: float, traced: bool = False) -> ProjectSnapshot:
    elements, crossings = _prisms(count)
    first, last = crossings[0], crossings[-1]
    centre = tuple(0.5 * (a + b) for a, b in zip(first, last))
    obs = B.observation_plane("OBS", position=centre, normal=B.unit((-SLOPE, 0.0, 1.0)),
                              radius_mm=0.5 * math.dist(first, last) + 5.0)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    targets = (Target(target_id="p", observation_id="OBS", metric="power_W", maximum=maximum),)
    return ProjectSnapshot(components=(src, _ghost_window(r0, traced), *elements), observations=(obs,), targets=targets)


@pytest.mark.parametrize("count", [1, 2])
def test_left_out_light_a_prism_turns_back_across_a_plane_counts_every_straight_leg(count: int):
    """An earlier version counted one crossing of a plane after each number of reflections: a window's ghost (5e-10 W, left out) that
    one prism turns back across the plane crosses it twice (1.995 x the ghost: the traced control at 1e-2), and the plane
    PASSed power maxima of 1.2 - 1.9 x the ghost (with two prisms, three crossings, 2.985 x: up to 2.9 x). Between two
    reflections the ghost now travels in at most L = T + 1 straight legs -- T the window's two faces and the prisms' faces,
    the surfaces that may turn it --, each crossing the plane at most once: its crossings carry at most L sum_{j=0}^{N}
    R^j times its power (N the window's front face and the prisms' faces, R their reflectance bounds: 5e-10 + 0.25 % per
    face, each met at most once between two reflections), times the square of the factor of the prisms' returns."""
    elements, crossings = _prisms(count)
    assert len(crossings) == count + 1
    control = run(_prism_project(count, 1e-2, 1.0, traced=True))
    (obs,) = control.observations
    assert len(obs.arrivals) == count + 1
    carried = sum(a.beam.power_W for a in obs.arrivals) / 1e-2
    assert carried == pytest.approx((1.995, 2.985)[count - 1], abs=1e-3)
    gains = _gains(_prism_project(count, LEAK, 1.0))
    faces = 2 * count
    assert (gains.reflectors, gains.legs) == (1 + faces, 3 + faces) and OBS not in gains.once
    reflectance = LEAK + faces * 0.0025
    assert gains.reflectance == pytest.approx(reflectance, rel=1e-12)
    power = (3 + faces) * sum(reflectance ** j for j in range(2 + faces))
    assert gains.crossings(OBS) == pytest.approx(power, rel=1e-12)
    assert gains.crossings(OBS) == pytest.approx(open_power(LEAK, *[0.0025] * faces, turning=2 + faces, plane=True), rel=1e-12)
    assert gains.crossings("P1/front") == pytest.approx(2.0, rel=1e-15)  # (met at most once: the least count)
    factor = gains.factor(OBS)
    assert len(gains.pairs) == (1, 6)[count - 1] and all(p.gain == pytest.approx(0.0025, rel=1e-9) for p in gains.pairs)
    assert factor == pytest.approx(1.0 + sum(p.returns for p in gains.pairs), rel=1e-12)
    outcome = run_physics(_prism_project(count, LEAK, 1.0), EngineConfiguration().numerical)
    bound = outcome.reach["OBS"][("S", 0)][FAINT_POWER]
    assert bound == pytest.approx(power * factor * factor * LEAK, rel=1e-6)
    assert bound >= carried * LEAK
    for k in (1.2, 1.9, 2.9, 0.99 * bound / LEAK):  # (formerly PASS up to 1.9 x, two prisms up to 2.9 x)
        assert _status(_prism_project(count, LEAK, k * LEAK)) == ["UNKNOWN"]
    assert _status(_prism_project(count, LEAK, 1.01 * bound)) == ["PASS"]


def _lens_project(r0: float, maximum: float, traced: bool = False) -> ProjectSnapshot:
    """The ghost window at the origin sends ``r0`` of the 1 W beam along +x;
    a thin lens (f 3 mm, radius 6 mm, axis +x through (30, 0, 3)) meets it 3 mm below its axis and turns it to (1, 0, 1)
    / sqrt(2). The transparent plane OBS, -x + 2 z = -10 (radius 25 mm about (30, 0, 10)), is crossed at (10, 0, 0) going
    one way and at (50, 0, 20) the other way; the main beam meets its plane at (0, 0, -5), outside it, and OBS does not
    meet the lens."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    lens = B.thin_lens("L", focal_length_mm=3.0, clear_radius_mm=6.0, position=(30.0, 0.0, 3.0), axis=(1.0, 0.0, 0.0))
    s5 = math.sqrt(5.0)
    obs = B.observation_plane("OBS", position=(30.0, 0.0, 10.0), normal=(-1.0 / s5, 0.0, 2.0 / s5), radius_mm=25.0)
    targets = (Target(target_id="p", observation_id="OBS", metric="power_W", maximum=maximum),
               Target(target_id="c", observation_id="OBS", metric="coherent_power_W", maximum=maximum))
    return ProjectSnapshot(components=(src, _ghost_window(r0, traced), lens), observations=(obs,), targets=targets)


def test_left_out_light_a_lens_turns_back_across_a_plane_counts_every_straight_leg():
    """An earlier version counted one crossing of the plane: the window's ghost (5e-10 W), which a thin lens turns back across the
    plane, crosses it twice (2 x the ghost; in phase 4 x), and the plane PASSed a power maximum of 1.2 x the ghost and
    coherent power maxima of 2.5 and 3.5 x. The window's two faces and the lens may turn it (T = 3): four straight legs between two
    reflections, the window's front face its only reflector (N 1, R 5e-10, no pair): the crossings carry at most 4 (1 +
    5e-10) x its power, the field 4 sqrt(2) (1 + sqrt(5e-10)) x one crossing's (32 x the ghost in phase)."""
    control = run(_lens_project(1e-2, 1.0, traced=True))
    (obs,) = control.observations
    assert [round(a.beam.power_W, 9) for a in obs.arrivals] == [0.01, 0.01]
    gains = _gains(_lens_project(LEAK, 1.0))
    assert (gains.pairs, gains.reflectors, gains.legs) == ((), 1, 4) and OBS not in gains.once and "L/lens" in gains.once
    power, field = 4.0 * (1.0 + LEAK), 4.0 * math.sqrt(2.0) * (1.0 + math.sqrt(LEAK))
    assert (gains.crossings(OBS), gains.crossing_field(OBS)) == pytest.approx((power, field), rel=1e-12)
    assert gains.crossing_field(OBS) == pytest.approx(open_paths(LEAK, turning=3, plane=True), rel=1e-12)
    outcome = run_physics(_lens_project(LEAK, 1.0), EngineConfiguration().numerical)
    reach = outcome.reach["OBS"][("S", 0)]
    assert reach[FAINT_POWER] == pytest.approx(power * LEAK, rel=1e-6)
    assert reach[FAINT_ROOT] == pytest.approx(field * math.sqrt(LEAK), rel=1e-6)
    assert reach[FAINT_POWER] >= 2.0 * LEAK and reach[FAINT_ROOT] ** 2 >= 4.0 * LEAK
    for maximum in (1.2 * LEAK, 2.5 * LEAK, 3.5 * LEAK):  # (formerly the power PASSed at all three, the coherent power at 2.5, 3.5 x)
        assert _status(_lens_project(LEAK, maximum)) == ["UNKNOWN", "UNKNOWN"]
    assert _status(_lens_project(LEAK, 4.01 * LEAK)) == ["PASS", "UNKNOWN"]
    assert _status(_lens_project(LEAK, 32.1 * LEAK)) == ["PASS", "PASS"]


# ------------------------------------------------------------------------------------------- the count written out


def test_the_open_path_crossings_count_the_reflectors_the_legs_and_the_surfaces_met_once():
    """The series written out: N = 3 surfaces whose reflectance bounds (0.25, 0.15, 0.1, met once each) sum to 0.5 give 1 +
    0.5 + 0.25 + 0.125 for the power; for the field the pieces of each reflection count add in phase, over the ordered
    sequences of distinct surfaces: after j reflections at most the sum of the products of their amplitude bounds
    sqrt(R_s), and at most sqrt(j! C(3, j) x their power, at most min(1, the sum of the products of R_s)) -- times
    sqrt(2) and the meetings ``m`` (1 on a surface met at most once, ``L`` on a plane), at least 2. Five surfaces of
    0.2 each (summing to 1): the power m (N + 1); the field sqrt(2) m (1 + sum_j min(j! C(5, j) 0.2^(j/2), sqrt(j!
    C(5, j) min(1, j! C(5, j) 0.2^j)))). In a scene a thin lens turns the light (two legs) but reflects nothing, and a beam
    dump and an observation plane neither; an uncoated window's two faces reflect 1 each (the reflectance just below the
    critical angle) and turn it; 50 faces reflecting 9.9e-10 each add 4.95e-8, met once each."""
    bounds = (0.25, 0.15, 0.1)
    weak = resonance.ReturnGains(LAM, reflectors=3, reflectance=0.5, legs=3, once=frozenset({"A"}),
                                 reflecting=tuple((math.sqrt(r), 1) for r in bounds))
    series = 1.875
    x = [math.sqrt(r) for r in bounds]
    amplitude = (x[0] + x[1] + x[2], 2.0 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2]), 6.0 * x[0] * x[1] * x[2])
    power = (0.5, 2.0 * (0.25 * 0.15 + 0.25 * 0.1 + 0.15 * 0.1), 6.0 * 0.25 * 0.15 * 0.1)
    root_series = 1.0 + sum(min(a, math.sqrt(n * min(1.0, q))) for a, n, q in zip(amplitude, (3, 6, 6), power))
    assert (weak.meetings("A"), weak.meetings("B"), weak.meetings()) == (1, 3, 3)
    assert weak.crossings("A") == 2.0 and weak.crossings("B") == pytest.approx(3.0 * series, rel=1e-15)
    assert weak.crossing_field("A") == pytest.approx(math.sqrt(2.0) * root_series, rel=1e-12)
    assert weak.crossing_field() == pytest.approx(3.0 * math.sqrt(2.0) * root_series, rel=1e-12)
    strong = resonance.ReturnGains(LAM, reflectors=5, reflectance=1.0, reflecting=((math.sqrt(0.2), 1),) * 5)
    ordered = [math.factorial(j) * math.comb(5, j) for j in range(1, 6)]
    strong_series = 1.0 + sum(min(n * 0.2 ** (j / 2.0), math.sqrt(n * min(1.0, n * 0.2 ** j)))
                              for j, n in zip(range(1, 6), ordered))
    assert strong.crossings() == 6.0 and strong.crossing_field() == pytest.approx(math.sqrt(2.0) * strong_series, rel=1e-12)
    assert resonance.ReturnGains(LAM).crossings() == 2.0 and resonance.ReturnGains(LAM).crossing_field() == 2.0
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.thin_lens("L", focal_length_mm=100.0, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0))
    dump = B.beam_dump("D", position=(200.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    plane = B.observation_plane("P", position=(100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    none = _gains(ProjectSnapshot(components=(src, lens, dump), observations=(plane,)))
    assert (none.reflectors, none.legs, none.once) == (0, 2, frozenset({"L/lens"}))
    assert (none.crossings("observation/P"), none.crossing_field("observation/P")) == (2.0, pytest.approx(2.0 * math.sqrt(2.0)))
    assert (none.crossings("L/lens"), none.crossing_field("L/lens"), none.crossing_field("D/dump")) == (
        2.0, 2.0, pytest.approx(2.0 * math.sqrt(2.0)))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 0.0, 0.0))
    bare = _gains(ProjectSnapshot(components=(src, lens, window, dump)))
    assert (bare.reflectors, bare.reflectance, bare.legs) == (2, pytest.approx(2.0, rel=1e-12), 4)
    assert bare.crossings("W/front") == 3.0 and bare.crossings() == 12.0
    assert bare.crossing_field("W/back") == pytest.approx(open_paths(1.0, 1.0), rel=1e-12)
    assert bare.crossing_field() == pytest.approx(open_paths(1.0, 1.0, turning=3, plane=True), rel=1e-12)
    r = 9.9e-10
    windows = []
    for k in range(25):
        w = B.window(f"W{k}", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=(20.0 + 5.0 * k, 0.0, 0.0))
        windows.append(w.model_copy(update={"surfaces": tuple(
            s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": _coat(r)})}) for s in w.surfaces)}))
    coated = _gains(ProjectSnapshot(components=(src, *windows)))
    assert (coated.reflectors, coated.legs) == (50, 51) and coated.reflectance == pytest.approx(50 * r, rel=1e-9)
    assert coated.crossing_field("W0/front") == 2.0 == open_paths(*[r] * 50)
    # Parallel weak interfaces admit a convergent sum over every return sequence,
    # including repeated surfaces; the distinct-surface open-path sum omits that tail.
    all_orders = 1.0 / (1.0 - 50 * math.sqrt(r))
    assert coated.closed_cycles and coated.closed_series == pytest.approx(all_orders, rel=1e-12)
    assert coated.crossing_field() == pytest.approx(51 * math.sqrt(2.0) * all_orders, rel=1e-12)
    assert coated.crossing_field() > open_paths(*[r] * 50, plane=True)


# ------------------------------------------- surfaces that do not turn the light: met as often per leg as a line meets them


def _element(iid: str, surface, rating: float | None = None) -> EngineeringComponentSnapshot:
    ratings = () if rating is None else (PowerRating(regime="CW", value=rating, unit="W/cm2", wavelength_nm=LAM,
                                                      value_definition="PEAK", status="DECLARED", basis="test fixture"),)
    return EngineeringComponentSnapshot(instance_id=iid, role="beam_splitter", surfaces=(surface,), power_ratings=ratings,
                                        pose=B._pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))


def _air_face(iid: str, r0: float, traced: bool = False, normal=(1.0, 0.0, -1.0)) -> EngineeringComponentSnapshot:
    """A refracting plane between air and air at the origin (normal ``normal``, coated ``r0``): it turns no light, and its
    reflection sends a beam along +z toward +x."""
    return _element(iid, PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=B.unit(normal),
                                      aperture=CircularAperture(radius_mm=12.7), medium_positive="air", medium_negative="air",
                                      interaction=RefractInteraction(coating=_coat(r0), trace_fresnel_reflection=traced)))


def _sag(**form) -> SurfaceG:
    """A lowered sag surface (vertex at the origin, axis +z) with the profile ``form`` (``dataclasses.replace`` fields of
    ``SurfaceG``: curvature, conic, coefficients, linear, cylindrical)."""
    surface = SagSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0),
                         curvature_per_mm=0.0, aperture=CircularAperture(radius_mm=10.0),
                         interaction=BeamSplitterInteraction(coating=_coat(1e-3)))
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    project = ProjectSnapshot(components=(src, _element("CS", surface)))
    cfg = NumericalConfig()
    scene, _ = lower_scene(project, cfg, MaterialLibrary(project.materials, cfg))
    (lowered,) = [s for s in scene.surfaces if s.surface_id == "CS/face"]
    return replace(lowered, **form)


def _points_on(surface: SurfaceG, start, direction, extent: float = 60.0) -> int:
    """The points the line ``start + t direction`` (the surface's local frame: x, y across its axis, z along it) meets the
    sag surface at, counted as sign changes of ``z - sag`` within the profile's domain (a lower count where it touches)."""
    t = np.linspace(-extent, extent, 120001)
    a, b, z = (start[k] + t * direction[k] for k in range(3))
    profile, inside = sag(surface, transverse_s(surface, a, b))
    f = z - profile
    both = inside[:-1] & inside[1:]
    return int(np.count_nonzero(both & (np.sign(f[:-1]) * np.sign(f[1:]) < 0.0)))


#: Sag profiles and the most points a straight line meets each at (``resonance._line_meetings``): the degree of the profile's
#: equation along a line -- e = 2p for asphere terms up to rho^(2p) (1 without), 2e on a curved conic base, max(e, 2) on a
#: paraboloid, e on a flat base; twice that with an axicon term.
SAG_FORMS = [
    (dict(), 1),  # (a flat sag: a plane)
    (dict(curvature=0.05), 2),  # (a sphere, R 20 mm)
    (dict(curvature=0.05, conic=-1.0), 2),  # (a paraboloid)
    (dict(curvature=0.05, conic=-2.0), 2),  # (a hyperboloid)
    (dict(curvature=0.05, conic=0.5), 2),  # (an oblate ellipsoid)
    (dict(curvature=0.05, cylindrical=True), 2),  # (a circular cylinder)
    (dict(curvature=0.05, coefficients=(0.0, 0.0)), 2),  # (zero asphere terms)
    (dict(coefficients=(1e-4,)), 4),  # (A4 on a flat base)
    (dict(curvature=0.2, conic=-1.0, coefficients=(-5e-4,)), 4),  # (A4 on a paraboloid: a W profile)
    (dict(curvature=0.05, coefficients=(1e-5,)), 8),  # (A4 on a sphere)
    (dict(curvature=0.05, coefficients=(1e-5, -1e-8)), 12),  # (A4 and A6 on a sphere)
    (dict(curvature=0.05, coefficients=(0.0, -1e-8)), 12),  # (A6 alone on a sphere)
    (dict(curvature=0.02, cylindrical=True, coefficients=(-2e-5,)), 8),  # (an acylinder)
    (dict(linear=0.3), 2),  # (a bare cone)
    (dict(curvature=0.1, conic=-1.0, linear=-0.5), 4),  # (a cone on a paraboloid)
    (dict(curvature=0.05, linear=0.3), 4),  # (a cone on a sphere)
    (dict(curvature=0.05, coefficients=(1e-5,), linear=0.3), 16),  # (a cone on an asphere)
]


@pytest.mark.parametrize(("form", "most"), SAG_FORMS)
def test_a_straight_line_meets_a_sag_surface_at_most_as_often_as_its_profile_allows(form: dict, most: int):
    """The meetings per straight leg of a surface that does not turn the light (module docstring of
    ``validation/resonance.py``, "Meetings per leg"): no line of 400 random ones through the surface's region meets it more
    often than its profile's degree along a line allows."""
    surface = _sag(**form)
    assert resonance._line_meetings(surface) == most
    rng = np.random.default_rng(20261017)
    found = 0
    for _ in range(400):
        start = rng.uniform(-15.0, 15.0, 3)
        direction = rng.normal(size=3)
        found = max(found, _points_on(surface, start, direction / np.linalg.norm(direction)))
    assert found <= most


def test_lines_reach_the_most_meetings_of_a_sphere_a_w_profile_and_a_cone():
    """The counts are reached: a chord of a sphere meets it twice, a line across a W profile (a paraboloid c = 0.2 / mm
    with A4 = -5e-4 / mm^3, z = 0.1 rho^2 - 5e-4 rho^4) four times, a line across a bare cone twice and one across a cone
    on a paraboloid (z = 0.05 rho^2 - 0.5 rho) four times."""
    along_x = (1.0, 0.0, 0.0)
    assert _points_on(_sag(curvature=0.05), (0.0, 0.0, 10.0), along_x) == 2 == resonance._line_meetings(_sag(curvature=0.05))
    w = _sag(curvature=0.2, conic=-1.0, coefficients=(-5e-4,))
    assert _points_on(w, (0.0, 0.0, 2.5), along_x) == 4 == resonance._line_meetings(w)
    assert _points_on(_sag(linear=0.3), (0.0, 0.0, 2.0), along_x) == 2
    cone = _sag(curvature=0.1, conic=-1.0, linear=-0.5)
    assert _points_on(cone, (0.0, 0.0, -0.6), along_x) == 4 == resonance._line_meetings(cone)


T_TL = 0.999  # (a tabulated thin lens's response: 1e-3 of the 1 W beam of unknown fate)


def _unknown_fate_lens() -> EngineeringComponentSnapshot:
    """A tabulated thin lens (f 1 km, T 0.999) on the 1 W beam along +z at (0, 0, -20): the light it does not deliver,
    1e-3 W of unknown fate whose power is known, counts with the left-out light and cannot be followed deeper, so its
    bound decides a rating on its own. It turns light: one leg more."""
    lens = B.thin_lens("TL", focal_length_mm=1e6, clear_radius_mm=5.0, position=(0.0, 0.0, -20.0), axis=(0.0, 0.0, 1.0))
    surface = lens.surfaces[0]
    return lens.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": TabulatedThinLensInteraction(
        wavelength_nm=(LAM,), focal_length_mm=(1e6,), transmittance=(T_TL,), focal_interpolation="points_only",
        transmittance_interpolation="points_only")}),)})


def _one_leg_project(form: str, r0: float, rating: float, traced: bool = False, unknown: bool = False) -> ProjectSnapshot:
    """A refracting plane between air and air at the origin reflects ``r0`` of a 1 W, 1064 nm beam along +z toward +x
    (5e-10: left out; a traced control otherwise); nothing else turns light (``unknown``: the lens ``TL`` on the beam
    before it), so between two reflections the light travels one straight leg, along the x axis, across a curved splitter
    CS (R 1e-3, a CW rating): ``sphere`` (R 20 mm, vertex 10 mm below the leg: a chord, met twice) or ``w`` (the W
    profile, vertex 2.5 mm below: met four times)."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    if form == "sphere":
        surface = SagSurface(surface_id="face", origin_mm=(50.0, 0.0, -10.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0),
                             curvature_per_mm=0.05, aperture=CircularAperture(radius_mm=19.0),
                             interaction=BeamSplitterInteraction(coating=_coat(1e-3)))
    else:
        surface = SagSurface(surface_id="face", origin_mm=(50.0, 0.0, -2.5), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0),
                             curvature_per_mm=0.2, conic=-1.0, aspheric_coefficients=(-5e-4,),
                             aperture=CircularAperture(radius_mm=14.0), interaction=BeamSplitterInteraction(coating=_coat(1e-3)))
    extra = (_unknown_fate_lens(),) if unknown else ()
    return ProjectSnapshot(components=(src, *extra, _air_face("M0", r0, traced), _element("CS", surface, rating)),
                           targets=(Target(target_id="u", device_id="CS", metric="power_rating_utilization", maximum=1.0),))


def _faint_bound(result, surface_id: str) -> tuple[float, float, float]:
    """(I_max, f, faint peak) on a surface between air and air: the diffraction limit 2 pi P* / lambda^2 of the left-out
    light's in-phase power (W/cm2), the surface's return factor and its ``faint_light_peak_W_per_cm2``."""
    face = next(s for s in result.exposures.surfaces if s.surface_id == surface_id)
    (line,) = result.exposures.faint_light
    i_max = 2.0 * math.pi * line.in_phase_power_W / (LAM * 1e-6) ** 2 * 100.0
    return i_max, face.faint_light_return_factor or 1.0, face.faint_light_peak_W_per_cm2


def _rating_outcome(project: ProjectSnapshot, surface_id: str) -> tuple[str, float | None]:
    """(the surface's rating status, its comparison's exposure upper bound)."""
    surface = next(s for s in run(project).exposures.surfaces if s.surface_id == surface_id)
    return surface.rating_status, surface.ratings[0].exposure_upper_bound


@pytest.mark.parametrize(("form", "most"), [("sphere", 2), ("w", 4)])
def test_left_out_light_meeting_a_curved_splitter_several_times_on_one_straight_leg_counts_every_meeting(form: str, most: int):
    """An earlier version counted every curved surface as met at most once between two reflections: the W-profile splitter, which
    one straight leg crosses four times (the traced control: four hits, passes 0 - 3), had c = 2, and a rating between 4
    and 16 x the diffraction limit of the light of unknown fate -- below its four meetings in phase -- was WITHIN_RATING.
    A surface that does not turn the light is met as often per straight leg as a straight line meets it: 2 on the sphere,
    4 on the W profile (``resonance._line_meetings``), times L = 2 (the lens TL); R = 2 x 5e-10 + m 1e-3 (the air face
    met once per leg, CS m times); c = sqrt(2) m (1 + a_1 + a_2), the pieces after one and two reflections in phase:
    a_1 at most the amplitude bounds summed, 2 sqrt(5e-10) + m sqrt(1e-3), and at most sqrt((2 + m) R); a_2 at most
    2 (2 sqrt(5e-10)) (m sqrt(1e-3)) and at most sqrt(2 (2 m) min(1, 2 (2 x 5e-10) (m 1e-3))). Only light left out as
    negligible reaches CS, of unknown size and state: its rating (no test beam or polarization stated; stated, they could
    not be checked against that light either) is not verified, and the bound shows in the exposure -- above every rating
    below the meetings in phase, within 1.01 x the bound."""
    control = run(_one_leg_project(form, 1e-2, 1e9, traced=True))
    face = next(s for s in control.exposures.surfaces if s.surface_id == "CS/face")
    assert sorted(h.pass_index for h in face.hits) == list(range(most)) and len({h.path_id for h in face.hits}) == 1
    assert _gains(_one_leg_project(form, LEAK, 1e9)).per_leg == {"CS/face": most}
    gains = _gains(_one_leg_project(form, LEAK, 1e9, unknown=True))
    assert (gains.reflectors, gains.legs, gains.once, gains.per_leg) == (2, 2, frozenset({"TL/lens"}), {"CS/face": most})
    m = 2 * most
    assert gains.meetings("CS/face") == m and gains.meetings("M0/face") == 2 and gains.meetings("TL/lens") == 1
    reflectance = 2.0 * LEAK + m * 1e-3
    assert gains.reflectance == pytest.approx(reflectance, rel=1e-12)
    x_air, x_cs = 2.0 * math.sqrt(LEAK), m * math.sqrt(1e-3)
    a_1 = min(x_air + x_cs, math.sqrt((2 + m) * min(1.0, reflectance)))
    a_2 = min(2.0 * x_air * x_cs, math.sqrt(2.0 * 2 * m * min(1.0, 2.0 * (2.0 * LEAK) * (m * 1e-3))))
    field = math.sqrt(2.0) * m * (1.0 + a_1 + a_2)
    assert gains.crossing_field("CS/face") == pytest.approx(field, rel=1e-12)
    i_max, f, peak = _faint_bound(run(_one_leg_project(form, LEAK, 1e9, unknown=True)), "CS/face")
    assert peak == pytest.approx((field * f) ** 2 * i_max, rel=1e-6)
    for k in (1.2, 4.01, 0.99 * most * most, 0.99 * field * field):  # (formerly WITHIN_RATING above 4 x)
        status, upper = _rating_outcome(_one_leg_project(form, LEAK, k * f * f * i_max, unknown=True), "CS/face")
        assert status == "CONDITIONS_NOT_VERIFIED" and upper > k * f * f * i_max
    status, upper = _rating_outcome(_one_leg_project(form, LEAK, 1.01 * peak, unknown=True), "CS/face")
    assert status == "CONDITIONS_NOT_VERIFIED" and upper <= 1.01 * peak


def _fields_once(surfaces: list[tuple[float, int]]) -> float:
    """``open_paths`` for reflecting surfaces given as (power reflectance bound, meetings): sqrt(2) times 1 plus, after each
    number j of reflections, the smaller of the ordered sum over sequences of j distinct surfaces of the products of
    m_s sqrt(R_s) and the square root of the ordered sums of the products of m_s (the pieces) and m_s R_s (their power,
    at most 1)."""
    from test_power_exposure_ratings import _ordered_sums
    amplitude = _ordered_sums([m * math.sqrt(r) for r, m in surfaces])
    pieces = _ordered_sums([float(m) for _, m in surfaces])
    power = _ordered_sums([m * r for r, m in surfaces])
    return math.sqrt(2.0) * (1.0 + sum(min(amplitude[j], math.sqrt(pieces[j] * min(1.0, power[j])))
                                       for j in range(1, len(surfaces) + 1)))


def _on_obs_project(kind: str, r0: float, rating: float, traced: bool = False, unknown: bool = False) -> ProjectSnapshot:
    """The window's ghost and the two prisms of ``_prisms`` (three crossings of the plane z = 0.25 (x - 10)) with a rated
    element CS on that plane instead of OBS (coated R 1e-3, a CW rating): ``bs_sag``, a sag splitter of curvature 1e-7 /
    mm (R 10 km), or ``refract_air``, a refracting plane between air and air (``unknown``: the lens ``TL`` on the beam)."""
    elements, crossings = _prisms(2)
    first, last = crossings[0], crossings[-1]
    centre = tuple(0.5 * (a + b) for a, b in zip(first, last))
    normal, radius = B.unit((-SLOPE, 0.0, 1.0)), 0.5 * math.dist(first, last) + 5.0
    if kind == "bs_sag":
        surface = SagSurface(surface_id="face", origin_mm=centre, normal=normal, u_axis=B.unit((1.0, 0.0, SLOPE)),
                             curvature_per_mm=1e-7, aperture=CircularAperture(radius_mm=radius),
                             interaction=BeamSplitterInteraction(coating=_coat(1e-3)))
    else:
        surface = PlaneSurface(surface_id="face", origin_mm=centre, normal=normal, aperture=CircularAperture(radius_mm=radius),
                               medium_positive="air", medium_negative="air", interaction=RefractInteraction(coating=_coat(1e-3)))
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    extra = (_unknown_fate_lens(),) if unknown else ()
    return ProjectSnapshot(components=(src, *extra, _ghost_window(r0, traced), *elements, _element("CS", surface, rating)),
                           targets=(Target(target_id="u", device_id="CS", metric="power_rating_utilization", maximum=1.0),))


@pytest.mark.parametrize(("kind", "most"), [("bs_sag", 2), ("refract_air", 1)])
def test_left_out_light_prisms_turn_back_across_a_surface_that_does_not_turn_it_counts_it_per_leg(kind: str, most: int):
    """An earlier version counted a curved splitter
    (every curved surface) and a refracting surface between media of one index (every refracting surface) as met at most
    once between two reflections: CS, which the window's ghost crosses three times on three straight legs (the traced
    control: three hits), had c = 2 and was WITHIN_RATING at ratings below the three meetings in phase (9 x the
    diffraction limit, times the square of the prisms' return factor). Neither turns the light: CS is met as often per leg
    as a straight line meets it, 2 on the splitter (a sphere of R 10 km), 1 on the plane, times L = 8 (the lens TL, the
    window's and the prisms' faces); R = 5e-10 + 4 x 0.25 % + m 1e-3; the field over the ordered sequences of distinct
    reflecting surfaces (``open_paths``, each met once: CS's m meetings are its weights). This remains the open-path
    contribution, but the physical scene can also return between CS and the nonparallel prism/window faces. The closed
    return screen cannot establish an all-orders upper bound there; a rating above the old open-path bound stays unknown
    (and not verified: only light left out as negligible, of unknown size and state, reaches CS, and its rating states no
    test beam or polarization)."""
    control = run(_on_obs_project(kind, 1e-2, 1e9, traced=True))
    face = next(s for s in control.exposures.surfaces if s.surface_id == "CS/face")
    assert len(face.hits) == 3
    gains = _gains(_on_obs_project(kind, LEAK, 1e9, unknown=True))
    assert (gains.reflectors, gains.legs) == (6, 8) and "CS/face" not in gains.once and "P1/front" in gains.once
    m = 8 * most
    assert gains.meetings("CS/face") == m and gains.per_leg == ({"CS/face": 2} if most == 2 else {})
    reflectance = LEAK + 4 * 0.0025 + m * 1e-3
    assert gains.reflectance == pytest.approx(reflectance, rel=1e-12)
    field = m * _fields_once([(LEAK, 1), (0.0025, 1), (0.0025, 1), (0.0025, 1), (0.0025, 1), (1e-3, m)])
    # Isolate the open-path contribution to retain the independent per-leg/count proof.
    open_only = replace(gains, closed_cycles=(), closed_series=None)
    assert open_only.crossing_field("CS/face") == pytest.approx(field, rel=1e-12)
    assert gains.closed_cycles and any("CS/face" in cycle for cycle in gains.closed_cycles)
    assert gains.closed_series is None and math.isinf(gains.crossing_field("CS/face"))
    result = run(_on_obs_project(kind, LEAK, 1e9, unknown=True))
    i_max, f, peak = _faint_bound(result, "CS/face")
    face = next(s for s in result.exposures.surfaces if s.surface_id == "CS/face")
    # (no bound: the peak is an estimate -- the open paths' field times the series 1 / (1 - G) around the one closed trail
    # the search found, G its round gain -- that names the possible returns it leaves out and decides nothing)
    gain = gains.search.cycle_gain
    assert 0.0 < gain < 1.0 and gains.nominal_returns_factor() == pytest.approx(1.0 / (1.0 - gain), rel=1e-12)
    assert f == pytest.approx(gains.factor("CS/face"), rel=1e-12)
    assert peak == pytest.approx((field / (1.0 - gain) * f) ** 2 * i_max, rel=1e-9)
    entry = face.value_basis["faint_light_peak_W_per_cm2"]
    assert entry.kind.value == "ESTIMATE" and entry.assumed == ()
    (omitted,) = entry.omitted
    assert omitted.part.value == "POSSIBLE_RETURN" and omitted.code == "EXPOSURE_BOUND_WITHHELD" and "CS/face" in omitted.at
    assert face.ratings[0].exposure_upper_bound is None and face.rating_status == "CONDITIONS_NOT_VERIFIED"
    assert "FAINT_LIGHT_RECIRCULATES" in face.reason_codes and "OUT_OF_MODEL_SCOPE" in face.reason_codes
    assert result.outcome.execution.outcome.value == "COMPLETED"
    assert result.targets[0].assessment.value == "INDETERMINATE"
    old_open_bound = (field * f) ** 2 * i_max
    for rating in (1.2 * f * f * i_max, 9.0 * f * f * i_max, 1.01 * old_open_bound, 1e30):
        assert _rating_outcome(_on_obs_project(kind, LEAK, rating, unknown=True), "CS/face") == ("CONDITIONS_NOT_VERIFIED", None)


def test_only_lenses_and_refracting_surfaces_between_different_media_are_met_once():
    """The surfaces met at most once between two reflections are those that turn the light: a thin lens, a refracting
    surface between media of different index (a window's faces). Every other surface counts per straight leg: a
    refracting plane between air and air and a plane splitter once per leg, a spherical mirror and a paraboloid splitter
    twice, the W profile four times; L = 4 (the lens and the window's two faces). R adds each bound times its meetings."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.thin_lens("L", focal_length_mm=100.0, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 0.0, 0.0))
    window = window.model_copy(update={"surfaces": tuple(
        s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": _coat(0.004)})}) for s in window.surfaces)})
    air = _air_face("A", 0.001, normal=(1.0, 0.0, 1.0))
    air = air.model_copy(update={"pose": B._pose((0.0, 80.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))})
    plane = B.beam_splitter("P", position=(0.0, -80.0, 0.0), normal=(0.0, 1.0, 0.0), coating=_coat(0.002))

    def curved(iid, at, **profile):
        return _element(iid, SagSurface(surface_id="face", origin_mm=at, normal=(0.0, 0.0, -1.0), u_axis=(1.0, 0.0, 0.0),
                                        aperture=CircularAperture(radius_mm=10.0), **profile))

    mirror = curved("M", (0.0, 0.0, 200.0), curvature_per_mm=0.01,
                    interaction=MirrorInteraction(coating=_coat(0.99), reflective_side="positive"))
    para = curved("Q", (0.0, 0.0, 300.0), curvature_per_mm=0.01, conic=-1.0, interaction=BeamSplitterInteraction(coating=_coat(0.003)))
    w = curved("V", (0.0, 0.0, 400.0), curvature_per_mm=0.2, conic=-1.0, aspheric_coefficients=(-5e-4,),
               interaction=BeamSplitterInteraction(coating=_coat(0.005)))
    gains = _gains(ProjectSnapshot(components=(src, lens, window, air, plane, mirror, para, w)))
    assert gains.legs == 4 and gains.once == frozenset({"L/lens", "W/front", "W/back"})
    assert gains.per_leg == {"M/face": 2, "Q/face": 2, "V/face": 4}
    meetings = {"L/lens": 1, "W/front": 1, "W/back": 1, "A/face": 4, "P/coating": 4, "M/face": 8, "Q/face": 8, "V/face": 16}
    assert {sid: gains.meetings(sid) for sid in meetings} == meetings and gains.meetings() == 4
    assert gains.reflectors == 7  # (every surface but the lens)
    reflectance = 2 * 0.004 + 4 * 0.001 + 4 * 0.002 + 8 * 0.99 + 8 * 0.003 + 16 * 0.005
    assert gains.reflectance == pytest.approx(reflectance, rel=1e-12)


def test_pieces_of_one_reflection_count_from_different_sequences_add_in_phase():
    """Review m1: after ``j`` reflections the left-out light is made of pieces from different sequences of the reflecting
    surfaces, mutually coherent, so their fields add. Two splitters of R = 0.5, met once each: after one reflection the
    two pieces reach a point in phase with sqrt(0.5) + sqrt(0.5) = 1.414 x one crossing of the whole piece's field (a
    balanced Mach-Zehnder brings its two arms together so), more than the square root of their total power, 1, which the
    field counted before; after two reflections the sequences (a, b) and (b, a) bring 2 x 0.5 = 1. So c = sqrt(2) (1 +
    sqrt(2) + 1), not sqrt(2) (1 + 1 + 1). Where many pieces share little power their number bounds the sum (Cauchy-Schwarz):
    four surfaces of R = 1 (bare faces) after two reflections: 12 ordered pairs of distinct surfaces sharing at most the
    whole power, sqrt(12), not 12."""
    split = resonance.ReturnGains(LAM, reflectors=2, reflectance=1.0, reflecting=((math.sqrt(0.5), 1), (math.sqrt(0.5), 1)))
    assert split.crossing_field() == pytest.approx(math.sqrt(2.0) * (2.0 + math.sqrt(2.0)), rel=1e-12)
    bare = resonance.ReturnGains(LAM, reflectors=4, reflectance=4.0, reflecting=((1.0, 1),) * 4)
    series = 1.0 + math.sqrt(4.0) + math.sqrt(12.0) + math.sqrt(24.0) + math.sqrt(24.0)
    assert bare.crossing_field() == pytest.approx(math.sqrt(2.0) * series, rel=1e-12)
