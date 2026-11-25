"""The tracer's partial-beam decisions, called directly (``optics/rays/decisions.py``): no scene is lowered, no beam is traced.

Each test builds the explicit inputs a decision reads -- a ``SceneIndex`` of a few hand-made surfaces, a ``BeamView``, the
step's ray parameters per surface (``times``) and a leg -- and pins what the tracer decides today for the residual classes
of the intersecting-plane decision (they document the present behaviour; changing it is a physics decision of its own):

* conservative refusal near physical surfaces: ``taken_before`` puts a transparent plane's power out of scope for any share
  above the line's negligible level, however small (residual 1), and now returns that share;
* grazing / escaping tails: ``leg_partial_crossings`` bounds a leg that leaves the scene by the energy flow, records the
  whole beam without a Gaussian bound for a plane within the reach of the pupil rays' envelope, and nothing beyond it
  (``NO_BOUND_NO_RAY``) or where the bound is negligible;
* the fold hit-point convention: ``crossing_group`` puts an observation plane within ``ray_min_advance_mm`` of a physical
  surface, on either side of it, before that surface (it records the beam arriving there);
* what an observation plane reads (``observation_reading``): active area, acceptance cone, quantized detected power.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from pydantic import ValidationError

from os_optics import NumericalConfig
from os_optics.optics.gaussian import gauss_legendre_rule, initial_state
from os_optics.optics.rays import decisions as D
from os_optics.optics.surfaces import Hits
from os_optics.scene.model import ApertureG, InteractionG, SurfaceG

CFG = NumericalConfig()
QUAD = gauss_legendre_rule(CFG.aperture_quad_sector_nodes)
X, Y, Z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
POWER = 1e-3
LEVEL = CFG.negligible_power_fraction * POWER  # the line's negligible level


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / math.sqrt(float(v @ v))


def _surface(sid: str, origin, normal, kind: str = "observe", *, absorbing: bool = False, area: ApertureG | None = None,
             acceptance: float | None = None) -> SurfaceG:
    n = _unit(normal)
    helper = Z if abs(float(n @ Z)) < 0.9 else X
    u = _unit(np.cross(helper, n))
    v = np.cross(n, u)
    observe = kind == "observe"
    inter = InteractionG(kind, observation_id=sid if observe else None, absorbing=absorbing, active_aperture=area,
                         acceptance_half_angle_deg=acceptance)
    return SurfaceG(surface_id=f"observation/{sid}" if observe else f"{sid}/face", owner_id=sid, geometry="plane",
                    origin=np.asarray(origin, dtype=float), normal=n, u_axis=u, v_axis=v, center=np.zeros(3), radius=0.0,
                    cap_axis=n, aperture=ApertureG(), mount_margin=0.0, medium_pos="air", medium_neg="air", interaction=inter)


def _gstate(w0: float = 0.5, wavelength_nm: float = 780.0, z_waist: float = 0.0):
    return initial_state(wavelength_nm=wavelength_nm, n=1.0, w0x=w0, w0y=w0, z_waist=z_waist, m2x=1.0, m2y=1.0, bx=Y, by=Z, cfg=CFG)


def _beam(o: np.ndarray, d: np.ndarray, *, gstate=None, alive=None, pupil: int | None = None) -> D.BeamView:
    """A beam of ``len(o)`` rays (ray 0 the chief ray, rays ``1 .. pupil`` pupil samples), 1 mW on a 1 mW line."""
    n = o.shape[0]
    pupil = n - 1 if pupil is None else pupil
    annulus = np.zeros(n, dtype=bool)
    annulus[1:1 + pupil] = True
    return D.BeamView("L#0", "L", 0, 780.0, o, d, np.ones(n, dtype=bool) if alive is None else alive, annulus,
                      _gstate() if gstate is None else gstate, POWER, False, 1.0, LEVEL)


def _leg(length: float, *, gstate=None, start=None, d=X, end_normal=X) -> D.Leg:
    """A leg along ``d`` from ``start`` (default the origin) between two planes across it, ``length`` apart."""
    start = np.zeros(3) if start is None else np.asarray(start, dtype=float)
    return D.Leg(start, d, length, d, 0.0, end_normal, length, 0.0, length, _gstate() if gstate is None else gstate)


def _plane_times(surfaces: list[SurfaceG], o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Every ray's parameter at every (unbounded) plane, ``inf`` where it runs along it or the plane lies behind it."""
    times = np.full((len(surfaces), o.shape[0]), np.inf)
    for i, s in enumerate(surfaces):
        along = d @ s.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((s.origin - o) @ s.normal) / along
        times[i] = np.where((np.abs(along) > CFG.parallel_dot_tol) & (t > 0.0), t, np.inf)
    return times


# --------------------------------------------------------------------------------------------- crossing group
def _times(*chief: float) -> np.ndarray:
    return np.array(chief, dtype=float)[:, None]


def test_a_plane_within_the_advance_guard_of_a_fold_mirror_records_the_beam_arriving_on_either_side():
    """Fold hit-point convention: a monitor within ``ray_min_advance_mm`` of the mirror's hit point, on the incident or on
    the reflected side of it, is crossed before the mirror acts (transparent planes before absorbing ones, then by id)."""
    index = D.SceneIndex.of([_surface("M", (10, 0, 0), (-1, 1, 0), "mirror"), _surface("T", (10, 0, 0), X),
                             _surface("D", (10, 0, 0), X, absorbing=True), _surface("far", (50, 0, 0), X)])
    guard = CFG.ray_min_advance_mm
    for t_plane in (10.0 - 0.5 * guard, 10.0 + 0.5 * guard):
        times = _times(10.0, t_plane, t_plane, 50.0)
        chief = int(np.argmin(times[:, 0]))
        assert D.crossing_group(index, chief, times, CFG) == ([1, 2, 0], 0)
    # just beyond the guard: the mirror alone (the plane is crossed after the reflection, if at all)
    times = _times(10.0, 10.0 + 2.0 * guard, 10.0 + 2.0 * guard, 50.0)
    assert D.crossing_group(index, 0, times, CFG) == ([0], 0)


def test_planes_of_one_crossing_are_ordered_by_kind_and_id_not_by_declaration():
    surfaces = [_surface("b", (5, 0, 0), X), _surface("dump", (5, 0, 0), X, absorbing=True), _surface("a", (5, 0, 0), _unit((1, 1, 0)))]
    index = D.SceneIndex.of(surfaces)
    times = _times(5.0, 5.0, 5.0 + 0.5 * CFG.ray_min_advance_mm)
    assert D.crossing_group(index, 0, times, CFG) == ([2, 0, 1], None)
    assert D.observation_crossing(index, 0, times, CFG) == [2, 0, 1]
    assert D.coincident_crossing(index, 0, times, CFG) is None


# ------------------------------------------------------------------------------------------ leg partial crossings
def _leg_scene(*planes: SurfaceG, end_x: float = 100.0):
    """A leg along +x from the origin to a mirror at ``end_x`` (index 0), and the observation planes after it."""
    return [_surface("E", (end_x, 0, 0), (-1, 0, 0), "mirror"), *planes]


def _decide(surfaces, beam, *, leg, prefilter=False, crossed=(), end_si=0):
    index = D.SceneIndex.of(surfaces)
    times = _plane_times(surfaces, beam.o, beam.d)
    candidates = D.leg_end_candidates(index, crossed, ())
    return {surfaces[c.surface_index].surface_id: c for c in D.leg_partial_crossings(
        index, beam, times, times, candidates, crossed, end_si, leg, "E/face", CFG, QUAD, prefilter=prefilter)}


def _chief_only() -> D.BeamView:
    return _beam(np.zeros((1, 3)), X[None, :])


def test_a_plane_along_the_leg_near_the_beam_records_the_gaussian_share_beyond_it():
    beam = _chief_only()
    near = _decide(_leg_scene(_surface("near", (0, 0.6, 0), Y)), beam, leg=_leg(100.0))["observation/near"]
    assert near.outcome == D.RECORDED and near.rays == 0
    assert near.share_W == near.bound_W == near.crossing.power_W
    # the part of the beam beyond the plane, its radius taken at its widest on the leg (about 0.8 % of the beam)
    expected = 0.5 * math.erfc(math.sqrt(2.0) * 0.6 / D.widest(_gstate(), (0.0, 100.0))) * POWER
    assert near.share_W == pytest.approx(expected, rel=1e-9)


def test_a_tail_below_the_negligible_level_is_not_recorded_and_its_bound_is_visible():
    """Residual 4 (grazing tails): the Gaussian share of a plane along the leg, far from the beam, is at most the line's
    negligible level: nothing is recorded (the plane reads a known zero for this leg); the bound that decided it is
    returned. With the pre-filter on, such a plane is skipped before its share is computed."""
    beam = _chief_only()
    surfaces = _leg_scene(_surface("far", (0, 5.0, 0), Y))
    far = _decide(surfaces, beam, leg=_leg(100.0))["observation/far"]
    assert far.outcome == D.NEGLIGIBLE and far.crossing is None
    assert far.share_W is not None and far.share_W <= LEVEL
    skipped = _decide(surfaces, beam, leg=_leg(100.0), prefilter=True)["observation/far"]
    assert skipped.outcome == D.PREFILTERED and skipped.share_W is None


def _leaving(gstate=None, start_normal=X) -> D.Leg:
    """A leg along +x from the origin that leaves the scene."""
    return D.Leg(np.zeros(3), X, math.inf, start_normal, 0.0, X, None, 0.0, math.inf, _gstate() if gstate is None else gstate)


def test_an_escaping_leg_is_bounded_by_the_energy_flow_out_of_the_scene():
    """Escaping tails: on a leg that leaves the scene the lines beyond a plane along it
    reach it as the beam widens: from the share beyond it at the leg's start (nil at the waist) to half the beam in the limit."""
    pupil = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    beam = _beam(pupil, np.repeat(X[None, :], 3, 0))
    surfaces = [_surface("tail", (0, 1.0, 0), Y)]  # (no end surface)
    tail = _decide(surfaces, beam, leg=_leaving(), end_si=None)["observation/tail"]
    start = 0.5 * math.erfc(math.sqrt(2.0) * 1.0 / 0.5)
    assert tail.outcome == D.RECORDED and tail.rays == 0
    assert tail.share_W == pytest.approx((0.5 - start) * POWER, rel=1e-12)
    # a pupil ray heading across the plane: the whole beam is the bound, as without a Gaussian state
    diverging = _beam(pupil, np.array([X, _unit((1, 0.05, 0)), X]))
    reached = _decide(surfaces, diverging, leg=_leaving(), end_si=None)["observation/tail"]
    assert reached.outcome == D.RECORDED and reached.rays == 1 and reached.crossing.power_W == POWER


def test_on_an_escaping_leg_a_plane_without_an_active_area_counts_what_crosses_it_within_the_scene():
    """``Leg.scene_end``: the share beyond the plane there. It is the reach of a laboratory scene, or the farthest along the ray
    of the physical surfaces, source points and active areas, and no plane's origin counts."""
    beam = _chief_only()
    surfaces = [_surface("tail", (0, 1.0, 0), Y)]
    leg = dataclasses.replace(_leaving(), scene_end=500.0)
    tail = _decide(surfaces, beam, leg=leg, end_si=None)["observation/tail"]
    w = D.widest(_gstate(), (500.0,))
    assert tail.share_W == pytest.approx((0.5 * math.erfc(math.sqrt(2.0) / w) - 0.5 * math.erfc(math.sqrt(2.0) / 0.5)) * POWER, rel=1e-9)
    # the prefilter holds the same stretch
    far = dataclasses.replace(_leaving(), scene_end=500.0)
    distant = _decide([_surface("far", (0, 50.0, 0), Y)], beam, leg=far, end_si=None, prefilter=True)["observation/far"]
    assert distant.outcome == D.PREFILTERED
    near = D.SceneIndex.of([_surface("a", (10, 0, 0), X, area=ApertureG("circle", p1=2.0)), _surface("M", (0, 0, 0), X, "mirror")],
                           (np.array([-10.0, 0.0, 0.0]),))
    assert near.reach_along(np.zeros(3), X) == 1000.0
    assert D.SceneIndex.of([]).reach_along(np.zeros(3), X) == 1000.0
    # an active area along the ray extends it by its own reach; a plane's origin, or a part off to the side, does not
    far = D.SceneIndex.of([_surface("a", (3000.0, 7.0, 0), X, area=ApertureG("circle", p1=2.0)), _surface("b", (9000.0, 0, 0), X),
                           _surface("side", (0.0, 5000.0, 0), Y, area=ApertureG("circle", p1=2.0))])
    assert far.reach_along(np.zeros(3), X) == pytest.approx(3002.0, rel=1e-12)
    assert far.reach_along(np.array([1000.0, 0.0, 0.0]), X) == pytest.approx(2002.0, rel=1e-12)


@pytest.mark.parametrize(("support", "reach"), ((998.0, 1000.0), (1000.0, 1000.0), (1000.5, 1000.5), (2000.0, 2000.0)))
def test_the_reach_is_the_declared_value_or_the_scene_s_support_whichever_is_farther(support: float, reach: float):
    """max(1000 mm, support) on both sides of the boundary: a part whose far edge lies at ``support`` along the ray."""
    part = _surface("a", (support - 2.0, 7.0, 0), X, area=ApertureG("circle", p1=2.0))
    assert D.SceneIndex.of([part]).reach_along(np.zeros(3), X) == pytest.approx(reach, rel=1e-12)


def test_the_reach_is_a_declared_value_of_the_run_and_cannot_be_set():
    """It reaches the decisions from the run's configuration, every run's manifest records it through the numerical tolerances
    that ``_manifest`` copies, and any other value is refused."""
    config = NumericalConfig()
    assert config.laboratory_reach_mm == 1000.0 and "laboratory_reach_mm" in config.model_dump()
    for value in (999.0, 1001.0):
        with pytest.raises(ValidationError):
            NumericalConfig(laboratory_reach_mm=value)
    from helpers import run
    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot
    laser = B.laser("L", wavelength_nm=780.0, waist_radius_mm=0.5, power_W=1e-3, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    plane = B.observation_plane("T", position=(10.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=5.0)
    result = run(ProjectSnapshot(components=(laser,), observations=(plane,)))
    assert result.manifest.numerical_tolerances["laboratory_reach_mm"] == 1000.0


def test_a_severely_truncated_beam_reaches_a_plane_ahead_of_the_leg_whatever_its_rays_say():
    """No Gaussian state for that reason, no live pupil ray near the plane: the whole beam, marked as diffracted, and not taken
    out of the beam on an absorbing plane; behind the leg's start nothing."""
    from os_optics.errors import ErrorCode

    invalid = _gstate().invalidated(ErrorCode.GAUSSIAN_SEVERE_TRUNCATION, "cut")
    pupil = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    beam = _beam(pupil, np.repeat(X[None, :], 2, 0), gstate=invalid)
    ahead = _surface("ahead", (50.0, 30.0, 0.0), Y, absorbing=True, area=ApertureG("circle", p1=2.0))
    behind = _surface("behind", (-50.0, 30.0, 0.0), Y, area=ApertureG("circle", p1=2.0))
    decided = _decide([ahead, behind], beam, leg=None, end_si=None)
    assert decided["observation/ahead"].crossing.diffracted and decided["observation/ahead"].crossing.power_W == POWER
    assert not decided["observation/ahead"].crossing.absorbing
    assert decided["observation/behind"].outcome == D.NO_BOUND_NO_RAY
    # light followed only to bound it keeps the rays' envelope
    index, times = D.SceneIndex.of([ahead]), _plane_times([ahead], beam.o, beam.d)
    (probe,) = D.leg_partial_crossings(index, beam, times, times, [0], (), None, None, None, CFG, QUAD, traced=False)
    assert probe.outcome == D.NO_BOUND_NO_RAY
    # another reason for no Gaussian state keeps the rays' envelope
    other = _beam(pupil, np.repeat(X[None, :], 2, 0), gstate=_gstate().invalidated(ErrorCode.GAUSSIAN_ABERRATION_DOMINATED, "rays"))
    assert _decide([ahead], other, leg=None, end_si=None)["observation/ahead"].outcome == D.NO_BOUND_NO_RAY


def test_an_escaping_leg_keeps_a_plane_across_it_or_tilted_far_from_its_far_field_dark():
    """A plane across the leg behind its start: every line meets it where the chief ray does, behind the start. A plane
    tilted 45 deg behind the start: the far field would have to run 45 deg off the axis, thousands of radii out."""
    beam = _chief_only()
    behind = _decide([_surface("behind", (-5, 0, 0), X)], beam, leg=_leaving(), end_si=None)["observation/behind"]
    tilted = _decide([_surface("tilted", (-5, 0, 0), _unit((1, 1, 0)))], beam, leg=_leaving(), end_si=None)["observation/tilted"]
    assert behind.outcome == D.NEGLIGIBLE and behind.share_W == 0.0
    assert tilted.outcome == D.NEGLIGIBLE and tilted.share_W <= LEVEL
    surfaces = [_surface("behind", (-5, 0, 0), X), _surface("tilted", (-5, 0, 0), _unit((1, 1, 0)))]
    skipped = _decide(surfaces, beam, leg=_leaving(), end_si=None, prefilter=True)
    assert {c.outcome for c in skipped.values()} == {D.PREFILTERED}


def test_without_a_gaussian_bound_the_pupil_rays_envelope_decides():
    """No Gaussian bound (``leg`` None) and no pupil ray crossing the plane: within three envelopes of the live pupil rays
    (they lie 0.5 mm from the chief ray: 1.5 mm) the plane may receive the light, and the whole beam bounds it; beyond,
    nothing says light gets there; and without a live ray nothing says where the light is."""
    pupil = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    beam = _beam(pupil, np.repeat(X[None, :], 3, 0))
    near = _decide([_surface("near", (0, 1.0, 0), Y)], beam, leg=None, end_si=None)["observation/near"]
    assert near.outcome == D.RECORDED and near.share_W is None and near.rays == 0 and near.crossing.power_W == POWER
    far = _decide([_surface("far", (0, 2.0, 0), Y)], beam, leg=None, end_si=None)["observation/far"]
    assert far.outcome == D.NO_BOUND_NO_RAY and far.share_W is None and far.bound_W == 0.0 and far.crossing is None
    dead = _beam(pupil, np.repeat(X[None, :], 3, 0), alive=np.array([True, False, False]))
    none = _decide([_surface("near", (0, 1.0, 0), Y)], dead, leg=None, end_si=None)["observation/near"]
    assert none.outcome == D.NO_BOUND_NO_RAY
    # an active area off the envelope's reach along the plane stays out of it too
    away = _surface("away", (0.0, 1.0, 50.0), Y, area=ApertureG("circle", p1=2.0))
    assert _decide([away], beam, leg=None, end_si=None)["observation/away"].outcome == D.NO_BOUND_NO_RAY


def test_a_plane_across_the_leg_beyond_its_end_is_not_crossed_first():
    beam = _chief_only()
    behind = _decide(_leg_scene(_surface("behind", (150, 0, 0), X)), beam, leg=_leg(100.0))["observation/behind"]
    assert behind.outcome == D.NEGLIGIBLE and behind.share_W == 0.0


# --------------------------------------------------------------------------------------------------- taken before
def _taken(surfaces, si, beam, *, lead=None, start=None, leg=None, no_leg=False):
    index = D.SceneIndex.of(surfaces)
    times = _plane_times(surfaces, beam.o, beam.d)
    t_block, nearest = D.first_blocking(index.blocking_ids, times, (si,))
    leg = None if no_leg else _leg(float(times[si, 0])) if leg is None else leg
    return D.taken_before(index, beam, times, si, times[si], t_block, nearest, leg, (), lead, start, CFG, QUAD)


def _fold_lead(x: float) -> D.NextSurface:
    """A 45 deg fold mirror whose chief-ray hit is at ``x`` on the axis, as its own plane."""
    return D.NextSurface("M/face", D.Sheet(np.array([x, 0.0, 0.0]), _unit((-1, 1, 0))))


@pytest.mark.parametrize(("gap_mm", "taken"), [(0.3, True), (1.2, True), (1.6, False)])
def test_a_tiny_share_meeting_the_next_surface_first_puts_the_whole_reading_out_of_scope(gap_mm, taken):
    """Residual 1 (conservative refusal): a transparent plane T just before a 45 deg fold mirror (its chief-ray hit
    ``gap_mm`` behind T). The part of a 0.5 mm beam more than ``gap_mm`` off the axis on the mirror's near side meets
    the mirror before T; any share above the line's negligible level -- at 1.2 mm about 8e-7 of the beam -- counts the
    mirror, so T's power is out of scope altogether (``power_scope``) rather than bounded by ``[P (1 - share), P]``. The
    share that decided it is returned."""
    surfaces = [_surface("M", (50.0 + gap_mm, 0, 0), (-1, 1, 0), "mirror"), _surface("T", (50.0, 0, 0), X)]
    (entry,) = _taken(surfaces, 1, _chief_only(), lead=_fold_lead(50.0 + gap_mm))
    assert (entry.surface_id, entry.why, entry.taken) == ("M/face", D.NEXT_SURFACE, taken)
    # the half-plane y < -gap of the beam (it meets the mirror x = x_M + y first), its radius at its widest on the leg
    expected = 0.5 * math.erfc(math.sqrt(2.0) * gap_mm / D.widest(_gstate(), (0.0, 50.0))) * POWER
    assert entry.share_W == pytest.approx(expected, rel=1e-9)
    assert (entry.share_W > LEVEL) is taken


def test_an_absorbing_plane_before_the_monitor_is_weighed_by_its_share_or_by_the_pupil_rays():
    """A 45 deg absorbing plane 1 mm before the monitor T, its active area a 0.2 mm disc on the axis: it takes the
    beam's Gaussian share on that area (about a fifth of a 0.5 mm beam) before T. Without a Gaussian bound (no leg) a
    live pupil ray that meets it first (``blocking``: here the whole plane) counts it; none: it is weighed and not taken."""
    beam = _beam(np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]]), np.repeat(X[None, :], 2, 0))
    dump = _surface("dump", (49.0, 0.0, 0.0), _unit((1, -1, 0)), absorbing=True, area=ApertureG("circle", p1=0.2))
    monitor = _surface("T", (50.0, 0, 0), X)
    (entry,) = _taken([dump, monitor], 1, beam)
    assert (entry.surface_id, entry.why, entry.taken) == ("observation/dump", D.ABSORBING_PLANE, True)
    assert 0.15 * POWER < entry.share_W < 0.25 * POWER
    index, times = D.SceneIndex.of([dump, monitor]), _plane_times([dump, monitor], beam.o, beam.d)
    none = (np.full(2, np.inf), np.full(2, -1))
    (unbounded,) = D.taken_before(index, beam, times, 1, times[1], *none, None, (), None, None, CFG, QUAD)
    assert unbounded.share_W is None and unbounded.taken  # (the pupil ray meets the dump at t = 49.4, before T at 50)
    missed = times.copy()
    missed[0] = np.inf  # (no live pupil ray meets the dump)
    (unseen,) = D.taken_before(index, beam, missed, 1, times[1], *none, None, (), None, None, CFG, QUAD)
    assert unseen.share_W is None and not unseen.taken


def test_a_surface_a_pupil_ray_meets_first_is_counted_once():
    """A live pupil ray meets the fold mirror before the monitor: the mirror is counted once, weighed as the next surface
    with its Gaussian share and the bound of what it takes; without a leg (no Gaussian bound) it is the pupil ray's
    surface, with no bound."""
    beam = _beam(np.array([[0.0, 0.0, 0.0], [0.0, -0.4, 0.0]]), np.repeat(X[None, :], 2, 0))  # (meets x = 50.3 + y at 49.9)
    surfaces = [_surface("M", (50.3, 0, 0), (-1, 1, 0), "mirror"), _surface("T", (50.0, 0, 0), X)]
    (entry,) = _taken(surfaces, 1, beam, lead=_fold_lead(50.3))
    assert (entry.surface_id, entry.why, entry.taken) == ("M/face", D.NEXT_SURFACE, True)
    assert entry.share_W is not None and entry.bound_W >= entry.share_W
    (entry,) = _taken(surfaces, 1, beam, lead=_fold_lead(50.3), no_leg=True)
    assert (entry.surface_id, entry.why, entry.share_W, entry.taken, entry.bound_W) == ("M/face", D.PUPIL_RAY_FIRST, None, True, None)


def test_a_reflecting_start_surface_counts_when_part_of_the_beam_leaves_it_beyond_the_plane():
    """The leg began at a fold mirror whose reflection sent part of the beam beyond the tilted plane T crossed on the leg."""
    beam = _chief_only()
    plane = _surface("T", (0.3, 0, 0), _unit((1, 1, 0)))
    start = D.StartSurface("F/face", (_unit((1, -1, 0)), 0.0))
    (entry,) = _taken([plane], 0, beam, start=start, leg=_leg(10.0))
    assert (entry.surface_id, entry.why, entry.taken) == ("F/face", D.START_SURFACE, True)
    # rays at y > 0.15 mm leave the mirror (x = y) beyond T (x = 0.3 - y): about 27 % of the beam
    expected = 0.5 * math.erfc(math.sqrt(2.0) * 0.15 / D.widest(_gstate(), (0.0, 10.0))) * POWER
    assert entry.share_W == pytest.approx(expected, rel=1e-9)
    (unbounded,) = _taken([plane], 0, beam, start=D.StartSurface("F/face", None), leg=_leg(10.0))
    assert unbounded.share_W is None and not unbounded.taken


# ---------------------------------------------------------------------------------------------- observation reading
def _hit_on(surf: SurfaceG, points: np.ndarray) -> Hits:
    rel = points - surf.origin
    n = points.shape[0]
    return Hits(np.ones(n), points, np.repeat(surf.normal[None, :], n, 0), np.ones(n, dtype=bool), rel @ surf.u_axis,
                rel @ surf.v_axis)


def test_an_absorbing_area_detects_the_beam_share_on_it_quantized():
    detector = _surface("D", (10, 0, 0), X, absorbing=True, area=ApertureG("circle", p1=0.6))
    beam = _chief_only()
    reading = D.observation_reading(beam, detector, _hit_on(detector, np.array([[10.0, 0.0, 0.0]])), 1.0, 0.0, None, CFG, QUAD)
    assert reading.counted and reading.area_fraction is None and reading.accepted is None
    quantum = CFG.power_ledger_fraction_resolution
    expected = math.exp(-2.0 * 0.6 ** 2 / 0.5 ** 2)  # (the round beam's power beyond 0.6 mm)
    assert reading.outside == pytest.approx(round(expected / quantum) * quantum, abs=quantum)
    assert reading.outside / quantum == round(reading.outside / quantum)
    assert reading.detected_W == POWER * (1.0 - reading.outside) and reading.detected_min_W is None


def test_a_beam_beside_a_small_area_does_not_count_and_its_fraction_is_reported():
    monitor = _surface("S", (10, 0, 0), X, area=ApertureG("circle", p1=0.1, offset_u=0.8), acceptance=5.0)
    reading = D.observation_reading(_chief_only(), monitor, _hit_on(monitor, np.array([[10.0, 0.0, 0.0]])), 1.0, 0.0, None, CFG, QUAD)
    assert not reading.counted and 0.0 < reading.area_fraction < CFG.capture_min_power_fraction
    assert reading.accepted is True and reading.tail_accepted is True and reading.detected_W is None


def test_a_detector_outside_its_acceptance_cone_detects_nothing():
    detector = _surface("D", (10, 0, 0), X, absorbing=True, acceptance=5.0)
    reading = D.observation_reading(_chief_only(), detector, _hit_on(detector, np.array([[10.0, 0.0, 0.0]])), math.cos(math.radians(10.0)),
                                    10.0, None, CFG, QUAD)
    assert reading.counted and reading.accepted is False and reading.detected_W == 0.0


def test_an_annulus_counted_whole_keeps_the_bound_of_what_may_fall_beside():
    detector = _surface("D", (10, 0, 0), X, absorbing=True, area=ApertureG("circle", p1=2.0))
    annulus = dataclasses.replace(_chief_only(), chief_obstructed=True, annulus_fraction=0.9)
    reading = D.observation_reading(annulus, detector, _hit_on(detector, np.array([[10.0, 0.0, 0.0]])), 1.0, 0.0,
                                    D.AnnulusOnArea(True, None, 0.01), CFG, QUAD)
    assert reading.counted and reading.outside == 0.0 and reading.detected_W == POWER and reading.detected_min_W == POWER * 0.99
