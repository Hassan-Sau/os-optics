"""Dark planes and surface coincidence, target independence and the incoherent sum at plane level.

* A plane with no traced arrival that followed or unknown-fate light may reach is UNKNOWN, never a known zero and not
  bounded, whatever a supported bound says; the bound of a leg that leaves the scene or
  of a nearly parallel plane is the paraxial beam's energy flow (``decisions.flow_crossing``), checked here against its own
  numerical integration of that flow over the plane, not against the engine.
* A plane through a physical surface's chief-ray hit point records the power arriving at that surface, and says so
  (``surface_coincidence``, ``SURFACE_COINCIDENT_ARRIVAL``).
* A plane's physical record does not depend on targets.
* A plane's ordinary power is labelled the incoherent sum where a coherent one exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.optics.gaussian import gauss_legendre_rule, initial_state
from os_optics.optics.rays import decisions as D
from os_optics.scene.model import ApertureG, InteractionG, SurfaceG

from helpers import arrivals, assert_dark_truncated, one_arrival, run

P = 1e-3
LAM = 780.0
Q = math.sqrt(0.5)
X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
CFG = NumericalConfig()
RULE = gauss_legendre_rule(CFG.aperture_quad_sector_nodes)


def _max(value: float, oid: str = "T", metric: str = "power_W", **kw) -> Target:
    return Target(target_id=f"{oid}{metric}max{value:g}", metric=metric, observation_id=oid, maximum=value, unit="W", **kw)


def _min(value: float, oid: str = "T", metric: str = "power_W", **kw) -> Target:
    return Target(target_id=f"{oid}{metric}min{value:g}", metric=metric, observation_id=oid, minimum=value, unit="W", **kw)


def _obs(result, oid: str = "T"):
    return next(o for o in result.observations if o.observation_id == oid)


# ----------------------------------------------------------------------------------- the independent energy flow
def _flow_through_disc(start, d, qx: complex, qy: complex, bx, by, centre, normal, radius: float, power: float = P,
                       s_range: tuple[float, float] = (-math.inf, math.inf), nodes: int = 600, after=None) -> float:
    """The power a paraxial Gaussian beam's energy flow carries across a disc, each crossing counted, by quadrature on the part
    of the disc on the stretch ``s_range`` of the chief ray (Gauss-Legendre along the chief ray's direction in the plane and
    along each chord across it): at each point the irradiance of the beam at that point's place along the chief ray, times
    the component along the normal of the local flow direction ``d + x kappa_x bx + y kappa_y by``, ``kappa`` the wavefront's
    curvature ``z / (z^2 + z_R^2)``. ``after``: planes (point, normal) the light has left: a point counts on their positive
    side. Written from the beam's own equations; nothing of the engine's bound."""
    scale = LAM * 1e-6 / math.pi
    d, n, start, centre, bx, by = (np.asarray(v, float) for v in (d, normal, start, centre, bx, by))
    e1 = d - (d @ n) * n
    e1 = e1 / math.sqrt(float(e1 @ e1))
    e2 = np.cross(n, e1)
    # (s is linear in the coordinate a along e1: the stretch cuts a to a range)
    s_centre, rate = float((centre - start) @ d), float(e1 @ d)
    low, high = max(-radius, (s_range[0] - s_centre) / rate), min(radius, (s_range[1] - s_centre) / rate)
    if low >= high:
        return 0.0
    t, weights = np.polynomial.legendre.leggauss(nodes)
    a = 0.5 * (high - low) * t + 0.5 * (high + low)
    half = np.sqrt(np.maximum(radius * radius - a * a, 0.0))
    b = half[:, None] * t[None, :]
    weight = (0.5 * (high - low) * weights)[:, None] * (half[:, None] * weights[None, :])
    points = centre - start + a[:, None, None] * e1 + b[..., None] * e2
    s = points @ d
    x, y = points @ bx, points @ by
    zx, zy = qx.real + s, qy.real + s
    wx2, wy2 = scale * (zx * zx + qx.imag * qx.imag) / qx.imag, scale * (zy * zy + qy.imag * qy.imag) / qy.imag
    irradiance = 2.0 * power / (math.pi * np.sqrt(wx2 * wy2)) * np.exp(-2.0 * (x * x / wx2 + y * y / wy2))
    kx, ky = zx / (zx * zx + qx.imag * qx.imag), zy / (zy * zy + qy.imag * qy.imag)
    along = float(d @ n) + x * kx * float(bx @ n) + y * ky * float(by @ n)
    keep = np.ones(s.shape, dtype=bool)
    for point, side in after or ():
        keep &= (points - (np.asarray(point, float) - start)) @ np.asarray(side, float) >= 0.0
    return float(np.sum(irradiance * np.abs(along) * weight * keep))


W0 = 0.02
Z_R = math.pi * W0 * W0 / (LAM * 1e-6)


def _reproduced_flux(theta_deg: float) -> float:
    """1 mW 780 nm, waist 20 um at the origin, direction ``theta_deg`` off +X toward +Y; T the 5 mm disc through (90, 2, 0)
    with normal +Y."""
    t = math.radians(theta_deg)
    d = (math.cos(t), math.sin(t), 0.0)
    return _flow_through_disc((0, 0, 0), d, complex(0.0, Z_R), complex(0.0, Z_R), (-math.sin(t), math.cos(t), 0.0), (0.0, 0.0, 1.0),
                              (90.0, 2.0, 0.0), Y, 5.0)


def _tail_scene(theta_deg: float, dump: bool, targets=(), radius: float | None = 5.0) -> ProjectSnapshot:
    t = math.radians(theta_deg)
    d = (math.cos(t), math.sin(t), 0.0)
    parts = [B.laser("L", wavelength_nm=LAM, waist_radius_mm=W0, power_W=P, position=(0.0, 0.0, 0.0), direction=d)]
    if dump:
        parts.append(B.beam_dump("DUMP", position=(100.0 * d[0], 100.0 * d[1], 0.0), axis=d, radius_mm=20.0))
    plane = B.observation_plane("T", position=(90.0, 2.0, 0.0), normal=Y, radius_mm=radius)
    return ProjectSnapshot(components=tuple(parts), observations=(plane,), targets=tuple(targets))


TAILS = ((0.0, False), (0.01, False), (-0.01, False), (0.0, True), (0.01, True), (-0.01, True), (0.2, True), (-0.2, True))


@pytest.mark.parametrize(("theta", "dump"), TAILS)
def test_a_diverging_beams_tail_through_a_nearly_parallel_monitor_is_not_a_known_zero(theta: float, dump: bool):
    """The escaping-tail fixture. The recorded false reading was T KNOWN_ZERO, NO_TRACED_ARRIVAL and a 1e-7 W maximum MET, for
    2.7e-7 W through the disc (or a bound below that: 6.7e-9 W at 0.01 deg, 5.6e-7 W for 1.7e-6 W at 0.2 deg).

    The plane reads BOUNDED [0, bound] with no power value. Its targets are graded on [0, bound] (targets only): the bound holds the independently integrated flux, so a maximum is MET only where
    that flux lies under it, a minimum above the bound is UNMET, and a maximum under the bound is not decided -- the
    recorded false MET stays closed."""
    flux = _reproduced_flux(theta)
    targets = (_max(1e-7), _max(1.0), _min(1e-3))
    result = run(_tail_scene(theta, dump, targets))
    obs = _obs(result)
    assert obs.arrivals == () and obs.total_power_W is None and not obs.power_complete
    assert obs.quantity_state.value == "BOUNDED" and obs.status.value == "PASS" and obs.power_lower_bound_W == 0.0
    # the bound holds the flux, tightly: the flow's strip of the plane around a disc wider than the beam
    assert flux <= obs.power_upper_bound_W <= 1.02 * flux
    assert obs.untraced_power_W == pytest.approx(obs.power_upper_bound_W, rel=1e-9)
    tight, loose, least = result.targets
    assert tight.assessment.value == ("MET" if obs.power_upper_bound_W <= 1e-7 else "INDETERMINATE")
    assert tight.assessment.value != "MET" or flux <= 1e-7  # (never met while the flux through the disc exceeds the maximum)
    assert loose.assessment.value == "MET" and least.assessment.value == "UNMET"
    for target in result.targets:
        assert target.value is None  # (no traced light: no number, only the interval)
        assert target.issue_codes
        assert "NO_TRACED_ARRIVAL" not in target.issue_codes


def test_the_fixture_flux_is_the_recorded_one():
    assert _reproduced_flux(0.0) == pytest.approx(2.7e-7, rel=0.01)
    # the same flux from the beam's share beyond the plane at the two ends of the disc, less the chords' parts of it
    strip = P * 0.5 * (math.erfc(math.sqrt(2.0) * 2.0 / (W0 * math.hypot(1.0, 95.0 / Z_R)))
                       - math.erfc(math.sqrt(2.0) * 2.0 / (W0 * math.hypot(1.0, 85.0 / Z_R))))
    assert _reproduced_flux(0.0) <= strip <= 1.02 * _reproduced_flux(0.0)


def _flat_plane_flux(w0: float, f0: float, dn: float, s_max: float, power: float = P) -> float:
    """The flux of a round beam of waist ``w0`` at the leg's start across a whole plane, within ``s_max`` of the start along the
    chief ray: the chief ray's signed distance from the plane is ``f0 + dn s``. Across the plane's direction transverse to the ray the
    Gaussian integrates in closed form; along the ray this is a numerical integral of the irradiance times the local flow
    direction's component along the normal, from the beam's own equations."""
    across = math.sqrt(1.0 - dn * dn)
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    s = np.linspace(0.0, s_max, 400001)
    w = w0 * np.sqrt(1.0 + (s / z_r) ** 2)
    rho = -(f0 + dn * s) / across
    kappa = s / (s * s + z_r * z_r)
    along = power * math.sqrt(2.0 / math.pi) / w * np.exp(-2.0 * rho * rho / (w * w)) * np.abs(dn + rho * across * kappa) / across
    return float(np.sum(0.5 * (along[1:] + along[:-1]) * np.diff(s)))


def _no_area_plane(origin=(0.0, 2.0, 0.0), normal=Y):
    return B.observation_plane("T", position=origin, normal=normal)


def test_a_plane_without_an_active_area_counts_the_reach_of_a_laboratory_scene():
    """The plane y = 2 mm, along an escaping 20 um beam, with no active area: within 1000 mm of the start the flux crosses it, by
    independent integration; the bound is the share beyond it there, not half the beam."""
    obs = _obs(run(_tail_scene(0.0, False, radius=None)))
    flux = _flat_plane_flux(W0, -2.0, 0.0, 1000.0)
    assert obs.quantity_state.value == "BOUNDED"
    assert obs.power_upper_bound_W == pytest.approx(P * 0.5 * math.erfc(math.sqrt(2.0) * 2.0 / (W0 * math.hypot(1.0, 1000.0 / Z_R))), rel=1e-9)
    assert obs.power_upper_bound_W == pytest.approx(flux, rel=1e-3)


@pytest.mark.parametrize(("tilt_deg", "within_300"), ((0.1, 4.53e-4), (0.5, 3.64e-4), (1.0, 2.43e-4), (2.0, 8.12e-5)))
def test_a_plane_nearly_parallel_to_an_escaping_beam_behind_the_source_is_not_a_known_zero(tilt_deg: float, within_300: float):
    """A plane with no active area through (-10, 0, 0), a few degrees from the beam: the chief ray crosses it behind the source, and
    the beam's far side crosses it within a few hundred mm. The independent flux within 300 mm is the review's."""
    a = math.radians(tilt_deg)
    assert _flat_plane_flux(0.5, 10.0 * math.sin(a), math.sin(a), 300.0) == pytest.approx(within_300, rel=0.01)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plane = _no_area_plane((-10.0, 0.0, 0.0), (math.sin(a), math.cos(a), 0.0))
    result = run(ProjectSnapshot(components=(laser,), observations=(plane,), targets=(_max(1e-9),)))
    assert _obs(result).quantity_state.value == "BOUNDED"
    assert _obs(result).power_upper_bound_W >= _flat_plane_flux(0.5, 10.0 * math.sin(a), math.sin(a), 1000.0)
    assert result.targets[0].assessment.value == "INDETERMINATE"


def _reach_reading(*parts, origin=(0.0, 2.0, 0.0), turn_deg: float = 0.0) -> tuple:
    """(state, bound) of the plane y = 2 mm along an escaping 20 um beam, the whole design turned about Z."""
    a = math.radians(turn_deg)

    def turned(v):
        return (math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1], v[2])

    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=W0, power_W=P, position=(0.0, 0.0, 0.0), direction=turned(X))
    obs = _obs(run(ProjectSnapshot(components=(laser, *(part(turned) for part in parts)), observations=(_no_area_plane(turned(origin), turned(Y)),))))
    return obs.quantity_state.value, obs.power_upper_bound_W


def test_that_reading_does_not_move_with_the_planes_origin_an_unrelated_part_or_a_turn_of_the_design():
    state, bound = _reach_reading()
    assert state == "BOUNDED"
    for origin in ((10.0, 2.0, 0.0), (50.0, 2.0, 0.0), (90.0, 2.0, 0.0), (300.0, 2.0, 0.0), (3000.0, 2.0, 40.0)):
        moved_state, moved_bound = _reach_reading(origin=origin)
        assert moved_state == state and moved_bound == pytest.approx(bound, rel=1e-9)
    aside = (lambda turned: B.beam_dump("D", position=turned((0.0, 0.0, 1000.0)), axis=turned(Y), radius_mm=5.0))
    assert _reach_reading(aside) == (state, bound)
    for turn in (22.5, 45.0, 77.0):
        turned_state, turned_bound = _reach_reading(turn_deg=turn)
        assert turned_state == state and turned_bound == pytest.approx(bound, rel=1e-9)


def test_a_part_of_the_scene_far_along_the_ray_extends_that_reach():
    """A dump beside the beam 3000 mm along it: the design reaches there, so the plane counts that far too; the flux agrees."""
    far = (lambda turned: B.beam_dump("D", position=turned((3000.0, 300.0, 0.0)), axis=turned(X), radius_mm=5.0))
    _, bound = _reach_reading(far)
    assert bound == pytest.approx(_flat_plane_flux(W0, -2.0, 0.0, 3005.0), rel=1e-2)
    assert bound > 1.2 * _reach_reading()[1]


@pytest.mark.parametrize("offset", (100.0, 500.0))
def test_a_plane_without_an_active_area_far_from_the_beam_stays_a_known_zero_and_says_how_far(offset: float):
    """The other side of the rule: 100 or 500 mm off a 0.5 mm beam nothing crosses within 1000 mm (independent flux below the
    line's level). A line would reach this plane hundreds of metres on; that is not counted, and the plane says so."""
    assert _flat_plane_flux(0.5, -offset, 0.0, 1000.0) < 1e-12
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    result = run(ProjectSnapshot(components=(laser,), observations=(_no_area_plane((0.0, offset, 0.0)),), targets=(_max(1e-6),)))
    obs = _obs(result)
    assert obs.quantity_state.value == "KNOWN_ZERO" and obs.total_power_W == 0.0
    assert result.targets[0].assessment.value == "MET" and result.targets[0].value == 0.0
    assert "within 1000 mm" in obs.message and "state an active area or end the leg on a dump" in obs.message
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert "within 1000 mm" in issue.message


def test_a_dark_plane_with_an_active_area_or_beside_no_escaping_beam_has_no_such_clause():
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    disc = _obs(run(ProjectSnapshot(components=(laser,), observations=(B.observation_plane("T", position=(30.0, 40.0, 0.0), normal=X, radius_mm=2.0),))))
    dumped = _obs(run(ProjectSnapshot(components=(laser, B.beam_dump("D", position=(100.0, 0.0, 0.0), axis=X, radius_mm=20.0)), observations=(_no_area_plane((0.0, 100.0, 0.0)),))))
    assert disc.message == dumped.message == "no light reaches this observation plane: it receives 0 W"


def test_common_planes_without_an_active_area_keep_their_readings_beside_an_arm_that_escapes():
    """A splitter's reflected arm and a fold's incident leg, each with a plane with no active area, and the other beam leaving
    the scene along that plane: the arrival's own power, KNOWN, as closed form (half of 1 mW; 1 mW)."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(-50.0, 0.0, 0.0), direction=X)
    splitter = B.npbs("BS", reflectance=0.5, position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0))
    arm = run(ProjectSnapshot(components=(laser, splitter), observations=(B.observation_plane("T", position=(0.0, 100.0, 0.0), normal=Y),),
                              targets=(_max(6e-4), _min(4e-4))))
    assert _obs(arm).quantity_state.value == "KNOWN" and _obs(arm).total_power_W == pytest.approx(0.5 * P)
    assert [t.assessment.value for t in arm.targets] == ["MET", "MET"]
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0))
    leg = run(ProjectSnapshot(components=(laser, fold), observations=(B.observation_plane("T", position=(-20.0, 0.0, 0.0), normal=X),),
                              targets=(_max(1.1e-3), _min(9e-4))))
    assert _obs(leg).quantity_state.value == "KNOWN" and _obs(leg).total_power_W == pytest.approx(P)
    assert [t.assessment.value for t in leg.targets] == ["MET", "MET"]


def test_a_plane_far_outside_the_far_field_stays_a_known_zero_on_an_escaping_leg():
    """50 mm off a collimated beam (far field 0.5 mrad, its radius 0.5 mm) on T's 50 mm of the leg: far beyond the beam."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plane = B.observation_plane("T", position=(50.0, 50.0, 0.0), normal=Y, radius_mm=50.0)
    behind = B.observation_plane("B", position=(-10.0, 0.0, 0.0), normal=X, radius_mm=50.0)
    tilted = B.observation_plane("R", position=(-10.0, 0.0, 0.0), normal=B.unit((1.0, 1.0, 0.0)), radius_mm=50.0)
    result = run(ProjectSnapshot(components=(laser,), observations=(plane, behind, tilted), targets=(_max(1e-9),)))
    for oid in ("T", "B", "R"):
        obs = _obs(result, oid)
        assert obs.quantity_state.value == "KNOWN_ZERO" and obs.total_power_W == 0.0 and obs.power_complete
    assert result.targets[0].assessment.value == "MET" and "NO_TRACED_ARRIVAL" in result.targets[0].issue_codes


def _surface(sid: str, origin, normal, area: ApertureG | None) -> SurfaceG:
    n = np.asarray(normal, float) / math.sqrt(float(np.dot(normal, normal)))
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, n)
    u = u / math.sqrt(float(u @ u))
    return SurfaceG(surface_id=f"observation/{sid}", owner_id=sid, geometry="plane", origin=np.asarray(origin, float), normal=n, u_axis=u,
                    v_axis=np.cross(n, u), center=np.zeros(3), radius=0.0, cap_axis=n, aperture=ApertureG(), mount_margin=0.0,
                    medium_pos="air", medium_neg="air", interaction=InteractionG("observe", observation_id=sid, active_aperture=area))


@pytest.mark.parametrize("seed", range(120))
def test_the_flow_bound_holds_the_independent_flux_for_random_beams_and_nearly_parallel_discs(seed: int):
    """Round and astigmatic beams with waists on or off the leg, discs off axis at small tilts in any direction, finite legs and
    legs that leave the scene: ``leg_share`` is never below the numerical flux, each crossing counted."""
    rng = np.random.default_rng(seed)
    round_beam = rng.random() < 0.5
    w0x = float(10 ** rng.uniform(-1.8, -0.7))
    w0y = w0x if round_beam else float(10 ** rng.uniform(-1.8, -0.7))
    z_waist = float(rng.choice([0.0, rng.uniform(-30.0, 120.0)]))
    d = np.array([1.0, 0.0, 0.0])
    # (the beam's axes at a random roll about the ray)
    roll = rng.uniform(0.0, 2.0 * math.pi)
    bx, by = np.array([0.0, math.cos(roll), math.sin(roll)]), np.array([0.0, -math.sin(roll), math.cos(roll)])
    g = initial_state(wavelength_nm=LAM, n=1.0, w0x=w0x, w0y=w0y, z_waist=z_waist, m2x=1.0, m2y=1.0, bx=bx, by=by, cfg=CFG)
    assert g.valid
    tilt_deg = float(rng.choice([0.0, 10 ** rng.uniform(-4.0, 0.5)]))
    tilt, phi = math.radians(tilt_deg), rng.uniform(0.0, 2.0 * math.pi)
    across = np.array([0.0, math.cos(phi), math.sin(phi)])
    n = math.sin(tilt) * d + math.cos(tilt) * across
    finite = rng.random() < 0.5
    length = float(rng.uniform(20.0, 200.0)) if finite else math.inf
    offset = float(rng.uniform(0.05, 4.0))
    centre = float(rng.uniform(-10.0, 200.0)) * d + offset * across + float(rng.uniform(-3.0, 3.0)) * np.cross(d, across)
    radius = float(rng.uniform(0.3, 20.0))
    surf = _surface("T", centre, n, ApertureG("circle", p1=radius))
    f0 = float((np.zeros(3) - surf.origin) @ surf.normal)
    dn = float(d @ surf.normal)
    if dn != 0.0 and 0.0 < -f0 / dn < length:  # (the chief ray would cross it on the leg: the other way, it does not)
        surf = _surface("T", centre, -math.sin(tilt) * d + math.cos(tilt) * across, ApertureG("circle", p1=radius))
    beam = D.BeamView("L#0", "L", 0, LAM, np.zeros((1, 3)), d[None, :], np.ones(1, dtype=bool), np.zeros(1, dtype=bool), g, P, False, 1.0,
                      CFG.negligible_power_fraction * P)
    end = (d, length, 0.0, length) if finite else (d, None, 0.0, math.inf)
    leg = D.Leg(np.zeros(3), d, length, d, 0.0, end[0], end[1], 0.0, end[3], g)
    share = D.leg_share(beam, surf, leg, CFG, RULE)
    flux = _flow_through_disc((0, 0, 0), d, g.qx, g.qy, bx, by, surf.origin, surf.normal, radius, s_range=(0.0, length))
    assert share >= flux * (1.0 - 1e-4) - 1e-16
    # (a plane this close to the chief ray's far field is not skipped by the pre-filter while its share is not negligible)
    index = D.SceneIndex.of([surf])
    if bool(D.beyond_reach(index, beam, leg, np.array([0], dtype=np.intp), CFG)[0]):
        assert share <= beam.negligible_W


@pytest.mark.parametrize("seed", range(60))
def test_the_flow_bound_holds_the_flux_after_a_tilted_start_surface_and_before_a_tilted_end_surface(seed: int):
    """A leg from a tilted surface (a fold's reflected leg) to a tilted surface or out of the scene, a disc nearly along it that
    may lie close to the start or the end: the flux counts only the points after the start plane and before the end plane. The
    hard cut makes the quadrature good to about a percent."""
    rng = np.random.default_rng(1000 + seed)
    w0 = float(10 ** rng.uniform(-1.6, -0.5))
    d = np.array([1.0, 0.0, 0.0])
    bx, by = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    g = initial_state(wavelength_nm=LAM, n=1.0, w0x=w0, w0y=w0, z_waist=float(rng.uniform(-40.0, 20.0)), m2x=1.0, m2y=1.0, bx=bx, by=by, cfg=CFG)
    assert g.valid

    def tilted(angle_deg: float) -> np.ndarray:
        a, phi = math.radians(angle_deg), rng.uniform(0.0, 2.0 * math.pi)
        return math.cos(a) * d + math.sin(a) * np.array([0.0, math.cos(phi), math.sin(phi)])

    start_n = tilted(float(rng.uniform(0.0, 50.0)))
    finite = rng.random() < 0.5
    length = float(rng.uniform(5.0, 60.0)) if finite else math.inf
    end_n = tilted(float(rng.uniform(0.0, 50.0)))
    tilt, phi = math.radians(float(rng.choice([0.0, rng.uniform(0.0, 2.0)]))), rng.uniform(0.0, 2.0 * math.pi)
    across = np.array([0.0, math.cos(phi), math.sin(phi)])
    centre = float(rng.uniform(-2.0, 30.0)) * d + float(rng.uniform(0.05, 2.0)) * across
    surf = _surface("T", centre, math.sin(tilt) * d + math.cos(tilt) * across, ApertureG("circle", p1=float(rng.uniform(1.0, 15.0))))
    f0 = float((np.zeros(3) - surf.origin) @ surf.normal)
    dn = float(d @ surf.normal)
    if dn != 0.0 and 0.0 < -f0 / dn < length:
        surf = _surface("T", centre, -math.sin(tilt) * d + math.cos(tilt) * across, surf.interaction.active_aperture)
    beam = D.BeamView("L#0", "L", 0, LAM, np.zeros((1, 3)), d[None, :], np.ones(1, dtype=bool), np.zeros(1, dtype=bool), g, P, False, 1.0,
                      CFG.negligible_power_fraction * P)
    leg = D.Leg(np.zeros(3), d, length, start_n, 0.0, end_n if finite else d, length if finite else None, 0.0, length, g)
    share = D.leg_share(beam, surf, leg, CFG, RULE)
    after = [(np.zeros(3), start_n)] + ([(length * d, -end_n)] if finite else [])
    flux = _flow_through_disc((0, 0, 0), d, g.qx, g.qy, bx, by, surf.origin, surf.normal, surf.interaction.active_aperture.p1, after=after)
    assert share >= 0.97 * flux - 1e-16


# ----------------------------------------------------------------------------------------------- unknown light at plane level
def _ghost_scene(targets=()) -> ProjectSnapshot:
    """An uncoated window tilted 5 deg sends a ghost back beside the laser: T on its leg, off the traced beam."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    t = math.radians(5.0)
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=12.0, material="N-BK7", position=(50.0, 0.0, 0.0), axis=(math.cos(t), math.sin(t), 0.0))
    plane = B.observation_plane("T", position=(20.0, -30.0 * math.tan(math.radians(10.0)), 0.0), normal=X, radius_mm=4.0)
    return ProjectSnapshot(components=(laser, plate), observations=(plane,), targets=tuple(targets))


def test_a_plane_only_followed_light_reaches_is_bounded_from_zero_and_its_power_targets_are_graded_on_the_bound():
    """A plane only a window's ghosts reach: its power is BOUNDED between 0 and the bound of the followed light, with no
    value. A power target on it is graded on [0, bound of the followed light] (targets only): a maximum at or above the bound is MET, a minimum above it is UNMET, a maximum under it is not
    decided. The coherent power's bound is the square of the summed field amplitudes, at least the power's."""
    targets = (_max(1e-3), _max(1e-9), _min(1.0), _max(1e-3, metric="coherent_power_W"), _min(1.0, metric="coherent_power_W"),
               _max(8e-5, metric="coherent_power_W"))
    result = run(_ghost_scene(targets))
    obs = _obs(result)
    assert obs.arrivals == () and obs.total_power_W is None
    assert obs.quantity_state.value == "BOUNDED" and obs.power_lower_bound_W == 0.0 and obs.status.value == "PASS"
    # (the followed light as data: the window's ghosts, each with the face it left the beam at and its power here)
    ghosts = [f for f in obs.followed_light if f.kind == "UNTRACED_REFLECTION"]
    assert ghosts and {f.surface_id for f in ghosts} <= {"W/front", "W/back"}
    assert {f.kind for f in obs.followed_light} <= {"UNTRACED_REFLECTION", "ACTIVE_AREA_TAIL"}
    assert math.fsum(f.power_W for f in obs.followed_light) == pytest.approx(obs.untraced_power_W, rel=1e-12)
    assert obs.untraced_power_W > 1e-6 and obs.power_upper_bound_W >= obs.untraced_power_W  # (evidence, beside the state)
    assert 7e-5 < obs.power_upper_bound_W < 1e-4
    assert [t.assessment.value for t in result.targets] == ["MET", "INDETERMINATE", "UNMET", "MET", "UNMET", "INDETERMINATE"]
    for target in result.targets:
        assert target.value is None
        assert "UNTRACED_LIGHT_BOUNDED" in target.issue_codes
        assert "at most" in target.message
    # (two mutually coherent ghosts may add their fields: up to twice the sum of their powers for equal ones, so a
    # maximum between the power's bound and that is decided for the power only)
    between = run(_ghost_scene((_max(1.2e-4), _max(1.2e-4, metric="coherent_power_W"))))
    assert [t.assessment.value for t in between.targets] == ["MET", "INDETERMINATE"]


def test_a_target_whose_lines_are_dark_on_a_lit_plane_is_indeterminate_too():
    """B lights T; A's traced beam does not, and A's ghost may: A's power on T has no value and lies under the ghost's
    bound, so a maximum above that bound is MET (targets only)."""
    t = math.radians(5.0)
    a = B.laser("A", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=12.0, material="N-BK7", position=(50.0, 0.0, 0.0), axis=(math.cos(t), math.sin(t), 0.0))
    y = -30.0 * math.tan(2.0 * t)
    b = B.laser("B", wavelength_nm=532.0, waist_radius_mm=0.5, power_W=P, position=(20.0, y, -20.0), direction=(0.0, 0.0, 1.0))
    plane = B.observation_plane("T", position=(20.0, y, 0.0), normal=B.unit((1.0, 0.0, 1.0)), radius_mm=4.0)
    result = run(ProjectSnapshot(components=(a, plate, b), observations=(plane,), targets=(
        _max(1e-3, source_id="A"), _min(1e-3, source_id="B", wavelength_nm=None), )))
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["Tpower_Wmax0.001"].assessment.value == "MET"
    assert by_id["Tpower_Wmax0.001"].value is None
    assert by_id["Tpower_Wmin0.001"].value == pytest.approx(P)
    tight = run(ProjectSnapshot(components=(a, plate, b), observations=(plane,), targets=(_max(1e-9, source_id="A"),)))
    assert tight.targets[0].assessment.value == "INDETERMINATE" and tight.targets[0].value is None


def test_a_plane_where_only_light_left_out_as_negligible_may_add_stays_bounded():
    """Nothing followed, a supported interval: BOUNDED, and a target outside it is decided."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=12.0, material="N-BK7", position=(50.0, 0.0, 0.0))
    off = B.observation_plane("T", position=(30.0, 40.0, 0.0), normal=X, radius_mm=2.0)
    result = run(ProjectSnapshot(components=(laser, plate), observations=(off,), targets=(_max(1e-3), _max(1e-16))))
    obs = _obs(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.untraced_power_W == 0.0 and obs.power_upper_bound_W > 0.0
    # (this reading of BOUNDED: the traced zero as total, no lower end, nothing followed, the plane's status UNKNOWN --
    # unlike a plane only followed light reaches, total null, lower 0, status PASS; the results reference states both)
    assert obs.total_power_W == 0.0 and obs.power_lower_bound_W is None and obs.followed_light == ()
    assert obs.status.value == "UNKNOWN"
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["Tpower_Wmax0.001"].assessment.value == "MET"
    assert by_id["Tpower_Wmax1e-16"].assessment.value == "INDETERMINATE"
    from pathlib import Path

    guide = Path(__file__).resolve().parents[1] / "docs" / "engine" / "RESULTS_AND_EXPORT.md"
    text = " ".join(guide.read_text(encoding="utf-8").split())
    assert ("`total_power_W` 0.0 (the traced zero), `power_lower_bound_W` null, `followed_light` empty, "
            "`untraced_power_W` 0, status `UNKNOWN`") in text
    assert "`power_lower_bound_W` 0, status `PASS`" in text
    # (the headless results reference states the same reading)
    assert obs.power_complete is False
    from os_optics.results import basis_of

    entry = basis_of(obs, "total_power_W")
    assert entry.kind.value == "LOWER_BOUND" and [o.part.value for o in entry.omitted] == ["POSSIBLE_RETURN"]
    assert entry.omitted[0].upper == pytest.approx(obs.power_upper_bound_W, rel=1e-12)
    assert "FAINT_LIGHT_BOUNDED" in by_id["Tpower_Wmax1e-16"].issue_codes
    readme = " ".join((Path(__file__).resolve().parents[1] / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    assert ("is the stated truncation of a value while the most it brings to the plane stays within the lines' negligible "
            "level: then it changes no state or status of the plane") in readme
    assert ("a plane with no traced arrival then reads `BOUNDED` [0, `power_upper_bound_W`] with `total_power_W` 0.0 (the "
            "traced zero), `power_lower_bound_W` null, `followed_light` empty, `untraced_power_W` 0, `power_complete` false "
            "and status `UNKNOWN`, the `value_basis` entry of `total_power_W` a `LOWER_BOUND` whose omission is that "
            "light's `POSSIBLE_RETURN` up to the bound; its `power_W` targets name `FAINT_LIGHT_BOUNDED`") in readme
    assert "a plane that only light left out as negligible may reach above its level, from the traced 0 to that light's bound" in readme


def test_a_truly_dark_plane_stays_a_known_zero_with_a_notice():
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plane = B.observation_plane("T", position=(30.0, 40.0, 0.0), normal=X, radius_mm=2.0)
    result = run(ProjectSnapshot(components=(laser,), observations=(plane,), targets=(_max(1e-9), _min(1e-9))))
    assert _obs(result).quantity_state.value == "KNOWN_ZERO" and _obs(result).total_power_W == 0.0
    assert [t.assessment.value for t in result.targets] == ["MET", "UNMET"]
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert issue.status.value == "NOT_APPLICABLE" and _obs(result).status.value == "PASS"


def test_the_shared_contract_accepts_a_dark_bounded_plane_without_a_value():
    from os_optics._vendor.shared_contracts.optical_design.scientific_result import ObservationQuantityV1

    obs = _obs(run(_ghost_scene()))
    record = ObservationQuantityV1(observation_id="T", quantity_state=obs.quantity_state.value, total_power_W=obs.total_power_W,
                                   power_complete=obs.power_complete, untraced_power_W=obs.untraced_power_W,
                                   power_upper_bound_W=obs.power_upper_bound_W)
    assert record.quantity_state == "BOUNDED" and record.total_power_W is None and record.power_upper_bound_W == obs.power_upper_bound_W
    # the run's own record of that plane: the interval is its null value and its upper bound (the lower end 0 is the
    # Engine result's; the record states a lower bound beside a value only)
    from os_optics.local.record import scientific_projection

    (quantity,) = scientific_projection(run(_ghost_scene()), ()).observation_quantities
    assert quantity.quantity_state == "BOUNDED" and quantity.total_power_W is None and quantity.power_lower_bound_W is None
    assert quantity.power_upper_bound_W == obs.power_upper_bound_W and obs.power_lower_bound_W == 0.0


# -------------------------------------------------------------------------------- the record does not depend on targets
def _dark_scene(targets) -> ProjectSnapshot:
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(-10.0, 0.0, 0.0), direction=X)
    plane = B.observation_plane("T", position=(30.0, 40.0, 0.0), normal=X, radius_mm=2.0)
    return ProjectSnapshot(components=(laser,), observations=(plane,), targets=tuple(targets))


def _followed_scene(targets) -> ProjectSnapshot:
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(-10.0, 0.0, 0.0), direction=X)
    plane = B.observation_plane("T", position=(0.1, 0.0, 0.0), normal=X, radius_mm=50.0)
    detector = B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=(Q, Q, 0.0), radius_mm=50.0, absorbing=True)
    return ProjectSnapshot(components=(laser,), observations=(plane, detector), targets=tuple(targets))


def _record(result) -> tuple:
    observations = [o.model_dump(mode="json") for o in result.observations]
    plane_issues = [{k: v for k, v in i.model_dump(mode="json").items() if k != "affected_targets"} for i in result.issues
                    if i.code.value in ("OBSERVATION_NOT_REACHED", "ZERO_POWER")]
    return observations, plane_issues


@pytest.mark.parametrize(("scene", "state", "issue_status", "plane_status"), (
    (_dark_scene, "KNOWN_ZERO", "NOT_APPLICABLE", "PASS"), (_followed_scene, "BOUNDED", "NOT_APPLICABLE", "PASS")))
def test_adding_or_changing_a_target_leaves_the_plane_and_its_issue_identical(scene, state: str, issue_status: str, plane_status: str):
    bare = run(scene(()))
    loose = run(scene((_max(1.0),)))
    strict = run(scene((_min(1.0),)))
    reference = _record(bare)
    assert _record(loose) == reference
    assert _record(strict) == reference
    assert _obs(bare).quantity_state.value == state
    (issue,) = [i for i in bare.issues if i.code.value == "OBSERVATION_NOT_REACHED" and i.location == "observation:T"]
    assert issue.status.value == issue_status
    assert _obs(bare).status.value == plane_status


def test_a_met_target_on_a_dark_plane_is_met_and_the_run_summary_does_not_warn():
    result = run(_dark_scene((_max(1e-9),)))
    assert result.targets[0].assessment.value == "MET" and result.targets[0].status.value == "PASS"
    assert result.status.value == "PASS"  # (the dark plane's notice raises no status)


# ----------------------------------------------------------------------------------------------- D2: the marker
Z50 = (50.0, 0.0, 0.0)


def _source(w: float = 0.5):
    return B.laser("SRC", wavelength_nm=LAM, power_W=P, waist_radius_mm=w, position=(0.0, 0.0, 0.0), direction=X)


def _plane_at(x: float = 50.0, normal=X, **kw):
    kw.setdefault("radius_mm", 50.0)
    return B.observation_plane("T", position=(x, 0.0, 0.0), normal=normal, **kw)


def _fold(**kw):
    return B.mirror("M", position=Z50, normal=(-Q, Q, 0.0), **kw)


def _marker(arrival) -> tuple:
    c = arrival.surface_coincidence
    return None if c is None else (c.surface_id, c.role, c.surface_event)


def test_a_plane_on_a_normal_incidence_mirror_records_the_arriving_and_the_returning_beam():
    mirror = B.mirror("M", position=Z50, normal=(-1.0, 0.0, 0.0), reflectance=0.95)
    result = run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(),), targets=(_min(0.9e-3),)))
    first, second = sorted(arrivals(result, "T"), key=lambda a: a.crossing_index)
    assert (first.beam.path_id, first.crossing_index, _marker(first)) == ("SRC#0", 0, ("M/face", "ARRIVING", "REFLECTIVE"))
    assert (second.beam.path_id, second.crossing_index, _marker(second)) == ("SRC#0", 1, ("M/face", "RETURNING", "REFLECTIVE"))
    assert first.beam.power_W == pytest.approx(P) and second.beam.power_W == pytest.approx(0.95 * P)
    assert first.surface_coincidence.crossing_power_upper_bound_W is None  # (the plane lies in the mirror's face)
    assert second.surface_coincidence.crossing_power_upper_bound_W is None
    (target,) = result.targets
    assert target.assessment.value == "MET" and "SURFACE_COINCIDENT_ARRIVAL" in target.issue_codes
    (issue,) = [i for i in result.issues if i.code.value == "SURFACE_COINCIDENT_ARRIVAL"]
    assert issue.status.value == "NOT_APPLICABLE" and issue.location == "observation:T"
    assert issue.numerical_context["surface_id"] == "M/face" and issue.numerical_context["roles"] == ["ARRIVING", "RETURNING"]


def test_a_plane_on_a_plate_splitter_records_the_return_on_the_child_path():
    plate = B.npbs("BS", reflectance=0.5, position=Z50, normal=(-1.0, 0.0, 0.0))
    result = run(ProjectSnapshot(components=(_source(), plate), observations=(_plane_at(),)))
    by_key = {(a.beam.path_id, a.crossing_index): a for a in arrivals(result, "T")}
    arriving = by_key[("SRC#0", 0)]
    assert _marker(arriving) == ("BS/face", "ARRIVING", "REFLECTIVE") or _marker(arriving)[1:] == ("ARRIVING", "REFLECTIVE")
    returning = next(a for key, a in by_key.items() if key != ("SRC#0", 0))
    assert returning.beam.path_id != "SRC#0" and returning.crossing_index == 0
    assert _marker(returning)[1:] == ("RETURNING", "REFLECTIVE") and returning.beam.power_W == pytest.approx(0.5 * P, rel=1e-3)
    assert len(by_key) == 2


@pytest.mark.parametrize("w", (0.5, 0.02))
def test_a_beam_normal_plane_through_a_fold_keeps_its_arrival_value_and_known_reading(w: float):
    """One arriving arrival of 1 mW: the leaving beam lies along the plane and is no second arrival. Its light in or near the
    plane is not added to the reading (the contract's later per-side report gives it to the plane's other side), however
    strongly it diverges (a 20 um waist)."""
    result = run(ProjectSnapshot(components=(_source(w), _fold()), observations=(_plane_at(),), targets=(_min(0.9e-3),)))
    arrival = one_arrival(result, "T")
    assert _marker(arrival) == ("M/face", "ARRIVING", "REFLECTIVE") and arrival.beam.power_W == pytest.approx(P)
    obs = _obs(result)
    assert obs.quantity_state.value == "KNOWN" and obs.power_complete and obs.total_power_W == pytest.approx(P)
    assert obs.untraced_power_W == 0.0
    (target,) = result.targets
    assert target.assessment.value == "MET" and target.value == pytest.approx(P)
    assert "SURFACE_COINCIDENT_ARRIVAL" in target.issue_codes
    # the literal plane: half of the beam reaches it before the mirror
    assert 0.5 * P <= arrival.surface_coincidence.crossing_power_upper_bound_W <= 0.52 * P


def test_the_literal_bound_holds_the_lines_that_reach_a_tilted_plane_before_a_fold():
    """T tilted 10 deg through a 45 deg fold's hit point on a strongly diverging beam (20 um waist, 50 mm before it). Each energy
    flow line of a Monte Carlo sample of the profile is followed to T and to the mirror on its own curved place in the
    beam: the share that reaches T first is under the bound."""
    tilt = math.radians(10.0)
    normal = (math.cos(tilt), math.sin(tilt), 0.0)
    result = run(ProjectSnapshot(components=(_source(0.02), _fold()), observations=(_plane_at(normal=normal),)))
    bound = one_arrival(result, "T").surface_coincidence.crossing_power_upper_bound_W

    rng = np.random.default_rng(3)
    zeta = rng.normal(0.0, 0.5, 200000)  # (its coordinate across the fold's line in the profile, in units of the 1/e^2 radius)
    w = lambda s: 0.02 * np.sqrt(1.0 + (s / (math.pi * 0.02 * 0.02 / (LAM * 1e-6))) ** 2)  # noqa: E731

    def first_root(f):
        # (the root nearest the hit point on the stretch the K-radii envelope spans, by bisection from both sides of s = 50)
        lo, hi = np.full(zeta.shape, 30.0), np.full(zeta.shape, 70.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = f(mid) * f(lo) > 0.0
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    # the plane: (x - 50) cos + y sin = 0; the mirror: y = x - 50; the line: y = zeta w(x), with x along the chief ray
    at_plane = first_root(lambda s: (s - 50.0) * math.cos(tilt) + zeta * w(s) * math.sin(tilt))
    at_mirror = first_root(lambda s: zeta * w(s) - (s - 50.0))
    share = float(np.mean(at_plane < at_mirror))
    assert share * P <= bound
    assert bound <= 0.6 * P


def test_the_mirror_tilted_so_the_leaving_ray_keeps_the_arrivals_sign_stays_one_known_arrival():
    q = (-Q * math.cos(1e-4) + Q * math.sin(1e-4), Q * math.cos(1e-4) + Q * math.sin(1e-4), 0.0)
    mirror = B.mirror("M", position=Z50, normal=q)
    result = run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(),)))
    assert _marker(one_arrival(result, "T")) == ("M/face", "ARRIVING", "REFLECTIVE")
    assert _obs(result).quantity_state.value == "KNOWN"


def test_a_beam_normal_plane_through_a_45_degree_splitter_keeps_its_reading_on_both_branches():
    split = B.npbs("BS", reflectance=0.5, position=Z50, normal=(-Q, Q, 0.0))
    result = run(ProjectSnapshot(components=(_source(0.02), split), observations=(_plane_at(),)))
    arrival = one_arrival(result, "T")
    assert _marker(arrival)[1:] == ("ARRIVING", "REFLECTIVE") and arrival.beam.power_W == pytest.approx(P)
    assert _obs(result).quantity_state.value == "KNOWN"


def test_the_literal_bound_counts_light_that_passes_beside_a_small_mirror():
    small = _fold(clear_radius_mm=0.5)
    (arrival,) = arrivals(run(ProjectSnapshot(components=(_source(), small), observations=(_plane_at(),))), "T")
    # (half reaches the plane first; of the rest the part beyond the mirror's edge crosses it as well)
    assert arrival.surface_coincidence.crossing_power_upper_bound_W > 0.55 * P


def test_an_absorbing_plane_on_a_mirror_stops_the_beam_there():
    mirror = B.mirror("M", position=Z50, normal=(-1.0, 0.0, 0.0), reflectance=0.95)
    result = run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(absorbing=True),)))
    arrival = one_arrival(result, "T")
    assert _marker(arrival) == ("M/face", "ARRIVING", "ABSORBING") and arrival.surface_coincidence.crossing_power_upper_bound_W is None
    assert [p.termination for p in result.paths] == ["DETECTED"]


def test_a_plane_one_nanometre_before_a_fold_reads_an_interval_and_no_marker():
    """Outside the coincidence window the plane is an ordinary one the mirror takes about half the beam before: [P - S, P]
    with S at least half of it, and a 0.9 mW minimum not decided (it was out of model scope)."""
    result = run(ProjectSnapshot(components=(_source(), _fold()), observations=(_plane_at(50.0 - 1e-6),), targets=(_min(0.9e-3),)))
    arrival = one_arrival(result, "T")
    assert arrival.surface_coincidence is None and arrival.power_out_of_scope_at == ()
    assert arrival.power_taken_before_at == ("M/face",) and arrival.power_taken_before_upper_W >= 0.49 * arrival.beam.power_W
    assert result.targets[0].assessment.value == "INDETERMINATE"
    assert "SURFACE_COINCIDENT_ARRIVAL" not in result.targets[0].issue_codes
    assert not [i for i in result.issues if i.code.value == "SURFACE_COINCIDENT_ARRIVAL"]


def test_a_plane_one_nanometre_before_a_normal_incidence_mirror_has_two_arrivals_and_no_marker():
    mirror = B.mirror("M", position=Z50, normal=(-1.0, 0.0, 0.0), reflectance=0.95)
    result = run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(50.0 - 1e-6),)))
    assert [a.surface_coincidence for a in arrivals(result, "T")] == [None, None]


def test_a_plane_on_a_lens_vertex_is_marked_transmissive():
    lens = B.singlet_lens("L", r1_mm=51.5, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=6.0, material="N-BK7", position=(52.0, 0.0, 0.0))
    result = run(ProjectSnapshot(components=(_source(), lens), observations=(_plane_at(50.0),), targets=(_min(0.9e-3),)))
    arrival = one_arrival(result, "T")
    assert arrival.surface_coincidence is not None
    assert arrival.surface_coincidence.role == "ARRIVING" and arrival.surface_coincidence.surface_event == "TRANSMISSIVE"
    assert arrival.surface_coincidence.crossing_power_upper_bound_W is None  # (not a flat surface)
    assert "SURFACE_COINCIDENT_ARRIVAL" in result.targets[0].issue_codes


def test_the_marker_does_not_change_the_value_the_plane_reads():
    """The arrival equals what a plane just on the incident side reads; the marker is the only difference."""
    mirror = B.mirror("M", position=Z50, normal=(-1.0, 0.0, 0.0), reflectance=0.95)
    at = arrivals(run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(),))), "T")
    before = arrivals(run(ProjectSnapshot(components=(_source(), mirror), observations=(_plane_at(50.0 - 1e-3),))), "T")
    assert [a.beam.power_W for a in at] == pytest.approx([a.beam.power_W for a in before])


# ------------------------------------------------------------------------------------------------- the label
def _two_beams(targets=(), second_nm: float = LAM) -> ProjectSnapshot:
    a = B.laser("A", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X, coherence_group="G")
    b = B.laser("B", wavelength_nm=second_nm, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X, coherence_group="G")
    return ProjectSnapshot(components=(a, b), observations=(_plane_at(),), targets=tuple(targets))


def test_the_ordinary_power_of_two_beams_of_one_wavelength_is_labelled_the_incoherent_sum():
    result = run(_two_beams((_min(1e-3), _min(1.5e-3, source_id="A"))))
    obs = _obs(result)
    assert obs.power_basis == "INCOHERENT_SUM" and obs.superpositions
    assert '"power_basis"' in obs.model_dump_json()
    by_id = {t.target_id: t for t in result.targets}
    summed = by_id["Tpower_Wmin0.001"]
    assert summed.assessment.value == "MET" and summed.value == pytest.approx(2.0 * P)
    assert "POWER_INCOHERENT_SUM" in summed.issue_codes
    assert "sum of the powers" in summed.message and "coherent_power_W" in summed.message
    one = by_id["Tpower_Wmin0.0015"]  # (one beam selected: nothing is summed; it is unmet on its own)
    assert one.assessment.value == "UNMET" and "POWER_INCOHERENT_SUM" not in one.issue_codes


def test_no_label_without_a_coherent_quantity_on_the_plane():
    result = run(_two_beams((_min(1e-3),), second_nm=532.0))
    assert _obs(result).power_basis is None and '"power_basis"' not in _obs(result).model_dump_json()
    assert "POWER_INCOHERENT_SUM" not in result.targets[0].issue_codes
    single = run(ProjectSnapshot(components=(_source(),), observations=(_plane_at(),), targets=(_min(1e-4),)))
    assert _obs(single).power_basis is None and "POWER_INCOHERENT_SUM" not in single.targets[0].issue_codes


def test_the_label_does_not_change_the_assessment():
    unmet = run(_two_beams((_min(3e-3),)))
    assert unmet.targets[0].assessment.value == "UNMET" and "POWER_INCOHERENT_SUM" in unmet.targets[0].issue_codes


# --------------------------------------------------------------- mutually incoherent beams have no fringe at any point
def _incoherent_pair(targets) -> ProjectSnapshot:
    q = math.sqrt(0.5)
    a = B.laser("A", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(-10.0 * q, 10.0 * q, 0.0), direction=(q, -q, 0.0), coherence_group="G1")
    b = B.laser("B", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(-10.0 * q, -10.0 * q, 0.0), direction=(q, q, 0.0), coherence_group="G2")
    return ProjectSnapshot(components=(a, b), observations=(B.observation_plane("T", position=(0.0, 0.0, 0.0), normal=X, radius_mm=50.0),),
                           targets=tuple(targets))


def test_the_local_contrast_of_mutually_incoherent_beams_reads_as_their_visibility():
    targets = tuple(Target(target_id=m, metric=m, observation_id="T", maximum=0.5, source_id="A", source_id_b="B")
                    for m in ("fringe_visibility", "fringe_visibility_peak"))
    result = run(_incoherent_pair(targets))
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["fringe_visibility"].value == 0.0 and by_id["fringe_visibility_peak"].value == 0.0
    assert by_id["fringe_visibility"].assessment.value == by_id["fringe_visibility_peak"].assessment.value == "MET"
    assert "INCOHERENT_SOURCES" in by_id["fringe_visibility_peak"].issue_codes


def test_followed_light_of_their_wavelength_leaves_the_local_contrast_of_incoherent_beams_open():
    """A window behind the plane sends a ghost of each beam back across it: that light may interfere with either beam, and with
    no irradiance formed for the pair the local contrast lies between 0 and 1, at either bound of a maximum."""
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=25.0, material="N-BK7", position=(20.0, 0.0, 0.0))
    base = _incoherent_pair(())
    targets = (Target(target_id="peak", metric="fringe_visibility_peak", observation_id="T", maximum=0.5, source_id="A", source_id_b="B"),)
    result = run(base.model_copy(update={"components": (*base.components, plate), "targets": targets}))
    (target,) = result.targets
    assert target.value == 0.0 and target.assessment.value == "INDETERMINATE"
    assert "the contrast there is between 0 and 1" in target.message


# ------------------------------------------------------------------------ a selector that selects no arrival
@pytest.mark.parametrize("parameters", ({"path_id": "L#0"}, {"path_id": "L#0", "crossing_index": 0}, {"crossing_index": 0}))
@pytest.mark.parametrize("metric", ("power_W", "coherent_power_W"))
def test_a_path_or_crossing_selector_does_not_turn_the_paths_own_light_into_a_known_zero(parameters, metric):
    """The fixture's beam L#0 puts 2.7e-7 W through the disc (independent flux), and the target names that beam."""
    flux = _reproduced_flux(0.0)
    targets = (Target(target_id="low", metric=metric, observation_id="T", maximum=1e-7, unit="W", parameters=parameters),
               Target(target_id="high", metric=metric, observation_id="T", minimum=1e-9, unit="W", parameters=parameters))
    result = run(_tail_scene(0.0, False, targets))
    assert _obs(result).power_upper_bound_W >= flux
    for target in result.targets:
        assert target.assessment.value == "INDETERMINATE" and target.value is None
        assert "NO_TRACED_ARRIVAL" not in target.issue_codes and "UNTRACED_LIGHT_BOUNDED" in target.issue_codes


def test_a_path_selector_on_a_plane_only_a_ghost_reaches_is_not_decided():
    """About 2 x ((n - 1) / (n + 1))^2 of 1 mW, n = 1.511: 8e-5 W of the path's own ghosts reach T."""
    result = run(_ghost_scene((Target(target_id="p", metric="power_W", observation_id="T", maximum=1e-7, unit="W", parameters={"path_id": "L#0"}),)))
    assert _obs(result).untraced_power_W == pytest.approx(2.0 * ((1.511 - 1.0) / (1.511 + 1.0)) ** 2 * P, rel=0.1)
    assert result.targets[0].assessment.value == "INDETERMINATE" and result.targets[0].value is None


def test_a_selector_keeps_its_selected_arrivals_reading_and_another_lines_light_does_not_touch_a_named_path():
    """The exemption stands where the selection holds an arrival: the arrival's power, whatever passes beside it."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=12.0, material="N-BK7", position=(50.0, 0.0, 0.0))
    plane = B.observation_plane("T", position=(20.0, 0.0, 0.0), normal=X, radius_mm=10.0)
    target = Target(target_id="p", metric="power_W", observation_id="T", minimum=0.999e-3, maximum=1.001e-3, unit="W", parameters={"path_id": "L#0"})
    (result,) = run(ProjectSnapshot(components=(laser, plate), observations=(plane,), targets=(target,))).targets
    assert result.value == pytest.approx(P) and result.assessment.value == "MET"


# ----------------------------------------------------------------------------- a plane through a hit with no arrival there
def _fold_error(error: float, targets=()) -> tuple[ProjectSnapshot, float]:
    """A 1 mW, 0.5 mm beam folded at (50, 0, 0), the mirror ``error`` rad off 45 deg. T: the plane x = 50 through the hit point,
    its 20 mm disc 200 mm along it: the arriving chief ray is beside the disc, so no arrival. Returns the scene and the flux of
    the leaving beam through the disc, by the independent integration."""
    a = math.radians(45.0) + error
    n = np.array((-math.sin(a), math.cos(a), 0.0))
    d = np.array((1.0, 0.0, 0.0)) - 2.0 * float(np.array((1.0, 0.0, 0.0)) @ n) * n
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    mirror = B.mirror("M", position=(50.0, 0.0, 0.0), normal=tuple(n), clear_radius_mm=20.0)
    plane = B.observation_plane("T", position=(50.0, 200.0, 0.0), normal=X, radius_mm=20.0)
    z_r = math.pi * 0.25 / (LAM * 1e-6)
    q = complex(50.0, z_r)
    flux = _flow_through_disc((50.0, 0.0, 0.0), d, q, q, np.cross((0.0, 0.0, 1.0), d), (0.0, 0.0, 1.0), (50.0, 200.0, 0.0), X, 20.0,
                              s_range=(0.0, math.inf))
    return ProjectSnapshot(components=(laser, mirror), observations=(plane,), targets=tuple(targets)), flux


@pytest.mark.parametrize(("error", "recorded"), ((-1e-5, 1.18e-6), (-1e-4, 1.17e-5), (-1e-3, 3.56e-5), (-2e-3, 2.16e-6)))
def test_a_plane_through_a_folds_hit_point_with_no_arrival_there_keeps_the_leaving_legs_light(error: float, recorded: float):
    scene, flux = _fold_error(error, (_max(1e-9),))
    assert flux == pytest.approx(recorded, rel=0.05)
    result = run(scene)
    obs = _obs(result)
    assert obs.arrivals == () and obs.quantity_state.value == "BOUNDED" and obs.power_upper_bound_W >= flux
    assert result.targets[0].assessment.value == "INDETERMINATE"


def test_that_plane_stays_dark_with_the_mirror_exactly_at_45_degrees():
    """The beam leaves along the plane 200 mm from nothing: 1.3e-17 W through the disc, below the line's negligible level."""
    scene, flux = _fold_error(0.0, (_max(1e-9),))
    assert flux < 1e-12
    result = run(scene)
    assert _obs(result).arrivals == () and result.targets[0].assessment.value == "MET"


# ---------------------------------------------------------------------- behind a severe hard-edge truncation
def _bessel_j1(v: np.ndarray) -> np.ndarray:
    tau, weights = np.polynomial.legendre.leggauss(400)
    tau = 0.5 * math.pi * (tau + 1.0)
    return np.sum(np.cos(tau[..., None, None] - v[None, ...] * np.sin(tau)[..., None, None]) * (0.5 * weights)[..., None, None], axis=0)


def _airy_flux(a_mm: float) -> float:
    """Fraunhofer: the power of a pinhole of radius ``a_mm`` at x = 50 (a 0.5 mm waist at the origin) into the 20 mm disc through
    (150, 5, 0) with normal +Y. No engine code: the Airy pattern per steradian times the disc's solid angle."""
    k = 2.0 * math.pi / (LAM * 1e-6)
    w = 0.5 * math.hypot(1.0, 50.0 / (math.pi * 0.25 / (LAM * 1e-6)))
    through = P * (1.0 - math.exp(-2.0 * a_mm * a_mm / (w * w)))
    t, weights = np.polynomial.legendre.leggauss(300)
    xs, zs = 20.0 * t, 20.0 * t
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    weight = np.outer(20.0 * weights, 20.0 * weights) * (xx * xx + zz * zz <= 400.0)
    dx, dz = xx + 100.0, zz
    r = np.sqrt(dx * dx + 25.0 + dz * dz)
    v = k * a_mm * np.sqrt(25.0 + dz * dz) / r
    irradiance = through * (k * a_mm) ** 2 / math.pi * (_bessel_j1(v) / v) ** 2
    return float(np.sum(irradiance * (5.0 / r) / (r * r) * weight))


def _pinhole(a_mm: float, *planes, targets=()):
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    stop = B.aperture_stop("S", position=(50.0, 0.0, 0.0), axis=X, radius_mm=a_mm)
    return ProjectSnapshot(components=(laser, stop), observations=tuple(planes), targets=tuple(targets))


@pytest.mark.parametrize("a_mm", (0.3, 0.1))
def test_past_a_severely_truncating_stop_a_side_monitor_is_not_a_known_zero(a_mm: float):
    """A spatial filter: its diffraction into a 20 mm disc beside the beam is about 1e-7 W by the Airy pattern, far above a
    1e-12 W maximum. The beam after the stop has no Gaussian state, and its rays say nothing about that."""
    flux = _airy_flux(a_mm)
    assert 5e-8 < flux < 1e-6
    side = B.observation_plane("T", position=(150.0, 5.0, 0.0), normal=Y, radius_mm=20.0)
    result = run(_pinhole(a_mm, side, targets=(_max(1e-12),)))
    obs = _obs(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_upper_bound_W >= flux
    (target,) = result.targets
    assert target.assessment.value == "INDETERMINATE" and "GAUSSIAN_SEVERE_TRUNCATION" in target.issue_codes
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert issue.numerical_context["untraced_light"] == ["DIFFRACTED_BEAM"]


def test_the_bound_of_that_plane_decides_in_the_direction_it_proves_and_the_documents_say_so():
    """The side monitor reads BOUNDED from 0 to the beam's power (total power null, status PASS, DIFFRACTED_BEAM followed):
    a power maximum at the bound is met, a minimum above it is not, every other limit there is undecided. The Agent guide
    and the results reference state exactly that (not UNKNOWN)."""
    from pathlib import Path

    side = B.observation_plane("T", position=(150.0, 5.0, 0.0), normal=Y, radius_mm=20.0)
    obs = _obs(run(_pinhole(0.1, side)))
    assert (obs.quantity_state.value, obs.status.value, obs.total_power_W, obs.power_lower_bound_W) == ("BOUNDED", "PASS", None, 0.0)
    assert [light.kind for light in obs.followed_light] == ["DIFFRACTED_BEAM"]
    bound = obs.power_upper_bound_W
    targets = (Target(target_id="at", observation_id="T", metric="power_W", maximum=bound),
               Target(target_id="above", observation_id="T", metric="power_W", minimum=1.5 * bound),
               Target(target_id="inside", observation_id="T", metric="power_W", maximum=0.5 * bound),
               Target(target_id="radius", observation_id="T", metric="beam_radius_mm", maximum=10.0))
    graded = {t.target_id: t.assessment.value for t in run(_pinhole(0.1, side, targets=targets)).targets}
    assert graded == {"at": "MET", "above": "UNMET", "inside": "INDETERMINATE", "radius": "INDETERMINATE"}
    repo = Path(__file__).resolve().parents[1]
    for name in ("docs/engine/RESULTS_AND_EXPORT.md",
                 "docs/engine/PHYSICS_SCOPE.md", "docs/engine/KNOWN_LIMITATIONS.md"):
        if not (repo / name).is_file():
            continue
        text = " ".join((repo / name).read_text(encoding="utf-8").split())
        assert "reads `BOUNDED` from 0 to the beam's power" in text, name
        assert "A `power_W` maximum at or above that bound is `MET` and a minimum above it `UNMET`" in text, name
        assert "within that reach on a plane with no active area, reads `UNKNOWN`" not in text, name


#: Sentences of the documents that said a plane reached only by light the probes followed reads UNKNOWN with every target
#: undecided; the run reads it BOUNDED [0, bound] and decides its power targets in the direction the bound proves.
STALE_UNREACHED_PLANE = ("it stays `UNKNOWN`", "plane `UNKNOWN`, a 1e-7 W maximum", "reads `UNKNOWN` when that light may reach the area",
                         "every target on the plane is `INDETERMINATE`, a maximum or a minimum")


def test_the_documents_and_the_schema_state_the_bounded_reading_of_a_plane_only_followed_light_reaches():
    """The tail scene (the reference example of the results reference) reads BOUNDED with status PASS and its partial
    crossing followed; the four documents and the exported schema's QuantityState description say so and none says the
    plane stays UNKNOWN or that its bound decides nothing."""
    from pathlib import Path

    from os_optics.cli import main

    obs = _obs(run(_tail_scene(0.0, False, (_max(1e-7),))))
    assert (obs.quantity_state.value, obs.status.value, obs.total_power_W, obs.power_lower_bound_W) == ("BOUNDED", "PASS", None, 0.0)
    assert [light.kind for light in obs.followed_light] == ["PARTIAL_CROSSING"]
    repo = Path(__file__).resolve().parents[1]
    for name in ("docs/engine/RESULTS_AND_EXPORT.md",
                 "docs/engine/PHYSICS_SCOPE.md", "docs/engine/KNOWN_LIMITATIONS.md"):
        if not (repo / name).is_file():
            continue
        text = " ".join((repo / name).read_text(encoding="utf-8").split())
        for stale in STALE_UNREACHED_PLANE:
            assert stale not in text, (name, stale)
    results = " ".join((repo / "docs/engine/RESULTS_AND_EXPORT.md").read_text(encoding="utf-8").split())
    assert "bound 2.73e-7 W, plane `BOUNDED`, a 1e-7 W maximum `INDETERMINATE`" in results
    assert "* `BOUNDED`, [0, `power_upper_bound_W`]: only light the probe traces followed reaches it" in results
    import contextlib
    import io
    import json

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert main(["schema", "result"]) == 0
    description = " ".join(json.loads(printed.getvalue())["$defs"]["QuantityState"]["description"].split())
    assert "decides nothing" not in description
    assert "a plane no traced beam reaches that only light the trace followed reaches under a supported bound, [0, bound]" in description


def test_that_refusal_leaves_the_lit_plane_a_plane_behind_the_stop_and_the_beam_alone():
    """The plane the beam reaches keeps its reading, a plane behind the stop stays dark, and an absorbing side monitor takes
    nothing out of the beam (aperture diffraction is not propagated)."""
    centre = B.observation_plane("C", position=(150.0, 0.0, 0.0), normal=X, radius_mm=20.0)
    behind = B.observation_plane("B", position=(20.0, 10.0, 0.0), normal=X, radius_mm=2.0)
    side = B.observation_plane("T", position=(100.0, 5.0, 0.0), normal=Y, radius_mm=20.0, absorbing=True)
    result = run(_pinhole(0.1, centre, behind, side))
    arrival = one_arrival(result, "C")
    assert arrival.power_out_of_scope_at == () and _obs(result, "C").quantity_state.value == "KNOWN"
    assert _obs(result, "C").total_power_W == pytest.approx(P * (1.0 - math.exp(-2.0 * 0.01 / (0.5 * math.hypot(1.0, 50.0 / (math.pi * 0.25 / (LAM * 1e-6)))) ** 2)), rel=1e-3)
    assert _obs(result, "B").quantity_state.value == "KNOWN_ZERO"
    assert _obs(result, "T").quantity_state.value == "BOUNDED"


def test_a_ghost_a_rim_truncates_keeps_the_rays_envelope():
    """Two uncoated singlets before crossed polarizers: the detector behind them is reached by no light above the level. Ghosts a
    lens's rim cuts severely carry about 1e-11 W; they do not refuse it (they are followed only to bound what may arrive)."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=1.0, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    comps = [laser, *(B.singlet_lens(f"L{i}", r1_mm=60.0, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7",
                                     position=(20.0 + 25.0 * i, 0.0, 0.0)) for i in range(2)),
             B.polarizer("P1", position=(70.0, 0.0, 0.0), axis=X, angle_deg=0.0), B.polarizer("P2", position=(80.0, 0.0, 0.0), axis=X, angle_deg=90.0)]
    detector = B.observation_plane("T", position=(110.0, 0.0, 0.0), normal=X, radius_mm=12.0, absorbing=True)
    result = run(ProjectSnapshot(components=tuple(comps), observations=(detector,), targets=(_max(1e-9),)))
    assert _obs(result).quantity_state.value == "BOUNDED" and result.targets[0].assessment.value == "MET"


def test_a_mild_truncation_keeps_its_gaussian_state_and_a_side_monitor_stays_dark():
    """The stop cuts the tail of the beam at 3 radii: the Gaussian state stays, aperture diffraction is not propagated."""
    side = B.observation_plane("T", position=(150.0, 5.0, 0.0), normal=Y, radius_mm=20.0)
    result = run(_pinhole(1.5, side, targets=(_max(1e-12),)))
    assert _obs(result).quantity_state.value == "KNOWN_ZERO"
    assert result.targets[0].assessment.value == "MET"


# ------------------------------------------------------------------------------- a beam metric on an UNKNOWN plane
@pytest.mark.parametrize(("metric", "limits"), (("beam_radius_x_mm", {"maximum": 100.0}), ("active_area_power_fraction", {"minimum": 0.5})))
@pytest.mark.parametrize("scene", ("tail", "ghost"))
def test_a_beam_metric_on_an_unknown_plane_is_not_decided(metric, limits, scene):
    target = Target(target_id="m", metric=metric, observation_id="T", **limits)
    result = run(_tail_scene(0.0, False, (target,)) if scene == "tail" else _ghost_scene((target,)))
    assert _obs(result).quantity_state.value == "BOUNDED"
    (read,) = result.targets
    assert read.assessment.value == "INDETERMINATE" and read.value is None
    assert "UNTRACED_LIGHT_BOUNDED" in read.issue_codes and "OBSERVATION_NOT_REACHED" not in read.issue_codes


def test_a_power_target_is_undecided_only_inside_the_interval_of_a_plane_no_traced_beam_reaches():
    """A plane with no traced arrival that followed light may reach states no power; a power target on it is graded on
    [0, bound] (targets only): a maximum above the bound is MET, a
    minimum inside the interval is not decided. A lit plane keeps its own decisions on its interval, and a plane nothing
    reaches keeps its known zero: a maximum MET, a minimum UNMET."""
    undecided = run(_ghost_scene((_max(1.0, oid="T"), _min(1e-12, oid="T"))))
    assert [t.assessment.value for t in undecided.targets] == ["MET", "INDETERMINATE"]
    assert _obs(undecided).quantity_state.value == "BOUNDED" and _obs(undecided).total_power_W is None
    lit = run(_lit_with_ghost((_max(1.0), _min(1e-6))))
    assert _obs(lit).arrivals and [t.assessment.value for t in lit.targets] == ["MET", "MET"]
    dark = run(_dark_scene((_max(1.0, oid="T"), _min(1e-12, oid="T"))))
    assert _obs(dark).quantity_state.value == "KNOWN_ZERO" and [t.assessment.value for t in dark.targets] == ["MET", "UNMET"]


def test_a_beam_metric_on_a_dark_plane_still_fails_for_want_of_a_beam():
    result = run(_dark_scene((Target(target_id="m", metric="beam_radius_x_mm", observation_id="T", maximum=100.0),)))
    assert result.targets[0].assessment.value == "UNMET" and "OBSERVATION_NOT_REACHED" in result.targets[0].issue_codes


@pytest.mark.parametrize("metric", ("fringe_visibility", "beam_crossing_angle_deg", "centroid_separation_mm"))
def test_a_pair_metric_on_an_unknown_plane_is_not_decided_and_on_a_dark_plane_fails(metric: str):
    """Two ghosts cross the plane, and could form the pair: not decided. Where nothing reaches the plane there is no pair."""
    target = Target(target_id="m", metric=metric, observation_id="T", maximum=1000.0)
    ghosts = run(_ghost_scene((target,)))
    assert _obs(ghosts).quantity_state.value == "BOUNDED"
    (read,) = ghosts.targets
    assert read.assessment.value == "INDETERMINATE" and "UNTRACED_LIGHT_BOUNDED" in read.issue_codes
    assert "OBSERVATION_NOT_REACHED" not in read.issue_codes
    (dark,) = run(_dark_scene((target,))).targets
    assert dark.assessment.value == "UNMET" and "OBSERVATION_NOT_REACHED" in dark.issue_codes


def test_a_dark_plane_only_unbounded_faint_light_may_reach_has_no_value_and_decides_nothing():
    """An uncoated lens and a disc off the beam: nothing followed reaches it, and the surfaces facing each other leave the light
    left out as negligible with no finite bound. Not known to be dark: no 0 W beside UNKNOWN, and no UNMET for want of a beam."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    lens = B.singlet_lens("L1", r1_mm=51.5, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=12.0, material="N-BK7", position=(52.0, 0.0, 0.0))
    plane = B.observation_plane("T", position=(30.0, 40.0, 0.0), normal=X, radius_mm=2.0)
    targets = (_max(1e-3), Target(target_id="m", metric="beam_radius_x_mm", observation_id="T", maximum=100.0))
    result = run(ProjectSnapshot(components=(laser, lens), observations=(plane,), targets=targets))
    obs = _obs(result)
    assert obs.quantity_state.value == "UNKNOWN" and obs.power_upper_bound_W is None and obs.untraced_power_W == 0.0
    assert obs.total_power_W is None and [s.power_W for s in obs.power_by_wavelength] == [None]
    for target in result.targets:
        assert target.assessment.value == "INDETERMINATE" and "FAINT_LIGHT_BOUNDED" in target.issue_codes


def test_a_diffracted_plane_is_told_what_makes_it_readable():
    side = B.observation_plane("T", position=(150.0, 5.0, 0.0), normal=Y, radius_mm=20.0)
    result = run(_pinhole(0.1, side))
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert "open the aperture" in issue.suggested_action and "max_paths" not in issue.suggested_action


# -------------------------------------------------------------------------------- a plane's issues and the threshold
def _lit_with_ghost(targets) -> ProjectSnapshot:
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plate = B.window("W", thickness_mm=3.0, clear_radius_mm=12.0, material="N-BK7", position=(50.0, 0.0, 0.0))
    return ProjectSnapshot(components=(laser, plate), observations=(B.observation_plane("T", position=(20.0, 0.0, 0.0), normal=X, radius_mm=10.0),),
                           targets=tuple(targets))


def test_a_threshold_adds_no_issue_to_a_lit_planes_record():
    """A window's ghost crosses the lit plane: the limit 1.0001 mW lies inside the plane's interval, 1 W above it. What the limit
    leaves undecided is the target's; the plane's issues and compact codes are the same with or without it."""
    bare, wide, tight = (run(_lit_with_ghost(targets)) for targets in ((), (_max(1.0),), (_max(1.0001e-3),)))
    plane_issues = [[i.model_dump(mode="json") for i in r.issues if i.location == "observation:T"] for r in (bare, wide, tight)]
    assert plane_issues[0] == plane_issues[1] == plane_issues[2]
    assert [o.model_dump(mode="json") for o in bare.observations] == [o.model_dump(mode="json") for o in tight.observations]
    assert tight.targets[0].assessment.value == "INDETERMINATE"
    (issue,) = [i for i in tight.issues if i.code.value == "INCOMPLETE_TRACE"]
    assert issue.location == f"target:{tight.targets[0].target_id}" and issue.affected_targets == (tight.targets[0].target_id,)
    assert issue.numerical_context["observation_id"] == "T"
    assert not [i for i in wide.issues if i.code.value == "INCOMPLETE_TRACE"]


# ----------------------------------------------------------------------------- what the chief ray does at the surface
def _face(interaction, medium_negative: str = "air", medium_positive: str = "air"):
    from os_optics.contracts.geometry import Pose3D
    from os_optics.contracts.optics import CircularAperture, PlaneSurface
    from os_optics.contracts.project import EngineeringComponentSnapshot

    surface = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=20.0),
                           medium_positive=medium_positive, medium_negative=medium_negative, interaction=interaction)
    return EngineeringComponentSnapshot(instance_id="F", role="window", pose=Pose3D.from_axis_direction((50.0, 0.0, 0.0), (Q, -Q, 0.0)), surfaces=(surface,))


def test_a_stops_plate_that_takes_the_chief_ray_is_absorbing_at_the_hit():
    """A beam 1 mm off a 0.2 mm hole: the plate stops its chief ray, and the plane through that hit records the arrival."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 1.0, 0.0), direction=X)
    stop = B.aperture_stop("S", position=(50.0, 0.0, 0.0), axis=X, radius_mm=0.2)
    plane = B.observation_plane("T", position=(50.0, 1.0, 0.0), normal=X, radius_mm=5.0)
    result = run(ProjectSnapshot(components=(laser, stop), observations=(plane,)))
    assert [p.termination for p in result.paths] == ["CLIPPED"]
    assert one_arrival(result, "T").surface_coincidence.surface_event == "ABSORBING"


def test_total_internal_reflection_is_reflective_at_the_hit():
    """A beam in N-BK7 at 45 deg on a glass/air face, past the critical angle (41.4 deg): all of it reflects."""
    from os_optics.contracts.optics import RefractInteraction

    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X, medium="N-BK7")
    plane = B.observation_plane("T", position=(50.0, 0.0, 0.0), normal=X, radius_mm=20.0, medium="N-BK7")
    result = run(ProjectSnapshot(components=(laser, _face(RefractInteraction(), medium_negative="N-BK7")), observations=(plane,)))
    assert _marker(one_arrival(result, "T")) == ("F/face", "ARRIVING", "REFLECTIVE")


@pytest.mark.parametrize(("reflectance", "event"), ((0.9, "REFLECTIVE"), (0.01, "TRANSMISSIVE")))
def test_a_coated_refracting_face_is_read_by_what_it_does(reflectance: float, event: str):
    from os_optics.contracts.materials import ConstantCoating
    from os_optics.contracts.optics import RefractInteraction

    coating = ConstantCoating(reflectance_s=reflectance, reflectance_p=reflectance, transmittance_s=1.0 - reflectance, transmittance_p=1.0 - reflectance)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    plane = B.observation_plane("T", position=(50.0, 0.0, 0.0), normal=X, radius_mm=20.0)
    face = _face(RefractInteraction(coating=coating), medium_positive="N-BK7")
    result = run(ProjectSnapshot(components=(laser, face), observations=(plane,)))
    assert _marker(one_arrival(result, "T")) == ("F/face", "ARRIVING", event)


# ------------------------------------------------------------------------------------- the literal bound's limit
def test_no_literal_bound_where_the_beam_starts_partly_behind_the_surface():
    """A 7.4 mm beam, a mirror 78.9 deg from it 31 mm from the source: within 3 radii the mirror's plane reaches back past the
    source (3 x 7.4 x tan 78.9 > 31), so part of the beam starts behind it, on neither side of the half-plane."""
    a = math.radians(78.9)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=7.4, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    mirror = B.mirror("M", position=(31.0, 0.0, 0.0), normal=(-math.cos(a), math.sin(a), 0.0), clear_radius_mm=300.0)
    plane = B.observation_plane("T", position=(31.0, 0.0, 0.0), normal=X, radius_mm=800.0)
    arrival = one_arrival(run(ProjectSnapshot(components=(laser, mirror), observations=(plane,))), "T")
    assert arrival.surface_coincidence.role == "ARRIVING" and arrival.surface_coincidence.crossing_power_upper_bound_W is None


# ---------------------------------------------------------------------------------- an active area off the leg
def test_an_active_area_wholly_before_the_source_receives_nothing_from_the_leg():
    """A rectangle behind the source on a plane the rays' picture crosses in front: no point of it lies on the leg."""
    a = math.radians(2.0)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.02, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    dump = B.beam_dump("D", position=(100.0, 0.0, 0.0), axis=X, radius_mm=20.0)
    plane = B.observation_plane("T", position=(-20.0, 0.0, 0.0), normal=(math.sin(a), math.cos(a), 0.0), half_width_mm=5.0, half_height_mm=5.0)
    result = run(ProjectSnapshot(components=(laser, dump), observations=(plane,), targets=(_max(1e-12),)))
    assert _obs(result).quantity_state.value == "KNOWN_ZERO"
    assert result.targets[0].assessment.value == "MET"
