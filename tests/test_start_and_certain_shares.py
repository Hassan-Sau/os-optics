"""The share a reflection the leg began with sends beyond a plane, the energy flow of the share the next surface takes
first, and the share a stopping surface certainly takes before a plane (an interval, not a refusal).

* A monitor just after a fold mirror: part of the reflected beam leaves the mirror beyond the monitor's plane and never
  crosses it. That share is bounded (``decisions.start_flow_share``: the beam's energy-flow lines at their crossing of the
  mirror's plane, behind the chief ray's start too), so the plane reads the beam within ``[P - S, P]``.
* The surface the chief ray meets next: its bound also holds the energy-flow lines that meet it at chief-ray parameters
  beyond the plane's chief crossing (a tilted plane).
* The back of an uncoated mirror or an absorber the beam meets first stops that light: the lines that certainly meet it
  first before the plane bound the plane's power from above, ``P - L`` (``lower_share.certain_meeting_share``), so a
  minimum the beam provably misses is unmet.

The references are this module's own: the paraxial Gaussian beam's energy-flow lines (a line at ``(xi, eta)`` of the
profile, density ``exp(-2 (xi^2 + eta^2))``, lies ``xi w(s), eta w(s)`` off the chief ray) followed to their own crossings of
the surfaces (fixed-point iteration, bisection), the set of lines per profile row found by bisection and its Gaussian mass
integrated exactly; or the closed form of a collimated beam."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import run
from os_optics import builders as B
from os_optics.contracts import (CircularAperture, EngineeringComponentSnapshot, MirrorInteraction, Pose3D, ProjectSnapshot,
                                      SagSurface, SphereSurface, Target)

LAM = 780.0
SOURCE_W = 1e-3
LAUNCH_MM = 30.0
X = (1.0, 0.0, 0.0)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------------------------- the reference's lines
def _rows_share(indicator, rows: int = 241, cols: int = 321) -> float:
    """The Gaussian mass of the lines ``(xi, eta)`` with ``indicator(xi, eta)``: per ``eta`` row (trapezoid over +-6.5) the
    indicator's changes on a grid of ``xi`` over +-8, each refined by bisection, the mass between them exact (``erf``)."""
    etas = np.linspace(-6.5, 6.5, rows)
    xs = np.linspace(-8.0, 8.0, cols)
    xi, eta = np.meshgrid(xs, etas, indexing="xy")
    ind = indicator(xi.ravel(), eta.ravel()).reshape(xi.shape)
    ri, rj = np.nonzero(ind[:, :-1] != ind[:, 1:])
    a, b, e, ia = xs[rj].copy(), xs[rj + 1].copy(), etas[ri], ind[ri, rj].copy()
    for _ in range(45):
        m = 0.5 * (a + b)
        same = indicator(m, e) == ia if len(m) else np.zeros(0, dtype=bool)
        a, b = np.where(same, m, a), np.where(same, b, m)
    roots = 0.5 * (a + b)
    mass = np.zeros(len(etas))
    for i in range(len(etas)):
        edges = [-math.inf, *np.sort(roots[ri == i]).tolist(), math.inf]
        inside = bool(ind[i, 0])
        for lo, hi in zip(edges[:-1], edges[1:]):
            if inside:
                mass[i] += 0.5 * ((math.erf(math.sqrt(2.0) * hi) if math.isfinite(hi) else 1.0)
                                  - (math.erf(math.sqrt(2.0) * lo) if math.isfinite(lo) else -1.0))
            inside = not inside
    density = np.exp(-2.0 * etas * etas) * math.sqrt(2.0 / math.pi)
    return float(np.sum(mass * density) * (etas[1] - etas[0]))


def _width(w0: float, waist: float):
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    return lambda s: w0 * np.sqrt(1.0 + ((s - waist) / z_r) ** 2)


def _plane_hit(src, d, e1, e2, width, xi, eta, normal, point) -> np.ndarray:
    """Each line's chief-ray parameter at its crossing of the plane through ``point`` with ``normal`` (fixed point)."""
    n = np.asarray(normal, float)
    across = xi * float(e1 @ n) + eta * float(e2 @ n)
    s = np.full(xi.shape, float((np.asarray(point) - src) @ n) / float(d @ n))
    for _ in range(400):
        s_new = (float((np.asarray(point) - src) @ n) - width(s) * across) / float(d @ n)
        if np.max(np.abs(s_new - s), initial=0.0) < 1e-14:
            return s_new
        s = s_new
    return s


def _at(src, d, e1, e2, width, s, xi, eta) -> np.ndarray:
    return src + s[..., None] * d + width(s)[..., None] * (xi[..., None] * e1 + eta[..., None] * e2)


# ------------------------------------------------------------------------------------------ a monitor after a fold
def _after_fold(mirror_n, w0: float, waist: float, t_point, t_normal, targets=(), sphere_r: float | None = None) -> ProjectSnapshot:
    if sphere_r is None:
        mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=tuple(float(v) for v in mirror_n), reflectance=0.99, clear_radius_mm=20.0)
    else:
        mirror = B.spherical_mirror("M", position=(0.0, 0.0, 0.0), normal=tuple(float(v) for v in mirror_n), radius_of_curvature_mm=sphere_r,
                                    clear_radius_mm=12.0)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0, power_W=SOURCE_W, position=(-LAUNCH_MM, 0.0, 0.0), direction=X,
                    waist_position_z_mm=waist)
    t = B.observation_plane("T", position=tuple(float(v) for v in t_point), normal=tuple(float(v) for v in t_normal), radius_mm=60.0)
    return ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=targets)


def _beyond_share(mirror_n, w0: float, waist: float, t_point, t_normal, sphere_r: float | None = None) -> float:
    """The share of the incident beam's lines whose point on the mirror lies beyond T (they leave it beyond the plane)."""
    src, d = np.array([-LAUNCH_MM, 0.0, 0.0]), np.array(X)
    e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    width, n = _width(w0, waist), np.asarray(mirror_n, float)

    def beyond(xi, eta):
        if sphere_r is None:
            s = _plane_hit(src, d, e1, e2, width, xi, eta, n, (0.0, 0.0, 0.0))
        else:  # (the cap near the vertex: bisection on the sign change of |p - C|^2 - R^2 within 6 mm of the vertex)
            centre = sphere_r * n
            lo, hi = np.full(xi.shape, LAUNCH_MM - 6.0), np.full(xi.shape, LAUNCH_MM + 6.0)

            def f(s):
                q = _at(src, d, e1, e2, width, s, xi, eta) - centre
                return np.sum(q * q, axis=-1) - sphere_r * sphere_r

            f_lo = f(lo)
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                left = np.sign(fm) != np.sign(f_lo)
                hi, lo, f_lo = np.where(left, mid, hi), np.where(left, lo, mid), np.where(left, f_lo, fm)
            s = 0.5 * (lo + hi)
        return (_at(src, d, e1, e2, width, s, xi, eta) - np.asarray(t_point)) @ np.asarray(t_normal) > 0.0

    return _rows_share(beyond, rows=121 if sphere_r is not None else 241, cols=161 if sphere_r is not None else 321)


def _reflected(result):
    (obs,) = result.observations
    return next(a for a in obs.arrivals if any(e.startswith("M/") and "reflect" in e for e in a.lineage))


@pytest.mark.parametrize("gap", (0.1, 0.5, 1.0))
def test_a_monitor_just_after_a_fold_reads_the_beam_within_the_share_the_mirror_sends_beyond_it(gap: float):
    """A collimated 0.5 mm beam, a 45 deg fold, T across the reflected leg ``gap`` after the hit: the mirror sends
    0.5 erfc(sqrt(2) gap / w) of the beam beyond T. That share is within the bound, which is tight, the plane is BOUNDED
    (it was OUT_OF_MODEL_SCOPE), and a minimum below the lower end is met."""
    q = math.sqrt(0.5)
    target = Target(target_id="t", metric="power_W", observation_id="T", minimum=0.99e-3 * (1.0 - 0.5 * math.erfc(math.sqrt(2.0) * gap / 0.5)) * 0.99,
                    unit="W")
    result = run(_after_fold((-q, q, 0.0), 0.5, LAUNCH_MM, (0.0, gap, 0.0), (0.0, 1.0, 0.0), (target,)))
    arrival = _reflected(result)
    share = 0.5 * math.erfc(math.sqrt(2.0) * gap / 0.5) * arrival.beam.power_W
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("M/face",)
    assert share <= arrival.power_taken_before_upper_W * (1.0 + 1e-9)
    assert arrival.power_taken_before_upper_W <= 1.001 * share + 1e-15
    assert result.observations[0].quantity_state.value == "BOUNDED"
    assert result.targets[0].assessment.value == "MET"


def test_the_start_bound_holds_the_energy_flow_share_behind_the_chief_rays_start():
    """24 seeded poses: folds 10-80 deg (the mirror tilted out of plane too), waists 20 um-1 mm from 20 mm before the source
    to 15 mm past the mirror (beams converging onto the mirror: their lines behind the chief ray's start are wider), gaps
    0.01-4 mm, T tilted up to 72 deg. The reference share is never above the bound, a plane without a bound has a share at
    most the line's negligible level, and the bound is within a few per cent of it where it decides."""
    rng = np.random.default_rng(3741)
    bounded = 0
    for _ in range(24):
        fold = math.radians(float(rng.uniform(10.0, 80.0)))
        mirror_n = _unit((-math.cos(fold), math.sin(fold), float(rng.uniform(-0.1, 0.1))))
        d_out = np.array(X) - 2.0 * float(np.array(X) @ mirror_n) * mirror_n
        w0 = float(np.exp(rng.uniform(math.log(0.02), math.log(1.0))))
        waist = float(rng.uniform(-20.0, 45.0))
        gap = float(np.exp(rng.uniform(math.log(0.01), math.log(4.0))))
        t_normal = _unit(d_out + rng.normal(size=3) * float(rng.uniform(0.0, 0.6)))
        if float(t_normal @ d_out) < 0.3:
            t_normal = d_out
        arrival = _reflected(run(_after_fold(mirror_n, w0, waist, gap * d_out, t_normal)))
        truth = _beyond_share(mirror_n, w0, waist, gap * d_out, t_normal) * arrival.beam.power_W
        if arrival.power_taken_before_upper_W is not None:
            bounded += 1
            assert truth <= arrival.power_taken_before_upper_W * (1.0 + 1e-9) + 1e-18, (fold, w0, waist, gap)
        else:
            assert arrival.power_out_of_scope_at == () and truth <= 1e-9 * SOURCE_W, (fold, w0, waist, gap)
    assert bounded >= 12


def test_a_spherical_fold_mirror_is_bounded_by_its_tangent_plane():
    """A concave R 75 mm mirror at 26.5 deg incidence, T 0.21 mm after the hit: the plane across the ray that stands for the
    cap reaches beyond T (a bound of the whole beam); the tangent plane at the hit with the cap's departure from it bounds the
    share within a few times the reference (1.5e-5 W)."""
    fold = math.radians(26.5)
    mirror_n = _unit((-math.cos(fold), math.sin(fold), 0.0))
    d_out = np.array(X) - 2.0 * float(np.array(X) @ mirror_n) * mirror_n
    arrival = _reflected(run(_after_fold(mirror_n, 0.457, 12.96, 0.208 * d_out, d_out, sphere_r=75.0)))
    truth = _beyond_share(mirror_n, 0.457, 12.96, 0.208 * d_out, d_out, sphere_r=75.0) * arrival.beam.power_W
    assert arrival.power_taken_before_at == ("M/face",)
    assert truth <= arrival.power_taken_before_upper_W <= 10.0 * truth


def test_without_a_gaussian_bound_the_pupil_rays_that_leave_the_mirror_beyond_the_plane_refuse_it():
    """A convex R 176 mm mirror at 66 deg incidence (the Gaussian state is not valid there, ``GAUSSIAN_LARGE_AOI``), T
    0.014 mm after the hit: 47 % of the beam leaves the mirror beyond T, and pupil rays do. The plane's power was read as
    the whole beam; it is out of model scope at the mirror."""
    rng = np.random.default_rng(5)
    fold = math.radians(66.35020466217662)
    mirror_n = _unit((-math.cos(fold), math.sin(fold), 0.0))
    d_out = np.array(X) - 2.0 * float(np.array(X) @ mirror_n) * mirror_n
    del rng
    result = run(_after_fold(mirror_n, 0.15, -1.42, 0.0138 * d_out, d_out, sphere_r=-176.48))
    arrival = _reflected(result)
    assert arrival.power_out_of_scope_at == ("M/face",)
    assert arrival.power_taken_before_upper_W is None


# ---------------------------------------------------------------------- the next surface before a tilted plane
def test_the_next_surface_bound_holds_lines_that_meet_it_beyond_the_planes_chief_crossing():
    """A beam dump tilted 34.6 deg, T 0.47 mm before it and tilted 16 deg: lines of the diverging beam meet the dump at
    chief-ray parameters beyond T's chief crossing, where the beam is wider. The parallel rays at the widest radius up to that
    crossing gave 2.50906e-5 W against the lines' 2.50913e-5 W; the energy flow's bound holds it."""
    w0, waist, tilt, gap = 0.4879701652864531, 16.44617269930908, math.radians(34.63462449883081), 0.4655528869071869
    n_dump = np.array([-math.cos(tilt), math.sin(tilt), 0.0])
    t_normal = _unit(np.array(X) + np.array([-0.04908865, 0.26556031, 0.08407139]))
    src, d = np.array([-40.0, 0.0, 0.0]), np.array(X)
    e1, e2 = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    width = _width(w0, waist)
    t_point = np.array([-gap, 0.0, 0.0])

    def first(xi, eta):  # (the dump's disc of radius 12 mm before the line's own crossing of T)
        s_d = _plane_hit(src, d, e1, e2, width, xi, eta, n_dump, (0.0, 0.0, 0.0))
        s_t = _plane_hit(src, d, e1, e2, width, xi, eta, t_normal, t_point)
        p = _at(src, d, e1, e2, width, s_d, xi, eta)
        return (s_d < s_t) & (np.sum(p * p, axis=-1) <= 144.0)

    truth = _rows_share(first) * SOURCE_W
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=w0, position=tuple(src), direction=X, waist_position_z_mm=waist)
    dump = B.beam_dump("M", position=(0.0, 0.0, 0.0), axis=tuple(n_dump), radius_mm=12.0)
    t = B.observation_plane("T", position=tuple(t_point), normal=tuple(t_normal), radius_mm=10.0)
    (obs,) = run(ProjectSnapshot(components=(laser, dump), observations=(t,))).observations
    (arrival,) = obs.arrivals
    assert truth == pytest.approx(2.50913e-5, rel=1e-4)
    assert truth <= arrival.power_taken_before_upper_W
    assert arrival.power_taken_before_lower_W <= truth  # (the dump stops what it meets: a certain share)


# ---------------------------------------------------------------------------------------- a stopping surface met first
def _rim_x(kind: str) -> float:
    """The bowl's near rim: x = -26.5 mm on the sag bowls' support, R sin(2.85 deg - 34.85 deg) on the spherical cap."""
    return 50.0 * math.sin(math.radians(2.85 - 34.85)) if kind == "sphere" else -26.5


def _bowl(kind: str, delta: float, target_min: float, hole: tuple[float, float] | None = None) -> ProjectSnapshot:
    """A concave R 50 mm bowl (vertex at the origin, axis +Z) reflecting on its inside, its back uncoated (blocking); a
    0.3 mm beam along +X ``delta`` above the near rim (x = -26.5 mm) lands on the far side; T normal +X at x = 0. ``hole``:
    a circular hole (centre x, radius) in a sag bowl's support."""
    radius, rim_x = 50.0, _rim_x(kind)
    height = radius - math.sqrt(radius * radius - rim_x * rim_x) + delta
    if kind == "sphere":
        tilt, half = math.radians(2.85), math.radians(34.85)
        face = SphereSurface(surface_id="S1", center_mm=(0.0, 0.0, radius), radius_mm=radius, cap_axis=(math.sin(tilt), 0.0, -math.cos(tilt)),
                             aperture_radius_mm=radius * math.sin(half), interaction=MirrorInteraction(reflective_side="negative"))
    else:
        face = SagSurface(surface_id="S1", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=X, curvature_per_mm=1.0 / radius,
                          cylindrical=kind == "cylinder", aperture=CircularAperture(radius_mm=28.5, center_offset_mm=(rim_x + 28.5, 0.0)),
                          **({} if hole is None else {"holes": (CircularAperture(radius_mm=hole[1], center_offset_mm=(hole[0], 0.0)),)}),
                          interaction=MirrorInteraction(reflective_side="positive"))
    mirror = EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D(), surfaces=(face,))
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=0.3, position=(-60.0, 0.0, height), direction=X)
    t = B.observation_plane("T", position=(0.0, 0.0, height), normal=X, radius_mm=10.0)
    return ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=(
        Target(target_id="t", metric="power_W", observation_id="T", minimum=target_min, unit="W"),))


def _bowl_share(kind: str, delta: float, hole: tuple[float, float] | None = None) -> float:
    """The share of the lines that meet the bowl's back (outside its sphere / cylinder) on its support (less ``hole``)
    before x = 0."""
    radius, rim_x = 50.0, _rim_x(kind)
    height = radius - math.sqrt(radius * radius - rim_x * rim_x) + delta
    src, d = np.array([-60.0, 0.0, height]), np.array(X)
    e1, e2 = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    width = _width(0.3, 0.0)
    centre = np.array([0.0, 0.0, radius])
    tilt, half = math.radians(2.85), math.radians(34.85)
    axis = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])

    def f(p):
        q = p - centre
        return (q[..., 0] ** 2 + q[..., 2] ** 2 if kind == "cylinder" else np.sum(q * q, axis=-1)) - radius * radius

    def on(p):
        if kind == "sphere":
            q = p - centre
            ax = q @ axis
            return (ax >= 0.0) & (np.sum(q * q, axis=-1) - ax * ax <= (radius * math.sin(half)) ** 2)
        opening = np.zeros(p.shape[:-1], dtype=bool) if hole is None else (p[..., 0] - hole[0]) ** 2 + p[..., 1] ** 2 <= hole[1] ** 2
        return (p[..., 2] < radius) & ((p[..., 0] + 26.5 - 28.5) ** 2 + p[..., 1] ** 2 <= 28.5 ** 2) & ~opening

    def stopped(xi, eta):  # (stepping along each line from x = -35 mm, every line still outside, to x = 0 by 0.01 mm, each
        # sign change refined by bisection)
        hit = np.zeros(xi.shape, dtype=bool)
        decided = np.zeros(xi.shape, dtype=bool)
        s_prev = np.full(xi.shape, 25.0)
        f_prev = f(_at(src, d, e1, e2, width, s_prev, xi, eta))
        assert bool((f_prev > 0.0).all())
        for s_val in np.arange(25.01, 60.0 + 1e-9, 0.01):
            s = np.full(xi.shape, s_val)
            fv = f(_at(src, d, e1, e2, width, s, xi, eta))
            change = (np.sign(fv) != np.sign(f_prev)) & ~decided
            if change.any():
                idx = np.flatnonzero(change)
                lo, hi, f_lo = s_prev[idx], s[idx], f_prev[idx]
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    fm = f(_at(src, d, e1, e2, width, mid, xi[idx], eta[idx]))
                    left = np.sign(fm) != np.sign(f_lo)
                    hi, lo, f_lo = np.where(left, mid, hi), np.where(left, lo, mid), np.where(left, f_lo, fm)
                ok = on(_at(src, d, e1, e2, width, 0.5 * (lo + hi), xi[idx], eta[idx]))
                hit[idx[ok]] = f_prev[idx[ok]] > 0.0
                decided[idx[ok]] = True
            s_prev, f_prev = s, fv
        return hit

    return _rows_share(stopped, rows=61, cols=81)


@pytest.mark.parametrize("kind", ("sphere", "rotational", "cylinder"))
def test_a_beam_skimming_a_bowls_near_rim_misses_a_minimum_it_provably_misses(kind: str):
    """1.2 w above the near rim, 0.82 % of the beam meets the bowl's blocking back before T: T receives 0.99176 mW of the 1
    mW it records. The certain share (``power_taken_before_lower_W``) is at most that, so a 0.995 mW minimum is UNMET
    (it was INDETERMINATE); the bound from above still holds the share."""
    result = run(_bowl(kind, 0.36, 0.995e-3))
    (obs,) = result.observations
    (arrival,) = obs.arrivals
    truth = _bowl_share(kind, 0.36) * SOURCE_W
    assert truth == pytest.approx(0.0082e-3, rel=0.03)
    assert arrival.power_taken_before_lower_W <= truth <= arrival.power_taken_before_upper_W
    assert arrival.power_taken_before_lower_W >= 0.5 * truth
    assert result.targets[0].assessment.value == "UNMET"


def test_the_bowls_back_blocks_and_its_front_gives_no_certain_share():
    """The side the certain share reads: the chief ray sent onto the bowl's back is blocked (``BLOCKED_BY_BACK_SIDE``); a
    plane before a fold mirror's reflecting front gets no certain share (that light is reflected, not stopped)."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=0.3, position=(0.0, 0.0, -20.0), direction=(0.0, 0.0, 1.0))
    bowl = _bowl("rotational", 0.36, 0.0).components[1]
    result = run(ProjectSnapshot(components=(laser, bowl), observations=()))
    assert "BLOCKED_BY_BACK_SIDE" in {issue.code.value for issue in result.issues}
    q = math.sqrt(0.5)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-q, q, 0.0), clear_radius_mm=20.0)
    beam = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=SOURCE_W, position=(-30.0, 0.0, 0.0), direction=X)
    t = B.observation_plane("T", position=(-0.5, 0.0, 0.0), normal=X, radius_mm=50.0)
    (obs,) = run(ProjectSnapshot(components=(beam, mirror), observations=(t,))).observations
    incoming = next(a for a in obs.arrivals if a.crossing_index == 0)
    assert incoming.power_taken_before_at == ("M/face",) and incoming.power_taken_before_lower_W is None


@pytest.mark.parametrize("hole", ((-26.0, 0.45), (-26.2, 0.25)))
def test_a_hole_where_the_beam_meets_the_bowls_back_gives_no_certain_share(hole: tuple[float, float]):
    """The sag bowl 1.2 w above its near rim with a hole where most of the lines meet its back: they pass it, so only
    4.8e-6 / 5.4e-6 W of the 8.4e-6 W meet the back, less than the 6.3e-6 W the cells on the support without the hole
    certainly give. A surface with holes states no certain share; the bound from above still holds, and the 0.995 mW
    minimum (0.99523 mW arrive with the first hole) is not decided."""
    result = run(_bowl("rotational", 0.36, 0.995e-3, hole))
    (arrival,) = result.observations[0].arrivals
    truth = _bowl_share("rotational", 0.36, hole) * SOURCE_W
    assert truth < 0.7 * _bowl_share("rotational", 0.36) * SOURCE_W
    assert arrival.power_taken_before_at == ("M/S1",) and arrival.power_taken_before_lower_W is None
    assert truth <= arrival.power_taken_before_upper_W
    assert result.targets[0].assessment.value == "INDETERMINATE"


# ------------------------------------------------------------------------------- a surface beside the chief ray met first
@pytest.mark.parametrize(("offset", "tilt_deg"), ((0.3, 0.0), (0.45, 30.0), (0.5, 60.0)))
def test_a_surface_beside_the_chief_ray_that_pupil_rays_meet_first_bounds_the_plane(offset: float, tilt_deg: float):
    """A beam dump (a disc of radius 5 mm) beside a 0.5 mm beam, its edge ``offset`` mm above the chief ray, 10 mm before
    T and tilted ``tilt_deg`` about Y: pupil rays meet it first, the chief ray passes it. The share of the beam whose lines
    may meet it before T bounds what it takes (``possible_meeting_share``): the plane reads [P - S, P] (it was out of model
    scope), S holds the lines that meet the disc, and a 0.5 mW minimum is met."""
    w0, tilt = 0.5, math.radians(tilt_deg)
    normal = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
    centre = np.array([-10.0, 0.0, offset + 5.0 * math.cos(tilt)])  # (its lowest point ``offset`` above the chief ray)
    src, d = np.array([-60.0, 0.0, 0.0]), np.array(X)
    e1, e2 = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    width = _width(w0, 0.0)

    def meets(xi, eta):  # (the disc before the line's crossing of T at x = 0)
        s_d = _plane_hit(src, d, e1, e2, width, xi, eta, normal, centre)
        p = _at(src, d, e1, e2, width, s_d, xi, eta) - centre
        return (s_d < 60.0) & (np.sum(p * p, axis=-1) <= 25.0)

    truth = _rows_share(meets) * SOURCE_W
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=w0, position=tuple(src), direction=X)
    dump = B.beam_dump("D", position=tuple(centre), axis=tuple(-normal), radius_mm=5.0)
    t = B.observation_plane("T", position=(0.0, 0.0, 0.0), normal=X, radius_mm=20.0)
    result = run(ProjectSnapshot(components=(laser, dump), observations=(t,), targets=(
        Target(target_id="t", metric="power_W", observation_id="T", minimum=0.5e-3, unit="W"),)))
    (arrival,) = result.observations[0].arrivals
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("D/dump",)
    assert truth <= arrival.power_taken_before_upper_W <= 1.5 * truth + 1e-7 * SOURCE_W
    assert result.targets[0].assessment.value == "MET"
