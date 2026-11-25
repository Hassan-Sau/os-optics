"""The sphere batch of the tracer gives, for every (sphere, ray) pair, the ray parameter ``intersect`` gives, bit for bit.

``SphereBatch.times`` computes the ``(spheres, rays)`` table of candidate hit times in one vectorized pass; the tracer
chooses the next surface from it and evaluates only the chosen surface with ``intersect``. A candidate time that differed
from ``intersect(...).t`` in its last bit could order two surfaces differently, so the table is compared here with
``intersect`` of each surface over randomized spheres and rays with ``==`` on the bytes (``inf`` rows included): convex
and concave caps, rays that miss the sphere, rays that meet the sphere off its cap, rays inside the mount ring but outside
the clear aperture, rays beyond the mount ring, rays that start on the surface, rays from inside the sphere, and rays whose
rows hold no finite point (a retired ray).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from os_optics.config import NumericalConfig
from os_optics.math.vectors import dot
from os_optics.optics.surfaces.geometry import SphereBatch, intersect
from os_optics.scene.model import ApertureG, InteractionG, SurfaceG

CFG = NumericalConfig()


def _bits(x) -> bytes:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / math.sqrt(float(dot(v, v)))


def _sphere(rng: np.random.Generator, k: int) -> SurfaceG:
    """A spherical cap as the scene lowering writes one: vertex, cap axis from the centre to the vertex, clear radius and
    mount ring. Radii from a steep half-ball to a nearly flat cap; clear apertures up to the whole hemisphere."""
    radius = float(10.0 ** rng.uniform(0.3, 3.5))
    cap_axis = _unit(rng.standard_normal(3))
    vertex = rng.standard_normal(3) * 40.0
    center = vertex - radius * cap_axis
    helper = np.array([1.0, 0.0, 0.0]) if abs(cap_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = _unit(np.cross(cap_axis, helper))
    v_axis = np.cross(cap_axis, u_axis)
    aperture_radius = float(min(radius, 10.0 ** rng.uniform(0.0, 1.6)) * rng.uniform(0.2, 1.0))
    mount_margin = float(rng.choice([0.0, 0.5, 2.0, 1.0e-9])) if k % 3 else 0.0
    return SurfaceG(surface_id=f"e{k}/s", owner_id=f"e{k}", geometry="sphere", origin=vertex, normal=cap_axis, u_axis=u_axis, v_axis=v_axis,
                    center=center, radius=radius, cap_axis=cap_axis, aperture=ApertureG("circle", aperture_radius), mount_margin=mount_margin,
                    medium_pos="air", medium_neg="glass", interaction=InteractionG(kind="refract"), aperture_radius=aperture_radius)


def _cap_points(rng: np.random.Generator, s: SurfaceG, count: int, reach: float) -> np.ndarray:
    """Points on the sphere of ``s`` at transverse distances up to ``reach`` x (clear radius + mount ring) from its axis (on
    the cap side)."""
    rho = np.minimum(rng.uniform(0.0, reach, count) * (s.aperture_radius + s.mount_margin), s.radius)
    phi = rng.uniform(0.0, 2.0 * math.pi, count)
    axial = np.sqrt(np.maximum(s.radius * s.radius - rho * rho, 0.0))
    return (s.center + axial[:, None] * s.cap_axis + (rho * np.cos(phi))[:, None] * s.u_axis + (rho * np.sin(phi))[:, None] * s.v_axis)


def _rays(rng: np.random.Generator, spheres: list[SurfaceG], per_kind: int) -> tuple[np.ndarray, np.ndarray]:
    """Rays of every kind named in the module docstring, for every sphere in turn (each is also a ray for all the others)."""
    origins, directions = [], []
    for s in spheres:
        # aimed at the cap: inside the clear aperture, in the mount ring, beyond the ring (1.6 x the support)
        for reach in (0.6, 1.0, 1.6):
            target = _cap_points(rng, s, per_kind, reach)
            start = target + rng.standard_normal((per_kind, 3)) * rng.choice([0.5, 30.0, 400.0], size=(per_kind, 1))
            origins.append(start)
            directions.append(target - start)
        # along the cap axis, at points out to the edge of the clear aperture and out to the edge of the support (the rows
        # on the edges themselves: test_rays_at_the_edges_of_a_cap_have_the_bits_of_intersect)
        for edge in (s.aperture_radius, s.aperture_radius + s.mount_margin):
            scale = edge / (s.aperture_radius + s.mount_margin)
            rim = _cap_points(rng, s, per_kind, scale)
            start = rim - rng.uniform(1.0, 80.0, (per_kind, 1)) * s.cap_axis * rng.choice([-1.0, 1.0], size=(per_kind, 1))
            origins.append(start)
            directions.append(rim - start)
        # starting on the surface (t = 0 is not an advance: the other root, or a miss), going either way
        on_surface = _cap_points(rng, s, per_kind, 1.0)
        origins.append(on_surface)
        directions.append(rng.standard_normal((per_kind, 3)))
        # starting on the surface along a direction it was reached by (a relaunch after an interaction)
        start = on_surface - 25.0 * s.cap_axis + rng.standard_normal((per_kind, 3))
        origins.append(on_surface)
        directions.append(on_surface - start)
        # from inside the ball, from the centre, tangent to the sphere, and through the back of the ball (off the cap)
        origins.append(s.center + rng.standard_normal((per_kind, 3)) * 0.3 * s.radius)
        directions.append(rng.standard_normal((per_kind, 3)))
        origins.append(np.broadcast_to(s.center, (per_kind, 3)).copy())
        directions.append(rng.standard_normal((per_kind, 3)))
        tangent_point = s.center + s.radius * np.array([_unit(v) for v in rng.standard_normal((per_kind, 3))])
        tangent_dir = np.cross(tangent_point - s.center, rng.standard_normal((per_kind, 3)))
        origins.append(tangent_point - 3.0 * np.array([_unit(v) for v in tangent_dir]))
        directions.append(tangent_dir)
        back = s.center - s.radius * s.cap_axis + rng.standard_normal((per_kind, 3))
        origins.append(back - 50.0 * s.cap_axis)
        directions.append(np.broadcast_to(s.cap_axis, (per_kind, 3)) + 0.02 * rng.standard_normal((per_kind, 3)))
    # anywhere, any direction (most miss every sphere)
    origins.append(rng.standard_normal((4 * per_kind, 3)) * 200.0)
    directions.append(rng.standard_normal((4 * per_kind, 3)))
    o, d = np.vstack(origins), np.vstack(directions)
    d = d / np.sqrt(dot(d, d))[:, None]
    return np.ascontiguousarray(o), np.ascontiguousarray(d)


def _expected(spheres: list[SurfaceG], o: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.stack([intersect(s, o, d, CFG).t for s in spheres])


@pytest.mark.parametrize("seed", range(8))
def test_the_sphere_batch_table_has_the_bits_of_intersect(seed):
    rng = np.random.default_rng(1000 + seed)
    spheres = [_sphere(rng, k) for k in range(int(rng.integers(1, 9)))]
    batch = SphereBatch(spheres)
    o, d = _rays(rng, spheres, 40)
    got = batch.times(o, d, CFG)
    want = _expected(spheres, o, d)
    assert got.shape == want.shape == (len(spheres), o.shape[0]) and got.dtype == np.float64
    assert _bits(got) == _bits(want)
    assert (got == want).all()  # (no NaN in a table of times)
    # every kind of row is present: hits inside the clear aperture, hits on the mount ring only, and misses
    inside = np.stack([intersect(s, o, d, CFG).inside for s in spheres])
    hit = np.isfinite(want)
    assert (hit & inside).any() and (~hit).any()
    if any(s.mount_margin > 1.0e-6 for s in spheres):
        assert (hit & ~inside).any()
    assert (want[hit] > CFG.ray_min_advance_mm).all()


def test_the_sphere_batch_gives_the_same_table_one_ray_or_one_sphere_at_a_time():
    """A one-ray packet (a probe without pupil rays) and a one-sphere scene take the same arithmetic as the full table."""
    rng = np.random.default_rng(77)
    spheres = [_sphere(rng, k) for k in range(6)]
    batch = SphereBatch(spheres)
    o, d = _rays(rng, spheres, 6)
    want = _expected(spheres, o, d)
    for i in range(o.shape[0]):
        assert _bits(batch.times(o[i:i + 1], d[i:i + 1], CFG)) == _bits(want[:, i:i + 1]), i
    for k, s in enumerate(spheres):
        assert _bits(SphereBatch([s]).times(o, d, CFG)) == _bits(want[k:k + 1])
    # rows of a larger array that are not contiguous in memory (a packet's rows are sliced and reversed on the way)
    assert _bits(batch.times(o[::-2], d[::-2], CFG)) == _bits(want[:, ::-2])
    assert _bits(batch.times(np.asfortranarray(o), np.asfortranarray(d), CFG)) == _bits(want)


def test_a_ray_that_starts_on_a_sphere_does_not_meet_it_where_it_starts():
    """The start point's own root is within ``ray_min_advance_mm``: the batch, like ``intersect``, takes the other root (a
    ray going into the ball of a cap that spans it) or none."""
    rng = np.random.default_rng(5)
    spheres = [_sphere(rng, k) for k in range(5)]
    batch = SphereBatch(spheres)
    for k, s in enumerate(spheres):
        start = _cap_points(rng, s, 200, 0.9)
        d = rng.standard_normal((200, 3))
        d = d / np.sqrt(dot(d, d))[:, None]
        got = batch.times(start, d, CFG)[k]
        want = intersect(s, start, d, CFG).t
        assert _bits(got) == _bits(want)
        finite = np.isfinite(got)
        assert (got[finite] > CFG.ray_min_advance_mm).all()
        chord = -2.0 * dot(d, start - s.center)  # (the other root of a ray that starts on the sphere)
        assert np.allclose(got[finite], chord[finite], rtol=1.0e-9, atol=1.0e-9)


def _ulps_around(x: float, count: int) -> np.ndarray:
    """``x`` and the ``count`` floats on either side of it."""
    below, above, lo, hi = [], [], x, x
    for _ in range(count):
        lo, hi = float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf))
        below.append(lo)
        above.append(hi)
    return np.array(below[::-1] + [x] + above)


def _axial_cap(aperture_radius: float, mount_margin: float, radius: float = 50.0) -> SurfaceG:
    """A cap about the z axis, vertex at z = ``radius`` (centre at the origin)."""
    z = np.array([0.0, 0.0, 1.0])
    return SurfaceG(surface_id="e/s", owner_id="e", geometry="sphere", origin=radius * z, normal=z, u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 1.0, 0.0]), center=np.zeros(3), radius=radius, cap_axis=z,
                    aperture=ApertureG("circle", aperture_radius), mount_margin=mount_margin, medium_pos="air", medium_neg="glass",
                    interaction=InteractionG(kind="refract"), aperture_radius=aperture_radius)


@pytest.mark.parametrize("aperture_radius, mount_margin", [(10.0, 0.0), (10.0, 0.5), (7.3, 2.0), (3.0, 1.0e-9), (49.0, 0.0)])
def test_rays_on_the_edge_of_the_support_hit_up_to_it_and_miss_beyond_it(aperture_radius, mount_margin):
    """Rays along the axis of a cap at 8 ulps either side of the support's edge (clear radius + mount ring + edge
    tolerance) and of the clear aperture's edge: the comparison that turns a hit into a miss. The batch gives the bits of
    ``intersect``, and the rows straddle the edge: the point is carried exactly, so every row up to the edge hits and
    every row beyond it misses; at the clear aperture's edge (with a mount ring) the rows hit, inside and outside it."""
    s = _axial_cap(aperture_radius, mount_margin)
    reach = (aperture_radius + mount_margin) + CFG.aperture_edge_tol_mm
    rho = _ulps_around(reach, 8)
    o = np.stack([rho, np.zeros_like(rho), np.full_like(rho, 120.0)], axis=1)
    d = np.broadcast_to(np.array([0.0, 0.0, -1.0]), o.shape).copy()
    got, want = SphereBatch([s]).times(o, d, CFG)[0], intersect(s, o, d, CFG).t
    assert _bits(got) == _bits(want)
    assert np.isfinite(got[:9]).all() and np.isinf(got[9:]).all()
    if mount_margin > 1.0e-6:
        clear = _ulps_around(aperture_radius + CFG.aperture_edge_tol_mm, 8)
        o = np.stack([clear, np.zeros_like(clear), np.full_like(clear, 120.0)], axis=1)
        hit = intersect(s, o, d, CFG)
        assert _bits(SphereBatch([s]).times(o, d, CFG)[0]) == _bits(hit.t)
        assert np.isfinite(hit.t).all() and hit.inside[:9].all() and not hit.inside[9:].any()


@pytest.mark.parametrize("seed", range(4))
def test_rays_at_the_edges_of_a_cap_have_the_bits_of_intersect(seed):
    """Caps in any orientation, rays along the cap axis at points ``edge (1 + k 2^-52)`` (``k`` in -8..8) from the axis, for
    the clear aperture's edge and the support's edge with and without the edge tolerance. Rounding of the rotated frame
    decides each row's side of the edge, so only the bits are compared (both forms round alike)."""
    rng = np.random.default_rng(4000 + seed)
    spheres = [_sphere(rng, k) for k in range(6)]
    batch = SphereBatch(spheres)
    origins = []
    for s in spheres:
        for edge in (s.aperture_radius, s.aperture_radius + s.mount_margin,
                     (s.aperture_radius + s.mount_margin) + CFG.aperture_edge_tol_mm):
            rho = np.minimum(edge * (1.0 + np.arange(-8, 9) * 2.0 ** -52), s.radius)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            axial = np.sqrt(np.maximum(s.radius * s.radius - rho * rho, 0.0))
            rim = s.center + axial[:, None] * s.cap_axis + (rho * math.cos(phi))[:, None] * s.u_axis + (rho * math.sin(phi))[:, None] * s.v_axis
            origins.append(rim + 30.0 * s.cap_axis)
    o = np.vstack(origins)
    d = np.vstack([np.broadcast_to(-s.cap_axis, (51, 3)) for s in spheres])
    got = batch.times(o, d, CFG)
    assert _bits(got) == _bits(_expected(spheres, o, d))
    own = np.concatenate([got[k, 51 * k:51 * (k + 1)] for k in range(len(spheres))])
    assert np.isfinite(own).any() and np.isinf(own).any()


def test_a_start_just_before_the_surface_meets_it_beyond_the_minimum_advance():
    """Rays along the axis of a cap starting ``ray_min_advance_mm (1 + k 10^-3)`` (``k`` in -5..5) before its vertex: the
    near root is taken beyond the minimum advance and skipped within it (then the ray leaves the ball off the cap and
    misses); the batch gives the bits of ``intersect`` on both sides."""
    s = _axial_cap(10.0, 0.5)
    offsets = CFG.ray_min_advance_mm * (1.0 + np.arange(-5, 6) * 1.0e-3)
    o = np.stack([np.zeros_like(offsets), np.zeros_like(offsets), s.radius + offsets], axis=1)
    d = np.broadcast_to(np.array([0.0, 0.0, -1.0]), o.shape).copy()
    got, want = SphereBatch([s]).times(o, d, CFG)[0], intersect(s, o, d, CFG).t
    assert _bits(got) == _bits(want)
    assert np.isfinite(got).any() and np.isinf(got).any()
    assert (got[np.isfinite(got)] > CFG.ray_min_advance_mm).all()


def test_rows_without_a_finite_point_miss_every_sphere_as_in_intersect():
    """A retired ray's row may hold ``nan`` or ``inf`` (it had no hit on an earlier surface): its times are ``inf`` in both."""
    rng = np.random.default_rng(9)
    spheres = [_sphere(rng, k) for k in range(4)]
    batch = SphereBatch(spheres)
    o, d = _rays(rng, spheres, 3)
    o, d = o.copy(), d.copy()
    o[1], o[5], d[7] = np.nan, np.inf, np.nan
    o[9, 1], d[11, 2] = -np.inf, np.inf
    d[13] = 0.0  # (no direction at all)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # (inf - inf and 0 x inf in those rows, in both forms)
        got = batch.times(o, d, CFG)
        want = _expected(spheres, o, d)
    assert _bits(got) == _bits(want)
    assert np.isinf(got[:, [1, 5, 7, 9, 11]]).all()
