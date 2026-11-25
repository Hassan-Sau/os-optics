"""Phase 2: vector / rotation / pose / RNG foundation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.math import (
    DeterministicRng,
    Pose,
    matrix_to_quat,
    normalize,
    parallel_transport_basis,
    quat_from_axis_angle,
    quat_multiply,
    quat_to_matrix,
    reflect,
    rotation_matrices_between,
    transverse_basis,
)


def test_normalize_zero_vector_is_flagged_not_nan():
    unit, ok = normalize(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0], [np.nan, 0.0, 1.0]]), CFG.zero_vector_tol)
    assert list(ok) == [False, True, False]
    assert np.all(np.isfinite(unit))
    np.testing.assert_allclose(unit[1], [0.6, 0.0, 0.8])


@pytest.mark.parametrize("axis,angle,expected", [
    ((0, 0, 1), 90.0, (0.0, 1.0, 0.0)),   # x -> y under +90 deg about z (right-handed)
    ((0, 1, 0), 90.0, (0.0, 0.0, -1.0)),  # x -> -z under +90 about y
])
def test_axis_angle_right_handed(axis, angle, expected):
    m = quat_to_matrix(quat_from_axis_angle(np.array(axis, float), math.radians(angle)))
    np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), expected, atol=1e-15)


def test_matrix_quat_roundtrip_all_branches():
    rng = np.random.default_rng(1234)  # test-only randomness, seeded
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = quat_to_matrix(q)
        q2 = matrix_to_quat(m)
        np.testing.assert_allclose(quat_to_matrix(q2), m, atol=1e-13)
    for axis in np.eye(3):  # 180-degree rotations exercise the non-trace branches
        m = quat_to_matrix(quat_from_axis_angle(axis, math.pi))
        np.testing.assert_allclose(quat_to_matrix(matrix_to_quat(m)), m, atol=1e-13)


def test_quaternion_composition_matches_matrix_product():
    qa = quat_from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.7)
    qb = quat_from_axis_angle(np.array([-1.0, 0.5, 0.2]), -1.1)
    np.testing.assert_allclose(quat_to_matrix(quat_multiply(qa, qb)), quat_to_matrix(qa) @ quat_to_matrix(qb), atol=1e-14)


def test_pose_compose_inverse():
    a = Pose.from_quat(np.array([1.0, 2.0, 3.0]), quat_from_axis_angle(np.array([0, 0, 1.0]), 0.3))
    b = Pose.from_quat(np.array([-4.0, 0.5, 2.0]), quat_from_axis_angle(np.array([0, 1.0, 0]), -0.9))
    p = np.array([[0.3, -1.0, 2.0]])
    np.testing.assert_allclose(a.compose(b).apply_points(p), a.apply_points(b.apply_points(p)), atol=1e-14)
    ident = a.compose(a.inverse())
    np.testing.assert_allclose(ident.rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(ident.translation, 0.0, atol=1e-14)


def test_from_axis_direction_frame():
    pose = Pose.from_axis_direction(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), CFG.zero_vector_tol)
    np.testing.assert_allclose(pose.rotation @ [0, 0, 1], [1, 0, 0], atol=1e-15)  # local z = axis
    np.testing.assert_allclose(pose.rotation @ [0, 1, 0], [0, 0, 1], atol=1e-15)  # local y = up
    assert np.linalg.det(pose.rotation) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Pose.from_axis_direction(np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), CFG.zero_vector_tol)
    with pytest.raises(ValueError):
        Pose.from_axis_direction(np.zeros(3), np.zeros(3), np.array([0, 0, 1.0]), CFG.zero_vector_tol)


def test_transverse_basis_right_handed_and_deterministic():
    rng = np.random.default_rng(7)
    d = rng.normal(size=(50, 3))
    d, _ = normalize(d, CFG.zero_vector_tol)
    e1, e2 = transverse_basis(d, CFG.zero_vector_tol)
    np.testing.assert_allclose(np.cross(e1, e2), d, atol=1e-14)
    np.testing.assert_allclose(np.einsum("ni,ni->n", e1, d), 0.0, atol=1e-14)
    e1b, e2b = transverse_basis(d, CFG.zero_vector_tol)
    assert np.array_equal(e1, e1b) and np.array_equal(e2, e2b)


def test_rotation_between_including_antiparallel_and_parallel():
    a = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    b = np.array([[0, 1.0, 0], [0, 1.0, 0], [0, 0, -1.0], [1.0, 0, 0]])
    fb = np.array([[0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    rot = rotation_matrices_between(a, b, CFG.antiparallel_tol, fb)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", rot, a), b, atol=1e-14)
    for r in rot:
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0)


@pytest.mark.parametrize("theta", [1.6e-6, 1e-5, 3e-4, 0.05, 0.44, 0.46, 1.2])
def test_rotation_between_nearly_opposite_directions_stays_orthonormal(theta):
    """A beam reflected a microradian off normal incidence: ``b`` is ``-a`` turned by ``theta``, ``1 + a.b`` about
    ``theta^2 / 2`` -- just above ``antiparallel_tol`` for the smallest. Rodrigues' ``1 / (1 + c)`` there carries the
    rounding of ``c`` magnified (a beam axis transported through it read 1.0007 long); the rotation is orthonormal to
    rounding on both sides of the switch to its axis-angle form (``1 + c`` = 0.1), and takes ``a`` onto ``b`` to the
    resolution of its axis ``a x b`` (a few ``eps / theta``: the axis of a turn by nearly pi is that ill-conditioned)."""

    rng = np.random.default_rng(7)
    for _ in range(40):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        u = np.cross(a, rng.normal(size=3))
        u /= np.linalg.norm(u)
        b = -math.cos(theta) * a + math.sin(theta) * u
        b /= np.linalg.norm(b)
        assert 1.0 + float(a @ b) > CFG.antiparallel_tol
        (r,) = rotation_matrices_between(a[None, :], b[None, :], CFG.antiparallel_tol, u[None, :])
        np.testing.assert_allclose(r @ r.T, np.eye(3), rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(r @ a, b, rtol=0.0, atol=max(1e-14, 2e-16 / theta))


def test_a_frame_reflected_back_on_itself_many_times_keeps_its_length_and_orthogonality():
    """A beam reflected exactly back on itself (a cavity, a retro-reflection) is turned by pi about its fallback axis, the
    transported frame's own axis, which is unit and perpendicular to the beam only to rounding. The rotation takes that
    axis's part perpendicular to the beam, made unit, so it is orthogonal: over 60 bounces the frame's lengths and
    orthogonality move by the rounding of each bounce, about ``eps`` (with the axis as given an error grew about 5-fold per
    bounce, |bx|^2 - 1 reaching -0.69 after 24)."""

    g = np.array([0.95, 0.12, -0.21, 0.18])
    w, x, y, z = g / np.linalg.norm(g)
    turn = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    d, bx, by = turn @ np.array([0.0, 0.0, 1.0]), turn @ np.array([1.0, 0.0, 0.0]), turn @ np.array([0.0, 1.0, 0.0])
    bx = bx * (1.0 + 3e-15) + 2e-15 * d  # (not unit, not quite perpendicular to the beam)
    start = (float(bx @ bx), float(by @ by), float(bx @ by))
    for bounce in range(60):
        (r,) = rotation_matrices_between(d[None, :], -d[None, :], CFG.antiparallel_tol, bx[None, :])
        np.testing.assert_allclose(r @ r.T, np.eye(3), rtol=0.0, atol=4e-15)
        bx, by, d = r @ bx, r @ by, -d
        drift = 1e-15 * (bounce + 1)  # (the rounding of each bounce, about eps)
        assert abs(float(bx @ bx) - start[0]) < drift and abs(float(by @ by) - start[1]) < drift, bounce
        assert abs(float(bx @ by) - start[2]) < drift, bounce
        assert abs(float(np.cross(bx, by) @ d) - 1.0) < 1e-14 + drift, bounce


def test_parallel_transport_keeps_orthonormality_over_many_steps():
    d = np.array([[0.0, 0.0, 1.0]])
    e1, e2 = transverse_basis(d, CFG.zero_vector_tol)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        nd = rng.normal(size=(1, 3))
        nd, _ = normalize(nd, CFG.zero_vector_tol)
        e1, e2 = parallel_transport_basis(e1, e2, d, nd, CFG.antiparallel_tol)
        d = nd
    np.testing.assert_allclose(np.cross(e1, e2), d, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ni,ni->n", e1, e1), 1.0, atol=1e-12)


def test_reflect_45_degree_mirror_closed_form():
    n = np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0)
    out = reflect(np.array([1.0, 0.0, 0.0]), n)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-15)


def test_rng_is_deterministic_and_seed_sensitive():
    a = [DeterministicRng(42).normal() for _ in range(1)]
    r1, r2, r3 = DeterministicRng(42), DeterministicRng(42), DeterministicRng(43)
    s1 = [r1.normal() for _ in range(64)]
    s2 = [r2.normal() for _ in range(64)]
    s3 = [r3.normal() for _ in range(64)]
    assert s1 == s2 and s1 != s3 and a[0] == s1[0]
    u = [DeterministicRng(5).uniform() for _ in range(3)]
    assert all(0.0 <= x < 1.0 for x in u)


def test_rng_statistics_reasonable():
    r = DeterministicRng(2024)
    xs = np.array([r.normal() for _ in range(20000)])
    assert abs(xs.mean()) < 0.03 and abs(xs.std() - 1.0) < 0.03
    with pytest.raises(ValueError):
        DeterministicRng(-1)


def test_rng_golden_stream_is_stable():
    """Pin the raw stream: guards against accidental change of the transform (recorded from this implementation)."""
    r = DeterministicRng(12345)
    first = [r.uniform() for _ in range(3)]
    bg = np.random.PCG64(12345)
    expected = [(int(bg.random_raw()) >> 11) / 2**53 for _ in range(3)]
    assert first == expected
