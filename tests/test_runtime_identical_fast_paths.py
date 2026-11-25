"""The fast forms of the runtime-identical arithmetic give the same bits as the plain forms they replace.

``math.vectors`` / ``math.exact_trig`` evaluate one vector, one matrix or one angle in Python floats (each ``+ - x /`` and
``sqrt`` is the same correctly rounded IEEE operation as NumPy's), ``matvec`` / ``matmul`` accumulate over the inner index
for all entries at once, the plane batch of the tracer computes its ``(planes, rays)`` table along the rays and keeps the
products of unchanged directions, the exponential stops its series where the remaining terms cannot change the sum, and
the Newton iteration of the Gauss-Legendre nodes stops once it repeats itself. Each is compared here with the plain form,
bit for bit (``tobytes``: signed zeros, subnormals and infinities included).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.config import NumericalConfig
from os_optics.math import exact_trig
from os_optics.math.exact_linalg import power2
from os_optics.math.vectors import dot, matmul, matvec
from os_optics.optics.gaussian.truncation import LEGENDRE_NEWTON_MAX_STEPS, gauss_legendre_rule
from os_optics.optics.surfaces.geometry import PlaneBatch, support_of
from os_optics.scene.model import ApertureG


def _bits(x) -> bytes:
    """The bytes of ``x`` as float64, every NaN written as one quiet NaN: which NaN an invalid operation yields (sign,
    payload) is not specified by IEEE-754 or WebAssembly and never reaches a result; every other value is compared bit
    for bit (signed zeros, subnormals, infinities)."""
    a = np.array(x, dtype=np.float64)
    a[np.isnan(a)] = np.nan
    return a.tobytes()


def _plain_dot(a, b):
    """``math.vectors.dot`` without the one-vector fast path (the general broadcast form)."""
    a, b = np.asarray(a), np.asarray(b)
    total = a[..., 0] * b[..., 0]
    for i in range(1, max(a.shape[-1], b.shape[-1])):
        total = total + a[..., i] * b[..., i]
    return total


SPECIAL = [0.0, -0.0, 1.0, -1.0, 1e-310, -1e-310, 1e308, -1e308, math.inf, -math.inf, math.nan, 2.0 ** -1074, 0.1, 1.0 / 3.0]


def _vectors(rng: np.random.Generator, count: int, size: int) -> np.ndarray:
    scale = 10.0 ** rng.integers(-12, 12, size=(count, size))
    values = rng.standard_normal((count, size)) * scale
    special = rng.random((count, size)) < 0.1
    values[special] = rng.choice(SPECIAL, size=int(special.sum()))
    return values


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_the_dot_of_two_vectors_has_the_bits_of_the_broadcast_form(size):
    rng = np.random.default_rng(size)
    a, b = _vectors(rng, 3000, size), _vectors(rng, 3000, size)
    with np.errstate(all="ignore"):  # (the special values overflow and multiply inf by 0 on purpose)
        plain = _plain_dot(a, b)
        for i in range(a.shape[0]):
            got = dot(a[i], b[i])
            assert type(got) is np.float64
            assert _bits(got) == _bits(plain[i]), (a[i], b[i])


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_the_dot_of_rows_with_one_vector_has_the_bits_of_the_broadcast_form(size):
    rng = np.random.default_rng(100 + size)
    rows = _vectors(rng, 2000, size)
    with np.errstate(all="ignore"):
        for vector in _vectors(rng, 40, size):
            for got, want in ((dot(rows, vector), _plain_dot(rows, vector)), (dot(vector, rows), _plain_dot(vector, rows))):
                assert got.shape == want.shape == (2000,) and _bits(got) == _bits(want)
        assert dot(rows[:0], rows[0]).shape == (0,)


def test_the_pupil_samples_have_the_bits_of_one_angle_at_a_time():
    from os_optics.math.exact_trig import cis, log
    from os_optics.optics.rays.pupil import pupil_offsets

    for w_x, w_y, rings, points in ((1.0, 1.0, 5, 100), (0.37, 2.5e-3, 3, 7), (1.0, 1.0, 1, 1), (2.0, 3.0, 0, 4), (2.0, 3.0, 4, 0)):
        want = []
        for k in range(rings):
            rho = math.sqrt(-0.5 * log(1.0 - (k + 0.5) / rings))
            offset = 0.5 if (k % 2) else 0.0
            for m in range(points):
                c = cis(2.0 * math.pi * (m + offset) / points)
                want.append((w_x * rho * c.real, w_y * rho * c.imag))
        got = pupil_offsets(w_x, w_y, rings, points)
        assert got.shape == (rings * points, 2) and _bits(got) == _bits(np.array(want, dtype=np.float64).reshape(-1, 2))


def test_matvec_and_matmul_have_the_bits_of_rows_of_dots():
    rng = np.random.default_rng(11)
    with np.errstate(all="ignore"):
        for shape_m, shape_v in [((3, 3), (3,)), ((2, 2), (2,)), ((2, 3), (3,)), ((40, 3, 3), (40, 3)), ((3, 3), (40, 3)),
                                 ((40, 3, 3), (3,))]:
            m = _vectors(rng, int(np.prod(shape_m)) // shape_m[-1], shape_m[-1]).reshape(shape_m)
            v = _vectors(rng, max(1, int(np.prod(shape_v)) // shape_v[-1]), shape_v[-1]).reshape(shape_v)
            want = np.stack([_plain_dot(m[..., i, :], v) for i in range(m.shape[-2])], axis=-1)
            got = matvec(m, v)
            assert got.shape == want.shape and got.flags.c_contiguous and _bits(got) == _bits(want), (shape_m, shape_v)
        for shape_a, shape_b in [((3, 3), (3, 3)), ((2, 2), (2, 2)), ((2, 3), (3, 4)), ((40, 3, 3), (40, 3, 3)), ((3, 3), (40, 3, 3))]:
            a = _vectors(rng, int(np.prod(shape_a)) // shape_a[-1], shape_a[-1]).reshape(shape_a)
            b = _vectors(rng, int(np.prod(shape_b)) // shape_b[-1], shape_b[-1]).reshape(shape_b)
            bt = np.swapaxes(b, -1, -2)
            want = np.stack([np.stack([_plain_dot(a[..., i, :], bt[..., j, :]) for j in range(bt.shape[-2])], axis=-1)
                             for i in range(a.shape[-2])], axis=-2)
            got = matmul(a, b)
            assert got.shape == want.shape and got.flags.c_contiguous and _bits(got) == _bits(want), (shape_a, shape_b)


def _plain_times(batch: PlaneBatch, o, d, cfg: NumericalConfig):
    """The ``(rays, planes)`` table of ``PlaneBatch.times`` as first written (broadcast dots), transposed."""
    o3, d3 = o[:, None, :], d[:, None, :]
    denom = _plain_dot(d3, batch.normal[None, :, :])
    numer = _plain_dot(batch.origin[None, :, :] - o3, batch.normal[None, :, :])
    ok = np.abs(denom) > cfg.parallel_dot_tol
    t = np.where(ok, numer / np.where(ok, denom, 1.0), np.inf)
    valid = ok & (t > cfg.ray_min_advance_mm)
    t = np.where(valid, t, np.inf)
    ts = np.where(valid, t, 0.0)
    u = (_plain_dot(o3, batch.u_axis[None, :, :]) - batch.origin_u[None, :]) + ts * _plain_dot(d3, batch.u_axis[None, :, :]) - batch.off_u[None, :]
    v = (_plain_dot(o3, batch.v_axis[None, :, :]) - batch.origin_v[None, :]) + ts * _plain_dot(d3, batch.v_axis[None, :, :]) - batch.off_v[None, :]
    tol = cfg.aperture_edge_tol_mm
    circle = (u * u + v * v) <= (batch.p1[None, :] + tol) ** 2
    rect = (np.abs(u) <= batch.p1[None, :] + tol) & (np.abs(v) <= batch.p2[None, :] + tol)
    support = np.where(batch.shape[None, :] == 1, circle, np.where(batch.shape[None, :] == 2, rect, True))
    return np.where(valid & support, t, np.inf).T


class _Plane:
    """The fields of a plane ``SurfaceG`` that ``PlaneBatch`` reads."""

    def __init__(self, rng: np.random.Generator, shape: str) -> None:
        self.origin = rng.standard_normal(3) * 50.0
        n = rng.standard_normal(3)
        self.normal = n / math.sqrt(float(dot(n, n)))
        helper = np.array([1.0, 0.0, 0.0]) if abs(self.normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(self.normal, helper)
        self.u_axis = u / math.sqrt(float(dot(u, u)))
        self.v_axis = np.cross(self.normal, self.u_axis)
        p1, p2 = float(rng.uniform(1.0, 30.0)), float(rng.uniform(1.0, 30.0))
        self.aperture = ApertureG(shape, p1, p2, float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0)), 0.0)
        self.outline = None
        self.mount_margin = float(rng.uniform(0.0, 3.0))


@pytest.mark.parametrize("shapes", [("none",) * 5, ("circle",) * 6, ("rectangle",) * 4, ("none", "circle", "rectangle", "circle", "none", "rectangle", "none")])
def test_the_plane_batch_table_has_the_bits_of_the_broadcast_form(shapes):
    rng = np.random.default_rng(len(shapes))
    cfg = NumericalConfig()
    planes = [_Plane(rng, shape) for shape in shapes]
    assert [support_of(p).shape for p in planes] == list(shapes)
    batch = PlaneBatch(planes)
    d = rng.standard_normal((300, 3))
    d /= np.sqrt(_plain_dot(d, d))[:, None]
    d[:5] = planes[0].u_axis  # parallel to the first plane
    for step in range(4):  # the same directions from new origins (the kept direction products), then new directions
        o = rng.standard_normal((300, 3)) * 40.0
        if step == 3:
            d = d[::-1].copy()
        got = batch.times(o, d, cfg)
        want = _plain_times(batch, o, d, cfg)
        assert got.shape == want.shape == (len(shapes), 300)
        assert _bits(np.ascontiguousarray(got)) == _bits(np.ascontiguousarray(want)), step
        assert np.isfinite(got).any() and not np.isfinite(got).all()


def _exp_full(x):
    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x / exact_trig._LN2)
    r = (x - k * exact_trig._LN2_HI) - k * exact_trig._LN2_LO
    total = np.ones_like(r)
    term = np.ones_like(r)
    for n in range(1, exact_trig._EXP_TERMS):
        term = term * r / n
        total = total + term
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(total, np.clip(k, -exact_trig._EXP_MAX_SCALE, exact_trig._EXP_MAX_SCALE).astype(np.int64))


def test_the_short_exponential_series_has_the_bits_of_the_full_one():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.uniform(-0.4, 0.4, 400_000), rng.uniform(-750.0, 710.0, 400_000), rng.standard_normal(200_000) * 1e-6,
                        -np.abs(rng.standard_normal(100_000)) * 10.0 ** rng.uniform(0.0, 15.0, 100_000),
                        np.arange(-1100, 1100) * exact_trig._LN2 * 0.5, np.array(SPECIAL)])
    with np.errstate(all="ignore"):
        for chunk in np.array_split(x, 50):
            assert _bits(exact_trig.exp(chunk)) == _bits(_exp_full(chunk))
        for value in x[::997]:
            assert _bits(exact_trig.exp(value)) == _bits(_exp_full(value))
    # the bound behind it: the first omitted term is below half the spacing of doubles at the partial sum
    first_omitted = exact_trig._EXP_SHORT_MAX_R ** exact_trig._EXP_SHORT_TERMS / math.factorial(exact_trig._EXP_SHORT_TERMS)
    assert first_omitted < 2.0 ** -54 and math.exp(-exact_trig._EXP_SHORT_MAX_R) > 0.5  # (spacing 2^-53 on [0.5, 1])
    assert exact_trig._EXP_SHORT_MAX_R > 0.5 * exact_trig._LN2


def test_a_medium_without_absorption_attenuates_by_exactly_one():
    """The tracer skips the attenuation series where the absorption is 0: the exponent ``-0.5 * 0 * t`` of every ray is
    +-0 (``t`` finite), whose exponential is exactly 1."""
    t = np.array([0.0, 1e-300, 0.25, 150.0, 1e300])
    exponent = -0.5 * 0.0 * t
    assert not exponent.any()
    assert _bits(exact_trig.exp(exponent)) == _bits(np.ones(t.shape)) == _bits(exact_trig.exp(-exponent))


def test_one_angle_in_python_floats_has_the_bits_of_the_array_form():
    rng = np.random.default_rng(3)
    cosines = np.concatenate([rng.uniform(-0.2, 1.2, 20_000), 1.0 - 10.0 ** -rng.uniform(1.0, 17.0, 5000),
                              np.array([0.0, -0.0, 1.0, 1.0 - 1e-16, 0.5, math.nan, math.inf, -math.inf, 2.0, -3.0])])
    array_form = exact_trig.acos_deg(cosines)  # (the array path)
    for c, want in zip(cosines.tolist(), array_form.tolist()):
        got = exact_trig.acos_deg(c)
        assert type(got) is np.float64 and _bits(got) == _bits(want), c
        assert _bits(exact_trig.acos_deg(np.float64(c))) == _bits(want)
    ys = np.concatenate([rng.standard_normal(20_000) * 10.0 ** rng.uniform(-5, 5, 20_000), np.array(SPECIAL), np.zeros(len(SPECIAL))])
    xs = np.concatenate([rng.standard_normal(20_000) * 10.0 ** rng.uniform(-5, 5, 20_000), np.zeros(len(SPECIAL)), np.array(SPECIAL)])
    with np.errstate(all="ignore"):
        array_form = exact_trig.atan2(ys, xs)
        for y, x, want in zip(ys.tolist(), xs.tolist(), array_form.tolist()):
            got = exact_trig.atan2(y, x)
            assert isinstance(got, np.ndarray) and got.ndim == 0 and _bits(got) == _bits(want), (y, x)


def test_the_power_of_a_jones_vector_has_the_bits_of_its_complex_components():
    rng = np.random.default_rng(9)
    for _ in range(2000):
        jones = (rng.standard_normal(2) + 1j * rng.standard_normal(2)) * 10.0 ** rng.uniform(-8, 8)
        x, y = complex(jones[0]), complex(jones[1])
        want = (x.real * x.real + x.imag * x.imag) + (y.real * y.real + y.imag * y.imag)
        assert _bits(power2(jones)) == _bits(want) and _bits(power2(jones[None, :][0])) == _bits(want)


def _gauss_legendre_all_steps(n: int):
    """The rule with every one of the ``LEGENDRE_NEWTON_MAX_STEPS`` Newton steps taken (no early stop but the zero step)."""
    guess, _ = exact_trig.cos_sin(np.pi * (np.arange(1, n + 1) - 0.25) / (n + 0.5))
    x = guess
    derivative = np.ones_like(x)
    for _ in range(LEGENDRE_NEWTON_MAX_STEPS):
        p_prev, p = np.ones_like(x), x
        for k in range(2, n + 1):
            p_prev, p = p, ((2 * k - 1) * x * p - (k - 1) * p_prev) / k
        derivative = n * (x * p - p_prev) / ((x - 1.0) * (x + 1.0))
        step = p / derivative
        x = x - step
        if not bool(np.any(step != 0.0)):
            break
    weights = 2.0 / ((1.0 - x) * (1.0 + x) * derivative * derivative)
    return x[::-1].copy(), weights[::-1].copy()


@pytest.mark.parametrize("nodes", [*range(1, 41), 48, 64, 96, 128])
def test_the_gauss_legendre_rule_ends_on_the_nodes_of_every_newton_step(nodes):
    """Newton's last-bit corrections alternate between neighbouring doubles for many node counts (48 included): the rule
    stops once the nodes repeat and still ends on the nodes and weights of all ``LEGENDRE_NEWTON_MAX_STEPS`` steps."""
    x, w = gauss_legendre_rule(nodes)
    want_x, want_w = _gauss_legendre_all_steps(nodes)
    assert _bits(x) == _bits(want_x) and _bits(w) == _bits(want_w)
    assert float(np.sum(w)) == pytest.approx(2.0, abs=1e-13)


def _plain_atan_unit(t):
    """``exact_trig._atan_unit`` as first written: ``total = total + term / (2k + 1)``, ``term = -term * u2``."""
    t = np.asarray(t, dtype=np.float64)
    big = t > exact_trig._TAN_15_DEG
    u = np.where(big, (t * exact_trig._SQRT3 - 1.0) / (exact_trig._SQRT3 + t), t)
    u2 = u * u
    total = np.zeros_like(u)
    term = u
    for k in range(exact_trig._ATAN_TERMS):
        total = total + term / (2 * k + 1)
        term = -term * u2
    return np.where(big, exact_trig._PI_OVER_6 + total, total)


def test_the_arctangent_series_in_place_has_the_bits_of_the_plain_one():
    """``_atan_unit`` adds in place and multiplies by ``-u^2`` once negated: ``(-term) * u2`` and ``term * (-u2)`` are the
    same correctly rounded product, so every argument keeps its bits."""
    rng = np.random.default_rng(11)
    t = np.concatenate([rng.random(200_000), rng.random(100_000) ** 8, 10.0 ** rng.uniform(-300.0, 0.0, 100_000),
                        np.array([exact_trig._TAN_15_DEG, np.nextafter(exact_trig._TAN_15_DEG, 0.0), np.nextafter(exact_trig._TAN_15_DEG, 1.0),
                                  *SPECIAL])])
    with np.errstate(all="ignore"):
        for chunk in np.array_split(t, 20):
            assert _bits(exact_trig._atan_unit(chunk)) == _bits(_plain_atan_unit(chunk))
        assert _bits(exact_trig._atan_unit(np.float64(0.3))) == _bits(_plain_atan_unit(0.3))


def _incidence_rows(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``count`` unit directions about a unit normal, tilted from 0 (exactly normal) through the near-normal switch
    (``1 - cos`` of 1e-4, about 0.81 deg) to large angles; per-row normals a few 1e-9 rad apart; their ``|n . d|``."""
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    scale = rng.choice([0.0, 1e-17, 1e-12, 1e-9, 1e-6, 1e-3, 0.0141, 0.01414, 0.0142, 0.02, 0.3, 1.5], size=count)
    d = normal[None, :] + rng.standard_normal((count, 3)) * scale[:, None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    normals = normal[None, :] + rng.standard_normal((count, 3)) * 1e-9
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normal, d, normals


def test_the_angles_of_incidence_of_many_rows_have_the_bits_of_one_row_at_a_time():
    """``incidence_deg`` of an array takes one arctangent of every row's own argument (near the normal ``|n x d| / |n . d|``,
    away from it ``acos_deg``'s): whatever mix of rows an array holds, each row has the bits of that row alone (the
    Python-float paths), with one normal for all rows and with a normal per row."""
    rng = np.random.default_rng(29)
    for trial in range(300):
        count = int(rng.integers(2, 40))
        normal, d, normals = _incidence_rows(rng, count)
        if trial % 5 == 0:
            d = -d
        for n in (normal, normals):
            cos = np.minimum(np.abs(np.sum(d * np.broadcast_to(n, d.shape), axis=1)), 1.0)
            if trial % 7 == 0:
                cos[0] = math.nan
            many = exact_trig.incidence_deg(cos, d, n)
            assert many.shape == cos.shape and many.dtype == np.float64
            for k in range(count):
                row_normal = n if n.ndim == 1 else n[k]
                single = exact_trig.incidence_deg(float(cos[k]), d[k], row_normal)
                assert _bits(many[k]) == _bits(single), (trial, k, float(cos[k]))
                assert _bits(exact_trig.incidence_deg(cos[k:k + 1], d[k:k + 1], row_normal)) == _bits([single])


def test_an_array_of_angles_of_incidence_takes_one_arctangent(monkeypatch):
    """The rows near the normal and those away from it share one ``_atan_unit`` call; one angle takes one arctangent in
    Python floats, in the form it returns."""
    rng = np.random.default_rng(31)
    normal, d, normals = _incidence_rows(rng, 25)
    d[0], d[1] = normal, np.array([0.3, 0.1, 0.9486832980505138])  # (one row at the normal, one far from it)
    calls = {"array": 0, "float": 0}
    array_form, float_form = exact_trig._atan_unit, exact_trig._atan_unit_float

    def counted_array(t):
        calls["array"] += 1
        return array_form(t)

    def counted_float(t):
        calls["float"] += 1
        return float_form(t)

    monkeypatch.setattr(exact_trig, "_atan_unit", counted_array)
    monkeypatch.setattr(exact_trig, "_atan_unit_float", counted_float)
    for n in (normal, normals):
        cos = np.minimum(np.abs(np.sum(d * np.broadcast_to(n, d.shape), axis=1)), 1.0)
        calls.update(array=0, float=0)
        exact_trig.incidence_deg(cos, d, n)
        assert calls == {"array": 1, "float": 0}
    for row in (normal, d[1]):
        calls.update(array=0, float=0)
        exact_trig.incidence_deg(min(abs(float(row @ normal)), 1.0), row, normal)
        assert calls == {"array": 0, "float": 1}
