"""An independent reference for ray tracing through two-variable sag surfaces: standard-library ``decimal`` at 60 digits.

Nothing here imports the Engine or the contracts. The sag is coded from the published polar formulas (radial polynomial from
its factorial series, ``cos(m theta)`` / ``sin(m theta)`` by the Chebyshev recurrence from ``cos(theta) = x / r``), the surface
normal comes from a central difference at step ``1e-25`` mm (error of order ``1e-50``, far below the comparison), the ray /
surface intersection is plain bisection on ``z(t) - sag(x(t), y(t))``, and refraction is the vector form of Snell's law.
"""

from __future__ import annotations

from decimal import Decimal as D, localcontext
from math import factorial

DIGITS = 60
STEP = D("1e-25")
BISECTIONS = 220


def noll_order(j: int) -> tuple[int, int]:
    """Noll's (n, m) by walking his ordering: radial order by radial order, |m| ascending, even j the cosine of a pair."""

    index, n = 0, 0
    while True:
        azimuths = list(range(n % 2, n + 1, 2))
        for m in azimuths:
            for _ in range(1 if m == 0 else 2):
                index += 1
                if index == j:
                    return n, 0 if m == 0 else (m if j % 2 == 0 else -m)
        n += 1


def _power(value: D, exponent: int) -> D:
    return value ** exponent if exponent else D(1)


def radial(n: int, m: int, rho: D) -> D:
    total = D(0)
    for s in range((n - m) // 2 + 1):
        total += D((-1) ** s * factorial(n - s)) / D(factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)) * _power(rho, n - 2 * s)
    return total


def zernike_noll(j: int, x: D, y: D, radius: D) -> D:
    """Noll's ``Z_j`` at ``(x, y)``: unit RMS over the disc of ``radius``."""

    n, m = noll_order(j)
    r = (x * x + y * y).sqrt()
    rho = r / radius
    cos_t, sin_t = (x / r, y / r) if r > 0 else (D(1), D(0))
    size = abs(m)
    c_prev, s_prev, c_now, s_now = D(1), D(0), cos_t, sin_t
    if size == 0:
        c_now, s_now = D(1), D(0)
    for _ in range(size - 1):
        c_prev, s_prev, c_now, s_now = c_now, s_now, c_now * cos_t - s_now * sin_t, s_now * cos_t + c_now * sin_t
    if m == 0:
        return D(n + 1).sqrt() * radial(n, 0, rho)
    return D(2 * (n + 1)).sqrt() * radial(n, size, rho) * (c_now if m > 0 else s_now)


def base_sag(x: D, y: D, radius_mm: str | None, conic: str | None) -> D:
    if radius_mm is None:
        return D(0)
    c = 1 / D(radius_mm)
    k = D(conic) if conic is not None else D(0)
    s = x * x + y * y
    return c * s / (1 + (1 - (1 + k) * c * c * s).sqrt())


def zernike_surface(radius_mm: str | None, conic: str | None, norm_radius: str, terms: dict[int, str]):
    r = D(norm_radius)
    values = {j: D(v) for j, v in terms.items()}

    def sag(x: D, y: D) -> D:
        return base_sag(x, y, radius_mm, conic) + sum((v * zernike_noll(j, x, y, r) for j, v in values.items()), D(0))

    return sag


def xy_surface(radius_mm: str | None, conic: str | None, norm_radius: str, terms: dict[tuple[int, int], str]):
    r = D(norm_radius)
    values = {key: D(v) for key, v in terms.items()}

    def sag(x: D, y: D) -> D:
        return base_sag(x, y, radius_mm, conic) + sum((v * _power(x / r, i) * _power(y / r, j) for (i, j), v in values.items()), D(0))

    return sag


def biconic_surface(radius_x: str, radius_y: str, conic_x: str, conic_y: str):
    cx, cy, kx, ky = 1 / D(radius_x), 1 / D(radius_y), D(conic_x), D(conic_y)

    def sag(x: D, y: D) -> D:
        return (cx * x * x + cy * y * y) / (1 + (1 - (1 + kx) * cx * cx * x * x - (1 + ky) * cy * cy * y * y).sqrt())

    return sag


def toroid_surface(radius_y: str, conic: str, y2: str, radius_of_rotation: str):
    """Zemax's construction, taken literally: the curve ``z(y)`` in the YZ plane turned about an axis ``R`` from the vertex:
    ``z = R - sign(R) sqrt((R - z(y))^2 - x^2)``."""

    c, k, a, big_r = 1 / D(radius_y), D(conic), D(y2), D(radius_of_rotation)

    def sag(x: D, y: D) -> D:
        curve = c * y * y / (1 + (1 - (1 + k) * c * c * y * y).sqrt()) + a * y * y
        root = ((big_r - curve) ** 2 - x * x).sqrt()
        return big_r - root if big_r > 0 else big_r + root

    return sag


def _gradient(sag, x: D, y: D) -> tuple[D, D]:
    return (sag(x + STEP, y) - sag(x - STEP, y)) / (2 * STEP), (sag(x, y + STEP) - sag(x, y - STEP)) / (2 * STEP)


def hit_surface(sag, vertex_z: D, origin: tuple[D, D, D], direction: tuple[D, D, D], t_max: D) -> tuple[D, tuple[D, D, D]]:
    """The first crossing of ``z = vertex_z + sag(x, y)`` along the ray, by bisection from ``t = 0`` (below the surface)."""

    def f(t: D) -> D:
        x, y, z = (origin[k] + t * direction[k] for k in range(3))
        return z - vertex_z - sag(x, y)

    low, high = D(0), t_max
    f_low = f(low)
    if f_low * f(high) > 0:
        raise ValueError("the bracket does not hold a crossing")
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        f_middle = f(middle)
        if f_low * f_middle <= 0:
            high = middle
        else:
            low, f_low = middle, f_middle
    t = (low + high) / 2
    return t, tuple(origin[k] + t * direction[k] for k in range(3))  # type: ignore[return-value]


def refract(direction: tuple[D, D, D], normal: tuple[D, D, D], n1: D, n2: D) -> tuple[D, D, D]:
    """Snell's law in vector form; ``normal`` unit, either orientation."""

    cos_i = sum(direction[k] * normal[k] for k in range(3))
    if cos_i < 0:
        normal, cos_i = tuple(-v for v in normal), -cos_i  # type: ignore[assignment]
    ratio = n1 / n2
    sin2_t = ratio * ratio * (1 - cos_i * cos_i)
    cos_t = (1 - sin2_t).sqrt()
    return tuple(ratio * direction[k] + (cos_t - ratio * cos_i) * normal[k] for k in range(3))  # type: ignore[return-value]


def trace_through(surfaces: list[tuple[object, D, D]], origin: tuple[str, str, str], direction: tuple[str, str, str], n_start: str = "1") -> dict:
    """A ray through ``(sag, vertex_z, index after)`` in order, coaxial along +Z: the hit point on every surface and the
    direction after each one, as decimals."""

    with localcontext() as context:
        context.prec = DIGITS
        o = tuple(D(v) for v in origin)
        d = tuple(D(v) for v in direction)
        length = sum(v * v for v in d).sqrt()
        d = tuple(v / length for v in d)
        n = D(n_start)
        hits, directions = [], []
        for sag, vertex_z, n_after in surfaces:
            _, point = hit_surface(sag, D(vertex_z), o, d, D(1000))
            gx, gy = _gradient(sag, point[0], point[1])
            w = (1 + gx * gx + gy * gy).sqrt()
            normal = (-gx / w, -gy / w, 1 / w)
            d = refract(d, normal, n, D(n_after))
            hits.append(point)
            directions.append(d)
            n = D(n_after)
            o = point
        return {"hits": hits, "directions": directions}
