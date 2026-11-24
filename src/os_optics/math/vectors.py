"""Vector helpers (NumPy only). All functions are pure and vectorized over a leading axis.

Zero-length vectors never produce NaN: ``normalize`` returns an ``ok`` mask and a zero vector for
the failing rows; callers decide how to report it.

Same bits in every runtime: ``dot``, ``matvec`` and ``matmul`` sum separately rounded elementwise products in index order.
NumPy's ``einsum`` and ``matmul`` (``@`` on stacked or 2-D operands) may fuse a product and a sum into one fused
multiply-add, or use another summation order, in a native build on a CPU with FMA, and do not in WebAssembly (Pyodide), so
their last bit differs between runtimes; geometry that reaches a result (intersections, path lengths, directions, frames)
uses these helpers instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_F64 = np.dtype(np.float64)


def correctly_rounded_sum(values: Iterable[float]) -> float:
    """The sum of ``values`` correctly rounded (``math.fsum``): the same bits on every Python and runtime, in any order. The
    built-in ``sum`` compensates floats from Python 3.12 and folds them before, so its last bits depend on the interpreter.
    Where ``fsum`` has no finite answer to give (an overflow, or ``inf - inf``) the sum is the IEEE result of adding them."""
    items = list(values)
    try:
        return math.fsum(items) if items else 0.0
    except (OverflowError, ValueError):
        total = 0.0
        for value in items:
            total = total + value
        return total


def as_vec3(v: object) -> Array:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"expected shape (3,), got {a.shape}")
    return a


def _dot_floats(x: list[float], y: list[float]) -> float:
    """``x_0 y_0 + x_1 y_1 + ...`` in Python floats, summed in index order (``dot``'s operations)."""
    total = x[0] * y[0]
    for i in range(1, len(x)):
        total = total + x[i] * y[i]
    return total


def cross(a: Array, b: Array) -> Array:
    """``a x b`` over the last axis (broadcast over the others): ``numpy.cross``'s operations, ``(a1 b2 - a2 b1, a2 b0 -
    a0 b2, a0 b1 - a1 b0)`` from separately rounded products, so the same bits, without its per-call overhead (one vector or
    one row is computed in Python floats; float64 rows component by component). Other operands go to ``numpy.cross``."""
    if type(a) is np.ndarray and type(b) is np.ndarray and a.dtype == _F64 and b.dtype == _F64 \
            and a.ndim in (1, 2) and b.ndim in (1, 2) and a.shape[-1] == 3 and b.shape[-1] == 3:
        rows_a, rows_b = (a.shape[0] if a.ndim == 2 else 0), (b.shape[0] if b.ndim == 2 else 0)
        if (a.ndim == 1 or rows_a == 1) and (b.ndim == 1 or rows_b == 1):
            a0, a1, a2 = (a[0] if a.ndim == 2 else a).tolist()
            b0, b1, b2 = (b[0] if b.ndim == 2 else b).tolist()
            out = [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0]
            return np.array([out] if (a.ndim == 2 or b.ndim == 2) else out)
        if a.ndim == 2 and (b.ndim == 1 or rows_b == rows_a) or b.ndim == 2 and a.ndim == 1:
            a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
            b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
            out = np.empty((max(rows_a, rows_b), 3))
            np.multiply(a1, b2, out=out[:, 0])
            out[:, 0] -= a2 * b1
            np.multiply(a2, b0, out=out[:, 1])
            out[:, 1] -= a0 * b2
            np.multiply(a0, b1, out=out[:, 2])
            out[:, 2] -= a1 * b0
            return out
    return np.cross(a, b)


def dot(a: Array, b: Array) -> Array:
    """``sum_i a_i b_i`` over the last axis (broadcast over the others), summed in index order from separately rounded
    elementwise products (no fused multiply-add, no BLAS).

    Two float64 vectors are summed in Python floats: each ``*`` and ``+`` is the same correctly rounded IEEE operation as
    NumPy's, so the result has the same bits, without NumPy's per-call overhead; so is one row ``(1, k)`` against one row or
    one vector (a one-ray packet). Rows ``(n, k)`` against rows of the same shape or one vector ``(k,)`` form all products
    in one array operation and add its columns in index order (the same products and sums, fewer array operations). A NaN
    result is a NaN in every form; its sign bit is not fixed (adding two NaNs keeps either one's sign)."""
    if type(a) is np.ndarray and type(b) is np.ndarray and a.dtype == _F64 and b.dtype == _F64:
        if a.ndim == 1 and a.shape == b.shape and a.shape[0]:
            return np.float64(_dot_floats(a.tolist(), b.tolist()))
        if a.ndim == 2 and a.shape[0] == 1 and a.shape[1] and (b.shape == a.shape or b.shape == a.shape[1:]):
            return np.array([_dot_floats(a[0].tolist(), (b[0] if b.ndim == 2 else b).tolist())])
        if b.ndim == 2 and b.shape[0] == 1 and b.shape[1] and a.shape == b.shape[1:]:
            return np.array([_dot_floats(a.tolist(), b[0].tolist())])
        if (a.ndim == 2 and (b.shape == a.shape or b.shape == a.shape[1:]) and a.shape[1] > 1
                or b.ndim == 2 and a.shape == b.shape[1:] and b.shape[1] > 1):
            products = a * b
            total = products[:, 0] + products[:, 1]
            for i in range(2, products.shape[1]):
                total += products[:, i]
            return total
        if a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0] and b.shape[0]:
            y = b.tolist()
            total = a[:, 0] * y[0]
            for i in range(1, len(y)):
                total = total + a[:, i] * y[i]
            return total
        if b.ndim == 2 and a.ndim == 1 and b.shape[1] == a.shape[0] and a.shape[0]:
            x = a.tolist()
            total = x[0] * b[:, 0]
            for i in range(1, len(x)):
                total = total + x[i] * b[:, i]
            return total
    a = np.asarray(a)
    b = np.asarray(b)
    total = a[..., 0] * b[..., 0]
    for i in range(1, max(a.shape[-1], b.shape[-1])):
        total = total + a[..., i] * b[..., i]
    return total


#: ``|d x n|`` (the sine of the angle of incidence) at or below which an interface has no resolved plane of incidence. A
#: cross product of two unit vectors carries an absolute error of a few 1e-16 per component, so the direction of ``d x n``
#: is uncertain by about 1e-16 / ``|d x n|`` (a design turned as a whole about an oblique axis, or written with rounded
#: decimal positions, meets an exactly normal surface at 1e-16 .. 1e-10). Below this bound the s and p coefficients of an
#: interface differ from their common normal-incidence value by at most first order in the angle (< 1e-6 rad), so the
#: ray's own transverse axis ``e1`` -- carried from the source's frame through every interaction, hence turned with the
#: design -- is used as s instead (``incidence_s_axis``); the result is then the same for a design and for that design
#: turned rigidly. Above it the plane of incidence is resolved to ``TRANSPORT_BASIS_RESOLUTION``.
NORMAL_INCIDENCE_SIN = 1e-6
#: The absolute resolution of what follows the plane of incidence of a beam that met a surface near normal incidence: the
#: components of its transport basis (``basis.e1`` / ``e2``), of its Jones vector in that basis, of its Gaussian axes and its
#: Stokes ``s1`` / ``s2``. The s axis ``d x n / |d x n|`` turns by up to 8 units of roundoff over ``|d x n|`` with the last
#: bits of the poses and of the surfaces met, at most ``8 * 2^-52 / NORMAL_INCIDENCE_SIN`` = 1.8e-9 rad (4.4e-6 from normal,
#: a fold prism's beam on a lens: 3e-11). The lab-frame field, the powers, positions and directions do not follow it: the
#: s and p coefficients there differ only at second order in the angle. A larger bound would resolve the basis better
#: but give the normal-incidence rules to beams up to that angle, whose s and p (and tangential and sagittal Gaussian)
#: treatments would then follow the beam's own axes instead of the plane of incidence (1e-3: Gaussian radii and footprints
#: of beams 1e-5 .. 1e-3 from normal move by 1e-6 relative with the poses' last bits).
TRANSPORT_BASIS_RESOLUTION = 8 * 2.0**-52 / NORMAL_INCIDENCE_SIN


def incidence_s_axis(d: Array, n: Array, fallback: Array, zero_tol: float) -> tuple[Array, NDArray[np.bool_]]:
    """Unit ``s`` axes (rows) perpendicular to the plane of incidence of directions ``d`` on normals ``n``: ``d x n``
    normalized where ``|d x n| > NORMAL_INCIDENCE_SIN``, else ``fallback`` (the ray's ``e1``); either made exactly
    perpendicular to ``d``. Returns ``(s, resolved)``: ``resolved`` is False where the plane of incidence is not resolved
    (normal incidence)."""
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    dxn = cross(d, np.atleast_2d(np.asarray(n, dtype=np.float64)))
    size = np.sqrt(dot(dxn, dxn))
    resolved = size > NORMAL_INCIDENCE_SIN
    s = np.where(resolved[:, None], dxn / np.where(resolved, size, 1.0)[:, None], np.atleast_2d(fallback))
    s, ok = normalize(s - dot(s, d)[:, None] * d, zero_tol)
    return s, resolved & ok


def matvec(m: Array, v: Array) -> Array:
    """``m v`` for matrices ``(..., r, c)`` and vectors ``(..., c)`` (broadcast), each row a ``dot``: entry ``i`` is
    ``(m_i0 v_0 + m_i1 v_1) + ...``, accumulated over ``k`` for all rows at once (C-contiguous result)."""
    m = np.asarray(m)
    v = np.asarray(v)
    if m.ndim == 2 and v.ndim == 1 and type(m) is np.ndarray and type(v) is np.ndarray and m.dtype == _F64 and v.dtype == _F64 \
            and m.shape[1] == v.shape[0] and v.shape[0]:
        y = v.tolist()
        rows = []
        for row in m.tolist():
            total = row[0] * y[0]
            for k in range(1, len(y)):
                total = total + row[k] * y[k]
            rows.append(total)
        return np.array(rows, dtype=np.float64)
    total = m[..., :, 0] * v[..., None, 0]
    for k in range(1, max(m.shape[-1], v.shape[-1])):
        total = total + m[..., :, k] * v[..., None, k]
    return np.ascontiguousarray(total)


def matmul(a: Array, b: Array) -> Array:
    """``a b`` for matrices ``(..., r, k)`` and ``(..., k, c)`` (broadcast), each entry a ``dot``: entry ``(i, j)`` is
    ``(a_i0 b_0j + a_i1 b_1j) + ...``, accumulated over ``k`` for all entries at once (C-contiguous result)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2 and b.ndim == 2 and type(a) is np.ndarray and type(b) is np.ndarray and a.dtype == _F64 and b.dtype == _F64 \
            and a.shape[1] == b.shape[0] and a.shape[1]:
        cols = b.T.tolist()
        out = []
        for row in a.tolist():
            entries = []
            for col in cols:
                total = row[0] * col[0]
                for k in range(1, len(row)):
                    total = total + row[k] * col[k]
                entries.append(total)
            out.append(entries)
        return np.array(out, dtype=np.float64).reshape(a.shape[0], b.shape[1])
    total = a[..., :, 0, None] * b[..., None, 0, :]
    for k in range(1, max(a.shape[-1], b.shape[-2])):
        total = total + a[..., :, k, None] * b[..., None, k, :]
    return np.ascontiguousarray(total)


def norm(v: Array) -> Array:
    return np.sqrt(dot(v, v))


def normalize(v: Array, zero_tol: float) -> tuple[Array, Array]:
    """Return ``(unit, ok)``. Rows with ``|v| <= zero_tol`` (or non-finite) give a zero row and ``ok=False``.

    One row ``(1, k)`` (a one-ray packet) with ``zero_tol >= 0`` is computed in Python floats: the same ``dot``, correctly
    rounded square root and divisions, so the same bits."""
    if type(v) is np.ndarray and v.dtype == _F64 and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] and type(zero_tol) is float \
            and zero_tol >= 0.0:
        row = v[0].tolist()
        size = math.sqrt(_dot_floats(row, row))
        if math.isfinite(size) and size > zero_tol:
            return np.array([[x / size for x in row]]), np.array([True])
        return np.zeros(v.shape), np.array([False])
    n = norm(v)
    ok = np.isfinite(n) & (n > zero_tol)
    safe = np.where(ok, n, 1.0)
    unit = np.where(ok[..., None], v / safe[..., None], 0.0)
    return unit, ok


def transverse_basis(d: Array, zero_tol: float) -> tuple[Array, Array]:
    """Deterministic right-handed transverse basis ``(e1, e2)`` with ``e1 x e2 = d``.

    The helper axis is the coordinate axis least aligned with ``d`` (first minimal component, so ties are
    resolved deterministically). ``d`` must be unit length.
    """
    d = np.atleast_2d(d)
    idx = np.argmin(np.abs(d), axis=-1)
    helper = np.zeros_like(d)
    helper[np.arange(d.shape[0]), idx] = 1.0
    e1, _ = normalize(cross(helper, d), zero_tol)
    e2 = cross(d, e1)
    return e1, e2


#: ``1 + a.b`` below which ``rotation_matrices_between`` takes the axis-angle form: Rodrigues' ``[v]x^2 / (1 + c)`` leaves the
#: matrix about ``eps / (1 + c)`` off orthonormal (8e-13 at ``1 + c`` = 1.25e-3), the axis-angle form a few ``eps``.
ROTATION_AXIS_FORM_BELOW = 0.1


def rotation_matrices_between(a: Array, b: Array, antiparallel_tol: float, fallback_axis: Array) -> Array:
    """Rotation matrices (N,3,3) taking unit vectors ``a`` onto unit vectors ``b`` (minimal rotation).

    Rodrigues form ``I + [v]x + [v]x^2 / (1 + c)`` with ``v = a x b``, ``c = a.b``; stable as ``c -> 1``. Where ``1 + c``
    is below ``ROTATION_AXIS_FORM_BELOW`` (``b`` nearly ``-a``: ``1 + c`` there carries the rounding of ``c``, a relative
    error of ``eps / (1 + c)`` that left the matrix up to 7e-4 off orthonormal next to ``antiparallel_tol``), the same
    rotation in its axis-angle form ``c I + [v]x + (1 - c) k k^T``, ``k = v / |v|``, which stays orthonormal to rounding.
    For ``1 + c <= antiparallel_tol`` the rotation is by pi about ``fallback_axis``: ``R = 2 u u^T - I``, ``u`` its part
    perpendicular to ``a`` made unit (a fallback axis is unit and perpendicular only to rounding, and ``R`` would then
    not be orthogonal; a fallback axis along ``a``: the transverse basis's ``e1``). ``fallback_axis`` is per-row (N,3).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    n = a.shape[0]
    c = dot(a, b)
    v = cross(a, b)
    vx = np.zeros((n, 3, 3))
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    anti = (1.0 + c) <= antiparallel_tol
    denom = np.where(anti, 1.0, 1.0 + c)
    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    rot = eye + vx + matmul(vx, vx) / denom[:, None, None]
    near = ~anti & ((1.0 + c) < ROTATION_AXIS_FORM_BELOW)
    if np.any(near):
        size = np.sqrt(dot(v, v))
        k = v / np.where(size > 0.0, size, 1.0)[:, None]
        axis_angle = c[:, None, None] * eye + vx + (1.0 - c)[:, None, None] * (k[:, :, None] * k[:, None, :])
        rot = np.where(near[:, None, None], axis_angle, rot)
    if np.any(anti):
        # (the fallback axis made exactly perpendicular to ``a`` and unit: with ``|u|^2 = 1 + e`` the matrix scales ``u`` by
        # ``1 + 2e``, so a beam reflected back on itself again and again would grow its transported frame ~5-fold in ``e``
        # per bounce)
        f = np.broadcast_to(np.atleast_2d(fallback_axis), a.shape)
        u = f - dot(f, a)[:, None] * a
        size = np.sqrt(dot(u, u))
        helper, _e2 = transverse_basis(a, 0.0)
        u = np.where((size > 0.0)[:, None], u / np.where(size > 0.0, size, 1.0)[:, None], helper)
        pi_rot = 2.0 * (u[:, :, None] * u[:, None, :]) - eye
        rot = np.where(anti[:, None, None], pi_rot, rot)
    return rot


def parallel_transport_basis(
    e1: Array, e2: Array, d_old: Array, d_new: Array, antiparallel_tol: float
) -> tuple[Array, Array]:
    """Rotation-minimizing transport of the transverse basis ``(e1,e2)`` from ``d_old`` to ``d_new``.

    Jones components are unchanged by this transport (no polarization rotation). For antiparallel
    directions the rotation axis is ``e1`` (deterministic)."""
    rot = rotation_matrices_between(d_old, d_new, antiparallel_tol, fallback_axis=e1)
    n1 = matvec(rot, e1)
    n2 = matvec(rot, e2)
    return n1, n2


def reflect(d: Array, n: Array) -> Array:
    """Mirror reflection ``d - 2 (d.n) n`` (n unit)."""
    return d - 2.0 * dot(d, n)[..., None] * n
