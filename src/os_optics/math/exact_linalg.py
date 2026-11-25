"""Small complex linear algebra from ``+ - x /`` and ``sqrt`` only, for values that must have the same bits in every runtime.

LAPACK (``numpy.linalg``) is a different library in CPython builds (OpenBLAS, MKL, Accelerate) and in Pyodide, and its
results may differ in the last bits. The network export reports singular values and element Jones matrices; these helpers
compute them deterministically. Accuracy: about ``1e-15`` relative to the largest singular value.

Complex products: numpy's complex multiply and BLAS dot products may fuse ``a b - c d`` into one fused multiply-add on a
CPU with FMA (native x86-64 builds) and not in WebAssembly (Pyodide), so the last bit of a complex product can differ.
``cmul`` and ``power2`` form them from separate real multiplications and additions (each correctly rounded on its own):
the Jones arithmetic of the traced beams uses them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

CArray = NDArray[np.complex128]
Array = NDArray[np.float64]

#: One-sided Jacobi: a column pair counts as orthogonal when |a_p^H a_q| <= this x ||a_p|| ||a_q||.
JACOBI_ORTHOGONALITY = 1e-15
#: Sweeps over every column pair (quadratic convergence: a handful suffice for the small matrices exported).
JACOBI_MAX_SWEEPS = 60


def symmetric_eigenvalues_2x2(a: float, b: float, c: float) -> tuple[float, float]:
    """Eigenvalues (ascending) of the real symmetric matrix ``[[a, b], [b, c]]`` from ``+ - x /`` and ``sqrt`` only (identical
    bits in every runtime, unlike LAPACK's): the larger from the mean and the half-difference, the smaller from the
    determinant when the sum would cancel."""

    mean = 0.5 * (a + c)
    half = 0.5 * (a - c)
    root = math.sqrt(half * half + b * b)
    big = mean + root
    small = (a * c - b * b) / big if big > 0.0 else mean - root
    return small, big


def abs2(z: CArray) -> Array:
    """``|z|^2`` elementwise (``re^2 + im^2``: no ``hypot``)."""

    z = np.asarray(z, dtype=np.complex128)
    return z.real * z.real + z.imag * z.imag


def cmul(a: CArray | complex, b: CArray | complex) -> CArray:
    """``a * b`` elementwise (complex), from separately rounded real products: no fused multiply-add."""

    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    out = np.empty(np.broadcast(a, b).shape, dtype=np.complex128)
    out.real = np.subtract(np.multiply(ar, br), np.multiply(ai, bi))
    out.imag = np.add(np.multiply(ar, bi), np.multiply(ai, br))
    return out


def power2(jones: CArray) -> float:
    """``|J1|^2 + |J2|^2`` of one Jones vector, summed in a fixed order from separate real products."""

    x, y = (jones.tolist() if type(jones) is np.ndarray and jones.shape == (2,) else (jones[0], jones[1]))
    x, y = complex(x), complex(y)
    return (x.real * x.real + x.imag * x.imag) + (y.real * y.real + y.imag * y.imag)


def cvdot(a: CArray, b: CArray) -> complex:
    """``sum(conj(a_i) b_i)`` of two complex vectors (``numpy.vdot``), summed in index order from separately rounded real
    products (no BLAS, no fused multiply-add)."""

    re_sum, im_sum = 0.0, 0.0
    for x, y in zip(np.asarray(a, dtype=np.complex128).ravel(), np.asarray(b, dtype=np.complex128).ravel()):
        xr, xi, yr, yi = float(x.real), float(x.imag), float(y.real), float(y.imag)
        re_sum += xr * yr + xi * yi
        im_sum += xr * yi - xi * yr
    return complex(re_sum, im_sum)


def singular_values(matrix: CArray) -> Array:
    """The ``min(m, n)`` singular values of an ``m x n`` complex matrix, largest first: one-sided (Hestenes) Jacobi
    rotations orthogonalize the columns (of the conjugate transpose when ``n > m``), the singular values are the column
    norms."""

    a = np.array(matrix, dtype=np.complex128)
    if a.size == 0:
        return np.zeros(0)
    if a.shape[1] > a.shape[0]:
        a = a.conj().T.copy()
    n = a.shape[1]
    for _ in range(JACOBI_MAX_SWEEPS):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                ap, aq = a[:, p], a[:, q]
                alpha = float(np.sum(abs2(ap)))
                beta = float(np.sum(abs2(aq)))
                gamma = complex(np.sum(ap.conj() * aq))
                g = math.sqrt(gamma.real * gamma.real + gamma.imag * gamma.imag)
                if g == 0.0 or g <= JACOBI_ORTHOGONALITY * math.sqrt(alpha * beta):
                    continue
                rotated = True
                phase = gamma / g  # e^{i phi}: a_q e^{-i phi} makes the pair's inner product real (g)
                zeta = (beta - alpha) / (2.0 * g)
                t = (1.0 if zeta >= 0.0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                aq_real = aq * phase.conjugate()
                a[:, p], a[:, q] = c * ap - s * aq_real, s * ap + c * aq_real
        if not rotated:
            break
    norms = np.sqrt(np.sum(abs2(a), axis=0))
    return np.sort(norms)[::-1][: min(matrix.shape)]


def inverse_2x2(m: CArray) -> CArray | None:
    """The inverse of a complex 2 x 2 matrix (``None`` when singular)."""

    a, b, c, d = complex(m[0, 0]), complex(m[0, 1]), complex(m[1, 0]), complex(m[1, 1])
    det = a * d - b * c
    if det == 0:
        return None
    return np.array([[d / det, -b / det], [-c / det, a / det]], dtype=np.complex128)


def det2(m: Array) -> float:
    """Determinant of a real 2 x 2 matrix (two separately rounded products: no LAPACK)."""

    return float(m[0, 0]) * float(m[1, 1]) - float(m[0, 1]) * float(m[1, 0])


def solve2(m: Array, b: NDArray) -> NDArray:
    """``m^-1 b`` of a regular real 2 x 2 matrix and a (real or complex) 2-vector, by Cramer's rule (no LAPACK)."""

    det = det2(m)
    m00, m01, m10, m11 = float(m[0, 0]), float(m[0, 1]), float(m[1, 0]), float(m[1, 1])
    b0, b1 = (complex(b[0]), complex(b[1])) if np.iscomplexobj(b) else (float(b[0]), float(b[1]))  # plain Python arithmetic
    return np.array([(m11 * b0 - m01 * b1) / det, (m00 * b1 - m10 * b0) / det])


def eigvalsh2(m: Array) -> tuple[float, float]:
    """Eigenvalues (ascending) of a real symmetric 2 x 2 matrix in closed form (no LAPACK)."""

    a, b, d = float(m[0, 0]), 0.5 * (float(m[0, 1]) + float(m[1, 0])), float(m[1, 1])
    mean, half = 0.5 * (a + d), 0.5 * (a - d)
    r = math.sqrt(half * half + b * b)
    return mean - r, mean + r


__all__ = ["abs2", "cmul", "cvdot", "det2", "eigvalsh2", "inverse_2x2", "power2", "singular_values", "solve2"]
