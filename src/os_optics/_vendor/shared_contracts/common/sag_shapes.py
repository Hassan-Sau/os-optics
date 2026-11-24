"""Two-variable sag shapes: Zernike, XY polynomial, biconic, toroid (pure arithmetic, no I/O, no libm trigonometry).

One kernel, written with ``+ - * /`` and one injected square root, evaluates a shape on ``float``, on arrays and on
``Decimal``: the contract checks, the Engine's tracer and the display meshes compute the same formulas. ``x``, ``y`` are the
surface's local transverse coordinates (mm), ``z`` the sag along its axis (mm).

Zernike (``kind = "ZERNIKE"``). ``z = sum value_j Z_j(rho, theta)``, ``rho = sqrt(x^2 + y^2) / R``, ``x = R rho cos(theta)``,
``y = R rho sin(theta)``, ``R`` the normalization radius, with ``Z_j = N R_n^m(rho) cos(m theta)`` (``m > 0``), ``N R_n^|m|
sin(|m| theta)`` (``m < 0``), ``N R_n^0`` (``m = 0``):

    R_n^m(rho) = sum_k (-1)^k (n - k)! / (k! ((n + m)/2 - k)! ((n - m)/2 - k)!) rho^(n - 2k)

* ``indexing = NOLL``, ``normalization = RMS_ORTHONORMAL``: Noll's single index (R. J. Noll, J. Opt. Soc. Am. 66, 207, 1976),
  ``N = sqrt(n + 1)`` for ``m = 0`` and ``sqrt(2 (n + 1))`` otherwise, so each term has unit RMS over the unit disc;
* ``indexing = FRINGE``, ``normalization = PEAK_UNIT``: the 37-term Fringe (University of Arizona) ordering, ``N = 1``, so each
  term has unit peak.

Each term is expanded once into integer monomials of ``(x / R, y / R)`` (``zernike_monomials``): ``rho^(2q) = (X^2 + Y^2)^q``,
``rho^m cos(m theta) = Re (X + i Y)^m`` and ``rho^m sin(m theta) = Im (X + i Y)^m``, so no trigonometric function is ever
evaluated. The terms are not referenced to the vertex: a term with ``m = 0`` moves the point at ``(0, 0)`` by
``value N R_n^0(0)`` and a coma term tilts the surface there. Piston and tilt (``n <= 1``) are not accepted: they state a
pose, not a shape.

XY polynomial (``kind = "XY_POLYNOMIAL"``). ``z = sum value (x / R)^i (y / R)^j`` over ``i + j >= 2``, ``R`` the normalization
radius; each coefficient is a length (mm).

Biconic (``kind = "BICONIC"``). ``z = (c_x x^2 + c_y y^2) / (1 + S)``, ``S = sqrt(1 - (1 + k_x) c_x^2 x^2 - (1 + k_y) c_y^2 y^2)``,
``c_x = 1 / radius_x_mm``, ``c_y = 1 / radius_y_mm`` (0 when absent), ``k_x``, ``k_y`` the conic constants.

Toroid (``kind = "TOROIDAL"``). A curve in the YZ plane, ``z_y(y) = c y^2 / (1 + T) + sum_i alpha_i y^(2i)``,
``T = sqrt(1 - (1 + k) c^2 y^2)``, ``c = 1 / radius_y_mm`` (0 when absent), ``alpha_1, alpha_2, ... = y_terms``, rotated about
the axis parallel to Y through ``z = R`` (``R = radius_of_rotation_mm``; absent: no rotation, a surface straight along X):

    z = z_y + p x^2 / (1 + V),     p = 1 / (R - z_y),     V = sqrt(1 - p^2 x^2)

the cancellation-free form of ``z = R - sign(R) sqrt((R - z_y)^2 - x^2)``.

Slopes are returned with the root multiplied through, ``(z, h_x, h_y, w, ok)``: the normal on the ``+z`` side is parallel to
``(-h_x, -h_y, w)`` and ``dz/dx = h_x / w``, ``dz/dy = h_y / w``, finite at the edge of the domain (``w = 0``). ``ok`` says
where the shape is defined; ``z`` and the slopes are meaningless elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import comb, factorial
from typing import Any

ZERNIKE = "ZERNIKE"
XY_POLYNOMIAL = "XY_POLYNOMIAL"
BICONIC = "BICONIC"
TOROIDAL = "TOROIDAL"
#: The shapes that add to a surface's radius, conic and aspheric terms (biconic and toroid replace them).
ADDITIVE_KINDS = frozenset({ZERNIKE, XY_POLYNOMIAL})
NOLL = "NOLL"
FRINGE = "FRINGE"
RMS_ORTHONORMAL = "RMS_ORTHONORMAL"
PEAK_UNIT = "PEAK_UNIT"
#: The normalization each indexing is published with; another pairing would silently rescale every coefficient.
NORMALIZATION_OF_INDEXING = {NOLL: RMS_ORTHONORMAL, FRINGE: PEAK_UNIT}
#: Highest radial order of a Zernike term and of an XY monomial (``i + j``); Noll index 231 is the last of radial order 20.
MAX_RADIAL_ORDER = 20
MAX_NOLL_INDEX = 231
MAX_FRINGE_INDEX = 37
#: The most terms one shape holds (all monomials of order 2 .. 20, or Noll 4 .. 231).
MAX_SAG_TERMS = 231
#: The most ``y_terms`` of a toroid: ``y^2`` .. ``y^16`` (the even-asphere column count plus the ``y^2`` term).
MAX_TOROID_Y_TERMS = 8

# Reason codes of a refused shape: the first word of the message.
SAG_COMBINATION_NOT_SUPPORTED = "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"
SAG_NORMALIZATION_MISMATCH = "SAG_NORMALIZATION_MISMATCH"
SAG_TERM_NOT_SUPPORTED = "SAG_TERM_NOT_SUPPORTED"
SAG_APERTURE_BEYOND_NORMALIZATION = "SAG_APERTURE_BEYOND_NORMALIZATION"
SAG_OUTSIDE_DOMAIN = "SAG_OUTSIDE_DOMAIN"

#: ``(n, m)`` of Fringe index 1 .. 37 (``m > 0`` cos, ``m < 0`` sin).
FRINGE_ORDERS: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0), (3, 3), (3, -3), (4, 2), (4, -2), (5, 1), (5, -1), (6, 0),
    (4, 4), (4, -4), (5, 3), (5, -3), (6, 2), (6, -2), (7, 1), (7, -1), (8, 0), (5, 5), (5, -5), (6, 4), (6, -4), (7, 3), (7, -3), (8, 2),
    (8, -2), (9, 1), (9, -1), (10, 0), (12, 0),
)


# ------------------------------------------------------------------------------------------- Zernike as integers


def noll_order(index: int) -> tuple[int, int]:
    """``(n, m)`` of Noll index ``index >= 1``: radial order ``n`` from ``n (n + 1) / 2 < index <= (n + 1) (n + 2) / 2``, ``|m|``
    ascending within it, even ``index`` the cosine term and odd ``index`` the sine term of a pair."""

    n = 0
    while (n + 1) * (n + 2) // 2 < index:
        n += 1
    position = index - n * (n + 1) // 2 - 1
    m = (position + 1) // 2 * 2 if n % 2 == 0 else position // 2 * 2 + 1
    if m == 0 or index % 2 == 0:
        return n, m
    return n, -m


def fringe_order(index: int) -> tuple[int, int]:
    """``(n, m)`` of Fringe index ``1 .. 37``."""

    return FRINGE_ORDERS[index - 1]


def zernike_order(indexing: str, index: int) -> tuple[int, int]:
    return noll_order(index) if indexing == NOLL else fringe_order(index)


def zernike_monomials(n: int, m: int) -> tuple[tuple[int, int, int], ...]:
    """``R_n^|m| cos / sin`` of the module docstring as integer monomials ``(i, j, c)``: ``sum c X^i Y^j``, sorted by ``(i, j)``,
    without zero coefficients."""

    size = abs(m)
    angular: list[tuple[int, int, int]] = []  # Re / Im (X + i Y)^|m|
    for k in range(size + 1):
        if (k % 2 == 1) == (m < 0):
            angular.append((size - k, k, comb(size, k) * (-1) ** (k // 2)))
    terms: dict[tuple[int, int], int] = {}
    for s in range((n - size) // 2 + 1):
        radial = (-1) ** s * factorial(n - s) // (factorial(s) * factorial((n + size) // 2 - s) * factorial((n - size) // 2 - s))
        q = (n - 2 * s - size) // 2  # (X^2 + Y^2)^q
        for t in range(q + 1):
            for i, j, c in angular:
                key = (i + 2 * (q - t), j + 2 * t)
                terms[key] = terms.get(key, 0) + radial * comb(q, t) * c
    return tuple((i, j, c) for (i, j), c in sorted(terms.items()) if c != 0)


def zernike_norm_squared(normalization: str, n: int, m: int) -> int:
    """``N^2`` of the module docstring."""

    if normalization == PEAK_UNIT:
        return 1
    return n + 1 if m == 0 else 2 * (n + 1)


# ------------------------------------------------------------------------------------------------ term checks


def zernike_terms_problem(indexing: str, normalization: str, indices: Sequence[int]) -> str | None:
    """Why these Zernike terms are not accepted, with its reason code (``None``: they are)."""

    expected = NORMALIZATION_OF_INDEXING[indexing]
    if normalization != expected:
        return (f"{SAG_NORMALIZATION_MISMATCH}: {indexing} coefficients are published with {expected} normalization, not {normalization}; "
                "another pairing would rescale every coefficient")
    limit = MAX_NOLL_INDEX if indexing == NOLL else MAX_FRINGE_INDEX
    if len(set(indices)) != len(indices):
        return f"{SAG_TERM_NOT_SUPPORTED}: a {indexing} index appears twice"
    for index in indices:
        if not 1 <= index <= limit:
            return f"{SAG_TERM_NOT_SUPPORTED}: {indexing} index {index} is outside 1 .. {limit}"
        if zernike_order(indexing, index)[0] <= 1:
            return (f"{SAG_TERM_NOT_SUPPORTED}: {indexing} index {index} is piston or tilt, which moves the surface as a whole: state it "
                    "in the surface's pose")
    return None


def xy_terms_problem(powers: Sequence[tuple[int, int]]) -> str | None:
    """Why these XY monomial powers are not accepted, with its reason code (``None``: they are)."""

    if len(set(powers)) != len(powers):
        return f"{SAG_TERM_NOT_SUPPORTED}: an XY monomial appears twice"
    for i, j in powers:
        if i < 0 or j < 0 or i + j > MAX_RADIAL_ORDER:
            return f"{SAG_TERM_NOT_SUPPORTED}: x^{i} y^{j} is outside the powers i, j >= 0, i + j <= {MAX_RADIAL_ORDER}"
        if i + j < 2:
            return f"{SAG_TERM_NOT_SUPPORTED}: x^{i} y^{j} is piston or tilt, which moves the surface as a whole: state it in the surface's pose"
    return None


# ------------------------------------------------------------------------------------------------- arithmetic


@dataclass(frozen=True)
class SagOps:
    """How numbers are made and rooted: ``number`` converts a published value, ``root`` is ``sqrt(max(value, 0))``."""

    number: Callable[[Any], Any]
    root: Callable[[Any], Any]


def _float_root(value: float) -> float:
    return math.sqrt(value) if value > 0.0 else 0.0


def _decimal_root(value: Decimal) -> Decimal:
    return value.sqrt() if value > 0 else Decimal(0)


#: Plain floats (``math.sqrt`` is correctly rounded on every runtime).
FLOAT_OPS = SagOps(float, _float_root)
#: Exact decimals: the caller holds the decimal context.
DECIMAL_OPS = SagOps(Decimal, _decimal_root)


@dataclass(frozen=True)
class SagForm:
    """A shape ready to evaluate: one of the polynomial (``terms``: ``(i, j, coefficient)`` of ``(x scale)^i (y scale)^j``), the
    biconic (``curvature_x``, ``curvature_y`` and the conic factors ``conic_x = (1 + k_x) c_x^2``, ``conic_y``) or the toroid
    (``curvature``, ``conic_y = (1 + k) c^2``, ``y_terms``, ``rotation`` the radius of rotation, ``None`` for none)."""

    kind: str
    terms: tuple[tuple[int, int, Any], ...] = ()
    scale: Any = None
    curvature_x: Any = None
    curvature_y: Any = None
    conic_x: Any = None
    conic_y: Any = None
    curvature: Any = None
    y_terms: tuple[Any, ...] = ()
    rotation: Any = None

    @property
    def additive(self) -> bool:
        return self.kind in ADDITIVE_KINDS


def _inverse(value: Any, ops: SagOps) -> Any:
    return ops.number(0) if value is None else 1 / ops.number(value)


def _conic_factor(curvature: Any, conic: Any, ops: SagOps) -> Any:
    k = ops.number(0) if conic is None else ops.number(conic)
    return (1 + k) * curvature * curvature


def sag_form(shape: Any, ops: SagOps = FLOAT_OPS) -> SagForm:
    """The form of a contract shape (any object with the fields of ``ZernikeSagV1`` .. ``ToroidalSagV1``; its numbers may be
    decimal strings or floats)."""

    if shape.kind == BICONIC:
        c_x, c_y = _inverse(shape.radius_x_mm, ops), _inverse(shape.radius_y_mm, ops)
        return SagForm(BICONIC, curvature_x=c_x, curvature_y=c_y, conic_x=_conic_factor(c_x, shape.conic_x, ops),
                       conic_y=_conic_factor(c_y, shape.conic_y, ops))
    if shape.kind == TOROIDAL:
        c = _inverse(shape.radius_y_mm, ops)
        return SagForm(TOROIDAL, curvature=c, conic_y=_conic_factor(c, shape.conic, ops),
                       y_terms=tuple(ops.number(a) for a in shape.y_terms or ()),
                       rotation=None if shape.radius_of_rotation_mm is None else ops.number(shape.radius_of_rotation_mm))
    scale = 1 / ops.number(shape.normalization_radius_mm)
    if shape.kind == XY_POLYNOMIAL:
        merged = {(t.x_power, t.y_power): ops.number(t.value_mm) for t in shape.terms}
    else:
        merged = {}
        for term in shape.terms:
            n, m = zernike_order(shape.indexing, term.index)
            weight = ops.number(term.value_mm) * ops.root(ops.number(zernike_norm_squared(shape.normalization, n, m)))
            for i, j, c in zernike_monomials(n, m):
                merged[(i, j)] = merged.get((i, j), ops.number(0)) + weight * c
    return SagForm(shape.kind, terms=tuple((i, j, c) for (i, j), c in sorted(merged.items())), scale=scale)


def _powers(value: Any, top: int) -> list[Any]:
    out = [value * 0 + 1]
    for _ in range(top):
        out.append(out[-1] * value)
    return out


def _polynomial(form: SagForm, x: Any, y: Any, slopes: bool) -> tuple[Any, Any, Any]:
    """``(z, dz/dx, dz/dy)`` of a polynomial form (the slopes ``None`` unless asked for)."""

    big_x, big_y = x * form.scale, y * form.scale
    px = _powers(big_x, max(i for i, _, _ in form.terms))
    py = _powers(big_y, max(j for _, j, _ in form.terms))
    z = big_x * 0
    if not slopes:
        for i, j, c in form.terms:
            z = z + c * (px[i] * py[j])
        return z, None, None
    zx, zy = big_x * 0, big_x * 0
    for i, j, c in form.terms:
        z = z + c * (px[i] * py[j])
        if i:
            zx = zx + (c * i) * (px[i - 1] * py[j])
        if j:
            zy = zy + (c * j) * (px[i] * py[j - 1])
    return z, zx * form.scale, zy * form.scale


def _toroid_curve(form: SagForm, y: Any, ops: SagOps) -> tuple[Any, Any, Any]:
    """``(z_y, T, T dz_y/dy)`` of a toroid's YZ curve; the domain mask is ``1 - (1 + k) c^2 y^2 >= 0``."""

    y2 = y * y
    arg = 1 - form.conic_y * y2
    root = ops.root(arg)
    z = form.curvature * y2 / (1 + root)
    power = y2
    derivative = y * 0
    previous = y2 * 0 + 1
    for i, alpha in enumerate(form.y_terms, start=1):
        z = z + alpha * power
        derivative = derivative + (2 * i * alpha) * (previous * y)
        previous = power
        power = power * y2
    return z, root, form.curvature * y + root * derivative


def sag_value(form: SagForm, x: Any, y: Any, ops: SagOps = FLOAT_OPS) -> tuple[Any, Any]:
    """``(z, ok)`` at ``(x, y)``."""

    if form.additive:
        z, _, _ = _polynomial(form, x, y, False)
        return z, x == x
    if form.kind == BICONIC:
        arg = 1 - form.conic_x * (x * x) - form.conic_y * (y * y)
        return (form.curvature_x * (x * x) + form.curvature_y * (y * y)) / (1 + ops.root(arg)), arg >= 0
    z_y, _, _ = _toroid_curve(form, y, ops)
    arg_y = 1 - form.conic_y * (y * y)
    if form.rotation is None:
        return z_y, arg_y >= 0
    distance = form.rotation - z_y
    p = _safe_inverse(distance)
    arg = 1 - p * p * (x * x)
    return z_y + p * (x * x) / (1 + ops.root(arg)), (arg_y >= 0) & (arg >= 0) & (distance * form.rotation > 0)


def sag_slopes(form: SagForm, x: Any, y: Any, ops: SagOps = FLOAT_OPS) -> tuple[Any, Any, Any, Any, Any]:
    """``(z, h_x, h_y, w, ok)`` at ``(x, y)`` (module docstring)."""

    if form.additive:
        z, zx, zy = _polynomial(form, x, y, True)
        return z, zx, zy, x * 0 + 1, x == x
    if form.kind == BICONIC:
        x2, y2 = x * x, y * y
        arg = 1 - form.conic_x * x2 - form.conic_y * y2
        root = ops.root(arg)
        u = 1 / (1 + root)
        q = form.curvature_x * x2 + form.curvature_y * y2
        return (q * u, (2 * form.curvature_x * x * root + q * form.conic_x * x * u) * u,
                (2 * form.curvature_y * y * root + q * form.conic_y * y * u) * u, root, arg >= 0)
    z_y, root_y, h_y = _toroid_curve(form, y, ops)
    arg_y = 1 - form.conic_y * (y * y)
    if form.rotation is None:
        return z_y, x * 0, h_y, root_y, arg_y >= 0
    distance = form.rotation - z_y
    p = _safe_inverse(distance)
    arg = 1 - p * p * (x * x)
    root_x = ops.root(arg)
    return (z_y + p * (x * x) / (1 + root_x), p * x * root_y, h_y, root_y * root_x,
            (arg_y >= 0) & (arg >= 0) & (distance * form.rotation > 0))


def _safe_inverse(value: Any) -> Any:
    """``1 / value``, 0 where ``value`` is 0 (outside the domain): a scalar or an array."""

    if hasattr(value, "shape"):
        nonzero = value != 0
        return nonzero * (1 / (value + (~nonzero) * 1.0))
    return 1 / value if value != 0 else value * 0


# ------------------------------------------------------------------------------------------------ sizes and words


def polynomial_bound(terms: Iterable[tuple[int, int, Any]], reach_scaled: Any) -> Any:
    """A bound of ``|sum c X^i Y^j|`` over ``X^2 + Y^2 <= reach_scaled^2``: ``sum |c| reach_scaled^(i + j)``."""

    total = reach_scaled * 0
    for i, j, c in terms:
        power = reach_scaled * 0 + 1
        for _ in range(i + j):
            power = power * reach_scaled
        total = total + abs(c) * power
    return total


def sag_shape_text(document: dict[str, Any]) -> str:
    """One line of a shape's JSON form, for a person reading a surface's facts (values as published, in mm)."""

    def stated(label: str, key: str, unit: str = "") -> list[str]:
        return [] if document.get(key) is None else [f"{label} {document[key]}{unit}"]

    kind = document["kind"]
    if kind == ZERNIKE:
        terms = ", ".join(f"Z{t['index']} {t['value_mm']}" for t in document["terms"])
        return f"Zernike {document['indexing']} {document['normalization']}, R {document['normalization_radius_mm']} mm: {terms} mm"
    if kind == XY_POLYNOMIAL:
        terms = ", ".join(f"x{t['x_power']}y{t['y_power']} {t['value_mm']}" for t in document["terms"])
        return f"XY polynomial, R {document['normalization_radius_mm']} mm: {terms} mm"
    if kind == BICONIC:
        parts = [*stated("Rx", "radius_x_mm", " mm"), *stated("Ry", "radius_y_mm", " mm"), *stated("kx", "conic_x"), *stated("ky", "conic_y")]
        return "Biconic: " + ", ".join(parts)
    parts = [*stated("Ry", "radius_y_mm", " mm"), *stated("k", "conic"), *stated("R rotation", "radius_of_rotation_mm", " mm")]
    if document.get("y_terms"):
        parts.append("y terms " + ", ".join(str(a) for a in document["y_terms"]))
    return "Toroid: " + ", ".join(parts)


__all__ = [
    "ADDITIVE_KINDS",
    "BICONIC",
    "DECIMAL_OPS",
    "FLOAT_OPS",
    "FRINGE",
    "FRINGE_ORDERS",
    "MAX_FRINGE_INDEX",
    "MAX_NOLL_INDEX",
    "MAX_RADIAL_ORDER",
    "MAX_SAG_TERMS",
    "MAX_TOROID_Y_TERMS",
    "NOLL",
    "NORMALIZATION_OF_INDEXING",
    "PEAK_UNIT",
    "RMS_ORTHONORMAL",
    "SAG_APERTURE_BEYOND_NORMALIZATION",
    "SAG_COMBINATION_NOT_SUPPORTED",
    "SAG_NORMALIZATION_MISMATCH",
    "SAG_OUTSIDE_DOMAIN",
    "SAG_TERM_NOT_SUPPORTED",
    "SagForm",
    "SagOps",
    "TOROIDAL",
    "XY_POLYNOMIAL",
    "ZERNIKE",
    "fringe_order",
    "noll_order",
    "polynomial_bound",
    "sag_form",
    "sag_shape_text",
    "sag_slopes",
    "sag_value",
    "xy_terms_problem",
    "zernike_monomials",
    "zernike_norm_squared",
    "zernike_order",
    "zernike_terms_problem",
]
