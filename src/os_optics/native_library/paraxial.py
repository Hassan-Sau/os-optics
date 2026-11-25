"""Paraxial first-order optics of coaxial surface sequences, in exact decimal arithmetic.

A surface ``k`` at vertex ``z_k`` with vertex curvature ``c_k`` (``1 / radius``; a conic or a cylinder has the same
paraxial power in its curved section) between indices ``n_k`` (before) and ``n'_k`` (after) refracts a paraxial ray
``(y, u)`` -- height and reduced angle ``n u`` -- by the matrix ``[[1, 0], [-(n'_k - n_k) c_k, 1]]``; a gap ``d`` in index
``n`` transfers it by ``[[1, d / n], [0, 1]]``. The product from the first to the last vertex, ``[[A, B], [C, D]]`` (air
on both sides), gives

* effective focal length ``EFL = -1 / C``;
* back focal length (last vertex to rear focal point) ``BFL = -A / C``;
* front focal length (front focal point to first vertex) ``FFL = -D / C``;
* rear principal plane ``z_last + BFL - EFL``, front principal plane ``z_first - FFL + EFL``.

These are the textbook Gaussian-optics relations (e.g. Hecht, *Optics*, 5th ed., sec. 6.2; W. J. Smith, *Modern Optical
Engineering*, 4th ed., ch. 2-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

from .numbers import ONE, WORK_CONTEXT, ZERO

Matrix = tuple[Decimal, Decimal, Decimal, Decimal]
IDENTITY: Matrix = (ONE, ZERO, ZERO, ONE)


@dataclass(frozen=True)
class ParaxialSurface:
    z_mm: Decimal
    curvature: Decimal
    n_before: Decimal
    n_after: Decimal


@dataclass(frozen=True)
class FirstOrder:
    """First-order properties of a surface sequence (lengths in mm, element frame)."""

    efl: Decimal
    bfl: Decimal
    ffl: Decimal
    front_principal_z: Decimal
    rear_principal_z: Decimal
    first_z: Decimal
    last_z: Decimal

    @property
    def rear_focus_z(self) -> Decimal:
        return self.last_z + self.bfl

    @property
    def front_focus_z(self) -> Decimal:
        return self.first_z - self.ffl


def multiply(left: Matrix, right: Matrix) -> Matrix:
    a, b, c, d = left
    e, f, g, h = right
    with localcontext(WORK_CONTEXT):
        return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def transfer(distance: Decimal, index: Decimal = ONE) -> Matrix:
    with localcontext(WORK_CONTEXT):
        return ONE, distance / index, ZERO, ONE


def refraction(power: Decimal) -> Matrix:
    return ONE, ZERO, -power, ONE


def thin_lens(focal_length: Decimal) -> Matrix:
    with localcontext(WORK_CONTEXT):
        return refraction(ONE / focal_length)


def sequence_matrix(surfaces: list[ParaxialSurface] | tuple[ParaxialSurface, ...]) -> Matrix:
    """The reduced ABCD matrix from the first to the last vertex."""

    matrix = IDENTITY
    with localcontext(WORK_CONTEXT):
        for k, surface in enumerate(surfaces):
            if k:
                previous = surfaces[k - 1]
                matrix = multiply(transfer(surface.z_mm - previous.z_mm, previous.n_after), matrix)
            matrix = multiply(refraction((surface.n_after - surface.n_before) * surface.curvature), matrix)
    return matrix


def first_order(matrix: Matrix, first_z: Decimal, last_z: Decimal) -> FirstOrder:
    """First-order properties of a system matrix (air on both sides); ``ValueError`` for an afocal system."""

    a, _b, c, d = matrix
    if c == 0:
        raise ValueError("the system has no power (afocal)")
    with localcontext(WORK_CONTEXT):
        efl, bfl, ffl = -ONE / c, -a / c, -d / c
        return FirstOrder(efl=efl, bfl=bfl, ffl=ffl, front_principal_z=first_z - ffl + efl, rear_principal_z=last_z + bfl - efl,
                          first_z=first_z, last_z=last_z)


def sequence_first_order(surfaces: list[ParaxialSurface] | tuple[ParaxialSurface, ...]) -> FirstOrder:
    return first_order(sequence_matrix(surfaces), surfaces[0].z_mm, surfaces[-1].z_mm)


__all__ = ["FirstOrder", "IDENTITY", "Matrix", "ParaxialSurface", "first_order", "multiply", "refraction", "sequence_first_order",
           "sequence_matrix", "thin_lens", "transfer"]
