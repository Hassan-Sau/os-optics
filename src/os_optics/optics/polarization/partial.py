"""Partially polarized light as the incoherent sum of two orthogonal fully polarized components.

A source line of power ``L`` and degree of polarization ``P`` whose polarized part has the unit Jones vector ``J`` has the
coherency (polarization) matrix

    C = L [P J J^H + (1 - P) / 2 I] = L [(1 + P) / 2 J J^H + (1 - P) / 2 J_perp J_perp^H],
    J_perp = (-conj(J_y), conj(J_x))

(the eigen-decomposition of ``C``: ``J_perp`` is orthogonal to ``J``, the opposite point of the Poincare sphere). The Engine
models every element by a Jones matrix ``M`` (no element depolarizes light), so behind a sequence of elements the coherency
is ``M C M^H = L [(1 + P) / 2 (M J)(M J)^H + (1 - P) / 2 (M J_perp)(M J_perp)^H]``: the two components travel as two fully
polarized beams of powers ``(1 + P) / 2 L`` and ``(1 - P) / 2 L`` along the same routes, and wherever they arrive their
powers and Stokes parameters add. The decomposition is exact for every non-depolarizing system; an element that
depolarizes (a Mueller matrix that is not a Jones matrix) is outside the model.

Their amplitudes are uncorrelated, so no interference term exists between them, along one route or two: each component
gets a coherence group of its own (``component_group``: the second one's is the source's group with ``GROUP_MARK``),
so every coherent sum of the Engine (two-beam pairs, plane superpositions, irradiance maps, surface exposures) adds their
powers, and two-beam pairs are formed within one component only. Within a component the beams of the line are mutually
coherent as for a fully polarized source (its declared coherence length holds).

Unpolarized light (``P = 0``) is the sum of equal powers in ``J`` and ``J_perp`` for any unit ``J``; the Project's
``UNPOLARIZED`` kind takes ``J`` along the source's local +X. A component's share is exact binary arithmetic:
``w0 = (1 + P) / 2`` and ``w1 = 1 - w0`` (exact, ``w0`` in ``[1/2, 1]``), so ``w0 + w1 = 1``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ...scene.model import SourceG

#: What the coherence group of a line's second polarization component appends to its source's group (the mark its path ids
#: carry too: ``optics/rays/path_ids.py`` ``COMPONENT_MARK``).
GROUP_MARK = ":p1"


def partially_polarized(degree_of_polarization: float | None) -> float | None:
    """The degree of polarization of partially polarized light (``< 1``), or None for fully polarized light (None or 1)."""
    if degree_of_polarization is None or degree_of_polarization >= 1.0:
        return None
    return float(degree_of_polarization)


def orthogonal_jones(jones: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """``(-conj(J_y), conj(J_x))``: the unit Jones vector orthogonal to ``jones`` (``J_perp^H J = 0``), no negative zeros."""
    jx, jy = complex(jones[0]), complex(jones[1])
    return np.array([complex(-jy.real + 0.0, jy.imag + 0.0), complex(jx.real + 0.0, -jx.imag + 0.0)], dtype=np.complex128)


def component_shares(degree_of_polarization: float) -> tuple[float, float]:
    """The shares of a line's power of its two components, ``((1 + P) / 2, 1 - (1 + P) / 2)``: they sum to 1 exactly."""
    w0 = (1.0 + degree_of_polarization) / 2.0
    return w0, 1.0 - w0


def component_group(coherence_group: str, polarization_component: int) -> str:
    """The coherence group of a component's beams: the source's group for component 0, a group of its own for component 1."""
    return coherence_group if polarization_component == 0 else coherence_group + GROUP_MARK


def source_components(src: SourceG) -> tuple[tuple[SourceG, float], ...]:
    """What a source launches for each of its lines: ``(source, share of the line's power)`` -- the source itself with share
    1.0 when it is fully polarized; its two polarization components (each a ``SourceG`` with its own Jones vector, coherence
    group and ``polarization_component``) when it is partially polarized."""
    if src.degree_of_polarization is None:
        return ((src, 1.0),)
    w0, w1 = component_shares(src.degree_of_polarization)
    first = replace(src, polarization_component=0)
    second = replace(src, polarization_component=1, jones=orthogonal_jones(src.jones),
                     coherence_group=component_group(src.coherence_group, 1))
    return ((first, w0), (second, w1))


def component_pair(a: object, b: object) -> bool:
    """Whether two arrivals (records with ``source_id``, ``spectral_index`` and ``polarization_component``) are the two
    mutually incoherent components of one partially polarized line: they form no two-beam pair."""
    ca, cb = getattr(a, "polarization_component", None), getattr(b, "polarization_component", None)
    return (ca is not None and cb is not None and ca != cb and getattr(a, "source_id") == getattr(b, "source_id")
            and getattr(a, "spectral_index") == getattr(b, "spectral_index"))


__all__ = [
    "GROUP_MARK",
    "component_group",
    "component_pair",
    "component_shares",
    "orthogonal_jones",
    "partially_polarized",
    "source_components",
]
