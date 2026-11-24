"""Observer of a probe trace that keeps, for each interaction, the beam's 3D field just before and just after it.

It implements ``optics.rays.tracer.TraceObserver`` and is observation only: it copies the state the tracer produced and
never writes to a packet. Before / after are taken at the same point of the surface, so the pair holds the interaction's own
Jones action (including a Gaussian truncation by the surface's clear aperture) without any propagation phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Lineage events that are not optical interactions of an element.
_NOT_ELEMENT_EVENTS = frozenset({"observe", "absorbed"})


@dataclass(frozen=True)
class ElementEvent:
    path_id: str
    occurrence: int  # n-th interaction of this path at this surface (a double pass meets a surface twice)
    surface_id: str
    event: str
    d_in: NDArray[np.float64]
    d_out: NDArray[np.float64]
    e_in: NDArray[np.complex128]  # 3D field (flux-normalized) before the interaction
    e_out: NDArray[np.complex128]  # after it (on the output branch)
    aoi_deg: float

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.path_id, self.occurrence, self.surface_id, self.event)


def _field(pk) -> NDArray[np.complex128]:
    return (pk.jones[0, 0] * pk.e1[0] + pk.jones[0, 1] * pk.e2[0]).astype(np.complex128)


def _split(entry: str) -> tuple[str, str]:
    surface, _sep, event = entry.rpartition(":")
    return surface, event


class ElementRecorder:
    """Collects ``ElementEvent`` s of one trace (``events``, in trace order)."""

    def __init__(self) -> None:
        self.events: list[ElementEvent] = []
        self._before: dict[str, tuple[str, NDArray[np.float64], NDArray[np.complex128], int]] = {}
        self._visits: dict[tuple[str, str], int] = {}

    def _occurrence(self, path_id: str, surface_id: str) -> int:
        key = (path_id, surface_id)
        n = self._visits.get(key, 0)
        self._visits[key] = n + 1
        return n

    def packet_started(self, pk, surface_id: str | None) -> None:
        """Nothing to keep: a packet's first interaction is recorded from its ``packet_advanced``."""

    def pupil_rays_diverged(self, pk, rays, next_surface_ids, delivered) -> None:
        """Nothing to keep: the element matrices use the chief ray only."""

    def packet_advanced(self, pk, surface_id: str | None) -> None:
        if surface_id is not None:
            self._before[pk.path_id] = (surface_id, pk.d[0].copy(), _field(pk), len(pk.lineage))

    def packet_interacted(self, pk) -> None:
        before = self._before.pop(pk.path_id, None)
        if before is None:
            return
        surface_id, d_in, e_in, n_lineage = before
        if len(pk.lineage) <= n_lineage:
            return  # no interaction took place (a clipped beam)
        surface, event = _split(pk.lineage[-1])
        if surface != surface_id or event in _NOT_ELEMENT_EVENTS:
            return
        aoi = pk.segments[-1].angle_of_incidence_deg if pk.segments else None
        self.events.append(ElementEvent(pk.path_id, self._occurrence(pk.path_id, surface_id), surface_id, event, d_in, pk.d[0].copy(),
                                        e_in, _field(pk), float(aoi) if aoi is not None else 0.0))

    def branch_spawned(self, parent, child) -> None:
        before = self._before.get(parent.path_id)
        if before is None or not child.lineage:
            return
        surface_id, d_in, e_in, _n = before
        surface, event = _split(child.lineage[-1])
        if surface != surface_id:
            return
        d_out = child.d[0].copy()
        aoi = math.degrees(math.acos(min(1.0, max(-1.0, -float(d_in @ d_out))))) / 2.0  # reflection: 2 x AOI between the rays
        self.events.append(ElementEvent(child.path_id, self._occurrence(child.path_id, surface_id), surface_id, event, d_in, d_out, e_in, _field(child), aoi))

    def path_finished(self, pk, path) -> None:
        self._before.pop(pk.path_id, None)

    def path_dropped(self, pk) -> None:
        self._before.pop(pk.path_id, None)
