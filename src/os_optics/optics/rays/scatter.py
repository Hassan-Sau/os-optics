"""Scattered light: the ledger kinds of diffuse surface scatter and bulk scattering, and where a surface scatters a beam.

Surface scatter (``contracts.optics.SurfaceScatter``): where a beam meets a surface that scatters, ``fraction`` of the power
that meets it leaves as incoherent, unpolarized light with the stated angular law about the surface normal on the stated side;
the beam goes on with the rest (its field scaled by ``sqrt(1 - fraction)``, its direction, Gaussian state and polarization
unchanged) and meets the surface's own interaction. The scattered power is booked ``SURFACE_SCATTER`` and the event recorded
as a ``ScatterOrigin``: the run traces it afterwards as a secondary extended emitter, a point at the beam's hit point
(``optics/rays/scatter_trace.py``), and the rays' fates replace that booking in the ledger. Scattered light the run does not
trace stays ``SURFACE_SCATTER``: light of unknown fate, which may reach any plane (light a sampled ray scatters again, light a
probe trace scatters, the origins beyond the run's ``max_scatter_origins``, a run that reads the beams only).

Bulk scattering (``scattering_coefficient_per_mm`` of a material): a traced beam's power decays as ``exp(-(alpha + mu_s) L)``
along a path ``L`` in the medium; the decay splits into absorption and scattering in the ratio ``alpha : mu_s`` (both remove
power at the same point of the path, so the split is exact), booked ``BULK_ABSORPTION`` and ``BULK_SCATTER``. The scattered
light is not followed: ``BULK_SCATTER`` has an unknown fate, so the power of every plane its source line lights is a lower
bound and a plane it does not light is not dark.

Model scope: geometric, incoherent, single scatter per traced beam; no diffraction, speckle or coherence of the scattered
light; the scatter leaves from the beam's chief-ray hit point (the footprint is not resolved: valid where the beam is small
against the distances to what the scattered light reaches, ``ScatterOrigin.beam_radius_mm`` states it).
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

#: Ledger kind of diffusely scattered surface light (the run's scatter rays replace it where it traces them).
SINK_SURFACE_SCATTER = "SURFACE_SCATTER"
#: Ledger kind of light a medium scatters out of a traced beam (not followed).
SINK_BULK_SCATTER = "BULK_SCATTER"

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ScatterOrigin:
    """A beam meeting a scattering surface: what the scattered light is and where it starts. ``axis``: the surface normal on the
    scatter side (unit, design frame); ``medium``: the medium the scattered light leaves into; ``power_W``: the scattered power
    (``fraction`` x ``incident_power_W``); ``beam_radius_mm``: the beam's largest Gaussian radius there (``None``: not known);
    ``power_scope``: the surfaces behind which the beam's power is out of model scope (``_Packet.power_scope``): the scattered power
    is then that of its chief ray, not of the light, and so is everything the scattered rays bring to a plane or a surface."""

    ordinal: int
    source_id: str
    source_owner_id: str
    spectral_index: int
    wavelength_nm: float
    path_id: str
    surface_id: str
    surface_owner_id: str
    point: Array
    axis: Array
    medium: str
    incident_power_W: float
    fraction: float
    power_W: float
    side: str
    law: str
    exponent: float
    sample_count: int
    seed: int
    beam_radius_mm: float | None
    power_scope: tuple[str, ...] = ()


def bulk_shares(absorption_per_mm: float, scattering_per_mm: float, lost_W: float) -> tuple[tuple[str, float], ...]:
    """The ledger entries of ``lost_W`` that a medium took from a beam over a path: all ``BULK_ABSORPTION`` where it does not
    scatter, else absorption and scattering in the ratio of their coefficients (the scattered share by difference, so the two
    add to ``lost_W``)."""

    if scattering_per_mm == 0.0:
        return (("BULK_ABSORPTION", lost_W),)
    scattered = lost_W * (scattering_per_mm / (absorption_per_mm + scattering_per_mm))
    absorbed = lost_W - scattered
    return tuple((kind, power) for kind, power in (("BULK_ABSORPTION", absorbed), (SINK_BULK_SCATTER, scattered)) if power > 0.0)


__all__ = ["SINK_BULK_SCATTER", "SINK_SURFACE_SCATTER", "ScatterOrigin", "bulk_shares"]
