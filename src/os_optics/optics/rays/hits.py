"""``SurfaceHit``: the beam state where a traced packet meets a physical surface (the per-surface exposure input).

The tracer records one ``SurfaceHit`` each time a packet's chief ray reaches a physical (non-observation) surface, before
the surface's clear aperture or interaction acts: the incident power, the Gaussian state there (both transverse radii,
not only the larger one a path segment keeps), the chief ray's angle of incidence and the local surface normal, and the
live pupil rays (for the same aberration check an observation applies), the packet's lineage up to the surface, and its
field and coherence (Jones vector, chief-ray optical path, coherence group and length: mutually coherent beams are
combined by their amplitudes, a rating's polarization is checked against the Jones state). Probe
traces of untraced light record theirs too (``ProbeOutcome.surface_hits``: the untraced light that reaches a surface).
``validation/exposures.py`` turns them into ``ValidationResult.exposures``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..gaussian.state import GState
from .envelope import NO_SPREAD, Spread

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


@dataclass(frozen=True)
class HeldData:
    """Data of a surface the trace read outside the range it is stated for, held at its nearest stated point
    (``Tracer._live_coating``): the surface, the code of the issue that states it, and per parameter outside its range
    ``(parameter, unit, value asked for, value the data was read at, stated minimum, stated maximum)``."""

    surface_id: str
    code: str
    parameters: tuple[tuple[str, str, float, float, float, float], ...]


@dataclass(frozen=True)
class SurfaceHit:
    surface_id: str
    owner_id: str
    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    power_W: float
    gstate: GState
    position: Array
    direction: Array
    normal: Array  # unit surface normal at the chief-ray hit (any orientation)
    aoi_deg: float
    cos_aoi: float = 1.0  # |d . n| of the chief ray (the cosine of ``aoi_deg``, without libm)
    chief_obstructed: bool = False
    # live pupil rays at the hit (ArrivalRecord's names: the aberration check reads them alike)
    pupil_points: Array = field(default_factory=lambda: np.zeros((0, 3)))
    pupil_directions: Array = field(default_factory=lambda: np.zeros((0, 3)))
    pupil_launch: Array = field(default_factory=lambda: np.zeros((0, 2)))
    lineage: tuple[str, ...] = ()  # '<surface_id>:<event>' interactions from the source up to this surface
    power_scope: tuple[str, ...] = ()  # surfaces behind which the beam's power is not the light's (``_Packet.scope_behind``)
    # of them, where every one took a bounded share of the beam (``_Packet.taken_path_at``): those surfaces and the most
    # power they took together (W): ``power_W`` is then an upper value of the light's, at most that much above it
    power_taken_at: tuple[str, ...] = ()
    power_taken_upper_W: float | None = None
    # A beam whose chief ray misses the surface while its footprint reaches the support (``Tracer`` ``watch``): a lower bound
    # of the transverse distance from the chief ray to the support's nearest point (mm); ``position`` / ``gstate`` are the
    # chief ray's crossing of the surface's reference plane. ``None``: an ordinary hit. ``inf`` with an invalid ``gstate``:
    # live pupil rays meet the surface, how much of the beam is not known.
    miss_distance_mm: float | None = None
    # the beam's field and coherence there (the exposure combines mutually coherent beams and checks a rating's polarization):
    # Jones vector in the transverse basis (e1, e2), the chief ray's optical path from its source (mm; ``opl_defined`` False
    # behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical path), its source's coherence group and the
    # line's declared coherence length (None: not declared)
    jones: CArray = field(default_factory=lambda: np.zeros(2, dtype=np.complex128))
    e1: Array = field(default_factory=lambda: np.zeros(3))
    e2: Array = field(default_factory=lambda: np.zeros(3))
    opl_mm: float = 0.0
    opl_defined: bool = True
    coherence_group: str = ""
    coherence_length_mm: float | None = None
    # the surface is a plane (every point of it near the hit lies in the plane of ``normal`` through ``position``: the
    # exposure bounds two beams' relative delay across it by the tangential part of their directions)
    surface_flat: bool = False
    # the chief ray's group optical path from its source (sum of n_g x length, mm; None: a medium's group index is not
    # available), its group-delay dispersion (sum of k2 x length, mm^2) and whether it crossed a dispersive medium (n_g != n):
    # two beams' degree of coherence follows their relative group delay, not their phase delay
    group_opl_mm: float | None = None
    gdd_mm2: float = 0.0
    dispersive: bool = False
    # how much the envelope delay can vary over the beam's ray families of each size (``optics/rays/envelope.py``: per
    # ``COHERENCE_K``, the group path's and the group-delay dispersion's spread; ``None``: not bounded)
    envelope_spread: Spread | None = NO_SPREAD
    # the passes on its way whose group delay the model does not state (``optics/interference/coherence.py``: two beams'
    # relative delay is known only when these agree)
    delays: tuple[str, ...] = ()
