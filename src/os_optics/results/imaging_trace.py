"""An analysis-only trace of a dense, regular pupil sampling of a run's sources (``results.imaging``).

The run's own Engine request (``engine_request.json``) is lowered again and the chosen sources are traced with a square grid
of launch points over a disc of ``radius`` beam radii in place of their equal-power rings (``SourceG.pupil_layout``; the
tracer's one entry point for it is the launch of a source's pupil rays). Nothing else of the request changes, the untraced-light
probes and interference are not run, and the run's files, result and hashes are untouched. The rays overfill the system's
aperture stop on purpose: the apertures stop the rays outside it, which leaves the vignetted pupil.

The grid is regular, so each ray keeps its grid index for the solid angle its tube fills in image space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.request import ValidationRequest
from ..math.vectors import dot
from ..optics.materials.library import MaterialLibrary
from ..optics.polarization.partial import source_components
from ..optics.rays.path_ids import root_path_id
from ..optics.rays.tracer import Tracer
from ..scene.lower import lower_scene
from ..validation.ray_paths import _RayPathRecorder
from .rays import FORWARD_COS_MIN, OBSERVATION_PREFIX

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LaunchGrid:
    """A square grid of launch points within a disc: ``offsets`` (N, 2) in beam radii, ``index`` (N, 2) the grid columns and
    rows, ``step`` the pitch in beam radii."""

    offsets: Array
    index: NDArray[np.intp]
    step: float
    count: int
    radius: float


def launch_grid(count: int, radius: float) -> LaunchGrid:
    """``count`` points across the disc's diameter (cell centres), those inside the disc of ``radius``."""

    step = 2.0 * radius / count
    axis = (np.arange(count) - 0.5 * (count - 1)) * step
    xx, yy = np.meshgrid(axis, axis)
    inside = xx * xx + yy * yy <= radius * radius
    columns, rows = np.meshgrid(np.arange(count), np.arange(count))
    return LaunchGrid(np.stack((xx[inside], yy[inside]), axis=1), np.stack((columns[inside], rows[inside]), axis=1).astype(np.intp),
                      step, count, radius)


@dataclass(frozen=True)
class FieldRays:
    """One field (a source's spectral line) at one crossing of the image plane, from the dense trace.

    ``arrived`` marks the launched rays that reach the plane; ``points`` / ``directions`` are where they cross it and their
    directions after it; ``opl`` their optical path from the source's phase front to the plane (``launch`` term included);
    ``None`` arrays and ``opl_defined`` false when the path's optical path is not defined (an ideal or tabulated thin lens, a
    Faraday rotator, a grating). ``chief_*`` is the source axis's ray (behind a clear aperture whose rim it fell on, the
    passing pupil ray promoted to carry the beam's reference, ``Tracer._promote_reference``); ``previous_point`` where it met
    the last surface before the plane."""

    source_id: str
    wavelength_nm: float
    path_id: str
    crossing_index: int
    crossings: int
    grid: LaunchGrid
    arrived: NDArray[np.bool_]
    points: Array
    directions: Array
    opl: Array | None
    opl_defined: bool
    chief_point: Array
    chief_direction: Array
    previous_point: Array | None
    medium_index: float
    source_jones: NDArray[np.complex128]
    path_power_W: float | None
    # each launched ray's irradiance where it crosses the plane relative to its launch (|J|^2 of the field the trace carries
    # over that of the source's Jones vector: a sample's amplitude, the interfaces' and coatings' s and p transmissions); for
    # a partially polarized or unpolarized source the sum over its two incoherent components, each by its share of the line
    irradiance: NDArray[np.float64] | None = None
    # the same per component that reaches the plane (each times its share of the line): mutually incoherent, each
    # component's field propagates to the image apart and their point spread functions add in irradiance
    component_irradiance: tuple[NDArray[np.float64], ...] = ()
    # why the polarization components arriving on this route cannot be combined (``trace_fields``); None when they can
    problem: str | None = None


def _launch_terms(points: Array, directions: Array, origin: Array, axis: Array, n_source: float) -> Array:
    """Optical path from the source's phase front to each ray's start point (``results.rays``' launch term)."""

    rho = points - origin
    rho = rho - dot(rho, axis)[:, None] * axis[None, :]
    cos = dot(directions, axis)
    cos = np.where(cos > FORWARD_COS_MIN, cos, 1.0)
    sigma = (directions - cos[:, None] * axis[None, :]) / cos[:, None]
    return n_source * dot(sigma, rho) / (np.sqrt(1.0 + dot(sigma, sigma)) + 1.0)


#: Two polarization components share a route when their rays cross the plane at the same points within this distance.
ROUTE_TOLERANCE_MM = 1e-9


def trace_fields(request: ValidationRequest | dict, observation_id: str, wanted: set[str], preferred: dict[tuple[str, float], str],
                 grid: LaunchGrid, crossing_index: int | None, arriving: set[str] | None = None) -> dict[tuple[str, float], FieldRays]:
    """Trace the sources ``wanted`` with ``grid`` and read their rays at ``observation_id``: ``{(source id, wavelength):
    FieldRays}``, of the route ``preferred`` names for a line (the run's strongest path at the plane), else of the strongest route
    that reaches the plane. ``crossing_index`` ``None``: the last crossing.

    A route is a line's path less the root of its polarization component: the two components of a partially polarized or
    unpolarized source travel it as ``<source>#<i><branch>`` and ``<source>#<i>:p1<branch>``. Either may reach the plane alone (a
    polarizer passes one). ``arriving``: the path ids of the run's arrivals at the plane, which say which components reach it; a
    component the run has no arrival of there adds nothing, one it has but this trace does not reach is not retraced
    (``POLARIZATION_COMPONENT_NOT_RETRACED``), and two components whose rays do not cross the plane together are not added
    (``POLARIZATION_COMPONENT_ROUTES_DIFFER``)."""

    if not isinstance(request, ValidationRequest):
        request = ValidationRequest.model_validate(request)
    cfg: NumericalConfig = request.configuration.numerical
    project = request.project
    library = MaterialLibrary(project.materials, cfg)
    scene, _poses = lower_scene(project, cfg, library)
    sources = [replace(src, pupil_layout=grid.offsets, ring_count=1, points_per_ring=int(grid.offsets.shape[0]))
               for src in scene.sources if src.source_id in wanted]
    scene = replace(scene, sources=sources, emitters=[])
    recorder = _RayPathRecorder(keep_fields=True)
    Tracer(scene, library, cfg, include_pupil=True, recorder=recorder).run()
    by_path = {path.path_id: (path, rec) for path, rec in recorder.finished}
    by_source = {src.source_id: src for src in sources}
    surface = OBSERVATION_PREFIX + observation_id
    count = grid.offsets.shape[0]
    routes: dict[tuple[str, float, int, str], dict[int, tuple]] = {}
    for path, rec in recorder.finished:
        steps = [k for k, s in enumerate(rec.surfaces) if s == surface]
        if not steps or path.source_id not in wanted:
            continue
        root1 = root_path_id(path.source_id, path.spectral_index, 1)
        component = 1 if path.path_id.startswith(root1) else 0
        root = root1 if component else root_path_id(path.source_id, path.spectral_index)
        routes.setdefault((path.source_id, path.wavelength_nm, path.spectral_index, path.path_id[len(root):]), {})[component] = (path, rec, steps)
    out: dict[tuple[str, float], FieldRays] = {}
    for (source_id, wavelength, spectral_index, branch), members in sorted(routes.items(), key=lambda item: item[0]):
        key = (source_id, wavelength)
        ids = {c: root_path_id(source_id, spectral_index, c) + branch for c in (0, 1)}
        if key in preferred and preferred[key] not in ids.values():
            continue
        power = math.fsum(m[0].final_power_W or 0.0 for m in members.values())
        if key in out and (out[key].path_power_W or 0.0) >= power:
            continue
        src = by_source[source_id]
        components = source_components(src)  # ((the source, 1.0),) or its two polarization components with their shares
        reaching = [c for c in range(len(components)) if arriving is None or ids[c] in arriving] or sorted(members)
        problem = None
        if any(c not in members for c in reaching):
            problem = "POLARIZATION_COMPONENT_NOT_RETRACED"
        present = [c for c in reaching if c in members]
        reference = present[0] if present else min(members)
        path, rec, steps = members[reference]
        index = len(steps) - 1 if crossing_index is None else crossing_index
        if index < 0 or index >= len(steps):
            continue
        step = steps[index]
        arrived = rec.reached[step][1:1 + count].copy()
        points = rec.points[step][1:1 + count].copy()
        parts: list[NDArray[np.float64]] = []
        for c in present or [reference]:
            other_path, other, other_steps = members[c]
            if c != reference:
                other_step = other_steps[index] if len(other_steps) == len(steps) else None
                same = (other_step is not None and np.array_equal(other.reached[other_step][1:1 + count], arrived)
                        and bool(np.all(np.abs(other.points[other_step][1:1 + count][arrived] - points[arrived]) <= ROUTE_TOLERANCE_MM)))
                if not same:
                    problem = problem or "POLARIZATION_COMPONENT_ROUTES_DIFFER"
                    continue
            else:
                other_step = step
            component_src, share = components[c] if c < len(components) else components[0]
            parts.append(share * _relative_irradiance(other.fields[other_step][1:1 + count], component_src.jones))
        irradiance = np.sum(parts, axis=0) if parts else np.zeros(count)
        root_path, root = path, rec
        while root_path.parent_path_id is not None and root_path.parent_path_id in by_path:
            root_path, root = by_path[root_path.parent_path_id]
        n_source = library.evaluator(src.medium).index(path.wavelength_nm)[0]
        directions = rec.directions[step][1:1 + count].copy()
        opl = None
        if rec.opl_ok[step] and n_source is not None:
            launch = _launch_terms(root.points[0][1:1 + count], root.directions[0][1:1 + count], root.points[0][0], root.directions[0][0],
                                   float(n_source))
            opl = rec.opl[step][1:1 + count] + launch
        ends = [s for s in path.segments if s.end_surface_id == surface]
        medium = ends[min(index, len(ends) - 1)].refractive_index if ends else 1.0
        out[key] = FieldRays(
            source_id=path.source_id, wavelength_nm=path.wavelength_nm, path_id=path.path_id, crossing_index=index, crossings=len(steps),
            grid=grid, arrived=arrived, points=points, directions=directions, opl=opl, opl_defined=opl is not None,
            chief_point=rec.points[step][0].copy(), chief_direction=rec.directions[step][0].copy(),
            previous_point=rec.points[step - 1][0].copy() if step > 0 else None, medium_index=float(medium),
            source_jones=np.array(src.jones, dtype=np.complex128), path_power_W=power, irradiance=irradiance,
            component_irradiance=tuple(parts), problem=problem)
    return out


def _relative_irradiance(fields: NDArray[np.complex128], launched: tuple[complex, complex]) -> NDArray[np.float64]:
    """|J|^2 of each ray's field over that of the Jones vector it was launched with."""

    jones = np.array(launched, dtype=np.complex128)
    return np.real(np.sum(fields * np.conj(fields), axis=1)) / float(np.real(np.vdot(jones, jones)))


__all__ = ["FieldRays", "LaunchGrid", "launch_grid", "trace_fields"]
