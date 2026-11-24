"""Pupil-ray samples at observation planes and the geometric wavefront (optical path difference) they define.

Both read the run's ray data (``ray_data.json``: ``validation.ray_paths.ray_paths_document``): for every beam path the
chief and pupil rays at each step of the trace (source or branch point, then every surface the chief ray reached). A
crossing of observation plane ``ID`` is the step whose surface id is ``observation/ID``; the path's ``c``-th such step is
crossing ``c`` (``crossing_index``). Pupil rays only ever leave a beam, so a pupil ray present at a step was present at
every earlier step of its path.

Wavefront model (geometric optics): every pupil ray starts on the source plane with optical path length 0, travelling
along the normal of the source's Gaussian phase front (curvature 1/R of the beam at the source plane; the rays of a round
beam meet in the centre of curvature). Its optical path from that phase front to the observation plane is
``W0 + OPL``, where ``W0`` is the optical path from the phase front to its start point on the source plane
(``n_source * (sigma . rho) / (sqrt(1 + |sigma|^2) + 1)`` for transverse launch offset ``rho`` and launch slope ``sigma``;
exact for round beams). Each ray is then continued along its direction, in the medium at the plane, to the reference
surface through the chief ray's hit point: the plane perpendicular to the chief ray, or a sphere about a centre. The
optical path difference of a ray is its optical path to the reference surface minus the chief ray's, in vacuum
wavelengths; the mean over the pupil rays (piston, equal power weights) is removed. Diffraction is not included: this
is the aberration of the traced geometric ray bundle, not a diffracted field.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..contracts.results import ArrivalResult, ObservationResult, ValidationResult
from ..units import NM_TO_MM
from .tables import _row
from ..math.vectors import dot
from ..validation.ray_paths import RAY_DATA_FULL, ray_data_recording

OBSERVATION_PREFIX = "observation/"
# Wavefront requests need at least this many pupil rays on the plane (one ring of the default sampling).
MIN_WAVEFRONT_RAYS = 8
# A reference sphere needs a radius of at least this (mm); a smaller one is a centre on the chief ray's hit point.
MIN_REFERENCE_RADIUS_MM = 1e-9
# The two Gaussian waists count as one point when their distances to the plane agree within this fraction of the
# Rayleigh range (otherwise the beam is astigmatic and the Gaussian waist is not one centre).
WAIST_MATCH_FRACTION_OF_RAYLEIGH = 0.01
# ray_focus: the rays' least-squares convergence point is undefined when the normal matrix's smallest eigenvalue is below
# this fraction of its largest (parallel rays: a collimated bundle has no focus).
RAY_FOCUS_MIN_EIGEN_FRACTION = 1e-12
# A ray must reach the reference plane travelling forward along the chief ray (cosine above this).
FORWARD_COS_MIN = 1e-9

WAVEFRONT_MODEL = (
    "Geometric optics: optical path differences of the traced pupil rays from the source's Gaussian phase front to a "
    "reference surface through the chief ray's hit point (plane perpendicular to the chief ray, or a sphere about a "
    "centre), relative to the chief ray, in vacuum wavelengths, piston (equal-weight mean) removed. Diffraction is not "
    "included."
)

RAY_DATA_NOT_AVAILABLE = "RAY_DATA_NOT_AVAILABLE"
CROSSING_NOT_FOUND = "CROSSING_NOT_FOUND"
RAY_OPL_UNDEFINED = "RAY_OPL_UNDEFINED"
TOO_FEW_RAYS = "TOO_FEW_RAYS"
REFERENCE_UNDEFINED = "REFERENCE_UNDEFINED"
RAY_MISSES_REFERENCE = "RAY_MISSES_REFERENCE"
RAY_DATA_INCONSISTENT = "RAY_DATA_INCONSISTENT"


def crossing_steps(path: dict[str, Any], observation_id: str) -> list[int]:
    """The steps of a ray-data path at which its chief ray crosses ``observation_id`` (index = crossing index)."""

    surface = OBSERVATION_PREFIX + observation_id
    return [k for k, s in enumerate(path["chief"]["surface_ids"]) if s == surface]


def observation_crossings(ray_data: dict[str, Any]) -> dict[str, list[str]]:
    """Path id -> the observation ids its chief ray crosses, in order."""

    return {p["path_id"]: [s[len(OBSERVATION_PREFIX):] for s in p["chief"]["surface_ids"] if s and s.startswith(OBSERVATION_PREFIX)]
            for p in ray_data.get("paths", [])}


def _counted(observation: ObservationResult) -> dict[tuple[str, int], bool]:
    counted = {(a.beam.path_id, a.crossing_index): True for a in observation.arrivals}
    counted.update({(c.path_id, c.crossing_index): False for c in observation.outside_active_area})
    return counted


def _plane_axes(observation: ObservationResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    plane = observation.plane
    if plane is None:
        return None
    return np.array(plane.origin_mm), np.array(plane.u_axis), np.array(plane.v_axis), np.array(plane.normal)


def _ray_row(observation: ObservationResult, path: dict[str, Any], crossing: int, counted: bool | None, axes: Any,
             point: list[float], direction: list[float], opl: float, *, ray: dict[str, Any] | None, weight: float | None,
             opl_defined: bool) -> dict[str, Any]:
    p, d = np.array(point), np.array(direction)
    if axes is None:
        u = v = du = dv = dn = None
    else:
        origin, ua, va, na = axes
        u, v = float(dot((p - origin), ua)), float(dot((p - origin), va))
        du, dv, dn = float(dot(d, ua)), float(dot(d, va)), float(dot(d, na))
    return _row("rays", {
        "observation_id": observation.observation_id, "path_id": path["path_id"], "source_id": path["source_id"],
        "wavelength_nm": path["wavelength_nm"], "crossing_index": crossing, "counted": counted,
        "ray_index": 0 if ray is None else ray["ray_index"], "chief": ray is None,
        "ring_index": None if ray is None else ray["ring_index"], "ring_position_index": None if ray is None else ray["ring_position_index"],
        "launch_offset_x": 0.0 if ray is None else ray["launch_offset"][0], "launch_offset_y": 0.0 if ray is None else ray["launch_offset"][1],
        "power_weight": weight, "u_mm": u, "v_mm": v,
        "position_x_mm": point[0], "position_y_mm": point[1], "position_z_mm": point[2],
        "direction_x": direction[0], "direction_y": direction[1], "direction_z": direction[2],
        "direction_u": du, "direction_v": dv, "direction_n": dn, "opl_mm": opl, "opl_defined": opl_defined,
        "ray_end": None if ray is None else ray["end"], "ray_end_surface_id": None if ray is None else ray["end_surface_id"],
    })


def _arrived(path: dict[str, Any], step: int) -> list[dict[str, Any]]:
    """The pupil rays of ``path`` present at ``step`` (their own step lists are a prefix of the chief's)."""

    chief_surfaces = path["chief"]["surface_ids"]
    return [r for r in path["pupil"]["rays"] if len(r["points_mm"]) > step and r["surface_ids"][step] == chief_surfaces[step]]


def ray_rows(result: ValidationResult, ray_data: dict[str, Any], *, monitor: str | None = None, path: str | None = None,
             crossing_index: int | None = None) -> list[dict[str, Any]]:
    """Chief and pupil rays at every crossing of an observation plane (filtered by monitor / path / crossing index)."""

    rows = []
    paths = ray_data.get("paths", [])
    for observation in result.observations:
        if monitor is not None and observation.observation_id != monitor:
            continue
        axes, counted = _plane_axes(observation), _counted(observation)
        for p in paths:
            if path is not None and p["path_id"] != path:
                continue
            for crossing, step in enumerate(crossing_steps(p, observation.observation_id)):
                if crossing_index is not None and crossing != crossing_index:
                    continue
                chief = p["chief"]
                flag = counted.get((p["path_id"], crossing))
                opl_ok = bool(p["opl_defined"][step])
                weight = p["pupil"]["power_weight_per_ray"]
                rows.append(_ray_row(observation, p, crossing, flag, axes, chief["points_mm"][step], chief["directions"][step],
                                     chief["opl_mm"][step], ray=None, weight=None, opl_defined=opl_ok))
                for ray in _arrived(p, step):
                    rows.append(_ray_row(observation, p, crossing, flag, axes, ray["points_mm"][step], ray["directions"][step],
                                         ray["opl_mm"][step], ray=ray, weight=weight, opl_defined=opl_ok))
    return rows


# ------------------------------------------------------------------------------------------------------------ wavefront


#: A query that needs pupil rays, on a run whose ray data records the chief rays only (``ray_data`` ``summary``).
PUPIL_RAYS_NOT_RECORDED = "PUPIL_RAYS_NOT_RECORDED"
PUPIL_RAYS_NOT_RECORDED_MESSAGE = ("this run recorded the chief rays only (ray data recording 'summary'): run the Project "
                                   "again with --ray-data full (ray_data=\"full\") to record its pupil rays")


def _refusal(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"status": "UNAVAILABLE", "reason": code, "message": message, **context}


def _launch_terms(ray_data: dict[str, Any], path: dict[str, Any], result: ValidationResult) -> tuple[dict[int, float], str | None]:
    """Ray index -> optical path from the source's phase front to the ray's start point on the source plane (mm)."""

    by_id = {p["path_id"]: p for p in ray_data["paths"]}
    root = path
    while root["parent_path_id"] is not None:
        if root["parent_path_id"] not in by_id:
            return {}, "a parent path of this beam is missing from the ray data"
        root = by_id[root["parent_path_id"]]
    engine_root = next((p for p in result.paths if p.path_id == root["path_id"]), None)
    if engine_root is None or not engine_root.segments or root["chief"]["surface_ids"][0] is not None:
        return {}, "the source path of this beam is missing"
    n_source = engine_root.segments[0].refractive_index
    origin, axis = np.array(root["chief"]["points_mm"][0]), np.array(root["chief"]["directions"][0])
    terms = {}
    for ray in root["pupil"]["rays"]:
        if not ray["points_mm"] or ray["surface_ids"][0] is not None:
            continue
        rho = np.array(ray["points_mm"][0]) - origin
        rho = rho - float(dot(rho, axis)) * axis
        d = np.array(ray["directions"][0])
        cos = float(dot(d, axis))
        if cos <= FORWARD_COS_MIN:
            return {}, "a pupil ray leaves the source backwards"
        sigma = (d - cos * axis) / cos
        terms[ray["ray_index"]] = n_source * float(dot(sigma, rho)) / (math.sqrt(1.0 + float(dot(sigma, sigma))) + 1.0)
    return terms, None


def _medium_index(result: ValidationResult, path_id: str, observation_id: str, crossing: int) -> float | None:
    engine_path = next((p for p in result.paths if p.path_id == path_id), None)
    if engine_path is None:
        return None
    ends = [s for s in engine_path.segments if s.end_surface_id == OBSERVATION_PREFIX + observation_id]
    return ends[crossing].refractive_index if crossing < len(ends) else None


def _ray_focus(points: np.ndarray, directions: np.ndarray) -> np.ndarray | None:
    """Least-squares point closest to every ray (lines through ``points`` along ``directions``)."""

    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    normal = projectors.sum(axis=0)
    eigen = np.linalg.eigvalsh(normal)
    if eigen[0] <= RAY_FOCUS_MIN_EIGEN_FRACTION * eigen[-1]:
        return None
    return np.linalg.solve(normal, dot(projectors, points[:, None, :]).sum(axis=0))


def _gaussian_waist(arrival: ArrivalResult | None, hit: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray | None, str]:
    g = None if arrival is None else arrival.beam.gaussian
    if g is None or not g.valid or g.waist_distance_x_mm is None or g.waist_distance_y_mm is None:
        return None, "the beam has no valid Gaussian state at this crossing"
    rayleigh = 0.5 * ((g.rayleigh_range_x_mm or 0.0) + (g.rayleigh_range_y_mm or 0.0))
    if abs(g.waist_distance_x_mm - g.waist_distance_y_mm) > WAIST_MATCH_FRACTION_OF_RAYLEIGH * rayleigh:
        return None, "the beam is astigmatic: its x and y waists are at different distances"
    distance = 0.5 * (g.waist_distance_x_mm + g.waist_distance_y_mm)
    return hit - distance * direction, ""


def _to_plane(points: np.ndarray, directions: np.ndarray, hit: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    cos = dot(directions, normal)
    if bool((cos <= FORWARD_COS_MIN).any()):
        return None
    return dot(hit - points, normal) / cos


def _to_sphere(points: np.ndarray, directions: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray | None:
    rel = points - center
    b = dot(directions, rel)
    c = dot(rel, rel) - radius * radius
    disc = b * b - c
    if bool((disc < 0.0).any()):
        return None
    root = np.sqrt(disc)
    near, far = -b - root, -b + root
    return np.where(np.abs(near) <= np.abs(far), near, far)


def wavefront(result: ValidationResult, ray_data: dict[str, Any] | None, observation_id: str, path_id: str, *, crossing_index: int = 0,
              reference: str = "plane", center: Any = None, min_rays: int = MIN_WAVEFRONT_RAYS) -> dict[str, Any]:
    """The geometric wavefront of one beam at one crossing of an observation plane (module docstring).

    ``reference``: ``"plane"`` (perpendicular to the chief ray through its hit point) or ``"sphere"`` through the chief
    ray's hit point about ``center``: a design-frame point ``(x, y, z)`` in mm, ``"gaussian_waist"`` (the beam's Gaussian
    waist at this crossing; a stigmatic beam only -- a geometric sphere about the waist, so on a plane within the
    Rayleigh range, where the Gaussian phase front is nearly flat, it reports a large curvature residual: use ``plane``
    there) or ``"ray_focus"`` (the least-squares convergence point of the arrived pupil rays). Returns ``status`` ``OK`` with the summary and one row per pupil ray, or ``UNAVAILABLE`` with a
    ``reason`` code and a ``message``."""

    context = {"observation_id": observation_id, "path_id": path_id, "crossing_index": crossing_index, "model": WAVEFRONT_MODEL}
    if ray_data is None or ray_data.get("status") != "TRACED":
        return _refusal(RAY_DATA_NOT_AVAILABLE, "this run carries no ray data (ray_data.json)", **context)
    if ray_data_recording(ray_data) != RAY_DATA_FULL:
        return _refusal(PUPIL_RAYS_NOT_RECORDED, PUPIL_RAYS_NOT_RECORDED_MESSAGE, **context)
    observation = next((o for o in result.observations if o.observation_id == observation_id), None)
    path = next((p for p in ray_data["paths"] if p["path_id"] == path_id), None)
    steps = [] if observation is None or path is None else crossing_steps(path, observation_id)
    if crossing_index < 0 or crossing_index >= len(steps):
        return _refusal(CROSSING_NOT_FOUND, f"path {path_id!r} has no crossing {crossing_index} of observation {observation_id!r}", **context)
    step = steps[crossing_index]
    if not path["opl_defined"][step]:
        return _refusal(RAY_OPL_UNDEFINED, "per-ray optical path lengths are not defined after an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating", **context)
    rays = _arrived(path, step)
    if not rays or len(rays) < min_rays:  # RunResults refuses a min_rays below 1 (RESULTS_QUERY_INVALID)
        return _refusal(TOO_FEW_RAYS, f"{len(rays)} pupil rays reach the plane; at least {min_rays} are needed", **context)
    n_plane = _medium_index(result, path_id, observation_id, crossing_index)
    launch, problem = _launch_terms(ray_data, path, result)
    if n_plane is None or problem is not None or any(r["ray_index"] not in launch for r in rays):
        return _refusal(RAY_DATA_INCONSISTENT, problem or "the ray data and the Engine result do not describe the same beam", **context)
    chief = path["chief"]
    hit, direction = np.array(chief["points_mm"][step]), np.array(chief["directions"][step])
    points = np.array([r["points_mm"][step] for r in rays])
    directions = np.array([r["directions"][step] for r in rays])
    opl = np.array([r["opl_mm"][step] + launch[r["ray_index"]] for r in rays])
    reference_doc: dict[str, Any] = {"kind": reference}
    if reference == "plane":
        t = _to_plane(points, directions, hit, direction)
        reference_doc.update(point_mm=hit.tolist(), normal=direction.tolist())
    elif reference == "sphere":
        arrival = next((a for a in (observation.arrivals if observation else ()) if a.beam.path_id == path_id
                        and a.crossing_index == crossing_index), None)
        if isinstance(center, str) and center == "gaussian_waist":
            point, why = _gaussian_waist(arrival, hit, direction)
        elif isinstance(center, str) and center == "ray_focus":
            point, why = _ray_focus(points, directions), "the pupil rays are parallel: a collimated bundle has no focus"
        elif center is not None and not isinstance(center, str) and len(center) == len("xyz"):
            point, why = np.array([float(c) for c in center]), ""
        else:
            return _refusal(REFERENCE_UNDEFINED, "a sphere reference needs center: a point (x, y, z) in mm, 'gaussian_waist' or "
                                                 "'ray_focus'", **context)
        if point is None or not bool(np.isfinite(point).all()):
            return _refusal(REFERENCE_UNDEFINED, why, **context)
        radius = float(np.sqrt(dot(hit - point, hit - point)))
        if radius < MIN_REFERENCE_RADIUS_MM:
            return _refusal(REFERENCE_UNDEFINED, "the sphere centre lies on the chief ray's hit point", **context)
        t = _to_sphere(points, directions, point, radius)
        reference_doc.update(center_mm=point.tolist(), radius_mm=radius,
                             center_from=center if isinstance(center, str) else "point",
                             converging=bool(float(dot((point - hit), direction)) > 0.0))
    else:
        return _refusal(REFERENCE_UNDEFINED, f"unknown reference {reference!r}: 'plane' or 'sphere'", **context)
    if t is None:
        return _refusal(RAY_MISSES_REFERENCE, "a pupil ray does not reach the reference surface", **context)
    wavelength_mm = path["wavelength_nm"] * NM_TO_MM
    chief_opl = chief["opl_mm"][step]  # the chief ray starts on the phase front and lies on both reference surfaces
    opd_mm = opl + n_plane * t - chief_opl
    piston = float(opd_mm.mean())
    opd_mm = opd_mm - piston
    opd_waves = opd_mm / wavelength_mm
    origin = np.array(observation.plane.origin_mm) if observation.plane else None
    rows = []
    for ray, waves, mm, p in zip(rays, opd_waves, opd_mm, points):
        u = v = None
        if origin is not None:
            u, v = float(dot(p - origin, np.array(observation.plane.u_axis))), float(dot(p - origin, np.array(observation.plane.v_axis)))
        rows.append(_row("wavefront", {
            "ray_index": ray["ray_index"], "ring_index": ray["ring_index"], "ring_position_index": ray["ring_position_index"],
            "pupil_x": ray["launch_offset"][0], "pupil_y": ray["launch_offset"][1], "u_mm": u, "v_mm": v,
            "opd_waves": float(waves), "opd_mm": float(mm),
        }))
    return {
        "status": "OK", **context, "wavelength_nm": path["wavelength_nm"], "medium_index": n_plane, "reference": reference_doc,
        "ray_count": len(rays), "pupil_ray_count": path["pupil"]["ray_count"], "piston_waves": piston / wavelength_mm,
        "rms_waves": float(math.sqrt(float(np.mean(opd_waves**2)))), "pv_waves": float(opd_waves.max() - opd_waves.min()),
        "pupil_coordinates": "launch offsets at the source plane in units of the source's 1/e^2 radii (x along the source frame "
                             "x, y along y)",
        "rays": rows,
    }
