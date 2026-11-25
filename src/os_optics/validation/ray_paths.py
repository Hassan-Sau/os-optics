"""Chief and pupil ray paths of the Engine's own trace: ``validate_with_ray_paths`` / ``trace_ray_paths``.

``ValidationResult.paths`` carries each beam path's chief-ray segments. The tracer also carries deterministic pupil
sample rays alongside the chief (8 rings x 12 points by default). This module exports where each of
those rays went, for drawing a focusing beam:

* ``validate_with_ray_paths(request)`` runs ``validate`` once with a ``TraceObserver`` attached to its physics trace.
  The observer copies the per-ray positions, directions and alive masks the tracer produced at each step (source or
  branch point, every surface reached, every interaction, path end). It never writes to the tracer, so the returned
  ``ValidationResult`` is exactly ``validate(request)``'s.
* The ``RayPathTrace`` carries the run's ``input_hash`` and ``trace_hash`` (equal to its ``ValidationManifest``'s), which
  binds the rays to that result.
* Nothing here refracts, intersects or propagates: every point is a tracer position.

Pupil ray ends: ``PATH_END`` (alive when its path ended; ``exit_direction`` is set only when the path escaped),
``STOPPED_AT_SURFACE`` (removed at its last point: aperture, grazing incidence, absorber, detector, ...) or
``NOT_FOLLOWED`` (it would next reach a different surface than the chief, which the tracer does not follow; its
``exit_direction`` is its direction at its last point and ``next_surface_id`` the surface it would reach, ``None`` if
none; ``counted_with_chief`` says it would otherwise have reached the chief's surface inside its clear aperture, so the
path's power and Gaussian beam still count it). A branch carries only the rays alive where it starts.

Per point every ray also carries the optical path length from the source (``opl_mm``, mm: sum of n x geometric length,
the same accumulation as the chief's ``OpticalPath.optical_path_length_mm``) and its propagation direction leaving the
point (``directions``: after that point's interaction; a ray stopped there keeps the direction it arrived with).
``opl_defined[k]`` is False from the step after a thin lens, a Faraday rotator or a diffraction grating on (its per-ray optical path is not modelled). Every pupil
ray carries the same share ``1 / pupil_ray_count`` of the sampled power (equal-power rings, ``optics/rays/pupil.py``) and its
``launch_offset``: its source-plane offset in units of the source's 1/e^2 radii (x along the source frame x, y along y).

``ray_paths_document(trace)`` is the JSON document of a trace (the run directory's ``ray_data.json``). Its recording level
(``RAY_DATA_LEVELS``) says what it holds: ``full`` (the default) every chief and pupil ray as above; ``summary`` the chief
rays only, with each path's pupil sampling (ray count, rings, power weight per ray) but not its pupil rays -- about 1 % of
the full document's size. The trace and its sampling are the same either way: the level decides only what is written,
so the Engine input and result hashes do not depend on it. A ``summary`` document states it (``"recording": "summary"``);
a query that needs pupil rays refuses it (``PUPIL_RAYS_NOT_RECORDED``) instead of reading fewer rays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..contracts.optics import OpticalPath
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..evaluation.capabilities import SECONDARY_RAYS
from ..optics.rays.path_ids import BRANCH_SUFFIX_PATTERN  # noqa: F401  (the path-id grammar, for the optical view)
from ..optics.rays.pupil import pupil_offsets
from ..optics.rays.sampled import SampledTrace
from ..optics.rays.tracer import ESCAPED, TraceObserver, _Packet
from ..run_control import checkpoint, phase
from .pipeline import _schema_ok, evaluate

if TYPE_CHECKING:
    from ..optics.rays.scatter_trace import ScatterTrace

RAY_PATHS_VERSION = "2.0"
RAY_DATA_SCHEMA = "opensci.ray_data"
#: Ray data recording levels: every chief and pupil ray, or the chief rays only (module docstring).
RAY_DATA_FULL = "full"
RAY_DATA_SUMMARY = "summary"
RAY_DATA_LEVELS = (RAY_DATA_FULL, RAY_DATA_SUMMARY)

TRACED = "TRACED"
NOT_TRACED = "NOT_TRACED"
#: Why a recorder that holds part of a stopped beam trace gives no ray paths, and what to do.
BEAM_RECORD_OPEN = ("the recorder holds part of a beam trace a monitor stopped: give the recorder to the evaluation that "
                    "continues it (recorder=...), then read the ray paths of that evaluation's result")
#: The same for the secondary rays traced from a beam trace.
SECONDARY_RECORD_OPEN = ("the recorder holds part of the secondary rays a monitor stopped: give the recorder to the evaluation "
                         "that continues it (recorder=...), then read the ray paths of that evaluation's result")
PATH_END = "PATH_END"
STOPPED_AT_SURFACE = "STOPPED_AT_SURFACE"
NOT_FOLLOWED = "NOT_FOLLOWED"

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PupilRayPath:
    """One pupil sample ray of a path. ``ray_index`` 1..N (0 is the chief); ``ring_index`` / ``ring_position_index``
    locate it in the source's pupil rings. ``points_mm``: its tracer positions (design frame, mm) at the steps it reached;
    ``end``: ``PATH_END`` | ``STOPPED_AT_SURFACE`` | ``NOT_FOLLOWED`` (module docstring); ``next_surface_id``: for a
    ``NOT_FOLLOWED`` ray, the surface it would have reached next instead of the chief's (``None``: none, or another end);
    ``counted_with_chief``: a ``NOT_FOLLOWED`` ray that would otherwise have reached the chief's surface inside its clear
    aperture (the path's power and Gaussian beam count it as travelling with the chief)."""

    ray_index: int
    ring_index: int
    ring_position_index: int
    points_mm: tuple[Vec3, ...]
    end: str
    end_surface_id: str | None
    exit_direction: Vec3 | None
    next_surface_id: str | None = None
    counted_with_chief: bool = False
    surface_ids: tuple[str | None, ...] = ()
    directions: tuple[Vec3, ...] = ()
    opl_mm: tuple[float | None, ...] = ()  # (None at a step whose ``RayPath.opl_defined`` is false)
    launch_offset: tuple[float, float] = (0.0, 0.0)
    # the steps of the path (``RayPath.chief_surface_ids`` indices) at which the ray was recorded (``points_mm``): an
    # observation plane a live ray does not cross before the surface it meets has no point of it (not exported)
    steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class RayPath:
    """Chief and pupil rays of one Engine ``OpticalPath`` (same ``path_id``). ``chief_points_mm[k]`` is the chief position
    at step ``k`` and ``chief_surface_ids[k]`` its surface (``None`` at a source). ``pupil_ray_count`` is the number of pupil
    rays the packet was created with; ``pupil_rays`` holds those alive at this path's start."""

    path_id: str
    source_id: str
    source_component_id: str
    spectral_index: int
    wavelength_nm: float
    parent_path_id: str | None
    termination: str
    termination_surface_id: str | None
    chief_points_mm: tuple[Vec3, ...]
    chief_surface_ids: tuple[str | None, ...]
    chief_exit_direction: Vec3 | None
    pupil_ray_count: int
    pupil_rays: tuple[PupilRayPath, ...]
    chief_directions: tuple[Vec3, ...] = ()
    chief_opl_mm: tuple[float | None, ...] = ()  # (None at a step whose ``opl_defined`` is false)
    opl_defined: tuple[bool, ...] = ()
    ring_count: int = 0
    points_per_ring: int = 0


@dataclass(frozen=True)
class RayPathTrace:
    """Every path the Engine kept for one request, sorted by ``path_id``. ``status`` ``NOT_TRACED`` (with ``message``,
    no paths, ``trace_hash`` None) when the validation run produced no trace."""

    input_hash: str
    trace_hash: str | None
    status: str
    message: str = ""
    paths: tuple[RayPath, ...] = ()
    ray_paths_version: str = RAY_PATHS_VERSION
    # the extended emitters' rays of this trace, for the model-scope check (``local/validity.py``): not part of the document
    sampled: SampledTrace | None = field(default=None, compare=False, repr=False)
    # the rays of the light the surfaces scattered, origin by origin (``ScatterLine.scope``, ``rows``), for the same check:
    # not part of the document
    scattered: ScatterTrace | None = field(default=None, compare=False, repr=False)


@dataclass
class _Recording:
    """Per-step copies for one packet: step ``k`` starts with ``points[k]`` / ``reached[k]`` at ``surfaces[k]`` and is
    closed with ``survived[k]`` / ``directions[k]`` (the per-ray state after its interaction)."""

    source_component_id: str
    points_per_ring: int
    pupil_count: int = 0
    ring_count: int = 0
    surfaces: list[str | None] = field(default_factory=list)
    points: list[NDArray[np.float64]] = field(default_factory=list)
    opl: list[NDArray[np.float64]] = field(default_factory=list)
    opl_ok: list[bool] = field(default_factory=list)
    # with ``keep_fields``: each ray's Jones vector at the step's start (its field: what a sample's amplitude and the coatings
    # left of the launched field), read by the imaging analysis' pupil amplitudes; a run's ray data does not keep them
    keep_fields: bool = False
    fields: list[NDArray[np.complex128]] = field(default_factory=list)
    reached: list[NDArray[np.bool_]] = field(default_factory=list)
    survived: list[NDArray[np.bool_]] = field(default_factory=list)
    directions: list[NDArray[np.float64]] = field(default_factory=list)
    # ray index -> (surface a DIVERGED ray would reach instead, whether it would otherwise have travelled with the chief)
    diverged_to: dict[int, tuple[str | None, bool]] = field(default_factory=dict)
    # the rays still in the beam when the path ended (a ray retired as the beam leaves the scene is not)
    final: NDArray[np.bool_] | None = None

    def start(self, surface_id: str | None, pk: _Packet) -> None:
        self.surfaces.append(surface_id)
        self.points.append(pk.o.copy())
        self.opl.append(pk.opl.copy())
        self.opl_ok.append(not pk.opl_undefined)
        if self.keep_fields:
            self.fields.append(pk.jones.copy())
        self.reached.append(pk.alive.copy())

    def close_step(self, survived: NDArray[np.bool_], directions: NDArray[np.float64]) -> None:
        if len(self.survived) < len(self.points):
            self.survived.append(survived.copy())
            self.directions.append(directions.copy())


class _RayPathRecorder:
    """``TraceObserver`` that keeps a ``_Recording`` per packet (by ``path_id``, unique within a trace).

    The record belongs to the traces it observed last: the paths of the beam trace (``beam``) and the paths and sampled
    rays of the secondary rays traced from it (``secondary``, ``sampled``) are kept apart. A beam trace that starts again
    (``beam_trace_started``) drops both records, secondary rays that start again (``secondary_rays_started``) drop theirs:
    a recorder never holds the rays of two traces. A beam trace that started and did not finish (a monitor stopped it)
    leaves the record open (``beam_open``) until the evaluation that continues it is given the recorder and finishes the
    trace (``beam_trace_finished``), and so do secondary rays that started and did not finish (``secondary_open``,
    ``secondary_rays_finished``); ``recorded_ray_paths`` answers ``NOT_TRACED`` for an open record."""

    def __init__(self, keep_fields: bool = False) -> None:
        self.keep_fields = keep_fields
        self._live: dict[str, _Recording] = {}
        self.beam: list[tuple[OpticalPath, _Recording]] = []
        self.secondary: list[tuple[OpticalPath, _Recording]] = []
        self._current = self.beam
        self.sampled: SampledTrace | None = None
        self.scattered: ScatterTrace | None = None
        # (a tracer driven directly, without beam_trace_started, records a whole trace: the record is not open)
        self.beam_open = False
        self.secondary_open = False

    @property
    def finished(self) -> list[tuple[OpticalPath, _Recording]]:
        """Every path recorded: the beam trace's, then the secondary rays'."""
        return [*self.beam, *self.secondary]

    def beam_trace_started(self) -> None:
        """A beam trace starts: what an earlier trace left here (its paths, and the secondary rays traced from it) is dropped."""
        self._live.clear()
        self.beam.clear()
        self.secondary.clear()
        self.sampled = None
        self.scattered = None
        self._current = self.beam
        self.beam_open = True
        self.secondary_open = False

    def beam_trace_finished(self) -> None:
        """The beam trace that started last finished: its record is whole."""
        self.beam_open = False

    def secondary_rays_started(self) -> None:
        """The secondary rays of the observed beam trace start: what an earlier pass of them left here is dropped."""
        self._live.clear()
        self.secondary.clear()
        self.sampled = None
        self.scattered = None
        self._current = self.secondary
        self.secondary_open = True

    def secondary_rays_finished(self) -> None:
        """The secondary rays that started last finished: their record is whole."""
        self.secondary_open = False

    def sampled_finished(self, sampled: SampledTrace) -> None:
        self.sampled = sampled

    def scattered_finished(self, scattered: ScatterTrace) -> None:
        """The rays of the light the surfaces scattered are traced (``scatter_trace.ScatterTrace``). Read only."""
        self.scattered = scattered

    def packet_started(self, pk: _Packet, surface_id: str | None) -> None:
        rec = _Recording(pk.src.owner_id, pk.src.points_per_ring, pk.pupil_total, pk.src.ring_count, keep_fields=self.keep_fields)
        rec.start(surface_id, pk)
        self._live[pk.path_id] = rec

    def pupil_rays_diverged(self, pk: _Packet, rays: NDArray[np.intp], next_surface_ids: tuple[str | None, ...],
                            delivered: NDArray[np.bool_]) -> None:
        rec = self._live.get(pk.path_id)
        if rec is not None:
            rec.diverged_to.update((int(r), (surface, bool(ok))) for r, surface, ok in zip(rays, next_surface_ids, delivered))

    def packet_advanced(self, pk: _Packet, surface_id: str | None) -> None:
        rec = self._live.get(pk.path_id)
        if rec is None:
            return
        # Only the start step (source or branch point) can still be open here: nothing has interacted since it started,
        # so every ray present there leaves it in the direction the packet travels. Later steps close in packet_interacted.
        rec.close_step(rec.reached[-1], pk.d)
        if surface_id is not None:
            rec.start(surface_id, pk)

    def packet_interacted(self, pk: _Packet) -> None:
        rec = self._live.get(pk.path_id)
        if rec is not None:
            rec.close_step(pk.alive, pk.d)

    def branch_spawned(self, parent: _Packet, child: _Packet) -> None:
        """Nothing to record: a branch's ray paths start at its ``packet_started``."""

    def path_finished(self, pk: _Packet, path: OpticalPath) -> None:
        rec = self._live.pop(pk.path_id, None)
        if rec is not None:
            rec.close_step(pk.alive, pk.d)
            rec.final = pk.alive.copy()
            self._current.append((path, rec))

    def path_dropped(self, pk: _Packet) -> None:
        self._live.pop(pk.path_id, None)


def _vec(v: NDArray[np.float64]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _pupil_rays(rec: _Recording, termination: str) -> tuple[PupilRayPath, ...]:
    last_step = len(rec.points) - 1
    rays = []
    launch = pupil_offsets(1.0, 1.0, rec.ring_count, rec.points_per_ring) if rec.pupil_count else np.zeros((0, 2))
    for i in range(1, rec.pupil_count + 1):
        reached_steps = [k for k in range(len(rec.points)) if bool(rec.reached[k][i])]
        if not reached_steps:  # removed before this branch started: it belongs to an ancestor path only
            continue
        j = reached_steps[-1]  # (a live ray an observation plane does not sample is not recorded there, and goes on)
        points = tuple(_vec(rec.points[k][i]) for k in reached_steps)
        exit_direction, next_surface, counted = None, None, False
        if not bool(rec.survived[j][i]):
            end, end_surface = STOPPED_AT_SURFACE, rec.surfaces[j]
        elif bool(rec.survived[last_step][i]) and (rec.final is None or bool(rec.final[i])):
            end, end_surface = PATH_END, None
            if termination == ESCAPED:
                exit_direction = _vec(rec.directions[j][i])
        else:
            end, end_surface = NOT_FOLLOWED, rec.surfaces[j]
            exit_direction = _vec(rec.directions[j][i])
            next_surface, counted = rec.diverged_to.get(i, (None, False))
        ring, position = divmod(i - 1, rec.points_per_ring)
        rays.append(PupilRayPath(
            i, ring, position, points, end, end_surface, exit_direction, next_surface, counted,
            surface_ids=tuple(rec.surfaces[k] for k in reached_steps),
            directions=tuple(_vec(rec.directions[k][i]) for k in reached_steps),
            opl_mm=tuple(float(rec.opl[k][i]) if rec.opl_ok[k] else None for k in reached_steps),
            launch_offset=(float(launch[i - 1][0]), float(launch[i - 1][1])), steps=tuple(reached_steps),
        ))
    return tuple(rays)


def _ray_path(path: OpticalPath, rec: _Recording) -> RayPath:
    return RayPath(
        path_id=path.path_id, source_id=path.source_id, source_component_id=rec.source_component_id,
        spectral_index=path.spectral_index, wavelength_nm=path.wavelength_nm, parent_path_id=path.parent_path_id,
        termination=path.termination, termination_surface_id=path.termination_surface_id,
        chief_points_mm=tuple(_vec(p[0]) for p in rec.points), chief_surface_ids=tuple(rec.surfaces),
        chief_exit_direction=_vec(rec.directions[-1][0]) if path.termination == ESCAPED else None,
        pupil_ray_count=rec.pupil_count, pupil_rays=_pupil_rays(rec, path.termination),
        chief_directions=tuple(_vec(d[0]) for d in rec.directions), chief_opl_mm=tuple(float(o[0]) if ok else None for o, ok in zip(rec.opl, rec.opl_ok)),
        opl_defined=tuple(rec.opl_ok), ring_count=rec.ring_count, points_per_ring=rec.points_per_ring,
    )


def validate_with_ray_paths(request: ValidationRequest) -> tuple[ValidationResult, RayPathTrace]:
    """``validate(request)`` plus the chief / pupil ray paths of its one physics trace (never raises, like ``validate``).

    The result is exactly ``validate(request)``'s. The trace is ``NOT_TRACED`` when that run produced no trace: an
    unsupported ``project_schema_version``, ``run_physics`` disabled, or a run that failed (``message`` names the issue).
    """
    recorder = _RayPathRecorder()
    return recorded_ray_paths(request, evaluate(request, recorder=recorder), recorder)


def ray_path_recorder() -> TraceObserver:
    """A recorder of the beam trace's rays: pass it to ``os_optics.evaluate(request, ..., recorder=recorder)`` (with the
    ``EvaluationState`` of every later evaluation that reuses that trace), then read ``recorded_ray_paths``. It holds the
    record of the latest beam trace it observed and of the latest secondary rays traced from it: a trace made again for
    it, or continued after a monitor stopped it, replaces what the recorder held. After a monitor stopped an evaluation
    inside the beam trace or the secondary rays, give the recorder to the evaluation that continues it: a recorder that
    holds part of a beam trace, or of the secondary rays traced from it, a monitor stopped answers ``NOT_TRACED`` in
    ``recorded_ray_paths``."""
    return _RayPathRecorder()


def recorded_ray_paths(request: ValidationRequest, result: ValidationResult, recorder: TraceObserver) -> tuple[ValidationResult, RayPathTrace]:
    """``result`` and the ``RayPathTrace`` of the beam trace ``recorder`` observed (``ray_path_recorder``) in an evaluation
    of ``request`` that gave ``result``, or in an earlier evaluation over the same state whose trace it reused. The
    extended emitters' kept rays are part of it when ``result`` is of an evaluation that ran the secondary rays (the rays
    of scattered light are not recorded as paths; what the model-scope check reads of them is, ``scattered``): the paths
    are then those ``result.manifest.trace_hash`` covers, whatever else
    the recorder observed before. ``NOT_TRACED`` when the recorder holds part of a beam trace, or of the secondary rays
    traced from it, a monitor stopped and the evaluation that continued it was not given the recorder."""
    assert isinstance(recorder, _RayPathRecorder)
    manifest = result.manifest
    input_hash = "" if manifest is None else manifest.input_hash
    trace_hash = None if manifest is None else manifest.trace_hash
    if not _schema_ok(request.project.project_schema_version):
        return result, RayPathTrace(input_hash, None, NOT_TRACED, "unsupported project_schema_version")
    if not request.configuration.run_physics:
        return result, RayPathTrace(input_hash, None, NOT_TRACED, "run_physics is disabled in the request")
    if trace_hash is None:
        message = "; ".join(f"{i.code.value}: {i.message}" for i in result.issues)
        return result, RayPathTrace(input_hash, None, NOT_TRACED, message)
    if recorder.beam_open:
        return result, RayPathTrace(input_hash, None, NOT_TRACED, BEAM_RECORD_OPEN)
    with_secondary = result.evaluation is None or SECONDARY_RAYS in result.evaluation.executed
    if with_secondary and recorder.secondary_open:
        return result, RayPathTrace(input_hash, None, NOT_TRACED, SECONDARY_RECORD_OPEN)
    finished = sorted(recorder.finished if with_secondary else recorder.beam, key=lambda item: item[0].path_id)
    with phase("ray_paths", total=len(finished), unit="beam paths"):
        paths = tuple(_counted(_ray_path(path, rec)) for path, rec in finished)
    return result, RayPathTrace(input_hash, trace_hash, TRACED, "", paths, sampled=recorder.sampled if with_secondary else None,
                                scattered=recorder.scattered if with_secondary else None)


def _counted(path: RayPath) -> RayPath:
    checkpoint()  # (one more path's rays gathered: the run's monitor may stop the run here, os_optics/run_control.py)
    return path


def trace_ray_paths(request: ValidationRequest) -> RayPathTrace:
    """The ``RayPathTrace`` of ``validate_with_ray_paths(request)``."""
    return validate_with_ray_paths(request)[1]


def _pupil_ray_document(ray: PupilRayPath) -> dict[str, Any]:
    return {
        "ray_index": ray.ray_index, "ring_index": ray.ring_index, "ring_position_index": ray.ring_position_index,
        "launch_offset": list(ray.launch_offset), "surface_ids": list(ray.surface_ids),
        "points_mm": [list(p) for p in ray.points_mm], "directions": [list(d) for d in ray.directions], "opl_mm": list(ray.opl_mm),
        "end": ray.end, "end_surface_id": ray.end_surface_id,
        "exit_direction": None if ray.exit_direction is None else list(ray.exit_direction),
        "next_surface_id": ray.next_surface_id, "counted_with_chief": ray.counted_with_chief,
    }


def ray_paths_document(trace: RayPathTrace, recording: str = RAY_DATA_FULL) -> dict[str, Any]:
    """The JSON document of a ``RayPathTrace`` (``opensci.ray_data`` ``RAY_PATHS_VERSION``): every chief and pupil ray the
    Engine traced, per path, with points (design frame, mm), directions and optical path lengths. Plain JSON values.
    ``recording``: ``full`` (every ray) or ``summary`` (the chief rays and each path's pupil sampling, no pupil rays;
    module docstring)."""

    document = ray_paths_parts(trace, recording)
    return {**document, "paths": list(document["paths"].items)}


def ray_paths_parts(trace: RayPathTrace, recording: str = RAY_DATA_FULL) -> dict[str, Any]:
    """``ray_paths_document`` with its ``paths`` given path by path (``manifest.streaming.Elements``): written or hashed
    without holding the whole document."""

    from ..manifest.streaming import Elements

    if recording not in RAY_DATA_LEVELS:
        raise ValueError(f"ray data recording is one of {', '.join(RAY_DATA_LEVELS)}, not {recording!r}")
    full = recording == RAY_DATA_FULL
    return {
        "schema": RAY_DATA_SCHEMA, "schema_version": trace.ray_paths_version, "status": trace.status, "message": trace.message,
        "input_hash": trace.input_hash, "trace_hash": trace.trace_hash, "length_unit": "mm", "frame": "DESIGN",
        **({} if full else {"recording": RAY_DATA_SUMMARY}),
        "paths": Elements(_path_document(path, full) for path in trace.paths),
    }


def _path_document(path: RayPath, full: bool) -> dict[str, Any]:
    return {
        "path_id": path.path_id, "source_id": path.source_id, "source_component_id": path.source_component_id,
        "spectral_index": path.spectral_index, "wavelength_nm": path.wavelength_nm, "parent_path_id": path.parent_path_id,
        "termination": path.termination, "termination_surface_id": path.termination_surface_id,
        "chief": {"surface_ids": list(path.chief_surface_ids), "points_mm": [list(p) for p in path.chief_points_mm],
                  "directions": [list(d) for d in path.chief_directions], "opl_mm": list(path.chief_opl_mm),
                  "exit_direction": None if path.chief_exit_direction is None else list(path.chief_exit_direction)},
        "opl_defined": list(path.opl_defined),
        "pupil": {"ray_count": path.pupil_ray_count, "ring_count": path.ring_count, "points_per_ring": path.points_per_ring,
                  "power_weight_per_ray": (1.0 / path.pupil_ray_count) if path.pupil_ray_count else None,
                  **({"rays": [_pupil_ray_document(r) for r in path.pupil_rays]} if full else {})},
    }


def ray_data_recording(document: Any) -> str:
    """The recording level of a ray data document (``full`` where it states none)."""

    level = document.get("recording", RAY_DATA_FULL) if isinstance(document, dict) else RAY_DATA_FULL
    return level if isinstance(level, str) else RAY_DATA_FULL
