"""The estimates of a local run's memory and run-file size: before the trace from its beams, after the trace from the
beam paths and ray steps it counted.

A run's memory and run files grow with the light it traces: every beam (one emission line of a Gaussian-beam source)
launches a chief ray and its pupil rays, every surface a beam splits at adds a beam path, and each path is recorded ray by
ray, evaluated on the planes it reaches and drawn in the 3D scene. Two estimates are made (``RunMonitor.preflight``):

* **Before the trace** (``estimate_run_resources``), from the beams alone, with the per-beam costs measured on a reference
  scene (``REFERENCE``: 2500 independent 852 nm beams through a shared objective, two paths a beam, 8 x 12 pupil rays)::

      memory    = the process's memory once the Project is lowered (its catalog read) + beams x RUN_MEMORY_PER_BEAM x
                  (pupil rays + 1) / REFERENCE_RAYS (+ the run files, when they are held in memory: the JSON API); in a
                  browser (Pyodide) the WebAssembly memory once lowered + beams x RUN_MEMORY_PER_BEAM_WASM x (pupil rays + 1)
                  / REFERENCE_RAYS
      run files = beams x (the bytes per beam of every file but ray_data.json, and ray_data.json's at the recording level)

  It holds for beams like the reference's and is a bound in neither direction: a beam that splits into more paths or
  meets more surfaces, and a Project with elements beside its beams, costs more, and only the trace says how much; beams
  that meet no element cost less (``PRE_TRACE_MEASURED_RATIO``: the measured runs wrote 0.57 to 234 x the run files
  estimated from their beams).
* **After the trace** (``estimate_after_trace``), from what the trace counted -- its beam paths and their steps (a path's segments: one to each
  surface or plane it reaches; ``run_control.traced``) -- and the Project's optical elements, with the costs fitted to measured runs
  (``AFTER_TRACE_MEASURED``; full and summary ray data, the scene with and without pupil rays drawn)::

      ray_data.json   full:    paths x RAY_FULL_BYTES_PER_PATH x (pupil rays + 1) / REFERENCE_RAYS
                               + steps x pupil rays x RAY_FULL_BYTES_PER_PUPIL_STEP
                      summary: paths x RAY_SUMMARY_BYTES_PER_PATH + steps x RAY_SUMMARY_BYTES_PER_STEP
      other files     OTHER_FILES_BYTES + steps x OTHER_FILES_BYTES_PER_STEP + elements x OTHER_FILES_BYTES_PER_ELEMENT (steps x SCENE_PUPIL_RING_BYTES_PER_STEP less without
                      pupil rays drawn)
      memory          the memory after the trace + paths x MEMORY_PER_PATH + steps x MEMORY_PER_STEP + elements x
                      MEMORY_PER_ELEMENT (+ the run files, when they are held in memory)

  stated with a range whose high end covers the measured runs (``AFTER_TRACE_MEASURED_RATIO``: 0.66 to 1.42 x the
  estimate): run files up to ``AFTER_TRACE_FILES_HIGH`` x the estimate; memory up to the memory after the trace plus ``AFTER_TRACE_MEMORY_HIGH`` x
  the estimated growth + ``AFTER_TRACE_MEMORY_MARGIN``.

  The low end of the range is not fitted: it is what the run writes for certain (``certain_run_file_bytes``) -- the Project
  file and the Engine request file, whose bytes are known exactly once the Project is lowered, and in the result file each
  traced beam path and each of its steps (``OpticalPath`` and ``RaySegment`` records) at no fewer bytes than the shortest
  record of its kind (every field at its shortest JSON value; ``shortest_record_bytes``) -- and the memory read when the trace
  ended (the run's peak is at least that reading).

``RunMonitor`` refuses a run before the trace when its estimate from its beams exceeds a limit, and after the trace only
when the low end does: the run would write at least that much, or has used at least that much memory. The checks during
the run (the memory in use, the bytes written) stop a run that reaches a limit whatever an estimate said. The memory costs
are those of a CPython process (Linux resident set, Windows working set); after the trace they are used in a browser too,
where they were not measured.

Each estimate states how it was made, as fields (``estimate_basis``): ``method`` (``PER_BEAM_REFERENCE`` /
``COUNTED_PATHS_AND_STEPS``), ``reference`` (a key of ``REFERENCES``: the measured runs its costs come from),
``measured_ratio_low`` / ``measured_ratio_high`` (the run files the measured runs wrote, as multiples of their estimate's
``artifact_bytes``), ``memory_platform`` (whose memory costs it uses: ``CPYTHON`` or ``BROWSER_WASM``; ``None`` without a
memory estimate) and ``lower_bounds`` (the fields of the estimate that are counted, not fitted: the run needs at least
that much). ``estimate_statement`` words a refusal from them.
"""

from __future__ import annotations

import enum
import json
import sys
import typing
from typing import Any

from pydantic import BaseModel

from ..contracts.optics import OpticalPath, RaySegment
from ..contracts.request import ValidationRequest
from ..manifest.streaming import Elements, json_bytes
from ..validation.ray_paths import RAY_DATA_FULL

#: Where the per-beam costs below were measured.
REFERENCE = ("2500 independent 852 nm beams through a shared objective, two paths a beam, 8 x 12 pupil rays; os-optics "
             "0.13.0, Linux x86-64")
#: The pupil rays and the chief ray of a beam in the reference (``NumericalConfig`` 8 rings x 12 points, + 1).
REFERENCE_RAYS = 97
#: Memory a beam adds to the run (bytes): the trace's recorded rays, results and 3D scene of the reference -- the resident
#: set of a CPython process.
RUN_MEMORY_PER_BEAM = 603_573
#: The same in a browser (Pyodide): the WebAssembly memory a beam adds, run files held in memory included, measured with
#: 625 beams of the reference scene in Chrome (Designer, os-optics 0.13.0).
RUN_MEMORY_PER_BEAM_WASM = 327_262
#: Run-file bytes a beam adds: ``ray_data.json`` at each recording level, and every other file together (the scene drawn
#: with the outer ring of pupil rays, the default; without pupil rays it is ``SCENE_PUPIL_RING_BYTES_PER_BEAM`` less).
RAY_DATA_BYTES_PER_BEAM = {"full": 314_786, "summary": 3_344}
OTHER_FILE_BYTES_PER_BEAM = 57_707
SCENE_PUPIL_RING_BYTES_PER_BEAM = 19_980
#: The run files of the measured runs against the estimate before the trace (``tests/test_run_control.py``
#: ``MEASURED_RUNS`` with the beams of each): from 0.57 x (three beams that meet no element, full ray data: 0.58 rounded down) over 1.0 x
#: (the reference scene), 1.0 to 2.2 x (the reference designs of ``project example`` with full ray data) and 5.8 / 30 /
#: 60 / 92 x (one beam through 4 / 16 / 32 / 64 window-and-tap stages: 5 / 17 / 23 / 23 beam paths) to 234 x (one beam
#: and 300 waveplates beside it, summary ray data). The estimate is a bound in neither direction.
PRE_TRACE_MEASURED_RATIO = (0.57, 234.0)

#: Where the costs after the trace were fitted (os-optics 0.13.0, Windows 10, CPython 3.14; run-file bytes are the same on
#: every platform; tests/test_run_control.py holds the measurements).
AFTER_TRACE_MEASURED = ("the reference scene at 625 and 2500 beams, the 10 reference designs and one beam through 4 to 64 "
                        "window-and-tap stages")
#: ray_data.json at the full level: bytes per beam path (at REFERENCE_RAYS rays a beam) and per step of each pupil ray.
RAY_FULL_BYTES_PER_PATH = 67_851
RAY_FULL_BYTES_PER_PUPIL_STEP = 320
#: ray_data.json at the summary level: bytes per beam path and per step.
RAY_SUMMARY_BYTES_PER_PATH = 851
RAY_SUMMARY_BYTES_PER_STEP = 270
#: Every other run file together (result, record, request, Project, scene JSON and GLB, manifest): a fixed part and the
#: bytes per step and per optical element (the beam paths add nothing beyond their steps); the scene's outer ring of pupil
#: rays per step.
OTHER_FILES_BYTES = 80_642
OTHER_FILES_BYTES_PER_STEP = 9_811
OTHER_FILES_BYTES_PER_ELEMENT = 62_169
SCENE_PUPIL_RING_BYTES_PER_STEP = 3_343
#: The run files of the measured runs against the estimate after the trace (``tests/test_run_control.py``
#: ``MEASURED_RUNS``): from 0.66 x (a small design recorded with summary ray data and no pupil rays drawn; many thin
#: elements beside the beam 0.76 x) to 1.42 x (many lenses beside the beam). The high end of the range stated; its low
#: end is what the run writes for certain (``certain_run_file_bytes``).
AFTER_TRACE_MEASURED_RATIO = (0.66, 1.42)
AFTER_TRACE_FILES_HIGH = 1.5
#: Memory a run adds after its trace (its peak above the memory read when the trace ends), per beam path, step and
#: optical element; the measured runs added 0.97 to 1.42 x this, or less than 16 MiB on Windows (working set); on Linux
#: (resident set) one beam through 4 stages added 19 MiB against 6 MB estimated: the margin of the high end covers it.
MEMORY_PER_PATH = 130_602
MEMORY_PER_STEP = 14_689
MEMORY_PER_ELEMENT = 607_384
AFTER_TRACE_MEMORY_HIGH = 1.6
AFTER_TRACE_MEMORY_MARGIN = 32 << 20

#: How an estimate was made (its ``method``): from the per-beam costs of a reference scene, before the trace; from the
#: beam paths and steps the trace counted, after it.
PER_BEAM_REFERENCE = "PER_BEAM_REFERENCE"
COUNTED_PATHS_AND_STEPS = "COUNTED_PATHS_AND_STEPS"
#: The measured runs an estimate's costs come from (its ``reference``): what was measured, as a refusal states it.
REFERENCES = {
    "INDEPENDENT_BEAMS_2500": "2500 independent beams, two paths each, 8 x 12 pupil rays",
    "INDEPENDENT_BEAMS_625_BROWSER": "625 independent beams, two paths each, 8 x 12 pupil rays, in a browser",
    "MEASURED_RUNS": AFTER_TRACE_MEASURED,
}
#: Whose memory costs an estimate uses (its ``memory_platform``): a CPython process (Linux resident set, Windows working
#: set), or the WebAssembly memory of a browser runtime.
CPYTHON = "CPYTHON"
BROWSER_WASM = "BROWSER_WASM"


def estimate_basis(method: str, reference: str, ratio: tuple[float, float], memory_platform: str | None,
                   lower_bounds: tuple[str, ...] = ()) -> dict[str, Any]:
    """The fields that state how an estimate was made (module docstring)."""
    return {"method": method, "reference": reference, "measured_ratio_low": ratio[0], "measured_ratio_high": ratio[1],
            "memory_platform": memory_platform, "lower_bounds": list(lower_bounds)}


def estimate_statement(estimate: dict[str, Any], *, memory: bool = False) -> str:
    """How ``estimate`` was made and what lets the run go on, in the words of a refusal, from its fields alone
    (``memory``: of its memory, else of its run files)."""
    if estimate.get("method") == COUNTED_PATHS_AND_STEPS and memory:
        return "read after the trace: the process's memory when the trace ended; a higher limit lets the run complete"
    if estimate.get("method") == COUNTED_PATHS_AND_STEPS:
        return (f"counted after the trace: the Project and Engine request files and, in the result, its "
                f"{estimate.get('paths')} beam paths of {estimate.get('steps')} steps in all, each at least as large as the "
                f"shortest record of its kind; a higher limit lets the run complete")
    measured = REFERENCES.get(str(estimate.get("reference")), "a reference scene")
    low, high = estimate.get("measured_ratio_low"), estimate.get("measured_ratio_high")
    ratio = f"; measured runs wrote {low:g} to {high:g} times such an estimate of their run files" if low and high else ""
    return (f"estimated before the trace for {estimate.get('beams')} beams of {estimate.get('rays_per_beam')} rays from the "
            f"per-beam costs measured on {measured}{ratio}; the estimate after the trace counts what this run needs; a "
            f"smaller ray_data recording, a scene without pupil rays drawn, a higher limit or no pre-flight refusal lets "
            f"the run start")


def beam_count(request: ValidationRequest) -> int:
    """The beams of a lowered Project: one per emission line of each Gaussian-beam source (a continuous spectrum's
    quadrature lines included); an extended emitter's rays are not beams."""
    return sum(len(c.source.spectrum) or 1 for c in request.project.components if c.source is not None)


def estimate_run_resources(request: ValidationRequest, *, ray_data: str = RAY_DATA_FULL, pupil_rays_drawn: bool = True,
                           base_memory_bytes: int | None = None, files_in_memory: bool = False,
                           browser: bool | None = None) -> dict[str, Any]:
    """The estimate before the trace (module docstring): ``stage`` ``before_trace``, ``beams``, ``rays_per_beam``,
    ``memory_bytes`` (``None`` when ``base_memory_bytes``, the process's memory once the Project is lowered, is not
    known), ``artifact_bytes`` and how it was made (``estimate_basis``)."""
    cfg = request.configuration.numerical
    browser = sys.platform == "emscripten" if browser is None else browser
    beams = beam_count(request)
    rays = cfg.pupil_ring_count * cfg.pupil_points_per_ring + 1
    scale = rays / REFERENCE_RAYS
    files = beams * (OTHER_FILE_BYTES_PER_BEAM - (0 if pupil_rays_drawn else SCENE_PUPIL_RING_BYTES_PER_BEAM)
                     + round(RAY_DATA_BYTES_PER_BEAM.get(ray_data, RAY_DATA_BYTES_PER_BEAM[RAY_DATA_FULL]) * scale))
    memory = None
    if base_memory_bytes is not None and browser:  # (WebAssembly memory: the files held in memory are in the per-beam cost)
        memory = base_memory_bytes + round(beams * RUN_MEMORY_PER_BEAM_WASM * scale)
    elif base_memory_bytes is not None:
        memory = base_memory_bytes + round(beams * RUN_MEMORY_PER_BEAM * scale) + (files if files_in_memory else 0)
    platform = None if memory is None else BROWSER_WASM if browser else CPYTHON
    return {"stage": "before_trace", "beams": beams, "rays_per_beam": rays, "memory_bytes": memory, "artifact_bytes": files,
            **estimate_basis(PER_BEAM_REFERENCE, "INDEPENDENT_BEAMS_625_BROWSER" if browser else "INDEPENDENT_BEAMS_2500",
                             PRE_TRACE_MEASURED_RATIO, platform)}


def optical_elements(request: ValidationRequest) -> int:
    """The components of a lowered Project that are not light sources (optical elements and instruments)."""
    return sum(1 for c in request.project.components if c.source is None)


def after_trace_files(paths: int, steps: int, elements: int, *, rays_per_beam: int = REFERENCE_RAYS,
                      ray_data: str = RAY_DATA_FULL, pupil_rays_drawn: bool = True) -> float:
    """The run files estimated after the trace (module docstring) of ``paths`` beam paths of ``steps`` steps in all and
    ``elements`` optical elements, ``rays_per_beam`` rays a beam."""
    if ray_data == RAY_DATA_FULL:
        ray_bytes = (paths * RAY_FULL_BYTES_PER_PATH * rays_per_beam / REFERENCE_RAYS
                     + steps * (rays_per_beam - 1) * RAY_FULL_BYTES_PER_PUPIL_STEP)
    else:
        ray_bytes = paths * RAY_SUMMARY_BYTES_PER_PATH + steps * RAY_SUMMARY_BYTES_PER_STEP
    other = (OTHER_FILES_BYTES + steps * OTHER_FILES_BYTES_PER_STEP + elements * OTHER_FILES_BYTES_PER_ELEMENT
             - (0 if pupil_rays_drawn else steps * SCENE_PUPIL_RING_BYTES_PER_STEP))
    return ray_bytes + max(other, 0)


def after_trace_memory_growth(paths: int, steps: int, elements: int) -> float:
    """The memory a run is estimated to add after its trace (module docstring)."""
    return paths * MEMORY_PER_PATH + steps * MEMORY_PER_STEP + elements * MEMORY_PER_ELEMENT


def _json_length(value: Any) -> int:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    return len(json.dumps(value.value if isinstance(value, enum.Enum) else value, ensure_ascii=False).encode("utf-8"))


def _shortest_value(annotation: Any) -> Any:
    """A value of the type ``annotation`` whose JSON text is no longer than that of any value of the type: ``null``, ``""``,
    ``0``, ``0.0``, ``true``, an empty array or object, the shortest literal or enum value, a fixed-length array of shortest
    values, a record of shortest values (``_shortest_record``), the shortest of a union's. A float's JSON text has at least
    three characters (``0.0``; every finite float is written with a point or an exponent) and a string's at least its two
    quotes, so no value is written shorter."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        return _shortest_value(args[0])
    if origin in (typing.Union, type(int | None)):  # (Optional[X] and X | None)
        return min((_shortest_value(arg) for arg in args), key=_json_length)
    if origin is typing.Literal:
        return min(args, key=_json_length)
    if origin in (tuple, list) or annotation in (tuple, list):
        return tuple(_shortest_value(arg) for arg in args) if origin is tuple and args and args[-1] is not Ellipsis else ()
    if origin is dict or annotation is dict:
        return {}
    shortest = {type(None): None, bool: True, int: 0, float: 0.0, str: ""}
    if annotation in shortest:
        return shortest[annotation]
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return min(annotation, key=_json_length)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _shortest_record(annotation)
    raise TypeError(f"no shortest JSON value is known for {annotation!r}")


def _shortest_record(model: type[BaseModel]) -> BaseModel:
    """The record of ``model`` with every field at its shortest JSON value (``_shortest_value``); a field the record omits
    when absent (``OmitAbsentFields.absent_omitted``) is absent."""
    omitted = getattr(model, "absent_omitted", ())
    return model.model_construct(**{name: None if name in omitted else _shortest_value(info.annotation)
                                    for name, info in model.model_fields.items()})


def _array_element_bytes(record: dict[str, Any]) -> int:
    """The bytes one more element ``record`` adds to an array member of a run file's canonical JSON (``json_parts``): its
    separator, line break, indentation and text."""
    return len(json_bytes({"a": Elements([record, record])})) - len(json_bytes({"a": Elements([record])}))


def shortest_record_bytes() -> tuple[int, int]:
    """``(path, step)``: the fewest bytes a traced beam path's record (``OpticalPath``) adds to the result file's
    ``paths``, without its steps, and the fewest each of its steps (``RaySegment`` in its ``segments``) adds -- the shortest
    records (``_shortest_record``) in the file's own JSON form."""
    segment = _shortest_record(RaySegment)

    def path_bytes(count: int) -> int:
        record = _shortest_record(OpticalPath).model_copy(update={"segments": (segment,) * count})
        return _array_element_bytes(record.model_dump(mode="json", warnings=False))

    step = path_bytes(2) - path_bytes(1)
    return min(path_bytes(0), path_bytes(1) - step), step


def certain_run_file_bytes(paths: int, steps: int, fixed_file_bytes: int) -> int:
    """The run-file bytes a run writes for certain once its trace has ended with ``paths`` beam paths of ``steps`` steps in
    all: ``fixed_file_bytes`` (the Project file and the Engine request file, known exactly once the Project is lowered) and,
    in the result file, every path and step at no fewer bytes than its shortest record (``shortest_record_bytes``). The
    result file holds every path the trace kept with all of its segments (a local run's request sets ``include_paths``)."""
    path, step = shortest_record_bytes()
    return fixed_file_bytes + paths * path + steps * step


def estimate_after_trace(request: ValidationRequest, *, paths: int, steps: int, ray_data: str = RAY_DATA_FULL,
                         pupil_rays_drawn: bool = True, memory_bytes: int | None = None,
                         files_in_memory: bool = False, fixed_file_bytes: int | None = None,
                         certain_files: bool = True) -> dict[str, Any]:
    """The estimate after the trace (module docstring), from the ``paths`` it traced and their ``steps`` (segments) in all:
    ``stage`` ``after_trace``, ``beams``, ``rays_per_beam``, ``paths``, ``steps``, ``elements``, ``artifact_bytes`` with
    ``artifact_bytes_low`` (what the run writes for certain: ``certain_run_file_bytes``; ``fixed_file_bytes`` the bytes of
    its Project and Engine request files, ``None`` when the run writes no run files: then 0; with ``certain_files`` false
    it is not counted and reads ``None``: no limit would be checked against it) / ``artifact_bytes_high``,
    ``memory_bytes`` with ``memory_bytes_low`` (``memory_bytes``, the memory read when the trace ended) /
    ``memory_bytes_high`` (``None`` when that reading is not known) and how it was made (``estimate_basis``)."""
    cfg = request.configuration.numerical
    rays = cfg.pupil_ring_count * cfg.pupil_points_per_ring + 1
    elements = optical_elements(request)
    files = after_trace_files(paths, steps, elements, rays_per_beam=rays, ray_data=ray_data, pupil_rays_drawn=pupil_rays_drawn)
    certain = None
    if certain_files:
        certain = 0 if fixed_file_bytes is None else certain_run_file_bytes(paths, steps, fixed_file_bytes)
    estimate: dict[str, Any] = {
        "stage": "after_trace", "beams": beam_count(request), "rays_per_beam": rays, "paths": paths, "steps": steps,
        "elements": elements, "artifact_bytes": round(max(files, certain or 0)), "artifact_bytes_low": certain,
        "artifact_bytes_high": round(max(files * AFTER_TRACE_FILES_HIGH, certain or 0)), "memory_bytes": None,
        "memory_bytes_low": None, "memory_bytes_high": None,
    }
    if memory_bytes is not None:
        growth = after_trace_memory_growth(paths, steps, elements)
        held = (files, files * AFTER_TRACE_FILES_HIGH) if files_in_memory else (0, 0)
        estimate.update(memory_bytes=round(memory_bytes + growth + held[0]), memory_bytes_low=memory_bytes,
                        memory_bytes_high=round(memory_bytes + AFTER_TRACE_MEMORY_HIGH * growth + AFTER_TRACE_MEMORY_MARGIN + held[1]))
    counted = tuple(name for name in ("artifact_bytes_low", "memory_bytes_low") if estimate[name] is not None)
    estimate.update(estimate_basis(COUNTED_PATHS_AND_STEPS, "MEASURED_RUNS", AFTER_TRACE_MEASURED_RATIO,
                                   None if memory_bytes is None else CPYTHON, counted))
    return estimate


__all__ = ["AFTER_TRACE_FILES_HIGH", "AFTER_TRACE_MEASURED", "AFTER_TRACE_MEASURED_RATIO", "AFTER_TRACE_MEMORY_HIGH", "AFTER_TRACE_MEMORY_MARGIN",
           "BROWSER_WASM", "COUNTED_PATHS_AND_STEPS", "CPYTHON",
           "MEMORY_PER_ELEMENT", "MEMORY_PER_PATH", "MEMORY_PER_STEP", "OTHER_FILES_BYTES", "OTHER_FILES_BYTES_PER_ELEMENT",
           "OTHER_FILES_BYTES_PER_STEP", "OTHER_FILE_BYTES_PER_BEAM", "PER_BEAM_REFERENCE", "PRE_TRACE_MEASURED_RATIO", "REFERENCES",
           "RAY_DATA_BYTES_PER_BEAM", "RAY_FULL_BYTES_PER_PATH", "RAY_FULL_BYTES_PER_PUPIL_STEP", "RAY_SUMMARY_BYTES_PER_PATH",
           "RAY_SUMMARY_BYTES_PER_STEP", "REFERENCE", "REFERENCE_RAYS", "RUN_MEMORY_PER_BEAM", "RUN_MEMORY_PER_BEAM_WASM",
           "SCENE_PUPIL_RING_BYTES_PER_BEAM", "SCENE_PUPIL_RING_BYTES_PER_STEP", "after_trace_files", "after_trace_memory_growth",
           "beam_count", "certain_run_file_bytes", "estimate_after_trace", "estimate_basis",
           "estimate_run_resources", "estimate_statement", "optical_elements", "shortest_record_bytes"]
