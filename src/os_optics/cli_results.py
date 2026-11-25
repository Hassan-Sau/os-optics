"""``os-optics results ...`` and ``os-optics export ...``: the scientific results of a run directory from the shell
(``os_optics.results``). ``register(subparsers)`` adds both commands; every command prints JSON (``export`` prints
the exported file, or a short JSON summary with ``--out``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

USAGE_ERROR = 64
NOT_EVALUATED = 2
RUN_HELP = "a run directory written by 'os-optics project run --out-dir'"
RESULTS_DESCRIPTION = (
    "Read the scientific results of a run directory on this device; nothing is simulated again. Each command answers one "
    "question, and its own --help says what it prints: 'describe', 'targets', 'freshness', 'monitors' / 'monitor', "
    "'paths', 'power-budget', 'rays', 'wavefront', 'irradiance', 'sampled-light', 'sample-emission', 'detection', "
    "'handoff', 'exposures', 'grating-orders', 'nonlinear' and 'configurations' read what the run computed; "
    "'pupil-field', 'focal-field', 'imaging', 'field', 'ghosts', 'cavity' and 'detector' compute an analysis from the "
    "run's light; 'metrics' and 'columns' describe the target metrics and the export columns. A run answers separately "
    "whether the calculation completed (outcome.execution), whether the targets are met (outcome.requirements and each "
    "target's assessment: MET, UNMET, INDETERMINATE or NOT_APPLICABLE) and whether each number is usable (the quantity's "
    "state; a number that is a bound or an estimate carries its value_basis entry: kind LOWER_BOUND, UPPER_BOUND or "
    "ESTIMATE, and what it omits, assumes or exceeds); the run's status is a diagnostic summary. Each command that reads "
    "a run directory checks the run's files first, as 'project verify' does, and names a run that does not verify on "
    "standard error; --no-verify reads them as stored. 'freshness' reads the run's record as stored and checks the files "
    "only with --verify; 'configurations' reads the comparison as stored (check it with 'project verify'). Output is "
    "JSON. Python: "
    "os_optics.results.load_run(RUN). Positions are design-frame millimetres; u / v are observation-plane coordinates "
    "(see 'describe')."
)
#: How a command of 'results' reads its run (the closing sentence of each command's description).
READ_AS_STORED = ("The run's files are checked first, as 'project verify' does (--no-verify reads them as stored). "
                  "Output is JSON.")
#: The description of each command of 'results' that has no description of its own elsewhere: what it computes or reads
#: and what it prints (``_described``).
RESULT_COMMAND_DESCRIPTIONS = {
    "describe": "Describe a run: its identity, status, the result types it holds, the frame and sign conventions and its "
                "files. Positions are design-frame millimetres; u / v are observation-plane coordinates.",
    "targets": "Every target outcome: its assessment (MET, UNMET, INDETERMINATE or NOT_APPLICABLE, the answer that "
               "decides), value, unit, bounds, error against the target, diagnostic status and issue codes, with the "
               "value's value_basis entry where the number is a bound or an estimate (such a number decides only in the "
               "direction it proves), and validity where a check of the model the metric needs failed or stays open (a "
               "Gaussian beam past an accuracy limit: no value, and the parameter, observed value, limit, setting and where). "
               "Also the run's outcome, the counts per status and per assessment, the run identity, "
               "and in_model_scope with every out-of-model finding in out_of_model_issues: model_scope_issues, Engine "
               "issues and targets whose status is OUT_OF_MODEL_SCOPE.",
    "monitors": "Every observation plane: frame, active area, absorbing behaviour, arrivals, its power with the "
                "quantity_state (KNOWN, KNOWN_ZERO, BOUNDED, UNKNOWN, OUT_OF_MODEL_SCOPE) and the bounds "
                "(power_lower_bound_W, power_upper_bound_W, untraced_power_W), the power per wavelength, and the "
                "value_basis entry of a power that is a bound or an estimate.",
    "power-budget": "The power ledger of the run: the launched power, every sink and termination, the power at each "
                    "monitor and the residual.",
    "monitor": "One observation plane in full: every arriving beam's state (position, direction, Gaussian beam, "
               "polarization, phase, power), the crossings outside its active area, the coherent sums where beams "
               "overlap, the plane's power with its state, bounds and value_basis, and the targets on it.",
    "paths": "The beam-path tree: each path's branches, terminations, power and optical path length; --path prints one "
             "path.",
    "rays": "The chief and pupil rays at observation-plane crossings: u, v, xyz, directions and optical path length of each "
            "ray. --monitor, --path and --crossing-index select rows; --offset / --limit print one page {total, offset, "
            "limit, rows}. The pupil rays need a run written with --ray-data full (the default).",
    "wavefront": "The optical path difference of each pupil ray of one path at a plane crossing, against a reference "
                 "plane (default) or sphere through the chief ray's hit point, with its RMS and PV (geometric optics, no "
                 "diffraction). It needs the pupil rays (--ray-data full, the default) and at least --min-rays of them.",
    "sampled-light": "The collection fraction of each extended emitter line at the observation planes, with the same "
                     "fraction from half of its rays: their difference shows the sampling noise.",
    "detection": "The photocurrent, shot / dark / thermal noise and SNR of each observation plane that states a "
                 "detector, and with a lock-in its demodulated signal, noise and SNR; each quantity with its state.",
    "sample-emission": "How each fluorescence or Raman sample's emission follows from the excitation delivered to it: the "
                       "absorbed share, the quantum yield or Raman conversion, the line powers and photon rates, and its "
                       "collection at each plane.",
    "irradiance": "The irradiance grid at a plane: the arriving beams from the paraxial Gaussian model (incoherent and "
                  "coherent sums) and the sampled rays of extended emitters binned (incoherent), with untraced_fraction, "
                  "the share of the light that may reach the plane and is not in the map. --roi adds the uniformity "
                  "(min/mean, std/mean) and how far the region's power may move; --samples fixes the grid; "
                  "--max-untraced-fraction sets the share above which the coherent map and a uniformity are withheld.",
    "handoff": "The beams at an observation plane in its (u, v, n) frame (q, waist, curvature, M^2, Jones vector, power) "
               "with the TEM00 field-construction convention, for an external diffraction or full-wave tool; --element "
               "adds that element's clear apertures in the same frame.",
    "pupil-field": "The scalar low-NA focal field of a pupil mask, with the power at every stage and the state of every "
                   "quantity. The request states the illumination, or RUN with --monitor gives the plane's beams "
                   "(HANDOFF_BEAMS). A document whose focal quantities are unavailable says why (status PARTIAL, exit "
                   "0); exit 2 when no document can be made (status UNAVAILABLE).",
    "focal-field": "The vector focal field of an ideal aplanatic objective at any numerical aperture: |E|^2 and S_z maps, "
                   "peak, widths, encircled energy and the energy balance, with convergence evidence. The request states "
                   "the illumination, or RUN with --monitor gives the plane's beams (HANDOFF_BEAMS).",
    "imaging": "The point spread function (diffraction and geometric), the diffraction and geometric MTF, the encircled "
               "energy and the wavefront of every field at an image plane: the run's light traced again with a dense "
               "pupil grid. Each quantity states its model basis, validity and convergence; a quantity outside its model "
               "states why and the others are reported.",
    "field": "The coherent field of a sampled-field or speckle source at a plane: angular-spectrum propagation in free "
             "space, or the Collins integral of the source axis's first-order transfer through elements; irradiance maps, "
             "power, centroid and speckle contrast over realizations, with the model's conditions and convergence. "
             "--field-out writes the field of realization 0 as an .npz a pupil-field FIELD illumination reads.",
    "ghosts": "The ghost images at an image plane: every lens or window face's reflection followed up to a bounce order, "
              "each ghost's bounce surfaces, direction, focus position, relative power and irradiance, and a bound on "
              "the ghosts not listed (proven where every ghost surface met a valid Gaussian beam; "
              "completeness.not_followed names what it does not count).",
    "cavity": "The cavity analysis of the run's Project, bound to the run: stability, Gaussian eigenmode, resonances and "
              "the resonant build-up of light injected through a mirror, with the Project's targets of a cavity metric "
              "assessed on its quantities. The request is the one 'project cavity --request' takes.",
    "detector": "Photons, electrons, ADU, saturation, noise and SNR per pixel of a stated detector, from a plane's "
                "irradiance (RUN with --monitor) or a focal field (--field: the reply of 'results pupil-field' or "
                "'results irradiance'); what the detector does not state stays unavailable. A --field document that "
                "lacks a member the response reads (grid, grid.u_mm / v_mm, a map of their shape) or holds one of the "
                "wrong form is refused with IRRADIANCE_INVALID, naming the member.",
    "exposures": "The beam exposure of every element surface the beams meet (power, beam radii, AOI, peak irradiance / "
                 "fluence, P / d) and its comparison with the elements' power ratings, condition by condition. An "
                 "exposure or a ratio that is a bound or an estimate carries its value_basis entry; an estimate decides "
                 "no rating verdict.",
    "grating-orders": "Every diffraction order of every beam's pass through a grating: its direction, and the state of its "
                      "power, polarization and phase.",
    "nonlinear": "Every beam's pass into a crystal that generates its second harmonic: the harmonic power, conversion "
                 "efficiency, phase mismatch, walk-off, the model's checks and the acceptance bandwidths.",
    "metrics": "What every target metric means: unit, definition, sign convention, operands, frame, validity and the result "
               "field it reads. --name prints one; --include-stage-metrics adds the layout / control metrics of Engine "
               "requests. It reads no run. Output is JSON.",
    "columns": "The columns of an export kind ('os-optics export KIND'), each with its unit and meaning. It reads no run. "
               "Output is JSON.",
}


def _described(name: str) -> str:
    """The description of the 'results' command ``name``: what it prints, then how it reads the run."""

    text = RESULT_COMMAND_DESCRIPTIONS[name]
    return text if text.endswith("Output is JSON.") else f"{text} {READ_AS_STORED}"
RESULTS_EPILOG = (
    "exit codes: 0 = answered (pupil-field: also status PARTIAL, a document whose focal quantities are UNAVAILABLE with their "
    "reason), 2 = the requested wavefront / irradiance / field cannot be computed from this run (JSON with status "
    "UNAVAILABLE and a reason code), 64 = usage error, unreadable run directory or unknown monitor / path (JSON error), "
    "70 = an internal error of os-optics on a run that verifies (JSON error INTERNAL_ERROR with the exception type; traceback on stderr)"
)
ANALYSIS_EPILOG = (
    "exit codes: 0 = the analysis ran to its end (execution COMPLETED; each quantity states its own state, and its reason "
    "where it is not known or outside its model), 2 = partial (a resource budget stopped the analysis early: "
    "execution.budgets), 1 = the request was refused (JSON error with its code), 64 = usage error or unreadable run directory, "
    "70 = an internal error of os-optics on a run that verifies (JSON error INTERNAL_ERROR with the exception type; traceback on stderr)"
)
#: 'results cavity': the cavity analysis has no budget, so an answered request completed.
CAVITY_EPILOG = (
    "exit codes: 0 = the analysis was answered (each quantity states its own state: one outside its model or not known carries the "
    "reason in the JSON), 1 = the request was refused (JSON error with its code), 64 = usage error or unreadable run directory, "
    "65 = the run's Project cannot be lowered here (JSON with its problems, e.g. CATALOG_NOT_AVAILABLE: the catalog release it "
    "pins is not on this device; run 'os-optics catalog sync' or 'os-optics catalog import FILE'), "
    "70 = an internal error of os-optics on a run that verifies (JSON error INTERNAL_ERROR with the exception type; traceback on stderr)"
)
EXPORT_DESCRIPTION = (
    "Export one kind of result of a run directory as a file: monitors, arrivals (full beam state per arrival), rays (chief "
    "and pupil rays at observation planes), paths, segments, power-budget, superpositions, wavefront (needs --monitor and "
    "--path), irradiance (needs --monitor), handoff (needs --monitor; --element adds an element's apertures to the json form), "
    "metrics, exposures (one row per beam meeting an element surface), power-ratings (one row per surface and power "
    "rating) or grating-orders (one row per diffraction order of a pass through a grating). Formats: csv (header row; units in the column names, e.g. "
    "beam_radius_x_mm; empty field = undefined; lists joined by ';'), jsonl (one JSON object per row), json (rows with the "
    "column descriptions) and, for the irradiance grid, npz (NumPy arrays). Without --out the file is written to standard "
    "output. 'os-optics results columns KIND' lists every column with its unit and meaning."
)
EXPORT_EPILOG = (
    "exit codes: 0 = exported, 2 = the wavefront / irradiance cannot be computed from this run (JSON with the reason on "
    "stdout), 64 = usage error, unreadable run directory, unknown monitor / path or unwritable --out, 70 = an internal "
    "error of os-optics on a run that verifies (JSON error INTERNAL_ERROR with the exception type; traceback on stderr)"
)
EXPORT_KIND_CHOICES = ("monitors", "arrivals", "rays", "paths", "segments", "power-budget", "superpositions", "wavefront", "irradiance",
                       "sampled-light", "sampled-crossings",
                       "metrics", "handoff", "exposures", "power-ratings", "grating-orders", "nonlinear")
UNAVAILABLE_CODES = ("RAY_DATA_NOT_AVAILABLE", "PUPIL_RAYS_NOT_RECORDED", "CROSSING_NOT_FOUND", "RAY_OPL_UNDEFINED", "TOO_FEW_RAYS", "REFERENCE_UNDEFINED",
                     "RAY_MISSES_REFERENCE", "RAY_DATA_INCONSISTENT", "WAVELENGTH_REQUIRED", "NO_ARRIVALS", "GAUSSIAN_NOT_AVAILABLE",
                     "GRID_INVALID", "ROI_INVALID", "LIMIT_INVALID", "POWER_OUT_OF_MODEL_SCOPE",
                     "POWER_TAKEN_BEFORE_PLANE")
CENTER_HELP = ("sphere centre: x,y,z in design-frame mm, 'gaussian_waist' (the beam's Gaussian waist; a geometric sphere about it, "
               "so within the Rayleigh range, where the Gaussian phase front is nearly flat, prefer --reference plane) or "
               "'ray_focus' (where the pupil rays converge, least squares)")


def _json_out(doc: object) -> None:
    text = json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False)
    try:
        print(text)
    except UnicodeEncodeError:  # an output encoding without Chinese characters: the same JSON with \u escapes
        print(json.dumps(doc, indent=2, allow_nan=False))


#: What reading a run directory whose files have the wrong structure raises (a run that does not verify may hold anything).
UNEXPECTED_STRUCTURE = (AttributeError, KeyError, TypeError, ValueError, RecursionError)


def _verifies(results: Any, run_dir: str | None) -> bool:
    """Whether what a command read is known to be good: no run at all, a run loaded with its check that verified, or --
    when the run was read with ``--no-verify`` or could not be loaded -- a run directory that verifies (checked now)."""

    if run_dir is None:
        return True
    if results is not None and results.integrity_problems is not None:
        return results.integrity_problems == ()
    from .local.artifacts import verify_run_artifacts

    try:
        return verify_run_artifacts(run_dir) == ()
    except Exception:  # (a directory that cannot even be checked does not verify)
        return False


def _failed(exc: BaseException, results: Any = None, run_dir: str | None = None) -> int:
    """An exception no results command documents. On a run that does not verify the files may hold anything:
    ``RUN_NOT_READABLE`` (exit 64). With no run read, or a run that verifies, the input is good and the failure is
    os-optics's own: ``INTERNAL_ERROR`` with the exception's type, the traceback on standard error, exit 70."""

    if not _verifies(results, run_dir):
        return _error("RUN_NOT_READABLE", f"the run files have an unexpected structure ({type(exc).__name__}); check them "
                      f"with 'os-optics project verify {run_dir}'")
    from .cli_exit import INTERNAL_ERROR_EXIT, internal_error_document, internal_error_traceback

    print(internal_error_traceback(exc), end="", file=sys.stderr)
    _json_out(internal_error_document(exc))
    return INTERNAL_ERROR_EXIT


def _error(code: str, message: str) -> int:
    _json_out({"status": "ERROR", "error": {"code": code, "message": message}})
    return USAGE_ERROR


def _limit(text: str) -> float | str:
    """``--max-untraced-fraction``: a number of at least 0, or ``none`` (no limit)."""
    if text.strip().lower() == "none":
        return "none"
    value = float(text)
    if not value >= 0.0:
        raise argparse.ArgumentTypeError("a number of at least 0, or 'none'")
    return value


def _limit_option(args: argparse.Namespace) -> dict[str, Any]:
    """The ``max_untraced_fraction`` keyword of an irradiance call: absent when the option is not given."""
    stated = getattr(args, "max_untraced_fraction", None)
    return {} if stated is None else {"max_untraced_fraction": None if stated == "none" else stated}


def _center_uv(text: str | None) -> list[float] | None:
    """``u,v`` in mm, or ``None`` when absent or not two numbers."""
    if text is None:
        return None
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        return None
    return values if len(values) == 2 else None


def _roi(text: str | None) -> dict[str, Any] | None:
    """A JSON object, or ``None`` when absent or not one."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _center(text: str | None) -> Any:
    if text is None or text in ("gaussian_waist", "ray_focus"):
        return text
    parts = text.split(",")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return values if len(values) == len("xyz") else None


def _load(run_dir: str, verify: bool = True) -> Any:
    """The results of ``run_dir`` with its files checked first (as ``project verify`` does) and a run that does not verify
    named on stderr; ``verify`` false (``--no-verify``) reads them as stored."""

    from .results import load_run

    results = load_run(run_dir, verify=verify)
    if results.integrity_problems:
        print(f"os-optics: warning: {run_dir} does not verify ({len(results.integrity_problems)} problems; see "
              "'os-optics project verify'); the results are shown as stored", file=sys.stderr)
    if results.later_attempt is not None:
        attempt = results.later_attempt
        print(f"os-optics: note: a later run into {run_dir} stopped in phase {attempt['phase']} ({attempt['cause']}) and left "
              f"this run as it was; it ran {'the same Project' if attempt['same_project'] else 'another Project'} "
              "(run_stopped.json)", file=sys.stderr)
    return results


def _unavailable(doc: dict[str, Any]) -> int:
    _json_out(doc)
    return 0 if doc.get("status") == "OK" else NOT_EVALUATED


def _read_json(path: str) -> Any:
    """The JSON document of a file; ``OSError`` / ``ValueError`` when it cannot be read."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pupil_field(args: argparse.Namespace) -> int:
    from .results import PupilFieldError, ResultsError
    from .results.pupil_field import pupil_field as focal_field  # (the function: the package attribute may be the module)

    if (args.run_dir is None) != (args.monitor is None):
        return _error("USAGE", "a run's beams need both RUN and --monitor; a request with its own illumination needs neither")
    try:
        request = _read_json(args.request)
        files = {Path(name).name: Path(name).read_bytes() for name in args.mask_file}
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request or a mask file: {exc}")
    results = None
    try:
        if args.run_dir is None:
            doc = focal_field(request, files)
        else:
            results = _load(args.run_dir, args.verify)
            doc = results.pupil_field(args.monitor, request, files)
    except (PupilFieldError, ResultsError) as exc:
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    # (a PARTIAL document is an answer: its pupil stage is stated and its focal quantities say why they are not)
    _json_out(doc)
    return NOT_EVALUATED if doc.get("status") == "UNAVAILABLE" else 0


#: Exit code of an analysis whose request is refused (an invalid request, beams no field can be built from).
ANALYSIS_REFUSED = 1


def _analysis_exit(doc: Any) -> int:
    """Print an analysis document; 0 when it ran to its end, 2 when a budget stopped it early (``execution.complete``)."""

    _json_out(doc.model_dump(mode="json", by_alias=True))
    return 0 if doc.execution.complete else NOT_EVALUATED


def _focal_field(args: argparse.Namespace) -> int:
    from .results import FocalFieldError, ResultsError
    from .results.focal_field import focal_field  # (the function, read from its module)

    if (args.run_dir is None) != (args.monitor is None):
        return _error("USAGE", "a run's beams need both RUN and --monitor; a request with its own illumination needs neither")
    try:
        request = _read_json(args.request)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request: {exc}")
    results = None
    try:
        if args.run_dir is None:
            doc = focal_field(request)
        else:
            results = _load(args.run_dir, args.verify)
            doc = results.focal_field(args.monitor, request)
    except FocalFieldError as exc:
        _json_out({"status": "ERROR", "error": {"code": exc.code, "message": exc.message}})
        return ANALYSIS_REFUSED
    except ResultsError as exc:
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return _analysis_exit(doc)


def _ghosts(args: argparse.Namespace) -> int:
    from .results import GhostError, ResultsError

    try:
        request = None if args.request is None else _read_json(args.request)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request: {exc}")
    results = None
    try:
        results = _load(args.run_dir, args.verify)
        doc = results.ghosts(args.monitor, request)
    except GhostError as exc:
        _json_out({"status": "ERROR", "error": {"code": exc.code, "message": exc.message}})
        return ANALYSIS_REFUSED
    except ResultsError as exc:
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return _analysis_exit(doc)


def _cavity(args: argparse.Namespace) -> int:
    from .results import ResultsError
    from .results.model import RunProjectRefused

    try:
        request = _read_json(args.request)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request: {exc}")
    results = None
    try:
        results = _load(args.run_dir, args.verify)
        doc = results.cavity(request)
    except RunProjectRefused as exc:  # (the run's Project cannot be lowered here: its problems, as 'project cavity' gives them)
        _json_out(exc.refusal.as_dict())
        return LOCAL_INPUT_ERROR
    except ResultsError as exc:
        if exc.code == "CAVITY_ANALYSIS_INVALID":
            _json_out({"status": "ERROR", "error": {"code": exc.code, "message": exc.message}})
            return ANALYSIS_REFUSED
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return _analysis_exit(doc)


def _field(args: argparse.Namespace) -> int:
    from .results import FieldError, ResultsError
    from .results.field import field_npz

    try:
        request = None if args.request is None else _read_json(args.request)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request: {exc}")
    if args.field_out is not None:
        request = {**(request or {}), "include_field": True}
    results = None
    try:
        results = _load(args.run_dir, args.verify)
        doc = results.field(args.monitor, request)
        if args.field_out is not None and "field_re" in doc.arrays:
            Path(args.field_out).write_bytes(field_npz(doc))
    except FieldError as exc:
        _json_out({"status": "ERROR", "error": {"code": exc.code, "message": exc.message}})
        return ANALYSIS_REFUSED
    except ResultsError as exc:
        return _error(exc.code, exc.message)
    except OSError as exc:
        return _error("USAGE", f"cannot write the field file: {exc}")
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return _analysis_exit(doc)


def _imaging(args: argparse.Namespace) -> int:
    from .results import ImagingError, ResultsError

    try:
        request = None if args.request is None else _read_json(args.request)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the request: {exc}")
    results = None
    try:
        results = _load(args.run_dir, args.verify)
        doc = results.imaging(args.monitor, request)
    except ImagingError as exc:
        _json_out({"status": "ERROR", "error": {"code": exc.code, "message": exc.message}})
        return ANALYSIS_REFUSED
    except ResultsError as exc:
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return _analysis_exit(doc)


def _detector(args: argparse.Namespace) -> int:
    from .results import DetectorError, ResultsError, detector_response

    if (args.field is None) == (args.run_dir is None) or (args.run_dir is not None and args.monitor is None):
        return _error("USAGE", "the irradiance is RUN with --monitor, or --field FILE (the reply of 'results pupil-field' or 'results irradiance')")
    center_uv = _center_uv(args.center_uv_mm)
    if args.center_uv_mm is not None and center_uv is None:
        return _error("USAGE", "--center-uv-mm is u,v (two numbers, mm)")
    try:
        detector = _read_json(args.detector)
        field = None if args.field is None else _read_json(args.field)
    except (OSError, ValueError) as exc:
        return _error("USAGE", f"cannot read the detector or the field: {exc}")
    results = None
    try:
        if field is None:
            results = _load(args.run_dir, args.verify)
            field = results.irradiance_map(args.monitor, wavelength_nm=args.wavelength_nm, half_extent_mm=args.half_extent_mm, samples=args.samples,
                                           source_id=args.source_id, center_uv_mm=center_uv, **_limit_option(args))
        return _unavailable(detector_response(field, detector, args.composition))
    except (DetectorError, ResultsError) as exc:
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        if args.field is not None:  # (the caller's own document: not input known to be good)
            return _error("IRRADIANCE_INVALID", f"{args.field} is not an irradiance document 'results irradiance' or 'results "
                          f"pupil-field' gives ({type(exc).__name__})")
        return _failed(exc, results, args.run_dir)


def _results(args: argparse.Namespace) -> int:
    from .results import MIN_WAVEFRONT_RAYS, ResultsError, columns
    from .validation.descriptors import metric_descriptors

    command = args.results_command
    if command == "metrics":
        descriptors = [d.model_dump(mode="json") for d in metric_descriptors(include_stage_metrics=args.include_stage_metrics)]
        if args.name:
            descriptors = [d for d in descriptors if d["name"] == args.name]
            if not descriptors:
                return _error("METRIC_NOT_FOUND", f"no metric named {args.name!r}")
        _json_out(descriptors)
        return 0
    if command == "columns":
        _json_out({"kind": args.kind, "columns": columns(args.kind)})
        return 0
    if command == "freshness":
        return _freshness(args)
    if command == "configurations":
        return _configurations(args)
    if command == "rays" and ((args.offset is not None and args.offset < 0) or (args.limit is not None and args.limit < 0)):
        return _error("USAGE", "--offset and --limit must be non-negative")
    if command == "wavefront" and args.center is not None and _center(args.center) is None:
        return _error("USAGE", "--center is x,y,z (three numbers, mm), gaussian_waist or ray_focus")
    results = None
    try:
        results = _load(args.run_dir, args.verify)
        if command == "describe":
            _json_out(results.describe())
        elif command == "monitors":
            _json_out(results.monitors())
        elif command == "monitor":
            _json_out(results.monitor(args.observation_id))
        elif command == "paths":
            _json_out(results.paths(args.path))
        elif command == "power-budget":
            _json_out(results.power_budget())
        elif command == "targets":
            _json_out(results.targets())
        elif command == "rays":
            rows = results.rays(args.monitor, args.path, args.crossing_index)
            if args.offset is None and args.limit is None:
                _json_out(rows)
            else:
                offset = args.offset or 0
                page = rows[offset:] if args.limit is None else rows[offset:offset + args.limit]
                _json_out({"total": len(rows), "offset": offset, "limit": args.limit, "rows": page})
        elif command == "handoff":
            _json_out(results.handoff(args.monitor, args.path, args.crossing_index, args.element))
        elif command == "exposures":
            _json_out(results.exposures(args.element))
        elif command == "grating-orders":
            _json_out(results.grating_orders(args.element))
        elif command == "nonlinear":
            _json_out(results.nonlinear(args.element))
        elif command == "wavefront":
            return _unavailable(results.wavefront(args.monitor, args.path, crossing_index=args.crossing_index or 0,
                                                  reference=args.reference or "plane", center=_center(args.center),
                                                  min_rays=MIN_WAVEFRONT_RAYS if args.min_rays is None else args.min_rays))
        elif command == "sampled-light":
            _json_out(results.sampled_light(args.monitor))
        elif command == "detection":
            _json_out(results.detection(args.monitor))
        elif command == "sample-emission":
            _json_out(results.sample_emission())
        else:  # irradiance
            center_uv = _center_uv(args.center_uv_mm)
            roi = _roi(args.roi)
            if (args.center_uv_mm is not None and center_uv is None) or (args.roi is not None and roi is None):
                return _error("USAGE", "--center-uv-mm is u,v (two numbers, mm); --roi is a JSON object")
            doc = results.irradiance_map(args.monitor, wavelength_nm=args.wavelength_nm, half_extent_mm=args.half_extent_mm,
                                         samples=args.samples, source_id=args.source_id, center_uv_mm=center_uv, roi=roi, **_limit_option(args))
            return _unavailable(doc)
    except ResultsError as exc:
        if exc.code in UNAVAILABLE_CODES:
            return _unavailable({"status": "UNAVAILABLE", "reason": exc.code, "message": exc.message})
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    return 0


FRESHNESS_EXIT = {"CURRENT": 0, "HISTORICAL": 1, "UNDETERMINED": 2}
LOCAL_INPUT_ERROR = 65
CONFIGURATIONS_DESCRIPTION = (
    "The configurations of a run of a Project with configurations side by side (nothing is simulated again): per "
    "configuration its execution (outcome, termination, exhausted budgets) and requirements, each target's value, status "
    "and assessment per configuration, and per configuration the power at every observation plane (with its quantity "
    "state) and every arriving beam's chief-ray position, Gaussian radii and spot RMS radius. --paraxial adds the effective "
    "and back focal lengths of the listed components in each configuration. Output: the CONFIGURATIONS analysis document "
    "(JSON), or with --table one row per quantity and one column per configuration. Each configuration's own results are "
    "read from its run directory, RUN/configurations/ID, with every other 'results' command. There is no overall verdict: "
    "each configuration's targets are its own. The run is read as stored: 'project verify RUN' checks its files."
)
CONFIGURATIONS_EPILOG = (
    "exit codes: 0 = every configuration's run completed, 2 = a configuration's run did not complete (its termination is "
    "listed; the other configurations' values stand), 1 = the paraxial request was refused or the run cannot be read as "
    "a run of configurations (JSON error), 64 = usage error"
)
CONFIGURATIONS_EXIT = {"COMPLETED": 0, "PARTIAL": NOT_EVALUATED}
REFUSED = 1


def _configurations(args: argparse.Namespace) -> int:
    from .analysis.configurations import comparison_table
    from .local import LocalRunError
    from .local.configurations import comparison_from_files, run_reader

    paraxial = None
    if args.paraxial is not None:
        if args.wavelength_nm is None:
            return _error("USAGE", "--paraxial needs --wavelength-nm")
        paraxial = {"components": [c for c in args.paraxial.split(",") if c], "wavelength_nm": args.wavelength_nm}
    read = run_reader(args.run_dir)
    if read is None:
        _json_out({"status": "ERROR", "error": {"code": "CONFIGURATIONS_RUN_INVALID",
                                                "message": f"{args.run_dir} is not the run directory of a Project with configurations"}})
        return REFUSED
    try:
        comparison = comparison_from_files(read, paraxial=paraxial)
    except LocalRunError as exc:
        _json_out(exc.as_dict())
        return REFUSED
    except OSError as exc:
        return _error("RUN_NOT_READABLE", f"a file of the run cannot be read ({exc}); check it with 'os-optics project verify "
                      f"{args.run_dir}'")
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, None, args.run_dir)
    _json_out(comparison_table(comparison) if args.table else comparison.model_dump(mode="json", by_alias=True))
    return CONFIGURATIONS_EXIT[comparison.execution.outcome.value]


def _freshness(args: argparse.Namespace) -> int:
    from .local import LocalRunError
    from .local.freshness import RunNotReadable, run_freshness

    try:
        project = Path(args.project).read_bytes()
    except OSError as exc:
        print(f"error: cannot read the project: {exc}", file=sys.stderr)
        return USAGE_ERROR
    try:
        report = run_freshness(project, args.run_dir, verify=args.verify)
    except LocalRunError as exc:
        _json_out(exc.as_dict())
        return LOCAL_INPUT_ERROR
    except RunNotReadable as exc:
        return _error(exc.as_dict()["code"], exc.message)
    _json_out(report)
    return FRESHNESS_EXIT[report["freshness"]]


def _export(args: argparse.Namespace) -> int:
    from .contracts.results import ValidationResult, ValidationSummary
    from .errors import Status
    from .results import ResultsError, RunResults

    if args.format == "npz" and not args.out:
        return _error("USAGE", "the npz format is binary: give --out FILE")
    if args.center is not None and _center(args.center) is None:
        return _error("USAGE", "--center is x,y,z (three numbers, mm), gaussian_waist or ray_focus")
    if (args.center_uv_mm is not None and _center_uv(args.center_uv_mm) is None) or (args.roi is not None and _roi(args.roi) is None):
        return _error("USAGE", "--center-uv-mm is u,v (two numbers, mm); --roi is a JSON object")
    filters = {"monitor": args.monitor, "path": args.path, "crossing_index": args.crossing_index, "reference": args.reference,
               "center": _center(args.center), "min_rays": args.min_rays, "wavelength_nm": args.wavelength_nm,
               "half_extent_mm": args.half_extent_mm, "samples": args.samples, "element": args.element, "source_id": args.source_id,
               "center_uv_mm": _center_uv(args.center_uv_mm), "roi": _roi(args.roi)}
    results = None
    try:
        if args.kind == "metrics" and args.run_dir is None:  # the metric descriptors need no run
            results = RunResults(result=ValidationResult(status=Status.NOT_APPLICABLE, summary=ValidationSummary()))
        elif args.run_dir is None:
            return _error("USAGE", f"export {args.kind} needs a run directory")
        else:
            results = _load(args.run_dir, args.verify)
        data = results.export(args.kind, args.format, **{k: v for k, v in filters.items() if v is not None}, **_limit_option(args))
    except ResultsError as exc:
        if exc.code in UNAVAILABLE_CODES:
            return _unavailable({"status": "UNAVAILABLE", "reason": exc.code, "message": exc.message})
        return _error(exc.code, exc.message)
    except UNEXPECTED_STRUCTURE as exc:
        return _failed(exc, results, args.run_dir)
    if args.out:
        try:
            Path(args.out).write_bytes(data)
        except OSError as exc:
            print(f"error: cannot write the export: {exc}", file=sys.stderr)
            return USAGE_ERROR
        _json_out({"kind": args.kind, "format": args.format, "out": args.out, "bytes": len(data)})
        return 0
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


VERIFY_HELP = ("also check the run's files (hashes and cross-references, as 'project verify' does) and report it in "
               "integrity; without it integrity.checked is false. The check never changes the freshness")
NO_VERIFY_HELP = ("read the run's files as stored, without checking their hashes and cross-references first (the check reads "
                  "every file, ray data included); integrity.checked is then false")


def _add_verify(commands: argparse._SubParsersAction) -> None:
    """``--no-verify`` on every command of ``commands`` that reads a run directory (they check its files by default);
    ``freshness`` checks them only with ``--verify``."""

    for name, parser in commands.choices.items():
        # ('configurations' reads the comparison document of a run of configurations, which 'project verify' checks)
        if name == "configurations" or not any(action.dest == "run_dir" for action in parser._actions):
            continue
        if name == "freshness":
            parser.add_argument("--verify", action="store_true", help=VERIFY_HELP)
        else:
            parser.add_argument("--no-verify", dest="verify", action="store_false", help=NO_VERIFY_HELP)


def _add_filters(parser: argparse.ArgumentParser, *, wavefront: bool, irradiance: bool) -> None:
    if wavefront:
        parser.add_argument("--crossing-index", type=int, default=None,
                            help="which crossing of the plane by the path: 0 = first (default), 1 = second (folded / double pass)")
        parser.add_argument("--reference", choices=["plane", "sphere"], default=None,
                            help="wavefront reference surface through the chief ray's hit point: 'plane' (perpendicular to the chief "
                                 "ray; default) or 'sphere' (about --center)")
        parser.add_argument("--center", help=CENTER_HELP)
        parser.add_argument("--min-rays", type=int, default=None, help="refuse a wavefront with fewer arrived pupil rays: an integer, at least 1 (default 8)")
    if irradiance:
        parser.add_argument("--wavelength-nm", type=float, default=None, help="irradiance: the wavelength (when several arrive and it is "
                                                                              "not given: their sum on one grid)")
        parser.add_argument("--half-extent-mm", type=float, default=None,
                            help="irradiance: half width of the square grid (default: every beam out to 2.5 beam radii; for sampled "
                                 "rays the active area, or the half width holding 95%% of their power)")
        parser.add_argument("--samples", type=int, default=None,
                            help="irradiance: grid points per axis of a beam map, bins per axis of a sampled-ray map, 2 to 1001 "
                                 "(when not given: 101 for a beam map, or the count up to 1001 that resolves the beams' fringes; a "
                                 "sampled-ray map takes about 16 crossings per bin). State it to fix the count")
        parser.add_argument("--max-untraced-fraction", type=_limit, default=None,
                            help="irradiance: the share of the map's light that may be missing from it (untraced_fraction) above which the "
                                 "coherent map, a uniformity and the detector quantities that light can move are withheld (at or "
                                 "below it they are stated with untraced_fraction and how far that light may move them): a "
                                 "number of at least 0 (default 0.1), or 'none' for no limit")
        parser.add_argument("--source-id", default=None, help="irradiance: the light of this source alone")
        parser.add_argument("--center-uv-mm", default=None, help="irradiance: the grid's centre on the plane, u,v in mm")
        parser.add_argument("--roi", default=None,
                            help='irradiance: JSON region for the uniformity (min/mean, std/mean): {"shape":"rectangle","center_uv_mm":[u,v],'
                                 '"half_width_mm":[a,b]} or {"shape":"circle","center_uv_mm":[u,v],"radius_mm":r}')


def register(subparsers: argparse._SubParsersAction) -> None:
    results = subparsers.add_parser("results", help="read a run directory's scientific results: monitors, beams, rays, wavefront, "
                                                    "power budget, exposures and power ratings, metric meanings",
                                    description=RESULTS_DESCRIPTION,
                                    epilog=RESULTS_EPILOG)
    sub = results.add_subparsers(dest="results_command", required=True)
    for name, text in (("describe", "identity, status, available results, conventions and files of a run"),
                       ("targets", "every target outcome (value, unit, bounds, status, issue codes, value_basis, validity), counts per status, "
                                   "in_model_scope with every out-of-model finding, and the run identity"),
                       ("monitors", "every observation plane: frame, active area, absorbing behaviour, arrivals, power"),
                       ("power-budget", "the power ledger: input, every sink, power at each monitor, residual")):
        p = sub.add_parser(name, help=text, description=_described(name), epilog=RESULTS_EPILOG)
        p.add_argument("run_dir", help=RUN_HELP)
        p.set_defaults(handler=_results)
    monitor = sub.add_parser("monitor", help="one observation plane in full: every arrival's beam state, outside crossings, "
                                             "coherent sums, targets", description=_described("monitor"), epilog=RESULTS_EPILOG)
    monitor.add_argument("run_dir", help=RUN_HELP)
    monitor.add_argument("observation_id", help="the observation plane id")
    monitor.set_defaults(handler=_results)
    paths = sub.add_parser("paths", help="the beam-path tree (branches, terminations, power, optical path length)",
                           description=_described("paths"), epilog=RESULTS_EPILOG)
    paths.add_argument("run_dir", help=RUN_HELP)
    paths.add_argument("--path", help="only this path id")
    paths.set_defaults(handler=_results)
    rays = sub.add_parser("rays", help="chief and pupil rays at observation-plane crossings (u, v, xyz, directions, OPL)",
                          description=_described("rays"), epilog=RESULTS_EPILOG)
    rays.add_argument("run_dir", help=RUN_HELP)
    rays.add_argument("--monitor", help="only this observation plane")
    rays.add_argument("--path", help="only this path id")
    rays.add_argument("--crossing-index", type=int, default=None, help="only this crossing (0 = first)")
    rays.add_argument("--offset", type=int, default=None, help="skip this many rows; with --offset or --limit the output is one page "
                                                                "{total, offset, limit, rows}")
    rays.add_argument("--limit", type=int, default=None, help="print at most this many rows (see --offset)")
    rays.set_defaults(handler=_results)
    fresh = sub.add_parser("freshness", help="is the run CURRENT for a Project, or HISTORICAL (with the reasons)?",
                           description="Compare a run with a Project without running anything: CURRENT when the Project's "
                                       "project_hash and the engine_input_hash it lowers to equal the run's (its Local "
                                       "Validation record), else HISTORICAL with the reasons PHYSICS_CHANGED (the design apart from "
                                       "its targets changed: its optics, sources, observation planes or compute_budget; or the "
                                       "change cannot be attributed to the targets alone), "
                                       "TARGETS_CHANGED, OTHER_PROJECT_CHANGED (another Project field, such as the design "
                                       "intent), CATALOG_CHANGED (the same Project gives another Engine input: another catalog "
                                       "release) and ENGINE_CHANGED (added to the reasons of a HISTORICAL run that another Engine "
                                       "build made); UNDETERMINED (PROJECT_NOT_CHECKED): the same Project, which no longer passes the "
                                       "check. RUN is a run directory or a local_validation.json file (a bare record cannot tell "
                                       "a target change from a physics change: PHYSICS_CHANGED). A run of the same Project and "
                                       "Engine input made by another Engine build stays CURRENT; engine_build_matches says "
                                       "whether the installed build made it. Beside this aggregate the reply states the two "
                                       "identities separately: physics (CURRENT when the Engine input without its targets is the "
                                       "run's, whatever the targets did; HISTORICAL with PHYSICS_CHANGED or CATALOG_CHANGED) and "
                                       "requirements (CURRENT when the physics and the targets are the run's). A re-run is needed "
                                       "only when physics is not CURRENT; after an edit of the targets alone the stored quantities "
                                       "are the Project's. For a run of configurations, coverage lists the configurations declared, "
                                       "run, not run and no longer declared, and physics is taken over those that were run. Prints "
                                       "the run's and the Project's identities. "
                                       "Freshness is read from the run's record as stored; --verify also checks the run's files "
                                       "and reports it in integrity (checked false without it), which never changes the "
                                       "freshness.",
                           epilog="exit codes: 0 = CURRENT, 1 = HISTORICAL, 2 = UNDETERMINED, 64 = unreadable project or run, "
                                  "65 = the document is not a valid Project")
    fresh.add_argument("run_dir", help=RUN_HELP + " or its local_validation.json")
    fresh.add_argument("--project", required=True, help="the Project JSON file to compare the run with")
    fresh.set_defaults(handler=_results)
    configurations = sub.add_parser("configurations", help="the configurations of a run of a Project with configurations side by "
                                                           "side: execution, targets, plane powers, beam metrics",
                                    description=CONFIGURATIONS_DESCRIPTION, epilog=CONFIGURATIONS_EPILOG)
    configurations.add_argument("run_dir", help="a run directory of a Project with configurations ('project run --out-dir')")
    configurations.add_argument("--table", action="store_true", help="one row per quantity, one column per configuration")
    configurations.add_argument("--paraxial", metavar="IDS", help="comma-separated component ids, in the order light meets "
                                                                  "them: add their effective and back focal lengths per configuration")
    configurations.add_argument("--wavelength-nm", type=float, help="the vacuum wavelength of --paraxial")
    configurations.set_defaults(handler=_results)
    wave = sub.add_parser("wavefront", help="optical path difference per pupil ray at a plane crossing, with RMS / PV (geometric "
                                            "optics, no diffraction)", description=_described("wavefront"), epilog=RESULTS_EPILOG)
    wave.add_argument("run_dir", help=RUN_HELP)
    wave.add_argument("--monitor", required=True, help="the observation plane id")
    wave.add_argument("--path", required=True, help="the beam path id ('results paths' lists them)")
    _add_filters(wave, wavefront=True, irradiance=False)
    wave.set_defaults(handler=_results)
    sampled = sub.add_parser("sampled-light", help="collection fraction of each extended emitter line at the observation planes, with "
                                                   "the same fraction from half of the rays", description=_described("sampled-light"),
                             epilog=RESULTS_EPILOG)
    sampled.add_argument("run_dir", help=RUN_HELP)
    sampled.add_argument("--monitor", help="only this observation plane")
    sampled.set_defaults(handler=_results)
    detection = sub.add_parser("detection", help="photocurrent, shot / dark / thermal noise and SNR of each observation plane that states "
                                                 "a detector, and with a lock-in its demodulated signal, noise and SNR; each quantity "
                                                 "with its status", description=_described("detection"), epilog=RESULTS_EPILOG)
    detection.add_argument("run_dir", help=RUN_HELP)
    detection.add_argument("--monitor", help="only this observation plane")
    detection.set_defaults(handler=_results)
    emission = sub.add_parser("sample-emission", help="how each fluorescence or Raman sample's emission follows from the excitation "
                                                      "delivered to it: absorbed share, quantum yield or Raman conversion, line powers "
                                                      "and photon rates, and its collection at each plane",
                              description=_described("sample-emission"), epilog=RESULTS_EPILOG)
    emission.add_argument("run_dir", help=RUN_HELP)
    emission.set_defaults(handler=_results)
    irr = sub.add_parser("irradiance", help="irradiance grid at a plane: beams from the paraxial Gaussian model (incoherent and coherent "
                                            "sums), sampled rays of extended emitters binned (incoherent); --roi adds the uniformity",
                         description=_described("irradiance"), epilog=RESULTS_EPILOG)
    irr.add_argument("run_dir", help=RUN_HELP)
    irr.add_argument("--monitor", required=True, help="the observation plane id")
    _add_filters(irr, wavefront=False, irradiance=True)
    irr.set_defaults(handler=_results)
    handoff = sub.add_parser("handoff", help="the beams at an observation plane in its (u, v, n) frame (q, waist, curvature, M^2, Jones, "
                                             "power) with the TEM00 field-construction convention; --element adds an element's apertures",
                             description=_described("handoff"), epilog=RESULTS_EPILOG)
    handoff.add_argument("run_dir", help=RUN_HELP)
    handoff.add_argument("--monitor", required=True, help="the observation plane id")
    handoff.add_argument("--path", help="only this path id")
    handoff.add_argument("--crossing-index", type=int, default=None, help="only this crossing (0 = first)")
    handoff.add_argument("--element", help="a component instance id: its clear apertures in the plane's (u, v, n) frame")
    handoff.set_defaults(handler=_results)
    pupil = sub.add_parser("pupil-field", help="the scalar low-NA focal field of a pupil mask, with the power at every stage and the status of every quantity; "
                                               "RUN with --monitor reads the plane's beams for a HANDOFF_BEAMS illumination",
                           description=_described("pupil-field"), epilog=RESULTS_EPILOG)
    pupil.add_argument("run_dir", nargs="?", help=RUN_HELP + " (only for a HANDOFF_BEAMS illumination)")
    pupil.add_argument("--request", required=True, help="the request JSON file ('os-optics schema pupil-field-request' prints its schema)")
    pupil.add_argument("--monitor", help="the observation plane whose beams light the pupil")
    pupil.add_argument("--mask-file", action="append", default=[], metavar="FILE", help="a FILE mask's .npy / .npz file or a FIELD illumination's .npz file, named by its file name in the request (repeat for each)")
    pupil.set_defaults(handler=_pupil_field)
    focal = sub.add_parser("focal-field", help="the vector focal field of an ideal aplanatic objective at any numerical aperture: |E|^2 and S_z maps, peak, "
                                                "widths, encircled energy and the energy balance, with convergence evidence; RUN with --monitor reads the "
                                                "plane's beams for a HANDOFF_BEAMS illumination",
                           description=_described("focal-field"), epilog=ANALYSIS_EPILOG)
    focal.add_argument("run_dir", nargs="?", help=RUN_HELP + " (only for a HANDOFF_BEAMS illumination)")
    focal.add_argument("--request", required=True, help="the request JSON file ('os-optics schema focal-field-request' prints its schema)")
    focal.add_argument("--monitor", help="the observation plane whose beams light the objective's pupil")
    focal.set_defaults(handler=_focal_field)
    image = sub.add_parser("imaging", help="point spread function, diffraction and geometric MTF, encircled energy and wavefront of every field at an "
                                           "image plane: the run's light traced again with a dense pupil grid, with model basis, validity and convergence",
                           description=_described("imaging"), epilog=ANALYSIS_EPILOG)
    image.add_argument("run_dir", help=RUN_HELP)
    image.add_argument("--monitor", required=True, help="the image plane (an observation plane id)")
    image.add_argument("--request", help="the request JSON file ('os-optics schema imaging-request' prints its schema); omitted: every field with the defaults")
    image.set_defaults(handler=_imaging)
    fld = sub.add_parser("field", help="the coherent field of a sampled-field or speckle source at a plane: angular-spectrum propagation in free space or "
                                       "the Collins integral of the source axis's first-order transfer through elements, irradiance maps, power, centroid "
                                       "and speckle contrast over realizations, with model conditions and convergence",
                         description=_described("field"), epilog=ANALYSIS_EPILOG)
    fld.add_argument("run_dir", help=RUN_HELP)
    fld.add_argument("--monitor", required=True, help="the observation plane (an observation plane id)")
    fld.add_argument("--request", help="the request JSON file ('os-optics schema field-request' prints its schema); omitted: the defaults")
    fld.add_argument("--field-out", metavar="FILE", help="also write the field of realization 0 over the region as an .npz (field_re, field_im, "
                                                         "u_mm, v_mm, pitch_mm; model_scope_json where a model-scope finding reaches "
                                                         "the field): a pupil-field FIELD illumination file")
    fld.set_defaults(handler=_field)
    ghost = sub.add_parser("ghosts", help="ghost images at an image plane: every lens or window face's reflection followed up to a bounce order, each ghost's "
                                          "bounce surfaces, direction, focus position, relative power and irradiance, with a bound on the ghosts not listed (proven where every ghost surface met a valid "
                                          "Gaussian beam)",
                           description=_described("ghosts"), epilog=ANALYSIS_EPILOG)
    ghost.add_argument("run_dir", help=RUN_HELP)
    ghost.add_argument("--monitor", required=True, help="the image plane (an observation plane id)")
    ghost.add_argument("--request", help="the request JSON file ('os-optics schema ghost-request' prints its schema); omitted: every field, two bounces")
    ghost.set_defaults(handler=_ghosts)
    cavity = sub.add_parser("cavity", help="the cavity analysis of the run's Project (stability, eigenmode, resonances, the resonant build-up of "
                                           "injected light, targets on its quantities with the Project's cavity targets), bound to the run",
                            description=_described("cavity"), epilog=CAVITY_EPILOG)
    cavity.add_argument("run_dir", help=RUN_HELP)
    cavity.add_argument("--request", required=True, help="the request JSON file, as 'os-optics project cavity --request' takes it")
    cavity.set_defaults(handler=_cavity)
    detector = sub.add_parser("detector", help="photons, electrons, ADU, saturation, noise and SNR per pixel of a stated detector, from a plane's irradiance or a focal "
                                               "field; what the detector does not state stays unavailable",
                              description=_described("detector"), epilog=RESULTS_EPILOG)
    detector.add_argument("run_dir", nargs="?", help=RUN_HELP + " (with --monitor; or --field)")
    detector.add_argument("--detector", required=True, help="the detector JSON file ('os-optics schema detector-model' prints its schema)")
    detector.add_argument("--monitor", help="the observation plane whose irradiance to read")
    detector.add_argument("--field", help="a JSON file holding an irradiance document instead of a run: the reply of 'results pupil-field' or 'results irradiance'")
    detector.add_argument("--composition", choices=["COHERENT", "INCOHERENT"], help="the map to read when the irradiance document holds two different ones")
    _add_filters(detector, wavefront=False, irradiance=True)
    detector.set_defaults(handler=_detector)
    exposures = sub.add_parser("exposures", help="the beam exposure of every element surface (power, beam radii, AOI, peak irradiance "
                                                 "/ fluence, P / d) and its comparison with the elements' power ratings",
                               description=_described("exposures"), epilog=RESULTS_EPILOG)
    exposures.add_argument("run_dir", help=RUN_HELP)
    exposures.add_argument("--element", help="only this element (component instance id)")
    exposures.set_defaults(handler=_results)
    grating_orders = sub.add_parser("grating-orders", help="every diffraction order of every beam's pass through a grating: its direction, and the "
                                                           "states of its power, polarization and phase", description=_described("grating-orders"), epilog=RESULTS_EPILOG)
    grating_orders.add_argument("run_dir", help=RUN_HELP)
    grating_orders.add_argument("--element", help="only this element (component instance id)")
    grating_orders.set_defaults(handler=_results)
    nonlinear = sub.add_parser("nonlinear", help="every beam's pass into a crystal that generates its second harmonic: harmonic power, "
                                                 "conversion efficiency, phase mismatch, walk-off, the model's checks and the "
                                                 "acceptance bandwidths", description=_described("nonlinear"), epilog=RESULTS_EPILOG)
    nonlinear.add_argument("run_dir", help=RUN_HELP)
    nonlinear.add_argument("--element", help="only this element (component instance id)")
    nonlinear.set_defaults(handler=_results)
    metrics = sub.add_parser("metrics", help="what every target metric means: unit, definition, sign convention, operands, frame, "
                                             "validity", description=_described("metrics"), epilog=RESULTS_EPILOG)
    metrics.add_argument("--name", help="only this metric")
    metrics.add_argument("--json", action="store_true", help="JSON output (the default)")
    metrics.add_argument("--include-stage-metrics", action="store_true", help="also the layout / control metrics of Engine requests")
    metrics.set_defaults(handler=_results)
    cols = sub.add_parser("columns", help="the columns of an export kind with their units and meaning", description=_described("columns"),
                          epilog=RESULTS_EPILOG)
    cols.add_argument("kind", choices=list(EXPORT_KIND_CHOICES))
    cols.set_defaults(handler=_results)
    _add_verify(sub)

    export = subparsers.add_parser("export", help="export a run's results as CSV / JSON Lines / JSON (NPZ for the irradiance grid)",
                                   description=EXPORT_DESCRIPTION, epilog=EXPORT_EPILOG)
    export.add_argument("kind", choices=list(EXPORT_KIND_CHOICES), help="what to export")
    export.add_argument("run_dir", nargs="?", help=RUN_HELP + " (not needed for 'metrics')")
    export.add_argument("--format", choices=["csv", "jsonl", "json", "npz"], default="csv", help="file format (default csv)")
    export.add_argument("--out", help="write the file here and print a JSON summary (default: the file on standard output)")
    export.add_argument("--monitor", help="only this observation plane (arrivals, rays, superpositions; required for wavefront, "
                                          "irradiance)")
    export.add_argument("--path", help="only this path id (arrivals, rays, paths, segments, handoff; required for wavefront)")
    export.add_argument("--element", help="handoff: a component instance id whose apertures the json form adds; exposures, "
                                          "power-ratings: only this element")
    _add_filters(export, wavefront=True, irradiance=True)
    export.add_argument("--no-verify", dest="verify", action="store_false", help=NO_VERIFY_HELP)
    export.set_defaults(handler=_export)
