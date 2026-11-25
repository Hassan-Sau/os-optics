"""OS Optics (``os_optics``) by OpenSci: deterministic, stateless, offline 3D optical / layout / control engineering engine.

Pure-optical design on this device (public optical catalog -> Project -> simulation -> local validation record)::

    from os_optics.local import example_project, load_public_catalog, run_optical_project
    run = run_optical_project(example_project())

See ``help(os_optics.local)``. The scientific results of a run directory (monitors, beams, rays, wavefronts, power
budget, exports): ``help(os_optics.results)``. Low-level Engine use::

    from os_optics import ValidationRequest, validate
    result = validate(ValidationRequest(project=...))

A selective evaluation runs the capabilities it names and what they require, and keeps their products for the next one
(``help(os_optics.evaluation)``)::

    from os_optics import evaluate
    from os_optics.evaluation import EvaluationState
    state = EvaluationState()
    readings = evaluate(request, capabilities=("readings",), state=state)
    complete = evaluate(request, state=state)    # runs what the readings did not
"""

import sys as _sys

from ._version import MIN_PYTHON as _MIN_PYTHON

if _sys.version_info < _MIN_PYTHON:
    raise ImportError(
        "os-optics needs Python {0}.{1} or newer; this interpreter is Python {2} ({3}). Install it with a supported "
        "Python: 'uv tool install \"os-optics[protected]\"' (uv selects or downloads a compatible Python), or create "
        "a virtual environment from Python {0}.{1}+ and run 'python -m pip install \"os-optics[protected]\"' in it."
        .format(*_MIN_PYTHON, ".".join(str(p) for p in _sys.version_info[:3]), _sys.executable)
    )

from ._version import DISTRIBUTION_NAME, ENGINE_VERSION, MODEL_VERSIONS, NUMERICAL_CONFIG_VERSION, SCHEMA_VERSION, __version__

# (about a second of imports: a Ctrl-C in them while the os-optics command is starting ends it with one line on stderr and
# exit code 130 instead of a traceback; a program importing os_optics gets its KeyboardInterrupt as always: _command.py)
try:
    from .config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig
    from .contracts.request import EngineConfiguration, ValidationRequest
    from .contracts.results import ValidationResult
    from .errors import ErrorCode, Status, ValidationIssue
    from .validation import (
        RAY_PATHS_VERSION,
        PupilRayPath,
        RayPath,
        RayPathTrace,
        evaluate,
        trace_ray_paths,
        validate,
        validate_json,
        validate_with_ray_paths,
    )
except KeyboardInterrupt:
    from ._command import exit_interrupted_while_starting as _exit_interrupted_while_starting

    _exit_interrupted_while_starting()
    raise

__all__ = [
    "__version__",
    "DISTRIBUTION_NAME",
    "ENGINE_VERSION",
    "SCHEMA_VERSION",
    "NUMERICAL_CONFIG_VERSION",
    "MODEL_VERSIONS",
    "NumericalConfig",
    "DEFAULT_NUMERICAL_CONFIG",
    "EngineConfiguration",
    "ValidationRequest",
    "ValidationResult",
    "ValidationIssue",
    "ErrorCode",
    "Status",
    "evaluate",
    "validate",
    "validate_json",
    "validate_with_ray_paths",
    "trace_ray_paths",
    "RAY_PATHS_VERSION",
    "PupilRayPath",
    "RayPath",
    "RayPathTrace",
]
