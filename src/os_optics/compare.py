"""Cross-runtime result comparison (native CPython vs Pyodide vs a reference runtime).

Rules: canonical input hash identical; every status / code / string / integer / boolean identical; floats equal
within ``cross_runtime_rel_tol`` / ``cross_runtime_abs_tol``; timestamps, runtime descriptors and JSON key order are not compared.
``result_hash`` is reported as information only (it is quantized but a value straddling a rounding boundary can differ).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .config import NumericalConfig

# Fields that legitimately differ between runtimes.
IGNORED_PATHS = frozenset({"manifest.runtime", "manifest.result_hash", "manifest.trace_hash"})


@dataclass(frozen=True)
class Difference:
    path: str
    left: Any
    right: Any
    kind: str  # "type" | "value" | "float" | "length" | "keys"

    def __str__(self) -> str:
        return f"{self.path}: {self.kind}: {self.left!r} != {self.right!r}"


def compare_results(
    left: str | dict, right: str | dict, cfg: NumericalConfig | None = None, *, extra_ignored_paths: frozenset[str] = frozenset()
) -> list[Difference]:
    """Return the list of differences (empty = equivalent within tolerance).

    ``extra_ignored_paths`` adds dotted paths to skip (e.g. ``manifest.engine_build_id`` when comparing against a stored snapshot
    produced by an earlier source revision)."""
    cfg = cfg or NumericalConfig()
    a = json.loads(left) if isinstance(left, (str, bytes)) else left
    b = json.loads(right) if isinstance(right, (str, bytes)) else right
    diffs: list[Difference] = []
    _walk(a, b, "", cfg, diffs, IGNORED_PATHS | extra_ignored_paths)
    return diffs


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ----------------------------------------------------------------------------------------- wrapped-phase fields
#
# Two *result* fields are angles wrapped to (-π, π] modulo 2π, so they are periodic on a circle, not unbounded scalars:
#   * ``observations[*].arrivals[*].beam.phase_rad``        (wrapped in validation/observations.py `beam_state`)
#   * ``interference[*].phase_difference_rad``              (wrapped in optics/interference/two_beam.py `_wrap`)
#
# For such a field the generic ``|a-b| <= abs + rel*max(|a|,|b|)`` rule is wrong near the ±π seam: the relative part
# vanishes for angles near zero, and the SAME physical angle expressed as ``+π-ε`` and ``-π+ε`` compares as ~2π apart.
# These two fields are therefore compared with the shortest circular distance, keeping the SAME tolerance values but
# reinterpreted on the natural 2π angular scale: ``abs_tol + rel_tol * 2π``. No new NumericalConfig field is added.


def _is_wrapped_phase_path(path: str) -> bool:
    """Classify a dotted result path as one of the two normatively-wrapped angular result fields.

    Matches on the exact public result suffixes only; every other ``*_rad`` field (Gaussian divergence / far-field
    half angles, ``fd_step_rotation_rad`` in the manifest, input-contract ``retardance_rad`` / ``reflection_phase_rad`` /
    ``transmission_phase_rad`` / ``phase_s_rad`` / ``phase_p_rad``) is an unbounded scalar and must keep generic
    tolerance. The two suffixes below are unique in the result schema, so the bare suffix match is precise and cannot
    reach a non-phase field.
    """
    return path.endswith(".beam.phase_rad") or path.endswith(".phase_difference_rad")


def _angular_distance_rad(a: float, b: float) -> float:
    """Shortest circular distance between two wrapped angles, in [0, π]."""
    return abs(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)


def _walk(a: Any, b: Any, path: str, cfg: NumericalConfig, out: list[Difference], ignored: frozenset[str]) -> None:
    if path in ignored:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(Difference(path, sorted(set(a) - set(b)), sorted(set(b) - set(a)), "keys"))
        for k in sorted(set(a) & set(b)):
            _walk(a[k], b[k], f"{path}.{k}" if path else k, cfg, out, ignored)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(Difference(path, len(a), len(b), "length"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", cfg, out, ignored)
        return
    if isinstance(a, float) or isinstance(b, float):
        if _is_number(a) and _is_number(b):
            fa, fb = float(a), float(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                out.append(Difference(path, a, b, "float"))
            elif _is_wrapped_phase_path(path):
                if _angular_distance_rad(fa, fb) > cfg.cross_runtime_abs_tol + cfg.cross_runtime_rel_tol * (2.0 * math.pi):
                    out.append(Difference(path, a, b, "float"))
            elif abs(fa - fb) > cfg.cross_runtime_abs_tol + cfg.cross_runtime_rel_tol * max(abs(fa), abs(fb)):
                out.append(Difference(path, a, b, "float"))
            return
    if type(a) is not type(b) and not (_is_number(a) and _is_number(b)):
        out.append(Difference(path, a, b, "type"))
        return
    if a != b:
        out.append(Difference(path, a, b, "value"))
