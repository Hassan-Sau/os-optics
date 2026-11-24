"""Canonical serialization and hashing (SHA-256).

Hashes are unkeyed SHA-256 digests that identify content for integrity and reproducibility checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def quantize_float(value: float, sig_digits: int, abs_floor: float) -> float:
    """Round to ``sig_digits`` significant digits; magnitudes below ``abs_floor`` (and -0.0) become 0.0."""
    if not math.isfinite(value):
        raise ValueError("non-finite float cannot be canonicalized")
    if abs(value) < abs_floor:
        return 0.0
    return float(f"{value:.{sig_digits - 1}e}")


def quantize(obj: Any, sig_digits: int, abs_floor: float) -> Any:
    """Recursively quantize every float in a JSON-compatible structure."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return quantize_float(obj, sig_digits, abs_floor)
    if isinstance(obj, dict):
        return {str(k): quantize(v, sig_digits, abs_floor) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [quantize(v, sig_digits, abs_floor) for v in obj]
    raise TypeError(f"unsupported type for canonical serialization: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, UTF-8-safe, ints/floats via ``repr`` shortest form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_exact(obj: Any) -> str:
    """Hash of a JSON structure with floats kept exactly (used for inputs: identical JSON -> identical floats)."""
    return sha256_hex(canonical_json(_normalize_negative_zero(obj)))


def hash_quantized(obj: Any, sig_digits: int, abs_floor: float) -> str:
    """Hash of a JSON structure with floats quantized (used for results)."""
    return sha256_hex(canonical_json(quantize(obj, sig_digits, abs_floor)))


def _normalize_negative_zero(obj: Any) -> Any:
    if isinstance(obj, float):
        return 0.0 if obj == 0.0 else obj
    if isinstance(obj, dict):
        return {k: _normalize_negative_zero(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_negative_zero(v) for v in obj]
    return obj
