"""osc1: SHA-256(UTF-8(JCS(value))) exact canonical-data hash profile.

``osc1`` is RFC 8785 (JSON Canonicalization Scheme) applied to the JSON-domain
value model below, with no byte prefix -- domain separation lives inside
hashed contract data via ``schema``/``schema_version``. This module is the
Python reference implementation.

Accepted input is the strict Python mirror of the JSON value domain: ``None``,
``bool``, ``int``, ``float``, ``str``, ``list``, ``dict[str, ...]`` -- checked
by exact type (``type(x) is ...``), not ``isinstance``, so subclasses (enums,
``NamedTuple``, custom ``dict``/``list`` subclasses) are rejected rather than
silently accepted. ``osc1`` is not a general Python-object serializer: convert
Pydantic contracts with ``model_dump(mode="json")`` and exact decimals with
``decimal_to_string`` (:mod:`shared_contracts.common.decimal_string`) before
calling it. It never inspects a value for domain meaning (e.g. pose sign
canonicalization): that happens, if at all, in the caller, before osc1 sees
the data.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ._osc1_numbers import format_ecmascript_number


MAX_SAFE_INTEGER = (1 << 53) - 1
MIN_SAFE_INTEGER = -MAX_SAFE_INTEGER

_SURROGATE_LOW = 0xD800
_SURROGATE_HIGH = 0xDFFF

_CONTROL_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


class Osc1Error(ValueError):
    """Base class for every osc1 canonicalization failure."""


class Osc1UnsupportedTypeError(Osc1Error):
    """Raised when a value's Python type is outside the osc1 JSON-only domain."""


class Osc1NonFiniteFloatError(Osc1Error):
    """Raised for ``NaN`` or +/-``Infinity``, which JCS forbids."""


class Osc1IntegerOutOfRangeError(Osc1Error):
    """Raised for a Python ``int`` outside +/-(2**53 - 1)."""


class Osc1InvalidUnicodeError(Osc1Error):
    """Raised for a string containing a lone UTF-16 surrogate code point."""


class Osc1InvalidObjectKeyError(Osc1Error):
    """Raised when a dict passed to osc1 has a non-``str`` key."""


def canonicalize_osc1_text(value: Any) -> str:
    """Return the RFC 8785 canonical JSON text for ``value``.

    Raises an :class:`Osc1Error` subclass if ``value`` (or anything nested in
    it) is outside the osc1 value domain.
    """

    return _encode(value)


def canonicalize_osc1_bytes(value: Any) -> bytes:
    """Return the exact UTF-8 bytes that osc1 hashes for ``value``."""

    return canonicalize_osc1_text(value).encode("utf-8")


def osc1_digest(value: Any) -> str:
    """Return ``SHA-256(UTF-8(JCS(value)))`` as a lowercase 64-hex-character string."""

    return hashlib.sha256(canonicalize_osc1_bytes(value)).hexdigest()


def _encode(value: Any) -> str:
    if value is None:
        return "null"

    value_type = type(value)

    if value_type is bool:
        return "true" if value else "false"

    if value_type is int:
        if not (MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER):
            raise Osc1IntegerOutOfRangeError(
                f"integer {value} is outside the exactly interoperable IEEE-754 "
                f"range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
            )
        return str(value)

    if value_type is float:
        if not math.isfinite(value):
            raise Osc1NonFiniteFloatError(f"non-finite float is not representable in JSON: {value!r}")
        return format_ecmascript_number(value)

    if value_type is str:
        return _encode_string(value)

    if value_type is list:
        return "[" + ",".join(_encode(item) for item in value) + "]"

    if value_type is dict:
        for key in value:
            if type(key) is not str:
                raise Osc1InvalidObjectKeyError(
                    f"osc1 object keys must be str, got {type(key).__name__}"
                )
        sorted_keys = sorted(value.keys(), key=_utf16_code_units)
        return "{" + ",".join(f"{_encode_string(k)}:{_encode(value[k])}" for k in sorted_keys) + "}"

    raise Osc1UnsupportedTypeError(
        f"unsupported type for osc1 canonicalization: {value_type.__name__}"
    )


def _validate_unicode(value: str) -> None:
    for ch in value:
        code_point = ord(ch)
        if _SURROGATE_LOW <= code_point <= _SURROGATE_HIGH:
            raise Osc1InvalidUnicodeError(
                f"lone surrogate U+{code_point:04X} is not valid Unicode and cannot be osc1-hashed"
            )


def _encode_string(value: str) -> str:
    _validate_unicode(value)
    parts = ['"']
    for ch in value:
        escape = _CONTROL_ESCAPES.get(ch)
        if escape is not None:
            parts.append(escape)
        elif ord(ch) < 0x20:
            parts.append(f"\\u{ord(ch):04x}")
        else:
            parts.append(ch)
    parts.append('"')
    return "".join(parts)


def _utf16_code_units(value: str) -> tuple[int, ...]:
    """RFC 8785 §3.2.3: sort object keys by their UTF-16BE code-unit sequence."""

    _validate_unicode(value)
    encoded = value.encode("utf-16-be")
    return tuple(encoded[i] << 8 | encoded[i + 1] for i in range(0, len(encoded), 2))
