"""Canonicalization helpers for map/set-like tuple fields.

A dict/map concept represented as a tuple (to stay deeply immutable) must
still have exactly one canonical byte
identity per logical map -- otherwise two callers who build the "same" data
in a different construction order get two different osc1 hashes for it, and
duplicate keys can silently overwrite one another downstream with no
contract-level signal. These helpers are used across
`design_part`/`designer_project`: sort a raw `field_validator(mode="
before")` sequence by an extracted key, and separately report duplicate keys
for a clear `model_validator(mode="after")` error.

Fields whose order is genuinely physical/semantic (placed components, polygon
vertices, a route/process sequence) must NOT use these helpers -- reordering
them changes the design, it does not merely re-express the same data.
"""

from __future__ import annotations

from typing import Any


def item_key(item: Any, *fields: str) -> tuple[Any, ...]:
    """Canonical sort key from a raw dict or a model instance exposing the same attributes."""

    if isinstance(item, dict):
        return tuple(item[field] for field in fields)
    return tuple(getattr(item, field) for field in fields)


#: JSON names of the Python types a raw key can have (for the refusal of keys that cannot be ordered).
_JSON_TYPE_NAMES = {str: "string", int: "number", float: "number", bool: "boolean", type(None): "null", dict: "object", list: "array"}


def _has_fields(item: Any, fields: tuple[str, ...]) -> bool:
    if isinstance(item, dict):
        return all(field in item for field in fields)
    return all(hasattr(item, field) for field in fields) and not isinstance(item, (str, bytes, int, float, list, tuple))


def canonicalize_by_key(value: Any, *fields: str) -> Any:
    """Sort a raw `list`/`tuple` sequence by `item_key(item, *fields)`.

    Passes through anything else unchanged (an already-validated tuple from a prior pass, or a
    non-sequence the field's own type validation will go on to reject with its own error). A sequence
    with an item that is not an object carrying every key field (``1``, ``null``, ``{"value": 1}``) is
    also passed through unsorted: the item's own validation then refuses it with a located error. Key
    values that cannot be compared with each other (``1`` next to ``"a"``, ``null`` next to a string)
    raise ``ValueError``, which the model reports as a validation error -- never a crash.
    """

    if not isinstance(value, (list, tuple)):
        return value
    if not all(_has_fields(item, fields) for item in value):
        return value
    keys = [item_key(item, *fields) for item in value]
    try:
        order = sorted(range(len(value)), key=lambda index: keys[index])
    except TypeError:
        kinds = sorted({_JSON_TYPE_NAMES.get(type(part), type(part).__name__) for key in keys for part in key})
        raise ValueError(f"the items' {'/'.join(fields)} values must be of one comparable type, not a mix of {', '.join(kinds)}") from None
    return [value[index] for index in order]


def duplicate_keys(items: Any, *fields: str) -> list[tuple[Any, ...]]:
    """The canonical keys that appear more than once in `items`, in first-seen order."""

    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    for item in items:
        key = item_key(item, *fields)
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates
