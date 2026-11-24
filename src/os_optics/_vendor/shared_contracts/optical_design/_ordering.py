"""Fail-safe canonical ordering for `mode="before"` validators.

A before-validator sees raw, unvalidated input. Sorting must never raise a raw ``KeyError`` /
``AttributeError`` / ``TypeError`` on malformed items: a malformed submission has to surface as a
pydantic ``ValidationError`` (a client error), so on any failure the value is returned unchanged
and ordinary field validation reports what is wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def sort_raw(value: Any, key: Callable[[Callable[[Any, str], Any], Any], Any]) -> Any:
    """Sort a raw list/tuple with ``key(field_getter, item)``; pass anything unsortable through unchanged."""

    if not isinstance(value, (list, tuple)):
        return value
    try:
        return sorted(value, key=lambda item: key(_field, item))
    except (KeyError, AttributeError, TypeError, ValueError):
        return value
