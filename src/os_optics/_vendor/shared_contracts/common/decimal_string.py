"""Canonical exact-decimal wire representation (a decimal number carried as a JSON string).

``DecimalString`` is the strict wire type: a value must already be in canonical spelling. ``OpticalDecimal`` is the
type of the numbers an author writes into an optical model (a Project's declared models and the catalog classes they
reuse): it reads any finite decimal spelling, a JSON number or a ``Decimal`` and stores the canonical string, so the
payload and its hash are one per value whatever the spelling.
"""

from __future__ import annotations

import math
import re
from decimal import Context, Decimal, localcontext
from typing import Annotated

from pydantic import BeforeValidator, Field, StringConstraints


DECIMAL_STRING_PATTERN = (
    r"^(?:0(?:\.[0-9]*[1-9])?|[1-9][0-9]*(?:\.[0-9]*[1-9])?|-(?:0\.[0-9]*[1-9]|[1-9][0-9]*(?:\.[0-9]*[1-9])?))$"
)

DecimalString = Annotated[
    str,
    StringConstraints(strict=True, pattern=DECIMAL_STRING_PATTERN),
    Field(
        description=(
            "Canonical decimal string: no exponent, leading plus/zero, trailing "
            "fractional zero, or negative zero."
        )
    ),
]

#: Longest decimal spelling (characters) and most significant digits an ``OpticalDecimal`` reads.
MAX_DECIMAL_INPUT_LENGTH = 400
#: Largest decimal exponent (``Decimal.adjusted``, either sign) an ``OpticalDecimal`` reads: beyond a float's range.
MAX_DECIMAL_EXPONENT = 400

_CANONICAL = re.compile(DECIMAL_STRING_PATTERN)
_SPELLING = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# Exact for every value read: more digits than any accepted input holds, so normalize() never rounds.
_EXACT = Context(prec=MAX_DECIMAL_INPUT_LENGTH + 2 * MAX_DECIMAL_EXPONENT + 10)


def decimal_to_string(value: Decimal) -> str:
    """Return the canonical DecimalString for a finite Decimal, without using float."""

    if not value.is_finite():
        raise ValueError("DecimalString requires a finite decimal")
    if value.is_zero():
        return "0"

    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def read_decimal(value: object) -> Decimal | None:
    """The exact value of a decimal as an author writes it -- a string in any finite decimal spelling (``"100.0"``,
    ``"1e2"``, ``"+5"``, ``"-0"``), an integer, a float (read through its shortest repr, so ``0.1`` is ``0.1``) or a
    ``Decimal`` -- or ``None`` for anything else (a boolean, NaN, infinity, other text, another type)."""

    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if len(value) > MAX_DECIMAL_INPUT_LENGTH or _SPELLING.fullmatch(value) is None:
            return None
        number = Decimal(value)
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        number = Decimal(repr(value))
    elif isinstance(value, Decimal):
        number = value
    else:
        return None
    if not number.is_finite():
        return None
    if not number.is_zero() and (len(number.as_tuple().digits) > MAX_DECIMAL_INPUT_LENGTH
                                 or abs(number.adjusted()) > MAX_DECIMAL_EXPONENT):
        return None
    return number


def canonical_decimal(value: object) -> object:
    """``OpticalDecimal``'s reader: the canonical DecimalString of a decimal written in any spelling ``read_decimal``
    accepts. A string already in canonical spelling is returned unchanged; a boolean, NaN, infinity or a string that
    is not a decimal is refused; any other type is passed on for the string type to refuse."""

    if isinstance(value, str) and _CANONICAL.fullmatch(value) is not None:
        return value
    if isinstance(value, bool):
        raise ValueError("a decimal number is a JSON number or a decimal string, not a boolean")
    if not isinstance(value, (str, int, float, Decimal)):
        return value
    number = read_decimal(value)
    if number is None:
        raise ValueError(f"a decimal number is a finite JSON number or a decimal string such as \"0.25\" or \"2.5e-3\", "
                         f"got {value!r}")
    with localcontext(_EXACT):
        return decimal_to_string(number)


OpticalDecimal = Annotated[
    str,
    BeforeValidator(canonical_decimal, json_schema_input_type=str | int | float),
    StringConstraints(strict=True, pattern=DECIMAL_STRING_PATTERN),
    Field(
        description=(
            "Decimal number: a JSON number or a decimal string in any finite spelling (\"100\", \"100.0\", \"1e2\"); "
            "stored as the canonical decimal string (no exponent, leading plus/zero, trailing fractional zero or "
            "negative zero)."
        )
    ),
]
