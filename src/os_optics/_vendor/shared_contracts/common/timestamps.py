"""Strict UTC ISO-8601 wire timestamp (every contract time is UTC ISO 8601).

A plain ``datetime`` field lets Pydantic accept any offset (``+08:00``, no offset at all, a naive
value) and re-serializes it in whatever form the stdlib prefers -- two callers describing the same
instant in different offsets would produce different wire bytes for supposedly the same fact. This
package's convention for every other wire primitive (``DecimalString``, ``OpenSciId``, ...) is a
strict, pattern-constrained ``str`` with exactly one canonical textual form; ``UtcTimestamp`` applies
the same discipline to timestamps: the literal ``Z`` suffix is mandatory, no other offset spelling
(``+00:00``, ``-00:00``) is accepted, and an optional fractional-seconds component is allowed but
never required.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints


UTC_TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$"

UtcTimestamp = Annotated[
    str,
    StringConstraints(strict=True, pattern=UTC_TIMESTAMP_PATTERN),
    Field(description="Strict UTC ISO-8601 timestamp; the literal 'Z' suffix is mandatory, no other offset form is accepted."),
]
