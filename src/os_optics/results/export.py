"""File formats of the result exports: CSV, JSON Lines, JSON and (dense grids only) NPZ, all built in memory as bytes.

* ``csv``: UTF-8, one header line with the column names (units in the names, e.g. ``beam_radius_x_mm``), then one line
  per row, ``\\n`` line ends. Numbers are written in Python's shortest round-trip form (parsing them gives back the exact
  value), booleans as ``true`` / ``false``, an undefined value as an empty field, a list as its items joined by ``;`` and
  an object (the ``value_basis`` column) as its compact JSON form with sorted keys.
  A field containing a comma, a quote or a line break is quoted (``"..."``, inner quotes doubled). A text field that a
  spreadsheet would read as a formula (it starts with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return, e.g. an id
  chosen as ``=SUM(1+1)``) is written with a leading ``'``, so opening the file never evaluates it; numbers, negative
  ones included, are written as they are.
* ``jsonl``: one JSON object per row and line, keys in column order (lists stay JSON lists).
* ``json``: one document ``{"schema": "opensci.results_export", "schema_version", "kind", "run", "filters", "columns",
  "rows", "summary"}`` with the column descriptions (name, unit, description).
* ``npz``: a NumPy ``.npz`` archive (uncompressed ZIP of ``.npy`` arrays, fixed timestamps: the same data always gives the
  same bytes), for dense grids only (``irradiance``).

No value is NaN or infinite: undefined quantities are empty / ``null``.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from .tables import COLUMNS, LIST_SEPARATOR, columns_of

EXPORT_SCHEMA = "opensci.results_export"
EXPORT_SCHEMA_VERSION = "1.0"
FORMATS = ("csv", "jsonl", "json", "npz")
NPZ_KINDS = ("irradiance",)
# ZIP entries of an .npz carry this fixed timestamp (the earliest a ZIP can record), so archives are reproducible.
NPZ_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
CSV_SPECIAL = (",", '"', "\n", "\r")
#: The first characters that make a spreadsheet read a text cell as a formula (CSV formula injection).
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _is_text(value: Any) -> bool:
    """A text value (not a number): an id, a code, a name -- or a list whose first item is one."""

    if isinstance(value, (list, tuple)):
        return bool(value) and _is_text(value[0])
    return isinstance(value, str)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    text_value = _is_text(value)
    if isinstance(value, (list, tuple)):
        value = LIST_SEPARATOR.join(_cell(v) for v in value)
    text = str(value)
    if text_value and text.startswith(FORMULA_START):
        text = "'" + text
    if any(c in text for c in CSV_SPECIAL) or text != text.strip():
        return '"' + text.replace('"', '""') + '"'
    return text


def csv_bytes(kind: str, rows: list[dict[str, Any]]) -> bytes:
    names = [name for name, _unit, _text in COLUMNS[kind]]
    lines = [",".join(names)] + [",".join(_cell(row[name]) for name in names) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")


def json_document(kind: str, rows: list[dict[str, Any]], *, run: dict[str, Any], filters: dict[str, Any],
                  summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"schema": EXPORT_SCHEMA, "schema_version": EXPORT_SCHEMA_VERSION, "kind": kind, "run": run, "filters": filters,
            "columns": columns_of(kind), "rows": rows, "summary": summary or {}}


def json_bytes(document: Any) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """An ``.npz`` archive of ``arrays`` (``numpy.load(io.BytesIO(data))`` reads it), byte-reproducible."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=NPZ_TIMESTAMP)
            info.external_attr = 0
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def arrays_of(data: bytes) -> dict[str, np.ndarray]:
    """The arrays of an ``.npy`` file (under ``"array"``) or an ``.npz`` archive, from its bytes; never unpickles. ``ValueError``
    when the bytes are neither."""

    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"not an .npy or .npz file: {exc}") from None
    if isinstance(loaded, np.ndarray):
        return {"array": loaded}
    with loaded:
        return {name: loaded[name] for name in loaded.files}
