"""Canonical JSON and its hashes, produced piece by piece (large run documents).

A run's largest documents -- the Engine result with thousands of paths, beam pairs or issues, the ray data with every
pupil ray -- are objects whose bulk sits in a few top-level arrays. ``canonical_json_bytes`` and ``hash_quantized`` build the
whole document as a Python tree, then a quantized copy, then one string and its bytes: several whole-document copies at
once. The functions here produce the same bytes and the same digests from the document's top-level members, an array
member element by element (``Elements``), so at any time only one element is held beside the output.

* ``json_parts(parts)`` yields the text of ``canonical_json_bytes`` (``json.dumps(sort_keys=True, indent=2,
  ensure_ascii=False, allow_nan=False)`` plus a final newline) of the object whose members are ``parts``.
* ``hash_quantized_parts(parts, sig_digits, abs_floor)`` is ``hash_quantized`` of the same object.

``parts`` maps each member name to its JSON value, or to ``Elements`` (an iterable of the array's elements, each a JSON
value, read once). JSON strings never hold a raw line break (``json`` escapes it), so a nested value's indented text is
the value's own text with every line break followed by its depth's indentation: the bytes are exactly those of the whole
document (``tests/test_streamed_run_files.py``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .canonical import canonical_json, quantize

#: ``canonical_json_bytes``' indentation (``optical_view/artifacts.py`` ``JSON_INDENT``): two spaces per level.
INDENT = "  "


class Elements:
    """A JSON array member given element by element (each element a JSON value; iterated once)."""

    def __init__(self, items: Iterable[Any]) -> None:
        self.items = items


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=len(INDENT), ensure_ascii=False, allow_nan=False)


def _members(parts: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(((str(name), value) for name, value in parts.items()), key=lambda item: item[0])


def json_parts(parts: Mapping[str, Any]) -> Iterator[str]:
    """The text of ``canonical_json_bytes`` of the object with members ``parts`` (module docstring), in pieces."""

    members = _members(parts)
    if not members:
        yield "{}\n"
        return
    for k, (name, value) in enumerate(members):
        yield ("{" if k == 0 else ",") + "\n" + INDENT + json.dumps(name, ensure_ascii=False) + ": "
        if isinstance(value, Elements):
            first = True
            for element in value.items:
                yield ("[" if first else ",") + "\n" + 2 * INDENT + _text(element).replace("\n", "\n" + 2 * INDENT)
                first = False
            yield "[]" if first else "\n" + INDENT + "]"
        else:
            yield _text(value).replace("\n", "\n" + INDENT)
    yield "\n}\n"


def json_bytes(parts: Mapping[str, Any]) -> bytes:
    """``json_parts`` as one UTF-8 byte string (each piece encoded as it is produced)."""

    return b"".join(piece.encode("utf-8") for piece in json_parts(parts))


def _quantized_parts(parts: Mapping[str, Any], sig_digits: int, abs_floor: float) -> Iterator[str]:
    yield "{"
    for k, (name, value) in enumerate(_members(parts)):
        yield ("," if k else "") + canonical_json(name) + ":"
        if isinstance(value, Elements):
            yield "["
            for n, element in enumerate(value.items):
                yield ("," if n else "") + canonical_json(quantize(element, sig_digits, abs_floor))
            yield "]"
        else:
            yield canonical_json(quantize(value, sig_digits, abs_floor))
    yield "}"


def hash_quantized_parts(parts: Mapping[str, Any], sig_digits: int, abs_floor: float) -> str:
    """``hash_quantized`` of the object with members ``parts`` (module docstring), one piece at a time."""

    digest = hashlib.sha256()
    for piece in _quantized_parts(parts, sig_digits, abs_floor):
        digest.update(piece.encode("utf-8"))
    return digest.hexdigest()


def dict_parts(document: Mapping[str, Any]) -> dict[str, Any]:
    """The members of a JSON object as read, each array member given element by element."""

    return {name: Elements(value) if isinstance(value, list) else value for name, value in document.items()}


def model_parts(model: Any, *, update: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The members of ``model.model_dump(mode="json")`` (a pydantic model; ``update``: members replaced first, as
    ``model_copy(update=...)``), every non-empty tuple member of models dumped element by element: the same JSON as the
    whole dump, which serializes a nested model as its own ``model_dump`` does."""

    if update:
        model = model.model_copy(update=dict(update))
    streamed = {name for name, value in model if isinstance(value, tuple) and value
                and all(hasattr(item, "model_dump") for item in value)}
    head = model.model_dump(mode="json", exclude=streamed)
    for name in streamed:
        head[name] = Elements(item.model_dump(mode="json") for item in getattr(model, name))
    return head


__all__ = ["Elements", "dict_parts", "hash_quantized_parts", "json_bytes", "json_parts", "model_parts"]
