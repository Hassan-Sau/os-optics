"""`FrozenStringMap`: a deeply-immutable, hashable `str -> str` mapping that is wire-identical to a
plain JSON object.

`HashRef.params` must be deeply immutable/hashable like every other contract field, but its wire
JSON shape (`object<string,string> | null`) is normative: a plain object, e.g.
``{"sig_digits": "9", "abs_floor": "0.000000000001"}``. A `dict[str, str]` field satisfies the wire
shape but not deep immutability; a `tuple[{key, value}, ...]` field satisfies deep immutability but
breaks the wire shape. `FrozenStringMap` is a tuple-backed `Mapping[str, str]` with a custom Pydantic
core schema and JSON Schema, so it is BOTH: internally an immutable, hashable, sorted tuple of pairs
(never a `dict`), and externally -- on the wire and in the generated JSON Schema -- an ordinary JSON
object with string values, byte-identical to what `dict[str, str] | None` already produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


class FrozenStringMap(Mapping[str, str]):
    """Immutable, hashable `str -> str` mapping; serializes as a plain JSON object, sorted by key.

    Every key and every value must be EXACTLY `str` (`type(x) is str`, not `isinstance` -- the same
    exact-type discipline `osc1` itself uses for its JSON-domain checks, so a `bool`/`int`/`None`/str
    subclass such as an enum member is rejected rather than silently accepted or coerced). This keeps
    the `dict[str, str]` wire contract exact: `{1: "x"}`, `{"x": 1}`, `{"x": True}` and
    `{"x": None}` are all invalid constructions, never quietly stringified.
    """

    __slots__ = ("_pairs",)

    def __init__(self, items: Mapping[str, str] | Iterable[tuple[str, str]] = ()) -> None:
        pairs = tuple(items.items()) if isinstance(items, Mapping) else tuple(items)
        for key, value in pairs:
            if type(key) is not str:
                raise ValueError(f"FrozenStringMap keys must be str, got {type(key).__name__}: {key!r}")
            if type(value) is not str:
                raise ValueError(f"FrozenStringMap values must be str, got {type(value).__name__}: {value!r}")
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            raise ValueError(f"duplicate key(s) in FrozenStringMap: {duplicates}")
        object.__setattr__(self, "_pairs", tuple(sorted(pairs, key=lambda pair: pair[0])))

    def __getitem__(self, key: str) -> str:
        for candidate_key, value in self._pairs:
            if candidate_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self._pairs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenStringMap):
            return self._pairs == other._pairs
        if isinstance(other, Mapping):
            return dict(self._pairs) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenStringMap({dict(self._pairs)!r})"

    @classmethod
    def _validate(cls, value: Any) -> FrozenStringMap:
        if isinstance(value, FrozenStringMap):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        raise ValueError(f"expected a str-keyed mapping for FrozenStringMap, got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: dict(value.items()),
                when_used="always",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_obj: Any, handler: Any) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": {"type": "string"}}
