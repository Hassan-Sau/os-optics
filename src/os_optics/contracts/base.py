"""Shared base types for public contracts (Pydantic v2, frozen, strict, JSON-friendly)."""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer


class StrictModel(BaseModel):
    """Base of every public contract: immutable, no unknown fields, no NaN/Inf."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, ser_json_inf_nan="null")


def schema_description(text: str) -> ConfigDict:
    """The ``model_config`` of a ``StrictModel`` whose JSON schema carries ``text`` as its description instead of the class
    docstring: the schema states what the record is, its availability and where its model is documented; the docstring
    keeps the model, its derivations and the cases, for the reader of the code and of ``help()``."""

    return ConfigDict(json_schema_extra={"description": text})


class OmitAbsentFields:
    """Mixin of a contract whose newer optional fields (``absent_omitted``) are serialized only when given (not ``None``,
    not empty), so a document that does not use them keeps its exact earlier JSON form and hash."""

    absent_omitted: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for name in self.absent_omitted:
                if name in data and (data[name] is None or data[name] == () or data[name] == []):
                    data.pop(name)
        return data


Vector3 = Annotated[tuple[float, float, float], Field(description="Cartesian vector (x, y, z).")]
Vector2 = Annotated[tuple[float, float], Field(description="2D vector (u, v).")]
Quaternion = Annotated[
    tuple[float, float, float, float],
    Field(description="Quaternion (w, x, y, z), Hamilton, active rotation; non-zero, taken by its direction (normalized)."),
]
Complex = Annotated[tuple[float, float], Field(description="Complex number as [real, imag].")]

# No '/', no leading whitespace and no control character (C0, DEL or C1), like the Project contract's identifiers.
Id = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*$",
        description="Stable identifier.",
    ),
]


def is_finite_tuple(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(v) for v in values)


def unit_direction(values: tuple[float, ...], tol: float, name: str) -> tuple[float, ...]:
    """A stated direction as a unit vector: its components unchanged when the squared length is within ``tol`` of 1, else
    divided by the length (a direction is taken by its direction, whatever its length); a zero-length one is refused."""
    square = sum(v * v for v in values)
    if not math.isfinite(square) or square == 0.0:
        raise ValueError(f"{name} must be a non-zero direction")
    if abs(square - 1.0) <= tol:
        return values
    length = math.sqrt(square)
    return tuple(v / length for v in values)
