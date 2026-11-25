"""Parameters of native component classes: their published description and their parsing from JSON values.

A parameter is described by a ``ParameterSpec`` (name, type, unit, constraints, default) and parsed from a plain JSON
value: a decimal is a decimal string, an integer or a JSON number (read through its shortest representation, so ``0.1``
is exactly ``0.1``); a wavelength band or a pair is a two-element list (or ``"a,b"``); a material, a lens or a layer
stack is a JSON object (see ``materials.py`` and the class descriptions). Every problem of a request is collected and
reported together (``NativeComponentError``).
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from .._vendor.shared_contracts.common.text import check_author_text, nfc
from .errors import NativeComponentError, NativeErrorCode, NativeProblem
from .numbers import text, to_decimal


class ParameterType(StrEnum):
    DECIMAL = "decimal"
    ENUM = "enum"
    BOOLEAN = "boolean"
    TEXT = "text"
    MODEL_ID = "model_id"
    INSTANCE_ID = "instance_id"
    MATERIAL = "material"
    COATING = "coating"
    BAND = "wavelength_band"
    PAIR = "decimal_pair"
    TRIPLE = "decimal_triple"
    LENS = "lens"
    LAYERS = "layers"
    JSON = "json"


#: Model ids (``ProjectOpticalModelV1.model_id``) and instance ids written by the library.
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
INSTANCE_ID_PATTERN = re.compile(r"^[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]{0,127}$")
#: The longest text parameter (a display name, as the Project's ``AuthorName``): any Unicode text, stored in NFC.
MAX_TEXT_LENGTH = 200
#: The most layers a layered plate may have (a SURFACE_SEQUENCE holds at most 16 surfaces).
MAX_LAYERS = 15


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter of a native class.

    ``default``: the value used when the parameter is absent (a JSON value, parsed like a given one); ``default_rule``:
    when the default is computed from other parameters, how (the value is then chosen by the class). A parameter with
    neither and ``required=False`` is simply optional (absent means "not used").
    """

    name: str
    type: ParameterType
    description: str
    unit: str | None = None
    required: bool = False
    default: Any = None
    default_rule: str | None = None
    minimum: Decimal | None = None
    exclusive_minimum: Decimal | None = None
    maximum: Decimal | None = None
    exclusive_maximum: Decimal | None = None
    nonzero: bool = False
    choices: tuple[str, ...] = ()
    lens_classes: tuple[str, ...] = field(default=())

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type.value, "required": self.required, "description": self.description}
        if self.unit is not None:
            out["unit"] = self.unit
        if self.default is not None:
            out["default"] = self.default
        if self.default_rule is not None:
            out["default_rule"] = self.default_rule
        constraints: dict[str, Any] = {}
        for key in ("minimum", "exclusive_minimum", "maximum", "exclusive_maximum"):
            value = getattr(self, key)
            if value is not None:
                constraints[key] = text(value)
        if self.nonzero:
            constraints["nonzero"] = True
        if constraints:
            out["constraints"] = constraints
        if self.choices:
            out["choices"] = list(self.choices)
        if self.lens_classes:
            out["lens_classes"] = list(self.lens_classes)
        return out


def _range_problem(spec: ParameterSpec, value: Decimal) -> str | None:
    if spec.nonzero and value == 0:
        return "must not be zero"
    if spec.minimum is not None and value < spec.minimum:
        return f"must be at least {text(spec.minimum)}"
    if spec.exclusive_minimum is not None and value <= spec.exclusive_minimum:
        return f"must be greater than {text(spec.exclusive_minimum)}"
    if spec.maximum is not None and value > spec.maximum:
        return f"must be at most {text(spec.maximum)}"
    if spec.exclusive_maximum is not None and value >= spec.exclusive_maximum:
        return f"must be smaller than {text(spec.exclusive_maximum)}"
    return None


def _decimals(raw: Any, count: int) -> tuple[Decimal, ...] | None:
    if isinstance(raw, str):
        raw = [part for part in raw.split(",")]
    if isinstance(raw, Mapping) and count == 2 and set(raw) == {"min_nm", "max_nm"}:
        raw = [raw["min_nm"], raw["max_nm"]]
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        return None
    values = tuple(to_decimal(v) for v in raw)
    return None if any(v is None for v in values) else values  # type: ignore[return-value]


def parse_value(spec: ParameterSpec, raw: Any) -> Any:
    """The parsed value of ``raw`` for ``spec``; raises ``ValueError`` with a plain message when it is not valid."""

    kind = spec.type
    if kind is ParameterType.DECIMAL:
        value = to_decimal(raw)
        if value is None:
            raise ValueError(f"expected a decimal number, got {raw!r}")
        problem = _range_problem(spec, value)
        if problem:
            raise ValueError(f"{problem} (got {text(value)})")
        return value
    if kind is ParameterType.ENUM:
        if not isinstance(raw, str) or raw.upper() not in spec.choices:
            raise ValueError(f"expected one of {list(spec.choices)}, got {raw!r}")
        return raw.upper()
    if kind is ParameterType.BOOLEAN:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        raise ValueError(f"expected true or false, got {raw!r}")
    if kind is ParameterType.TEXT:
        value = nfc(raw) if isinstance(raw, str) else None
        if value is None or not 1 <= len(value) <= MAX_TEXT_LENGTH:
            raise ValueError(f"expected text of 1 to {MAX_TEXT_LENGTH} characters")
        return check_author_text(value)
    if kind is ParameterType.MODEL_ID:
        if not isinstance(raw, str) or not MODEL_ID_PATTERN.match(raw):
            raise ValueError("expected a model id: a letter, then letters, digits, '_' or '-' (at most 64)")
        return raw
    if kind is ParameterType.INSTANCE_ID:
        if not isinstance(raw, str) or not INSTANCE_ID_PATTERN.match(raw):
            raise ValueError("expected a component instance id (no '/', no whitespace at the start, at most 128 characters)")
        return raw
    if kind in (ParameterType.BAND, ParameterType.PAIR, ParameterType.TRIPLE):
        count = 3 if kind is ParameterType.TRIPLE else 2
        values = _decimals(raw, count)
        if values is None:
            raise ValueError(f"expected {count} decimal numbers, as a list or 'a,b{',c' if count == 3 else ''}'")
        for value in values:
            problem = _range_problem(spec, value)
            if problem:
                raise ValueError(f"each value {problem} (got {text(value)})")
        if kind is ParameterType.BAND and not values[0] < values[1]:
            raise ValueError("a wavelength band is [min_nm, max_nm] with min_nm < max_nm")
        if kind is ParameterType.TRIPLE and not values[0] < values[1] < values[2]:
            raise ValueError("expected three strictly increasing values")
        return values
    if kind is ParameterType.MATERIAL:
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, Mapping):
            return dict(raw)
        raise ValueError("expected a public catalog material id (e.g. 'N-BK7') or a material object")
    if kind is ParameterType.COATING:
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, Mapping):
            return dict(raw)
        raise ValueError("expected a public catalog coating id or a declared coating object")
    if kind is ParameterType.LENS:
        if not isinstance(raw, Mapping) or set(raw) - {"class_id", "parameters"} or not isinstance(raw.get("class_id"), str):
            raise ValueError('expected {"class_id": <lens class>, "parameters": {...}}')
        if raw["class_id"] not in spec.lens_classes:
            raise ValueError(f"the lens class must be one of {list(spec.lens_classes)}")
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("'parameters' must be an object")
        return {"class_id": raw["class_id"], "parameters": dict(parameters)}
    if kind is ParameterType.LAYERS:
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= MAX_LAYERS:
            raise ValueError(f'expected a list of 1 to {MAX_LAYERS} layers {{"material": ..., "thickness_mm": ...}}')
        layers = []
        for index, layer in enumerate(raw):
            if not isinstance(layer, Mapping) or set(layer) != {"material", "thickness_mm"}:
                raise ValueError(f'layer {index}: expected exactly {{"material": ..., "thickness_mm": ...}}')
            thickness = to_decimal(layer["thickness_mm"])
            if thickness is None or thickness <= 0:
                raise ValueError(f"layer {index}: thickness_mm must be a positive decimal")
            material = layer["material"]
            if not (isinstance(material, str) and material.strip()) and not isinstance(material, Mapping):
                raise ValueError(f"layer {index}: material must be a catalog material id or a material object")
            layers.append({"material": material.strip() if isinstance(material, str) else dict(material), "thickness_mm": thickness})
        return tuple(layers)
    if kind is ParameterType.JSON:
        if isinstance(raw, str) and raw.strip()[:1] in ("{", "["):
            try:
                raw = json.loads(raw)
            except ValueError as exc:
                raise ValueError(f"not valid JSON: {exc}") from None
        if not isinstance(raw, (Mapping, list, tuple, str)):
            raise ValueError("expected a JSON object, list or text")
        return copy.deepcopy(raw)
    raise ValueError(f"unsupported parameter type {kind}")  # pragma: no cover - the enum is exhaustive


def parse_parameters(specs: tuple[ParameterSpec, ...], raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Parsed values of every parameter of ``specs`` (absent optional ones are ``None``), or ``NativeComponentError``
    listing every unknown, missing and invalid parameter."""

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_INVALID, "parameters must be a JSON object")])
    by_name = {spec.name: spec for spec in specs}
    problems = [NativeProblem(NativeErrorCode.PARAMETER_UNKNOWN, f"unknown parameter; this class takes {sorted(by_name)}", str(name))
                for name in raw if name not in by_name]
    values: dict[str, Any] = {}
    for spec in specs:
        given = raw.get(spec.name)
        if given is None:
            if spec.required:
                problems.append(NativeProblem(NativeErrorCode.PARAMETER_MISSING, f"required: {spec.description}", spec.name))
                continue
            given = spec.default
        if given is None:
            values[spec.name] = None
            continue
        try:
            values[spec.name] = parse_value(spec, given)
        except ValueError as exc:
            problems.append(NativeProblem(NativeErrorCode.PARAMETER_INVALID, str(exc), spec.name))
    if problems:
        raise NativeComponentError(problems)
    return values


__all__ = ["MAX_LAYERS", "MODEL_ID_PATTERN", "ParameterSpec", "ParameterType", "parse_parameters", "parse_value"]
