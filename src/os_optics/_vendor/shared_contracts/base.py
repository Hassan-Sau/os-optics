"""Strict base model of the OpenSci wire contracts."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer


class StrictContractModel(BaseModel):
    """Immutable JSON-safe model with unknown fields and non-finite floats rejected."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        ser_json_inf_nan="null",
        serialize_by_alias=True,
    )


class OmitAbsentFields:
    """Mixin (listed before ``StrictContractModel``): the optional fields named in ``_omit_when_none`` are serialized only
    when they are given, so every document that does not use a field added later keeps its exact earlier payload and hash
    -- also in payloads dumped without ``exclude_none`` (``ProjectDocumentV1.payload()``,
    ``VerificationInputV1.input_fingerprint()``), which keep their other ``null`` fields."""

    _omit_when_none: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            for name in self._omit_when_none:
                if name in data and data[name] is None:
                    del data[name]
        return data
