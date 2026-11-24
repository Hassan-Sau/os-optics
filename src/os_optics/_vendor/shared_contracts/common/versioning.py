"""Self-describing schema headers and compatibility primitives."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from ..base import StrictContractModel
from .identifiers import SchemaId


SchemaVersion = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"),
    Field(description="Contract version as MAJOR.MINOR."),
]


# The engine's (os-optics) own, independent version axis (the Engine keeps its x.y.z). This is NOT `SchemaVersion` and must never be confused with it: every
# generic contract (e.g. `opensci.project`) is versioned `MAJOR.MINOR` via `SchemaVersion`
# above; only `os_optics.SCHEMA_VERSION` (and anything citing it verbatim, e.g.
# `EngineIdentityV1.schema_version`) uses this stricter `MAJOR.MINOR.PATCH` grammar. A wire
# value here must equal the Engine's own `SCHEMA_VERSION` string exactly -- never truncated to
# `MAJOR.MINOR` to fit the generic type.
EngineSchemaVersion = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"),
    Field(description="The engine's (os-optics) own schema version as MAJOR.MINOR.PATCH (independent of the generic MAJOR.MINOR SchemaVersion)."),
]


class SchemaHeader(StrictContractModel):
    """Mandatory self-description fields embedded by persisted contract objects."""

    schema_id: SchemaId = Field(alias="schema")
    schema_version: SchemaVersion


class CompatibilityStatus(StrEnum):
    """Result of checking a schema header before authoritative processing."""

    COMPATIBLE = "compatible"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    NEWER_VERSION = "newer_unknown_version"
    UNSUPPORTED_MAJOR = "unsupported_major"


def check_schema_compatibility(
    header: SchemaHeader,
    *,
    expected_schema: str,
    supported_version: str,
) -> CompatibilityStatus:
    """Classify a validated header; older same-major versions are compatible."""

    if header.schema_id != expected_schema:
        return CompatibilityStatus.UNSUPPORTED_SCHEMA
    received_major, received_minor = (int(part) for part in header.schema_version.split("."))
    supported_major, supported_minor = (int(part) for part in supported_version.split("."))
    if received_major != supported_major:
        return CompatibilityStatus.UNSUPPORTED_MAJOR
    if received_minor > supported_minor:
        return CompatibilityStatus.NEWER_VERSION
    return CompatibilityStatus.COMPATIBLE
