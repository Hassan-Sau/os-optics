"""The entry schema of the interop capability catalog (``os_optics.interop``).

An entry describes one scientific workflow that combines OpenSci with tools the user runs in an environment of their own
choice: which tools fit, what OpenSci provides and what the tool computes, which OpenSci data and entry points the
workflow uses, the inputs only the user can give, the assumptions and limits, and how far OpenSci has checked the
combination (``status``, ``verified``). OpenSci does not ship, host or run any of these tools.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INTEROP_SCHEMA = "opensci.interop_catalog"
INTEROP_SCHEMA_VERSION = "1.0"

Status = Literal["GUIDANCE_ONLY", "DATA_CONTRACT_AVAILABLE", "VERIFIED_EXAMPLE", "PLANNED", "DEPRECATED"]

#: What each status means (``interop_catalog()["statuses"]``).
STATUS_MEANINGS: dict[str, str] = {
    "GUIDANCE_ONLY": "The entry states which OpenSci data the tool needs, how to obtain it and what the tool adds. OpenSci has not "
                     "executed this combination.",
    "DATA_CONTRACT_AVAILABLE": "OpenSci publishes a versioned export or data contract, with stated units, frames and conventions, that "
                               "carries the data this workflow needs. Executed checks, when there are any, are listed under verified.",
    "VERIFIED_EXAMPLE": "A named example with pinned tool versions was executed and its result checked against an independent "
                        "reference; verified lists the tool, versions, date and example.",
    "PLANNED": "The OpenSci data this workflow needs is not published yet; the entry states what is missing.",
    "DEPRECATED": "The tool is no longer maintained upstream. The entry is kept so earlier work can be reproduced; use the "
                  "recommended replacement for new work.",
}

Direction = Literal[
    "OPTIMIZATION_TOLERANCE",
    "IMAGING_SCALAR_DIFFRACTION",
    "DOE_VECTOR_INVERSE_DESIGN",
    "CAVITIES_INTERFEROMETERS",
    "QUANTUM_PHOTONICS",
    "MICRO_NANO_PIC",
    "METASURFACE_VIRTUAL_ELEMENT",
    "COATINGS_MATERIALS_POLARIZATION",
    "CAD_BREP_GEOMETRY",
    "SPECIALIST",
]

#: The directions in catalog order, with a short title each.
DIRECTION_TITLES: dict[str, str] = {
    "OPTIMIZATION_TOLERANCE": "Automated design, tolerancing and robustness",
    "IMAGING_SCALAR_DIFFRACTION": "Imaging and scalar diffraction",
    "DOE_VECTOR_INVERSE_DESIGN": "DOE / SLM, vector fields and inverse design",
    "CAVITIES_INTERFEROMETERS": "Cavities and complex interferometers",
    "QUANTUM_PHOTONICS": "Quantum photonic systems",
    "MICRO_NANO_PIC": "Micro- and nano-photonic devices and photonic integrated circuits",
    "METASURFACE_VIRTUAL_ELEMENT": "Metasurface response as a virtual optical element",
    "COATINGS_MATERIALS_POLARIZATION": "Coatings, materials and polarization statistics",
    "CAD_BREP_GEOMETRY": "Exact CAD geometry and data exchange",
    "SPECIALIST": "Specialist topics and engineering verification",
}

_ID = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
TOOL_NAME_MAX = 80
TITLE_MAX = 120
COMMIT_MAX = 40
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolV1(_Model):
    """A candidate tool. ``license`` is quoted as the upstream distribution reports it (``license_source``) on
    ``license_checked``; it is a fact about the tool, never a statement of its terms for any particular use."""

    name: str = Field(min_length=1, max_length=TOOL_NAME_MAX)
    role: Literal["RECOMMENDED", "ALTERNATIVE", "CROSS_CHECK", "COMPANION", "HISTORICAL"]
    url: str = Field(description="The tool's official website, documentation or source repository.")
    maintainer: Literal["UPSTREAM_PROJECT", "THIRD_PARTY_WRAPPER", "COMMERCIAL_VENDOR"] = Field(
        "UPSTREAM_PROJECT", description="THIRD_PARTY_WRAPPER: a package by another party that wraps the upstream library.")
    distribution: Literal["OPEN_SOURCE", "COMMERCIAL", "FREE_WITH_REGISTRATION"] = "OPEN_SOURCE"
    maintenance: Literal["ACTIVE", "ARCHIVED", "LEGACY_ENVIRONMENT"] = "ACTIVE"
    license: str | None = Field(None, description="The license as the upstream distribution reports it (None: not checked).")
    license_source: str | None = None
    license_checked: str | None = Field(None, description="Date the license was read (YYYY-MM-DD).")
    computes: str = Field(min_length=1, description="What this tool computes in the workflow.")
    notes: tuple[str, ...] = ()

    @field_validator("url")
    @classmethod
    def _https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("a tool url is an https:// link")
        return value

    @model_validator(mode="after")
    def _license(self) -> ToolV1:
        if (self.license is None) != (self.license_checked is None) or (self.license is None) != (self.license_source is None):
            raise ValueError("license, license_source and license_checked are given together")
        if self.license_checked is not None and not _DATE.match(self.license_checked):
            raise ValueError("license_checked is YYYY-MM-DD")
        return self


class EntryPointV1(_Model):
    """An existing OpenSci entry point the workflow uses, by its exact name: a Python attribute path
    (``os_optics.results.load_run``), a ``json_api`` function name or a CLI command line."""

    kind: Literal["PYTHON", "JSON_API", "CLI"]
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class VerifiedRecordV1(_Model):
    """One executed check: the tool and its version, the Python it ran on, the date, the case in the public kit repository
    (JunLin615/os-optics-kit) and the commit of that repository's history it ran at, and the OpenSci Engine version it
    used."""

    tool: str
    version: str
    python: str
    date: str
    example: str = Field(description="Path of the case in the public kit repository (JunLin615/os-optics-kit).")
    companion_commit: str = Field(min_length=7, max_length=COMMIT_MAX)
    engine_version: str
    reference: str = Field(description="The independent reference the result was checked against.")
    outcome: str

    @field_validator("date")
    @classmethod
    def _date(cls, value: str) -> str:
        if not _DATE.match(value):
            raise ValueError("date is YYYY-MM-DD")
        return value


class InteropEntryV1(_Model):
    entry_id: str
    title: str = Field(min_length=1, max_length=TITLE_MAX)
    direction: Direction
    status: Status
    scenario: str = Field(min_length=1)
    tools: tuple[ToolV1, ...] = Field(min_length=1)
    opensci_provides: tuple[str, ...] = Field(min_length=1)
    tool_computes: tuple[str, ...] = Field(min_length=1)
    entry_points: tuple[EntryPointV1, ...] = ()
    data: tuple[str, ...] = Field((), description="The OpenSci exports and documents the workflow reads (kinds, schema ids).")
    user_inputs: tuple[str, ...] = Field((), description="What only the user can provide: OpenSci never infers it.")
    assumptions: tuple[str, ...] = ()
    limits: tuple[str, ...] = Field((), description="What the workflow cannot do, and what OpenSci refuses (with the code).")
    conventions: tuple[str, ...] = Field((), description="Where the units, frames, phase and normalization conventions are stated.")
    runtime: tuple[str, ...] = Field((), description="Where and how the tool runs: local, WSL, a separate environment, commercial software.")
    verified: tuple[VerifiedRecordV1, ...] = ()
    replaced_by: str | None = Field(None, description="DEPRECATED entries: the entry to use instead.")

    @field_validator("entry_id")
    @classmethod
    def _entry_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("entry_id is lower-case letters, digits and '-'")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> InteropEntryV1:
        if self.status == "VERIFIED_EXAMPLE" and not self.verified:
            raise ValueError("a VERIFIED_EXAMPLE entry lists at least one verified record")
        if self.status in ("GUIDANCE_ONLY", "PLANNED") and self.verified:
            raise ValueError(f"a {self.status} entry has no verified records")
        if (self.status == "DEPRECATED") != (self.replaced_by is not None):
            raise ValueError("replaced_by is given exactly for a DEPRECATED entry")
        if self.status == "DEPRECATED" and not any(t.maintenance == "ARCHIVED" or t.role == "HISTORICAL" for t in self.tools):
            raise ValueError("a DEPRECATED entry names its archived or historical tool")
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names are unique within an entry")
        return self

    def summary(self) -> dict[str, object]:
        """The compact row of ``interop list``."""

        return {"entry_id": self.entry_id, "title": self.title, "direction": self.direction, "status": self.status,
                "tools": [t.name for t in self.tools], "verified": len(self.verified)}
