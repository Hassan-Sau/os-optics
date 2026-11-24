"""Structured, localized findings of the optical-solid checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

#: The findings that make a design invalid input (the run is refused before tracing). ``OPTICAL_BODIES_TOUCH`` is an
#: ERROR only where light would cross from one body into the other with no air gap (a contact on traced surfaces inside
#: their clear apertures); anywhere else (rims, stop plates, substrate faces) a contact is INFO.
ERROR_CODES = ("SAG_DOMAIN_INSIDE_OUTLINE", "SURFACES_INTERSECT", "MEDIUM_BOUNDARY_INCONSISTENT", "OPTICAL_BODIES_OVERLAP",
               "MEMBER_OUTSIDE_ENVELOPE", "OPTICAL_BODIES_TOUCH")
#: Where on the bodies a domain, thickness or envelope ERROR lies (``SolidIssue.zone``): ``CLEAR_APERTURE`` -- two surfaces
#: cross, or a profile ends, inside a clear aperture (the part of a surface the trace follows light over);
#: ``OUTSIDE_CLEAR_APERTURE`` -- between the clear apertures and the outline (the body beyond the traced surfaces);
#: ``UNIT_ENVELOPE`` -- against a compound unit's declared envelope (a member outside it, another body inside it).
ZONE_CLEAR_APERTURE = "CLEAR_APERTURE"
ZONE_OUTSIDE_CLEAR_APERTURE = "OUTSIDE_CLEAR_APERTURE"
ZONE_UNIT_ENVELOPE = "UNIT_ENVELOPE"
ZONES = (ZONE_CLEAR_APERTURE, ZONE_OUTSIDE_CLEAR_APERTURE, ZONE_UNIT_ENVELOPE)
REPORT_SCHEMA = "opensci.optical_solids_report"
REPORT_VERSION = "1.0"


@dataclass(frozen=True)
class SolidIssue:
    """One finding: a stable ``code`` (the ERROR ones are ``local.errors.LocalRunErrorCode`` values; ``KNIFE_EDGE``,
    ``EDGE_THICKNESS_IMPLAUSIBLE`` (a warning, or INFO in a declared model's review band) and ``EDGE_FAR_BEYOND_CLEAR_APERTURE``
    are report-only findings,
    ``OPTICAL_BODIES_TOUCH`` an ERROR only on traced surfaces), its
    ``severity``, a factual ``message`` and where: the component ``instance_id`` (and compound ``member_id``), the
    ``surface_id`` (and ``other_surface_id``) it concerns, the other body of a pair (``other_region`` ``UNIT_ENVELOPE``: the
    other component's compound-unit envelope), the position in the design frame
    (``position_mm``) or in the element frame (``local_position_mm``), and the measured value (``value_mm``: an overlap
    depth, a thickness, a profile's domain radius) against ``limit_mm`` where one applies. ``zone`` (one of ``ZONES``), on
    the ``SAG_DOMAIN_INSIDE_OUTLINE``, ``SURFACES_INTERSECT`` and envelope ERROR findings: whether the finding lies inside
    a clear aperture, between the clear apertures and the outline, or against a unit envelope."""

    code: str
    severity: str
    message: str
    instance_id: str
    member_id: str | None = None
    surface_id: str | None = None
    other_instance_id: str | None = None
    other_member_id: str | None = None
    other_surface_id: str | None = None
    #: what of the other component is met when it is not its optical body: ``UNIT_ENVELOPE`` (a compound unit's envelope)
    other_region: str | None = None
    position_mm: tuple[float, ...] | None = None
    local_position_mm: tuple[float, ...] | None = None
    value_mm: float | None = None
    limit_mm: float | None = None
    zone: str | None = None

    @property
    def location(self) -> str:
        where = f"component {self.instance_id}"
        if self.member_id is not None:
            where += f" member {self.member_id}"
        if self.surface_id is not None:
            where += f" surface {self.surface_id}"
        return where

    def as_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        for key in ("position_mm", "local_position_mm"):
            if key in data:
                data[key] = list(data[key])
        data["location"] = self.location
        return data

    def details(self) -> dict[str, Any]:
        """The structured fields of the finding (no code, severity, message or location: those are the problem's own)."""

        return {k: v for k, v in self.as_dict().items() if k not in ("code", "severity", "message", "location")}


@dataclass(frozen=True)
class OpticalSolidsReport:
    """Every finding of one check, and per element the thickness of every gap between consecutive surfaces."""

    issues: tuple[SolidIssue, ...]
    elements: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    pairs_checked: int = 0

    @property
    def errors(self) -> tuple[SolidIssue, ...]:
        return tuple(i for i in self.issues if i.severity == ERROR)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"schema": REPORT_SCHEMA, "schema_version": REPORT_VERSION, "status": "VALID" if self.valid else "INVALID",
                "issues": [i.as_dict() for i in self.issues], "elements": [dict(e) for e in self.elements], "pairs_checked": self.pairs_checked}

    def local_run_problems(self) -> list[Any]:
        """The ERROR findings as ``LocalRunProblem`` for a ``LocalRunError``: code, message, location and ``details``, the
        finding's structured fields (``SolidIssue.details``)."""

        from ..local.errors import LocalRunErrorCode, LocalRunProblem

        return [LocalRunProblem(LocalRunErrorCode(i.code), i.message, i.location, i.details()) for i in self.errors]
