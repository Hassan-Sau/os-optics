"""Optical-solid validity: are the optical bodies of a design physically possible solids? Checked on the compiled
elements before any ray is traced, for every design intent, independently of catalog identity and of rays.

Per element (``checks.element_issues``):

* ``SAG_DOMAIN_INSIDE_OUTLINE`` (error): a surface's profile (a steep sphere or conic) is not defined everywhere inside
  the outline it is traced and drawn over (a profile is never extended or flattened beyond its domain);
* ``SURFACES_INTERSECT`` (error): two consecutive surfaces of an element cross somewhere over its whole outline -- in the
  clear aperture or between it and the edge (a negative edge thickness), through glass or through an air gap of the
  model; ``KNIFE_EDGE`` (warning): the glass between them thins to zero; the centre, minimum (and where) and edge
  thickness of every gap are reported as facts. Each surface is checked over its own mechanical edge (held flat beyond
  it) and each gap over its own outline (``SurfaceSequenceModelV1.gap_outline_mm``). Over the clear aperture of either
  surface (its exact shape, holes left out) the own profiles of the two surfaces bounding glass are compared too, none
  held flat: where they cross there, ``SURFACES_INTERSECT`` names the clear aperture, the thickness between the profiles
  and the surface whose flat hold keeps the drawn body positive (fact ``clear_aperture_min_thickness_mm``);
* ``EDGE_THICKNESS_IMPLAUSIBLE``: a glass gap thicker at its edge than 1.5 times its clear diameter (a warning, for any
  model: beyond every published product); for a declared model between its review ratio (0.35 by default) and that, an
  INFO finding (``checks.EdgeThicknessBand``; the ratio and the element are named);
  ``EDGE_FAR_BEYOND_CLEAR_APERTURE`` (warning): a surface's glass reaching beyond 3 times its clear radius (and by more
  than 2 mm) -- both the sign of an outline that is not the element's mechanical edge (``checks.EDGE_*`` limits);
* ``MEDIUM_BOUNDARY_INCONSISTENT`` (error): the media on both sides of the compiled surfaces do not chain (a sequence's
  ``-Z`` / ``+Z`` media against its model, a surface set's front / back against its solid).

Per pair of placed bodies (``checks.pair_issues``): ``OPTICAL_BODIES_OVERLAP`` (error) when two optical solids share
volume deeper than ``TOLERANCE_MM`` (1e-6 mm), ``OPTICAL_BODIES_TOUCH`` when they meet: an error where light would cross
from one body into the other (traced surfaces of both inside their clear apertures, checked at the contact and, on one
axis, at the axial surface points -- the tracer cannot pass two surfaces at one point), info anywhere else. Exact for two
solids of revolution on one axis (lens trains: the largest axial overlap of their spans over the common radii); any other
pair is searched on sampled boundary points of both bodies (points well outside the other body's box skipped) and
refined on the deepest ones. The bodies are optical solids only
(``bodies``): the glass between surfaces, a surface set's solid, the physical design body of a thin coated element (behind
a mirror's coated face), an aperture stop's stated plate minus its opening --
never mounts, posts or boards (lab mechanics stay in ``layout`` / ``assembly``, for a lab assembly only).

Per compound unit: ``MEMBER_OUTSIDE_ENVELOPE`` (error, ``checks.envelope_issues``) when a member reaches outside the
unit's declared envelope; ``OPTICAL_BODIES_OVERLAP`` (error, ``checks.intrusion_issues``, ``other_region``
``UNIT_ENVELOPE``) when another component's body lies inside it: the envelope is the unit's whole boundary (a telescope's
tube), only its members occupy it; touching it is allowed.

Entry point (the lowering calls it for every compiled element before building the Engine request)::

    from os_optics.optical_solids import PlacedElement, UnitEnvelope, inspect_optical_solids

    report = inspect_optical_solids(
        [PlacedElement(element, component.nominal_pose.position_mm, component.nominal_pose.quaternion_wxyz,
                       member_id=None, design_geometry=None) for element, component in ...],
        units=[UnitEnvelope(instance_id, envelope_geometry, unit_position_mm, unit_quaternion_wxyz)],  # compound units
    )
    problems.extend(report.local_run_problems())   # the ERROR findings as LocalRunProblem (stable codes, locations)

``element`` is a compiled element (``.instance_id``, ``.model``, ``.surfaces``, ``.solid``); a compound member is the
compiled member placed at its design-frame pose with its ``member_id``, ``instance_id`` its unit's component.
``inspect_lowered(lowered, project)`` does this for a ``LoweredProject``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..run_control import checkpoint
from .bodies import OpticalBody, body_of, design_solid
from .checks import (
    DEFAULT_EDGE_BAND,
    TOLERANCE_MM,
    EdgeThicknessBand,
    PlacedBody,
    contact,
    element_issues,
    envelope_issues,
    intrusion_issues,
    pair_issues,
)
from .report import ERROR, ERROR_CODES, INFO, WARNING, OpticalSolidsReport, SolidIssue

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class PlacedElement:
    """A compiled element at its design-frame pose. ``member_id``: a member of the compound unit ``instance_id`` (the
    element's own ``instance_id`` unless ``instance_id`` is given); ``design_geometry``: the physical body of an element
    whose model has no solid (a ``DesignGeometryV1``), when known."""

    element: Any
    position_mm: Vec3
    quaternion_wxyz: Quat
    member_id: str | None = None
    design_geometry: Any = None
    instance_id: str | None = None


@dataclass(frozen=True)
class UnitEnvelope:
    """A compound unit's declared envelope (``DesignGeometryV1`` in the unit frame) and the unit's design-frame pose."""

    instance_id: str
    envelope: Any
    position_mm: Vec3
    quaternion_wxyz: Quat


def inspect_optical_solids(placed: Sequence[PlacedElement], *, units: Sequence[UnitEnvelope] = (),
                           edge_band: EdgeThicknessBand = DEFAULT_EDGE_BAND) -> OpticalSolidsReport:
    """Every element check, every pair of placed bodies (members of one unit included) and every unit envelope.
    ``edge_band``: when an edge thickness is reported as implausible."""

    issues: list[SolidIssue] = []
    elements: list[dict[str, Any]] = []
    bodies: list[PlacedBody] = []
    for item in placed:
        body = body_of(item.element, member_id=item.member_id, design_geometry=item.design_geometry, instance_id=item.instance_id)
        found, gaps = element_issues(body, edge_band)
        issues += found
        entry: dict[str, Any] = {"instance_id": body.instance_id, "model_kind": body.model_kind}
        if body.member_id is not None:
            entry["member_id"] = body.member_id
        if gaps:
            entry["gaps"] = gaps
        elements.append(entry)
        bodies.append(PlacedBody(body, tuple(float(v) for v in item.position_mm), tuple(float(v) for v in item.quaternion_wxyz)))  # type: ignore[arg-type]
        checkpoint()  # (a monitored run: one body checked; os_optics.run_control)
    pairs = 0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if bodies[i].body.pieces and bodies[j].body.pieces:
                pairs += 1
                issues += pair_issues(bodies[i], bodies[j])
                checkpoint()  # (one pair of bodies checked)
    for unit in units:
        members = [b for b in bodies if b.body.instance_id == unit.instance_id and b.body.member_id is not None]
        issues += envelope_issues(unit.instance_id, unit.envelope, tuple(float(v) for v in unit.position_mm),  # type: ignore[arg-type]
                                  tuple(float(v) for v in unit.quaternion_wxyz), members)  # type: ignore[arg-type]
        others = [b for b in bodies if b.body.instance_id != unit.instance_id and b.body.pieces]
        issues += intrusion_issues(unit.instance_id, unit.envelope, tuple(float(v) for v in unit.position_mm),  # type: ignore[arg-type]
                                   tuple(float(v) for v in unit.quaternion_wxyz), others)  # type: ignore[arg-type]
        checkpoint()  # (one unit's envelope checked)
    return OpticalSolidsReport(tuple(issues), tuple(elements), pairs)


def _design_geometry(element: Any) -> Any:
    """The physical body of a compiled element: its identity's design geometry (a catalog product's, a declared model's
    stated or derived body; ``None`` for a thin element)."""

    identity = getattr(element, "identity", None)
    return getattr(identity, "design_geometry", None) if identity is not None else getattr(element, "design_geometry", None)


def _compose(position, q, local_position, local_q) -> tuple[Vec3, Quat]:
    w1, x1, y1, z1 = (float(v) for v in q)
    n = math.sqrt(w1 * w1 + x1 * x1 + y1 * y1 + z1 * z1)
    w1, x1, y1, z1 = w1 / n, x1 / n, y1 / n, z1 / n
    m = ((1 - 2 * (y1 * y1 + z1 * z1), 2 * (x1 * y1 - w1 * z1), 2 * (x1 * z1 + w1 * y1)),
         (2 * (x1 * y1 + w1 * z1), 1 - 2 * (x1 * x1 + z1 * z1), 2 * (y1 * z1 - w1 * x1)),
         (2 * (x1 * z1 - w1 * y1), 2 * (y1 * z1 + w1 * x1), 1 - 2 * (x1 * x1 + y1 * y1)))
    lp = [float(v) for v in local_position]
    p = tuple(sum(m[i][j] * lp[j] for j in range(3)) + float(position[i]) for i in range(3))
    w2, x2, y2, z2 = (float(v) for v in local_q)
    out = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
           w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    return p, out  # type: ignore[return-value]


def placed_elements(elements: Sequence[Any], poses: dict[str, Any]) -> tuple[list[PlacedElement], list[UnitEnvelope]]:
    """Every optical body of compiled elements at its design-frame pose -- a compound unit's members at the unit's pose
    composed with their member pose -- and every unit's envelope. ``poses``: instance id -> a pose (``position_mm``,
    ``quaternion_wxyz``)."""

    placed: list[PlacedElement] = []
    units: list[UnitEnvelope] = []
    for element in elements:
        pose = poses[element.instance_id]
        position, q = tuple(pose.position_mm), tuple(pose.quaternion_wxyz)
        if getattr(element, "is_compound", False):
            for member in element.members:
                mp, mq = _compose(position, q, member.member_pose.position_mm, member.member_pose.quaternion_wxyz)
                placed.append(PlacedElement(member, mp, mq, member_id=member.member_id, design_geometry=_design_geometry(member),
                                            instance_id=element.instance_id))
            envelope = _design_geometry(element)
            if envelope is not None:
                units.append(UnitEnvelope(element.instance_id, envelope, position, q))  # type: ignore[arg-type]
        else:
            placed.append(PlacedElement(element, position, q, design_geometry=_design_geometry(element)))  # type: ignore[arg-type]
    return placed, units


def inspect_elements(elements: Sequence[Any], poses: dict[str, Any], *, edge_band: EdgeThicknessBand = DEFAULT_EDGE_BAND) -> OpticalSolidsReport:
    """``inspect_optical_solids`` of compiled elements (catalog products, declared models, compound units) at ``poses``."""

    placed, units = placed_elements(elements, poses)
    return inspect_optical_solids(placed, units=units, edge_band=edge_band)


def inspect_lowered(lowered: Any, project: Any, *, edge_band: EdgeThicknessBand | None = None) -> OpticalSolidsReport:
    """``inspect_optical_solids`` of a ``LoweredProject``'s compiled elements at their Project components' nominal poses
    (the report the lowering computed, when it did and no other ``edge_band`` is asked for)."""

    if getattr(lowered, "solids", None) is not None and edge_band is None:
        return lowered.solids
    return inspect_elements(lowered.elements, {c.instance_id: c.nominal_pose for c in project.components},
                            edge_band=edge_band or DEFAULT_EDGE_BAND)


__all__ = [
    "DEFAULT_EDGE_BAND",
    "ERROR",
    "ERROR_CODES",
    "EdgeThicknessBand",
    "INFO",
    "TOLERANCE_MM",
    "WARNING",
    "OpticalBody",
    "OpticalSolidsReport",
    "PlacedBody",
    "PlacedElement",
    "SolidIssue",
    "UnitEnvelope",
    "body_of",
    "contact",
    "design_solid",
    "element_issues",
    "envelope_issues",
    "inspect_elements",
    "inspect_lowered",
    "inspect_optical_solids",
    "intrusion_issues",
    "pair_issues",
    "placed_elements",
]
