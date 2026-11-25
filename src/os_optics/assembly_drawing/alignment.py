"""Optical spacing and the alignment procedure of a drawing, derived only from the package facts.

``beam_paths`` names every point of each nominal beam path of the preview scene (``beam_reference.beam_axes``) by the
design anchor the axis names there (``through``; a scene that does not name them: the anchor at that point, when they
agree to ``ANCHOR_MATCH_MM``): the customer's light source, an OpenSci optic (named with its Customer Purchase BOM line,
e.g. ``L2 (B05)``), a customer-supplied component or an observation plane. ``spacing_rows`` are the design distances
between consecutive elements of each path (printed to 0.1 mm), measured between their optical centres -- the points the
alignment text names and the optic's position on the board refers to: an OpenSci optic's ``OPTICAL_CENTRE_DATUM``
(the body centre on the axis of an optic the beam passes through; a mirror's or splitter's coated face centre), a
customer-supplied optic the beam passes through at the centre of its reserved envelope on its axis, any other element at
its beam point. Optically identical designs therefore print identical spacings whichever way a lens is inserted. An
element the beam enters and leaves at two points is one element of the path.

``alignment_items`` is a short procedure that states nothing the design does not (for an Order OpenSci assembles, it
first says that OpenSci aligns the optics): bring each customer light source's
beam onto the design axis through two reference points on it (the source point and the first element's point, at the
stated axis height), place the optics in beam order, centre each on the beam, set the spacings to the table values, then
use the delivered adjustments -- and, for a delivered translation of an optic along the beam, that it is the adjustment
that sets the spacing to the element before it. The beam order lists optics only; the customer's own equipment on a
path (a detector, a beam dump) is named apart, where the beam reaches it.

Only ``+ - * /`` and ``sqrt`` are used (deterministic on every runtime).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .._vendor.shared_contracts.assembly_preview.scene import OPTICAL_CENTRE_DATUM, TRANSMISSIVE_ROLES
from .geometry import placement
from .model import DrawingFacts
from .steps import Words, delivered, optic_name, safe, world_axis
from .text import mm, value_text

#: A beam path point and a design anchor are the same point when every coordinate agrees to this (mm).
ANCHOR_MATCH_MM = 1e-3
#: A delivered translation moves an optic along the beam when its direction and the beam agree to this cosine.
ALONG_BEAM_COS = 0.999
#: Customer-supplied components that are equipment the beam ends at, not optics of the beam order.
EQUIPMENT_ROLES = frozenset({"detector", "dump"})
#: A unit direction within this of an axis is printed as that axis (+X, -Y ...).
AXIS_ALIGNED = 1e-9


@dataclass(frozen=True)
class PathPoint:
    instance_id: str | None  # the anchor at this point (None: no design element there)
    label: str
    position: tuple[float, float, float]  # the nominal beam point
    reference: tuple[float, float, float] | None = None  # the optical centre the spacing is measured to (None: the beam point)

    @property
    def at(self) -> tuple[float, float, float]:
        """The point the spacing table measures to."""

        return self.reference if self.reference is not None else self.position


@dataclass(frozen=True)
class NamedPath:
    path_key: str
    wavelength_nm: float | None
    points: tuple[PathPoint, ...]


@dataclass(frozen=True)
class SpacingRow:
    path_key: str
    a: PathPoint
    b: PathPoint
    distance_mm: float

    def result(self) -> dict | None:
        if self.a.instance_id is None or self.b.instance_id is None:
            return None
        return {"path_key": self.path_key, "from_instance_id": self.a.instance_id, "to_instance_id": self.b.instance_id,
                "distance_mm": mm(self.distance_mm)}


def _same(p, q) -> bool:  # noqa: ANN001
    return all(abs(p[k] - q[k]) <= ANCHOR_MATCH_MM for k in range(3))


def _unit(a, b) -> tuple[float, float, float]:  # noqa: ANN001
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    return (d[0] / n, d[1] / n, d[2] / n) if n > 0.0 else (0.0, 0.0, 0.0)


def _length(a, b) -> float:  # noqa: ANN001
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def anchor_label(facts: DrawingFacts, anchor) -> str:  # noqa: ANN001 - the vendored DesignAnchorV1
    """An anchor as the sheet names it: an OpenSci optic with its BOM line (``L2 (B05)``), else its design id."""

    if anchor.realized_by_part_ref is not None:
        return f"{safe(anchor.instance_id)} ({facts.part(anchor.realized_by_part_ref).line_ref})"
    return safe(anchor.instance_id)


def optical_centre(facts: DrawingFacts, anchor) -> tuple[float, float, float] | None:  # noqa: ANN001 - DesignAnchorV1
    """The optical centre of a design anchor in the assembly world, when the package states it: an OpenSci optic's
    ``OPTICAL_CENTRE_DATUM``; a customer-supplied optic of a transmissive role, the centre of its reserved envelope on
    its axis."""

    if anchor.realized_by_part_ref is not None:
        part = facts.part(anchor.realized_by_part_ref).part
        datum = next((d for d in part.datums if d.datum_key == OPTICAL_CENTRE_DATUM), None)
        if datum is None:
            return None
        pose = placement(tuple(part.pose_in_world.position_mm), tuple(part.pose_in_world.quaternion_wxyz))  # type: ignore[arg-type]
        return tuple(pose.apply(tuple(datum.pose_in_part.position_mm)))  # type: ignore[arg-type,return-value]
    supply = anchor.customer_supply
    if not anchor.customer_supplied or supply is None or supply.optical_role not in TRANSMISSIVE_ROLES:
        return None
    envelope = supply.reserved_envelope
    if envelope is None:
        return None
    frame = placement(tuple(anchor.position_mm), tuple(supply.orientation_wxyz))  # type: ignore[arg-type]
    return tuple(frame.apply((0.0, 0.0, (envelope.z_min_mm + envelope.z_max_mm) / 2.0)))  # type: ignore[return-value]


def beam_paths(facts: DrawingFacts, w: Words) -> list[NamedPath]:
    anchors = facts.scene.beam_reference.design_anchors
    by_id = {a.instance_id: a for a in anchors}
    out = []
    for axis in facts.scene.beam_reference.beam_axes:
        points: list[PathPoint] = []
        for k, p in enumerate(axis.points_mm):
            if axis.through is not None:
                anchor = by_id.get(axis.through[k])
            else:
                anchor = next((a for a in anchors if _same(p, a.position_mm)), None)
            if anchor is not None and points and points[-1].instance_id == anchor.instance_id:
                continue  # entered and left at two points: one element of the path (entered at the first)
            points.append(PathPoint(anchor.instance_id if anchor is not None else None,
                                    anchor_label(facts, anchor) if anchor is not None else w("spacing.end"), tuple(p),  # type: ignore[arg-type]
                                    optical_centre(facts, anchor) if anchor is not None else None))
        out.append(NamedPath(axis.path_key, axis.wavelength_nm, tuple(points)))
    return out


def spacing_rows(facts: DrawingFacts, w: Words) -> list[SpacingRow]:
    return [SpacingRow(path.path_key, a, b, _length(a.at, b.at))
            for path in beam_paths(facts, w) for a, b in zip(path.points, path.points[1:])]


def _source_anchor(facts: DrawingFacts, instance_id: str | None):  # noqa: ANN202
    anchor = next((a for a in facts.scene.beam_reference.design_anchors if a.instance_id == instance_id), None)
    if anchor is None or not anchor.customer_supplied:
        return None
    supply = anchor.customer_supply
    return anchor if supply is None or supply.optical_role == "source" else None


def alignment_items(facts: DrawingFacts, w: Words) -> list[str]:
    """The numbered alignment procedure (texts in the drawing's locale)."""

    paths = beam_paths(facts, w)
    reference = facts.scene.beam_reference
    table = facts.board.customer_table
    items: list[str] = []
    if facts.detail.detail.release.delivery_mode == "OPENSCI_ASSEMBLY":
        items.append(w("align.opensci_assembly"))
    fed = [p for p in paths if len(p.points) >= 2 and _source_anchor(facts, p.points[0].instance_id) is not None]
    if not fed:
        items.append(w("align.none"))
    height = ""
    if reference.optical_axis_height_mm is not None:
        height = w("align.height_table" if table else "align.height", h=mm(reference.optical_axis_height_mm))
    for path in fed:
        r1, r2 = path.points[0], path.points[1]
        items.append(w("align.input", source=r1.label, height=height, x1=mm(r1.position[0]), y1=mm(r1.position[1]),
                       z1=mm(r1.position[2]), x2=mm(r2.position[0]), y2=mm(r2.position[1]), z2=mm(r2.position[2]),
                       first=r2.label, direction=world_axis(_axis_direction(r1.position, r2.position))))
    equipment = {a.instance_id: a for a in reference.design_anchors
                 if a.customer_supplied and a.customer_supply is not None and a.customer_supply.optical_role in EQUIPMENT_ROLES}
    elements = {a.instance_id for a in reference.design_anchors if a.instance_id not in equipment
                and (a.realized_by_part_ref is not None or (a.customer_supplied and _source_anchor(facts, a.instance_id) is None))}
    placed: list[str] = []
    for path in paths:
        order = [p.label for p in path.points if p.instance_id in elements]
        if order:
            items.append(w("align.order", path=path.path_key, order=", ".join(order)))
        for k, point in enumerate(path.points):
            if k == 0 or point.instance_id not in equipment or point.instance_id in placed:
                continue
            placed.append(point.instance_id)
            anchor = equipment[point.instance_id]
            items.append(w("align.equipment", role=w.enum("role", anchor.customer_supply.optical_role), id=point.label,
                           previous=path.points[k - 1].label, x=mm(point.position[0]), y=mm(point.position[1]),
                           z=mm(point.position[2])))
    items.append(w("align.centre"))
    if any(len(p.points) >= 2 for p in paths):
        items.append(w("align.spacing"))
    items.append(w("align.adjust"))
    items += _along_beam_items(facts, w, paths)
    return items


def _axis_direction(a, b) -> tuple[float, float, float]:  # noqa: ANN001
    u = _unit(a, b)
    for k in range(3):  # an axis-aligned direction is printed as +X / -Y ...
        if abs(abs(u[k]) - 1.0) <= AXIS_ALIGNED:
            return tuple(1.0 if (j == k and u[k] > 0) else -1.0 if j == k else 0.0 for j in range(3))  # type: ignore[return-value]
    return tuple(float(value_text(c)) for c in u)  # type: ignore[return-value]


def _along_beam_items(facts: DrawingFacts, w: Words, paths: list[NamedPath]) -> list[str]:
    """For each delivered translation of an OpenSci optic along the beam that enters it: the spacing it sets."""

    anchors = {a.realized_by_part_ref: a for a in facts.scene.beam_reference.design_anchors if a.realized_by_part_ref}
    items = []
    for dof in facts.scene.dof_implementations:
        if not delivered(dof) or dof.axis.value not in ("Tx", "Ty", "Tz") or dof.optic_part_ref not in anchors:
            continue
        if dof.travel_min is None or dof.travel_max is None:
            continue
        anchor = anchors[dof.optic_part_ref]
        for path in paths:
            hit = next((k for k, p in enumerate(path.points) if k > 0 and p.instance_id == anchor.instance_id), None)
            if hit is None:
                continue
            a, b = path.points[hit - 1], path.points[hit]
            u = _unit(a.position, b.position)
            d = dof.world_direction
            if abs(u[0] * d[0] + u[1] * d[1] + u[2] * d[2]) < ALONG_BEAM_COS:
                continue
            impl = facts.part(dof.implemented_by_part_ref)
            items.append(w("align.adjust_along", axis=dof.axis.value, optic=optic_name(facts, dof),
                           mechanism=w.enum("mechanism", dof.mechanism.value), line=impl.line_ref, lo=mm(dof.travel_min),
                           hi=mm(dof.travel_max), a=a.label, b=b.label, d=mm(_length(a.at, b.at))))
            break
    return items
