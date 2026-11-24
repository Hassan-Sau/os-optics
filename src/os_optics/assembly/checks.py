"""The local assembly layout checks, run on a package that has ALREADY passed the canonical package verifier.

The checker never re-resolves anything: parts, poses, support tree, mates, breadboard and geometry are those of
the resolved assembly and are only measured here. Each check returns its issues and how many subjects it
evaluated; ``result.py`` turns them into the ``AssemblyLocalValidationV1`` document.

A part is *placed* when its pose is a finite unit-quaternion pose and its mesh can be read (its vertices are in the
world); it is *solid* when, in addition, its mesh is a closed, outward-oriented surface. The checks that use only
vertices (breadboard containment and fixation, the support gap, the box-apart broad phase, the vertex test of a
reserved space) need a placed part; the volume-based ones (overlap measure, solid intervals along an axis) need solid
parts. A check that needs a part that is not placed (or not solid) reports a ``*_NOT_EVALUATED`` issue (UNKNOWN) for
it instead of skipping it silently.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .._vendor.shared_contracts.assembly_preview import (
    AssemblyCheckKindV1,
    AssemblyIssueCodeV1,
    AssemblyPreviewSceneV1,
    CheckerTolerancesV1,
    ISSUE_RULES,
    MateV1,
    MetricNameV1,
    MetricV1,
    PartCategoryV1,
    PreviewPartV1,
    SupportRelationV1,
)
from .interfaces import engagement_zone, needs_zone
from .mesh import MeshError, MeshErrorCode, PartMesh, read_part_mesh
from .overlap import EMPTY, OverlapMeasure, bounds_of, box_gap, box_intersection, measure_overlap, solid_intervals_along
from .transforms import Rigid, Vec3, frame_along, gap_degrees, half_angle_sin, length, quaternion_norm, rotation_gap, sub

METRIC_DECIMALS = 6
_I = AssemblyIssueCodeV1
_MESH_CODES = {
    MeshErrorCode.UNREADABLE: AssemblyIssueCodeV1.GEOMETRY_UNREADABLE,
    MeshErrorCode.NOT_CLOSED: AssemblyIssueCodeV1.GEOMETRY_NOT_CLOSED,
    MeshErrorCode.DEGENERATE: AssemblyIssueCodeV1.GEOMETRY_DEGENERATE,
}


@dataclass(frozen=True)
class Finding:
    """One issue before it becomes an ``AssemblyIssueV1`` (the message and severity come from ``ISSUE_RULES``)."""

    code: AssemblyIssueCodeV1
    part_refs: tuple[str, ...] = ()
    mate_ref: str | None = None
    design_instance_id: str | None = None
    metrics: tuple[tuple[MetricNameV1, float], ...] = ()

    @property
    def check(self) -> AssemblyCheckKindV1:
        return ISSUE_RULES[self.code].check


@dataclass
class CheckOutcome:
    check: AssemblyCheckKindV1
    evaluated: int = 0
    findings: list[Finding] = field(default_factory=list)
    applicable: bool = True


@dataclass
class _Placed:
    part: PreviewPartV1
    pose: Rigid | None
    mesh: PartMesh | None = None
    world: tuple[Vec3, ...] | None = None  # mesh vertices in the world

    @property
    def placed(self) -> bool:
        return self.world is not None

    @property
    def solid(self) -> bool:
        return self.placed and self.mesh is not None and self.mesh.solid


def _metric(name: MetricNameV1, value: float) -> tuple[MetricNameV1, float]:
    return name, round(value, METRIC_DECIMALS) + 0.0  # + 0.0 folds -0.0


def _refs(*refs: str) -> tuple[str, ...]:
    return tuple(sorted(set(refs)))


def _pose_problem(position: Vec3, quaternion: tuple[float, float, float, float], tol: float) -> AssemblyIssueCodeV1 | None:
    if not all(math.isfinite(c) for c in (*position, *quaternion)):
        return AssemblyIssueCodeV1.NON_FINITE_POSE
    if abs(quaternion_norm(quaternion) - 1.0) > tol:
        return AssemblyIssueCodeV1.NON_UNIT_QUATERNION
    return None


class AssemblyChecker:
    """Runs every check once over one verified package's scene and geometry."""

    def __init__(self, scene: AssemblyPreviewSceneV1, geometry: dict[str, bytes], tolerances: CheckerTolerancesV1):
        self.scene = scene
        self.geometry = geometry
        self.tol = tolerances
        self.parts: dict[str, _Placed] = {}
        self.bad_datums: set[tuple[str, str]] = set()
        self.bad_mates: set[str] = set()
        self.mates = {m.mate_ref: m for m in scene.mates}
        self.board_ref = scene.breadboard.part_ref

    # ----------------------------------------------------------------------------------------- scene / geometry

    def scene_transforms(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.SCENE_TRANSFORMS)
        tol = self.tol.quaternion_norm_tolerance
        for part in sorted(self.scene.parts, key=lambda p: p.part_ref):
            pose = part.pose_in_world
            problem = _pose_problem(pose.position_mm, pose.quaternion_wxyz, tol)
            out.evaluated += 1
            problems = set()  # one issue per part and code: the pose and any datum frame of the part
            if problem is not None:
                problems.add(problem)
                self.parts[part.part_ref] = _Placed(part, None)
            else:
                self.parts[part.part_ref] = _Placed(part, Rigid.from_pose(pose.position_mm, pose.quaternion_wxyz))
            for datum in part.datums:
                out.evaluated += 1
                datum_problem = _pose_problem(datum.pose_in_part.position_mm, datum.pose_in_part.quaternion_wxyz, tol)
                if datum_problem is not None:
                    self.bad_datums.add((part.part_ref, datum.datum_key))
                    problems.add(datum_problem)
            for code in sorted(problems):
                metrics = ((_metric(MetricNameV1.QUATERNION_NORM, quaternion_norm(pose.quaternion_wxyz)),)
                           if code is problem is _I.NON_UNIT_QUATERNION else ())
                out.findings.append(Finding(code, (part.part_ref,), metrics=metrics))
        for mate in sorted(self.scene.mates, key=lambda m: m.mate_ref):
            out.evaluated += 1
            problem = _pose_problem(mate.b_in_a.position_mm, mate.b_in_a.quaternion_wxyz, tol)
            if problem is not None:
                self.bad_mates.add(mate.mate_ref)
                out.findings.append(Finding(problem, _refs(mate.part_a_ref, mate.part_b_ref), mate_ref=mate.mate_ref))
        return out

    def part_geometry(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.PART_GEOMETRY)
        declared = sum(p.geometry.triangle_count for p in self.scene.parts)
        if declared > self.tol.max_total_triangles:
            out.findings.append(Finding(_I.GEOMETRY_BUDGET_EXCEEDED, metrics=(_metric(MetricNameV1.TRIANGLE_COUNT, declared),)))
            return out
        for ref in sorted(self.parts):
            entry = self.parts[ref]
            try:
                entry.mesh = read_part_mesh(self.geometry[entry.part.geometry.asset_path], require_solid=False)
            except MeshError as exc:
                out.findings.append(Finding(_MESH_CODES[exc.code], (ref,)))
                continue
            if entry.mesh.defect is not None:  # its vertices still serve the checks that need only them
                out.findings.append(Finding(_MESH_CODES[entry.mesh.defect], (ref,)))
            else:
                out.evaluated += 1
            if entry.pose is not None:
                entry.world = entry.pose.apply_all(entry.mesh.vertices)
        return out

    @staticmethod
    def support_structure(scene: AssemblyPreviewSceneV1) -> CheckOutcome:
        """Tree rooted at the breadboard, acyclic, every support through a mate joining child and parent: the scene
        contract refuses any other package, so a verified package passes; counted, not re-derived."""

        return CheckOutcome(AssemblyCheckKindV1.SUPPORT_STRUCTURE, evaluated=sum(1 for p in scene.parts if p.support is not None))

    # ---------------------------------------------------------------------------------------------- helpers

    def _datum_frame(self, part_ref: str, datum_key: str) -> Rigid | None:
        entry = self.parts[part_ref]
        if entry.pose is None or (part_ref, datum_key) in self.bad_datums:
            return None
        datum = next(d for d in entry.part.datums if d.datum_key == datum_key)
        return entry.pose.compose(Rigid.from_pose(datum.pose_in_part.position_mm, datum.pose_in_part.quaternion_wxyz))

    def _mate_frames(self, mate: MateV1) -> tuple[Rigid, Rigid] | None:
        """World frames of datum B as the mate predicts it (A o b_in_a) and as part B places it."""

        if mate.mate_ref in self.bad_mates:
            return None
        frame_a, frame_b = self._datum_frame(mate.part_a_ref, mate.datum_a), self._datum_frame(mate.part_b_ref, mate.datum_b)
        if frame_a is None or frame_b is None:
            return None
        b_in_a = Rigid.from_pose(mate.b_in_a.position_mm, mate.b_in_a.quaternion_wxyz)
        return frame_a.compose(b_in_a), frame_b

    def _support_edges(self) -> list[tuple[str, str, MateV1, SupportRelationV1]]:
        """(child, parent, mate, relation) for every supported part, ordered by child."""

        edges = []
        for ref in sorted(self.parts):
            support = self.parts[ref].part.support
            if support is not None:
                edges.append((ref, support.parent_ref, self.mates[support.mate_ref], support.relation))
        return edges

    def _triangles(self, ref: str, frame: Rigid) -> list[tuple[Vec3, Vec3, Vec3]]:
        """The part's triangles expressed in ``frame`` (a world frame)."""

        entry = self.parts[ref]
        vertices = frame.inverse().compose(entry.pose).apply_all(entry.mesh.vertices)  # type: ignore[arg-type, union-attr]
        return [(vertices[a], vertices[b], vertices[c]) for a, b, c in entry.mesh.triangles]  # type: ignore[union-attr]

    def _measure(self, ref_a: str, ref_b: str, frame: Rigid, zone=None) -> OverlapMeasure:
        """Overlap of two solid parts in ``frame``. Columns where a part's winding does not close are left out of every
        sum (``inconsistent_columns``): the measure then counts less overlap than there is, never more."""

        assert self.parts[ref_a].solid and self.parts[ref_b].solid, "the overlap measure needs closed, outward-oriented meshes"
        tris_a, tris_b = self._triangles(ref_a, frame), self._triangles(ref_b, frame)
        box_a = bounds_of(tuple(p for t in tris_a for p in t))
        box_b = bounds_of(tuple(p for t in tris_b for p in t))
        region = box_intersection(box_a, box_b)
        if region is None:
            return EMPTY
        return measure_overlap(tris_a, tris_b, region, spacing=self.tol.sample_spacing_mm,
                               max_columns=self.tol.max_columns_per_axis, noise=self.tol.overlap_noise_mm,
                               deep=self.tol.collision_penetration_mm, zone=zone)

    def _severity(self, measure: OverlapMeasure, smaller_volume: float) -> str:
        """'none' (contact), 'minor', 'collision' or -- for a measure with skipped columns -- 'not_evaluated'.

        A collision is any of: a penetration and volume both at the collision thresholds; a deep core (overlap at least
        the collision penetration thick along all three axes) of at least the collision volume; or an overlap burying at
        least ``buried_fraction`` of the smaller part (a thin plate entirely inside a block is thin, but not minor).

        With skipped columns the deep core and the volume are lower bounds (a skipped column can only add overlap), so
        those two criteria still prove a collision; the penetration is a median, which more columns can move either
        way, and no clear or minor result can be claimed: anything else is 'not_evaluated'."""

        tol = self.tol
        if measure.deep_volume >= tol.collision_volume_mm3:
            return "collision"
        if measure.inconsistent_columns:
            return "collision" if measure.volume >= tol.buried_fraction * smaller_volume else "not_evaluated"
        if measure.penetration < tol.overlap_noise_mm or measure.volume < tol.overlap_noise_volume_mm3:
            return "none"
        if measure.penetration >= tol.collision_penetration_mm and measure.volume >= tol.collision_volume_mm3:
            return "collision"
        if measure.volume >= tol.buried_fraction * smaller_volume:
            return "collision"
        return "minor"

    def _smaller_volume(self, ref_a: str, ref_b: str) -> float:
        assert self.parts[ref_a].solid and self.parts[ref_b].solid, "a volume needs closed, outward-oriented meshes"
        return min(self.parts[ref_a].mesh.volume_mm3, self.parts[ref_b].mesh.volume_mm3)  # type: ignore[union-attr]

    @staticmethod
    def _overlap_metrics(measure: OverlapMeasure, *, engaged: bool) -> tuple[tuple[MetricNameV1, float], ...]:
        """The pair's overlap measure; a measure with skipped columns also states how many columns were cast and
        skipped (its volumes then count the evaluated columns only)."""

        metrics = [_metric(MetricNameV1.OVERLAP_VOLUME_MM3, measure.volume), _metric(MetricNameV1.PENETRATION_MM, measure.penetration),
                   _metric(MetricNameV1.DEEP_OVERLAP_VOLUME_MM3, measure.deep_volume)]
        if engaged:
            metrics.append(_metric(MetricNameV1.ENGAGED_VOLUME_MM3, measure.engaged_volume))
        if measure.inconsistent_columns:
            metrics += [_metric(MetricNameV1.COLUMNS_CAST, measure.columns),
                        _metric(MetricNameV1.COLUMNS_SKIPPED, measure.inconsistent_columns)]
        return tuple(sorted(metrics, key=lambda m: m[0].value))

    # ---------------------------------------------------------------------------------------- attachment

    def support_attachment(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.SUPPORT_ATTACHMENT)
        tol = self.tol
        minor_sin, detached_sin = half_angle_sin(tol.mate_angle_tolerance_deg), half_angle_sin(tol.detachment_angle_deg)
        for mate in sorted(self.scene.mates, key=lambda m: m.mate_ref):
            refs = _refs(mate.part_a_ref, mate.part_b_ref)
            frames = self._mate_frames(mate)
            if frames is None:
                out.findings.append(Finding(_I.ATTACHMENT_NOT_EVALUATED, refs, mate_ref=mate.mate_ref))
                continue
            out.evaluated += 1
            predicted, actual = frames
            offset = length(sub(predicted.translation, actual.translation))
            turn = rotation_gap(predicted.rotation, actual.rotation)
            metrics = tuple(sorted((_metric(MetricNameV1.ANGLE_ERROR_DEG, gap_degrees(turn)),
                                    _metric(MetricNameV1.POSITION_ERROR_MM, offset)), key=lambda m: m[0].value))
            if offset >= tol.detachment_distance_mm or turn >= detached_sin:
                out.findings.append(Finding(_I.MATE_SEPARATED, refs, mate_ref=mate.mate_ref, metrics=metrics))
            elif offset > tol.mate_position_tolerance_mm or turn > minor_sin:
                out.findings.append(Finding(_I.MATE_MISALIGNED, refs, mate_ref=mate.mate_ref, metrics=metrics))
        for child, parent, mate, _relation in self._support_edges():
            refs = _refs(child, parent)
            c, p = self.parts[child], self.parts[parent]
            if not (c.placed and p.placed):
                if mate.mate_ref not in {f.mate_ref for f in out.findings if f.code is _I.ATTACHMENT_NOT_EVALUATED}:
                    out.findings.append(Finding(_I.ATTACHMENT_NOT_EVALUATED, refs, mate_ref=mate.mate_ref))
                continue
            out.evaluated += 1
            # Gap between the child's geometry and its support's box, in the support's frame: a lower bound of
            # the true distance, so a value at the detachment tolerance proves the part floats.
            child_in_parent = p.pose.inverse().compose(c.pose).apply_all(c.mesh.vertices)  # type: ignore[union-attr, arg-type]
            gap = box_gap(bounds_of(child_in_parent), (p.mesh.bounds_min, p.mesh.bounds_max))  # type: ignore[union-attr]
            if gap >= tol.detachment_distance_mm:
                out.findings.append(Finding(_I.SUPPORT_GEOMETRY_SEPARATED, refs, mate_ref=mate.mate_ref,
                                            metrics=(_metric(MetricNameV1.GAP_MM, gap),)))
        return out

    # -------------------------------------------------------------------------------------------- the board

    def board_containment(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.BOARD_CONTAINMENT)
        board = self.scene.breadboard
        (x0, y0), (x1, y1) = board.outline_min_xy_mm, board.outline_max_xy_mm
        tol = self.tol.board_outline_tolerance_mm
        for ref in sorted(self.parts):
            if ref == self.board_ref:
                continue
            entry = self.parts[ref]
            if not entry.placed:
                out.findings.append(Finding(_I.CONTAINMENT_NOT_EVALUATED, (ref,)))
                continue
            out.evaluated += 1
            xs = [p[0] for p in entry.world]  # type: ignore[union-attr]
            ys = [p[1] for p in entry.world]  # type: ignore[union-attr]
            excursion = max(0.0, x0 - min(xs), max(xs) - x1, y0 - min(ys), max(ys) - y1)
            if excursion <= tol:
                continue
            fully_outside = min(xs) > x1 + tol or max(xs) < x0 - tol or min(ys) > y1 + tol or max(ys) < y0 - tol
            on_board = entry.part.support is not None and entry.part.support.parent_ref == self.board_ref
            code = _I.PART_OUTSIDE_BREADBOARD if (on_board or fully_outside) else _I.PART_OVERHANGS_BREADBOARD
            out.findings.append(Finding(code, (ref,), metrics=(_metric(MetricNameV1.OUTSIDE_DISTANCE_MM, excursion),)))
        for anchor in sorted(self.scene.beam_reference.design_anchors, key=lambda a: a.instance_id):
            if anchor.customer_supplied:
                out.findings.append(Finding(_I.CUSTOMER_SUPPLIED_COMPONENT, design_instance_id=anchor.instance_id))
        return out

    def board_fixation(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.BOARD_FIXATION)
        grid = self.scene.breadboard.hole_grid
        fixations = sorted((f for f in self.scene.fixations if f.board_holes), key=lambda f: f.part_ref)
        if not fixations or grid is None:
            out.applicable = False
            return out
        tol = self.tol.board_outline_tolerance_mm
        for fixation in fixations:
            entry = self.parts[fixation.part_ref]
            if not entry.placed:
                out.findings.append(Finding(_I.FIXATION_NOT_EVALUATED, (fixation.part_ref,)))
                continue
            out.evaluated += 1
            hull = convex_hull([(p[0], p[1]) for p in entry.world])  # type: ignore[union-attr]
            worst = max(distance_to_polygon((grid.origin_xy_mm[0] + i * grid.pitch_mm[0], grid.origin_xy_mm[1] + j * grid.pitch_mm[1]), hull)
                        for i, j in fixation.board_holes)
            if worst > tol:
                out.findings.append(Finding(_I.FIXATION_HOLE_NOT_UNDER_PART, (fixation.part_ref,),
                                            metrics=(_metric(MetricNameV1.HOLE_DISTANCE_MM, worst),)))
        return out

    # ----------------------------------------------------------------------------------- contact / collision

    def _expected_pairs(self) -> list[tuple[str, str, MateV1, SupportRelationV1 | None]]:
        """(child, parent, mate, relation): every support edge, then every other mate (part B treated as parent)."""

        pairs: list[tuple[str, str, MateV1, SupportRelationV1 | None]] = list(self._support_edges())
        used = {mate.mate_ref for _, _, mate, _ in pairs}
        for mate in sorted(self.scene.mates, key=lambda m: m.mate_ref):
            if mate.mate_ref not in used:
                pairs.append((mate.part_a_ref, mate.part_b_ref, mate, None))
        return pairs

    def expected_engagement(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.EXPECTED_ENGAGEMENT)
        for child, parent, mate, relation in self._expected_pairs():
            refs = _refs(child, parent)
            child_datum, parent_datum = (mate.datum_a, mate.datum_b) if mate.part_a_ref == child else (mate.datum_b, mate.datum_a)
            frame = self._datum_frame(parent, parent_datum)
            child_frame = self._datum_frame(child, child_datum)
            if frame is None or child_frame is None or not (self.parts[child].solid and self.parts[parent].solid):
                out.findings.append(Finding(_I.ENGAGEMENT_NOT_EVALUATED, refs, mate_ref=mate.mate_ref))
                continue
            child_tris = self._triangles(child, frame)
            zs = [p[2] for t in child_tris for p in t]
            zone = engagement_zone(relation, mate.interface_designation, child_datum_z=frame.inverse().apply(child_frame.translation)[2],
                                   child_z_range=(min(zs), max(zs)), tolerances=self.tol)
            measure = self._measure(child, parent, frame, zone)
            severity = self._severity(measure, self._smaller_volume(child, parent))
            if severity == "not_evaluated":
                out.findings.append(Finding(_I.ENGAGEMENT_NOT_EVALUATED, refs, mate_ref=mate.mate_ref,
                                            metrics=self._overlap_metrics(measure, engaged=zone is not None)))
                continue
            out.evaluated += 1
            if severity == "none":
                continue
            if severity == "minor":
                code = _I.ENGAGEMENT_MINOR_OVERLAP
            elif zone is None and needs_zone(relation):
                code = _I.ENGAGEMENT_NOT_BOUNDED
            else:
                code = _I.ENGAGEMENT_EXCEEDS_INTERFACE
            out.findings.append(Finding(code, refs, mate_ref=mate.mate_ref, metrics=self._overlap_metrics(measure, engaged=zone is not None)))
        return out

    def part_collision(self) -> CheckOutcome:
        out = CheckOutcome(AssemblyCheckKindV1.PART_COLLISION)
        expected = {frozenset((child, parent)) for child, parent, _, _ in self._expected_pairs()}
        world_boxes = {ref: bounds_of(e.world) for ref, e in self.parts.items() if e.placed}  # type: ignore[arg-type]
        for a, b in itertools.combinations(sorted(self.parts), 2):
            if frozenset((a, b)) in expected:
                continue
            if not (self.parts[a].placed and self.parts[b].placed):
                out.findings.append(Finding(_I.PAIR_NOT_EVALUATED, (a, b)))
                continue
            if box_intersection(world_boxes[a], world_boxes[b]) is None:
                out.evaluated += 1  # broad phase: world boxes apart
                continue
            if not (self.parts[a].solid and self.parts[b].solid):  # boxes meet: only a volume measure can tell
                out.findings.append(Finding(_I.PAIR_NOT_EVALUATED, (a, b)))
                continue
            measure = self._measure(a, b, self.parts[a].pose)  # type: ignore[arg-type]
            severity = self._severity(measure, self._smaller_volume(a, b))
            if severity == "not_evaluated":
                out.findings.append(Finding(_I.PAIR_NOT_EVALUATED, (a, b), metrics=self._overlap_metrics(measure, engaged=False)))
                continue
            out.evaluated += 1
            if severity != "none":
                code = _I.PART_INTERFERENCE if severity == "collision" else _I.MINOR_OVERLAP
                out.findings.append(Finding(code, (a, b), metrics=self._overlap_metrics(measure, engaged=False)))
        self._reserved_spaces(out)
        return out

    def _reserved_spaces(self, out: CheckOutcome) -> None:
        """No part may intrude into the space reserved for a customer-supplied component (its reserved envelope: the
        cylinder about the component's local +Z, ``customer_supply.reserved_envelope``). A part intrudes when one of its
        mesh vertices lies inside the cylinder, or when the cylinder's axis runs through the part's solid, by more than
        the overlap noise (``RESERVED_SPACE_INTRUSION``, FAIL)."""

        noise = self.tol.overlap_noise_mm
        for anchor in sorted(self.scene.beam_reference.design_anchors, key=lambda a: a.instance_id):
            supply = anchor.customer_supply
            envelope = None if supply is None else supply.reserved_envelope
            if not anchor.customer_supplied or envelope is None:
                continue
            frame = Rigid.from_pose(tuple(anchor.position_mm), tuple(supply.orientation_wxyz))  # type: ignore[arg-type]
            inverse = frame.inverse()
            radius, z0, z1 = envelope.diameter_mm / 2.0, envelope.z_min_mm, envelope.z_max_mm
            axis_start = frame.apply((0.0, 0.0, z0))
            column = frame_along(axis_start, sub(frame.apply((0.0, 0.0, z1)), axis_start))
            for ref in sorted(self.parts):
                entry = self.parts[ref]
                if not entry.placed:
                    continue
                local = inverse.apply_all(entry.world)  # type: ignore[arg-type]
                inside = any(z0 + noise < p[2] < z1 - noise and math.hypot(p[0], p[1]) < radius - noise for p in local)
                if not inside and not entry.solid:
                    continue  # the axis test needs a solid; the vertex test found nothing
                out.evaluated += 1
                if not inside and column is not None:
                    intervals, ok = solid_intervals_along(self._triangles(ref, column))
                    inside = ok and sum(max(0.0, min(hi, z1 - z0) - max(lo, 0.0)) for lo, hi in intervals) > noise
                if inside:
                    out.findings.append(Finding(_I.RESERVED_SPACE_INTRUSION, (ref,), design_instance_id=anchor.instance_id))

    # ------------------------------------------------------------------------------------------- beam axis

    def _designed_on(self, axis) -> set[str]:  # noqa: ANN001 - the vendored BeamAxisV1
        """The optical elements a beam axis crosses by design: those realizing a design anchor the axis names
        (``through``); for an axis that does not name them, those realizing a design anchor on its polyline."""

        anchors = self.scene.beam_reference.design_anchors
        if axis.through is not None:
            visited = set(axis.through)
            return {a.realized_by_part_ref for a in anchors if a.realized_by_part_ref is not None and a.instance_id in visited}
        points = tuple(tuple(p) for p in axis.points_mm)
        tol = self.tol.overlap_noise_mm
        return {a.realized_by_part_ref for a in anchors
                if a.realized_by_part_ref is not None and any(
                    point_segment_distance(tuple(a.position_mm), points[k], points[k + 1]) <= tol for k in range(len(points) - 1))}

    def beam_axis_clearance(self) -> CheckOutcome:
        """The nominal beam centre line must not run through a mechanical part, nor through an optical element it does
        not cross by design (another chain's optic: several beams on one breadboard). The optics of a beam are
        those realizing the design anchors it names (an older scene: the anchors on its polyline).

        One exact ray per axis segment of the nominal beam centre line; an obstruction is a WARNING (the beam's own
        extent comes from the user's optical run)."""

        out = CheckOutcome(AssemblyCheckKindV1.BEAM_AXIS_CLEARANCE)
        segments = [(axis.points_mm[k], axis.points_mm[k + 1], self._designed_on(axis))
                    for axis in self.scene.beam_reference.beam_axes for k in range(len(axis.points_mm) - 1)]
        if not segments:
            out.applicable = False
            return out
        optical = {ref for ref in self.parts if self.parts[ref].part.category is PartCategoryV1.OPTICAL_ELEMENT}
        blocked: dict[str, float] = {}
        not_evaluated: set[str] = set()
        for start, end, designed in segments:
            direction = sub(end, start)
            frame = frame_along(start, direction)
            if frame is None:
                continue
            out.evaluated += 1
            seg_box = bounds_of((start, end))
            span = length(direction)
            for ref in sorted(self.parts):
                if ref in optical and ref in designed:
                    continue
                entry = self.parts[ref]
                if not entry.placed:
                    not_evaluated.add(ref)
                    continue
                world_box = bounds_of(entry.world)  # type: ignore[arg-type]
                if box_gap(world_box, seg_box) > 0.0:
                    continue
                if not entry.solid:  # the axis crosses its box: only its solid can tell
                    not_evaluated.add(ref)
                    continue
                intervals, ok = solid_intervals_along(self._triangles(ref, frame))
                if not ok:
                    not_evaluated.add(ref)
                    continue
                inside = sum(max(0.0, min(hi, span) - max(lo, 0.0)) for lo, hi in intervals)
                if inside > self.tol.overlap_noise_mm:
                    blocked[ref] = blocked.get(ref, 0.0) + inside
        for ref in sorted(not_evaluated):
            out.findings.append(Finding(_I.BEAM_AXIS_NOT_EVALUATED, (ref,)))
        for ref in sorted(blocked):
            out.findings.append(Finding(_I.BEAM_AXIS_OBSTRUCTED, (ref,), metrics=(_metric(MetricNameV1.AXIS_LENGTH_IN_PART_MM, blocked[ref]),)))
        return out

    def run(self) -> list[CheckOutcome]:
        """Every check after PACKAGE_INTEGRITY, in ``ASSEMBLY_CHECK_ORDER`` (transforms and geometry come first
        because the other checks use their placements)."""

        return [
            self.scene_transforms(),
            self.part_geometry(),
            self.support_structure(self.scene),
            self.support_attachment(),
            self.board_containment(),
            self.board_fixation(),
            self.expected_engagement(),
            self.part_collision(),
            self.beam_axis_clearance(),
        ]


# ------------------------------------------------------------------------------------------- planar helpers


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull (Andrew's monotone chain); collinear points dropped."""

    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def turn(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> float:
    """Distance of ``p`` from the segment ``a``-``b`` (3-D)."""

    d = sub(b, a)
    span = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    w = sub(p, a)
    t = 0.0 if span == 0.0 else max(0.0, min(1.0, (w[0] * d[0] + w[1] * d[1] + w[2] * d[2]) / span))
    return length((w[0] - t * d[0], w[1] - t * d[1], w[2] - t * d[2]))


def distance_to_polygon(point: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    """0 inside (or on) a counter-clockwise convex polygon, else the distance to its boundary."""

    if not hull:
        return math.inf
    x, y = point
    inside = len(hull) >= 3
    best = math.inf
    for k, (ax, ay) in enumerate(hull):
        bx, by = hull[(k + 1) % len(hull)]
        ex, ey = bx - ax, by - ay
        if ex * (y - ay) - ey * (x - ax) < 0.0:
            inside = False
        span = ex * ex + ey * ey
        t = 0.0 if span == 0.0 else max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / span))
        dx, dy = ax + t * ex - x, ay + t * ey - y
        best = min(best, math.sqrt(dx * dx + dy * dy))
    return 0.0 if inside else best
