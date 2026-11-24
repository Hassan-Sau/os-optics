"""The facts one drawing shows, read from ONE verified Assembly Preview Package and its verified Assembly Detail Package.

``DrawingFacts`` holds no geometry or number of its own: identities, BOM line references, support relations, mates,
settings, fixations, manual operations, DOF delivery, assembly notes and the breadboard come from the preview scene; part
outlines are the package GLBs placed with their exact poses; dimension values are the detail verifier's measurements
from those poses. Missing required facts fail closed (``MISSING_REQUIRED_FACT``).

The breadboard is drawn from its semantic section (outline, thickness, top, hole grid) rather than its dense part
mesh; ``board_problems`` cross-checks that section against the breadboard GLB (placed by its pose) and a disagreement
refuses the drawing (``BOARD_GEOMETRY_MISMATCH``).

Every part names its Customer Purchase BOM line, except the customer's own optical table: when the scene's breadboard
is ``supplied_by = CUSTOMER`` (the customer did not buy the OpenSci breadboard), the board part is the mounting area on
that table. It has no BOM line: it gets no balloon and no legend row, and the sheet names it ``TABLE`` and asks for the
mounting area to be marked on the table.

A *support chain* (one per optical element, e.g. optic -> mount -> post -> post holder) is one subtree of the
scene's support tree standing on the breadboard: a part whose support parent is the breadboard, with everything it
carries. Chains are numbered 1..n in assembly order; they come from the support relations only, never from identities.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._vendor.shared_contracts.assembly_detail import VerifiedAssemblyDetailV1
from .._vendor.shared_contracts.assembly_preview import AssemblyPreviewPackageV1
from .._vendor.shared_contracts.assembly_preview.scene import (
    FinishV1,
    PartCategoryV1,
    PreviewPartV1,
    SupportRelationV1,
)
from ..assembly.mesh import MeshError, read_part_mesh
from .errors import AssemblyDrawingError, DrawingErrorCode
from .geometry import PosedMesh, box_mesh, load_posed_mesh, placement

#: Allowed disagreement between the semantic breadboard and its mesh (the mesh's hole markers stand
#: 0.05 mm proud of the top; float32 coordinates round at ~1e-5 mm).
BOARD_CROSS_CHECK_MM = 0.5


#: How the sheet names the customer's own optical table (it has no Customer Purchase BOM line).
CUSTOMER_TABLE_LABEL = "TABLE"


@dataclass(frozen=True)
class BoardFacts:
    part_ref: str
    x0: float
    y0: float
    x1: float
    y1: float
    top_z: float
    thickness: float
    grid_origin: tuple[float, float] | None
    grid_pitch: tuple[float, float] | None
    grid_count: tuple[int, int] | None
    thread: str | None
    customer_table: bool = False  # the customer's own optical table, not an OpenSci breadboard

    def hole_xy(self, i: int, j: int) -> tuple[float, float]:
        assert self.grid_origin is not None and self.grid_pitch is not None
        return (self.grid_origin[0] + i * self.grid_pitch[0], self.grid_origin[1] + j * self.grid_pitch[1])

    def slab(self, x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None) -> PosedMesh:
        """The board slab as a closed box, optionally cropped to a world x / y range (for a broken-out view)."""

        x0, x1 = (max(self.x0, x_range[0]), min(self.x1, x_range[1])) if x_range else (self.x0, self.x1)
        y0, y1 = (max(self.y0, y_range[0]), min(self.y1, y_range[1])) if y_range else (self.y0, self.y1)
        return box_mesh(self.part_ref, (x0, y0, self.top_z - self.thickness), (x1, y1, self.top_z))


@dataclass(frozen=True)
class PartFacts:
    part: PreviewPartV1
    line_ref: str
    mesh: PosedMesh | None  # None for the breadboard (drawn from BoardFacts)

    @property
    def part_ref(self) -> str:
        return self.part.part_ref

    @property
    def sku(self) -> str:
        return self.part.identity.opensci_sku

    @property
    def on_bom(self) -> bool:
        """The part is a Customer Purchase BOM line (every part but the customer's own optical table)."""

        return self.part.commercial_line_ref is not None

    @property
    def display_name(self) -> str:
        return self.part.identity.display_name


@dataclass(frozen=True)
class ChainFacts:
    """One support chain: a part standing on the breadboard and everything it carries (in assembly order)."""

    number: int  # 1..n in assembly order of the chains' first parts
    root_ref: str
    part_refs: tuple[str, ...]
    optic_refs: tuple[str, ...]
    instance_ids: tuple[str, ...]  # design instance ids realized by its optics (beam reference anchors)


@dataclass(frozen=True)
class DrawingFacts:
    preview: AssemblyPreviewPackageV1
    detail: VerifiedAssemblyDetailV1
    parts: tuple[PartFacts, ...]  # in assembly order
    board: BoardFacts
    chains: tuple[ChainFacts, ...] = ()

    @property
    def scene(self):  # noqa: ANN201 - the vendored scene model
        return self.preview.scene

    def part(self, part_ref: str) -> PartFacts:
        for facts in self.parts:
            if facts.part_ref == part_ref:
                return facts
        raise KeyError(part_ref)

    def line_ref(self, part_ref: str) -> str:
        return self.part(part_ref).line_ref

    def bom_parts(self) -> tuple[PartFacts, ...]:
        """Every part that is a Customer Purchase BOM line (ballooned and in the parts legend)."""

        return tuple(p for p in self.parts if p.on_bom)

    def stack(self) -> tuple[PartFacts, ...]:
        """Every part but the breadboard (the parts it carries)."""

        return tuple(p for p in self.parts if p.mesh is not None)

    def chain_of(self, part_ref: str) -> ChainFacts | None:
        return next((c for c in self.chains if part_ref in c.part_refs), None)


def support_chains(parts: tuple[PartFacts, ...], board_ref: str, anchors) -> tuple[ChainFacts, ...]:  # noqa: ANN001
    """The support chains of ``parts`` (in assembly order): every subtree standing on the breadboard ``board_ref``."""

    children: dict[str, list[str]] = {}
    for facts in parts:
        if facts.part.support is not None:
            children.setdefault(facts.part.support.parent_ref, []).append(facts.part_ref)
    order = {facts.part_ref: n for n, facts in enumerate(parts)}
    by_ref = {facts.part_ref: facts for facts in parts}
    chains = []
    for root in children.get(board_ref, []):
        members, stack = [], [root]
        while stack:
            ref = stack.pop()
            if ref in members:  # pragma: no cover - the scene contract refuses a cyclic support tree
                continue
            members.append(ref)
            stack.extend(children.get(ref, []))
        members.sort(key=lambda r: order[r])
        optics = tuple(r for r in members if by_ref[r].part.category is PartCategoryV1.OPTICAL_ELEMENT)
        ids = tuple(a.instance_id for a in anchors if a.realized_by_part_ref in optics)
        chains.append((order[members[0]], root, tuple(members), optics, ids))
    chains.sort()
    return tuple(ChainFacts(n, root, members, optics, ids) for n, (_, root, members, optics, ids) in enumerate(chains, start=1))


def _missing(problem: str) -> AssemblyDrawingError:
    return AssemblyDrawingError(DrawingErrorCode.MISSING_REQUIRED_FACT, (problem,))


def board_problems(board: BoardFacts, glb: bytes, pose_position: tuple[float, float, float],
                   pose_quaternion: tuple[float, float, float, float]) -> list[str]:
    """Disagreements between the semantic breadboard and its own mesh placed by its pose."""

    try:
        mesh = read_part_mesh(glb)
    except MeshError as exc:
        return [f"breadboard geometry is unreadable: {exc.code.value}"]
    pose = placement(pose_position, pose_quaternion)
    world = pose.apply_all(tuple(mesh.vertices[i] for i in sorted({i for t in mesh.triangles for i in t})))
    lo = tuple(min(p[k] for p in world) for k in range(3))
    hi = tuple(max(p[k] for p in world) for k in range(3))
    tol = BOARD_CROSS_CHECK_MM
    problems = []
    if abs(lo[0] - board.x0) > tol or abs(lo[1] - board.y0) > tol or abs(hi[0] - board.x1) > tol or abs(hi[1] - board.y1) > tol:
        problems.append("breadboard outline does not match the breadboard geometry")
    if abs(lo[2] - (board.top_z - board.thickness)) > tol:
        problems.append("breadboard underside (top_z_mm - thickness_mm) does not match the breadboard geometry")
    if not (board.top_z - tol <= hi[2] <= board.top_z + tol):
        problems.append("breadboard top_z_mm does not match the breadboard geometry")
    return problems


def drawing_facts(detail: VerifiedAssemblyDetailV1) -> DrawingFacts:
    """Collect and cross-check the facts of a verified detail (and the preview it was verified against)."""

    preview = detail.preview
    scene = preview.scene
    sb = scene.breadboard
    grid = sb.hole_grid
    board = BoardFacts(
        part_ref=sb.part_ref, x0=sb.outline_min_xy_mm[0], y0=sb.outline_min_xy_mm[1], x1=sb.outline_max_xy_mm[0],
        y1=sb.outline_max_xy_mm[1], top_z=sb.top_z_mm, thickness=sb.thickness_mm,
        grid_origin=tuple(grid.origin_xy_mm) if grid else None, grid_pitch=tuple(grid.pitch_mm) if grid else None,  # type: ignore[arg-type]
        grid_count=tuple(grid.count) if grid else None, thread=grid.thread_designation if grid else None,  # type: ignore[arg-type]
        customer_table=sb.supplied_by == "CUSTOMER")
    for fixation in scene.fixations:
        if grid is not None and fixation.mode.value != "MANUAL_BASE_FIXATION" and not fixation.board_holes:
            raise _missing(f"fixation of {fixation.part_ref!r} names no board hole on the gridded breadboard")
    parts = []
    for part in sorted(scene.parts, key=lambda p: (p.assembly_order, p.part_ref)):
        if part.commercial_line_ref is None and not (board.customer_table and part.part_ref == sb.part_ref):
            raise _missing(f"part {part.part_ref!r} has no Customer Purchase BOM line reference")
        if part.part_ref == sb.part_ref:
            if part.category is not PartCategoryV1.BREADBOARD:
                raise _missing("the breadboard part is not a BREADBOARD")
            problems = board_problems(board, preview.files[part.geometry.asset_path], part.pose_in_world.position_mm,
                                      part.pose_in_world.quaternion_wxyz)
            if problems:
                raise AssemblyDrawingError(DrawingErrorCode.BOARD_GEOMETRY_MISMATCH, problems)
            parts.append(PartFacts(part, part.commercial_line_ref or CUSTOMER_TABLE_LABEL, None))
            continue
        pose = placement(part.pose_in_world.position_mm, part.pose_in_world.quaternion_wxyz)
        try:
            mesh = load_posed_mesh(part.part_ref, preview.files[part.geometry.asset_path], pose,
                                   glass=part.appearance.finish is FinishV1.GLASS)
        except MeshError as exc:
            raise AssemblyDrawingError(DrawingErrorCode.PART_GEOMETRY_UNREADABLE,
                                       (f"geometry of {part.part_ref!r} is not a readable closed triangle mesh: {exc.code.value}",)) from None
        parts.append(PartFacts(part, part.commercial_line_ref, mesh))  # type: ignore[arg-type]
    chains = support_chains(tuple(parts), sb.part_ref, scene.beam_reference.design_anchors)
    return DrawingFacts(preview=preview, detail=detail, parts=tuple(parts), board=board, chains=chains)


RELATION_ACTION = {
    SupportRelationV1.STANDS_ON: "STAND_AND_FIX",
    SupportRelationV1.INSERTED_INTO: "INSERT",
    SupportRelationV1.MOUNTED_ON: "MOUNT",
    SupportRelationV1.SEATED_IN: "SEAT",
    SupportRelationV1.FASTENED_TO: "FASTEN",
}
