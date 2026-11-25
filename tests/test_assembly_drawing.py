"""Local Detailed Assembly Drawing (``os_optics.assembly_drawing``): semantic and visual acceptance on the
reference pair, determinism, pose authority, the diagnostic sheets of a FAILED or incomplete check, the fail-closed matrix, the output self-scan,
the JSON API and the CLI.

The reference pair is ``tests/fixtures/assembly_drawing`` (an Assembly Preview Package bound to a persisted
commercial resolution + its Assembly Detail Package). Every expectation is derived from those packages' own facts
(scene, detail, measured values), never from hard-coded hashes, positions or hole indices.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import time

import pytest
from assembly_helpers import apply, box, datum_world, part, set_geometry, without_last_triangles
from deep_json import UNREADABLE_DEPTH, deep_array
from drawing_helpers import (
    DETAIL_JSON,
    FORBIDDEN_TOKENS,
    PREVIEW_ZIP,
    RAISE_MM,
    SUNK_POST_Z_MM,
    detail_bytes,
    detail_doc,
    edited_preview,
    encoded_vocabulary_hits,
    enlarged_board,
    forbidden_hits,
    groups,
    metadata,
    path_bounds,
    preview,
    preview_files,
    raised_optic_stack,
    rebound_detail,
    sunk_post,
    svg_tree,
    texts,
    withheld_dofs,
)

from os_optics._vendor.shared_contracts.assembly_detail import (
    DRAWING_DESCRIPTION,
    DrawingStatusV1,
    AssemblyDrawingManifestV1,
    AssemblyDrawingResultV1,
    detail_content_hash_of,
    load_assembly_detail,
    verify_drawing_outputs,
)
from os_optics._vendor.shared_contracts.assembly_preview import (
    AssemblyPreviewPackageV1,
    content_hash_of,
    json_file_bytes,
    package_archive_bytes,
)
from os_optics.assembly import validate_assembly_layout
from os_optics.assembly_drawing import (
    DRAWING_FILES,
    AssemblyDrawingError,
    DrawingErrorCode,
    generate_assembly_drawing,
    load_assembly_detail_package,
    verify_assembly_drawing,
)
from os_optics.assembly_drawing import generate as generate_module
from os_optics.assembly_drawing.model import drawing_facts
import drawing_text_check as privacy
from os_optics.assembly_drawing import sheet as sheet_module
from os_optics.assembly_drawing import steps as steps_module
from os_optics.assembly_drawing.sheet import SheetBuilder
from os_optics.assembly_drawing.i18n import EN as EN_MESSAGES
from os_optics.assembly_drawing.steps import assembly_steps

#: The English words of every DOF mechanism (the message catalog's ``mechanism.*`` templates).
MECHANISM_WORDS = {k.split(".", 1)[1]: v for k, v in EN_MESSAGES.items() if k.startswith("mechanism.")}
from os_optics.assembly_drawing import i18n
from os_optics.assembly_drawing.generate import GENERATOR_VERSION
from os_optics.assembly_drawing.text import mm, value_text
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.manifest import engine_build_id

SVG, PDF, RESULT, MANIFEST = DRAWING_FILES
VIEW_KEYS = ("FRONT", "SIDE", "PLAN", "PICTORIAL")
SVG_PATH = "{http://www.w3.org/2000/svg}path"


@pytest.fixture(scope="module")
def ref():
    return generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)


@pytest.fixture(scope="module")
def svg(ref):
    return svg_tree(ref.files[SVG])


@pytest.fixture(scope="module")
def scene():
    return preview().scene


@pytest.fixture(scope="module")
def verified_detail():
    return load_assembly_detail(detail_bytes(), preview())


@pytest.fixture(scope="module")
def sunk():
    package = edited_preview(sunk_post)
    return package, rebound_detail(package)


@pytest.fixture(scope="module")
def unbounded():
    """The reference with the optic's seat named by a designation the checker does not know: the seat engagement is
    not bounded, so the local check is UNKNOWN (incomplete)."""

    def unknown_seat(scene):
        next(m for m in scene["mates"] if m["mate_ref"] == "mate-1")["interface_designation"] = "UNC-QX"

    package = edited_preview(unknown_seat)
    return package, rebound_detail(package)


@pytest.fixture(scope="module")
def raised():
    package = edited_preview(raised_optic_stack)
    return package, rebound_detail(package)


@pytest.fixture(scope="module")
def withheld():
    """(package, drawing) of the reference with one DOF fixed and one nominal travel."""

    package = edited_preview(withheld_dofs)
    return package, generate_assembly_drawing(package, rebound_detail(package))


def all_text(svg_root) -> str:
    return "\n".join(texts(svg_root))


def insertion_setting(scene):
    return next(s for s in scene.settings if s.kind.value == "INSERTION_DEPTH")


def the_fixation(scene):
    """The reference's one fixation, its one board hole (i, j) and that hole's world (x, y) from the board's hole grid."""

    (fixation,) = scene.fixations
    ((i, j),) = fixation.board_holes
    grid = scene.breadboard.hole_grid
    return fixation, (i, j), (grid.origin_xy_mm[0] + i * grid.pitch_mm[0], grid.origin_xy_mm[1] + j * grid.pitch_mm[1])


def world_box(scene, ref):
    """World AABB of a part's scene geometry bounds, placed by its pose."""

    p = scene.part(ref)
    pose = (p.pose_in_world.position_mm, p.pose_in_world.quaternion_wxyz)
    corners = [apply(pose, c) for c in itertools.product(*zip(p.geometry.bounds_min_mm, p.geometry.bounds_max_mm))]
    return [min(c[k] for c in corners) for k in range(3)], [max(c[k] for c in corners) for k in range(3)]


def dimension_of(result, kind: str):
    (dim,) = [d for d in result.dimensions if d.kind.value == kind]
    return dim


def dof_table_rows(root, dofs) -> dict[str, str]:
    """The DOF delivery table's text split into one row per DOF (rows follow the scene's DOF order), keyed by axis."""

    (table,) = groups(root, "dof-table")
    text = " ".join(texts(table))
    starts = [text.index(f" {d.axis.value} {d.design_instance_id} ") + 1 for d in dofs]
    assert starts == sorted(starts)
    return {d.axis.value: text[a:b].strip() for d, a, b in zip(dofs, starts, starts[1:] + [len(text)])}


def world_axis_word(direction) -> str:
    """``+X`` .. ``-Z`` of an axis-aligned unit direction."""

    (k,) = [k for k in range(3) if abs(direction[k]) > 0.5]
    return ("+" if direction[k] > 0 else "-") + "XYZ"[k]


# ============================================================================== the reference sheet


def test_reference_drawing_is_a_deliverable_detailed_assembly_drawing(ref):
    result = ref.result
    assert result.drawing_status.value == "DETAILED_ASSEMBLY_DRAWING" and result.delivery_ready is True
    assert result.validation.status.value == validate_assembly_layout(PREVIEW_ZIP).status.value == "PASS"
    assert result.validation.result_hash == validate_assembly_layout(PREVIEW_ZIP).result_hash
    assert result.description == DRAWING_DESCRIPTION and result.sheet == "A3_LANDSCAPE_420x297_MM"
    assert set(ref.files) == set(DRAWING_FILES)
    assert verify_drawing_outputs(ref.files) == () and verify_assembly_drawing(ref.files) == ()
    assert AssemblyDrawingResultV1.model_validate_json(ref.files[RESULT]) == result
    assert AssemblyDrawingManifestV1.model_validate_json(ref.files[MANIFEST]) == ref.manifest
    assert ref.files[RESULT] == json_file_bytes(result.payload())


def test_result_and_manifest_bind_the_exact_sources(ref, verified_detail):
    package = preview()
    binding = ref.result.binding
    manifest_binding = package.manifest.binding
    assert binding.preview_package_content_hash == package.content_hash
    assert binding.preview_package_revision == package.manifest.package_revision
    assert binding.detail_content_hash == verified_detail.content_hash == detail_doc()["detail_content_hash"]
    assert binding.commercial_resolution_id == manifest_binding.commercial_resolution_id
    assert binding.commercial_bom == manifest_binding.commercial_bom
    assert binding.project == manifest_binding.project
    assert binding.local_validation_hash == manifest_binding.local_validation_hash
    assert binding.release == verified_detail.detail.release
    assert ref.manifest.binding == binding and ref.manifest.drawing_result_hash == ref.result.result_hash
    assert ref.result.generator.engine_build_id == ref.manifest.generator.engine_build_id == engine_build_id()
    assert ref.result.generator.generator_version == GENERATOR_VERSION == "4.0.0"
    files = {f.path: f for f in ref.manifest.files}
    for name in (SVG, PDF, RESULT):
        assert files[name].sha256 == hashlib.sha256(ref.files[name]).hexdigest()
        assert files[name].size_bytes == len(ref.files[name])


def test_svg_metadata_binds_hashes_but_not_the_engine_build(ref, svg):
    meta = metadata(svg)
    package = preview()
    binding = package.manifest.binding
    assert meta["preview_package_content_hash"] == package.content_hash
    assert meta["detail_content_hash"] == detail_doc()["detail_content_hash"]
    assert meta["commercial_resolution_id"] == binding.commercial_resolution_id
    assert meta["commercial_bom_hash"] == binding.commercial_bom.bom_hash.digest
    assert meta["project_hash"] == binding.project.project_hash.digest
    assert meta["drawing_status"] == "DETAILED_ASSEMBLY_DRAWING" and meta["delivery_ready"] is True
    assert meta["generator_version"] == GENERATOR_VERSION and meta["validation_status"] == "PASS" and meta["locale"] == "en"
    assert meta["validation_interference"] == "NO_INTERFERENCE"
    raw = ref.files[SVG].decode("utf-8")
    assert engine_build_id() not in raw and "engine_build_id" not in raw and "runtime_kind" not in raw
    assert ref.result.validation.result_hash not in raw  # it covers the runtime kind
    assert set(meta["dimensions"]) == {d.dim_ref for d in ref.result.dimensions}


def test_all_four_views_are_drawn_and_show_the_parts(ref, svg, scene):
    views = {v.view.value: v for v in ref.result.views}
    assert tuple(views) == VIEW_KEYS
    holder, post, mount = "b04-post-holder-base", "b03-post", "b02-optic-mount"
    (optic,) = [p.part_ref for p in scene.parts if p.category.value == "OPTICAL_ELEMENT"]
    for key, view in views.items():
        assert view.visible_segment_count > 0, key
        assert set(view.part_refs) >= {holder, mount}, key
        view_group = groups(svg, "view", view=key)
        assert len(view_group) == 1
        assert sum(1 for _ in view_group[0].iter(SVG_PATH)) > 0
        drawn = {g.get("data-part-ref") for g in groups(svg, "part", view=key)}
        assert drawn <= {p.part_ref for p in scene.parts}
    for key in ("FRONT", "SIDE", "PICTORIAL"):
        assert post in views[key].part_refs, key
    # PLAN draws no hidden edges: the post, wholly under its mount seen from above, has no outline there
    (post_lo, post_hi), (mount_lo, mount_hi) = world_box(scene, post), world_box(scene, mount)
    assert all(mount_lo[k] <= post_lo[k] and post_hi[k] <= mount_hi[k] for k in (0, 1)) and post_hi[2] <= mount_lo[2]
    assert post not in views["PLAN"].part_refs
    assert not list(groups(svg, "part", view="PLAN", part_ref=post)[0].iter(SVG_PATH))
    # the optic sits inside its mount: dashed (hidden) in the elevations, seen only through the mount's bore in the
    # pictorial (ballooned there on a visible point of it)
    for key in ("FRONT", "SIDE"):
        (outline,) = groups(svg, "part", view=key, part_ref=optic)
        paths = list(outline.iter(SVG_PATH))
        assert paths and all(p.get("stroke-dasharray") for p in paths), key
    assert {v.value for b in ref.result.balloons if b.part_ref == optic for v in b.views} == {"PICTORIAL"}
    assert views["FRONT"].hidden_segment_count > 0 and views["SIDE"].hidden_segment_count > 0  # hidden lines dashed
    assert views["PLAN"].hidden_segment_count == 0 and views["PICTORIAL"].hidden_segment_count == 0
    assert views["FRONT"].scale == views["SIDE"].scale  # one scale for the elevations (third-angle arrangement)


def test_balloons_label_every_part_by_its_commercial_bom_line(ref, svg, scene):
    lines = {p.part_ref: p.commercial_line_ref for p in scene.parts}
    assert sorted(lines.values()) == ["B01", "B02", "B03", "B04", "B05"]
    assert {b.part_ref: b.commercial_line_ref for b in ref.result.balloons} == lines
    balloons = groups(svg, "balloon")
    assert {g.get("data-part-ref") for g in balloons} == set(lines)
    for g in balloons:
        assert texts(g) == [lines[g.get("data-part-ref")]] == [g.get("data-line-ref")]
    for b in ref.result.balloons:
        assert {v.value for v in b.views} == {g.get("data-view") for g in balloons if g.get("data-part-ref") == b.part_ref}


def test_balloon_leaders_end_on_visible_points_of_their_part(verified_detail):
    facts = drawing_facts(verified_detail)
    builder = SheetBuilder(facts, drawing_status=DrawingStatusV1.DETAILED_ASSEMBLY_DRAWING, validation_status="WARNING", validation_issues=[], checker_version="1.0.0",
                           steps=assembly_steps(facts), generator_version="1.0.0", validation_interference="NO_INTERFERENCE")
    builder.build()
    checked = 0
    for part_ref, anchors in builder.record.balloon_anchors.items():
        for key, anchor in anchors.items():
            vd = builder.views[key]
            first = vd.scene.first_mesh(*vd.to_uv(anchor))
            assert first is not None and vd.mesh_refs[first[0]] == part_ref, (part_ref, key)
            checked += 1
    assert checked >= len(facts.parts)


def test_every_dimension_is_drawn_with_its_measured_value_and_basis(ref, svg, verified_detail, scene):
    by_ref = {d.dim_ref: d for d in verified_detail.detail.dimensions}
    assert {d.dim_ref for d in ref.result.dimensions} == set(by_ref)
    for dim in ref.result.dimensions:
        source = by_ref[dim.dim_ref]
        measured = verified_detail.measured[dim.dim_ref].value_mm
        assert dim.value_mm == value_text(measured) and dim.basis == source.basis and dim.kind == source.kind
        (group,) = groups(svg, "dimension", dim_ref=dim.dim_ref)
        assert group.get("data-view") == dim.views[0].value and group.get("data-basis") == source.basis.value
        (label,) = texts(group)
        shown = mm(abs(measured))
        if source.basis.value == "RESOLVED":
            assert label == shown
        elif source.basis.value == "CUSTOMER_DESIGN":
            assert label.startswith(f"{shown} C")
        elif source.basis.value == "NOMINAL":
            assert label.startswith(f"{shown} N"), label  # a nominal value is always marked
        else:
            assert label.startswith(f"({shown})"), label  # a reference dimension, in parentheses
    # every value, recomputed from the scene's own facts (hole grid, poses, settings, design anchors), not by the verifier
    fixation, _, hole = the_fixation(scene)
    corner = scene.breadboard.outline_min_xy_mm
    holder = scene.part(fixation.part_ref).pose_in_world.position_mm
    (optic,) = [p for p in scene.parts if p.category.value == "OPTICAL_ELEMENT"]
    lens = optic.pose_in_world.position_mm
    # the optic's place on the board and its distance from the source refer to its optical centre (the
    # middle of the lens on its axis, the point the optical spacing is measured to), not to its first vertex
    centre, _ = datum_world(scene.payload(), optic.part_ref, "optical_centre")
    (source,) = [a.position_mm for a in scene.beam_reference.design_anchors if a.customer_supplied]
    axis = {"WORLD_X": 0, "WORLD_Y": 1, "WORLD_Z": 2}
    expected = {
        "FIXATION_HOLE_POSITION": lambda k: hole[k] - corner[k],
        "FIXATION_OFFSET": lambda k: hole[k] - holder[k],
        "INSERTION_DEPTH": lambda k: float(insertion_setting(scene).value),
        "OPTICAL_AXIS_HEIGHT": lambda k: lens[k] - scene.breadboard.top_z_mm,
        "OPTIC_BOARD_POSITION": lambda k: centre[k] - corner[k],
        "EXTERNAL_SOURCE_STANDOFF": lambda k: centre[k] - source[k],
    }
    assert {d.kind.value for d in by_ref.values()} == set(expected)
    values = {d.dim_ref: d.value_mm for d in ref.result.dimensions}
    assert values == {d.dim_ref: value_text(expected[d.kind.value](axis[d.axis.value])) for d in by_ref.values()}


def test_the_insertion_depth_reads_as_a_nominal_value(ref, svg, scene):
    setting = insertion_setting(scene)
    depth = value_text(float(setting.value))
    assert setting.basis.value == "NOMINAL"
    dim = dimension_of(ref.result, "INSERTION_DEPTH")
    assert dim.basis.value == "NOMINAL" and dim.value_mm == depth and dim.views[0].value == "FRONT"
    (group,) = groups(svg, "dimension", dim_ref=dim.dim_ref)
    (label,) = texts(group)
    assert label.startswith(f"{depth} N")
    assert all(p.get("stroke") == "#b45309" for p in group.iter(SVG_PATH) if p.get("stroke-dasharray"))
    (row,) = groups(svg, "dimension-row", dim_ref=dim.dim_ref)
    assert "NOMINAL" in texts(row) and f"{depth} NOM" in texts(row)
    (insert,) = [s for s in ref.result.assembly_steps if s.action.value == "INSERT"]
    assert f"insertion depth {depth} mm (nominal, set at assembly)" in insert.text
    assert depth not in [t.strip() for t in texts(svg)]  # the depth is always printed with its basis


def test_optical_axis_height_and_beam_axis(ref, svg, scene):
    (dim,) = [d for d in ref.result.dimensions if d.kind.value == "OPTICAL_AXIS_HEIGHT"]
    assert dim.value_mm == value_text(scene.beam_reference.optical_axis_height_mm) == "125"
    assert dim.views[0].value == "FRONT"
    for key in ("FRONT", "PLAN", "PICTORIAL", "SIDE"):
        (beam,) = groups(svg, "beam-axis", view=key)
        assert beam.get("data-basis") == "DESIGN_NOMINAL_AXIS"
    front = groups(svg, "beam-axis", view="FRONT")[0]
    assert any(p.get("stroke") == "#c8102e" for p in front.iter("{http://www.w3.org/2000/svg}path"))
    (label,) = groups(svg, "beam-label", view="FRONT")
    assert texts(label) == ["DESIGN BEAM AXIS 780 nm"] and label.get("data-path-keys") == "beam-1"
    assert "optical axis height 125 mm above the board top" in all_text(svg)


def test_fixation_hole_callout(svg, scene):
    fixation, (i, j), (hx, hy) = the_fixation(scene)
    assert fixation.part_ref == "b04-post-holder-base" and fixation.mode.value == "SLOT"
    owner, board = scene.part(fixation.part_ref), scene.part(scene.breadboard.part_ref)
    thread = fixation.fastener.thread_designation
    assert thread == "M6x1.0" and not fixation.fastener.on_commercial_bom
    # the resolution's own statement: standard fixing screws supplied with the order, not separate item lines
    fastener = f"standard {thread} screws and washers, supplied with the order (not separate item lines)"
    hole = f"{i},{j}"
    # the reference sheet: the hole symbol in PLAN and FRONT, and the fixation stated once in the notes; a callout needs a
    # free board corner in PLAN, which the compact reference board does not have, so the hole is tagged instead
    assert groups(svg, "fixation-hole", view="PLAN", hole=hole) and groups(svg, "fixation-hole", view="FRONT", hole=hole)
    notes = " ".join(texts(groups(svg, "notes")[0]))
    assert (f"BASE FIXING: {owner.commercial_line_ref} {owner.identity.opensci_sku} is fixed through its slot on BREADBOARD "
            f"{board.commercial_line_ref}, hole ({i + 1}, {j + 1}) at x {mm(hx)} y {mm(hy)}.") in notes
    assert f"Fixing screws: {fastener}." in notes
    # holes are numbered for people: (column, row) from (1, 1), stated with its origin
    assert f"Hole numbers (column, row) count from (1, 1), the hole at x {mm(scene.breadboard.hole_grid.origin_xy_mm[0])}" in notes
    assert "zero-based" not in all_text(svg)
    (tag,) = groups(svg, "fixation-tag", view="PLAN")
    assert tag.get("data-hole") == hole and tag.get("data-part-ref") == fixation.part_ref and tag.get("data-mode") == "SLOT"
    assert texts(tag) == [f"{owner.commercial_line_ref} ({i + 1}, {j + 1})"] and not groups(svg, "fixation-callout")
    assert not re.search(r"(?i)M6x1\.0\s*x\s*\d|length\s+\d|not a BOM line", all_text(svg))  # no invented length, no "buy your own"
    # the same assembly on a larger breadboard (every pose and hole kept): the PLAN callout of the hole
    package = enlarged_board()
    root = svg_tree(generate_assembly_drawing(package, rebound_detail(package)).files[SVG])
    (callout,) = groups(root, "fixation-callout", view="PLAN")
    assert callout.get("data-hole") == hole and callout.get("data-mode") == "SLOT" == fixation.mode.value
    assert callout.get("data-part-ref") == fixation.part_ref
    lines = texts(callout)
    assert lines[0] == f"FIXING HOLE ({i + 1}, {j + 1})"
    joined = " ".join(lines)
    assert f"{owner.commercial_line_ref}: base fixed through its slot at x {mm(hx)} y {mm(hy)}" in joined
    assert fastener in joined
    assert not groups(root, "fixation-tag")
    assert not re.search(r"(?i)M6x1\.0\s*x\s*\d|length\s+\d", all_text(root))


def test_the_customer_supplied_source_is_marked_as_customer_supplied(svg, scene):
    (anchor,) = [a for a in scene.beam_reference.design_anchors if a.customer_supplied]
    (group,) = groups(svg, "external-source", view="FRONT")
    assert group.get("data-instance-id") == anchor.instance_id and group.get("data-supply") == "CUSTOMER_SUPPLIED"
    (label,) = groups(svg, "customer-supplied-label", view="FRONT")
    assert texts(label) == [f"{anchor.instance_id}: YOUR LIGHT SOURCE", "supplied by you, at its design position"]
    (optic,) = [p for p in scene.parts if p.category.value == "OPTICAL_ELEMENT"]
    centre, _ = datum_world(scene.payload(), optic.part_ref, "optical_centre")  # to the optic's centre
    distance = max(abs(o - a) for o, a in zip(centre, anchor.position_mm))  # along the beam
    (standoff,) = [g for g in groups(svg, "dimension") if g.get("data-kind") == "EXTERNAL_SOURCE_STANDOFF"]
    assert texts(standoff)[0].startswith(f"{mm(distance)} C")  # C = customer design


def test_title_block_and_notices(ref, svg):
    text = all_text(svg)
    (banner,) = groups(svg, "notice-banner")
    assert banner.get("data-status") == "DETAILED_ASSEMBLY_DRAWING" and texts(banner)[0] == "DETAILED ASSEMBLY DRAWING"
    (block,) = groups(svg, "title-block")
    block_text = texts(block)
    assert "DETAILED ASSEMBLY DRAWING" in block_text
    # ONE check statement: the status and interference of the local assembly check; its counts stay in the data
    (check,) = groups(block, "check-statement")
    evaluated = len(ref.validation.checks)
    passed = sum(1 for c in ref.validation.checks if c.status.value == "PASS")
    assert texts(check) == ["ASSEMBLY CHECK", "PASS - no interference"]
    assert (check.get("data-checks-passed"), check.get("data-checks-evaluated")) == (str(passed), str(evaluated))
    assert (ref.result.validation.checks_passed, ref.result.validation.checks_evaluated) == (passed, evaluated) == (10, 10)
    assert "RESOLUTION CHECKS PASSED" not in block_text and "Resolution checks passed" not in all_text(svg)
    # the identities of the sources are data (SVG metadata, drawing result, manifest), not title-block cells
    binding = ref.result.binding
    assert not groups(block, "references") and "GENERATOR" not in block_text and "REFERENCES" not in " ".join(block_text)
    joined = " ".join(block_text)
    for digest in (binding.preview_package_content_hash, binding.detail_content_hash, binding.commercial_bom.bom_hash.digest,
                   binding.project.project_hash.digest, binding.commercial_resolution_id):
        assert digest[:12] not in joined
    meta = metadata(svg)
    assert (meta["commercial_resolution_id"], meta["commercial_bom_hash"]) == (binding.commercial_resolution_id,
                                                                            binding.commercial_bom.bom_hash.digest)
    notice = " ".join(texts(groups(svg, "drawing-notice")[0]))
    assert notice.split() == DRAWING_DESCRIPTION.split()
    for phrase in ("not a manufacturing", "proxy", "proxies", "engineering review", "engineering preview", "approximate",
                   "strict", "not supplied", "not modelled", "preview assumption", "supplier", "bom authority", "gross"):
        assert phrase not in text.lower(), phrase


def test_parts_legend_lists_the_commercial_bom_lines(ref, svg, scene):
    (legend,) = groups(svg, "parts-legend")
    assert legend.get("data-source") == "CUSTOMER_PURCHASE_BOM"
    assert legend.get("data-bom-hash") == preview().manifest.binding.commercial_bom.bom_hash.digest
    rows = groups(svg, "legend-row")
    expected = {p.commercial_line_ref: (p.part_ref, p.identity.opensci_sku) for p in scene.parts}
    assert {r.get("data-line-ref"): (r.get("data-part-ref"), r.get("data-sku")) for r in rows} == expected
    heading = " ".join(texts(legend))
    assert "Customer Purchase BOM lines" in heading and "quantities and prices" in heading
    assert not any(re.fullmatch(r"(?i)qty|quantity|price", t.strip()) for t in texts(legend))
    assert [r.commercial_line_ref for r in ref.result.parts_legend.rows] == sorted(expected)
    assert all(r.opensci_sku.startswith(("OS-OPT-", "OS-MECH-")) for r in ref.result.parts_legend.rows)


def _adjust_step_facts(step, dof, lines) -> None:
    """An ADJUST step names its DOF, the optic, the mechanism and the part carrying it, the moved part, travel and basis."""

    assert step.part_ref == dof.implemented_by_part_ref and step.commercial_line_ref == lines[dof.implemented_by_part_ref]
    control = {"MANUAL": "manual", "NUMERICAL": "motorized"}[dof.delivered_control.value]
    assert (f"{control} adjustment {dof.axis.value} of {dof.design_instance_id} ({lines[dof.optic_part_ref]}) along world "
            f"{world_axis_word(dof.world_direction)} with the {MECHANISM_WORDS[dof.mechanism.value]} of "
            f"{lines[dof.implemented_by_part_ref]}") in step.text
    if dof.moved_part_ref != dof.implemented_by_part_ref:
        assert f"it moves {lines[dof.moved_part_ref]} with everything it carries" in step.text
    nominal = " (nominal)" if dof.travel_basis.value == "NOMINAL" else ""
    assert f"travel {mm(dof.travel_min)} to +{mm(dof.travel_max)} mm{nominal}" in step.text
    as_manual = "; the design specifies motorized adjustment"
    assert (as_manual in step.text) == (dof.status.value == "DELIVERED_AS_MANUAL")


def test_assembly_steps_follow_the_resolved_assembly(ref, scene, withheld):
    steps = ref.result.assembly_steps
    delivered = [d for d in scene.dof_implementations if d.status.value != "NOT_DELIVERED"]
    assert delivered == list(scene.dof_implementations)  # the reference delivers every retained DOF
    assert [s.action.value for s in steps] == (["PLACE", "STAND_AND_FIX", "INSERT", "MOUNT", "SEAT"] + ["ADJUST"] * len(delivered)
                                               + ["PLACE_CUSTOMER_SUPPLIED"])
    ordered = sorted(scene.parts, key=lambda p: p.assembly_order)
    assert [s.part_ref for s in steps[:5]] == [p.part_ref for p in ordered]
    assert [s.commercial_line_ref for s in steps[:5]] == [p.commercial_line_ref for p in ordered]
    mates = {m.mate_ref: m for m in scene.mates}
    for step, p in zip(steps[1:5], ordered[1:]):
        assert p.identity.opensci_sku in step.text
        assert f"(interface {mates[p.support.mate_ref].interface_designation})" in step.text
        parent = scene.part(p.support.parent_ref)
        assert parent.commercial_line_ref in step.text
    _, (i, j), (hx, hy) = the_fixation(scene)
    assert f"fix it through its slot at hole ({i + 1}, {j + 1}) (x {mm(hx)} y {mm(hy)})" in steps[1].text
    assert "fixing screws supplied (NOTES)" in steps[1].text
    # one ADJUST step per delivered DOF, in scene order: the mount's own translation adjusters and the base slot
    assert {d.mechanism.value for d in delivered} >= {"TRANSLATION_MOUNT", "BASE_SLOT_TRAVEL"}
    lines = {p.part_ref: p.commercial_line_ref for p in scene.parts}
    for step, dof in zip(steps[5:-1], delivered, strict=True):
        _adjust_step_facts(step, dof, lines)
    # a fixed DOF gets no step (nothing pretends to adjust it); a nominal travel says so
    package, drawing = withheld
    dofs = package.scene.dof_implementations
    (undelivered,) = [d for d in dofs if d.status.value == "NOT_DELIVERED"]
    still = [d for d in dofs if d.status.value != "NOT_DELIVERED"]
    adjust = [s for s in drawing.result.assembly_steps if s.action.value == "ADJUST"]
    assert len(adjust) == len(still) and any(d.travel_basis.value == "NOMINAL" for d in still)
    for step, dof in zip(adjust, still, strict=True):
        _adjust_step_facts(step, dof, lines)
    assert all(undelivered.axis.value not in s.text for s in drawing.result.assembly_steps)


def test_dof_delivery_and_notes(ref, svg, scene, withheld):
    lines = {p.part_ref: p.commercial_line_ref for p in scene.parts}

    def expected_row(dof) -> str:
        label = f"{dof.axis.value} {dof.design_instance_id}"
        requested = {"MANUAL": "manual", "NUMERICAL": "motorized"}[dof.requested_control.value]
        if dof.status.value == "NOT_DELIVERED":
            return f"{label} FIXED: design: {requested}; fixed in this assembly"
        delivery = dof.delivered_control.value
        if dof.status.value == "DELIVERED_AS_MANUAL":
            delivery += f" (design: {requested})"
        nominal = " (nominal)" if dof.travel_basis.value == "NOMINAL" else ""
        return (f"{label} {delivery}: {MECHANISM_WORDS[dof.mechanism.value]} of {lines[dof.implemented_by_part_ref]}, along world "
                f"{world_axis_word(dof.world_direction)}, travel {mm(dof.travel_min)} to +{mm(dof.travel_max)} mm{nominal}")

    rows = dof_table_rows(svg, scene.dof_implementations)
    assert rows == {d.axis.value: expected_row(d) for d in scene.dof_implementations}
    assert "MANUAL (design: motorized)" in " ".join(rows.values()) and "FIXED" not in " ".join(rows.values())
    assert "translation adjusters of the mount" in " ".join(rows.values())
    # the DOF delivery cases the reference does not have: a fixed DOF, a nominal travel
    package, drawing = withheld
    root = svg_tree(drawing.files[SVG])
    rows = dof_table_rows(root, package.scene.dof_implementations)
    assert rows == {d.axis.value: expected_row(d) for d in package.scene.dof_implementations}
    assert "FIXED" in " ".join(rows.values()) and "(nominal)" in " ".join(rows.values())

    def notes_follow_the_scene(source, result, root_) -> None:
        codes = list(dict.fromkeys(d.code.value for d in source.notes))  # first occurrence order
        assert [c.value for c in result.notes] == codes
        notes = " ".join(texts(groups(root_, "notes")[0]))
        for code in codes:
            assert i18n.CATALOG.enum("en", "note_code", code) in notes

    notes_follow_the_scene(scene, ref.result, svg)
    notes_follow_the_scene(package.scene, drawing.result, root)
    assert "DOF_FIXED" in {c.value for c in drawing.result.notes}
    assert "DOF_FIXED" not in {c.value for c in ref.result.notes}


def pdf_literal(text: str) -> bytes:
    """``text`` as a PDF literal string (ASCII; backslash and parentheses escaped)."""

    return b"(" + re.sub(r"([\\()])", r"\\\1", text).encode("ascii") + b")"


def test_pdf_is_the_same_sheet(ref, svg, scene):
    pdf = ref.files[PDF]
    assert pdf.startswith(b"%PDF-1.4\n") and pdf.endswith(b"%%EOF\n")
    depth = mm(float(insertion_setting(scene).value))
    _, (i, j), _ = the_fixation(scene)
    hole_lines = [t for t in texts(svg) if f"({i + 1}, {j + 1})" in t]  # the notes' fixing sentence, the steps
    assert hole_lines
    for needle in (b"(DETAILED ASSEMBLY DRAWING)", b"(B03)", pdf_literal(f"{depth} NOMINAL"), *map(pdf_literal, hole_lines)):
        assert needle in pdf, needle
    xref = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    entries = pdf[xref:].split(b"\n")[3:10]
    for number, entry in enumerate(entries, start=1):
        offset = int(entry.split()[0])
        assert pdf[offset:].startswith(f"{number} 0 obj".encode()), number
    assert b"CreationDate" not in pdf and b"/ID" not in pdf


# ============================================================================== determinism


def test_generation_is_deterministic_for_every_source_form(ref):
    package = preview()
    again = generate_assembly_drawing(PREVIEW_ZIP.read_bytes(), detail_bytes())
    from_map = generate_assembly_drawing(preview_files(), load_assembly_detail(detail_bytes(), package))
    from_package = generate_assembly_drawing(package, DETAIL_JSON)
    for other in (again, from_map, from_package):
        assert dict(other.files) == dict(ref.files)


def test_svg_and_pdf_bytes_do_not_depend_on_the_engine_build(ref, monkeypatch):
    monkeypatch.setattr(generate_module, "engine_build_id", lambda: "0.0.0+0000000000000000")
    other = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)
    assert other.files[SVG] == ref.files[SVG] and other.files[PDF] == ref.files[PDF]
    assert other.files[RESULT] != ref.files[RESULT] and other.result.generator.engine_build_id == "0.0.0+0000000000000000"


def test_reference_generation_time_is_practical():
    started = time.perf_counter()
    generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)
    assert time.perf_counter() - started < 8.0  # ~1.4 s on the development machine, validation included


# ============================================================================== privacy


def test_no_output_carries_forbidden_vocabulary(ref, sunk):
    """The supplier denylist scan runs HERE (tests are not shipped); so does the positive allow-list (the generation
    hook of conftest.py runs it on every drawing a test generates)."""

    diag = generate_assembly_drawing(*sunk, diagnostic=True)
    for drawing in (ref, diag):
        for name, data in drawing.files.items():
            text = data.decode("latin-1" if name == PDF else "utf-8")
            assert forbidden_hits(text) == [], name
        skus = set(re.findall(r"\bOS-[A-Z0-9]+(?:-[A-Z0-9]+)*", drawing.files[SVG].decode()))
        assert skus and all(s.startswith(("OS-OPT-", "OS-MECH-")) for s in skus)


def test_every_output_token_comes_from_the_inputs_the_templates_or_is_a_number(ref, sunk, verified_detail):
    generator = ref.result.generator.model_dump(mode="json")
    allowed = privacy.allowed_vocabulary(verified_detail, ref.validation, (str(v) for v in generator.values()))
    assert privacy.unexpected_text(ref.files, allowed) == []
    inputs = privacy.input_vocabulary(verified_detail, ref.validation)
    seen = set().union(*(privacy.tokens(text) for _, text in privacy.output_strings(ref.files)))
    assert {"b03", "m6x1", "tx"} <= seen & inputs  # identities, designations, enum values of the inputs
    # the reference optic's generated public identity (read from the fixture: the catalog's SKUs are never pinned)
    optic_words = privacy.tokens(verified_detail.preview.scene.part("b01-optic").identity.opensci_sku) - {"os", "opt"}
    assert optic_words and optic_words <= seen & inputs
    assert {"breadboard", "fixation", "nominal", "resolved"} <= seen & privacy.template_vocabulary()
    for strings in (privacy.output_strings(ref.files), privacy.output_strings(generate_assembly_drawing(*sunk, diagnostic=True).files)):
        assert all(where.split(":")[0] in DRAWING_FILES for where, _ in strings)
    pieces = {where.split(": ", 1)[1].split(" ")[0] for where, _ in privacy.output_strings(ref.files) if where.startswith(SVG)}
    assert {"<title>", "<desc>", "<metadata>", "<text>", "<g>"} <= pieces


def test_the_number_and_geometry_grammars_are_narrow():
    for token in ("0", "137", "20x30", "a2d48ab308655948b63b47649269476e13bb409c8acf40cd7375665736892abc"):
        assert privacy.is_number(token), token
    for token in ("xq412", "75a", "a2d48ab3086559", "x30", "20x", "0x1f"):
        assert not privacy.is_number(token), token
    svg = (b'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="420mm">\n'
           b'<path d="M1 2 L3 4 Z"/>\n<text x="1" y="2" transform="rotate(-90 1 2)">B03 &amp; B04</text>\n'
           b'<path d="M1 2 zqxjv"/>\n<text x="1">a<tspan>b</tspan></text>\n</svg>\n')
    strings = list(privacy._svg_strings(svg))
    assert [where for where, text in strings if text == privacy.REJECTED] == [
        "<path> d (not geometry)", "line 6 (not the generator's SVG layout)"]
    assert ("<text> text", "B03 & B04") in strings
    rejected = [where for where, text in privacy._svg_strings(b"<svg/>") if text == privacy.REJECTED]
    assert rejected == ["file (not the generator's SVG layout)"]


def _unexpected_after_injection() -> list[str]:
    """Generate the reference with an injected token (the generation hook off) and run the build-time check on it:
    its problems name the file and the place, never the token."""

    with privacy.unchecked():
        drawing = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)
    generator = drawing.result.generator.model_dump(mode="json").values()
    problems = privacy.drawing_problems(drawing.files, load_assembly_detail(detail_bytes(), preview()), drawing.validation,
                                        (str(v) for v in generator))
    assert problems and all("zqxjv" not in p.lower() for p in problems)  # the offending token is never echoed
    return problems


def test_a_token_injected_into_a_template_is_reported(monkeypatch):
    monkeypatch.setitem(i18n.EN, "title", i18n.EN["title"] + " zqxjv")
    problems = _unexpected_after_injection()
    assert any(p.startswith(f"{SVG}: <title>") for p in problems) and any(p.startswith(f"{PDF}: file") for p in problems)


def test_a_token_injected_into_a_template_table_is_reported(monkeypatch):
    for key in [k for k in i18n.EN if k.startswith("mechanism.")]:
        monkeypatch.setitem(i18n.EN, key, "zqxjv mechanism")
    problems = _unexpected_after_injection()
    assert any(p.startswith(f"{RESULT}: assembly_steps") for p in problems) and any(p.startswith(f"{SVG}: <text>") for p in problems)


def test_the_generation_hook_fails_a_test_whose_drawing_carries_an_unexpected_token(monkeypatch):
    monkeypatch.setitem(i18n.EN, "title", i18n.EN["title"] + " zqxjv")
    with pytest.raises(AssertionError, match="<title>"):
        generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)


def test_a_token_injected_into_the_canvas_is_reported(monkeypatch):
    original = SheetBuilder.build

    def build(self):
        canvas = original(self)
        canvas.text((10.0, 10.0), "OS-OPT-ZQXJV-1", 2.0)
        return canvas

    monkeypatch.setattr(sheet_module.SheetBuilder, "build", build)
    problems = _unexpected_after_injection()
    assert [p.split(":")[0] for p in problems] == [PDF, SVG]


def test_a_token_injected_into_a_group_attribute_or_the_metadata_is_reported(monkeypatch):
    monkeypatch.setattr(generate_module, "SVG_METADATA_SCHEMA", "opensci.zqxjv")
    assert any(p.startswith(f"{SVG}: <metadata> text") for p in _unexpected_after_injection())


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
@pytest.mark.parametrize("fixture", ["reference", "customer_supplied", "two_lenses", "four_lenses", "multichain", "sunk",
                                     "unbounded"])
def test_every_drawing_fixture_passes_the_text_gate(fixture, locale, sunk, unbounded):
    """The build-time gate over every drawing fixture (and both diagnostic statuses), in both locales: every output
    token comes from the verified inputs, the templates or is a number, and the output set verifies."""

    from drawing_helpers import FIXTURES, MULTI_CHAIN, multi_chain_preview, projected_detail, replicated_chains

    if fixture == "sunk":
        package, detail = sunk
    elif fixture == "unbounded":
        package, detail = unbounded
    elif fixture == "multichain":
        package = replicated_chains(multi_chain_preview("four_lenses"), 2)  # several chains on a larger sheet
        detail = projected_detail(package, MULTI_CHAIN["four_lenses"][1].read_bytes())
    else:
        package = (FIXTURES / f"{fixture}.assembly-preview.zip").read_bytes()
        detail = (FIXTURES / f"{fixture}.assembly-detail.json").read_bytes()
    with privacy.unchecked():
        drawing = generate_assembly_drawing(package, detail, locale=locale, diagnostic=fixture in ("sunk", "unbounded"))
    verified = load_assembly_detail_package(detail, package)
    generator = drawing.result.generator.model_dump(mode="json").values()
    assert privacy.drawing_problems(drawing.files, verified, drawing.validation, (str(v) for v in generator)) == []


def test_the_client_check_is_bounded_by_the_inputs_not_by_a_denylist():
    """Documents the limit: a word in a verified PUBLIC input field is carried (here a neutral, unknown word). A
    private token the server wrongly put into a public field is stopped server-side (a projection leak
    guards), not by the client, which carries no supplier vocabulary."""

    package = edited_preview(lambda scene: part(scene, "b02-optic-mount")["identity"].update(display_name="Mount zqxjv grade"))
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    assert "Mount zqxjv grade" in all_text(svg_tree(drawing.files[SVG]))


SOURCE_DIR = generate_module.__file__.rsplit("generate.py", 1)[0]


def test_the_generator_source_carries_no_supplier_vocabulary_in_any_form():
    """No supplier token in the shipped generator: not as text, not hashed (SHA-256 / SHA-1 / MD5 of its lowercase,
    upper-case and separator-stripped forms), not as a character-class regex shape; and no 64-hex literal at all."""

    from pathlib import Path

    sources = {p.name: p.read_text(encoding="utf-8") for p in Path(SOURCE_DIR).glob("*.py")}
    assert {"generate.py", "sheet.py"} <= set(sources) and "privacy.py" not in sources  # the text gate is a test helper
    assert {name: encoded_vocabulary_hits(text) for name, text in sources.items()} == {name: [] for name in sources}


def test_the_encoded_vocabulary_scan_finds_hashed_and_shaped_tokens():
    """The scan above would have caught the old hashed self-scan (digest + character-class shape per token)."""

    for token in FORBIDDEN_TOKENS:
        flat_form = re.sub(r"[^0-9a-z]+", "", token.lower())
        assert f"hashed:{token}" in encoded_vocabulary_hits(f'X = "{hashlib.sha256(flat_form.encode()).hexdigest()}"'), token
        assert f"hashed:{token}" in encoded_vocabulary_hits(hashlib.md5(token.lower().encode()).hexdigest())
    assert "shape:ZK-75A" in encoded_vocabulary_hits('(FLAT, r"[a-z]{2}[0-9]{2}[a-z]", ...)')
    assert "shape:QB-3" in encoded_vocabulary_hits(r'(LITERAL, r"[a-z]{2}\-[0-9]", ...)')
    assert encoded_vocabulary_hits("OS-OPT-ACHROMAT b03-post M6x1.0 [A-Za-z0-9]+") == []


def test_the_title_block_states_the_release_and_the_bom_as_facts(svg, ref):
    """The title block states the release basis in customer words (the Customer Purchase BOM digest is SVG metadata); no
    identity state, release policy or projector identity is drawn or recorded."""

    text = all_text(svg)
    (block,) = groups(svg, "title-block")
    block_text = texts(block)
    assert "RELEASE" in block_text and "released by OpenSci for this design" in block_text
    assert not groups(block, "order-number")  # a direct release names no Order Number and no customer
    assert metadata(svg)["commercial_bom_hash"] == ref.result.binding.commercial_bom.bom_hash.digest
    for word in ("identity_state", "COMMERCIAL)", "policy", "opensci.commercial.", "projector", "pilot", "demo"):
        assert word.lower() not in text.lower(), word
    result = ref.result.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert result["binding"]["release"] == {"basis": "DIRECT_RELEASE"}
    assert result["parts_legend"]["source"] == "CUSTOMER_PURCHASE_BOM"
    assert set(result["binding"]["commercial_bom"]) == {"bom_hash"}
    assert "DEMO_IDENTITY_NOT_DRAWABLE" not in DrawingErrorCode.__members__


# ============================================================================== pose authority


def test_a_moved_part_moves_its_outline_balloon_and_dimensions(ref, svg, raised):
    package, detail = raised
    moved = generate_assembly_drawing(package, detail)
    other = svg_tree(moved.files[SVG])

    def paper_scale(drawing) -> float:
        label = {v.view.value: v.scale for v in drawing.result.views}["FRONT"]
        a, b = (float(x) for x in label.split(":"))
        return a / b

    def relative(root, drawing, ref_part):  # world offset of a part's FRONT outline from the (unmoved) base's
        s = paper_scale(drawing)
        (g,) = groups(root, "part", view="FRONT", part_ref=ref_part)
        (base,) = groups(root, "part", view="FRONT", part_ref="b04-post-holder-base")
        (x0, y0, _, _), (bx0, by0, _, by1) = path_bounds(g), path_bounds(base)
        return (x0 - bx0) / s, (by1 - y0) / s

    for ref_part in ("b03-post", "b02-optic-mount"):
        (ax, ay), (bx, by) = relative(svg, ref, ref_part), relative(other, moved, ref_part)
        assert abs(ax - bx) < 0.05 and abs((by - ay) - RAISE_MM) < 0.05, (ref_part, ay, by)
    anchor_a = next(g.get("data-anchor") for g in groups(svg, "balloon", view="FRONT", part_ref="b03-post"))
    anchor_b = next(g.get("data-anchor") for g in groups(other, "balloon", view="FRONT", part_ref="b03-post"))
    assert anchor_a != anchor_b
    scene = preview().scene
    depth = value_text(float(insertion_setting(scene).value) - RAISE_MM)
    assert dimension_of(moved.result, "INSERTION_DEPTH").value_mm == depth
    assert dimension_of(moved.result, "OPTICAL_AXIS_HEIGHT").value_mm == value_text(scene.beam_reference.optical_axis_height_mm + RAISE_MM)
    holes = [(d.axis, d.value_mm) for d in ref.result.dimensions if d.kind.value == "FIXATION_HOLE_POSITION"]
    assert holes and [(d.axis, d.value_mm) for d in moved.result.dimensions if d.kind.value == "FIXATION_HOLE_POSITION"] == holes  # unmoved
    (label,) = texts(groups(other, "dimension", dim_ref=dimension_of(moved.result, "INSERTION_DEPTH").dim_ref)[0])
    assert label.startswith(f"{depth} N")  # the basis tag survives any scale (N = NOMINAL)
    assert moved.result.binding.preview_package_content_hash == package.content_hash != ref.result.binding.preview_package_content_hash


def test_a_detail_not_rebound_to_the_moved_preview_is_refused(raised):
    package, _ = raised
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, DETAIL_JSON)
    assert caught.value.code is DrawingErrorCode.DETAIL_BINDING_MISMATCH


# ============================================================================== FAIL / diagnostic / inconclusive


def test_a_failed_validation_is_refused_by_default(sunk):
    package, detail = sunk
    assert validate_assembly_layout(package).status.value == "FAIL"
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, detail)
    assert caught.value.code is DrawingErrorCode.DRAWING_REFUSED_VALIDATION_FAILED
    assert "request diagnostic mode" in caught.value.problems[0]
    assert any("PART_INTERFERENCE" in t or "MATE_SEPARATED" in t for t in caught.value.problems[1:])
    # asked for, the diagnostic sheet lists the same issues
    drawing = generate_assembly_drawing(package, detail, diagnostic=True)
    assert drawing.result.drawing_status is DrawingStatusV1.DIAGNOSTIC_VALIDATION_FAILED and not drawing.delivery_ready
    root = svg_tree(drawing.files[SVG])
    (issues,) = groups(root, "diagnostic-issues")
    assert any("PART_INTERFERENCE" in t or "MATE_SEPARATED" in t for t in texts(issues))
    meta = metadata(root)
    assert meta["validation_status"] == "FAIL" and meta["validation_interference"] == drawing.validation.summary.interference.value
    assert meta["validation_checks_not_passed"] == [c.check.value for c in drawing.validation.checks
                                                     if c.status.value not in ("PASS", "NOT_APPLICABLE")]


def test_the_drawing_contract_states_the_refusal_and_the_diagnostic_sheet_on_request():
    """The contract module (shared and the copy the Engine ships, read through help()) says what the generator does: a check
    that did not pass is refused unless a diagnostic sheet is asked for, which the status then selects."""
    import importlib

    for name in ("os_optics._vendor.shared_contracts.assembly_detail.drawing", "shared_contracts.assembly_detail.drawing"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        text = " ".join((module.__doc__ or "").split())
        for part in ("FAIL with ``DRAWING_REFUSED_VALIDATION_FAILED``",
                     "UNKNOWN / OUT_OF_MODEL_SCOPE with ``DRAWING_REFUSED_VALIDATION_INCONCLUSIVE``",
                     "unless the caller asks for a diagnostic sheet"):
            assert part in text, (name, part)
        assert {code.value for code in DrawingErrorCode} >= {"DRAWING_REFUSED_VALIDATION_FAILED", "DRAWING_REFUSED_VALIDATION_INCONCLUSIVE"}


def test_a_failed_package_draws_as_not_deliverable(sunk):
    package, detail = sunk
    drawing = generate_assembly_drawing(package, detail, diagnostic=True)
    assert drawing.result.drawing_status.value == "DIAGNOSTIC_VALIDATION_FAILED" and drawing.result.delivery_ready is False
    assert drawing.manifest.delivery_ready is False and drawing.result.validation.status.value == "FAIL"
    assert verify_assembly_drawing(drawing.files) == ()
    root = svg_tree(drawing.files[SVG])
    (banner,) = groups(root, "notice-banner")
    assert banner.get("data-status") == "DIAGNOSTIC_VALIDATION_FAILED"
    assert texts(banner)[0] == "DIAGNOSTIC - LOCAL ASSEMBLY CHECK FAILED"
    assert groups(root, "diagnostic-watermark") and groups(root, "diagnostic-issues")
    assert "DIAGNOSTIC - LOCAL ASSEMBLY CHECK FAILED" in all_text(root)
    (check,) = groups(root, "check-statement")
    assert texts(check)[-1].startswith("FAILED: ") and texts(check)[-1].endswith("NOT FOR DELIVERY")
    assert metadata(root)["drawing_status"] == "DIAGNOSTIC_VALIDATION_FAILED" and metadata(root)["delivery_ready"] is False
    assert b"(DIAGNOSTIC - LOCAL ASSEMBLY CHECK FAILED)" in drawing.files[PDF]
    bore_top = datum_world(package.scene.payload(), "b04-post-holder-base", "bore_top")[0][2]
    assert datum_world(package.scene.payload(), "b03-post", "bottom")[0][2] == SUNK_POST_Z_MM
    # measured from the sunk pose (bore top down to the sunk post bottom), not taken from the reference
    depth = dimension_of(drawing.result, "INSERTION_DEPTH").value_mm
    assert depth == value_text(bore_top - SUNK_POST_Z_MM) != value_text(float(insertion_setting(preview().scene).value))


def test_an_inconclusive_validation_is_refused_by_default_and_drawn_incomplete_on_request(unbounded, ref):
    package, detail = unbounded
    validation = validate_assembly_layout(package)
    assert validation.status.value == "UNKNOWN"
    assert [i.code.value for i in validation.issues if i.code.value != "CUSTOMER_SUPPLIED_COMPONENT"] == ["ENGAGEMENT_NOT_BOUNDED"]
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, detail)
    assert caught.value.code is DrawingErrorCode.DRAWING_REFUSED_VALIDATION_INCONCLUSIVE
    assert "is UNKNOWN" in caught.value.problems[0] and "ENGAGEMENT_NOT_BOUNDED: b01-optic, b02-optic-mount" in caught.value.problems
    drawing = generate_assembly_drawing(package, detail, diagnostic=True)
    result = drawing.result
    assert result.drawing_status is DrawingStatusV1.DIAGNOSTIC_VALIDATION_INCOMPLETE and result.delivery_ready is False
    assert drawing.manifest.drawing_status is DrawingStatusV1.DIAGNOSTIC_VALIDATION_INCOMPLETE and drawing.manifest.delivery_ready is False
    assert result.validation.status.value == "UNKNOWN" and verify_assembly_drawing(drawing.files) == ()
    root = svg_tree(drawing.files[SVG])
    (banner,) = groups(root, "notice-banner")
    assert banner.get("data-status") == "DIAGNOSTIC_VALIDATION_INCOMPLETE"
    assert texts(banner)[0] == "DIAGNOSTIC - LOCAL ASSEMBLY CHECK INCOMPLETE"
    assert groups(root, "diagnostic-watermark")
    (issues,) = groups(root, "diagnostic-issues")
    assert "ENGAGEMENT_NOT_BOUNDED: b01-optic, b02-optic-mount" in texts(issues)
    (check,) = groups(root, "check-statement")
    assert check.get("data-status") == "UNKNOWN"
    assert texts(check)[-1].startswith("INCOMPLETE: ") and texts(check)[-1].endswith("NOT FOR DELIVERY")
    meta = metadata(root)
    assert meta["drawing_status"] == "DIAGNOSTIC_VALIDATION_INCOMPLETE" and meta["delivery_ready"] is False
    assert meta["validation_interference"] == "NOT_DETERMINED" and meta["validation_checks_not_passed"] == ["EXPECTED_ENGAGEMENT"]
    assert b"(DIAGNOSTIC - LOCAL ASSEMBLY CHECK INCOMPLETE)" in drawing.files[PDF]
    # the drawing content is the reference's: same views, dimensions, balloons and legend (the seat step names the
    # edited designation)
    for field in ("views", "dimensions", "balloons", "parts_legend"):
        assert getattr(result, field) == getattr(ref.result, field), field
    zh = generate_assembly_drawing(package, detail, locale="zh-CN", diagnostic=True)
    assert texts(groups(svg_tree(zh.files[SVG]), "notice-banner")[0])[0] == "诊断图 - 本地装配检查未完成"


def test_the_reference_drawing_has_no_diagnostic_metadata(svg):
    assert "validation_checks_not_passed" not in metadata(svg)


def test_a_mesh_the_hidden_line_step_cannot_use_is_refused_by_its_own_code():
    """With diagnostic mode, the drawing's own need for closed part shells is still refused by its own code."""

    def open_mount(scene, geometry):
        set_geometry(scene, geometry, "b02-optic-mount", without_last_triangles(box((-5.1, -15.25, 0.0), (5.1, 15.25, 37.25))))

    base = preview()
    scene = copy.deepcopy(base.scene.payload())
    geometry = {p: bytes(d) for p, d in base.files.items() if p.startswith("geometry/")}
    open_mount(scene, geometry)
    from assembly_helpers import rebuild

    package = rebuild(scene, geometry, like=base)
    status = validate_assembly_layout(package).status.value
    assert status in ("UNKNOWN", "OUT_OF_MODEL_SCOPE"), status
    detail = rebound_detail(package)
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, detail, diagnostic=True)
    assert caught.value.code is DrawingErrorCode.PART_GEOMETRY_UNREADABLE
    with pytest.raises(AssemblyDrawingError) as caught:  # (without it the inconclusive check is refused first)
        generate_assembly_drawing(package, detail)
    assert caught.value.code is DrawingErrorCode.DRAWING_REFUSED_VALIDATION_INCONCLUSIVE
    assert DrawingErrorCode.DRAWING_REFUSED_VALIDATION_FAILED.value == "DRAWING_REFUSED_VALIDATION_FAILED"
    assert DrawingErrorCode.DRAWING_REFUSED_VALIDATION_INCONCLUSIVE.value == "DRAWING_REFUSED_VALIDATION_INCONCLUSIVE"


# ============================================================================== fail closed


def _rehashed(doc: dict) -> bytes:
    content = {k: v for k, v in doc.items() if k != "detail_content_hash"}
    return json_file_bytes({**content, "detail_content_hash": detail_content_hash_of(content)})


def _edit(fn) -> bytes:
    doc = detail_doc()
    fn(doc)
    return _rehashed(doc)


def _flip_digest(d: str) -> str:
    return ("0" if d[0] != "0" else "1") + d[1:]


def _at(kind: str) -> int:
    """Index of the reference detail's (first) dimension of ``kind``."""

    return next(k for k, d in enumerate(detail_doc()["dimensions"]) if d["kind"] == kind)


DEPTH, HEIGHT, HOLE = _at("INSERTION_DEPTH"), _at("OPTICAL_AXIS_HEIGHT"), _at("FIXATION_HOLE_POSITION")
DETAIL_CASES = {
    "not_json": (b"{not json", DrawingErrorCode.DETAIL_PACKAGE_MALFORMED),
    "not_an_object": (b"[]\n", DrawingErrorCode.DETAIL_PACKAGE_MALFORMED),
    "not_canonical": (json.dumps(json.loads(DETAIL_JSON.read_bytes()), indent=4).encode(), DrawingErrorCode.DETAIL_PACKAGE_MALFORMED),
    "unknown_field": (_edit(lambda d: d.update(extra="x")), DrawingErrorCode.DETAIL_PACKAGE_MALFORMED),
    "unsupported_version": (_edit(lambda d: d.update(schema_version="1.0")), DrawingErrorCode.DETAIL_UNSUPPORTED_VERSION),
    "unsupported_schema": (_edit(lambda d: d.update(schema="opensci.other")), DrawingErrorCode.DETAIL_UNSUPPORTED_VERSION),
    "tampered_value": (DETAIL_JSON.read_bytes().replace(b'"preview_package_revision": 1', b'"preview_package_revision": 2'),
                       DrawingErrorCode.DETAIL_CONTENT_HASH_MISMATCH),
    "wrong_revision": (_edit(lambda d: d["binding"].update(preview_package_revision=2)), DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "wrong_preview": (_edit(lambda d: d["binding"].update(preview_package_content_hash=_flip_digest(d["binding"]["preview_package_content_hash"]))),
                      DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "wrong_resolution": (_edit(lambda d: d["binding"].update(commercial_resolution_id="00000000-0000-4000-8000-000000000999")),
                         DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "wrong_project": (_edit(lambda d: d["binding"]["project"]["project_hash"].update(
        digest=_flip_digest(d["binding"]["project"]["project_hash"]["digest"]))), DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "wrong_bom_digest": (_edit(lambda d: d["binding"]["commercial_bom"]["bom_hash"].update(
        digest=_flip_digest(d["binding"]["commercial_bom"]["bom_hash"]["digest"]))), DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "wrong_local_validation": (_edit(lambda d: d["binding"]["local_validation_hash"].update(
        digest=_flip_digest(d["binding"]["local_validation_hash"]["digest"]))), DrawingErrorCode.DETAIL_BINDING_MISMATCH),
    "unknown_datum": (_edit(lambda d: d["dimensions"][DEPTH]["from_point"].update(datum_key="nowhere")),
                      DrawingErrorCode.DETAIL_DIMENSION_INVALID),
    "wrong_axis_height_rule": (_edit(lambda d: d["dimensions"][HEIGHT].update(axis="WORLD_X")), DrawingErrorCode.DETAIL_DIMENSION_INVALID),
    "hole_off_grid": (_edit(lambda d: d["dimensions"][HOLE]["to_point"].update(hole=[40, 5])), DrawingErrorCode.DETAIL_DIMENSION_INVALID),
    "preview_basis_claimed_resolved": (_edit(lambda d: d["dimensions"][DEPTH].update(basis="RESOLVED")),
                                       DrawingErrorCode.DETAIL_DIMENSION_INVALID),
    "insertion_depth_not_shown": (_edit(lambda d: d["dimensions"].pop(DEPTH)), DrawingErrorCode.MISSING_REQUIRED_FACT),
}


@pytest.mark.parametrize("case", sorted(DETAIL_CASES))
def test_a_bad_detail_package_is_refused_with_a_stable_code(case):
    data, code = DETAIL_CASES[case]
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(PREVIEW_ZIP, data)
    assert caught.value.code is code, caught.value.problems
    with pytest.raises(AssemblyDrawingError) as again:
        load_assembly_detail_package(data, PREVIEW_ZIP)
    assert again.value.code is code
    envelope = json_api.generate_assembly_drawing(PREVIEW_ZIP.read_bytes(), data)
    assert envelope["ok"] is False and envelope["error"]["code"] == code.value
    assert all(forbidden_hits(p["message"]) == [] for p in envelope["error"]["problems"])


def test_an_expected_detail_hash_is_enforced():
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, expected_detail_content_hash="0" * 64)
    assert caught.value.code is DrawingErrorCode.DETAIL_CONTENT_HASH_MISMATCH
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, expected_preview_content_hash="0" * 64)
    assert caught.value.code is DrawingErrorCode.PREVIEW_PACKAGE_INVALID
    drawing = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, expected_detail_content_hash=detail_doc()["detail_content_hash"],
                                        expected_preview_content_hash=preview().content_hash)
    assert drawing.delivery_ready


def test_verified_inputs_are_used_as_they_are(ref, verified_detail, monkeypatch):
    """Bytes are verified where they enter; the verified package and detail are not verified again (with both
    verifiers disabled the drawing is the reference's), and the drawing is not re-verified after it is built."""

    from os_optics import assembly as assembly_module

    package = verified_detail.preview

    def never(*_args, **_kwargs):
        raise AssertionError("verified input verified again")

    for module, name in ((assembly_module, "load_assembly_preview_package"), (generate_module, "load_assembly_detail"),
                         (generate_module, "verify_drawing_outputs")):
        monkeypatch.setattr(module, name, never)
    drawing = generate_assembly_drawing(package, verified_detail)
    assert drawing.files == ref.files
    # a detail verified against another preview package is not this preview's
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(edited_preview(raised_optic_stack), verified_detail)
    assert caught.value.code is DrawingErrorCode.DETAIL_BINDING_MISMATCH
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, verified_detail, expected_detail_content_hash="0" * 64)
    assert caught.value.code is DrawingErrorCode.DETAIL_CONTENT_HASH_MISMATCH


def test_the_callers_validation_is_used_only_for_this_package_and_checker(ref, verified_detail, sunk, monkeypatch):
    package = verified_detail.preview
    calls = []
    original = generate_module.validate_assembly_layout
    monkeypatch.setattr(generate_module, "validate_assembly_layout", lambda p, **k: calls.append(p) or original(p, **k))
    # the caller's own validation of this package (here with a recorded tolerance of its choice) is the one drawn
    from os_optics.assembly import DEFAULT_TOLERANCES

    own = original(package, tolerances=DEFAULT_TOLERANCES.model_copy(update={"max_total_triangles": 200_000}))
    drawing = generate_assembly_drawing(package, verified_detail, validation=own)
    assert calls == [] and drawing.validation is own and drawing.result.validation.result_hash == own.result_hash
    # a validation of another package, or of another checker build, is ignored and the check runs here
    other = original(sunk[0])
    for change in ({"checker_version": "0.0.1"}, {"engine_build_id": "another-build"}):
        stale = own.model_copy(update={"checker": own.checker.model_copy(update=change)})
        for foreign in (other, stale):
            calls.clear()
            drawing = generate_assembly_drawing(package, verified_detail, validation=foreign)
            assert len(calls) == 1 and drawing.files == ref.files
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, verified_detail, validation={"status": "PASS"})  # type: ignore[arg-type]
    assert caught.value.code is DrawingErrorCode.INPUT_INVALID


def test_a_bad_preview_package_is_refused():
    files = preview_files()
    scene = files["scene.json"]
    tampered = {**files, "scene.json": scene.replace(b'"thickness_mm": 12.7', b'"thickness_mm": 12.8')}
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(tampered, DETAIL_JSON)
    assert caught.value.code is DrawingErrorCode.PREVIEW_PACKAGE_INVALID
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package_archive_bytes(files)[:4000], DETAIL_JSON)
    assert caught.value.code is DrawingErrorCode.PREVIEW_PACKAGE_INVALID


def test_a_preview_without_commercial_binding_cannot_be_drawn():
    from assembly_helpers import FIXTURE_ZIP  # the reference package before its commercial resolution was persisted

    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(FIXTURE_ZIP, DETAIL_JSON)
    assert caught.value.code is DrawingErrorCode.PREVIEW_NOT_COMMERCIALLY_BOUND


def test_a_part_without_a_commercial_bom_line_fails_closed():
    package = edited_preview(lambda scene: part(scene, "b02-optic-mount").pop("commercial_line_ref"))
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, rebound_detail(package))
    assert caught.value.code is DrawingErrorCode.MISSING_REQUIRED_FACT


def test_a_breadboard_that_disagrees_with_its_geometry_fails_closed():
    package = edited_preview(lambda scene: scene["breadboard"].update(thickness_mm=20.0))
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(package, rebound_detail(package))
    assert caught.value.code is DrawingErrorCode.BOARD_GEOMETRY_MISMATCH


def test_tampered_outputs_do_not_verify(ref, tmp_path):
    out = ref.write(tmp_path / "drawing")
    assert sorted(p.name for p in out.iterdir()) == sorted(DRAWING_FILES)
    assert verify_assembly_drawing(out) == ()
    svg_path = out / SVG
    svg_path.write_bytes(svg_path.read_bytes().replace(b"B03", b"B09", 1))
    assert any("SHA-256" in p for p in verify_assembly_drawing(out))
    (out / "extra.txt").write_bytes(b"x")
    assert verify_assembly_drawing(out) != ()
    assert verify_assembly_drawing(tmp_path / "missing") != ()


# ============================================================================== JSON API (the Designer's Pyodide worker)


def test_json_api_generates_the_same_files_from_bytes_file_map_and_memoryview(ref):
    archive = PREVIEW_ZIP.read_bytes()
    for package, detail in ((archive, detail_bytes()), (preview_files(), detail_bytes()),
                            (memoryview(archive), memoryview(detail_bytes()))):
        reply = json_api.generate_assembly_drawing(package, detail)
        assert reply["ok"] is True and reply["drawing_status"] == "DETAILED_ASSEMBLY_DRAWING" and reply["delivery_ready"] is True
        assert reply["validation_status"] == "PASS"
        assert reply["files"] == dict(ref.files)
        assert reply["result"] == ref.result.payload() and reply["manifest"] == ref.manifest.payload()
        json.dumps({k: v for k, v in reply.items() if k != "files"})  # JSON-safe apart from the file bytes
    assert json_api.verify_assembly_drawing(reply["files"]) == {"ok": True, "valid": True, "problems": []}
    broken = {**reply["files"], SVG: reply["files"][SVG] + b" "}
    assert json_api.verify_assembly_drawing(broken)["valid"] is False


def test_json_api_verifies_a_detail_package(verified_detail):
    reply = json_api.verify_assembly_detail_package(PREVIEW_ZIP.read_bytes(), detail_bytes())
    assert reply["ok"] is True and reply["detail_content_hash"] == verified_detail.content_hash
    assert reply["preview_package_content_hash"] == preview().content_hash
    assert {d["dim_ref"]: d["value_mm"] for d in reply["dimensions"]} == {k: m.value_mm for k, m in verified_detail.measured.items()}
    bad = json_api.verify_assembly_detail_package(PREVIEW_ZIP.read_bytes(), DETAIL_CASES["wrong_resolution"][0])
    assert bad == {"ok": False, "error": {"code": "DETAIL_BINDING_MISMATCH", "problems": bad["error"]["problems"]}}


def test_json_api_refuses_paths_and_bad_options(sunk):
    for package, detail in ((str(PREVIEW_ZIP), detail_bytes()), (PREVIEW_ZIP.read_bytes(), str(DETAIL_JSON)),
                            ({"scene.json": "text"}, detail_bytes())):
        reply = json_api.generate_assembly_drawing(package, detail)
        assert reply["ok"] is False and reply["error"]["code"] == "INPUT_INVALID"
    for options in ({"diagnostic": "yes"}, {"unknown": True}, ["locale"]):
        reply = json_api.generate_assembly_drawing(PREVIEW_ZIP.read_bytes(), detail_bytes(), options)
        assert reply["ok"] is False and reply["error"]["code"] == "INPUT_INVALID"
    package, detail = sunk
    files = dict(package.files)
    refused = json_api.generate_assembly_drawing(files, detail)
    assert refused["ok"] is False and refused["error"]["code"] == "DRAWING_REFUSED_VALIDATION_FAILED"
    diag = json_api.generate_assembly_drawing(files, detail, {"diagnostic": True})
    assert diag["ok"] is True and diag["drawing_status"] == "DIAGNOSTIC_VALIDATION_FAILED" and diag["delivery_ready"] is False


def test_json_api_schemas_of_the_drawing_contracts():
    kinds = json_api.capabilities()["schema_kinds"]
    for kind in ("assembly-detail", "assembly-drawing-result", "assembly-drawing-manifest"):
        assert kind in kinds
        reply = json_api.json_schema(kind)
        assert reply["ok"] is True and reply["schema"]["type"] == "object"


# ============================================================================== CLI


def test_cli_draws_and_verifies(ref, tmp_path, capsys):
    out = tmp_path / "drawing"
    assert cli_main(["assembly", "draw", str(PREVIEW_ZIP), "--detail", str(DETAIL_JSON), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "DETAILED_ASSEMBLY_DRAWING" and summary["delivery_ready"] is True
    assert summary["result_hash"] == ref.result.result_hash
    assert {p.name: p.read_bytes() for p in out.iterdir()} == dict(ref.files)
    assert cli_main(["assembly", "verify-drawing", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "VERIFIED"
    (out / RESULT).write_bytes((out / RESULT).read_bytes() + b"\n")
    assert cli_main(["assembly", "verify-drawing", str(out)]) == 66
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"


def test_cli_draws_diagnostics(sunk, tmp_path, capsys):
    package, detail = sunk
    archive = tmp_path / "sunk.zip"
    archive.write_bytes(package.archive_bytes())
    detail_path = tmp_path / "sunk.detail.json"
    detail_path.write_bytes(detail)
    args = ["assembly", "draw", str(archive), "--detail", str(detail_path), "--out", str(tmp_path / "diag")]
    assert cli_main(args) == 65  # refused without --diagnostic: nothing is written
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "DRAWING_REFUSED_VALIDATION_FAILED"
    assert not (tmp_path / "diag").exists() or not any((tmp_path / "diag").iterdir())
    assert cli_main([*args, "--diagnostic"]) == 2  # the diagnostic sheet is written; the check did not pass
    summary = json.loads(capsys.readouterr().out)  # (the summary says which sheet)
    assert summary["status"] == "DIAGNOSTIC_VALIDATION_FAILED" and summary["delivery_ready"] is False
    assert summary["validation_status"] == "FAIL" and set(summary["files"]) < {p.name for p in (tmp_path / "diag").iterdir()}
    assert cli_main(["assembly", "verify-drawing", str(tmp_path / "diag")]) == 0
    capsys.readouterr()
    assert cli_main(["assembly", "draw", str(PREVIEW_ZIP), "--detail", str(detail_path), "--out", str(tmp_path / "x")]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "DETAIL_BINDING_MISMATCH"


# ============================================================================== steps print measured lengths


def test_the_insertion_depth_step_prints_the_measured_value_not_the_stored_setting(verified_detail):
    import dataclasses

    measured = dict(verified_detail.measured)
    (dim,) = [d for d in verified_detail.detail.dimensions if d.kind.value == "INSERTION_DEPTH"]
    stored = float(insertion_setting(verified_detail.preview.scene).value)
    assert measured[dim.dim_ref].value_mm == pytest.approx(stored)
    measured[dim.dim_ref] = dataclasses.replace(measured[dim.dim_ref], value_mm=stored - 0.3)
    shifted = dataclasses.replace(verified_detail, measured=measured)
    assert float(insertion_setting(shifted.preview.scene).value) == stored  # the setting itself is unchanged
    (step,) = [s for s in assembly_steps(drawing_facts(shifted)) if s.action == "INSERT"]
    assert f"insertion depth {mm(stored - 0.3)} mm (nominal, set at assembly)" in step.text
    assert f"insertion depth {mm(stored)} mm" not in step.text


def test_an_insertion_setting_without_its_dimension_is_a_missing_fact(verified_detail):
    import dataclasses

    (dim,) = [d for d in verified_detail.detail.dimensions if d.kind.value == "INSERTION_DEPTH"]
    others = tuple(d for d in verified_detail.detail.dimensions if d.dim_ref != dim.dim_ref)
    stripped = dataclasses.replace(verified_detail, detail=verified_detail.detail.model_copy(update={"dimensions": others}))
    with pytest.raises(ValueError, match="INSERTION_DEPTH setting"):
        assembly_steps(drawing_facts(stripped))


# ============================================================================== write / verify-drawing directories


def test_write_replaces_its_own_files_and_leaves_other_files_alone(ref, sunk, tmp_path, monkeypatch):
    """A drawing is regenerated from the same packages, so writing it again replaces its four files; a foreign file stays
    byte-identical, and a write interrupted before the files are moved into place leaves the earlier four verifying."""

    out = tmp_path / "drawing"
    out.mkdir()
    (out / "notes.txt").write_bytes(b"mine")
    ref.write(out)
    diagnostic = generate_assembly_drawing(*sunk, diagnostic=True)
    diagnostic.write(out)  # another drawing's set replaces the earlier one
    assert {name: (out / name).read_bytes() for name in DRAWING_FILES} == dict(diagnostic.files)
    ref.write(out)
    assert {name: (out / name).read_bytes() for name in DRAWING_FILES} == dict(ref.files)

    from pathlib import Path

    def interrupted(_self, _target):
        raise OSError("interrupted")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        diagnostic.write(out)
    monkeypatch.undo()
    assert {name: (out / name).read_bytes() for name in DRAWING_FILES} == dict(ref.files)
    assert sorted(p.name for p in out.iterdir()) == sorted([*DRAWING_FILES, "notes.txt"])  # no temporary file left
    assert (out / "notes.txt").read_bytes() == b"mine"
    assert verify_assembly_drawing(out) != ()  # the library verifier is strict: exactly the four files
    assert verify_assembly_drawing({name: (out / name).read_bytes() for name in DRAWING_FILES}) == ()


def test_cli_verify_drawing_reads_the_four_files_and_warns_about_others(ref, tmp_path, capsys):
    out = ref.write(tmp_path / "drawing")
    (out / "notes.txt").write_bytes(b"mine")
    (out / "sub").mkdir()
    assert cli_main(["assembly", "verify-drawing", str(out)]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["status"] == "VERIFIED" and reply["problems"] == []
    assert reply["warnings"] == ["notes.txt: not a drawing file, ignored", "sub: not a drawing file, ignored"]
    (out / MANIFEST).unlink()
    assert cli_main(["assembly", "verify-drawing", str(out)]) == 66
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID"
    assert cli_main(["assembly", "verify-drawing", str(tmp_path / "missing")]) == 66
    capsys.readouterr()


def test_cli_draw_replaces_an_earlier_drawing(ref, tmp_path, capsys):
    out = tmp_path / "drawing"
    args = ["assembly", "draw", str(PREVIEW_ZIP), "--detail", str(DETAIL_JSON), "--out", str(out)]
    assert cli_main(args) == 0
    capsys.readouterr()
    assert cli_main(args) == 0  # regenerated from the same packages: its own files are replaced
    capsys.readouterr()
    assert cli_main(args + ["--overwrite"]) == 0  # an agent script's flag is accepted and changes nothing
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli_main(["assembly", "draw", "--help"])
    assert "--overwrite" not in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in out.iterdir()} == dict(ref.files)


# ============================================================================== untrusted input fuzzing

_ID, _VERSION = detail_doc()["schema"], detail_doc()["schema_version"]


def _nested(depth: int) -> bytes:
    return ('{"schema": "%s", "schema_version": "%s", "x": ' % (_ID, _VERSION) + "[" * depth + "]" * depth + "}").encode()


def _with_revision(raw: str) -> bytes:
    return DETAIL_JSON.read_bytes().replace(b'"preview_package_revision": 1', b'"preview_package_revision": ' + raw.encode())


MALFORMED = DrawingErrorCode.DETAIL_PACKAGE_MALFORMED
FUZZ_DETAILS = {
    "deep_900": (_nested(900), {MALFORMED}),
    "deep_5000": (_nested(5000), {MALFORMED}),
    "deep_unreadable": (_nested(UNREADABLE_DEPTH), {MALFORMED}),
    "deep_top_level": (deep_array().encode(), {MALFORMED}),
    "deep_in_a_field": (_with_revision("[" * 5000 + "]" * 5000), {MALFORMED}),
    "nan": (_with_revision("NaN"), {MALFORMED}),
    "infinity": (_with_revision("Infinity"), {MALFORMED}),
    "minus_infinity": (_with_revision("-Infinity"), {MALFORMED}),
    "huge_int_over_digit_limit": (_with_revision("1" * 5000), {MALFORMED}),
    "huge_int": (_with_revision("1" * 400), {MALFORMED, DrawingErrorCode.DETAIL_CONTENT_HASH_MISMATCH}),
    "huge_float": (_with_revision("1e400"), {MALFORMED}),
    "lone_surrogate_escape": (('{"schema": "%s", "schema_version": "%s", "x": "\\ud800"}' % (_ID, _VERSION)).encode(), {MALFORMED}),
    "lone_surrogate_in_value": (DETAIL_JSON.read_bytes().replace(b'"DIRECT_RELEASE"', b'"\\udfff"', 1), {MALFORMED}),
    "encoded_surrogate_bytes": (DETAIL_JSON.read_bytes().replace(b'"DIRECT_RELEASE"', b'"\xed\xa0\x80"', 1), {MALFORMED}),
    "bad_utf8": (b'{"schema": "\xff\xfe"}', {MALFORMED}),
    "utf16": (DETAIL_JSON.read_text(encoding="utf-8").encode("utf-16"), {MALFORMED}),
    "empty": (b"", {MALFORMED}),
    "nul_bytes": (b"\x00" * 64, {MALFORMED}),
    "duplicate_key": (DETAIL_JSON.read_bytes().replace(b"{", b'{"schema": "x", ', 1), {MALFORMED}),
}


@pytest.mark.parametrize("case", sorted(FUZZ_DETAILS))
def test_fuzzed_detail_packages_are_refused_with_a_stable_code_at_every_entry_point(case, tmp_path, capsys):
    data, codes = FUZZ_DETAILS[case]
    for call in (lambda: load_assembly_detail_package(data, PREVIEW_ZIP), lambda: generate_assembly_drawing(PREVIEW_ZIP, data)):
        with pytest.raises(AssemblyDrawingError) as caught:
            call()
        assert caught.value.code in codes, (case, caught.value.code, caught.value.problems)
    archive = PREVIEW_ZIP.read_bytes()
    for reply in (json_api.generate_assembly_drawing(archive, data), json_api.verify_assembly_detail_package(archive, data)):
        assert reply["ok"] is False and DrawingErrorCode(reply["error"]["code"]) in codes, (case, reply)
        json.dumps(reply)  # a plain JSON envelope
    path = tmp_path / "detail.json"
    path.write_bytes(data)
    assert cli_main(["assembly", "draw", str(PREVIEW_ZIP), "--detail", str(path), "--out", str(tmp_path / "out")]) == 65
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "ERROR" and DrawingErrorCode(error["error"]["code"]) in codes
    assert not (tmp_path / "out").exists()


def _resigned(files: dict[str, bytes]) -> dict[str, bytes]:
    """Re-sign a package's manifest over whatever its files now hold (sizes, SHA-256, content hash)."""

    manifest = json.loads(files["manifest.json"])
    for entry in manifest["files"]:
        entry["size_bytes"] = len(files[entry["path"]])
        entry["sha256"] = hashlib.sha256(files[entry["path"]]).hexdigest()
    content = {k: v for k, v in manifest.items() if k != "package_content_hash"}
    return {**files, "manifest.json": json_file_bytes({**content, "package_content_hash": content_hash_of(content)})}


def _glb(document: bytes) -> bytes:
    import struct

    chunk = document + b" " * (-len(document) % 4)
    return struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(chunk)) + struct.pack("<II", len(chunk), 0x4E4F534A) + chunk


def _deep_glb(files: dict[str, bytes]) -> dict[str, bytes]:
    path = next(p for p in sorted(files) if p.startswith("geometry/"))
    return _resigned({**files, path: _glb(b'{"asset": ' + b"[" * 5000 + b"]" * 5000 + b"}")})


FUZZ_PREVIEWS = {
    "deep_glb_json": _deep_glb,
    "deep_scene": lambda f: _resigned({**f, "scene.json": b'{"x": ' + b"[" * 5000 + b"]" * 5000 + b"}"}),
    "nan_scene": lambda f: _resigned({**f, "scene.json": f["scene.json"].replace(b'"thickness_mm": 12.7', b'"thickness_mm": NaN')}),
    "bad_utf8_scene": lambda f: _resigned({**f, "scene.json": b'{"\xff": 1}'}),
    "deep_manifest": lambda f: {**f, "manifest.json": deep_array().encode()},
    "surrogate_manifest": lambda f: {**f, "manifest.json": b'{"schema": "\\ud800"}'},
}


@pytest.mark.parametrize("case", sorted(FUZZ_PREVIEWS))
def test_fuzzed_preview_packages_are_refused_with_a_stable_code_at_every_entry_point(case, tmp_path, capsys):
    files = FUZZ_PREVIEWS[case](preview_files())
    for call in (lambda: load_assembly_detail_package(DETAIL_JSON, files), lambda: generate_assembly_drawing(files, DETAIL_JSON)):
        with pytest.raises(AssemblyDrawingError) as caught:
            call()
        assert caught.value.code is DrawingErrorCode.PREVIEW_PACKAGE_INVALID, (case, caught.value.problems)
    archive = package_archive_bytes(files)
    for package in (files, archive):
        for reply in (json_api.generate_assembly_drawing(package, detail_bytes()),
                      json_api.verify_assembly_detail_package(package, detail_bytes())):
            assert reply == {"ok": False, "error": reply["error"]} and reply["error"]["code"] == "PREVIEW_PACKAGE_INVALID", case
    path = tmp_path / "preview.zip"
    path.write_bytes(archive)
    assert cli_main(["assembly", "draw", str(path), "--detail", str(DETAIL_JSON), "--out", str(tmp_path / "out")]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "PREVIEW_PACKAGE_INVALID"


def test_fuzzed_drawing_outputs_never_raise(ref, tmp_path, capsys):
    deep = b'{"x": ' + deep_array().encode() + b"}"
    for index, data in enumerate((deep, b"\xff\xfe", b'{"x": NaN}', b'"\\ud800"')):
        for name in (RESULT, MANIFEST):
            files = {**ref.files, name: data}
            assert verify_assembly_drawing(files) != ()
            reply = json_api.verify_assembly_drawing(files)
            assert reply["ok"] is True and reply["valid"] is False
            out = tmp_path / f"{index}-{name}"
            out.mkdir()
            for file_name, content in files.items():
                (out / file_name).write_bytes(content)
            assert cli_main(["assembly", "verify-drawing", str(out)]) == 66
            assert json.loads(capsys.readouterr().out)["status"] == "INVALID"
