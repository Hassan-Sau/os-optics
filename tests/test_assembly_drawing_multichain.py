"""The local Detailed Assembly Drawing of several support chains on one breadboard (v0.13).

Inputs: the 2- and 4-chain pairs produced by the generalized commercial resolution -> preview projection -> detail projection
projection (``tests/fixtures/assembly_drawing/{two,four}_lenses.*``), and 8- / 12-chain assemblies synthesized from the
4-chain package (``drawing_helpers.replicated_chains``; the detail projected with the server's fixed rule table).
Every expectation is derived from the packages' own facts.
"""

from __future__ import annotations

import re

import pytest
from drawing_helpers import (
    MULTI_CHAIN,
    forbidden_hits,
    groups,
    metadata,
    multi_chain_preview,
    projected_detail,
    replicated_chains,
    svg_tree,
    texts,
)

from os_optics._vendor.shared_contracts.assembly_detail import (
    DimensionKindV1,
    DrawingSheetV1,
    load_assembly_detail,
    verify_drawing_outputs,
)
from os_optics._vendor.shared_contracts.assembly_preview.scene import DofMechanismV1
from os_optics.assembly import ASSEMBLY_CHECKER_VERSION
from os_optics.assembly_drawing import DRAWING_FILES, DrawingStatusV1, generate_assembly_drawing, verify_assembly_drawing
from os_optics.assembly_drawing import layout as L
import drawing_text_check as privacy
from os_optics.assembly_drawing.model import drawing_facts
from os_optics.assembly_drawing.i18n import EN as EN_MESSAGES
from os_optics.assembly_drawing.steps import assembly_steps

#: The English words of every DOF mechanism (the message catalog's ``mechanism.*`` templates).
MECHANISM_WORDS = {k.split(".", 1)[1]: v for k, v in EN_MESSAGES.items() if k.startswith("mechanism.")}
from os_optics.assembly_drawing.text import mm, value_text

SVG, PDF, RESULT, MANIFEST = DRAWING_FILES
#: The sheet each input is laid out on: the smallest that fits its views at legible scales and its text.
EXPECTED_SHEET = {"two_lenses": "A2_LANDSCAPE_594x420_MM", "four_lenses": "A2_LANDSCAPE_594x420_MM",
                  "eight": "A1_LANDSCAPE_841x594_MM", "twelve": "A1_LANDSCAPE_841x594_MM"}
SHEET_MM = {"A3_LANDSCAPE_420x297_MM": ("420", "297"), "A2_LANDSCAPE_594x420_MM": ("594", "420"),
            "A1_LANDSCAPE_841x594_MM": ("841", "594"), "A0_LANDSCAPE_1189x841_MM": ("1189", "841")}
CHAINS = {"two_lenses": 2, "four_lenses": 4, "eight": 8, "twelve": 12}


def _inputs(name: str):
    if name in MULTI_CHAIN:
        preview_zip, detail_json, _ = MULTI_CHAIN[name]
        return multi_chain_preview(name), detail_json.read_bytes()
    rows = {"eight": 2, "twelve": 3}[name]
    package = replicated_chains(multi_chain_preview("four_lenses"), rows)
    return package, projected_detail(package, MULTI_CHAIN["four_lenses"][1].read_bytes())


_CACHE: dict[str, tuple] = {}


def drawn(name: str):
    """(preview, detail bytes, drawing, parsed SVG) of one input, generated once per test session."""

    if name not in _CACHE:
        package, detail = _inputs(name)
        drawing = generate_assembly_drawing(package, detail)
        _CACHE[name] = (package, detail, drawing, svg_tree(drawing.files[SVG]))
    return _CACHE[name]


REAL = ("two_lenses", "four_lenses")
ALL = REAL + ("eight", "twelve")


# ============================================================================== the drawing of N chains


@pytest.mark.parametrize("name", ALL)
def test_multi_chain_drawing_is_a_deliverable_detailed_assembly_drawing(name):
    package, _, drawing, _ = drawn(name)
    assert len(drawing_facts(load_assembly_detail(_inputs(name)[1], package)).chains) == CHAINS[name]
    assert drawing.result.drawing_status.value == "DETAILED_ASSEMBLY_DRAWING" and drawing.delivery_ready is True
    assert drawing.validation.status.value in ("PASS", "WARNING")
    assert verify_drawing_outputs(drawing.files) == () and verify_assembly_drawing(drawing.files) == ()
    assert drawing.manifest.drawing_result_hash == drawing.result.result_hash


@pytest.mark.parametrize("name", ALL)
def test_the_sheet_is_the_smallest_that_fits_and_every_output_says_so(name):
    _, _, drawing, svg = drawn(name)
    sheet = drawing.result.sheet
    assert sheet is DrawingSheetV1(EXPECTED_SHEET[name])
    width, height = SHEET_MM[sheet.value]
    assert (svg.get("width"), svg.get("height")) == (f"{width}mm", f"{height}mm")
    assert svg.get("viewBox") == f"0 0 {width} {height}"
    (block,) = groups(svg, "title-block")
    assert sheet.value[:2] in texts(block)  # the SIZE cell
    points = [f"{float(v) * 72.0 / 25.4:.4f}" for v in (width, height)]
    assert f"/MediaBox [0 0 {points[0]} {points[1]}]".encode() in drawing.files[PDF]
    for key in ("FRONT", "SIDE", "PLAN"):
        a, b = (float(x) for x in {v.view.value: v.scale for v in drawing.result.views}[key].split(":"))
        assert a / b >= min(L.MIN_ELEVATION_SCALE, L.MIN_PLAN_SCALE)  # legible: not shrunk to fit a smaller sheet


def test_a_smaller_sheet_is_not_skipped():
    """Two chains do not fit the A3 sheet (the v0.12 failure: notes and steps), four need text columns on A2."""

    from os_optics.assembly_drawing import sheet as sheet_module

    for name, refused in (("two_lenses", [("A3_LANDSCAPE_420x297_MM", 1)]),
                          ("four_lenses", [("A3_LANDSCAPE_420x297_MM", 1), ("A2_LANDSCAPE_594x420_MM", 1)])):
        package, detail = _inputs(name)
        verified = load_assembly_detail(detail, package)
        facts = drawing_facts(verified)
        builder = sheet_module.SheetBuilder(facts, drawing_status=DrawingStatusV1.DETAILED_ASSEMBLY_DRAWING, validation_status="WARNING", validation_issues=[],
                                            checker_version=ASSEMBLY_CHECKER_VERSION, steps=assembly_steps(facts),
                                            generator_version="1.1.0", validation_interference="NO_INTERFERENCE")
        for key, columns in refused:
            geometry = L.SheetGeometry(next(s for s in L.SHEETS if s.key == key), columns)
            assert builder._text_fits(geometry) is None, (name, key)


@pytest.mark.parametrize("name", ALL)
def test_every_bom_line_is_ballooned(name):
    package, _, drawing, svg = drawn(name)
    lines = {p.part_ref: p.commercial_line_ref for p in package.scene.parts}
    assert {b.part_ref: b.commercial_line_ref for b in drawing.result.balloons} == lines
    assert all(b.views for b in drawing.result.balloons)
    balloons = groups(svg, "balloon")
    assert {g.get("data-part-ref") for g in balloons} == set(lines)
    for g in balloons:
        assert texts(g) == [lines[g.get("data-part-ref")]] == [g.get("data-line-ref")]
    assert len(drawing.result.parts_legend.rows) == len(lines)


def test_large_assemblies_get_one_stacked_balloon_group_per_chain():
    for name in ("four_lenses", "eight", "twelve"):
        package, detail, _, svg = drawn(name)
        facts = drawing_facts(load_assembly_detail(detail, package))
        stacked = [g for g in groups(svg, "balloon") if g.get("data-stack")]
        by_chain: dict[str, list] = {}
        for g in stacked:
            by_chain.setdefault(g.get("data-stack"), []).append(g)
        assert len(by_chain) >= len(facts.chains) - 1, name  # a chain without a free place is ballooned one by one
        for number, members in by_chain.items():
            chain = facts.chains[int(number) - 1]
            assert [g.get("data-part-ref") for g in members] == list(chain.part_refs)  # in assembly order
            assert {g.get("data-view") for g in members} == {"PLAN"}
            assert sum(1 for g in members for _ in g.iter("{http://www.w3.org/2000/svg}path")) == 1  # one leader
            assert members[0].get("data-leader-part-ref") in chain.part_refs


@pytest.mark.parametrize("name", ALL)
def test_every_dimension_is_drawn_with_its_measured_value(name):
    package, detail, drawing, svg = drawn(name)
    verified = load_assembly_detail(detail, package)
    by_ref = {d.dim_ref: d for d in verified.detail.dimensions}
    assert {d.dim_ref for d in drawing.result.dimensions} == set(by_ref)
    drawn_refs: dict[str, object] = {}
    for g in groups(svg, "dimension"):
        for ref in (g.get("data-dim-refs") or g.get("data-dim-ref")).split():
            assert ref not in drawn_refs, ref  # each dimension is drawn exactly once
            drawn_refs[ref] = g
    assert set(drawn_refs) == set(by_ref)
    for dim in drawing.result.dimensions:
        measured = verified.measured[dim.dim_ref].value_mm
        assert dim.value_mm == value_text(measured) and dim.views
        group = drawn_refs[dim.dim_ref]
        assert group.get("data-view") == dim.views[0].value
        (label,) = texts(group)
        shown = mm(abs(measured))
        count = group.get("data-count")
        if group.get("data-style") == "ORDINATE":
            assert label.split()[0].strip("()") == shown  # a coordinate from the datum, marked with its basis
        else:
            prefix = f"{count} x " if count else ""
            assert label.startswith(prefix + (f"({shown})" if dim.basis.value == "REFERENCE" else shown)), label
    rows = {g.get("data-dim-ref") for g in groups(svg, "dimension-row")}
    assert rows == set(by_ref)  # and every one is in the dimension table
    assert set(metadata(svg)["dimensions"]) == set(by_ref)


@pytest.mark.parametrize("name", REAL)
def test_the_assembly_controlling_dimensions_of_every_chain(name):
    """One optical axis height per optic, one insertion depth per post, the fixation hole and offset of every holder and
    every optic's position on the board -- each drawn in its view."""

    package, detail, drawing, _ = drawn(name)
    facts = drawing_facts(load_assembly_detail(detail, package))
    by_kind: dict[str, set[str]] = {}
    for dim in drawing.result.dimensions:
        by_kind.setdefault(dim.kind.value, set()).add(dim.subject_part_ref)
    optics = {r for c in facts.chains for r in c.optic_refs}
    posts = {p.part_ref for p in package.scene.parts if p.support and p.support.relation.value == "INSERTED_INTO"}
    holders = {f.part_ref for f in package.scene.fixations}
    assert by_kind["OPTICAL_AXIS_HEIGHT"] == by_kind["OPTIC_BOARD_POSITION"] == optics and len(optics) == CHAINS[name]
    assert by_kind["INSERTION_DEPTH"] == posts and len(posts) == CHAINS[name]
    assert by_kind["FIXATION_HOLE_POSITION"] == by_kind["FIXATION_OFFSET"] == holders
    views = {d.dim_ref: d.views[0].value for d in drawing.result.dimensions}
    for dim in drawing.result.dimensions:
        if dim.kind in (DimensionKindV1.OPTICAL_AXIS_HEIGHT, DimensionKindV1.INSERTION_DEPTH):
            assert views[dim.dim_ref] == "FRONT"
        if dim.kind in (DimensionKindV1.FIXATION_HOLE_POSITION, DimensionKindV1.OPTIC_BOARD_POSITION):
            assert views[dim.dim_ref] == "PLAN"


def test_repeated_dimensions_are_drawn_once_as_n_x_value():
    _, _, drawing, svg = drawn("four_lenses")
    heights = [g for g in groups(svg, "dimension") if g.get("data-kind") == "OPTICAL_AXIS_HEIGHT"]
    assert len(heights) == 1 and heights[0].get("data-count") == "4" and texts(heights[0]) == ["4 x 125"]
    (depth,) = [g for g in groups(svg, "dimension") if g.get("data-kind") == "INSERTION_DEPTH"]
    assert texts(depth)[0].startswith("4 x 47 N")  # still marked as a nominal value
    assert "n x = the same dimension of n features" in " ".join(texts(groups(svg, "dimension-table")[0]))


def test_crowded_plan_dimensions_become_ordinates_from_the_board_corner():
    for name in ("four_lenses", "twelve"):
        _, _, _, svg = drawn(name)
        ordinates = [g for g in groups(svg, "dimension") if g.get("data-style") == "ORDINATE"]
        assert ordinates and {g.get("data-view") for g in ordinates} == {"PLAN"}
        assert {g.get("data-kind") for g in ordinates} <= {"FIXATION_HOLE_POSITION", "OPTIC_BOARD_POSITION"}
        datums = groups(svg, "dimension-datum", view="PLAN")
        assert datums and all(texts(d) == ["0"] for d in datums)
    _, _, _, svg = drawn("two_lenses")  # few lines: stacked as in v0.12
    assert not [g for g in groups(svg, "dimension") if g.get("data-style") == "ORDINATE"]


@pytest.mark.parametrize("name", REAL)
def test_the_views_frame_the_whole_assembly(name):
    package, detail, drawing, svg = drawn(name)
    facts = drawing_facts(load_assembly_detail(detail, package))
    views = {v.view.value: v for v in drawing.result.views}
    assert {c.root_ref for c in facts.chains} <= set(views["PLAN"].part_refs)  # no chain hides another seen from above
    assert {c.root_ref for c in facts.chains} <= set(views["PICTORIAL"].part_refs)
    for chain in facts.chains:  # every chain is seen in the elevations (its optic may sit edge-on inside its mount)
        seen = set(views["FRONT"].part_refs) | set(views["SIDE"].part_refs)
        assert set(chain.part_refs) - set(chain.optic_refs) <= seen
    width = float(svg.get("viewBox").split()[2])
    for key in ("FRONT", "SIDE", "PLAN", "PICTORIAL"):  # every view's drawing stays on the left of the text columns
        (view,) = groups(svg, "view", view=key)
        xs = [float(n) for p in view.iter("{http://www.w3.org/2000/svg}path") for n in re.findall(r"-?\d+(?:\.\d+)?", p.get("d", ""))[0::2]]
        geometry = next(g for g in L.sheet_geometries() if g.spec.key == drawing.result.sheet.value)
        assert 0.0 < min(xs) and max(xs) < min(c[0] for c in geometry.text_columns()) < width, key


@pytest.mark.parametrize("name", REAL)
def test_steps_are_grouped_by_support_chain(name):
    package, detail, drawing, svg = drawn(name)
    facts = drawing_facts(load_assembly_detail(detail, package))
    steps = drawing.result.assembly_steps
    assert steps[0].action.value == "PLACE" and steps[0].part_ref == package.scene.breadboard.part_ref
    chain_of = {r: c.number for c in facts.chains for r in c.part_refs}
    sequence = [chain_of[s.part_ref] for s in steps[1:] if s.part_ref is not None]
    assert sequence == sorted(sequence)  # chain by chain
    for chain in facts.chains:
        mine = [s for s in steps if chain_of.get(s.part_ref) == chain.number]
        assert [s.part_ref for s in mine if s.action.value != "ADJUST"] == list(chain.part_refs)
        assert [s.action.value for s in mine[:4]] == ["STAND_AND_FIX", "INSERT", "MOUNT", "SEAT"]
        (insert,) = [s for s in mine if s.action.value == "INSERT"]
        assert "insertion depth 47 mm (nominal, set at assembly)" in insert.text
    headings = groups(svg, "step-chain")
    assert [h.get("data-chain") for h in headings] == [str(c.number) for c in facts.chains]
    for heading, chain in zip(headings, facts.chains):
        assert texts(heading) == [f"OPTIC STACK {chain.number} ({', '.join(chain.instance_ids)}): "
                                  + " / ".join(facts.line_ref(r) for r in chain.part_refs)]
    assert all(len(s.text) <= 600 for s in steps)


@pytest.mark.parametrize("name", REAL)
def test_notes_group_repeated_facts(name):
    package, _, _, svg = drawn(name)
    notes = " ".join(texts(groups(svg, "notes")[0]))
    assert f"{CHAINS[name]} optic stacks on one breadboard" in notes
    assert notes.count("BASE FIXING (through its slot)") == 1
    assert notes.count("Fixing screws: standard M6x1.0 screws and washers, supplied with the order (not separate item lines).") == 1
    for fixation in package.scene.fixations:
        i, j = fixation.board_holes[0]
        assert f"hole ({i + 1}, {j + 1})" in notes
    assert notes.count("YOUR LIGHT SOURCES") == 1
    assert "DESIGN BEAM AXES" in notes
    for internal in ("SUPPORT CHAIN", "FIXATIONS in", "zero-based", "height closure", "intra chain envelope"):
        assert internal.lower() not in " ".join(texts(svg)).lower(), internal
    tags = groups(svg, "fixation-tag", view="PLAN")
    assert {t.get("data-part-ref") for t in tags} == {f.part_ref for f in package.scene.fixations}
    assert not groups(svg, "fixation-callout")


@pytest.mark.parametrize("name", ("four_lenses", "eight"))
def test_generation_is_deterministic(name):
    package, detail, drawing, _ = drawn(name)
    again = generate_assembly_drawing(package.archive_bytes(), detail)
    assert dict(again.files) == dict(drawing.files)


@pytest.mark.parametrize("name", ALL)
def test_every_output_token_is_allowed_and_no_supplier_vocabulary_appears(name):
    package, detail, drawing, _ = drawn(name)
    verified = load_assembly_detail(detail, package)
    generator = drawing.result.generator.model_dump(mode="json")
    allowed = privacy.allowed_vocabulary(verified, drawing.validation, (str(v) for v in generator.values()))
    assert privacy.unexpected_text(drawing.files, allowed) == []
    for file_name, data in drawing.files.items():
        assert forbidden_hits(data.decode("latin-1" if file_name == PDF else "utf-8")) == [], file_name


def test_text_flows_through_the_text_columns():
    _, _, drawing, svg = drawn("twelve")
    leads = {round(float(next(g.iter("{http://www.w3.org/2000/svg}text")).get("x")), 2)
             for kind in ("dimension-table", "customer-supplied-row", "step") for g in groups(svg, kind)}
    assert len(leads) >= 2  # the text continues in a further text column, one column pitch to the right
    ordered = sorted(leads)
    assert all(abs(b - a - (L.TEXT_COLUMN_W + L.TEXT_COLUMN_GAP)) < 0.01 for a, b in zip(ordered, ordered[1:]))
    assert ordered[-1] + L.TEXT_COLUMN_W <= float(svg.get("viewBox").split()[2]) - L.MARGIN + 0.01
    (legend,) = groups(svg, "parts-legend")
    assert len(groups(legend, "legend-row")) == len(drawing.result.balloons)


# ============================================================================== the single-chain sheet


def test_one_chain_keeps_the_single_chain_sheet_and_wording():
    from drawing_helpers import DETAIL_JSON, PREVIEW_ZIP, enlarged_board, rebound_detail

    drawing = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)
    svg = svg_tree(drawing.files[SVG])
    assert drawing.result.sheet is DrawingSheetV1.A3_LANDSCAPE_420x297_MM and svg.get("width") == "420mm"
    assert not groups(svg, "step-chain") and not [g for g in groups(svg, "balloon") if g.get("data-stack")]
    assert not [g for g in groups(svg, "dimension") if g.get("data-count") or g.get("data-style")]
    notes = " ".join(texts(groups(svg, "notes")[0]))
    assert "optic stacks" not in notes and notes.count("BASE FIXING: ") == 1  # one fixing note
    # one fixation is called out where a board corner is free, otherwise (the compact reference board) its hole is tagged
    assert len(groups(svg, "fixation-tag")) == 1 and not groups(svg, "fixation-callout")
    roomy = enlarged_board()
    roomy_svg = svg_tree(generate_assembly_drawing(roomy, rebound_detail(roomy)).files[SVG])
    assert groups(roomy_svg, "fixation-callout") and not groups(roomy_svg, "fixation-tag")


def test_the_a3_geometry_is_the_v012_sheet():
    a3 = L.SheetGeometry(L.SHEETS[0], 1)
    assert a3.is_v012_a3 and a3.frame == L.FRAME and a3.left == L.LEFT and a3.right == L.RIGHT and a3.banner == L.BANNER
    assert a3.text_columns() == [L.RIGHT] and (a3.spec.width, a3.spec.height) == (420.0, 297.0)
    assert [g.spec.size for g in L.sheet_geometries()] == ["A3", "A2", "A2", "A1", "A1", "A1", "A0", "A0", "A0", "A0"]
    for geometry in L.sheet_geometries():
        columns = geometry.text_columns()
        assert columns[-1][2] == geometry.spec.width - L.MARGIN and geometry.view_region[2] < columns[0][0]


def test_every_dof_mechanism_has_template_words():
    """Scene 1.1 adds TRANSLATION_MOUNT and ROTATION_MOUNT: a drawing of such a scene must not fail on a missing word."""

    assert set(MECHANISM_WORDS) == {m.value for m in DofMechanismV1}

