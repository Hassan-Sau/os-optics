"""The local Detailed Assembly Drawing of a layout on the customer's own optical table.

When the customer does not buy the OpenSci breadboard (``include_breadboard = false``), the preview scene keeps the
resolved board as its root on the same hole grid, but as the customer's table: ``breadboard.supplied_by = CUSTOMER``,
the ``CUSTOMER_SUPPLIED`` identity ``OS-CUSTOMER-OPTICAL-TABLE`` and no Customer Purchase BOM line. The drawing draws
the mounting area, gives it no balloon and no parts-legend row, and its first step marks the mounting area on the table.
"""

from __future__ import annotations

import pytest
from drawing_helpers import edited_preview, groups, metadata, rebound_detail, svg_tree, texts

from os_optics._vendor.shared_contracts.assembly_preview import CUSTOMER_OPTICAL_TABLE_SKU
from os_optics.assembly_drawing import generate_assembly_drawing, verify_assembly_drawing
from os_optics.viewer.assembly import assembly_view_scene

SVG = "assembly_drawing.svg"


def customer_table(scene: dict) -> None:
    board_ref = scene["breadboard"]["part_ref"]
    board = next(p for p in scene["parts"] if p["part_ref"] == board_ref)
    board["identity"] = {"opensci_sku": CUSTOMER_OPTICAL_TABLE_SKU, "identity_kind": "CUSTOMER_SUPPLIED",
                         "display_name": "Your optical table, mounting area 150 mm x 150 mm, M6 holes on a 25 mm grid"}
    del board["commercial_line_ref"]
    scene["breadboard"]["supplied_by"] = "CUSTOMER"


@pytest.fixture(scope="module")
def table():
    package = edited_preview(customer_table)
    return package, generate_assembly_drawing(package, rebound_detail(package))


def test_the_customer_table_is_drawn_but_never_ballooned_or_listed(table) -> None:
    package, drawing = table
    board_ref = package.scene.breadboard.part_ref
    result = drawing.result
    assert verify_assembly_drawing(drawing.files) == ()
    on_bom = sorted(p.part_ref for p in package.scene.parts if p.commercial_line_ref is not None)
    assert board_ref not in on_bom and len(on_bom) == len(package.scene.parts) - 1
    assert sorted(b.part_ref for b in result.balloons) == on_bom
    assert sorted(r.part_ref for r in result.parts_legend.rows) == on_bom
    assert CUSTOMER_OPTICAL_TABLE_SKU not in {r.opensci_sku for r in result.parts_legend.rows}
    first = result.assembly_steps[0]
    assert (first.part_ref, first.commercial_line_ref) == (board_ref, None)
    assert first.text.startswith("Mark the mounting area 150 mm x 150 mm on your optical table (M6x1.0 holes on a 25 mm grid)")
    assert any("on your optical table" in step.text for step in result.assembly_steps[1:])
    root = svg_tree(drawing.files[SVG])
    sheet_text = " ".join(t for el in root.iter() for t in texts(el))
    assert "YOUR OPTICAL TABLE" in sheet_text and "Heights from the table top." in sheet_text
    assert "on your optical table, hole" in sheet_text and "zero-based" not in sheet_text
    assert not [g for g in groups(root, "balloon") if g.get("data-part-ref") == board_ref]
    assert not [g for g in groups(root, "legend-row") if g.get("data-part-ref") == board_ref]
    assert board_ref not in metadata(root)["balloons"]


def test_the_opensci_breadboard_drawing_is_unchanged() -> None:
    """Without ``supplied_by`` the board is a BOM line, ballooned and listed as before."""

    package = edited_preview(lambda scene: None)
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    board_ref = package.scene.breadboard.part_ref
    assert board_ref in {b.part_ref for b in drawing.result.balloons}
    assert board_ref in {r.part_ref for r in drawing.result.parts_legend.rows}
    assert drawing.result.assembly_steps[0].text.startswith("Place ")


def test_the_viewer_marks_the_customer_table(table) -> None:
    package, _ = table
    view = assembly_view_scene(package)
    board = next(p for p in view.parts if p.part_id == package.scene.breadboard.part_ref)
    assert board.facts["supplied_by"] == "CUSTOMER" and "line_ref" not in board.facts
    assert board.sku == CUSTOMER_OPTICAL_TABLE_SKU
    others = [p for p in view.parts if p.part_id != package.scene.breadboard.part_ref]
    assert others and all("supplied_by" not in p.facts for p in others)
