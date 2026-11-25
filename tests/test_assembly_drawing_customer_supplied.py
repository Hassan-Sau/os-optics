"""Customer-supplied components on the local Detailed Assembly Drawing and in the local assembly check.

A customer-supplied design component -- the customer's light source, or an optical component OpenSci does not supply
-- is a customer-supplied design anchor of the preview scene (``customer_supply``: role, reason, OpenSci product,
orientation, reserved envelope). The drawing generalizes the customer light-source mechanism: every OpenSci part is
drawn as resolved, each customer-supplied optical component is a dashed, labelled placeholder at its design position
(its reserved envelope when its shape is known), listed in a customer-supplied components table apart from the parts
legend and placed by a ``PLACE_CUSTOMER_SUPPLIED`` step. The local layout check reports it as CUSTOMER_SUPPLIED_COMPONENT
(never a failure by itself) and fails a part that intrudes into its reserved space (RESERVED_SPACE_INTRUSION).
"""

from __future__ import annotations

import pytest
from drawing_helpers import edited_preview, groups, rebound_detail, svg_tree, texts

from os_optics._vendor.shared_contracts.assembly_detail import CustomerSuppliedModelV1
from os_optics._vendor.shared_contracts.assembly_preview import note_message, AssemblyNoteCodeV1
from os_optics.assembly import validate_assembly_layout
from os_optics.assembly_drawing import generate_assembly_drawing, verify_assembly_drawing
from os_optics.viewer.assembly import assembly_view_scene

SVG = "assembly_drawing.svg"
SKU = "OS-OPT-POL-LIN-D12P7-TEST"
#: The catalog model the design used for the customer's own polarizer (the detail's ``customer_supplied_models``).
MODEL = CustomerSuppliedModelV1(instance_id="P1", opensci_sku=SKU, display_name="Linear polarizer D12.7 mm",
                                display_name_zh_cn="线偏振片 D12.7 mm")


def _source_and_optic(scene: dict) -> tuple[dict, dict]:
    anchors = scene["beam_reference"]["design_anchors"]
    source = next(a for a in anchors if a.get("customer_supplied"))
    optic = next(a for a in anchors if a.get("realized_by_part_ref"))
    return source, optic


def add_customer_component(instance_id: str, supply: dict, at: float = 0.85):
    """A customer-supplied (design-only) component on the source's beam, ``at`` of the way from the source to the first
    optic (on the board), facing along the beam; the nominal beam axis visits it (as the resolver's trace does).
    ``supply``: its ``customer_supply`` without the orientation (taken from the source)."""

    def edit(scene: dict) -> None:
        source, optic = _source_and_optic(scene)
        position = [s + at * (o - s) for s, o in zip(source["position_mm"], optic["position_mm"], strict=True)]
        for axis in scene["beam_reference"]["beam_axes"]:
            if max(abs(p - q) for p, q in zip(axis["points_mm"][0], source["position_mm"], strict=True)) < 1e-6:
                axis["points_mm"].insert(1, position)
                if axis.get("through") is not None:
                    axis["through"].insert(1, instance_id)
        full = supply | {"orientation_wxyz": list(source["customer_supply"]["orientation_wxyz"])}
        scene["beam_reference"]["design_anchors"].append(
            {"instance_id": instance_id, "anchor_kind": "COMPONENT", "position_mm": position, "customer_supplied": True,
             "customer_supply": full})
        scene["beam_reference"]["design_anchors"].sort(key=lambda a: (a["anchor_kind"], a["instance_id"]))
        scene["notes"].append({"code": "CUSTOMER_SUPPLIED_COMPONENT", "severity": "INFO", "design_instance_id": instance_id,
                               "message": note_message(AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT,
                                                       reserved_space="reserved_envelope" in supply)})
    return edit


def add_customer_polarizer(at: float = 0.85, *, envelope: bool = True, diameter: float = 12.7):
    """A customer-supplied catalog polarizer (NOT_ORDERABLE) with the catalog's design geometry as its envelope."""

    supply: dict = {"reason": "NOT_ORDERABLE", "optical_role": "polarizer", "opensci_sku": SKU}
    if envelope:
        supply["reserved_envelope"] = {"diameter_mm": diameter, "z_min_mm": 0.0, "z_max_mm": 4.7}
    return add_customer_component("P1", supply, at)


def add_declared_lens(at: float = 0.85, *, diameter: float = 20.0):
    """The customer's own lens whose optical model the Project declares (NOT_IN_CATALOG, no OpenSci product): the
    resolution reserves the declared model's geometry (basis PROJECT_DECLARED_GEOMETRY), as the server's projector does
    for a component with an ``optical_model_id``."""

    return add_customer_component("OWN1", {
        "reason": "NOT_IN_CATALOG", "optical_role": "lens",
        "reserved_envelope": {"basis": "PROJECT_DECLARED_GEOMETRY", "diameter_mm": diameter, "z_min_mm": 0.0, "z_max_mm": 4.0}}, at)


@pytest.fixture(scope="module")
def supplied():
    package = edited_preview(add_customer_polarizer())
    return package, generate_assembly_drawing(package, rebound_detail(package, models=(MODEL,)))


def test_the_light_source_is_a_customer_supplied_anchor_with_its_supply() -> None:
    package = edited_preview(lambda scene: None)
    sources = [a for a in package.scene.beam_reference.design_anchors if a.customer_supplied]
    assert sources and all(a.customer_supply.reason.value == "CUSTOMER_LIGHT_SOURCE" for a in sources)
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    rows = drawing.result.customer_supplied_components
    assert [r.design_instance_id for r in rows] == sorted(a.instance_id for a in sources)
    assert all(r.optical_role == "source" and not r.reserved_envelope for r in rows)
    steps = [s for s in drawing.result.assembly_steps if s.action.value == "PLACE_CUSTOMER_SUPPLIED"]
    assert [s.design_instance_id for s in steps] == [r.design_instance_id for r in rows]
    assert steps[0].text.startswith("Place your own light source") and steps == list(drawing.result.assembly_steps[-len(steps):])


def test_a_customer_supplied_optic_is_a_labelled_placeholder_listed_apart(supplied) -> None:
    package, drawing = supplied
    assert verify_assembly_drawing(drawing.files) == ()
    result = drawing.result
    assert drawing.delivery_ready
    # every OpenSci part is still listed and ballooned exactly as before; the polarizer is not a BOM part
    on_bom = sorted(p.part_ref for p in package.scene.parts if p.commercial_line_ref is not None)
    assert sorted(r.part_ref for r in result.parts_legend.rows) == on_bom
    assert SKU not in {r.opensci_sku for r in result.parts_legend.rows}
    row = next(r for r in result.customer_supplied_components if r.design_instance_id == "P1")
    anchor = next(a for a in package.scene.beam_reference.design_anchors if a.instance_id == "P1")
    assert (row.optical_role, row.reason.value, row.opensci_sku, row.reserved_envelope) == ("polarizer", "NOT_ORDERABLE", SKU, True)
    assert row.position_mm == pytest.approx(anchor.position_mm, abs=1e-3)
    frame = package.scene.beam_reference.design_frame_in_world
    w, x, y, z = frame.quaternion_wxyz
    v = row.design_position_mm  # back into the world by the design frame
    t = (2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0]))
    world = (v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0]))
    assert [a + b for a, b in zip(world, frame.position_mm, strict=True)] == pytest.approx(anchor.position_mm, abs=1e-3)
    step = next(s for s in result.assembly_steps if s.design_instance_id == "P1")
    assert step.action.value == "PLACE_CUSTOMER_SUPPLIED" and step.part_ref is None and step.commercial_line_ref is None
    # the customer's own part, never presented as the OpenSci product: the product is only the model the design used
    assert step.text.startswith(f"Place your own polarizer P1 (modelled as Linear polarizer D12.7 mm {SKU}) at its design position x ")
    assert "mm from the board's -X edge" in step.text and "mm above the board top" in step.text  # located
    assert row.model_display_name == "Linear polarizer D12.7 mm"
    assert "its support is yours; its space is kept free in the layout." in step.text
    root = svg_tree(drawing.files[SVG])
    placeholders = groups(root, "customer-supplied", instance_id="P1")
    assert placeholders and {g.get("data-reason") for g in placeholders} == {"NOT_ORDERABLE"}
    assert all(g.get("data-outline") == "RESERVED_ENVELOPE" for g in placeholders)
    labels = [t for g in groups(root, "customer-supplied-label") for t in texts(g)]
    assert any(t.startswith("P1") for t in labels) and not any(SKU in t for t in labels)  # never labelled by the model's SKU
    table = groups(root, "customer-supplied-table")
    table_text = " ".join(texts(table[0]))
    assert "P1" in table_text and f"your own polarizer (modelled as Linear polarizer D12.7 mm {SKU})" in table_text
    assert "outline drawn: space kept free, diameter 12.7 mm" in table_text
    sheet = " ".join(t for el in root.iter() for t in texts(el))
    assert "YOUR OWN COMPONENTS" in sheet
    assert "simulation-only product, not supplied by OpenSci" in sheet


def test_an_unknown_shape_is_the_standard_symbol() -> None:
    package = edited_preview(add_customer_polarizer(envelope=False))
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    row = next(r for r in drawing.result.customer_supplied_components if r.design_instance_id == "P1")
    assert row.reserved_envelope is False
    step = next(s for s in drawing.result.assembly_steps if s.design_instance_id == "P1")
    assert step.text.endswith("its support is yours.")


def test_the_layout_check_reports_customer_components_and_reserved_space_intrusions(supplied) -> None:
    package, _ = supplied
    result = validate_assembly_layout(package.archive_bytes())
    codes = {(i.code.value, i.design_instance_id) for i in result.issues}
    assert ("CUSTOMER_SUPPLIED_COMPONENT", "P1") in codes
    assert not [i for i in result.issues if i.code.value == "RESERVED_SPACE_INTRUSION"]
    assert result.status.value != "FAIL"
    # a reserved envelope placed on the first optic: that optic (and whatever else is inside) intrudes
    intruded = edited_preview(add_customer_polarizer(at=1.0, diameter=25.4))
    bad = validate_assembly_layout(intruded.archive_bytes())
    hits = [i for i in bad.issues if i.code.value == "RESERVED_SPACE_INTRUSION"]
    assert hits and all(i.design_instance_id == "P1" and len(i.part_refs) == 1 for i in hits)
    assert bad.status.value == "FAIL"


def test_the_viewer_marks_what_the_customer_supplies(supplied) -> None:
    package, _ = supplied
    view = assembly_view_scene(package)
    marker = next(m for m in view.markers if m.marker_id == "P1")
    assert marker.marker_class == "CUSTOMER_SUPPLIED"
    assert (marker.facts["optical_role"], marker.facts["reason"], marker.facts["opensci_sku"]) == ("polarizer", "NOT_ORDERABLE", SKU)


def test_a_declared_optic_is_a_placeholder_at_its_design_position_with_its_declared_space() -> None:
    """A customer-supplied component with a Project-declared optical model (reserved envelope of basis
    PROJECT_DECLARED_GEOMETRY) goes through the local layout check and the Detailed Assembly Drawing like any other
    customer-supplied optic: a CUSTOMER_SUPPLIED_COMPONENT finding, its declared space kept free (an intruding part
    fails), and a labelled placeholder with that outline at its design position, never an OpenSci product."""

    package = edited_preview(add_declared_lens())
    anchor = next(a for a in package.scene.beam_reference.design_anchors if a.instance_id == "OWN1")
    assert anchor.customer_supply.reserved_envelope.basis == "PROJECT_DECLARED_GEOMETRY"
    assert anchor.customer_supply.opensci_sku is None

    result = validate_assembly_layout(package.archive_bytes())
    assert [(i.code.value, i.design_instance_id) for i in result.issues if i.design_instance_id == "OWN1"] == [
        ("CUSTOMER_SUPPLIED_COMPONENT", "OWN1")]
    assert result.status.value != "FAIL"
    intruded = validate_assembly_layout(edited_preview(add_declared_lens(at=1.0, diameter=25.4)).archive_bytes())
    hits = [i for i in intruded.issues if i.code.value == "RESERVED_SPACE_INTRUSION"]
    assert hits and all(i.design_instance_id == "OWN1" for i in hits) and intruded.status.value == "FAIL"

    drawing = generate_assembly_drawing(package, rebound_detail(package))
    assert verify_assembly_drawing(drawing.files) == () and drawing.delivery_ready
    row = next(r for r in drawing.result.customer_supplied_components if r.design_instance_id == "OWN1")
    assert (row.optical_role, row.reason.value, row.opensci_sku, row.model_display_name, row.reserved_envelope) == (
        "lens", "NOT_IN_CATALOG", None, None, True)
    assert row.position_mm == pytest.approx(anchor.position_mm, abs=1e-3)
    on_bom = sorted(p.part_ref for p in package.scene.parts if p.commercial_line_ref is not None)
    assert sorted(r.part_ref for r in drawing.result.parts_legend.rows) == on_bom
    step = next(s for s in drawing.result.assembly_steps if s.design_instance_id == "OWN1")
    assert step.action.value == "PLACE_CUSTOMER_SUPPLIED" and step.part_ref is None and step.commercial_line_ref is None
    assert step.text.startswith("Place your own lens OWN1 at its design position x ") and "modelled as" not in step.text
    assert step.text.endswith("its space is kept free in the layout.")
    root = svg_tree(drawing.files[SVG])
    placeholders = groups(root, "customer-supplied", instance_id="OWN1")
    assert placeholders and all((g.get("data-reason"), g.get("data-outline")) == ("NOT_IN_CATALOG", "RESERVED_ENVELOPE")
                                for g in placeholders)
    table_text = " ".join(texts(groups(root, "customer-supplied-table")[0]))
    assert "OWN1 your own lens; not an OpenSci catalog product" in table_text
    assert "outline drawn: space kept free, diameter 20 mm, 0 to 4 mm along its axis" in table_text
