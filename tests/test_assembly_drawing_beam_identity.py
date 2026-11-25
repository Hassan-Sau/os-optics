"""The drawing and the local beam-axis check name the elements of a beam axis by the ids the scene gives
(``BeamAxisV1.through``), never by matching coordinates.

* an optic on a laterally shifted axis (after a plate splitter the physical axis lies ~1 mm off the nominal point) keeps
  its name, its place in the beam order and its spacing row, and is not an obstruction of its own beam;
* two light sources at one point (a two-colour laser modelled as two sources): each path starts at its own source and
  each source gets its own input step;
* an element the beam enters and leaves at two points (an offset return) is one element of the path;
* a scene that does not name its axes (an older package) is still read by position.
"""

from __future__ import annotations

import copy

from drawing_helpers import edited_preview, rebound_detail

from os_optics.assembly import validate_assembly_layout
from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.assembly_drawing.alignment import alignment_items, beam_paths, spacing_rows
from os_optics.assembly_drawing.model import drawing_facts
from os_optics.assembly_drawing.steps import Words
from os_optics._vendor.shared_contracts.assembly_detail import load_assembly_detail

SHIFT_MM = 1.0


def _facts(package):
    return drawing_facts(load_assembly_detail(rebound_detail(package), package))


def shifted_axis(scene: dict) -> None:
    """The reference beam after the source runs 1 mm off the anchors in +Y (a plate splitter's lateral shift)."""

    (axis,) = scene["beam_reference"]["beam_axes"]
    axis["points_mm"] = [[x, y + SHIFT_MM, z] for x, y, z in axis["points_mm"][1:]]
    axis["points_mm"].insert(0, scene["beam_reference"]["design_anchors"][1]["position_mm"])  # SRC stays where it is


def unnamed(scene: dict) -> None:
    for axis in scene["beam_reference"]["beam_axes"]:
        axis.pop("through", None)


def test_an_optic_on_a_shifted_axis_keeps_its_name_its_order_and_its_spacing():
    package = edited_preview(shifted_axis)
    facts = _facts(package)
    w = Words("en")
    (path,) = beam_paths(facts, w)
    assert [p.instance_id for p in path.points] == ["SRC", "L1", "focal_plane"]
    rows = [(r.a.instance_id, r.b.instance_id) for r in spacing_rows(facts, w)]
    assert rows == [("SRC", "L1"), ("L1", "focal_plane")]
    text = " ".join(alignment_items(facts, w))
    assert "Place the optics in beam order along beam-1: L1 (B01)." in text
    assert "end of the design beam" not in text
    # the optic the axis names is on its own beam, not an obstruction of it
    result = validate_assembly_layout(package)
    assert not [i for i in result.issues if i.code.value == "BEAM_AXIS_OBSTRUCTED"]
    drawing = generate_assembly_drawing(package.archive_bytes(), rebound_detail(package))
    assert [(r.from_instance_id, r.to_instance_id) for r in drawing.result.optical_spacing] == rows


def test_an_older_unnamed_axis_is_read_by_position():
    facts = _facts(edited_preview(unnamed))
    (path,) = beam_paths(facts, Words("en"))
    assert [p.instance_id for p in path.points] == ["SRC", "L1", "focal_plane"]
    # and an unnamed shifted axis still loses the shifted points (the behaviour the ids replace)
    lost = _facts(edited_preview(lambda s: (shifted_axis(s), unnamed(s))))
    (path,) = beam_paths(lost, Words("en"))
    assert [p.instance_id for p in path.points] == ["SRC", None, None]


def colocated_sources(scene: dict) -> None:
    """A second customer light source SRC_A at SRC's point feeding its own 532 nm axis (matched by position, both paths
    would start at SRC, the first anchor there)."""

    reference = scene["beam_reference"]
    source = next(a for a in reference["design_anchors"] if a["instance_id"] == "SRC")
    reference["design_anchors"].append({**copy.deepcopy(source), "instance_id": "SRC_A"})
    reference["design_anchors"].sort(key=lambda a: (a["anchor_kind"], a["instance_id"]))
    (axis,) = reference["beam_axes"]
    reference["beam_axes"].append({**copy.deepcopy(axis), "path_key": "beam-2", "wavelength_nm": 532.0,
                                   "through": ["SRC_A", *axis["through"][1:]]})
    note = next(n for n in scene["notes"] if n["code"] == "EXTERNAL_LIGHT_SOURCE")
    scene["notes"].append({**note, "design_instance_id": "SRC_A"})
    scene["notes"].sort(key=lambda n: (n["code"], n.get("design_instance_id") or "", n.get("dof_axis") or ""))


def test_two_sources_at_one_point_each_start_their_own_path_and_input_step():
    facts = _facts(edited_preview(colocated_sources))
    w = Words("en")
    assert {p.path_key: p.points[0].instance_id for p in beam_paths(facts, w)} == {"beam-1": "SRC", "beam-2": "SRC_A"}
    inputs = [i for i in alignment_items(facts, w) if i.startswith("Bring the input beam of")]
    assert [i.split(" onto ")[0] for i in inputs] == ["Bring the input beam of SRC", "Bring the input beam of SRC_A"]


def offset_return(scene: dict) -> None:
    """L1 entered and left at two points of beam-1 (the axis names it twice)."""

    (axis,) = scene["beam_reference"]["beam_axes"]
    x, y, z = axis["points_mm"][1]
    axis["points_mm"].insert(2, [x + 0.5, y, z])
    axis["through"].insert(2, "L1")


def test_an_element_entered_and_left_at_two_points_is_one_element_of_the_path():
    facts = _facts(edited_preview(offset_return))
    w = Words("en")
    (path,) = beam_paths(facts, w)
    assert [p.instance_id for p in path.points] == ["SRC", "L1", "focal_plane"]
    assert "beam order along beam-1: L1 (B01)." in " ".join(alignment_items(facts, w))
