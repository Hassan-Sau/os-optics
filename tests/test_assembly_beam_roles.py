"""The assembly presentations draw a beam axis's SECONDARY stretches (light that only weakly reaches an arm:
the leak of a polarizing splitter, a dichroic's leak) as the Optical Simulation View does: thinner and fainter.

* the Experimental Lab Assembly View (``viewer.assembly_view_scene``): ``SECONDARY`` runs beside the ``DESIGN_AXIS`` runs;
* the Detailed Assembly Drawing: the stretch in the thin, short-dashed beam style (``S_BEAM_SECONDARY``); its spacing
  rows and the beam order are unchanged (the weak arm is still light on the table);
* an axis without ``segment_roles`` (every package written before them) is drawn as before.
"""

from __future__ import annotations

from drawing_helpers import edited_preview, preview, rebound_detail

from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.assembly_drawing.canvas import S_BEAM_SECONDARY
from os_optics.viewer import assembly_view_scene

DASH = "stroke-dasharray=\"" + " ".join(f"{v:g}" for v in S_BEAM_SECONDARY.dash) + "\""


def secondary_tail(scene: dict) -> None:
    """The reference beam's last stretch (lens -> focal plane) only weakly lit, and noted (scene 3.1 fields)."""

    scene["schema_version"] = "3.1"
    (axis,) = scene["beam_reference"]["beam_axes"]
    axis["segment_roles"] = ["PRIMARY"] * (len(axis["points_mm"]) - 2) + ["SECONDARY"]
    axis["notes"] = ["POLARIZATION_DEPENDENT"]


def _svgs(drawing) -> str:
    return "".join(data.decode("utf-8") for name, data in drawing.files.items() if name.endswith(".svg"))


def test_the_assembly_view_draws_a_secondary_stretch_as_secondary():
    plain = assembly_view_scene(preview())
    (beam,) = plain.beams
    assert [role for role, _ in beam.runs] == ["DESIGN_AXIS"] and "notes" not in beam.facts
    marked = assembly_view_scene(edited_preview(secondary_tail))
    (beam,) = marked.beams
    assert [role for role, _ in beam.runs] == ["DESIGN_AXIS", "SECONDARY"]
    assert beam.runs[0][1][-1].tolist() == beam.runs[1][1][0].tolist()  # the runs meet at the lens
    assert beam.facts["notes"] == ["POLARIZATION_DEPENDENT"]


def test_the_drawing_draws_a_secondary_stretch_thinner_and_keeps_its_spacing():
    package = edited_preview(secondary_tail)
    plain = generate_assembly_drawing(preview().archive_bytes(), rebound_detail(preview()))
    marked = generate_assembly_drawing(package.archive_bytes(), rebound_detail(package))
    assert DASH not in _svgs(plain).replace(",", " ")
    assert DASH in _svgs(marked).replace(",", " ")
    rows = lambda d: [(r.path_key, r.from_instance_id, r.to_instance_id) for r in d.result.optical_spacing]  # noqa: E731
    assert rows(marked) == rows(plain)
