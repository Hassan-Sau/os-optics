"""The seat step of an optic whose two faces differ (``PreviewPartV1.faces``) says which face meets the
incoming beam, derived from the optic's pose and the direction of the beam axis that names it; an optic with alike faces
(or a scene that does not state them) gets no such clause.

The reference lens is a plano-convex lens with its convex face at its origin (``front``), its local +Z along the beam:
the beam meets the convex face first.
"""

from __future__ import annotations

from drawing_helpers import edited_preview, preview, rebound_detail

from os_optics.assembly_drawing import generate_assembly_drawing

PLANO_CONVEX = {"front": {"shape": "CONVEX", "radius_mm": 51.5}, "back": {"shape": "PLANO"}}


def _lens(scene: dict) -> dict:
    (lens,) = [p for p in scene["parts"] if p["category"] == "OPTICAL_ELEMENT"]
    return lens


def _seat(package, locale: str = "en") -> str:  # noqa: ANN001
    drawing = generate_assembly_drawing(package.archive_bytes(), rebound_detail(package), locale=locale)
    lens = next(p for p in package.scene.parts if p.category.value == "OPTICAL_ELEMENT")
    (step,) = [s.text for s in drawing.result.assembly_steps if s.part_ref == lens.part_ref]
    return step


def _travel(package) -> str:  # noqa: ANN001
    (axis,) = package.scene.beam_reference.beam_axes
    d = [b - a for a, b in zip(axis.points_mm[0], axis.points_mm[1], strict=True)]
    k = max(range(3), key=lambda i: abs(d[i]))
    return ("+" if d[k] > 0 else "-") + "XYZ"[k]


def _opposite(axis: str) -> str:
    return ("-" if axis[0] == "+" else "+") + axis[1]


def test_the_reference_lens_meets_the_beam_with_its_convex_face():
    package = preview()
    assert _lens(package.scene.payload())["faces"] == PLANO_CONVEX
    back = _opposite(_travel(package))
    assert f"; turn it with its convex face toward the incoming beam (toward world {back});" in _seat(package)
    assert f"；使其凸面朝向入射光束（朝向世界坐标 {back}）；" in _seat(package, "zh-CN")


def test_the_same_lens_turned_round_meets_it_with_its_flat_face():
    """The faces swapped in the lens's frame is the lens turned round in the same place: the other face meets the beam."""

    def turned(scene: dict) -> None:
        faces = _lens(scene)["faces"]
        faces["front"], faces["back"] = faces["back"], faces["front"]

    package = edited_preview(turned)
    back = _opposite(_travel(package))
    assert f"turn it with its flat face toward the incoming beam (toward world {back})" in _seat(package)
    assert f"使其平面朝向入射光束（朝向世界坐标 {back}）" in _seat(package, "zh-CN")
    assert _seat(package) != _seat(preview())


def test_two_curved_faces_are_told_apart_by_their_radius():
    def biconvex(scene: dict) -> None:
        _lens(scene)["faces"] = {"front": {"shape": "CONVEX", "radius_mm": 60.0}, "back": {"shape": "CONVEX", "radius_mm": 200.0}}

    assert "turn it with its convex face of radius 60 mm toward the incoming beam" in _seat(edited_preview(biconvex))
    assert "使其半径 60 mm 的凸面朝向入射光束" in _seat(edited_preview(biconvex), "zh-CN")


def test_without_a_beam_axis_naming_the_lens_its_front_face_direction_is_stated():
    def unnamed(scene: dict) -> None:
        for axis in scene["beam_reference"]["beam_axes"]:
            axis.pop("through", None)

    package = edited_preview(unnamed)
    back = _opposite(_travel(package))  # the lens's local +Z runs along the beam: its front face looks back along it
    step = _seat(package)
    assert f"turn it with its convex face toward world {back};" in step and "incoming beam" not in step


def test_a_lens_with_alike_faces_gets_no_face_clause():
    def alike(scene: dict) -> None:
        _lens(scene).pop("faces")

    step = _seat(edited_preview(alike))
    assert "turn it with its" not in step
    assert "使其" not in _seat(edited_preview(alike), "zh-CN")
