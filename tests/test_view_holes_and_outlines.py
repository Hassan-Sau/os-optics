"""The 3D view of surfaces with holes through the element and of non-convex clear outlines (``optical_view/region_mesh.py``).

References computed here, independent of the mesher: the shoelace area of the stated rings, the area of a circle, point-in-
polygon tests of the drawn triangles' centroids against the hole rings, and the profile's sag at the drawn vertices.
"""

from __future__ import annotations

import math

import pytest

from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.local import lower_project
from os_optics.optical_view.builder import declared_body
from os_optics.optical_view.optic_mesh import draw_body
from os_optics.optical_view.region_mesh import subdivided, triangulate
from os_optics.optical_view.surface_mesh import CHORD_TOLERANCE_MM

from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import project, source


def shoelace(ring) -> float:
    return 0.5 * abs(sum(ring[i - 1][0] * ring[i][1] - ring[i][0] * ring[i - 1][1] for i in range(len(ring))))


def inside(point, ring) -> bool:
    x, y = point
    hit = False
    for i in range(len(ring)):
        (x0, y0), (x1, y1) = ring[i - 1], ring[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def triangles_2d(mesh):
    """(area, centroid) of every triangle of a mesh, in the element frame's X / Y (a coaxial surface's local X / Y)."""

    out = []
    p = mesh.positions
    for t in range(0, len(mesh.indices), 3):
        a, b, c = p[mesh.indices[t]], p[mesh.indices[t + 1]], p[mesh.indices[t + 2]]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        out.append((area, ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)))
    return out


def drawing_of(model: dict, materials=None):
    document = with_models(project([source(position=(-50.0, 0.0, 0.0)), declared_component("W", "window", model["model_id"])], []),
                           [model], materials)
    (element,) = lower_project(document).elements
    body = declared_body(element, ProjectDocumentV2.model_validate(document))
    return body, draw_body(body)


HOLES = [{"center_mm": ["4", "0"], "radius_mm": "2"}, {"center_mm": ["-3", "-3"], "radius_mm": "1"}]


def test_a_window_with_holes_is_drawn_open_through_them() -> None:
    model = {"model_id": "window", "model": sequence([surface(1, "0", clear="9", holes=HOLES), surface(2, "3", clear="9", holes=HOLES)],
                                                     ["N-BK7"])}
    body, drawing = drawing_of(model, [nbk7()])
    for drawn, stated in zip(drawing.surfaces, body.surfaces, strict=True):
        rings = stated.hole_outlines_mm
        assert len(rings) == 2 and drawn.facts["holes_drawn"] == 2
        clear = [t for material, mesh in drawn.meshes if material not in ("optic_rim",) for t in triangles_2d(mesh)]
        rim = [t for material, mesh in drawn.meshes if material == "optic_rim" for t in triangles_2d(mesh)]
        for _area, centroid in clear + rim:
            assert not any(inside(centroid, ring) for ring in rings)
        holes_area = sum(shoelace(ring) for ring in rings)
        clear_area = sum(a for a, _c in clear)
        assert clear_area + holes_area == pytest.approx(math.pi * 81.0, rel=2e-3)
        assert clear_area + holes_area + sum(a for a, _c in rim) == pytest.approx(math.pi * 100.0, rel=2e-3)
    # the edge wall still meets both faces' outline rows
    assert any(material == "optic_edge" for material, _mesh in drawing.body)


def test_a_holed_lens_is_drawn_on_its_curved_profile_around_the_hole() -> None:
    """A plano-convex lens (R 40 mm) bored through on its axis: its curved face is drawn on the sphere, around the hole."""

    hole = [{"center_mm": ["0", "0"], "radius_mm": "2"}]
    model = {"model_id": "window", "model": sequence([surface(1, "0", radius="40", clear="9", holes=hole), surface(2, "5", clear="9", holes=hole)],
                                                     ["N-BK7"])}
    body, drawing = drawing_of(model, [nbk7()])
    curved = drawing.surfaces[0]
    assert curved.facts["max_deviation_mm"] <= CHORD_TOLERANCE_MM
    ring = body.surfaces[0].hole_outlines_mm[0]
    for material, mesh in curved.meshes:
        if material == "optic_rim":
            continue
        for x, y, z in mesh.positions:
            assert z == pytest.approx(40.0 - math.sqrt(1600.0 - x * x - y * y), abs=1e-9)
        for _area, centroid in triangles_2d(mesh):
            assert math.hypot(*centroid) > 2.0 - CHORD_TOLERANCE_MM and not inside(centroid, ring)


def test_a_non_convex_clear_outline_is_drawn_to_its_shape() -> None:
    corners = [["-6", "-6"], ["6", "-6"], ["6", "-1"], ["-1", "-1"], ["-1", "6"], ["-6", "6"]]  # an L
    shaped = {"clear_shape": {"kind": "POLYGON", "vertices_mm": corners}}
    model = {"model_id": "window", "model": sequence([surface(1, "0", clear="9", **shaped), surface(2, "3", clear="9", **shaped)], ["N-BK7"])}
    _body, drawing = drawing_of(model, [nbk7()])
    polygon = [(float(x), float(y)) for x, y in corners]
    for drawn in drawing.surfaces:
        clear = [t for material, mesh in drawn.meshes if material != "optic_rim" for t in triangles_2d(mesh)]
        assert sum(a for a, _c in clear) == pytest.approx(shoelace(polygon), rel=1e-12)
        for _area, centroid in clear:
            assert inside(centroid, polygon)
        rim = [t for material, mesh in drawn.meshes if material == "optic_rim" for t in triangles_2d(mesh)]
        for _area, centroid in rim:
            assert not inside(centroid, polygon)


def test_ear_clipping_covers_a_polygon_with_holes_exactly() -> None:
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]
    holes = [[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)], [(6.0, 1.0), (8.0, 1.0), (7.0, 3.0)]]
    points, triangles = triangulate(outer, holes)
    expected = shoelace(outer) - sum(shoelace(h) for h in holes)

    def total(points, triangles) -> float:
        return sum(0.5 * ((points[b][0] - points[a][0]) * (points[c][1] - points[a][1])
                          - (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])) for a, b, c in triangles)

    assert total(points, triangles) == pytest.approx(expected, rel=1e-12)  # (counter-clockwise: every area positive)
    assert all(0.5 * ((points[b][0] - points[a][0]) * (points[c][1] - points[a][1])
                      - (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])) > 0.0 for a, b, c in triangles)
    finer, split = subdivided(points, triangles)
    assert len(split) == 4 * len(triangles) and total(finer, split) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("radius", [None, "40"], ids=["window", "plano-convex"])
def test_a_holed_surface_with_its_own_edge_is_drawn_open_through_its_holes_and_flat_beyond_its_edge(radius) -> None:
    """S1 reaches its own edge (8 mm) inside the element's outline (10 mm): its clear aperture (7 mm) minus the holes, its rim
    to the edge, then its flat flange at the edge's sag out to the glass outline. The areas add up to pi 49, pi 64 and pi 100
    (the drawn chords), no triangle lies in a hole, every flange vertex lies at the sag of the edge point, and the edge wall
    still meets both faces' rows."""

    holes = [{"center_mm": ["3", "0"], "radius_mm": "1.5"}, {"center_mm": ["-2", "-2"], "radius_mm": "1"}]
    first = surface(1, "0", radius=radius, clear="7", holes=holes, edge_radius_mm="8")
    model = {"model_id": "window", "model": sequence([first, surface(2, "6", clear="9")], ["N-BK7"])}
    body, drawing = drawing_of(model, [nbk7()])
    drawn, stated = drawing.surfaces[0], body.surfaces[0]
    rings = stated.hole_outlines_mm
    assert len(rings) == 2 and drawn.facts["holes_drawn"] == 2
    parts = {material: [t for m, mesh in drawn.meshes if m == material for t in triangles_2d(mesh)] for material, _ in drawn.meshes}
    flange = parts.pop("optic_edge")
    rim = parts.pop("optic_rim")
    clear = [t for triangles in parts.values() for t in triangles]
    for _area, centroid in clear + rim + flange:
        assert not any(inside(centroid, ring) for ring in rings)
    holes_area = sum(shoelace(ring) for ring in rings)
    clear_area = sum(a for a, _c in clear)
    assert clear_area + holes_area == pytest.approx(math.pi * 49.0, rel=2e-3)
    assert clear_area + holes_area + sum(a for a, _c in rim) == pytest.approx(math.pi * 64.0, rel=2e-3)
    assert clear_area + holes_area + sum(a for a, _c in rim + flange) == pytest.approx(math.pi * 100.0, rel=2e-3)
    for _area, centroid in flange:
        assert 8.0 - 0.01 < math.hypot(*centroid) < 10.0
    held = 0.0 if radius is None else 40.0 - math.sqrt(1600.0 - 64.0)
    for material, mesh in drawn.meshes:
        if material == "optic_edge":
            for _x, _y, z in mesh.positions:
                assert z == pytest.approx(held, abs=1e-9)
    assert any(material == "optic_edge" for material, _mesh in drawing.body)
