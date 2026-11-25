"""A closed facet surface the trace follows bounds solids whose sides hold (``optics/surfaces/facet_topology.py``).

Several separate shells, a hollow solid (the cavity's facets face into it) and a solid inside a cavity are traced: a beam
along a line through them crosses every face once, each a Fresnel transmission at normal incidence, ``(4 n / (n + 1)^2)``
per face (independent reference). Shells one inside the other that both face outward, a lone shell facing inward,
overlapping or touching shells, a face pushed through another and a surface that touches itself at a corner are refused
before the trace (``FACET_SURFACE_NOT_TRACED``, OUT_OF_MODEL_SCOPE) with the reason, not met at run time as a medium
mismatch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import validate
from os_optics import builders as B
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.optics.surfaces.facet_topology import facets_cross, shells, solid_problem
from os_optics.optics.surfaces.facets import planar_patches

from facet_helpers import GLASS, N_GLASS, convex_solid, extruded, part, request, solid, unit

# This topology/Fresnel reference uses a test-only ideal glass with known zero bulk loss.
IDEAL_GLASS = ConstantIndexMaterial(material_id=GLASS, refractive_index=N_GLASS, valid_wavelength_nm=(400.0, 1100.0),
                                    absorption_coefficient_per_mm=0.0)
WL = 633.0
T_FACE = 4.0 * N_GLASS / (N_GLASS + 1.0) ** 2  # Fresnel power transmittance at normal incidence, either way
CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def _cube(centre=(0.0, 0.0, 0.0), size: float = 10.0):
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    points = [tuple(centre[k] + size * (c[k] - 0.5) for k in range(3)) for c in corners]
    vertices, triangles, _faces = convex_solid(points, CUBE_FACES)
    return list(vertices), [tuple(t) for t in triangles]


def _joined(*meshes, inward=()):
    """One vertex list and triangle list of several meshes; the meshes numbered in ``inward`` turned to face inward."""
    vertices, triangles = [], []
    for k, (v, t) in enumerate(meshes):
        offset = len(vertices)
        vertices += list(v)
        triangles += [(a + offset, c + offset, b + offset) if k in inward else (a + offset, b + offset, c + offset) for a, b, c in t]
    return tuple(vertices), tuple(triangles)


def _problem(vertices, triangles):
    return solid_problem(vertices, triangles, [p.triangles for p in planar_patches(vertices, triangles)])


def _through(mesh, *, start=(0.3, 0.2, -40.0), direction=(0.0, 0.0, 1.0)):
    vertices, triangles = mesh
    d = unit(direction)
    laser = B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=0.2, position=start, direction=tuple(d))
    plane = B.observation_plane("o", position=tuple(float(x) for x in np.array(start) + 80.0 * d), normal=tuple(d))
    return validate(request(laser, part("P", solid("body", vertices, triangles)), observations=(plane,), materials=(IDEAL_GLASS,)))


@pytest.mark.parametrize(("name", "mesh", "faces_met"), [
    ("two separate cubes on the beam", _joined(_cube((0, 0, -10)), _cube((0, 0, 10))), 4),
    ("a hollow cube", _joined(_cube(), _cube(size=4.0), inward=(1,)), 4),
    ("a cube inside the cavity of a hollow cube", _joined(_cube(size=12.0), _cube(size=6.0), _cube(size=2.0), inward=(1,)), 6),
])
def test_separate_hollow_and_nested_solids_are_traced_face_by_face(name, mesh, faces_met):
    assert _problem(*mesh) is None
    result = _through(mesh)
    # (six parallel faces at normal incidence return ghost beams between every pair: the trace may reach its path limit,
    # MAX_PATHS, which ends the run's search for further ghosts, not the direct beam)
    assert result.outcome.execution.termination == "COMPLETED" or "MAX_PATHS" in {i.code.value for i in result.issues}, name
    (obs,) = result.observations
    (arrival,) = [a for a in obs.arrivals if all(step.endswith(":refract") for step in a.lineage)]
    assert [step.split(":")[1] for step in arrival.lineage] == ["refract"] * faces_met
    assert arrival.beam.power_W == pytest.approx(1e-3 * T_FACE ** faces_met, rel=1e-12)
    assert np.allclose(arrival.beam.direction, (0.0, 0.0, 1.0), atol=1e-15, rtol=0)
    assert "MEDIUM_MISMATCH" not in {i.code.value for i in result.issues}
    assert result.power_budget.conserved


def _pinched():
    a, b = _cube(), _cube((10.0, 10.0, 10.0))
    vertices = a[0] + b[0]
    triangles = a[1] + [tuple(x + 8 for x in t) for t in b[1]]
    # (corner 6 of the first cube, at (5, 5, 5), is corner 0 of the second: one vertex)
    return tuple(vertices), tuple(tuple(6 if x == 8 else x for x in t) for t in triangles)


def _pushed_through():
    vertices, triangles = _cube()
    vertices[6] = (5.0, 5.0, -8.0)  # (the top corner below the bottom face)
    return tuple(vertices), tuple(triangles)


@pytest.mark.parametrize(("name", "mesh", "reason"), [
    ("shells one inside the other, both facing outward", _joined(_cube(), _cube(size=4.0)), "faces outward and lies inside 1 other shell"),
    ("a lone shell facing inward beside a solid", _joined(_cube(size=20.0), _cube((40.0, 0.0, 0.0), 4.0), inward=(1,)),
     "faces inward and lies inside no other shell"),
    ("two overlapping cubes", _joined(_cube(), _cube((5.0, 3.0, 2.0))), "cross or overlap each other"),
    ("two cubes touching along a face", _joined(_cube(), _cube((10.0, 0.0, 0.0))), "cross or overlap each other"),
    ("a corner pushed through the opposite face", _pushed_through(), "cross or overlap each other"),
    ("two cubes sharing one corner vertex", _pinched(), "touch themselves at vertex 6"),
])
def test_a_surface_that_does_not_bound_one_consistent_solid_is_refused_with_its_reason(name, mesh, reason):
    vertices, triangles = mesh
    problem = _problem(vertices, triangles)
    assert problem is not None and reason in problem, name
    result = _through(mesh, start=(0.3, 0.2, -40.0))
    assert result.status.value == "OUT_OF_MODEL_SCOPE" and result.outcome.execution.termination == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in result.issues if i.code.value == "FACET_SURFACE_NOT_TRACED"]
    assert "do not bound one solid" in issue.message and reason in issue.message
    assert issue.location == "component:P/surface:body"


def test_the_shells_of_a_surface_are_its_edge_connected_facets():
    vertices, triangles = _joined(_cube(), _cube((30.0, 0.0, 0.0)), _cube((0.0, 30.0, 0.0)))
    assert [len(s) for s in shells(triangles)] == [12, 12, 12]
    assert shells(triangles)[1][0] == 12


def test_facets_crossing_is_judged_on_the_common_part_of_the_two_triangles():
    """Independent cases: triangles crossing through each other's interior, meeting along a line through a shared corner,
    touching at a point, sharing a corner in one plane side by side, and overlapping in one plane."""
    contact = 1e-9
    base = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0))
    through = ((1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (3.0, -2.0, 0.0))  # its edge from z -1 to z +1 pierces the base at (1, 1)
    assert facets_cross(base, through, contact)
    assert not facets_cross(base, ((1.0, 1.0, 0.5), (1.0, 2.0, 1.0), (2.0, 1.0, 1.0)), contact)  # above it
    assert not facets_cross(base, ((1.0, 1.0, 0.0), (1.0, 2.0, 1.0), (2.0, 1.0, 1.0)), contact)  # touches it at one point
    # one shared corner: a fold along a line from it crosses; a separate angle around it does not
    assert facets_cross(base, ((0.0, 0.0, 0.0), (3.0, 1.0, 1.0), (1.0, 3.0, -1.0)), contact, corner=(0.0, 0.0, 0.0))
    assert not facets_cross(base, ((0.0, 0.0, 0.0), (-3.0, 1.0, 1.0), (-1.0, 3.0, -1.0)), contact, corner=(0.0, 0.0, 0.0))
    # in one plane, sharing a corner: side by side, or one inside the other's angle
    assert not facets_cross(base, ((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (-4.0, 0.0, 0.0)), contact, corner=(0.0, 0.0, 0.0))
    assert facets_cross(base, ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 2.0, 0.0)), contact, corner=(0.0, 0.0, 0.0))
    # in one plane, no shared corner: overlapping, or apart
    assert facets_cross(base, ((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (1.0, 5.0, 0.0)), contact)
    assert not facets_cross(base, ((3.0, 3.0, 0.0), (7.0, 3.0, 0.0), (3.0, 7.0, 0.0)), contact)


def test_rounded_and_many_sided_solids_pass_the_solid_check():
    sides = 96
    ring = [(5.0 * math.cos(2 * math.pi * k / sides), 5.0 * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices, triangles, _faces = extruded(ring, 6.0)
    rounded = tuple(tuple(round(c, 6) for c in v) for v in vertices)
    assert _problem(rounded, triangles) is None
    cube = np.round(np.array(_cube()[0]) @ np.array([[0.96, -0.28, 0.0], [0.28, 0.96, 0.0], [0.0, 0.0, 1.0]]).T, 4)
    assert _problem(tuple(tuple(float(c) for c in row) for row in cube), tuple(_cube()[1])) is None
