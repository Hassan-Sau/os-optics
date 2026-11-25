"""A ``FACET_SOLID`` placed as a member of a ``COMPOUND`` unit traces as the same solid placed directly by its Project pose.

The faces of a facet solid are found from its vertices as written: which facets lie in one plane is judged against the
coordinates' rounding (``optics/surfaces/facets.py``). A member's rigid placement keeps the vertices as written and places
the faces found from them (``MeshSurface.placement``), so a part of rounded or single-precision (float32) coordinates keeps
its faces in a unit. References: the same part placed directly (component pose = unit pose o member pose) gives the same
power, exit direction and position, optical path, polarization, Gaussian beam, notices, traced faces and run summary; at
normal incidence through two parallel faces of a cube the power is the Fresnel transmittance ``(4 n / (n + 1)^2)^2`` of
N-BK7 (its Sellmeier formula) and the direction is unchanged; a placement composed of two placements is the product of
their rigid motions (computed here with matrices)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from os_optics import validate
from os_optics import builders as B
from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics.config import NumericalConfig
from os_optics.contracts.geometry import Pose3D
from os_optics.contracts.optics import MirrorInteraction
from os_optics.local import run_optical_project
from os_optics.local.compile import placed_surface
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.scene.lower import facet_patches, facet_patch_ids, lower_scene

from declared_helpers import compound_model, cylinder, declared_component, nbk7, with_models
from facet_helpers import convex_solid, extruded, glass, part, request, solid
from local_helpers import catalog_document, project, source
from test_facet_solid_project import HIGH_REFLECTOR, _decimal, _nbk7_index

WL = 633.0
BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _rotation(axis, angle_deg: float) -> np.ndarray:
    a = np.array(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(angle_deg)
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def _quaternion(axis, angle_deg: float) -> tuple[float, float, float, float]:
    a = np.array(axis, float) / np.linalg.norm(axis)
    h = math.radians(angle_deg) / 2.0
    return (math.cos(h), *(float(c) * math.sin(h) for c in a))  # type: ignore[return-value]


def _quaternion_of(m: np.ndarray) -> tuple[float, float, float, float]:
    """The unit quaternion of a rotation matrix (largest-component branch)."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return (w, x, y, z)


def _frame_to(d) -> np.ndarray:
    """A rotation whose third column (local +z) is ``d``."""
    d = np.array(d, float) / np.linalg.norm(d)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, d)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(d, x), d])


def _rounded(points: np.ndarray, rounding) -> np.ndarray:
    if rounding == "float32":
        return points.astype(np.float32).astype(float)
    return points if rounding is None else np.round(points, rounding)


def _cube(rounding):
    """A 10 mm cube centred on the origin, turned 17 deg about (1, 2, 3), its corners rounded (CAD export)."""
    corners = np.array([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)], float) - 5.0
    points = _rounded((_rotation((1, 2, 3), 17.0) @ corners.T).T, rounding)
    vertices, triangles, _faces = convex_solid([tuple(p) for p in points], CUBE_FACES)
    return points, vertices, triangles


def _model(model_id: str, vertices, triangles, **fields) -> dict:
    model = {"model_kind": "FACET_SOLID", "vertices_mm": [[_decimal(c) for c in v] for v in vertices],
             "triangles": [list(t) for t in triangles], "approximation": "FACETED_EXACT", "medium": "N-BK7",
             "valid_band_nm": dict(BAND), "status": "NOMINAL"}
    model.update(fields)
    return {"model_id": model_id, "model": model}


# unit pose U and member pose M (the member turned 11 deg about Z and moved by (0.1, 0.2, 0.3) mm in the unit)
UNIT_POSES = {"unit at the origin": (IDENTITY, (0.0, 0.0, 0.0)),
              "unit turned and moved": (_quaternion((-1.0, 0.5, 2.0), 23.0), (2.5, -1.25, 0.75))}
MEMBER = (_quaternion((0.0, 0.0, 1.0), 11.0), (0.1, 0.2, 0.3))


def _compose(outer, inner):
    """``outer o inner`` of two ``(quaternion, position)`` poses, by matrices."""
    (qa, pa), (qb, pb) = outer, inner
    ra, rb = (np.array(PoseV1(quaternion_wxyz=q).quaternion_wxyz) for q in (qa, qb))  # (validated as the contract takes them)
    ma, mb = _matrix(ra), _matrix(rb)
    return _quaternion_of(ma @ mb), tuple(float(c) for c in ma @ np.array(pb) + np.array(pa))


def _matrix(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in np.array(q) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _through_face(points: np.ndarray, face: list[int]):
    """``(beam start, beam direction, plane point, plane normal)`` in the part's frame: a beam along the inward normal of
    ``face`` at its centre from 40 mm outside, an observation plane 40 mm beyond the face, facing along the beam."""
    centre = points[face].mean(axis=0)
    normal = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, centre - points.mean(axis=0)) < 0.0:
        normal = -normal  # outward
    return centre + 40.0 * normal, -normal, centre - 40.0 * normal, -normal


def _documents(model: dict, beam, unit_pose, member_pose, *, array: dict | None = None, cell_offset=(0.0, 0.0, 0.0), coatings=None):
    """The part placed directly (pose ``unit o member``) and as the one member of a unit, with the same global source and
    observation plane: ``beam`` = ``(start, direction, plane point, plane normal)`` in the part's frame."""
    q_direct, p_direct = _compose(unit_pose, (member_pose[0], tuple(np.array(member_pose[1]) + np.array(cell_offset))))
    m = _matrix(q_direct)
    start, direction, at, facing = (m @ np.array(beam[0]) + np.array(p_direct), m @ np.array(beam[1]), m @ np.array(beam[2]) + np.array(p_direct),
                                    m @ np.array(beam[3]))
    q_beam, q_plane = _quaternion_of(_frame_to(direction)), _quaternion_of(_frame_to(facing))
    laser = source(position=tuple(float(c) for c in start), wavelength_nm=WL, waist_mm=0.3, quaternion=q_beam, power_w=1e-3)
    back = {"observation_id": "back", "pose": {"position_mm": [float(c) for c in at], "quaternion_wxyz": list(q_plane)},
            "aperture": {"shape": "circle", "radius_mm": 25.0}}
    materials = [nbk7()]
    direct = with_models(project([laser, declared_component("C", "prism", model["model_id"], position=p_direct, quaternion=q_direct)], [back]),
                         [model], materials, coatings)
    member = ("M1", model["model_id"], member_pose[1], member_pose[0])
    unit = compound_model("UNIT", [member], cylinder("120", "-60", "60"))
    if array is not None:
        unit["model"]["members"][0]["array"] = array
    compound = with_models(project([laser, declared_component("U", "prism", "UNIT", position=unit_pose[1], quaternion=unit_pose[0])], [back]),
                           [model, unit], materials, coatings)
    return direct, compound, direction


def _arrival(run):
    (obs,) = [o for o in run.result.observations if o.observation_id == "back"]
    (arrival,) = obs.arrivals
    return arrival


def _facets_row(run) -> dict:
    rows = run.summary()["element_models"]
    for row in rows:
        if "facets" in row:
            return row["facets"]
        for member in row.get("members", ()):
            if "facets" in member:
                return member["facets"]
    raise AssertionError("no facet solid in the run summary")


def _traced_face_counts(run) -> set[int]:
    """The number of planar patches the trace takes on each facet surface of the part (one per cell of an array)."""
    (component,) = [c for c in run.request.project.components if c.instance_id in ("C", "U")]
    return {len(facet_patch_ids(component.instance_id, surface)) for surface in component.surfaces}


def _field(beam) -> np.ndarray:
    j = beam.polarization
    ex, ey = complex(*j.ex), complex(*j.ey)
    return ex * np.array(beam.basis.e1) + ey * np.array(beam.basis.e2)


def _assert_same_trace(direct, compound):
    a, b = _arrival(direct), _arrival(compound)
    assert b.beam.power_W == pytest.approx(a.beam.power_W, rel=1e-12)
    assert np.allclose(b.beam.direction, a.beam.direction, atol=1e-12, rtol=0)
    assert np.allclose(b.beam.position_mm, a.beam.position_mm, atol=1e-9, rtol=0)
    assert b.beam.optical_path_length_mm == pytest.approx(a.beam.optical_path_length_mm, rel=1e-12)
    # the field direction (Jones vector in its basis, as a 3D complex vector): near-normal incidence leaves the plane of
    # incidence ill-conditioned, so the two placements' last-bit differences in the face normals show at 1e-12
    assert np.allclose(_field(b.beam), _field(a.beam), atol=1e-11, rtol=0)
    assert a.beam.gaussian.valid and b.beam.gaussian.valid
    for field in ("q_x_mm", "q_y_mm"):
        assert np.allclose(getattr(b.beam.gaussian, field), getattr(a.beam.gaussian, field), rtol=1e-9, atol=0)
    # the same faces met (patch ids are the facet solid's own: the member prefix aside), the same notices
    assert [s.split("facets.")[1] for s in b.lineage] == [s.split("facets.")[1] for s in a.lineage]
    assert Counter((i.code.value, i.status.value) for i in compound.result.issues) == Counter((i.code.value, i.status.value) for i in direct.result.issues)
    assert compound.status == direct.status
    # the run summary states the faces the trace takes
    assert _facets_row(compound) == _facets_row(direct)
    assert _traced_face_counts(compound) == _traced_face_counts(direct) == {_facets_row(direct)["traced_faces"]}
    return a, b


@pytest.mark.parametrize("unit_pose", list(UNIT_POSES), ids=list(UNIT_POSES))
@pytest.mark.parametrize("rounding", [8, 6, 4, "float32"])
def test_a_rounded_cube_traces_the_same_as_a_compound_member_and_placed_directly(rounding, unit_pose):
    points, vertices, triangles = _cube(rounding)
    direct_doc, compound_doc, d = _documents(_model("CUBE", vertices, triangles), _through_face(points, CUBE_FACES[0]), UNIT_POSES[unit_pose], MEMBER)
    direct, compound = run_optical_project(direct_doc), run_optical_project(compound_doc)
    a, _b = _assert_same_trace(direct, compound)
    # independent references: six faces, two Fresnel transmissions at normal incidence, direction kept, no facet edge met
    assert _facets_row(direct)["traced_faces"] == 6
    n = _nbk7_index(WL)
    t = 4.0 * n / (n + 1.0) ** 2
    assert a.beam.power_W == pytest.approx(1e-3 * t * t, rel=1e-9)
    assert np.allclose(a.beam.direction, d, atol=1e-4, rtol=0)  # (the rounded faces' planes tilt by about rounding / size)
    codes = {i.code.value for i in compound.result.issues}
    assert "FACET_EDGE_DIVIDES_BEAM" not in codes and "GAUSSIAN_UNDEFINED_AFTER_FACETS" not in codes
    assert compound.result.power_budget.conserved


def test_a_cell_of_an_arrayed_compound_member_traces_as_the_part_placed_directly():
    points, vertices, triangles = _cube(6)
    array = {"layout": "RECTANGULAR", "pitch_mm": ["30", "30"], "counts": [2, 1]}
    direct_doc, compound_doc, _d = _documents(_model("CUBE", vertices, triangles), _through_face(points, CUBE_FACES[2]),
                                              UNIT_POSES["unit turned and moved"], MEMBER, array=array, cell_offset=(15.0, 0.0, 0.0))
    direct, compound = run_optical_project(direct_doc), run_optical_project(compound_doc)
    _a, b = _assert_same_trace(direct, compound)
    assert all(step.startswith("U/M1_0_1-facets.t") for step in b.lineage)


def test_coated_mirror_faces_of_a_placed_penta_prism_keep_their_coating_and_geometry():
    """An N-BK7 penta prism with a high reflector (R 0.99) on its two mirrored faces, coordinates rounded to 5 decimals, as
    a turned unit's member: the light reflects on the coated faces (the patches of the mirrored facet group) and leaves
    with ``T0^2 R^2`` of its power, as the part placed directly."""
    s, b = 10.0, 4.0
    c, sn = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    a = (s - b / math.sqrt(2.0)) / (c - sn)
    corners = [(0.0, 0.0), (s, 0.0), (s + a * sn, a * c), (s + a * sn - b / math.sqrt(2.0), a * c + b / math.sqrt(2.0)), (0.0, s)]
    vertices, triangles, faces = extruded(corners, 8.0)
    turn = _rotation((0.3, -1.0, 0.4), 31.0)
    points = _rounded(np.array(vertices) @ turn.T, 5)
    vertices = [tuple(p) for p in points]
    mirrors = faces[3] + faces[5]
    model = _model("PP", vertices, triangles, facet_groups=[{"triangles": mirrors, "traced": "REFLECTED", "coating_id": HIGH_REFLECTOR}])
    records = {c["coating_id"]: c for c in catalog_document()["coatings"]}
    # in the prism's own frame light enters the z = 0 face along +z at (3, 0.5) and leaves the x = 0 face along -x
    beam = (turn @ np.array([3.0, 0.5, -40.0]), turn @ np.array([0.0, 0.0, 1.0]), turn @ np.array([-40.0, 0.5, 5.0]), turn @ np.array([-1.0, 0.0, 0.0]))
    direct_doc, compound_doc, _d = _documents(model, beam, UNIT_POSES["unit turned and moved"], MEMBER, coatings=[records[HIGH_REFLECTOR]])
    direct, compound = run_optical_project(direct_doc), run_optical_project(compound_doc)
    for run in (direct, compound):
        assert run.result.power_budget.conserved
    out_direct, out_compound = _arrival(direct), _arrival(compound)
    assert [step.split(":")[1] for step in out_compound.lineage] == ["refract", "reflect", "reflect", "refract"]
    n = _nbk7_index(WL)
    t0 = 4.0 * n / (n + 1.0) ** 2
    assert out_compound.beam.power_W == pytest.approx(1e-3 * t0 * t0 * 0.99 * 0.99, rel=1e-6)
    assert out_compound.beam.power_W == pytest.approx(out_direct.beam.power_W, rel=1e-12)
    assert np.allclose(out_compound.beam.direction, out_direct.beam.direction, atol=1e-12, rtol=0)
    # the reflections take place on patches of the coated mirror group, with its coating
    (component,) = [c for c in compound.request.project.components if c.instance_id == "U"]
    (surface,) = component.surfaces
    interactions = dict(facet_patches("U", surface))
    for step in out_compound.lineage:
        surface_id, event = step.split(":")
        inter = interactions[surface_id]
        assert isinstance(inter, MirrorInteraction) == (event == "reflect")
        if event == "reflect":
            assert inter.coating is not None
    assert _facets_row(compound) == _facets_row(direct)


# ------------------------------------------------------------------------------------------------ the Engine contract


def _ray_through(surface, at, direction, *, back: float = 40.0):
    d = np.array(direction, float) / np.linalg.norm(direction)
    laser = B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=0.3, position=tuple(np.array(at, float) - back * d), direction=tuple(d))
    plane = B.observation_plane("o", position=tuple(float(x) for x in np.array(at, float) + back * d), normal=tuple(d))
    result = validate(request(part("P", surface), laser, observations=(plane,)))
    (obs,) = result.observations
    (arrival,) = obs.arrivals
    return arrival


def test_a_placed_facet_surface_traces_as_its_vertices_moved_by_the_placement():
    """Full-precision coordinates (the faces are the same either way): a facet surface with a ``placement`` traces as the
    surface whose vertices are moved by it."""
    points, _vertices, _triangles = _cube(None)
    vertices, triangles, _faces = convex_solid([tuple(p) for p in points], CUBE_FACES)
    q, p = _quaternion((0.2, 1.0, -0.7), 41.0), (1.5, -2.0, 0.25)
    m = _matrix(q)
    moved = [tuple(float(c) for c in m @ np.array(v) + np.array(p)) for v in vertices]
    placed = solid("cube", vertices, triangles).model_copy(update={"placement": Pose3D(position_mm=p, quaternion_wxyz=q)})
    at, direction = np.array(p) + m @ np.array([0.3, -0.8, 0.0]), m @ np.array([0.1, 0.05, 1.0])
    a, b = _ray_through(placed, at, direction), _ray_through(solid("cube", moved, triangles), at, direction)
    assert b.beam.power_W == pytest.approx(a.beam.power_W, rel=1e-12)
    assert np.allclose(a.beam.direction, b.beam.direction, atol=1e-12, rtol=0)
    assert np.allclose(a.beam.position_mm, b.beam.position_mm, atol=1e-9, rtol=0)
    assert a.beam.optical_path_length_mm == pytest.approx(b.beam.optical_path_length_mm, rel=1e-12)
    assert [s.split(":")[0] for s in a.lineage] == [s.split(":")[0] for s in b.lineage]
    assert "placement" not in solid("cube", moved, triangles).model_dump(mode="json")  # (omitted when absent)


def test_placing_a_placed_facet_surface_composes_the_two_placements():
    points, vertices, triangles = _cube(6)
    inner = PoseV1(position_mm=(0.1, 0.2, 0.3), quaternion_wxyz=MEMBER[0])
    outer = PoseV1(position_mm=(2.5, -1.25, 0.75), quaternion_wxyz=tuple(2.0 * c for c in UNIT_POSES["unit turned and moved"][0]))  # (not unit)
    twice = placed_surface(placed_surface(solid("cube", vertices, triangles), inner), outer)
    assert twice.vertices_mm == tuple(tuple(v) for v in vertices)  # the vertices as written
    q, p = _compose((outer.quaternion_wxyz, outer.position_mm), (inner.quaternion_wxyz, inner.position_mm))
    assert np.allclose(_matrix(twice.placement.quaternion_wxyz), _matrix(q), atol=1e-15, rtol=0)
    assert np.allclose(twice.placement.position_mm, p, atol=1e-14, rtol=0)
    assert math.isclose(math.fsum(c * c for c in twice.placement.quaternion_wxyz), 1.0, abs_tol=1e-15)
    # lowered: six faces, each the plane of its corners moved by the composed rigid motion
    cfg = NumericalConfig()
    scene, _poses = lower_scene(request(part("P", twice)).project, cfg, MaterialLibrary((glass(),), cfg))
    planes = [s for s in scene.surfaces if s.owner_id == "P"]
    assert len(planes) == 6
    m = _matrix(q)
    moved = points @ m.T + np.array(p)
    for face in CUBE_FACES:
        normal = np.cross(moved[face[1]] - moved[face[0]], moved[face[2]] - moved[face[0]])
        normal /= np.linalg.norm(normal)
        centre = moved[face].mean(axis=0)
        if np.dot(normal, centre - np.array(p)) < 0.0:
            normal = -normal
        (match,) = [s for s in planes if float(np.dot(s.normal, normal)) > 1.0 - 1e-9]
        assert abs(float(np.dot(normal, np.array(match.origin) - centre))) < 1e-5  # (6-decimal corners: their face's plane within the rounding)


ANTI_REFLECTION = "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10"


def test_the_coating_validity_of_every_face_of_a_placed_member_is_checked_as_for_the_part_placed_directly():
    """A rounded cube with an anti-reflection coating characterized for 0-10 deg, met at 20 deg on a face centre: each
    face the beam crosses (entry and exit) is outside the coating's angles, in the unit as placed directly; the power is
    the coating's ``0.995^2`` (the stated transmittance, used as stated), and no face is taken for an untraced edge."""
    points, vertices, triangles = _cube(6)
    face = CUBE_FACES[0]
    _start, d, _at, _facing = _through_face(points, face)
    centre = points[face].mean(axis=0)
    axis = np.cross(d, (0.3, 0.9, 0.1))
    tilted = _rotation(axis, 20.0) @ d
    beam = (centre - 40.0 * tilted, tilted, centre + 40.0 * tilted, tilted)
    records = {c["coating_id"]: c for c in catalog_document()["coatings"]}
    model = _model("CUBE", vertices, triangles, coating_id=ANTI_REFLECTION)
    direct_doc, compound_doc, _d = _documents(model, beam, UNIT_POSES["unit turned and moved"], MEMBER, coatings=[records[ANTI_REFLECTION]])
    runs = {name: run_optical_project(doc) for name, doc in (("direct", direct_doc), ("compound", compound_doc))}
    found = {}
    for name, run in runs.items():
        assert _arrival(run).beam.power_W == pytest.approx(1e-3 * 0.995 ** 2, rel=1e-12)
        found[name] = sorted((i.code, i.message.split(" surface ")[1].split(" ")[0].removeprefix("M1-")) for i in run.model_scope_issues)
    assert found["compound"] == found["direct"]
    assert [code for code, _face in found["direct"]] == ["COATING_AOI_OUTSIDE_RANGE"] * 2
    assert {face for _code, face in found["direct"]} == {s.split("/")[1].split(":")[0] for s in _arrival(runs["direct"]).lineage}
