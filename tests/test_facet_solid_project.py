"""A Project's declared FACET_SOLID end to end: the contract (``FacetSolidModelV1``) compiles to one Engine facet surface
(``local/compile.py``), the run traces its facets, the results name the facet faces, and the Optical Simulation View draws
the facets (3D and section). References: a solid N-BK7 corner cube retroreflects (exit direction -input, power the Fresnel
transmittance of N-BK7 at entry and exit, from its Sellmeier formula); a penta prism with two mirrored faces deviates by
90 degrees; a tessellated rod states that the Gaussian beam is not defined behind its facets."""

from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics.local import json_api, run_optical_project
from os_optics._vendor.shared_contracts.common.decimal_string import decimal_to_string
from os_optics.optical_view.glb import parse_glb, scene_glb_bytes
from os_optics.optical_view.optic_mesh import draw_body
from os_optics.optical_view.section import element_section

from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, with_models
from facet_helpers import convex_solid, extruded
from local_helpers import plane, project, source

BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
WAVELENGTH_NM = 632.8
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _decimal(x: float) -> str:
    return "0" if x == 0.0 else decimal_to_string(Decimal(repr(float(x))))


def _solid_model(model_id: str, vertices, triangles, **fields) -> dict:
    model = {"model_kind": "FACET_SOLID", "vertices_mm": [[_decimal(c) for c in v] for v in vertices],
             "triangles": [list(t) for t in triangles], "approximation": "FACETED_EXACT", "medium": "N-BK7",
             "valid_band_nm": dict(BAND), "status": "NOMINAL"}
    model.update(fields)
    return {"model_id": model_id, "model": model}


def _nbk7_index(wavelength_nm: float) -> float:
    """N-BK7 from its Sellmeier formula (the independent reference of the declared material)."""
    l2 = (wavelength_nm / 1000.0) ** 2
    b = [float(x) for x in NBK7_SELLMEIER["b"]]
    c = [float(x) for x in NBK7_SELLMEIER["c_um2"]]
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _rotation_to(a, b) -> np.ndarray:
    a, b = np.array(a, float) / np.linalg.norm(a), np.array(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _corner_cube_design():
    rotation = _rotation_to((1, 1, 1), (-1, 0, 0))  # the front face looks along -x: light travelling +x enters it
    corners = np.array([(0, 0, 0), (20, 0, 0), (0, 20, 0), (0, 0, 20)], float) @ rotation.T
    vertices, triangles, _faces = convex_solid([tuple(c) for c in corners], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    entry = np.array([9.0, 7.0, 4.0]) @ rotation.T
    document = project([source(position=(-60.0, float(entry[1]), float(entry[2])), wavelength_nm=WAVELENGTH_NM, waist_mm=0.2),
                        declared_component("CC", "prism", "CC", quaternion=IDENTITY)],
                       [plane("back", (-30.0, 0.0, 0.0), radius_mm=25.0)])
    return with_models(document, [_solid_model("CC", vertices, triangles, source="corner cube, CAD export")], [nbk7()])


def _arrival(run, observation_id: str, crossing: int):
    (obs,) = [o for o in run.result.observations if o.observation_id == observation_id]
    (arrival,) = [a for a in obs.arrivals if a.crossing_index == crossing]
    return arrival


def test_a_declared_corner_cube_retroreflects_through_the_project_path():
    document = _corner_cube_design()
    checked = json_api.check_project(document)
    assert checked["ok"]
    assert checked["declared_models"] == [{"instance_id": "CC", "optical_model_id": "CC", "model_kind": "FACET_SOLID",
                                          "engine_surfaces": ["facets"], "engine_materials": ["project:N-BK7"]}]
    run = run_optical_project(document)
    assert run.result.outcome.execution.termination == "COMPLETED"
    back = _arrival(run, "back", 1)
    assert np.allclose(back.beam.direction, (-1.0, 0.0, 0.0), atol=1e-12, rtol=0)
    assert [step.split(":")[1] for step in back.lineage] == ["refract", "tir", "tir", "tir", "refract"]
    assert all(step.startswith("CC/facets.t") for step in back.lineage)
    n = _nbk7_index(WAVELENGTH_NM)
    t = 4.0 * n / (n + 1.0) ** 2  # normal incidence on the front face, in and out
    assert back.beam.power_W == pytest.approx(1e-3 * t * t, rel=1e-9)
    assert run.result.power_budget.conserved
    # the run states the faces it traces and how flat they are (corners within the floor of their planes)
    (row,) = [m for m in run.summary()["element_models"] if m["instance_id"] == "CC"]
    assert row["facets"]["traced_faces"] == 4 and row["facets"]["max_face_deviation_mm"] <= row["facets"]["coplanar_tolerance_mm"]
    # the view draws the facets (four triangles), in 3D and in section
    (element,) = [e for e in run.optical_view.elements if e.element_id == "CC"]
    body = element.geometry
    assert body.model_kind == "FACET_SOLID" and len(body.facets.triangles) == 4 and body.surfaces == ()
    drawing = draw_body(body)
    assert sum(len(mesh.indices) // 3 for _material, mesh in drawing.meshes()) == 4
    gltf, _binary = parse_glb(scene_glb_bytes(run.optical_view))
    assert any(node.get("name") == "element:CC" for node in gltf["nodes"])
    section = element_section(body, "XZ")
    assert section["surfaces"] and all(len(piece["points"]) == 2 for s in section["surfaces"] for piece in s["segments"])


def test_a_declared_penta_prism_with_mirrored_faces_deviates_by_90_degrees():
    s, b = 10.0, 4.0
    c, sn = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    a = (s - b / math.sqrt(2.0)) / (c - sn)
    corners = [(0.0, 0.0), (s, 0.0), (s + a * sn, a * c), (s + a * sn - b / math.sqrt(2.0), a * c + b / math.sqrt(2.0)), (0.0, s)]
    vertices, triangles, faces = extruded(corners, 8.0)
    mirrored = {"triangles": faces[3] + faces[5], "traced": "REFLECTED"}
    tilt = math.radians(2.0)
    quaternion = (math.cos(-tilt / 2.0), 0.0, math.sin(-tilt / 2.0), 0.0)  # source +z turned 2 degrees toward -x about y
    document = project([source(position=(3.0 + 40.0 * math.sin(tilt), 0.5, -40.0 * math.cos(tilt)), wavelength_nm=WAVELENGTH_NM,
                               waist_mm=0.2, quaternion=quaternion),
                        declared_component("PP", "prism", "PP", quaternion=IDENTITY)],
                       [plane("out", (-40.0, 0.0, 5.0), radius_mm=25.0)])
    document = with_models(document, [_solid_model("PP", vertices, triangles, facet_groups=[mirrored])], [nbk7()])
    run = run_optical_project(document)
    out = _arrival(run, "out", 0)
    d_in = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
    assert math.degrees(math.acos(float(np.dot(out.beam.direction, d_in)))) == pytest.approx(90.0, abs=1e-9)
    assert [step.split(":")[1] for step in out.lineage] == ["refract", "reflect", "reflect", "refract"]
    (element,) = [e for e in run.optical_view.elements if e.element_id == "PP"]
    assert [g.role for g in element.geometry.facets.groups] == ["REFLECT"]
    materials = {material for material, _mesh in draw_body(element.geometry).meshes()}
    assert materials == {"optic_glass", "mirror_face"}


HIGH_REFLECTOR = "DIELECTRIC_HIGH_REFLECTOR.400-1100R0.99T0.AOI0-45"
ANTI_REFLECTION = "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10"


@pytest.mark.parametrize("coated", ["mirrors", "faces"])
def test_coatings_on_facet_groups_and_on_the_other_facets_set_the_power_through_a_penta_prism(coated):
    """A penta prism of N-BK7 met at normal incidence: with a high reflector (R 0.99) on its two mirrored faces the power out
    is ``T0^2 R^2`` (``T0 = 4 n / (n + 1)^2`` at the bare entry and exit faces); with an anti-reflection coating (T 0.995) as
    the solid's coating and bare mirrors, ``0.995^2``."""
    from local_helpers import catalog_document

    s, b = 10.0, 4.0
    c, sn = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    a = (s - b / math.sqrt(2.0)) / (c - sn)
    corners = [(0.0, 0.0), (s, 0.0), (s + a * sn, a * c), (s + a * sn - b / math.sqrt(2.0), a * c + b / math.sqrt(2.0)), (0.0, s)]
    vertices, triangles, faces = extruded(corners, 8.0)
    mirrors = {"triangles": faces[3] + faces[5], "traced": "REFLECTED"}
    if coated == "mirrors":
        model = _solid_model("PP", vertices, triangles, facet_groups=[{**mirrors, "coating_id": HIGH_REFLECTOR}])
    else:
        model = _solid_model("PP", vertices, triangles, facet_groups=[mirrors], coating_id=ANTI_REFLECTION)
    records = {c["coating_id"]: c for c in catalog_document()["coatings"]}
    document = project([source(position=(3.0, 0.5, -40.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.2, quaternion=IDENTITY),
                        declared_component("PP", "prism", "PP", quaternion=IDENTITY)],
                       [plane("out", (-40.0, 0.0, 5.0), radius_mm=25.0)])
    document = with_models(document, [model], [nbk7()], [records[HIGH_REFLECTOR], records[ANTI_REFLECTION]])
    run = run_optical_project(document)
    out = _arrival(run, "out", 0)
    n = _nbk7_index(WAVELENGTH_NM)
    t0 = 4.0 * n / (n + 1.0) ** 2
    expected = 1e-3 * (t0 * t0 * 0.99 * 0.99 if coated == "mirrors" else 0.995 * 0.995)
    assert out.beam.power_W == pytest.approx(expected, rel=1e-12)
    assert [step.split(":")[1] for step in out.lineage] == ["refract", "reflect", "reflect", "refract"]
    assert run.result.power_budget.conserved


def test_a_declared_tessellated_rod_states_no_gaussian_beam_behind_its_facets():
    sides, radius = 36, 5.0
    ring = [(radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices, triangles, _faces = extruded(ring, 6.0)
    chord = radius * (1.0 - math.cos(math.pi / sides))
    model = _solid_model("ROD", vertices, triangles, approximation="TESSELLATED_FROM_CAD", max_chord_deviation_mm=_decimal(round(chord, 6)),
                         source="a round rod, CAD tessellation")
    document = project([source(position=(1.3, 0.0, -40.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.05, quaternion=IDENTITY),
                        declared_component("ROD", "lens", "ROD", quaternion=IDENTITY)],
                       [plane("out", (0.0, 0.0, 40.0), IDENTITY, radius_mm=25.0)])
    run = run_optical_project(with_models(document, [model], [nbk7()]))
    codes = {i.code.value for i in run.result.issues}
    assert "GAUSSIAN_UNDEFINED_AFTER_FACETS" in codes
    out = _arrival(run, "out", 0)
    assert out.beam.gaussian is not None and not out.beam.gaussian.valid
    assert all(step.startswith("ROD/facets.t") for step in out.lineage)
    assert run.result.power_budget.conserved
    payload = json.loads(json.dumps(model))
    assert payload["model"]["approximation"] == "TESSELLATED_FROM_CAD"


def test_a_facet_solid_member_of_a_compound_unit_is_placed_by_its_pose():
    from declared_helpers import compound_model, cylinder

    base = _corner_cube_design()
    unit = compound_model("UNIT", [("M1", "CC", (0.0, 5.0, 0.0), IDENTITY)], cylinder("80", "-60", "60"))
    laser = base["components"][0]
    x, y, z = laser["nominal_pose"]["position_mm"]
    laser = {**laser, "nominal_pose": {**laser["nominal_pose"], "position_mm": [x, y + 5.0, z]}}  # (the member is 5 mm along y)
    document = {**base, "optical_models": [base["optical_models"][0], unit],
                "components": [laser, declared_component("U", "prism", "UNIT", quaternion=IDENTITY)]}
    run = run_optical_project(document)
    back = _arrival(run, "back", 1)
    assert np.allclose(back.beam.direction, (-1.0, 0.0, 0.0), atol=1e-12, rtol=0)
    assert [step.split(":")[0] for step in back.lineage][0].startswith("U/M1-facets.t")


def _shoelace(polygon) -> float:
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1])))


def _cube_corners(centre, size: float, turn_deg: float = 0.0):
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    out = []
    for x, y, z in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]:
        px, py, pz = size * (x - 0.5), size * (y - 0.5), size * (z - 0.5)
        out.append((centre[0] + c * px + s * pz, centre[1] + py, centre[2] - s * px + c * pz))  # (turned about y)
    return out


CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


@pytest.mark.parametrize(("name", "cubes", "areas"), [
    ("a cube turned 30 deg about y", [((0.0, 0.0, 0.0), 10.0, False)], [100.0]),
    ("two separate cubes", [((-8.0, 0.0, 0.0), 6.0, False), ((8.0, 0.0, 0.0), 6.0, False)], [36.0, 36.0]),
    ("a hollow cube (outlines only)", [((0.0, 0.0, 0.0), 10.0, False), ((0.0, 0.0, 0.0), 4.0, True)], []),
])
def test_the_section_of_a_facet_solid_fills_its_medium_inside_closed_outlines(name, cubes, areas):
    """The section of a facet solid in its XZ plane: the pieces of its facets close into outlines, each filled with the
    solid's medium (area: the cut of the cube, independent of its turn about y); a cavity keeps the outlines only."""
    vertices, triangles = [], []
    for centre, size, inward in cubes:
        v, t, _faces = convex_solid(_cube_corners(centre, size, 30.0 if len(cubes) == 1 else 0.0), CUBE_FACES)
        offset = len(vertices)
        vertices += list(v)
        triangles += [(a + offset, c + offset, b + offset) if inward else (a + offset, b + offset, c + offset) for a, b, c in t]
    document = project([source(position=(-60.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.2),
                        declared_component("S", "prism", "S", quaternion=IDENTITY)], [plane("back", (60.0, 0.0, 0.0), radius_mm=25.0)])
    run = run_optical_project(with_models(document, [_solid_model("S", vertices, triangles)], [nbk7()]))
    (element,) = [e for e in run.optical_view.elements if e.element_id == "S"]
    section = element_section(element.geometry, "XZ")
    assert section["surfaces"], name
    # (the section's points are rounded to 1e-6 mm)
    assert sorted(_shoelace(m["polygon"]) for m in section["media"]) == pytest.approx(sorted(areas), rel=1e-6)
    assert all(m["medium"] == "N-BK7" for m in section["media"])
