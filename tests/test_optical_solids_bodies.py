"""Optical bodies of the solid check against closed-form references (final-review fixes B3, M7, M8, M9, m11, n9).

* B3: a round design body whose -Z face lies below the element origin (``z_min < 0``) keeps its true axial extent: a
  flat mirror's substrate is the 4 mm of glass behind its coated face, a concave mirror's substrate lies behind its
  surface (the air in front of it is air) and an off-axis mirror's blank is solid; census of every catalog reflector;
* M7: a declared single curved mirror without a stated body has its derived body on its non-reflecting (-Z) side in
  the solid check, the 3D mesh and the exact section;
* M8: an aperture stop with a stated plate body is the plate minus its opening, plus its central obstruction;
* M9: a rectangular stop plate's quick-rejection radius reaches its corners (closed-form ``hypot`` of the half widths);
* m11 / n9: the section draws a decentred obstruction's true chord and a rectangular mirror's substrate.
"""

from __future__ import annotations

import math
from decimal import Decimal
from types import SimpleNamespace

import pytest

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import compound_model, cylinder, declared_component, nbk7, sequence, stop_model, surface, with_models
from local_helpers import catalog_document, plane, project, source
from os_optics.local import json_api, load_project, load_public_catalog, lower_project
from os_optics.local.compile import compile_element
from os_optics.optical_solids import PlacedElement, body_of, inspect_optical_solids
from os_optics.optical_solids.bodies import Convex, Plate, Slab, design_solid
from os_optics.optical_view.builder import declared_body
from os_optics.optical_view.optic_mesh import draw_body

I = (1.0, 0.0, 0.0, 0.0)
FLAT = "OS-OPT-MIR-FLAT-D12P7-AG-450-12000-R95-AOI0T10-T4-CA11P43-HK9L"
CONCAVE = "OS-OPT-MIR-CONCAVE-D25P4-F25P4-AG-450-12000-R95-AOI0T10-T4P7-CA22P86-HK9L"
OAP = "OS-OPT-MIR-OAP-D12P7-RFL101P6-A90-AG-450-20000-R96-AOI43P3T46P6-T19P1-CA11P43"
SMALL_LENS = "OS-OPT-LENS-BCX-D6-F12-T2P3-R12-NBK7-AR400-700-W587P6"
RECT_MIRROR = "OS-OPT-MIR-FLAT-R25P4X25P4-AG-450-12000-R95-AOI0T10-T4-CA22P86X22P86-HK9L"


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


def _placed(catalog, iid: str, sku: str, position=(0.0, 0.0, 0.0)) -> PlacedElement:
    product = catalog.get(sku)
    return PlacedElement(compile_element(iid, product, catalog), position, I, design_geometry=product.design_geometry)


def _sag(radius: float, rho: float) -> float:
    return radius - math.copysign(math.sqrt(radius * radius - rho * rho), radius)


def _z_extent(geometry) -> tuple[float, float]:
    if geometry.kind == "CYLINDER":
        return float(Decimal(geometry.z_min_mm)), float(Decimal(geometry.z_max_mm))
    if geometry.kind == "BOX":
        c, h = Decimal(geometry.center_mm[2]), Decimal(geometry.half_extents_mm[2])
        return float(c - h), float(c + h)
    zs = [float(Decimal(v[2])) for v in geometry.vertices_mm]
    return min(zs), max(zs)


def _ct(catalog, sku: str) -> float:
    return float(Decimal(catalog.get(sku).simulation_models[0].surfaces[-1].vertex_z_mm))


# ------------------------------------------------------------------------------------------------ B3


def test_a_round_design_body_below_the_origin_keeps_its_axial_extent(catalog):
    for sku in (FLAT, CONCAVE, OAP):
        geometry = catalog.get(sku).design_geometry
        expected = (float(Decimal(geometry.z_min_mm)), float(Decimal(geometry.z_max_mm)))
        assert expected[0] < 0.0
        assert Convex(design_solid(geometry), None, "DESIGN_GEOMETRY").z_range() == pytest.approx(expected, abs=1e-12)


def test_a_lens_inside_a_flat_mirrors_substrate_overlaps_by_its_thickness(catalog):
    ct = _ct(catalog, SMALL_LENS)  # its glass spans [-3, -3 + ct] on the axis, inside the mirror's glass [-4, 0]
    report = inspect_optical_solids([_placed(catalog, "M", FLAT), _placed(catalog, "L", SMALL_LENS, (0.0, 0.0, -3.0))])
    (issue,) = report.issues
    assert issue.code == "OPTICAL_BODIES_OVERLAP" and issue.value_mm == pytest.approx(ct, abs=1e-6)
    off_axis = inspect_optical_solids([_placed(catalog, "M", FLAT), _placed(catalog, "L", SMALL_LENS, (1.0, 0.0, -3.0))])
    assert [i.code for i in off_axis.issues] == ["OPTICAL_BODIES_OVERLAP"] and off_axis.issues[0].value_mm > 1.0


def test_a_lens_in_the_air_in_front_of_a_concave_mirror_is_valid(catalog):
    # the lens's front vertex 0.3 mm in front of the mirror's vertex; at its edge (r = 3) its surface is at
    # 0.3 + sag_12(3) = 0.681 mm, the mirror's at sag_50.8(3) = 0.089 mm: air between them everywhere
    assert 0.3 + _sag(12.0, 3.0) > _sag(50.8, 3.0)
    for position in ((0.0, 0.0, 0.3), (5.0, 0.0, 0.9)):
        report = inspect_optical_solids([_placed(catalog, "M", CONCAVE), _placed(catalog, "L", SMALL_LENS, position)])
        assert report.valid and report.issues == (), position


def test_a_lens_inside_an_off_axis_mirror_blank_overlaps(catalog):
    ct = _ct(catalog, SMALL_LENS)
    report = inspect_optical_solids([_placed(catalog, "O", OAP), _placed(catalog, "L", SMALL_LENS, (0.0, 0.0, -5.0))])
    (issue,) = report.issues
    assert issue.code == "OPTICAL_BODIES_OVERLAP" and issue.value_mm == pytest.approx(ct, abs=1e-6)


def test_census_every_catalog_reflector_has_its_substrate_behind_its_coated_face(catalog):
    """Every single coated surface's substrate fills the design body from the surface to the body's far face: behind a
    mirror its -Z face (the face-normal frame's back), behind a transmitting coated plane the face farther from it."""

    counted = 0
    for product in catalog.products:
        for model in product.simulation_models:
            if model.model_kind != "SURFACE_SEQUENCE" or len(model.surfaces) != 1 or product.design_geometry is None:
                continue
            geometry = product.design_geometry
            z_min, z_max = _z_extent(geometry)
            vertex = float(Decimal(model.surfaces[0].vertex_z_mm))
            body = body_of(compile_element("E", product, catalog, model_kind=model.model_kind), design_geometry=geometry)
            (substrate,) = [p for p in body.pieces if isinstance(p, Slab)]
            flat = substrate.front if substrate.front.surface_id == "substrate" else substrate.back
            if model.surfaces[0].traced.value == "REFLECTED":
                assert flat.vertex_z == z_min and substrate.front is flat, product.optical_sku
                counted += 1
            else:
                assert flat.vertex_z == (z_min if vertex - z_min >= z_max - vertex else z_max), product.optical_sku
            # a point in the glass just behind the coated face's vertex is inside, one in front of it is not
            inward = -1.0 if flat.vertex_z < vertex else 1.0
            assert body.depth((0.0, 0.0, vertex + inward * 1e-3)) > 0.0 > body.depth((0.0, 0.0, vertex - inward * 1e-3)), product.optical_sku
    assert counted > 200  # flat, concave and dichroic mirrors of the seed release


# ------------------------------------------------------------------------------------------------ M7


COATING = next(c for c in catalog_document()["coatings"] if c["coating_id"].startswith("DIELECTRIC_HIGH_REFLECTOR"))


def _mirror_model(radius: str) -> dict:
    return {"model_id": "cm", "model": sequence([surface(1, "0", radius=radius, clear="12", traced="REFLECTED", coating_id=COATING["coating_id"])],
                                                [], edge="12.7")}


def _design(components: list[dict], models: list[dict]) -> dict:
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), *components], [plane("P", (-100.0, 0.0, 0.0), radius_mm=20.0)])
    return with_models(document, models, [nbk7()], coatings=[COATING])


def _unit_with_mirror(radius: str, member: tuple[str, dict, tuple]) -> dict:
    name, model, position = member
    unit = compound_model("u", [("M", "cm", (0, 0, 0), I), (name, model["model_id"], position, I)], cylinder("30", "-5", "5"))
    return _design([declared_component("U", "lens", "u")], [_mirror_model(radius), model, unit])


def test_a_declared_concave_mirror_has_glass_only_behind_its_surface():
    edge_sag = _sag(50.0, 12.7)
    document = _design([declared_component("M", "mirror", "cm")], [_mirror_model("50")])
    (element,) = lower_project(load_project(document), check_solids=False).elements
    geometry = element.identity.design_geometry
    assert (float(geometry.z_min_mm), float(geometry.z_max_mm)) == (0.0, pytest.approx(edge_sag, abs=1e-9))  # the bounding body
    body = body_of(element, design_geometry=geometry)
    (substrate,) = body.pieces
    assert substrate.front.surface_id == "substrate" and substrate.front.vertex_z == 0.0 and substrate.back.surface_id == "S1"
    assert body.depth((0.0, 0.0, 1.0)) < 0.0  # on the axis in front of the vertex: the air in the mirror's cup
    assert body.depth((12.0, 0.0, 0.5)) > 0.0  # behind the surface (sag(12) = 1.463 mm): the substrate
    # the 3D view draws the substrate's flat face at z = 0, facing -Z
    drawing = draw_body(declared_body(element, load_project(document)))
    (face,) = [mesh for material, mesh in drawing.body if material == "mirror_substrate"]
    assert min(p[2] for p in face.positions) == 0.0 and max(p[2] for p in face.positions) == pytest.approx(edge_sag, abs=1e-9)
    flat = [tuple(n) for n in face.normals if n[0] == 0.0 and n[1] == 0.0]
    assert flat and set(flat) == {(0.0, 0.0, -1.0)}  # the flat face looks away from the reflecting side
    # the section's substrate polygon closes along z = 0 (behind the surface), not along the bounding body's far face
    polygon = json_api.optical_section(document, "M", "XZ")["section"]["elements"][0]["substrate"]["polygon"]
    assert [z for _, z in polygon[-2:]] == [0.0, 0.0] and max(z for _, z in polygon) == pytest.approx(edge_sag, abs=1e-6)


def test_bodies_in_front_of_a_declared_concave_mirror_are_valid_and_crossing_its_rim_is_not():
    small = {"model_id": "sl", "model": sequence([surface(1, "0", radius="20", clear="2.5"), surface(2, "0.6", radius="-20", clear="2.5")],
                                                 ["N-BK7"], edge="3")}
    # a small lens 0.5 mm in front of the vertex: its back surface is at 0.5 + 0 .. its rim, the mirror at sag(3) = 0.09 mm
    assert json_api.inspect_optical_solids(_unit_with_mirror("50", ("L", small, (0, 0, 0.5))))["status"] == "VALID"
    # a stop plate (r 12.7) 3 mm in front of the vertex clears the rim (sag(12.7) = 1.64 mm) ...
    for z, status in (("3", "VALID"), ("1", "INVALID")):
        stop = stop_model("st", opening_radius_mm="4", edge_radius_mm="12.7")  # its plane at the member's position
        report = json_api.inspect_optical_solids(_unit_with_mirror("50", ("S", stop, (0, 0, float(z)))))
        assert report["status"] == status, z
    # ... one 1 mm in front of it crosses the mirror's surface beyond rho = sqrt(2 R z - z^2) = 9.95 mm, where the glass
    # behind the surface rises above the plate
    (issue,) = report["issues"]
    x, y, _ = issue["position_mm"]
    assert issue["code"] == "OPTICAL_BODIES_OVERLAP" and math.hypot(y, _) > math.sqrt(2 * 50 * 1 - 1)
    convex = json_api.inspect_optical_solids(_unit_with_mirror("-50", ("L", small, (0, 0, 0.5))))  # control: glass behind a convex mirror
    assert convex["status"] == "VALID"


# ------------------------------------------------------------------------------------------------ M8


def _window(model_id: str, radius: str, thickness: str = "2") -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", clear=str(float(radius) * 0.9)), surface(2, thickness, clear=str(float(radius) * 0.9))],
                                                    ["N-BK7"], edge=radius)}


def _stop_unit(stop: dict, window: dict, position) -> dict:
    unit = compound_model("u", [("S", stop["model_id"], (0, 0, 0), I), ("W", window["model_id"], position, I)], cylinder("30", "-5", "10"))
    return _design([declared_component("U", "lens", "u")], [stop, window, unit])


def test_a_native_iris_is_open_where_light_passes():
    built = json_api.build_native_model("aperture.circular", {"opening_diameter_mm": "12", "plate_diameter_mm": "25.4", "thickness_mm": "1",
                                                              "model_id": "iris", "instance_id": "S"})
    assert built["ok"] and built["optical_model"]["design_geometry"]["z_max_mm"] == "1"
    stop = built["optical_model"]
    # a window of radius 5 inside the opening (r 6), sharing the plate's z range: nothing of the stop is there
    assert json_api.inspect_optical_solids(_stop_unit(stop, _window("w", "5"), (0, 0, 0.4)))["status"] == "VALID"
    # a window of radius 8 at z = 0.4: it shares the plate's glass between r = 6 and r = 8 over z in [0.4, 1]
    report = json_api.inspect_optical_solids(_stop_unit(stop, _window("w", "8"), (0, 0, 0.4)))
    (issue,) = report["issues"]
    assert issue["code"] == "OPTICAL_BODIES_OVERLAP" and issue["value_mm"] == pytest.approx(0.6, abs=1e-6)


def test_a_stated_stop_plate_keeps_its_obstruction_and_a_decentred_opening():
    stated = cylinder("25.4", "0", "1")
    annular = {**stop_model("st", opening_radius_mm="6", obstruction_radius_mm="2", edge_radius_mm="12.7"), "design_geometry": stated}
    report = json_api.inspect_optical_solids(_stop_unit(annular, _window("w", "1.5"), (0, 0, 0.4)))  # inside the obstruction disc
    # the small window is 2 mm thick on a 2.7 mm clear diameter (74 %): a declared model between the review ratio 35 % and
    # 1.5 clear diameters is an INFO finding with its ratio, besides the overlap
    (issue,) = [i for i in report["issues"] if i["code"] != "EDGE_THICKNESS_IMPLAUSIBLE"]
    (edge,) = [i for i in report["issues"] if i["code"] == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert edge["severity"] == "INFO" and edge["value_mm"] == pytest.approx(2.0) and edge["limit_mm"] == pytest.approx(0.35 * 2.7)
    assert issue["code"] == "OPTICAL_BODIES_OVERLAP" and issue["value_mm"] == pytest.approx(0.6, abs=1e-6)
    decentred = {**stop_model("st", opening_radius_mm="3", opening_center_mm=["2", "0"], edge_radius_mm="12.7"), "design_geometry": stated}
    assert json_api.inspect_optical_solids(_stop_unit(decentred, _window("w", "1"), (2, 0, 0.4)))["status"] == "VALID"
    crossing = json_api.inspect_optical_solids(_stop_unit(decentred, _window("w", "1"), (-1.5, 0, 0.4)))  # x in [-2.5, -0.5]: over the plate
    assert crossing["status"] == "INVALID" and [i["code"] for i in crossing["issues"] if i["severity"] == "ERROR"] == ["OPTICAL_BODIES_OVERLAP"]
    declared = load_project(_stop_unit(decentred, _window("w", "1"), (2, 0, 0.4))).optical_model("st")
    body = body_of(SimpleNamespace(instance_id="S", model=declared.model, surfaces=(), solid=None), design_geometry=declared.design_geometry)
    (plate,) = body.pieces
    assert isinstance(plate, Plate)
    assert plate.depth((2.0, 0.0, 0.5)) < 0.0 < plate.depth((-2.0, 0.0, 0.5))  # the opening's centre is open, the plate beside it is not


def test_the_section_of_a_stop_plate_is_open_at_its_opening():
    stop = {**stop_model("st", opening_radius_mm="6", obstruction_radius_mm="2", edge_radius_mm="12.7"), "design_geometry": cylinder("25.4", "0", "1")}
    document = _design([declared_component("S", "aperture", "st")], [stop])
    entry = json_api.optical_section(document, "S", "XZ")["section"]["elements"][0]
    spans = sorted((min(t for t, _ in p), max(t for t, _ in p)) for p in entry["plate"]["polygons"])
    assert spans == [(-12.7, -6.0), (-2.0, 2.0), (6.0, 12.7)]
    assert all(sorted({z for _, z in p}) == [0.0, 1.0] for p in entry["plate"]["polygons"])


# ------------------------------------------------------------------------------------------------ M9


def _rect_stop(hx: float, hy: float) -> SimpleNamespace:
    model = SimpleNamespace(model_kind="APERTURE_STOP", opening_radius_mm="2", opening_half_widths_mm=None, opening_center_mm=None,
                            obstruction_radius_mm=None, edge_radius_mm=None, edge_half_widths_mm=(str(hx), str(hy)), plane_z_mm="0")
    return SimpleNamespace(instance_id="S", model=model, surfaces=(), solid=None)


def _flat_window(radius: float, thickness: float) -> SimpleNamespace:
    s1 = SimpleNamespace(surface_id="S1", vertex_z_mm=str(-thickness / 2), radius_mm=None, conic=None, aspheric_coefficients=None,
                         cylinder_axis=None, clear_radius_mm=str(radius * 0.9), clear_half_widths_mm=None, aperture_center_mm=None)
    s2 = SimpleNamespace(**{**s1.__dict__, "surface_id": "S2", "vertex_z_mm": str(thickness / 2)})
    model = SimpleNamespace(model_kind="SURFACE_SEQUENCE", edge_radius_mm=str(radius), edge_half_widths_mm=None, surfaces=(s1, s2), media=("N-BK7",))
    return SimpleNamespace(instance_id="W", model=model, surfaces=(), solid=None)


@pytest.mark.parametrize(("hx", "hy"), [(20.0, 2.0), (20.0, 10.0), (30.0, 30.0)])
def test_a_rectangular_plate_reaches_its_corners(hx, hy):
    assert body_of(_rect_stop(hx, hy)).radius >= math.hypot(hx, hy)


@pytest.mark.parametrize("beyond", [2.5, 3.0, 3.4])
def test_a_rectangular_plate_corner_inside_a_window_is_an_overlap(beyond):
    # the window (r 5, 2 mm thick, centred on the plate's plane) holds the plate's corner (20, 10): depth 5 - beyond * sqrt(2)
    report = inspect_optical_solids([PlacedElement(_rect_stop(20.0, 10.0), (0.0, 0.0, 0.0), I),
                                     PlacedElement(_flat_window(5.0, 2.0), (20.0 + beyond, 10.0 + beyond, 0.0), I)])
    (issue,) = report.issues
    assert issue.code == "OPTICAL_BODIES_OVERLAP" and issue.value_mm == pytest.approx(min(5.0 - beyond * math.sqrt(2.0), 1.0), abs=1e-5)


# ------------------------------------------------------------------------------------------------ m11 / n9


@pytest.mark.parametrize(("centre_y", "chord"), [("1", math.sqrt(1.5 ** 2 - 1.0)), ("0", 1.5), ("2", None)])
def test_the_section_draws_a_decentred_obstructions_chord(centre_y, chord):
    stop = stop_model("st", opening_radius_mm="3", opening_center_mm=["0", centre_y], obstruction_radius_mm="1.5", edge_radius_mm="8")
    document = _design([declared_component("S", "aperture", "st")], [stop])
    (entry,) = json_api.optical_section(document, "S", "XZ")["section"]["elements"][0]["surfaces"]
    if chord is None:
        assert "obstruction_mm" not in entry and all(s["zone"] != "OBSTRUCTION" for s in entry["segments"])
    else:
        assert entry["obstruction_mm"] == [pytest.approx(-chord, abs=1e-6), pytest.approx(chord, abs=1e-6)]


def test_the_section_draws_a_rectangular_mirrors_substrate(catalog, real_catalog_store):
    from compound_objective import ALONG_X, project_of
    from compound_objective import source as objective_source
    from os_optics._vendor.shared_contracts.common.geometry import PoseV1
    from os_optics._vendor.shared_contracts.optical_design import OpticalComponentInstanceV1, OpticalProductRefV1

    geometry = catalog.get(RECT_MIRROR).design_geometry
    assert geometry.kind == "BOX"
    back = float(Decimal(geometry.center_mm[2]) - Decimal(geometry.half_extents_mm[2]))
    component = OpticalComponentInstanceV1(instance_id="M", optical_role="mirror", product=OpticalProductRefV1(optical_sku=RECT_MIRROR),
                                           nominal_pose=PoseV1(position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X))
    document = project_of([objective_source(-60.0, 1.0), component], 150.0).payload()
    entry = json_api.optical_section(document, "M", "XZ")["section"]["elements"][0]
    polygon = entry["substrate"]["polygon"]
    assert back == -4.0 and [z for _, z in polygon[-2:]] == [back, back]
    assert sorted({t for t, _ in polygon[-2:]}) == [-12.7, 12.7]


# ------------------------------------------------------------------------------------------------ M1 / m2


def _meniscus(stated: dict | None = None) -> dict:
    model = {"model_id": "men", "model": sequence([surface(1, "0", radius="20", clear="9"), surface(2, "3", radius="25", clear="9")], ["N-BK7"], edge="10")}
    if stated is not None:
        model["design_geometry"] = stated
    return model


def test_a_derived_body_spans_the_surfaces_over_the_whole_outline():
    """S2 (R 25, vertex 3) rises to 3 + sag_25(10) = 5.0871 mm at the 10 mm edge, beyond its 9 mm clear aperture: the
    derived body is the contract's one derivation, rounded outward to 1e-9 mm."""

    from os_optics._vendor.shared_contracts.optical_design import ProjectOpticalModelV1
    from os_optics.local.compile import declared_design_geometry

    declared = ProjectOpticalModelV1.model_validate(_meniscus())
    geometry = declared_design_geometry(declared)
    top = 3.0 + _sag(25.0, 10.0)
    assert float(geometry.z_min_mm) == 0.0 and top <= float(geometry.z_max_mm) < top + 1e-9
    assert (Decimal(geometry.z_min_mm), Decimal(geometry.z_max_mm)) == declared.model.outline_z_range_mm()


def test_a_stated_body_that_misses_the_surfaces_is_extended_to_hold_them():
    """A stated cylinder that holds the vertices, not the rim (S2 rises to 5.087122 mm at the 10 mm edge): the body used is
    the smallest cylinder that holds the stated one and the surfaces over the whole outline -- here the derived body --
    and the check states both."""

    from os_optics._vendor.shared_contracts.optical_design import ProjectOpticalModelV1
    from os_optics.local.compile import declared_design_geometry

    lens = declared_component("M", "lens", "men")
    design = _design([lens], [_meniscus(cylinder("20", "0", "3"))])
    checked = json_api.check_project(design)
    assert checked["ok"] is True
    (notice,) = [w for w in checked["warnings"] if w["code"] == "DESIGN_GEOMETRY_EXTENDED"]
    assert notice["location"] == "optical model men" and notice["details"]["stated"]["z_max_mm"] == "3"
    used = notice["details"]["used"]
    assert (used["kind"], used["diameter_mm"], used["z_min_mm"]) == ("CYLINDER", "20", "0")
    assert 5.0871215 <= float(used["z_max_mm"]) <= 5.0871216  # (3 + 25 - sqrt(25^2 - 10^2), rounded up to 1e-9 mm)
    derived = declared_design_geometry(ProjectOpticalModelV1.model_validate(_meniscus()))
    assert used == derived.model_dump(mode="json")  # (the stated body lies inside the glass's own outline body)
    (element_,) = lower_project(design).elements
    assert element_.identity.design_geometry.model_dump(mode="json") == used
    # a stated body that already holds the surfaces is used as stated, without a notice
    held = json_api.check_project(_design([lens], [_meniscus(cylinder("20", "0", "5.0872"))]))
    assert held["ok"] and not [w for w in held["warnings"] if w["code"] == "DESIGN_GEOMETRY_EXTENDED"]
    # a wider stated body keeps its width; only its length grows
    wide = json_api.check_project(_design([lens], [_meniscus(cylinder("30", "-1", "3"))]))
    (notice,) = [w for w in wide["warnings"] if w["code"] == "DESIGN_GEOMETRY_EXTENDED"]
    assert (notice["details"]["used"]["diameter_mm"], notice["details"]["used"]["z_min_mm"]) == ("30", "-1")


def test_a_neighbour_inside_the_extension_of_a_stated_body_overlaps_it():
    """The extension is glass: a window placed where the stated body ended but the meniscus's rim still is overlaps it."""

    lens = declared_component("M", "lens", "men")
    window = declared_component("W", "window", "win", (0.0, 0.0, 4.0))
    refused = json_api.check_project(_design([lens, window], [_meniscus(cylinder("20", "0", "3")), WINDOW]))
    assert refused["ok"] is False
    assert "OPTICAL_BODIES_OVERLAP" in {p["code"] for p in refused["error"]["problems"]}


# ------------------------------------------------------------------------------------------------ m1 / m12

ALONG_X = (0.5, 0.5, 0.5, 0.5)
PCX = {"model_id": "pcx", "model": sequence([surface(1, "0", radius="52"), surface(2, "4.5")], ["N-BK7"])}  # clear 9, edge 10
WINDOW = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"])}


def _contact_unit(gap: str) -> dict:
    unit = compound_model("u", [("L1", "pcx", (0, 0, 0), I), ("W", "win", (0, 0, 4.5 + float(gap)), I)], cylinder("22", "-1", "10"))
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=0.5), declared_component("U", "lens", "u", (0, 0, 0), ALONG_X)],
                       [plane("scr", (100.0, 0.0, 0.0), radius_mm=100.0)])
    return with_models(document, [PCX, WINDOW, unit], [nbk7()])


def test_bodies_touching_where_light_crosses_are_refused_with_structured_details():
    """A plano-convex lens's flat back against a window: the tracer cannot pass the window's entrance surface at zero
    distance from the lens's exit surface, so the design is refused (merge into one cemented sequence, or separate)."""

    refused = json_api.check_project(_contact_unit("0"))
    (problem,) = refused["error"]["problems"]
    assert (problem["code"], problem["location"]) == ("OPTICAL_BODIES_TOUCH", "component U member L1 surface S2")
    details = problem["details"]
    assert (details["other_instance_id"], details["other_member_id"], details["other_surface_id"]) == ("U", "W", "S1")
    assert details["position_mm"] == pytest.approx([4.5, 0.0, 0.0], abs=1e-9) and details["value_mm"] == 0.0
    assert "cemented" in problem["message"] and "separate them by more than 1e-06 mm" in problem["message"]
    assert refused["optical_solids"]["status"] == "INVALID"  # the whole report travels with the refusal
    assert json_api.check_project(_contact_unit("0.001"))["ok"]


def test_a_stop_plate_against_a_lens_rim_is_a_contact_inside_its_clear_aperture_a_refusal():
    for opening, severity in (("9.5", "INFO"), ("8", "ERROR")):  # the lens's clear radius is 9, its edge 10
        stop = stop_model("st", opening_radius_mm=opening, edge_radius_mm="12.7")
        unit = compound_model("u", [("L1", "pcx", (0, 0, 0), I), ("S", "st", (0, 0, 4.5), I)], cylinder("30", "-1", "10"))
        document = with_models(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), declared_component("U", "lens", "u")],
                                       [plane("scr", (100.0, 0.0, 0.0), radius_mm=100.0)]), [PCX, stop, unit], [nbk7()])
        report = json_api.inspect_optical_solids(document)
        assert [(i["code"], i["severity"]) for i in report["issues"]] == [("OPTICAL_BODIES_TOUCH", severity)], opening
        assert report["status"] == ("VALID" if severity == "INFO" else "INVALID")


def test_an_overlap_refusal_carries_the_other_body_position_and_depth():
    lens = declared_component("A", "lens", "pcx", (0, 0, 0), ALONG_X)
    window = declared_component("B", "window", "win", (4.0, 0, 0), ALONG_X)  # 0.5 mm into the lens's back
    document = with_models(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), lens, window],
                                   [plane("scr", (100.0, 0.0, 0.0), radius_mm=100.0)]), [PCX, WINDOW], [nbk7()])
    refused = json_api.check_project(document)
    (problem,) = refused["error"]["problems"]
    assert problem["code"] == "OPTICAL_BODIES_OVERLAP" and problem["location"] == "component A"
    assert problem["details"]["other_instance_id"] == "B" and problem["details"]["value_mm"] == pytest.approx(0.5, abs=1e-6)
    assert len(problem["details"]["position_mm"]) == 3
    assert refused["optical_solids"]["issues"][0]["code"] == "OPTICAL_BODIES_OVERLAP"
    run = json_api.run_project(document)
    assert not run["ok"] and run["error"]["problems"][0]["details"] == problem["details"]


# ------------------------------------------------------------------------------------------------ n8


def test_a_simple_declared_lens_is_drawn_within_the_chord_tolerance_under_the_vertex_cap():
    """A sphere R 9 over a 12 x 10 mm rectangular outline (round 4.5 mm aperture): doubling the mesh again would exceed
    the per-surface vertex budget, so the rest of the budget goes to directions. Independent check: the exact sphere at
    every drawn triangle's centroid and edge midpoints, along the normal."""

    import numpy as np

    from os_optics.optical_view.optic_mesh import MAX_SURFACE_VERTICES

    lens = {"model_id": "m", "model": {**sequence([surface(1, "0", radius="9", clear="4.5"), surface(2, "5", clear="4.5")], ["N-BK7"]),
                                       "edge_half_widths_mm": ["6", "5"]}}
    lens["model"].pop("edge_radius_mm")
    document = _design([declared_component("L", "lens", "m")], [lens])
    (element,) = lower_project(load_project(document), check_solids=False).elements
    drawing = draw_body(declared_body(element, load_project(document)))
    s1 = drawing.surfaces[0]
    assert "chord_tolerance_met" not in s1.facts and s1.facts["max_deviation_mm"] <= 0.002
    worst, count = 0.0, 0
    for _, mesh in s1.meshes:
        p = np.asarray(mesh.positions, float)
        tri = p[np.asarray(mesh.indices, int).reshape(-1, 3)]
        count += len(p)
        probes = np.concatenate([tri.mean(axis=1), 0.5 * (tri[:, 0] + tri[:, 1]), 0.5 * (tri[:, 1] + tri[:, 2]), 0.5 * (tri[:, 0] + tri[:, 2])])
        rho2 = probes[:, 0] ** 2 + probes[:, 1] ** 2
        exact = 9.0 - np.sqrt(81.0 - rho2)
        worst = max(worst, float(np.max(np.abs(probes[:, 2] - exact) * np.sqrt(81.0 - rho2) / 9.0)))
    assert worst <= 0.002 + 1e-9 and count <= MAX_SURFACE_VERTICES
