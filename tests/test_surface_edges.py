"""Per-surface mechanical edges of a ``SURFACE_SEQUENCE``, end to end.

A surface may state its own edge (``edge_radius_mm`` on the surface); the element outline is the largest edge. Beyond its
edge a surface is held flat at its rim height out to the outline of the glass it bounds. Checked here:

* prescription import: each row's ``mech_semi_diameter`` becomes that surface's edge (no collapse to the largest one),
  so a lens whose steep surface is edged small imports with its prescribed thickness;
* compile: each surface blocks light out to the outline of its own glass, so two air-spaced lenses in one sequence trace
  exactly like the same lenses as two elements (a ray outside the small lens passes it and meets the large one);
* optical solids: the glass is held at each surface's edge (thickness, profile domain), with structured plausibility
  warnings for an outline that is not the element's edge;
* view: the mesh of an edged lens is closed, its rim ends at the edge and its wall spans the real edge thickness; the
  section draws the flange.

Fixture data: every physical value below is made up for the test.
"""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

import pytest

from os_optics._vendor.shared_contracts.design_catalog import SurfaceSequenceModelV1
from os_optics._vendor.shared_contracts.optical_view import OpticBodyGeometryV1
from os_optics.local import load_public_catalog, lower_project, run_optical_project
from os_optics.optical_solids import checks
from os_optics.optical_solids.bodies import body_of
from os_optics.optical_view.optic_mesh import draw_body
from os_optics.optical_view.section import _sequence_section
from os_optics.prescription import PrescriptionErrorCode as E, PrescriptionImportError, import_prescription
from os_optics.project_document import parse_project_document

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import arrivals, plane, project, source

SEED = load_public_catalog(REAL_CATALOG.read_bytes())

# A biconvex N-BK7 lens, R = +-20 mm, 6 mm thick, clear semi-diameter 8 mm. Edged at 11 mm on both sides the glass would
# cross itself (each sag at 11 mm is 3.30 mm: 6 - 2 x 3.30 < 0); with the back surface edged at 9 mm (sag 2.14 mm) and
# held flat beyond it, the edge is 6 - 3.30 - 2.14 = 0.56 mm thick.
EDGED = """surface,radius,thickness,material,semi_diameter,mech_semi_diameter
1,20,6,N-BK7,8,11
2,-20,10,,8,9
"""


def _sag(r: float, rho: float) -> float:
    return r - math.copysign(math.sqrt(r * r - rho * rho), r)


# ------------------------------------------------------------------------------------------------- prescription


def test_lowering_keeps_each_surfaces_mechanical_edge():
    result = import_prescription(EDGED, catalog=SEED)
    lens = next(m for m in result.optical_models if m.model_id == "prescription-L1").model
    assert isinstance(lens, SurfaceSequenceModelV1)
    assert lens.edge_radius_mm == "11"
    assert lens.surfaces[0].edge_radius_mm is None  # edged at the outline: no own edge
    assert lens.surfaces[1].edge_radius_mm == "9"
    member = result.report["members"][0]
    assert member["edge_radius_mm"] == "11" and member["surface_edge_radii_mm"] == ["11", "9"]
    # the solid is held at the back edge: positive edge thickness, as computed by hand
    issues, gaps = checks.element_issues(body_of(SimpleNamespace(instance_id="L1", model=lens)))
    assert not [i for i in issues if i.severity == "ERROR"]
    assert gaps[0]["edge_thickness_mm"] == pytest.approx(6.0 - _sag(20.0, 11.0) - _sag(20.0, 9.0), abs=1e-6)


def test_without_the_smaller_edge_the_same_lens_crosses_itself():
    with pytest.raises(PrescriptionImportError) as refused:
        import_prescription(EDGED.replace("8,9\n", "8,11\n"), catalog=SEED)
    assert refused.value.problems[0].code is E.EDGE_INVALID


def test_a_surface_whose_profile_ends_before_its_own_edge_is_refused():
    with pytest.raises(PrescriptionImportError) as refused:
        import_prescription(EDGED.replace("1,20,6,", "1,20,10,").replace("2,-20,10,,8,9", "2,-8.8,10,,8,9"), catalog=SEED)
    problem = refused.value.problems[0]
    assert problem.code is E.EDGE_INVALID and "its mechanical semi-diameter 9 mm" in problem.message


# ------------------------------------------------------------------------------------------------------ compile


def _element(model: SurfaceSequenceModelV1):
    document = with_models(project([source(), declared_component("E", "lens", "m")], [plane("P", (100.0, 0.0, 0.0))]),
                           [{"model_id": "m", "model": model.model_dump(mode="json", exclude_none=True)}], [nbk7()])
    return lower_project(document).elements[0]


def _group(edges: tuple[str | None, ...], outline: str) -> dict:
    """Two air-spaced N-BK7 lenses in one sequence: the meniscus L1 = S1 (R 30) - S2 (R 20), clear radius 4 mm, and 1 mm
    behind it the plano-convex L2 = S3 (plane) - S4 (R -40), clear radius 6 mm. L1's back surface bends toward L2: at
    5 mm from the axis it is 0.36 mm clear of L2's face, at 8 mm it would cut 0.67 mm into it."""

    rows = [("0", "30", "4"), ("3", "20", "4"), ("4", None, "6"), ("7", "-40", "6")]
    surfaces = []
    for i, ((z, radius, clear), edge) in enumerate(zip(rows, edges), start=1):
        extra = {"edge_radius_mm": edge} if edge is not None else {}
        surfaces.append(surface(i, z, radius=radius, clear=clear, **extra))
    return sequence(surfaces, ["N-BK7", "AIR", "N-BK7"], edge=outline)


def _solid(model: SurfaceSequenceModelV1):
    """The optical body of a model (no Engine surfaces: the medium chain is not checked)."""

    return body_of(SimpleNamespace(instance_id="E", model=model))


def _support(engine_surface) -> float:
    radius = engine_surface.aperture_radius_mm if engine_surface.geometry == "sphere" else engine_surface.aperture.radius_mm
    return radius + engine_surface.mount_margin_mm


def test_each_surface_blocks_light_out_to_its_own_glass():
    model = SurfaceSequenceModelV1.model_validate(_group(("5", "5", None, None), "8"))
    element = _element(model)
    assert [round(_support(s), 12) for s in element.surfaces] == [5.0, 5.0, 8.0, 8.0]


def _run(models: list[dict], components: list[dict], beam_y: float):
    document = with_models(project([source(position=(-50.0, beam_y, 0.0), waist_mm=0.2, wavelength_nm=632.8), *components],
                                   [plane("P", (60.0, 0.0, 0.0), radius_mm=30.0)]), models, [nbk7()])
    return run_optical_project(document)


@pytest.mark.parametrize("beam_y", [0.0, 3.0, 6.5])
def test_a_group_in_one_sequence_traces_like_its_elements(beam_y):
    """A beam 6.5 mm off axis passes outside L1 (edge 5 mm) and meets L2's rim (clear 6 mm, edge 8 mm): blocked, as it
    is when L1 and L2 are two elements; on axis and at 3 mm both builds deliver the same light to the same place."""

    group = {"model_id": "group", "model": _group(("5", "5", None, None), "8")}
    one = _run([group], [declared_component("G", "lens", "group", quaternion=(0.5, 0.5, 0.5, 0.5))], beam_y)
    l1 = {"model_id": "l1", "model": sequence([surface(1, "0", radius="30", clear="4"), surface(2, "3", radius="20", clear="4")], ["N-BK7"],
                                              edge="5")}
    l2 = {"model_id": "l2", "model": sequence([surface(1, "0", clear="6"), surface(2, "3", radius="-40", clear="6")], ["N-BK7"], edge="8")}
    two = _run([l1, l2], [declared_component("L1", "lens", "l1", quaternion=(0.5, 0.5, 0.5, 0.5)),
                          declared_component("L2", "lens", "l2", position=(4.0, 0.0, 0.0), quaternion=(0.5, 0.5, 0.5, 0.5))], beam_y)
    got_one, got_two = arrivals(one.result, "P"), arrivals(two.result, "P")
    assert len(got_one) == len(got_two)
    if beam_y == 6.5:
        assert got_one == []
    for a, b in zip(got_one, got_two):
        assert a.beam.power_W == pytest.approx(b.beam.power_W, rel=1e-12)
        assert a.uv_mm == pytest.approx(b.uv_mm, abs=1e-9)


def test_the_group_in_one_sequence_needs_no_air_gap_inflation():
    """With one edge for the whole sequence (8 mm), L1's back surface would cut into L2 (only a wider air gap would
    pass); with L1 edged at its own 5 mm the same spacing is valid."""

    issues, _ = checks.element_issues(_solid(SurfaceSequenceModelV1.model_validate(_group((None, None, None, None), "8"))))
    assert "SURFACES_INTERSECT" in [i.code for i in issues]
    issues, _ = checks.element_issues(_solid(SurfaceSequenceModelV1.model_validate(_group(("5", "5", None, None), "8"))))
    assert not [i for i in issues if i.severity == "ERROR"], [i.message for i in issues]


# ----------------------------------------------------------------------------------------------- plausibility


def _lens(front_edge: str | None, clear: str, outline: str, thickness: str) -> SurfaceSequenceModelV1:
    extra = {"edge_radius_mm": front_edge} if front_edge is not None else {}
    return SurfaceSequenceModelV1.model_validate(
        sequence([surface(1, "0", radius="200", clear=clear, **extra), surface(2, thickness, clear=clear)], ["N-BK7"], edge=outline))


def test_an_implausible_edge_thickness_is_a_structured_warning():
    """12.2 mm of glass at the edge over a 5.9 mm clear diameter (206 %): a warning that names the gap and the ratio,
    beyond the catalog-calibrated 1.5 (``checks.EDGE_THICKNESS_RATIO_LIMIT``); between a declared model's review ratio 0.35
    and that it is an INFO finding (``test_edge_review.py``)."""

    issues, _ = checks.element_issues(_solid(_lens(None, "2.95", "4", "12.2")))
    found = [i for i in issues if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert len(found) == 1 and found[0].severity == "WARNING"
    assert found[0].surface_id == "S1" and found[0].other_surface_id == "S2" and "% of its clear diameter 5.9 mm" in found[0].message
    assert found[0].limit_mm == pytest.approx(checks.EDGE_THICKNESS_RATIO_LIMIT * 5.9)
    issues, _ = checks.element_issues(_solid(_lens(None, "2.95", "4", "2")))  # 1.96 mm at the edge: 33 %
    assert "EDGE_THICKNESS_IMPLAUSIBLE" not in [i.code for i in issues]


def test_the_edge_thickness_warning_names_the_edge_field_of_the_elements_shape():
    """A rectangular element states a surface's own edge with edge_half_widths_mm, a round one with edge_radius_mm."""

    rows = [surface(1, "0", radius="200", clear="2.95"), surface(2, "12.2", clear="2.95")]
    plate = sequence(rows, ["N-BK7"])
    plate.pop("edge_radius_mm")
    plate["edge_half_widths_mm"] = ["4", "4"]
    issues, _ = checks.element_issues(_solid(SurfaceSequenceModelV1.model_validate(plate)))
    (found,) = [i for i in issues if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert found.message.endswith("states its own edge_half_widths_mm")
    issues, _ = checks.element_issues(_solid(_lens(None, "2.95", "4", "12.2")))
    (found,) = [i for i in issues if i.code == "EDGE_THICKNESS_IMPLAUSIBLE"]
    assert found.message.endswith("states its own edge_radius_mm")


def test_glass_far_beyond_the_clear_aperture_is_a_structured_warning():
    issues, _ = checks.element_issues(_solid(_lens(None, "2", "9", "3")))
    found = [i for i in issues if i.code == "EDGE_FAR_BEYOND_CLEAR_APERTURE"]
    assert [i.surface_id for i in found] == ["S1", "S2"] and all(i.severity == "WARNING" for i in found)
    assert "4.50 times its clear radius 2 mm" in found[0].message
    issues, _ = checks.element_issues(_solid(_lens(None, "2", "5", "3")))
    assert "EDGE_FAR_BEYOND_CLEAR_APERTURE" not in [i.code for i in issues]


def test_the_profile_domain_is_checked_out_to_each_surfaces_own_edge():
    steep = SurfaceSequenceModelV1.model_validate(
        sequence([surface(1, "0", radius="9.5", clear="8", edge_radius_mm="9"), surface(2, "8", clear="8")], ["N-BK7"], edge="12"))
    issues, _ = checks.element_issues(_solid(steep))
    assert "SAG_DOMAIN_INSIDE_OUTLINE" not in [i.code for i in issues]
    too_far = SurfaceSequenceModelV1.model_validate(
        sequence([surface(1, "0", radius="9.5", clear="8"), surface(2, "8", clear="8")], ["N-BK7"], edge="12"))
    issues, _ = checks.element_issues(_solid(too_far))
    assert "SAG_DOMAIN_INSIDE_OUTLINE" in [i.code for i in issues]


# ------------------------------------------------------------------------------------------------------- view


def _body(front_edge: float | None, back_edge: float | None, outline: float = 12.0) -> OpticBodyGeometryV1:
    s1 = {"surface_id": "S1", "role": "REFRACT", "vertex_z_mm": 0.0, "radius_mm": 15.0, "clear_radius_mm": 9.0}
    s2 = {"surface_id": "S2", "role": "REFRACT", "vertex_z_mm": 6.0, "radius_mm": 40.0, "clear_radius_mm": 9.0}
    if front_edge is not None:
        s1["edge_radius_mm"] = front_edge
    if back_edge is not None:
        s2["edge_radius_mm"] = back_edge
    return OpticBodyGeometryV1.model_validate({"diameter_mm": 2 * outline, "z_min_mm": 0.0, "z_max_mm": 6.0, "model_kind": "SURFACE_SEQUENCE",
                                               "fidelity": "SURFACE_PRESCRIPTION", "edge_radius_mm": outline,
                                               "basis": "PROJECT_DECLARED_MODEL", "surfaces": [s1, s2], "media": ["N-BK7"]})


def _open_edges(meshes) -> int:
    def key(p):
        return tuple(round(c, 9) for c in p)

    uses: Counter = Counter()
    for _, mesh in meshes:
        for t in range(0, len(mesh.indices), 3):
            tri = [key(mesh.positions[mesh.indices[t + i]]) for i in range(3)]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if a != b:
                    uses[frozenset((a, b))] += 1
    return sum(1 for n in uses.values() if n != 2)


@pytest.mark.parametrize("edges", [(None, None), (10.0, None), (None, 11.0), (10.0, 12.0)])
def test_an_edged_lens_mesh_is_closed(edges):
    assert _open_edges(draw_body(_body(*edges)).meshes()) == 0


def test_the_rim_ends_at_the_edge_and_the_wall_spans_the_real_edge_thickness():
    """A meniscus whose front surface (R 15) is edged at 10 mm inside a 12 mm element: the front profile stops at 10 mm,
    its flange is flat at that rim height out to 12 mm, and the edge wall runs from there to the back surface at 12 mm --
    not a cylinder from the front profile's (absent) height at 12 mm."""

    drawing = draw_body(_body(10.0, None))
    front, back = drawing.surfaces
    rim = 15.0 - math.sqrt(15.0 ** 2 - 10.0 ** 2)
    on_profile = [p for m, mesh in front.meshes if m != "optic_edge" for p in mesh.positions]
    assert max(math.hypot(p[0], p[1]) for p in on_profile) == pytest.approx(10.0, abs=1e-9)
    flange = [p for m, mesh in front.meshes if m == "optic_edge" for p in mesh.positions]
    assert flange and all(p[2] == pytest.approx(rim, abs=1e-9) for p in flange)
    assert max(math.hypot(p[0], p[1]) for p in flange) == pytest.approx(12.0, abs=1e-9)
    (wall,) = [mesh for m, mesh in drawing.body if m == "optic_edge"]
    zs = [p[2] for p in wall.positions]
    assert min(zs) == pytest.approx(rim, abs=1e-9)
    assert max(zs) == pytest.approx(6.0 + 40.0 - math.sqrt(40.0 ** 2 - 12.0 ** 2), abs=1e-9)
    assert all(math.hypot(p[0], p[1]) == pytest.approx(12.0, abs=1e-9) for p in wall.positions)


def test_the_section_draws_the_flange_and_the_glass_over_the_gap_outline():
    section = _sequence_section(_body(10.0, None), "X", 1e-4)
    front = section["surfaces"][0]
    rim = 15.0 - math.sqrt(15.0 ** 2 - 10.0 ** 2)
    points = [p for seg in front["segments"] for p in seg["points"]]
    assert front["edge_mm"] == [-10.0, 10.0]
    assert max(abs(t) for t, _ in points) == pytest.approx(12.0)
    assert all(z == pytest.approx(rim, abs=1e-6) for t, z in points if abs(t) >= 10.0)
    (medium,) = section["media"]
    assert medium["edge_thickness_mm"][1] == pytest.approx(6.0 + 40.0 - math.sqrt(40.0 ** 2 - 12.0 ** 2) - rim, abs=1e-6)


# -------------------------------------------------------------------------------------------- one entry point


def test_project_documents_are_read_by_one_function():
    document = with_models(project([source(), declared_component("E", "lens", "m")], [plane("P", (100.0, 0.0, 0.0))]),
                           [{"model_id": "m", "model": _group(("5", "5", None, None), "8")}], [nbk7()])
    parsed = parse_project_document(document)
    assert parsed.optical_models[0].model.surfaces[0].edge_radius_mm == "5"
    assert parsed.payload()["optical_models"][0]["model"]["surfaces"][2].get("edge_radius_mm") is None
    assert Decimal(parsed.optical_models[0].model.edge_radius_mm) == 8
