"""Optical-solid validity (``os_optics.optical_solids``), against closed-form references.

* the seed catalog has no invalid optical solid (census of every published model);
* P1 a biconvex lens whose surfaces cross between the clear aperture and the edge: ``SURFACES_INTERSECT`` with the
  closed-form edge thickness ``CT - 2 (R - sqrt(R^2 - r^2))``; P2 the same crossing inside the clear aperture (a model the
  catalog contract already refuses); a knife edge (zero edge thickness) is a warning, not an error;
* P3 a conic whose domain ``R / sqrt(1 + k)`` ends inside the outline: ``SAG_DOMAIN_INSIDE_OUTLINE``;
* P4 two placed achromats: overlapping by exactly the axial interpenetration of their vertices, touching vertex to
  vertex (``OPTICAL_BODIES_TOUCH``, valid), apart, turned around, and tilted off the axis (the sampled search);
* a compiled element whose surfaces' media do not chain: ``MEDIUM_BOUNDARY_INCONSISTENT``;
* members of a compound unit inside / outside its envelope;
* the JSON API and the CLI report the same findings with stable codes and locations; lab mechanics is never needed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from catalog_store_helpers import REAL_CATALOG
from compound_objective import AGAINST_X, ALONG_X, objective_project, pair_project, thick_achromat
from os_optics._vendor.shared_contracts.design_catalog import CylinderGeometryV1, PrescriptionSurfaceV1, SurfaceSequenceModelV1
from os_optics.local import LocalRunError, LocalRunErrorCode, load_public_catalog, lower_project
from os_optics.local.compile import compile_element
from os_optics.optical_solids import (
    TOLERANCE_MM,
    PlacedElement,
    UnitEnvelope,
    body_of,
    element_issues,
    inspect_lowered,
    inspect_optical_solids,
)


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


def test_the_seed_catalog_has_no_invalid_optical_solid(catalog):
    thinnest = math.inf
    for product in catalog.products:
        for model in product.simulation_models:
            element = compile_element("E", product, catalog, model_kind=model.model_kind)
            issues, gaps = element_issues(body_of(element, design_geometry=product.design_geometry))
            assert not issues, (product.optical_sku, [i.message for i in issues])
            thinnest = min([thinnest] + [g["min_thickness_mm"] for g in gaps if g["medium"] != "AIR"])
    assert 0.0 < thinnest < 1.0  # the thinnest glass edge of the release is reported, and positive


# ------------------------------------------------------------------------------------------------ single elements


def _biconvex(catalog, *, r1: str, r2: str, ct: str, clear: str, edge: str) -> dict:
    base = next(p for p in sorted(catalog.products, key=lambda p: p.optical_sku)
                if p.form.value == "BICONVEX_LENS" and p.simulation_models[0].model_kind == "SURFACE_SEQUENCE")
    model = base.simulation_models[0].model_dump(mode="json")
    model["edge_radius_mm"] = edge
    model["surfaces"] = model["surfaces"][:2]
    model["surfaces"][0].update({"vertex_z_mm": "0", "radius_mm": r1, "clear_radius_mm": clear})
    model["surfaces"][1].update({"vertex_z_mm": ct, "radius_mm": r2, "clear_radius_mm": clear})
    model["media"] = model["media"][:1]
    return model


def _element(model) -> SimpleNamespace:
    return SimpleNamespace(instance_id="L1", model=model, surfaces=(), solid=None)


def _unvalidated(model: dict) -> SurfaceSequenceModelV1:
    surfaces = tuple(PrescriptionSurfaceV1.model_construct(**s) for s in model["surfaces"])
    return SurfaceSequenceModelV1.model_construct(**{**model, "surfaces": surfaces, "media": tuple(model["media"])})


def _edge_thickness(ct: float, r: float, edge: float) -> float:
    return ct - 2.0 * (r - math.sqrt(r * r - edge * edge))


def test_p1_surfaces_crossing_beyond_the_clear_aperture_are_refused(catalog):
    model = SurfaceSequenceModelV1.model_validate(_biconvex(catalog, r1="20", r2="-20", ct="2", clear="5", edge="12.7"))  # the contract accepts it
    issues, gaps = element_issues(body_of(_element(model)))
    (issue,) = issues
    assert (issue.code, issue.severity, issue.surface_id, issue.other_surface_id) == ("SURFACES_INTERSECT", "ERROR", "S1", "S2")
    assert issue.value_mm == pytest.approx(_edge_thickness(2.0, 20.0, 12.7), abs=1e-6)
    assert issue.local_position_mm[:2] == pytest.approx((12.7, 0.0), abs=1e-6) and "between the clear aperture and the outline" in issue.message
    assert issue.location == "component L1 surface S1"
    # the two profiles are 2 - 2 x 0.635 = 0.73 mm apart at the 5 mm clear radius: the finding lies beyond the traced surfaces
    assert issue.zone == "OUTSIDE_CLEAR_APERTURE" and issue.as_dict()["zone"] == "OUTSIDE_CLEAR_APERTURE" and issue.details()["zone"] == issue.zone
    assert gaps[0]["clear_aperture_min_thickness_mm"] == pytest.approx(_edge_thickness(2.0, 20.0, 5.0), abs=1e-6)
    (gap,) = gaps
    assert (gap["centre_thickness_mm"], gap["edge_thickness_mm"]) == (2.0, pytest.approx(_edge_thickness(2.0, 20.0, 12.7), abs=1e-6))


def test_p2_surfaces_crossing_inside_the_clear_aperture_are_refused(catalog):
    data = _biconvex(catalog, r1="20", r2="-20", ct="2", clear="12", edge="12.7")
    with pytest.raises(ValueError):
        SurfaceSequenceModelV1.model_validate(data)  # the catalog contract refuses it too
    issues, _ = element_issues(body_of(_element(_unvalidated(data))))
    assert _codes(issues) == ["SURFACES_INTERSECT"]
    # it crosses inside the clear aperture already: zero thickness where 2 = 2 (20 - sqrt(400 - r^2))
    crossing = math.sqrt(400.0 - 19.0 ** 2)
    assert crossing < 12.0
    # the thinnest point reported is at the 12.7 mm outline, beyond the 12 mm clear radius; the zone is the clear aperture
    # all the same, because the surfaces cross inside it
    assert issues[0].local_position_mm[0] == pytest.approx(12.7, abs=1e-6) and issues[0].zone == "CLEAR_APERTURE"


def test_a_knife_edge_is_a_warning_not_an_error(catalog):
    with localcontext() as ctx:
        ctx.prec = 30
        ct = 2 * (Decimal(20) - (Decimal(400) - Decimal(100)).sqrt())  # zero edge thickness at the 10 mm edge
    model = SurfaceSequenceModelV1.model_validate(_biconvex(catalog, r1="20", r2="-20", ct=str(ct.quantize(Decimal("1e-15"))), clear="9", edge="10"))
    issues, gaps = element_issues(body_of(_element(model)))
    assert [(i.code, i.severity) for i in issues] == [("KNIFE_EDGE", "WARNING")]
    assert abs(gaps[0]["min_thickness_mm"]) <= TOLERANCE_MM
    assert issues[0].zone is None and "zone" not in issues[0].as_dict()  # (the zone is stated on the ERROR findings only)


def test_p3_a_profile_domain_ending_inside_the_outline_is_refused(catalog):
    data = _biconvex(catalog, r1="10", r2="-20", ct="6", clear="5", edge="12.7")
    data["surfaces"][0]["conic"] = "0.5"
    data["surfaces"][1].pop("radius_mm")
    model = SurfaceSequenceModelV1.model_validate(data)  # the contract checks the clear aperture only
    issues, _ = element_issues(body_of(_element(model)))
    domain = [i for i in issues if i.code == "SAG_DOMAIN_INSIDE_OUTLINE"]
    (issue,) = domain
    assert issue.surface_id == "S1" and issue.value_mm == pytest.approx(10.0 / math.sqrt(1.5), abs=1e-6) and issue.limit_mm == pytest.approx(12.7)
    assert issue.zone == "OUTSIDE_CLEAR_APERTURE"  # the profile ends at 8.16 mm, beyond the 5 mm clear radius
    data["surfaces"][0]["clear_radius_mm"] = "9"  # ... and inside a 9 mm clear aperture
    (inside,) = [i for i in element_issues(body_of(_element(_unvalidated(data))))[0] if i.code == "SAG_DOMAIN_INSIDE_OUTLINE"]
    assert inside.zone == "CLEAR_APERTURE"
    # within its domain the conic also reaches the plane at z = 6 (sag(rho_max) = 10 / 1.5 > 6): the surfaces cross there too
    assert "SURFACES_INTERSECT" in _codes(issues)


def test_a_medium_chain_that_does_not_close_is_refused(catalog):
    product = thick_achromat(catalog)
    element = compile_element("L1", product, catalog)
    first = element.surfaces[0]
    swapped = first.model_copy(update={"medium_positive": first.medium_negative, "medium_negative": first.medium_positive})
    tampered = dataclasses.replace(element, surfaces=(swapped, *element.surfaces[1:]))
    issues, _ = element_issues(body_of(tampered, design_geometry=product.design_geometry))
    assert _codes(issues) == ["MEDIUM_BOUNDARY_INCONSISTENT"] and issues[0].surface_id == "S1"
    assert not element_issues(body_of(element, design_geometry=product.design_geometry))[0]


# ------------------------------------------------------------------------------------------------ pairs


def _report(catalog, project):
    return inspect_lowered(lower_project(project, catalog, check_solids=False), project)


def test_the_lowering_refuses_overlapping_bodies_before_tracing(catalog):
    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    with pytest.raises(LocalRunError) as refused:
        lower_project(pair_project(catalog, thickness / 2.0), catalog)
    assert [(p.code, p.location) for p in refused.value.problems] == [(LocalRunErrorCode.OPTICAL_BODIES_OVERLAP, "component L1")]
    with pytest.raises(LocalRunError) as touching:  # vertex to vertex on the axis: light would cross with no air gap
        lower_project(pair_project(catalog, thickness), catalog)
    (problem,) = touching.value.problems
    assert (problem.code, problem.location) == (LocalRunErrorCode.OPTICAL_BODIES_TOUCH, "component L1 surface S3")
    assert problem.details["other_instance_id"] == "L2" and problem.details["other_surface_id"] == "S1"
    assert touching.value.optical_solids["status"] == "INVALID"
    apart = lower_project(pair_project(catalog, thickness + 1e-3), catalog)  # a micrometre of air is a valid design
    assert apart.solids.valid and apart.solids.issues == ()


def test_p4_overlapping_lenses_are_refused_with_the_exact_interpenetration(catalog):
    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    report = _report(catalog, pair_project(catalog, thickness / 2.0))
    (issue,) = report.issues
    assert (issue.code, issue.instance_id, issue.other_instance_id) == ("OPTICAL_BODIES_OVERLAP", "L1", "L2")
    # coaxial: the two glass bodies share the axial span [t/2, t] on the axis, and at no radius more than that
    assert issue.value_mm == pytest.approx(thickness / 2.0, abs=1e-6)
    assert not report.valid and [p.code for p in report.local_run_problems()] == [LocalRunErrorCode.OPTICAL_BODIES_OVERLAP]
    assert report.local_run_problems()[0].location == "component L1"


def test_p4_lenses_touching_vertex_to_vertex_on_the_axis_are_refused(catalog):
    """m1: the trace cannot pass two surfaces at one point (the second one lies closer than the tracer's minimum advance),
    so a contact where light crosses from one body into the other is refused with an actionable message."""

    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    touching = _report(catalog, pair_project(catalog, thickness))
    (issue,) = touching.issues
    assert (issue.code, issue.severity, issue.surface_id, issue.other_surface_id) == ("OPTICAL_BODIES_TOUCH", "ERROR", "S3", "S1")
    assert not touching.valid and "cemented" in issue.message and "separate them" in issue.message
    assert issue.position_mm == pytest.approx((thickness, 0.0, 0.0), abs=1e-6)
    apart = _report(catalog, pair_project(catalog, thickness + 0.5))
    assert apart.issues == () and apart.valid and apart.pairs_checked == 1


def test_p4_a_lens_turned_around_into_another_is_refused(catalog):
    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    # L2 faces -X with its first vertex at x = 1.5 t: its glass spans [t/2, 3t/2] on the axis, L1's [0, t]
    report = _report(catalog, pair_project(catalog, 1.5 * thickness, AGAINST_X))
    (issue,) = report.issues
    assert issue.code == "OPTICAL_BODIES_OVERLAP" and issue.value_mm == pytest.approx(thickness / 2.0, abs=1e-6)


def test_p4_a_tilted_lens_cutting_another_is_refused(catalog):
    half = math.radians(20.0) / 2.0
    tilt = (math.cos(half), math.sin(half), 0.0, 0.0)
    w1, x1, y1, z1 = ALONG_X
    w2, x2, y2, z2 = tilt
    q = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    thickness = float(Decimal(thick_achromat(catalog).simulation_models[0].surfaces[-1].vertex_z_mm))
    report = _report(catalog, pair_project(catalog, thickness * 0.8, q, 3.0))
    (issue,) = report.issues
    assert issue.code == "OPTICAL_BODIES_OVERLAP" and issue.value_mm > 0.5
    # the reported point lies inside both glass bodies
    from os_optics.optical_solids.checks import PlacedBody

    lowered = lower_project(pair_project(catalog, thickness * 0.8, q, 3.0), catalog, check_solids=False)
    product = thick_achromat(catalog)
    poses = {"L1": ((0.0, 0.0, 0.0), ALONG_X), "L2": ((thickness * 0.8, 3.0, 0.0), q)}
    for element in lowered.elements:
        placed = PlacedBody(body_of(element, design_geometry=product.design_geometry), *poses[element.instance_id])
        assert placed.body.depth(placed.to_local(issue.position_mm)) > 0.0


def test_the_compound_objective_is_valid(catalog):
    report = _report(catalog, objective_project(catalog))
    assert report.valid and report.issues == () and report.pairs_checked == 3
    for element in report.elements:
        assert all(g["min_thickness_mm"] > 0.0 for g in element["gaps"])


# ------------------------------------------------------------------------------------------------ compound units


def test_members_must_lie_inside_their_unit_envelope(catalog):
    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    l1, l2 = compile_element("L1", product, catalog), compile_element("L2", product, catalog)
    members = [PlacedElement(l1, (0.0, 0.0, 0.0), ALONG_X, member_id="L1", design_geometry=product.design_geometry, instance_id="OBJ"),
               PlacedElement(l2, (thickness + 1.0, 0.0, 0.0), ALONG_X, member_id="L2", design_geometry=product.design_geometry, instance_id="OBJ")]

    def envelope(z_max: float) -> UnitEnvelope:
        return UnitEnvelope("OBJ", CylinderGeometryV1(diameter_mm="30", z_min_mm="-1", z_max_mm=format(Decimal(repr(z_max)).normalize(), "f"), status="NOMINAL"), (0.0, 0.0, 0.0), ALONG_X)

    roomy = inspect_optical_solids(members, units=[envelope(2 * thickness + 2.0)])
    assert roomy.valid and roomy.issues == ()
    tight = inspect_optical_solids(members, units=[envelope(thickness + 1.5)])
    (issue,) = tight.issues
    assert (issue.code, issue.instance_id, issue.member_id, issue.zone) == ("MEMBER_OUTSIDE_ENVELOPE", "OBJ", "L2", "UNIT_ENVELOPE")
    assert issue.value_mm == pytest.approx(thickness - 0.5, abs=1e-3)  # L2's far vertex at 2t + 1, the envelope ends at t + 1.5
    assert issue.location == "component OBJ member L2"


# ------------------------------------------------------------------------------------------------ JSON API / CLI


def test_the_json_api_reports_solids_with_stable_codes(catalog, real_catalog_store):
    from os_optics.local import json_api

    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    valid = json_api.inspect_optical_solids(objective_project(catalog).payload())
    assert valid["ok"] and valid["status"] == "VALID" and valid["schema"] == "opensci.optical_solids_report"
    invalid = json_api.inspect_optical_solids(pair_project(catalog, thickness / 2.0).payload())
    assert invalid["ok"] and invalid["status"] == "INVALID"
    (issue,) = invalid["issues"]
    assert issue["code"] == "OPTICAL_BODIES_OVERLAP" and issue["location"] == "component L1" and issue["other_instance_id"] == "L2"
    assert json.loads(json.dumps(invalid)) == invalid  # JSON-safe


def test_the_cli_checks_a_project_file(catalog, real_catalog_store, tmp_path):
    product = thick_achromat(catalog)
    thickness = float(Decimal(product.simulation_models[0].surfaces[-1].vertex_z_mm))
    for name, project, code, status in (("ok", objective_project(catalog), 0, "VALID"), ("bad", pair_project(catalog, thickness / 2.0), 1, "INVALID")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(project.payload()), encoding="utf-8")
        out = subprocess.run([sys.executable, "-m", "os_optics", "project", "solids", str(path), "--catalog", str(REAL_CATALOG)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == code, out.stderr
        assert json.loads(out.stdout)["status"] == status
