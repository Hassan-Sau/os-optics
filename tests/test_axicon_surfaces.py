"""Axicon (cone) surfaces: the ``linear_sag_coefficient`` term of a surface profile.

A ray parallel to the axis that refracts at a cone of base angle ``alpha`` (inside glass of index ``n``, out to air) leaves
at ``beta = asin(n sin alpha) - alpha`` from the axis toward it; a ring of such rays at height ``h`` crosses the axis
``h / tan(beta)`` behind the cone -- the Bessel zone of a beam of radius ``R`` ends ``R / tan(beta)`` behind it. These
references are computed here from Snell's law and the Sellmeier formula only, never with the Engine's own helpers.

Also: the apex (a ray along the axis passes undeviated), the cylinder (roof) variant, meshes on the exact cone, the
optical-solid check, prescription import, a catalog 3.1 product, Gaussian metrics out of model scope behind an axicon,
and the acylinder (a conic / asphere on a cylinder) against an independent 1-D computation.

Fixture data: every physical value below (shapes, slopes, the catalog product) is made up for the test.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics.contracts.optics import SagSurface, CircularAperture, RefractInteraction
from os_optics.local import json_api, load_public_catalog, run_optical_project
from os_optics.local.compile import ProjectResolver, compile_declared, paraxial_matrix
from os_optics.optical_view import surface_mesh as sm
from os_optics.optical_view.optic_mesh import draw_body
from os_optics.prescription import PrescriptionImportError, import_prescription

from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, sequence, surface, with_models
from local_helpers import catalog_document, plane, project, source

WAVELENGTH_NM = 780.0
SLOPE = 0.2  # tan(alpha): base angle alpha = atan(0.2) = 11.3 deg
THICKNESS = 5.0
X_AXIS = np.array([1.0, 0.0, 0.0])


def nbk7_index(wavelength_nm: float) -> float:
    """Sellmeier index of N-BK7 from its published coefficients (independent of the Engine's material code)."""

    lam2 = (wavelength_nm / 1000.0) ** 2
    b = [float(v) for v in NBK7_SELLMEIER["b"]]
    c = [float(v) for v in NBK7_SELLMEIER["c_um2"]]
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def deflection(alpha: float, n: float) -> float:
    """Snell: a ray parallel to the axis, inside glass, meets the cone at incidence alpha and leaves at alpha + beta."""

    return math.asin(n * math.sin(alpha)) - alpha


def axicon_model(slope: str = f"-{SLOPE}", **surface_fields) -> dict:
    """A plano-axicon: flat front at z = 0, cone at z = 5 with its apex on the axis, falling back toward the front."""

    return {"model_id": "ax", "model": sequence([surface(1, "0"), surface(2, "5", linear_sag_coefficient=slope, **surface_fields)],
                                                ["N-BK7"])}


def axicon_project(*, height_mm: float = 0.0, lateral_mm: float = 0.0, targets=None, model=None, waist_mm: float = 0.5) -> dict:
    # the element's local Z is the global X axis, local X the global Y, local Y the global Z (local_helpers.ALONG_X)
    document = project([source(position=(-200.0, lateral_mm, height_mm), waist_mm=waist_mm, wavelength_nm=WAVELENGTH_NM),
                        declared_component("AX", "lens", "ax")], [plane("P", (80.0, 0.0, 0.0), radius_mm=25.0)], targets)
    return with_models(document, [model or axicon_model()], [nbk7()])


def _after(run, surface_id: str = "AX/S2"):
    """(hit point, direction) of the chief ray where it leaves ``surface_id``."""

    path = run.result.paths[0]
    for k, segment in enumerate(path.segments):
        if segment.end_surface_id == surface_id:
            leaving = path.segments[k + 1]
            d = np.subtract(leaving.end_mm, leaving.start_mm)
            return np.array(leaving.start_mm), d / np.linalg.norm(d)
    raise AssertionError(f"the chief ray does not reach {surface_id}")


@pytest.mark.parametrize("height_mm", [1.0, 3.0, 7.5])
def test_deflection_matches_snell_at_the_cone(height_mm: float) -> None:
    run = run_optical_project(axicon_project(height_mm=height_mm))
    hit, d = _after(run)
    n = nbk7_index(WAVELENGTH_NM)
    beta = deflection(math.atan(SLOPE), n)
    angle = math.acos(float(d @ X_AXIS))
    assert angle == pytest.approx(beta, abs=1e-12)
    assert d[2] < 0.0 and abs(d[1]) < 1e-15  # toward the axis, in the meridional plane
    # the ray meets the cone where the sag says: x = 5 - 0.2 h
    assert hit[0] == pytest.approx(THICKNESS - SLOPE * height_mm, abs=1e-12) and hit[2] == pytest.approx(height_mm, abs=1e-12)


def test_bessel_zone_length_is_the_axis_crossing_of_the_rim_rays() -> None:
    height = 6.0
    run = run_optical_project(axicon_project(height_mm=height))
    hit, d = _after(run)
    crossing = hit[0] + hit[2] / math.tan(math.acos(float(d @ X_AXIS)))  # where the chief ray meets the axis
    beta = deflection(math.atan(SLOPE), nbk7_index(WAVELENGTH_NM))
    assert crossing - hit[0] == pytest.approx(height / math.tan(beta), rel=1e-12)
    # the ray path really crosses there: its z falls to 0 at the computed x
    s = (crossing - hit[0]) / d[0]
    assert abs(hit[2] + s * d[2]) < 1e-9


def test_the_apex_passes_a_ray_along_the_axis_undeviated_and_pupil_rays_form_a_cone() -> None:
    run = run_optical_project(axicon_project())
    _, d = _after(run)
    assert np.allclose(d, X_AXIS, atol=1e-15)
    beta = deflection(math.atan(SLOPE), nbk7_index(WAVELENGTH_NM))
    (ray_path,) = run.ray_paths.paths
    angles = []
    for ray in ray_path.pupil_rays:
        if ray.exit_direction is None:
            continue
        exit_dir = np.array(ray.exit_direction)
        if abs(float(exit_dir @ X_AXIS) - 1.0) > 1e-12:  # off-apex rays: all on the cone of half-angle beta
            angles.append(math.acos(float(exit_dir @ X_AXIS)))
    assert angles and max(abs(a - beta) for a in angles) < 1e-12


def test_roof_variant_deflects_across_its_straight_axis_only() -> None:
    # a cylinder straight along local X (global Y): the curved coordinate is local Y (global Z)
    roof = axicon_model(cylinder_axis="X")
    beta = deflection(math.atan(SLOPE), nbk7_index(WAVELENGTH_NM))
    across = run_optical_project(axicon_project(height_mm=4.0, model=roof))
    _, d = _after(across)
    assert math.acos(float(d @ X_AXIS)) == pytest.approx(beta, abs=1e-12) and abs(d[1]) < 1e-15
    along = run_optical_project(axicon_project(lateral_mm=4.0, model=roof))  # on the ridge: no slope across it
    _, d = _after(along)
    assert np.allclose(d, X_AXIS, atol=1e-15)
    both = run_optical_project(axicon_project(height_mm=-2.0, lateral_mm=3.0, model=roof))
    hit, d = _after(both)
    assert abs(d[1]) < 1e-15 and d[2] > 0.0 and math.acos(float(d @ X_AXIS)) == pytest.approx(beta, abs=1e-12)
    assert hit[0] == pytest.approx(THICKNESS - SLOPE * 2.0, abs=1e-12)


def test_gaussian_metrics_are_out_of_model_scope_behind_an_axicon_while_power_is_traced() -> None:
    targets = [{"target_id": "power", "observation_id": "P", "metric": "power_W", "minimum": 0.0},
               {"target_id": "radius", "observation_id": "P", "metric": "beam_radius_mm", "maximum": 5.0},
               {"target_id": "spot", "observation_id": "P", "metric": "spot_rms_mm", "minimum": 0.0}]
    run = run_optical_project(axicon_project(targets=targets))
    status = {t.target_id: t.status.value for t in run.result.targets}
    assert status == {"power": "PASS", "radius": "OUT_OF_MODEL_SCOPE", "spot": "PASS"}
    codes = {(i.code.value, i.status.value) for i in run.result.issues}
    assert ("GAUSSIAN_UNDEFINED_AFTER_AXICON", "OUT_OF_MODEL_SCOPE") in codes
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.beam.gaussian.valid is False and arrival.beam.gaussian.reason == "GAUSSIAN_UNDEFINED_AFTER_AXICON"
    assert arrival.first_order is None  # a cone is not a linear map about the chief ray
    # two Fresnel interfaces of N-BK7 at normal incidence (the front) and at alpha (the cone): the power is the Engine's own
    assert 0.9 * 0.001 < run.result.targets[0].value < 0.001


def test_mesh_lies_on_the_cone_within_the_drawing_tolerance() -> None:
    run = run_optical_project(axicon_project())
    (element,) = [e for e in run.optical_view.elements if e.element_id == "AX"]
    geom = element.geometry
    cone = geom.surfaces[1]
    assert cone.linear_sag_coefficient == -SLOPE
    drawing = draw_body(geom)
    meshes = drawing.surfaces[1].meshes
    profile = sm.Profile.of(cone)
    assert sm.max_deviation(meshes, profile, sm.Placement.coaxial(THICKNESS)) <= sm.CHORD_TOLERANCE_MM
    worst, apex_drawn = 0.0, False
    for material, mesh in meshes:
        for x, y, z in mesh.positions:
            rho = math.hypot(x, y)
            if rho <= 11.43:
                worst = max(worst, abs(z - (THICKNESS - SLOPE * rho)))
            apex_drawn = apex_drawn or rho == 0.0
    assert worst < 1e-12 and apex_drawn  # every vertex on the cone, one at its apex


def test_optical_solid_check_finds_a_cone_that_cuts_through_the_front_face() -> None:
    sound = json_api.inspect_optical_solids(axicon_project())
    assert sound["ok"] is True
    # clear aperture 9 mm, edge 10 mm: the cone stays in front of S1 over the clear aperture (z = 5 - 0.52 * 9 = 0.32) but
    # cuts through it before the edge (z = 5 - 5.2 = -0.2)
    steep = axicon_model(slope="-0.52")
    report = json_api.inspect_optical_solids(axicon_project(model=steep))
    text = str(report)
    assert "SURFACES_INTERSECT" in text


def test_paraxial_matrix_refuses_an_axicon() -> None:
    document = axicon_project()
    from os_optics.local.catalog import load_project

    loaded = load_project(document)
    element = compile_declared("AX", loaded.optical_models[0], ProjectResolver(loaded))
    with pytest.raises(ValueError, match="axicon"):
        paraxial_matrix(element, WAVELENGTH_NM)


def test_a_catalog_3_1_axicon_product_is_read_and_traced() -> None:
    document = catalog_document()
    window = next(p for p in document["products"] if p["optical_sku"] == "OS-OPT-TEST-WIN-NBK7-T3")
    product = copy.deepcopy(window)
    # the cone carries a small A4 term: the catalog's ASPHERIC_LENS form needs a conic or polynomial surface
    product.update(optical_sku="OS-OPT-TEST-AXICON", category="LENS", form="ASPHERIC_LENS", display_name="Fixture axicon",
                   display_name_zh_cn="测试锥透镜", summary="Fixture axicon.")
    # fixture specifications (a LENS product publishes a focal length and design wavelength)
    product["specifications"].update(focal_length_mm={"value": "100", "status": "NOMINAL"},
                                     design_wavelength_nm={"value": "780", "status": "NOMINAL"})
    model = product["simulation_models"][0]
    model["surfaces"][1].update(vertex_z_mm="5", linear_sag_coefficient=f"-{SLOPE}", aspheric_coefficients=["0.0001"])
    product["design_geometry"]["z_max_mm"] = "5"
    product["specifications"]["thickness_mm"]["value"] = "5"
    document["products"].append(product)
    document["schema_version"] = "3.1"
    document["categories_offered"] = sorted(set(document["categories_offered"]) | {"LENS"})
    catalog = load_public_catalog(document)
    assert catalog.get("OS-OPT-TEST-AXICON").simulation_models[0].surfaces[1].is_axicon
    component = {"instance_id": "AX", "optical_role": "lens", "product": {"optical_sku": "OS-OPT-TEST-AXICON"},
                 "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}
    doc = project([source(position=(-200.0, 0.0, 3.0), waist_mm=0.5), component], [plane("P", (80.0, 0.0, 0.0), radius_mm=25.0)])
    run = run_optical_project(doc, catalog=document)
    hit, d = _after(run)
    local_slope = SLOPE - 4.0 * 0.0001 * 3.0**3  # |dz / d rho| of -0.2 rho + A4 rho^4 at rho = 3
    beta = deflection(math.atan(local_slope), nbk7_index(780.0))
    assert math.acos(float(d @ X_AXIS)) == pytest.approx(beta, abs=1e-12)
    assert hit[0] == pytest.approx(THICKNESS - SLOPE * 3.0 + 0.0001 * 3.0**4, abs=1e-12)


AXICON_CSV = """#! format = csv-surface-table/1
#! units = mm
[surfaces]
surface,type,radius,thickness,material,semi_diameter,linear_sag
OBJ,,,inf,,,
1,,,5,GLASS,10,
2,AXICON,,40,,10,-0.2
IMA,,,,,,
[materials]
name,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm
GLASS,SELLMEIER,1.03961212,0.231792344,1.01046945,0.00600069867,0.0200179144,103.560653,400,1100
"""


def test_prescription_import_reads_the_axicon_term() -> None:
    result = import_prescription(AXICON_CSV, glass_source="inline", model_id="ax")
    lens = next(m.model for m in result.optical_models if m.model.model_kind == "SURFACE_SEQUENCE")
    assert [s.linear_sag_coefficient for s in lens.surfaces] == [None, "-0.2"]
    assert lens.surfaces[1].sag_at(Decimal(10)) == Decimal("-2.0")
    bad = AXICON_CSV.replace("2,AXICON,,40,,10,-0.2", "2,STANDARD,,40,,10,-0.2")
    with pytest.raises(PrescriptionImportError, match="belongs to type AXICON"):
        import_prescription(bad, glass_source="inline", model_id="ax")


# ------------------------------------------------------------------------------------------------ acylinder

ACYL_RADIUS, ACYL_CONIC = 30.0, -1.0  # a parabolic cylinder, curved along local Y (global Z)


def acylinder_model() -> dict:
    return {"model_id": "ax", "model": sequence([surface(1, "0", radius="30", conic="-1", cylinder_axis="X"),
                                                 surface(2, "6")], ["N-BK7"])}


def sag_1d(y: float) -> float:
    c = 1.0 / ACYL_RADIUS
    return c * y * y / (1.0 + math.sqrt(1.0 - (1.0 + ACYL_CONIC) * c * c * y * y))


def test_acylinder_traces_its_conic_profile_against_an_independent_1d_computation() -> None:
    height = 5.0
    run = run_optical_project(axicon_project(height_mm=height, lateral_mm=2.0, model=acylinder_model()))
    path = run.result.paths[0]
    first = next(s for s in path.segments if s.end_surface_id == "AX/S1")
    assert first.end_mm[0] == pytest.approx(sag_1d(height), abs=1e-12)  # the hit lies on the 1-D sag, whatever x (straight axis)
    # refraction in the (x, z) plane at the 1-D slope dz/dy, independently: the surface normal (-z'(h), 1) in (axial, height)
    slope = height / ACYL_RADIUS  # z'(y) of a paraboloid: c y
    normal = np.array([1.0, -slope]) / math.hypot(1.0, slope)  # (axial, height), toward +axial
    d_in = np.array([1.0, 0.0])
    eta = 1.0 / nbk7_index(WAVELENGTH_NM)
    cos_i = float(d_in @ normal)
    against = -normal  # oriented against the incident ray: d_in . against = -cos_i
    t = eta * d_in + (eta * cos_i - math.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))) * against
    t = t / np.linalg.norm(t)
    inside = next(s for s in path.segments if s.start_mm == first.end_mm)
    d = np.subtract(inside.end_mm, inside.start_mm)
    d = d / np.linalg.norm(d)
    assert d[0] == pytest.approx(t[0], abs=1e-12) and d[2] == pytest.approx(t[1], abs=1e-12) and abs(d[1]) < 1e-15


def test_acylinder_low_level_surface_has_the_conic_on_its_curved_axis() -> None:
    s = SagSurface(surface_id="S", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(0.0, 1.0, 0.0),
                   curvature_per_mm=1.0 / ACYL_RADIUS, conic=ACYL_CONIC, cylindrical=True, aperture=CircularAperture(radius_mm=9.0),
                   interaction=RefractInteraction())
    assert "linear_sag_coefficient" not in s.model_dump()  # no axicon term: the earlier serialized form


def test_a_zero_axicon_term_is_no_axicon_term() -> None:
    """``linear_sag_coefficient`` 0 (or -0.0) is the surface without an axicon term: one form, one serialization."""

    fields = dict(surface_id="S", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(0.0, 1.0, 0.0), curvature_per_mm=0.02,
                  aperture=CircularAperture(radius_mm=9.0), interaction=RefractInteraction())
    plain = SagSurface(**fields)
    for zero in (0.0, -0.0, 0):
        surface = SagSurface(**fields, linear_sag_coefficient=zero)
        assert surface.linear_sag_coefficient is None and surface == plain and surface.model_dump() == plain.model_dump()
