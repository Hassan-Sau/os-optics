"""``os_optics.local.compile``: the generic public-model -> Engine-element rules (surfaces, media, materials,
coatings, interactions, apertures, polarization elements) and their exact numbers."""

from __future__ import annotations

import math
from decimal import Context, Decimal, localcontext

import pytest

from os_optics.contracts import (
    BeamSplitterInteraction,
    ConstantCoating,
    MirrorInteraction,
    PlaneSurface,
    PolarizerInteraction,
    RefractInteraction,
    RetarderInteraction,
    SellmeierMaterial,
    SphereSurface,
    TabulatedCoating,
    TabulatedIndexMaterial,
)
from os_optics.local import (
    LocalRunError,
    PublicCoatingV1,
    compile_element,
    engine_coating,
    engine_material,
    engine_material_id,
    load_public_catalog,
    lower_project,
    paraxial_matrix,
)
from os_optics.local.compile import POLARIZATION_SURFACE_ID, select_model
from os_optics._vendor.shared_contracts.optical_design.optical_model import PropagationEnvironmentV1

#: The paraxial checks below compare with independent matrices in a unit-index surrounding medium: compile for vacuum (a
#: compiled element otherwise records air at 20 C and 101325 Pa, the default environment, and its matrix is relative to it).
VACUUM_ENVIRONMENT = PropagationEnvironmentV1(pressure_pa="0")

from local_helpers import (
    CONCAVE_MIRROR,
    DICHROIC,
    FLAT_MIRROR,
    HWP,
    PCV_FN50,
    PCX_F100,
    PLATE_BS,
    POLARIZER,
    WINDOW,
    catalog_document,
    element,
    lens_matrix,
    load_test_catalog,
    plane,
    product_document,
    project,
    source,
)

EDGE, CLEAR = 12.7, 11.43
DEFAULT_COATING_RANGE = ((100.0, 20000.0), 90.0)


def _compiled(sku: str, document: dict | None = None):
    catalog = load_public_catalog(document) if document is not None else load_test_catalog()
    return compile_element("X", catalog.get(sku), catalog, environment=VACUUM_ENVIRONMENT)


def _constant(r: float, t: float) -> ConstantCoating:
    return ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=t, transmittance_p=t)


def _values(coating):
    """A compiled coating without the record it carries (``declared_validity``: its own test is in ``test_coating_use.py``)."""
    return coating.model_copy(update={"declared_validity": None})


def _plain(interaction):
    """A compiled interaction with its coating's values only (``_values``)."""
    return interaction.model_copy(update={"coating": _values(interaction.coating)})


def test_a_plano_convex_lens_is_a_sphere_then_a_plane_between_air_and_its_glass():
    element_ = _compiled(PCX_F100)
    s1, s2 = element_.surfaces
    assert isinstance(s1, SphereSurface) and isinstance(s2, PlaneSurface)
    assert (s1.surface_id, s1.center_mm, s1.radius_mm, s1.cap_axis) == ("S1", (0.0, 0.0, 51.5), 51.5, (0.0, 0.0, -1.0))
    assert (s1.aperture_radius_mm, s1.mount_margin_mm) == (CLEAR, float(Decimal("12.7") - Decimal("11.43")))
    # centre on +Z: the side away from the centre is before the surface (air)
    assert (s1.medium_positive, s1.medium_negative) == ("air", "catalog:N-BK7")
    assert (s2.surface_id, s2.origin_mm, s2.normal) == ("S2", (0.0, 0.0, 3.6), (0.0, 0.0, 1.0))
    assert (s2.medium_positive, s2.medium_negative) == ("air", "catalog:N-BK7")
    assert s2.aperture.radius_mm == CLEAR and s2.aperture.radius_mm + s2.mount_margin_mm == pytest.approx(EDGE, abs=1e-12)
    for surface in (s1, s2):
        assert _plain(surface.interaction) == RefractInteraction(coating=_constant(0.005, 0.995), trace_fresnel_reflection=False)
        assert (surface.interaction.coating.valid_wavelength_nm, surface.interaction.coating.max_aoi_deg) == DEFAULT_COATING_RANGE
    (material,) = element_.materials
    assert material == SellmeierMaterial(material_id="catalog:N-BK7", valid_wavelength_nm=(300.0, 2500.0),
                                         b=(1.03961212, 0.231792344, 1.01046945), c_um2=(0.00600069867, 0.0200179144, 103.560653))
    assert [(e.surface_id, e.traced, e.coating.coating_id, e.valid_aoi_deg.max_deg) for e in element_.envelopes] == [
        ("S1", "TRANSMITTED", "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10", "10"),
        ("S2", "TRANSMITTED", "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10", "10"),
    ]
    assert (element_.optical_sku, element_.model_kind, element_.valid_band_nm.min_nm) == (PCX_F100, "SURFACE_SEQUENCE", "400")
    assert element_.engine_surface_id("S2") == "X/S2"


def test_a_negative_radius_puts_the_centre_on_minus_z_and_the_glass_on_the_positive_side():
    s1, _ = _compiled(PCV_FN50).surfaces
    assert (s1.center_mm, s1.radius_mm, s1.cap_axis) == ((0.0, 0.0, -25.7), 25.7, (0.0, 0.0, 1.0))
    assert (s1.medium_positive, s1.medium_negative) == ("catalog:N-BK7", "air")


def test_an_uncoated_window_has_bare_fresnel_interfaces():
    element_ = _compiled(WINDOW)
    assert [s.interaction for s in element_.surfaces] == [RefractInteraction(), RefractInteraction()]
    assert [(e.coating, e.valid_aoi_deg) for e in element_.envelopes] == [(None, None), (None, None)]
    assert [(s.medium_positive, s.medium_negative) for s in element_.surfaces] == [("catalog:N-BK7", "air"), ("air", "catalog:N-BK7")]


def test_mirrors_reflect_on_the_side_that_faces_local_plus_z():
    (flat,) = _compiled(FLAT_MIRROR).surfaces
    assert isinstance(flat, PlaneSurface) and flat.origin_mm == (0.0, 0.0, 0.0)
    assert _plain(flat.interaction) == MirrorInteraction(reflective_side="positive", coating=_constant(0.99, 0.0))
    (concave,) = _compiled(CONCAVE_MIRROR).surfaces
    assert isinstance(concave, SphereSurface) and concave.center_mm == (0.0, 0.0, 200.0)
    assert concave.interaction.reflective_side == "negative"  # the +Z side is inside the sphere
    convex_document = catalog_document()
    product_document(CONCAVE_MIRROR, convex_document)["simulation_models"][0]["surfaces"][0]["radius_mm"] = "-200"
    (convex,) = _compiled(CONCAVE_MIRROR, convex_document | {"catalog_version": "2026-01-01.2"}).surfaces
    assert convex.center_mm == (0.0, 0.0, -200.0) and convex.interaction.reflective_side == "positive"


def test_a_plate_splitter_traces_the_reflection_of_its_coated_face_only():
    back, face = _compiled(PLATE_BS).surfaces
    assert _plain(back.interaction) == RefractInteraction(coating=_constant(0.01, 0.99), trace_fresnel_reflection=False)
    assert _plain(face.interaction) == RefractInteraction(coating=_constant(0.5, 0.5), trace_fresnel_reflection=True)
    assert (back.origin_mm, face.origin_mm) == ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0))
    assert (face.medium_positive, face.medium_negative) == ("air", "catalog:N-BK7")


def test_a_splitting_plane_with_air_on_both_sides_is_a_thin_beam_splitter():
    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    model = product_document(PLATE_BS, document)["simulation_models"][0]
    model["surfaces"] = [model["surfaces"][1] | {"surface_id": "S1"}]
    model["media"] = []
    model["approximations"] = ["COATING_AT_SPECIFIED_BOUND", "EDGE_AS_OPAQUE_RIM", "SUBSTRATE_NOT_MODELLED"]
    product_document(PLATE_BS, document)["specifications"]["materials"] = []
    (surface,) = _compiled(PLATE_BS, document).surfaces
    assert _plain(surface.interaction) == BeamSplitterInteraction(coating=_constant(0.5, 0.5))


def test_several_coating_bands_become_an_aoi_independent_table_held_flat_to_the_engine_range():
    _, face = _compiled(DICHROIC).surfaces
    coating = face.interaction.coating
    assert isinstance(coating, TabulatedCoating)
    assert coating.wavelength_nm == (100.0, 400.0, 600.0, 650.0, 1100.0, 20000.0)
    assert coating.aoi_deg == (0.0, 90.0)
    assert coating.reflectance_s == coating.reflectance_p == ((0.95, 0.95),) * 3 + ((0.1, 0.1),) * 3
    assert coating.transmittance_s == coating.transmittance_p == ((0.05, 0.05),) * 3 + ((0.9, 0.9),) * 3


def test_single_wavelength_bands_are_one_node_and_a_single_band_is_constant():
    def coating(bands):
        return PublicCoatingV1.model_validate({
            "coating_id": "TEST.C", "kind": "DICHROIC", "display_name": "test", "basis": "test",
            "valid_aoi_deg": {"min_deg": "0", "max_deg": "10", "status": "SPECIFIED"},
            "bands": [{"min_nm": lo, "max_nm": hi, "reflectance": r, "transmittance": t, "status": "SPECIFIED"} for lo, hi, r, t in bands],
        })

    table = engine_coating(coating([("1064", "1064", "0.99", "0.01"), ("1500", "1600", "0.02", "0.98")]))
    assert table.wavelength_nm == (100.0, 1064.0, 1500.0, 1600.0, 20000.0)
    assert [row[0] for row in table.reflectance_s] == [0.99, 0.99, 0.02, 0.02, 0.02]
    wide = engine_coating(coating([("50", "60", "0.3", "0.7"), ("30000", "40000", "0.6", "0.4")]))
    assert wide.wavelength_nm == (50.0, 60.0, 30000.0, 40000.0)  # nothing to extend beyond the outermost bands
    assert _values(engine_coating(coating([("1064", "1064", "0.99", "0.01")]))) == _constant(0.99, 0.01)


def test_polarization_elements_are_one_jones_plane():
    (plate,) = _compiled(HWP).surfaces
    assert (plate.surface_id, plate.origin_mm, plate.normal) == (POLARIZATION_SURFACE_ID, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert plate.interaction == RetarderInteraction(fast_axis_local=(1.0, 0.0, 0.0), retardance_rad=math.pi, design_wavelength_nm=633.0,
                                                    achromatic=False, transmittance=0.99)
    assert (plate.medium_positive, plate.medium_negative) == ("air", "air")
    (polarizer,) = _compiled(POLARIZER).surfaces
    assert polarizer.interaction == PolarizerInteraction(axis_local=(1.0, 0.0, 0.0), transmittance_parallel=0.8, transmittance_perpendicular=0.0008)
    envelope = _compiled(POLARIZER).envelopes[0]
    assert (envelope.traced, envelope.coating, envelope.valid_aoi_deg.max_deg) == ("POLARIZATION_ELEMENT", None, "10")
    assert _compiled(HWP).materials == ()
    achromatic = catalog_document() | {"catalog_version": "2026-01-01.2"}
    product_document(HWP, achromatic)["simulation_models"][0]["retardance_dispersion"] = "CONSTANT"
    product_document(HWP, achromatic)["simulation_models"][0]["approximations"] = ["CONSTANT_RETARDANCE", "THIN_ELEMENT"]
    assert _compiled(HWP, achromatic).surfaces[0].interaction.achromatic is True


def test_a_tabulated_material_becomes_a_pchip_engine_material_on_its_published_band():
    document = catalog_document()
    material = next(m for m in document["materials"] if m["material_id"] == "FUSED_SILICA")
    material["dispersion"] = {"kind": "TABULATED", "wavelength_nm": ["200", "700", "1200", "4000"],
                              "refractive_index": ["1.55", "1.455", "1.448", "1.39"], "interpolation": "PCHIP", "max_interpolation_error": "0.0001"}
    model = engine_material(load_public_catalog(document | {"catalog_version": "2026-01-01.2"}).material("FUSED_SILICA"))
    assert model == TabulatedIndexMaterial(material_id="catalog:FUSED_SILICA", valid_wavelength_nm=(210.0, 3710.0),
                                           wavelength_nm=(200.0, 700.0, 1200.0, 4000.0), refractive_index=(1.55, 1.455, 1.448, 1.39),
                                           interpolation="pchip")


def test_catalog_materials_never_replace_engine_built_ins():
    assert engine_material_id("AIR") == "air" and engine_material_id("N-BK7") == "catalog:N-BK7"
    document = project([source(), element("L1", "lens", PCX_F100), element("D1", "splitter", DICHROIC, (100.0, 0.0, 0.0))],
                       [plane("p", (200.0, 0.0, 0.0))])
    materials = lower_project(document, catalog_document()).request.project.materials
    assert [m.material_id for m in materials] == ["air", "catalog:FUSED_SILICA", "catalog:N-BK7"]  # (air: the environment's, stated)


def _uncoated_mirror_catalog() -> dict:
    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    del product_document(FLAT_MIRROR, document)["simulation_models"][0]["surfaces"][0]["coating_id"]
    return document


def test_a_reflecting_surface_without_a_coating_is_the_ideal_reflector():
    """A reflecting surface that states no coating compiles to the Engine's ideal mirror: R = 1, reflection phases pi (s)
    and 0 (p). (A splitting surface with the same medium on both sides keeps its refusal: test_catalog3_physics.py.)"""

    catalog = load_public_catalog(_uncoated_mirror_catalog())
    compiled = compile_element("M1", catalog.get(FLAT_MIRROR), catalog)
    (surface,), (envelope,) = compiled.surfaces, compiled.envelopes
    assert surface.interaction == MirrorInteraction(reflective_side="positive")
    assert surface.interaction.coating is None and surface.interaction.phases is None
    assert (surface.interaction.phase_s_rad, surface.interaction.phase_p_rad) == (math.pi, 0.0)
    assert envelope.ideal_reflector and envelope.coating is None and envelope.valid_aoi_deg is None
    coated = _compiled(FLAT_MIRROR)
    assert not coated.envelopes[0].ideal_reflector and coated.surfaces[0].interaction.coating is not None


def test_an_ideal_reflector_folds_the_whole_power_and_the_run_says_which_surface_it_is():
    """The same fold with the published coating and without one: the coated mirror delivers the published reflectance of
    what the ideal one delivers, the ideal run lists the surface in its model description, and nothing is checked at a
    surface that states no angle range."""

    from os_optics.local import run_optical_project  # noqa: PLC0415

    from local_helpers import ALONG_Y, FOLD_X_TO_Y, TEST_CATALOG_VERSION  # noqa: PLC0415

    def fold(catalog: dict, version: str | None):
        document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=633.0),
                            element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=version)],
                           [plane("screen", (0.0, 60.0, 0.0), ALONG_Y, radius_mm=40.0)])
        return run_optical_project(document, catalog=catalog)

    coated, ideal = fold(catalog_document(), TEST_CATALOG_VERSION), fold(_uncoated_mirror_catalog(), None)
    (coating_id,) = {s["coating_id"] for s in product_document(FLAT_MIRROR)["simulation_models"][0]["surfaces"]}
    band = next(c for c in catalog_document()["coatings"] if c["coating_id"] == coating_id)["bands"][0]
    reflectance = float(band["reflectance"])
    assert 0.9 < reflectance < 1.0
    p_coated = coated.result.observations[0].total_power_W
    p_ideal = ideal.result.observations[0].total_power_W
    assert p_coated == pytest.approx(reflectance * p_ideal, rel=1e-12)
    assert p_ideal == pytest.approx(1e-3, rel=1e-9)  # the source's 1 mW: no coating loss is booked at the fold
    assert ideal.model_scope_issues == () and ideal.status.value == "PASS"
    assert [m.get("ideal_reflectors") for m in ideal.element_models] == [["S1"]]
    assert ["ideal_reflectors" in m for m in coated.element_models] == [False]
    assert ideal.summary()["element_models"][0]["ideal_reflectors"] == ["S1"]


def test_model_selection_is_explicit():
    catalog = load_test_catalog()
    product = catalog.get(PCX_F100)
    assert select_model(product).model_kind == "SURFACE_SEQUENCE"
    assert select_model(product, "SURFACE_SEQUENCE") is product.simulation_models[0]
    with pytest.raises(LocalRunError) as exc:
        compile_element("L1", product, catalog, model_kind="THIN_LENS")
    assert [c.value for c in exc.value.codes] == ["SIMULATION_MODEL_UNSUPPORTED"]


def test_numbers_are_the_nearest_binary_of_the_published_decimals_whatever_the_callers_context():
    with localcontext(Context(prec=3)):
        narrow = _compiled(PCX_F100)
    wide = _compiled(PCX_F100)
    assert narrow.surfaces == wide.surfaces
    assert narrow.surfaces[0].mount_margin_mm == float(Decimal("1.27"))


def test_paraxial_matrix_is_the_independent_prescription_matrix():
    element_ = _compiled(PCX_F100)
    for wavelength in (450.0, 780.0, 1064.0):
        matrix = paraxial_matrix(element_, wavelength)
        expected = lens_matrix(PCX_F100, wavelength)
        assert matrix[0][0] == pytest.approx(expected[0, 0], rel=1e-12) and matrix[1][0] == pytest.approx(expected[1, 0], rel=1e-12)
        assert matrix[0][1] == pytest.approx(expected[0, 1], rel=1e-12) and matrix[1][1] == pytest.approx(expected[1, 1], rel=1e-12)
    assert paraxial_matrix(_compiled(HWP), 633.0) == ((1.0, 0.0), (0.0, 1.0))
    assert paraxial_matrix(_compiled(FLAT_MIRROR), 633.0) == ((1.0, 0.0), (0.0, 1.0))
    with pytest.raises(ValueError):
        paraxial_matrix(element_, 3000.0)  # outside N-BK7's published dispersion data


# ------------------------------------------------------------------------------------- schema-3.0 constructs


def _catalog3():
    from catalog3_products import catalog3_document

    return load_public_catalog(catalog3_document())


def test_polarization_resolved_and_spectrum_coatings_compile_per_polarization_and_per_node():
    from catalog3_products import PBS, SPECTRUM

    catalog = _catalog3()
    split = engine_coating(catalog.coating(PBS))
    assert isinstance(split, ConstantCoating)
    assert (split.reflectance_s, split.reflectance_p, split.transmittance_s, split.transmittance_p) == (0.995, 0.05, 0.0003, 0.95)
    table = engine_coating(catalog.coating(SPECTRUM))
    assert isinstance(table, TabulatedCoating) and table.aoi_deg == (0.0, 90.0)
    assert table.wavelength_nm == (100.0, 400.0, 500.0, 550.0, 600.0, 700.0, 20000.0)  # the spectrum's nodes, held flat beyond
    assert [row[0] for row in table.transmittance_s] == [0.001, 0.001, 0.4, 0.855, 0.3, 0.001, 0.001]
    assert table.transmittance_s == table.transmittance_p and table.reflectance_s == table.reflectance_p
    assert [row[0] for row in table.reflectance_s] == [0.999, 0.999, 0.6, 0.145, 0.7, 0.999, 0.999]


def test_profiles_apertures_and_outlines_compile_to_sag_surfaces_and_mount_outlines():
    from os_optics.contracts import SagSurface
    from os_optics.contracts.optics import RectangularAperture

    from catalog3_products import CYLINDER, CYLINDER_RADIUS_MM, OAP, PARENT_FOCAL_MM, RECT_WINDOW

    catalog = _catalog3()
    lens = compile_element("L", catalog.get(CYLINDER), catalog)
    front, back = lens.surfaces
    # straight along local Y, curved along local X: the Engine's u axis (the curved one) is local X
    assert isinstance(front, SagSurface) and front.cylindrical and front.u_axis == (1.0, 0.0, 0.0)
    assert front.curvature_per_mm == 1.0 / CYLINDER_RADIUS_MM and front.medium_negative == "air" and front.medium_positive == "catalog:N-BK7"
    # an 18 x 11 clear aperture in a 20 x 12 outline: unequal rings, so the outline itself bounds the mount
    assert front.aperture == RectangularAperture(half_width_mm=9.0, half_height_mm=5.5)
    assert front.mount_margin_mm == 0.0 and front.mount_outline == RectangularAperture(half_width_mm=10.0, half_height_mm=6.0)
    assert isinstance(back, PlaneSurface) and back.aperture_u_axis == (1.0, 0.0, 0.0)
    window = compile_element("W", catalog.get(RECT_WINDOW), catalog)
    assert window.surfaces[0].mount_margin_mm == 2.0 and window.surfaces[0].mount_outline is None  # equal rings of 2 mm
    mirror = compile_element("M", catalog.get(OAP), catalog)
    (face,) = mirror.surfaces
    assert isinstance(face, SagSurface) and face.conic == -1.0 and face.curvature_per_mm == 1.0 / (2.0 * PARENT_FOCAL_MM)
    assert face.aperture.center_offset_mm == (2.0 * PARENT_FOCAL_MM, 0.0) and face.interaction.reflective_side == "positive"


def test_a_cylinder_straight_along_x_has_its_engine_u_axis_along_y():
    from os_optics.contracts import SagSurface
    from os_optics.contracts.optics import RectangularAperture

    from catalog3_products import CYLINDER, catalog3_document, cylindrical_lens

    product = cylindrical_lens()
    surface = product["simulation_models"][0]["surfaces"][0]
    surface["cylinder_axis"] = "X"
    surface["clear_half_widths_mm"] = ["9", "5.5"]
    paraxial = product["simulation_models"][0].get("paraxial")
    assert paraxial is None
    catalog = load_public_catalog(catalog3_document(product))
    front = compile_element("L", catalog.get(CYLINDER), catalog).surfaces[0]
    assert isinstance(front, SagSurface) and front.u_axis == (0.0, 1.0, 0.0)
    assert front.aperture == RectangularAperture(half_width_mm=5.5, half_height_mm=9.0)  # (x, y) -> (y, -x)


def test_paraxial_matrix_per_section_axis():
    from catalog3_products import CUBE, CYLINDER, CYLINDER_RADIUS_MM, catalog3_document

    catalog = _catalog3()
    lens = compile_element("L", catalog.get(CYLINDER), catalog, environment=VACUUM_ENVIRONMENT)
    material = next(m for m in catalog3_document()["materials"] if m["material_id"] == "N-BK7")
    b = [float(v) for v in material["dispersion"]["b"]]
    c = [float(v) for v in material["dispersion"]["c_um2"]]
    lam2 = 0.5876 ** 2
    n = math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))
    (a, bb), (cc, d) = paraxial_matrix(lens, 587.6, axis="X")
    assert cc == pytest.approx(-(n - 1.0) / CYLINDER_RADIUS_MM, rel=1e-12) and bb == pytest.approx(4.0 / n, rel=1e-12)
    assert paraxial_matrix(lens, 587.6, axis="Y") == ((1.0, pytest.approx(4.0 / n, rel=1e-12)), (0.0, 1.0))
    with pytest.raises(ValueError, match="cylindrical"):
        paraxial_matrix(lens, 587.6)
    with pytest.raises(ValueError, match="SURFACE_SET"):
        paraxial_matrix(compile_element("C", catalog.get(CUBE), catalog), 587.6)
