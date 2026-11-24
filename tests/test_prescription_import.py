"""Prescription import: csv-surface-table/1 -> IR -> Project-native optical models.

Independent references: the paraxial focal length / back focal distance of each imported unit, compiled member by member
through the catalog compile path, is checked against a y-nu trace written here from the fixture's own numbers with an
own Sellmeier evaluation; the imported members are compared with natively authored models of the same numbers; the inch
fixture states the same system as the mm fixture and must import to identical models; every error code is exercised.
The fixtures (``tests/fixtures/prescriptions/``) are systems constructed for these tests (their comments say how).
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from catalog_store_helpers import REAL_CATALOG
from os_optics.local import compile as C
from os_optics.local import json_api, load_public_catalog
from os_optics._vendor.shared_contracts.design_catalog import PublicCoatingV1, PublicMaterialV1, SurfaceSequenceModelV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.prescription import (
    PRESCRIPTION_FORMATS,
    PrescriptionErrorCode as E,
    PrescriptionImportError,
    import_prescription,
    merge_into_project,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prescriptions"
SEED = load_public_catalog(REAL_CATALOG.read_bytes())
FIXTURE_CATALOG = load_public_catalog()  # the suite's default store: N-BK7 and fused silica only


def text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def imported(name_or_text: str, **kwargs):
    source = text(name_or_text) if name_or_text.endswith(".csv") else name_or_text
    kwargs.setdefault("catalog", SEED)
    return import_prescription(source, **kwargs)


def models_by_id(result) -> dict:
    return {m.model_id: m for m in result.optical_models}


# ------------------------------------------------------------------------------------- independent reference


def sellmeier(b, c, wavelength_nm: float) -> float:
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


# Sellmeier coefficients written here (the fixture's XSK16; SCHOTT N-BK7, N-SF5, F2 as published in the seed catalog).
GLASS = {
    "XSK16": ((1.34317774, 0.241144399, 0.994317969), (0.00704687339, 0.0229005, 92.7508526)),
    "F2": ((1.34533359, 0.209073176, 0.937357162), (0.00997743871, 0.0470450767, 111.886764)),
    "N-BK7": ((1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)),
    "N-SF5": ((1.52481889, 0.187085527, 1.42729015), (0.011254756, 0.0588995392, 129.141675)),
}
# (radius or None, thickness to the next surface, medium after) per real surface, from the fixture rows.
COOKE = [(22.01359, 3.25896, "XSK16"), (-435.7604, 6.00756, None), (-22.21328, 0.99997, "F2"), (20.29192, 2.0, None),
         (None, 2.75, None), (79.6836, 2.95208, "XSK16"), (-18.39533, None, None)]
DOUBLET_SINGLET = [(None, 5.08, None), (60.96, 6.35, "N-BK7"), (-43.18, 2.54, "N-SF5"), (-127.0, 12.7, None), (76.2, 3.81, "N-BK7"),
                   (None, None, None)]


def edlen_air_index(wavelength_nm: float) -> float:
    """Air at 20 C and 101325 Pa (Edlen 1953 in wavenumber form, scaled from 15 C as OpticStudio does): the medium of a
    native Project without a stated environment (native lowering uses physical air)."""

    sigma2 = (1000.0 / wavelength_nm) ** 2
    n15 = 1e-8 * (6432.8 + 2949810.0 / (146.0 - sigma2) + 25540.0 / (41.0 - sigma2))
    return 1.0 + n15 / (1.0 + 0.0034785 * 5.0)


def ynu_trace(rows, wavelength_nm: float, *, physical_air: bool = True) -> tuple[float, float]:
    """EFL and BFD (from the last vertex) for an object at infinity: a marginal ray y = 1, n u = 0, in air of index
    ``edlen_air_index`` (``physical_air``; else unit-index air, the Engine-level snapshot's built-in medium). With image
    space of index n', EFL = n' / power and BFD = n' y / power."""

    air = edlen_air_index(wavelength_nm) if physical_air else 1.0

    def n(medium):
        return air if medium is None else sellmeier(*GLASS[medium], wavelength_nm)

    y, nu, before = 1.0, 0.0, None
    for index, (radius, _t, after) in enumerate(rows):
        if index:
            previous_t, previous_medium = rows[index - 1][1], rows[index - 1][2]
            y += previous_t * nu / n(previous_medium)
        if radius is not None:
            nu -= y * (n(after) - n(before)) / radius
        before = after
    return -air / nu, -air * y / nu


def compiled_members(result):
    """(member_id, z, compiled element or None for a stop, model) of the imported unit, in order, compiled by the runtime's own
    declared-model path (``compile.compile_declared`` with the Project resolver)."""

    by_id = models_by_id(result)
    resolver = C.ProjectResolver(SimpleNamespace(model_materials=result.model_materials, model_coatings=result.model_coatings,
                                                 optical_models=tuple(by_id.values())))
    members = []
    for member in by_id[result.model_id].model.members:
        declared = by_id[member.model_id]
        model = declared.model
        z = member.pose.position_mm[2]
        if model.model_kind == "SURFACE_SEQUENCE":
            element = C.compile_declared(member.member_id, declared, resolver, member_id=member.member_id)
            members.append((member.member_id, z, element, model))
        else:
            members.append((member.member_id, z, None, model))
    return members


def system_matrix(result, wavelength_nm: float):
    """The unit's paraxial matrix from its first to its last vertex: compiled element matrices and the air between them."""

    matrix, plane = C.IDENTITY, None
    for _member_id, z, element, model in compiled_members(result):
        if element is None:
            continue
        if plane is not None:
            matrix = C._multiply(((1.0, z - plane), (0.0, 1.0)), matrix)
        matrix = C._multiply(C.paraxial_matrix(element, wavelength_nm), matrix)
        plane = z + float(Decimal(model.surfaces[-1].vertex_z_mm))
    return matrix


# ------------------------------------------------------------------------------------------- the fixtures


def test_cooke_triplet_imports_as_three_elements_and_a_stop_with_exact_values():
    result = imported("cooke_triplet.csv", model_id="cooke")
    by_id = models_by_id(result)
    unit = by_id["cooke"].model
    assert unit.model_kind == "COMPOUND"
    assert [(m.member_id, m.model_id, m.pose.position_mm) for m in unit.members] == [
        ("L1", "cooke-L1", (0.0, 0.0, 0.0)), ("L2", "cooke-L2", (0.0, 0.0, 9.26652)), ("STOP", "cooke-STOP", (0.0, 0.0, 12.26649)),
        ("L3", "cooke-L3", (0.0, 0.0, 15.01649))]
    l1 = by_id["cooke-L1"].model
    assert [(s.surface_id, s.vertex_z_mm, s.radius_mm, s.clear_radius_mm) for s in l1.surfaces] == [
        ("S1", "0", "22.01359", "9"), ("S2", "3.25896", "-435.7604", "9")]
    assert l1.media == ("XSK16",) and l1.edge_radius_mm == "10" and l1.fidelity.value == "SURFACE_PRESCRIPTION"
    assert by_id["cooke-L2"].model.media == ("F2",) and by_id["cooke-L2"].model.surfaces[1].vertex_z_mm == "0.99997"
    stop = by_id["cooke-STOP"].model
    assert (stop.model_kind, stop.opening_radius_mm, stop.edge_radius_mm, stop.plane_z_mm) == ("APERTURE_STOP", "5", "10", "0")
    assert unit.envelope.model_dump(mode="json") == {"kind": "CYLINDER", "diameter_mm": "20", "z_min_mm": "0", "z_max_mm": "17.96857",
                                                    "status": "DERIVED"}
    assert [(p.port_id, p.point_mm, p.direction) for p in unit.ports] == [("IN", ("0", "0", "0"), ("0", "0", "1")),
                                                                         ("OUT", ("0", "0", "17.96857"), ("0", "0", "1"))]
    materials = {m.material_id: m for m in result.model_materials}
    xsk16 = materials["XSK16"]
    assert xsk16.dispersion.model_dump(mode="json") == {"kind": "SELLMEIER", "b": ["1.34317774", "0.241144399", "0.994317969"],
                                                         "c_um2": ["0.00704687339", "0.0229005", "92.7508526"]}
    assert (xsk16.valid_band_nm.min_nm, xsk16.valid_band_nm.max_nm, xsk16.display_name) == ("310", "2500", "Crown XSK16 (fixture Sellmeier)")
    f2 = materials["F2"]
    published = SEED.material("F2")
    assert f2.dispersion.model_dump() == published.dispersion.model_dump()
    assert f2.valid_band_nm == published.valid_band_nm
    assert f2.basis == f"{published.basis}; copied from public optical catalog {SEED.catalog_version}"
    report = result.report
    assert report["counts"] == {"surfaces": 7, "elements": 3, "cemented_elements": 0, "aperture_stops": 1, "dummy_surfaces": 0, "members": 4,
                                "materials": 2, "materials_from_catalog": 1, "materials_defined_in_file": 1, "coatings": 0}
    assert report["image_plane_z_mm"] == "60.17635" and report["object"] == {"at_infinity": True, "distance_mm": None}
    assert report["wavelength_band_nm"]["min_nm"] == "320" and report["wavelength_band_nm"]["max_nm"] == "2500"
    assert report["catalog_version"] == SEED.catalog_version and report["warnings"] == []
    assert any("plate radius 10 mm" in a for a in report["assumptions"])
    assert result.component.optical_model_id == "cooke" and result.component.custom_part is not None
    assert result.component.optical_role.value == "lens"


def test_doublet_and_singlet_import_a_cemented_element_a_coating_and_a_front_stop():
    result = imported("doublet_singlet.csv", model_id="obj")
    by_id = models_by_id(result)
    assert [m.member_id for m in by_id["obj"].model.members] == ["STOP", "L1", "L2"]
    doublet = by_id["obj-L1"].model
    assert doublet.media == ("N-BK7", "N-SF5") and [s.vertex_z_mm for s in doublet.surfaces] == ["0", "6.35", "8.89"]
    assert [s.radius_mm for s in doublet.surfaces] == ["60.96", "-43.18", "-127"]
    singlet = by_id["obj-L2"].model
    assert [s.coating_id for s in singlet.surfaces] == ["AR_400-1100R0.01T0.99_AOI0-10"] * 2 and singlet.surfaces[1].radius_mm is None
    assert {a.value for a in singlet.approximations} >= {"CATALOG_MATERIAL_DISPERSION", "COATING_PHASE_NOT_MODELLED", "EDGE_AS_OPAQUE_RIM"}
    assert result.model_coatings == (SEED.coating("AR_400-1100R0.01T0.99_AOI0-10"),)
    assert (doublet.valid_band_nm.min_nm, doublet.valid_band_nm.max_nm, doublet.valid_band_nm.status.value) == ("450", "850", "SPECIFIED")
    stop = by_id["obj-STOP"].model
    assert (stop.opening_radius_mm, stop.edge_radius_mm) == ("6.35", "12.7")
    assert result.report["counts"]["cemented_elements"] == 1 and result.report["image_plane_z_mm"] == "81.28"


@pytest.mark.parametrize("name, rows", [("cooke_triplet.csv", COOKE), ("doublet_singlet.csv", DOUBLET_SINGLET)])
@pytest.mark.parametrize("wavelength_nm", [486.1327, 587.5618, 656.2725])
def test_paraxial_focal_length_matches_an_independent_ynu_trace(name, rows, wavelength_nm):
    result = imported(name)
    matrix = system_matrix(result, wavelength_nm)
    efl, bfd = -1.0 / matrix[1][0], -matrix[0][0] / matrix[1][0]
    reference_efl, reference_bfd = ynu_trace(rows, wavelength_nm)
    assert efl == pytest.approx(reference_efl, rel=1e-12)
    assert bfd == pytest.approx(reference_bfd, rel=1e-12)


def test_independent_reference_values_are_physical():
    """Sanity of the references themselves: a positive objective of the expected size."""

    cooke_efl, cooke_bfd = ynu_trace(COOKE, 587.5618)
    objective_efl, _ = ynu_trace(DOUBLET_SINGLET, 587.5618)
    assert 45.0 < cooke_efl < 55.0 and 35.0 < cooke_bfd < 50.0
    assert 40.0 < objective_efl < 80.0


def test_imported_members_equal_natively_authored_models():
    """The imported doublet is exactly the SURFACE_SEQUENCE an author would write from the same numbers."""

    result = imported("doublet_singlet.csv", model_id="obj")
    native = SurfaceSequenceModelV1.model_validate({
        "fidelity": "SURFACE_PRESCRIPTION",
        "surfaces": [
            {"surface_id": "S1", "vertex_z_mm": "0", "radius_mm": "60.96", "clear_radius_mm": "10.16", "traced": "TRANSMITTED", "status": "SPECIFIED"},
            {"surface_id": "S2", "vertex_z_mm": "6.35", "radius_mm": "-43.18", "clear_radius_mm": "10.16", "traced": "TRANSMITTED", "status": "SPECIFIED"},
            {"surface_id": "S3", "vertex_z_mm": "8.89", "radius_mm": "-127", "clear_radius_mm": "10.16", "traced": "TRANSMITTED", "status": "SPECIFIED"},
        ],
        "media": ["N-BK7", "N-SF5"], "edge_radius_mm": "12.7",
        "valid_band_nm": {"min_nm": "450", "max_nm": "850", "status": "SPECIFIED"},
        "approximations": ["NOMINAL_PRESCRIPTION", "IDEAL_SURFACE_FIGURE", "CATALOG_MATERIAL_DISPERSION", "GHOST_REFLECTIONS_NOT_TRACED",
                           "EDGE_AS_OPAQUE_RIM"],
        "status": "SPECIFIED",
    })
    assert models_by_id(result)["obj-L1"].model == native
    (_, _, element, _), = [m for m in compiled_members(result) if m[0] == "L1"]
    assert [s.geometry for s in element.surfaces] == ["sphere", "sphere", "sphere"]
    assert [m.material_id for m in element.materials] == ["project:N-BK7", "project:N-SF5"]  # declared media never alias catalog ones


def _model_dump(result) -> list:
    """The imported models (their display names cite the source's SHA-256 and rows, which differ with the extra column)."""
    return [m.model_dump(mode="json", exclude={"display_name"}) for m in result.optical_models] + [
        m.model_dump(mode="json") for m in (*result.model_materials, *result.model_coatings)]


def test_a_named_or_blank_extra_column_is_skipped_and_listed():
    plain = imported(singlet())
    vendor = singlet(extra_header=",vendor_pn", extra1=",LA1131", extra2=",LA1131")
    # a column with values the caller did not name is refused, and the message names the option
    (problem,) = problems_of(vendor)
    assert problem["code"] == "COLUMN_UNKNOWN" and problem["location"]["column"] == "vendor_pn" and "ignore_columns" in problem["message"]
    for named in (imported(vendor, ignore_columns=("vendor_pn",)), imported(vendor, ignore_columns=("Vendor PN",)),
                  imported("#! ignore_columns = vendor_pn, price\n" + vendor)):
        assert _model_dump(named) == _model_dump(plain)
        assert named.report["ignored_columns"] == [{"section": "surfaces", "column": "vendor_pn", "reason": "DECLARED"}]
    blank = imported(singlet(extra_header=",remarks", extra1=",", extra2=","))
    assert _model_dump(blank) == _model_dump(plain)
    assert blank.report["ignored_columns"] == [{"section": "surfaces", "column": "remarks", "reason": "BLANK"}]
    assert "ignored_columns" not in plain.report  # a table without extra columns keeps its report form
    # a column the import reads cannot be ignored (its values would silently change the lens)
    (problem,) = problems_of(singlet(extra_header=",conic", extra1=",-1", extra2=","), ignore_columns=("k",))
    assert problem["code"] == "DIRECTIVE_INVALID" and problem["location"]["column"] == "conic"
    answer = json_api.import_prescription(vendor, {"ignore_columns": ["vendor_pn"]})
    assert answer["ok"] is True and answer["report"]["ignored_columns"][0]["reason"] == "DECLARED"
    assert json_api.import_prescription(vendor, {"ignore_columns": "vendor_pn"})["error"]["problems"][0]["code"] == "OPTIONS_INVALID"


def test_inch_prescription_imports_to_exactly_the_mm_models():
    mm = imported("doublet_singlet.csv", model_id="obj")
    inch = imported("doublet_singlet_inch.csv", model_id="obj")
    # the display names cite the source rows (the inch file has one more directive line); everything else is identical
    dump = lambda result: [m.model_dump(mode="json", exclude={"display_name"}) for m in result.optical_models] + [
        m.model_dump(mode="json") for m in (*result.model_materials, *result.model_coatings)]
    assert dump(inch) == dump(mm)
    assert inch.report["input_unit"] == "in" and any("1 in = 25.4 mm" in a for a in inch.report["assumptions"])
    assert inch.report["warnings"] == []


SINGLET = """surface,{radius},thickness,material,semi_diameter{extra_header}
1,{r1},{t},N-BK7,{sd}{extra1}
2,{r2},10,,{sd}{extra2}
"""


def singlet(r1="50", r2="-50", t="5", sd="10", radius="radius", extra_header="", extra1="", extra2="", directives=""):
    return directives + SINGLET.format(r1=r1, r2=r2, t=t, sd=sd, radius=radius, extra_header=extra_header, extra1=extra1, extra2=extra2)


@pytest.mark.parametrize("unit, values, expected", [
    ("m", ("0.05", "0.005", "0.01"), ("50", "5", "10")), ("cm", ("5", "0.5", "1"), ("50", "5", "10")),
    ("in", ("2", "0.2", "0.4"), ("50.8", "5.08", "10.16")), ("mm", ("50", "5", "10"), ("50", "5", "10")),
    ("inches", ("2.5", "0.25", "0.5"), ("63.5", "6.35", "12.7")), ("metre", ("0.125", "0.0125", "0.025"), ("125", "12.5", "25"))])
def test_lengths_convert_exactly(unit, values, expected):
    r, t, sd = values
    doc = f"#! units = {unit}\nsurface,radius,thickness,material,semi_diameter\n1,{r},{t},N-BK7,{sd}\n2,inf,{t},,{sd}\n"
    result = imported(doc, wavelength_band_nm=(Decimal(500), Decimal(600)))
    lens = models_by_id(result)["prescription-L1"].model
    assert (lens.surfaces[0].radius_mm, lens.surfaces[1].vertex_z_mm, lens.surfaces[0].clear_radius_mm) == expected


def test_units_option_applies_when_the_file_states_none():
    result = imported(singlet(r1="2", r2="-2", t="0.2", sd="0.5"), units="in", wavelength_band_nm=(Decimal(500), Decimal(600)))
    lens = models_by_id(result)["prescription-L1"].model
    assert (lens.surfaces[0].radius_mm, lens.surfaces[1].vertex_z_mm, lens.surfaces[0].clear_radius_mm) == ("50.8", "5.08", "12.7")


def test_a_units_option_never_silently_overrides_the_files_units():
    """R3 #9: the option agrees with the file (any spelling) -> read in that unit; it disagrees -> UNITS_CONFLICT at the
    directive row (json_api and CLI alike); no option -> the file's unit."""

    inch = singlet(r1="2", r2="-2", t="0.2", sd="0.5", directives="#! units = in\n")
    for option in (None, "in", "inches"):
        lens = models_by_id(imported(inch, units=option))["prescription-L1"].model
        assert lens.surfaces[0].radius_mm == "50.8"
    answer = json_api.import_prescription(text("cooke_triplet.csv"), {"units": "in"})
    assert answer["ok"] is False and answer["error"]["problems"][0]["code"] == "UNITS_CONFLICT"
    assert "drop the units option" in answer["error"]["problems"][0]["message"]
    assert json_api.import_prescription(text("doublet_singlet_inch.csv"), {"units": "mm"})["error"]["problems"][0]["code"] == "UNITS_CONFLICT"
    assert imported("doublet_singlet_inch.csv").report["input_unit"] == "in"


def test_curvature_and_aspheric_conversions_state_their_rounding():
    exact = imported(singlet(radius="curvature", r1="0.02", r2="-0.02"))
    assert models_by_id(exact)["prescription-L1"].model.surfaces[0].radius_mm == "50" and exact.report["warnings"] == []
    rounded = imported(singlet(radius="curvature", r1="0.03", r2="-0.02"))
    assert models_by_id(rounded)["prescription-L1"].model.surfaces[0].radius_mm == "33.33333333333333333333333333"
    assert [w["code"] for w in rounded.report["warnings"]] == ["VALUE_ROUNDED"]
    assert rounded.report["warnings"][0]["location"] == {"row": 2, "column": "curvature", "surface": "1", "section": "surfaces"}
    # A4 in m^-3 -> mm^-3 divides by 1000^3 exactly; in in^-3 by 25.4^3 = 16387.064 (rounded to 28 digits)
    metres = imported("#! units = m\n" + singlet(r1="0.05", r2="-0.05", t="0.005", sd="0.01", extra_header=",a4,a6",
                                                 extra1=",0.002,3", extra2=",,"))
    assert models_by_id(metres)["prescription-L1"].model.surfaces[0].aspheric_coefficients == ("0.000000000002", "0.000000000000003")
    inches = imported("#! units = in\n" + singlet(r1="2", r2="-2", t="0.2", sd="0.4", extra_header=",a4", extra1=",0.001", extra2=","))
    a4 = models_by_id(inches)["prescription-L1"].model.surfaces[0].aspheric_coefficients[0]
    assert Decimal(a4) == pytest.approx(Decimal("0.001") / Decimal("16387.064"), rel=Decimal("1e-27"))
    assert [w["code"] for w in inches.report["warnings"]] == ["VALUE_ROUNDED"]


def test_conic_aspheric_and_cylinder_profiles_are_carried():
    doc = singlet(extra_header=",conic,a4,a6,cylinder_axis", extra1=",-1,0,1e-7,", extra2=",,,,")
    lens = models_by_id(imported(doc))["prescription-L1"].model
    assert (lens.surfaces[0].conic, lens.surfaces[0].aspheric_coefficients) == ("-1", ("0", "0.0000001"))
    cylinder = models_by_id(imported(singlet(r2="inf", extra_header=",cylinder_axis,type", extra1=",X,CYLINDER", extra2=",,")))
    assert cylinder["prescription-L1"].model.surfaces[0].cylinder_axis == "X"


def test_crossed_cylinder_elements_import():
    """Two cylindrical elements with different axes (an anamorphic pair) are separate elements and import."""

    doc = """surface,radius,thickness,material,semi_diameter,cylinder_axis
1,50,4,N-BK7,8,X
2,inf,10,,8,
3,80,4,N-BK7,8,Y
4,inf,20,,8,
"""
    by_id = models_by_id(imported(doc))
    assert (by_id["prescription-L1"].model.surfaces[0].cylinder_axis, by_id["prescription-L2"].model.surfaces[0].cylinder_axis) == ("X", "Y")


def test_inline_constant_index_material_and_absorption():
    doc = singlet() .replace("N-BK7", "MY-GLASS") + "[materials]\nname,kind,index,min_nm,max_nm,absorption_per_mm,basis\n" \
        "my-glass,CONSTANT_INDEX,1.5,400,700,0.001,Measured at 589 nm\n"
    result = imported(doc, glass_source="inline")
    (material,) = result.model_materials
    assert material.model_dump(mode="json", exclude_none=True) == {
        "material_id": "MY-GLASS", "display_name": "MY-GLASS", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.5"},
        "valid_band_nm": {"min_nm": "400", "max_nm": "700", "status": "SPECIFIED"}, "absorption_coefficient_per_mm": "0.001",
        "basis": "Measured at 589 nm"}
    assert result.report["catalog_version"] is None and result.report["wavelength_band_nm"]["basis"].startswith("the intersection")


def test_the_project_keeps_the_source_format_and_sha256():
    """The unit's display_name ends with the format and the SHA-256 of the imported text (the report's source_sha256),
    whatever the length of the lens name; every member names its rows and the format. Deterministic."""

    source = text("cooke_triplet.csv")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    result = imported(source, model_id="cooke")
    assert result.report["source_sha256"] == digest
    by_id = models_by_id(result)
    assert by_id["cooke"].display_name.endswith(f" (csv-surface-table/1, source SHA-256 {digest})")
    assert all("csv-surface-table/1)" in by_id[f"cooke-{m}"].display_name and "row" in by_id[f"cooke-{m}"].display_name
               for m in ("L1", "L2", "STOP", "L3"))
    assert imported(source, model_id="cooke").optical_models == result.optical_models
    renamed = source.replace("#! name = Cooke triplet (OpenSci test fixture)", "#! name = " + ("Long lens name " * 12).strip())
    named = models_by_id(imported(renamed, model_id="cooke"))["cooke"].display_name
    named_digest = hashlib.sha256(renamed.encode("utf-8")).hexdigest()
    assert len(named) == 200 and named.endswith(f"... (csv-surface-table/1, source SHA-256 {named_digest})")


def test_loose_glass_names_resolve_with_a_warning_and_unused_materials_are_reported():
    doc = singlet().replace("N-BK7", "nbk7") + "[materials]\nname,index,min_nm,max_nm\nSPARE,1.4,400,700\n"
    result = imported(doc)
    assert [m.material_id for m in result.model_materials] == ["N-BK7"]
    assert [w["code"] for w in result.report["warnings"]] == ["GLASS_NAME_MATCHED", "MATERIAL_UNUSED"]


def test_dummy_surfaces_stop_on_a_lens_and_several_stops():
    doc = """surface,radius,thickness,material,semi_diameter,stop
1,50,5,N-BK7,10,yes
2,-50,3,,10,
3,,3,,10,
4,,2,,4,yes
5,,2,,4,yes
6,50,5,N-BK7,10,
7,-50,10,,10,
"""
    result = imported(doc)
    unit = models_by_id(result)["prescription"].model
    assert [m.member_id for m in unit.members] == ["L1", "STOP1", "STOP2", "L2"]
    assert [m.pose.position_mm[2] for m in unit.members] == [0.0, 11.0, 13.0, 15.0]
    assert result.report["counts"]["dummy_surfaces"] == 1
    assert "DUMMY_SURFACE" in [w["code"] for w in result.report["warnings"]]
    assert any("surface 1's clear aperture" in a for a in result.report["assumptions"])


def test_an_element_without_mechanical_data_has_its_outline_at_its_largest_clear_aperture():
    result = imported(singlet().replace("2,-50,10,,10", "2,-50,10,,8"))
    lens = models_by_id(result)["prescription-L1"].model
    assert lens.edge_radius_mm == "10" and "largest clear semi-diameter" in result.report["members"][0]["edge_basis"]
    # both profiles reach the outline: no surface is edged anywhere else
    assert [s.edge_radius_mm for s in lens.surfaces] == [None, None] and "derived_edges" not in result.report["members"][0]


EDGE_HEADER = "surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n"
#: A steep front face (R 6 mm, clear 5 mm) on a lens whose plane back face is clear to 7 mm: the sphere ends 6 mm from the
#: axis, inside the 7 mm outline.
STEEP_FRONT = EDGE_HEADER + "1,6,7,N-BK7,5,\n2,,10,,7,\n"
#: A small lens cemented on a wide plate: its front face (R 20 mm, clear 5 mm) would cut through the cemented plane 2 mm
#: behind it at sqrt(76) = 8.7 mm from the axis, inside the unit's 12 mm outline. Both faces of the small lens are clear to
#: 5 mm: edged there, its glass is 2 - 0.635 = 1.365 mm thick at its edge.
SMALL_ON_PLATE = EDGE_HEADER + "1,20,2,N-BK7,5,\n2,,3,N-SF5,5,\n3,,10,,12,\n"


@pytest.mark.parametrize("table, derived, edges", [(STEEP_FRONT, ("1",), ["5", "7"]), (SMALL_ON_PLATE, ("1", "2"), ["5", "5", "12"])],
                         ids=["steep_front", "small_on_plate"])
def test_a_surface_without_mechanical_data_whose_profile_cannot_reach_the_outline_is_edged_at_its_clear_aperture(table, derived, edges):
    """The outline is the largest clear semi-diameter; a surface whose profile ends before it, or leaves no glass there, is
    edged at its own clear semi-diameter and held flat beyond it. The model is the one the same table gives with that edge
    written in the mech_semi_diameter column, and the report states the derived edge as data."""

    result = imported(table)
    member = result.report["members"][0]
    lens = models_by_id(result)["prescription-L1"].model
    assert lens.edge_radius_mm == edges[-1] and [s.edge_radius_mm for s in lens.surfaces] == [*edges[:-1], None]
    assert member["derived_edges"] == [{"surface": label, "edge_radius_mm": "5", "basis": "CLEAR_SEMI_DIAMETER"} for label in derived]
    assert member["surface_edge_radii_mm"] == edges and f"surface{'s' if len(derived) > 1 else ''} {', '.join(derived)} edged at" in member["edge_basis"]
    assert all(Decimal(g["edge_thickness_mm"]) > 0 for g in member["edge_thickness"])
    if table is SMALL_ON_PLATE:  # the edge thickness of the small lens, by hand: 2 - (20 - sqrt(20^2 - 5^2))
        assert Decimal(member["edge_thickness"][0]["edge_thickness_mm"]) == pytest.approx(Decimal(repr(2.0 - (20.0 - math.sqrt(375.0)))), abs=Decimal("1e-4"))
    written = imported(table.replace(",5,\n", ",5,5\n"))
    assert "derived_edges" not in written.report["members"][0]
    assert [m.model for m in written.optical_models[1:]] == [m.model for m in result.optical_models[1:]]
    assert written.optical_models[0].model == result.optical_models[0].model  # the unit: members, envelope, ports


def test_a_derived_edge_changes_no_reading():
    """The trace of the import with the derived edge is the trace of the table with the edge written: the same arrivals at
    the plane, bit for bit."""

    from os_optics.local import evaluate_project

    def readings(table: str):
        project = merge_into_project(base_project(), imported(table, model_id="obj"), instance_id="OBJ",
                                     pose={"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
        evaluation = evaluate_project(project, outputs=("readings", "solid_check"), catalog=SEED)
        assert evaluation.optical_solids.valid
        return [a.model_dump(mode="json") for o in evaluation.result.observations for a in o.arrivals]

    derived, written = readings(STEEP_FRONT), readings(STEEP_FRONT.replace(",N-BK7,5,\n", ",N-BK7,5,5\n"))
    assert derived and derived == written


def test_a_stated_mechanical_semi_diameter_is_never_replaced():
    """A surface that states a mechanical semi-diameter its profile does not reach is refused, as before."""

    with pytest.raises(PrescriptionImportError) as refused:
        imported(STEEP_FRONT.replace(",N-BK7,5,\n", ",N-BK7,5,7\n"))
    (problem,) = refused.value.problems
    assert problem.code is E.EDGE_INVALID and "does not reach" in problem.message and problem.location.row == 2


@pytest.mark.parametrize("table, needle", [
    # 4 mm thick: the R 6 mm sphere (sag 6 mm at its end) crosses the back plane inside the back face's 7 mm clear aperture
    (EDGE_HEADER + "1,6,4,N-BK7,5,\n2,,10,,7,\n", "does not reach element L1's outline radius 7 mm"),
    # the R 20 mm sphere crosses the back plane 2 mm behind it at 8.7 mm, inside the back face's 12 mm clear aperture
    (EDGE_HEADER + "1,20,2,N-BK7,5,\n2,,10,,12,\n", "has no thickness at radius"),
], ids=["profile_ends_in_the_back_clear_aperture", "no_glass_in_the_back_clear_aperture"])
def test_an_edge_at_the_clear_aperture_is_derived_only_for_a_valid_solid(table, needle):
    """Where the two profiles of a glass cross inside the clear aperture of either surface, no edge makes the element a
    solid the run accepts: the table is refused as it stands, by the finding of the element as stated."""

    from os_optics.optical_solids import element_issues
    from os_optics.optical_solids.bodies import body_of

    with pytest.raises(PrescriptionImportError) as refused:
        imported(table)
    (problem,) = refused.value.problems
    assert problem.code is E.EDGE_INVALID and needle in problem.message and "largest clear semi-diameter (no mechanical data)" in problem.message
    # the same table with the edge written imports, and the optical-solid check names the crossing in the clear aperture
    lens = models_by_id(imported(table.replace(",N-BK7,5,\n", ",N-BK7,5,5\n")))["prescription-L1"].model
    issues, _ = element_issues(body_of(SimpleNamespace(instance_id="L1", model=lens)))
    assert [(i.code, i.zone) for i in issues if i.severity == "ERROR"] == [("SURFACES_INTERSECT", "CLEAR_APERTURE")]


def _accepted_tables() -> list:
    tables = [pytest.param(path.read_text(encoding="utf-8"), id=path.stem) for path in sorted(FIXTURES.glob("*.csv"))]
    return tables + [pytest.param(STEEP_FRONT, id="steep_front"), pytest.param(SMALL_ON_PLATE, id="small_on_plate")]


@pytest.mark.parametrize("table", _accepted_tables())
def test_a_model_the_importer_accepts_has_no_solid_error_outside_a_clear_aperture(table):
    """The importer and the optical-solid check agree on the body: every element of an accepted table passes the solid
    check's domain and thickness rules beyond the clear apertures (a table that writes its mechanical semi-diameters can
    still state profiles that cross inside a clear aperture: the two sample that differently), and an element with a
    derived edge passes them all. A fixture table imports as before: none of them needs a derived edge."""

    from os_optics.optical_solids import element_issues
    from os_optics.optical_solids.bodies import body_of

    try:
        result = imported(table)
    except PrescriptionImportError:
        result = import_prescription(table, glass_source="inline")
    derived = [m["member_id"] for m in result.report["members"] if "derived_edges" in m]
    assert bool(derived) == (table in (STEEP_FRONT, SMALL_ON_PLATE))
    for declared in result.optical_models:
        if declared.model.model_kind != "SURFACE_SEQUENCE":
            continue
        issues, _ = element_issues(body_of(SimpleNamespace(instance_id=declared.model_id, model=declared.model)))
        assert [(i.code, i.zone) for i in issues if i.severity == "ERROR" and (derived or i.zone != "CLEAR_APERTURE")] == []


def test_a_derived_stop_plate_clears_the_neighbouring_element():
    doc = """surface,radius,thickness,material,semi_diameter,mech_semi_diameter,stop
1,-60,4,N-BK7,7,8,
2,15,1,,7,8,
STO,,6,,3,,yes
4,40,4,N-BK7,8,8,
5,-40,20,,8,8,
"""
    result = imported(doc)
    plate = Decimal(models_by_id(result)["prescription-STOP"].model.edge_radius_mm)
    # surface 2 (R 15, centre after it) reaches the stop plane 1 mm behind its vertex at r = sqrt(2 * 15 * 1 - 1)
    assert Decimal(3) < plate <= Decimal(29).sqrt() and plate > Decimal(29).sqrt() - Decimal("0.2")
    assert any("Aperture stop STO: plate radius" in a for a in result.report["assumptions"])


# ------------------------------------------------------------------------------------- Project round trip


def base_project() -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Test laser."},
        "nominal_pose": {"position_mm": [0.0, 0.0, -50.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 1.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "image", "pose": {"position_mm": [0.0, 0.0, 81.28], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                          "aperture": {"shape": "circle", "radius_mm": 10.0}}]}


def test_round_trip_csv_to_project_to_json_and_back():
    result = imported("doublet_singlet.csv", model_id="obj")
    project = merge_into_project(base_project(), result, instance_id="OBJ", pose={"position_mm": [0.0, 0.0, 0.0],
                                                                                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    text_form = json.dumps(project.payload())
    again = ProjectDocumentV2.model_validate(json.loads(text_form))
    assert again.payload() == project.payload() and again.content_hash() == project.content_hash()
    component = next(c for c in again.components if c.instance_id == "OBJ")
    assert component.optical_model_id == "obj" and component.product is None
    assert {m.model_id for m in again.optical_models} == {"obj", "obj-STOP", "obj-L1", "obj-L2"}
    # spacings, signs and materials survive: re-derive the vertex positions of every surface in the unit frame
    by_id = {m.model_id: m.model for m in again.optical_models}
    vertices = [(Decimal(repr(member.pose.position_mm[2])) + Decimal(s.vertex_z_mm), s.radius_mm)
                for member in by_id["obj"].members if by_id[member.model_id].model_kind == "SURFACE_SEQUENCE"
                for s in by_id[member.model_id].surfaces]
    assert vertices == [(Decimal("5.08"), "60.96"), (Decimal("11.43"), "-43.18"), (Decimal("13.97"), "-127"), (Decimal("26.67"), "76.2"),
                        (Decimal("30.48"), None)]
    assert [m.material_id for m in again.model_materials] == ["N-BK7", "N-SF5"]


def test_the_last_thickness_only_places_the_image():
    """R2 m9: the last surface's thickness is the image distance, never geometry: 0 (image on the last vertex), negative
    (a virtual image, with a warning), inf (afocal: no image plane) and blank are all imported; the elements are the same."""

    def run(last: str):
        return imported(HEADER + f"1,50,5,N-BK7,10\n2,-50,{last},,10\nIMA,,,,\n")

    reference = run("40")
    assert reference.report["image_plane_z_mm"] == "45"
    for last, image_z in (("0", "5"), ("-40", "-35"), ("inf", None), ("", None)):
        result = run(last)
        assert result.optical_models[1:] == reference.optical_models[1:], last  # the unit's name cites the source digest
        assert result.report["image_plane_z_mm"] == image_z, last
    assert [w["code"] for w in run("-40").report["warnings"]] == ["IMAGE_BEFORE_LAST_SURFACE"]
    assert any("image is at infinity" in a for a in run("inf").report["assumptions"])
    afocal = imported(HEADER + "1,50,5,N-BK7,10\n2,-50,inf,,10\n")  # no image row: still only the image distance
    assert afocal.report["image_plane_z_mm"] is None


def test_a_non_ascii_label_is_a_structured_refusal_everywhere(tmp_path, capsys):
    """R2 M4: a label the contract's text pattern cannot hold is refused at its row (VALUE_INVALID naming the character), in
    the Python API, the JSON envelope and the CLI (exit 65) -- never an uncaught validation error."""

    from os_optics.cli import main

    source = HEADER + "Fl\u00e4che1,50,5,N-BK7,10\n2,-50,5,,10\n"
    problems = problems_of(source)
    assert [p["code"] for p in problems] == ["VALUE_INVALID"] and "character 3 is 'ä' (U+00E4)" in problems[0]["message"]
    assert "byte-order mark" in problems_of(HEADER + "1,50,5,N-BK7,10\n\ufeff2,-50,5,,10\n")[0]["message"]
    answer = json_api.import_prescription(source, {})
    assert answer["ok"] is False and answer["error"]["problems"][0]["location"] == {"row": 2, "column": "surface", "section": "surfaces"}
    bad = tmp_path / "bad.csv"
    bad.write_text(source, encoding="utf-8")
    assert main(["project", "import-prescription", str(bad), "--catalog", str(REAL_CATALOG)]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["problems"][0]["code"] == "VALUE_INVALID"


def test_any_contract_refusal_of_the_import_is_a_structured_refusal(monkeypatch):
    """Belt and braces for M4: a contract refusal no check foresaw still comes back as the MODEL_INVALID envelope."""

    from pydantic import ValidationError

    from os_optics.prescription import api

    def refuse(*_args, **_kwargs):
        try:
            SurfaceSequenceModelV1.model_validate({})
        except ValidationError as exc:
            raise exc from None

    monkeypatch.setattr(api, "import_prescription", refuse)
    answer = api.import_prescription_json(singlet(), {})
    assert answer["ok"] is False and answer["error"]["code"] == "PRESCRIPTION_IMPORT_INVALID"
    assert answer["error"]["problems"][0]["code"] == "MODEL_INVALID"


def test_a_second_import_shares_identical_materials_and_refuses_conflicts():
    first = imported("doublet_singlet.csv", model_id="a")
    project = merge_into_project(base_project(), first)
    second = imported("doublet_singlet.csv", model_id="b")
    both = merge_into_project(project, second, instance_id="B", pose={"position_mm": [0.0, 30.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    assert [m.material_id for m in both.model_materials] == ["N-BK7", "N-SF5"] and len(both.optical_models) == 8
    with pytest.raises(PrescriptionImportError) as exc:
        merge_into_project(both, first, instance_id="C")
    assert set(exc.value.codes) == {E.PROJECT_INVALID} and "already declares optical model 'a'" in str(exc.value)
    with pytest.raises(PrescriptionImportError) as exc:
        merge_into_project({"schema": "opensci.project"}, first)
    assert exc.value.codes == (E.PROJECT_INVALID,)


# --------------------------------------------------------------------------------------------- JSON API, CLI


def test_json_api_import_prescription(real_catalog_store):
    caps = json_api.capabilities()
    assert caps["prescription_formats"] == list(PRESCRIPTION_FORMATS) == ["csv-surface-table/1"]
    answer = json_api.import_prescription(text("cooke_triplet.csv"), {"model_id": "cooke"})
    json.dumps(answer, allow_nan=False)
    assert answer["ok"] is True and answer["model_id"] == "cooke" and answer["format"] == "csv-surface-table/1"
    assert [m["model_id"] for m in answer["optical_models"]] == ["cooke", "cooke-L1", "cooke-L2", "cooke-STOP", "cooke-L3"]
    assert answer["model_coatings"] is None and answer["component"]["optical_model_id"] == "cooke"
    merged = json_api.import_prescription(text("doublet_singlet.csv").encode("utf-8"), {
        "model_id": "obj", "into": base_project(), "instance_id": "OBJ",
        "pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}, "wavelength_band_nm": {"min_nm": "500", "max_nm": 700}})
    assert merged["ok"] is True and merged["project_hash"] == ProjectDocumentV2.model_validate(merged["project"]).content_hash()
    assert merged["report"]["wavelength_band_nm"]["basis"] == "as requested"
    assert [w["code"] for w in merged["report"]["warnings"]] == ["BAND_OVERRIDDEN"]
    refused = json_api.import_prescription("surface,radius,thickness,material,semi_diameter\n1,abc,5,N-BK7,10\n2,-50,5,,10\n")
    assert refused == {"ok": False, "error": {"code": "PRESCRIPTION_IMPORT_INVALID", "problems": [{
        "code": "VALUE_INVALID", "message": "radius 'abc' is not a number; give a radius, or blank / inf for a plane",
        "location": {"row": 2, "column": "radius", "surface": "1", "section": "surfaces"}}]}}
    for options, code in (({"colour": 1}, "OPTIONS_INVALID"), ({"pose": {}}, "OPTIONS_INVALID"), ({"wavelength_band_nm": [1, 2]}, "OPTIONS_INVALID"),
                          ({"model_id": "9x"}, "OPTIONS_INVALID"), ({"format": "zemax-zmx"}, "FORMAT_UNSUPPORTED"), ("x", "OPTIONS_INVALID")):
        answer = json_api.import_prescription(text("cooke_triplet.csv"), options)
        assert answer["ok"] is False and answer["error"]["problems"][0]["code"] == code, options
    assert json_api.import_prescription(42)["error"]["problems"][0]["code"] == "TEXT_INVALID"


def test_json_api_import_without_a_catalog_names_the_way_out(empty_catalog_store):
    answer = json_api.import_prescription(text("doublet_singlet.csv"))
    assert answer["ok"] is False
    problems = answer["error"]["problems"]
    assert {p["code"] for p in problems} == {"GLASS_UNKNOWN", "COATING_UNKNOWN"}
    assert "catalog sync" in problems[0]["message"] and "[materials]" in problems[0]["message"]
    inline = json_api.import_prescription(text("cooke_triplet.csv").replace(",F2,", ",XSK16,"))
    assert inline["ok"] is True and inline["report"]["catalog_version"] is None


def test_cli_import_prescription(tmp_path, capsys):
    from os_optics.cli import main

    fixture = str(FIXTURES / "doublet_singlet.csv")
    assert main(["project", "import-prescription", fixture, "--catalog", str(REAL_CATALOG), "--model-id", "obj"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["model_id"] == "obj" and "project" not in printed
    project_file = tmp_path / "project.json"
    project_file.write_text(json.dumps(base_project()), encoding="utf-8")
    out = tmp_path / "out.json"
    assert main(["project", "import-prescription", fixture, "--catalog", str(REAL_CATALOG), "--into", str(project_file), "--instance-id", "OBJ",
                 "--pose", "0,0,5", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["project_written"] == str(out)
    written = ProjectDocumentV2.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert next(c for c in written.components if c.instance_id == "OBJ").nominal_pose.position_mm == (0.0, 0.0, 5.0)
    bad = tmp_path / "bad.csv"
    bad.write_text("surface,radius,thickness,material,semi_diameter\n1,50,-5,N-BK7,10\n2,-50,5,,10\n", encoding="utf-8")
    assert main(["project", "import-prescription", str(bad), "--catalog", str(REAL_CATALOG)]) == 65
    refused = json.loads(capsys.readouterr().out)
    assert refused["error"]["problems"][0]["code"] == "THICKNESS_INVALID"
    extra = tmp_path / "extra.csv"
    extra.write_text(singlet(extra_header=",vendor_pn", extra1=",LA1131", extra2=",LA1131"), encoding="utf-8")
    assert main(["project", "import-prescription", str(extra), "--catalog", str(REAL_CATALOG)]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["problems"][0]["code"] == "COLUMN_UNKNOWN"
    assert main(["project", "import-prescription", str(extra), "--catalog", str(REAL_CATALOG), "--ignore-column", "vendor_pn"]) == 0
    assert json.loads(capsys.readouterr().out)["report"]["ignored_columns"] == [
        {"section": "surfaces", "column": "vendor_pn", "reason": "DECLARED"}]
    assert main(["project", "import-prescription", str(tmp_path / "missing.csv")]) == 64
    assert main(["project", "import-prescription", fixture, "--pose", "1,2,3"]) == 64
    capsys.readouterr()


# ------------------------------------------------------------------------------------------- error codes


def problems_of(source, **kwargs) -> list[dict]:
    with pytest.raises(PrescriptionImportError) as exc:
        imported(source, **kwargs)
    return [p.as_dict() for p in exc.value.problems]


HEADER = "surface,radius,thickness,material,semi_diameter\n"
ERROR_CASES = [
    # (id, text, options, code, expected location subset)
    ("options", singlet(), {"model_id": "-bad"}, E.OPTIONS_INVALID, {}),
    ("glass-source", singlet(), {"glass_source": "web"}, E.OPTIONS_INVALID, {}),
    ("band-option", singlet(), {"wavelength_band_nm": (Decimal(700), Decimal(600))}, E.OPTIONS_INVALID, {}),
    ("format-option", singlet(), {"format": "codev-seq"}, E.FORMAT_UNSUPPORTED, {}),
    ("format-directive", "#! format = zemax-zmx\n" + singlet(), {}, E.FORMAT_UNSUPPORTED, {"row": 1}),
    ("empty", "  \n", {}, E.TEXT_INVALID, {}),
    ("quote", HEADER + '1,"50,5,N-BK7,10\n', {}, E.TEXT_INVALID, {"row": 2}),
    ("directive", "#! colour = blue\n" + singlet(), {}, E.DIRECTIVE_INVALID, {"row": 1}),
    ("directive-band", "#! wavelength_min_nm = 400\n" + singlet(), {}, E.DIRECTIVE_INVALID, {"row": 1}),
    ("units", "#! units = furlong\n" + singlet(), {}, E.UNITS_INVALID, {"row": 1}),
    ("units-option", singlet(), {"units": "parsec"}, E.UNITS_INVALID, {}),
    ("section", singlet() + "[glasses]\nname\n", {}, E.SECTION_INVALID, {"row": 4}),
    ("column-unknown", singlet(extra_header=",power", extra1=",1", extra2=","), {}, E.COLUMN_UNKNOWN, {"row": 1, "column": "power"}),
    ("column-duplicate", "surface,radius,radius,thickness,material,semi_diameter\n1,50,50,5,N-BK7,10\n2,-50,-50,5,,10\n", {}, E.COLUMN_DUPLICATE,
     {"row": 1}),
    ("radius-and-curvature", "surface,radius,curvature,thickness,material,semi_diameter\n1,50,,5,N-BK7,10\n2,-50,,5,,10\n", {},
     E.COLUMN_DUPLICATE, {"row": 1}),
    ("column-missing", "surface,radius,thickness,material\n1,50,5,N-BK7\n2,-50,5,\n", {}, E.COLUMN_MISSING, {"row": 1}),
    ("field-missing", singlet(sd=""), {}, E.FIELD_MISSING, {"row": 2, "column": "semi_diameter", "surface": "1"}),
    ("thickness-missing", HEADER + "1,50,,N-BK7,10\n2,-50,,,10\n", {}, E.FIELD_MISSING, {"row": 2, "column": "thickness"}),
    ("value", singlet(r1="fifty"), {}, E.VALUE_INVALID, {"row": 2, "column": "radius", "surface": "1"}),
    ("radius-zero", singlet(r1="0"), {}, E.VALUE_INVALID, {"row": 2, "column": "radius"}),
    ("stop-word", singlet(extra_header=",stop", extra1=",maybe", extra2=","), {}, E.VALUE_INVALID, {"column": "stop"}),
    ("row", singlet(extra1=",surplus"), {}, E.ROW_INVALID, {"row": 2}),
    ("object-not-first", HEADER + "1,50,5,N-BK7,10\nOBJ,inf,inf,,\n2,-50,5,,10\n", {}, E.SURFACE_ORDER_INVALID, {"row": 3, "surface": "OBJ"}),
    ("after-image", HEADER + "1,50,5,N-BK7,10\nIMA,,,,\n2,-50,5,,10\n", {}, E.SURFACE_ORDER_INVALID, {"row": 4}),
    ("duplicate-label", HEADER + "1,50,5,N-BK7,10\n1,-50,5,,10\n", {}, E.SURFACE_ORDER_INVALID, {"row": 3}),
    ("no-surface", HEADER + "OBJ,inf,inf,,\nIMA,,,,\n", {}, E.SURFACE_ORDER_INVALID, {}),
    ("only-air", HEADER + "1,,5,,10\n2,,5,,10\n", {}, E.SURFACE_ORDER_INVALID, {}),
    ("thickness-negative", singlet(t="-5"), {}, E.THICKNESS_INVALID, {"row": 2, "column": "thickness"}),
    ("thickness-zero", singlet(t="0"), {}, E.THICKNESS_INVALID, {"row": 2, "column": "thickness"}),
    ("thickness-infinite", singlet(t="inf"), {}, E.THICKNESS_INVALID, {"row": 2, "column": "thickness"}),
    ("glass-unknown", singlet().replace("N-BK7", "UNOBTAINIUM"), {}, E.GLASS_UNKNOWN, {"row": 2, "column": "material", "surface": "1"}),
    ("glass-inline-only", singlet(), {"glass_source": "inline"}, E.GLASS_UNKNOWN, {"row": 2, "column": "material"}),
    ("material-kind", singlet().replace("N-BK7", "G1") + "[materials]\nname,min_nm,max_nm\nG1,400,700\n", {}, E.MATERIAL_INVALID,
     {"row": 6, "section": "materials"}),
    ("material-duplicate", singlet().replace("N-BK7", "G1") + "[materials]\nname,index,min_nm,max_nm\nG1,1.5,400,700\ng1,1.6,400,700\n", {},
     E.MATERIAL_INVALID, {"row": 7}),
    ("material-name", singlet() + "[materials]\nname,index,min_nm,max_nm\nmy glass!,1.5,400,700\n", {}, E.MATERIAL_INVALID, {"row": 6}),
    ("material-band-order", singlet().replace("N-BK7", "G1") + "[materials]\nname,index,min_nm,max_nm\nG1,1.5,700,400\n", {},
     E.MATERIAL_INVALID, {"row": 6, "column": "max_nm"}),
    # (a band that only reaches beyond the glass's is narrowed, tests/test_out_of_range_data.py; one wholly outside it is refused)
    ("outside-band", singlet(), {"wavelength_band_nm": (Decimal(100), Decimal(200))}, E.MATERIAL_OUTSIDE_BAND,
     {"row": 2, "column": "material"}),
    ("no-common-band", singlet().replace("N-BK7", "G1") .replace("2,-50,10,,10", "2,-50,5,G2,10\n3,inf,10,,10")
     + "[materials]\nname,index,min_nm,max_nm\nG1,1.5,400,500\nG2,1.6,600,700\n", {}, E.MATERIAL_OUTSIDE_BAND, {"row": 2}),
    ("coating", singlet(extra_header=",coating", extra1=",AR_FOO", extra2=","), {}, E.COATING_UNKNOWN, {"row": 2, "column": "coating"}),
    ("coating-definition", singlet(extra_header=",coating", extra1=",OWN", extra2=",") + "[coatings]\nname,kind,wavelength_nm,reflectance,"
     "transmittance,min_aoi_deg,max_aoi_deg\nOWN,ANTI_REFLECTION,400,0.2,0.9,0,10\nOWN,,1100,0.01,0.99,,\n", {}, E.COATING_INVALID,
     {"row": 6, "column": "name", "section": "coatings"}),
    ("surface-profile", singlet(r1="8"), {}, E.SURFACE_INVALID, {"row": 2, "surface": "1"}),
    ("surfaces-intersect", singlet(r1="10", r2="-10", t="0.5", sd="5"), {}, E.SURFACES_INTERSECT, {"row": 2}),
    ("elements-intersect", HEADER + "1,50,3,N-BK7,3\n2,10,0.2,,3\n3,-10,3,N-BK7,3\n4,-50,10,,3\n", {}, E.SURFACES_INTERSECT, {"row": 4}),
    ("edge-mechanical", singlet(extra_header=",mech_semi_diameter", extra1=",9", extra2=","), {}, E.EDGE_INVALID,
     {"row": 2, "column": "mech_semi_diameter"}),
    ("edge-thickness", singlet(r1="20", r2="-20", t="1.5", sd="5", extra_header=",mech_semi_diameter", extra1=",8", extra2=",8"), {},
     E.EDGE_INVALID, {"row": 3}),
    ("edge-profile", singlet(r1="10", sd="5", extra_header=",mech_semi_diameter", extra1=",12", extra2=","), {}, E.EDGE_INVALID, {"row": 2}),
    ("bodies", "surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n1,50,3,N-BK7,3,10\n2,30,1,,3,10\n3,-30,3,N-BK7,3,10\n"
     "4,-50,10,,3,10\n", {}, E.BODIES_INTERSECT, {"row": 4}),
    ("stop-plate", "surface,radius,thickness,material,semi_diameter,mech_semi_diameter,stop\n1,-60,4,N-BK7,7,8,\n2,15,1,,7,8,\n"
     "STO,,6,,3,8,yes\n4,40,4,N-BK7,8,8,\n5,-40,20,,8,8,\n", {}, E.BODIES_INTERSECT, {"row": 4, "surface": "STO"}),
    ("stop-in-opening", "surface,radius,thickness,material,semi_diameter,stop\n1,-60,4,N-BK7,7,\n2,15,0.2,,7,\nSTO,,6,,3,yes\n"
     "4,40,4,N-BK7,8,\n5,-40,20,,8,\n", {}, E.BODIES_INTERSECT, {"row": 4}),
    ("decenter", singlet(extra_header=",decenter_x", extra1=",0.1", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"row": 2, "column": "decenter_x"}),
    ("tilt", singlet(extra_header=",tilt_y", extra1=",", extra2=",2"), {}, E.UNSUPPORTED_FEATURE, {"row": 3, "column": "tilt_y"}),
    ("a2", singlet(extra_header=",a2", extra1=",0.001", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"column": "a2"}),
    ("coordinate-break", singlet(extra_header=",type", extra1=",COORDBRK", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"column": "type"}),
    ("grin", singlet(extra_header=",type", extra1=",GRINSUR1", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"column": "type"}),
    ("diffractive", singlet(extra_header=",type", extra1=",DIFFRACTION_GRATING", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"column": "type"}),
    ("non-sequential", singlet(extra_header=",type", extra1=",NSC_OBJECT", extra2=","), {}, E.UNSUPPORTED_FEATURE, {"column": "type"}),
    ("unknown-type", singlet(extra_header=",type", extra1=",WOBBLY", extra2=","), {}, E.VALUE_INVALID, {"column": "type"}),
    ("mirror", HEADER + "1,-100,10,MIRROR,10\n", {}, E.UNSUPPORTED_FEATURE, {"row": 2, "column": "material"}),
    ("image-in-glass", HEADER + "1,50,5,N-BK7,10\n", {}, E.UNSUPPORTED_FEATURE, {"row": 2, "column": "material"}),
    ("object-in-glass", HEADER + "OBJ,inf,inf,N-BK7,\n1,50,5,N-BK7,10\n2,-50,5,,10\n", {}, E.UNSUPPORTED_FEATURE, {"row": 2}),
    ("coated-air-plane", HEADER.replace("\n", ",coating\n") + "1,50,5,N-BK7,10,\n2,-50,5,,10,\n3,,5,,10,AR_400-1100R0.01T0.99_AOI0-10\n",
     {}, E.UNSUPPORTED_FEATURE, {"row": 4, "column": "coating"}),
    ("too-many-surfaces", HEADER + "".join(f"{i},{'' if i % 2 else ''},1,{'N-BK7' if i < 17 else ''},5\n" for i in range(1, 19)), {},
     E.UNSUPPORTED_FEATURE, {"row": 2}),
    ("mixed-cylinders", singlet(extra_header=",cylinder_axis", extra1=",X", extra2=",Y"), {}, E.MODEL_INVALID, {"row": 2}),
    # final-review regressions (R2 M4, m9, n7; R3 #9)
    ("units-conflict", singlet(r1="2", r2="-2", t="0.2", sd="0.4", directives="#! units = in\n"), {"units": "mm"}, E.UNITS_CONFLICT,
     {"row": 1}),
    ("label-not-ascii", HEADER + "Fl\u00e4che1,50,5,N-BK7,10\n2,-50,5,,10\n", {}, E.VALUE_INVALID, {"row": 2, "column": "surface"}),
    ("label-superscript", HEADER + "S\u00b91,50,5,N-BK7,10\n2,-50,5,,10\n", {}, E.VALUE_INVALID, {"row": 2, "column": "surface"}),
    ("label-stray-bom", HEADER + "\ufeff1,50,5,N-BK7,10\n2,-50,5,,10\n", {}, E.VALUE_INVALID, {"row": 2, "column": "surface"}),
    ("label-too-long", HEADER + "A" * 201 + ",50,5,N-BK7,10\n2,-50,5,,10\n", {}, E.VALUE_INVALID, {"row": 2, "column": "surface"}),
    ("number-overflow", singlet(r1="1e400"), {}, E.VALUE_INVALID, {"row": 2, "column": "radius", "surface": "1"}),
    ("number-underflow", singlet(t="1e-400"), {}, E.VALUE_INVALID, {"row": 2, "column": "thickness"}),
    ("material-pole-in-band", singlet().replace("N-BK7", "G1") + "[materials]\nname,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm\n"
     "G1,SELLMEIER,1,0,0,0.25,0,0,400,700\n", {}, E.MATERIAL_INVALID, {"row": 6, "column": "name", "section": "materials"}),
    ("material-index-below-1", singlet().replace("N-BK7", "G1") + "[materials]\nname,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm\n"
     "G1,SELLMEIER,-0.5,0,0,0.01,0,0,400,700\n", {}, E.MATERIAL_INVALID, {"row": 6, "section": "materials"}),
    ("material-index-imaginary", singlet().replace("N-BK7", "G1") + "[materials]\nname,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm\n"
     "G1,SELLMEIER,-2,0,0,0.01,0,0,400,700\n", {}, E.MATERIAL_INVALID, {"row": 6, "section": "materials"}),
    ("material-constant-below-1", singlet().replace("N-BK7", "G1") + "[materials]\nname,kind,index,min_nm,max_nm\nG1,CONSTANT_INDEX,0.5,400,700\n",
     {}, E.MATERIAL_INVALID, {"row": 6, "section": "materials"}),
    ("type-plane-with-radius", singlet(extra_header=",type", extra1=",PLANE", extra2=","), {}, E.VALUE_INVALID, {"row": 2, "column": "type"}),
    ("type-standard-with-asphere", singlet(extra_header=",type,a4", extra1=",STANDARD,1e-5", extra2=",,"), {}, E.VALUE_INVALID,
     {"row": 2, "column": "type"}),
    ("type-sphere-with-conic", singlet(extra_header=",type,conic", extra1=",SPHERE,-1", extra2=",,"), {}, E.VALUE_INVALID,
     {"row": 2, "column": "type"}),
    ("type-cylinder-plane", singlet(r2="inf", extra_header=",type,cylinder_axis", extra1=",,", extra2=",CYLINDER,X"), {}, E.VALUE_INVALID,
     {"row": 3, "column": "type"}),
    ("coated-stop", HEADER.replace("\n", ",stop,coating\n") + "STO,,5,,5,yes,AR_400-1100R0.01T0.99_AOI0-10\n1,50,5,N-BK7,10,,\n2,-50,5,,10,,\n",
     {}, E.UNSUPPORTED_FEATURE, {"row": 2, "column": "coating", "surface": "STO"}),
]


@pytest.mark.parametrize("case_id, source, options, code, location", ERROR_CASES, ids=[c[0] for c in ERROR_CASES])
def test_every_refusal_is_structured_and_located(case_id, source, options, code, location):
    problems = problems_of(source, **options)
    first = next((p for p in problems if p["code"] == code.value), None)
    assert first is not None, problems
    assert first["message"]
    assert {k: first.get("location", {}).get(k) for k in location} == location, first


def test_every_error_code_is_exercised():
    codes = {case[3] for case in ERROR_CASES} | {E.GLASS_AMBIGUOUS, E.PROJECT_INVALID}
    assert codes == set(E)


def test_an_ambiguous_glass_name_lists_the_candidates():
    materials = [SEED.material("N-BK7"), SEED.material("N-BK7").model_copy(update={"material_id": "NBK7"})]
    duck = SimpleNamespace(catalog_version="2026-01-01.9", materials=materials,
                           material=lambda i: next((m for m in materials if m.material_id == i), None), coating=lambda i: None)
    problems = problems_of(singlet().replace("N-BK7", "N BK7"), catalog=lambda: duck)
    assert problems[0]["code"] == "GLASS_AMBIGUOUS" and "N-BK7, NBK7" in problems[0]["message"]


def test_unknown_glass_suggests_published_names():
    (problem,) = problems_of(singlet().replace("N-BK7", "N-BK8"))
    assert problem["code"] == "GLASS_UNKNOWN" and "closest published: N-BK7" in problem["message"]
    assert "[materials]" in problem["message"]


def test_every_problem_of_the_table_step_is_reported_together():
    """A file with several table problems (a bad number, a bad thickness, a missing field) reports all of them at once;
    later steps (glasses and coatings, geometry) are checked only once the table is clean."""

    doc = HEADER + "1,abc,-1,N-BK7,\n2,-50,5,,10\n"
    codes = [p["code"] for p in problems_of(doc)]
    assert codes == ["VALUE_INVALID", "THICKNESS_INVALID", "FIELD_MISSING"]
    # an unknown glass in the same file is reported once the table is fixed, not together with the table problems
    assert [p["code"] for p in problems_of(doc.replace("N-BK7", "N-BK8"))] == codes


def test_excel_style_csv_is_read():
    """Quoted cells, CRLF line ends, a byte-order mark, trailing empty cells, comment rows and aliased headers."""

    doc = ("﻿#! units = mm,,,\r\n# comment,,,\r\n\"Surf\",\"Glass\",R,T,SD,,\r\n\"1\",\"N-BK7\",50,5,10,,\r\n2,,\"-50\",10,10,,\r\n,,,,,,\r\n")
    lens = models_by_id(imported(doc))["prescription-L1"].model
    assert [s.radius_mm for s in lens.surfaces] == ["50", "-50"] and lens.media == ("N-BK7",)


def test_fixture_catalog_resolves_n_bk7_only():
    """Glass resolution uses whichever catalog is given: the suite's small fixture catalog publishes N-BK7."""

    assert models_by_id(imported(singlet(), catalog=FIXTURE_CATALOG))["prescription-L1"].model.media == ("N-BK7",)
    assert problems_of(singlet().replace("N-BK7", "N-SF5"), catalog=FIXTURE_CATALOG)[0]["code"] == "GLASS_UNKNOWN"


# --------------------------------------------------------------------- Engine trace of the imported members


def test_engine_traces_the_imported_elements_to_the_paraxial_focus():
    """The compiled member surfaces, placed at their member poses, traced by the Engine: the first-order transfer at the
    independently computed back focal plane has the paraxial power and no A term (a collimated input focuses there)."""

    from os_optics import validate
    from os_optics.contracts.components import EngineeringComponentSnapshot
    from os_optics.contracts.geometry import Pose3D
    from os_optics.contracts.project import Observation, ProjectSnapshot
    from os_optics.contracts.request import ValidationRequest
    from os_optics.local.lowering import lower_project

    result = imported("doublet_singlet.csv")
    wavelength = 632.8
    # an Engine-level snapshot whose surfaces name the built-in unit-index air (DESIGN: the engine layer's ``air``)
    efl, bfd = ynu_trace(DOUBLET_SINGLET, wavelength, physical_air=False)
    lowered = lower_project(base_project() | {"observations": []}, FIXTURE_CATALOG)
    source = next(c for c in lowered.request.project.components if c.source is not None)
    components, materials = [source], {}
    last_vertex = 0.0
    for member_id, z, element, model in compiled_members(result):
        if element is None:
            continue
        components.append(EngineeringComponentSnapshot(instance_id=f"OBJ-{member_id}", role="lens",
                                                       pose=Pose3D(position_mm=(0.0, 0.0, z)), surfaces=element.surfaces))
        materials.update({m.material_id: m for m in element.materials})
        last_vertex = z + float(Decimal(model.surfaces[-1].vertex_z_mm))
    focus = Observation(observation_id="focus", pose=Pose3D(position_mm=(0.0, 0.0, last_vertex + bfd)))
    request = ValidationRequest(project=ProjectSnapshot(components=tuple(components), observations=(focus,),
                                                        materials=tuple(materials[k] for k in sorted(materials))),
                                configuration=lowered.request.configuration)
    outcome = validate(request)
    (observation,) = outcome.observations
    arrival = observation.arrivals[0]
    matrix = arrival.first_order.matrix
    assert matrix[2][0] == pytest.approx(-1.0 / efl, rel=1e-5)
    assert abs(matrix[0][0]) < 1e-4
    # The fixture's imported catalog glass states no measured or explicitly assumed bulk absorption. Its geometry,
    # first-order matrix and optical path are still defined, but its transmitted *power* is only upper-bounded; a
    # positive value would silently assume zero loss and falsely promote that assumption to a measurement.
    assert arrival.beam.power_W is None
    assert arrival.power_upper_bound_W is not None and arrival.power_upper_bound_W > 0.0


# ------------------------------------------------- end to end through the Project runtime (needs declared-model lowering)

DECLARED_MODELS = "optical_models" in (json_api.capabilities().get("project_features") or [])


@pytest.mark.skipif(not DECLARED_MODELS, reason="this runtime does not lower Project-declared optical models yet "
                    "(capabilities()['project_features'] lacks 'optical_models')")
@pytest.mark.parametrize("name", ["cooke_triplet.csv", "doublet_singlet.csv"])
def test_imported_unit_runs_end_to_end(real_catalog_store, name):
    answer = json_api.import_prescription(text(name), {"model_id": "rx", "into": base_project(), "instance_id": "RX"})
    assert answer["ok"] is True, answer
    project = answer["project"]
    image_z = float(answer["report"]["image_plane_z_mm"])
    project["observations"][0]["pose"]["position_mm"] = [0.0, 0.0, image_z]
    checked = json_api.check_project(project)
    assert checked["ok"] is True, checked
    run = json_api.run_project(project)
    assert run["ok"] is True, run


# ------------------------------------------------------------------------------- one catalog glass, every route (R2 M5)


def _nbk7_product():
    return next(p for p in SEED.products if len(p.simulation_models) == 1
                and "N-BK7" in (p.simulation_models[0].material_ids() if hasattr(p.simulation_models[0], "material_ids") else ()))


def test_a_catalog_glass_copied_by_every_route_and_release_is_one_material():
    """The native library (from the suite's fixture catalog release), the import (from the seed release) and a declared copy
    of a catalog product all copy N-BK7 with ``catalog_copy.copy_catalog_material``; the copies differ only in provenance
    (``basis`` names their release), so they are one material in a Project, in either order. Copies whose physics differs
    are refused, naming what differs."""

    from os_optics.catalog_copy import copy_catalog_material, physics_difference
    from os_optics.local import declared_copy
    from os_optics.native_library import NativeComponentError, add_to_project, build_native_model

    native = build_native_model("lens.singlet", {"focal_length_mm": "100", "design_wavelength_nm": "632.8", "material": "N-BK7",
                                                 "diameter_mm": "25.4"})
    assert FIXTURE_CATALOG.catalog_version != SEED.catalog_version
    with_native = add_to_project(base_project(), native, instance_id="NAT")
    merged = merge_into_project(with_native, imported(singlet(), model_id="rx"), instance_id="RX")
    assert [m.material_id for m in merged.model_materials] == ["N-BK7"]
    assert merged.model_materials[0].basis.endswith(FIXTURE_CATALOG.catalog_version)  # the Project's own entry is kept
    imported_first = merge_into_project(base_project(), imported(singlet(), model_id="rx"), instance_id="RX").payload()
    both = add_to_project(imported_first, native, instance_id="NAT")
    assert [m["material_id"] for m in both["model_materials"]] == ["N-BK7"]
    assert both["model_materials"][0]["basis"].endswith(SEED.catalog_version)
    _model, materials, _coatings = declared_copy(_nbk7_product(), SEED)
    copied = next(m for m in materials if m.material_id == "N-BK7")
    assert copied == copy_catalog_material(SEED.material("N-BK7"), SEED.catalog_version)  # catalog_model copies the same way
    assert physics_difference(copied, merged.model_materials[0]) is None
    assert copied.basis != merged.model_materials[0].basis

    other = {**copied.model_dump(mode="json", exclude_none=True), "valid_band_nm": {"min_nm": "290", "max_nm": "2500", "status": "SPECIFIED"}}
    project = {**json.loads(json.dumps(with_native)), "model_materials": [other]}
    with pytest.raises(PrescriptionImportError) as exc:
        merge_into_project(project, imported(singlet(), model_id="rx"))
    assert exc.value.codes == (E.PROJECT_INVALID,)
    assert "different physics: the valid band (290-2500 nm vs 300-2500 nm) differ" in str(exc.value)
    with pytest.raises(NativeComponentError) as native_exc:
        add_to_project(project, native, instance_id="NAT2")
    assert native_exc.value.problems[0].code.value == "NATIVE_PARAMETER_INVALID" and "the valid band" in native_exc.value.problems[0].message
    absorbing = {**other, "valid_band_nm": copied.valid_band_nm.model_dump(mode="json"), "absorption_coefficient_per_mm": "0.001"}
    assert physics_difference(absorbing, copied) == "the bulk absorption (0.001 vs 0 per mm)"
    resellmeier = {**absorbing, "absorption_coefficient_per_mm": "0", "dispersion": {**absorbing["dispersion"], "b": ["1.04", "0.231792344", "1.01046945"]}}
    assert physics_difference(resellmeier, copied) == "the SELLMEIER dispersion data"
    renamed = {**resellmeier, "dispersion": copied.dispersion.model_dump(mode="json"), "display_name": "Another name", "basis": "Elsewhere."}
    assert physics_difference(renamed, copied) is None  # names and provenance are not physics; an explicit 0 absorption is none
