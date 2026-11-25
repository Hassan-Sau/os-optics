"""A small synthetic catalog-schema-3.6 release for the material-resolution tests (``test_material_resolution_*``).

The suite's frozen fixture catalog (``FIXTURE_CATALOG``, schema 3.0: N-BK7 and fused silica, no identities) plus design
materials that state their identity (``MaterialIdentityV1``), built here: CDGM D-K9 and its anneal variant D-K9-25, OHARA
S-FPL53 (a law on air wavelengths, relative to air, with a measured internal transmittance), SCHOTT F5 and CDGM F5 (one
designation, two makers), the released id ``F2`` (SCHOTT F2) and CDGM F2, a literature model of no maker (``PMMA``), and an
infrared material with a measured index
table and extinction coefficients (``TESTIR``). The coefficients are plausible glass values written for these tests, not
manufacturer data: the tests check resolution and verbatim copying, not the glasses' indices.
"""

from __future__ import annotations

import json
from decimal import Decimal

from catalog_store_helpers import FIXTURE_CATALOG
from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from os_optics._vendor.shared_contracts.design_catalog.material_physics import STANDARD_REFERENCE_AIR, IndexReferenceV1

CATALOG_VERSION = "2026-11-01.1"
BK7 = (("1.03961212", "0.231792344", "1.01046945"), ("0.00600069867", "0.0200179144", "103.560653"))
F5 = (("1.3104463", "0.19603426", "0.96612977"), ("0.00958633048", "0.0457627627", "115.011883"))
F2 = (("1.34533359", "0.209073176", "0.937357162"), ("0.00997743871", "0.0470450767", "111.886764"))
FPL53 = (("0.983021", "0.0695688", "1.11145"), ("0.00492234", "0.0193581", "264.275"))


def sellmeier(coefficients) -> dict:
    b, c = coefficients
    return {"kind": "SELLMEIER", "b": list(b), "c_um2": list(c)}


def tweak(coefficients, b1: str):
    b, c = coefficients
    return (b1, *b[1:]), c


def identity(maker: str, designation: str, edition: str, **more) -> dict:
    return {"manufacturer": maker, "designation": designation, "catalogue_tags": [maker], "source_edition": edition,
            "ordering_status": {"code": "Standard", "design_availability": "CURRENT", "as_of": edition}, **more}


def band(low="310", high="2325") -> dict:
    return {"min_nm": low, "max_nm": high, "status": "SPECIFIED"}


FPL53_REFERENCE = IndexReferenceV1(wavelength="AIR", index="RELATIVE_TO_AIR", reference_air=STANDARD_REFERENCE_AIR,
                                   source_band_nm=("365", "1014"), stated_by="CATALOGUE_FORMAT", basis="AGF: air wavelengths, relative index.")
FPL53_BAND = tuple(f"{v:f}" for v in FPL53_REFERENCE.vacuum_band())

DESIGN_MATERIALS = [
    {"material_id": "CDGM_D-K9", "display_name": "CDGM D-K9", "dispersion": sellmeier(BK7), "valid_band_nm": band(),
     "status": "SPECIFIED", "basis": "Test Sellmeier law (CDGM D-K9 stand-in).", "identity": identity("CDGM", "D-K9", "CDGM 2026-09")},
    {"material_id": "CDGM_D-K9-25", "display_name": "CDGM D-K9-25", "dispersion": sellmeier(tweak(BK7, "1.0396")), "valid_band_nm": band(),
     "status": "SPECIFIED", "basis": "Test Sellmeier law (CDGM D-K9-25 stand-in).",
     "identity": identity("CDGM", "D-K9-25", "CDGM 2026-09", variant_of="CDGM_D-K9", variant="annealing rate 25 C/h"),
     "thermo_optic": {"kind": "SCHOTT_DN_DT", "d0": "0.00000186", "d1": "0.0000000131", "d2": "-0.0000000000137", "e0": "0.000000434",
                      "e1": "0.000000000639", "lambda_tk_um": "0.17", "reference_temperature_c": "20",
                      "valid_temperature_c": ["-40", "80"], "basis": "Test TD line."}},
    {"material_id": "OHARA_S-FPL53", "display_name": "OHARA S-FPL53", "dispersion": sellmeier(FPL53),
     "valid_band_nm": band(*FPL53_BAND), "status": "SPECIFIED", "basis": "Test Sellmeier law (OHARA S-FPL53 stand-in).",
     "identity": identity("OHARA", "S-FPL53", "OHARA 2024-02", aliases=["SFPL53"]),
     "index_reference": FPL53_REFERENCE.model_dump(mode="json", exclude_none=True),
     "bulk_loss": {"kind": "INTERNAL_TRANSMITTANCE", "wavelength_nm": ["400", "700", "1000"], "internal_transmittance": ["0.99", "0.998", "0.997"],
                   "thickness_mm": ["10", "10", "10"], "basis": "Test internal transmittance (10 mm)."}},
    {"material_id": "SCHOTT_F5", "display_name": "SCHOTT F5", "dispersion": sellmeier(F5), "valid_band_nm": band(), "status": "SPECIFIED",
     "basis": "Test Sellmeier law (SCHOTT F5 stand-in).", "identity": identity("SCHOTT", "F5", "SCHOTT 2017-01")},
    {"material_id": "CDGM_F5", "display_name": "CDGM F5", "dispersion": sellmeier(tweak(F5, "1.3104")), "valid_band_nm": band(),
     "status": "SPECIFIED", "basis": "Test Sellmeier law (CDGM F5 stand-in).", "identity": identity("CDGM", "F5", "CDGM 2026-09")},
    {"material_id": "F2", "display_name": "F2", "dispersion": sellmeier(F2), "valid_band_nm": band(), "status": "SPECIFIED",
     "basis": "Test Sellmeier law (SCHOTT F2 stand-in).", "identity": identity("SCHOTT", "F2", "SCHOTT 2017-01")},
    {"material_id": "CDGM_F2", "display_name": "CDGM F2", "dispersion": sellmeier(tweak(F2, "1.3453")), "valid_band_nm": band(),
     "status": "SPECIFIED", "basis": "Test Sellmeier law (CDGM F2 stand-in).", "identity": identity("CDGM", "F2", "CDGM 2026-09")},
    {"material_id": "PMMA_SULTANOVA2009", "display_name": "PMMA (Sultanova 2009)", "dispersion": sellmeier((("1.1819",), ("0.011313",))),
     "valid_band_nm": band("437", "1052"), "status": "SPECIFIED", "basis": "Test stand-in for the Sultanova et al. 2009 literature model.",
     "identity": {"designation": "PMMA", "source_edition": "Sultanova et al. 2009"}},
    {"material_id": "TESTIR", "display_name": "Test IR glass",
     "dispersion": {"kind": "TABULATED", "wavelength_nm": ["2000", "6000", "8000", "10000", "12000", "14000"],
                    "refractive_index": ["2.52", "2.505", "2.502", "2.498", "2.494", "2.489"], "origin": "MEASURED",
                    "measurement": {"temperature_c": "20", "basis": "Test measured table."}},
     "valid_band_nm": band("2000", "14000"), "status": "SPECIFIED", "basis": "Test measured index table.",
     "identity": identity("TESTMAKER", "TESTIR", "Test datasheet 2026"),
     "bulk_loss": {"kind": "EXTINCTION_COEFFICIENT", "wavelength_nm": ["8000", "10000", "12000"], "k": ["0.00001", "0.000005", "0.00001"],
                   "basis": "Test extinction coefficients."}},
]


def catalog_document() -> dict:
    document = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8"))
    document["schema_version"] = "3.6"
    document["catalog_version"] = CATALOG_VERSION
    document["materials"] = document["materials"] + DESIGN_MATERIALS
    return document


def catalog() -> PublicOpticalCatalogV2:
    return PublicOpticalCatalogV2.model_validate(catalog_document())


def decimal(value: str) -> Decimal:
    return Decimal(value)
