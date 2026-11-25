"""Focal-length discovery on the public catalog: ``catalog list`` and ``catalog_products`` apply one definition
(``FocalLengthQuery``), so an Agent finds a negative or a positive lens by its signed nominal focal length without parsing
SKUs or inspecting records. Expected products are worked out from the raw catalog document, independently of the filter."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from os_optics import cli
from os_optics.cli import main
from os_optics.local import clear_installed_catalogs, install_catalog, json_api

from local_helpers import FLAT_MIRROR, PCV_FN50, PCX_F50, PCX_F100, catalog_document

PCX_F150 = "OS-OPT-TEST-LENS-PCX-F150"
BEYOND_F150_BAND_NM = "1064"  # above the band of PCX_F150 (it ends at 700 nm)


def _document() -> dict:
    """The test catalog with a +150 mm lens valid only to 700 nm, so the catalog holds both lenses of a Galilean expander."""

    document = catalog_document()
    lens = copy.deepcopy(next(p for p in document["products"] if p["optical_sku"] == PCX_F100))
    lens["optical_sku"] = PCX_F150
    lens["specifications"]["focal_length_mm"]["value"] = "150"
    lens["simulation_models"][0]["valid_band_nm"]["max_nm"] = "700"
    document["products"].append(lens)
    return document


DOCUMENT = _document()


def _expected(low=None, high=None, *, wavelength_nm=None) -> tuple[list[str], int]:
    """From the raw document: the SKUs in [low, high], and how many products valid at the wavelength publish no focal length."""

    valid = [p for p in DOCUMENT["products"]
             if wavelength_nm is None or any(Decimal(m["valid_band_nm"]["min_nm"]) <= Decimal(wavelength_nm) <= Decimal(m["valid_band_nm"]["max_nm"])
                                             for m in p["simulation_models"])]
    published = [(p["optical_sku"], Decimal(p["specifications"]["focal_length_mm"]["value"])) for p in valid if "focal_length_mm" in p["specifications"]]
    skus = sorted(sku for sku, focal in published if (low is None or focal >= Decimal(low)) and (high is None or focal <= Decimal(high)))
    return skus, len(valid) - len(published)


@pytest.fixture
def catalog_file(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
    return path


@pytest.fixture
def installed():
    install_catalog(json.dumps(DOCUMENT).encode())
    yield
    clear_installed_catalogs()


def _list(catalog_file, capsys, *flags: str) -> dict:
    code = main(["catalog", "--catalog", str(catalog_file), "list", *flags])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def _skus(reply: dict) -> list[str]:
    return [row["optical_sku"] for row in reply["products"]]


@pytest.mark.parametrize(
    "flags",
    [
        ("--focal-length-mm", "-50"),  # the reported command line
        ("--focal-length-mm=-50",),
        ("--focal-length-mm", "-50.0"),
        ("--focal-length-mm", "-5e1"),
        ("--focal-length-mm", "-5E+1"),
        ("--focal-length-mm", "-50."),
        ("--focal-length-mm", "-50", "--focal-tolerance-mm", "0"),
        ("--focal-length-mm", "-49", "--focal-tolerance-mm", "1"),  # the bounds are inclusive
        ("--focal-min-mm", "-50", "--focal-max-mm", "-50"),
        ("--focal-min-mm", "-.6e2", "--focal-max-mm", "-40"),
        ("--focal-min-mm=-60", "--focal-max-mm=-40"),
        ("--focal-sign", "negative"),
        ("--focal-max-mm", "0", "--focal-sign", "negative"),
    ],
)
def test_catalog_list_takes_a_negative_focal_length(catalog_file, capsys, flags):
    reply = _list(catalog_file, capsys, *flags)
    assert _skus(reply) == [PCV_FN50] and reply["total"] == 1 and reply["products"][0]["focal_length_mm"] == "-50"


@pytest.mark.parametrize(
    ("flags", "params", "low", "high"),
    [
        (("--focal-length-mm", "100"), {"focal_length_mm": "100"}, "100", "100"),
        (("--focal-length-mm", "100.0"), {"focal_length_mm": 100.0}, "100", "100"),
        (("--focal-length-mm", "120", "--focal-tolerance-mm", "30"), {"focal_length_mm": 120, "focal_tolerance_mm": 30}, "90", "150"),
        (("--focal-min-mm", "75"), {"focal_min_mm": "75"}, "75", None),
        (("--focal-max-mm", "75"), {"focal_max_mm": 75}, None, "75"),
        (("--focal-min-mm", "-60", "--focal-max-mm", "60"), {"focal_min_mm": -60, "focal_max_mm": 60}, "-60", "60"),
        (("--focal-sign", "positive"), {"focal_sign": "positive"}, "0.001", None),
        (("--focal-min-mm", "1000"), {"focal_min_mm": "1000"}, "1000", None),  # nothing there: an empty answer, not a refusal
    ],
)
def test_the_cli_and_the_json_api_answer_alike(catalog_file, capsys, installed, flags, params, low, high):
    expected, without = _expected(low, high)
    reply = _list(catalog_file, capsys, *flags)
    assert _skus(reply) == expected and reply["total"] == len(expected)
    assert reply["focal_filter"]["excluded_without_focal_length"] == without
    api = json_api.catalog_products(summary=True, **params)
    assert api["ok"] is True and _skus(api) == expected and api["total"] == len(expected) and api["focal_filter"] == reply["focal_filter"]
    assert reply["products"] == api["products"]


def test_a_filtered_response_echoes_the_query_and_what_it_compared(catalog_file, capsys):
    reply = _list(catalog_file, capsys, "--focal-length-mm", "-50", "--focal-tolerance-mm", "5", "--category", "LENS")
    focal = reply["focal_filter"]
    assert (focal["focal_length_mm"], focal["focal_tolerance_mm"], focal["focal_min_mm"], focal["focal_max_mm"], focal["focal_sign"]) == ("-50", "5", "-55", "-45", None)
    assert focal["excluded_without_focal_length"] == 0  # every lens of this catalog publishes one
    assert "nominal focal length" in focal["basis"] and "design_wavelength_nm" in focal["basis"] and "does not change the value" in focal["basis"]
    assert _list(catalog_file, capsys, "--focal-sign", "negative")["focal_filter"]["focal_min_mm"] is None


def test_an_unfiltered_response_has_no_focal_filter(catalog_file, capsys, installed):
    assert "focal_filter" not in _list(catalog_file, capsys, "--category", "LENS")
    assert "focal_filter" not in json_api.catalog_products(category="LENS", summary=True)


def test_missing_data_never_matches_and_is_counted(catalog_file, capsys):
    everything = _list(catalog_file, capsys)
    assert FLAT_MIRROR in _skus(everything)
    expected, without = _expected()
    reply = _list(catalog_file, capsys, "--focal-min-mm", "-1000000", "--focal-max-mm", "1000000")
    assert _skus(reply) == expected and FLAT_MIRROR not in _skus(reply)
    assert without > 0 and reply["focal_filter"]["excluded_without_focal_length"] == without == everything["total"] - reply["total"]
    # counted among the products the other filters kept
    assert _list(catalog_file, capsys, "--focal-sign", "negative", "--form", "FLAT_MIRROR")["focal_filter"]["excluded_without_focal_length"] == 1
    assert _list(catalog_file, capsys, "--focal-sign", "positive", "--category", "LENS")["focal_filter"]["excluded_without_focal_length"] == 0


def test_the_wavelength_filter_selects_products_and_leaves_the_value_alone(catalog_file, capsys):
    near_150 = ("--focal-length-mm", "150", "--focal-tolerance-mm", "10")
    assert _skus(_list(catalog_file, capsys, *near_150)) == [PCX_F150]
    assert _skus(_list(catalog_file, capsys, *near_150, "--wavelength-nm", "633")) == [PCX_F150]
    outside = _list(catalog_file, capsys, *near_150, "--wavelength-nm", BEYOND_F150_BAND_NM)
    assert _skus(outside) == [] and outside["total"] == 0
    # the same row at any wavelength: nominal focal length and its design wavelength, no dispersion evaluated
    (at_450,) = _list(catalog_file, capsys, "--focal-length-mm", "-50", "--wavelength-nm", "450")["products"]
    (at_633,) = _list(catalog_file, capsys, "--focal-length-mm", "-50", "--wavelength-nm", "633")["products"]
    assert at_450 == at_633 and (at_450["focal_length_mm"], at_450["design_wavelength_nm"]) == ("-50", "587.6")
    expected, without = _expected("0.001", None, wavelength_nm=BEYOND_F150_BAND_NM)
    reply = _list(catalog_file, capsys, "--focal-sign", "positive", "--wavelength-nm", BEYOND_F150_BAND_NM)
    assert _skus(reply) == expected and PCX_F150 not in expected and reply["focal_filter"]["excluded_without_focal_length"] == without


def test_paging_slices_the_filtered_products(catalog_file, capsys, installed):
    expected, _ = _expected("0.001", None)
    assert len(expected) == 3
    positive = ("--focal-sign", "positive")
    pages = [_list(catalog_file, capsys, *positive, "--offset", str(offset), "--limit", "2") for offset in (0, 2, 4)]
    assert [_skus(page) for page in pages] == [expected[:2], expected[2:], []]
    assert {page["total"] for page in pages} == {3} and [page["offset"] for page in pages] == [0, 2, 4]
    assert _skus(_list(catalog_file, capsys, *positive, "--limit", "0")) == [] and _list(catalog_file, capsys, *positive, "--limit", "0")["total"] == 3
    api = json_api.catalog_products(focal_sign="positive", offset=1, limit=1, summary=True)
    assert _skus(api) == expected[1:2] and api["total"] == 3 and (api["offset"], api["limit"]) == (1, 1)


def test_full_documents_filter_alike(installed):
    reply = json_api.catalog_products(focal_length_mm=-50)
    assert [p["optical_sku"] for p in reply["products"]] == [PCV_FN50]
    assert reply["products"][0]["specifications"]["focal_length_mm"]["value"] == "-50"


@pytest.mark.parametrize("value", ["-50", "-50.0", "-5e1", "-.5e2"])
def test_every_form_of_a_negative_number_is_a_value_on_the_command_line(catalog_file, capsys, value):
    assert _skus(_list(catalog_file, capsys, "--focal-length-mm", value)) == [PCV_FN50]


@pytest.mark.parametrize("value", ["-50mm", "-1e2x"])
def test_a_negative_looking_value_with_a_unit_reaches_the_refusal_and_not_the_parser(catalog_file, capsys, value):
    assert main(["catalog", "--catalog", str(catalog_file), "list", "--focal-length-mm", value]) == 64
    assert capsys.readouterr().err.startswith("error: FOCAL_VALUE_INVALID: ")


@pytest.mark.parametrize("value", ["-Inf", "--5"])
def test_what_does_not_start_like_a_number_stays_an_option(catalog_file, capsys, value):
    """A usage error on every supported Python: neither is a number, and an option cannot take them."""

    with pytest.raises(SystemExit) as stopped:
        main(["catalog", "--catalog", str(catalog_file), "list", "--focal-length-mm", value])
    assert stopped.value.code == 64


def test_the_negative_value_rule_uses_no_private_argparse_attribute():
    assert "_negative_number_matcher" not in Path(cli.__file__).read_text(encoding="utf-8")


REFUSALS = [
    (("--focal-length-mm", "50mm"), {"focal_length_mm": "50mm"}, "FOCAL_VALUE_INVALID", "focal_length_mm"),
    (("--focal-length-mm", "-50mm"), {"focal_length_mm": "-50mm"}, "FOCAL_VALUE_INVALID", "focal_length_mm"),
    (("--focal-min-mm", "nan"), {"focal_min_mm": "nan"}, "FOCAL_VALUE_INVALID", "focal_min_mm"),
    (("--focal-length-mm", "-50", "--focal-min-mm", "-60"), {"focal_length_mm": -50, "focal_min_mm": -60}, "FOCAL_BOUNDS_CONFLICT", "focal_length_mm"),
    (("--focal-tolerance-mm", "5"), {"focal_tolerance_mm": 5}, "FOCAL_TOLERANCE_WITHOUT_VALUE", "focal_tolerance_mm"),
    (("--focal-length-mm", "-50", "--focal-tolerance-mm", "-5"), {"focal_length_mm": -50, "focal_tolerance_mm": -5}, "FOCAL_TOLERANCE_NEGATIVE", "focal_tolerance_mm"),
    (("--focal-min-mm", "-40", "--focal-max-mm", "-60"), {"focal_min_mm": -40, "focal_max_mm": -60}, "FOCAL_RANGE_INVERTED", "focal_min_mm"),
    (("--focal-min-mm", "40", "--focal-max-mm", "60", "--focal-sign", "negative"), {"focal_min_mm": 40, "focal_max_mm": 60, "focal_sign": "negative"},
     "FOCAL_SIGN_CONTRADICTS_RANGE", "focal_sign"),
]


@pytest.mark.parametrize(("flags", "params", "reason", "parameter"), REFUSALS)
def test_a_refused_query_gives_its_reason_on_both_surfaces(catalog_file, capsys, installed, flags, params, reason, parameter):
    assert main(["catalog", "--catalog", str(catalog_file), "list", *flags]) == 64
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith(f"error: {reason}: ")
    reply = json_api.catalog_products(**params)
    (problem,) = reply["error"]["problems"]
    assert reply["ok"] is False and problem["code"] == "CATALOG_QUERY_INVALID"
    assert problem["details"] == {"reason": reason} and problem["location"] == parameter and problem["message"] in captured.err


def test_an_unknown_sign_is_refused_on_both_surfaces(catalog_file, capsys, installed):
    with pytest.raises(SystemExit) as refused:
        main(["catalog", "--catalog", str(catalog_file), "list", "--focal-sign", "NEGATIVE"])
    assert refused.value.code == 64
    capsys.readouterr()
    assert json_api.catalog_products(focal_sign="NEGATIVE")["error"]["problems"][0]["details"] == {"reason": "FOCAL_SIGN_UNKNOWN"}


def test_the_help_states_the_semantics(capsys):
    with pytest.raises(SystemExit) as shown:
        main(["catalog", "list", "--help"])
    assert shown.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for flag in ("--focal-length-mm", "--focal-tolerance-mm", "--focal-min-mm", "--focal-max-mm", "--focal-sign"):
        assert flag in text
    assert "signed nominal focal length" in text and "millimetres" in text and "focal_length_tolerance_percent" in text
    assert "does not change the value compared" in text and "excluded_without_focal_length" in text and "never matches" in text


def test_a_galilean_expander_lookup_finds_both_lenses_by_focal_length(catalog_file, capsys, installed):
    """A customer's request: a Galilean beam expander, a negative lens near -50 mm and a positive lens near +150 mm.

    The lookup assumes no SKU spelling and no number of products: each lens comes back with the sign and the focal length
    asked for, and the pair expands the beam by f2 / |f1|. Hand reference: M = 150 / 50 = 3, lens spacing f1 + f2 = 100 mm."""

    tolerance = "10"
    negative = _list(catalog_file, capsys, "--category", "LENS", "--focal-length-mm", "-50", "--focal-tolerance-mm", tolerance)
    positive = _list(catalog_file, capsys, "--category", "LENS", "--focal-length-mm", "150", "--focal-tolerance-mm", tolerance)
    assert negative["total"] >= 1 and positive["total"] >= 1
    for reply, nominal in ((negative, Decimal(-50)), (positive, Decimal(150))):
        for row in reply["products"]:
            assert abs(Decimal(row["focal_length_mm"]) - nominal) <= Decimal(tolerance) and row["category"] == "LENS"
    f1, f2 = Decimal(negative["products"][0]["focal_length_mm"]), Decimal(positive["products"][0]["focal_length_mm"])
    assert f1 < 0 < f2 and f2 / -f1 == 3 and f1 + f2 == 100

    # the same lookup in the JSON API and in Python, by bounds
    api = json_api.catalog_products(category="LENS", focal_min_mm=-60, focal_max_mm=-40, summary=True)
    assert _skus(api) == _skus(negative)
    hosted, _ = json_api.call_method("catalog_products", {"category": "LENS", "focal_min_mm": "-60", "focal_max_mm": "-40", "summary": True})  # the Designer's worker
    assert _skus(hosted) == _skus(negative)
    assert {"focal_length_mm", "focal_tolerance_mm", "focal_min_mm", "focal_max_mm", "focal_sign"} <= set(json_api.capabilities()["json_api_methods"]["catalog_products"]["parameters"])
    from os_optics.local import load_public_catalog

    catalog = load_public_catalog()
    assert [p.optical_sku for p in catalog.select(category="LENS", focal_min_mm="140", focal_max_mm="160")] == _skus(positive)
    assert PCX_F50 not in _skus(negative) + _skus(positive)
