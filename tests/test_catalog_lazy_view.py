"""A release of the local catalog store is read without validating the whole document again (``StoredCatalog``): the
store validated it when it entered and recorded by which catalog contract; a reader with that contract builds the
records a run asks for one by one, and the whole catalog on first use by anything that walks it. Every result is the
one the fully validated catalog gives."""

from __future__ import annotations

import hashlib
import json

import pytest

from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from os_optics.catalog import import_release
from os_optics.catalog.store import CatalogStore, contract_identity, validated_here
from os_optics.local import LocalRunError, json_api, load_public_catalog, lower_project
from os_optics.local import catalog as catalog_module
from os_optics.local.catalog import StoredCatalog
from os_optics.local.example import example_project_of_kind

from catalog_store_helpers import FIXTURE_CATALOG, REAL_CATALOG
from local_helpers import PCX_F100, TEST_CATALOG_VERSION, catalog_document, element, plane, project, source

FIXTURE_BYTES = FIXTURE_CATALOG.read_bytes()


@pytest.fixture(autouse=True)
def _fresh_process_memo():
    """Each test reads the store as a new process does (the parsed-catalog memo is per process)."""

    catalog_module._PARSED_CATALOGS.clear()
    yield
    catalog_module._PARSED_CATALOGS.clear()


def _count_full_validations(monkeypatch) -> list[int]:
    calls: list[int] = []
    parse = catalog_module._parse_catalog

    def counted(document):
        calls.append(1)
        return parse(document)

    monkeypatch.setattr(catalog_module, "_parse_catalog", counted)
    return calls


def _index(store_root) -> dict:
    return json.loads((store_root / "state.json").read_text(encoding="utf-8"))


def _lens_project() -> dict:
    return project([source(), element("L1", "lens", PCX_F100)], [plane("P", (150.0, 0.0, 0.0))])


def test_a_stored_release_records_the_contract_that_validated_it(empty_catalog_store):
    import_release(FIXTURE_BYTES)
    record = _index(empty_catalog_store)["releases"][TEST_CATALOG_VERSION]
    identity = contract_identity()
    assert len(identity) == 64 and record["validated_by"] == identity
    (release,) = CatalogStore(empty_catalog_store).read_state().releases.values()
    assert validated_here(release)


def test_a_run_reads_a_validated_release_without_validating_the_whole_document(empty_catalog_store, monkeypatch):
    import_release(FIXTURE_BYTES)
    calls = _count_full_validations(monkeypatch)
    catalog = load_public_catalog()
    assert isinstance(catalog, StoredCatalog) and catalog.catalog_version == TEST_CATALOG_VERSION and catalog.schema_version == "3.0"
    assert load_public_catalog() is catalog  # one object per release in a process (the compiled-element cache keys on it)
    assert load_public_catalog(catalog) is catalog
    lowered = lower_project(_lens_project())
    assert lowered.catalog_version == TEST_CATALOG_VERSION and calls == []
    full = PublicOpticalCatalogV2.model_validate(catalog_document())
    assert lowered.engine_input_hash == lower_project(_lens_project(), full).engine_input_hash
    # the first use that walks the catalog builds the whole model, once
    assert len(catalog.products) == len(full.products) and catalog.products == full.products
    assert catalog.select(category="LENS") == full.select(category="LENS")
    assert calls == [1]
    assert catalog.get(PCX_F100) is catalog.full().get(PCX_F100)


def _every_record_equal(view: StoredCatalog, full: PublicOpticalCatalogV2) -> None:
    for product in full.products:
        assert view.get(product.optical_sku) == product, product.optical_sku
    for material in full.materials:
        assert view.material(material.material_id) == material, material.material_id
    for coating in full.coatings:
        assert view.coating(coating.coating_id) == coating, coating.coating_id
    assert view._full is None  # (compared record by record: the whole model was never built)
    assert view.get("OS-OPT-NOT-PUBLISHED") is None and view.material("NOPE") is None and view.coating("NOPE") is None
    assert view.get(None) is None  # type: ignore[arg-type]
    assert view.summary() == json_api.catalog_summary(full)


def test_every_record_built_on_its_own_is_the_record_of_the_whole_catalog():
    """The catalog-level validation sorts and cross-checks; it does not rewrite a product, material or coating."""

    _every_record_equal(StoredCatalog(json.loads(FIXTURE_BYTES)), PublicOpticalCatalogV2.model_validate_json(FIXTURE_BYTES))


def test_every_record_of_the_seed_release_built_on_its_own_is_the_record_of_the_whole_catalog():
    data = REAL_CATALOG.read_bytes()
    _every_record_equal(StoredCatalog(json.loads(data)), PublicOpticalCatalogV2.model_validate_json(data))


@pytest.mark.parametrize("kind", ["focus", "folded", "beam_expander"])
def test_the_example_projects_lower_to_the_same_request_through_the_view(real_catalog_store, kind):
    view = load_public_catalog()
    assert isinstance(view, StoredCatalog)
    full = PublicOpticalCatalogV2.model_validate_json(REAL_CATALOG.read_bytes())
    document = example_project_of_kind(kind, full)
    through_view, through_model = lower_project(document, view), lower_project(document, full)
    assert through_view.engine_input_hash == through_model.engine_input_hash
    assert through_view.simulated_parts == through_model.simulated_parts
    assert through_view.elements == through_model.elements


def test_a_release_stored_by_another_contract_is_validated_in_full_once_and_recorded(empty_catalog_store, monkeypatch):
    import_release(FIXTURE_BYTES)
    path = empty_catalog_store / "state.json"
    for stale in ("0" * 64, None):  # an Engine with another catalog contract; an index written before the record existed
        index = _index(empty_catalog_store)
        if stale is None:
            del index["releases"][TEST_CATALOG_VERSION]["validated_by"]
        else:
            index["releases"][TEST_CATALOG_VERSION]["validated_by"] = stale
        path.write_text(json.dumps(index), encoding="utf-8")
        catalog_module._PARSED_CATALOGS.clear()
        calls = _count_full_validations(monkeypatch)
        catalog = load_public_catalog()
        assert isinstance(catalog, StoredCatalog) and calls == [1]
        assert catalog.get(PCX_F100) is catalog.full().get(PCX_F100) and calls == [1]
        assert _index(empty_catalog_store)["releases"][TEST_CATALOG_VERSION]["validated_by"] == contract_identity()
        catalog_module._PARSED_CATALOGS.clear()  # the next process
        load_public_catalog().get(PCX_F100)
        assert calls == [1]


def test_a_stored_release_that_is_not_valid_is_refused_in_full_when_no_contract_vouches_for_it(empty_catalog_store):
    """A file found on recovery or written by another Engine gets today's full validation: a bad product refuses the load."""

    document = catalog_document()
    next(p for p in document["products"] if p["optical_sku"] == PCX_F100)["simulation_models"][0]["surfaces"][0]["radius_mm"] = "not a number"
    data = json.dumps(document, sort_keys=True).encode("utf-8")
    import_release(FIXTURE_BYTES)
    (empty_catalog_store / "releases" / f"{TEST_CATALOG_VERSION}.json").write_bytes(data)
    index = _index(empty_catalog_store)
    index["releases"][TEST_CATALOG_VERSION].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), validated_by="0" * 64)
    (empty_catalog_store / "state.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(LocalRunError) as refused:
        load_public_catalog()
    assert [p.code.value for p in refused.value.problems] == ["CATALOG_INVALID"]
    assert _index(empty_catalog_store)["releases"][TEST_CATALOG_VERSION]["validated_by"] == "0" * 64  # nothing was recorded


def test_a_record_that_does_not_build_is_reported_when_it_is_used_and_names_itself():
    document = catalog_document()
    next(p for p in document["products"] if p["optical_sku"] == PCX_F100)["simulation_models"][0]["surfaces"][0]["radius_mm"] = "not a number"
    view = StoredCatalog(document)
    other = next(p["optical_sku"] for p in document["products"] if p["optical_sku"] != PCX_F100)
    assert view.get(other).optical_sku == other
    with pytest.raises(LocalRunError) as refused:
        view.get(PCX_F100)
    assert [p.code.value for p in refused.value.problems] == ["CATALOG_INVALID"]
    assert f"optical_sku {PCX_F100} of public catalog {TEST_CATALOG_VERSION}" in str(refused.value)


def test_the_count_commands_read_a_validated_release_without_building_it(empty_catalog_store, monkeypatch):
    from os_optics import cli_doctor  # noqa: PLC0415

    import_release(FIXTURE_BYTES)
    full = PublicOpticalCatalogV2.model_validate_json(FIXTURE_BYTES)
    calls = _count_full_validations(monkeypatch)
    assert json_api.catalog_summary(load_public_catalog()) == json_api.catalog_summary(full)
    assert calls == []
    monkeypatch.setattr(PublicOpticalCatalogV2, "model_validate_json", classmethod(lambda cls, data: pytest.fail("validated in full")))
    entry = cli_doctor._catalog()
    assert (entry["status"], entry["products"], entry["catalog_version"]) == ("OK", len(full.products), TEST_CATALOG_VERSION)


def test_a_record_is_built_in_the_contracts_decimal_context_whatever_the_callers_is(empty_catalog_store):
    """A record built on first use inside a caller's narrow decimal context is the record of the whole catalog, and an
    explicit catalog first read there is read the same."""

    from decimal import Context, localcontext  # noqa: PLC0415

    import_release(FIXTURE_BYTES)
    reference = lower_project(_lens_project(), catalog_document())
    catalog_module._PARSED_CATALOGS.clear()
    with localcontext(Context(prec=3)):
        stored = lower_project(_lens_project())
        catalog_module._PARSED_CATALOGS.clear()
        explicit = lower_project(_lens_project(), catalog_document())
    assert stored.engine_input_hash == explicit.engine_input_hash == reference.engine_input_hash
    assert stored.elements == reference.elements


def test_explicit_bytes_of_a_stored_release_are_validated_in_full_and_are_the_same_object(empty_catalog_store, monkeypatch):
    import_release(FIXTURE_BYTES)
    calls = _count_full_validations(monkeypatch)
    stored = load_public_catalog()
    assert calls == []
    assert load_public_catalog(FIXTURE_BYTES) is stored and calls == [1]  # an explicit catalog: the whole document, once
    assert load_public_catalog(FIXTURE_BYTES) is stored and calls == [1]
