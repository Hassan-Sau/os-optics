"""Which catalog a local run uses, the in-memory install of the JSON API, and ``update_project_catalog``.

Resolution order: an explicit catalog > a catalog installed in memory (``install_catalog``; the release a Project pins
when installed, else the last installed) > the local store (the pinned release, else the active one).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json

import pytest

from os_optics.catalog import import_release
from os_optics.cli import main as cli_main
from os_optics.local import (
    LocalRunError,
    catalog_resolution_status,
    install_catalog,
    installed_catalogs,
    load_public_catalog,
    pinned_catalog_version,
    run_optical_project,
    update_project_catalog,
)
from os_optics.local import json_api

from catalog_store_helpers import FIXTURE_CATALOG
from local_helpers import PCX_F100, TEST_CATALOG_VERSION, catalog_document, element, plane, project, source

FIXTURE_BYTES = FIXTURE_CATALOG.read_bytes()
NEXT_VERSION = "2026-01-02.1"


def _release(version: str, edit=None) -> bytes:
    document = catalog_document()
    document["catalog_version"] = version
    if edit is not None:
        edit(document)
    return json.dumps(document, sort_keys=True).encode("utf-8")


def _lens_project(version: str | None = TEST_CATALOG_VERSION) -> dict:
    return project([source(), element("L1", "lens", PCX_F100, catalog_version=version)], [plane("P", (150.0, 0.0, 0.0))])


# ---- resolution order ----------------------------------------------------------------------------------------------------


def test_the_default_is_the_stores_active_release_and_an_explicit_catalog_wins(empty_catalog_store):
    import_release(FIXTURE_BYTES)
    import_release(_release(NEXT_VERSION), activate=True)
    assert load_public_catalog().catalog_version == NEXT_VERSION
    assert load_public_catalog(catalog_version=TEST_CATALOG_VERSION).catalog_version == TEST_CATALOG_VERSION
    assert load_public_catalog(catalog_document()).catalog_version == TEST_CATALOG_VERSION
    assert catalog_resolution_status()["active"] == {"catalog_version": NEXT_VERSION, "schema_version": "3.0",
                                                      "sha256": hashlib.sha256(_release(NEXT_VERSION)).hexdigest(), "source": "store"}


def test_a_run_uses_the_release_its_components_pin(empty_catalog_store):
    import_release(FIXTURE_BYTES)
    import_release(_release(NEXT_VERSION))  # active
    assert pinned_catalog_version(_lens_project()) == TEST_CATALOG_VERSION
    assert run_optical_project(_lens_project()).catalog_version == TEST_CATALOG_VERSION
    assert run_optical_project(_lens_project(None)).catalog_version == NEXT_VERSION  # unpinned: the active release
    mixed = _lens_project()
    mixed["components"].append(element("L2", "lens", PCX_F100, (50.0, 0.0, 0.0), catalog_version=NEXT_VERSION))
    assert pinned_catalog_version(mixed) is None
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(mixed)
    assert "CATALOG_VERSION_MISMATCH" in [c.value for c in exc.value.codes]


def test_an_installed_catalog_takes_precedence_over_the_store(empty_catalog_store):
    import_release(_release(NEXT_VERSION))
    installed = install_catalog(FIXTURE_BYTES, expected_sha256=hashlib.sha256(FIXTURE_BYTES).hexdigest())
    assert installed.summary()["catalog_version"] == TEST_CATALOG_VERSION and installed.summary()["product_count"] == 10
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    # a pinned release that is not installed: the active installed one (the run reports the mismatch), never the store
    assert load_public_catalog(catalog_version=NEXT_VERSION).catalog_version == TEST_CATALOG_VERSION
    install_catalog(_release(NEXT_VERSION))
    assert load_public_catalog().catalog_version == NEXT_VERSION
    assert run_optical_project(_lens_project()).catalog_version == TEST_CATALOG_VERSION  # pinned and installed
    assert [i.catalog.catalog_version for i in installed_catalogs()] == [TEST_CATALOG_VERSION, NEXT_VERSION]
    install_catalog(FIXTURE_BYTES)  # re-installing makes it active again
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION


def test_install_refuses_what_it_cannot_use():
    with pytest.raises(LocalRunError) as sha:
        install_catalog(FIXTURE_BYTES, expected_sha256="0" * 64)
    assert sha.value.codes[0].value == "CATALOG_INVALID"
    with pytest.raises(LocalRunError) as newer:
        install_catalog(_release("2026-01-03.1", lambda d: d.update(schema_version="4.0")))
    assert newer.value.codes[0].value == "ENGINE_UPGRADE_REQUIRED"
    with pytest.raises(LocalRunError) as junk:
        install_catalog("not bytes")
    assert junk.value.codes[0].value == "CATALOG_INVALID"


# ---- JSON API ------------------------------------------------------------------------------------------------------------


def test_json_api_install_and_status(empty_catalog_store):
    assert json_api.catalog_status() == {"ok": True, "available": False, "active": None, "installed": [], "supported_schema_versions": ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6"]}
    capabilities = json_api.capabilities()
    assert capabilities["ok"] is True and capabilities["catalog"] is None and capabilities["catalog_version"] is None
    for call in (lambda: json_api.catalog_products(limit=1), lambda: json_api.catalog_product(PCX_F100), json_api.catalog_materials,
                 json_api.catalog_coatings, lambda: json_api.check_project(_lens_project())):
        refused = call()
        assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "CATALOG_NOT_AVAILABLE"
    digest = hashlib.sha256(FIXTURE_BYTES).hexdigest()
    assert json_api.install_catalog(memoryview(FIXTURE_BYTES), digest) == {
        "ok": True, "catalog_version": TEST_CATALOG_VERSION, "schema_version": "3.0", "sha256": digest,
        "size_bytes": len(FIXTURE_BYTES), "product_count": 10}
    status = json_api.catalog_status()
    assert status["active"] == {"catalog_version": TEST_CATALOG_VERSION, "schema_version": "3.0", "sha256": digest, "source": "installed"}
    assert status["installed"] == [{"catalog_version": TEST_CATALOG_VERSION, "sha256": digest}]
    assert json_api.capabilities()["catalog"]["product_count"] == 10
    assert json_api.catalog_products(limit=1)["catalog_version"] == TEST_CATALOG_VERSION
    assert json_api.check_project(_lens_project())["ok"] is True
    refused = json_api.install_catalog(b"{}")
    assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "CATALOG_INVALID"


# ---- update_project_catalog ----------------------------------------------------------------------------------------------


def _changed_lens(document: dict) -> None:
    lens = next(p for p in document["products"] if p["optical_sku"] == PCX_F100)
    lens["specifications"]["focal_length_tolerance_percent"]["value"] = "2"


def _reworded_lens(document: dict) -> None:
    lens = next(p for p in document["products"] if p["optical_sku"] == PCX_F100)
    lens.update(display_name=lens["display_name"] + " rev B", summary=lens["summary"] + " Revised.",
                notices=[*lens.get("notices", ()), "A new notice."])


def _withdrawn_lens(document: dict) -> None:
    document["products"] = [p for p in document["products"] if p["optical_sku"] != PCX_F100]
    document["categories_offered"] = sorted({p["category"] for p in document["products"]})


def test_unchanged_products_are_repinned_to_the_active_release(empty_catalog_store):
    import_release(FIXTURE_BYTES)
    import_release(_release(NEXT_VERSION))
    update = update_project_catalog(_lens_project())
    assert (update.status, update.catalog_version) == ("UPDATED", NEXT_VERSION)
    assert [r["instance_id"] for r in update.repinned] == ["L1"] and update.changed == update.withdrawn == ()
    assert pinned_catalog_version(update.project) == NEXT_VERSION
    assert run_optical_project(update.project).catalog_version == NEXT_VERSION
    again = update_project_catalog(update.project)
    assert again.status == "CURRENT" and again.project == update.project
    report = update.as_dict()
    assert "unchanged" in report["summary"] and [s["action"] for s in report["next_steps"]] == ["RERUN", "SUBMIT"]
    assert again.as_dict()["next_steps"] == []


def test_a_product_whose_words_alone_changed_is_repinned(empty_catalog_store):
    """Names, summary and notices are not engineering content: a release that only rewords the lens re-pins it like an
    unchanged product, without --include-changed."""

    import_release(FIXTURE_BYTES)
    update = update_project_catalog(_lens_project(), _release(NEXT_VERSION, _reworded_lens))
    assert (update.status, update.catalog_version) == ("UPDATED", NEXT_VERSION)
    assert [r["instance_id"] for r in update.repinned] == ["L1"] and update.changed == update.withdrawn == ()


def test_changed_and_withdrawn_products_are_reported_and_keep_their_pin(empty_catalog_store):
    import_release(FIXTURE_BYTES)
    changed_release = _release(NEXT_VERSION, _changed_lens)
    update = update_project_catalog(_lens_project(), changed_release)
    assert update.status == "NEEDS_ATTENTION" and [r["optical_sku"] for r in update.changed] == [PCX_F100]
    assert pinned_catalog_version(update.project) == TEST_CATALOG_VERSION and update.repinned == ()
    assert [s["action"] for s in update.next_steps] == ["INCLUDE_CHANGED"] and "--include-changed" in update.next_steps[0]["command"]
    forced = update_project_catalog(_lens_project(), changed_release, include_changed=True)
    # re-pinned, but never presented as a clean update: the product changed, so the design must be run and reviewed
    assert forced.status == "UPDATED_UNVERIFIED" and pinned_catalog_version(forced.project) == NEXT_VERSION and len(forced.changed) == 1
    assert [s["action"] for s in forced.next_steps] == ["RERUN_AND_REVIEW", "SUBMIT"]
    assert "see next_steps" in forced.summary and "review" in forced.next_steps[0]["message"]  # what to do is in next_steps
    withdrawn = update_project_catalog(_lens_project(), _release(NEXT_VERSION, _withdrawn_lens), include_changed=True)
    assert withdrawn.status == "NEEDS_ATTENTION" and [r["optical_sku"] for r in withdrawn.withdrawn] == [PCX_F100]
    assert [s["action"] for s in withdrawn.next_steps] == ["CHOOSE_ANOTHER_PRODUCT"]


def _supply_facts_changed(document: dict) -> None:
    """The next release only states how OpenSci supplies the lens (a schema 3.1 release)."""

    document["schema_version"] = "3.1"
    for product in document["products"]:
        product["opensci_mounting"], product["delivered_form"] = "NOT_AVAILABLE", "OPTIC"
    lens = next(p for p in document["products"] if p["optical_sku"] == PCX_F100)
    lens["commercial_readiness"] = "SIMULATION_ONLY" if lens["commercial_readiness"] == "ORDERABLE" else "ORDERABLE"


def test_a_product_whose_supply_facts_alone_changed_is_unchanged(empty_catalog_store):
    """Whether OpenSci supplies or mounts a product never changes its simulation: it re-pins as unchanged."""

    import_release(FIXTURE_BYTES)
    update = update_project_catalog(_lens_project(), _release(NEXT_VERSION, _supply_facts_changed))
    assert (update.status, update.changed) == ("UPDATED", ()) and [r["instance_id"] for r in update.repinned] == ["L1"]


def test_an_old_release_that_is_not_on_this_device_is_unverified(empty_catalog_store):
    update = update_project_catalog(_lens_project(), _release(NEXT_VERSION))
    # (the old release may still be obtained -- a download, an import: the component keeps its pin until it is asked for)
    assert update.status == "NEEDS_ATTENTION" and [r["instance_id"] for r in update.unverified] == ["L1"]
    assert update.kept_unverified == update.unverified and pinned_catalog_version(update.project) == TEST_CATALOG_VERSION
    # unverified is explained, with the command that moves the component anyway and what to do afterwards
    report = update.as_dict()
    assert report["old_releases"] == [{"catalog_version": TEST_CATALOG_VERSION, "reason": "NOT_OBTAINED"}]
    (step,) = report["next_steps"]
    assert step["action"] == "INCLUDE_UNVERIFIED" and step["command"].endswith("--out PROJECT --include-changed")
    assert f"{TEST_CATALOG_VERSION} is not on this device" in step["message"] and "run the Project again" in step["message"]
    forced = update_project_catalog(_lens_project(), _release(NEXT_VERSION), include_changed=True)
    assert forced.status == "UPDATED_UNVERIFIED" and pinned_catalog_version(forced.project) == NEXT_VERSION
    assert [r["instance_id"] for r in forced.unverified] == ["L1"] and "could not be compared" in forced.summary
    assert [s["action"] for s in forced.next_steps] == ["RERUN_AND_REVIEW", "SUBMIT"]
    assert "cannot be compared" in forced.next_steps[0]["message"]


@pytest.mark.parametrize("interval", ["0", "3600"])
def test_the_unverified_step_is_number_correct_and_names_the_way_to_the_old_release(empty_catalog_store, monkeypatch, interval):
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", interval)
    monkeypatch.setenv("OPENSCI_CATALOG_URL", "http://127.0.0.1:9")  # nothing listens: the old release cannot be fetched
    update = update_project_catalog(_lens_project(), _release(NEXT_VERSION))
    (step,) = update.next_steps
    message = step["message"]
    # one component: 'its' / 'it', never 'their' / 'them'
    assert ": its product data cannot be compared" in message and "to move it to" in message
    assert " them " not in message and "their" not in message
    assert update.summary == "1 catalog component keeps its old catalog release; see next_steps"
    if interval == "0":  # automatic catalog access is off: say so, and how to bring the old release in
        assert "automatic catalog access is off" in message and "could not be downloaded" not in message
        assert f"'os-optics catalog sync --release {TEST_CATALOG_VERSION}'" in message and "catalog import" in message
        assert "once the catalog service is reachable" not in message
    else:
        assert "could not be downloaded" in message and "once the catalog service is reachable" in message


def test_update_catalog_command_and_json_api(empty_catalog_store, tmp_path):
    import_release(FIXTURE_BYTES)
    import_release(_release(NEXT_VERSION))
    path = tmp_path / "project.json"
    path.write_text(json.dumps(_lens_project()), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(["project", "update-catalog", str(path), "--out", str(path)])
    report = json.loads(out.getvalue())
    assert code == 0 and report["status"] == "UPDATED" and report["catalog_version"] == NEXT_VERSION
    assert report["project_written"] is True and report["next_steps"][0]["action"] == "RERUN"
    assert pinned_catalog_version(path) == NEXT_VERSION
    path.write_text(json.dumps(_lens_project()), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(["project", "update-catalog", str(path)])
    report = json.loads(out.getvalue())
    # without --out nothing is written, and the report says so first
    assert code == 0 and report["project_written"] is False and report["next_steps"][0]["action"] == "WRITE_PROJECT"
    assert report["next_steps"][0]["command"] == f"os-optics project update-catalog {path} --out {path}"
    assert pinned_catalog_version(path) == TEST_CATALOG_VERSION
    result = json_api.update_project_catalog(_lens_project())
    assert result["ok"] is True and result["status"] == "UPDATED" and result["project"]["components"][1]["product"]["catalog_version"] == NEXT_VERSION


def _retire(store_root, version: str, schema_version: str = "2.1") -> None:
    """Record ``version`` in the store as written in an earlier catalog schema (a release kept on the device from
    before an Engine upgrade)."""

    state_file = store_root / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["releases"][version]["schema_version"] = schema_version
    state_file.write_text(json.dumps(state), encoding="utf-8")


def test_a_release_in_an_earlier_catalog_schema_is_not_available_rather_than_invalid(empty_catalog_store, monkeypatch):
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "0")
    import_release(FIXTURE_BYTES)
    _retire(empty_catalog_store, TEST_CATALOG_VERSION)
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    assert [c.value for c in exc.value.codes] == ["CATALOG_NOT_AVAILABLE"]
    assert "earlier format" in str(exc.value) and "os-optics catalog sync" in str(exc.value)


def test_update_catalog_reports_an_old_release_in_an_earlier_schema_as_unreadable(empty_catalog_store, monkeypatch):
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "0")
    import_release(FIXTURE_BYTES)
    _retire(empty_catalog_store, TEST_CATALOG_VERSION)
    update = update_project_catalog(_lens_project(), _release(NEXT_VERSION))
    report = update.as_dict()
    # (no download or retry makes this Engine read the old release: the component is moved, and stays listed with the reason)
    assert update.status == "UPDATED_UNVERIFIED" and [r["instance_id"] for r in update.unverified] == ["L1"]
    assert pinned_catalog_version(update.project) == NEXT_VERSION and update.kept_unverified == ()
    assert report["old_releases"] == [{"catalog_version": TEST_CATALOG_VERSION, "reason": "NOT_READABLE"}]
    assert [s["action"] for s in report["next_steps"]] == ["RERUN_AND_REVIEW", "SUBMIT"]
    step = report["next_steps"][0]
    assert "earlier catalog format" in step["message"] and "once the catalog service is reachable" not in step["message"]
    forced = update_project_catalog(_lens_project(), _release(NEXT_VERSION), include_changed=True)
    assert forced.status == "UPDATED_UNVERIFIED" and pinned_catalog_version(forced.project) == NEXT_VERSION
    assert forced.as_dict() == report
