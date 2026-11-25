"""The local catalog store and its synchronization with the catalog service (``os_optics.catalog``).

Against a local stand-in of ``GET /api/v1/catalog`` + ``GET /api/v1/catalog/releases/<v>`` (``catalog_service_stub``):
first-use sync, heartbeat throttling and conditional requests (304), gzip downloads, refusal of bytes that differ from
the manifest, a newer catalog schema (``ENGINE_UPGRADE_REQUIRED``), offline fallback, pinned-release fetch, the
never-overwrite rule, origin resolution, verify / repair, import, the browser runtime (no network) and the CLI.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from os_optics import catalog as catalog_api
from os_optics.catalog import CatalogError, CatalogErrorCode, CatalogStore
from os_optics.cli import main as cli_main
from os_optics.deployment import default_service_origin
from os_optics.local import LocalRunError, load_public_catalog, run_optical_project

from catalog_service_stub import CatalogServiceStub
from catalog_store_helpers import FIXTURE_CATALOG
from local_helpers import TEST_CATALOG_VERSION

FIXTURE_BYTES = FIXTURE_CATALOG.read_bytes()
NEXT_VERSION = "2026-01-02.1"


def _release(version: str, *, edit=None) -> bytes:
    document = json.loads(FIXTURE_BYTES)
    document["catalog_version"] = version
    if edit is not None:
        edit(document)
    return (json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


NEXT_BYTES = _release(NEXT_VERSION)


@pytest.fixture
def stub():
    with CatalogServiceStub() as service:
        yield service


@pytest.fixture
def online(empty_catalog_store, stub, monkeypatch):
    """An empty store of this test, automatic checks on (hourly), pointed at the stub service."""

    monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url)
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "3600")
    return CatalogStore(empty_catalog_store)


def _age_last_check(store: CatalogStore, seconds: float = 7200.0) -> None:
    state = store.read_state()
    store.write_state(replace(state, checked_at=state.checked_at - seconds))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- first use, heartbeat, conditional requests --------------------------------------------------------------------------


def test_first_use_syncs_once_and_later_uses_are_offline_within_the_interval(online, stub, capsys):
    stub.state.publish(FIXTURE_BYTES)
    catalog = load_public_catalog()
    assert catalog.catalog_version == TEST_CATALOG_VERSION
    # the automatic download is announced by one line on stderr (release, size, service), stdout stays clean
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    notice = captured.err
    assert notice.startswith(f"os-optics: downloaded public optical catalog {TEST_CATALOG_VERSION} (")
    assert f" MB) from {stub.url} into {online.root}" in notice and "active release now" in notice
    assert stub.state.paths() == ["/api/v1/catalog", f"/api/v1/catalog/releases/{TEST_CATALOG_VERSION}"]
    assert (online.root / "releases" / f"{TEST_CATALOG_VERSION}.json").read_bytes() == FIXTURE_BYTES
    for _ in range(3):
        assert load_public_catalog() is catalog
    assert len(stub.state.requests) == 2  # throttled: no request within the interval
    assert capsys.readouterr().err == ""  # nothing downloaded, nothing announced
    status = catalog_api.status().as_dict()
    assert status["active"]["catalog_version"] == TEST_CATALOG_VERSION and status["active"]["source"] == "sync"
    assert status["up_to_date"] is True and status["service"]["current"]["catalog_version"] == TEST_CATALOG_VERSION
    # status uses no network: reachable is the last recorded contact, with its time, and says it was not checked now
    service = status["service"]
    assert service["checked_by_this_command"] is False and service["reachable"] is True
    assert service["reachable_as_of"] == service["last_contact_at"] == service["current_as_of"] is not None
    assert service["manifest_url"] == f"{stub.url}/api/v1/catalog" and service["last_known"] is None


def test_a_due_check_is_conditional_and_downloads_only_a_new_release(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    load_public_catalog()
    _age_last_check(online)
    load_public_catalog()
    assert stub.state.requests[-1]["path"] == "/api/v1/catalog" and stub.state.requests[-1]["if_none_match"]
    assert len(stub.state.requests) == 3  # 304: nothing downloaded
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z")
    _age_last_check(online)
    assert load_public_catalog().catalog_version == NEXT_VERSION
    assert stub.state.paths()[-2:] == ["/api/v1/catalog", f"/api/v1/catalog/releases/{NEXT_VERSION}"]
    assert sorted(online.read_state().releases) == [TEST_CATALOG_VERSION, NEXT_VERSION]


def test_automatic_checks_can_be_turned_off(online, stub, monkeypatch):
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "0")
    stub.state.publish(FIXTURE_BYTES)
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE"
    assert "catalog sync" in str(exc.value) and "catalog import" in str(exc.value) and "OPENSCI_CATALOG_CHECK_INTERVAL=0" in str(exc.value)
    assert stub.state.requests == []
    catalog_api.sync()  # an explicit sync still works
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION


def test_gzip_downloads_are_decoded_and_verified(online, stub):
    stub.state.gzip = True
    stub.state.publish(FIXTURE_BYTES)
    report = catalog_api.sync()
    assert stub.state.requests[-1]["accept_encoding"] == "gzip"
    assert report.active.sha256 == catalog_api.store.sha256_hex(FIXTURE_BYTES)
    assert (online.root / "releases" / f"{TEST_CATALOG_VERSION}.json").read_bytes() == FIXTURE_BYTES


def test_a_relative_release_path_and_a_mirror_manifest_url_resolve_like_links(online, stub, monkeypatch):
    stub.state.release_path_template = "releases/{catalog_version}.json"
    stub.state.publish(FIXTURE_BYTES)
    monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url + "/mirror/manifest.json")
    catalog_api.sync()
    assert stub.state.paths() == ["/mirror/manifest.json", f"/mirror/releases/{TEST_CATALOG_VERSION}.json"]


# ---- integrity -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tamper", ["sha256", "size", "bytes"])
def test_bytes_that_differ_from_the_manifest_are_never_used(online, stub, tamper):
    if tamper == "sha256":
        stub.state.publish(FIXTURE_BYTES, sha256="0" * 64)
    elif tamper == "size":
        stub.state.publish(FIXTURE_BYTES, size_bytes=len(FIXTURE_BYTES) - 1)
    else:
        stub.state.publish(FIXTURE_BYTES)
        stub.state.releases[TEST_CATALOG_VERSION] = FIXTURE_BYTES.replace(b"Synthetic", b"Synthetiq", 1)
    with pytest.raises(CatalogError) as exc:
        catalog_api.sync()
    assert exc.value.code is CatalogErrorCode.CATALOG_INTEGRITY_FAILED
    assert online.read_state().releases == {} and not list((online.root / "releases").glob("*.json"))
    with pytest.raises(LocalRunError) as unavailable:  # the automatic first use fails the same way, then says what to do
        load_public_catalog()
    assert unavailable.value.codes[0].value == "CATALOG_NOT_AVAILABLE" and "CATALOG_INTEGRITY_FAILED" in str(unavailable.value)


def test_a_release_label_always_denotes_the_same_bytes(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    altered = _release(TEST_CATALOG_VERSION, edit=lambda d: d["products"][0].update(display_name="changed"))
    stub.state.publish(altered)  # the service now claims different bytes under the same label
    with pytest.raises(CatalogError) as exc:
        catalog_api.sync()
    assert exc.value.code is CatalogErrorCode.CATALOG_RELEASE_CONFLICT
    with pytest.raises(CatalogError) as imported:
        catalog_api.import_release(altered)
    assert imported.value.code is CatalogErrorCode.CATALOG_RELEASE_CONFLICT
    assert CatalogStore(online.root).read_release(online.read_state().active_release) == FIXTURE_BYTES


def test_verify_reports_a_damaged_release_and_sync_repairs_it(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    assert catalog_api.verify(deep=True)["ok"] is True
    path = online.root / "releases" / f"{TEST_CATALOG_VERSION}.json"
    path.write_bytes(FIXTURE_BYTES + b" ")
    report = catalog_api.verify()
    assert report["ok"] is False and report["releases"][0]["problem"]["code"] == "CATALOG_INTEGRITY_FAILED"
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE" and "CATALOG_INTEGRITY_FAILED" in str(exc.value)
    catalog_api.sync()
    assert path.read_bytes() == FIXTURE_BYTES and catalog_api.verify()["ok"] is True


# ---- a lost index: recovery never guesses the active release ------------------------------------------------------------

OLDER, NEWER = "2026-01-05.9", "2026-01-05.10"  # .10 follows .9 (numeric sequence, not text order)


def _lose_index(root: Path, how: str) -> None:
    if how == "corrupt":
        (root / "state.json").write_text("{not json", encoding="utf-8")
    else:
        (root / "state.json").unlink()


def test_release_labels_order_numerically_on_the_sequence_number():
    labels = ["2026-01-05.10", "2026-01-05.9", "2026-01-06.1", "2026-01-05.1", "2025-12-31.20"]
    assert sorted(labels, key=catalog_api.catalog_version_key) == ["2025-12-31.20", "2026-01-05.1", "2026-01-05.9", "2026-01-05.10", "2026-01-06.1"]


@pytest.mark.parametrize("how", ["corrupt", "missing"])
def test_a_lost_index_is_rebuilt_with_no_active_release_and_offline_use_takes_the_newest_stored_one(empty_catalog_store, how, capsys):
    """Which release was active cannot be recovered from the files. Offline, an unpinned use takes the newest stored
    release (.10 follows .9), records how it became active and says so once; a pinned use still gets exactly its
    release; ``catalog use`` selects another."""

    catalog_api.import_release(_release(OLDER))  # the active release
    catalog_api.import_release(_release(NEWER), activate=False)  # newer, stored for a pinned Project
    store = CatalogStore(empty_catalog_store)
    _lose_index(store.root, how)
    state = store.read_state()
    assert state.active is None and state.active_unknown and state.recovered_at is not None and state.active_basis is None
    assert state.stored_versions() == [NEWER, OLDER] and state.newest_readable() == NEWER
    assert json.loads((store.root / "state.json").read_text(encoding="utf-8"))["active"] is None  # the rebuilt index is written
    # before any unpinned use: nothing is active yet, and the status says what the next use will do (a note, not a problem)
    status = catalog_api.status().as_dict()
    assert status["available"] is True and status["active"] is None and status["active_basis"] is None and status["problems"] == []
    assert [r["catalog_version"] for r in status["releases"]] == [NEWER, OLDER]
    (note,) = status["notes"]
    assert f"uses public catalog {NEWER}" in note and f"{NEWER}, {OLDER}" in note and "os-optics catalog use <catalog_version>" in note
    assert catalog_api.verify()["ok"] is True
    code, doctor = _cli("doctor")
    assert doctor["catalog"]["status"] == "NEWEST_STORED_ON_FIRST_USE" and doctor["catalog"]["ready"] is True
    assert doctor["catalog"]["catalog_version"] == NEWER and doctor["catalog"]["stored_releases"] == [NEWER, OLDER]
    # a Project pinned to a stored release runs against exactly that release, and chooses nothing
    assert load_public_catalog(catalog_version=OLDER).catalog_version == OLDER
    assert store.read_state().active is None
    capsys.readouterr()
    assert load_public_catalog().catalog_version == NEWER
    announced = capsys.readouterr().err
    assert f"using {NEWER}, the newest of the stored releases {NEWER}, {OLDER}" in announced and "catalog use" in announced
    state = store.read_state()
    assert (state.active, state.active_basis, state.recovered_at) == (NEWER, "NEWEST_STORED", None)
    assert load_public_catalog().catalog_version == NEWER and capsys.readouterr().err == ""  # announced once
    status = catalog_api.status().as_dict()
    assert status["active"]["catalog_version"] == NEWER and status["active_basis"] == "NEWEST_STORED" and status["notes"] == []
    # the explicit choice
    chosen = catalog_api.use_release(OLDER)
    assert chosen.active.catalog_version == OLDER and chosen.state.active_basis == "CHOSEN" and chosen.as_dict()["problems"] == []
    assert load_public_catalog().catalog_version == OLDER


def test_a_release_imported_without_activation_is_used_when_no_release_is_active(empty_catalog_store, capsys):
    catalog_api.import_release(_release(OLDER), activate=False)
    assert CatalogStore(empty_catalog_store).read_state().active is None
    assert load_public_catalog().catalog_version == OLDER
    assert f"using {OLDER}, the newest (select another" in capsys.readouterr().err  # one stored release: nothing to choose among
    assert CatalogStore(empty_catalog_store).read_state().active_basis == "NEWEST_STORED"


def test_a_store_whose_releases_this_engine_reads_none_of_still_reports_active_unknown(empty_catalog_store):
    """Releases stored, none active, and every one in a catalog schema this Engine does not read: nothing is chosen."""

    catalog_api.import_release(_release(OLDER), activate=False)
    store = CatalogStore(empty_catalog_store)
    index = json.loads((store.root / "state.json").read_text(encoding="utf-8"))
    index["releases"][OLDER]["schema_version"] = "2.0"  # as recorded for a release kept from before an Engine upgrade
    (store.root / "state.json").write_text(json.dumps(index), encoding="utf-8")
    state = store.read_state()
    assert state.active_unknown and state.newest_readable() is None
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE" and "CATALOG_ACTIVE_UNKNOWN" in str(exc.value)
    assert store.read_state().active is None
    status = catalog_api.status().as_dict()
    assert status["available"] is False and [p["code"] for p in status["problems"]] == ["CATALOG_ACTIVE_UNKNOWN"]
    assert catalog_api.verify()["ok"] is False


def test_online_recovery_converges_to_the_services_current_release(online, stub):
    stub.state.publish(_release(NEWER), current=False, published_at="2026-10-01T00:00:00Z")
    stub.state.publish(_release(OLDER), published_at="2026-10-02T00:00:00Z")  # the service's current release is .9
    catalog_api.sync()
    catalog_api.sync(NEWER)  # a newer label, stored for a pinned Project
    assert online.read_state().active == OLDER
    _lose_index(online.root, "corrupt")
    downloads = len([p for p in stub.state.paths() if "/releases/" in p])
    assert load_public_catalog().catalog_version == OLDER  # the automatic check settles it: the service's current release
    state = online.read_state()
    assert state.active == OLDER and state.recovered_at is None and sorted(state.releases) == sorted([OLDER, NEWER])
    assert len([p for p in stub.state.paths() if "/releases/" in p]) == downloads  # both were held: nothing downloaded
    _lose_index(online.root, "missing")
    assert catalog_api.sync().active.catalog_version == OLDER  # an explicit sync converges the same way


def test_offline_recovery_uses_the_newest_stored_release_until_the_service_answers(online, stub, monkeypatch, capsys):
    """The service's current release is the older one; the index is lost and the service is out of reach: the run takes the
    newest stored release and records the failed check. When the service answers again, its current release is active."""

    stub.state.publish(_release(OLDER))
    catalog_api.sync()
    catalog_api.import_release(_release(NEWER), activate=False)
    _lose_index(online.root, "corrupt")
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{_free_port()}")
    capsys.readouterr()
    assert load_public_catalog().catalog_version == NEWER
    assert f"using {NEWER}" in capsys.readouterr().err
    state = online.read_state()
    assert (state.active, state.active_basis) == (NEWER, "NEWEST_STORED")
    assert state.last_check_error["code"] == "CATALOG_SERVICE_UNREACHABLE"
    monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url)
    assert catalog_api.sync().active.catalog_version == OLDER
    state = online.read_state()
    assert (state.active, state.active_basis) == (OLDER, "SERVICE_CURRENT")
    assert load_public_catalog().catalog_version == OLDER


def test_use_refuses_a_release_that_is_not_stored(empty_catalog_store):
    catalog_api.import_release(FIXTURE_BYTES)
    with pytest.raises(CatalogError) as missing:
        catalog_api.use_release(NEXT_VERSION)
    assert missing.value.code is CatalogErrorCode.CATALOG_NOT_AVAILABLE and f"catalog sync --release {NEXT_VERSION}" in missing.value.message
    with pytest.raises(CatalogError) as label:
        catalog_api.use_release("latest")
    assert label.value.code is CatalogErrorCode.CATALOG_INVALID
    (empty_catalog_store / "releases" / f"{TEST_CATALOG_VERSION}.json").write_bytes(FIXTURE_BYTES + b" ")
    with pytest.raises(CatalogError) as damaged:
        catalog_api.use_release(TEST_CATALOG_VERSION)
    assert damaged.value.code is CatalogErrorCode.CATALOG_INTEGRITY_FAILED


# ---- compatibility and offline -------------------------------------------------------------------------------------------


def test_a_newer_catalog_schema_keeps_the_stored_release_and_reports_an_engine_upgrade(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z", schema_version="4.0")
    report = catalog_api.sync()
    assert report.upgrade_required is True and report.active.catalog_version == TEST_CATALOG_VERSION
    assert [p["code"] for p in report.as_dict()["problems"]] == ["ENGINE_UPGRADE_REQUIRED"]
    assert f"/api/v1/catalog/releases/{NEXT_VERSION}" not in stub.state.paths()  # never downloaded
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    checked = catalog_api.check().as_dict()
    assert checked["upgrade_required"] is True and checked["up_to_date"] is False and checked["service"]["reachable"] is True


def test_another_service_s_upgrade_answer_is_only_last_known(online, stub, monkeypatch):
    """Review: service A's current release needs a newer engine; with service B configured
    (not yet answered, or unreachable) neither status nor check asserts A's upgrade, up-to-dateness or current release
    for B; A's answer stays under last_known with its URL and time."""

    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z", schema_version="4.0")
    assert catalog_api.sync().upgrade_required is True  # service A, confirmed
    other = f"http://127.0.0.1:{_free_port()}"
    monkeypatch.setenv("OPENSCI_CATALOG_URL", other)
    for report in (catalog_api.status(), catalog_api.check()):  # B not yet asked; B asked and unreachable
        document = report.as_dict()
        assert report.upgrade_required is False and document["upgrade_required"] is False
        assert "ENGINE_UPGRADE_REQUIRED" not in [p["code"] for p in document["problems"]]
        assert document["up_to_date"] is None and document["service"]["current"] is None
        assert document["service"]["manifest_url"] == f"{other}/api/v1/catalog"
        last_known = document["service"]["last_known"]
        assert last_known["manifest_url"] == f"{stub.url}/api/v1/catalog" and last_known["as_of"] is not None
        assert last_known["current"]["schema_version"] == "4.0"
    assert catalog_api.check().as_dict()["service"]["reachable"] is False
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION  # the stored release stays in use
    monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url)  # back to A: its confirmed answer is current again
    assert catalog_api.status().upgrade_required is True


def test_an_empty_store_takes_the_newest_compatible_retained_release(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z", schema_version="4.0")
    report = catalog_api.sync()
    assert report.active.catalog_version == TEST_CATALOG_VERSION and report.upgrade_required is True
    stub.state.entries = [e for e in stub.state.entries if e["catalog_version"] == NEXT_VERSION]
    online_empty = CatalogStore(online.root / "other")
    with pytest.raises(CatalogError) as exc:
        catalog_api.sync(environ={"OPENSCI_CATALOG_DIR": str(online_empty.root), "OPENSCI_CATALOG_URL": stub.url})
    assert exc.value.code is CatalogErrorCode.ENGINE_UPGRADE_REQUIRED


def _publish_design_material(document: dict) -> None:
    """Catalog schema 3.6: the release also publishes a design material (a material no product names)."""

    design = dict(json.loads(json.dumps(document["materials"][0])), material_id="TEST-DESIGN-GLASS", display_name="TEST-DESIGN-GLASS")
    document["materials"].append(design)
    document["schema_version"] = "3.6"


DESIGN_MATERIAL_BYTES = _release(NEXT_VERSION, edit=_publish_design_material)


def test_a_catalog_schema_3_6_release_is_read_by_this_engine(online, stub):
    """A release that publishes design materials states catalog schema 3.6; this os-optics reads it."""

    assert "3.6" in catalog_api.SUPPORTED_SCHEMA_VERSIONS
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    stub.state.publish(DESIGN_MATERIAL_BYTES, published_at="2026-10-02T00:00:00Z")
    report = catalog_api.sync()
    assert report.upgrade_required is False and report.active.catalog_version == NEXT_VERSION
    catalog = load_public_catalog()
    assert catalog.catalog_version == NEXT_VERSION and catalog.schema_version == "3.6"
    assert "TEST-DESIGN-GLASS" in {m.material_id for m in catalog.materials}


def test_a_reader_of_schema_3_5_never_downloads_a_3_6_release(online, stub, monkeypatch):
    """An os-optics whose supported list ends at 3.5 (0.13.0) sees 3.6 in the manifest, reports an engine upgrade before
    downloading and keeps its stored release; an empty store takes the newest retained release it reads."""

    from os_optics.catalog import service as catalog_service
    from os_optics.catalog import store as catalog_store

    through_3_5 = ("3.0", "3.1", "3.2", "3.3", "3.4", "3.5")
    monkeypatch.setattr(catalog_store, "SUPPORTED_SCHEMA_VERSIONS", through_3_5)
    monkeypatch.setattr(catalog_service, "SUPPORTED_SCHEMA_VERSIONS", through_3_5)
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    stub.state.publish(DESIGN_MATERIAL_BYTES, published_at="2026-10-02T00:00:00Z")
    report = catalog_api.sync()
    assert report.upgrade_required is True and report.active.catalog_version == TEST_CATALOG_VERSION
    assert [p["code"] for p in report.as_dict()["problems"]] == ["ENGINE_UPGRADE_REQUIRED"]
    assert f"/api/v1/catalog/releases/{NEXT_VERSION}" not in stub.state.paths()  # never downloaded
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    with pytest.raises(CatalogError) as pinned:
        catalog_api.sync(NEXT_VERSION)
    assert pinned.value.code is CatalogErrorCode.ENGINE_UPGRADE_REQUIRED
    assert f"/api/v1/catalog/releases/{NEXT_VERSION}" not in stub.state.paths()
    online_empty = CatalogStore(online.root / "other")
    fresh = catalog_api.sync(environ={"OPENSCI_CATALOG_DIR": str(online_empty.root), "OPENSCI_CATALOG_URL": stub.url})
    assert fresh.active.catalog_version == TEST_CATALOG_VERSION and fresh.upgrade_required is True
    assert f"/api/v1/catalog/releases/{NEXT_VERSION}" not in stub.state.paths()


def test_offline_the_stored_release_stays_in_use_silently(online, stub, monkeypatch):
    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    _age_last_check(online)
    offline = f"http://127.0.0.1:{_free_port()}"
    monkeypatch.setenv("OPENSCI_CATALOG_URL", offline)
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    state = online.read_state()
    assert state.last_check_error["code"] == "CATALOG_SERVICE_UNREACHABLE"
    checked = catalog_api.check()
    assert checked.reachable is False
    # an unreachable service is never reported as up to date; the URL is the one actually checked, and what the
    # previous service said is kept, labelled with its time
    document = checked.as_dict()
    assert document["up_to_date"] is None and document["service"]["current"] is None
    assert document["service"]["manifest_url"] == f"{offline}/api/v1/catalog" and document["service"]["reachable"] is False
    last_known = document["service"]["last_known"]
    assert last_known["manifest_url"] == f"{stub.url}/api/v1/catalog" and last_known["current"]["catalog_version"] == TEST_CATALOG_VERSION
    assert last_known["as_of"] is not None
    status = catalog_api.status().as_dict()  # no network: the failed attempt is what it reports
    assert status["service"]["reachable"] is False and status["up_to_date"] is None and status["service"]["current"] is None
    with pytest.raises(CatalogError) as exc:
        catalog_api.sync()
    assert exc.value.code is CatalogErrorCode.CATALOG_SERVICE_UNREACHABLE


def test_an_empty_store_tries_a_newly_configured_service_at_once(online, stub, monkeypatch, capsys):
    """The first use failed against an unreachable URL; with the real service configured, the very next use
    downloads (the one-minute retry pause only spares the same unreachable service)."""

    stub.state.publish(FIXTURE_BYTES)
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{_free_port()}")
    with pytest.raises(LocalRunError):
        load_public_catalog()
    with pytest.raises(LocalRunError):  # the same unreachable service: not retried within the minute
        load_public_catalog()
    monkeypatch.setenv("OPENSCI_CATALOG_URL", stub.url)
    capsys.readouterr()
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    assert "downloaded public optical catalog" in capsys.readouterr().err


def test_status_never_lends_another_service_s_reachability(online, stub, monkeypatch):
    """After a sync with one service, pointing at another (here one that is down) must not
    report it reachable from the first one's record before anything was asked."""

    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    assert catalog_api.status().as_dict()["service"]["reachable"] is True
    offline = f"http://127.0.0.1:{_free_port()}"
    monkeypatch.setenv("OPENSCI_CATALOG_URL", offline)
    service = catalog_api.status().as_dict()["service"]
    assert service["manifest_url"] == f"{offline}/api/v1/catalog"
    assert service["reachable"] is None and service["reachable_as_of"] is None and service["current"] is None
    assert service["last_known"]["manifest_url"] == f"{stub.url}/api/v1/catalog"


def test_an_empty_store_offline_names_the_ways_to_get_a_catalog(online, monkeypatch):
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{_free_port()}")
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    message = str(exc.value)
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE"
    assert "os-optics catalog sync" in message and "os-optics catalog import" in message and "CATALOG_SERVICE_UNREACHABLE" in message


def test_the_service_without_a_catalog_endpoint_is_a_service_error(online, stub):
    stub.state.manifest_status = 404
    with pytest.raises(CatalogError) as exc:
        catalog_api.sync()
    assert exc.value.code is CatalogErrorCode.CATALOG_SERVICE_ERROR and exc.value.details["http_status"] == 404


# ---- pinned releases -----------------------------------------------------------------------------------------------------


def _pinned_project(version: str) -> dict:
    from os_optics.local import example_project_of_kind

    document = example_project_of_kind("focus", catalog=FIXTURE_CATALOG).payload()
    for component in document["components"]:
        if "product" in component:
            component["product"]["catalog_version"] = version
    return document


def test_running_a_pinned_project_uses_the_pinned_release(online, stub):
    stub.state.publish(FIXTURE_BYTES, current=False, published_at="2026-10-01T00:00:00Z")
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z")
    catalog_api.sync()
    assert online.read_state().active == NEXT_VERSION and TEST_CATALOG_VERSION not in online.read_state().releases
    run = run_optical_project(_pinned_project(TEST_CATALOG_VERSION))
    assert run.catalog_version == TEST_CATALOG_VERSION
    assert f"/api/v1/catalog/releases/{TEST_CATALOG_VERSION}" in stub.state.paths()
    assert online.read_state().active == NEXT_VERSION  # fetching a pinned release does not change the active one
    assert run_optical_project(_pinned_project(NEXT_VERSION)).catalog_version == NEXT_VERSION


def _pinned_check(version: str, tmp_path: Path) -> None:
    """``os-optics project check`` of a Project pinned to ``version`` (the CLI / json_api path)."""

    path = tmp_path / f"pinned-{version}.json"
    path.write_text(json.dumps(_pinned_project(version)), encoding="utf-8")
    code, checked = _cli("project", "check", str(path))
    assert code == 0 and checked["catalog_version"] == version


def _pinned_run(version: str, tmp_path: Path) -> None:
    """``run_optical_project`` of a Project pinned to ``version`` (the Python API path)."""

    assert run_optical_project(_pinned_project(version)).catalog_version == version


def _publish_both(stub) -> None:
    stub.state.publish(FIXTURE_BYTES, current=False, published_at="2026-10-01T00:00:00Z")
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z")


def _ready(capsys) -> tuple[dict, dict]:
    """``catalog status`` and ``doctor`` after the command under test: both exit 0 with the catalog OK."""

    capsys.readouterr()
    code, status = _cli("catalog", "status")
    assert code == 0 and status["available"] is True and status["problems"] == []
    doctor_code = cli_main(["doctor"])
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["catalog"]["status"] == "OK" and doctor["ready"]["local_design"] is True and doctor_code == 0
    assert doctor["ready"]["catalog"] is True
    return status, doctor


@pytest.mark.parametrize("entry", [_pinned_check, _pinned_run], ids=["cli-project-check", "python-run"])
def test_a_first_pinned_fetch_into_an_empty_store_becomes_the_active_release(online, stub, tmp_path, capsys, entry):
    """The first command on a fresh store is a pinned Project; afterwards ``catalog status`` and ``doctor`` are
    ready, as after first-use synchronization."""

    _publish_both(stub)
    entry(TEST_CATALOG_VERSION, tmp_path)
    notice = capsys.readouterr().err
    assert f"downloaded public optical catalog {TEST_CATALOG_VERSION}" in notice and "it is the active release now" in notice
    state = online.read_state()
    assert state.active == TEST_CATALOG_VERSION and sorted(state.releases) == [TEST_CATALOG_VERSION]
    status, _ = _ready(capsys)
    assert status["active"]["catalog_version"] == TEST_CATALOG_VERSION
    assert status["up_to_date"] is False and status["service"]["current"]["catalog_version"] == NEXT_VERSION
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION  # within the interval: the stored active release


@pytest.mark.parametrize("entry", [_pinned_check, _pinned_run], ids=["cli-project-check", "python-run"])
def test_a_pinned_fetch_never_changes_an_active_release(online, stub, tmp_path, capsys, entry):
    _publish_both(stub)
    catalog_api.sync()
    assert online.read_state().active == NEXT_VERSION
    capsys.readouterr()
    entry(TEST_CATALOG_VERSION, tmp_path)
    assert "the active release is unchanged" in capsys.readouterr().err
    state = online.read_state()
    assert state.active == NEXT_VERSION and sorted(state.releases) == [TEST_CATALOG_VERSION, NEXT_VERSION]
    _ready(capsys)


def test_only_the_first_pinned_fetch_into_an_empty_store_is_activated(online, stub, tmp_path, capsys):
    """Pinned A into an empty store -> A active; a later pinned B is stored, A stays active."""

    _publish_both(stub)
    _pinned_run(NEXT_VERSION, tmp_path)
    assert online.read_state().active == NEXT_VERSION
    capsys.readouterr()
    _pinned_check(TEST_CATALOG_VERSION, tmp_path)
    assert "the active release is unchanged" in capsys.readouterr().err
    state = online.read_state()
    assert state.active == NEXT_VERSION and sorted(state.releases) == [TEST_CATALOG_VERSION, NEXT_VERSION]
    _ready(capsys)


@pytest.mark.parametrize("how", ["imported-without-activation", "index-lost"])
def test_a_pinned_fetch_does_not_choose_among_stored_releases_with_none_active(online, stub, tmp_path, capsys, how):
    """A store that already holds releases but none active is not empty: a pinned fetch chooses no active release. The
    status says what the next unpinned use will take (a note, not a problem)."""

    _publish_both(stub)
    if how == "index-lost":
        catalog_api.sync()
        _lose_index(online.root, "missing")
    else:
        catalog_api.import_release(NEXT_BYTES, activate=False)
    assert online.read_state().active_unknown
    capsys.readouterr()
    _pinned_run(TEST_CATALOG_VERSION, tmp_path)
    assert "the active release is unchanged" in capsys.readouterr().err
    state = online.read_state()
    assert state.active is None and sorted(state.releases) == [TEST_CATALOG_VERSION, NEXT_VERSION]
    code, status = _cli("catalog", "status")
    assert code == 0 and status["problems"] == [] and status["active"] is None and status["available"] is True
    assert f"uses public catalog {NEXT_VERSION}" in status["notes"][0]


def test_a_pinned_release_that_cannot_be_obtained_is_not_available(online, stub, monkeypatch):
    stub.state.publish(NEXT_BYTES)
    catalog_api.sync()
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(_pinned_project(TEST_CATALOG_VERSION))
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE"
    message = str(exc.value)
    # the service said it does not distribute that release, so a fetch is never suggested; the Project moves
    assert TEST_CATALOG_VERSION in message and "no longer distributes it" in message
    assert "catalog sync --release" not in message
    # (update-catalog moves the components of a release that can never be compared and says what to do: no flag is advised)
    assert "project update-catalog PROJECT --out PROJECT" in message and "next_steps" in message
    assert "--include-changed" not in message and "review the new Local Validation" not in message
    with pytest.raises(CatalogError) as missing:
        catalog_api.sync(TEST_CATALOG_VERSION)
    assert missing.value.code is CatalogErrorCode.CATALOG_RELEASE_NOT_FOUND
    with pytest.raises(CatalogError) as detailed:
        catalog_api.release_bytes(TEST_CATALOG_VERSION)
    assert detailed.value.details["last_attempt"] == "CATALOG_RELEASE_NOT_FOUND"
    assert detailed.value.details["next_steps"][0].startswith("os-optics project update-catalog")
    # offline, the release may still be obtainable later: then fetching it is the first suggestion
    monkeypatch.setenv("OPENSCI_CATALOG_URL", f"http://127.0.0.1:{_free_port()}")
    with pytest.raises(LocalRunError) as offline:
        run_optical_project(_pinned_project(TEST_CATALOG_VERSION))
    assert f"catalog sync --release {TEST_CATALOG_VERSION}" in str(offline.value) and "update-catalog" in str(offline.value)


def test_a_project_of_an_older_engine_moves_to_the_current_release(online, stub, tmp_path):
    """The release a 0.3.x Project pins is not distributed. Nothing can ever compare its products with the current
    ones, so update-catalog moves the Project in one step -- never UPDATED: the components are listed as unverified with
    the reason, and the next step is to run again and review."""

    stub.state.publish(NEXT_BYTES)
    project = tmp_path / "project.json"
    project.write_text(json.dumps(_pinned_project(TEST_CATALOG_VERSION)), encoding="utf-8")
    code, report = _cli("project", "update-catalog", str(project), "--out", str(project))
    assert code == 0 and report["status"] == "UPDATED_UNVERIFIED" and report["project_written"] is True
    assert report["unverified"] and report["unverified"] == report["repinned"] and report["changed"] == []
    assert report["old_releases"] == [{"catalog_version": TEST_CATALOG_VERSION, "reason": "NOT_DISTRIBUTED"}]
    assert [s["action"] for s in report["next_steps"]] == ["RERUN_AND_REVIEW", "SUBMIT"]
    assert "no longer distributes it" in report["next_steps"][0]["message"] and "see next_steps" in report["summary"]
    assert run_optical_project(json.loads(project.read_text(encoding="utf-8"))).catalog_version == NEXT_VERSION


def test_sync_of_a_named_release_activates_it_only_when_none_is_active(online, stub, capsys):
    """After the store was removed, the advised 'catalog sync --release X' leaves a usable active catalog."""

    stub.state.publish(FIXTURE_BYTES, current=False, published_at="2026-10-01T00:00:00Z")
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z")
    code, synced = _cli("catalog", "sync", "--release", TEST_CATALOG_VERSION)
    assert code == 0 and synced["active"]["catalog_version"] == TEST_CATALOG_VERSION
    assert synced["notes"] == [f"public catalog {TEST_CATALOG_VERSION} is the active release now (no release was active)"]
    assert synced["service"]["reachable"] is True and synced["problems"] == []
    code, current = _cli("catalog", "sync")  # the plain sync still moves to the service's current release
    assert code == 0 and current["active"]["catalog_version"] == NEXT_VERSION
    code, again = _cli("catalog", "sync", "--release", TEST_CATALOG_VERSION)  # stored: no request, the active stays
    assert code == 0 and again["active"]["catalog_version"] == NEXT_VERSION and again["service"]["checked_by_this_command"] is False
    assert f"'os-optics catalog use {TEST_CATALOG_VERSION}'" in again["notes"][0]
    assert capsys.readouterr().err == ""  # explicit commands print their JSON only


def test_a_removed_store_recovers_with_the_advised_commands(online, stub):
    import shutil

    stub.state.publish(FIXTURE_BYTES)
    catalog_api.sync()
    shutil.rmtree(online.root)
    code, refused = _cli("catalog", "use", TEST_CATALOG_VERSION)
    assert code == 1 and f"catalog sync --release {TEST_CATALOG_VERSION}' (it becomes the active release)" in refused["error"]["message"]
    code, synced = _cli("catalog", "sync", "--release", TEST_CATALOG_VERSION)
    assert code == 0 and synced["active"]["catalog_version"] == TEST_CATALOG_VERSION
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION


# ---- origin, import, browser runtime -------------------------------------------------------------------------------------


def test_the_service_origin_resolution_order():
    assert catalog_api.manifest_url({}) == default_service_origin() + "/api/v1/catalog"
    assert catalog_api.manifest_url({"OPENSCI_API_URL": "https://api.example.test/api/v1/"}) == "https://api.example.test/api/v1/catalog"
    assert catalog_api.manifest_url({"OPENSCI_API_URL": "https://a.example.test", "OPENSCI_CATALOG_URL": "https://b.example.test"}) \
        == "https://b.example.test/api/v1/catalog"
    assert catalog_api.manifest_url({"OPENSCI_CATALOG_URL": "http://localhost:8080/x/manifest.json"}) == "http://localhost:8080/x/manifest.json"
    for bad in ("http://catalog.example.test", "ftp://x", "https://user:pw@x.example.test", "https://x.example.test/?q=1"):
        with pytest.raises(CatalogError) as exc:
            catalog_api.manifest_url({"OPENSCI_CATALOG_URL": bad})
        assert exc.value.code is CatalogErrorCode.CATALOG_ORIGIN_INVALID


def test_import_makes_a_file_the_active_release_for_an_offline_device(empty_catalog_store):
    report = catalog_api.import_release(FIXTURE_CATALOG)
    assert report.active.catalog_version == TEST_CATALOG_VERSION and report.active.source == "import"
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    catalog_api.import_release(NEXT_BYTES, activate=False)
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    with pytest.raises(CatalogError) as exc:
        catalog_api.import_release(b'{"schema": "something else"}')
    assert exc.value.code is CatalogErrorCode.CATALOG_INVALID
    newer = _release("2026-01-03.1", edit=lambda d: d.update(schema_version="4.0"))
    with pytest.raises(CatalogError) as upgrade:
        catalog_api.import_release(newer)
    assert upgrade.value.code is CatalogErrorCode.ENGINE_UPGRADE_REQUIRED


def test_the_browser_runtime_never_uses_the_network(online, stub, monkeypatch):
    stub.state.publish(FIXTURE_BYTES)
    monkeypatch.setattr(sys, "platform", "emscripten")
    with pytest.raises(LocalRunError) as exc:
        load_public_catalog()
    assert exc.value.codes[0].value == "CATALOG_NOT_AVAILABLE" and "install_catalog" in str(exc.value)
    with pytest.raises(CatalogError) as sync_refused:
        catalog_api.sync()
    assert sync_refused.value.code is CatalogErrorCode.CATALOG_SYNC_UNAVAILABLE
    assert stub.state.requests == []


# ---- command line --------------------------------------------------------------------------------------------------------


def _cli(*argv: str) -> tuple[int, dict]:
    """``os-optics ARGV``: exit code and the JSON document on stdout."""

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(list(argv))
    return code, json.loads(out.getvalue())


def test_catalog_commands(online, stub, tmp_path):
    code, status = _cli("catalog", "status")
    assert code == 1 and status["available"] is False and status["automatic_check_interval_s"] == 3600
    stub.state.publish(FIXTURE_BYTES)
    code, checked = _cli("catalog", "check")
    assert code == 0 and checked["service"]["current"]["catalog_version"] == TEST_CATALOG_VERSION and checked["up_to_date"] is False
    code, synced = _cli("catalog", "sync")
    assert code == 0 and synced["active"]["catalog_version"] == TEST_CATALOG_VERSION and synced["up_to_date"] is True
    code, verified = _cli("catalog", "verify", "--deep")
    assert code == 0 and verified["ok"] is True
    release = tmp_path / "next.json"
    release.write_bytes(NEXT_BYTES)
    code, imported = _cli("catalog", "import", str(release))
    assert code == 0 and imported["active"]["catalog_version"] == NEXT_VERSION
    code, listed = _cli("catalog", "list", "--limit", "1")
    assert code == 0 and listed["catalog_version"] == NEXT_VERSION
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    code, refused = _cli("catalog", "import", str(bad))
    assert code == 65 and refused["error"]["code"] == "CATALOG_INVALID"
    stub.state.manifest_status = 503
    code, failed = _cli("catalog", "sync")
    assert code == 1 and failed["error"]["code"] == "CATALOG_SERVICE_ERROR"
    code, used = _cli("catalog", "use", TEST_CATALOG_VERSION)
    assert code == 0 and used["active"]["catalog_version"] == TEST_CATALOG_VERSION
    code, refused_use = _cli("catalog", "use", "2026-01-09.1")
    assert code == 1 and refused_use["error"]["code"] == "CATALOG_NOT_AVAILABLE"


# ---- releases in an earlier catalog schema (kept from before an Engine upgrade) -------------------------------------------


def _mark_schema(store: CatalogStore, version: str, schema_version: str) -> None:
    state_file = store.root / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["releases"][version]["schema_version"] = schema_version
    state_file.write_text(json.dumps(state), encoding="utf-8")


def test_an_active_release_in_an_earlier_schema_is_replaced_at_once_not_after_the_interval(online, stub):
    stub.state.publish(FIXTURE_BYTES)
    assert load_public_catalog().catalog_version == TEST_CATALOG_VERSION
    _mark_schema(online, TEST_CATALOG_VERSION, "2.1")  # checked just now, but unreadable by this Engine
    stub.state.publish(NEXT_BYTES, published_at="2026-10-02T00:00:00Z")
    assert load_public_catalog().catalog_version == NEXT_VERSION
    assert online.read_state().active == NEXT_VERSION


def test_a_pinned_release_the_service_has_only_in_an_earlier_schema_moves_the_project(online, stub):
    from local_helpers import PCX_F100, element, plane, project, source
    from os_optics.local import update_project_catalog

    old_version = "2025-12-01.1"
    stub.state.publish(_release(old_version), published_at="2025-12-01T00:00:00Z", current=False, schema_version="2.0")
    stub.state.publish(FIXTURE_BYTES)
    pinned = project([source(), element("L1", "lens", PCX_F100, catalog_version=old_version)], [plane("P", (150.0, 0.0, 0.0))])
    with pytest.raises(LocalRunError) as exc:
        run_optical_project(pinned)
    message = str(exc.value)
    assert [c.value for c in exc.value.codes] == ["CATALOG_NOT_AVAILABLE"]
    assert "earlier format" in message and "update-catalog" in message and "sync --release" not in message
    report = update_project_catalog(pinned).as_dict()
    assert report["old_releases"] == [{"catalog_version": old_version, "reason": "NOT_READABLE"}]
    # (it can never be compared: the Project is moved in this one step and the component stays listed as unverified)
    assert report["status"] == "UPDATED_UNVERIFIED" and [r["instance_id"] for r in report["unverified"]] == ["L1"]
    step, submit = report["next_steps"]
    assert (step["action"], submit["action"]) == ("RERUN_AND_REVIEW", "SUBMIT")
    assert "written in an earlier catalog format" in step["message"] and "reachable" not in step["message"]


def test_a_stored_release_in_a_newer_schema_is_not_called_an_earlier_format(empty_catalog_store):
    from os_optics.local.catalog import exact_catalog_lookup

    catalog_api.import_release(FIXTURE_BYTES)
    _mark_schema(CatalogStore(empty_catalog_store), TEST_CATALOG_VERSION, "4.0")
    assert exact_catalog_lookup(TEST_CATALOG_VERSION) == (None, "ENGINE_UPGRADE_REQUIRED")
