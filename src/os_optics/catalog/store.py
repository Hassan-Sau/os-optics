"""The local catalog store: verified public optical catalog releases kept on this device.

Layout (``$OPENSCI_CATALOG_DIR``, default ``~/.opensci/catalog/``)::

    releases/<catalog_version>.json   the exact bytes of one release (as published; verified by SHA-256)
    state.json                        the index: active release, every stored release's SHA-256 / size / schema
                                      version / source / the catalog contract that validated its whole document
                                      (``validated_by``), and the last check against the catalog service
    store.lock                        the lock file of concurrent store operations (``locking``)

A ``catalog_version`` always denotes the same bytes: a release is written once, after its SHA-256 (and, for a download,
its published size) and its content were verified, and is never replaced by different bytes. Files are written to a
temporary name unique to the writer and renamed into place, so an interrupted write never leaves a partial release
under its real name.

Concurrency: several processes and threads may use one store at once. Every change of the index is a read-modify-write
of the *current* index under the store's lock (``CatalogStore.update_state``), never a write of an index read earlier,
so concurrent operations never lose a release another one stored, and a release file is written under the same lock.

Recovery: an index that is missing or unreadable while release files exist is rebuilt from the files (each re-verified
as a readable release). Which release was active cannot be recovered from the files -- a release stored for a pinned
Project or imported without activation may be newer than the active one -- so a rebuilt index has **no** active
release until one is chosen: the next catalog check against the service selects the service's current release, and
where no check settles it an unpinned use takes the newest stored release this Engine reads and says so
(``active_basis`` ``NEWEST_STORED``; ``catalog use`` selects another). Only a store none of whose releases this Engine
reads fails with ``CATALOG_ACTIVE_UNKNOWN``. Release labels ``YYYY-MM-DD.N`` are ordered by date, then numerically by ``N`` (``catalog_version_key``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .._vendor.shared_contracts.design_catalog import PUBLIC_OPTICAL_CATALOG_SCHEMA_ID, PUBLIC_OPTICAL_CATALOG_SCHEMA_VERSIONS, PublicOpticalCatalogV2
from .errors import CatalogError, CatalogErrorCode
from .locking import LOCK_TIMEOUT_SECONDS, StoreLock, temporary_path

ENV_CATALOG_DIR = "OPENSCI_CATALOG_DIR"
DEFAULT_STORE_DIR = Path(".opensci") / "catalog"
STATE_FILE = "state.json"
RELEASES_DIR = "releases"
STATE_SCHEMA = "opensci.engine.catalog_store"
STATE_VERSION = "1"
#: The public optical catalog schema versions this Engine reads (``PublicOpticalCatalogV2``): 3.0, 3.1, 3.2 and 3.3.
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = PUBLIC_OPTICAL_CATALOG_SCHEMA_VERSIONS
SOURCE_SYNC = "sync"
SOURCE_IMPORT = "import"
CATALOG_VERSION_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*")
MAX_REPORTED_PROBLEMS = 5
#: How long a writer retries renaming its temporary file onto a release file another process is reading (Windows refuses
#: to replace an open file); store operations hold the store lock meanwhile.
REPLACE_RETRY_SECONDS = 2.0
REPLACE_RETRY_PAUSE_SECONDS = 0.02


#: How a release became the active one (``StoreState.active_basis``): ``CHOSEN`` (``catalog use``, an import, a download of
#: a named release into a store with none active), ``SERVICE_CURRENT`` (a check made the catalog service's current release
#: active) and ``NEWEST_STORED`` (releases were stored with none active, and an unpinned use took the newest one this
#: Engine reads; the next successful check replaces it with the service's current release).
ACTIVE_BASES = ("CHOSEN", "SERVICE_CURRENT", "NEWEST_STORED")


def catalog_version_key(catalog_version: str) -> tuple[str, int]:
    """The order of release labels ``YYYY-MM-DD.N``: by date, then numerically by ``N`` (``2026-10-05.10`` follows
    ``2026-10-05.9``)."""

    date, _, sequence = catalog_version.partition(".")
    return date, int(sequence)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_text(epoch_seconds: float | None) -> str | None:
    """``YYYY-MM-DDTHH:MM:SSZ`` of a POSIX time (None stays None)."""

    return None if epoch_seconds is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


@dataclass(frozen=True)
class StoredRelease:
    """One release held in the store."""

    catalog_version: str
    sha256: str
    size_bytes: int
    schema_version: str
    source: str
    added_at: float
    #: The identity of the catalog contract that validated the whole document (``contract_identity()`` of the Engine that
    #: did); ``None`` in an index written before it was recorded. A reader whose own identity equals it reads the release
    #: without validating the whole document again (``os_optics.local.catalog``).
    validated_by: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"catalog_version": self.catalog_version, "schema_version": self.schema_version, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "source": self.source, "added_at": utc_text(self.added_at)}

    def record(self) -> dict[str, Any]:
        record = {"sha256": self.sha256, "size_bytes": self.size_bytes, "schema_version": self.schema_version,
                  "source": self.source, "added_at": self.added_at}
        if self.validated_by is not None:
            record["validated_by"] = self.validated_by
        return record


@dataclass(frozen=True)
class StoreState:
    """The store index (``state.json``)."""

    active: str | None = None
    releases: dict[str, StoredRelease] = field(default_factory=dict)
    manifest_url: str | None = None
    manifest_etag: str | None = None
    release_path_template: str | None = None
    server_current: dict[str, Any] | None = None
    #: When the catalog service was last asked (successfully or not; paces the automatic checks).
    checked_at: float | None = None
    last_check_error: dict[str, Any] | None = None
    #: When the catalog service last answered (``server_current`` is what it said then).
    contacted_at: float | None = None
    #: When the index was last rebuilt from the release files with no active release (None once one is chosen).
    recovered_at: float | None = None
    #: How the active release became active (``ACTIVE_BASES``; None: no active release, or an index written before this
    #: was recorded).
    active_basis: str | None = None

    @property
    def active_release(self) -> StoredRelease | None:
        return self.releases.get(self.active) if self.active is not None else None

    @property
    def usable_active(self) -> bool:
        """An active release this Engine can read: one in an earlier catalog schema (kept from before an Engine
        upgrade) counts as no active release, so the automatic check replaces it at once."""

        active = self.active_release
        return active is not None and not is_retired_schema(active.schema_version)

    @property
    def active_unknown(self) -> bool:
        """Releases are stored but none is the active one (until a check, ``catalog use`` or an import chooses one)."""

        return self.active is None and bool(self.releases)

    def stored_versions(self) -> list[str]:
        """The stored release labels, newest first (``catalog_version_key``)."""

        return sorted(self.releases, key=catalog_version_key, reverse=True)

    def activated(self, catalog_version: str, basis: str = "CHOSEN") -> StoreState:
        """This index with the stored release ``catalog_version`` as the active one, and how it became active
        (``ACTIVE_BASES``)."""

        return replace(self, active=catalog_version, recovered_at=None, active_basis=basis)

    def newest_readable(self) -> str | None:
        """The newest stored release this Engine reads (``catalog_version_key``), or ``None`` when it reads none of them."""

        return next((version for version in self.stored_versions()
                     if self.releases[version].schema_version in SUPPORTED_SCHEMA_VERSIONS), None)

    def document(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "schema_version": STATE_VERSION,
            "active": self.active,
            "releases": {version: self.releases[version].record() for version in sorted(self.releases, key=catalog_version_key)},
            "manifest_url": self.manifest_url,
            "manifest_etag": self.manifest_etag,
            "release_path_template": self.release_path_template,
            "server_current": self.server_current,
            "checked_at": self.checked_at,
            "last_check_error": self.last_check_error,
            "contacted_at": self.contacted_at,
            "recovered_at": self.recovered_at,
            "active_basis": self.active_basis,
        }


def store_directory(environ: Any = None) -> Path:
    """``$OPENSCI_CATALOG_DIR``, else ``~/.opensci/catalog``."""

    env = os.environ if environ is None else environ
    named = str(env.get(ENV_CATALOG_DIR, "") or "").strip()
    if named:
        return Path(named).expanduser()
    try:
        return Path.home() / DEFAULT_STORE_DIR
    except RuntimeError:  # no home directory (a sandbox): the working directory, never a guess elsewhere
        return Path.cwd() / DEFAULT_STORE_DIR


def release_header(data: bytes) -> dict[str, Any]:
    """The parsed JSON document of a release; ``CATALOG_INVALID`` when it is not a JSON object of the public catalog."""

    try:
        document = json.loads(data)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise CatalogError(CatalogErrorCode.CATALOG_INVALID, f"the catalog document is not valid JSON: {exc}") from None
    if not isinstance(document, dict) or document.get("schema") != PUBLIC_OPTICAL_CATALOG_SCHEMA_ID:
        raise CatalogError(CatalogErrorCode.CATALOG_INVALID, f"not a public optical catalog document (schema {PUBLIC_OPTICAL_CATALOG_SCHEMA_ID} expected)")
    return document


def _schema_key(schema_version: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in str(schema_version).split("."))
    except ValueError:
        return None


def is_retired_schema(schema_version: Any) -> bool:
    """Whether ``schema_version`` is an earlier catalog schema than every one this Engine reads (a release kept on the
    device from before an Engine upgrade): a current release replaces it, no newer Engine is needed."""

    key = _schema_key(schema_version)
    return key is not None and all(key < (_schema_key(s) or ()) for s in SUPPORTED_SCHEMA_VERSIONS)


def check_compatible(schema_version: Any, catalog_version: Any) -> None:
    """Refuse a release this Engine cannot read: ``ENGINE_UPGRADE_REQUIRED`` for a newer (or unknown) catalog schema,
    ``CATALOG_NOT_AVAILABLE`` for an earlier one (``is_retired_schema``: a current release replaces it)."""

    if schema_version in SUPPORTED_SCHEMA_VERSIONS:
        return
    details = {"catalog_version": catalog_version, "schema_version": schema_version, "supported_schema_versions": list(SUPPORTED_SCHEMA_VERSIONS)}
    if is_retired_schema(schema_version):
        details["retired"] = True
        raise CatalogError(
            CatalogErrorCode.CATALOG_NOT_AVAILABLE,
            f"public catalog {catalog_version} is written in catalog schema {schema_version}, an earlier format this "
            f"os-optics no longer reads (it reads schema {', '.join(SUPPORTED_SCHEMA_VERSIONS)}); get a current "
            "release with `os-optics catalog sync` (or `os-optics catalog import <file>`)",
            details,
        )
    raise CatalogError(
        CatalogErrorCode.ENGINE_UPGRADE_REQUIRED,
        f"public catalog {catalog_version} is written in catalog schema {schema_version}; this os-optics reads "
        f"schema {', '.join(SUPPORTED_SCHEMA_VERSIONS)}. Upgrade os-optics to use it.",
        details,
    )


_CONTRACT_IDENTITY: list[str] = []


def contract_identity() -> str:
    """The identity of the catalog contract this Engine validates a release with: a digest of the bundled contract
    modules (``_vendor/MANIFEST.json``, which lists the SHA-256 of each), the catalog schema versions it reads and the
    version of the validation library. A release whose index entry records the same identity (``StoredRelease.validated_by``)
    was validated in full by exactly these rules."""

    if not _CONTRACT_IDENTITY:
        import pydantic

        manifest = Path(__file__).resolve().parent.parent / "_vendor" / "MANIFEST.json"
        try:
            contracts = sha256_hex(manifest.read_bytes())
        except OSError:  # no manifest beside the contracts: no identity can be shown, every read validates in full
            contracts = ""
        _CONTRACT_IDENTITY.append(
            sha256_hex("|".join((contracts, ",".join(SUPPORTED_SCHEMA_VERSIONS), pydantic.VERSION)).encode("utf-8")) if contracts else "")
    return _CONTRACT_IDENTITY[0]


def validated_here(release: StoredRelease) -> bool:
    """Whether the whole document of ``release`` was validated by this Engine's catalog contract (``contract_identity``)."""

    identity = contract_identity()
    return bool(identity) and release.validated_by == identity


def validate_release(data: bytes) -> tuple[str, str]:
    """``(catalog_version, schema_version)`` of a release this Engine can read; refuses (``ENGINE_UPGRADE_REQUIRED``
    for a newer schema, ``CATALOG_INVALID`` otherwise) anything else. The whole document is validated."""

    document = release_header(data)
    check_compatible(document.get("schema_version"), document.get("catalog_version"))
    try:
        catalog = PublicOpticalCatalogV2.model_validate(document)
    except ValidationError as exc:
        first = exc.errors()[:MAX_REPORTED_PROBLEMS]
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in first)
        raise CatalogError(CatalogErrorCode.CATALOG_INVALID, f"not a valid public optical catalog: {problems}") from None
    return catalog.catalog_version, catalog.schema_version


def _file_has(path: Path, digest: str) -> bool:
    """Whether the file at ``path`` holds exactly the bytes of SHA-256 ``digest``."""

    try:
        return sha256_hex(path.read_bytes()) == digest
    except OSError:
        return False


class CatalogStore:
    """The store at one directory (see the module docstring)."""

    def __init__(self, root: Path | str, *, lock_timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        self.root = Path(root)
        self.lock_timeout = lock_timeout

    @classmethod
    def from_environment(cls, environ: Any = None) -> CatalogStore:
        return cls(store_directory(environ))

    def __repr__(self) -> str:
        return f"CatalogStore({str(self.root)!r})"

    @property
    def releases_dir(self) -> Path:
        return self.root / RELEASES_DIR

    def release_file(self, catalog_version: str) -> Path:
        if not CATALOG_VERSION_PATTERN.fullmatch(catalog_version):
            raise CatalogError(CatalogErrorCode.CATALOG_INVALID, f"{catalog_version!r} is not a catalog version label")
        return self.releases_dir / f"{catalog_version}.json"

    def lock(self) -> StoreLock:
        """The store's exclusive, re-entrant lock (``locking.StoreLock``; ``CATALOG_STORE_BUSY`` after the bounded wait)."""

        return StoreLock(self.root, self.lock_timeout)

    # ------------------------------------------------------------------------------------------------ state

    def read_state(self) -> StoreState:
        """The current index; an absent store is empty. An unreadable or missing index is rebuilt from the release files
        present (module docstring: no active release) and the rebuilt index is written."""

        if not self.root.is_dir():
            return StoreState()
        try:
            with self.lock():
                return self._load_state()
        except CatalogError as exc:
            if exc.code is not CatalogErrorCode.CATALOG_STORE_ERROR:
                raise
        # the lock cannot be created (a read-only store): nobody can write the index either, so read it as it is
        return self._load_state(persist=False)

    def update_state(self, change: Callable[[StoreState], StoreState]) -> StoreState:
        """Apply ``change`` to the current index under the store lock and write the result (when it differs). ``change``
        may raise ``CatalogError`` to refuse (nothing is written)."""

        with self.lock():
            current = self._load_state()
            updated = change(current)
            if updated != current:
                self._write_state(updated)
            return updated

    def write_state(self, state: StoreState) -> None:
        """Replace the index with ``state`` (under the store lock). Store operations use ``update_state``."""

        with self.lock():
            self._write_state(state)

    def _load_state(self, *, persist: bool = True) -> StoreState:
        path = self.root / STATE_FILE
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return self._recover(persist) if self.releases_dir.is_dir() else StoreState()
        except OSError as exc:
            raise CatalogError(CatalogErrorCode.CATALOG_STORE_ERROR, f"cannot read the catalog store index {path}: {exc}") from None
        try:
            document = json.loads(raw)
            releases = {
                version: StoredRelease(version, str(record["sha256"]), int(record["size_bytes"]), str(record["schema_version"]),
                                       str(record.get("source", SOURCE_SYNC)), float(record.get("added_at", 0.0)),
                                       record["validated_by"] if isinstance(record.get("validated_by"), str) else None)
                for version, record in document["releases"].items()
            }
            if not all(CATALOG_VERSION_PATTERN.fullmatch(version) for version in releases):
                raise ValueError("not a release label")
            active = document.get("active")
            checked_at = float(document["checked_at"]) if document.get("checked_at") is not None else None
            last_error = document.get("last_check_error") if isinstance(document.get("last_check_error"), dict) else None
            server_current = document.get("server_current") if isinstance(document.get("server_current"), dict) else None
            if document.get("contacted_at") is not None:
                contacted_at = float(document["contacted_at"])
            else:  # an index written before contacted_at was recorded
                contacted_at = checked_at if server_current is not None and last_error is None else None
            return StoreState(
                active=active if active in releases else None,
                releases=releases,
                manifest_url=document.get("manifest_url"),
                manifest_etag=document.get("manifest_etag"),
                release_path_template=document.get("release_path_template"),
                server_current=server_current,
                checked_at=checked_at,
                last_check_error=last_error,
                contacted_at=contacted_at,
                recovered_at=float(document["recovered_at"]) if document.get("recovered_at") is not None else None,
                active_basis=document.get("active_basis") if active in releases and document.get("active_basis") in ACTIVE_BASES else None,
            )
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
            return self._recover(persist)

    def _recover(self, persist: bool) -> StoreState:
        state = self._rebuilt()
        if persist:
            try:
                self._write_state(state)
            except CatalogError:  # an unwritable store: rebuilt again on the next read
                pass
        return state

    def _rebuilt(self) -> StoreState:
        """An index recovered from the release files (each re-verified as a readable release; others are ignored),
        with no active release (module docstring) and no record of the catalog service."""

        releases: dict[str, StoredRelease] = {}
        for path in sorted(self.releases_dir.glob("*.json")) if self.releases_dir.is_dir() else ():
            try:
                data = path.read_bytes()
                version, schema_version = validate_release(data)
            except (OSError, CatalogError):
                continue
            if path.name == f"{version}.json":
                releases[version] = StoredRelease(version, sha256_hex(data), len(data), schema_version, SOURCE_IMPORT, path.stat().st_mtime,
                                                  contract_identity() or None)
        return StoreState(active=None, releases=releases, recovered_at=time.time() if releases else None)

    def _write_state(self, state: StoreState) -> None:
        self._write_atomic(self.root / STATE_FILE, (json.dumps(state.document(), indent=2, sort_keys=True) + "\n").encode("utf-8"))

    def _write_atomic(self, path: Path, data: bytes) -> None:
        temporary = temporary_path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(data)
            deadline = time.monotonic() + REPLACE_RETRY_SECONDS
            while True:
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:  # Windows: another process has the target open for reading
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(REPLACE_RETRY_PAUSE_SECONDS)
        except OSError as exc:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise CatalogError(CatalogErrorCode.CATALOG_STORE_ERROR, f"cannot write the catalog store at {self.root}: {exc}") from None

    # ------------------------------------------------------------------------------------------------ releases

    def read_release(self, release: StoredRelease) -> bytes:
        """The exact bytes of a stored release, verified against its recorded SHA-256."""

        path = self.release_file(release.catalog_version)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CatalogError(CatalogErrorCode.CATALOG_INTEGRITY_FAILED,
                               f"stored catalog release {release.catalog_version} cannot be read: {exc}",
                               {"catalog_version": release.catalog_version}) from None
        if sha256_hex(data) != release.sha256:
            raise CatalogError(CatalogErrorCode.CATALOG_INTEGRITY_FAILED,
                               f"stored catalog release {release.catalog_version} differs from the release it was verified as; "
                               "run 'os-optics catalog verify' and 'os-optics catalog sync' to repair it",
                               {"catalog_version": release.catalog_version, "expected_sha256": release.sha256})
        return data

    def file_intact(self, release: StoredRelease) -> bool:
        try:
            self.read_release(release)
        except CatalogError:
            return False
        return True

    def add_release(self, data: bytes, *, source: str, expected_sha256: str | None = None, expected_size: int | None = None,
                    expected_version: str | None = None, activate: bool = False,
                    activate_first: bool = False) -> tuple[StoreState, StoredRelease]:
        """Verify ``data`` and store it (unless already held), making it the active release when ``activate``, or with
        ``activate_first`` when the index held no release at all (the first release of an empty store; a store holding
        releases with none active is not empty, and its active release is never guessed). Returns the updated index
        (written) and the release.

        Refuses bytes that differ from ``expected_sha256`` / ``expected_size`` (``CATALOG_INTEGRITY_FAILED``), a release
        whose label differs from ``expected_version``, a document this Engine cannot read, and a different document
        under a ``catalog_version`` already held (``CATALOG_RELEASE_CONFLICT``). The bytes are verified before the store
        lock is taken; the release file and the index are written under it, against the current index."""

        digest = sha256_hex(data)
        if expected_sha256 is not None and digest != expected_sha256:
            raise CatalogError(CatalogErrorCode.CATALOG_INTEGRITY_FAILED,
                               f"the downloaded catalog has SHA-256 {digest}, not the published {expected_sha256}; it was not used",
                               {"expected_sha256": expected_sha256, "sha256": digest})
        if expected_size is not None and len(data) != expected_size:
            raise CatalogError(CatalogErrorCode.CATALOG_INTEGRITY_FAILED,
                               f"the downloaded catalog has {len(data)} bytes, not the published {expected_size}; it was not used",
                               {"expected_size_bytes": expected_size, "size_bytes": len(data)})
        version, schema_version = validate_release(data)
        if expected_version is not None and version != expected_version:
            raise CatalogError(CatalogErrorCode.CATALOG_INTEGRITY_FAILED,
                               f"the downloaded document is catalog {version}, not the requested {expected_version}; it was not used")
        stored: list[StoredRelease] = []

        def change(state: StoreState) -> StoreState:
            held = state.releases.get(version)
            if held is not None and held.sha256 != digest:
                raise CatalogError(
                    CatalogErrorCode.CATALOG_RELEASE_CONFLICT,
                    f"this device already holds a different public catalog {version}; a catalog version always denotes the "
                    "same document, so the stored release was kept",
                    {"catalog_version": version, "stored_sha256": held.sha256, "offered_sha256": digest},
                )
            # (these bytes were validated in full just above: the entry records by which contract)
            release = (replace(held, validated_by=contract_identity() or None) if held is not None else
                       StoredRelease(version, digest, len(data), schema_version, source, time.time(), contract_identity() or None))
            path = self.release_file(version)
            if not _file_has(path, digest):  # absent, or damaged on disk: (re)write the verified bytes
                self._write_atomic(path, data)
            stored.append(release)
            first = activate_first and state.active is None and not state.releases
            state = replace(state, releases={**state.releases, version: release})
            return state.activated(version) if activate or first else state

        state = self.update_state(change)
        return state, stored[-1]

    def mark_validated(self, release: StoredRelease) -> None:
        """Record that the whole document of ``release`` was validated by this Engine's catalog contract (a release stored
        by an Engine with another contract, read and validated here). A store that cannot be written keeps its index: the
        next reader validates the document again."""

        identity = contract_identity()
        if not identity:
            return

        def change(state: StoreState) -> StoreState:
            held = state.releases.get(release.catalog_version)
            if held is None or held.sha256 != release.sha256 or held.validated_by == identity:
                return state
            return replace(state, releases={**state.releases, release.catalog_version: replace(held, validated_by=identity)})

        try:
            self.update_state(change)
        except CatalogError:
            pass
