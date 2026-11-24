"""The public optical catalogs the test suite uses, and local catalog stores seeded from them.

The Engine bundles no catalog: a local run reads the local catalog store (``os_optics.catalog``). The suite never
depends on whatever store or service the developer's machine has: ``conftest.py`` points ``OPENSCI_CATALOG_DIR`` at a
temporary store seeded from an explicit file, with automatic checks off (``OPENSCI_CATALOG_CHECK_INTERVAL=0``):

* ``FIXTURE_CATALOG`` (``tests/fixtures/catalog/test_optical_catalog.v2.json``, release ``2026-01-01.1``): the small,
  frozen synthetic catalog -- the default active release of the suite;
* ``REAL_CATALOG``: one pinned release of the public optical catalog (``PUBLIC_CATALOG_VERSION``, verified by its
  SHA-256 and size), used only where a test genuinely needs the real catalog (reference designs / examples, parity,
  catalog physics, the release QA probe): such tests use the ``real_catalog_store`` fixture (the store whose active
  release is this one) or pass ``REAL_CATALOG`` explicitly.

The catalog is versioned data, not part of this repository: the suite reads the release from the file named by
``OS_OPTICS_TEST_PUBLIC_CATALOG`` or, by default, from ``.cache/public-catalog/`` at the repository root, and downloads it
there once from the public catalog service (``PUBLIC_CATALOG_URL``, the same release file ``os-optics catalog sync``
downloads) when it is missing. ``python tests/catalog_store_helpers.py`` fetches (or verifies) it ahead of a run.

A release is a version axis like the Engine's: a new release with its products unchanged must fail nothing. A test that
needs a design on the pinned release takes the committed design and re-pins it at test time (``pinned_to_seed``) or reads
the release (``catalog_version_of(REAL_CATALOG)``). A test of catalog behaviour (resolution, sync, retention, refusal of
an old release) states its own releases.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURE_CATALOG = TESTS_DIR / "fixtures" / "catalog" / "test_optical_catalog.v2.json"
#: The pinned public catalog release (schema 3.2) the real-catalog tests run against.
PUBLIC_CATALOG_VERSION = "2026-10-28.1"
PUBLIC_CATALOG_SHA256 = "7d7f5f0a6fa1db7ce32503a6a1d86c5bdcd167035782b47b73c2ac00fc1acf30"
PUBLIC_CATALOG_SIZE = 13463067
PUBLIC_CATALOG_URL = f"https://opensci-optics.com/api/v1/catalog/releases/{PUBLIC_CATALOG_VERSION}"
ENV_PUBLIC_CATALOG = "OS_OPTICS_TEST_PUBLIC_CATALOG"
REAL_CATALOG = Path(os.environ.get(ENV_PUBLIC_CATALOG) or
                    REPO_ROOT / ".cache" / "public-catalog" / f"public_optical_catalog.{PUBLIC_CATALOG_VERSION}.json")
ENV_CATALOG_DIR = "OPENSCI_CATALOG_DIR"
ENV_CHECK_INTERVAL = "OPENSCI_CATALOG_CHECK_INTERVAL"
CATALOG_URL_VARIABLES = ("OPENSCI_CATALOG_URL", "OPENSCI_API_URL")


def _verified(data: bytes) -> bool:
    return len(data) == PUBLIC_CATALOG_SIZE and hashlib.sha256(data).hexdigest() == PUBLIC_CATALOG_SHA256


def ensure_public_catalog(timeout: float = 300.0) -> Path:
    """``REAL_CATALOG``, downloaded from ``PUBLIC_CATALOG_URL`` first if it is missing; its SHA-256 and size are checked.

    Raises ``RuntimeError`` (with what to do) when the file is wrong or cannot be fetched."""
    if REAL_CATALOG.is_file():
        if not _verified(REAL_CATALOG.read_bytes()):
            raise RuntimeError(f"{REAL_CATALOG} is not public catalog release {PUBLIC_CATALOG_VERSION} "
                               f"(sha256 {PUBLIC_CATALOG_SHA256}); delete it or point {ENV_PUBLIC_CATALOG} at the release")
        return REAL_CATALOG
    if os.environ.get(ENV_PUBLIC_CATALOG):
        raise RuntimeError(f"{ENV_PUBLIC_CATALOG} names {REAL_CATALOG}, which does not exist")
    request = urllib.request.Request(PUBLIC_CATALOG_URL, headers={"User-Agent": "os-optics-test-suite"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
            data = response.read(PUBLIC_CATALOG_SIZE + 1)
    except OSError as exc:
        raise RuntimeError(f"cannot download public catalog release {PUBLIC_CATALOG_VERSION} from {PUBLIC_CATALOG_URL} "
                           f"({exc}); download it there yourself and set {ENV_PUBLIC_CATALOG} to the file") from exc
    if not _verified(data):
        raise RuntimeError(f"{PUBLIC_CATALOG_URL} did not return release {PUBLIC_CATALOG_VERSION} "
                           f"(sha256 {PUBLIC_CATALOG_SHA256}, {PUBLIC_CATALOG_SIZE} bytes)")
    REAL_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    partial = REAL_CATALOG.with_name(f"{REAL_CATALOG.name}.{os.getpid()}.part")
    partial.write_bytes(data)
    os.replace(partial, REAL_CATALOG)
    return REAL_CATALOG


def catalog_version_of(path: Path) -> str:
    return json.loads(path.read_bytes())["catalog_version"]


def pinned_to_seed(document: dict) -> dict:
    """A copy of a committed design with every product pinned to the current seed release; nothing else changes."""

    def repin(value):
        if isinstance(value, dict):
            pinned = {key: repin(item) for key, item in value.items()}
            if isinstance(value.get("product"), dict) and "catalog_version" in value["product"]:
                pinned["product"] = {**value["product"], "catalog_version": catalog_version_of(REAL_CATALOG)}
            return pinned
        return [repin(item) for item in value] if isinstance(value, list) else value

    return repin(document)


def seed_store(root: Path, *files: Path, active: Path) -> Path:
    """A catalog store at ``root`` holding ``files`` (imported, verified), with ``active`` as the active release."""

    from os_optics.catalog import import_release

    environ = {ENV_CATALOG_DIR: str(root), ENV_CHECK_INTERVAL: "0"}
    for path in files:
        if path != active:
            import_release(path, activate=False, environ=environ)
    import_release(active, activate=True, environ=environ)
    return root


def store_environment(root: Path) -> dict[str, str]:
    """``os.environ`` for a subprocess that must use the store at ``root`` without contacting any service."""

    env = {k: v for k, v in os.environ.items() if k not in CATALOG_URL_VARIABLES}
    env.update({ENV_CATALOG_DIR: str(root), ENV_CHECK_INTERVAL: "0"})
    return env


if __name__ == "__main__":  # python tests/catalog_store_helpers.py: fetch (or verify) the pinned public catalog release
    try:
        print(ensure_public_catalog())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
