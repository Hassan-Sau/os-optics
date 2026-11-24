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
import time
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


def _download(timeout: float) -> bytes:
    """The release file from ``PUBLIC_CATALOG_URL``, read within ``timeout`` seconds in all; raises ``RuntimeError``."""
    request = urllib.request.Request(PUBLIC_CATALOG_URL, headers={"User-Agent": "os-optics-test-suite"})
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    received = 0
    try:
        # a stalled connection fails within a minute; the deadline bounds the whole attempt
        with urllib.request.urlopen(request, timeout=min(timeout, 60.0)) as response:  # noqa: S310 (fixed https URL)
            while received <= PUBLIC_CATALOG_SIZE:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"not complete after {timeout:g} s ({received} of {PUBLIC_CATALOG_SIZE} bytes)")
                chunk = response.read(min(1 << 20, PUBLIC_CATALOG_SIZE + 1 - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot download public catalog release {PUBLIC_CATALOG_VERSION} from {PUBLIC_CATALOG_URL} "
                           f"({exc}); download it there yourself and set {ENV_PUBLIC_CATALOG} to the file") from exc
    data = b"".join(chunks)
    if not _verified(data):
        raise RuntimeError(f"{PUBLIC_CATALOG_URL} did not return release {PUBLIC_CATALOG_VERSION} "
                           f"(sha256 {PUBLIC_CATALOG_SHA256}, {PUBLIC_CATALOG_SIZE} bytes)")
    return data


def ensure_public_catalog(timeout: float = 300.0, attempts: int = 1, pause: float = 15.0) -> Path:
    """``REAL_CATALOG``, downloaded from ``PUBLIC_CATALOG_URL`` first if it is missing; its SHA-256 and size are checked.

    A download is tried up to ``attempts`` times, each limited to ``timeout`` seconds in all, ``pause`` seconds apart.
    Raises ``RuntimeError`` (with what to do) when the file is wrong or cannot be fetched: after the last attempt, the
    message of that attempt."""
    if REAL_CATALOG.is_file():
        if not _verified(REAL_CATALOG.read_bytes()):
            raise RuntimeError(f"{REAL_CATALOG} is not public catalog release {PUBLIC_CATALOG_VERSION} "
                               f"(sha256 {PUBLIC_CATALOG_SHA256}); delete it or point {ENV_PUBLIC_CATALOG} at the release")
        return REAL_CATALOG
    if os.environ.get(ENV_PUBLIC_CATALOG):
        raise RuntimeError(f"{ENV_PUBLIC_CATALOG} names {REAL_CATALOG}, which does not exist")
    for attempt in range(1, attempts + 1):
        try:
            data = _download(timeout)
            break
        except RuntimeError as exc:
            if attempt == attempts:
                raise
            print(f"attempt {attempt} of {attempts}: {exc}; retrying in {pause:g} s", file=sys.stderr, flush=True)
            time.sleep(pause)
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
    import argparse

    parser = argparse.ArgumentParser(description="Fetch (or verify) the pinned public catalog release.")
    parser.add_argument("--attempts", type=int, default=1, help="download attempts (default 1)")
    parser.add_argument("--attempt-timeout", type=float, default=300.0,
                        help="seconds per download attempt (default 300)")
    parser.add_argument("--pause", type=float, default=15.0, help="seconds between attempts (default 15)")
    args = parser.parse_args()
    try:
        print(ensure_public_catalog(timeout=args.attempt_timeout, attempts=max(1, args.attempts), pause=args.pause))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
