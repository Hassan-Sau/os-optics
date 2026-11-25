"""Suite-wide setup: a temporary local catalog store (see ``catalog_store_helpers``).

Before any test module is imported (module-level catalog loads included), ``OPENSCI_CATALOG_DIR`` points at a temporary
store whose active release is the small fixture catalog, automatic catalog checks are off and no catalog service URL is
set, so no test reads the developer's own store or reaches the network. ``real_catalog_store`` switches a test to a
second store whose active release is the pinned public catalog release (see ``catalog_store_helpers``).

The pinned public release is fetched only when a collected test module needs it: a module that names ``REAL_CATALOG`` or
``real_catalog_store``, or imports a test helper that does. Every other test runs offline. A module that needs the release
when it cannot be obtained is not imported: it reports a failing test with the reason and what to do, carrying the module's
markers so that a ``-m`` selection still selects it, and the other modules still run; nothing is skipped silently.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from catalog_store_helpers import (CATALOG_URL_VARIABLES, ENV_CATALOG_DIR, ENV_CHECK_INTERVAL, FIXTURE_CATALOG, REAL_CATALOG,
                                   ensure_public_catalog, seed_store)

_STORES: dict[str, Path] = {}
TESTS_DIR = Path(__file__).resolve().parent
_NEEDS = re.compile(r"\bREAL_CATALOG\b|\breal_catalog_store\b|\bensure_public_catalog\b")
_IMPORTS = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w]*)\s+import|import\s+([A-Za-z_][\w]*))", re.M)


@functools.cache
def _modules_needing_the_public_catalog() -> frozenset[str]:
    """Names of the modules of ``tests/`` that read the pinned public release, directly or through a helper they import."""
    sources = {path.stem: path.read_text(encoding="utf-8") for path in TESTS_DIR.glob("*.py") if path.stem != "conftest"}
    imports = {name: {a or b for a, b in _IMPORTS.findall(text)} & set(sources) for name, text in sources.items()}
    needing = {name for name, text in sources.items() if _NEEDS.search(text) and name != "catalog_store_helpers"}
    while True:
        more = {name for name, used in imports.items() if used & needing} - needing
        if not more:
            return frozenset(needing)
        needing |= more


def _real_store() -> Path:
    """The store whose active release is the pinned public release, fetched and seeded on first use."""
    if "real" not in _STORES:
        ensure_public_catalog()
        _STORES["real"] = seed_store(_STORES["root"] / "real", FIXTURE_CATALOG, REAL_CATALOG, active=REAL_CATALOG)
    return _STORES["real"]


class _PublicCatalogMissingItem(pytest.Item):
    """Stands for the tests of a module that needs the pinned public release when it cannot be obtained: it fails, naming
    why and what to do, and the other modules still run."""

    def __init__(self, *, reason: str, **kwargs):
        super().__init__(**kwargs)
        self.reason = reason

    def runtest(self):
        pytest.fail(f"this module needs the pinned public catalog release: {self.reason}", pytrace=False)

    def reportinfo(self):
        return self.path, None, f"{self.path.name}: needs the pinned public catalog release"


def _marker_sets(source: str, registered: set[str]) -> list[tuple[str, ...]]:
    """The marker sets the module's tests carry, read from its source: the module's own ``pytestmark`` on every set, plus
    one set per further registered marker a test of it carries. A ``-m`` selection then selects the failing stand-in
    wherever it would have selected a test of the module."""
    module = {name for line in re.findall(r"^pytestmark\s*=.*$", source, re.M)
              for name in re.findall(r"pytest\.mark\.(\w+)", line) if name in registered}
    used = {name for name in re.findall(r"pytest\.mark\.(\w+)", source) if name in registered}
    return [tuple(sorted(module))] + [tuple(sorted(module | {name})) for name in sorted(used - module)]


class _PublicCatalogMissing(pytest.File):
    """A test module that needs the pinned public release when it cannot be obtained: failing items, not imported.

    The items carry the module's registered markers (``install``, ``pyodide``, ``golden``), so ``-m install`` or
    ``-m pyodide`` selects and fails them instead of deselecting the module's tests silently."""

    reason = ""

    def collect(self):
        registered = {line.split(":", 1)[0].strip() for line in self.config.getini("markers")}
        for marks in _marker_sets(self.path.read_text(encoding="utf-8"), registered):
            name = "public_catalog_missing" + (f"[{'-'.join(marks)}]" if marks else "")
            item = _PublicCatalogMissingItem.from_parent(self, name=name, reason=self.reason)
            for mark in marks:
                item.add_marker(mark)
            yield item


def pytest_pycollect_makemodule(module_path: Path, parent):
    if module_path.parent != TESTS_DIR or module_path.stem not in _modules_needing_the_public_catalog():
        return None
    try:
        _real_store()
    except RuntimeError as exc:
        node = _PublicCatalogMissing.from_parent(parent, path=module_path)
        node.reason = str(exc)
        return node
    return None


def pytest_configure(config: pytest.Config) -> None:
    root = Path(tempfile.mkdtemp(prefix="opensci-catalog-stores-"))
    _STORES["root"] = root
    for name in CATALOG_URL_VARIABLES:
        os.environ.pop(name, None)
    os.environ[ENV_CHECK_INTERVAL] = "0"
    _STORES["fixture"] = seed_store(root / "fixture", FIXTURE_CATALOG, active=FIXTURE_CATALOG)
    os.environ[ENV_CATALOG_DIR] = str(_STORES["fixture"])


def pytest_unconfigure(config: pytest.Config) -> None:
    root = _STORES.get("root")
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _drawing_text_gate():
    """Every drawing and layout any test generates is checked: each output token comes from its inputs, its templates
    or is a number, and the output set verifies (``drawing_text_check``; the build-time gate, not a runtime check)."""

    from drawing_text_check import install_generation_hook

    with pytest.MonkeyPatch.context() as patch:
        install_generation_hook(patch)
        yield


@pytest.fixture(autouse=True)
def _isolated_catalog_environment(monkeypatch: pytest.MonkeyPatch):
    """Every test starts on the fixture store with automatic checks off and nothing installed in memory."""

    from os_optics.local import clear_installed_catalogs

    monkeypatch.setenv(ENV_CATALOG_DIR, str(_STORES["fixture"]))
    monkeypatch.setenv(ENV_CHECK_INTERVAL, "0")
    for name in CATALOG_URL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    clear_installed_catalogs()
    yield
    clear_installed_catalogs()


@pytest.fixture
def real_catalog_store(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use the store whose active release is the pinned public catalog release (``REAL_CATALOG``)."""

    try:
        store = _real_store()
    except RuntimeError as exc:
        pytest.fail(f"this test needs the pinned public catalog release: {exc}", pytrace=False)
    monkeypatch.setenv(ENV_CATALOG_DIR, str(store))
    return store


@pytest.fixture
def fixture_catalog_store() -> Path:
    """The default store (active release: the small fixture catalog)."""

    return _STORES["fixture"]


@pytest.fixture
def empty_catalog_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty store of this test's own."""

    root = tmp_path / "catalog-store"
    monkeypatch.setenv(ENV_CATALOG_DIR, str(root))
    return root
