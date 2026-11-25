"""The vendored contract bundle (``src/os_optics/_vendor``) is complete, unaltered and self-contained.

The wheel ships the pure-optical design contracts, the customer/device-facing ``assembly_preview`` package contract, the
``assembly_detail`` package / local drawing output contract, the ``protected_assembly_delivery`` envelope/grant contract
a client opens, the ``client_assembly_artifact`` upload manifest and the ``optical_layout`` package contract, so that
``pip install os-optics`` alone supports local design. ``MANIFEST.json`` names every packaged file with its SHA-256: a
file edited without updating the manifest, a missing file and an unexpected one all fail here.

The public optical catalog is NOT bundled: it is versioned data the Engine synchronizes into a local catalog store
(``os_optics.catalog``), so a catalog release never requires an Engine release.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "src" / "os_optics" / "_vendor"
#: The package files the bundle has besides the manifest's: its own package markers.
GENERATED = {"__init__.py", "shared_contracts/__init__.py", "MANIFEST.json"}
FAMILIES = {"assembly_detail", "assembly_preview", "client_assembly_artifact", "common", "design_catalog", "optical_design",
            "optical_layout", "optical_view", "protected_assembly_delivery"}


def _committed() -> dict[str, bytes]:
    return {path.relative_to(VENDOR_DIR).as_posix(): path.read_bytes().replace(b"\r\n", b"\n")
            for path in sorted(VENDOR_DIR.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def test_the_manifest_lists_every_vendored_file_with_its_digest():
    committed = _committed()
    manifest = json.loads(committed["MANIFEST.json"])
    assert set(manifest) == {"files"} and manifest["files"]
    assert set(manifest["files"]) == set(committed) - GENERATED
    for path, entry in manifest["files"].items():
        assert set(entry) == {"sha256"}, path
        assert hashlib.sha256(committed[path]).hexdigest() == entry["sha256"], f"{path} differs from MANIFEST.json"


def test_the_bundle_holds_the_contract_families_and_no_catalog_document():
    committed = _committed()
    assert [path for path in committed if path.endswith(".json") and path != "MANIFEST.json"] == []
    assert "shared_contracts/design_catalog/manifest.py" in committed
    families = {Path(path).parts[1] for path in committed if path.startswith("shared_contracts/") and len(Path(path).parts) > 2}
    assert families == FAMILIES


def test_vendored_modules_import_only_the_bundle_the_standard_library_and_pydantic():
    """Every import of a vendored module is relative (inside the bundle), from the standard library, pydantic or -- the
    protected-delivery envelope only, imported lazily -- the optional ``cryptography`` (the ``protected`` extra): no
    absolute ``shared_contracts`` import (it would not resolve inside the wheel) and no server-side package."""
    import sys

    allowed = set(sys.stdlib_module_names) | {"pydantic", "pydantic_core", "typing_extensions", "annotated_types"}
    for path, data in _committed().items():
        if not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(data.decode("utf-8"), filename=path)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            optional = {"cryptography"} if path.startswith("shared_contracts/protected_assembly_delivery/") else set()
            bad = [name for name in names if name.split(".")[0] not in allowed | optional]
            assert bad == [], f"{path}:{node.lineno} imports {bad}"


def test_the_engine_reads_the_schema_of_the_pinned_public_catalog_release():
    """The pinned public release the suite runs against is one this Engine can read, so a fresh store synced from the
    service works; the catalog itself is never part of the bundle."""
    from catalog_store_helpers import REAL_CATALOG

    from os_optics.catalog import SUPPORTED_SCHEMA_VERSIONS, validate_release

    version, schema_version = validate_release(REAL_CATALOG.read_bytes())
    assert schema_version in SUPPORTED_SCHEMA_VERSIONS and version == json.loads(REAL_CATALOG.read_bytes())["catalog_version"]
