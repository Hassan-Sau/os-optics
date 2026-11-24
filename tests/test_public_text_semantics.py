"""Customer-visible text of the design workflow states product facts.

The published package is public: every docstring and string constant a user can print, the CLI help, the JSON Schemas
``os-optics schema`` prints and the published catalog are product text. These static checks keep phrases that
describe implementation layers, sourcing or development stages out of those surfaces, and check that every product of
the repository's seed catalog release carries its Simplified Chinese name.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
from pathlib import Path

import pytest

import os_optics
from os_optics.cli import main as cli_main
from os_optics.local import load_public_catalog

from catalog_store_helpers import REAL_CATALOG
from os_optics.local.json_api import json_schema

PACKAGE = Path(os_optics.__file__).resolve().parent

# The assembly, drawing and protected-delivery surfaces have their own text checks.
ASSEMBLY_SURFACES = (
    "assembly/", "assembly_drawing/", "protected_delivery.py", "client_artifacts.py", "cli.py",
    "_vendor/shared_contracts/assembly_preview/", "_vendor/shared_contracts/assembly_detail/",
    "_vendor/shared_contracts/protected_assembly_delivery/", "_vendor/shared_contracts/client_assembly_artifact/",
    # the optics-only Optical Layout Package contract; the layout drawing's own output text is checked by
    # tests/test_optical_layout.py
    "_vendor/shared_contracts/optical_layout/",
)

# Phrases that describe sourcing, implementation layers or development stages instead of the product.
FORBIDDEN_PHRASES = (
    "supplier", "self-reported", "self_reported", "resolved_by_private_assembly", "private assembly", "private data",
    "from public data", "reserved for opensci-internal", "opensci-internal", "design_part_ref", "not a laboratory assembly",
    "nothing here names where it came from", "procurement", "reverse mapping", "reverse-mapping", "proxy geometry",
    "does not validate", "not guaranteed", "engineering review", "in this version", "not yet", "numerical_conventions",
    "physics_scope", "provisional", "v0.1", "round 1", "round 2", "private resolver", "the resolver",
    "delivery layer", "vendored", "persisted commercial resolution", "mirrors ``", "neutral wire", "renderer-neutral",
    "first version", "self-declared", "not trusted", "which are not checked", "no fulfillment responsibility",
    "builder concern", "custom-part placeholder", "engine-input builder", "server-pinned", "internal, numpy",
    "private local", "private random", "private address", "127.0.0.1:8750", "controlled verification",
)

JSON_SCHEMA_KINDS = ("project", "local-validation", "catalog", "optical-scene", "view-settings", "linear-optical-network", "focal-field-request",
                     "analysis-result", "imaging-request", "ghost-request")
CLI_HELP_ARGV = (
    ["--help"], ["catalog", "--help"], ["catalog", "list", "--help"], ["catalog", "show", "--help"], ["project", "--help"],
    ["project", "example", "--help"], ["project", "run", "--help"], ["project", "verify", "--help"], ["schema", "--help"],
    ["validate", "--help"], ["results", "--help"], ["results", "wavefront", "--help"], ["results", "irradiance", "--help"],
    ["results", "metrics", "--help"], ["results", "configurations", "--help"], ["export", "--help"], ["project", "network", "--help"], ["results", "handoff", "--help"],
    ["project", "paraxial", "--help"], ["project", "cavity", "--help"], ["interop", "--help"], ["interop", "list", "--help"], ["interop", "show", "--help"],
    ["view", "--help"], ["project", "solids", "--help"], ["project", "evaluate", "--help"],
    ["results", "focal-field", "--help"], ["results", "imaging", "--help"], ["results", "ghosts", "--help"], ["results", "cavity", "--help"],
)


def _hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def _shipped_strings():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith(ASSEMBLY_SURFACES):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield rel, node.lineno, node.value


# Every hit fails: nothing in the design workflow is exempt.
ALLOWED: set[tuple[str, str]] = set()


def test_docstrings_and_string_constants_of_the_design_workflow_state_product_facts() -> None:
    found = [f"{rel}:{line}: {hits}" for rel, line, text in _shipped_strings()
             if (hits := [h for h in _hits(text) if (rel, h) not in ALLOWED])]
    assert found == []


@pytest.mark.parametrize("argv", CLI_HELP_ARGV, ids=lambda argv: " ".join(argv))
def test_cli_help_states_product_facts(argv: list[str]) -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
        cli_main(argv)
    assert exc.value.code == 0
    text = out.getvalue()
    assert text.strip()
    assert _hits(text) == []
    assert "cannot simulate from public data" not in text


@pytest.mark.parametrize("kind", JSON_SCHEMA_KINDS)
def test_printed_json_schemas_state_product_facts(kind: str) -> None:
    result = json_schema(kind)
    assert result["ok"] is True
    assert _hits(json.dumps(result["schema"])) == []


def test_seed_catalog_states_product_facts() -> None:
    document = REAL_CATALOG.read_bytes().decode("utf-8")
    for phrase in ("supplier", "resolved_by_private_assembly", "private", "procurement", "design interpretation",
                   "design_interpretation", "ar_near_normal", "glass catalogue", "an equivalent glass", "too high", "pointwise",
                   "(not modelled)", '"approximation"'):
        assert phrase not in document.lower(), phrase
    assert json.loads(document)["mechanical_accessories"] == "CONFIGURED_PER_DESIGN"


_ZH = re.compile(r"[一-鿿]")


def test_every_seed_catalog_product_has_a_simplified_chinese_name() -> None:
    catalog = load_public_catalog(REAL_CATALOG)
    assert catalog.schema_version == "3.2"  # the seed bounds its coatings' group delays (schema 3.2) besides its 3.1 fields
    for product in catalog.products:
        name = product.display_name_zh_cn
        assert _ZH.search(name), product.optical_sku
        # the same specification numbers as the English name
        numbers = sorted(re.findall(r"-?[0-9]+(?:\.[0-9]+)?", name))
        assert numbers == sorted(re.findall(r"-?[0-9]+(?:\.[0-9]+)?", product.display_name)), product.optical_sku


def test_catalog_rows_carry_both_names() -> None:
    from os_optics.local.json_api import catalog_products

    page = catalog_products(limit=5, summary=True)
    assert page["ok"] is True and page["mechanical_accessories"] == "CONFIGURED_PER_DESIGN"
    for row in page["products"]:
        assert row["display_name"] and _ZH.search(row["display_name_zh_cn"])


def test_catalog_list_prints_valid_json_on_a_console_without_chinese_characters() -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "-m", "os_optics", "catalog", "list", "--limit", "2"], capture_output=True, env=env,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    rows = json.loads(proc.stdout.decode("cp1252"))["products"]
    assert all(_ZH.search(row["display_name_zh_cn"]) for row in rows)
