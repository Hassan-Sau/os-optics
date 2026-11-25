"""Test helpers for ``os_optics.protected_delivery`` and ``os_optics.client_artifacts``.

* the committed protected delivery vectors (``tests/fixtures/protected_assembly_delivery``: fixed key, nonce and
  ``now``), wrapped into the Commercial API delivery document;
* ``sealed_reference_deliveries()``: the reference pair of ``tests/fixtures/assembly_drawing`` sealed with the
  vendored canonical ``seal_assembly_preview`` / ``seal_assembly_detail`` (what the server delivers), valid now, as
  delivery documents. Test keys only: derived from fixed labels, never used outside tests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from os_optics._vendor.shared_contracts.protected_assembly_delivery import (
    b64url_decode,
    parse_utc,
    seal_assembly_detail,
    seal_assembly_preview,
)
from os_optics.assembly import load_assembly_preview
from os_optics.assembly_drawing import load_assembly_detail_package
from os_optics.protected_delivery import DELIVERY_SCHEMA_ID, DELIVERY_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/fixtures/protected_assembly_delivery"
PREVIEW_VECTOR_FILE = VECTORS / "vector_v1.json"
DETAIL_VECTOR_FILE = VECTORS / "detail_vector_v1.json"
FIXTURES = ROOT / "tests/fixtures/assembly_drawing"
PREVIEW_ZIP = FIXTURES / "reference.assembly-preview.zip"
DETAIL_JSON = FIXTURES / "reference.assembly-detail.json"
TEST_ORGANIZATION_ID = "00000000-0000-4000-8000-0000000000f8"


def load_vector(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def vector_now(vector: dict[str, Any]) -> datetime:
    return parse_utc(vector["now"])


def header_of(envelope: dict[str, Any]) -> dict[str, Any]:
    return json.loads(envelope["header"])


def delivery_document(kind: str, envelope: dict[str, Any], grant: dict[str, Any]) -> dict[str, Any]:
    """The Commercial API delivery document around one envelope + grant (JSON objects)."""

    return {
        "schema": DELIVERY_SCHEMA_ID,
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "delivery": kind,
        "commercial_resolution_id": header_of(envelope)["binding"]["commercial_resolution_id"],
        "envelope": envelope,
        "grant": grant,
    }


def vector_document(vector: dict[str, Any], kind: str) -> dict[str, Any]:
    return delivery_document(kind, vector["envelope"], vector["grant"])


def vector_preview_package(detail_vector: dict[str, Any]):
    """The verified preview package the detail vector supplements."""

    return load_assembly_preview(b64url_decode(detail_vector["preview_archive"]))


def _test_bytes(label: str, size: int) -> bytes:
    return hashlib.sha256(f"opensci-engine test only: {label}".encode("ascii")).digest()[:size]


def sealed_reference_deliveries(now: datetime | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(preview delivery document, detail delivery document) of the reference pair, valid at ``now``."""

    moment = (now or datetime.now(UTC)).replace(microsecond=0)
    issued, expires = moment - timedelta(minutes=1), moment + timedelta(minutes=9)
    preview = load_assembly_preview(PREVIEW_ZIP)
    detail = load_assembly_detail_package(DETAIL_JSON, preview).detail
    envelope, grant = seal_assembly_preview(
        preview, key=_test_bytes("preview key", 32), nonce=_test_bytes("preview nonce", 12),
        envelope_id="00000000-0000-4000-8000-0000000000e8", grant_id="00000000-0000-4000-8000-0000000000c8",
        organization_id=TEST_ORGANIZATION_ID, issued_at=issued, expires_at=expires)
    detail_envelope, detail_grant = seal_assembly_detail(
        detail, preview, key=_test_bytes("detail key", 32), nonce=_test_bytes("detail nonce", 12),
        envelope_id="00000000-0000-4000-8000-0000000000e9", grant_id="00000000-0000-4000-8000-0000000000c9",
        organization_id=TEST_ORGANIZATION_ID, issued_at=issued, expires_at=expires)
    return (delivery_document("ASSEMBLY_PREVIEW", json.loads(envelope.to_json_bytes()), json.loads(grant.to_json_bytes())),
            delivery_document("ASSEMBLY_DETAIL", json.loads(detail_envelope.to_json_bytes()),
                              json.loads(detail_grant.to_json_bytes())))
