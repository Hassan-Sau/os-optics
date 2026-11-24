"""``os_optics.protected_delivery``: opening a Commercial API protected delivery document on this device.

The committed protected delivery vectors (fixed key / nonce / ``now``) open through the wrapper to exactly their plaintext;
the chain-L reference pair sealed with the vendored canonical contract opens and feeds the local assembly check and the drawing; the wrapper is strict
(schema, version, keys, delivery kind, resolution id) and every refusal is a ``ProtectedDeliveryError`` with a stable
code that never carries the key; a missing ``cryptography`` names the ``protected`` extra; and ``import os_optics``
/ ``os_optics.local`` neither import the new modules nor need ``cryptography``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import pytest

from delivery_helpers import (
    DETAIL_JSON,
    DETAIL_VECTOR_FILE,
    PREVIEW_VECTOR_FILE,
    PREVIEW_ZIP,
    TEST_ORGANIZATION_ID,
    header_of,
    load_vector,
    sealed_reference_deliveries,
    vector_document,
    vector_now,
    vector_preview_package,
)
from os_optics import protected_delivery as pd
from os_optics._vendor.shared_contracts.protected_assembly_delivery import b64url_decode, b64url_encode, parse_utc
from os_optics.assembly import load_assembly_preview, validate_assembly_layout
from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.protected_delivery import (
    CryptographyUnavailableError,
    ProtectedDeliveryDocumentError,
    ProtectedDeliveryError,
    ProtectedDeliveryErrorCode,
    open_detail_delivery,
    open_preview_delivery,
)

C = ProtectedDeliveryErrorCode
PREVIEW_VECTOR = load_vector(PREVIEW_VECTOR_FILE)
DETAIL_VECTOR = load_vector(DETAIL_VECTOR_FILE)
PREVIEW_NOW = vector_now(PREVIEW_VECTOR)
DETAIL_NOW = vector_now(DETAIL_VECTOR)


def preview_doc() -> dict:
    return copy.deepcopy(vector_document(PREVIEW_VECTOR, "ASSEMBLY_PREVIEW"))


def detail_doc() -> dict:
    return copy.deepcopy(vector_document(DETAIL_VECTOR, "ASSEMBLY_DETAIL"))


@pytest.fixture(scope="module")
def vector_preview():
    return vector_preview_package(DETAIL_VECTOR)


def refused(code: ProtectedDeliveryErrorCode, call) -> ProtectedDeliveryError:
    with pytest.raises(ProtectedDeliveryError) as info:
        call()
    assert info.value.code is code, (info.value.code, str(info.value))
    return info.value


# ---- the committed vectors ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["mapping", "text", "bytes"])
def test_the_preview_vector_opens_to_exactly_its_plaintext(form):
    document = preview_doc()
    value = {"mapping": document, "text": json.dumps(document), "bytes": json.dumps(document).encode("utf-8")}[form]
    package = open_preview_delivery(value, now=PREVIEW_NOW)
    header = header_of(PREVIEW_VECTOR["envelope"])
    assert isinstance(package, pd.AssemblyPreviewPackageV1)
    assert package.archive_bytes() == b64url_decode(PREVIEW_VECTOR["plaintext"])
    assert package.content_hash == header["plaintext"]["package_content_hash"] == PREVIEW_VECTOR["grant"]["package_content_hash"]
    assert package.manifest.binding.commercial_resolution_id == document["commercial_resolution_id"]


def test_the_detail_vector_opens_to_exactly_its_plaintext(vector_preview):
    detail = open_detail_delivery(detail_doc(), preview=vector_preview, now=DETAIL_NOW)
    header = header_of(DETAIL_VECTOR["envelope"])
    assert isinstance(detail, pd.VerifiedAssemblyDetailV1)
    assert detail.detail_bytes == b64url_decode(DETAIL_VECTOR["plaintext"])
    assert detail.content_hash == header["plaintext"]["detail_content_hash"]
    assert vector_preview.content_hash == header["plaintext"]["preview_package_content_hash"]


def test_expected_binding_values_pass_through():
    header = header_of(PREVIEW_VECTOR["envelope"])["binding"]
    package = open_preview_delivery(preview_doc(), now=PREVIEW_NOW, expected_resolution_id=header["commercial_resolution_id"],
                                    expected_project_hash=header["project_hash"], expected_organization_id=header["organization_id"])
    assert package.content_hash == PREVIEW_VECTOR["grant"]["package_content_hash"]
    for pin in ({"expected_resolution_id": "00000000-0000-4000-8000-000000000999"},
                {"expected_project_hash": "0" * 64}, {"expected_organization_id": TEST_ORGANIZATION_ID}):
        refused(C.BINDING_MISMATCH, lambda: open_preview_delivery(preview_doc(), now=PREVIEW_NOW, **pin))


# ---- the real reference pair, end to end -----------------------------------------------------------------------------------------
def test_the_sealed_reference_pair_opens_and_feeds_the_local_checks():
    preview_delivery, detail_delivery = sealed_reference_deliveries()
    preview = open_preview_delivery(json.dumps(preview_delivery))  # default now: the current UTC time
    reference = load_assembly_preview(PREVIEW_ZIP)
    assert preview.content_hash == reference.content_hash and preview.archive_bytes() == reference.archive_bytes()
    detail = open_detail_delivery(json.dumps(detail_delivery).encode("utf-8"), preview=preview,
                                  expected_organization_id=TEST_ORGANIZATION_ID)
    assert detail.detail_bytes.replace(b"\r\n", b"\n") == DETAIL_JSON.read_bytes().replace(b"\r\n", b"\n")
    assert validate_assembly_layout(preview).result_hash == validate_assembly_layout(reference).result_hash
    drawing = generate_assembly_drawing(preview, detail)
    assert dict(drawing.files) == dict(generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON).files)


# ---- the wrapper is strict ---------------------------------------------------------------------------------------------------------
def test_a_delivery_of_the_other_kind_is_refused_and_named(vector_preview):
    err = refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(detail_doc(), now=DETAIL_NOW))
    assert isinstance(err, ProtectedDeliveryDocumentError) and "open_detail_delivery" in str(err)
    err = refused(C.ENVELOPE_MALFORMED, lambda: open_detail_delivery(preview_doc(), preview=vector_preview, now=PREVIEW_NOW))
    assert "open_preview_delivery" in str(err)
    # a document that claims the right kind but carries the other kind's envelope / grant
    swapped = preview_doc() | {"envelope": DETAIL_VECTOR["envelope"], "commercial_resolution_id": detail_doc()["commercial_resolution_id"]}
    refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(swapped, now=PREVIEW_NOW))
    refused(C.GRANT_MALFORMED, lambda: open_preview_delivery(preview_doc() | {"grant": DETAIL_VECTOR["grant"]}, now=PREVIEW_NOW))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="opensci.protected_delivery"),
    lambda d: d.update(schema_version="1.0"),
    lambda d: d.update(schema_version=1),
    lambda d: d.update(delivery="ASSEMBLY_DRAWING"),
    lambda d: d.update(delivery="assembly_preview"),
    lambda d: d.update(extra="x"),
    lambda d: d.pop("commercial_resolution_id"),
    lambda d: d.pop("grant"),
    lambda d: d.update(commercial_resolution_id=None),
    lambda d: d.update(envelope=json.dumps(d["envelope"])),
    lambda d: d.update(grant=[d["grant"]]),
], ids=["schema", "version-text", "version-int", "kind-unknown", "kind-case", "extra-key", "no-resolution", "no-grant",
        "resolution-null", "envelope-as-text", "grant-as-list"])
def test_a_malformed_delivery_document_is_refused(mutate):
    document = preview_doc()
    mutate(document)
    err = refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(document, now=PREVIEW_NOW))
    assert isinstance(err, ProtectedDeliveryDocumentError) and err.reason


@pytest.mark.parametrize("value", [b"\xff", "not json", "[]", "null", json.dumps([1]), 7])
def test_a_delivery_that_is_not_a_json_object_is_refused(value):
    refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(value, now=PREVIEW_NOW))


def test_a_resolution_id_other_than_the_envelope_binding_is_refused(vector_preview):
    other = "00000000-0000-4000-8000-000000000999"
    err = refused(C.BINDING_MISMATCH, lambda: open_preview_delivery(preview_doc() | {"commercial_resolution_id": other}, now=PREVIEW_NOW))
    assert isinstance(err, ProtectedDeliveryDocumentError) and other not in str(err)
    refused(C.BINDING_MISMATCH, lambda: open_detail_delivery(detail_doc() | {"commercial_resolution_id": other}, preview=vector_preview,
                                                             now=DETAIL_NOW))


# ---- the contract's refusals surface unchanged --------------------------------------------------------------------------------------
def _reheader(envelope: dict, edit) -> dict:
    header = json.loads(envelope["header"])
    edit(header)
    return envelope | {"header": json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True)}


def test_a_tampered_header_fails_authentication():
    """A header edited consistently with its grant (so every binding check passes) still fails the AEAD tag."""

    later = (parse_utc(PREVIEW_VECTOR["grant"]["expires_at"]) + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    document = preview_doc()
    document["envelope"] = _reheader(document["envelope"], lambda h: h.update(expires_at=later))
    document["grant"]["expires_at"] = later
    refused(C.DECRYPTION_FAILED, lambda: open_preview_delivery(document, now=PREVIEW_NOW))
    # edited without the grant: the grant no longer belongs to the envelope
    document = preview_doc()
    document["envelope"] = _reheader(document["envelope"], lambda h: h.update(expires_at=later))
    refused(C.GRANT_ENVELOPE_MISMATCH, lambda: open_preview_delivery(document, now=PREVIEW_NOW))
    # not canonical header text
    document = preview_doc()
    document["envelope"]["header"] = document["envelope"]["header"].replace(",", ", ", 1)
    refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(document, now=PREVIEW_NOW))


def test_a_tampered_ciphertext_or_key_fails_authentication(vector_preview):
    document = detail_doc()
    data = bytearray(b64url_decode(document["envelope"]["ciphertext"]))
    data[0] ^= 1
    document["envelope"]["ciphertext"] = b64url_encode(bytes(data))
    refused(C.DECRYPTION_FAILED, lambda: open_detail_delivery(document, preview=vector_preview, now=DETAIL_NOW))
    document = detail_doc()
    key = document["grant"]["key"]
    document["grant"]["key"] = b64url_encode(hashlib.sha256(b"another key").digest())
    err = refused(C.DECRYPTION_FAILED, lambda: open_detail_delivery(document, preview=vector_preview, now=DETAIL_NOW))
    assert key not in str(err) and document["grant"]["key"] not in str(err)


def test_an_expired_or_not_yet_valid_grant_is_refused():
    header = header_of(PREVIEW_VECTOR["envelope"])
    expires, issued = parse_utc(header["expires_at"]), parse_utc(header["issued_at"])
    refused(C.GRANT_EXPIRED, lambda: open_preview_delivery(preview_doc(), now=expires))
    refused(C.GRANT_EXPIRED, lambda: open_preview_delivery(preview_doc(), now=issued - timedelta(minutes=6)))
    assert open_preview_delivery(preview_doc(), now=expires - timedelta(seconds=1)).content_hash
    refused(C.GRANT_EXPIRED, lambda: open_preview_delivery(preview_doc()))  # default now: long after the vector's window
    with pytest.raises(ValueError, match="timezone-aware"):
        open_preview_delivery(preview_doc(), now=datetime(2026, 9, 27, 12, 5))


def test_a_detail_against_another_preview_is_refused():
    other = load_assembly_preview(PREVIEW_ZIP)
    refused(C.PREVIEW_PACKAGE_MISMATCH, lambda: open_detail_delivery(detail_doc(), preview=other, now=DETAIL_NOW))


def test_no_refusal_carries_the_key():
    keys = (PREVIEW_VECTOR["grant"]["key"], DETAIL_VECTOR["grant"]["key"])
    for call in (lambda: open_preview_delivery(preview_doc()),
                 lambda: open_preview_delivery(detail_doc(), now=DETAIL_NOW),
                 lambda: open_preview_delivery(preview_doc() | {"grant": DETAIL_VECTOR["grant"]}, now=PREVIEW_NOW)):
        with pytest.raises(ProtectedDeliveryError) as info:
            call()
        assert not any(key in str(info.value) or key in repr(info.value.args) for key in keys)


# ---- cryptography is optional -------------------------------------------------------------------------------------------------------
def block_cryptography(monkeypatch) -> None:
    """Make every ``import cryptography...`` raise ImportError (a ``None`` entry in ``sys.modules``), including the
    submodules this process already imported."""

    for name in [m for m in sys.modules if m == "cryptography" or m.startswith("cryptography.")] + ["cryptography"]:
        monkeypatch.setitem(sys.modules, name, None)


def test_missing_cryptography_names_the_protected_extra(monkeypatch, vector_preview):
    block_cryptography(monkeypatch)
    assert pd.cryptography_available() is False
    for call in (lambda: open_preview_delivery(preview_doc(), now=PREVIEW_NOW),
                 lambda: open_detail_delivery(detail_doc(), preview=vector_preview, now=DETAIL_NOW)):
        err = refused(C.CRYPTO_UNAVAILABLE, call)
        assert isinstance(err, CryptographyUnavailableError) and 'pip install "os-optics[protected]"' in str(err)
    # the wrapper's structural checks still run first, without cryptography
    refused(C.ENVELOPE_MALFORMED, lambda: open_preview_delivery(detail_doc(), now=PREVIEW_NOW))


def test_the_contract_level_crypto_refusal_is_also_translated(monkeypatch):
    block_cryptography(monkeypatch)
    monkeypatch.setattr(pd, "cryptography_available", lambda: True)  # the probe passes, the contract's lazy import fails
    err = refused(C.CRYPTO_UNAVAILABLE, lambda: open_preview_delivery(preview_doc(), now=PREVIEW_NOW))
    assert isinstance(err, CryptographyUnavailableError) and "os-optics[protected]" in str(err)


BLOCKED_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, json, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError("blocked for this test")
            return None

    sys.meta_path.insert(0, Block())
    import os_optics
    import os_optics.local
    base = sorted(m for m in sys.modules if m.startswith(("os_optics.protected_delivery", "os_optics.client_artifacts",
                                                          "os_optics._vendor.shared_contracts.protected_assembly_delivery",
                                                          "os_optics._vendor.shared_contracts.client_assembly_artifact")))
    import os_optics.client_artifacts
    import os_optics.protected_delivery as pd
    print(json.dumps({"base": base, "cryptography": [m for m in sys.modules if m.split(".")[0] == "cryptography"],
                      "available": pd.cryptography_available()}))
    """
)


def test_the_engine_and_the_new_modules_import_without_cryptography(tmp_path):
    script = tmp_path / "blocked.py"
    script.write_text(BLOCKED_IMPORT_SCRIPT, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert json.loads(proc.stdout) == {"base": [], "cryptography": [], "available": False}
