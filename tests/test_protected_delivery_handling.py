"""Handling of protected delivery documents in memory (``ProtectedDelivery``, ``open_delivery``, ``redact_delivery``).

A delivery straight from an HTTP response body opens without being written anywhere; ``repr`` / ``str`` / ``redacted()``
never carry the grant's key; the object cannot be pickled or copied and can be discarded; the open functions keep the
raw document out of their own frames, so a traceback rendered with local variables does not show the key; the checks are
as strict as for a plain document.
"""

from __future__ import annotations

import copy
import json
import pickle
import traceback
from datetime import UTC, datetime, timedelta

import pytest

from delivery_helpers import sealed_reference_deliveries
from os_optics.protected_delivery import (
    REDACTED,
    ProtectedDelivery,
    ProtectedDeliveryDocumentError,
    ProtectedDeliveryError,
    ProtectedDeliveryErrorCode,
    VerifiedAssemblyDetailV1,
    open_delivery,
    open_detail_delivery,
    open_preview_delivery,
    redact_delivery,
)

C = ProtectedDeliveryErrorCode


@pytest.fixture(scope="module")
def pair():
    return sealed_reference_deliveries()


def keys_of(*documents) -> list[str]:
    return [d["grant"]["key"] for d in documents]


def frame_locals(error: BaseException) -> str:
    texts, tb = [], error.__traceback__
    while tb is not None:
        if "os_optics" in tb.tb_frame.f_code.co_filename:
            texts += [f"{name}={value!r}" for name, value in tb.tb_frame.f_locals.items()]
        tb = tb.tb_next
    return "\n".join(texts)


def test_a_response_body_opens_in_memory(pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    preview_doc, detail_doc = pair
    held = ProtectedDelivery.from_response(json.dumps(preview_doc).encode("utf-8"))
    preview = open_delivery(held)
    assert preview.manifest.binding.commercial_resolution_id == held.commercial_resolution_id
    assert open_preview_delivery(held).content_hash == preview.content_hash
    detail = open_delivery(json.dumps(detail_doc), preview=preview)
    assert isinstance(detail, VerifiedAssemblyDetailV1)
    assert open_detail_delivery(ProtectedDelivery(detail_doc), preview=preview).content_hash == detail.content_hash
    with pytest.raises(TypeError):
        open_delivery(detail_doc)  # a detail needs its preview
    assert list(tmp_path.iterdir()) == []


def test_repr_str_and_redacted_never_show_the_key(pair):
    for document in pair:
        held = ProtectedDelivery(document)
        texts = [repr(held), str(held), f"{held}", repr([held]), json.dumps(held.redacted()), json.dumps(redact_delivery(document))]
        for text in texts:
            assert document["grant"]["key"] not in text and document["envelope"]["ciphertext"] not in text
        redacted = held.redacted()
        assert redacted["grant"]["key"] == REDACTED and redacted["grant"]["grant_id"] == document["grant"]["grant_id"]
        assert redacted["envelope"]["header"] == document["envelope"]["header"]
        assert document["grant"]["key"] != REDACTED  # the caller's document is not changed


def test_a_held_delivery_cannot_be_pickled_copied_or_reused(pair):
    held = ProtectedDelivery(pair[0])
    for dump in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            dump(held)
    held.discard()
    assert held.discarded and pair[0]["grant"]["key"] not in repr(held)
    with pytest.raises(ProtectedDeliveryDocumentError) as refused:
        open_preview_delivery(held)
    assert refused.value.code is C.ENVELOPE_MALFORMED


def test_the_wrapper_is_as_strict_as_the_document_check(pair):
    preview_doc, detail_doc = pair
    with pytest.raises(ProtectedDeliveryDocumentError) as wrong_kind:
        open_preview_delivery(ProtectedDelivery(detail_doc))
    assert "open_detail_delivery" in str(wrong_kind.value)
    for bad in (b"not json", b"[]", json.dumps({**preview_doc, "extra": 1}), json.dumps({**preview_doc, "delivery": "X"})):
        with pytest.raises(ProtectedDeliveryDocumentError) as refused:
            ProtectedDelivery(bad)
        assert refused.value.code is C.ENVELOPE_MALFORMED
        assert preview_doc["grant"]["key"] not in str(refused.value)
    mismatch = {**preview_doc, "commercial_resolution_id": "00000000-0000-4000-8000-000000000999"}
    with pytest.raises(ProtectedDeliveryDocumentError) as other:
        open_delivery(ProtectedDelivery(mismatch))
    assert other.value.code is C.BINDING_MISMATCH


def test_a_refused_open_keeps_the_key_out_of_the_traceback_and_its_locals():
    stale_preview, stale_detail = sealed_reference_deliveries(now=datetime.now(UTC) - timedelta(hours=2))
    for opener in (lambda: open_preview_delivery(ProtectedDelivery(stale_preview)),
                   lambda: open_delivery(json.dumps(stale_preview).encode("utf-8"))):
        with pytest.raises(ProtectedDeliveryError) as expired:
            opener()
        error = expired.value
        assert error.code is C.GRANT_EXPIRED
        rendered = "".join(traceback.format_exception(error)) + frame_locals(error)
        assert "open_protected_assembly_preview" in rendered
        for key in keys_of(stale_preview, stale_detail):
            assert key not in rendered
