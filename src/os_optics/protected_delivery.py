"""Open a protected Assembly Preview / Assembly Detail delivery on this device.

After a commercial resolution, the OpenSci Commercial API delivers the Assembly Preview Package and (once the Order
is paid and confirmed) the Assembly Detail Package as an encrypted envelope together with a short-lived decryption
grant. This module opens such a delivery in memory and hands back the verified packages the rest of the Engine
consumes::

    from os_optics.protected_delivery import open_detail_delivery, open_preview_delivery

    preview = open_preview_delivery(preview_delivery_json)                    # AssemblyPreviewPackageV1, verified
    detail = open_detail_delivery(detail_delivery_json, preview=preview)      # VerifiedAssemblyDetailV1, verified vs preview

    from os_optics.assembly import validate_assembly_layout
    from os_optics.assembly_drawing import generate_assembly_drawing
    validation = validate_assembly_layout(preview)
    drawing = generate_assembly_drawing(preview, detail)

``delivery`` is the Commercial API delivery document, as a mapping or its JSON text / bytes, of exactly this shape::

    {"schema": "opensci.commercial_api.protected_delivery", "schema_version": "1",
     "delivery": "ASSEMBLY_PREVIEW" | "ASSEMBLY_DETAIL",
     "commercial_resolution_id": "<the resolution the envelope is bound to>",
     "envelope": {<the envelope JSON object>}, "grant": {<the decryption grant JSON object, key included>}}

The wrapper is checked strictly (exactly these keys, this schema and version, the delivery kind the function opens,
the resolution id equal to the envelope's authenticated binding); then the delivery contract does everything
else: grant bound to the envelope, the caller's
``expected_*`` resolution / Project / organization, the validity window at ``now`` (default: the current UTC time),
AEAD authentication of ciphertext and header, the plaintext's size and SHA-256, and the canonical package verifier.
Every refusal is a ``ProtectedDeliveryError`` with a stable ``ProtectedDeliveryErrorCode``; its message never carries
the key, the plaintext or input values.

Nothing is written anywhere: the decrypted packages exist only in the returned objects. Keep the grant out of logs,
files, error reports and URLs; a grant opens its envelope until its ``expires_at``. Each package states its usage terms
(``preview.manifest.usage``, ``detail.detail.usage``): this Project's assembly, not for redistribution. The Detailed
Assembly Drawing generated from them is yours to keep and use.

Decryption needs PyCA ``cryptography``, which is the optional extra ``pip install "os-optics[protected]"``. It is
imported only when a delivery is opened; ``import os_optics`` and ``os_optics.local`` never need it. Without
it, opening raises ``CryptographyUnavailableError`` (code ``CRYPTO_UNAVAILABLE``) naming the extra.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ._vendor.shared_contracts.assembly_detail import VerifiedAssemblyDetailV1
from ._vendor.shared_contracts.assembly_preview import AssemblyPreviewPackageV1
from ._vendor.shared_contracts.protected_assembly_delivery import (
    ProtectedDeliveryError,
    ProtectedDeliveryErrorCode,
    open_protected_assembly_detail,
    open_protected_assembly_preview,
    parse_detail_envelope,
    parse_detail_grant,
    parse_envelope,
    parse_grant,
)

DELIVERY_SCHEMA_ID = "opensci.commercial_api.protected_delivery"
DELIVERY_SCHEMA_VERSION = "1"
PREVIEW_DELIVERY = "ASSEMBLY_PREVIEW"
DETAIL_DELIVERY = "ASSEMBLY_DETAIL"
DELIVERY_KEYS = frozenset({"schema", "schema_version", "delivery", "commercial_resolution_id", "envelope", "grant"})
PROTECTED_EXTRA = "os-optics[protected]"
#: What ``repr`` / ``redacted()`` show instead of a grant's key.
REDACTED = "<redacted>"


class ProtectedDeliveryDocumentError(ProtectedDeliveryError):
    """The Commercial API delivery document around the envelope and grant is not the documented shape.

    ``code`` is ``ENVELOPE_MALFORMED`` (not a delivery document of the kind this function opens) or
    ``BINDING_MISMATCH`` (the document names another resolution than the envelope's authenticated binding);
    ``reason`` is a fixed text that never carries input values."""

    def __init__(self, code: ProtectedDeliveryErrorCode, reason: str) -> None:
        super().__init__(code)
        self.reason = reason
        ValueError.__init__(self, f"{self.code}: {self.message} ({reason})")


class CryptographyUnavailableError(ProtectedDeliveryError):
    """``CRYPTO_UNAVAILABLE``: the optional AES-GCM dependency is not installed in this environment."""

    def __init__(self) -> None:
        super().__init__(ProtectedDeliveryErrorCode.CRYPTO_UNAVAILABLE)
        ValueError.__init__(self, f"{self.code}: {self.message}; install it with: pip install \"{PROTECTED_EXTRA}\"")


def cryptography_available() -> bool:
    """Whether protected deliveries can be opened here (the ``protected`` extra is installed)."""

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401 -- availability probe only
    except ImportError:
        return False
    return True


def _document(delivery: Any, kind: str | None) -> dict[str, Any]:
    """The checked delivery document (``kind`` ``None``: either delivery kind). Refusals never carry input values."""

    malformed = ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED
    if isinstance(delivery, ProtectedDelivery):
        document = delivery._held()
        if kind is not None and document["delivery"] != kind:
            raise _wrong_kind(kind, document["delivery"])
        return document
    if isinstance(delivery, (str, bytes, bytearray)):
        try:
            delivery = json.loads(delivery)
        except (ValueError, RecursionError):
            raise ProtectedDeliveryDocumentError(malformed, "the delivery document is not JSON") from None
    if not isinstance(delivery, Mapping):
        raise ProtectedDeliveryDocumentError(malformed, "the delivery document is not a JSON object")
    document = dict(delivery)
    if set(document) != DELIVERY_KEYS:
        raise ProtectedDeliveryDocumentError(malformed, "the delivery document must have exactly the keys "
                                             + ", ".join(sorted(DELIVERY_KEYS)))
    if document["schema"] != DELIVERY_SCHEMA_ID or document["schema_version"] != DELIVERY_SCHEMA_VERSION:
        raise ProtectedDeliveryDocumentError(malformed, f"the delivery document is not {DELIVERY_SCHEMA_ID} "
                                             f"version {DELIVERY_SCHEMA_VERSION}")
    if kind is None and document["delivery"] not in (PREVIEW_DELIVERY, DETAIL_DELIVERY):
        raise ProtectedDeliveryDocumentError(malformed, f"not an {PREVIEW_DELIVERY} or {DETAIL_DELIVERY} delivery")
    if kind is not None and document["delivery"] != kind:
        raise _wrong_kind(kind, document["delivery"])
    if not isinstance(document["commercial_resolution_id"], str):
        raise ProtectedDeliveryDocumentError(malformed, "commercial_resolution_id is not a string")
    if not isinstance(document["envelope"], Mapping) or not isinstance(document["grant"], Mapping):
        raise ProtectedDeliveryDocumentError(malformed, "envelope and grant must be JSON objects")
    return document


def _wrong_kind(kind: str, found: Any) -> ProtectedDeliveryDocumentError:
    other = "open_detail_delivery" if kind == PREVIEW_DELIVERY else "open_preview_delivery"
    known = found in (PREVIEW_DELIVERY, DETAIL_DELIVERY)
    return ProtectedDeliveryDocumentError(ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED,
                                          f"not an {kind} delivery" + (f"; open it with {other}" if known else ""))


def _redacted_grant(grant: Any) -> Any:
    if not isinstance(grant, Mapping):
        return REDACTED
    return {name: (REDACTED if name == "key" else value) for name, value in grant.items()}


def redact_delivery(delivery: Any) -> dict[str, Any]:
    """A copy of a delivery document that is safe to log or show: the grant's ``key`` is replaced by ``<redacted>`` and
    the envelope's ciphertext by its length. Raises ``ProtectedDeliveryDocumentError`` for anything that is not a
    delivery document."""

    document = _document(delivery, None)
    envelope = dict(document["envelope"])
    if isinstance(envelope.get("ciphertext"), str):
        envelope["ciphertext"] = f"<{len(envelope['ciphertext'])} characters>"
    return {**document, "envelope": envelope, "grant": _redacted_grant(document["grant"])}


class ProtectedDelivery:
    """One Commercial API delivery document held in memory, e.g. straight from an HTTP response body::

        delivery = ProtectedDelivery(response_body)        # bytes, text or a mapping; checked like open_* does
        package = open_delivery(delivery)                   # or open_preview_delivery / open_detail_delivery
        delivery.discard()                                  # drop the document once it is opened

    ``repr`` and ``str`` show only the delivery kind and resolution id; the grant never appears there, and the object
    cannot be pickled or copied. ``redacted()`` is a loggable copy. Nothing is written anywhere."""

    __slots__ = ("_document", "commercial_resolution_id", "delivery")

    def __init__(self, document: Any) -> None:
        held = _document(document, None)
        self._document: dict[str, Any] | None = held
        self.delivery: str = held["delivery"]
        self.commercial_resolution_id: str = held["commercial_resolution_id"]

    @classmethod
    def from_response(cls, body: bytes | bytearray | str) -> ProtectedDelivery:
        """The delivery of an HTTP response body (the JSON bytes of a ``201`` delivery response)."""

        return cls(body)

    def _held(self) -> dict[str, Any]:
        if self._document is None:
            raise ProtectedDeliveryDocumentError(ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED,
                                                 "the delivery was discarded")
        return self._document

    @property
    def discarded(self) -> bool:
        return self._document is None

    def discard(self) -> None:
        """Drop the held document (envelope and grant). Opening it afterwards raises ``ENVELOPE_MALFORMED``."""

        self._document = None

    def redacted(self) -> dict[str, Any]:
        """``redact_delivery`` of the held document."""

        return redact_delivery(self._held())

    def __repr__(self) -> str:
        state = "discarded" if self._document is None else f"grant={REDACTED}"
        return (f"ProtectedDelivery(delivery={self.delivery!r}, commercial_resolution_id={self.commercial_resolution_id!r}, "
                f"{state})")

    __str__ = __repr__

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("a ProtectedDelivery cannot be pickled or copied")


DeliveryDocument = ProtectedDelivery | Mapping[str, Any] | str | bytes | bytearray


def _holder(delivery: Any, kind: str) -> ProtectedDelivery:
    held = delivery if isinstance(delivery, ProtectedDelivery) else ProtectedDelivery(delivery)
    if held._held()["delivery"] != kind:
        raise _wrong_kind(kind, held.delivery)
    return held


def _check_resolution(held: ProtectedDelivery, binding_resolution_id: str) -> None:
    if held.commercial_resolution_id != binding_resolution_id:
        raise ProtectedDeliveryDocumentError(ProtectedDeliveryErrorCode.BINDING_MISMATCH,
                                             "commercial_resolution_id differs from the envelope binding")


def _open(call: Any) -> Any:
    if not cryptography_available():
        raise CryptographyUnavailableError()
    try:
        return call()
    except ProtectedDeliveryError as exc:
        if exc.code is ProtectedDeliveryErrorCode.CRYPTO_UNAVAILABLE:
            raise CryptographyUnavailableError() from None
        raise


# The open functions hold the document only inside a ProtectedDelivery and the grant only parsed (both mask the key in
# ``repr``), so a traceback rendered with local variables does not show the key.


def open_preview_delivery(
    delivery: DeliveryDocument,
    *,
    now: datetime | None = None,
    expected_resolution_id: str | None = None,
    expected_project_hash: str | None = None,
    expected_organization_id: str | None = None,
) -> AssemblyPreviewPackageV1:
    """Open an ``ASSEMBLY_PREVIEW`` delivery document in memory and return the verified Assembly Preview Package.

    ``now`` (timezone-aware; default: the current UTC time) is checked against the grant's validity window. The
    ``expected_*`` values, when given, pin the envelope's binding. Raises ``ProtectedDeliveryError``."""

    held = _holder(delivery, PREVIEW_DELIVERY)
    delivery = held
    envelope = parse_envelope(held._held()["envelope"])
    _check_resolution(held, envelope.parsed_header().binding.commercial_resolution_id)
    grant = parse_grant(held._held()["grant"])
    moment = datetime.now(UTC) if now is None else now
    return _open(lambda: open_protected_assembly_preview(
        envelope, grant, now=moment, expected_resolution_id=expected_resolution_id,
        expected_project_hash=expected_project_hash, expected_organization_id=expected_organization_id))


def open_detail_delivery(
    delivery: DeliveryDocument,
    *,
    preview: AssemblyPreviewPackageV1,
    now: datetime | None = None,
    expected_resolution_id: str | None = None,
    expected_project_hash: str | None = None,
    expected_organization_id: str | None = None,
) -> VerifiedAssemblyDetailV1:
    """Open an ``ASSEMBLY_DETAIL`` delivery document in memory and return the detail verified against ``preview``.

    ``preview`` is the verified Assembly Preview Package the detail supplements (``open_preview_delivery`` or
    ``os_optics.assembly.load_assembly_preview``). ``detail_bytes`` of the result is what
    ``os_optics.assembly_drawing.generate_assembly_drawing`` takes. Raises ``ProtectedDeliveryError``."""

    held = _holder(delivery, DETAIL_DELIVERY)
    delivery = held
    envelope = parse_detail_envelope(held._held()["envelope"])
    _check_resolution(held, envelope.parsed_header().binding.commercial_resolution_id)
    grant = parse_detail_grant(held._held()["grant"])
    moment = datetime.now(UTC) if now is None else now
    return _open(lambda: open_protected_assembly_detail(
        envelope, grant, now=moment, preview=preview, expected_resolution_id=expected_resolution_id,
        expected_project_hash=expected_project_hash, expected_organization_id=expected_organization_id))


def open_delivery(
    delivery: DeliveryDocument,
    *,
    preview: AssemblyPreviewPackageV1 | None = None,
    now: datetime | None = None,
    expected_resolution_id: str | None = None,
    expected_project_hash: str | None = None,
    expected_organization_id: str | None = None,
) -> AssemblyPreviewPackageV1 | VerifiedAssemblyDetailV1:
    """Open either delivery kind: ``open_preview_delivery`` for ``ASSEMBLY_PREVIEW``, ``open_detail_delivery`` (which
    needs ``preview``) for ``ASSEMBLY_DETAIL``. ``delivery`` may be the raw HTTP response body. Raises
    ``ProtectedDeliveryError``."""

    held = delivery if isinstance(delivery, ProtectedDelivery) else ProtectedDelivery(delivery)
    delivery = held
    pins = {"now": now, "expected_resolution_id": expected_resolution_id, "expected_project_hash": expected_project_hash,
            "expected_organization_id": expected_organization_id}
    if held.delivery == PREVIEW_DELIVERY:
        return open_preview_delivery(held, **pins)
    if preview is None:
        raise TypeError("opening an ASSEMBLY_DETAIL delivery needs preview=<the verified Assembly Preview Package>")
    return open_detail_delivery(held, preview=preview, **pins)


__all__ = [
    "DELIVERY_SCHEMA_ID",
    "DELIVERY_SCHEMA_VERSION",
    "DETAIL_DELIVERY",
    "PREVIEW_DELIVERY",
    "PROTECTED_EXTRA",
    "REDACTED",
    "AssemblyPreviewPackageV1",
    "CryptographyUnavailableError",
    "DeliveryDocument",
    "ProtectedDelivery",
    "ProtectedDeliveryDocumentError",
    "ProtectedDeliveryError",
    "ProtectedDeliveryErrorCode",
    "VerifiedAssemblyDetailV1",
    "cryptography_available",
    "open_delivery",
    "open_detail_delivery",
    "open_preview_delivery",
    "redact_delivery",
]
