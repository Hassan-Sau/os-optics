"""Protected Assembly Detail envelope, decryption grant, seal and open.

The protected preview design (``envelope``), applied to the second protected document: the Assembly Detail Package
(``assembly_detail``), the bounded drawing supplement of exactly one verified Assembly Preview Package::

    detail envelope = {"schema": "opensci.protected_assembly_delivery.detail_envelope", "schema_version": "2.0",
                       "header": "<detail header JSON text>", "ciphertext": "<AES-256-GCM ciphertext || 16-byte tag>"}
    detail grant    = {"schema": "opensci.protected_assembly_delivery.detail_grant", ..., "key": "<32 bytes>", ...}

Exactly as for the preview: one fresh random 256-bit key and 96-bit nonce per envelope (a key encrypts one message),
the canonical header text as the AEAD additional authenticated data, a short-lived grant that is the only place the
key exists (``SecretStr``, ``hide_input_in_errors``), a lifetime of at most 30 minutes and never beyond the
authorization token, and the same client clock-skew allowance and strict base64url.

What differs, and why the two can never be confused:

* the plaintext is the detail's canonical file bytes (``detail_file_bytes``: ``application/json``, at most
  ``MAX_DETAIL_BYTES``), identified by its ``detail_content_hash`` AND the ``preview_package_content_hash`` of the one
  preview package it supplements. Opening needs that verified preview package: the decrypted detail is verified
  against it (``load_assembly_detail``), and a preview whose content hash is not the header's is refused before
  decryption (``PREVIEW_PACKAGE_MISMATCH``);
* the header names the ``release_basis`` of the detail (``ReleaseBasisV1``) and repeats the detail's own usage terms
  (``DetailUsageTermsV1``);
* every schema id differs from the preview's (``detail_envelope`` / ``detail_header`` / ``detail_grant``) and the
  models are closed, so a preview grant never parses as a detail grant, a detail envelope never parses as a preview
  envelope, and vice versa; the AAD (the header text) differs as well.

``cryptography`` is imported lazily.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, SecretStr, StringConstraints, field_validator, model_validator

from ..assembly_detail import (
    MAX_DETAIL_BYTES,
    AssemblyDetailPackageError,
    AssemblyDetailPackageV1,
    DetailUsageTermsV1,
    ReleaseBasisV1,
    VerifiedAssemblyDetailV1,
    detail_file_bytes,
    load_assembly_detail,
)
from ..assembly_preview import AssemblyPreviewPackageV1
from ..base import StrictContractModel
from ..common.hashes import Sha256Digest
from ._common import (
    CIPHER_ALGORITHM,
    GRANT_SENSITIVITY,
    KEY_BYTES,
    SCHEMA_VERSION,
    B64Url,
    CipherParamsV1,
    DeliveryBindingV1,
    HeaderText,
    ProtectedDeliveryError,
    ProtectedDeliveryErrorCode,
    UtcSecond,
    Uuid,
    b64url_decode,
    b64url_encode,
    canonical_text,
    check_expected_binding,
    check_key_and_nonce,
    check_lifetime,
    check_window,
    ciphertext_fits,
    decode_ciphertext,
    decrypt,
    encrypt,
    format_utc,
    max_ciphertext_chars,
    parse_model,
)

DETAIL_ENVELOPE_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.detail_envelope"] = (
    "opensci.protected_assembly_delivery.detail_envelope")
DETAIL_HEADER_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.detail_header"] = (
    "opensci.protected_assembly_delivery.detail_header")
DETAIL_GRANT_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.detail_grant"] = (
    "opensci.protected_assembly_delivery.detail_grant")
DETAIL_PLAINTEXT_FORMAT: Literal["opensci.assembly_detail.document"] = "opensci.assembly_detail.document"
DETAIL_MEDIA_TYPE: Literal["application/json"] = "application/json"

_MAX_DETAIL_CIPHERTEXT_CHARS = max_ciphertext_chars(MAX_DETAIL_BYTES)


# ------------------------------------------------------------------------------------------ contract


class ProtectedDetailPlaintextV1(StrictContractModel):
    """What decrypts out: the detail's canonical file bytes, and the one preview package they supplement."""

    format: Literal["opensci.assembly_detail.document"] = DETAIL_PLAINTEXT_FORMAT
    media_type: Literal["application/json"] = DETAIL_MEDIA_TYPE
    size_bytes: int = Field(strict=True, ge=1, le=MAX_DETAIL_BYTES)
    sha256: Sha256Digest
    detail_content_hash: Sha256Digest
    preview_package_content_hash: Sha256Digest


class ProtectedDetailEnvelopeHeaderV1(StrictContractModel):
    """Everything readable before decryption; all of it is authenticated as AAD."""

    schema_id: Literal["opensci.protected_assembly_delivery.detail_header"] = Field(DETAIL_HEADER_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    envelope_id: Uuid
    cipher: CipherParamsV1
    plaintext: ProtectedDetailPlaintextV1
    binding: DeliveryBindingV1
    release_basis: ReleaseBasisV1
    issued_at: UtcSecond
    expires_at: UtcSecond
    usage: DetailUsageTermsV1 = DetailUsageTermsV1()

    @model_validator(mode="after")
    def _lifetime(self) -> ProtectedDetailEnvelopeHeaderV1:
        check_lifetime(self.issued_at, self.expires_at)
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def text(self) -> str:
        """The canonical header text: sorted keys, no whitespace, ASCII."""

        return canonical_text(self.payload())


class ProtectedAssemblyDetailEnvelopeV1(StrictContractModel):
    """The encrypted detail. Safe to cache briefly; useless without the matching detail grant."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_id: Literal["opensci.protected_assembly_delivery.detail_envelope"] = Field(DETAIL_ENVELOPE_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    header: HeaderText
    ciphertext: Annotated[B64Url, StringConstraints(max_length=_MAX_DETAIL_CIPHERTEXT_CHARS)]

    @model_validator(mode="after")
    def _check(self) -> ProtectedAssemblyDetailEnvelopeV1:
        header = self.parsed_header()
        if header.text() != self.header:
            raise ValueError("header is not in canonical form")
        if not ciphertext_fits(self.ciphertext, header.plaintext.size_bytes):
            raise ValueError("ciphertext length does not match the header")
        return self

    def parsed_header(self) -> ProtectedDetailEnvelopeHeaderV1:
        return ProtectedDetailEnvelopeHeaderV1.model_validate_json(self.header)

    def aad(self) -> bytes:
        """The AEAD additional authenticated data: exactly the header text bytes."""

        return self.header.encode("ascii")

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def to_json_bytes(self) -> bytes:
        return canonical_text(self.payload()).encode("ascii")


class DetailDecryptionGrantV1(StrictContractModel):
    """SECRET. The content key of exactly one detail envelope, bound to it, to its detail and preview package and to
    its resolution. Same handling as ``PreviewDecryptionGrantV1``: conforming runtimes refuse to open after
    ``expires_at`` (policy enforcement, not cryptographic revocation); pass it in memory, never log it or put it in a
    URL or file name, and delete any local file it had to pass through once the envelope is opened. ``repr`` and
    ``model_dump`` mask the key; ``to_json_bytes`` is the one deliberate serializer; never use it as a request-body
    model (parse with ``parse_detail_grant``)."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_id: Literal["opensci.protected_assembly_delivery.detail_grant"] = Field(DETAIL_GRANT_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    grant_id: Uuid
    envelope_id: Uuid
    detail_content_hash: Sha256Digest
    preview_package_content_hash: Sha256Digest
    commercial_resolution_id: Uuid
    project_hash: Sha256Digest
    organization_id: Uuid
    issued_at: UtcSecond
    expires_at: UtcSecond
    algorithm: Literal["AES-256-GCM"] = CIPHER_ALGORITHM
    key: SecretStr = Field(description="Base64url of the 32-byte AES-256-GCM content key. SECRET.")
    sensitivity: Literal["SECRET_SHORT_LIVED_DECRYPTION_KEY"] = GRANT_SENSITIVITY

    @field_validator("key")
    @classmethod
    def _key(cls, value: SecretStr) -> SecretStr:
        try:
            size = len(b64url_decode(value.get_secret_value()))
        except ValueError:
            raise ValueError("key is not canonical base64url") from None
        if size != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes")
        return value

    @model_validator(mode="after")
    def _lifetime(self) -> DetailDecryptionGrantV1:
        check_lifetime(self.issued_at, self.expires_at)
        return self

    def key_bytes(self) -> bytes:
        return b64url_decode(self.key.get_secret_value())

    def to_json_bytes(self) -> bytes:
        """SENSITIVE: the wire form, key included."""

        payload = self.model_dump(mode="json", by_alias=True)
        payload["key"] = self.key.get_secret_value()
        return canonical_text(payload).encode("ascii")


# ------------------------------------------------------------------------------------------ seal / open


def seal_assembly_detail(
    detail: AssemblyDetailPackageV1,
    preview: AssemblyPreviewPackageV1,
    *,
    key: bytes,
    nonce: bytes,
    envelope_id: str,
    grant_id: str,
    organization_id: str,
    issued_at: datetime,
    expires_at: datetime,
) -> tuple[ProtectedAssemblyDetailEnvelopeV1, DetailDecryptionGrantV1]:
    """Encrypt one detail package, verified against its preview package, into an envelope + grant. ``key`` must be
    fresh random bytes used for this one envelope only.

    The binding (resolution id, Project hash, preview package content hash) and the release basis are read from the
    verified detail and preview, never from the caller; only the organization, ids and times are the caller's."""

    check_key_and_nonce(key, nonce)
    plaintext = detail_file_bytes(detail)
    # The exact bytes the client will receive must themselves verify against this preview package.
    verified = load_assembly_detail(plaintext, preview, expected_content_hash=detail.detail_content_hash)
    binding = verified.detail.binding
    header = ProtectedDetailEnvelopeHeaderV1(
        envelope_id=envelope_id,
        cipher=CipherParamsV1(nonce=b64url_encode(nonce)),
        plaintext=ProtectedDetailPlaintextV1(size_bytes=len(plaintext), sha256=hashlib.sha256(plaintext).hexdigest(),
                                             detail_content_hash=verified.content_hash,
                                             preview_package_content_hash=preview.content_hash),
        binding=DeliveryBindingV1(commercial_resolution_id=binding.commercial_resolution_id,
                                  project_hash=binding.project.project_hash.digest, organization_id=organization_id),
        release_basis=verified.detail.release.basis,
        issued_at=format_utc(issued_at),
        expires_at=format_utc(expires_at),
    )
    text = header.text()
    ciphertext = encrypt(key, nonce, plaintext, text.encode("ascii"))
    envelope = ProtectedAssemblyDetailEnvelopeV1.model_validate({
        "schema": DETAIL_ENVELOPE_SCHEMA_ID, "schema_version": SCHEMA_VERSION, "header": text,
        "ciphertext": b64url_encode(ciphertext)})
    grant = DetailDecryptionGrantV1.model_validate({
        "schema": DETAIL_GRANT_SCHEMA_ID, "schema_version": SCHEMA_VERSION, "grant_id": grant_id, "envelope_id": envelope_id,
        "detail_content_hash": header.plaintext.detail_content_hash,
        "preview_package_content_hash": header.plaintext.preview_package_content_hash,
        "commercial_resolution_id": header.binding.commercial_resolution_id, "project_hash": header.binding.project_hash,
        "organization_id": header.binding.organization_id, "issued_at": header.issued_at, "expires_at": header.expires_at,
        "algorithm": CIPHER_ALGORITHM, "key": b64url_encode(key), "sensitivity": GRANT_SENSITIVITY,
    })
    return envelope, grant


def parse_detail_envelope(
    value: ProtectedAssemblyDetailEnvelopeV1 | Mapping[str, Any] | str | bytes,
) -> ProtectedAssemblyDetailEnvelopeV1:
    return parse_model(ProtectedAssemblyDetailEnvelopeV1, value, ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED)


def parse_detail_grant(value: DetailDecryptionGrantV1 | Mapping[str, Any] | str | bytes) -> DetailDecryptionGrantV1:
    return parse_model(DetailDecryptionGrantV1, value, ProtectedDeliveryErrorCode.GRANT_MALFORMED)


def _grant_fits(header: ProtectedDetailEnvelopeHeaderV1, grant: DetailDecryptionGrantV1) -> bool:
    return (grant.envelope_id, grant.detail_content_hash, grant.preview_package_content_hash,
            grant.commercial_resolution_id, grant.project_hash, grant.organization_id, grant.issued_at, grant.expires_at,
            grant.algorithm) == (
        header.envelope_id, header.plaintext.detail_content_hash, header.plaintext.preview_package_content_hash,
        header.binding.commercial_resolution_id, header.binding.project_hash, header.binding.organization_id,
        header.issued_at, header.expires_at, header.cipher.algorithm)


def open_protected_assembly_detail(
    envelope: ProtectedAssemblyDetailEnvelopeV1 | Mapping[str, Any] | str | bytes,
    grant: DetailDecryptionGrantV1 | Mapping[str, Any] | str | bytes,
    *,
    now: datetime,
    preview: AssemblyPreviewPackageV1,
    expected_resolution_id: str | None = None,
    expected_project_hash: str | None = None,
    expected_organization_id: str | None = None,
) -> VerifiedAssemblyDetailV1:
    """Decrypt in memory and return the detail verified against ``preview``, or raise ``ProtectedDeliveryError``.

    ``preview`` is the verified Assembly Preview Package the client already opened (``open_protected_assembly_preview``
    or a local package verified by ``verify_assembly_preview_package``). Success means all of: well-formed envelope and grant, grant bound to this
    envelope, the caller's expected resolution / Project / organization (when given), inside the validity window,
    ``preview`` being the package the header names, AEAD authentication of ciphertext and header, the plaintext's
    SHA-256 and size, ``load_assembly_detail`` against ``preview`` with the header's detail content hash, and the
    detail naming the header's resolution and Project. Nothing is written anywhere."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    parsed = parse_detail_envelope(envelope)
    header = parsed.parsed_header()
    secret = parse_detail_grant(grant)
    if not _grant_fits(header, secret):
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.GRANT_ENVELOPE_MISMATCH)
    binding = header.binding
    check_expected_binding(binding, resolution_id=expected_resolution_id, project_hash=expected_project_hash,
                           organization_id=expected_organization_id)
    check_window(header.issued_at, header.expires_at, now)
    if not isinstance(preview, AssemblyPreviewPackageV1) or preview.content_hash != header.plaintext.preview_package_content_hash:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.PREVIEW_PACKAGE_MISMATCH)
    data = decode_ciphertext(parsed.ciphertext)
    plaintext = decrypt(secret.key_bytes(), b64url_decode(header.cipher.nonce), data, parsed.aad())
    if len(plaintext) != header.plaintext.size_bytes or hashlib.sha256(plaintext).hexdigest() != header.plaintext.sha256:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.PACKAGE_HASH_MISMATCH)
    try:
        verified = load_assembly_detail(plaintext, preview, expected_content_hash=header.plaintext.detail_content_hash)
    except AssemblyDetailPackageError as exc:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.DETAIL_INVALID, problems=exc.problems) from None
    detail_binding = verified.detail.binding
    if (detail_binding.commercial_resolution_id, detail_binding.project.project_hash.digest) != (
            binding.commercial_resolution_id, binding.project_hash):
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.BINDING_MISMATCH)
    if verified.detail.release.basis is not header.release_basis:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.DETAIL_INVALID,
                                     problems=("the detail's release basis is not the envelope's",))
    return verified
