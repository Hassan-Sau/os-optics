"""Protected Assembly Preview envelope, decryption grant, seal and open.

Wire forms (JSON; binary fields are base64url without padding, canonical)::

    envelope = {"schema": "opensci.protected_assembly_delivery.envelope", "schema_version": "2.0",
                "header": "<header JSON text>", "ciphertext": "<AES-256-GCM ciphertext || 16-byte tag>"}
    grant    = {"schema": "opensci.protected_assembly_delivery.grant", ..., "key": "<32 bytes>", ...}

The header is carried as its exact JSON text, and the AEAD additional authenticated data is exactly those ASCII
bytes: a browser passes ``TextEncoder().encode(envelope.header)`` to WebCrypto and never re-serializes JSON, so no
cross-language canonicalization is involved. Everything a reader may see before decryption (ids, hashes, times,
nonce, usage) is inside the authenticated header, so changing any of it makes decryption fail.

Key and nonce: each call of ``seal_assembly_preview`` needs a fresh random key and nonce; the key exists only in the
grant.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, SecretStr, StringConstraints, field_validator, model_validator

from ..assembly_preview import (
    AssemblyPreviewPackageError,
    AssemblyPreviewPackageV1,
    UsageTermsV1,
    load_assembly_preview_package,
    read_package_archive,
)
from ..base import StrictContractModel
from ..common.hashes import Sha256Digest
from ._common import (  # noqa: F401 -- also re-exports the preview's public names now defined in _common
    CIPHER_ALGORITHM,
    CLOCK_SKEW_ALLOWANCE,
    DEFAULT_GRANT_LIFETIME,
    GRANT_SENSITIVITY,
    KEY_BYTES,
    MAX_GRANT_LIFETIME,
    MAX_HEADER_CHARS,
    MIN_GRANT_LIFETIME,
    NONCE_BYTES,
    SCHEMA_VERSION,
    TAG_BYTES,
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
    parse_utc,
)

ENVELOPE_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.envelope"] = "opensci.protected_assembly_delivery.envelope"
HEADER_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.header"] = "opensci.protected_assembly_delivery.header"
GRANT_SCHEMA_ID: Literal["opensci.protected_assembly_delivery.grant"] = "opensci.protected_assembly_delivery.grant"
PLAINTEXT_FORMAT: Literal["opensci.assembly_preview.archive"] = "opensci.assembly_preview.archive"
ARCHIVE_MEDIA_TYPE: Literal["application/zip"] = "application/zip"

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
_MAX_CIPHERTEXT_CHARS = max_ciphertext_chars(MAX_ARCHIVE_BYTES)

# Module-level aliases of the shared helpers.
_canonical_text = canonical_text
_check_lifetime = check_lifetime


# ------------------------------------------------------------------------------------------ contract


class ProtectedPlaintextV1(StrictContractModel):
    """What decrypts out: the Assembly Preview Package archive (``package_archive_bytes``) and its identity."""

    format: Literal["opensci.assembly_preview.archive"] = PLAINTEXT_FORMAT
    media_type: Literal["application/zip"] = ARCHIVE_MEDIA_TYPE
    size_bytes: int = Field(strict=True, ge=1, le=MAX_ARCHIVE_BYTES)
    sha256: Sha256Digest
    package_content_hash: Sha256Digest


class ProtectedEnvelopeHeaderV1(StrictContractModel):
    """Everything readable before decryption; all of it is authenticated as AAD."""

    schema_id: Literal["opensci.protected_assembly_delivery.header"] = Field(HEADER_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    envelope_id: Uuid
    cipher: CipherParamsV1
    plaintext: ProtectedPlaintextV1
    binding: DeliveryBindingV1
    issued_at: UtcSecond
    expires_at: UtcSecond
    usage: UsageTermsV1 = UsageTermsV1()

    @model_validator(mode="after")
    def _lifetime(self) -> ProtectedEnvelopeHeaderV1:
        check_lifetime(self.issued_at, self.expires_at)
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def text(self) -> str:
        """The canonical header text: sorted keys, no whitespace, ASCII."""

        return canonical_text(self.payload())


class ProtectedAssemblyPreviewEnvelopeV1(StrictContractModel):
    """The encrypted package. Safe to cache briefly; useless without the matching grant."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_id: Literal["opensci.protected_assembly_delivery.envelope"] = Field(ENVELOPE_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    header: HeaderText
    ciphertext: Annotated[B64Url, StringConstraints(max_length=_MAX_CIPHERTEXT_CHARS)]

    @model_validator(mode="after")
    def _check(self) -> ProtectedAssemblyPreviewEnvelopeV1:
        header = self.parsed_header()
        if header.text() != self.header:
            raise ValueError("header is not in canonical form")
        if not ciphertext_fits(self.ciphertext, header.plaintext.size_bytes):
            raise ValueError("ciphertext length does not match the header")
        return self

    def parsed_header(self) -> ProtectedEnvelopeHeaderV1:
        return ProtectedEnvelopeHeaderV1.model_validate_json(self.header)

    def aad(self) -> bytes:
        """The AEAD additional authenticated data: exactly the header text bytes."""

        return self.header.encode("ascii")

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def to_json_bytes(self) -> bytes:
        return canonical_text(self.payload()).encode("ascii")


class PreviewDecryptionGrantV1(StrictContractModel):
    """SECRET. The content key of exactly one envelope, bound to it and to its resolution. Conforming OpenSci
    runtimes refuse to open the envelope after ``expires_at``; that is policy enforcement, not cryptographic
    revocation: whoever keeps the envelope and this key can decrypt it at any time. Pass it to the user's runtime in
    memory; never log it or put it in a URL or file name, and do not keep it: if a workflow has to hand it over as a
    local file, delete that file once the envelope is opened.
    ``repr`` and ``model_dump`` mask the key; ``to_json_bytes`` is the one deliberate way to serialize it.

    Never use it as a request-body model of a web framework: ``hide_input_in_errors`` masks ``str()`` of a validation
    error, but ``ValidationError.errors()`` / ``.json()`` still carry the input, key included. Parse grants with
    ``parse_grant``, which raises ``GRANT_MALFORMED`` and nothing else."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_id: Literal["opensci.protected_assembly_delivery.grant"] = Field(GRANT_SCHEMA_ID, alias="schema")
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    grant_id: Uuid
    envelope_id: Uuid
    package_content_hash: Sha256Digest
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
    def _lifetime(self) -> PreviewDecryptionGrantV1:
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


def seal_assembly_preview(
    package: AssemblyPreviewPackageV1,
    *,
    key: bytes,
    nonce: bytes,
    envelope_id: str,
    grant_id: str,
    organization_id: str,
    issued_at: datetime,
    expires_at: datetime,
) -> tuple[ProtectedAssemblyPreviewEnvelopeV1, PreviewDecryptionGrantV1]:
    """Encrypt one verified package into an envelope + grant. ``key`` must be fresh random bytes used for this one
    envelope only (a key encrypts exactly one message, so a nonce can never repeat under it).

    The binding (resolution id, Project hash) is read from the package's own manifest, never from the caller."""

    check_key_and_nonce(key, nonce)
    archive = package.archive_bytes()
    # The exact bytes the client will receive must themselves be one valid package with this identity.
    verified = load_assembly_preview_package(read_package_archive(archive), expected_content_hash=package.content_hash)
    resolution_id = verified.manifest.binding.commercial_resolution_id
    if resolution_id is None:
        raise ValueError("only a package bound to a commercial resolution can be delivered")
    header = ProtectedEnvelopeHeaderV1(
        envelope_id=envelope_id,
        cipher=CipherParamsV1(nonce=b64url_encode(nonce)),
        plaintext=ProtectedPlaintextV1(size_bytes=len(archive), sha256=hashlib.sha256(archive).hexdigest(),
                                       package_content_hash=verified.content_hash),
        binding=DeliveryBindingV1(commercial_resolution_id=resolution_id,
                                  project_hash=verified.manifest.binding.project.project_hash.digest,
                                  organization_id=organization_id),
        issued_at=format_utc(issued_at),
        expires_at=format_utc(expires_at),
    )
    text = header.text()
    ciphertext = encrypt(key, nonce, archive, text.encode("ascii"))
    envelope = ProtectedAssemblyPreviewEnvelopeV1.model_validate(
        {"schema": ENVELOPE_SCHEMA_ID, "schema_version": SCHEMA_VERSION, "header": text, "ciphertext": b64url_encode(ciphertext)})
    grant = PreviewDecryptionGrantV1.model_validate({
        "schema": GRANT_SCHEMA_ID, "schema_version": SCHEMA_VERSION, "grant_id": grant_id, "envelope_id": envelope_id,
        "package_content_hash": header.plaintext.package_content_hash,
        "commercial_resolution_id": header.binding.commercial_resolution_id, "project_hash": header.binding.project_hash,
        "organization_id": header.binding.organization_id, "issued_at": header.issued_at, "expires_at": header.expires_at,
        "algorithm": CIPHER_ALGORITHM, "key": b64url_encode(key), "sensitivity": GRANT_SENSITIVITY,
    })
    return envelope, grant


def parse_envelope(value: ProtectedAssemblyPreviewEnvelopeV1 | Mapping[str, Any] | str | bytes) -> ProtectedAssemblyPreviewEnvelopeV1:
    return parse_model(ProtectedAssemblyPreviewEnvelopeV1, value, ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED)


def parse_grant(value: PreviewDecryptionGrantV1 | Mapping[str, Any] | str | bytes) -> PreviewDecryptionGrantV1:
    return parse_model(PreviewDecryptionGrantV1, value, ProtectedDeliveryErrorCode.GRANT_MALFORMED)


def _grant_fits(header: ProtectedEnvelopeHeaderV1, grant: PreviewDecryptionGrantV1) -> bool:
    return (grant.envelope_id, grant.package_content_hash, grant.commercial_resolution_id, grant.project_hash,
            grant.organization_id, grant.issued_at, grant.expires_at, grant.algorithm) == (
        header.envelope_id, header.plaintext.package_content_hash, header.binding.commercial_resolution_id,
        header.binding.project_hash, header.binding.organization_id, header.issued_at, header.expires_at,
        header.cipher.algorithm)


def open_protected_assembly_preview(
    envelope: ProtectedAssemblyPreviewEnvelopeV1 | Mapping[str, Any] | str | bytes,
    grant: PreviewDecryptionGrantV1 | Mapping[str, Any] | str | bytes,
    *,
    now: datetime,
    expected_resolution_id: str | None = None,
    expected_project_hash: str | None = None,
    expected_organization_id: str | None = None,
) -> AssemblyPreviewPackageV1:
    """Decrypt in memory and return the verified Assembly Preview Package, or raise ``ProtectedDeliveryError``.

    Success means all of: well-formed envelope and grant, grant bound to this envelope, the caller's expected
    resolution / Project / organization (when given), inside the validity window, AEAD authentication of ciphertext
    and header, the plaintext's SHA-256 and size, and ``verify_assembly_preview_package`` on the decrypted files with
    the header's Project hash, package content hash and resolution id. Nothing is written anywhere."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    parsed = parse_envelope(envelope)
    header = parsed.parsed_header()
    secret = parse_grant(grant)
    if not _grant_fits(header, secret):
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.GRANT_ENVELOPE_MISMATCH)
    binding = header.binding
    check_expected_binding(binding, resolution_id=expected_resolution_id, project_hash=expected_project_hash,
                           organization_id=expected_organization_id)
    check_window(header.issued_at, header.expires_at, now)
    data = decode_ciphertext(parsed.ciphertext)
    archive = decrypt(secret.key_bytes(), b64url_decode(header.cipher.nonce), data, parsed.aad())
    if len(archive) != header.plaintext.size_bytes or hashlib.sha256(archive).hexdigest() != header.plaintext.sha256:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.PACKAGE_HASH_MISMATCH)
    problems: tuple[str, ...] = ()
    try:
        package = load_assembly_preview_package(read_package_archive(archive), expected_project_hash=binding.project_hash,
                                                expected_content_hash=header.plaintext.package_content_hash)
    except AssemblyPreviewPackageError as exc:
        problems = exc.problems
    except Exception:  # noqa: BLE001 -- zipfile raises its own errors (bad CRC, header/name mismatch): same refusal
        problems = ("the archive could not be read",)
    if problems:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.PACKAGE_INVALID, problems=problems)
    if package.manifest.binding.commercial_resolution_id != binding.commercial_resolution_id:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.BINDING_MISMATCH)
    return package
