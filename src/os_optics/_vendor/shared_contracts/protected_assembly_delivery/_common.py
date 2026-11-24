"""The generic pieces shared by every protected-delivery envelope (Assembly Preview and Assembly Detail).

Use the names exported by ``envelope`` (preview) and ``detail_envelope`` (detail).

* strict base64url, whole-second UTC text, the canonical header text (sorted keys, no whitespace, ASCII);
* the grant lifetime policy and the client clock-skew allowance;
* ``CipherParamsV1`` and ``DeliveryBindingV1`` (the non-secret resolution / Project / organization binding);
* the fail-closed error codes and ``ProtectedDeliveryError``;
* AES-256-GCM through the maintained ``cryptography`` package, imported lazily (never at module import) so the
  contract parses and checks bindings without it.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, ValidationError, field_validator

from ..base import StrictContractModel
from ..common.hashes import Sha256Digest

SCHEMA_VERSION: Literal["2.0"] = "2.0"
CIPHER_ALGORITHM: Literal["AES-256-GCM"] = "AES-256-GCM"
GRANT_SENSITIVITY: Literal["SECRET_SHORT_LIVED_DECRYPTION_KEY"] = "SECRET_SHORT_LIVED_DECRYPTION_KEY"

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
MAX_HEADER_CHARS = 4096

#: Grant lifetime policy. The server also caps a grant at its authorization token's own expiry.
DEFAULT_GRANT_LIFETIME = timedelta(minutes=10)
MAX_GRANT_LIFETIME = timedelta(minutes=30)
MIN_GRANT_LIFETIME = timedelta(seconds=30)
#: A client clock may run behind the server's by this much when a grant is first used.
CLOCK_SKEW_ALLOWANCE = timedelta(minutes=5)

_UUID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_B64URL = re.compile(r"^[A-Za-z0-9_-]*$")

Uuid = Annotated[str, StringConstraints(strict=True, pattern=_UUID), Field(description="Lowercase canonical UUID.")]
UtcSecond = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
    Field(description="UTC instant, whole seconds: YYYY-MM-DDTHH:MM:SSZ."),
]
B64Url = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]*$"),
                   Field(description="Base64url (RFC 4648 section 5) without padding.")]
HeaderText = Annotated[str, StringConstraints(strict=True, min_length=2, max_length=MAX_HEADER_CHARS, pattern=r"^[\x20-\x7e]+$")]


def max_ciphertext_chars(max_plaintext_bytes: int) -> int:
    """Longest unpadded base64url text of ``max_plaintext_bytes`` plus the tag."""

    return -(-(max_plaintext_bytes + TAG_BYTES) * 4 // 3)


# ------------------------------------------------------------------------------------------ encodings


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    """Strict: base64url alphabet, no padding, and the canonical encoding of its bytes (no stray trailing bits)."""

    if not isinstance(text, str) or not _B64URL.match(text) or len(text) % 4 == 1:
        raise ValueError("not canonical base64url")
    data = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    if b64url_encode(data) != text:
        raise ValueError("not canonical base64url")
    return data


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, _UTC_FORMAT).replace(tzinfo=UTC)


def format_utc(moment: datetime) -> str:
    """Whole-second UTC text of a timezone-aware instant (sub-second part dropped)."""

    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime(_UTC_FORMAT)


def canonical_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def check_lifetime(issued_at: str, expires_at: str) -> None:
    lifetime = parse_utc(expires_at) - parse_utc(issued_at)
    if not timedelta(0) < lifetime <= MAX_GRANT_LIFETIME:
        raise ValueError(f"lifetime must be positive and at most {int(MAX_GRANT_LIFETIME.total_seconds())} s")


def ciphertext_fits(ciphertext: str, plaintext_size: int) -> bool:
    return len(ciphertext) % 4 != 1 and len(ciphertext) * 3 // 4 == plaintext_size + TAG_BYTES


# ------------------------------------------------------------------------------------------ shared models


class CipherParamsV1(StrictContractModel):
    algorithm: Literal["AES-256-GCM"] = CIPHER_ALGORITHM
    nonce: B64Url
    tag_bits: Literal[128] = 128

    @field_validator("nonce")
    @classmethod
    def _nonce(cls, value: str) -> str:
        if len(b64url_decode(value)) != NONCE_BYTES:
            raise ValueError(f"nonce must be {NONCE_BYTES} bytes")
        return value


class DeliveryBindingV1(StrictContractModel):
    """The one resolution / Project / organization this envelope belongs to (non-secret identifiers only)."""

    commercial_resolution_id: Uuid
    project_hash: Sha256Digest
    organization_id: Uuid


# ------------------------------------------------------------------------------------------ errors


class ProtectedDeliveryErrorCode(StrEnum):
    ENVELOPE_MALFORMED = "ENVELOPE_MALFORMED"
    GRANT_MALFORMED = "GRANT_MALFORMED"
    GRANT_ENVELOPE_MISMATCH = "GRANT_ENVELOPE_MISMATCH"
    BINDING_MISMATCH = "BINDING_MISMATCH"
    GRANT_EXPIRED = "GRANT_EXPIRED"
    DECRYPTION_FAILED = "DECRYPTION_FAILED"
    PACKAGE_HASH_MISMATCH = "PACKAGE_HASH_MISMATCH"
    PACKAGE_INVALID = "PACKAGE_INVALID"
    CRYPTO_UNAVAILABLE = "CRYPTO_UNAVAILABLE"
    # protected Assembly Detail delivery
    PREVIEW_PACKAGE_MISMATCH = "PREVIEW_PACKAGE_MISMATCH"
    DETAIL_INVALID = "DETAIL_INVALID"


_MESSAGES = {
    ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED: "the protected package envelope is malformed",
    ProtectedDeliveryErrorCode.GRANT_MALFORMED: "the decryption grant is malformed",
    ProtectedDeliveryErrorCode.GRANT_ENVELOPE_MISMATCH: "the decryption grant does not belong to this envelope",
    ProtectedDeliveryErrorCode.BINDING_MISMATCH: "the protected package belongs to another project or resolution",
    ProtectedDeliveryErrorCode.GRANT_EXPIRED: "the decryption grant is expired or not yet valid",
    ProtectedDeliveryErrorCode.DECRYPTION_FAILED: "the protected package could not be decrypted",
    ProtectedDeliveryErrorCode.PACKAGE_HASH_MISMATCH: "the decrypted package does not match the envelope",
    ProtectedDeliveryErrorCode.PACKAGE_INVALID: "the decrypted package is not a valid assembly preview package",
    ProtectedDeliveryErrorCode.CRYPTO_UNAVAILABLE: "the 'cryptography' package is required to open protected packages",
    ProtectedDeliveryErrorCode.PREVIEW_PACKAGE_MISMATCH: "the assembly detail belongs to another assembly preview package",
    ProtectedDeliveryErrorCode.DETAIL_INVALID: "the decrypted document is not a valid assembly detail package of this preview",
}


class ProtectedDeliveryError(ValueError):
    """A fail-closed refusal to open a protected package. The message is fixed per code and never carries key,
    plaintext or input values; ``problems`` lists the preview package verifier's findings for ``PACKAGE_INVALID`` (and the
    detail verifier's findings for ``DETAIL_INVALID``) only."""

    def __init__(self, code: ProtectedDeliveryErrorCode, *, problems: tuple[str, ...] = ()) -> None:
        self.code = ProtectedDeliveryErrorCode(code)
        self.message = _MESSAGES[self.code]
        self.problems = problems
        super().__init__(f"{self.code}: {self.message}")


# ------------------------------------------------------------------------------------------ AEAD


def aesgcm(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:  # pragma: no cover - exercised only where cryptography is absent
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.CRYPTO_UNAVAILABLE) from None
    return AESGCM(key)


def encrypt(key: bytes, nonce: bytes, data: bytes, aad: bytes) -> bytes:
    return aesgcm(key).encrypt(nonce, data, aad)


def decrypt(key: bytes, nonce: bytes, data: bytes, aad: bytes) -> bytes:
    cipher = aesgcm(key)
    from cryptography.exceptions import InvalidTag

    try:
        return cipher.decrypt(nonce, data, aad)
    except InvalidTag:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.DECRYPTION_FAILED) from None


def check_key_and_nonce(key: Any, nonce: Any) -> None:
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes")
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
        raise ValueError(f"nonce must be {NONCE_BYTES} bytes")


# ------------------------------------------------------------------------------------------ open steps


def parse_model(model: type[Any], value: Any, code: ProtectedDeliveryErrorCode) -> Any:
    if isinstance(value, model):
        return value
    try:
        if isinstance(value, (str, bytes, bytearray)):
            return model.model_validate_json(value)
        if isinstance(value, Mapping):
            return model.model_validate(dict(value))
    except (ValidationError, ValueError):
        pass
    raise ProtectedDeliveryError(code) from None


def check_expected_binding(binding: DeliveryBindingV1, *, resolution_id: str | None, project_hash: str | None,
                           organization_id: str | None) -> None:
    for expected, actual in ((resolution_id, binding.commercial_resolution_id), (project_hash, binding.project_hash),
                             (organization_id, binding.organization_id)):
        if expected is not None and expected != actual:
            raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.BINDING_MISMATCH)


def check_window(issued_at: str, expires_at: str, now: datetime) -> None:
    if not parse_utc(issued_at) - CLOCK_SKEW_ALLOWANCE <= now < parse_utc(expires_at):
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.GRANT_EXPIRED)


def decode_ciphertext(ciphertext: str) -> bytes:
    try:
        return b64url_decode(ciphertext)
    except ValueError:
        raise ProtectedDeliveryError(ProtectedDeliveryErrorCode.ENVELOPE_MALFORMED) from None
