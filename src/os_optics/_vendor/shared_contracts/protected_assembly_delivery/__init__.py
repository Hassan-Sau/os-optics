"""Protected delivery of one Assembly Preview Package and its Assembly Detail Package.

A project/resolution-scoped AES-256-GCM envelope around the exact Assembly Preview Package archive, and a separate short-lived
decryption grant that carries the one-off content key::

    ProtectedAssemblyPreviewEnvelopeV1   header (authenticated, readable) + ciphertext; may be cached briefly
    PreviewDecryptionGrantV1             the per-envelope content key + the same binding; SECRET, short-lived

    seal_assembly_preview(package, key=..., nonce=..., ...)        server side: one fresh key per envelope
    open_protected_assembly_preview(envelope, grant, now=...)      client side: decrypt in memory, then verify

and, for the Assembly Detail Package (the same design, distinct schemas, never interchangeable)::

    ProtectedAssemblyDetailEnvelopeV1    detail header (authenticated) + ciphertext of the detail JSON
    DetailDecryptionGrantV1              the per-envelope content key; SECRET, short-lived

    seal_assembly_detail(detail, preview, key=..., nonce=..., ...)            server side
    open_protected_assembly_detail(envelope, grant, now=..., preview=...)      client side: decrypt, verify vs preview

Each envelope is bound to one Project and resolution and authenticated end to end: a modified envelope, a grant of
another envelope, or a grant past its expiry does not open. Every header carries the usage terms of its package.

The AEAD primitive is the ``cryptography`` package, imported lazily: parsing and binding checks work without it.
"""

from .detail_envelope import (
    DETAIL_ENVELOPE_SCHEMA_ID,
    DETAIL_GRANT_SCHEMA_ID,
    DETAIL_HEADER_SCHEMA_ID,
    DETAIL_MEDIA_TYPE,
    DETAIL_PLAINTEXT_FORMAT,
    DetailDecryptionGrantV1,
    ProtectedAssemblyDetailEnvelopeV1,
    ProtectedDetailEnvelopeHeaderV1,
    ProtectedDetailPlaintextV1,
    open_protected_assembly_detail,
    parse_detail_envelope,
    parse_detail_grant,
    seal_assembly_detail,
)
from .envelope import (
    ARCHIVE_MEDIA_TYPE,
    CIPHER_ALGORITHM,
    CLOCK_SKEW_ALLOWANCE,
    DEFAULT_GRANT_LIFETIME,
    ENVELOPE_SCHEMA_ID,
    GRANT_SCHEMA_ID,
    GRANT_SENSITIVITY,
    HEADER_SCHEMA_ID,
    KEY_BYTES,
    MAX_ARCHIVE_BYTES,
    MAX_GRANT_LIFETIME,
    MIN_GRANT_LIFETIME,
    NONCE_BYTES,
    PLAINTEXT_FORMAT,
    SCHEMA_VERSION,
    TAG_BYTES,
    CipherParamsV1,
    DeliveryBindingV1,
    PreviewDecryptionGrantV1,
    ProtectedAssemblyPreviewEnvelopeV1,
    ProtectedDeliveryError,
    ProtectedDeliveryErrorCode,
    ProtectedEnvelopeHeaderV1,
    ProtectedPlaintextV1,
    b64url_decode,
    b64url_encode,
    format_utc,
    open_protected_assembly_preview,
    parse_envelope,
    parse_grant,
    parse_utc,
    seal_assembly_preview,
)

__all__ = [
    "ARCHIVE_MEDIA_TYPE",
    "CIPHER_ALGORITHM",
    "CLOCK_SKEW_ALLOWANCE",
    "DEFAULT_GRANT_LIFETIME",
    "DETAIL_ENVELOPE_SCHEMA_ID",
    "DETAIL_GRANT_SCHEMA_ID",
    "DETAIL_HEADER_SCHEMA_ID",
    "DETAIL_MEDIA_TYPE",
    "DETAIL_PLAINTEXT_FORMAT",
    "ENVELOPE_SCHEMA_ID",
    "GRANT_SCHEMA_ID",
    "GRANT_SENSITIVITY",
    "HEADER_SCHEMA_ID",
    "KEY_BYTES",
    "MAX_ARCHIVE_BYTES",
    "MAX_GRANT_LIFETIME",
    "MIN_GRANT_LIFETIME",
    "NONCE_BYTES",
    "PLAINTEXT_FORMAT",
    "SCHEMA_VERSION",
    "TAG_BYTES",
    "CipherParamsV1",
    "DeliveryBindingV1",
    "DetailDecryptionGrantV1",
    "PreviewDecryptionGrantV1",
    "ProtectedAssemblyDetailEnvelopeV1",
    "ProtectedAssemblyPreviewEnvelopeV1",
    "ProtectedDeliveryError",
    "ProtectedDeliveryErrorCode",
    "ProtectedDetailEnvelopeHeaderV1",
    "ProtectedDetailPlaintextV1",
    "ProtectedEnvelopeHeaderV1",
    "ProtectedPlaintextV1",
    "b64url_decode",
    "b64url_encode",
    "format_utc",
    "open_protected_assembly_detail",
    "open_protected_assembly_preview",
    "parse_detail_envelope",
    "parse_detail_grant",
    "parse_envelope",
    "parse_grant",
    "parse_utc",
    "seal_assembly_detail",
    "seal_assembly_preview",
]
