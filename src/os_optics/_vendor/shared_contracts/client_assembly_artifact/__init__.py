"""The manifest of one client-generated assembly artifact set.

The user device computes (local assembly validation, detailed assembly drawing); OpenSci stores what it computed::

    client-local computation -> artifact files -> ClientAssemblyArtifactManifestV1 -> authorized upload
      -> server verifies bytes + binding -> durable, immutable artifact set

The manifest binds exactly one artifact set to one commercial resolution (Project hash, resolution id,
Customer Purchase BOM digest), to the opaque source packages / results it was generated from, to the generator
that produced it, and to the exact bytes of every file (role, canonical filename, media type, size, SHA-256).
``artifact_set_digest`` is the osc1 digest of the manifest without that field: the one structured identity of the
set. The organization is deliberately absent: the uploader's organization is taken from the authenticated request,
never from the document.

Deliberately bounded, not a file bucket: three artifact kinds, five file roles, one fixed filename and media type per
role, explicit per-role and per-set byte budgets.
"""

from .manifest import (
    CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_ID,
    CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    JSON_MEDIA_TYPE,
    KIND_RULES,
    MAX_ARTIFACT_FILES,
    MAX_ARTIFACT_SET_BYTES,
    MAX_JSON_BYTES,
    MAX_PDF_BYTES,
    MAX_SVG_BYTES,
    PDF_MEDIA_TYPE,
    ROLE_RULES,
    SVG_MEDIA_TYPE,
    ArtifactFileRoleV1,
    ArtifactFormatV1,
    ArtifactKindRule,
    ArtifactRoleRule,
    ClientArtifactBindingV1,
    ClientArtifactFileV1,
    ClientArtifactGeneratorV1,
    ClientArtifactKindV1,
    ClientArtifactSourceBindingV1,
    ClientAssemblyArtifactManifestV1,
    SourceBindingKindV1,
    artifact_set_digest_of,
    build_client_artifact_manifest,
)

__all__ = [
    "CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_ID",
    "CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "JSON_MEDIA_TYPE",
    "KIND_RULES",
    "MAX_ARTIFACT_FILES",
    "MAX_ARTIFACT_SET_BYTES",
    "MAX_JSON_BYTES",
    "MAX_PDF_BYTES",
    "MAX_SVG_BYTES",
    "PDF_MEDIA_TYPE",
    "ROLE_RULES",
    "SVG_MEDIA_TYPE",
    "ArtifactFileRoleV1",
    "ArtifactFormatV1",
    "ArtifactKindRule",
    "ArtifactRoleRule",
    "ClientArtifactBindingV1",
    "ClientArtifactFileV1",
    "ClientArtifactGeneratorV1",
    "ClientArtifactKindV1",
    "ClientArtifactSourceBindingV1",
    "ClientAssemblyArtifactManifestV1",
    "SourceBindingKindV1",
    "artifact_set_digest_of",
    "build_client_artifact_manifest",
]
