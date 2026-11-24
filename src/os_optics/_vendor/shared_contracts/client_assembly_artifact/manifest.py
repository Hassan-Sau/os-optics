"""``ClientAssemblyArtifactManifestV1``: identity, binding and exact file digests of one client artifact set.

Kinds, roles and budgets:

=============================  ====================================================  ===============================
kind                           files (all required)                                  source bindings (all required)
=============================  ====================================================  ===============================
ASSEMBLY_LOCAL_VALIDATION      assembly_local_validation.json                        ASSEMBLY_PREVIEW_PACKAGE,
                                                                                     ASSEMBLY_LOCAL_VALIDATION
ASSEMBLY_DRAWING               drawing_manifest.json, assembly_drawing.svg,          ASSEMBLY_PREVIEW_PACKAGE,
                               assembly_drawing.pdf, drawing_result.json             ASSEMBLY_DETAIL_PACKAGE,
                                                                                     ASSEMBLY_LOCAL_VALIDATION
ASSEMBLY_DRAWING_DIAGNOSTIC    as ASSEMBLY_DRAWING                                   as ASSEMBLY_DRAWING
=============================  ====================================================  ===============================

The drawing kinds are exactly the four outputs of the local drawing generator (``shared_contracts.assembly_detail``:
``AssemblyDrawingManifestV1`` / ``AssemblyDrawingResultV1``), which always writes all four; a diagnostic drawing
(``DIAGNOSTIC_VALIDATION_FAILED`` or ``DIAGNOSTIC_VALIDATION_INCOMPLETE``) is the same file set. Every drawing is generated from one preview package,
its detail package, and the local assembly validation the generator ran, so all three bindings are required.

A source binding is ``(kind, digest)`` and nothing else: the digest is the source's own content identity
(``package_content_hash`` of an Assembly Preview Package, ``detail_content_hash`` of an Assembly Detail Package,
``result_hash`` of an ``AssemblyLocalValidationV1``).

Byte budgets: JSON 2 MiB, SVG 4 MiB, PDF 16 MiB per file; 32 MiB per set. A declared size outside its budget is
refused while the manifest is parsed, before any file byte is read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from ..base import StrictContractModel
from ..common.hashes import Sha256Digest, osc1_digest
from ..common.identifiers import BuildVersion, OpenSciId


CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_ID: Literal["opensci.client_assembly_artifact_manifest"] = (
    "opensci.client_assembly_artifact_manifest"
)
CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_VERSION: Literal["1.0"] = "1.0"

JSON_MEDIA_TYPE: Literal["application/json"] = "application/json"
SVG_MEDIA_TYPE: Literal["image/svg+xml"] = "image/svg+xml"
PDF_MEDIA_TYPE: Literal["application/pdf"] = "application/pdf"

MIB = 1024 * 1024
MAX_JSON_BYTES = 2 * MIB
MAX_SVG_BYTES = 4 * MIB
MAX_PDF_BYTES = 16 * MIB
MAX_ARTIFACT_SET_BYTES = 32 * MIB

CanonicalUuid = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
    Field(description="Lowercase canonical UUID text."),
]


class ClientArtifactKindV1(StrEnum):
    """What the client generated: one value per kind of assembly artifact."""

    ASSEMBLY_LOCAL_VALIDATION = "ASSEMBLY_LOCAL_VALIDATION"  # AssemblyLocalValidationV1
    ASSEMBLY_DRAWING = "ASSEMBLY_DRAWING"  # detailed assembly drawing (SVG / PDF + its manifest / result)
    ASSEMBLY_DRAWING_DIAGNOSTIC = "ASSEMBLY_DRAWING_DIAGNOSTIC"  # diagnostic drawing: local validation FAILED or incomplete


class SourceBindingKindV1(StrEnum):
    """What an artifact set was generated from. The digest is that source's own content identity."""

    ASSEMBLY_PREVIEW_PACKAGE = "ASSEMBLY_PREVIEW_PACKAGE"  # preview manifest.package_content_hash
    ASSEMBLY_DETAIL_PACKAGE = "ASSEMBLY_DETAIL_PACKAGE"  # AssemblyDetailPackageV1.detail_content_hash
    ASSEMBLY_LOCAL_VALIDATION = "ASSEMBLY_LOCAL_VALIDATION"  # AssemblyLocalValidationV1.result_hash


class ArtifactFileRoleV1(StrEnum):
    VALIDATION_RESULT = "VALIDATION_RESULT"
    DRAWING_MANIFEST = "DRAWING_MANIFEST"
    DRAWING_SVG = "DRAWING_SVG"
    DRAWING_PDF = "DRAWING_PDF"
    DRAWING_RESULT = "DRAWING_RESULT"


class ArtifactFormatV1(StrEnum):
    """The file format (``application/json`` / ``image/svg+xml`` / ``application/pdf``)."""

    JSON = "JSON"
    SVG = "SVG"
    PDF = "PDF"


@dataclass(frozen=True)
class ArtifactRoleRule:
    filename: str
    media_type: str
    format: ArtifactFormatV1
    max_bytes: int


@dataclass(frozen=True)
class ArtifactKindRule:
    required_roles: frozenset[ArtifactFileRoleV1]
    optional_roles: frozenset[ArtifactFileRoleV1]
    #: every one of these must be bound, and nothing else may be
    required_sources: frozenset[SourceBindingKindV1]

    @property
    def roles(self) -> frozenset[ArtifactFileRoleV1]:
        return self.required_roles | self.optional_roles


_JSON, _SVG, _PDF = ArtifactFormatV1.JSON, ArtifactFormatV1.SVG, ArtifactFormatV1.PDF
R, S = ArtifactFileRoleV1, SourceBindingKindV1

#: The exact filename, media type, format and byte budget of every role. Nothing else is ever accepted.
ROLE_RULES: dict[ArtifactFileRoleV1, ArtifactRoleRule] = {
    R.VALIDATION_RESULT: ArtifactRoleRule("assembly_local_validation.json", JSON_MEDIA_TYPE, _JSON, MAX_JSON_BYTES),
    R.DRAWING_MANIFEST: ArtifactRoleRule("drawing_manifest.json", JSON_MEDIA_TYPE, _JSON, MAX_JSON_BYTES),
    R.DRAWING_SVG: ArtifactRoleRule("assembly_drawing.svg", SVG_MEDIA_TYPE, _SVG, MAX_SVG_BYTES),
    R.DRAWING_PDF: ArtifactRoleRule("assembly_drawing.pdf", PDF_MEDIA_TYPE, _PDF, MAX_PDF_BYTES),
    R.DRAWING_RESULT: ArtifactRoleRule("drawing_result.json", JSON_MEDIA_TYPE, _JSON, MAX_JSON_BYTES),
}
MAX_ARTIFACT_FILES = len(ROLE_RULES)

_DRAWING = ArtifactKindRule(
    required_roles=frozenset({R.DRAWING_MANIFEST, R.DRAWING_SVG, R.DRAWING_PDF, R.DRAWING_RESULT}),
    optional_roles=frozenset(),
    required_sources=frozenset({S.ASSEMBLY_PREVIEW_PACKAGE, S.ASSEMBLY_DETAIL_PACKAGE, S.ASSEMBLY_LOCAL_VALIDATION}),
)

KIND_RULES: dict[ClientArtifactKindV1, ArtifactKindRule] = {
    ClientArtifactKindV1.ASSEMBLY_LOCAL_VALIDATION: ArtifactKindRule(
        required_roles=frozenset({R.VALIDATION_RESULT}),
        optional_roles=frozenset(),
        required_sources=frozenset({S.ASSEMBLY_PREVIEW_PACKAGE, S.ASSEMBLY_LOCAL_VALIDATION}),
    ),
    ClientArtifactKindV1.ASSEMBLY_DRAWING: _DRAWING,
    ClientArtifactKindV1.ASSEMBLY_DRAWING_DIAGNOSTIC: _DRAWING,
}
del R, S


class ClientArtifactSourceBindingV1(StrictContractModel):
    kind: SourceBindingKindV1
    digest: Sha256Digest


class ClientArtifactBindingV1(StrictContractModel):
    """The commercial resolution the set belongs to, and what it was generated from; each value must match that
    resolution."""

    project_hash: Sha256Digest
    commercial_resolution_id: CanonicalUuid
    customer_bom_digest: Sha256Digest
    sources: tuple[ClientArtifactSourceBindingV1, ...] = Field(min_length=1, max_length=len(SourceBindingKindV1))

    @model_validator(mode="after")
    def _sorted(self) -> ClientArtifactBindingV1:
        kinds = [s.kind.value for s in self.sources]
        if kinds != sorted(set(kinds)):
            raise ValueError("sources must be unique by kind and sorted by kind")
        return self

    def source(self, kind: SourceBindingKindV1) -> ClientArtifactSourceBindingV1 | None:
        return next((s for s in self.sources if s.kind is kind), None)


class ClientArtifactGeneratorV1(StrictContractModel):
    """The client software that produced the files."""

    generator_id: OpenSciId
    generator_version: BuildVersion
    build_id: BuildVersion | None = None


class ClientArtifactFileV1(StrictContractModel):
    role: ArtifactFileRoleV1
    filename: str = Field(max_length=64, description="The role's fixed file name (for example assembly_drawing.svg); never a path.")
    media_type: Literal["application/json", "image/svg+xml", "application/pdf"]
    size_bytes: int = Field(ge=1, le=MAX_PDF_BYTES)
    sha256: Sha256Digest

    @model_validator(mode="after")
    def _role(self) -> ClientArtifactFileV1:
        rule = ROLE_RULES[self.role]
        if self.filename != rule.filename:
            raise ValueError(f"{self.role.value}: filename must be {rule.filename!r}")
        if self.media_type != rule.media_type:
            raise ValueError(f"{self.role.value}: media_type must be {rule.media_type!r}")
        if self.size_bytes > rule.max_bytes:
            raise ValueError(f"{self.role.value}: size_bytes exceeds the {rule.max_bytes}-byte budget")
        return self


class ClientAssemblyArtifactManifestV1(StrictContractModel):
    """One client-generated artifact set: kind, binding, generator, exact files, and its content identity."""

    schema_id: Literal["opensci.client_assembly_artifact_manifest"] = Field(
        CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_ID, alias="schema")
    schema_version: Literal["1.0"] = CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_VERSION
    artifact_kind: ClientArtifactKindV1
    binding: ClientArtifactBindingV1
    generator: ClientArtifactGeneratorV1
    files: tuple[ClientArtifactFileV1, ...] = Field(min_length=1, max_length=MAX_ARTIFACT_FILES)
    locale: Literal["en", "zh-CN"] | None = Field(
        None, description="A drawing set: the language its sheet is written in (its drawing_result.json locale).")
    artifact_set_digest: Sha256Digest = Field(description="osc1 digest of this manifest without artifact_set_digest.")

    @model_validator(mode="after")
    def _check(self) -> ClientAssemblyArtifactManifestV1:
        names = [f.filename for f in self.files]
        if names != sorted(set(names)):
            raise ValueError("files must be unique and sorted by filename")
        rule = KIND_RULES[self.artifact_kind]
        roles = {f.role for f in self.files}
        if missing := rule.required_roles - roles:
            raise ValueError(f"{self.artifact_kind.value} requires file role(s) {sorted(r.value for r in missing)}")
        if extra := roles - rule.roles:
            raise ValueError(f"{self.artifact_kind.value} does not accept file role(s) {sorted(r.value for r in extra)}")
        if sum(f.size_bytes for f in self.files) > MAX_ARTIFACT_SET_BYTES:
            raise ValueError(f"the artifact set exceeds the {MAX_ARTIFACT_SET_BYTES}-byte budget")
        sources = {s.kind for s in self.binding.sources}
        if extra_sources := sources - rule.required_sources:
            raise ValueError(f"{self.artifact_kind.value} does not accept source binding(s) "
                             f"{sorted(s.value for s in extra_sources)}")
        if missing_sources := rule.required_sources - sources:
            raise ValueError(f"{self.artifact_kind.value} requires source binding(s) "
                             f"{sorted(s.value for s in missing_sources)}")
        if self.locale is not None and self.artifact_kind is ClientArtifactKindV1.ASSEMBLY_LOCAL_VALIDATION:
            raise ValueError("only a drawing artifact set names the language of its sheet")
        if self.artifact_set_digest != artifact_set_digest_of(self.content_payload()):
            raise ValueError("artifact_set_digest does not match the manifest content")
        return self

    def payload(self) -> dict[str, Any]:
        """Canonical serialized form (``None`` fields omitted)."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def content_payload(self) -> dict[str, Any]:
        """The hashed part of the manifest: everything except ``artifact_set_digest``."""

        payload = self.payload()
        del payload["artifact_set_digest"]
        return payload

    def file(self, role: ArtifactFileRoleV1) -> ClientArtifactFileV1 | None:
        return next((f for f in self.files if f.role is role), None)


def artifact_set_digest_of(content_payload: dict[str, Any]) -> str:
    """``artifact_set_digest`` = osc1 digest of the manifest payload without that field."""

    return osc1_digest(content_payload)


def build_client_artifact_manifest(
    artifact_kind: ClientArtifactKindV1 | str,
    *,
    project_hash: str,
    commercial_resolution_id: str,
    customer_bom_digest: str,
    sources: Mapping[SourceBindingKindV1 | str, str],
    generator: ClientArtifactGeneratorV1 | Mapping[str, Any],
    files: Mapping[str, bytes],
    locale: str | None = None,
) -> ClientAssemblyArtifactManifestV1:
    """Client-side helper: the manifest of ``files`` (canonical filename -> exact bytes) for one artifact kind (a drawing
    set also names the language of its sheet, ``locale``).

    The role of each file is looked up by its canonical filename; sizes and SHA-256 digests are computed from the
    bytes. This is how a local runtime (for example the local drawing generator) wraps its outputs for upload.
    """

    kind = ClientArtifactKindV1(artifact_kind)
    by_name = {rule.filename: role for role, rule in ROLE_RULES.items() if role in KIND_RULES[kind].roles}
    entries = []
    for filename in sorted(files):
        role = by_name.get(filename)
        if role is None:
            raise ValueError(f"{filename!r} is not a file of a {kind.value} artifact set")
        data = files[filename]
        entries.append({"role": role.value, "filename": filename, "media_type": ROLE_RULES[role].media_type,
                        "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    content = {
        "schema": CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_ID,
        "schema_version": CLIENT_ASSEMBLY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": kind.value,
        "binding": {
            "project_hash": project_hash,
            "commercial_resolution_id": commercial_resolution_id,
            "customer_bom_digest": customer_bom_digest,
            "sources": [{"kind": SourceBindingKindV1(k).value, "digest": d}
                        for k, d in sorted(sources.items(), key=lambda item: SourceBindingKindV1(item[0]).value)],
        },
        "generator": ClientArtifactGeneratorV1.model_validate(generator).model_dump(mode="json", exclude_none=True),
        "files": entries,
        **({"locale": locale} if locale is not None else {}),
    }
    return ClientAssemblyArtifactManifestV1.model_validate({**content, "artifact_set_digest": artifact_set_digest_of(content)})
