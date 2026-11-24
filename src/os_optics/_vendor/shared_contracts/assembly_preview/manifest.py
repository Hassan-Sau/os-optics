"""Assembly Preview Package manifest: identity, binding, usage terms and file digests.

The manifest (``manifest.json``) is the package's single identity record:

* ``binding`` ties the package to exactly one design / resolution: the Project hash (osc1, the same hash
  the user's Local Validation names), the Local Validation record hash, the commercial submission hash
  (when present), the resolution hash, and the Customer Purchase BOM hash (when
  one exists);
* ``usage`` states the terms of use of the package (``UsageTermsV1``);
* ``files`` lists every other file with its size and SHA-256;
* ``package_content_hash`` is the osc1 digest of the manifest payload without that field, so it covers
  the binding, the revision, the usage terms and every file digest.

The package is plain bytes addressed by relative paths. The protected delivery (``protected_assembly_delivery``)
wraps the archive bytes in an encrypted envelope bound to one Project and resolution; the inner
``package_content_hash`` stays the package identity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..base import StrictContractModel
from ..common.hashes import HashProfile, HashRef, Sha256Digest, osc1_digest, require_hash_profile
from ..common.identifiers import OpenSciId
from ..optical_design.local_validation import ValidatedProjectRefV1


ASSEMBLY_PREVIEW_MANIFEST_SCHEMA_ID: Literal["opensci.assembly_preview.manifest"] = "opensci.assembly_preview.manifest"
ASSEMBLY_PREVIEW_MANIFEST_SCHEMA_VERSION: Literal["3.0"] = "3.0"
PREVIEW_USAGE_STATEMENT: Literal[
    "Assembly data of this Project's resolved assembly, for viewing and checking the assembly. Not for redistribution."
] = "Assembly data of this Project's resolved assembly, for viewing and checking the assembly. Not for redistribution."

MANIFEST_PATH = "manifest.json"
SCENE_PATH = "scene.json"
JSON_MEDIA_TYPE: Literal["application/json"] = "application/json"
GLB_MEDIA_TYPE: Literal["model/gltf-binary"] = "model/gltf-binary"
MAX_PACKAGE_FILES = 257

RESOLUTION_HASH_SUBJECT = "assembly_resolution"
LOCAL_VALIDATION_HASH_SUBJECT = "local_validation_record"
COMMERCIAL_SUBMISSION_HASH_SUBJECT = "commercial_submission"
COMMERCIAL_BOM_HASH_SUBJECT = "customer_purchase_bom"

PackagePath = Annotated[
    str,
    StringConstraints(strict=True, max_length=80, pattern=r"^(scene\.json|geometry/[a-z][a-z0-9]*(-[a-z0-9]+)*\.glb)$"),
    Field(description="Relative path inside the package: 'scene.json' or 'geometry/<part_ref>.glb'."),
]


class PermittedUseV1(StrEnum):
    """What a delivered assembly package may be used for."""

    VIEW_ASSEMBLY = "VIEW_ASSEMBLY"
    VALIDATE_ASSEMBLY = "VALIDATE_ASSEMBLY"
    GENERATE_ASSEMBLY_DRAWING = "GENERATE_ASSEMBLY_DRAWING"


PREVIEW_PERMITTED_USES: tuple[PermittedUseV1, ...] = (PermittedUseV1.VIEW_ASSEMBLY, PermittedUseV1.VALIDATE_ASSEMBLY)


class UsageTermsV1(StrictContractModel):
    """Terms of use of an Assembly Preview Package: this Project's assembly, viewing and checking, no redistribution."""

    scope: Literal["THIS_PROJECT"] = "THIS_PROJECT"
    permitted_uses: tuple[PermittedUseV1, ...] = PREVIEW_PERMITTED_USES
    redistribution: Literal["NOT_PERMITTED"] = "NOT_PERMITTED"
    statement: Literal[
        "Assembly data of this Project's resolved assembly, for viewing and checking the assembly. Not for redistribution."
    ] = PREVIEW_USAGE_STATEMENT

    @model_validator(mode="after")
    def _complete(self) -> UsageTermsV1:
        if self.permitted_uses != PREVIEW_PERMITTED_USES:
            raise ValueError("permitted_uses of an assembly preview package are exactly (VIEW_ASSEMBLY, VALIDATE_ASSEMBLY)")
        return self


def _osc1_subject(value: HashRef | None, subject: str, field_name: str) -> HashRef | None:
    if value is None:
        return None
    require_hash_profile(value, HashProfile.OSC1, field_name=field_name)
    if value.subject != subject:
        raise ValueError(f"{field_name} subject must be {subject!r}")
    return value


class ResolutionRefV1(StrictContractModel):
    """The Assembly Resolution this package realizes, by its hash."""

    resolution_hash: HashRef

    @field_validator("resolution_hash")
    @classmethod
    def _hash(cls, value: HashRef) -> HashRef:
        return _osc1_subject(value, RESOLUTION_HASH_SUBJECT, "resolution_hash")  # type: ignore[return-value]


class CommercialBomRefV1(StrictContractModel):
    """The authoritative Customer Purchase BOM this package's ``commercial_line_ref`` values point into."""

    bom_hash: HashRef

    @field_validator("bom_hash")
    @classmethod
    def _hash(cls, value: HashRef) -> HashRef:
        return _osc1_subject(value, COMMERCIAL_BOM_HASH_SUBJECT, "bom_hash")  # type: ignore[return-value]


class PackageBindingV1(StrictContractModel):
    project: ValidatedProjectRefV1
    local_validation_hash: HashRef | None = None
    commercial_submission_hash: HashRef | None = None
    resolution: ResolutionRefV1
    commercial_bom: CommercialBomRefV1 | None = None
    order_ref: OpenSciId | None = None
    commercial_resolution_id: OpenSciId | None = Field(
        None, description="Id of the commercial resolution this package belongs to.")

    @field_validator("local_validation_hash")
    @classmethod
    def _lv(cls, value: HashRef | None) -> HashRef | None:
        return _osc1_subject(value, LOCAL_VALIDATION_HASH_SUBJECT, "local_validation_hash")

    @field_validator("commercial_submission_hash")
    @classmethod
    def _submission(cls, value: HashRef | None) -> HashRef | None:
        return _osc1_subject(value, COMMERCIAL_SUBMISSION_HASH_SUBJECT, "commercial_submission_hash")


class PackageFileV1(StrictContractModel):
    path: PackagePath
    media_type: Literal["application/json", "model/gltf-binary"]
    size_bytes: int = Field(ge=0)
    sha256: Sha256Digest

    @model_validator(mode="after")
    def _media(self) -> PackageFileV1:
        expected = JSON_MEDIA_TYPE if self.path.endswith(".json") else GLB_MEDIA_TYPE
        if self.media_type != expected:
            raise ValueError(f"{self.path}: media_type must be {expected!r}")
        return self


class AssemblyPreviewManifestV1(StrictContractModel):
    schema_id: Literal["opensci.assembly_preview.manifest"] = Field(ASSEMBLY_PREVIEW_MANIFEST_SCHEMA_ID, alias="schema")
    schema_version: Literal["3.0"] = ASSEMBLY_PREVIEW_MANIFEST_SCHEMA_VERSION
    package_revision: int = Field(ge=1, description="Revision of the package for this resolution (1 = first revision).")
    usage: UsageTermsV1 = UsageTermsV1()
    binding: PackageBindingV1
    files: tuple[PackageFileV1, ...] = Field(min_length=1, max_length=MAX_PACKAGE_FILES - 1)
    package_content_hash: Sha256Digest

    @model_validator(mode="after")
    def _check(self) -> AssemblyPreviewManifestV1:
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("manifest files must be unique and sorted by path")
        if SCENE_PATH not in paths:
            raise ValueError("manifest files must include scene.json")
        expected = content_hash_of(self.content_payload())
        if self.package_content_hash != expected:
            raise ValueError("package_content_hash does not match the manifest content")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def content_payload(self) -> dict[str, Any]:
        """The hashed part of the manifest: everything except ``package_content_hash``."""

        payload = self.payload()
        del payload["package_content_hash"]
        return payload


def content_hash_of(content_payload: dict[str, Any]) -> str:
    """``package_content_hash`` = osc1 digest of the manifest payload without that field."""

    return osc1_digest(content_payload)
