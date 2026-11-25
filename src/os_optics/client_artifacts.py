"""Wrap what this device computed into an upload for the OpenSci client artifact store.

The device computes (the local assembly validation, the Detailed Assembly Drawing); OpenSci stores what it computed
as an immutable artifact set, bound to one commercial resolution. Each set is described by a
``ClientAssemblyArtifactManifestV1`` (the artifact manifest contract):
kind, binding (Project hash, resolution id, Customer Purchase BOM digest, source digests), generator, and the exact
size and SHA-256 of every file; ``artifact_set_digest`` is its osc1 identity. The helpers below take every value from
the real Engine outputs, never from the caller::

    from os_optics.client_artifacts import drawing_artifact_upload, validation_artifact_upload

    upload = validation_artifact_upload(preview, validation)   # ASSEMBLY_LOCAL_VALIDATION: assembly_local_validation.json
    upload = drawing_artifact_upload(drawing)                   # ASSEMBLY_DRAWING: the four drawing files, unchanged
    upload.manifest.artifact_set_digest
    upload.manifest_json_bytes()                                # the manifest document the server accepts
    upload.multipart_fields()                                   # [("manifest", ...), ("file", ...), ...] for HTTP

The HTTP upload is ``multipart/form-data`` with exactly one ``manifest`` part (``application/json``) and one ``file``
part per manifest file whose filename is the manifest filename; ``multipart_fields()`` returns exactly that, in the
``(name, (filename, bytes, media_type))`` form common HTTP clients accept. The organization is never in the manifest:
the server takes it from the authenticated request. This module does no I/O and no network access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ._vendor.shared_contracts.assembly_detail import DrawingStatusV1
from ._vendor.shared_contracts.assembly_preview import AssemblyLocalValidationV1, AssemblyPreviewPackageV1, json_file_bytes
from ._vendor.shared_contracts.client_assembly_artifact import (
    ClientArtifactKindV1,
    ClientAssemblyArtifactManifestV1,
    SourceBindingKindV1,
    build_client_artifact_manifest,
)
from .assembly import PackageSource, load_assembly_preview, result_file_bytes
from .assembly_drawing import AssemblyDrawing

MANIFEST_PART = "manifest"
FILE_PART = "file"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_MEDIA_TYPE = "application/json"
VALIDATION_FILENAME = "assembly_local_validation.json"

#: The drawing statuses each drawing artifact kind carries (the server refuses any other pairing).
DRAWING_KIND_STATUS: dict[ClientArtifactKindV1, frozenset[DrawingStatusV1]] = {
    ClientArtifactKindV1.ASSEMBLY_DRAWING: frozenset({DrawingStatusV1.DETAILED_ASSEMBLY_DRAWING}),
    ClientArtifactKindV1.ASSEMBLY_DRAWING_DIAGNOSTIC: frozenset({DrawingStatusV1.DIAGNOSTIC_VALIDATION_FAILED,
                                                                 DrawingStatusV1.DIAGNOSTIC_VALIDATION_INCOMPLETE}),
}


def drawing_kind_for(status: DrawingStatusV1) -> ClientArtifactKindV1:
    """The artifact kind a drawing of ``status`` is uploaded as."""

    return next(kind for kind, statuses in DRAWING_KIND_STATUS.items() if status in statuses)


@dataclass(frozen=True)
class ClientArtifactUpload:
    """One artifact set ready to upload: its manifest and the exact bytes of every file it lists."""

    manifest: ClientAssemblyArtifactManifestV1
    files: dict[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        files = {str(name): bytes(data) for name, data in self.files.items()}
        listed = {entry.filename: entry for entry in self.manifest.files}
        if set(files) != set(listed):
            raise ValueError("the files are not exactly the manifest's files")
        for name, entry in listed.items():
            if len(files[name]) != entry.size_bytes or hashlib.sha256(files[name]).hexdigest() != entry.sha256:
                raise ValueError(f"{name}: bytes do not match the manifest")
        object.__setattr__(self, "files", files)

    @property
    def artifact_kind(self) -> ClientArtifactKindV1:
        return self.manifest.artifact_kind

    @property
    def artifact_set_digest(self) -> str:
        return self.manifest.artifact_set_digest

    def manifest_payload(self) -> dict[str, Any]:
        """The manifest document (``model_dump(mode="json", by_alias=True, exclude_none=True)``, its canonical form)."""

        return self.manifest.payload()

    def manifest_json_bytes(self) -> bytes:
        """The manifest document as strict UTF-8 JSON (sorted keys, LF): what the upload's ``manifest`` part carries."""

        return json_file_bytes(self.manifest_payload())

    def multipart_fields(self) -> list[tuple[str, tuple[str, bytes, str]]]:
        """``[("manifest", ("manifest.json", <manifest bytes>, "application/json")), ("file", (filename, bytes,
        media_type)), ...]``: the multipart/form-data parts of the upload, files in manifest order."""

        parts = [(MANIFEST_PART, (MANIFEST_FILENAME, self.manifest_json_bytes(), MANIFEST_MEDIA_TYPE))]
        parts += [(FILE_PART, (entry.filename, self.files[entry.filename], entry.media_type)) for entry in self.manifest.files]
        return parts


def drawing_artifact_upload(drawing: AssemblyDrawing,
                            *, kind: ClientArtifactKindV1 | str = ClientArtifactKindV1.ASSEMBLY_DRAWING) -> ClientArtifactUpload:
    """The upload of one generated drawing (``os_optics.assembly_drawing.generate_assembly_drawing``).

    ``kind`` is ``ASSEMBLY_DRAWING`` for a ``DETAILED_ASSEMBLY_DRAWING`` (the default) or
    ``ASSEMBLY_DRAWING_DIAGNOSTIC`` for a ``DIAGNOSTIC_VALIDATION_FAILED`` or ``DIAGNOSTIC_VALIDATION_INCOMPLETE`` one
    (``drawing_kind_for``); any other pairing is refused
    (``ValueError``). The binding, the three source digests (preview package, detail package, the local assembly
    validation the generator ran) and the generator all come from the drawing's own result; the four files are uploaded
    unchanged."""

    if not isinstance(drawing, AssemblyDrawing):
        raise TypeError("drawing must be the AssemblyDrawing returned by generate_assembly_drawing")
    artifact_kind = ClientArtifactKindV1(kind)
    statuses = DRAWING_KIND_STATUS.get(artifact_kind)
    if statuses is None:
        raise ValueError(f"{artifact_kind.value} is not a drawing artifact kind; use one of "
                         + ", ".join(k.value for k in DRAWING_KIND_STATUS))
    result = drawing.result
    if result.drawing_status not in statuses:
        raise ValueError(f"a {result.drawing_status.value} drawing cannot be uploaded as {artifact_kind.value}; upload it as "
                         f"{drawing_kind_for(result.drawing_status).value}")
    binding = result.binding
    manifest = build_client_artifact_manifest(
        artifact_kind,
        project_hash=binding.project.project_hash.digest,
        commercial_resolution_id=binding.commercial_resolution_id,
        customer_bom_digest=binding.commercial_bom.bom_hash.digest,
        sources={SourceBindingKindV1.ASSEMBLY_PREVIEW_PACKAGE: binding.preview_package_content_hash,
                 SourceBindingKindV1.ASSEMBLY_DETAIL_PACKAGE: binding.detail_content_hash,
                 SourceBindingKindV1.ASSEMBLY_LOCAL_VALIDATION: result.validation.result_hash},
        generator={"generator_id": result.generator.generator_id, "generator_version": result.generator.generator_version,
                   "build_id": result.generator.engine_build_id},
        files=dict(drawing.files),
        locale=result.locale,  # the set names its sheet's language (two languages of one drawing are told apart)
    )
    return ClientArtifactUpload(manifest=manifest, files=dict(drawing.files))


def validation_artifact_upload(preview: PackageSource, result: AssemblyLocalValidationV1) -> ClientArtifactUpload:
    """The ``ASSEMBLY_LOCAL_VALIDATION`` upload of one local assembly validation result
    (``os_optics.assembly.validate_assembly_layout``, or ``AssemblyDrawing.validation``) over ``preview``, the
    verified package it checked.

    ``preview`` is anything ``os_optics.assembly.load_assembly_preview`` accepts (it is verified again). The
    binding comes from the package manifest (Project hash, resolution id, Customer Purchase BOM digest), the sources
    are the package content hash and the result hash, the generator is the result's checker, and the one file is the
    result's exact ``assembly_local_validation.json`` bytes. A result of another package, or a package not bound to a
    commercial resolution and its Customer Purchase BOM, is refused (``ValueError``)."""

    if not isinstance(result, AssemblyLocalValidationV1):
        raise TypeError("result must be the AssemblyLocalValidationV1 returned by validate_assembly_layout")
    package = preview if isinstance(preview, AssemblyPreviewPackageV1) else load_assembly_preview(preview)
    if result.package.package_content_hash != package.content_hash:
        raise ValueError("the validation result is of another assembly preview package")
    binding = package.manifest.binding
    if binding.commercial_resolution_id is None or binding.commercial_bom is None:
        raise ValueError("the assembly preview package is not bound to a commercial resolution and its Customer "
                         "Purchase BOM")
    files = {VALIDATION_FILENAME: result_file_bytes(result)}
    manifest = build_client_artifact_manifest(
        ClientArtifactKindV1.ASSEMBLY_LOCAL_VALIDATION,
        project_hash=binding.project.project_hash.digest,
        commercial_resolution_id=binding.commercial_resolution_id,
        customer_bom_digest=binding.commercial_bom.bom_hash.digest,
        sources={SourceBindingKindV1.ASSEMBLY_PREVIEW_PACKAGE: package.content_hash,
                 SourceBindingKindV1.ASSEMBLY_LOCAL_VALIDATION: result.result_hash},
        generator={"generator_id": result.checker.checker_id, "generator_version": result.checker.checker_version,
                   "build_id": result.checker.engine_build_id},
        files=files,
    )
    return ClientArtifactUpload(manifest=manifest, files=files)


__all__ = [
    "DRAWING_KIND_STATUS",
    "drawing_kind_for",
    "ClientArtifactKindV1",
    "ClientArtifactUpload",
    "ClientAssemblyArtifactManifestV1",
    "SourceBindingKindV1",
    "drawing_artifact_upload",
    "validation_artifact_upload",
]
