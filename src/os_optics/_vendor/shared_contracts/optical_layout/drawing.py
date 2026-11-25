"""Optical Layout Drawing outputs: the local output manifest of one rendered Optical Layout Package.

An OpenSci runtime (``opensci_engine.optical_layout``) renders an ``OpticalLayoutPackageV1`` on the customer's device
into three files::

    optical_layout.svg              one landscape sheet (A3, or the smallest larger A-series sheet the layout needs)
    optical_layout.pdf              the same sheet as a one-page PDF
    optical_layout_manifest.json    OpticalLayoutDrawingManifestV1: what was rendered (the package's
                                    ``layout_content_hash`` and binding), in which language, by which build, and every
                                    other file's size, media type and SHA-256

The sheet is the deliverable of an optics-only purchase: the relative optical layout -- plan and elevation views of
the elements and nominal beam paths, an element table (identity, relative position, axis direction, retained DOFs), a
spacing table along each beam path, notes and a title block (Order Number and customer once ordered).
``verify_layout_drawing_outputs`` checks a received file set; it cannot re-render the sheet (that needs the package).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from ..assembly_detail.drawing import DrawingSheetV1
from ..assembly_preview.package import json_file_bytes
from ..base import StrictContractModel
from ..common.hashes import Sha256Digest, osc1_digest
from ..common.identifiers import BuildVersion
from ..optical_design.local_validation import ENGINE_PACKAGE_NAME, EnginePackageName
from .package import OpticalLayoutBindingV1

OPTICAL_LAYOUT_DRAWING_MANIFEST_SCHEMA_ID: Literal["opensci.optical_layout.drawing_manifest"] = (
    "opensci.optical_layout.drawing_manifest")
OPTICAL_LAYOUT_DRAWING_SCHEMA_VERSION: Literal["1.0"] = "1.0"
OPTICAL_LAYOUT_GENERATOR_ID: Literal["opensci.engine.optical_layout"] = "opensci.engine.optical_layout"

LAYOUT_SVG_PATH = "optical_layout.svg"
LAYOUT_PDF_PATH = "optical_layout.pdf"
LAYOUT_MANIFEST_PATH = "optical_layout_manifest.json"
LAYOUT_DRAWING_FILES = (LAYOUT_SVG_PATH, LAYOUT_PDF_PATH, LAYOUT_MANIFEST_PATH)
MAX_LAYOUT_FILE_BYTES = 16 * 1024 * 1024

_FILE_SPEC: dict[str, tuple[str, str]] = {
    LAYOUT_PDF_PATH: ("LAYOUT_PDF", "application/pdf"),
    LAYOUT_SVG_PATH: ("LAYOUT_SVG", "image/svg+xml"),
}


class OpticalLayoutGeneratorRefV1(StrictContractModel):
    generator_id: Literal["opensci.engine.optical_layout"] = OPTICAL_LAYOUT_GENERATOR_ID
    generator_version: BuildVersion
    engine_package: EnginePackageName = ENGINE_PACKAGE_NAME
    engine_version: BuildVersion
    engine_build_id: BuildVersion


class OpticalLayoutDrawingFileV1(StrictContractModel):
    path: Literal["optical_layout.pdf", "optical_layout.svg"]
    role: Literal["LAYOUT_PDF", "LAYOUT_SVG"]
    media_type: Literal["application/pdf", "image/svg+xml"]
    size_bytes: int = Field(strict=True, ge=1, le=MAX_LAYOUT_FILE_BYTES)
    sha256: Sha256Digest

    @model_validator(mode="after")
    def _spec(self) -> OpticalLayoutDrawingFileV1:
        if (self.role, self.media_type) != _FILE_SPEC[self.path]:
            raise ValueError(f"{self.path}: role/media_type must be {_FILE_SPEC[self.path]}")
        return self


class OpticalLayoutDrawingManifestV1(StrictContractModel):
    """The local output record of one Optical Layout Drawing: the package it renders (``layout_content_hash`` and its
    ``binding``), the language, the sheet, the build and the two drawing files."""

    schema_id: Literal["opensci.optical_layout.drawing_manifest"] = Field(OPTICAL_LAYOUT_DRAWING_MANIFEST_SCHEMA_ID, alias="schema")
    schema_version: Literal["1.0"] = OPTICAL_LAYOUT_DRAWING_SCHEMA_VERSION
    generator: OpticalLayoutGeneratorRefV1
    locale: Literal["en", "zh-CN"]
    layout_content_hash: Sha256Digest
    binding: OpticalLayoutBindingV1
    sheet: DrawingSheetV1
    files: tuple[OpticalLayoutDrawingFileV1, ...] = Field(min_length=2, max_length=2)
    manifest_content_hash: Sha256Digest = Field(description="osc1 digest of this manifest without manifest_content_hash.")

    @model_validator(mode="after")
    def _check(self) -> OpticalLayoutDrawingManifestV1:
        if [f.path for f in self.files] != sorted(_FILE_SPEC):
            raise ValueError(f"files must be exactly {sorted(_FILE_SPEC)} in that order")
        if self.manifest_content_hash != osc1_digest(self.hashed_payload()):
            raise ValueError("manifest_content_hash does not match the manifest")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def hashed_payload(self) -> dict[str, Any]:
        payload = self.payload()
        del payload["manifest_content_hash"]
        return payload


def layout_drawing_manifest_for(*, generator: Mapping[str, str], locale: str, layout_content_hash: str,
                                binding: OpticalLayoutBindingV1, sheet: str,
                                files: Mapping[str, bytes]) -> OpticalLayoutDrawingManifestV1:
    """The manifest of the rendered ``files`` (the SVG and PDF bytes)."""

    entries = []
    for path in sorted(_FILE_SPEC):
        role, media = _FILE_SPEC[path]
        data = bytes(files[path])
        entries.append({"path": path, "role": role, "media_type": media, "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    content = {
        "schema": OPTICAL_LAYOUT_DRAWING_MANIFEST_SCHEMA_ID,
        "schema_version": OPTICAL_LAYOUT_DRAWING_SCHEMA_VERSION,
        "generator": dict(generator),
        "locale": locale,
        "layout_content_hash": layout_content_hash,
        "binding": binding.model_dump(mode="json", by_alias=True, exclude_none=True),
        "sheet": sheet,
        "files": entries,
    }
    return OpticalLayoutDrawingManifestV1.model_validate({**content, "manifest_content_hash": osc1_digest(content)})


def verify_layout_drawing_outputs(files: Mapping[str, bytes]) -> tuple[str, ...]:
    """Every problem of a received Optical Layout Drawing output set (exactly the three files); empty = valid. Never
    raises: the manifest (schema, canonical bytes, content hash) and each file's size and SHA-256."""

    if set(files) != set(LAYOUT_DRAWING_FILES):
        return (f"optical layout drawing outputs must be exactly {sorted(LAYOUT_DRAWING_FILES)}",)
    if any(not isinstance(v, (bytes, bytearray)) for v in files.values()):
        return ("optical layout drawing output content must be bytes",)
    try:
        manifest = OpticalLayoutDrawingManifestV1.model_validate_json(bytes(files[LAYOUT_MANIFEST_PATH]))
    except (ValidationError, ValueError, RecursionError) as exc:
        return (f"{LAYOUT_MANIFEST_PATH} is not a valid optical layout drawing manifest: {type(exc).__name__}",)
    problems: list[str] = []
    if bytes(files[LAYOUT_MANIFEST_PATH]) != json_file_bytes(manifest.payload()):
        problems.append(f"{LAYOUT_MANIFEST_PATH} is not in canonical file form")
    for entry in manifest.files:
        data = bytes(files[entry.path])
        if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
            problems.append(f"{entry.path}: size or SHA-256 does not match the manifest")
    if not bytes(files[LAYOUT_PDF_PATH]).startswith(b"%PDF-"):
        problems.append(f"{LAYOUT_PDF_PATH} is not a PDF")
    if b"<svg" not in bytes(files[LAYOUT_SVG_PATH])[:512]:
        problems.append(f"{LAYOUT_SVG_PATH} is not an SVG document")
    return tuple(problems)


__all__ = [
    "LAYOUT_DRAWING_FILES",
    "LAYOUT_MANIFEST_PATH",
    "LAYOUT_PDF_PATH",
    "LAYOUT_SVG_PATH",
    "OPTICAL_LAYOUT_DRAWING_MANIFEST_SCHEMA_ID",
    "OPTICAL_LAYOUT_DRAWING_SCHEMA_VERSION",
    "OPTICAL_LAYOUT_GENERATOR_ID",
    "OpticalLayoutDrawingFileV1",
    "OpticalLayoutDrawingManifestV1",
    "OpticalLayoutGeneratorRefV1",
    "layout_drawing_manifest_for",
    "verify_layout_drawing_outputs",
]
