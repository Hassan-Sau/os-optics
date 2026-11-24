"""``os_optics.client_artifacts``: the upload of what this device computed, from real Engine outputs.

Over the committed chain-L reference pair (``tests/fixtures/assembly_drawing``): ``generate_assembly_drawing`` ->
``drawing_artifact_upload`` and the local assembly check's result -> ``validation_artifact_upload``. The manifest must re-validate with the
vendored canonical ``ClientAssemblyArtifactManifestV1`` from its JSON bytes (so ``artifact_set_digest`` re-derives), list
exactly the uploaded bytes, and take every binding from the Engine outputs themselves.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from delivery_helpers import DETAIL_JSON, PREVIEW_ZIP
from os_optics._vendor.shared_contracts.client_assembly_artifact import (
    KIND_RULES,
    ROLE_RULES,
    ClientArtifactKindV1,
    ClientAssemblyArtifactManifestV1,
    SourceBindingKindV1,
    artifact_set_digest_of,
    build_client_artifact_manifest,
)
from os_optics.assembly import load_assembly_preview, result_file_bytes, validate_assembly_layout
from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.client_artifacts import ClientArtifactUpload, drawing_artifact_upload, validation_artifact_upload
from os_optics.manifest import engine_build_id

DRAWING_FILES = {"assembly_drawing.pdf", "assembly_drawing.svg", "drawing_manifest.json", "drawing_result.json"}


@pytest.fixture(scope="module")
def preview():
    return load_assembly_preview(PREVIEW_ZIP)


@pytest.fixture(scope="module")
def drawing(preview):
    return generate_assembly_drawing(preview, DETAIL_JSON)


def reparsed(upload: ClientArtifactUpload) -> ClientAssemblyArtifactManifestV1:
    document = json.loads(upload.manifest_json_bytes().decode("utf-8"))
    manifest = ClientAssemblyArtifactManifestV1.model_validate(document)
    assert manifest == upload.manifest
    assert manifest.artifact_set_digest == artifact_set_digest_of(manifest.content_payload()) == upload.artifact_set_digest
    return manifest


def assert_files_match(manifest: ClientAssemblyArtifactManifestV1, files: dict[str, bytes]) -> None:
    assert [f.filename for f in manifest.files] == sorted(files)
    for entry in manifest.files:
        data = files[entry.filename]
        assert (entry.size_bytes, entry.sha256) == (len(data), hashlib.sha256(data).hexdigest())
        assert (entry.role, entry.media_type) == next((role, rule.media_type) for role, rule in ROLE_RULES.items()
                                                       if rule.filename == entry.filename)


# ---- drawing ----------------------------------------------------------------------------------------------------------------------
def test_the_drawing_upload_is_the_four_real_files_bound_from_the_drawing(drawing, preview):
    upload = drawing_artifact_upload(drawing)
    manifest = reparsed(upload)
    assert manifest.artifact_kind is ClientArtifactKindV1.ASSEMBLY_DRAWING
    assert upload.files == dict(drawing.files) and set(upload.files) == DRAWING_FILES
    assert_files_match(manifest, upload.files)
    result = drawing.result
    assert (manifest.binding.project_hash, manifest.binding.commercial_resolution_id, manifest.binding.customer_bom_digest) == (
        preview.manifest.binding.project.project_hash.digest, preview.manifest.binding.commercial_resolution_id,
        preview.manifest.binding.commercial_bom.bom_hash.digest)
    assert {s.kind: s.digest for s in manifest.binding.sources} == {
        SourceBindingKindV1.ASSEMBLY_PREVIEW_PACKAGE: preview.content_hash,
        SourceBindingKindV1.ASSEMBLY_DETAIL_PACKAGE: result.binding.detail_content_hash,
        SourceBindingKindV1.ASSEMBLY_LOCAL_VALIDATION: drawing.validation.result_hash,
    }
    assert set(s.kind for s in manifest.binding.sources) == KIND_RULES[ClientArtifactKindV1.ASSEMBLY_DRAWING].required_sources
    assert manifest.generator.model_dump() == {"generator_id": "opensci.engine.assembly_drawing",
                                               "generator_version": result.generator.generator_version,
                                               "build_id": engine_build_id()}


def test_the_drawing_upload_is_deterministic_and_equals_the_canonical_builder(drawing):
    upload = drawing_artifact_upload(drawing, kind="ASSEMBLY_DRAWING")
    assert upload.manifest_json_bytes() == drawing_artifact_upload(drawing).manifest_json_bytes()
    result = drawing.result
    canonical = build_client_artifact_manifest(
        "ASSEMBLY_DRAWING", project_hash=result.binding.project.project_hash.digest,
        commercial_resolution_id=result.binding.commercial_resolution_id,
        customer_bom_digest=result.binding.commercial_bom.bom_hash.digest,
        sources={"ASSEMBLY_PREVIEW_PACKAGE": result.binding.preview_package_content_hash,
                 "ASSEMBLY_DETAIL_PACKAGE": result.binding.detail_content_hash,
                 "ASSEMBLY_LOCAL_VALIDATION": result.validation.result_hash},
        generator={"generator_id": result.generator.generator_id, "generator_version": result.generator.generator_version,
                   "build_id": result.generator.engine_build_id},
        files=dict(drawing.files), locale=result.locale)
    assert upload.manifest == canonical and upload.manifest.locale == "en"


def test_the_multipart_fields_are_one_manifest_part_and_one_part_per_file(drawing):
    upload = drawing_artifact_upload(drawing)
    fields = upload.multipart_fields()
    assert fields[0] == ("manifest", ("manifest.json", upload.manifest_json_bytes(), "application/json"))
    assert [(name, filename, media) for name, (filename, _, media) in fields[1:]] == [
        ("file", entry.filename, entry.media_type) for entry in upload.manifest.files]
    assert {filename: data for _, (filename, data, _) in fields[1:]} == dict(drawing.files)
    assert upload.manifest_json_bytes().endswith(b"\n") and b"\r" not in upload.manifest_json_bytes()
    assert "build_id" in json.loads(upload.manifest_json_bytes())["generator"]


def test_a_drawing_kind_must_match_the_drawing_status(drawing, preview):
    assert drawing.result.drawing_status.value == "DETAILED_ASSEMBLY_DRAWING"
    with pytest.raises(ValueError, match="upload it as ASSEMBLY_DRAWING"):
        drawing_artifact_upload(drawing, kind=ClientArtifactKindV1.ASSEMBLY_DRAWING_DIAGNOSTIC)
    with pytest.raises(ValueError, match="not a drawing artifact kind"):
        drawing_artifact_upload(drawing, kind="ASSEMBLY_LOCAL_VALIDATION")
    with pytest.raises(ValueError):
        drawing_artifact_upload(drawing, kind="DRAWING")
    with pytest.raises(TypeError):
        drawing_artifact_upload(dict(drawing.files))


# ---- local validation ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["package", "zip-path", "zip-bytes"])
def test_the_validation_upload_is_the_r4c_file_bound_from_the_package(preview, drawing, source):
    result = validate_assembly_layout(preview)
    assert result.result_hash == drawing.validation.result_hash  # the generator ran the same local assembly check
    given = {"package": preview, "zip-path": PREVIEW_ZIP, "zip-bytes": PREVIEW_ZIP.read_bytes()}[source]
    upload = validation_artifact_upload(given, result)
    manifest = reparsed(upload)
    assert manifest.artifact_kind is ClientArtifactKindV1.ASSEMBLY_LOCAL_VALIDATION
    assert upload.files == {"assembly_local_validation.json": result_file_bytes(result)}
    assert_files_match(manifest, upload.files)
    binding = preview.manifest.binding
    assert (manifest.binding.project_hash, manifest.binding.commercial_resolution_id, manifest.binding.customer_bom_digest) == (
        binding.project.project_hash.digest, binding.commercial_resolution_id, binding.commercial_bom.bom_hash.digest)
    assert {s.kind: s.digest for s in manifest.binding.sources} == {
        SourceBindingKindV1.ASSEMBLY_PREVIEW_PACKAGE: preview.content_hash,
        SourceBindingKindV1.ASSEMBLY_LOCAL_VALIDATION: result.result_hash}
    assert manifest.generator.model_dump() == {"generator_id": result.checker.checker_id,
                                               "generator_version": result.checker.checker_version,
                                               "build_id": engine_build_id()}
    assert upload.manifest_json_bytes() == validation_artifact_upload(preview, drawing.validation).manifest_json_bytes()


def test_a_validation_of_another_package_is_refused(preview, drawing):
    other = drawing.validation.model_copy(update={"package": drawing.validation.package.model_copy(
        update={"package_content_hash": "0" * 64})})
    with pytest.raises(ValueError, match="another assembly preview package"):
        validation_artifact_upload(preview, other)
    with pytest.raises(TypeError):
        validation_artifact_upload(preview, drawing.result)


def test_an_upload_must_carry_exactly_the_manifest_bytes(drawing):
    upload = drawing_artifact_upload(drawing)
    files = dict(upload.files)
    with pytest.raises(ValueError, match="exactly the manifest's files"):
        ClientArtifactUpload(manifest=upload.manifest, files={k: v for k, v in files.items() if k != "assembly_drawing.pdf"})
    with pytest.raises(ValueError, match="do not match the manifest"):
        ClientArtifactUpload(manifest=upload.manifest, files=files | {"assembly_drawing.svg": files["assembly_drawing.svg"] + b" "})
    assert "%PDF" not in repr(upload)  # the file bytes stay out of the repr
