"""The one place the Engine's tests take producer identity out of committed test data.

Committed test data pins content. It does not pin which build recorded it, so a version bump, a schema bump or any Engine
source edit (the build id digests the sources) leaves every pin where it is, and a change of the numbers still fails.

Identity, in a ``ValidationResult`` manifest:

- ``engine_version``, ``engine_build_id``, ``schema_version``, ``project_schema_version`` name the producer. A snapshot does
  not store them.
- ``input_hash`` and ``project_hash`` are exact hashes of documents that contain ``project_schema_version``, so they move with
  the schema version alone. A snapshot stores them with that stamp masked, under ``*_identity_masked``: the digest still
  moves with every other field of the request and with the canonicalization of ``hash_exact``.
- ``runtime`` describes the host and is never stored.

Everything else in the manifest is content and is compared exactly: ``design_projection_hash`` covers only the baseplate,
components, observations and materials; ``result_hash`` and ``trace_hash`` cover the quantized result body and trace events,
which carry no identity (``test_fixtures_cli_freeze.py::test_manifest_hashes_bind_the_stamped_request`` moves the stamp and
watches each of them); ``numerical_config_version``, ``manifest_version`` and ``model_versions`` move only with a change of
numerical defaults or physics models.

How a binding is tested: the real hash is recomputed from its parts and is shown to move when a part moves; the pin is the
masked digest. Keep the two together so neither a drifting lowering nor a broken binding gets through.

A request file leaves ``project.project_schema_version`` out: ``ProjectSnapshot`` defaults it to the current
``SCHEMA_VERSION``, so every reader (pydantic, the ``validate`` command, ``validate_json``, the Pyodide harness) stamps the
schema of the Engine reading it. ``_schema_ok`` in ``validation/pipeline.py`` compares the major version only, so a stored stamp
would pass silently across a minor bump (with a stale hash) and be refused at the next major one.
"""

from __future__ import annotations

import copy
from typing import Any

from os_optics.manifest import hash_exact

#: What stands in for ``project_schema_version`` in a masked digest. Not a version on purpose: it cannot be mistaken for one
#: and can never coincide with the schema the Engine stamps.
IDENTITY_MASK = "<identity-masked>"
#: Manifest fields a snapshot never stores.
IDENTITY_MANIFEST_FIELDS = ("runtime", "engine_version", "engine_build_id", "schema_version", "project_schema_version")
#: Manifest hashes that bind the schema stamp: a snapshot stores each as ``<name>_identity_masked``.
IDENTITY_BOUND_MANIFEST_HASHES = ("input_hash", "project_hash")


def _request_document(request: Any) -> dict:
    return request.model_dump(mode="json") if hasattr(request, "model_dump") else copy.deepcopy(request)


def request_without_identity(request: Any) -> dict:
    """The request document a request file stores: no ``project_schema_version``, which a reader stamps as its own."""

    document = _request_document(request)
    document["project"].pop("project_schema_version", None)
    return document


def mask_request_identity(request: Any) -> dict:
    """The request document with its schema stamp masked and every other field exactly as it is."""

    document = _request_document(request)
    document["project"]["project_schema_version"] = IDENTITY_MASK
    return document


def masked_input_hash(request: Any) -> str:
    """``hash_exact`` of the request with its schema stamp masked: the pin of what a Project lowers to."""

    return hash_exact(mask_request_identity(request))


def masked_project_hash(request: Any) -> str:
    return hash_exact(mask_request_identity(request)["project"])


def mask_result_identity(result: dict, request: Any) -> dict:
    """The content of a ``ValidationResult`` document as a snapshot stores and compares it (see the module docstring)."""

    document = copy.deepcopy(result)
    manifest = document["manifest"]
    for field in IDENTITY_MANIFEST_FIELDS:
        manifest.pop(field, None)
    masked = {"input_hash": masked_input_hash(request), "project_hash": masked_project_hash(request)}
    for field in IDENTITY_BOUND_MANIFEST_HASHES:
        # a result refused before its Project was read has no project hash: it stays empty, which is content too
        manifest[f"{field}_identity_masked"] = masked[field] if manifest.pop(field) else ""
    return document
