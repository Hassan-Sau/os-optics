"""`CatalogManifestV1`: which public optical catalog releases an OpenSci service distributes.

The public optical catalog (``PublicOpticalCatalogV2``) is versioned data, not part of any Engine release. A service
publishes the releases it serves in one small manifest (``GET /api/v1/catalog``):

* ``current``: the release new designs should use (status ``CURRENT``);
* ``retained``: every release the service still serves, newest first (by ``published_at``), the current one included,
  so a saved Project that pins an older ``catalog_version`` can still be reproduced;
* ``release_path_template``: where the exact bytes of one release are fetched, with ``{catalog_version}`` in place of
  the release label. It is resolved like a link against the manifest's own URL (an absolute path such as
  ``/api/v1/catalog/releases/{catalog_version}`` or a relative one such as ``releases/{catalog_version}.json``).

A release entry names the release (``catalog_version``), the public catalog ``schema_version`` its document is written
in (a client that cannot read that schema keeps its cached release and reports that it needs an upgrade), the
lowercase hexadecimal SHA-256 and the size of the exact document bytes (a client verifies both before using a
download), and when it was published. A ``catalog_version`` always denotes the same bytes: a client never accepts a
different document under a label it already holds.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StringConstraints, model_validator

from ..base import StrictContractModel
from ..common.hashes import Sha256Digest
from ..common.timestamps import UtcTimestamp
from .optical_catalog import CatalogVersion

CATALOG_MANIFEST_SCHEMA_ID = "opensci.design_catalog.catalog_manifest"
CATALOG_MANIFEST_VERSION = "1.0"
#: The placeholder ``release_path_template`` carries exactly once.
CATALOG_VERSION_PLACEHOLDER = "{catalog_version}"

CatalogSchemaVersion = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9]+\.[0-9]+$"),
    Field(description="The public optical catalog schema_version of a release document, '<major>.<minor>'."),
]

ReleasePathTemplate = Annotated[
    str,
    StringConstraints(strict=True, min_length=len(CATALOG_VERSION_PLACEHOLDER), max_length=512, pattern=r"^[^\s?#]+$"),
    Field(description="Path of one release, with '{catalog_version}' in place of the label; resolved against the manifest URL."),
]


def _instant(timestamp: str) -> tuple[str, str]:
    """A sort key of a ``UtcTimestamp`` (``...SS[.ffffff]Z``) in time order: fractional seconds padded to six digits."""

    seconds, _, fraction = timestamp[:-1].partition(".")
    return seconds, fraction.ljust(6, "0")


class CatalogReleaseStatus(StrEnum):
    """``CURRENT``: the release new designs use (exactly one); ``RETAINED``: an older release still served."""

    CURRENT = "CURRENT"
    RETAINED = "RETAINED"


class CatalogReleaseEntryV1(StrictContractModel):
    """One distributed public optical catalog release."""

    catalog_version: CatalogVersion
    schema_version: CatalogSchemaVersion
    sha256: Sha256Digest
    size_bytes: StrictInt = Field(gt=0)
    published_at: UtcTimestamp
    status: CatalogReleaseStatus


class CatalogManifestV1(StrictContractModel):
    """The public optical catalog releases available for download: the current release and the retained earlier
    releases, each with its schema version, SHA-256 and size."""

    schema_id: Literal["opensci.design_catalog.catalog_manifest"] = Field(alias="schema")
    manifest_version: Literal["1.0"]
    current: CatalogReleaseEntryV1
    retained: tuple[CatalogReleaseEntryV1, ...] = Field(min_length=1)
    release_path_template: ReleasePathTemplate

    @model_validator(mode="after")
    def _consistent(self) -> CatalogManifestV1:
        if self.current.status is not CatalogReleaseStatus.CURRENT:
            raise ValueError("current must have status CURRENT")
        versions = [entry.catalog_version for entry in self.retained]
        if len(set(versions)) != len(versions):
            raise ValueError("retained lists a catalog_version more than once")
        current = [entry for entry in self.retained if entry.status is CatalogReleaseStatus.CURRENT]
        if current != [self.current]:
            raise ValueError("retained must include the current release, identical to current, as its only CURRENT entry")
        published = [_instant(entry.published_at) for entry in self.retained]
        if published != sorted(published, reverse=True):
            raise ValueError("retained must be ordered newest first (published_at descending)")
        if self.release_path_template.count(CATALOG_VERSION_PLACEHOLDER) != 1:
            raise ValueError(f"release_path_template must contain {CATALOG_VERSION_PLACEHOLDER} exactly once")
        return self

    def release(self, catalog_version: str) -> CatalogReleaseEntryV1 | None:
        """The retained release labelled ``catalog_version``, if this manifest serves it."""

        return next((entry for entry in self.retained if entry.catalog_version == catalog_version), None)

    def release_path(self, catalog_version: str) -> str:
        """``release_path_template`` for ``catalog_version`` (still to be resolved against the manifest URL)."""

        return self.release_path_template.replace(CATALOG_VERSION_PLACEHOLDER, catalog_version)
