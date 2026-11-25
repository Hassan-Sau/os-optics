"""Errors of the local public optical catalog store and its synchronization (stable codes)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class CatalogErrorCode(StrEnum):
    """Why a catalog operation could not complete.

    * ``CATALOG_NOT_AVAILABLE``: no usable catalog release is on this device (or not the one a Project pins) and none
      could be obtained; the message says how to get one (``os-optics catalog sync`` / ``catalog import``).
    * ``CATALOG_ACTIVE_UNKNOWN``: catalog releases are stored on this device, none is selected as the active one
      (e.g. the store index was lost and rebuilt: which release was active cannot be recovered from the files), the
      catalog service could not settle it and this Engine reads none of them (with a readable one, an unpinned use takes
      the newest); the message lists the stored releases and the commands that choose one
      (``os-optics catalog sync``, ``catalog use <catalog_version>``, ``catalog import <file>``).
    * ``ENGINE_UPGRADE_REQUIRED``: the release is written in a catalog schema this Engine version cannot read.
    * ``CATALOG_INVALID``: a document that is not a valid public optical catalog release.
    * ``CATALOG_INTEGRITY_FAILED``: bytes differ from their published SHA-256 / size (a download, or a stored file
      that changed on disk); they are never used.
    * ``CATALOG_RELEASE_CONFLICT``: a different document under a ``catalog_version`` this device already holds; the
      stored release is kept (a release label always denotes the same bytes).
    * ``CATALOG_RELEASE_NOT_FOUND``: the service does not distribute that ``catalog_version``.
    * ``CATALOG_SERVICE_UNREACHABLE`` / ``CATALOG_SERVICE_ERROR``: the catalog service could not be reached, or
      answered with something that is not a valid catalog manifest / release.
    * ``CATALOG_ORIGIN_INVALID``: ``OPENSCI_CATALOG_URL`` / ``OPENSCI_API_URL`` is not a usable service URL.
    * ``CATALOG_STORE_ERROR``: the local catalog store cannot be read or written.
    * ``CATALOG_STORE_BUSY``: another os-optics operation kept the local catalog store locked for longer than the
      bounded wait; nothing was changed, try again.
    * ``CATALOG_SYNC_UNAVAILABLE``: this runtime has no network access for catalog synchronization (the browser
      runtime: the host installs the catalog instead).
    """

    CATALOG_NOT_AVAILABLE = "CATALOG_NOT_AVAILABLE"
    CATALOG_ACTIVE_UNKNOWN = "CATALOG_ACTIVE_UNKNOWN"
    ENGINE_UPGRADE_REQUIRED = "ENGINE_UPGRADE_REQUIRED"
    CATALOG_INVALID = "CATALOG_INVALID"
    CATALOG_INTEGRITY_FAILED = "CATALOG_INTEGRITY_FAILED"
    CATALOG_RELEASE_CONFLICT = "CATALOG_RELEASE_CONFLICT"
    CATALOG_RELEASE_NOT_FOUND = "CATALOG_RELEASE_NOT_FOUND"
    CATALOG_SERVICE_UNREACHABLE = "CATALOG_SERVICE_UNREACHABLE"
    CATALOG_SERVICE_ERROR = "CATALOG_SERVICE_ERROR"
    CATALOG_ORIGIN_INVALID = "CATALOG_ORIGIN_INVALID"
    CATALOG_STORE_ERROR = "CATALOG_STORE_ERROR"
    CATALOG_STORE_BUSY = "CATALOG_STORE_BUSY"
    CATALOG_SYNC_UNAVAILABLE = "CATALOG_SYNC_UNAVAILABLE"


class CatalogError(Exception):
    """A catalog operation that could not complete: ``code`` (``CatalogErrorCode``), ``message``, ``details``."""

    def __init__(self, code: CatalogErrorCode, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = CatalogErrorCode(code)
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"{self.code.value}: {message}")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": self.details}
