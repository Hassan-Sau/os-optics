"""Move a Project to another public catalog release (``update_project_catalog``).

A Project's catalog components pin the ``catalog_version`` they were designed against, so a saved Project always runs
against the same product data. When OpenSci publishes a new catalog release, ``update_project_catalog`` re-pins every
component whose product is published unchanged in the target release (the active one by default): the product's public
document -- specifications, simulation models, geometry -- must be identical, so the design's simulation cannot change
(its supply facts -- whether OpenSci supplies and mounts it, its delivered form -- do not count). It reports the components whose product changed or was withdrawn, and those it could not compare because the
release they were designed with is not on this device and cannot be obtained (``unverified``: e.g. a release the
catalog service no longer distributes). A component whose old release can never be compared -- the service no longer
distributes it (``NOT_DISTRIBUTED``) or it is written in a catalog format this Engine no longer reads
(``NOT_READABLE``) -- is re-pinned and stays listed in ``unverified`` with that reason: waiting or asking again cannot
change it. A changed product, and a component whose old release may still be obtained (``NOT_OBTAINED``), keep their
pin unless ``include_changed=True`` re-pins them too.

Statuses (``ProjectCatalogUpdate.status``) and what to do next (``next_steps`` of the report says the same):

* ``CURRENT``: every catalog component already pins the target release; nothing to do.
* ``UPDATED``: every catalog component now pins the target release and every product is unchanged: run the Project
  again (the updated Project needs its own Local Validation) and submit that run.
* ``UPDATED_UNVERIFIED``: every catalog component now pins the target release, but for some of them the product
  changed (``changed``) or could not be compared (``unverified``), so the simulation may differ: run the Project again
  and review its Local Validation before submitting.
* ``NEEDS_ATTENTION``: some components keep their old pin: ``changed`` ones and ``unverified`` ones whose old release
  may still be obtained (re-run with ``include_changed`` / ``--include-changed`` to move them) or ``withdrawn`` ones
  (choose another product).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2, PublicOpticalProductV2
from .._vendor.shared_contracts.optical_design import ProjectDocumentV2
from .catalog import (
    RELEASE_NOT_DISTRIBUTED,
    RELEASE_UNREADABLE,
    automatic_catalog_access_off,
    exact_catalog_lookup,
    load_project,
    load_public_catalog,
)

UPDATE_SCHEMA = "opensci.engine.project_catalog_update"
UPDATE_SCHEMA_VERSION = "1"
CURRENT = "CURRENT"
UPDATED = "UPDATED"
UPDATED_UNVERIFIED = "UPDATED_UNVERIFIED"
NEEDS_ATTENTION = "NEEDS_ATTENTION"
#: Why an old release could not be compared (``old_releases`` of the report).
NOT_DISTRIBUTED = "NOT_DISTRIBUTED"
NOT_OBTAINED = "NOT_OBTAINED"
NOT_READABLE = "NOT_READABLE"
#: Reasons a later attempt cannot change (no point retrying once the service is reachable).
FINAL_REASONS = frozenset({NOT_DISTRIBUTED, NOT_READABLE})
UPDATE_COMMAND = "os-optics project update-catalog PROJECT --out PROJECT"
RUN_COMMAND = "os-optics project run PROJECT --out-dir RUN_DIR"
SUBMIT_COMMAND = "os-optics resolve RUN_DIR"


def _content(product: PublicOpticalProductV2) -> str:
    # its engineering content: whether OpenSci supplies or mounts a product does not change its simulation
    return json.dumps(product.engineering_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _skus(rows: tuple[dict[str, Any], ...]) -> str:
    return ", ".join(f"{r['instance_id']} ({r['optical_sku']})" for r in rows)


@dataclass(frozen=True)
class ProjectCatalogUpdate:
    """The outcome of ``update_project_catalog``.

    ``status``: ``CURRENT``, ``UPDATED``, ``UPDATED_UNVERIFIED`` or ``NEEDS_ATTENTION`` (module docstring).
    ``repinned`` / ``changed`` / ``withdrawn`` / ``unverified`` list ``{"instance_id", "optical_sku",
    "from_catalog_version"}``; ``unverified``: the release the component was designed with is not on this device and
    could not be obtained, so whether its product changed could not be checked. ``old_releases`` says why per release:
    ``NOT_DISTRIBUTED`` (the catalog service does not distribute it any more), ``NOT_READABLE`` (it is written in an
    earlier catalog format this Engine no longer reads) or ``NOT_OBTAINED`` (e.g. no access to the service right
    now). ``include_changed``: whether changed / unverified components were re-pinned too.
    """

    project: ProjectDocumentV2
    catalog_version: str
    status: str
    repinned: tuple[dict[str, Any], ...]
    changed: tuple[dict[str, Any], ...]
    withdrawn: tuple[dict[str, Any], ...]
    unverified: tuple[dict[str, Any], ...]
    old_releases: tuple[tuple[str, str], ...] = ()
    include_changed: bool = False
    #: automatic catalog access was off (``OPENSCI_CATALOG_CHECK_INTERVAL=0``): a missing old release was not downloaded
    access_off: bool = False

    def _obtainable(self) -> list[str]:
        """The old releases of unverified components that a download could still bring (not retired, still distributed)."""

        reasons = dict(self.old_releases)
        return sorted(v for v in {r["from_catalog_version"] for r in self.unverified} if reasons.get(v) not in FINAL_REASONS)

    def _final(self, row: dict[str, Any]) -> bool:
        """The component's old release can never be compared (``FINAL_REASONS``): it was moved without being asked."""

        return dict(self.old_releases).get(row["from_catalog_version"]) in FINAL_REASONS

    @property
    def kept_unverified(self) -> tuple[dict[str, Any], ...]:
        """The unverified components that keep their old pin: none with ``include_changed``, else those whose old release
        may still be obtained."""

        return () if self.include_changed else tuple(row for row in self.unverified if not self._final(row))

    def _unverified_meaning(self, rows: tuple[dict[str, Any], ...] | None = None) -> str:
        rows = self.unverified if rows is None else rows
        reasons = dict(self.old_releases)
        parts = []
        for version in sorted({r["from_catalog_version"] for r in rows}):
            reason = reasons.get(version)
            if reason == NOT_READABLE:
                parts.append(f"release {version} is written in an earlier catalog format this os-optics does not read")
            elif reason == NOT_DISTRIBUTED:
                parts.append(f"release {version} is not on this device and the catalog service no longer distributes it")
            elif self.access_off:
                parts.append(f"release {version} is not on this device and automatic catalog access is off")
            else:
                parts.append(f"release {version} is not on this device and could not be downloaded")
        whose = "its" if len(rows) == 1 else "their"
        return f"{whose} product data cannot be compared with {self.catalog_version} because {' and '.join(parts)}"

    @property
    def summary(self) -> str:
        """One sentence: what happened and whether anything is left to do."""

        target = self.catalog_version
        if self.status == CURRENT:
            return f"the Project already uses public catalog {target}; nothing to do"
        if self.status == UPDATED:
            return f"every catalog component now uses public catalog {target} and its product data is unchanged"
        if self.status == UPDATED_UNVERIFIED:
            moved = len(self.changed) + len(self.unverified)
            verb = "was moved although its" if moved == 1 else "were moved although their"
            return (f"every catalog component now uses public catalog {target}, but {moved} of them {verb} product data "
                    "changed or could not be compared: the simulation may differ; see next_steps")
        kept = len(self.withdrawn) + (0 if self.include_changed else len(self.changed)) + len(self.kept_unverified)
        if kept == 1:
            return "1 catalog component keeps its old catalog release; see next_steps"
        return f"{kept} catalog components keep their old catalog release; see next_steps"

    @property
    def next_steps(self) -> list[dict[str, str]]:
        """What to do next, in order (``action``, ``message``, ``command``)."""

        target = self.catalog_version
        steps: list[dict[str, str]] = []
        if self.status == NEEDS_ATTENTION:
            kept = self.kept_unverified
            if kept:
                obtainable = self._obtainable()
                if not obtainable:
                    retry = ""
                elif self.access_off:
                    fetch = " and ".join(f"'os-optics catalog sync --release {v}'" for v in obtainable)
                    retry = (f" (or, to compare first, download the old release with {fetch}, or add it with 'os-optics "
                             "catalog import <catalog file>', and run this command again)")
                else:
                    retry = " (or run this command again once the catalog service is reachable)"
                them = "it" if len(kept) == 1 else "them"
                steps.append({"action": "INCLUDE_UNVERIFIED",
                              "message": f"{_skus(kept)}: {self._unverified_meaning(kept)}. Run the command again "
                                         f"with --include-changed to move {them} to {target}, then run the Project again and "
                                         f"review its Local Validation{retry}",
                              "command": f"{UPDATE_COMMAND} --include-changed"})
            if self.changed and not self.include_changed:
                steps.append({"action": "INCLUDE_CHANGED",
                              "message": f"{_skus(self.changed)}: the product data changed in {target}. To use the new data, "
                                         "run the command again with --include-changed, then run the Project again and "
                                         "compare its Local Validation with the previous one",
                              "command": f"{UPDATE_COMMAND} --include-changed"})
            if self.withdrawn:
                steps.append({"action": "CHOOSE_ANOTHER_PRODUCT",
                              "message": f"{_skus(self.withdrawn)}: not published in {target}. Choose another product for "
                                         f"{'this component' if len(self.withdrawn) == 1 else 'these components'} (browse "
                                         "with 'os-optics catalog list'), then run this "
                                         "command again",
                              "command": "os-optics catalog list --category <category>"})
            return steps
        if self.status == UPDATED_UNVERIFIED:
            moved = []
            if self.changed:
                moved.append(f"{_skus(self.changed)} (product data changed)")
            if self.unverified:
                moved.append(f"{_skus(self.unverified)} ({self._unverified_meaning()})")
            steps.append({"action": "RERUN_AND_REVIEW",
                          "message": f"moved to {target}: {'; '.join(moved)}. Run the Project again and review the new "
                                     "Local Validation (status and target values) before submitting",
                          "command": RUN_COMMAND})
        elif self.status == UPDATED:
            steps.append({"action": "RERUN",
                          "message": "run the updated Project again: it needs its own Local Validation before it is "
                                     "submitted (the result is expected to be unchanged)",
                          "command": RUN_COMMAND})
        if steps:
            steps.append({"action": "SUBMIT", "message": "submit the new run when you want prices and an assembly",
                          "command": SUBMIT_COMMAND})
        return steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": UPDATE_SCHEMA,
            "schema_version": UPDATE_SCHEMA_VERSION,
            "status": self.status,
            "summary": self.summary,
            "catalog_version": self.catalog_version,
            "project_hash": self.project.content_hash(),
            "repinned": list(self.repinned),
            "changed": list(self.changed),
            "withdrawn": list(self.withdrawn),
            "unverified": list(self.unverified),
            "old_releases": [{"catalog_version": version, "reason": reason} for version, reason in self.old_releases],
            "next_steps": self.next_steps,
        }


def update_project_catalog(project: Any, catalog: Any = None, *, include_changed: bool = False) -> ProjectCatalogUpdate:
    """Re-pin ``project``'s catalog components to ``catalog`` (default: the active catalog) where their product is
    published unchanged there; report changed / withdrawn / unverified products (module docstring). With
    ``include_changed=True`` components whose product changed (or whose old release could not be obtained) are re-pinned
    too and still reported (status ``UPDATED_UNVERIFIED``), so the design can be re-run against the new data; a
    component whose old release can never be compared is re-pinned either way (module docstring). Never
    edits anything but ``product.catalog_version``."""

    document = load_project(project)
    target: PublicOpticalCatalogV2 = load_public_catalog(catalog)
    payload = document.payload()
    old_catalogs: dict[str, PublicOpticalCatalogV2 | None] = {}
    old_reasons: dict[str, str] = {}
    repinned: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    attention = False
    for component, entry in zip(document.components, payload["components"]):
        ref = component.product
        if ref is None or ref.catalog_version == target.catalog_version:
            continue
        row = {"instance_id": component.instance_id, "optical_sku": ref.optical_sku, "from_catalog_version": ref.catalog_version}
        new = target.get(ref.optical_sku)
        if new is None:
            withdrawn.append(row)
            attention = True
            continue
        old_product = None
        if ref.catalog_version is not None:
            if ref.catalog_version not in old_catalogs:
                old_catalogs[ref.catalog_version], reason = exact_catalog_lookup(ref.catalog_version)
                if old_catalogs[ref.catalog_version] is None:
                    old_reasons[ref.catalog_version] = {RELEASE_NOT_DISTRIBUTED: NOT_DISTRIBUTED,
                                                        RELEASE_UNREADABLE: NOT_READABLE}.get(reason, NOT_OBTAINED)
            old_catalog = old_catalogs[ref.catalog_version]
            old_product = old_catalog.get(ref.optical_sku) if old_catalog is not None else None
            if old_catalog is None:
                unverified.append(row)
            elif old_product is None or _content(old_product) != _content(new):
                changed.append(row)
            # (an old release that can never be compared: keeping the pin would keep a design that can never run again)
            never = old_catalog is None and old_reasons.get(ref.catalog_version) in FINAL_REASONS
            if (old_catalog is None or old_product is None or _content(old_product) != _content(new)) and not never:
                if not include_changed:
                    attention = True
                    continue
        entry["product"]["catalog_version"] = target.catalog_version
        repinned.append(row)
    updated = ProjectDocumentV2.model_validate(payload)
    if attention:
        status = NEEDS_ATTENTION
    elif not repinned:
        status = CURRENT
    else:
        status = UPDATED_UNVERIFIED if changed or unverified else UPDATED
    reasons = tuple(sorted(old_reasons.items()))
    access_off = NOT_OBTAINED in old_reasons.values() and automatic_catalog_access_off()
    return ProjectCatalogUpdate(updated, target.catalog_version, status, tuple(repinned), tuple(changed), tuple(withdrawn),
                                tuple(unverified), reasons, bool(include_changed), access_off)


__all__ = ["CURRENT", "NEEDS_ATTENTION", "UPDATED", "UPDATED_UNVERIFIED", "ProjectCatalogUpdate", "update_project_catalog"]
