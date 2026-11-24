"""Interop capability catalog: which external scientific tools work with OpenSci data, for what, and how far it is checked.

Every entry names a scenario, candidate tools with their official links and licenses as reported upstream, what OpenSci
provides and what the tool computes, the OpenSci exports and exact entry points the workflow uses, the inputs only the
user can give, assumptions and limits (with OpenSci's refusal codes), where the conventions are stated, the runtime
(local, separate environment, WSL, commercial software) and its status:

* ``GUIDANCE_ONLY``: what the tool needs and where to get it; OpenSci has not executed the combination;
* ``DATA_CONTRACT_AVAILABLE``: a versioned OpenSci export with stated conventions carries the data the workflow needs;
* ``VERIFIED_EXAMPLE``: a named example with pinned versions was executed and checked against an independent reference
  (``verified``: tool, version, Python, date, the case's path in the public kit repository os-optics-kit and the commit it ran at);
* ``PLANNED``: the data the workflow needs is not published yet;
* ``DEPRECATED``: the tool is no longer maintained upstream (``replaced_by`` names the entry to use).

OpenSci ships, hosts and runs none of these tools: the user (or the user's own Agent) installs them where they choose.
External results stay in their own files next to the run they used and never change the run's hashes or its Local
Validation.

Use::

    from os_optics import interop

    interop.interop_catalog()                 # {"schema", "schema_version", "statuses", "directions", "entries": [...]}
    interop.interop_entry("quantum-photonics")
    interop.entry_ids()

The same from a shell: ``os-optics interop list [--status S] [--direction D] [--json]`` and ``os-optics
interop show ENTRY_ID [--json]``; from a JSON host: ``json_api.interop_catalog(entry_id=None)``.
"""

from __future__ import annotations

from typing import Any

from .entries import CHECKED, ENTRIES
from .model import (
    DIRECTION_TITLES,
    INTEROP_SCHEMA,
    INTEROP_SCHEMA_VERSION,
    STATUS_MEANINGS,
    EntryPointV1,
    InteropEntryV1,
    ToolV1,
    VerifiedRecordV1,
)

#: The entries by id, in catalog order.
INTEROP_ENTRIES: dict[str, InteropEntryV1] = {entry.entry_id: entry for entry in ENTRIES}
if len(INTEROP_ENTRIES) != len(ENTRIES):  # pragma: no cover - guarded by the tests
    raise RuntimeError("interop entry ids are not unique")


class InteropEntryUnknown(KeyError):
    """No entry with this id: ``entry_id`` and the known ids."""

    def __init__(self, entry_id: object) -> None:
        super().__init__(entry_id)
        self.entry_id = entry_id
        self.message = f"no interop entry {entry_id!r}; entries: {entry_ids()}"


def entry_ids() -> list[str]:
    """The entry ids, in catalog order."""

    return list(INTEROP_ENTRIES)


def interop_entry(entry_id: str) -> dict[str, Any]:
    """One entry as JSON (``InteropEntryUnknown`` when there is none)."""

    entry = INTEROP_ENTRIES.get(entry_id) if isinstance(entry_id, str) else None
    if entry is None:
        raise InteropEntryUnknown(entry_id)
    return entry.model_dump(mode="json")


def interop_catalog(*, status: str | None = None, direction: str | None = None) -> dict[str, Any]:
    """The whole catalog as JSON, optionally only the entries of one ``status`` / ``direction``."""

    entries = [e for e in ENTRIES if status in (None, e.status) and direction in (None, e.direction)]
    return {
        "schema": INTEROP_SCHEMA,
        "schema_version": INTEROP_SCHEMA_VERSION,
        "checked": CHECKED,
        "statuses": dict(STATUS_MEANINGS),
        "directions": dict(DIRECTION_TITLES),
        "entries": [e.model_dump(mode="json") for e in entries],
    }


def entry_text(entry: dict[str, Any]) -> str:
    """A readable rendering of one entry (``interop show`` without ``--json``)."""

    lines = [f"{entry['entry_id']}: {entry['title']}", f"status: {entry['status']} -- {STATUS_MEANINGS[entry['status']]}",
             f"direction: {DIRECTION_TITLES[entry['direction']]}", "", entry["scenario"], "", "Tools:"]
    for tool in entry["tools"]:
        facts = [tool["role"].lower().replace("_", " ")]
        if tool["maintainer"] != "UPSTREAM_PROJECT":
            facts.append(tool["maintainer"].lower().replace("_", " "))
        if tool["distribution"] != "OPEN_SOURCE":
            facts.append(tool["distribution"].lower().replace("_", " "))
        if tool["maintenance"] != "ACTIVE":
            facts.append(tool["maintenance"].lower().replace("_", " "))
        if tool["license"] is not None:
            facts.append(f"license {tool['license']} ({tool['license_source']}, {tool['license_checked']})")
        lines.append(f"  - {tool['name']} [{'; '.join(facts)}] {tool['url']}")
        lines.append(f"    computes: {tool['computes']}")
        lines.extend(f"    note: {note}" for note in tool["notes"])
    sections = (("OpenSci provides", entry["opensci_provides"]), ("The tool computes", entry["tool_computes"]),
                ("OpenSci data", entry["data"]), ("Inputs you provide", entry["user_inputs"]), ("Assumptions", entry["assumptions"]),
                ("Limits", entry["limits"]), ("Conventions", entry["conventions"]), ("Runtime", entry["runtime"]))
    for heading, items in sections:
        if items:
            lines.extend(["", f"{heading}:", *(f"  - {item}" for item in items)])
    if entry["entry_points"]:
        lines.extend(["", "Entry points:", *(f"  - {p['kind'].lower()} {p['name']}: {p['purpose']}" for p in entry["entry_points"])])
    lines.extend(["", "Verified:"])
    if entry["verified"]:
        lines.extend(f"  - {v['tool']} {v['version']} (Python {v['python']}, {v['date']}, engine {v['engine_version']}): {v['example']} "
                     f"@ {v['companion_commit']}; {v['outcome']}" for v in entry["verified"])
    else:
        lines.append("  - no executed check recorded")
    if entry["replaced_by"]:
        lines.extend(["", f"Use instead: {entry['replaced_by']}"])
    return "\n".join(lines)


__all__ = [
    "DIRECTION_TITLES",
    "INTEROP_ENTRIES",
    "INTEROP_SCHEMA",
    "INTEROP_SCHEMA_VERSION",
    "STATUS_MEANINGS",
    "EntryPointV1",
    "InteropEntryUnknown",
    "InteropEntryV1",
    "ToolV1",
    "VerifiedRecordV1",
    "entry_ids",
    "entry_text",
    "interop_catalog",
    "interop_entry",
]
