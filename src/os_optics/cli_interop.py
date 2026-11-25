"""``os-optics interop list|show``: the interop capability catalog (``os_optics.interop``) from a shell."""

from __future__ import annotations

import argparse
import json

from .interop.model import DIRECTION_TITLES, STATUS_MEANINGS

USAGE_ERROR = 64
INTEROP_DESCRIPTION = (
    "External scientific tools that work with OpenSci data: optimization and tolerancing, scalar diffraction and imaging, "
    "DOE / vector / inverse design, cavities and interferometers, quantum photonics, micro-nano devices and PICs, metasurface "
    "responses as virtual elements, coatings, materials and polarization, exact CAD geometry and specialist topics. Each "
    "entry states the scenario, the candidate tools with their official links and licenses as reported upstream, what "
    "OpenSci provides and what the tool computes, the OpenSci exports and entry points to use, the inputs you provide, "
    "assumptions and limits, the runtime, and its status: "
    + "; ".join(f"{status} = {meaning}" for status, meaning in STATUS_MEANINGS.items())
    + " Python: os_optics.interop; JSON hosts: json_api.interop_catalog."
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add ``os-optics interop``."""

    interop = subparsers.add_parser("interop", help="external scientific tools that work with OpenSci data: what, how, status",
                                    description=INTEROP_DESCRIPTION)
    sub = interop.add_subparsers(dest="interop_command", required=True)
    listing = sub.add_parser("list", help="every entry: id, status, title, tools", description=INTEROP_DESCRIPTION)
    listing.add_argument("--status", choices=list(STATUS_MEANINGS), help="only entries with this status")
    listing.add_argument("--direction", choices=list(DIRECTION_TITLES), help="only entries of this direction")
    listing.add_argument("--json", action="store_true", help="print the catalog as JSON (every field of every entry)")
    listing.set_defaults(handler=_list)
    show = sub.add_parser("show", help="one entry in full", description=INTEROP_DESCRIPTION,
                          epilog="exit codes: 0 = shown, 64 = unknown entry id")
    show.add_argument("entry_id", help="an entry id from 'os-optics interop list'")
    show.add_argument("--json", action="store_true", help="print the entry as JSON")
    show.set_defaults(handler=_show)


def _list(args: argparse.Namespace) -> int:
    from .interop import interop_catalog

    catalog = interop_catalog(status=args.status, direction=args.direction)
    if args.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=True))
        return 0
    rows = [(e["entry_id"], e["status"], e["title"], ", ".join(t["name"] for t in e["tools"])) for e in catalog["entries"]]
    width_id = max([len("ENTRY"), *(len(r[0]) for r in rows)])
    width_status = max([len("STATUS"), *(len(r[1]) for r in rows)])
    print(f"{'ENTRY'.ljust(width_id)}  {'STATUS'.ljust(width_status)}  TITLE (TOOLS)")
    for entry_id, status, title, tools in rows:
        print(f"{entry_id.ljust(width_id)}  {status.ljust(width_status)}  {title} ({tools})")
    print(f"\n{len(rows)} entries (checked {catalog['checked']}). 'os-optics interop show ENTRY' gives one in full.")
    return 0


def _show(args: argparse.Namespace) -> int:
    from .interop import InteropEntryUnknown, entry_text, interop_entry

    try:
        entry = interop_entry(args.entry_id)
    except InteropEntryUnknown as exc:
        print(json.dumps({"status": "ERROR", "error": {"code": "INTEROP_ENTRY_UNKNOWN", "message": exc.message}}, indent=2))
        return USAGE_ERROR
    print(json.dumps(entry, indent=2, ensure_ascii=True) if args.json else entry_text(entry))
    return 0
