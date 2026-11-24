"""``os-optics project network PROJECT``: export the Project's optics as a linear optical network (JSON)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

USAGE_ERROR = 64
NOT_EXPORTABLE = 65
NETWORK_DESCRIPTION = (
    "Export the Project's optics as a linear optical network (opensci-linear-optical-network/v1) for a quantum-optics or "
    "classical simulator of your choice: input modes (each source, lab H / V), output modes (each beam on an output "
    "observation plane, lab H / V), the complex transfer matrix T, every beam path with its optical path length and Jones "
    "amplitudes, per-element Jones matrices, delay / coherence hints, the stated conventions and a contract hash. One "
    "wavelength per export (--wavelength-nm when the sources emit several). PROJECT may also be an Engine request JSON. "
    "A layout that is not a linear optical network (a cavity, partially overlapping beams, an ideal thin lens, ...) is "
    "refused with structured problems; a partial network is never written."
)


def register_project(project_commands: argparse._SubParsersAction) -> None:
    """Add ``os-optics project network``."""

    network = project_commands.add_parser(
        "network", help="export the Project as a linear optical network (modes, transfer matrix, paths) for a quantum simulator",
        description=NETWORK_DESCRIPTION,
        epilog="exit codes: 0 = network written, 64 = unreadable or unwritable file, 65 = not exportable or a Project that "
               "cannot be simulated (JSON with status ERROR and structured problems on stdout)")
    network.add_argument("project", help="path to a Project JSON file (or an Engine request JSON)")
    network.add_argument("--wavelength-nm", type=float, help="the spectral line to export (required when the sources emit several)")
    network.add_argument("--observation", action="append", dest="observation_ids", metavar="OBSERVATION_ID",
                         help="an output plane (repeat; default: every observation plane)")
    network.add_argument("--source", action="append", dest="source_ids", metavar="SOURCE_ID",
                         help="a source with input modes (repeat; default: every source)")
    network.add_argument("--configuration", dest="configuration_id", metavar="ID",
                         help="the configuration to export, of a Project that declares configurations")
    network.add_argument("--catalog", help="use this public optical catalog JSON instead of the release the Project pins / the active one")
    network.add_argument("--out", help="write the network JSON here and print a short summary (default: print the network)")
    network.set_defaults(handler=_network)


def _network(args: argparse.Namespace) -> int:
    from .local.json_api import linear_optical_network

    try:
        document = json.loads(Path(args.project).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {args.project}: {exc}")
        return USAGE_ERROR
    options = {key: value for key, value in (("wavelength_nm", args.wavelength_nm), ("observation_ids", args.observation_ids),
                                              ("source_ids", args.source_ids)) if value is not None}
    reply = linear_optical_network(document, options or None, catalog=args.catalog, configuration_id=args.configuration_id)
    if not reply["ok"]:
        print(json.dumps({"status": "ERROR", "error": reply["error"]}, indent=2))
        return NOT_EXPORTABLE
    text = json.dumps(reply["network"], indent=1, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
        return 0
    try:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}")
        return USAGE_ERROR
    network = reply["network"]
    matrix = network["transfer_matrix"]
    print(json.dumps({"status": "EXPORTED", "out": args.out, "schema": network["schema"], "wavelength_nm": network["wavelength_nm"],
                      "input_modes": len(matrix["columns"]), "output_modes": len(matrix["rows"]), "paths": len(network["paths"]),
                      "lossless": matrix["lossless"], "contract_hash": network["contract_hash"]}, indent=2))
    return 0
