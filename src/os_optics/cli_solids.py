"""``os-optics project solids PROJECT``: are the Project's optical bodies physically possible solids?

Prints the optical-solid report (``os_optics.optical_solids``) of the Project's compiled elements at their nominal
poses: per element the profile domain over the outline, the thickness between consecutive surfaces over the whole
outline and the medium chain; per pair of elements whether their optical bodies overlap or touch. No ray is traced and
no mount, post or board is needed. Exit 0 = VALID, 1 = INVALID (the findings say what and where), 65 = the Project
cannot be lowered (the refusal is printed), 64 = unreadable file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

USAGE_ERROR = 64
LOCAL_INPUT_ERROR = 65
INVALID = 1
SOLIDS_DESCRIPTION = (
    "Check the Project's optical bodies before any simulation: every surface profile defined over its element's outline, "
    "a positive thickness between consecutive surfaces everywhere over the outline (the edge thickness included), media "
    "that chain from surface to surface, and no two placed optical bodies sharing volume (touching is allowed). Prints the "
    "report as JSON: status VALID / INVALID, findings with stable codes, component, surface and position, and the centre, "
    "minimum and edge thickness of every gap."
)


def register_project(project_commands: argparse._SubParsersAction) -> None:
    """Add ``os-optics project solids``."""

    solids = project_commands.add_parser("solids", help="check that the Project's optical bodies are possible solids (no simulation)",
                                         description=SOLIDS_DESCRIPTION,
                                         epilog="exit codes: 0 = VALID, 1 = INVALID, 64 = unreadable file, 65 = the Project cannot be lowered")
    solids.add_argument("project", help="path to a Project JSON file")
    solids.add_argument("--catalog", help="use this public optical catalog JSON instead of the release the Project pins / the active one")
    solids.add_argument("--edge-review-ratio", type=float, metavar="RATIO",
                        help="list EDGE_THICKNESS_IMPLAUSIBLE (INFO) when the glass of a Project's own lens model is thicker at its edge "
                             "than RATIO times its clear diameter (default 0.35); above 1.5 it is a warning for any model")
    solids.set_defaults(handler=_solids)


def _solids(args: argparse.Namespace) -> int:
    from .local import LocalRunError, load_project, lower_project
    from .optical_solids import EdgeThicknessBand, inspect_lowered

    ratio = getattr(args, "edge_review_ratio", None)
    if ratio is not None and not (math.isfinite(ratio) and ratio > 0.0):
        print("error: --edge-review-ratio must be a positive number", file=sys.stderr)
        return USAGE_ERROR

    try:
        with open(args.project, encoding="utf-8") as handle:
            document = json.load(handle)
        catalog = None
        if args.catalog:
            with open(args.catalog, encoding="utf-8") as handle:
                catalog = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {args.project if not args.catalog else 'the input'}: {exc}", file=sys.stderr)
        return USAGE_ERROR
    try:
        project = load_project(document)
        lowered = lower_project(project, catalog, check_solids=False)
    except LocalRunError as exc:
        print(json.dumps(exc.as_dict(), indent=2, ensure_ascii=True))
        return LOCAL_INPUT_ERROR
    report = inspect_lowered(lowered, project, edge_band=None if ratio is None else EdgeThicknessBand(review_ratio=ratio))
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=True))
    return 0 if report.valid else INVALID
