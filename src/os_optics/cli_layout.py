"""``os-optics layout draw``: the Optical Layout Drawing of an optics-only purchase, rendered on this device.

``layout draw PACKAGE_JSON --out DIR [--locale zh-CN|en]`` renders an Optical Layout Package file
(``os_optics.optical_layout``) offline. ``order layout ORDER_ID`` / ``resolution layout RESOLUTION_ID``
(``cli_commerce``) fetch the package from the OpenSci service first, save it as ``optical-layout.json`` in --out (the
file ``layout draw`` takes) and render it the same way (``write_layout``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

USAGE_ERROR = 64
LOCAL_INPUT_ERROR = 65
LOCALE_CHOICES = ("en", "zh-CN")
#: The file a fetched Optical Layout Package is saved to in --out (what ``layout draw`` renders again offline).
PACKAGE_FILE = "optical-layout.json"

LAYOUT_DESCRIPTION = (
    "The Optical Layout Drawing of an optics-only purchase: the relative optical layout of a validated design -- each "
    "optical element's identity (OpenSci product and BOM line, or your own component), its position relative to the "
    "datum element, its orientation, the nominal beam paths and the spacings along them -- for building your own "
    "mounting around the optics. No height above a table or breadboard is implied. 'draw PACKAGE_JSON' renders an "
    "Optical Layout Package file on this device (no network) into optical_layout.svg, optical_layout.pdf and "
    "optical_layout_manifest.json in --out; for an Order use 'os-optics order layout ORDER_ID --out DIR' (after "
    "an optics-only resolution, 'os-optics resolution layout RESOLUTION_ID --out DIR'), which fetch the package "
    "with your Account token. An optics-only Order has no Detailed Assembly Drawing: this drawing is its layout "
    "deliverable. A run directory has no package: its 2D layout is 'os-optics view RUN_DIR --section ALL --snapshot DIR', a "
    "true-scale section with the stop, the image plane and the traced rays. An earlier layout's files in --out are replaced. "
    "Exit codes: 0 = written, 64 = usage error / unwritable --out, 65 = no package was given (OPTICAL_LAYOUT_PACKAGE_REQUIRED) or the package "
    "is not usable (JSON error on stdout)."
)
FETCH_DESCRIPTION = (
    "The Optical Layout Drawing of an optics-only purchase: the relative optical layout of the validated design -- each "
    "optical element's identity (OpenSci product and BOM line, or your own component), its position relative to the "
    "datum element, its orientation, the nominal beam paths and the spacings along them -- for building your own "
    "mounting around the optics. No height above a table or breadboard is implied. Fetches the Optical Layout Package "
    "with your Account token, saves it as optical-layout.json in --out (render it again offline with 'os-optics "
    "layout draw DIR/optical-layout.json --out DIR2 --locale en|zh-CN') and renders optical_layout.svg, optical_layout.pdf "
    "and optical_layout_manifest.json next to it. An optics-only Order has no Detailed Assembly Drawing: this drawing is "
    "its layout deliverable ('order layout' once ordered; 'resolution layout' for an optics-only resolution before "
    "ordering). An earlier layout's files in --out are replaced. Exit codes: 0 = written, 64 = usage error / unwritable "
    "--out, 65 = the package "
    "is not usable (JSON error on stdout)."
)


def summary(drawing: Any, out: str) -> dict[str, Any]:
    manifest = drawing.manifest
    binding = manifest.binding
    return {"status": "WRITTEN", "locale": manifest.locale, "sheet": manifest.sheet.value,
            "layout_content_hash": manifest.layout_content_hash, "commercial_resolution_id": binding.commercial_resolution_id,
            "order_number": binding.order_number, "purchase_scope": binding.purchase_scope,
            "manifest_content_hash": manifest.manifest_content_hash, "files": {f.path: f.sha256 for f in manifest.files},
            "out": out}


def write_layout(package: Any, out: str, *, locale: str, extra: dict[str, Any] | None = None,
                 save_package: bool = False) -> int:
    """Render ``package`` (a document, bytes or path), write it to ``out`` (with ``save_package`` also the package itself,
    ``optical-layout.json``) and print the summary; the exit code."""

    from ._vendor.shared_contracts.assembly_preview import json_file_bytes
    from .optical_layout import OpticalLayoutError, generate_optical_layout

    try:
        drawing = generate_optical_layout(package, locale=locale)
    except OpticalLayoutError as exc:
        print(json.dumps({"status": "ERROR", "error": exc.as_dict()}, indent=2, ensure_ascii=True))
        return LOCAL_INPUT_ERROR
    # the verified package, in canonical file form (what 'layout draw' renders again), written with the drawing as one set
    package_file = {PACKAGE_FILE: json_file_bytes(drawing.package.model_dump(mode="json", by_alias=True))} if save_package else None
    try:
        drawing.write(out, package_file)
    except OSError as exc:
        print(f"error: cannot write the layout drawing: {exc}", file=sys.stderr)
        return USAGE_ERROR
    saved = {"package_file": str(Path(out) / PACKAGE_FILE)} if save_package else {}
    print(json.dumps({**(extra or {}), **summary(drawing, out), **saved}, indent=2, ensure_ascii=True))
    return 0


def _draw(args: argparse.Namespace) -> int:
    from .optical_layout.generate import package_required

    if args.package is None:
        print(json.dumps({"status": "ERROR", "error": package_required("no package was given").as_dict()}, indent=2, ensure_ascii=True))
        return LOCAL_INPUT_ERROR
    if args.out is None:
        print("error: give --out DIR", file=sys.stderr)
        return USAGE_ERROR
    return write_layout(Path(args.package), args.out, locale=args.locale)


def add_output_options(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--out", required=required, help="output directory (created if needed); only the layout files are "
                                                         "written, replacing an earlier layout's files there")
    parser.add_argument("--locale", choices=LOCALE_CHOICES, default="en",
                        help="the language of the sheet: en (default) or zh-CN (Simplified Chinese)")
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)  # accepted; the files are always replaced


def register(sub: Any) -> None:
    """Add ``layout draw`` to the ``os-optics`` parser."""

    p_layout = sub.add_parser("layout", help="the Optical Layout Drawing of an optics-only purchase (relative optical layout)",
                              description=LAYOUT_DESCRIPTION)
    layout_sub = p_layout.add_subparsers(dest="layout_command", required=True)
    draw = layout_sub.add_parser("draw", help="render an Optical Layout Package file on this device (SVG, PDF, manifest)",
                                 description=LAYOUT_DESCRIPTION)
    draw.add_argument("package", metavar="PACKAGE_JSON", nargs="?", help="the Optical Layout Package (optical-layout.json)")
    add_output_options(draw, required=False)
    draw.set_defaults(handler=_draw)


__all__ = ["FETCH_DESCRIPTION", "LAYOUT_DESCRIPTION", "PACKAGE_FILE", "add_output_options", "register", "summary", "write_layout"]
