"""``os-optics project import-prescription``: a lens prescription file -> Project-native optical models.

Prints the import as JSON (``os_optics.prescription.api``): the ``COMPOUND`` unit and its member models, the
materials and coatings they use, the suggested component and the import report. With ``--into PROJECT.json`` the import
is added to that Project (``--instance-id``, ``--pose``) and the updated Project is written to ``--out`` (or printed).
The CSV format is described in ``os_optics.prescription.csv_table``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .prescription import PRESCRIPTION_FORMATS, DEFAULT_MODEL_ID

USAGE_ERROR = 64
IMPORT_REFUSED = 65
DESCRIPTION = (
    "Import a sequential lens prescription (a CSV surface table, format csv-surface-table/1) as Project-native optical models: one "
    "COMPOUND unit whose members are the physical elements (SURFACE_SEQUENCE) and aperture stops (APERTURE_STOP), with the "
    "materials and coatings they use. Glasses the file does not define are copied from the public optical catalog."
)


def register_project(project_commands: argparse._SubParsersAction) -> None:
    """Add ``os-optics project import-prescription``."""

    command = project_commands.add_parser(
        "import-prescription", help="import a lens prescription (CSV surface table) as Project-native optical models",
        description=DESCRIPTION,
        epilog="exit codes: 0 = imported, 65 = the prescription (or the Project) is refused (JSON problems on stdout), "
               "64 = usage error / unreadable or unwritable file")
    command.add_argument("file", help="the prescription file (e.g. lens.csv)")
    command.add_argument("--format", default=PRESCRIPTION_FORMATS[0], choices=list(PRESCRIPTION_FORMATS), help="the file's format")
    command.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="id of the imported unit; members are <id>-L1, <id>-STOP, ...")
    command.add_argument("--units", help="length unit to read the file in (mm, cm, m, in; default: the file's own units "
                         "directive, else mm); a file that states another unit is refused")
    command.add_argument("--wavelength-min-nm", help="the band the models are valid for (with --wavelength-max-nm)")
    command.add_argument("--wavelength-max-nm")
    command.add_argument("--glass-source", default="catalog", choices=["catalog", "inline"],
                         help="catalog: glasses the file does not define come from the public optical catalog; inline: only the file's own")
    command.add_argument("--catalog", help="use this public optical catalog JSON instead of the active release")
    command.add_argument("--ignore-column", action="append", default=[], metavar="NAME",
                         help="an extra column of the file the import does not read (repeatable); listed in report.ignored_columns")
    command.add_argument("--into", help="add the import to this Project JSON file")
    command.add_argument("--instance-id", help="with --into: the component's instance id (default: the model id)")
    command.add_argument("--pose", help="with --into: the component's pose, 'x,y,z' (mm) or 'x,y,z,w,qx,qy,qz' (position and unit quaternion)")
    command.add_argument("--out", help="with --into: write the updated Project here (default: include it in the printed result)")
    command.set_defaults(handler=_import)


def _out(doc: object) -> None:
    print(json.dumps(doc, indent=2, ensure_ascii=True))


def _pose(text: str) -> dict | None:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        return None
    if len(values) == 3:
        return {"position_mm": values, "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    if len(values) == 7:
        return {"position_mm": values[:3], "quaternion_wxyz": values[3:]}
    return None


def _import(args: argparse.Namespace) -> int:
    from .local import load_public_catalog
    from .prescription.api import import_prescription_json

    try:
        text = Path(args.file).read_bytes()
        into = json.loads(Path(args.into).read_text(encoding="utf-8")) if args.into else None
        catalog_bytes = Path(args.catalog).read_bytes() if args.catalog else None
    except (OSError, ValueError) as exc:
        print(f"error: cannot read an input file: {exc}", file=sys.stderr)
        return USAGE_ERROR
    if (args.wavelength_min_nm is None) != (args.wavelength_max_nm is None):
        print("error: give both --wavelength-min-nm and --wavelength-max-nm", file=sys.stderr)
        return USAGE_ERROR
    if into is None and (args.instance_id or args.pose or args.out):
        print("error: --instance-id, --pose and --out need --into PROJECT.json", file=sys.stderr)
        return USAGE_ERROR
    options: dict = {"format": args.format, "model_id": args.model_id, "glass_source": args.glass_source}
    if args.units is not None:
        options["units"] = args.units
    if args.ignore_column:
        options["ignore_columns"] = list(args.ignore_column)
    if args.wavelength_min_nm is not None:
        options["wavelength_band_nm"] = {"min_nm": args.wavelength_min_nm, "max_nm": args.wavelength_max_nm}
    if into is not None:
        options["into"] = into
        if args.instance_id:
            options["instance_id"] = args.instance_id
        if args.pose:
            pose = _pose(args.pose)
            if pose is None:
                print("error: --pose is 'x,y,z' or 'x,y,z,w,qx,qy,qz'", file=sys.stderr)
                return USAGE_ERROR
            options["pose"] = pose

    def catalog():
        if catalog_bytes is not None:
            return load_public_catalog(catalog_bytes)
        return load_public_catalog()

    result = import_prescription_json(text, options, catalog=catalog)
    if not result["ok"]:
        _out(result)
        return IMPORT_REFUSED
    if args.out and "project" in result:
        try:
            Path(args.out).write_text(json.dumps(result["project"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write the Project: {exc}", file=sys.stderr)
            return USAGE_ERROR
        result = {**{k: v for k, v in result.items() if k != "project"}, "project_written": args.out}
    _out(result)
    return 0
