"""``os-optics native``: the native component library from the shell (``os_optics.native_library``).

``native list`` lists the classes, ``native describe CLASS`` prints one class (parameters with units, constraints and
defaults, physics, approximations, references) and ``native build CLASS --param name=value ...`` prints the built model,
its materials / coatings, a ready component instance and the design report -- or, with ``--into PROJECT.json``, adds
them to that Project file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

USAGE_ERROR = 64
NATIVE_INPUT_ERROR = 65
NATIVE_DESCRIPTION = (
    "Parametric optical components, independent of any catalog product (stops, slits, pinholes, annular apertures, central "
    "obstructions, ideal and designed lenses, achromats, axicons, lenses of a tabulated response, windows, cuvettes, wedge "
    "prisms, flat and curved mirrors, hollow roof mirrors, corner-cube retroreflectors, plate splitters, variable "
    "neutral-density filters, polarizers, waveplates, acousto-optic deflectors, telescopes and 4f relays) as Project optical models. A built "
    "component is simulated like any catalog optic; it is not priced or sourced (in an Order it is customer-supplied). "
    "'native list' also names the field models (a spatial light modulator's pupil mask, a camera's response), which act on the light "
    "a Project delivers and have no optical model of their own. "
    "Values are exact decimals; a request that cannot be built lists every problem."
)
BUILD_DESCRIPTION = (
    "Build one native component from its class and parameters: the Project optical model (and a compound unit's member "
    "models), the model materials and coatings it uses, and a component template that places it. Without --into the "
    "build is printed as JSON; with --into it is added to a Project file. In an Order a built component is "
    "customer-supplied. A request that cannot be built lists every problem, each naming its parameter."
)
BUILD_EPILOG = (
    "Parameter values: decimals as text (--param focal_length_mm=100), lists and objects as JSON (--param "
    "'valid_band_nm=[400,700]', --param 'material={\"constant_index\":\"1.5\",\"valid_band_nm\":[400,700]}'); a material may "
    "also be a public catalog material id (--param material=N-BK7). 'os-optics native describe CLASS' lists a class's "
    "parameters. Exit codes: 0 = built, 64 = usage error or unwritable file, 65 = refused (JSON error on stdout)."
)


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("native", help="native component library: list / describe classes, build a component model",
                                   description=NATIVE_DESCRIPTION)
    sub = parser.add_subparsers(dest="native_command", required=True)
    p_list = sub.add_parser("list", help="list the native component classes")
    p_list.add_argument("--category",
                        help="only this category (STOP, LENS, PLATE, PRISM, MIRROR, SPLITTER, FILTER, POLARIZATION, GRATING, MODULATOR, COMPOUND)")
    p_describe = sub.add_parser("describe", help="print one class: parameters, physics, approximations, references")
    p_describe.add_argument("class_id", help="a class id, e.g. lens.singlet, or a family of one class, e.g. grating")
    p_build = sub.add_parser("build", help="build one component: its Project optical model, materials, coatings and component",
                             description=BUILD_DESCRIPTION, epilog=BUILD_EPILOG)
    p_build.add_argument("class_id", help="a class id, e.g. aperture.circular")
    p_build.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="one parameter (repeat for each)")
    p_build.add_argument("--params-json", metavar="FILE", help="read the parameters from a JSON object file ('-' for stdin)")
    p_build.add_argument("--into", metavar="PROJECT", help="add the models / materials / coatings and the component to this Project "
                                                           "JSON file (rewritten in place when the result is a valid Project)")
    p_build.add_argument("--instance-id", help="with --into: the component's instance id (default: the model id)")
    p_build.add_argument("--position", metavar="X,Y,Z", help="with --into: the component's position in mm (default 0,0,0)")
    p_build.add_argument("--quaternion", metavar="W,X,Y,Z", help="with --into: the component's orientation (default 1,0,0,0)")
    p_build.add_argument("-o", "--output", help="write the build JSON here instead of stdout")
    for p in (p_list, p_describe, p_build):
        p.set_defaults(handler=command)
    return parser


def _out(doc: object) -> None:
    print(json.dumps(doc, indent=2, ensure_ascii=False))


def _refused(problems: list[dict[str, Any]]) -> int:
    _out({"status": "ERROR", "error": {"code": "NATIVE_COMPONENT_INVALID", "problems": problems}})
    return NATIVE_INPUT_ERROR


def _parameters(args: argparse.Namespace) -> dict[str, Any] | str:
    """The parameters given on the command line, or a usage message."""

    params: dict[str, Any] = {}
    if args.params_json:
        try:
            text = sys.stdin.read() if args.params_json == "-" else Path(args.params_json).read_text(encoding="utf-8")
            loaded = json.loads(text)
        except (OSError, ValueError) as exc:
            return f"cannot read --params-json: {exc}"
        if not isinstance(loaded, dict):
            return "--params-json must hold a JSON object"
        params.update(loaded)
    given: set[str] = set()
    for item in args.param:  # a --param replaces the same name from --params-json
        name, sep, value = item.partition("=")
        if not sep or not name:
            return f"--param takes NAME=VALUE, got {item!r}"
        if name in given:
            return f"parameter {name!r} is given twice"
        given.add(name)
        if value.lstrip().startswith(("{", "[")):
            try:
                params[name] = json.loads(value)
            except ValueError as exc:
                return f"--param {name}: not valid JSON ({exc})"
        else:
            params[name] = value
    return params


def _numbers(text: str | None, count: int, label: str) -> list[float] | str | None:
    if text is None:
        return None
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        return f"{label} takes {count} comma-separated numbers"
    return values if len(values) == count else f"{label} takes {count} comma-separated numbers"


def command(args: argparse.Namespace) -> int:
    from .native_library import NativeComponentError, add_to_project, build_native_model, describe_native_class, native_components, native_field_models

    if args.native_command == "list":
        rows = [{key: c[key] for key in ("class_id", "category", "title", "model_kind", "optical_role")} for c in native_components()
                if args.category is None or c["category"] == args.category.upper()]
        fields = [] if args.category else [{key: m[key] for key in ("model_id", "category", "title", "cli", "schema_kind")} for m in native_field_models()]
        _out({"classes": rows, "field_models": fields})
        return 0
    if args.native_command == "describe":
        field_model = next((m for m in native_field_models() if m["model_id"] == args.class_id), None)
        if field_model is not None:
            _out(field_model)
            return 0
        try:
            _out(describe_native_class(args.class_id))
        except NativeComponentError as exc:
            return _refused([p.as_dict() for p in exc.problems])
        return 0
    params = _parameters(args)
    position = _numbers(args.position, 3, "--position")
    quaternion = _numbers(args.quaternion, 4, "--quaternion")
    for problem in (params, position, quaternion):
        if isinstance(problem, str):
            print(f"error: {problem}", file=sys.stderr)
            return USAGE_ERROR
    if not args.into and (args.instance_id or position or quaternion):
        print("error: --instance-id, --position and --quaternion need --into", file=sys.stderr)
        return USAGE_ERROR
    try:
        build = build_native_model(args.class_id, params)  # type: ignore[arg-type]
    except NativeComponentError as exc:
        return _refused([p.as_dict() for p in exc.problems])
    if args.into:
        from .local.catalog import load_project
        from .local.errors import LocalRunError

        try:
            project = load_project(Path(args.into))
        except LocalRunError as exc:
            return _refused([{"code": p.code.value, "message": p.message, "parameter": "project"} for p in exc.problems])
        try:
            merged = add_to_project(project, build, instance_id=args.instance_id, position_mm=position, quaternion_wxyz=quaternion)  # type: ignore[arg-type]
        except NativeComponentError as exc:
            return _refused([p.as_dict() for p in exc.problems])
        try:
            Path(args.into).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.into}: {exc.strerror}", file=sys.stderr)
            return USAGE_ERROR
        component = merged["components"][-1]
        _out({"project": args.into, "component": component["instance_id"], "optical_model_id": build.optical_model.model_id,
              "added_models": [m.model_id for m in build.models], "materials": [m.material_id for m in build.model_materials],
              "coatings": [c.coating_id for c in build.model_coatings], "design": build.design})
        return 0
    document = build.as_dict()
    if args.output:
        try:
            Path(args.output).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.output}: {exc.strerror}", file=sys.stderr)
            return USAGE_ERROR
        _out({"class_id": build.class_id, "optical_model_id": build.optical_model.model_id, "output": args.output})
        return 0
    _out(document)
    return 0
