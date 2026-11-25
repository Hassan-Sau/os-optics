"""JSON-in / JSON-out prescription import (the body of ``local.json_api.import_prescription`` and of the CLI).

``import_prescription_json(text, options, catalog)`` -> ``{"ok": True, "format", "model_id", "optical_models",
"model_materials", "model_coatings", "component", "report"}`` (plus ``"environment"`` when the prescription states its
reference environment -- ``merge_into_project`` sets it on the Project --, and ``"project"`` and ``"project_hash"`` when
the options name a Project to import into; ``report.glasses`` says how each glass resolved, ``report.wavelengths`` how the
file wrote its wavelengths), or ``{"ok": False, "error": {"code": "PRESCRIPTION_IMPORT_INVALID",
"problems": [{"code", "message", "location": {"row", "column", "surface", "section"}}]}}``.

Options (all optional): ``format`` (default ``csv-surface-table/1``), ``model_id`` (default ``prescription``),
``wavelength_band_nm`` (``{"min_nm", "max_nm"}``, decimal strings or numbers), ``units`` (the length unit to read the
file in; default: the unit the file states, else ``mm``; a file stating another unit is refused with ``UNITS_CONFLICT``), ``glass_source`` (``catalog`` default, or ``inline``), ``into`` (a Project document),
``instance_id`` and ``pose`` (``{"position_mm": [x, y, z], "quaternion_wxyz": [w, x, y, z]}``; only with ``into``),
``ignore_columns`` (a list of extra column names the import does not read; listed in ``report.ignored_columns``),
``edge_review_ratio`` (edge thickness over clear diameter above which an element's fact row says ``above_review_ratio``;
default 0.35).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.common.geometry import PoseV1
from . import DEFAULT_MODEL_ID, PrescriptionImportError, import_prescription, merge_into_project
from .csv_table import FORMAT_ID, NUMBER_PATTERN
from .errors import PrescriptionErrorCode as E, refuse
from .glass import CatalogSource

OPTION_KEYS = ("format", "model_id", "wavelength_band_nm", "units", "glass_source", "into", "instance_id", "pose", "ignore_columns",
               "edge_review_ratio")


def _decimal_option(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise refuse(E.OPTIONS_INVALID, f"{label} must be a number or a decimal string")
    text = str(value).strip()
    if not NUMBER_PATTERN.match(text):
        raise refuse(E.OPTIONS_INVALID, f"{label} {value!r} is not a number")
    return Decimal(text)


def _band(value: Any) -> tuple[Decimal, Decimal] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"min_nm", "max_nm"} or not {"min_nm", "max_nm"} <= set(value):
        raise refuse(E.OPTIONS_INVALID, "wavelength_band_nm is {\"min_nm\": ..., \"max_nm\": ...}")
    return _decimal_option(value["min_nm"], "wavelength_band_nm.min_nm"), _decimal_option(value["max_nm"], "wavelength_band_nm.max_nm")


def import_prescription_json(text: Any, options: Any = None, catalog: CatalogSource | None = None) -> dict[str, Any]:
    """Module docstring."""

    try:
        options = {} if options is None else options
        if not isinstance(options, dict):
            raise refuse(E.OPTIONS_INVALID, "options must be an object")
        unknown = sorted(set(options) - set(OPTION_KEYS))
        if unknown:
            raise refuse(E.OPTIONS_INVALID, f"unknown option(s) {', '.join(map(repr, unknown))}; options: {', '.join(OPTION_KEYS)}")
        into = options.get("into")
        if into is None and (options.get("instance_id") is not None or options.get("pose") is not None):
            raise refuse(E.OPTIONS_INVALID, "instance_id and pose place the import in a Project: give the Project as 'into'")
        pose = None
        if options.get("pose") is not None:
            try:
                pose = PoseV1.model_validate(options["pose"])
            except ValidationError as exc:
                raise refuse(E.OPTIONS_INVALID, "pose is not a valid pose: " + "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or 'pose'}: {str(e['msg']).removeprefix('Value error, ')}"
                    for e in exc.errors(include_url=False))) from None
        instance_id = options.get("instance_id")
        if instance_id is not None and not isinstance(instance_id, str):
            raise refuse(E.OPTIONS_INVALID, "instance_id must be a string")
        ignore = options.get("ignore_columns", [])
        if not isinstance(ignore, list) or not all(isinstance(c, str) for c in ignore):
            raise refuse(E.OPTIONS_INVALID, "ignore_columns is a list of column names")
        if not isinstance(text, (str, bytes)):
            raise refuse(E.TEXT_INVALID, "the prescription must be text")
        imported = import_prescription(
            text, format=options.get("format", FORMAT_ID), model_id=options.get("model_id", DEFAULT_MODEL_ID),
            wavelength_band_nm=_band(options.get("wavelength_band_nm")), units=options.get("units"),
            glass_source=options.get("glass_source", "catalog"), catalog=catalog, ignore_columns=tuple(ignore),
            edge_review_ratio=options.get("edge_review_ratio"))
        result: dict[str, Any] = {"ok": True, **imported.as_dict()}
        if into is not None:
            project = merge_into_project(into, imported, instance_id=instance_id, pose=pose)
            result["project"] = project.payload()
            result["project_hash"] = project.content_hash()
        return result
    except PrescriptionImportError as exc:
        return {"ok": False, "error": exc.as_dict()}
    except ValidationError as exc:  # a contract refusal the checks above did not foresee: still a structured refusal
        problem = refuse(E.MODEL_INVALID, "the imported models are refused by the Project contract: " + "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'the document'}: {str(e['msg']).removeprefix('Value error, ')}"
            for e in exc.errors(include_url=False)))
        return {"ok": False, "error": problem.as_dict()}


__all__ = ["OPTION_KEYS", "import_prescription_json"]
