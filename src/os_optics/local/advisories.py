"""Advisory warnings of a Project check: facts about an accepted Project that decide part of its run's outcome.

``project_warnings(project, lowered)`` lists, before any run:

* ``TARGET_INVALID``, ``TARGET_PARAMETER_INVALID``, ``UNKNOWN_SOURCE_REFERENCE`` (location ``target <target_id>[
  parameters.<key>]``): a requirement that cannot be read as written (``lowering.target_problems``). The run reports
  that target ``UNKNOWN`` (assessment ``INDETERMINATE``) with the code and evaluates everything else as for the Project
  without it; ``details.target_id`` names it;
* ``TARGET_UNIT_MISMATCH`` (location ``target <target_id>``): the target states a ``unit`` other than its metric's unit
  (``target_metrics()``); the run reports that target's value in the metric's unit and does not compare the target's
  bounds: its status is ``UNKNOWN`` (a target the run cannot evaluate at all keeps that outcome). A cavity metric's unit is in ``validation.scope.CAVITY_METRIC_UNITS``: the cavity
  analysis gives that target no value (``INDETERMINATE``);
* ``SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND`` (location ``component <instance_id>[ member <member_id>]``, one per optical
  body): no line of any source of the Project lies inside the wavelengths the component's published or declared model
  is characterized for, so every beam that meets it is outside its model, whichever path it takes. That is all a check
  knows before a trace: a body some source line fits gets no warning, and which of the other lines reach it is
  reported by the run (``model_scope_issues``, per element and surface). For most models the wavelengths are the band
  (``valid_band_nm``, with the tolerance of a single design wavelength in force), and the run reports
  ``MODEL_WAVELENGTH_OUTSIDE_BAND`` in ``model_scope_issues``. For a ``TABULATED_THIN_LENS`` it is its table's domain as
  the Engine applies it (``response_at``: the rows' span; only the rows, each within 0.5 %, for ``POINTS_ONLY`` and a
  one-row table), and the run reports ``MODEL_WAVELENGTH_OUTSIDE_BAND`` as an Engine issue (``result.issues``, status
  ``OUT_OF_MODEL_SCOPE``) and in ``model_scope_issues``. A ``FARADAY_ROTATOR`` likewise, with its table's domain (the rows'
  span; only the rows, exactly, for ``POINTS_ONLY`` and a one-row table): that light passes the element straight on, with
  its power and polarization not computed behind it (power ``None``, bounded by the incident power).
  ``details``: ``instance_id``, ``member_id``, ``source_ids`` and ``wavelengths_nm`` (every source and line of the
  Project), ``lines`` (per source its lines), ``band_nm`` and ``reported_in`` (``model_scope_issues`` /
  ``engine_issues``: both).

* ``FRAME_SYMMETRY_DERIVED`` and ``DESIGN_GEOMETRY_EXTENDED`` (location ``optical model <model_id>``): a declared
  model's stated frame symmetry differs from what the model is (the derived one is used: ``details.stated``,
  ``details.derived``), or its stated cylinder or box ``design_geometry`` does not hold its surfaces (the smallest body
  of that kind that holds both is used: ``details.stated``, ``details.used``) (``lowering.model_notices``). The
  Engine request does not contain either, so the run's numbers are those of the Project stating the used facts.

* ``INPUT_NOT_SIMULATED`` (location ``materials`` or ``component <instance_id>``): an input the simulation does not read
  -- a ``selected_parameters`` entry that is not an attenuator position, an entry of the Project's ``materials`` list
  (``lowering.inputs_not_simulated``). The run's numbers are those of the Project without it; ``details`` names the
  ``field`` and the ``key``.

* ``CONFIGURATION_MOVES_FIXED_DOF`` (location ``configuration <id> component <instance_id>``, ``configuration_warnings``):
  a configuration of a laboratory assembly moves a component along or about a DOF it does not retain
  (``details.motions``). The configuration is simulated as declared; an assembly request of the Project is refused with
  the same code (retain those DOFs, or declare ``design_intent`` ``OPTICAL_SYSTEM``).

They never refuse a Project: a Project the check accepts stays accepted. Each warning is ``{code, message, location,
details}``.
"""

from __future__ import annotations

from typing import Any, Callable

from .compile import DESIGN_GEOMETRY_EXTENDED, FRAME_SYMMETRY_DERIVED

TARGET_UNIT_MISMATCH = "TARGET_UNIT_MISMATCH"
SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND = "SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND"
INPUT_NOT_SIMULATED = "INPUT_NOT_SIMULATED"
CONFIGURATION_MOVES_FIXED_DOF = "CONFIGURATION_MOVES_FIXED_DOF"
#: The codes of a requirement that cannot be read as written (``lowering.target_problems``).
TARGET_PROBLEM_CODES = ("TARGET_INVALID", "TARGET_PARAMETER_INVALID", "UNKNOWN_SOURCE_REFERENCE")
WARNING_CODES = (*TARGET_PROBLEM_CODES, TARGET_UNIT_MISMATCH, SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND, FRAME_SYMMETRY_DERIVED,
                 DESIGN_GEOMETRY_EXTENDED, INPUT_NOT_SIMULATED, CONFIGURATION_MOVES_FIXED_DOF)


def _source_lines(project: Any) -> list[tuple[str, float]]:
    """``(source instance_id, vacuum wavelength_nm)`` of every line of every source, in Project order."""

    lines: list[tuple[str, float]] = []
    for component in project.components:
        source = component.source_model
        if source is None:
            continue
        if getattr(source, "spectral_density", None) is not None:
            continue  # (a continuous spectrum's quadrature lines are not stated wavelengths)
        lines.extend((component.instance_id, float(wavelength)) for wavelength in source.wavelengths_nm)
    return lines


def _unit_warnings(project: Any) -> list[dict[str, Any]]:
    from ..validation.scope import CAVITY_METRIC_UNITS
    from .lowering import target_metrics

    units = {**target_metrics(), **CAVITY_METRIC_UNITS}
    out = []
    for target in project.targets:
        metric_unit = units.get(target.metric)
        if target.unit is None or not metric_unit or target.unit == metric_unit:
            continue
        consequence = ("the cavity analysis gives that target no value; its assessment is INDETERMINATE" if target.metric in CAVITY_METRIC_UNITS else
                       f"the run reports that target's value in {metric_unit!r} and does not compare the target's bounds; its status is UNKNOWN "
                       "(a target the run cannot evaluate at all keeps that outcome)")
        out.append({
            "code": TARGET_UNIT_MISMATCH,
            "message": (f"target {target.target_id} declares unit {target.unit!r}; metric {target.metric} is in {metric_unit!r}, so "
                        f"{consequence}. Set unit to {metric_unit!r} and give its bounds in {metric_unit}."),
            "location": f"target {target.target_id}",
            "details": {"target_id": target.target_id, "metric": target.metric, "unit": target.unit, "metric_unit": metric_unit},
        })
    return out


def _wavelength_domain(body: Any) -> tuple[Callable[[float], bool], str, str]:
    """``(in_domain(wavelength_nm), domain text, where the run reports a line outside it)`` of one optical body.

    A tabulated thin lens: its table exactly as the Engine applies it (``optics.surfaces.tabulated_lens.response_at``,
    as ``compile.paraxial_matrix`` does); the run's model-scope check skips it and the Engine reports the line as an
    issue of the run. A Faraday rotator: the same, with ``optics.polarization.faraday.FaradayResponse.at``. Every other model: ``validity.py``'s own band rule."""

    from ..optics.polarization.faraday import FaradayResponse
    from ..optics.surfaces.tabulated_lens import ThinLensTable, response_at
    from .compile import FARADAY_ROTATOR, TABULATED_THIN_LENS
    from .validity import _band_text, _decimal, _in_band, wavelength_tolerance

    if getattr(body.model, "model_kind", None) == TABULATED_THIN_LENS:
        interaction = body.surfaces[0].interaction
        table = ThinLensTable(interaction.wavelength_nm, interaction.focal_length_mm, interaction.transmittance,
                              interaction.focal_interpolation, interaction.transmittance_interpolation, interaction.point_tolerance)
        return (lambda w: response_at(table, w)[2]), table.domain_text(), "engine_issues"
    if getattr(body.model, "model_kind", None) == FARADAY_ROTATOR:
        interaction = body.surfaces[0].interaction
        response = FaradayResponse(interaction.wavelength_nm, interaction.rotation_rad, interaction.transmittance, interaction.interpolation)
        return (lambda w: response.at(w) is not None), response.domain_text(), "engine_issues"
    band, tolerance = body.valid_band_nm, wavelength_tolerance(body)
    return ((lambda w: _in_band(band, _decimal(w), tolerance)), f"the {_band_text(band.min_nm, band.max_nm, tolerance)} band",
            "model_scope_issues")


_REPORTED_IN = {"model_scope_issues": "in model_scope_issues",
                "engine_issues": "as an Engine issue of the run (result.issues, status OUT_OF_MODEL_SCOPE) and in model_scope_issues"}


def _band_warnings(project: Any, lowered: Any) -> list[dict[str, Any]]:
    """One warning per optical body no source line of the Project fits (module docstring)."""

    lines = _source_lines(project)
    out: list[dict[str, Any]] = []
    if not lines:
        return out
    source_ids = list(dict.fromkeys(s for s, _ in lines))
    wavelengths = sorted({w for _, w in lines})
    by_source = [{"source_id": source_id, "wavelengths_nm": sorted({w for s, w in lines if s == source_id})} for source_id in source_ids]
    for element in lowered.elements:
        for body in element.bodies():
            band = body.valid_band_nm
            if band is None:  # (an aperture stop, a compound unit; a grating's directions hold at every wavelength)
                continue
            in_domain, domain, reported_in = _wavelength_domain(body)
            if any(in_domain(w) for w in wavelengths):
                continue  # (some line fits: which of the others reach the body is known after the trace, and reported there)
            member = f" member {body.member_id}" if body.member_id is not None else ""
            published = "published" if body.identity.is_catalog else "declared"
            listed = ", ".join(f"{w:g} nm" for w in wavelengths)
            out.append({
                "code": SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND,
                "message": (f"no source line of the Project ({listed}) lies inside {domain} of the {published} model of "
                            f"{element.instance_id}{member} ({body.label}): every beam that meets {element.instance_id} is outside "
                            f"that model, and the run reports MODEL_WAVELENGTH_OUTSIDE_BAND {_REPORTED_IN[reported_in]}."),
                "location": f"component {element.instance_id}{member}",
                "details": {"instance_id": element.instance_id, "member_id": body.member_id, "source_ids": source_ids,
                            "wavelengths_nm": wavelengths, "lines": by_source,
                            "band_nm": {"min_nm": band.min_nm, "max_nm": band.max_nm}, "reported_in": reported_in},
            })
    return out


#: What the simulation reads instead of an input it passes over, by the input's field.
_READ_INSTEAD = {
    "selected_parameters": "a source is simulated from its source_model, a catalog product from its published model and a "
                           "declared element from its optical model",
    "materials": "the media of a declared model are the Project's model_materials, a catalog product's those of its release",
}


def _input_warnings(lowered: Any) -> list[dict[str, Any]]:
    out = []
    for row in lowered.inputs_not_simulated:
        what = (f"selected parameter {row['key']!r} of {row['location']}" if row["field"] == "selected_parameters" else
                f"materials entry {row['key']!r}")
        out.append({
            "code": INPUT_NOT_SIMULATED,
            "message": f"{what} is not simulation input: {_READ_INSTEAD[row['field']]}. The run's results are those of the "
                       "Project without it.",
            "location": row["location"],
            "details": {"field": row["field"], "key": row["key"]},
        })
    return out


def _target_warnings(lowered: Any) -> list[dict[str, Any]]:
    out = []
    for problem in lowered.target_problems:
        row = problem.as_dict()
        row["message"] = f"{problem.message.rstrip('.')}. The run reports this target UNKNOWN (INDETERMINATE) until it is corrected."
        row["details"] = {**row.get("details", {}), "target_id": problem.location.split(" ", 2)[1]}
        out.append(row)
    return out


def _model_warnings(lowered: Any) -> list[dict[str, Any]]:
    out = []
    for notice in lowered.model_notices:
        details = {key: value for key, value in notice.items() if key != "code"}
        model_id = notice["model_id"]
        if notice["code"] == FRAME_SYMMETRY_DERIVED:
            message = (f"model {model_id} states frame symmetry {notice['stated']}, but the model is {notice['derived']}: the run, "
                       f"the view and the solid check use {notice['derived']}. Set frame.symmetry to {notice['derived']}.")
        else:
            message = (f"the design_geometry of model {model_id} does not hold its surfaces over the whole outline: the solid "
                       f"check, the view and the reserved envelope use the smallest {notice['used']['kind'].lower()} that holds "
                       "both (details.used). Omit design_geometry to have it derived, or state a body that holds the surfaces.")
        out.append({"code": notice["code"], "message": message, "location": f"optical model {model_id}", "details": details})
    return out


def project_warnings(project: Any, lowered: Any) -> list[dict[str, Any]]:
    """The advisory warnings of an accepted Project (module docstring): targets first, then elements in Project order,
    then the declared models' replaced facts, then the inputs the simulation does not read."""

    return (_target_warnings(lowered) + _unit_warnings(project) + _band_warnings(project, lowered) + _model_warnings(lowered)
            + _input_warnings(lowered))


def configuration_warnings(project: Any, configuration_id: str) -> list[dict[str, Any]]:
    """``CONFIGURATION_MOVES_FIXED_DOF`` of one configuration of ``project`` (module docstring)."""

    out = []
    for cid, instance_id, motions in project.configuration_fixed_dof_motions():
        if cid != configuration_id:
            continue
        out.append({
            "code": CONFIGURATION_MOVES_FIXED_DOF,
            "message": (f"configuration {cid}: component {instance_id} {'; '.join(motions)}. The configuration is simulated as "
                        "declared; an assembly request is refused, because in a laboratory assembly a configuration moves a "
                        "component only along and about its retained DOFs. Retain those DOFs, or declare design_intent "
                        "OPTICAL_SYSTEM."),
            "location": f"configuration {cid} component {instance_id}",
            "details": {"configuration_id": cid, "instance_id": instance_id, "motions": list(motions)},
        })
    return out


__all__ = ["CONFIGURATION_MOVES_FIXED_DOF", "INPUT_NOT_SIMULATED", "SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND",
           "TARGET_UNIT_MISMATCH", "WARNING_CODES", "configuration_warnings", "project_warnings"]
