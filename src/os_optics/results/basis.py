"""Reading the value basis of a result: ``basis_of(record, field)`` and the projections that carry it.

``basis_of`` returns the entry a record stores for a field (``value_basis[field]``, else ``value_basis["*"]``), else the
entry derived from the record's flat fields -- the facts a result already holds in its own fields (``quantity_state``,
``power_complete``, ``untraced_power_W``, ``power_upper_bound_W``, ``power_lower_bound_W``, ``power_out_of_scope_at``,
``power_taken_before_*``, ``detected_power_min_W``, ``coherence_factor_min`` / ``_max``, ...) and does not store twice --,
else ``None``. ``None`` means the number is the model's value (or there is no number). Runs written before the value
basis existed answer in the same shape.

The derivation (``DERIVATIONS``; one function per carrier and field):

=====================================  =======================================  ===============================================
record / field                         flat facts                               entry
=====================================  =======================================  ===============================================
``ObservationResult.total_power_W``    state ``KNOWN`` / ``KNOWN_ZERO`` /        none
                                       ``NOT_APPLICABLE``, or no number
                                       ``power_out_of_scope_at`` of a power     ``ESTIMATE``: ``BEAM_PART_ELSEWHERE`` at those
                                       (state ``OUT_OF_MODEL_SCOPE``)           surfaces; for a surface whose data a beam met
                                                                                outside its stated range (the beam's stored
                                                                                entry), ``HELD_AT_NEAREST_STATED_POINT`` and
                                                                                the range checks of that entry instead
                                       state ``NOT_COMPUTED``                   ``LOWER_BOUND``: ``NOT_ACCOUNTED``
                                       ``power_lower_bound_W`` (part of a       ``UPPER_BOUND``: ``BEAM_PART_ELSEWHERE`` with
                                       beam taken before the plane)             ``upper`` = value - lower bound; ``ESTIMATE``
                                                                                when left-out light may also add
                                       left-out light may add (``BOUNDED`` /    ``LOWER_BOUND``: ``NOT_FOLLOWED`` (untraced
                                       ``UNKNOWN``)                             light) or ``POSSIBLE_RETURN`` (only light left
                                                                                out as negligible, whose returns or crossings
                                                                                may bring more than the negligible level);
                                                                                ``upper`` = ``power_upper_bound_W`` - value
                                                                                when bounded
``SpectralPower.power_W``              ``complete`` false                       as the plane's power, per line
``ArrivalResult`` ``beam.power_W``,    ``power_out_of_scope_at``                ``ESTIMATE``: ``BEAM_PART_ELSEWHERE``
``detected_power_W``                   ``power_taken_before_at``                ``UPPER_BOUND``: ``BEAM_PART_ELSEWHERE``,
                                                                                ``upper`` = ``power_taken_before_upper_W``
                                       ``detected_power_min_W``                 ``UPPER_BOUND``: ``BEAM_PART_ELSEWHERE``,
                                                                                ``upper`` = value - minimum
``InterferenceResult.coherence_factor``  ``coherence_factor_min`` / ``_max``    ``ESTIMATE``: ``UNSTATED_GROUP_DELAY_ZERO``
(and the visibilities and peak irradiance stated at it, the margin, and the combined power -- not where
``counter_propagating``: Pa + Pb has no cross term)
``CoherentSuperposition.total_power_W``  ``total_power_min_W`` / ``_max_W``     ``ESTIMATE``: ``UNSTATED_GROUP_DELAY_ZERO``
=====================================  =======================================  ===============================================

A model-scope finding of the run (a published model used outside its validity, ``local.validity``) is not a flat fact of
the Engine result: the results add it (``scoped_observation``) where it names a plane. The plane's ``total_power_W`` (it
reads every beam on the plane), the power of each arrival the finding reaches (``affected_arrivals``) and the power of each
line whose light it reaches carry an ``ESTIMATE`` with the finding's validity checks of that light (``validity_reach``),
merged with the entry above; an arrival or a line the finding does not reach keeps its own. An arrival the finding reaches
carries its checks on every other number whose kind of quantity the finding touches (``arrival_field_kinds``,
``finding_quantity_kinds``): its Gaussian state under ``"*"``, its position, direction, path lengths, spot and polarization
state -- not on power alone. A coherent sum whose beams it reaches carries them on its powers, a two-beam interference
record (``scoped_interference``) on the numbers of the pair metrics whose kind it touches. A finding that lists no arrival
on the plane (sampled light) reaches a line only where sampled light of that wavelength is there and its wavelength check
names it.

Carriage (``CARRIAGE``): every projection that carries one of these quantities carries its entry under ``value_basis``,
keyed by the projection's own name of the number, or carries null for the number; the Local Validation record projection
is registered bare by decision (it carries no entry; under the decision rule no verdict in a record rests on a number that
is not the model's value). ``row_bases`` builds the ``value_basis`` member of a row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..contracts.basis import WHOLE_RECORD, Assumed, Omitted, ValidityCheck, ValueBasis, merge_bases
from ..contracts.detection import DetectionQuantity
from ..contracts.optics import GaussianBeamState
from ..contracts.results import (
    ArrivalResult,
    CoherentSuperposition,
    ExposureHit,
    InterferenceResult,
    ObservationResult,
    PolarizationMixture,
    RatingComparison,
    SampledLightAtPlane,
    SpectralPower,
    TargetResult,
)
from ..errors.status import AssumedInput, OmittedPart, QuantityState, ValueKind

_NO_ENTRY = frozenset({QuantityState.KNOWN, QuantityState.KNOWN_ZERO, QuantityState.NOT_APPLICABLE})


def _left_out(value: float, upper_bound: float | None, *, faint_only: bool) -> Omitted:
    """The light a counted power leaves out: light not followed, or (``faint_only``) only light left out as negligible --
    which leaves a power incomplete only where its returns or crossings may bring more than the lines' negligible level
    (``validation/power.LineFate.faint_accounted``): its possible return."""

    upper = None if upper_bound is None else max(0.0, upper_bound - value)
    if not faint_only:
        return Omitted(part=OmittedPart.NOT_FOLLOWED, upper=upper)
    return Omitted(part=OmittedPart.POSSIBLE_RETURN, upper=upper)


def _held(scope: Iterable[str], stored: Iterable[ValueBasis]) -> ValueBasis | None:
    """The entry of a plane's power behind the surfaces ``scope`` when beams on it state, in their stored entries, that they
    met the data of some of those surfaces outside its stated range and went on with its values at the nearest stated point
    (``validation/observations.py`` ``held_basis``): those assumptions and range checks, and ``BEAM_PART_ELSEWHERE`` for the
    other surfaces. ``None`` when no beam states one."""

    entries = [entry for entry in stored if any(a.input is AssumedInput.HELD_AT_NEAREST_STATED_POINT for a in entry.assumed)]
    if not entries:
        return None
    assumed = tuple(dict.fromkeys(a for entry in entries for a in entry.assumed))
    held_at = {surface for a in assumed for surface in a.at}
    rest = tuple(surface for surface in dict.fromkeys(scope) if surface not in held_at)
    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=assumed,
                      validity=tuple(dict.fromkeys(check for entry in entries for check in entry.validity)),
                      omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=rest),) if rest else ())


def _plane_power(o: ObservationResult) -> ValueBasis | None:
    state, value = o.quantity_state, o.total_power_W
    if value is None or state is None or state in _NO_ENTRY:
        return None
    scope = sorted({s for a in o.arrivals for s in a.power_out_of_scope_at}
                   | {s for line in o.sampled_light for s in line.power_out_of_scope_at})
    if state is QuantityState.OUT_OF_MODEL_SCOPE or scope:
        held = _held(scope, (entry for a in o.arrivals if (entry := stored_basis(a, "beam.power_W")) is not None))
        if held is not None:
            return held
        return ValueBasis(kind=ValueKind.ESTIMATE, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(scope)),))
    if state is QuantityState.NOT_COMPUTED:
        return ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.NOT_ACCOUNTED),))
    omitted: list[Omitted] = []
    taken = o.power_lower_bound_W is not None
    if taken:
        at = tuple(sorted({s for a in o.arrivals for s in a.power_taken_before_at}))
        omitted.append(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=at, upper=max(0.0, value - o.power_lower_bound_W)))
    adds = not o.power_complete
    if adds:
        untraced = o.untraced_power_W or 0.0
        omitted.append(_left_out(value, o.power_upper_bound_W, faint_only=untraced == 0.0))
    if not omitted:
        return None
    kind = ValueKind.ESTIMATE if taken and adds else (ValueKind.UPPER_BOUND if taken else ValueKind.LOWER_BOUND)
    return ValueBasis(kind=kind, omitted=tuple(omitted))


def _spectral_power(s: SpectralPower) -> ValueBasis | None:
    if s.power_W is None:
        return None
    if s.power_out_of_scope_at:
        return ValueBasis(kind=ValueKind.ESTIMATE,
                          omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(s.power_out_of_scope_at)),))
    if s.complete:
        return None
    if s.untraced_power_W is None:
        return ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.NOT_ACCOUNTED),))
    return ValueBasis(kind=ValueKind.LOWER_BOUND,
                      omitted=(_left_out(s.power_W, None, faint_only=s.untraced_power_W == 0.0),))


def _arrival_power(a: ArrivalResult, detected: bool) -> ValueBasis | None:
    value = a.detected_power_W if detected else a.beam.power_W
    if value is None:
        return None
    if a.power_out_of_scope_at:
        return ValueBasis(kind=ValueKind.ESTIMATE,
                          omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(a.power_out_of_scope_at)),))
    omitted: list[Omitted] = []
    if a.power_taken_before_at:
        omitted.append(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(a.power_taken_before_at),
                               upper=a.power_taken_before_upper_W))
    if detected and a.detected_power_min_W is not None:
        # (an annular beam counted whole: the capture level decides that the share beside the area is left out)
        omitted.append(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, upper=max(0.0, value - a.detected_power_min_W),
                               setting="capture_min_power_fraction"))
    if not omitted:
        return None
    return ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=tuple(omitted))


_GROUP_DELAY_ZERO = Assumed(input=AssumedInput.UNSTATED_GROUP_DELAY_ZERO)


def _coherence(i: InterferenceResult, field: str) -> ValueBasis | None:
    if getattr(i, field) is None or i.coherence_factor_min is None or i.coherence_factor_max is None:
        return None
    if field == "combined_power_collinear_W" and i.counter_propagating:
        return None  # (Pa + Pb: beams crossing the plane in opposite directions have no cross term, whatever their coherence)
    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=(_GROUP_DELAY_ZERO,))


def _superposition(s: CoherentSuperposition) -> ValueBasis | None:
    if s.total_power_W is None or s.total_power_min_W is None or s.total_power_max_W is None:
        return None
    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=(_GROUP_DELAY_ZERO,))


#: The derivation table: ``(record type, field) -> entry from the record's flat fields``.
DERIVATIONS: dict[tuple[type, str], Callable[[Any], ValueBasis | None]] = {
    (ObservationResult, "total_power_W"): _plane_power,
    (SpectralPower, "power_W"): _spectral_power,
    (ArrivalResult, "beam.power_W"): lambda a: _arrival_power(a, detected=False),
    (ArrivalResult, "detected_power_W"): lambda a: _arrival_power(a, detected=True),
    **{(InterferenceResult, name): (lambda i, name=name: _coherence(i, name))
       for name in ("coherence_factor", "fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2",
                    "coherence_margin", "combined_power_collinear_W")},
    (CoherentSuperposition, "total_power_W"): _superposition,
}


def stored_basis(record: Any, field: str) -> ValueBasis | None:
    """The entry ``record`` stores for ``field`` (or for ``"*"``), without derivation."""

    bases = getattr(record, "value_basis", None)
    if not bases:
        return None
    if isinstance(bases, ValueBasis):  # a record that is one quantity (contracts.analysis.Quantity)
        return bases
    return bases.get(field) or bases.get(WHOLE_RECORD)


def basis_of(record: Any, field: str) -> ValueBasis | None:
    """What the number in ``record``'s ``field`` is: the stored entry, else the entry derived from the record's flat fields
    (``DERIVATIONS``), else ``None`` -- the model's value, or no number at all (``field`` holds null)."""

    stored = stored_basis(record, field)
    if stored is not None:
        return stored
    derive = DERIVATIONS.get((type(record), field))
    return None if derive is None else derive(record)


def bases_of(record: Any) -> dict[str, ValueBasis]:
    """Every entry of ``record``: the stored ones and those derived from its flat fields for the fields it stores none for."""

    out: dict[str, ValueBasis] = {}
    stored = getattr(record, "value_basis", None)
    if isinstance(stored, Mapping):
        out.update(stored)
    for (kind, field), derive in DERIVATIONS.items():
        if isinstance(record, kind) and field not in out and WHOLE_RECORD not in out:
            entry = derive(record)
            if entry is not None:
                out[field] = entry
    return out


def dump_with_bases(record: Any, **dump: Any) -> dict[str, Any]:
    """``record.model_dump(mode="json", **dump)`` with its derived entries added to ``value_basis`` (the stored ones are in the
    dump already)."""

    doc = record.model_dump(mode="json", **dump)
    entries = bases_of(record)
    if entries:
        doc["value_basis"] = {key: entry.model_dump(mode="json") for key, entry in entries.items()}
    return doc


def row_bases(pairs: Iterable[tuple[str, Any, str]]) -> dict[str, Any]:
    """The ``value_basis`` member of a projection row: ``{"value_basis": {row key: entry}}`` for each ``(row key, record,
    field)`` whose number has an entry (``basis_of``), or ``{}`` when none has, so the row keeps its earlier form."""

    entries = {key: entry.model_dump(mode="json") for key, record, field in pairs if (entry := basis_of(record, field)) is not None}
    return {"value_basis": entries} if entries else {}


def row_basis_cell(pairs: Iterable[tuple[str, Any, str]]) -> dict[str, Any] | None:
    """The ``value_basis`` column of a table row: the entries keyed by column name, or ``None`` (an empty cell)."""

    return row_bases(pairs).get("value_basis")


# ----------------------------------------------------------------------------------------------- model-scope findings

Arrival = tuple[str, str, int]


def _finding_row(issue: Any) -> Mapping[str, Any]:
    """A model-scope finding (``local.validity.ModelScopeIssue``, its record form or its row) as its row."""

    if hasattr(issue, "as_dict"):
        return issue.as_dict()
    if hasattr(issue, "model_dump"):
        return issue.model_dump(mode="json")
    return issue


def _arrival_keys(rows: Any) -> set[Arrival]:
    return {(a["observation_id"], a["path_id"], a["crossing_index"]) for a in rows or ()
            if isinstance(a, Mapping) and {"observation_id", "path_id", "crossing_index"} <= a.keys()}


#: A sampled line at a plane: ``(observation_id, source_id, spectral_index)`` (a finding's ``affected_sampled_lines``).
Line = tuple[str, str, int]


def _line_keys(rows: Any) -> set[Line]:
    return {(r["observation_id"], r["source_id"], r["spectral_index"]) for r in rows or ()
            if isinstance(r, Mapping) and {"observation_id", "source_id", "spectral_index"} <= r.keys()}


def plane_lines(o: ObservationResult, wavelength_nm: float | None = None) -> set[Line]:
    """The sampled lines on the plane ``o`` (of ``wavelength_nm`` only, when given)."""

    return {(o.observation_id, line.source_id, line.spectral_index) for line in o.sampled_light
            if wavelength_nm is None or line.wavelength_nm == wavelength_nm}


#: The parameter of a validity check on the wavelength of the light (``ValidityCheck.observed``: that wavelength, nm).
WAVELENGTH_CHECK = "wavelength_nm"


def scope_checks(findings: Iterable[Any], plane: str, light: set[Arrival] | None = None, *,
                 unlocalised: bool = False, wavelength_nm: float | None = None,
                 touches: frozenset[str] | None = None, lines: set[Line] | None = None) -> tuple[ValidityCheck, ...]:
    """The validity checks of the model-scope findings ``findings`` that qualify a number read on the plane ``plane``.

    ``light`` ``None``: the plane's total, which reads every beam on it -- every finding that names the plane
    (``affected_observation_ids``), with the checks whose light reached an arrival on it (``validity_reach``). ``light`` a
    set of arrivals ``(observation_id, path_id, crossing_index)``: an arrival's power, or a line's -- the findings whose
    ``affected_arrivals`` include one of them, with the checks whose light is among them. ``lines``: the sampled lines
    ``(observation_id, source_id, spectral_index)`` the number reads -- the findings whose ``affected_sampled_lines``
    include one of them, with the checks whose light is among them (with ``light`` ``None``: every line the finding lists on
    the plane). ``unlocalised``: for a finding that lists no sampled line (one of an earlier Engine), a finding that names
    the plane but lists no arrival on it (light sampled by an extended emitter, named by its planes only) also qualifies
    the number; for a line (``wavelength_nm``) only where a wavelength check of the finding observed that line's
    wavelength, or the finding states no wavelength check (its other checks stay). A finding that does not say per check,
    or whose checks name none of that light, gives all of its checks; one listed without checks (a run of an earlier Engine)
    a check of its code at its element. ``touches``: the kinds of quantity the number reads (``QuantityKind`` values): only
    the findings whose code touches one of them (``finding_quantity_kinds``) qualify it."""

    from ..local.validity import finding_quantity_kinds

    checks: list[ValidityCheck] = []
    for issue in findings:
        row = _finding_row(issue)
        if plane not in (row.get("affected_observation_ids") or ()):
            continue
        if touches is not None and not touches & {kind.value for kind in finding_quantity_kinds(row["code"])}:
            continue
        listed = {a for a in _arrival_keys(row.get("affected_arrivals")) if a[0] == plane}
        reached = listed if light is None else listed & light
        listed_lines = {line for line in _line_keys(row.get("affected_sampled_lines")) if line[0] == plane}
        reached_lines = listed_lines if light is None else listed_lines & (lines or set())
        # (a finding of an earlier Engine lists no sampled line: it names sampled light by its planes only)
        legacy = "affected_sampled_lines" not in row and unlocalised and not listed
        if light is not None and not reached and not reached_lines and not legacy:
            continue
        stated = [ValidityCheck.model_validate(c) for c in row.get("validity") or ()]
        if light is not None and legacy and wavelength_nm is not None and any(c.parameter == WAVELENGTH_CHECK for c in stated):
            # (admitted only as light without listed arrivals: on a line, its wavelength checks of that line alone)
            stated = [c for c in stated if c.parameter != WAVELENGTH_CHECK or c.observed == wavelength_nm]
            if not any(c.parameter == WAVELENGTH_CHECK for c in stated):
                continue
        reach = row.get("validity_reach") or ()
        if (reached or reached_lines) and len(reach) == len(stated):
            stated = [c for c, met in zip(stated, reach, strict=True)
                      if not reached.isdisjoint(_arrival_keys(met.get("affected_arrivals")))
                      or not reached_lines.isdisjoint(_line_keys(met.get("affected_sampled_lines")))] or stated
        for check in stated or [ValidityCheck(parameter=row["code"].lower(), within=False, at=(row["instance_id"],), code=row["code"])]:
            if check not in checks:
                checks.append(check)
    return tuple(checks)


def _with_checks(entry: ValueBasis | None, checks: tuple[ValidityCheck, ...]) -> ValueBasis | None:
    """``entry`` with an ``ESTIMATE`` stating ``checks`` merged in: the number is the model's outside a model's validity."""

    if not checks:
        return entry
    return merge_bases(entry, ValueBasis(kind=ValueKind.ESTIMATE, validity=checks), monotone=None)


#: The kinds of quantity a Gaussian beam state's numbers are read as (its radii, waists, Rayleigh ranges, divergences,
#: curvature radii: ``GEOMETRY_ONLY_METRICS``).
GAUSSIAN_KINDS = frozenset({"GEOMETRY"})
_ARRIVAL_PREFIX = "monitor:arrivals[]."
#: Arrival fields stated in the lab frame that no metric reads by that name: the chief ray's position and direction (the
#: geometry ``position_u_mm`` / ``direction_cosine_u`` read in the plane's frame).
_LAB_FRAME_FIELDS = {"beam.position_mm": GAUSSIAN_KINDS, "beam.direction": GAUSSIAN_KINDS}
def arrival_field_kinds() -> dict[str, frozenset[str]]:
    """An arrival's numbers beside its power and its Gaussian state, by the path ``monitor()`` gives them (``uv_mm``,
    ``pupil_rms_radius_mm``, ``polarization_state.s1``, ``beam.optical_path_length_mm`` ...), each with the kinds of
    quantity the metrics that read it read (``RESULT_FIELDS``, ``metric_quantity_kinds``), and the chief ray's lab-frame
    position and direction (``GEOMETRY``)."""

    from ..validation.descriptors import RESULT_FIELDS
    from ..validation.quantity_kinds import metric_quantity_kinds

    kinds: dict[str, set[str]] = {}
    for metric, paths in RESULT_FIELDS.items():
        for path in paths:
            if not path.startswith(_ARRIVAL_PREFIX):
                continue
            field = path.removeprefix(_ARRIVAL_PREFIX).split("[", 1)[0]
            if field.startswith("beam.gaussian.") or field in ("beam.power_W", "detected_power_W"):
                continue
            kinds.setdefault(field, set()).update(kind.value for kind in metric_quantity_kinds(metric))
    return {**{field: frozenset(k) for field, k in sorted(kinds.items()) if k}, **_LAB_FRAME_FIELDS}


def _value_at(record: Any, field: str) -> Any:
    """The number (or vector of numbers) a dotted ``field`` names on ``record``, or ``None``."""

    value = record
    for name in field.split("."):
        value = getattr(value, name, None)
        if value is None:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, tuple) and value and all(isinstance(v, (int, float)) for v in value)):
        return value
    return None


def scoped_observation(o: ObservationResult, findings: Iterable[Any]) -> ObservationResult:
    """The plane ``o`` as the results report it under the run's model-scope findings ``findings`` (the Engine result itself
    is not changed): where a finding names the plane, its ``total_power_W`` -- which reads every beam on it -- carries an
    ``ESTIMATE`` with the checks of the light it reaches there (``scope_checks``), beside the entry derived from the
    plane's own facts; so do the power of each arrival the finding reaches (``beam.power_W``, ``detected_power_W``), its
    other numbers whose kind of quantity the finding touches (``arrival_field_kinds``; its Gaussian state under ``"*"``),
    the power of each line whose light it reaches (``power_by_wavelength``), the powers of each coherent sum of light it
    reaches (``superpositions``), the detector's quantities that read light it reaches (``detection``,
    ``_scoped_detection``), the Stokes values of each polarization mixture of light it reaches (``_scoped_mixture``) and
    the power and collection fractions of each sampled line it lists (``sampled_light``). An arrival, a line, a sum, a
    mixture or a detector reading it does not reach keeps its entry. A number that is null gets none."""

    named = [row for row in map(_finding_row, findings or ()) if o.observation_id in (row.get("affected_observation_ids") or ())]
    if not named:
        return o
    plane = o.observation_id
    update: dict[str, Any] = {}
    checks = scope_checks(named, plane)
    if o.total_power_W is not None and checks:
        update["value_basis"] = {**(o.value_basis or {}), "total_power_W": _with_checks(basis_of(o, "total_power_W"), checks)}
    arrivals = []
    field_kinds = arrival_field_kinds()
    for a in o.arrivals:
        light = {(plane, a.beam.path_id, a.crossing_index)}
        checks = scope_checks(named, plane, light)
        bases = {field: _with_checks(basis_of(a, field), checks)
                 for field, value in (("beam.power_W", a.beam.power_W), ("detected_power_W", a.detected_power_W))
                 if checks and value is not None}
        changes: dict[str, Any] = {}
        if checks:
            # (its other numbers: each takes the findings that touch a kind of quantity a metric reading it reads)
            for field, kinds in field_kinds.items():
                value = _value_at(a, field)
                if value is not None and (field_checks := scope_checks(named, plane, light, touches=kinds)):
                    bases[field] = _with_checks(basis_of(a, field), field_checks)
            gaussian = a.beam.gaussian
            geometry = scope_checks(named, plane, light, touches=GAUSSIAN_KINDS) if gaussian is not None and gaussian.valid else ()
            if geometry:
                own = gaussian.value_basis or {}
                scoped = {key: _with_checks(entry, geometry) for key, entry in own.items()}
                scoped[WHOLE_RECORD] = _with_checks(own.get(WHOLE_RECORD), geometry)
                changes["beam"] = a.beam.model_copy(update={"gaussian": gaussian.model_copy(update={"value_basis": scoped})})
        if bases:
            changes["value_basis"] = {**(a.value_basis or {}), **bases}
        arrivals.append(a.model_copy(update=changes) if changes else a)
    lines = []
    for s in o.power_by_wavelength:
        light = {(plane, a.beam.path_id, a.crossing_index) for a in o.arrivals if a.beam.wavelength_nm == s.wavelength_nm}
        # (the line's sampled light: the sampled lines of its wavelength on the plane)
        sampled = plane_lines(o, s.wavelength_nm)
        checks = scope_checks(named, plane, light, lines=sampled, unlocalised=bool(sampled), wavelength_nm=s.wavelength_nm)
        lines.append(s.model_copy(update={"value_basis": {**(s.value_basis or {}), "power_W": _with_checks(basis_of(s, "power_W"), checks)}})
                     if checks and s.power_W is not None else s)
    sampled_rows = []
    for line in o.sampled_light:
        checks = scope_checks(named, plane, set(), lines={(plane, line.source_id, line.spectral_index)}, unlocalised=True,
                              wavelength_nm=line.wavelength_nm)
        bases = {field: _with_checks(basis_of(line, field), checks) for field in SAMPLED_LINE_FIELDS if checks}
        sampled_rows.append(line.model_copy(update={"value_basis": {**(line.value_basis or {}), **bases}}) if bases else line)
    update["sampled_light"] = tuple(sampled_rows)
    if o.detection is not None:
        update["detection"] = _scoped_detection(o, named)
    if o.polarization_mixtures:
        update["polarization_mixtures"] = tuple(_scoped_mixture(m, plane, named) for m in o.polarization_mixtures)
    sums = []
    for s in o.superpositions:
        # (a coherent sum of Gaussian beams: the findings that reach one of its beams; sampled light is in no coherent sum)
        light = {(plane, path_id, index) for path_id, index in zip(s.path_ids, s.crossing_indices or (0,) * len(s.path_ids))}
        checks = scope_checks(named, plane, light)
        bases = {field: _with_checks(basis_of(s, field), checks) for field in SUPERPOSITION_FIELDS
                 if checks and getattr(s, field) is not None}
        sums.append(s.model_copy(update={"value_basis": {**(s.value_basis or {}), **bases}}) if bases else s)
    return o.model_copy(update={**update, "arrivals": tuple(arrivals), "power_by_wavelength": tuple(lines),
                                "superpositions": tuple(sums)})


def _scoped_detection(o: ObservationResult, named: list[Mapping[str, Any]]) -> Any:
    """The plane's detection under the findings ``named`` that name it: each quantity that reads the plane's light
    (``LIGHT_QUANTITIES``) and has a value carries an ``ESTIMATE`` with the checks of the light the detector reads (its
    contributions' sources and wavelengths: those arrivals, and the plane's sampled light where a sampled line is among
    them; every arrival and line read: the plane total's checks), merged into ``value_basis["value"]``. A detector that reads
    only light no finding reaches stays unmarked; the bandwidths and the dark noise are not marked. ``power_state`` stays
    the plane's (it says what the Engine computed)."""

    from ..validation.detection import LIGHT_QUANTITIES

    detection, plane = o.detection, o.observation_id
    read = {(c.source_id, c.wavelength_nm) for c in detection.contributions}
    light = {(plane, a.beam.path_id, a.crossing_index) for a in o.arrivals if (a.beam.source_id, a.beam.wavelength_nm) in read}
    sampled = {(plane, line.source_id, line.spectral_index) for line in o.sampled_light if (line.source_id, line.wavelength_nm) in read}
    if len(light) == len(o.arrivals) and len(sampled) == len(o.sampled_light):
        checks = scope_checks(named, plane)
    else:
        checks = scope_checks(named, plane, light, lines=sampled, unlocalised=bool(sampled))
    if not checks:
        return detection
    quantities = tuple(
        q.model_copy(update={"value_basis": {**(q.value_basis or {}), "value": _with_checks(basis_of(q, "value"), checks)}})
        if q.name in LIGHT_QUANTITIES and q.value is not None else q for q in detection.quantities)
    return detection.model_copy(update={"quantities": quantities})


#: The numbers of a polarization mixture, each with the metrics that read it (``POLARIZATION_METRICS``, ``power_W``).
MIXTURE_FIELDS = {"stokes_W": ("power_W", "stokes_s1", "stokes_s2", "stokes_s3"),
                  "degree_of_polarization": ("degree_of_polarization",),
                  "polarization_state.s1": ("stokes_s1",), "polarization_state.s2": ("stokes_s2",),
                  "polarization_state.s3": ("stokes_s3",), "polarization_state.ellipticity_deg": ("polarization_ellipticity_deg",),
                  "polarization_state.azimuth_in_plane_deg": ("polarization_azimuth_deg",),
                  "polarization_state.degree_of_polarization": ("degree_of_polarization",)}


def _scoped_mixture(mix: Any, plane: str, named: list[Mapping[str, Any]]) -> Any:
    """A polarization mixture under the findings ``named``: each of its numbers carries an ``ESTIMATE`` with the checks of
    the findings that reach one of its components on its route (``path_ids`` at ``crossing_index``) and touch a kind of
    quantity a metric reading that number reads (``MIXTURE_FIELDS``). Its ``quantity_state`` stays: the values are the
    model's numbers outside a model's validity, not withheld."""

    from ..validation.quantity_kinds import metric_quantity_kinds

    light = {(plane, path_id, mix.crossing_index) for path_id in mix.path_ids}
    bases = {}
    for field, metrics in MIXTURE_FIELDS.items():
        if _value_at(mix, field) is None:
            continue
        kinds = frozenset(kind.value for metric in metrics for kind in metric_quantity_kinds(metric))
        checks = scope_checks(named, plane, light, touches=kinds)
        if checks:
            bases[field] = _with_checks(basis_of(mix, field), checks)
    return mix.model_copy(update={"value_basis": {**(mix.value_basis or {}), **bases}}) if bases else mix


#: The numbers of a sampled line at a plane that read the light carried there: a model-scope finding that reaches the line
#: qualifies them (its ``emitted_W`` is the source's own emission).
SAMPLED_LINE_FIELDS = ("power_W", "collection_fraction", "collection_fraction_half_samples", "collection_fraction_difference",
                       "collection_fraction_sigma")


#: The numbers of a coherent sum a model-scope finding on one of its beams qualifies (``coherent_power_W`` reads them).
SUPERPOSITION_FIELDS = ("total_power_W", "total_power_min_W", "total_power_max_W", "incoherent_power_W", "interference_power_W")


def scoped_interference(row: InterferenceResult, findings: Iterable[Any]) -> InterferenceResult:
    """A two-beam interference record as the results report it under the run's model-scope findings ``findings``: where a
    finding reaches one of its two beams on its plane, each of its numbers a pair metric of the same name reads
    (``PAIR_FIELDS``) carries an ``ESTIMATE`` with the checks of the findings that touch a kind of quantity that metric
    reads, as a target on that metric does."""

    from ..validation.descriptors import PAIR_FIELDS
    from ..validation.quantity_kinds import metric_quantity_kinds

    plane = row.observation_id
    named = [r for r in map(_finding_row, findings or ()) if plane in (r.get("affected_observation_ids") or ())]
    if not named:
        return row
    light = {(plane, row.path_id_a, row.crossing_index_a), (plane, row.path_id_b, row.crossing_index_b)}
    bases = {}
    for field in PAIR_FIELDS:
        if _value_at(row, field) is None:
            continue
        checks = scope_checks(named, plane, light, touches=frozenset(kind.value for kind in metric_quantity_kinds(field)))
        if checks:
            bases[field] = _with_checks(basis_of(row, field), checks)
    return row.model_copy(update={"value_basis": {**(row.value_basis or {}), **bases}}) if bases else row


# ----------------------------------------------------------------------------------------------- carriage


@dataclass(frozen=True)
class Projection:
    """A projection of results that carries quantities: where it is, the records it reads and how it carries their basis.
    ``CARRIES``: under ``value_basis``, keyed by its own name of each number. ``BARE``: by decision, without the entry."""

    name: str
    carriage: Literal["CARRIES", "BARE"]
    records: tuple[type, ...]


#: Every projection that carries a quantity of a result (the carriage test runs a result with qualified values through
#: each). A new projection of a quantity is added here.
CARRIAGE: tuple[Projection, ...] = (
    Projection("RunResults.targets", "CARRIES", (TargetResult,)),
    Projection("RunResults.monitors", "CARRIES", (ObservationResult, ArrivalResult, SampledLightAtPlane)),
    Projection("RunResults.monitor", "CARRIES", (ObservationResult, ArrivalResult, SpectralPower, CoherentSuperposition,
                                                 InterferenceResult, TargetResult, GaussianBeamState, SampledLightAtPlane,
                                                 PolarizationMixture)),
    Projection("RunResults.superpositions", "CARRIES", (CoherentSuperposition,)),
    Projection("RunResults.exposures", "CARRIES", (ExposureHit, RatingComparison)),
    Projection("RunResults.detection", "CARRIES", (DetectionQuantity,)),
    Projection("RunResults.sampled_light", "CARRIES", (SampledLightAtPlane,)),
    Projection("tables.sampled_light", "CARRIES", (SampledLightAtPlane,)),
    Projection("tables.sampled_crossings", "CARRIES", (SampledLightAtPlane,)),
    Projection("tables.monitors", "CARRIES", (ObservationResult,)),
    Projection("tables.arrivals", "CARRIES", (ArrivalResult, GaussianBeamState)),
    Projection("tables.superpositions", "CARRIES", (CoherentSuperposition,)),
    Projection("LocalOpticalRun.summary", "CARRIES", (TargetResult,)),
    Projection("json_api.run_summary", "CARRIES", (TargetResult, ObservationResult, ArrivalResult, GaussianBeamState,
                                                   SampledLightAtPlane)),
    Projection("json_api.evaluate_project", "CARRIES", (TargetResult, ObservationResult, ArrivalResult, GaussianBeamState,
                                                       SampledLightAtPlane)),
    Projection("json_api.workspace_run", "CARRIES", (TargetResult,)),
    Projection("cli_workspace.run", "CARRIES", (TargetResult,)),
    Projection("local.record.scientific_projection", "BARE", (TargetResult, ObservationResult)),
)



__all__ = [
    "CARRIAGE",
    "DERIVATIONS",
    "Projection",
    "bases_of",
    "basis_of",
    "dump_with_bases",
    "row_basis_cell",
    "row_bases",
    "scope_checks",
    "scoped_observation",
    "stored_basis",
]
