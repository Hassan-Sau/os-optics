"""The value-basis contract: types and validators, the carriers, the reader and its derivation, carriage, rule T1, notices.

A number is the model's value only when no ``value_basis`` entry names it or the entry's kind is ``MODEL``; an entry feeds
nothing but the reader. These tests hold the contract itself; the producers that fill entries test their own values.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from os_optics.contracts.analysis import Interval, IntervalKind, Quantity, ValidityCheck as AnalysisValidityCheck
from os_optics.contracts.basis import (
    ACCOUNTING_PARTS,
    Assumed,
    Omitted,
    Requirement,
    ValidityCheck,
    ValueBasis,
    grade_point,
    merge_bases,
)
from os_optics.contracts.optics import GaussianBeamState
from os_optics.contracts.project import Target
from os_optics.contracts.results import (
    ArrivalResult,
    CoherentSuperposition,
    ExposureHit,
    InterferenceResult,
    ObservationResult,
    PolarizationMixture,
    SampledLightAtPlane,
    SpectralPower,
    TargetResult,
)
from os_optics.errors import ErrorCode, Status, notice, worst
from os_optics.errors.status import AssumedInput, OmittedPart, QuantityState, RequirementAssessment, ValueKind
from os_optics.results import CARRIAGE, DERIVATIONS, basis_of
from os_optics.results import quantity as dict_quantity
from os_optics.results.basis import bases_of, scoped_observation

SNAPSHOTS = Path(__file__).parent / "fixtures" / "snapshots"
LOWER = ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.NOT_FOLLOWED, upper=1e-6),))
ESTIMATE = ValueBasis(kind=ValueKind.ESTIMATE, validity=(ValidityCheck(parameter="far_field_half_angle_rad", maximum=0.3,
                                                                       observed=0.42, within=False,
                                                                       setting="gaussian_scope_divergence_rad"),))


def _snapshots() -> list[tuple[str, dict]]:
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(SNAPSHOTS.glob("*.json"))]


# ------------------------------------------------------------------------------------------------ types (B2)


def test_an_entry_states_something():
    with pytest.raises(ValidationError, match="states at least one"):
        ValueBasis(kind=ValueKind.LOWER_BOUND)


def test_a_model_entry_holds_only_passing_checks_and_accounting_omissions():
    ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(parameter="p", maximum=0.5, observed=0.42, within=True, setting="s"),))
    ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, setting="negligible_power_fraction"),))
    with pytest.raises(ValidationError, match="within their range"):
        ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(parameter="p", within=False),))
    with pytest.raises(ValidationError, match="within their range"):
        ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(parameter="p", within=None),))
    with pytest.raises(ValidationError, match="accounting parts"):
        ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(part=OmittedPart.NOT_FOLLOWED),))
    with pytest.raises(ValidationError, match="no assumption"):
        ValueBasis(kind=ValueKind.MODEL, assumed=(Assumed(input=AssumedInput.FULL_COHERENCE),))


def test_the_accounting_parts_are_the_levels_and_every_other_part_is_an_uncertainty():
    assert ACCOUNTING_PARTS == {OmittedPart.BELOW_NEGLIGIBLE_LEVEL, OmittedPart.BELOW_POWER_FLOOR}
    assert Omitted(part=OmittedPart.BELOW_POWER_FLOOR).accounting
    assert not any(Omitted(part=part).accounting for part in OmittedPart if part not in ACCOUNTING_PARTS)


def test_the_entries_have_no_prose_field_and_omit_what_they_do_not_state():
    for model in (ValueBasis, Omitted, Assumed, ValidityCheck):
        assert not {"message", "reason", "note", "description", "statement"} & set(model.model_fields)
    assert json.loads(LOWER.model_dump_json()) == {"kind": "LOWER_BOUND", "omitted": [{"part": "NOT_FOLLOWED", "upper": 1e-6}]}
    # an upper bound of a part is never negative
    with pytest.raises(ValidationError):
        Omitted(part=OmittedPart.NOT_FOLLOWED, upper=-1.0)


def test_validity_check_moved_and_keeps_its_form():
    assert AnalysisValidityCheck is ValidityCheck
    assert json.loads(ValidityCheck(parameter="x", minimum=1.0, observed=2.0, within=True).model_dump_json()) == {
        "parameter": "x", "unit": None, "minimum": 1.0, "maximum": None, "observed": 2.0, "within": True}
    assert ValidityCheck(parameter="x", setting="s", at=("L1/front",), code="GAUSSIAN_HIGH_NA").model_dump(mode="json")["at"] == ["L1/front"]


def test_every_member_of_the_two_enums_is_defined_in_one_sentence_in_the_code():
    source = (Path(__file__).parents[1] / "src" / "os_optics" / "errors" / "status.py").read_text(encoding="utf-8")
    for enum in (OmittedPart, AssumedInput, ValueKind):
        for member in enum:
            line = f'    {member.name} = "{member.value}"'
            assert line in source
            before = source[:source.index(line)].rstrip().splitlines()[-1].strip()
            assert before.startswith("#:"), member


# ------------------------------------------------------------------------------------------ carriers (B3)


def _target(**update) -> TargetResult:
    return TargetResult(**{"target_id": "t", "metric": "power_W", "status": "UNKNOWN", "value": 1e-3, **update})


def test_a_carrier_key_names_a_value_field_that_holds_a_number():
    assert _target(value_basis={"value": LOWER}).value_basis == {"value": LOWER}
    with pytest.raises(ValidationError, match="names no field"):
        _target(value_basis={"valu": LOWER})
    with pytest.raises(ValidationError, match="without a value"):
        _target(value=None, value_basis={"value": LOWER})
    with pytest.raises(ValidationError, match="holds no number"):
        _target(value_basis={"metric": LOWER})


def test_an_empty_entry_map_is_no_entry_and_is_omitted():
    assert _target(value_basis={}).value_basis is None
    assert "value_basis" not in _target(value_basis={}).model_dump(mode="json")


def test_the_whole_record_key_and_a_dotted_key():
    g = GaussianBeamState(valid=False, status="OUT_OF_MODEL_SCOPE", reason="GAUSSIAN_HIGH_NA", beam_radius_x_mm=1.52,
                          value_basis={"*": ESTIMATE})
    assert basis_of(g, "beam_radius_x_mm") == ESTIMATE and basis_of(g, "rayleigh_range_x_mm") == ESTIMATE
    assert json.loads(g.model_dump_json())["value_basis"]["*"]["kind"] == "ESTIMATE"


def _plane(state: str, value: float | None = 1e-3, **update) -> ObservationResult:
    return ObservationResult(**{"observation_id": "mon", "status": "PASS", "total_power_W": value, "quantity_state": state,
                                "power_complete": state in ("KNOWN", "KNOWN_ZERO"), **update})


def test_the_state_word_governs_the_entries_of_its_quantity():
    _plane("UNKNOWN", value_basis={"total_power_W": LOWER})
    for state in ("KNOWN_ZERO", "NOT_APPLICABLE"):
        with pytest.raises(ValidationError, match="carries no value basis"):
            _plane(state, 0.0, value_basis={"total_power_W": LOWER})
    with pytest.raises(ValidationError, match="only a MODEL"):
        _plane("KNOWN", value_basis={"total_power_W": LOWER})
    model = ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(parameter="p", within=True, setting="s"),))
    assert _plane("KNOWN", value_basis={"total_power_W": model}).value_basis == {"total_power_W": model}


@pytest.mark.parametrize("carrier", [ArrivalResult, SpectralPower, CoherentSuperposition, InterferenceResult, ExposureHit,
                                     TargetResult, ObservationResult, GaussianBeamState])
def test_every_carrier_has_the_field_and_omits_it_when_absent(carrier):
    assert "value_basis" in carrier.model_fields


def test_a_result_without_entries_keeps_its_bytes():
    """Every stored record of the fixture snapshots (written before the contract existed) reads and writes back unchanged."""

    checked = 0
    for name, doc in _snapshots():
        for o in doc.get("observations", ()):
            assert ObservationResult.model_validate(o).model_dump(mode="json") == o, name
            checked += 1
        for t in doc.get("targets", ()):
            assert TargetResult.model_validate(t).model_dump(mode="json") == t, name
            checked += 1
        for i in doc.get("interference", ()):
            assert InterferenceResult.model_validate(i).model_dump(mode="json") == i, name
            checked += 1
        exposures = doc.get("exposures") or {}
        for s in exposures.get("surfaces", ()):
            for hit in s.get("hits", ()):
                assert ExposureHit.model_validate(hit).model_dump(mode="json") == hit, name
                checked += 1
    assert checked > 100


# ------------------------------------------------------------------------------ the analysis Quantity (B3)


def test_an_analysis_quantity_carries_a_value_off_the_valued_states_exactly_with_its_entry():
    base = {"quantity_id": "w", "name": "beam radius", "unit": "mm", "reason": "GAUSSIAN_HIGH_NA"}
    with pytest.raises(ValidationError, match="without a value_basis"):
        Quantity(**base, state="OUT_OF_MODEL_SCOPE", value=1.52)
    q = Quantity(**base, state="OUT_OF_MODEL_SCOPE", value=1.52, value_basis=ESTIMATE)
    assert q.model_dump(mode="json")["value_basis"]["kind"] == "ESTIMATE"
    with pytest.raises(ValidationError, match="but no value"):
        Quantity(**base, state="OUT_OF_MODEL_SCOPE", value_basis=ESTIMATE)
    with pytest.raises(ValidationError, match="only a MODEL"):
        Quantity(quantity_id="p", name="p", state="KNOWN", value=1.0, value_basis=LOWER)
    plain = Quantity(quantity_id="p", name="p", state="KNOWN", value=1.0)
    assert "value_basis" not in plain.model_dump(mode="json")


def test_an_analysis_quantity_decides_in_the_direction_its_kind_proves():
    finesse = Quantity(quantity_id="f", name="finesse", state="BOUNDED", value=100.0, reason="LOSS_NOT_STATED",
                       interval=Interval(upper=100.0, kind=IntervalKind.NOMINAL),
                       value_basis=ValueBasis(kind=ValueKind.UPPER_BOUND, assumed=(Assumed(input=AssumedInput.NO_UNSTATED_LOSS),)))
    assert finesse.decides(150.0) and not finesse.decides(50.0)
    lower = Quantity(quantity_id="p", name="p", state="UNKNOWN", value=1.0, reason="R", value_basis=LOWER)
    assert lower.decides(0.5) and not lower.decides(2.0)
    estimate = Quantity(quantity_id="w", name="w", state="OUT_OF_MODEL_SCOPE", value=1.0, reason="R", value_basis=ESTIMATE)
    assert not estimate.decides(0.0) and not estimate.decides(10.0)


def test_a_dict_quantity_carries_one_entry_and_an_unavailable_one_a_value_only_with_it():
    assert dict_quantity.modeled("W", "b", value=1.0, value_basis=LOWER)["value_basis"]["kind"] == "LOWER_BOUND"
    assert "value_basis" not in dict_quantity.modeled("W", "b", value=1.0)
    with pytest.raises(ValueError):
        dict_quantity.unavailable("mm", "R", "m", value=1.0)
    doc = dict_quantity.unavailable("mm", "R", "m", value=1.0, value_basis=ESTIMATE)
    assert doc["value"] == 1.0 and doc["value_basis"]["kind"] == "ESTIMATE"


# ---------------------------------------------------------------------------------------- derivation (B5)

_DERIVED_KINDS = {
    QuantityState.BOUNDED: {ValueKind.LOWER_BOUND, ValueKind.UPPER_BOUND, ValueKind.ESTIMATE},
    QuantityState.UNKNOWN: {ValueKind.LOWER_BOUND, ValueKind.ESTIMATE},
    QuantityState.OUT_OF_MODEL_SCOPE: {ValueKind.ESTIMATE},
    QuantityState.NOT_COMPUTED: {ValueKind.LOWER_BOUND},
}


def test_the_derived_entries_agree_with_the_state_words_on_every_result_of_the_fixture_scenes():
    seen: dict[str, int] = {}
    for name, doc in _snapshots():
        for raw in doc.get("observations", ()):
            o = ObservationResult.model_validate(raw)
            entry = basis_of(o, "total_power_W")
            state = o.quantity_state
            if o.total_power_W is None or state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO, QuantityState.NOT_APPLICABLE):
                assert entry is None, (name, o.observation_id)
            else:
                assert entry is not None and entry.kind in _DERIVED_KINDS[state], (name, o.observation_id, state, entry)
                # the same entry stored on the record is accepted by the carrier rules
                ObservationResult.model_validate({**raw, "value_basis": {"total_power_W": entry.model_dump(mode="json")}})
                if o.power_upper_bound_W is not None and entry.kind is ValueKind.LOWER_BOUND:
                    assert sum(part.upper for part in entry.omitted) == pytest.approx(o.power_upper_bound_W - o.total_power_W)
                seen[state.value] = seen.get(state.value, 0) + 1
            for s in o.power_by_wavelength:
                derived = basis_of(s, "power_W")
                assert (derived is None) == (s.power_W is None or (s.complete and not s.power_out_of_scope_at)), (name, s)
            for a in o.arrivals:
                for field in ("beam.power_W", "detected_power_W"):
                    derived = basis_of(a, field)
                    if derived is not None:
                        ArrivalResult.model_validate({**a.model_dump(mode="json"), "value_basis": {field: derived.model_dump(mode="json")}})
            if state is QuantityState.KNOWN:
                assert all(basis_of(a, "beam.power_W") is None for a in o.arrivals), (name, o.observation_id)
    # (the fixture scenes' lit planes whose power light not followed may add are BOUNDED, with an open upper end where no
    # bound is supported; a plane UNKNOWN with a number -- a traced 0 W with an open upper end -- occurs in none of them, and
    # its derivation is pinned on stated records by test_the_derivation_of_each_flat_fact)
    assert seen.get("BOUNDED", 0) >= 10, seen


def test_the_derivation_of_each_flat_fact():
    unknown = _plane("UNKNOWN", untraced_power_W=2e-6)
    assert basis_of(unknown, "total_power_W") == ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.NOT_FOLLOWED),))
    returning = _plane("UNKNOWN", untraced_power_W=0.0)
    assert basis_of(returning, "total_power_W").omitted[0].part is OmittedPart.POSSIBLE_RETURN
    # (a plane only light left out as negligible leaves incomplete: its returns or crossings may bring more than the
    # negligible level there -- an uncertainty, not the accounting omission a KNOWN plane carries)
    faint = _plane("BOUNDED", untraced_power_W=0.0, power_upper_bound_W=1.5e-3)
    entry = basis_of(faint, "total_power_W")
    assert entry.kind is ValueKind.LOWER_BOUND and entry.omitted[0].part is OmittedPart.POSSIBLE_RETURN
    assert entry.omitted[0].upper == pytest.approx(5e-4)
    narrowed = _plane("NOT_COMPUTED", untraced_power_W=None)
    assert basis_of(narrowed, "total_power_W").omitted[0].part is OmittedPart.NOT_ACCOUNTED
    taken = _plane("BOUNDED", power_lower_bound_W=0.8e-3, power_complete=True)
    entry = basis_of(taken, "total_power_W")
    assert entry.kind is ValueKind.UPPER_BOUND and entry.omitted[0].upper == pytest.approx(0.2e-3)
    assert basis_of(_plane("KNOWN"), "total_power_W") is None
    assert basis_of(_plane("UNKNOWN", None), "total_power_W") is None
    # a stored entry wins over the derived one
    stored = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.GAUSSIAN_PROFILE),))
    assert basis_of(_plane("UNKNOWN", untraced_power_W=2e-6, value_basis={"total_power_W": stored}), "total_power_W") == stored
    assert set(bases_of(unknown)) == {"total_power_W"}
    ranged = CoherentSuperposition(wavelength_nm=1064.0, path_ids=("a", "b"), incoherent_power_W=2.0, total_power_W=3.0,
                                   total_power_min_W=2.5, total_power_max_W=3.5, status="PASS")
    assert basis_of(ranged, "total_power_W").assumed[0].input is AssumedInput.UNSTATED_GROUP_DELAY_ZERO
    assert {field for _type, field in DERIVATIONS} >= {"total_power_W", "power_W", "beam.power_W", "detected_power_W", "coherence_factor"}


# --------------------------------------------------------------------------------------------- merge (B6)


def test_merge_bases_unions_the_lists_and_keeps_a_kind_only_where_it_is_proven():
    upper = ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=("L1/front",)),))
    assert merge_bases(None, None) is None
    assert merge_bases(LOWER, None).kind is ValueKind.LOWER_BOUND
    assert merge_bases(LOWER, LOWER).omitted == LOWER.omitted
    assert merge_bases(LOWER, None, monotone="decreasing").kind is ValueKind.UPPER_BOUND
    assert merge_bases(LOWER, upper).kind is ValueKind.ESTIMATE
    assert merge_bases(LOWER, None, monotone=None).kind is ValueKind.ESTIMATE
    assert merge_bases(LOWER, ESTIMATE).kind is ValueKind.ESTIMATE
    merged = merge_bases(LOWER, upper, ESTIMATE)
    assert merged.omitted == LOWER.omitted + upper.omitted and merged.validity == ESTIMATE.validity
    model = ValueBasis(kind=ValueKind.MODEL, validity=(ValidityCheck(parameter="p", within=True, setting="s"),))
    assert merge_bases(model, None).kind is ValueKind.MODEL
    assert merge_bases(model, LOWER).kind is ValueKind.LOWER_BOUND


# ------------------------------------------------------------------------------------------------- T1 (B7)

MINIMUM = Requirement(minimum=1.0)
MAXIMUM = Requirement(maximum=1.0)
WINDOW = Requirement(minimum=0.5, maximum=1.5)


@pytest.mark.parametrize(("value", "requirement", "basis", "measured", "expected"), [
    # no entry or MODEL: graded as before
    (2.0, MINIMUM, None, Status.PASS, Status.PASS),
    (0.5, MINIMUM, None, Status.PASS, Status.FAIL),
    (2.0, MINIMUM, None, Status.WARNING, Status.WARNING),
    (2.0, MINIMUM, None, Status.NOT_APPLICABLE, Status.PASS),
    # a lower bound: a minimum it meets is met with a caveat, a maximum it exceeds is unmet, anything else undecided
    (2.0, MINIMUM, LOWER, Status.PASS, Status.WARNING),
    (0.5, MINIMUM, LOWER, Status.PASS, Status.UNKNOWN),
    (2.0, MAXIMUM, LOWER, Status.PASS, Status.FAIL),
    (0.5, MAXIMUM, LOWER, Status.PASS, Status.UNKNOWN),
    (1.0, WINDOW, LOWER, Status.PASS, Status.UNKNOWN),
    # an upper bound: the mirror image
    (0.5, MAXIMUM, ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE),)), Status.PASS, Status.WARNING),
    (0.5, MINIMUM, ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE),)), Status.PASS, Status.FAIL),
    # an estimate decides nothing: out of model scope when a validity check failed, else unknown
    (2.0, MINIMUM, ESTIMATE, Status.PASS, Status.OUT_OF_MODEL_SCOPE),
    (0.5, MINIMUM, ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.GAUSSIAN_PROFILE),)), Status.PASS, Status.UNKNOWN),
    # a number the measurement does not call known decides nothing, also where it misses (decision B7: never FAIL)
    (0.5, MINIMUM, None, Status.UNKNOWN, Status.UNKNOWN),
    (0.5, MINIMUM, None, Status.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE),
    (2.0, MINIMUM, None, Status.UNKNOWN, Status.UNKNOWN),
])
def test_rule_t1(value, requirement, basis, measured, expected):
    assert grade_point(value, requirement, basis=basis, measured=measured) is expected


def test_the_requirement_reads_a_target_value_with_its_tolerance():
    r = Requirement(target_value=1.0, tolerance=0.1)
    assert r.meets(1.05) and r.below(0.8) and r.above(1.2) and not r.open_above and not r.open_below
    assert grade_point(1.05, r, basis=LOWER) is Status.UNKNOWN  # a lower bound inside the window: more may arrive
    assert grade_point(1.2, r, basis=LOWER) is Status.FAIL


def _evaluate(monkeypatch, mv):
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics

    monkeypatch.setattr(metrics, "_measure", lambda *args, **kwargs: mv)
    target = Target(target_id="w", metric="beam_radius_mm", observation_id="mon", minimum=3.0)
    return metrics._evaluate(target, {}, [], None, None, NumericalConfig())


def test_a_point_value_the_measurement_calls_unknown_and_that_misses_its_bound_is_indeterminate(monkeypatch):
    """Decision B7: before it, a missed bound outranked the measurement's UNKNOWN and the target read FAIL / UNMET."""

    from os_optics.validation.metrics import MetricValue

    result = _evaluate(monkeypatch, MetricValue(2.0, "mm", Status.UNKNOWN, ("GAUSSIAN_HIGH_NA",), "the beam radius is not known here"))
    assert result.status is Status.UNKNOWN and result.assessment is RequirementAssessment.INDETERMINATE
    assert result.value == 2.0 and result.issue_codes == ("GAUSSIAN_HIGH_NA",)
    assert result.message == "the beam radius is not known here; value 2 mm is below the minimum 3 mm"
    known = _evaluate(monkeypatch, MetricValue(2.0, "mm", Status.PASS))
    assert known.status is Status.FAIL and known.assessment is RequirementAssessment.UNMET
    assert ErrorCode.TARGET_NOT_MET.value in known.issue_codes


def test_a_measured_entry_reaches_the_target_under_value_and_grades_it(monkeypatch):
    from os_optics.validation.metrics import MetricValue

    result = _evaluate(monkeypatch, MetricValue(4.0, "mm", Status.PASS, basis=LOWER))
    assert result.value_basis == {"value": LOWER}
    assert result.status is Status.WARNING and result.assessment is RequirementAssessment.MET
    missed = _evaluate(monkeypatch, MetricValue(2.0, "mm", Status.PASS, basis=ESTIMATE))
    assert missed.status is Status.OUT_OF_MODEL_SCOPE and missed.error is None and missed.value == 2.0
    assert missed.value_basis == {"value": ESTIMATE}


# ------------------------------------------------------------------------------------------- notices (B6)


def test_a_notice_counts_toward_no_status_and_sits_at_its_quantity():
    issue = notice(ErrorCode.GAUSSIAN_HIGH_NA, "the far-field half angle exceeds the first-order limit", location="observation:mon",
                   observed=0.42, limit=0.3)
    assert issue.status is Status.NOT_APPLICABLE and issue.location == "observation:mon" and issue.affected_targets == ()
    assert issue.numerical_context == {"observed": 0.42, "limit": 0.3}
    assert worst([Status.PASS, issue.status]) is Status.PASS
    with pytest.raises(ValueError):
        notice(ErrorCode.GAUSSIAN_HIGH_NA, "m", location="")


# ------------------------------------------------------------------------------------------- carriage (B8)

SENTINEL = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.GAUSSIAN_PROFILE, value=0.25),))
SENTINEL_JSON = SENTINEL.model_dump(mode="json")


@pytest.fixture
def qualified(monkeypatch):
    """Every target value and every plane and arrival power of a run carries ``SENTINEL``: the targets as stored entries
    (through the measurement), the powers as derived ones."""

    from os_optics.results import basis as reader
    from os_optics.validation import metrics

    measure = metrics._measure

    def measured(*args, **kwargs):
        mv = measure(*args, **kwargs)
        return mv if mv.value is None else dataclasses.replace(mv, basis=SENTINEL)

    monkeypatch.setattr(metrics, "_measure", measured)
    derivations = dict(reader.DERIVATIONS)
    derivations[(ObservationResult, "total_power_W")] = lambda o: None if o.total_power_W is None else SENTINEL
    derivations[(ArrivalResult, "beam.power_W")] = lambda a: None if a.beam.power_W is None else SENTINEL
    derivations[(SampledLightAtPlane, "power_W")] = lambda line: SENTINEL
    derivations[(PolarizationMixture, "degree_of_polarization")] = lambda m: None if m.degree_of_polarization is None else SENTINEL
    monkeypatch.setattr(reader, "DERIVATIONS", derivations)


def _carries(row: dict, key: str) -> bool:
    return row.get(key) is None or row.get("value_basis", {}).get(key) == SENTINEL_JSON


def _sampled_results():
    """A run whose plane reads an extended emitter's light (``test_sampled_sources``)."""

    from test_sampled_sources import _component, _intensity

    from os_optics.local import EvaluationSession
    from test_model_scope_per_beam import OUTPUTS, lens

    document = lens(532)
    emitter = _component(_intensity([[1.0]], 0.1, rays=16, wavelength_nm=532), "EM")
    emitter["nominal_pose"]["position_mm"][0] = -100
    document["components"].append(emitter)
    return EvaluationSession().evaluate(document, outputs=OUTPUTS).results()


def _mixture_results():
    """A run whose plane reads a partially polarized source's light (its ``polarization_mixtures``)."""

    from os_optics.local import EvaluationSession
    from test_model_scope_per_beam import OUTPUTS, lens

    document = lens(532)
    document["components"][0]["source_model"]["polarization"] = {
        "kind": "PARTIAL", "degree_of_polarization": 0.5, "polarized": {"kind": "LINEAR", "angle_deg": 0.0}}
    document["targets"] = [{"target_id": "dop", "metric": "degree_of_polarization", "observation_id": "early", "minimum": 0.0,
                            "unit": "1"}]
    return EvaluationSession().evaluate(document, outputs=OUTPUTS).results()


def _cell(row: dict) -> dict:
    return {**row, "value_basis": row["value_basis"] or {}}


def test_every_projection_that_carries_a_quantity_carries_its_entry(qualified, tmp_path):
    from os_optics.local import json_api, load_project, run_optical_project
    from os_optics.local.artifacts import run_artifact_files
    from os_optics.local.record import scientific_projection
    from os_optics.results import results_from_files

    from local_helpers import plane, project, source

    document = load_project(project(
        [source()], [plane("screen", (100.0, 0.0, 0.0))],
        [{"target_id": "p", "metric": "power_W", "observation_id": "screen", "target_value": 0.001,
          "tolerance_abs": 1e-6, "unit": "W"}],
    ))
    run = run_optical_project(document)
    workspace = json_api.workspace_init(document.payload())["workspace"]
    assert run.result.targets and any(t.value is not None for t in run.result.targets)
    assert run.result.observations[0].total_power_W > 0.0
    results = results_from_files(run_artifact_files(run), verify=False)
    monitor = run.result.observations[0].observation_id
    sampled = _sampled_results()
    assert sampled.sampled_light() and sampled.table("sampled-crossings")[0]
    mixtures = _mixture_results()
    assert mixtures.monitor("early")["polarization_mixtures"]
    checks = {
        "RunResults.targets": lambda: all(_carries(r, "value") for r in results.targets()["targets"]),
        "RunResults.monitors": lambda: all(_carries(r, "total_power_W") and all(_carries(a, "power_W") for a in r["arrivals"])
                                           for r in results.monitors()),
        "RunResults.monitor": lambda: (all(_carries(m, "degree_of_polarization") for m in mixtures.monitor("early")["polarization_mixtures"])
                                       and _carries(results.monitor(monitor), "total_power_W")
                                       and all(t.get("value") is None or t["value_basis"]["value"] == SENTINEL_JSON
                                               for t in results.monitor(monitor)["targets"])
                                       and all(a["beam"]["power_W"] is None or a["value_basis"]["beam.power_W"] == SENTINEL_JSON
                                               for a in results.monitor(monitor)["arrivals"])),
        "RunResults.superpositions": lambda: all("value_basis" in r for r in results.superpositions()),
        "RunResults.exposures": lambda: results.exposures()["available"] is False or "surfaces" in results.exposures(),
        # (each detector's quantities as the plane under the run's model-scope findings states them, value_basis included)
        "RunResults.detection": lambda: all(
            row["quantities"] == [x.model_dump(mode="json") for x in scoped_observation(next(
                o for o in run.result.observations if o.observation_id == row["observation_id"]),
                run.model_scope_issues).detection.quantities]
            for row in results.detection()),
        # (the sampled lines of a run with sampled light: ``sampled`` below)
        "RunResults.sampled_light": lambda: all(_carries(r, "power_W") for r in sampled.sampled_light()),
        "tables.sampled_light": lambda: all(_carries(_cell(r), "power_W") for r in sampled.table("sampled-light")[0]),
        "tables.sampled_crossings": lambda: all(_carries(_cell(r), "power_W") for r in sampled.table("sampled-crossings")[0]),
        "tables.monitors": lambda: all(r["total_power_W"] is None or r["value_basis"]["total_power_W"] == SENTINEL_JSON
                                       for r in results.table("monitors")[0]),
        "tables.arrivals": lambda: all(r["power_W"] is None or r["value_basis"]["power_W"] == SENTINEL_JSON
                                       for r in results.table("arrivals")[0]),
        "tables.superpositions": lambda: all("value_basis" in r for r in results.table("superpositions")[0]),
        "LocalOpticalRun.summary": lambda: all(_carries(t, "value") for t in run.summary()["targets"]),
        "json_api.run_summary": lambda: (all(_carries(t, "value") for t in json_api.run_summary(run)["targets"])
                                         and all(_carries(o, "total_power_W") and all(_carries(a, "power_W") for a in o["arrivals"])
                                                 for o in json_api.run_summary(run)["observations"])),
        "json_api.evaluate_project": lambda: _reply_carries(json_api.evaluate_project(document)),
        "json_api.workspace_run": lambda: _workspace_reply_carries(json_api.workspace_run(workspace)),
        "cli_workspace.run": lambda: _workspace_reply_carries(_cli_workspace_run(workspace, tmp_path)),
        "local.record.scientific_projection": lambda: "value_basis" not in json.dumps(
            scientific_projection(run.result, run.model_scope_issues).model_dump(mode="json")),
    }
    assert set(checks) == {p.name for p in CARRIAGE}, "a registered projection has no check here (or the reverse)"
    for projection in CARRIAGE:
        assert checks[projection.name](), projection.name
    # the CSV form of the column is the entries' compact JSON
    csv = results.export("monitors", "csv").decode("utf-8")
    assert '"{""total_power_W"":{""assumed"":[{""input"":""GAUSSIAN_PROFILE"",""value"":0.25}],""kind"":""ESTIMATE""}}"' in csv


def test_a_power_target_carries_the_entry_of_the_power_it_reads():
    """A ``power_W`` target reads its beams' received powers; where one is not the model's value -- a share a surface may
    take before the plane, an annular beam counted whole on an active area that may miss part of it -- the target's value
    carries that power's entry under ``value`` (an upper bound with ``BEAM_PART_ELSEWHERE``), so a reader by allow-list
    never takes the upper end of the interval it is graded on as the model's value. Scenes of the taken-share and
    annulus tests, read back through every projection that carries a target."""
    from test_diagnostics_round3 import _defocused, _t
    from test_taken_before_interval import LAUNCH_MM, _fold

    from helpers import one_arrival, run

    taken = run(_fold(1.0, 45.0, 0.5, LAUNCH_MM, targets=(
        Target(target_id="t", metric="power_W", observation_id="T", minimum=0.5e-3, unit="W", parameters={"crossing_index": 0}),)))
    annular = run(_defocused(1.0, _t("a", "OUT", minimum=1e-4)))
    for result, arrival, field in ((taken, next(a for a in taken.observations[0].arrivals if a.crossing_index == 0), "beam.power_W"),
                                   (annular, one_arrival(annular, "OUT"), "detected_power_W")):
        (target,) = result.targets
        read = basis_of(arrival, field)
        assert read is not None and read.kind is ValueKind.UPPER_BOUND
        assert target.value is not None and target.value_basis["value"] == read, (target.target_id, target.value_basis)
        assert any(o.part is OmittedPart.BEAM_PART_ELSEWHERE and o.upper > 0.0 for o in read.omitted)
        assert TargetResult.model_validate_json(target.model_dump_json()).value_basis == target.value_basis


def test_an_estimate_with_a_proven_interval_is_decided_on_it_as_the_guides_say(monkeypatch):
    """A coherent sum of 1 W whose left-out field (at most 1e-5 sqrt(W)) brings a rise above the level: the value is an
    ``ESTIMATE`` (``POSSIBLE_RETURN``) graded on the interval that field leaves, about [0.99998, 1.00002] W. A maximum of
    2 W and a minimum of 0.5 W are met, a maximum of 0.5 W unmet -- each still with the ``ESTIMATE`` entry -- and a
    maximum inside the interval is not decided. The README and the run help say an estimate's number alone decides
    nothing and the proven interval beside it decides."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics as M
    from os_optics.validation.power import LineFate

    fate = LineFate(input_W=1.0, unknown_W={}, below_floor_W=0.0, floor_significant=False, faint_root=1e-5, faint_W=5e-10,
                    level_W=1e-9)
    mv = M._untraced_interval(M.MetricValue(1.0, "W"), fate, None, coherent=True)
    assert mv.basis.kind is ValueKind.ESTIMATE and [o.part for o in mv.basis.omitted] == [OmittedPart.POSSIBLE_RETURN]
    low, high = mv.untraced
    assert low == pytest.approx((1.0 - 1e-5) ** 2, rel=1e-15) and high == pytest.approx((1.0 + 1e-5) ** 2, rel=1e-15)
    monkeypatch.setattr(M, "_measure", lambda *args, **kwargs: mv)
    for limits, assessment in (({"maximum": 2.0}, "MET"), ({"minimum": 0.5}, "MET"), ({"maximum": 0.5}, "UNMET"),
                               ({"maximum": 1.0}, "INDETERMINATE")):
        target = Target(target_id="t", metric="coherent_power_W", observation_id="P", unit="W", **limits)
        got = M._evaluate(target, {}, [], None, None, NumericalConfig())
        assert got.assessment.value == assessment, limits
        assert got.value == 1.0 and got.value_basis["value"].kind is ValueKind.ESTIMATE, limits
    root = Path(__file__).resolve().parents[1]
    readme = " ".join((root / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    assert "an estimate decides none" not in readme
    assert ("An estimate's number alone decides no target: where the result also states a proven interval the value lies in "
            "(the interval in the target's message, the range over degrees of coherence 0 to 1, the range coating "
            "group-delay bounds leave, the interval light the trace did not follow leaves), the target is `MET` or `UNMET` "
            "when that whole interval meets or misses it and otherwise `INDETERMINATE`; the value keeps its `ESTIMATE` entry "
            "either way") in readme
    from os_optics.cli import RUN_DESCRIPTION

    help_text = " ".join(RUN_DESCRIPTION.split())
    assert "estimate decides none." not in help_text
    assert ("an estimate's number alone decides none, and a proven interval the result states beside it (the target's "
            "message) decides the target when the whole interval meets or misses it") in help_text


def _workspace_reply_carries(reply: dict) -> bool:
    """A workspace run reply: every target value carries its entry, the same as its summary's row (the JSON API's at its
    quantization), and its failed checks where the row has them; a value kept as an estimate carries its error against
    ``target_value`` beside it (W5), the same as the summary's row."""
    assert reply["ok"], reply
    rows = {row["target_id"]: row for row in reply["summary"]["targets"]} if "summary" in reply else None
    keys = ("value", "target_value", "error", "value_basis", "validity", "assessment")
    estimates = [t for t in reply["targets"] if (t.get("value_basis") or {}).get("value", {}).get("kind") == "ESTIMATE"
                 and isinstance(t["value"], float | int) and isinstance(t.get("target_value"), float | int)]
    return (any(t["value"] is not None for t in reply["targets"])
            and all(_carries(t, "value") for t in reply["targets"])
            and bool(estimates) and all(isinstance(t.get("error"), float | int) for t in estimates)
            and (rows is None or all({k: rows[t["target_id"]].get(k) for k in keys} == {k: t.get(k) for k in keys}
                                     for t in reply["targets"])))


def _cli_workspace_run(workspace: dict, tmp_path) -> dict:
    """``os-optics workspace run`` (its default reply, and with ``--json`` its summary beside it)."""
    import contextlib
    import io

    from os_optics.cli import main

    path = tmp_path / "ws.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    replies = []
    for extra in ([], ["--json"]):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            main(["workspace", "run", str(path), "-o", str(tmp_path / "out.json"), *extra])
        replies.append(json.loads(printed.getvalue()))
    plain, full = replies
    assert plain["targets"] == full["targets"]
    return full


def _reply_carries(reply: dict) -> bool:
    assert reply["ok"], reply
    return (all(_carries(t, "value") for t in reply["targets"])
            and all(_carries(m, "total_power_W") and all(_carries(a, "power_W") for a in m["arrivals"]) for m in reply["monitors"]))


def test_a_registered_bare_projection_is_the_record_and_nothing_else():
    assert [p.name for p in CARRIAGE if p.carriage == "BARE"] == ["local.record.scientific_projection"]
