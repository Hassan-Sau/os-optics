"""The shared analysis result contract (contracts/analysis.py) and its builder (os_optics.analysis); the analyses' rows of
the capability table (``evaluation.capabilities.ANALYSES``): each Python call with every argument a caller passes, the
options of a request stated as options, and the text that tells a caller who named an analysis as a capability how to
run it."""

from __future__ import annotations

import importlib
import inspect
import json
import re

import pytest
from pydantic import ValidationError

from os_optics.analysis import completed, document, settings_hash
from os_optics.contracts.analysis import (
    AnalysisResultV1,
    Convergence,
    ConvergenceMethod,
    Interval,
    IntervalKind,
    ModelBasis,
    Quantity,
    ValidityCheck,
)
from os_optics.errors.status import QuantityState
from os_optics.manifest import engine_build_id

BASIS = ModelBasis(model_id="m", name="test model", version="1",
                   validity=(ValidityCheck(parameter="na", maximum=0.2, observed=0.1, within=True),))


def _q(**kwargs: object) -> Quantity:
    base: dict[str, object] = {"quantity_id": "q", "name": "q", "unit": "W"}
    base.update(kwargs)
    return Quantity(**base)  # type: ignore[arg-type]


def test_known_quantity_needs_a_value() -> None:
    with pytest.raises(ValidationError):
        _q(state=QuantityState.KNOWN)
    assert _q(state=QuantityState.KNOWN, value=1.0).value == 1.0


def test_unavailable_quantity_has_a_reason_and_no_value() -> None:
    with pytest.raises(ValidationError):
        _q(state=QuantityState.OUT_OF_MODEL_SCOPE)
    with pytest.raises(ValidationError):
        _q(state=QuantityState.UNKNOWN, value=1.0, reason="X")
    assert _q(state=QuantityState.OUT_OF_MODEL_SCOPE, reason="NA_ABOVE_SCALAR_LIMIT").value is None


def test_bounded_quantity_needs_an_interval_and_known_zero_is_zero() -> None:
    with pytest.raises(ValidationError):
        _q(state=QuantityState.BOUNDED)
    with pytest.raises(ValidationError):
        _q(state=QuantityState.KNOWN_ZERO, value=1e-3)
    with pytest.raises(ValidationError):
        Interval(lower=2.0, upper=1.0, kind=IntervalKind.PROVEN)


def test_only_a_resolved_value_or_a_proven_interval_decides() -> None:
    resolved = _q(state=QuantityState.KNOWN, value=0.50,
                  convergence=Convergence(method=ConvergenceMethod.REFINEMENT, resolution=0.01))
    assert resolved.decides(0.45)
    assert not resolved.decides(0.495)
    proven = _q(state=QuantityState.BOUNDED, interval=Interval(lower=0.6, upper=0.8, kind=IntervalKind.PROVEN))
    assert proven.decides(0.5) and proven.decides(0.9) and not proven.decides(0.7)
    nominal = _q(state=QuantityState.BOUNDED, interval=Interval(lower=0.6, upper=0.8, kind=IntervalKind.NOMINAL))
    assert not nominal.decides(0.5)
    assert not _q(state=QuantityState.UNKNOWN, reason="R").decides(0.5)


def test_document_binds_identity_and_checks_references() -> None:
    settings = {"grid": 64, "wavelength_nm": 633.0}
    doc = document("IMAGING", "1", settings, completed(), engine_input_hash="abc", models=(BASIS,),
                   quantities=(_q(state=QuantityState.KNOWN, values_ref="mtf", basis="m"),), arrays={"mtf": [1.0, 0.5]})
    assert doc.identity.physical.engine_build_id == engine_build_id()
    assert doc.identity.physical.settings_hash == settings_hash(settings)
    assert doc.identity.requirements is None
    dumped = json.loads(doc.model_dump_json(by_alias=True))
    assert dumped["schema"] == "opensci.analysis_result"
    assert AnalysisResultV1.model_validate(dumped) == doc
    with pytest.raises(ValidationError):
        document("IMAGING", "1", settings, completed(), quantities=(_q(state=QuantityState.KNOWN, value=1.0, basis="nope"),))
    with pytest.raises(ValidationError):
        document("IMAGING", "1", settings, completed(), quantities=(_q(state=QuantityState.KNOWN, values_ref="nope"),))


def test_a_document_takes_the_build_id_its_caller_holds(monkeypatch) -> None:
    """An analysis of a run or a session states the build id its caller already has: the Engine's sources are read to
    identify the build only when none is given."""
    import os_optics.analysis as analysis
    from os_optics.analysis.configurations import configurations_comparison
    from os_optics.local import example_project

    project = example_project()
    identified: list[int] = []

    def identify() -> str:
        identified.append(1)
        return "build-identified-here"

    monkeypatch.setattr(analysis, "engine_build_id", identify)
    held = document("IMAGING", "1", {"grid": 64}, completed(), engine_build_id="build-of-the-session")
    assert held.identity.physical.engine_build_id == "build-of-the-session" and identified == []
    compared = configurations_comparison(project, [], engine_input_hash="abc", engine_build_id="build-of-the-session")
    assert compared.identity.physical.engine_build_id == "build-of-the-session" and identified == []
    assert document("IMAGING", "1", {"grid": 64}, completed()).identity.physical.engine_build_id == "build-identified-here"
    assert configurations_comparison(project, [], engine_input_hash="abc").identity.physical.engine_build_id == "build-identified-here"
    assert identified == [1, 1]
    # (the build id is the only difference: the same settings give the same settings hash)
    assert held.identity.physical.settings_hash == settings_hash({"grid": 64})


def test_a_threshold_moves_only_the_requirement_identity() -> None:
    a = document("IMAGING", "1", {"grid": 64}, completed(), targets_hash="t1")
    b = document("IMAGING", "1", {"grid": 64}, completed(), targets_hash="t2")
    assert a.identity.physical == b.identity.physical
    assert a.identity.requirements != b.identity.requirements


# ------------------------------------------------------------------------------------------------- the analyses' rows

#: Arguments that carry a run's identity on to the analysis document: the RunResults method passes them, a caller does not.
IDENTITY_ARGUMENTS = frozenset({"run_identity", "config", "engine_build_id", "run_result_hash"})
CALL = re.compile(r"(os_optics(?:\.\w+)+)\(([^()]*)\)")


def _callable(dotted: str):
    """The object a dotted name of a row reaches as a caller reaches it: attribute by attribute (a package's export, not a
    submodule of the same name)."""

    head, *names = dotted.split(".")
    target = importlib.import_module(head)
    for index, name in enumerate(names):
        if not hasattr(target, name):
            importlib.import_module(".".join([head, *names[:index + 1]]))
        target = getattr(target, name)
    return target


def _rows():
    from os_optics.evaluation.capabilities import ANALYSES, EXPORTS

    return (*ANALYSES, *EXPORTS)


def test_every_call_of_an_analysis_row_names_the_arguments_of_its_callable():
    calls = 0
    for row in _rows():
        if not row.entry_point:
            continue
        for dotted, arguments in CALL.findall(row.entry_point):
            parameters = inspect.signature(_callable(dotted)).parameters
            named = [a.strip().lstrip("*").split("=")[0] for a in arguments.split(",") if a.strip()]
            assert set(named) <= set(parameters), (row.id, dotted, named)
            stated = {n for n, p in parameters.items() if n != "self" and p.kind is not p.VAR_KEYWORD} - IDENTITY_ARGUMENTS
            assert stated <= set(named), (row.id, dotted, sorted(stated - set(named)))
            if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):  # (said in the row's summary)
                assert "further keyword arguments" in row.summary, (row.id, dotted)
            calls += 1
    assert calls >= 14


def test_the_rows_state_a_request_option_as_an_option():
    from os_optics.evaluation.capabilities import ANALYSIS_BY_ID

    for name, options in (("imaging", ("convergence false", "diffraction false")), ("ghosts", ("convergence false",))):
        method = ANALYSIS_BY_ID[name].method
        assert "leaves out" not in method and "A request can leave out" in method, method
        for option in options:
            assert option in method, (name, option)


def test_the_refusal_of_an_analysis_names_a_results_object_only_where_its_call_takes_one():
    from os_optics.evaluation.capabilities import ANALYSES
    from os_optics.evaluation.plan import EvaluationPlanError, plan_evaluation

    for row in ANALYSES:
        with pytest.raises(EvaluationPlanError) as refused:
            plan_evaluation((row.id,), level="local", what="outputs")
        message = refused.value.message
        assert row.entry_point in message and (row.command or "") in message, row.id
        takes = "RunResults" in row.entry_point
        assert ("(RunResults: Evaluation.results()" in message) is takes, (row.id, message)
        assert ("load_run(RUN)" in message) is takes, (row.id, message)
        assert ("(RUN: a run directory" in message) is ("RUN" in message.split(" (RUN: ")[0]), (row.id, message)
        if takes and row.requires:
            assert f"of an evaluation that computed {', '.join(row.requires)}," in message, (row.id, message)
