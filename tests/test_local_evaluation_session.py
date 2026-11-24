"""The local surface of composable evaluation (``os_optics.local.evaluate``): a session's evaluations are compositions of the
local run's own stages -- the complete one is ``run_optical_project`` -- a narrowed one enters only the phases of what it
computes, its retained quantities equal the complete run's, its results refuse what it did not compute, and a session
reuses catalog, compiled elements and Engine products exactly as far as they stay valid."""

from __future__ import annotations

import copy
import json

import pytest

from os_optics.evaluation import EvaluationPlanError
from os_optics.local import (EvaluationSession, LocalRunError, evaluate_project, example_project, example_project_of_kind, load_project,
                              run_optical_project, verify_run_artifacts)
from os_optics.local.artifacts import run_artifact_files
from os_optics.local.record import local_validation_record
from os_optics.results import NOT_COMPUTED, ResultsError, load_run
from os_optics.run_control import STOP_CALLER, RunStopped, monitored
from os_optics.run_monitor import RunMonitor

from declared_helpers import declared_project
from local_helpers import ALONG_X, WINDOW, catalog_document, element, plane, project, source, with_lossless_bulk

#: The phases a readings evaluation never enters (the probe pass, the bounds, the exposures, the record, the view, files).
SKIPPED_BY_READINGS = ("untraced_light", "return_analysis", "bounds", "exposures", "scene", "record", "write")


class _Phases:
    """A run monitor that records the phases entered, and stops the run at the ``at``-th checkpoint of ``stop_in``."""

    def __init__(self, stop_in: str | None = None, at: int = 1) -> None:
        self.stop_in, self.at, self.entered, self.stack, self.steps = stop_in, at, [], [], 0

    def enter(self, name: str, total: int | None, unit: str) -> None:
        self.entered.append(name)
        self.stack.append(name)

    def leave(self, name: str) -> None:
        self.stack.pop()

    def unwind(self, name: str) -> None:
        self.stack.pop()

    def step(self, count: int) -> None:
        if self.stop_in is not None and self.stack and self.stack[-1] == self.stop_in:
            self.steps += 1
            if self.steps == self.at:
                raise RunStopped(STOP_CALLER, "stopped by the test", phase=self.stop_in)


def _moved(document: dict, instance_id: str, dx_mm: float) -> dict:
    out = copy.deepcopy(document)
    component = next(c for c in out["components"] if c["instance_id"] == instance_id)
    component["nominal_pose"]["position_mm"][0] += dx_mm
    return out


def _window_document() -> dict:
    """A beam half over a window its chief ray misses (``BEAM_PARTIALLY_INTERCEPTED``), with a beam-radius target on the
    detector plane (read from the plane readings) and a power target (read from the plane bounds)."""

    edge = 12.7  # (the window's published edge radius, tests/test_local_beam_scope.py)
    radius = {"target_id": "radius", "metric": "beam_radius_mm", "observation_id": "D", "maximum": 10.0, "unit": "mm"}
    power = {"target_id": "power", "metric": "power_W", "observation_id": "D", "minimum": 0.0009, "unit": "W"}
    return project([source(waist_mm=3.0), element("W", "window", WINDOW, (0.0, edge + 0.01, 0.0), ALONG_X)],
                   [plane("D", (100.0, 0.0, 0.0), radius_mm=25.0)], [radius, power])


def test_run_optical_project_is_the_complete_evaluation():
    document = example_project_of_kind("two_beam_interference")
    reference = run_optical_project(document)
    evaluation = EvaluationSession().evaluate(document, outputs=("complete",))
    assert evaluation.complete and evaluation.result.evaluation is None
    run = evaluation.run()
    assert run.local_validation.payload() == reference.local_validation.payload()
    assert run.result.manifest.result_hash == reference.result.manifest.result_hash
    assert run.result.manifest.trace_hash == reference.result.manifest.trace_hash
    assert run.optical_view.content_hash() == reference.optical_view.content_hash()
    # (every run file, the manifest included, byte for byte: the run files hold no timing)
    assert run_artifact_files(run) == run_artifact_files(reference)


def test_a_readings_evaluation_enters_only_the_phases_of_what_it_computes():
    monitor = _Phases()
    session = EvaluationSession()
    with monitored(monitor):
        evaluation = session.evaluate(example_project_of_kind("two_beam_interference"))
    assert evaluation.scope.requested == ("readings",)
    assert evaluation.scope.executed == ("scene", "beam_trace", "secondary_rays", "plane_readings", "ray_paths", "model_scope")
    # (the Engine-level scope names the Engine capabilities the local evaluation asked for by id; the caller's is ev.scope)
    assert evaluation.result.evaluation.requested == evaluation.result.evaluation.executed == evaluation.scope.executed[:4]
    # (a number at the local level comes with its model-scope findings: plane_readings named alone brings them)
    alone = session.evaluate(example_project_of_kind("two_beam_interference"), outputs=("plane_readings",))
    assert alone.scope.requested == ("plane_readings",) and alone.scope.executed == evaluation.scope.executed
    assert ("model_scope", "plane_readings") in {(a.capability, a.needed_by) for a in alone.scope.added}
    assert not set(SKIPPED_BY_READINGS) & set(monitor.entered)
    assert {"lower", "trace", "observation_results", "ray_paths", "model_scope"} <= set(monitor.entered)
    # (the optical-solid check is a capability readings leave out)
    assert evaluation.lowered.solids is None and "solid_check" not in session.statistics()["executions"]
    assert evaluation.local_validation is None and evaluation.optical_view is None and not evaluation.complete
    not_computed = {row["capability"] for row in evaluation.not_computed()}
    assert {"solid_check", "untraced_light", "plane_bounds", "surface_exposures", "beam_pairs", "requirements", "record", "view"} <= not_computed


def test_retained_quantities_equal_the_complete_run_with_the_model_scope_applied():
    document = _window_document()
    catalog = catalog_document()
    readings = evaluate_project(document, outputs=("readings",), targets=("radius",), catalog=catalog)
    complete = load_run(run_optical_project(document, catalog=catalog))
    results = readings.results()
    # the findings that decide whether a number applies come with the readings
    assert [i["code"] for i in results.targets()["model_scope_issues"]] == ["BEAM_PARTIALLY_INTERCEPTED"]
    assert readings.status == "OUT_OF_MODEL_SCOPE"
    mine = {row["target_id"]: row for row in results.targets()["targets"]}
    theirs = {row["target_id"]: row for row in complete.targets()["targets"]}
    assert set(mine) == {"radius"}
    for key in ("value", "status", "assessment", "applicability_issue_codes", "unit"):
        assert mine["radius"][key] == theirs["radius"][key], key
    assert mine["radius"]["assessment"] == "INDETERMINATE"
    assert results.targets()["in_model_scope"] is False is complete.targets()["in_model_scope"]
    assert [(t["target_id"], t["reason"]) for t in results.targets()["not_evaluated"]] == [("power", "NOT_REQUESTED")]
    # the plane's readings are the complete run's; its power state is not computed
    (mine_plane,), (their_plane,) = results.monitors(), complete.monitors()
    assert mine_plane["quantity_state"] == "OUT_OF_MODEL_SCOPE" == their_plane["quantity_state"]  # (a finding names the plane)
    assert mine_plane["total_power_W"] == their_plane["total_power_W"] and mine_plane["arrivals"] == their_plane["arrivals"]
    assert mine_plane["power_complete"] is False
    beams = lambda doc: [{k: v for k, v in a.items() if k not in ("power_upper_bound_W",)} for a in doc["arrivals"]]  # noqa: E731
    assert beams(results.monitor("D")) == beams(complete.monitor("D"))


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_planes_without_bounds_are_not_computed_never_zero(lossless):
    catalog = catalog_document()
    session = EvaluationSession(catalog=with_lossless_bulk(catalog) if lossless else catalog)
    evaluation = session.evaluate(example_project(catalog))
    results = evaluation.results()
    (plane_row,) = results.monitors()
    assert plane_row["quantity_state"] == "NOT_COMPUTED" and plane_row["power_complete"] is False
    (observation,) = evaluation.result.observations
    if lossless:
        assert observation.total_power_W > 0.0
    else:
        assert observation.total_power_W is None
        (arrival,) = observation.arrivals
        assert arrival.beam.power_W is None and arrival.beam.gaussian.beam_radius_x_mm > 0.0
    # (the untraced power was not computed: null in every row, table and export, never 0.0)
    plane = results.monitor("focal_plane")
    for row in (plane_row, plane):
        assert row["untraced_power_W"] is None and "power_upper_bound_W" not in row
        assert [line["untraced_power_W"] for line in row["power_by_wavelength"]] == [None]
    (table_row,), _summary = results.table("monitors")
    assert table_row["untraced_power_W"] is None and table_row["quantity_state"] == "NOT_COMPUTED"
    header, values = results.export("monitors", "csv").decode("utf-8").splitlines()[:2]
    assert dict(zip(header.split(","), values.split(",")))["untraced_power_W"] == ""


@pytest.mark.parametrize("query, capability", [
    (lambda r: r.power_budget(), "power_ledger"), (lambda r: r.exposures(), "surface_exposures"),
    (lambda r: r.targets(), "requirements"), (lambda r: r.detection(), "plane_bounds"),
    (lambda r: r.table("power-budget"), "power_ledger"), (lambda r: r.export("exposures", "csv"), "surface_exposures"),
])
def test_a_query_of_what_was_not_computed_names_the_capability_to_request(query, capability):
    results = EvaluationSession().evaluate(example_project()).results()
    with pytest.raises(ResultsError) as refused:
        query(results)
    assert refused.value.code == NOT_COMPUTED == "NOT_COMPUTED" and repr(capability) in refused.value.message


def test_the_plane_readings_and_requirements_always_come_with_the_model_scope():
    """However the selection is spelled, a number of a Project evaluation comes with the findings that say whether it
    applies: excluding them, or the ray paths they read, is refused -- never graded MET without them."""
    for outputs in (("plane_readings",), ("requirements",), ("readings",), ("plane_readings", "requirements")):
        for excluded in ("ray_paths", "model_scope"):
            with pytest.raises(EvaluationPlanError) as refused:
                EvaluationSession().evaluate(example_project(), outputs=outputs, exclude=(excluded,))
            assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and refused.value.context["capability"] == excluded
    # (the half-intercepted window: the radius is INDETERMINATE, as the run says, from any spelling)
    for outputs in (("plane_readings", "requirements"), ("requirements",)):
        window = evaluate_project(_window_document(), outputs=outputs, targets=("radius",), catalog=catalog_document())
        assert {"ray_paths", "model_scope"} <= set(window.scope.executed)
        (radius,) = window.results().targets()["targets"]
        assert radius["assessment"] == "INDETERMINATE" and window.status == "OUT_OF_MODEL_SCOPE"


def test_a_model_scope_finding_names_every_declared_target_it_affects_whatever_the_evaluation_assessed():
    """Which targets a finding affects is the Project's statement (the plane or device each target names): the readings,
    which assess no target, and a selection of one target name the targets the complete run names."""
    from os_optics.local import json_api

    document, catalog = _window_document(), catalog_document()
    complete = run_optical_project(document, catalog=catalog)
    (reference,) = complete.model_scope_issues
    assert reference.code == "BEAM_PARTIALLY_INTERCEPTED" and reference.affected_target_ids == ("power", "radius")
    for outputs, targets in ((("readings",), None), (("requirements",), ("radius",)), (("requirements",), None),
                             (("complete",), None)):
        evaluation = evaluate_project(document, outputs=outputs, targets=targets, catalog=catalog)
        assert evaluation.model_scope_issues == complete.model_scope_issues, (outputs, targets)
        (described,) = evaluation.results().describe()["status"]["model_scope_issues"]
        assert described["affected_target_ids"] == ["power", "radius"] and described["affected_observation_ids"] == ["D"]
    reply = json_api.evaluate_project(document, catalog=catalog)
    assert reply["targets"] == [] and reply["model_scope_issues"][0]["affected_target_ids"] == ["power", "radius"]
    # (a selected target is graded by the finding as in the run; the record keeps reading the findings per assessed target)
    selected = evaluate_project(document, outputs=("requirements",), targets=("radius",), catalog=catalog)
    (radius,) = selected.results().targets()["targets"]
    assert radius["assessment"] == "INDETERMINATE" and radius["applicability_issue_codes"] == ["BEAM_PARTIALLY_INTERCEPTED"]
    assert selected.run().local_validation.payload() == complete.local_validation.payload()


def test_ray_queries_read_the_ray_paths_and_the_plane_readings():
    traced = EvaluationSession().evaluate(example_project(), outputs=("beam_trace",))
    assert traced.ray_paths is None and traced.model_scope_issues is None
    # (the path list is the beam trace's and the secondary rays': refused until both ran, then answered)
    with pytest.raises(ResultsError) as refused:
        traced.results().paths()
    assert refused.value.code == NOT_COMPUTED and "'secondary_rays'" in refused.value.message
    assert traced.extend(outputs=("secondary_rays",)).results().paths()
    with pytest.raises(ResultsError) as refused:
        traced.results().rays(monitor="focal_plane")
    assert refused.value.code == NOT_COMPUTED and "'ray_paths'" in refused.value.message and "'plane_readings'" in refused.value.message
    # (the ray paths without the plane readings: the crossings of a plane are the readings' -- refused, never [])
    recorded = EvaluationSession().evaluate(example_project_of_kind("two_beam_interference"), outputs=("ray_paths",))
    assert recorded.scope.executed == ("scene", "beam_trace", "ray_paths")
    results = recorded.results()
    for query in (lambda: results.rays(), lambda: results.rays(monitor="crossing_plane"),
                  lambda: results.wavefront("crossing_plane", "SRC#0"), lambda: results.table("rays"),
                  lambda: results.table("wavefront", monitor="crossing_plane", path="SRC#0")):
        with pytest.raises(ResultsError) as refused:
            query()
        assert refused.value.code == NOT_COMPUTED and "'plane_readings'" in refused.value.message
    assert recorded.extend(outputs=("plane_readings",)).results().rays()


@pytest.mark.parametrize(("kind", "lossless"), [
    ("two_beam_interference", False), ("focus", True), ("folded", True), ("focus", False), ("folded", False),
])
def test_a_complete_evaluations_results_are_those_of_its_run_directory(kind, lossless, tmp_path):
    """Every table and export, the power budget and the status block of a complete evaluation held in memory equal those
    read from the run directory it writes (rows in the same order, the same bytes)."""
    from os_optics.results import EXPORT_KINDS

    catalog = catalog_document()
    document = example_project_of_kind(kind, catalog)
    evaluation = EvaluationSession(catalog=with_lossless_bulk(catalog) if lossless else catalog).evaluate(document, outputs=("complete",))
    assert evaluation.complete
    evaluation.write_run(tmp_path / "run")
    memory, disk = evaluation.results(), load_run(tmp_path / "run")
    plane = memory.monitors()[0]["observation_id"]
    path = memory.arrivals(monitor=plane)[0]["path_id"]
    filters = {"wavefront": {"monitor": plane, "path": path}, "irradiance": {"monitor": plane, "samples": 8},
               "handoff": {"monitor": plane}}
    for export_kind in EXPORT_KINDS:
        for fmt in ("csv", "json"):
            options = filters.get(export_kind, {})
            if export_kind == "irradiance" and not lossless and kind in ("focus", "folded"):
                with pytest.raises(ResultsError) as from_memory:
                    memory.export(export_kind, fmt, **options)
                with pytest.raises(ResultsError) as from_disk:
                    disk.export(export_kind, fmt, **options)
                assert from_memory.value.code == from_disk.value.code
                assert from_memory.value.message == from_disk.value.message
            else:
                assert memory.export(export_kind, fmt, **options) == disk.export(export_kind, fmt, **options), (export_kind, fmt)
    assert [(row["category"], row["name"]) for row in memory.power_budget()["rows"]] == [
        (row["category"], row["name"]) for row in disk.power_budget()["rows"]]
    assert memory.power_budget() == disk.power_budget()
    ours, theirs = memory.describe(), disk.describe()
    assert ours["status"] == theirs["status"] and set(ours["status"]) == {
        "engine_status", "validation_status", "simulation_status", "model_scope_issues", "targets"}
    assert all("assessment" in row for row in ours["status"]["targets"]) and ours["status"]["targets"]
    # (everything but what describes the directory itself: its integrity check and its files)
    for key in theirs:
        if key not in ("integrity", "files"):
            assert ours[key] == theirs[key], key
    import json

    assert json.loads(json.dumps(ours["status"])) == ours["status"]  # (plain JSON values)


def test_a_complete_evaluation_answers_every_query_from_memory():
    document = example_project()
    evaluation = EvaluationSession().evaluate(document, outputs=("complete",))
    mine, theirs = evaluation.results(), load_run(run_optical_project(document))
    assert mine.targets()["targets"] == theirs.targets()["targets"] and mine.monitors() == theirs.monitors()
    assert mine.power_budget() == theirs.power_budget() and mine.exposures() == theirs.exposures()
    assert mine.rays(monitor="focal_plane") == theirs.rays(monitor="focal_plane")
    assert mine.identity()["result_hash"] == theirs.identity()["result_hash"]
    assert mine.identity()["project_hash"] == theirs.identity()["project_hash"]
    assert "evaluation" not in mine.describe()


def test_extend_reuses_what_was_computed_and_equals_a_fresh_evaluation():
    document = example_project_of_kind("two_beam_interference")
    session = EvaluationSession()
    readings = session.evaluate(document)
    extended = readings.extend(outputs=("plane_bounds",))
    assert set(extended.reused) == {"scene", "beam_trace", "secondary_rays", "plane_readings"}
    assert "beam_trace" not in extended.executed and "plane_bounds" in extended.executed
    executions = session.statistics()["executions"]
    assert all(executions[c] == 1 for c in ("scene", "beam_trace", "secondary_rays", "plane_readings", "untraced_light", "plane_bounds"))
    assert session.statistics()["prepared"] == 1  # (the extension reads the lowering it extends)
    fresh = EvaluationSession().evaluate(document, outputs=("readings", "plane_bounds"))
    assert extended.result.manifest.result_hash == fresh.result.manifest.result_hash
    assert extended.result.model_dump(mode="json") == fresh.result.model_dump(mode="json")
    run = extended.run()
    assert run.result.manifest.result_hash == run_optical_project(document).result.manifest.result_hash
    assert session.statistics()["executions"]["beam_trace"] == 1 and session.statistics()["executions"]["solid_check"] == 1


@pytest.mark.parametrize("first", [("beam_trace",), ("secondary_rays",), ("untraced_light",)])
def test_an_evaluation_made_without_the_ray_paths_is_not_traced_again_when_a_later_one_reads_them(first):
    """The session observes every trace it makes: an evaluation that leaves the ray paths out has none, and its
    extension and its run read the rays of the retained trace instead of tracing (and probing) again."""
    document = example_project_of_kind("two_beam_interference")
    session = EvaluationSession()
    evaluation = session.evaluate(document, outputs=first)
    assert evaluation.ray_paths is None and "ray_paths" not in evaluation.executed
    assert "ray_paths" in {row["capability"] for row in evaluation.not_computed()}
    traced = dict(session.statistics()["executions"])
    extended = evaluation.extend(outputs=("readings",))
    assert set(first) <= set(extended.reused) and "beam_trace" in extended.reused
    run = evaluation.run()
    stats = session.statistics()
    assert stats["invalidations"] == {}
    assert all(stats["executions"][c] == 1 for c in traced)  # (scene, beam trace, and what the first selection added)
    fresh = EvaluationSession().evaluate(document, outputs=(*first, "readings"))
    assert extended.ray_paths == fresh.ray_paths and extended.ray_paths is not None
    assert extended.result.model_dump(mode="json") == fresh.result.model_dump(mode="json")
    reference = run_optical_project(document)
    assert run.result.manifest.result_hash == reference.result.manifest.result_hash and run.ray_paths == reference.ray_paths


def test_a_session_reuses_compiled_elements_and_invalidates_engine_products_when_a_pose_changes():
    document = example_project().payload()
    session = EvaluationSession()
    session.evaluate(document)
    assert session.statistics()["compiled_elements"] == 1 and session.statistics()["reused_elements"] == 0
    moved = session.evaluate(_moved(document, "L1", 1.0))
    stats = session.statistics()
    assert stats["compiled_elements"] == 1 and stats["reused_elements"] == 1  # (a pose is no part of a compiled element)
    assert stats["invalidations"] == {"inputs": 1} and stats["executions"]["beam_trace"] == 2 and moved.reused == ()
    assert stats["catalogs"] == 1
    # (the values: the moved candidate and the one moved back equal fresh evaluations)
    assert moved.result.manifest.result_hash == evaluate_project(_moved(document, "L1", 1.0)).result.manifest.result_hash
    assert session.evaluate(document).result.manifest.result_hash == evaluate_project(document).result.manifest.result_hash


def test_a_changed_target_threshold_reuses_every_product():
    document = example_project().payload()
    session = EvaluationSession()
    first = session.evaluate(document, outputs=("readings", "requirements"))
    stricter = copy.deepcopy(document)
    stricter["targets"][1]["maximum"] = stricter["targets"][1]["maximum"] / 10.0
    second = session.evaluate(stricter, outputs=("readings", "requirements"))
    stats = session.statistics()
    assert set(second.reused) == {"scene", "beam_trace", "secondary_rays", "plane_readings"}
    assert stats["invalidations"] == {} and stats["executions"]["beam_trace"] == 1 and stats["executions"]["requirements"] == 2
    assert second.ray_paths is not None and second.ray_paths.paths == first.ray_paths.paths  # (the observed trace, not traced again)
    statuses = {t.target_id: t.status.value for t in second.result.targets}
    fresh = {t.target_id: t.status.value for t in evaluate_project(stricter, outputs=("readings", "requirements")).result.targets}
    assert statuses == fresh


def test_a_material_change_compiles_again_and_invalidates_the_engine_products():
    document = declared_project(example_project().payload(), catalog_document())
    session = EvaluationSession()
    first = session.evaluate(document)
    changed = copy.deepcopy(document)
    changed["model_materials"][0]["dispersion"]["b"][0] = "1.05"
    second = session.evaluate(changed)
    stats = session.statistics()
    assert stats["compiled_elements"] == 2 and stats["reused_elements"] == 0
    assert stats["invalidations"] == {"inputs": 1}
    fresh = evaluate_project(changed).result.manifest.result_hash
    assert second.result.manifest.result_hash == fresh != first.result.manifest.result_hash


def test_a_compute_budget_change_and_another_build_invalidate_the_engine_products():
    document = example_project().payload()
    session = EvaluationSession()
    session.evaluate(document)
    budget = {**copy.deepcopy(document), "compute_budget": {"max_paths": 256}}
    second = session.evaluate(budget)
    assert session.statistics()["invalidations"] == {"numerical_configuration": 1}
    fresh = evaluate_project(budget)
    assert second.result.manifest.result_hash == fresh.result.manifest.result_hash
    session.state.build_id = "0.0.0+another-build"
    evaluation = session.evaluate(budget)
    assert session.statistics()["invalidations"] == {"numerical_configuration": 1, "engine_build": 1}
    assert evaluation.result.manifest.engine_build_id == "0.0.0+another-build" and evaluation.reused == ()
    assert evaluation.result.manifest.result_hash == fresh.result.manifest.result_hash
    # (the trace of the rebound state is recorded anew: its ray paths are a fresh evaluation's, once each, and stay so)
    assert evaluation.ray_paths == fresh.ray_paths and len(evaluation.ray_paths.paths) == len(evaluation.result.paths)
    again = session.evaluate(budget)
    assert again.ray_paths == fresh.ray_paths and again.reused
    assert evaluation.run().optical_view is not None


def test_another_catalog_object_compiles_its_elements_again():
    from os_optics.local import load_public_catalog, lower_project

    cache: dict = {}
    first = load_public_catalog(catalog_document())
    other = first.model_copy(deep=True)
    document = example_project(first)
    lower_project(document, first, compiled=cache)
    ((key, compiled),) = cache.items()
    lower_project(document, first, compiled=cache)
    assert list(cache) == [key] and cache[key] is compiled
    lower_project(document, other, compiled=cache)
    # (the dictionary holds the elements of the latest lowering: the other object's, compiled anew, in place of the first's)
    ((other_key, recompiled),) = cache.items()
    assert other_key != key and recompiled is not compiled


def test_a_session_holds_the_compiled_elements_of_the_latest_project_it_lowered():
    """A loop of changed prescriptions does not grow what the session keeps; a candidate that only moves an element still
    reuses the compiled elements, and the counters count each compiled element once."""
    document = declared_project(example_project().payload(), catalog_document())
    session = EvaluationSession()
    session.evaluate(document)
    elements = len(session._compiled)
    assert elements == 1 and session.statistics()["compiled_elements"] == 1
    candidate = document
    for step in range(1, 6):
        candidate = copy.deepcopy(document)
        surface = next(s for s in candidate["optical_models"][0]["model"]["surfaces"] if "radius_mm" in s)
        surface["radius_mm"] = repr(float(surface["radius_mm"]) * (1.0 + 1e-4 * step))
        evaluation = session.evaluate(candidate)
        stats = session.statistics()
        assert len(session._compiled) == elements  # (the earlier prescription's elements are dropped)
        assert stats["compiled_elements"] == elements * (step + 1) and stats["reused_elements"] == 0
    assert evaluation.result.manifest.result_hash == evaluate_project(candidate).result.manifest.result_hash
    # (a pose is no part of a compiled element: the moved candidate compiles nothing)
    moved = session.evaluate(_moved(candidate, "L1", 1.0))
    stats = session.statistics()
    assert stats["compiled_elements"] == 6 and stats["reused_elements"] == 1 and len(session._compiled) == elements
    assert moved.result.manifest.result_hash == evaluate_project(_moved(candidate, "L1", 1.0)).result.manifest.result_hash
    # (the first prescription again: the session holds the latest Project's elements, so it is compiled again)
    session.evaluate(document)
    assert session.statistics()["compiled_elements"] == 7 and len(session._compiled) == elements


def test_a_catalog_assigned_to_the_session_is_read_by_its_next_evaluation():
    base = catalog_document()
    other = copy.deepcopy(base)  # (the same release label, another lens radius)
    product = next(p for p in other["products"] if p["optical_sku"] == "OS-OPT-TEST-LENS-PCX-F100")
    surface = next(s for s in product["simulation_models"][0]["surfaces"] if s["surface_id"] == "S1")
    surface["radius_mm"] = "77.25"
    document = example_project(base).payload()
    session = EvaluationSession(catalog=base)
    first = session.evaluate(document, outputs=("readings", "requirements"))
    session.catalog = other
    second = session.evaluate(document, outputs=("readings", "requirements"))
    fresh = EvaluationSession(catalog=other).evaluate(document, outputs=("readings", "requirements"))
    assert second.result.manifest.result_hash == fresh.result.manifest.result_hash != first.result.manifest.result_hash
    assert session.statistics()["catalogs"] == 2 and session.statistics()["compiled_elements"] == 2


@pytest.mark.parametrize("as_path", [False, True], ids=["str", "Path"])
def test_a_catalog_file_given_to_the_session_is_read_by_its_content(tmp_path, as_path):
    """A session given a catalog file reads the file at each evaluation and keys it by its resolved path and the digest of
    its bytes: the file edited on disk (same release label, same ids) gives the fresh session's result, the unchanged file
    reuses the compiled elements, and the session holds one catalog per file."""
    from pathlib import Path

    base = catalog_document()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    document = example_project(base).payload()
    given = path if as_path else str(path)
    session = EvaluationSession(catalog=given)
    first = session.evaluate(document, outputs=("readings", "requirements"))
    again = session.evaluate(document, outputs=("readings", "requirements"))
    stats = session.statistics()
    assert again.result.manifest.result_hash == first.result.manifest.result_hash
    assert stats["catalogs"] == 1 and stats["compiled_elements"] == 1 and stats["reused_elements"] == 1
    product = next(p for p in base["products"] if p["optical_sku"] == "OS-OPT-TEST-LENS-PCX-F100")
    surface = next(s for s in product["simulation_models"][0]["surfaces"] if s["surface_id"] == "S1")
    surface["radius_mm"] = "77.25"
    path.write_text(json.dumps(base), encoding="utf-8")
    second = session.evaluate(document, outputs=("readings", "requirements"))
    fresh = EvaluationSession(catalog=str(path)).evaluate(document, outputs=("readings", "requirements"))
    assert fresh.result.manifest.result_hash != first.result.manifest.result_hash
    assert second.result.manifest.result_hash == fresh.result.manifest.result_hash
    stats = session.statistics()
    assert stats["catalogs"] == 1 and stats["compiled_elements"] == 2
    # (the same file named through another spelling of its path is the same file)
    session.catalog = Path(tmp_path, ".", "catalog.json")
    session.evaluate(document, outputs=("readings", "requirements"))
    assert session.statistics()["catalogs"] == 1 and session.statistics()["compiled_elements"] == 2


def test_an_interrupted_evaluation_leaves_the_session_usable_and_the_next_call_continues():
    catalog = with_lossless_bulk(catalog_document())
    document = example_project_of_kind("two_beam_interference", catalog)
    session = EvaluationSession(catalog=catalog)
    monitor = _Phases(stop_in="untraced_light")
    with pytest.raises(RunStopped) as stopped:
        session.evaluate(document, outputs=("complete",), monitor=monitor)
    assert stopped.value.phase == "untraced_light" and stopped.value.stopped is not None
    assert session.state.retained == ("scene", "beam_trace", "secondary_rays", "plane_readings", "power_ledger")
    evaluation = session.evaluate(document, outputs=("complete",))
    assert {"scene", "beam_trace", "secondary_rays", "plane_readings", "power_ledger"} <= set(evaluation.reused)
    assert session.statistics()["executions"]["beam_trace"] == 1 and session.statistics()["executions"]["untraced_light"] == 1
    assert evaluation.run().result.manifest.result_hash == run_optical_project(document, catalog=catalog).result.manifest.result_hash


def test_a_stop_inside_the_trace_records_the_next_trace_anew():
    document = example_project_of_kind("two_beam_interference")
    session = EvaluationSession()
    with pytest.raises(RunStopped):
        session.evaluate(document, monitor=_Phases(stop_in="trace"))
    evaluation = session.evaluate(document)
    fresh = EvaluationSession().evaluate(document)
    assert evaluation.ray_paths == fresh.ray_paths
    assert evaluation.result.manifest.result_hash == fresh.result.manifest.result_hash


class _Interrupted(RunMonitor):
    """A run monitor whose process is interrupted (Ctrl-C) on entering the phase ``at``."""

    def __init__(self, at: str) -> None:
        super().__init__()
        self.at = at

    def enter(self, name: str, total: int | None, unit: str) -> None:
        super().enter(name, total, unit)
        if name == self.at:
            raise KeyboardInterrupt


@pytest.mark.parametrize("call", [
    lambda ev, monitor, tmp: ev.extend(outputs=("plane_bounds",), monitor=monitor),
    lambda ev, monitor, tmp: ev.run(monitor=monitor),
    lambda ev, monitor, tmp: ev.write_run(tmp / "run", monitor=monitor),
], ids=["extend", "run", "write_run"])
def test_an_interrupt_of_an_extension_or_a_run_stops_it_with_a_record_as_evaluate_does(call, tmp_path):
    document = example_project_of_kind("two_beam_interference")
    with pytest.raises(RunStopped) as reference:
        EvaluationSession().evaluate(document, outputs=("complete",), monitor=_Interrupted("untraced_light"))
    session = EvaluationSession()
    evaluation = session.evaluate(document)
    with pytest.raises(RunStopped) as stopped:
        call(evaluation, _Interrupted("untraced_light"), tmp_path)
    record = stopped.value.stopped
    assert record is not None and stopped.value.cause == STOP_CALLER == reference.value.cause
    assert record.stop.phase == "untraced_light" == reference.value.stopped.stop.phase
    assert record.engine_input_hash == evaluation.engine_input_hash == reference.value.stopped.engine_input_hash
    assert not (tmp_path / "run" / "manifest.json").exists()
    # (the session stays usable: the evaluation that follows continues and equals a fresh one)
    continued = evaluation.extend(outputs=("plane_bounds",))
    fresh = EvaluationSession().evaluate(document, outputs=("readings", "plane_bounds"))
    assert continued.result.manifest.result_hash == fresh.result.manifest.result_hash
    # (without a monitor that records a stop the interrupt is the caller's own)
    class _Raw(_Phases):
        def enter(self, name, total, unit):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), monitored(_Raw()):
        evaluation.extend(outputs=("surface_exposures",))


@pytest.mark.parametrize("call, method, direct", [
    (lambda ev, tmp: ev.run(), "run()", "run_optical_project(project)"),
    (lambda ev, tmp: ev.write_run(tmp / "run"), "write_run(output_dir)", "run_optical_project(project, output_dir=output_dir)"),
], ids=["run", "write_run"])
@pytest.mark.parametrize("excluded", ["beam_pairs", "view", "record", "solid_check"])
def test_a_record_or_run_files_with_an_exclusion_they_need_is_refused(call, method, direct, excluded, tmp_path):
    """run() / write_run() of an evaluation made with exclude= are refused before anything runs, in the words of the
    call that was made: it has neither outputs nor exclude, so the message names the exclusion of the evaluation and the
    two calls that give the complete run."""
    session = EvaluationSession()
    evaluation = session.evaluate(example_project(), outputs=("readings",), exclude=(excluded,))
    before = session.statistics()
    with pytest.raises(EvaluationPlanError) as refused:
        call(evaluation, tmp_path)
    message = refused.value.message
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and refused.value.context["capability"] == excluded
    assert message.startswith(f"{method} makes the complete run, which computes {excluded}, and this evaluation was made "
                              f"with exclude={excluded}:")
    assert message.endswith(f"Evaluate the Project again without exclude and call {method} on that evaluation (in the same "
                            f"session it reuses the Engine products this one left), or make the complete run with {direct}.")
    assert "from outputs" not in message and "do not request" not in message and "requested" not in message
    assert not (tmp_path / "run").exists()
    assert session.statistics() == before  # (nothing ran and nothing was counted: no optical-solid check before the refusal)
    # (both ways out give the run)
    again = session.evaluate(example_project(), outputs=("readings",))
    reference = run_optical_project(example_project())
    assert again.run().result.manifest.result_hash == reference.result.manifest.result_hash
    assert set(again.reused) >= {"scene", "beam_trace", "plane_readings"}


def test_an_extension_keeps_the_exclusions_and_is_refused_before_anything_runs():
    session = EvaluationSession()
    evaluation = session.evaluate(example_project(), outputs=("readings",), exclude=("beam_pairs",))
    before = session.statistics()
    with pytest.raises(EvaluationPlanError) as refused:
        evaluation.extend(outputs=("complete",))
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and refused.value.context["capability"] == "beam_pairs"
    assert "the preset complete includes it" in refused.value.message
    assert session.statistics() == before  # (no optical-solid check was made or counted for the refused extension)
    extended = evaluation.extend(outputs=("power_ledger",))
    assert "power_ledger" in extended.executed and extended.scope.excluded == ("beam_pairs",)


def test_a_record_requested_with_an_exclusion_it_needs_is_refused():
    with pytest.raises(EvaluationPlanError) as direct:
        EvaluationSession().evaluate(example_project(), outputs=("record",), exclude=("untraced_light",))
    assert direct.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and "record requires untraced_light" in direct.value.message


def test_an_excluded_capability_has_one_reason_in_every_statement_of_a_project_evaluation():
    """EXCLUDED wherever the evaluation reports it -- its scope, the Engine result's scope, its results and the JSON reply
    -- also when the exclusion was written as a preset; what was only not asked for is NOT_REQUESTED."""
    from os_optics.local import json_api

    document = example_project_of_kind("two_beam_interference")
    for outputs, exclude, excluded in ((("beam_trace",), ("readings",), {"plane_readings", "model_scope"}),
                                       (("readings",), ("beam_pairs", "view"), {"beam_pairs", "view"}),
                                       (("beam_trace", "solid_check"), ("readings", "record"), {"plane_readings", "model_scope", "record"})):
        evaluation = EvaluationSession().evaluate(document, outputs=outputs, exclude=exclude)
        stated = {row["capability"]: row["reason"] for row in evaluation.describe_scope()["not_computed"]}
        assert {c for c, reason in stated.items() if reason == "EXCLUDED"} == excluded, (outputs, exclude)
        assert set(stated.values()) == {"EXCLUDED", "NOT_REQUESTED"} and evaluation.describe_scope()["excluded"] == list(exclude)
        engine = {n.capability: n.reason for n in evaluation.result.evaluation.not_computed}
        assert engine == {c: stated[c] for c in engine} and engine  # (the Engine capabilities among them, the same reason)
        results = evaluation.results().describe()["evaluation"]["not_computed"]
        assert {row["capability"]: row["reason"] for row in results} == stated
        reply = json_api.evaluate_project(document.payload(), list(outputs), list(exclude))
        assert {row["capability"]: row["reason"] for row in reply["evaluation"]["not_computed"]} == stated


def test_a_narrowed_evaluation_writes_a_run_only_after_computing_the_rest(tmp_path):
    document = example_project()
    evaluation = EvaluationSession().evaluate(document)
    run = evaluation.write_run(tmp_path / "run")
    assert (tmp_path / "run" / "manifest.json").is_file() and run.artifacts is not None
    assert evaluation.session.statistics()["executions"]["run_files"] == 1  # (write_run alone writes, and counts it)
    assert run_artifact_files(run) == run_artifact_files(run_optical_project(document))


def test_unknown_ids_and_targets_are_refused_before_anything_runs():
    monitor = _Phases()
    with pytest.raises(EvaluationPlanError) as unknown, monitored(monitor):
        EvaluationSession().evaluate(example_project(), outputs=("no_such",))
    assert unknown.value.code == "UNKNOWN_CAPABILITY" and monitor.entered == []
    with pytest.raises(EvaluationPlanError) as target:
        EvaluationSession().evaluate(example_project(), targets=("no_such_target",))
    assert target.value.code == "UNKNOWN_TARGET"


def test_an_evaluation_writes_nothing_so_run_files_is_no_output():
    monitor = _Phases()
    with pytest.raises(EvaluationPlanError) as refused, monitored(monitor):
        EvaluationSession().evaluate(example_project(), outputs=("run_files",))
    assert refused.value.code == "UNKNOWN_CAPABILITY" and "Evaluation.write_run(output_dir)" in refused.value.message
    assert monitor.entered == []
    session = EvaluationSession()
    complete = session.evaluate(example_project(), outputs=("complete",))
    assert complete.complete and complete.not_computed() == [] and complete.scope.complete
    assert "run_files" not in complete.scope.executed and "run_files" not in session.statistics()["executions"]


def test_a_record_needs_every_declared_target(tmp_path):
    """record / complete with a selection that leaves a declared target out is refused; run(), write_run() and an
    extension to the record evaluate every declared target, dropping the selection."""
    document = example_project_of_kind("two_beam_interference")
    declared = [t["target_id"] for t in document.payload()["targets"]]
    for outputs in (("record",), ("complete",)):
        with pytest.raises(EvaluationPlanError) as refused:
            EvaluationSession().evaluate(document, outputs=outputs, targets=declared[:1])
        assert refused.value.code == "TARGET_SELECTION_INCOMPLETE" and refused.value.context["targets"] == declared[1:]
    narrowed = EvaluationSession().evaluate(document, targets=declared[:1])
    assert [t.target_id for t in narrowed.result.targets] == declared[:1]
    for whole in (narrowed.extend(outputs=("record",)), narrowed.extend(outputs=("complete",))):
        assert whole.local_validation is not None and whole.result.evaluation is None
        assert sorted(t.target_id for t in whole.local_validation.simulation.target_outcomes) == sorted(declared)
    assert [t.target_id for t in narrowed.extend(outputs=("plane_bounds",)).result.targets] == declared[:1]
    run = narrowed.run()
    assert sorted(t.target_id for t in run.local_validation.simulation.target_outcomes) == sorted(declared)
    written = narrowed.write_run(tmp_path / "run")
    assert run_artifact_files(written) == run_artifact_files(run_optical_project(document))
    # (the builder itself makes no record of a narrowed result)
    with pytest.raises(LocalRunError) as builder:
        local_validation_record(narrowed.project, narrowed.lowered, narrowed.result, scope_issues=narrowed.model_scope_issues)
    assert builder.value.problems[0].code.value == "BINDING_INCONSISTENT"


def test_a_narrowed_result_has_no_record_form():
    """A narrowed evaluation has no persisted or submitted form (a narrowed result is not a complete result): the
    record builder refuses it (above), and the record contract refuses each of its additions, so no reader of a Local
    Validation record, the commercial service among them, takes a narrowed result for a complete one."""
    from pydantic import ValidationError

    from os_optics._vendor.shared_contracts.optical_design import LocalValidationRecordV1

    narrowed = EvaluationSession().evaluate(example_project())  # readings: the plane bounds do not run
    assert narrowed.result.evaluation is not None
    assert {o.quantity_state.value for o in narrowed.result.observations} == {"NOT_COMPUTED"}
    record = EvaluationSession().evaluate(example_project(), outputs=("complete",)).local_validation.model_dump(mode="json")
    LocalValidationRecordV1.model_validate(record)
    assert record["simulation"]["observation_quantities"]

    scoped = copy.deepcopy(record)
    scoped["simulation"]["evaluation"] = narrowed.result.evaluation.model_dump(mode="json")
    with pytest.raises(ValidationError) as refused:
        LocalValidationRecordV1.model_validate(scoped)
    assert [(e["type"], e["loc"]) for e in refused.value.errors()] == [("extra_forbidden", ("simulation", "evaluation"))]

    state = copy.deepcopy(record)
    state["simulation"]["observation_quantities"][0]["quantity_state"] = "NOT_COMPUTED"
    with pytest.raises(ValidationError) as refused:
        LocalValidationRecordV1.model_validate(state)
    assert [(e["type"], e["loc"]) for e in refused.value.errors()] == [
        ("literal_error", ("simulation", "observation_quantities", 0, "quantity_state"))]


def test_write_run_leaves_the_earlier_run_as_it_was_when_it_stops_or_fails(tmp_path, monkeypatch):
    document = example_project().payload()
    moved = _moved(document, "L1", 1.0)
    for name, failure in (("stopped", None), ("failed", RuntimeError("the test fails the run"))):
        out = tmp_path / name
        run_optical_project(document, output_dir=out)
        assert not verify_run_artifacts(out)
        earlier = {p.name: p.read_bytes() for p in out.iterdir()}
        evaluation = EvaluationSession().evaluate(moved)
        if failure is None:
            with pytest.raises(RunStopped) as stopped:
                evaluation.write_run(out, monitor=_Phases(stop_in="untraced_light"))
            assert stopped.value.stopped is not None and (out / "run_stopped.json").is_file()
            assert sorted(p.name for p in out.iterdir()) == sorted([*earlier, "run_stopped.json"])
        else:
            from os_optics.local import evaluate as evaluate_module

            def failing(*args, **kwargs):
                raise failure

            with monkeypatch.context() as patched:
                patched.setattr(evaluate_module.stages, "run_view", failing)
                with pytest.raises(RuntimeError):
                    evaluation.write_run(out)
        # the earlier run -- of the Project before the move -- is byte for byte as it was, and it verifies
        assert {p.name: p.read_bytes() for p in out.iterdir() if p.name != "run_stopped.json"} == earlier
        assert verify_run_artifacts(out) == ()
        assert json.loads(earlier["manifest.json"])["project"]["project_hash"]["digest"] == load_project(document).content_hash()


def test_write_run_is_monitored_as_run_optical_project_is(tmp_path):
    document = example_project()
    # (the estimate before the rest is computed models the run that is written, with its ray data; the readings' trace
    # is reused, so no estimate after a trace follows)
    by_run, by_write = RunMonitor(), RunMonitor()
    run_optical_project(document, output_dir=tmp_path / "a", ray_data="summary", monitor=by_run)
    EvaluationSession().evaluate(document).write_run(tmp_path / "b", ray_data="summary", monitor=by_write)
    assert by_write.estimates[0]["artifact_bytes"] == by_run.estimates[0]["artifact_bytes"]
    assert by_write.estimates[0]["stage"] == by_run.estimates[0]["stage"]
    # (the artifact limit holds while the files are written: a complete evaluation's write stops at it)
    complete = EvaluationSession().evaluate(document, outputs=("complete",))
    with pytest.raises(RunStopped) as stopped:
        complete.write_run(tmp_path / "c", monitor=RunMonitor(max_artifact_bytes=1000, preflight=False))
    assert stopped.value.cause == "ARTIFACT_LIMIT" and not (tmp_path / "c" / "manifest.json").exists()
    assert (tmp_path / "c" / "run_stopped.json").is_file()


def _steep_lens_document(front_edge: str | None = None) -> dict:
    """A lens whose front profile (R 6 mm, clear 5 mm) ends inside its 7 mm outline: no solid over the ring from 6 to 7 mm
    (``SAG_DOMAIN_INSIDE_OUTLINE``, ``SURFACES_INTERSECT`` there). With the front edged at 5 mm it is a valid solid with
    the same traced surfaces."""

    from declared_helpers import declared_component, nbk7, sequence, surface, with_models

    extra = {"edge_radius_mm": front_edge} if front_edge is not None else {}
    model = sequence([surface(1, "0", radius="6", clear="5", **extra), surface(2, "4", clear="5")], ["N-BK7"], edge="7")
    return with_models(project([source(), declared_component("E", "lens", "m", quaternion=(0.5, 0.5, 0.5, 0.5))],
                               [plane("P", (100.0, 0.0, 0.0), radius_mm=30.0)]), [{"model_id": "m", "model": model}], [nbk7()])


def _plane_powers(evaluation) -> list[float]:
    return [a.beam.power_W for o in evaluation.result.observations for a in o.arrivals]


def test_a_requested_solid_check_reports_its_findings_beside_the_readings():
    """An evaluation that asks for ``solid_check`` without the record is not refused for an ERROR finding: it carries the
    report (INVALID, each finding with its zone, the thickness facts of every gap) beside the readings, which are those
    of the same lens with a valid body."""

    from os_optics.local import json_api

    session = EvaluationSession()
    evaluation = session.evaluate(_steep_lens_document(), outputs=("readings", "solid_check"))
    report = evaluation.optical_solids
    assert not report.valid and report.as_dict()["status"] == "INVALID"
    assert [(i.code, i.severity, i.zone) for i in report.issues] == [
        ("SAG_DOMAIN_INSIDE_OUTLINE", "ERROR", "OUTSIDE_CLEAR_APERTURE"), ("SURFACES_INTERSECT", "ERROR", "OUTSIDE_CLEAR_APERTURE")]
    (gap,) = report.elements[0]["gaps"]
    # centre 4 mm; at the 5 mm clear radius the sphere's sag is 6 - sqrt(11): 4 - (6 - sqrt(11)) mm of glass
    assert gap["centre_thickness_mm"] == 4.0 and gap["clear_aperture_min_thickness_mm"] == pytest.approx(4.0 - (6.0 - 11.0 ** 0.5), abs=1e-6)
    powers = _plane_powers(evaluation)
    assert powers and powers == _plane_powers(session.evaluate(_steep_lens_document("5"), outputs=("readings", "solid_check")))
    assert powers == _plane_powers(EvaluationSession().evaluate(_steep_lens_document(), outputs=("readings",)))
    assert "solid_check" in evaluation.scope.executed and EvaluationSession().evaluate(_steep_lens_document()).optical_solids is None
    reply = json_api.evaluate_project(_steep_lens_document(), ["readings", "solid_check"])
    assert reply["ok"] and reply["optical_solids"]["status"] == "INVALID"
    assert [(i["code"], i["zone"]) for i in reply["optical_solids"]["issues"]] == [
        ("SAG_DOMAIN_INSIDE_OUTLINE", "OUTSIDE_CLEAR_APERTURE"), ("SURFACES_INTERSECT", "OUTSIDE_CLEAR_APERTURE")]
    assert "optical_solids" not in json_api.evaluate_project(_steep_lens_document(), ["readings"])


@pytest.mark.parametrize("outputs", [("readings", "record"), ("complete",)])
def test_the_record_and_the_complete_run_are_refused_for_a_body_that_is_no_solid(outputs):
    """The record is made of valid optical bodies only: asked for directly, by extending an evaluation that reported the
    findings, by its ``run()``, or by ``run_optical_project``."""

    codes = ["SAG_DOMAIN_INSIDE_OUTLINE", "SURFACES_INTERSECT"]

    def refused(call) -> None:
        with pytest.raises(LocalRunError) as exc:
            call()
        assert [p.code.value for p in exc.value.problems] == codes and exc.value.optical_solids["status"] == "INVALID"
        assert [p.details["zone"] for p in exc.value.problems] == ["OUTSIDE_CLEAR_APERTURE"] * 2

    session = EvaluationSession()
    refused(lambda: session.evaluate(_steep_lens_document(), outputs=outputs))
    reported = session.evaluate(_steep_lens_document(), outputs=("readings", "solid_check"))
    refused(lambda: reported.extend(outputs))
    refused(reported.run)
    refused(lambda: run_optical_project(_steep_lens_document()))
    # a lowering made without refusing is refused when the record is made of it
    lowered = session.lowering(_steep_lens_document(), check_solids=False)
    refused(lambda: session.evaluation(_steep_lens_document(), outputs, lowered=lowered))


def test_write_run_states_what_a_failure_while_the_run_is_moved_into_place_leaves():
    """``Evaluation.write_run`` writes through the same commit as ``run_optical_project``: its docstring limits "leaves
    output_dir as it found it" to a stop or a failure before the run is moved into place, and states what a failure during
    the move leaves, as ``run_optical_project`` does."""
    from os_optics.local.evaluate import Evaluation

    text = " ".join((Evaluation.write_run.__doc__ or "").split())
    assert ("a run that stops, or fails before the run is moved into place, leaves ``output_dir`` as it found it") in text
    assert ("a failure or Ctrl-C while the run is moved into place (a file held open elsewhere) leaves what a process killed "
            "there leaves -- no ``manifest.json``, so no run that verifies") in text
    assert "a run that stops or fails leaves ``output_dir`` as it found it" not in text


# ---- behaviours of the parallel implementation's tests, on this API: a declared validity band, a view alone, threads ----

def _lens(wavelength_nm: float) -> dict:
    """A 1 mW Gaussian source through an ideal thin lens published for 400-1100 nm onto planes at x=50 and x=100, the source
    at ``wavelength_nm`` (``fixtures/ideal_lens_band.project.json``)."""
    from pathlib import Path

    document = json.loads((Path(__file__).resolve().parent / "fixtures" / "ideal_lens_band.project.json").read_text(encoding="utf-8"))
    document["components"][0]["source_model"]["wavelength_nm"] = wavelength_nm
    return document


def test_a_changed_validity_band_of_a_declared_model_is_read_by_the_next_evaluation():
    """The declared model's band is part of what a session evaluates: widening it clears the finding (and equals a fresh
    evaluation), narrowing it back restores it -- nothing of the earlier scope is kept."""
    from os_optics.local import json_api
    from os_optics.local.json_api import _evaluation_reply

    document = _lens(1200.0)
    document["targets"] = [{"target_id": "spot", "metric": "spot_rms_mm", "observation_id": "early", "maximum": 1.0, "unit": "mm"}]
    outputs = ("readings", "requirements")
    session = EvaluationSession()

    def assessed(reply: dict) -> dict[str, str]:
        return {t["target_id"]: t["assessment"] for t in reply["targets"]}

    before = _evaluation_reply(session.evaluate(document, outputs=outputs))
    assert assessed(before) == {"spot": "INDETERMINATE"}
    assert {i["code"] for i in before["model_scope_issues"]} == {"MODEL_WAVELENGTH_OUTSIDE_BAND"}
    widened = copy.deepcopy(document)
    widened["optical_models"][0]["model"]["valid_band_nm"]["max_nm"] = "1500"
    after = _evaluation_reply(session.evaluate(widened, outputs=outputs))
    assert assessed(after) == {"spot": "MET"} and after["model_scope_issues"] == []
    fresh = json_api.evaluate_project(widened, list(outputs))
    assert after["targets"] == fresh["targets"] and after["identity"] == fresh["identity"]
    again = _evaluation_reply(session.evaluate(document, outputs=outputs))
    assert assessed(again) == {"spot": "INDETERMINATE"} and again["identity"] == before["identity"]


def test_a_view_alone_assesses_no_target_and_draws_the_complete_runs_beams(monkeypatch):
    """``outputs=("view",)`` builds the scene without the targets, the plane bounds or the record, and draws what the complete
    run draws: the scenes differ only in ``engine_run``, whose result hash names the narrower result (the trace hash is the
    same)."""
    import os_optics.validation.pipeline as pipeline

    complete = run_optical_project(example_project())
    # (the pipeline calls the name it imported: a rename fails here instead of skipping the guard)
    monkeypatch.setattr(pipeline, "assess_targets", lambda *a, **k: pytest.fail("a view assessed targets"))
    ev = EvaluationSession().evaluate(example_project().payload(), outputs=("view",))
    assert ev.optical_view is not None and ev.local_validation is None and ev.result.targets == ()
    assert {"plane_bounds", "requirements", "record"} <= set(ev.scope.not_executed)
    want, got = complete.optical_view.payload(), ev.optical_view.payload()
    assert got["engine_run"]["trace_hash"] == want["engine_run"]["trace_hash"]
    assert got["engine_run"]["result_hash"] != want["engine_run"]["result_hash"]
    assert {k: v for k, v in got.items() if k != "engine_run"} == {k: v for k, v in want.items() if k != "engine_run"}


def test_a_cancelled_evaluation_in_one_thread_leaves_another_threads_evaluation_untouched():
    """The run monitor is the thread's own (a context variable): cancelling one evaluation does not stop another running at
    the same time, whose result equals a fresh evaluation's."""
    from concurrent.futures import ThreadPoolExecutor

    stopped = RunMonitor()
    stopped.cancel()
    document = _lens(532.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cancelled = pool.submit(evaluate_project, document, ("readings",), (), None, stopped)
        healthy = pool.submit(evaluate_project, document, ("readings",))
        with pytest.raises(RunStopped):
            cancelled.result()
        result = healthy.result()
    assert result.result.manifest.result_hash == evaluate_project(document, ("readings",)).result.manifest.result_hash
