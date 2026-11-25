"""The capability table and the evaluation plan (``os_optics.evaluation``).

The table is consistent (known ids, prerequisites before what requires them, phases and controls that exist), every target
metric names the capabilities it reads, and ``plan_evaluation`` adds what a selection requires, names what it added and
refuses what an exclusion makes impossible.
"""

from __future__ import annotations

import pytest

from os_optics import EngineConfiguration, NumericalConfig
from os_optics.contracts import Target
from os_optics.evaluation import (
    ANALYSES,
    CAPABILITIES,
    PRESETS,
    EvaluationPlanError,
    capability,
    capability_ids,
    capability_table,
    enabled_capabilities,
    plan_evaluation,
)
from os_optics.evaluation.capabilities import ENGINE_PRESET_MEMBERS, EXPORTS, LOCAL_PRESET_MEMBERS
from os_optics.run_control import RUN_PHASES
from os_optics.validation.descriptors import metric_descriptors
from os_optics.validation.metrics import ALL_SUPPORTED, METRIC_REQUIRES

from fixture_scenes import FIXTURES

ENGINE_IDS = ("scene", "beam_trace", "secondary_rays", "plane_readings", "power_ledger", "untraced_light", "plane_bounds",
              "surface_exposures", "beam_pairs", "layout", "control", "requirements")
LOCAL_IDS = ("solid_check", "ray_paths", "model_scope", "record", "view")
READINGS = ("scene", "beam_trace", "secondary_rays", "plane_readings")


def _request(name: str):
    return FIXTURES[name]()


# ----------------------------------------------------------------------------------------------- the table


def test_the_table_holds_the_public_capability_ids_once_each():
    ids = [row.id for row in CAPABILITIES]
    assert len(ids) == len(set(ids))
    assert capability_ids("engine") == ENGINE_IDS
    assert set(capability_ids("local")) == set(LOCAL_IDS)
    assert capability_table() is CAPABILITIES
    assert not set(ids) & {row.id for row in ANALYSES} and not set(ids) & set(PRESETS)
    assert [row.id for row in EXPORTS] == ["run_files"] and not set(ids) & {row.id for row in EXPORTS}
    # (what a Project evaluation runs with the plane readings and the requirements: their model-scope findings)
    assert {row.id: row.local_requires for row in CAPABILITIES if row.local_requires} == {
        "plane_readings": ("model_scope",), "requirements": ("model_scope",)}
    assert capability("beam_trace").requires == ("scene",) and capability("imaging").kind == "analysis"
    with pytest.raises(KeyError):
        capability("no_such_capability")


def test_every_prerequisite_is_a_capability_that_runs_before_what_requires_it():
    """Execution order is the table's order: a prerequisite stands before every row that requires it, so the graph is acyclic."""
    seen: set[str] = set()
    for row in CAPABILITIES:
        for need in (*row.requires, *row.traced_requires, *row.rated_requires):
            assert need in seen, f"{row.id} requires {need}, which is not an earlier row"
        seen.add(row.id)
    for row in ANALYSES:
        assert set(row.requires) <= seen and row.entry_point.startswith("os_optics.")
    for members in (*ENGINE_PRESET_MEMBERS.values(), *LOCAL_PRESET_MEMBERS.values()):
        assert set(members) <= seen
    assert set(ENGINE_PRESET_MEMBERS) == set(LOCAL_PRESET_MEMBERS) == set(PRESETS)


def test_a_row_names_phases_controls_and_request_flags_that_exist():
    controls = set(NumericalConfig.model_fields)
    flags = set(EngineConfiguration.model_fields)
    for row in (*CAPABILITIES, *ANALYSES, *EXPORTS):
        assert row.level in ("engine", "local") and row.kind in ("computation", "recording", "presentation", "analysis", "export")
        assert row.summary and row.method and row.version == "1"
        assert set(row.phases) <= set(RUN_PHASES), row.id
        assert set(row.controls) <= controls, row.id
        assert not row.request_flag or row.request_flag in flags, row.id
    assert all(row.level == "engine" for row in CAPABILITIES if row.request_flag)


def test_every_target_metric_names_the_capabilities_it_reads():
    assert set(METRIC_REQUIRES) == set(ALL_SUPPORTED)
    for metric, requires in METRIC_REQUIRES.items():
        assert set(requires) <= set(ENGINE_IDS), metric
        assert len(requires) == len(set(requires))
    described = {d.name: d for d in metric_descriptors(include_stage_metrics=True)}
    assert set(described) == set(METRIC_REQUIRES)
    for name, descriptor in described.items():
        assert descriptor.requires == METRIC_REQUIRES[name]
    assert METRIC_REQUIRES["spot_rms_mm"] == ("plane_readings",)
    assert METRIC_REQUIRES["power_W"] == METRIC_REQUIRES["coherent_power_W"] == METRIC_REQUIRES["detector_snr"] == ("plane_bounds",)
    assert METRIC_REQUIRES["stokes_s1"] == ("plane_readings", "power_ledger")
    assert METRIC_REQUIRES["fringe_period_nm"] == ("beam_pairs",)
    assert METRIC_REQUIRES["fringe_visibility"] == ("beam_pairs", "plane_bounds")
    assert METRIC_REQUIRES["power_rating_utilization"] == ("surface_exposures", "untraced_light")
    assert METRIC_REQUIRES["shg_power_W"] == ("power_ledger",) and METRIC_REQUIRES["dof_travel_margin_mm"] == ()
    assert METRIC_REQUIRES["layout_min_clearance_mm"] == ("layout",) and METRIC_REQUIRES["numerical_rank"] == ("control",)


# ----------------------------------------------------------------------------------------------- the plan


def test_readings_is_the_plane_readings_with_what_they_require():
    plan = plan_evaluation(("readings",))
    assert plan.requested == ("readings",) and plan.executed == READINGS and not plan.complete
    assert [(a.capability, a.needed_by) for a in plan.added] == [
        ("scene", "beam_trace"), ("beam_trace", "plane_readings"), ("secondary_rays", "plane_readings")]
    assert plan.not_executed == ENGINE_IDS[4:]
    assert plan_evaluation("readings") == plan
    assert plan_evaluation(("plane_readings", "readings", "plane_readings")).requested == ("readings", "plane_readings")


def test_complete_is_every_capability_the_request_enables():
    request = _request("interference_405_500")  # (physics only: no layout checks, no control analysis)
    plan = plan_evaluation(("complete",), request=request)
    assert plan.complete and plan.added == () and plan.executed == plan.complete_scope == enabled_capabilities(request)
    assert "layout" not in plan.executed and "control" not in plan.executed
    assert {t.target_id for t in plan.targets} == {t.target_id for t in request.project.targets} and not plan.unselected_targets
    with_layout = _request("benchmark_standard")
    assert plan_evaluation(("complete",), request=with_layout).executed == tuple(c for c in ENGINE_IDS if c != "control")
    with_control = _request("control_recoverability")
    assert plan_evaluation(("complete",), request=with_control).executed == tuple(c for c in ENGINE_IDS if c != "layout")
    assert plan_evaluation(("complete",)).executed == ENGINE_IDS


def test_a_plus_c_and_b_only_compositions():
    request = _request("interference_405_500")
    a_and_c = plan_evaluation(("plane_readings", "beam_pairs"), exclude=("untraced_light",), request=request)
    assert a_and_c.executed == (*READINGS, "beam_pairs") and a_and_c.excluded == ("untraced_light",)
    assert a_and_c.excluded_capabilities == ("untraced_light",)
    # (a preset in exclude: the plan keeps the caller's words and the capabilities they stand for)
    by_preset = plan_evaluation(("beam_trace",), exclude=("readings",))
    assert by_preset.excluded == ("readings",) and by_preset.excluded_capabilities == ("plane_readings",)
    assert plan_evaluation(("beam_trace",), request=request).executed == ("scene", "beam_trace")
    assert plan_evaluation(("untraced_light",), request=request).executed == ("scene", "beam_trace", "secondary_rays", "untraced_light")
    assert plan_evaluation(("surface_exposures",), request=request).executed == (*READINGS, "power_ledger", "surface_exposures")
    assert plan_evaluation(("plane_bounds",), request=request).executed == (*READINGS, "power_ledger", "untraced_light", "plane_bounds")


def test_layout_alone_reads_the_traced_beams_only_when_the_request_traces():
    traced = _request("layout_example")
    assert plan_evaluation(("layout",), request=traced).executed == ("scene", "beam_trace", "secondary_rays", "layout")
    untraced = traced.model_copy(update={"configuration": traced.configuration.model_copy(update={"run_physics": False})})
    plan = plan_evaluation(("layout",), request=untraced)
    assert plan.executed == ("layout",) and "scene" not in plan.complete_scope


def test_the_requirements_add_what_each_selected_target_reads():
    request = _request("interference_405_500")
    plan = plan_evaluation(("requirements",), request=request)
    assert plan.executed == (*READINGS, "power_ledger", "untraced_light", "plane_bounds", "beam_pairs", "requirements")
    added = {a.capability: a.needed_by for a in plan.added}
    assert added["beam_pairs"] == "target:period" and added["plane_bounds"] == "target:visibility"
    assert added["untraced_light"] == "plane_bounds"
    geometric = plan_evaluation(("readings",), targets=("period", "opd"), request=request)
    assert geometric.executed == (*READINGS, "beam_pairs", "requirements")
    assert {a.capability: a.needed_by for a in geometric.added}["requirements"] == "targets"
    assert [t.target_id for t in geometric.targets] == ["period", "opd"]
    assert geometric.unselected_targets == ("half_angle", "visibility", "coherence", "exposure") and not geometric.complete
    without = plan_evaluation(("readings",), request=request)
    assert without.targets == () and len(without.unselected_targets) == len(request.project.targets)


def test_the_exposures_require_the_untraced_light_of_a_request_that_states_a_power_rating():
    from os_optics.contracts import PowerRating

    assert capability("surface_exposures").rated_requires == ("untraced_light",)
    assert "untraced_light" not in capability("surface_exposures").requires
    unrated = _request("mzi_window")
    assert not any(c.power_ratings for c in unrated.project.components)
    plain = plan_evaluation(("surface_exposures",), request=unrated)
    assert plain.executed == (*READINGS, "power_ledger", "surface_exposures")
    rating = PowerRating(regime="CW", value=100.0, unit="W/cm2", wavelength_nm=633.0, status="DECLARED", basis="test rating",
                         polarization="LINEAR", polarization_condition="linear")
    parts = list(unrated.project.components)
    parts[-1] = parts[-1].model_copy(update={"power_ratings": (rating,)})
    rated = unrated.model_copy(update={"project": unrated.project.model_copy(update={"components": tuple(parts)})})
    plan = plan_evaluation(("surface_exposures",), request=rated)
    assert plan.executed == (*READINGS, "power_ledger", "untraced_light", "surface_exposures")
    assert {a.capability: a.needed_by for a in plan.added}["untraced_light"] == "surface_exposures"
    with pytest.raises(EvaluationPlanError) as refused:
        plan_evaluation(("surface_exposures",), exclude=("untraced_light",), request=rated)
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and refused.value.context["needed_by"] == "surface_exposures"
    # (the same exclusion is a plan for the design without ratings; the table alone plans for a design that may state one)
    assert "untraced_light" not in plan_evaluation(("surface_exposures",), exclude=("untraced_light",), request=unrated).executed
    assert "untraced_light" in plan_evaluation(("surface_exposures",)).executed


def test_a_threshold_never_changes_the_plan():
    request = _request("keplerian_expander")
    moved = tuple(t.model_copy(update={"minimum": None, "maximum": 1e9, "target_value": None, "tolerance_abs": None,
                                       "tolerance_rel": None}) for t in request.project.targets)
    other = request.model_copy(update={"project": request.project.model_copy(update={"targets": moved})})
    assert plan_evaluation(("requirements",), request=other) == plan_evaluation(("requirements",), request=request)


def test_an_exclusion_something_requires_is_refused_with_both_ways_out():
    request = _request("interference_405_500")
    with pytest.raises(EvaluationPlanError) as direct:
        plan_evaluation(("plane_bounds",), exclude=("untraced_light",), request=request)
    assert direct.value.code == "EXCLUDED_CAPABILITY_REQUIRED"
    assert direct.value.context == {"capability": "untraced_light", "needed_by": "plane_bounds", "origin": "plane_bounds"}
    assert direct.value.message == (
        "untraced_light is excluded, and plane_bounds requires untraced_light: the evaluation plan runs nothing that is excluded "
        "and computes nothing in its place. Remove untraced_light from exclude, or do not request plane_bounds.")
    with pytest.raises(EvaluationPlanError) as by_target:
        plan_evaluation(("readings", "requirements"), exclude=("untraced_light",), request=request)
    assert by_target.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and by_target.value.context["origin"] == "target:visibility"
    assert by_target.value.message == (
        "untraced_light is excluded, and plane_bounds requires untraced_light, and target 'visibility' (metric fringe_visibility) "
        "reads plane_bounds: the evaluation plan runs nothing that is excluded and computes nothing in its place. Remove "
        "untraced_light from exclude, or leave target 'visibility' out of the evaluation (select the others with targets, or do "
        "not request requirements).")
    # (the same selection without the target that reads the excluded capability is a plan)
    allowed = plan_evaluation(("readings",), targets=("period", "half_angle", "opd", "coherence"), exclude=("untraced_light",),
                              request=request)
    assert "untraced_light" not in allowed.executed and "plane_bounds" not in allowed.executed
    with pytest.raises(EvaluationPlanError) as both:
        plan_evaluation(("beam_trace",), exclude=("beam_trace",))
    assert both.value.code == "EXCLUDED_CAPABILITY_REQUIRED"
    assert both.value.message == (
        "beam_trace is both requested and excluded: the evaluation plan cannot run it and leave it out. Remove beam_trace from "
        "exclude, or remove it from the requested capabilities.")


RULE = "the evaluation plan runs nothing that is excluded and computes nothing in its place"


def _refusal(*args, **kwargs) -> EvaluationPlanError:
    with pytest.raises(EvaluationPlanError) as refused:
        plan_evaluation(*args, **kwargs)
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED"
    return refused.value


def test_a_refused_exclusion_names_the_preset_the_caller_wrote():
    """A capability that came with a preset is named with that preset, in the request and in the exclusions: the way out
    never names an id the caller did not write. What else of the plan requires the excluded capability is named, and the
    second way out is then one that plans: capabilities that do not require it."""
    request = _request("interference_405_500")
    readers = "targets 'period', 'half_angle', 'opd', 'visibility', 'coherence' and 'exposure' read it"
    without_readers = "(the requirements with targets that do not read it)"
    everything = _refusal(("complete",), exclude=("beam_pairs",), request=request)
    assert everything.context == {"capability": "beam_pairs", "needed_by": "", "origin": "complete"}
    assert everything.message == (f"beam_pairs is excluded, and the preset complete includes it, and {readers}: {RULE}. Remove "
                                  "beam_pairs from exclude, or name capabilities that do not require beam_pairs instead of "
                                  f"complete {without_readers}.")
    # (nothing else of the plan requires the excluded capability: the message is the plain one)
    untargeted = request.model_copy(update={"project": request.project.model_copy(update={"targets": ()})})
    assert _refusal(("complete",), exclude=("beam_pairs",), request=untargeted).message == (
        f"beam_pairs is excluded, and the preset complete includes it: {RULE}. Remove beam_pairs from exclude, or name the "
        "capabilities you want instead of complete.")
    # (the default selection of the local workflow: readings come with the models' applicability, and the plane readings
    # require it, so naming them instead of the preset would be refused in turn)
    scope = _refusal(("readings",), exclude=("model_scope",), request=request, level="local", what="outputs")
    assert scope.message == (f"model_scope is excluded, and the preset readings includes it, and plane_readings requires it: "
                             f"{RULE}. Remove model_scope from exclude, or name capabilities that do not require model_scope "
                             "instead of readings.")
    chain = _refusal(("readings",), exclude=("ray_paths",), request=request, level="local", what="outputs")
    assert chain.context == {"capability": "ray_paths", "needed_by": "model_scope", "origin": "readings"}
    assert chain.message == (f"ray_paths is excluded, and model_scope requires ray_paths, and the preset readings includes "
                             f"model_scope (plane_readings requires ray_paths as well): {RULE}. Remove ray_paths from exclude, "
                             "or name capabilities that do not require ray_paths instead of readings.")
    # (both ways out the messages name can be followed: the capability the message says requires the excluded one is
    # refused, the preset's other capabilities plan)
    for excluded in ("model_scope", "ray_paths"):
        again = _refusal(("plane_readings",), exclude=(excluded,), request=request, level="local", what="outputs")
        assert again.context["capability"] == excluded
        allowed = plan_evaluation(("scene", "beam_trace", "secondary_rays"), exclude=(excluded,), request=request,
                                  level="local", what="outputs")
        assert allowed.executed == ("scene", "beam_trace", "secondary_rays")
        assert plan_evaluation(("readings",), request=request, level="local", what="outputs").runs(excluded)
    trace = _refusal(("readings",), exclude=("beam_trace",), request=request)
    assert trace.message == (f"beam_trace is excluded, and plane_readings requires beam_trace, and the preset readings includes "
                             f"plane_readings (secondary_rays requires beam_trace as well): {RULE}. Remove beam_trace from "
                             "exclude, or name capabilities that do not require beam_trace instead of readings.")
    # (an excluded preset is named as the caller wrote it, with the member that is needed)
    by_preset = _refusal(("complete",), exclude=("readings",), request=request)
    assert by_preset.context == {"capability": "plane_readings", "needed_by": "", "origin": "complete", "excluded_by": "readings"}
    assert by_preset.message == (f"plane_readings is excluded (by the preset readings in exclude), and the preset complete "
                                 f"includes it, and power_ledger, plane_bounds, surface_exposures and beam_pairs require it, and "
                                 f"{readers}: {RULE}. Remove readings from exclude, or name capabilities that do not require "
                                 f"plane_readings instead of complete {without_readers}.")
    whole = _refusal(("readings",), exclude=("complete",), request=request)
    assert whole.context["excluded_by"] == "complete" and whole.context["origin"] == "readings"
    assert whole.message == (f"scene is excluded (by the preset complete in exclude), and beam_trace requires scene, and "
                             f"plane_readings requires beam_trace, and the preset readings includes plane_readings "
                             f"(secondary_rays requires scene as well): {RULE}. "
                             "Remove complete from exclude, or name capabilities that do not require scene instead of readings.")
    # (a Project evaluation's requirements require the model scope whatever the targets: no target is named for them)
    selected = _refusal(("readings",), exclude=("model_scope",), targets=("period",), request=request, level="local",
                        what="outputs")
    assert selected.message == (f"model_scope is excluded, and the preset readings includes it, and plane_readings, beam_pairs "
                                f"and requirements require it: {RULE}. Remove model_scope from exclude, or name capabilities "
                                "that do not require model_scope instead of readings.")
    # (a capability the caller named by its id is the caller's own word, whatever preset also includes it)
    named = _refusal(("readings", "plane_readings"), exclude=("plane_readings",), request=request, what="outputs")
    assert named.context["origin"] == "plane_readings"
    assert named.message == ("plane_readings is both requested and excluded: the evaluation plan cannot run it and leave it out. "
                             "Remove plane_readings from exclude, or remove it from outputs.")
    # (a Project evaluation's plane readings and requirements come with the models' applicability, however the request is
    # spelled: excluding the model scope or the ray paths it reads is refused; an Engine evaluation has no model scope)
    for outputs in (("plane_readings",), ("requirements",)):
        for excluded in ("model_scope", "ray_paths"):
            refused = _refusal(outputs, exclude=(excluded,), request=request, level="local", what="outputs")
            assert refused.context["capability"] == excluded and refused.context["origin"] == outputs[0]
            assert refused.message.endswith(f"Remove {excluded} from exclude, or do not request {outputs[0]}.")
    assert _refusal(("plane_readings",), exclude=("ray_paths",), request=request, level="local", what="outputs").message == (
        f"ray_paths is excluded, and model_scope requires ray_paths, and plane_readings requires model_scope: {RULE}. Remove "
        "ray_paths from exclude, or do not request plane_readings.")
    assert plan_evaluation(("plane_readings",), request=request).executed == READINGS


def test_an_unknown_id_is_reported_under_the_parameter_the_caller_wrote():
    for what in ("requested", "capabilities", "outputs"):
        with pytest.raises(EvaluationPlanError) as found:
            plan_evaluation(("observations",), what=what)
        assert found.value.code == "UNKNOWN_CAPABILITY" and found.value.message.startswith(f"{what}: 'observations' is not a")
    from os_optics import evaluate

    with pytest.raises(EvaluationPlanError) as engine:
        evaluate(_request("interference_405_500"), capabilities=("observations",))
    assert engine.value.message.startswith("capabilities: 'observations' is not a capability id or a preset")


def test_unknown_ids_targets_and_capabilities_the_request_does_not_enable_are_refused():
    request = _request("interference_405_500")
    for bad, words in (("no_such", "not a capability id"), ("record", "local workflow"), ("imaging", "is an analysis")):
        with pytest.raises(EvaluationPlanError) as found:
            plan_evaluation((bad,), request=request)
        assert found.value.code == "UNKNOWN_CAPABILITY" and words in found.value.message and found.value.context == {"capability": bad}
    with pytest.raises(EvaluationPlanError) as excluded:
        plan_evaluation(("readings",), exclude=("no_such",))
    assert excluded.value.code == "UNKNOWN_CAPABILITY" and excluded.value.message.startswith("exclude:")
    with pytest.raises(EvaluationPlanError) as off:
        plan_evaluation(("layout",), request=request)
    assert off.value.code == "CAPABILITY_NOT_ENABLED" and "run_layout is false" in off.value.message
    with pytest.raises(EvaluationPlanError) as target:
        plan_evaluation(("readings",), targets=("nobody",), request=request)
    assert target.value.code == "UNKNOWN_TARGET" and "period" in target.value.message


def test_a_target_on_a_capability_the_request_does_not_enable_is_planned_as_missing_it():
    request = _request("layout_example")
    clearance = Target(target_id="clear", metric="layout_min_clearance_mm", minimum=0.0)
    request = request.model_copy(update={"project": request.project.model_copy(update={"targets": (clearance,)})})
    off = request.model_copy(update={"configuration": request.configuration.model_copy(update={"run_layout": False})})
    plan = plan_evaluation(("complete",), request=off)
    assert "layout" not in plan.executed and plan.complete
    missing = [t for t in plan.targets if t.missing]
    assert missing and all(t.missing == ("layout",) for t in missing)
    assert all(not t.missing for t in plan_evaluation(("complete",), request=request).targets)


def test_no_exported_name_of_the_package_is_also_one_of_its_submodules():
    """A package attribute that is a function for one import order and a submodule for another breaks its callers: the
    table's accessors do not share a name with ``capabilities.py``, ``plan.py`` or ``state.py``."""
    import importlib
    import pkgutil

    import os_optics.evaluation as package

    submodules = {module.name for module in pkgutil.iter_modules(package.__path__)}
    assert submodules == {"capabilities", "plan", "state"} and not submodules & set(package.__all__)
    for name in submodules:
        importlib.import_module(f"os_optics.evaluation.{name}")
    assert callable(package.capability_table) and callable(package.evaluate) and callable(package.plan_evaluation)
    assert package.evaluate is importlib.import_module("os_optics").evaluate


def test_the_local_level_adds_the_local_workflow_to_the_presets():
    request = _request("keplerian_expander")
    complete = plan_evaluation(("complete",), request=request, level="local")
    assert complete.executed[0] == "solid_check" and complete.executed[-4:] == ("ray_paths", "model_scope", "record", "view")
    assert "run_files" not in complete.executed and complete.complete and complete.not_executed == ()
    readings = plan_evaluation(("readings",), request=request, level="local")
    assert readings.executed == (*READINGS, "ray_paths", "model_scope")
    assert plan_evaluation(("plane_readings",), request=request, level="local").executed == readings.executed
    with pytest.raises(EvaluationPlanError) as record:
        plan_evaluation(("record",), exclude=("untraced_light",), request=request, level="local")
    assert record.value.code == "EXCLUDED_CAPABILITY_REQUIRED" and record.value.context["needed_by"] == "record"
    # (the run directory is written by Evaluation.write_run: an evaluation writes nothing, at either level)
    for level in ("local", "engine"):
        with pytest.raises(EvaluationPlanError) as files:
            plan_evaluation(("run_files",), request=request, level=level, what="outputs")
        assert files.value.code == "UNKNOWN_CAPABILITY" and files.value.context == {"capability": "run_files"}
        assert "Evaluation.write_run(output_dir)" in files.value.message and "os-optics project run" in files.value.message
    assert capability("run_files").kind == "export" and "run_files" not in enabled_capabilities(request, "local")


def test_a_project_evaluation_neither_lists_nor_accepts_the_capabilities_of_an_engine_request_only():
    """The layout checks and the control analysis run on an Engine request (os_optics.evaluate); a Project's request runs
    neither, so the local level does not list them and a request for one names where they run."""
    request = _request("benchmark_standard")  # (an Engine request with envelopes: the layout checks are enabled)
    assert "layout" in enabled_capabilities(request) and "layout" not in enabled_capabilities(request, "local")
    assert {"layout", "control"}.isdisjoint(enabled_capabilities(None, "local"))
    for name, flag in (("layout", "run_layout"), ("control", "run_control")):
        for asked in ({"requested": (name,), "what": "outputs"}, {"requested": ("readings",), "exclude": (name,)}):
            with pytest.raises(EvaluationPlanError) as refused:
                plan_evaluation(level="local", **asked)
            assert refused.value.code == "UNKNOWN_CAPABILITY" and refused.value.context == {"capability": name}
            assert f"os_optics.evaluate(request, capabilities=['{name}'])" in refused.value.message and flag in refused.value.message
        with pytest.raises(EvaluationPlanError) as unknown:
            plan_evaluation(("no_such",), level="local", what="outputs")
        assert f", {name}," not in unknown.value.message and "run_files" not in unknown.value.message


def test_an_analysis_named_as_a_capability_says_how_it_is_run():
    """Each analysis row names Python calls that exist, with their arguments, and an os-optics command; the refusal of an
    analysis named as an output repeats both."""
    import importlib
    import inspect

    for row in ANALYSES:
        for call in row.entry_point.split(", or "):
            path, arguments = call.split(" for ")[0].split("(", 1)
            module_name, _, rest = path.partition(".results.") if ".results." in path else path.partition(".local.")
            target = importlib.import_module(f"{module_name}.{'results' if '.results.' in path else 'local'}")
            for part in rest.split("."):
                target = getattr(target, part)
            parameters = inspect.signature(target).parameters
            for argument in arguments.rstrip(")").split(", "):
                assert argument.split("=")[0] in parameters, (row.id, call)
        assert row.command.startswith("os-optics "), row.id
        with pytest.raises(EvaluationPlanError) as refused:
            plan_evaluation((row.id,), level="local", what="outputs")
        message = refused.value.message
        assert message.startswith(f"outputs: {row.id!r} is an analysis, not a capability of the evaluation plan")
        assert row.entry_point in message and row.command in message


def test_a_record_with_a_selection_that_leaves_a_declared_target_out_is_refused():
    """The record is made of a complete evaluation: every capability the request enables (an exclusion is refused) and
    every declared target (a selection that leaves one out is refused the same way, before anything runs)."""
    request = _request("interference_405_500")
    declared = tuple(t.target_id for t in request.project.targets)
    with pytest.raises(EvaluationPlanError) as record:
        plan_evaluation(("record",), targets=("period", "opd"), request=request, level="local")
    assert record.value.code == "TARGET_SELECTION_INCOMPLETE"
    assert record.value.context == {"capability": "record", "needed_by": "", "origin": "record",
                                    "targets": ["half_angle", "visibility", "coherence", "exposure"]}
    assert record.value.message == (
        "record is made of a complete evaluation: it needs every declared target, and the selection of targets leaves out "
        "'half_angle', 'visibility', 'coherence', 'exposure'. Give no selection of targets (every declared target is then "
        "evaluated) or name every declared target, or do not request record.")
    with pytest.raises(EvaluationPlanError) as complete:
        plan_evaluation(("complete",), targets=declared[:-1], request=request, level="local")
    assert complete.value.code == "TARGET_SELECTION_INCOMPLETE" and complete.value.context["origin"] == "complete"
    assert complete.value.message == (
        "record is made of a complete evaluation (the preset complete includes record): it needs every declared target, and "
        "the selection of targets leaves out 'exposure'. Give no selection of targets (every declared target is then "
        "evaluated) or name every declared target, or name the capabilities you want instead of complete.")
    with pytest.raises(EvaluationPlanError) as nothing:
        plan_evaluation(("record",), targets=(), request=request, level="local")
    assert nothing.value.code == "TARGET_SELECTION_INCOMPLETE" and nothing.value.context["targets"] == list(declared)
    # (a selection that names every declared target is planned; so is no selection)
    for outputs in (("record",), ("complete",)):
        for selection in (declared, None):
            plan = plan_evaluation(outputs, targets=selection, request=request, level="local")
            assert plan.unselected_targets == () and plan.runs("record") and len(plan.targets) == len(declared)
    # (an Engine evaluation has no capability made of a complete evaluation: a selection of its complete scope is a plan)
    engine = plan_evaluation(("complete",), targets=("period",), request=request)
    assert not engine.complete and engine.unselected_targets == declared[1:]
