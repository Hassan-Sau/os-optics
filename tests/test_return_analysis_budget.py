"""The return-light analysis runs only where a bound reads it, and its closed-return screen has a work budget.

Light the trace leaves out as negligible is bounded through the returns of the surfaces that may reflect it
(``validation/resonance.py``, ``validation/closed_returns.py``). That analysis (a) runs per wavelength on the first read of
a bound that needs it -- a run whose lines leave nothing out never runs it, and reading it later or not at all changes no
number -- and (b) stops its closed-return screen at ``NumericalConfig.max_return_analysis_work`` cone tests
(``ComputeBudgetV1.max_return_analysis_work``). A screen that stops there does not know whether the left-out light may
circulate, so every upper bound that reads it is withheld (no bound, the quantity UNKNOWN, a dependent target
INDETERMINATE), the run is PARTIAL with the budget named in its budget usage, and one RETURN_ANALYSIS_BUDGET issue
(UNKNOWN) names the planes and surfaces. Never the open-path count, and never a target MET on a withheld bound. The limit
is the run's: its wavelengths share it (``test_closed_return_search``).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local.lowering import pure_optical_engine_configuration
from os_optics.validation import closed_returns as closed_module
from os_optics.validation import resonance
from os_optics.validation.return_work import PAIR_SCREEN_WORK, ReturnWork, WorkCounter

from helpers import run
from test_closed_returns import faint_ring
from test_return_pairs import LAM, _coat, _constant, _scene, _surface

BUDGET = "max_return_analysis_work"


def three_mirrors(*, plane: bool = False) -> ProjectSnapshot:
    """The faint ring with its third mirror moved off the others' cones: three reflectors, no closed return
    (``test_closed_returns.test_three_reflectors_do_not_alone_make_a_closed_path``), so the exposure of ``B/face`` has a
    supported upper bound once the screen has finished -- and, with ``plane``, a transparent plane beside the ring with a
    power maximum on it."""
    project = faint_ring()
    parts = list(project.components)
    parts[3] = _coat(B.mirror("C", position=(50, 3000, 0), normal=(0, 0, 1), clear_radius_mm=3), _constant(1 - 1e-6))
    if not plane:
        return project.model_copy(update={"components": tuple(parts)})
    side = B.observation_plane("side", position=(50, -40, 0), normal=(0, 1, 0), radius_mm=10)
    maximum = Target(target_id="side-power", observation_id="side", metric="power_W", maximum=1e-3)
    return project.model_copy(update={"components": tuple(parts), "observations": (side,), "targets": (maximum,)})


def _issue(result):
    found = [i for i in result.issues if i.code.value == "RETURN_ANALYSIS_BUDGET"]
    assert len(found) <= 1
    return found[0] if found else None


def test_work_is_deterministic_and_stops_exactly_at_its_limit():
    scene, library = _scene(faint_ring())
    faces = [resonance._face(i, s) for i, s in enumerate(scene.surfaces)]

    def screen(limit):
        counter = WorkCounter(limit)
        found = closed_module.closed_returns(resonance._Scene(scene, library, LAM, faces, None, work=counter))
        return found, counter.record()

    (cycles, series), work = screen(None)
    assert cycles and work.steps > 0 and not work.exhausted and work.limit is None
    assert screen(None) == ((cycles, series), work)
    # (a limit of exactly the work done finishes; one less stops there, with no series)
    assert screen(work.steps) == ((cycles, series), ReturnWork(work.steps, work.steps))
    (_, stopped_series), stopped = screen(work.steps - 1)
    assert stopped == ReturnWork(work.steps - 1, work.steps - 1, True, "closed_returns") and stopped_series is None
    # the whole analysis: every pair examined costs PAIR_SCREEN_WORK units, then the closed-return screen
    whole = resonance.ReturnAnalysis(scene, library, [LAM])[LAM]
    assert whole.work.steps >= work.steps + PAIR_SCREEN_WORK
    assert resonance.ReturnAnalysis(scene, library, [LAM], work_limit=whole.work.steps)[LAM] ==         resonance.ReturnGains(**{**whole.__dict__, "work": ReturnWork(whole.work.steps, whole.work.steps)})


def test_an_exhausted_analysis_bounds_nothing_and_never_falls_back_to_the_open_path_count():
    scene, library = _scene(three_mirrors())
    whole = resonance.ReturnAnalysis(scene, library, [LAM])[LAM]
    assert whole.screened and not whole.closed_cycles and math.isfinite(whole.crossing_field("B/face"))
    # stopped in the closed-return screen: the pairs' factors stand, the open paths' counts bound nothing
    closed = resonance.ReturnAnalysis(scene, library, [LAM], work_limit=whole.work.steps - 1)[LAM]
    assert closed.work.exhausted and closed.work.stage == "closed_returns"
    assert closed.crossing_field("B/face") == math.inf and closed.crossings("B/face") == math.inf
    assert closed.factor("B/face") == whole.factor("B/face") and closed.pairs == whole.pairs
    assert closed.withheld_by_work == {"B/face"}
    # stopped in the pair screen: nothing is known
    cut = resonance.ReturnAnalysis(scene, library, [LAM], work_limit=1)[LAM]
    assert cut.work == ReturnWork(0, 1, True, "pairs") and not cut.pairs
    assert cut.factor("B/face") == cut.power_factor("B/face") == cut.exposure_factor("B/face") == math.inf
    assert cut.crossing_field("P") == cut.crossings(None) == math.inf
    assert cut.withheld_by_work == {"B/face", "P", ""}


def test_return_analysis_is_read_on_demand_and_only_where_light_was_left_out(monkeypatch):
    analysed = []
    original = resonance._gains_at

    def counting(*args, **kwargs):
        analysed.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(resonance, "_gains_at", counting)
    # one laser straight onto a plane: every line traced to its end, nothing left out, no analysis
    lone = ProjectSnapshot(
        components=(B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0, 0, 0), direction=(1, 0, 0)),),
        observations=(B.observation_plane("P", position=(100, 0, 0), normal=(1, 0, 0), radius_mm=5),))
    result = run(lone)
    assert analysed == [] and result.observations[0].quantity_state.value == "KNOWN"
    # the ring leaves light out at its wavelength: analysed once, however many bounds read it
    run(faint_ring())
    assert analysed == [LAM]


def test_membership_and_length_do_not_analyse():
    scene, library = _scene(faint_ring())
    analysis = resonance.ReturnAnalysis(scene, library, [LAM, LAM, 532.0])
    assert len(analysis) == 2 and LAM in analysis and 633.0 not in analysis and list(analysis) == [532.0, LAM]
    assert analysis.analysed == ()
    with pytest.raises(KeyError):
        analysis[633.0]
    assert analysis.get(LAM) is analysis[LAM] and [g.wavelength_nm for g in analysis.analysed] == [LAM]


def test_default_budget_keeps_the_bound_and_reports_no_return_analysis_budget():
    result = run(three_mirrors(plane=True))
    surface = _surface(result, "B/face")
    assert surface.ratings[0].exposure_upper_bound is not None
    assert _issue(result) is None
    budgets = result.outcome.execution.budgets
    assert budgets.limits[BUDGET] == NumericalConfig().max_return_analysis_work and BUDGET not in budgets.exhausted
    assert result.outcome.execution.outcome.value == "COMPLETED"


def _work_of(project: ProjectSnapshot) -> int:
    scene, library = _scene(project)
    return resonance.ReturnAnalysis(scene, library, [LAM])[LAM].work.steps


@pytest.mark.parametrize("stage", ["pairs", "closed_returns"])
def test_exhausted_budget_withholds_the_bounds_it_reads_and_leaves_the_execution_completed(stage):
    complete = run(three_mirrors(plane=True))
    limit = 1 if stage == "pairs" else _work_of(three_mirrors(plane=True)) - 1
    result = run(three_mirrors(plane=True), numerical=NumericalConfig(max_return_analysis_work=limit))
    surface = _surface(result, "B/face")
    assert surface.ratings[0].exposure_upper_bound is None and surface.ratings[0].exposure_value is None
    assert surface.rating_status != "WITHIN_RATING"
    issue = _issue(result)
    # (a notice: the analysis is auxiliary. What it withholds stays unknown on the quantities themselves -- the surface's
    # rating status above, the plane's null upper bound below -- and the run's status takes nothing from the notice)
    assert issue is not None and issue.status.value == "NOT_APPLICABLE"
    assert surface.rating_status == "UNKNOWN" or surface.ratings[0].status != "WITHIN_RATING"
    context = issue.numerical_context
    assert context["limit"] == limit and context["wavelength_nm"] == [LAM] and context["stage"] == [stage]
    assert context["work_done"] == [0 if stage == "pairs" else limit]
    assert "B/face" in context["affected_surface_ids"]
    execution = result.outcome.execution
    # (the primary trace exhausted nothing: COMPLETED, with the auxiliary analysis's budget in the record)
    assert execution.outcome.value == "COMPLETED" and execution.termination.value == "COMPLETED" and execution.complete
    assert execution.budgets.exhausted == (BUDGET,)
    assert execution.budgets.limits[BUDGET] == limit
    assert execution.budgets.exhausted_by_phase == {"return_analysis": (BUDGET,)}
    # the side plane, which only left-out light may cross: once the screen finished, that light's bound is within the
    # line's negligible level -- the stated truncation of a known 0 W, the maximum graded on it; without the screen
    # nothing bounds its returns, so more than that level may arrive: the power is not known and the maximum undecided
    complete_plane, plane = complete.observations[0], result.observations[0]
    assert context["affected_observation_ids"] == ["side"] and execution.budgets.affected_observation_ids == ("side",)
    assert complete_plane.quantity_state.value == "KNOWN_ZERO" and complete_plane.power_upper_bound_W is not None
    assert complete.targets[0].assessment.value == "MET"
    assert complete.targets[0].value_basis["value"].omitted[0].upper == pytest.approx(complete_plane.power_upper_bound_W, rel=1e-12)
    assert plane.power_upper_bound_W is None and "power_upper_bound_W" in plane.model_fields_set
    assert plane.quantity_state.value == "UNKNOWN" and plane.total_power_W is None
    assert result.targets[0].assessment.value == "INDETERMINATE" and result.targets[0].value is None
    # what does not read the screen keeps its value: the traced light and the power budget
    assert result.paths == complete.paths and result.power_budget == complete.power_budget


def test_a_project_budget_reaches_the_engine_and_absent_keeps_the_default():
    assert pure_optical_engine_configuration({BUDGET: 7}).numerical.max_return_analysis_work == 7
    assert pure_optical_engine_configuration(None).numerical.max_return_analysis_work == NumericalConfig().max_return_analysis_work


BUDGET_CAUSE = "the return-light analysis reached its work limit"
PAIR_CAUSE = "two surfaces facing each other"


@pytest.mark.parametrize("stage", ["pairs", "closed_returns"])
def test_a_withheld_bound_names_the_exhausted_budget_not_a_cavity(stage):
    # Every message of a quantity the exhausted analysis leaves unbounded names its real cause, the work limit
    # (RETURN_ANALYSIS_BUDGET): the plane's power and coherent-power targets, and a rating condition read against that
    # light. Here no pair is known to hold the light (the complete run bounds it), so "two surfaces facing each other"
    # would be false.
    from os_optics.contracts import PowerRating

    project = three_mirrors(plane=True)
    parts = list(project.components)
    parts[2] = parts[2].model_copy(update={"power_ratings": (PowerRating(
        regime="CW", value=100.0, unit="W/cm2", wavelength_nm=LAM, status="DECLARED", basis="verifier fixture rating, not product data",
        polarization="LINEAR", polarization_condition="linear"),)})
    targets = (Target(target_id="side-power", observation_id="side", metric="power_W", maximum=1e-3),
               Target(target_id="side-coh", observation_id="side", metric="coherent_power_W", maximum=1e-3))
    project = project.model_copy(update={"components": tuple(parts), "targets": targets})
    limit = 1 if stage == "pairs" else _work_of(three_mirrors(plane=True)) - 1
    result = run(project, numerical=NumericalConfig(max_return_analysis_work=limit))
    assert _issue(result) is not None
    for target in result.targets:
        assert target.assessment.value == "INDETERMINATE"
        assert BUDGET_CAUSE in target.message and "RETURN_ANALYSIS_BUDGET" in target.message, target.message
        assert PAIR_CAUSE not in target.message, target.message
        # (its codes name the same cause: the faint light is not bounded, so not FAINT_LIGHT_BOUNDED)
        assert "RETURN_ANALYSIS_BUDGET" in target.issue_codes and "FAINT_LIGHT_BOUNDED" not in target.issue_codes, target.issue_codes
    rating = _surface(result, "B/face").ratings[0]
    texts = [c.message for c in rating.conditions] + list(rating.unverified_conditions)
    assert any("RETURN_ANALYSIS_BUDGET" in text for text in texts), texts
    assert not any(PAIR_CAUSE in text for text in texts), texts
    # the complete run bounds the same light: no unbounded wording at all
    complete = run(project)
    assert all(t.assessment.value == "MET" for t in complete.targets)
    assert all("FAINT_LIGHT_BOUNDED" in t.issue_codes for t in complete.targets)
    assert not any(BUDGET_CAUSE in (t.message or "") for t in complete.targets)


def _facet_rod(sides: int):
    """A glass rod of ``sides`` flat side faces and two flat caps (``sides + 2`` planar patches, every one refracting)."""
    from facet_helpers import extruded, glass, part, request, solid

    polygon = [(5.0 * math.cos(2 * math.pi * k / sides), 5.0 * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices, triangles, _faces = extruded(polygon, 3.0)
    project = request(part("R", solid("rod", vertices, triangles)), materials=(glass(),)).project
    return _scene(project)


def test_the_pair_screen_of_a_solid_of_many_facets_costs_work_by_the_facets_each_pair_considers(monkeypatch):
    # review M3: every pair of a closed solid's facets has the solid's other refracting facets between it, so a pair
    # costs more than PAIR_SCREEN_WORK; charging it as PAIR_SCREEN_WORK let a 191-facet solid spend 2 minutes within the
    # default limit. Each pair is charged its cones through the facets it considers (pair_cone_work) and each class of
    # parallel facets the legs and images its momenta test (momentum_work).
    from os_optics.validation import return_work

    sides = 24
    scene, library = _facet_rod(sides)
    spent: list[tuple[str, int]] = []
    cone, momentum = return_work.pair_cone_work, return_work.momentum_work
    monkeypatch.setattr(resonance, "pair_cone_work", lambda n: (spent.append(("cones", n)), cone(n))[1])
    monkeypatch.setattr(resonance, "momentum_work", lambda n: (spent.append(("momenta", n)), momentum(n))[1])
    whole = resonance.ReturnAnalysis(scene, library, [LAM])[LAM]
    pairs = (sides + 2) * (sides + 1) // 2
    assert whole.screened and len(whole.pairs) == pairs
    cones = [n for kind, n in spent if kind == "cones"]
    assert len(cones) == pairs and min(cones) >= sides // 2  # (each pair considers at least half the other facets)
    assert [n for kind, n in spent if kind == "momenta"]  # (opposite side faces are parallel: their classes are charged)
    charged = pairs * PAIR_SCREEN_WORK + sum(cone(n) for n in cones) + sum(momentum(n) for kind, n in spent if kind == "momenta")
    assert whole.work.steps >= charged
    # at PAIR_SCREEN_WORK a pair the screen no longer finishes: it stops in the pair stage and bounds nothing
    cut = resonance.ReturnAnalysis(scene, library, [LAM], work_limit=pairs * PAIR_SCREEN_WORK)[LAM]
    assert cut.work.exhausted and cut.work.stage == "pairs" and not cut.pairs
    assert math.isinf(cut.factor("R/rod.t0"))
