"""Regression: a faintly fed ring must not inherit an open-path bound."""

import math

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot

from helpers import run
from test_return_pairs import _coat, _constant, _gains, _rating, _ring, _surface, _targets, LAM


def faint_ring(corners=3, rating=100.0):
    if corners == 3:
        p = _ring()
        components = list(p.components)
    else:
        points = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]

        def direction(a, b):
            length = math.dist(a, b)
            return tuple((y - x) / length for x, y in zip(a, b))

        components = [B.laser("S", wavelength_nm=LAM, power_W=1, waist_radius_mm=.5,
                              position=(-400, 0, 0), direction=(1, 0, 0))]
        for i, point in enumerate(points):
            mirror = B.mirror_for_reflection(chr(65 + i), position=point,
                                            d_in=direction(points[i - 1], point),
                                            d_out=direction(point, points[(i + 1) % 4]), clear_radius_mm=3)
            components.append(_coat(mirror, _constant(1 - 1e-6), both=i == 0))
    components[1] = _coat(components[1], _constant(1 - 1e-12), both=True)
    components[2] = components[2].model_copy(update={"power_ratings": (_rating(rating),)})
    return ProjectSnapshot(components=tuple(components), targets=_targets("B"))


@pytest.mark.parametrize("corners", [3, 4])
def test_faint_ring_withholds_unsupported_bound_and_names_its_cause(corners):
    result = run(faint_ring(corners))
    surface = _surface(result, "B/face")
    assert surface.ratings[0].exposure_upper_bound is None
    assert surface.ratings[0].exposure_value is None
    assert surface.rating_status != "WITHIN_RATING"
    assert result.targets[0].assessment.value == "INDETERMINATE"
    assert "EXPOSURE_BOUND_WITHHELD" in surface.reason_codes and "OUT_OF_MODEL_SCOPE" not in surface.reason_codes
    # one notice for the run (not an issue per surface): it names the return, every surface it leaves without a bound,
    # where closed returns can be at all (the screened group) and the round gain bound of the trail found
    assert not [i for i in result.issues if i.code.value == "OUT_OF_MODEL_SCOPE"]
    (cause,) = [i for i in result.issues if i.numerical_context.get("cause") == "CLOSED_RETURN_NOT_BOUNDED"
                and i.numerical_context["quantity"] == "exposure_upper_bound"]
    context = cause.numerical_context
    assert (cause.code.value, cause.status.value, cause.location) == ("EXPOSURE_BOUND_WITHHELD", "NOT_APPLICABLE", "exposures")
    assert any("B/face" in cycle for cycle in context["closed_return_surfaces"])
    assert context["surface_count"] == len(result.exposures.surfaces) and set(context["surface_ids"]) == {
        s.surface_id for s in result.exposures.surfaces}
    assert "B" in context["element_ids"] and cause.affected_targets == (result.targets[0].target_id,)
    # (the one screened group: the ring's mirrors -- closed returns of this kind run only among them)
    assert [sorted(group.split(", ")) for group in context["closed_return_candidates"]] == [[f"{m}/face" for m in "ABCD"[:corners]]]
    # (the trail's round gain bound: the product of its mirrors' amplitude reflectances, sqrt(R) each)
    assert context["closed_return_gain"] == pytest.approx(math.sqrt((1 - 1e-12) * (1 - 1e-6) ** (corners - 1)), rel=1e-12)
    # (where other closed returns can run is a fact of the message; no step inside the tool obtains the bound: no action)
    assert "runs only among the surfaces" in cause.message and "run again" not in cause.message
    assert cause.suggested_action is None
    # An unavailable exposure bound does not turn a completed execution into a failure.
    assert result.outcome.execution.outcome.value == "COMPLETED"
    assert result.paths and result.power_budget is not None


def test_ring_result_does_not_depend_on_optional_rating_target():
    project = faint_ring()
    with_target = run(project)
    without_target = run(project.model_copy(update={"targets": ()}))
    assert with_target.exposures == without_target.exposures
    assert with_target.outcome.execution == without_target.outcome.execution
    # Changing only an optional threshold cannot restore the unsupported quantity.
    changed = project.model_copy(update={"targets": tuple(t.model_copy(update={"maximum": 1e20})
                                                        for t in project.targets)})
    assert run(changed).exposures == with_target.exposures


def test_ring_with_resource_exhaustion_stays_unavailable():
    result = run(faint_ring(), numerical=NumericalConfig(max_trace_steps=1))
    assert result.outcome.execution.outcome.value == "PARTIAL"
    surface = _surface(result, "B/face")
    assert surface.ratings[0].exposure_upper_bound is None
    assert result.targets[0].assessment.value == "INDETERMINATE"


def test_three_reflectors_do_not_alone_make_a_closed_path():
    project = faint_ring()
    # Move the third mirror off the aperture cones of both other mirrors.
    parts = list(project.components)
    parts[3] = _coat(B.mirror("C", position=(50, 3000, 0), normal=(0, 0, 1), clear_radius_mm=3),
                     _constant(1 - 1e-6))
    project = project.model_copy(update={"components": tuple(parts)})
    assert not _gains(project).closed_cycles
    surface = _surface(run(project), "B/face")
    assert surface.ratings[0].exposure_upper_bound is not None
    assert "OUT_OF_MODEL_SCOPE" not in surface.reason_codes


def test_ordinary_weak_parallel_window_stack_has_all_orders_bound():
    source = B.laser("S", wavelength_nm=LAM, power_W=1, waist_radius_mm=.5,
                     position=(-100, 0, 0), direction=(1, 0, 0))
    windows = tuple(B.window(f"W{i}", thickness_mm=3, clear_radius_mm=12.7, material="N-BK7",
                            position=(i * 30, 0, 0)) for i in range(2))
    project = ProjectSnapshot(components=(source, *windows))
    gains = _gains(project)
    assert gains.closed_cycles
    assert gains.closed_series is not None and math.isfinite(gains.closed_series)
    assert math.isfinite(gains.crossing_field("W1/front"))


def test_traced_power_is_retained_when_only_its_exposure_bound_is_unavailable():
    result = run(faint_ring())
    coupler = _surface(result, "A/face")
    assert coupler.hits and coupler.incident_power_W > .99
    assert result.power_budget.total_input_W == pytest.approx(1)


def test_folded_window_return_keeps_both_passes_of_each_fold_mirror():
    source = B.laser("S", wavelength_nm=LAM, power_W=1, waist_radius_mm=.5,
                     position=(-100, 0, 0), direction=(1, 0, 0))
    first = B.window("W1", thickness_mm=3, clear_radius_mm=12.7, material="N-BK7", position=(0, 0, 0))
    a = _coat(B.mirror_for_reflection("M1", position=(100, 0, 0), d_in=(1, 0, 0),
                                     d_out=(0, 0, 1), clear_radius_mm=12.7), _constant(.995))
    b = _coat(B.mirror_for_reflection("M2", position=(100, 0, 100), d_in=(0, 0, 1),
                                     d_out=(1, 0, 0), clear_radius_mm=12.7), _constant(.995))
    last = B.window("W2", thickness_mm=3, clear_radius_mm=12.7, material="N-BK7", position=(150, 0, 100))
    gains = _gains(ProjectSnapshot(components=(source, first, a, b, last)))
    assert gains.closed_cycles and gains.closed_series is None
    assert any("M1/face" in cycle and "M2/face" in cycle for cycle in gains.closed_cycles)
    assert not math.isfinite(gains.crossing_field("W2/front"))


def test_auxiliary_ring_interaction_limit_is_recorded_without_double_counting_and_the_execution_completes():
    project = faint_ring()
    parts = list(project.components)
    parts[1] = _coat(parts[1], _constant(.99), both=True)
    project = project.model_copy(update={"components": tuple(parts)})
    result = run(project, numerical=NumericalConfig(max_interactions_per_path=5))
    execution = result.outcome.execution
    # (only the probes of untraced light met the limit: the primary trace completed)
    assert execution.outcome.value == "COMPLETED" and execution.complete
    assert execution.termination.value == "COMPLETED"
    assert execution.budgets.exhausted == ("max_interactions_per_path",)
    limits = [i for i in result.issues if i.code.value == "MAX_INTERACTIONS"]
    assert limits and all(i.status.value == "NOT_APPLICABLE" and i.numerical_context["phase"] == "untraced_light" for i in limits)
    assert execution.budgets.exhausted_by_phase == {"untraced_light": ("max_interactions_per_path",)}
    assert execution.budgets.affected_source_ids == ("S",)
    assert execution.budgets.affected_path_ids
    # Probe light has already left the primary trace as a physical coating transmission.
    # Its subsequent truncation is diagnostic, not a second addition to primary omitted power.
    assert execution.budgets.untraced_power_W is None
    limited = [i for i in result.issues if i.code.value == "MAX_INTERACTIONS"]
    assert limited and all(i.numerical_context["phase"] == "untraced_light" for i in limited)
    assert all(i.numerical_context["untraced_power_W"] > 0 for i in limited)
    assert set(execution.budgets.affected_path_ids) == {
        path for issue in limited for path in issue.numerical_context["affected_path_ids"]}
    assert all("MAX_INTERACTIONS" in _surface(result, f"{name}/face").reason_codes for name in "ABC")
    assert result.power_budget.total_input_W == pytest.approx(1)


def test_a_closed_trail_of_gain_below_1_reports_its_series_beside_the_withheld_count():
    """A leaky ring: the coupler A (R 1 - 1e-12) and two mirrors of R 0.25 (amplitude 0.5 each). The search finds the one
    closed trail; its round gain bound is the product of the three amplitude reflectances, G = 0.25 (closed form). No
    series over every closed return of the group is proven, so the open paths' counts stay withheld (inf) and no surface
    gets an exposure upper bound. Beside that, the returns around the one trail found are a geometric series,
    1 / (1 - G) = 4 / 3, and the left-out light's peak with the open paths' count times that factor is reported on the
    rated face as an ESTIMATE whose entry names the return it does not bound -- in no proven field and no status."""
    project = faint_ring(3)
    weak = _constant(0.25)
    parts = tuple(_coat(c, weak) if c.instance_id in ("B", "C") else c for c in project.components)
    parts = tuple(c.model_copy(update={"power_ratings": (_rating(100.0),)}) if c.instance_id == "B" else c for c in parts)
    project = project.model_copy(update={"components": parts})
    gains = _gains(project)
    gain = math.sqrt(1.0 - 1e-12) * 0.5 * 0.5
    assert gains.closed_cycles and gains.closed_series is None and gains.closed_return_kinds == ("path",)
    assert gains.search.cycle_gain == pytest.approx(gain, rel=1e-12)
    assert gains.nominal_returns_factor() == pytest.approx(1.0 / (1.0 - gain), rel=1e-12) == pytest.approx(4.0 / 3.0, rel=1e-9)
    assert math.isinf(gains.crossing_field("B/face")) and math.isinf(gains.crossings("B/face"))  # (the proven counts: withheld)
    open_paths = gains._open_crossing_field("B/face")
    assert math.isfinite(open_paths) and gains.nominal_crossing_field("B/face") == pytest.approx(open_paths / (1.0 - gain), rel=1e-12)
    result = run(project)
    surface = _surface(result, "B/face")
    (comparison,) = surface.ratings
    assert comparison.exposure_upper_bound is None and surface.rating_status != "WITHIN_RATING"
    assert result.targets[0].assessment.value == "INDETERMINATE"
    (line,) = result.exposures.faint_light
    limit = 2.0 * math.pi * line.in_phase_power_W / (LAM * 1e-6) ** 2 * 100.0  # (the diffraction limit in air, W/cm2)
    assert surface.faint_light_peak_W_per_cm2 == pytest.approx(gains.nominal_crossing_field("B/face") ** 2 * limit, rel=1e-9)
    entry = surface.value_basis["faint_light_peak_W_per_cm2"]
    assert entry.kind.value == "ESTIMATE" and entry.assumed == ()
    (omitted,) = entry.omitted
    assert (omitted.part.value, omitted.code) == ("POSSIBLE_RETURN", "EXPOSURE_BOUND_WITHHELD")
    assert {"A/face", "B/face", "C/face"} <= set(omitted.at)
    assert "faint_light_return_factor" not in surface.value_basis  # (no pair: nothing estimated there)


def test_a_closed_trail_that_keeps_all_its_light_reports_no_series():
    """The ring of the reference case (round gain bound within 1e-6 of 1): the trail's series 1 / (1 - G) is about 7e5
    and is still reported, with its entry; a trail whose gain bound is 1 has none -- the quantitative answer of a lossless
    ring is a cavity's, not a series."""
    lossless = faint_ring(3)
    one = _constant(1.0)
    parts = tuple(_coat(c, one, both=c.instance_id == "A") if c.instance_id in ("A", "B", "C") else c for c in lossless.components)
    gains = _gains(lossless.model_copy(update={"components": parts}))
    if gains.closed_cycles and gains.search.cycle_gain is not None:
        assert gains.search.cycle_gain == 1.0 and gains.nominal_returns_factor() is None and gains.nominal_crossing_field("B/face") is None
    faint = _gains(faint_ring(3))
    assert faint.search.cycle_gain < 1.0 and faint.nominal_returns_factor() == pytest.approx(1.0 / (1.0 - faint.search.cycle_gain), rel=1e-12)
