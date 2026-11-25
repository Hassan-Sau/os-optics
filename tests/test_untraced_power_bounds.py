"""Power requirements when part of the light was not traced (F-EV-1).

The traced light at a plane is a lower bound of a summed power; on an absorbing plane the traced light plus the untraced
power is an upper bound (each untraced watt is received at most once), on a transparent plane nothing bounds it from
above. A requirement is decided only where the whole interval agrees: a minimum passes when the lower end meets it and
fails when the upper end misses it, a maximum passes when the upper end meets it and fails when the lower end exceeds it;
anything else is UNKNOWN, never a WARNING. ``coherent_power_W`` uses the field-norm interval
``[(sqrt(P) - sqrt(U))^2, (sqrt(P) + sqrt(U))^2]``. Every number below is computed here from the stated inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from os_optics import NumericalConfig, ValidationRequest, validate
from os_optics.builders import laser, observation_plane
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status
from os_optics.local import json_api, lower_project

from helpers import run

X = (1.0, 0.0, 0.0)
A_W = 1e-3  # traced (source A)
B_W = 0.5e-3  # not launched (max_paths=1): untraced
FIXTURES = Path(__file__).parent / "fixtures" / "composition"


def _power(target_id: str, *, metric: str = "power_W", **limits) -> Target:
    return Target(target_id=target_id, metric=metric, observation_id="det", unit="W", **limits)


def _two_sources(*targets: Target, absorbing: bool = True):
    a = laser("A", wavelength_nm=780.0, power_W=A_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)
    b = laser("B", wavelength_nm=780.0, power_W=B_W, waist_radius_mm=0.5, position=(-100.0, 3.0, 0.0), direction=X)
    det = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=absorbing)
    snapshot = ProjectSnapshot(components=(a, b), observations=(det,), targets=targets)
    return {t.target_id: t for t in run(snapshot, numerical=NumericalConfig(max_paths=1)).targets}


def test_power_w_on_an_absorbing_plane_is_graded_on_traced_to_traced_plus_untraced():
    # the plane receives between 1 mW (A, traced) and 1.5 mW (A + the untraced B)
    t = _two_sources(_power("min_met", minimum=0.9e-3), _power("min_missed", minimum=1.6e-3), _power("min_open", minimum=1.2e-3),
                     _power("max_met", maximum=1.6e-3), _power("max_exceeded", maximum=0.9e-3), _power("max_open", maximum=1.2e-3),
                     _power("band_met", target_value=1.25e-3, tolerance_abs=0.3e-3),
                     _power("band_open", target_value=1.25e-3, tolerance_abs=0.1e-3))
    expected = {"min_met": Status.PASS, "min_missed": Status.FAIL, "min_open": Status.UNKNOWN, "max_met": Status.PASS,
                "max_exceeded": Status.FAIL, "max_open": Status.UNKNOWN, "band_met": Status.PASS, "band_open": Status.UNKNOWN}
    assert {k: v.status for k, v in t.items()} == expected
    for result in t.values():
        assert result.value == pytest.approx(A_W, rel=1e-12)  # the traced light's value
        assert "MAX_PATHS" in result.issue_codes
    assert t["min_missed"].issue_codes == ("MAX_PATHS", "TARGET_NOT_MET") and "upper end" in t["min_missed"].message
    assert t["max_exceeded"].issue_codes == ("MAX_PATHS", "TARGET_NOT_MET") and "lower end" in t["max_exceeded"].message
    assert "between 0.001 W and 0.0015 W" in t["min_open"].message and "not decided" in t["min_open"].message
    assert not [r for r in t.values() if r.status is Status.WARNING]


def test_a_transparent_plane_has_no_upper_bound():
    # untraced light may cross a transparent plane more than once: at least 1 mW, no upper end
    t = _two_sources(_power("min_met", minimum=0.9e-3), _power("min_high", minimum=1e3), _power("max_huge", maximum=1e3),
                     _power("max_exceeded", maximum=0.9e-3), absorbing=False)
    assert {k: v.status for k, v in t.items()} == {"min_met": Status.PASS, "min_high": Status.UNKNOWN,
                                                     "max_huge": Status.UNKNOWN, "max_exceeded": Status.FAIL}
    assert "at least 0.001 W" in t["max_huge"].message


def test_coherent_power_with_untraced_power_of_unknown_fate_is_not_decided():
    """B is not launched (the trace budget): power of unknown fate, which may reach the plane in any number of beams. Their
    fields may add up to more than (sqrt P + sqrt U)^2 or cancel the traced light: the coherent sum is bounded by nothing
    (0 to unbounded), so no coherent requirement is decided (no bound for an unknown multiplicity; an earlier
    (sqrt P + sqrt U)^2 interval was not rigorous for several beams)."""
    root = math.sqrt(A_W), math.sqrt(B_W)
    low, high = (root[0] - root[1]) ** 2, (root[0] + root[1]) ** 2  # the former single-field interval 0.0858 mW .. 2.914 mW
    eps = 1e-9
    t = _two_sources(*(_power(name, metric="coherent_power_W", **limits) for name, limits in (
        ("min_low_end", {"minimum": low * (1.0 - 1e-6)}), ("min_above_low", {"minimum": low * (1.0 + 1e-6)}),
        ("min_above_high", {"minimum": high * (1.0 + 1e-6)}), ("max_high_end", {"maximum": high * (1.0 + 1e-6)}),
        ("max_below_high", {"maximum": high * (1.0 - 1e-6)}), ("max_below_low", {"maximum": low * (1.0 - 1e-6)}))))
    assert {k: v.status for k, v in t.items()} == dict.fromkeys(t, Status.UNKNOWN)
    assert t["min_low_end"].value == pytest.approx(A_W, rel=eps) and "nothing bounds the coherent sum" in t["min_low_end"].message
    assert all("TARGET_NOT_MET" not in v.issue_codes and "MAX_PATHS" in v.issue_codes for v in t.values())


def test_several_coherent_untraced_contributions_bound_by_the_sum_of_their_amplitudes():
    """A weak traced beam (1 % folded by M1 and M2) and the 99 % M1's coating transmits, which crosses a bare window before
    the plane: the window's double reflection reaches the plane as a second contribution of the same line. Mutually
    coherent fields add their amplitudes: the coherent sum can reach (sqrt P + sqrt U1 + sqrt U2)^2, more than the single-
    field (sqrt P + sqrt(U1 + U2))^2; a maximum between the two is not decided. The light the run leaves out as negligible
    adds its field too (``FAINT_ROOT``: twice the square root of its power, from either side)."""
    from os_optics.local import lower_project, run_optical_project
    from os_optics.validation.pipeline import run_physics
    from os_optics.validation.untraced import FAINT_ROOT, ROOT_SUM, hazards

    from test_exposure_bounds import _folded_partial_mirror
    from declared_helpers import declared_component, nbk7, sequence, surface

    document = _folded_partial_mirror(observation=True)
    document["components"].append(declared_component("W", "window", "win", position=(50.0, 0.0, 0.0)))
    document["optical_models"].append({"model_id": "win", "model": sequence([surface(1, "0", clear="10"), surface(2, "3", clear="10")],
                                                                            ["N-BK7"], edge="12")})
    # This synthetic coherent-field fixture explicitly assumes lossless glass: otherwise unknown bulk loss, not
    # the separate amplitudes of untraced reflections, would decide the result.
    document["model_materials"] = [{**nbk7(), "absorption_coefficient_per_mm": "0"}]
    document["targets"] = [{"target_id": "c", "metric": "coherent_power_W", "observation_id": "DET", "maximum": 10.0}]
    request = lower_project(document).request
    outcome = run_physics(request.project, request.configuration.numerical)
    ((line, per),) = [(k, v) for k, v in outcome.reach["DET"].items()]
    traced = sum(a.beam.power_W for a in next(o for o in outcome.observations if o.observation_id == "DET").arrivals)
    untraced = sum(hazards(outcome.reach, "DET", [line]).values())  # (the kinds of light, not the amplitude sums beside them)
    assert per[FAINT_ROOT] > 0.0
    single, several = (math.sqrt(traced) + math.sqrt(untraced)) ** 2, (math.sqrt(traced) + per[ROOT_SUM] + per[FAINT_ROOT]) ** 2
    assert per[ROOT_SUM] > math.sqrt(untraced) * (1.0 + 1e-3) and several > single  # several separate contributions
    between = 0.5 * (single + several)
    for maximum, expected in ((between, "UNKNOWN"), (several * (1.0 + 1e-6), "PASS")):
        document["targets"][0]["maximum"] = maximum
        (result,) = run_optical_project(document).result.targets
        assert result.status.value == expected, maximum  # (the single-field bound would have PASSed `between`)
    assert result.value == pytest.approx(traced, rel=1e-9)


def test_complete_traces_are_unchanged():
    a = laser("A", wavelength_nm=780.0, power_W=A_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)
    det = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(a,), observations=(det,), targets=(_power("p", minimum=0.9e-3), _power("q", maximum=0.9e-3))))
    p, q = result.targets
    assert (p.status, p.issue_codes, p.message) == (Status.PASS, (), "")
    assert (q.status, q.issue_codes) == (Status.FAIL, ("TARGET_NOT_MET",))


def test_a_utilization_from_an_incomplete_trace_of_its_own_line_is_not_decided():
    """Power of unknown fate of the rated beam's own line (the per-path interaction limit) may reach the window and, coherent
    with the traced beam, add to it or cancel it there: the utilization has no bound either way, so no limit is decided.
    (Before the coherent combination the traced value was taken as a lower bound: FAIL / PASS. Light of unknown fate of
    another, incoherent source leaves the lower bound: ``test_coherent_exposure``.)"""
    from test_power_exposure_ratings import _document, _independent, _peak_w_per_cm2, _rating

    power, w = _independent()["W/S1"]
    rating_value = 0.5 * _peak_w_per_cm2(power, w)  # the traced light alone gives a utilization of about 2
    targets = [{"target_id": "fails", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0},
               {"target_id": "open", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1000.0},
               {"target_id": "met", "metric": "power_rating_utilization", "device_instance_id": "W", "minimum": 1.5}]
    document = _document([_rating("CW", rating_value, "W/cm2")], targets=targets)
    # Isolate the interaction-limit question from unrelated unstated bulk absorption in this synthetic window.
    document["model_materials"][0]["absorption_coefficient_per_mm"] = "0"
    lowered = lower_project(document)
    cfg = lowered.request.configuration.numerical.model_copy(update={"max_interactions_per_path": 2})
    request = lowered.request.model_copy(update={"configuration": lowered.request.configuration.model_copy(update={"numerical": cfg})})
    result = validate(ValidationRequest.model_validate_json(request.model_dump_json()))
    assert not result.exposures.trace_complete
    t = {r.target_id: r for r in result.targets}
    assert {k: (v.status, v.assessment.value) for k, v in t.items()} == {
        "fails": (Status.UNKNOWN, "INDETERMINATE"), "open": (Status.UNKNOWN, "INDETERMINATE"), "met": (Status.UNKNOWN, "INDETERMINATE")}
    # (the traced beam's utilization is reported as an estimate that names the budget it leaves out: the independent
    # peak over the rated value, 2; it decides none of the three)
    for target in t.values():
        assert target.value == pytest.approx(2.0, rel=1e-6)
        entry = target.value_basis["value"]
        assert entry.kind.value == "ESTIMATE"
        assert [(o.part.value, o.code) for o in entry.omitted] == [("BUDGET", "MAX_INTERACTIONS")]
    assert "MAX_INTERACTIONS" in t["open"].issue_codes and "not known" in t["open"].message


def test_the_evidence_mzi_cascade_of_eight_cells_is_unknown_where_the_trace_budget_decides():
    """Case 9 of the composition evidence (F-EV-1): eight Mach-Zehnder cells imported into one Project. Each cell sends half
    the light on, so END receives P / 2^8 = 3.906 uW (own value: incoherent path sum, as ``power_W`` adds beams); the trace
    budget (512 paths) brings 1.0986 uW to END and leaves 4.761 uW untraced. None of the three END requirements can be
    decided from [1.0986, 5.859] uW, and the run is UNKNOWN, also when it holds only the maximum the full system violates."""

    project = json.loads((FIXTURES / "mzi_cascade_x8.project.json").read_text(encoding="utf-8"))
    own = 1e-3 * 0.5 ** 8
    project["targets"].append({"target_id": "T_END_MAX_HALF", "metric": "power_W", "observation_id": "END", "maximum": 0.5 * own})
    reply = json_api.run_project(project)
    assert reply["ok"], reply
    summary = reply["summary"]
    rows = {t["target_id"]: t for t in summary["targets"]}
    end = next(o for o in summary["observations"] if o["observation_id"] == "END")
    budget = next(i for i in summary["issues"] if i["code"] == "MAX_PATHS" and i.get("location") is None)
    traced, untraced = end["total_power_W"], budget["numerical_context"]["untraced_power_W"]
    assert end["power_complete"] is False
    assert traced == pytest.approx(1.0986e-6, rel=1e-4) and untraced == pytest.approx(4.761e-6, rel=1e-3)
    assert traced < 0.5 * own < 0.99 * own < 1.01 * own < traced + untraced
    assert {k: rows[k]["status"] for k in ("T_END_MIN", "T_END_MAX", "T_END_MAX_HALF")} == {
        "T_END_MIN": "UNKNOWN", "T_END_MAX": "UNKNOWN", "T_END_MAX_HALF": "UNKNOWN"}
    assert all(rows[k]["value"] == pytest.approx(traced, rel=1e-12) for k in ("T_END_MIN", "T_END_MAX", "T_END_MAX_HALF"))
    # a cell's dump minimum of 0 W is met by the traced light alone
    assert {rows[f"s{k}_T_DUMP"]["status"] for k in range(8)} == {"PASS"}
    assert reply["status"] == "UNKNOWN"
    # the budget issue names the requirements it leaves undecided (F-EV-2)
    assert budget["affected_targets"] == ["T_END_MIN", "T_END_MAX", "T_END_MAX_HALF"]
    only_half = {**project, "targets": [t for t in project["targets"] if t["target_id"] == "T_END_MAX_HALF"]}
    alone = json_api.run_project(only_half)
    assert alone["status"] == "UNKNOWN" and alone["summary"]["targets"][0]["status"] == "UNKNOWN"
