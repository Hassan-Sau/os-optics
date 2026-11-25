"""``ValidationIssue.affected_targets`` names the targets an issue leaves undecided (F-EV-2): the run's
trace budget, the per-path interaction limit, a plane no traced beam reaches while light was not followed, the beam-pair
limit. Targets decided anyway (on a bound, or by light the issue does not concern) are not named."""

from __future__ import annotations

import pytest

from os_optics import NumericalConfig
from os_optics.builders import laser, npbs, observation_plane
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status

from helpers import run

X = (1.0, 0.0, 0.0)
MX = (-1.0, 0.0, 0.0)


def _issues(result, code: str) -> dict[str | None, tuple[str, ...]]:
    return {i.location: i.affected_targets for i in result.issues if i.code.value == code}


def test_the_run_trace_budget_names_the_targets_of_the_untraced_lines():
    a = laser("A", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)
    b = laser("B", wavelength_nm=780.0, power_W=0.5e-3, waist_radius_mm=0.5, position=(-100.0, 3.0, 0.0), direction=X)
    det = observation_plane("det", position=(50.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    targets = (Target(target_id="b", metric="power_W", observation_id="det", source_id="B", minimum=1e-6),
               Target(target_id="b_radius", metric="beam_radius_mm", observation_id="det", source_id="B", maximum=1.0),
               Target(target_id="total_open", metric="power_W", observation_id="det", maximum=1.2e-3),
               Target(target_id="total_met", metric="power_W", observation_id="det", maximum=1.0),
               Target(target_id="a", metric="power_W", observation_id="det", source_id="A", minimum=1e-6))
    result = run(ProjectSnapshot(components=(a, b), observations=(det,), targets=targets), numerical=NumericalConfig(max_paths=1))
    status = {t.target_id: t.status for t in result.targets}
    assert status == {"b": Status.UNKNOWN, "b_radius": Status.UNKNOWN, "total_open": Status.UNKNOWN, "total_met": Status.PASS,
                      "a": Status.PASS}
    assert _issues(result, "MAX_PATHS") == {None: ("b", "b_radius", "total_open")}
    # consistent with the rows: every UNKNOWN target with MAX_PATHS is named, nothing else
    assert {t.target_id for t in result.targets if t.status is Status.UNKNOWN and "MAX_PATHS" in t.issue_codes} == {"b", "b_radius", "total_open"}


def test_the_interaction_limit_and_an_unknown_plane_name_their_targets():
    """Two partial reflectors facing each other (an etalon) stop at the per-path interaction limit; a plane beside the beam
    receives no traced light, and whether the untraced light reaches it is not known."""
    src = laser("SRC", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)
    plates = (npbs("P0", reflectance=0.5, position=(0.0, 0.0, 0.0), normal=MX), npbs("P1", reflectance=0.5, position=(20.0, 0.0, 0.0), normal=X))
    det = observation_plane("det", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    side = observation_plane("side", position=(150.0, 40.0, 0.0), normal=X, radius_mm=5.0, absorbing=True)
    # Exact-cap free propagation delivers .25 + .0625 mW to det. Choose a
    # threshold above that traced lower bound, inside the unresolved interval.
    targets = (Target(target_id="det_open", metric="power_W", observation_id="det", maximum=0.32e-3),
               Target(target_id="det_met", metric="power_W", observation_id="det", minimum=1e-6),
               Target(target_id="side_power", metric="power_W", observation_id="side", maximum=1e-9),
               Target(target_id="det_radius", metric="beam_radius_mm", observation_id="det", maximum=5.0,
                      parameters={"path_id": "SRC#0"}))
    result = run(ProjectSnapshot(components=(src, *plates), observations=(det, side), targets=targets),
                 numerical=NumericalConfig(max_interactions_per_path=4))
    rows = {t.target_id: t for t in result.targets}
    assert rows["det_open"].value == pytest.approx(0.3125e-3)
    assert rows["det_open"].status is Status.UNKNOWN and "MAX_INTERACTIONS" in rows["det_open"].issue_codes
    assert rows["det_met"].status is Status.PASS
    assert rows["side_power"].status is Status.UNKNOWN and "MAX_INTERACTIONS" in rows["side_power"].issue_codes
    assert rows["det_radius"].status is Status.PASS  # selected primary beam: decided, not named
    limits = _issues(result, "MAX_INTERACTIONS")
    assert limits and all(named == ("det_open", "side_power") for named in limits.values())
    assert _issues(result, "OBSERVATION_NOT_REACHED") == {"observation:side": ("side_power",)}


def test_the_beam_pair_limit_leaves_unselected_pairs_unknown_and_names_them():
    src = laser("SRC", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X,
                coherence_length_mm=1e6)
    plates = (npbs("P0", reflectance=0.5, position=(0.0, 0.0, 0.0), normal=MX), npbs("P1", reflectance=0.5, position=(20.0, 0.0, 0.0), normal=X))
    det = observation_plane("det", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0)
    targets = (Target(target_id="opd", metric="optical_path_difference_mm", observation_id="det", minimum=-1e6),)
    result = run(ProjectSnapshot(components=(src, *plates), observations=(det,), targets=targets),
                 numerical=NumericalConfig(max_interactions_per_path=10, max_interference_pairs=1))
    (row,) = result.targets
    assert row.status is Status.UNKNOWN and row.issue_codes == ("MAX_BEAM_PAIRS",)
    assert _issues(result, "MAX_BEAM_PAIRS") == {"observation:det": ("opd",)}
    # (the cap of the pair listing is a notice: the target beyond it is UNKNOWN by itself, and the run's budget record
    # names the plane)
    (capped,) = [i for i in result.issues if i.code.value == "MAX_BEAM_PAIRS"]
    assert capped.status is Status.NOT_APPLICABLE and capped.numerical_context["limit"] == 1
    assert "det" in result.outcome.execution.budgets.affected_observation_ids
    assert "max_interference_pairs" in result.outcome.execution.budgets.exhausted
