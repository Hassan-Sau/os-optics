"""The observation's own power bound covers the light the run's probes followed to the plane.

On a plane the main beam reaches, a ghost of the same source line may arrive beside it. ``ObservationResult.
power_upper_bound_W`` and ``untraced_power_W`` include that light, so the observation and a ``power_W`` target on the plane
state the same interval, whatever the target's threshold.

Reference, independent of the Engine: an uncoated N-BK7 window at normal incidence (SCHOTT Sellmeier, Fresnel
``R = ((n - 1) / (n + 1))^2``, ``T = 1 - R``) passes ``T^2 P`` in its main beam and ``T^2 R^2 P`` in its first forward
ghost (reflected at the second face, then at the first); all its forward orders sum to the plate transmittance
``T^2 / (1 - R^2)``. A collimated beam far inside its Rayleigh range puts every order on the absorbing detector.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status

from helpers import run

X = (1.0, 0.0, 0.0)
P_W = 1e-3


def _n_bk7(wavelength_nm: float) -> float:
    l2 = (wavelength_nm * 1e-3) ** 2
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


N = _n_bk7(633.0)
R = ((N - 1.0) / (N + 1.0)) ** 2
T = 1.0 - R
TRACED = T * T * P_W
PLATE = T * T / (1.0 - R * R) * P_W  # every forward order of the plate


def _project(*targets: Target, absorbing: bool = True) -> ProjectSnapshot:
    source = B.laser("SRC", wavelength_nm=633.0, power_W=P_W, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=X)
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(20.0, 0.0, 0.0))
    det = B.observation_plane("DET", position=(100.0, 0.0, 0.0), normal=X, radius_mm=5.0, absorbing=absorbing)
    return ProjectSnapshot(components=(source, window), observations=(det,), targets=targets)


def _power(target_id: str, **limits) -> Target:
    return Target(target_id=target_id, metric="power_W", observation_id="DET", unit="W", **limits)


def _det(result):
    return next(o for o in result.observations if o.observation_id == "DET")


def test_the_bound_of_a_lit_plane_holds_the_window_ghosts_the_probes_followed():
    """The main beam arrives with T^2 P; the bound reaches the plate's whole transmittance T^2 / (1 - R^2) P (it used to be
    the traced power plus the light left out as negligible only, 1e-13 W above it)."""
    det = _det(run(_project()))
    assert det.total_power_W == pytest.approx(TRACED, rel=1e-9)
    assert det.untraced_power_W == pytest.approx(PLATE - TRACED, rel=1e-6)
    assert not det.power_complete and det.quantity_state.value == "BOUNDED"
    # an upper bound: at least every order of the plate, and above it only by the light left out as negligible
    assert PLATE * (1.0 - 1e-12) <= det.power_upper_bound_W <= PLATE + 1e-8 * P_W


def test_the_observation_bound_and_a_power_target_state_one_interval_whatever_the_threshold():
    """A maximum between the traced power and the plate's transmittance is not decided by the target, and the observation's
    bound does not decide it either; no target, a minimum or a maximum leave the observation's numbers unchanged."""
    between = 0.5 * (TRACED + PLATE)
    runs = {"none": run(_project()), "max_between": run(_project(_power("p", maximum=between))),
            "min": run(_project(_power("p", minimum=1e-4))), "max_lax": run(_project(_power("p", maximum=1.0)))}
    target = runs["max_between"].targets[0]
    assert (target.status, target.value) == (Status.UNKNOWN, pytest.approx(TRACED, rel=1e-9))
    assert "UNTRACED_LIGHT_BOUNDED" in target.issue_codes
    assert _det(runs["max_between"]).power_upper_bound_W > between
    assert runs["max_lax"].targets[0].status is Status.PASS and runs["min"].targets[0].status is Status.PASS
    facts = {name: (_det(r).total_power_W, _det(r).untraced_power_W, _det(r).power_upper_bound_W, _det(r).quantity_state)
             for name, r in runs.items()}
    assert len(set(facts.values())) == 1, facts
    # the target's upper end is the observation's bound: a maximum just above it is met, one just below it is not decided
    upper = _det(runs["none"]).power_upper_bound_W
    assert run(_project(_power("p", maximum=upper * (1.0 + 1e-9)))).targets[0].status is Status.PASS
    assert run(_project(_power("p", maximum=upper * (1.0 - 1e-9)))).targets[0].status is Status.UNKNOWN


def test_a_transparent_plane_counts_the_followed_ghosts_in_its_untraced_power():
    """On a transparent plane the same followed ghost light is part of the plane's untraced power."""
    det = _det(run(_project(absorbing=False)))
    assert det.total_power_W == pytest.approx(TRACED, rel=1e-9)
    assert det.untraced_power_W == pytest.approx(PLATE - TRACED, rel=1e-6)
    assert det.power_upper_bound_W is None or det.power_upper_bound_W >= PLATE * (1.0 - 1e-12)


def _uncoated_pair(absorbing: bool) -> ProjectSnapshot:
    """An uncoated N-BK7 f50 / f100 Keplerian pair (Fresnel reflectance on all four faces) onto a plane 101 mm behind it."""
    source = B.laser("SRC", wavelength_nm=780.0, power_W=P_W, waist_radius_mm=1.0, position=(-200.0, 0.0, 0.0), direction=X)
    l1 = B.singlet_lens("L1", r1_mm=25.8, r2_mm=math.inf, thickness_mm=2.6, clear_radius_mm=6.0, material="N-BK7",
                        position=(0.0, 0.0, 0.0))
    l2 = B.singlet_lens("L2", r1_mm=math.inf, r2_mm=-51.5, thickness_mm=2.2, clear_radius_mm=6.0, material="N-BK7",
                        position=(151.7, 0.0, 0.0))
    out = B.observation_plane("OUT", position=(252.8, 0.0, 0.0), normal=X, radius_mm=12.5, absorbing=absorbing)
    return ProjectSnapshot(components=(source, l1, l2), observations=(out,),
                           targets=(Target(target_id="p", metric="power_W", observation_id="OUT", unit="W", maximum=1.0),))


def test_a_plane_whose_power_has_no_upper_bound_says_why_and_so_does_its_power_target():
    """On a transparent plane behind bare faces the light left out as negligible may cross the plane without bound
    (the faces' return factor), so the plane's power is BOUNDED from the traced light with an open upper end; the
    observation's message and its entry say which light is unbounded, and the power target names it next to the bounded
    ghost light. On an absorbing plane the same light is bounded and the observation states the bound."""
    result = run(_uncoated_pair(absorbing=False))
    out = next(o for o in result.observations if o.observation_id == "OUT")
    assert out.quantity_state.value == "BOUNDED" and out.power_upper_bound_W is None and out.untraced_power_W > 0.0
    assert "negligible may add power without bound" in out.message and "no upper bound" in out.message
    entry = out.value_basis["total_power_W"]
    assert entry.kind.value == "LOWER_BOUND" and [o.part.value for o in entry.omitted] == ["NOT_FOLLOWED", "POSSIBLE_RETURN"]
    assert entry.omitted[0].upper == pytest.approx(out.untraced_power_W, rel=1e-12) and entry.omitted[1].upper is None
    (target,) = result.targets
    assert target.status is Status.UNKNOWN
    assert target.message.index("adds at most") < target.message.index("without bound") < target.message.index("at least")
    absorbing = run(_uncoated_pair(absorbing=True))
    out = next(o for o in absorbing.observations if o.observation_id == "OUT")
    assert out.quantity_state.value == "BOUNDED" and out.message == ""
    assert out.power_upper_bound_W == pytest.approx(out.total_power_W + out.untraced_power_W, rel=1e-6)
    assert out.value_basis is None  # (the flat fields hold the followed light and its bound)


def _windows(count: int, *targets: Target, absorbing: bool) -> ProjectSnapshot:
    """A row of ``count`` uncoated windows in one beam (36 AR windows, shrunk) onto a detector."""
    source = B.laser("SRC", wavelength_nm=633.0, power_W=P_W, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=X)
    windows = tuple(B.window(f"W{k}", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(20.0 + 10.0 * k, 0.0, 0.0))
                    for k in range(count))
    det = B.observation_plane("DET", position=(30.0 + 10.0 * count, 0.0, 0.0), normal=X, radius_mm=5.0, absorbing=absorbing)
    return ProjectSnapshot(components=(source, *windows), observations=(det,), targets=targets)


@pytest.mark.parametrize("absorbing", [False, True])
@pytest.mark.parametrize("max_paths", [2, None])
def test_the_observation_and_its_power_target_agree_whatever_the_probes_budget(max_paths, absorbing):
    """36 AR windows (interactions raised, the probes' trace budget reached): the detector read KNOWN with its
    bound at the traced power while its power_W target said 0.09 W of light the probes did not follow may arrive, without
    an upper bound on a transparent plane; a larger budget then turned the detector UNKNOWN. The observation now states
    the target's interval at every budget: no upper bound where the target has none, else the target's upper end."""
    from os_optics.config import NumericalConfig

    numerical = NumericalConfig() if max_paths is None else NumericalConfig(max_paths=max_paths)

    def outcome(maximum: float):
        result = run(_windows(4, _power("p", maximum=maximum), absorbing=absorbing), numerical=numerical)
        return _det(result), result.targets[0], result

    det, lax, result = outcome(1.0)
    if max_paths is not None:  # (the probes could not follow every ghost: light of unknown fate at the lit plane)
        assert "untraced_light" in result.outcome.execution.budgets.exhausted_by_phase
        assert not det.power_complete and det.untraced_power_W > 0.0
    if det.power_upper_bound_W is None:
        assert det.quantity_state.value == "BOUNDED" and lax.status is Status.UNKNOWN and "at least" in lax.message
        assert det.value_basis["total_power_W"].kind.value == "LOWER_BOUND"
        assert lax.value == det.total_power_W and lax.value_basis["value"].kind.value == "LOWER_BOUND"
    else:
        assert lax.status is Status.PASS and det.power_upper_bound_W > det.total_power_W
        assert outcome(det.power_upper_bound_W * (1.0 + 1e-9))[1].status is Status.PASS
        assert outcome(det.power_upper_bound_W * (1.0 - 1e-9))[1].status is Status.UNKNOWN
    if absorbing:  # (each watt is absorbed once: always bounded)
        assert det.power_upper_bound_W is not None


def _budget_runs(project: ProjectSnapshot):
    """The same scene with the default budget and with a path budget too small for the probes to follow every ghost."""
    from os_optics.config import NumericalConfig

    return {"default": run(project), "max_paths_4": run(project, numerical=NumericalConfig(max_paths=4))}


def _followed_light_codes(result, observation_id: str) -> set[str]:
    """The codes the plane's targets and its issue name for light the trace does not count (the budget's own excluded)."""
    codes = {code for t in result.targets if t.observation_id == observation_id for code in t.issue_codes}
    for issue in result.issues:
        if issue.location == f"observation:{observation_id}":
            codes |= set(issue.numerical_context.get("untraced_causes") or ())
    return codes - {"MAX_PATHS", "FAINT_LIGHT_BOUNDED"}


def _assert_budget_reported_on_its_own(runs) -> None:
    """The small budget is reported where it belongs (the probes' phase, a MAX_PATHS issue and target code); the default
    budget follows every ghost."""
    small, default = runs["max_paths_4"], runs["default"]
    assert "untraced_light" in small.outcome.execution.budgets.exhausted_by_phase
    (issue,) = [i for i in small.issues if i.code.value == "MAX_PATHS"]
    assert issue.location == "untraced_light" and issue.numerical_context["phase"] == "untraced_light"
    assert all("MAX_PATHS" in t.issue_codes for t in small.targets)
    assert not default.outcome.execution.budgets.exhausted_by_phase
    assert not any("MAX_PATHS" in t.issue_codes for t in default.targets)
    for result in runs.values():  # (no model was used outside its validity: in_model_scope stays true)
        assert all(t.status is not Status.OUT_OF_MODEL_SCOPE for t in result.targets)
        assert all(i.status is not Status.OUT_OF_MODEL_SCOPE for i in result.issues)


@pytest.mark.parametrize("absorbing", [True, False])
def test_followed_ghost_light_on_a_lit_plane_keeps_its_code_whatever_the_probes_budget(absorbing):
    """The window's ghosts the probes followed to the detector were OUT_OF_MODEL_SCOPE when the probes also ran out of
    paths for other light, UNTRACED_LIGHT_BOUNDED otherwise. The same light now has one code; the budget keeps MAX_PATHS."""
    runs = _budget_runs(_windows(1, _power("p", maximum=1.0), absorbing=absorbing))
    _assert_budget_reported_on_its_own(runs)
    for result in runs.values():
        assert _followed_light_codes(result, "DET") == {"UNTRACED_LIGHT_BOUNDED"}
        assert _det(result).untraced_power_W > 0.0


def test_followed_ghost_light_on_a_dark_plane_keeps_its_code_whatever_the_probes_budget():
    """The same on a pick-off detector only the window's untraced reflection reaches (no traced beam): its maximum target
    and its OBSERVATION_NOT_REACHED issue name that followed reflection UNTRACED_LIGHT_BOUNDED at both budgets."""
    tilt = math.radians(10.0)
    axis = (math.cos(tilt), math.sin(tilt), 0.0)
    reflected = (-math.cos(2.0 * tilt), -math.sin(2.0 * tilt), 0.0)  # (X mirrored in the window's face normal)
    pick_at = tuple(o - 1.5 * a + 60.0 * r for o, a, r in zip((50.0, 0.0, 0.0), axis, reflected))
    source = B.laser("SRC", wavelength_nm=633.0, power_W=P_W, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=X)
    window = B.window("WIN", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(50.0, 0.0, 0.0), axis=axis)
    main = B.observation_plane("MAIN", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    pick = B.observation_plane("PICK", position=pick_at, normal=reflected, radius_mm=5.0, absorbing=True)
    target = Target(target_id="pick_max", metric="power_W", observation_id="PICK", unit="W", maximum=1e-9)
    runs = _budget_runs(ProjectSnapshot(components=(source, window), observations=(pick, main), targets=(target,)))
    _assert_budget_reported_on_its_own(runs)
    for result in runs.values():
        assert _followed_light_codes(result, "PICK") == {"UNTRACED_LIGHT_BOUNDED"}
        assert result.targets[0].status is Status.UNKNOWN
