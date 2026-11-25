"""Physical faint-light evidence stays independent of optional requirements."""
import json
from pathlib import Path

import pytest
from os_optics import builders as B
from os_optics import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.contracts.results import ObservationResult
from os_optics.local import run_optical_project
from os_optics.local.artifacts import verify_run_artifacts, cross_artifact_problems
from os_optics.results import load_run
from os_optics._vendor.shared_contracts.optical_design import LocalValidationRecordV1
from helpers import run
from test_faint_light_bounds import _leaky
from catalog_store_helpers import REAL_CATALOG


def scene(*, maximum=1e-15, absorbing=True, leak=1e-12, minimum=None):
    source = B.laser('S', wavelength_nm=532, power_W=1, waist_radius_mm=.5,
                     position=(-100, 0, 0), direction=(1, 0, 0))
    mirror = _leaky(B.mirror_for_reflection('M', position=(0, 0, 0), d_in=(1, 0, 0),
                                           d_out=(0, 1, 0), clear_radius_mm=20), leak)
    plane = B.observation_plane('O', position=(100, 0, 0), normal=(1, 0, 0), radius_mm=20, absorbing=absorbing)
    targets = () if maximum is None and minimum is None else (
        Target(target_id='p', observation_id='O', metric='power_W', maximum=maximum, minimum=minimum),)
    return ProjectSnapshot(components=(source, mirror), observations=(plane,), targets=targets)


@pytest.mark.parametrize('maximum, minimum, assessment', [
    (None, None, None),
    (1e-15, None, 'INDETERMINATE'),  # (a maximum below the left-out bound: 0 W meets it, 1e-12 W does not)
    (1e-11, None, 'MET'),            # (met across [0, 1e-12] W)
    (None, 5e-13, 'INDETERMINATE'),  # (a minimum the left-out light could reach)
    (None, 1e-11, 'UNMET'),          # (a minimum above the whole interval)
])
def test_a_leak_below_the_negligible_level_is_the_stated_truncation_of_a_known_zero(maximum, minimum, assessment):
    """A coating that transmits 1e-12 of a 1 W line: below the line's negligible level (1e-9 of its input), so the trace
    leaves it out. The plane behind the mirror is a known zero with that light's bound beside it, whatever the target. The
    target reports the traced 0 W with the entry that states the truncation and its bound, and is graded on the value and
    the bound together: a limit inside [0, 1e-12] W is not decided by the truncation."""
    result = run(scene(maximum=maximum, minimum=minimum))
    obs, = result.observations
    assert obs.arrivals == () and obs.total_power_W == 0
    assert obs.untraced_power_W == 0 and obs.power_complete
    assert obs.power_upper_bound_W == pytest.approx(1e-12, rel=1e-12, abs=0)
    assert obs.quantity_state.value == 'KNOWN_ZERO'
    assert 'receives 0 W' in obs.message
    assert obs.power_by_wavelength[0].complete
    stored = ObservationResult.model_validate_json(obs.model_dump_json())
    assert stored == obs
    if assessment is not None:
        target, = result.targets
        assert target.value == 0.0 and target.assessment.value == assessment
        assert target.status.value == {'MET': 'PASS', 'UNMET': 'FAIL', 'INDETERMINATE': 'UNKNOWN'}[assessment]
        entry = target.value_basis['value']
        omitted, = entry.omitted
        assert entry.kind.value == 'MODEL' and omitted.part.value == 'BELOW_NEGLIGIBLE_LEVEL'
        assert omitted.upper == pytest.approx(1e-12, rel=1e-12, abs=0) and omitted.setting == 'negligible_power_fraction'
        if assessment == 'INDETERMINATE':
            assert 'between 0 W and 1e-12 W' in target.message and 'not decided' in target.message


def test_a_leak_above_the_negligible_level_is_followed_light_not_a_truncation():
    """The same 1e-12 leak with the level lowered to 1e-13 of the line: the leak is followed to the plane, which is then no
    known zero (its power there is stated as untraced light)."""
    lowered = NumericalConfig(energy_conservation_tol=1e-13, negligible_power_fraction=1e-13)
    obs, = run(scene(maximum=None), numerical=lowered).observations
    assert obs.quantity_state.value != 'KNOWN_ZERO' and not obs.power_complete
    assert obs.untraced_power_W == pytest.approx(1e-12, rel=1e-9, abs=0)


def test_genuine_dark_control_is_still_known_zero():
    obs, = run(scene(leak=0)).observations
    assert obs.quantity_state.value == 'KNOWN_ZERO'
    assert obs.power_complete and obs.total_power_W == obs.power_upper_bound_W == 0


def test_transparent_plane_uses_crossing_bound_instead_of_absorbed_energy():
    obs, = run(scene(absorbing=False)).observations
    assert obs.quantity_state.value == 'KNOWN_ZERO' and obs.power_complete
    assert obs.power_upper_bound_W > 1e-12
    assert obs.total_power_W == 0


def test_repeated_ring_crossings_have_no_supported_finite_bound():
    from test_closed_returns import faint_ring
    project = faint_ring().model_copy(update={'observations': (
        B.observation_plane('O', position=(50, 0, 0), normal=(1, 0, 0), radius_mm=20),), 'targets': ()})
    result = run(project)
    obs, = result.observations
    # (only light left out as negligible may circulate there, and nothing bounds its returns: more than the negligible
    # level may arrive, so the plane is no known zero -- its power is not known)
    assert obs.quantity_state.value == 'UNKNOWN' and obs.total_power_W is None
    assert obs.power_upper_bound_W is None and 'power_upper_bound_W' in obs.model_fields_set and not obs.power_complete
    assert 'finite upper bound' in obs.message
    notice, = [i for i in result.issues if i.code.value == 'EXPOSURE_BOUND_WITHHELD' and i.location == 'observation:O']
    assert notice.status.value == 'NOT_APPLICABLE' and notice.numerical_context['cause'] == 'CLOSED_RETURN_NOT_BOUNDED'
    dark, = [i for i in result.issues if i.code.value == 'OBSERVATION_NOT_REACHED']
    assert dark.numerical_context['faint_power_upper_bound_W'] is None


def test_persisted_faint_bound_survives_readers_and_compact_evidence(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/regressions/faint_leak_project.json'
    local = run_optical_project(fixture, catalog=REAL_CATALOG, output_dir=tmp_path)
    assert verify_run_artifacts(tmp_path) == ()
    quantity, = local.local_validation.simulation.observation_quantities
    assert quantity.quantity_state == 'KNOWN_ZERO' and quantity.power_upper_bound_W == pytest.approx(1e-12, rel=1e-12, abs=0)
    for reader in (load_run(local), load_run(tmp_path)):
        obs, = reader.monitors()
        assert obs['power_upper_bound_W'] == pytest.approx(1e-12, rel=1e-12, abs=0) and obs['quantity_state'] == 'KNOWN_ZERO'
    body = local.local_validation.payload()
    body['simulation']['observation_quantities'][0]['power_upper_bound_W'] = 2e-12
    altered = LocalValidationRecordV1.model_validate(body)
    assert cross_artifact_problems(altered, local.optical_view, local.result)
    raw = json.loads((tmp_path / 'simulation_result.json').read_text())['observations'][0]
    raw.pop('power_upper_bound_W')
    assert 'power_upper_bound_W' not in ObservationResult.model_validate(raw).model_dump()


def test_optional_thresholds_do_not_change_quantity_evidence():
    evidence = []
    for maximum in (None, 1e-15, 1e-11):
        obs, = run(scene(maximum=maximum)).observations
        evidence.append((obs.total_power_W, obs.power_complete, obs.untraced_power_W,
                         obs.power_upper_bound_W, obs.quantity_state, obs.power_by_wavelength))
    assert evidence[0] == evidence[1] == evidence[2]


@pytest.mark.parametrize('absorbing', [True, False])
def test_path_budget_keeps_absorbing_energy_bound_but_not_unknown_transparent_crossings(absorbing):
    source = B.laser('S', spectrum=tuple((700 + i, 1) for i in range(40)), power_W=.04,
                     waist_radius_mm=.5, position=(-100, 0, 0), direction=(1, 0, 0))
    plane = B.observation_plane('O', position=(100, 0, 0), normal=(1, 0, 0),
                                radius_mm=20, absorbing=absorbing)
    project = ProjectSnapshot(components=(source,), observations=(plane,), targets=())
    partial = run(project, numerical=NumericalConfig(max_paths=8))
    obs, = partial.observations
    assert not obs.power_complete
    assert obs.total_power_W == pytest.approx(.008, rel=1e-12, abs=0)
    assert obs.untraced_power_W == pytest.approx(.032, rel=1e-12, abs=0)
    # (the traced value is the lower end either way; the lines the budget did not launch may cross a transparent plane
    # more than once: an open upper end there, each watt once on an absorbing plane)
    assert obs.quantity_state.value == 'BOUNDED'
    budget, = obs.value_basis['total_power_W'].omitted
    assert obs.value_basis['total_power_W'].kind.value == 'LOWER_BOUND' and (budget.part.value, budget.code) == ('BUDGET', 'MAX_PATHS')
    if absorbing:
        assert obs.power_upper_bound_W == pytest.approx(.04, rel=1e-12, abs=0)
        assert budget.upper == pytest.approx(.032, rel=1e-12, abs=0)
    else:
        assert obs.power_upper_bound_W is None and budget.upper is None
    full, = run(project, numerical=NumericalConfig(max_paths=64)).observations
    assert full.quantity_state.value == 'KNOWN' and full.power_complete
    assert full.total_power_W == pytest.approx(.04, rel=1e-12, abs=0)
    assert full.power_upper_bound_W == full.total_power_W


def _graded(monkeypatch, mv, **limits):
    from os_optics.validation import metrics
    monkeypatch.setattr(metrics, "_measure", lambda *args, **kwargs: mv)
    return metrics._evaluate(Target(target_id='p', observation_id='O', metric=limits.pop('metric', 'power_W'), **limits),
                             {}, [], None, None, NumericalConfig())


def _truncated(upper):
    from os_optics.contracts.basis import Omitted, ValueBasis
    from os_optics.errors.status import OmittedPart, ValueKind
    return ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=upper,
                                                             setting='negligible_power_fraction'),))


@pytest.mark.parametrize('limits, expected', [
    # power_W known within [0.9, 1.0] W (an annular beam's ends), light left out as negligible adds at most 0.05 W: [0.9, 1.05]
    (dict(maximum=1.02), 'UNKNOWN'),   # met on the counted interval, not at 1.05 W
    (dict(maximum=1.1), 'PASS'),       # met on the whole widened interval (the interval's own WARNING aside)
    (dict(maximum=0.8), 'FAIL'),       # missed at the lower end already
    (dict(minimum=1.03), 'UNKNOWN'),   # missed on the counted interval, reachable with the left-out light
    (dict(minimum=1.2), 'FAIL'),
    # coherent power: that light's field may lower the sum too, by less than the bound: [0.85, 1.05]
    (dict(metric='coherent_power_W', minimum=0.88), 'UNKNOWN'),
    (dict(metric='coherent_power_W', minimum=0.8), 'PASS'),
])
def test_an_interval_value_is_graded_with_the_bound_of_the_light_left_out(monkeypatch, limits, expected):
    """The same rule where the counted value is itself an interval (``_evaluate``'s interval branch): the bound widens it
    for the decision; the message keeps the counted interval and adds the bound only where it leaves the target undecided."""
    from os_optics.errors import Status
    from os_optics.validation.metrics import MetricValue
    coherent = limits.get('metric') == 'coherent_power_W'
    mv = MetricValue(1.0, 'W', Status.PASS, (), '', untraced=(0.9, 1.0), basis=_truncated(0.05)) if coherent else \
        MetricValue(1.0, 'W', Status.PASS, (), '', lower=0.9, basis=_truncated(0.05))
    result = _graded(monkeypatch, mv, **limits)
    got = 'PASS' if result.status in (Status.PASS, Status.WARNING) else result.status.value
    assert got == expected and result.value == 1.0 and result.value_basis == {'value': _truncated(0.05)}
    if expected == 'UNKNOWN':
        assert 'at most 0.05 W' in result.message and 'not decided' in result.message
    else:
        assert 'left out as negligible' not in result.message
