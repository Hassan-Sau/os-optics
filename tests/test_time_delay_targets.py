"""Time-domain targets on the trace: a beam's group delay and phase delay from its source (``group_delay_ps``,
``phase_delay_ps``) and two beams' relative envelope delay (``time_delay_ps``) -- the pump-probe delay of a delay line.

Independent references, written out here: ``2 dx / c`` for a retro mirror moved by ``dx`` in air (index 1 exactly), the
distance over c in free space, the N-BK7 group index ``n_g = n - lambda dn/dlambda`` of the Schott Sellmeier formula by
central differences for a plate (``(n_g - 1) t / c`` of extra group delay, ``(n - 1) t / c`` of phase delay), and the
network export's per-path group delay.
"""

from __future__ import annotations

import json

import pytest

from os_optics.local import run_optical_project

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import declared_component, declared_project, nbk7, sequence, surface, thin_lens_model, with_models
from local_helpers import ALONG_X, plane, project, source
from michelson_helpers import expected, michelson
from test_group_delay_and_returns import _bk7_group

C_MM_PER_PS = 0.299792458
WAVELENGTH_NM = 780.0


def _target(metric: str, observation: str = "OUT", **fields) -> dict:
    return {"target_id": metric, "metric": metric, "observation_id": observation, "minimum": -1e9, "unit": "ps", **fields}


def _values(run_) -> dict:
    return {t.target_id: t for t in run_.result.targets}


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(REAL_CATALOG.read_text(encoding="utf-8"))


def test_free_space_group_and_phase_delay_are_the_distance_over_c():
    document = project([source(position=(-200.0, 0.0, 0.0))], [plane("OUT", (100.0, 0.0, 0.0))],
                       [_target("group_delay_ps"), _target("phase_delay_ps")])
    values = _values(run_optical_project(document, catalog=None))
    assert values["group_delay_ps"].value == pytest.approx(300.0 / C_MM_PER_PS, rel=1e-12)
    assert values["phase_delay_ps"].value == pytest.approx(300.0 / C_MM_PER_PS, rel=1e-12)
    assert values["group_delay_ps"].status.value == "PASS"


def _plate_project(thickness: str = "10", *, lens: bool = False) -> dict:
    window = {"model_id": "plate", "model": sequence([surface(1, "0"), surface(2, thickness)], ["N-BK7"])}
    components = [source(position=(-200.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM),
                  declared_component("W", "window", "plate", (0.0, 0.0, 0.0), ALONG_X)]
    models = [window]
    if lens:
        components.append(declared_component("L", "lens", "thin", (50.0, 0.0, 0.0), ALONG_X))
        models.append(thin_lens_model("thin", "500"))
    document = project(components, [plane("OUT", (100.0, 0.0, 0.0))], [_target("group_delay_ps"), _target("phase_delay_ps")])
    return with_models(document, models, [nbk7()])


def test_a_glass_plate_adds_its_group_delay_not_its_phase_delay():
    values = _values(run_optical_project(_plate_project(), catalog=None))
    n, n_g, _ = _bk7_group(WAVELENGTH_NM)
    assert values["group_delay_ps"].value == pytest.approx((300.0 + (n_g - 1.0) * 10.0) / C_MM_PER_PS, rel=1e-9)
    assert values["phase_delay_ps"].value == pytest.approx((300.0 + (n - 1.0) * 10.0) / C_MM_PER_PS, rel=1e-9)
    assert values["group_delay_ps"].value - values["phase_delay_ps"].value == pytest.approx((n_g - n) * 10.0 / C_MM_PER_PS, rel=1e-6)


def test_an_element_without_a_stated_delay_reports_the_modelled_paths_group_delay_ungraded():
    """An ideal thin lens (no thickness: its own delay is not stated) behind the plate: the group delay reported is the
    modelled path's -- the air and the plate, the reference of the plate-only case above, the number the arrival holds
    over c -- with the pass named, and it is not graded. The phase delay keeps no value (no optical path is stated behind
    such an element)."""
    run_ = run_optical_project(_plate_project(lens=True), catalog=None)
    values = _values(run_)
    group, phase = values["group_delay_ps"], values["phase_delay_ps"]
    n, n_g, _ = _bk7_group(WAVELENGTH_NM)
    assert group.value == pytest.approx((300.0 + (n_g - 1.0) * 10.0) / C_MM_PER_PS, rel=1e-9)
    (arrival,) = run_.result.observations[0].arrivals
    assert group.value == arrival.group_path_length_mm / C_MM_PER_PS
    assert (group.status.value, group.assessment.value) == ("UNKNOWN", "INDETERMINATE") and "GROUP_DELAY_UNDEFINED" in group.issue_codes
    assert "L/" in group.message  # (names the pass)
    entry = group.value_basis["value"]
    assert entry.kind.value == "ESTIMATE"
    (assumed,) = entry.assumed
    assert (assumed.input.value, assumed.code) == ("UNSTATED_GROUP_DELAY_ZERO", "GROUP_DELAY_UNDEFINED")
    assert assumed.at and all(step.startswith("L/") for step in assumed.at)
    assert phase.value is None and "PHASE_UNDEFINED" in phase.issue_codes
    # a limit the number misses is not graded either: the element's own delay is not in it
    missed = _plate_project(lens=True)
    missed["targets"] = [_target("group_delay_ps", minimum=1e6)]
    (undecided,) = run_optical_project(missed, catalog=None).result.targets
    assert (undecided.status.value, undecided.assessment.value, undecided.value) == ("UNKNOWN", "INDETERMINATE", group.value)
    assert "TARGET_NOT_MET" not in undecided.issue_codes


def _ideal_delay_project(catalog: dict, extra_mm: float = 0.0, **fields) -> dict:
    """Declared copies preserve the real prescription/coatings, with lossless glass assumed only for delay tests."""
    document = declared_project(michelson(catalog, extra_mm, targets=[_target("time_delay_ps", "out", **fields)]), catalog)
    for material in document["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"  # Synthetic timing reference, not catalog evidence.
    return document


def _delay(catalog: dict, extra_mm: float, **fields) -> object:
    return _values(run_optical_project(_ideal_delay_project(catalog, extra_mm, **fields), catalog=None))["time_delay_ps"]


def test_a_delay_line_moves_the_pair_delay_by_twice_the_mirror_travel_over_c(catalog):
    base = _delay(catalog, 0.0)
    for dx in (0.015, 0.3, 1.5):  # 100 fs, 2 ps and 10 ps of delay
        moved = _delay(catalog, dx)
        assert abs(moved.value - base.value) == pytest.approx(2.0 * dx / C_MM_PER_PS, rel=1e-9)


def test_the_pair_delay_has_the_coatings_bound_and_its_sign_follows_the_operands(catalog):
    want = expected(catalog)
    forward = _delay(catalog, 0.0)
    # arm R is path .0 (operand a), arm T path .1: a - b = -(group path T - R) / c; the coatings' bounds give its interval
    assert forward.value == pytest.approx(-want.group_difference_mm / C_MM_PER_PS, rel=1e-6)
    assert forward.status.value == "PASS" and "within 2.3 ps" in forward.message
    reverse = _delay(catalog, 0.0, parameters=[{"key": "path_id_a", "value": "SRC#0.1"}, {"key": "path_id_b", "value": "SRC#0.0"}])
    assert reverse.value == -forward.value
    tight = _ideal_delay_project(catalog, minimum=-12.0)
    graded = _values(run_optical_project(tight, catalog=None))["time_delay_ps"]
    assert graded.status.value == "UNKNOWN"  # the interval [-15.24, -10.64] ps straddles the limit


def test_the_unmodified_catalog_does_not_claim_a_known_delay_target_without_bulk_loss(catalog):
    """Only the declared timing-reference copies assume zero loss; actual catalog glass has no loss source."""
    result = run_optical_project(michelson(catalog, targets=[_target("time_delay_ps", "out")]), catalog=catalog).result
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and target.value is None
    # Unknown bulk loss prevents a supported two-arm selection; unstated coating phase also prevents
    # treating the pair's envelope timing as a decisive catalog-only number.
    assert target.issue_codes == ("PHASE_UNDEFINED",)
    assert any(issue.code.value == "BULK_LOSS_NOT_ESTABLISHED" for issue in result.issues)
    assert len(result.observations[0].arrivals) == 2  # The two physical arm routes still exist.


def test_a_pair_whose_coatings_state_no_group_delay_reports_the_modelled_paths_relative_delay(catalog):
    """The Michelson with a catalog whose coatings publish no group-delay bound: the two arms meet the splitter's coating
    differently, so the relative delay is that of the two modelled paths (the same group-path difference as with the
    bounds, from the independent arm lengths), reported with the passes named and not graded; its sign follows the
    target's operands."""
    import copy

    bare = copy.deepcopy(catalog)
    for coating in bare["coatings"]:
        coating.pop("group_delay_bound_fs", None)
    want = expected(catalog)
    forward = _delay(bare, 0.0)
    assert forward.value == pytest.approx(-want.group_difference_mm / C_MM_PER_PS, rel=1e-6)
    assert forward.value == pytest.approx(_delay(catalog, 0.0).value, rel=1e-12)
    assert (forward.status.value, forward.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert "GROUP_DELAY_UNDEFINED" in forward.issue_codes
    (assumed,) = forward.value_basis["value"].assumed
    assert forward.value_basis["value"].kind.value == "ESTIMATE" and assumed.input.value == "UNSTATED_GROUP_DELAY_ZERO" and assumed.at
    reverse = _delay(bare, 0.0, parameters=[{"key": "path_id_a", "value": "SRC#0.1"}, {"key": "path_id_b", "value": "SRC#0.0"}])
    assert reverse.value == -forward.value
def test_two_sources_have_no_common_time_origin():
    document = project([source("A", (-200.0, 0.0, 0.0)), source("B", (-200.0, 0.5, 0.0))], [plane("OUT", (100.0, 0.0, 0.0))],
                       [_target("time_delay_ps", source_id="A", source_id_b="B")])
    (target,) = run_optical_project(document, catalog=None).result.targets
    assert target.value is None and "GROUP_DELAY_UNDEFINED" in target.issue_codes


def test_the_group_delay_agrees_with_the_network_export():
    from os_optics.local import lower_project
    from os_optics.network.export import build_linear_optical_network

    document = _plate_project()
    run_ = run_optical_project(document, catalog=None)
    lowered = lower_project(document, catalog=None)
    network = build_linear_optical_network(lowered.request.project, engine_input_hash="sha256:" + "0" * 64)
    (path,) = [p for p in network.paths if p.output_beam is not None] or network.paths[:1]
    assert path.group_delay_ps == pytest.approx(_values(run_)["group_delay_ps"].value, rel=1e-9)
