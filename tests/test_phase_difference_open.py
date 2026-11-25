"""The phase difference of two beams where a pass one of them makes has no stated phase at the line.

The trace gives a coating that states no phase the default convention (pi for a reflector, ...) and an element without a
thickness no phase of its own; neither is a statement about the part. Where such a pass is one the two beams do not share,
their phase difference -- and every result built on it -- is stated on that convention with an ESTIMATE entry naming the
passes, and a target on it is reported and not decided; passes both beams make alike cancel, and coatings that state their
phases at the line give the value (``optics/interference/coherence.py``, ``phase_open``).
"""

from __future__ import annotations

import copy
import math

import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

from os_optics.local import example_project_of_kind, run_optical_project
from os_optics.optics.interference.coherence import delay_key, pass_phase_stated, phase_open

from local_helpers import ALONG_Y, FLAT_MIRROR, catalog_document, element
from test_bounded_coating_coherence import FOLD_Y_TO_X, _ideal_run, _stated_phase_mach_zehnder

PHASE_TARGET = {"target_id": "phase", "metric": "phase_difference_rad", "observation_id": "out", "target_value": 0.0,
                "tolerance_abs": 10.0, "unit": "rad"}


def test_shared_passes_cancel_and_a_stated_phase_is_recorded():
    common = delay_key("M0/S1", "reflect", True, 45.0, state="1.000000,0.000000,0.000000")
    stated = delay_key("BS/S2", "split_reflect", True, 45.0, phase_stated=True)
    unstated = delay_key("M1/S1", "reflect", True, 45.0)
    assert stated.endswith("#phase") and pass_phase_stated(stated) and not pass_phase_stated(unstated)
    assert pass_phase_stated(delay_key("BS/S2", "split_reflect", True, 45.0, bound_fs=1000.0, phase_stated=True))
    assert not phase_open((common, stated), (stated, common))  # the same passes: their phases cancel
    assert not phase_open((common, stated), (common,))  # the pass that differs states its phase
    assert phase_open((common, unstated), (common,))


def test_an_unstated_differing_phase_states_the_phase_difference_on_the_default_phases():
    """A declared lossless reference isolates the trace's default interface phases. The phase difference names the
    passes using that convention and its target remains indeterminate for any tolerance."""
    catalog = catalog_document()
    document = example_project_of_kind("two_beam_interference", catalog).payload()
    document["targets"].append({**PHASE_TARGET, "observation_id": "crossing_plane"})
    run = _ideal_run(document, catalog)
    (pair,) = run.result.interference
    assert pair.phase_difference_rad is not None and "PHASE_UNDEFINED" not in pair.issue_codes
    assert pair.fringe_period_nm == pytest.approx(780.0 / (air_index(780.0) * math.sqrt(2.0)), rel=1e-6)
    entry = pair.value_basis["phase_difference_rad"]
    (assumed,) = entry.assumed
    assert entry.kind.value == "ESTIMATE" and assumed.input.value == "UNSTATED_PHASE_AS_INTERFACE"
    assert set(assumed.at) <= {"BS1/S1:refract:+", "BS1/S2:refract:+", "BS1/S2:coating_reflection:+", "M1/S1:reflect:+",
                               "M2/S1:reflect:+"} and assumed.at
    assert not [i for i in run.result.issues if i.code.value == "PHASE_UNDEFINED"]
    target = next(t for t in run.result.targets if t.target_id == "phase")
    assert target.assessment.value == "INDETERMINATE" and target.value == pair.phase_difference_rad
    assert target.value_basis["value"] == entry
    others = {t.target_id: t.status.value for t in run.result.targets if t.target_id != "phase"}
    assert set(others.values()) == {"PASS"}  # (fringe period, balance, overlap, coherence length)


@pytest.mark.parametrize("bs2_reflection_rad", [repr(math.pi / 2), repr(-math.pi / 2)])
def test_stated_phases_give_the_phase_difference_without_a_delay_bound(bs2_reflection_rad):
    """Declared copies of the catalog parts whose coatings state their phases (phase tables), no group-delay bounds: the
    coherence is stated at no added delay at those passes (an entry), the phase difference has its value with no entry,
    and BS2's reflection phase moves it by pi."""
    run = run_optical_project(_stated_phase_mach_zehnder(catalog_document(), bs2_reflection_rad), catalog=catalog_document())
    (pair,) = run.result.interference
    assert "COHERENCE_UNDEFINED" not in pair.issue_codes and "PHASE_UNDEFINED" not in pair.issue_codes
    assert pair.value_basis["coherence_factor"].assumed[0].input.value == "UNSTATED_GROUP_DELAY_ZERO"
    assert "phase_difference_rad" not in pair.value_basis
    entries = [e for o in run.result.observations for a in o.arrivals for e in a.group_delay_unknown_at]
    assert entries and all(e.endswith("#phase") for e in entries)
    assert pair.phase_difference_rad is not None


def test_bs2_phase_moves_the_stated_phase_difference_by_pi():
    values = []
    for phase in (repr(math.pi / 2), repr(-math.pi / 2)):
        run = run_optical_project(_stated_phase_mach_zehnder(catalog_document(), phase), catalog=catalog_document())
        values.append(run.result.interference[0].phase_difference_rad)
    gap = abs(values[0] - values[1])
    assert min(gap, 2.0 * math.pi - gap) == pytest.approx(math.pi, abs=1e-5)


def test_a_coating_both_arms_meet_alike_cancels():
    """A fold mirror of unstated phase before the first splitter: both arms make that pass alike, so it cancels, and the
    phase difference has its value; a target on it is decided."""
    catalog = catalog_document()  # _ideal_run declares a test-only alpha=0 copy to isolate coating phase
    document = _stated_phase_mach_zehnder(catalog, repr(math.pi / 2))
    laser = next(c for c in document["components"] if c["instance_id"] == "SRC")
    laser["nominal_pose"]["position_mm"] = [-100.0, -60.0, 0.0]
    laser["nominal_pose"]["quaternion_wxyz"] = list(ALONG_Y)
    document["components"].append(element("M0", "mirror", FLAT_MIRROR, (-100.0, 0.0, 0.0), FOLD_Y_TO_X,
                                          catalog_version=catalog["catalog_version"]))
    document["targets"] = [copy.deepcopy(PHASE_TARGET)]
    run = run_optical_project(document, catalog=catalog)
    (pair,) = run.result.interference
    arrivals = [a for o in run.result.observations for a in o.arrivals]
    shared = [e for e in arrivals[0].group_delay_unknown_at if e.startswith("M0/")]
    assert shared and all(shared[0] in a.group_delay_unknown_at for a in arrivals) and not pass_phase_stated(shared[0])
    differing = set(arrivals[0].group_delay_unknown_at) ^ set(arrivals[1].group_delay_unknown_at)
    assert differing and all(pass_phase_stated(e) for e in differing)  # the passes that differ state their phases
    assert pair.phase_difference_rad is not None and "PHASE_UNDEFINED" not in pair.issue_codes
    (target,) = run.result.targets
    assert target.status.value == "PASS" and target.value == pair.phase_difference_rad
