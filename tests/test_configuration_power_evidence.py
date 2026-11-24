"""A configuration comparison carries each run's proven power interval. The catalog fixture's unstated glass absorption
leaves the upper end open at both monitors. Close to the fold the mirror can take part of the direct light before it reaches
the plane, so its lower bound is below the minimum. Farther from the fold the direct light proves the minimum.
"""

from __future__ import annotations

import json

import pytest

from os_optics.cli import main
from os_optics.contracts.analysis import AnalysisResultV1, IntervalKind
from os_optics.errors.status import QuantityState
from os_optics.local import example_project_of_kind

from catalog_store_helpers import REAL_CATALOG

ALONG_X = [0.5, 0.5, 0.5, 0.5]
MINIMUM_W = 0.9e-3


def _fold_configurations() -> dict:
    """The folded reference design (a 1 mW, 780 nm, 0.5 mm beam onto a 45 deg mirror) with a monitor 0.2 mm before the mirror in
    configuration ``near`` and 5 mm before it in ``far``, and a 0.9 mW minimum on it."""

    document = example_project_of_kind("folded", REAL_CATALOG).payload()
    document["observations"].append({"observation_id": "M", "pose": {"position_mm": [-0.2, 0.0, 0.0], "quaternion_wxyz": ALONG_X},
                                     "aperture": {"shape": "circle", "radius_mm": 10.0}})
    document["targets"].append({"target_id": "m_power", "metric": "power_W", "observation_id": "M", "minimum": MINIMUM_W, "unit": "W"})
    document["configurations"] = [
        {"configuration_id": "near"},
        {"configuration_id": "far", "observation_poses": [
            {"observation_id": "M", "pose": {"position_mm": [-5.0, 0.0, 0.0], "quaternion_wxyz": ALONG_X}}]},
    ]
    return document


@pytest.fixture
def runs(tmp_path, real_catalog_store):
    root = tmp_path
    path = root / "fold.json"
    path.write_text(json.dumps(_fold_configurations()), encoding="utf-8")
    out = root / "run"
    main(["project", "run", str(path), "--out-dir", str(out), "--status-contract", "execution"])
    comparison = AnalysisResultV1.model_validate(json.loads((out / "comparison.json").read_text(encoding="utf-8")))
    records = {cid: json.loads((out / "configurations" / cid / "local_validation.json").read_text(encoding="utf-8"))
               for cid in ("near", "far")}
    return comparison, records


def _plane(record: dict, observation_id: str) -> dict:
    return next(q for q in record["simulation"]["observation_quantities"] if q["observation_id"] == observation_id)


def _target(record: dict, target_id: str) -> dict:
    return next(t for t in record["simulation"]["target_outcomes"] if t["target_id"] == target_id)


def test_a_plane_before_the_fold_does_not_promote_traced_power_to_a_lower_bound(runs):
    comparison, records = runs
    plane = _plane(records["near"], "M")
    assert plane["quantity_state"] == "BOUNDED" and not plane["power_complete"]
    lower = plane["power_lower_bound_W"]
    assert 0.0 < lower < MINIMUM_W < plane["total_power_W"]  # a share may be taken by the mirror before this plane
    assert plane["power_upper_bound_W"] is None  # unknown bulk loss and return paths prevent a finite upper bound
    assert _target(records["near"], "m_power")["assessment"] == "INDETERMINATE"

    q = comparison.quantity("near|M|power_W")
    assert q.state is QuantityState.BOUNDED and q.value is None
    assert q.interval.kind is IntervalKind.PROVEN and q.interval.lower == lower and q.interval.upper is None
    assert q.depends_on == tuple(plane["issue_codes"])
    assert not q.decides(MINIMUM_W)


def test_a_plane_far_from_the_fold_meets_its_minimum_without_claiming_known_power(runs):
    comparison, records = runs
    plane = _plane(records["far"], "M")
    q = comparison.quantity("far|M|power_W")
    assert plane["quantity_state"] == q.state.value == "BOUNDED" and not plane["power_complete"]
    assert plane["total_power_W"] > MINIMUM_W  # the direct source certainly reaches this upstream plane
    assert plane.get("power_lower_bound_W") is None and plane["power_upper_bound_W"] is None
    assert q.value is None and q.interval.kind is IntervalKind.PROVEN
    assert q.interval.lower == plane["total_power_W"] and q.interval.upper is None
    assert q.depends_on == tuple(plane["issue_codes"]) and q.decides(MINIMUM_W)
    # The direct light proves this minimum with the upper end open. The comparison also carries the run's target verdict.
    assert _target(records["far"], "m_power")["assessment"] == "MET"
    rows = [row for row in comparison.arrays["targets"] if row["target_id"] == "m_power"]
    assert len(rows) == 1
    far = next(c for c in rows[0]["configurations"] if c["configuration_id"] == "far")
    assert far["assessment"] == "MET"


def test_the_comparisons_target_assessments_are_the_runs(runs):
    comparison, records = runs
    table = comparison.arrays
    rows = json.dumps(table)
    for cid, record in records.items():
        assessment = _target(record, "m_power")["assessment"]
        assert f'"{cid}"' in rows and assessment in rows


def test_a_sampled_value_keeps_its_resolution_and_is_never_a_proven_interval():
    """A plane read from pupil samples (``power_sampling_resolution_W``): the comparison keeps the resolution as convergence evidence,
    so a threshold within it is not decided, and an interval resting on a sampled value is not PROVEN."""

    from types import SimpleNamespace

    from os_optics.analysis.configurations import _power

    base = {"observation_id": "M", "issue_codes": (), "total_power_W": 0.75e-3, "power_upper_bound_W": None,
            "power_lower_bound_W": None, "power_sampling_resolution_W": 0.12e-3}
    known = _power("c", SimpleNamespace(**base, quantity_state="KNOWN"))
    assert known.convergence is not None and known.convergence.resolution == 0.12e-3
    assert not known.decides(0.8e-3) and known.decides(0.5e-3)
    bounded = _power("c", SimpleNamespace(**{**base, "power_lower_bound_W": 0.7e-3, "power_upper_bound_W": 0.8e-3}, quantity_state="BOUNDED"))
    assert bounded.interval.kind is IntervalKind.SAMPLING and not bounded.decides(0.5e-3)
