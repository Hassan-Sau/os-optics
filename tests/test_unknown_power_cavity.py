"""A branch of unknown power in a cavity: no power floor ends it, so the budgets do, and what they report of it is not a zero.

The geometry: a grating with no stated response between mirrors facing each other. Expected by the budgets' own definitions: one path up to
``max_interactions_per_path`` surfaces with full mirrors, ``max_paths`` paths with partial ones; the ledger holds the launched power once, in
``UNMODELLED_RESPONSE``; a plane in the cavity is not known and has no upper bound, since that light may cross it any number of times.
"""

from __future__ import annotations

import pytest

from os_optics.local import run_optical_project

from declared_helpers import declared_component, sequence, surface, with_models
from local_helpers import AGAINST_X, ALONG_X, POWER_W, catalog_document, plane, project, source

COATINGS = catalog_document()["coatings"]
HR = next(c for c in COATINGS if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
SPLITTER = next(c for c in COATINGS if c["kind"] == "BEAMSPLITTER")
GRATING = {"model_id": "g", "model": {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "100", "orders": [0],
                                      "clear_half_widths_mm": ["20", "20"], "edge_half_widths_mm": ["21", "21"], "status": "SPECIFIED"}}
TARGET = [{"target_id": "p", "metric": "power_W", "observation_id": "IN", "minimum": 1e-6, "unit": "W"}]


def _mirror(model_id: str, coating: dict, traced: str) -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", clear="30", traced=traced, coating_id=coating["coating_id"])], [], edge="31")}


def _cavity(coating: dict, traced: str, mirrors: tuple[tuple[float, tuple], ...]):
    parts = [declared_component(f"M{k}", "mirror", f"m{k}", (x, 0.0, 0.0), pose) for k, (x, pose) in enumerate(mirrors)]
    document = with_models(project([source("S", (-5.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X), *parts],
                                   [plane("IN", (25.0, 0.0, 0.0), radius_mm=25.0)], TARGET),
                           [GRATING, *(_mirror(f"m{k}", coating, traced) for k in range(len(mirrors)))], None, [coating], design_intent="OPTICAL_SYSTEM")
    return run_optical_project(document, catalog=None).result


def _plane_is_not_known(result) -> None:
    (obs,) = result.observations
    assert obs.total_power_W is None and obs.power_complete is False and obs.power_upper_bound_W is None
    assert all(arrival.beam.power_W is None for arrival in obs.arrivals)
    (target,) = result.targets
    assert target.status.value == "UNKNOWN" and target.value is None
    assert result.power_budget.conserved
    assert dict(result.power_budget.total_sinks) == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=1e-12)}


def test_the_interaction_limit_ends_a_branch_of_unknown_power_and_says_so():
    result = _cavity(HR, "REFLECTED", ((50.0, AGAINST_X), (-10.0, ALONG_X)))
    _plane_is_not_known(result)
    (limit,) = [i for i in result.issues if i.code.value == "MAX_INTERACTIONS"]
    assert " 0 W" not in limit.message and "not known" in limit.message
    assert "untraced_power_W" not in limit.numerical_context
    assert limit.numerical_context["power_upper_bound_W"] == pytest.approx(POWER_W)


def test_the_path_budget_ends_branches_of_unknown_power_and_says_so():
    result = _cavity(SPLITTER, "TRANSMITTED_AND_REFLECTED", ((50.0, AGAINST_X), (-10.0, ALONG_X), (20.0, AGAINST_X)))
    assert len(result.paths) == 512
    _plane_is_not_known(result)
    (budget,) = [i for i in result.issues if i.code.value == "MAX_PATHS"]
    assert " 0 W" not in budget.message and "0 %" not in budget.message and "a power that is not known" in budget.message
    branches = budget.numerical_context["untraced_branches"]
    assert branches > 0 and budget.numerical_context["untraced_unknown_power_paths"] == branches
    assert "untraced_power_W" not in budget.numerical_context
