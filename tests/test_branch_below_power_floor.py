"""A splitting surface whose reflected branch starts at or below the power floor (``power_epsilon_W``): the Engine ends that
branch where it starts (ZERO_POWER, no segment) while the transmitted beam goes on. The optical view, the ray data and the
run files take it like any other path; this used to raise IndexError in the optical view."""

from __future__ import annotations

import pytest

from os_optics.local import LocalValidationStatusV1, verify_run_artifacts

from declared_helpers import declared_project
from local_helpers import FOLD_X_TO_Y, PLATE_BS, catalog_document, element, plane, project, run, source

SPLITTER_COATING = "BEAMSPLITTER.400-1100R0.5T0.5.AOI45-45"


def test_a_reflected_branch_below_the_power_floor_ends_where_it_starts(tmp_path):
    document = catalog_document() | {"catalog_version": "2026-01-01.2"}
    band = next(c for c in document["coatings"] if c["coating_id"] == SPLITTER_COATING)["bands"][0]
    band["reflectance"], band["transmittance"] = "0.01", "0.99"  # a 1 % sampler
    project_document = project([source(position=(-100.0, 0.0, 0.0), power_w=5e-14),  # reflected: 5e-16 W, below the 1e-15 W floor
                                element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None)],
                               [plane("t", (60.0, 0.0, 0.0), radius_mm=40.0)],
                               [{"target_id": "t", "metric": "power_W", "observation_id": "t", "minimum": 0.0, "unit": "W"}])
    ideal = declared_project(project_document, document)
    glass = next(m for m in ideal["model_materials"] if m["material_id"] == "N-BK7")
    assert "absorption_coefficient_per_mm" not in glass and "bulk_loss" not in glass
    glass["absorption_coefficient_per_mm"] = "0"  # test-only ideal glass; isolates the power-floor branch
    result = run(ideal, catalog=document, output_dir=tmp_path / "run")
    paths = {p.path_id: p for p in result.result.paths}
    assert paths["SRC#0.0"].termination == "ZERO_POWER" and paths["SRC#0.0"].segments == ()
    assert paths["SRC#0"].termination == "ESCAPED"
    # 1 % of this 5e-14 W line stopped at the power floor without being followed: the traced light at "t" is a lower bound
    # of its power, which meets the minimum of 0 W whatever that light does (PASS, ZERO_POWER named), and the plane's
    # spectral power says the value is incomplete (unknown is never read as complete)
    assert result.status is LocalValidationStatusV1.PASS
    (target,) = result.result.targets
    assert target.status.value == "PASS" and "ZERO_POWER" in target.issue_codes
    (spectral,) = result.result.observations[0].power_by_wavelength
    # (the plate's other face reflects 1 % of the transmitted beam: an untraced reflection whose probe stops at the same
    # floor, light of unknown fate that may reach the plane as well -- the target counts it too)
    ghost = result.result.power_budget.lines[0].sinks["UNTRACED_REFLECTION"]
    assert not spectral.complete
    assert spectral.untraced_power_W == pytest.approx(paths["SRC#0.0"].final_power_W + ghost, rel=1e-12, abs=0)
    assert verify_run_artifacts(tmp_path / "run") == ()
