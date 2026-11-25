"""A native annular Project keeps its aperture proof when branches drop pupil rays."""

import json
import math
from pathlib import Path

import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

from catalog_store_helpers import REAL_CATALOG, pinned_to_seed
from os_optics.local import run_optical_project
from os_optics.local.artifacts import verify_run_artifacts
from os_optics.optics.rays.tracer import Tracer


def test_annular_project_probes_keep_opening_geometry_without_discarded_ray_indices(tmp_path, monkeypatch):
    checked = []
    branch = Tracer._branch

    def check_branch(self, *args, **kwargs):
        before = len(self.out.untraced_branches)
        branch(self, *args, **kwargs)
        for emitted in self.out.untraced_branches[before:]:
            packet = emitted.seed
            if packet.pupil_total == 0 and packet.opening is not None:
                # Check the actual one-ray probe seed, rather than waiting for a later aperture to
                # need _edge_sign: _whole_annulus_inside can already prove containment from the envelope.
                opening = packet.opening
                assert opening.radius == 4.0
                assert opening.ray_offsets.shape == (1, 2)
                assert packet.gstate.valid
                for axis in (packet.gstate.bx, packet.gstate.by):
                    signs = self._edge_sign(packet, opening, axis, packet.d[0])
                    checked.append(signs)
                    # The opening survives, but one reference ray cannot determine its map's orientation.
                    assert signs == (1.0, -1.0)

    monkeypatch.setattr(Tracer, "_branch", check_branch)
    project = pinned_to_seed(json.loads((Path(__file__).parent / "fixtures/regressions/annular_probe_project.json").read_bytes()))
    run = run_optical_project(project, catalog=REAL_CATALOG, output_dir=tmp_path)
    assert checked
    assert verify_run_artifacts(tmp_path) == ()
    assert run.result.outcome.execution.outcome.value == "COMPLETED"
    assert run.result.outcome.execution.budgets.exhausted == ()
    assert {issue.code.value for issue in run.result.issues} == {
        "GAUSSIAN_CENTRAL_OBSTRUCTION", "BULK_LOSS_NOT_ESTABLISHED"}
    assert run.result.power_budget.conserved
    (plane,) = run.result.observations
    # The catalog glass contributes no known arrival power. The probes leave unresolved returns that may cross this
    # transparent plane repeatedly, so the individual arrival's bound cannot cap the plane's total.
    assert plane.quantity_state.value == "UNKNOWN" and not plane.power_complete
    assert plane.power_lower_bound_W is None and plane.power_upper_bound_W is None
    assert plane.value_basis is None
    assert "bulk absorption is not established" in plane.message and "path ended out of model scope" in plane.message
    assert plane.untraced_power_W > 1e-5
    assert len(plane.arrivals) == 1
    # Independent Gaussian integral at the stop, 20 mm from a 1 mm waist, followed by
    # the catalog lens's two 99% transmitting AR faces: this bounds—not measures—the light through
    # its glass, whose bulk loss is unstated. The annulus remains non-Gaussian.
    zr = math.pi * air_index(780.0) / (780e-6)  # (the Rayleigh range in the Project's air, 20 C and 101325 Pa: pi w0^2 n / lambda)
    w2 = 1 + (20 / zr) ** 2
    transmitted = math.exp(-2 * .5**2 / w2) - math.exp(-2 * 4**2 / w2)
    assert plane.total_power_W is None
    assert plane.arrivals[0].beam.power_W is None
    assert plane.arrivals[0].power_upper_bound_W == pytest.approx(.001 * transmitted * .99**2, rel=1e-9)
    assert run.status.value == "OUT_OF_MODEL_SCOPE"
