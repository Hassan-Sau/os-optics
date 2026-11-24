"""A beam whose power cannot be computed behind a surface goes on as geometry.

Where the geometric pupil-sample basis gives no power (every sample cut while the chief ray passes, samples that do not
follow the beam's diffraction, an annulus cut behind a central obstruction) the path used to end there and every plane
behind lost the beam. The beam now goes on: its power at that surface is booked once to light of unknown fate
(``TERMINATED_OUT_OF_MODEL``, the same amount as before), and from there on it carries no power, only the bound of what
it may carry -- what arrived, since every later element passes at most what arrives. No plane, pair or exposure behind
reads a number from it.

References: the ledger's closure (energy conservation), the closed-form Gaussian share a round stop passes
(``1 - exp(-2 a^2 / w^2)``), and the same scene without the second stop for the geometry (positions and directions).
"""

from __future__ import annotations

import math

import pytest

from helpers import arrivals, assert_power_not_computed, run
from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target

P0 = 1e-3
X = (1.0, 0.0, 0.0)
S2 = math.sqrt(0.5)


def _scene(second_stop: float | None = 0.1) -> ProjectSnapshot:
    """A collimated 1 mm beam through a stop at one beam radius (13.5 % cut: no valid Gaussian state behind it), a second
    stop of ``second_stop`` mm that cuts every pupil sample while the chief ray passes, a 50:50 splitter, and a plane on
    each of its two beams; the reflected one after a fold mirror."""
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0.0, 0.0, 0.0), direction=X, power_W=P0)
    parts = [laser, B.aperture_stop("S1", position=(10.0, 0.0, 0.0), axis=X, radius_mm=1.0)]
    if second_stop is not None:
        parts.append(B.aperture_stop("S2", position=(20.0, 0.0, 0.0), axis=X, radius_mm=second_stop))
    parts.append(B.npbs("BS", position=(40.0, 0.0, 0.0), normal=(-S2, S2, 0.0), reflectance=0.5, clear_radius_mm=12.7))
    parts.append(B.mirror("M", position=(40.0, 30.0, 0.0), normal=(S2, -S2, 0.0), clear_radius_mm=12.7))
    planes = (B.observation_plane("T", position=(70.0, 0.0, 0.0), normal=X, radius_mm=10.0),
              B.observation_plane("R", position=(70.0, 30.0, 0.0), normal=X, radius_mm=10.0))
    targets = tuple(Target(target_id=f"p{p.observation_id}", metric="power_W", observation_id=p.observation_id, minimum=1e-5, unit="W")
                    for p in planes)
    return ProjectSnapshot(components=tuple(parts), observations=planes, targets=targets)


def _passed_first_stop() -> float:
    w2 = 1.0 + (10.0 / (math.pi / 633e-6)) ** 2
    return P0 * (1.0 - math.exp(-2.0 / w2))


def test_the_geometry_goes_on_through_both_branches_and_no_power_is_stated_behind():
    result = run(_scene())
    assert result.outcome.execution.outcome.value == "COMPLETED"
    # the ledger: the first stop's cut, and what reached the second stop booked once as light of unknown fate
    (line,) = result.power_budget.lines
    assert line.conserved and set(line.sinks) == {"APERTURE_TRUNCATION", "TERMINATED_OUT_OF_MODEL"}
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(_passed_first_stop(), rel=1e-9)
    assert line.sinks["APERTURE_TRUNCATION"] + line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(P0, rel=1e-12)
    (issue,) = [i for i in result.issues if i.numerical_context.get("power_upper_bound_W") is not None]
    assert (issue.code.value, issue.status.value, issue.location) == ("GAUSSIAN_SEVERE_TRUNCATION", "OUT_OF_MODEL_SCOPE", "surface:S2/stop")
    assert issue.numerical_context["power_upper_bound_W"] == line.sinks["TERMINATED_OUT_OF_MODEL"]
    # both branches reach their planes where the beam without the second stop does, with no power and the bound of what arrived
    reference = run(_scene(None))
    for plane_id in ("T", "R"):
        (arrival,) = arrivals(result, plane_id)
        (clear,) = arrivals(reference, plane_id)
        assert arrival.beam.position_mm == pytest.approx(clear.beam.position_mm, abs=1e-12)
        assert arrival.beam.direction == pytest.approx(clear.beam.direction, abs=1e-12)
        assert clear.beam.power_W == pytest.approx(0.5 * _passed_first_stop(), rel=1e-9)
        assert arrival.beam.power_W is None and arrival.detected_power_W is None
        assert arrival.power_upper_bound_W == line.sinks["TERMINATED_OUT_OF_MODEL"]  # (not halved: a bound of what arrived)
        plane = next(o for o in result.observations if o.observation_id == plane_id)
        assert (plane.total_power_W, plane.power_complete, plane.quantity_state.value) == (None, False, "UNKNOWN")
        assert all(s.power_W is None for s in plane.power_by_wavelength)
    assert sorted(p.termination for p in result.paths) == ["ESCAPED", "ESCAPED"] and all(p.final_power_W is None for p in result.paths)
    assert sorted(tuple(step for step in p.lineage if not step.startswith("S2/")) for p in result.paths) == sorted(
        tuple(p.lineage) for p in reference.paths)
    for target in result.targets:
        assert (target.value, target.assessment.value) == (None, "INDETERMINATE")
    # no exposure, pair or coherent sum states a number from the beam behind the stop
    behind = {"BS", "M"}
    for surface in result.exposures.surfaces:
        if surface.surface_id.split("/")[0] in behind:
            assert not [h for h in surface.hits if h.incident_power_W]
    assert not [r for r in result.interference if r.combined_power_collinear_W is not None]


def test_a_budget_that_cuts_the_branch_of_unknown_power_changes_nothing_known():
    """With one path allowed the reflected branch (no power of its own) is not traced; the known amounts are those of the
    unlimited run, and nothing known is reported as untraced."""
    full = run(_scene())
    limited = run(_scene(), numerical=NumericalConfig(max_paths=1))
    assert dict(limited.power_budget.lines[0].sinks) == dict(full.power_budget.lines[0].sinks)
    assert limited.power_budget.conserved
    assert len(limited.paths) == 1 and limited.paths[0].final_power_W is None
    assert_power_not_computed(limited, "T")
    assert (limited.outcome.execution.budgets.untraced_power_W or 0.0) == 0.0


def test_an_absorbing_plane_takes_the_beam_whose_power_is_not_computed():
    """The same beam onto an absorbing detector: the path ends there; the detector states no power, at most the bound."""
    scene = _scene()
    detector = B.observation_plane("T", position=(30.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(scene.model_copy(update={"observations": (detector,), "targets": (scene.targets[0],)}))
    obs, arrival = assert_power_not_computed(result, "T", state=None)
    assert [p.termination for p in result.paths] == ["DETECTED"]
    assert arrival.power_upper_bound_W == pytest.approx(_passed_first_stop(), rel=1e-9)
    (target,) = result.targets
    assert (target.value, target.assessment.value) == (None, "INDETERMINATE")
