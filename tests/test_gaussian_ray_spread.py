"""PHYSICS_SCOPE P4 assumption 2b: every arrival's Gaussian beam is checked against the Engine's own pupil rays.

* The bound is exact without aberration: through an ideal (slope-form, aberration-free) thin lens the ratio of the rays'
  RMS spread to the Gaussian bound never exceeds 1 -- for waists before, at and after the source, elliptical and M^2 > 1
  beams, planes before, at and far beyond the focus, and a tilted plane (two-ray decomposition of the Gaussian beam).
* A plano-convex singlet turned the wrong way round focuses a 2.2 mm beam into an aberration-dominated spot: the Engine
  withholds that arrival's Gaussian state (``GAUSSIAN_ABERRATION_DOMINATED``, ``OUT_OF_MODEL_SCOPE``), at the focus and at
  a plane far beyond it (through its Gaussian waist plane), while the power is still evaluated. The same lens the right
  way round with a small beam stays in scope, and the limit is ``NumericalConfig.gaussian_max_ray_spread_ratio``.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.validation.observations import gaussian_ray_spread
from os_optics.validation.pipeline import run_physics

from helpers import CFG, issue_codes, one_arrival, run

F_MM = 100.0
LENS_X = 50.0
WAVELENGTH = 780.0
UNCHECKED = NumericalConfig(gaussian_max_ray_spread_ratio=1e9)


def _thin_lens_project(*, waist_position_mm: float, w0y: float | None = None, m2: float = 1.0, planes=()) -> ProjectSnapshot:
    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=WAVELENGTH, waist_radius_mm=1.5, waist_radius_y_mm=w0y, m2=m2, waist_position_z_mm=waist_position_mm,
                    position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
            B.thin_lens("Lens", focal_length_mm=F_MM, clear_radius_mm=12.7, position=(LENS_X, 0.0, 0.0)),
        ),
        observations=tuple(planes),
    )


def _planes(focus_x: float):
    distances = (10.0, focus_x - 5.0, focus_x, focus_x + 0.5, focus_x + 60.0, 600.0)
    planes = [B.observation_plane(f"P{i}", position=(x, 0.0, 0.0), normal=(1.0, 0.0, 0.0)) for i, x in enumerate(distances)]
    tilt = math.radians(30.0)
    planes.append(B.observation_plane("tilted", position=(focus_x + 1.0, 0.0, 0.0), normal=(math.cos(tilt), math.sin(tilt), 0.0)))
    return planes


@pytest.mark.parametrize("waist_position_mm, w0y, m2", [
    (0.0, None, 1.0),          # waist (flat wavefront) at the source
    (-300.0, None, 1.0),       # diverging from a waist 300 mm behind the source
    (400.0, None, 1.0),        # converging toward a waist beyond the lens
    (0.0, 0.8, 1.0),           # elliptical beam
    (-100.0, None, 1.7),       # M^2 > 1
])
def test_without_aberration_the_rays_never_spread_wider_than_the_gaussian_bound(waist_position_mm, w0y, m2):
    project = _thin_lens_project(waist_position_mm=waist_position_mm, w0y=w0y, m2=m2, planes=_planes(LENS_X + F_MM))
    outcome = run_physics(project, CFG)
    spreads = [gaussian_ray_spread(rec, CFG) for rec in outcome.trace.arrivals]
    assert len(spreads) == len(project.observations) and all(s is not None for s in spreads)
    assert max(s.ratio for s in spreads) <= 1.0 + 1e-9
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in {i.code.value for i in outcome.issues}


def _singlet_project(*, flat_side_first: bool, waist_mm: float, plane_x: float) -> ProjectSnapshot:
    radius = 13.0  # N-BK7 plano-convex, f ~ 25 mm
    r1, r2 = (math.inf, -radius) if flat_side_first else (radius, math.inf)
    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=WAVELENGTH, waist_radius_mm=waist_mm, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
            B.singlet_lens("Lens", r1_mm=r1, r2_mm=r2, thickness_mm=5.0, clear_radius_mm=11.0, material="N-BK7", position=(0.0, 0.0, 0.0)),
        ),
        observations=(B.observation_plane("P", position=(plane_x, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),),
        targets=(
            Target(target_id="radius", metric="beam_radius_mm", observation_id="P", maximum=1.0, unit="mm"),
            Target(target_id="power", metric="power_W", observation_id="P", minimum=1e-9, unit="W"),
        ),
    )


def _gaussian_waist_x(flat_side_first: bool, waist_mm: float) -> float:
    """Where the Engine's own Gaussian puts the waist behind the lens (read on a plane 60 mm after the lens, with the check
    relaxed so the Gaussian state is reported)."""

    probe = run(_singlet_project(flat_side_first=flat_side_first, waist_mm=waist_mm, plane_x=60.0), numerical=UNCHECKED)
    g = one_arrival(probe, "P").beam.gaussian
    assert g is not None and g.valid
    return 60.0 - g.waist_distance_x_mm


def test_an_aberration_dominated_focus_withholds_the_gaussian_metrics_but_keeps_the_power():
    waist_x = _gaussian_waist_x(True, 2.2)
    result = run(_singlet_project(flat_side_first=True, waist_mm=2.2, plane_x=waist_x))
    arrival = one_arrival(result, "P")
    assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
    assert arrival.beam.gaussian.reason == "GAUSSIAN_ABERRATION_DOMINATED"
    assert arrival.pupil_rms_radius_mm > 0.01  # the rays' own spot: tens of micrometres, not the ~3 um Gaussian waist
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_ABERRATION_DOMINATED"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location == "observation:P"
    context = issue.numerical_context
    assert context["ray_spread_ratio"] > 10.0 and context["ray_spread_limit"] == CFG.gaussian_max_ray_spread_ratio
    assert context["geometric_rms_mm"] > 10.0 * context["gaussian_bound_rms_mm"]
    targets = {t.target_id: t for t in result.targets}
    assert targets["radius"].status.value == "OUT_OF_MODEL_SCOPE" and targets["radius"].value is None
    assert targets["power"].status.value == "PASS" and targets["power"].value > 0.0
    assert result.status.value == "OUT_OF_MODEL_SCOPE"


def test_the_waist_plane_check_catches_the_aberrated_waist_from_a_plane_far_beyond_it():
    result = run(_singlet_project(flat_side_first=True, waist_mm=2.2, plane_x=150.0))
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_ABERRATION_DOMINATED"]
    assert issue.numerical_context["at_waist"] is True and issue.numerical_context["axial_distance_mm"] < -100.0


def test_the_same_lens_the_right_way_round_with_a_small_beam_stays_in_scope():
    waist_x = _gaussian_waist_x(False, 0.5)
    for plane_x in (waist_x, waist_x + 100.0):
        result = run(_singlet_project(flat_side_first=False, waist_mm=0.5, plane_x=plane_x))
        assert "GAUSSIAN_ABERRATION_DOMINATED" not in issue_codes(result)
        assert one_arrival(result, "P").beam.gaussian.valid
    outcome = run_physics(_singlet_project(flat_side_first=False, waist_mm=0.5, plane_x=waist_x), CFG)
    (rec,) = outcome.trace.arrivals
    assert gaussian_ray_spread(rec, CFG).ratio < 0.5 * CFG.gaussian_max_ray_spread_ratio


def test_the_limit_is_the_numerical_configurations():
    waist_x = _gaussian_waist_x(True, 2.2)
    outcome = run_physics(_singlet_project(flat_side_first=True, waist_mm=2.2, plane_x=waist_x), CFG)
    (rec,) = outcome.trace.arrivals
    ratio = gaussian_ray_spread(rec, CFG).ratio
    lenient = NumericalConfig(gaussian_max_ray_spread_ratio=2.0 * ratio)
    result = run(_singlet_project(flat_side_first=True, waist_mm=2.2, plane_x=waist_x), numerical=lenient)
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in issue_codes(result)
    assert CFG.gaussian_max_ray_spread_ratio == 2.0
    with pytest.raises(ValueError):
        NumericalConfig(gaussian_max_ray_spread_ratio=1.0)  # at or below the aberration-free bound every beam would be flagged


def test_a_beam_clipped_on_one_side_stays_below_the_subset_bound():
    """A beam passing an ideal lens near its edge loses its outer rays on one side (87 of 96 arrive): without aberration
    the ratio of any subset of the pupil sample stays <= sqrt(2), below the limit."""

    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=WAVELENGTH, waist_radius_mm=1.5, position=(0.0, 11.6, 0.0), direction=(1.0, 0.0, 0.0)),
            B.thin_lens("Lens", focal_length_mm=F_MM, clear_radius_mm=12.7, position=(LENS_X, 0.0, 0.0)),
        ),
        observations=tuple(B.observation_plane(f"P{i}", position=(x, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
                           for i, x in enumerate((60.0, 149.0, 150.0, 151.0, 400.0))),
    )
    outcome = run_physics(project, CFG)
    assert all(rec.pupil_points.shape[0] < rec.pupil_total for rec in outcome.trace.arrivals)
    assert max(gaussian_ray_spread(rec, CFG).ratio for rec in outcome.trace.arrivals) <= math.sqrt(2.0)
    assert "GAUSSIAN_ABERRATION_DOMINATED" not in {i.code.value for i in outcome.issues}
