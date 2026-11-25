"""A central obstruction the chief ray falls on : the annulus around it is transmitted.

Before, the chief ray inside the obstruction was treated like a chief ray on an opaque rim: the whole beam was booked
CLIPPED and every pupil ray stopped, although the annulus carries most of the power. Now:

* the annulus's power is exact for the incident Gaussian: T = exp(-2 a^2 / w^2) - exp(-2 R^2 / w^2) for a centred beam
  (obstruction radius a, opening radius R, beam radius w at the stop); the obstruction's share 1 - exp(-2 a^2 / w^2) is
  booked CLIPPED;
* the pupil rays outside the obstruction go on as the annulus's geometric rays; the chief ray is an unpowered reference
  axis (``chief_ray_obstructed``), so its phase is withheld;
* the transmitted beam is not a Gaussian beam: Gaussian metrics are OUT_OF_MODEL_SCOPE (GAUSSIAN_CENTRAL_OBSTRUCTION);
  diffraction by the obstruction is not modelled;
* behind it the annulus is carried only while the whole envelope of the unobstructed Gaussian beam (it contains the
  annulus) passes each aperture or active area; when an aperture, an active area or the surfaces take part of it, the
  power that would continue is not known: ANNULAR_BEAM_NOT_FOLLOWED, and the targets behind it are UNKNOWN (never a power
  from where the pupil samples happen to land; the samples reach only 1.18 w).

References are independent: the closed-form annulus integral above, and the pupil ring radii rho_k = sqrt(-ln(1 - (k + 1/2)
/ K) / 2) w0 of the equal-power sampling (rays launched at the waist stay parallel, so they keep these radii).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import (
    ApertureStopInteraction,
    CircularAperture,
    EngineeringComponentSnapshot,
    PlaneSurface,
    ProjectSnapshot,
    Target,
)
from os_optics.errors import Status

from helpers import assert_power_not_computed, issue_codes, one_arrival, rayleigh_range, run

X = (1.0, 0.0, 0.0)
P_W = 1e-3
LAM, W0 = 633.0, 1.0
STOP_X, OUT_X = 10.0, 30.0
OPENING_MM = 4.0
RINGS, PER_RING = 8, 12
QUANTUM = 1e-12  # NumericalConfig.power_ledger_fraction_resolution


def _w(z_mm: float) -> float:
    return W0 * math.sqrt(1.0 + (z_mm / rayleigh_range(W0, LAM)) ** 2)


def _annulus(a: float, r: float, w: float) -> float:
    return math.exp(-2.0 * a * a / (w * w)) - math.exp(-2.0 * r * r / (w * w))


def _ring_radii() -> list[float]:
    return [W0 * math.sqrt(-0.5 * math.log(1.0 - (k + 0.5) / RINGS)) for k in range(RINGS)]


def _stop(instance_id: str, x: float, radius: float, obstruction: float = 0.0, plate_mm: float = 10.0) -> EngineeringComponentSnapshot:
    """A circular stop in a ``plate_mm`` wide opaque plate (light beyond the opening meets the plate; 0: a bare opening)."""
    surface = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=radius, obstruction_radius_mm=obstruction), mount_margin_mm=plate_mm,
                           medium_positive="air", medium_negative="air", interaction=ApertureStopInteraction())
    return EngineeringComponentSnapshot(instance_id=instance_id, role="aperture", pose=B._pose((x, 0.0, 0.0), X), surfaces=(surface,))


def _project(obstruction: float, *, offset_mm: float = 0.0, extra=(), targets=(), out_radius: float | None = None,
             absorbing: bool = False) -> ProjectSnapshot:
    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W0, position=(0.0, offset_mm, 0.0), direction=X)
    laser = laser.model_copy(update={"source": laser.source.model_copy(update={"pupil_ring_count": RINGS,
                                                                                "pupil_points_per_ring": PER_RING})})
    out = B.observation_plane("OUT", position=(OUT_X, 0.0, 0.0), normal=X, radius_mm=out_radius, absorbing=absorbing)
    return ProjectSnapshot(components=(laser, _stop("STOP", STOP_X, OPENING_MM, obstruction), *extra), observations=(out,),
                           targets=tuple(targets))


def _t(target_id: str, metric: str, unit: str, **kw) -> Target:
    return Target(target_id=target_id, metric=metric, observation_id="OUT", unit=unit, **kw)


@pytest.mark.parametrize("a", [0.5, 0.1])
def test_the_annulus_is_transmitted_with_its_exact_power(a):
    w = _w(STOP_X)
    transmitted = _annulus(a, OPENING_MM, w)
    result = run(_project(a, targets=(_t("power", "power_W", "W", target_value=P_W * transmitted, tolerance_rel=1e-9),
                                      _t("radius", "beam_radius_mm", "mm", maximum=10.0),
                                      _t("phase", "phase_rad", "rad", minimum=-4.0),
                                      _t("spot", "spot_rms_mm", "mm", maximum=10.0))))
    targets = {t.target_id: t for t in result.targets}
    # power through the annulus, exactly (a 0.1 mm obstruction passes 98 %, a 0.5 mm one 60.65 %; it used to be 0)
    assert targets["power"].status is Status.PASS
    arrival = one_arrival(result, "OUT")
    assert arrival.beam.power_W == pytest.approx(P_W * transmitted, rel=1e-9)
    (line,) = result.power_budget.lines
    assert line.sinks["CLIPPED"] == pytest.approx(P_W * (1.0 - math.exp(-2.0 * a * a / (w * w))), abs=P_W * QUANTUM)
    assert "APERTURE_TRUNCATION" not in line.sinks  # exp(-2 R^2 / w^2) ~ 1e-14 is below the ledger's resolution
    assert line.sinks["ESCAPED"] == pytest.approx(P_W * transmitted, rel=1e-9) and line.conserved
    assert [p.termination for p in result.paths] == ["ESCAPED"]
    # the annulus's geometric rays: the pupil rings outside the obstruction (rays launched at the waist stay parallel)
    radii = [r for r in _ring_radii() if r >= a]
    assert arrival.pupil_ray_count == RINGS * PER_RING and arrival.pupil_rays_arrived == len(radii) * PER_RING
    assert targets["spot"].value == pytest.approx(math.sqrt(sum(r * r for r in radii) / len(radii)), rel=1e-9)
    # the chief ray is a reference axis: no Gaussian metric, no phase; the Gaussian is out of model scope
    assert arrival.chief_ray_obstructed and arrival.beam.phase_rad is None
    assert arrival.beam.gaussian.valid is False and arrival.beam.gaussian.reason == "GAUSSIAN_CENTRAL_OBSTRUCTION"
    for target_id in ("radius", "phase"):
        assert (targets[target_id].status, targets[target_id].issue_codes) == (Status.OUT_OF_MODEL_SCOPE, ("GAUSSIAN_CENTRAL_OBSTRUCTION",))
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_CENTRAL_OBSTRUCTION"]
    assert issue.status is Status.OUT_OF_MODEL_SCOPE and issue.location == "surface:STOP/stop"
    assert issue.numerical_context["transmitted_fraction"] == pytest.approx(transmitted, rel=1e-9)
    assert issue.numerical_context["pupil_rays_continuing"] == len(radii) * PER_RING
    assert "diffraction by the obstruction is not modelled" in issue.message
    assert "CLIPPED" not in issue_codes(result)  # no "beam blocked" failure


def test_a_chief_ray_in_the_annulus_is_a_gaussian_truncation_as_before():
    result = run(_project(0.5, offset_mm=1.0))
    arrival = one_arrival(result, "OUT")
    assert not arrival.chief_ray_obstructed and arrival.beam.gaussian.valid
    assert "GAUSSIAN_CENTRAL_OBSTRUCTION" not in issue_codes(result) and "GAUSSIAN_TRUNCATION" in issue_codes(result)


def test_without_an_obstruction_nothing_changes():
    result = run(_project(0.0))
    arrival = one_arrival(result, "OUT")
    assert arrival.beam.power_W == pytest.approx(P_W, rel=1e-12) and not arrival.chief_ray_obstructed
    assert result.status is Status.PASS


def test_a_second_obstruction_behind_the_first_leaves_the_power_not_computed():
    """The envelope that bounds the annulus is the unobstructed beam, whose centre a second obstruction cuts: where the
    shadow of the first one lies is not tracked (it is the geometric shadow; diffraction fills it), so the power behind it
    is not computed. The rays go on: the plane behind records the arrival with no power, at most the annulus's."""
    result = run(_project(0.5, extra=(_stop("STOP2", 20.0, OPENING_MM, 0.5),), targets=(_t("power", "power_W", "W", maximum=1.0),)))
    _obs, arrival = assert_power_not_computed(result, "OUT")
    assert arrival.power_upper_bound_W == pytest.approx(P_W * _annulus(0.5, OPENING_MM, _w(STOP_X)), rel=1e-9)
    (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert issue.location == "surface:STOP2/stop"
    assert (result.targets[0].status, result.targets[0].value) == (Status.UNKNOWN, None)


def test_an_annulus_cut_further_on_goes_on_with_a_power_that_is_not_computed():
    """A 0.6 mm stop behind the obstruction cuts the annulus (its outer rings reach 1.18 mm): the power that passes it is
    not computed, so the plane behind it is UNKNOWN (never the full annulus, never 0 W). The rays the stop passes go on and
    reach the plane: an arrival with no power, at most the annulus's, booked once to light of unknown fate."""
    w = _w(STOP_X)
    result = run(_project(0.5, extra=(_stop("STOP2", 20.0, 0.6),), targets=(_t("power", "power_W", "W", maximum=1.0),)))
    _obs, arrival = assert_power_not_computed(result, "OUT")
    assert arrival.pupil_rays_arrived == sum(1 for r in _ring_radii() if 0.5 < r <= 0.6) * PER_RING
    (line,) = result.power_budget.lines
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(P_W * _annulus(0.5, OPENING_MM, w), rel=1e-9)
    (target,) = result.targets
    assert (target.status, target.value, target.issue_codes) == (Status.UNKNOWN, None, ("OUT_OF_MODEL_SCOPE",))
    (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert issue.status is Status.OUT_OF_MODEL_SCOPE and issue.location == "surface:STOP2/stop"
    assert issue.numerical_context["pupil_rays_cut"] == sum(1 for r in _ring_radii() if r > 0.6) * PER_RING


def test_a_detector_in_the_shadow_of_the_obstruction_is_not_a_known_zero():
    """A 0.3 mm absorbing detector on the axis behind a 0.5 mm obstruction: no pupil ray lands on it, but it lies inside the
    envelope of the beam (and diffraction puts light in the geometric shadow, the Poisson spot): the power on it is not
    known, never 0 W from where the samples land."""
    result = run(_project(0.5, out_radius=0.3, absorbing=True, targets=(_t("power", "power_W", "W", maximum=1e-9),)))
    (target,) = result.targets
    assert (target.status, target.value) == (Status.UNKNOWN, None)
    # the detector ends the path; what it detects of the annulus is not computed, at most the annulus's power
    obs, arrival = assert_power_not_computed(result, "OUT", state="BOUNDED")
    assert [p.termination for p in result.paths] == ["DETECTED"]
    assert obs.power_upper_bound_W == arrival.power_upper_bound_W == pytest.approx(P_W * _annulus(0.5, OPENING_MM, _w(STOP_X)), rel=1e-9)


def test_a_detector_that_takes_the_whole_annulus_detects_its_power_and_one_that_takes_part_states_none():
    """'Whole' is the envelope of the unobstructed beam: at most negligible_power_fraction (1e-9) of it outside (radius 4 mm:
    exp(-32) ~ 1e-14), or on an active area at most capture_min_power_fraction of the annulus's power beside it (3 mm:
    exp(-18) / 0.6065 ~ 2.5e-8: counted, with an INCOMPLETE_ENVELOPE warning stating the bound); 0.8 mm takes part of it."""
    w = _w(STOP_X)
    whole = run(_project(0.5, out_radius=4.0, absorbing=True, targets=(_t("power", "power_W", "W", minimum=0.0),)))
    assert whole.targets[0].value == pytest.approx(P_W * _annulus(0.5, OPENING_MM, w), rel=1e-9)
    almost = run(_project(0.5, out_radius=3.0, absorbing=True, targets=(_t("power", "power_W", "W", minimum=0.0),)))
    assert almost.targets[0].value == pytest.approx(P_W * _annulus(0.5, OPENING_MM, w), rel=1e-9)
    (warning,) = [i for i in almost.issues if i.code.value == "INCOMPLETE_ENVELOPE"]
    assert warning.status is Status.WARNING
    assert warning.numerical_context["annulus_power_beside_bound"] == pytest.approx(
        math.exp(-18.0 / _w(OUT_X) ** 2) / _annulus(0.5, OPENING_MM, w), rel=1e-6)
    part = run(_project(0.5, out_radius=0.8, absorbing=True, targets=(_t("power", "power_W", "W", minimum=0.0),)))
    (target,) = part.targets
    obs, arrival = assert_power_not_computed(part, "OUT", state="BOUNDED")
    assert obs.power_upper_bound_W == arrival.power_upper_bound_W == pytest.approx(P_W * _annulus(0.5, OPENING_MM, w), rel=1e-9)
    # (no value; the plane's proven interval [0, bound] decides a minimum of 0 W, nothing above it)
    assert (target.status, target.assessment.value, target.value) == (Status.PASS, "MET", None)
    (above,) = run(_project(0.5, out_radius=0.8, absorbing=True, targets=(_t("power", "power_W", "W", minimum=1e-4),))).targets
    assert (above.assessment.value, above.value) == ("INDETERMINATE", None)


def test_an_annulus_that_splits_between_surfaces_goes_on_with_the_rays_that_stay():
    """A bare 0.6 mm opening (no plate): the inner rays of the annulus meet it, the outer ones pass beside it. The light
    splits up: the power that goes on with the reference axis is not computed; the rays that part are retired and the
    others go on to the plane."""
    result = run(_project(0.5, extra=(_stop("RING", 20.0, 0.6, plate_mm=0.0),), targets=(_t("power", "power_W", "W", maximum=1.0),)))
    _obs, arrival = assert_power_not_computed(result, "OUT")
    assert arrival.pupil_rays_arrived == sum(1 for r in _ring_radii() if 0.5 < r <= 0.6) * PER_RING
    (path,) = result.paths
    assert path.geometric_path_length_mm == pytest.approx(OUT_X, abs=1e-9)  # (it goes on to the plane and beyond the scene)
    assert result.targets[0].status is Status.UNKNOWN
    (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert issue.location == "path:SRC#0" and "RING/stop" in issue.message
