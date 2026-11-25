"""Final-review (R2) diagnostics: known zero vs light the trace does not follow, the obstructed annulus bounded by its
envelope, a truncated beam at a later aperture, and the smaller semantics (B1, B2, M2, m8, n6).

* **B1** -- a plane no traced beam reaches is a known zero only where none of the light the trace does not follow can
  arrive: an untraced Fresnel reflection or coating transmission, the part of a beam booked ``CLIPPED`` at an aperture rim
  that passes the opening, a beam's tail on an active area it is not counted on. Such a plane is ``UNKNOWN`` with the cause
  and its power is ``null``; a branch whose free flight provably misses the plane keeps the known zero.
* **B2** -- behind a central obstruction the annulus is carried only while the whole envelope of the unobstructed beam
  passes each aperture / active area; never from where the pupil samples land, never without samples.
* **M2** -- a beam without a valid Gaussian (severely truncated) meets an aperture that cuts its rays: not followed.
* **m8 / n6** -- control readings honour the acceptance cone; a crossing / wavelength selection of a source that never
  makes it is ``TARGET_SELECTION_EMPTY``; a negligible below-floor remainder of a weak line does not make a summed power
  incomplete; the acceptance cone is two-sided; monitors spend the step budget.

References are independent of the Engine: the Fresnel equations with the SCHOTT Sellmeier formula of N-BK7, the Gaussian
beam radius ``w(z) = w0 sqrt(1 + (z / z_R)^2)``, the closed-form annulus ``exp(-2 a^2 / w^2) - exp(-2 R^2 / w^2)`` and the
power of an off-centre Gaussian inside a disc of radius ``a`` (beam axis at distance ``d`` from its centre),
``F(a; d, w) = int_0^a (4 r / w^2) exp(-2 (r^2 + d^2) / w^2) I0(4 r d / w^2) dr`` (Gauss-Legendre, 400 nodes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import (
    ApertureStopInteraction,
    CircularAperture,
    ConstantCoating,
    ControlSpec,
    EngineeringComponentSnapshot,
    JonesVector,
    MirrorInteraction,
    PlaneSurface,
    ProjectSnapshot,
    SensorSpec,
    Target,
)
from os_optics.control.readings import evaluate, sensed_power_W
from os_optics.errors import Status
from os_optics.validation.pipeline import CHIEF_RAYS, COMPLETE_PHYSICS, run_physics

from helpers import CFG, issue_codes, one_arrival, rayleigh_range, run

X = (1.0, 0.0, 0.0)
P_W = 1e-3


# ------------------------------------------------------------------------------------------------- independent references
def _w(w0: float, wavelength_nm: float, z_mm: float) -> float:
    return w0 * math.sqrt(1.0 + (z_mm / rayleigh_range(w0, wavelength_nm)) ** 2)


def _annulus(a: float, r: float, w: float) -> float:
    return math.exp(-2.0 * a * a / (w * w)) - math.exp(-2.0 * r * r / (w * w))


def _disc(a: float, d: float, w: float, nodes: int = 400) -> float:
    """Power fraction of a round Gaussian (radius w) inside a disc of radius a whose centre is d from the beam axis."""
    x, weights = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * a * (x + 1.0)
    f = (4.0 * r / (w * w)) * np.exp(-2.0 * (r * r + d * d) / (w * w)) * np.i0(4.0 * r * d / (w * w))
    return float(0.5 * a * np.sum(weights * f))


def _n_bk7(wavelength_nm: float) -> float:
    """SCHOTT Sellmeier formula of N-BK7 (B1..3, C1..3 from the SCHOTT data sheet)."""
    l2 = (wavelength_nm * 1e-3) ** 2
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _fresnel(n: float, aoi_rad: float) -> tuple[float, float]:
    ci = math.cos(aoi_rad)
    ct = math.sqrt(1.0 - (math.sin(aoi_rad) / n) ** 2)
    rs = (ci - n * ct) / (ci + n * ct)
    rp = (n * ci - ct) / (n * ci + ct)
    return rs * rs, rp * rp


# ----------------------------------------------------------------------------------------------------------- builders
def _laser(sid: str = "SRC", *, power: float = P_W, lam: float = 633.0, w0: float = 1.0, pos=(0.0, 0.0, 0.0), direction=X,
           rings: int = 8, per: int = 12, **kw):
    laser = B.laser(sid, wavelength_nm=lam, power_W=power, waist_radius_mm=w0, position=pos, direction=direction, **kw)
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pupil_ring_count": rings, "pupil_points_per_ring": per})})


def _stop(iid: str, x: float, radius: float, obstruction: float = 0.0, *, plate_mm: float = 10.0, y: float = 0.0):
    surface = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=radius, obstruction_radius_mm=obstruction), mount_margin_mm=plate_mm,
                           medium_positive="air", medium_negative="air", interaction=ApertureStopInteraction())
    return EngineeringComponentSnapshot(instance_id=iid, role="aperture", pose=B._pose((x, y, 0.0), X), surfaces=(surface,))


def _leaky_mirror(iid: str, position, normal):
    surface = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=12.7),
                           medium_positive="air", medium_negative="air",
                           interaction=MirrorInteraction(reflective_side="positive", coating=ConstantCoating(
                               reflectance_s=0.99, reflectance_p=0.99, transmittance_s=0.01, transmittance_p=0.01)))
    return EngineeringComponentSnapshot(instance_id=iid, role="mirror", pose=B._pose(position, B.unit(normal)), surfaces=(surface,))


def _t(target_id: str, observation_id: str, metric: str = "power_W", unit: str = "W", **kw) -> Target:
    return Target(target_id=target_id, metric=metric, observation_id=observation_id, unit=unit, **kw)


def _obs(result, observation_id: str):
    return next(o for o in result.observations if o.observation_id == observation_id)


def _assert_not_known(result, observation_id: str, code: str, untraced_W: float, rel: float):
    """The plane no traced beam reaches is no known zero: only light the probes followed reaches it, so its power is BOUNDED
    between 0 and its upper bound with no value, ``code`` names that light, and ``followed_light`` lists it."""
    obs = _obs(result, observation_id)
    assert obs.status is Status.PASS and not obs.arrivals and not obs.power_complete
    assert obs.quantity_state.value == "BOUNDED" and obs.power_lower_bound_W == 0.0 and obs.power_upper_bound_W >= obs.untraced_power_W
    assert obs.followed_light and math.fsum(f.power_W for f in obs.followed_light) == pytest.approx(obs.untraced_power_W, rel=1e-12)
    assert obs.total_power_W is None
    assert [(s.power_W, s.complete) for s in obs.power_by_wavelength] == [(None, False)]
    assert obs.untraced_power_W == pytest.approx(untraced_W, rel=rel)
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED" and i.location == f"observation:{observation_id}"]
    assert issue.status is Status.NOT_APPLICABLE and code in issue.numerical_context["untraced_causes"]  # (a notice)
    # (followed untraced light is UNTRACED_LIGHT_BOUNDED, not OUT_OF_MODEL_SCOPE, in the plane's issue and its targets alike)
    for target in result.targets:
        if target.observation_id == observation_id:
            assert (target.status, target.value) == (Status.UNKNOWN, None) and code in target.issue_codes


# ---------------------------------------------------------------------------------------------------------- B1
def test_a_pick_off_detector_in_an_untraced_fresnel_reflection_is_not_a_known_zero():
    """A detector in the first-surface reflection of an uncoated 10 deg N-BK7 window (reflections not traced): it used to
    read a known 0 W and PASS a 1 nW maximum; the window's reflections (4.4 % + 3.5 %) may reach it: UNKNOWN, power null."""
    th = math.radians(10.0)
    axis = (math.cos(th), math.sin(th), 0.0)
    n = np.array(axis)
    r = np.array(X) - 2.0 * float(np.array(X) @ n) * n
    pick_pos = tuple(np.array((50.0, 0.0, 0.0)) - 1.5 * n + 60.0 * r)
    source = _laser(w0=0.5)
    main = B.observation_plane("MAIN", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    pick = B.observation_plane("PICK", position=pick_pos, normal=tuple(r), radius_mm=5.0, absorbing=True)
    window = B.window("WIN", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(50.0, 0.0, 0.0), axis=axis)
    result = run(ProjectSnapshot(components=(source, window), observations=(pick, main),
                                 targets=(_t("pick_max", "PICK", maximum=1e-9),)))
    # independent: Fresnel at 10 deg for the source's own s / p mix (plane of incidence x-y: s along z)
    free = one_arrival(run(ProjectSnapshot(components=(source,), observations=(main,))), "MAIN").beam
    field = complex(*free.polarization.ex) * np.array(free.basis.e1) + complex(*free.polarization.ey) * np.array(free.basis.e2)
    s_share = abs(field[2]) ** 2 / float(np.sum(np.abs(field) ** 2))
    rs, rp = _fresnel(_n_bk7(633.0), th)
    ps, pp = s_share * P_W, (1.0 - s_share) * P_W
    reflected = rs * ps + rp * pp + rs * (1.0 - rs) * ps + rp * (1.0 - rp) * pp  # first face, then the second face from inside
    (line,) = result.power_budget.lines
    assert line.sinks["UNTRACED_REFLECTION"] == pytest.approx(reflected, rel=1e-6)
    # both reflections reach the pick-off (followed through the window: the second leaves through the first face, T = 1 - R)
    # (and, followed to every order, each further internal bounce: the plate's reflectance R + T^2 R / (1 - R^2))
    arriving = sum(share * (r + (1.0 - r) ** 2 * r / (1.0 - r * r)) for share, r in ((ps, rs), (pp, rp)))
    _assert_not_known(result, "PICK", "UNTRACED_LIGHT_BOUNDED", arriving, rel=1e-6)
    assert result.status is Status.UNKNOWN  # (only a maximum on PICK: it used to PASS)
    # the plane the traced beam reaches is unchanged: its power counts the traced beam (the declared approximation)
    main_obs = _obs(result, "MAIN")
    assert not main_obs.power_complete and main_obs.total_power_W == pytest.approx((1 - rs) ** 2 * ps + (1 - rp) ** 2 * pp, rel=1e-6)
    assert main_obs.quantity_state.value == "BOUNDED" and main_obs.power_upper_bound_W > main_obs.total_power_W


def _leak_project(pd_position, *targets):
    source = _laser(w0=0.5)
    mirror = _leaky_mirror("M", (50.0, 0.0, 0.0), (-1.0, 1.0, 0.0))
    dump = B.observation_plane("DUMP", position=(50.0, 50.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=10.0, absorbing=True)
    pd = B.observation_plane("PD", position=pd_position, normal=X, radius_mm=2.0, absorbing=True)
    return ProjectSnapshot(components=(source, mirror), observations=(pd, dump), targets=targets)


def test_a_leak_photodiode_behind_a_partly_transmitting_mirror_is_not_a_known_zero():
    """R 0.99 / T 0.01 mirror: the 1 % the coating transmits (not followed) goes straight on to the photodiode."""
    result = run(_leak_project((70.0, 0.0, 0.0), _t("leak_max", "PD", maximum=1e-9)))
    _assert_not_known(result, "PD", "UNTRACED_LIGHT_BOUNDED", 0.01 * P_W, rel=1e-12)
    assert _obs(result, "DUMP").total_power_W == pytest.approx(0.99 * P_W, rel=1e-12)


def test_an_untraced_branch_whose_free_flight_misses_the_plane_keeps_the_known_zero():
    """The same 1 % leaves along +x from (50, 0, 0); a 2 mm photodiode 30 mm to the side is out of its envelope (6 w is
    about 3 mm there) and nothing redirects it: a known zero, the maximum passes."""
    result = run(_leak_project((70.0, 30.0, 0.0), _t("leak_max", "PD", maximum=1e-9)))
    (target,) = result.targets
    assert (target.status, target.value, target.issue_codes) == (Status.PASS, 0.0, ("NO_TRACED_ARRIVAL",))
    obs = _obs(result, "PD")
    assert obs.power_complete and obs.total_power_W == 0.0 and [(s.power_W, s.complete) for s in obs.power_by_wavelength] == [(0.0, True)]


@pytest.mark.parametrize("offset_mm", [1.1, 1.5])
def test_the_part_of_a_clipped_beam_that_passes_the_opening_reaches_the_detector(offset_mm):
    """The chief ray falls on the plate beside a 1 mm opening (CLIPPED, a FAIL); the Gaussian power inside the opening goes
    on as the pupil rays that pass, the one nearest their centroid its reference ray: the detector behind it reads
    that power (not a known zero, and no longer only untraced light), and a 1 nW maximum fails."""
    source = _laser(w0=1.0, pos=(0.0, offset_mm, 0.0))
    det = B.observation_plane("DET", position=(40.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(source, _stop("STOP", 20.0, 1.0)), observations=(det,),
                                 targets=(_t("det_max", "DET", maximum=1e-9),)))
    passing = P_W * _disc(1.0, offset_mm, _w(1.0, 633.0, 20.0))
    assert passing == pytest.approx(P_W * (0.327 if offset_mm == 1.1 else 0.1133), rel=3e-3)  # (the review's values)
    (arrival,) = _obs(result, "DET").arrivals
    assert arrival.reference_ray_promoted_at == ("STOP/stop",)
    assert arrival.beam.power_W == pytest.approx(passing, rel=1e-6)
    assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
    assert arrival.detected_power_W is not None and arrival.detected_power_W <= arrival.beam.power_W * (1.0 + 1e-12)
    (target,) = result.targets
    assert target.status is Status.FAIL
    assert "OBSERVATION_NOT_REACHED" not in issue_codes(result)
    assert result.status is Status.FAIL  # the CLIPPED issue itself
    assert result.power_budget.conserved


def test_a_beam_tail_below_the_capture_threshold_is_not_a_known_zero():
    """A beam 2.2 mm beside a 0.5 mm detector: 1.43e-4 of it falls on the area, below capture_min_power_fraction (1e-3), so
    it is not an arrival; but that tail is not negligible: UNKNOWN (INCOMPLETE_ENVELOPE) with its power, not 0 W."""
    source = _laser(w0=1.0, pos=(0.0, 2.2, 0.0))
    det = B.observation_plane("DET", position=(40.0, 0.0, 0.0), normal=X, radius_mm=0.5, absorbing=True)
    result = run(ProjectSnapshot(components=(source,), observations=(det,), targets=(_t("det_min", "DET", minimum=1e-8),)))
    tail = _disc(0.5, 2.2, _w(1.0, 633.0, 40.0))
    assert 1e-4 < tail < CFG.capture_min_power_fraction
    _assert_not_known(result, "DET", "INCOMPLETE_ENVELOPE", P_W * tail, rel=1e-6)
    # far beside it (12 w) the tail is far below the line's closure tolerance: a known zero
    far = run(ProjectSnapshot(components=(_laser(w0=1.0, pos=(0.0, 12.0, 0.0)),), observations=(det,),
                              targets=(_t("det_max", "DET", maximum=1e-9),)))
    assert (far.targets[0].status, far.targets[0].value) == (Status.PASS, 0.0)


def _tilted_detector(half_angle_deg: float, tilt_deg: float = 25.0):
    tilt = math.radians(tilt_deg)
    return B.observation_plane("DET", position=(40.0, 0.0, 0.0), normal=(math.cos(tilt), math.sin(tilt), 0.0), radius_mm=0.5,
                               absorbing=True, acceptance_half_angle_deg=half_angle_deg)


def _tail_run(offset_mm: float, half_angle_deg: float, *, w0: float = 1.0):
    return run(ProjectSnapshot(components=(_laser(w0=w0, pos=(0.0, offset_mm, 0.0)),), observations=(_tilted_detector(half_angle_deg),),
                               targets=(_t("det_max", "DET", maximum=1e-9),)))


def test_a_beam_tail_outside_the_detector_acceptance_cone_stays_a_known_zero():
    """633 nm, 1 mW, w0 1 mm; a 0.5 mm detector at 40 mm whose normal is tilted 25 deg, acceptance half angle 10 deg; a
    1 nW maximum. Centred, the beam reaches the area at 25 deg: counted, not detected (0 W, complete, PASS). Offset 2.2 mm,
    only its Gaussian tail falls on the area (below the capture fraction); that tail travels within ~1e-3 deg of the chief
    direction (collimated: R ~ 6.3 m at 40 mm, spread 6 x 0.2 mrad), i.e. at 25 deg, outside the cone like the centred beam:
    the detector cannot see it, so it is a known zero too (it used to be UNKNOWN with ~1.03e-7 W of untraced power)."""
    centred = _tail_run(0.0, 10.0)
    assert one_arrival(centred, "DET").detected_power_W == 0.0
    obs = _obs(centred, "DET")
    assert obs.power_complete and obs.total_power_W == 0.0
    assert (centred.targets[0].status, centred.targets[0].value) == (Status.PASS, 0.0)
    offset = _tail_run(2.2, 10.0)
    obs = _obs(offset, "DET")
    assert not obs.arrivals and obs.power_complete and obs.total_power_W == 0.0 and not obs.untraced_power_W
    assert [(s.power_W, s.complete) for s in obs.power_by_wavelength] == [(0.0, True)]
    (target,) = offset.targets
    assert (target.status, target.value) == (Status.PASS, 0.0)


def test_a_beam_tail_inside_or_at_the_edge_of_the_acceptance_cone_may_still_arrive():
    """The same tail with a 30 deg cone (25 deg is inside) is light the detector sees: UNKNOWN with the tail's power, as
    without a cone. With the cone's edge at the chief's 25 deg the tail's directions straddle it: not decided, UNKNOWN.
    A strongly diverging beam (w0 10 um: R ~ 40 mm, theta_0 ~ 1.2 deg) whose chief ray is 1 deg outside a 24 deg cone
    reaches the area's points along wavefront normals tilted by up to ~3.4 deg (plus its angular spread): not provably
    outside, UNKNOWN."""
    for half_angle in (30.0, 25.0):
        result = _tail_run(2.2, half_angle)
        obs = _obs(result, "DET")
        assert obs.quantity_state.value == "BOUNDED" and obs.total_power_W is None and obs.untraced_power_W > 0.0
        assert (result.targets[0].status, result.targets[0].value) == (Status.UNKNOWN, None)
    reference = _obs(_tail_run(2.2, 30.0), "DET").untraced_power_W
    no_cone = run(ProjectSnapshot(components=(_laser(w0=1.0, pos=(0.0, 2.2, 0.0)),),
                                  observations=(_tilted_detector(10.0).model_copy(update={"acceptance_half_angle_deg": None}),)))
    assert _obs(no_cone, "DET").untraced_power_W == pytest.approx(reference, rel=1e-12)
    diverging = _tail_run(1.7, 24.0, w0=0.01)
    obs = _obs(diverging, "DET")
    assert not obs.arrivals and obs.quantity_state.value == "BOUNDED" and obs.untraced_power_W > 0.0
    assert (diverging.targets[0].status, diverging.targets[0].value) == (Status.UNKNOWN, None)


def test_a_target_on_a_line_whose_untraced_light_cannot_arrive_stays_a_known_zero():
    """Per source line: the leak photodiode's 1 % comes from source A; a target that selects source B, which a dump stops
    before the mirror, is a known zero although A's leak may reach the plane."""
    a = _laser("A", w0=0.5)
    b = _laser("B", w0=0.5, pos=(0.0, -40.0, 0.0))
    stop_b = B.observation_plane("STOPB", position=(10.0, -40.0, 0.0), normal=X, radius_mm=5.0, absorbing=True)
    project = _leak_project((70.0, 0.0, 0.0), _t("leak_a", "PD", source_id="A", maximum=1e-9),
                            _t("leak_b", "PD", source_id="B", maximum=1e-9))
    project = project.model_copy(update={"components": (a, b, project.components[1]), "observations": (*project.observations, stop_b)})
    result = run(project)
    t = {x.target_id: x for x in result.targets}
    assert (t["leak_a"].status, t["leak_a"].value) == (Status.UNKNOWN, None)
    assert (t["leak_b"].status, t["leak_b"].value, t["leak_b"].issue_codes) == (Status.PASS, 0.0, ("NO_TRACED_ARRIVAL",))


# ------------------------------------------------------------------------------------------------------------- B2 (S02)
def _obstructed(obstruction: float, *, lam=633.0, w0=1.0, stop_x=10.0, extra=(), out_x=30.0, out_radius=None, absorbing=False,
                targets=(), out_pos=None):
    out = B.observation_plane("OUT", position=out_pos or (out_x, 0.0, 0.0), normal=X, radius_mm=out_radius, absorbing=absorbing)
    return ProjectSnapshot(components=(_laser(lam=lam, w0=w0), _stop("STOP", stop_x, 4.0, obstruction), *extra), observations=(out,),
                           targets=tuple(targets))


def _ends_not_followed(result, surface_id: str, annulus_W: float):
    """The annulus's power is not computed from ``surface_id`` on: the issue there states the bound (the annulus's power),
    the ledger books it once to light of unknown fate, the rays go on with no power (the path ends where the geometry does),
    and no target on the plane has a value or fails."""
    (path,) = result.paths
    assert path.termination in ("ESCAPED", "DETECTED") and path.final_power_W is None
    (line,) = result.power_budget.lines
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(annulus_W, rel=1e-9) and line.conserved
    (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert issue.location == f"surface:{surface_id}"
    assert issue.numerical_context["power_upper_bound_W"] == pytest.approx(annulus_W, rel=1e-9)
    for target in result.targets:
        assert target.value is None and target.status is not Status.FAIL
    assert _obs(result, "OUT").total_power_W is None
    for arrival in _obs(result, "OUT").arrivals:
        assert arrival.beam.power_W is None and arrival.power_upper_bound_W == pytest.approx(annulus_W, rel=1e-9)


def test_far_field_annulus_behind_a_later_stop_is_not_followed():
    """w0 0.5 mm at 1064 nm, 0.3 mm obstruction at z = 2000 mm (w = 1.444 mm), then a 0.6 mm stop: the samples (launched at
    the waist, kept parallel) all pass it and the annulus used to be carried whole (0.917 mW); scaled with the beam only
    0.207 mW would pass. The envelope does not fit the stop: not followed."""
    w = _w(0.5, 1064.0, 2000.0)
    result = run(_obstructed(0.3, lam=1064.0, w0=0.5, stop_x=2000.0, out_x=2020.0, extra=(_stop("STOP2", 2010.0, 0.6),),
                             targets=(_t("p", "OUT", maximum=1.0),)))
    _ends_not_followed(result, "STOP2/stop", P_W * _annulus(0.3, 4.0, w))
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_CENTRAL_OBSTRUCTION"]
    assert issue.numerical_context["transmitted_fraction"] == pytest.approx(_annulus(0.3, 4.0, w), rel=1e-9)


def test_an_obstruction_wider_than_every_sampled_ring_leaves_the_annulus_to_its_bounds():
    """A 0.7 mm obstruction covers every pupil ring (the outermost at 0.589 mm): no ray follows the annulus (62.5 % of the
    power), so its bounds alone decide. Its exact power is booked at the obstruction; the 50 um pinhole behind it takes part
    of the envelope, so nothing behind it is a value (a pinhole and a detector in the shadow used to detect 0.625 mW); an
    unbounded plane receives the annulus's exact power."""
    w = _w(0.5, 1064.0, 2000.0)
    result = run(_obstructed(0.7, lam=1064.0, w0=0.5, stop_x=2000.0, out_x=2020.0, out_radius=0.05, absorbing=True,
                             extra=(_stop("PIN", 2010.0, 0.05),), targets=(_t("p", "OUT", maximum=1.0),)))
    _ends_not_followed(result, "PIN/stop", P_W * _annulus(0.7, 4.0, w))
    free = run(_obstructed(0.7, lam=1064.0, w0=0.5, stop_x=2000.0, out_x=2020.0, targets=(_t("p", "OUT", minimum=0.0),)))
    assert free.targets[0].value == pytest.approx(P_W * _annulus(0.7, 4.0, w), rel=1e-9)
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_CENTRAL_OBSTRUCTION"]
    assert issue.numerical_context["transmitted_fraction"] == pytest.approx(_annulus(0.7, 4.0, w), rel=1e-9)
    assert issue.numerical_context["pupil_rays_continuing"] == 0


@pytest.mark.parametrize("absorbing", [False, True])
def test_a_stop_or_detector_that_takes_the_outer_tail_of_the_annulus_is_not_followed(absorbing):
    """Near field, 0.5 mm obstruction, then a 1.2 mm stop (or a 1.2 mm detector): every sample passes (they reach 1.18 mm),
    but 0.0563 mW of the annulus lies beyond 1.2 mm (0.5504 vs 0.6065 mW): not followed, never the whole annulus."""
    w = _w(1.0, 633.0, 10.0)
    assert _annulus(0.5, 1.2, w) == pytest.approx(0.550396, abs=1e-6)
    if absorbing:
        result = run(_obstructed(0.5, out_radius=1.2, absorbing=True, targets=(_t("p", "OUT", maximum=1.0),)))
        surface = "observation/OUT"
    else:
        result = run(_obstructed(0.5, extra=(_stop("STOP2", 20.0, 1.2),), targets=(_t("p", "OUT", maximum=1.0),)))
        surface = "STOP2/stop"
    _ends_not_followed(result, surface, P_W * _annulus(0.5, 4.0, w))


def test_without_pupil_rays_the_annulus_is_carried_by_its_bounds_alone():
    """run_physics without pupil rays (control readings, probes, the network recorder) has no ray of the annulus: the
    envelope bound decides alone. A 0.1 mm pinhole on the axis takes part of it (it used to detect the whole annulus,
    0.607 mW): not followed; a 10 mm detector takes all of it: the exact annulus, as with pupil rays (follow-up N2)."""
    w = _w(1.0, 633.0, 10.0)
    det = B.observation_plane("OUT", position=(30.0, 0.0, 0.0), normal=X, radius_mm=0.1, absorbing=True)
    project = ProjectSnapshot(components=(_laser(), _stop("STOP", 10.0, 4.0, 0.5), _stop("PIN", 20.0, 0.1)), observations=(det,))
    out = run_physics(project, CFG, plan=CHIEF_RAYS)
    # (the power behind the pinhole is not computed; the reference axis goes on to the detector, which states no power)
    assert [(p.termination, p.termination_surface_id, p.final_power_W) for p in out.trace.paths] == [("DETECTED", "observation/OUT", None)]
    (arrival,) = out.observations[0].arrivals
    assert arrival.beam.power_W is None and arrival.power_upper_bound_W == pytest.approx(P_W * _annulus(0.5, 4.0, w), rel=1e-9)
    assert out.observations[0].total_power_W is None
    assert [i.location for i in out.trace.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"] == ["surface:PIN/stop"]
    wide = det.model_copy(update={"aperture": CircularAperture(radius_mm=10.0)})
    for plan in (COMPLETE_PHYSICS.without("beam_pairs"), CHIEF_RAYS):  # (with pupil rays and without)
        whole = run_physics(ProjectSnapshot(components=(_laser(), _stop("STOP", 10.0, 4.0, 0.5)), observations=(wide,)), CFG,
                            plan=plan)
        (arrival,) = whole.observations[0].arrivals
        assert arrival.detected_power_W == pytest.approx(P_W * _annulus(0.5, 4.0, w), rel=1e-12)


def test_an_annulus_focused_by_a_lens_is_detected_whole_only_where_the_envelope_fits():
    """Obstruction 0.5 mm, f = 100 mm thin lens 10 mm behind it: a 0.2 mm detector at the focus (about 10 focal spot
    radii) takes the whole envelope, so it detects the annulus's exact power; a 0.3 mm detector halfway (envelope ~0.5 mm)
    takes part of it: not followed (it used to detect the whole annulus or nothing, from the samples)."""
    w = _w(1.0, 633.0, 10.0)
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=10.0, position=(20.0, 0.0, 0.0))
    focus = run(_obstructed(0.5, extra=(lens,), out_x=120.0, out_radius=0.2, absorbing=True, targets=(_t("p", "OUT", minimum=0.0),)))
    assert focus.targets[0].value == pytest.approx(P_W * _annulus(0.5, 4.0, w), rel=1e-9)
    halfway = run(_obstructed(0.5, extra=(lens,), out_x=70.0, out_radius=0.3, absorbing=True, targets=(_t("p", "OUT", minimum=0.0),)))
    _ends_not_followed(halfway, "observation/OUT", P_W * _annulus(0.5, 4.0, w))


def test_an_obstruction_at_a_focus_stops_every_sample_and_is_not_followed():
    """A 10 um obstruction at the focus (w_f = 20 um): the annulus carries 61 % of the power but every ray converges onto
    the obstruction, so its bounds alone decide; a 50 um detector 30 mm behind on the axis (the envelope is ~6 mm there)
    used to detect 0.611 mW: not followed."""
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=10.0, position=(20.0, 0.0, 0.0))
    det = B.observation_plane("OUT", position=(150.0, 0.0, 0.0), normal=X, radius_mm=0.05, absorbing=True)
    result = run(ProjectSnapshot(components=(_laser(), lens, _stop("STOP", 120.0, 4.0, 0.01)), observations=(det,),
                                 targets=(_t("p", "OUT", maximum=1.0),)))
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_CENTRAL_OBSTRUCTION"]
    w_focus = 633.0e-6 * 100.0 / (math.pi * 1.0)
    assert issue.numerical_context["transmitted_fraction"] == pytest.approx(math.exp(-2.0 * 0.01**2 / w_focus**2), rel=1e-3)
    _ends_not_followed(result, "observation/OUT", P_W * issue.numerical_context["transmitted_fraction"])


@pytest.mark.parametrize("phi_deg", [0.0, 7.5, 15.0])
def test_a_small_detector_inside_the_lit_annulus_is_never_a_dark_shadow(phi_deg):
    """A 50 um detector at r = 0.7 mm, between the samples: about 1.9 uW falls on it; it used to read a known 0 W."""
    y, z = 0.7 * math.cos(math.radians(phi_deg)), 0.7 * math.sin(math.radians(phi_deg))
    result = run(_obstructed(0.5, out_pos=(30.0, y, z), out_radius=0.05, absorbing=True, targets=(_t("p", "OUT", maximum=1e-9),)))
    _ends_not_followed(result, "observation/OUT", P_W * _annulus(0.5, 4.0, _w(1.0, 633.0, 10.0)))


def test_a_transparent_monitor_that_takes_part_of_the_annulus_is_not_followed():
    """A transparent 0.8 mm monitor counts a crossing beam whole; behind an obstruction part of the annulus (0.33 of 0.61 mW)
    falls on it, so whether and how much it receives is not known: not followed (an unbounded monitor counts the whole
    annulus, ``test_central_obstruction.py``)."""
    result = run(_obstructed(0.5, out_radius=0.8, targets=(_t("p", "OUT", maximum=1.0),
                                                          _t("f", "OUT", "active_area_power_fraction", "1", minimum=0.0))))
    _ends_not_followed(result, "observation/OUT", P_W * _annulus(0.5, 4.0, _w(1.0, 633.0, 10.0)))


# ------------------------------------------------------------------------------------------------------------------- M2
def test_a_severely_truncated_beam_at_a_later_aperture_continues_on_its_pupil_samples():
    """The chief ray passes beside a 0.5 mm obstruction (beam 0.8 mm off centre): the transmitted fraction is exact (0.85505)
    and the Gaussian is out of scope. A 1.0 mm stop on the beam behind it passes 0.7322 mW (an independent integration of the
    beam over the stop less the obstruction's shadow); the beam continues on the share of its pupil samples the stop passes
    (GEOMETRIC_POWER_BASIS), whose sampling resolution covers that value, and the stop's cut is booked as aperture
    truncation."""
    w = _w(1.0, 633.0, 10.0)
    transmitted = _disc(4.0, 0.8, w) - _disc(0.5, 0.8, w)
    assert transmitted == pytest.approx(0.8550456033, abs=1e-9)
    alone = run(ProjectSnapshot(components=(_laser(pos=(0.0, 0.8, 0.0)), _stop("STOP", 10.0, 4.0, 0.5)),
                                observations=(B.observation_plane("OUT", position=(30.0, 0.0, 0.0), normal=X),),
                                targets=(_t("p", "OUT", maximum=1.0),)))
    assert alone.targets[0].value == pytest.approx(P_W * transmitted, rel=1e-9)
    result = run(ProjectSnapshot(components=(_laser(pos=(0.0, 0.8, 0.0)), _stop("STOP", 10.0, 4.0, 0.5), _stop("STOP2", 20.0, 1.0, y=0.8)),
                                 observations=(B.observation_plane("OUT", position=(30.0, 0.0, 0.0), normal=X),),
                                 targets=(_t("p", "OUT", maximum=1.0),)))
    (path,) = result.paths
    assert path.termination == "ESCAPED"
    (arrival,) = result.observations[0].arrivals
    reference_W = 0.7322142e-3  # scipy dblquad of the beam over the 1.0 mm stop less the 0.5 mm obstruction's shadow
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    assert abs(arrival.beam.power_W - reference_W) <= arrival.power_sampling_resolution_W
    (line,) = result.power_budget.lines
    assert line.sinks["ESCAPED"] + line.sinks["APERTURE_TRUNCATION"] == pytest.approx(P_W, rel=1e-12)
    assert result.targets[0].status is Status.PASS and result.targets[0].value == arrival.beam.power_W
    issues = [i for i in result.issues if i.code.value == "GEOMETRIC_POWER_BASIS" and i.location == "surface:STOP2/stop"]
    assert len(issues) == 1


# ------------------------------------------------------------------------------------------------------------------- m8
def test_control_readings_honour_the_acceptance_cone():
    """A beam at 20 deg on an absorbing detector with a 10 deg cone is not detected (0 W): the control reading of its power
    is 0 (the sensor is lost), not the incident 1 mW."""
    d = (math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0)
    source = _laser(w0=0.5, pos=(0.0, -100.0 * math.tan(math.radians(20.0)), 0.0), direction=d)
    det = B.observation_plane("DET", position=(100.0, 0.0, 0.0), normal=X, radius_mm=5.0, absorbing=True, acceptance_half_angle_deg=10.0)
    project = ProjectSnapshot(components=(source,), observations=(det,))
    out = run_physics(project, CFG, plan=CHIEF_RAYS)
    (arrival,) = out.observations[0].arrivals
    assert arrival.detected_power_W == 0.0 and sensed_power_W(arrival) == 0.0
    spec = ControlSpec(actuator_refs=("SRC/x",), sensors=(SensorSpec(sensor_id="p", observation_id="DET", quantity="power_W", target_value=0.0),))
    reading = evaluate(project, spec, {}, CFG)
    assert reading.lost and reading.detected_fraction == 0.0
    inside = det.model_copy(update={"acceptance_half_angle_deg": 30.0})
    reading = evaluate(project.model_copy(update={"observations": (inside,)}), spec, {}, CFG)
    assert not reading.lost and reading.values[0] == pytest.approx(P_W, rel=1e-12)


def _retro_scene(*targets):
    """Source A crosses MON once; source B is sent back by a mirror and crosses it twice (crossing indices 0, 1)."""
    a = _laser("A", w0=0.5)
    b = _laser("B", w0=0.5, pos=(0.0, 5.0, 0.0))
    mirror = B.mirror("M", position=(100.0, 5.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=2.0)
    mon = B.observation_plane("MON", position=(50.0, 0.0, 0.0), normal=X)
    return ProjectSnapshot(components=(a, b, mirror), observations=(mon,), targets=targets)


@pytest.mark.parametrize("extra", [{}, {"wavelength_nm": 633.0}])
def test_a_crossing_index_the_selected_source_never_makes_is_an_empty_selection(extra):
    """crossing_index 1 of source A does not occur (only B crosses MON twice): TARGET_SELECTION_EMPTY, not a known 0 W."""
    result = run(_retro_scene(_t("a1", "MON", source_id="A", maximum=1e-9, parameters={"crossing_index": 1}, **extra),
                              _t("b1", "MON", source_id="B", minimum=0.0, parameters={"crossing_index": 1}, **extra)))
    t = {x.target_id: x for x in result.targets}
    assert (t["a1"].status, t["a1"].value, t["a1"].issue_codes) == (Status.UNKNOWN, None, ("TARGET_SELECTION_EMPTY",))
    assert "crossing indices 0" in t["a1"].message
    assert t["b1"].status is Status.PASS and t["b1"].value > 0.0


def test_an_unknown_plane_exports_null_power_and_a_known_dark_plane_zero():
    """In the exported JSON a plane where nothing arrives and the power is not known has null powers (a reader of the value
    alone must not see 0 W); a known dark plane keeps 0.0."""
    unknown = run(_leak_project((70.0, 0.0, 0.0))).model_dump(mode="json")
    dark = run(_leak_project((70.0, 30.0, 0.0))).model_dump(mode="json")
    pd_unknown = next(o for o in unknown["observations"] if o["observation_id"] == "PD")
    pd_dark = next(o for o in dark["observations"] if o["observation_id"] == "PD")
    assert pd_unknown["total_power_W"] is None and pd_unknown["power_by_wavelength"][0]["power_W"] is None
    assert pd_dark["total_power_W"] == 0.0 and pd_dark["power_by_wavelength"][0]["power_W"] == 0.0


# ------------------------------------------------------------------------------------------------------------------- n6
def test_a_negligible_below_floor_remainder_retains_a_bound_on_summed_power():
    """Lines of 1 W and 1e-12 W through a polarizer at 89.9 deg: the weak line ends at the floor with 3e-18 W.
    The traced sum remains Malus' cos^2(89.9 deg) W, while the omitted remainder gives a positive upper bound;
    the weak line alone is not known (ZERO_POWER)."""
    source = B.laser("L", spectrum=((633.0, 1.0), (780.0, 1e-12)), power_W=1.0, waist_radius_mm=0.3, position=(0, 0, 0), direction=X,
                     polarization=JonesVector.linear(89.9))
    pol = B.polarizer("POL", position=(20.0, 0.0, 0.0), axis=X, angle_deg=0.0)
    det = B.observation_plane("DET", position=(60.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(source, pol), observations=(det,), targets=(
        _t("all", "DET", maximum=1.0), _t("weak", "DET", wavelength_nm=780.0, maximum=1e-30))))
    t = {x.target_id: x for x in result.targets}
    malus = math.cos(math.radians(89.9)) ** 2
    # (FAINT_LIGHT_BOUNDED: the light the run leaves out as negligible is the stated truncation of the traced value)
    assert (t["all"].status, t["all"].issue_codes) == (Status.PASS, ("FAINT_LIGHT_BOUNDED",))
    assert t["all"].value == pytest.approx(malus, rel=1e-9)
    assert (t["weak"].status, t["weak"].issue_codes) == (Status.UNKNOWN, ("ZERO_POWER",))
    obs = result.observations[0]
    assert obs.power_complete and t["all"].value == obs.total_power_W
    assert obs.quantity_state.value == "KNOWN" and obs.power_upper_bound_W > obs.total_power_W
    entry = t["all"].value_basis["value"]
    (omitted,) = entry.omitted
    assert entry.kind.value == "MODEL" and omitted.part.value == "BELOW_NEGLIGIBLE_LEVEL"
    assert omitted.upper == pytest.approx(obs.power_upper_bound_W - obs.total_power_W, rel=1e-6)
    assert [(s.wavelength_nm, s.complete) for s in obs.power_by_wavelength] == [(633.0, True), (780.0, False)]


def test_the_acceptance_cone_is_two_sided():
    """An observation plane has no front: the AOI is measured from the plane's normal line, so a beam travelling against
    the normal at 0 deg is inside a 10 deg cone (detected), at 30 deg outside it (not detected) -- on either side."""
    det = B.observation_plane("DET", position=(50.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=10.0, absorbing=True,
                              acceptance_half_angle_deg=10.0)
    head_on = run(ProjectSnapshot(components=(_laser(w0=0.5),), observations=(det,)))
    assert one_arrival(head_on, "DET").detected_power_W == pytest.approx(P_W, rel=1e-12)
    d = (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0)
    oblique = run(ProjectSnapshot(components=(_laser(w0=0.5, pos=(0.0, -50.0 * math.tan(math.radians(30.0)), 0.0), direction=d),),
                                  observations=(det,)))
    assert one_arrival(oblique, "DET").detected_power_W == 0.0


def test_monitors_spend_the_step_budget_but_not_the_interaction_limit():
    """Two-mirror cavity: with the default budget the path stops after 64 reflections with or without monitors (the
    interaction limit); with max_trace_steps = 100 each monitor crossing is a step, so 2 monitors stop it after 33."""
    def cavity(monitors: int):
        m1 = B.mirror("M1", position=(-50.0, 0, 0), normal=(1.0, 0, 0), clear_radius_mm=10.0)
        m2 = B.mirror("M2", position=(50.0, 0, 0), normal=(-1.0, 0, 0), clear_radius_mm=10.0)
        planes = tuple(B.observation_plane(f"MON{k}", position=(10.0 + 6.0 * k, 0, 0), normal=X) for k in range(monitors))
        return ProjectSnapshot(components=(_laser(w0=0.5), m1, m2), observations=planes)

    def reflections(result):
        (path,) = result.paths
        return path.termination, sum(1 for step in path.lineage if step.endswith(":reflect"))

    assert reflections(run(cavity(0))) == reflections(run(cavity(2))) == ("MAX_INTERACTIONS", 64)
    small = NumericalConfig(max_trace_steps=100)
    assert reflections(run(cavity(0), numerical=small)) == ("MAX_INTERACTIONS", 64)
    assert reflections(run(cavity(2), numerical=small)) == ("TRACE_BUDGET", 33)


# ------------------------------------------------------------------------------------- interaction limit across branches
def test_a_splitter_cavity_followed_through_its_reflected_branches_stops_at_the_interaction_limit():
    """Two R 0.9 / T 0.1 plate splitters facing each other, the source between them: the light that stays in the cavity is
    the reflected branch of a branch of a branch ... A branch is born at a physical interaction, so it counts it: the
    reflected lineage stops after 64 splits (the limit), like the two-mirror cavity followed by one path, instead of
    being followed through ~250 reflections to the power floor. References: P R^64 left in the cavity, and the parent of
    the last branch escapes carrying its transmitted P R^63 T without requiring another interaction."""
    r, t = 0.9, 0.1
    coating = ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=t, transmittance_p=t)
    s1 = B.beam_splitter("S1", position=(-50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), coating=coating)
    s2 = B.beam_splitter("S2", position=(50.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), coating=coating)
    result = run(ProjectSnapshot(components=(_laser(w0=0.5), s1, s2)))
    limit = CFG.max_interactions_per_path
    stopped = sorted((p for p in result.paths if p.termination == "MAX_INTERACTIONS"), key=lambda p: p.final_power_W)
    assert [len(p.lineage) for p in stopped] == [limit]
    assert all(step.endswith(":split_reflect") for step in stopped[0].lineage)
    assert stopped[0].final_power_W == pytest.approx(P_W * r**limit, rel=1e-9)
    last_escape = next(p for p in result.paths if p.termination == "ESCAPED" and len(p.lineage) == limit)
    assert last_escape.final_power_W == pytest.approx(P_W * r ** (limit - 1) * t, rel=1e-9)
    (line,) = result.power_budget.lines
    assert line.sinks["INTERACTION_LIMIT"] == pytest.approx(P_W * r**limit, rel=1e-9)
    assert "BELOW_POWER_FLOOR" not in line.sinks and line.conserved
    assert {(i.code.value, i.status) for i in result.issues if i.code.value == "MAX_INTERACTIONS"} == {("MAX_INTERACTIONS", Status.UNKNOWN)}
    assert len(result.paths) == limit + 1  # the root and one branch per split
