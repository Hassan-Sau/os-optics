"""Containments: where the model cannot state a beam's power, a plane's reading or a share taken
before it, the result says so (an interval that holds the truth, or out of model scope) instead of a wrong value.

* an absorbing plane weighs the light a surface takes before it: ``[D - S, D]``, no certain share;
* a certain share is not counted where another counted surface only turns light;
* a beam outside a plane's acceptance cone adds no share taken;
* without a Gaussian state a reflecting start surface that reaches beyond the plane, or a next surface that reaches before it,
  puts the plane's power out of model scope;
* a surface beside the chief ray that the beam's tail may meet while no pupil ray does bounds the plane and, by the same
  bound in W, every plane behind the leg's end (the tail beside a waist);
* an obstructing absorber with holes proves no later hole dark; a curved obstruction met off its axis puts the
  annulus's power out of model scope.

References are closed forms of the paraxial Gaussian beam (straight edges: ``0.5 erfc(sqrt(2) a / w)``) or direct
integrations of its profile; no Engine code."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import run
from os_optics.results import basis_of
from os_optics import builders as B
from os_optics.contracts import (AbsorberInteraction, CircularAperture, EngineeringComponentSnapshot, MirrorInteraction,
                                      PlaneSurface, ProjectSnapshot, RectangularAperture, SagSurface, Target, ThinLensInteraction)

P0, LAM = 1e-3, 780.0
X = (1.0, 0.0, 0.0)


def _w(w0: float, z: float) -> float:
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    return w0 * math.sqrt(1.0 + (z / z_r) ** 2)


def _target(minimum: float, observation: str = "T") -> Target:
    return Target(target_id=f"min_{observation}", metric="power_W", observation_id=observation, minimum=minimum, unit="W")


def _verdict(result, observation: str = "T"):
    target = next(t for t in result.targets if t.observation_id == observation)
    return target.assessment.value if target.assessment else None


# -------------------------------------------------------------------------------- shares taken before a plane
@pytest.mark.parametrize("kind", ("dump", "fold"))
@pytest.mark.parametrize("y0_w", (0.1, 0.5))
def test_an_absorbing_plane_reads_its_detected_power_within_the_share_a_surface_takes_before_it(kind: str, y0_w: float):
    """A 0.3 mm beam, an absorbing plane T just before a 45 deg dump (or fold) the chief ray meets next: the surface takes
    ``0.5 erfc(sqrt(2) y0 / w)`` of the beam before T (42 % at 0.1 w). T detected the whole beam with no flag; now it reads
    ``[D - S, D]`` with ``S`` above the share taken and no certain share (the light certainly taken may be light that would
    have missed the active area), so a minimum between the truth and ``D`` is not met."""
    w0, tilt = 0.3, math.radians(45.0)
    gap = y0_w * w0 * math.tan(tilt)
    stopped = 0.5 * math.erfc(math.sqrt(2.0) * y0_w * w0 / _w(w0, 40.0 - gap))
    truth = P0 * (1.0 - stopped)
    laser = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=(-40.0, 0, 0), direction=X)
    n = (-math.cos(tilt), math.sin(tilt), 0.0)
    other = (B.beam_dump("D", position=(0, 0, 0), axis=n, radius_mm=12.0) if kind == "dump"
             else B.mirror("D", position=(0, 0, 0), normal=n, clear_radius_mm=12.0, reflectance=0.99))
    t = B.observation_plane("T", position=(-gap, 0, 0), normal=X, radius_mm=10.0, absorbing=True)
    result = run(ProjectSnapshot(components=(laser, other), observations=(t,), targets=(_target(0.99 * P0),)))
    (obs,) = result.observations
    (arrival,) = obs.arrivals
    assert arrival.detected_power_W == pytest.approx(P0, rel=1e-6)
    assert arrival.power_taken_before_at == ("D/" + ("dump" if kind == "dump" else "face"),)
    assert arrival.power_taken_before_upper_W >= P0 * stopped
    assert arrival.power_taken_before_lower_W is None
    assert obs.quantity_state.value == "BOUNDED"
    assert obs.power_lower_bound_W <= truth
    assert _verdict(result) != "MET"


@pytest.mark.parametrize("z0_w", (0.5, 1.0))
def test_a_lens_beside_the_chief_ray_before_a_dump_leaves_no_certain_share(z0_w: float):
    """A thin lens beside the chief ray (covering y < -0.05 mm) turns the lines through it over the dump the chief ray meets
    next: the dump stops ``P(z < -z0) P(y > -0.05)`` of the beam before T. Its certain share counted lines the lens turns
    away first (a false UNMET at 0.995 of the truth); with the lens counted no share is certain: ``[P - S, P]``."""
    w0, tilt, y_edge = 0.3, math.radians(45.0), -0.05
    z0 = z0_w * w0
    gap = z0 * math.tan(tilt)
    stopped = 0.5 * math.erfc(math.sqrt(2.0) * z0 / _w(w0, -2.0 + gap)) * (1.0 - 0.5 * math.erfc(math.sqrt(2.0) * abs(y_edge) / w0))
    truth = P0 * (1.0 - stopped)
    aperture = RectangularAperture(half_width_mm=10.0, half_height_mm=10.0, center_offset_mm=(15.0, y_edge - 10.0))
    surface = PlaneSurface(surface_id="lens", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=aperture, mount_margin_mm=0.0,
                           medium_positive="air", medium_negative="air", interaction=ThinLensInteraction(focal_length_mm=150.0, transmittance=1.0))
    lens = EngineeringComponentSnapshot(instance_id="Lz", role="lens", pose=B._pose((-20.0, 0.0, 15.0), X), surfaces=(surface,))
    laser = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=(-40.0, 0, 0), direction=X, waist_position_z_mm=38.0)
    dump = B.beam_dump("D", position=(0, 0, 0), axis=(-math.cos(tilt), 0.0, math.sin(tilt)), radius_mm=12.0)
    t = B.observation_plane("T", position=(-gap, 0, 0), normal=X, radius_mm=10.0)
    result = run(ProjectSnapshot(components=(laser, dump, lens), observations=(t,), targets=(_target(0.995 * truth),)))
    (arrival,) = result.observations[0].arrivals
    assert set(arrival.power_taken_before_at) == {"Lz/lens", "D/dump"}
    assert arrival.power_taken_before_lower_W is None
    assert arrival.power_taken_before_upper_W >= P0 * stopped
    assert _verdict(result) != "UNMET"


def test_a_beam_outside_the_acceptance_cone_adds_no_share_taken_to_the_plane():
    """Beam A crosses T within its 5 deg acceptance cone; beam B crosses it at 30 deg (received 0) just before a dump that
    takes part of B. T receives exactly A's 1 mW: minima of 0.9999 mW and 0.99 mW are met (B's shares were subtracted)."""
    w0, ang = 0.3, math.radians(30.0)
    a_laser = B.laser("A", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=(-40.0, 0, 3.0), direction=X)
    d_b = np.array([math.cos(ang), math.sin(ang), 0.0])
    hit_b = np.array([0.0, 0.0, -3.0]) + 0.08 * d_b
    b_laser = B.laser("B", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=tuple(hit_b - 40.0 * d_b), direction=tuple(d_b))
    perp = np.array([-math.sin(ang), math.cos(ang), 0.0])
    n_d = -d_b * math.cos(math.pi / 4) + perp * math.sin(math.pi / 4)
    dump = B.beam_dump("D", position=tuple(hit_b), axis=tuple(n_d), radius_mm=5.0)
    t = B.observation_plane("T", position=(0, 0, 0), normal=X, radius_mm=10.0, acceptance_half_angle_deg=5.0)
    for minimum in (0.9999e-3, 0.99e-3):
        result = run(ProjectSnapshot(components=(a_laser, b_laser, dump), observations=(t,), targets=(_target(minimum),)))
        outside = next(a for a in result.observations[0].arrivals if a.within_acceptance is False)
        assert outside.power_taken_before_upper_W is not None  # (B's share taken stays on its arrival)
        assert _verdict(result) == "MET"
        assert result.targets[0].value == pytest.approx(P0, rel=1e-12)


# -------------------------------------------------------------------------------- a monitor after a fold
def test_a_monitor_after_a_fold_without_a_gaussian_state_is_out_of_scope_where_the_mirror_reaches_beyond_it():
    """An elliptical beam (M^2 1 / 2.1) folded out of plane at 77 deg has no Gaussian state (twisted astigmatism); 17.8 % of
    it leaves the mirror beyond a monitor 0.25 mm after the fold, while every pupil ray crosses the monitor (near the waist
    they lie well inside the beam). The monitor read the whole beam, exact; the mirror's support reaches beyond it, so its
    power is out of model scope. A monitor beyond the mirror's reach reads the beam.

    Register row T-04, withheld: the beam has no Gaussian state, so its power is its pupil samples' share, and near a waist the
    samples lie well inside the beam and do not sample the part that leaves beyond the monitor."""
    n = (-0.217551273, 0.967588023, -0.12823752)
    tn = (0.956828365, -0.064123474, 0.283491904)
    tp = np.array((0.22612451, 0.105151786, -0.013936101))
    for place, scoped in ((tp, True), (tp + 25.0 * np.array(tn), False)):
        laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=0.056094031752207646, waist_radius_y_mm=0.14037063609031555, m2=1.0,
                        m2_y=2.101783895176921, power_W=P0, position=(-30, 0, 0), direction=X, waist_position_z_mm=9.987034811675102)
        mirror = B.mirror("M", position=(0, 0, 0), normal=n, reflectance=0.99, clear_radius_mm=20.0)
        t = B.observation_plane("T", position=tuple(float(v) for v in place), normal=tn, radius_mm=60.0)
        result = run(ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=(_target(0.98e-3),)))
        (arrival,) = result.observations[0].arrivals
        assert arrival.beam.gaussian is None or not arrival.beam.gaussian.valid
        if scoped:
            assert "M/face" in arrival.power_out_of_scope_at
            assert result.targets[0].status.value == "OUT_OF_MODEL_SCOPE"
        else:
            assert arrival.power_out_of_scope_at == ()
            assert _verdict(result) == "MET"


# ------------------------------------------------------------------------------------------------ the tail beside a waist
TAIL_N = (0.75095129, 0.57052169, -0.33253143)
TAIL_C = np.array((-2.30460535, -1.44934377, 0.91578842))
TAIL_R = 2.2339150461982724


def _tail_scene(shift: float, fold: bool) -> ProjectSnapshot:
    """A 0.088 mm waist 38 mm after the source at x = -60; a tilted dump disc whose edge passes 0.05 mm (in its plane) beside
    the chief ray 3.8 mm before the monitor T at x = 0 (``shift`` mm farther off in -y); with ``fold`` a mirror at x = 20
    turns the beam to +y onto the plane T2."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=0.088, position=(-60.0, 0, 0), direction=X, waist_position_z_mm=38.0)
    dump = B.beam_dump("D", position=tuple(float(v) for v in TAIL_C + np.array([0.0, -shift, 0.0])), axis=TAIL_N, radius_mm=TAIL_R)
    components = [laser, dump]
    observations = [B.observation_plane("T", position=(0, 0, 0), normal=X, radius_mm=30)]
    if fold:
        components.append(B.mirror("M", position=(20.0, 0, 0), normal=(-1, 1, 0), clear_radius_mm=12.7))
        observations.append(B.observation_plane("T2", position=(20.0, 20.0, 0), normal=(0, 1, 0), radius_mm=30))
    return ProjectSnapshot(components=tuple(components), observations=tuple(observations),
                           targets=tuple(_target(0.9e-3, o.observation_id) for o in observations))


def _tail_reference() -> float:
    """The fraction of the beam whose energy-flow lines meet the dump disc before T: lines ``xi w(s)`` off the chief ray
    (paraxial, ``s`` from the source), each followed to the disc's plane (bisection), counted where the crossing lies within
    the disc and before x = 0; the profile's mass by a 2-D midpoint rule over +-4.5 radii."""
    n, c = np.array(TAIL_N), TAIL_C
    z_r = math.pi * 0.088 ** 2 / (LAM * 1e-6)
    h = 0.01
    xi = np.arange(-4.5, 4.5, h) + h / 2
    gy, gz = np.meshgrid(xi, xi, indexing="ij")
    weight = np.exp(-2.0 * (gy * gy + gz * gz))

    def point(s):
        w = 0.088 * np.sqrt(1.0 + ((s - 38.0) / z_r) ** 2)
        return np.stack([s - 60.0 + 0.0 * gy, gy * w, gz * w], axis=-1)

    def side(s):
        return (point(s) - c) @ n

    lo, hi = np.full(gy.shape, 0.0), np.full(gy.shape, 60.0)
    f_lo = side(lo)
    crosses = np.sign(f_lo) != np.sign(side(hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        same = np.sign(side(mid)) == np.sign(f_lo)
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    p = point(0.5 * (lo + hi))
    on_disc = np.linalg.norm((p - c) - ((p - c) @ n)[..., None] * n, axis=-1) <= TAIL_R
    meets = crosses & on_disc & (p[..., 0] < 0.0)
    return float(weight[meets].sum() / weight.sum())


def test_a_dump_beside_the_chief_ray_that_no_pupil_ray_meets_bounds_the_plane_and_the_beam_behind():
    """Near a waist the pupil rays pass well inside the beam: none meets the dump although 23 % of the beam's energy-flow
    lines do (reference), and T read 1 mW exact (a 0.9 mW minimum MET). T reads ``[P - S, P]`` with ``S`` above that
    share (not MET). Behind the fold the beam still counts that light: T2 reads ``[P - S', P]`` with ``S'`` the bound of the
    share the dump may take on the whole leg to the mirror, which holds the light that is there, ``P (1 - share)`` (the
    mirror is an ideal reflector), and is not MET either. The dump 1.5 mm farther off: unchanged."""
    share = _tail_reference()
    assert 0.2 < share < 0.27
    result = run(_tail_scene(0.0, True))
    by_plane = {o.observation_id: o for o in result.observations}
    (t,) = by_plane["T"].arrivals
    assert t.pupil_rays_arrived == t.pupil_ray_count
    assert t.power_taken_before_at == ("D/dump",) and t.power_taken_before_upper_W >= share * P0
    assert by_plane["T"].quantity_state.value == "BOUNDED"
    assert _verdict(result, "T") != "MET"
    (t2,) = by_plane["T2"].arrivals
    assert t2.power_out_of_scope_at == () and t2.power_taken_before_at == ("D/dump",)
    assert share * P0 <= t2.power_taken_before_upper_W < P0
    assert by_plane["T2"].quantity_state.value == "BOUNDED"
    assert by_plane["T2"].power_lower_bound_W == pytest.approx(P0 - t2.power_taken_before_upper_W, rel=1e-12)
    assert by_plane["T2"].power_lower_bound_W <= P0 * (1.0 - share) <= t2.beam.power_W == pytest.approx(P0, rel=1e-12)
    entry = basis_of(t2, "beam.power_W")
    assert entry.kind.value == "UPPER_BOUND"
    assert [(o.part.value, o.at, o.upper) for o in entry.omitted] == [("BEAM_PART_ELSEWHERE", ("D/dump",), t2.power_taken_before_upper_W)]
    target = next(x for x in result.targets if x.observation_id == "T2")
    assert (target.status.value, target.assessment.value, target.issue_codes) == ("UNKNOWN", "INDETERMINATE", ("POWER_TAKEN_BEFORE_PLANE",))
    (issue,) = [i for i in result.issues if i.location == "surface:D/dump"]
    assert (issue.code.value, issue.status.value) == ("POWER_TAKEN_BEFORE_PLANE", "WARNING")
    context = issue.numerical_context
    assert context["next_surface_id"] == "M/face" and context["surface_ids"] == ["D/dump"]
    assert context["tail_power_upper_W"] == t2.power_taken_before_upper_W == pytest.approx(context["tail_fraction_upper"] * P0, rel=1e-12)
    assert not any(i.status.value == "OUT_OF_MODEL_SCOPE" for i in result.issues)
    clear = run(_tail_scene(1.5, True))
    for obs in clear.observations:
        (a,) = obs.arrivals
        assert a.power_taken_before_at == () and a.power_out_of_scope_at == () and obs.quantity_state.value == "KNOWN"
    assert all(_verdict(clear, o) == "MET" for o in ("T", "T2"))


# -------------------------------------------------------------------------------- obstructions and holes
S2 = math.sqrt(0.5)


def test_an_obstructing_absorber_with_a_hole_proves_no_later_hole_dark():
    """A 15 mm beam; an absorber disc r 6 with a hole r 1.5 at 4 mm obstructs it; a fold mirror's hole r 1.5 lies in line with
    the absorber's hole, inside the absorber's outline. The light through the absorber's hole leaves through the mirror's:
    the plane read the whole annulus, 1.7 % high (a 7.40e-4 W minimum passed against 7.31e-4 W). The annulus is now not
    followed past the holed mirror (``ANNULAR_BEAM_NOT_FOLLOWED``): no value.

    Register row T-06, withheld: the annulus's power is a Gaussian share beside one support; with a hole in that support and a
    second holed surface behind it the share that passes both is not that integral."""
    hy, hr, mr = 4.0, 1.5, 1.5
    dump_s = PlaneSurface(surface_id="dump", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=6.0),
                          aperture_u_axis=(1.0, 0.0, 0.0), holes=(CircularAperture(radius_mm=hr, center_offset_mm=(hy, 0.0)),),
                          interaction=AbsorberInteraction())
    dump = EngineeringComponentSnapshot(instance_id="DUMP", role="dump", pose=B._pose((0.0, 0.0, 0.0), X), surfaces=(dump_s,))
    mirror_s = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(S2, 0.0, -S2),
                            aperture=RectangularAperture(half_width_mm=90.0, half_height_mm=70.0), aperture_u_axis=(S2, 0.0, S2),
                            holes=(CircularAperture(radius_mm=mr, center_offset_mm=(hy / S2, 0.0)),), interaction=MirrorInteraction())
    mirror = EngineeringComponentSnapshot(instance_id="FOLD", role="mirror", pose=B._pose((50.0, 0.0, 0.0), X), surfaces=(mirror_s,))
    laser = B.laser("SRC", wavelength_nm=633.0, power_W=P0, waist_radius_mm=15.0, position=(-200.0, 0.0, 0.0), direction=X, waist_position_z_mm=200.0)
    plane = B.observation_plane("OUT", position=(50.0, 100.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=80.0)
    result = run(ProjectSnapshot(components=(laser, dump, mirror), observations=(plane,), targets=(_target(7.40e-4, "OUT"),)))
    codes = {i.code.value for i in result.issues}
    assert "GAUSSIAN_CENTRAL_OBSTRUCTION" in codes and "ANNULAR_BEAM_NOT_FOLLOWED" in codes
    assert result.targets[0].status.value != "PASS"
    assert result.observations[0].quantity_state.value not in ("KNOWN", "BOUNDED")


@pytest.mark.parametrize("tilt_deg", (0.0, 20.0, 35.0))
def test_a_curved_obstruction_met_off_its_axis_puts_the_annulus_power_out_of_scope(tilt_deg: float):
    """A convex absorber cap (R 12 mm, support r 8) inside a 15 mm beam whose chief ray passes 3 mm off its vertex. Along the
    cap's axis the share on its support is its shadow (unchanged); at 20 and 35 deg the share mapped onto the plane at the
    chief ray's hit is off the shadow's by +1.8 / +4.6 %, which a stated lower bound carried (a 6.9e-4 W minimum passed
    against 6.74e-4 W): the annulus's power behind the cap is now out of model scope, the cap named.

    Register row T-07, withheld: (off the axis) the shadow of a sagged support along a tilted beam is not the support mapped
    onto the plane at the chief ray's hit, and the mapped share is no bound of it."""
    t = math.radians(tilt_deg)
    d = (math.cos(t), math.sin(t), 0.0)
    surf = SagSurface(surface_id="dump", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), curvature_per_mm=1 / 12.0,
                      conic=0.0, aperture=CircularAperture(radius_mm=8.0), interaction=AbsorberInteraction())
    cap = EngineeringComponentSnapshot(instance_id="DUMP", role="dump", pose=B._pose((0.0, 0.0, 0.0), X), surfaces=(surf,))
    start = np.array([0.0, 3.0, 0.0]) - 200.0 * np.array(d)
    laser = B.laser("SRC", wavelength_nm=633.0, power_W=P0, waist_radius_mm=15.0, position=tuple(start), direction=d, waist_position_z_mm=200.0)
    plane = B.observation_plane("OUT", position=tuple(np.array([0.0, 3.0, 0.0]) + 100.0 * np.array(d)), normal=d, radius_mm=80.0)
    result = run(ProjectSnapshot(components=(laser, cap), observations=(plane,), targets=(_target(6.9e-4, "OUT"),)))
    (arrival,) = result.observations[0].arrivals
    if tilt_deg == 0.0:
        assert "DUMP/dump" not in arrival.power_out_of_scope_at
        assert result.observations[0].quantity_state.value != "OUT_OF_MODEL_SCOPE"
    else:
        assert "DUMP/dump" in arrival.power_out_of_scope_at
        assert result.observations[0].quantity_state.value == "OUT_OF_MODEL_SCOPE"
        assert result.targets[0].status.value == "OUT_OF_MODEL_SCOPE"


# -------------------------------------------------------------------------------- planes bounded by light taken before them
def test_a_plane_bounded_by_light_taken_before_it_states_no_map_handoff_power_or_detector_response():
    """The tail scene: T is BOUNDED, ``[P - S, P]`` with ``D/dump`` taking 23.5 % of the beam first. Its irradiance map stated the
    whole beam (1 mW against 0.765 mW arriving) and the detector response and the handoff's power followed; now the map is
    ``UNAVAILABLE`` (``POWER_TAKEN_BEFORE_PLANE``, the surfaces and paths named), the detector response refuses through it, the
    handoff states no power (geometry and Gaussian parameters stay) and a field built from it is refused. Clear of the beam
    (1.5 mm farther off) the map's grid power is the beam's and the handoff states it.

    Register row T-08, withheld: the profile on the plane is not the unclipped beam's, and a clip of share s can move the peak
    by about 2 sqrt(s)."""
    from os_optics.results.detector import detector_response
    from os_optics.results.handoff import handoff
    from os_optics.results.irradiance import irradiance_map
    from os_optics.results.pupil_field import beams_of_handoff  # (os_optics: the earlier name would rebind results.pupil_field to the module)

    result = run(_tail_scene(0.0, False))
    doc = irradiance_map(result, "T")
    assert (doc["status"], doc["reason"], doc["quantity_state"]) == ("UNAVAILABLE", "POWER_TAKEN_BEFORE_PLANE", "BOUNDED")
    assert doc["power_taken_before_at"] == ["D/dump"] and doc["path_ids"] == ["S#0"] and "incoherent_W_per_mm2" not in doc
    detector = {"pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0, "pixel_count": [4, 4]}
    response = detector_response(doc, detector)
    assert response["status"] == "UNAVAILABLE" and response["upstream_reason"] == "POWER_TAKEN_BEFORE_PLANE"
    (beam,) = handoff(result, None, "T")["beams"]
    assert beam["power_W"] is None and beam["gaussian"] is not None
    assert beam["validity"]["beam_parameters"] == "POWER_TAKEN_BEFORE_PLANE" and beam["validity"]["power_taken_before_at"] == ["D/dump"]
    # (the only beam of the plane: no field under either composition, the refusal it was)
    for composition in ("COHERENT", "INCOHERENT"):
        with pytest.raises(Exception) as refused:
            beams_of_handoff(handoff(result, None, "T"), LAM, composition=composition)
        assert getattr(refused.value, "reason", getattr(refused.value, "code", None)) == "POWER_TAKEN_BEFORE_PLANE"

    clear = run(_tail_scene(1.5, False))
    doc = irradiance_map(clear, "T")
    assert doc["status"] == "OK" and doc["grid_power_incoherent_W"] == pytest.approx(P0, rel=1e-3)
    (beam,) = handoff(clear, None, "T")["beams"]
    assert beam["power_W"] == pytest.approx(P0, rel=1e-12) and "power_taken_before_at" not in beam["validity"]


@pytest.mark.parametrize(("tilt_deg", "gap"), [(45.0, 0.5), (45.0, 0.01), (30.0, 0.01)])
def test_parallel_tilted_plates_close_together_take_nothing_from_each_others_legs(tilt_deg: float, gap: float):
    """Two 2 mm N-BK7 plates (radius 12.7 mm) tilted together, ``gap`` mm apart along their normal, a 1 mm beam through both:
    the first plate's faces lie wholly behind the plane of the second's back face where the light leaves it, so nothing of the
    leg after the second plate can meet them. The tail check weighed them by balls about as large as the plates and counted
    a share (a 0.5 mW minimum 100 mm on undecided at 45 deg / 0.01 mm); tested by the plates' own outlines it counts none:
    the plane reads the beam with no share taken, as without the check."""
    a = math.radians(tilt_deg)
    axis = (math.cos(a), math.sin(a), 0.0)
    sep = 2.0 + gap
    plates = (B.window("P1", thickness_mm=2.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0), axis=axis),
              B.window("P2", thickness_mm=2.0, clear_radius_mm=12.7, material="N-BK7", position=(sep * axis[0], sep * axis[1], 0.0), axis=axis))
    laser = B.laser("L", wavelength_nm=LAM, power_W=P0, waist_radius_mm=1.0, position=(-60.0, 0, 0), direction=X)
    out = B.observation_plane("OUT", position=(100.0, 0, 0), normal=X, radius_mm=40)
    result = run(ProjectSnapshot(components=(laser, *plates), observations=(out,), targets=(_target(0.5e-3, "OUT"),)))
    main = max(result.observations[0].arrivals, key=lambda arrival: arrival.beam.power_W)
    assert main.power_taken_before_at == () and main.power_out_of_scope_at == ()
    assert not any(i.code.value == "OUT_OF_MODEL_SCOPE" for i in result.issues)
    assert _verdict(result, "OUT") == "MET"


# ------------------------------------------------------------------------------------------------ crystal twins at a polarizer
def _plate_before_polarizer(axis: tuple[str, str, str], waves: float, polarizer: bool = True):
    """A 1 mW 780 nm beam polarized along design Y (0 deg), a Project-declared calcite plate at normal incidence with its optic axis
    ``axis`` in its faces (element frame), its thickness ``waves`` waves of retardance ``(n_o - n_e) L / lambda``; the test linear
    polarizer 30 mm on (axis along the input polarization) and the plane OUT behind it; power_W targets >= 3e-4 and <= 1e-5 W."""
    from declared_helpers import declared_component, with_models
    from local_helpers import ALONG_X, POLARIZER, catalog_document, element, plane, project, source
    from test_project_crystals import CALCITE, N_E, N_O, displacer

    from os_optics.local import run_optical_project

    length = f"{waves * 780.0e-6 / (N_O - N_E):.7f}".rstrip("0").rstrip(".")
    model = displacer("PL", axis=axis, length=float(length))
    model["model"]["surfaces"][1]["vertex_z_mm"] = length
    components = [source(wavelength_nm=780.0, waist_mm=0.3, polarization_deg=0.0),
                  declared_component("P0", "prism", "PL", position=(0.0, 0.0, 0.0), quaternion=ALONG_X)]
    if polarizer:
        components.append(element("POL", "polarizer", POLARIZER, position=(30.0, 0.0, 0.0)))
    doc = with_models(project(components, [plane("OUT", (60.0, 0.0, 0.0), radius_mm=20.0)]), [model], [CALCITE])
    doc["targets"] = [{"target_id": "open", "metric": "power_W", "observation_id": "OUT", "minimum": 3e-4, "unit": "W"},
                      {"target_id": "ext", "metric": "power_W", "observation_id": "OUT", "maximum": 1e-5, "unit": "W"}]
    return run_optical_project(doc, catalog=catalog_document()).result


@pytest.mark.parametrize("waves", [214.5, 215.0])
def test_a_crystal_wave_plate_before_a_polarizer_states_no_power_behind_it(waves: float):
    """Calcite plate with its axis at 45 deg in its faces: its ordinary and extraordinary waves (equal shares) leave it overlapping
    with a relative phase ``2 pi waves``; the polarizer passes their fields' sum. At the half-wave thickness about 1e-7 W
    (plus the polarizer's leakage) arrives and at the full-wave 7.24e-4 W, while the sum of the two waves' powers read 3.62e-4 W
    at every thickness (a >= 3e-4 minimum MET, a <= 1e-5 maximum UNMET at the half wave). The power behind the polarizer is out
    of model scope: no target is decided, the polarizer named.

    Register row T-10, withheld: the two waves are one coherent field and the polarizer passes the sum of their fields; the
    trace carries no relative phase between them, and the sum of their powers is no bound of the result."""
    result = _plate_before_polarizer(("1", "1", "0"), waves)
    out = next(o for o in result.observations if o.observation_id == "OUT")
    assert len(out.arrivals) == 2 and all("POL/element" in a.power_out_of_scope_at for a in out.arrivals)
    assert {t.target_id: t.status.value for t in result.targets} == {"open": "OUT_OF_MODEL_SCOPE", "ext": "OUT_OF_MODEL_SCOPE"}
    assert all(t.value is None for t in result.targets)
    assert any(i.code.value == "OUT_OF_MODEL_SCOPE" and i.location == "surface:POL/element" and "polarizer" in i.message
               for i in result.issues)


def test_a_crystal_plate_with_its_axis_along_the_polarization_and_a_displacer_without_a_polarizer_keep_their_powers():
    """Controls. Axis along the input polarization (0 deg): one wave, nothing to mix: the polarizer passes it, decided (the
    plate's two Fresnel faces and the polarizer's 0.8). A beam displacer (axis at 45 deg to the beam) with 45 deg input and no
    polarizer: its two orthogonally polarized beams reach OUT walked off from each other, their powers add exactly (no
    interference term in a power), decided. A displacer's separated beams crossing a polarizer are taken as overlapping (no
    overlap test): out of model scope there.

    Register row T-11, withheld: (the displacer before a polarizer) whether the two separated beams still overlap at the
    polarizer is not tested; separated beams add as powers, overlapping ones as fields."""
    from local_helpers import POWER_W
    from test_project_crystals import N_O, fresnel_t

    one = _plate_before_polarizer(("1", "0", "0"), 214.5)
    (arrival,) = next(o for o in one.observations if o.observation_id == "OUT").arrivals
    assert arrival.power_out_of_scope_at == ()
    assert {t.target_id: t.status.value for t in one.targets} == {"open": "PASS", "ext": "FAIL"}

    def displaced(polarizer: bool):
        from local_helpers import POLARIZER, catalog_document, element, source
        from test_project_crystals import displacer, scene

        from os_optics.local import run_optical_project

        doc = scene(source(wavelength_nm=780.0, waist_mm=0.3, polarization_deg=45.0), ("BD", displacer(), 0.0))
        if polarizer:
            doc["components"].append(element("POL", "polarizer", POLARIZER, position=(50.0, 0.0, 0.0)))
        return run_optical_project(doc, catalog=catalog_document()).result

    free = displaced(False)
    beams = next(o for o in free.observations if o.observation_id == "OUT").arrivals
    assert len(beams) == 2 and not any(a.power_out_of_scope_at for a in beams)
    assert next(a for a in beams if abs(a.beam.position_mm[1]) < 1e-9).beam.power_W == pytest.approx(
        POWER_W * 0.5 * fresnel_t(N_O) ** 2, rel=1e-12)
    crossed = displaced(True)
    assert all("POL/element" in a.power_out_of_scope_at for a in next(o for o in crossed.observations if o.observation_id == "OUT").arrivals)
