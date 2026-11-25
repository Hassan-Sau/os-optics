"""Part of a beam reaches an observation plane its chief ray does not cross (the observation-plane residual).

The packet's power follows its chief ray. Where an observation plane T meets the beam inside its footprint behind (along
the chief ray) the surface that stops or turns the beam -- a tilted absorbing plane, an absorber, a fold mirror -- the
light on one side of the beam reaches T first, the chief ray never. T then has no arrival, but it is not dark: the
tracer records the partial crossing (``tracer.PartialCrossing``) with the most power that part can carry -- the Gaussian
beam's share of it, its tail beyond the pupil rays and a small active area included. T's power is then UNKNOWN, never a
known 0 W (``validation/untraced.py``: ``PARTIAL_CROSSING``), with that bound and the untraced power beside it as evidence,
and every target on it is INDETERMINATE whatever the bound says. An absorbing T takes that part
out of the beam: from there on the beam's power is out of model scope (``power_out_of_scope_at``). Where the beam does not
reach T before the surface, T stays a known zero. Nothing depends on the declaration order or on a target's threshold.

The geometry is the same throughout: a 1 mW Gaussian beam (waist 0.5 mm at the source, 10 mm before the origin) along
+X, a 45 deg surface through the origin, and T normal +X at x = c (or 45 deg through (c, 0, 0) before a plane at x = 0):
the beam's light on the far side of the line where T meets the surface, c from the chief ray, reaches T first: its share
is 0.5 erfc(sqrt(2) c / w), w the beam's radius at the end of the 10 mm leg.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import Tracer
from os_optics.scene.lower import lower_scene

from helpers import arrivals, one_arrival, run

SOURCE_W = 1e-3
LAM = 780.0
W0 = 0.5
Q = math.sqrt(0.5)
X = (1.0, 0.0, 0.0)
DIAGONAL = (Q, Q, 0.0)
CROSSED = (1e-3, 0.1)
CLEAR = 2.0
#: the beam's 1/e^2 radius 10 mm from its waist (the widest on the leg)
W_END = W0 * math.sqrt(1.0 + (10.0 / (math.pi * W0 * W0 / (LAM * 1e-6))) ** 2)


def _part(c: float, power_W: float = SOURCE_W) -> float:
    """The Gaussian beam's power beyond a line c from its centre."""
    return power_W * 0.5 * math.erfc(math.sqrt(2.0) * c / W_END)


def _laser():
    return B.laser("L", wavelength_nm=LAM, waist_radius_mm=W0, power_W=SOURCE_W, position=(-10.0, 0.0, 0.0), direction=X)


def _monitor(dx: float, oid: str = "T", **kwargs):
    kwargs.setdefault("radius_mm", 50.0)
    return B.observation_plane(oid, position=(dx, 0.0, 0.0), normal=X, **kwargs)


def _detector():
    return B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=DIAGONAL, radius_mm=50.0, absorbing=True)


def _maximum(value: float, oid: str = "T") -> Target:
    return Target(target_id=f"{oid}max", metric="power_W", observation_id=oid, maximum=value, unit="W")


def _observation(result, oid: str = "T"):
    return next(o for o in result.observations if o.observation_id == oid)


def _not_reached(result, oid: str = "T"):
    (issue,) = [i for i in result.issues if i.location == f"observation:{oid}" and i.code.value == "OBSERVATION_NOT_REACHED"]
    return issue


def _trace(project: ProjectSnapshot):
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    return Tracer(scene, library, cfg).run()


def _assert_bounded_from_zero(obs, bound_W: float) -> None:
    assert obs.arrivals == ()
    assert obs.total_power_W is None  # never a known 0 W
    assert not obs.power_complete
    assert obs.quantity_state.value == "BOUNDED" and obs.power_lower_bound_W == 0.0
    assert {f.reach for f in obs.followed_light} == {"PARTIAL_CROSSING"}
    assert obs.power_upper_bound_W == pytest.approx(bound_W, rel=1e-9)
    assert obs.untraced_power_W == pytest.approx(bound_W, rel=1e-9)


# ------------------------------------------------------------------------------------- a transparent plane, bounded


@pytest.mark.parametrize("dx", CROSSED)
@pytest.mark.parametrize("reverse", (False, True))
def test_a_plane_part_of_the_beam_crosses_before_a_tilted_detector_is_not_a_known_zero(dx: float, reverse: bool):
    observations = (_detector(), _monitor(dx))
    result = run(ProjectSnapshot(components=(_laser(),), observations=observations[::-1] if reverse else observations,
                                 targets=(_maximum(1e-9),)))
    (detected,) = arrivals(result, "D")
    assert detected.detected_power_W == pytest.approx(SOURCE_W, rel=1e-12)
    assert detected.power_out_of_scope_at == ()  # (a transparent plane takes nothing out of the beam)
    assert [p.termination for p in result.paths] == ["DETECTED"]
    _assert_bounded_from_zero(_observation(result), _part(dx))
    (target,) = result.targets
    assert target.value is None
    assert target.assessment.value == "INDETERMINATE"
    assert "NO_TRACED_ARRIVAL" not in target.issue_codes
    assert "UNTRACED_LIGHT_BOUNDED" in target.issue_codes
    issue = _not_reached(result)
    assert issue.status.value == "NOT_APPLICABLE"  # (a notice: the plane's interval and its followed_light carry the fact)
    assert issue.numerical_context["untraced_light"] == ["PARTIAL_CROSSING"]
    assert issue.numerical_context["partial_before_surface_id"] == "observation/D"
    assert issue.message.startswith("part of a traced beam reaches this observation plane before observation plane D, while "
                                    "the beam's chief ray does not cross it: it receives between 0 W and ")
    assert issue.suggested_action == ("place the plane where the beam's chief ray crosses it, or outside the beam's envelope, to "
                                      "read its power; a finite bound decides a power target on this plane only in the direction it "
                                      "proves (a maximum at or above it is met, a minimum above it is not); other power limits and beam "
                                      "metrics are not decided")


@pytest.mark.parametrize("dx", CROSSED)
def test_the_partial_crossing_does_not_depend_on_the_declaration_order(dx: float):
    results = [run(ProjectSnapshot(components=(_laser(),), observations=order, targets=(_maximum(1e-9),)))
               for order in ((_detector(), _monitor(dx)), (_monitor(dx), _detector()))]
    by_id = [{o.observation_id: o for o in r.observations} for r in results]
    assert by_id[0] == by_id[1]
    assert results[0].targets == results[1].targets


def test_the_trace_records_which_part_of_the_beam_reaches_the_plane_first():
    out = _trace(ProjectSnapshot(components=(_laser(),), observations=(_detector(), _monitor(0.1))))
    (part,) = out.partial_crossings
    assert (part.observation_id, part.ends_at, part.path_id, part.absorbing) == ("T", "observation/D", "L#0", False)
    assert part.power_W == pytest.approx(_part(0.1), rel=1e-9)
    assert 0 < part.rays < part.pupil_rays  # (the pupil rays below y = -0.1 mm)


@pytest.mark.parametrize("reverse", (False, True))
def test_a_plane_the_beam_does_not_reach_before_the_detector_stays_a_known_zero(reverse: bool):
    """2 mm behind: the beam's share beyond 2 mm (4 radii) is about 6e-19 W, below the line's negligible level."""
    observations = (_detector(), _monitor(CLEAR))
    result = run(ProjectSnapshot(components=(_laser(),), observations=observations[::-1] if reverse else observations,
                                 targets=(_maximum(1e-9),)))
    obs = _observation(result)
    assert obs.total_power_W == 0.0 and obs.power_complete and obs.quantity_state.value == "KNOWN_ZERO"
    (target,) = result.targets
    assert target.value == 0.0 and target.assessment.value == "MET" and "NO_TRACED_ARRIVAL" in target.issue_codes


def test_the_planes_availability_does_not_depend_on_a_target_or_its_threshold():
    def observed(targets):
        result = run(ProjectSnapshot(components=(_laser(),), observations=(_detector(), _monitor(1e-3)), targets=targets))
        obs = _observation(result)
        return (obs.quantity_state, obs.total_power_W, obs.power_complete, obs.power_upper_bound_W, obs.untraced_power_W,
                obs.power_by_wavelength), result

    reference, _ = observed(())
    strict, strict_run = observed((_maximum(1e-9),))
    loose, loose_run = observed((_maximum(1.0),))
    assert strict == loose == reference
    # (followed light only: BOUNDED from 0 with no value)
    assert reference[0].value == "BOUNDED" and reference[1] is None and 1e-9 < reference[3] < 1.0
    # the plane's record is the same with any target; a target is graded on [0, bound]
    # (targets only): a maximum under the bound is not decided, one above it is met
    assert strict_run.targets[0].assessment.value == "INDETERMINATE"
    assert loose_run.targets[0].assessment.value == "MET" and loose_run.targets[0].value is None


@pytest.mark.parametrize("dx", CROSSED)
def test_a_plane_part_of_the_beam_crosses_before_an_absorber_is_not_a_known_zero(dx: float):
    dump = B.beam_dump("DUMP", position=(0.0, 0.0, 0.0), axis=DIAGONAL)
    result = run(ProjectSnapshot(components=(_laser(), dump), observations=(_monitor(dx),), targets=(_maximum(1e-9),)))
    assert [p.termination for p in result.paths] == ["ABSORBED"]
    _assert_bounded_from_zero(_observation(result), _part(dx))
    assert result.targets[0].assessment.value == "INDETERMINATE"
    assert _not_reached(result).numerical_context["partial_before_surface_id"] == "DUMP/dump"


def test_a_plane_part_of_the_beam_crosses_before_a_fold_mirror_is_not_a_known_zero():
    """The 45 deg mirror folds the beam to +Y, parallel to T (normal +X, 0.1 mm behind the hit point along the beam): the
    chief ray never crosses T, but the light above y = 0.1 mm crosses it before it meets the mirror. So does the folded
    beam's own flow on the way out of the scene: on T's 50 mm of that leg the share beyond T grows as the beam widens."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0))
    result = run(ProjectSnapshot(components=(_laser(), fold), observations=(_monitor(0.1),), targets=(_maximum(1e-9),)))
    assert [p.termination for p in result.paths] == ["ESCAPED"]
    obs = _observation(result)
    assert obs.arrivals == () and obs.total_power_W is None and obs.quantity_state.value == "BOUNDED"

    def radius(z: float) -> float:
        return W0 * math.sqrt(1.0 + (z / (math.pi * W0 * W0 / (LAM * 1e-6))) ** 2)

    folded = SOURCE_W * 0.5 * (math.erfc(math.sqrt(2.0) * 0.1 / radius(60.0)) - math.erfc(math.sqrt(2.0) * 0.1 / radius(10.0)))
    assert folded == pytest.approx(2.5e-7, rel=0.05)
    assert _part(0.1) + 0.99 * folded <= obs.power_upper_bound_W <= _part(0.1) + 2.0 * folded  # (it starts 0.1 mm along that leg)
    assert result.targets[0].assessment.value == "INDETERMINATE"
    clear = run(ProjectSnapshot(components=(_laser(), fold), observations=(_monitor(CLEAR),), targets=(_maximum(1e-9),)))
    assert _observation(clear).quantity_state.value == "KNOWN_ZERO"


def test_part_of_a_beam_crossing_a_plane_beside_its_active_area_is_not_light_on_it():
    """T's active area (0.05 mm radius) is 5 mm from where the light crosses it: no light reaches the area, T stays dark."""
    beside = B.observation_plane("T", position=(0.1, 5.0, 0.0), normal=X, radius_mm=0.05)
    result = run(ProjectSnapshot(components=(_laser(),), observations=(_detector(), beside), targets=(_maximum(1e-9),)))
    assert _observation(result).quantity_state.value == "KNOWN_ZERO"
    assert result.targets[0].assessment.value == "MET"


# ------------------------------------------------------------------ beyond the pupil rays (review M2): the beam's tail


def _tilted_before_flat_detector(c: float, **monitor):
    """T at 45 deg through (c, 0, 0) and an absorbing D normal +X at the origin, which the chief ray meets first: the light
    above y = c reaches T first (``c09`` of the review)."""
    monitor.setdefault("radius_mm", 50.0)
    t = B.observation_plane("T", position=(c, 0.0, 0.0), normal=DIAGONAL, **monitor)
    d = B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=X, radius_mm=50.0, absorbing=True)
    return ProjectSnapshot(components=(_laser(),), observations=(t, d), targets=(_maximum(1e-9),))


@pytest.mark.parametrize("c", (0.3, 0.6, 0.7, 0.9))
def test_the_beams_tail_beyond_the_pupil_rays_is_bounded_by_its_gaussian_share(c: float):
    """The outermost pupil ring is 0.59 mm from the chief ray: from c = 0.6 no pupil ray reaches T first, but 8.2, 2.6 and
    0.16 uW of the Gaussian beam do -- the bound is that share, not 0 W and not the whole beam."""
    result = run(_tilted_before_flat_detector(c))
    _assert_bounded_from_zero(_observation(result), _part(c))
    assert result.targets[0].assessment.value == "INDETERMINATE"
    (part,) = _trace(_tilted_before_flat_detector(c)).partial_crossings
    assert (part.rays == 0) is (c >= 0.6)


@pytest.mark.parametrize(("centre", "radius"), (((-0.15, 0.35, 0.0), 0.03), ((-0.15, 0.35, 0.05), 0.01), ((-0.2, 0.4, 0.1), 0.03)))
def test_a_small_active_area_between_the_pupil_rays_is_bounded_by_the_gaussian_power_on_it(centre, radius):
    """A 10 or 30 um detector area on T, in the part of the beam that reaches T first, between pupil rays: about the
    Gaussian irradiance there times the area seen along the beam (pi r^2 cos 45 deg)."""
    t = B.observation_plane("T", position=centre, normal=DIAGONAL, radius_mm=radius)
    d = B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=X, radius_mm=50.0, absorbing=True)
    result = run(ProjectSnapshot(components=(_laser(),), observations=(t, d), targets=(_maximum(1e-9),)))
    rho2 = centre[1] ** 2 + centre[2] ** 2
    expected = 2.0 * SOURCE_W / (math.pi * W_END * W_END) * math.exp(-2.0 * rho2 / (W_END * W_END)) * math.pi * radius * radius * Q
    obs = _observation(result)
    assert obs.quantity_state.value == "BOUNDED"
    assert obs.power_upper_bound_W == pytest.approx(expected, rel=0.02)
    assert result.targets[0].assessment.value == "INDETERMINATE"


# --------------------------------------------------------------------- an absorbing plane part of the beam meets (B1)


@pytest.mark.parametrize("reverse", (False, True))
def test_an_absorbing_plane_part_of_the_beam_meets_first_is_bounded_and_so_is_the_detector_behind_it(reverse: bool):
    """c17 of the review: an absorbing T at 45 deg through (0.3, 0, 0) takes the light above y = 0.3 mm (11.5 %) out of the
    beam before the detector D at x = 0, which the chief ray meets. T is bounded by that share, not a known 0 W; D's
    detected power (the beam's, which follows the chief ray) is the upper end of ``[P - S, P]`` with ``S`` the bound of T's
    share: the interval holds the light that reaches D, ``P - part(0.3)``, and a minimum between its ends is not decided."""
    t = B.observation_plane("T", position=(0.3, 0.0, 0.0), normal=DIAGONAL, radius_mm=50.0, absorbing=True)
    d = B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=X, radius_mm=50.0, absorbing=True)
    observations = (t, d)
    result = run(ProjectSnapshot(components=(_laser(),), observations=observations[::-1] if reverse else observations,
                                 targets=(_maximum(1e-9), Target(target_id="dmin", metric="power_W", observation_id="D",
                                                                  minimum=0.95e-3, unit="W"))))
    _assert_bounded_from_zero(_observation(result), _part(0.3))
    targets = {t.target_id: t for t in result.targets}
    assert targets["Tmax"].assessment.value == "INDETERMINATE"
    detector = one_arrival(result, "D")
    assert detector.power_out_of_scope_at == () and detector.power_taken_before_at == ("observation/T",)
    assert (1.0 - 1e-9) * _part(0.3) <= detector.power_taken_before_upper_W <= 1.001 * _part(0.3)  # (the erfc's rounding)
    plane = _observation(result, "D")
    assert plane.quantity_state.value == "BOUNDED"
    assert plane.power_lower_bound_W <= (SOURCE_W - _part(0.3)) * (1.0 + 1e-9) and plane.total_power_W == pytest.approx(SOURCE_W, rel=1e-12)
    assert (targets["dmin"].assessment.value, targets["dmin"].issue_codes) == ("INDETERMINATE", ("POWER_TAKEN_BEFORE_PLANE",))
    assert targets["dmin"].value == pytest.approx(SOURCE_W, rel=1e-12)


def test_the_beam_behind_an_absorbing_plane_that_took_part_of_it_is_bounded_downstream():
    """An absorbing T normal +X 0.1 mm behind a fold mirror's hit point takes the light above y = 0.1 mm; the folded beam
    goes on to E: E reads the beam on ``[P - S, P]`` with ``S`` the bound of T's share, which holds the light that is
    there, ``P - part(0.1)``; T is bounded."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0))
    t = _monitor(0.1, absorbing=True)
    e = B.observation_plane("E", position=(0.0, 50.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0)
    result = run(ProjectSnapshot(components=(_laser(), fold), observations=(t, e), targets=(_maximum(1e-9),)))
    obs = _observation(result)
    assert obs.arrivals == () and obs.total_power_W is None and obs.quantity_state.value == "BOUNDED"
    assert _part(0.1) < obs.power_upper_bound_W <= 1.001 * _part(0.1)  # (and the folded beam's flow on its way to E)
    arrival = one_arrival(result, "E")
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("observation/T",)
    assert (1.0 - 1e-9) * _part(0.1) <= arrival.power_taken_before_upper_W <= 1.001 * _part(0.1)
    plane = _observation(result, "E")
    assert plane.quantity_state.value == "BOUNDED"
    assert plane.power_lower_bound_W <= (SOURCE_W - _part(0.1)) * (1.0 + 1e-9) and plane.total_power_W == pytest.approx(SOURCE_W, rel=1e-12)


# ----------------------------------------------------------------- a plane along the beam (parallel to its chief ray)

#: a strongly diverging beam: waist 10 um at the source (x = -10 mm), its radius about 25 um per mm further on
NARROW = 0.01
ALONG_Y = (0.0, 1.0, 0.0)


def _narrow_radius(s: float) -> float:
    return NARROW * math.sqrt(1.0 + (s / (math.pi * NARROW * NARROW / (LAM * 1e-6))) ** 2)


def _beyond(offset: float, radius: float) -> float:
    return SOURCE_W * 0.5 * math.erfc(math.sqrt(2.0) * offset / radius)


def _along_the_beam(x_centre: float, **area) -> ProjectSnapshot:
    """T normal +Y through (x_centre, 2, 0): parallel to the beam, 2 mm off its axis; an absorbing D at x = 90 mm ends
    the leg 100 mm from the waist."""
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=NARROW, power_W=SOURCE_W, position=(-10.0, 0.0, 0.0), direction=X)
    t = B.observation_plane("T", position=(x_centre, 2.0, 0.0), normal=ALONG_Y, **area)
    d = B.observation_plane("D", position=(90.0, 0.0, 0.0), normal=X, radius_mm=50.0, absorbing=True)
    return ProjectSnapshot(components=(laser,), observations=(t, d), targets=(_maximum(1e-9),))


def test_a_plane_along_a_diverging_beam_is_bounded_by_the_light_that_crosses_it_on_its_active_area():
    """The beam's energy flows along lines that keep their place in its profile, so the light that crosses T on its 2 mm
    active area (48-52 mm from the waist) is the share beyond T where the beam is widest there less where it is narrowest
    -- about 0.6 uW, not the 54 uW beyond T at the end of the leg."""
    result = run(_along_the_beam(40.0, radius_mm=2.0))
    _assert_bounded_from_zero(_observation(result), _beyond(2.0, _narrow_radius(52.0)) - _beyond(2.0, _narrow_radius(48.0)))
    assert result.targets[0].assessment.value == "INDETERMINATE"


def test_a_plane_along_the_beam_whose_active_area_lies_beyond_the_legs_end_stays_a_known_zero():
    result = run(_along_the_beam(140.0, radius_mm=2.0))
    assert _observation(result).quantity_state.value == "KNOWN_ZERO"
    assert result.targets[0].assessment.value == "MET"


def test_a_plane_along_the_beam_without_an_active_area_is_bounded_by_the_share_beyond_it_where_the_beam_is_widest():
    result = run(_along_the_beam(40.0))
    _assert_bounded_from_zero(_observation(result), _beyond(2.0, _narrow_radius(100.0)))


# ------------------------------------------- an absorbing plane acts on its active area only

ALONG_D = (Q, -Q, 0.0)  # (along a 45 deg plane, away from the beam)


def _absorbing_far_area(through) -> object:
    """An absorbing plane at 45 deg through ``through`` whose 1 mm active area lies 20 mm along the plane, out of the beam."""
    centre = (through[0] + 20.0 * ALONG_D[0], through[1] + 20.0 * ALONG_D[1], 0.0)
    return B.observation_plane("D", position=centre, normal=DIAGONAL, radius_mm=1.0, absorbing=True)


def test_rays_crossing_an_absorbing_plane_beside_its_active_area_are_not_taken_out_of_the_beam():
    """Case c16 with an absorbing D: the rays that cross D's plane before the fold mirror, outside its active area,
    go on; E behind the mirror equals the scene without D (96/96 pupil rays, the same spot), D receives nothing."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99)
    e = B.observation_plane("E", position=(0.0, 60.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0)
    targets = (Target(target_id="erms", metric="spot_rms_mm", observation_id="E", maximum=0.34, unit="mm"),)
    control = run(ProjectSnapshot(components=(_laser(), fold), observations=(e,), targets=targets))
    result = run(ProjectSnapshot(components=(_laser(), fold), observations=(_absorbing_far_area((0.3, 0.3)), e),
                                 targets=(*targets, _maximum(1e-9, "D"))))
    assert one_arrival(result, "E") == one_arrival(control, "E")
    assert one_arrival(result, "E").pupil_rays_arrived == 96
    assert {t.target_id: t.assessment.value for t in result.targets} == {"erms": "UNMET", "Dmax": "MET"}


def test_a_plane_tied_with_an_absorbing_plane_whose_area_is_elsewhere_is_not_out_of_scope():
    """T (normal +X) and an absorbing D through one point, D's active area 20 mm away: D neither stops nor takes part of
    the beam, so T's power is the beam's (a 0.9 mW minimum MET) and E behind gets the whole beam."""
    t = B.observation_plane("T", position=(0.0, 0.0, 0.0), normal=X, radius_mm=50.0)
    e = B.observation_plane("E", position=(30.0, 0.0, 0.0), normal=X, radius_mm=50.0)
    tmin = Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.9e-3, unit="W")
    result = run(ProjectSnapshot(components=(_laser(),), observations=(t, _absorbing_far_area((0.0, 0.0)), e), targets=(tmin,)))
    assert one_arrival(result, "T").power_out_of_scope_at == ()
    assert one_arrival(result, "E").pupil_rays_arrived == 96
    assert result.targets[0].assessment.value == "MET"


# --------------------------------------------- a curved end surface; a physical surface met first


def _convex_end(c: float, convex: bool):
    """r2/curved_end.py: a collimated 1 mm beam onto a spherical mirror (R 5 mm, vertex at the origin), T normal
    (1, 0.2, 0) through (c, 0, 0), crossed by the chief ray only beyond the vertex."""
    mirror = B.spherical_mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=-5.0 if convex else 5.0,
                                clear_radius_mm=4.0)
    t = B.observation_plane("T", position=(c, 0.0, 0.0), normal=B.unit((1.0, 0.2, 0.0)), radius_mm=50.0)
    return ProjectSnapshot(components=(B.laser("L", wavelength_nm=LAM, waist_radius_mm=1.0, power_W=SOURCE_W, position=(-10.0, 0.0, 0.0),
                                               direction=X), mirror), observations=(t,), targets=(_maximum(1e-6),))


@pytest.mark.parametrize("c", (0.4, 0.5))
def test_a_plane_beside_a_convex_end_surface_is_bounded_by_what_the_bulge_lets_through(c: float):
    """Off-axis rays meet a convex surface later than its tangent plane: the light that crosses T in between (2.95 uW at
    c = 0.5 mm, 10 uW at 0.4 mm, by Monte Carlo) used to be left out of the bound (0.29 nW, 32 nW: a false MET). The end is
    now the plane across the ray as far as the sphere reaches within the beam's envelope: the bound covers it."""
    result = run(_convex_end(c, convex=True))
    obs = _observation(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_upper_bound_W >= 10e-6
    assert result.targets[0].assessment.value == "INDETERMINATE"


def test_a_plane_beside_a_concave_end_surface_keeps_its_tight_bound():
    """A concave surface curves toward the beam: off-axis rays stop earlier; the bound of the leg to the mirror stays the
    tangent-plane share (0.29 nW). The focused beam it sends back has no valid Gaussian state (far field 0.4 rad), and T
    lies within three envelopes of its pupil rays, so that leg's bound is the whole beam, as evidence: the 1 uW maximum on a
    plane with no traced arrival stays INDETERMINATE."""
    project = _convex_end(0.5, convex=False)
    arriving, leaving = _trace(project).partial_crossings
    assert (arriving.ends_at, leaving.ends_at) == ("M/face", None)
    assert arriving.power_W < 1e-9 and leaving.power_W == pytest.approx(SOURCE_W, rel=1e-12)
    assert run(project).targets[0].assessment.value == "INDETERMINATE"


def _before_fold(gap: float, w: float):
    """r2/partial_miss.py: T normal +X a distance ``gap`` before a 45 deg fold mirror's hit point, a beam of radius ``w``."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=1.0, clear_radius_mm=20.0)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=w, power_W=SOURCE_W, position=(-30.0, 0.0, 0.0), direction=X)
    t = B.observation_plane("T", position=(-gap, 0.0, 0.0), normal=X, radius_mm=50.0)
    return ProjectSnapshot(components=(laser, fold), observations=(t,), targets=(
        Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.9e-3, unit="W"),))


@pytest.mark.parametrize(("gap", "w"), ((0.1, 0.5), (0.5, 3.0), (2.0, 3.0)))
def test_a_plane_just_before_a_fold_mirror_that_part_of_the_beam_meets_first_reads_an_interval(gap: float, w: float):
    """The mirror reflects the near part of the beam before it reaches T (63 % to 91 % of it arrives: 1 - 0.5 erfc(sqrt(2)
    gap / w)): T records the chief ray's 1 mW with the bounded share the mirror takes first, so its power lies in
    [1 mW - S, 1 mW] with S at least that share; a 0.9 mW minimum is met only where the lower end clears it (it used to read
    1 mW: a false MET; then it was out of model scope)."""
    result = run(_before_fold(gap, w))
    arrival = one_arrival(result, "T")
    share = 0.5 * math.erfc(math.sqrt(2.0) * gap / w) * SOURCE_W
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("M/face",)
    assert share <= arrival.power_taken_before_upper_W * (1.0 + 1e-9)
    (target,) = result.targets
    lower = SOURCE_W - arrival.power_taken_before_upper_W
    assert target.assessment.value == ("MET" if lower >= 0.9e-3 else "INDETERMINATE")
    assert target.message.startswith("M/face takes at most")


def test_a_plane_well_before_a_fold_mirror_reads_the_beam():
    result = run(_before_fold(20.0, 3.0))
    assert one_arrival(result, "T").power_out_of_scope_at == ()
    assert result.targets[0].assessment.value == "MET"


# ----------------------------------------- legs that start or end at a sag surface (never weaker than a plane)

from os_optics.contracts import CircularAperture, EngineeringComponentSnapshot, MirrorInteraction, Pose3D, SagSurface  # noqa: E402

SAG_PROFILES = {
    "flat": {"curvature_per_mm": 0.0},
    "concave_R1000": {"curvature_per_mm": 1e-3},
    "paraboloid_R200": {"curvature_per_mm": 1.0 / 200.0, "conic": -1.0},
    "cylinder_R200": {"curvature_per_mm": 1.0 / 200.0, "cylindrical": True},
    "axicon": {"curvature_per_mm": 0.0, "linear_sag_coefficient": 0.01},
}


def _sag_mirror(normal, u_axis, **profile):
    face = SagSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=normal, u_axis=u_axis, aperture=CircularAperture(radius_mm=12.7),
                      interaction=MirrorInteraction(reflective_side="positive"), **profile)
    return EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D(), surfaces=(face,))


def _laser_30():
    return B.laser("L", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=W0, position=(-30.0, 0.0, 0.0), direction=X)


@pytest.mark.parametrize("profile", sorted(SAG_PROFILES))
def test_the_tail_that_crosses_a_plane_before_a_sag_end_surface_is_bounded(profile: str):
    """r3/sag_end.py: the beam ends on a sag mirror at the origin (facing -X); T (x + y = 0.7) is crossed by the tail
    beyond 0.7 mm (no pupil ray) before the mirror: 2.56 uW, bounded like a plane mirror's, never a known 0 W."""
    t = B.observation_plane("T", position=(0.7, 0.0, 0.0), normal=DIAGONAL, radius_mm=50.0)
    result = run(ProjectSnapshot(components=(_laser_30(), _sag_mirror((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), **SAG_PROFILES[profile])),
                                 observations=(t,), targets=(_maximum(1e-9),)))
    obs = _observation(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_upper_bound_W >= _part(0.7, SOURCE_W) * (1.0 - 1e-6)
    assert result.targets[0].assessment.value == "INDETERMINATE"


@pytest.mark.parametrize("profile", ("flat", "paraboloid_R200", "cylinder_R200", "axicon"))
def test_the_tail_that_crosses_a_plane_after_a_sag_start_surface_is_bounded(profile: str):
    """r3/sag_start.py: a 45 deg sag fold sends the beam to +Y; absorbing D at y = 40 stops it; T (x + y = 40.7) is crossed
    by the tail first: bounded (2.6 uW), as after a plane fold."""
    d = B.observation_plane("D", position=(0.0, 40.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0, absorbing=True)
    t = B.observation_plane("T", position=(0.0, 40.7, 0.0), normal=DIAGONAL, radius_mm=50.0)
    fold = _sag_mirror((-Q, Q, 0.0), (Q, Q, 0.0), **SAG_PROFILES[profile])
    result = run(ProjectSnapshot(components=(_laser_30(), fold), observations=(d, t), targets=(_maximum(1e-9),)))
    obs = _observation(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_upper_bound_W >= 2e-6
    assert result.targets[0].assessment.value == "INDETERMINATE"


def test_an_absorbing_plane_after_a_sag_fold_that_takes_part_of_the_beam_bounds_a_later_planes_power():
    """r3/sag_start_absorbing.py: after a flat sag fold, an absorbing D (x + y = 40.7) takes the 2.56 uW tail before T at
    y = 40, which the chief ray crosses first: T's power lies between the beam's less the bounded share D takes and the
    beam's (it was out of model scope), so a minimum within that share of the beam is not decided."""
    d = B.observation_plane("D", position=(0.0, 40.7, 0.0), normal=DIAGONAL, radius_mm=50.0, absorbing=True)
    t = B.observation_plane("T", position=(0.0, 40.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0)
    fold = _sag_mirror((-Q, Q, 0.0), (Q, Q, 0.0), **SAG_PROFILES["flat"])
    tmin = Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.999999e-3, unit="W")
    result = run(ProjectSnapshot(components=(_laser_30(), fold), observations=(d, t), targets=(tmin,)))
    arrival = one_arrival(result, "T")
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("observation/D",)
    assert arrival.power_taken_before_upper_W >= 2.56e-6
    assert _observation(result, "T").quantity_state.value == "BOUNDED"
    assert result.targets[0].assessment.value == "INDETERMINATE"
    assert "POWER_TAKEN_BEFORE_PLANE" in result.targets[0].issue_codes


# ------------------------------------- a plane just after a fold mirror; an ND plate met first; the text


def _after_fold(gap: float, w: float):
    """r3/after_fold.py: T normal +Y a distance ``gap`` after a 45 deg fold mirror's hit point on the reflected leg: the
    part of the beam that reflects beyond T's plane never crosses T."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99, clear_radius_mm=20.0)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=w, power_W=SOURCE_W, position=(-30.0, 0.0, 0.0), direction=X)
    t = B.observation_plane("T", position=(0.0, gap, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0)
    return ProjectSnapshot(components=(laser, fold), observations=(t,), targets=(
        Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.891e-3, unit="W"),))


@pytest.mark.parametrize(("gap", "w"), ((0.1, 0.5), (0.5, 3.0), (2.0, 3.0)))
def test_a_plane_just_after_a_fold_mirror_that_part_of_the_beam_leaves_beyond_reads_it_within_that_share(gap: float, w: float):
    """0.649 / 0.624 / 0.900 mW arrive, the engine records the beam's 0.99 mW: the share the mirror sends beyond the plane,
    0.5 erfc(sqrt(2) gap / w) of it, is bounded (the energy flow's lines at the mirror, ``decisions.start_flow_share``), so
    the plane reads [0.99 mW - S, 0.99 mW]: a 0.891 mW minimum is not decided at 0.1 / 0.5 mm and met at 2 mm (where 0.900
    mW arrive; it was once MET at all three, then out of model scope)."""
    result = run(_after_fold(gap, w))
    arrival = one_arrival(result, "T")
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("M/face",)
    share = 0.5 * math.erfc(math.sqrt(2.0) * gap / w) * arrival.beam.power_W
    assert share <= arrival.power_taken_before_upper_W * (1.0 + 1e-9) and arrival.power_taken_before_upper_W <= 1.001 * share
    assert result.targets[0].assessment.value == ("MET" if gap == 2.0 else "INDETERMINATE")


def test_a_plane_well_after_a_fold_mirror_reads_the_beam():
    result = run(_after_fold(20.0, 3.0))
    assert one_arrival(result, "T").power_out_of_scope_at == ()
    assert result.targets[0].assessment.value == "MET"


def test_a_plate_met_first_is_named_as_met_first_not_as_a_variable_attenuator():
    """r3/nd_met_first.py: T 0.1 mm before a 45 deg fixed-density ND plate: part of the beam meets the plate first. The
    cause is that part (a bounded share for a power, OUT_OF_MODEL_SCOPE for a coherent power), not a density varying across
    the beam (it was ND_DENSITY_VARIES_ACROSS_BEAM)."""
    from os_optics.contracts import PlaneSurface  # noqa: PLC0415
    from os_optics.contracts.optics import FixedDensity, NeutralDensityInteraction  # noqa: PLC0415

    plate = EngineeringComponentSnapshot(instance_id="ND", role="filter", pose=B._pose((0.0, 0.0, 0.0), (Q, -Q, 0.0)), surfaces=(
        PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=12.7),
                     interaction=NeutralDensityInteraction(law=FixedDensity(optical_density=1.0))),))
    t = B.observation_plane("T", position=(-0.1, 0.0, 0.0), normal=X, radius_mm=50.0)
    result = run(ProjectSnapshot(components=(_laser_30(), plate), observations=(t,), targets=(
        Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.5e-3, unit="W"),)))
    (target,) = result.targets
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" not in target.issue_codes and "POWER_TAKEN_BEFORE_PLANE" in target.issue_codes
    assert target.message.startswith("ND/face takes at most")
    assert "variable attenuator" not in target.message
    coherent = Target(target_id="c", metric="coherent_power_W", observation_id="T", minimum=0.5e-3, unit="W")
    result = run(ProjectSnapshot(components=(_laser_30(), plate), observations=(t,), targets=(coherent,)))
    (target,) = result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and "ND_DENSITY_VARIES_ACROSS_BEAM" not in target.issue_codes
    assert target.message.startswith("part of a beam of this target does not reach this plane with the beam: it meets ND/face")


def test_a_lit_plane_a_partial_crossing_also_reaches_states_the_bound_once():
    """A lit T behind a lens that part of the beam also crosses before the lens: the target names that part and its
    bound once, then the interval."""
    lens = B.thin_lens("F", position=(0.0, 0.0, 0.0), focal_length_mm=200.0, clear_radius_mm=10.0)
    t = B.observation_plane("T", position=(0.3, 0.0, 0.0), normal=DIAGONAL, radius_mm=50.0)
    result = run(ProjectSnapshot(components=(_laser(), lens), observations=(t,), targets=(_maximum(2e-3),)))
    message = result.targets[0].message
    assert message.startswith("the traced light gives 0.001 W, and part of a traced beam of the target's source lines reaches it "
                              "where the beam's chief ray does not cross it: that part adds at most ")
    assert message.count("at most") == 1 and "followed through the optics" not in message


# ------------------------- sag surfaces: the reach over the part the beam can meet; the causes apart


def _oap():
    """The catalog's 90 deg OAP (RFL 101.6 mm, clear radius 5.715 mm 101.6 mm off the parent axis) at contract level:
    the beam comes in along -Z and leaves along -X."""
    return EngineeringComponentSnapshot(instance_id="OAP", role="mirror", pose=Pose3D(), surfaces=(SagSurface(
        surface_id="S1", origin_mm=(-101.6, 0.0, -50.8), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), curvature_per_mm=1.0 / 101.6,
        conic=-1.0, aperture=CircularAperture(radius_mm=5.715, center_offset_mm=(101.6, 0.0)), mount_margin_mm=0.635,
        interaction=MirrorInteraction(reflective_side="positive")),))


def _down_beam():
    return B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=0.3, position=(0.0, 0.0, 40.0), direction=(0.0, 0.0, -1.0))


@pytest.mark.parametrize("gap", (1.0, 2.0, 4.0))
def test_monitors_close_before_and_after_an_off_axis_paraboloid_read_the_beam(gap: float):
    """T1 before the OAP and T2 after it, 1-4 mm away: the whole beam crosses both (the share beyond
    1 mm is 1.3e-11). Taken over the whole off-axis support the paraboloid's depth refused both."""
    t1 = B.observation_plane("T1", position=(0.0, 0.0, gap), normal=(0.0, 0.0, -1.0), radius_mm=30.0)
    t2 = B.observation_plane("T2", position=(-gap, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=30.0)
    result = run(ProjectSnapshot(components=(_down_beam(), _oap()), observations=(t1, t2), targets=(
        Target(target_id="t1", metric="power_W", observation_id="T1", minimum=0.5e-3, unit="W"),
        Target(target_id="t2", metric="power_W", observation_id="T2", minimum=0.5e-3, unit="W"))))
    assert one_arrival(result, "T1").power_out_of_scope_at == () and one_arrival(result, "T2").power_out_of_scope_at == ()
    assert [t.assessment.value for t in result.targets] == ["MET", "MET"]


def test_a_plane_behind_an_off_axis_paraboloid_stays_a_known_zero():
    """T3: 10 mm behind the OAP no light arrives -- a known 0 W (the whole support's depth gave it a
    whole-beam bound)."""
    t3 = B.observation_plane("T3", position=(0.0, 0.0, -10.0), normal=(0.0, 0.0, 1.0), radius_mm=5.0)
    result = run(ProjectSnapshot(components=(_down_beam(), _oap()), observations=(t3,), targets=(_maximum(1e-6, "T3"),)))
    assert _observation(result, "T3").quantity_state.value == "KNOWN_ZERO"
    assert result.targets[0].assessment.value == "MET"


def test_a_plane_between_two_close_folds_names_each_surface_by_what_it_does():
    """T between fold M (the leg's start, +X to +Y) and fold N (its end, +Y to -X), 1 mm from each, a 0.5 mm beam: part
    of the beam is reflected by M beyond T and part meets N before T; the message says which, once each."""
    m = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=1.0, clear_radius_mm=5.0)
    n = B.mirror("N", position=(0.0, 2.0, 0.0), normal=(-Q, -Q, 0.0), reflectance=1.0, clear_radius_mm=5.0)
    laser = B.laser("L", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=0.5, position=(-30.0, 0.0, 0.0), direction=X)
    t = B.observation_plane("T", position=(0.0, 1.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0)
    result = run(ProjectSnapshot(components=(laser, m, n), observations=(t,), targets=(
        Target(target_id="tmin", metric="power_W", observation_id="T", minimum=0.5e-3, unit="W"),)))
    assert set(one_arrival(result, "T").power_out_of_scope_at) == {"M/face", "N/face"}
    assert result.targets[0].message.startswith("part of a beam of this target does not reach this plane with the beam: it meets "
                                                "N/face before the plane or is reflected by M/face beyond the plane, while ")


# ------------------------------- the curved surfaces' planes against the engine's own intersection


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / math.sqrt(float(v @ v))


def _basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(a - (a @ n) * n)
    return u, np.cross(n, u)


def _random_sag(rng):
    """A conic (sphere-like, paraboloid, hyperboloid, ellipsoid), strong even-asphere terms, an axicon or a pure
    polynomial, rotational or cylindrical, on a centred, off-axis or rectangular support with or without a mount ring."""
    from os_optics.scene.model import ApertureG, InteractionG, SurfaceG  # noqa: PLC0415

    n = _unit(rng.normal(size=3))
    u, v = _basis(n)
    turn = rng.uniform(0.0, 2.0 * math.pi)
    u, v = math.cos(turn) * u + math.sin(turn) * v, -math.sin(turn) * u + math.cos(turn) * v
    kind = int(rng.integers(0, 5))
    c = float(rng.choice([-1, 1]) * 10 ** rng.uniform(-2.7, -0.7))
    k = float(rng.choice([0.0, -1.0, rng.uniform(-6.0, -1.01), rng.uniform(0.01, 3.0)]))
    coefficients, linear = (), 0.0
    if kind == 1:
        coefficients = tuple(float(rng.choice([-1, 1]) * 10 ** rng.uniform(-6.0, -2.5) / 10 ** (1.6 * j))
                             for j in range(int(rng.integers(1, 5))))
    elif kind == 2:
        linear = float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.8))
    elif kind == 3:
        c, coefficients = 0.0, (float(rng.choice([-1, 1]) * 10 ** rng.uniform(-5.0, -3.0)),)
    domain = 1.0 / math.sqrt((1.0 + k) * c * c) if (1.0 + k) * c * c > 0.0 else 60.0
    if rng.random() < 0.6:
        off = (float(rng.uniform(-1, 1) * min(domain, 120.0)), float(rng.uniform(-1, 1) * min(domain, 40.0)))
        if rng.random() < 0.5:
            off = (0.0, 0.0)
        aperture = ApertureG("circle", float(rng.uniform(1.0, 25.0)), 0.0, off[0], off[1])
    else:
        aperture = ApertureG("rectangle", float(rng.uniform(1.0, 25.0)), float(rng.uniform(1.0, 25.0)),
                             float(rng.uniform(-30.0, 30.0)) if rng.random() < 0.4 else 0.0, 0.0)
    margin = float(rng.choice([0.0, rng.uniform(0.1, 3.0)]))
    return SurfaceG("S/face", "S", "sag", rng.normal(size=3) * 20.0, n, u, v, np.zeros(3), 0.0, n, aperture, margin, "air", "air",
                    InteractionG("mirror"), curvature=c, conic=k, coefficients=coefficients, cylindrical=bool(rng.random() < 0.3),
                    linear=linear)


def _random_sphere(rng):
    from os_optics.scene.model import ApertureG, InteractionG, SurfaceG  # noqa: PLC0415

    radius = float(10 ** rng.uniform(0.5, 2.6))
    axis = _unit(rng.normal(size=3))
    clear = float(min(rng.uniform(1.0, 30.0), radius * rng.uniform(0.2, 1.0)))
    return SurfaceG("S/face", "S", "sphere", np.zeros(3), axis, *_basis(axis), rng.normal(size=3) * 20.0, radius, axis,
                    ApertureG("circle", clear), float(rng.choice([0.0, rng.uniform(0.1, 2.0)])), "air", "air", InteractionG("mirror"),
                    aperture_radius=clear)


def _point_on(rng, surf, near_edge: bool):
    """A point of the surface (the chief ray's hit), near its support's edge or anywhere, and on a sag surface its
    support coordinates."""
    from os_optics.optics.surfaces import sag as sg  # noqa: PLC0415
    from os_optics.optics.surfaces.geometry import support_of  # noqa: PLC0415

    for _ in range(400):
        if surf.geometry == "sag":
            sup = support_of(surf)
            half_v = sup.p1 if sup.shape == "circle" else sup.p2
            if near_edge and sup.shape == "circle":
                phi, f = rng.uniform(0.0, 2.0 * math.pi), 1.0 - rng.uniform(0.0, 0.12)
                a, b = sup.offset_u + f * sup.p1 * math.cos(phi), sup.offset_v + f * sup.p1 * math.sin(phi)
            elif near_edge:
                a = sup.offset_u + rng.choice([-1, 1]) * (1.0 - rng.uniform(0.0, 0.12)) * sup.p1
                b = sup.offset_v + rng.uniform(-1, 1) * half_v
            else:
                a, b = sup.offset_u + rng.uniform(-1, 1) * sup.p1, sup.offset_v + rng.uniform(-1, 1) * half_v
                if sup.shape == "circle" and (a - sup.offset_u) ** 2 + (b - sup.offset_v) ** 2 > sup.p1 ** 2:
                    continue
            z, ok = sg.sag(surf, sg.transverse_s(surf, np.array([a]), np.array([b])))
            if bool(ok[0]) and math.isfinite(float(z[0])):
                return surf.origin + a * surf.u_axis + b * surf.v_axis + float(z[0]) * surf.normal, (a, b)
            continue
        radius, axis = abs(surf.radius), surf.cap_axis
        support = surf.aperture_radius + surf.mount_margin
        rim = math.sqrt(max(radius * radius - support * support, 0.0))
        w = _unit(rng.normal(size=3))
        if near_edge:
            w = _unit(_unit(w - (w @ axis) * axis) * support / radius + axis * rim / radius * rng.uniform(1.0, 1.05))
        if float(w @ axis) * radius >= rim:
            return surf.center + radius * w, None
    return None, None


def _all_roots(surf, starts, d, cfg):
    """(N, 6): the ray parameters (from ``starts`` along ``d``) of every root of each line, from 400 mm back; inf padded."""
    from os_optics.optics.surfaces.geometry import intersect  # noqa: PLC0415

    count = starts.shape[0]
    out = np.full((count, 6), np.inf)
    o, base = starts - 400.0 * d, np.full(count, -400.0)
    rays = np.repeat(d[None, :], count, axis=0)
    alive = np.ones(count, dtype=bool)
    for j in range(6):
        if not alive.any():
            break
        t = np.full(count, np.inf)
        t[alive] = intersect(surf, o[alive], rays[alive], cfg).t
        hit = alive & np.isfinite(t)
        step = np.where(hit, t, 0.0)
        out[hit, j] = base[hit] + t[hit]
        base = base + step
        o = o + step[:, None] * rays
        alive = hit
    return out


def _direction(rng, surf, point, axis, n_t):
    """A leg direction: random, near the axis' plane, grazing the tangent plane at the point, along the axis, or along a
    chord of the surface (a line that may meet it twice); ``None``: no chord found."""
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return _unit(rng.normal(size=3))
    if mode == 1:
        u, v = _basis(axis)
        return _unit(axis * rng.uniform(-0.06, 0.06) + u * rng.normal() + v * rng.normal())
    if mode == 2:
        u, v = _basis(n_t)
        return _unit(n_t * rng.choice([-1, 1]) * rng.uniform(0.03, 0.35) + u * rng.normal() + v * rng.normal())
    if mode == 3:
        return _unit(rng.choice([-1, 1]) * axis + rng.normal(size=3) * rng.uniform(0.0, 0.3))
    other, _ = _point_on(rng, surf, False)
    if other is None or float(np.linalg.norm(point - other)) < 1e-6:
        return None
    chord = point - other
    return _unit(chord * rng.choice([1, -1]) + rng.normal(size=3) * 0.05 * float(np.linalg.norm(chord)))


def _curved_plane_violations(trials: int, seed: int, sag_share: float = 0.75) -> dict[str, int]:
    """Random sag surfaces and spheres (``_random_sag``, ``_random_sphere``), a point P on each (the chief ray's hit),
    leg directions d (``_direction``) and a reach r. Each ray P + rho + t d (|rho| <= r: 8 rings x 48) is intersected
    with the surface along its whole line by the engine's own ``intersect``; with ``t_P`` the ray's crossing of the
    plane through P (``_curved_plane``'s normal), the counts of rays (beyond 1e-7 mm) where
    * C (``_curved_plane``, side +1 / -1: a leg's end / start surface; ``_start_beyond``): the root that continues P's
      (nearest ``t_P``) lies after ``t_P + ahead`` or before ``t_P - back``;
    * M (``_curved_plane(..., -1, every_hit=True)``: the surface a ray meets first, ``_taken_before``): the ray's first
      root after the leg start (just after the chief line's last root before P) lies before ``t_P - back`` -- on a sag
      surface and on a sphere (the other root of a line on the cap, the near rim of a steep cap)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from os_optics.optics.surfaces import sag as sg  # noqa: PLC0415

    cfg, rng = NumericalConfig(), np.random.default_rng(seed)
    counts = {"configs": 0, "C rays": 0, "M rays": 0, "C": 0, "M": 0}
    for _ in range(trials):
        surf = _random_sag(rng) if rng.random() < sag_share else _random_sphere(rng)
        point, ab = _point_on(rng, surf, bool(rng.random() < 0.4))
        if point is None:
            continue
        tracer = Tracer.__new__(Tracer)
        tracer.scene, tracer.cfg = SimpleNamespace(surfaces=[surf]), cfg
        axis = surf.normal if surf.geometry == "sag" else surf.cap_axis
        n_t = sg.local_normals(surf, np.array([ab[0]]), np.array([ab[1]]))[0] if ab is not None else _unit(point - surf.center)
        for _ in range(4):
            d = _direction(rng, surf, point, axis, n_t)
            if d is None:
                continue
            reach = float(10 ** rng.uniform(-1.5, 0.9))
            chief = _all_roots(surf, point[None, :], d, cfg)[0]
            chief = chief[np.isfinite(chief)]
            if chief.size == 0 or float(np.min(np.abs(chief))) > 1e-6:
                continue
            before = chief[chief < -1e-6]
            s_start = float(before.max()) + 1e-6 if before.size else -399.0
            e1, e2 = _basis(d)
            rings, phi = np.meshgrid(np.linspace(0.0, 1.0, 9)[1:] * reach, np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False),
                                     indexing="ij")
            rho = rings.ravel()[:, None] * (np.cos(phi).ravel()[:, None] * e1 + np.sin(phi).ravel()[:, None] * e2)
            roots = _all_roots(surf, point + rho, d, cfg)
            finite = np.where(np.isfinite(roots), roots, 1e300)
            counts["configs"] += 1
            checks = [(1.0, False), (-1.0, False), (-1.0, True)]
            for side, every_hit in checks:
                normal, back, ahead = tracer._curved_plane(0, point, d, reach, side, every_hit=every_hit)
                if not math.isfinite(ahead if side > 0.0 else back):
                    continue
                t_p = -(rho @ normal) / float(d @ normal)
                if every_hit:
                    first = np.where(roots > s_start, roots, np.inf).min(axis=1)
                    has = np.isfinite(first)
                    counts["M rays"] += int(has.sum())
                    counts["M"] += int((has & ((t_p - back) - first > 1e-7)).sum())
                    continue
                cont = roots[np.arange(roots.shape[0]), np.argmin(np.abs(finite - t_p[:, None]), axis=1)]
                has = np.isfinite(cont)
                excess = cont - (t_p + ahead) if side > 0.0 else (t_p - back) - cont
                counts["C rays"] += int(has.sum())
                counts["C"] += int((has & (excess > 1e-7)).sum())
    return counts


def test_the_curved_surface_planes_hold_the_hit_each_caller_bounds():
    """``_curved_plane_violations``: no ray of the envelope meets the surface outside the bound its caller relies on,
    neither the root that continues the chief ray's hit (a leg's start and end surfaces) nor a ray's first hit on a sag
    surface or a sphere met first (the near rim of a steep surface the beam passes; the sphere)."""
    counts = _curved_plane_violations(60, 3)
    assert counts["configs"] > 120 and counts["C rays"] > 50000 and counts["M rays"] > 20000, counts
    assert counts["C"] == 0 and counts["M"] == 0, counts


def test_a_spheres_met_first_sheet_holds_every_first_hit():
    """Spheres only (``_curved_plane_violations`` with ``sag_share`` 0): no ray of the envelope meets the cap before the
    met-first sheet, also where its first hit is the other root of its line (formerly 2124 of these 97 225 rays
    met the cap before the sheet)."""
    counts = _curved_plane_violations(100, 5, sag_share=0.0)
    assert counts["configs"] > 250 and counts["M rays"] > 60000, counts
    assert counts["C"] == 0 and counts["M"] == 0, counts


def _rim_skim(delta0: float, cylindrical: bool) -> ProjectSnapshot:
    """r5/skim_edge2.py: a concave sag mirror (R 50 mm, axis +Z, vertex at the origin; support radius 28.5 mm centred
    2 mm across the axis, near rim at x = -26.5 mm). A 0.3 mm beam along +X passes ``delta0`` above the near rim and
    lands on the far side (AOI 57 deg); T normal +X at x = 0 lies between the rim and the hit; T >= 0.995 mW."""
    radius, support, rim_x, w = 50.0, 28.5, -26.5, 0.3
    height = radius - math.sqrt(radius * radius - rim_x * rim_x) + delta0
    face = SagSurface(surface_id="S1", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=X, curvature_per_mm=1.0 / radius,
                      cylindrical=cylindrical, aperture=CircularAperture(radius_mm=support, center_offset_mm=(rim_x + support, 0.0)),
                      interaction=MirrorInteraction(reflective_side="positive"))
    mirror = EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D(), surfaces=(face,))
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=w, position=(-60.0, 0.0, height), direction=X)
    t = B.observation_plane("T", position=(0.0, 0.0, height), normal=X, radius_mm=10.0)
    return ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=(
        Target(target_id="t", metric="power_W", observation_id="T", minimum=0.995e-3, unit="W"),))


@pytest.mark.parametrize("cylindrical", (True, False))
@pytest.mark.parametrize("delta0", (0.36, 0.45))
def test_a_beam_that_skims_a_mirrors_near_rim_is_bounded_at_a_plane_before_its_far_side(delta0: float, cylindrical: bool):
    """1.2 w / 1.5 w above the rim, the lowest 0.82 % / 0.14 % of the beam meets the rim (from behind) about 50 mm before
    the chief ray's hit, and no pupil ray is that low: 0.99176 / 0.99865 mW reach T with the beam, which records 1 mW.
    T's power is bounded below by the beam's less the rim's bounded share (the sag surface itself, cell by cell), and above
    by the beam's less the share whose lines certainly meet the rim's blocking back first (``power_taken_before_lower_W``):
    at 1.2 w that misses 0.995 mW (UNMET; INDETERMINATE before the certain share), at 1.5 w the bound is tight enough to meet
    it (it was MET at both: the met-first sheet held only the hit that continues the chief ray's)."""
    result = run(_rim_skim(delta0, cylindrical))
    arrival = one_arrival(result, "T")
    share = {0.36: 0.0082, 0.45: 0.0014}[delta0] * SOURCE_W  # (the rim's share, independently integrated)
    assert arrival.power_taken_before_at == ("M/S1",)
    assert arrival.power_taken_before_lower_W <= share <= arrival.power_taken_before_upper_W  # (the bounds hold the rim's share)
    lower, upper = SOURCE_W - arrival.power_taken_before_upper_W, SOURCE_W - arrival.power_taken_before_lower_W
    expected = "MET" if lower >= 0.995e-3 else "UNMET" if upper < 0.995e-3 else "INDETERMINATE"
    assert result.targets[0].assessment.value == expected == {0.36: "UNMET", 0.45: "MET"}[delta0]


def _sphere_rim_skim(delta0: float) -> ProjectSnapshot:
    """review5/skim_sphere.py: the bowl of ``_rim_skim`` as a spherical cap (R 50 mm, centre (0, 0, 50), its axis tilted
    2.85 deg toward +X, half-angle 34.85 deg: near rim at x = -26.5 mm). A 0.3 mm beam along +X passes ``delta0`` above
    the near rim and lands on the far side (AOI 57 deg); T normal +X at x = 0; T >= 0.995 mW."""
    from os_optics.contracts import SphereSurface  # noqa: PLC0415

    radius, tilt, half = 50.0, math.radians(2.85), math.radians(34.85)
    rim_x = radius * math.sin(tilt - half)
    height = radius - math.sqrt(radius * radius - rim_x * rim_x) + delta0
    cap = SphereSurface(surface_id="S1", center_mm=(0.0, 0.0, radius), radius_mm=radius, cap_axis=(math.sin(tilt), 0.0, -math.cos(tilt)),
                        aperture_radius_mm=radius * math.sin(half), interaction=MirrorInteraction(reflective_side="negative"))
    mirror = EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D(), surfaces=(cap,))
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=0.3, position=(-60.0, 0.0, height), direction=X)
    t = B.observation_plane("T", position=(0.0, 0.0, height), normal=X, radius_mm=10.0)
    return ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=(
        Target(target_id="t", metric="power_W", observation_id="T", minimum=0.995e-3, unit="W"),))


@pytest.mark.parametrize("delta0", (0.36, 0.45))
def test_a_beam_that_skims_a_spherical_mirrors_near_rim_is_bounded_at_a_plane_before_its_far_side(delta0: float):
    """1.2 w / 1.5 w above the rim, 0.82 % / 0.13 % of the beam meets the near rim of the cap (the other root of its line)
    before T: 0.9918 / 0.9987 mW reach T with the beam, which records 1 mW. T's power is bounded below by the beam's less
    the cap's bounded share (here the whole cap: up to the whole beam) and above by the beam's less the share whose lines
    certainly meet the cap's blocking back first: at 1.2 w that misses 0.995 mW (UNMET; it was MET when the met-first sheet
    of a sphere held only the roots nearest the chief ray's hit, then INDETERMINATE), at 1.5 w nothing is decided."""
    result = run(_sphere_rim_skim(delta0))
    arrival = one_arrival(result, "T")
    share = {0.36: 0.0082, 0.45: 0.0013}[delta0] * SOURCE_W
    assert arrival.power_taken_before_at == ("M/S1",)
    assert arrival.power_taken_before_lower_W <= share <= arrival.power_taken_before_upper_W
    assert result.targets[0].assessment.value == {0.36: "UNMET", 0.45: "INDETERMINATE"}[delta0]


def _catalog_runs():
    """The catalog's plano-concave cylinder lens and 90 deg OAP through the local runtime."""
    import json  # noqa: PLC0415

    from catalog_store_helpers import REAL_CATALOG  # noqa: PLC0415
    from os_optics.local import load_public_catalog, run_optical_project  # noqa: PLC0415
    from os_optics.local.example import BEAM_ALONG_X, _element, _plane, _project, _source  # noqa: PLC0415

    catalog = load_public_catalog(json.loads(REAL_CATALOG.read_text(encoding="utf-8")))
    by_sku = {p.optical_sku: p for p in catalog.products}
    cylinder = next(p for s, p in by_sku.items() if s.startswith("OS-OPT-LENS-CYL-D25P4-FN50-T2-RN25P8-NBK7-UNC"))
    oap = next(p for s, p in by_sku.items() if s.startswith("OS-OPT-MIR-OAP-D12P7-RFL101P6-A90-AG"))
    down, to_minus_x = (0.0, 1.0, 0.0, 0.0), (Q, 0.0, -Q, 0.0)

    def lens_run(gap):
        return run_optical_project(_project(
            [_source("SRC", [-60.0, 0.0, 0.0], waist_radius_mm=0.5), _element("L", "lens", cylinder, catalog, [0.0, 0.0, 0.0], BEAM_ALONG_X)],
            [_plane("T", [-gap, 0.0, 0.0], BEAM_ALONG_X)],
            [{"target_id": "t", "metric": "power_W", "observation_id": "T", "minimum": 0.9e-3, "unit": "W"}]), catalog=catalog)

    def oap_run(gap):
        return run_optical_project(_project(
            [_source("SRC", [0.0, 0.0, 40.0], waist_radius_mm=0.3, quaternion_wxyz=down),
             _element("M", "mirror", oap, catalog, [0.0, 0.0, 0.0], (1.0, 0.0, 0.0, 0.0))],
            [_plane("T1", [0.0, 0.0, gap], down), _plane("T2", [-gap, 0.0, 0.0], to_minus_x)],
            [{"target_id": "t1", "metric": "power_W", "observation_id": "T1", "minimum": 0.5e-3, "unit": "W"},
             {"target_id": "t2", "metric": "power_W", "observation_id": "T2", "minimum": 0.5e-3, "unit": "W"}]), catalog=catalog)

    return lens_run, oap_run


@pytest.mark.parametrize("gap", (0.5, 1.0, 2.0, 3.0))
def test_a_monitor_close_before_a_catalog_cylinder_lens_reads_the_beam(gap: float):
    """OS-OPT-LENS-CYL-D25P4-FN50-T2-RN25P8-NBK7-UNC (concave R 25.8 mm toward the light), w 0.5 mm, T 0.5-3 mm before
    its vertex: the whole 1 mW reaches T before any unknown catalog-glass loss; the later trace warns about
    missing bulk loss, but the upstream target remains decisive (the face's sag cannot refuse it)."""
    lens_run, _ = _catalog_runs()
    result = lens_run(gap)
    assert result.status.value == "WARNING"
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in result.result.issues)
    assert result.result.targets[0].status.value == "PASS"
    arrival = one_arrival(result.result, "T")
    assert arrival.beam.power_W == pytest.approx(1e-3, rel=1e-12)
    assert arrival.power_out_of_scope_at == ()


@pytest.mark.parametrize("gap", (1.0, 2.0, 4.0))
def test_monitors_close_to_a_catalog_off_axis_paraboloid_read_the_beam(gap: float):
    """OS-OPT-MIR-OAP-D12P7-RFL101P6-A90-AG-..., w 0.3 mm: T1 before it and T2 after it, 1-4 mm away, each read the beam
    and the run passes."""
    _, oap_run = _catalog_runs()
    result = oap_run(gap)
    assert result.status.value == "PASS"
    assert one_arrival(result.result, "T1").power_out_of_scope_at == () and one_arrival(result.result, "T2").power_out_of_scope_at == ()
