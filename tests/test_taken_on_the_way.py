"""A bounded share a surface takes beside the chief ray travels with the beam as an interval.

A Gaussian beam passes an absorbing disc beside its chief ray, then a 50:50 splitter; a plane reads each of the two beams
behind it. The beam's power follows its chief ray and still counts the light the disc absorbed. The planes read each beam
on ``[P - S, P]``: ``S`` is the bound of the share the disc took, in W, carried unscaled into both branches (every later
element is passive: the light missing behind it is at most the light taken).

Reference, no Engine code: the flux of a round Gaussian beam of radius ``w`` through a disc of radius ``a`` whose centre
lies ``d`` off the beam's axis, ``int_0^a (4 r / w^2) exp(-2 (r^2 + d^2) / w^2) I0(4 r d / w^2) dr`` (Gauss-Legendre),
with ``w`` the closed-form beam radius at the disc's plane. The light behind the splitter is then ``R (1 - share) P``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import run
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.results import basis_of

P0, LAM = 2e-3, 780.0
X = (1.0, 0.0, 0.0)
S2 = math.sqrt(0.5)


def _disc(radius: float, offset: float, w: float, nodes: int = 800) -> float:
    x, weights = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * radius * (x + 1.0)
    # (exponentially scaled Bessel form: exp(-2 (r - d)^2 / w^2) * [I0(z) e^-z], finite for a disc many radii out)
    z = 4.0 * r * offset / (w * w)
    big = np.maximum(z, 1.0)
    series = (1.0 + 1.0 / (8.0 * big) + 9.0 / (128.0 * big ** 2) + 225.0 / (3072.0 * big ** 3)) / np.sqrt(2.0 * math.pi * big)
    scaled_i0 = np.where(z < 500.0, np.i0(np.minimum(z, 500.0)) * np.exp(-np.minimum(z, 500.0)), series)
    f = (4.0 * r / (w * w)) * np.exp(-2.0 * (r - offset) ** 2 / (w * w)) * scaled_i0
    return float(0.5 * radius * np.sum(weights * f))


def _radius_at(w0: float, z_from_waist: float) -> float:
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    return w0 * math.sqrt(1.0 + (z_from_waist / z_r) ** 2)


def _scene(w0: float, waist_x: float, radius: float, offset: float, *, disc_x: float = -50.0) -> ProjectSnapshot:
    """The source at x = -200 with its waist at ``waist_x``; the disc at ``disc_x``, its centre ``offset`` from the axis;
    a 50:50 splitter at the origin; plane T behind it and plane R on its reflection."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=(-200.0, 0.0, 0.0), direction=X,
                    waist_position_z_mm=waist_x + 200.0)
    dump = B.beam_dump("D", position=(disc_x, offset, 0.0), axis=X, radius_mm=radius)
    splitter = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=(-S2, S2, 0.0), reflectance=0.5, clear_radius_mm=60.0)
    planes = (B.observation_plane("T", position=(80.0, 0.0, 0.0), normal=X), B.observation_plane("R", position=(0.0, 80.0, 0.0), normal=(0.0, 1.0, 0.0)))
    return ProjectSnapshot(components=(laser, dump, splitter), observations=planes,
                           targets=tuple(Target(target_id=f"min_{p.observation_id}", metric="power_W", observation_id=p.observation_id,
                                                minimum=0.45 * P0, unit="W") for p in planes))


def _cases() -> list[tuple[float, float, float, float]]:
    """Waist radius, waist position, disc radius and the distance of the disc's edge from the chief ray in beam radii at the
    disc: edges from 0.4 to 2.6 radii out (pupil rays meet the nearer ones; only the tail meets the farther ones)."""
    rng = np.random.default_rng(521050)
    out = []
    for edge in (0.4, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.3, 2.6):
        w0 = float(rng.uniform(0.3, 2.5))
        waist_x = float(rng.uniform(-400.0, 300.0))  # (converging, diverging or near its waist at the disc)
        out.append((w0, waist_x, float(rng.uniform(1.0, 8.0)), edge))
    return out


@pytest.mark.parametrize("w0, waist_x, radius, edge", _cases())
def test_the_interval_behind_the_next_surface_holds_the_independent_flux_on_both_branches(w0: float, waist_x: float, radius: float,
                                                                                          edge: float):
    w = _radius_at(w0, -50.0 - waist_x)
    offset = radius + edge * w
    share = _disc(radius, offset, w)
    assert 1e-8 < share < 0.3
    result = run(_scene(w0, waist_x, radius, offset))
    assert result.power_budget.conserved
    by_plane = {o.observation_id: o for o in result.observations}
    for plane_id in ("T", "R"):
        plane = by_plane[plane_id]
        (arrival,) = plane.arrivals
        stated = arrival.beam.power_W
        assert stated == pytest.approx(0.5 * P0, rel=1e-9)  # (the chief ray's power: the disc's light still counted)
        there = 0.5 * P0 * (1.0 - share)  # (the light that is there)
        assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("D/dump",)
        # the bound is the disc's share of the whole beam, in W, on either branch -- not half of it
        assert share * P0 <= arrival.power_taken_before_upper_W
        assert plane.quantity_state.value == "BOUNDED"
        assert plane.power_lower_bound_W == pytest.approx(max(stated - arrival.power_taken_before_upper_W, 0.0), rel=1e-9, abs=1e-18)
        assert plane.power_lower_bound_W <= there <= plane.total_power_W == stated
        entry = basis_of(arrival, "beam.power_W")
        assert entry is not None and entry.kind.value == "UPPER_BOUND"
        assert [(o.part.value, o.at, o.upper) for o in entry.omitted] == [("BEAM_PART_ELSEWHERE", ("D/dump",), arrival.power_taken_before_upper_W)]
    (issue,) = [i for i in result.issues if i.location == "surface:D/dump"]
    assert (issue.code.value, issue.status.value) == ("POWER_TAKEN_BEFORE_PLANE", "WARNING")
    context = issue.numerical_context
    bound_W = context.get("taken_power_upper_W", context.get("tail_power_upper_W"))
    assert bound_W is not None and bound_W >= share * P0
    assert all(a.power_taken_before_upper_W == pytest.approx(min(bound_W, a.beam.power_W), rel=1e-12)
               for o in result.observations for a in o.arrivals)
    assert ("pupil_rays_met" in context) == (edge < 1.15)  # (the pupil rays reach 1.18 beam radii at most)
    if "pupil_rays_met" in context:  # (pupil rays meet the disc: the issue also states the flux through its support)
        assert context["support_fraction"] == pytest.approx(share, rel=1e-6)
        assert context["taken_fraction_upper"] >= context["support_fraction"] * (1.0 - 1e-9)
    assert not any(i.status.value == "OUT_OF_MODEL_SCOPE" for i in result.issues)
    # a minimum between the two ends is not decided; the value is the beam's
    for target in result.targets:
        if by_plane[target.observation_id].power_lower_bound_W < 0.45 * P0:
            assert (target.assessment.value, target.issue_codes) == ("INDETERMINATE", ("POWER_TAKEN_BEFORE_PLANE",))
        else:
            assert target.assessment.value == "MET"


def test_a_disc_clear_of_the_beam_changes_nothing():
    """The disc eight beam radii out: its share is below the negligible level, nothing is listed and every power is known."""
    w = _radius_at(1.0, -50.0 - 0.0)
    result = run(_scene(1.0, 0.0, 3.0, 3.0 + 8.0 * w))
    for plane in result.observations:
        (arrival,) = plane.arrivals
        assert arrival.power_taken_before_at == () and arrival.power_out_of_scope_at == () and arrival.power_taken_before_upper_W is None
        assert plane.quantity_state.value == "KNOWN" and plane.power_lower_bound_W is None
        assert basis_of(arrival, "beam.power_W") is None
    assert not [i for i in result.issues if i.location == "surface:D/dump"]
    assert all(t.assessment.value == "MET" for t in result.targets)


def test_an_exposure_behind_the_disc_states_the_incident_power_as_an_upper_value():
    """The splitter's faces behind the disc: the incident power stays the beam's, with the share taken as its entry; the
    peak irradiance is not stated (a clip of share s can ripple the peak by about 2 sqrt(s))."""
    w = _radius_at(1.0, -50.0)
    result = run(_scene(1.0, 0.0, 3.0, 3.0 + 1.5 * w))
    hits = [h for s in result.exposures.surfaces if s.surface_id.startswith("BS/") for h in s.hits]
    assert hits
    share = _disc(3.0, 3.0 + 1.5 * w, w)
    for hit in hits:
        assert hit.status.value == "OUT_OF_MODEL_SCOPE" and hit.peak_irradiance_W_per_cm2 is None
        entry = basis_of(hit, "incident_power_W")
        assert entry.kind.value == "UPPER_BOUND"
        (omitted,) = entry.omitted
        assert (omitted.part.value, omitted.at, omitted.code) == ("BEAM_PART_ELSEWHERE", ("D/dump",), "POWER_TAKEN_BEFORE_PLANE")
        assert share * P0 <= omitted.upper <= hit.incident_power_W


# ------------------------------------------------------------------------------------- tilted discs, focused beams
def _flow_share(w0: float, waist_s: float, normal: np.ndarray, centre: np.ndarray, radius: float, s_end: float) -> float:
    """The fraction of the beam whose energy-flow lines meet a tilted disc: lines ``xi w(s)`` off the chief ray (paraxial,
    ``s`` from the source at the origin along x), each followed to the disc's plane (bisection) and counted where the
    crossing lies within the disc; the profile's mass by a 2-D midpoint rule over +-4.5 radii."""
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    h = 0.02
    xi = np.arange(-4.5, 4.5, h) + h / 2
    gy, gz = np.meshgrid(xi, xi, indexing="ij")
    weight = np.exp(-2.0 * (gy * gy + gz * gz))

    def point(s):
        w = w0 * np.sqrt(1.0 + ((s - waist_s) / z_r) ** 2)
        return np.stack([s + 0.0 * gy, gy * w, gz * w], axis=-1)

    def side(s):
        return (point(s) - centre) @ normal

    lo, hi = np.full(gy.shape, 0.0), np.full(gy.shape, s_end)
    f_lo = side(lo)
    crosses = np.sign(f_lo) != np.sign(side(hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        same = np.sign(side(mid)) == np.sign(f_lo)
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    p = point(0.5 * (lo + hi))
    on_disc = np.linalg.norm((p - centre) - ((p - centre) @ normal)[..., None] * normal, axis=-1) <= radius
    return float(weight[crosses & on_disc].sum() / weight.sum())


def _tilted_cases() -> list[tuple[float, float, float, float, float, float, float]]:
    rng = np.random.default_rng(521051)
    out = []
    for edge in (0.3, 0.6, 0.9, 1.2, 1.5, 1.9):
        out.append((float(rng.uniform(0.06, 0.3)), float(rng.uniform(10.0, 70.0)), float(rng.uniform(20.0, 45.0)),
                    float(rng.uniform(15.0, 65.0)), float(rng.uniform(0.0, 360.0)), float(rng.uniform(1.0, 3.0)), edge))
    return out


@pytest.mark.parametrize("w0, waist_s, cross_s, tilt_deg, turn_deg, radius, edge", _tilted_cases())
def test_a_tilted_disc_beside_a_focused_beam_is_bounded_behind_the_fold(w0: float, waist_s: float, cross_s: float, tilt_deg: float,
                                                                         turn_deg: float, radius: float, edge: float):
    """A focused beam (waists of 0.06-0.3 mm, converging or diverging at the disc); the disc's plane, tilted 15-65 deg to
    the beam, crosses the chief ray ``cross_s`` from the source, the disc's edge ``edge`` beam radii from it; a fold mirror
    80 mm from the source and a plane behind it. The plane's interval holds the light the flow lines leave."""
    tilt, turn = math.radians(tilt_deg), math.radians(turn_deg)
    across = np.array([0.0, math.cos(turn), math.sin(turn)])
    normal = math.cos(tilt) * np.array([1.0, 0.0, 0.0]) + math.sin(tilt) * across
    in_plane = math.cos(tilt) * across - math.sin(tilt) * np.array([1.0, 0.0, 0.0])
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    w = w0 * math.sqrt(1.0 + ((cross_s - waist_s) / z_r) ** 2)
    centre = np.array([cross_s, 0.0, 0.0]) + (radius + edge * w) * in_plane
    share = _flow_share(w0, waist_s, normal, centre, radius, 70.0)
    laser = B.laser("S", wavelength_nm=LAM, power_W=P0, waist_radius_mm=w0, position=(0.0, 0.0, 0.0), direction=X, waist_position_z_mm=waist_s)
    dump = B.beam_dump("D", position=tuple(float(v) for v in centre), axis=tuple(float(v) for v in normal), radius_mm=radius)
    mirror = B.mirror("M", position=(80.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0), clear_radius_mm=25.0)
    plane = B.observation_plane("T", position=(80.0, 30.0, 0.0), normal=(0.0, 1.0, 0.0))
    result = run(ProjectSnapshot(components=(laser, dump, mirror), observations=(plane,)))
    (obs,) = result.observations
    (arrival,) = obs.arrivals
    assert arrival.beam.power_W == pytest.approx(P0, rel=1e-9)
    there = P0 * (1.0 - share)
    if arrival.power_out_of_scope_at:  # (not bounded: no value is stated, as before)
        assert obs.quantity_state.value == "OUT_OF_MODEL_SCOPE"
        return
    if share * P0 > 1e-9 * P0 * 10.0:
        assert arrival.power_taken_before_at == ("D/dump",)
    if arrival.power_taken_before_at:
        assert share * P0 <= arrival.power_taken_before_upper_W * (1.0 + 1e-3)  # (the reference's midpoint rule: 1e-3)
        assert obs.power_lower_bound_W <= there * (1.0 + 1e-3) and there <= obs.total_power_W
        assert obs.quantity_state.value == "BOUNDED"
    else:
        assert share <= 1e-8 and obs.quantity_state.value == "KNOWN"


@pytest.mark.parametrize("edge", [0.4, 2.0])
def test_a_mirror_beside_the_chief_ray_is_no_share_taken_on_the_way(edge: float):
    """A flat mirror disc where the absorbing disc was, its edge 0.4 beam radii out (pupil rays meet it) or 2 radii out (only
    the beam's tail): it does not stop the light it meets -- it turns that share, which may reach a plane behind (an
    acceptance cone, an active area, another branch), so the power stated behind it is no upper end and ``[P - S, P]`` would
    not hold. The planes behind it are out of model scope with the mirror named, as for a share without a bound; the same
    disc absorbing gives the interval (the test above)."""

    w0, radius = 1.0, 3.0
    w = _radius_at(w0, -50.0)
    offset = radius + edge * w
    scene = _scene(w0, 0.0, radius, offset)
    mirror = B.mirror("D", position=(-50.0, offset, 0.0), normal=(-S2, -S2, 0.0), clear_radius_mm=radius)
    scene = scene.model_copy(update={"components": tuple(mirror if c.instance_id == "D" else c for c in scene.components)})
    result = run(scene)
    for plane in result.observations:
        (arrival,) = plane.arrivals
        assert arrival.power_taken_before_at == () and arrival.power_out_of_scope_at == ("D/face",), plane.observation_id
        assert plane.quantity_state.value == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in result.issues if i.location == "surface:D/face" and i.code.value == "OUT_OF_MODEL_SCOPE"]
    assert "may reach a plane behind" in issue.message
    assert not [i for i in result.issues if i.code.value == "POWER_TAKEN_BEFORE_PLANE"]
    assert all(t.assessment.value == "INDETERMINATE" for t in result.targets)
