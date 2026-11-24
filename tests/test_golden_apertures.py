"""Golden cases 9, 10, 13: aperture clipping, projected aperture under oblique incidence, detector acceptance / capture.

References: closed forms (erf / exp) for centred apertures and an independent high-accuracy reference (scipy adaptive quadrature
with an erf inner integral, product of erf differences for rectangles) for off-centre / oblique / obstructed cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot
from os_optics.contracts.optics import CircularAperture
from os_optics.optics.gaussian import initial_state, propagate, transmission_fraction
from os_optics.scene.model import ApertureG

from helpers import CFG, arrivals, issue_codes, one_arrival, rayleigh_range, run

pytestmark = pytest.mark.golden


def _w_at(z: float, w0: float, lam: float) -> float:
    return w0 * math.sqrt(1.0 + (z / rayleigh_range(w0, lam)) ** 2)


# ---------------------------------------------------------------------------------------------------------------
# Golden 8: aperture clipping (centred circular aperture).  Reference: T = 1 - exp(-2 a^2 / w^2), w = w(z) at the stop.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("a_over_w,expect_status", [(3.0, "PASS"), (2.0, "PASS"), (1.5, "WARNING"), (1.0, "OUT_OF_MODEL_SCOPE")])
def test_golden_circular_aperture_clipping(a_over_w, expect_status):
    lam, w0, z = 633.0, 1.0, 40.0
    w = _w_at(z, w0, lam)
    a = a_over_w * w
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), power_W=4e-3),
            B.aperture_stop("Stop", position=(z, 0, 20), axis=(1, 0, 0), radius_mm=a),
        ),
        observations=(B.observation_plane("after", position=(z + 30, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
    )
    res = run(project)
    arr = one_arrival(res, "after")
    t_expected = 1.0 - math.exp(-2.0 * a * a / (w * w))
    assert arr.beam.power_W == pytest.approx(4e-3 * t_expected, rel=1e-9)
    gauss = arr.beam.gaussian
    assert gauss.status.value == expect_status if expect_status != "PASS" else gauss.status.value == "PASS"
    if expect_status == "OUT_OF_MODEL_SCOPE":
        assert not gauss.valid and gauss.reason == "GAUSSIAN_SEVERE_TRUNCATION"
        assert "GAUSSIAN_SEVERE_TRUNCATION" in issue_codes(res)
        assert res.status.value == "OUT_OF_MODEL_SCOPE"  # never a pseudo-exact PASS
    elif expect_status == "WARNING":
        assert gauss.valid and "GAUSSIAN_TRUNCATION" in issue_codes(res)
    else:
        assert gauss.valid and res.status.value == "PASS"


def test_golden_rectangular_aperture_clipping():
    """Centred rectangle, round beam: T = erf(sqrt2 a / w) erf(sqrt2 b / w)."""
    lam, w0, z = 532.0, 1.0, 20.0
    w = _w_at(z, w0, lam)
    ha, hb = 1.8 * w, 2.6 * w
    from os_optics.contracts import (
        ApertureStopInteraction, EngineeringComponentSnapshot, PlaneSurface, RectangularAperture,
    )

    stop = EngineeringComponentSnapshot(
        instance_id="Slit", pose=B.Pose3D.from_axis_direction((z, 0, 20), (1, 0, 0)),
        surfaces=(PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=(0, 0, 1),
                               aperture=RectangularAperture(half_width_mm=ha, half_height_mm=hb), mount_margin_mm=50.0,
                               interaction=ApertureStopInteraction()),),
    )
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3), stop),
        observations=(B.observation_plane("after", position=(z + 5, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
    )
    arr = one_arrival(run(project), "after")
    expected = erf(math.sqrt(2.0) * ha / w) * erf(math.sqrt(2.0) * hb / w)
    assert arr.beam.power_W == pytest.approx(1e-3 * expected, rel=1e-9)


def _reference_fraction(state, aperture: ApertureG, chief_uv, u_axis, v_axis, cos_alpha, **_ignored):
    """Independent high-accuracy reference for beam axes aligned with the plane axes (u, v).

    In plane coordinates the intensity is separable with widths wu = w_x/|bx.u|, wv = w_y/|by.v| (the plane cuts the beam
    obliquely, cos(alpha) cancels the normalisation).  Circle: 1D adaptive quadrature over u with an erf inner integral;
    rectangle: product of erf differences.  (A dense-grid indicator integral was tried first and is only 1e-4 accurate.)"""
    from scipy.integrate import quad

    bxu, byv = float(state.bx @ u_axis), float(state.by @ v_axis)
    assert abs(float(state.bx @ v_axis)) < 1e-12 and abs(float(state.by @ u_axis)) < 1e-12, "reference needs aligned axes"
    wu, wv = state.radius("x") / abs(bxu), state.radius("y") / abs(byv)
    cu, cv = chief_uv
    au, av = aperture.offset_u - cu, aperture.offset_v - cv  # aperture centre relative to the beam centre

    def cdf(x, w):  # cumulative distribution of the normalised intensity exp(-2 x^2/w^2)
        return 0.5 * (1.0 + erf(math.sqrt(2.0) * x / w))

    def one_axis_density(x):
        return math.sqrt(2.0 / math.pi) / wu * math.exp(-2.0 * x * x / (wu * wu))

    if aperture.shape == "rectangle":
        return (cdf(au + aperture.p1, wu) - cdf(au - aperture.p1, wu)) * (cdf(av + aperture.p2, wv) - cdf(av - aperture.p2, wv))

    def disk(radius):
        if radius <= 0:
            return 0.0

        def integrand(x):
            h2 = radius * radius - (x - au) ** 2
            if h2 <= 0:
                return 0.0
            h = math.sqrt(h2)
            return one_axis_density(x) * (cdf(av + h, wv) - cdf(av - h, wv))

        pts = [au - radius, au, au + radius, 0.0]
        val, _err = quad(integrand, au - radius, au + radius, points=sorted(set(pts)), epsabs=1e-13, epsrel=1e-13, limit=400)
        return val

    return disk(aperture.p1) - disk(aperture.obstruction)


def _state(wx=0.8, wy=0.5, lam=633.0, bx=None, by=None):
    bx = np.array([1.0, 0, 0]) if bx is None else bx
    by = np.array([0, 1.0, 0]) if by is None else by
    st = initial_state(wavelength_nm=lam, n=1.0, w0x=wx, w0y=wy, z_waist=0.0, m2x=1.0, m2y=1.0, bx=bx, by=by, cfg=CFG)
    return st


@pytest.mark.parametrize(
    "aperture,chief_uv",
    [
        (ApertureG("circle", 0.9, 0, 0.0, 0.0, 0.0), (0.0, 0.0)),
        (ApertureG("circle", 0.9, 0, 0.1, -0.2, 0.0), (0.15, 0.05)),
        (ApertureG("circle", 0.9, 0, 0.0, 0.0, 0.3), (0.05, 0.0)),  # obstruction
        (ApertureG("circle", 0.6, 0, 1.0, 0.8, 0.0), (0.0, 0.0)),  # beam centre outside the aperture
        (ApertureG("rectangle", 0.7, 0.5, 0.05, 0.02, 0.0), (0.1, -0.1)),
        (ApertureG("rectangle", 2.0, 0.3, 0.0, 0.0, 0.0), (0.0, 0.0)),
    ],
)
def test_aperture_transmission_matches_independent_quadrature(aperture, chief_uv):
    st = _state()
    u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    got = transmission_fraction(st, aperture, chief_uv, u, v, 1.0, CFG)
    ref = _reference_fraction(st, aperture, chief_uv, u, v, 1.0)
    assert got == pytest.approx(ref, abs=1e-8)


def test_aperture_quadrature_is_converged():
    """Doubling every quadrature resolution changes the result by < 1e-10 (numerical accuracy evidence)."""
    st = _state(0.8, 0.5)
    u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    fine = NumericalConfig(aperture_quad_angular=1024, aperture_quad_sector_nodes=96)
    for ap, uv in [(ApertureG("circle", 0.9, 0, 0.1, -0.2, 0.0), (0.15, 0.05)), (ApertureG("circle", 0.6, 0, 1.0, 0.8, 0.0), (0.0, 0.0)),
                   (ApertureG("rectangle", 0.7, 0.5, 0.05, 0.02, 0.0), (0.1, -0.1))]:
        assert transmission_fraction(st, ap, uv, u, v, 1.0, CFG) == pytest.approx(transmission_fraction(st, ap, uv, u, v, 1.0, fine), abs=1e-9)


@pytest.mark.parametrize("radius_mm", [11.43, 12.7, 25.0])
@pytest.mark.parametrize("offset_uv", [(0.0, 0.0), (0.3, -0.2)])
def test_an_unclipped_beam_is_transmitted_exactly_on_every_runtime(radius_mm, offset_uv):
    """A ~1 mm beam in a >10 mm aperture loses ~exp(-200) of its power: the quadrature sum lands a few ulps below 1 on
    some runtimes (SIMD cos/sin/sum) and exactly on 1 on others. A loss below cross_runtime_abs_tol is that noise and is
    not applied, so the transmitted power is exactly the incident power everywhere (the native and browser (Pyodide) runtimes agree bit for bit)."""

    st = _state(1.0013140503265083, 1.0013140503265083, lam=780.0)
    u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    got = transmission_fraction(st, ApertureG("circle", radius_mm, 0, offset_uv[0], offset_uv[1], 0.0), (0.0, 0.0), u, v, 1.0, CFG)
    assert got == 1.0


def test_a_real_loss_above_the_noise_floor_is_still_applied():
    st = _state(1.0, 1.0)
    u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    # loss exp(-2 a^2 / w^2): a = 3.5 w -> 2.2e-11 (kept), a = 4 w -> 1.3e-14 (below the 1e-12 floor: noise)
    kept = transmission_fraction(st, ApertureG("circle", 3.5, 0, 0.0, 0.0, 0.0), (0.0, 0.0), u, v, 1.0, CFG)
    assert 1.0 - kept == pytest.approx(math.exp(-2.0 * 3.5**2), rel=1e-3)
    assert transmission_fraction(st, ApertureG("circle", 4.0, 0, 0.0, 0.0, 0.0), (0.0, 0.0), u, v, 1.0, CFG) == 1.0


# ---------------------------------------------------------------------------------------------------------------
# Golden 9: projected aperture under oblique incidence.
# Reference: the aperture plane tilted by theta from the beam: the round beam footprint on the plane is an ellipse with semi-axes
# w/cos(theta) (tangential) and w (sagittal); the aperture as seen by the beam is a cos(theta) x a ellipse.  Transmission is the
# integral of the intensity over the *plane* aperture (dense-grid reference below); the chief-ray clipping distance uses a cos(theta).
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0, 75.0])
def test_golden_projected_aperture_oblique_incidence(theta_deg):
    th = math.radians(theta_deg)
    lam, w0, z = 633.0, 1.0, 30.0
    w = _w_at(z, w0, lam)
    a = 1.4 * w  # transmission ~ 0.86 on-axis: keeps the model valid only for the *power*; compare against brute force
    d = np.array([math.cos(th), math.sin(th), 0.0])  # beam direction; aperture plane normal = +x
    # brute force in the plane frame: plane normal (1,0,0), in-plane axes u = y-ish, v = z
    plane_u, plane_v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    state = _state(w, w, lam, bx=np.array([-math.sin(th), math.cos(th), 0.0]), by=np.array([0.0, 0.0, 1.0]))
    ref = _reference_fraction(state, ApertureG("circle", a, 0, 0.0, 0.0, 0.0), (0.0, 0.0), plane_u, plane_v, math.cos(th))
    src = (-z * d[0], -z * d[1], 20.0)
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(src[0] + 200.0, src[1], src[2]), direction=tuple(d), power_W=1e-3),
            B.aperture_stop("Stop", position=(200.0, 0.0, 20.0), axis=(1, 0, 0), radius_mm=a),
        ),
        observations=(B.observation_plane("after", position=(230.0, 0, 20), normal=(1, 0, 0), radius_mm=200.0),),
    )
    res = run(project)
    arr = one_arrival(res, "after")
    assert arr.beam.power_W / 1e-3 == pytest.approx(ref, abs=1e-8)
    # obliquity makes the footprint wider in the tangential direction: transmission falls with theta at fixed a
    assert 0.0 < ref <= 1.0


def test_chief_ray_clipping_reports_projected_aperture():
    th = math.radians(60.0)
    a, w0 = 5.0, 0.5
    d = np.array([math.cos(th), math.sin(th), 0.0])
    off = 5.5  # beam centre lands 5.5 mm from the aperture centre (> a): clipped
    src = np.array([-40.0 * d[0], -40.0 * d[1] + off, 20.0])
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=w0, position=tuple(src), direction=tuple(d)),
                    B.aperture_stop("Stop", position=(0.0, 0.0, 20.0), axis=(1, 0, 0), radius_mm=a)),
        observations=(B.observation_plane("after", position=(50, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
    )
    res = run(project)
    clip = next(i for i in res.issues if i.code.value == "CLIPPED")
    ctx = clip.numerical_context
    assert ctx["aoi_deg"] == pytest.approx(60.0, abs=1e-9)
    assert ctx["projected_aperture_tangential_mm"] == pytest.approx(a * math.cos(th), rel=1e-12)
    assert ctx["aperture_radius_mm"] == a and ctx["center_to_edge_mm"] == pytest.approx(a - off, abs=1e-9)
    assert ctx["suggested_min_aperture_radius_mm"] > off
    assert clip.location == "surface:Stop/stop" and clip.suggested_action
    # (the chief ray on the rim is a physical outcome the trace completes: a WARNING; the run is not failed by it)
    assert clip.status.value == "WARNING" and res.status.value != "FAIL"
    # the part of the beam inside the opening goes on: its power is the Gaussian integral over the opening (independent
    # quadrature), carried by the pupil rays that pass, one of them its reference ray from the stop on
    w = _w_at(40.0, w0, 532.0)
    state = _state(w, w, 532.0, bx=np.array([-math.sin(th), math.cos(th), 0.0]), by=np.array([0.0, 0.0, 1.0]))
    ref = _reference_fraction(state, ApertureG("circle", a, 0, 0.0, 0.0, 0.0), (off, 0.0), np.array([0.0, 1.0, 0.0]),
                              np.array([0.0, 0.0, 1.0]), math.cos(th))
    assert 0.1 < ref < 0.2  # (the edge 0.5 footprint radii inside the chief ray: about erfc(sqrt(2) / 2) / 2)
    assert ctx["transmitted_fraction"] == pytest.approx(ref, abs=1e-8)
    assert 0 < ctx["pupil_rays_passing"] < ctx["pupil_rays_live"] and ctx["reference_shift_mm"] > 0.0
    assert [p.termination for p in res.paths] == ["ESCAPED"]  # (it leaves beside the plane behind the stop)
    assert res.power_budget.conserved
    assert res.power_budget.total_sinks["ESCAPED"] == pytest.approx(1e-3 * ref, rel=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Golden 13: detector acceptance and capture.
# Reference: capture margin = R_active - (rho + w_radial); acceptance margin = acceptance - AOI; active-area power fraction by
# independent dense-grid integration; beam radius w(z) closed form.
# ---------------------------------------------------------------------------------------------------------------
def _detector_project(offset_mm=0.0, tilt_deg=0.0, active=3.0, acceptance=10.0):
    th = math.radians(tilt_deg)
    lam, w0, path = 633.0, 0.4, 100.0
    d = (1.0, 0.0, 0.0)
    # detector plane normal tilted by tilt_deg about z; the beam travels along +x
    normal = (-math.cos(th), -math.sin(th), 0.0)
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, offset_mm, 20), direction=d),),
        observations=(B.observation_plane("det", position=(path, 0, 20), normal=normal, radius_mm=active,
                                          acceptance_half_angle_deg=acceptance),),
    ), _w_at(path, w0, lam)


def test_golden_detector_capture_margin_and_fraction():
    project, w = _detector_project(offset_mm=1.2, active=3.0)
    arr = one_arrival(run(project), "det")
    assert arr.capture_margin_mm == pytest.approx(3.0 - (1.2 + w), rel=1e-10)
    assert arr.acceptance_margin_deg == pytest.approx(10.0, abs=1e-9) and arr.within_acceptance
    st = _state(w, w, 633.0)
    ref = _reference_fraction(st, ApertureG("circle", 3.0, 0, 0.0, 0.0, 0.0), (0.0, 1.2), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0)
    # (the plane frame u,v differ from st's frame but the beam is round, so the fraction is identical)
    assert arr.active_area_power_fraction == pytest.approx(ref, abs=1e-8)


def test_detector_beam_partly_outside_is_flagged_with_a_notice_and_the_captured_fraction():
    project, w = _detector_project(offset_mm=2.6, active=3.0)
    res = run(project)
    arr = one_arrival(res, "det")
    assert arr.capture_margin_mm < 0 and arr.active_area_power_fraction < 1.0
    # partial capture is a measurement : a notice that carries the captured fraction, and no status of the run
    (overfill,) = [i for i in res.issues if i.code.value == "INCOMPLETE_ENVELOPE"]
    assert overfill.status.value == "NOT_APPLICABLE" and res.status.value == "PASS"
    assert overfill.numerical_context["active_area_power_fraction"] == arr.active_area_power_fraction


def test_detector_beam_missing_active_area_is_not_an_arrival_and_reports_the_miss():
    project, w = _detector_project(offset_mm=5.0, active=3.0)
    res = run(project)
    # The tiny Gaussian tail is left out as negligible: the stated truncation of a known zero, with its finite bound.
    assert res.status.value == "PASS" and res.observations[0].arrivals == ()
    assert res.observations[0].total_power_W == 0.0 and res.observations[0].power_complete
    assert res.observations[0].quantity_state.value == "KNOWN_ZERO"
    assert 0 < res.observations[0].power_upper_bound_W < 1e-20
    miss = next(i for i in res.issues if i.code.value == "OBSERVATION_NOT_REACHED")
    assert miss.status.value == "NOT_APPLICABLE"
    assert miss.numerical_context["faint_power_upper_bound_W"] == pytest.approx(res.observations[0].power_upper_bound_W, rel=1e-12)
    assert miss.numerical_context["nearest_miss_mm"] == pytest.approx(5.0 - 3.0, abs=1e-9)
    assert miss.numerical_context["beams_crossing_outside_active_area"] == 1


def test_partially_overlapping_beam_still_counts_as_an_arrival():
    """Chief ray just outside the active area but the beam edge overlaps it: an arrival with a negative capture margin."""
    w = _w_at(100.0, 0.4, 633.0)
    project, _ = _detector_project(offset_mm=3.0 + 0.9 * w, active=3.0)
    arr = one_arrival(run(project), "det")
    assert 1e-3 < arr.active_area_power_fraction < 0.5 and arr.capture_margin_mm < 0


@pytest.mark.parametrize("tilt,ok", [(4.0, True), (9.9, True), (10.1, False), (25.0, False)])
def test_detector_acceptance_angle(tilt, ok):
    project, _ = _detector_project(tilt_deg=tilt, acceptance=10.0)
    res = run(project)
    arr = one_arrival(res, "det")
    assert arr.aoi_deg == pytest.approx(tilt, abs=1e-9)
    assert arr.within_acceptance is ok
    assert arr.acceptance_margin_deg == pytest.approx(10.0 - tilt, abs=1e-9)
    assert ("MONITOR_MISSED" in issue_codes(res)) is (not ok)
    # outside the acceptance cone the beam is not detected : the plane receives 0 W from it, stated by a notice
    assert res.observations[0].total_power_W == (pytest.approx(arr.beam.power_W, rel=1e-12) if ok else 0.0)
    assert res.status.value == "PASS" and res.observations[0].status.value == "PASS"
    if not ok:
        (missed,) = [i for i in res.issues if i.code.value == "MONITOR_MISSED"]
        assert missed.status.value == "NOT_APPLICABLE" and missed.numerical_context["aoi_deg"] == pytest.approx(tilt, abs=1e-9)
