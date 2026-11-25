"""Golden cases 12 and 15: waveplate + PBS, Malus, Fresnel / Brewster / TIR / Snell, and 3D Jones basis transport.

Independent references:
* Jones-matrix products written in the test with plain numpy;
* ideal mirror: E_out = -(I - 2 n n^T) E_in (perfect conductor: tangential components flip, normal component preserved);
* Fresnel amplitudes (Born & Wolf) evaluated in the test with their own sin/cos algebra."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import (
    EngineeringComponentSnapshot, JonesVector, PlaneSurface, ProjectSnapshot, RefractInteraction, CircularAperture,
)

from helpers import issue_codes, one_arrival, run

from _frozen_poses import Q_45_NEG, Q_NEG_Y, pin_pose

pytestmark = pytest.mark.golden

INV = 1.0 / math.sqrt(2.0)


def field_of(arrival) -> np.ndarray:
    """3D complex electric field of an arrival (Jones in its transverse basis)."""
    jx = complex(*arrival.beam.polarization.ex)
    jy = complex(*arrival.beam.polarization.ey)
    return jx * np.array(arrival.beam.basis.e1) + jy * np.array(arrival.beam.basis.e2)


# ---------------------------------------------------------------------------------------------------------------
# Golden 10: half-wave plate + PBS.  Beam along +x; local x of the elements = global y (in the PBS plane of incidence = p),
# local y = global z (s).  Linear input at 0 deg (p).  HWP fast axis at alpha rotates the polarization by 2 alpha.
# Reference (Jones algebra): P_transmitted = cos^2(2 alpha), P_reflected = sin^2(2 alpha).
# ---------------------------------------------------------------------------------------------------------------
def _hwp_pbs_project(alpha_deg, *, pbs_kwargs=None, pol_deg=0.0, qwp=False):
    plate = (B.quarter_waveplate if qwp else B.half_waveplate)(
        "WP", position=(20, 0, 20), axis=(1, 0, 0), fast_axis_angle_deg=alpha_deg, design_wavelength_nm=635.0)
    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=635.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=10e-3,
                    polarization=JonesVector.linear(pol_deg)),
            plate,
            pin_pose(B.pbs("PBS", position=(60, 0, 20), normal=(-INV, INV, 0), **(pbs_kwargs or {})), Q_45_NEG),
        ),
        observations=(B.observation_plane("T", position=(120, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      pin_pose(B.observation_plane("R", position=(60, 60, 20), normal=(0, -1, 0), radius_mm=10.0), Q_NEG_Y)),
    )


@pytest.mark.parametrize("alpha", [0.0, 10.0, 22.5, 30.0, 45.0, 67.5, 90.0])
def test_golden_hwp_plus_pbs(alpha):
    res = run(_hwp_pbs_project(alpha))
    p_t = one_arrival(res, "T").beam.power_W if any(a for a in res.observations[0].arrivals) else 0.0
    p_r = one_arrival(res, "R").beam.power_W if any(a for a in res.observations[1].arrivals) else 0.0
    a = math.radians(2.0 * alpha)
    # independent Jones product: HWP(alpha) applied to (1, 0) (fast axis alpha, retardance pi)
    c, s = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    rot = np.array([[c, -s], [s, c]])
    hwp = rot @ np.diag([np.exp(-0.5j * math.pi), np.exp(0.5j * math.pi)]) @ rot.T
    out = hwp @ np.array([1.0, 0.0])
    assert abs(out[0]) ** 2 == pytest.approx(math.cos(a) ** 2, abs=1e-12)  # sanity of the reference itself
    assert p_t == pytest.approx(10e-3 * abs(out[0]) ** 2, abs=1e-14)
    assert p_r == pytest.approx(10e-3 * abs(out[1]) ** 2, abs=1e-14)
    assert p_t + p_r == pytest.approx(10e-3, rel=1e-12)


def test_hwp_pbs_extinction_leakage():
    kw = dict(extinction_reflect_p=0.01, extinction_transmit_s=0.002)
    alpha = 15.0
    res = run(_hwp_pbs_project(alpha, pbs_kwargs=kw))
    a = math.radians(2 * alpha)
    p_t = 10e-3 * (0.99 * math.cos(a) ** 2 + 0.002 * math.sin(a) ** 2)  # T_p cos^2 + T_s sin^2 (orthogonal components add)
    p_r = 10e-3 * (0.998 * math.sin(a) ** 2 + 0.01 * math.cos(a) ** 2)
    assert one_arrival(res, "T").beam.power_W == pytest.approx(p_t, rel=1e-12)
    assert one_arrival(res, "R").beam.power_W == pytest.approx(p_r, rel=1e-12)


def test_qwp_at_45_makes_circular_light_split_evenly_at_pbs():
    res = run(_hwp_pbs_project(45.0, qwp=True))
    assert one_arrival(res, "T").beam.power_W == pytest.approx(5e-3, rel=1e-12)
    assert one_arrival(res, "R").beam.power_W == pytest.approx(5e-3, rel=1e-12)
    # the light between the plate and the PBS is circular: |E1| = |E2|, 90 deg phase
    proj = _hwp_pbs_project(45.0, qwp=True)
    proj = proj.model_copy(update={"observations": (B.observation_plane("mid", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=10.0),)})
    mid = one_arrival(run(proj), "mid")
    jx, jy = complex(*mid.beam.polarization.ex), complex(*mid.beam.polarization.ey)
    assert abs(jx) == pytest.approx(abs(jy), abs=1e-12)
    assert abs(math.sin(np.angle(jy) - np.angle(jx))) == pytest.approx(1.0, abs=1e-12)


def test_qwp_convention_linear_to_right_circular():
    """Input linear at 0 deg, QWP fast axis at +45 deg: output (1, -i)/sqrt2 up to global phase (fast axis leads by pi/2)."""
    proj = _hwp_pbs_project(45.0, qwp=True)
    proj = proj.model_copy(update={"observations": (B.observation_plane("mid", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=10.0),)})
    mid = one_arrival(run(proj), "mid")
    jx, jy = complex(*mid.beam.polarization.ex), complex(*mid.beam.polarization.ey)
    ratio = jy / jx
    # independent Jones: R(45) diag(e^{-i pi/4}, e^{+i pi/4}) R(-45) (1,0)
    c = s = INV
    rot = np.array([[c, -s], [s, c]])
    out = rot @ np.diag([np.exp(-0.25j * math.pi), np.exp(0.25j * math.pi)]) @ rot.T @ np.array([1.0, 0.0])
    assert ratio == pytest.approx(out[1] / out[0], abs=1e-12)
    assert ratio == pytest.approx(-1j, abs=1e-12)


@pytest.mark.parametrize("phi", [0.0, 20.0, 45.0, 70.0, 90.0])
def test_malus_law_two_polarizers(phi):
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=2e-3,
                    polarization=JonesVector.linear(0.0)),
            B.polarizer("P1", position=(10, 0, 20), axis=(1, 0, 0), angle_deg=0.0),
            B.polarizer("P2", position=(20, 0, 20), axis=(1, 0, 0), angle_deg=phi),
        ),
        observations=(B.observation_plane("d", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    arr = res.observations[0].arrivals
    p = arr[0].beam.power_W if arr else 0.0
    assert p == pytest.approx(2e-3 * math.cos(math.radians(phi)) ** 2, abs=1e-15)


def test_polarizer_extinction_ratio():
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                    polarization=JonesVector.linear(90.0)),
            B.polarizer("P", position=(10, 0, 20), axis=(1, 0, 0), angle_deg=0.0, t_par=0.9, t_perp=1e-4),
        ),
        observations=(B.observation_plane("d", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    assert one_arrival(run(proj), "d").beam.power_W == pytest.approx(1e-3 * 1e-4, rel=1e-12)


def test_retarder_wavelength_scaling_and_achromatic():
    """Zero-order plate: delta = delta0 * lambda0/lambda. A nominal HWP at 635 nm used at 532 nm gives delta = pi*635/532."""
    def out_power(wl, achromatic):
        proj = ProjectSnapshot(
            components=(
                B.laser("L", wavelength_nm=wl, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                        polarization=JonesVector.linear(0.0)),
                B.waveplate("W", position=(10, 0, 20), axis=(1, 0, 0), fast_axis_angle_deg=45.0, retardance_rad=math.pi,
                            design_wavelength_nm=635.0, achromatic=achromatic),
                B.polarizer("P", position=(20, 0, 20), axis=(1, 0, 0), angle_deg=90.0),
            ),
            observations=(B.observation_plane("d", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
        )
        arr = run(proj).observations[0].arrivals
        return arr[0].beam.power_W if arr else 0.0

    delta = math.pi * 635.0 / 532.0
    # linear 0 deg through retarder at 45 deg then analyzer at 90 deg: P = P0 * sin^2(delta/2)... (Jones product)
    c = s = INV
    rot = np.array([[c, -s], [s, c]])
    jones = rot @ np.diag([np.exp(-0.5j * delta), np.exp(0.5j * delta)]) @ rot.T @ np.array([1.0, 0.0])
    assert out_power(532.0, False) == pytest.approx(1e-3 * abs(jones[1]) ** 2, rel=1e-12)
    assert out_power(532.0, True) == pytest.approx(1e-3, rel=1e-12)  # achromatic HWP: full rotation to 90 deg


# ---------------------------------------------------------------------------------------------------------------
# Fresnel: plate at oblique incidence. Reference: Born & Wolf amplitude coefficients evaluated in the test.
# ---------------------------------------------------------------------------------------------------------------
def _fresnel_plate_project(aoi_deg, pol_deg, n_name="N-BK7", t=3.0, wl=633.0):
    th = math.radians(aoi_deg)
    d = (math.cos(th), math.sin(th), 0.0)
    plate = B.window("W", thickness_mm=t, clear_radius_mm=25.0, material=n_name, position=(50, 0, 20), axis=(1, 0, 0))
    src = (50 - 40 * d[0], -40 * d[1], 20.0)
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=wl, waist_radius_mm=0.5, position=src, direction=d, polarization=JonesVector.linear(pol_deg)),
                    plate),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=200.0),),
    )


def _bk7_n(wl):
    l2 = (wl * 1e-3) ** 2
    b = (1.03961212, 0.231792344, 1.01046945)
    c = (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _fresnel_T(n1, n2, th1):
    s1 = math.sin(th1)
    th2 = math.asin(n1 * s1 / n2)
    c1, c2 = math.cos(th1), math.cos(th2)
    ts = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    tp = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    f = n2 * c2 / (n1 * c1)
    return f * ts**2, f * tp**2, th2


@pytest.mark.parametrize("aoi", [0.0, 30.0, 56.0, 70.0])
@pytest.mark.parametrize("pol,kind", [(90.0, "s"), (0.0, "p")])
def test_fresnel_transmission_through_plate(aoi, pol, kind):
    n = _bk7_n(633.0)
    res = run(_fresnel_plate_project(aoi, pol))
    th = math.radians(aoi)
    ts1, tp1, th2 = _fresnel_T(1.0, n, th)
    ts2, tp2, _ = _fresnel_T(n, 1.0, th2)
    expected = (ts1 * ts2) if kind == "s" else (tp1 * tp2)
    assert one_arrival(res, "d").beam.power_W == pytest.approx(1e-3 * expected, rel=1e-12)
    if aoi == 0.0:
        r = ((n - 1) / (n + 1)) ** 2
        assert expected == pytest.approx((1 - r) ** 2, rel=1e-12)  # normal incidence closed form


def test_brewster_angle_p_polarization_lossless_through_plate():
    n = _bk7_n(633.0)
    brewster = math.degrees(math.atan(n))
    res = run(_fresnel_plate_project(brewster, 0.0))
    assert one_arrival(res, "d").beam.power_W == pytest.approx(1e-3, rel=1e-12)  # r_p = 0 at both surfaces
    assert run(_fresnel_plate_project(brewster, 90.0)).observations[0].arrivals[0].beam.power_W < 1e-3  # s still reflects


def test_plate_output_field_3d_matches_independent_fresnel_vectors():
    """Complete complex 3D field after the plate (parallel plate: direction restored): E = t_s E_s s_hat + t_p E_p p_hat."""
    aoi, pol = 50.0, 35.0
    n = _bk7_n(633.0)
    th = math.radians(aoi)
    d = np.array([math.cos(th), math.sin(th), 0.0])
    res = run(_fresnel_plate_project(aoi, pol))
    arr = one_arrival(res, "d")
    # input field: local x = y_hat x z... beam d, up = z: local y = z_hat, local x = y x d
    e_loc_y = np.array([0.0, 0.0, 1.0])
    e_loc_x = np.cross(e_loc_y, d)
    e_in = math.cos(math.radians(pol)) * e_loc_x + math.sin(math.radians(pol)) * e_loc_y
    s_hat = np.array([0.0, 0.0, 1.0])  # plane of incidence = xy-plane -> s = z
    p_hat = np.cross(d, s_hat)
    e_s, e_p = e_in @ s_hat, e_in @ p_hat
    th2 = math.asin(math.sin(th) / n)
    c1, c2 = math.cos(th), math.cos(th2)
    def t_coeffs(n1, n2, c1, c2):
        return 2 * n1 * c1 / (n1 * c1 + n2 * c2), 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    ts1, tp1 = t_coeffs(1.0, n, c1, c2)
    ts2, tp2 = t_coeffs(n, 1.0, c2, c1)
    f1, f2 = n * c2 / c1, c1 / (n * c2)  # energy-flux factors n2 cos2/(n1 cos1) for each interface
    tot_s = ts1 * ts2 * math.sqrt(f1 * f2)
    tot_p = tp1 * tp2 * math.sqrt(f1 * f2)
    expected = tot_s * e_s * s_hat + tot_p * e_p * p_hat
    got = field_of(arr)
    np.testing.assert_allclose(got, expected, atol=1e-13)
    np.testing.assert_allclose(arr.beam.direction, d, atol=1e-13)


@pytest.mark.parametrize("theta_deg,n1,n2", [(20.0, 1.0, 1.5), (35.0, 1.5, 1.0), (10.0, 1.0, 2.4)])
def test_snell_refraction_angle(theta_deg, n1, n2):
    """Single interface: refracted direction from an independent Snell evaluation; energy R + T = 1."""
    from os_optics.contracts import ConstantIndexMaterial

    th = math.radians(theta_deg)
    m1 = ConstantIndexMaterial(material_id="m1", valid_wavelength_nm=(300.0, 900.0), refractive_index=n1,
                               absorption_coefficient_per_mm=0.0)
    m2 = ConstantIndexMaterial(material_id="m2", valid_wavelength_nm=(300.0, 900.0), refractive_index=n2,
                               absorption_coefficient_per_mm=0.0)
    iface = EngineeringComponentSnapshot(
        instance_id="I", pose=B.Pose3D.from_axis_direction((0, 0, 20), (1, 0, 0)),
        surfaces=(PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=(0, 0, -1), aperture=CircularAperture(radius_mm=100.0),
                               medium_positive="m1", medium_negative="m2", interaction=RefractInteraction(trace_fresnel_reflection=True)),),
    )
    d = (math.cos(th), math.sin(th), 0.0)
    proj = ProjectSnapshot(
        materials=(m1, m2),
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.2, position=(-30 * d[0], -30 * d[1], 20.0), direction=d, medium="m1"), iface),
        observations=(B.observation_plane("T", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=500.0, medium="m2"),
                      B.observation_plane("R", position=(-60, 0, 20), normal=(-1, 0, 0), radius_mm=500.0, medium="m1")),
    )
    res = run(proj)
    th2 = math.asin(n1 * math.sin(th) / n2)
    t_arr = one_arrival(res, "T")
    np.testing.assert_allclose(t_arr.beam.direction, (math.cos(th2), math.sin(th2), 0.0), atol=1e-13)
    r_arr = one_arrival(res, "R")
    np.testing.assert_allclose(r_arr.beam.direction, (-math.cos(th), math.sin(th), 0.0), atol=1e-13)
    assert t_arr.beam.power_W + r_arr.beam.power_W == pytest.approx(1e-3, rel=1e-12)


def _tir_project(delta_rad):
    """Beam inside glass (n=1.5) hitting a glass/air plane at critical angle + delta."""
    from os_optics.contracts import ConstantIndexMaterial

    n = 1.5
    crit = math.asin(1.0 / n)
    th = crit + delta_rad
    glass = ConstantIndexMaterial(material_id="glass", valid_wavelength_nm=(300.0, 900.0), refractive_index=n,
                                  absorption_coefficient_per_mm=0.0)
    iface = EngineeringComponentSnapshot(
        instance_id="I", pose=B.Pose3D.from_axis_direction((0, 0, 20), (1, 0, 0)),
        surfaces=(PlaneSurface(surface_id="s", origin_mm=(0, 0, 0), normal=(0, 0, -1), aperture=CircularAperture(radius_mm=100.0),
                               medium_positive="glass", medium_negative="air", interaction=RefractInteraction()),),
    )
    d = (math.cos(th), math.sin(th), 0.0)
    return ProjectSnapshot(
        materials=(glass,),
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.2, position=(-30 * d[0], -30 * d[1], 20.0), direction=d, medium="glass"), iface),
        observations=(B.observation_plane("T", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=1e6, medium="air"),
                      B.observation_plane("R", position=(-60, 0, 20), normal=(-1, 0, 0), radius_mm=500.0, medium="glass")),
    ), crit


def test_total_internal_reflection_and_critical_boundary():
    proj, crit = _tir_project(+0.01)  # above the critical angle
    res = run(proj)
    assert res.observations[0].arrivals == ()  # nothing transmitted
    r = one_arrival(res, "R")
    assert r.beam.power_W == pytest.approx(1e-3, rel=1e-12)  # |r| = 1
    th = crit + 0.01
    np.testing.assert_allclose(r.beam.direction, (-math.cos(th), math.sin(th), 0.0), atol=1e-13)
    # just below the critical angle: transmission resumes, grazing refracted ray (tiny cos(theta_t))
    # margin = 1 - (n1 sin(theta)/n2)^2 = 6.7e-4 < tir_warn_margin (1e-3): TIR_BOUNDARY warning, but light transmits
    proj2, _ = _tir_project(-3e-4)
    res2 = run(proj2)
    assert "TIR_BOUNDARY" in issue_codes(res2)
    arr2 = res2.observations[0].arrivals[0]
    # laser polarization 0 deg = in the plane of incidence (p): independent Fresnel power for glass -> air just below critical
    th_c = math.asin(1.0 / 1.5) - 3e-4
    _ts, t_p, _ = _fresnel_T(1.5, 1.0, th_c)
    assert arr2.beam.power_W == pytest.approx(1e-3 * t_p, rel=1e-10)
    assert 0.0 < t_p < 1.0


# ---------------------------------------------------------------------------------------------------------------
# Golden 12/15: 3D Jones basis transport. Ideal mirrors: E_out = -(I - 2 n n^T) E_in per reflection (exact vector reference).
# A chain of mirrors folding the beam through non-coplanar directions must carry an arbitrary elliptical polarization state exactly.
# ---------------------------------------------------------------------------------------------------------------
def _unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


DIRECTIONS = [_unit(v) for v in [(1, 0, 0), (0.2, 1, 0.3), (-0.5, 0.4, 1.0), (0.7, -0.6, -0.4), (0.1, 0.3, -1.0)]]


def _mirror_chain_project(pol: JonesVector, spacing=60.0):
    comps = [B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.4, position=(0, 0, 0), direction=tuple(DIRECTIONS[0]), polarization=pol)]
    pos = np.zeros(3)
    normals = []
    for k in range(1, len(DIRECTIONS)):
        pos = pos + spacing * DIRECTIONS[k - 1]
        comps.append(B.mirror_for_reflection(f"M{k}", position=tuple(pos), d_in=tuple(DIRECTIONS[k - 1]), d_out=tuple(DIRECTIONS[k]),
                                             clear_radius_mm=20.0))
        n = DIRECTIONS[k] - DIRECTIONS[k - 1]
        normals.append(n / np.linalg.norm(n))
    end = pos + spacing * DIRECTIONS[-1]
    obs = B.observation_plane("end", position=tuple(end), normal=tuple(-DIRECTIONS[-1]), radius_mm=30.0)
    return ProjectSnapshot(components=tuple(comps), observations=(obs,)), normals


@pytest.mark.parametrize("pol", [JonesVector.linear(0.0), JonesVector.linear(90.0), JonesVector.linear(33.0), JonesVector.right_circular(),
                                 JonesVector.left_circular(), JonesVector.elliptical(25.0, 20.0)])
def test_golden_jones_basis_transport_through_noncoplanar_mirrors(pol):
    proj, normals = _mirror_chain_project(pol)
    res = run(proj)
    assert res.status.value in ("PASS", "WARNING"), res.issues
    arr = one_arrival(res, "end")
    # reference: start from the source frame's 3D field (local x = up x d, local y = up' ...)
    from os_optics.contracts import Pose3D

    pose = Pose3D.from_axis_direction((0, 0, 0), tuple(DIRECTIONS[0])).to_pose()
    ex, ey = pose.rotation[:, 0], pose.rotation[:, 1]
    e = complex(*pol.ex) * ex + complex(*pol.ey) * ey
    for n in normals:
        e = -(e - 2.0 * (e @ n) * n)
    got = field_of(arr)
    np.testing.assert_allclose(got, e, atol=1e-12)
    # basis remains orthonormal and right-handed, and transverse to the final direction
    b = arr.beam.basis
    e1, e2, d = np.array(b.e1), np.array(b.e2), np.array(b.direction)
    np.testing.assert_allclose(np.cross(e1, e2), d, atol=1e-12)
    np.testing.assert_allclose([e1 @ d, e2 @ d, e1 @ e2, np.linalg.norm(e1) - 1, np.linalg.norm(e2) - 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(d, DIRECTIONS[-1], atol=1e-12)
    assert abs(got @ d) < 1e-12  # the field stays transverse
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-12)  # ideal mirrors: no loss


def test_jones_transport_does_not_rotate_polarization_in_free_propagation_and_lens_deflection():
    """Parallel transport through an off-centre ideal lens: the 3D field is rotated by the minimal rotation d -> d'."""
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.4, position=(0, 8.0, 20), direction=(1, 0, 0), polarization=JonesVector.linear(30.0)),
            B.thin_lens("Lens", focal_length_mm=40.0, clear_radius_mm=20.0, position=(50, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("d", position=(120, 0, 20), normal=(1, 0, 0), radius_mm=100.0),),
    )
    arr = one_arrival(run(proj), "d")
    d_in = np.array([1.0, 0, 0])
    d_out = _unit((1.0, -8.0 / 40.0, 0.0))  # slope after the lens: s' = 0 - h/f with h = +8 (y offset)
    np.testing.assert_allclose(arr.beam.direction, d_out, atol=1e-13)
    # independent Rodrigues rotation taking d_in -> d_out applied to the input field
    e_in = math.cos(math.radians(30)) * np.cross([0, 0, 1.0], d_in) + math.sin(math.radians(30)) * np.array([0, 0, 1.0])
    v = np.cross(d_in, d_out)
    c = d_in @ d_out
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx / (1 + c)
    np.testing.assert_allclose(field_of(arr), rot @ e_in, atol=1e-13)


def test_retroreflection_basis_flip_is_deterministic_and_orthonormal():
    """Normal-incidence mirror sends the beam back on itself (antiparallel transport branch)."""
    proj = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.4, position=(0, 0, 20), direction=(1, 0, 0), polarization=JonesVector.linear(40.0)),
            B.mirror("M", position=(50, 0, 20), normal=(-1, 0, 0), clear_radius_mm=10.0),
        ),
        observations=(B.observation_plane("back", position=(-5, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    arr = one_arrival(run(proj), "back")
    np.testing.assert_allclose(arr.beam.direction, (-1, 0, 0), atol=1e-15)
    e_in = math.cos(math.radians(40)) * np.cross([0, 0, 1.0], [1.0, 0, 0]) + math.sin(math.radians(40)) * np.array([0, 0, 1.0])
    np.testing.assert_allclose(field_of(arr), -e_in, atol=1e-13)  # normal-incidence mirror: E -> -E
