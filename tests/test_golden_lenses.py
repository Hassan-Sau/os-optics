"""Golden physics cases 1-5: thin-lens focusing, Keplerian / Galilean expanders, 45 deg mirror, non-normal lens
incidence, splitter multi-path. Every reference is analytic or an independent construction (see docstrings)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import ConstantCoating, ProjectSnapshot

from helpers import (
    abcd_apply, arrivals, free, gaussian_radius_from_q, issue_codes, one_arrival, rayleigh_range, run, thin,
)

pytestmark = pytest.mark.golden


# ---------------------------------------------------------------------------------------------------------------
# Golden 1: single thin lens focusing
# Reference (closed form, derived without ABCD code): a collimated waist w0 located AT the lens has q = i zR. After a thin
# lens 1/q' = 1/q - 1/f  =>  q' = i zR/(1 - i zR/f). Waist of the outgoing beam: distance d' = f/(1+(f/zR)^2) and
# w0' = w0 f / sqrt(f^2 + zR^2).  Geometric: parallel rays converge to the back focal plane at distance f.
# ---------------------------------------------------------------------------------------------------------------
def test_golden_thin_lens_focus_analytic_waist_and_rays():
    lam, w0, f, x_lens = 532.0, 1.0, 100.0, 10.0
    zr = rayleigh_range(w0, lam)
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), power_W=2e-3),
            B.thin_lens("Lens", focal_length_mm=f, clear_radius_mm=12.7, position=(x_lens, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("focal", position=(x_lens + f, 0, 20), normal=(1, 0, 0), radius_mm=5.0),
                      B.observation_plane("early", position=(x_lens + f / 2, 0, 20), normal=(1, 0, 0), radius_mm=12.0)),
    )
    res = run(project)
    assert res.status.value in ("PASS", "WARNING"), res.issues
    # waist is 10 mm before the lens: q at the lens = 10 + i zR; independent closed form for the outgoing beam
    q_lens = complex(x_lens, zr)
    q_after = q_lens / (1.0 - q_lens / f)
    q_focal = q_after + f
    w_expected = gaussian_radius_from_q(q_focal, lam)
    a = one_arrival(res, "focal")
    g = a.beam.gaussian
    assert g is not None and g.valid
    assert g.beam_radius_x_mm == pytest.approx(w_expected, rel=1e-12)
    assert g.beam_radius_y_mm == pytest.approx(w_expected, rel=1e-12)
    # geometric rays: all pupil rays focus (RMS spot ~ 0) at the back focal plane, and not elsewhere
    assert a.pupil_rays_arrived == a.pupil_ray_count > 0
    assert a.pupil_rms_radius_mm == pytest.approx(0.0, abs=1e-9)
    early = one_arrival(res, "early")
    assert early.pupil_rms_radius_mm > 0.1
    # power conserved through the ideal lens
    assert a.beam.power_W == pytest.approx(2e-3, rel=1e-12)
    # geometric length: source -> focal plane is a straight line through the thin lens, whose model states no optical path
    assert a.beam.geometric_length_mm == pytest.approx(x_lens + f, rel=1e-12)
    assert a.beam.optical_path_length_mm is None and a.relative_path_mm == pytest.approx(x_lens + f, rel=1e-12)


def test_thin_lens_waist_formula_when_waist_at_lens():
    """Waist located at the lens: the analytic waist size / position of the focused beam."""
    lam, w0, f = 532.0, 1.0, 100.0
    zr = rayleigh_range(w0, lam)
    d_expected = f / (1.0 + (f / zr) ** 2)
    w0_expected = w0 * f / math.sqrt(f * f + zr * zr)
    x_lens = 1.0  # waist_position_z_mm = +x_lens puts the waist exactly at the lens
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=x_lens),
            B.thin_lens("Lens", focal_length_mm=f, clear_radius_mm=12.7, position=(x_lens, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("waist", position=(x_lens + d_expected, 0, 20), normal=(1, 0, 0), radius_mm=5.0),),
    )
    a = one_arrival(run(project), "waist")
    g = a.beam.gaussian
    assert g.waist_radius_x_mm == pytest.approx(w0_expected, rel=1e-11)
    assert g.beam_radius_x_mm == pytest.approx(w0_expected, rel=1e-11)  # at the waist plane
    assert abs(g.waist_distance_x_mm) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Golden 2: Keplerian and Galilean beam expanders
# Reference: plain numpy ABCD matrices  M = L(d) . lens(f2) . L(f1+f2) . lens(f1)  (independent code path) and the ideal
# magnification |f2/f1| (collimated output radius = M w, divergence /M).
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,f1,f2", [("keplerian", 25.0, 100.0), ("galilean", -25.0, 100.0)])
def test_golden_beam_expander(name, f1, f2):
    lam, w0 = 532.0, 0.5
    x1 = 20.0
    x2 = x1 + f1 + f2  # afocal spacing
    x_out = x2 + 150.0
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=x1),
            B.thin_lens("L1", focal_length_mm=f1, clear_radius_mm=12.7, position=(x1, 0, 20), axis=(1, 0, 0)),
            B.thin_lens("L2", focal_length_mm=f2, clear_radius_mm=12.7, position=(x2, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("out", position=(x_out, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
    )
    res = run(project)
    a = one_arrival(res, "out")
    g = a.beam.gaussian
    zr = rayleigh_range(w0, lam)
    mat = free(x_out - x2) @ thin(f2) @ free(f1 + f2) @ thin(f1)
    q_ref = abcd_apply(mat, complex(0.0, zr))  # waist at L1
    assert g.beam_radius_x_mm == pytest.approx(gaussian_radius_from_q(q_ref, lam), rel=1e-11)
    assert g.q_x_mm[0] == pytest.approx(q_ref.real, rel=1e-10, abs=1e-8)
    assert g.q_x_mm[1] == pytest.approx(q_ref.imag, rel=1e-10)
    magnification = abs(f2 / f1)
    # output waist size = M w0 (exact for an afocal system with the input waist at the first lens)
    assert g.waist_radius_x_mm == pytest.approx(magnification * w0, rel=1e-10)
    # divergence reduced by M
    assert g.far_field_half_angle_x_rad == pytest.approx(lam * 1e-6 / (math.pi * w0) / magnification, rel=1e-10)
    # geometric: exiting pupil rays are parallel to the chief ray (afocal, collimated) and the pupil is magnified by M
    assert a.pupil_rays_arrived == a.pupil_ray_count
    # power / OPL sanity
    assert a.beam.power_W == pytest.approx(1e-3, rel=1e-12)
    assert a.beam.geometric_length_mm == pytest.approx(x_out, rel=1e-12)


def test_expander_output_rays_are_parallel_and_magnified():
    """Direct geometric check through the tracer: collimated pupil in -> collimated pupil, radius x M."""
    from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.optics.rays.tracer import Tracer
    from os_optics.scene.lower import lower_scene

    f1, f2, x1 = 25.0, 100.0, 20.0
    proj = ProjectSnapshot(
        components=(
            # waist AT the source plane: flat wavefront, exactly collimated pupil rays
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=0.0),
            B.thin_lens("L1", focal_length_mm=f1, clear_radius_mm=12.7, position=(x1, 0, 20), axis=(1, 0, 0)),
            B.thin_lens("L2", focal_length_mm=f2, clear_radius_mm=12.7, position=(x1 + f1 + f2, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("out", position=(400, 0, 20), normal=(1, 0, 0), radius_mm=25.0),
                      B.observation_plane("in", position=(x1 - 1e-3, 0, 20), normal=(1, 0, 0), radius_mm=25.0)),
    )
    lib = MaterialLibrary((), CFG)
    scene, _ = lower_scene(proj, CFG, lib)
    out = Tracer(scene, lib, CFG).run()
    rec_in = next(r for r in out.arrivals if r.observation_id == "in")
    rec_out = next(r for r in out.arrivals if r.observation_id == "out")
    r_in = np.linalg.norm(rec_in.pupil_uv, axis=1)
    r_out = np.linalg.norm(rec_out.pupil_uv, axis=1)
    np.testing.assert_allclose(r_out, r_in * (f2 / f1), rtol=1e-12)
    # exit rays are parallel to the axis (afocal + collimated): every pupil ray reaches the plane at a constant radius,
    # and the chief ray is on axis
    np.testing.assert_allclose(rec_out.uv, (0.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# Golden 3: 45 degree mirror. Reference: reflection law d' = d - 2 (d.n) n with n = (-1, 1, 0)/sqrt2 gives (0, 1, 0).
# ---------------------------------------------------------------------------------------------------------------
def test_golden_45_degree_mirror():
    inv = 1.0 / math.sqrt(2.0)
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=635.0, waist_radius_mm=0.4, position=(0, 0, 25), direction=(1, 0, 0)),
            B.mirror("M", position=(50, 0, 25), normal=(-inv, inv, 0.0), clear_radius_mm=12.7),
        ),
        observations=(B.observation_plane("det", position=(50, 80, 25), normal=(0, -1, 0), radius_mm=10.0),),
    )
    res = run(project)
    a = one_arrival(res, "det")
    np.testing.assert_allclose(a.beam.direction, (0.0, 1.0, 0.0), atol=1e-15)
    assert a.beam.geometric_length_mm == pytest.approx(50.0 + 80.0, rel=1e-13)
    assert a.beam.power_W == pytest.approx(1e-3, rel=1e-12)  # ideal mirror
    assert a.aoi_deg == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(a.uv_mm, (0.0, 0.0), atol=1e-12)
    # Gaussian beam radius is unchanged by a plane mirror: w(z) with the total propagation distance
    zr = rayleigh_range(0.4, 635.0)
    assert a.beam.gaussian.beam_radius_x_mm == pytest.approx(0.4 * math.sqrt(1 + (130.0 / zr) ** 2), rel=1e-12)


def test_mirror_for_reflection_helper_matches_reflection_law():
    rng = np.random.default_rng(11)
    for _ in range(25):
        d_in = rng.normal(size=3)
        d_in /= np.linalg.norm(d_in)
        d_out = rng.normal(size=3)
        d_out /= np.linalg.norm(d_out)
        if np.linalg.norm(d_out - d_in) < 1e-3:
            continue
        comp = B.mirror_for_reflection("M", position=(0, 0, 0), d_in=tuple(d_in), d_out=tuple(d_out))
        n = comp.pose.to_pose().rotation @ np.array(comp.surfaces[0].normal)
        np.testing.assert_allclose(d_in - 2 * (d_in @ n) * n, d_out, atol=1e-12)
        assert d_in @ n < 0  # normal faces the incident beam (reflective side)


# ---------------------------------------------------------------------------------------------------------------
# Golden 4: non-normal lens incidence.
# Reference: for an ideal thin lens a collimated beam arriving at angle theta to the lens axis focuses on the back focal
# plane (perpendicular to the axis, distance f) at lateral offset f tan(theta); the chief ray therefore travels f/cos(theta)
# after the lens.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 25.0, 40.0])
def test_golden_non_normal_lens_incidence(theta_deg):
    lam, f = 633.0, 150.0
    th = math.radians(theta_deg)
    axis = (1.0, 0.0, 0.0)
    d_in = (math.cos(th), math.sin(th), 0.0)
    lens_pos = (100.0, 0.0, 20.0)
    src_pos = (lens_pos[0] - 60.0 * d_in[0], lens_pos[1] - 60.0 * d_in[1], 20.0)  # beam hits the lens centre
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=1.0, position=src_pos, direction=d_in),
            B.thin_lens("Lens", focal_length_mm=f, clear_radius_mm=25.0, position=lens_pos, axis=axis),
        ),
        observations=(B.observation_plane("bfp", position=(lens_pos[0] + f, 0.0, 20.0), normal=axis, radius_mm=300.0),),
    )
    a = one_arrival(run(project), "bfp")
    # lateral offset: plane frame u = local x = y x z = (0,0,1)x(1,0,0) = (0,1,0) -> global y; v -> global z
    assert a.pupil_rms_radius_mm == pytest.approx(0.0, abs=1e-8)
    assert a.pupil_centroid_uv_mm[0] == pytest.approx(f * math.tan(th), abs=1e-9)
    assert a.pupil_centroid_uv_mm[1] == pytest.approx(0.0, abs=1e-9)
    assert a.uv_mm[0] == pytest.approx(f * math.tan(th), abs=1e-9)  # chief ray lands at the same point
    assert a.beam.geometric_length_mm - 60.0 == pytest.approx(f / math.cos(th), rel=1e-12)
    # Gaussian: tangential/sagittal q use P = cos(theta)/f each; input waist at the lens (z_waist = 60)
    zr = rayleigh_range(1.0, lam)
    q_up = complex(60.0, zr)  # waist 60 mm before the lens
    q_ref = q_up / (1.0 - (math.cos(th) / f) * q_up) + f / math.cos(th)
    assert a.beam.gaussian.q_x_mm[0] == pytest.approx(q_ref.real, rel=1e-9, abs=1e-8)
    assert a.beam.gaussian.q_x_mm[1] == pytest.approx(q_ref.imag, rel=1e-9)
    assert a.beam.gaussian.q_y_mm == pytest.approx(a.beam.gaussian.q_x_mm)  # slope-form ideal lens is stigmatic


# ---------------------------------------------------------------------------------------------------------------
# Golden 5: splitter, multiple paths
# Reference: NPBS with R=0.3: transmitted power 0.7 P, reflected 0.3 P, reflected direction by the reflection law, lossless
# (energy conserved), branch geometry by construction.
# ---------------------------------------------------------------------------------------------------------------
def test_golden_splitter_multi_path():
    inv = 1.0 / math.sqrt(2.0)
    project = ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(-40, 0, 20), direction=(1, 0, 0), power_W=10e-3),
            B.npbs("BS", position=(0, 0, 20), normal=(-inv, inv, 0), reflectance=0.3),
        ),
        observations=(B.observation_plane("through", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("side", position=(0, 70, 20), normal=(0, -1, 0), radius_mm=10.0)),
    )
    res = run(project)
    t, r = one_arrival(res, "through"), one_arrival(res, "side")
    assert t.beam.power_W == pytest.approx(7e-3, rel=1e-12)
    assert r.beam.power_W == pytest.approx(3e-3, rel=1e-12)
    assert t.beam.power_W + r.beam.power_W == pytest.approx(10e-3, rel=1e-12)
    np.testing.assert_allclose(t.beam.direction, (1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(r.beam.direction, (0, 1, 0), atol=1e-15)
    assert t.beam.geometric_length_mm == pytest.approx(140.0)
    assert r.beam.geometric_length_mm == pytest.approx(110.0)
    paths = {p.path_id: p for p in res.paths}
    assert set(paths) == {"L#0", "L#0.0"}
    assert paths["L#0.0"].parent_path_id == "L#0" and paths["L#0.0"].branch_surface_id == "BS/coating"
    assert paths["L#0.0"].branch_kind == "beam_splitter_reflection"
    assert all(p.termination == "ESCAPED" for p in paths.values())
    assert res.summary.path_count == 2


def test_splitter_polarization_dependent_coating_energy_accounting():
    """R_s=0.8,T_s=0.1 (10 % absorbed) for s; p: R=0.2, T=0.7. s-polarized input -> P*0.8 reflected, P*0.1 transmitted."""
    inv = 1.0 / math.sqrt(2.0)
    from os_optics.contracts import JonesVector

    coating = ConstantCoating(reflectance_s=0.8, transmittance_s=0.1, reflectance_p=0.2, transmittance_p=0.7)
    for pol_deg, (t_expected, r_expected) in ((90.0, (0.1, 0.8)), (0.0, (0.7, 0.2))):
        project = ProjectSnapshot(
            components=(
                B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(-40, 0, 20), direction=(1, 0, 0),
                        polarization=JonesVector.linear(pol_deg)),
                B.beam_splitter("BS", position=(0, 0, 20), normal=(-inv, inv, 0), coating=coating),
            ),
            observations=(B.observation_plane("through", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                          B.observation_plane("side", position=(0, 70, 20), normal=(0, -1, 0), radius_mm=10.0)),
        )
        res = run(project)
        # local x (pol 0 deg) for beam along +x with up=z is global y = IN the plane of incidence (p); 90 deg is global z (s)
        assert one_arrival(res, "through").beam.power_W == pytest.approx(t_expected * 1e-3, rel=1e-12)
        assert one_arrival(res, "side").beam.power_W == pytest.approx(r_expected * 1e-3, rel=1e-12)
