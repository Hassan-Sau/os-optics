"""Golden cases 6, 7, 11: Mach-Zehnder OPL/OPD + output-port interference, 405 nm / 500 nm two-beam interference,
multiple independent lasers. References are closed forms or independent constructions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from os_optics import builders as B
from os_optics.contracts import ConstantIndexMaterial, JonesVector, ProjectSnapshot, Target

from helpers import NM, arrivals, issue_codes, one_arrival, rayleigh_range, run, stated_splitter

from _frozen_poses import Q_45_NEG, Q_45_POS, Q_POS_Y, pin_pose

pytestmark = pytest.mark.golden

_BK7 = dict(b=(1.03961212, 0.231792344, 1.01046945), c=(0.00600069867, 0.0200179144, 103.560653))


def bk7_index(wavelength_nm: float) -> float:
    """Test-local Sellmeier evaluation (catalogue coefficients), independent of the engine's material code."""
    l2 = (wavelength_nm * 1e-3) ** 2
    return math.sqrt(1.0 + sum(b * l2 / (l2 - c) for b, c in zip(_BK7["b"], _BK7["c"])))


# ---------------------------------------------------------------------------------------------------------------
# Golden 6: Mach-Zehnder style OPL comparison and output-port interference.
# Geometry (all in the plane z = 20): laser -> BS1(0,0) ; arm A: mirror (L,0) -> BS2(L,L) ; arm B: mirror (0,L) -> BS2,
# a BK7 window of thickness t in arm B.  s-polarized light (E || z) so the ideal mirrors give -1 in both arms.
# Reference (independent): OPL_A = 50 + 2L, OPL_B = 50 + 2L + (n - 1) t ; unitary BS  t = sqrt(T), r = i sqrt(R):
#   port +x : E = -t r (e^{i phiA} + a_B e^{i phiB})     port +y : E = -(t t e^{i phiA} + r r a_B e^{i phiB})
# with a_B = (1-R_F) the two-surface window amplitude factor (R_F = ((n-1)/(n+1))^2 per surface, no ghosts).
# ---------------------------------------------------------------------------------------------------------------
def _mzi_project(window_thickness_mm: float, wavelength_nm: float = 532.0, L: float = 100.0):
    inv = 1.0 / math.sqrt(2.0)
    comps = [
        B.laser("L0", wavelength_nm=wavelength_nm, waist_radius_mm=0.8, position=(-50, 0, 20), direction=(1, 0, 0),
                power_W=1e-3, coherence_length_mm=1e4, polarization=JonesVector.linear(90.0)),
        pin_pose(stated_splitter(B.npbs("BS1", position=(0, 0, 20), normal=(-inv, inv, 0), reflectance=0.5)), Q_45_NEG),
        pin_pose(B.mirror("MA", position=(L, 0, 20), normal=(-inv, inv, 0), clear_radius_mm=12.7), Q_45_NEG),
        pin_pose(B.mirror("MB", position=(0, L, 20), normal=(inv, -inv, 0), clear_radius_mm=12.7), Q_45_POS),
        pin_pose(stated_splitter(B.npbs("BS2", position=(L, L, 20), normal=(-inv, inv, 0), reflectance=0.5)), Q_45_NEG),
    ]
    if window_thickness_mm > 0:
        comps.append(B.window("WB", thickness_mm=window_thickness_mm, clear_radius_mm=12.7, material="N-BK7",
                              position=(L / 2, L, 20), axis=(1, 0, 0)))
    return ProjectSnapshot(
        components=tuple(comps),
        observations=(B.observation_plane("port_x", position=(L + 60, L, 20), normal=(1, 0, 0), radius_mm=15.0),
                      pin_pose(B.observation_plane("port_y", position=(L, L + 60, 20), normal=(0, 1, 0), radius_mm=15.0), Q_POS_Y)),
    )


def test_golden_mzi_opl_and_geometry():
    L, t, lam = 100.0, 2.0, 532.0
    res = run(_mzi_project(t, lam, L))
    n = bk7_index(lam)
    opl_a = 50 + 2 * L + 60
    opl_b = 50 + 2 * L + 60 + (n - 1.0) * t
    x = {a.beam.path_id: a for a in arrivals(res, "port_x")}
    y = {a.beam.path_id: a for a in arrivals(res, "port_y")}
    assert len(x) == 2 and len(y) == 2  # four leaf paths, two per output port
    # path structure: A = through BS1 (path L0#0 continues), B = reflected at BS1 (child .0)
    # port_x: A reflected at BS2 (child L0#0.0), B transmitted at BS2 (child L0#0.0's parent chain -> L0#0.0)
    all_opl = sorted(a.beam.optical_path_length_mm for a in list(x.values()) + list(y.values()))
    assert all_opl[0] == pytest.approx(opl_a, rel=1e-12)
    assert all_opl[1] == pytest.approx(opl_a, rel=1e-12)
    assert all_opl[2] == pytest.approx(opl_b, rel=1e-12)
    assert all_opl[3] == pytest.approx(opl_b, rel=1e-12)
    # geometric length is unaffected by the glass
    lens = sorted(a.beam.geometric_length_mm for a in list(x.values()) + list(y.values()))
    assert lens == pytest.approx([opl_a] * 2 + [opl_a] * 2, rel=1e-12)
    # OPD reported by the interference analysis = (n-1) t (sign by pair order)
    for oid in ("port_x", "port_y"):
        pair = next(i for i in res.interference if i.observation_id == oid)
        assert abs(pair.optical_path_difference_mm) == pytest.approx((n - 1.0) * t, rel=1e-9)


@pytest.mark.parametrize("thickness", [0.0, 0.2, 0.4123, 0.75, 1.0])
def test_golden_mzi_output_port_power_vs_opd(thickness):
    lam, L = 532.0, 100.0
    res = run(_mzi_project(thickness, lam, L))
    n = bk7_index(lam)
    k0 = 2.0 * math.pi / (lam * NM)
    r_f = ((n - 1.0) / (n + 1.0)) ** 2
    a_b = (1.0 - r_f) ** 2 if thickness > 0 else 1.0  # power factor of the two-surface window (no ghosts); amplitude = sqrt(power)
    amp_b = math.sqrt(a_b)
    t_amp, r_amp = math.sqrt(0.5), 1j * math.sqrt(0.5)
    phi_a = 0.0
    phi_b = k0 * (n - 1.0) * thickness if thickness > 0 else 0.0
    ea, eb = np.exp(1j * phi_a), amp_b * np.exp(1j * phi_b)
    p0 = 1e-3
    p_x = p0 * abs(t_amp * r_amp * (ea + eb)) ** 2  # mirrors: (-1) each arm -> common factor
    p_y = p0 * abs(t_amp * t_amp * ea + r_amp * r_amp * eb) ** 2
    px_pair = next(i for i in res.interference if i.observation_id == "port_x")
    py_pair = next(i for i in res.interference if i.observation_id == "port_y")
    # abs = 2e-12 W (2e-9 of the input power): the two arms' Gaussian envelopes differ at the 1e-7 level after the glass
    # (reduced propagation t/n), a second-order effect the perfect-overlap reference formula ignores.
    assert px_pair.combined_power_collinear_W == pytest.approx(p_x, rel=1e-9, abs=2e-12)
    assert py_pair.combined_power_collinear_W == pytest.approx(p_y, rel=1e-9, abs=2e-12)
    # energy: incoherent sum of the two ports equals the power leaving through the window losses only
    total_incoherent = px_pair.incoherent_power_sum_W + py_pair.incoherent_power_sum_W
    assert total_incoherent == pytest.approx(p0 * 0.25 * (2.0 + 2.0 * a_b), rel=1e-9)


# ---------------------------------------------------------------------------------------------------------------
# Golden 7: 405 nm two-beam interference with 500 nm period.
# Reference: theta = arcsin(lambda / (2 Lambda)) = 23.8911 deg (independent closed form); beams placed symmetrically about
# the substrate normal by an independent geometric construction (mirrors positioned by the test); expected outputs:
# crossing angle 2 theta, in-plane period lambda / (2 sin theta) = 500 nm, OPD 0 (equal-arm solve by brentq), s-polarization
# overlap 1, visibility 1, coherence margin 1.
# ---------------------------------------------------------------------------------------------------------------
LAM_405 = 405.0
PERIOD = 500.0
THETA = math.asin(LAM_405 / (2.0 * PERIOD))


def _rot(a: float, v: np.ndarray) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])


def _two_beam_geometry(z=20.0, opd_mm=0.0, arm_ratio_r=0.5):
    beta = math.radians(45.0)
    b = np.array([math.cos(beta), math.sin(beta), 0.0])
    d1, d2 = _rot(THETA, b), _rot(-THETA, b)
    ma = np.array([100.0, 0.0, z])

    def path_diff(s1: float) -> float:
        s_pt = ma + s1 * d1
        s2 = s_pt[0] / d2[0]
        mb = np.array([0.0, s_pt[1] - s2 * d2[1], z])
        return (50 + 100 + s1) - (50 + np.linalg.norm(mb - np.array([0, 0, z])) + s2) - opd_mm

    s1 = brentq(path_diff, 50, 600)
    s_pt = ma + s1 * d1
    s2 = s_pt[0] / d2[0]
    mb = np.array([0.0, s_pt[1] - s2 * d2[1], z])
    return d1, d2, ma, mb, s_pt


def _interference_project(*, pol_deg=90.0, opd_mm=0.0, reflectance=0.5, coherence_length_mm=500.0, power_W=1e-3,
                          waist_mm=1.0, targets=(), extra=()):
    d1, d2, ma, mb, s_pt = _two_beam_geometry(opd_mm=opd_mm)
    n_sub = -(d1 + d2) / np.linalg.norm(d1 + d2)
    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=LAM_405, waist_radius_mm=waist_mm, position=(-50, 0, 20), direction=(1, 0, 0),
                    power_W=power_W, coherence_length_mm=coherence_length_mm, polarization=JonesVector.linear(pol_deg)),
            pin_pose(stated_splitter(B.npbs("BS", position=(0, 0, 20), normal=(-1, 1, 0), reflectance=reflectance)), Q_45_NEG),
            B.mirror_for_reflection("MA", position=tuple(ma), d_in=(1, 0, 0), d_out=tuple(d1)),
            B.mirror_for_reflection("MB", position=tuple(mb), d_in=(0, 1, 0), d_out=tuple(d2)),
            *extra,
        ),
        observations=(B.observation_plane("sub", position=tuple(s_pt), normal=tuple(n_sub), radius_mm=10.0),),
        targets=targets,
    ), d1, d2, s_pt


def test_golden_405nm_500nm_period_interference():
    targets = (
        Target(target_id="period", metric="fringe_period_nm", observation_id="sub", target_value=500.0, tolerance_abs=1e-6, unit="nm"),
        Target(target_id="half", metric="half_angle_deg", observation_id="sub", target_value=math.degrees(THETA), tolerance_abs=1e-9),
        Target(target_id="opd", metric="optical_path_difference_mm", observation_id="sub", target_value=0.0, tolerance_abs=1e-9),
        Target(target_id="vis", metric="fringe_visibility", observation_id="sub", minimum=0.999),
        Target(target_id="coh", metric="coherence_margin", observation_id="sub", minimum=0.9),
    )
    proj, d1, d2, s_pt = _interference_project(targets=targets)
    res = run(proj)
    assert res.status.value == "PASS", [(t.target_id, t.status.value, t.message) for t in res.targets]
    pair = res.interference[0]
    assert pair.status.value == "PASS"
    assert pair.beam_crossing_angle_deg == pytest.approx(2.0 * math.degrees(THETA), abs=1e-10)
    assert pair.fringe_period_nm == pytest.approx(500.0, abs=1e-8)
    assert pair.fringe_spacing_3d_nm == pytest.approx(500.0, abs=1e-8)  # symmetric about the normal: same as in-plane
    assert pair.polarization_overlap == pytest.approx(1.0, abs=1e-12)
    assert pair.spatial_overlap_ratio == pytest.approx(1.0, abs=1e-9)
    assert pair.fringe_visibility == pytest.approx(1.0, abs=1e-9)
    assert pair.intensity_balance == pytest.approx(1.0, abs=1e-9)
    assert pair.optical_path_length_mm_a == pytest.approx(pair.optical_path_length_mm_b, abs=1e-9)
    # irradiance reference: both beams at the centre; transverse-flux intensity 2P/(pi w^2) each with w(z) at the path length z
    z = pair.optical_path_length_mm_a - 50.0  # geometric path from the source origin: laser sits at x=-50 => path incl. first 50
    z = pair.optical_path_length_mm_a
    w = 1.0 * math.sqrt(1.0 + (z / rayleigh_range(1.0, LAM_405)) ** 2)
    each = 2.0 * 0.5e-3 / (math.pi * w * w)
    assert pair.mean_irradiance_W_per_mm2 == pytest.approx(2.0 * each, rel=1e-9)
    assert pair.peak_irradiance_W_per_mm2 == pytest.approx(4.0 * each, rel=1e-9)


def test_405_500_fringe_period_from_independent_wave_vector_construction():
    """Fringe period from the difference of the two 3D wave vectors built in the test (not from the engine's formula path)."""
    proj, d1, d2, s_pt = _interference_project()
    res = run(proj)
    k = 2.0 * math.pi / (LAM_405 * NM)
    nu = -(d1 + d2) / np.linalg.norm(d1 + d2)
    dk = k * (d1 - d2)
    dk_par = dk - (dk @ nu) * nu
    assert res.interference[0].fringe_period_nm == pytest.approx(2.0 * math.pi / np.linalg.norm(dk_par) / NM, abs=1e-7)
    # fringe direction is along dk_par (unit), lying in the plane
    fd = np.array(res.interference[0].fringe_direction)
    np.testing.assert_allclose(fd, dk_par / np.linalg.norm(dk_par), atol=1e-12)
    assert abs(fd @ nu) < 1e-12


def test_p_polarization_overlap_is_cos_crossing_angle():
    """E in the plane of the beams (p-like): overlap = cos(2 theta); visibility scales with it."""
    proj, *_ = _interference_project(pol_deg=0.0)
    pair = run(proj).interference[0]
    assert pair.polarization_overlap == pytest.approx(math.cos(2.0 * THETA), abs=1e-12)
    assert pair.fringe_visibility == pytest.approx(math.cos(2.0 * THETA), abs=1e-9)


def test_unequal_arms_visibility_matches_two_beam_formula():
    r = 0.3
    proj, *_ = _interference_project(reflectance=r)
    pair = run(proj).interference[0]
    # transmitted arm gets T = 0.7, reflected arm R = 0.3 (mirrors ideal): V = 2 sqrt(P1 P2)/(P1+P2)
    assert pair.fringe_visibility == pytest.approx(2.0 * math.sqrt(0.7 * 0.3), abs=1e-9)
    assert pair.intensity_balance == pytest.approx(0.3 / 0.7, abs=1e-9)


@pytest.mark.parametrize("opd", [0.0, 3.0, 10.0, 20.0])
def test_visibility_decays_with_opd_over_coherence_length(opd):
    lc = 10.0
    proj, *_ = _interference_project(opd_mm=opd, coherence_length_mm=lc)
    pair = run(proj).interference[0]
    assert pair.optical_path_difference_mm == pytest.approx(opd, abs=1e-9)
    assert pair.coherence_factor == pytest.approx(math.exp(-((opd / lc) ** 2)), rel=1e-9)
    assert pair.coherence_margin == pytest.approx(1.0 - opd / lc, rel=1e-9, abs=1e-12)
    if opd < 20.0:
        assert pair.fringe_visibility == pytest.approx(math.exp(-((opd / lc) ** 2)), rel=1e-8)
        return
    # at 20 mm mirror MB stands 11.7 mm from the splitter and its 12.7 mm disc comes within 1.28 mm (1.28 beam radii) of the
    # incoming beam's axis: about 0.5 % of the beam (half-plane 0.5 erfc(sqrt(2) 1.28)) meets MB's back before the splitter
    # while no pupil ray does. The share is bounded, but a mirror turns the light it meets rather than stopping it, so the
    # beam's power behind the splitter is out of model scope, and the pair's power quantities (visibility among them) are
    # withheld while its geometry and coherence stand
    _d1, _d2, _ma, mb, _s = _two_beam_geometry(opd_mm=opd)
    assert mb[1] == pytest.approx(11.7497, abs=1e-3)
    assert pair.fringe_visibility is None and "OUT_OF_MODEL_SCOPE" in pair.issue_codes


def test_exposure_time_from_dose_and_irradiance():
    targets = (Target(target_id="t", metric="estimated_exposure_time_s", observation_id="sub", minimum=0.0,
                      parameters={"dose_mj_per_cm2": 100.0}),)
    proj, *_ = _interference_project(targets=targets)
    res = run(proj)
    pair = res.interference[0]
    expected = 100.0 / (pair.mean_irradiance_W_per_mm2 * 1e5)
    assert res.targets[0].value == pytest.approx(expected, rel=1e-12)
    assert res.targets[0].unit == "s"


def test_missing_coherence_definition_is_unknown_not_pass():
    # a coherence margin needs a coherence length: a source that states none is taken as monochromatic (the visibility has a
    # value with its entry) and has none, so a margin target stays UNKNOWN and says what to declare
    proj, *_ = _interference_project(coherence_length_mm=None,
                                     targets=(Target(target_id="c", metric="coherence_margin", observation_id="sub", minimum=0.5),
                                              Target(target_id="p", metric="fringe_period_nm", observation_id="sub",
                                                     target_value=500.0, tolerance_abs=1e-3)))
    res = run(proj)
    by_id = {t.target_id: t for t in res.targets}
    assert by_id["c"].status.value == "UNKNOWN" and by_id["c"].value is None
    assert by_id["p"].status.value in ("PASS", "UNKNOWN") and by_id["p"].value == pytest.approx(500.0, abs=1e-6)
    assert "COHERENCE_UNDEFINED" not in issue_codes(res)
    assert res.interference[0].fringe_visibility is not None and "fringe_visibility" in res.interference[0].value_basis
    assert res.status.value == "UNKNOWN"  # the coherence target cannot be evaluated
    assert "declare the source's coherence_length_mm or linewidth_nm" in by_id["c"].message
    assert "COHERENCE_UNDEFINED" in by_id["c"].issue_codes


def test_undeclared_coherence_is_taken_as_monochromatic_with_its_basis():
    """A source that states no coherence is taken as monochromatic: the coherence-dependent quantities have values at a
    degree of coherence of 1, equal to a declared coherence length of 1e12 mm, each with an entry that names the stand-in;
    the coherence length and margin are not reported, and no COHERENCE_UNDEFINED issue is raised."""
    targets = (Target(target_id="p", metric="fringe_period_nm", observation_id="sub", target_value=500.0, tolerance_abs=1e-3),)
    res = run(_interference_project(coherence_length_mm=None, targets=targets)[0])
    long = run(_interference_project(coherence_length_mm=1e12, targets=targets)[0])
    assert res.status.value == "PASS" and [t.status.value for t in res.targets] == ["PASS"]
    assert not [i for i in res.issues if i.code.value == "COHERENCE_UNDEFINED"]
    pair, reference = res.interference[0], long.interference[0]
    assert pair.status.value == "PASS" and "COHERENCE_UNDEFINED" not in pair.issue_codes
    assert pair.fringe_period_nm == pytest.approx(500.0, abs=1e-6) and pair.optical_path_difference_mm is not None
    assert pair.coherence_factor == 1.0 and (pair.coherence_length_mm, pair.coherence_margin) == (None, None)
    for name in ("fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2"):
        assert getattr(pair, name) == pytest.approx(getattr(reference, name), rel=1e-12)
        entry = pair.value_basis[name]
        assert entry.kind.value == "UPPER_BOUND" and entry.assumed[0].input.value == "FULL_COHERENCE"
    assert reference.value_basis is None


@pytest.mark.parametrize("metric", ["coherence_length_mm", "coherence_margin", "coherence_factor", "fringe_visibility",
                                    "fringe_visibility_peak", "peak_irradiance_W_per_mm2"])
def test_every_coherence_metric_reads_the_declared_coherence_or_the_monochromatic_stand_in(metric):
    from os_optics.validation.metrics import COHERENCE_METRICS

    assert metric in COHERENCE_METRICS
    targets = (Target(target_id="m", metric=metric, observation_id="sub", minimum=0.0),)
    without, *_ = _interference_project(coherence_length_mm=None, targets=targets)
    with_, *_ = _interference_project(coherence_length_mm=10.0, targets=targets)
    missing, declared = run(without), run(with_)
    assert declared.targets[0].value is not None and declared.targets[0].status.value == "PASS"
    if metric in ("coherence_length_mm", "coherence_margin"):  # (a monochromatic stand-in has neither)
        assert missing.targets[0].value is None and missing.targets[0].status.value == "UNKNOWN"
    else:  # (a minimum of 0 holds for every degree of coherence from 0 to 1)
        assert missing.targets[0].value is not None and missing.targets[0].assessment.value == "MET"
        assert missing.targets[0].value_basis["value"].assumed[0].input.value == "FULL_COHERENCE"


def test_coherence_length_from_linewidth():
    """L_c = (2 sqrt(ln2)/pi) lambda^2 / d_lambda (|gamma| = 1/e at OPD = L_c for a Gaussian spectrum)."""
    from os_optics.scene.lower import coherence_length_from_linewidth_mm

    lc = coherence_length_from_linewidth_mm(405.0, 0.01)
    assert lc == pytest.approx((2.0 * math.sqrt(math.log(2.0)) / math.pi) * 405.0**2 / 0.01 * 1e-6, rel=1e-14)
    # Gaussian spectrum check: |gamma|(tau) = exp(-pi^2 dnu^2 tau^2/(4 ln2)); at OPD = L_c it equals exp(-1)
    c = 299792458.0
    dnu = c * (0.01e-9) / (405e-9) ** 2
    tau = lc * 1e-3 / c
    assert math.exp(-(math.pi**2) * dnu**2 * tau**2 / (4.0 * math.log(2.0))) == pytest.approx(math.exp(-1.0), rel=1e-9)


def test_beams_that_do_not_overlap_are_reported_not_passed():
    # move the substrate observation far from the crossing so the spatial overlap collapses: shift by translating MB laterally
    proj, d1, d2, s_pt = _interference_project(waist_mm=0.05)  # small beams, then displace one beam via mirror MB offset
    comps = list(proj.components)
    mb = comps[3]
    from os_optics.contracts import Pose3D

    # the beam reaching MB travels along +y, so shifting MB along x displaces the reflected beam laterally
    moved = mb.model_copy(update={"pose": Pose3D(position_mm=(mb.pose.position_mm[0] + 3.0, mb.pose.position_mm[1], mb.pose.position_mm[2]),
                                                 quaternion_wxyz=mb.pose.quaternion_wxyz)})
    comps[3] = moved
    res = run(proj.model_copy(update={"components": tuple(comps)}))
    pair = res.interference[0]
    assert pair.spatial_overlap_ratio is not None and pair.spatial_overlap_ratio < 1e-4
    # reported (not PASS) without failing the run by itself: an overlap-dependent target fails on its own numbers
    assert "BEAMS_NOT_OVERLAPPING" in pair.issue_codes and pair.status.value == "NOT_APPLICABLE"
    assert pair.fringe_visibility is not None and pair.fringe_visibility < 1e-4


def test_collinear_beams_have_no_defined_fringe_period():
    res = run(_mzi_project(0.0))
    pair = next(i for i in res.interference if i.observation_id == "port_x")
    assert pair.fringe_period_nm is None and "NO_FRINGES" in pair.issue_codes
    assert pair.beam_crossing_angle_deg == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Golden 11: multiple independent lasers.
# Reference: each source is traced independently; incoherent sources add in power and never interfere; results do not
# depend on the order of components in the project.
# ---------------------------------------------------------------------------------------------------------------
def _two_laser_project(order: str):
    lasers = [
        B.laser("LA", wavelength_nm=405.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=2e-3),
        B.laser("LB", wavelength_nm=532.0, waist_radius_mm=0.7, position=(0, 10, 20), direction=(1, 0, 0), power_W=5e-3),
        B.laser("LC", wavelength_nm=405.0, waist_radius_mm=0.5, position=(0, -10, 20), direction=(1, 0, 0), power_W=3e-3,
                coherence_length_mm=100.0),
    ]
    lens = B.thin_lens("Lens", focal_length_mm=100.0, clear_radius_mm=25.0, position=(50, 0, 20), axis=(1, 0, 0))
    comps = lasers + [lens]
    if order == "reversed":
        comps = comps[::-1]
    return ProjectSnapshot(components=tuple(comps),
                           observations=(B.observation_plane("det", position=(300, 0, 20), normal=(1, 0, 0), radius_mm=40.0),))


def test_golden_multiple_independent_lasers():
    res = run(_two_laser_project("normal"))
    by_src = {a.beam.source_id: a for a in arrivals(res, "det")}
    assert set(by_src) == {"LA", "LB", "LC"}
    assert {k: v.beam.power_W for k, v in by_src.items()} == pytest.approx({"LA": 2e-3, "LB": 5e-3, "LC": 3e-3}, rel=1e-12)
    assert sum(a.beam.power_W for a in by_src.values()) == pytest.approx(10e-3, rel=1e-12)
    # independent lasers do not interfere, even at the same wavelength (LA and LC): their pair is counted on the plane
    # (incoherent_pairs, one INCOHERENT_SOURCES notice), not analysed -- never PASS (a target naming them reads its analysis:
    # tests/test_incoherent_visibility.py)
    assert not any({i.source_id_a, i.source_id_b} == {"LA", "LC"} for i in res.interference)
    assert [(p.observation_id, p.pair_count, p.coherence_group_count) for p in res.incoherent_pairs] == [("det", 1, 3)]
    (notice,) = [i for i in res.issues if i.code.value == "INCOHERENT_SOURCES"]
    assert notice.status.value == "NOT_APPLICABLE" and notice.numerical_context["pair_count"] == 1
    # beams of different wavelengths never interfere: such pairs are not analysed at all
    assert not any({i.source_id_a, i.source_id_b} == {"LA", "LB"} for i in res.interference)
    # each beam is focused with its own q (waist at the source, lens at 50 mm). Lasers LB / LC sit 10 mm off axis, so the ideal
    # lens (slope form) deflects their chief rays by slope -h/f and the path after the lens is 250 sqrt(1 + (h/f)^2).
    for sid, (lam, w0, h) in {"LA": (405.0, 0.5, 0.0), "LB": (532.0, 0.7, 10.0), "LC": (405.0, 0.5, -10.0)}.items():
        zr = rayleigh_range(w0, lam)
        q = complex(0.0, zr) + 50.0
        q = q / (1.0 - q / 100.0) + 250.0 * math.sqrt(1.0 + (h / 100.0) ** 2)
        assert by_src[sid].beam.gaussian.q_x_mm[0] == pytest.approx(q.real, rel=1e-9)
        assert by_src[sid].beam.gaussian.q_x_mm[1] == pytest.approx(q.imag, rel=1e-9)
        assert by_src[sid].beam.geometric_length_mm == pytest.approx(50.0 + 250.0 * math.sqrt(1.0 + (h / 100.0) ** 2), rel=1e-12)


def test_component_order_does_not_change_physics():
    a = run(_two_laser_project("normal"))
    b = run(_two_laser_project("reversed"))
    da = {x.beam.source_id: x.beam.model_dump() for x in arrivals(a, "det")}
    db = {x.beam.source_id: x.beam.model_dump() for x in arrivals(b, "det")}
    assert da == db
