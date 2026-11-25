"""Linear optical network export: reference networks against independent textbook matrices, the classical-limit
self-check, structured refusals and determinism.

References never use Engine code: 2 x 2 beam-splitter algebra (t = sqrt(T), r = i sqrt(R) for s; the mirror law for the
polarization of a fold), the Fresnel window loss and Jones calculus. Only gauge-invariant quantities are compared (|T_ij|,
permanents of 2 x 2 blocks, T^H T, cycle phases), because a row / column phase is a choice of reference plane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot, ValidationRequest
from os_optics.contracts.network import LinearOpticalNetworkV1
from os_optics.local.network import export_request_network
from os_optics.manifest import hash_exact
from os_optics.network import NetworkErrorCode, NetworkExportError, contract_hash_of, lab_basis, mode_overlap

from generate_fixtures import REQ_DIR
from test_golden_interference import _mzi_project, bk7_index
from test_golden_polarization import _hwp_pbs_project

INV = 1.0 / math.sqrt(2.0)
NM_TO_MM = 1e-6


def quantization(wavelength_nm: float, amplitude_sum: float = 1.0) -> float:
    """The bound the declared quantization (optical path lengths rounded to 1e-9 mm before the phases) puts on |T_ij|^2."""
    return amplitude_sum ** 2 * 2 * math.pi / (wavelength_nm * NM_TO_MM) * 1e-9 + 1e-8


def _request(project: ProjectSnapshot) -> ValidationRequest:
    return ValidationRequest(project=project)


def export(project: ProjectSnapshot, **options) -> LinearOpticalNetworkV1:
    return export_request_network(_request(project), options=options or None)


def refusal(project: ProjectSnapshot, **options) -> set[str]:
    with pytest.raises(NetworkExportError) as info:
        export(project, **options)
    return {p.code.value for p in info.value.problems}


def matrix(net: LinearOpticalNetworkV1) -> np.ndarray:
    return np.array([[complex(*z) for z in row] for row in net.transfer_matrix.entries])


def entry(net: LinearOpticalNetworkV1, row: str, col: str) -> complex:
    tm = net.transfer_matrix
    return complex(*tm.entries[tm.rows.index(row)][tm.columns.index(col)])


# ---------------------------------------------------------------------------------------------------- mzi_window


def test_mzi_window_fixture_exports_the_textbook_network():
    """Acceptance 1: the frozen mzi_window request. V (s) input; BS r = i/sqrt2, t = 1/sqrt2 on both sides; mirrors -1 for s;
    the BK7 window transmits (1 - R_F) in amplitude (its two faces' untraced reflections are loss) and adds (n - 1) t."""
    request = ValidationRequest.model_validate_json((REQ_DIR / "mzi_window.json").read_text(encoding="utf-8"))
    net = export_request_network(request)
    lam, t = 532.0, 0.4123
    n = bk7_index(lam)
    k0 = 2 * math.pi / (lam * NM_TO_MM)
    rf = ((n - 1) / (n + 1)) ** 2
    tt, rr = INV, 1j * INV
    ea = -1.0
    eb = -(1 - rf) * np.exp(1j * k0 * (n - 1) * t)
    ref = {"port_x": tt * rr * ea + rr * tt * eb, "port_y": tt * tt * ea + rr * rr * eb}
    assert net.transfer_matrix.columns == ("in:L0#0:H", "in:L0#0:V")
    assert net.transfer_matrix.rows == ("out:port_x:0:H", "out:port_x:0:V", "out:port_y:0:H", "out:port_y:0:V")
    for port, amplitude in ref.items():
        assert abs(entry(net, f"out:{port}:0:V", "in:L0#0:V")) ** 2 == pytest.approx(abs(amplitude) ** 2, abs=quantization(lam))
        assert abs(entry(net, f"out:{port}:0:H", "in:L0#0:V")) == 0.0  # V never becomes H at in-plane folds
        assert abs(entry(net, f"out:{port}:0:V", "in:L0#0:H")) == 0.0
    tm = matrix(net)
    loss = (1 + (1 - rf) ** 2) / 2  # half the light crosses the window, which keeps (1 - R_F)^2 of it
    assert tm.conj().T @ tm == pytest.approx(loss * np.eye(2), abs=2e-9)
    assert not net.transfer_matrix.lossless
    assert net.transfer_matrix.column_transmission == pytest.approx((loss, loss), abs=2e-9)
    # the same for H (p) light: its p reflections at the splitters have the opposite sign (r_p = -r_s), as at a mirror
    for port in ("port_x", "port_y"):
        assert abs(entry(net, f"out:{port}:0:H", "in:L0#0:H")) == pytest.approx(abs(entry(net, f"out:{port}:0:V", "in:L0#0:V")), abs=1e-8)
    # four paths, two per output beam, with the window's extra optical path in the B arm
    assert len(net.paths) == 4
    opd = {p.output_beam: [] for p in net.paths}
    for p in net.paths:
        opd[p.output_beam].append(p.opl_mm)
    for beam, opls in opd.items():
        assert abs(opls[0] - opls[1]) == pytest.approx((n - 1) * t, abs=2e-9)
    check = net.classical_check
    assert check.status == "PASS" and all(plane.deviation <= 1e-6 for column in check.columns for plane in column.planes)
    column = check.columns[0]
    assert dict(column.lost_path_power_by_kind)["UNTRACED_REFLECTION"] == pytest.approx(0.5 * (1 - (1 - rf) ** 2), abs=1e-9)
    assert column.ledger_residual == pytest.approx(0.0, abs=1e-9)
    assert net.provenance.engine_input_hash == hash_exact(request.model_dump(mode="json"))
    assert net.provenance.project_hash is None and net.provenance.catalog_version is None


def test_mzi_element_matrices_are_the_textbook_jones_matrices():
    net = export(_mzi_project(0.0))
    by = {(e.surface_id, e.event): e for e in net.elements}
    m = lambda e: np.array([[complex(*z) for z in row] for row in e.matrix])  # noqa: E731
    assert m(by[("MA/face", "reflect")]) == pytest.approx(np.diag([1.0, -1.0]), abs=1e-9)  # mirror law at a horizontal fold
    assert m(by[("BS1/coating", "split_transmit")]) == pytest.approx(INV * np.eye(2), abs=1e-9)
    # r_s = i/sqrt2 (V is s), r_p = -i/sqrt2 (H is p): exactly the mirror's matrix times -i/sqrt2
    assert m(by[("BS1/coating", "split_reflect")]) == pytest.approx(-1j * INV * np.diag([1.0, -1.0]), abs=1e-9)
    assert by[("BS1/coating", "split_reflect")].aoi_deg == pytest.approx(45.0, abs=1e-9)


def test_lossless_mzi_is_unitary_for_both_polarizations_and_matches_the_analytic_phase():
    lam, L = 532.0, 100.0
    net = export(_mzi_project(0.0, lam, L))
    tm = matrix(net)
    assert tm.conj().T @ tm == pytest.approx(np.eye(2), abs=2e-9)
    assert net.transfer_matrix.lossless
    # equal arms: all the light leaves port_y (t t e + r r e = 0 at port_y? -> t^2 + r^2 = 0; port_x gets 2 t r)
    assert abs(entry(net, "out:port_x:0:V", "in:L0#0:V")) ** 2 == pytest.approx(1.0, abs=2e-9)
    assert abs(entry(net, "out:port_y:0:V", "in:L0#0:V")) ** 2 == pytest.approx(0.0, abs=2e-9)
    assert abs(entry(net, "out:port_x:0:H", "in:L0#0:H")) ** 2 == pytest.approx(1.0, abs=2e-9)


# ---------------------------------------------------------------------------------------------------- Michelson


def _michelson(l1: float = 80.0, l2: float = 80.25, *, back_plane: bool = True) -> ProjectSnapshot:
    observations = [B.observation_plane("out", position=(0, -50, 20), normal=(0, -1, 0), radius_mm=15.0)]
    if back_plane:
        observations.append(B.observation_plane("back", position=(-30, 0, 20), normal=(1, 0, 0), radius_mm=15.0))
    return ProjectSnapshot(components=(
        B.laser("L0", wavelength_nm=633.0, waist_radius_mm=0.6, position=(-60, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                coherence_length_mm=1e4, polarization=JonesVector.linear(90.0)),
        B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5),
        B.mirror("M1", position=(l1, 0, 20), normal=(-1, 0, 0)),
        B.mirror("M2", position=(0, l2, 20), normal=(0, -1, 0)),
    ), observations=tuple(observations))


def test_michelson_output_follows_the_textbook_fringe():
    l1, l2, lam = 80.0, 80.25, 633.0
    net = export(_michelson(l1, l2), observation_ids=["out"])
    k0 = 2 * math.pi / (lam * NM_TO_MM)
    t, r = INV, 1j * INV
    for pol in ("H", "V"):
        # arm 1: transmitted, mirror (-1 for s, the H/V mirror sign cancels in |.|), reflected; arm 2: reflected, mirror, transmitted
        expected = abs(t * r + r * t * np.exp(1j * k0 * 2 * (l2 - l1))) ** 2
        assert abs(entry(net, f"out:out:0:{pol}", f"in:L0#0:{pol}")) ** 2 == pytest.approx(expected, abs=quantization(lam))
    # the light going back to the laser leaves the network: by path powers, half of it; coherently 1 - |T|^2
    column = net.classical_check.columns[0]
    assert column.lost_path_power_fraction == pytest.approx(0.5, abs=1e-9) and column.ledger_residual == pytest.approx(0.0, abs=1e-9)
    assert dict(column.lost_path_power_by_kind) == {"ESCAPED": pytest.approx(0.5, abs=1e-9)}


def test_a_plane_the_beam_crosses_before_further_optics_is_not_an_output_port():
    assert refusal(_michelson()) == {NetworkErrorCode.OUTPUT_NOT_TERMINAL.value}


# ---------------------------------------------------------------------------------------------------- HWP + PBS


@pytest.mark.parametrize("alpha", [0.0, 22.5, 30.0])
def test_hwp_and_pbs_match_the_jones_product(alpha):
    net = export(_hwp_pbs_project(alpha))
    c, s = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    rot = np.array([[c, -s], [s, c]])
    hwp = rot @ np.diag([np.exp(-0.5j * math.pi), np.exp(0.5j * math.pi)]) @ rot.T  # in (H, V) of the beam (fast axis from H toward V)
    # ideal PBS: transmits H (p), reflects V (s)
    for j, pin in enumerate(("H", "V")):
        out = hwp[:, j]
        assert abs(entry(net, "out:T:0:H", f"in:L#0:{pin}")) ** 2 == pytest.approx(abs(out[0]) ** 2, abs=2e-9)
        assert abs(entry(net, "out:T:0:V", f"in:L#0:{pin}")) == pytest.approx(0.0, abs=1e-9)
        assert abs(entry(net, "out:R:0:V", f"in:L#0:{pin}")) ** 2 == pytest.approx(abs(out[1]) ** 2, abs=2e-9)
        assert abs(entry(net, "out:R:0:H", f"in:L#0:{pin}")) == pytest.approx(0.0, abs=1e-9)
    tm = matrix(net)
    assert tm.conj().T @ tm == pytest.approx(np.eye(2), abs=2e-9)
    # gauge invariant: the relative phase of the two columns within one row is the waveplate's
    if 0.0 < alpha < 45.0:
        got = entry(net, "out:T:0:H", "in:L#0:H") / entry(net, "out:T:0:H", "in:L#0:V")
        assert np.angle(got / (hwp[0, 0] / hwp[0, 1])) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------------- HOM


def _hom(polarization_deg: float = 0.0, *, d2_offset: float = 0.0) -> ProjectSnapshot:
    pol = JonesVector.linear(polarization_deg)
    return ProjectSnapshot(components=(
        B.laser("A", wavelength_nm=780.0, waist_radius_mm=0.5, position=(-50, 0, 20), direction=(1, 0, 0), power_W=1e-3, polarization=pol),
        B.laser("Bsrc", wavelength_nm=780.0, waist_radius_mm=0.5, position=(0, -50, 20), direction=(0, 1, 0), power_W=1e-3, polarization=pol),
        B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5),
    ), observations=(
        B.observation_plane("D1", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
        B.observation_plane("D2", position=(d2_offset, 60, 20), normal=(0, 1, 0), radius_mm=10.0),
    ))


def test_two_source_beam_splitter_is_unitary_with_a_vanishing_permanent():
    """Hong-Ou-Mandel as a gauge-invariant statement: per polarization, the 2 x 2 block of a 50:50 splitter has permanent 0."""
    net = export(_hom(45.0))
    tm = matrix(net)
    assert tm.shape == (4, 4)
    assert tm.conj().T @ tm == pytest.approx(np.eye(4), abs=2e-9)
    for pol in ("H", "V"):
        a, b = f"in:A#0:{pol}", f"in:Bsrc#0:{pol}"
        d1, d2 = f"out:D1:0:{pol}", f"out:D2:0:{pol}"
        per = entry(net, d1, a) * entry(net, d2, b) + entry(net, d1, b) * entry(net, d2, a)
        assert abs(per) < 1e-8
        assert abs(entry(net, d1, a)) ** 2 == pytest.approx(0.5, abs=2e-9)
    pairs = [p for p in net.hints.path_pairs if not p.same_source]
    assert pairs and all(p.coherence_length_mm is None for p in pairs)  # independent lasers without declared coherence
    (h_re, h_im), (v_re, v_im) = net.input_modes[0].source_polarization
    assert (h_re, h_im, v_re, v_im) == pytest.approx((INV, 0.0, INV, 0.0), abs=1e-12)  # 45 deg from H toward V


# ---------------------------------------------------------------------------------------------------- refusals


def _rotated(component, angle_rad: float):
    """``component`` turned by ``angle_rad`` about the design z axis through its position."""
    w, x, y, z = component.pose.quaternion_wxyz
    c, s = math.cos(angle_rad / 2), math.sin(angle_rad / 2)
    q = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)
    return component.model_copy(update={"pose": component.pose.model_copy(update={"quaternion_wxyz": q})})


def _with(project: ProjectSnapshot, instance_id: str, replace) -> ProjectSnapshot:
    return project.model_copy(update={"components": tuple(replace(c) if c.instance_id == instance_id else c for c in project.components)})


def test_a_tilted_recombining_splitter_is_refused_as_partial_mode_overlap():
    """BS2 turned by 0.1 mrad: the arms meet at 0.2 mrad, mode overlap exp(-(pi w theta / lambda)^2) ~ 0.4."""
    tilted = _with(_mzi_project(0.0), "BS2", lambda c: _rotated(c, 1e-4))
    assert refusal(tilted) == {NetworkErrorCode.PARTIAL_MODE_OVERLAP.value}


def test_beams_tilted_far_apart_are_separate_modes():
    """BS2 turned by 0.5 mrad: the arms meet at 1 mrad, overlap ~3e-10 -- two orthogonal output modes per port, and the
    Engine's whole-plane superposition (no cross term left) agrees."""
    net = export(_with(_mzi_project(0.0), "BS2", lambda c: _rotated(c, 5e-4)))
    assert net.transfer_matrix.rows == tuple(f"out:{port}:{beam}:{pol}" for port in ("port_x", "port_y") for beam in (0, 1) for pol in "HV")
    tm = matrix(net)
    assert tm.conj().T @ tm == pytest.approx(np.eye(2), abs=1e-8)
    assert all(abs(abs(z) ** 2 - 0.25) < 1e-8 or abs(z) < 1e-9 for z in tm.flatten())


def test_an_offset_beam_is_refused_as_partial_mode_overlap():
    shifted = _with(_mzi_project(0.0), "MB", lambda c: c.model_copy(update={"pose": c.pose.model_copy(
        update={"position_mm": (c.pose.position_mm[0], c.pose.position_mm[1] + 0.4 * INV, c.pose.position_mm[2])})}))
    assert refusal(shifted) == {NetworkErrorCode.PARTIAL_MODE_OVERLAP.value}


def test_a_loose_mode_tolerance_is_caught_by_the_classical_limit_check():
    """Deliberately inconsistent: a 20 urad tilt accepted as one mode (tolerance 0.1) -- the Engine's coherent
    superposition over the plane (its fringes) disagrees with |T J|^2."""
    tilted = _with(_mzi_project(0.0), "BS2", lambda c: _rotated(c, 2e-5))
    assert refusal(tilted, mode_overlap_tolerance=0.1) == {NetworkErrorCode.CLASSICAL_LIMIT_MISMATCH.value}


def test_mixed_wavelengths_are_refused_unless_one_line_is_chosen():
    project = _hom()
    other = project.components[1]
    project = _with(project, "Bsrc", lambda c: c.model_copy(update={"source": c.source.model_copy(update={"wavelength_nm": 633.0})}))
    assert other.source.wavelength_nm == 780.0
    assert refusal(project) == {NetworkErrorCode.MIXED_WAVELENGTHS.value}
    net = export(project, wavelength_nm=633.0)
    assert net.transfer_matrix.columns == ("in:Bsrc#0:H", "in:Bsrc#0:V")
    assert any(n.code == "SOURCE_WITHOUT_LINE" for n in net.hints.notes)
    assert refusal(project, wavelength_nm=532.0) == {NetworkErrorCode.WAVELENGTH_NOT_EMITTED.value}


def test_an_ideal_thin_lens_on_a_path_is_refused():
    project = ProjectSnapshot(components=(
        B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
        B.thin_lens("TL", position=(50, 0, 20), axis=(1, 0, 0), focal_length_mm=100.0, clear_radius_mm=10.0),
    ), observations=(B.observation_plane("o", position=(150, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    assert refusal(project) == {NetworkErrorCode.PHASE_UNDEFINED_IDEAL_LENS.value}


def test_a_resonator_is_refused():
    project = ProjectSnapshot(components=(
        B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
        B.npbs("M1", position=(50, 0, 20), normal=(-1, 0, 0), reflectance=0.99),
        B.npbs("M2", position=(60, 0, 20), normal=(-1, 0, 0), reflectance=0.99),
    ), observations=(B.observation_plane("o", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    assert refusal(project) == {NetworkErrorCode.CAVITY_NOT_SUPPORTED.value}


def test_unknown_options_are_refused():
    assert refusal(_hom(), observation_ids=["nope"]) == {NetworkErrorCode.OPTIONS_INVALID.value}
    assert refusal(_hom(), source_ids=["nope"]) == {NetworkErrorCode.OPTIONS_INVALID.value}
    assert refusal(_hom(), mode_overlap_tolerance=0.5) == {NetworkErrorCode.OPTIONS_INVALID.value}


# ---------------------------------------------------------------------------------------------------- determinism, conventions


def test_the_export_is_deterministic_and_its_hash_covers_the_body():
    first = export(_mzi_project(0.4123))
    second = export(_mzi_project(0.4123))
    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(mode="json", by_alias=True)
    document = first.model_dump(mode="json", by_alias=True)
    assert contract_hash_of(document) == first.contract_hash
    document["transfer_matrix"]["entries"][0][0][0] += 1e-9
    assert contract_hash_of(document) != first.contract_hash
    assert document["schema"] == "opensci-linear-optical-network/v1"


def test_lab_basis_is_right_handed_with_v_up_and_a_vertical_fallback():
    for d in ([1, 0, 0], [0, -1, 0], [INV, INV, 0], [0.8, 0, 0.6]):
        h, v = lab_basis(d)
        assert np.cross(h, v) == pytest.approx(np.array(d) / np.linalg.norm(d))
        assert v[2] > 0
    h, v = lab_basis([0, 0, 1])
    assert v == pytest.approx([0, 1, 0]) and h == pytest.approx([1, 0, 0])
    # within 45 deg of vertical V is +Y projected: right-handed too, and continuous across the old 1 mrad switch
    for d in ([0.6, 0, 0.8], [0, -0.3, -0.95]):
        h, v = lab_basis(d)
        assert np.cross(h, v) == pytest.approx(np.array(d) / np.linalg.norm(d)) and v[1] > 0
    for azimuth in (0.0, 0.7, 2.0):
        below, above = ([math.sin(t) * math.cos(azimuth), math.sin(t) * math.sin(azimuth), math.cos(t)] for t in (0.99e-3, 1.01e-3))
        assert np.array(lab_basis(below)) == pytest.approx(np.array(lab_basis(above)), abs=1e-4)


def test_mode_overlap_of_tilted_and_offset_gaussians_matches_the_closed_form():
    """|<a|b>|^2 = exp(-(pi w theta / lambda)^2) for a tilt theta and exp(-(d / w)^2) for an offset d of equal beams at their
    waist (each source sits on the plane's upstream side with its waist on the plane)."""
    from os_optics.validation.pipeline import CHIEF_RAYS, run_physics

    def beam(direction, position):
        project = ProjectSnapshot(components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=position, direction=direction,
                                                      power_W=1e-3, waist_position_z_mm=10.0),),
                                  observations=(B.observation_plane("o", position=(0, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
        return run_physics(project, NumericalConfig(), plan=CHIEF_RAYS).trace.arrivals[0]

    a = beam((1, 0, 0), (-10, 0, 20))
    theta = 2e-4
    b = beam((math.cos(theta), math.sin(theta), 0), (-10 * math.cos(theta), -10 * math.sin(theta), 20))  # through the same waist point
    assert mode_overlap(a, b) == pytest.approx(math.exp(-(math.pi * 0.5 * theta / 633e-6) ** 2), rel=1e-6)
    c = beam((1, 0, 0), (-10, 0.2, 20))
    assert mode_overlap(a, c) == pytest.approx(math.exp(-(0.2 / 0.5) ** 2), rel=1e-9)
    assert mode_overlap(a, a) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------------- continuous spectrum

GAUSSIAN_800 = {"shape": {"kind": "gaussian", "center_nm": 800.0, "fwhm_nm": 20.0}, "band_nm": (770.0, 830.0), "line_count": 15}


def _continuous_michelson(delta_mm: float, *, stated_phases: bool) -> ProjectSnapshot:
    """``_michelson``'s layout lit by a Gaussian continuous spectrum (800 nm, 20 nm FWHM, 15 quadrature lines), arm difference
    ``delta_mm``; ``stated_phases``: the splitter states constant phases (r = i / sqrt 2, t = 1 / sqrt 2), so the run knows the
    paths' group delays and the Engine's superposition of the source's own beams reads the spectrum's coherence."""
    from os_optics.contracts import SpectralDensity, SpectralLine
    from os_optics.contracts.optics import BeamSplitterInteraction, InterfacePhases
    from os_optics.optics.interference import spectrum

    density = SpectralDensity.model_validate(GAUSSIAN_800)
    lines = spectrum.quadrature_lines(density)
    laser = B.laser("L0", wavelength_nm=800.0, waist_radius_mm=0.6, position=(-60, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                    polarization=JonesVector.linear(90.0))
    src = laser.source.model_copy(update={"wavelength_nm": None, "spectral_density": density, "linewidth_nm": spectrum.bin_width_nm(density),
                                          "spectrum": tuple(SpectralLine(wavelength_nm=w, weight=p) for w, p in lines)})
    laser = laser.model_copy(update={"source": type(src).model_validate(src.model_dump())})
    bs = B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5)
    if stated_phases:
        surface = bs.surfaces[0]
        phases = InterfacePhases(reflection_s_rad=(math.pi / 2,), reflection_p_rad=(math.pi / 2,), transmission_s_rad=(0.0,),
                                 transmission_p_rad=(0.0,))
        interaction = BeamSplitterInteraction.model_validate(surface.interaction.model_copy(update={"phases": phases}).model_dump())
        bs = bs.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": interaction}),)})
    return ProjectSnapshot(components=(
        laser, bs, B.mirror("M1", position=(80.0, 0, 20), normal=(-1, 0, 0)), B.mirror("M2", position=(0, 80.0 + delta_mm, 20), normal=(0, -1, 0)),
    ), observations=(B.observation_plane("out", position=(0, -50, 20), normal=(0, -1, 0), radius_mm=15.0),))


def _reference_coherence_length_mm() -> float:
    """The smallest D with |gamma(D)| = 1/e for GAUSSIAN_800, independent of the Engine: gamma(D) = int S(lambda) exp(2 pi i D /
    lambda) d lambda / int S over the band, S = exp(-4 ln 2 ((lambda - 800) / 20)^2), dense trapezoid (400 001 points), then
    bisection of the first crossing."""
    lam = np.linspace(770.0, 830.0, 400_001)
    s = np.exp(-4.0 * math.log(2.0) * ((lam - 800.0) / 20.0) ** 2)
    norm = np.trapezoid(s, lam) if hasattr(np, "trapezoid") else np.trapz(s, lam)

    def gamma(d_mm: float) -> float:
        z = s * np.exp(2j * math.pi * d_mm / (lam * 1e-6))
        integral = np.trapezoid(z, lam) if hasattr(np, "trapezoid") else np.trapz(z, lam)
        return abs(integral) / norm

    low, high = 0.0, 0.001
    while gamma(high) > math.exp(-1.0):
        low, high = high, high + 0.001
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if gamma(middle) > math.exp(-1.0) else (low, middle)
    return 0.5 * (low + high)


def test_a_continuous_spectrum_exports_the_spectrum_coherence_length_not_the_quadrature_bin():
    # Each quadrature line is traced as a Gaussian line one bin (4 nm) wide; its coherence length (2 sqrt(ln 2) / pi) lambda^2 / w
    # = 0.0848 mm is a property of the discretization. The network's input mode and path-pair hint carry the spectrum's
    # (0.01696 mm): at the 0.03 mm path difference the two arms are 1.77 coherence lengths apart, not 0.35.
    delta = 0.015
    net = export(_continuous_michelson(delta, stated_phases=False), wavelength_nm=800.0)
    reference = _reference_coherence_length_mm()
    bin_length = 2.0 * math.sqrt(math.log(2.0)) / math.pi * 800.0 ** 2 / 4.0 * NM_TO_MM
    assert bin_length == pytest.approx(0.0848033163, rel=1e-9)
    assert {m.mode_id for m in net.input_modes} == {"in:L0#7:H", "in:L0#7:V"}
    for mode in net.input_modes:
        assert mode.coherence_length_mm == pytest.approx(reference, rel=1e-6)
    (pair,) = net.hints.path_pairs
    assert abs(pair.opd_mm) == pytest.approx(2 * delta, abs=1e-9)
    assert pair.coherence_length_mm == pytest.approx(reference, rel=1e-6)
    assert pair.opd_over_coherence_length == pytest.approx(2 * delta / reference, rel=1e-6)
    assert pair.opd_over_coherence_length > 1.7
    (note,) = [n for n in net.hints.notes if n.code == "CONTINUOUS_SPECTRUM"]
    assert note.location == "source:L0" and "15 quadrature lines of 4 nm" in note.message
    assert not any(n.code == "COHERENCE_BEYOND_SCAN" for n in net.hints.notes)


def test_a_continuous_spectrum_line_is_exported_as_one_frequency_when_the_group_delays_are_known():
    # With the splitter's phases stated the paths' group delays are known, and the Engine's superposition of the source's two
    # beams reads the spectrum's |gamma| (0.045 at 0.03 mm). The network is one frequency: its classical-limit check now traces
    # the line alone (it used to compare the line's network with the whole spectrum's fringe contrast and refuse the export as
    # CLASSICAL_LIMIT_MISMATCH). The line's fringe follows the textbook Michelson.
    from os_optics.network.export import export_scope_projects
    from os_optics.contracts import SourceSpec
    from os_optics.contracts.network import NetworkExportOptionsV1

    delta = 0.0151
    project = _continuous_michelson(delta, stated_phases=True)
    net = export(project, wavelength_nm=800.0)
    k0 = 2 * math.pi / (800.0 * NM_TO_MM)
    t, r = INV, 1j * INV
    for pol in ("H", "V"):
        expected = abs(t * r + r * t * np.exp(1j * k0 * 2 * delta)) ** 2
        assert abs(entry(net, f"out:out:0:{pol}", f"in:L0#7:{pol}")) ** 2 == pytest.approx(expected, abs=quantization(800.0))
    for column in net.classical_check.columns:
        for plane in column.planes:
            assert plane.deviation <= plane.allowance
    # the model-scope runs of the export are the exported line alone too: a source the contract accepts
    for scoped in export_scope_projects(project, NumericalConfig(), NetworkExportOptionsV1(wavelength_nm=800.0)):
        (laser,) = [c for c in scoped.components if c.source is not None]
        assert laser.source.spectral_density is None and laser.source.wavelength_nm == 800.0
        SourceSpec.model_validate(laser.source.model_dump())
