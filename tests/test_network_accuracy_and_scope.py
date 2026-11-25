"""Linear optical network export: final-review regressions (precision, passivity, scope, refusal causes).

Every reference is independent of the Engine: 2 x 2 splitter algebra (t = sqrt(T), r_s = i sqrt(R)), the mirror sign for s
light (-1), optical path lengths from the layout's own coordinates, the Gaussian power through a circular aperture
(1 - exp(-2 a^2 / w^2)) and Jones calculus. Only gauge-invariant quantities are compared (|T_ij|, T^H T, singular values)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import ConstantCoating, JonesVector, ProjectSnapshot
from os_optics.network import NetworkErrorCode, NetworkExportError

from test_golden_interference import _mzi_project
from test_golden_polarization import _hwp_pbs_project
from test_network_export import INV, NM_TO_MM, entry, export, matrix, refusal


def _laser(**overrides):
    kwargs = dict(wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3)
    kwargs.update(overrides)
    instance_id = kwargs.pop("instance_id", "L")
    return B.laser(instance_id, **kwargs)


def _plane(observation_id="o", x=100.0, **kwargs):
    return B.observation_plane(observation_id, position=(x, 0, 20), normal=(1, 0, 0), **kwargs)


def recipe_matrix(net) -> np.ndarray:
    """T rebuilt from the documented recipe: sum over paths of amplitudes exp(i k0 reference_opl_mm) exp(i k0 opl_offset_mm)."""
    k0 = 2 * math.pi / (net.wavelength_nm * NM_TO_MM)
    reference = {m.mode_id.rsplit(":", 1)[0]: m.reference_opl_mm for m in net.output_modes}
    tm = net.transfer_matrix
    t = np.zeros((len(tm.rows), len(tm.columns)), dtype=complex)
    for path in net.paths:
        phase = np.exp(1j * k0 * reference[path.output_beam]) * np.exp(1j * k0 * path.opl_offset_mm)
        line = next(m.spectral_index for m in net.input_modes if m.source_id == path.source_id)
        for r, pout in enumerate("HV"):
            for c, pin in enumerate("HV"):
                t[tm.rows.index(f"{path.output_beam}:{pout}"), tm.columns.index(f"in:{path.source_id}#{line}:{pin}")] += (
                    complex(*path.amplitudes[r][c]) * phase)
    return t


# ---------------------------------------------------------------------------------------------------- M3: quantization


def _michelson_both_ports(l1, l2, out_y, back_x):
    return ProjectSnapshot(components=(
        B.laser("L0", wavelength_nm=633.0, waist_radius_mm=0.6, position=(-60, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                coherence_length_mm=1e4, polarization=JonesVector.linear(90.0)),
        B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5),
        B.mirror("M1", position=(l1, 0, 20), normal=(-1, 0, 0)),
        B.mirror("M2", position=(0, l2, 20), normal=(0, -1, 0)),
    ), observations=(B.observation_plane("out", position=(0, -out_y, 20), normal=(0, -1, 0), radius_mm=15.0),
                     B.observation_plane("back", position=(-back_x, 0, 20), normal=(-1, 0, 0), radius_mm=15.0)))


def test_a_lossless_michelson_with_sub_micron_positions_stays_unitary():
    """Optical path lengths rounded independently per path (1e-9 mm, k0 q ~ 1e-5 rad) made 3 of 12 such
    layouts lossless: false with singular values 1.0000019. With the per-beam reference and 1e-12 mm offsets every trial is
    unitary to 2e-9 and matches the textbook Michelson (both ports) to 1e-8."""
    lam = 633.0
    k0 = 2 * math.pi / (lam * NM_TO_MM)
    rng = np.random.default_rng(7)
    t, r, m = INV, 1j * INV, -1.0  # s light: splitter t, r = i/sqrt2; mirror -1
    for _trial in range(12):
        l1 = 80.0 + rng.uniform(0, 1e-3)
        l2 = l1 + 0.25 + (math.pi / 2) / (2 * k0) + rng.uniform(-1, 1) * 1e-7
        out_y, back_x = 50.0 + rng.uniform(0, 1e-3), 70.0 + rng.uniform(0, 1e-3)
        net = export(_michelson_both_ports(l1, l2, out_y, back_x))
        tm = matrix(net)
        assert net.transfer_matrix.lossless and net.transfer_matrix.lossless_tolerance == pytest.approx(1e-6)
        assert np.abs(tm.conj().T @ tm - np.eye(2)).max() < 2e-9
        assert max(net.transfer_matrix.singular_values) <= 1.0 + 2e-9
        out = t * m * r * np.exp(1j * k0 * (60 + 2 * l1 + out_y)) + r * m * t * np.exp(1j * k0 * (60 + 2 * l2 + out_y))
        back = t * m * t * np.exp(1j * k0 * (60 + 2 * l1 + back_x)) + r * m * r * np.exp(1j * k0 * (60 + 2 * l2 + back_x))
        assert abs(entry(net, "out:out:0:V", "in:L0#0:V")) ** 2 == pytest.approx(abs(out) ** 2, abs=1e-8)
        assert abs(entry(net, "out:back:0:V", "in:L0#0:V")) ** 2 == pytest.approx(abs(back) ** 2, abs=1e-8)
        # the documented recipe reproduces T (to the entry quantum), and the offsets are the layout's arm difference
        assert recipe_matrix(net) == pytest.approx(tm, abs=3e-9)
        offsets = sorted(abs(p.opl_offset_mm) for p in net.paths)
        assert offsets[-1] == pytest.approx(2 * (l2 - l1), abs=1e-9)


def test_the_conventions_state_the_offset_quantum_and_the_real_phase_accuracy():
    net = export(_mzi_project(0.0))
    quanta = {q.quantity: (q.resolution, q.unit) for q in net.conventions.quantization}
    assert any("relative to its output beam's reference" in k and v == (1e-12, "mm") for k, v in quanta.items())
    assert any(k.startswith("element angles of incidence") and v == (1e-6, "deg") for k, v in quanta.items())
    assert any(k.startswith("wavelength_nm") and v[1] == "12 significant digits" for k, v in quanta.items())
    assert "k0 x 5e-13 mm" in net.conventions.propagation_phase


# ---------------------------------------------------------------------------------------------------- m4: passivity


def _hom_with_phase(phase: float) -> ProjectSnapshot:
    coat = ConstantCoating(reflectance_s=0.5, reflectance_p=0.5, transmittance_s=0.5, transmittance_p=0.5)
    return ProjectSnapshot(components=(
        _laser(instance_id="A", wavelength_nm=780.0, position=(-50, 0, 20)),
        _laser(instance_id="Bs", wavelength_nm=780.0, position=(0, -50, 20), direction=(0, 1, 0)),
        B.beam_splitter("BS", position=(0, 0, 20), normal=(-INV, INV, 0), coating=coat, reflection_phase_rad=phase)),
        observations=(B.observation_plane("D1", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("D2", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=10.0)))


@pytest.mark.parametrize("phase", [0.0, math.pi / 4])
def test_a_splitter_that_creates_light_is_refused(phase):
    """[[t, r], [r, t]] with t = 1/sqrt2, r = e^{i phi}/sqrt2 has singular values |1 +- e^{i phi}| / sqrt2
    (1.414 / 0 for phi = 0): gain, which was exported silently as lossless: false. The splitter itself is refused: its
    largest singular value squared |1 + e^{i phi}|^2 / 2 exceeds 1."""
    with pytest.raises(Exception) as info:
        export(_hom_with_phase(phase))
    problems = info.value.problems
    assert {p.code for p in problems} == {NetworkErrorCode.NETWORK_NOT_PASSIVE}
    assert problems[0].location.startswith("surface:BS")
    assert f"put out up to {abs(1 + np.exp(1j * phase)) ** 2 / 2:.9g} of the light" in problems[0].message


def test_the_unitary_splitter_phase_stays_exportable():
    net = export(_hom_with_phase(math.pi / 2))
    assert net.transfer_matrix.lossless and net.transfer_matrix.singular_values == pytest.approx((1.0,) * 4, abs=2e-9)


# ---------------------------------------------------------------------------------------------------- m3: normal incidence


@pytest.mark.parametrize("roll_deg", [0.0, 30.0, 90.0])
def test_a_polarizing_splitter_at_normal_incidence_is_refused(roll_deg):
    """Lab-vertical light through a PBS at normal incidence gave 1.0 / 0.75 / 0.0 for source rolls 0 / 30 /
    90 deg (the s axis is the ray's internal e1 there). The export refuses it; F1 makes only a non-polarizing splitter
    independent of that axis."""
    a = math.radians(roll_deg)
    up = (0.0, -math.sin(a), math.cos(a))
    project = ProjectSnapshot(components=(_laser(up=up), B.pbs("P", position=(50, 0, 20), normal=(-1, 0, 0))),
                              observations=(_plane("fwd"),))
    assert refusal(project) == {NetworkErrorCode.POLARIZING_NORMAL_INCIDENCE.value}


def test_a_non_polarizing_splitter_at_normal_incidence_is_exported():
    """|t|^2 = 1 - R for every polarization and source roll (the reflection goes back toward the laser and is lost)."""
    for up in ((0.0, 0.0, 1.0), (0.0, -0.5, math.sqrt(0.75))):
        project = ProjectSnapshot(components=(_laser(up=up), B.npbs("S", position=(50, 0, 20), normal=(-1, 0, 0), reflectance=0.36)),
                                  observations=(_plane("fwd"),))
        tm = matrix(export(project))
        assert np.abs(tm) == pytest.approx(0.8 * np.eye(2), abs=1e-9)


def test_a_coating_with_physical_data_near_normal_incidence_is_exported():
    """A tabulated coating with R_s = R_p at 0 deg (its s / p split grows with the angle, as a real coating's) met at 0.5 deg
    is continuous there: exported, |t|^2 by bilinear interpolation of the table."""
    from os_optics.contracts import TabulatedCoating

    table = TabulatedCoating(wavelength_nm=(600.0, 700.0), aoi_deg=(0.0, 10.0),
                             reflectance_s=((0.3, 0.32),) * 2, reflectance_p=((0.3, 0.28),) * 2,
                             transmittance_s=((0.7, 0.68),) * 2, transmittance_p=((0.7, 0.72),) * 2)
    tilt = math.radians(0.5)
    project = ProjectSnapshot(components=(_laser(), B.beam_splitter("S", position=(50, 0, 20), normal=(-math.cos(tilt), -math.sin(tilt), 0),
                                                                     coating=table)),
                              observations=(_plane("fwd"),))
    net = export(project)
    ts, tp = 0.7 - 0.02 * 0.05, 0.7 + 0.02 * 0.05
    assert sorted(net.transfer_matrix.column_transmission) == pytest.approx(sorted((ts, tp)), abs=1e-6)


# ---------------------------------------------------------------------------------------------------- m5: classical check


def test_the_classical_check_covers_the_declared_h_and_v_inputs():
    """Each input column is checked against its own Engine trace. The predicted fractions are the Jones
    product HWP(22.5 deg) then an ideal PBS (H transmitted, V reflected): 1/2 each for H and for V input."""
    net = export(_hwp_pbs_project(22.5))
    planes = net.classical_check.columns[0].planes
    assert sorted({(p.input, p.observation_id) for p in planes}) == [(i, o) for i in ("H", "V", "declared") for o in ("R", "T")]
    for p in planes:
        assert p.deviation <= p.allowance
        if p.input in ("H", "V"):
            assert p.predicted_fraction == pytest.approx(0.5, abs=1e-8) and p.engine_fraction == pytest.approx(0.5, abs=1e-8)
    assert "not detected" in type(net.classical_check).__doc__


# ---------------------------------------------------------------------------------------------------- m6: scope gaps


def test_observation_and_source_ids_with_a_colon_are_exported():
    project = ProjectSnapshot(components=(_laser(instance_id="L:1"),), observations=(_plane("port:x", radius_mm=10.0),))
    net = export(project)
    assert net.transfer_matrix.rows == ("out:port:x:0:H", "out:port:x:0:V")
    assert net.transfer_matrix.columns == ("in:L:1#0:H", "in:L:1#0:V")
    assert all(p.predicted_fraction == pytest.approx(1.0, abs=2e-9) for p in net.classical_check.columns[0].planes)


def test_a_beam_truncated_above_the_warning_fraction_is_refused():
    """An aperture passing 91 % of one MZI arm was exported with amplitude sqrt(0.91) as the full Gaussian
    mode (its mode-projected amplitude is 0.91). Above the Engine's 1e-3 warning fraction the export now refuses."""
    w = 0.8
    a = w * math.sqrt(-math.log(0.09) / 2)  # exp(-2 a^2 / w^2) = 0.09
    mzi = _mzi_project(0.0, 532.0, 100.0)
    mzi = mzi.model_copy(update={"components": mzi.components + (B.aperture_stop("AP", position=(50, 100, 20), axis=(1, 0, 0), radius_mm=a),)})
    assert refusal(mzi) == {NetworkErrorCode.MODE_TRUNCATED.value}


def test_a_beam_truncated_below_the_warning_fraction_is_exported_with_its_loss():
    """exp(-2 a^2 / w^2) = 3.4e-4 of the power is cut (w: the Gaussian radius 50 mm from the 0.5 mm waist): below 1e-3,
    exported as loss (amplitude error <= 1.7e-4)."""
    w0, a, z = 0.5, 1.0, 50.0
    w = w0 * math.sqrt(1 + (z * 633e-6 / (math.pi * w0 ** 2)) ** 2)
    project = ProjectSnapshot(components=(_laser(), B.aperture_stop("A", position=(50, 0, 20), axis=(1, 0, 0), radius_mm=a)),
                              observations=(_plane(radius_mm=10.0),))
    net = export(project)
    passed = 1 - math.exp(-2 * a ** 2 / w ** 2)
    assert net.transfer_matrix.column_transmission == pytest.approx((passed, passed), abs=2e-9)


@pytest.mark.parametrize("declared_deg", [0.0, 90.0])
def test_scope_checks_see_the_branch_the_declared_polarization_does_not_feed(declared_deg):
    """A PBS whose reflected (s) arm is clipped by 5 %: with declared H only the V probe sees that arm. Both
    declarations are now refused alike (each input column is checked)."""
    w = 0.5
    a = w * math.sqrt(-math.log(0.05) / 2)
    project = ProjectSnapshot(components=(
        _laser(position=(-50, 0, 20), waist_radius_mm=w, polarization=JonesVector.linear(declared_deg)),
        B.pbs("P", position=(0, 0, 20), normal=(-INV, INV, 0)),
        B.aperture_stop("A", position=(0, 30, 20), axis=(0, 1, 0), radius_mm=a)),
        observations=(_plane("T", x=60.0, radius_mm=10.0),
                      B.observation_plane("R", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=10.0)))
    assert refusal(project) == {NetworkErrorCode.MODE_TRUNCATED.value}


@pytest.mark.parametrize("declared_deg", [0.0, 90.0])
def test_warnings_of_both_probes_reach_the_document(declared_deg):
    """A waveplate crossed at 30 deg in the PBS's reflected arm warns RETARDER_OBLIQUE_INCIDENCE whatever the declared
    polarization (with declared H no power reaches that arm in the declared trace)."""
    tilt = math.radians(30.0)
    project = ProjectSnapshot(components=(
        _laser(position=(-50, 0, 20), polarization=JonesVector.linear(declared_deg)),
        B.pbs("P", position=(0, 0, 20), normal=(-INV, INV, 0)),
        B.half_waveplate("W", position=(0, 30, 20), axis=(math.sin(tilt), math.cos(tilt), 0), fast_axis_angle_deg=0.0,
                         design_wavelength_nm=633.0)),
        observations=(_plane("T", x=60.0, radius_mm=10.0),
                      B.observation_plane("R", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=20.0)))
    net = export(project)
    assert "RETARDER_OBLIQUE_INCIDENCE" in {w.code for w in net.validity.warnings}


@pytest.mark.parametrize("absorbing", [False, True])
def test_an_output_beam_outside_the_acceptance_cone_is_refused_for_that_cause(absorbing):
    """Transparent plane -> was CLASSICAL_LIMIT_MISMATCH, absorbing -> OUTPUT_NOT_TERMINAL."""
    tilt = 0.3  # rad (17.2 deg) against a 5 deg acceptance half-angle
    project = ProjectSnapshot(components=(_laser(),), observations=(B.observation_plane(
        "o", position=(100, 0, 20), normal=(math.cos(tilt), math.sin(tilt), 0), radius_mm=10.0, acceptance_half_angle_deg=5.0,
        absorbing=absorbing),))
    assert refusal(project) == {NetworkErrorCode.OUTSIDE_ACCEPTANCE.value}
    # the problem names the plane, so the export can be repeated without it
    with pytest.raises(NetworkExportError) as info:
        export(project)
    assert [(p.observation_id, p.as_dict()["observation_id"]) for p in info.value.problems] == [("o", "o")]


def _pbs_with_reflected_arm_aperture(aperture: bool) -> ProjectSnapshot:
    """A PBS fed at 45 deg; in its reflected arm an aperture passing half the beam (GAUSSIAN_SEVERE_TRUNCATION, outside the
    Engine's Gaussian model) before plane R; plane T behind the transmitted arm."""
    w = 0.5
    components = [_laser(position=(-50, 0, 20), waist_radius_mm=w, polarization=JonesVector.linear(45.0)),
                  B.pbs("P", position=(0, 0, 20), normal=(-INV, INV, 0))]
    if aperture:
        components.append(B.aperture_stop("A", position=(0, 30, 20), axis=(0, 1, 0), radius_mm=w * math.sqrt(-math.log(0.5) / 2)))
    return ProjectSnapshot(components=tuple(components), observations=(
        _plane("T", x=60.0, radius_mm=10.0), B.observation_plane("R", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=10.0)))


def test_an_out_of_model_finding_refuses_only_an_export_whose_outputs_receive_that_light():
    project = _pbs_with_reflected_arm_aperture(True)
    with pytest.raises(NetworkExportError) as info:
        export(project)  # plane R receives the cut beam
    assert [p.as_dict() for p in info.value.problems] == [{
        "code": "MODE_TRUNCATED", "message": "an aperture cuts the beam severely (GAUSSIAN_SEVERE_TRUNCATION): the beam is no longer "
        "one Gaussian mode; enlarge the aperture or reduce the beam", "location": "surface:A/stop", "observation_id": "R"}]
    # without plane R the cut arm reaches no output: exported, the finding listed, T that of the layout without the aperture
    net = export(project, observation_ids=["T"])
    assert [(w.code, w.location) for w in net.validity.warnings] == [("GAUSSIAN_SEVERE_TRUNCATION", "surface:A/stop")]
    reference = export(_pbs_with_reflected_arm_aperture(False), observation_ids=["T"])
    assert net.transfer_matrix == reference.transfer_matrix and net.paths == reference.paths
    assert abs(entry(net, "out:T:0:H", "in:L#0:H")) == pytest.approx(1.0, abs=1e-8)  # p passes the PBS whole


def _crystal_project(y_mm: float) -> ProjectSnapshot:
    from os_optics.contracts import CircularAperture, EngineeringComponentSnapshot, PlaneSurface, RefractInteraction, UniaxialMaterial
    from os_optics.contracts.materials import ConstantIndexLaw

    calcite = UniaxialMaterial(material_id="calcite", valid_wavelength_nm=(400.0, 1000.0), ordinary=ConstantIndexLaw(refractive_index=1.6557),
                               extraordinary=ConstantIndexLaw(refractive_index=1.4849), optic_axis=(INV, INV, 0.0))
    faces = tuple(PlaneSurface(surface_id=sid, origin_mm=(x, 0.0, 0.0), normal=(n, 0.0, 0.0), aperture=CircularAperture(radius_mm=10.0),
                               medium_positive="air", medium_negative="calcite", interaction=RefractInteraction())
                  for sid, x, n in (("front", -5.0, -1.0), ("back", 5.0, 1.0)))
    crystal = EngineeringComponentSnapshot(instance_id="X", pose=B.Pose3D(position_mm=(50.0, y_mm, 20.0)), surfaces=faces)
    return ProjectSnapshot(components=(_laser(), crystal), materials=(calcite,), observations=(_plane(radius_mm=10.0),))


def test_a_crystal_refuses_the_export_only_on_an_exported_path():
    """The export has no phase model of a uniaxial crystal's split: a crystal the beam crosses refuses (each face, with the
    plane it feeds); the same crystal beside the beam leaves the network that of the layout without it."""
    with pytest.raises(NetworkExportError) as info:
        export(_crystal_project(0.0))
    assert [(p.code.value, p.location, p.observation_id) for p in info.value.problems] == [
        ("OUT_OF_MODEL_SCOPE", "surface:X/back", "o"), ("OUT_OF_MODEL_SCOPE", "surface:X/front", "o")]
    beside = export(_crystal_project(60.0))
    alone = export(ProjectSnapshot(components=(_laser(),), observations=(_plane(radius_mm=10.0),)))
    assert beside.transfer_matrix == alone.transfer_matrix and not beside.validity.warnings


def test_a_beam_at_the_45_degree_basis_switch_is_flagged_and_a_steep_beam_noted():
    """The vertical-beam fallback switched at 1 mrad, where +Z projected is tiny and a near-threshold beam
    turned H / V by 90 deg. The switch is at 45 deg (both references well conditioned) and a mode at it is a warning."""
    at_switch = ProjectSnapshot(components=(_laser(direction=(INV, 0.0, INV)),), observations=(B.observation_plane(
        "o", position=(100 * INV, 0, 20 + 100 * INV), normal=(INV, 0, INV), radius_mm=10.0),))
    net = export(at_switch)
    assert {(w.code, w.location) for w in net.validity.warnings} >= {("POLARIZATION_BASIS_SWITCH", "source:L"),
                                                                     ("POLARIZATION_BASIS_SWITCH", "observation:o")}
    steep = ProjectSnapshot(components=(_laser(direction=(0.0, 0.6, 0.8)),), observations=(B.observation_plane(
        "o", position=(0, 60, 100), normal=(0.0, 0.6, 0.8), radius_mm=10.0),))
    net = export(steep)
    assert any(n.code == "VERTICAL_BEAM_BASIS" for n in net.hints.notes) and not net.validity.warnings
    assert net.output_modes[0].basis_v == pytest.approx((0.0, 0.8, -0.6), abs=1e-12)  # +Y projected, (0, 1, 0) - 0.6 d, normalized


def test_a_network_without_any_output_light_says_so():
    """A beam dump in front of the only output: T has no rows, with a note."""
    project = ProjectSnapshot(components=(_laser(), B.beam_dump("D", position=(50, 0, 20), axis=(-1, 0, 0))),
                              observations=(_plane(radius_mm=10.0),))
    net = export(project)
    assert net.transfer_matrix.rows == () and net.transfer_matrix.column_transmission == (0.0, 0.0)
    assert not net.transfer_matrix.lossless and any(n.code == "NO_OUTPUT_REACHED" for n in net.hints.notes)
