"""Coherent N-beam superposition on observation planes: agreement with the two-beam analysis, energy conservation of a
lossless Mach-Zehnder interferometer for any arm difference, cos^2 fringes, and N = 3 phasor sums.

The interferometer's splitters state their phases (constant over wavelength: no group delay), so the two arms' relative
envelope delay is known and a declared coherence length gives their degree of coherence; with a splitter, retarder or
coating whose phase is not stated, the delay -- and the coherent sum -- is not known (``test_undetermined_group_delays.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.builders import laser, mirror_for_reflection, npbs, observation_plane
from os_optics.contracts import (
    ArrivalResult,
    BeamState,
    GaussianBeamState,
    JonesVector,
    ObservationPlaneFrame,
    PolarizationBasis,
    ProjectSnapshot,
    Target,
)
from os_optics.contracts.optics import InterfacePhases
from os_optics.config import NumericalConfig
from os_optics.errors import Status
from os_optics.validation.superposition import irradiance, superpose

from helpers import run

X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
L = 100.0
Q = math.sqrt(0.5)


def _stated(component, transmission_rad: float = 0.0):
    """A splitter with its phases stated, constant over wavelength: r_s = i sqrt(R), r_p = -i sqrt(R) (the default
    lossless splitter's) and t = sqrt(T) exp(i ``transmission_rad``) for s and p."""
    s = component.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2.0,), reflection_p_rad=(3.0 * math.pi / 2.0,),
                             transmission_s_rad=(transmission_rad,), transmission_p_rad=(transmission_rad,))
    return component.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})}),)})


def _mzi(phase_rad: float, coherence_length_mm: float | None = 1000.0):
    """Square Mach-Zehnder (BS1 at the origin, mirrors at (L, 0) and (0, L), BS2 at (L, L)) with a lossless phase shift in
    arm 1: BS1 states the phase of its transmitted light (arm 1) as ``phase_rad`` -- a pure phase, the power and
    polarization unchanged (the light of the unused input port is not traced, so BS1 needs no partner phase)."""
    src = laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(-50.0, 0.0, 0.0), direction=X,
                coherence_length_mm=coherence_length_mm)
    bs1 = _stated(npbs("BS1", reflectance=0.5, position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0)), phase_rad)
    m1 = mirror_for_reflection("M1", position=(L, 0.0, 0.0), d_in=X, d_out=Y)
    m2 = mirror_for_reflection("M2", position=(0.0, L, 0.0), d_in=Y, d_out=X)
    bs2 = _stated(npbs("BS2", reflectance=0.5, position=(L, L, 0.0), normal=(-Q, Q, 0.0)))
    out_x = observation_plane("OUT_X", position=(L + 60.0, L, 0.0), normal=X)
    out_y = observation_plane("OUT_Y", position=(L, L + 60.0, 0.0), normal=Y)
    return ProjectSnapshot(components=(src, bs1, m1, m2, bs2), observations=(out_x, out_y))


def _total(result, oid: str) -> float:
    obs = next(o for o in result.observations if o.observation_id == oid)
    (s,) = obs.superpositions
    assert s.status is Status.PASS
    return s.total_power_W


@pytest.mark.parametrize("phase_rad", [0.0, 0.3, math.pi / 2, 2.0, math.pi])
def test_mach_zehnder_outputs_conserve_energy_for_any_arm_phase(phase_rad):
    result = run(_mzi(phase_rad))
    assert _total(result, "OUT_X") + _total(result, "OUT_Y") == pytest.approx(1e-3, rel=1e-9)


def test_mach_zehnder_port_power_follows_cos_squared_of_the_arm_phase():
    reference = _total(run(_mzi(0.0)), "OUT_X")
    for phase in (0.3, math.pi / 2, 2.0, math.pi):
        p = _total(run(_mzi(phase)), "OUT_X")
        # P(phi) = P0 cos^2((phi + phi0) / 2) with the port's static phase phi0 fixed by P(0)
        phi0 = 2.0 * math.acos(math.sqrt(min(1.0, reference / 1e-3)))
        candidates = [1e-3 * math.cos((phase + s * phi0) / 2.0) ** 2 for s in (1.0, -1.0)]
        assert min(abs(p - c) for c in candidates) < 1e-12


def test_superposition_matches_the_two_beam_collinear_combination():
    result = run(_mzi(0.7))
    obs = next(o for o in result.observations if o.observation_id == "OUT_Y")
    pair = next(i for i in result.interference if i.observation_id == "OUT_Y")
    assert obs.superpositions[0].total_power_W == pytest.approx(pair.combined_power_collinear_W, rel=1e-9)


def test_undeclared_coherence_is_taken_as_monochromatic_with_its_basis():
    # no stated coherence: the line is taken as monochromatic. The coherent sum equals the same design's with a coherence
    # length of 1e12 mm (a degree of coherence of 1 to rounding), carries the stand-in in its entry, and states the range
    # every degree of coherence from 0 to 1 gives (from the incoherent sum to the value)
    stated = run(_mzi(0.7, coherence_length_mm=1e12))
    (s,) = next(o for o in run(_mzi(0.7, coherence_length_mm=None)).observations if o.observation_id == "OUT_Y").superpositions
    middle = 0.5 * (s.total_power_W + s.incoherent_power_W)
    targets = (Target(target_id="t", metric="coherent_power_W", observation_id="OUT_Y", minimum=0.0),
               Target(target_id="tight", metric="coherent_power_W", observation_id="OUT_Y", minimum=middle))
    result = run(_mzi(0.7, coherence_length_mm=None).model_copy(update={"targets": targets}))
    (reference,) = next(o for o in stated.observations if o.observation_id == "OUT_Y").superpositions
    assert s.total_power_W == pytest.approx(reference.total_power_W, rel=1e-12) and "COHERENCE_UNDEFINED" not in s.issue_codes
    assert reference.value_basis is None
    entry = s.value_basis["total_power_W"]
    assert entry.kind.value == "ESTIMATE" and [(a.input.value, a.at, a.value) for a in entry.assumed] == [
        ("FULL_COHERENCE", ("source:L",), 1.0)]
    assert s.total_power_min_W == pytest.approx(min(s.incoherent_power_W, s.total_power_W), rel=1e-12)
    assert s.total_power_max_W == pytest.approx(max(s.incoherent_power_W, s.total_power_W), rel=1e-12)
    met, open_ = result.targets
    assert met.assessment.value == "MET" and met.value == pytest.approx(s.total_power_W, rel=1e-12)
    assert met.value_basis["value"].kind.value == "ESTIMATE"
    # (a minimum inside the range is not decided: the source's own degree of coherence may lie anywhere in it)
    assert s.total_power_min_W < middle < s.total_power_max_W and open_.assessment.value == "INDETERMINATE"
    assert not any(i.code.value == "COHERENCE_UNDEFINED" for i in result.issues)


# ---------------------------------------------------------------------------------- synthetic N-beam arrivals


def _arrival(path_id: str, opl_mm: float, power_W: float = 1e-3, w: float = 0.5) -> ArrivalResult:
    basis = PolarizationBasis(direction=(0.0, 0.0, 1.0), e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0))
    g = GaussianBeamState(valid=True, status=Status.PASS, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), wavelength_nm=500.0, medium_index=1.0,
                          beam_radius_x_mm=w, beam_radius_y_mm=w)
    beam = BeamState(source_id="S", path_id=path_id, spectral_index=0, wavelength_nm=500.0, position_mm=(0.0, 0.0, 0.0),
                     direction=(0.0, 0.0, 1.0), power_W=power_W, geometric_length_mm=opl_mm, optical_path_length_mm=opl_mm,
                     polarization=JonesVector.linear(0.0), basis=basis, gaussian=g)
    return ArrivalResult(beam=beam, uv_mm=(0.0, 0.0), aoi_deg=0.0, coherence_group="g", coherence_length_mm=1e6,
                         group_path_length_mm=opl_mm, group_delay_dispersion_mm2=0.0)  # (in air: the group path is the optical path)


PLANE = ObservationPlaneFrame(origin_mm=(0.0, 0.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0), normal=(0.0, 0.0, 1.0),
                              pose_frame="DESIGN")
LAMBDA_MM = 500e-6


@pytest.mark.parametrize(("phases", "expected_factor"), [((0.0, 0.0, 0.0), 9.0), ((0.0, 1.0 / 3.0, 2.0 / 3.0), 0.0),
                                                          ((0.0, 0.5, 0.0), 1.0)])
def test_three_collinear_beams_add_as_phasors(phases, expected_factor):
    members = [_arrival(f"S#0.{k}", 10.0 + p * LAMBDA_MM) for k, p in enumerate(phases)]
    s = superpose(members, PLANE, NumericalConfig())
    assert s.total_power_W == pytest.approx(expected_factor * 1e-3, abs=1e-12)


def test_irradiance_map_integrates_to_the_superposed_power():
    members = [_arrival("S#0.0", 10.0), _arrival("S#0.1", 10.0 + 0.2 * LAMBDA_MM, power_W=0.5e-3)]
    total = superpose(members, PLANE, NumericalConfig()).total_power_W
    edge, n = 3.0, 401
    axis = np.linspace(-edge, edge, n)
    uu, vv = np.meshgrid(axis, axis)
    values = irradiance(members, PLANE, np.column_stack([uu.ravel(), vv.ravel()]), NumericalConfig())
    integral = float(values.sum()) * (axis[1] - axis[0]) ** 2
    assert integral == pytest.approx(total, rel=1e-6)


def test_tilted_beams_interference_term_is_the_gaussian_fourier_factor():
    theta = 1e-3
    a = _arrival("S#0.0", 10.0)
    tilted = (math.sin(theta), 0.0, math.cos(theta))
    b = _arrival("S#0.1", 10.0)
    b = b.model_copy(update={"beam": b.beam.model_copy(update={"direction": tilted})})
    s = superpose([a, b], PLANE, NumericalConfig())
    dk = 2.0 * math.pi / LAMBDA_MM * math.sin(theta)
    # equal round beams centred at one point, equal phase there: 2 P exp(-dk^2 w^2 / 8), flux factor (1 + cos theta) / 2
    expected = 2.0 * 1e-3 * math.exp(-(dk**2) * 0.5**2 / 8.0) * (1.0 + math.cos(theta)) / 2.0
    assert s.interference_power_W == pytest.approx(expected, rel=1e-6)
