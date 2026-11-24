"""Readers that used a crystal's ordinary index alone: the cavity and paraxial analyses, and the return-light bounds.

* Cavity (probe p6): a symmetric two-mirror cavity (R 500 / 500 mm, L 200 mm, 1064 nm) with a 10 mm calcite plate, optic axis in
  its face, at the centre. Light polarized across the axis (ordinary) and along it (extraordinary) has two families of modes;
  without a stated polarization no single free spectral range is stated. References, independent of the Engine: the group
  index ``n - lambda dn/dlambda`` of the Ghosh Sellmeier laws by a central difference; the eigenmode from NumPy matrices of the
  round trip with the plate's reduced length ``t / n_o`` (ordinary), ``t n_e / n_o^2`` along the optic axis and ``t / n_e``
  across it (extraordinary: the wave-vector surface ``k_c^2 / n_o^2 + k_perp^2 / n_e^2 = k0^2``).
* Paraxial analysis: an element holding a crystal has no single matrix and is refused.
* Return-light bounds (``validation/resonance.py``): a crystal's face reflects over its two indices (a positive crystal's
  extraordinary wave reflects more than its ordinary one), and a cemented face between two cuts of one crystal with equal
  ordinary indices is not a non-reflecting, non-turning face. Reference: the s amplitude reflectance of every pair of indices
  in the crystals' ranges on a dense grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.config import NumericalConfig
from os_optics.local.cavity import cavity_analysis
from os_optics.local.paraxial import ParaxialError, paraxial_analysis
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.scene.lower import lower_scene
from os_optics.validation import resonance as R

from declared_helpers import declared_component
from local_helpers import ALONG_X
from test_birefringent_crystals import _wedge, crystal, displacer as engine_displacer, project as engine_project, source as engine_source
from test_cavity_analysis import LAMBDA_NM, two_mirror_cavity
from test_project_crystals import CALCITE, CALCITE_E, CALCITE_O, displacer, sellmeier

C_MM_PER_S = 299792458.0e3
THICKNESS = 10.0
LENGTH = 200.0
RADIUS = 500.0


def cavity(axis=("1", "0", "0")) -> dict:
    doc = two_mirror_cavity("500", "500", LENGTH, extra=(declared_component("X", "prism", "BD", (95.0, 0.0, 0.0), ALONG_X),))
    doc["optical_models"].append(displacer("BD", axis=axis))
    doc["model_materials"] = [CALCITE]
    return doc


def quantities(analysis) -> dict:
    return {q.quantity_id: q for q in analysis.quantities}


def group_index(terms) -> float:
    h = 1e-3
    return sellmeier(terms, LAMBDA_NM) - LAMBDA_NM * (sellmeier(terms, LAMBDA_NM + h) - sellmeier(terms, LAMBDA_NM - h)) / (2.0 * h)


def eigenmode(reduced: float) -> tuple[float, float, float]:
    """(half trace, round-trip Gouy phase, mode radius at M1) of the cavity with the plate's reduced length ``reduced``."""
    gap = (LENGTH - THICKNESS) / 2.0

    def free(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    mirror = np.array([[1.0, 0.0], [-2.0 / RADIUS, 1.0]])
    one_way = free(gap) @ free(reduced) @ free(gap)
    rt = mirror @ one_way @ mirror @ one_way
    m = (rt[0, 0] + rt[1, 1]) / 2.0
    a, b, d = rt[0, 0], rt[0, 1], rt[1, 1]
    inv_q = complex((d - a) / (2.0 * b), -math.sqrt(1.0 - m * m) / abs(b))
    w = math.sqrt(-LAMBDA_NM * 1e-6 / (math.pi * inv_q.imag))
    theta = math.atan2(math.sqrt(1.0 - m * m) * np.sign(b), m) % (2.0 * math.pi)
    return m, theta, w


INDEX_DEPENDENT = ("round_trip_half_trace", "g1g2", "round_trip_gouy_phase_rad", "mode_radius_mm.M1", "mode_radius_mm.X",
                   "gap_1.waist_radius_mm", "round_trip_group_path_mm", "free_spectral_range_Hz", "transverse_mode_spacing_Hz",
                   "linewidth_fwhm_Hz")


def test_a_cavity_with_a_crystal_states_no_single_free_spectral_range_without_a_polarization() -> None:
    """Probe p6: every quantity read through the crystal's index was KNOWN from n_o alone (725.41 MHz; 731.70 MHz for e light)."""
    q = quantities(cavity_analysis(cavity(), {"components": ["M1", "X", "M2"], "wavelength_nm": LAMBDA_NM}))
    for name in INDEX_DEPENDENT:
        assert q[name].state.value == "OUT_OF_MODEL_SCOPE" and q[name].value is None, name
        assert q[name].reason == "POLARIZATION_NOT_STATED", name
    assert "n_o = 1.64246, n_e = 1.47964" in q["free_spectral_range_Hz"].message
    assert q["cavity_length_mm"].value == LENGTH and q["round_trip_power_fraction"].state.value == "KNOWN"


@pytest.mark.parametrize("polarization", ["ORDINARY", "EXTRAORDINARY"])
def test_each_eigen_polarization_has_its_group_path_and_free_spectral_range(polarization: str) -> None:
    analysis = cavity_analysis(cavity(), {"components": ["M1", "X", "M2"], "wavelength_nm": LAMBDA_NM, "polarization": polarization})
    q = quantities(analysis)
    n_g = group_index(CALCITE_O if polarization == "ORDINARY" else CALCITE_E)
    path = 2.0 * (LENGTH + (n_g - 1.0) * THICKNESS)
    assert q["round_trip_group_path_mm"].value == pytest.approx(path, rel=1e-9)
    assert q["free_spectral_range_Hz"].value == pytest.approx(C_MM_PER_S / path, rel=1e-9)
    # (the reviewer's numbers: 725.41 MHz ordinary, 731.70 MHz extraordinary)
    assert q["free_spectral_range_Hz"].value / 1e6 == pytest.approx(725.41 if polarization == "ORDINARY" else 731.70, abs=0.01)
    assert analysis.settings["polarization"] == polarization


def test_the_eigenmode_of_each_polarization_follows_its_reduced_plate_length() -> None:
    n_o, n_e = sellmeier(CALCITE_O, LAMBDA_NM), sellmeier(CALCITE_E, LAMBDA_NM)
    request = {"components": ["M1", "X", "M2"], "wavelength_nm": LAMBDA_NM}
    for section in ("X", "Y"):
        q = quantities(cavity_analysis(cavity(), {**request, "polarization": "ORDINARY", "axis": section}))
        m, theta, w = eigenmode(THICKNESS / n_o)
        assert q["round_trip_half_trace"].value == pytest.approx(m, rel=1e-9)
        assert q["round_trip_gouy_phase_rad"].value == pytest.approx(theta, rel=1e-9)
        assert q["mode_radius_mm.M1"].value == pytest.approx(w, rel=1e-9)
    found = []
    for section in ("X", "Y"):
        q = quantities(cavity_analysis(cavity(), {**request, "polarization": "EXTRAORDINARY", "axis": section}))
        found.append((q["round_trip_half_trace"].value, q["round_trip_gouy_phase_rad"].value, q["mode_radius_mm.M1"].value))
    along, across = eigenmode(THICKNESS * n_e / (n_o * n_o)), eigenmode(THICKNESS / n_e)
    # the optic axis lies along one section axis: that section diffracts with t n_e / n_o^2, the other with t / n_e
    match = (found[0], found[1]) if found[0][0] == pytest.approx(along[0], rel=1e-9) else (found[1], found[0])
    for got, want in zip(match, (along, across), strict=True):
        assert got == pytest.approx(want, rel=1e-9)


def test_a_crystal_whose_axis_leaves_its_face_has_no_linear_eigenmode() -> None:
    q = quantities(cavity_analysis(cavity(("1", "0", "1")), {"components": ["M1", "X", "M2"], "wavelength_nm": LAMBDA_NM,
                                                             "polarization": "ORDINARY"}))
    for name in INDEX_DEPENDENT:
        assert q[name].state.value == "OUT_OF_MODEL_SCOPE" and q[name].reason == "CRYSTAL_EIGENMODES_UNDEFINED", name


def test_the_paraxial_analysis_refuses_an_element_holding_a_crystal() -> None:
    with pytest.raises(ParaxialError) as caught:
        paraxial_analysis(cavity(), {"components": ["X"], "wavelength_nm": LAMBDA_NM})
    (problem,) = caught.value.problems
    assert problem["code"] == "PARAXIAL_BIREFRINGENT_ELEMENT" and problem["location"] == "component X"


# ------------------------------------------------------------------------------------------------ return-light bounds


def rs_max(n1_range, n2_range, high_rad: float, samples: int = 201) -> float:
    """The largest s amplitude reflectance over a grid of the two index ranges at incidence angles up to ``high_rad`` (1 where
    one of them reflects totally)."""
    best = 0.0
    for n1 in np.linspace(*n1_range, samples):
        for n2 in np.linspace(*n2_range, samples):
            for a in np.linspace(0.0, high_rad, 21):
                s = n1 / n2 * math.sin(a)
                if s >= 1.0:
                    return 1.0
                c, t = math.cos(a), math.sqrt(1.0 - s * s)
                best = max(best, abs((n1 * c - n2 * t) / (n1 * c + n2 * t)))
    return best


@pytest.mark.parametrize("n_o, n_e", [(1.5427, 1.5518), (1.958, 2.168), (1.6557, 1.4849)], ids=["quartz", "yvo4", "calcite"])
def test_a_bare_crystal_face_reflects_up_to_its_larger_fresnel_coefficient(n_o: float, n_e: float) -> None:
    low, high = sorted((n_o, n_e))
    for angle in (0.0, 0.3, 0.6):
        bound = R._interface_range((1.0, 1.0), (low, high), 0.0, angle)
        assert bound >= rs_max((1.0, 1.0), (low, high), angle) - 1e-15
        assert bound == pytest.approx(rs_max((1.0, 1.0), (low, high), angle), rel=1e-12)
    # a positive crystal (quartz, YVO4): its extraordinary wave reflects more than the ordinary index alone gives
    if n_e > n_o:
        assert R._interface_range((1.0, 1.0), (low, high), 0.0, 0.0) > (n_o - 1.0) / (n_o + 1.0)


def test_a_cemented_face_between_two_cuts_of_one_crystal_reflects_and_turns() -> None:
    """The calcite Wollaston of the crystal tests: both prisms have n_o = 1.6557; the diagonal was read as a face between equal
    indices (no reflection, no turn), while an o wave becomes an e wave there."""
    cfg = NumericalConfig()
    materials, p1, p2 = _wedge(math.radians(15.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    snapshot = engine_project(materials, [engine_source(polarization_deg=45.0), p1, p2])
    library = MaterialLibrary(snapshot.materials, cfg)
    scene, _ = lower_scene(snapshot, cfg, library)
    (diagonal,) = [s for s in scene.surfaces if s.surface_id.endswith("diagonal")]
    r, t = R._coefficients(diagonal, R.EITHER, 0.0, 0.2, library, 633.0)
    assert r == pytest.approx(rs_max((1.4849, 1.6557), (1.4849, 1.6557), 0.2, 101), rel=1e-6)
    assert r > 0.05 and t == 1.0
    assert R._index_range(library, "calcite_a", 633.0) == (1.4849, 1.6557)


def test_a_crystal_plate_keeps_finite_return_bounds() -> None:
    """A calcite plate (two parallel faces) in a run with bounds: its faces' factors stay finite (reflectances below 1 over its
    index range), so the run's planes keep their power bounds."""
    material, plate = engine_displacer((0.0, 1.0, 0.0))
    from test_birefringent_crystals import out_plane
    from helpers import run
    result = run(engine_project([material], [engine_source(), plate], out_plane(60.0)))
    assert result.power_budget.conserved
    (obs,) = result.observations
    (power,) = obs.power_by_wavelength
    assert power.power_W > 0.0 and math.isfinite(power.untraced_power_W)


def test_the_return_screen_reads_crystal_plates_and_cemented_faces() -> None:
    """The crystal fixture (a calcite displacer, then a calcite Wollaston): the displacer's two faces keep a finite bound of their
    returns (their walk now bounded with the walk-off), and the Wollaston's diagonal -- equal ordinary indices on both sides --
    turns light, so its entrance and exit faces no longer count as parallel planes with nothing turning light between them
    (that read gave a finite 2.0)."""
    from fixture_scenes import trace_birefringent_crystals
    from os_optics.validation.resonance import ReturnAnalysis

    request = trace_birefringent_crystals()
    cfg = NumericalConfig()
    library = MaterialLibrary(request.project.materials, cfg)
    scene, _ = lower_scene(request.project, cfg, library)
    pairs = {p.surfaces: p for p in ReturnAnalysis(scene, library, {633.0}, lambda: None, 10**9)[633.0].pairs}
    assert math.isfinite(pairs[("D/front", "D/back")].returns)
    assert pairs[("W/entrance", "W/exit")].returns == math.inf
    assert "W/diagonal" in pairs[("W/entrance", "W/exit")].crossed
