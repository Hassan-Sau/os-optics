"""Native crystal classes: ``splitter.beam_displacer`` (a uniaxial crystal plate) and ``polarizer.glan`` (two air-spaced crystal
prisms, a compound unit), built from a declared calcite and traced through the local Engine path.

References are computed here, independent of the Engine and of the library's own design report: calcite's Sellmeier indices
(G. Ghosh, Opt. Commun. 163, 95 (1999)), the walk-off ``tan(theta_S) = (n_o / n_e)^2 tan(theta)``, the critical angles
``asin(1 / n)`` and the isotropic Fresnel transmittances (normal incidence ``4 n / (1 + n)^2``; s-polarized at an angle).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.contracts import UniaxialMaterial
from os_optics.local import json_api, lower_project, run_optical_project
from os_optics.native_library import NativeComponentError, add_to_project, build_native_model, describe_native_class

from local_helpers import ALONG_X, POWER_W, plane, project, source

WAVELENGTH = 633.0
CALCITE_O = ((0.73358749, 0.96464345, 1.82831454), (0.0, 0.0194325203, 120.0))
CALCITE_E = ((0.35859695, 0.8242783, 0.14429128), (0.0, 0.0106689543, 120.0))
CALCITE = {"ordinary": {"sellmeier": {"b": ["0.73358749", "0.96464345", "1.82831454"], "c_um2": ["0", "0.0194325203", "120"]}},
           "extraordinary": {"sellmeier": {"b": ["0.35859695", "0.8242783", "0.14429128"], "c_um2": ["0", "0.0106689543", "120"]}},
           "valid_band_nm": [400, 2000], "material_id": "CALCITE", "display_name": "Calcite",
           "absorption_coefficient_per_mm": "0"}  # synthetic lossless crystal assumption for power/Snell tests
#: Parameters of every crystal class (shared with test_native_library.py: every class builds; Pyodide parity).
CLASSES_CRYSTAL = {
    "splitter.beam_displacer": {"material": CALCITE, "design_wavelength_nm": "633", "length_mm": "10"},
    "polarizer.glan": {"material": CALCITE, "design_wavelength_nm": "633", "aperture_mm": "10"},
    "polarizer.glan_thompson": {"material": CALCITE, "design_wavelength_nm": "633", "aperture_mm": "10", "gap_angle_deg": "75",
                                "cement": {"constant_index": "1.55", "valid_band_nm": [300, 2500], "material_id": "CEMENT",
                                           "basis": "synthetic lossless cement", "absorption_coefficient_per_mm": "0"}},
    "splitter.wollaston": {"material": CALCITE, "design_wavelength_nm": "633", "aperture_mm": "10", "wedge_angle_deg": "20"},
    "splitter.rochon": {"material": CALCITE, "design_wavelength_nm": "633", "aperture_mm": "10", "wedge_angle_deg": "20"},
}


def sellmeier(terms, wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(*terms)))


N_O, N_E = sellmeier(CALCITE_O, WAVELENGTH), sellmeier(CALCITE_E, WAVELENGTH)


def t_normal(n: float) -> float:
    return 4.0 * n / (1.0 + n) ** 2


def t_s(n1: float, n2: float, cos_i: float) -> float:
    sin_t = n1 / n2 * math.sqrt(1.0 - cos_i * cos_i)
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    return 1.0 - ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2


def place(build, *, polarization_deg: float = 45.0, planes=None) -> dict:
    """The element at the origin, its local +Z along design +X (local +X = design +Y, local +Y = design +Z), a 633 nm source at
    x = -100 polarized at ``polarization_deg`` from design +Y toward +Z, and the plane OUT at x = 60."""

    src = source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH, waist_mm=0.3, polarization_deg=polarization_deg)
    document = project([src], planes or [plane("OUT", (60.0, 0.0, 0.0), radius_mm=40.0)])
    return add_to_project(document, build, instance_id="X", position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


def outputs(result, oid: str = "OUT"):
    obs = next(o for o in result.observations if o.observation_id == oid)
    return obs.arrivals


@pytest.mark.parametrize("class_id", sorted(CLASSES_CRYSTAL))
def test_each_crystal_class_builds_describes_itself_and_enters_a_project(class_id: str) -> None:
    build = build_native_model(class_id, CLASSES_CRYSTAL[class_id])
    description = describe_native_class(class_id)
    assert description["model_kind"] == build.optical_model.model.model_kind
    assert description["references"]
    ProjectDocumentV2.model_validate(place(build))
    (crystal,) = (m for m in build.model_materials if m.is_crystal)
    assert {m.material_id for m in build.model_materials if not m.is_crystal} <= {"CEMENT"}
    reply = json_api.build_native_model(class_id, CLASSES_CRYSTAL[class_id])
    assert reply["ok"] and reply["optical_model"]["model_id"] == build.optical_model.model_id
    assert json_api.build_native_model(class_id, CLASSES_CRYSTAL[class_id]) == reply


def test_the_beam_displacer_separates_the_beams_by_its_walk_off() -> None:
    """45 deg cut, 10 mm: the extraordinary beam (polarized along local X = design Y) leaves displaced by ``L tan(rho)`` along
    local +X (design +Y); the ordinary beam (local Y = design Z) goes straight. Powers at normal incidence."""

    build = build_native_model("splitter.beam_displacer", CLASSES_CRYSTAL["splitter.beam_displacer"])
    theta = math.radians(45.0)
    rho = math.atan((N_O / N_E) ** 2 * math.tan(theta)) - theta
    assert float(build.design["walk_off_deg"]) == pytest.approx(math.degrees(rho), abs=1e-9)
    assert float(build.design["displacement_mm"]) == pytest.approx(10.0 * math.tan(rho), abs=1e-6)
    result = run_optical_project(place(build), catalog=None).result
    assert result.power_budget.conserved
    beams = outputs(result)
    assert len(beams) == 2
    shifted = [a for a in beams if a.beam.position_mm[1] > 1e-9]
    straight = [a for a in beams if abs(a.beam.position_mm[1]) <= 1e-12]
    assert len(shifted) == 1 and len(straight) == 1
    np.testing.assert_allclose(shifted[0].beam.position_mm, (60.0, 10.0 * math.tan(rho), 0.0), atol=1e-12)
    n_theta = 1.0 / math.sqrt(math.cos(theta) ** 2 / N_O**2 + math.sin(theta) ** 2 / N_E**2)
    assert shifted[0].beam.power_W == pytest.approx(0.5 * POWER_W * t_normal(n_theta) ** 2, rel=1e-12)
    assert straight[0].beam.power_W == pytest.approx(0.5 * POWER_W * t_normal(N_O) ** 2, rel=1e-12)


def test_a_displacement_gives_the_crystal_length() -> None:
    build = build_native_model("splitter.beam_displacer", {"material": CALCITE, "design_wavelength_nm": "633", "displacement_mm": "2.7"})
    theta = math.radians(45.0)
    rho = math.atan((N_O / N_E) ** 2 * math.tan(theta)) - theta
    assert float(build.design["length_mm"]) == pytest.approx(math.ceil(2.7 / math.tan(rho) * 1000.0) / 1000.0, abs=1e-12)
    with pytest.raises(NativeComponentError, match="exactly one of length_mm"):
        build_native_model("splitter.beam_displacer", {"material": CALCITE, "design_wavelength_nm": "633"})


def test_the_glan_foucault_passes_the_extraordinary_wave_with_the_exact_fresnel_transmittance() -> None:
    """Axes along local Y (design Z), perpendicular to the plane of incidence at the gap: the design-Z polarized half crosses
    entrance (normal), gap out (s, crystal -> air at A_g), gap in (s, air -> crystal) and exit (normal); the ordinary half is
    totally reflected. The split is exact here: no approximation warning."""

    build = build_native_model("polarizer.glan", {**CLASSES_CRYSTAL["polarizer.glan"], "variant": "GLAN_FOUCAULT"})
    angle = math.radians(float(build.design["gap_angle_deg"]))
    assert math.degrees(math.asin(1.0 / N_O)) < math.degrees(angle) < math.degrees(math.asin(1.0 / N_E))
    run_ = run_optical_project(place(build), catalog=None)
    result = run_.result
    assert result.power_budget.conserved
    (through,) = outputs(result)
    expected = 0.5 * POWER_W * t_normal(N_E) ** 2 * t_s(N_E, 1.0, math.cos(angle)) * t_s(1.0, N_E, math.sqrt(1.0 - (N_E * math.sin(angle)) ** 2))
    assert through.beam.power_W == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(through.beam.direction, (1.0, 0.0, 0.0), atol=1e-12)
    j = through.beam.polarization
    field = complex(*j.ex) * np.array(through.beam.basis.e1) + complex(*j.ey) * np.array(through.beam.basis.e2)
    assert abs(field[2]) / np.linalg.norm(field) == pytest.approx(1.0, abs=1e-12)  # along design Z = local Y
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" not in {i.code for i in result.issues}
    assert any(step.endswith("tir_o") for p in result.paths for step in p.lineage)


def test_the_glan_taylor_passes_light_polarized_along_its_axes() -> None:
    """Axes along local X (design Y): the transmitted beam is the design-Y polarized half; its power is the p-polarized gap
    transmission to within the stated face approximation (warned)."""

    build = build_native_model("polarizer.glan", CLASSES_CRYSTAL["polarizer.glan"])
    request = lower_project(place(build), catalog=None).request
    crystals = sorted(m.material_id for m in request.project.materials if isinstance(m, UniaxialMaterial))
    assert crystals == ["project:CALCITE@X-A", "project:CALCITE@X-B"]
    result = run_optical_project(place(build), catalog=None).result
    assert result.power_budget.conserved
    (through,) = outputs(result)
    angle = math.radians(float(build.design["gap_angle_deg"]))

    def t_p(n1: float, n2: float, cos_i: float) -> float:
        sin_t = n1 / n2 * math.sqrt(1.0 - cos_i * cos_i)
        cos_t = math.sqrt(1.0 - sin_t * sin_t)
        return 1.0 - ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    reference = 0.5 * POWER_W * t_normal(N_E) ** 2 * t_p(N_E, 1.0, math.cos(angle)) * t_p(1.0, N_E, math.sqrt(1.0 - (N_E * math.sin(angle)) ** 2))
    assert through.beam.power_W == pytest.approx(reference, rel=0.05)
    j = through.beam.polarization
    e1, e2 = np.array(through.beam.basis.e1), np.array(through.beam.basis.e2)
    field = complex(*j.ex) * e1 + complex(*j.ey) * e2
    assert abs(field[1]) / np.linalg.norm(field) == pytest.approx(1.0, abs=1e-9)  # along design Y = local X
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" in {i.code for i in result.issues}


def test_a_glan_polarizer_needs_a_negative_crystal_and_a_gap_angle_in_the_window() -> None:
    positive = {**CALCITE, "ordinary": CALCITE["extraordinary"], "extraordinary": CALCITE["ordinary"], "material_id": "POSITIVE"}
    with pytest.raises(NativeComponentError, match="needs a negative crystal"):
        build_native_model("polarizer.glan", {**CLASSES_CRYSTAL["polarizer.glan"], "material": positive})
    with pytest.raises(NativeComponentError, match="between the critical angles"):
        build_native_model("polarizer.glan", {**CLASSES_CRYSTAL["polarizer.glan"], "gap_angle_deg": "30"})


def test_a_crystal_class_refuses_an_isotropic_material() -> None:
    with pytest.raises(NativeComponentError, match="a crystal is a material document with birefringence"):
        build_native_model("splitter.beam_displacer", {"material": {"constant_index": "1.5", "valid_band_nm": [400, 700]},
                                                      "design_wavelength_nm": "633", "length_mm": "10"})
