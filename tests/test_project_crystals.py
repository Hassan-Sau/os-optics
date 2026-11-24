"""Uniaxial crystals declared in a Project: a crystal material (``model_materials`` with ``birefringence``) and the optic axis of
the element made of it (``optic_axes``), lowered to the Engine's crystal model.

References are independent of the Engine: calcite's ordinary and extraordinary indices from its Sellmeier formulas (G. Ghosh,
Opt. Commun. 163, 95 (1999)) evaluated here, the walk-off angle ``tan(theta_S) = (n_o / n_e)^2 tan(theta)`` of the
extraordinary ray (``theta`` the wave normal's angle to the optic axis, ``theta_S`` the ray's), the extraordinary phase index
``1 / n(theta)^2 = cos^2(theta) / n_o^2 + sin^2(theta) / n_e^2`` and the isotropic Fresnel transmittance at normal incidence
``4 n / (1 + n)^2``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.contracts import UniaxialMaterial
from os_optics.local import lower_project, run_optical_project
from os_optics.local.errors import LocalRunError

from declared_helpers import declared_component, with_models
from local_helpers import ALONG_X, POWER_W, plane, project, source

WAVELENGTH = 780.0
LENGTH = 10.0
CALCITE_O = ((0.73358749, 0.96464345, 1.82831454), (0.0, 0.0194325203, 120.0))
CALCITE_E = ((0.35859695, 0.8242783, 0.14429128), (0.0, 0.0106689543, 120.0))
CALCITE = {
    "material_id": "CALCITE",
    "display_name": "Calcite",
    "dispersion": {"kind": "SELLMEIER", "b": ["0.73358749", "0.96464345", "1.82831454"], "c_um2": ["0", "0.0194325203", "120"]},
    "valid_band_nm": {"min_nm": "400", "max_nm": "2000", "status": "SPECIFIED"},
    "absorption_coefficient_per_mm": "0",  # Ideal test-only calcite: measure walk-off/Fresnel, not bulk absorption.
    "birefringence": {"kind": "UNIAXIAL", "extraordinary": {"kind": "SELLMEIER", "b": ["0.35859695", "0.8242783", "0.14429128"],
                                                            "c_um2": ["0", "0.0106689543", "120"]}},
}


def sellmeier(terms, wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(*terms)))


N_O, N_E = sellmeier(CALCITE_O, WAVELENGTH), sellmeier(CALCITE_E, WAVELENGTH)


def fresnel_t(n: float) -> float:
    return 4.0 * n / (1.0 + n) ** 2


def displacer(model_id: str = "BD", axis=("1", "0", "1"), length: float = LENGTH) -> dict:
    """A calcite plate (SURFACE_SEQUENCE, 10 x 10 mm faces normal to local Z) with the optic axis ``axis`` in its frame."""

    face = {"clear_half_widths_mm": ["4.5", "4.5"], "traced": "TRANSMITTED", "status": "SPECIFIED"}
    return {
        "model_id": model_id, "display_name": "Calcite beam displacer",
        "model": {"model_kind": "SURFACE_SEQUENCE", "fidelity": "SURFACE_PRESCRIPTION",
                  "surfaces": [{"surface_id": "S1", "vertex_z_mm": "0", **face}, {"surface_id": "S2", "vertex_z_mm": f"{length:g}", **face}],
                  "media": ["CALCITE"], "edge_half_widths_mm": ["5", "5"],
                  "valid_band_nm": {"min_nm": "400", "max_nm": "2000", "status": "SPECIFIED"},
                  "approximations": ["NOMINAL_PRESCRIPTION", "IDEAL_SURFACE_FIGURE", "GHOST_REFLECTIONS_NOT_TRACED", "EDGE_AS_OPAQUE_RIM"],
                  "status": "SPECIFIED"},
        "optic_axes": [{"material_id": "CALCITE", "direction": list(axis)}],
    }


def scene(src: dict, *models_at: tuple[str, dict, float], observations=None) -> dict:
    """``src`` along design +X from x = -200; each declared model at its x (element local +Z = design +X, local +X = design +Y)."""

    components = [src] + [declared_component(f"X{k}", "prism", model["model_id"], position=(x, 0.0, 0.0), quaternion=ALONG_X)
                          for k, (_, model, x) in enumerate(models_at)]
    document = project(components, observations or [plane("OUT", (100.0, 0.0, 0.0), radius_mm=40.0)])
    unique = {model["model_id"]: model for _, model, _ in models_at}
    return with_models(document, list(unique.values()), [CALCITE])


def partial(dop: float, angle_deg: float = 0.0) -> dict:
    laser = source(wavelength_nm=WAVELENGTH, waist_mm=0.3)
    laser["source_model"]["polarization"] = {"kind": "PARTIAL", "degree_of_polarization": dop,
                                             "polarized": {"kind": "LINEAR", "angle_deg": angle_deg}}
    return laser


def by_wave(result, oid: str = "OUT") -> dict:
    """Arrivals by the crystal wave of their path (``o`` / ``e``), each a list (a partial source has two components per wave)."""

    obs = next(o for o in result.observations if o.observation_id == oid)
    found: dict[str, list] = {}
    for arrival in obs.arrivals:
        waves = "".join(step.rsplit("_", 1)[-1] for step in arrival.lineage if step.endswith(("_o", "_e")))
        found.setdefault(waves, []).append(arrival)
    return found


def test_the_crystal_lowers_to_the_engine_crystal_model_with_the_element_axis() -> None:
    request = lower_project(scene(source(wavelength_nm=WAVELENGTH), ("BD", displacer(), 0.0)), catalog=None).request
    (crystal,) = [m for m in request.project.materials if isinstance(m, UniaxialMaterial)]
    assert crystal.material_id == "project:CALCITE@X0"
    assert crystal.optic_axis_component == "X0"
    assert crystal.optic_axis == pytest.approx((math.sqrt(0.5), 0.0, math.sqrt(0.5)), abs=1e-16)
    assert crystal.ordinary.b == CALCITE_O[0] and crystal.ordinary.c_um2 == CALCITE_O[1]
    assert crystal.extraordinary.b == CALCITE_E[0] and crystal.extraordinary.c_um2 == CALCITE_E[1]
    assert crystal.valid_wavelength_nm == (400.0, 2000.0)
    component = next(c for c in request.project.components if c.instance_id == "X0")
    assert {(s.medium_negative, s.medium_positive) for s in component.surfaces} == {("air", "project:CALCITE@X0"),
                                                                                   ("project:CALCITE@X0", "air")}


@pytest.mark.parametrize("psi_deg", [0.0, 30.0, 90.0])
def test_a_calcite_beam_displacer_declared_in_a_project_splits_two_beams_with_the_walk_off_and_power(psi_deg: float) -> None:
    """Optic axis at 45 deg to the beam in the element's X-Z plane: the extraordinary beam (polarized along the element's local X,
    design Y) leaves displaced by ``L tan(rho)`` away from the axis (design -Y), parallel to the input; the ordinary beam goes
    straight. Powers at normal incidence: ``P_e = cos^2(psi) T(n(theta))^2``, ``P_o = sin^2(psi) T(n_o)^2``."""

    result = run_optical_project(scene(source(wavelength_nm=WAVELENGTH, waist_mm=0.3, polarization_deg=psi_deg), ("BD", displacer(), 0.0)),
                                 catalog=None).result
    assert result.power_budget.conserved
    theta = math.radians(45.0)
    rho = math.atan((N_O / N_E) ** 2 * math.tan(theta)) - theta
    n_theta = 1.0 / math.sqrt(math.cos(theta) ** 2 / N_O**2 + math.sin(theta) ** 2 / N_E**2)
    psi = math.radians(psi_deg)
    expected = {"e": math.cos(psi) ** 2 * fresnel_t(n_theta) ** 2, "o": math.sin(psi) ** 2 * fresnel_t(N_O) ** 2}
    found = by_wave(result)
    assert set(found) == {wave for wave, share in expected.items() if share > 1e-20}
    for wave, arrivals in found.items():
        (arrival,) = arrivals
        assert arrival.beam.power_W == pytest.approx(POWER_W * expected[wave], rel=1e-12)
        np.testing.assert_allclose(arrival.beam.direction, (1.0, 0.0, 0.0), atol=1e-14)
        shift = -LENGTH * math.tan(rho) if wave == "e" else 0.0
        np.testing.assert_allclose(arrival.beam.position_mm, (100.0, shift, 0.0), atol=1e-12)


def test_a_partially_polarized_source_reports_its_degree_before_and_behind_the_displacer() -> None:
    """P = 0.4, the polarized part along the extraordinary direction: before the crystal the route's degree of polarization is
    0.4; behind it each beam is fully polarized (degree 1) and carries ``(1 +- P) / 2`` of the power times its two faces'
    transmittance."""

    dop = 0.4
    observations = [plane("BEFORE", (-50.0, 0.0, 0.0), radius_mm=40.0), plane("OUT", (100.0, 0.0, 0.0), radius_mm=40.0)]
    result = run_optical_project(scene(partial(dop), ("BD", displacer(), 0.0), observations=observations), catalog=None).result
    before = next(o for o in result.observations if o.observation_id == "BEFORE")
    (mixture,) = before.polarization_mixtures
    assert mixture.degree_of_polarization == pytest.approx(dop, rel=1e-12)
    out = next(o for o in result.observations if o.observation_id == "OUT")
    mixtures = {m.path_ids[0]: m for m in out.polarization_mixtures}
    assert len(mixtures) == 2
    theta = math.radians(45.0)
    n_theta = 1.0 / math.sqrt(math.cos(theta) ** 2 / N_O**2 + math.sin(theta) ** 2 / N_E**2)
    found = by_wave(result)
    power = {wave: sum(a.beam.power_W for a in arrivals) for wave, arrivals in found.items()}
    assert power["e"] == pytest.approx(POWER_W * (1 + dop) / 2 * fresnel_t(n_theta) ** 2, rel=1e-12)
    assert power["o"] == pytest.approx(POWER_W * (1 - dop) / 2 * fresnel_t(N_O) ** 2, rel=1e-12)
    for m in mixtures.values():
        assert m.quantity_state.value == "KNOWN"
        assert m.degree_of_polarization == pytest.approx(1.0, abs=1e-12)


def test_one_crystal_cut_two_ways_is_two_engine_materials() -> None:
    """The same calcite in two elements with different axes: each element has its own crystal, axis in its own frame."""

    turned = displacer("BD90", axis=("0", "1", "1"))
    request = lower_project(scene(source(wavelength_nm=WAVELENGTH), ("BD", displacer(), 0.0), ("BD90", turned, 30.0)), catalog=None).request
    crystals = {m.material_id: m for m in request.project.materials if isinstance(m, UniaxialMaterial)}
    assert set(crystals) == {"project:CALCITE@X0", "project:CALCITE@X1"}
    assert crystals["project:CALCITE@X1"].optic_axis == pytest.approx((0.0, math.sqrt(0.5), math.sqrt(0.5)), abs=1e-16)
    assert crystals["project:CALCITE@X1"].optic_axis_component == "X1"


def test_two_displacers_turned_90_degrees_split_into_four_beams() -> None:
    """The second displacer's axis lies in the element's Y-Z plane: each beam of the first meets it as one wave (the first's
    extraordinary beam is the second's ordinary one and the reverse), so two beams leave, displaced along both axes."""

    result = run_optical_project(scene(source(wavelength_nm=WAVELENGTH, waist_mm=0.3, polarization_deg=45.0), ("BD", displacer(), 0.0),
                                       ("BD90", displacer("BD90", axis=("0", "1", "1")), 30.0)), catalog=None).result
    assert result.power_budget.conserved
    theta = math.radians(45.0)
    shift = LENGTH * math.tan(math.atan((N_O / N_E) ** 2 * math.tan(theta)) - theta)
    found = by_wave(result)
    assert set(found) == {"eo", "oe"}
    np.testing.assert_allclose(found["eo"][0].beam.position_mm, (100.0, -shift, 0.0), atol=1e-12)
    # local Y of the element is design +Z: the second displacer moves its extraordinary beam to design -Z
    np.testing.assert_allclose(found["oe"][0].beam.position_mm, (100.0, 0.0, -shift), atol=1e-12)


def test_a_crystal_element_is_drawn_in_the_optical_view() -> None:
    from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
    from os_optics.optical_view import build_optical_simulation_scene

    document = scene(source(wavelength_nm=WAVELENGTH), ("BD", displacer(), 0.0))
    lowered = lower_project(document, catalog=None)
    view = build_optical_simulation_scene(project=ProjectDocumentV2.model_validate(document), catalog=None, engine_request=lowered.request)
    (element,) = [e for e in view.elements if e.element_id == "X0"]
    assert element.optical_model_id == "BD"
    (medium,) = [m for m in view.display_media if m.medium_id == "project:CALCITE@X0"]
    assert medium.refractive_index == pytest.approx(sellmeier(CALCITE_O, 587.5618), abs=5e-5)  # drawn with its ordinary index (4 decimals)


def test_a_curved_crystal_face_is_refused_by_the_project_contract() -> None:
    model = displacer()
    model["model"]["surfaces"][1]["radius_mm"] = "-40"
    with pytest.raises(LocalRunError, match="surface S2 bounds the crystal CALCITE and is curved"):
        lower_project(scene(source(wavelength_nm=WAVELENGTH), ("BD", model, 0.0)), catalog=None)


@pytest.mark.parametrize("with_targets", [True, False])
def test_an_observation_plane_inside_a_declared_crystal_is_refused_by_project_check_and_run(tmp_path, capsys, with_targets) -> None:
    # A declared crystal is an Engine material per element that cuts it (project:CALCITE@X0), and light is traced in a crystal
    # only between its own faces. A plane whose medium is the crystal used to pass the check and fail in the Engine
    # (UNKNOWN_MATERIAL); it is refused before the Engine, with the code the Engine gives a plane in an Engine-request crystal.
    import json

    from os_optics.cli import main

    document = scene(source(wavelength_nm=WAVELENGTH), ("BD", displacer(), 0.0))
    document["observations"][0]["medium"] = "CALCITE"
    if not with_targets:
        document["targets"] = []
    expected = [{"code": "BIREFRINGENT_MEDIUM_UNSUPPORTED",
                 "message": "observation OUT lies inside the uniaxial crystal CALCITE: light is observed outside crystals; place the "
                            "observation plane outside the crystal and give it the medium there (for example air)",
                 "location": "observation OUT", "details": {"material_id": "CALCITE"}}]
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    for command in ("check", "run"):
        assert main(["project", command, str(path)]) == 65
        reply = json.loads(capsys.readouterr().out)
        assert reply["status"] == "ERROR" and reply["problems"] == expected, reply
    with pytest.raises(LocalRunError) as info:
        lower_project(document, catalog=None)
    assert [p.as_dict() for p in info.value.problems] == expected
    # the same plane in air, behind the crystal, is accepted
    document["observations"][0]["medium"] = "air"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "check", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ACCEPTED"


def test_a_project_without_crystals_lowers_as_before() -> None:
    plain = copy.deepcopy(CALCITE)
    del plain["birefringence"]
    model = displacer()
    del model["optic_axes"]
    document = scene(source(wavelength_nm=WAVELENGTH), ("BD", model, 0.0))
    document["model_materials"] = [plain]
    request = lower_project(document, catalog=None).request
    assert [m.material_id for m in request.project.materials] == ["air", "project:CALCITE"]  # (with the environment's air)
