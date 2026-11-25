"""Joined members of a COMPOUND unit (``CompoundModelV1.joints``): Wollaston, Rochon and Glan-Thompson prisms declared in a
Project and traced through the local Engine path.

References are computed here, independent of the Engine and of the native library's design report: calcite's Sellmeier
indices (G. Ghosh, Opt. Commun. 163, 95 (1999)), the vector Snell law at the joint and the exit face (each wave there travels
perpendicular to its crystal's optic axis or along it, so its index does not depend on its direction), and the isotropic
s-polarized Fresnel transmittances for the Glan-Thompson, whose optic axes are perpendicular to the plane of incidence.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.local import LocalRunError, lower_project, run_optical_project
from os_optics.local.compile import ProjectResolver, compile_declared
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.native_library import build_native_model

from local_helpers import POWER_W
from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index
from test_native_crystals import CALCITE, CLASSES_CRYSTAL, N_E, N_O, outputs, place, t_normal, t_s

THETA_DEG = 20.0
#: The native library designs for air at 20 C and 101325 Pa: its design report states angles in that air, computed with
#: the indices relative to it; the runs here are in vacuum (``local_helpers.project``), with the absolute indices.
N_AIR = air_index(633.0)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.dot(a, b))))


def refract(d: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Vector Snell law (``normal`` against the incident ray or not: oriented here)."""

    n = normal if np.dot(d, normal) < 0 else -normal
    cos_i = -float(np.dot(d, n))
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    return eta * d + (eta * cos_i - math.sqrt(k)) * n


def exit_direction(n_a: float, n_b: float) -> np.ndarray:
    """The axial beam (local +Z) through a split prism (joint normal ``(-sin theta, 0, cos theta)``, exit face normal +Z), with
    index ``n_a`` in prism A and ``n_b`` in prism B, as a design-frame direction (local Z = design X, local X = design Y)."""

    theta = math.radians(THETA_DEG)
    inside = refract(np.array([0.0, 0.0, 1.0]), np.array([-math.sin(theta), 0.0, math.cos(theta)]), n_a, n_b)
    out = refract(inside, np.array([0.0, 0.0, 1.0]), n_b, 1.0)
    out = out / np.linalg.norm(out)
    return np.array([out[2], out[0], out[1]])


def polarized_along(arrival) -> np.ndarray:
    j = arrival.beam.polarization
    field = complex(*j.ex) * np.array(arrival.beam.basis.e1) + complex(*j.ey) * np.array(arrival.beam.basis.e2)
    return np.abs(field) / np.linalg.norm(field)


def test_a_wollaston_prism_sends_the_two_polarizations_to_either_side_by_snells_law() -> None:
    """X-polarized light (local X = design Y) is extraordinary in prism A (axis X) and ordinary in prism B (axis Y); Y-polarized
    light (design Z) the other way round."""

    build = build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])
    assert [j.model_dump() for j in build.optical_model.model.joints] == [
        {"member_a": "A", "surface_a": "S2", "member_b": "B", "surface_b": "S1", "cement_material_id": None}]
    result = run_optical_project(place(build), catalog=None).result
    assert result.power_budget.conserved
    beams = outputs(result)
    assert len(beams) == 2
    along_y = next(a for a in beams if polarized_along(a)[2] < 1e-9)  # (in the design X-Y plane, across its direction)
    along_z = next(a for a in beams if polarized_along(a)[2] > 0.999999)
    np.testing.assert_allclose(along_y.beam.direction, exit_direction(N_E, N_O), atol=1e-12)
    np.testing.assert_allclose(along_z.beam.direction, exit_direction(N_O, N_E), atol=1e-12)
    separation = math.degrees(math.acos(float(np.dot(along_y.beam.direction, along_z.beam.direction))))
    design_separation = _angle_deg(exit_direction(N_E / N_AIR, N_O / N_AIR), exit_direction(N_O / N_AIR, N_E / N_AIR))
    assert design_separation == pytest.approx(float(build.design["separation_deg"]), abs=1e-8)
    assert separation == pytest.approx(design_separation, rel=1e-3)
    assert separation == pytest.approx(2.0 * math.degrees(math.asin((N_O - N_E) * math.tan(math.radians(THETA_DEG)))), rel=0.01)
    # each half of the 45-degree input, within the face approximation the run states (prism A's axis lies in the plane of incidence)
    for arrival in beams:
        assert arrival.beam.power_W == pytest.approx(0.5 * POWER_W * 0.9, rel=0.05)
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" in {i.code for i in result.issues}


def test_a_rochon_prism_passes_one_polarization_undeviated() -> None:
    build = build_native_model("splitter.rochon", CLASSES_CRYSTAL["splitter.rochon"])
    result = run_optical_project(place(build), catalog=None).result
    assert result.power_budget.conserved
    beams = outputs(result)
    along_y = next(a for a in beams if polarized_along(a)[2] < 1e-9)  # (in the design X-Y plane, across its direction)
    along_z = next(a for a in beams if polarized_along(a)[2] > 0.999999)
    np.testing.assert_allclose(along_y.beam.direction, (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(along_z.beam.direction, exit_direction(N_O, N_E), atol=1e-12)
    deviation = math.degrees(math.acos(float(along_z.beam.direction[0])))
    design_deviation = math.degrees(math.acos(float(exit_direction(N_O / N_AIR, N_E / N_AIR)[0])))
    assert design_deviation == pytest.approx(abs(float(build.design["exit_angle_y_polarized_deg"])), abs=1e-8)
    assert deviation == pytest.approx(design_deviation, rel=1e-3)


def test_a_glan_thompson_passes_the_extraordinary_wave_with_the_exact_fresnel_transmittance() -> None:
    """Axes along local Y (design Z), perpendicular to the plane of incidence at the cement layer: the design-Z polarized half
    crosses the entrance (normal), the layer's two faces (s, calcite -> cement -> calcite at the gap angle) and the exit
    (normal); the ordinary half is totally reflected at the layer. Exact: no approximation warning."""

    parameters = CLASSES_CRYSTAL["polarizer.glan_thompson"]
    build = build_native_model("polarizer.glan_thompson", parameters)
    n_c = 1.55
    angle = math.radians(75.0)
    assert math.degrees(math.asin(n_c / N_O)) == pytest.approx(float(build.design["critical_angle_ordinary_deg"]), abs=1e-8)
    assert build.optical_model.model.joints[0].cement_material_id == "CEMENT"
    request = lower_project(place(build), catalog=None).request
    assert "project:CEMENT" in {m.material_id for m in request.project.materials}
    result = run_optical_project(place(build), catalog=None).result
    assert result.power_budget.conserved
    through = next(a for a in outputs(result) if abs(a.beam.direction[0] - 1.0) < 1e-12)
    cos_c = math.sqrt(1.0 - (N_E * math.sin(angle) / n_c) ** 2)
    expected = 0.5 * POWER_W * t_normal(N_E) ** 2 * t_s(N_E, n_c, math.cos(angle)) * t_s(n_c, N_E, cos_c)
    assert through.beam.power_W == pytest.approx(expected, rel=1e-9)
    assert polarized_along(through)[2] == pytest.approx(1.0, abs=1e-12)
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" not in {i.code for i in result.issues}
    assert any(step.endswith("tir_o") for p in result.paths for step in p.lineage)


def test_an_optical_contact_is_one_interface_between_the_two_crystals() -> None:
    """The unit's Engine surfaces: prism A's joint face carries prism B's crystal on its front side; prism B's joint face is gone."""

    build = build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])
    project = ProjectDocumentV2.model_validate(place(build))
    unit = project.optical_model(build.optical_model.model_id)
    element = compile_declared("X", unit, ProjectResolver(project))
    ids = [s.surface_id for s in element.surfaces]
    assert ids == ["A-S1", "A-S2", "B-S2"]
    joint = element.surfaces[1]
    assert (joint.medium_negative, joint.medium_positive) == ("project:CALCITE@X-A", "project:CALCITE@X-B")
    assert [e.surface_id for e in element.envelopes] == ids
    assert [s.surface_id for s in element.members[1].surfaces] == ["B-S1", "B-S2"]  # the member keeps its own body


def _misplaced(shift_mm: float, *, cement: bool = False, turn: bool = False) -> dict:
    build = build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])
    document = copy.deepcopy(place(build))
    unit = next(m for m in document["optical_models"] if m["model"]["model_kind"] == "COMPOUND")
    member = unit["model"]["members"][1]
    member["pose"]["position_mm"][2] += shift_mm
    box = unit["model"]["envelope"]
    box["half_extents_mm"][2] = repr(float(box["half_extents_mm"][2]) + shift_mm)
    if turn:
        member["pose"]["quaternion_wxyz"] = [math.cos(0.005), 0.0, math.sin(0.005), 0.0]
    if cement:
        unit["model"]["joints"][0]["cement_material_id"] = "CEMENT"
        document["model_materials"].append({"material_id": "CEMENT", "display_name": "Test cement", "basis": "test-only lossless cement",
                                            "absorption_coefficient_per_mm": "0", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.55"},
                                            "valid_band_nm": {"min_nm": "300", "max_nm": "2500", "status": "SPECIFIED"}})
    return document


def test_a_joint_whose_faces_do_not_meet_is_refused() -> None:
    with pytest.raises(LocalRunError, match="an optical contact's faces coincide"):
        lower_project(_misplaced(0.01), catalog=None)
    with pytest.raises(LocalRunError, match="the faces do not face each other"):
        lower_project(_misplaced(0.0, turn=True), catalog=None)
    with pytest.raises(LocalRunError, match="the cement layer between the faces is 0 mm thick"):
        lower_project(_misplaced(0.0, cement=True), catalog=None)
    with pytest.raises(LocalRunError, match="the cement layer between the faces is .* mm thick"):
        lower_project(_misplaced(5.0, cement=True), catalog=None)


def test_a_cement_layer_of_a_declared_unit_is_traced_between_its_faces() -> None:
    """The same Wollaston with prism B moved by 0.02 mm along Z and a cement layer: the layer (index 1.55, between parallel
    faces) leaves the exit directions unchanged (n sin is kept across it)."""

    document = _misplaced(0.02, cement=True)
    result = run_optical_project(document, catalog=None).result
    assert result.power_budget.conserved
    beams = outputs(result)
    along_y = next(a for a in beams if polarized_along(a)[2] < 1e-9)  # (in the design X-Y plane, across its direction)
    np.testing.assert_allclose(along_y.beam.direction, exit_direction(N_E, N_O), atol=1e-12)
    assert any("project:CEMENT" == s.medium_id for p in result.paths for s in p.segments)



@pytest.mark.parametrize("gradient", [{"gradient": {"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": "0.3"}},
                                      {"gradient": {"kind": "RADIAL_QUADRATIC", "gradient_constant_per_mm": "3"}},
                                      {"axial_gradient": {"kind": "AXIAL_POLYNOMIAL", "index_terms": ["0.05"]}}],
                         ids=["radial", "radial_strong", "axial"])
def test_a_cement_of_a_gradient_index_material_is_refused(gradient: dict) -> None:
    """A cement layer is traced as one homogeneous medium: a cement with a gradient index (radial or axial) is refused by
    the contract (PROJECT_INVALID), never traced as if uniform (before, every such run gave the homogeneous cement's
    result hash with no issue)."""

    document = _misplaced(0.02, cement=True)
    next(m for m in document["model_materials"] if m["material_id"] == "CEMENT").update(gradient)
    with pytest.raises(LocalRunError, match="cement CEMENT has a gradient index; a cement layer is homogeneous") as refused:
        run_optical_project(document, catalog=None)
    assert refused.value.as_dict()["problems"][0]["code"] == "PROJECT_INVALID"


def test_split_prisms_refuse_an_isotropic_or_unsplitting_material() -> None:
    from os_optics.native_library import NativeComponentError

    isotropic = {"constant_index": "1.5", "valid_band_nm": [400, 2000]}
    with pytest.raises(NativeComponentError):
        build_native_model("splitter.wollaston", {**CLASSES_CRYSTAL["splitter.wollaston"], "material": isotropic})
    flat = {**CALCITE, "extraordinary": CALCITE["ordinary"], "material_id": "FLAT"}
    with pytest.raises(NativeComponentError, match="no birefringence"):
        build_native_model("splitter.rochon", {**CLASSES_CRYSTAL["splitter.rochon"], "material": flat})
    with pytest.raises(NativeComponentError, match="state gap_angle_deg"):
        build_native_model("polarizer.glan_thompson", {k: v for k, v in CLASSES_CRYSTAL["polarizer.glan_thompson"].items()
                                                       if k != "gap_angle_deg"})


# ------------------------------------------------------------------------- the run's model scope behind a joint (TR-1)
# Behind a joint the light is in the next member's crystal (an optical contact) or in the cement layer, bounded by that
# body's faces: a beam that leaves through them is inside the model's scope; one that leaves a body through its side is
# not (RAY_LEFT_ELEMENT_THROUGH_EDGE). Expected here from the geometry: on the axis every beam crosses the joint and leaves
# through the exit face; with prism B slid 8 mm along the joint plane (the faces still coincide) a beam 3 mm off the axis
# crosses the joint where prism B's exit face is not.


def _cemented() -> dict:
    return _misplaced(0.02, cement=True)


@pytest.mark.parametrize("label", ["splitter.wollaston", "splitter.rochon", "polarizer.glan_thompson", "polarizer.glan",
                                   "splitter.beam_displacer", "cemented wollaston"])
def test_a_joined_prism_run_is_within_the_model_scope(label: str) -> None:
    document = _cemented() if label == "cemented wollaston" else place(build_native_model(label, CLASSES_CRYSTAL[label]))
    run = run_optical_project(document, catalog=None)
    assert run.model_scope_issues == ()
    assert run.engine_status == "WARNING" and run.status == "WARNING"
    if label in ("splitter.wollaston", "splitter.rochon", "polarizer.glan_thompson", "cemented wollaston"):
        # (the path does cross the joint into the next body: the segment after A-S2 is in B's crystal or in the cement)
        media = {s.medium_id for p in run.result.paths for s in p.segments}
        assert "project:CALCITE@X-B" in media


def test_light_crossing_the_joint_backwards_is_within_the_model_scope() -> None:
    """The Wollaston traced from its exit face: prism B, the joint, prism A."""

    from local_helpers import AGAINST_X, plane, source

    document = place(build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"]), planes=[plane("OUT", (-60.0, 0.0, 0.0), AGAINST_X, radius_mm=40.0)])
    document["components"][0] = source(position=(100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.3, quaternion=AGAINST_X, polarization_deg=45.0)
    run = run_optical_project(document, catalog=None)
    assert outputs(run.result)
    lineage = next(p for p in run.result.paths if p.parent_path_id is None).lineage
    assert [step.split(":")[0] for step in lineage[:3]] == ["X/B-S2", "X/A-S2", "X/A-S1"]
    assert run.model_scope_issues == ()


def _slid(cement: bool) -> dict:
    """Prism B slid 8 mm along the joint plane (direction (cos 20, 0, sin 20) in the unit frame), the source 3 mm off the axis."""

    document = _cemented() if cement else copy.deepcopy(place(build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])))
    unit = next(m for m in document["optical_models"] if m["model"]["model_kind"] == "COMPOUND")
    member = unit["model"]["members"][1]
    theta = math.radians(THETA_DEG)
    member["pose"]["position_mm"][0] -= 8.0 * math.cos(theta)
    member["pose"]["position_mm"][2] -= 8.0 * math.sin(theta)
    box = unit["model"]["envelope"]
    box["half_extents_mm"] = ["13", "5", f"{float(box['half_extents_mm'][2]) + 8.0:.6f}".rstrip("0")]
    document["components"][0]["nominal_pose"]["position_mm"][1] = 3.0
    return document


def test_light_leaving_the_next_prism_through_its_side_is_reported() -> None:
    """Optical contact: behind the joint the beam is in prism B's crystal and reaches no face of prism B."""

    run = run_optical_project(_slid(cement=False), catalog=None)
    path = next(p for p in run.result.paths if p.parent_path_id is None)
    assert [s.medium_id for s in path.segments][2:] == ["project:CALCITE@X-B", "project:CALCITE@X-B"]
    (issue,) = [i for i in run.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE"]
    assert (issue.instance_id, issue.member_id, issue.surface_id) == ("X", "B", "B-S1")
    assert "reaches observation/OUT, then leaves the scene" in issue.message
    assert run.status in ("OUT_OF_MODEL_SCOPE", "FAIL")


def test_light_leaving_a_cement_layer_through_its_side_is_reported() -> None:
    run = run_optical_project(_slid(cement=True), catalog=None)
    path = next(p for p in run.result.paths if p.parent_path_id is None)
    assert path.segments[2].medium_id == "project:CEMENT" and path.segments[2].end_surface_id == "observation/OUT"
    (issue,) = [i for i in run.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE"]
    assert (issue.member_id, issue.surface_id) == ("A", "A-S2")
    assert run.status == "OUT_OF_MODEL_SCOPE"


def test_the_rays_of_an_extended_emitter_are_read_by_the_body_behind_the_joint() -> None:
    """The sampled-ray branch of the check reads the same bodies (``SampledScope.exits``: entry surface, end surface,
    medium): into prism B across the joint and out through B-S2 is in scope; leaving prism B's crystal for the scene, or prism
    A's crystal ending on prism B's exit face, is not."""

    from os_optics.local.validity import element_scope_issues
    from os_optics.optics.rays.sampled import SampledLine, SampledScope, SampledTrace
    from os_optics.validation.ray_paths import RayPathTrace

    lowered = lower_project(place(build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])), catalog=None)
    result = run_optical_project(place(build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])), catalog=None).result
    line = SampledLine(source_id="EM", owner_id="EM", spectral_index=0, wavelength_nm=633.0, emitted_W=1.0, weight_W=1.0, sample_count=10, seed=1)

    def issues(exits):
        trace = RayPathTrace(input_hash="h", trace_hash=None, status="TRACED", sampled=SampledTrace(lines=[line], scope=SampledScope(exits=exits)))
        return [(i.member_id, i.surface_id, i.code) for i in element_scope_issues(result, lowered.elements, ray_paths=trace)
                if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE"]

    b, a = "project:CALCITE@X-B", "project:CALCITE@X-A"
    assert issues({("X/A-S2", "X/B-S2", "EM", 0, b): 5, ("X/A-S1", "X/A-S2", "EM", 0, a): 5, ("X/B-S2", "X/A-S2", "EM", 0, b): 2}) == []
    assert issues({("X/A-S2", None, "EM", 0, b): 3}) == [("B", "B-S1", "RAY_LEFT_ELEMENT_THROUGH_EDGE")]
    assert issues({("X/A-S1", "X/B-S2", "EM", 0, a): 3}) == [("A", "A-S1", "RAY_LEFT_ELEMENT_THROUGH_EDGE")]
