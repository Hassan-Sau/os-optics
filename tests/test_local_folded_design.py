"""The folded reference design (``example_folded_project``: source -> 45 deg fold mirror -> lens -> focal plane) on the
active catalog, its record and run directory, and its Optical Simulation View. The products are whatever the example
queries pick from the active release, so nothing here names a SKU.

Physics references are closed-form: specular reflection ``d' = d - 2 (d . n) n``, and a flat mirror adds no optical
power, so folding the beam must not move the focus along the path or change the spot.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from decimal import Decimal

import pytest

from os_optics import Status, validate_with_ray_paths
from os_optics.local import (
    AOI_TOLERANCE_DEG,
    LocalValidationStatusV1,
    ProjectDocumentV2,
    check_local_validation_binding,
    example_folded_project,
    example_project,
    json_api,
    load_public_catalog,
    run_optical_project,
    verify_run_artifacts,
)
from os_optics.optical_view import OpticalViewBindingError, build_optical_simulation_scene, parse_glb
from os_optics.optical_view import geometry as g
from os_optics._vendor.shared_contracts.optical_view import GeometryBasisV1, OpticBodyGeometryV1

from helpers import issue_codes, one_arrival


@pytest.fixture(scope="module")
def folded_run():
    return run_optical_project(example_folded_project())


@pytest.fixture(scope="module")
def straight_run():
    return run_optical_project(example_project())


def _component(project, instance_id):
    return next(c for c in project.components if c.instance_id == instance_id)


def _mirror_reflectance_at(wavelength_nm: str) -> float:
    catalog = load_public_catalog()
    product = catalog.get(_component(example_folded_project(), "M1").product.optical_sku)
    (surface,) = product.simulation_models[0].surfaces
    return float(Decimal(catalog.coating(surface.coating_id).band_at(Decimal(wavelength_nm)).reflectance))


def test_the_folded_design_keeps_the_focus_on_the_folded_leg_with_bounded_power(folded_run, straight_run):
    # The active public lens has no stated bulk loss: focus remains decidable,
    # while transmitted power has only the lossless upper envelope.
    assert folded_run.status is LocalValidationStatusV1.UNKNOWN and folded_run.engine_status is Status.UNKNOWN
    assert folded_run.model_scope_issues == () and issue_codes(folded_run.result) == {"BULK_LOSS_NOT_ESTABLISHED"}
    assert [(t.target_id, t.status.value) for t in folded_run.result.targets] == [("focus_position", "PASS"), ("spot_radius", "PASS")]
    arrival = one_arrival(folded_run.result, "focal_plane")
    focal_plane = folded_run.project.observations[0].pose.position_mm
    assert arrival.beam.position_mm == pytest.approx(tuple(focal_plane), abs=1e-9)
    straight = one_arrival(straight_run.result, "focal_plane")
    assert arrival.beam.power_W is straight.beam.power_W is None
    assert arrival.power_upper_bound_W == pytest.approx(straight.power_upper_bound_W * _mirror_reflectance_at("780"), rel=1e-12)
    assert arrival.beam.optical_path_length_mm == pytest.approx(straight.beam.optical_path_length_mm, abs=1e-9)
    # The fold adds no power: same 200 mm source-to-lens path, so the Gaussian focus is the straight design's.
    folded = arrival.beam.gaussian
    assert folded.valid and folded.waist_distance_x_mm == pytest.approx(straight.beam.gaussian.waist_distance_x_mm, abs=1e-9)
    assert folded.waist_radius_x_mm == pytest.approx(straight.beam.gaussian.waist_radius_x_mm, rel=1e-9)
    assert folded.beam_radius_x_mm == pytest.approx(folded.beam_radius_y_mm, rel=1e-9)  # a flat mirror adds no astigmatism
    for folded_target, straight_target in zip(folded_run.result.targets, straight_run.result.targets, strict=True):
        assert folded_target.value == pytest.approx(straight_target.value, rel=1e-9, abs=1e-9)


def test_the_chief_ray_is_folded_at_45_degrees_from_x_into_y(folded_run):
    (path,) = folded_run.result.paths
    (reflect,) = [s for s in path.segments if s.end_event == "reflect"]
    assert reflect.end_surface_id == "M1/S1"
    assert reflect.angle_of_incidence_deg == pytest.approx(45.0, abs=1e-9) and abs(reflect.angle_of_incidence_deg - 45.0) < AOI_TOLERANCE_DEG
    into, out = path.segments[0], path.segments[1]
    assert into.start_mm == pytest.approx(tuple(folded_run.project.components[0].nominal_pose.position_mm), abs=1e-12)
    assert into.end_mm == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert g.unit(g.sub(out.end_mm, out.start_mm)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    lens_y = _component(folded_run.project, "L1").nominal_pose.position_mm[1]
    assert out.end_surface_id == "L1/S1" and out.end_mm == pytest.approx((0.0, lens_y, 0.0), abs=1e-9)  # the first lens vertex


def test_every_optical_axis_is_horizontal_and_every_element_level():
    project = example_folded_project()
    poses = [c.nominal_pose for c in project.components] + [o.pose for o in project.observations]
    for pose in poses:
        q = tuple(pose.quaternion_wxyz)
        assert g.rotate(q, (0.0, 0.0, 1.0))[2] == pytest.approx(0.0, abs=1e-15)  # optical axis / face normal in the XY plane
        assert g.rotate(q, (0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0), abs=1e-15)  # local +Y is design up
    mirror = _component(project, "M1")
    s = math.sqrt(0.5)
    assert g.rotate(tuple(mirror.nominal_pose.quaternion_wxyz), (0.0, 0.0, 1.0)) == pytest.approx((-s, s, 0.0), abs=1e-15)
    assert [(d.axis.value, d.control.value) for d in mirror.retained_dofs] == [("Rx", "MANUAL"), ("Ry", "MANUAL")]
    lens = _component(project, "L1")
    assert [(d.axis.value, d.control.value) for d in lens.retained_dofs] == [("Tx", "MANUAL"), ("Ty", "MANUAL"), ("Tz", "NUMERICAL")]
    assert lens.product.optical_sku == _component(example_project(), "L1").product.optical_sku  # the same queried lens


def test_the_folded_record_binds_the_mirror_and_the_lens(folded_run):
    record = folded_run.local_validation
    assert check_local_validation_binding(record, folded_run.project) == () and folded_run.binding_problems() == ()
    catalog_version = load_public_catalog().catalog_version
    skus = {c.instance_id: c.product.optical_sku for c in folded_run.project.components if c.product is not None}
    assert [(p.instance_id, p.optical_sku, p.catalog_version, p.simulation_model.model_kind, p.simulation_model.model_version)
            for p in record.simulated_parts] == [("L1", skus["L1"], catalog_version, "SURFACE_SEQUENCE", "2"),
                                                 ("M1", skus["M1"], catalog_version, "SURFACE_SEQUENCE", "2")]
    assert [(d.instance_id, d.axis.value, d.control.value) for d in record.retained_dofs] == [
        ("L1", "Tx", "MANUAL"), ("L1", "Ty", "MANUAL"), ("L1", "Tz", "NUMERICAL"), ("M1", "Rx", "MANUAL"), ("M1", "Ry", "MANUAL")]
    assert record.validation_status is LocalValidationStatusV1.UNKNOWN  # catalog glass has no published bulk loss
    assert record.input_hash.digest == folded_run.engine_input_hash


def test_the_folded_run_is_deterministic(folded_run):
    again = run_optical_project(example_folded_project())
    assert again.engine_input_hash == folded_run.engine_input_hash
    assert again.local_validation.payload() == folded_run.local_validation.payload()
    assert again.optical_view.content_hash() == folded_run.optical_view.content_hash()
    assert again.summary() == folded_run.summary()


def test_the_folded_run_directory_verifies(tmp_path):
    run = run_optical_project(example_folded_project(), output_dir=tmp_path / "run")
    assert verify_run_artifacts(tmp_path / "run") == ()
    assert [p["instance_id"] for p in run.artifacts.manifest["simulated_parts"]] == ["M1", "L1"]


def test_the_folded_example_is_exposed_like_the_reference_example():
    reply = json_api.example_project_document(folded=True)
    assert reply["kind"] == "folded" and reply["project"] == example_folded_project().payload()
    assert reply["project_hash"] == example_folded_project().content_hash()
    assert json_api.example_project_document(kind="folded") == reply
    assert json_api.example_project_document()["project"] == example_project().payload()
    checked = json_api.check_project(reply["project"])
    assert checked["ok"] is True and checked["engine_input_hash"] == run_optical_project(example_folded_project()).engine_input_hash
    proc = subprocess.run([sys.executable, "-m", "os_optics", "project", "example", "--folded"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert ProjectDocumentV2.model_validate(json.loads(proc.stdout)).payload() == example_folded_project().payload()


def test_a_folded_design_with_the_mirror_turned_past_its_range_is_never_passed():
    document = example_folded_project().payload()
    theta = math.radians(50.0)
    z = (-math.cos(theta), math.sin(theta), 0.0)
    y = (0.0, 0.0, 1.0)
    next(c for c in document["components"] if c["instance_id"] == "M1")["nominal_pose"]["quaternion_wxyz"] = list(g.matrix_to_quat((g.cross(y, z), y, z)))
    run = run_optical_project(document)
    assert run.status in (LocalValidationStatusV1.FAIL, LocalValidationStatusV1.OUT_OF_MODEL_SCOPE)


# ---- the Optical Simulation View ---------------------------------------------------------------------------------------------------
def test_the_view_draws_the_mirror_body_at_its_pose(folded_run):
    scene = folded_run.optical_view
    assert [e.element_id for e in scene.elements] == ["SRC", "M1", "L1"]
    mirror = next(e for e in scene.elements if e.element_id == "M1")
    product = load_public_catalog().get(_component(folded_run.project, "M1").product.optical_sku)
    assert (mirror.optical_sku, mirror.label, mirror.optical_role.value) == (product.optical_sku, product.display_name, "mirror")
    geom = mirror.geometry
    assert isinstance(geom, OpticBodyGeometryV1) and geom.basis is GeometryBasisV1.PUBLIC_CATALOG_DESIGN_GEOMETRY
    body = product.design_geometry
    assert (geom.diameter_mm, geom.z_min_mm, geom.z_max_mm) == (float(body.diameter_mm), float(body.z_min_mm), float(body.z_max_mm))
    assert (geom.category, geom.reference_axis, geom.surfaces[-1].role.value, geom.surfaces[-1].vertex_z_mm) == (
        product.category, "FACE_NORMAL", "REFLECT", 0.0)  # the coated face at the element origin
    lens = next(e for e in scene.elements if e.element_id == "L1").geometry
    assert isinstance(lens, OpticBodyGeometryV1) and {s.role.value for s in lens.surfaces} == {"REFRACT"}
    (beam,) = scene.beam_paths
    assert (0.0, 0.0, 0.0) in beam.chief_ray.points_mm  # the chief ray turns on the mirror face
    marker = next(m for m in scene.axis_markers if m.element_id == "M1")
    s = math.sqrt(0.5)
    assert g.unit(g.scale(g.sub(marker.end_mm, marker.start_mm), 0.5)) == pytest.approx((-s, s, 0.0), abs=1e-6)  # along the face normal


def test_the_glb_carries_the_mirror_node(folded_run):
    from os_optics.optical_view import scene_glb_bytes

    gltf, _ = parse_glb(scene_glb_bytes(folded_run.optical_view))
    node = next(n for n in gltf["nodes"] if n.get("name") == "element:M1")
    assert node["extras"]["geometry_kind"] == "OPTIC_BODY"
    assert node["extras"]["optical_sku"] == _component(folded_run.project, "M1").product.optical_sku
    assert {m["name"] for m in gltf["materials"]} >= {"mirror_substrate", "mirror_face"}


def test_the_view_refuses_a_mirror_that_is_not_its_published_model(folded_run):
    request = folded_run.request
    components = list(request.project.components)
    index = next(i for i, c in enumerate(components) if c.instance_id == "M1")
    (surface,) = components[index].surfaces
    tampered_coating = surface.interaction.coating.model_copy(update={"reflectance_s": 0.5, "reflectance_p": 0.5})
    tampered_surface = surface.model_copy(update={"interaction": surface.interaction.model_copy(update={"coating": tampered_coating})})
    components[index] = components[index].model_copy(update={"surfaces": (tampered_surface,)})
    tampered = request.model_copy(update={"project": request.project.model_copy(update={"components": tuple(components)})})
    result, ray_paths = validate_with_ray_paths(tampered)
    with pytest.raises(OpticalViewBindingError, match="not its public model"):
        build_optical_simulation_scene(project=folded_run.project, catalog=load_public_catalog(), engine_request=tampered,
                                       engine_result=result, ray_paths=ray_paths)
