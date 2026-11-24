"""One local run -> one run directory: simulation result, Local Validation record and Optical Simulation View all come
from the same single Engine execution, and a run directory whose files do not belong together is detected.

* ``run_optical_project(project, output_dir=...)`` / ``run.write_artifacts`` / ``os-optics project run --out-dir``
  write ``manifest.json``, ``project.json``, ``engine_request.json``, ``simulation_result.json``,
  ``local_validation.json``, ``optical_scene.json`` and ``optical_scene.glb``;
* the scene and the record carry equal Project, Engine input and Engine result hashes, and the scene's trace hash is
  the result's (cross-artifact assertion);
* ``verify_run_artifacts`` / ``os-optics project verify`` re-check it all and catch a scene of another run;
* FAIL / OUT_OF_MODEL_SCOPE runs produce complete, honest run directories; a view that cannot be bound to its run is an
  error, never a scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal

import pytest

import os_optics.local.run as local_run
from os_optics import trace_ray_paths
from os_optics.cli import main
from os_optics.local import (
    MODEL_SCOPE_CODES,
    LocalRunError,
    LocalRunErrorCode,
    LocalValidationStatusV1,
    example_project,
    load_public_catalog,
    run_optical_project as _run_optical_project,
    verify_run_artifacts,
)
from os_optics.local.artifacts import MANIFEST_FILE, RUN_FILES, run_artifact_files
from os_optics.optical_view import parse_glb
from local_helpers import catalog_document


def _artifact_catalog():
    """Test catalog's glass is assumed lossless for deterministic PASS/FAIL artifact examples."""
    document = catalog_document()
    document["schema_version"] = "3.6"  # Required for material bulk-loss evidence.
    for material in document["materials"]:
        material["bulk_loss"] = {"kind": "INTERNAL_TRANSMITTANCE",
                                 "wavelength_nm": [material["valid_band_nm"]["min_nm"], material["valid_band_nm"]["max_nm"]],
                                 "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
                                 "basis": "Assumed lossless synthetic test fixture."}
    return document


def run_optical_project(project, **kwargs):
    return _run_optical_project(project, catalog=_artifact_catalog(), **kwargs)


def _artifact_catalog_file(directory):
    path = directory / "synthetic-catalog.json"
    path.write_text(json.dumps(_artifact_catalog()), encoding="utf-8")
    return path


RUN_FILE_NAMES = {"manifest.json", "project.json", "engine_request.json", "simulation_result.json", "ray_data.json", "local_validation.json",
                  "optical_scene.json", "optical_scene.glb"}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return run_optical_project(example_project(), output_dir=out)


def _lens():
    """The public lens the example design queried from the active catalog."""

    sku = next(c for c in example_project().components if c.instance_id == "L1").product.optical_sku
    return load_public_catalog().get(sku)


def _document(**changes):
    document = example_project().payload()
    for mutate in changes.values():
        mutate(document)
    return document


def _rewrite(path, data: bytes) -> None:
    path.write_bytes(data)


def _refresh_manifest(directory) -> None:
    """Make the manifest's file digests match whatever is on disk (a forger who also edits the manifest)."""
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    for entry in manifest["files"]:
        data = (directory / entry["name"]).read_bytes()
        entry["bytes"], entry["sha256"] = len(data), hashlib.sha256(data).hexdigest()
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---- the run directory ------------------------------------------------------------------------------------------------------
def test_one_run_writes_the_complete_run_directory(reference):
    directory = reference.artifacts.directory
    assert {p.name for p in directory.iterdir()} == RUN_FILE_NAMES
    assert set(reference.artifacts.paths) == {role for role, *_ in RUN_FILES} | {"manifest"}
    manifest = reference.artifacts.manifest
    assert manifest["schema"] == "opensci.local_run" and manifest["status"]["validation_status"] == "PASS"
    assert [f["name"] for f in manifest["files"]] == [name for _role, name, *_ in RUN_FILES]
    assert manifest["binding"] == {"one_engine_execution": True, "record_binding_problems": [], "cross_artifact_problems": []}
    assert verify_run_artifacts(directory) == ()


def test_the_manifest_names_the_run_identity(reference):
    manifest, record, result = reference.artifacts.manifest, reference.local_validation, reference.result
    assert manifest["project"]["project_hash"]["digest"] == record.project.project_hash.digest == reference.project.content_hash()
    catalog_version = load_public_catalog().catalog_version
    assert manifest["catalog_version"] == catalog_version
    assert manifest["simulated_parts"] == [{"instance_id": "L1", "optical_sku": _lens().optical_sku, "catalog_version": catalog_version,
                                            "simulation_model": {"model_kind": "SURFACE_SEQUENCE", "model_version": "2"}}]
    assert [(d["instance_id"], d["axis"]) for d in manifest["retained_dofs"]] == [("L1", "Tx"), ("L1", "Ty"), ("L1", "Tz")]
    engine = manifest["engine_run"]
    assert engine["engine_package"] == record.runtime.engine_package == "os-optics"
    assert engine["engine_version"] == result.manifest.engine_version and engine["engine_build_id"] == result.manifest.engine_build_id
    assert engine["input_hash"]["digest"] == record.input_hash.digest and engine["result_hash"]["digest"] == record.result_hash.digest
    assert engine["trace_hash"] == result.manifest.trace_hash
    assert manifest["optical_view"]["scene_content_hash_osc1"] == reference.optical_view.content_hash()


def test_each_file_is_the_run_object_it_claims_to_be(reference):
    paths = reference.artifacts.paths
    assert json.loads(paths["project"].read_bytes()) == reference.project.payload()
    assert json.loads(paths["local_validation"].read_bytes()) == reference.local_validation.payload()
    assert json.loads(paths["simulation_result"].read_bytes()) == reference.result.model_dump(mode="json")
    assert json.loads(paths["engine_request"].read_bytes()) == reference.request.model_dump(mode="json")
    assert json.loads(paths["optical_scene"].read_bytes()) == reference.optical_view.payload()
    gltf, _ = parse_glb(paths["optical_scene_glb"].read_bytes())
    assert gltf["asset"]["extras"]["scene_content_hash"] == reference.optical_view.content_hash()
    # the 3D file alone names the Engine run of the record: input and result hash
    assert gltf["asset"]["extras"]["engine_input_hash"] == reference.local_validation.input_hash.digest
    assert gltf["asset"]["extras"]["engine_result_hash"] == reference.local_validation.result_hash.digest
    assert gltf["asset"]["generator"].startswith("os-optics ")


# ---- same Engine run ----------------------------------------------------------------------------------------------------------
def test_scene_and_local_validation_come_from_the_same_engine_run(reference):
    scene, record, manifest = reference.optical_view, reference.local_validation, reference.result.manifest
    assert scene.engine_run.input_hash == record.input_hash
    assert scene.engine_run.result_hash == record.result_hash
    assert scene.project == record.project
    assert scene.engine_run.trace_hash == manifest.trace_hash == reference.ray_paths.trace_hash
    assert reference.ray_paths.input_hash == manifest.input_hash == reference.engine_input_hash
    assert scene.engine_run.engine_version == record.runtime.engine_version == manifest.engine_version


def test_rays_of_another_run_are_an_error_never_a_scene(monkeypatch):
    """If the Engine execution handed the view rays of a different request, the run refuses to draw them."""
    other = run_optical_project(_document(power=lambda d: d["components"][0]["source_model"].update(power_w=0.002)))
    real = local_run.traced_evaluation
    monkeypatch.setattr(local_run, "traced_evaluation", lambda request, **kw: (real(request, **kw)[0], other.ray_paths))
    with pytest.raises(LocalRunError) as err:
        run_optical_project(example_project())
    assert err.value.codes == (LocalRunErrorCode.OPTICAL_VIEW_BINDING_FAILED,)
    assert "different Engine inputs" in str(err.value)


def test_a_scene_from_another_run_is_detected_in_a_run_directory(reference, tmp_path):
    other_dir = tmp_path / "other"
    run_optical_project(_document(power=lambda d: d["components"][0]["source_model"].update(power_w=0.002)), output_dir=other_dir)
    forged = tmp_path / "forged"
    reference.write_artifacts(forged)
    for name in ("optical_scene.json", "optical_scene.glb"):
        _rewrite(forged / name, (other_dir / name).read_bytes())
    assert any("size or SHA-256" in p for p in verify_run_artifacts(forged))
    manifest = json.loads((forged / MANIFEST_FILE).read_bytes())
    manifest["optical_view"]["scene_content_hash_osc1"] = json.loads((other_dir / MANIFEST_FILE).read_bytes())["optical_view"]["scene_content_hash_osc1"]
    (forged / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_manifest(forged)
    problems = verify_run_artifacts(forged)
    assert "the optical scene and the local validation record have different Engine input hashes" in problems
    assert "the optical scene and the local validation record have different Engine result hashes" in problems


def test_an_edited_record_is_detected(reference, tmp_path):
    reference.write_artifacts(tmp_path)
    record = json.loads((tmp_path / "local_validation.json").read_bytes())
    record["result_hash"]["digest"] = "0" * 64
    _rewrite(tmp_path / "local_validation.json", json.dumps(record).encode())
    _refresh_manifest(tmp_path)
    problems = verify_run_artifacts(tmp_path)
    assert "the optical scene and the local validation record have different Engine result hashes" in problems
    assert "the Engine result's manifest hashes differ from the local validation record's" in problems


def test_an_edited_result_or_request_is_detected(reference, tmp_path):
    reference.write_artifacts(tmp_path)
    result = json.loads((tmp_path / "simulation_result.json").read_bytes())
    result["targets"][0]["status"] = "PASS" if result["targets"][0]["status"] != "PASS" else "FAIL"
    _rewrite(tmp_path / "simulation_result.json", json.dumps(result).encode())
    request = json.loads((tmp_path / "engine_request.json").read_bytes())
    request["project"]["components"][0]["source"]["power_W"] = 0.5
    _rewrite(tmp_path / "engine_request.json", json.dumps(request).encode())
    _refresh_manifest(tmp_path)
    problems = verify_run_artifacts(tmp_path)
    assert "engine_request.json: does not hash to the record's Engine input_hash" in problems
    assert any(p.startswith("simulation_result.json: does not match its own manifest result_hash") for p in problems)


def test_missing_or_foreign_directories_are_reported(tmp_path):
    assert verify_run_artifacts(tmp_path) == ("not a run directory: it has no manifest.json (write one with 'os-optics "
                                              "project run PROJECT --out-dir DIR')",)
    assert verify_run_artifacts(tmp_path / "nodir")[0].startswith("not a run directory")
    (tmp_path / MANIFEST_FILE).write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")
    assert verify_run_artifacts(tmp_path) == ("manifest.json: not an opensci.local_run 1.1 or 1.2 manifest",)


# ---- regeneration -------------------------------------------------------------------------------------------------------------
def test_writing_again_is_deterministic_and_safe(reference, tmp_path):
    first = reference.write_artifacts(tmp_path / "a")
    rerun = run_optical_project(example_project(), output_dir=tmp_path / "b")
    for name in RUN_FILE_NAMES:
        assert (first.directory / name).read_bytes() == (rerun.artifacts.directory / name).read_bytes(), name
    # re-using a directory of another run replaces every run file and leaves foreign files alone
    (tmp_path / "b" / "notes.txt").write_text("mine", encoding="utf-8")
    other = run_optical_project(_document(power=lambda d: d["components"][0]["source_model"].update(power_w=0.002)), output_dir=tmp_path / "b")
    assert verify_run_artifacts(tmp_path / "b") == ()
    assert json.loads((tmp_path / "b" / "local_validation.json").read_bytes()) == other.local_validation.payload()
    assert (tmp_path / "b" / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not list((tmp_path / "b").glob("*.partial")) and not list((tmp_path / "b").glob(".*.partial"))


def test_in_memory_files_equal_the_written_files(reference):
    files = run_artifact_files(reference)
    assert set(files) == RUN_FILE_NAMES
    for name, data in files.items():
        assert (reference.artifacts.directory / name).read_bytes() == data


# ---- ray data -----------------------------------------------------------------------------------------------------------------
def test_ray_data_is_the_trace_of_the_same_engine_run(reference):
    from os_optics.validation.ray_paths import ray_paths_document

    ray_data = json.loads(reference.artifacts.paths["ray_data"].read_bytes())
    assert ray_data == json.loads(json.dumps(ray_paths_document(reference.ray_paths)))
    manifest, result = reference.artifacts.manifest, reference.result.manifest
    assert ray_data["schema"] == "opensci.ray_data" and ray_data["status"] == "TRACED"
    assert ray_data["input_hash"] == result.input_hash and ray_data["trace_hash"] == result.trace_hash == manifest["ray_data"]["trace_hash"]
    assert [p["path_id"] for p in ray_data["paths"]] == [p.path_id for p in reference.result.paths]
    assert manifest["schema_version"] == "1.2" and manifest["ray_data"]["path_count"] == len(reference.result.paths)


def _copy_run(reference, tmp_path):
    directory = tmp_path / "copy"
    directory.mkdir()
    for path in reference.artifacts.directory.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    return directory


@pytest.mark.parametrize("tamper", ["trace_hash", "input_hash", "status", "paths"])
def test_ray_data_of_another_run_is_detected(reference, tmp_path, tamper):
    directory = _copy_run(reference, tmp_path)
    ray_data = json.loads((directory / "ray_data.json").read_bytes())
    if tamper == "paths":
        ray_data["paths"] = ray_data["paths"] + [dict(ray_data["paths"][0], path_id="ghost")]
    elif tamper == "status":
        ray_data["status"] = "NOT_TRACED"
    else:
        ray_data[tamper] = "0" * 64
    (directory / "ray_data.json").write_text(json.dumps(ray_data), encoding="utf-8")
    _refresh_manifest(directory)  # a forger who also fixes the file digests
    problems = verify_run_artifacts(directory)
    assert problems and all(p.startswith("ray_data.json: ") for p in problems), problems


def test_a_run_directory_must_carry_its_ray_data(reference, tmp_path):
    """One run-directory format: a manifest relabelled to another version, or one that drops the ray data, does not verify."""
    directory = _copy_run(reference, tmp_path)
    (directory / "ray_data.json").unlink()
    manifest = json.loads((directory / MANIFEST_FILE).read_bytes())
    manifest["schema_version"] = "1.0"
    manifest.pop("ray_data")
    manifest["files"] = [f for f in manifest["files"] if f["role"] != "ray_data"]
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_run_artifacts(directory) == ("manifest.json: not an opensci.local_run 1.1 or 1.2 manifest",)
    manifest["schema_version"] = "1.1"
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    assert "manifest.json: the ray_data file is not listed as ray_data.json" in verify_run_artifacts(directory)


# ---- honest outcomes --------------------------------------------------------------------------------------------------------
def _fail_document():
    return _document(tight=lambda d: d["targets"][1].update(maximum=0.01))


def _out_of_band_document():
    """The example with a source 1 nm beyond the lens model's band (still inside its glass data): the Engine traces it; the
    published model is not characterized there."""

    beyond = float(Decimal(_lens().simulation_models[0].valid_band_nm.max_nm) + 1)

    def shift(d):
        d["components"][0]["source_model"]["wavelength_nm"] = beyond
        d["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "focal_plane", "minimum": 0.0, "unit": "W"}]
    return _document(shift=shift)


def _out_of_scope_metric_document():
    return _document(psf=lambda d: d["targets"].append({"target_id": "psf_width", "metric": "psf", "observation_id": "focal_plane", "maximum": 0.05}))


@pytest.mark.parametrize(("document", "status", "engine_status"), [
    (_fail_document, "FAIL", "FAIL"),
    (_out_of_band_document, "OUT_OF_MODEL_SCOPE", "PASS"),
    (_out_of_scope_metric_document, "OUT_OF_MODEL_SCOPE", "OUT_OF_MODEL_SCOPE"),
], ids=["fail", "published-model-validity", "engine-out-of-scope-metric"])
def test_non_passing_runs_write_complete_honest_run_directories(document, status, engine_status, tmp_path):
    run = run_optical_project(document(), output_dir=tmp_path)
    manifest = run.artifacts.manifest
    assert (manifest["status"]["validation_status"], manifest["status"]["simulation_status"]) == (status, engine_status)
    assert run.status is LocalValidationStatusV1(status)
    assert verify_run_artifacts(tmp_path) == ()
    assert run.optical_view.engine_run.result_hash == run.local_validation.result_hash
    if status != engine_status:
        assert manifest["status"]["model_scope_issues"] and manifest["status"]["model_scope_issues"][0]["code"] in MODEL_SCOPE_CODES


def _on_rim(waist_mm: float):
    """The example with its beam's chief ray on the opaque rim between the lens's clear aperture and its edge."""
    model = _lens().simulation_models[0]
    rim = float((Decimal(model.surfaces[0].clear_radius_mm) + Decimal(model.edge_radius_mm)) / 2)

    def move(d):
        d["components"][0]["nominal_pose"]["position_mm"][1] = rim
        d["components"][0]["source_model"]["waist_radius_mm"] = waist_mm
    return _document(off_axis=move)


def test_a_clipped_beam_is_drawn_stopping_at_the_lens():
    """A beam whose chief ray hits the opaque rim between the clear aperture and the edge, with none of its pupil rays
    inside the opening (a 0.1 mm beam 0.3 mm beyond it), stops at the lens in the view; the clip is a WARNING (a physical
    outcome), not a failed run."""
    run = run_optical_project(_on_rim(0.1))
    assert run.status is not LocalValidationStatusV1.FAIL
    assert {i.status.value for i in run.result.issues if i.code.value == "CLIPPED"} == {"WARNING"}
    (path,) = run.optical_view.beam_paths
    assert path.termination == "CLIPPED" and path.chief_ray.end.value == "STOPPED_AT_SURFACE"
    assert run.optical_view.engine_run.input_hash == run.local_validation.input_hash


def test_a_vignetted_beam_goes_on_as_the_rays_that_pass_the_lens():
    """The 1 mm beam on the same rim: part of it falls inside the clear aperture and goes on -- drawn past the
    lens, its reference ray a pupil ray that passed; the clip is a WARNING, not a failed run."""
    run = run_optical_project(_on_rim(1.0))
    assert run.status is not LocalValidationStatusV1.FAIL
    (path,) = run.optical_view.beam_paths
    assert path.termination != "CLIPPED"
    clip = next(i for i in run.result.issues if i.code.value == "CLIPPED")
    assert 0 < clip.numerical_context["pupil_rays_passing"] < clip.numerical_context["pupil_rays_live"]
    assert 0.0 < clip.numerical_context["transmitted_fraction"] < 0.5  # (the chief ray outside the opening: under half)
    (arrival,) = [a for o in run.result.observations for a in o.arrivals]
    assert arrival.reference_ray_promoted_at == ("L1/S1",) and arrival.first_order is None
    assert run.result.power_budget.conserved


def test_unsupported_inputs_still_fail_closed_before_any_artifact(tmp_path):
    document = _document(sku=lambda d: d["components"][1]["product"].update(optical_sku="OS-OPT-NOT-PUBLISHED"))
    with pytest.raises(LocalRunError) as err:
        run_optical_project(document, output_dir=tmp_path / "run")
    assert LocalRunErrorCode.OPTICAL_SKU_NOT_PUBLISHED in err.value.codes
    assert not (tmp_path / "run").exists()


# ---- CLI ------------------------------------------------------------------------------------------------------------------------
def _write_project(tmp_path, document) -> str:
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_cli_project_run_writes_the_run_directory(tmp_path, capsys, reference):
    out = tmp_path / "run"
    assert main(["project", "run", _write_project(tmp_path, example_project().payload()), "--catalog", str(_artifact_catalog_file(tmp_path)),
                 "--out-dir", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS" and summary["artifacts"]["optical_scene_glb"] == str(out / "optical_scene.glb")
    assert summary["artifacts"]["local_validation"] == str(out / "local_validation.json")
    assert {p.name for p in out.iterdir()} == RUN_FILE_NAMES
    for name in RUN_FILE_NAMES:
        assert (out / name).read_bytes() == (reference.artifacts.directory / name).read_bytes(), name
    assert main(["project", "verify", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"run_dir": str(out), "status": "CONSISTENT", "problems": []}


def test_cli_verify_reports_an_inconsistent_directory(tmp_path, capsys, reference):
    reference.write_artifacts(tmp_path)
    (tmp_path / "optical_scene.glb").write_bytes(b"not a glb")
    assert main(["project", "verify", str(tmp_path)]) == 66
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "INCONSISTENT" and any("optical_scene.glb" in p for p in report["problems"])


def test_cli_run_exit_code_is_the_status_and_the_view_selection_is_display_only(tmp_path, capsys):
    project = _write_project(tmp_path, _fail_document())
    assert main(["project", "run", project, "--catalog", str(_artifact_catalog_file(tmp_path)), "--out-dir", str(tmp_path / "all"),
                 "--pupil-rays", "ALL"]) == 1
    capsys.readouterr()
    assert main(["project", "run", project, "--catalog", str(_artifact_catalog_file(tmp_path)), "--out-dir", str(tmp_path / "outer")]) == 1
    capsys.readouterr()
    scene_all = json.loads((tmp_path / "all" / "optical_scene.json").read_bytes())
    scene_outer = json.loads((tmp_path / "outer" / "optical_scene.json").read_bytes())
    assert len(scene_all["beam_paths"][0]["pupil_rays"]) == 96 and len(scene_outer["beam_paths"][0]["pupil_rays"]) == 12
    for name in ("local_validation.json", "simulation_result.json"):
        assert (tmp_path / "all" / name).read_bytes() == (tmp_path / "outer" / name).read_bytes()


def test_cli_schema_and_help_name_the_view(capsys):
    assert main(["schema", "optical-scene"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "OPTICAL_SIMULATION_VIEW" in json.dumps(schema)
    with pytest.raises(SystemExit):
        main(["project", "run", "--help"])
    text = capsys.readouterr().out
    assert "--out-dir" in text and "optical_scene.glb" in text and "local_validation.json" in text


def test_summary_without_artifacts_and_record_only_output_still_work(tmp_path, capsys):
    record = tmp_path / "record.json"
    assert main(["project", "run", _write_project(tmp_path, example_project().payload()), "--catalog", str(_artifact_catalog_file(tmp_path)),
                 "-o", str(record)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifacts"] is None and summary["optical_view"]["beam_paths"] == 1
    assert json.loads(record.read_bytes())["validation_status"] == "PASS"


def test_trace_ray_paths_reproduces_the_runs_rays(reference):
    again = trace_ray_paths(reference.request)
    assert again == reference.ray_paths
    assert copy.deepcopy(reference.ray_paths) == reference.ray_paths


# ---- a run is verified once, where it crosses a trust boundary -------------------------------------------------------------------
def test_the_lowering_binds_the_catalog_as_resolve_simulated_parts_does():
    """The server recomputes ``resolve_simulated_parts`` at intake, so the binding the lowering computes itself
    (``LoweredProject.simulated_parts``) must be that function's, for every reference design. A run no longer compares
    the two each time: this test holds the property."""

    from local_helpers import DICHROIC, FOLD_X_TO_Y, PCX_F100, PLATE_BS, catalog_document, element, plane, project, source

    from os_optics._vendor.shared_contracts.optical_design import resolve_simulated_parts
    from os_optics.local import EXAMPLE_KINDS, example_project_of_kind, load_project, lower_project
    from os_optics.local.run import project_catalog

    cases = [(example_project_of_kind(kind), None) for kind in EXAMPLE_KINDS]
    cases.append((load_project(project(
        [source(), element("L1", "lens", PCX_F100, (0.0, 0.0, 0.0)), element("BS", "splitter", PLATE_BS, (60.0, 0.0, 0.0), FOLD_X_TO_Y),
         element("DM", "filter", DICHROIC, (120.0, 0.0, 0.0), FOLD_X_TO_Y)],
        [plane("screen", (200.0, 0.0, 0.0), radius_mm=40.0)])), catalog_document()))
    bound = 0
    for document, given in cases:
        catalog = project_catalog(document, given)
        lowered = lower_project(document, catalog)
        expected = resolve_simulated_parts(document, catalog) if catalog is not None else ()
        assert lowered.simulated_parts == expected, document.content_hash()
        assert _run_optical_project(document, catalog=given).simulated_parts == expected
        bound += len(expected)
    assert bound >= len(cases)  # (the designs do bind catalog products)


def _unbound_record(monkeypatch):
    """Every run made while this is in force gets a record naming another Project: the internal defect the run's own
    binding refusal used to stop."""

    real = local_run.local_validation_record

    def another_project(document, lowered, result, **kwargs):
        record = real(document, lowered, result, **kwargs)
        digest = record.project.project_hash.model_copy(update={"digest": "0" * 64})
        return record.model_copy(update={"project": record.project.model_copy(update={"project_hash": digest})})

    monkeypatch.setattr(local_run, "local_validation_record", another_project)


def test_a_run_whose_record_does_not_bind_is_returned_and_written_with_the_problems_named(tmp_path, monkeypatch, reference):
    """The run is not refused inside the process: its numbers stay readable. The manifest names the binding problems,
    ``verify`` reports them, and every other byte of the run is the consistent run's."""

    from os_optics.results import load_run

    _unbound_record(monkeypatch)
    out = tmp_path / "run"
    run = run_optical_project(example_project(), output_dir=out)
    monkeypatch.undo()
    problems = run.binding_problems()
    assert problems and all("project" in problem.lower() for problem in problems)
    manifest = json.loads((out / MANIFEST_FILE).read_bytes())
    assert manifest["binding"]["record_binding_problems"] == list(problems)
    assert manifest["binding"]["cross_artifact_problems"] == ["the optical scene and the local validation record name different Projects"]
    assert manifest["binding"]["one_engine_execution"] is True
    reported = verify_run_artifacts(out)
    assert any(problem.startswith("local_validation.json:") for problem in reported)
    assert main(["project", "verify", str(out)]) == 66
    # the numbers are the consistent run's, readable as stored
    assert run.result.manifest.result_hash == reference.result.manifest.result_hash
    assert load_run(out).targets()["targets"] == load_run(reference.artifacts.directory).targets()["targets"]
    # a consistent run's manifest states no problem, as before
    assert reference.artifacts.manifest["binding"] == {"one_engine_execution": True, "record_binding_problems": [],
                                                       "cross_artifact_problems": []}


def test_a_run_whose_record_does_not_bind_is_refused_wherever_it_leaves_the_device(tmp_path, monkeypatch, reference, capsys):
    """The check moved to the trust boundary: a submission made from the run object, from its directory or from its files
    is refused; the consistent run is submitted from all three with the same document."""

    from os_optics.local.submission import RunDirectoryInconsistentError, submission_document

    _unbound_record(monkeypatch)
    out = tmp_path / "run"
    run = run_optical_project(example_project(), output_dir=out)
    monkeypatch.undo()
    files = {path.name: path.read_bytes() for path in out.iterdir()}
    for given in (run, out, str(out), files, run_artifact_files(run)):
        with pytest.raises(RunDirectoryInconsistentError) as refused:
            submission_document(given)
        assert refused.value.problems
    with pytest.raises(RunDirectoryInconsistentError) as refused:
        submission_document(run)
    assert tuple(refused.value.problems) == run.binding_problems()
    good = reference.artifacts.directory
    documents = [submission_document(given) for given in (reference, good, {path.name: path.read_bytes() for path in good.iterdir()})]
    assert documents[0] == documents[1] == documents[2]
    capsys.readouterr()
