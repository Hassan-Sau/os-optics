"""``json_api.optical_scene_glb``: the glTF binary of a scene a headless JSON host received.

The reference is the run directory: a complete evaluation's scene is the run's scene, so its GLB must be the run's
``optical_scene.glb`` byte for byte; a narrowed evaluation's scene gives the GLB the Engine's writer makes of it.
"""

from __future__ import annotations

from os_optics._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from os_optics.local import json_api
from os_optics.optical_view.glb import scene_glb_bytes


def _example() -> dict:
    return json_api.example_project_document()["project"]


def test_the_glb_of_a_complete_evaluation_scene_is_the_run_directory_glb():
    project = _example()
    run = json_api.run_project(project)
    evaluation = json_api.evaluate_project(project, ["complete"])
    assert run["ok"] and evaluation["ok"]
    reply, files = json_api.call_method("optical_scene_glb", {"scene": evaluation["optical_scene"]})
    assert reply == {"ok": True}
    assert files == {"optical_scene.glb": run["files"]["optical_scene.glb"]}


def test_the_glb_of_a_view_through_a_prepared_handle_is_the_writer_bytes_of_its_scene():
    project = _example()
    handle = json_api.prepare_evaluation(project)["handle"]
    try:
        view = json_api.evaluate_prepared(handle, ["view"])
        assert view["ok"]
        reply, files = json_api.call_method("optical_scene_glb", {"scene": view["optical_scene"]})
        assert reply == {"ok": True}
        glb = files["optical_scene.glb"]
        assert glb[:4] == b"glTF"
        assert glb == scene_glb_bytes(OpticalSimulationSceneV1.model_validate(view["optical_scene"]))
    finally:
        json_api.release_evaluation(handle)


def test_a_document_that_is_not_a_scene_is_refused_with_what_does_not_fit():
    refused = json_api.optical_scene_glb({"schema": "opensci.optical_simulation_scene"})
    assert refused["ok"] is False
    assert refused["error"]["code"] == "INPUT_INVALID"
    assert "optical scene" in refused["error"]["message"]
    assert refused["error"]["problems"][0]["code"] == "INPUT_INVALID"
    assert json_api.optical_scene_glb(None)["error"]["code"] == "INPUT_INVALID"
