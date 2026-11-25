"""``json_api.run_project(output_dir=...)`` serializes each headless run file once, writes those
bytes and checks the written directory against them without parsing it again."""

from __future__ import annotations

def test_run_project_serializes_once_and_does_not_reparse_the_written_directory(tmp_path, monkeypatch):
    from os_optics.local import artifacts, example_project, json_api

    calls = {"ray_data": 0, "load": 0}
    parts_of = artifacts.ray_paths_parts

    def counting(*args, **kwargs):
        calls["ray_data"] += 1
        return parts_of(*args, **kwargs)

    def no_load(*args, **kwargs):  # pragma: no cover - the test fails when called
        calls["load"] += 1
        raise AssertionError("the written run directory must not be parsed again")

    monkeypatch.setattr(artifacts, "ray_paths_parts", counting)
    monkeypatch.setattr(artifacts, "_load_json", no_load)
    reply = json_api.run_project(example_project().payload(), output_dir=str(tmp_path / "run"))
    assert reply["ok"] is True and reply["verify"] == []
    assert calls == {"ray_data": 1, "load": 0}
    for name, data in reply["files"].items():
        assert (tmp_path / "run" / name).read_bytes() == data


def test_a_written_file_that_differs_is_reported(tmp_path):
    from os_optics.local import example_project, run_optical_project
    from os_optics.local.artifacts import run_artifact_files, write_run_files, written_run_problems

    files = run_artifact_files(run_optical_project(example_project()))
    write_run_files(files, tmp_path)
    assert written_run_problems(files, tmp_path) == ()
    (tmp_path / "ray_data.json").write_bytes(files["ray_data.json"][:-1] + b" ")
    (tmp_path / "optical_scene.glb").unlink()
    assert written_run_problems(files, tmp_path) == (
        "ray_data.json: size or SHA-256 differs from the run's ray_data.json", "optical_scene.glb: missing")
