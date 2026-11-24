"""Same Project + public model + Engine trace + view version/settings -> same scene and same artifact bytes, in process
and across interpreter processes with different hash seeds, for the reference design and the folded multi-element
design (every OPTIC_BODY kind: lenses, mirror, window, splitter, polarizer, waveplate). Settings are part of the scene
identity. (Moved from the earlier ``opensci-optical-view`` package.)"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from optical_view_reference import build_scene, folded_project
from optical_view_reference import catalog, engine_request, engine_result, engine_run, project, ray_paths, scene  # noqa: F401  (pytest fixtures)

from os_optics._vendor.shared_contracts.optical_view import OpticalViewSettingsV1, PupilRaySelectionV1
from os_optics.optical_view import build_optical_simulation_scene, optical_view_files

TESTS_DIR = Path(__file__).resolve().parent

_CHILD = """
import hashlib, json, sys
sys.path.insert(0, {tests!r})
from optical_view_reference import build_scene, folded_project, public_catalog, reference_project
from os_optics.optical_view import optical_view_files
catalog = public_catalog()
out = {{}}
for name, project in (("reference", reference_project(catalog)), ("folded", folded_project(catalog))):
    scene, _request = build_scene(project, catalog)
    out[name] = {{file: hashlib.sha256(data).hexdigest() for file, data in optical_view_files(scene).items()}}
print(json.dumps(out, sort_keys=True))
"""


def _digests(scene) -> dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest() for name, data in optical_view_files(scene).items()}


def test_rebuilding_gives_identical_scene_and_bytes(scene, project, catalog, engine_request):
    again = build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request)
    assert again == scene and again.content_hash() == scene.content_hash()
    assert optical_view_files(again) == optical_view_files(scene)


def test_bytes_do_not_depend_on_the_process_or_hash_seed(scene, catalog):
    folded, _ = build_scene(folded_project(catalog), catalog)
    expected = {"reference": _digests(scene), "folded": _digests(folded)}
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", _CHILD.format(tests=str(TESTS_DIR))], capture_output=True,
                             text=True, env=env, timeout=300, check=True)
        assert json.loads(out.stdout) == expected


def test_settings_are_part_of_the_scene_identity(scene, project, catalog, engine_request, engine_result):
    all_rays = build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                              settings=OpticalViewSettingsV1(pupil_ray_selection=PupilRaySelectionV1.ALL))
    assert all_rays.content_hash() != scene.content_hash()
    assert len(all_rays.beam_paths[0].pupil_rays) == all_rays.beam_paths[0].pupil_ray_count_traced > len(scene.beam_paths[0].pupil_rays)
    assert all_rays.builder.settings.pupil_ray_selection is PupilRaySelectionV1.ALL
    # the same Engine points: the default selection is a subset
    shown = {r.ray_index: r.points_mm for r in scene.beam_paths[0].pupil_rays}
    assert {r.ray_index: r.points_mm for r in all_rays.beam_paths[0].pupil_rays if r.ray_index in shown} == shown
