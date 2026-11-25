"""The Optical Simulation View as files, in memory: the scene JSON and its GLB rendering.

``optical_scene.json`` is the renderer-independent scene (the document a browser view consumes); ``optical_scene.glb`` is the
self-contained 3D file for any glTF viewer. This module performs no I/O: ``os_optics.local`` writes these files
into a run directory next to the Project, the Engine result and the Local Validation record, under one manifest.
"""

from __future__ import annotations

import json
from typing import Any

from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from .glb import scene_glb_bytes

SCENE_FILE = "optical_scene.json"
GLB_FILE = "optical_scene.glb"
JSON_INDENT = 2


def canonical_json_bytes(value: Any) -> bytes:
    """Sorted-key, indented UTF-8 JSON with a final newline (NaN / Infinity refused)."""

    return (json.dumps(value, sort_keys=True, indent=JSON_INDENT, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def scene_json_bytes(scene: OpticalSimulationSceneV1) -> bytes:
    return canonical_json_bytes(scene.payload())


def optical_view_files(scene: OpticalSimulationSceneV1) -> dict[str, bytes]:
    """``{"optical_scene.json": ..., "optical_scene.glb": ...}`` for one scene."""

    return {SCENE_FILE: scene_json_bytes(scene), GLB_FILE: scene_glb_bytes(scene)}
