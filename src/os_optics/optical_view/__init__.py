"""Optical Simulation View: what is happening optically in one local Engine run.

Normal use goes through the local workflow, which builds the view of its own run::

    from os_optics.local import run_optical_project

    run = run_optical_project(project, output_dir="run")   # writes run/optical_scene.json and run/optical_scene.glb
    run.optical_view                                         # the OpticalSimulationSceneV1

The scene (``OpticalSimulationSceneV1``) is renderer-independent JSON: optical elements (public catalog design geometry or
symbolic markers), observation / target planes with the Engine's beam footprints, and the Engine-traced chief and pupil
ray polylines of every beam path. ``optical_scene.glb`` is its headless glTF 2.0 rendering. Every ray point comes from
``os_optics.validate_with_ray_paths`` -- the same Engine execution that produced the validation result -- and the
scene records that run's input, result and trace hashes. The view never traces or refracts anything itself.
"""

from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1, OpticalViewSettingsV1, PupilRaySelectionV1
from .artifacts import GLB_FILE, SCENE_FILE, canonical_json_bytes, optical_view_files, scene_json_bytes
from .builder import OpticalViewBindingError, build_optical_simulation_scene
from .glb import GlbFormatError, parse_glb, read_accessor, scene_glb_bytes
from .identity import BUILDER_PACKAGE, BUILDER_VERSION

__all__ = [
    "BUILDER_PACKAGE",
    "BUILDER_VERSION",
    "GLB_FILE",
    "SCENE_FILE",
    "GlbFormatError",
    "OpticalSimulationSceneV1",
    "OpticalViewBindingError",
    "OpticalViewSettingsV1",
    "PupilRaySelectionV1",
    "build_optical_simulation_scene",
    "canonical_json_bytes",
    "optical_view_files",
    "parse_glb",
    "read_accessor",
    "scene_glb_bytes",
    "scene_json_bytes",
]
