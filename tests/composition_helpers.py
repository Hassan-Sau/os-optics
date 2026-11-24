"""Small modules for the composition tests, built from the native component library (no catalog products).

Every physical value here is a test fixture: an ideal 632.8 nm laser, native flat mirrors / ideal thin lenses / plate
splitters with declared reflectances. They exercise geometry and bookkeeping, not supplier data.

Frames: the design frame of every module has its beam along +X at z = 0 (``ALONG_X``: local +Z -> design +X).
"""

from __future__ import annotations

import copy
import math
from typing import Any

from local_helpers import ALONG_X, ALONG_Y, FOLD_X_TO_Y, source

from os_optics.local import json_api

WAVELENGTH_NM = 632.8
MIRROR = {"diameter_mm": "25.4", "reflectance": "0.98", "valid_band_nm": [400, 700]}
SPLITTER = {"diameter_mm": "25.4", "reflectance": "0.5", "transmittance": "0.5", "valid_band_nm": [600, 700], "thickness_mm": "1"}


def build(class_id: str, parameters: dict[str, Any], *, model_id: str | None = None) -> dict[str, Any]:
    params = dict(parameters)
    if model_id is not None:
        params["model_id"] = model_id
    reply = json_api.build_native_model(class_id, params)
    assert reply["ok"], reply
    return reply


def place(build_reply: dict[str, Any], instance_id: str, position, quaternion) -> dict[str, Any]:
    component = copy.deepcopy(build_reply["component_template"])
    component["instance_id"] = instance_id
    component["nominal_pose"] = {"position_mm": [float(v) for v in position], "quaternion_wxyz": [float(v) for v in quaternion]}
    return component


def project_of(components, observations=(), targets=(), builds=(), design_intent: str | None = "OPTICAL_SYSTEM") -> dict[str, Any]:
    models, materials, coatings = [], [], []
    for reply in builds:
        for model in (*reply["member_models"], reply["optical_model"]):
            if model not in models:
                models.append(model)
        for material in reply["model_materials"] or ():
            if material not in materials:
                materials.append(material)
        for coating in reply["model_coatings"] or ():
            if coating not in coatings:
                coatings.append(coating)
    document: dict[str, Any] = {"schema": "opensci.project", "project_schema_version": "2.0", "components": list(components),
                                "observations": list(observations), "targets": list(targets)}
    if design_intent is not None:
        document["design_intent"] = design_intent
    if models:
        document["optical_models"] = models
    if materials:
        document["model_materials"] = materials
    if coatings:
        document["model_coatings"] = coatings
    return document


def screen(observation_id: str, position, quaternion=ALONG_X, *, radius_mm: float = 12.5, absorbing: bool = False,
           device: str | None = None) -> dict[str, Any]:
    plane = {"observation_id": observation_id, "pose": {"position_mm": [float(v) for v in position], "quaternion_wxyz": list(quaternion)},
             "aperture": {"shape": "circle", "radius_mm": radius_mm}}
    if absorbing:
        plane["absorbing"] = True
    if device is not None:
        plane["device_instance_id"] = device
    return plane


def laser(instance_id: str = "SRC", position=(-100.0, 0.0, 0.0), quaternion=ALONG_X, power_w: float = 0.001) -> dict[str, Any]:
    return source(instance_id, position, wavelength_nm=WAVELENGTH_NM, waist_mm=0.5, quaternion=quaternion, power_w=power_w)


def power_target(target_id: str, observation_id: str, minimum: float, source_id: str | None = None) -> dict[str, Any]:
    target: dict[str, Any] = {"target_id": target_id, "metric": "power_W", "observation_id": observation_id, "minimum": minimum}
    if source_id is not None:
        target["source_id"] = source_id
    return target


def fold_module(*, mirror_model_id: str = "fold-mirror", absorbing_plane: bool = True, reflectance: str | None = None) -> dict[str, Any]:
    """A 90 degree fold: a test laser along +X, a flat mirror at the origin turning the beam to +Y, a test screen 100 mm
    after it with a test power target, a free monitor plane at y = 50 mm that stays in the system, and a carried
    centroid target on it. ``reflectance``: another mirror reflectance (a revised module)."""

    mirror = build("mirror.flat", {**MIRROR, **({"reflectance": reflectance} if reflectance is not None else {})}, model_id=mirror_model_id)
    project = project_of(
        [laser(), place(mirror, "M1", (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
        [screen("OUT", (0.0, 100.0, 0.0), ALONG_Y, absorbing=absorbing_plane), screen("MON", (0.0, 50.0, 0.0), ALONG_Y, radius_mm=20.0)],
        [power_target("T_OUT", "OUT", 0.0009, "SRC"),
         {"target_id": "T_MON_CENTER", "metric": "spot_centroid_u_mm", "observation_id": "MON", "target_value": 0.0, "tolerance_abs": 0.5}],
        [mirror])
    return project


FOLD_DESCRIPTOR = {
    "module_id": "fold-90", "title": "90 degree fold", "revision": "r1",
    "fixtures": [{"object_kind": "COMPONENT", "object_id": "SRC", "role": "TEST_SOURCE", "default_action": "REMOVE"},
                 {"object_kind": "OBSERVATION", "object_id": "OUT", "role": "TEST_OBSERVATION", "default_action": "REMOVE"},
                 {"object_kind": "TARGET", "object_id": "T_OUT", "role": "TEST_TARGET", "default_action": "REMOVE"}],
    "interfaces": [{"port_id": "IN", "role": "INPUT", "point_mm": [-20.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0], "beam_reference": "SRC"},
                   {"port_id": "OUT", "role": "OUTPUT", "point_mm": [0.0, 20.0, 0.0], "direction": [0.0, 1.0, 0.0]}],
    "requirements": [{"target_id": "T_MON_CENTER", "scope": "CARRY_TO_PARENT"}],
}


def fold_workspace() -> dict[str, Any]:
    reply = json_api.workspace_init(fold_module(), {"module": FOLD_DESCRIPTOR})
    assert reply["ok"], reply
    return reply["workspace"]


def lens_module(focal_length_mm: str = "50") -> dict[str, Any]:
    """An ideal thin lens at the origin (axis +X) with a test laser and a test focal screen."""

    lens = build("lens.ideal_thin", {"focal_length_mm": focal_length_mm, "clear_aperture_diameter_mm": "20"}, model_id="focus-lens")
    f = float(focal_length_mm)
    return project_of([laser(), place(lens, "L1", (0.0, 0.0, 0.0), ALONG_X)], [screen("FOCUS", (f, 0.0, 0.0), ALONG_X, absorbing=True)],
                      [{"target_id": "T_SPOT", "metric": "beam_radius_mm", "observation_id": "FOCUS", "maximum": 0.1}], [lens])


LENS_DESCRIPTOR = {
    "module_id": "focus-50", "title": "Focusing stage",
    "fixtures": [{"object_kind": "COMPONENT", "object_id": "SRC", "role": "TEST_SOURCE", "default_action": "REMOVE"}],
    "requirements": [{"target_id": "T_SPOT", "scope": "CARRY_TO_PARENT"}],
}


def lens_workspace(focal_length_mm: str = "50") -> dict[str, Any]:
    reply = json_api.workspace_init(lens_module(focal_length_mm), {"module": LENS_DESCRIPTOR})
    assert reply["ok"], reply
    return reply["workspace"]


def system_with_laser(position=(-100.0, 0.0, 0.0)) -> dict[str, Any]:
    """A parent workspace holding only the system's own laser (along +X)."""

    reply = json_api.workspace_init(project_of([laser("LASER", position)], [], [], []))
    assert reply["ok"], reply
    return reply["workspace"]


def turn_z(degrees: float) -> tuple[float, float, float, float]:
    """Independent reference quaternion (libm) of a turn about +Z."""

    half = math.radians(degrees) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def ok(reply: dict[str, Any]) -> dict[str, Any]:
    assert reply["ok"], reply
    return reply
