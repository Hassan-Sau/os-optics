"""A six-element objective assembled the low-level way: one ``SURFACE_SEQUENCE`` component per element and an
``APERTURE_STOP`` component, each with its own pose along design +X, an image plane, and collimated sources (one per
field angle) 200 mm upstream. The elements come from importing ``fixtures/section/six_singlets.csv``."""

from __future__ import annotations

import math
from pathlib import Path

from os_optics.local import json_api

AXIS_X = (0.5, 0.5, 0.5, 0.5)  # local +Z along design +X, local +X along design +Y
CSV = Path(__file__).parent / "fixtures" / "section" / "six_singlets.csv"
ELEMENTS = ("L1", "L2", "L3", "L4", "L5", "L6")
SOURCE_UPSTREAM_MM = 200.0
PUPIL_X_MM = 32.0  # the fields are aimed at this place on the axis, near the entrance pupil
WAVELENGTH_NM = 587.6
WAIST_MM = 8.0


def _about_z(q, degrees: float):
    """The rotation of ``degrees`` about design +Z applied after ``q`` (Hamilton product, wxyz)."""

    half = math.radians(degrees) / 2.0
    w0, z0 = math.cos(half), math.sin(half)
    w, x, y, z = q
    return (w0 * w - z0 * z, w0 * x - z0 * y, w0 * y + z0 * x, w0 * z + z0 * w)


def source(instance_id: str, field_deg: float = 0.0, waist_mm: float = WAIST_MM) -> dict:
    y = -math.tan(math.radians(field_deg)) * (PUPIL_X_MM + SOURCE_UPSTREAM_MM)
    return {"instance_id": instance_id, "optical_role": "source", "custom_part": {"custom_part_key": "collimated", "description": "collimated"},
            "nominal_pose": {"position_mm": [-SOURCE_UPSTREAM_MM, y, 0.0], "quaternion_wxyz": list(_about_z(AXIS_X, field_deg))},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM, "power_w": 0.001, "waist_radius_mm": waist_mm,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}


def observation(observation_id: str, x_mm: float, radius_mm: float = 25.0) -> dict:
    return {"observation_id": observation_id, "pose": {"position_mm": [x_mm, 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)},
            "aperture": {"shape": "circle", "radius_mm": radius_mm}, "absorbing": False, "medium": "air"}


def six_singlet_project(fields_deg=(0.0,), *, waist_mm: float = WAIST_MM, order=None, extra_observations=()) -> dict:
    """The Project document; ``order``: the element ids in the order the components are listed (the stop stays between)."""

    imported = json_api.import_prescription(CSV.read_text(encoding="utf-8"), {"model_id": "gauss"})
    assert imported["ok"], imported
    models = {m["model_id"]: m for m in imported["optical_models"]}
    members = {m["member_id"]: m for m in models["gauss"]["model"]["members"]}
    components = [source("SRC" if len(fields_deg) == 1 else f"FIELD{k}", field, waist_mm) for k, field in enumerate(fields_deg)]
    for member_id in order or members:
        member = members[member_id]
        role = "aperture" if member_id == "STOP" else "lens"
        components.append({"instance_id": member_id, "optical_role": role, "optical_model_id": member["model_id"],
                           "custom_part": {"custom_part_key": member_id.lower(), "description": member_id},
                           "nominal_pose": {"position_mm": [member["pose"]["position_mm"][2], 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)}})
    image_x = float(imported["report"]["image_plane_z_mm"])
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": components,
            "optical_models": [m for k, m in models.items() if k != "gauss"], "model_materials": imported["model_materials"],
            "model_coatings": imported["model_coatings"],
            "observations": [observation("image", image_x), *extra_observations],
            "targets": [{"target_id": "p", "metric": "power_W", "observation_id": "image", "minimum": 1e-12, "unit": "W"}]}
