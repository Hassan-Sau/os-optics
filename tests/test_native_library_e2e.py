"""End-to-end: Projects that use native component models run through the local Engine path (one physics path with
catalog optics). These need the runtime to lower declared optical models; on a runtime that does not advertise
``project_features: ["optical_models", ...]`` the module is skipped.

Independent references: the singlet's focus from its own paraxial block (checked against an independent y-nu trace in
``test_native_library.py``), Gaussian-beam propagation (ABCD on the complex q parameter, written here) for the telescope,
and the clipped fraction of a Gaussian through a slit (``erf``) / a large opening (no loss).
"""

from __future__ import annotations

import math

import pytest

from os_optics.local import json_api, run_optical_project
from os_optics.native_library import add_to_project, build_native_model

pytestmark = pytest.mark.skipif("optical_models" not in json_api.capabilities().get("project_features", []),
                                reason="this runtime does not lower Project-declared optical models yet")

AXIS_X = [0.5, 0.5, 0.5, 0.5]  # local +Z along global +X (the reference designs' convention)
WAVELENGTH_NM = 780.0
WAIST_MM = 1.0


def _project(source_x: float = -200.0) -> dict:
    return {
        "schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
        "components": [{
            "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
            "nominal_pose": {"position_mm": [source_x, 0.0, 0.0], "quaternion_wxyz": AXIS_X},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM, "power_w": 0.001, "waist_radius_mm": WAIST_MM,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}},
        }],
    }


def _observe(document: dict, observation_id: str, x: float, metrics: list[tuple[str, str]]) -> dict:
    document.setdefault("observations", []).append({
        "observation_id": observation_id, "pose": {"position_mm": [x, 0.0, 0.0], "quaternion_wxyz": AXIS_X},
        "aperture": {"shape": "circle", "radius_mm": 25.0}, "absorbing": False, "medium": "air"})
    for target_id, metric in metrics:
        document.setdefault("targets", []).append({"target_id": target_id, "metric": metric, "observation_id": observation_id,
                                                   "minimum": -1e9, "unit": json_api.capabilities()["target_metrics"][metric]})
    return document


def _values(document: dict) -> dict[str, float | None]:
    run = run_optical_project(document)
    return {t.target_id: t.value for t in run.result.targets}


def test_native_singlet_focuses_at_its_paraxial_back_focus():
    build = build_native_model("lens.singlet", {"focal_length_mm": "100", "design_wavelength_nm": str(WAVELENGTH_NM), "material": "N-BK7",
                                                "diameter_mm": "25.4", "form": "BEST_FORM"})
    paraxial = build.optical_model.model.paraxial
    last_vertex = float(build.optical_model.model.surfaces[-1].vertex_z_mm)
    document = add_to_project(_project(), build, instance_id="L1", quaternion_wxyz=AXIS_X)
    focus_x = last_vertex + float(paraxial.back_focal_length_mm)
    document = _observe(document, "focus", focus_x, [("waist", "waist_distance_x_mm"), ("radius", "beam_radius_mm")])
    values = _values(document)
    # Gaussian waist of a 1 mm beam focused by f = 100 mm sits within ~f (f / z_R)^2 of the geometric focus (z_R = 4 m)
    assert abs(values["waist"]) < 0.1
    assert values["radius"] == pytest.approx(WAVELENGTH_NM * 1e-6 * 100.0 / (math.pi * WAIST_MM), rel=0.05)


def _q_propagate(q: complex, matrix) -> complex:
    (a, b), (c, d) = matrix
    return (a * q + b) / (c * q + d)


def test_native_galilean_telescope_expands_the_beam_by_its_magnification():
    build = build_native_model("compound.galilean_telescope", {
        "lens1": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "-25", "clear_aperture_diameter_mm": "10"}},
        "lens2": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "100", "clear_aperture_diameter_mm": "40"}}})
    document = add_to_project(_project(-100.0), build, instance_id="EXPANDER", quaternion_wxyz=AXIS_X)
    after = 75.0 + 50.0
    document = _observe(document, "out", after, [("radius", "beam_radius_mm")])
    values = _values(document)
    # independent Gaussian reference: q at the source waist, through free space, f1, gap 75 mm, f2, free space
    wavelength_mm = WAVELENGTH_NM * 1e-6
    q = complex(0.0, math.pi * WAIST_MM ** 2 / wavelength_mm)
    for matrix in (((1, 100.0), (0, 1)), ((1, 0), (1 / 25.0, 1)), ((1, 75.0), (0, 1)), ((1, 0), (-1 / 100.0, 1)), ((1, 50.0), (0, 1))):
        q = _q_propagate(q, matrix)
    radius = math.sqrt(-wavelength_mm / (math.pi * (1 / q).imag))
    assert values["radius"] == pytest.approx(radius, rel=0.01)
    assert values["radius"] == pytest.approx(4.0 * WAIST_MM, rel=0.02)


def test_native_stops_pass_or_clip_the_beam_geometrically():
    def power_after(class_id: str, parameters: dict) -> float:
        build = build_native_model(class_id, parameters)
        document = add_to_project(_project(-50.0), build, instance_id="STOP", quaternion_wxyz=AXIS_X)
        document = _observe(document, "after", 20.0, [("power", "power_W")])
        return _values(document)["power"] or 0.0

    assert power_after("aperture.circular", {"opening_diameter_mm": "12"}) == pytest.approx(0.001, rel=1e-6)
    clipped = power_after("aperture.slit", {"slit_width_mm": "0.4", "slit_height_mm": "20"})
    # a Gaussian of radius w through a slit of half width a: erf(sqrt(2) a / w); w ~ 1 mm at 50 mm from a 1 mm waist
    expected = math.erf(math.sqrt(2.0) * 0.2 / WAIST_MM) * 0.001
    assert 0.0 < clipped < 0.5 * 0.001
    assert clipped == pytest.approx(expected, rel=0.25)
