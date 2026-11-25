"""The model-scope checks of an extended emitter read every ray of the sequence, not only the rays kept as whole paths.

Each scene puts a small element where one of the kept rays does not reach it (asserted: no kept path meets it), so the check below can
only come from the rays that are not kept. Expected: the same code the Gaussian beam gets in the same geometry, and the angle of incidence
from the geometry by hand, within the spread the emitter and the element's size allow.
"""

from __future__ import annotations

import math

import numpy as np

from declared_helpers import declared_component, sequence, surface, with_models
from local_helpers import ALONG_X, ALONG_Y, plane, project, source
from os_optics.local import run_optical_project
from test_paraxial_mirrors_compounds import HR, _material


def emitter(rays: int = 4096, seed: int = 5, position=(0.0, 0.0, 0.0)) -> dict:
    return {"instance_id": "EM", "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "e"},
            "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(ALONG_X)},
            "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 633.0, "power_w": 2.0, "emitting_surface": {"shape": "DISC", "diameter_mm": 0.2},
                             "angular_law": "LAMBERTIAN", "sampling": {"sample_count": rays, "seed": seed}}}


def quaternion_of(x_axis, y_axis, z_axis) -> tuple[float, float, float, float]:
    m = np.column_stack([x_axis, y_axis, z_axis])
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0], q[i + 1], q[j + 1], q[k + 1] = (m[k, j] - m[j, k]) / s, 0.25 * s, (m[i, j] + m[j, i]) / s, (m[i, k] + m[k, i]) / s
    return tuple(q)  # type: ignore[return-value]


#: The small mirror of ``_oblique_mirror``: 100 mm from the emitter, its normal 60 deg off the light.
MIRROR_POSE = quaternion_of(np.array([0.0, 0.0, 1.0]), np.cross(np.array([-0.5, math.sqrt(3.0) / 2.0, 0.0]), np.array([0.0, 0.0, 1.0])), np.array([-0.5, math.sqrt(3.0) / 2.0, 0.0]))


def _oblique_mirror(lit, clear: str = "2") -> dict:
    """A mirror whose HR coating is stated up to 45 deg, met at 60 deg; an absorbing plane on the reflected axis."""

    normal = np.array([-0.5, math.sqrt(3.0) / 2.0, 0.0])
    reflected = np.array([1.0, 0.0, 0.0]) - 2.0 * normal * float(np.dot([1.0, 0.0, 0.0], normal))
    across = np.cross(reflected, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    screen = {**plane("D", tuple(np.array([100.0, 0.0, 0.0]) + 80.0 * reflected), quaternion_of(across, np.cross(reflected, across), reflected), radius_mm=40.0), "absorbing": True}
    mirror = {"model_id": "m", "model": sequence([surface(1, "0", clear=clear, traced="REFLECTED", coating_id=HR["coating_id"])], [], edge=str(int(clear) + 1))}
    return with_models(project([lit, declared_component("M", "mirror", "m", (100.0, 0.0, 0.0), MIRROR_POSE)], [screen]), [mirror], None, [HR], design_intent="OPTICAL_SYSTEM")


def _kept_meeting(result, surface_prefix: str) -> int:
    return sum(1 for path in result.paths if any((segment.end_surface_id or "").startswith(surface_prefix) for segment in path.segments))


def test_a_ray_that_is_not_kept_is_checked_against_a_coatings_angles():
    run = run_optical_project(_oblique_mirror(emitter()), catalog=None)
    assert _kept_meeting(run.result, "M/") == 0
    assert run.result.observations[0].sampled_light[0].counted_crossing_count > 0
    (issue,) = [i for i in run.model_scope_issues if i.code == "COATING_AOI_OUTSIDE_RANGE"]
    assert issue.instance_id == "M" and issue.affected_observation_ids == ("D",)
    # 60 deg, within the 2 mm mirror's angular size from 100 mm and the emitter's 0.1 mm radius
    angles = [float(token) for token in issue.message.replace(",", " ").split() if token.replace(".", "", 1).isdigit() and 50.0 < float(token) < 70.0]
    assert angles and all(abs(angle - 60.0) <= 1.5 for angle in angles)


def test_the_answer_does_not_depend_on_which_rays_are_kept():
    """A larger mirror, where a kept ray does meet it: the same code from the same check."""

    kept = run_optical_project(_oblique_mirror(emitter(seed=1), clear="18"), catalog=None)
    assert _kept_meeting(kept.result, "M/") > 0
    assert "COATING_AOI_OUTSIDE_RANGE" in {i.code for i in kept.model_scope_issues}


def test_the_same_geometry_reads_the_same_for_a_gaussian_beam():
    laser = source("S", (0.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5)
    assert "COATING_AOI_OUTSIDE_RANGE" in {i.code for i in run_optical_project(_oblique_mirror(laser), catalog=None).model_scope_issues}


def test_a_mirror_inside_its_stated_angles_stays_in_scope():
    """The same small mirror facing the light at 30 deg: inside the coating's range, for every ray."""

    normal = np.array([-math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0])
    pose = quaternion_of(np.array([0.0, 0.0, 1.0]), np.cross(normal, [0.0, 0.0, 1.0]), normal)
    reflected = np.array([1.0, 0.0, 0.0]) - 2.0 * normal * float(np.dot([1.0, 0.0, 0.0], normal))
    across = np.cross(reflected, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    screen = {**plane("D", tuple(np.array([100.0, 0.0, 0.0]) + 80.0 * reflected), quaternion_of(across, np.cross(reflected, across), reflected), radius_mm=40.0), "absorbing": True}
    mirror = {"model_id": "m", "model": sequence([surface(1, "0", clear="2", traced="REFLECTED", coating_id=HR["coating_id"])], [], edge="3")}
    document = with_models(project([emitter(), declared_component("M", "mirror", "m", (100.0, 0.0, 0.0), pose)], [screen]), [mirror], None, [HR], design_intent="OPTICAL_SYSTEM")
    run = run_optical_project(document, catalog=None)
    assert run.result.observations[0].sampled_light[0].counted_crossing_count > 0
    assert [i.code for i in run.model_scope_issues] == []


def _rotator(valid: tuple[str, str] = ("0", "0.5")) -> dict:
    return {"model_id": "fr", "model": {"model_kind": "FARADAY_ROTATOR", "model_version": "1", "rows": [{"wavelength_nm": "633", "rotation_deg": "30", "transmittance": "1"}],
                                        "interpolation": "POINTS_ONLY", "clear_radius_mm": "1", "edge_radius_mm": "2",
                                        "valid_aoi_deg": {"min_deg": valid[0], "max_deg": valid[1], "status": "SPECIFIED"}, "operating_conditions": "fixture",
                                        "evidence": {"method": "x", "tools": [{"name": "t", "version": "1"}], "source_data_sha256": "ab" * 32, "fit_model": "x",
                                                     "validation_summary": "x", "date": "2026-10-30"}, "status": "SPECIFIED"}}


def test_a_ray_that_is_not_kept_is_checked_against_a_faraday_rotators_angles():
    """Rays through a small rotator 40 mm away, off its axis by up to atan(1 / 40) = 1.4 deg, against a stated 0.5 deg."""

    document = with_models(project([emitter(), declared_component("F", "other_optical", "fr", (40.0, 0.0, 0.0), ALONG_X)], [{**plane("D", (80.0, 0.0, 0.0), radius_mm=40.0), "absorbing": True}]),
                           [_rotator()], None, design_intent="OPTICAL_SYSTEM")
    run = run_optical_project(document, catalog=None)
    assert _kept_meeting(run.result, "F/") == 0
    assert "ELEMENT_AOI_OUTSIDE_RANGE" in {i.code for i in run.model_scope_issues}


def test_a_ray_that_is_not_kept_is_checked_for_crossing_a_body_through_its_edge():
    """A small plate turned edge-on to the light: a ray crosses its body with no traced surface in the way."""

    plate = {"model_id": "p", "model": sequence([surface(1, "0", clear="1.5"), surface(2, "2", clear="1.5")], ["N-BK7"], edge="2")}
    document = with_models(project([emitter(), declared_component("P", "other_optical", "p", (30.0, 0.0, -1.0), ALONG_Y)], [{**plane("D", (80.0, 0.0, 0.0), radius_mm=40.0), "absorbing": True}]),
                           [plate], [_material()], design_intent="OPTICAL_SYSTEM")
    run = run_optical_project(document, catalog=None)
    assert _kept_meeting(run.result, "P/") == 0
    (issue,) = [i for i in run.model_scope_issues if i.code == "RAY_CROSSES_ELEMENT_BODY"]
    assert issue.instance_id == "P" and "rays of EM" in issue.message


def test_a_ray_that_is_not_kept_is_checked_for_leaving_a_body_through_its_edge():
    """A long thin rod on the axis: light entering its front face leaves through the side wall, which the model does not describe."""

    rod = {"model_id": "r", "model": sequence([surface(1, "0", clear="1"), surface(2, "60", clear="1")], ["N-BK7"], edge="1.2")}
    document = with_models(project([emitter(position=(0.0, 0.0, 0.0)), declared_component("R", "other_optical", "r", (30.0, 0.0, 0.0), ALONG_X)], [plane("D", (200.0, 0.0, 0.0), radius_mm=150.0)]),
                           [rod], [_material()], design_intent="OPTICAL_SYSTEM")
    run = run_optical_project(document, catalog=None)
    assert _kept_meeting(run.result, "R/") == 0
    (issue,) = [i for i in run.model_scope_issues if i.code == "RAY_LEFT_ELEMENT_THROUGH_EDGE"]
    assert issue.instance_id == "R" and "rays of EM" in issue.message


def test_a_scope_issue_of_an_emitter_names_the_target_on_the_plane_its_rays_reach():
    document = _oblique_mirror(emitter())
    document["targets"] = [{"target_id": "p", "metric": "power_W", "observation_id": "D", "minimum": 0.0, "unit": "W"}]
    run = run_optical_project(document, catalog=None)
    (issue,) = [i for i in run.model_scope_issues if i.code == "COATING_AOI_OUTSIDE_RANGE"]
    assert issue.affected_target_ids == ("p",)
