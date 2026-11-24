"""The geometric pupil-sample power basis: a beam without a valid Gaussian state whose pupil rays a clear aperture cuts.

Before, such a beam ended at that aperture (``TRUNCATED_BEAM_NOT_FOLLOWED``) and every plane behind it lost the field. Now
the aperture passes the share of the live, equal-power pupil samples that pass it; the rays are traced on, the cut power is
booked ``APERTURE_TRUNCATION`` and handed on as untraced light, the power carries a sampling resolution
(``ArrivalResult.power_sampling_resolution_W``) and a power target within it is not decided. The Gaussian beam metrics stay
unavailable. A beam with a valid Gaussian state is not touched (the request fixtures' snapshots pin that)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from helpers import assert_power_not_computed, one_arrival
from os_optics import ValidationRequest
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local import json_api, run_optical_project
from os_optics.validation import validate

SECTION = Path(__file__).parent / "fixtures" / "section"
AXIS_X = (0.5, 0.5, 0.5, 0.5)  # local +Z along design +X
FIELDS_DEG = (0.0, 8.5, 12.0)
UPSTREAM_MM = 100.0
SOURCE_W = 1e-3


# ------------------------------------------------------------------------------------------------- analytic: two stops


def _two_stops(a1: float, a2: float, rings: int = 8, per_ring: int = 12, *, targets: tuple[Target, ...] = ()) -> ProjectSnapshot:
    """A collimated 1 mW beam (w = 1 mm, z_R 5 m) through a stop of radius ``a1`` w (13.5 % cut at 1 w: the Gaussian state is
    invalid behind it) and then a concentric stop of radius ``a2`` w < ``a1``: what passes both is ``P (1 - exp(-2 a2^2))``."""
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0), power_W=SOURCE_W)
    laser = laser.model_copy(update={"source": laser.source.model_copy(update={"pupil_ring_count": rings, "pupil_points_per_ring": per_ring})})
    return ProjectSnapshot(name="two stops", components=(
        laser, B.aperture_stop("S1", position=(10, 0, 20), axis=(1, 0, 0), radius_mm=a1),
        B.aperture_stop("S2", position=(20, 0, 20), axis=(1, 0, 0), radius_mm=a2)),
        observations=(B.observation_plane("after", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=50.0),), targets=targets)


def _run(project: ProjectSnapshot):
    return validate(ValidationRequest(project=project))


@pytest.mark.parametrize("a2", (0.2, 0.35, 0.5, 0.65, 0.8))
def test_the_sampled_power_behind_a_second_stop_lies_within_its_resolution_of_the_gaussian_integral(a2: float):
    result = _run(_two_stops(1.0, a2))
    arrival = one_arrival(result, "after")
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * a2 * a2))
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
    assert abs(arrival.beam.power_W - truth) <= arrival.power_sampling_resolution_W
    assert arrival.power_sampling_resolution_W > 0.0
    codes = [i.code.value for i in result.issues]
    assert "GEOMETRIC_POWER_BASIS" in codes and "TRUNCATED_BEAM_NOT_FOLLOWED" not in codes
    assert result.power_budget.conserved


def test_the_sampled_power_converges_with_the_pupil_samples():
    """The resolution halves with each doubling of rings and points, and keeps covering the error, which shrinks."""
    a2 = 0.5
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * a2 * a2))
    rows = []
    for rings, per_ring in ((8, 12), (16, 24), (32, 48)):
        arrival = one_arrival(_run(_two_stops(1.0, a2, rings, per_ring)), "after")
        rows.append((abs(arrival.beam.power_W - truth), arrival.power_sampling_resolution_W))
    assert all(error <= resolution for error, resolution in rows)
    assert rows[0][1] > rows[1][1] > rows[2][1]
    assert rows[2][0] < rows[0][0]


def test_the_cut_power_is_booked_and_the_ledger_closes():
    """The first stop takes its exact Gaussian share; the second, the cut samples' share of what reaches it: the two
    APERTURE_TRUNCATION entries and the detected-free transmitted power make up the source power."""
    result = _run(_two_stops(1.0, 0.5))
    budget = result.power_budget
    arrival = one_arrival(result, "after")
    cuts = {e.element_id: e.power_W for e in budget.entries if e.kind == "APERTURE_TRUNCATION"}
    first, second = cuts["S1"], cuts["S2"]
    assert first == pytest.approx(SOURCE_W * math.exp(-2.0), rel=1e-5)  # (the first stop: the Gaussian integral)
    # (the second: rings 3-6 of the 84 live samples, of the power that reaches it)
    assert second == pytest.approx((SOURCE_W - first) * 48 / 84, rel=1e-12)
    assert arrival.beam.power_W + first + second == pytest.approx(SOURCE_W, rel=1e-12)
    assert budget.conserved


@pytest.mark.parametrize(("minimum", "assessment"), ((0.1e-3, "MET"), (0.38e-3, "INDETERMINATE"), (0.7e-3, "UNMET")))
def test_a_power_target_within_the_sampling_resolution_is_not_decided(minimum: float, assessment: str):
    target = Target(target_id="p", metric="power_W", observation_id="after", minimum=minimum, unit="W")
    result = _run(_two_stops(1.0, 0.5, targets=(target,)))
    (got,) = result.targets
    assert got.assessment.value == assessment
    assert "GEOMETRIC_POWER_BASIS" in got.issue_codes


def test_a_stop_that_cuts_every_pupil_sample_leaves_the_power_not_computed():
    """No sample passes while the chief ray does: the power that passes has no sample to share it, and is not computed. The
    chief ray goes on alone: the plane behind records the arrival with no power, at most what reached the second stop,
    ``P (1 - exp(-2))`` (the first stop at one beam radius)."""
    result = _run(_two_stops(1.0, 0.1))
    _obs, arrival = assert_power_not_computed(result, "after")
    w2 = 1.0 + (10.0 / (math.pi / 633e-6)) ** 2  # (the beam radius squared at the first stop, 10 mm from the 1 mm waist)
    assert arrival.power_upper_bound_W == pytest.approx(SOURCE_W * (1.0 - math.exp(-2.0 / w2)), rel=1e-9)
    assert arrival.pupil_rays_arrived == 0 and arrival.beam.position_mm[0] == 40.0
    (issue,) = [i for i in result.issues if i.numerical_context.get("power_upper_bound_W") is not None]
    assert (issue.code.value, issue.location, issue.numerical_context["pupil_rays_cut"]) == ("GAUSSIAN_SEVERE_TRUNCATION", "surface:S2/stop", 84)


def test_a_beam_with_a_valid_gaussian_state_keeps_the_gaussian_power():
    """A mild first stop (2 %, Gaussian state valid) and a second one: the exact Gaussian fraction, no sampling."""
    result = _run(_two_stops(1.4, 1.2))
    arrival = one_arrival(result, "after")
    assert arrival.power_basis is None and arrival.power_sampling_resolution_W is None
    assert "power_basis" not in arrival.model_dump(mode="json")


# ---------------------------------------------------------------------------------------- fast objectives, every field


def _fast_project(csv_name: str) -> dict:
    text = (SECTION / csv_name).read_text(encoding="utf-8")
    imported = json_api.import_prescription(text, {"model_id": "fast"})
    assert imported["ok"], imported
    # Inline fast-objective prescription media are ideal test glass: the geometric-power test
    # does not infer zero absorption from a material source that has not supplied loss data.
    for material in imported["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    models = {m["model_id"]: m for m in imported["optical_models"]}
    members = {m["member_id"]: m for m in models["fast"]["model"]["members"]}
    # the beam's 1/e^2 radius is the stop's semi-diameter: the stop cuts 13.5 % of it (GAUSSIAN_SEVERE_TRUNCATION)
    waist = next(float(row.split(",")[4]) for row in text.splitlines() if row.startswith("STO,"))
    components = []
    for k, field in enumerate(FIELDS_DEG):
        half = math.radians(field) / 2.0
        w0, z0 = math.cos(half), math.sin(half)
        w, x, y, z = AXIS_X
        q = (w0 * w - z0 * z, w0 * x - z0 * y, w0 * y + z0 * x, w0 * z + z0 * w)
        components.append({
            "instance_id": f"FIELD{k}", "optical_role": "source", "custom_part": {"custom_part_key": "collimated", "description": "field"},
            "nominal_pose": {"position_mm": [-UPSTREAM_MM, -math.tan(math.radians(field)) * UPSTREAM_MM, 0.0], "quaternion_wxyz": list(q)},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 587.6, "power_w": SOURCE_W, "waist_radius_mm": waist,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}})
    for member_id, member in members.items():
        components.append({"instance_id": member_id, "optical_role": "aperture" if member_id == "STOP" else "lens",
                           "optical_model_id": member["model_id"], "custom_part": {"custom_part_key": member_id.lower(), "description": member_id},
                           "nominal_pose": {"position_mm": [member["pose"]["position_mm"][2], 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)}})
    image_x = float(imported["report"]["image_plane_z_mm"])
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": components,
            "optical_models": [m for key, m in models.items() if key != "fast"], "model_materials": imported["model_materials"],
            "model_coatings": imported["model_coatings"],
            "observations": [{"observation_id": "image", "pose": {"position_mm": [image_x, 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)},
                              "aperture": {"shape": "circle", "radius_mm": 40.0}, "absorbing": True, "medium": "air"}],
            "targets": [{"target_id": f"p{k}", "metric": "power_W", "observation_id": "image", "source_id": f"FIELD{k}",
                         "minimum": 1e-5, "unit": "W"} for k in range(len(FIELDS_DEG))]}


@pytest.fixture(scope="module", params=("fast_f1p4.csv", "fast_f1p2.csv"))
def fast_run(request):
    return request.param, run_optical_project(_fast_project(request.param))


def test_a_fast_objective_reports_every_field_with_gaussian_metrics_unavailable(fast_run):
    """F/1.4 and F/1.2 front-stop objectives, fields 0 / 8.5 / 12 deg, the beam filling the stop (13.5 % cut there): every
    field reaches the image with its geometric spot, angle of incidence and power; the Gaussian metrics carry their reason;
    the field a later clear aperture cuts carries the geometric power basis (before, it ended there with no arrival)."""
    name, run = fast_run
    result = run.result
    (image,) = result.observations
    by_field = {a.beam.source_id: a for a in image.arrivals}
    assert set(by_field) == {f"FIELD{k}" for k in range(len(FIELDS_DEG))}, name
    for arrival in by_field.values():
        assert arrival.pupil_rays_arrived > 0 and arrival.pupil_rms_radius_mm is not None
        assert math.isfinite(arrival.aoi_deg) and arrival.beam.power_W > 0.0
        assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
        assert arrival.beam.gaussian.reason is not None
    geometric = [a for a in by_field.values() if a.power_basis == "GEOMETRIC_PUPIL_SAMPLES"]
    assert geometric and all(0.0 < a.power_sampling_resolution_W < a.beam.power_W for a in geometric)
    assert by_field["FIELD2"].power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    codes = [i.code.value for i in result.issues]
    assert "GEOMETRIC_POWER_BASIS" in codes and "TRUNCATED_BEAM_NOT_FOLLOWED" not in [p.termination for p in result.paths]
    assert all(t.value is not None for t in result.targets)


def test_a_fast_objective_power_budget_reconciles_to_the_source_power(fast_run):
    """Per line, the detected, truncated, untraced-reflection and budget sinks make up the launched power within the
    ledger's tolerance; the cut part is booked APERTURE_TRUNCATION, and outside the glass it is also recorded as untraced
    light going on."""
    _name, run = fast_run
    budget = run.result.power_budget
    assert budget.conserved and abs(budget.residual_W) <= budget.tolerance_W
    assert budget.total_input_W == pytest.approx(len(FIELDS_DEG) * SOURCE_W, rel=1e-12)
    assert budget.total_sinks["APERTURE_TRUNCATION"] > 0.0
    for line in budget.lines:
        assert line.conserved


def test_a_fast_objective_ray_paths_reach_every_surface_for_every_field(fast_run):
    """The per-ray paths (spot, angles, aperture use at each surface) exist for each field from the stop to the image:
    every field's chief ray meets every lens surface and is detected, and pupil rays reach the image with it."""
    name, run = fast_run
    lens_surfaces = {s for p in run.ray_paths.paths for s in p.chief_surface_ids if s is not None and s.startswith("L")}
    roots = [p for p in run.ray_paths.paths if p.parent_path_id is None]
    assert {p.source_id for p in roots} == {f"FIELD{k}" for k in range(len(FIELDS_DEG))}, name
    for path in roots:
        assert path.termination == "DETECTED"
        assert lens_surfaces <= set(path.chief_surface_ids)
        assert sum(ray.end_surface_id == "observation/image" for ray in path.pupil_rays) > 0
