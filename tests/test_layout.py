"""Layout (2D/2.5D, Shapely): rotated footprints, keepout, physical intersection, plate boundary, beam envelope, optical height,
moving envelope, tool / cable regions. Shapely does 2D booleans only; references are closed-form geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from os_optics import builders as B
from os_optics.contracts import (
    ActuatorSpec, BaseplateSnapshot, EngineeringComponentSnapshot, EngineeringEnvelopeSnapshot, PolygonRegion, Pose3D, ProjectSnapshot,
)
from os_optics.layout.geometry import sweep_translation

from helpers import issue_codes, run


def _sq(instance_id, x, y, *, size=10.0, yaw=0.0, z=(0.0, 20.0), **env):
    return EngineeringComponentSnapshot(
        instance_id=instance_id, pose=Pose3D.from_yaw_deg((x, y, 0.0), yaw),
        envelope=EngineeringEnvelopeSnapshot(footprint_mm=((-size / 2, -size / 2), (size / 2, -size / 2), (size / 2, size / 2), (-size / 2, size / 2)),
                                             z_range_mm=z, **env),
    )


def layout(*comps, plate=None, **kw):
    res = run(ProjectSnapshot(components=tuple(comps), baseplate=plate, **kw), layout=True)
    return res.layout, res


def kinds(lay):
    return [i.kind for i in lay.issues if i.status.value != "PASS"]


# ---- rotated footprint ---------------------------------------------------------------------------------------------------------
def test_rotated_footprint_bounds_45_degrees():
    lay, _ = layout(_sq("A", 0.0, 0.0, yaw=45.0))
    minx, miny, maxx, maxy = lay.rotated_bounds["A"]
    h = 10.0 * math.sqrt(2.0) / 2.0  # half diagonal of a 10 mm square
    assert (minx, miny, maxx, maxy) == pytest.approx((-h, -h, h, h), abs=1e-12)


def test_rotated_footprint_is_used_not_the_unrotated_aabb():
    """B (yaw 45) at (10, 10): its AABB [2.93, 17.07]^2 overlaps A's AABB [-5, 5]^2 but the true polygons are 2.071 mm apart."""
    h = 5.0 * math.sqrt(2.0)
    expected_gap = (10.0 - h) / math.sqrt(2.0)
    lay, _ = layout(_sq("A", 0.0, 0.0), _sq("B", 10.0, 10.0, yaw=45.0))
    assert "PHYSICAL_INTERSECTION" not in kinds(lay)
    assert lay.min_clearance_mm == pytest.approx(expected_gap, abs=1e-12) and lay.min_clearance_pair == ("A", "B")
    # an AABB check (which would report a collision) is contradicted by the actual rotated geometry
    a_box, b_box = lay.rotated_bounds["A"], lay.rotated_bounds["B"]
    assert b_box[0] < a_box[2] and b_box[1] < a_box[3]  # AABBs do overlap
    # requiring more clearance than the true gap gives a keepout violation with the exact deficit
    lay2, _ = layout(_sq("A", 0.0, 0.0, required_clearance_mm=3.0), _sq("B", 10.0, 10.0, yaw=45.0))
    issue = next(i for i in lay2.issues if i.kind == "KEEPOUT_VIOLATION")
    assert issue.clearance_mm == pytest.approx(expected_gap, abs=1e-12) and issue.required_clearance_mm == 3.0
    assert issue.deficit_mm == pytest.approx(3.0 - expected_gap, abs=1e-12)


def test_keepout_deficit_matches_plan_example_and_suggested_move_resolves_it():
    """Clearance 17, required 18 -> deficit 1 with structured suggested moves."""
    # A at 0, B at 27: gap = 27 - 10 = 17 mm, required 18 mm
    lay, res = layout(_sq("A", 0.0, 0.0, required_clearance_mm=18.0), _sq("B", 27.0, 0.0))
    issue = next(i for i in lay.issues if i.kind == "KEEPOUT_VIOLATION")
    assert (issue.clearance_mm, issue.required_clearance_mm, issue.deficit_mm) == pytest.approx((17.0, 18.0, 1.0), abs=1e-9)
    assert set(issue.device_ids) == {"A", "B"}
    assert issue.contact_or_nearest_points == ((5.0, 0.0), (22.0, 0.0)) or issue.contact_or_nearest_points[0][0] == pytest.approx(5.0)
    assert set(issue.rotated_bounds) == {"A", "B"}
    # the layout's own outcome; it reaches the run's status only through a declared layout target
    assert lay.status.value == "FAIL" and res.status.value == "NOT_APPLICABLE"  # no optics to assess
    # applying the first suggested move (move B away) reaches the required clearance
    (dev, (dx, dy)), = issue.suggested_moves[0].items()
    assert dev == "B" and (dx, dy) == pytest.approx((1.0, 0.0), abs=1e-5)
    moved, _ = layout(_sq("A", 0.0, 0.0, required_clearance_mm=18.0), _sq("B", 27.0 + dx, dy))
    assert "KEEPOUT_VIOLATION" not in kinds(moved)
    # the alternative suggestion moves A instead
    (dev2, (dx2, dy2)), = issue.suggested_moves[1].items()
    assert dev2 == "A" and dx2 == pytest.approx(-1.0, abs=1e-5)


def test_physical_intersection_reports_overlap_and_moves():
    lay, res = layout(_sq("A", 0.0, 0.0), _sq("B", 6.0, 0.0))
    issue = next(i for i in lay.issues if i.kind == "PHYSICAL_INTERSECTION")
    assert issue.clearance_mm == 0.0 and "overlap" in issue.message and issue.suggested_moves
    (dev, (dx, dy)), = issue.suggested_moves[0].items()
    assert dx == pytest.approx(4.0, abs=1e-5)  # separate to touching: 10 - 6
    assert lay.status.value == "FAIL" and res.status.value == "NOT_APPLICABLE"  # no optics to assess


def test_a_layout_finding_reaches_the_run_status_through_a_declared_layout_target():
    """Passing optics with a blocked tool-access region: the run passes and ``layout.status`` fails; with a target of at
    most 0 layout violations the target and the run fail."""
    from os_optics.contracts import Target

    tool = PolygonRegion(region_id="hexkey", polygon_mm=((-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)), z_range_mm=(20.0, 60.0))
    comps = (_sq("Mount", 0.0, 0.0, z=(0.0, 25.0), tool_access=(tool,)), _sq("Above", 1.0, 1.0, size=2.0, z=(30.0, 50.0)))
    lay, res = layout(*comps)
    assert "TOOL_ACCESS_BLOCKED" in kinds(lay) and lay.status.value == "FAIL"
    assert res.status.value == "NOT_APPLICABLE"  # (no optics: nothing else to assess)
    _, gated = layout(*comps, targets=(Target(target_id="v", metric="layout_violation_count", target_value=0.0, tolerance_abs=0.0),))
    assert [t.status.value for t in gated.targets] == ["FAIL"] and gated.status.value == "FAIL"


def test_touching_footprints_are_not_an_intersection():
    lay, _ = layout(_sq("A", 0.0, 0.0), _sq("B", 10.0, 0.0))
    assert "PHYSICAL_INTERSECTION" not in kinds(lay)


def test_height_layering_2p5d():
    lay, _ = layout(_sq("A", 0.0, 0.0, z=(0.0, 20.0)), _sq("B", 2.0, 0.0, z=(25.0, 40.0)))  # overlapping footprints, disjoint heights
    assert kinds(lay) == []
    lay2, _ = layout(_sq("A", 0.0, 0.0, z=(0.0, 20.0)), _sq("B", 2.0, 0.0, z=(15.0, 40.0)))
    assert "PHYSICAL_INTERSECTION" in kinds(lay2)


def test_keepout_margin_dilation():
    lay, _ = layout(_sq("A", 0.0, 0.0, keepout_margin_mm=5.0), _sq("B", 13.0, 0.0))  # gap 3 < 5
    issue = next(i for i in lay.issues if i.kind == "KEEPOUT_VIOLATION")
    assert issue.deficit_mm == pytest.approx(2.0, abs=1e-9)


def test_explicit_keepout_polygon():
    region = PolygonRegion(region_id="swing", polygon_mm=((5.0, -5.0), (25.0, -5.0), (25.0, 5.0), (5.0, 5.0)), z_range_mm=(0.0, 20.0))
    lay, _ = layout(_sq("A", 0.0, 0.0, keepout_polygons=(region,)), _sq("B", 20.0, 0.0, size=4.0))
    assert "KEEPOUT_VIOLATION" in kinds(lay)
    lay2, _ = layout(_sq("A", 0.0, 0.0, keepout_polygons=(region,)), _sq("B", 20.0, 30.0, size=4.0))
    assert kinds(lay2) == []


def test_invalid_and_degenerate_footprints():
    bow = EngineeringComponentSnapshot(
        instance_id="Bow", envelope=EngineeringEnvelopeSnapshot(footprint_mm=((0, 0), (10, 10), (10, 0), (0, 10)), z_range_mm=(0, 5)))
    flat = EngineeringComponentSnapshot(
        instance_id="Flat", envelope=EngineeringEnvelopeSnapshot(footprint_mm=((0, 0), (10, 0), (20, 0)), z_range_mm=(0, 5)))
    lay, _ = layout(bow, flat, _sq("A", 50.0, 50.0))
    bad = [i for i in lay.issues if i.kind == "INVALID_FOOTPRINT"]
    assert {i.device_ids[0] for i in bad} == {"Bow", "Flat"} and all(i.status.value == "FAIL" for i in bad)


def test_tilted_pose_footprint_is_projected():
    """A component rolled 60 deg about x has its footprint y-extent foreshortened by cos(60) in plan view."""
    comp = EngineeringComponentSnapshot(
        instance_id="T", pose=Pose3D.from_euler_zyx_deg((0, 0, 0), roll_deg=60.0),
        envelope=EngineeringEnvelopeSnapshot(footprint_mm=((-5, -5), (5, -5), (5, 5), (-5, 5)), z_range_mm=(0, 1)))
    lay, _ = layout(comp)
    minx, miny, maxx, maxy = lay.rotated_bounds["T"]
    assert (maxx - minx, maxy - miny) == pytest.approx((10.0, 10.0 * math.cos(math.radians(60.0))), abs=1e-12)


# ---- plate boundary ---------------------------------------------------------------------------------------------------------------
def test_plate_boundary_and_edge_margin():
    plate = BaseplateSnapshot(outline_mm=((0, 0), (100, 0), (100, 100), (0, 100)), edge_margin_mm=5.0)
    inside, _ = layout(_sq("A", 50.0, 50.0), plate=plate)
    assert kinds(inside) == []
    near_edge, _ = layout(_sq("A", 9.0, 50.0), plate=plate)  # left edge at x = 4 < margin 5
    issue = next(i for i in near_edge.issues if i.kind == "PLATE_BOUNDARY_VIOLATION")
    assert issue.device_ids == ("A",) and issue.suggested_moves
    (dev, (dx, dy)), = issue.suggested_moves[0].items()
    assert dx > 0
    outside, _ = layout(_sq("A", 98.0, 50.0), plate=BaseplateSnapshot(outline_mm=plate.outline_mm))
    assert "PLATE_BOUNDARY_VIOLATION" in kinds(outside)


# ---- beam envelope / optical height ------------------------------------------------------------------------------------------------
def _beam_scene(obstacle_y):
    laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0),
                    envelope=B.rect_envelope(20, 20, 30, base_z_mm=0.0, optical=True))
    stop = B.aperture_stop("End", position=(200, 0, 20), axis=(1, 0, 0), radius_mm=30.0, envelope=B.rect_envelope(6, 60, 40, optical=True))
    obstacle = EngineeringComponentSnapshot(
        instance_id="Post", pose=Pose3D(position_mm=(100.0, obstacle_y, 0.0)),
        envelope=EngineeringEnvelopeSnapshot(footprint_mm=((-3, -3), (3, -3), (3, 3), (-3, 3)), z_range_mm=(0.0, 40.0)))
    return laser, stop, obstacle


def test_beam_envelope_blocked_by_element_in_the_beam_path():
    zr = math.pi * 1.0**2 / 532e-6
    w = math.sqrt(1 + (100.0 / zr) ** 2)  # beam radius ~ 1.0 at the obstacle
    k = 2.0
    laser, stop, obstacle = _beam_scene(obstacle_y=0.0)
    lay, res = layout(laser, stop, obstacle)
    assert "BEAM_ENVELOPE_BLOCKED" in kinds(lay)
    # obstacle just outside k*w: half-width 3 -> nearest face at y_obstacle - 3; free when y - 3 > k*w
    y_free = 3.0 + k * 1.02 + 0.05
    lay2, _ = layout(*_beam_scene(obstacle_y=y_free))
    assert "BEAM_ENVELOPE_BLOCKED" not in kinds(lay2)
    y_touch = 3.0 + k * 1.02 - 0.5
    lay3, _ = layout(*_beam_scene(obstacle_y=y_touch))
    assert "BEAM_ENVELOPE_BLOCKED" in kinds(lay3)


def test_elements_the_beam_interacts_with_are_not_reported_as_blockers():
    laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), envelope=B.rect_envelope(20, 20, 30, optical=True))
    lens = B.thin_lens("Lens", focal_length_mm=100.0, clear_radius_mm=10.0, position=(50, 0, 20), axis=(1, 0, 0), envelope=B.rect_envelope(8, 30, 40, optical=True))
    lay, _ = layout(laser, lens)
    assert "BEAM_ENVELOPE_BLOCKED" not in kinds(lay)


def test_optical_height_mismatch():
    def scene(nominal_z):
        laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0))
        env = EngineeringEnvelopeSnapshot(frame_pose=Pose3D.plan_frame_for_optical_axis(),
                                          footprint_mm=((-4, -15), (4, -15), (4, 15), (-4, 15)), z_range_mm=(0, 40),
                                          optical_center_local_mm=(0.0, 0.0, 0.0), optical_height_tolerance_mm=0.5)
        lens = B.thin_lens("Lens", focal_length_mm=100.0, clear_radius_mm=10.0, position=(50, 0, nominal_z), axis=(1, 0, 0), envelope=env)
        return laser, lens

    ok, _ = layout(*scene(20.0))
    assert "OPTICAL_HEIGHT_MISMATCH" not in kinds(ok)
    bad, _ = layout(*scene(21.5))
    issue = next(i for i in bad.issues if i.kind == "OPTICAL_HEIGHT_MISMATCH")
    assert issue.clearance_mm == pytest.approx(1.5, abs=1e-9) and issue.required_clearance_mm == 0.5 and issue.device_ids == ("Lens",)


# ---- moving envelope ---------------------------------------------------------------------------------------------------------------
def test_translation_sweep_area_matches_closed_form():
    sq = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    swept = sweep_translation(sq, 0.0, 0.0, 12.0, 0.0)
    assert swept.area == pytest.approx(10.0 * 10.0 + 10.0 * 12.0, abs=1e-9)  # w^2 + w L
    diag = sweep_translation(sq, 0.0, 0.0, 6.0, 8.0)  # slanted sweep, length 10: area = w^2 + w*L*|sin/cos combination|
    assert diag.area == pytest.approx(100.0 + 10.0 * (6.0 + 8.0), abs=1e-9)  # for an axis-aligned square: w^2 + w(|dx| + |dy|)


def test_moving_envelope_collision_and_clear_cases():
    stage_act = ActuatorSpec(actuator_id="x", dof="translate", axis_local=(1.0, 0.0, 0.0), travel_min=-2.0, travel_max=15.0)
    stage = _sq("Stage", 0.0, 0.0)
    stage = stage.model_copy(update={"actuators": (stage_act,)})
    near = _sq("Near", 25.0, 0.0)  # static gap 15 > 0 but sweep to +15 -> x in [10, 20] hits Near's [20, 30]? touch at 20: keep clear
    lay_touch, _ = layout(stage, near)
    assert "MOVING_ENVELOPE_VIOLATION" not in kinds(lay_touch)  # sweep reaches x = 20 exactly: touching, not overlapping
    hit = _sq("Near", 23.0, 0.0)
    lay_hit, _ = layout(stage, hit)
    assert "MOVING_ENVELOPE_VIOLATION" in kinds(lay_hit)
    assert {"Stage", "Near"} <= set(next(i for i in lay_hit.issues if i.kind == "MOVING_ENVELOPE_VIOLATION").device_ids) | {"Near"}
    plate = BaseplateSnapshot(outline_mm=((-20, -20), (18, -20), (18, 20), (-20, 20)))
    lay_plate, _ = layout(stage, plate=plate)
    assert "MOVING_ENVELOPE_VIOLATION" in kinds(lay_plate)  # sweeps beyond the plate edge at x = 18


def test_rotation_sweep_envelope():
    act = ActuatorSpec(actuator_id="rz", dof="rotate", axis_local=(0.0, 0.0, 1.0), pivot_local_mm=(0.0, 0.0, 0.0), travel_min=-45.0, travel_max=45.0)
    rot = _sq("R", 0.0, 0.0).model_copy(update={"actuators": (act,)})
    # the rotating square's corner reaches 5*sqrt(2) = 7.07 mm from the pivot (at +-45 deg)
    lay_ok, _ = layout(rot, _sq("N", 20.0, 0.0))
    assert "MOVING_ENVELOPE_VIOLATION" not in kinds(lay_ok)  # reach 7.07 < 20 - 5 = 15
    lay_bad, _ = layout(rot, _sq("N", 8.5, 0.0, size=4.0))  # its left edge (6.5) is inside the 7.07 mm reach
    assert "MOVING_ENVELOPE_VIOLATION" in kinds(lay_bad)


# ---- tool access / cable keepout -----------------------------------------------------------------------------------------------------
def test_tool_access_and_cable_regions():
    tool = PolygonRegion(region_id="hexkey", polygon_mm=((-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)), z_range_mm=(20.0, 60.0))
    cable = PolygonRegion(region_id="cable", polygon_mm=((5.0, -2.0), (30.0, -2.0), (30.0, 2.0), (5.0, 2.0)), z_range_mm=(0.0, 10.0))
    owner = _sq("Mount", 0.0, 0.0, z=(0.0, 25.0), tool_access=(tool,), cable_keepout=(cable,))
    lay, _ = layout(owner, _sq("Above", 1.0, 1.0, size=2.0, z=(30.0, 50.0)))
    assert "TOOL_ACCESS_BLOCKED" in kinds(lay)  # a component above the mount blocks tool access
    lay2, _ = layout(owner, _sq("Cab", 20.0, 0.0, size=6.0, z=(0.0, 5.0)))
    assert "CABLE_KEEPOUT_VIOLATION" in kinds(lay2)
    lay3, _ = layout(owner, _sq("Cab", 20.0, 0.0, size=6.0, z=(12.0, 30.0)))  # above the cable region's height band
    assert "CABLE_KEEPOUT_VIOLATION" not in kinds(lay3)


def test_layout_result_is_deterministic_and_sorted():
    a = layout(_sq("B", 6.0, 0.0), _sq("A", 0.0, 0.0), _sq("C", 40.0, 40.0))[0]
    b = layout(_sq("A", 0.0, 0.0), _sq("C", 40.0, 40.0), _sq("B", 6.0, 0.0))[0]
    assert a.model_dump() == b.model_dump()
    assert list(a.rotated_bounds) == sorted(a.rotated_bounds)


def test_layout_dependency_unavailable_is_reported_not_crashed(monkeypatch):
    """If Shapely cannot be imported the layout stage becomes UNKNOWN / DEPENDENCY_UNAVAILABLE and physics still runs."""
    import builtins
    import sys

    for name in [m for m in sys.modules if m.startswith(("os_optics.layout", "os_optics.layout")) or m == "shapely" or m.startswith("shapely.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shapely" or name.startswith("shapely."):
            raise ImportError("simulated: shapely is not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    proj = ProjectSnapshot(components=(_sq("A", 0.0, 0.0),))
    res = run(proj, layout=True)
    assert res.layout is None and "DEPENDENCY_UNAVAILABLE" in issue_codes(res) and res.status.value == "UNKNOWN"
