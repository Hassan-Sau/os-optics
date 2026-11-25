"""A plane through a physical surface's chief-ray hit point  records the same crossings as a plane just on the
incident side, and is never reported as not reached."""

from __future__ import annotations

import math

import pytest

from os_optics.builders import laser, mirror, npbs, observation_plane, polarizer, singlet_lens, window
from os_optics.contracts import ProjectSnapshot

from helpers import arrivals, issue_codes, run

X = (1.0, 0.0, 0.0)
MX = (-1.0, 0.0, 0.0)
WAVELENGTH_NM = 780.0
SOURCE_W = 1e-3
R_MIRROR = 0.95


def _source(power_W: float = SOURCE_W):
    return laser("SRC", wavelength_nm=WAVELENGTH_NM, power_W=power_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=X)


def _arrival_keys(result, oid: str):
    return sorted((a.beam.path_id, a.crossing_index, round(a.beam.power_W, 15), tuple(round(v, 9) for v in a.beam.direction))
                  for a in arrivals(result, oid))


def _same_as_incident_side(components, plane_x: float, normal=X, **kwargs):
    at = run(ProjectSnapshot(components=components, observations=(observation_plane("mon", position=(plane_x, 0.0, 0.0), normal=normal,
                                                                                    radius_mm=10.0, **kwargs),)))
    before = run(ProjectSnapshot(components=components, observations=(observation_plane("mon", position=(plane_x - 1e-3, 0.0, 0.0),
                                                                                        normal=normal, radius_mm=10.0, **kwargs),)))
    assert "OBSERVATION_NOT_REACHED" not in issue_codes(at)
    assert _arrival_keys(at, "mon") == _arrival_keys(before, "mon")
    return at, before


def test_plane_on_a_mirror_face_records_the_incident_and_the_reflected_beam():
    at, _ = _same_as_incident_side((_source(), mirror("M", position=(50.0, 0.0, 0.0), normal=MX, reflectance=R_MIRROR)), 50.0)
    powers = sorted(a.beam.power_W for a in arrivals(at, "mon"))
    assert powers == [pytest.approx(SOURCE_W * R_MIRROR), pytest.approx(SOURCE_W)]
    # the mirror still reflects: the reflected path escapes back past the source
    assert sorted(p.termination for p in at.paths) == ["ESCAPED"]


def test_plane_at_a_lens_vertex_records_the_incident_beam():
    lens = singlet_lens("L", r1_mm=51.5, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=6.0, material="N-BK7", position=(2.0, 0.0, 0.0))
    at, _ = _same_as_incident_side((_source(), lens), 0.0)
    assert len(arrivals(at, "mon")) == 1


def test_plane_on_a_window_face_and_a_polarizer_plane():
    plate = window("W", thickness_mm=5.0, clear_radius_mm=10.0, material="N-BK7", position=(2.5, 0.0, 0.0))
    _same_as_incident_side((_source(), plate), 0.0)
    pol = polarizer("P", position=(30.0, 0.0, 0.0), axis=X, angle_deg=0.0)
    at, _ = _same_as_incident_side((_source(), pol), 30.0)
    assert len(arrivals(at, "mon")) == 1


def test_beam_normal_plane_at_a_45_degree_splitter_centre():
    q = math.sqrt(0.5)
    split = npbs("BS", reflectance=0.5, position=(40.0, 0.0, 0.0), normal=(-q, q, 0.0))
    at, _ = _same_as_incident_side((_source(), split), 40.0)
    (arrival,) = arrivals(at, "mon")
    assert arrival.beam.power_W == pytest.approx(SOURCE_W)


def test_absorbing_detector_on_a_mirror_face_absorbs_the_incident_beam():
    at, _ = _same_as_incident_side((_source(), mirror("M", position=(50.0, 0.0, 0.0), normal=MX, reflectance=R_MIRROR)), 50.0,
                                   absorbing=True)
    (path,) = at.paths
    assert path.termination == "DETECTED"
    assert dict(at.power_budget.total_sinks)["DETECTED"] == pytest.approx(SOURCE_W)


def test_plane_just_behind_a_mirror_within_the_advance_tolerance_is_not_lost():
    components = (_source(), mirror("M", position=(50.0, 0.0, 0.0), normal=MX, reflectance=R_MIRROR))
    result = run(ProjectSnapshot(components=components, observations=(observation_plane("mon", position=(50.0 + 5e-8, 0.0, 0.0),
                                                                                        normal=X, radius_mm=10.0),)))
    assert len(arrivals(result, "mon")) == 2


# ------------------------------------------------------------------------------------- published-model validity (local runs)


def test_a_plane_on_a_catalog_lens_exit_face_does_not_read_as_light_leaving_through_the_edge():
    from local_helpers import PCX_F100, element, plane, project, run as local_run, source

    for x in (0.0, 3.6, 1.8):  # entry vertex, plano exit face, inside the glass
        document = project([source(position=(-100.0, 0.0, 0.0)), element("E", "lens", PCX_F100)], [plane("mon", (x, 0.0, 0.0), radius_mm=40.0)])
        result = local_run(document)
        assert result.model_scope_issues == ()
        assert len(arrivals(result.result, "mon")) == 1


def test_a_plane_on_the_back_face_of_a_reversed_splitter_keeps_the_met_from_behind_finding():
    """A plane through the back-face hit point adds a zero-length segment; the arrival side is still judged from the
    direction of travel. Light from behind a lossless coating is in model scope, so the finding is pinned on
    a copy of the plate splitter whose coating absorbs (R 0.45, T 0.5), where it sits on the coated face S2."""

    from local_helpers import (FOLD_X_TO_Y, PLATE_BS, catalog_document, element, plane, project, quat_multiply,
                               run as local_run, scope_codes, source)

    catalog = catalog_document() | {"catalog_version": "2026-01-01.2"}
    coating = next(c for c in catalog["coatings"] if c["coating_id"] == "BEAMSPLITTER.400-1100R0.5T0.5.AOI45-45")
    coating["bands"][0]["reflectance"] = "0.45"
    reversed_fold = quat_multiply((0.0, 0.0, 0.0, 1.0), FOLD_X_TO_Y)
    components = [source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0),
                  element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), reversed_fold, catalog_version=None)]
    plain = local_run(project(components, [plane("far", (80.0, 0.0, 0.0), radius_mm=40.0)]), catalog=catalog)
    assert scope_codes(plain) == [("BS", "ELEMENT_MET_FROM_BEHIND", "S2")]
    back_x = plain.result.paths[0].segments[0].end_mm[0]
    result = local_run(project(components, [plane("mon", (back_x, 0.0, 0.0), radius_mm=40.0)]), catalog=catalog)
    assert scope_codes(result) == scope_codes(plain)
    assert len(arrivals(result.result, "mon")) >= 1


def _plane_step(ray_path, oid: str) -> int:
    return list(ray_path.chief_surface_ids).index(f"observation/{oid}")


@pytest.mark.parametrize("mirror_normal", [(-math.sqrt(0.5), math.sqrt(0.5), 0.0), (-math.cos(0.3), math.sin(0.3), 0.0)])
def test_the_ray_data_of_a_plane_through_a_tilted_mirror_lie_on_the_plane(mirror_normal):
    """A beam-normal plane through a tilted mirror's hit point records the pupil rays where they cross the plane:
    on it (x = 0), with the optical path length to it (every ray of the collimated, normal beam: 100 mm from the waist
    plane), not at the mirror surface behind or before it."""

    from os_optics import EngineConfiguration, ValidationRequest, validate_with_ray_paths  # noqa: PLC0415

    snapshot = ProjectSnapshot(components=(_source(), mirror("M", position=(0.0, 0.0, 0.0), normal=mirror_normal)),
                               observations=(observation_plane("mon", position=(0.0, 0.0, 0.0), normal=X, radius_mm=10.0),))
    result, trace = validate_with_ray_paths(ValidationRequest(project=snapshot, configuration=EngineConfiguration()))
    assert "OBSERVATION_NOT_REACHED" not in issue_codes(result)
    (ray_path,) = [p for p in trace.paths if p.path_id == "SRC#0"]
    step = _plane_step(ray_path, "mon")
    rays = [r for r in ray_path.pupil_rays if len(r.points_mm) > step and r.surface_ids[step] == "observation/mon"]
    assert len(rays) > 4 and max(abs(r.points_mm[step][1]) for r in rays) > 0.1  # rays off the chief, where the mirror is off the plane
    for r in rays:
        assert r.points_mm[step][0] == pytest.approx(0.0, abs=1e-9)
        assert r.opl_mm[step] == pytest.approx(100.0, abs=1e-9)
    # the next step starts at the mirror: the rays reach it after the plane
    mirror_step = [r for r in rays if len(r.surface_ids) > step + 1 and r.surface_ids[step + 1] == "M/face"]
    assert len(mirror_step) == len(rays)
    for r in mirror_step:
        x, y, _ = r.points_mm[step + 1]
        assert x * mirror_normal[0] + y * mirror_normal[1] == pytest.approx(0.0, abs=1e-9)
