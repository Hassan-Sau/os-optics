"""A normal-incidence (0 deg) mirror of the seed catalog release in designs a customer builds: retro-reflection onto the
source and back-reflection onto a beamsplitter.

The catalog has flat mirrors whose reflecting coating is specified at exactly 0 deg incidence. Used as specified, such a
mirror sends the beam straight back along the incoming beam: onto the source, or back onto the beamsplitter that fed it
(a double-pass tap). Both run end to end -- lowering, the one Engine run, the validity check and the Optical Simulation
View -- in model scope. The reflected-only arm has the published numeric power;
transmission through public glass with unstated bulk loss has its coated upper bound.

A beam path that crosses one observation plane more than once -- a retro-reflection back through a monitor plane, a
double-pass arm, a U-turn through one large plane -- is drawn by the Optical Simulation View with one footprint per
crossing (scene schema 3.0, ``BeamFootprintV1.crossing_index``), whatever the mirror angle; the local run no longer
refuses it.

Products are selected by query from the seed release (no SKU named): an AOI-0 flat mirror, a plate beamsplitter at
45 deg whose coating bands cover the mirror's wavelength, and a flat mirror specified at 45 deg incidence.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from os_optics.local import LocalValidationStatusV1, json_api, load_public_catalog, run_optical_project
from os_optics.optical_view import parse_glb, scene_glb_bytes
from os_optics.optical_view import geometry as g
from os_optics.viewer.optical import optical_scene_from

from catalog_store_helpers import REAL_CATALOG

CATALOG = load_public_catalog(REAL_CATALOG)  # the real release: the AOI-0 mirrors exist only there
POWER_W = 0.001
S = math.sqrt(0.5)
SOURCE_X = -100.0
ALONG = {  # pose quaternion (w, x, y, z) whose local +Z is the direction and local +Y is design +Z
    (1, 0): (0.5, 0.5, 0.5, 0.5),
    (-1, 0): (-0.5, -0.5, 0.5, 0.5),
    (0, -1): (S, S, 0.0, 0.0),
    (0, 1): (0.0, 0.0, S, S),
}
#: local +Z = (-1, 1, 0) / sqrt 2: a plate met at 45 deg by a beam along +X, reflecting it into +Y
SPLITTER_POSE = (-0.27059805007309856, -0.27059805007309856, 0.6532814824381883, 0.6532814824381884)


def _coating(surface):
    return CATALOG.coating(surface.coating_id) if surface.coating_id else None


def _normal_incidence_mirrors():
    found = []
    for product in sorted(CATALOG.products, key=lambda p: p.optical_sku):
        model = product.simulation_models[0]
        if product.form.value != "FLAT_MIRROR" or len(model.surfaces) != 1:
            continue
        coating = _coating(model.surfaces[0])
        if coating is not None and Decimal(coating.valid_aoi_deg.min_deg) == Decimal(coating.valid_aoi_deg.max_deg) == 0:
            found.append(product)
    return found


def _plate_splitter_at(wavelength: Decimal):
    """A plate beamsplitter specified at 45 deg: coated splitting face S2 at z = 0, AR face S1 behind it."""

    for product in sorted(CATALOG.products, key=lambda p: p.optical_sku):
        model = product.simulation_models[0]
        if product.form.value != "PLATE_BEAMSPLITTER" or model.model_kind != "SURFACE_SEQUENCE":
            continue  # (a SURFACE_SET plate places its faces by pose, not by vertex)
        faces = {s.traced.value: s for s in model.surfaces}
        if set(faces) != {"TRANSMITTED", "TRANSMITTED_AND_REFLECTED"} or Decimal(faces["TRANSMITTED_AND_REFLECTED"].vertex_z_mm) != 0:
            continue
        coatings = [_coating(s) for s in faces.values()]
        if all(c is not None and c.band_at(wavelength) is not None and Decimal(c.valid_aoi_deg.min_deg) <= 45 <= Decimal(c.valid_aoi_deg.max_deg)
               for c in coatings):
            return product, faces["TRANSMITTED_AND_REFLECTED"], faces["TRANSMITTED"]
    return None


def _wavelength(mirror) -> Decimal:
    band = mirror.simulation_models[0].valid_band_nm
    return (Decimal(band.min_nm) + Decimal(band.max_nm)) / 2


def _pair():
    mirrors = _normal_incidence_mirrors()
    # a failure, not a skip: a generator regression that dropped the AOI-0 mirrors must not pass silently
    assert mirrors, "the seed release has no flat mirror specified at 0 deg incidence"
    for mirror in mirrors:
        splitter = _plate_splitter_at(_wavelength(mirror))
        if splitter is not None:
            return mirror, *splitter
    raise AssertionError("no plate beamsplitter of the seed release covers the wavelength of an AOI-0 mirror")


MIRROR, SPLITTER, SPLIT_FACE, AR_FACE = _pair()
WAVELENGTH = _wavelength(MIRROR)
R_MIRROR = float(_coating(MIRROR.simulation_models[0].surfaces[0]).band_at(WAVELENGTH).reflectance)
_SPLIT = _coating(SPLIT_FACE).band_at(WAVELENGTH)
R_SPLIT, T_SPLIT = float(_SPLIT.reflectance), float(_SPLIT.transmittance)
T_AR = float(_coating(AR_FACE).band_at(WAVELENGTH).transmittance)


def _source(wavelength_nm: Decimal = WAVELENGTH) -> dict:
    return {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "customer-laser", "description": "Customer laser."},
            "nominal_pose": {"position_mm": [SOURCE_X, 0.0, 0.0], "quaternion_wxyz": list(ALONG[(1, 0)])},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": float(wavelength_nm), "power_w": POWER_W, "waist_radius_mm": 0.5,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}


def _element(instance_id: str, role: str, product, position, quaternion) -> dict:
    return {"instance_id": instance_id, "optical_role": role,
            "product": {"optical_sku": product.optical_sku, "catalog_version": CATALOG.catalog_version},
            "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)}}


def _plane(observation_id: str, position, facing, absorbing: bool = False) -> dict:
    return {"observation_id": observation_id, "pose": {"position_mm": list(position), "quaternion_wxyz": list(ALONG[facing])},
            "absorbing": absorbing}


def _project(components, observations, targets=()) -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": components, "observations": observations,
            "targets": list(targets)}


def _arrivals(result, observation_id: str):
    (observation,) = [o for o in result.observations if o.observation_id == observation_id]
    return observation.arrivals


def _footprints(run, observation_id: str):
    (plane,) = [p for p in run.optical_view.observation_planes if p.observation_id == observation_id]
    return plane.footprints


def test_the_catalog_mirror_is_specified_at_normal_incidence_only():
    (surface,) = MIRROR.simulation_models[0].surfaces
    assert surface.traced.value == "REFLECTED" and MIRROR.frame.is_reflector
    aoi = _coating(surface).valid_aoi_deg
    assert (Decimal(aoi.min_deg), Decimal(aoi.max_deg)) == (0, 0)


def test_a_normal_incidence_mirror_retro_reflects_the_beam_onto_its_source():
    """Source -> AOI-0 mirror 100 mm away -> straight back through the source; the plane 20 mm behind the source sees
    only the returning beam."""

    document = _project([_source(), _element("M1", "mirror", MIRROR, (0.0, 0.0, 0.0), ALONG[(-1, 0)])],
                        [_plane("BEHIND_SOURCE", (SOURCE_X - 20.0, 0.0, 0.0), (-1, 0))],
                        [{"target_id": "returned_power", "metric": "power_W", "observation_id": "BEHIND_SOURCE",
                          "minimum": 0.9 * R_MIRROR * POWER_W, "unit": "W"}])
    run = run_optical_project(document, catalog=CATALOG)
    assert run.status is LocalValidationStatusV1.PASS and run.model_scope_issues == ()
    assert [(t.target_id, t.status.value) for t in run.result.targets] == [("returned_power", "PASS")]
    (arrival,) = _arrivals(run.result, "BEHIND_SOURCE")
    assert arrival.beam.direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert arrival.beam.position_mm == pytest.approx((SOURCE_X - 20.0, 0.0, 0.0), abs=1e-9)  # back along the incoming axis
    assert arrival.beam.power_W == pytest.approx(R_MIRROR * POWER_W, rel=1e-12)
    # the Optical Simulation View draws it: one footprint, and the chief ray turns around at the mirror
    (footprint,) = _footprints(run, "BEHIND_SOURCE")
    assert footprint.path_id == arrival.beam.path_id
    (path,) = run.optical_view.beam_paths
    points = path.chief_ray.points_mm
    assert points[0] == pytest.approx((SOURCE_X, 0.0, 0.0)) and points[-1] == pytest.approx((SOURCE_X - 20.0, 0.0, 0.0))
    assert max(p[0] for p in points) == pytest.approx(0.0, abs=1e-9)


def _tap_design(extra_observations=()) -> dict:
    """Source -> 45 deg plate beamsplitter -> (reflected arm) AOI-0 end mirror -> back onto the splitter's coated face:
    the transmitted half of the return leaves through the TAP port, the reflected half goes back onto the source."""

    return _project(
        [_source(), _element("BS", "splitter", SPLITTER, (0.0, 0.0, 0.0), SPLITTER_POSE),
         _element("M1", "mirror", MIRROR, (0.0, 60.0, 0.0), ALONG[(0, -1)])],
        [_plane("THRU", (60.0, 0.0, 0.0), (1, 0)), _plane("TAP", (0.0, -60.0, 0.0), (0, -1)),
         _plane("BEHIND_SOURCE", (SOURCE_X - 20.0, 0.0, 0.0), (-1, 0)), *extra_observations],
        [{"target_id": "tap_power", "metric": "power_W", "observation_id": "TAP",
          "minimum": 0.9 * R_SPLIT * R_MIRROR * T_SPLIT * T_AR * POWER_W, "unit": "W"}],
    )


def test_a_back_reflection_onto_a_beamsplitter_runs_in_scope_with_the_coating_powers():
    run = run_optical_project(_tap_design(), catalog=CATALOG)
    assert run.status is LocalValidationStatusV1.UNKNOWN and run.model_scope_issues == ()
    assert [(t.target_id, t.status.value, t.assessment.value) for t in run.result.targets] == [("tap_power", "UNKNOWN", "INDETERMINATE")]
    expected = {
        "THRU": (T_SPLIT * T_AR, (1.0, 0.0, 0.0), True),  # glass traversal: only the zero-loss upper envelope
        "TAP": (R_SPLIT * R_MIRROR * T_SPLIT * T_AR, (0.0, -1.0, 0.0), True),
        "BEHIND_SOURCE": (R_SPLIT * R_MIRROR * R_SPLIT, (-1.0, 0.0, 0.0), False),  # all-reflected, no substrate pass
    }
    for observation_id, (fraction, direction, bounded) in expected.items():
        main = max(_arrivals(run.result, observation_id), key=lambda a: a.beam.power_W or a.power_upper_bound_W or 0.0)
        if bounded:
            assert main.beam.power_W is None
            assert main.power_upper_bound_W == pytest.approx(fraction * POWER_W, rel=1e-9), observation_id
        else:
            assert main.power_upper_bound_W is None
            assert main.beam.power_W == pytest.approx(fraction * POWER_W, rel=1e-9), observation_id
        assert main.beam.direction == pytest.approx(direction, abs=1e-9), observation_id
        # every arrival of a plane is drawn, one footprint per path
        footprints = _footprints(run, observation_id)
        assert sorted(f.path_id for f in footprints) == sorted(a.beam.path_id for a in _arrivals(run.result, observation_id))
    # the return meets the end mirror exactly at normal incidence and lands back on the source axis
    returned = max(_arrivals(run.result, "BEHIND_SOURCE"), key=lambda a: a.beam.power_W or a.power_upper_bound_W or 0.0)
    assert returned.beam.position_mm[1:] == pytest.approx((0.0, 0.0), abs=1e-9)


def _crossings(run, observation_id: str) -> list[tuple[str, int]]:
    return [(f.path_id, f.crossing_index) for f in _footprints(run, observation_id)]


def _check_repeated_crossings_are_drawn(run, observation_id: str) -> None:
    """Every Engine arrival of the plane is one footprint (same path, crossing, position and power), in the scene, in its
    GLB (a filled disc for a first crossing, a ring for a later one), in the local viewer and in the json_api summary."""

    arrivals = sorted(_arrivals(run.result, observation_id), key=lambda a: (a.beam.path_id, a.crossing_index))
    footprints = _footprints(run, observation_id)
    assert [(f.path_id, f.crossing_index) for f in footprints] == [(a.beam.path_id, a.crossing_index) for a in arrivals]
    for footprint, arrival in zip(footprints, arrivals, strict=True):
        assert footprint.chief_uv_mm == pytest.approx(arrival.uv_mm, abs=1e-6)
        assert footprint.power_W == pytest.approx(arrival.beam.power_W, rel=1e-8)
    gltf, binary = parse_glb(scene_glb_bytes(run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n.get("name") == f"observation:{observation_id}"]
    assert [(f["path_id"], f["crossing_index"]) for f in node["extras"]["footprints"]] == _crossings(run, observation_id)
    primitives = gltf["meshes"][node["mesh"]]["primitives"][1:]  # the plane, then one primitive per footprint
    names = [gltf["materials"][p["material"]]["name"] for p in primitives]
    assert [n.startswith("footprint_repeat_") for n in names] == [f.crossing_index > 0 for f in footprints]
    viewer = optical_scene_from(run.optical_view, gltf, binary)
    (part,) = [p for p in viewer.parts if p.part_id == observation_id]
    assert [(f["path_id"], f["crossing_index"]) for f in part.facts["footprints"]] == _crossings(run, observation_id)
    summary = json_api.run_summary(run)
    (row,) = [o for o in summary["observations"] if o["observation_id"] == observation_id]
    assert sorted((a["path_id"], a["crossing_index"]) for a in row["arrivals"]) == _crossings(run, observation_id)


def test_a_double_pass_arm_through_a_monitor_plane_is_drawn_with_both_crossings():
    """A non-absorbing monitor plane inside the double-pass arm is crossed by the same beam path going out and
    coming back. The local run passes and the view draws both crossings: crossing 0 going out (+Y), crossing 1 coming
    back (-Y) at the same spot with the end mirror's reflectance applied."""

    run = run_optical_project(_tap_design([_plane("ARM", (0.0, 30.0, 0.0), (0, 1))]), catalog=CATALOG)
    assert run.status is LocalValidationStatusV1.UNKNOWN and run.model_scope_issues == ()  # unknown loss in the splitter's transmitted arm
    out, back = sorted(_arrivals(run.result, "ARM"), key=lambda a: a.crossing_index)
    assert (out.crossing_index, back.crossing_index) == (0, 1) and out.beam.path_id == back.beam.path_id
    assert out.beam.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert back.beam.direction == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert out.beam.power_W == pytest.approx(R_SPLIT * POWER_W, rel=1e-9)
    assert back.beam.power_W == pytest.approx(R_MIRROR * out.beam.power_W, rel=1e-12)
    assert back.uv_mm == pytest.approx(out.uv_mm, abs=1e-9)  # straight back along the arm
    assert _crossings(run, "ARM") == [(out.beam.path_id, 0), (out.beam.path_id, 1)]
    _check_repeated_crossings_are_drawn(run, "ARM")
    # the other planes keep their own single crossings
    for observation_id in ("THRU", "TAP", "BEHIND_SOURCE"):
        assert all(index == 0 for _, index in _crossings(run, observation_id)), observation_id


def test_a_retro_reflection_through_a_monitor_plane_is_drawn_with_both_crossings():
    """Source -> non-absorbing monitor plane -> AOI-0 mirror -> straight back through the same plane onto the source:
    two footprints of one path at the same spot (crossing 0: the full power along +X; crossing 1: R times it along -X).
    The monitor consumes nothing, so the plane behind the source sees the returning beam with the mirror's power."""

    document = _project([_source(), _element("M1", "mirror", MIRROR, (0.0, 0.0, 0.0), ALONG[(-1, 0)])],
                        [_plane("MONITOR", (-50.0, 0.0, 0.0), (1, 0)), _plane("BEHIND_SOURCE", (SOURCE_X - 20.0, 0.0, 0.0), (-1, 0))])
    run = run_optical_project(document, catalog=CATALOG)
    assert run.status is LocalValidationStatusV1.PASS and run.model_scope_issues == ()
    (path,) = run.result.paths
    first, second = sorted(_arrivals(run.result, "MONITOR"), key=lambda a: a.crossing_index)
    assert (first.crossing_index, second.crossing_index) == (0, 1) and first.beam.path_id == second.beam.path_id == path.path_id
    assert first.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12) and first.beam.power_W == pytest.approx(POWER_W, rel=1e-12)
    assert second.beam.direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert second.beam.power_W == pytest.approx(R_MIRROR * POWER_W, rel=1e-12)
    assert first.uv_mm == pytest.approx((0.0, 0.0), abs=1e-9) and second.uv_mm == pytest.approx((0.0, 0.0), abs=1e-9)
    (behind,) = _arrivals(run.result, "BEHIND_SOURCE")
    assert behind.beam.power_W == pytest.approx(R_MIRROR * POWER_W, rel=1e-12)
    _check_repeated_crossings_are_drawn(run, "MONITOR")
    (monitor,) = [p for p in run.optical_view.observation_planes if p.observation_id == "MONITOR"]
    assert monitor.absorbing_extent.value == "NONE"


def _fold_mirror_and_wavelength():
    """The first flat mirror (by SKU) with one reflecting surface whose coating is published at 45 deg incidence, and a
    wavelength inside its model band and its coating band."""

    for product in sorted(CATALOG.products, key=lambda p: p.optical_sku):
        model = product.simulation_models[0]
        if product.form.value != "FLAT_MIRROR" or len(product.simulation_models) != 1 or len(model.surfaces) != 1:
            continue
        coating = _coating(model.surfaces[0])
        if coating is None or not Decimal(coating.valid_aoi_deg.min_deg) <= 45 <= Decimal(coating.valid_aoi_deg.max_deg):
            continue
        wavelength = _wavelength(product)
        if coating.band_at(wavelength) is not None:
            return product, wavelength
    raise AssertionError("the seed release has no flat mirror specified at 45 deg incidence")


FOLD_MIRROR, FOLD_WAVELENGTH = _fold_mirror_and_wavelength()
U_TURN_OFFSET_MM = 40.0


def _face_normal_pose(normal) -> tuple[float, float, float, float]:
    """Quaternion whose local +Z is ``normal`` (in the XY plane) and whose local +Y is design +Z."""

    z = g.unit(normal)
    y = (0.0, 0.0, 1.0)
    return g.matrix_to_quat((g.cross(y, z), y, z))


def _u_turn(absorbing: bool) -> dict:
    """Source along +X -> M1 (45 deg, into +Y) -> M2 (45 deg, into -X): the beam comes back parallel to itself,
    ``U_TURN_OFFSET_MM`` to the side, through the large plane WIDE between the source and the mirrors."""

    return _project(
        [_source(FOLD_WAVELENGTH),
         _element("M1", "mirror", FOLD_MIRROR, (0.0, 0.0, 0.0), _face_normal_pose((-1.0, 1.0, 0.0))),
         _element("M2", "mirror", FOLD_MIRROR, (0.0, U_TURN_OFFSET_MM, 0.0), _face_normal_pose((-1.0, -1.0, 0.0)))],
        [{**_plane("WIDE", (-50.0, 0.0, 0.0), (1, 0), absorbing=absorbing),
          "aperture": {"shape": "rectangle", "half_width_mm": 60.0, "half_height_mm": 10.0}}],
    )


def test_a_u_turn_through_one_large_plane_with_two_45_degree_mirrors_is_drawn_with_both_crossings():
    """Two footprints of one path on the large monitor plane, ``U_TURN_OFFSET_MM`` apart along the plane's u axis
    (design +Y), the second with both mirrors' reflectance applied."""

    reflectance = float(_coating(FOLD_MIRROR.simulation_models[0].surfaces[0]).band_at(FOLD_WAVELENGTH).reflectance)
    run = run_optical_project(_u_turn(absorbing=False), catalog=CATALOG)  # not refused: the view draws both crossings
    # the out and back crossings are two separate spots: the two-beam analysis reports them as not overlapping without failing
    assert run.status is LocalValidationStatusV1.PASS and run.model_scope_issues == ()
    (pair,) = [i for i in run.result.interference if i.observation_id == "WIDE"]
    assert (pair.crossing_index_a, pair.crossing_index_b) == (0, 1) and pair.status.value == "NOT_APPLICABLE"
    (path,) = run.result.paths
    first, second = sorted(_arrivals(run.result, "WIDE"), key=lambda a: a.crossing_index)
    assert (first.crossing_index, second.crossing_index) == (0, 1) and first.beam.path_id == second.beam.path_id == path.path_id
    # plane frame: u = design +Y, v = design +Z, normal = design +X
    assert first.uv_mm == pytest.approx((0.0, 0.0), abs=1e-9) and second.uv_mm == pytest.approx((U_TURN_OFFSET_MM, 0.0), abs=1e-9)
    assert first.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert second.beam.direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert second.beam.power_W == pytest.approx(reflectance * reflectance * POWER_W, rel=1e-12)
    _check_repeated_crossings_are_drawn(run, "WIDE")
    (wide,) = [p for p in run.optical_view.observation_planes if p.observation_id == "WIDE"]
    assert [f.chief_uv_mm for f in wide.footprints] == pytest.approx([(0.0, 0.0), (U_TURN_OFFSET_MM, 0.0)], abs=1e-6)
    assert wide.absorbing_extent.value == "NONE"


def test_an_absorbing_plane_stops_the_u_turn_at_its_first_crossing():
    """The same U-turn with the large plane absorbing on its active area: the beam is detected at its first crossing
    and never reaches the mirrors; the scene and the summary mark the plane as a detector over its active area."""

    run = run_optical_project(_u_turn(absorbing=True), catalog=CATALOG)
    (arrival,) = _arrivals(run.result, "WIDE")
    assert arrival.crossing_index == 0 and arrival.beam.power_W == pytest.approx(POWER_W, rel=1e-12)
    (wide,) = [p for p in run.optical_view.observation_planes if p.observation_id == "WIDE"]
    assert wide.absorbing_extent.value == "ACTIVE_AREA" and _crossings(run, "WIDE") == [(arrival.beam.path_id, 0)]
    gltf, _ = parse_glb(scene_glb_bytes(run.optical_view))
    (node,) = [n for n in gltf["nodes"] if n.get("name") == "observation:WIDE"]
    plane_material = gltf["materials"][gltf["meshes"][node["mesh"]]["primitives"][0]["material"]]["name"]
    assert node["extras"]["absorbing_extent"] == "ACTIVE_AREA" and plane_material == "plane_observation_absorbing"
    summary = json_api.run_summary(run)
    (row,) = [o for o in summary["observations"] if o["observation_id"] == "WIDE"]
    assert row["absorbing_extent"] == "ACTIVE_AREA"
    (line,) = summary["power_budget"]["lines"]
    assert line["input_power_W"] == pytest.approx(POWER_W) and line["sinks"] == {"DETECTED": pytest.approx(POWER_W)}
    assert line["conserved"] is True and summary["power_budget"]["conserved"] is True
