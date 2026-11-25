"""A model-scope finding affects the targets whose metric reads a kind of quantity the finding touches.

The two tables (``validity.MODEL_SCOPE_AFFECTS`` per finding code, ``quantity_kinds.METRIC_QUANTITY_KINDS`` per metric)
are checked against the trace itself: a metric entered as fixed by the geometry gives the same number, bit for bit, when
a coating's reflectance changes; a coefficient finding's scene has the same chief-ray points and directions as the
same scene with the published range widened to hold it.
"""

from __future__ import annotations

import math

import pytest

from local_helpers import (ALONG_X, DICHROIC, FLAT_MIRROR, FOLD_X_TO_Y, PCX_F100, PLATE_BS, POLARIZER, catalog_document, element, plane,
                           product_document, project, quat_multiply, run, scope_codes, source, with_lossless_bulk)
from os_optics.cli_exit import run_exit_code
from os_optics.local import validity
from os_optics.local.evaluate import capability_descriptor
from os_optics.optical_view import geometry as g
from os_optics.local.validity import MODEL_SCOPE_AFFECTS, MODEL_SCOPE_CODES, finding_applies, finding_quantity_kinds
from os_optics.validation.metrics import ALL_SUPPORTED, METRIC_REQUIRES, OBSERVATION_METRICS, PAIR_METRICS, POWER_DERIVED_METRICS
from os_optics.validation.alignment import ALIGNMENT_METRICS
from os_optics.validation.polarization_mixture import POLARIZATION_METRICS
from os_optics.validation.quantity_kinds import (ALL_QUANTITY_KINDS, GEOMETRY_ONLY_METRICS, LIGHT_INDEPENDENT_METRICS,
                                                      METRIC_QUANTITY_KINDS, QuantityKind, metric_quantity_kinds)

COEFFICIENT_CODES = ("COATING_WAVELENGTH_OUTSIDE_BANDS", "COATING_AOI_OUTSIDE_RANGE", "COATING_DATA_OUTSIDE_RANGE", "ELEMENT_AOI_OUTSIDE_RANGE",
                     "ELEMENT_MET_FROM_BEHIND")
HR_COATING = "DIELECTRIC_HIGH_REFLECTOR.400-1100R0.99T0.AOI0-45"
AR_COATING = "ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10"
SPLITTER_COATING = "BEAMSPLITTER.400-1100R0.5T0.5.AOI45-45"
DICHROIC_COATING = "DICHROIC.400-600R0.95T0.05_650-1100R0.1T0.9.AOI45-45"
OTHER_CATALOG = "2026-01-01.2"
TILT_20_ABOUT_DESIGN_Z = (math.cos(math.radians(10.0)), 0.0, 0.0, math.sin(math.radians(10.0)))
REVERSED_FOLD = quat_multiply((0.0, 0.0, 0.0, 1.0), FOLD_X_TO_Y)


# ------------------------------------------------------------------------------------------------ the tables


def test_every_metric_and_every_finding_code_has_an_explicit_entry():
    assert set(METRIC_QUANTITY_KINDS) == set(ALL_SUPPORTED) == set(METRIC_REQUIRES)
    assert set(MODEL_SCOPE_AFFECTS) == set(MODEL_SCOPE_CODES)
    assert GEOMETRY_ONLY_METRICS <= set(ALL_SUPPORTED) and LIGHT_INDEPENDENT_METRICS <= set(ALL_SUPPORTED)
    assert not GEOMETRY_ONLY_METRICS & LIGHT_INDEPENDENT_METRICS


def test_what_the_tables_do_not_hold_reads_and_touches_every_kind():
    assert metric_quantity_kinds("a_metric_added_later") == ALL_QUANTITY_KINDS
    assert finding_quantity_kinds("A_CODE_ADDED_LATER") == ALL_QUANTITY_KINDS
    assert finding_applies("A_CODE_ADDED_LATER", "position_u_mm") and finding_applies("COATING_AOI_OUTSIDE_RANGE", "a_metric_added_later")


def test_only_the_coefficient_codes_leave_the_geometry():
    narrowed = {code for code, kinds in MODEL_SCOPE_AFFECTS.items() if kinds != ALL_QUANTITY_KINDS}
    assert narrowed == set(COEFFICIENT_CODES)
    assert all(MODEL_SCOPE_AFFECTS[code] == {QuantityKind.POWER, QuantityKind.POLARIZATION} for code in COEFFICIENT_CODES)


def test_no_metric_that_reads_a_power_a_polarization_or_a_coating_delay_is_entered_as_geometry():
    reads_light_amounts = (set(POWER_DERIVED_METRICS) | set(POLARIZATION_METRICS)
                           | {"phase_rad", "phase_difference_rad", "group_delay_ps", "time_delay_ps", "coherence_factor",
                              "polarization_overlap", "spatial_overlap_ratio", "spot_rms_mm", "spot_centroid_u_mm", "spot_centroid_v_mm",
                              "active_area_power_fraction", "capture_margin_mm", "power_rating_utilization"})
    assert reads_light_amounts <= set(ALL_SUPPORTED)
    assert all(metric_quantity_kinds(metric) == ALL_QUANTITY_KINDS for metric in reads_light_amounts)
    assert all(metric_quantity_kinds(metric) == {QuantityKind.GEOMETRY} for metric in GEOMETRY_ONLY_METRICS)
    assert all(metric_quantity_kinds(metric) == frozenset() for metric in LIGHT_INDEPENDENT_METRICS)


def test_a_declared_limit_is_touched_by_no_finding_and_a_power_by_every_one():
    for code in MODEL_SCOPE_CODES:
        assert finding_applies(code, "power_W") and finding_applies(code, "stokes_s1")
        assert not finding_applies(code, "dof_travel_margin_mm")
        assert finding_applies(code, "position_u_mm") is (code not in COEFFICIENT_CODES)


def test_the_capability_descriptor_publishes_both_tables():
    scope = capability_descriptor("model_scope")["quantity_kinds"]
    assert scope["kinds"] == ["GEOMETRY", "POWER", "POLARIZATION"]
    assert scope["findings"] == {code: [k.value for k in QuantityKind if k in MODEL_SCOPE_AFFECTS[code]] for code in MODEL_SCOPE_CODES}
    assert scope["metrics"] == {metric: [k.value for k in QuantityKind if k in METRIC_QUANTITY_KINDS[metric]]
                                for metric in sorted(METRIC_QUANTITY_KINDS)}
    assert scope["findings"]["COATING_AOI_OUTSIDE_RANGE"] == ["POWER", "POLARIZATION"]
    assert scope["metrics"]["position_u_mm"] == ["GEOMETRY"] and scope["metrics"]["dof_drift_mm"] == []
    assert "quantity_kinds" not in capability_descriptor("plane_readings")


# ------------------------------------------------------------------------------------------------ scenes


def _level_quaternion(axis):
    z = g.unit(axis)
    y = (0.0, 0.0, 1.0)
    return list(g.matrix_to_quat((g.cross(y, z), y, z)))


def _target(target_id: str, metric: str, **fields) -> dict:
    return {"target_id": target_id, "metric": metric, "observation_id": fields.pop("observation_id", "screen"), **fields}


def _mirror(aoi_deg: float, targets=(), *, catalog=None, retained=()):
    """A flat mirror met at ``aoi_deg`` (its coating is published for 0-45 deg) and a plane square to the reflected beam."""

    theta = math.radians(aoi_deg)
    normal = (-math.cos(theta), math.sin(theta), 0.0)
    reflected = (-math.cos(2 * theta), math.sin(2 * theta), 0.0)
    mirror = element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), _level_quaternion(normal),
                     catalog_version=None if catalog is not None else "2026-01-01.1")
    mirror["retained_dofs"] = [dict(d) for d in retained]
    document = project([source(position=(-100.0, 0.0, 0.0)), mirror],
                       [plane("screen", g.scale(reflected, 100.0), _level_quaternion(reflected))], list(targets))
    return run(document) if catalog is None else run(document, catalog=catalog)


def _catalog(edit) -> dict:
    document = catalog_document() | {"catalog_version": OTHER_CATALOG}
    edit(document)
    return document


def _coating(document: dict, coating_id: str) -> dict:
    return next(c for c in document["coatings"] if c["coating_id"] == coating_id)


def _geometry(result) -> list:
    """Every chief-ray segment's ends, length and angle of incidence, and every arrival's point and direction."""

    paths = [(p.path_id, p.parent_path_id, [(s.start_mm, s.end_mm, s.length_mm, s.end_surface_id, s.end_event, s.angle_of_incidence_deg)
                                            for s in p.segments]) for p in result.paths]
    arrivals = [(o.observation_id, [(a.beam.path_id, a.uv_mm, a.aoi_deg, a.beam.geometric_length_mm, a.beam.optical_path_length_mm)
                                    for a in o.arrivals]) for o in result.observations]
    return [paths, arrivals]


# ------------------------------------------------------------------------------------------------ what a finding names


TX = {"axis": "Tx", "travel": {"min": -2.0, "max": 3.0}, "unit": "mm"}


def TRAVEL(device: str) -> dict:
    return {"target_id": "travel", "metric": "dof_travel_margin_mm", "device_instance_id": device, "minimum": 0.0,
            "parameters": [{"key": "axis", "value": "Tx"}, {"key": "required_min", "value": -1.0}, {"key": "required_max", "value": 1.0}]}


BEHIND_A_COATING_ANGLE = [
    _target("centred", "position_u_mm", minimum=-0.01, maximum=0.01, unit="mm"),
    _target("off_axis", "position_u_mm", minimum=1.0, unit="mm"),
    _target("radius", "beam_radius_x_mm", maximum=5.0, unit="mm"),
    _target("length", "geometric_path_length_mm", target_value=200.0, tolerance_abs=1e-6, unit="mm"),
    _target("power", "power_W", minimum=1e-6, unit="W"),
    _target("s1", "stokes_s1", minimum=-1.0),
    TRAVEL("M1"),
]


@pytest.fixture(scope="module")
def beyond_the_coating_angle():
    return _mirror(50.0, BEHIND_A_COATING_ANGLE, retained=[TX])


def test_a_coefficient_finding_names_the_targets_that_read_power_or_polarization(beyond_the_coating_angle):
    local = beyond_the_coating_angle
    assert scope_codes(local) == [("M1", "COATING_AOI_OUTSIDE_RANGE", "S1")]
    (issue,) = local.model_scope_issues
    assert issue.affected_observation_ids == ("screen",) and issue.affected_element_ids == ("M1",)
    assert issue.affected_target_ids == ("power", "s1")


def test_a_target_the_finding_does_not_touch_keeps_the_engines_assessment(beyond_the_coating_angle):
    local = beyond_the_coating_angle
    engine = {t.target_id: t for t in local.result.targets}
    recorded = {t.target_id: t for t in local.local_validation.simulation.target_outcomes}
    assert {i: t.assessment.value for i, t in engine.items()} == {
        "centred": "MET", "off_axis": "UNMET", "radius": "MET", "length": "MET", "power": "MET", "s1": "MET", "travel": "MET"}
    for target_id in ("centred", "off_axis", "radius", "length", "travel"):
        assert recorded[target_id].assessment == engine[target_id].assessment.value, target_id
        assert recorded[target_id].applicability_issue_codes is None, target_id
        assert recorded[target_id].value == engine[target_id].value
    for target_id in ("power", "s1"):
        assert recorded[target_id].assessment == "INDETERMINATE"
        assert recorded[target_id].applicability_issue_codes == ("COATING_AOI_OUTSIDE_RANGE",)
    requirements = local.local_validation.simulation.outcome.requirements
    assert requirements.met == ("centred", "radius", "length", "travel") and requirements.unmet == ("off_axis",)
    assert requirements.indeterminate == ("power", "s1")
    # the plane's own quantity is its power, which the finding touches
    (quantity,) = local.local_validation.simulation.observation_quantities
    assert quantity.quantity_state == "OUT_OF_MODEL_SCOPE" and "COATING_AOI_OUTSIDE_RANGE" in quantity.issue_codes


def test_the_requirements_exit_code_follows_the_targets_the_finding_touches():
    met = _mirror(50.0, BEHIND_A_COATING_ANGLE[:1])
    assert scope_codes(met) == [("M1", "COATING_AOI_OUTSIDE_RANGE", "S1")] and run_exit_code("requirements", met) == 0
    assert met.status.value == "OUT_OF_MODEL_SCOPE"  # the diagnostic summary still says a model was used outside its data
    assert run_exit_code("requirements", _mirror(50.0, BEHIND_A_COATING_ANGLE[1:2])) == 1
    assert run_exit_code("requirements", _mirror(50.0, BEHIND_A_COATING_ANGLE[4:5])) == 2


def test_a_finding_that_touches_the_geometry_names_every_target_that_reads_light():
    targets = [_target("centred", "position_u_mm", minimum=-0.01, maximum=0.01, unit="mm"),
               _target("power", "power_W", minimum=0.0, unit="W"),
               TRAVEL("L1")]
    lens = element("L1", "lens", PCX_F100, (0.0, 0.0, 0.0), ALONG_X)
    lens["retained_dofs"] = [dict(TX)]
    local = run(project([source(wavelength_nm=1200.0), lens], [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)], targets))
    codes = {i.code for i in local.model_scope_issues}
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in codes and "COATING_WAVELENGTH_OUTSIDE_BANDS" in codes
    by_code = {}
    for issue in local.model_scope_issues:
        by_code.setdefault(issue.code, set()).update(issue.affected_target_ids)
    assert by_code["MODEL_WAVELENGTH_OUTSIDE_BAND"] == {"centred", "power"}
    assert by_code["COATING_WAVELENGTH_OUTSIDE_BANDS"] == {"power"}
    recorded = {t.target_id: t for t in local.local_validation.simulation.target_outcomes}
    assert recorded["centred"].applicability_issue_codes == ("MODEL_WAVELENGTH_OUTSIDE_BAND",)
    assert recorded["centred"].assessment == "INDETERMINATE"
    assert recorded["power"].applicability_issue_codes == ("COATING_WAVELENGTH_OUTSIDE_BANDS", "MODEL_WAVELENGTH_OUTSIDE_BAND")
    assert recorded["travel"].applicability_issue_codes is None


# ------------------------------------------------------------------------------------------------ independent references


def _values(local) -> dict:
    return {t.target_id: t.value for t in local.result.targets}


def test_a_geometry_metric_of_one_beam_is_the_same_number_whatever_the_coating_reflects():
    """The reference: the same scene with the mirror coating's reflectance 0.99 and 0.37. Every metric entered as fixed by
    the geometry gives the same bits; the power does not."""

    metrics = sorted(GEOMETRY_ONLY_METRICS & set(OBSERVATION_METRICS))
    targets = [_target(m, m, minimum=-1e12) for m in metrics] + [_target("power", "power_W", minimum=0.0, unit="W")]
    published = _mirror(30.0, targets, catalog=_catalog(lambda d: None))
    dimmer = _mirror(30.0, targets, catalog=_catalog(lambda d: _coating(d, HR_COATING)["bands"][0].update(reflectance="0.37")))
    assert published.model_scope_issues == () and dimmer.model_scope_issues == ()
    before, after = _values(published), _values(dimmer)
    assert after["power"] == pytest.approx(before["power"] * 0.37 / 0.99, rel=1e-12)
    valued = [m for m in metrics if before[m] is not None]
    assert len(valued) >= len(metrics) - 4, {m: before[m] for m in metrics}  # (the first-order image metrics need a traced field)
    for metric in metrics:
        assert after[metric] == before[metric], metric
    assert _geometry(dimmer.result) == _geometry(published.result)


def _two_arms(targets, catalog):
    """A lossless-bulk plate splitter and a fold mirror send two beams to the same plane."""

    bounce = (0.0, 50.0, 0.0)
    out = g.unit((60.0, -50.0, 0.0))
    normal = g.unit(g.sub(out, (0.0, 1.0, 0.0)))
    document = project(
        [source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0),
         element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None),
         element("M1", "mirror", FLAT_MIRROR, bounce, _level_quaternion(normal), catalog_version=None)],
        [plane("screen", (60.0, 0.0, 0.0), radius_mm=40.0)], list(targets))
    return run(document, catalog=with_lossless_bulk(catalog))


def test_a_geometry_metric_of_two_beams_is_the_same_number_whatever_the_coating_reflects():
    metrics = sorted(GEOMETRY_ONLY_METRICS & (set(PAIR_METRICS) | set(ALIGNMENT_METRICS)))
    bare = _two_arms((), _catalog(lambda d: None))
    first, second = (a.beam.path_id for a in next(o for o in bare.result.observations if o.observation_id == "screen").arrivals)
    operands = [{"key": "path_id_a", "value": first}, {"key": "path_id_b", "value": second}]
    targets = [_target(m, m, minimum=-1e12, parameters=operands) for m in metrics] + [
        _target("balance", "intensity_balance", minimum=0.0, parameters=operands)]
    published = _two_arms(targets, _catalog(lambda d: None))
    dimmer = _two_arms(targets, _catalog(lambda d: _coating(d, HR_COATING)["bands"][0].update(reflectance="0.37")))
    assert len(next(o for o in published.result.observations if o.observation_id == "screen").arrivals) == 2
    before, after = _values(published), _values(dimmer)
    assert before["balance"] is not None and after["balance"] != before["balance"]
    valued = [m for m in metrics if before[m] is not None]
    assert {"beam_crossing_angle_deg", "half_angle_deg", "centroid_separation_mm", "angular_separation_rad"} <= set(valued), before
    for metric in metrics:
        assert after[metric] == before[metric], metric
    assert _geometry(dimmer.result) == _geometry(published.result)


def _widen_angles(document: dict, coating_id: str) -> None:
    _coating(document, coating_id)["valid_aoi_deg"] = {"min_deg": "0", "max_deg": "89", "status": "SPECIFIED"}


def _coefficient_scenes():
    """Per coefficient code: the scene that has the finding, and the edit of the published data that holds the scene."""

    def mirror(catalog):
        return _mirror(50.0, catalog=catalog)

    def lens(catalog):
        tilted = quat_multiply(TILT_20_ABOUT_DESIGN_Z, ALONG_X)
        return run(project([source(wavelength_nm=780.0), element("L1", "lens", PCX_F100, (0.0, 0.0, 0.0), tilted, catalog_version=None)],
                           [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)]), catalog=catalog)

    def polarizer(catalog):
        tilted = quat_multiply(TILT_20_ABOUT_DESIGN_Z, ALONG_X)
        return run(project([source(position=(-50.0, 0.0, 0.0)), element("POL", "polarizer", POLARIZER, (0.0, 0.0, 0.0), tilted, catalog_version=None)],
                           [plane("out", (50.0, 0.0, 0.0), radius_mm=30.0)]), catalog=catalog)

    def reversed_splitter(catalog):
        return run(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0),
                            element("BS", "splitter", PLATE_BS, (0.0, 0.0, 0.0), REVERSED_FOLD, catalog_version=None)],
                           [plane("far", (60.0, 0.0, 0.0), radius_mm=40.0)]), catalog=catalog)

    def dichroic(catalog):  # 625 nm lies in the gap between the dichroic coating's two bands (400-600 nm, 650-1100 nm)
        return run(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=625.0),
                            element("DM", "filter", DICHROIC, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=None)],
                           [plane("t", (60.0, 0.0, 0.0)), plane("r", (0.0, 60.0, 0.0), _level_quaternion((0.0, 1.0, 0.0)))]), catalog=catalog)

    def no_gap(document):
        _coating(document, DICHROIC_COATING)["bands"][1]["min_nm"] = "610"

    def lossy(document):
        _coating(document, SPLITTER_COATING)["bands"][0]["reflectance"] = "0.45"

    def wide_polarizer(document):
        product_document(POLARIZER, document)["simulation_models"][0]["valid_aoi_deg"] = {"min_deg": "0", "max_deg": "45", "status": "DESIGN_BASIS"}

    return {
        "COATING_AOI_OUTSIDE_RANGE": (mirror, lambda d: None, lambda d: _widen_angles(d, HR_COATING)),
        "COATING_AOI_OUTSIDE_RANGE/lens": (lens, lambda d: None, lambda d: _widen_angles(d, AR_COATING)),
        "COATING_WAVELENGTH_OUTSIDE_BANDS": (dichroic, lambda d: None, no_gap),
        "ELEMENT_AOI_OUTSIDE_RANGE": (polarizer, lambda d: None, wide_polarizer),
        "ELEMENT_MET_FROM_BEHIND": (reversed_splitter, lossy, lambda d: None),
    }


@pytest.mark.parametrize("case", sorted(_coefficient_scenes()))
def test_a_coefficient_finding_leaves_every_chief_ray_point_and_direction_as_inside_the_published_range(case):
    scene, outside, inside = _coefficient_scenes()[case]
    code = case.split("/")[0]
    found = scene(_catalog(outside))
    held = scene(_catalog(inside))
    assert {c for _, c, _ in scope_codes(found)} == {code}, scope_codes(found)
    assert held.model_scope_issues == ()
    assert len(found.result.paths) == len(held.result.paths) > 0
    assert _geometry(found.result) == _geometry(held.result)


def _mirror_with_engine_coating(valid_wavelength_nm: tuple[float, float]):
    """COATING_DATA_OUTSIDE_RANGE has no catalog scene (a catalog coating outside its bands is
    COATING_WAVELENGTH_OUTSIDE_BANDS): a catalog fold mirror whose Engine coating is stated for ``valid_wavelength_nm``,
    met by the source's 780 nm beam, which the Engine carries on with the coating's values at the nearest stated point.
    Returns the model-scope codes of the run and its result."""

    from os_optics import ValidationRequest, validate
    from os_optics.contracts import ConstantCoating
    from os_optics.local import lower_project
    from os_optics.local.validity import element_poses, element_scope_issues

    document = project([source(position=(-100.0, 0.0, 0.0)), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("screen", (0.0, 100.0, 0.0), _level_quaternion((0.0, 1.0, 0.0)))])
    lowered = lower_project(document, catalog_document())
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M1"]
    coating = ConstantCoating(reflectance_s=0.9, reflectance_p=0.9, transmittance_s=0.0, transmittance_p=0.0,
                              valid_wavelength_nm=valid_wavelength_nm)
    faces = tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": coating})})
                  if getattr(s.interaction, "coating", None) is not None else s for s in mirror.surfaces)
    components = tuple(c.model_copy(update={"surfaces": faces}) if c.instance_id == "M1" else c for c in lowered.request.project.components)
    request = lowered.request.model_copy(update={"project": lowered.request.project.model_copy(update={"components": components})})
    result = validate(ValidationRequest.model_validate_json(request.model_dump_json()))
    return {i.code for i in element_scope_issues(result, lowered.elements, poses=element_poses(lowered))}, result


def test_a_coating_whose_data_ran_out_leaves_every_chief_ray_point_and_direction_as_inside_its_data():
    found_codes, found = _mirror_with_engine_coating((400.0, 500.0))
    held_codes, held = _mirror_with_engine_coating((400.0, 900.0))
    assert found_codes == {"COATING_DATA_OUTSIDE_RANGE"} and held_codes == set()
    assert len(found.paths) == len(held.paths) > 0
    assert _geometry(found) == _geometry(held)


def test_the_table_names_exactly_the_codes_the_scenes_cover():
    # (COATING_DATA_OUTSIDE_RANGE: the Engine-coating scene above)
    assert {case.split("/")[0] for case in _coefficient_scenes()} | {"COATING_DATA_OUTSIDE_RANGE"} == set(COEFFICIENT_CODES)
    assert validity.MODEL_SCOPE_AFFECTS is MODEL_SCOPE_AFFECTS
