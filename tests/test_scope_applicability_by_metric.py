"""A model-scope finding names the targets whose metric it can change.

A finding about a coating's response (its reflectance, transmittance or phases outside what is published, or not
determined from behind) changes power, phase and polarization behind the surface, and nothing about where the light goes
or its Gaussian beam. The first test is the reason: with every coating coefficient of a scene changed, every geometry
metric keeps its value bit for bit while the power changes. The others pin what follows: a geometry target behind such a
finding keeps the Engine's assessment, a power target does not decide, and a finding that changes the geometry (a thin
lens outside its band) still reaches every target.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics._vendor.shared_contracts.optical_design import TargetOutcomeV1
from os_optics.contracts import (
    CircularAperture,
    ConstantCoating,
    EngineeringComponentSnapshot,
    MirrorInteraction,
    PlaneSurface,
    Pose3D,
    ProjectSnapshot,
    RefractInteraction,
    Target,
)
from os_optics.errors import RequirementAssessment, Status, assessment_of
from os_optics.local.validity import MODEL_SCOPE_AFFECTS, MODEL_SCOPE_CODES, finding_applies
from os_optics.validation.quantity_kinds import GEOMETRY_ONLY_METRICS, QuantityKind
from os_optics.optical_view import geometry as g
from os_optics.validation.metrics import OBSERVATION_METRICS, POWER_DERIVED_METRICS

from helpers import run
from local_helpers import FLAT_MIRROR, element, plane, project, source
from local_helpers import run as run_local

SQRT_HALF = math.sqrt(0.5)
#: The findings about a coating's response (its reflectance, transmittance and phases outside what is published, or not
#: determined from behind): they leave where the light goes and its Gaussian beam as traced.
COATING_RESPONSE_CODES = frozenset({"COATING_WAVELENGTH_OUTSIDE_BANDS", "COATING_AOI_OUTSIDE_RANGE", "COATING_DATA_OUTSIDE_RANGE",
                                    "ELEMENT_MET_FROM_BEHIND"})
#: The geometry metrics this test's scene measures on its plane: the chief ray's position, direction and angle, its
#: geometric length, its wavelength, the Gaussian radii, waists, Rayleigh ranges and divergences.
GEOMETRY_METRICS = frozenset({
    "position_u_mm", "position_v_mm", "direction_cosine_u", "direction_cosine_v", "aoi_deg", "acceptance_margin_deg",
    "geometric_path_length_mm", "wavelength_nm", "beam_radius_x_mm", "beam_radius_y_mm", "beam_radius_mm", "waist_radius_x_mm",
    "waist_radius_y_mm", "waist_distance_x_mm", "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm",
    "divergence_half_angle_x_rad", "divergence_half_angle_y_rad",
})


def _coated_scene(reflectance: float, transmittance: float, phase_rad: float) -> ProjectSnapshot:
    """A beam through a coated plano-convex lens (both faces coated) and off a coated 45 deg fold mirror onto a plane with an
    acceptance cone; one target per geometry metric and one on the power."""

    lens_coat = ConstantCoating(reflectance_s=1.0 - transmittance, reflectance_p=1.0 - transmittance,
                                transmittance_s=transmittance, transmittance_p=0.9 * transmittance)
    glass = B.singlet_lens("L", r1_mm=80.0, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=8.0, material="N-BK7", position=(30, 0, 20))
    lens = glass.model_copy(update={"surfaces": tuple(
        s.model_copy(update={"interaction": RefractInteraction(coating=lens_coat)}) for s in glass.surfaces)})
    mirror_coat = ConstantCoating(reflectance_s=reflectance, reflectance_p=0.8 * reflectance, transmittance_s=0.0, transmittance_p=0.0)
    face = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=12.0),
                        interaction=MirrorInteraction(coating=mirror_coat, phase_s_rad=phase_rad, phase_p_rad=0.5 * phase_rad))
    mirror = EngineeringComponentSnapshot(instance_id="M", role="mirror", surfaces=(face,),
                                          pose=Pose3D.from_axis_direction((70.0, 0.0, 20.0), (-SQRT_HALF, SQRT_HALF, 0.0)))
    targets = tuple(Target(target_id=metric, metric=metric, observation_id="out", minimum=-1e9) for metric in sorted(GEOMETRY_METRICS))
    return ProjectSnapshot(
        components=(B.laser("S", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0.3, 20), direction=(1, 0, 0),
                            polarization=B.JonesVector.linear(30.0)), lens, mirror),
        observations=(B.observation_plane("out", position=(70.0, 40.0, 20.0), normal=(0.0, 1.0, 0.0), radius_mm=20.0,
                                          acceptance_half_angle_deg=10.0),),
        targets=(*targets, Target(target_id="power", metric="power_W", observation_id="out", minimum=0.0)))


def test_no_geometry_metric_depends_on_a_coating_coefficient():
    """Two runs of one scene whose coatings differ in every coefficient (R, T, s against p, the mirror's phases): each
    geometry metric has a value in both and it is the same bits; the power differs. So a finding about a coating's response
    cannot change what a geometry metric reads."""

    first, second = run(_coated_scene(0.9, 0.99, math.pi)), run(_coated_scene(0.4, 0.6, 0.3))
    one, two = ({t.target_id: t for t in r.targets} for r in (first, second))
    assert GEOMETRY_METRICS <= set(OBSERVATION_METRICS) and not GEOMETRY_METRICS & POWER_DERIVED_METRICS
    for metric in sorted(GEOMETRY_METRICS):
        assert one[metric].value is not None and one[metric].status.value == "PASS", (metric, one[metric].message)
        assert one[metric].value == two[metric].value, metric
    assert one["power"].value != two["power"].value and one["power"].value > 0.0 and two["power"].value > 0.0
    (a,), (b,) = first.observations[0].arrivals, second.observations[0].arrivals
    assert a.beam.position_mm == b.beam.position_mm and a.beam.direction == b.beam.direction
    assert a.beam.gaussian.model_dump() == b.beam.gaussian.model_dump() and a.beam.polarization != b.beam.polarization


def test_the_table_of_findings_and_metrics():
    """The one rule (``finding_applies`` over ``MODEL_SCOPE_AFFECTS``): a coating-response finding spares every metric that
    reads geometry alone, the metrics above among them; it reaches every metric that reads power, phase or polarization."""

    assert GEOMETRY_METRICS <= GEOMETRY_ONLY_METRICS and COATING_RESPONSE_CODES <= set(MODEL_SCOPE_CODES)
    assert all(MODEL_SCOPE_AFFECTS[code] == {QuantityKind.POWER, QuantityKind.POLARIZATION} for code in COATING_RESPONSE_CODES)
    for code in COATING_RESPONSE_CODES:
        for metric in OBSERVATION_METRICS:
            assert finding_applies(code, metric) is (metric not in GEOMETRY_ONLY_METRICS), (code, metric)
    for code in ("MODEL_WAVELENGTH_OUTSIDE_BAND", "MATERIAL_DATA_OUTSIDE_RANGE"):  # (findings that change the geometry)
        assert all(finding_applies(code, metric) for metric in GEOMETRY_METRICS), code
    # a metric or a code the table does not know is affected: nothing is spared by default
    assert finding_applies("COATING_AOI_OUTSIDE_RANGE", "a_new_metric") and finding_applies("A_NEW_CODE", "position_u_mm")
    assert all(finding_applies(code, "power_W") and finding_applies(code, "phase_rad") for code in COATING_RESPONSE_CODES)


def _level(axis) -> list[float]:
    z, y = g.unit(axis), (0.0, 0.0, 1.0)
    return list(g.matrix_to_quat((g.cross(y, z), y, z)))


def _mirror_beyond_its_angle_range(aoi_deg: float):
    """A catalog mirror characterized for 0-45 deg met at ``aoi_deg``, with a position, a beam-size and a power target on the
    plane behind it."""

    theta = math.radians(aoi_deg)
    normal, reflected = (-math.cos(theta), math.sin(theta), 0.0), (-math.cos(2 * theta), math.sin(2 * theta), 0.0)
    targets = [{"target_id": "where", "metric": "position_u_mm", "observation_id": "screen", "target_value": 0.0, "tolerance_abs": 1e-6},
               {"target_id": "size", "metric": "beam_radius_x_mm", "observation_id": "screen", "maximum": 5.0},
               {"target_id": "power", "metric": "power_W", "observation_id": "screen", "minimum": 1e-6}]
    return run_local(project([source(position=(-100.0, 0.0, 0.0)), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), _level(normal))],
                             [plane("screen", g.scale(reflected, 100.0), _level(reflected))], targets))


def test_a_coating_met_beyond_its_angle_range_leaves_position_and_size_targets_decided():
    """At 50 deg the mirror's coating is outside its 0-45 deg range: the finding names the plane and, of its three targets,
    the power target only. The record keeps the Engine's assessment of the position and beam-size targets (their values
    are those of the 45 deg fold turned: on the axis, the same beam radius) and holds the power target undecided."""

    inside, beyond = _mirror_beyond_its_angle_range(45.0), _mirror_beyond_its_angle_range(50.0)
    assert inside.model_scope_issues == ()
    (issue,) = beyond.model_scope_issues
    assert issue.code == "COATING_AOI_OUTSIDE_RANGE" and issue.affected_observation_ids == ("screen",)
    assert issue.affected_target_ids == ("power",)
    rows = {t["target_id"]: t for t in beyond.local_validation.simulation.model_dump(mode="json")["target_outcomes"]}
    reference = {t["target_id"]: t for t in inside.local_validation.simulation.model_dump(mode="json")["target_outcomes"]}
    assert (rows["power"]["assessment"], rows["power"]["applicability_issue_codes"]) == ("INDETERMINATE", ["COATING_AOI_OUTSIDE_RANGE"])
    assert rows["power"]["status"] == "PASS" and rows["power"]["value"] == pytest.approx(reference["power"]["value"], rel=1e-9)
    for name in ("where", "size"):
        assert (rows[name]["status"], rows[name]["assessment"]) == ("PASS", "MET") and not rows[name].get("applicability_issue_codes"), name
    assert rows["where"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert rows["size"]["value"] == pytest.approx(reference["size"]["value"], rel=1e-9)
    outcome = beyond.local_validation.simulation.model_dump(mode="json")["outcome"]["requirements"]
    assert sorted(outcome["met"]) == ["size", "where"] and outcome["indeterminate"] == ["power"]


def test_the_assessment_has_one_rule_and_the_record_contract_agrees_with_it():
    """``assessment_of(status, applicability codes)`` is the rule; the record contract's own validator accepts exactly its
    answer for every status, with and without an applicability code."""

    for status in Status:
        for codes in ((), ("COATING_AOI_OUTSIDE_RANGE",)):
            expected = assessment_of(status, codes)
            assert expected is (RequirementAssessment.INDETERMINATE if codes else assessment_of(status))
            for candidate in RequirementAssessment:
                row = dict(target_id="t", metric="power_W", status=status.value, assessment=candidate.value, applicability_issue_codes=codes)
                if candidate is expected:
                    assert TargetOutcomeV1(**row).assessment == expected.value
                else:
                    with pytest.raises(ValueError):
                        TargetOutcomeV1(**row)
