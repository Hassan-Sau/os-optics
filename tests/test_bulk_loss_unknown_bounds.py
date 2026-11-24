"""Unknown bulk absorption is an interval, never a silently lossless window.

The 2 mm ideal R=0/T=1 plate isolates bulk loss from face losses. A measured
zero, an explicitly assumed zero and an absent loss have different meanings.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from os_optics._vendor.shared_contracts.optical_design.optical_model import ProjectModelMaterialV1
from os_optics.local import EvaluationSession, run_optical_project
from os_optics.local.lowering import lower_project
from os_optics.contracts.request import ValidationRequest
from os_optics.validation.pipeline import validate
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.local.compile import engine_material
from os_optics.config import DEFAULT_NUMERICAL_CONFIG
from os_optics.contracts.results import PowerBudgetLine
from os_optics.contracts.optics import OpticalPath
from os_optics.optics.rays.path_ids import child_path_id
from os_optics.optics.rays.tracer import DETECTED, SINK_BULK_LOSS_UNKNOWN, PowerFlow, SourceInput, TraceOutput
from os_optics.validation.power import line_fate
from os_optics.validation.untraced import Probes, hazards, plane_reach

from declared_helpers import declared_component, sequence, surface, with_models
from local_helpers import plane, project, source
from test_material_physics_engine import HWS3 as _HWS3, HWS9 as _HWS9, air_index, STANDARD
from test_tabulated_thin_lens import ttl_model

# The independent catalogue row fixtures use abbreviated ids; here exercise the
# identifiers by which the two CDGM materials are copied into a Project.
HWS3 = _HWS3.model_copy(update={"material_id": "CDGM_HWS3"})
HWS9 = _HWS9.model_copy(update={"material_id": "CDGM_HWS9"})


# Page 1, UMICORE GASIR 1 revision 02: index table at 20 C. Its page 2
# external transmission including Fresnel surfaces is NOT internal loss data.
GASIR1 = ProjectModelMaterialV1.model_validate({
    "material_id": "UMICORE_GASIR1", "display_name": "UMICORE GASIR 1",
    "dispersion": {"kind": "TABULATED", "wavelength_nm": ["1064", "1540", "2000", "3500", "5000", "8000", "10000", "14000"],
                   "refractive_index": ["2.5796", "2.5424", "2.5267", "2.512", "2.5071", "2.4999", "2.4944", "2.4787"],
                   "origin": "MEASURED", "measurement": {"temperature_c": "20", "basis": "UMICORE GASIR 1 revision 02"}},
    "valid_band_nm": {"min_nm": "1064", "max_nm": "14000", "status": "SPECIFIED"},
    "basis": "UMICORE GASIR 1 revision 02: no measured internal bulk loss"})


AR = {"coating_id": "AR0", "kind": "ANTI_REFLECTION", "display_name": "Ideal lossless surfaces",
      "bands": [{"min_nm": "1000", "max_nm": "14050", "reflectance": "0", "transmittance": "1", "status": "DESIGN_BASIS"}],
      "valid_aoi_deg": {"min_deg": "0", "max_deg": "89", "status": "DESIGN_BASIS"},
      "aoi_reference": "AIR_EQUIVALENT", "basis": "Declared ideal surfaces R=0 T=1"}


def _document(material: ProjectModelMaterialV1, nm: float):
    glass = material.material_id
    window = {"model_id": "w", "model": sequence(
        [surface(1, "0", coating_id="AR0"), surface(2, "2", coating_id="AR0")], [glass],
        band={"min_nm": str(int(nm - 1)), "max_nm": str(int(nm + 1)), "status": "SPECIFIED"})}
    detector = {**plane("D", (30, 0, 0), radius_mm=20), "absorbing": True}
    document = with_models(project(
        [source(position=(-50, 0, 0), wavelength_nm=nm, waist_mm=0.5), declared_component("W", "window", "w")],
        [detector], [{"target_id": "min", "metric": "power_W", "observation_id": "D", "minimum": 0.0009, "unit": "W"},
                     {"target_id": "max", "metric": "power_W", "observation_id": "D", "maximum": 0.0009, "unit": "W"},
                     {"target_id": "len", "metric": "geometric_path_length_mm", "observation_id": "D", "minimum": 80, "unit": "mm"},
                     {"target_id": "min_impossible", "metric": "power_W", "observation_id": "D", "minimum": 0.0011, "unit": "W"},
                     {"target_id": "max_safe", "metric": "power_W", "observation_id": "D", "maximum": 0.0011, "unit": "W"}],
        environment=None), [window], [material.model_dump(mode="json", exclude_none=True)], [AR])
    return document


def _run(material: ProjectModelMaterialV1, nm: float):
    return run_optical_project(_document(material, nm)).result


@pytest.mark.parametrize("material,nm", [(GASIR1, 10000.0), (HWS9, 2200.0)])
def test_unknown_loss_bounds_detector_power_but_not_geometry(material, nm):
    result = _run(material, nm)
    observation = result.observations[0]
    (arrival,) = observation.arrivals
    assert arrival.beam.power_W is None
    assert arrival.power_upper_bound_W == pytest.approx(0.001)
    assert observation.total_power_W is None
    assert observation.power_lower_bound_W is None  # zero is the implicit lower end of the unknown-power interval
    assert observation.power_upper_bound_W == pytest.approx(0.001)
    assert observation.quantity_state.value == "BOUNDED"
    assert "between 0 W and 0.001 W" in observation.message
    assert observation.power_complete is False
    assert observation.power_by_wavelength[0].power_W is None
    assert observation.power_by_wavelength[0].complete is False
    assert [result.targets[i].status.value for i in (0, 1)] == ["UNKNOWN", "UNKNOWN"]
    assert all(result.targets[i].assessment.value == "INDETERMINATE" for i in (0, 1))
    assert all(result.targets[i].value is None for i in (0, 1))
    assert all("BULK_LOSS_NOT_ESTABLISHED" in result.targets[i].issue_codes for i in (0, 1))
    assert result.targets[2].status.value == "PASS"
    assert result.targets[3].status.value == "FAIL"  # upper bound disproves minimum
    assert result.targets[4].status.value == "PASS"  # upper bound proves maximum
    assert arrival.beam.geometric_length_mm == pytest.approx(80)
    assert arrival.beam.optical_path_length_mm is not None
    assert "BULK_LOSS_NOT_ESTABLISHED" in [i.code.value for i in result.issues]
    assert result.power_budget.conserved
    assert dict(result.power_budget.total_sinks)["BULK_LOSS_UNKNOWN"] == pytest.approx(0.001)


def test_known_upstream_crossing_is_graded_without_downstream_bulk_loss():
    document = _document(HWS9, 2200.0)
    document["observations"].insert(0, plane("UP", (-5, 0, 0), radius_mm=20))
    first_crossing = [{"key": "path_id", "value": "SRC#0"}, {"key": "crossing_index", "value": 0}]
    missing_crossing = [first_crossing[0], {"key": "crossing_index", "value": 1}]
    upstream = {"metric": "power_W", "observation_id": "UP", "unit": "W", "maximum": 0.0011}
    document["targets"].extend([
        {**upstream, "target_id": "up_crossing", "parameters": first_crossing},
        {**upstream, "target_id": "up_crossing_too_high", "parameters": first_crossing, "maximum": 0.0009},
        {**upstream, "target_id": "up_all_light"},
        {**upstream, "target_id": "up_missing_crossing", "parameters": missing_crossing},
        {**upstream, "target_id": "down_crossing", "observation_id": "D", "maximum": 0.0009,
         "parameters": first_crossing},
    ])

    result = run_optical_project(document).result
    before, after = result.observations
    (arrival,) = before.arrivals
    assert (arrival.beam.path_id, arrival.crossing_index) == ("SRC#0", 0)
    assert arrival.beam.power_W == pytest.approx(0.001)
    assert after.quantity_state.value == "BOUNDED" and after.total_power_W is None
    assert after.power_upper_bound_W == pytest.approx(0.001)
    targets = {target.target_id: target for target in result.targets}
    for name, status in (("up_crossing", "PASS"), ("up_crossing_too_high", "FAIL")):
        target = targets[name]
        assert target.status.value == status and target.value == pytest.approx(0.001)
        assert "BULK_LOSS_NOT_ESTABLISHED" not in target.issue_codes
    for name in ("up_all_light", "up_missing_crossing", "min", "max", "down_crossing"):
        target = targets[name]
        assert target.status.value == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
        assert "BULK_LOSS_NOT_ESTABLISHED" in target.issue_codes


def test_evaluation_extends_readings_into_unknown_bulk_power_bounds():
    document = _document(GASIR1, 10000.0)
    session = EvaluationSession()
    readings = session.evaluate(document, outputs=("readings",), targets=("len",))
    (plane,) = readings.result.observations
    (arrival,) = plane.arrivals
    assert plane.quantity_state.value == "NOT_COMPUTED"
    assert plane.total_power_W is None and plane.untraced_power_W is None
    assert arrival.beam.power_W is None
    assert arrival.beam.geometric_length_mm == pytest.approx(80)
    assert arrival.beam.optical_path_length_mm is not None
    (geometry,) = readings.result.targets
    assert geometry.target_id == "len" and geometry.status.value == "PASS"

    extended = readings.extend(("plane_bounds",))
    (bounded,) = extended.result.observations
    assert bounded.quantity_state.value == "BOUNDED"
    assert bounded.total_power_W is None and bounded.power_lower_bound_W is None
    assert bounded.power_upper_bound_W == pytest.approx(0.001)
    assert extended.result.targets == (geometry,)

    full = extended.run().result
    assert full.manifest.result_hash == run_optical_project(document).result.manifest.result_hash
    assert [target.status.value for target in full.targets] == ["UNKNOWN", "UNKNOWN", "PASS", "FAIL", "PASS"]
    assert all("BULK_LOSS_NOT_ESTABLISHED" in target.issue_codes for target in full.targets[:2])
    assert session.statistics()["executions"]["beam_trace"] == 1


def _blocked_document(material, nm):
    document = _document(material, nm)
    opaque = {**AR, "coating_id": "OPAQUE", "kind": "DIELECTRIC_HIGH_REFLECTOR",
              "bands": [{**AR["bands"][0], "reflectance": "1", "transmittance": "0"}],
              "basis": "Declared perfectly reflecting, zero-transmission filter"}
    blocker = {"model_id": "block", "model": sequence([surface(1, "0", coating_id="OPAQUE")], [],
               band={"min_nm": str(int(nm - 1)), "max_nm": str(int(nm + 1)), "status": "SPECIFIED"})}
    document["optical_models"].append(blocker)
    document["model_coatings"].append(opaque)
    document["components"].append(declared_component("B", "filter", "block", (10, 0, 0)))
    return document


@pytest.mark.parametrize("material,nm", [(GASIR1, 10000.0), (HWS9, 2200.0)])
def test_opaque_filter_does_not_invent_zero_from_unknown_upstream_bulk(material, nm):
    document = _blocked_document(material, nm)
    document["observations"].append(plane("PRE", (5, 0, 0), radius_mm=20))  # before the opaque coating
    result = run_optical_project(document).result
    observation = next(o for o in result.observations if o.observation_id == "D")
    before = next(o for o in result.observations if o.observation_id == "PRE")
    assert before.arrivals[0].power_upper_bound_W == pytest.approx(.001)
    assert before.total_power_W is None and not before.power_complete
    (arrival,) = observation.arrivals
    assert arrival.beam.power_W is None and arrival.power_upper_bound_W == 0.0
    assert arrival.beam.geometric_length_mm == pytest.approx(80)
    assert arrival.beam.optical_path_length_mm is not None
    # The chief ray is blocked, but energy could leave the unmeasured bulk via
    # a route the bounded trace did not book (scattering, response, bypass).
    assert observation.total_power_W is None and observation.power_upper_bound_W == pytest.approx(.001)
    assert observation.quantity_state.value == "BOUNDED" and not observation.power_complete
    assert result.targets[0].status.value == "UNKNOWN"
    assert result.targets[1].status.value == "UNKNOWN"
    assert "BULK_LOSS_NOT_ESTABLISHED" in [issue.code.value for issue in result.issues]
    assert result.power_budget.conserved
    assert result.power_budget.lines[0].sinks["BULK_LOSS_UNKNOWN"] == pytest.approx(.001)


@pytest.mark.parametrize("bypass", ["bulk_scatter", "surface_scatter", "unmodelled_response"])
def test_unbooked_post_bulk_bypass_never_closes_an_opaque_chief_ray(bypass):
    """Physical off-axis energy may bypass a finite blocker even where its chief-ray T is zero.

    These response paths are not booked separately after bulk power becomes
    unknown. The unresolved bulk bucket must therefore survive to the detector.
    """
    document = _blocked_document(GASIR1, 10000.0)
    document["observations"][0]["aperture"]["radius_mm"] = 40
    if bypass == "bulk_scatter":
        document["model_materials"][0].update(scattering_coefficient_per_mm="0.02", anisotropy_g="0.8")
        missing_sink = "BULK_SCATTER"
    elif bypass == "surface_scatter":
        document["optical_models"][0]["model"]["surfaces"][1]["scatter"] = {
            "bsdf": "LAMBERTIAN", "fraction": "0.1", "side": "TRANSMIT", "sample_count": 1024}
        missing_sink = "SURFACE_SCATTER"
    else:
        response = ttl_model(
            rows=[{"wavelength_nm": "10000", "focal_length_mm": "500", "transmittance": "0.5"}],
            focal_interpolation="POINTS_ONLY", transmittance_interpolation="POINTS_ONLY")
        document["optical_models"].append(response)
        document["components"].append(declared_component("TL", "lens", response["model_id"], (5, 0, 0)))
        missing_sink = "UNMODELLED_RESPONSE"
    result = run_optical_project(document).result
    detector = next(o for o in result.observations if o.observation_id == "D")
    assert detector.quantity_state.value != "KNOWN_ZERO" and not detector.power_complete
    assert detector.power_upper_bound_W is not None and detector.power_upper_bound_W > 0.0009
    assert result.targets[1].status.value == "UNKNOWN"  # the nominally opaque filter proves no maximum
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.issues}
    (line,) = result.power_budget.lines
    assert line.conserved and line.sinks["BULK_LOSS_UNKNOWN"] == pytest.approx(.001)
    assert missing_sink not in line.sinks  # separate bypass power was not booked after uncertainty opened


def test_compacted_child_id_never_turns_unknown_bulk_into_known_zero():
    """Deep compacted descendants cannot justify a zero from a zero-upper parent.

    Without a complete bypass certificate, even a single leaf path with upper
    zero cannot discharge an earlier BULK_LOSS_UNKNOWN ledger bucket.
    """
    root = "SRC#0"
    parent = root + ".0" * 48
    child = child_path_id(parent, root, 0)
    assert "~" in child and not child.startswith(parent + ".")
    path = OpticalPath(path_id=parent, source_id="SRC", spectral_index=0, wavelength_nm=2200.0,
                       segments=(), termination=DETECTED, final_power_W=None,
                       geometric_path_length_mm=0.0, optical_path_length_mm=0.0)
    branch = path.model_copy(update={"path_id": child, "parent_path_id": parent})
    flow = PowerFlow(SINK_BULK_LOSS_UNKNOWN, .001, "SRC", 0, 2200.0, parent, None, None, "CDGM_HWS9")
    trace = TraceOutput(paths=[path, branch], inputs=[SourceInput("SRC", "SRC", 0, 2200.0, .001)], power_flows=[flow])
    scene = SimpleNamespace(surfaces=[SimpleNamespace(
        surface_id="observation/D", interaction=SimpleNamespace(kind="observe", observation_id="D", absorbing=True))])
    arrivals = [SimpleNamespace(observation_id="D", source_id="SRC", spectral_index=0, path_id=parent,
                                power_bound_W=0.0),
                SimpleNamespace(observation_id="D", source_id="SRC", spectral_index=0, path_id=child,
                                power_bound_W=.0005)]
    reach, issues = plane_reach(trace, scene, None, arrivals, (), ("D",), DEFAULT_NUMERICAL_CONFIG, probes=Probes())
    assert not issues
    line = PowerBudgetLine(source_id="SRC", spectral_index=0, wavelength_nm=2200.0,
                           input_power_W=.001, sinks={SINK_BULK_LOSS_UNKNOWN: .001})
    fate = line_fate([line], DEFAULT_NUMERICAL_CONFIG, hazards(reach, "D", [("SRC", 0)]))
    assert fate.untraced_W == pytest.approx(.001) and not fate.complete
    # A leaf with zero geometric arrival is *also* not a blanket certificate
    # that unmodelled scattering or surface-response light cannot bypass it.
    trace.paths = [path]
    leaf_reach, _ = plane_reach(trace, scene, None, arrivals[:1], (), ("D",), DEFAULT_NUMERICAL_CONFIG, probes=Probes())
    leaf_fate = line_fate([line], DEFAULT_NUMERICAL_CONFIG, hazards(leaf_reach, "D", [("SRC", 0)]))
    assert leaf_fate.untraced_W == pytest.approx(.001) and not leaf_fate.complete


def test_nonzero_filter_transmission_cannot_use_the_opaque_zero_proof():
    nm = 2200.0
    document = _document(HWS9, nm)
    document["targets"] = [{"target_id": "leak", "metric": "power_W", "observation_id": "D", "maximum": 1e-9, "unit": "W"}]
    coating = {**AR, "coating_id": "LEAK", "kind": "DIELECTRIC_HIGH_REFLECTOR",
               "bands": [{**AR["bands"][0], "reflectance": "0.999", "transmittance": "0.001"}],
               "basis": "Declared 0.1% transmission, not an opaque filter"}
    blocker = {"model_id": "leaky", "model": sequence([surface(1, "0", coating_id="LEAK")], [],
               band={"min_nm": "2199", "max_nm": "2201", "status": "SPECIFIED"})}
    document["model_coatings"].append(coating)
    document["optical_models"].append(blocker)
    document["components"].append(declared_component("B", "filter", "leaky", (10, 0, 0)))
    result = run_optical_project(document).result
    (arrival,) = result.observations[0].arrivals
    assert arrival.beam.power_W is None and arrival.power_upper_bound_W > 1e-9
    assert result.observations[0].power_upper_bound_W > 1e-9
    assert result.targets[0].status.value == "UNKNOWN"
    assert result.power_budget.conserved


@pytest.mark.parametrize("material,expected", [(HWS3, .988), (HWS9, .998)])
def test_manufacturer_internal_transmission_stays_decisive(material, expected):
    vacuum = 10000 * air_index(10000 * air_index(10000, *STANDARD), *STANDARD)
    result = _run(material, vacuum)
    obs = result.observations[0]
    assert obs.total_power_W == pytest.approx(expected * .001, rel=1e-10)
    assert obs.power_complete
    assert obs.quantity_state.value == "KNOWN"
    assert result.targets[0].status.value == "PASS"
    assert result.targets[1].status.value == "FAIL"
    assert "BULK_LOSS_NOT_ESTABLISHED" not in [i.code.value for i in result.issues]


def test_known_and_unknown_wavelengths_add_only_known_power_to_observation_value():
    vacuum = 10000 * air_index(10000 * air_index(10000, *STANDARD), *STANDARD)
    document = _document(HWS9, 2200)
    document["components"].insert(1, source("KNOWN", (-50, 0, 0), wavelength_nm=vacuum, waist_mm=.5))
    result = run_optical_project(document).result
    obs = result.observations[0]
    arrivals = {a.beam.source_id: a for a in obs.arrivals}
    assert set(arrivals) == {"SRC", "KNOWN"}
    assert arrivals["SRC"].beam.power_W is None and arrivals["SRC"].power_upper_bound_W == pytest.approx(.001)
    assert arrivals["KNOWN"].beam.power_W == pytest.approx(.000998)
    assert obs.total_power_W == pytest.approx(.000998)  # not .001998: unknown is no measured contribution
    assert obs.power_lower_bound_W == pytest.approx(.000998)
    assert obs.power_upper_bound_W == pytest.approx(.001998)
    assert obs.quantity_state.value == "BOUNDED" and not obs.power_complete
    spectral = {row.wavelength_nm: row for row in obs.power_by_wavelength}
    assert spectral[2200].power_W is None and spectral[vacuum].power_W == pytest.approx(.000998)
    assert result.targets[0].status.value == "PASS"  # known contribution alone proves minimum .9mW
    assert result.targets[1].status.value == "FAIL"  # known contribution alone disproves maximum .9mW
    assert result.power_budget.conserved


@pytest.mark.parametrize("material,nm", [(GASIR1, 10000.0), (HWS9, 2200.0)])
def test_serialized_engine_request_keeps_absent_and_out_of_band_loss_unknown(material, nm):
    request = lower_project(_document(material, nm)).request
    assert request.project.materials
    before = next(m for m in request.project.materials if m.material_id == f"project:{material.material_id}")
    serialized = request.model_dump_json()
    after = ValidationRequest.model_validate_json(serialized)
    restored = next(m for m in after.project.materials if m.material_id == before.material_id)
    assert "absorption_coefficient_per_mm" not in before.model_fields_set
    assert "absorption_coefficient_per_mm" not in restored.model_fields_set
    result = validate(after)
    assert result.observations[0].total_power_W is None
    assert result.observations[0].power_upper_bound_W == pytest.approx(.001)
    assert all(t.status.value == "UNKNOWN" for t in result.targets[:2])


def test_explicitly_assumed_and_measured_zero_are_not_unstated():
    model = engine_material(GASIR1, "project:")
    assert MaterialLibrary([model], DEFAULT_NUMERICAL_CONFIG).loss_unstated("project:UMICORE_GASIR1")
    # The declared scalar zero is an assumption, not a manufacturer measurement.
    assumed_doc = GASIR1.model_dump(mode="json", exclude_none=True)
    assumed_doc["absorption_coefficient_per_mm"] = "0"
    assumed = ProjectModelMaterialV1.model_validate(assumed_doc)
    assumed_model = engine_material(assumed, "project:")
    assert "absorption_coefficient_per_mm" in assumed_model.model_dump(mode="json")
    assert not MaterialLibrary([type(assumed_model).model_validate_json(assumed_model.model_dump_json())],
                               DEFAULT_NUMERICAL_CONFIG).loss_unstated("project:UMICORE_GASIR1")
    assumed_run = _run(assumed, 10000)
    assert assumed_run.observations[0].total_power_W == pytest.approx(.001)
    assert all(i.code.value != "BULK_LOSS_NOT_ESTABLISHED" for i in assumed_run.issues)
    measured_doc = HWS9.model_dump(mode="json", exclude_none=True)
    measured_doc["bulk_loss"]["internal_transmittance"] = ["1"] * len(measured_doc["bulk_loss"]["internal_transmittance"])
    measured = ProjectModelMaterialV1.model_validate(measured_doc)
    assert not MaterialLibrary([engine_material(measured, "project:")], DEFAULT_NUMERICAL_CONFIG).loss_unstated("project:CDGM_HWS9")
    vacuum = 10000 * air_index(10000 * air_index(10000, *STANDARD), *STANDARD)
    measured_run = _run(measured, vacuum)
    assert measured_run.observations[0].total_power_W == pytest.approx(.001)
    assert all(i.code.value != "BULK_LOSS_NOT_ESTABLISHED" for i in measured_run.issues)
