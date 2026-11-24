"""Position-dependent neutral-density attenuators (``NEUTRAL_DENSITY`` models; ``optics/attenuation.py``).

Every law is checked against its analytic transmittance ``10^-OD`` written out here: the chief-ray transmittance carried
by the beam, the per-point law at arbitrary points of the plane, the wheel rotation / gradient shift chosen with the
component's position parameter, the uncoated parts, the reflected branch and the power ledger, and the density range
across a beam's 1/e^2 footprint (out of model scope beyond ``ND_FOOTPRINT_OD_TOLERANCE``). Every model below is fixture
data declared in the test Project.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.local import LocalRunError, LocalRunErrorCode, lower_project, run_optical_project
from os_optics.optics.attenuation import ND_FOOTPRINT_OD_TOLERANCE, AttenuatorG, transmission_error_estimate

from declared_helpers import compound_model, cylinder, declared_component, with_models
from local_helpers import NM, plane, project, rayleigh_range, source

BAND = {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}
POWER_W = 0.001
SOURCE_X = -100.0


def nd_model(law: dict, model_id: str = "nd", **extra) -> dict:
    model = {"model_kind": "NEUTRAL_DENSITY", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "law": law, "clear_radius_mm": "20",
             "edge_radius_mm": "22", "valid_band_nm": BAND, "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}
    model.update(extra)
    return {"model_id": model_id, "model": model}


GRADIENT = {"kind": "LINEAR_GRADIENT", "od_start": "0.5", "od_end": "2.5", "x_start_mm": "-10", "x_end_mm": "10"}
ROTARY = {"kind": "LINEAR_ROTARY", "od_start": "0.1", "od_end": "3.1", "arc_start_deg": "0", "arc_end_deg": "270"}
STEPS = {"kind": "STEPPED_ROTARY", "steps": [{"start_deg": "0", "end_deg": "90", "optical_density": "0.5"},
                                             {"start_deg": "90", "end_deg": "180", "optical_density": "1"},
                                             {"start_deg": "200", "end_deg": "300", "optical_density": "2"}]}


def _document(model: dict, *, beam_yz=(0.0, 0.0), waist_mm: float = 0.2, parameters: list[dict] | None = None,
              wavelength_nm: float = 632.8) -> dict:
    """A beam along +X at design (y, z) = ``beam_yz`` through the attenuator at the origin (local X = design +Y, local Y =
    design +Z) to a detector plane."""
    laser = source(position=(SOURCE_X, *beam_yz), wavelength_nm=wavelength_nm, waist_mm=waist_mm, power_w=POWER_W)
    nd = declared_component("ND", "filter", model["model_id"])
    if parameters is not None:
        nd["selected_parameters"] = parameters
    document = project([laser, nd], [plane("det", (50.0, *beam_yz), radius_mm=20.0)],
                       [{"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0, "unit": "W"}])
    return with_models(document, [model])


def _position(key: str, value: str, unit: str | None) -> list[dict]:
    parameter = {"semantic_key": key, "value_kind": "DECIMAL", "value": value}
    if unit is not None:
        parameter["unit_code"] = unit
    return [parameter]


def _power(run) -> float:
    """The traced power reaching the detector (the arrival's: a power target is out of model scope behind a variation)."""
    (arrival,) = run.result.observations[0].arrivals
    (target,) = run.result.targets
    if not arrival.power_out_of_scope_at:
        assert target.value == arrival.beam.power_W
    return arrival.beam.power_W


def _gradient_od(x: float, offset: float = 0.0) -> float:
    s = x - offset
    return 0.5 + 2.0 * (s + 10.0) / 20.0


def _rotary_od(azimuth_deg: float, rotation: float = 0.0) -> float | None:
    a = (azimuth_deg - rotation) % 360.0
    return 0.1 + 3.0 * a / 270.0 if a <= 270.0 else None


# ------------------------------------------------------------------------------------------------ fixed density


def test_a_fixed_density_transmits_ten_to_minus_od_and_absorbs_the_rest():
    run = run_optical_project(_document(nd_model({"kind": "FIXED", "optical_density": "1.3"})))
    assert _power(run) == pytest.approx(POWER_W * 10.0 ** -1.3, rel=1e-12)
    sinks = run.result.power_budget.total_sinks
    assert sinks["ATTENUATOR_ABSORPTION"] == pytest.approx(POWER_W * (1.0 - 10.0 ** -1.3), rel=1e-12)
    assert run.status.value == "PASS" and run.model_scope_issues == ()


def test_a_reflecting_attenuator_traces_its_reflection_and_absorbs_one_minus_r_minus_t():
    run = run_optical_project(_document(nd_model({"kind": "FIXED", "optical_density": "1"}, reflectance="0.05")))
    t = 0.1
    assert _power(run) == pytest.approx(POWER_W * t, rel=1e-12)
    reflected = [p for p in run.result.paths if p.branch_kind == "attenuator_reflection"]
    assert len(reflected) == 1 and reflected[0].final_power_W == pytest.approx(POWER_W * 0.05, rel=1e-12)
    assert reflected[0].segments[0].end_mm is None or reflected[0].termination == "ESCAPED"
    sinks = run.result.power_budget.total_sinks
    assert sinks["ATTENUATOR_ABSORPTION"] == pytest.approx(POWER_W * (1.0 - 0.05 - t), rel=1e-12)
    assert run.result.power_budget.conserved


# ------------------------------------------------------------------------------------------------ linear gradient


@pytest.mark.parametrize("offset", [0.0, 2.5, -7.0])
def test_a_linear_gradient_follows_the_beam_position_and_the_offset_parameter(offset: float):
    y0 = 3.0  # local X of the hit point
    run = run_optical_project(_document(nd_model(GRADIENT, position_parameter="nd_offset_mm"), beam_yz=(y0, 0.0),
                                        parameters=_position("nd_offset_mm", f"{offset:g}", "mm")))
    assert _power(run) == pytest.approx(POWER_W * 10.0 ** -_gradient_od(y0, offset), rel=1e-12)


def test_beyond_the_gradient_the_uncoated_transmittance_applies_or_the_beam_is_out_of_scope():
    with_uncoated = nd_model(GRADIENT, uncoated_transmittance="0.9", position_parameter="nd_offset_mm")
    run = run_optical_project(_document(with_uncoated, beam_yz=(15.0, 0.0)))
    assert _power(run) == pytest.approx(POWER_W * 0.9, rel=1e-12)
    run = run_optical_project(_document(nd_model(GRADIENT), beam_yz=(15.0, 0.0)))
    codes = {(i.code.value, i.status.value) for i in run.result.issues}
    assert ("ND_UNCOATED_REGION", "OUT_OF_MODEL_SCOPE") in codes
    assert run.result.power_budget.total_sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(POWER_W, rel=1e-12)
    (target,) = run.result.targets
    # (no value; the plane's proven interval [0, bound] meets the target's minimum of 0 W, and nothing above it)
    assert target.value is None and target.assessment.value == "MET"
    # (the light passes on straight with a power that is not computed, at most what arrived)
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.beam.power_W is None and arrival.power_upper_bound_W == pytest.approx(POWER_W, rel=1e-12)


def test_the_density_range_across_the_footprint_is_reported_beyond_the_tolerance():
    waist = 2.0  # a large beam on a steep gradient: 0.1 OD / mm over a 4 mm diameter
    y0 = 1.0
    run = run_optical_project(_document(nd_model(GRADIENT), beam_yz=(y0, 0.0), waist_mm=waist))
    (issue,) = [i for i in run.result.issues if i.code.value == "ND_DENSITY_VARIES_ACROSS_BEAM"]
    w = waist * math.sqrt(1.0 + (-SOURCE_X / rayleigh_range(waist, 632.8)) ** 2)  # radius at the attenuator (independent)
    ctx = issue.numerical_context
    assert ctx["od_min"] == pytest.approx(_gradient_od(y0 - w), rel=1e-9) and ctx["od_max"] == pytest.approx(_gradient_od(y0 + w), rel=1e-9)
    assert ctx["od_variation"] == pytest.approx(0.1 * 2.0 * w, rel=1e-9) and ctx["od_variation"] > ND_FOOTPRINT_OD_TOLERANCE
    assert ctx["transmission_error_estimate"] == pytest.approx(transmission_error_estimate(0.2 * w), rel=1e-12)
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and run.result.status.value == "OUT_OF_MODEL_SCOPE"
    # the chief-ray transmittance is still the beam's traced power; its Gaussian state is withheld from there on, and its power
    # is out of model scope: a power target on it has no value (M2)
    assert _power(run) == pytest.approx(POWER_W * 10.0 ** -_gradient_od(y0), rel=1e-12)
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.beam.gaussian.valid is False and arrival.beam.gaussian.reason == "ND_DENSITY_VARIES_ACROSS_BEAM"
    assert arrival.power_out_of_scope_at == ("ND/attenuator",)
    (target,) = run.result.targets
    assert (target.status.value, target.value) == ("OUT_OF_MODEL_SCOPE", None)
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" in target.issue_codes and "ND/attenuator" in target.message
    small = run_optical_project(_document(nd_model(GRADIENT), beam_yz=(y0, 0.0), waist_mm=0.2))
    assert not any(i.code.value == "ND_DENSITY_VARIES_ACROSS_BEAM" for i in small.result.issues)


def test_the_beam_averaged_transmittance_bound_matches_a_numerical_average():
    """The stated estimate exp((ln 10 dOD / 2)^2 / 8) - 1 is the exact beam-averaged / chief-ray transmittance ratio minus
    one for a Gaussian behind a linear gradient (checked by quadrature)."""
    delta, w = 0.4, 1.0
    b = delta / (2.0 * w)
    x = np.linspace(-6.0 * w, 6.0 * w, 20001)
    weight = np.exp(-2.0 * x**2 / w**2)
    ratio = float(np.sum(weight * 10.0 ** (-b * x)) / np.sum(weight))
    assert ratio - 1.0 == pytest.approx(transmission_error_estimate(delta), rel=1e-6)


# ------------------------------------------------------------------------------------------------ rotary laws


@pytest.mark.parametrize(("azimuth", "rotation"), [(30.0, 0.0), (135.0, 0.0), (30.0, -100.0), (250.0, 45.0)])
def test_a_linear_rotary_wheel_follows_the_hit_azimuth_and_the_wheel_rotation(azimuth: float, rotation: float):
    r = 8.0
    beam = (r * math.cos(math.radians(azimuth)), r * math.sin(math.radians(azimuth)))  # local (X, Y) = design (y, z)
    run = run_optical_project(_document(nd_model(ROTARY, position_parameter="nd_rotation_deg", uncoated_transmittance="0.92"),
                                        beam_yz=beam, parameters=_position("nd_rotation_deg", f"{rotation:g}", None)))
    od = _rotary_od(azimuth, rotation)
    assert _power(run) == pytest.approx(POWER_W * (0.92 if od is None else 10.0 ** -od), rel=1e-9)


@pytest.mark.parametrize(("azimuth", "expected_t"), [(45.0, 10.0 ** -0.5), (120.0, 0.1), (190.0, 0.95), (250.0, 0.01)])
def test_a_stepped_wheel_gives_each_sector_its_density(azimuth: float, expected_t: float):
    r = 8.0
    beam = (r * math.cos(math.radians(azimuth)), r * math.sin(math.radians(azimuth)))
    run = run_optical_project(_document(nd_model(STEPS, uncoated_transmittance="0.95"), beam_yz=beam))
    assert _power(run) == pytest.approx(POWER_W * expected_t, rel=1e-9)


def test_the_law_is_evaluated_at_every_point_of_the_plane():
    """``AttenuatorG.transmittance`` per point (what every ray of a packet gets) against the analytic laws."""
    origin, normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    x_axis, y_axis = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(7)
    points = np.column_stack([rng.uniform(-15, 15, 200), rng.uniform(-15, 15, 200), np.zeros(200)])
    gradient = AttenuatorG("linear_gradient", origin, normal, x_axis, y_axis, 1.5, 0.0, 0.8, od_start=0.5, od_end=2.5, start=-10.0, end=10.0)
    want = [10.0 ** -_gradient_od(p[0], 1.5) if -10.0 <= p[0] - 1.5 <= 10.0 else 0.8 for p in points]
    assert gradient.transmittance(points) == pytest.approx(want, rel=1e-12)
    rotary = AttenuatorG("linear_rotary", origin, normal, x_axis, y_axis, 30.0, 0.0, None, od_start=0.1, od_end=3.1, start=0.0, end=270.0)
    got = rotary.transmittance(points)
    for p, t in zip(points, got):
        od = _rotary_od(math.degrees(math.atan2(p[1], p[0])), 30.0)
        assert (math.isnan(t) if od is None else t == pytest.approx(10.0 ** -od, rel=1e-12))
    fixed = AttenuatorG("fixed", origin, normal, x_axis, y_axis, 0.0, 0.0, None, optical_density=2.0)
    assert fixed.transmittance(points) == pytest.approx([0.01] * 200, rel=1e-15)


# ------------------------------------------------------------------------------------------------ Project rules


def test_position_parameters_are_checked():
    model = nd_model(GRADIENT, position_parameter="nd_offset_mm")
    wrong_unit = _document(model, parameters=_position("nd_offset_mm", "1", "deg"))
    with pytest.raises(LocalRunError) as caught:
        lower_project(wrong_unit)
    assert [p.code for p in caught.value.problems] == [LocalRunErrorCode.SELECTED_PARAMETERS_UNSUPPORTED]
    other_key = _document(model, parameters=_position("nd_rotation_deg", "1", "deg"))
    with pytest.raises(LocalRunError) as caught:
        lower_project(other_key)
    assert "nd_offset_mm" in caught.value.problems[0].message
    (nd,) = [c for c in lower_project(_document(model)).request.project.components if c.instance_id == "ND"]
    assert nd.surfaces[0].interaction.position == 0.0 and nd.surfaces[0].surface_id == "attenuator"


def test_the_band_is_checked_and_the_view_draws_the_attenuator(tmp_path):
    run = run_optical_project(_document(nd_model({"kind": "FIXED", "optical_density": "1"}), wavelength_nm=1200.0))
    assert [i.code for i in run.model_scope_issues] == ["MODEL_WAVELENGTH_OUTSIDE_BAND"]
    run = run_optical_project(_document(nd_model(GRADIENT, position_parameter="nd_offset_mm"),
                                        parameters=_position("nd_offset_mm", "2", "mm")), output_dir=str(tmp_path / "run"))
    (element,) = [e for e in run.optical_view.elements if e.element_id == "ND"]
    assert element.geometry.model_kind == "NEUTRAL_DENSITY"
    assert [(s.surface_id, s.role.value) for s in element.geometry.surfaces] == [("attenuator", "ATTENUATOR")]
    assert (tmp_path / "run" / "optical_scene.glb").exists()


def test_a_rectangular_attenuator_in_a_compound_unit_turns_its_law_with_the_member():
    law = {"kind": "LINEAR_GRADIENT", "od_start": "0.5", "od_end": "2.5", "x_start_mm": "-10", "x_end_mm": "10"}
    model = nd_model(law, clear_half_widths_mm=["12", "8"], edge_half_widths_mm=["13", "9"])
    del model["model"]["clear_radius_mm"], model["model"]["edge_radius_mm"]
    s, c = math.sqrt(0.5), math.sqrt(0.5)
    unit = compound_model("unit", [("A", "nd", (0.0, 0.0, 0.0), (c, 0.0, 0.0, s))], cylinder("40", "-1", "1"))  # member turned 90 deg
    laser = source(position=(SOURCE_X, 0.0, 3.0), wavelength_nm=632.8, waist_mm=0.2, power_w=POWER_W)
    document = with_models(project([laser, declared_component("U", "filter", "unit")], [plane("det", (50.0, 0.0, 3.0), radius_mm=20.0)],
                                   [{"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0, "unit": "W"}]),
                           [unit, copy.deepcopy(model)])
    run = run_optical_project(document)
    # design z = unit local Y; the member's local X is the unit's local Y after the 90 deg turn about Z: law x = 3 mm
    assert _power(run) == pytest.approx(POWER_W * 10.0 ** -_gradient_od(3.0), rel=1e-9)


def test_a_wheels_stated_linearity_is_named_with_its_model_and_does_not_change_the_results():
    """PV-O3: a LINEAR_ROTARY wheel that states its linearity (+-5 %) runs on the nominal linear law, and the run names that in
    the element's model row (``notes``); the results equal those of the same wheel without the statement."""

    beam = {"beam_yz": (5.0, 0.0)}
    stated = run_optical_project(_document(nd_model({**ROTARY, "linearity_percent": "5"}), **beam))
    plain = run_optical_project(_document(nd_model(ROTARY), **beam))
    (row,) = [r for r in stated.element_models if r["instance_id"] == "ND"]
    assert row["notes"] == ["The optical density follows the nominal linear law of azimuth from 0.1 to 3.1; the stated linearity "
                            "of \u00b15 % is not applied to the results."]
    assert "notes" not in next(r for r in plain.element_models if r["instance_id"] == "ND")
    assert [(t.status, t.value) for t in stated.result.targets] == [(t.status, t.value) for t in plain.result.targets]
    assert stated.local_validation.result_hash == plain.local_validation.result_hash
