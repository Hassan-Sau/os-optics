"""A neutral-density filter wheel coated on an annulus around an uncoated hub (``coated_inner_radius_mm``).

Every case is checked against the analytic transmittance written out here: ``10^-OD`` of the law on the coated annulus,
``uncoated_transmittance`` inside the hub (or out of model scope without one), and the out-of-scope rule for a beam whose
1/e^2 footprint crosses the hub's edge (a density step larger than ``ND_FOOTPRINT_OD_TOLERANCE``), including a hub that
lies inside a large footprint without touching any of its ring samples. Every model below is fixture data declared in
the test Project.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics.contracts.optics import FixedDensity, NeutralDensityInteraction
from os_optics.local import lower_project, run_optical_project
from os_optics.optics.attenuation import ND_FOOTPRINT_OD_TOLERANCE, AttenuatorG

from test_neutral_density import POWER_W, _document, _position, _power, nd_model

INNER = 4.75
UNCOATED_T = 0.95
ROTARY = {"kind": "LINEAR_ROTARY", "od_start": "0.02", "od_end": "2", "arc_start_deg": "0", "arc_end_deg": "270"}


def _wheel(**extra) -> dict:
    fields = {"coated_inner_radius_mm": f"{INNER:g}", "uncoated_transmittance": f"{UNCOATED_T:g}", "position_parameter": "nd_rotation_deg"}
    fields.update(extra)
    return nd_model(ROTARY, **{k: v for k, v in fields.items() if v is not None})


def _rotary_od(azimuth_deg: float, rotation: float = 0.0) -> float | None:
    a = (azimuth_deg - rotation) % 360.0
    return 0.02 + 1.98 * a / 270.0 if a <= 270.0 else None


def _at(radius: float, azimuth_deg: float) -> tuple[float, float]:
    return radius * math.cos(math.radians(azimuth_deg)), radius * math.sin(math.radians(azimuth_deg))


def _issue_codes(run) -> set[str]:
    return {i.code.value for i in run.result.issues}


def test_the_annulus_lowers_onto_the_attenuator_interaction():
    lowered = lower_project(_document(_wheel()))
    (nd,) = [c for c in lowered.request.project.components if c.instance_id == "ND"]
    interaction = nd.surfaces[0].interaction
    assert interaction.coated_inner_radius_mm == INNER and interaction.uncoated_transmittance == UNCOATED_T
    # an attenuator without a hub keeps its earlier serialized form (the field is omitted)
    plain = NeutralDensityInteraction(law=FixedDensity(optical_density=1.0))
    assert "coated_inner_radius_mm" not in plain.model_dump(mode="json")
    with pytest.raises(ValueError, match="position-dependent"):
        NeutralDensityInteraction(law=FixedDensity(optical_density=1.0), coated_inner_radius_mm=2.0)


@pytest.mark.parametrize(("radius", "azimuth", "rotation"), [(8.0, 0.0, 0.0), (8.0, 90.0, 0.0), (6.0, 200.0, 30.0), (9.0, 300.0, 0.0)])
def test_on_the_annulus_the_law_applies(radius: float, azimuth: float, rotation: float):
    run = run_optical_project(_document(_wheel(), beam_yz=_at(radius, azimuth), parameters=_position("nd_rotation_deg", f"{rotation:g}", None)))
    od = _rotary_od(azimuth, rotation)
    assert _power(run) == pytest.approx(POWER_W * (UNCOATED_T if od is None else 10.0 ** -od), rel=1e-9)
    assert run.result.status.value == "PASS"


@pytest.mark.parametrize(("radius", "azimuth"), [(0.0, 0.0), (2.0, 90.0), (3.5, 180.0)])
def test_inside_the_hub_the_plate_is_uncoated(radius: float, azimuth: float):
    run = run_optical_project(_document(_wheel(), beam_yz=_at(radius, azimuth), waist_mm=0.2))
    assert _power(run) == pytest.approx(POWER_W * UNCOATED_T, rel=1e-12)
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" not in _issue_codes(run)


def test_inside_the_hub_without_an_uncoated_transmittance_the_beam_is_out_of_model_scope():
    run = run_optical_project(_document(_wheel(uncoated_transmittance=None), beam_yz=_at(2.0, 90.0), waist_mm=0.2))
    assert "ND_UNCOATED_REGION" in _issue_codes(run) and run.result.status.value == "OUT_OF_MODEL_SCOPE"
    assert run.result.power_budget.total_sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(POWER_W, rel=1e-12)


def test_a_footprint_straddling_the_inner_radius_is_out_of_model_scope():
    # at azimuth 90 deg the coating has OD 0.68 next to the hub's 0.022: a step of 0.66 across the hub's edge
    straddling = run_optical_project(_document(_wheel(), beam_yz=_at(INNER, 90.0), waist_mm=0.5))
    (issue,) = [i for i in straddling.result.issues if i.code.value == "ND_DENSITY_VARIES_ACROSS_BEAM"]
    ctx = issue.numerical_context
    assert ctx["od_min"] == pytest.approx(-math.log10(UNCOATED_T), rel=1e-9)
    assert ctx["od_variation"] > ND_FOOTPRINT_OD_TOLERANCE and issue.status.value == "OUT_OF_MODEL_SCOPE"
    # the same beam well out on the annulus stays in scope (the law changes by ~0.05 OD across it there)
    clear = run_optical_project(_document(_wheel(), beam_yz=_at(8.0, 90.0), waist_mm=0.5))
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" not in _issue_codes(clear) and clear.result.status.value == "PASS"
    # without an uncoated transmittance the crossing is out of scope too (part of the beam meets an undefined plate)
    undefined = run_optical_project(_document(_wheel(uncoated_transmittance=None), beam_yz=_at(INNER + 0.3, 90.0), waist_mm=0.5))
    assert "ND_DENSITY_VARIES_ACROSS_BEAM" in _issue_codes(undefined)


def test_a_hub_inside_a_large_footprint_is_seen_even_when_no_ring_sample_falls_on_it():
    origin, normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    x_axis, y_axis = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # a shallow gradient (0.05 OD across the beam) with a 1 mm hub; the beam (1/e^2 radius 5 mm) is centred 3 mm away
    gradient = AttenuatorG("linear_gradient", origin, normal, x_axis, y_axis, 0.0, 0.0, UNCOATED_T, od_start=0.5, od_end=0.6,
                           start=-10.0, end=10.0, inner_radius=1.0)
    center, direction = np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 1.0])
    points = gradient.footprint_points(center, direction, x_axis, y_axis, 5.0, 5.0)
    ring = points[1:-1]
    assert np.all(np.hypot(ring[:, 0], ring[:, 1]) > 1.0)  # no ring sample on the hub
    assert np.allclose(points[-1], origin)  # the hub centre is inside the footprint: sampled itself
    od_min, od_max, defined = gradient.footprint_density_range(points)
    assert defined and od_min == pytest.approx(-math.log10(UNCOATED_T)) and od_max - od_min > ND_FOOTPRINT_OD_TOLERANCE
    # a footprint beside the hub adds its point nearest the hub centre along the line toward it (on the 1/e^2 ellipse)
    beside = gradient.footprint_points(np.array([0.0, 4.0, 0.0]), direction, x_axis, y_axis, 2.0, 2.0)
    assert np.allclose(beside[-1], [0.0, 2.0, 0.0])


def test_the_hub_moves_with_a_shifted_gradient_and_stays_on_the_axis_of_a_turned_wheel():
    origin, normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    x_axis, y_axis = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    shifted = AttenuatorG("linear_gradient", origin, normal, x_axis, y_axis, 5.0, 0.0, 0.9, od_start=0.5, od_end=2.5, start=-10.0,
                          end=10.0, inner_radius=1.0)
    t = shifted.transmittance(np.array([[5.5, 0.0, 0.0], [0.5, 0.0, 0.0], [7.0, 0.0, 0.0]]))
    assert t[0] == pytest.approx(0.9) and t[1] == pytest.approx(10.0 ** -(0.5 + 2.0 * (0.5 - 5.0 + 10.0) / 20.0))
    assert t[2] == pytest.approx(10.0 ** -(0.5 + 2.0 * (7.0 - 5.0 + 10.0) / 20.0))
    turned = AttenuatorG("linear_rotary", origin, normal, x_axis, y_axis, 30.0, 0.0, None, od_start=0.02, od_end=2.0, start=0.0,
                         end=270.0, inner_radius=INNER)
    od, coated = turned.density(np.array([[0.0, 4.0, 0.0], [0.0, 6.0, 0.0]]))
    assert not coated[0] and coated[1] and od[1] == pytest.approx(_rotary_od(90.0, 30.0))


def test_an_unplaced_declared_wheel_with_a_hub_lowers():
    document = _document(_wheel())
    document["optical_models"].append({"model_id": "spare", "model": copy.deepcopy(_wheel()["model"])})
    assert [e.instance_id for e in lower_project(document).elements] == ["ND"]
