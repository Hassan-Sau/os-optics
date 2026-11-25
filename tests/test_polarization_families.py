"""Polarization model families: retardance stated per wavelength (a dual-wavelength plate), the circular polarizer (a linear
polarizer and a retarder behind it, an isolator against its own reflection) and a rotation parameter of a polarization
element's axes (the half-wave plate of a tunable splitter / attenuator).

References: plain-numpy Jones matrices written here (retarder ``R(a) diag(e^{-i d/2}, e^{i d/2}) R(-a)``, polarizer
``diag(sqrt(T), sqrt(T/ER))``), Malus' law ``cos^2(2 theta)`` of a half-wave plate turned by ``theta`` before a polarizer. Fixture
data (transmittances, extinction ratios, retardances) are made up for the test.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot
from os_optics.local import lower_project

from declared_helpers import declared_component, with_models
from helpers import run as run_request
from local_helpers import POWER_W, plane, project, run, source
from test_polarization_extensions import _turned, _value, chain, out_power, polarizer, retarder


def tabulated_retarder(rows: list[tuple[str, str]], interpolation: str = "POINTS_ONLY", design: tuple[str, str] | None = None) -> dict:
    design = design or rows[0]
    model = retarder(None, waves=design[1])
    model["model"].update(retardance_dispersion="TABULATED", design_wavelength_nm=design[0], retardance_interpolation=interpolation,
                          retardance_rows=[{"wavelength_nm": w, "retardance_waves": r} for w, r in rows])
    model["model"]["transmittance"] = _value("1")
    return model


def circular(axis_deg: str = "45", offset: str = "1", transmittance: str = "0.8", extinction: str = "1000") -> dict:
    model = polarizer(None, transmittance=transmittance, extinction=extinction)
    model["model_id"] = "cp"
    model["model"].update(element="CIRCULAR_POLARIZER", retardance_waves="0.25", design_wavelength_nm="780", retardance_dispersion="CONSTANT",
                          retarder_axis_deg=axis_deg, retarder_offset_mm=offset)
    return model


def _stokes(arrival) -> np.ndarray:
    j = arrival.beam.polarization
    ex, ey = complex(*j.ex), complex(*j.ey)
    s0 = abs(ex) ** 2 + abs(ey) ** 2
    return np.array([s0, abs(ex) ** 2 - abs(ey) ** 2, 2 * (ex.conjugate() * ey).real, 2 * (ex.conjugate() * ey).imag]) / s0


def _out(run_):
    (obs,) = [o for o in run_.result.observations if o.observation_id == "OUT"]
    return obs.arrivals


# ------------------------------------------------------------------------------------------- tabulated retardance


DUAL = [("532", "1"), ("1064", "0.5")]


@pytest.mark.parametrize("wavelength_nm,through", [(1064.0, 1.0), (532.0, 0.0)])
def test_a_dual_wavelength_plate_is_a_half_wave_plate_at_one_line_and_a_full_wave_at_the_other(wavelength_nm: float, through: float) -> None:
    """Plate axis at 45 deg between crossed polarizers: half wave at 1064 nm turns the light by 90 deg (all of it passes the
    crossed analyzer), a full wave at 532 nm leaves it unchanged (none passes, up to the analyzer's extinction)."""
    plate = tabulated_retarder(DUAL, design=("1064", "0.5"))
    analyzer = polarizer(None, transmittance="1", extinction="1000000")
    analyzer["model_id"] = "an"
    run_ = run(chain(source(wavelength_nm=wavelength_nm), ("W", "waveplate", plate, 45.0), ("A", "polarizer", analyzer, 90.0)))
    expected = POWER_W * (through + (1.0 - through) / 1e6)
    assert out_power(run_) == pytest.approx(expected, rel=1e-9)
    assert run_.model_scope_issues == ()


def test_a_linear_retardance_table_interpolates_and_a_points_only_table_refuses_other_wavelengths() -> None:
    """LINEAR rows (700 nm: 0.25 wave, 900 nm: 0.75 wave): at 800 nm a half wave; POINTS_ONLY: 800 nm is outside the stated
    response, the light stops at the plate (MODEL_WAVELENGTH_OUTSIDE_BAND, out of model scope, RETARDANCE_UNAVAILABLE)."""
    rows = [("700", "0.25"), ("900", "0.75")]
    analyzer = polarizer(None, transmittance="1", extinction="1000000")
    analyzer["model_id"] = "an"
    linear = run(chain(source(wavelength_nm=800.0), ("W", "waveplate", tabulated_retarder(rows, "LINEAR"), 45.0), ("A", "polarizer", analyzer, 90.0)))
    assert out_power(linear) == pytest.approx(POWER_W, rel=1e-9)
    points = run(chain(source(wavelength_nm=800.0), ("W", "waveplate", tabulated_retarder(rows), 45.0), ("A", "polarizer", analyzer, 90.0)))
    # (behind the plate the light goes on straight with a power and polarization that are not computed, at most what arrived)
    (behind,) = _out(points)
    assert behind.beam.power_W is None and behind.power_upper_bound_W == pytest.approx(POWER_W, rel=1e-12)
    codes = {i.code.value: i for i in points.result.issues}
    assert codes["MODEL_WAVELENGTH_OUTSIDE_BAND"].status.value == "OUT_OF_MODEL_SCOPE"
    assert "700.0, 900.0 nm exactly" in codes["MODEL_WAVELENGTH_OUTSIDE_BAND"].message
    assert codes["MODEL_WAVELENGTH_OUTSIDE_BAND"].numerical_context["power_upper_bound_W"] == behind.power_upper_bound_W
    assert [(p.termination, p.final_power_W) for p in points.result.paths] == [("ESCAPED", None)]


def test_the_tabulated_retarder_lowers_to_its_rows() -> None:
    lowered = lower_project(chain(source(wavelength_nm=1064.0), ("W", "waveplate", tabulated_retarder(DUAL, design=("1064", "0.5")), 0.0)))
    (component,) = [c for c in lowered.request.project.components if c.instance_id == "W"]
    inter = component.surfaces[0].interaction
    assert inter.retardance_table_nm == (532.0, 1064.0)
    assert inter.retardance_table_rad == pytest.approx((2.0 * math.pi, math.pi), rel=1e-15)
    assert inter.retardance_interpolation == "points_only"


@pytest.mark.parametrize("change,message", [
    ({"retardance_rows": None}, "TABULATED retardance_dispersion states retardance_rows"),
    ({"retardance_dispersion": "CONSTANT"}, "belong to a TABULATED"),
    ({"design_wavelength_nm": "800"}, "design point"),
])
def test_a_tabulated_retarder_states_consistent_rows(change: dict, message: str) -> None:
    from os_optics._vendor.shared_contracts.design_catalog import PolarizationElementModelV1

    model = tabulated_retarder(DUAL, design=("1064", "0.5"))["model"]
    model.update(change)
    with pytest.raises(ValueError, match=message):
        PolarizationElementModelV1.model_validate({k: v for k, v in model.items() if v is not None})


# ------------------------------------------------------------------------------------------------ circular polarizer


@pytest.mark.parametrize("axis_deg", ["45", "-45"])
def test_a_circular_polarizer_makes_circular_light_of_its_retarder_axis_handedness(axis_deg: str) -> None:
    """Light polarized at 30 deg to the polarizer axis: the polarizer passes T cos^2(30) of it (plus T/ER of the rest), the
    quarter-wave retarder at +-45 deg turns it circular: normalized S3 = +-1 (opposite for the two retarder axes, the same
    convention as the plain Jones product of the test). The element is two planes 1 mm apart."""
    run_ = run(chain(source(polarization_deg=30.0), ("C", "polarizer", circular(axis_deg), 0.0)))
    (arrival,) = _out(run_)
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    assert arrival.beam.power_W == pytest.approx(POWER_W * 0.8 * (c * c + s * s / 1000.0), rel=1e-12)
    stokes = _stokes(arrival)
    # the reference: polarizer then retarder at +-45 deg, in the element's (x, y) axes
    a = math.radians(float(axis_deg))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    qwp = rot @ np.diag([np.exp(-1j * math.pi / 4), np.exp(1j * math.pi / 4)]) @ rot.T
    out = qwp @ np.diag([math.sqrt(0.8), math.sqrt(0.8 / 1000.0)]) @ np.array([c, s])
    ref = np.array([abs(out[0]) ** 2 - abs(out[1]) ** 2, 2 * (out[0].conjugate() * out[1]).real, 2 * (out[0].conjugate() * out[1]).imag]) / np.vdot(out, out).real
    # the arrival's basis is the source's (x along the polarizer axis at 0 deg, y = n x x): the same Stokes parameters
    np.testing.assert_allclose(stokes[1:], ref, atol=1e-9)


def test_the_two_retarder_axes_give_opposite_handedness() -> None:
    plus = _stokes(_out(run(chain(source(), ("C", "polarizer", circular("45"), 0.0))))[0])
    minus = _stokes(_out(run(chain(source(), ("C", "polarizer", circular("-45"), 0.0))))[0])
    assert plus[3] == pytest.approx(-minus[3], abs=1e-12)
    assert abs(plus[3]) == pytest.approx(1.0, abs=1e-6)


def test_a_circular_polarizer_isolates_its_own_reflection() -> None:
    """Polarizer side toward the source, a mirror at normal incidence behind the retarder: the returning light crosses the
    retarder again (a half wave in all: turned by 90 deg) and the polarizer blocks it. Returned power = P T (T / ER); with the
    retarder removed (a linear polarizer only) the polarizer passes it back: P T^2."""
    lowered = lower_project(chain(source(), ("C", "polarizer", circular("45"), 0.0)))
    linear = lower_project(chain(source(), ("C", "polarizer", polarizer(None), 0.0)))
    returned = {}
    for name, low in (("circular", lowered), ("linear", linear)):
        project_ = low.request.project
        mirror = B.mirror("M", position=(30.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
        back = B.observation_plane("BACK", position=(-100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=50.0)
        snapshot = ProjectSnapshot(materials=project_.materials, components=(*project_.components, mirror), observations=(back,))
        result = run_request(snapshot)
        returned[name] = sum(a.beam.power_W for o in result.observations for a in o.arrivals if a.beam.direction[0] < 0.0)
    assert returned["linear"] == pytest.approx(POWER_W * 0.8 * 0.8, rel=1e-9)
    assert returned["circular"] == pytest.approx(POWER_W * 0.8 * 0.8 / 1000.0, rel=1e-9)


@pytest.mark.parametrize("change,message", [
    ({"retarder_offset_mm": None}, "states retarder_axis_deg and retarder_offset_mm"),
    ({"retarder_offset_mm": "0"}, "retarder_offset_mm is not 0"),
    ({"extinction_ratio": None}, "has an extinction_ratio"),
])
def test_a_circular_polarizer_states_its_parts(change: dict, message: str) -> None:
    from os_optics._vendor.shared_contracts.design_catalog import PolarizationElementModelV1

    model = copy.deepcopy(circular()["model"])
    model.update(change)
    with pytest.raises(ValueError, match=message):
        PolarizationElementModelV1.model_validate({k: v for k, v in model.items() if v is not None})


# ------------------------------------------------------------------------------------------------ rotation parameter


def _rotatable(model: dict) -> dict:
    out = copy.deepcopy(model)
    out["model"]["position_parameter"] = "axis_rotation_deg"
    return out


def _with_angle(document: dict, instance_id: str, angle_deg: float) -> dict:
    for component in document["components"]:
        if component["instance_id"] == instance_id:
            component["selected_parameters"] = [{"semantic_key": "axis_rotation_deg", "value_kind": "DECIMAL", "value": f"{angle_deg:g}",
                                                 "unit_code": "deg"}]
    return document


@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 22.5, 30.0, 45.0])
def test_a_rotatable_half_wave_plate_before_a_polarizer_is_a_tunable_attenuator(theta_deg: float) -> None:
    """HWP (axis_rotation_deg selected per component) then a polarizer along the input: T = cos^2(2 theta) (Malus law of the
    turned polarization), the same as the plate placed turned by theta."""
    hwp = _rotatable(retarder(None, waves="0.5", transmittance="1"))
    analyzer = polarizer(None, transmittance="1", extinction="1000000")
    analyzer["model_id"] = "an"
    document = _with_angle(chain(source(), ("W", "waveplate", hwp, 0.0), ("A", "polarizer", analyzer, 0.0)), "W", theta_deg)
    run_ = run(document)
    c = math.cos(math.radians(2.0 * theta_deg))
    assert out_power(run_) == pytest.approx(POWER_W * (c * c + (1.0 - c * c) / 1e6), rel=1e-9, abs=1e-15)
    placed = run(chain(source(), ("W", "waveplate", retarder(None, waves="0.5", transmittance="1"), theta_deg), ("A", "polarizer", analyzer, 0.0)))
    assert out_power(run_) == pytest.approx(out_power(placed), rel=1e-9, abs=1e-15)


def test_the_rotation_parameter_turns_a_compound_members_axis() -> None:
    """A tunable splitter as a COMPOUND unit: the rotatable half-wave plate member and a polarizer member; the component's
    axis_rotation_deg turns only the plate."""
    from declared_helpers import compound_model, cylinder

    hwp = _rotatable(retarder(None, waves="0.5", transmittance="1"))
    analyzer = polarizer(None, transmittance="1", extinction="1000000")
    analyzer["model_id"] = "an"
    unit = compound_model("tunable", [("HWP", "wp", (0.0, 0.0, -5.0), (1.0, 0.0, 0.0, 0.0)), ("POL", "an", (0.0, 0.0, 5.0), (1.0, 0.0, 0.0, 0.0))],
                          cylinder("30", "-10", "10"))
    document = project([source(), declared_component("T", "other_optical", "tunable", position=(20.0, 0.0, 0.0), quaternion=_turned(0.0))],
                       [plane("OUT", (100.0, 0.0, 0.0))])
    document = with_models(document, [hwp, analyzer, unit])
    document = _with_angle(document, "T", 22.5)
    assert out_power(run(document)) == pytest.approx(POWER_W * (0.5 + 0.5 / 1e6), rel=1e-9)
    lowered = lower_project(document)
    (component,) = [c for c in lowered.request.project.components if c.instance_id == "T"]
    axes = {s.surface_id: (s.interaction.kind, getattr(s.interaction, "fast_axis_local", None) or getattr(s.interaction, "axis_local", None))
            for s in component.surfaces}
    assert axes["HWP-element"][1] == pytest.approx((math.cos(math.radians(22.5)), math.sin(math.radians(22.5)), 0.0), abs=1e-15)
    assert axes["POL-element"][1] == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)


def test_a_rotation_on_an_element_without_the_parameter_is_refused() -> None:
    from os_optics.local import LocalRunError

    analyzer = polarizer(None)
    document = _with_angle(chain(source(), ("A", "polarizer", analyzer, 0.0)), "A", 10.0)
    with pytest.raises(LocalRunError, match="axis_rotation_deg"):
        lower_project(document)
