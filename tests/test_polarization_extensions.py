"""Polarization extensions (engine part): band-resolved polarizer / retarder transmittance (CAT-02), elliptical and
circular source polarization, and a Fresnel-rhomb golden of the total-internal-reflection phase.

Independent references: the polarization ellipse written out from its time-domain field, plain-numpy Jones products, and
the Fresnel TIR amplitudes (Born & Wolf) evaluated here. Fixture data: transmittances and extinction ratios below are made
up for the test.
"""

from __future__ import annotations

import cmath
import copy
import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import (
    CircularAperture,
    ConstantIndexMaterial,
    EngineeringComponentSnapshot,
    JonesVector,
    PlaneSurface,
    ProjectSnapshot,
    RefractInteraction,
)
from os_optics.local import lower_project
from os_optics.local.lowering import source_jones
from os_optics._vendor.shared_contracts.optical_design import EllipticalPolarizationV1

from declared_helpers import BAND, declared_component, with_models
from helpers import one_arrival, run as run_request
from local_helpers import ALONG_X, POWER_W, catalog_document, plane, project, quat_multiply, run, source, turn_about_z

INV = 1.0 / math.sqrt(2.0)


def _value(value: str) -> dict:
    return {"value": value, "status": "SPECIFIED"}


def polarizer(bands: list[tuple[str, str, str, str | None]] | None, transmittance: str = "0.8", extinction: str = "1000") -> dict:
    model = {"model_kind": "POLARIZATION_ELEMENT", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
             "transmittance": _value(transmittance), "extinction_ratio": _value(extinction), "clear_radius_mm": "10", "edge_radius_mm": "12.7",
             "valid_band_nm": dict(BAND), "valid_aoi_deg": {"min_deg": "0", "max_deg": "10", "status": "SPECIFIED"},
             "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}
    if bands is not None:
        model["transmittance_bands"] = [{"min_nm": lo, "max_nm": hi, "transmittance": _value(t),
                                         **({"extinction_ratio": _value(er)} if er is not None else {}), "status": "SPECIFIED"}
                                        for lo, hi, t, er in bands]
    return {"model_id": "pol", "model": model}


def retarder(bands: list[tuple[str, str, str]] | None, waves: str = "0.25", transmittance: str = "0.5") -> dict:
    model = {"model_kind": "POLARIZATION_ELEMENT", "fidelity": "IDEAL_THIN_ELEMENT", "element": "RETARDER", "plane_z_mm": "0",
             "retardance_waves": waves, "design_wavelength_nm": "780", "retardance_dispersion": "CONSTANT",
             "transmittance": _value(transmittance), "clear_radius_mm": "10", "edge_radius_mm": "12.7", "valid_band_nm": dict(BAND),
             "valid_aoi_deg": {"min_deg": "0", "max_deg": "10", "status": "SPECIFIED"}, "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}
    if bands is not None:
        model["transmittance_bands"] = [{"min_nm": lo, "max_nm": hi, "transmittance": _value(t), "status": "SPECIFIED"} for lo, hi, t in bands]
    return {"model_id": "wp", "model": model}


def _turned(angle_deg: float) -> tuple[float, float, float, float]:
    """ALONG_X turned by ``angle_deg`` about the element's own local +Z (the beam axis): its local +X axis at that angle
    from the source's local +X toward local +Y."""

    half = math.radians(angle_deg) / 2.0
    return quat_multiply(ALONG_X, turn_about_z(math.cos(half), math.sin(half)))


def chain(src: dict, *elements: tuple[str, str, dict, float]) -> dict:
    """``src`` along +X, then each ``(instance_id, role, declared model, axis angle deg)`` at x = 0, 20, 40 ..., and OUT."""

    components = [src] + [declared_component(iid, role, model["model_id"], position=(20.0 * k, 0.0, 0.0), quaternion=_turned(angle))
                          for k, (iid, role, model, angle) in enumerate(elements)]
    document = project(components, [plane("OUT", (100.0, 0.0, 0.0))])
    return with_models(document, [model for _, _, model, _ in elements])


def out_power(run_) -> float:
    (obs,) = [o for o in run_.result.observations if o.observation_id == "OUT"]
    return sum(a.beam.power_W for a in obs.arrivals)


def elliptical_source(orientation_deg: float, ellipticity_deg: float, **kwargs) -> dict:
    laser = source(**kwargs)
    laser["source_model"]["polarization"] = {"kind": "ELLIPTICAL", "orientation_deg": orientation_deg, "ellipticity_deg": ellipticity_deg}
    return laser


# ---------------------------------------------------------------------------------------------------------- CAT-02

BANDS = [("400", "700", "0.8", "1000"), ("700", "1100", "0.9", "5000")]


@pytest.mark.parametrize("wavelength_nm, t_par, ratio", [(600.0, 0.8, 1000.0), (780.0, 0.9, 5000.0), (700.0, 0.8, 1000.0)],
                         ids=["first-band", "second-band", "shared-boundary-takes-the-smaller"])
def test_a_banded_polarizer_transmits_each_lines_own_band(wavelength_nm, t_par, ratio):
    along = run(chain(source(wavelength_nm=wavelength_nm), ("P", "polarizer", polarizer(BANDS), 0.0)))
    crossed = run(chain(source(wavelength_nm=wavelength_nm, polarization_deg=90.0), ("P", "polarizer", polarizer(BANDS), 0.0)))
    assert out_power(along) == pytest.approx(POWER_W * t_par, rel=1e-12)
    assert out_power(crossed) == pytest.approx(POWER_W * t_par / ratio, rel=1e-9)
    assert along.model_scope_issues == () and crossed.model_scope_issues == ()
    # the power budget books the band's rejection (1 - T / ER of the crossed light) as POLARIZER_REJECTION
    assert crossed.result.power_budget.lines[0].sinks["POLARIZER_REJECTION"] == pytest.approx(POWER_W * (1.0 - t_par / ratio), rel=1e-9)
    assert along.result.power_budget.lines[0].sinks["ELEMENT_TRANSMISSION_LOSS"] == pytest.approx(POWER_W * (1.0 - t_par), rel=1e-9)


def test_band_extinction_ratios_default_to_the_elements_when_the_bands_state_none():
    bands = [("400", "700", "0.8", None), ("700", "1100", "0.9", None)]
    crossed = run(chain(source(wavelength_nm=780.0, polarization_deg=90.0), ("P", "polarizer", polarizer(bands), 0.0)))
    assert out_power(crossed) == pytest.approx(POWER_W * 0.9 / 1000.0, rel=1e-9)


def test_a_banded_retarder_transmits_its_bands_value_and_a_multi_line_source_gets_each_bands():
    lines = source()
    lines["source_model"].pop("wavelength_nm")
    lines["source_model"].pop("power_w")
    lines["source_model"]["spectral_lines"] = [{"wavelength_nm": 520.0, "power_w": 0.001}, {"wavelength_nm": 900.0, "power_w": 0.002}]
    model = retarder([("400", "500", "0.5"), ("500", "1100", "0.94")])
    run_ = run(chain(lines, ("WP", "waveplate", model, 30.0)))
    (obs,) = [o for o in run_.result.observations if o.observation_id == "OUT"]
    by_line = {round(a.beam.wavelength_nm): a.beam.power_W for a in obs.arrivals}
    assert by_line == pytest.approx({520: 0.001 * 0.94, 900: 0.002 * 0.94}, rel=1e-12)
    blue = copy.deepcopy(lines)
    blue["source_model"]["spectral_lines"][0]["wavelength_nm"] = 450.0
    (obs,) = [o for o in run(chain(blue, ("WP", "waveplate", model, 30.0))).result.observations if o.observation_id == "OUT"]
    assert {round(a.beam.wavelength_nm): a.beam.power_W for a in obs.arrivals} == pytest.approx({450: 0.001 * 0.5, 900: 0.002 * 0.94}, rel=1e-12)


def test_without_bands_the_engine_input_is_unchanged():
    lowered = lower_project(chain(source(), ("P", "polarizer", polarizer(None), 0.0)))
    assert "transmittance_bands" not in lowered.request.model_dump_json()
    banded = lower_project(chain(source(), ("P", "polarizer", polarizer(BANDS), 0.0)))
    (element,) = [c for c in banded.request.project.components if c.instance_id == "P"]
    bands = element.surfaces[0].interaction.transmittance_bands
    assert [(b.min_nm, b.max_nm, b.transmittance_parallel, b.transmittance_perpendicular) for b in bands] == [
        (400.0, 700.0, 0.8, 0.8 / 1000.0), (700.0, 1100.0, 0.9, 0.9 / 5000.0)]


# ------------------------------------------------------------------------------------------- elliptical source


def _ellipse_field(jones: complex, jones_y: complex, phase: float) -> np.ndarray:
    """The real field (E_x, E_y) at time phase ``omega t`` of the Jones vector under exp(-i omega t)."""

    factor = cmath.exp(-1j * phase)
    return np.array([(jones * factor).real, (jones_y * factor).real])


def test_the_elliptical_jones_vector_has_the_stated_ellipse_and_handedness():
    """Independent of the formula: sample the time-domain field E(t) = Re[J exp(-i w t)]. Its largest excursion lies along
    orientation_deg, minor / major = |tan(chi)|, and for chi < 0 it turns clockwise for an observer looking toward the
    source (from +Z toward the source: X to the right, Y up), i.e. right-handed; chi = -45 is (1, -i) / sqrt 2. The sign is
    the Stokes s3 = 2 Im(conj(Ex) Ey) = sin(2 chi) the results report (M1: one convention for input and output)."""

    for theta, chi in ((0.0, 45.0), (0.0, -45.0), (30.0, 20.0), (-60.0, -10.0), (75.0, 33.0), (90.0, 45.0)):
        j = source_jones(EllipticalPolarizationV1(kind="ELLIPTICAL", orientation_deg=theta, ellipticity_deg=chi))
        assert j.norm == pytest.approx(1.0, abs=1e-15)
        jx, jy = complex(*j.ex), complex(*j.ey)
        samples = np.array([_ellipse_field(jx, jy, p) for p in np.linspace(0.0, 2 * math.pi, 3600, endpoint=False)])
        radii = np.hypot(samples[:, 0], samples[:, 1])
        major = samples[int(np.argmax(radii))]
        if abs(chi) < 45.0:
            assert math.degrees(math.atan2(major[1], major[0])) % 180.0 == pytest.approx(theta % 180.0, abs=0.06)
        assert radii.min() / radii.max() == pytest.approx(abs(math.tan(math.radians(chi))), abs=1e-5)
        turn = samples[0, 0] * samples[1, 1] - samples[0, 1] * samples[1, 0]  # (E(0) x E(dt)) . z: > 0 counterclockwise seen from +z
        assert (turn < 0) == (chi < 0)  # clockwise toward the source = right-handed
        s3 = 2.0 * (jx.conjugate() * jy).imag
        assert s3 == pytest.approx(math.sin(math.radians(2.0 * chi)), abs=1e-15)
    rhc = source_jones(EllipticalPolarizationV1(kind="ELLIPTICAL", orientation_deg=0.0, ellipticity_deg=-45.0))
    reference = JonesVector.right_circular()
    assert (*rhc.ex, *rhc.ey) == pytest.approx((*reference.ex, *reference.ey), abs=2e-16)


def _qwp(angle_deg: float) -> np.ndarray:
    """Quarter-wave plate, fast axis at ``angle_deg``: J = e^{-i pi/4} a a^T + e^{+i pi/4} b b^T (independent numpy)."""

    a = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    b = np.array([-a[1], a[0]])
    return cmath.exp(-1j * math.pi / 4) * np.outer(a, a) + cmath.exp(1j * math.pi / 4) * np.outer(b, b)


@pytest.mark.parametrize("chi", [-45.0, 45.0], ids=["right", "left"])
def test_circular_light_through_a_quarter_wave_plate_at_45_deg_is_linear(chi):
    """RHC (1, -i) / sqrt 2 through a QWP with its fast axis at +45 deg leaves linear along local Y (up to phase), LHC
    along local X (independent Jones product): a polarizer along that direction then passes all the light (the
    waveplate's 0.5 transmittance aside), a crossed one none."""

    j_in = np.array([INV, -1j * INV]) if chi < 0 else np.array([INV, 1j * INV])
    out = _qwp(45.0) @ j_in
    out = out * cmath.exp(-1j * cmath.phase(out[0] if abs(out[0]) > 1e-9 else out[1]))  # remove the common phase
    assert np.allclose(out.imag, 0.0, atol=1e-12)  # linear
    direction = math.degrees(math.atan2(out[1].real, out[0].real))
    expected_direction = 90.0 if chi < 0 else 0.0
    offset = (direction - expected_direction) % 180.0  # a line has no sign: -1e-15 deg and 180 deg are 0 deg
    assert min(offset, 180.0 - offset) == pytest.approx(0.0, abs=1e-9)
    wp = retarder(None, waves="0.25", transmittance="0.5")
    for offset, expected in ((0.0, 0.5), (90.0, 0.0)):
        run_ = run(chain(elliptical_source(0.0, chi), ("WP", "waveplate", wp, 45.0),
                         ("P", "polarizer", polarizer(None, transmittance="1", extinction="1000000000"), direction + offset)))
        assert out_power(run_) == pytest.approx(POWER_W * expected, abs=POWER_W * 1e-9)


@pytest.mark.parametrize("theta, chi", [(0.0, 45.0), (20.0, 30.0), (-50.0, -15.0)])
@pytest.mark.parametrize("beta", [0.0, 35.0, 90.0, 130.0])
def test_an_elliptical_beam_through_a_polarizer_follows_the_ellipse(theta, chi, beta):
    """Power through an ideal polarizer at beta: P [cos^2 chi cos^2(theta - beta) + sin^2 chi sin^2(theta - beta)] (the
    ellipse's projection); circular light gives half the power at every angle."""

    run_ = run(chain(elliptical_source(theta, chi), ("P", "polarizer", polarizer(None, transmittance="1", extinction="1000000000"), beta)))
    c, s = math.cos(math.radians(chi)), math.sin(math.radians(chi))
    d = math.radians(theta - beta)
    expected = c * c * math.cos(d) ** 2 + s * s * math.sin(d) ** 2
    assert out_power(run_) == pytest.approx(POWER_W * (expected + (1.0 - expected) * 1e-9), abs=1e-15)
    if abs(chi) == 45.0:
        assert out_power(run_) == pytest.approx(POWER_W * 0.5, rel=1e-8)


def test_a_linear_ellipse_is_the_linear_source_bit_for_bit():
    linear = lower_project(chain(source(polarization_deg=30.0), ("P", "polarizer", polarizer(None), 0.0)))
    flat = lower_project(chain(elliptical_source(30.0, 0.0), ("P", "polarizer", polarizer(None), 0.0)))
    (a,) = [c for c in linear.request.project.components if c.source is not None]
    (b,) = [c for c in flat.request.project.components if c.source is not None]
    assert a.source.polarization == b.source.polarization


# --------------------------------------------------------------------------------------- Fresnel rhomb (TIR) golden


def _bk7_index(wavelength_nm: float) -> float:
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _tir_amplitudes(n: float, theta: float) -> tuple[complex, complex]:
    """Born & Wolf (r_s, r_p) glass (n) -> air beyond the critical angle, cos(theta_t) = +i sqrt(n^2 sin^2 - 1)."""

    cos_t = 1j * math.sqrt(n * n * math.sin(theta) ** 2 - 1.0)
    return (n * math.cos(theta) - cos_t) / (n * math.cos(theta) + cos_t), (math.cos(theta) - n * cos_t) / (math.cos(theta) + n * cos_t)


def _rhomb_angle(n: float) -> float:
    """The larger angle at which one TIR gives 45 deg of retardance: tan(delta / 2) = cos(theta) sqrt(sin^2 theta - 1 / n^2)
    / sin^2 theta = tan(22.5 deg), by bisection between the maximum of the retardance and grazing incidence."""

    def f(theta: float) -> float:
        return math.cos(theta) * math.sqrt(math.sin(theta) ** 2 - 1.0 / n ** 2) / math.sin(theta) ** 2 - math.tan(math.pi / 8)

    peak = math.acos(math.sqrt((n * n - 1.0) / (n * n + 1.0)))  # where the single-reflection retardance is largest
    lo, hi = peak, math.radians(89.9)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2.0


def _rhomb(theta: float, n: float, wavelength_nm: float) -> ProjectSnapshot:
    """A Fresnel rhomb in the XY plane: entrance face at x = 0 (normal incidence), TIR face A at x = 40 with outward normal
    (cos theta, sin theta, 0), the parallel face B a distance L along the reflected beam, exit face beyond; the beam leaves
    along +X, displaced, with linear 45 deg (s = Z, p in the XY plane) input."""

    length = 30.0
    n_a = np.array([math.cos(theta), math.sin(theta), 0.0])
    p_a = np.array([40.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0]) - 2.0 * math.cos(theta) * n_a
    p_b = p_a + length * d1
    x_exit = p_b[0] + 0.5 * (p_a[0] + 1.3 * length - p_b[0])
    glass = ConstantIndexMaterial(material_id="rhomb-glass", valid_wavelength_nm=(400.0, 1000.0), refractive_index=n,
                                  absorption_coefficient_per_mm=0.0)

    def face(surface_id, origin, normal, radius=None):
        return PlaneSurface(surface_id=surface_id, origin_mm=tuple(float(v) for v in origin), normal=tuple(float(v) for v in normal),
                            aperture=CircularAperture(radius_mm=radius) if radius else None,
                            medium_positive="air" if surface_id != "in" else "rhomb-glass",
                            medium_negative="rhomb-glass" if surface_id != "in" else "air", interaction=RefractInteraction())

    rhomb = EngineeringComponentSnapshot(
        instance_id="RH", pose=B.Pose3D(position_mm=(0.0, 0.0, 0.0), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)),
        surfaces=(face("in", (0, 0, 0), (1, 0, 0)), face("A", p_a, n_a, 0.4 * length), face("B", p_b, -n_a, 0.4 * length),
                  face("out", (x_exit, 0, 0), (1, 0, 0))))
    return ProjectSnapshot(
        materials=(glass,),
        components=(B.laser("L", wavelength_nm=wavelength_nm, waist_radius_mm=0.3, position=(-50.0, 0.0, 0.0), direction=(1, 0, 0),
                            polarization=JonesVector.linear(45.0)), rhomb),
        observations=(B.observation_plane("before", position=(-10.0, 0.0, 0.0), normal=(1, 0, 0), radius_mm=5.0),
                      B.observation_plane("after", position=(x_exit + 20.0, float(p_b[1]), 0.0), normal=(1, 0, 0), radius_mm=5.0)))


@pytest.mark.golden
def test_golden_fresnel_rhomb_two_tir_reflections_give_a_quarter_wave():
    """Two TIRs in BK7 at the Fresnel-rhomb angle: the s / p phase difference grows by exactly pi / 2 (independent
    closed form for the angle, Born & Wolf amplitudes for the phases), |r| = 1, and 45 deg linear light leaves circular."""

    wavelength = 632.8
    n = _bk7_index(wavelength)
    theta = _rhomb_angle(n)
    r_s, r_p = _tir_amplitudes(n, theta)
    assert abs(r_s) == pytest.approx(1.0, abs=1e-15) and abs(r_p) == pytest.approx(1.0, abs=1e-15)
    per_reflection = cmath.phase(r_p / r_s)
    assert abs(per_reflection) == pytest.approx(math.pi / 4, abs=1e-12)  # the closed form and the amplitudes agree
    result = run_request(_rhomb(theta, n, wavelength))
    z, minus_y = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])  # s and p = d x s for d = +X, before and after

    def components(arrival) -> tuple[complex, complex]:
        jx, jy = complex(*arrival.beam.polarization.ex), complex(*arrival.beam.polarization.ey)
        e = jx * np.array(arrival.beam.basis.e1) + jy * np.array(arrival.beam.basis.e2)
        return complex(e @ z), complex(e @ minus_y)

    before, after = one_arrival(result, "before"), one_arrival(result, "after")
    s0, p0 = components(before)
    s1, p1 = components(after)
    retardance = cmath.phase((p1 / s1) / (p0 / s0))
    assert retardance == pytest.approx(cmath.phase((r_p / r_s) ** 2), abs=1e-9)
    assert abs(retardance) == pytest.approx(math.pi / 2, abs=1e-9)
    assert after.beam.power_W == pytest.approx(before.beam.power_W * (1 - ((n - 1) / (n + 1)) ** 2) ** 2, rel=1e-9)  # two normal faces
    assert abs(s1) == pytest.approx(abs(p1), rel=1e-9)  # circular
    assert [s.end_event for p in result.paths for s in p.segments if s.end_surface_id in ("RH/A", "RH/B")] == ["tir", "tir"]


# ------------------------------------------------------------------------------------------ cross-runtime case


def coating_and_polarization_project() -> dict:
    """One Project with every feature of this module and of the coating tests: an elliptical source, a tilted window with
    an angle-resolved coating and wavelength-tabulated phases, a banded half-wave plate and a banded polarizer (the
    Pyodide parity harness runs it in the browser runtime too)."""

    from test_coating_tables_and_phases import angular, coating, window_project

    phase = {"wavelength_nm": ["400", "1100"], "transmission_s_rad": ["0", "0.7"], "transmission_p_rad": ["1.4", "0"], "status": "DECLARED"}
    document = window_project(coating(angular_spectrum=angular(), phase=phase), polarization_deg=0.0)
    document["components"][0] = elliptical_source(20.0, 30.0)
    plate = retarder([("400", "700", "0.9"), ("700", "1100", "0.95")], waves="0.5", transmittance="0.9")
    pol = polarizer(BANDS)
    for k, (iid, role, model, angle) in enumerate((("WP", "waveplate", plate, 45.0), ("P", "polarizer", pol, 10.0))):
        document["components"].append(declared_component(iid, role, model["model_id"], position=(40.0 + 20.0 * k, -1.5, 0.0),
                                                         quaternion=_turned(angle)))
        document["optical_models"].append(model)
    document["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "OUT", "minimum": 0.0, "unit": "W"}]
    return document


def test_the_cross_runtime_case_runs_in_scope():
    run_ = run(coating_and_polarization_project())
    assert run_.model_scope_issues == () and run_.status.value == "PASS"
    assert 0.0 < out_power(run_) < POWER_W


@pytest.mark.parametrize("polarization", [{"kind": "RANDOM"}, {"kind": "CIRCULAR", "handedness": "RIGHT"}])
def test_a_source_polarization_of_another_kind_is_refused_naming_the_kinds_there_are(polarization):
    """A polarization of a kind the contract does not have used to be refused with the LINEAR fields only ("kind:
    Input should be 'LINEAR'; angle_deg: required field missing"); the refusal names the kind and the kinds there are (UNPOLARIZED is one: test_partial_polarization.py)."""
    from os_optics.local import json_api

    document = project([source()], [plane("OUT", (100.0, 0.0, 0.0))])
    document["components"][0]["source_model"]["polarization"] = polarization
    reply = json_api.check_project(document, catalog=catalog_document())
    (problem,) = reply["error"]["problems"]
    assert problem["code"] == "PROJECT_INVALID"
    assert problem["message"] == (f"components[0].source_model.polarization.kind: {polarization['kind']!r} is not a source "
                                  "polarization kind: the kinds are LINEAR (angle_deg), ELLIPTICAL (orientation_deg, "
                                  "ellipticity_deg), PARTIAL (polarized, degree_of_polarization) or UNPOLARIZED")
