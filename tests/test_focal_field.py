"""Vector focal field of an ideal aplanatic objective (``os_optics.results.focal_field``, ``os_optics.diffraction``), against references
derived here or recorded independently, with tolerances set before the first run:

* the independent 852 nm Richards-Wolf table (x-polarized Gaussian pupil, w = 10 mm, F = 20 mm; 1/e^2 radii of |E|^2 normalized on axis), 1e-4;
* low NA, uniform pupil: the Airy pattern -- FWHM ``0.5145 lambda / NA``, encircled energy ``1 - J0^2 - J1^2``, peak ``P pi NA^2 / lambda^2`` -- within
  ``O(NA^2)`` (3e-3 at NA 0.05);
* low NA, unclipped Gaussian pupil: the Gaussian waist ``lambda F / (pi w)`` and its peak ``2 P / (pi w0^2)`` (1e-3);
* polarization symmetries: x and y input exchange the x and y widths (1e-9), circular input gives equal widths (1e-9), E_y is odd in x and y and
  E_z odd in x for x input;
* the longitudinal share of the electric energy against a one-dimensional quadrature of the sphere written here (1e-9);
* the pupil power of a clipped Gaussian ``P (1 - exp(-2 a^2 / w^2))`` (1e-12) and the focal-plane flux against it (energy balance, 1e-3);
* an off-axis input focuses at ``F l`` with the on-axis spot's widths (1e-6); incoherent foci carry their beams' powers (1e-3);
* a window: transmission ``(1 - R)^2`` at low NA (1e-3), the focus between the paraxial ``t (1 - 1 / n)`` and the marginal-ray shift, and the Strehl ratio
  against the Marechal estimate of the plate's residual phase (0.01); a Zernike spherical term against Marechal (0.01).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import integrate, optimize, special

from os_optics.cli import main as cli_main
from os_optics.contracts.analysis import AnalysisResultV1
from os_optics.errors.status import QuantityState
from os_optics.local import json_api
from os_optics.results import FocalFieldError, focal_field, load_run

from local_helpers import plane, project, run, source

#: Independent ideal Richards-Wolf reference at 852 nm, F = 20 mm, Gaussian pupil radius 10 mm, x-polarized: (NA, x radius um, y radius um).
REFERENCE_852 = ((0.15, 2.369095, 2.354114), (0.30, 1.236011, 1.205938), (0.40, 0.968638, 0.928573), (0.50, 0.819312, 0.769604),
                 (0.65, 0.698480, 0.635844))


def request(na: float = 0.65, *, wavelength: float = 852.0, focal: float = 20.0, illumination: dict | None = None, polarization: dict | None = None,
            index: float = 1.0, window: dict | None = None, phase: dict | None = None, sampling: dict | None = None, output: dict | None = None) -> dict:
    objective = {"model": "IDEAL_APLANATIC", "numerical_aperture": na, "focal_length_mm": focal, "focal_medium_index": index}
    if window is not None:
        objective["window"] = window
    doc = {"wavelength_nm": wavelength, "objective": objective,
           "illumination": illumination or {"kind": "GAUSSIAN", "power_w": 1.0, "radius_mm": 10.0},
           "polarization": polarization or {"kind": "LINEAR", "angle_deg": 0.0}}
    for key, value in (("pupil_phase", phase), ("sampling", sampling), ("output", output)):
        if value is not None:
            doc[key] = value
    return doc


def value(doc: AnalysisResultV1, quantity_id: str) -> float:
    q = doc.quantity(quantity_id)
    assert q.state is QuantityState.KNOWN, (quantity_id, q.state, q.reason, q.message)
    return float(q.value)


@pytest.fixture(scope="module")
def na065() -> AnalysisResultV1:
    return focal_field(request(0.65))


@pytest.mark.parametrize(("na", "x_um", "y_um"), REFERENCE_852)
def test_the_independent_852_nm_table(na, x_um, y_um):
    doc = focal_field(request(na))
    assert value(doc, "radius_1e2_x") == pytest.approx(x_um, rel=1e-4)
    assert value(doc, "radius_1e2_y") == pytest.approx(y_um, rel=1e-4)


def test_every_value_converged_and_was_checked_by_the_independent_method(na065):
    doc = na065
    assert doc.analysis == "FOCAL_FIELD" and doc.execution.complete
    assert doc.settings["method"] == "bessel" and doc.settings["reference_method"] == "direct"
    for q in doc.quantities:
        if q.state is QuantityState.KNOWN and q.convergence is not None and q.convergence.method.value == "REFINEMENT":
            assert q.convergence.converged, q.quantity_id
    for row in doc.arrays["reference"]:
        assert abs(row["relative_difference"]) < 1e-9, row
    energy = doc.arrays["energy_balance"]
    assert max(energy["parseval_relative_residuals"]) < 1e-12
    assert abs(value(doc, "energy_relative_residual")) < 1e-3
    # a clipped Gaussian: the pupil passes P (1 - exp(-2 a^2 / w^2)), a = F NA
    a = 20.0 * 0.65
    assert value(doc, "pupil_power_W") == pytest.approx(1.0 - math.exp(-2.0 * a * a / 100.0), rel=2e-9)
    assert value(doc, "clipped_power_W") == pytest.approx(math.exp(-2.0 * a * a / 100.0), rel=1e-6)
    # the Gaussian description does not hold at this aperture (far-field angle w / F = 0.5 rad, pupil radius 1.3 w): the waist is
    # the formula's number lambda F / (pi w) -- by hand below --, an estimate that states the two checks it fails and decides
    # nothing; the widths stay
    waist = doc.quantity("gaussian_waist_radius")
    assert waist.state is QuantityState.OUT_OF_MODEL_SCOPE and waist.reason == "NON_GAUSSIAN_FOCUS"
    assert waist.value == pytest.approx(0.852 * 20.0 / (math.pi * 10.0), rel=1e-9)  # um: 852 nm, F = 20 mm, w = 10 mm
    entry = waist.value_basis
    assert entry.kind.value == "ESTIMATE" and not entry.assumed and not entry.omitted
    assert [(c.parameter, c.within, c.code) for c in entry.validity] == [
        ("far_field_half_angle_rad", False, "NON_GAUSSIAN_FOCUS"), ("pupil_radius_over_beam_radius", False, "NON_GAUSSIAN_FOCUS")]
    assert entry.validity[0].observed == pytest.approx(0.5, rel=1e-9) and entry.validity[1].observed == pytest.approx(1.3, rel=1e-9)
    assert entry.validity[1].minimum == 2.0 and not waist.decides(0.0) and not waist.decides(1e9)
    assert value(doc, "strehl_ratio") == 1.0


def _airy_radius_for(fraction: float) -> float:
    """``v`` with ``1 - J0(v)^2 - J1(v)^2 = fraction``."""
    return optimize.brentq(lambda v: 1.0 - special.j0(v) ** 2 - special.j1(v) ** 2 - fraction, 0.1, 3.8)


def test_low_na_uniform_pupil_gives_the_airy_pattern():
    na, wavelength = 0.05, 1000.0
    doc = focal_field(request(na, wavelength=wavelength, focal=50.0, illumination={"kind": "UNIFORM", "power_w": 2.0}))
    lam_um = wavelength * 1e-3
    assert value(doc, "fwhm_x") == pytest.approx(0.5145 * lam_um / na, rel=3e-3)
    assert value(doc, "fwhm_y") == pytest.approx(0.5145 * lam_um / na, rel=3e-3)
    k_na = 2.0 * math.pi * na / lam_um
    assert value(doc, "encircled_radius_50") == pytest.approx(_airy_radius_for(0.5) / k_na, rel=3e-3)
    assert value(doc, "encircled_radius_80") == pytest.approx(_airy_radius_for(0.8) / k_na, rel=3e-3)
    assert value(doc, "peak_electric_irradiance") == pytest.approx(2.0 * math.pi * na * na / (lam_um * lam_um), rel=3e-3)
    assert value(doc, "pupil_power_W") == pytest.approx(2.0, rel=2e-9)


def test_low_na_unclipped_gaussian_focus_is_the_gaussian_waist():
    w, focal, wavelength = 1.0, 100.0, 1064.0
    doc = focal_field(request(0.05, wavelength=wavelength, focal=focal, illumination={"kind": "GAUSSIAN", "power_w": 0.5, "radius_mm": w}))
    w0_um = wavelength * 1e-6 * focal / (math.pi * w) * 1e3
    assert value(doc, "gaussian_waist_radius") == pytest.approx(w0_um, rel=2e-9)
    assert value(doc, "radius_1e2_x") == pytest.approx(w0_um, rel=1e-3)
    assert value(doc, "radius_1e2_y") == pytest.approx(w0_um, rel=1e-3)
    assert value(doc, "peak_electric_irradiance") == pytest.approx(2.0 * 0.5 / (math.pi * w0_um * w0_um), rel=1e-3)


def test_polarization_symmetries(na065):
    y_input = focal_field(request(0.65, polarization={"kind": "LINEAR", "angle_deg": 90.0}))
    for a, b in (("fwhm_x", "fwhm_y"), ("radius_1e2_x", "radius_1e2_y")):
        assert value(y_input, a) == pytest.approx(value(na065, b), rel=3e-9)
    assert value(y_input, "fwhm_z") == pytest.approx(value(na065, "fwhm_z"), rel=3e-9)
    circular = focal_field(request(0.65, polarization={"kind": "CIRCULAR", "handedness": "RIGHT"}))
    assert value(circular, "fwhm_x") == pytest.approx(value(circular, "fwhm_y"), rel=3e-9)
    assert value(circular, "radius_1e2_x") == pytest.approx(value(circular, "radius_1e2_y"), rel=3e-9)
    unpolarized = focal_field(request(0.65, polarization={"kind": "UNPOLARIZED"}))
    assert value(unpolarized, "fwhm_x") == pytest.approx(value(unpolarized, "fwhm_y"), rel=3e-9)
    assert value(unpolarized, "fwhm_x") == pytest.approx(value(circular, "fwhm_x"), rel=3e-9)
    # field components of x input: E_y odd in x and in y, E_z odd in x and even in y
    doc = focal_field(request(0.65, output={"include_components": True, "samples": 21}))
    plane_ = doc.arrays["maps"]["planes"][0]["field_sqrt_W_per_um2"]
    ey = np.array(plane_["Ey"]["re"]) + 1j * np.array(plane_["Ey"]["im"])
    ez = np.array(plane_["Ez"]["re"]) + 1j * np.array(plane_["Ez"]["im"])
    ex = np.array(plane_["Ex"]["re"]) + 1j * np.array(plane_["Ex"]["im"])
    scale = np.abs(ex).max()
    assert np.abs(ey + ey[:, ::-1]).max() < 1e-6 * scale and np.abs(ey + ey[::-1, :]).max() < 1e-6 * scale
    assert np.abs(ez + ez[:, ::-1]).max() < 1e-6 * scale and np.abs(ez - ez[::-1, :]).max() < 1e-6 * scale
    assert np.abs(ez).max() > 0.1 * scale  # the longitudinal field is substantial at NA 0.65


def test_the_longitudinal_share_against_a_quadrature_of_the_sphere(na065):
    """For x input, a_z = A t sin(theta) cos(phi) and |a|^2 = A^2 cos(theta) (A the pupil amplitude): the plane integrals of |E_z|^2 and |E|^2 are
    f'^2 Int |a_z|^2 / cos(theta) dOmega and f'^2 Int |a|^2 / cos(theta) dOmega."""

    focal, w, na = 20.0, 10.0, 0.65

    def amp2(theta: float) -> float:
        h = focal * math.sin(theta)
        return math.exp(-2.0 * h * h / (w * w))

    top = math.asin(na)
    axial = math.pi * integrate.quad(lambda t: amp2(t) * math.sin(t) ** 3, 0.0, top, epsabs=0, epsrel=1e-13)[0]
    total = 2.0 * math.pi * integrate.quad(lambda t: amp2(t) * math.sin(t), 0.0, top, epsabs=0, epsrel=1e-13)[0]
    assert value(na065, "longitudinal_energy_fraction") == pytest.approx(axial / total, rel=3e-9)
    grid = na065.arrays["energy_balance"]["longitudinal_energy_fraction_grid"][1]
    assert grid == pytest.approx(axial / total, rel=2e-3)


def test_an_off_axis_input_focuses_at_f_l_with_the_on_axis_shape(na065):
    tilted = focal_field(request(0.65, illumination={"kind": "GAUSSIAN", "power_w": 1.0, "radius_mm": 10.0, "direction_cosines": [0.005, -0.002]},
                                 sampling={"theta_nodes": 48, "phi_nodes": 48, "grid_cells": 96}))
    assert tilted.settings["method"] == "grid" and tilted.settings["reference_method"] == "direct"
    assert value(tilted, "peak_position_x") == pytest.approx(100.0, abs=2e-4)  # reported to 1e-4 lambda / NA
    assert value(tilted, "peak_position_y") == pytest.approx(-40.0, abs=2e-4)
    for name in ("fwhm_x", "fwhm_y", "fwhm_z", "radius_1e2_x", "radius_1e2_y", "peak_electric_irradiance", "encircled_radius_50"):
        assert value(tilted, name) == pytest.approx(value(na065, name), rel=2e-3), name
    for row in tilted.arrays["reference"]:
        assert abs(row["relative_difference"]) < 2e-3, row
    for q in tilted.quantities:
        if q.convergence is not None and q.convergence.converged is not None:
            assert q.convergence.converged, q.quantity_id


def test_incoherent_foci_carry_their_beams_powers():
    beams = [{"label": "a", "power_w": 1.0, "radius_mm": [2.0, 2.0], "direction_cosines": [0.0005, 0.0]},
             {"label": "b", "power_w": 2.0, "radius_mm": [2.0, 2.0], "direction_cosines": [-0.0005, 0.0]}]
    doc = focal_field(request(0.2, illumination={"kind": "BEAMS", "beams": beams, "composition": "INCOHERENT"},
                              sampling={"theta_nodes": 32, "phi_nodes": 32, "grid_cells": 64}, output={"samples": 21}))
    assert value(doc, "a.peak_position_x") == pytest.approx(10.0, abs=5e-4)
    assert value(doc, "b.peak_position_x") == pytest.approx(-10.0, abs=5e-4)
    assert value(doc, "b.power_within_disc_W") / value(doc, "a.power_within_disc_W") == pytest.approx(2.0, rel=1e-3)
    assert value(doc, "b.peak_electric_irradiance") / value(doc, "a.peak_electric_irradiance") == pytest.approx(2.0, rel=1e-3)
    assert doc.quantity("a.strehl_ratio").state is QuantityState.NOT_APPLICABLE


def test_a_window_shifts_the_focus_and_transmits_its_fresnel_share():
    t, n_w = 1.0, 1.5
    doc = focal_field(request(0.1, illumination={"kind": "UNIFORM", "power_w": 1.0}, window={"thickness_mm": t, "refractive_index": n_w}))
    r = ((n_w - 1.0) / (n_w + 1.0)) ** 2
    assert value(doc, "transmitted_power_W") / value(doc, "pupil_power_W") == pytest.approx((1.0 - r) ** 2, rel=1e-3)
    paraxial = t * (1.0 - 1.0 / n_w) * 1e3
    theta = math.asin(0.1)
    marginal = t * (1.0 - math.tan(math.asin(math.sin(theta) / n_w)) / math.tan(theta)) * 1e3
    assert paraxial < value(doc, "peak_position_z") < marginal
    assert value(doc, "strehl_ratio") > 0.99


def _marechal(phase, weight, top: float) -> float:
    """``exp(-sigma^2)`` of the phase over the pupil (weights ``weight``) after removing its best fit ``a + b cos(theta)`` (piston and focus)."""

    theta, w = np.polynomial.legendre.leggauss(400)
    theta = 0.5 * top * (theta + 1.0)
    w = 0.5 * top * w * weight(theta)
    basis = np.stack((np.ones_like(theta), np.cos(theta)))
    phi = phase(theta)
    gram = (basis * w) @ basis.T
    coefficients = np.linalg.solve(gram, (basis * w) @ phi)
    residual = phi - coefficients @ basis
    sigma2 = float((w * residual * residual).sum() / w.sum())
    return math.exp(-sigma2)


def test_the_window_strehl_ratio_against_marechal():
    t, n_w, na, wavelength = 1.0, 1.5, 0.3, 852.0
    doc = focal_field(request(na, wavelength=wavelength, illumination={"kind": "UNIFORM", "power_w": 1.0},
                              window={"thickness_mm": t, "refractive_index": n_w, "surfaces": "IDEAL_ANTIREFLECTION"}))
    k0 = 2.0 * math.pi / (wavelength * 1e-6)

    def plate(theta):
        return k0 * t * (np.sqrt(n_w * n_w - np.sin(theta) ** 2) - np.cos(theta))

    # uniform pupil: amplitude sqrt(cos) on the sphere, dOmega = sin dtheta: weight cos(theta) sin(theta) (1 + cos(theta)) / 2 for the axial field
    estimate = _marechal(plate, lambda th: np.sqrt(np.cos(th)) * np.sin(th) * (1.0 + np.cos(th)), math.asin(na))
    strehl = value(doc, "strehl_ratio")
    assert 0.5 < strehl < 0.99
    assert strehl == pytest.approx(estimate, abs=0.01)


def test_a_zernike_spherical_term_against_marechal():
    waves = 0.04
    doc = focal_field(request(0.1, illumination={"kind": "UNIFORM", "power_w": 1.0}, phase={"kind": "ZERNIKE", "terms": [{"n": 4, "m": 0, "coefficient_waves": waves}]}))
    assert value(doc, "strehl_ratio") == pytest.approx(math.exp(-(2.0 * math.pi * waves) ** 2), abs=0.01)
    assert doc.settings["method"] == "bessel"
    astig = focal_field(request(0.1, illumination={"kind": "UNIFORM", "power_w": 1.0}, phase={"kind": "ZERNIKE", "terms": [{"n": 2, "m": 2, "coefficient_waves": waves}]},
                                sampling={"theta_nodes": 32, "phi_nodes": 32, "grid_cells": 64}))
    assert astig.settings["method"] == "grid"
    assert value(astig, "strehl_ratio") == pytest.approx(math.exp(-(2.0 * math.pi * waves) ** 2), abs=0.01)


def test_an_annular_pupil_lengthens_the_focus():
    full = focal_field(request(0.5, illumination={"kind": "UNIFORM", "power_w": 1.0}, polarization={"kind": "CIRCULAR", "handedness": "LEFT"}))
    ring = focal_field(request(0.5, illumination={"kind": "ANNULAR", "power_w": 1.0, "inner_radius_fraction": 0.5},
                               polarization={"kind": "CIRCULAR", "handedness": "LEFT"}))
    assert value(ring, "pupil_power_W") == pytest.approx(1.0, rel=2e-9)
    assert value(ring, "fwhm_z") > 1.2 * value(full, "fwhm_z")
    assert value(ring, "fwhm_x") < value(full, "fwhm_x")


def test_outside_the_debye_domain_the_focal_quantities_are_out_of_scope_and_the_pupil_stays():
    doc = focal_field(request(0.1, wavelength=1000.0, focal=0.05, illumination={"kind": "UNIFORM", "power_w": 1.0}))
    # the analysis ran to its end (the pupil ledger); the focal quantities say they are outside the model
    assert doc.execution.complete and doc.execution.outcome.value == "COMPLETED" and doc.execution.reason == ""
    assert doc.quantity("peak_electric_irradiance").state is QuantityState.OUT_OF_MODEL_SCOPE
    assert doc.quantity("peak_electric_irradiance").reason == "DEBYE_APPROXIMATION_INVALID"
    assert value(doc, "pupil_power_W") == pytest.approx(1.0, rel=2e-9)
    assert not doc.models[0].validity[0].within


@pytest.mark.parametrize("change", [
    {"polarization": None}, {"objective": {"model": "IDEAL_APLANATIC", "numerical_aperture": 1.2, "focal_length_mm": 2.0}},
    {"illumination": {"kind": "GAUSSIAN", "power_w": -1.0, "radius_mm": 1.0}},
    {"pupil_phase": {"kind": "ZERNIKE", "terms": [{"n": 3, "m": 2, "coefficient_waves": 0.1}]}},
])
def test_an_invalid_request_is_refused(change):
    body = {**request(), **change}
    if body.get("polarization") is None:
        body.pop("polarization")
    with pytest.raises(FocalFieldError) as exc:
        focal_field(body)
    assert exc.value.code == "FOCAL_FIELD_INVALID"


def test_beams_without_a_handoff_or_a_phase_are_refused():
    with pytest.raises(FocalFieldError) as exc:
        focal_field({**request(), "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}})
    assert exc.value.code == "FOCAL_FIELD_INVALID"  # a handoff illumination takes no stated polarization
    body = request()
    body.pop("polarization")
    body["illumination"] = {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}
    with pytest.raises(FocalFieldError) as exc:
        focal_field(body)
    assert exc.value.code == "HANDOFF_REQUIRED"
    # several coherent beams add with their phases: one without a phase leaves the pattern undetermined
    beams = [{"label": "a", "power_w": 1.0, "radius_mm": [3.0, 3.0], "phase_rad": 0.0}, {"label": "b", "power_w": 1.0, "radius_mm": [3.0, 3.0]}]
    with pytest.raises(FocalFieldError) as exc:
        focal_field(request(illumination={"kind": "BEAMS", "beams": beams, "composition": "COHERENT"}))
    assert exc.value.code == "PHASE_NOT_DEFINED" and "INCOHERENT" in exc.value.message and "b" in exc.value.message


def test_one_coherent_beam_without_a_phase_is_the_same_field_with_an_undetermined_constant_phase():
    """One beam is the same field under both compositions, up to a constant phase no |E|^2 depends on: without a stated phase
    it was refused (PHASE_NOT_DEFINED); it is computed with the phase taken as 0, and the document says so."""

    sampling = {"theta_nodes": 48, "phi_nodes": 48, "grid_cells": 64}
    beam = {"label": "a", "power_w": 1.0, "radius_mm": [3.0, 3.0]}
    coherent = focal_field(request(0.5, illumination={"kind": "BEAMS", "beams": [beam], "composition": "COHERENT"}, sampling=sampling))
    stated = focal_field(request(0.5, illumination={"kind": "BEAMS", "beams": [{**beam, "phase_rad": 0.0}], "composition": "COHERENT"}, sampling=sampling))
    assert coherent.arrays["phase_reference"] == "UNDETERMINED_CONSTANT" and "phase_reference" not in stated.arrays
    assert coherent.execution.complete
    assert [q.model_dump() for q in coherent.quantities] == [q.model_dump() for q in stated.quantities]
    assert coherent.arrays["maps"] == stated.arrays["maps"]


# ------------------------------------------------------------------------------------------------- a run's beams


def _pupil_plane_run(m2: float = 1.0) -> dict:
    laser = source(position=(-100.0, 0.0, 0.0), wavelength_nm=852.0, power_w=2e-3, waist_mm=4.0)
    laser["source_model"]["m2"] = m2
    return project([laser], [plane("PUPIL", (0.0, 0.0, 0.0), radius_mm=20.0)])


def _handoff_request() -> dict:
    body = request(0.5, sampling={"theta_nodes": 48, "phi_nodes": 48, "grid_cells": 64})
    body.pop("polarization")
    body["illumination"] = {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}
    return body


def test_a_runs_beam_lights_the_pupil_with_its_power_and_jones_vector():
    results = load_run(run(_pupil_plane_run()))
    beam = results.handoff("PUPIL")["beams"][0]
    doc = results.focal_field("PUPIL", _handoff_request())
    assert doc.identity.physical.engine_input_hash == results.identity()["engine_input_hash"]
    assert doc.identity.physical.run_result_hash == results.identity()["result_hash"]
    assert "handoff_sha256" in doc.settings
    label = beam["path_id"]
    stated = focal_field(request(0.5, illumination={"kind": "GAUSSIAN", "power_w": beam["power_W"], "radius_mm": beam["gaussian"]["beam_radius_x_mm"]},
                                 polarization={"kind": "LINEAR", "angle_deg": 0.0}))
    assert value(doc, "peak_electric_irradiance") == pytest.approx(value(stated, "peak_electric_irradiance"), rel=1e-3)
    assert doc.quantities[0].quantity_id == "incident_power_W" and value(doc, "incident_power_W") == pytest.approx(2e-3, rel=3e-9)
    assert label
    # the same through the JSON API
    reply = json_api.focal_field(_handoff_request(), results.handoff("PUPIL"))
    assert reply["ok"] is True and reply["analysis"] == "FOCAL_FIELD"
    assert reply["quantities"] == json.loads(doc.model_dump_json(by_alias=True))["quantities"]


def test_a_beam_that_is_not_tem00_builds_no_focal_field():
    results = load_run(run(_pupil_plane_run(m2=1.5)))
    with pytest.raises(FocalFieldError) as exc:
        results.focal_field("PUPIL", _handoff_request())
    assert exc.value.code == "BEAM_NOT_TEM00"
    reply = json_api.focal_field(_handoff_request(), results.handoff("PUPIL"))
    assert reply["ok"] is False and reply["error"]["code"] == "BEAM_NOT_TEM00"


def test_the_gaussian_limit_of_the_run_holds_for_a_field_built_on_it():
    """A stated Gaussian pupil of 6.4 mm before an NA 0.65, 20 mm objective has a far-field half angle of 0.32 rad: outside the
    Engine's default Gaussian limit (0.3 rad), so no Gaussian waist is stated. A Project that sets
    gaussian_scope_divergence_rad to 0.4 states it, lambda F / (pi w), through the run and through the JSON API with the
    run's handoff document; the check names the setting and the analysis records the limit in force."""

    wanted = request(0.65, illumination={"kind": "GAUSSIAN", "power_w": 1.0, "radius_mm": 6.4}, sampling={"theta_nodes": 48, "phi_nodes": 48, "grid_cells": 64})
    default = load_run(run(_pupil_plane_run())).focal_field("PUPIL", wanted)
    assert default.quantity("gaussian_waist_radius").state is QuantityState.OUT_OF_MODEL_SCOPE
    check = next(c for m in default.models if m.model_id == "paraxial_gaussian_focus" for c in m.validity if c.parameter == "far_field_half_angle_rad")
    assert check.maximum == 0.3 and check.observed == pytest.approx(0.32) and check.within is False and check.setting == "gaussian_scope_divergence_rad"
    assert "numerical_settings" not in default.settings
    doc = _pupil_plane_run()
    doc["numerical_settings"] = {"gaussian_scope_divergence_rad": 0.4}
    results = load_run(run(doc))
    raised = results.focal_field("PUPIL", wanted)
    assert value(raised, "gaussian_waist_radius") == pytest.approx(852.0e-6 * 20.0 / (math.pi * 6.4) * 1e3, rel=2e-9)
    check = next(c for m in raised.models if m.model_id == "paraxial_gaussian_focus" for c in m.validity if c.parameter == "far_field_half_angle_rad")
    assert check.maximum == 0.4 and check.within is True and check.setting == "gaussian_scope_divergence_rad"
    assert raised.settings["numerical_settings"] == {"gaussian_scope_divergence_rad": 0.4}
    reply = json_api.focal_field(wanted, results.handoff("PUPIL"))
    assert reply["ok"] is True and reply["quantities"] == json.loads(raised.model_dump_json(by_alias=True))["quantities"]
    assert reply["settings"]["numerical_settings"] == {"gaussian_scope_divergence_rad": 0.4}
    # inside the default limit nothing is named: the document of a default run is as before
    inside = focal_field(request(0.05, wavelength=1064.0, focal=100.0, illumination={"kind": "GAUSSIAN", "power_w": 0.5, "radius_mm": 1.0}))
    assert all(c.setting is None for m in inside.models for c in m.validity) and "numerical_settings" not in inside.settings


def test_an_incoherent_focal_field_leaves_out_a_beam_no_field_follows_from_and_names_it():
    """Two beams on the pupil plane, one with M^2 = 1.5: the incoherent focal field is that of the TEM00 beam (the analysis of
    a request naming it alone, quantity for quantity) and the document lists the other as excluded; a coherent one refuses."""

    a = source("A", position=(-100.0, 0.0, 0.0), wavelength_nm=852.0, power_w=2e-3, waist_mm=4.0)
    b = source("B", position=(-100.0, 1.0, 0.0), wavelength_nm=852.0, power_w=1e-3, waist_mm=4.0)
    b["source_model"]["m2"] = 1.5
    results = load_run(run(project([a, b], [plane("PUPIL", (0.0, 0.0, 0.0), radius_mm=20.0)])))
    paths = {beam["source_id"]: beam["path_id"] for beam in results.handoff("PUPIL")["beams"]}
    body = _handoff_request()
    doc = results.focal_field("PUPIL", body)
    alone = results.focal_field("PUPIL", {**body, "illumination": {**body["illumination"], "path_ids": [paths["A"]]}})
    assert doc.arrays["illumination"]["complete"] is False and "illumination" not in alone.arrays
    (row,) = doc.arrays["illumination"]["excluded_beams"]
    assert row["path_id"] == paths["B"] and row["reason"] == "BEAM_NOT_TEM00" and row["power_W"] == pytest.approx(1e-3, rel=1e-9)
    assert value(doc, "incident_power_W") == pytest.approx(2e-3, rel=3e-9)
    assert doc.quantity("incident_power_W").name == "power of the beams the field is built from"
    assert {q.quantity_id: (q.state, q.value) for q in doc.quantities} == {q.quantity_id: (q.state, q.value) for q in alone.quantities}
    with pytest.raises(FocalFieldError) as exc:
        results.focal_field("PUPIL", {**body, "illumination": {"kind": "HANDOFF_BEAMS", "composition": "COHERENT"}})
    assert exc.value.code == "BEAM_NOT_TEM00" and "INCOHERENT" in exc.value.message


def test_discovery_and_the_command_line(tmp_path, capsys):
    caps = json_api.capabilities()
    assert caps["field_analyses"]["focal_field"]["schema_kind"] == "focal-field-request"
    assert {"focal-field-request", "analysis-result"} <= set(caps["schema_kinds"])
    assert json_api.json_schema("focal-field-request")["schema"]["title"] == "FocalFieldRequestV1"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request(0.5, output={"samples": 11})), encoding="utf-8")
    assert cli_main(["results", "focal-field", "--request", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "FOCAL_FIELD" and out["execution"]["outcome"] == "COMPLETED"
    path.write_text(json.dumps(request(0.1, wavelength=1000.0, focal=0.05, illumination={"kind": "UNIFORM", "power_w": 1.0})), encoding="utf-8")
    assert cli_main(["results", "focal-field", "--request", str(path)]) == 0
    outside = json.loads(capsys.readouterr().out)
    assert outside["execution"]["outcome"] == "COMPLETED"
    peak = next(q for q in outside["quantities"] if q["quantity_id"] == "peak_electric_irradiance")
    assert peak["state"] == "OUT_OF_MODEL_SCOPE" and peak["reason"] == "DEBYE_APPROXIMATION_INVALID" and peak["value"] is None
    path.write_text(json.dumps({"wavelength_nm": 852.0}), encoding="utf-8")
    assert cli_main(["results", "focal-field", "--request", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "FOCAL_FIELD_INVALID"
    assert cli_main(["results", "focal-field", "--request", str(path), "--monitor", "M"]) == 64
    capsys.readouterr()


def test_the_document_is_reproducible(na065):
    again = focal_field(request(0.65))
    assert again.model_dump(mode="json") == na065.model_dump(mode="json")
    assert again.identity.physical.settings_hash == na065.identity.physical.settings_hash
