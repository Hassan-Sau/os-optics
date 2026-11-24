"""Thin complex-transmission samples (amplitude and phase over a plane), against closed-form references.

References, computed here independently of the Engine:

* a uniform sample ``t = a exp(i phi)``: power ``a^2 P`` and optical path ``+ phi lambda / (2 pi)``;
* a linear phase ``phi = g u``: the local grating equation, ``sin(theta') = sin(theta) + lambda g / (2 pi)`` in the plane of the
  ramp (air on both sides);
* an amplitude linear across the plane, ``A = a0 + c u``: a Gaussian beam of 1/e^2 radius ``w`` keeps ``<A^2> = a0^2 + c^2 <u^2>``,
  ``<u^2> = w^2 / 4`` at normal incidence and ``w^2 / (4 cos^2(theta))`` along the plane of incidence at ``theta`` (the footprint
  is stretched by ``1 / cos(theta)``); a bilinear map of a linear function is that function;
* a Zernike defocus ``phi = c sqrt(3) (2 rho^2 - 1)`` (Noll 4, unit RMS) over a normalization radius ``R`` is the phase of a thin
  lens of focal length ``f = k0 R^2 / (4 sqrt(3) c)`` (``phi = -k0 r^2 / (2 f)`` up to a constant): the rays meet at ``f``; its
  gradient ``4 sqrt(3) |c| r / R^2`` reaches ``k0`` at ``r_cut``, beyond which the thin element's transmitted light is evanescent:
  the share of a Gaussian beam of radius ``w`` that propagates is ``1 - exp(-2 r_cut^2 / w^2)``.
"""

from __future__ import annotations

import math

import pytest
from declared_helpers import declared_component, stop_model, with_models
from local_helpers import plane, project, run, source
from pydantic import ValidationError

from os_optics import EngineConfiguration, ValidationRequest, validate
from os_optics import builders as B
from os_optics.contracts import (ComplexTransmissionInteraction, EngineeringComponentSnapshot, LinearPhaseRamp, PhaseStep, PlaneSurface,
                                 ProjectSnapshot, SampledTransmissionMap, Target, ZernikePhase, ZernikePhaseTerm)
from os_optics.contracts.optics import RectangularAperture
from os_optics.local import LocalRunError

WL = 633.0
LAM_MM = WL * 1e-6
K0 = 2.0 * math.pi / LAM_MM


def _scene(interaction, *, det_z: float = 50.0, waist: float = 1.0, tilt_deg: float = 0.0, half: float = 8.0) -> ProjectSnapshot:
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    src = B.laser("S", wavelength_nm=WL, power_W=1.0, waist_radius_mm=waist, position=(-100.0 * s, 0.0, -100.0 * c), direction=(s, 0.0, c))
    surf = PlaneSurface(surface_id="t", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                        aperture=RectangularAperture(half_width_mm=half, half_height_mm=half), aperture_u_axis=(1.0, 0.0, 0.0),
                        interaction=interaction)
    sample = EngineeringComponentSnapshot(instance_id="SAMPLE", role="other_optical", surfaces=(surf,))
    det = B.observation_plane("D", position=(0.0, 0.0, det_z), normal=(0.0, 0.0, 1.0), absorbing=True)
    return ProjectSnapshot(name="sample", components=(src, sample), observations=(det,))


def _arrival(snapshot: ProjectSnapshot):
    result = validate(ValidationRequest(project=snapshot, configuration=EngineConfiguration(run_layout=False, run_control=False)))
    (arrival,) = result.observations[0].arrivals
    return arrival, result


def _beam_radius_at_sample(waist: float = 1.0, distance: float = 100.0) -> float:
    z_r = math.pi * waist * waist / LAM_MM
    return waist * math.sqrt(1.0 + (distance / z_r) ** 2)


def _linear_map(a0: float, c: float, *, nodes: int = 17, pitch: float = 1.0, phase_slope: float | None = None) -> SampledTransmissionMap:
    us = [(i - (nodes - 1) / 2) * pitch for i in range(nodes)]
    amplitude = tuple(a0 + c * u for _v in us for u in us)
    phase = None if phase_slope is None else tuple(phase_slope * u for _v in us for u in us)
    return SampledTransmissionMap(nodes_u=nodes, nodes_v=nodes, pitch_mm=(pitch, pitch), amplitude=amplitude, phase_rad=phase)


def test_a_uniform_sample_scales_the_power_and_adds_its_optical_path():
    plain, _ = _arrival(_scene(ComplexTransmissionInteraction()))
    arrival, result = _arrival(_scene(ComplexTransmissionInteraction(amplitude=0.8, phase_rad=1.3)))
    assert arrival.beam.power_W == pytest.approx(0.64, rel=1e-12)
    assert arrival.beam.optical_path_length_mm - plain.beam.optical_path_length_mm == pytest.approx(1.3 / K0, abs=1e-12)
    assert arrival.beam.gaussian.valid and arrival.beam.direction == plain.beam.direction
    (line,) = result.power_budget.lines
    assert line.conserved and line.sinks["ELEMENT_TRANSMISSION_LOSS"] == pytest.approx(0.36, rel=1e-12)


@pytest.mark.parametrize("tilt_deg", [0.0, 20.0])
def test_a_linear_phase_steers_by_the_local_grating_equation(tilt_deg):
    g = 50.0  # rad/mm
    arrival, result = _arrival(_scene(ComplexTransmissionInteraction(phase_terms=(LinearPhaseRamp(gradient_rad_per_mm=(g, 0.0)),)),
                                      tilt_deg=tilt_deg))
    sin_out = math.sin(math.radians(tilt_deg)) + LAM_MM * g / (2.0 * math.pi)
    assert arrival.beam.direction[0] == pytest.approx(sin_out, abs=1e-12)
    assert abs(arrival.beam.direction[1]) < 1e-15
    assert arrival.beam.gaussian.valid  # (a linear phase keeps the beam Gaussian: no phase structure across it)
    assert not [i for i in result.issues if i.code.startswith("TRANSMISSION_MAP")]


def test_a_sampled_linear_phase_steers_as_the_analytic_ramp():
    g = 40.0
    analytic, _ = _arrival(_scene(ComplexTransmissionInteraction(phase_terms=(LinearPhaseRamp(gradient_rad_per_mm=(g, 0.0)),))))
    sampled, _ = _arrival(_scene(ComplexTransmissionInteraction(sampled_map=_linear_map(1.0, 0.0, phase_slope=g))))
    assert sampled.beam.direction == pytest.approx(analytic.beam.direction, abs=1e-12)


def test_an_evanescent_phase_gradient_ends_the_path_out_of_model_scope():
    g = 1.01 * 2.0 * math.pi / LAM_MM
    _arrival_result = validate(ValidationRequest(project=_scene(ComplexTransmissionInteraction(
        phase_terms=(LinearPhaseRamp(gradient_rad_per_mm=(g, 0.0)),))), configuration=EngineConfiguration(run_layout=False, run_control=False)))
    assert not _arrival_result.observations[0].arrivals
    issue = next(i for i in _arrival_result.issues if i.code == "TRANSMISSION_MAP_EVANESCENT")
    assert issue.status == "OUT_OF_MODEL_SCOPE"
    (line,) = _arrival_result.power_budget.lines
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(1.0)


def _defocus_reaching_k0(r_cut: float, radius: float = 12.0) -> ComplexTransmissionInteraction:
    """A Zernike defocus whose phase gradient reaches ``k0`` at the radius ``r_cut``: ``|grad phi| = 4 sqrt(3) |c| r / R^2``."""
    coefficient = K0 * radius * radius / (4.0 * math.sqrt(3.0) * r_cut)
    return ComplexTransmissionInteraction(phase_terms=(ZernikePhase(indexing="NOLL", normalization="RMS_ORTHONORMAL", normalization_radius_mm=radius,
                                                                    terms=(ZernikePhaseTerm(index=4, value_rad=-coefficient),)),))


def _sampled_scene(interaction, rings: int, per_ring: int, *, minimum_W: float = 0.95) -> ProjectSnapshot:
    snapshot = _scene(interaction, det_z=0.02)
    src = snapshot.components[0]
    src = src.model_copy(update={"source": src.source.model_copy(update={"pupil_ring_count": rings, "pupil_points_per_ring": per_ring})})
    return snapshot.model_copy(update={"components": (src, *snapshot.components[1:]),
                                       "targets": (Target(target_id="p", metric="power_W", observation_id="D", minimum=minimum_W, unit="W"),)})


@pytest.mark.parametrize("r_cut", [0.5, 0.8, 1.2])
def test_a_sample_that_turns_the_beam_s_outer_part_beyond_grazing_books_the_evanescent_share(r_cut):
    """The chief ray propagates, the outer part of the beam is turned beyond grazing. The thin-element share that propagates is
    ``1 - exp(-2 r_cut^2 / w^2)``; before, the arrival kept the full 1 W and a 0.95 W minimum was MET (at r_cut 1.2 every pupil
    ray propagated, the Gaussian tail beyond r_cut did not)."""
    w = _beam_radius_at_sample()
    truth = 1.0 - math.exp(-2.0 * r_cut * r_cut / (w * w))
    resolutions = []
    for rings, per_ring in ((8, 12), (32, 32)):
        arrival, result = _arrival(_sampled_scene(_defocus_reaching_k0(r_cut), rings, per_ring))
        assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES" and not arrival.beam.gaussian.valid
        assert abs(arrival.beam.power_W - truth) <= arrival.power_sampling_resolution_W
        resolutions.append(arrival.power_sampling_resolution_W)
        issue = next(i for i in result.issues if i.code == "TRANSMISSION_MAP_EVANESCENT")
        assert issue.status == "OUT_OF_MODEL_SCOPE"
        (line,) = result.power_budget.lines
        assert line.conserved
        # the turned samples' share leaves the beam as light of unknown fate (a phase-only sample loses nothing else)
        assert line.sinks.get("TERMINATED_OUT_OF_MODEL", 0.0) == pytest.approx(1.0 - arrival.beam.power_W, abs=1e-12)
        assert result.targets[0].assessment != "MET"
        # (the target names the cause: the sample turned light beyond grazing, not an aperture)
        assert "a thin sample turned part of the beam beyond grazing" in result.targets[0].message
        assert "aperture that cut them" not in result.targets[0].message
    assert resolutions[1] < resolutions[0]  # (more pupil samples narrow it)


@pytest.mark.parametrize("r_cut", [4.0, 5.0])
def test_a_sample_that_turns_only_a_negligible_tail_beyond_grazing_keeps_the_decided_power(r_cut):
    """The phase turns the Gaussian tail beyond ``r_cut`` beyond grazing, a share ``exp(-2 r_cut^2 / w^2)`` (1.3e-14 at 4 w,
    2e-22 at 5 w) far below the line's negligible level (``negligible_power_fraction`` x 1 W). Before, any turned footprint node
    put the beam on the pupil-sample basis: 1 W within 0.125 W, the 0.95 W minimum INDETERMINATE. Now the share is left out as
    negligible with a bound at least the thin-element share, and the beam keeps its power: the minimum is MET."""
    from os_optics.config import NumericalConfig
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.optics.rays.tracer import Tracer
    from os_optics.scene.lower import lower_scene

    w = _beam_radius_at_sample()
    share = math.exp(-2.0 * r_cut * r_cut / (w * w))
    snapshot = _sampled_scene(_defocus_reaching_k0(r_cut), 8, 12)
    arrival, result = _arrival(snapshot)
    assert arrival.power_basis is None and arrival.power_sampling_resolution_W is None
    assert arrival.beam.power_W == pytest.approx(1.0, rel=1e-12)
    assert not any(i.code == "TRANSMISSION_MAP_EVANESCENT" for i in result.issues)
    assert result.targets[0].assessment == "MET"
    cfg = NumericalConfig()
    library = MaterialLibrary(snapshot.materials, cfg)
    scene, _ = lower_scene(snapshot, cfg, library)
    left = Tracer(scene, library, cfg, include_pupil=True).run().left_out_W[("S", 0)]
    assert share <= left <= cfg.negligible_power_fraction * 1.0


def test_a_sample_whose_turned_tail_is_above_the_negligible_level_stays_on_the_sampled_basis():
    """At ``r_cut`` 3 w the turned share (1.5e-8) exceeds the negligible level (1e-9 of the line): the pupil-sample basis stays."""
    arrival, result = _arrival(_sampled_scene(_defocus_reaching_k0(3.0), 8, 12))
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    assert any(i.code == "TRANSMISSION_MAP_EVANESCENT" for i in result.issues)


def test_a_sample_that_turns_no_part_of_the_beam_beyond_grazing_keeps_the_gaussian_power():
    arrival, result = _arrival(_sampled_scene(_defocus_reaching_k0(6.0), 8, 12, minimum_W=0.5))  # (k0 is reached at 6 w)
    assert arrival.power_basis is None and arrival.beam.power_W == pytest.approx(1.0, rel=1e-12)
    assert not any(i.code == "TRANSMISSION_MAP_EVANESCENT" for i in result.issues)


@pytest.mark.parametrize("tilt_deg", [0.0, 30.0])
def test_the_power_through_an_amplitude_map_is_the_gaussian_weighted_mean_of_its_square(tilt_deg):
    a0, c = 0.5, 0.05
    arrival, result = _arrival(_scene(ComplexTransmissionInteraction(sampled_map=_linear_map(a0, c)), tilt_deg=tilt_deg))
    w = _beam_radius_at_sample()
    cos_t = math.cos(math.radians(tilt_deg))
    reference = a0 * a0 + c * c * w * w / (4.0 * cos_t * cos_t)
    assert arrival.beam.power_W == pytest.approx(reference, rel=1e-9)
    assert not arrival.power_out_of_scope_at
    assert arrival.beam.gaussian.valid  # (an amplitude map alone leaves the phase a plane)


def test_a_zernike_defocus_focuses_the_rays_and_the_beam_is_no_longer_gaussian():
    radius, coefficient = 12.0, -40.0
    focal = K0 * radius * radius / (4.0 * math.sqrt(3.0) * -coefficient)
    zernike = ZernikePhase(indexing="NOLL", normalization="RMS_ORTHONORMAL", normalization_radius_mm=radius,
                           terms=(ZernikePhaseTerm(index=4, value_rad=coefficient),))
    arrival, result = _arrival(_scene(ComplexTransmissionInteraction(phase_terms=(zernike,)), det_z=focal))
    assert arrival.beam.power_W == pytest.approx(1.0, rel=1e-12)  # (a phase-only sample)
    assert not arrival.beam.gaussian.valid
    assert any(i.code == "TRANSMISSION_MAP_PHASE_STRUCTURE" and i.status == "OUT_OF_MODEL_SCOPE" for i in result.issues)
    w = _beam_radius_at_sample()
    assert arrival.pupil_rms_radius_mm < 1e-6 * w  # (the rays meet at the thin-lens focus; sin vs tan at NA ~ 2e-4)


def test_a_phase_step_across_the_beam_keeps_the_rays_and_drops_the_gaussian_beam():
    plain, _ = _arrival(_scene(ComplexTransmissionInteraction()))
    across, result = _arrival(_scene(ComplexTransmissionInteraction(phase_terms=(PhaseStep(phase_rad=math.pi),))))
    assert not across.beam.gaussian.valid and across.beam.direction == plain.beam.direction
    assert across.beam.power_W == pytest.approx(1.0, rel=1e-12)
    # an edge far from the beam: its phase is flat across the footprint (a constant on one side)
    beside, _ = _arrival(_scene(ComplexTransmissionInteraction(phase_terms=(PhaseStep(phase_rad=math.pi, edge_offset_mm=7.5),))))
    assert beside.beam.gaussian.valid
    assert beside.beam.optical_path_length_mm == pytest.approx(plain.beam.optical_path_length_mm, abs=1e-12)


@pytest.mark.parametrize("fields, message", [
    ({"nodes_u": 2, "nodes_v": 2, "pitch_mm": (1.0, 1.0)}, "amplitude, phase_rad or both"),
    ({"nodes_u": 2, "nodes_v": 2, "pitch_mm": (1.0, 1.0), "amplitude": (1.0, 1.0, 1.0)}, "4 values"),
    ({"nodes_u": 2, "nodes_v": 2, "pitch_mm": (1.0, 1.0), "amplitude": (1.0, 1.0, 1.0, 1.2)}, "passive"),
    ({"nodes_u": 2, "nodes_v": 2, "pitch_mm": (0.0, 1.0), "phase_rad": (0.0, 0.0, 0.0, 0.0)}, "pitch"),
])
def test_invalid_maps_are_refused(fields, message):
    with pytest.raises(ValidationError, match=message):
        SampledTransmissionMap(**fields)


def test_a_map_that_does_not_cover_the_clear_aperture_is_refused():
    small = _linear_map(1.0, 0.0, nodes=5)  # (covers +-2 mm; the aperture reaches 8 mm)
    result = validate(ValidationRequest(project=_scene(ComplexTransmissionInteraction(sampled_map=small)),
                                        configuration=EngineConfiguration(run_layout=False, run_control=False)))
    assert result.status == "FAIL" and not result.observations
    assert any("does not cover the clear aperture" in i.message for i in result.issues)


# ------------------------------------------------------------------------------------------------ the Project's sample


def _sample_document(sample: dict | None) -> dict:
    fields = {"opening_half_widths_mm": ["5", "5"], "edge_half_widths_mm": ["10", "10"]}
    if sample is not None:
        fields["sample"] = sample
    doc = project([source("SRC", (-100.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=0.5),
                   declared_component("CELL", "other_optical", "cell", (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))],
                  [plane("D", (50.0, 0.0, 0.0), radius_mm=5.0)])
    return with_models(doc, [stop_model("cell", **fields)])


def _project_arrival(sample: dict | None):
    result = run(_sample_document(sample)).result
    (arrival,) = result.observations[0].arrivals
    return arrival, result


def test_a_project_sample_reproduces_the_uniform_power_and_optical_path():
    plain, _ = _project_arrival(None)
    arrival, _ = _project_arrival({"amplitude": "0.9", "phase_rad": "0.5", "basis": "measured with a phase-shifting interferometer"})
    assert arrival.beam.power_W == pytest.approx(plain.beam.power_W * 0.81, rel=1e-12)
    assert arrival.beam.optical_path_length_mm - plain.beam.optical_path_length_mm == pytest.approx(0.5 * 632.8e-6 / (2.0 * math.pi), abs=1e-12)


def test_a_project_sample_map_modulates_the_wavefront():
    nodes, pitch, slope = 11, 1.0, 30.0
    xs = [(i - (nodes - 1) / 2) * pitch for i in range(nodes)]
    sample = {"sampled_map": {"nodes_x": nodes, "nodes_y": nodes, "pitch_mm": ["1", "1"],
                              "phase_rad": [str(int(slope * x)) for _y in xs for x in xs]}}
    arrival, _ = _project_arrival(sample)
    # the element frame's x is the design frame's ... the beam turns by the grating equation in the element's x direction
    turned = math.hypot(arrival.beam.direction[1], arrival.beam.direction[2])
    assert turned == pytest.approx(632.8e-6 * slope / (2.0 * math.pi), abs=1e-12)


def test_project_sample_refusals():
    with pytest.raises(LocalRunError, match="amplitude"):
        run(_sample_document({"amplitude": "1.2"}))
    with pytest.raises(LocalRunError, match="LINEAR_RAMP phase states gradient_rad_per_mm"):
        run(_sample_document({"phase_terms": [{"kind": "LINEAR_RAMP", "phase_rad": "1"}]}))
    with pytest.raises(LocalRunError, match="does not cover"):
        run(_sample_document({"sampled_map": {"nodes_x": 2, "nodes_y": 2, "pitch_mm": ["1", "1"], "phase_rad": ["0", "0", "0", "0"]}}))


def test_the_recorded_homogeneous_plate_with_a_sample_behind_it():
    # the recorded sample geometry: a 5 mm plate of index 1.6 and absorption 0.01 / mm at 632.8 nm (independent value of the
    # plate alone: (1 - R)^2 exp(-0.05) x 1 mW = 0.852612768013 mW), then a thin sample of amplitude 0.9 and phase 0.5 rad
    from declared_helpers import sequence, surface

    plate_glass = {"material_id": "SAMPLE-GLASS", "display_name": "Sample glass", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.6"},
                   "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}, "absorption_coefficient_per_mm": "0.01"}
    plate = {"model_id": "plate", "model": sequence([surface(1, "0", clear="9"), surface(2, "5", clear="9")], ["SAMPLE-GLASS"])}
    reflectance = (0.6 / 2.6) ** 2
    independent = 1e-3 * (1.0 - reflectance) ** 2 * math.exp(-0.05)
    assert independent == pytest.approx(0.852612768013e-3, rel=1e-11)
    for sample, factor in ((None, 1.0), ({"amplitude": "0.9", "phase_rad": "0.5"}, 0.81)):
        doc = _sample_document(sample)
        doc["components"].append(declared_component("PLATE", "window", "plate", (-50.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)))
        doc["optical_models"].append(plate)
        doc["model_materials"] = [plate_glass]
        result = run(doc).result
        (arrival,) = result.observations[0].arrivals
        assert arrival.beam.power_W == pytest.approx(independent * factor, rel=1e-12)
