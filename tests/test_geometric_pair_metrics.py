"""Geometric pair metrics of two beams of the same or different wavelengths (``validation/alignment.py``):
centroid / normalized / angular / waist-position separation and the envelope overlap ratio, against geometry computed here
by hand (straight chief rays, Gaussian radii from q = z + i z_R, an ideal thin lens's waist formula). They are geometric:
never WAVELENGTH_MISMATCH, no phase, no coherence."""

from __future__ import annotations

import math

import pytest

from os_optics.local import lower_project, run_optical_project

from declared_helpers import declared_component, stop_model, thin_lens_model, with_models
from local_helpers import ALONG_X, NM, plane, project, quat_multiply, rayleigh_range, source, turn_about_z

GEOMETRIC = ("centroid_separation_mm", "normalized_centroid_separation", "angular_separation_rad", "geometric_overlap_ratio",
             "waist_position_separation_mm")
W0 = 0.5


def _targets(**selection) -> list[dict]:
    return [{"target_id": f"t{i}", "metric": metric, "observation_id": "det", "minimum": 0.0, **selection}
            for i, metric in enumerate(GEOMETRIC)]


def _values(run) -> dict[str, float | None]:
    return {t.metric: t.value for t in run.result.targets}


def _radius(z: float, wavelength_nm: float, waist: float = W0) -> float:
    return waist * math.sqrt(1.0 + (z / rayleigh_range(waist, wavelength_nm)) ** 2)


def test_two_sources_of_different_wavelengths_offset_and_tilted():
    alpha = 0.002  # rad, source B turned about design z
    q_b = quat_multiply(turn_about_z(math.cos(alpha / 2.0), math.sin(alpha / 2.0)), ALONG_X)
    a = source("A", position=(-100.0, 0.0, 0.0), wavelength_nm=532.0, waist_mm=W0)
    b = source("B", position=(-100.0, 0.3, 0.0), wavelength_nm=1064.0, waist_mm=W0, quaternion=q_b)
    run = run_optical_project(project([a, b], [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)], _targets(source_id="A", source_id_b="B")))
    values = _values(run)
    y_b = 0.3 + 100.0 * math.tan(alpha)  # B's chief ray on the plane x = 0
    assert values["centroid_separation_mm"] == pytest.approx(y_b, rel=1e-9)
    assert values["angular_separation_rad"] == pytest.approx(alpha, rel=1e-9)
    w_a = _radius(100.0, 532.0)
    w_b = _radius(100.0 / math.cos(alpha), 1064.0)
    assert values["normalized_centroid_separation"] == pytest.approx(y_b / math.sqrt(w_a**2 + w_b**2), rel=1e-5)
    # round beams at (near) normal incidence: overlap = 2 w_a w_b / (w_a^2 + w_b^2) exp(-d^2 / (w_a^2 + w_b^2))
    expected = 2.0 * w_a * w_b / (w_a**2 + w_b**2) * math.exp(-(y_b**2) / (w_a**2 + w_b**2))
    assert values["geometric_overlap_ratio"] == pytest.approx(expected, rel=1e-5)
    # both waists at the sources: (W_a - W_b) . m with m = (cos(alpha / 2), sin(alpha / 2), 0), the unit mean direction
    assert values["waist_position_separation_mm"] == pytest.approx(0.3 * math.sin(alpha / 2.0), rel=1e-6)
    assert all(t.status.value == "PASS" for t in run.result.targets)
    assert not any(i.code.value == "WAVELENGTH_MISMATCH" for t in run.result.targets for i in t.issue_codes)


def test_the_interference_pair_metric_of_the_same_beams_stays_not_applicable():
    a = source("A", position=(-100.0, 0.0, 0.0), wavelength_nm=532.0, waist_mm=W0)
    b = source("B", position=(-100.0, 0.3, 0.0), wavelength_nm=1064.0, waist_mm=W0)
    targets = [{"target_id": "interf", "metric": "spatial_overlap_ratio", "observation_id": "det", "source_id": "A", "source_id_b": "B",
                "minimum": 0.0},
               {"target_id": "geo", "metric": "geometric_overlap_ratio", "observation_id": "det", "source_id": "A", "source_id_b": "B",
                "minimum": 0.5}]
    run = run_optical_project(project([a, b], [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)], targets))
    interf, geo = run.result.targets
    assert interf.status.value == "NOT_APPLICABLE" and interf.issue_codes == ("WAVELENGTH_MISMATCH",)
    assert geo.status.value == "PASS" and geo.value > 0.5


def test_two_lines_of_one_source_through_an_ideal_thin_lens():
    """An achromatic thin lens focuses two lines of one source to different planes (z_R differs): the waist behind a lens
    of focal length f for an input waist at distance d is at f + f^2 (d - f) / ((d - f)^2 + z_R^2)."""
    f, d, z_det = 100.0, 200.0, 50.0
    laser = source(position=(-d, 0.0, 0.0), waist_mm=W0)
    model = laser["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": 532.0, "power_w": 0.001}, {"wavelength_nm": 1064.0, "power_w": 0.001}]
    document = with_models(project([laser, declared_component("L", "lens", "thin")], [plane("det", (z_det, 0.0, 0.0), radius_mm=20.0)],
                                   _targets(wavelength_nm=532.0, wavelength_nm_b=1064.0)), [thin_lens_model("thin", "100")])
    run = run_optical_project(document)
    values = _values(run)

    def waist_after(wavelength_nm: float) -> float:
        zr = rayleigh_range(W0, wavelength_nm)
        return f + f * f * (d - f) / ((d - f) ** 2 + zr**2)

    def radius_at_det(wavelength_nm: float) -> float:
        q = complex(d, rayleigh_range(W0, wavelength_nm))
        q = 1.0 / (1.0 / q - 1.0 / f) + z_det
        return math.sqrt(wavelength_nm * NM / math.pi * abs(q) ** 2 / q.imag)

    assert values["waist_position_separation_mm"] == pytest.approx(abs(waist_after(532.0) - waist_after(1064.0)), rel=1e-6)
    w1, w2 = radius_at_det(532.0), radius_at_det(1064.0)
    assert values["geometric_overlap_ratio"] == pytest.approx(2.0 * w1 * w2 / (w1**2 + w2**2), rel=1e-9)
    assert values["centroid_separation_mm"] == pytest.approx(0.0, abs=1e-12)
    assert values["normalized_centroid_separation"] == 0.0 and values["angular_separation_rad"] == pytest.approx(0.0, abs=1e-12)


def test_selection_problems_are_reported():
    laser = source(position=(-100.0, 0.0, 0.0), waist_mm=W0)
    model = laser["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": 532.0, "power_w": 0.001}, {"wavelength_nm": 1064.0, "power_w": 0.001}]
    det = [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)]
    same = run_optical_project(project([laser], det, [{"target_id": "t", "metric": "centroid_separation_mm", "observation_id": "det",
                                                         "wavelength_nm": 532.0, "maximum": 1.0}]))
    (target,) = same.result.targets
    assert target.status.value == "UNKNOWN" and "same beam" in target.message
    ambiguous = run_optical_project(project([laser], det, [{"target_id": "t", "metric": "centroid_separation_mm", "observation_id": "det",
                                                              "maximum": 1.0}]))
    assert ambiguous.result.targets[0].status.value == "UNKNOWN" and ambiguous.result.targets[0].issue_codes == ("INVALID_INPUT",)
    missing = run_optical_project(project([laser], det, [{"target_id": "t", "metric": "centroid_separation_mm", "observation_id": "det",
                                                            "wavelength_nm": 532.0, "wavelength_nm_b": 777.0, "maximum": 1.0}]))
    assert missing.result.targets[0].issue_codes == ("TARGET_SELECTION_EMPTY",) and "777 nm" in missing.result.targets[0].message


def test_a_beam_without_a_valid_gaussian_state_is_out_of_scope_for_envelope_metrics():
    a = source("A", position=(-100.0, 0.0, 0.0), wavelength_nm=532.0, waist_mm=W0)
    b = source("B", position=(-100.0, 5.0, 0.0), wavelength_nm=1064.0, waist_mm=W0)
    stop = declared_component("STOP", "aperture", "stop", (-50.0, 0.0, 0.0))
    document = with_models(project([a, b, stop], [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)], _targets(source_id="A", source_id_b="B")),
                           [stop_model("stop", plane_z_mm="0", opening_radius_mm="0.3", edge_radius_mm="2")])
    values = {t.metric: (t.status.value, t.issue_codes) for t in run_optical_project(document).result.targets}
    assert values["geometric_overlap_ratio"] == ("OUT_OF_MODEL_SCOPE", ("GAUSSIAN_SEVERE_TRUNCATION",))
    assert values["normalized_centroid_separation"][0] == "OUT_OF_MODEL_SCOPE"
    assert values["centroid_separation_mm"][0] == "PASS" and values["angular_separation_rad"][0] == "PASS"


def test_wavelength_nm_b_reaches_the_engine_only_for_geometric_pair_metrics():
    laser = source(position=(-100.0, 0.0, 0.0), waist_mm=W0)
    det = [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)]
    target = {"target_id": "t", "metric": "geometric_overlap_ratio", "observation_id": "det", "wavelength_nm": 780.0,
              "wavelength_nm_b": 780.0, "minimum": 0.0}
    (engine_target,) = lower_project(project([laser], det, [target])).request.project.targets
    assert engine_target.wavelength_nm_b == 780.0
    plain = {"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0}
    request = lower_project(project([laser], det, [plain])).request.model_dump(mode="json")
    assert "wavelength_nm_b" not in request["project"]["targets"][0]
    # (a selector its metric cannot read: the requirement is carried as not assessed, with the problem)
    lowered = lower_project(project([laser], det, [dict(plain, wavelength_nm_b=780.0)]))
    (marked,) = lowered.request.project.targets
    assert marked.not_assessable is not None and marked.not_assessable.code == "TARGET_INVALID"
    assert "wavelength_nm_b" in lowered.target_problems[0].message
