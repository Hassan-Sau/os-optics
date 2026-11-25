"""The imaging analysis of a partially polarized or unpolarized source: its two incoherent polarization components may reach the
image plane alone (a polarizer passes either) or together; a component that reaches the plane is analysed, and together their
point spread functions add in irradiance by their shares of the light (``results/imaging_trace.py``, routes ``<source>#<i><branch>`` / ``<source>#<i>:p1<branch>``).

The combination is checked against the components traced on their own: a source polarized along each component's state, with
the result weighted by the component shares."""

from __future__ import annotations

import numpy as np
import pytest

from os_optics.errors.status import QuantityState
from os_optics.results import load_run

from declared_helpers import declared_component
from local_helpers import run
from test_imaging import ENGINE, _hyperbolic, _lens_project, _line
from test_imaging_pupil_amplitude import _curves
from test_polarization_extensions import _turned, polarizer

WL = 587.5618
LINE_POWER_W = 0.000458720936769186  # what the singlet delivers to its image plane through the polarizer (either orientation)


def _singlet_behind_polarizer(angle_deg: float) -> dict:
    """F/8 40 mm hyperbolic singlet, an UNPOLARIZED source, an ideal linear polarizer (T = 1, extinction 1e12) 5 mm before it."""

    doc = _lens_project(_hyperbolic(8.0, 40.0, WL), waist_mm=1.0)
    # Inline prescription glass here is an ideal test medium, not a measured catalog loss.
    for material in doc["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    doc["components"][0]["source_model"]["polarization"] = {"kind": "UNPOLARIZED"}
    doc["components"].append(declared_component("P", "polarizer", "pol", position=(-6.5, 0.0, 0.0), quaternion=_turned(angle_deg)))
    doc["optical_models"].append(polarizer(None, transmittance="1", extinction="1000000000000"))
    return doc


@pytest.mark.parametrize(("angle_deg", "path_id"), [(0.0, "SRC#0"), (90.0, "SRC#0:p1")])
def test_either_component_alone_is_analysed(angle_deg: float, path_id: str):
    """At 0 deg only the first component reaches the image plane, at 90 deg only the second: each is analysed (it was NO_ARRIVAL
    for the second), with the run's arrival and power, and the aberration-free singlet's Airy MTF."""

    results = load_run(run(_singlet_behind_polarizer(angle_deg)))
    (arrival,) = results._observation("image").arrivals
    assert arrival.beam.path_id == path_id and arrival.beam.power_W == pytest.approx(LINE_POWER_W, rel=1e-12)
    doc = results.imaging("image", {"launch_grid": 32, "psf_samples": 32})
    assert all(q.reason != "NO_ARRIVAL" for q in doc.quantities)
    p = _line(doc, "SRC#")
    assert doc.quantity(p + "strehl_ratio").state is QuantityState.KNOWN
    assert doc.quantity(p + "strehl_ratio").value == pytest.approx(1.0, abs=1e-2)
    f, t, s = _curves(doc)
    cutoff = 1.0 / (WL * 1e-6 * 8.0)
    nu = np.minimum(f / cutoff, 1.0)
    airy = (2.0 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1.0 - nu * nu))
    inside = f < 0.9 * cutoff
    assert np.abs(t[inside] - airy[inside]).max() < 0.01 and np.abs(s[inside] - airy[inside]).max() < 0.01


def _double_gauss(polarization: dict) -> object:
    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    project = _lens_project(text, waist_mm=3.0)
    for material in project["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"  # Test-only dispersion without assumed real-world loss.
    project["components"][0]["source_model"]["polarization"] = polarization
    return load_run(run(project)).imaging("image")


@pytest.fixture(scope="module")
def linear_mtfs():
    """The double Gauss on axis with the source polarized along x (an unpolarized source's first component) and along y (its
    second): their tangential and sagittal MTF differ by the surfaces' s and p transmissions (0.0125)."""

    return {name: _curves(_double_gauss({"kind": "LINEAR", "angle_deg": angle})) for name, angle in (("x", 0), ("y", 90))}


@pytest.mark.parametrize(("polarization", "share_x"), [
    ({"kind": "UNPOLARIZED"}, 0.5),
    ({"kind": "PARTIAL", "degree_of_polarization": 0.5, "polarized": {"kind": "LINEAR", "angle_deg": 0}}, 0.75),
])
def test_both_components_add_by_their_shares(linear_mtfs, polarization: dict, share_x: float):
    """Both components reach the plane on one route: the MTF is that of the components traced alone, weighted by their shares
    (their point spread functions add; on axis here the components' OTFs are real and positive, so the MTFs add too)."""

    f, t, s = _curves(_double_gauss(polarization))
    fx, tx, sx = linear_mtfs["x"]
    fy, ty, sy = linear_mtfs["y"]
    assert np.array_equal(f, fx) and np.array_equal(f, fy)
    expected_t = share_x * tx + (1.0 - share_x) * ty
    expected_s = share_x * sx + (1.0 - share_x) * sy
    assert np.abs(t - expected_t).max() < 1e-4 and np.abs(s - expected_s).max() < 1e-4
    assert np.abs(tx - sx).max() > 1e-2  # (the components alone do differ: the test resolves the weighting)


def test_components_whose_rays_part_are_not_added(monkeypatch):
    """Where the two components' rays do not cross the plane at the same points they are not added as one pupil: the field
    states why (POLARIZATION_COMPONENT_ROUTES_DIFFER), not NO_ARRIVAL, for the spot and the wavefront as for the diffraction
    quantities: none of them describes one component alone."""

    import os_optics.results.imaging_trace as imaging_trace

    monkeypatch.setattr(imaging_trace, "ROUTE_TOLERANCE_MM", -1.0)  # (no two crossings agree: every pair reads as parted)
    doc = load_run(run(_singlet_behind_polarizer(45.0))).imaging("image", {"launch_grid": 32, "psf_samples": 32})
    p = _line(doc, "SRC#")
    for name in ("strehl_ratio", "image_space_na", "geometric_spot_rms_radius", "geometric_mtf_tangential", "opd_rms_waves"):
        assert doc.quantity(p + name).reason == "POLARIZATION_COMPONENT_ROUTES_DIFFER", name
    assert all(q.reason != "NO_ARRIVAL" for q in doc.quantities)
