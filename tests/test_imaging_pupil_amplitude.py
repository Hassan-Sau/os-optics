"""The imaging analysis weights its pupil by each ray's field: a thin sample of varying amplitude before an aberration-free F/8 lens
changes its point spread function and MTF as the scalar pupil-autocorrelation of that amplitude says (``results/imaging.py``,
each ray's Jones vector recorded where it crosses the image plane)."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics.results import load_run

from declared_helpers import declared_component, stop_model
from local_helpers import run
from test_imaging import _hyperbolic, _lens_project, _line

WL = 587.5618
F, FNUM = 40.0, 8.0
A = F / (2.0 * FNUM)  # the lens's semi-aperture (its rim is the stop): 2.5 mm
NODES, PITCH = 21, 0.5  # the sampled map: 21 x 21 nodes, 0.5 mm apart (+-5 mm), linear in u
SLOPE, MEAN = 0.09, 0.5  # t(u) = MEAN + SLOPE u: 0.275 .. 0.725 across the pupil (0.05 .. 0.95 over the map)


def _amplitude(u: np.ndarray) -> np.ndarray:
    return MEAN + SLOPE * u


def _project(varying: bool) -> dict:
    doc = _lens_project(_hyperbolic(FNUM, F, WL), waist_mm=1.0)
    xs = [(i - (NODES - 1) // 2) * Decimal("0.5") for i in range(NODES)]
    amplitude = [str((Decimal("0.5") + Decimal("0.09") * x if varying else Decimal("0.5")).normalize()) for _y in xs for x in xs]
    sample = {"sampled_map": {"nodes_x": NODES, "nodes_y": NODES, "pitch_mm": [f"{PITCH}", f"{PITCH}"], "amplitude": amplitude}}
    doc["components"].append(declared_component("CELL", "other_optical", "cell", (-10.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)))
    doc["optical_models"].append(stop_model("cell", opening_half_widths_mm=["5", "5"], edge_half_widths_mm=["10", "10"], sample=sample))
    return doc


def _reference_mtf(frequencies: np.ndarray, along_u: bool, varying: bool) -> np.ndarray:
    """The scalar diffraction MTF of the pupil ``t(u)`` on a disc of radius A: the normalized autocorrelation of the pupil
    function at the shear ``lambda F nu`` (along u or across it), integrated on a fine grid of the analytic functions."""

    h = A / 600.0
    x = np.arange(-A, A + h, h)
    xx, yy = np.meshgrid(x, x, indexing="ij")  # xx along u
    disc = xx * xx + yy * yy < A * A

    def pupil(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        inside = px * px + py * py < A * A
        return np.where(inside, _amplitude(px) if varying else MEAN, 0.0)

    p0 = np.where(disc, pupil(xx, yy), 0.0)
    norm = float((p0 * p0).sum())
    out = []
    for nu in frequencies:
        s = WL * 1e-6 * F * float(nu)
        shifted = pupil(xx - s, yy) if along_u else pupil(xx, yy - s)
        out.append(float((p0 * shifted).sum()) / norm)
    return np.array(out)


@pytest.fixture(scope="module")
def documents():
    return {varying: load_run(run(_project(varying))).imaging("image", {"launch_grid": 96}) for varying in (False, True)}


def _curves(doc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _line(doc, "SRC#")
    f = np.array(doc.arrays[p + "frequencies_cyc_per_mm"], dtype=float)
    keep = [i for i, v in enumerate(doc.arrays[p + "mtf_tangential"]) if v is not None and doc.arrays[p + "mtf_sagittal"][i] is not None]
    t = np.array([doc.arrays[p + "mtf_tangential"][i] for i in keep])
    s = np.array([doc.arrays[p + "mtf_sagittal"][i] for i in keep])
    return f[keep], t, s


def test_a_varying_sample_amplitude_changes_the_mtf_as_the_pupil_autocorrelation_says(documents):
    cutoff = 1.0 / (WL * 1e-6 * FNUM)
    for varying, doc in documents.items():
        f, t, s = _curves(doc)
        inside = f < 0.9 * cutoff
        along = _reference_mtf(f[inside], True, varying)
        across = _reference_mtf(f[inside], False, varying)
        engine = (t[inside], s[inside])
        # the plane's tangential axis is one of the pupil's axes: the pair matches the reference pair in either order
        best = min(max(np.abs(engine[0] - along).max(), np.abs(engine[1] - across).max()),
                   max(np.abs(engine[0] - across).max(), np.abs(engine[1] - along).max()))
        assert best < 0.01, (varying, best)
    f_u, t_u, s_u = _curves(documents[False])
    f_v, t_v, s_v = _curves(documents[True])
    half = int(np.argmin(np.abs(f_u - 0.5 * cutoff)))
    # at half the cutoff the uniform pupil gives the Airy value (2 / pi)(acos 0.5 - 0.5 sqrt(0.75)) = 0.391; the varying one less
    # along its gradient
    airy = (2.0 / math.pi) * (math.acos(0.5) - 0.5 * math.sqrt(0.75))
    assert t_u[half] == pytest.approx(airy, abs=0.01) and s_u[half] == pytest.approx(airy, abs=0.01)
    assert min(t_v[half], s_v[half]) < airy - 0.03


def test_the_psf_and_encircled_energy_follow_the_amplitude(documents):
    uniform, varying = documents[False], documents[True]
    p, q = _line(uniform, "SRC#"), _line(varying, "SRC#")
    assert uniform.quantity(p + "encircled_radius_80").value != varying.quantity(q + "encircled_radius_80").value
    assert uniform.arrays[p + "psf"] != varying.arrays[q + "psf"]


def test_an_unpolarized_source_weights_the_pupil_with_both_components():
    """An unpolarized source is traced as two incoherent orthogonal components whose point spread functions add; their sum is
    rotationally symmetric on axis: the double Gauss's tangential and sagittal MTF agree, as the mean of the x- and
    y-polarized MTF (each component's PSF adds in irradiance)."""

    from test_imaging import ENGINE

    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    docs = {}
    for name, polarization in (("x", {"kind": "LINEAR", "angle_deg": 0}), ("y", {"kind": "LINEAR", "angle_deg": 90}),
                               ("unpolarized", {"kind": "UNPOLARIZED"})):
        project = _lens_project(text, waist_mm=3.0)
        project["components"][0]["source_model"]["polarization"] = polarization
        docs[name] = load_run(run(project)).imaging("image")
    curves = {name: _curves(doc) for name, doc in docs.items()}
    _f, t, s = curves["unpolarized"]
    assert np.abs(t - s).max() < 1e-3  # (x alone: 0.0125)
    _fx, tx, sx = curves["x"]
    assert np.abs(tx - sx).max() > 5e-3
