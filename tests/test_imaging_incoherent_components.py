"""The imaging analysis of a partially polarized or unpolarized source propagates each polarization component's pupil to the image
apart and adds their point spread functions in irradiance (``results/imaging.py`` ``amplitudes``: one field per component), with
each component's transmitted share of the light. Adding the components' pupil irradiances into one field first is a different,
coherent pupil, and wrong where the components' pupils differ.

The case: an aberration-free F/8 singlet whose rear surface carries a synthetic coating that passes s and p equally at normal
incidence (0.5) and p only off it (s 0.01, p 1): an x- and a y-polarized beam then have very different pupils (a cross of
transmission turned by 90 deg), equal powers, and very different MTF along a given axis. The references are the single-component
runs' own point spread functions, added on one grid with their transmitted-power weights, and the OTF of that sum."""

from __future__ import annotations

import numpy as np
import pytest

import importlib

from os_optics.results import load_run

from local_helpers import run
from test_coating_tables_and_phases import coating
from test_imaging import _hyperbolic, _lens_project, _line
from test_polarization_extensions import _turned, polarizer
from declared_helpers import declared_component

WL, GRID = 587.5618, 64
LAUNCH_RADII = 3.0  # (the source's 1 mm beam radius: the 2.5 mm lens rim is the stop)
HALF_UM, SAMPLES = 150.0, 301
im = importlib.import_module("os_optics.results.imaging")  # (the module; ``os_optics.results.imaging`` is also a function)
X, Y = {"kind": "LINEAR", "angle_deg": 0}, {"kind": "LINEAR", "angle_deg": 90}


def _coating() -> dict:
    t_s, t_p = (("0.5", "0.01", "0.01"),) * 2, (("0.5", "1", "1"),) * 2
    zero = (("0", "0", "0"),) * 2
    return coating("SYN", angular_spectrum={"wavelength_nm": ["300", "2500"], "aoi_deg": ["0", "0.1", "45"],
                                            "reflectance_s": [list(r) for r in zero], "reflectance_p": [list(r) for r in zero],
                                            "transmittance_s": [list(r) for r in t_s], "transmittance_p": [list(r) for r in t_p],
                                            "status": "DECLARED"})


def _project(polarization: dict, *, split_front: bool = False, weak_y_polarizer: bool = False) -> dict:
    doc = _lens_project(_hyperbolic(8.0, 40.0, WL), waist_mm=1.0)
    # Inline prescription glass is synthetic here: isolate polarization and coating pupils from bulk loss.
    for material in doc["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    doc["components"][0]["source_model"]["polarization"] = polarization
    surfaces = doc["optical_models"][1]["model"]["surfaces"]
    surfaces[1]["coating_id"] = "SYN"
    if split_front:  # (both branches of the front surface traced: the image plane receives a child path of each component)
        surfaces[0]["traced"] = "TRANSMITTED_AND_REFLECTED"
    doc.setdefault("model_coatings", []).append(_coating())
    if weak_y_polarizer:  # (y passes a quarter of what x passes: unequal transmitted powers)
        doc["components"].append(declared_component("P", "polarizer", "pol", position=(-6.5, 0.0, 0.0), quaternion=_turned(0.0)))
        doc["optical_models"].append(polarizer(None, transmittance="1", extinction="4"))
    return doc


def _psf(results) -> tuple[np.ndarray, float]:
    """The point spread function (1/mm^2) of the run's line on a fixed grid, by the analysis's own field at a fixed launch disc,
    and the power the run delivers to the image plane."""

    request = im.parse_request({"launch_grid": GRID, "launch_radius_beam_radii": LAUNCH_RADII})
    preferred = {}
    for arrival in results._observation("image").arrivals:
        preferred[(arrival.beam.source_id, arrival.beam.wavelength_nm)] = arrival.beam.path_id
    rays = im._trace(results, "image", "SRC", preferred, GRID, LAUNCH_RADII, None)[WL]
    level = im._Level(rays, request, results._observation("image").plane)
    a = np.linspace(-HALF_UM, HALF_UM, SAMPLES) * 1e-3
    power = sum(a_.beam.power_W for a_ in results._observation("image").arrivals)
    return level.psf_map(a), power


def _mtf(psf: np.ndarray, frequencies: np.ndarray, axis: int) -> np.ndarray:
    """|OTF| along one map axis: the Fourier transform of the line-spread function, normalized at zero frequency."""

    a = np.linspace(-HALF_UM, HALF_UM, SAMPLES) * 1e-3
    lsf = psf.sum(axis=1 - axis)
    otf = np.array([np.sum(lsf * np.exp(-2j * np.pi * f * a)) for f in frequencies])
    return np.abs(otf) / np.sum(lsf)


@pytest.fixture(scope="module")
def singles():
    return {name: _psf(load_run(run(_project(pol)))) for name, pol in (("x", X), ("y", Y))}


def _reference(parts: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    """The incoherent sum of normalized component PSFs, each by its share of the line times its transmitted power."""

    weights = np.array([share * power for _psf_, power, share in parts])
    weights = weights / weights.sum()
    return sum(w * (psf / psf.sum()) for w, (psf, _p, _s) in zip(weights, parts))


def test_unpolarized_light_adds_the_components_point_spread_functions(singles):
    (px, wx), (py, wy) = singles["x"], singles["y"]
    assert wx == pytest.approx(wy, rel=1e-12)  # (equal transmitted powers: equal weights)
    psf, _ = _psf(load_run(run(_project({"kind": "UNPOLARIZED"}))))
    reference = _reference([(px, wx, 0.5), (py, wy, 0.5)])
    measured = psf / psf.sum()
    assert np.abs(measured - reference).max() <= 1e-3 * reference.max()
    assert measured.max() == pytest.approx(reference.max(), rel=1e-3)  # (a single pupil of the summed irradiance: +20.7 %)
    # the components alone differ strongly: the test resolves the composition
    assert np.abs(px / px.sum() - py / py.sum()).max() > 0.2 * reference.max()


def test_unequal_shares_and_transmitted_powers_weight_the_components():
    singles = {name: _psf(load_run(run(_project(pol, weak_y_polarizer=True)))) for name, pol in (("x", X), ("y", Y))}
    (px, wx), (py, wy) = singles["x"], singles["y"]
    assert wy == pytest.approx(0.25 * wx, rel=1e-6)
    partial = {"kind": "PARTIAL", "degree_of_polarization": 0.5, "polarized": X}  # shares 0.75 (x) / 0.25 (y)
    results = load_run(run(_project(partial, weak_y_polarizer=True)))
    psf, power = _psf(results)
    assert power == pytest.approx(0.75 * wx + 0.25 * wy, rel=1e-9)
    reference = _reference([(px, wx, 0.75), (py, wy, 0.25)])
    assert np.abs(psf / psf.sum() - reference).max() <= 1e-3 * reference.max()


def test_the_documents_mtf_strehl_and_encircled_energy_are_those_of_the_summed_psf(singles):
    (px, wx), (py, wy) = singles["x"], singles["y"]
    reference = _reference([(px, wx, 0.5), (py, wy, 0.5)])
    doc = load_run(run(_project({"kind": "UNPOLARIZED"}))).imaging("image", {"launch_grid": GRID, "launch_radius_beam_radii": LAUNCH_RADII})
    line = _line(doc, "SRC#")
    f = np.array(doc.arrays[line + "frequencies_cyc_per_mm"], dtype=float)
    keep = [i for i, v in enumerate(doc.arrays[line + "mtf_tangential"]) if v is not None and f[i] < 150.0]
    engine = (np.array([doc.arrays[line + "mtf_tangential"][i] for i in keep]), np.array([doc.arrays[line + "mtf_sagittal"][i] for i in keep]))
    along, across = _mtf(reference, f[keep], 0), _mtf(reference, f[keep], 1)
    # the plane's tangential axis is one of the map's axes: the pair matches the reference pair in either order
    best = min(max(np.abs(engine[0] - along).max(), np.abs(engine[1] - across).max()),
               max(np.abs(engine[0] - across).max(), np.abs(engine[1] - along).max()))
    assert best < 0.01, best
    # at half the cutoff the components' summed PSF gives 0.352; one pupil of their summed irradiance would give 0.391
    half = int(np.argmin(np.abs(f[keep] - 0.5 / (WL * 1e-6 * 8.0))))
    assert along[half] == pytest.approx(0.3518, abs=0.005) and engine[0][half] < 0.37
    assert doc.quantity(line + "strehl_ratio").value == pytest.approx(1.0, abs=1e-3)  # (no phase error: peak over the same rays' peak)
    # the 80 % encircled-energy radius of the reference map about its peak
    a = np.linspace(-HALF_UM, HALF_UM, SAMPLES)
    r = np.hypot(*np.meshgrid(a, a, indexing="ij"))
    order = np.argsort(r.ravel())
    cumulative = np.cumsum(reference.ravel()[order]) / reference.sum()
    r80 = r.ravel()[order][np.searchsorted(cumulative, 0.8)]
    assert doc.quantity(line + "encircled_radius_80").value == pytest.approx(r80, abs=2.0 * (a[1] - a[0]))


def test_components_are_associated_by_route_among_traced_branches():
    """The front surface's reflections are traced as branches of each component (``SRC#0.<k>`` / ``SRC#0:p1.<k>``, back toward
    the source): the image plane's two arrivals are the components' transmitted routes, which are found and added, while the
    sibling branches of either component are not taken for its partner."""

    singles = {name: _psf(load_run(run(_project(pol, split_front=True)))) for name, pol in (("x", X), ("y", Y))}
    results = load_run(run(_project({"kind": "UNPOLARIZED"}, split_front=True)))
    ids = sorted(a.beam.path_id for a in results._observation("image").arrivals)
    assert ids == ["SRC#0", "SRC#0:p1"]
    branches = {p.path_id for p in results.result.paths}
    assert any(b.startswith("SRC#0.") for b in branches) and any(b.startswith("SRC#0:p1.") for b in branches)
    psf, _ = _psf(results)
    (px, wx), (py, wy) = singles["x"], singles["y"]
    reference = _reference([(px, wx, 0.5), (py, wy, 0.5)])
    assert np.abs(psf / psf.sum() - reference).max() <= 1e-3 * reference.max()
