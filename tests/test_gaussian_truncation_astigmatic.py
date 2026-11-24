"""Gaussian aperture truncation of strongly astigmatic and turned beams: accuracy against independent references, a cost
that does not grow with the astigmatism, and the libm-free error function behind it.

A cylindrical lens focuses a beam into a line (aspect ratio 100:1 and more). Sizing the rectangle quadrature by the
narrowest Gaussian scale on both axes made one run of a design with the catalog's rectangular cylindrical lens take
10 minutes (or fail allocating 7 GiB); the polar sums of the circle integral, exact for a nearly round footprint, lost 17 %
of such a beam inside a 20 mm disk.

References (test only, scipy): a rectangle met by a beam with the rectangle's axes is the product of two ``erf``
intervals; a turned rectangle is an adaptive quadrature over ``u`` of the closed-form integral over ``v`` (the order of
integration fixed here, whatever the Engine chooses); a disk is an adaptive quadrature, in the beam's own axes, over the
disk's angular parameter ``p = pc + R sin(phi)`` of the closed-form chord integral.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erf, erfc

from os_optics.config import NumericalConfig
from os_optics.math.exact_erf import ERFC_SERIES_LIMIT, erf_interval
from os_optics.math.exact_erf import erfc as exact_erfc
from os_optics.optics.gaussian import gauss_legendre_rule, initial_state, transmission_fraction
from os_optics.optics.gaussian.truncation import CIRCLE_POLAR_MAX_ASPECT
from os_optics.scene.model import ApertureG

pytestmark = pytest.mark.golden

CFG = NumericalConfig()
RULE = gauss_legendre_rule(CFG.aperture_quad_sector_nodes)
U, V = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
W_LONG = 0.5
#: The line focus met by the probe of OS-OPT-LENS-CYL-R12X10-F40-...: 0.5 mm x 4.593 um (aspect 109).
LINE_FOCUS = (0.5, 0.004593)


def _beam(w_x: float, w_y: float, theta_deg: float = 0.0):
    t = math.radians(theta_deg)
    return initial_state(wavelength_nm=633.0, n=1.0, w0x=w_x, w0y=w_y, z_waist=0.0, m2x=1.0, m2y=1.0,
                         bx=np.array([math.cos(t), math.sin(t), 0.0]), by=np.array([-math.sin(t), math.cos(t), 0.0]), cfg=CFG)


def _fraction(beam, aperture: ApertureG, cfg: NumericalConfig = CFG, rule=RULE) -> float:
    return transmission_fraction(beam, aperture, (0.0, 0.0), U, V, 1.0, cfg, rule)


def _floor(t: float) -> float:
    return 1.0 if 1.0 - t <= CFG.cross_runtime_abs_tol else t


def _rectangle_reference(w_x, w_y, theta_deg, centre, half_u, half_v) -> float:
    t = math.radians(theta_deg)
    bx, by = (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))
    a = bx[0] * bx[0] / w_x**2 + by[0] * by[0] / w_y**2
    b = bx[0] * bx[1] / w_x**2 + by[0] * by[1] / w_y**2
    c = bx[1] * bx[1] / w_x**2 + by[1] * by[1] / w_y**2
    u_lo, u_hi, v_lo, v_hi = centre[0] - half_u, centre[0] + half_u, centre[1] - half_v, centre[1] + half_v
    if abs(b) <= 1e-15 * math.sqrt(a * c):
        p_u = 0.5 * (erf(math.sqrt(2 * a) * u_hi) - erf(math.sqrt(2 * a) * u_lo))
        p_v = 0.5 * (erf(math.sqrt(2 * c) * v_hi) - erf(math.sqrt(2 * c) * v_lo))
        return p_u * p_v
    m = 1.0 / (w_x * w_x * w_y * w_y * c)  # det S / c; det S = 1 / (w_x w_y)^2 for in-plane beam axes
    slope = b / c

    def integrand(u):
        p = 0.5 * (erf(math.sqrt(2 * c) * (v_hi + slope * u)) - erf(math.sqrt(2 * c) * (v_lo + slope * u)))
        return math.sqrt(2 * m / math.pi) * math.exp(-2 * m * u * u) * p

    reach = 8.0 / math.sqrt(m)
    lo, hi = max(u_lo, -reach), min(u_hi, reach)
    if hi <= lo:
        return 0.0
    edge_scale = 1.0 / (math.sqrt(c) * abs(slope))  # where v*(u) crosses an edge, the inner integral changes over this
    points = sorted({x for x in (0.0, *(-v / slope + j * edge_scale for v in (v_lo, v_hi) for j in range(-8, 9))) if lo < x < hi})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _ = quad(integrand, lo, hi, points=points or None, epsabs=1e-17, epsrel=1e-14, limit=4000)
    return value


def _disk_reference(w_x, w_y, theta_deg, centre, radius) -> float:
    t = math.radians(theta_deg)
    pc = centre[0] * math.cos(t) + centre[1] * math.sin(t)  # disk centre in the beam's axes
    qc = -centre[0] * math.sin(t) + centre[1] * math.cos(t)

    def integrand(phi):
        p, h = pc + radius * math.sin(phi), radius * math.cos(phi)
        chord = 0.5 * (erf(math.sqrt(2.0) * (qc + h) / w_y) - erf(math.sqrt(2.0) * (qc - h) / w_y))
        return h * math.sqrt(2.0 / math.pi) / w_x * math.exp(-2.0 * p * p / (w_x * w_x)) * chord

    points = []
    if abs(qc) < radius:  # a chord end crosses the beam axis: the chord integral changes over w_y / (R |sin(phi)|)
        end = math.acos(abs(qc) / radius)
        width = w_y / (radius * max(math.sin(end), 1e-12))
        points += [s * end + j * width for s in (1.0, -1.0) for j in range(-8, 9)]
    if abs(pc) < radius:
        points.append(math.asin(-pc / radius))
    for step in range(1, 40):  # the ends, where a chord is shorter than a few beam radii
        x = math.pi / 2 - step * min(w_y / radius, 0.02)
        points += [x, -x]
    points = sorted({x for x in points if -math.pi / 2 < x < math.pi / 2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _ = quad(integrand, -math.pi / 2, math.pi / 2, points=points, epsabs=1e-17, epsrel=1e-14, limit=8000)
    return value


# ---------------------------------------------------------------------------------------------------------------------
# The error function
# ---------------------------------------------------------------------------------------------------------------------
def test_erfc_matches_an_independent_implementation():
    x = np.concatenate([np.linspace(0.0, 27.0, 200001), np.linspace(ERFC_SERIES_LIMIT - 0.01, ERFC_SERIES_LIMIT + 0.01, 2001)])
    got, want = exact_erfc(x), erfc(x)
    assert np.max(np.abs(got - want)) < 2e-15
    tail = (x >= ERFC_SERIES_LIMIT) & (want > 1e-300)
    assert np.max(np.abs(got[tail] - want[tail]) / want[tail]) < 4e-15  # far tails keep their relative accuracy
    assert exact_erfc(np.array([0.0]))[0] == 1.0 and exact_erfc(np.array([60.0, 1e300]))[0] == 0.0


def test_erf_interval_matches_an_independent_implementation():
    rng = np.random.default_rng(11)
    a = rng.uniform(-9.0, 9.0, 100000)
    b = a + rng.exponential(1.0, a.size)
    want = np.where(a >= 0, 0.5 * (erfc(a) - erfc(b)), np.where(b <= 0, 0.5 * (erfc(-b) - erfc(-a)), 0.5 * (erf(b) - erf(a))))
    assert np.max(np.abs(erf_interval(a, b) - want)) < 2e-15
    far = (a > 4.0) & (b - a > 0.5)  # one tail, interval not narrow: relative accuracy
    assert np.max(np.abs(erf_interval(a[far], b[far]) - want[far]) / want[far]) < 1e-13


# ---------------------------------------------------------------------------------------------------------------------
# Rectangles
# ---------------------------------------------------------------------------------------------------------------------
RECTANGLES = [((0.0, 0.0), 5.4, 4.5), ((0.3, -0.2), 0.7, 0.4), ((1.2, 0.0), 0.5, 2.0), ((0.0, 0.004), 3.0, 0.005), ((0.0, 0.0), 0.002, 3.0)]


@pytest.mark.parametrize("aspect", [1.0, 10.0, 109.0, 1e3, 1e4])
@pytest.mark.parametrize("theta_deg", [0.0, 90.0])
@pytest.mark.parametrize(("centre", "half_u", "half_v"), RECTANGLES)
def test_rectangle_with_the_beams_axes_is_the_erf_product(aspect, theta_deg, centre, half_u, half_v):
    w_x, w_y = W_LONG, W_LONG / aspect
    got = _fraction(_beam(w_x, w_y, theta_deg), ApertureG("rectangle", half_u, half_v, centre[0], centre[1], 0.0))
    assert got == pytest.approx(_floor(_rectangle_reference(w_x, w_y, theta_deg, centre, half_u, half_v)), abs=1e-14)


@pytest.mark.parametrize("aspect", [2.0, 10.0, 109.0, 1e3, 1e4])
@pytest.mark.parametrize("theta_deg", [1e-3, 0.5, 7.0, 30.0, 45.0, 123.0])
@pytest.mark.parametrize(("centre", "half_u", "half_v"), RECTANGLES)
def test_turned_astigmatic_beam_on_a_rectangle(aspect, theta_deg, centre, half_u, half_v):
    w_x, w_y = W_LONG, W_LONG / aspect
    got = _fraction(_beam(w_x, w_y, theta_deg), ApertureG("rectangle", half_u, half_v, centre[0], centre[1], 0.0))
    assert got == pytest.approx(_floor(_rectangle_reference(w_x, w_y, theta_deg, centre, half_u, half_v)), abs=1e-13)


# ---------------------------------------------------------------------------------------------------------------------
# Circles
# ---------------------------------------------------------------------------------------------------------------------
DISKS = [((0.0, 0.0), 20.0), ((0.0, 0.0), 0.6), ((0.2, 0.001), 0.4), ((1.0, 0.0), 0.6), ((0.3, 0.0), 0.35), ((0.7, 0.7), 0.5),
         ((0.0, 0.0), 0.003), ((0.0, 0.02), 0.01)]


@pytest.mark.parametrize("aspect", [1.0, 1.9, 2.1, 3.0, 10.0, 109.0, 1e3, 1e4])
@pytest.mark.parametrize("theta_deg", [0.0, 33.0, 90.0])
@pytest.mark.parametrize(("centre", "radius"), DISKS)
def test_astigmatic_beam_on_a_disk(aspect, theta_deg, centre, radius):
    """Both sides of the switch from the polar form (aspect <= 2) to the principal frame."""
    w_x, w_y = W_LONG, W_LONG / aspect
    got = _fraction(_beam(w_x, w_y, theta_deg), ApertureG("circle", radius, 0.0, centre[0], centre[1], 0.0))
    assert got == pytest.approx(_floor(_disk_reference(w_x, w_y, theta_deg, centre, radius)), abs=1e-13)
    assert CIRCLE_POLAR_MAX_ASPECT == 2.0


@pytest.mark.parametrize("aspect", [10.0, 109.0])
def test_astigmatic_beam_through_an_annulus(aspect):
    w_x, w_y = W_LONG, W_LONG / aspect
    beam = _beam(w_x, w_y, 20.0)
    got = _fraction(beam, ApertureG("circle", 0.9, 0.0, 0.05, -0.02, 0.25))
    want = _disk_reference(w_x, w_y, 20.0, (0.05, -0.02), 0.9) - _disk_reference(w_x, w_y, 20.0, (0.05, -0.02), 0.25)
    assert got == pytest.approx(want, abs=1e-13)


def test_a_line_focus_inside_a_large_aperture_loses_nothing():
    """Regression: the polar sums gave 0.826 for this beam centred in a 20 mm disk (and 0.127 at 1000:1)."""
    for w_x, w_y in (LINE_FOCUS, (0.5, 0.0005)):
        for theta_deg in (0.0, 30.0):
            beam = _beam(w_x, w_y, theta_deg)
            assert _fraction(beam, ApertureG("circle", 20.0, 0.0, 0.0, 0.0, 0.0)) == 1.0
            assert _fraction(beam, ApertureG("rectangle", 5.4, 4.5, 0.0, 0.0, 0.0)) == 1.0


# ---------------------------------------------------------------------------------------------------------------------
# Convergence and cost
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0.0, 7.0, 45.0])
def test_astigmatic_quadrature_is_converged(theta_deg):
    """Doubling the Gauss-Legendre nodes and widening the support to 8 beam radii changes nothing above rounding."""
    fine = NumericalConfig(aperture_quad_sector_nodes=96, aperture_quad_extent_sigmas=8.0)
    fine_rule = gauss_legendre_rule(96)
    for aspect in (10.0, 109.0, 1e4):
        beam = _beam(W_LONG, W_LONG / aspect, theta_deg)
        for aperture in (ApertureG("rectangle", 3.0, 0.005, 0.0, 0.004, 0.0), ApertureG("rectangle", 0.7, 0.4, 0.3, -0.2, 0.0),
                         ApertureG("circle", 0.4, 0.0, 0.2, 0.001, 0.0), ApertureG("circle", 0.6, 0.0, 1.0, 0.0, 0.0)):
            assert _fraction(beam, aperture) == pytest.approx(_fraction(beam, aperture, fine, fine_rule), abs=1e-14)


def test_the_cost_does_not_grow_with_the_astigmatism():
    """The line focus of the cylindrical-lens case on its 10.8 x 9 mm rectangle took minutes per call (a 31392 x 31392
    grid); every call here is a few milliseconds."""
    cases = [(w_x, w_y, theta) for w_x, w_y in (LINE_FOCUS, (0.5, 0.5e-4), (0.5, 0.25)) for theta in (0.0, 0.5, 45.0)]
    apertures = [ApertureG("rectangle", 5.4, 4.5, 0.0, 0.0, 0.0), ApertureG("rectangle", 4.5, 5.4, 0.3, 0.001, 0.0),
                 ApertureG("circle", 20.0, 0.0, 0.0, 0.0, 0.0), ApertureG("circle", 0.4, 0.0, 0.2, 0.001, 0.25)]
    beams = [_beam(*case) for case in cases]
    start = time.perf_counter()
    for beam in beams:
        for aperture in apertures:
            _fraction(beam, aperture)
    assert time.perf_counter() - start < 5.0  # 36 calls; about 0.1 s on a development machine


CYLINDER_SKU = "OS-OPT-LENS-CYL-R12X10-F40-T2P6-R20P7-NBK7-UNC-W587P6"
ALONG_X = (0.5, 0.5, 0.5, 0.5)  # local +Z -> design +X, local +Y -> design +Z
LINE_FOCUS_X_MM = 41.7  # the waist of the focused meridian, 41.7 mm behind the lens's origin


def line_focus_project() -> dict:
    """The release probe's one-element design of the catalog's rectangular cylindrical lens (a collimated 0.5 mm beam along
    +X; the lens's uncoated, rated faces send untraced light that focuses to a line on its rectangular faces) and three
    planes at the line focus (0.5 mm x ~17 um): an off-centre 20 um pinhole (the disk integral of a ~30:1 footprint, in the
    beam's principal frame), a 0.6 x 0.02 mm slit turned 30 deg about the beam (the rectangle integral of a turned
    astigmatic footprint) and a 0.3 mm pinhole on the line (beam centre inside, chord ends crossing the beam)."""
    from decimal import Decimal

    from catalog_store_helpers import REAL_CATALOG
    from local_helpers import quat_multiply
    from os_optics.local import load_public_catalog

    catalog = load_public_catalog(REAL_CATALOG)
    product = catalog.get(CYLINDER_SKU)
    band = product.simulation_models[0].valid_band_nm
    wavelength = float((Decimal(band.min_nm) + Decimal(band.max_nm)) / 2)  # the probe's line: the model band's centre
    half = math.radians(30.0) / 2
    turned = [float(format(v, ".15g")) for v in quat_multiply((math.cos(half), math.sin(half), 0.0, 0.0), ALONG_X)]

    def plane(observation_id: str, x: float, aperture: dict, quaternion=ALONG_X) -> dict:
        return {"observation_id": observation_id, "pose": {"position_mm": [x, 0.0, 0.0], "quaternion_wxyz": list(quaternion)},
                "aperture": aperture, "absorbing": False}

    source = {"instance_id": "S1", "optical_role": "source",
              "custom_part": {"custom_part_key": "probe-source", "description": "Probe Gaussian source."},
              "nominal_pose": {"position_mm": [-100.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)},
              "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": wavelength, "power_w": 0.001, "waist_radius_mm": 0.5,
                               "waist_position_mm": 100.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}
    lens = {"instance_id": "E1", "optical_role": "lens", "product": {"optical_sku": CYLINDER_SKU, "catalog_version": catalog.catalog_version},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}
    observations = [plane("P1", 22.6, {"shape": "circle", "radius_mm": 20.0}),
                    plane("pinhole", LINE_FOCUS_X_MM, {"shape": "circle", "radius_mm": 0.02, "center_offset_mm": [0.1, 0.005]}),
                    plane("slit", LINE_FOCUS_X_MM + 0.1, {"shape": "rectangle", "half_width_mm": 0.3, "half_height_mm": 0.01}, turned),
                    plane("wide_pinhole", LINE_FOCUS_X_MM + 0.2, {"shape": "circle", "radius_mm": 0.3})]
    targets = [{"target_id": f"{o['observation_id']}_captured", "metric": "active_area_power_fraction", "observation_id": o["observation_id"],
                "minimum": 0.0, "unit": "1"} for o in observations]
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": [source, lens], "observations": observations,
            "targets": targets}


def _load_probe():
    script = Path(__file__).resolve().parents[1] / "scripts" / "optical_simulation_probe.py"
    spec = importlib.util.spec_from_file_location("optical_simulation_probe_fx7", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_rectangular_cylindrical_lens_probe_runs_in_seconds():
    """The release probe's one-element design of the catalog's rectangular cylindrical lens (collimated beam along +X;
    its uncoated, rated faces send untraced light that focuses to a line on the rectangular faces): 602 s before."""
    from catalog_store_helpers import REAL_CATALOG
    from os_optics.local import load_public_catalog

    catalog = load_public_catalog(REAL_CATALOG)
    product = next((p for p in catalog.products if p.optical_sku == CYLINDER_SKU), None)
    if product is None:
        pytest.skip(f"{CYLINDER_SKU} is not in the seed catalog release")
    probe = _load_probe()
    start = time.perf_counter()
    outcome = probe.probe_one(product, catalog)
    elapsed = time.perf_counter() - start
    # Its catalog glass states no bulk loss: the line-focus geometry is in scope, while transmitted power remains
    # bounded/unknown rather than silently assuming zero absorption. The probe must still complete in seconds.
    assert outcome == {"status": "IN_SCOPE", "run_status": "UNKNOWN", "wavelengths": 1,
                       "power_evidence": "UNKNOWN_BULK_LOSS", "codes": ["BULK_LOSS_NOT_ESTABLISHED"]}
    assert elapsed < 10.0  # about 0.2 s on a development machine


def _plane_axes(quaternion) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = quaternion
    u = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    v = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    return u, v


def test_line_focus_capture_fractions_match_the_references():
    """End to end: the fractions of the line focus captured by the pinholes and the turned slit (the Engine's beam state at
    each plane, expressed in that plane's axes) equal the independent references; the run with the whole bench turned about
    an oblique axis captures the same fractions."""
    from catalog_store_helpers import REAL_CATALOG
    from os_optics.local import load_public_catalog, run_optical_project
    from test_packaging_and_pyodide import _turned

    catalog = load_public_catalog(REAL_CATALOG)
    document = line_focus_project()
    run = run_optical_project(document, catalog=catalog)
    assert run.status.value == "UNKNOWN"  # The cylinder's catalog glass states no bulk absorption.
    assert {i.code.value for i in run.result.issues if i.status.value not in ("PASS", "NOT_APPLICABLE")} == {
        "BULK_LOSS_NOT_ESTABLISHED"}
    assert "INCOMPLETE_ENVELOPE" in {i.code.value for i in run.result.issues if i.status.value == "NOT_APPLICABLE"}
    poses = {o["observation_id"]: o["pose"]["quaternion_wxyz"] for o in document["observations"]}
    references = {"pinhole": lambda wx, wy, th: _disk_reference(wx, wy, th, (0.1, 0.005), 0.02),
                  "slit": lambda wx, wy, th: _rectangle_reference(wx, wy, th, (0.0, 0.0), 0.3, 0.01),
                  "wide_pinhole": lambda wx, wy, th: _disk_reference(wx, wy, th, (0.0, 0.0), 0.3),
                  "P1": lambda wx, wy, th: _floor(_disk_reference(wx, wy, th, (0.0, 0.0), 20.0))}
    captured = {}
    for observation in run.result.observations:
        (arrival,) = observation.arrivals
        assert arrival.beam.power_W is None and arrival.power_upper_bound_W > 0
        beam = arrival.beam.gaussian
        u, v = _plane_axes(poses[observation.observation_id])
        e1 = np.array(beam.e1)
        theta = math.degrees(math.atan2(float(e1 @ v), float(e1 @ u)))
        want = references[observation.observation_id](beam.beam_radius_x_mm, beam.beam_radius_y_mm, theta)
        assert arrival.active_area_power_fraction == pytest.approx(want, abs=1e-12), observation.observation_id
        captured[observation.observation_id] = arrival.active_area_power_fraction
    assert 0.03 < captured["pinhole"] < 0.05 and 0.05 < captured["slit"] < 0.08 and 0.7 < captured["wide_pinhole"] < 0.8
    turned = run_optical_project(_turned(document), catalog=catalog)
    assert turned.status.value == "UNKNOWN"
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in turned.result.issues}
    for observation in turned.result.observations:
        assert observation.arrivals[0].active_area_power_fraction == pytest.approx(captured[observation.observation_id], abs=1e-12)
