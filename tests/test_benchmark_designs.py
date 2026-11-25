"""Benchmark optical designs built from the repository's real seed catalog release, checked against independent physics.

Each benchmark is a small multi-component design a customer (or their Agent) would build: products are chosen by catalog
query only (category / form / band / published specification values -- never a hard-coded optical SKU, and a missing
product family is a failure, not a skip), placed by their published frames (transmissive elements: local +Z along the
beam; face-normal elements: local +Z = the coated face's normal toward the incoming light; ``PORTS`` elements: the ``IN``
port's direction along the incoming beam), simulated with the public local workflow (``run_optical_project``) and read
back through the results layer (``os_optics.results.load_run``) the way a customer reads a run. Every expected
number is computed here with plain ``math`` / NumPy from the published catalog values (coating s / p values, spectra,
Sellmeier coefficients, surface prescriptions, paraxial blocks, specifications) -- never with the runtime's own compile,
paraxial or tracing helpers.
"""

from __future__ import annotations

import csv
import io
import json
import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics.cli import main as cli_main
from os_optics.local import load_public_catalog, run_optical_project
from os_optics.results import load_run

from catalog_store_helpers import REAL_CATALOG
from local_helpers import NM, POWER_W, element, free, plane, project, q_through, rayleigh_range, source

CATALOG = load_public_catalog(REAL_CATALOG)
VERSION = CATALOG.catalog_version
DOCUMENT = json.loads(REAL_CATALOG.read_bytes())
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ catalog queries


def _model(product):
    return product.simulation_models[0]


def _band(product) -> tuple[float, float]:
    band = _model(product).valid_band_nm
    return float(band.min_nm), float(band.max_nm)


def _covers(product, *wavelengths: float) -> bool:
    low, high = _band(product)
    return all(low <= w <= high for w in wavelengths)


def _of_form(form: str) -> list:
    found = sorted((p for p in CATALOG.products if p.form.value == form), key=lambda p: p.optical_sku)
    # a failure, not a skip: a generator regression that dropped a product family must not pass silently
    assert found, f"the seed release has no {form} products"
    return found


def _spec(product, name: str) -> float | None:
    value = getattr(product.specifications, name, None)
    return None if value is None else float(value.value)


def _coating(coating_id: str):
    coating = CATALOG.coating(coating_id)
    assert coating is not None, coating_id
    return coating


def _band_values(coating_id: str, wavelength_nm: float):
    """The published band of a band coating that holds ``wavelength_nm``."""

    coating = _coating(coating_id)
    band = next(b for b in coating.bands if float(b.min_nm) <= wavelength_nm <= float(b.max_nm))
    return band


def _transmittance(coating_id: str, wavelength_nm: float) -> float:
    return float(_band_values(coating_id, wavelength_nm).transmittance)


def _reflectance(coating_id: str, wavelength_nm: float) -> float:
    return float(_band_values(coating_id, wavelength_nm).reflectance)


def _aoi_covers(coating_id: str, aoi_deg: float) -> bool:
    aoi = _coating(coating_id).valid_aoi_deg
    return float(aoi.min_deg) <= aoi_deg <= float(aoi.max_deg)


def _surface(product, surface_id: str):
    return next(s for s in _model(product).surfaces if s.surface_id == surface_id)


def _sellmeier(material_id: str, wavelength_nm: float) -> float:
    material = next(m for m in DOCUMENT["materials"] if m["material_id"] == material_id)
    assert material["dispersion"]["kind"] == "SELLMEIER", material_id
    b = [float(v) for v in material["dispersion"]["b"]]
    c = [float(v) for v in material["dispersion"]["c_um2"]]
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def _fresnel_normal(n: float) -> float:
    return ((n - 1.0) / (n + 1.0)) ** 2


def _sellmeier_glass(product) -> bool:
    return all(CATALOG.material(m).dispersion.kind == "SELLMEIER" for m in _model(product).media)


#: common laser lines a benchmark picks its wavelength from (the first one inside the products' bands)
LASER_LINES_NM = (632.8, 780.0, 1064.0, 1550.0)


def _laser_line(*products) -> float:
    line = next((w for w in LASER_LINES_NM if all(_covers(p, w) for p in products)), None)
    assert line is not None, f"no common laser line inside the bands of {[p.optical_sku for p in products]}"
    return line


# --------------------------------------------------------------------------------------------- frames and poses


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _triad(a, b) -> np.ndarray:
    """Orthonormal columns (a, b', a x b') from a primary vector and a secondary one (b' = b made perpendicular to a)."""

    e1 = _unit(a)
    e2 = np.asarray(b, dtype=float) - float(np.dot(b, e1)) * e1
    e2 = _unit(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def _orient(local_a, global_a, local_b, global_b) -> np.ndarray:
    """The rotation R (local -> design) taking local_a onto global_a and local_b's direction (made perpendicular) onto
    global_b's; the two pairs must make the same angle."""

    la, lb, ga, gb = (np.asarray(v, dtype=float) for v in (local_a, local_b, global_a, global_b))
    assert abs(float(np.dot(_unit(la), _unit(lb))) - float(np.dot(_unit(ga), _unit(gb)))) < 1e-9, "inconsistent port angles"
    return _triad(ga, gb) @ _triad(la, lb).T


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""

    m = rotation
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q, dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def _rotation_about(axis, degrees: float) -> np.ndarray:
    k = _unit(axis)
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return c * np.eye(3) + s * cross + (1.0 - c) * np.outer(k, k)


def _port(product, port_id: str) -> tuple[np.ndarray, np.ndarray]:
    ports = product.frame.ports
    assert ports, f"{product.optical_sku} is not placed by ports"
    port = next(p for p in ports if p.port_id == port_id)
    return np.array([float(v) for v in port.point_mm]), np.array([float(v) for v in port.direction])


def _place_by_ports(product, beam_point, beam_direction, out_port: str, out_direction, *, local_secondary=None, global_secondary=None):
    """Pose (position, quaternion, rotation) of a ``PORTS`` element whose ``IN`` port takes the beam travelling along
    ``beam_direction`` through ``beam_point`` and whose ``out_port`` leaves along ``out_direction`` (a retro element whose
    output is antiparallel to its input is oriented by an extra local / design vector pair instead)."""

    in_point, in_direction = _port(product, "IN")
    _, local_out = _port(product, out_port)
    if local_secondary is None:
        rotation = _orient(in_direction, beam_direction, local_out, out_direction)
    else:
        rotation = _orient(in_direction, beam_direction, local_secondary, global_secondary)
    assert np.allclose(rotation @ local_out, _unit(out_direction), atol=1e-9), "the output port does not leave along the requested direction"
    position = np.asarray(beam_point, dtype=float) - rotation @ in_point
    return tuple(float(v) for v in position), _quaternion(rotation), rotation


def _out_point(product, out_port: str, position, rotation) -> np.ndarray:
    point, _ = _port(product, out_port)
    return np.asarray(position, dtype=float) + rotation @ point


def _axis_pose(direction, up=Z) -> tuple[float, float, float, float]:
    """Quaternion of a transmissive element / plane / source whose local +Z points along ``direction`` (local +Y as close
    to ``up`` as possible)."""

    return _quaternion(_orient(Z, direction, Y, up))


def _face_pose(normal, up=Z) -> tuple[float, float, float, float]:
    """Quaternion of a face-normal element whose local +Z (the coated face's normal) points along ``normal``."""

    other = up if abs(float(np.dot(_unit(normal), _unit(up)))) < 0.9 else X
    return _quaternion(_orient(Z, normal, Y, np.asarray(other) - float(np.dot(other, _unit(normal))) * _unit(normal)))


# ---------------------------------------------------------------------------------------------- project helpers


def _source(instance_id: str, position, direction=X, *, wavelength_nm: float | None = None, lines=None, waist_mm: float = 1.0,
            polarization_deg: float = 0.0, coherence_length_mm: float | None = None, up=Z) -> dict:
    component = source(instance_id, tuple(position), wavelength_nm=wavelength_nm or 1.0, waist_mm=waist_mm,
                       quaternion=_axis_pose(direction, up), polarization_deg=polarization_deg)
    model = component["source_model"]
    if lines is not None:
        del model["wavelength_nm"], model["power_w"]
        model["spectral_lines"] = [{"wavelength_nm": w, "power_w": p} for w, p in lines]
    if coherence_length_mm is not None:
        model["coherence_length_mm"] = coherence_length_mm
    return component


def _element(instance_id: str, role: str, product, position, quaternion) -> dict:
    return element(instance_id, role, product.optical_sku, tuple(position), tuple(quaternion), catalog_version=VERSION)


def _plane(observation_id: str, position, direction, *, radius_mm: float = 12.5, absorbing: bool = False, up=Z) -> dict:
    observation = plane(observation_id, tuple(float(v) for v in position), _axis_pose(direction, up), radius_mm=radius_mm)
    if absorbing:
        observation["absorbing"] = True
    return observation


def _run(components, observations, targets=None, **kwargs):
    run = run_optical_project(project(components, observations, targets), catalog=CATALOG, **kwargs)
    return run, load_run(run)


def _in_scope(run) -> None:
    assert run.model_scope_issues == (), (run.status, run.model_scope_issues)
    if run.status.value == "UNKNOWN":
        assert "BULK_LOSS_NOT_ESTABLISHED" in {i.code.value for i in run.result.issues}
    else:
        assert run.status.value in ("PASS", "WARNING"), run.status


def _catalog_power(arrival: dict, theoretical_no_bulk_loss_W: float, *, rel: float = 1e-9) -> None:
    """Published faces bound a path, but unstated substrate absorption cannot make its power known."""
    assert arrival["beam"]["power_W"] is None  # this seed's substrate has no published bulk-loss law
    assert arrival["power_upper_bound_W"] == pytest.approx(theoretical_no_bulk_loss_W, rel=rel)


def _arrivals(results, monitor: str) -> list[dict]:
    return results.monitor(monitor)["arrivals"]


def _single(results, monitor: str) -> dict:
    found = _arrivals(results, monitor)
    assert len(found) == 1, [a["lineage"] for a in found]
    return found[0]


def _budget_closes(results) -> dict:
    budget = results.power_budget()
    assert budget["conserved"] is True
    assert sum(budget["total_sinks"].values()) + budget["residual_W"] == pytest.approx(budget["total_input_W"], rel=1e-12)
    assert abs(budget["residual_W"]) <= budget["tolerance_W"]
    return budget


# ================================================================================ 1. PBS cube attenuator (HWP + PBS)


def _attenuator_parts():
    """The first (by SKU) zero-order half-wave plate and polarizing cube splitter that share a wavelength: the plate's
    design wavelength inside the cube's band, the cube publishing polarization-resolved splitter values."""

    for plate in _of_form("HALF_WAVE_PLATE"):
        wavelength = float(_model(plate).design_wavelength_nm)
        if not _covers(plate, wavelength):
            continue
        for cube in _of_form("POLARIZING_CUBE_BEAMSPLITTER"):
            if not _covers(cube, wavelength):
                continue
            band = _band_values(_surface(cube, "S2").coating_id, wavelength)
            if band.transmittance_p is not None and _spec(cube, "extinction_ratio") is not None:
                return plate, cube, wavelength
    pytest.fail("no half-wave plate and polarizing cube splitter of a common wavelength in the seed release")


def _attenuator_run(theta_deg: float):
    plate, cube, wavelength = _attenuator_parts()
    # the source is polarized along design +Y: p for the cube, whose plane of incidence (IN along +X, OUT_R along +Y) is
    # design XY; the plate's fast axis (local +X) starts along design Z (perpendicular to it) and turns by theta about
    # the beam, which turns the polarization by 2 theta
    plate_rotation = _rotation_about(X, theta_deg) @ _orient(Z, X, X, Z)
    cube_position, cube_q, _ = _place_by_ports(cube, (0.0, 0.0, 0.0), X, "OUT_R", Y)
    components = [
        _source("SRC", (-150.0, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=1.0, polarization_deg=0.0),
        _element("HWP", "waveplate", plate, (-80.0, 0.0, 0.0), _quaternion(plate_rotation)),
        _element("PBS", "splitter", cube, cube_position, cube_q),
    ]
    observations = [_plane("T", (120.0, 0.0, 0.0), X, absorbing=True), _plane("R", (0.0, 120.0, 0.0), Y, absorbing=True)]
    return _run(components, observations)


def test_pbs_attenuator_follows_the_half_wave_plate_law_with_the_published_s_p_values():
    """Linear light through a half-wave plate turned by theta reaches the cube with its polarization turned by 2 theta:
    the transmitted port carries T_p cos^2(2 theta) + T_s sin^2(2 theta), the reflected one R_p cos^2 + R_s sin^2 (the
    cube's published s / p values), times the plate's published transmittance and the cube faces' AR transmittance."""

    plate, cube, wavelength = _attenuator_parts()
    split = _band_values(_surface(cube, "S2").coating_id, wavelength)
    tp, ts, rp, rs = (float(v) for v in (split.transmittance_p, split.transmittance_s, split.reflectance_p, split.reflectance_s))
    t_plate = float(_model(plate).transmittance.value)
    t_in, t_out_t, t_out_r = (_transmittance(_surface(cube, s).coating_id, wavelength) for s in ("S1", "S3", "S4"))
    powers = {}
    for theta in (0.0, 10.0, 22.5, 30.0, 45.0):
        run, results = _attenuator_run(theta)
        _in_scope(run)
        transmitted, reflected = _single(results, "T"), _single(results, "R")
        c2, s2 = math.cos(math.radians(2 * theta)) ** 2, math.sin(math.radians(2 * theta)) ** 2
        common = POWER_W * t_plate * t_in
        expected = (common * (tp * c2 + ts * s2) * t_out_t, common * (rp * c2 + rs * s2) * t_out_r)
        _catalog_power(transmitted, expected[0])
        _catalog_power(reflected, expected[1])
        powers[theta] = tuple(a["beam"]["power_W"] if a["beam"]["power_W"] is not None else a["power_upper_bound_W"]
                              for a in (transmitted, reflected))
        assert powers[theta][0] / powers[0.0][0] == pytest.approx((tp * c2 + ts * s2) / tp, rel=1e-9)
        assert powers[theta][1] / powers[0.0][1] == pytest.approx((rp * c2 + rs * s2) / rp, rel=1e-9)
        budget = _budget_closes(results)
        if transmitted["beam"]["power_W"] is None:
            assert reflected["beam"]["power_W"] is None and budget["total_sinks"]["BULK_LOSS_UNKNOWN"] > 0
            assert budget["total_sinks"].get("DETECTED", 0.0) == 0.0
        else:
            assert budget["total_sinks"]["DETECTED"] == pytest.approx(sum(powers[theta]), rel=1e-12)
            # The plate's known loss, AR residuals and splitter-coating loss apply to measured-loss materials.
            assert budget["total_sinks"]["ELEMENT_TRANSMISSION_LOSS"] == pytest.approx(POWER_W * (1 - t_plate), rel=1e-9)
            lossless_split = POWER_W * t_plate * t_in * ((tp + rp) * c2 + (ts + rs) * s2)
            assert budget["total_sinks"].get("COATING_ABSORPTION", 0.0) == pytest.approx(POWER_W * t_plate * t_in - lossless_split, abs=1e-15)
        if theta == 45.0:
            # all of the light is s now: the reflected beam is polarized perpendicular to the plane of incidence (design Z)
            state = reflected["polarization_state"]
            assert state["handedness"] == "LINEAR" and abs(state["major_axis"][2]) == pytest.approx(1.0, abs=1e-9)
    # the extinction ratio recovered from theta = 0 (p) and 45 deg (s) at the transmitted port is the cube's published one
    extinction = powers[0.0][0] / powers[45.0][0]
    assert extinction == pytest.approx(tp / ts, rel=1e-9)
    assert extinction == pytest.approx(_spec(cube, "extinction_ratio"), rel=1e-3)


# ================================================================== 2. Mach-Zehnder of two cube splitters and two mirrors

ARM_MM = 60.0
COHERENCE_LENGTH_MM = 1000.0


def _mach_zehnder_parts():
    """A polarization-independent cube splitter of a balanced split (|R - T| <= 0.05) and a flat mirror characterized at
    45 deg, sharing a laser line: the first cube and then the first mirror (by SKU) that do."""

    for wavelength in LASER_LINES_NM:
        cubes = [c for c in _of_form("CUBE_BEAMSPLITTER") if _covers(c, wavelength)
                 and "POLARIZATION_INDEPENDENT_COATING" in [a.value for a in _model(c).approximations]
                 and abs(_reflectance(_surface(c, "S2").coating_id, wavelength) - _transmittance(_surface(c, "S2").coating_id, wavelength)) <= 0.05]
        mirrors = [m for m in _of_form("FLAT_MIRROR") if _covers(m, wavelength) and _aoi_covers(_model(m).surfaces[0].coating_id, 45.0)]
        if cubes and mirrors:
            return cubes[0], mirrors[0], wavelength
    pytest.fail("no balanced cube splitter and 45 deg flat mirror sharing a laser line")


def _mach_zehnder(extra_path_mm: float = 0.0, *, catalog=CATALOG):
    """Cube C1 splits a +X beam into +X (T arm) and +Y (R arm); mirror M1 folds the T arm to +Y, M2 the R arm to +X; cube
    C2, turned so that its OUT_T face takes the R arm and its OUT_R face the T arm, recombines them. ``extra_path_mm``
    lengthens the T arm (M1 moved along its normal by -extra / sqrt 2)."""

    cube, mirror, wavelength = _mach_zehnder_parts()
    c1_position, c1_q, _ = _place_by_ports(cube, (0.0, 0.0, 0.0), X, "OUT_R", Y)
    c1 = np.asarray(c1_position)
    m1_normal, m2_normal = _unit((-1.0, 1.0, 0.0)), _unit((1.0, -1.0, 0.0))
    m1 = c1 + ARM_MM * np.asarray(X) - extra_path_mm / math.sqrt(2.0) * m1_normal
    m2 = c1 + ARM_MM * np.asarray(Y)
    c2 = c1 + ARM_MM * (np.asarray(X) + np.asarray(Y))
    c2_rotation = _orient(Z, (-1.0, 0.0, 0.0), X, (0.0, -1.0, 0.0))  # local +Z -> -X (OUT_T face meets the +X arm), +X -> -Y
    components = [
        _source("SRC", (-80.0, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=1.0, coherence_length_mm=COHERENCE_LENGTH_MM),
        _element("C1", "splitter", cube, c1_position, c1_q),
        _element("M1", "mirror", mirror, m1, _face_pose(m1_normal)),
        _element("M2", "mirror", mirror, m2, _face_pose(m2_normal)),
        _element("C2", "splitter", cube, c2, _quaternion(c2_rotation)),
    ]
    observations = [_plane("BRIGHT", c2 + 60.0 * np.asarray(X), X), _plane("DARK", c2 + 60.0 * np.asarray(Y), Y)]
    run = run_optical_project(project(components, observations), catalog=catalog)
    return run, load_run(run)



def test_closed_return_screen_prunes_impossible_cube_walks(monkeypatch):
    """A real two-cube MZI must not enumerate every edge-simple walk before checking its cone.

    Count geometry work rather than imposing a machine-dependent time limit. The guard fails promptly
    if pruning regresses, while the existing five-phase test checks the primary interference physics.
    The screen examines trails shortest first (every trail of 4 surfaces, then of 5, ...), re-examining the shorter
    ones at each length: 24,616 cone tests here, finding the 4-surface return through both cubes.
    """
    from os_optics.validation import closed_returns, resonance

    screen = closed_returns.closed_returns
    counts = []

    def measured(scene):
        overlap = resonance._overlap
        checks = 0

        def counted(*args):
            nonlocal checks
            checks += 1
            assert checks <= 40_000, "closed-return screen expanded too many impossible cube walks"
            return overlap(*args)

        with monkeypatch.context() as patch:
            patch.setattr(resonance, "_overlap", counted)
            result = screen(scene)
        counts.append(checks)
        # The speedup must retain the conservative warning about possible inter-element returns.
        assert result[0] and result[1] is None
        return result

    monkeypatch.setattr(closed_returns, "closed_returns", measured)
    _, results = _mach_zehnder()
    assert counts and all(count > 0 for count in counts)
    assert len(results.monitor("BRIGHT")["arrivals"]) == 2


def _single_frequency(run, observation_id: str) -> dict:
    """The coherent sum of a plane's beams at one frequency: the catalog cubes' and mirrors' coatings bound their group
    delays (catalog schema 3.2) but state no phase, so the phase between the arms is open (the run's coherent sum has no
    value, ``PHASE_UNDEFINED``, only its range over every phase); taken alike (the records of those passes cleared), the
    fields add as at one frequency."""
    from os_optics import NumericalConfig
    from os_optics.validation.superposition import superpositions

    obs = next(o for o in run.result.observations if o.observation_id == observation_id)
    single = tuple(a.model_copy(update={"group_delay_unknown_at": ()}) for a in obs.arrivals)
    (total,) = superpositions(single, obs.plane, NumericalConfig())
    return total.model_dump(mode="json")


@pytest.fixture(scope="module", params=[False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def mach_zehnder_catalog(request):
    from local_helpers import with_lossless_bulk

    lossless = request.param
    return (load_public_catalog(with_lossless_bulk(DOCUMENT)) if lossless else CATALOG), lossless


def test_mach_zehnder_output_follows_cos2_of_the_arm_length_and_its_budget_closes(mach_zehnder_catalog):
    """Five arm OPDs and coating-only power bounds. The lossless fixture isolates the phase scan."""
    catalog, lossless = mach_zehnder_catalog
    cube, mirror, wavelength = _mach_zehnder_parts()
    r_c, t_c = (f(_surface(cube, "S2").coating_id, wavelength) for f in (_reflectance, _transmittance))
    t_ar = _transmittance(_surface(cube, "S1").coating_id, wavelength)
    assert all(_transmittance(_surface(cube, s).coating_id, wavelength) == t_ar for s in ("S3", "S4", "S5"))
    r_m = _reflectance(_model(mirror).surfaces[0].coating_id, wavelength)
    # each arm meets one cube transmission and one cube reflection, one mirror and four AR faces
    arm = POWER_W * t_ar**4 * t_c * r_c * r_m
    totals = {}
    for step in range(5):
        extra = step * wavelength * NM / 8.0
        run, results = _mach_zehnder(extra, catalog=catalog)
        bright = results.monitor("BRIGHT")
        assert len(bright["arrivals"]) == 2
        for arrival in bright["arrivals"]:
            if lossless:
                assert arrival["beam"]["power_W"] == pytest.approx(arm, rel=1e-9)
            else:
                _catalog_power(arrival, arm)
            assert arrival["beam"]["direction"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
        if not lossless:
            assert run.status.value == "UNKNOWN" and run.model_scope_issues == ()
            assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
            # Coherent sums cannot be constructed from a beam without established power.
            assert bright["superpositions"] == bright["interference"] == []
            opl = [arrival["beam"]["optical_path_length_mm"] for arrival in bright["arrivals"]]
            assert abs(opl[0] - opl[1]) == pytest.approx(extra, abs=1e-9)
            dark = results.monitor("DARK")
            assert len(dark["arrivals"]) == 2
            for arrival, theoretical in zip(sorted(dark["arrivals"], key=lambda a: a["power_upper_bound_W"]),
                                            sorted(POWER_W * t_ar**4 * r_m * x * x for x in (t_c, r_c))):
                _catalog_power(arrival, theoretical)
            _budget_closes(results)
            continue  # a phase curve cannot be reported without actual power (nor the cube's unstated phase)
        (withheld,) = bright["superpositions"]
        # (the coherent power at the trace's default phases, its entry naming the passes whose phase is not stated)
        assert withheld["total_power_W"] is not None and "PHASE_UNDEFINED" not in withheld["issue_codes"]
        assert withheld["value_basis"]["total_power_W"]["kind"] == "ESTIMATE"
        superposition = _single_frequency(run, "BRIGHT")
        # the same sum with the pass records dropped (every pass taken as stated): the convention's own number
        assert withheld["total_power_W"] == pytest.approx(superposition["total_power_W"], rel=1e-9, abs=1e-15)
        assert superposition["status"] in ("PASS", "WARNING") and superposition["incoherent_power_W"] == pytest.approx(2 * arm, rel=1e-9)
        # the open phase leaves the coherent power anywhere in its range over every phase, the one-frequency sum included
        assert withheld["total_power_min_W"] <= superposition["total_power_W"] <= withheld["total_power_max_W"]
        totals[step] = superposition["total_power_W"]
        (pair,) = bright["interference"]
        # the Engine's own optical path difference is the arm-length change (sign by its operand order)
        assert abs(pair["optical_path_difference_mm"]) == pytest.approx(extra, abs=1e-9)
        budget = _budget_closes(results)
        if step == 0:
            by_element = {line["element_id"]: line["sinks"] for line in budget["by_element"]}
            # what the first cube takes: the AR residual at its entrance and at each exit face, and its coating's absorption
            assert by_element["C1"]["UNTRACED_REFLECTION"] == pytest.approx(POWER_W * (1 - t_ar) * (1 + t_ar * (t_c + r_c)), rel=1e-9)
            assert by_element["C1"]["COATING_ABSORPTION"] == pytest.approx(POWER_W * t_ar * (1 - r_c - t_c), rel=1e-9)
            # the second cube: the AR residual where each arm enters, where the two BRIGHT beams leave through its IN face
            # and where the two DARK beams (T then T, R then R) leave through its fourth face (opposite OUT_R), and the
            # coating's absorption of both arms
            at_c2 = POWER_W * t_ar**2 * r_m * (t_c + r_c)
            dark_inside = POWER_W * t_ar**3 * r_m * (t_c**2 + r_c**2)
            assert by_element["C2"]["COATING_ABSORPTION"] == pytest.approx(at_c2 * t_ar * (1 - r_c - t_c), rel=1e-9)
            assert by_element["C2"]["UNTRACED_REFLECTION"] == pytest.approx((1 - t_ar) * (at_c2 + 2 * arm / t_ar + dark_inside), rel=1e-9)
            # each DARK beam crossed the fourth face's anti-reflection coating
            dark_powers = sorted(a["beam"]["power_W"] for a in results.monitor("DARK")["arrivals"])
            assert dark_powers == pytest.approx(sorted(POWER_W * t_ar**4 * r_m * x * x for x in (t_c, r_c)), rel=1e-9)
        # the complementary port: the two other combinations (T then T, R then R) interfere destructively at equal arms
        dark = results.monitor("DARK")
        dark_sum = _single_frequency(run, "DARK")
        assert len(dark["arrivals"]) == 2 and dark_sum["total_power_W"] == pytest.approx(
            dark_sum["incoherent_power_W"] * 2 * math.sin(math.pi * step / 8.0) ** 2, rel=1e-6, abs=1e-6 * arm)
    if not lossless:
        return
    assert set(totals) == set(range(5))
    # equal arms add in phase at this port (each arm: one splitter transmission and one reflection); the extra path
    # turns the phase by 2 pi extra / lambda: P = 4 P_arm cos^2(pi extra / lambda)
    for step, total in totals.items():
        assert total == pytest.approx(4 * arm * math.cos(math.pi * step / 8.0) ** 2, rel=1e-6, abs=1e-6 * arm)
    assert totals[2] == pytest.approx(2 * arm, rel=1e-6)  # lambda / 4: the mean
    assert totals[4] < 1e-6 * arm  # lambda / 2: dark


def test_mach_zehnder_reaches_both_output_ports_inside_the_model(mach_zehnder_catalog):
    """The recombining cube traces both output ports in scope. Probe limits never stop the primary trace."""
    catalog, lossless = mach_zehnder_catalog
    run, results = _mach_zehnder(catalog=catalog)
    assert len(_arrivals(results, "BRIGHT")) == len(_arrivals(results, "DARK")) == 2
    assert run.model_scope_issues == ()
    # Independent untraced-light probes are auxiliary. Catalog bulk loss still leaves the planes' powers unknown.
    execution = run.result.outcome.execution
    assert execution.outcome.value == "COMPLETED" and execution.complete
    if lossless or execution.budgets.exhausted:
        assert execution.budgets.exhausted == ("max_trace_steps",)
        assert execution.budgets.exhausted_by_phase == {"untraced_light": ("max_trace_steps",)}
        assert execution.budgets.untraced_power_W is None
    if lossless:
        assert all(o.quantity_state.value == "BOUNDED" and not o.power_complete and o.power_upper_bound_W is None
                   and o.value_basis["total_power_W"].kind.value == "LOWER_BOUND" for o in run.result.observations)
    else:
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
        assert all(o.quantity_state.value == "UNKNOWN" and not o.power_complete and o.power_upper_bound_W is None
                   for o in run.result.observations)


# ================================================================================ 3. 4f imaging relay with fold prisms

RELAY_WAIST_MM = 0.5


def _relay_lenses():
    """Two achromatic doublets of one paraxial wavelength whose specified focal lengths are f and 2 f (the first such
    pair by focal length, then SKU), both at least 25 mm across."""

    doublets = [d for d in _of_form("ACHROMATIC_DOUBLET") if (_spec(d, "outer_diameter_mm") or 0) >= 25.0 and _model(d).paraxial is not None
                and _covers(d, float(_model(d).paraxial.wavelength_nm)) and (_spec(d, "focal_length_mm") or 0) >= 50.0]
    for first in sorted(doublets, key=lambda d: (_spec(d, "focal_length_mm"), d.optical_sku)):
        wavelength = _model(first).paraxial.wavelength_nm
        second = [d for d in doublets
                  if _model(d).paraxial.wavelength_nm == wavelength and _spec(d, "focal_length_mm") == 2 * _spec(first, "focal_length_mm")]
        if second:
            return first, second[0], float(wavelength)
    pytest.fail("no pair of achromatic doublets of focal lengths f and 2 f at one paraxial wavelength")


def _fold_prism(wavelength: float):
    """The first right-angle prism (by SKU) published for fold use (OUT at 90 deg to IN) valid at ``wavelength``."""

    for prism in _of_form("RIGHT_ANGLE_PRISM"):
        (_, d_in), (_, d_out) = _port(prism, "IN"), _port(prism, "OUT")
        if abs(float(np.dot(d_in, d_out))) < 1e-9 and _covers(prism, wavelength):
            return prism
    pytest.fail(f"no fold right-angle prism valid at {wavelength} nm")


def _reflector(prism) -> tuple[np.ndarray, np.ndarray]:
    """Local normal and origin of the face that turns the beam: the one surface oblique to both ports."""

    (_, d_in), (_, d_out) = _port(prism, "IN"), _port(prism, "OUT")
    (surface,) = [s for s in _model(prism).surfaces
                  if all(0.1 < abs(float(np.dot([float(v) for v in s.normal], d))) < 0.9 for d in (d_in, d_out))]
    return np.array([float(v) for v in surface.normal]), np.array([float(v) for v in surface.origin_mm])


def _glass_path(prism) -> float:
    """Chief-ray path inside the prism from the IN port to the OUT port via the turning face (element frame)."""

    (p_in, d_in), (p_out, _) = _port(prism, "IN"), _port(prism, "OUT")
    normal, origin = _reflector(prism)
    hit = p_in + float(np.dot(origin - p_in, normal) / np.dot(d_in, normal)) * d_in
    return float(np.linalg.norm(hit - p_in) + np.linalg.norm(p_out - hit))


def _relay(folds: int, targets=None):
    """Source (the object, its waist on it) -> doublet f -> doublet 2 f in a 4f arrangement placed from the published
    paraxial data (front / back focal lengths); ``folds`` right-angle prisms after the second doublet turn the beam
    +X -> +Y -> +X. The image plane sits where the air-equivalent distance (glass path / n) adds up to the second
    doublet's back focal length. Returns the run, its results, the predicted A block and the magnification."""

    first, second, wavelength = _relay_lenses()
    p1, p2 = _model(first).paraxial, _model(second).paraxial
    last1 = float(max(Decimal(s.vertex_z_mm) for s in _model(first).surfaces))
    last2 = float(max(Decimal(s.vertex_z_mm) for s in _model(second).surfaces))
    x1 = float(p1.front_focal_length_mm)  # the object (source) in the first doublet's front focal plane
    x2 = x1 + last1 + float(p1.back_focal_length_mm) + float(p2.front_focal_length_mm)
    point, direction = np.array([x2 + last2, 0.0, 0.0]), np.asarray(X)  # the second doublet's last vertex
    reduced = float(p2.back_focal_length_mm)  # air-equivalent distance left to the image
    components = [_source("SRC", (0.0, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=RELAY_WAIST_MM),
                  _element("L1", "lens", first, (x1, 0.0, 0.0), _axis_pose(X)),
                  _element("L2", "lens", second, (x2, 0.0, 0.0), _axis_pose(X))]
    transport = np.eye(3)  # how the folds carry transverse vectors (Householder reflections)
    lead = 20.0
    for k in range(folds):
        prism = _fold_prism(wavelength)
        out_direction = np.asarray(Y) if k % 2 == 0 else np.asarray(X)
        entry = point + lead * direction
        position, quaternion, rotation = _place_by_ports(prism, entry, direction, "OUT", out_direction)
        components.append(_element(f"P{k + 1}", "prism", prism, position, quaternion))
        n = _sellmeier(_model(prism).body.medium, wavelength)
        reduced -= lead + _glass_path(prism) / n
        normal = rotation @ _reflector(prism)[0]
        transport = (np.eye(3) - 2.0 * np.outer(normal, normal)) @ transport
        point, direction = _out_point(prism, "OUT", position, rotation), out_direction
        lead = 15.0
    image = point + reduced * direction
    assert reduced > 10.0
    run, results = _run(components, [_plane("IMG", image, direction, radius_mm=10.0)], targets)
    magnification = float(p2.effective_focal_length_mm) / float(p1.effective_focal_length_mm)
    plane_frame = results.monitor("IMG")["plane"]
    # predicted A block: the 4f relay inverts (x -> -m x), the folds reflect; source frame x / y -> plane u / v
    source_axes = (np.asarray(Y), np.asarray(Z))  # the source's local x / y (``_axis_pose(X)``)
    predicted = np.array([[-magnification * float(np.dot(transport @ e, plane_frame[axis])) for e in source_axes]
                          for axis in ("u_axis", "v_axis")])
    return run, results, predicted, magnification


@pytest.mark.parametrize("folds", [0, 1, 2])
def test_4f_relay_images_with_the_doublets_magnification_and_folds_set_the_parity(folds: int):
    run, results, predicted, magnification = _relay(folds)
    _in_scope(run)
    arrival = _single(results, "IMG")
    first_order = arrival["first_order"]
    matrix = np.array(first_order["matrix"])
    # the image plane located from the published paraxial data is conjugate to the source plane: B ~ 0 (mm / rad; the
    # B block is reported to 1e-4 mm and the published focal lengths carry 1e-4 mm)
    assert np.abs(matrix[:2, 2:]).max() < 5e-4
    assert np.allclose(matrix[:2, :2], predicted, atol=2e-5 * magnification)
    assert first_order["transverse_magnification"] == pytest.approx(magnification, rel=1e-5)
    assert first_order["reflection_count"] == folds
    assert first_order["handedness_preserved"] is (folds % 2 == 0)
    if folds % 2 == 0:
        # rotation of the source frame onto (u, v) (the beam travels along the plane normal): the angle of the A block
        expected = math.degrees(math.atan2(predicted[1, 0], predicted[0, 0]))
        assert (first_order["image_rotation_deg"] - expected + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-3)
    else:
        assert first_order["image_rotation_deg"] is None and first_order["mirror_axis_deg"] is not None
    if folds == 0:
        assert abs(first_order["image_rotation_deg"]) == pytest.approx(180.0, abs=1e-3)  # a 4f relay inverts the image
    # the Gaussian image: the source waist relayed (w0 m), on the image plane up to 0.05 mm -- the image-side Rayleigh range
    # is m^2 z_R (metres), so the 1e-4 mm rounding of the published focal lengths (a residual power C ~ 1e-8 / mm)
    # moves the waist by about A C z_R^2 / D^2 ~ 0.02 mm
    gaussian = arrival["beam"]["gaussian"]
    for axis in ("x", "y"):
        assert gaussian[f"waist_radius_{axis}_mm"] == pytest.approx(RELAY_WAIST_MM * magnification, rel=1e-4)
        assert abs(gaussian[f"waist_distance_{axis}_mm"]) < 0.05


def test_an_image_orientation_requirement_fails_with_one_fold_and_passes_with_two():
    not_mirrored = [{"target_id": "not_mirrored", "metric": "image_parity", "observation_id": "IMG", "source_id": "SRC",
                     "target_value": 1.0, "tolerance_abs": 0.5}]
    mirrored, _, _, _ = _relay(1, not_mirrored)
    (target,) = mirrored.result.targets
    assert target.status.value == "FAIL" and target.value == -1.0
    rotation = {"target_id": "rotation", "metric": "image_rotation_deg", "observation_id": "IMG", "source_id": "SRC"}
    _, _, predicted, _ = _relay(2)
    expected_rotation = math.degrees(math.atan2(predicted[1, 0], predicted[0, 0]))
    expected_rotation = 180.0 if expected_rotation <= -180.0 + 1e-6 else expected_rotation  # reported in (-180, 180]
    fixed, _, _, _ = _relay(2, not_mirrored + [{**rotation, "target_value": expected_rotation, "tolerance_abs": 0.01}])
    by_id = {t.target_id: t for t in fixed.result.targets}
    assert by_id["not_mirrored"].status.value == "PASS" and by_id["not_mirrored"].value == 1.0
    assert by_id["rotation"].status.value == "PASS" and by_id["rotation"].value == pytest.approx(expected_rotation, abs=1e-3)


# ===================================================================== 4. aspheric vs spherical singlet at one aperture

#: the beam's 1/e^2 radius as a fraction of the smaller clear radius: the outer pupil ring (1.18 radii) stays inside it
BEAM_FILL = 0.55
#: the beam's 1/e^2 radius as a fraction of the smallest surface radius: inside the Gaussian model's paraxial validity
#: (``NumericalConfig.gaussian_max_w_over_radius`` = 0.2), so the run stays in model scope
BEAM_OVER_RADIUS = 0.19
FOCUSING_APPROACH_MM = 30.0


def _singlet_pair():
    """The fastest aspheric lens (largest diameter / focal length, then SKU) evaluated at its design wavelength, paired
    with the plano-convex lens of the same outer diameter and the nearest specified focal length (within 5 percent)
    valid there, of a Sellmeier glass (the independent third-order check evaluates Sellmeier indices)."""

    candidates = []
    for asphere in _of_form("ASPHERIC_LENS"):
        paraxial, design = _model(asphere).paraxial, _spec(asphere, "design_wavelength_nm")
        if paraxial is None or design is None or float(paraxial.wavelength_nm) != design or not _covers(asphere, design):
            continue
        focal, diameter = float(paraxial.effective_focal_length_mm), _spec(asphere, "outer_diameter_mm")
        spheres = [p for p in _of_form("PLANO_CONVEX_LENS") if _spec(p, "outer_diameter_mm") == diameter and _covers(p, design)
                   and _sellmeier_glass(p)
                   and _spec(p, "focal_length_mm") is not None and abs(_spec(p, "focal_length_mm") - focal) <= 0.05 * focal]
        if spheres:
            sphere = min(spheres, key=lambda p: (abs(_spec(p, "focal_length_mm") - focal), p.optical_sku))
            candidates.append((-diameter / focal, asphere.optical_sku, asphere, sphere, design))
    assert candidates, "no aspheric lens with a plano-convex lens of the same diameter and focal length"
    _, _, asphere, sphere, wavelength = min(candidates, key=lambda c: (c[0], c[1]))
    return asphere, sphere, wavelength


def _seidel_spherical(product, wavelength: float, height: float) -> tuple[float, float]:
    """Third-order spherical aberration W040 = S_I / 8 (mm, at ray height ``height``) and the paraxial back focal distance
    of a spherical SURFACE_SEQUENCE lens for a collimated input: a paraxial marginal-ray trace through the published
    vertices, radii and Sellmeier indices with S_I = -sum A^2 h delta(u / n), A = n i (Welford)."""

    model = _model(product)
    media = [1.0, *(_sellmeier(m, wavelength) for m in model.media), 1.0]
    h, u, s_i = height, 0.0, 0.0
    surfaces = list(model.surfaces)
    for k, surface in enumerate(surfaces):
        assert surface.conic is None and not surface.aspheric_coefficients
        c = 1.0 / float(surface.radius_mm) if surface.radius_mm is not None else 0.0
        n, n_next = media[k], media[k + 1]
        u_next = (n * u - h * c * (n_next - n)) / n_next
        a = n * (h * c + u)
        s_i -= a * a * h * (u_next / n_next - u / n)
        u = u_next
        if k + 1 < len(surfaces):
            h += (float(surfaces[k + 1].vertex_z_mm) - float(surface.vertex_z_mm)) * u
    return s_i / 8.0, -h / u


def _best_focus_rms(rays: list[dict]) -> tuple[float, float]:
    """Smallest RMS spot radius of a ray bundle over planes parallel to the monitor (each ray carried along its direction,
    equal weights), and the axial shift where it occurs."""

    position = np.array([[r["u_mm"], r["v_mm"]] for r in rays])
    slope = np.array([[r["direction_u"] / r["direction_n"], r["direction_v"] / r["direction_n"]] for r in rays])
    a, b = position - position.mean(axis=0), slope - slope.mean(axis=0)
    shift = -float(np.sum(a * b) / np.sum(b * b))
    return float(np.sqrt(np.mean(np.sum((a + shift * b) ** 2, axis=1)))), shift


def _focus(product, wavelength: float, waist: float):
    model = _model(product)
    last_vertex = float(max(Decimal(s.vertex_z_mm) for s in model.surfaces))
    bfl = float(model.paraxial.back_focal_length_mm)
    monitor_x = last_vertex + 0.6 * bfl  # in the converging beam, before the focus
    components = [_source("SRC", (-FOCUSING_APPROACH_MM, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=waist),
                  _element("LENS", "lens", product, (0.0, 0.0, 0.0), _axis_pose(X))]
    run, results = _run(components, [_plane("MON", (monitor_x, 0.0, 0.0), X, radius_mm=25.0)])
    return run, results, last_vertex, monitor_x


def test_an_asphere_focuses_much_tighter_than_a_plano_convex_lens_of_the_same_focal_length():
    asphere, sphere, wavelength = _singlet_pair()
    clear = min(float(s.clear_radius_mm) for p in (asphere, sphere) for s in _model(p).surfaces)
    radius = min(abs(float(s.radius_mm)) for p in (asphere, sphere) for s in _model(p).surfaces if s.radius_mm is not None)
    waist = min(BEAM_FILL * clear, BEAM_OVER_RADIUS * radius)
    found = {}
    for product in (asphere, sphere):
        run, results, last_vertex, monitor_x = _focus(product, wavelength, waist)
        if product is asphere:
            _in_scope(run)  # its focus is diffraction limited: the Gaussian beam model holds
        else:
            # the spherical lens' focus is aberration dominated: its Gaussian beam quantities leave the model's scope, said
            # explicitly; the traced rays (spot, wavefront) are geometric optics and stay valid
            assert run.status.value == "OUT_OF_MODEL_SCOPE" and run.model_scope_issues == ()
            assert {issue.code for issue in run.result.issues if issue.status.value == "OUT_OF_MODEL_SCOPE"} == {"GAUSSIAN_ABERRATION_DOMINATED"}
        arrival = _single(results, "MON")
        rays = [r for r in results.rays(monitor="MON") if not r["chief"]]
        assert len(rays) == arrival["pupil_ray_count"] == arrival["pupil_rays_arrived"]  # every pupil ray inside the clear apertures
        rms, _ = _best_focus_rms(rays)
        wave = results.wavefront("MON", arrival["beam"]["path_id"], reference="sphere", center="ray_focus")
        assert wave["status"] == "OK" and wave["reference"]["converging"] is True
        found[product.form.value] = {"rms_mm": rms, "rms_waves": wave["rms_waves"], "results": results, "arrival": arrival,
                                     "last_vertex": last_vertex}
    spherical, aspheric = found["PLANO_CONVEX_LENS"], found["ASPHERIC_LENS"]
    # the seed release's pair: about 3 nm against 13 um geometric RMS and 1e-4 against 0.28 waves RMS (factors ~4000)
    assert spherical["rms_mm"] > 1000 * aspheric["rms_mm"]
    assert spherical["rms_waves"] > 1000 * aspheric["rms_waves"]
    # the plano-convex lens: the independent paraxial trace reproduces its published back focal length, and its spherical
    # aberration (the h^4 term of the wavefront about the paraxial focus, fitted with h^2 and h^6) is the third-order W040
    paraxial = _model(sphere).paraxial
    assert _seidel_spherical(sphere, float(paraxial.wavelength_nm), 1.0)[1] == pytest.approx(float(paraxial.back_focal_length_mm), abs=1e-3)
    focus = (spherical["last_vertex"] + _seidel_spherical(sphere, wavelength, 1.0)[1], 0.0, 0.0)
    wave = spherical["results"].wavefront("MON", spherical["arrival"]["beam"]["path_id"], reference="sphere", center=focus)
    assert wave["status"] == "OK"
    heights = np.array([math.hypot(r["pupil_x"], r["pupil_y"]) * waist for r in wave["rays"]])  # collimated: launch height
    design = np.column_stack([np.ones_like(heights), heights**2, heights**4, heights**6])
    fitted = np.linalg.lstsq(design, np.array([r["opd_mm"] for r in wave["rays"]]), rcond=None)[0][2] * heights.max() ** 4
    w040, _ = _seidel_spherical(sphere, wavelength, float(heights.max()))
    assert w040 > 2 * wavelength * NM  # several waves: well resolved
    # the OPD is the ray's optical path minus the chief's (positive lags): the opposite sign of the classical W
    assert -fitted == pytest.approx(w040, rel=0.05)


# ======================================================================================= 5. off-axis parabolic mirror

def _oap():
    """The first off-axis parabolic mirror (by SKU) that publishes its reflected focal length and covers a laser line."""

    for mirror in _of_form("OFF_AXIS_PARABOLIC_MIRROR"):
        if _spec(mirror, "focal_length_mm") is not None and any(_covers(mirror, w) for w in LASER_LINES_NM):
            return mirror, _laser_line(mirror)
    pytest.fail("no off-axis parabolic mirror with a published reflected focal length")


def _thin_lens_q(q_in: complex, focal_mm: float) -> complex:
    return q_through(np.array([[1.0, 0.0], [-1.0 / focal_mm, 1.0]]), q_in)


def test_off_axis_parabola_focuses_a_collimated_beam_at_its_reflected_focal_length():
    mirror, wavelength = _oap()
    rfl = _spec(mirror, "focal_length_mm")
    waist, approach = 1.0, 200.0
    # IN (-Z local) along design +X through the aperture centre; OUT toward design +Y
    position, quaternion, rotation = _place_by_ports(mirror, (0.0, 0.0, 0.0), X, "OUT", Y)
    out_direction = rotation @ _port(mirror, "OUT")[1]
    focus = np.asarray(position) + rotation @ _port(mirror, "OUT")[0] + rfl * out_direction
    components = [_source("SRC", (-approach, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=waist),
                  _element("OAP", "mirror", mirror, position, quaternion)]
    # the waist (a flat phase front) on the source: every pupil ray starts parallel to the parent axis
    run, results = _run(components, [_plane("FOCUS", focus, out_direction, absorbing=True)])
    _in_scope(run)
    arrival = _single(results, "FOCUS")
    assert arrival["uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9)  # the chief ray of the parent-axis beam meets the focus
    assert arrival["beam"]["direction"] == pytest.approx(list(out_direction), abs=1e-12)
    assert arrival["pupil_rms_radius_mm"] < 1e-6  # a paraboloid is stigmatic for light along its axis
    reflectance = _reflectance(_model(mirror).surfaces[0].coating_id, wavelength)
    assert arrival["beam"]["power_W"] == pytest.approx(POWER_W * reflectance, rel=1e-9)
    # Gaussian focusing (both axes; the section is stigmatic): q' = q / (1 - q / f) with q = d + i z_R on the mirror
    q_out = _thin_lens_q(complex(approach, rayleigh_range(waist, wavelength)), rfl)
    expected_waist = math.sqrt(wavelength * NM * q_out.imag / math.pi)
    assert expected_waist == pytest.approx(wavelength * NM * rfl / (math.pi * waist), rel=1e-3)  # the far-field lambda f / (pi w)
    gaussian = arrival["beam"]["gaussian"]
    for axis in ("x", "y"):
        assert gaussian[f"waist_radius_{axis}_mm"] == pytest.approx(expected_waist, rel=1e-6)
        assert gaussian[f"waist_distance_{axis}_mm"] == pytest.approx(rfl + q_out.real, abs=1e-6)
    _budget_closes(results)


# ============================================================================================= 6. cylindrical lens


def _cylinder():
    """The first positive cylindrical lens (by SKU) whose paraxial block is published inside its band, of one Sellmeier
    glass (the straight meridian is checked as a slab of that glass)."""

    for lens in _of_form("CYLINDRICAL_LENS"):
        paraxial = _model(lens).paraxial
        if paraxial is not None and paraxial.axis in ("X", "Y") and float(paraxial.effective_focal_length_mm) > 0 \
                and _covers(lens, float(paraxial.wavelength_nm)) and len(_model(lens).media) == 1 and _sellmeier_glass(lens):
            return lens
    pytest.fail("no positive cylindrical lens with a published paraxial block")


def _principal_matrix(paraxial, last_vertex_mm: float) -> np.ndarray:
    """First vertex -> last vertex ray-transfer matrix from the published paraxial block alone: air-equivalent travel to
    the front principal plane, the thin lens of the effective focal length, then from the rear principal plane on."""

    f = float(paraxial.effective_focal_length_mm)
    h, h_rear = float(paraxial.front_principal_plane_z_mm), float(paraxial.rear_principal_plane_z_mm)
    return free(last_vertex_mm - h_rear) @ np.array([[1.0, 0.0], [-1.0 / f, 1.0]]) @ free(h)


@pytest.mark.parametrize("turn_deg", [0.0, 90.0], ids=["curved-along-Z", "curved-along-Y"])
def test_cylindrical_lens_focuses_only_its_curved_axis_where_its_paraxial_block_says(turn_deg: float):
    lens = _cylinder()
    model, paraxial = _model(lens), _model(lens).paraxial
    wavelength = float(paraxial.wavelength_nm)
    last_vertex = float(max(Decimal(s.vertex_z_mm) for s in model.surfaces))
    bfl = float(paraxial.back_focal_length_mm)
    waist, approach = 2.0, 100.0
    # local +Z along design +X; the curved local axis lands on design Z (turn 0) or Y (turn 90 deg about the beam)
    rotation = _rotation_about(X, turn_deg) @ _orient(Z, X, Y, Z)
    curved_local = Y if paraxial.axis == "Y" else X
    curved = rotation @ np.array(curved_local)
    screen = last_vertex + bfl
    components = [_source("SRC", (-approach, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=waist),
                  _element("CYL", "lens", lens, (0.0, 0.0, 0.0), _quaternion(rotation))]
    run, results = _run(components, [_plane("LINE", (screen, 0.0, 0.0), X, radius_mm=25.0)])
    _in_scope(run)
    arrival = _single(results, "LINE")
    gaussian = arrival["beam"]["gaussian"]
    curved_key = "x" if abs(float(np.dot(gaussian["e1"], curved))) > 0.5 else "y"
    straight_key = "y" if curved_key == "x" else "x"
    q0 = complex(approach, rayleigh_range(waist, wavelength))  # at the first vertex; the waist (flat phase front) on the source
    # curved meridian: the published paraxial block (principal planes + EFL) carries q from the first to the last vertex
    q_curved = q_through(_principal_matrix(paraxial, last_vertex), q0)
    # (the paraxial block is published to 1e-4 mm: the predicted waist carries a few 1e-5 mm of that rounding)
    assert gaussian[f"waist_distance_{curved_key}_mm"] == pytest.approx(bfl + q_curved.real, abs=2e-4)
    assert gaussian[f"waist_radius_{curved_key}_mm"] == pytest.approx(math.sqrt(wavelength * NM * q_curved.imag / math.pi), rel=1e-4)
    # straight meridian: a plane-parallel slab of the lens glass, reduced thickness t / n
    (glass,) = model.media
    q_straight = q0 + last_vertex / _sellmeier(glass, wavelength)
    assert gaussian[f"waist_distance_{straight_key}_mm"] == pytest.approx(bfl + q_straight.real, abs=1e-6)
    assert gaussian[f"waist_radius_{straight_key}_mm"] == pytest.approx(waist, rel=1e-9)
    # the line focus: the pupil rays collapse along the curved axis and keep their spread along the straight one
    rays = [r for r in results.rays(monitor="LINE") if not r["chief"]]
    assert len(rays) == arrival["pupil_ray_count"] == arrival["pupil_rays_arrived"]
    along = {name: np.array([r[f"{name}_mm"] for r in rays]) for name in ("u", "v")}
    curved_name = "u" if abs(float(np.dot(results.monitor("LINE")["plane"]["u_axis"], curved))) > 0.5 else "v"
    straight_name = "v" if curved_name == "u" else "u"
    spread_straight = float(np.sqrt(np.mean(along[straight_name] ** 2)))
    # the documented equal-power ring layout: ring k of K at rho_k = sqrt(-ln(1 - (k + 1/2) / K) / 2) 1/e^2 radii, evenly
    # spaced points (per-axis mean of cos^2 = 1/2); the continuous Gaussian would give w / 2
    rings = len({r["ring_index"] for r in rays})
    sampled = math.sqrt(sum(-math.log(1.0 - (k + 0.5) / rings) / 2.0 for k in range(rings)) / rings / 2.0)
    assert spread_straight == pytest.approx(waist * sampled, rel=1e-9)
    assert float(np.sqrt(np.mean(along[curved_name] ** 2))) < spread_straight / 50


# ============================================================================ 7. two-colour routing through a bandpass


def _spectrum_bandpass():
    """The first bandpass filter (by SKU) modelled by a typical measured transmittance curve with a published centre."""

    for flt in _of_form("BANDPASS_FILTER"):
        coating_ids = [s.coating_id for s in _model(flt).surfaces if s.coating_id]
        spectra = [_coating(c) for c in coating_ids if _coating(c).spectrum is not None]
        if len(spectra) == 1 and spectra[0].spectrum.status.value == "TYPICAL" and _spec(flt, "center_wavelength_nm") is not None:
            return flt, spectra[0]
    pytest.fail("no bandpass filter with a typical spectrum curve in the seed release")


def _interpolate(spectrum, wavelength_nm: float) -> float:
    nodes = [float(w) for w in spectrum.wavelength_nm]
    values = [float(t) for t in spectrum.transmittance]
    k = max(i for i, w in enumerate(nodes) if w <= wavelength_nm)
    if k == len(nodes) - 1:
        return values[-1]
    fraction = (wavelength_nm - nodes[k]) / (nodes[k + 1] - nodes[k])
    return values[k] + fraction * (values[k + 1] - values[k])


def _blocked_line(spectrum, centre_nm: float, fwhm_nm: float) -> float:
    """A wavelength deep in the blocking band: the midpoint of the lowest-transmittance pair of neighbouring nodes at
    least five bandwidths from the centre (the nearest such pair on ties)."""

    nodes = [float(w) for w in spectrum.wavelength_nm]
    values = [float(t) for t in spectrum.transmittance]
    pairs = [(max(values[i], values[i + 1]), abs((nodes[i] + nodes[i + 1]) / 2 - centre_nm), (nodes[i] + nodes[i + 1]) / 2)
             for i in range(len(nodes) - 1) if min(abs(nodes[i] - centre_nm), abs(nodes[i + 1] - centre_nm)) > 5 * fwhm_nm]
    return min(pairs)[2]


def _two_colour_run(filter_product, lines):
    components = [_source("SRC", (-100.0, 0.0, 0.0), X, lines=[(w, POWER_W) for w in lines], waist_mm=1.0),
                  _element("BPF", "filter", filter_product, (0.0, 0.0, 0.0), _axis_pose(X))]
    return _run(components, [_plane("DET", (50.0, 0.0, 0.0), X, absorbing=True)])


def test_bandpass_filter_passes_its_centre_line_and_blocks_the_other_as_its_typical_curve_says():
    flt, coating = _spectrum_bandpass()
    centre, fwhm = _spec(flt, "center_wavelength_nm"), _spec(flt, "bandwidth_fwhm_nm")
    blocked = _blocked_line(coating.spectrum, centre, fwhm)
    assert _covers(flt, centre, blocked)
    run, results = _two_colour_run(flt, (centre, blocked))
    _in_scope(run)
    by_wavelength = {row["wavelength_nm"]: row["power_W"] for row in results.monitor("DET")["power_by_wavelength"]}
    assert set(by_wavelength) == {centre, blocked}
    for wavelength in (centre, blocked):
        # one transmitting plane (substrate not modelled): the line power times the curve, linearly interpolated
        assert by_wavelength[wavelength] == pytest.approx(POWER_W * _interpolate(coating.spectrum, wavelength), rel=1e-9, abs=1e-18)
    assert by_wavelength[centre] > 0.5 * POWER_W and by_wavelength[blocked] < 1e-3 * POWER_W
    budget = _budget_closes(results)
    lines = {line["wavelength_nm"]: line for line in budget["lines"]}
    for wavelength in (centre, blocked):
        assert lines[wavelength]["input_power_W"] == pytest.approx(POWER_W, rel=1e-12) and lines[wavelength]["conserved"] is True
        assert lines[wavelength]["sinks"]["DETECTED"] == pytest.approx(by_wavelength[wavelength], rel=1e-12, abs=1e-18)


def test_a_line_outside_the_filters_measured_curve_is_out_of_model_scope():
    flt, coating = _spectrum_bandpass()
    centre = _spec(flt, "center_wavelength_nm")
    outside = float(coating.spectrum.wavelength_nm[-1]) + 100.0
    run, _ = _two_colour_run(flt, (centre, outside))
    assert run.status.value == "OUT_OF_MODEL_SCOPE"
    issues = [(issue.instance_id, issue.code) for issue in run.model_scope_issues]
    assert issues and all(instance == "BPF" for instance, _ in issues), issues
    # both the element's model band and its coating's measured span are exceeded, each said once
    assert sorted(code for _, code in issues) == ["COATING_WAVELENGTH_OUTSIDE_BANDS", "MODEL_WAVELENGTH_OUTSIDE_BAND"], issues
    assert all(f"{outside:.1f} nm" in issue.message for issue in run.model_scope_issues)


# =================================================================== 8. double pass through one monitor (retro prism)


def _retro_prism():
    """The first right-angle prism (by SKU) published for retro use: its OUT port antiparallel to IN, legs reflecting."""

    for prism in _of_form("RIGHT_ANGLE_PRISM"):
        (_, d_in), (_, d_out) = _port(prism, "IN"), _port(prism, "OUT")
        reflecting = [s for s in _model(prism).surfaces if s.traced.value == "REFLECTED"]
        if float(np.dot(d_in, d_out)) < -1 + 1e-9 and len(reflecting) == 2 and any(_covers(prism, w) for w in LASER_LINES_NM):
            return prism, _laser_line(prism)
    pytest.fail("no retro right-angle prism with reflecting legs in the seed release")


RETRO_MONITOR_X = -40.0


def _retro_run(**kwargs):
    prism, wavelength = _retro_prism()
    # IN along design +X through the origin; the prism's height (local Y) along design Z
    position, quaternion, rotation = _place_by_ports(prism, (0.0, 0.0, 0.0), X, "OUT", (-1.0, 0.0, 0.0), local_secondary=Y, global_secondary=Z)
    components = [_source("SRC", (-120.0, 0.0, 0.0), X, wavelength_nm=wavelength, waist_mm=0.5), _element("RETRO", "prism", prism, position, quaternion)]
    run, results = _run(components, [_plane("MON", (RETRO_MONITOR_X, 0.0, 0.0), X, radius_mm=20.0)], **kwargs)
    return run, results, _out_point(prism, "OUT", position, rotation)


def test_a_retro_prism_double_pass_reports_both_crossings_of_one_monitor():
    prism, wavelength = _retro_prism()
    run, results, out_point = _retro_run()
    monitor_x = RETRO_MONITOR_X
    _in_scope(run)
    crossings = sorted(_arrivals(results, "MON"), key=lambda a: a["crossing_index"])
    assert [a["crossing_index"] for a in crossings] == [0, 1]
    assert crossings[0]["beam"]["path_id"] == crossings[1]["beam"]["path_id"]
    outbound, back = crossings
    assert outbound["beam"]["direction"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert back["beam"]["direction"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    # the return leaves the OUT port antiparallel, displaced by the port offset
    assert back["beam"]["position_mm"] == pytest.approx([monitor_x, out_point[1], out_point[2]], abs=1e-9)
    # power: the bare hypotenuse twice at normal incidence (Fresnel of the body glass) and the two coated legs
    n = _sellmeier(_model(prism).body.medium, wavelength)
    legs = [_reflectance(s.coating_id, wavelength) for s in _model(prism).surfaces if s.traced.value == "REFLECTED"]
    # (the retro path enters and leaves the hypotenuse off its centre: the clear aperture clips a far Gaussian tail, < 1e-7)
    _catalog_power(back, POWER_W * (1 - _fresnel_normal(n)) ** 2 * legs[0] * legs[1], rel=1e-7)
    assert outbound["beam"]["power_W"] == pytest.approx(POWER_W, rel=1e-12)
    assert back["first_order"]["reflection_count"] == 2 and back["first_order"]["handedness_preserved"]
    # the results layer keeps the two crossings apart: rays per crossing, both complete
    for index in (0, 1):
        rays = results.rays(monitor="MON", crossing_index=index)
        assert {r["crossing_index"] for r in rays} == {index} and len(rays) == 1 + back["pupil_ray_count"]
    budget = _budget_closes(results)
    assert budget["total_sinks"].get("APERTURE_TRUNCATION", 0.0) < 1e-7 * POWER_W


# ============================================================================ the results CLI on a benchmark run


def _cli(argv, capsys):
    code = cli_main(argv)
    return code, capsys.readouterr().out


def test_the_results_cli_reads_a_benchmark_run_directory(tmp_path, capsys):
    run, _, _ = _retro_run(output_dir=tmp_path / "run")
    directory = str(tmp_path / "run")
    code, out = _cli(["results", "describe", directory], capsys)
    described = json.loads(out)
    assert code == 0 and described["run"]["trace_hash"] == run.result.manifest.trace_hash
    code, out = _cli(["results", "monitors", directory], capsys)
    (monitor,) = json.loads(out)
    assert code == 0 and monitor["observation_id"] == "MON" and monitor == json.loads(json.dumps(load_run(directory).monitors()[0]))
    target = tmp_path / "arrivals.csv"
    code, out = _cli(["export", "arrivals", directory, "--format", "csv", "--out", str(target)], capsys)
    assert code == 0 and json.loads(out)["bytes"] == target.stat().st_size
    rows = list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8"))))
    # the double pass: one path, crossings 0 and 1 of the monitor, each with the power the API reports
    assert sorted(int(r["crossing_index"]) for r in rows) == [0, 1] and len({r["path_id"] for r in rows}) == 1
    api = {a["crossing_index"]: a for a in load_run(directory).monitor("MON")["arrivals"]}
    for row in rows:
        arrival = api[int(row["crossing_index"])]
        if arrival["beam"]["power_W"] is None:
            assert row["power_W"] == "" and arrival["power_upper_bound_W"] is not None
        else:
            assert float(row["power_W"]) == pytest.approx(arrival["beam"]["power_W"], rel=1e-12)
