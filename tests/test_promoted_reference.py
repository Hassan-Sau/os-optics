"""A beam whose chief ray falls on the rim beyond a clear aperture while some of its pupil rays pass.

Before, the whole beam was ``CLIPPED`` there and every plane behind it lost the field: every later decision is taken on the
reference ray (row 0, the chief ray), and no surviving ray carried that role. Now the passing pupil sample nearest the
centroid of the passing samples becomes the reference ray; the beam goes on with the power that passes (the Gaussian
integral over the opening for a Gaussian beam, else the geometric pupil-sample share), its Gaussian metrics unavailable.

Independent references: the Gaussian power on an offset disc (Gauss-Legendre in r with the Bessel-I0 azimuthal integral),
the Gaussian power on the overlap of two discs (a dense polar sum), the far-field spread ``L lambda / (pi w0)``, and a
sequential Monte-Carlo trace of the section fixtures written here (vector Snell, the prescription's spheres and clear
semi-diameters, the glasses' Sellmeier terms from the CSV) for the fast objectives' fields.

The geometric basis also books the share of pupil samples that part from the beam before the surface its reference ray
meets (beside an element's support, or onto another surface): before, those samples were dropped and their power stayed in
the beam (about +20 to +30 % on the fast objective's vignetted fields, +2 to +7 % on its oblique fields)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from helpers import assert_power_not_computed, issue_codes, one_arrival
from os_optics import ValidationRequest
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot
from os_optics.local import json_api, run_optical_project
from os_optics.validation import validate

SECTION = Path(__file__).parent / "fixtures" / "section"
SOURCE_W = 1e-3
AXIS_X = (0.5, 0.5, 0.5, 0.5)  # local +Z along design +X


def _laser(w0: float, wavelength_nm: float, y: float, rings: int = 8, per_ring: int = 12):
    laser = B.laser("L", wavelength_nm=wavelength_nm, waist_radius_mm=w0, position=(0, y, 0), direction=(1, 0, 0), power_W=SOURCE_W)
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pupil_ring_count": rings, "pupil_points_per_ring": per_ring})})


def _run(*components, planes=()):
    return validate(ValidationRequest(project=ProjectSnapshot(name="promoted", components=components, observations=planes)))


def _plane(x: float, absorbing: bool = False):
    return B.observation_plane("out", position=(x, 0, 0), normal=(1, 0, 0), radius_mm=50.0, absorbing=absorbing)


def _disc(a: float, d: float, w: float, nodes: int = 400) -> float:
    """Power fraction of a round Gaussian (1/e^2 radius ``w``) inside a disc of radius ``a`` whose centre is ``d`` off the beam
    axis (Gauss-Legendre in r, the azimuthal integral as the Bessel function I0)."""
    x, weights = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * a * (x + 1.0)
    f = (4.0 * r / (w * w)) * np.exp(-2.0 * (r * r + d * d) / (w * w)) * np.i0(4.0 * r * d / (w * w))
    return float(0.5 * a * np.sum(weights * f))


def _w(w0: float, wavelength_nm: float, z: float) -> float:
    z_r = math.pi * w0 * w0 / (wavelength_nm * 1e-6)
    return w0 * math.sqrt(1.0 + (z / z_r) ** 2)


# ------------------------------------------------------------------------------------- a Gaussian beam beside a stop


def test_a_gaussian_beam_whose_chief_ray_misses_the_opening_goes_on_with_the_power_inside_it():
    """A 1 mm beam whose axis passes 1.3 mm off the centre of a 1 mm stop (outside the opening): the beam goes on with the
    Gaussian power inside the opening, its reference ray the passing sample nearest the passing samples' centroid."""
    w0, offset = 1.0, 1.3
    result = _run(_laser(w0, 633.0, -offset, 32, 48), B.aperture_stop("S1", position=(10, 0, 0), axis=(1, 0, 0), radius_mm=1.0),
                  planes=(_plane(40.0),))
    truth = _disc(1.0, offset, _w(w0, 633.0, 10.0))
    arrival = one_arrival(result, "out")
    assert arrival.reference_ray_promoted_at == ("S1/stop",)
    assert arrival.beam.power_W == pytest.approx(SOURCE_W * truth, rel=1e-6)
    assert arrival.power_basis is None  # (the Gaussian integral: no sample decided the power)
    assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
    assert arrival.beam.gaussian.reason == "GAUSSIAN_SEVERE_TRUNCATION"
    assert arrival.first_order is None and "reference" in arrival.first_order_issue
    # the reference ray is a passing sample: inside the opening, on the side of the beam the opening holds
    assert -1.0 < arrival.beam.position_mm[1] < -0.3
    # the passing samples' centroid is the light's (the intensity-weighted centroid of the Gaussian on the disc, dense sum)
    r = np.linspace(0.0, 1.0, 1201)[1:] - 0.5 / 1200
    phi = np.linspace(0.0, 2.0 * math.pi, 1441)[:-1]
    rr, pp = np.meshgrid(r, phi)
    y = rr * np.sin(pp)  # (disc centre at y = 0 in the plane's frame, the beam axis at y = -1.3)
    weight = rr * np.exp(-2.0 * (rr * rr + 2.0 * offset * y + offset * offset) / _w(w0, 633.0, 10.0) ** 2)
    centroid_y = float(np.sum(weight * y) / np.sum(weight))
    assert arrival.pupil_centroid_uv_mm is not None
    assert abs(abs(arrival.pupil_centroid_uv_mm[0]) - abs(centroid_y)) < 0.03 or abs(abs(arrival.pupil_centroid_uv_mm[1]) - abs(centroid_y)) < 0.03
    clip = next(i for i in result.issues if i.code.value == "CLIPPED")
    assert clip.status.value == "WARNING" and 0 < clip.numerical_context["pupil_rays_passing"] < clip.numerical_context["pupil_rays_live"]
    assert clip.numerical_context["transmitted_fraction"] == pytest.approx(truth, rel=1e-6)
    budget = result.power_budget
    assert budget.conserved
    cut = sum(e.power_W for e in budget.entries if e.kind == "APERTURE_TRUNCATION")
    assert cut + arrival.beam.power_W == pytest.approx(SOURCE_W, rel=1e-9)


@pytest.mark.parametrize("eps", [0.0, 1e-9, -1e-9])
def test_a_beam_centred_on_the_rim_of_a_stop_goes_on_with_the_exact_power_inside_it(eps: float):
    """The beam's axis on the rim of a 0.5 mm stop (1 mm beam, stop 10 mm from the waist), and 1 nm either side: the power is
    the Gaussian integral over the opening of the beam as it reaches the stop, ``w(10 mm) = w0 sqrt(1 + (z / z_R)^2)``
    (1.000002 mm), continuous across the rim. Reference: SciPy's adaptive quadrature of the radial disc integral with the
    exponentially scaled I0 (not the Gauss-Legendre rule of ``_disc``). (Taking the waist radius instead gives 2e-6 more:
    that is the beam's growth over 10 mm, not a quadrature error.)"""
    from scipy.integrate import quad
    from scipy.special import i0e

    w0, a, d = 1.0, 0.5, 0.5 + eps
    w = _w(w0, 633.0, 10.0)
    truth = quad(lambda r: 4.0 * r / (w * w) * math.exp(-2.0 * (r - d) ** 2 / (w * w)) * i0e(4.0 * r * d / (w * w)), 0.0, a,
                 epsabs=1e-14, epsrel=1e-12, limit=200)[0]
    result = _run(_laser(w0, 633.0, d), B.aperture_stop("S1", position=(10, 0, 0), axis=(1, 0, 0), radius_mm=a), planes=(_plane(100.0),))
    assert one_arrival(result, "out").beam.power_W == pytest.approx(SOURCE_W * truth, rel=1e-10)
    assert result.power_budget.conserved


def test_a_beam_with_no_pupil_ray_in_the_opening_is_still_clipped_whole():
    """A 0.1 mm beam 0.3 mm beyond the edge: no pupil ray passes, the beam is CLIPPED as before."""
    result = _run(_laser(0.1, 633.0, -1.3), B.aperture_stop("S1", position=(10, 0, 0), axis=(1, 0, 0), radius_mm=1.0),
                  planes=(_plane(40.0),))
    assert [p.termination for p in result.paths] == ["CLIPPED"]
    assert result.observations[0].arrivals == ()


# ------------------------------------------------------------------------------------- the geometric pupil-sample basis


def _overlap(a1: float, a2: float, c2: float, w: float, n: int = 2400) -> float:
    """Gaussian (1/e^2 radius ``w``, centred) power on the overlap of a centred disc ``a1`` and a disc ``a2`` centred at
    ``c2`` on y (a dense polar midpoint sum over the second disc)."""
    r = (np.arange(n) + 0.5) * (a2 / n)
    phi = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    rr, pp = np.meshgrid(r, phi)
    x, y = rr * np.cos(pp), c2 + rr * np.sin(pp)
    inside = x * x + y * y <= a1 * a1
    density = (2.0 / (math.pi * w * w)) * np.exp(-2.0 * (x * x + y * y) / (w * w))
    return float(np.sum(density * inside * rr) * (a2 / n) * (2.0 * math.pi / n))


def test_a_beam_without_a_gaussian_state_promotes_its_reference_on_the_sampled_basis_and_converges():
    """A centred 1 w stop takes 13.5 % (the Gaussian state is lost), then a 0.6 mm stop centred 0.8 mm off the axis (the
    chief ray falls outside it): what passes both is the Gaussian power on the overlap of the two discs. The sampled power
    lies within its resolution of it, and both shrink as the samples grow."""
    w = _w(1.0, 633.0, 20.0)
    truth = SOURCE_W * _overlap(1.0, 0.6, 0.8, w)
    rows = []
    for rings, per_ring in ((8, 12), (16, 24), (32, 48)):
        result = _run(_laser(1.0, 633.0, 0.0, rings, per_ring),
                      B.aperture_stop("S1", position=(10, 0, 0), axis=(1, 0, 0), radius_mm=1.0),
                      B.aperture_stop("S2", position=(20, 0.8, 0), axis=(1, 0, 0), radius_mm=0.6), planes=(_plane(40.0),))
        arrival = one_arrival(result, "out")
        assert arrival.reference_ray_promoted_at == ("S2/stop",)
        assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
        assert "GEOMETRIC_POWER_BASIS" in issue_codes(result) and result.power_budget.conserved
        rows.append((abs(arrival.beam.power_W - truth), arrival.power_sampling_resolution_W))
    assert all(error <= resolution for error, resolution in rows), rows
    assert rows[0][1] > rows[1][1] > rows[2][1]
    assert rows[2][0] < 0.03 * truth


def test_the_diffraction_check_keeps_a_transfer_behind_a_promoted_reference():
    """A 20 um waist at 780 nm (z_R 1.6 mm): a stop 20 mm on whose edge passes 5 um beside the chief ray (the Gaussian power
    inside the opening goes on, about half), then a 0.5 mm stop 30 mm later. The rays stay within 25 um while the beam is
    0.62 mm there: the relaunched differential rays carry the transfer on, so the spread the rays miss is ``L lambda /
    (pi w0)`` with L = 50 mm (the composed transfer of both legs) and the power behind that stop is not computed (at most
    what reached it), not passed whole; the rays go on to the plane."""
    w0 = 0.02
    result = _run(_laser(w0, 780.0, 0.0), B.aperture_stop("S1", position=(20, -1.005, 0), axis=(1, 0, 0), radius_mm=1.0),
                  B.aperture_stop("S2", position=(50, 0, 0), axis=(1, 0, 0), radius_mm=0.5), planes=(_plane(170.0),))
    clip = next(i for i in result.issues if i.code.value == "CLIPPED")
    assert 0.4 < clip.numerical_context["transmitted_fraction"] < 0.5
    _obs, arrival = assert_power_not_computed(result, "out")
    assert arrival.power_upper_bound_W == pytest.approx(SOURCE_W * clip.numerical_context["transmitted_fraction"], rel=1e-9)
    issue = next(i for i in result.issues if "diffraction_spread_mm" in i.numerical_context)
    assert issue.location == "surface:S2/stop"
    assert issue.numerical_context["diffraction_spread_mm"] == pytest.approx(50.0 * 780e-6 / (math.pi * w0), rel=2e-3)


# ------------------------------------------------------------------------------------- the pupil sampling of a Project


def _project(budget=None) -> dict:
    from composition_helpers import laser, project_of, screen  # noqa: PLC0415

    doc = project_of([laser("S")], [screen("screen", (100, 0, 0))])
    if budget is not None:
        doc["compute_budget"] = budget
    return doc


def test_the_pupil_sampling_is_a_project_setting_bound_into_the_engine_input():
    from os_optics._vendor.shared_contracts.optical_design.compute_budget import ComputeBudgetV1  # noqa: PLC0415
    from os_optics.local.project_ops import ProjectOpError, apply_project_ops  # noqa: PLC0415
    from pydantic import ValidationError  # noqa: PLC0415

    baseline = json_api.check_project(_project())
    edited = json_api.update_project(_project(), [{"op": "set_compute_budget",
                                                   "compute_budget": {"pupil_ring_count": 16, "pupil_points_per_ring": 24}}])
    assert edited["ok"], edited
    assert edited["compute_budget"]["pupil_ring_count"] == 16 and edited["compute_budget"]["pupil_points_per_ring"] == 24
    assert baseline["compute_budget"]["pupil_ring_count"] == 8 and baseline["compute_budget"]["pupil_points_per_ring"] == 12
    assert edited["engine_input_hash"] != baseline["engine_input_hash"]
    run = run_optical_project(edited["project"])
    (arrival,) = [a for o in run.result.observations for a in o.arrivals]
    assert arrival.pupil_ray_count == 16 * 24
    fields = json_api.capabilities()["compute_budget"]["fields"]
    # (the budgets' common ceiling, JSON's integer range: the run monitor, not the contract, bounds a run's size)
    assert fields["pupil_ring_count"] == {"minimum": 1, "maximum": 2**53 - 1, "default": 8}
    assert fields["pupil_points_per_ring"] == {"minimum": 1, "maximum": 2**53 - 1, "default": 12}
    assert ComputeBudgetV1.model_validate({"pupil_ring_count": 65, "pupil_points_per_ring": 257}).overrides() == {
        "pupil_ring_count": 65, "pupil_points_per_ring": 257}
    for name, bad in (("pupil_ring_count", 2**53), ("pupil_points_per_ring", 2**53), ("pupil_ring_count", 0)):
        with pytest.raises(ValidationError):
            ComputeBudgetV1.model_validate({name: bad})
        with pytest.raises(ProjectOpError):
            apply_project_ops(_project(), [{"op": "set_compute_budget", "compute_budget": {name: bad}}])


# ------------------------------------------------------------------------------------- the fast objectives, every field

AR = {"coating_id": "AR0", "kind": "ANTI_REFLECTION", "display_name": "Declared anti reflection R=0",
      "bands": [{"min_nm": "300", "max_nm": "2600", "reflectance": "0", "transmittance": "1", "status": "DESIGN_BASIS"}],
      "valid_aoi_deg": {"min_deg": "0", "max_deg": "89", "status": "DESIGN_BASIS"}, "aoi_reference": "AIR_EQUIVALENT",
      "basis": "Declared anti-reflection coating (lossless: T = 1 - R)"}


def _fast_project(csv_name: str, fields: tuple[float, ...], aimed: bool, rings: int, per_ring: int) -> dict:
    """The section fixture with lossless faces (so the power is the geometric throughput), collimated fields of a beam whose
    1/e^2 radius is the stop's, launched 100 mm before the stop: ``aimed`` through the stop's centre, otherwise from the axis
    (their chief rays fall beyond the stop's opening at 13 and 14 deg)."""
    text = (SECTION / csv_name).read_text(encoding="utf-8")
    imported = json_api.import_prescription(text, {"model_id": "fast"})
    assert imported["ok"], imported
    for model in imported["optical_models"]:
        for surface in model["model"].get("surfaces") or ():
            surface["coating_id"] = "AR0"
    # The independent throughput trace deliberately models ideal glass and
    # lossless faces. Declare the same zero bulk attenuation for these test-only
    # imported materials; an unstated real catalog loss is not zero by default.
    for material in imported["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    models = {m["model_id"]: m for m in imported["optical_models"]}
    members = {m["member_id"]: m for m in models["fast"]["model"]["members"]}
    waist = next(float(row.split(",")[4]) for row in text.splitlines() if row.startswith("STO,"))
    components = []
    for k, field in enumerate(fields):
        half = math.radians(field) / 2.0
        w0, z0 = math.cos(half), math.sin(half)
        w, x, y, z = AXIS_X
        q = (w0 * w - z0 * z, w0 * x - z0 * y, w0 * y + z0 * x, w0 * z + z0 * w)
        components.append({
            "instance_id": f"FIELD{k}", "optical_role": "source", "custom_part": {"custom_part_key": "collimated", "description": "field"},
            "nominal_pose": {"position_mm": [-100.0, -math.tan(math.radians(field)) * 100.0 if aimed else 0.0, 0.0], "quaternion_wxyz": list(q)},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 587.6, "power_w": SOURCE_W, "waist_radius_mm": waist,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}})
    for member_id, member in members.items():
        components.append({"instance_id": member_id, "optical_role": "aperture" if member_id == "STOP" else "lens",
                           "optical_model_id": member["model_id"], "custom_part": {"custom_part_key": member_id.lower(), "description": member_id},
                           "nominal_pose": {"position_mm": [member["pose"]["position_mm"][2], 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)}})
    image_x = float(imported["report"]["image_plane_z_mm"])
    # (in vacuum: the independent sequential trace below takes the medium between the lenses to have index 1)
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": components,
            "environment": {"pressure_pa": "0"}, "optical_models": [m for key, m in models.items() if key != "fast"], "model_materials": imported["model_materials"],
            "model_coatings": [AR], "compute_budget": {"pupil_ring_count": rings, "pupil_points_per_ring": per_ring},
            "observations": [{"observation_id": "image", "pose": {"position_mm": [image_x, 0.0, 0.0], "quaternion_wxyz": list(AXIS_X)},
                              "aperture": {"shape": "circle", "radius_mm": 40.0}, "absorbing": True, "medium": "air"}]}


def _sequential_reference(csv_name: str, field: float, aimed: bool, n: int = 400_000, launch: tuple[float, float] | None = None):
    """Monte-Carlo throughput, image centroid and RMS radius of the Gaussian beam through the CSV prescription (the stop and
    every clear semi-diameter a hard edge; air n = 1; vector Snell on the spheres). With ``launch`` (an offset in the
    source's (e1, e2) in mm), that one ray's image point and direction instead."""
    stop, surfaces, materials, x, image_x, section = None, [], {}, 0.0, None, None
    for line in (SECTION / csv_name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        cols = line.split(",")
        if cols[0] in ("surface", "name"):
            continue
        if section == "materials":
            materials[cols[0]] = [float(v) for v in cols[2:8]]
            continue
        if cols[0] == "OBJ" or image_x is not None:
            continue
        if cols[0] == "IMA":
            image_x = x
            continue
        if cols[0] == "STO":
            stop = (x, float(cols[4]))
        else:
            surfaces.append((x, None if cols[1] == "inf" else float(cols[1]), float(cols[4]), cols[3]))
        x += float(cols[2])
    lam2 = 0.5876 ** 2

    def index(name: str) -> float:
        if not name:
            return 1.0
        b1, b2, b3, c1, c2, c3 = materials[name]
        return math.sqrt(1 + b1 * lam2 / (lam2 - c1) + b2 * lam2 / (lam2 - c2) + b3 * lam2 / (lam2 - c3))

    th = math.radians(field)
    d0, e1, e2 = np.array([math.cos(th), math.sin(th), 0.0]), np.array([-math.sin(th), math.cos(th), 0.0]), np.array([0.0, 0.0, 1.0])
    src = np.array([-100.0, -math.tan(th) * 100.0 if aimed else 0.0, 0.0])
    if launch is None:
        xy = np.random.default_rng(437).normal(0.0, stop[1] / 2.0, size=(n, 2))
    else:
        xy = np.array([launch])
    o = src + xy[:, :1] * e1 + xy[:, 1:] * e2
    d = np.tile(d0, (o.shape[0], 1))
    p = o + ((stop[0] - o[:, 0]) / d[:, 0])[:, None] * d
    alive = p[:, 1] ** 2 + p[:, 2] ** 2 <= stop[1] ** 2
    o, n1 = p, 1.0
    for xv, radius, semi, material in surfaces:
        n2 = index(material)
        if radius is None:
            p = o + ((xv - o[:, 0]) / d[:, 0])[:, None] * d
            normal = np.tile(np.array([-1.0, 0.0, 0.0]), (o.shape[0], 1))
        else:
            centre = np.array([xv + radius, 0.0, 0.0])
            oc = o - centre
            b = np.sum(d * oc, axis=1)
            disc = b * b - (np.sum(oc * oc, axis=1) - radius * radius)
            alive &= disc >= 0.0
            p = o + (-b - math.copysign(1.0, radius) * np.sqrt(np.clip(disc, 0.0, None)))[:, None] * d
            normal = (p - centre) / radius
        alive &= p[:, 1] ** 2 + p[:, 2] ** 2 <= semi * semi
        normal = np.where((np.sum(d * normal, axis=1) > 0.0)[:, None], -normal, normal)
        cos_i = -np.sum(d * normal, axis=1)
        mu = n1 / n2
        sin2 = mu * mu * (1.0 - cos_i * cos_i)
        alive &= sin2 < 1.0
        d = mu * d + (mu * cos_i - np.sqrt(np.clip(1.0 - sin2, 0.0, None)))[:, None] * normal
        d = d / np.sqrt(np.sum(d * d, axis=1))[:, None]
        o, n1 = p, n2
    image = o + ((image_x - o[:, 0]) / d[:, 0])[:, None] * d
    if launch is not None:
        return bool(alive[0]), image[0], d[0]
    q = image[alive][:, 1:]
    centroid = q.mean(axis=0)
    return float(alive.mean()), centroid, math.sqrt(float(np.mean(np.sum((q - centroid) ** 2, axis=1))))


def _launch_offset(row: int, rings: int, per_ring: int, w: float) -> tuple[float, float]:
    """The source-plane offset of pupil sample ``row`` (1-based) of the equal-power ring layout (ring k at the radius holding
    (k + 1/2) / K of the power, alternate rings turned by half a step)."""
    k, m = divmod(row - 1, per_ring)
    rho = math.sqrt(-0.5 * math.log(1.0 - (k + 0.5) / rings))
    phi = 2.0 * math.pi * (m + (0.5 if k % 2 else 0.0)) / per_ring
    return w * rho * math.cos(phi), w * rho * math.sin(phi)


@pytest.mark.parametrize(("csv_name", "fields", "aimed"), (
    ("fast_f1p4.csv", (0.0, 8.5, 12.0, 16.0, 20.0), True), ("fast_f1p2.csv", (0.0, 8.5, 12.0), True),
    ("fast_f1p4.csv", (12.0, 13.0, 14.0), False), ("fast_f1p2.csv", (12.0,), False)))
def test_the_fast_objectives_keep_every_field_against_an_independent_sequential_trace(csv_name, fields, aimed):
    """F/1.4 and F/1.2 with lossless faces, every field: the image receives each field with its geometric power, spot and
    angle of incidence; the power lies within its sampling resolution of the independent trace's throughput (and within
    3 %), the spot's centroid within 0.02 mm and its RMS radius within 10 %. Fields launched from the axis put the chief ray
    beyond the stop's opening (F/1.4 13, 14 deg; F/1.2 12 deg) or a lens's (F/1.4 12 deg): they keep their field through a promoted reference ray,
    which is exactly the pupil sample the engine names (the independent trace of that one ray lands where the arrival is)."""
    rings, per_ring = 32, 48
    run = run_optical_project(_fast_project(csv_name, fields, aimed, rings, per_ring))
    result = run.result
    (image,) = result.observations
    by_field = {a.beam.source_id: a for a in image.arrivals}
    assert set(by_field) == {f"FIELD{k}" for k in range(len(fields))}
    budget = result.power_budget
    assert budget.conserved and all(line.conserved for line in budget.lines)
    waist = next(float(row.split(",")[4]) for row in (SECTION / csv_name).read_text(encoding="utf-8").splitlines() if row.startswith("STO,"))
    clips = {i.numerical_context["path_id"]: i for i in result.issues if i.code.value == "CLIPPED"}
    for k, field in enumerate(fields):
        arrival = by_field[f"FIELD{k}"]
        throughput, centroid, rms = _sequential_reference(csv_name, field, aimed)
        error = abs(arrival.beam.power_W - SOURCE_W * throughput)
        resolution = arrival.power_sampling_resolution_W or 0.0
        assert error <= resolution + 4.0 * SOURCE_W * math.sqrt(throughput * (1.0 - throughput) / 400_000), (field, arrival.beam.power_W, throughput)
        assert error <= 0.03 * SOURCE_W * throughput, (field, arrival.beam.power_W, throughput)
        assert arrival.pupil_centroid_uv_mm is not None and arrival.pupil_rms_radius_mm is not None
        assert abs(abs(arrival.pupil_centroid_uv_mm[0]) - abs(float(centroid[0]))) < 0.02, (field, arrival.pupil_centroid_uv_mm, centroid)
        assert arrival.pupil_rms_radius_mm == pytest.approx(rms, rel=0.1), (field, arrival.pupil_rms_radius_mm, rms)
        assert math.isfinite(arrival.aoi_deg) and arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
        if arrival.reference_ray_promoted_at:
            clip = clips[arrival.beam.path_id]
            row = clip.numerical_context["reference_pupil_ray"]
            passes, point, direction = _sequential_reference(csv_name, field, aimed, launch=_launch_offset(row, rings, per_ring, waist))
            assert passes
            assert arrival.beam.position_mm == pytest.approx(tuple(point), abs=1e-6)
            assert arrival.beam.direction == pytest.approx(tuple(direction), abs=1e-9)
    if not aimed:
        assert all(by_field[f"FIELD{k}"].reference_ray_promoted_at for k in range(len(fields)))


# ------------------------------------------------------------------------------------- chief-ray metrics behind it (BND-3)


def test_a_target_on_the_chief_ray_behind_a_promoted_reference_is_not_decided_on_one_pupil_sample():
    """The beam 1.3 mm beside the 1 mm stop, symmetric about the design plane z = 0: the light that passes is symmetric about
    it, but the reference ray is one passing sample (on a tie, one of a mirror pair), whose hit point and direction are not the
    beam's. A target on the chief-ray hit point, direction cosines or angle of incidence behind the stop is out of model scope
    (``CLIPPED``, INDETERMINATE), never decided on that sample: it reports the sample's value -- the arrival's own
    ``uv_mm``, ``direction_uvn`` and ``aoi_deg`` -- with the entry that says one ray stands for the beam. The spot centroid
    is the beam's position and is decided (0 across the symmetry plane, by symmetry); in front of the stop the same targets
    are decided on the chief ray."""
    from os_optics.contracts import Target

    chief = ("position_u_mm", "position_v_mm", "direction_cosine_u", "direction_cosine_v", "aoi_deg")
    targets = [Target(target_id=f"{plane}-{metric}", observation_id=plane, metric=metric, maximum=100.0)
               for plane in ("out", "front") for metric in (*chief, "spot_centroid_u_mm", "spot_centroid_v_mm", "power_W")]
    front = B.observation_plane("front", position=(5.0, 0, 0), normal=(1, 0, 0), radius_mm=50.0)
    project = ProjectSnapshot(name="promoted", components=(_laser(1.0, 633.0, -1.3),
                                                           B.aperture_stop("S1", position=(10, 0, 0), axis=(1, 0, 0), radius_mm=1.0)),
                              observations=(_plane(40.0), front), targets=tuple(targets))
    result = validate(ValidationRequest(project=project))
    arrival = one_arrival(result, "out")
    assert arrival.reference_ray_promoted_at == ("S1/stop",)
    by_id = {t.target_id: t for t in result.targets}
    sample = {"position_u_mm": arrival.uv_mm[0], "position_v_mm": arrival.uv_mm[1], "direction_cosine_u": arrival.direction_uvn[0],
              "direction_cosine_v": arrival.direction_uvn[1], "aoi_deg": arrival.aoi_deg}
    # (the sample is off the beam's axis: at least one of its coordinates is not the symmetric beam's 0)
    assert max(abs(arrival.uv_mm[0]), abs(arrival.uv_mm[1])) > 1e-3
    for metric in chief:
        behind, before = by_id[f"out-{metric}"], by_id[f"front-{metric}"]
        assert behind.status.value == "OUT_OF_MODEL_SCOPE" and behind.assessment.value == "INDETERMINATE", metric
        assert "CLIPPED" in behind.issue_codes and "S1/stop" in behind.message
        assert behind.value == sample[metric], metric
        entry = behind.value_basis["value"]
        assert entry.kind.value == "ESTIMATE", metric
        assert [(a.input.value, a.at, a.code) for a in entry.assumed] == [("SINGLE_RAY_FOR_BEAM", ("S1/stop",), "CLIPPED")], metric
        assert before.assessment.value == "MET" and before.value is not None and before.value_basis is None, metric
    centroid = [by_id["out-spot_centroid_u_mm"], by_id["out-spot_centroid_v_mm"]]
    assert all(t.assessment.value == "MET" for t in centroid)
    # (one of the plane's axes runs across the symmetry plane: the passing light's centroid is 0 there)
    assert min(abs(t.value) for t in centroid) < 1e-9
    assert by_id["out-power_W"].assessment.value == "MET"


def test_a_clipped_reference_whose_power_behind_is_not_computed_states_the_bound_not_a_power():
    """The scene above with the second stop moved so that its opening misses the chief ray (S2 at y = -0.52): the reference
    is promoted there on the sampled basis, whose samples do not follow the beam's diffraction, so the power behind is not
    computed. The CLIPPED notice states the bound it has (at most what reached the stop), never 0 W passing."""
    result = _run(_laser(0.02, 780.0, 0.0), B.aperture_stop("S1", position=(20, -1.005, 0), axis=(1, 0, 0), radius_mm=1.0),
                  B.aperture_stop("S2", position=(50, -0.52, 0), axis=(1, 0, 0), radius_mm=0.5), planes=(_plane(170.0),))
    (clip,) = [i for i in result.issues if i.code.value == "CLIPPED" and i.location == "surface:S2/stop"]
    context = clip.numerical_context
    assert "transmitted_power_W" not in context and "transmitted_fraction" not in context
    (not_computed,) = [i for i in result.issues if i.location == "surface:S2/stop" and "power_upper_bound_W" in i.numerical_context
                       and i.code.value != "CLIPPED"]
    assert context["power_upper_bound_W"] == not_computed.numerical_context["power_upper_bound_W"] > 0.0
    assert "power not computed, at most" in clip.message
