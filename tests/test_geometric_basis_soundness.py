"""Where the geometric pupil-sample power basis is sound, and where it withholds the power.

A beam without a valid Gaussian state carries its power on its equal-power pupil samples (``GEOMETRIC_POWER_BASIS``). The
cases here each had a wrong reading before:

* an absorbing detector with a small active area counted the whole beam when its chief ray fell on the area; it now takes
  the samples' share on the area within their sampling resolution, or withholds the power where no sample falls on it;
* a plane on the sampled basis stated its sampled value as a supported upper bound; it now states the resolution beside
  the value and takes no bound from it (on an absorbing plane the launched power stays a bound);
* the pupil rays of a small beam launched at its waist do not spread with its diffraction, so a later stop cut nothing of
  them; where the rays' picture misses the beam's spread near an edge the power is now withheld;
* an edge between the live samples and an earlier cut's edge passed the beam whole with no resolution;
* the polygon power integral could be a few ulps below zero.

The references are independent: the Gaussian integral over a disc ``1 - exp(-2 a^2 / w^2)``, the free-space Gaussian beam
``w(z) = w0 sqrt(1 + (z / z_R)^2)``, and the energy conservation of an absorbing plane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import assert_power_not_computed, one_arrival
from os_optics import ValidationRequest
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local.record import scientific_projection
from os_optics.validation import validate
from test_geometric_power_basis import SOURCE_W, _two_stops

LAM_MM = 633e-6


def _run(project: ProjectSnapshot):
    return validate(ValidationRequest(project=project))


def _codes(result) -> set[str]:
    return {i.code.value for i in result.issues}


# ------------------------------------------------------------------------------------- a detector's small active area


def _detector(a1: float, area_r: float, target: Target | None = None) -> ProjectSnapshot:
    """1 mW, w = 1 mm, collimated; a stop of ``a1`` w (the Gaussian state is invalid behind 1.0 w or behind the second
    stop), a stop of 0.8 w, and an absorbing detector of active-area radius ``area_r`` on the axis 20 mm later."""
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0), power_W=SOURCE_W)
    return ProjectSnapshot(name="area", components=(
        laser, B.aperture_stop("S1", position=(10, 0, 20), axis=(1, 0, 0), radius_mm=a1),
        B.aperture_stop("S2", position=(20, 0, 20), axis=(1, 0, 0), radius_mm=0.8)),
        observations=(B.observation_plane("det", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=area_r, absorbing=True),),
        targets=() if target is None else (target,))


@pytest.mark.parametrize("a1", (1.0, 1.6))
def test_a_small_active_area_detects_the_samples_share_not_the_whole_beam(a1: float):
    """The area (0.2 mm) takes 1 - exp(-0.08) of the beam: 0.077 mW. Before, the chief ray on it counted all 0.74 / 0.72 mW
    and a 0.3 mW minimum was MET. Now the detected power is the samples' share within its resolution, and that minimum is
    not decided; a 0.6 mW maximum is (the whole interval lies below it)."""
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * 0.2 * 0.2))
    minimum = Target(target_id="p", metric="power_W", observation_id="det", minimum=0.3e-3, unit="W")
    result = _run(_detector(a1, 0.2, minimum))
    arrival = one_arrival(result, "det")
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    assert arrival.detected_power_W < 0.2 * arrival.beam.power_W  # (12 of the 72 live samples)
    assert abs(arrival.detected_power_W - truth) <= arrival.power_sampling_resolution_W
    (got,) = result.targets
    assert got.assessment.value == "INDETERMINATE" and "GEOMETRIC_POWER_BASIS" in got.issue_codes
    maximum = Target(target_id="m", metric="power_W", observation_id="det", maximum=0.6e-3, unit="W")
    (got,) = _run(_detector(a1, 0.2, maximum)).targets
    assert got.assessment.value == "MET"
    assert result.power_budget.conserved


def test_an_active_area_no_sample_falls_on_states_no_power():
    """A 0.05 mm pinhole between the samples: no sample to share the power, so it is not computed -- the plane states no
    power (at most the beam's, which it bounds as an absorbing plane) and a target on it is undecided, not the whole beam.
    The arrival is recorded with its bound."""
    minimum = Target(target_id="p", metric="power_W", observation_id="det", minimum=0.3e-3, unit="W")
    result = _run(_detector(1.0, 0.05, minimum))
    obs, arrival = assert_power_not_computed(result, "det", state="BOUNDED")
    assert obs.power_upper_bound_W == arrival.power_upper_bound_W < SOURCE_W
    assert (result.targets[0].assessment.value, result.targets[0].value) == ("INDETERMINATE", None)


def test_an_active_area_that_holds_the_beam_detects_it_whole():
    """A 5 mm area holds every sample, far beyond a cell: the whole beam, no added resolution."""
    result = _run(_detector(1.0, 5.0))
    arrival = one_arrival(result, "det")
    assert arrival.detected_power_W == arrival.beam.power_W
    reference = one_arrival(_run(_detector(1.0, 50.0)), "det")
    assert arrival.power_sampling_resolution_W == reference.power_sampling_resolution_W


# --------------------------------------------------------------------------------------------- the plane's own state


def test_a_sampled_plane_states_its_resolution_and_no_bound_from_it():
    """The two-stop case (1.0 w, then 0.5 w): the plane's power 0.3706 mW is below the truth 0.3935 mW; it used to be its own
    'supported upper bound'. Now the plane states the resolution beside its KNOWN sampled value, the truth lies within it,
    and no upper bound is taken from it (null on this transparent plane); the record carries the same."""
    result = _run(_two_stops(1.0, 0.5))
    (obs,) = result.observations
    truth = SOURCE_W * (1.0 - math.exp(-0.5))
    arrival = obs.arrivals[0]
    assert obs.power_sampling_resolution_W == arrival.power_sampling_resolution_W > 0.0
    assert abs(obs.total_power_W - truth) <= obs.power_sampling_resolution_W
    assert obs.quantity_state.value == "KNOWN"
    assert "power_upper_bound_W" in obs.model_fields_set and obs.power_upper_bound_W is None
    (quantity,) = scientific_projection(result).observation_quantities
    assert quantity.quantity_state == "KNOWN" and quantity.power_upper_bound_W is None
    assert quantity.power_sampling_resolution_W == obs.power_sampling_resolution_W
    dumped = quantity.model_dump(mode="json")
    assert dumped["power_upper_bound_W"] is None and "power_sampling_resolution_W" in dumped


def test_a_plane_with_exact_power_omits_the_resolution():
    result = _run(_two_stops(1.4, 1.2))
    (obs,) = result.observations
    assert obs.power_sampling_resolution_W is None and "power_sampling_resolution_W" not in obs.model_dump(mode="json")
    assert obs.power_upper_bound_W == obs.total_power_W


def test_an_absorbing_sampled_plane_is_bounded_by_the_launched_power():
    """On an absorbing plane each watt is absorbed once: the launched 1 mW bounds the sampled plane's power, and the record
    accepts that bound above the KNOWN sampled value."""
    project = _two_stops(1.0, 0.5)
    project = project.model_copy(update={"observations": (
        B.observation_plane("after", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=50.0, absorbing=True),)})
    result = _run(project)
    (obs,) = result.observations
    assert obs.power_upper_bound_W == pytest.approx(SOURCE_W, rel=1e-15) and obs.quantity_state.value == "KNOWN"
    (quantity,) = scientific_projection(result).observation_quantities
    assert quantity.power_upper_bound_W == obs.power_upper_bound_W


def test_an_absorbing_plane_bound_never_exceeds_the_launched_power():
    """A plane beside the area the chief ray misses (the stop-filled beam's samples land on it): the untraced light bound
    was 1.6 mW for a 1 mW source; the launched power bounds it."""
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0), power_W=SOURCE_W)
    project = ProjectSnapshot(name="beside", components=(
        laser, B.aperture_stop("S1", position=(10, 0, 20), axis=(1, 0, 0), radius_mm=1.0),
        B.aperture_stop("S2", position=(20, 0, 20), axis=(1, 0, 0), radius_mm=0.8)),
        observations=(B.observation_plane("det", position=(40, 0.5, 20), normal=(1, 0, 0), radius_mm=0.45, absorbing=True),))
    (obs,) = _run(project).observations
    # (no traced beam; only light the probes followed may arrive: between 0 and the bound, no value)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_lower_bound_W == 0.0 and obs.total_power_W is None
    assert obs.power_upper_bound_W is not None and obs.power_upper_bound_W <= SOURCE_W


# -------------------------------------------------------------------------------- rays that do not follow diffraction


def _waist_beam(w0: float, a1: float, a2: float) -> ProjectSnapshot:
    """A 780 nm beam launched at its waist ``w0``: a stop of ``a1`` mm 20 mm on (it cuts the Gaussian severely), a stop of
    ``a2`` mm 30 mm after it, a plane 120 mm after that."""
    laser = B.laser("L", wavelength_nm=780.0, waist_radius_mm=w0, position=(0, 0, 0), direction=(1, 0, 0), power_W=SOURCE_W)
    return ProjectSnapshot(name="waist", components=(
        laser, B.aperture_stop("S1", position=(20, 0, 0), axis=(1, 0, 0), radius_mm=a1),
        B.aperture_stop("S2", position=(50, 0, 0), axis=(1, 0, 0), radius_mm=a2)),
        observations=(B.observation_plane("out", position=(170, 0, 0), normal=(1, 0, 0), radius_mm=40.0),))


def test_rays_launched_at_a_small_waist_do_not_bound_a_later_stop():
    """w0 = 20 um at 780 nm (z_R = 1.6 mm): the rays stay within 25 um while the beam is 0.62 mm at the second stop
    (``w0 sqrt(1 + (z / z_R)^2)``), so that 0.5 mm stop cuts none of them and the plane read 0.72 mW, KNOWN. The spread the
    rays miss is ``B lambda / (pi w0)`` with B = 50 mm: the power behind that stop is not computed (at most what reached
    it), the plane UNKNOWN; the rays go on to it."""
    w0 = 0.02
    result = _run(_waist_beam(w0, 0.2, 0.5))
    _obs, arrival = assert_power_not_computed(result, "out")
    assert arrival.pupil_rays_arrived == arrival.pupil_ray_count  # (the stop cuts none of the rays)
    issue = next(i for i in result.issues if "diffraction_spread_mm" in i.numerical_context)
    spread = 50.0 * 780e-6 / (math.pi * w0)
    z_r = math.pi * w0 * w0 / 780e-6
    beam = w0 * math.sqrt(1.0 + (50.0 / z_r) ** 2)
    assert issue.numerical_context["diffraction_spread_mm"] == pytest.approx(spread, rel=1e-3)
    assert spread == pytest.approx(beam, rel=2e-3)  # (far field: the spread is the beam)
    assert issue.numerical_context["pupil_rms_radius_mm"] < 0.05 * spread
    assert issue.location == "surface:S2/stop"
    # the samples' radius at the plane is the geometric rays' (far below the 120 mm beam there): an estimate, with the check
    (check,) = arrival.value_basis["pupil_rms_radius_mm"].validity
    assert arrival.value_basis["pupil_rms_radius_mm"].kind.value == "ESTIMATE"
    assert (check.parameter, check.within, check.at, check.code) == ("diffraction_spread_mm", False, ("S2/stop",), issue.code.value)
    assert check.observed == issue.numerical_context["diffraction_spread_mm"]
    assert check.maximum == pytest.approx(0.25 * math.sqrt(2.0) * issue.numerical_context["pupil_rms_radius_mm"], rel=1e-12)
    assert arrival.pupil_rms_radius_mm < 0.01 * 170.0 * 780e-6 / (math.pi * w0)


def test_rays_of_a_wide_beam_keep_the_geometric_basis():
    """The control: w0 = 1 mm (z_R = 4 m), the same stops scaled: the rays picture the beam, the sampled power stands within
    its resolution of the Gaussian integral of the second stop."""
    result = _run(_waist_beam(1.0, 1.0, 0.5))
    arrival = one_arrival(result, "out")
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * 0.25))
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES"
    assert abs(arrival.beam.power_W - truth) <= arrival.power_sampling_resolution_W


def test_a_crystal_beam_whose_rays_do_not_spread_is_withheld_at_a_clipping_stop():
    """A 20 um waist through a calcite displacer: its e-beam loses the Gaussian state (GAUSSIAN_ANISOTROPIC_MEDIUM) and its
    rays stay within 4 um of the chief ray while the beam is about 0.57 mm at a 0.5 mm stop 50 mm from the waist. The plane
    read a BOUNDED [0.601, 0.614] mW against a true 0.47 mW; its power there is not computed (at most what reached the
    stop, above the true value) and the plane is UNKNOWN."""
    from declared_helpers import declared_component, stop_model, with_models  # noqa: PLC0415
    from local_helpers import ALONG_X, plane, project, source  # noqa: PLC0415
    from os_optics.local import run_optical_project  # noqa: PLC0415
    from test_project_crystals import CALCITE, displacer  # noqa: PLC0415

    src = source(position=(-20.0, 0.0, 0.0), wavelength_nm=780.0, waist_mm=0.02, polarization_deg=30.0)
    stop = stop_model("ST", opening_radius_mm="0.5", opening_center_mm=["-1.0653", "0"], edge_radius_mm="20")
    comps = [src, declared_component("X0", "prism", "BD1", position=(0.0, 0.0, 0.0), quaternion=ALONG_X),
             declared_component("ST", "aperture", "ST", position=(30.0, 0.0, 0.0), quaternion=ALONG_X),
             declared_component("X2", "prism", "BD2", position=(60.0, 0.0, 0.0), quaternion=ALONG_X)]
    doc = with_models(project(comps, [plane("OUT", (150.0, 0.0, 0.0), radius_mm=40.0)]),
                      [displacer("BD1"), displacer("BD2", axis=("0", "1", "1")), stop], [CALCITE])
    result = run_optical_project(doc, catalog=None).result
    out = next(o for o in result.observations if o.observation_id == "OUT")
    assert out.quantity_state.value == "UNKNOWN" and out.total_power_W is None
    assert out.arrivals and all(a.beam.power_W is None and a.power_upper_bound_W > 0.47e-3 for a in out.arrivals)
    issue = next(i for i in result.issues if "diffraction_spread_mm" in i.numerical_context)
    assert issue.code.value == "GAUSSIAN_ANISOTROPIC_MEDIUM"
    assert issue.numerical_context["diffraction_spread_mm"] > 0.4


# ------------------------------------------------------------------------------- an edge between the live samples


def test_an_edge_between_the_live_samples_and_an_earlier_cut_is_in_the_resolution():
    """1.0 w, then a concentric 0.95 w stop: it lies between the outermost live ring (0.915 w) and the first stop's edge,
    cuts no sample, and the beam passed whole with no resolution (3.4 % high). Now the ring's share is in the resolution and
    the Gaussian integral ``1 - exp(-2 0.95^2)`` lies within it."""
    result = _run(_two_stops(1.0, 0.95))
    arrival = one_arrival(result, "after")
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * 0.95 * 0.95))
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES" and "GEOMETRIC_POWER_BASIS" in _codes(result)
    assert abs(arrival.beam.power_W - truth) <= arrival.power_sampling_resolution_W
    assert abs(arrival.beam.power_W - truth) > 0.02 * truth  # (the sampled value itself misses it)


def test_an_edge_beyond_a_cell_of_the_live_samples_adds_nothing():
    """A second stop of 1.6 w lies farther from the outermost live samples than their spacing: no resolution, the exact
    power of the first stop's Gaussian cut."""
    result = _run(_two_stops(1.0, 1.6))
    arrival = one_arrival(result, "after")
    assert arrival.power_basis is None
    # (the first stop's Gaussian integral, at its quadrature's accuracy as in test_geometric_power_basis)
    assert arrival.beam.power_W == pytest.approx(SOURCE_W * (1.0 - math.exp(-2.0)), abs=1e-5 * SOURCE_W)


# ----------------------------------------------------------------------------------------------- the polygon integral


def test_the_polygon_power_integral_is_never_negative():
    """A triangle far beside a narrow beam: the boundary terms cancel to a few ulps, which could be below zero; the region's
    integral of a positive Gaussian lies between 0 and the whole plane's ``pi / (2 sqrt(det S))``."""
    from os_optics.config import NumericalConfig  # noqa: PLC0415
    from os_optics.optics.gaussian import aperture_regions  # noqa: PLC0415
    from os_optics.optics.gaussian.truncation import gauss_legendre_rule  # noqa: PLC0415
    from os_optics.scene.model import ApertureG  # noqa: PLC0415

    cfg = NumericalConfig()
    rule = gauss_legendre_rule(cfg.aperture_quad_sector_nodes)
    tri = ApertureG(shape="polygon", vertices=((0.0, 0.0), (3.0, 0.1), (0.2, 2.5)))
    negative = 0
    for w in (0.05, 0.1):
        for cu, cv in ((1.0, -1.2), (-1.4, 1.0), (4.0, 3.0), (-1.5, 0.3)):
            s = np.diag([1.0 / (w * w), 1.0 / (w * w)])
            det = 1.0 / w ** 4
            raw = aperture_regions._outer_integral(tri, s, det, (cu, cv), cfg, rule)
            negative += raw < 0.0
            value = aperture_regions.region_integral(tri, s, det, (cu, cv), cfg, rule)
            assert 0.0 <= value <= math.pi / (2.0 * math.sqrt(det))
            assert value == max(raw, 0.0)
    assert negative  # (the quadrature does give such values)


# ------------------------------------------------------------------- an edge that holds an opening the beam passed earlier


def _stop_then(f_mm: float | None, a2: float) -> ProjectSnapshot:
    """``_two_stops``' collimated 1 mW beam (w = 1 mm) through a 1.0 w stop at x = 10 (the Gaussian integral cuts 13.5 % and
    the state is invalid behind it), an optional thin lens of focal length ``f_mm`` (clear radius 5 mm) at x = 12, and a
    stop of ``a2`` w at x = 20."""
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0), power_W=SOURCE_W)
    parts = [laser, B.aperture_stop("S1", position=(10, 0, 20), axis=(1, 0, 0), radius_mm=1.0)]
    if f_mm is not None:
        parts.append(B.thin_lens("LENS", focal_length_mm=f_mm, clear_radius_mm=5.0, position=(12, 0, 20)))
    parts.append(B.aperture_stop("S2", position=(20, 0, 20), axis=(1, 0, 0), radius_mm=a2))
    return ProjectSnapshot(name="held", components=tuple(parts),
                           observations=(B.observation_plane("after", position=(40, 0, 20), normal=(1, 0, 0), radius_mm=50.0),))


@pytest.mark.parametrize("f_mm", (None, 100.0))
def test_a_stop_that_holds_the_earlier_opening_adds_no_resolution(f_mm: float | None):
    """The same 1.0 w stop met again (rays parallel), or behind a converging lens (the opening's image shrinks to 0.92 w):
    all light that passed the first stop lies inside the second, which cuts nothing. Before, the outer ring's whole share
    (12 of 84 samples, 0.124 mW) was added as resolution; now the power is the first stop's Gaussian cut with none."""
    result = _run(_stop_then(f_mm, 1.0))
    arrival = one_arrival(result, "after")
    assert arrival.power_basis is None and arrival.power_sampling_resolution_W is None
    assert "GEOMETRIC_POWER_BASIS" not in _codes(result)
    assert arrival.beam.power_W == pytest.approx(SOURCE_W * (1.0 - math.exp(-2.0)), abs=1e-5 * SOURCE_W)


def test_a_stop_the_beam_spreads_beyond_keeps_the_resolution():
    """Behind a diverging lens the first opening's image (1.08 w) reaches past the second stop's edge, which cuts the light
    between 1 / 1.08 w and 1 w of the first stop: the frontier rule keeps the outer ring's share, and the geometric truth
    ``1 - exp(-2 (1 / 1.08)^2)`` lies within it."""
    result = _run(_stop_then(-100.0, 1.0))
    arrival = one_arrival(result, "after")
    assert arrival.power_basis == "GEOMETRIC_PUPIL_SAMPLES" and "GEOMETRIC_POWER_BASIS" in _codes(result)
    truth = SOURCE_W * (1.0 - math.exp(-2.0 / (1.08 * 1.08)))
    assert abs(arrival.beam.power_W - truth) <= arrival.power_sampling_resolution_W


def _nbk7_index(wavelength_um: float) -> float:
    """Schott's N-BK7 Sellmeier formula."""
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    l2 = wavelength_um * wavelength_um
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _plate(hole: tuple[str, str] | None, clear: str):
    """R2's plate: a 3 mm N-BK7 window (uncoated; the front face's reflection traced), a 780 nm 1 mW beam of 0.5 mm waist on
    the window's front face; a clear aperture of ``clear`` mm on both faces, or a hole at ``(x, 0)`` of radius ``r`` (mm)
    through both. Planes OUT 100 mm behind, BACK 50 mm before."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models  # noqa: PLC0415
    from local_helpers import AGAINST_X, ALONG_X, plane, project, source  # noqa: PLC0415
    from os_optics.local import run_optical_project  # noqa: PLC0415

    extra = {} if hole is None else {"holes": [{"center_mm": [hole[0], "0"], "radius_mm": hole[1]}]}
    s1 = surface(1, "0", clear=clear, traced="TRANSMITTED_AND_REFLECTED", **extra)
    s2 = surface(2, "3", clear=clear, **extra)
    model = {"model_id": "W", "model": sequence([s1, s2], ["N-BK7"])}
    src = source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0, waist_mm=0.5, waist_position_mm=100.0)
    comps = [src, declared_component("W", "window", "W", position=(0.0, 0.0, 0.0), quaternion=ALONG_X)]
    obs = [plane("OUT", (100.0, 0.0, 0.0), radius_mm=40.0), plane("BACK", (-50.0, 0.0, 0.0), AGAINST_X, radius_mm=40.0)]
    return run_optical_project(with_models(project(comps, obs), [model], [nbk7()]), catalog=None).result


def test_a_plate_whose_faces_share_one_clear_aperture_reads_the_exact_power():
    """R2's ``p3b_noholes.py 0 0.4``: the front face's 0.4 mm aperture cuts the 0.5 mm beam (27.8 %, the state is lost), the
    back face's equal aperture 3 mm later cuts nothing of what passed. Before, its edge added the outer ring (0.111 mW, 17 %)
    and the transparent plane fell from BOUNDED to UNKNOWN; now OUT reads the Gaussian cut times the two faces' Fresnel
    transmission, ``(1 - exp(-2 a^2 / w^2)) (1 - R)^2``, BOUNDED by the light the trace does not follow."""
    from os_optics.validation.outcome import quantity_state_of  # noqa: PLC0415

    result = _plate(None, "0.4")
    out = next(o for o in result.observations if o.observation_id == "OUT")
    n = _nbk7_index(0.78)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    truth = SOURCE_W * (1.0 - math.exp(-2.0 * 0.4 * 0.4 / 0.25)) * (1.0 - r) ** 2
    assert out.power_sampling_resolution_W is None
    assert out.total_power_W == pytest.approx(truth, rel=1e-5)
    assert quantity_state_of(out).value == "BOUNDED"
    # (the bound: traced + untraced light + the light left out as negligible, picometre-watts here)
    assert 0.0 <= out.power_upper_bound_W - (out.total_power_W + out.untraced_power_W) <= 1e-8 * SOURCE_W
    assert "GEOMETRIC_POWER_BASIS" not in _codes(result)


def _hole_share(cx: float, radius: float, w: float, cells: int = 1601) -> float:
    """The share of a Gaussian beam of 1/e^2 radius ``w`` (centred at 0) inside the disc of ``radius`` about ``(cx, 0)``: a
    midpoint sum over a square grid covering the disc."""
    xs = np.linspace(cx - radius, cx + radius, cells)
    ys = np.linspace(-radius, radius, cells)
    h = xs[1] - xs[0]
    x, y = np.meshgrid(xs, ys)
    inside = (x - cx) ** 2 + y * y <= radius * radius
    return float(np.sum(np.exp(-2.0 * (x * x + y * y) / (w * w))[inside]) * h * h * 2.0 / (math.pi * w * w))


def test_light_through_a_hole_is_counted_once_on_a_plane_beyond():
    """R2 m3 (``p3_holes_ghosts.py 0.7 0.6``): the light through a hole under the beam's edge (27.5 %) left the traced beam at
    the front face and is followed by its probe to OUT -- and it was counted a second time as part of the leg reaching OUT
    through the hole, as the whole 1 mW: untraced 1.276 mW for a 1 mW source. Now it is counted once: OUT's untraced light
    is the hole's share plus the ghost light, and with the traced power it stays below the launched power."""
    result = _plate(("0.7", "0.6"), "9")
    out = next(o for o in result.observations if o.observation_id == "OUT")
    share = SOURCE_W * _hole_share(0.7, 0.6, 0.5)
    assert share * (1.0 - 1e-3) <= out.untraced_power_W <= share + 0.01 * SOURCE_W
    assert out.total_power_W + out.untraced_power_W <= SOURCE_W


def test_a_beam_whose_chief_ray_passes_a_hole_reads_the_plane_within_the_share_the_plate_takes():
    """R2 m3, ``p3_holes_ghosts.py 0.2 0.6``: the chief ray passes the hole (centre 0.2 mm, radius 0.6 mm) while part of the
    0.5 mm beam meets the plate around it. The plane's power follows the chief ray (the whole beam); before R2 the run read
    PASS with no issue, then the plane was out of model scope. Since the cell bound of a surface beside the chief ray covers
    supports with holes (``lower_share.possible_meeting_share``), the plate's faces state the share they may take and the plane
    reads ``[P - S, P]`` (BOUNDED, ``power_taken_before_at``). Independent reference: the Gaussian share outside the hole
    (2-D grid), which crosses the uncoated 3 mm plate at normal incidence with ``(1 - R)^2`` and reaches the plane; the light
    through the hole reaches it whole: the plane's true power lies in the stated interval."""
    from os_optics.validation.outcome import quantity_state_of  # noqa: PLC0415

    result = _plate(("0.2", "0.6"), "9")
    out = next(o for o in result.observations if o.observation_id == "OUT")
    (arrival,) = out.arrivals
    assert quantity_state_of(out).value == "BOUNDED" and arrival.power_out_of_scope_at == ()
    assert "W/S1" in arrival.power_taken_before_at and arrival.pupil_rays_arrived < arrival.pupil_ray_count
    w, h = 0.5, 0.002
    grid = np.arange(-3.0, 3.0, h) + h / 2
    x, y = np.meshgrid(grid, grid)
    irradiance = 2.0 / (math.pi * w * w) * np.exp(-2.0 * (x * x + y * y) / (w * w))
    outside = float(irradiance[(x - 0.2) ** 2 + y ** 2 > 0.36].sum() * h * h)
    assert 0.10 < outside < 0.12
    n = _nbk7_index(0.78)
    truth = SOURCE_W * ((1.0 - outside) + outside * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2) ** 2)
    assert arrival.power_taken_before_upper_W >= SOURCE_W * outside
    assert out.power_lower_bound_W <= truth <= out.total_power_W == pytest.approx(SOURCE_W, rel=1e-12)


# --------------------------------------------------------------- a sampled value's resolution is not a bound (wording)


def test_a_target_on_a_sampled_plane_says_its_interval_is_the_resolution_not_a_bound():
    """RR N4 (``p_sampled_target_vs_plane.py``): the plane states no upper bound from its sampled value, and the maximum target
    is decided outside value plus or minus the resolution. The target's message now says that interval is the sampling
    resolution (convergence evidence), not a bound; the plane's bound field says the same."""
    from os_optics.contracts.results import ObservationResult  # noqa: PLC0415

    maximum = Target(target_id="max", metric="power_W", observation_id="after", maximum=0.7e-3, unit="W")
    result = _run(_two_stops(1.0, 0.5, targets=(maximum,)))
    (plane,) = result.observations
    (got,) = result.targets
    assert plane.power_upper_bound_W is None and plane.power_sampling_resolution_W is not None
    assert got.assessment.value == "MET"
    assert "not a proven bound" in got.message and "within that resolution the plane receives" in got.message
    assert "behind an aperture that cut them" not in got.message
    described = ObservationResult.model_fields["power_upper_bound_W"].description
    assert "convergence evidence, not a bound" in described


def test_the_affine_image_test_is_sound_for_every_shape():
    """``apertures.holds_affine_image``: whenever it says the outer region holds the image of the inner one, every point of
    the inner region (a dense grid of it, holes left out) maps inside the outer region -- the brute-force reference -- for
    each shape pair under shrinking, identity, growing, shifted and sheared maps; the identity and a shrink of an equal
    convex region are held, a growth is not."""
    from os_optics.optics import apertures  # noqa: PLC0415
    from os_optics.scene.model import ApertureG  # noqa: PLC0415

    hole = ApertureG(shape="circle", p1=0.3, offset_u=0.4)
    shapes = {
        "circle": ApertureG(shape="circle", p1=1.0),
        "rectangle": ApertureG(shape="rectangle", p1=1.0, p2=0.7),
        "ellipse": ApertureG(shape="ellipse", p1=1.0, p2=0.6, axis=(0.6, 0.8)),
        "d_shape": ApertureG(shape="d_shape", p1=1.0, p2=0.5, axis=(1.0, 0.0)),
        "polygon": ApertureG(shape="polygon", vertices=((-1.0, -0.8), (1.0, -0.6), (0.7, 0.9), (-0.9, 0.7))),
        "holed": ApertureG(shape="circle", p1=1.0, holes=(hole,)),
    }
    maps = {"shrink": ((0.9, 0.0, 0.0, 0.9), (0.0, 0.0)), "identity": ((1.0, 0.0, 0.0, 1.0), (0.0, 0.0)),
            "grow": ((1.05, 0.0, 0.0, 1.05), (0.0, 0.0)), "shift": ((0.95, 0.0, 0.0, 0.95), (0.04, -0.03)),
            "shear": ((0.9, 0.05, -0.04, 0.92), (0.01, 0.0))}
    grid = np.linspace(-1.2, 1.2, 241)
    gu, gv = (a.ravel() for a in np.meshgrid(grid, grid))
    for inner_name, inner in shapes.items():
        points = apertures.inside(inner, gu, gv, 0.0)
        pu, pv = gu[points], gv[points]
        for outer_name, outer in shapes.items():
            for map_name, (m, b) in maps.items():
                held = apertures.holds_affine_image(outer, inner, m, b, 0.0, 1e-9)
                if held:
                    qu, qv = m[0] * pu + m[1] * pv + b[0], m[2] * pu + m[3] * pv + b[1]
                    assert bool(apertures.inside(outer, qu, qv, 1e-9).all()), (inner_name, outer_name, map_name)
    # (a d_shape is held through its disc, so only a region holding that disc holds it: a sufficient test; a holed region
    # shrunk moves its hole's image off the hole, so the light near the hole's edge may fall in the hole: not held)
    for name, region in shapes.items():
        assert apertures.holds_affine_image(region, region, *maps["identity"], 0.0, 1e-9) == (name != "d_shape"), name
        assert apertures.holds_affine_image(region, region, *maps["shrink"], 0.0, 1e-9) == (name not in ("d_shape", "holed")), name
        assert not apertures.holds_affine_image(region, region, *maps["grow"], 0.0, 1e-9), name
