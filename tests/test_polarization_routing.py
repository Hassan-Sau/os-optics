"""Polarization-resolved routing of beam lines along a known path (``optics.polarization.routing``).

Independent check: for each representative design the routed share of every branch -- the product of the elements'
shares along the nominal path, computed without tracing -- equals the power the Engine's tracer delivers to that
branch's observation plane in a local run of the same design (relative 1e-9). Designs: a half-wave plate before a
polarizing cube splitter at several plate angles, an s-polarized input to the splitter, crossed polarizers, the shipped
``polarization`` example, a fold mirror turning the plane of incidence before the splitter, a fold mirror whose declared
coating states its reflection phases ahead of a polarizer with transmittance bands, and (Engine request level) a
circularly polarized source through a quarter-wave plate.
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from os_optics import builders as B
from os_optics._vendor.shared_contracts.design_catalog import OpticalForm, PolarizationElementModelV1
from os_optics.contracts import JonesVector, ProjectSnapshot
from os_optics.local import example as ex
from os_optics.local import json_api
from os_optics.local import example_project_of_kind, load_public_catalog, lower_project, run_optical_project
from os_optics.optics.polarization.routing import (
    POLARIZING_RATIO,
    ROUTE_PASS,
    ROUTE_SPLIT_REFLECT,
    LineState,
    PolarizationRouter,
)

from declared_helpers import BAND, declared_component, with_models
from helpers import one_arrival, run

POWER_W = ex.SOURCE_POWER_W
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _turn(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


@pytest.fixture
def catalog(real_catalog_store):
    """The repository's seed catalog, active for the local runs of the test."""

    return load_public_catalog()


@pytest.fixture
def hwp_and_pbs(catalog):
    """A zero-order half-wave plate and a polarizing cube splitter whose splitting coating covers the plate's design
    wavelength."""

    cubes = [p for p in catalog.products if p.form is OpticalForm.POLARIZING_CUBE_BEAMSPLITTER]
    for plate in sorted(catalog.products, key=lambda p: p.optical_sku):
        model = ex._model(plate)
        if plate.form is not OpticalForm.HALF_WAVE_PLATE or not isinstance(model, PolarizationElementModelV1):
            continue
        wavelength = Decimal(model.design_wavelength_nm)
        if not ex._covers(plate, catalog, wavelength, ex.NORMAL_AOI_DEG):
            continue
        for cube in sorted(cubes, key=lambda p: p.optical_sku):
            coatings = [catalog.coating(s.coating_id) for s in cube.simulation_models[0].surfaces]
            if all(c.band_at(wavelength) is not None for c in coatings):
                return plate, cube, float(wavelength)
    raise AssertionError("no half-wave plate with a polarizing cube splitter at its design wavelength")


def _routed(project, chain):
    """``chain``: (element, route, point, d_in, d_out) per step from the source; the product of the shares and the
    leaving state."""

    router = PolarizationRouter(lower_project(project.payload(), load_public_catalog()).request.project)
    source = next(c for c in project.components if c.source_model is not None)
    state = router.source_state(source.instance_id)
    share = 1.0
    for element, route, point, d_in, d_out in chain:
        passage = router.step(element, route, source.source_model.wavelengths_nm[0], point, d_in, d_out, state)
        assert passage is not None, element
        share, state = share * passage.share, passage.state
    return share, state


def _traced(project, directions: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """Compare the router to the trace's *upper* power where real catalog substrate loss is unstated.

    A lossless routing share is an upper bound, not a measured transmitted power; the ray geometry remains exact.
    """
    run = run_optical_project(project)
    traced = {}
    for observation in run.result.observations:
        assert observation.observation_id in directions
        powers = []
        for arrival in observation.arrivals:
            np.testing.assert_allclose(arrival.beam.direction, directions[observation.observation_id], atol=1e-12)
            if arrival.beam.power_W is None:
                assert observation.quantity_state.value == "UNKNOWN"
                assert arrival.power_upper_bound_W is not None and 0.0 <= arrival.power_upper_bound_W <= POWER_W
                powers.append(arrival.power_upper_bound_W)
            else:
                powers.append(arrival.beam.power_W)
        traced[observation.observation_id] = sum(powers) / POWER_W
    return traced


def _splitter_design(hwp_and_pbs, catalog, theta_deg: float | None, *, source_deg: float = 0.0):
    plate, cube, wavelength = hwp_and_pbs
    components = [ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=wavelength, polarization_deg=source_deg)]
    if theta_deg is not None:
        components.append(ex._element("HWP", "waveplate", plate, catalog, [-50.0, 0.0, 0.0], ex._quat_multiply(ex.BEAM_ALONG_X, _turn(theta_deg))))
    components.append(ex._element("PBS", "splitter", cube, catalog, [0.0, 0.0, 0.0], ex.BEAM_ALONG_X))
    return ex._project(components, [ex._plane("out_T", [60.0, 0.0, 0.0], ex.BEAM_ALONG_X),
                                    ex._plane("out_R", [0.0, 60.0, 0.0], ex.BEAM_ALONG_Y)], [])


@pytest.mark.parametrize("theta", [0.0, 10.0, 22.5, 30.0, 45.0, 67.5])
def test_hwp_and_pbs_routed_shares_equal_the_traced_branch_powers(hwp_and_pbs, catalog, theta):
    project = _splitter_design(hwp_and_pbs, catalog, theta)
    traced = _traced(project, {"out_T": X, "out_R": Y})
    hwp = ("HWP", ROUTE_PASS, (-50.0, 0.0, 0.0), X, X)
    t, _ = _routed(project, [hwp, ("PBS", ROUTE_PASS, (0.0, 0.0, 0.0), X, X)])
    r, _ = _routed(project, [hwp, ("PBS", ROUTE_SPLIT_REFLECT, (0.0, 0.0, 0.0), X, Y)])
    assert t == pytest.approx(traced["out_T"], rel=1e-9, abs=1e-15)
    assert r == pytest.approx(traced["out_R"], rel=1e-9, abs=1e-15)


def test_an_s_polarized_input_goes_down_the_reflected_arm(hwp_and_pbs, catalog):
    # the source's local +X is design +Y; 90 deg turns it to design +Z: perpendicular to the cube's plane of incidence
    project = _splitter_design(hwp_and_pbs, catalog, None, source_deg=90.0)
    traced = _traced(project, {"out_T": X, "out_R": Y})
    t, _ = _routed(project, [("PBS", ROUTE_PASS, (0.0, 0.0, 0.0), X, X)])
    r, state = _routed(project, [("PBS", ROUTE_SPLIT_REFLECT, (0.0, 0.0, 0.0), X, Y)])
    assert t == pytest.approx(traced["out_T"], rel=1e-9) and t < 1e-3
    assert r == pytest.approx(traced["out_R"], rel=1e-9) and r > 0.9
    field = np.array(state.field())
    assert abs(field[2]) == pytest.approx(1.0, abs=1e-12)  # still vertical (s) on the reflected arm


def test_the_shipped_polarization_example_routes_its_traced_power(catalog):
    project = example_project_of_kind("polarization")
    traced = _traced(project, {"detector": X})
    share, _ = _routed(project, [("HWP", ROUTE_PASS, (0.0, 0.0, 0.0), X, X), ("POL", ROUTE_PASS, (50.0, 0.0, 0.0), X, X)])
    assert share == pytest.approx(traced["detector"], rel=1e-9)


def test_crossed_polarizers_pass_only_their_leak(catalog):
    base = example_project_of_kind("polarization")
    payload = base.payload()
    polarizer = next(c for c in payload["components"] if c["instance_id"] == "POL")
    first = {**polarizer, "instance_id": "POL1", "nominal_pose": {**polarizer["nominal_pose"], "quaternion_wxyz": list(ex.BEAM_ALONG_X)}}
    second = {**polarizer, "instance_id": "POL2",
              "nominal_pose": {**polarizer["nominal_pose"], "quaternion_wxyz": ex._quat_multiply(ex.BEAM_ALONG_X, _turn(90.0))}}
    payload["components"] = [c for c in payload["components"] if c["instance_id"] not in ("HWP", "POL")] + [
        {**first, "nominal_pose": {**first["nominal_pose"], "position_mm": [0.0, 0.0, 0.0]}}, second]
    payload["targets"] = []
    project = type(base).model_validate(payload)
    traced = _traced(project, {"detector": X})
    share, _ = _routed(project, [("POL1", ROUTE_PASS, (0.0, 0.0, 0.0), X, X), ("POL2", ROUTE_PASS, (50.0, 0.0, 0.0), X, X)])
    assert share == pytest.approx(traced["detector"], rel=1e-9)
    assert share < 1e-3  # t_par * t_par / extinction ratio: below the display rule's blocking share


def test_a_fold_mirror_turning_the_plane_of_incidence_is_followed(hwp_and_pbs, catalog):
    """SRC along +X, polarized horizontally (along +Y) -> 45 deg fold mirror into +Y -> PBS whose reflected port points up
    (+Z). Behind the mirror the field lies along X, perpendicular to the cube's plane of incidence (Y-Z): s, so the light
    goes up."""

    plate, cube, wavelength = hwp_and_pbs
    mirror = ex._first(catalog, ex.OpticalCategory.MIRROR, OpticalForm.FLAT_MIRROR, Decimal(repr(wavelength)), ex.FOLD_AOI_DEG)
    up_reflecting = ex._quat_multiply(ex.BEAM_ALONG_Y, _turn(90.0))  # local +Z -> +Y, local +X (OUT_R) -> +Z
    project = ex._project(
        [ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=wavelength),
         ex._element("M1", "mirror", mirror, catalog, [0.0, 0.0, 0.0], ex.FOLD_X_TO_Y),
         ex._element("PBS", "splitter", cube, catalog, [0.0, 60.0, 0.0], up_reflecting)],
        [ex._plane("out_T", [0.0, 120.0, 0.0], ex.BEAM_ALONG_Y),
         ex._plane("out_R", [0.0, 60.0, 60.0], (1.0, 0.0, 0.0, 0.0))], [])
    traced = _traced(project, {"out_T": Y, "out_R": Z})
    fold = ("M1", ROUTE_PASS, (0.0, 0.0, 0.0), X, Y)
    t, _ = _routed(project, [fold, ("PBS", ROUTE_PASS, (0.0, 60.0, 0.0), Y, Y)])
    r, _ = _routed(project, [fold, ("PBS", ROUTE_SPLIT_REFLECT, (0.0, 60.0, 0.0), Y, Z)])
    assert t == pytest.approx(traced["out_T"], rel=1e-9, abs=1e-15)
    assert r == pytest.approx(traced["out_R"], rel=1e-9, abs=1e-15)
    assert r > 0.9 > 1e-3 > t


def _phased_fold_before_polarizer(phase: dict | None, polarizer_deg: float) -> dict:
    """A 45 deg linear source along +X, a fold mirror into +Y whose declared coating (R = 0.9; fixture data) states
    ``phase``, a declared polarizer with transmittance bands (0.5 at 400-700 nm, 0.8 at 700-1100 nm: the 780 nm line
    takes 0.8) turned by ``polarizer_deg`` about the beam, and the plane OUT."""

    built = json_api.build_native_model("mirror.flat", {"reflectance": "0.9", "valid_band_nm": ["400", "1100"], "diameter_mm": "25.4"})
    assert built["ok"], built
    mirror, (coating,) = built["optical_model"], built["model_coatings"]
    if phase is not None:
        coating = {**coating, "phase": phase}
    value = lambda v: {"value": v, "status": "SPECIFIED"}  # noqa: E731
    polarizer = {"model_id": "pol", "model": {
        "model_kind": "POLARIZATION_ELEMENT", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
        "transmittance": value("0.5"), "extinction_ratio": value("1000"), "clear_radius_mm": "10", "edge_radius_mm": "12.7",
        "valid_band_nm": dict(BAND), "valid_aoi_deg": {"min_deg": "0", "max_deg": "10", "status": "SPECIFIED"},
        "transmittance_bands": [{"min_nm": "400", "max_nm": "700", "transmittance": value("0.5"), "status": "SPECIFIED"},
                                {"min_nm": "700", "max_nm": "1100", "transmittance": value("0.8"), "status": "SPECIFIED"}],
        "approximations": ["THIN_ELEMENT"], "status": "SPECIFIED"}}
    src = ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=780.0, polarization_deg=45.0)
    document = {"schema": "opensci.project", "project_schema_version": "2.0", "targets": [],
                "components": [src, declared_component("M", "mirror", mirror["model_id"], quaternion=ex.FOLD_X_TO_Y),
                               declared_component("P", "polarizer", "pol", (0.0, 50.0, 0.0), ex._quat_multiply(ex.BEAM_ALONG_Y, _turn(polarizer_deg)))],
                "observations": [ex._plane("OUT", [0.0, 100.0, 0.0], ex.BEAM_ALONG_Y)]}
    return with_models(document, [mirror, polarizer], None, [coating])


QUARTER_WAVE_PHASES = {"reflection_s_rad": "0", "reflection_p_rad": "1.5707963267948966", "status": "DECLARED"}


@pytest.mark.parametrize("phase", [None, QUARTER_WAVE_PHASES], ids=["mirror-phases", "stated-phases"])
@pytest.mark.parametrize("polarizer_deg", [0.0, 30.0, 45.0, 90.0])
def test_stated_coating_phases_and_polarizer_bands_route_as_traced(phase, polarizer_deg):
    """The routed share through the fold mirror and the polarizer equals the traced power at OUT (relative 1e-9): the router
    applies the coating's stated reflection phases and the polarizer's band at the line's wavelength as the tracer does."""

    document = _phased_fold_before_polarizer(phase, polarizer_deg)
    result = run_optical_project(document).result
    traced = sum(a.beam.power_W for o in result.observations for a in o.arrivals) / POWER_W
    router = PolarizationRouter(lower_project(document, None).request.project)
    mirror = router.step("M", ROUTE_PASS, 780.0, (0.0, 0.0, 0.0), X, Y, router.source_state("SRC"))
    polarizer = router.step("P", ROUTE_PASS, 780.0, (0.0, 50.0, 0.0), Y, Y, mirror.state)
    assert mirror.share * polarizer.share == pytest.approx(traced, rel=1e-9, abs=1e-15)


def test_the_stated_phases_change_the_routed_share():
    """At a polarizer turned 45 deg from s and p the mirror's own (pi, 0) phases keep the reflected light linear (share
    ~0 or ~1 of 0.9 x 0.8), the stated quarter-wave difference makes it circular (share 0.9 x 0.8 / 2)."""

    shares = {}
    for name, phase in (("mirror", None), ("stated", QUARTER_WAVE_PHASES)):
        router = PolarizationRouter(lower_project(_phased_fold_before_polarizer(phase, 45.0), None).request.project)
        mirror = router.step("M", ROUTE_PASS, 780.0, (0.0, 0.0, 0.0), X, Y, router.source_state("SRC"))
        shares[name] = mirror.share * router.step("P", ROUTE_PASS, 780.0, (0.0, 50.0, 0.0), Y, Y, mirror.state).share
    assert shares["stated"] == pytest.approx(0.9 * 0.8 * (1.0 + 1.0 / 1000.0) / 2.0, rel=1e-9)
    assert min(abs(shares["mirror"]), abs(shares["mirror"] - 0.9 * 0.8)) < 1e-3


def test_a_tabulated_thin_lens_routes_its_transmittance_at_the_line_wavelength():
    """The 780 nm line through a tabulated thin lens (T = 0.85 at 760 nm, 0.9 at 800 nm, linear; fixture data) takes
    T(780) = 0.875, as the tracer delivers to the plane."""

    evidence = {"method": "fixture response", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "0" * 64,
                "fit_model": "fixture", "validation_summary": "fixture", "date": "2026-10-15"}
    lens = {"model_id": "ml", "model": {
        "model_kind": "TABULATED_THIN_LENS", "model_version": "1", "status": "DERIVED", "evidence": evidence,
        "rows": [{"wavelength_nm": "760", "focal_length_mm": "103", "transmittance": "0.85"},
                 {"wavelength_nm": "800", "focal_length_mm": "98", "transmittance": "0.9"}],
        "focal_interpolation": "LINEAR_IN_OPTICAL_POWER", "transmittance_interpolation": "LINEAR", "clear_radius_mm": "10",
        "edge_radius_mm": "12.7", "max_field_angle_deg": "5",
        "supported_metrics": [{"metric": "transmittance", "abs_error": "0.05", "basis": "fixture"}]}}
    document = with_models({"schema": "opensci.project", "project_schema_version": "2.0", "targets": [],
                            "components": [ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=780.0),
                                           declared_component("ML", "lens", "ml", (0.0, 0.0, 0.0), ex.BEAM_ALONG_X)],
                            "observations": [ex._plane("OUT", [50.0, 0.0, 0.0], ex.BEAM_ALONG_X)]}, [lens])
    traced = sum(a.beam.power_W for o in run_optical_project(document).result.observations for a in o.arrivals) / POWER_W
    router = PolarizationRouter(lower_project(document, None).request.project)
    passage = router.step("ML", ROUTE_PASS, 780.0, (0.0, 0.0, 0.0), X, X, router.source_state("SRC"))
    assert passage.share == pytest.approx(0.875, rel=1e-12) and passage.share == pytest.approx(traced, rel=1e-9)


# ------------------------------------------------------------------------------------- Engine request level designs

def _circular_qwp_pbs(handed: JonesVector, fast_axis_deg: float) -> ProjectSnapshot:
    inv = 1.0 / math.sqrt(2.0)
    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=635.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3,
                    polarization=handed),
            B.quarter_waveplate("QWP", position=(20, 0, 20), axis=(1, 0, 0), fast_axis_angle_deg=fast_axis_deg, design_wavelength_nm=635.0),
            B.pbs("PBS", position=(60, 0, 20), normal=(-inv, inv, 0)),
        ),
        observations=(B.observation_plane("T", position=(120, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("R", position=(60, 60, 20), normal=(0, 1, 0), radius_mm=10.0)),
    )


@pytest.mark.parametrize("handed", [JonesVector.right_circular(), JonesVector.left_circular(), JonesVector.elliptical(20.0, 30.0)])
@pytest.mark.parametrize("fast_axis_deg", [45.0, -45.0])
def test_a_circular_or_elliptical_source_through_a_quarter_wave_plate(handed, fast_axis_deg):
    project = _circular_qwp_pbs(handed, fast_axis_deg)
    result = run(project)
    powers = {o.observation_id: (o.arrivals[0].beam.power_W if o.arrivals else 0.0) / 1e-3 for o in result.observations}
    router = PolarizationRouter(project)
    state = router.source_state("L")
    qwp = router.step("QWP", ROUTE_PASS, 635.0, (20, 0, 20), X, X, state)
    t = router.step("PBS", ROUTE_PASS, 635.0, (60, 0, 20), X, X, qwp.state)
    r = router.step("PBS", ROUTE_SPLIT_REFLECT, 635.0, (60, 0, 20), X, Y, qwp.state)
    assert qwp.share * t.share == pytest.approx(powers["T"], abs=1e-12)
    assert qwp.share * r.share == pytest.approx(powers["R"], abs=1e-12)


def test_circular_light_becomes_linear_and_takes_one_arm():
    t = run(_circular_qwp_pbs(JonesVector.right_circular(), 45.0))
    shares = sorted((o.arrivals[0].beam.power_W if o.arrivals else 0.0) / 1e-3 for o in t.observations)
    assert shares == pytest.approx([0.0, 1.0], abs=1e-12)


# ---------------------------------------------------------------------------------------- not known / not followed

def test_without_a_known_polarization_the_unpolarized_share_and_its_range(hwp_and_pbs, catalog):
    project = _splitter_design(hwp_and_pbs, catalog, 0.0)
    router = PolarizationRouter(lower_project(project.payload(), catalog).request.project)
    _, cube, wavelength = hwp_and_pbs
    band = next(catalog.coating(s.coating_id).band_at(Decimal(repr(wavelength))) for s in cube.simulation_models[0].surfaces
                if s.traced.value == "TRANSMITTED_AND_REFLECTED")
    passage = router.step("PBS", ROUTE_PASS, wavelength, (0.0, 0.0, 0.0), X, X, None)
    lo, hi = passage.share_range
    ar = passage.share_range[1] / float(band.transmittance_p)  # the two anti-reflection faces
    assert (lo, hi) == pytest.approx((float(band.transmittance_s) * ar, float(band.transmittance_p) * ar), rel=1e-12)
    assert passage.share == passage.unpolarized_share == pytest.approx((lo + hi) / 2.0, rel=1e-12)
    assert passage.polarization_dependent and not passage.arriving_known
    # its transmitted port polarizes (extinction 1000:1): p leaves, in the plane of incidence (design +Y)
    assert np.abs(np.array(passage.state.field())) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    reflected = router.step("PBS", ROUTE_SPLIT_REFLECT, wavelength, (0.0, 0.0, 0.0), X, Y, None)
    assert reflected.polarization_dependent and reflected.state is None  # Rp / Rs = 0.1 / 0.95: not a polarizer
    # a half-wave plate passes every polarization alike: its share does not depend on it, and what leaves is not known
    plate = router.step("HWP", ROUTE_PASS, wavelength, (-50.0, 0.0, 0.0), X, X, None)
    assert not plate.polarization_dependent and plate.state is None


def test_a_polarizer_emits_a_known_polarization_whatever_arrives():
    project = ProjectSnapshot(components=(
        B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
        B.polarizer("P", position=(10, 0, 20), axis=(1, 0, 0), angle_deg=30.0, t_par=0.9, t_perp=1e-5),
    ))
    router = PolarizationRouter(project)
    passage = router.step("P", ROUTE_PASS, 532.0, (10, 0, 20), X, X, None)
    assert passage.share_range[0] <= POLARIZING_RATIO * passage.share_range[1]
    known = router.step("P", ROUTE_PASS, 532.0, (10, 0, 20), X, X, router.source_state("L"))
    # the same linear polarization along the axis, up to the leak of the crossed component (sqrt(t_perp / t_par))
    assert np.allclose(np.abs(passage.state.field()), np.abs(known.state.field()), atol=5e-3)
    axis = np.array([0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    assert abs(np.vdot(axis, np.array(passage.state.field()))) == pytest.approx(1.0, abs=1e-12)


def test_steps_that_cannot_be_followed_are_not_answered(hwp_and_pbs, catalog):
    project = _splitter_design(hwp_and_pbs, catalog, 0.0)
    router = PolarizationRouter(lower_project(project.payload(), catalog).request.project)
    _, _, wavelength = hwp_and_pbs
    state = router.source_state("SRC")
    assert isinstance(state, LineState)
    assert router.step("NOPE", ROUTE_PASS, wavelength, (0.0, 0.0, 0.0), X, X, state) is None  # no such element
    assert router.step("HWP", ROUTE_SPLIT_REFLECT, wavelength, (-50.0, 0.0, 0.0), X, X, state) is None  # nothing splits
    assert router.step("PBS", ROUTE_PASS, wavelength, (0.0, 0.0, 0.0), X, Y, state) is None  # it does not leave along +Y
    assert router.step("PBS", ROUTE_PASS, 300.0, (0.0, 0.0, 0.0), X, X, state) is None  # no coating data at 300 nm
    with pytest.raises(ValueError):
        router.step("PBS", "SIDEWAYS", wavelength, (0.0, 0.0, 0.0), X, X, state)


def test_routing_is_deterministic(hwp_and_pbs, catalog):
    project = _splitter_design(hwp_and_pbs, catalog, 22.5)
    request = lower_project(project.payload(), catalog).request.project
    a, b = PolarizationRouter(request), PolarizationRouter(request)
    state = a.source_state("SRC")
    one = a.step("HWP", ROUTE_PASS, hwp_and_pbs[2], (-50.0, 0.0, 0.0), X, X, state)
    two = b.step("HWP", ROUTE_PASS, hwp_and_pbs[2], (-50.0, 0.0, 0.0), X, X, state)
    assert one == two
