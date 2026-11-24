"""Coherence through coatings whose catalog data bound their group delay (catalog schema 3.2, ``group_delay_bound_fs``).

The two arms of a catalog-built interferometer meet the splitter's and the mirrors' coatings differently. Where every
coating whose pass differs between the arms publishes a group-delay bound, the arms' relative group delay is known within
the sum ``B`` of those bounds, and the degree of coherence within ``delta = 2 B / (sqrt(pi) L_c)`` of its value at the
traced delay (``optics/interference/coherence.py``). The coherence metrics are stated with that interval, whatever its
width, and targets on them are judged on it; a differing pass without a bound gives them on the trace's convention (no
added delay there, named in value_basis), reported and not deciding a target.

The runs copy the test catalog's optics into declared models and explicitly assume alpha=0 only there: the public
catalog schema does not state N-BK7's bulk loss. The checks below compute the expected numbers independently of the
Engine: the plate's group path from the catalog's Sellmeier coefficients and Snell's law, the coherence length from the Gaussian line's FWHM, the two-beam visibility
``V = 2 sqrt(P1 P2) / (P1 + P2) |gamma|`` with ``|gamma| = exp(-(D / L_c)^2)``, and the bound ``delta`` by integrating
the spectrum numerically against relative spectral phases of bounded slope. The bound values in the test catalog (1000 fs
for the high reflector and the splitter coating, 50 fs for the anti-reflection coating) are the coating classes' published group-delay
bounds.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index, air_index_slope

from os_optics.local import example_project_of_kind, run_optical_project
from os_optics.optics.interference.coherence import (
    coherence_spread,
    delay_bound_fs,
    delay_key,
    relative_delay_bound,
)

from declared_helpers import declared_project
from local_helpers import (
    ALONG_X,
    FLAT_MIRROR,
    FOLD_A,
    FOLD_B,
    FOLD_X_TO_Y,
    PLATE_BS,
    bounded_catalog_document,
    catalog_document,
    element,
    plane,
    product_document,
    project,
    sellmeier_index,
    source,
    thickness,
)

C_MM_PER_FS = 2.99792458e-4  # the speed of light, mm per femtosecond
WAVELENGTH_NM = 780.0
LINEWIDTH_NM = 0.001
FOLD_Y_TO_X = (FOLD_B, FOLD_B, FOLD_A, FOLD_A)


def _ideal_project(document: dict, catalog: dict) -> dict:
    """Declared copies of synthetic catalog parts with test-only alpha=0; public catalog schema forbids bulk loss."""
    declared = declared_project(document, catalog)
    for material in declared["model_materials"]:
        assert "absorption_coefficient_per_mm" not in material and "bulk_loss" not in material
        material["absorption_coefficient_per_mm"] = "0"  # assumed, not a claim about public catalog transmission
    return declared


def _ideal_run(document: dict, catalog: dict):
    return run_optical_project(_ideal_project(document, catalog), catalog=catalog)


bounded_catalog = bounded_catalog_document


def _coating(document: dict, sku: str, surface_id: str) -> dict:
    surface = next(s for s in product_document(sku, document)["simulation_models"][0]["surfaces"] if s["surface_id"] == surface_id)
    return next(c for c in document["coatings"] if c["coating_id"] == surface["coating_id"])


def _band(coating: dict) -> tuple[float, float]:
    (band,) = coating["bands"]
    return float(band["reflectance"]), float(band["transmittance"])


def coherence_length_mm(wavelength_nm: float, linewidth_nm: float) -> float:
    """``L_c`` of a Gaussian line of FWHM ``linewidth_nm``: its power spectrum ``exp(-((k - k0) L_c / 2)^2)`` has the FWHM
    ``4 sqrt(ln 2) / L_c`` in ``k = 2 pi / lambda``, i.e. ``2 pi linewidth / lambda^2``."""
    lam_mm, dlam_mm = wavelength_nm * 1e-6, linewidth_nm * 1e-6
    return 2.0 * math.sqrt(math.log(2.0)) * lam_mm * lam_mm / (math.pi * dlam_mm)


def plate_group_excess_mm(document: dict) -> tuple[float, float]:
    """``(phase, group)`` path excess of the arm transmitted through the plate splitter at 45 deg over the reflected arm in
    the reference interferometer (``two_beam_interference``: both arms 300 mm in air, the transmitted arm's mirror on its
    shifted axis): ``n L - n_air (x_exit + shift)`` and ``n_g L - n_g,air (x_exit + shift)``, with ``L`` the path inside the plate,
    in the example's air (20 C, 101325 Pa: ``air_index`` and its slope; the plate refracts by ``n / n_air``)."""
    t = thickness(PLATE_BS)
    n = sellmeier_index("N-BK7", WAVELENGTH_NM, document)
    h = 0.01
    dn = (sellmeier_index("N-BK7", WAVELENGTH_NM + h, document) - sellmeier_index("N-BK7", WAVELENGTH_NM - h, document)) / (2 * h)
    n_g = n - WAVELENGTH_NM * dn
    n_air = air_index(WAVELENGTH_NM)
    n_g_air = n_air - WAVELENGTH_NM * air_index_slope(WAVELENGTH_NM)[0]
    cos_i = math.sqrt(0.5)
    sin_t = cos_i * n_air / n
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    normal = np.array([-cos_i, cos_i])  # the coated face's normal, towards the incoming beam (+X)
    d_t = np.array([1.0, 0.0]) * n_air / n + (cos_i * n_air / n - cos_t) * normal
    inside = t / cos_t
    x_exit, y_exit = inside * d_t
    return n * inside - n_air * (x_exit + y_exit), n_g * inside - n_g_air * (x_exit + y_exit)


@pytest.fixture(scope="module")
def catalog() -> dict:
    return bounded_catalog()


@pytest.fixture(scope="module")
def interferometer(catalog) -> dict:
    document = example_project_of_kind("two_beam_interference", catalog).payload()
    document["targets"] += [
        {"target_id": "visibility", "metric": "fringe_visibility", "observation_id": "crossing_plane", "minimum": 0.5, "unit": "1"},
        {"target_id": "coherence", "metric": "coherence_factor", "observation_id": "crossing_plane", "minimum": 0.995, "unit": "1"},
        {"target_id": "coherence_tight", "metric": "coherence_factor", "observation_id": "crossing_plane", "minimum": 0.9999, "unit": "1"},
        {"target_id": "margin", "metric": "coherence_margin", "observation_id": "crossing_plane", "minimum": 0.99, "unit": "1"},
    ]
    return document


@pytest.fixture(scope="module")
def bounded_run(interferometer, catalog):
    return _ideal_run(interferometer, catalog)


# ------------------------------------------------------------------------------------------------ the bound itself


def test_a_pass_record_carries_its_coatings_bound_and_shared_passes_cancel():
    bounded = delay_key("M1/S1", "reflect", True, 45.0, state="1.000000,0.000000,0.000000", bound_fs=1000.0)
    assert bounded.endswith("<=1000.0fs") and delay_bound_fs(bounded) == 1000.0
    plain = delay_key("M1/S1", "reflect", True, 45.0)
    assert delay_bound_fs(plain) is None and "<=" not in plain
    shared = delay_key("BS/S2", "split_reflect", True, 45.0, bound_fs=50.0)
    other = delay_key("M2/S1", "reflect", True, 45.0, bound_fs=1000.0)
    assert relative_delay_bound((shared, bounded), (bounded, shared)) == 0.0  # the same passes: the delays cancel
    assert relative_delay_bound((shared, bounded), (shared, other)) == pytest.approx(2000.0 * C_MM_PER_FS, rel=1e-15)
    assert relative_delay_bound((shared, shared), (shared,)) == pytest.approx(50.0 * C_MM_PER_FS, rel=1e-15)  # counted with multiplicity
    assert relative_delay_bound((shared, bounded), (shared, plain)) is None  # a differing pass without a bound
    assert coherence_spread(1.0, 100.0) == pytest.approx(2.0 / (math.sqrt(math.pi) * 100.0), rel=1e-15)


@pytest.mark.parametrize("delay_over_lc", [0.0, 0.3, 1.0, 2.0])
def test_the_coherence_bound_holds_for_any_coating_dispersion_within_the_bound(delay_over_lc):
    """|gamma - gamma_traced| <= 2 B / (sqrt(pi) L_c) for relative spectral phases psi(k) of slope within [-B, B] (any
    shape: a step of the slope at the line centre -- the worst case --, an oscillating slope, a random one), by direct
    numerical integration of the Gaussian spectrum (independent of the Engine's code)."""
    lc, bound = 10.0, 0.05
    k = np.linspace(-12.0 / lc, 12.0 / lc, 400001)  # k - k0 (1/mm); the spectrum is negligible beyond
    weight = np.exp(-((k * lc / 2.0) ** 2))
    weight /= np.trapezoid(weight, k)
    delay = delay_over_lc * lc
    traced = np.trapezoid(weight * np.exp(1j * delay * k), k)
    assert abs(traced) == pytest.approx(math.exp(-(delay_over_lc ** 2)), abs=1e-9)
    rng = np.random.default_rng(409)
    slopes = [bound * np.sign(k), -bound * np.sign(k), bound * np.cos(k * lc * 3.0), bound * np.sin(k * lc * 7.0) * np.sign(k),
              np.repeat(rng.uniform(-bound, bound, 4001), 100)[: k.size]]
    delta = 2.0 * bound / (math.sqrt(math.pi) * lc)
    worst = 0.0
    for slope in slopes:
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(k))])
        psi -= np.interp(0.0, k, psi)  # the traced phase at the line centre
        gamma = np.trapezoid(weight * np.exp(1j * (delay * k + psi)), k)
        worst = max(worst, abs(gamma - traced))
        assert abs(gamma - traced) <= delta * (1.0 + 1e-6)
    if delay == 0.0:
        assert worst > 0.5 * delta  # at zero delay the worst slope comes close to the bound


# ------------------------------------------------------------------------- the catalog-built reference interferometer


def test_bounded_coatings_give_the_interferometer_its_coherence_with_the_bound_stated(bounded_run, catalog):
    run = bounded_run
    (pair,) = run.result.interference
    assert "COHERENCE_UNDEFINED" not in pair.issue_codes
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]

    # independent numbers: the arms' group path difference, the coherence length, the bound and delta
    phase_excess, group_excess = plate_group_excess_mm(catalog)
    lc = coherence_length_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    bound_fs = sum(float(_coating(catalog, sku, s)["group_delay_bound_fs"]) for sku, s in (
        (PLATE_BS, "S2"), (PLATE_BS, "S1"), (FLAT_MIRROR, "S1"),  # transmitted arm: splitter coating, AR face, its mirror
        (PLATE_BS, "S2"), (FLAT_MIRROR, "S1")))  # reflected arm: splitter coating (in reflection), its mirror
    assert bound_fs == 4050.0
    bound_mm = bound_fs * C_MM_PER_FS
    delta = 2.0 * bound_mm / (math.sqrt(math.pi) * lc)
    assert delta < 0.005
    gamma = math.exp(-((group_excess / lc) ** 2))

    sign = 1.0 if pair.optical_path_difference_mm > 0 else -1.0  # (operand order a, b)
    assert sign * pair.optical_path_difference_mm == pytest.approx(phase_excess, rel=1e-6)
    assert sign * pair.group_path_difference_mm == pytest.approx(group_excess, rel=1e-6)
    assert pair.coherence_length_mm == pytest.approx(lc, rel=1e-12)
    assert pair.group_delay_bound_mm == pytest.approx(bound_mm, rel=1e-12)
    assert pair.coherence_factor == pytest.approx(gamma, abs=1e-9)
    assert pair.coherence_factor_min == pytest.approx(gamma - delta, abs=1e-9)
    assert pair.coherence_factor_max == pytest.approx(min(1.0, gamma + delta), abs=1e-12)
    assert pair.coherence_margin == pytest.approx(1.0 - group_excess / lc, rel=1e-9)

    # the two-beam visibility V = 2 sqrt(P1 P2) / (P1 + P2) |gamma| (s-polarized arms: polarization overlap 1; equal
    # Gaussian beams centred on the plane: spatial overlap 1 to 1e-7)
    r_bs, t_bs = _band(_coating(catalog, PLATE_BS, "S2"))
    _, t_ar = _band(_coating(catalog, PLATE_BS, "S1"))
    r_m, _ = _band(_coating(catalog, FLAT_MIRROR, "S1"))
    p_transmitted, p_reflected = 1e-3 * t_bs * t_ar * r_m, 1e-3 * r_bs * r_m
    assert pair.incoherent_power_sum_W == pytest.approx(p_transmitted + p_reflected, rel=1e-12)
    visibility = 2.0 * math.sqrt(p_transmitted * p_reflected) / (p_transmitted + p_reflected) * gamma
    assert pair.polarization_overlap == pytest.approx(1.0, abs=1e-12) and pair.spatial_overlap_ratio > 1.0 - 1e-7
    assert pair.fringe_visibility == pytest.approx(visibility, rel=1e-6)
    assert pair.fringe_visibility_peak == pytest.approx(visibility, rel=1e-6)
    print(f"independent check: D_g = {group_excess:.9f} mm, L_c = {lc:.6f} mm, B = {bound_mm:.9f} mm ({bound_fs:g} fs), "
          f"|gamma| = {gamma:.12f}, delta = {delta:.9f}, V = {visibility:.12f}; engine: V = {pair.fringe_visibility:.12f}, "
          f"gamma in [{pair.coherence_factor_min:.9f}, {pair.coherence_factor_max:.9f}]")


def test_a_declared_coating_states_its_bound_like_a_catalog_coating(bounded_run, interferometer, catalog):
    """The same interferometer of declared copies of the catalog parts (``model_coatings`` with the coatings' bounds) gives
    the same coherence."""
    declared = _ideal_project(interferometer, catalog)
    assert {c.get("group_delay_bound_fs") for c in declared["model_coatings"]} == {"1000", "50"}
    run = run_optical_project(declared, catalog=catalog_document())
    (pair,), (want,) = run.result.interference, bounded_run.result.interference
    for name in ("group_path_difference_mm", "group_delay_bound_mm", "coherence_factor", "coherence_factor_min", "coherence_factor_max",
                 "fringe_visibility"):
        assert getattr(pair, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=1e-15), name


def test_targets_on_a_bounded_coherence_are_judged_on_its_interval(bounded_run):
    outcomes = {t.target_id: t for t in bounded_run.result.targets}
    for target_id in ("fringe_period", "intensity_balance", "polarization_overlap", "coherence_length", "visibility", "coherence",
                      "margin"):
        assert outcomes[target_id].status.value == "PASS", (target_id, outcomes[target_id].message)
    for target_id in ("visibility", "coherence", "margin"):
        message = outcomes[target_id].message
        assert "group delays are known within 4050 fs in all" in message and message.endswith("the target is met")
    # 0.9999 lies inside the degree of coherence's interval (0.9957..1): met at the traced delay, not decided
    tight = outcomes["coherence_tight"]
    assert tight.value > 0.9999 and tight.status.value == "UNKNOWN"
    assert "the coherence factor is between" in tight.message and "not decided" in tight.message
    assert bounded_run.status.value == "UNKNOWN"
    # each value is read at the traced delay (no added coating delay): it carries the pair's entry of that convention, as the
    # pair does (B8), while the interval grades it
    from os_optics.results import basis_of

    (pair,) = bounded_run.result.interference
    for target_id, field in (("visibility", "fringe_visibility"), ("coherence", "coherence_factor"),
                             ("coherence_tight", "coherence_factor"), ("margin", "coherence_margin")):
        entry = outcomes[target_id].value_basis["value"]
        assert entry == basis_of(pair, field), target_id
        assert entry.kind.value == "ESTIMATE" and [x.input.value for x in entry.assumed] == ["UNSTATED_GROUP_DELAY_ZERO"], target_id


def test_a_bound_not_small_beside_the_coherence_length_states_the_coherence_with_its_wider_interval(interferometer, catalog):
    """delta = 2 x 1.214 mm / (sqrt(pi) x 50 mm) = 0.027: the interval is the statement (no gate). The value at the traced
    delay is exp(-(D_g / L_c)^2) with the plate's group excess computed from the catalog's Sellmeier data; a target whose
    limit lies inside the interval is not decided."""
    document = copy.deepcopy(interferometer)
    model = document["components"][0]["source_model"]
    model.pop("linewidth_nm")
    model["coherence_length_mm"] = 50.0  # delta = 2 x 1.214 mm / (sqrt(pi) x 50 mm) = 0.027 > 0.01
    run = _ideal_run(document, catalog)
    (pair,) = run.result.interference
    _, group_excess = plate_group_excess_mm(catalog)
    gamma = math.exp(-((group_excess / 50.0) ** 2))
    delta = coherence_spread(4050.0 * C_MM_PER_FS, 50.0)
    assert delta == pytest.approx(0.0274, abs=5e-5)
    assert pair.group_delay_bound_mm == pytest.approx(4050.0 * C_MM_PER_FS, rel=1e-12)
    assert pair.coherence_factor == pytest.approx(gamma, abs=1e-9)
    assert pair.coherence_factor_min == pytest.approx(gamma - delta, abs=1e-9)
    assert pair.coherence_factor_max == pytest.approx(min(1.0, gamma + delta), abs=1e-12)
    assert pair.coherence_margin == pytest.approx(1.0 - group_excess / 50.0, rel=1e-9)
    assert pair.fringe_visibility is not None and pair.peak_irradiance_W_per_mm2 is not None
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    outcomes = {t.target_id: t for t in run.result.targets}
    # 0.995 and 0.9999 lie inside [gamma - delta, 1]: not decided; the visibility minimum of 0.5 lies below it: met
    for target_id in ("coherence", "coherence_tight"):
        assert outcomes[target_id].status.value == "UNKNOWN" and outcomes[target_id].value == pair.coherence_factor
    assert outcomes["visibility"].status.value == "PASS"
    assert outcomes["coherence_length"].status.value == "FAIL" and outcomes["coherence_length"].value == 50.0  # (min 100 mm)


def test_a_source_without_a_stated_coherence_is_not_given_a_degree_of_coherence_by_the_coatings_bounds(interferometer, catalog):
    """The coatings bound the arms' relative group delay, but a line that states no coherence is taken as monochromatic:
    its degree of coherence 1 is a stand-in, not a proven value, so the bounds give it no proven interval (no
    ``coherence_factor_min`` / ``_max``). The visibility, the coherence factor and the peak irradiance are the values at full
    coherence, upper ends with the ``FULL_COHERENCE`` entry; a minimum on them is judged on the interval every degree of
    coherence gives, from 0 to that value, so 0.5 and 0.995 are not decided (a degree of coherence of 0 gives no fringes)."""
    document = copy.deepcopy(interferometer)
    document["components"][0]["source_model"].pop("linewidth_nm")
    document["targets"] = [t for t in document["targets"] if t["target_id"] not in ("coherence_length", "margin")]
    run = _ideal_run(document, catalog)
    (pair,) = run.result.interference
    assert pair.group_delay_bound_mm == pytest.approx(4050.0 * C_MM_PER_FS, rel=1e-12)
    assert pair.coherence_factor == 1.0 and (pair.coherence_factor_min, pair.coherence_factor_max) == (None, None)
    for name in ("coherence_factor", "fringe_visibility"):
        entry = pair.value_basis[name]
        assert entry.kind.value == "UPPER_BOUND" and [a.input.value for a in entry.assumed] == ["FULL_COHERENCE"]
    outcomes = {t.target_id: t for t in run.result.targets}
    for target_id in ("visibility", "coherence", "coherence_tight"):
        target = outcomes[target_id]
        assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE"), (target_id, target.message)
        assert target.value is not None and target.value_basis["value"].kind.value == "UPPER_BOUND"
    assert outcomes["visibility"].value == pytest.approx(pair.fringe_visibility, rel=1e-12) and outcomes["visibility"].value > 0.5


def test_a_differing_coating_without_a_bound_states_the_coherence_on_the_traces_convention(catalog):
    """The mirrors' coatings bound no delay: the coherence is stated at the traced delay -- no added delay at their passes,
    named in value_basis -- with no interval, and a target on it is reported and not decided. Independent reference: the
    same interferometer whose mirrors bound their delays has the same traced group path difference and coherence (the
    bound only adds the interval)."""
    document = bounded_catalog(unbounded_kinds=("DIELECTRIC_HIGH_REFLECTOR",))
    interferometer = example_project_of_kind("two_beam_interference", document).payload()
    interferometer["targets"].append(
        {"target_id": "coherence", "metric": "coherence_factor", "observation_id": "crossing_plane", "minimum": 0.5, "unit": "1"})
    run = _ideal_run(interferometer, document)
    (pair,) = run.result.interference
    reference = _ideal_run(interferometer, catalog)
    (bounded,) = reference.result.interference
    _, group_excess = plate_group_excess_mm(catalog)
    lc = coherence_length_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    assert abs(pair.group_path_difference_mm) == pytest.approx(group_excess, rel=1e-6)
    assert pair.group_path_difference_mm == bounded.group_path_difference_mm
    assert pair.coherence_factor == bounded.coherence_factor == pytest.approx(math.exp(-((group_excess / lc) ** 2)), abs=1e-9)
    assert pair.fringe_visibility == bounded.fringe_visibility
    assert pair.group_delay_bound_mm is None and pair.coherence_factor_min is None and pair.coherence_factor_max is None
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    entry = pair.value_basis["coherence_factor"]
    assert entry.kind.value == "ESTIMATE"
    (assumed,) = entry.assumed
    assert (assumed.input.value, assumed.at, assumed.value, assumed.unit) == (
        "UNSTATED_GROUP_DELAY_ZERO", ("M1/S1:reflect:+", "M2/S1:reflect:+"), 0.0, "fs")  # only the unbounded passes
    for name in ("group_path_difference_mm", "coherence_margin", "fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2"):
        assert pair.value_basis[name] == entry, name
    (target,) = [t for t in run.result.targets if t.target_id == "coherence"]
    assert target.value == pair.coherence_factor and target.value_basis["value"] == entry
    assert target.assessment.value == "INDETERMINATE"


def test_a_catalog_without_bounds_records_its_passes_as_before():
    document = catalog_document()
    run = _ideal_run(example_project_of_kind("two_beam_interference", document).payload(), document)
    (pair,) = run.result.interference
    # (no bound: the coherence on the trace's convention, no interval)
    assert pair.group_delay_bound_mm is None and pair.coherence_factor_min is None
    assert pair.value_basis["coherence_factor"].assumed[0].input.value == "UNSTATED_GROUP_DELAY_ZERO"
    entries = [e for o in run.result.observations for a in o.arrivals for e in a.group_delay_unknown_at]
    assert entries and not any("<=" in e for e in entries)
    assert "group_delay_bound_mm" not in pair.model_dump(mode="json")


# ------------------------------------------------------------------------------ coherent power: a Mach-Zehnder


def mach_zehnder(catalog: dict, **source_model) -> dict:
    """A Mach-Zehnder of two catalog plate splitters and two flat mirrors (each arm crosses one plate once, so the arms'
    paths are equal): BS1 at the origin, M1 on the transmitted arm's shifted axis, M2 shifted by the same amount so that
    the second plate's lateral shift brings that arm onto the first's at BS2; both leave BS2 along +X onto ``out``."""
    shift = thickness(PLATE_BS) * math.sqrt(0.5) * (1.0 - math.sqrt(0.5) / math.sqrt(sellmeier_index("N-BK7", WAVELENGTH_NM, catalog) ** 2 - 0.5))
    version = catalog["catalog_version"]
    laser = source("SRC", (-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM)
    laser["source_model"].update(linewidth_nm=LINEWIDTH_NM, **source_model)
    components = [
        laser,
        element("BS1", "splitter", PLATE_BS, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=version),
        element("M1", "mirror", FLAT_MIRROR, (100.0, -shift, 0.0), FOLD_X_TO_Y, catalog_version=version),
        element("M2", "mirror", FLAT_MIRROR, (0.0, 100.0 + shift, 0.0), FOLD_Y_TO_X, catalog_version=version),
        element("BS2", "splitter", PLATE_BS, (100.0, 100.0, 0.0), FOLD_Y_TO_X, catalog_version=version),
    ]
    return project(components, [plane("out", (150.0, 100.0, 0.0), ALONG_X)],
                   [{"target_id": "power", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.0, "unit": "W"}])


def _arm_power(catalog: dict) -> float:
    r_bs, t_bs = _band(_coating(catalog, PLATE_BS, "S2"))
    _, t_ar = _band(_coating(catalog, PLATE_BS, "S1"))
    r_m, _ = _band(_coating(catalog, FLAT_MIRROR, "S1"))
    return 1e-3 * t_bs * t_ar * r_m * r_bs  # each arm: one transmission through a plate, one splitter reflection, a mirror


def test_a_mach_zehnder_of_coatings_without_stated_phases_states_its_coherent_power_on_the_default_phases(catalog):
    """The catalog coatings bound their group delays but state no phase: the degree of coherence and the visibility are
    known, the phase between the arms at the output is on the trace's default interface phases -- the coherent power is
    stated there with value_basis naming the passes, beside its range over every phase, ``2P (1 -+ min(1, |gamma| +
    delta))`` -- and a target inside that range is not decided. Independent reference: the same interferometer whose
    coatings state exactly the default phases (a splitter's reflection pi/2 from its transmission, a mirror's (pi, 0))
    gives the same power."""
    document = mach_zehnder(catalog)
    document["targets"].append({"target_id": "bright", "metric": "coherent_power_W", "observation_id": "out", "minimum": 5e-4,
                                "unit": "W"})
    run = _ideal_run(document, catalog)
    (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
    (sum_,) = observation.superpositions
    p_arm = _arm_power(catalog)
    assert [a.beam.power_W for a in observation.arrivals] == pytest.approx([p_arm, p_arm], rel=1e-12)
    # both arms: splitter coating (T and R), the AR face, a mirror -- none shared: B = 2 x (1000 + 50 + 1000 + 1000) fs
    bound_mm = 6100.0 * C_MM_PER_FS
    lc = coherence_length_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    delta = 2.0 * bound_mm / (math.sqrt(math.pi) * lc)
    (pair,) = run.result.interference
    assert pair.group_delay_bound_mm == pytest.approx(bound_mm, rel=1e-12)
    assert abs(pair.group_path_difference_mm) < 1e-9 and pair.coherence_factor == pytest.approx(1.0, abs=1e-12)
    assert pair.coherence_factor_min == pytest.approx(1.0 - delta, rel=1e-9)
    # the value at the default phases, its range over every phase beside it
    assert pair.combined_power_collinear_W == pytest.approx(sum_.total_power_W, rel=1e-9)
    assert pair.combined_power_collinear_min_W <= pair.combined_power_collinear_W <= pair.combined_power_collinear_max_W
    (phase,) = pair.value_basis["phase_difference_rad"].assumed
    assert phase.input.value == "UNSTATED_PHASE_AS_INTERFACE" and len(phase.at) == 8 and pair.phase_statement is None
    assert sum_.total_power_W is not None and "PHASE_UNDEFINED" not in sum_.issue_codes and sum_.phase_statement is None
    entry = sum_.value_basis["total_power_W"]
    assert entry.kind.value == "ESTIMATE" and {a.input.value for a in entry.assumed} == {"UNSTATED_PHASE_AS_INTERFACE",
                                                                                        "UNSTATED_GROUP_DELAY_ZERO"}
    assert sum_.incoherent_power_W == pytest.approx(2.0 * p_arm, rel=1e-12)
    assert sum_.total_power_min_W == pytest.approx(0.0, abs=1e-15)
    assert sum_.total_power_max_W == pytest.approx(4.0 * p_arm, rel=1e-9)  # (|gamma| + delta capped at 1)
    assert sum_.total_power_min_W <= sum_.total_power_W <= sum_.total_power_max_W
    stated = run_optical_project(_stated_phase_mach_zehnder(catalog, repr(math.pi / 2)), catalog=catalog_document())
    (stated_sum,) = [o for o in stated.result.observations if o.observation_id == "out"][0].superpositions
    assert sum_.total_power_W == pytest.approx(stated_sum.total_power_W, rel=1e-9, abs=1e-15)
    targets = {t.target_id: t for t in run.result.targets}
    # (judged on the proven range: 5e-4 W lies inside [0, 4P]; reported with the value at the default phases)
    assert targets["bright"].status.value == "UNKNOWN" and targets["bright"].value == sum_.total_power_W
    # (the target's entry is the sum's with the light the trace does not follow that may reach the plane)
    target_entry = targets["bright"].value_basis["value"]
    assert target_entry.kind.value == "ESTIMATE" and target_entry.assumed == entry.assumed
    assert "NOT_FOLLOWED" in {o.part.value for o in target_entry.omitted} and "not decided" in targets["bright"].message


def _stated_phase_mach_zehnder(catalog: dict, bs2_reflection_rad: str) -> dict:
    """The Mach-Zehnder of declared copies of the catalog parts whose coatings state their phases at the line (a flat
    phase table: the phase is stated, its derivative is not formed, so the group-delay bounds still apply), BS2 with its own
    coating: a lossless 50 / 50 splitter whose reflection phase is ``bs2_reflection_rad`` and transmission phase 0."""
    # The test-only declared material makes the phase experiment independent of unstated catalog bulk loss.
    document = _ideal_project(mach_zehnder(catalog), catalog)

    def table(**phases: str) -> dict:
        return {"wavelength_nm": ["400", "1100"], **{name: [value, value] for name, value in phases.items()}, "status": "DECLARED"}

    half_pi = repr(math.pi / 2)
    for coating in document["model_coatings"]:
        if coating["kind"] == "DIELECTRIC_HIGH_REFLECTOR":
            coating["phase"] = table(reflection_s_rad="3.141592653589793", reflection_p_rad="0")
        else:  # the splitter and anti-reflection coatings: lossless, reflection and transmission a quarter period apart
            coating["phase"] = table(reflection_s_rad=half_pi, reflection_p_rad=half_pi, transmission_s_rad="0", transmission_p_rad="0")
    splitter = next(c for c in document["model_coatings"] if c["kind"] == "BEAMSPLITTER")
    second = copy.deepcopy(splitter)
    second["coating_id"] = "BS2.SPLITTER"
    second["phase"] = table(reflection_s_rad=bs2_reflection_rad, reflection_p_rad=bs2_reflection_rad, transmission_s_rad="0",
                            transmission_p_rad="0")
    document["model_coatings"].append(second)
    model = copy.deepcopy(next(m for m in document["optical_models"] if m["model_id"] == next(
        c["optical_model_id"] for c in document["components"] if c["instance_id"] == "BS1")))
    model["model_id"] = "bs2"
    for surface in model["model"]["surfaces"]:
        if surface.get("coating_id") == splitter["coating_id"]:
            surface["coating_id"] = second["coating_id"]
    document["optical_models"].append(model)
    next(c for c in document["components"] if c["instance_id"] == "BS2")["optical_model_id"] = "bs2"
    return document


def test_a_splitter_phase_alone_moves_the_port_power_within_the_open_range(catalog):
    """Changing only BS2's stated reflection phase by pi turns the output port from bright to dark. With every differing
    coating's phase stated, each case states its power with the narrow range its group-delay bounds leave; both lie inside
    the range the same interferometer reports when its coatings state no phase."""
    open_run = _ideal_run(mach_zehnder(catalog), catalog)
    (open_sum,) = [o for o in open_run.result.observations if o.observation_id == "out"][0].superpositions
    lc = coherence_length_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    delta = 2.0 * 6100.0 * C_MM_PER_FS / (math.sqrt(math.pi) * lc)
    p_arm = _arm_power(catalog)
    powers = {}
    for name, phase in (("plus", repr(math.pi / 2)), ("minus", repr(-math.pi / 2))):
        run = run_optical_project(_stated_phase_mach_zehnder(catalog, phase), catalog=catalog_document())
        (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
        (sum_,) = observation.superpositions
        entries = [e for a in observation.arrivals for e in a.group_delay_unknown_at]
        assert entries and all("#phase<=" in e for e in entries)  # every pass bounded, its phase stated
        (pair,) = run.result.interference
        assert pair.combined_power_collinear_W == pytest.approx(sum_.total_power_W, rel=1e-9)
        assert sum_.total_power_W is not None and "PHASE_UNDEFINED" not in sum_.issue_codes
        # the cross term 2P |gamma| cos(phi) lies within 2P delta of its traced value and within +-2P
        traced = sum_.total_power_W - 2.0 * p_arm
        assert sum_.total_power_min_W == pytest.approx(2.0 * p_arm + max(traced - 2.0 * p_arm * delta, -2.0 * p_arm), abs=1e-15)
        assert sum_.total_power_max_W == pytest.approx(2.0 * p_arm + min(traced + 2.0 * p_arm * delta, 2.0 * p_arm), abs=1e-15)
        assert open_sum.total_power_min_W <= sum_.total_power_min_W and sum_.total_power_max_W <= open_sum.total_power_max_W
        powers[name] = sum_.total_power_W
    # the two cases differ by pi in the phase between the arms: one port bright (4P), one dark (0), up to the coherence
    assert sorted(powers.values()) == pytest.approx([0.0, 4.0 * p_arm], abs=1e-12)


def test_the_pair_states_its_combined_power_and_margin_at_the_traced_delay_as_estimates(catalog):
    """Every differing pass bounded and its phase stated, linewidth 0.05 nm (coherence length 6.45 mm against a delay bound
    of 1.83 mm): the coherence factor lies in its interval, so the pair's combined power and coherence margin, read at the
    traced delay with no added coating delay, are estimates of that convention -- as the plane's coherent sum of the same
    beams says (ESTIMATE, UNSTATED_GROUP_DELAY_ZERO), not the model's values."""

    from os_optics.results import basis_of

    document = _stated_phase_mach_zehnder(catalog, repr(math.pi / 2))
    for component in document["components"]:
        if component.get("source_model"):
            component["source_model"]["linewidth_nm"] = 0.05
    run = run_optical_project(document, catalog=catalog_document())
    (pair,) = run.result.interference
    (sum_,) = [o for o in run.result.observations if o.observation_id == "out"][0].superpositions
    assert pair.coherence_factor_min < pair.coherence_factor <= pair.coherence_factor_max
    assert pair.combined_power_collinear_W == pytest.approx(sum_.total_power_W, rel=1e-9)
    assert sum_.total_power_min_W < sum_.total_power_W  # (the plane states the range the delay bound leaves)
    for field in ("combined_power_collinear_W", "coherence_margin"):
        entry = basis_of(pair, field)
        assert entry is not None and entry.kind.value == "ESTIMATE", field
        assert "UNSTATED_GROUP_DELAY_ZERO" in [x.input.value for x in entry.assumed], field
    assert [x.input.value for x in basis_of(sum_, "total_power_W").assumed] == ["UNSTATED_GROUP_DELAY_ZERO"]


def test_a_coherent_power_target_behind_bounded_delays_carries_the_entry_of_the_traced_delays(catalog):
    """The same arms: the plane's coherent sum behind bounded coating delays alone stores no entry, and its value at the
    traced delays is the ESTIMATE of that convention (``basis_of``: UNSTATED_GROUP_DELAY_ZERO). A coherent_power_W target
    on the plane reports that value with the same entry -- not as the model's value -- and keeps the bounded-delay wording
    and its grading on the range."""

    from os_optics.results import basis_of
    from os_optics.validation import metrics as M

    document = _stated_phase_mach_zehnder(catalog, repr(math.pi / 2))
    for component in document["components"]:
        if component.get("source_model"):
            component["source_model"]["linewidth_nm"] = 0.05
    document["targets"] = [{"target_id": "power", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.0,
                            "unit": "W"}]
    run = run_optical_project(document, catalog=catalog_document())
    (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
    (sum_,) = observation.superpositions
    assert sum_.value_basis is None and sum_.total_power_min_W < sum_.total_power_W
    derived = basis_of(sum_, "total_power_W")
    assert derived.kind.value == "ESTIMATE" and [x.input.value for x in derived.assumed] == ["UNSTATED_GROUP_DELAY_ZERO"]
    from os_optics.config import NumericalConfig

    value = M._coherent_power(observation, list(observation.arrivals), "W", NumericalConfig())
    assert value.basis == derived
    (target,) = run.result.targets
    assert target.assessment.value == "MET", target.message
    entry = target.value_basis["value"]
    assert entry.kind.value == "ESTIMATE" and "UNSTATED_GROUP_DELAY_ZERO" in [x.input.value for x in entry.assumed]
    assert "group delays are known within their stated bounds" in target.message


def test_a_mixed_s_and_p_state_at_coatings_of_unstated_phase_states_the_visibility_on_the_default_phases(catalog):
    """At 45 deg linear polarization the oblique coatings meet each beam in a state mixing s and p: their unstated s and p
    phases change the beams' polarization, so the polarization overlap -- and the visibility -- are stated on the trace's
    default phases (value_basis names the passes), and the coherent power's range takes every overlap (up to 1)."""
    run = _ideal_run(mach_zehnder(catalog, polarization={"kind": "LINEAR", "angle_deg": 45.0}), catalog)
    (pair,) = run.result.interference
    assert pair.coherence_factor is not None and pair.fringe_visibility is not None
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    for name in ("polarization_overlap", "coherence_factor", "fringe_visibility"):
        entry = pair.value_basis[name]
        assert entry.kind.value == "ESTIMATE" and "UNSTATED_PHASE_AS_INTERFACE" in {a.input.value for a in entry.assumed}, name
    (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
    sum_ = observation.superpositions[0]
    p = sum_.incoherent_power_W
    assert sum_.total_power_min_W == pytest.approx(0.0, abs=1e-12) and sum_.total_power_max_W == pytest.approx(2.0 * p, rel=1e-6)
    assert sum_.total_power_min_W <= sum_.total_power_W <= sum_.total_power_max_W


def test_a_pulse_train_far_from_its_revivals_states_a_bounded_delay(catalog):
    """80 MHz: a period length of 3747 mm, against a relative delay within 1.83 um of zero (6100 fs of bounds): the nearest
    revival lies 3.7 m away, so its coherence there is nil and the arms' degree of coherence is stated with its interval.
    The revival coherence over the bounded delays is checked against the Gaussian line's closed form."""
    from os_optics.optics.interference.coherence import revival_coherence_within  # noqa: PLC0415

    run = _ideal_run(mach_zehnder(catalog, pulse={"repetition_rate_hz": 8.0e7, "pulse_duration_s": 1.0e-9, "duration_definition": "FWHM"}), catalog)
    (pair,) = run.result.interference
    assert pair.coherence_factor == pytest.approx(1.0, abs=1e-12) and pair.coherence_factor_min is not None
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    lc = coherence_length_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    period = 2.99792458e11 / 8.0e7
    bound = 6100.0 * C_MM_PER_FS
    far = revival_coherence_within(0.0, bound, 0.0, lc, (period,))
    assert far == pytest.approx(math.exp(-(((period - bound) / lc) ** 2)), rel=1e-9) and far < 2.0 ** -55
    # near a revival the bounded delays reach it: the closest approach max(0, gap - B) decides
    assert revival_coherence_within(period - 2.0 * bound, bound, 0.0, lc, (period,)) == pytest.approx(math.exp(-((bound / lc) ** 2)), rel=1e-12)
    assert revival_coherence_within(period - 0.5 * bound, bound, 0.0, lc, (period,)) == 1.0


def test_a_retro_reflected_pair_behind_bounded_coatings_states_its_combined_power_as_the_models_value(interferometer, catalog):
    """M2 turned to normal incidence sends its arm's beam back along itself; a plane on that arm sees the beam and its
    reflection cross it in opposite directions. Their combined power is Pa + Pb -- the power through the plane has no cross
    term, whatever the degree of coherence -- so it reads neither the traced delay nor any stand-in: no entry, stored or
    derived, although the coatings' bounds leave the pair's degree of coherence an interval (its coherence factor and
    margin stay ESTIMATE values at the traced delay). The pair says why: ``counter_propagating``."""

    from os_optics.results import basis_of
    from os_optics.results.basis import dump_with_bases

    document = copy.deepcopy(interferometer)
    s = math.sqrt(0.5)
    (m2,) = [c for c in document["components"] if c["instance_id"] == "M2"]
    m2["nominal_pose"]["quaternion_wxyz"] = [s, s, 0.0, 0.0]
    arm = copy.deepcopy(document["observations"][0])
    arm.update(observation_id="arm")
    arm["pose"] = {"position_mm": [0.0, 50.0, 0.0], "quaternion_wxyz": [s, -s, 0.0, 0.0]}
    document["observations"].append(arm)
    document["targets"] = [t for t in document["targets"] if t["observation_id"] != "crossing_plane"]
    run = _ideal_run(document, catalog)
    (pair,) = [p for p in run.result.interference if p.observation_id == "arm"]
    assert pair.counter_propagating is True and "NO_FRINGES" in pair.issue_codes
    assert pair.coherence_factor_min < pair.coherence_factor_max  # (the bounds leave an interval)
    assert pair.combined_power_collinear_W == pair.incoherent_power_sum_W
    assert basis_of(pair, "combined_power_collinear_W") is None
    assert "combined_power_collinear_W" not in dump_with_bases(pair).get("value_basis", {})
    for field in ("coherence_factor", "coherence_margin"):
        entry = basis_of(pair, field)
        assert entry is not None and entry.kind.value == "ESTIMATE", field
        assert [x.input.value for x in entry.assumed] == ["UNSTATED_GROUP_DELAY_ZERO"], field


def test_a_coherent_power_at_an_unstated_delay_is_decided_on_its_range_as_the_guides_say():
    """The Mach-Zehnder on a catalog whose coatings bound no group delay: the coherent power is stated at no added delay
    with an ESTIMATE entry (UNSTATED_GROUP_DELAY_ZERO), and a target on it is judged on the range every delay and phase
    leave (total_power_min_W / _max_W), widened by the other light that may reach the plane -- here from 0, and without an
    upper end: reflections the trace does not follow and light left out as negligible may reach the plane. A minimum of
    0 W lies outside it and is MET; a minimum of 0.1 W, and a maximum 1.5 times total_power_max_W (outside the sum's own
    range, inside the widened interval), are not decided. The guides say so, and keep "never deciding a target" for the
    pair's point quantities only."""
    from pathlib import Path

    catalog = catalog_document()
    document = mach_zehnder(catalog)
    plain = _ideal_run(copy.deepcopy(document), catalog).result
    (total,) = [s for o in plain.observations if o.observation_id == "out" for s in o.superpositions]
    assert total.total_power_max_W is not None and total.total_power_max_W > 0.0
    document["targets"] = [
        {"target_id": "zero", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.0, "unit": "W"},
        {"target_id": "inside", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.1, "unit": "W"},
        {"target_id": "above_range", "metric": "coherent_power_W", "observation_id": "out",
         "maximum": 1.5 * total.total_power_max_W, "unit": "W"}]
    graded = {t.target_id: t for t in _ideal_run(document, catalog).result.targets}
    assert {k: t.assessment.value for k, t in graded.items()} == {"zero": "MET", "inside": "INDETERMINATE",
                                                                  "above_range": "INDETERMINATE"}, {
        k: t.message for k, t in graded.items()}
    for target in graded.values():
        entry = target.value_basis["value"]
        assert entry.kind.value == "ESTIMATE" and "UNSTATED_GROUP_DELAY_ZERO" in [x.input.value for x in entry.assumed]
    root = Path(__file__).resolve().parents[1]
    results = " ".join((root / "docs/engine/RESULTS_AND_EXPORT.md").read_text(encoding="utf-8").split())
    assert ("(`coherence_factor`, `fringe_visibility`, the pair's peak irradiance) are stated at no added delay at those "
            "passes, with `value_basis` of kind `ESTIMATE` naming them (`UNSTATED_GROUP_DELAY_ZERO`): reported, never "
            "deciding a target. `coherent_power_W` reports its value at no added delay with that entry too, and a target on "
            "it is judged on `total_power_min_W` / `total_power_max_W`") in results
    assert "the pair's peak irradiance, `coherent_power_W`) are stated at no added delay" not in results
    widened = ("widened by the other light that may reach the plane: traced light outside the superposition (ghosts), "
               "surface reflections the trace does not follow (`UNTRACED_REFLECTION`) and light left out as negligible, "
               "which may leave the interval without an upper end. A limit outside that widened interval decides the "
               "target; a limit inside it is `INDETERMINATE`")
    assert widened in results
    assert "a limit outside that range decides it" not in results
