"""What two beams' sources state beyond one line's coherence (``optics/interference/source_coherence.py``).

* A continuous spectrum: two beams of a source that states a spectral density interfere with the spectrum's coherence, not with
  the coherence of the quadrature line (a bin), which is a property of the discretization. Reference: the spectrum's own degree
  of coherence ``gamma(D) = integral S(lambda) exp(i D (k - k_line)) dlambda / integral S`` from NumPy's complex exponential on a
  dense trapezoid grid, for a Gaussian, a Lorentzian and a top-hat spectrum on an ideal Michelson (splitter of stated constant
  phases, so the group path difference is twice the mirror offset).
* A partially polarized source sharing a coherence group with a polarized one: only its polarized part is coherent with the
  other source, so an unpolarized source gives the same result whichever way it is turned about its own axis, and a partially
  polarized one has the degree of coherence ``sqrt(2 P / (1 + P))`` with the other source (its component 0's coherent share).
* The two components of one partially polarized line are counted on their own, not as mutually incoherent sources.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot, SpectralDensity, SpectralLine, Target
from os_optics.contracts.optics import BeamSplitterInteraction, InterfacePhases
from os_optics.local import run_optical_project
from os_optics.optics.interference import spectrum as S

from helpers import run
from local_helpers import plane, project, source

C_GAUSSIAN = SpectralDensity(shape={"kind": "gaussian", "center_nm": 800.0, "fwhm_nm": 20.0}, band_nm=(770.0, 830.0), line_count=15)
C_LORENTZIAN = SpectralDensity(shape={"kind": "lorentzian", "center_nm": 800.0, "fwhm_nm": 2.0}, band_nm=(780.0, 820.0), line_count=16)
C_TOP_HAT = SpectralDensity(shape={"kind": "top_hat", "center_nm": 600.0, "fwhm_nm": 200.0}, band_nm=(500.0, 700.0), line_count=16)


def michelson(density: SpectralDensity, offset_mm: float, *targets: Target) -> ProjectSnapshot:
    """The reviewer's ideal Michelson: a lossless 50/50 splitter with constant stated phases, arm M1 longer by ``offset_mm``
    (group path difference ``2 offset``, no dispersion)."""
    lines = S.quadrature_lines(density)
    laser = B.laser("L", spectrum=tuple(lines), position=(-100.0, 0.0, 0.0), waist_radius_mm=0.5)
    src = laser.source.model_copy(update={"spectral_density": density, "linewidth_nm": S.bin_width_nm(density),
                                          "spectrum": tuple(SpectralLine(wavelength_nm=w, weight=p) for w, p in lines)})
    laser = laser.model_copy(update={"source": type(src).model_validate(src.model_dump())})
    bs = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)))
    surface = bs.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2,), reflection_p_rad=(math.pi / 2,), transmission_s_rad=(0.0,),
                             transmission_p_rad=(0.0,))
    interaction = BeamSplitterInteraction.model_validate(surface.interaction.model_copy(update={"phases": phases}).model_dump())
    bs = bs.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": interaction}),)})
    m1 = B.mirror("M1", position=(100.0 + offset_mm, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    m2 = B.mirror("M2", position=(0.0, 100.0, 0.0), normal=(0.0, -1.0, 0.0))
    out = B.observation_plane("out", position=(0.0, -50.0, 0.0), normal=(0.0, -1.0, 0.0), radius_mm=10.0)
    return ProjectSnapshot(project_id="m", components=(laser, bs, m1, m2), observations=(out,), targets=targets)


def density_values(density: SpectralDensity, lam: np.ndarray) -> np.ndarray:
    shape = density.shape
    if shape.kind == "gaussian":
        return np.exp(-4.0 * math.log(2.0) * ((lam - shape.center_nm) / shape.fwhm_nm) ** 2)
    if shape.kind == "lorentzian":
        return 1.0 / (1.0 + (2.0 * (lam - shape.center_nm) / shape.fwhm_nm) ** 2)
    return np.ones_like(lam)


def reference_gamma(density: SpectralDensity, delay_mm: float, reference_nm: float, samples: int = 400001) -> complex:
    """``integral S exp(i D (k - k_ref)) dlambda / integral S`` (dense trapezoid, NumPy's complex exponential)."""
    lam = np.linspace(*density.band_nm, samples)
    s = density_values(density, lam)
    k = 2.0 * np.pi / (lam * 1e-6)
    field = np.trapezoid(s * np.exp(1j * delay_mm * (k - 2.0 * np.pi / (reference_nm * 1e-6))), lam)
    return complex(field / np.trapezoid(s, lam))


@pytest.mark.parametrize("density, offset_mm", [(C_GAUSSIAN, 0.015), (C_GAUSSIAN, 0.004), (C_LORENTZIAN, 0.03), (C_TOP_HAT, 0.0005)],
                         ids=["gaussian-30um", "gaussian-8um", "lorentzian-60um", "top-hat-1um"])
def test_a_continuous_spectrum_interferes_with_the_spectrum_coherence_not_its_line(density: SpectralDensity, offset_mm: float) -> None:
    result = run(michelson(density, offset_mm))
    pairs = sorted(result.interference, key=lambda p: p.wavelength_nm)
    assert len(pairs) == density.line_count
    (spectrum,) = result.source_spectra
    for pair in pairs:
        delay = pair.group_path_difference_mm
        assert delay == pytest.approx(-2.0 * offset_mm, abs=1e-12)
        gamma = reference_gamma(density, delay, pair.wavelength_nm)
        # the coherence factor and the visibility: the spectrum's |gamma(D)| (the beams are balanced and overlap fully)
        assert pair.coherence_factor == pytest.approx(abs(gamma), abs=2e-7)
        assert pair.fringe_visibility == pytest.approx(abs(gamma), abs=1e-4)
        # the spectrum's coherence length, not the bin's (2 sqrt(ln 2) / pi) lambda^2 / w
        assert pair.coherence_length_mm == spectrum.coherence_length_mm
        assert pair.coherence_margin == pytest.approx(1.0 - abs(delay) / spectrum.coherence_length_mm, rel=1e-12)
        # the line's share of the combined power: P_a + P_b + 2 sqrt(P_a P_b) Re[gamma exp(i phi)] (phi the phase at the line)
        cross = (pair.combined_power_collinear_W - pair.incoherent_power_sum_W) / pair.incoherent_power_sum_W
        want = (gamma * complex(math.cos(pair.phase_difference_rad), math.sin(pair.phase_difference_rad))).real
        assert cross == pytest.approx(want, abs=2e-5)
    # the plane's coherent sum of each line agrees with the pair (the same model)
    for sup, pair in zip(sorted(result.observations[0].superpositions, key=lambda s: s.wavelength_nm), pairs, strict=True):
        assert sup.total_power_W == pytest.approx(pair.combined_power_collinear_W, rel=1e-9)


def test_the_reviewer_michelson_visibility_target_is_not_met_on_the_line_coherence() -> None:
    """Probe p8b: 30 um of path difference on a 20 nm Gaussian spectrum at 800 nm: each line's coherence of its 4 nm bin gave
    0.88 and decided ``fringe_visibility >= 0.5`` MET; the spectrum's |gamma(0.03 mm)| is 0.0446."""
    lines = S.quadrature_lines(C_GAUSSIAN)
    centre = min((w for w, _ in lines), key=lambda w: abs(w - 800.0))
    target = Target(target_id="vis", observation_id="out", metric="fringe_visibility", minimum=0.5, wavelength_nm=centre)
    result = run(michelson(C_GAUSSIAN, 0.015, target))
    (t,) = result.targets
    assert t.assessment.value == "UNMET" and t.status.value == "FAIL"
    assert t.value == pytest.approx(abs(reference_gamma(C_GAUSSIAN, 0.03, centre)), abs=1e-4)
    assert t.value == pytest.approx(0.0445573, abs=1e-6)


def test_the_pair_coherence_does_not_depend_on_the_number_of_lines() -> None:
    values = {}
    for count in (15, 31, 61):
        density = C_GAUSSIAN.model_copy(update={"line_count": count})
        result = run(michelson(density, 0.015))
        values[count] = sorted({round(p.coherence_factor, 9) for p in result.interference})
    assert values[15] == values[31] == values[61] == [pytest.approx(0.044557325, abs=2e-9)]


def test_dispersion_and_coating_bounds_read_the_spectrum() -> None:
    """``spectrum.pair_coherence`` with a group-delay-dispersion difference against the reference with the quadratic phase, and
    ``mean_wavenumber_offset`` against a dense trapezoid of ``|k - k_ref|``."""
    lam = np.linspace(770.0, 830.0, 400001)
    s = density_values(C_GAUSSIAN, lam)
    x = 2.0 * np.pi / (lam * 1e-6) - 2.0 * np.pi / (800.0 * 1e-6)
    for delay, dispersion in ((0.01, 0.0), (0.01, 2e-5), (0.0, -5e-5)):
        want = np.trapezoid(s * np.exp(1j * (delay * x + 0.5 * dispersion * x * x)), lam) / np.trapezoid(s, lam)
        got = S.pair_coherence(C_GAUSSIAN, 800.0, delay, dispersion)
        assert got.real == pytest.approx(want.real, abs=1e-8) and got.imag == pytest.approx(want.imag, abs=1e-8)
    want = np.trapezoid(s * np.abs(x), lam) / np.trapezoid(s, lam)
    assert S.mean_wavenumber_offset(C_GAUSSIAN, 800.0) == pytest.approx(want, rel=3e-6)  # (raised by OFFSET_MARGIN = 1e-6)
    assert S.mean_wavenumber_offset(C_GAUSSIAN, 800.0) >= want * (1.0 - 1e-9)
    assert S.pair_coherence(C_GAUSSIAN, 800.0, 1e6, 0.0) is None  # (beyond the integral's resolution: no value)


# ------------------------------------------------------------------------------------------------ partial polarization


def crossing(up_b, dop: float = 0.0, angle_b: float = 0.0, angle_a: float = 90.0, *targets: Target) -> ProjectSnapshot:
    """Probe p5: source A (linear, fully polarized) and source Bs (degree of polarization ``dop``) of one coherence group cross on
    a plane; ``up_b`` turns Bs about its own beam axis."""
    a = B.laser("A", wavelength_nm=633.0, position=(0.0, -5.0, 0.0), direction=B.unit((100.0, 5.0, 0.0)),
                polarization=JonesVector.linear(angle_a), coherence_length_mm=1000.0, coherence_group="g", waist_radius_mm=1.0)
    b = B.laser("Bs", wavelength_nm=633.0, position=(0.0, 5.0, 0.0), direction=B.unit((100.0, -5.0, 0.0)), up=up_b,
                polarization=JonesVector.linear(angle_b), coherence_length_mm=1000.0, coherence_group="g", waist_radius_mm=1.0)
    b = b.model_copy(update={"source": b.source.model_copy(update={"degree_of_polarization": dop})})
    out = B.observation_plane("P", position=(100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0)
    return ProjectSnapshot(project_id="p", components=(a, b), observations=(out,), targets=targets)


def test_an_unpolarized_source_in_a_shared_group_gives_the_same_result_turned_about_its_axis() -> None:
    """The unpolarized source's light has no fixed phase relation to the polarized source's: no pair of them has fringes, and
    the plane's coherent power is the sum of the powers, however the source is turned (p5 gave 0 or 0.94)."""
    results = [run(crossing(up)) for up in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), B.unit((0.0, 1.0, 1.0)))]
    for result in results:
        coherent = [p for p in result.interference if p.coherence_factor is not None]
        assert coherent and all(p.coherence_factor == 0.0 and p.fringe_visibility == 0.0 for p in coherent)
        (sup,) = result.observations[0].superpositions
        assert sup.total_power_W == pytest.approx(sup.incoherent_power_W, rel=1e-15)
        assert sup.total_power_W == pytest.approx(2e-3, rel=1e-6)
    # the run's verdict and its issues do not depend on the turn either
    assert len({(r.status.value, tuple(sorted(i.code.value for i in r.issues))) for r in results}) == 1


@pytest.mark.parametrize("dop", [0.25, 0.5, 0.9])
def test_a_partially_polarized_source_is_coherent_with_another_source_through_its_polarized_part(dop: float) -> None:
    """Bs's polarized part along A's polarization: the cross term of A (power P_A) with Bs (power P_B, degree P) is
    ``2 sqrt(P_A P P_B)`` times the fields' overlap, so its component 0 (power (1 + P) P_B / 2) pairs with A with the degree of
    coherence ``sqrt(2 P / (1 + P))``; its component 1 is in a group of its own."""
    result = run(crossing((0.0, 0.0, 1.0), dop, angle_b=90.0))
    (pair,) = [p for p in result.interference if p.coherence_factor is not None]
    assert {pair.path_id_a, pair.path_id_b} == {"A#0", "Bs#0"}
    assert pair.coherence_factor == pytest.approx(math.sqrt(2.0 * dop / (1.0 + dop)), rel=1e-9)
    # the plane's interference power: 2 sqrt(P_A P P_B) eta_sp cos(phi) of the whole Bs, independent of its decomposition
    (sup,) = result.observations[0].superpositions
    whole = run(crossing((0.0, 0.0, 1.0), 1.0, angle_b=90.0)).observations[0].superpositions[0]
    assert sup.interference_power_W == pytest.approx(math.sqrt(dop) * whole.interference_power_W, rel=1e-9)


# ------------------------------------------------------------------------------------------------ component pairs


def test_the_components_of_a_partially_polarized_line_are_not_counted_as_incoherent_sources() -> None:
    """Probe p1's counts: one source of degree 0.4 over 8 lines; each line's two components form same-wavelength pairs that were
    counted as pairs of mutually incoherent sources in 2 coherence groups."""
    s = source()
    s["source_model"]["polarization"] = {"kind": "PARTIAL", "polarized": {"kind": "LINEAR", "angle_deg": 30.0},
                                         "degree_of_polarization": 0.4}
    del s["source_model"]["wavelength_nm"]
    s["source_model"]["spectral_density"] = {"shape": {"kind": "GAUSSIAN", "center_nm": 780.0, "fwhm_nm": 20.0}, "line_count": 8}
    result = run_optical_project(project([s], [plane("OUT", (100.0, 0.0, 0.0))]), catalog=None).result
    assert not result.incoherent_pairs
    assert "INCOHERENT_SOURCES" not in {i.code.value for i in result.issues}
    (issue,) = [i for i in result.issues if i.code.value == "PARTIAL_POLARIZATION_COMPONENTS"]
    assert issue.numerical_context["pair_count"] == 8 and issue.status.value == "NOT_APPLICABLE"


def test_a_partial_source_beside_an_independent_one_counts_two_coherence_groups() -> None:
    a, b = source(), source("SRC_B", (-200.0, 5.0, 0.0))
    b["source_model"]["polarization"] = {"kind": "UNPOLARIZED"}
    result = run_optical_project(project([a, b], [plane("OUT", (100.0, 0.0, 0.0))]), catalog=None).result
    (counted,) = result.incoherent_pairs
    # three beams of one wavelength: A, B's two components; the components' pair is counted apart
    assert counted.pair_count == 2 and counted.coherence_group_count == 2
    (issue,) = [i for i in result.issues if i.code.value == "PARTIAL_POLARIZATION_COMPONENTS"]
    assert issue.numerical_context["pair_count"] == 1


# ------------------------------------------------------------------ one line states a coherence length, the other none


def _two_lasers(lc_a: float | None, lc_b: float | None, *targets: Target) -> ProjectSnapshot:
    """Two lasers of one coherence group crossing at 1 deg on a plane, about 1 mm of group path apart."""
    ang = math.radians(0.5)
    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, -100.0 * math.tan(ang), 0.0),
                direction=(1.0, math.tan(ang), 0.0), coherence_length_mm=lc_a, coherence_group="g")
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-101.0, 101.0 * math.tan(ang), 0.0),
                direction=(1.0, -math.tan(ang), 0.0), coherence_length_mm=lc_b, coherence_group="g")
    return ProjectSnapshot(components=(a, b), observations=(B.observation_plane("P", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),),
                           targets=targets)


@pytest.mark.parametrize("stated, unstated", [("A", "B"), ("B", "A")])
def test_one_stated_coherence_length_gives_an_upper_end_that_names_the_unstated_source_and_decides_nothing(stated, unstated):
    """The unstated source may have a much shorter coherence length: the pair's degree of coherence at the stated length is
    only an upper end. It is the value both sources stating that length give (a second calculation), with an UPPER_BOUND
    entry whose FULL_COHERENCE stand-in names the unstated source only; coherence length and margin have no value; minimum
    targets on the coherence factor and the fringe visibility are not decided; the plane's coherent sum carries the ESTIMATE
    entry that names the unstated source."""

    from os_optics.results import basis_of

    targets = (Target(target_id="cf", metric="coherence_factor", observation_id="P", minimum=0.9, unit="1"),
               Target(target_id="vis", metric="fringe_visibility", observation_id="P", minimum=0.9, unit="1"))
    lengths = {stated: 5.0, unstated: None}
    one = run(_two_lasers(lengths["A"], lengths["B"], *targets))
    both = run(_two_lasers(5.0, 5.0, *targets))
    (pair,), (reference,) = one.interference, both.interference
    assert 0.0 < reference.coherence_factor < 1.0 and basis_of(reference, "coherence_factor") is None
    assert pair.coherence_factor == reference.coherence_factor
    assert pair.coherence_length_mm is None and pair.coherence_margin is None
    for field in ("coherence_factor", "fringe_visibility"):
        entry = basis_of(pair, field)
        assert entry.kind.value == "UPPER_BOUND", field
        assert [(x.input.value, x.at) for x in entry.assumed] == [("FULL_COHERENCE", (f"source:{unstated}",))], field
    assert {t.target_id: t.assessment.value for t in one.targets} == {"cf": "INDETERMINATE", "vis": "INDETERMINATE"}
    assert {t.target_id: t.assessment.value for t in both.targets} == {"cf": "MET", "vis": "MET"}
    (sup,) = one.observations[0].superpositions
    entry = basis_of(sup, "total_power_W")
    assert entry.kind.value == "ESTIMATE" and [x.at for x in entry.assumed if x.input.value == "FULL_COHERENCE"] == [(f"source:{unstated}",)]
    # (crossing beams: the fringes average out over the plane, so the sum is the two powers whatever the coherence, the same
    # value and range as with both sources unstated)
    (unstated_sum,) = run(_two_lasers(None, None)).observations[0].superpositions
    assert (sup.total_power_W, sup.total_power_min_W, sup.total_power_max_W) == (
        unstated_sum.total_power_W, unstated_sum.total_power_min_W, unstated_sum.total_power_max_W)
    assert sup.total_power_W == pytest.approx(sup.incoherent_power_W, rel=1e-6)


def _folded_lasers(lc_a: float | None, lc_b: float | None, bound_fs: float, *targets: Target) -> ProjectSnapshot:
    """Two lasers of one coherence group crossing at 1 deg on a plane; B reaches it about 101 mm of group path later by a
    mirror whose coating bounds its group delay (``group_delay_bound_fs``), which A does not meet."""
    ang = math.radians(0.5)
    da, db = (math.cos(ang), math.sin(ang), 0.0), (math.cos(ang), -math.sin(ang), 0.0)
    m = tuple(-100.0 * x for x in db)
    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=tuple(-100.0 * x for x in da),  # type: ignore[arg-type]
                direction=da, coherence_length_mm=lc_a, coherence_group="g")
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(m[0], m[1] - 101.0, 0.0),
                direction=(0.0, 1.0, 0.0), coherence_length_mm=lc_b, coherence_group="g")
    mirror = B.mirror_for_reflection("M", position=m, d_in=(0.0, 1.0, 0.0), d_out=db, reflectance=1.0)  # type: ignore[arg-type]
    (surf,) = mirror.surfaces
    coating = surf.interaction.coating.model_copy(update={"group_delay_bound_fs": bound_fs})  # type: ignore[union-attr]
    surf = surf.model_copy(update={"interaction": surf.interaction.model_copy(update={"coating": coating})})
    return ProjectSnapshot(components=(a, b, mirror.model_copy(update={"surfaces": (surf,)})),
                           observations=(B.observation_plane("P", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),),
                           targets=targets)


@pytest.mark.parametrize("bound_fs", [10000.0, 50000.0])
def test_one_stated_coherence_length_behind_bounded_coating_delays_is_judged_up_to_the_end_the_bounds_allow(bound_fs):
    """One line states its coherence length (80 mm), the other none; the beams meet a coating of bounded group delay
    differently. The degree of coherence at the traced delay is no upper end: the bounds allow delays down to |D| - B, where
    the stated line's coherence is larger (closed form exp(-((|D| - B) / L)^2), no dispersion in air), and the change the
    bounds allow, 2 B / (sqrt(pi) L), grows without limit as the unstated line's own length L shrinks. The pair states the
    upper end as ``coherence_factor_max``: the line's whole degree of coherence, 1, above the end both lines stating 80 mm
    give (a second calculation) and the closed form -- and no lower end: the unstated line may be incoherent. Its quantities
    are ESTIMATE values at the traced delay, judged from 0 to the upper end: a maximum or a minimum between the traced value
    and the closed-form end is not decided; a maximum at the end is met; a minimum above the both-stated end, which that pair
    decides, is not decided; a maximum below the both-stated lower end is not decided."""

    from os_optics.results import basis_of

    one = run(_folded_lasers(80.0, None, bound_fs))
    both = run(_folded_lasers(80.0, 80.0, bound_fs))
    (pair,), (reference,) = one.interference, both.interference
    delay, bound = abs(pair.group_path_difference_mm), pair.group_delay_bound_mm
    assert delay == pytest.approx(101.0, rel=1e-9) and bound == pytest.approx(bound_fs * 2.99792458e-4, rel=1e-9)
    closed = math.exp(-((delay - bound) / 80.0) ** 2)
    g = pair.coherence_factor
    assert g == pytest.approx(math.exp(-(delay / 80.0) ** 2), rel=1e-9) and g < closed
    assert pair.coherence_factor_min is None
    assert pair.coherence_factor_max == 1.0 and 1.0 > reference.coherence_factor_max >= closed
    for field in ("coherence_factor", "fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2"):
        entry = basis_of(pair, field)
        assert entry.kind.value == "ESTIMATE", field
        assert [(x.input.value, x.at) for x in entry.assumed] == [
            ("FULL_COHERENCE", ("source:B",)), ("UNSTATED_GROUP_DELAY_ZERO", ("M/face:reflect:+",))], field
    middle = 0.5 * (g + closed)
    vis_middle = pair.fringe_visibility * middle / g
    mean = pair.mean_irradiance_W_per_mm2
    peak_middle = mean + (pair.peak_irradiance_W_per_mm2 - mean) * middle / g
    targets = (Target(target_id="cf_max", metric="coherence_factor", observation_id="P", maximum=middle, unit="1"),
               Target(target_id="cf_min", metric="coherence_factor", observation_id="P", minimum=middle, unit="1"),
               Target(target_id="vis_max", metric="fringe_visibility", observation_id="P", maximum=vis_middle, unit="1"),
               Target(target_id="peak_max", metric="peak_irradiance_W_per_mm2", observation_id="P", maximum=peak_middle,
                      unit="W/mm^2"),
               Target(target_id="cf_max_end", metric="coherence_factor", observation_id="P",
                      maximum=pair.coherence_factor_max, unit="1"),
               Target(target_id="cf_min_above", metric="coherence_factor", observation_id="P",
                      minimum=1.01 * reference.coherence_factor_max, unit="1"),
               Target(target_id="cf_max_low", metric="coherence_factor", observation_id="P", maximum=0.5 * g, unit="1"))
    graded = {t.target_id: t for t in run(_folded_lasers(80.0, None, bound_fs, *targets)).targets}
    expected = {"cf_max": "INDETERMINATE", "cf_min": "INDETERMINATE", "vis_max": "INDETERMINATE", "peak_max": "INDETERMINATE",
                "cf_max_end": "MET", "cf_min_above": "INDETERMINATE", "cf_max_low": "INDETERMINATE"}
    assert {k: t.assessment.value for k, t in graded.items()} == expected, {k: t.message for k, t in graded.items()}
    assert f"between 0 and {pair.coherence_factor_max:.6g}" in graded["cf_max"].message
    assert graded["cf_max"].value == g and graded["cf_max"].value_basis["value"].kind.value == "ESTIMATE"
    if reference.coherence_factor_min > 0.0:
        # (both lines stating 80 mm prove a lower end above 0.5 g; with one unstated, nothing does)
        assert reference.coherence_factor_min > 0.5 * g
        stated = {t.target_id: t for t in run(_folded_lasers(80.0, 80.0, bound_fs, *targets)).targets}
        assert stated["cf_max_low"].assessment.value == "UNMET"
        assert stated["cf_min_above"].assessment.value == "UNMET"  # (both stated: the end the bounds allow decides it)


def _combiner(lc_a: float | None, lc_b: float | None, back_mm: float, glass_mm: float, *targets: Target,
              fresnel: bool = False) -> ProjectSnapshot:
    """Two lasers of one coherence group made collinear by a lossless splitter of stated constant phases; only B crosses
    ``glass_mm`` of N-BK7 (a group-delay-dispersion difference), A starts ``back_mm`` further back; ``fresnel``: the glass's
    surface reflections are traced (no untraced reflection widens a target's interval)."""
    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0 - back_mm, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_length_mm=lc_a, coherence_group="g")
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, -100.0, 0.0),
                direction=(0.0, 1.0, 0.0), coherence_length_mm=lc_b, coherence_group="g")
    bs = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)))
    surface = bs.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2,), reflection_p_rad=(math.pi / 2,), transmission_s_rad=(0.0,),
                             transmission_p_rad=(0.0,))
    interaction = BeamSplitterInteraction.model_validate(surface.interaction.model_copy(update={"phases": phases}).model_dump())
    bs = bs.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": interaction}),)})
    parts = [a, b, bs]
    if glass_mm:
        parts.append(B.window("W", thickness_mm=glass_mm, clear_radius_mm=3.0, material="N-BK7", position=(0.0, -50.0, 0.0),
                              axis=(0.0, 1.0, 0.0), **({"trace_fresnel_reflection": True} if fresnel else {})))
    out = B.observation_plane("P", position=(50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0)
    return ProjectSnapshot(project_id="c", components=tuple(parts), observations=(out,), targets=targets)


#: A's extra distance that leaves the combiner's group path difference at 0.1 mm behind 20 mm of N-BK7 on B.
_COMBINER_BACK_MM = 10.734840972671236 + 0.1


def test_one_stated_coherence_length_with_a_dispersion_difference_is_judged_up_to_the_largest_shorter_length():
    """A states 0.03 mm, B none and crosses 20 mm of N-BK7 that A does not: the group-delay dispersions differ, and the
    degree of coherence (1 + b^2)^(-1/4) exp(-(D / L)^2 / (1 + b^2)) does not grow with L there. At D = 0.1 mm the value at
    0.03 mm is about 3e-5, while B stating 2 mm or 1.5 mm (both lines then at that length, a second calculation) gives
    about 0.06. The pair states that upper end as ``coherence_factor_max`` (at least each of those values), the entries are
    ESTIMATE (assumed FULL_COHERENCE at B), and targets are judged from 0 to that end: a maximum 0.001 is not decided (it was
    met on the stated length's value), a maximum at the end is met. The plane's coherent power range reaches the coherent
    power those lengths give, so a maximum below it is not decided either."""

    from os_optics.results import basis_of

    references = [run(_combiner(0.03, lb, _COMBINER_BACK_MM, 20.0)) for lb in (0.002, 0.0015)]
    gammas = [r.interference[0].coherence_factor for r in references]
    powers = [r.observations[0].superpositions[0].total_power_W for r in references]
    one = run(_combiner(0.03, None, _COMBINER_BACK_MM, 20.0))
    (pair,) = one.interference
    assert pair.group_path_difference_mm == pytest.approx(0.1, abs=1e-9)
    assert pair.coherence_factor < 1e-4 and min(gammas) > 0.06
    assert pair.coherence_factor_min is None and pair.coherence_factor_max >= max(gammas)
    for field in ("coherence_factor", "fringe_visibility", "fringe_visibility_peak", "peak_irradiance_W_per_mm2"):
        entry = basis_of(pair, field)
        assert entry.kind.value == "ESTIMATE", field
        assert [(x.input.value, x.at) for x in entry.assumed] == [("FULL_COHERENCE", ("source:B",))], field
    (total,) = one.observations[0].superpositions
    assert total.total_power_max_W >= max(powers) > total.total_power_W
    limit = 0.5 * (total.total_power_W + max(powers))
    targets = (Target(target_id="cf_max", metric="coherence_factor", observation_id="P", maximum=0.001, unit="1"),
               Target(target_id="cf_end", metric="coherence_factor", observation_id="P", maximum=pair.coherence_factor_max,
                      unit="1"),
               Target(target_id="vis_max", metric="fringe_visibility", observation_id="P", maximum=0.001, unit="1"),
               Target(target_id="power_max", metric="coherent_power_W", observation_id="P", maximum=limit, unit="W"))
    graded = {t.target_id: t for t in run(_combiner(0.03, None, _COMBINER_BACK_MM, 20.0, *targets)).targets}
    expected = {"cf_max": "INDETERMINATE", "cf_end": "MET", "vis_max": "INDETERMINATE", "power_max": "INDETERMINATE"}
    assert {k: t.assessment.value for k, t in graded.items()} == expected, {k: t.message for k, t in graded.items()}
    assert f"between 0 and {pair.coherence_factor_max:.6g}" in graded["cf_max"].message
    assert graded["cf_max"].value == pair.coherence_factor


def test_one_stated_coherence_length_without_a_dispersion_difference_keeps_the_stated_length_as_its_upper_end():
    """The same combiner without the glass: no dispersion difference, so the degree of coherence grows with the length and
    the value at the stated length is the upper end -- equal to both lines stating it, UPPER_BOUND, no coherence_factor_max."""

    from os_optics.results import basis_of

    (pair,) = run(_combiner(0.3, None, 0.0, 0.0)).interference
    (reference,) = run(_combiner(0.3, 0.3, 0.0, 0.0)).interference
    assert pair.coherence_factor == reference.coherence_factor and pair.coherence_factor_max is None
    assert basis_of(pair, "coherence_factor").kind.value == "UPPER_BOUND"


def _gaussian_line_gamma(delay: float, dispersion: float, length: float) -> complex:
    """The Gaussian line's complex degree of coherence in closed form with Python's complex arithmetic (an independent
    evaluation): ``(1 - i b)^(-1/2) exp(-(D / L)^2 / (1 - i b))``, ``b = 2 B / L^2``."""
    import cmath

    b = 2.0 * dispersion / (length * length)
    return (1.0 - 1j * b) ** -0.5 * cmath.exp(-((delay / length) ** 2) / (1.0 - 1j * b))


def _main_pair(result):
    """The combiner's direct pair (each source's strongest arrival on P), oriented as the pair states it."""
    main: dict = {}
    for x in result.observations[0].arrivals:
        if x.beam.source_id not in main or x.beam.power_W > main[x.beam.source_id].beam.power_W:
            main[x.beam.source_id] = x
    ids = {main["A"].beam.path_id, main["B"].beam.path_id}
    (pair,) = [x for x in result.interference if {x.path_id_a, x.path_id_b} == ids]
    a, b = (main["A"], main["B"]) if pair.path_id_a == main["A"].beam.path_id else (main["B"], main["A"])
    return pair, a, b


#: A's extra distance that leaves the combiner's group path difference at 79 nm behind 20 mm of N-BK7 on B (the phase of the
#: degree of coherence at 0.03 mm is then about -0.12 rad).
_NEAR_BACK_MM = 10.73492
#: The same at 0.1001 mm.
_FAR_BACK_MM = 10.83495


def test_a_line_pair_through_different_group_delay_dispersions_adds_its_complex_degree_of_coherence():
    """Both sources state 0.03 mm, B crosses 20 mm of N-BK7 that A does not (B = -1.14e-4 mm^2, b = -0.25), D = 79 nm. The
    two fields add to Pa + Pb + k Re[gamma exp(i phi)] with the Gaussian line's complex gamma (closed form above): its phase
    (about -0.12 rad here) shifts the fringes. The pair's combined power matches that within the rounding of the reported
    phase (1e-6 rad steps); the line model's |gamma| cos(phi) would be about 1.1e-4 W lower. The coherence factor stays
    |gamma| to the last digit. The plane's coherent sum is about 1.219e-3 W, so a coherent_power_W maximum of 1.15e-3 W is
    unmet (it was met on |gamma| cos(phi))."""
    from os_optics.optics.interference.coherence import degree_of_coherence

    target = Target(target_id="cp_max", metric="coherent_power_W", observation_id="P", maximum=1.15e-3, unit="W")
    result = run(_combiner(0.03, 0.03, _NEAR_BACK_MM, 20.0, target, fresnel=True))
    pair, a, b = _main_pair(result)
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = a.group_delay_dispersion_mm2 - b.group_delay_dispersion_mm2
    assert dispersion == pytest.approx(-1.1415e-4, rel=1e-3) and abs(delay) < 1e-4
    gamma = _gaussian_line_gamma(delay, dispersion, 0.03)
    assert abs(math.atan2(gamma.imag, gamma.real)) > 0.1
    pa, pb = a.beam.power_W, b.beam.power_W
    k = 2.0 * math.sqrt(pa * pb) * pair.polarization_overlap * pair.spatial_overlap_ratio
    expected = pa + pb + k * (gamma * complex(math.cos(pair.phase_difference_rad), math.sin(pair.phase_difference_rad))).real
    rounding = k * abs(gamma) * 0.5e-6 + 1e-15  # (the reported phase is rounded to 1e-6 rad)
    assert abs(pair.combined_power_collinear_W - expected) <= rounding
    line_model = pa + pb + k * abs(gamma) * math.cos(pair.phase_difference_rad)
    assert expected - line_model > 1e-4
    assert pair.coherence_factor == degree_of_coherence(delay, dispersion, 0.03)
    (total,) = result.observations[0].superpositions
    assert total.total_power_W == pytest.approx(1.219e-3, rel=1e-3) and total.total_power_min_W is None
    (graded,) = result.targets
    assert graded.assessment.value == "UNMET", graded.message


def test_the_lines_complex_degree_of_coherence_is_the_continuous_spectrums_of_the_same_gaussian():
    """``spectrum.pair_coherence`` integrates the spectral density with the relative phase D (k - k0) + B (k - k0)^2 / 2: for
    a density Gaussian in the wavenumber with the line's width (a table of 4001 rows over +-12 / L in k, per unit wavelength)
    it is the line's complex gamma, to the table's resolution, phase included. Without a dispersion difference the line's
    gamma is the real ``degree_of_coherence``, bit for bit."""
    from os_optics.optics.interference.coherence import complex_degree_of_coherence, degree_of_coherence

    lam, length = 633.0, 0.03
    k0 = 2.0 * math.pi / (lam * 1e-6)
    lo, hi = 2.0 * math.pi / (k0 + 12.0 / length) * 1e6, 2.0 * math.pi / (k0 - 12.0 / length) * 1e6
    rows = [lo + (hi - lo) * i / 4000 for i in range(4001)]
    density = [math.exp(-(((2.0 * math.pi / (x * 1e-6)) - k0) * length / 2.0) ** 2) * 2.0 * math.pi / (x * x) for x in rows]
    table = SpectralDensity(shape={"kind": "tabulated", "wavelength_nm": tuple(rows), "density": tuple(density)},
                            band_nm=(lo, hi), line_count=8)
    for delay, dispersion in [(0.1001, -1.1415e-4), (7.9e-5, -1.1415e-4), (0.02, 4e-4), (0.0, 3e-4), (0.03, 0.0)]:
        line = complex_degree_of_coherence(delay, dispersion, length)
        spectrum = S.pair_coherence(table, lam, delay, dispersion)
        assert abs(complex(line) - spectrum) < 3e-6, (delay, dispersion, line, spectrum)
        assert abs(complex(line) - _gaussian_line_gamma(delay, dispersion, length)) < 1e-15
    assert complex_degree_of_coherence(0.03, 0.0, length) == degree_of_coherence(0.03, 0.0, length)
    assert not isinstance(complex_degree_of_coherence(0.03, 0.0, length), complex)


def test_a_one_stated_pair_through_different_dispersions_ranges_every_phase_up_to_its_upper_end():
    """A states 0.03 mm, B none and crosses the glass, D = 0.1001 mm. A shorter length L' of B's line moves the phase of the
    complex gamma (towards -pi/4 here) as well as its magnitude: over L' up to 0.03 mm (closed form on 20001 lengths) the
    pair's power reaches about 1.006e-3 W, above Pa + Pb + k |gamma(L')| cos(phi) at every L' (about 9.59e-4 W). The pair's
    combined_power_collinear_min_W / _max_W and the plane's total_power_min_W / _max_W hold every one of those values, and a
    coherent_power_W maximum of 9.8e-4 W, above the value but below that reach, is not decided."""
    target = Target(target_id="cp_max", metric="coherent_power_W", observation_id="P", maximum=9.8e-4, unit="W")
    result = run(_combiner(0.03, None, _FAR_BACK_MM, 20.0, target, fresnel=True))
    pair, a, b = _main_pair(result)
    delay = a.group_path_length_mm - b.group_path_length_mm
    dispersion = a.group_delay_dispersion_mm2 - b.group_delay_dispersion_mm2
    pa, pb = a.beam.power_W, b.beam.power_W
    k = 2.0 * math.sqrt(pa * pb) * pair.polarization_overlap * pair.spatial_overlap_ratio
    turn = complex(math.cos(pair.phase_difference_rad), math.sin(pair.phase_difference_rad))
    powers = [pa + pb + k * (_gaussian_line_gamma(delay, dispersion, x) * turn).real
              for x in np.linspace(0.03 * 1e-4, 0.03, 20001)]
    assert max(powers) == pytest.approx(1.006e-3, rel=2e-3) and max(powers) > 9.8e-4 > pair.combined_power_collinear_W
    assert pair.combined_power_collinear_min_W <= min(powers) and max(powers) <= pair.combined_power_collinear_max_W
    (total,) = result.observations[0].superpositions
    others = total.total_power_W - pair.combined_power_collinear_W  # (the glass's traced reflections)
    assert 0.0 < others < 2e-5
    assert total.total_power_max_W >= others + max(powers) and total.total_power_min_W <= others + min(powers)
    (graded,) = result.targets
    assert graded.assessment.value == "INDETERMINATE", graded.message


def _stated_phase_fold(lc_a: float | None, lc_b: float | None, bound_fs: float | None, *targets: Target,
                      glass_mm: float = 0.0) -> ProjectSnapshot:
    """Two lasers of one coherence group made collinear by a lossless splitter of stated constant phases; only B meets a
    fold mirror whose coating bounds its group delay (``group_delay_bound_fs``; None: no bound) and states its phases
    (tabulated), so the pair's phase is stated and its delay known within the bound (B's group path 150 mm longer);
    ``glass_mm``: N-BK7 on B's arm after the mirror, its surface reflections traced."""
    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_length_mm=lc_a, coherence_group="g")
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-150.0, -100.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_length_mm=lc_b, coherence_group="g")
    bs = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)))
    surface = bs.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2,), reflection_p_rad=(math.pi / 2,), transmission_s_rad=(0.0,),
                             transmission_p_rad=(0.0,))
    interaction = BeamSplitterInteraction.model_validate(surface.interaction.model_copy(update={"phases": phases}).model_dump())
    bs = bs.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": interaction}),)})
    mirror = B.mirror_for_reflection("M", position=(0.0, -100.0, 0.0), d_in=(1.0, 0.0, 0.0), d_out=(0.0, 1.0, 0.0),
                                     reflectance=1.0)
    (surf,) = mirror.surfaces
    coating = surf.interaction.coating.model_copy(update={"group_delay_bound_fs": bound_fs})  # type: ignore[union-attr]
    stated = InterfacePhases(wavelength_nm=(400.0, 1100.0), reflection_s_rad=(math.pi, math.pi), reflection_p_rad=(0.0, 0.0),
                             transmission_s_rad=(0.0, 0.0), transmission_p_rad=(0.0, 0.0))
    surf = surf.model_copy(update={"interaction": surf.interaction.model_copy(update={"coating": coating, "phases": stated})})
    out = B.observation_plane("P", position=(50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0)
    parts = [a, b, bs, mirror.model_copy(update={"surfaces": (surf,)})]
    if glass_mm:
        parts.append(B.window("W", thickness_mm=glass_mm, clear_radius_mm=3.0, material="N-BK7", position=(0.0, -50.0, 0.0),
                              axis=(0.0, 1.0, 0.0), trace_fresnel_reflection=True))
    return ProjectSnapshot(project_id="c", components=tuple(parts), observations=(out,), targets=targets)


def test_a_one_stated_coherent_sum_behind_bounded_delays_names_the_traced_delay_as_its_pair_does():
    """A states 80 mm, B none; only B meets a mirror of 100 fs group-delay bound with stated phases. The pair's entries
    name FULL_COHERENCE at B and UNSTATED_GROUP_DELAY_ZERO at the mirror pass (the value is at the traced delay); the
    plane's coherent sum stands on the same two and its entry says so, and a coherent_power_W target carries it. The
    target's text names the stand-in at the other source's stated length and the traced delays, and the bounded passes --
    not "at full coherence": the value is at a degree of coherence of about 0.03, the range reaches about 2e-3 W at 1."""
    from os_optics.results import basis_of

    target = Target(target_id="cp", metric="coherent_power_W", observation_id="P", minimum=0.0, unit="W")
    result = run(_stated_phase_fold(80.0, None, 100.0, target))
    (pair,) = result.interference
    expected = {("FULL_COHERENCE", ("source:B",)), ("UNSTATED_GROUP_DELAY_ZERO", ("M/face:reflect:+",))}
    assert {(x.input.value, x.at) for x in basis_of(pair, "coherence_factor").assumed} == expected
    assert 0.02 < pair.coherence_factor < 0.04
    (total,) = result.observations[0].superpositions
    entry = basis_of(total, "total_power_W")
    assert entry.kind.value == "ESTIMATE" and {(x.input.value, x.at) for x in entry.assumed} == expected
    assert total.total_power_max_W > 1.9e-3 > 1.1e-3 > total.total_power_W
    (graded,) = result.targets
    assert graded.value_basis["value"] == entry
    assert "at full coherence" not in graded.message
    assert ("the coherent sum with source B taken as monochromatic, at the other source's stated coherence length and the "
            "traced delays, is") in graded.message
    assert "beams of it meet coatings differently whose group delays are known within their stated bounds" in graded.message


def test_a_one_stated_pair_behind_an_unbounded_stated_phase_pass_ranges_up_to_full_coherence():
    """A states 80 mm, B none; only B meets the mirror (tabulated phases, no group-delay bound) and 20 mm of N-BK7: the
    pair's phase is stated, its delay is not known, and the dispersions differ. The unbounded delay leaves any degree of
    coherence up to B's polarized share (1 here), so the pair's combined power ranges over
    Pa + Pb +- 2 sqrt(Pa Pb) eta_pol share eta_sp, about [0, 1.92e-3] W -- not +-1.7e-5 W about the value at the stated
    length's degree of coherence (0.018). It is at least as wide as the range behind a 100 fs bound on the same mirror (a
    tighter constraint), and agrees with the plane's total_power_max_W less the glass's traced reflections."""
    result = run(_stated_phase_fold(80.0, None, None, glass_mm=20.0))
    pair, a, b = _main_pair(result)
    assert pair.coherence_factor_max is None and 0.01 < pair.coherence_factor < 0.03
    pa, pb = a.beam.power_W, b.beam.power_W
    largest = 2.0 * math.sqrt(pa * pb) * pair.polarization_overlap * 1.0 * pair.spatial_overlap_ratio  # (share 1)
    assert pair.combined_power_collinear_max_W == pytest.approx(pa + pb + largest, rel=1e-12)
    assert pair.combined_power_collinear_min_W == pytest.approx(max(0.0, pa + pb - largest), abs=1e-15)
    low, high = pair.combined_power_collinear_min_W, pair.combined_power_collinear_max_W
    assert low <= pair.combined_power_collinear_W <= high
    assert high == pytest.approx(1.92e-3, rel=1e-2)
    bounded, _, _ = _main_pair(run(_stated_phase_fold(80.0, None, 100.0, glass_mm=20.0)))
    assert low <= bounded.combined_power_collinear_min_W and bounded.combined_power_collinear_max_W <= high
    (total,) = result.observations[0].superpositions
    others = total.total_power_W - pair.combined_power_collinear_W  # (the glass's traced reflections)
    assert 0.0 < others < 2e-5
    # (the plane's upper end: this pair's at full coherence, the reflections' powers, and at most every cross term a
    # reflection forms with another beam)
    powers = [x.beam.power_W for x in result.observations[0].arrivals]
    main = {a.beam.path_id, b.beam.path_id}
    cross = sum(2.0 * math.sqrt(p * q) for i, (x, p) in enumerate(zip(result.observations[0].arrivals, powers))
                for y, q in zip(result.observations[0].arrivals[i + 1:], powers[i + 1:]) if {x.beam.path_id, y.beam.path_id} != main)
    assert others + high <= total.total_power_max_W * (1.0 + 1e-12) <= (others + high + cross) * (1.0 + 1e-12)


def test_the_coherent_map_of_a_one_stated_pair_names_only_the_source_taken_as_monochromatic():
    """A states 0.03 mm, B none: the coherent irradiance map is at A's stated length with B taken as monochromatic, so its
    entry names FULL_COHERENCE at source:B only, as the pair and the plane's coherent sum do -- not at A, whose length is
    stated. The same holds for the map summed over the plane's wavelengths (a third laser at 532 nm, of its own group)."""
    from os_optics.results import basis_of
    from os_optics.results.irradiance import irradiance_map

    snapshot = _combiner(0.03, None, _COMBINER_BACK_MM, 20.0)
    result = run(snapshot)
    pair, _, _ = _main_pair(result)
    assert [x.at for x in basis_of(pair, "coherence_factor").assumed if x.input.value == "FULL_COHERENCE"] == [("source:B",)]
    (total,) = result.observations[0].superpositions
    assert [x.at for x in basis_of(total, "total_power_W").assumed if x.input.value == "FULL_COHERENCE"] == [("source:B",)]
    doc = irradiance_map(result, "P", samples=81)
    assert doc["status"] == "OK" and doc["coherent_W_per_mm2"] is not None
    (entry,) = doc["value_basis"]["coherent_W_per_mm2"]["assumed"]
    assert entry["input"] == "FULL_COHERENCE" and entry["at"] == ["source:B"]
    green = B.laser("C", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, 1.0, 0.0),
                    direction=(1.0, 0.0, 0.0), coherence_group="h")
    summed = irradiance_map(run(snapshot.model_copy(update={"components": (*snapshot.components, green)})), "P", samples=81)
    assert summed["status"] == "OK" and summed["wavelength_nm"] is None and summed["coherent_W_per_mm2"] is not None
    (entry,) = summed["value_basis"]["coherent_W_per_mm2"]["assumed"]
    assert entry["input"] == "FULL_COHERENCE" and entry["at"] == ["source:B"]


def test_a_one_stated_coherent_sum_through_different_dispersions_is_worded_at_the_stated_length():
    """The combiner with A stating 0.03 mm, B none, behind the glass (no bounded pass): the sum is at the other source's
    stated length, where the pair's degree of coherence is about 3e-5, not at full coherence; the target says so."""
    target = Target(target_id="cp", metric="coherent_power_W", observation_id="P", minimum=0.0, unit="W")
    result = run(_combiner(0.03, None, _FAR_BACK_MM, 20.0, target, fresnel=True))
    pair, _, _ = _main_pair(result)
    assert pair.coherence_factor < 1e-4
    (graded,) = result.targets
    assert "at full coherence" not in graded.message
    assert ("the coherent sum with source B taken as monochromatic, at the other source's stated coherence length and the "
            "traced delays, is") in graded.message
    assert "known within their stated bounds" not in graded.message and "does not state" not in graded.message


@pytest.mark.parametrize("delay, dispersion, length", [(0.1, 7.9e-5, 0.03), (0.1, -7.9e-5, 0.03), (1.0, 1e-3, 5.0),
                                                       (0.05, 2e-6, 0.004), (0.3, 1e-2, 0.2), (2.0, 1e-9, 1.0)])
def test_the_largest_degree_of_coherence_over_shorter_lengths_is_an_upper_end_of_a_dense_scan(delay, dispersion, length):
    """``largest_within_length`` against NumPy on 200001 lengths up to the stated one (an independent scan of the closed
    form): never below any of them, and within a part in 1e-6 of their largest (the scan's own resolution)."""
    from os_optics.optics.interference.coherence import largest_within_length

    lengths = np.linspace(length * 1e-4, length, 200001)
    b = 2.0 * dispersion / lengths ** 2
    u = 1.0 + b * b
    scan = u ** -0.25 * np.exp(-((delay / lengths) ** 2) / u)
    top = largest_within_length(delay, dispersion, length)
    assert top >= float(scan.max())
    assert top <= float(scan.max()) * (1.0 + 1e-6) + 1e-300


@pytest.mark.parametrize("lc_b", [5.0, None])
def test_the_peak_irradiance_end_the_bounds_allow_survives_a_vanishing_traced_coherence(lc_b):
    """Coherence lengths of 5 mm and a coating bound of 300000 fs: the traced delay (101 mm) leaves a degree of coherence of
    about 6e-178, so the peak irradiance at the traced delay equals the mean to the last digit. The bounds allow delays down
    to |D| - B (about 11 mm), where the stated line's degree of coherence is exp(-((|D| - B) / L)^2) (closed form, about
    0.0075): the peak irradiance may reach mean (1 + V / g exp(-((|D| - B) / L)^2)) with V / g the local visibility per unit
    degree of coherence (about 1, read from the stored visibility, which keeps its digits). The interval a target is judged
    on reaches at least that far: a maximum half-way between the mean and it is not decided; a maximum above the stated
    upper end at coherence_factor_max is met."""
    import re

    (pair,) = run(_folded_lasers(5.0, lc_b, 300000.0)).interference
    delay, bound, g = abs(pair.group_path_difference_mm), pair.group_delay_bound_mm, pair.coherence_factor
    mean = pair.mean_irradiance_W_per_mm2
    assert 0.0 < g < 1e-170 and pair.peak_irradiance_W_per_mm2 == mean  # (the excess is below the mean's last digit)
    assert pair.coherence_factor_max == 1.0 and (pair.coherence_factor_min is None) == (lc_b is None)
    reach = math.exp(-((delay - bound) / 5.0) ** 2)
    per_degree = pair.fringe_visibility_peak / g
    assert 0.007 < reach < 0.008 and per_degree == pytest.approx(1.0, abs=1e-3)
    reachable = mean * (1.0 + per_degree * reach)
    targets = (Target(target_id="half", metric="peak_irradiance_W_per_mm2", observation_id="P", maximum=0.5 * (mean + reachable),
                      unit="W/mm^2"),
               Target(target_id="above", metric="peak_irradiance_W_per_mm2", observation_id="P", maximum=2.0 * mean,
                      unit="W/mm^2"))
    graded = {t.target_id: t for t in run(_folded_lasers(5.0, lc_b, 300000.0, *targets)).targets}
    assert graded["half"].assessment.value == "INDETERMINATE", graded["half"].message
    assert graded["above"].assessment.value == "MET", graded["above"].message
    ends = re.search(r"the peak irradiance is between (\S+) W/mm\^2 and (\S+) W/mm\^2", graded["half"].message)
    assert ends is not None, graded["half"].message
    low, high = float(ends.group(1)), float(ends.group(2))  # (the message's six significant digits)
    assert low == pytest.approx(mean, rel=1e-5) and high >= reachable
    # (at the stated upper end of 1 the excess is the full local visibility of the mean)
    assert high == pytest.approx(mean * (1.0 + per_degree), rel=1e-5)


def test_the_engines_guides_state_the_one_stated_pair_and_its_bounded_delay_case():
    """The guides an Agent reads in the Engine (help of the local runtime, the lowering module and ``target_metrics()``,
    the coherence metrics' descriptors, the source model's docstring in both copies) state what the run above reports: a
    pair where one line states its coherence length has values at that length, behind coatings of bounded group delay an
    ``ESTIMATE`` with ``coherence_factor_max`` and no ``coherence_factor_min``, a target judged from 0 up to that end; and
    none of them still says every such value is at a degree of coherence of 1 judged on [0, 1]."""
    import importlib

    from os_optics import local
    from os_optics.local import lowering
    from os_optics.validation.descriptors import metric_descriptors

    descriptors = {d.name: d.validity for d in metric_descriptors()}
    texts = {"local": local.__doc__, "lowering": lowering.__doc__, "target_metrics": lowering.target_metrics.__doc__,
             **{f"descriptor {m}": descriptors[m] for m in ("coherence_factor", "fringe_visibility", "fringe_visibility_peak",
                                                            "peak_irradiance_W_per_mm2")}}
    for name in ("os_optics._vendor.shared_contracts.optical_design.source", "shared_contracts.optical_design.source"):
        try:
            texts[name] = importlib.import_module(name).__doc__
        except ModuleNotFoundError:  # (the shared-contracts package is not installed beside the Engine)
            continue
    for name, text in texts.items():
        flat = " ".join((text or "").split())
        assert "coherence length where" in flat or "coherence length, where" in flat, name
        assert "coherence_factor_max" in flat, name
        for stale in ("the degree of coherence of its beams is 1", "judged on the interval of degrees 0 to 1",
                      "the value stated at a degree of coherence of 1 with its value_basis entry and a target judged"):
            assert stale not in flat, (name, stale)
    for name in ("local", "lowering"):
        flat = " ".join(texts[name].split())
        assert "no ``coherence_factor_min``" in flat and "UNSTATED_GROUP_DELAY_ZERO" in flat, name
    # (the case a Project reaches -- one source, no stated coherence -- is an upper end at a degree of coherence of 1 with
    # no coherence_factor_min or _max also behind bounded coatings (test_bounded_coating_coherence); the ESTIMATE up to
    # coherence_factor_max belongs to the pair whose other source states its length, a pair of two sources only a shared
    # coherence_group forms)
    sources = [name for name in texts if name.endswith("optical_design.source")]
    assert sources, "the source model's docstring is read"
    for name in (*sources, "target_metrics"):
        flat = " ".join(texts[name].split())
        assert "with no ``coherence_factor_min`` or ``coherence_factor_max``" in flat or (
            "no ``coherence_factor_min`` or ``coherence_factor_max``" in flat and "``UPPER_BOUND``" in flat), name
        assert "``UPPER_BOUND``" in flat and "coherence_group" in flat, name
        for stale in ("judged from 0 up to that value (behind coatings of bounded group delay, where the values are "
                      "``ESTIMATE`` values at the traced delay, up to the value at the pair's ``coherence_factor_max``)",
                      "a target on one of them is judged from 0 up to that value (behind coatings of bounded group delay, up "
                      "to the value at the pair's ``coherence_factor_max``)"):
            assert stale not in flat, (name, stale)
    assert all("no coherence_factor_min" in " ".join(texts[f"descriptor {m}"].split())
               for m in ("coherence_factor", "peak_irradiance_W_per_mm2"))
    # (the plane's coherent power is judged on its range over degrees of coherence 0 to 1, which starts at the incoherent
    # side; "from 0 up to the larger" belongs to the pair's collinear combined power only)
    for name in sources:
        flat = " ".join(texts[name].split())
        assert ("``coherent_power_W``, ``CoherentSuperposition.total_power_W``) at a stand-in degree of coherence of 1 is an "
                "``ESTIMATE``") in flat, name
        assert "judged on ``total_power_min_W`` .. ``total_power_max_W``" in flat, name
        for found in range(flat.count("from 0 up to the larger")):
            at = flat.find("from 0 up to the larger") if found == 0 else flat.find("from 0 up to the larger", at + 1)
            assert "combined_power_collinear_W" in flat[max(0, at - 120):at], (name, flat[max(0, at - 120):at + 60])
        assert "The coherent power, which the cross term may lower or raise, is an ``ESTIMATE`` judged from 0" not in flat, name


def test_the_engine_readme_states_the_one_stated_pair_and_its_bounded_delay_case():
    """The customer README states what the run reports for a pair where one source states its coherence length: values at
    that length as upper ends, a target judged from 0 up to them; behind coatings of bounded group delay estimates with
    coherence_factor_max and no coherence_factor_min. It no longer says every such value is at a degree of coherence of 1
    judged on [0, 1]. Such a pair needs a shared coherence_group (low-level Engine API), and its coherent power is an
    estimate, not an upper end."""
    from pathlib import Path

    readme = " ".join((Path(__file__).resolve().parents[1] / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    assert ("At the other source's coherence length where that source states it, fringe visibility, peak irradiance and the "
            "coherence factor are upper ends") in readme
    assert "a target is judged from 0 up to that value (the peak irradiance from the mean irradiance)" in readme
    assert "the pair states `coherence_factor_max` and no `coherence_factor_min`" in readme
    assert "coherent power have values at a degree of coherence of 1" not in readme
    # (such a pair exists only within one coherence_group, which only the low-level Engine API sets; a Project's sources are
    # incoherent with each other)
    assert ("Two sources form such a pair only where they share a `coherence_group`, which only the low-level Engine API sets "
            "(`os_optics.builders`, `SourceSpec.coherence_group` of a `ProjectSnapshot`); a `source_model` Project gives each "
            "source its own group.") in readme
    assert ("Beams of two different source components are mutually incoherent, unless they share a `coherence_group` "
            "(low-level Engine API)") in readme
    # (the coherent power is no upper end there: an estimate graded up to the larger of it and the incoherent sum --
    # metrics._assumed_coherence)
    assert ("fringe visibility, peak irradiance and the coherence factor are upper ends (`UPPER_BOUND`, with `FULL_COHERENCE` "
            "naming only the unstated source)") in readme
    assert ("the coherent power (`combined_power_collinear_W`), which the cross term may lower or raise, is an `ESTIMATE` "
            "(assumed `FULL_COHERENCE`) judged from 0 up to the larger of the value and the beams' incoherent power sum") in readme
