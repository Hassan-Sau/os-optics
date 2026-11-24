"""Passes whose group delay the design does not state.

An element without a thickness in the model (an attenuator, an aperture that attenuates, a retarder, a thin lens, a
splitter plate) and a coating without a stated phase delay light by an amount the design does not state: two beams that
cross such passes differently have no known relative envelope delay, so no degree of coherence -- 1 in every bound, no
value for a coherence metric (``optics/interference/coherence.py``). A pass whose delay may depend on polarization (a
retarder, an oblique coating) agrees only for beams in the same state, and a coated face met beyond the critical angle
has the coating's unstated delay. Every power rating is fixture data declared in the test project (its value stated as a
peak irradiance), never product data.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, PowerRating, ProjectSnapshot, Target
from os_optics.contracts.materials import ConstantCoating
from os_optics.contracts.optics import ApertureStopInteraction, FaradayRotatorInteraction, FixedDensity, InterfacePhases, NeutralDensityInteraction

from helpers import run, stated_splitter
from test_group_delay_and_returns import _bk7_group

FIX = "verifier fixture rating, not product data"


def _rating(value: float, wavelength_nm: float) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength_nm, value_definition="PEAK", status="DECLARED",
                       basis=FIX)


def _surface(result, surface_id: str):
    return next(s for s in result.exposures.surfaces if s.surface_id == surface_id)


def _targets(*devices: str) -> tuple[Target, ...]:
    return tuple(Target(target_id=d.lower(), device_id=d, metric="power_rating_utilization", maximum=1.0) for d in devices)


def _thin_element_interferometer(kind: str, rating: float | None, *, lc: float = 0.3, stated: bool = True):
    """A Mach-Zehnder interferometer (532 nm, 1 W, w0 0.8 mm, L_c 0.3 mm) whose arm B
    holds a thin element -- an ND filter OD 0.1 or a full-wave retarder -- and whose arm A is longer by the envelope delay
    of 2 mm of N-BK7, (n_g - 1) 2 mm: a real 2 mm filter or wave plate puts the arms on their white-light fringe. The
    splitters state their phases (``stated``): only the thin element differs between the arms. A rated dump and a plane
    'pre' in front of it on the output port."""
    ng = _bk7_group(532.0)[1]
    g = (ng - 1.0) * 2.0
    delta = (2.0 * 100.0 * g + g * g) / (2.0 * (100.0 + g))
    u = (-delta / math.hypot(delta, 100.0), 100.0 / math.hypot(delta, 100.0), 0.0)
    inv = 1.0 / math.sqrt(2.0)
    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.8, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(90.0), coherence_length_mm=lc)
    bs1 = B.npbs("BS1", position=(0.0, 0.0, 0.0), normal=(-inv, inv, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    mb = B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7)
    ma = B.mirror_for_reflection("MA", position=(100.0 + delta, 0.0, 0.0), d_in=(1, 0, 0), d_out=u, clear_radius_mm=12.7)
    bisector = (1.0 - u[0], -u[1], 0.0)
    norm = math.sqrt(sum(x * x for x in bisector))
    bs2 = B.npbs("BS2", position=(100.0, 100.0, 0.0), normal=tuple(x / norm for x in bisector), reflectance=0.5, clear_radius_mm=12.7)
    if stated:
        bs1, bs2 = stated_splitter(bs1), stated_splitter(bs2)
    if kind == "nd":
        element = B.polarizer("X", position=(50.0, 100.0, 0.0), axis=(1.0, 0.0, 0.0), angle_deg=0.0, clear_radius_mm=12.7)
        s = element.surfaces[0]
        element = element.model_copy(update={"role": "attenuator", "surfaces": (s.model_copy(update={
            "interaction": NeutralDensityInteraction(law=FixedDensity(optical_density=0.1))}),)})
    elif kind in ("stop", "open_stop"):
        element = B.aperture_stop("X", position=(50.0, 100.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7)
        s = element.surfaces[0]
        element = element.model_copy(update={"surfaces": (s.model_copy(update={
            "interaction": ApertureStopInteraction(transmittance=10.0 ** -0.1 if kind == "stop" else 1.0)}),)})
    else:
        element = B.waveplate("X", position=(50.0, 100.0, 0.0), axis=(1.0, 0.0, 0.0), fast_axis_angle_deg=0.0,
                              retardance_rad=2.0 * math.pi, design_wavelength_nm=532.0, clear_radius_mm=12.7)
        if kind == "faraday":  # a full turn: the polarization is unchanged, as for the full-wave retarder
            s = element.surfaces[0]
            element = element.model_copy(update={"surfaces": (s.model_copy(update={"interaction": FaradayRotatorInteraction(
                wavelength_nm=(532.0,), rotation_rad=(2.0 * math.pi,), transmittance=(1.0,), interpolation="points_only")}),)})
    dump = B.beam_dump("D", position=(100.0 + 40.0 * u[0], 100.0 + 40.0 * u[1], 0.0), axis=u, radius_mm=12.7)
    targets: tuple[Target, ...] = ()
    if rating is not None:
        # (the test beam: either arm's 1/e^2 diameter at the dump, 50 + 100 + 100 + 40 mm from the 0.8 mm waist; s-polarized
        # linear light, as the source's through the 45 deg splitters and folds)
        beam_mm = 2.0 * 0.8 * math.sqrt(1.0 + (290.0 / (math.pi * 0.64 / (532.0 * 1e-6))) ** 2)
        rated = _rating(rating, 532.0).model_copy(update={"beam_diameter_mm": beam_mm, "beam_diameter_definition": "1/E2",
                                                          "polarization": "LINEAR", "polarization_condition": "linear"})
        dump = dump.model_copy(update={"power_ratings": (rated,)})
        targets = _targets("D")
    pre = B.observation_plane("pre", position=(100.0 + 39.0 * u[0], 100.0 + 39.0 * u[1], 0.0), normal=u, radius_mm=12.7)
    return ProjectSnapshot(components=(src, bs1, mb, ma, element, bs2, dump), observations=(pre,), targets=targets)


@pytest.mark.parametrize("kind", ["nd", "retarder", "stop", "faraday"])
def test_a_thin_element_in_one_arm_states_the_arms_coherence_on_the_convention_and_bounds_with_none(kind: str):
    """The thin element's delay was taken as 0 -- an ND filter, a retarder, an aperture that attenuates (OD 0.1): the arms 3.7 L_c apart, gamma 1.5e-6, the dump bounded by the incoherent
    sum and a rating between it and the white-light-fringe peak WITHIN / PASS. The arms' relative delay is not known: the
    dump's bounds reach the coherent peak (gamma 1) and the rating is not decided. The plane's coherence metrics are stated
    on the trace's convention (no added delay at the element: gamma 1.5e-6, against 1 for a real 2 mm filter) with an
    ESTIMATE entry naming the pass, so a peak-irradiance or coherence-factor maximum there is reported and not decided."""
    probe = run(_thin_element_interferometer(kind, 1e9))
    dump = _surface(probe, "D/dump")
    peaks = sorted((h.peak_irradiance_W_per_cm2 for h in dump.hits), reverse=True)
    coherent = (math.sqrt(peaks[0]) + math.sqrt(peaks[1])) ** 2
    (comparison,) = dump.ratings
    assert comparison.exposure_upper_bound >= coherent * (1.0 - 1e-12) and dump.coherent_groups[0].largest_coherence_factor == 1.0
    rating = 0.5 * (peaks[0] + peaks[1] + coherent)
    verdict = run(_thin_element_interferometer(kind, rating))
    assert _surface(verdict, "D/dump").ratings[0].status == "UNKNOWN" and verdict.targets[0].status.value == "UNKNOWN"
    (pair,) = [p for p in probe.interference if p.observation_id == "pre"]
    a, b = probe.observations[0].arrivals
    event = {"nd": "neutral_density", "retarder": "retarder", "stop": "aperture_stop", "faraday": "faraday_rotator"}[kind]
    (record,) = (*a.group_delay_unknown_at, *b.group_delay_unknown_at)  # (the splitters state their phases; the mirrors ideal)
    assert record.startswith(f"X/{'stop' if kind == 'stop' else 'plate'}:{event}:")
    assert "COHERENCE_UNDEFINED" not in pair.issue_codes
    g = (_bk7_group(532.0)[1] - 1.0) * 2.0  # (the arms' group path difference without the element: 2 mm of N-BK7's excess)
    assert abs(pair.group_path_difference_mm) == pytest.approx(g, rel=1e-6)
    assert pair.coherence_factor == pytest.approx(math.exp(-((pair.group_path_difference_mm / 0.3) ** 2)), rel=1e-9)
    assert pair.coherence_factor < 1e-5 and pair.peak_irradiance_W_per_mm2 is not None
    entry = pair.value_basis["coherence_factor"]
    (assumed,) = entry.assumed
    assert (entry.kind.value, assumed.input.value, assumed.at) == ("ESTIMATE", "UNSTATED_GROUP_DELAY_ZERO", (record.split("@")[0],))
    assert pair.value_basis["peak_irradiance_W_per_mm2"] == pair.value_basis["group_path_difference_mm"] == entry
    project = _thin_element_interferometer(kind, None)
    for metric, maximum in (("peak_irradiance_W_per_mm2", 0.66), ("coherence_factor", 0.5)):
        target = Target(target_id="t", metric=metric, observation_id="pre", maximum=maximum)
        (outcome,) = run(project.model_copy(update={"targets": (target,)})).targets
        assert outcome.assessment.value == "INDETERMINATE" and outcome.value == getattr(pair, metric)
        assert outcome.value_basis["value"] == entry


def test_a_thin_element_both_arms_share_keeps_their_coherence_known():
    """The same pass on both beams' way (an attenuator before the first splitter) delays both alike: their relative delay,
    and their degree of coherence, are known."""
    project = _thin_element_interferometer("nd", None)
    element = next(c for c in project.components if c.instance_id == "X")
    shared = element.model_copy(update={"pose": element.pose.model_copy(update={"position_mm": (-25.0, 0.0, 0.0)})})
    project = project.model_copy(update={"components": tuple(shared if c.instance_id == "X" else c for c in project.components)})
    result = run(project)
    (pair,) = [p for p in result.interference if p.observation_id == "pre"]
    a, b = result.observations[0].arrivals
    assert a.group_delay_unknown_at == b.group_delay_unknown_at and len(a.group_delay_unknown_at) == 1
    assert pair.group_path_difference_mm is not None and pair.coherence_factor is not None


@pytest.mark.parametrize("phases", [False, True])
def test_a_coating_without_a_stated_phase_states_the_coherence_on_the_convention(phases: bool):
    """An AR coating on a window in one arm (the other arm bare): the coating's group delay is not stated, so the arms'
    relative delay is not known -- the coherence is stated at no added delay there, with an entry naming the two coated
    passes -- unless the coating states its transmission phase constant over wavelength (no delay): then the same number
    with no entry (the independent reference: stating the convention's own phases gives the convention's value)."""
    project = _thin_element_interferometer("nd", None)
    ar = ConstantCoating(reflectance_s=0.0025, reflectance_p=0.0025, transmittance_s=0.9975, transmittance_p=0.9975)
    stated = InterfacePhases(transmission_s_rad=(0.0,), transmission_p_rad=(0.0,), reflection_s_rad=(0.0,), reflection_p_rad=(0.0,))
    window = B.window("X", thickness_mm=2.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 100.0, 0.0))
    update = {"coating": ar, **({"phases": stated} if phases else {})}
    window = window.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update=update)})
                                                         for s in window.surfaces)})
    project = project.model_copy(update={"components": tuple(window if c.instance_id == "X" else c for c in project.components)})
    result = run(project)
    (pair,) = [p for p in result.interference if p.observation_id == "pre"]
    assert pair.coherence_factor is not None and pair.coherence_factor > 0.99  # (the window's own group delay balances the arms)
    assert "COHERENCE_UNDEFINED" not in pair.issue_codes
    if phases:
        assert pair.value_basis is None
    else:
        (assumed,) = pair.value_basis["coherence_factor"].assumed
        assert assumed.input.value == "UNSTATED_GROUP_DELAY_ZERO" and len(assumed.at) == 2
        window = window.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(
            update={"phases": stated})}) for s in window.surfaces)})
        stated_run = run(project.model_copy(update={"components": tuple(window if c.instance_id == "X" else c for c in project.components)}))
        (reference,) = [p for p in stated_run.interference if p.observation_id == "pre"]
        assert pair.coherence_factor == reference.coherence_factor and reference.value_basis is None


def _polarization_interferometer(variant: str, *, same_state: bool = False):
    """A polarization Mach-Zehnder (532 nm, w0 0.8 mm, linear 45 deg, coherence length
    5 um): plate PBS1 splits s and p, two uncoated mirrors, plate PBS2 recombines them collinear; both splitters' phases
    stated, equal arms. On the common path behind PBS2: (A) a thin half-wave plate with its fast axis along one beam's
    polarization, then a polarizer at 45 deg; (B) a 45 deg fold coated R 0.99 without a stated phase, meeting one beam as
    s and the other as p, then the polarizer. ``same_state``: (B) with the polarizer at 45 deg before the fold instead,
    which puts both beams in one state there."""
    from os_optics.contracts import JonesVector

    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.8, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(45.0), coherence_length_mm=0.005)
    pbs1 = stated_splitter(B.pbs("PBS1", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)), clear_radius_mm=12.7))
    ma = B.mirror_for_reflection("MA", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7)
    mb = B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7)
    pbs2 = stated_splitter(B.pbs("PBS2", position=(100.0, 100.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)), clear_radius_mm=12.7))
    parts = [src, pbs1, ma, mb, pbs2]
    if variant == "A":
        parts.append(B.half_waveplate("HWP", position=(100.0, 150.0, 0.0), axis=(0.0, 1.0, 0.0), fast_axis_angle_deg=0.0,
                                      design_wavelength_nm=532.0, clear_radius_mm=12.7))
        parts.append(B.polarizer("POL", position=(100.0, 170.0, 0.0), axis=(0.0, 1.0, 0.0), angle_deg=45.0, clear_radius_mm=12.7))
        obs = B.observation_plane("OBS", position=(100.0, 200.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=15.0)
    else:
        if same_state:
            parts.append(B.polarizer("PRE", position=(100.0, 125.0, 0.0), axis=(0.0, 1.0, 0.0), angle_deg=45.0, clear_radius_mm=12.7))
        fold = B.mirror_for_reflection("F", position=(100.0, 150.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7)
        s = fold.surfaces[0]
        c = ConstantCoating(reflectance_s=0.99, reflectance_p=0.99, transmittance_s=0.01, transmittance_p=0.01)
        parts.append(fold.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": c})}),)}))
        if not same_state:
            parts.append(B.polarizer("POL", position=(130.0, 150.0, 0.0), axis=(1.0, 0.0, 0.0), angle_deg=45.0, clear_radius_mm=12.7))
        obs = B.observation_plane("OBS", position=(160.0, 150.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=15.0)
    targets = (Target(target_id="gamma", observation_id="OBS", metric="coherence_factor", minimum=0.9),
               Target(target_id="vis", observation_id="OBS", metric="fringe_visibility", minimum=0.9))
    return ProjectSnapshot(components=tuple(parts), observations=(obs,), targets=targets)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_a_common_pass_in_two_polarization_states_leaves_the_coherence_unknown(variant: str):
    """The verifier's polarization interferometer: the two beams cross the same thin half-wave plate (A) or the same
    coated 45 deg fold (B) in orthogonal states -- the plate delays its fast and slow axes, the coating its s and p
    components, by amounts the design does not state. Their records of that pass now carry the beam's state
    ('[s1,s2,s3]') and differ: their relative delay is not known, the coherence targets are UNKNOWN (once PASS)."""
    result = run(_polarization_interferometer(variant))
    a, b = next(o for o in result.observations if o.observation_id == "OBS").arrivals
    assert a.group_delay_unknown_at != b.group_delay_unknown_at
    states = [r for r in (*a.group_delay_unknown_at, *b.group_delay_unknown_at) if not r.startswith("POL/")]
    assert states and all("[" in r for r in states)
    assert all(t.status.value == "UNKNOWN" for t in result.targets)


def test_a_common_coated_pass_in_one_polarization_state_keeps_the_coherence_known():
    """Both beams turned to one state (a polarizer at 45 deg before the fold): the fold's unstated delay is the same for
    both and cancels -- their degree of coherence is known."""
    result = run(_polarization_interferometer("B", same_state=True))
    a, b = next(o for o in result.observations if o.observation_id == "OBS").arrivals
    assert sorted(a.group_delay_unknown_at) == sorted(b.group_delay_unknown_at)
    (pair,) = [p for p in result.interference if p.observation_id == "OBS"]
    assert pair.coherence_factor is not None


@pytest.mark.parametrize("coated", [False, True])
def test_a_coated_face_beyond_the_critical_angle_has_the_coating_s_unstated_delay(coated: bool):
    """A Mach-Zehnder in N-BK7 whose arm A is folded by total internal reflection at a
    glass-air face. Bare, the reflection's delay is known (the coherence follows); with an AR coating that states no phase
    the coating's own delay under total reflection is not stated -- its record ('coated_tir') leaves the arms' coherence
    unknown (once: the coating skipped, PASS)."""
    from os_optics.contracts import JonesVector
    from os_optics.contracts.optics import RefractInteraction

    glass = "N-BK7"
    ar = ConstantCoating(reflectance_s=0.005, reflectance_p=0.005, transmittance_s=0.995, transmittance_p=0.995)
    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.8, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(45.0), coherence_length_mm=0.005, medium=glass)
    bs1 = stated_splitter(B.npbs("BS1", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)), clear_radius_mm=12.7, medium=glass))
    ma = B.mirror_for_reflection("MA", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7, medium=glass)
    s = ma.surfaces[0]
    ma = ma.model_copy(update={"surfaces": (s.model_copy(update={"medium_negative": "air",
                                                                 "interaction": RefractInteraction(coating=ar if coated else None)}),)})
    mb = B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7, medium=glass)
    bs2 = stated_splitter(B.npbs("BS2", position=(100.0, 100.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)), clear_radius_mm=12.7, medium=glass))
    obs = B.observation_plane("OBS", position=(160.0, 100.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=15.0, medium=glass)
    target = Target(target_id="gamma", observation_id="OBS", metric="coherence_factor", minimum=0.9)
    result = run(ProjectSnapshot(components=(src, bs1, ma, mb, bs2), observations=(obs,), targets=(target,)))
    records = [r for a in next(o for o in result.observations if o.observation_id == "OBS").arrivals for r in a.group_delay_unknown_at]
    (outcome,) = result.targets
    if coated:
        assert any(":coated_tir:" in r for r in records) and outcome.status.value == "UNKNOWN"
    else:
        assert records == [] and outcome.status.value == "PASS"




def _thin_mach_zehnder(extra_mm: float, *, lc: float = 100.0) -> ProjectSnapshot:
    """A Mach-Zehnder of two thin 50 / 50 splitters whose phases are not stated (their passes are open in phase and bound
    no delay) and two ideal mirrors; mirror MA moved by ``(extra / 4, -extra / 4)`` (along its normal, ``extra / (2 sqrt 2)``)
    lengthens its arm's optical path by ``extra_mm / 2`` and moves that beam sideways in both ports alike. Both output
    ports of BS2 are observed: 'x' (+X) and 'y' (+Y)."""
    inv = 1.0 / math.sqrt(2.0)
    src = B.laser("S", wavelength_nm=532.0, power_W=1.0, waist_radius_mm=0.8, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(90.0), coherence_length_mm=lc)
    bs1 = B.npbs("BS1", position=(0.0, 0.0, 0.0), normal=(-inv, inv, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    ma = B.mirror_for_reflection("MA", position=(100.0 + extra_mm / 4.0, -extra_mm / 4.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0),
                                 clear_radius_mm=12.7)
    mb = B.mirror_for_reflection("MB", position=(0.0, 100.0, 0.0), d_in=(0, 1, 0), d_out=(1, 0, 0), clear_radius_mm=12.7)
    bs2 = B.npbs("BS2", position=(100.0, 100.0, 0.0), normal=(-inv, inv, 0.0), reflectance=0.5, clear_radius_mm=12.7)
    planes = (B.observation_plane("x", position=(160.0, 100.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=15.0),
              B.observation_plane("y", position=(100.0, 160.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=15.0))
    return ProjectSnapshot(components=(src, bs1, ma, mb, bs2), observations=planes)


def test_the_two_ports_of_a_mach_zehnder_on_the_default_phases_conserve_the_power():
    """Energy conservation on the trace's default splitter phases (r = i sqrt(R) beside t = sqrt(T): a unitary 2 x 2 matrix
    for each polarization): at every arm length the two output ports receive together exactly the 1 W launched -- their
    interference terms cancel -- each port's coherent power is stated (value_basis: the default phases at the splitters'
    passes, no added delay there) inside its range over every phase, and the power follows cos^2 of half the arm phase."""
    lam_mm = 532e-6
    totals = []
    for step in range(5):  # (the arm lengthened by lam / 8 each step: its phase advances by pi / 4)
        result = run(_thin_mach_zehnder(step * lam_mm / 4.0))
        sums = {o.observation_id: o.superpositions for o in result.observations}
        (x,), (y,) = sums["x"], sums["y"]
        assert x.incoherent_power_W + y.incoherent_power_W == pytest.approx(1.0, rel=1e-12)
        assert x.total_power_W + y.total_power_W == pytest.approx(1.0, rel=1e-9)
        for s in (x, y):
            assert s.total_power_min_W - 1e-15 <= s.total_power_W <= s.total_power_max_W + 1e-15
            entry = s.value_basis["total_power_W"]
            assert entry.kind.value == "ESTIMATE"
            assert {a.input.value for a in entry.assumed} == {"UNSTATED_PHASE_AS_INTERFACE", "UNSTATED_GROUP_DELAY_ZERO"}
        totals.append(x.total_power_W)
    # (the x port follows (1 + cos(phi - offset)) / 2 with phi the added arm phase: the offset from phi = 0 and pi / 2)
    phases = [step * math.pi / 4.0 for step in range(5)]
    offset = math.atan2(2.0 * totals[2] - 1.0, 2.0 * totals[0] - 1.0)
    for t, p in zip(totals, phases):
        assert t == pytest.approx(0.5 * (1.0 + math.cos(p - offset)), abs=1e-6)


def test_an_open_aperture_delays_nothing():
    """An aperture that passes everything inside it (transmittance 1) only bounds the beam: no undetermined delay, so the
    arms' coherence follows from their group paths (``group_path_difference_mm`` and ``coherence_factor`` have values)."""
    result = run(_thin_element_interferometer("open_stop", None))
    a, b = result.observations[0].arrivals
    assert a.group_delay_unknown_at == b.group_delay_unknown_at == ()
    (pair,) = [p for p in result.interference if p.observation_id == "pre"]
    assert pair.group_path_difference_mm is not None and pair.coherence_factor is not None
