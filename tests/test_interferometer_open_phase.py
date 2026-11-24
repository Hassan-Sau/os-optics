"""A catalog-built interferometer preserves geometry through unknown public-glass loss. Declared lossless copies
isolate the coating phase: unstated phases give an ESTIMATE at default interface phases beside the full coherent range,
and stated phases produce a bright-to-dark value.

The geometry is the recorded Michelson (``michelson_helpers``: the numbers below come from Snell's law and the catalog's
published index table, R, T and group-delay bounds, not from the Engine). The form of the pair's range is checked against a direct
numerical integration of the two fields over the plane.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from os_optics.config import NumericalConfig
from os_optics.errors import Status
from os_optics.local import example_project_of_kind, json_api, run_optical_project
from os_optics.local.artifacts import run_artifact_files
from os_optics.optics.gaussian.state import GState
from os_optics.optics.interference import analyse_pair
from os_optics.optics.rays.tracer import ArrivalRecord
from os_optics.results import columns, load_run
from os_optics.results.tables import monitor_row

from catalog_store_helpers import REAL_CATALOG
from declared_helpers import declared_project
from local_helpers import bounded_catalog_document
from michelson_helpers import SCAN_STEPS, WAVELENGTH_NM, expected, michelson, scan, stated_phase_michelson

ABSENT_WHERE_PHI_IS_KNOWN = ("combined_power_collinear_min_W", "combined_power_collinear_max_W", "phase_statement")
# recorded on the products of catalog 2026-10-28.1 and Engine 0.10.0 / 0.11.0; a release that moves these parts moves them
RECORDED = {"power_W": 0.0004702677828125, "range_W": (6.095e-9, 9.4053e-4), "group_delay_bound_mm": 0.6895226534,
            "coherence_factor": (0.999855, 0.997442, 1.0), "fringe_visibility": 0.999842}


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(REAL_CATALOG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def want(catalog):
    return expected(catalog)


def _on_plane(run):
    (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
    (superposition,) = observation.superpositions
    (pair,) = [i for i in run.result.interference if i.observation_id == "out"]
    return observation, superposition, pair


def _ideal_declared(document: dict) -> dict:
    """The old numeric phase experiment explicitly assumes lossless test glass.

    This declaration belongs to a copy in the Project, not to the real public
    catalog: its missing bulk-loss datum must continue to yield bounded power.
    """
    for material in document["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    return document


def _ideal_michelson(catalog: dict, extra_mm: float = 0.0, *, targets=None) -> dict:
    return _ideal_declared(declared_project(michelson(catalog, extra_mm, targets=targets), catalog))


@pytest.fixture(scope="module")
def open_scan(catalog, want):
    """Recorded coating-open scan on test-only declared copies with stated zero bulk loss."""
    return [run_optical_project(_ideal_michelson(catalog, extra), catalog=catalog) for extra in scan(want)]


@pytest.fixture(scope="module")
def stated_scan(catalog, want):
    """The same explicitly lossless copies, whose coatings additionally state their phases."""
    return [run_optical_project(_ideal_declared(stated_phase_michelson(catalog, extra)), catalog=catalog) for extra in scan(want)]


# ---------------------------------------------------------------------------------------- coatings that state no phase


def test_the_real_catalog_plate_keeps_bounded_power_and_the_michelson_path_geometry(catalog, want):
    """An unstated public bulk loss is not a declaration of zero absorption.

    It withholds numeric interference and ordinary power, but the traced arms,
    their independently calculated upper envelopes and their optical path
    difference survive unchanged. The exact coherent scans below deliberately
    add a *test-only* zero-loss assumption instead of editing public products.
    """
    run = run_optical_project(michelson(catalog), catalog=catalog)
    (observation,) = [o for o in run.result.observations if o.observation_id == "out"]
    assert run.status.value == "UNKNOWN" and run.model_scope_issues == ()
    assert observation.total_power_W is None and observation.quantity_state.value == "UNKNOWN"
    assert observation.superpositions == () and run.result.interference == ()
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
    assert len(observation.arrivals) == 2
    arm_t = next(a for a in observation.arrivals if "M1/S1:reflect" in a.lineage)
    arm_r = next(a for a in observation.arrivals if "M2/S1:reflect" in a.lineage)
    assert arm_t.beam.power_W is arm_r.beam.power_W is None
    assert arm_t.power_upper_bound_W == pytest.approx(want.power_t_W, rel=1e-12)
    assert arm_r.power_upper_bound_W == pytest.approx(want.power_r_W, rel=1e-12)
    assert arm_t.beam.optical_path_length_mm - arm_r.beam.optical_path_length_mm == pytest.approx(want.path_difference_mm, abs=1e-9)
    assert arm_t.beam.direction == pytest.approx(arm_r.beam.direction, abs=1e-12)
    assert {t.target_id: t.assessment.value for t in run.result.targets} == {"coherent": "INDETERMINATE", "ordinary": "MET"}


def test_the_ordinary_power_of_the_open_phase_scan_is_the_constant_incoherent_sum(open_scan, want):
    assert len(open_scan) == SCAN_STEPS
    for run in open_scan:
        observation, superposition, pair = _on_plane(run)
        assert observation.total_power_W == pytest.approx(want.sum_W, rel=1e-12)
        assert superposition.incoherent_power_W == pytest.approx(want.sum_W, rel=1e-12)
        assert pair.incoherent_power_sum_W == pytest.approx(want.sum_W, rel=1e-12)
        ordinary = next(t for t in run.result.targets if t.target_id == "ordinary")
        assert ordinary.value == pytest.approx(want.sum_W, rel=1e-12)


def test_the_open_phase_range_comes_from_the_group_delay_bounds_and_is_not_the_sum(open_scan, want):
    for run, extra in zip(open_scan, scan(want)):
        _, superposition, pair = _on_plane(run)
        low, high = want.sum_W - want.largest_cross_term_W, want.sum_W + want.largest_cross_term_W
        for name, value in (("pair", (pair.combined_power_collinear_min_W, pair.combined_power_collinear_max_W)),
                            ("plane", (superposition.total_power_min_W, superposition.total_power_max_W))):
            assert value[0] == pytest.approx(low, abs=1e-10), name
            assert value[1] == pytest.approx(high, rel=1e-8), name
            assert value[0] < 1e-3 * want.sum_W and value[1] > 1.99 * want.sum_W, name  # not the sum: from near 0 to near twice it
            assert value[0] < want.sum_W < value[1], name
        assert pair.group_delay_bound_mm == pytest.approx(want.bound_mm, rel=1e-12)
        assert pair.coherence_factor == pytest.approx(want.gamma(extra), abs=1e-8)
        assert pair.coherence_factor_min == pytest.approx(want.gamma(extra) - want.delta, abs=1e-8)
        assert pair.coherence_factor_max == 1.0  # (|gamma| + delta is above 1)
        assert pair.spatial_overlap_ratio == pytest.approx(want.spatial_overlap, abs=1e-7)


def test_the_open_phase_is_stated_on_the_default_phases_and_follows_the_two_beam_form(open_scan, want):
    """The phase between the arms and the coherent power are stated at the trace's default interface phases, each with an
    ESTIMATE entry; the value follows the closed two-beam form ``P_T + P_R + 2 sqrt(P_T P_R) |gamma| eta_sp cos(k0 OPD +
    psi)`` (``michelson_helpers``, independent of the Engine) for one constant ``psi`` over the scan -- the defaults' phase
    between the arms --, inside the range over every phase; a target on it is reported and judged on that range."""
    combined = []
    for run in open_scan:
        _, superposition, pair = _on_plane(run)
        assert pair.phase_difference_rad is not None and pair.combined_power_collinear_W is not None
        assert "PHASE_UNDEFINED" not in pair.issue_codes and pair.phase_statement is None
        assert pair.combined_power_collinear_min_W <= pair.combined_power_collinear_W <= pair.combined_power_collinear_max_W
        assert pair.value_basis["phase_difference_rad"].kind.value == "ESTIMATE"
        assert pair.value_basis["combined_power_collinear_W"].kind.value == "ESTIMATE"
        assert superposition.total_power_W == pytest.approx(pair.combined_power_collinear_W, rel=1e-8, abs=1e-12)
        assert superposition.status is not Status.UNKNOWN and "PHASE_UNDEFINED" not in superposition.issue_codes
        assert superposition.total_power_min_W <= superposition.total_power_W <= superposition.total_power_max_W
        target = next(t for t in run.result.targets if t.target_id == "coherent")
        # (the target's entry is the sum's, with the light the trace does not follow that may reach the plane)
        target_entry = target.value_basis["value"]
        assert target.value == superposition.total_power_W and target_entry.kind.value == "ESTIMATE"
        assert target_entry.assumed == superposition.value_basis["total_power_W"].assumed
        combined.append(pair.combined_power_collinear_W)
    extras = scan(want)
    scale = [2.0 * math.sqrt(want.power_t_W * want.power_r_W) * want.gamma(x) * want.spatial_overlap for x in extras]
    # (a lambda / 8 mirror step turns the arm phase by pi / 2: steps 0 and 1 give the cosine and the sine of the first phase)
    first = math.atan2(-(combined[1] - want.sum_W) / scale[1], (combined[0] - want.sum_W) / scale[0])
    psi = first - 2.0 * math.pi * (want.path_difference_mm + 2.0 * extras[0]) / (WAVELENGTH_NM * 1e-6)
    for value, extra in zip(combined, extras):
        assert value == pytest.approx(want.combined_W(extra, psi), abs=1e-11)


def test_the_basis_names_the_passes_that_leave_the_phase_open(open_scan, want):
    _, superposition, pair = _on_plane(open_scan[0])
    (phase,) = pair.value_basis["phase_difference_rad"].assumed
    assert (phase.input.value, phase.at) == ("UNSTATED_PHASE_AS_INTERFACE", tuple(sorted(want.open_passes)))
    plane_passes = {a.input.value: a.at for a in superposition.value_basis["total_power_W"].assumed}
    assert plane_passes["UNSTATED_PHASE_AS_INTERFACE"] == tuple(sorted(want.open_passes))
    assert superposition.phase_statement is None
    assert not [i for i in open_scan[0].result.issues if i.code.value == "PHASE_UNDEFINED"]


def test_the_optical_path_difference_gives_the_change_of_the_phase_across_the_scan(open_scan, want):
    """What the statement says: ``optical_path_difference_mm`` has its value and moves by twice the mirror's step."""
    differences = [abs(_on_plane(run)[2].optical_path_difference_mm) for run in open_scan]
    offsets = scan(want)
    for difference, offset in zip(differences, offsets):
        assert difference == pytest.approx(want.path_difference_mm + 2.0 * offset, abs=1e-9)


def test_a_bound_outside_the_whole_range_decides_a_target_on_the_open_phase(catalog):
    """As for every interval: a requirement the whole range meets or misses is decided; inside it, it is not."""
    document = _ideal_michelson(catalog, targets=[
        {"target_id": "inside", "metric": "coherent_power_W", "observation_id": "out", "minimum": 4e-4, "unit": "W"},
        {"target_id": "below_range", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.0, "unit": "W"},
        {"target_id": "above_range", "metric": "coherent_power_W", "observation_id": "out", "minimum": 0.1, "unit": "W"}])
    outcomes = {t.target_id: t for t in run_optical_project(document, catalog=catalog).result.targets}
    assert outcomes["inside"].status is Status.UNKNOWN and outcomes["inside"].assessment.value == "INDETERMINATE"
    # (each reports the value at the default phases with its entry; the verdict comes from the proven range)
    assert outcomes["below_range"].status is Status.PASS and outcomes["below_range"].value is not None
    assert outcomes["above_range"].status is Status.FAIL and outcomes["above_range"].value is not None
    assert outcomes["inside"].value_basis["value"].kind.value == "ESTIMATE"


def test_the_recorded_michelson_numbers_hold_on_the_seed(open_scan):
    for run in open_scan:
        observation, superposition, pair = _on_plane(run)
        assert observation.total_power_W == pytest.approx(RECORDED["power_W"], rel=1e-12)
        assert superposition.total_power_min_W == pytest.approx(RECORDED["range_W"][0], abs=1e-12)
        assert superposition.total_power_max_W == pytest.approx(RECORDED["range_W"][1], abs=1e-8)
        assert pair.group_delay_bound_mm == pytest.approx(RECORDED["group_delay_bound_mm"], abs=1e-10)
        assert (pair.coherence_factor, pair.coherence_factor_min, pair.coherence_factor_max) == pytest.approx(RECORDED["coherence_factor"], abs=1e-6)
        assert pair.fringe_visibility == pytest.approx(RECORDED["fringe_visibility"], abs=1e-6)


# ----------------------------------------------------------------------------------------- coatings that state their phases


def test_stated_phases_give_a_combined_power_that_goes_bright_to_dark(stated_scan, want):
    combined = []
    for run, extra in zip(stated_scan, scan(want)):
        observation, superposition, pair = _on_plane(run)
        assert observation.total_power_W == pytest.approx(want.sum_W, rel=1e-12)  # the ordinary power stays constant
        assert pair.combined_power_collinear_W == pytest.approx(want.combined_W(extra), abs=1e-11)
        assert superposition.total_power_W == pytest.approx(pair.combined_power_collinear_W, rel=1e-8, abs=1e-12)
        assert "PHASE_UNDEFINED" not in pair.issue_codes and "PHASE_UNDEFINED" not in superposition.issue_codes
        dumped = pair.model_dump(mode="json")
        assert not set(ABSENT_WHERE_PHI_IS_KNOWN) & set(dumped) and "phase_statement" not in superposition.model_dump(mode="json")
        combined.append(pair.combined_power_collinear_W)
    # a quarter wave of optical path per step: bright, half, dark, half, ...; the recorded form (about 2 x ordinary, about 0)
    for step in (0, 4, 8):
        assert combined[step] > 1.999 * want.sum_W
    for step in (2, 6):
        assert combined[step] < 2e-3 * want.sum_W
    for step in (1, 3, 5, 7):
        assert combined[step] == pytest.approx(want.sum_W, rel=1e-5)


def test_the_stated_cases_lie_inside_the_open_range(stated_scan, open_scan):
    (_, open_sum, _) = _on_plane(open_scan[0])
    for run in stated_scan:
        _, superposition, _ = _on_plane(run)
        assert open_sum.total_power_min_W <= superposition.total_power_W <= open_sum.total_power_max_W


def test_a_stated_phase_of_a_pass_one_arm_makes_moves_the_fringe(catalog, want):
    """Arm T crosses the anti-reflection face twice more than arm R: a stated transmission phase ``psi`` moves the fringe by 2 psi."""
    extra = scan(want)[0]
    run = run_optical_project(_ideal_declared(stated_phase_michelson(catalog, extra, face_transmission_rad="0.7")), catalog=catalog)
    _, _, pair = _on_plane(run)
    assert pair.combined_power_collinear_W == pytest.approx(want.combined_W(extra, 1.4), abs=1e-11)
    assert abs(pair.phase_difference_rad) == pytest.approx(1.4, abs=2e-6)


# ------------------------------------------------------------------------------------------- the form of the range


def _record(path_id: str, power_W: float, uv: tuple[float, float], radius_mm: float, angle_deg: float, delays: tuple[str, ...],
            group_mm: float, coherence_length_mm: float) -> ArrivalRecord:
    """A beam along +Z with its waist on the plane z = 0 at ``uv``, of ``radius_mm``, linearly polarized at ``angle_deg``."""
    e1, e2, normal = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    wavelength_nm = 780.0
    q = 1j * math.pi * radius_mm * radius_mm / (wavelength_nm * 1e-6)
    angle = math.radians(angle_deg)
    return ArrivalRecord(
        observation_id="out", surface_id="observation/out", path_id=path_id, source_id="SRC", spectral_index=0, wavelength_nm=wavelength_nm,
        medium="air", n=1.0, position=np.array([uv[0], uv[1], 0.0]), direction=normal, power_W=power_W, opl_mm=100.0, length_mm=100.0,
        jones=np.array([math.cos(angle) + 0j, math.sin(angle) + 0j]), e1=e1, e2=e2,
        gstate=GState(valid=True, status=Status.PASS, reason=None, reason_text="", wavelength_nm=wavelength_nm, n=1.0, qx=q, qy=q, m2x=1.0, m2y=1.0, bx=e1, by=e2),
        uv=uv, aoi_deg=0.0, pupil_uv=np.zeros((0, 2)), pupil_total=0, coherence_group="SRC", coherence_length_mm=coherence_length_mm, lineage=(),
        plane_normal=normal, plane_u=e1, plane_v=e2, group_opl_mm=group_mm, delays=delays)


def _numerical_range(a: dict, b: dict, gamma: float) -> tuple[float, float, float]:
    """``(Pa + Pb, least, largest)`` power through the plane of two partly coherent Gaussian beams over every phase, by direct summation
    of their intensities and fields on a grid (``2 gamma Re[exp(i phi) e_a* . e_b Sum sqrt(I_a I_b)] dA``)."""
    u = np.linspace(-12.0, 12.0, 1601)
    cell = (u[1] - u[0]) ** 2
    x, y = np.meshgrid(u, u)

    def intensity(beam: dict) -> np.ndarray:
        r2 = (x - beam["uv"][0]) ** 2 + (y - beam["uv"][1]) ** 2
        return 2.0 * beam["power"] / (math.pi * beam["radius"] ** 2) * np.exp(-2.0 * r2 / beam["radius"] ** 2)

    ia, ib = intensity(a), intensity(b)
    overlap = math.cos(math.radians(a["angle"] - b["angle"]))
    cross = overlap * float(np.sum(np.sqrt(ia * ib))) * cell
    phases = np.linspace(0.0, 2.0 * math.pi, 721)
    total = float(np.sum(ia + ib)) * cell + 2.0 * gamma * cross * np.cos(phases)
    return float(np.sum(ia + ib)) * cell, float(np.min(total)), float(np.max(total))


@pytest.mark.parametrize("beam_a, beam_b, delay_mm", [
    (dict(power=3e-3, uv=(0.0, 0.0), radius=1.0, angle=0.0), dict(power=1e-3, uv=(0.4, -0.2), radius=1.3, angle=30.0), 20.0),
    (dict(power=2e-3, uv=(0.0, 0.0), radius=1.0, angle=0.0), dict(power=2e-3, uv=(0.0, 0.0), radius=1.0, angle=0.0), 0.0),
    (dict(power=5e-3, uv=(0.5, 0.0), radius=0.8, angle=10.0), dict(power=1e-4, uv=(-0.5, 0.3), radius=0.6, angle=70.0), 60.0),
])
def test_the_pair_range_is_the_numerically_integrated_range_over_every_phase(beam_a, beam_b, delay_mm):
    """``Pa + Pb -+ 2 sqrt(Pa Pb) eta_pol g eta_sp``, ``g`` the largest degree of coherence the bounds allow, the least at least 0."""
    bound_fs, lc = 100.0, 100.0
    delays_a = (f"A/S1:reflect:+@45.000000<={bound_fs!r}fs",)
    delays_b = (f"B/S1:reflect:+@45.000000<={bound_fs!r}fs",)
    a = _record("a", beam_a["power"], beam_a["uv"], beam_a["radius"], beam_a["angle"], delays_a, 100.0 + delay_mm, lc)
    b = _record("b", beam_b["power"], beam_b["uv"], beam_b["radius"], beam_b["angle"], delays_b, 100.0, lc)
    pair, _ = analyse_pair(a, b, NumericalConfig())
    delta = 2.0 * (2.0 * bound_fs * 2.99792458e-4) / (math.sqrt(math.pi) * lc)
    gamma = min(1.0, math.exp(-((delay_mm / lc) ** 2)) + delta)
    power, least, largest = _numerical_range(beam_a, beam_b, gamma)
    assert power == pytest.approx(beam_a["power"] + beam_b["power"], rel=1e-9)
    assert pair.combined_power_collinear_min_W == pytest.approx(max(0.0, least), abs=1e-9 * power)
    assert pair.combined_power_collinear_max_W == pytest.approx(largest, rel=1e-8)
    # (the value at the default phases -- here the records' own, equal arms of 100 mm optical path -- inside that range)
    assert pair.combined_power_collinear_min_W <= pair.combined_power_collinear_W <= pair.combined_power_collinear_max_W
    (phase,) = pair.value_basis["phase_difference_rad"].assumed
    assert phase.at == ("A/S1:reflect:+", "B/S1:reflect:+") and pair.phase_statement is None


def test_a_phase_statement_lists_the_first_passes_and_counts_the_rest():
    """Where the passes of open phase also state no optical path (here thin lenses) the phase has no value and the statement
    names them; the coherence is stated at no added delay there (their delay is not bounded either)."""
    delays_a = tuple(f"S{i:02d}/lens:thin_lens:+@0.000000" for i in range(11))
    a = _record("a", 1e-3, (0.0, 0.0), 1.0, 0.0, delays_a, 100.0, 1000.0)
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 1000.0)
    pair, issues = analyse_pair(a, b, NumericalConfig())
    assert "S07/lens:thin_lens:+ and 3 more state no phase" in pair.phase_statement and "S08" not in pair.phase_statement
    assert len(next(i for i in issues if i.code.value == "PHASE_UNDEFINED").numerical_context["passes"]) == 8
    assert pair.phase_difference_rad is None and pair.combined_power_collinear_W is None
    assert len(pair.value_basis["coherence_factor"].assumed[0].at) == 11  # (the basis names every pass)


def test_a_phase_undefined_names_the_element_without_an_optical_path_and_never_a_coating_phase():
    """PHASE_UNDEFINED arises only where a pass one beam makes states neither its phase nor its optical path (here an ideal
    thin lens in one arm of a combiner of stated phases); a coating without a stated phase alone gives the phase on the
    default phases with an ESTIMATE entry (``test_phase_difference_open.py``). So the issue, the pair's statement and a
    target on the phase difference name that element and its remedy -- model it with its surfaces / thickness and
    material -- and never ask for the coating's phases, which do not clear the code."""

    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot, Target
    from os_optics.contracts.optics import BeamSplitterInteraction, InterfacePhases

    from helpers import run

    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0), coherence_length_mm=50.0, coherence_group="g")
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, -100.0, 0.0),
                direction=(0.0, 1.0, 0.0), coherence_length_mm=50.0, coherence_group="g")
    bs = B.npbs("BS", position=(0.0, 0.0, 0.0), normal=B.unit((-1.0, 1.0, 0.0)))
    surface = bs.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2,), reflection_p_rad=(math.pi / 2,), transmission_s_rad=(0.0,),
                             transmission_p_rad=(0.0,))
    interaction = BeamSplitterInteraction.model_validate(surface.interaction.model_copy(update={"phases": phases}).model_dump())
    bs = bs.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": interaction}),)})
    lens = B.thin_lens("TL", focal_length_mm=1e6, clear_radius_mm=5.0, position=(0.0, -50.0, 0.0), axis=(0.0, 1.0, 0.0))
    out = B.observation_plane("P", position=(50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0)
    target = Target(target_id="phi", metric="phase_difference_rad", observation_id="P", source_id="A", source_id_b="B",
                    minimum=0.0, unit="rad")
    result = run(ProjectSnapshot(project_id="c", components=(a, b, bs, lens), observations=(out,), targets=(target,)))
    (pair,) = [x for x in result.interference if "PHASE_UNDEFINED" in x.issue_codes]
    assert pair.phase_difference_rad is None and pair.combined_power_collinear_W is None
    assert pair.phase_statement.endswith("To obtain the value, model the element with its thickness and material.")
    (issue,) = [i for i in result.issues if i.code.value == "PHASE_UNDEFINED"]
    assert issue.suggested_action == "model the element with its thickness and material"
    (graded,) = result.targets
    assert graded.status.value == "UNKNOWN" and "thin lens, a Faraday rotator or a diffraction grating" in graded.message
    for text in (pair.phase_statement, issue.suggested_action, graded.message):
        assert "coating" not in text, text
    # A headless consumer receives structured phase reasons directly, not translated editor strings.
    assert pair.phase_difference_rad is None and "PHASE_UNDEFINED" in pair.issue_codes
    assert issue.suggested_action == "model the element with its thickness and material"
    from os_optics.contracts.results import ArrivalResult

    description = ArrivalResult.model_fields["group_delay_unknown_at"].description
    assert "their phase difference has no value (PHASE_UNDEFINED). Two" not in description
    assert "with an ESTIMATE entry (UNSTATED_PHASE_AS_INTERFACE)" in description


def test_beams_crossing_the_plane_in_opposite_directions_add_and_have_no_range():
    """Their flux through the plane has no cross term, whatever their phase: the value stands, the phase is stated on the
    default phases with its entry."""
    a = _record("a", 3e-3, (0.0, 0.0), 1.0, 0.0, ("A/S1:reflect:+@45.000000<=100.0fs",), 100.0, 100.0)
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 0.0, ("B/S1:reflect:+@45.000000<=100.0fs",), 100.0, 100.0)
    b.direction = -b.direction
    pair, _ = analyse_pair(a, b, NumericalConfig())
    assert pair.combined_power_collinear_W == pytest.approx(4e-3, rel=1e-15)
    assert pair.combined_power_collinear_min_W is None and pair.combined_power_collinear_max_W is None
    assert pair.phase_statement is None and "combined_power_collinear_W" not in (pair.value_basis or {})
    assert pair.value_basis["phase_difference_rad"].kind.value == "ESTIMATE"


# -------------------------------------------------------------------------------- the pair's range and the plane's range


def _turned_about_z(quaternion, angle_rad: float) -> list[float]:
    c, s = math.cos(angle_rad / 2.0), math.sin(angle_rad / 2.0)
    w, x, y, z = quaternion
    return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]


def test_the_pair_range_and_the_plane_range_agree_for_collinear_beams_and_a_tilt_narrows_the_plane_range(catalog, open_scan, want):
    """The pair's range uses the envelopes' overlap ``eta_sp``; the plane's the whole-plane integral with the fields' phases. For
    collinear beams they are one number. Two beams at a small angle within the pair's collinear limit (fringe period above 10
    beam diameters) average across their fringes on the plane: the plane's cross term is the pair's times
    ``exp(-(dk w)^2 / 8)`` of the angle ``dk / k0`` between them, ``w`` the beam's radius there."""
    _, superposition, pair = _on_plane(open_scan[0])
    assert superposition.total_power_min_W == pytest.approx(pair.combined_power_collinear_min_W, abs=1e-15)
    assert superposition.total_power_max_W == pytest.approx(pair.combined_power_collinear_max_W, rel=1e-12)

    tilt = 3e-5  # rad, of the mirror: the beams meet at twice that
    document = _ideal_michelson(catalog)
    pose = next(c for c in document["components"] if c["instance_id"] == "M1")["nominal_pose"]
    pose["quaternion_wxyz"] = _turned_about_z(pose["quaternion_wxyz"], tilt)
    _, superposition, pair = _on_plane(run_optical_project(document, catalog=catalog))
    assert pair.fringe_period_nm is not None and pair.combined_power_collinear_max_W is not None  # within the collinear limit
    radius = 0.5 * math.sqrt(1.0 + (350.0 * WAVELENGTH_NM * 1e-6 / (math.pi * 0.25)) ** 2)
    dk = 2.0 * math.pi / (WAVELENGTH_NM * 1e-6) * 2.0 * tilt
    factor = math.exp(-((dk * radius) ** 2) / 8.0)
    assert factor < 0.995
    half_pair = (pair.combined_power_collinear_max_W - pair.combined_power_collinear_min_W) / 2.0
    half_plane = (superposition.total_power_max_W - superposition.total_power_min_W) / 2.0
    assert half_plane == pytest.approx(half_pair * factor, rel=2e-3)
    assert half_plane < half_pair  # the pair's range holds the plane's


# --------------------------------------------------------------------------------- a non-collinear crossing: the shipped example


def test_the_shipped_example_states_the_open_phase_without_a_collinear_range():
    catalog = bounded_catalog_document()
    # Declare test-only copies, as public catalog materials cannot state bulk
    # loss. The frozen development catalog itself remains unchanged.
    example = example_project_of_kind("two_beam_interference", catalog).payload()
    run = run_optical_project(_ideal_declared(declared_project(example, catalog)), catalog=catalog)
    (pair,) = run.result.interference
    # (crossing beams: the phase is stated on the default phases with its entry; no collinear power, so no range)
    assert pair.fringe_period_nm is not None and pair.phase_difference_rad is not None and pair.combined_power_collinear_W is None
    assert pair.combined_power_collinear_min_W is None and pair.combined_power_collinear_max_W is None
    assert pair.phase_statement is None and pair.value_basis["phase_difference_rad"].kind.value == "ESTIMATE"
    dumped = pair.model_dump(mode="json")
    assert "combined_power_collinear_min_W" not in dumped and "combined_power_collinear_max_W" not in dumped


def test_the_shipped_catalog_example_has_the_same_form(catalog):
    document = example_project_of_kind("two_beam_interference", catalog).payload()
    public = run_optical_project(document, catalog=catalog)
    assert public.result.interference == ()  # the actual catalog lacks substrate bulk-loss evidence
    (public_plane,) = public.result.observations
    assert len(public_plane.arrivals) == 2
    (through_glass,) = [a for a in public_plane.arrivals if a.beam.power_W is None]
    (reflected_only,) = [a for a in public_plane.arrivals if a.beam.power_W is not None]
    assert through_glass.power_upper_bound_W is not None and through_glass.power_upper_bound_W > 0
    assert reflected_only.power_upper_bound_W is None and reflected_only.beam.power_W > 0
    # A test-only declared copy restores numeric fields for the independent
    # coating-phase exercise, without changing any published material.
    run = run_optical_project(_ideal_declared(declared_project(document, catalog)), catalog=catalog)
    (ideal_plane,) = run.result.observations
    for actual in public_plane.arrivals:
        ideal = next(a for a in ideal_plane.arrivals if a.lineage == actual.lineage)
        assert actual.beam.optical_path_length_mm == pytest.approx(ideal.beam.optical_path_length_mm, abs=1e-9)
        assert actual.beam.direction == pytest.approx(ideal.beam.direction, abs=1e-12)
    (pair,) = run.result.interference
    assert "PHASE_UNDEFINED" not in pair.issue_codes and pair.phase_difference_rad is not None and pair.phase_statement is None
    assert pair.coherence_factor is not None and pair.fringe_visibility is not None
    # the two beams cross at 90 deg: their fringes average out over the plane, so its coherent power has a value whatever the phase
    (observation,) = run.result.observations
    (superposition,) = observation.superpositions
    assert superposition.total_power_W == pytest.approx(superposition.incoherent_power_W, rel=1e-12)
    assert superposition.status is Status.PASS and superposition.phase_statement is None


# ---------------------------------------------------------------------------------------------- query outputs


def test_the_query_outputs_carry_the_range_and_the_statement(open_scan, stated_scan):
    results = load_run(open_scan[0])
    (row,) = results.superpositions("out")
    (_, superposition, pair) = _on_plane(open_scan[0])
    assert row["total_power_W"] == superposition.total_power_W is not None
    assert row["value_basis"]["total_power_W"]["kind"] == "ESTIMATE"
    assert row["total_power_min_W"] == superposition.total_power_min_W and row["total_power_max_W"] == superposition.total_power_max_W
    assert row["phase_statement"] is None
    assert {"total_power_min_W", "total_power_max_W", "phase_statement"} <= {c["name"] for c in columns("superpositions")}
    (monitor_pair,) = results.monitor("out")["interference"]
    assert monitor_pair["combined_power_collinear_min_W"] == pair.combined_power_collinear_min_W
    assert monitor_pair["combined_power_collinear_max_W"] == pair.combined_power_collinear_max_W
    assert monitor_pair["combined_power_collinear_W"] == pair.combined_power_collinear_W
    assert monitor_pair["value_basis"]["combined_power_collinear_W"]["kind"] == "ESTIMATE"

    (known,) = load_run(stated_scan[0]).superpositions("out")
    assert known["total_power_W"] is not None and known["phase_statement"] is None


def test_the_power_basis_sits_beside_the_plane_power_in_every_row(open_scan):
    """The label is produced with the observation result; each row reads it (``ObservationResult.power_basis``)."""
    run = open_scan[0]
    observation = next(o for o in run.result.observations if o.observation_id == "out")
    labelled = observation.model_copy(update={"power_basis": "INCOHERENT_SUM"})
    assert monitor_row(labelled)["power_basis"] == "INCOHERENT_SUM"
    assert "power_basis" in {c["name"] for c in columns("monitors")}
    assert monitor_row(observation.model_copy(update={"power_basis": None}))["power_basis"] is None
    result = run.result.model_copy(update={"observations": (labelled,)})
    numerical = run.request.configuration.numerical
    (row,) = json_api._observations(result, numerical.hash_float_sig_digits, numerical.hash_float_abs_floor)
    assert row["power_basis"] == "INCOHERENT_SUM"
    bare = result.model_copy(update={"observations": (observation.model_copy(update={"power_basis": None}),)})
    (row,) = json_api._observations(bare, numerical.hash_float_sig_digits, numerical.hash_float_abs_floor)
    assert "power_basis" not in row


def test_the_run_summary_states_the_coherent_power_beside_the_plane_power(open_scan, stated_scan, want):
    (row,) = json_api.run_summary(open_scan[0])["observations"]
    (superposition,) = row["superpositions"]
    assert superposition["total_power_min_W"] < row["total_power_W"] < superposition["total_power_max_W"]
    assert superposition["total_power_min_W"] <= superposition["total_power_W"] <= superposition["total_power_max_W"]
    assert "phase_statement" not in superposition
    assert row["total_power_W"] == pytest.approx(want.sum_W, rel=1e-9)
    (row,) = json_api.run_summary(stated_scan[2])["observations"]  # dark
    (superposition,) = row["superpositions"]
    assert superposition["total_power_W"] < 2e-3 * row["total_power_W"] and "phase_statement" not in superposition


def test_a_plane_with_one_beam_has_no_superposition_row_and_no_power_basis(catalog):
    (row,) = json_api.run_summary(run_optical_project(example_project_of_kind("focus", catalog).payload(), catalog=catalog))["observations"]
    assert len(row["arrivals"]) == 1 and "superpositions" not in row and "power_basis" not in row


def test_run_files_round_trip_the_new_fields(open_scan):
    """Absent where they do not apply, present and read back where they do."""
    run = open_scan[0]
    results = load_run(run_artifact_files(run))
    (row,) = results.superpositions("out")
    assert row["total_power_min_W"] is not None and row["value_basis"]["total_power_W"]["kind"] == "ESTIMATE"
