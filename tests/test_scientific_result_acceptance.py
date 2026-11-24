"""The "Acceptance scenarios" table of the scientific-result contract as a machine-checked matrix: one test per row the Engine can express, each asserting the
row's required outcome on the four independent answers of the scientific-result contract.

* execution: ``outcome.execution`` (did the calculation complete, and if not why);
* quantity: a plane's ``quantity_state`` and values, or a surface's exposure and rating status;
* target: the target's ``assessment`` and ``outcome.requirements``;
* aggregate: ``ValidationResult.status``, a summary of diagnostic severity that decides nothing (``errors/status.py``).

The expected physics comes from closed forms written here (a Gaussian beam's power through a round stop, its far-field angle), never
from the Engine under test.

Row "Non-assembly commercial request" is a check of the OpenSci service, not of the Engine.
Row "Internal-test assembly" is covered end to end by ``tests/test_assembly_validation.py`` (``os_optics.assembly``); here only its
presence is checked.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from os_optics import builders as B
from os_optics.cli import main
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local import example_project, json_api, run_freshness, run_optical_project, verify_run_artifacts

from helpers import CFG, one_arrival, run
from local_helpers import plane, project, source
from local_helpers import run as run_local
from test_closed_returns import faint_ring
from test_local_run_artifacts import _refresh_manifest
from test_pulse_count_input import S_ON_1, _count, _laser, _mirror_run, _pulsed_rating
from test_tail_dark_planes_and_surface_coincidence import _ghost_scene

P = 1e-3
LAM = 780.0
X = (1.0, 0.0, 0.0)
#: What a threshold may move on a run's result: the rest is physics.
THRESHOLD_FIELDS = {"status", "outcome", "summary", "targets", "manifest"}


def _laser_beam(wavelength_nm: float = LAM, waist_mm: float = 0.5) -> object:
    return B.laser("L", wavelength_nm=wavelength_nm, power_W=P, waist_radius_mm=waist_mm, position=(0.0, 0.0, 0.0), direction=X)


def _plane_t():
    return B.observation_plane("T", position=(100.0, 0.0, 0.0), normal=X, radius_mm=12.7)


def _off_axis_plane():
    """A plane far off the beam: nothing the laser sends can reach it."""
    return B.observation_plane("D", position=(100.0, 60.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=5.0)


def _target(target_id: str, metric: str = "power_W", observation_id: str = "T", unit: str = "W", **limits) -> Target:
    return Target(target_id=target_id, metric=metric, observation_id=observation_id, unit=unit, **limits)


def _answers(result, *, execution: str, aggregate: str, met=(), unmet=(), indeterminate=(), not_applicable=(), declared: bool = True) -> None:
    """The execution, the target partition and the aggregate, each on its own."""
    assert result.outcome.execution.outcome.value == execution
    assert result.status.value == aggregate
    requirements = result.outcome.requirements
    assert requirements.declared is declared
    assert (set(requirements.met), set(requirements.unmet), set(requirements.indeterminate), set(requirements.not_applicable)) == (
        set(met), set(unmet), set(indeterminate), set(not_applicable))


def _plane(result, observation_id: str = "T"):
    return next(o for o in result.observations if o.observation_id == observation_id)


def _assessment(result, target_id: str) -> str:
    return next(t for t in result.targets if t.target_id == target_id).assessment.value


def _physical(result) -> dict:
    """Everything of a run but what a threshold may move. ``affected_targets`` on an issue is left out: that context names the
    targets an issue leaves undecided (``validation/affected.py``), so it depends on them by design, and it changes no physics."""
    dump = result.model_dump(mode="json")
    physical = {key: value for key, value in dump.items() if key not in THRESHOLD_FIELDS}
    for issue in physical.get("issues", ()):
        (issue.get("numerical_context") or {}).pop("affected_targets", None)
        issue.pop("affected_targets", None)  # (the same list on the issue itself: the targets it leaves undecided)
    return physical


# ---------------------------------------------------------------------------------------------------- no targets
def test_no_targets_gives_useful_results_and_no_implicit_certification():
    result = run(ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(),)))
    _answers(result, execution="COMPLETED", aggregate="PASS", declared=False)
    assert result.outcome.execution.complete and result.targets == ()
    obs = _plane(result)
    assert obs.quantity_state.value == "KNOWN" and obs.total_power_W == P
    # a beam of 0.5 mm waist after 100 mm (its own closed form): the result is readable with nothing asked of it
    rayleigh = math.pi * 0.25 / (LAM * 1e-6)
    assert one_arrival(result, "T").beam.gaussian.beam_radius_x_mm == pytest.approx(0.5 * math.sqrt(1.0 + (100.0 / rayleigh) ** 2), rel=1e-9)


# --------------------------------------------------------------------------------------------- threshold changed
@pytest.mark.parametrize(("scene", "limits", "assessments"), [
    # (a lit plane: the threshold decides; a plane only followed light reaches: it decides outside [0, that light's bound])
    (lambda t: ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(),), targets=t), ({"minimum": 5e-4}, {"minimum": 2e-3}), ("MET", "UNMET")),
    (_ghost_scene, ({"maximum": 1.0}, {"maximum": 1e-12}), ("MET", "INDETERMINATE")),
])
def test_a_changed_threshold_moves_the_assessment_and_the_identity_of_the_requirement_only(scene, limits, assessments):
    a = run(scene((_target("p", **limits[0]),)))
    b = run(scene((_target("p", **limits[1]),)))
    assert (_assessment(a, "p"), _assessment(b, "p")) == assessments
    # the physical records are identical, field by field
    assert _physical(a) == _physical(b)
    assert a.outcome.execution == b.outcome.execution
    assert _plane(a).quantity_state == _plane(b).quantity_state and _plane(a).message == _plane(b).message
    # the identities differ where a threshold is part of what they name, and only there
    ma, mb = a.manifest.model_dump(mode="json"), b.manifest.model_dump(mode="json")
    assert {key for key in ma if ma[key] != mb[key]} == {"input_hash", "project_hash", "result_hash"}


def test_a_threshold_changed_after_the_run_makes_it_historical_for_that_reason(tmp_path):
    document = example_project().payload()
    directory = tmp_path / "run"
    run_optical_project(document, output_dir=directory)
    edited = copy.deepcopy(document)
    edited["targets"][0]["tolerance_abs"] = 0.5
    fresh = run_freshness(edited, directory)
    assert (fresh["freshness"], fresh["reasons"]) == ("HISTORICAL", ["TARGETS_CHANGED"])


# ------------------------------------------------------------------------------------------------- a missed target
def test_a_valid_result_that_misses_a_target_is_an_unmet_target_not_a_numerical_failure():
    result = run(ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(),), targets=(_target("p", minimum=2e-3),)))
    _answers(result, execution="COMPLETED", aggregate="FAIL", unmet=("p",))
    assert result.outcome.execution.termination.value == "COMPLETED"
    obs = _plane(result)
    assert obs.quantity_state.value == "KNOWN" and obs.total_power_W == P
    assert result.targets[0].value == P and "TARGET_NOT_MET" in result.targets[0].issue_codes
    assert result.issues == ()


# ------------------------------------------------------------------------------------------------- clipped capture
def _stop_power(radius_mm: float, z_mm: float = 50.0, waist_mm: float = 0.5) -> float:
    """The power of a centred Gaussian beam through a round stop: P (1 - exp(-2 r^2 / w^2)), w its 1/e^2 radius there."""
    rayleigh = math.pi * waist_mm * waist_mm / (LAM * 1e-6)
    w2 = waist_mm * waist_mm * (1.0 + (z_mm / rayleigh) ** 2)
    return P * (1.0 - math.exp(-2.0 * radius_mm * radius_mm / w2))


def _clipped(radius_mm: float):
    stop = B.aperture_stop("A", position=(50.0, 0.0, 0.0), axis=X, radius_mm=radius_mm)
    targets = (_target("captured", minimum=1e-4), _target("radius", metric="beam_radius_x_mm", unit="mm", minimum=0.0))
    return run(ProjectSnapshot(components=(_laser_beam(), stop), observations=(_plane_t(),), targets=targets))


def test_a_mildly_clipped_beam_keeps_its_captured_power_and_its_gaussian_quantities_with_a_warning():
    result = _clipped(0.8)
    _answers(result, execution="COMPLETED", aggregate="WARNING", met=("captured", "radius"))
    obs = _plane(result)
    assert obs.quantity_state.value == "KNOWN" and obs.total_power_W == pytest.approx(_stop_power(0.8), rel=1e-9)
    assert "GAUSSIAN_TRUNCATION" in {i.code.value for i in result.issues}


def test_a_severely_clipped_beam_keeps_its_captured_power_and_withholds_its_gaussian_quantities():
    result = _clipped(0.5)
    # a met power target says nothing about the downstream Gaussian quantity: each has its own answer
    _answers(result, execution="COMPLETED", aggregate="OUT_OF_MODEL_SCOPE", met=("captured",), indeterminate=("radius",))
    obs = _plane(result)
    assert obs.quantity_state.value == "KNOWN" and obs.total_power_W == pytest.approx(_stop_power(0.5), rel=1e-9)
    radius = next(t for t in result.targets if t.target_id == "radius")
    assert radius.value is None and "GAUSSIAN_SEVERE_TRUNCATION" in radius.issue_codes
    assert not one_arrival(result, "T").beam.gaussian.valid
    # (no number: the target and the state name the failed check -- the truncated share against its setting)
    assert radius.value_basis is None and one_arrival(result, "T").beam.gaussian.beam_radius_x_mm is None
    (check,) = radius.validity
    assert (check.parameter, check.setting, check.within, check.code) == (
        "truncated_power_fraction", "gaussian_clip_scope_fraction", False, "GAUSSIAN_SEVERE_TRUNCATION")
    assert check.observed == pytest.approx(1.0 - _stop_power(0.5) / P, rel=1e-6)
    assert one_arrival(result, "T").beam.gaussian.validity == (check,)


# ---------------------------------------------------------------------------------------------------------- high NA
def test_high_na_keeps_the_geometric_results_and_withholds_the_gaussian_metrics():
    waist_mm = 3e-4
    assert 405e-6 / (math.pi * waist_mm) > 0.30  # the far-field half angle above the paraxial limit
    laser = _laser_beam(405.0, waist_mm)
    targets = (_target("p", minimum=1e-4), _target("aoi", metric="aoi_deg", unit="deg", maximum=1.0),
               _target("radius", metric="beam_radius_x_mm", unit="mm", minimum=0.0))
    result = run(ProjectSnapshot(components=(laser,), observations=(_plane_t(),), targets=targets))
    _answers(result, execution="COMPLETED", aggregate="OUT_OF_MODEL_SCOPE", met=("p", "aoi"), indeterminate=("radius",))
    assert _plane(result).total_power_W == P and _plane(result).quantity_state.value == "KNOWN"
    arrival = one_arrival(result, "T")
    assert arrival.aoi_deg == 0.0 and arrival.beam.gaussian.reason == "GAUSSIAN_HIGH_NA"
    radius = next(t for t in result.targets if t.target_id == "radius")
    assert radius.value is None and radius.status.value == "OUT_OF_MODEL_SCOPE" and radius.value_basis is None
    (check,) = radius.validity
    assert (check.parameter, check.setting, check.within, check.code) == (
        "far_field_half_angle_rad", "gaussian_scope_divergence_rad", False, "GAUSSIAN_HIGH_NA")
    assert check.observed == pytest.approx(405e-6 / (math.pi * waist_mm), rel=1e-9)


# -------------------------------------------------------------------------------------------------- budget exhausted
def test_budget_exhaustion_names_its_cause_and_scope_with_no_blanket_known_zero():
    targets = (_target("lit", maximum=1.0), _target("dark", observation_id="D", maximum=1.0))
    project_ = ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(), _off_axis_plane()), targets=targets)
    result = run(project_, numerical=CFG.model_copy(update={"max_trace_steps": 1}))
    _answers(result, execution="PARTIAL", aggregate="UNKNOWN", indeterminate=("lit", "dark"))
    execution = result.outcome.execution
    assert execution.termination.value == "BUDGET_EXHAUSTED" and execution.budgets.exhausted == ("max_trace_steps",)
    assert execution.budgets.limits["max_trace_steps"] == 1 and execution.budgets.untraced_power_W == pytest.approx(P)
    dark = _plane(result, "D")
    assert dark.quantity_state.value == "UNKNOWN" and dark.total_power_W != 0.0 and not dark.power_complete
    # (the lit plane: its traced value is the lower end of its power; what the budget left out may cross it again, so the
    # upper end is open and the entry names the budget and the path it stopped)
    lit = _plane(result)
    assert lit.quantity_state.value == "BOUNDED" and lit.total_power_W == pytest.approx(P) and lit.power_upper_bound_W is None
    entry = lit.value_basis["total_power_W"]
    (omitted,) = entry.omitted
    assert entry.kind.value == "LOWER_BOUND" and (omitted.part.value, omitted.code, omitted.upper) == ("BUDGET", "MAX_PATHS", None)
    assert omitted.at and all(place.startswith("path:") for place in omitted.at)
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["dark"].value is None
    assert by_id["lit"].value == lit.total_power_W and by_id["lit"].value_basis["value"].kind.value == "LOWER_BOUND"
    assert "MAX_PATHS" in {i.code.value for i in result.issues}


# --------------------------------------------------------------------------------------------- a closed-path bound
@pytest.mark.parametrize("corners", [3, 4])
def test_a_closed_return_has_no_unsupported_bound_and_no_within_rating(corners: int):
    result = run(faint_ring(corners))
    _answers(result, execution="COMPLETED", aggregate="UNKNOWN", indeterminate=("b",))
    surface = next(s for s in result.exposures.surfaces if s.surface_id == "B/face")
    assert surface.ratings[0].exposure_upper_bound is None and surface.rating_status != "WITHIN_RATING"
    # (the missing bound is a notice of the run, not a finding per surface: it does not count toward the run's status)
    (named,) = [i for i in result.issues if i.numerical_context.get("cause") == "CLOSED_RETURN_NOT_BOUNDED"]
    assert (named.code.value, named.status.value) == ("EXPOSURE_BOUND_WITHHELD", "NOT_APPLICABLE")
    assert "B/face" in named.numerical_context["surface_ids"]
    assert any("B/face" in cycle for cycle in named.numerical_context["closed_return_surfaces"])
    assert not [i for i in result.issues if i.code.value == "OUT_OF_MODEL_SCOPE"]
    # the rest of the run stays: the traced path and the power at the surface it meets
    assert result.paths and result.power_budget is not None
    assert next(s for s in result.exposures.surfaces if s.surface_id == "A/face").incident_power_W == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------- a crossing-plane defect
def _crossing_fixture(order: tuple[str, str]) -> dict:
    """The self-contained intersecting-plane fixture: a 1 mW beam through two transparent planes at one point, normals +X and (X + Y) / sqrt(2)."""
    poses = {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.2705980500730985, 0.2705980500730985, 0.6532814824381882, 0.6532814824381883]}
    return {
        "schema": "opensci.project", "project_schema_version": "2.0",
        "components": [{
            "instance_id": "S", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Synthetic probe laser"},
            "nominal_pose": {"position_mm": [-10, 0, 0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 780, "power_w": 0.001, "waist_radius_mm": 0.5, "waist_position_mm": 0,
                             "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}},
        }],
        "observations": [{"observation_id": oid, "pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": poses[oid]},
                          "aperture": {"shape": "circle", "radius_mm": 50}} for oid in order],
        "targets": [{"target_id": "powerA", "observation_id": "A", "metric": "power_W", "unit": "W", "maximum": 1e-9}],
    }


@pytest.mark.parametrize("order", [("B", "A"), ("A", "B")])
def test_a_plane_crossing_another_at_one_point_reads_the_beam_and_misses_the_upper_bound(order: tuple[str, str]):
    result = run_local(_crossing_fixture(order)).result
    _answers(result, execution="COMPLETED", aggregate="FAIL", unmet=("powerA",))
    for oid in "AB":
        obs = _plane(result, oid)
        assert obs.quantity_state.value == "KNOWN" and obs.total_power_W == pytest.approx(P)
        (arrival,) = obs.arrivals
        assert arrival.pupil_rays_arrived == arrival.pupil_ray_count == 96
    assert result.targets[0].value == pytest.approx(P)


# -------------------------------------------------------------------------------- a known dark plane, an empty selection
def test_a_known_dark_plane_is_a_measured_zero_and_an_empty_selection_is_no_measurement():
    targets = (_target("max", observation_id="D", maximum=1e-6), _target("min", observation_id="D", minimum=1e-6),
               _target("radius", observation_id="D", metric="beam_radius_x_mm", unit="mm", minimum=0.0),
               _target("wavelength", wavelength_nm=532.0, maximum=1.0), _target("path", maximum=1.0, parameters={"path_id": "nope"}))
    result = run(ProjectSnapshot(components=(_laser_beam(),), observations=(_plane_t(), _off_axis_plane()), targets=targets))
    _answers(result, execution="COMPLETED", aggregate="FAIL", met=("max",), unmet=("min", "radius"), indeterminate=("wavelength", "path"))
    by_id = {t.target_id: t for t in result.targets}
    dark = _plane(result, "D")
    assert dark.quantity_state.value == "KNOWN_ZERO" and dark.total_power_W == 0.0 and dark.power_complete
    assert by_id["max"].value == 0.0 and "NO_TRACED_ARRIVAL" in by_id["max"].issue_codes
    # no beam, so no radius: not a zero radius
    assert by_id["radius"].value is None and "OBSERVATION_NOT_REACHED" in by_id["radius"].issue_codes
    for selection in ("wavelength", "path"):
        assert by_id[selection].value is None and "TARGET_SELECTION_EMPTY" in by_id[selection].issue_codes


def test_a_met_target_on_a_dark_plane_stands_beside_a_passing_aggregate():
    result = run(ProjectSnapshot(components=(_laser_beam(),), observations=(_off_axis_plane(),), targets=(_target("max", observation_id="D", maximum=1e-6),)))
    # a measured zero carries no propagating status: the plane reads PASS / KNOWN_ZERO, its issue is a notice, and the
    # aggregate holds no term for it
    _answers(result, execution="COMPLETED", aggregate="PASS", met=("max",))
    assert _plane(result, "D").status.value == "PASS" and _plane(result, "D").quantity_state.value == "KNOWN_ZERO"
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert issue.status.value == "NOT_APPLICABLE" and issue.location == "observation:D"


def test_a_plane_only_followed_light_reaches_is_no_zero_and_its_bound_grades_its_power_targets():
    """The plane states no power and stays UNKNOWN; its power targets are graded on [0, bound of the followed light]
    (targets only): a maximum above the bound is met, a minimum above it
    is not, a limit inside the interval is not decided."""
    result = run(_ghost_scene((_target("max", maximum=1e-3), _target("min", minimum=1.0), _target("inside", maximum=1e-9))))
    _answers(result, execution="COMPLETED", aggregate="FAIL", met=("max",), unmet=("min",), indeterminate=("inside",))
    obs = _plane(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.power_lower_bound_W == 0.0 and obs.followed_light and obs.total_power_W is None and obs.untraced_power_W > 0.0
    assert obs.power_upper_bound_W is not None and all(t.value is None for t in result.targets)


# ---------------------------------------------------------------------------------- missing pulse-count / rating evidence
def test_a_missing_pulse_count_leaves_the_comparison_undecided_and_the_exposure_available():
    rating = _pulsed_rating(1e30, **S_ON_1)
    face, target = _mirror_run((_laser(total=None),), (rating,))
    (comparison,) = face.ratings
    assert _count(comparison).state == "NOT_VERIFIABLE" and _count(comparison).reason == "DESIGN_INPUT_MISSING"
    assert face.rating_status == "CONDITIONS_NOT_VERIFIED" and comparison.status == "CONDITIONS_NOT_VERIFIED"
    # (reported, not graded: the ratio to the rated value with the count as a check that could not be made)
    assert target.assessment.value == "INDETERMINATE" and target.status.value == "UNKNOWN"
    assert target.value == comparison.utilization == comparison.exposure_value / comparison.rating_value
    entry = target.value_basis["value"]
    assert entry.kind.value == "ESTIMATE" and [(v.parameter, v.within) for v in entry.validity] == [("pulse_count", None)]
    assert "POWER_RATING_CONDITIONS_NOT_VERIFIED" in target.issue_codes
    # the peak fluence of a 1 W, 10 Hz beam on the mirror, from its own hit: 2 E / (pi w_x w_y), E = P / f
    (hit,) = face.hits
    fluence = 2.0 * (hit.incident_power_W / 10.0) / (math.pi * hit.beam_radius_x_mm * hit.beam_radius_y_mm) * 100.0
    assert comparison.exposure_value == pytest.approx(fluence, rel=1e-12)
    assert hit.peak_fluence_J_per_cm2 == pytest.approx(fluence, rel=1e-12)
    # stating the count decides the same comparison: nothing else was missing
    stated, decided = _mirror_run((_laser(total=100),), (rating,))
    assert stated.ratings[0].status == "WITHIN_RATING" and decided.assessment.value == "MET"
    assert stated.ratings[0].exposure_value == comparison.exposure_value


# ----------------------------------------------------------------------------------------- a tampered or a stale result
def _unmet_run(directory: Path) -> dict:
    document = project([source()], [plane("screen", (100.0, 0.0, 0.0), radius_mm=40.0)],
                       [{"target_id": "power", "metric": "power_W", "observation_id": "screen", "minimum": 1.0, "unit": "W"}])
    produced = run_optical_project(document, output_dir=directory)
    assert produced.result.outcome.requirements.unmet == ("power",)
    return document


def test_a_status_edited_after_the_run_is_not_upgraded_by_recomputing_the_file_digests(tmp_path):
    directory = tmp_path / "run"
    document = _unmet_run(directory)
    assert verify_run_artifacts(directory) == ()
    assert run_freshness(document, directory)["freshness"] == "CURRENT"
    path = directory / "simulation_result.json"
    result = json.loads(path.read_bytes())
    result["status"] = "PASS"
    result["targets"][0].update(status="PASS", assessment="MET")
    result["outcome"]["requirements"].update(met=["power"], unmet=[])
    path.write_text(json.dumps(result), encoding="utf-8")
    _refresh_manifest(directory)  # (a forger who also fixes the file digests)
    # every path that decides anything reports the edited file: the verification, the freshness asked to verify, and the
    # submission -- where the run's bytes leave the device
    problems = verify_run_artifacts(directory)
    assert any(p.startswith("simulation_result.json:") for p in problems)
    asked = run_freshness(document, directory, verify=True)
    assert asked["integrity"]["checked"] is True and asked["integrity"]["consistent"] is False
    assert asked["integrity"]["problems"] == list(problems)
    from os_optics.local.submission import RunDirectoryInconsistentError, submission_document

    with pytest.raises(RunDirectoryInconsistentError):
        submission_document(directory)
    with pytest.raises(RunDirectoryInconsistentError):
        submission_document({p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()})
    # freshness is the other question -- is this the run of this Project? -- and never changes with verify: the run's
    # outcome it reports is the record's, where the target is still unmet
    fresh = run_freshness(document, directory)
    assert (fresh["freshness"], fresh["reasons"]) == (asked["freshness"], asked["reasons"]) == ("CURRENT", [])
    assert fresh["integrity"] == {"checked": False, "consistent": None, "problems": []}
    assert fresh["run"]["outcome"]["requirements"]["unmet"] == ["power"] == asked["run"]["outcome"]["requirements"]["unmet"]


def test_a_run_of_another_design_is_stale_whatever_it_reports(tmp_path):
    directory = tmp_path / "run"
    document = _unmet_run(directory)
    moved = copy.deepcopy(document)
    moved["observations"][0]["pose"]["position_mm"][0] = 120.0
    fresh = run_freshness(moved, directory, verify=True)
    assert (fresh["freshness"], fresh["reasons"]) == ("HISTORICAL", ["PHYSICS_CHANGED"])
    # freshness and integrity are facts about the run's identity: no field calls the local run independently verified
    assert fresh["integrity"]["consistent"] is True and "OPENSCI_VERIFIED" not in json.dumps(fresh)
    assert run_freshness(moved, directory)["integrity"]["checked"] is False


# ------------------------------------------------------------------------------------------------ the rows outside it
def test_the_internal_test_assembly_path_is_present():
    """The chain itself is exercised by ``tests/test_assembly_validation.py``: no performance promise, no shutdown."""
    from os_optics.assembly import load_assembly_preview, validate_assembly_layout

    assert callable(load_assembly_preview) and callable(validate_assembly_layout)
    assert (Path(__file__).parent / "test_assembly_validation.py").is_file()


# ---------------------------------------------------------------------------------------------- the recorded findings
def test_r2_f3_an_uncoated_lens_pair_states_why_its_power_has_an_open_upper_end():
    """A lit transparent plane behind an uncoated pair reads
    BOUNDED from its traced value with an open upper end, and its entry says which light keeps its power from having an upper
    bound: the followed ghosts with their power, and the light left out as negligible, which nothing bounds there."""
    lenses = [B.singlet_lens(f"L{i}", r1_mm=60.0, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7", position=(20.0 + 25.0 * i, 0.0, 0.0))
              for i in range(2)]
    result = run(ProjectSnapshot(components=(_laser_beam(), *lenses), observations=(_plane_t(),), targets=(_target("p", minimum=1e-6),)))
    obs = _plane(result)
    assert obs.quantity_state.value == "BOUNDED" and obs.total_power_W > 0.0 and obs.power_upper_bound_W is None
    assert "power_upper_bound_W" in obs.model_fields_set and not obs.power_complete
    assert "no upper bound" in obs.message and "the traced beams bring" in obs.message
    entry = obs.value_basis["total_power_W"]
    followed, faint = entry.omitted
    assert entry.kind.value == "LOWER_BOUND"
    assert (followed.part.value, followed.code) == ("NOT_FOLLOWED", "UNTRACED_LIGHT_BOUNDED")
    assert followed.upper == pytest.approx(obs.untraced_power_W, rel=1e-12)
    assert set(followed.at) <= {f"surface:L{i}/{face}" for i in range(2) for face in ("front", "back")} and followed.at
    # (nothing bounds that light's returns between the lens's faces: its possible return, not an accounting truncation)
    assert (faint.part.value, faint.upper, faint.setting) == ("POSSIBLE_RETURN", None, "negligible_power_fraction")
    # the minimum reads the traced light: met, with the bound's absence stated on the plane
    assert result.targets[0].assessment.value == "MET"


def test_r2_f2_a_target_that_does_not_apply_is_not_reported_met_by_the_requirements_contract(tmp_path, capsys):
    """The requirements contract (the default) exits 2 on a NOT_APPLICABLE target; the aggregate reads it as nothing."""
    document = project([source("A", wavelength_nm=532.0), source("B", position=(-200.0, 0.3, 0.0), wavelength_nm=1064.0)],
                       [plane("det", (0.0, 0.0, 0.0), radius_mm=20.0)],
                       [{"target_id": "overlap", "metric": "spatial_overlap_ratio", "observation_id": "det", "source_id": "A", "source_id_b": "B", "minimum": 0.0}])
    produced = run_local(document)
    assert produced.result.outcome.requirements.not_applicable == ("overlap",)
    assert produced.result.targets[0].issue_codes == ("WAVELENGTH_MISMATCH",)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--status-contract", "requirements"]) == 2
    assert main(["project", "run", str(path), "--status-contract", "execution"]) == 0
    assert main(["project", "run", str(path)]) == 2
    assert main(["project", "run", str(path), "--status-contract", "aggregate"]) == 0
    capsys.readouterr()


def test_the_run_summary_carries_the_marker_and_the_power_basis():
    """The structured run summary carries surface markers and the Engine's power basis for headless readers."""
    a = B.laser("A", wavelength_nm=LAM, waist_radius_mm=0.5, power_W=P, position=(0.0, 0.0, 0.0), direction=X)
    fold = B.mirror("M", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    marked = run(ProjectSnapshot(components=(a, fold), observations=(_plane_t(),)))
    observations = json_api._observations(marked, 9, 1e-30)
    roles = [row["surface_coincidence"]["role"] for row in observations[0]["arrivals"]]
    assert roles == ["ARRIVING", "RETURNING"]
    assert observations[0]["arrivals"][0]["surface_coincidence"]["surface_id"] == "M/face"
