"""A Gaussian state beyond an accuracy limit publishes no number: it states the failed check.

The first-order (ABCD) model has accuracy limits held by caller settings: the far-field half angle
(``gaussian_scope_divergence_rad``), the beam radius against a surface's radius of curvature
(``gaussian_max_w_over_radius``), the angle of incidence on a focusing surface (``gaussian_max_aoi_deg``) and the share
an aperture truncates (``gaussian_clip_scope_fraction``). Beyond one of them the state is not valid for the rest of the
path, no radius, waist, q or angle is published, and the state -- and a target on a Gaussian metric -- names the failed
check in ``validity`` (quantity, observed value, limit, setting, where it was first exceeded, code). A limit the caller
raises gives a valid state whose ``value_basis["*"]`` is a ``MODEL`` entry listing the checks passed under it.

References are the closed-form ABCD law (``q' = (A q + B) / (C q + D)``, ``helpers.py``) and the Gaussian beam formulas,
no Engine code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import abcd_apply, free, gaussian_radius_from_q, one_arrival, rayleigh_range, run, thin
from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import ErrorCode, Status
from os_optics.optics.gaussian import ops
from os_optics.optics.gaussian.ops import LocalCurvature
from os_optics.optics.gaussian.state import check_divergence, initial_state
from os_optics.results import basis_of
from os_optics.results.irradiance import irradiance_map
from os_optics.settings_registry import unrecorded_settings

LAM = 1064.0
X = (1.0, 0.0, 0.0)
NUMBERS = ("q_x_mm", "q_y_mm", "beam_radius_x_mm", "beam_radius_y_mm", "waist_radius_x_mm", "waist_radius_y_mm",
           "waist_distance_x_mm", "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm",
           "far_field_half_angle_x_rad", "far_field_half_angle_y_rad", "local_divergence_x_rad", "local_divergence_y_rad",
           "curvature_radius_x_mm", "curvature_radius_y_mm", "m2_x", "m2_y", "medium_index", "wavelength_nm", "e1", "e2")


def _plane(x: float, **kwargs):
    return B.observation_plane("d", position=(x, 0.0, 0.0), normal=X, **kwargs)


def _no_number(g) -> None:
    assert not g.valid and g.value_basis is None
    for name in NUMBERS:
        assert getattr(g, name) is None, name


def _check(g, *, setting: str, parameter: str, code: str, at: str):
    """The failed check the state states, with no number beside it."""
    _no_number(g)
    (check,) = g.validity
    assert (check.setting, check.parameter, check.code, check.at, check.within) == (setting, parameter, code, (at,), False)
    assert check.observed > check.maximum == getattr(NumericalConfig(), setting)
    return check


def _model_entry(g):
    entry = basis_of(g, "beam_radius_x_mm")
    assert entry is not None and entry is basis_of(g, "q_x_mm") and set(g.value_basis) == {"*"} and not g.validity
    assert entry.kind.value == "MODEL" and not entry.omitted and not entry.assumed
    return entry


def _divergence(q: complex) -> float:
    """The far-field half angle of a fundamental beam of parameter ``q`` (mm): ``w0 / z_R = sqrt(lambda / (pi z_R))``."""
    return math.sqrt(LAM * 1e-6 / (math.pi * q.imag))


def test_a_source_beyond_the_divergence_limit_states_the_failed_check_and_no_number():
    """A 0.5 um waist at 1064 nm: far-field half angle lambda / (pi w0) = 0.68 rad against the limit 0.30."""
    w0, z = 0.5e-3, 0.01
    laser = B.laser("S", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=w0, position=(0.0, 0.0, 0.0), direction=X)
    result = run(ProjectSnapshot(components=(laser,), observations=(_plane(z),),
                                 targets=(Target(target_id="t", metric="beam_radius_x_mm", observation_id="d", maximum=1.0),)))
    arrival = one_arrival(result, "d")
    g = arrival.beam.gaussian
    assert (g.valid, g.status.value, g.reason) == (False, "OUT_OF_MODEL_SCOPE", "GAUSSIAN_HIGH_NA")
    theta = LAM * 1e-6 / (math.pi * w0)
    check = _check(g, setting="gaussian_scope_divergence_rad", parameter="far_field_half_angle_rad", code="GAUSSIAN_HIGH_NA", at="S")
    assert check.observed == pytest.approx(theta, rel=1e-12) and check.unit == "rad"
    assert unrecorded_settings(result) == ()
    # no pupil ray is launched from a source beyond the limit (its far field needs another launch model)
    assert arrival.pupil_ray_count == 0
    # the target has no value: it reports the code and the same failed check
    (target,) = result.targets
    assert (target.status.value, target.assessment.value) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE")
    assert target.value is None and target.value_basis is None and target.validity == (check,)
    assert target.issue_codes == ("GAUSSIAN_HIGH_NA",)


def _focused(numerical: NumericalConfig | None = None, *, behind: float = 3.0):
    """A collimated 2 mm beam through an ideal f = 5 mm lens (far-field half angle about 0.4 rad behind it), read 3 mm
    behind the lens."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=2.0, position=(-50.0, 0.0, 0.0), direction=X)
    lens = B.thin_lens("L", focal_length_mm=5.0, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0), axis=X)
    return run(ProjectSnapshot(components=(laser, lens), observations=(_plane(behind),)), numerical=numerical)


def test_a_lens_that_focuses_beyond_the_limit_is_named_with_the_angle_it_gives():
    result = _focused()
    g = one_arrival(result, "d").beam.gaussian
    assert (g.valid, g.reason) == (False, "GAUSSIAN_HIGH_NA")
    surface = next(i.location for i in result.issues if i.code.value == "GAUSSIAN_HIGH_NA").split(":", 1)[1]
    check = _check(g, setting="gaussian_scope_divergence_rad", parameter="far_field_half_angle_rad", code="GAUSSIAN_HIGH_NA", at=surface)
    q_lens = abcd_apply(thin(5.0) @ free(50.0), complex(0.0, rayleigh_range(2.0, LAM)))
    assert check.observed == pytest.approx(_divergence(q_lens), rel=1e-9)
    assert surface.startswith("L/")


def test_a_raised_limit_gives_a_valid_state_with_the_first_order_numbers_and_a_model_entry():
    """The same scene with the divergence limit raised to 1 rad: the state is valid, its numbers are the closed-form ABCD
    beam's, and its entry is the model's value under the caller's limit."""
    raised = _focused(NumericalConfig(gaussian_scope_divergence_rad=1.0, gaussian_warn_divergence_rad=0.9))
    valid = one_arrival(raised, "d").beam.gaussian
    assert valid.valid and not valid.validity
    q = abcd_apply(free(3.0) @ thin(5.0) @ free(50.0), complex(0.0, rayleigh_range(2.0, LAM)))
    assert complex(*valid.q_x_mm) == pytest.approx(q, rel=1e-12) and complex(*valid.q_y_mm) == pytest.approx(q, rel=1e-12)
    assert valid.beam_radius_x_mm == pytest.approx(gaussian_radius_from_q(q, LAM), rel=1e-12)
    entry = _model_entry(valid)
    (check,) = entry.validity
    assert (check.setting, check.within, check.maximum, check.code) == ("gaussian_scope_divergence_rad", True, 1.0, None)
    assert check.observed == pytest.approx(_divergence(q), rel=1e-9)
    assert unrecorded_settings(raised) == ()


def test_the_state_stays_not_valid_and_names_where_the_limit_was_first_exceeded():
    """Behind a second lens that would collimate the beam again the state is still not valid: no number, and the check of
    the first lens."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=2.0, position=(-50.0, 0.0, 0.0), direction=X)
    first = B.thin_lens("L1", focal_length_mm=5.0, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0), axis=X)
    second = B.thin_lens("L2", focal_length_mm=5.0, clear_radius_mm=12.7, position=(10.0, 0.0, 0.0), axis=X)
    result = run(ProjectSnapshot(components=(laser, first, second), observations=(_plane(60.0),)))
    g = one_arrival(result, "d").beam.gaussian
    assert (g.valid, g.reason) == (False, "GAUSSIAN_HIGH_NA")
    check = _check(g, setting="gaussian_scope_divergence_rad", parameter="far_field_half_angle_rad", code="GAUSSIAN_HIGH_NA",
                   at=g.validity[0].at[0])
    assert check.at[0].startswith("L1/")


def test_a_severely_truncated_beam_states_the_truncated_share():
    """A 2 mm beam through a 1.5 mm stop loses exp(-2 a^2 / w^2) = 0.32 of its power: the state behind names the truncated
    share against the limit, and the power is the truncated beam's."""
    laser = B.laser("S", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=2.0, position=(-50.0, 0.0, 0.0), direction=X)
    stop = B.aperture_stop("A", position=(0.0, 0.0, 0.0), axis=X, radius_mm=1.5)
    result = run(ProjectSnapshot(components=(laser, stop), observations=(_plane(40.0),)))
    arrival = one_arrival(result, "d")
    g = arrival.beam.gaussian
    assert (g.valid, g.reason) == (False, "GAUSSIAN_SEVERE_TRUNCATION")
    cut = math.exp(-2.0 * 1.5 ** 2 / _radius(2.0, 50.0) ** 2)
    check = _check(g, setting="gaussian_clip_scope_fraction", parameter="truncated_power_fraction", code="GAUSSIAN_SEVERE_TRUNCATION",
                   at=next(i.location for i in result.issues if i.code.value == "GAUSSIAN_SEVERE_TRUNCATION").split(":", 1)[1])
    assert check.observed == pytest.approx(cut, rel=1e-6)
    assert arrival.beam.power_W == pytest.approx(1e-3 * (1.0 - cut), rel=1e-6)


def _radius(w0: float, z: float) -> float:
    return w0 * math.sqrt(1.0 + (z / rayleigh_range(w0, LAM)) ** 2)


def test_what_is_keyed_on_validity_states_no_gaussian_beam():
    """The map, the handoff and the coherent sum state nothing from a state outside the model's scope."""
    from os_optics.results.handoff import handoff

    result = _focused()
    doc = irradiance_map(result, "d")
    assert (doc["status"], doc["reason"]) == ("UNAVAILABLE", "GAUSSIAN_NOT_AVAILABLE")
    (beam,) = handoff(result, None, "d")["beams"]
    assert beam["gaussian"] is None and beam["validity"]["beam_parameters"] == "UNAVAILABLE" and beam["medium_index"] is None


# ------------------------------------------------------------------------------------------------ the operations
def _state(w0: float = 0.5e-3):
    cfg = NumericalConfig()
    e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    return initial_state(wavelength_nm=LAM, n=1.0, w0x=w0, w0y=w0, z_waist=0.0, m2x=1.0, m2y=1.0, bx=e1, by=e2, cfg=cfg), cfg


def test_a_state_beyond_a_limit_is_not_propagated_and_keeps_its_first_check():
    state, cfg = _state()
    assert not state.valid and state.limit.code is ErrorCode.GAUSSIAN_HIGH_NA
    # no step carries it on: the operations return it as it is
    assert ops.propagate(state, 2.0) is state
    d = np.array([1.0, 0.0, 0.0])
    undefined = LocalCurvature(0.0, 0.0, state.bx, state.by, undefined=(ErrorCode.GAUSSIAN_UNDEFINED_AFTER_AXICON, "no usable curvature"))
    ended = ops.apply_local_curvature(state, kind="refract", d_in=d, d_out=d, n_or=-d, cos_i=1.0, n1=1.0, n2=1.5, cos_t=1.0, tir=False,
                                      curvature=undefined, cfg=cfg)
    # the code and the failed check stay the first limit's (a state that is not valid keeps its reason)
    assert ended is state and ended.reason is ErrorCode.GAUSSIAN_HIGH_NA
    contract = ended.to_contract()
    _no_number(contract)
    (check,) = contract.validity
    assert (check.code, check.setting, check.within) == ("GAUSSIAN_HIGH_NA", "gaussian_scope_divergence_rad", False)
    # a later limit does not replace the first
    assert check_divergence(state, cfg) is state
    truncated, code = ops.mark_truncation(state, 0.5, cfg)
    assert truncated is state and code is None


def test_a_valid_state_under_the_default_limits_carries_no_entry():
    state, cfg = _state(1.0)
    assert state.valid and state.limit is None and state.passed == ()
    contract = ops.propagate(state, 10.0).to_contract()
    assert contract.valid and contract.value_basis is None and "value_basis" not in contract.model_dump()
    assert "validity" not in contract.model_dump()
    invalid = state.invalidated(ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, "twisted")
    assert invalid.limit is None and invalid.to_contract().model_dump(exclude_none=True) == {
        "valid": False, "status": Status.OUT_OF_MODEL_SCOPE, "reason": "GAUSSIAN_TWISTED_ASTIGMATISM"}


def _too_strong(numerical: NumericalConfig | None = None):
    """A 3 mm beam on a concave mirror of 10 mm radius of curvature (w / R = 0.3 against the limit 0.2)."""
    laser = B.laser("S", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=3.0, position=(0.0, 0.0, 0.0), direction=X)
    mirror = B.spherical_mirror("M", position=(50.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=10.0, clear_radius_mm=9.0)
    return run(ProjectSnapshot(components=(laser, mirror), observations=(B.observation_plane("d", position=(48.0, 0.0, 0.0), normal=X),)),
               numerical=numerical)


def _steep(numerical: NumericalConfig | None = None):
    """A 0.3 mm beam on a concave mirror (R = 300 mm) at 65 deg (limit 60 deg)."""
    a = math.radians(65.0)
    laser = B.laser("S", wavelength_nm=532.0, power_W=1e-3, waist_radius_mm=0.3, position=(0.0, 0.0, 0.0), direction=X)
    mirror = B.spherical_mirror("M", position=(50.0, 0.0, 0.0), normal=(-math.cos(a), math.sin(a), 0.0), radius_of_curvature_mm=300.0,
                                clear_radius_mm=10.0)
    out = (math.cos(2.0 * a - math.pi), math.sin(2.0 * a), 0.0)
    plane = B.observation_plane("d", position=(50.0 - 20.0 * math.cos(2.0 * a), 20.0 * math.sin(2.0 * a), 0.0), normal=out)
    return run(ProjectSnapshot(components=(laser, mirror), observations=(plane,)), numerical=numerical)


def _behind_mirror(result):  # (the beam behind the mirror: the last crossing of the plane)
    return [a for o in result.observations for a in o.arrivals if "M/face:reflect" in a.lineage][-1].beam.gaussian


def test_a_surface_too_strong_states_its_check_and_a_raised_limit_meets_the_aberration_check():
    """Behind the strong mirror: under the default limit the state names w / R against its setting; with that limit raised
    the valid state is refused as aberration dominated (its pupil rays spread far wider than the first-order envelope)."""
    wide = {"gaussian_scope_divergence_rad": 1.4, "gaussian_warn_divergence_rad": 1.3}
    raised = _behind_mirror(_too_strong(NumericalConfig(**wide, gaussian_max_w_over_radius=0.5)))
    assert (raised.valid, raised.reason, raised.q_x_mm) == (False, "GAUSSIAN_ABERRATION_DOMINATED", None)
    limited = _behind_mirror(_too_strong(NumericalConfig(**wide)))
    assert (limited.valid, limited.reason) == (False, "GAUSSIAN_SURFACE_TOO_STRONG")
    check = _check(limited, setting="gaussian_max_w_over_radius", parameter="beam_radius_over_surface_radius",
                   code="GAUSSIAN_SURFACE_TOO_STRONG", at="M/face")
    assert check.observed == pytest.approx(_radius_532(3.0, 50.0) / 10.0, rel=1e-9)


def _radius_532(w0: float, z: float) -> float:
    z_r = math.pi * w0 * w0 / 532e-6
    return w0 * math.sqrt(1.0 + (z / z_r) ** 2)


def test_the_strength_limit_ends_the_step_and_a_raised_limit_applies_the_mirror_law():
    """The operation itself: a 3 mm beam on a concave mirror of 10 mm radius at normal incidence. Beyond the limit the step
    is not applied; with the limit raised it gives ``1/q' = 1/q - 2/R``."""
    state, cfg = _state(3.0)
    d = np.array([1.0, 0.0, 0.0])
    out = ops.apply_curved_or_flat_interface(state, kind="mirror", d_in=d, d_out=-d, n_or=-d, cos_i=1.0, n1=1.0, n2=1.0, cos_t=1.0,
                                             tir=False, radius_signed_mm=10.0, cfg=cfg)
    assert not out.valid and out.reason is ErrorCode.GAUSSIAN_SURFACE_TOO_STRONG and (out.qx, out.qy) == (state.qx, state.qy)
    limit = out.limit
    assert (limit.setting, limit.parameter, limit.maximum) == ("gaussian_max_w_over_radius", "beam_radius_over_surface_radius", 0.2)
    assert limit.observed == pytest.approx(0.3, rel=1e-12)
    raised = NumericalConfig(gaussian_max_w_over_radius=0.5, gaussian_scope_divergence_rad=1.4, gaussian_warn_divergence_rad=1.3)
    valid = ops.apply_curved_or_flat_interface(state, kind="mirror", d_in=d, d_out=-d, n_or=-d, cos_i=1.0, n1=1.0, n2=1.0, cos_t=1.0,
                                               tir=False, radius_signed_mm=10.0, cfg=raised)
    expected = 1.0 / (1.0 / state.qx - 2.0 / 10.0)
    assert valid.valid and valid.qx == pytest.approx(expected, rel=1e-12) and valid.qy == pytest.approx(expected, rel=1e-12)
    assert [(p.setting, p.observed) for p in valid.passed if p.setting == "gaussian_max_w_over_radius"] == [("gaussian_max_w_over_radius", limit.observed)]


@pytest.mark.parametrize("scene, setting, raised, code, parameter", [
    (_steep, "gaussian_max_aoi_deg", 70.0, "GAUSSIAN_LARGE_AOI", "aoi_deg"),
])
def test_each_accuracy_limit_states_its_check_and_a_raised_limit_gives_a_model_entry(scene, setting, raised, code, parameter):
    """Beyond the limit the state names the angle of incidence (65 deg) against its setting; with that one limit raised the
    state is valid with a MODEL entry listing the check passed (the divergence limits raised in both runs)."""
    wide = {"gaussian_scope_divergence_rad": 1.4, "gaussian_warn_divergence_rad": 1.3}
    limited = _behind_mirror(scene(NumericalConfig(**wide)))
    assert (limited.valid, limited.reason) == (False, code)
    check = _check(limited, setting=setting, parameter=parameter, code=code, at="M/face")
    assert check.observed == pytest.approx(65.0, rel=1e-9)
    valid = _behind_mirror(scene(NumericalConfig(**wide, **{setting: raised})))
    assert valid.valid
    within = {c.setting: c for c in _model_entry(valid).validity}
    assert all(c.within for c in within.values())
    assert within[setting].maximum == raised and within[setting].observed == pytest.approx(check.observed, rel=1e-12)


#: Every module that reads the published Gaussian block of an arrival (``beam.gaussian``). Each reads the numbers of a
#: valid state only (a state outside the model's scope publishes none), or carries the block's entry with them (the tables,
#: the JSON rows, a target's value). A module that starts reading the block is added here after the same check.
GAUSSIAN_BLOCK_READERS = {
    "analysis/configurations.py", "local/json_api.py", "optical_view/builder.py", "results/handoff.py", "results/irradiance.py",
    "results/basis.py", "results/rays.py", "results/tables.py", "validation/alignment.py", "validation/descriptors.py", "validation/metrics.py",
    "validation/superposition.py",
}


def test_the_readers_of_the_published_gaussian_block_are_the_checked_ones():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "os_optics"
    readers = {path.relative_to(src).as_posix() for path in src.rglob("*.py")
               if "_vendor" not in path.parts and "beam.gaussian" in path.read_text(encoding="utf-8")}
    assert readers == GAUSSIAN_BLOCK_READERS


def test_the_rows_of_a_state_outside_the_scope_carry_no_number():
    from os_optics.results.tables import arrival_row

    result = _focused()
    arrival = one_arrival(result, "d")
    row = arrival_row("d", arrival)
    assert row["gaussian_valid"] is False and row["beam_radius_x_mm"] is None
    assert "beam_radius_x_mm" not in (row.get("value_basis") or {})


def test_a_probe_of_untraced_light_keeps_the_failed_check_and_no_number():
    """A 0.5 um waist (far beyond the divergence limit) through an uncoated plate: the plate's double reflection, a probe,
    reaches the plane with the source's failed check and no number, like the traced beam."""
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.optics.rays.tracer import Tracer
    from os_optics.scene.lower import lower_scene
    from os_optics.validation.untraced import follow_untraced

    laser = B.laser("S", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=0.5e-3, position=(0.0, 0.0, 0.0), direction=X)
    plate = B.window("W", thickness_mm=1.0, clear_radius_mm=12.7, material="N-BK7", position=(2.0, 0.0, 0.0))
    project = ProjectSnapshot(components=(laser, plate), observations=(_plane(6.0),))
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    trace = Tracer(scene, library, cfg, include_pupil=True).run()
    traced = [a for a in trace.arrivals if a.observation_id == "d"]
    probes = follow_untraced(trace, scene, library, cfg)
    crossings = [a for _, outcome in probes.results for a in outcome.arrivals if a.observation_id == "d"]
    assert traced and crossings
    for a in (*traced, *crossings):
        assert not a.gstate.valid and a.gstate.reason is ErrorCode.GAUSSIAN_HIGH_NA
        assert a.gstate.limit is not None and a.gstate.limit.at == "S"
        contract = a.gstate.to_contract()
        _no_number(contract)
        assert contract.validity[0].code == "GAUSSIAN_HIGH_NA"


def test_the_summary_target_row_carries_the_failed_check():
    """A 0.5 um waist at 780 nm (far-field half angle 0.50 rad): the run summary's target row (Python and json_api) has no
    value and states the failed check."""
    from local_helpers import plane, project, run as run_document, source
    from os_optics.local import json_api

    document = project([source(position=(0.0, 0.0, 0.0), waist_mm=0.5e-3)], [plane("d", (0.01, 0.0, 0.0))],
                       [{"target_id": "t", "metric": "beam_radius_x_mm", "observation_id": "d", "maximum": 1.0}])
    summary = json_api.run_summary(run_document(document))
    (row,) = summary["targets"]
    assert row["value"] is None and "value_basis" not in row
    (check,) = row["validity"]
    assert (check["code"], check["setting"], check["within"], check["at"]) == ("GAUSSIAN_HIGH_NA", "gaussian_scope_divergence_rad", False, ["SRC"])
    assert check["observed"] == pytest.approx(780e-6 / (math.pi * 0.5e-3), rel=1e-6)


def test_the_results_targets_row_carries_the_failed_check(tmp_path):
    """The same run written to a directory: ``results targets`` (``RunResults.targets()``) and the JSON API's
    ``run_results({"kind": "targets"})`` -- the readers of a run directory -- give the target no value and the failed check
    with every structured fact, as the summary row does."""
    from local_helpers import plane, project, run as run_document, source
    from os_optics.local import json_api
    from os_optics.results import load_run

    document = project([source(position=(0.0, 0.0, 0.0), waist_mm=0.5e-3)], [plane("d", (0.01, 0.0, 0.0))],
                       [{"target_id": "t", "metric": "beam_radius_x_mm", "observation_id": "d", "maximum": 1.0}])
    out = tmp_path / "run"
    run_document(document, output_dir=str(out))
    files = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    reply = json_api.run_results(files, {"kind": "targets"})
    assert reply["ok"] is True, reply
    for row in (load_run(out).targets()["targets"][0], reply["result"]["targets"][0]):
        assert row["value"] is None and "value_basis" not in row
        (check,) = row["validity"]
        assert (check["parameter"], check["code"], check["setting"], check["within"], check["at"]) == (
            "far_field_half_angle_rad", "GAUSSIAN_HIGH_NA", "gaussian_scope_divergence_rad", False, ["SRC"])
        assert check["maximum"] == NumericalConfig().gaussian_scope_divergence_rad
        assert check["observed"] == pytest.approx(780e-6 / (math.pi * 0.5e-3), rel=1e-12)


def test_the_summary_arrival_row_states_the_failed_check_where_it_has_no_radius():
    """The JSON API's arrival row of a beam beyond the divergence limit has no radius, waist or Rayleigh range; it states
    the reason and the failed check (quantized like the row's other numbers), once per beam. A beam within the limits
    states neither."""
    from local_helpers import plane, project, run as run_document, source
    from os_optics.local import json_api

    def arrival(waist_mm: float) -> dict:
        document = project([source(position=(0.0, 0.0, 0.0), waist_mm=waist_mm)], [plane("d", (0.01, 0.0, 0.0))])
        (row,) = json_api.run_summary(run_document(document))["observations"]
        (beam,) = row["arrivals"]
        return beam

    beam = arrival(0.5e-3)
    assert beam["beam_radius_x_mm"] is None and beam["waist_radius_x_mm"] is None and beam["rayleigh_range_x_mm"] is None
    assert beam["gaussian_reason"] == "GAUSSIAN_HIGH_NA"
    (check,) = beam["gaussian_validity"]
    assert (check["parameter"], check["code"], check["setting"], check["within"], check["at"]) == (
        "far_field_half_angle_rad", "GAUSSIAN_HIGH_NA", "gaussian_scope_divergence_rad", False, ["SRC"])
    assert check["maximum"] == NumericalConfig().gaussian_scope_divergence_rad
    assert check["observed"] == pytest.approx(780e-6 / (math.pi * 0.5e-3), rel=1e-6)
    valid = arrival(0.5)
    assert valid["beam_radius_x_mm"] is not None and "gaussian_validity" not in valid and "gaussian_reason" not in valid
