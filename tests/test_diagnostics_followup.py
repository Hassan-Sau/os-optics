"""Follow-up review of the diagnostics fixes (N1, N2, N4, N5, N8).

* **N1** -- first-order untraced light (a ghost reflection, light a reflector's coating transmits, the part of a clipped
  beam that passes an opening) is followed by a probe trace *through* the optics it meets; a plane no traced beam of a
  line reaches keeps its known zero where that light provably cannot arrive (an ordinary lens or window upstream no
  longer turns dark planes UNKNOWN), and is UNKNOWN where it can, with the power arriving there (N8).
* **N2** -- control readings trace without pupil rays: behind a central obstruction the annulus is carried by its bounds
  alone, so the reading is the annulus's power (no false BEAM_LOST); where it is not followed the reading is not known
  and the control analysis is UNKNOWN, never a lost beam.
* **N4** -- the annulus's bound includes the opening that cut it (a Cassegrain-like annulus through a lens slightly larger
  than the opening keeps its exact power).
* **N5** -- an active area that holds all but less than capture_min_power_fraction of the annulus counts it, with a WARNING
  stating the bound.

References are independent of the Engine: the Fresnel equations at normal incidence with the SCHOTT Sellmeier formula of
N-BK7, the closed-form annulus ``exp(-2 a^2 / w^2) - exp(-2 R^2 / w^2)``, Gaussian q propagation (``q -> q + L``,
``1/q -> 1/q - 1/f``, ``w^2 = -lambda / (pi Im(1/q))``) and Malus' law.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import (
    ApertureStopInteraction,
    CircularAperture,
    EngineeringComponentSnapshot,
    PlaneSurface,
    ProjectSnapshot,
    Target,
)
from os_optics.errors import Status

from helpers import assert_dark_truncated, assert_power_not_computed, rayleigh_range, run
from test_control import _control, _mirror_project

X = (1.0, 0.0, 0.0)
P_W = 1e-3
NM = 1e-6


def _n_bk7(wavelength_nm: float) -> float:
    l2 = (wavelength_nm * 1e-3) ** 2
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _w(w0: float, lam: float, z: float) -> float:
    return w0 * math.sqrt(1.0 + (z / rayleigh_range(w0, lam)) ** 2)


def _annulus(a: float, r: float, w: float) -> float:
    return math.exp(-2.0 * a * a / (w * w)) - math.exp(-2.0 * r * r / (w * w))


def _laser(sid: str = "SRC", *, lam: float = 633.0, w0: float = 0.5, pos=(0.0, 0.0, 0.0), **kw):
    return B.laser(sid, wavelength_nm=lam, power_W=P_W, waist_radius_mm=w0, position=pos, direction=X, **kw)


def _stop(iid: str, x: float, radius: float, obstruction: float = 0.0, *, y: float = 0.0, z: float = 0.0):
    surface = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=radius, obstruction_radius_mm=obstruction), mount_margin_mm=10.0,
                           medium_positive="air", medium_negative="air", interaction=ApertureStopInteraction())
    return EngineeringComponentSnapshot(instance_id=iid, role="aperture", pose=B._pose((x, y, z), X), surfaces=(surface,))


def _t(target_id: str, observation_id: str, **kw) -> Target:
    return Target(target_id=target_id, metric="power_W", observation_id=observation_id, unit="W", **kw)


def _obs(result, observation_id: str):
    return next(o for o in result.observations if o.observation_id == observation_id)


def _faint_bounded(result, target_id: str, observation_id: str):
    """No traced arrival: a known 0 W. The light left out as negligible is its stated truncation, with a positive bound below
    the 1e-9 W target in the target's entry and in the plane's upper bound."""
    target = next(t for t in result.targets if t.target_id == target_id)
    upper = assert_dark_truncated(target)
    obs = _obs(result, observation_id)
    assert obs.power_complete and obs.total_power_W == 0.0 and obs.untraced_power_W == 0.0
    assert obs.quantity_state.value == "KNOWN_ZERO" and 0 < upper <= obs.power_upper_bound_W <= 1e-9


# ------------------------------------------------------------------------------------------------------------------- N1
def _crossed(window: bool):
    comps = [_laser(), B.polarizer("P1", position=(30.0, 0, 0), axis=X, angle_deg=0.0),
             B.polarizer("P2", position=(60.0, 0, 0), axis=X, angle_deg=90.0)]
    if window:
        comps.insert(1, B.window("W", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(10.0, 0, 0)))
    det = B.observation_plane("DET", position=(100.0, 0, 0), normal=X, radius_mm=5.0, absorbing=True)
    return ProjectSnapshot(components=tuple(comps), observations=(det,), targets=(_t("leak_max", "DET", maximum=1e-9),))


@pytest.mark.parametrize("window", [False, True])
def test_crossed_polarizers_retain_a_faint_bound_with_an_uncoated_window_upstream(window):
    """Malus: cos^2(90 deg) = 0 passes the crossed pair. The window's two ghosts travel back toward the source (the second
    through the first face) and leave the scene: they cannot reach the detector, so only the omitted faint remainder bounds the detector (with the window it
    used to be UNKNOWN, 82 uW 'may arrive')."""
    _faint_bounded(run(_crossed(window)), "leak_max", "DET")


def test_a_coherent_power_behind_the_crossed_pair_is_bounded_where_the_left_out_field_is_above_the_level():
    """The window scene: the light left out as negligible brings at most the level (1e-12 W of the 1 mW line) to the
    detector, so its power_W keeps the known 0 W. Its field, the amplitudes of several faint pieces added, bounds a coherent
    sum of nothing above that level (W1 at the sum, P = 0): coherent_power_W has no value and no entry, and is graded on
    [0, r^2] -- a maximum of 1e-9 W is met, one of 1e-11 W (inside that interval) is not decided."""
    snapshot = _crossed(True)
    coherent = tuple(Target(target_id=f"coh_{tag}", metric="coherent_power_W", observation_id="DET", unit="W", maximum=maximum)
                     for tag, maximum in (("max", 1e-9), ("tight", 1e-11)))
    result = run(snapshot.model_copy(update={"targets": snapshot.targets + coherent}))
    _faint_bounded(result, "leak_max", "DET")
    by_id = {t.target_id: t for t in result.targets}
    for target_id, outcome in (("coh_max", ("PASS", "MET")), ("coh_tight", ("UNKNOWN", "INDETERMINATE"))):
        target = by_id[target_id]
        assert target.value is None and not target.value_basis, target
        assert (target.status.value, target.assessment.value) == outcome
        assert "FAINT_LIGHT_BOUNDED" in target.issue_codes and "may, with a field of at most sqrt(" in target.message
    # (the plane keeps its known zero; its power_W target keeps 0.0 with a BELOW_NEGLIGIBLE_LEVEL entry; a minimum above
    # r^2 is unmet -- what the README and RESULTS_AND_EXPORT state for this case)
    plane = _obs(result, "DET")
    assert plane.quantity_state.value == "KNOWN_ZERO" and plane.total_power_W == 0.0 and plane.power_complete is True
    leak = by_id["leak_max"]
    assert leak.value == 0.0 and [o.part.value for o in leak.value_basis["value"].omitted] == ["BELOW_NEGLIGIBLE_LEVEL"]
    above = Target(target_id="coh_min", metric="coherent_power_W", observation_id="DET", unit="W", minimum=1e-9)
    (unmet,) = run(snapshot.model_copy(update={"targets": (above,)})).targets
    assert (unmet.value, unmet.assessment.value) == (None, "UNMET")
    root = Path(__file__).resolve().parents[1]
    readme = " ".join((root / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    results = " ".join((root / "docs" / "engine" / "RESULTS_AND_EXPORT.md").read_text(encoding="utf-8").split())
    assert "a `power_W` or `coherent_power_W` target there reports the traced value with that amount" not in readme
    assert ("A `coherent_power_W` target adds fields, so it reads the bound of that light's field at the sum instead: the rise "
            "(sqrt(P) + r)^2 - P on a lit plane and r^2 on a dark one") in readme
    assert ("on a dark plane the target has no value and no entry and is graded on [0, r^2]: a maximum at or above r^2 is "
            "`MET`, a minimum above it `UNMET`, any other limit `INDETERMINATE`. Either way the plane keeps `KNOWN` / "
            "`KNOWN_ZERO` and `power_complete` true.") in readme
    assert ("the value is an `ESTIMATE` whose `POSSIBLE_RETURN` entry holds the larger of the rise and the fall") in readme
    assert "`power_W` or `coherent_power_W` target on it evaluates 0 W" not in results
    assert ("The plane is `BOUNDED` from 0 (`UNKNOWN` without a bound) where the power that light brings exceeds the level. "
            "Its field r^2 may exceed the level while that power stays within it: then the plane stays `KNOWN_ZERO`") in results


def test_a_ghost_the_probe_budget_does_not_reach_is_not_a_known_zero():
    """With max_paths = 1 the probe trace follows the first ghost only; the second (R (1 - R) P at normal incidence, P the
    line's power) is not followed, so the detector is not a known zero: UNKNOWN with that power."""
    n = _n_bk7(633.0)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    result = run(_crossed(True), numerical=NumericalConfig(max_paths=1))
    (target,) = result.targets
    assert (target.status, target.value) == (Status.UNKNOWN, None)
    assert _obs(result, "DET").untraced_power_W == pytest.approx(r * (1.0 - r) * P_W, rel=1e-6)
    (budget,) = [i for i in result.issues if i.code.value == "MAX_PATHS" and i.location == "untraced_light"]
    assert budget.numerical_context["budget"] == ["max_paths"]
    assert budget.numerical_context["untraced_branches"] == 1
    assert budget.numerical_context["untraced_power_W"] == pytest.approx(r * (1.0 - r) * P_W, rel=1e-6)


def test_window_ghosts_reach_a_detector_behind_the_source_with_their_power_there():
    """A normal-incidence N-BK7 window 20 mm ahead of the source; a detector on the axis 20 mm behind the source (no traced
    beam crosses it) receives both ghosts: R P from the first face and (1 - R) R (1 - R) P from the second, which leaves
    through the first face, and every further internal bounce (ghosts of ghosts are followed too). The plane is UNKNOWN and states that power (the power arriving there, not at the
    ghosts' origin). A detector 30 mm beside it has no traced arrival but retains a faint-light bound."""
    n = _n_bk7(633.0)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(20.0, 0, 0))
    back = B.observation_plane("BACK", position=(-20.0, 0, 0), normal=X, radius_mm=5.0, absorbing=True)
    side = B.observation_plane("SIDE", position=(-20.0, 30.0, 0), normal=X, radius_mm=5.0, absorbing=True)
    result = run(ProjectSnapshot(components=(_laser(), window), observations=(back, side),
                                 targets=(_t("back_max", "BACK", maximum=1e-9), _t("side_max", "SIDE", maximum=1e-9))))
    target = next(t for t in result.targets if t.target_id == "back_max")
    assert (target.status, target.value, target.issue_codes) == (Status.UNKNOWN, None, ("UNTRACED_LIGHT_BOUNDED",))  # (followed reflections: R7)
    obs = _obs(result, "BACK")
    assert obs.total_power_W is None and not obs.power_complete
    # every internal bounce leaves toward it too (followed to every order): the plate's reflectance R + T^2 R / (1 - R^2)
    assert obs.untraced_power_W == pytest.approx((r + (1.0 - r) ** 2 * r / (1.0 - r * r)) * P_W, rel=1e-6)
    _faint_bounded(result, "side_max", "SIDE")


def test_dark_planes_beside_a_lens_retain_faint_bounds():
    """An uncoated N-BK7 singlet in front of a beam dump: its ghosts return along the axis toward the source. A side
    detector 60 mm off the axis facing it, and a detector 30 mm beside the source looking forward, receive no followed ghost, but retain bounds on omitted faint light (they used to be UNKNOWN, 'may arrive' about 4 % of the power)."""
    lens = B.singlet_lens("L1", r1_mm=51.5, r2_mm=math.inf, thickness_mm=3.6, clear_radius_mm=12.7, material="N-BK7",
                          position=(40.0, 0, 0))
    dump = B.observation_plane("DUMP", position=(200.0, 0, 0), normal=X, radius_mm=12.0, absorbing=True)
    side = B.observation_plane("SIDE", position=(100.0, 60.0, 0), normal=(0.0, -1.0, 0.0), radius_mm=5.0, absorbing=True)
    back = B.observation_plane("BACK", position=(-20.0, 30.0, 0), normal=X, radius_mm=5.0, absorbing=True)
    result = run(ProjectSnapshot(components=(_laser(), lens), observations=(dump, side, back),
                                 targets=(_t("side_max", "SIDE", maximum=1e-9), _t("back_max", "BACK", maximum=1e-9))))
    _faint_bounded(result, "side_max", "SIDE")
    _faint_bounded(result, "back_max", "BACK")


def test_another_sources_lens_ghosts_preserve_traced_power_with_faint_bounds():
    """Source B goes through its own lens into a dump; A lights detector DETA. B's target at DETA is a known zero, and DETA's
    total power (A's 1 mW) is complete (it used to be a WARNING, B's ghosts 'may arrive')."""
    a, b = _laser("A"), _laser("B", pos=(0.0, 40.0, 0.0))
    lens = B.singlet_lens("LB", r1_mm=51.5, r2_mm=math.inf, thickness_mm=3.6, clear_radius_mm=12.7, material="N-BK7",
                          position=(40.0, 40.0, 0))
    deta = B.observation_plane("DETA", position=(100.0, 0, 0), normal=X, radius_mm=5.0, absorbing=True)
    dumpb = B.observation_plane("DUMPB", position=(100.0, 40.0, 0), normal=X, radius_mm=12.0, absorbing=True)
    result = run(ProjectSnapshot(components=(a, b, lens), observations=(deta, dumpb),
                                 targets=(_t("b_at_a", "DETA", source_id="B", maximum=1e-9), _t("total", "DETA", minimum=1e-4))))
    t = {x.target_id: x for x in result.targets}
    assert_dark_truncated(t["b_at_a"])
    # (the light the run leaves out as negligible is the stated truncation of the traced value)
    assert (t["total"].status, t["total"].issue_codes) == (Status.PASS, ("FAINT_LIGHT_BOUNDED",))
    assert t["total"].value == pytest.approx(P_W, rel=1e-12)
    assert t["total"].value_basis["value"].kind.value == "MODEL"
    obs = _obs(result, "DETA")
    assert obs.power_complete and obs.quantity_state.value == "KNOWN"
    assert obs.power_upper_bound_W > obs.total_power_W


# ------------------------------------------------------------------------------------------------------------------- N2
def _obstructed_mirror_project(det_radius: float = 30.0):
    project = _mirror_project(det_radius=det_radius)
    return project.model_copy(update={"components": (*project.components, _stop("AN", 50.0, 4.0, 0.2, z=20.0))})


def test_control_through_a_central_obstruction_reads_the_annulus_not_a_lost_beam():
    """The suite's mirror-steering control Project with a 0.2 mm central obstruction in a 4 mm opening 50 mm from the waist:
    the control readings (no pupil rays) carry the annulus by its envelope bound, so the control analysis runs and passes
    (it used to FAIL with BEAM_LOST while the same run showed 0.7265 mW on the sensor)."""
    project = _obstructed_mirror_project()
    control, result = _control(project)
    assert control is not None and control.status is Status.PASS
    assert "BEAM_LOST" not in {i.code.value for i in result.issues}
    source_power = next(c.source.power_W for c in project.components if c.source is not None)
    assert _obs(result, "det").total_power_W == pytest.approx(source_power * _annulus(0.2, 4.0, _w(0.5, 633.0, 50.0)), rel=1e-9)


def test_a_control_reading_that_is_not_known_makes_the_analysis_unknown_not_a_lost_beam():
    """The same with a 0.3 mm sensor: it takes part of the annulus, so the annulus is not followed there and the reading is
    not known: the control analysis is UNKNOWN (OUT_OF_MODEL_SCOPE), no BEAM_LOST failure."""
    control, result = _control(_obstructed_mirror_project(det_radius=0.3))
    assert control is not None and control.status is Status.UNKNOWN
    codes = {(i.code.value, i.status) for i in result.issues if i.location == "control"}
    assert codes == {("OUT_OF_MODEL_SCOPE", Status.UNKNOWN)}
    assert "BEAM_LOST" not in {i.code.value for i in result.issues}


# --------------------------------------------------------------------------------------------------------------- N4 / N5
def _cassegrain(lens_radius: float):
    """w0 4 mm at 633 nm, a 3 mm obstruction in a 12 mm opening 10 mm on, an f 100 mm thin lens 10 mm behind it and a
    0.5 mm detector at the focus."""
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=lens_radius, position=(20.0, 0, 0))
    det = B.observation_plane("OUT", position=(120.0, 0, 0), normal=X, radius_mm=0.5, absorbing=True)
    return ProjectSnapshot(components=(_laser(w0=4.0), _stop("STOP", 10.0, 12.0, 3.0), lens), observations=(det,),
                           targets=(_t("p", "OUT", minimum=0.0),))


def test_a_lens_slightly_larger_than_the_opening_passes_the_whole_annulus():
    """The light beyond the 12 mm opening was removed at the stop: paraxially the opening maps inside 12 / w_stop envelope
    radii (3.0) everywhere behind it, i.e. inside 12.0 mm at the lens, so a 12.7 mm lens passes the whole annulus and the
    detector at the focus reads its exact power 0.3247 mW (the unobstructed envelope alone leaves 1.7e-9 outside 12.7 mm, so
    it used to be 'not followed'). An 11 mm lens cuts it: the power behind the lens is not computed (the rays that pass go
    on to the detector, which states no power, at most the annulus's)."""
    whole = run(_cassegrain(12.7))
    assert whole.targets[0].value == pytest.approx(P_W * _annulus(3.0, 12.0, _w(4.0, 633.0, 10.0)), rel=1e-9)
    assert whole.targets[0].value == pytest.approx(0.3246524579e-3, rel=1e-9)
    cut = run(_cassegrain(11.0))
    _obs, arrival = assert_power_not_computed(cut, "OUT", state=None)
    (issue,) = [i for i in cut.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert issue.location == "surface:LENS/lens"
    assert arrival.power_upper_bound_W == pytest.approx(P_W * _annulus(3.0, 12.0, _w(4.0, 633.0, 10.0)), rel=1e-9)
    assert cut.targets[0].value is None and cut.targets[0].status is not Status.FAIL


def test_a_detector_holding_all_but_a_capture_fraction_of_the_annulus_counts_it_with_the_bound():
    """Obstruction 0.5 mm, f 100 mm lens, a 1.0 mm detector halfway to the focus (envelope w ~ 0.5 mm): at most
    exp(-2 r^2 / w^2) / T ~ 5.5e-4 of the annulus can fall beside it (< capture_min_power_fraction 1e-3): it counts the
    annulus's power with an INCOMPLETE_ENVELOPE warning stating that bound (it used to be 'not followed'); a 0.3 mm detector
    takes part of it: not followed."""
    lam = 633.0
    zr = rayleigh_range(1.0, lam)
    q = complex(20.0, zr)  # at the lens (waist at the source)
    q = 1.0 / (1.0 / q - 1.0 / 100.0) + 50.0  # thin lens, then 50 mm
    w = math.sqrt(-lam * NM / (math.pi * (1.0 / q).imag))
    transmitted = _annulus(0.5, 4.0, _w(1.0, lam, 10.0))
    bound = math.exp(-2.0 * 1.0 / w**2) / transmitted
    assert 1e-4 < bound < 1e-3
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=10.0, position=(20.0, 0, 0))

    def project(radius):
        det = B.observation_plane("OUT", position=(70.0, 0, 0), normal=X, radius_mm=radius, absorbing=True)
        return ProjectSnapshot(components=(_laser(w0=1.0), _stop("STOP", 10.0, 4.0, 0.5), lens), observations=(det,),
                               targets=(_t("p", "OUT", minimum=0.0),))

    counted = run(project(1.0))
    assert counted.targets[0].value == pytest.approx(P_W * transmitted, rel=1e-9)
    (warning,) = [i for i in counted.issues if i.code.value == "INCOMPLETE_ENVELOPE"]
    assert warning.status is Status.WARNING
    assert warning.numerical_context["annulus_power_beside_bound"] == pytest.approx(bound, rel=1e-6)
    partial = run(project(0.3))
    _obs, arrival = assert_power_not_computed(partial, "OUT", state=None)
    assert [p.termination for p in partial.paths] == ["DETECTED"]  # (the detector takes part of the annulus: how much is not computed)
    assert arrival.power_upper_bound_W == pytest.approx(P_W * transmitted, rel=1e-9)
