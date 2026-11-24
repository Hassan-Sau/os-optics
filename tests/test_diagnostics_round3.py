"""Round-3 follow-up review of the diagnostics (F1, F3, F4).

* **F1** -- untraced light is followed to every order: a probe's own ghosts are probed in turn (strongest first, down to
  ``negligible_power_fraction`` x the line power). A detector that only a ghost of a ghost reaches is UNKNOWN with the power
  arriving there, never a known 0 W; where the probes' budget cannot follow the cascade, the plane is UNKNOWN and the run
  carries a ``MAX_PATHS`` issue for the untraced light.
* **F3** -- an annulus counted on an active area that may miss up to ``capture_min_power_fraction`` of it carries a lower
  bound (``detected_power_min_W``); a power target on it is judged on that interval and states ``INCOMPLETE_ENVELOPE``.
* **F4** -- the opening bound follows the opening's own position: a decentred annulus whose 12 mm opening lies inside a
  12.3 mm lens passes it whole.

References are independent: the Fresnel equations with the SCHOTT Sellmeier formula of N-BK7 (for the two-window double
ghost the product of the two windows' plate reflectances ``R + T^2 R / (1 - R^2)`` -- the front face and every
internal bounce leaving through it --, per polarization), Gaussian q propagation, the closed-form annulus and the off-centre disc integral
``F(a; d, w) = int_0^a (4 r / w^2) exp(-2 (r^2 + d^2) / w^2) I0(4 r d / w^2) dr``.
"""

from __future__ import annotations

import math

import numpy as np
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

from helpers import assert_power_not_computed, assert_dark_bounded, one_arrival, rayleigh_range, run

X = (1.0, 0.0, 0.0)
P_W = 1e-3
NM = 1e-6


def _n_bk7(wavelength_nm: float) -> float:
    l2 = (wavelength_nm * 1e-3) ** 2
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def _fresnel(n: float, aoi_rad: float) -> tuple[float, float]:
    ci = math.cos(aoi_rad)
    ct = math.sqrt(1.0 - (math.sin(aoi_rad) / n) ** 2)
    rs = (ci - n * ct) / (ci + n * ct)
    rp = (n * ci - ct) / (n * ci + ct)
    return rs * rs, rp * rp


def _w(w0: float, lam: float, z: float) -> float:
    return w0 * math.sqrt(1.0 + (z / rayleigh_range(w0, lam)) ** 2)


def _annulus(a: float, r: float, w: float) -> float:
    return math.exp(-2.0 * a * a / (w * w)) - math.exp(-2.0 * r * r / (w * w))


def _disc(a: float, d: float, w: float, nodes: int = 400) -> float:
    x, weights = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * a * (x + 1.0)
    f = (4.0 * r / (w * w)) * np.exp(-2.0 * (r * r + d * d) / (w * w)) * np.i0(4.0 * r * d / (w * w))
    return float(0.5 * a * np.sum(weights * f))


def _laser(*, w0: float = 0.5, pos=(0.0, 0.0, 0.0)):
    return B.laser("SRC", wavelength_nm=633.0, power_W=P_W, waist_radius_mm=w0, position=pos, direction=X)


def _stop(iid: str, x: float, radius: float, obstruction: float = 0.0):
    surface = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=radius, obstruction_radius_mm=obstruction), mount_margin_mm=10.0,
                           medium_positive="air", medium_negative="air", interaction=ApertureStopInteraction())
    return EngineeringComponentSnapshot(instance_id=iid, role="aperture", pose=B._pose((x, 0.0, 0.0), X), surfaces=(surface,))


def _t(target_id: str, observation_id: str, **kw) -> Target:
    return Target(target_id=target_id, metric="power_W", observation_id=observation_id, unit="W", **kw)


def _obs(result, observation_id: str):
    return next(o for o in result.observations if o.observation_id == observation_id)


def _reflect(d, n):
    d, n = np.asarray(d, float), np.asarray(n, float)
    return d - 2.0 * float(d @ n) * n


# ------------------------------------------------------------------------------------------------------------------- F1
TH1, TH2 = math.radians(10.0), math.radians(15.0)


def _two_windows(traced: bool) -> ProjectSnapshot:
    """W1 (N-BK7, 3 mm) at 10 deg in the beam; its ghost meets W2 at 15 deg 60 mm away; W2's reflection of that ghost goes
    to detector D 60 mm on. No traced beam reaches D (the review's case)."""
    n1 = np.array((math.cos(TH1), math.sin(TH1), 0.0))
    r1 = _reflect(X, n1)
    w2_pos = np.array((50.0, 0.0, 0.0)) + 60.0 * r1
    c, s = math.cos(TH2), math.sin(TH2)
    n2 = np.array((-c * r1[0] + s * r1[1], -s * r1[0] - c * r1[1], 0.0))
    r2 = _reflect(r1, n2)
    w1 = B.window("W1", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(50.0, 0.0, 0.0), axis=tuple(n1),
                  trace_fresnel_reflection=traced)
    w2 = B.window("W2", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=tuple(w2_pos), axis=tuple(n2),
                  trace_fresnel_reflection=traced)
    d = B.observation_plane("D", position=tuple(w2_pos + 60.0 * r2), normal=tuple(r2), radius_mm=8.0, absorbing=True)
    main = B.observation_plane("MAIN", position=(150.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)
    return ProjectSnapshot(components=(_laser(), w1, w2), observations=(d, main), targets=(_t("d_max", "D", maximum=1e-9),))


def test_a_detector_only_a_ghost_of_a_ghost_reaches_is_not_a_known_zero():
    """It used to read PASS 0.0 (complete, no issue) while 5.68 uW arrive. Now the probe of W1's ghost makes W2's ghost in
    turn and follows it to D: UNKNOWN with the power arriving there -- the double-Fresnel product of the two windows' plate
    reflectances R + T^2 R / (1 - R^2) (independent), and the engine's own full trace of both windows'
    reflections (consistency)."""
    result = run(_two_windows(traced=False))
    (target,) = result.targets
    assert (target.status, target.value, target.issue_codes) == (Status.UNKNOWN, None, ("UNTRACED_LIGHT_BOUNDED",))  # (followed reflections: R7)
    d = _obs(result, "D")
    assert d.total_power_W is None and not d.power_complete
    # independent: s / p shares of the source (plane of incidence x-y throughout: s along z)
    free = one_arrival(run(ProjectSnapshot(components=(_laser(),), observations=(_obs_plane(),))), "FREE").beam
    field = complex(*free.polarization.ex) * np.array(free.basis.e1) + complex(*free.polarization.ey) * np.array(free.basis.e2)
    s_share = abs(field[2]) ** 2 / float(np.sum(np.abs(field) ** 2))
    n = _n_bk7(633.0)
    (r1s, r1p), (r2s, r2p) = _fresnel(n, TH1), _fresnel(n, TH2)
    ghost = lambda r: r + (1.0 - r) ** 2 * r / (1.0 - r * r)  # noqa: E731  a plate's reflectance: every internal bounce
    reference = P_W * (s_share * ghost(r1s) * ghost(r2s) + (1.0 - s_share) * ghost(r1p) * ghost(r2p))
    assert reference == pytest.approx(5.68e-6, rel=5e-3)
    assert d.untraced_power_W == pytest.approx(reference, rel=1e-6)
    traced = run(_two_windows(traced=True))
    assert _obs(traced, "D").total_power_W == pytest.approx(d.untraced_power_W, rel=1e-6)
    assert result.model_dump_json() == run(_two_windows(traced=False)).model_dump_json()  # deterministic


def _obs_plane():
    return B.observation_plane("FREE", position=(40.0, 0.0, 0.0), normal=X, radius_mm=10.0, absorbing=True)


def test_ghost_cascades_the_probe_budget_cannot_follow_leave_dark_planes_unknown_with_a_budget_issue():
    """Two uncoated singlets in front of crossed polarizers (no light passes them: Malus). With the default budget the
    cascade of ghosts is followed to every order above the negligible level and the detector is a known zero; with 50 trace
    steps it is not: the detector is UNKNOWN and a MAX_PATHS issue names the untraced light not followed."""
    comps = [_laser(w0=1.0)]
    x = 20.0
    for i in range(2):
        comps.append(B.singlet_lens(f"L{i}", r1_mm=60.0, r2_mm=math.inf, thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7",
                                    position=(x, 0.0, 0.0)))
        x += 25.0
    comps += [B.polarizer("P1", position=(x, 0, 0), axis=X, angle_deg=0.0), B.polarizer("P2", position=(x + 10.0, 0, 0), axis=X, angle_deg=90.0)]
    det = B.observation_plane("DET", position=(x + 40.0, 0.0, 0.0), normal=X, radius_mm=12.0, absorbing=True)
    project = ProjectSnapshot(components=tuple(comps), observations=(det,), targets=(_t("d_max", "DET", maximum=1e-9),))
    full = run(project)
    assert_dark_bounded(full.targets[0])
    assert "MAX_PATHS" not in {i.code.value for i in full.issues}
    short = run(project, numerical=NumericalConfig(max_trace_steps=50))
    assert (short.targets[0].status, short.targets[0].value) == (Status.UNKNOWN, None)
    (budget,) = [i for i in short.issues if i.code.value == "MAX_PATHS" and i.location == "untraced_light"]
    assert budget.status is Status.NOT_APPLICABLE and budget.numerical_context["untraced_power_W"] > 0.0
    assert short.outcome.execution.outcome.value == "COMPLETED"  # (the probes' budget, not the primary trace's)
    assert budget.numerical_context["budget"] == ["max_trace_steps"]
    assert budget.numerical_context["phase"] == "untraced_light"
    assert budget.numerical_context["affected_source_ids"] and budget.numerical_context["affected_path_ids"]


# ------------------------------------------------------------------------------------------------------------------- F3
def _defocused(radius: float, *targets: Target) -> ProjectSnapshot:
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=10.0, position=(20.0, 0.0, 0.0))
    det = B.observation_plane("OUT", position=(70.0, 0.0, 0.0), normal=X, radius_mm=radius, absorbing=True)
    return ProjectSnapshot(components=(_laser(w0=1.0), _stop("STOP", 10.0, 4.0, 0.5), lens), observations=(det,), targets=targets)


def test_a_target_on_an_annulus_counted_within_the_capture_tolerance_is_judged_on_its_interval():
    """Obstruction 0.5 mm, f 100 mm lens, 1.0 mm detector halfway to the focus: the annulus is counted whole (0.6065 mW)
    but up to 5.5e-4 of it may miss the area; the true geometric value 0.6062 mW (annulus minus the Gaussian tail beyond
    1 mm) lies in [detected_power_min_W, detected_power_W]. A minimum or maximum inside that interval is UNKNOWN with
    INCOMPLETE_ENVELOPE (it used to PASS / FAIL on the upper end); limits outside it are decided, with the code and a
    WARNING."""
    lam = 633.0
    q = 1.0 / (1.0 / complex(20.0, rayleigh_range(1.0, lam)) - 1.0 / 100.0) + 50.0
    w = math.sqrt(-lam * NM / (math.pi * (1.0 / q).imag))
    annulus = P_W * _annulus(0.5, 4.0, _w(1.0, lam, 10.0))
    true = annulus - P_W * math.exp(-2.0 * 1.0 / w**2)
    midway = 0.5 * (true + annulus)
    result = run(_defocused(1.0, _t("min_mid", "OUT", minimum=midway), _t("max_mid", "OUT", maximum=midway),
                            _t("min_low", "OUT", minimum=0.5 * true), _t("max_high", "OUT", maximum=2.0 * annulus),
                            _t("max_low", "OUT", maximum=0.5 * true)))
    arrival = one_arrival(result, "OUT")
    assert arrival.detected_power_W == pytest.approx(annulus, rel=1e-9)
    # (the lower end is the annulus minus that tail: equal to the reference up to rounding of the last bit)
    assert arrival.detected_power_min_W <= true * (1.0 + 1e-12) and true <= arrival.detected_power_W
    t = {x.target_id: x for x in result.targets}
    for target_id in ("min_mid", "max_mid"):
        assert (t[target_id].status, t[target_id].issue_codes) == (Status.UNKNOWN, ("INCOMPLETE_ENVELOPE",))
    for target_id in ("min_low", "max_high"):
        assert (t[target_id].status, t[target_id].issue_codes) == (Status.WARNING, ("INCOMPLETE_ENVELOPE",))
    assert (t["max_low"].status, t["max_low"].issue_codes) == (Status.FAIL, ("INCOMPLETE_ENVELOPE", "TARGET_NOT_MET"))
    exact = run(_defocused(4.0, _t("p", "OUT", minimum=0.0)))  # whole envelope on it: exact, no bound
    assert one_arrival(exact, "OUT").detected_power_min_W is None and exact.targets[0].issue_codes == ()


def test_the_capture_level_that_decides_an_annulus_is_named_where_it_leaves_light_out():
    """The same annulus. At the default capture_min_power_fraction (1e-3) it is counted whole, the up to 5.5e-4 beside the
    area left out: the detected power's and the target's entries name that level as the setting of their
    BEAM_PART_ELSEWHERE omission, as the recording rule of the caller settings requires of a level that decides what is
    left out (a key of the run's numerical_tolerances). At 1e-5 the same share is more than the level allows: the annulus
    is not followed, and its ANNULAR_BEAM_NOT_FOLLOWED issue names the level, its value and the share beside the area.
    The guides state that rule, not that every non-default setting is named."""
    from pathlib import Path

    from os_optics.config import NumericalConfig as Config
    from os_optics.results import basis_of

    default = run(_defocused(1.0, _t("p", "OUT", minimum=1e-4)))
    arrival = one_arrival(default, "OUT")
    (part,) = basis_of(arrival, "detected_power_W").omitted
    assert part.part.value == "BEAM_PART_ELSEWHERE" and part.setting == "capture_min_power_fraction"
    assert part.upper == pytest.approx(arrival.detected_power_W - arrival.detected_power_min_W, rel=1e-12)
    (graded,) = default.targets
    assert [o.setting for o in graded.value_basis["value"].omitted] == ["capture_min_power_fraction"]
    assert "capture_min_power_fraction" in Config().model_dump()
    strict = run(_defocused(1.0, _t("p", "OUT", minimum=1e-4)), numerical=Config(capture_min_power_fraction=1e-5))
    (issue,) = [i for i in strict.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    context = issue.numerical_context
    assert context["setting"] == "capture_min_power_fraction" and context["capture_min_power_fraction"] == 1e-5
    assert 1e-5 < context["annulus_power_beside_bound"] <= 1e-3
    assert "more than capture_min_power_fraction (1e-05)" in issue.message


# ------------------------------------------------------------------------------------------------------------------- F4
@pytest.mark.parametrize(("lens_radius", "whole"), [(12.7, True), (12.3, True), (11.5, False)])
def test_a_decentred_annulus_is_bounded_by_where_its_opening_lies(lens_radius, whole):
    """w0 4 mm, chief ray 0.6 mm off the axis of a 3 mm obstruction in a 12 mm opening, a centred f 100 mm thin lens and a
    0.5 mm detector at the focus. The opening (centred on the axis) lies inside a 12.3 or 12.7 mm lens, so the annulus passes
    whole and the detector reads its exact power (the off-centre Gaussian between radii 3 and 12, the disc integral); an
    11.5 mm lens cuts it: not followed. (The bound used to grow by the chief ray's offset: 'not followed' at 12.7 mm.)"""
    lens = B.thin_lens("LENS", focal_length_mm=100.0, clear_radius_mm=lens_radius, position=(20.0, 0.0, 0.0))
    det = B.observation_plane("OUT", position=(120.0, 0.0, 0.0), normal=X, radius_mm=0.5, absorbing=True)
    result = run(ProjectSnapshot(components=(_laser(w0=4.0, pos=(0.0, 0.6, 0.0)), _stop("STOP", 10.0, 12.0, 3.0), lens),
                                 observations=(det,), targets=(_t("p", "OUT", minimum=0.0),)))
    w = _w(4.0, 633.0, 10.0)
    reference = P_W * (_disc(12.0, 0.6, w) - _disc(3.0, 0.6, w))
    assert reference == pytest.approx(0.34092667057e-3, rel=1e-9)
    if whole:
        assert result.targets[0].value == pytest.approx(reference, rel=1e-9)
    else:  # (the power behind the lens is not computed, at most the annulus's; the rays that pass go on)
        _obs, arrival = assert_power_not_computed(result, "OUT", state=None)
        (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
        assert issue.location == "surface:LENS/lens" and arrival.power_upper_bound_W == pytest.approx(reference, rel=1e-9)
