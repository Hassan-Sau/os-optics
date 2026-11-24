"""A plane the chief ray crosses while a surface takes a bounded share of the beam before it: an interval, not a refusal.

The plane reads the beam (power ``P``); the share the surfaces take before it is at most ``S``
(``ArrivalResult.power_taken_before_upper_W``: the larger of the parallel-ray and the paraxial energy-flow bounds), so the
crossing power lies in ``[P - S, P]``. The plane is ``BOUNDED`` (``power_lower_bound_W``), a ``power_W`` target is judged on
that interval (``POWER_TAKEN_BEFORE_PLANE``), and the other power-derived metrics stay out of model scope. The reference is
an independent integration of the paraxial Gaussian beam's energy-flow lines (each at a fixed place ``xi`` in the profile,
``xi w(s)`` off the chief ray): the lines that meet a tilted flat mirror before the plane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import run
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local.record import scientific_projection

LAM = 780.0
SOURCE_W = 1e-3
LAUNCH_MM = 30.0  # the source lies this far before the mirror's hit point


def _fold(gap: float, tilt_deg: float, w0: float, waist_mm: float, targets: tuple[Target, ...] = ()) -> ProjectSnapshot:
    """A 1 mW beam along +X onto a flat mirror at the origin whose normal is tilted ``tilt_deg`` from -X toward +Y; the plane
    T (normal +X) ``gap`` mm before the hit point. ``waist_mm``: the waist's place after the source."""
    tilt = math.radians(tilt_deg)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-math.cos(tilt), math.sin(tilt), 0.0), reflectance=0.99, clear_radius_mm=20.0)
    laser = B.laser("L", wavelength_nm=LAM, waist_radius_mm=w0, power_W=SOURCE_W, position=(-LAUNCH_MM, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                    waist_position_z_mm=waist_mm)
    t = B.observation_plane("T", position=(-gap, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0)
    return ProjectSnapshot(components=(laser, mirror), observations=(t,), targets=targets)


def _incoming(result):
    """T's arrival of the incoming beam (the reflected one may cross T too)."""
    (obs,) = result.observations
    return next(a for a in obs.arrivals if a.crossing_index == 0 and a.parent_path_id is None)


def _flow_share(gap: float, tilt_deg: float, w0: float, waist_mm: float) -> float:
    """The fraction of the beam whose energy-flow lines meet the mirror's plane before T. A line at ``xi`` (density
    ``exp(-2 xi^2)``) is ``xi w(s)`` across the chief ray along +Y at ``s`` from the source; the mirror's plane is
    ``x = y tan(tilt)``, so the line meets it before T when ``xi tan(tilt) w(s) <= s - LAUNCH_MM`` for some ``s`` up to T: when
    ``xi <= c``, ``c`` the largest ``(s - LAUNCH_MM) / (w(s) tan(tilt))`` there."""
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    s = np.linspace(0.0, LAUNCH_MM - gap, 400_001)
    w = w0 * np.sqrt(1.0 + ((s - waist_mm) / z_r) ** 2)
    c = float(np.max((s - LAUNCH_MM) / (w * math.tan(math.radians(tilt_deg)))))
    return 0.5 * math.erfc(math.sqrt(2.0) * -c) if c < 0.0 else 1.0


@pytest.mark.parametrize("gap", (0.2, 0.5, 1.0))
def test_a_plane_before_a_fold_reads_the_beam_within_the_share_the_mirror_takes_first(gap: float):
    """A collimated 0.5 mm beam, a 45 deg fold: the mirror takes 0.5 erfc(sqrt(2) gap / w) of it before T. That share
    is within the reported bound, the plane is BOUNDED and its lower end is the beam's power less the bound."""
    result = run(_fold(gap, 45.0, 0.5, LAUNCH_MM))
    arrival = _incoming(result)
    share = 0.5 * math.erfc(math.sqrt(2.0) * gap / 0.5)
    assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("M/face",)
    upper = arrival.power_taken_before_upper_W
    assert share * SOURCE_W <= upper * (1.0 + 1e-9)
    assert upper <= 1.01 * share * SOURCE_W + 1e-15  # (collimated: the bound is the half-plane share itself)
    (obs,) = result.observations
    assert obs.quantity_state.value == "BOUNDED"
    assert obs.power_lower_bound_W == pytest.approx(obs.total_power_W - upper, rel=1e-12)


def test_the_bound_holds_the_energy_flow_share_of_diverging_and_focused_beams():
    """60 poses: gaps 0.01-3 mm, tilts 10-80 deg, waists 20 um-1 mm anywhere from 20 mm before the source to 10 mm past
    the mirror. Wherever the plane reads the beam with a bound, the energy-flow lines that meet the mirror first carry
    no more than it; nowhere is a share above the line's negligible level read as nothing."""
    rng = np.random.default_rng(374)
    bounded = 0
    for _ in range(60):
        gap, tilt = float(rng.uniform(0.01, 3.0)), float(rng.uniform(10.0, 80.0))
        w0, waist = float(np.exp(rng.uniform(math.log(0.02), math.log(1.0)))), float(rng.uniform(-20.0, 40.0))
        arrival = _incoming(run(_fold(gap, tilt, w0, waist)))
        truth = _flow_share(gap, tilt, w0, waist) * SOURCE_W
        if arrival.power_taken_before_upper_W is not None:
            bounded += 1
            assert truth <= arrival.power_taken_before_upper_W * (1.0 + 1e-9) + 1e-18, (gap, tilt, w0, waist)
        elif not arrival.power_out_of_scope_at:
            assert truth <= 1e-9 * SOURCE_W, (gap, tilt, w0, waist)
    assert bounded >= 10


@pytest.mark.parametrize("selected", (True, False))
@pytest.mark.parametrize(("minimum", "assessment"), ((0.9e-3, "MET"), (0.99999e-3, "INDETERMINATE"), (1.001e-3, "UNMET")))
def test_a_power_target_is_judged_on_the_interval(minimum: float, assessment: str, selected: bool):
    """1 mm before the 45 deg fold of a 0.5 mm beam the mirror takes at most 3.2e-5 of it first: a minimum below the lower
    end is met, one within the interval is not decided, one above the beam's power is unmet. The value is the beam's
    power, the upper end of that interval: its entry says so -- an upper bound with the share M/face may take
    (``BEAM_PART_ELSEWHERE``, the arrival's own bound) for the incoming crossing; for the whole plane also the light the
    trace did not follow there, which may add (an estimate holding both)."""
    target = Target(target_id="t", metric="power_W", observation_id="T", minimum=minimum, unit="W",
                    parameters={"crossing_index": 0} if selected else {})
    result = run(_fold(1.0, 45.0, 0.5, LAUNCH_MM, targets=(target,)))
    (got,) = result.targets
    assert got.assessment.value == assessment
    assert "POWER_TAKEN_BEFORE_PLANE" in got.issue_codes
    assert "M/face takes at most" in got.message
    entry = got.value_basis["value"]
    taken = [o for o in entry.omitted if o.part.value == "BEAM_PART_ELSEWHERE"]
    assert [(o.at, o.upper) for o in taken] == [(("M/face",), _incoming(result).power_taken_before_upper_W)]
    if selected:
        assert entry.kind.value == "UPPER_BOUND" and entry.omitted == tuple(taken)
    else:
        assert entry.kind.value == "ESTIMATE" and {o.part.value for o in entry.omitted} == {"BEAM_PART_ELSEWHERE", "NOT_FOLLOWED"}


def test_a_taken_share_keeps_the_light_left_out_as_negligible_in_its_interval(monkeypatch):
    """A power of 1 mW of which M/face may take 3.2e-8 W before the plane (an upper bound, graded on [P - S, P]), on a plane
    whose only uncounted light is left out as negligible (at most 5e-10 W, within the level): its entry is the upper bound
    with that truncation, and since the grading reads no left-out bound from an entry that is not the model's value, the
    interval itself rises by that light. A maximum 2e-10 W above the beam's power is not decided; one above both is met."""
    from os_optics.config import NumericalConfig  # noqa: PLC0415
    from os_optics.contracts.basis import Omitted, ValueBasis  # noqa: PLC0415
    from os_optics.errors.status import OmittedPart, ValueKind  # noqa: PLC0415
    from os_optics.validation import metrics as M  # noqa: PLC0415
    from os_optics.validation.power import LineFate  # noqa: PLC0415

    power, share, faint = 1e-3, 3.2e-8, 5e-10
    upper = ValueBasis(kind=ValueKind.UPPER_BOUND, omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=("M/face",), upper=share),))
    mv = M.MetricValue(power, "W", untraced=(power - share, power), basis=upper)
    fate = LineFate(input_W=1e-3, unknown_W={}, below_floor_W=0.0, floor_significant=False, faint_W=faint, level_W=1e-12 * 1e3)
    assert fate.complete and fate.faint_accounted
    out = M._untraced_interval(mv, fate, None, coherent=False)
    assert out.untraced == (power - share, power + faint)
    assert out.basis.kind is ValueKind.UPPER_BOUND
    assert {o.part for o in out.basis.omitted} == {OmittedPart.BEAM_PART_ELSEWHERE, OmittedPart.BELOW_NEGLIGIBLE_LEVEL}

    def graded(maximum: float):
        target = Target(target_id="t", metric="power_W", observation_id="T", maximum=maximum, unit="W")
        monkeypatch.setattr(M, "_measure", lambda *args, **kwargs: out)
        return M._evaluate(target, {}, [], None, None, NumericalConfig())

    inside = graded(power + 2e-10)
    assert (inside.status.value, inside.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert inside.value_basis["value"] == out.basis
    assert graded(power + 1e-9).assessment.value == "MET"


def test_other_power_derived_metrics_stay_out_of_scope():
    target = Target(target_id="c", metric="coherent_power_W", observation_id="T", minimum=0.5e-3, unit="W", parameters={"crossing_index": 0})
    (got,) = run(_fold(0.2, 45.0, 0.5, LAUNCH_MM, targets=(target,))).targets
    assert got.status.value == "OUT_OF_MODEL_SCOPE" and got.value is None


def test_the_scientific_summary_states_the_bounded_plane():
    """The Local Validation record's observation quantity carries the BOUNDED state and its lower end (the mirror's
    untraced transmission may reach the plane too: the evidence is not complete, and the upper end is stated beside it)."""
    result = run(_fold(0.5, 45.0, 0.5, LAUNCH_MM))
    (quantity,) = scientific_projection(result).observation_quantities
    assert quantity.quantity_state == "BOUNDED"
    assert quantity.power_lower_bound_W == pytest.approx(result.observations[0].power_lower_bound_W, rel=0)
    assert "power_lower_bound_W" in quantity.model_dump(mode="json")


def test_a_plane_far_from_the_fold_keeps_its_exact_reading():
    """5 mm before the fold the share is below the negligible level: no bound, no interval, the field omitted."""
    result = run(_fold(5.0, 45.0, 0.5, LAUNCH_MM))
    arrival = _incoming(result)
    assert arrival.power_taken_before_upper_W is None and arrival.power_taken_before_at == ()
    assert "power_taken_before_upper_W" not in arrival.model_dump(mode="json")
    assert result.observations[0].quantity_state.value == "KNOWN"


# ------------------------------------------------------------------------------------------- a sag surface met first


def _flow_share_on_sag(radius: float, rim_x: float, support: float, height: float, w0: float, waist_mm: float, x_t: float,
                       cells: int = 240, steps: int = 8001, half: float = 5.0) -> float:
    """An upper estimate of the fraction of the beam (along +X from x = -60 at ``height``) whose energy-flow lines meet the
    concave sag mirror ``z = sag(sqrt(x^2 + y^2))`` inside its support (a disc of radius ``support`` centred at
    ``(rim_x + support, 0)``) before the plane x = ``x_t``. The line at ``(xi, eta)`` of the profile (density
    ``exp(-2 (xi^2 + eta^2))``) is ``(xi w(s), eta w(s))`` off the chief ray; it meets the surface where
    ``f(s) = height + eta w(s) - sag <= 0``. Every estimate errs toward meeting: the profile is cut into ``cells`` x ``cells``
    cells of exact Gaussian weight within ``half`` radii (the tails beyond counted as met), a cell is met when its lowest
    line (``f`` grows with ``eta``) at either edge or the middle of its ``xi`` range meets; ``s`` is stepped by ``h`` and a
    line counts as meeting where a step finds ``f <= L h / 2`` (``L`` a bound of ``|df/ds|``) within the support grown by
    the step. So the bound must hold this estimate exactly: no tolerance."""
    from scipy.special import erf  # noqa: PLC0415

    edges = np.linspace(-half, half, cells + 1)
    weight = 0.5 * (erf(math.sqrt(2.0) * edges[1:]) - erf(math.sqrt(2.0) * edges[:-1]))
    tails = 1.0 - (1.0 - math.erfc(math.sqrt(2.0) * half)) ** 2  # (beyond half radii in xi or eta)
    c = 1.0 / radius
    z_r = math.pi * w0 * w0 / (LAM * 1e-6)
    theta = w0 / z_r
    s = np.linspace(0.0, x_t + 60.0, steps)
    h = float(s[1] - s[0])
    w = w0 * np.sqrt(1.0 + ((s - waist_mm) / z_r) ** 2)
    x = s - 60.0
    rho_max = math.hypot(max(abs(rim_x), abs(rim_x + 2.0 * support)), half * float(w.max()))
    slope = c * rho_max / math.sqrt(1.0 - (c * rho_max) ** 2)  # (the sag's steepest within the support's reach)
    lipschitz = half * theta + slope * (1.0 + half * theta)
    met = np.zeros((cells, cells), dtype=bool)
    eta_low = edges[:-1]
    for xis in (edges[:-1], edges[1:], 0.5 * (edges[:-1] + edges[1:])):
        xi = xis[:, None] * w[None, :]
        rho2 = x[None, :] * x[None, :] + xi * xi
        grow = h * (1.0 + np.abs(xis)[:, None] * theta)
        inside = (np.sqrt((x[None, :] - (rim_x + support)) ** 2 + xi * xi) <= support + grow) & (c * c * rho2 < 1.0)
        sag = c * rho2 / (1.0 + np.sqrt(np.maximum(1.0 - c * c * rho2, 0.0)))
        # (met for every eta at or below the largest (sag - height + L h / 2) / w over the steps inside the support)
        reach = np.where(inside, (sag - height + 0.5 * lipschitz * h) / w[None, :], -np.inf).max(axis=1)
        met |= eta_low[None, :] <= reach[:, None]
    return float((weight[:, None] * weight[None, :] * met).sum()) + tails


@pytest.mark.parametrize(("radius", "delta_w", "w0", "waist_mm", "x_t"), (
    (39.1, 1.68, 0.117, 40.0, -12.2), (39.7, 2.51, 0.163, 10.0, -0.3), (44.3, 1.90, 0.113, 70.0, -3.5), (40.3, 0.89, 0.398, 30.0, -10.5)))
def test_a_sag_mirror_met_first_is_bounded_by_the_surface_itself(radius, delta_w, w0, waist_mm, x_t):
    """A beam skims the near rim of a concave sag mirror (an off-axis support) toward its far side; T lies between. The
    share bounded cell by cell on the surface itself holds the share of the energy-flow lines that meet it first (a plane
    standing for the surface would be before its whole support: the whole beam). The reference over-estimates the share
    (``_flow_share_on_sag``), so containment is asserted exactly."""
    from os_optics.contracts import CircularAperture, EngineeringComponentSnapshot, MirrorInteraction, Pose3D, SagSurface  # noqa: PLC0415

    support, rim_x = 0.45 * radius, -0.4 * radius
    height = radius - math.sqrt(radius * radius - rim_x * rim_x) + delta_w * w0
    face = SagSurface(surface_id="S1", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), curvature_per_mm=1.0 / radius,
                      aperture=CircularAperture(radius_mm=support, center_offset_mm=(rim_x + support, 0.0)),
                      interaction=MirrorInteraction(reflective_side="positive"))
    mirror = EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D(), surfaces=(face,))
    laser = B.laser("S", wavelength_nm=LAM, power_W=SOURCE_W, waist_radius_mm=w0, position=(-60.0, 0.0, height), direction=(1.0, 0.0, 0.0),
                    waist_position_z_mm=waist_mm)
    t = B.observation_plane("T", position=(x_t, 0.0, height), normal=(1.0, 0.0, 0.0), radius_mm=20.0)
    arrival = _incoming(run(ProjectSnapshot(components=(laser, mirror), observations=(t,))))
    truth = _flow_share_on_sag(radius, rim_x, support, height, w0, waist_mm, x_t) * SOURCE_W
    assert arrival.power_taken_before_at == ("M/S1",)
    assert truth <= arrival.power_taken_before_upper_W  # (an upper estimate of the true share: contained, no slack)
    assert arrival.power_taken_before_upper_W < 0.1 * SOURCE_W  # (not the whole beam)
