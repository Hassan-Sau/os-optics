"""The fringe visibility of two mutually incoherent beams is 0, and a visibility target on them is decided.

Two beams of different source components are mutually incoherent (README "Interference"): the cross term of their fields
averages to zero over the detection time, the irradiance on the plane is the sum of the two beams' irradiances, and the
time-averaged visibility ``(I_max - I_min) / (I_max + I_min)`` of their fringe is 0 -- the Engine's own formula with a
coherence factor of 0. A minimum above 0 is therefore not met and a maximum is met; light of the same wavelength the
trace does not follow may still form a fringe with one of the beams and bounds the visibility from above
(``metrics._visibility_light``: ``(V + 2 rho) / (1 - rho)^2`` with ``V = 0``).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status

from helpers import run

DIAG = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)


def _crossing(*targets: Target, window: bool = False) -> ProjectSnapshot:
    a = B.laser("A", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    b = B.laser("B", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=1.0, position=(0.0, -100.0, 0.0), direction=(0.0, 1.0, 0.0))
    comps = [a, b]
    if window:  # (its forward ghost travels with A's beam: light the trace does not follow, coherent with A)
        comps.append(B.window("W", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(-50.0, 0.0, 0.0)))
    plane = B.observation_plane("X", position=(0.0, 0.0, 0.0), normal=DIAG, radius_mm=12.5)
    return ProjectSnapshot(components=tuple(comps), observations=(plane,), targets=targets)


def _visibility(target_id: str, **limits) -> Target:
    return Target(target_id=target_id, metric="fringe_visibility", observation_id="X", source_id="A", source_id_b="B", unit="1",
                  **limits)


def test_two_lasers_have_no_fringe_so_a_visibility_minimum_is_not_met():
    result = run(_crossing(_visibility("vis_min", minimum=0.5), _visibility("vis_max", maximum=0.1)))
    t = {r.target_id: r for r in result.targets}
    assert (t["vis_min"].status, t["vis_min"].value) == (Status.FAIL, 0.0)
    assert t["vis_min"].assessment.value == "UNMET" and "INCOHERENT_SOURCES" in t["vis_min"].issue_codes
    assert (t["vis_max"].status, t["vis_max"].value) == (Status.PASS, 0.0)
    assert "mutually incoherent" in t["vis_max"].message
    # (the pair's own record is unchanged: no fringe, the incoherent sum of the two beams)
    (pair,) = result.interference
    assert pair.status is Status.NOT_APPLICABLE and pair.fringe_visibility is None
    assert pair.incoherent_power_sum_W == pytest.approx(2e-3, rel=1e-12)
    assert result.outcome.requirements.unmet == ("vis_min",) and result.outcome.requirements.met == ("vis_max",)


def test_ghost_light_of_one_laser_bounds_the_visibility_of_the_two_from_above():
    """A window in A's beam: its forward ghost (T^2 R^2 of A's power at normal incidence) may form a fringe with A. With
    ``R`` the summed field amplitude of that light and ``P = P_a + P_b`` the visibility lies in ``[0, 2 rho / (1 - rho)^2]``,
    ``rho = R / sqrt(P)``: a minimum of 0.5 is still not met; a maximum below that upper end is not decided."""
    l2 = 0.780 ** 2
    b, c = (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)
    n = math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))
    r = ((n - 1.0) / (n + 1.0)) ** 2
    ghost_root = math.sqrt((1.0 - r) ** 2 * r * r * 1e-3)  # the first forward ghost's field amplitude (sqrt(W))
    result = run(_crossing(_visibility("vis_min", minimum=0.5), _visibility("vis_tight", maximum=1e-6), window=True))
    t = {r.target_id: r for r in result.targets}
    assert (t["vis_min"].status, t["vis_min"].value) == (Status.FAIL, 0.0)
    assert "mutually incoherent" in t["vis_min"].message and "not met even at the upper end" in t["vis_min"].message
    assert t["vis_tight"].status is Status.UNKNOWN and t["vis_tight"].value == 0.0
    power = result.interference[0].incoherent_power_sum_W
    rho = ghost_root / math.sqrt(power)
    assert 2.0 * rho / (1.0 - rho) ** 2 < 0.5  # (the bound itself is far below the minimum)
