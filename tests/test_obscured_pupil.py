"""An obscured pupil: a classical Cassegrain telescope whose secondary mirror sits inside the incoming beam and whose primary
mirror has a central hole. The secondary's back is a central obstruction: the beam beside it goes on as an
annulus with its exact Gaussian power, reflects off the primary around the hole (the reference axis takes the primary's
continued profile there, the hole lying inside the secondary's shadow), off the secondary's front, and through the hole
to the focus.

Independent references, computed here: the annulus power ``exp(-2 a^2 / w^2)`` of a centred Gaussian beam of radius ``w``
beside a disc of radius ``a``; the Cassegrain geometry by the mirror equation (primary focal length ``f1``, separation
``d``, image ``b`` behind the primary: ``1/f2 = 1/(d + b) - 1/(f1 - d)``, secondary magnification ``m = (d + b)/(f1 - d)``)
and the classical Cassegrain's conics (paraboloid; hyperboloid ``k2 = -((m + 1)/(m - 1))^2``), which image an axial point
perfectly: the annulus's rays meet at the image point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import CircularAperture, EngineeringComponentSnapshot, MirrorInteraction, ProjectSnapshot, SagSurface

from helpers import issue_codes, one_arrival, run

X = (1.0, 0.0, 0.0)
P_W, LAM, W = 1e-3, 633.0, 15.0
F1, D, B_MM = 400.0, 300.0, 100.0
E = D + B_MM
M = E / (F1 - D)
F2 = 1.0 / (1.0 / E - 1.0 / (F1 - D))
K2 = -((M + 1.0) / (M - 1.0)) ** 2
SECONDARY_R, HOLE_R = 13.0, 10.0


def _mirror(instance_id: str, x: float, curvature: float, conic: float, radius: float, normal_z: float, hole: float | None = None):
    surface = SagSurface(surface_id="m", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, normal_z), u_axis=(1.0, 0.0, 0.0),
                         curvature_per_mm=curvature, conic=conic, aperture=CircularAperture(radius_mm=radius),
                         holes=None if hole is None else (CircularAperture(radius_mm=hole),),
                         interaction=MirrorInteraction(reflective_side="positive"))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="mirror", pose=B._pose((x, 0.0, 0.0), X), surfaces=(surface,))


def telescope(hole: float = HOLE_R, secondary: float = SECONDARY_R, y0: float = 0.0, z0: float = 0.0) -> ProjectSnapshot:
    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W, position=(-200.0, y0, z0), direction=X)
    # secondary at x = 0, convex toward the primary (its centre of curvature behind it, on -x); reflective side +x
    sec = _mirror("SEC", 0.0, 1.0 / (2.0 * F2), K2, secondary, 1.0)
    # primary at x = D, concave toward the secondary (its centre on -x, its reflective side), paraboloid with a central hole
    prim = _mirror("PRIM", D, 1.0 / (2.0 * F1), -1.0, 55.0, -1.0, hole)
    image = B.observation_plane("IMG", position=(D + B_MM, 0.0, 0.0), normal=X, radius_mm=20.0)
    return ProjectSnapshot(components=(laser, sec, prim), observations=(image,))


def test_the_cassegrain_reference_geometry() -> None:
    assert F2 == pytest.approx(-400.0 / 3.0) and M == pytest.approx(4.0)
    assert F1 * M == pytest.approx(1600.0)  # effective focal length


def test_the_annulus_beside_the_secondary_reaches_the_focus_through_the_primary_hole() -> None:
    result = run(telescope())
    arrival = one_arrival(result, "IMG")
    expected = P_W * math.exp(-2.0 * SECONDARY_R ** 2 / (W * W))
    assert arrival.beam.power_W == pytest.approx(expected, rel=1e-6)
    assert any(step.endswith(":obstruction") for step in arrival.lineage)
    assert "GAUSSIAN_CENTRAL_OBSTRUCTION" in issue_codes(result)
    assert "ANNULAR_BEAM_NOT_FOLLOWED" not in issue_codes(result)
    # a classical Cassegrain images the axial point perfectly: the annulus's rays meet on the axis at the image plane
    assert arrival.pupil_rays_arrived >= 6
    assert arrival.pupil_rms_radius_mm < 1e-6
    assert result.power_budget.conserved


def test_a_hole_wider_than_the_shadow_is_not_followed() -> None:
    """A 14 mm hole behind a 13 mm secondary: part of the annulus would pass through the hole; the annulus is not followed."""

    result = run(telescope(hole=14.0))
    assert "ANNULAR_BEAM_NOT_FOLLOWED" in issue_codes(result)
    assert result.power_budget.conserved


def test_a_hole_between_the_shadow_and_the_samples_is_refused_by_the_shadow_check() -> None:
    """A 13.3 mm hole behind the 13 mm secondary: every sampled annulus ray (the innermost at 0.915 w = 13.7 mm) meets the
    primary around it, but the ring of the annulus between 13 and 13.3 mm would pass through the hole; the shadow check does not
    prove the hole inside the shadow, so the annulus's power is not computed behind it (never a power taken from where the
    samples land): the rays go on and every arrival behind states no power, at most the annulus's."""

    result = run(telescope(hole=13.3))
    (issue,) = [i for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert "not proven to lie inside the shadow" in issue.message
    behind = [a for o in result.observations for a in o.arrivals]
    assert behind and all(a.beam.power_W is None and a.power_upper_bound_W == issue.numerical_context["power_upper_bound_W"] for a in behind)
    assert result.power_budget.conserved


def test_a_secondary_wider_than_the_beam_blocks_it() -> None:
    """A secondary much wider than the beam: no light passes beside it (negligible), the beam is blocked by its back."""

    result = run(telescope(secondary=54.0, hole=10.0))
    assert "BLOCKED_BY_BACK_SIDE" in issue_codes(result)
    assert not [a for o in result.observations for a in o.arrivals]


def disc(radius: float, offset: float, w: float, nodes: int = 600) -> float:
    """The share of a round Gaussian beam (1/e^2 radius ``w``) on a disc of ``radius`` whose centre lies ``offset`` from the
    beam's: ``int_0^a (4 r / w^2) exp(-2 (r^2 + d^2) / w^2) I0(4 r d / w^2) dr`` by Gauss-Legendre in r (NumPy's I0)."""

    x, weights = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * radius * (x + 1.0)
    f = (4.0 * r / (w * w)) * np.exp(-2.0 * (r * r + offset * offset) / (w * w)) * np.i0(4.0 * r * offset / (w * w))
    return float(0.5 * radius * np.sum(weights * f))


@pytest.mark.parametrize("y0, z0", [(1.5, 0.0), (2.0, 0.0), (5.0, 0.0), (3.0, 4.0)])
def test_a_decentred_beam_on_the_curved_secondary_goes_on_as_its_annulus(y0: float, z0: float) -> None:
    """The chief ray on the secondary's back 1.5-5 mm off its axis: pupil rays on the secondary's support nearer the axis meet
    its convex back beyond the plane across the chief ray at its hit; they meet the element and leave the beam there (before,
    they were carried on and the annulus was refused at the next step). Power: the exact annulus ``P (1 - disc(13, r0, 15))``."""

    result = run(telescope(y0=y0, z0=z0))
    assert "ANNULAR_BEAM_NOT_FOLLOWED" not in issue_codes(result)
    arrival = one_arrival(result, "IMG")
    assert arrival.beam.power_W == pytest.approx(P_W * (1.0 - disc(SECONDARY_R, math.hypot(y0, z0), W)), rel=1e-6)
    assert result.power_budget.conserved


def test_an_annulus_whose_envelope_reaches_past_the_next_clear_aperture_is_still_refused() -> None:
    """9 mm off the axis the unobstructed beam's envelope reaches past the primary's clear aperture (by a 1e-9 share): the
    annulus is not proven to pass the apertures whole, so it is not followed (honest refusal, ledger closed)."""

    result = run(telescope(y0=9.0))
    messages = [i.message for i in result.issues if i.code.value == "ANNULAR_BEAM_NOT_FOLLOWED"]
    assert messages and "does not contain the whole envelope of the unobstructed beam" in messages[0]
    assert result.power_budget.conserved


@pytest.mark.parametrize("y0", [14.0, 16.0, 20.0])
def test_a_beam_beside_the_secondary_is_out_of_scope_behind_its_back(y0: float) -> None:
    """The chief ray passes beside the 13 mm secondary while part of the beam meets its back: the light that would reach
    the primary there (outside its 10 mm hole) is still counted by the beam, whose power follows the chief ray. The image's
    power is therefore out of model scope behind the secondary (never a known value), and the issue states the Gaussian
    share on the secondary's support. Physically the image gets ``P (1 - disc(13, y0, 15))``, between the power stated
    less that share and the power stated. The share is bounded, but the secondary is a mirror: what it turns may reach a
    plane behind, so ``[P - S, P]`` would not hold (only a surface that stops the light it meets is a share taken on the
    way)."""

    from os_optics.validation.outcome import quantity_state_of

    result = run(telescope(y0=y0))
    arrival = one_arrival(result, "IMG")
    assert arrival.power_out_of_scope_at == ("SEC/m",)
    assert quantity_state_of(result.observations[0]).value == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in result.issues if i.code.value == "OUT_OF_MODEL_SCOPE" and i.numerical_context.get("next_surface_id") == "PRIM/m"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location == "surface:SEC/m"
    context = issue.numerical_context
    assert context["support_fraction"] == pytest.approx(disc(SECONDARY_R, y0, W), rel=1e-7)
    physical = P_W * (1.0 - disc(SECONDARY_R, y0, W))
    reported = arrival.beam.power_W
    # (as before, the hole's share is the only large loss counted: the rest is the primary's rim and the secondary's edge)
    assert P_W * (1.0 - disc(HOLE_R, y0, W)) * (1.0 - 1e-4) < reported <= P_W * (1.0 - disc(HOLE_R, y0, W))
    assert reported - context["support_power_W"] <= physical < reported
    # the equal-power samples' estimate of the share counted twice (on the secondary's back and inside the primary's aperture)
    assert context["met_fraction"] == pytest.approx(disc(SECONDARY_R, y0, W) - disc(HOLE_R, y0, W), abs=0.04)
    assert result.power_budget.conserved
