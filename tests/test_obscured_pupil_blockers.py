"""Obscured pupils by an absorber and by the opaque plate of an aperture stop inside the beam: as for a mirror's
back (``test_obscured_pupil.py``), the share of the Gaussian beam on the element's support is stopped there and the rest
goes on as an annulus with its exact power, its pupil rays beside the element carried on as its geometric rays.

Independent reference, computed here: the power of a Gaussian beam of radius ``w`` (1/e^2) on a disc of radius ``a`` whose
centre lies ``r0`` from the beam's axis,
``disc(a, r0, w) = int_0^a (4 r / w^2) exp(-2 (r^2 + r0^2) / w^2) I0(4 r r0 / w^2) dr`` (SciPy ``quad`` with ``i0e``), and its
centred value ``1 - exp(-2 a^2 / w^2)``.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad
from scipy.special import i0e

from os_optics import builders as B
from os_optics.contracts import AbsorberInteraction, CircularAperture, EngineeringComponentSnapshot, PlaneSurface, ProjectSnapshot
from os_optics.contracts.optics import SurfaceScatter

from helpers import issue_codes, one_arrival, run

X = (1.0, 0.0, 0.0)
P_W, LAM, W = 1e-3, 633.0, 15.0


def disc(a: float, r0: float, w: float) -> float:
    """The share of a Gaussian beam (radius ``w``) on a disc of radius ``a`` centred ``r0`` from its axis."""

    def f(r: float) -> float:
        x = 4.0 * r * r0 / (w * w)
        return 4.0 * r / (w * w) * math.exp(-2.0 * (r - r0) ** 2 / (w * w)) * i0e(x)

    return quad(f, 0.0, a, epsabs=1e-15, epsrel=1e-13, limit=200)[0]


def _scene(blocker: EngineeringComponentSnapshot, y0: float = 0.0) -> ProjectSnapshot:
    """A beam of waist 15 mm at x = 0 (where the blocker is) along +X, ``y0`` off the blocker's axis; a plane 100 mm behind."""

    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W, position=(-200.0, y0, 0.0), direction=X,
                    waist_position_z_mm=200.0)
    plane = B.observation_plane("OUT", position=(100.0, 0.0, 0.0), normal=X, radius_mm=60.0)
    return ProjectSnapshot(components=(laser, blocker), observations=(plane,))


def _sink(result, kind: str) -> float:
    return sum(line.sinks.get(kind, 0.0) for line in result.power_budget.lines)


@pytest.mark.parametrize("y0", [0.0, 2.0, 3.5])
def test_the_annulus_beside_an_absorber_carries_the_exact_gaussian_power(y0: float) -> None:
    """A beam dump of radius 5 mm in a 15 mm beam: the plane behind it receives ``1 - disc(5, y0, 15)`` of the beam, the dump
    absorbs ``disc``; the lineage names the obstruction and the Gaussian metrics are withheld (an annulus)."""

    result = run(_scene(B.beam_dump("DUMP", position=(0.0, 0.0, 0.0), axis=X, radius_mm=5.0), y0))
    arrival = one_arrival(result, "OUT")
    expected = 1.0 - disc(5.0, y0, W)
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-7)
    assert _sink(result, "ABSORBED") == pytest.approx(P_W * (1.0 - expected), rel=1e-6)
    assert any(step == "DUMP/dump:obstruction" for step in arrival.lineage)
    assert "GAUSSIAN_CENTRAL_OBSTRUCTION" in issue_codes(result)
    assert arrival.pupil_rays_arrived > 0
    assert result.power_budget.conserved
    if y0 == 0.0:
        assert expected == pytest.approx(math.exp(-2.0 * 25.0 / (W * W)), rel=1e-12)


def test_the_opaque_plate_of_a_stop_beside_its_opening_obstructs_the_beam() -> None:
    """An aperture stop's plate (outline radius 6 mm) with a 0.1 um opening 4 mm from its centre: the chief ray 1 mm from the
    plate's centre falls on the plate, the opening passes a negligible share (about 2 (1e-4)^2 / 15^2, below the ledger tolerance 1e-9 of the line), so the plate is a
    central obstruction: ``1 - disc(6, 1, 15)`` passes beside it, the plate's share is booked ``CLIPPED``."""

    plate = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                         aperture=CircularAperture(radius_mm=1e-4, center_offset_mm=(4.0, 0.0)),
                         mount_outline=CircularAperture(radius_mm=6.0), medium_positive="air", medium_negative="air",
                         interaction={"kind": "aperture_stop"})
    stop = EngineeringComponentSnapshot(instance_id="STOP", role="aperture", pose=B._pose((0.0, 0.0, 0.0), X), surfaces=(plate,))
    result = run(_scene(stop, 1.0))
    arrival = one_arrival(result, "OUT")
    expected = 1.0 - disc(6.0, 1.0, W)
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-6)
    assert any(step.endswith(":obstruction") for step in arrival.lineage)
    assert _sink(result, "CLIPPED") == pytest.approx(P_W * (1.0 - expected), rel=1e-5)
    assert result.power_budget.conserved


def test_a_stop_whose_opening_passes_light_keeps_its_own_rule() -> None:
    """A 1 mm opening at the plate's centre and the chief ray 2 mm off it, on the plate: the opening passes a non-negligible
    share, so the plate is not taken as an obstruction (no ``obstruction`` step)."""

    plate = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=1.0),
                         mount_outline=CircularAperture(radius_mm=6.0), medium_positive="air", medium_negative="air",
                         interaction={"kind": "aperture_stop"})
    stop = EngineeringComponentSnapshot(instance_id="STOP", role="aperture", pose=B._pose((0.0, 0.0, 0.0), X), surfaces=(plate,))
    result = run(_scene(stop, 2.0))
    lineages = [a.lineage for o in result.observations for a in o.arrivals]
    assert not any(step.endswith(":obstruction") for lineage in lineages for step in lineage)
    assert result.power_budget.conserved


def test_a_scattering_absorber_keeps_its_own_rule() -> None:
    """A beam dump that scatters back part of the light: its scattered light is followed by its own model, so it is not
    taken as a central obstruction."""

    surf = PlaneSurface(surface_id="dump", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=5.0),
                        medium_positive="air", medium_negative="air",
                        interaction=AbsorberInteraction(scatter=SurfaceScatter(fraction=0.01, side="reflect", sample_count=16)))
    dump = EngineeringComponentSnapshot(instance_id="DUMP", role="dump", pose=B._pose((0.0, 0.0, 0.0), X), surfaces=(surf,))
    result = run(_scene(dump))
    lineages = [a.lineage for o in result.observations for a in o.arrivals]
    assert not any(step.endswith(":obstruction") for lineage in lineages for step in lineage)


@pytest.mark.parametrize("side_y", [8.0, 40.0, 80.0])
def test_light_another_element_takes_beside_the_obstruction_is_not_counted_as_the_annulus_s(side_y: float) -> None:
    """A second absorber SIDE (radius 3 mm, centred ``side_y`` from the axis, 100 mm before the 5 mm dump) takes part of the beam
    beside the dump's support. The annulus's power is the Gaussian share beside the dump, ``1 - disc(5, 0, 15)``, which still
    counts the light SIDE absorbed (``disc(3, 8, 15)`` of the beam: the two supports do not overlap in the near-collimated beam),
    so it is not the light's: the plane's power is out of model scope behind SIDE, and every result read from it is withheld
    (``POWER_OUT_OF_MODEL_SCOPE``), as for a beam that meets an element beside its chief ray without an obstruction. At 40 mm
    SIDE (its edge 2.5 beam radii out) meets no pupil sample but still takes 8.6e-8 of the beam, which the beam's tail check
    on the leg to the dump finds with a bound: the plane reads the annulus on ``[P - S, P]``, which holds the light that is
    there, ``P (stated - disc(3, 40, 15))``; its map stays unavailable (``POWER_TAKEN_BEFORE_PLANE``). Out of the beam's
    envelope (80 mm) the annulus's power is stated."""

    from os_optics.results.irradiance import irradiance_map

    beam = _scene(B.beam_dump("DUMP", position=(0.0, 0.0, 0.0), axis=X, radius_mm=5.0))
    side = B.beam_dump("SIDE", position=(-100.0, side_y, 0.0), axis=X, radius_mm=3.0)
    result = run(beam.model_copy(update={"components": (*beam.components, side)}))
    arrival = one_arrival(result, "OUT")
    observation = next(o for o in result.observations if o.observation_id == "OUT")
    stated = 1.0 - disc(5.0, 0.0, W)
    assert any(step == "DUMP/dump:obstruction" for step in arrival.lineage)
    assert arrival.beam.power_W == pytest.approx(P_W * stated, rel=1e-7)
    if side_y == 40.0:
        taken = disc(3.0, 40.0, W)
        assert 1e-8 < taken < 1e-4
        assert arrival.power_out_of_scope_at == () and arrival.power_taken_before_at == ("SIDE/dump",)
        assert taken * P_W <= arrival.power_taken_before_upper_W < 1e-3 * P_W
        assert observation.quantity_state == "BOUNDED"
        assert observation.power_lower_bound_W <= P_W * (stated - taken) <= arrival.beam.power_W
        doc = irradiance_map(result, "OUT")
        assert (doc["status"], doc["reason"], doc["power_taken_before_at"]) == ("UNAVAILABLE", "POWER_TAKEN_BEFORE_PLANE", ["SIDE/dump"])
        (issue,) = [i for i in result.issues if i.location == "surface:SIDE/dump"]
        assert (issue.code.value, issue.status.value) == ("POWER_TAKEN_BEFORE_PLANE", "WARNING")
        assert issue.numerical_context["tail_power_upper_W"] == arrival.power_taken_before_upper_W
    elif side_y == 8.0:
        taken = disc(3.0, 8.0, W)
        assert taken > 0.02  # (the light that power still counts: the plane's light is at most stated - taken)
        assert arrival.power_out_of_scope_at == ("SIDE/dump",)
        assert observation.quantity_state == "OUT_OF_MODEL_SCOPE"
        doc = irradiance_map(result, "OUT")
        assert (doc["status"], doc["reason"], doc["power_out_of_scope_at"]) == ("UNAVAILABLE", "POWER_OUT_OF_MODEL_SCOPE", ["SIDE/dump"])
        assert any(i.code.value == "OUT_OF_MODEL_SCOPE" and i.location == "surface:SIDE/dump" for i in result.issues)
    else:
        assert arrival.power_out_of_scope_at == ()
        assert observation.quantity_state == "KNOWN"
    assert result.power_budget.conserved
