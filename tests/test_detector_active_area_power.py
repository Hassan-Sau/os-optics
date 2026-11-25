"""Absorbing observation planes with an active area : ``power_W``, ``total_power_W`` and ``power_by_wavelength`` are
the power the active area receives (the power ledger's ``DETECTED``); the incident beam power stays on the beam state and
``detected_power_W`` carries the detected part."""

from __future__ import annotations

import math

import pytest

from os_optics.builders import laser, observation_plane
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status

from helpers import one_arrival, run

X = (1.0, 0.0, 0.0)
WAVELENGTH_NM = 780.0
SOURCE_W = 1e-3


def _observation(result, oid: str):
    return next(o for o in result.observations if o.observation_id == oid)


def _detector(radius_mm: float, waist_mm: float, *, minimum: float | None = None, offset_y: float = 0.0):
    src = laser("SRC", wavelength_nm=WAVELENGTH_NM, power_W=SOURCE_W, waist_radius_mm=waist_mm, position=(-100.0, 0.0, 0.0), direction=X)
    pd = observation_plane("pd", position=(50.0, offset_y, 0.0), normal=X, radius_mm=radius_mm, absorbing=True)
    targets = () if minimum is None else (Target(target_id="p", metric="power_W", observation_id="pd", minimum=minimum, unit="W"),)
    return ProjectSnapshot(components=(src,), observations=(pd,), targets=targets)


def test_absorbing_active_area_power_is_the_detected_power():
    result = run(_detector(0.8, 1.0, minimum=0.9e-3))
    arrival = one_arrival(result, "pd")
    fraction = 1.0 - math.exp(-2.0 * 0.8**2 / arrival.beam.gaussian.beam_radius_x_mm**2)
    assert arrival.active_area_power_fraction == pytest.approx(fraction, rel=1e-6)
    detected = dict(result.power_budget.total_sinks)["DETECTED"]
    assert detected == pytest.approx(SOURCE_W * fraction, rel=1e-6)
    assert arrival.beam.power_W == pytest.approx(SOURCE_W, rel=1e-12)  # the incident beam
    assert arrival.detected_power_W == pytest.approx(detected, rel=1e-12)
    obs = _observation(result, "pd")
    assert obs.total_power_W == pytest.approx(detected, rel=1e-12)
    assert [p.power_W for p in obs.power_by_wavelength] == [pytest.approx(detected, rel=1e-12)]
    (target,) = result.targets
    assert target.value == pytest.approx(detected, rel=1e-12)
    assert target.status is Status.FAIL


def test_centred_beam_inside_the_active_area_reads_the_detected_power():
    result = run(_detector(0.7, 0.606, minimum=1.9e-3 / 2.0))
    arrival = one_arrival(result, "pd")
    detected = dict(result.power_budget.total_sinks)["DETECTED"]
    assert arrival.detected_power_W == pytest.approx(detected, rel=1e-12)
    assert result.targets[0].value == pytest.approx(detected, rel=1e-12)
    assert detected < SOURCE_W


def test_power_on_a_transparent_plane_with_an_aperture_is_the_crossing_beam():
    src = laser("SRC", wavelength_nm=WAVELENGTH_NM, power_W=SOURCE_W, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=X)
    mon = observation_plane("mon", position=(50.0, 0.0, 0.0), normal=X, radius_mm=0.8)
    result = run(ProjectSnapshot(components=(src,), observations=(mon,),
                                 targets=(Target(target_id="p", metric="power_W", observation_id="mon", minimum=0.0, unit="W"),)))
    arrival = one_arrival(result, "mon")
    assert arrival.detected_power_W is None
    assert result.targets[0].value == pytest.approx(SOURCE_W, rel=1e-12)




def _coherent_target(oid: str, minimum: float) -> Target:
    return Target(target_id="c", metric="coherent_power_W", observation_id=oid, minimum=minimum, unit="W")


def test_coherent_power_of_one_beam_on_an_active_area_is_its_detected_power():
    """Coherent_power_W equals power_W for a single beam, so on an absorbing plane with an active area it is the
    detected power: a 0.9 mW minimum fails where 0.72 mW reaches the active area (it read the 1 mW incident beam)."""

    snapshot = _detector(0.8, 1.0, minimum=0.9e-3)
    snapshot = snapshot.model_copy(update={"targets": snapshot.targets + (_coherent_target("pd", 0.9e-3),)})
    result = run(snapshot)
    arrival = one_arrival(result, "pd")
    power, coherent = result.targets
    assert coherent.value == pytest.approx(arrival.detected_power_W, rel=1e-12)
    assert coherent.value == pytest.approx(power.value, rel=1e-12)
    assert coherent.value < 0.9e-3 < SOURCE_W
    assert coherent.status is Status.FAIL


def _mzi_on(out_x):
    """The coherent_superposition Mach-Zehnder (arm phase 0.7 rad) with ``out_x`` as its OUT_X plane."""

    from test_coherent_superposition import _mzi

    snapshot = _mzi(0.7)
    observations = (out_x,) + tuple(o for o in snapshot.observations if o.observation_id != "OUT_X")
    return snapshot.model_copy(update={"observations": observations, "targets": (_coherent_target("OUT_X", 0.0),)})


def test_coherent_power_of_two_beams_on_an_active_area_is_the_detected_coherent_sum():
    """Two co-directional coherent beams on an absorbing active area: the coherent sum, coherent_power_W and the two-beam
    combined power are in detected power. Both arms have the same Gaussian footprint, so the detected sum is the active-area
    fraction of the whole-plane sum (each field scaled by sqrt(detected / incident))."""

    transparent = run(_mzi_on(observation_plane("OUT_X", position=(160.0, 100.0, 0.0), normal=X)))
    detector = run(_mzi_on(observation_plane("OUT_X", position=(160.0, 100.0, 0.0), normal=X, radius_mm=0.4, absorbing=True)))
    whole = _observation(transparent, "OUT_X").superpositions[0].total_power_W
    obs = _observation(detector, "OUT_X")
    fractions = {a.active_area_power_fraction for a in obs.arrivals}
    assert len(obs.arrivals) == 2 and max(fractions) - min(fractions) < 1e-9 and max(fractions) < 0.99
    (fraction,) = {round(f, 9) for f in fractions}
    (s,) = obs.superpositions
    assert s.incoherent_power_W == pytest.approx(sum(a.detected_power_W for a in obs.arrivals), rel=1e-12)
    assert s.total_power_W == pytest.approx(fraction * whole, rel=1e-6)
    assert detector.targets[0].value == pytest.approx(s.total_power_W, rel=1e-12)
    (pair,) = [i for i in detector.interference if i.observation_id == "OUT_X"]
    assert pair.combined_power_collinear_W == pytest.approx(s.total_power_W, rel=1e-9)
    assert transparent.targets[0].value == pytest.approx(whole, rel=1e-12)
