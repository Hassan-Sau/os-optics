"""Counter-propagating beams on one observation plane : a beam and its own retro-reflection crossing a plane in
opposite directions have no interference term in the power through the plane (the time-averaged Poynting flux of
anti-parallel fields has no cross term), so the coherent sums, the irradiance grid and the two-beam combined power do not
depend on a sub-wavelength mirror position."""

from __future__ import annotations

import pytest

from os_optics.builders import laser, mirror, npbs, observation_plane
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.results.irradiance import irradiance_map

from helpers import run, stated_splitter

X = (1.0, 0.0, 0.0)
MX = (-1.0, 0.0, 0.0)
WAVELENGTH_NM = 780.0
SOURCE_W = 1e-3
R_MIRROR = 0.95


def _observation(result, oid: str):
    return next(o for o in result.observations if o.observation_id == oid)


def _retro(shift_mm: float, *, targets: tuple[Target, ...] = ()):
    src = laser("SRC", wavelength_nm=WAVELENGTH_NM, power_W=SOURCE_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                direction=X, coherence_length_mm=1000.0)
    retro = mirror("M", position=(50.0 + shift_mm, 0.0, 0.0), normal=MX, reflectance=R_MIRROR)
    monitor = observation_plane("mon", position=(0.0, 0.0, 0.0), normal=X, radius_mm=5.0)
    return ProjectSnapshot(components=(src, retro), observations=(monitor,), targets=targets)


SHIFTS_MM = tuple(k * WAVELENGTH_NM / 16.0 * 1e-6 for k in range(8))  # lambda/16 steps over half a wavelength


def test_counter_propagating_beams_have_no_interference_term_in_the_coherent_sum():
    target = Target(target_id="coherent", metric="coherent_power_W", observation_id="mon", maximum=SOURCE_W, unit="W")
    values = []
    for shift in SHIFTS_MM:
        result = run(_retro(shift, targets=(target,)))
        obs = _observation(result, "mon")
        assert len(obs.arrivals) == 2
        assert {a.direction_uvn[2] > 0 for a in obs.arrivals} == {True, False}
        (s,) = obs.superpositions
        assert s.interference_power_W == 0.0
        assert s.total_power_W == pytest.approx(SOURCE_W * (1.0 + R_MIRROR), rel=1e-12)
        values.append(result.targets[0].value)
    assert max(values) - min(values) <= 1e-15
    assert max(values) <= SOURCE_W * (1.0 + R_MIRROR) * (1.0 + 1e-12)


def test_counter_propagating_pair_has_no_cross_term_in_the_collinear_power():
    for shift in SHIFTS_MM[:4]:
        result = run(_retro(shift))
        (pair,) = [p for p in result.interference if p.observation_id == "mon"]
        assert pair.combined_power_collinear_W == pytest.approx(pair.incoherent_power_sum_W, rel=1e-15)
        assert pair.combined_power_collinear_W == pytest.approx(SOURCE_W * (1.0 + R_MIRROR), rel=1e-12)
        assert "NO_FRINGES" in pair.issue_codes


def test_no_fringes_text_does_not_claim_a_power_modulation_for_counter_propagating_beams():
    result = run(_retro(SHIFTS_MM[1]))
    messages = [i.message for i in result.issues if i.code.value == "NO_FRINGES"]
    assert messages
    assert all("power modulation" not in m for m in messages)
    assert any("opposite directions" in m for m in messages)


def test_counter_propagating_irradiance_grid_power_equals_the_incoherent_power():
    for shift in SHIFTS_MM[:3]:
        grid = irradiance_map(run(_retro(shift)), "mon", samples=81)
        assert grid["coherent_W_per_mm2"] is not None
        assert grid["grid_power_coherent_W"] == pytest.approx(grid["grid_power_incoherent_W"], rel=1e-12)


def test_same_direction_beams_still_interfere():
    """Two collinear same-direction beams (the transmitted beam and a double reflection between two splitters) keep
    their cross term."""
    src = laser("SRC", wavelength_nm=WAVELENGTH_NM, power_W=SOURCE_W, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0),
                direction=X, coherence_length_mm=1000.0)
    s1 = stated_splitter(npbs("S1", reflectance=0.3, position=(0.0, 0.0, 0.0), normal=MX))  # (phases stated: delays known)
    s2 = stated_splitter(npbs("S2", reflectance=0.3, position=(10.0, 0.0, 0.0), normal=X))
    out = observation_plane("out", position=(60.0, 0.0, 0.0), normal=X, radius_mm=5.0)
    result = run(ProjectSnapshot(components=(src, s1, s2), observations=(out,)))
    (s,) = _observation(result, "out").superpositions
    assert abs(s.interference_power_W) > 1e-6


