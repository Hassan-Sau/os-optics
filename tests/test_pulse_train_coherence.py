"""A pulse train's pulses may be mutually coherent (``optics/interference/coherence.py``, "Pulse trains").

A mode-locked laser emits copies of one pulse with a fixed phase: two fields of its train are coherent again where one
field's pulse overlaps a later pulse of the other, at relative group delays near every whole multiple of the period
length ``c / f_rep``. A source's data do not say whether its pulses are mutually coherent, so a pulsed source counts as if
they were: the exposure bounds take the larger coherence of the delay and of its distance to the nearest non-zero
multiple, and a coherence metric has no value where such a revival could change it.

The reference is a Mach-Zehnder (``test_golden_interference._mzi_project``) with a 10 mm N-BK7 plate in one arm: its arms'
relative group delay ``D`` is read from a continuous run, and the train's repetition rate is set so that the period length
is ``D`` (a revival at the delay) or ``3 D`` (none near it: the nearest multiple is ``2 D`` away). The declared
coherence length of 1 mm is far below ``D`` (about 5 mm).
"""

from __future__ import annotations

import math

import pytest

from os_optics.contracts import PowerRating, PulseTrain, Target
from os_optics.optics.interference.coherence import (C_MM_PER_S, REVIVAL_NEGLIGIBLE, degree_of_coherence, nearest_delay,
                                                          period_length_mm, revival_coherence, revival_distance)

from helpers import run
from test_golden_interference import _mzi_project

FIXTURE = "test fixture rating, not product data"
LENGTH_MM = 1.0


def test_the_distance_to_the_nearest_revival_and_its_coherence():
    """``revival_distance``: the distance from |D| to the nearest non-zero multiple of the period length (written out);
    ``nearest_delay`` the smaller of |D| and that; ``revival_coherence`` the Gaussian-spectrum coherence there."""
    assert period_length_mm(1e9) == pytest.approx(C_MM_PER_S / 1e9, rel=1e-15) == pytest.approx(299.792458, rel=1e-15)
    assert revival_distance(0.0, 30.0) == 30.0  # (the nearest non-zero multiple of 0 is one period away)
    assert revival_distance(29.0, 30.0) == 1.0 and revival_distance(-31.5, 30.0) == 1.5 and revival_distance(44.0, 30.0) == 14.0
    assert revival_distance(76.0, 30.0) == 14.0 and revival_distance(89.0, 30.0) == 1.0
    assert nearest_delay(29.0, (30.0,)) == 1.0 and nearest_delay(2.0, (30.0,)) == 2.0 and nearest_delay(-5.0, ()) == 5.0
    assert nearest_delay(29.0, (100.0, 30.0)) == 1.0
    assert revival_coherence(29.0, 0.0, 1.0, (30.0,)) == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert revival_coherence(29.0, 0.0, 1.0, ()) == 0.0
    assert revival_coherence(29.0, 0.0, 1.0, (30.0,)) == degree_of_coherence(1.0, 0.0, 1.0)


def _arms(pulse: PulseTrain | None, rating: float | None = None):
    """The Mach-Zehnder with a 10 mm plate in arm B, its laser declaring a 1 mm coherence length, optionally pulsed and
    with a CW-or-pulsed fixture rating on the second splitter, where the two arms meet."""
    project = _mzi_project(10.0, 1064.0)
    comps = []
    for c in project.components:
        if c.source is not None:
            c = c.model_copy(update={"source": c.source.model_copy(update={"coherence_length_mm": LENGTH_MM, "pulse": pulse})})
        if c.instance_id == "BS2" and rating is not None:
            extra = (dict(regime="CW", unit="W/cm2") if pulse is None else
                     dict(regime="PULSED", unit="J/cm2", pulse_duration_s=pulse.pulse_duration_s))
            c = c.model_copy(update={"power_ratings": (PowerRating(value=rating, wavelength_nm=1064.0, status="DECLARED",
                                                                    basis=FIXTURE, **extra),)})
        comps.append(c)
    return project.model_copy(update={"components": tuple(comps)})


def _delay() -> float:
    pair = next(i for i in run(_arms(None)).interference if i.observation_id == "port_x")
    assert pair.group_path_difference_mm is not None and 4.0 < abs(pair.group_path_difference_mm) < 7.0
    return abs(pair.group_path_difference_mm)


def _train(period_mm: float) -> PulseTrain:
    return PulseTrain(repetition_rate_hz=C_MM_PER_S / period_mm, pulse_duration_s=1e-13)


def test_a_pulse_train_coherent_again_at_its_period_leaves_the_coherence_metrics_open():
    """Continuous light: the arms are 5 coherence lengths apart, coherence factor exp(-(D / L_c)^2) (written out). A train
    whose period length is D overlaps the next pulse of one arm with the other arm's: the coherence factor, fringe visibility
    peak irradiance and coherence margin have no value (``COHERENCE_UNDEFINED``, the message names the period length) and a
    target on them is UNKNOWN -- a margin target 'the arms do not interfere' (maximum -1) PASSes for continuous light and is
    UNKNOWN for the train --, while the group path difference and the coherence length stay. A train whose nearest revival
    is 2 D away reports the continuous light's values."""
    delay = _delay()
    cw = next(i for i in run(_arms(None)).interference if i.observation_id == "port_x")
    assert cw.coherence_factor == pytest.approx(math.exp(-(delay / LENGTH_MM) ** 2), rel=1e-6)
    revived = run(_arms(_train(delay)))
    pair = next(i for i in revived.interference if i.observation_id == "port_x")
    assert pair.coherence_factor is None and pair.fringe_visibility is None and pair.peak_irradiance_W_per_mm2 is None
    assert "COHERENCE_UNDEFINED" in pair.issue_codes
    assert abs(pair.group_path_difference_mm) == pytest.approx(delay, rel=1e-12)
    assert pair.coherence_margin is None and pair.coherence_length_mm == LENGTH_MM
    assert cw.coherence_margin == pytest.approx(1.0 - delay / LENGTH_MM, rel=1e-12) and cw.coherence_margin < -1.0
    margin = Target(target_id="m", metric="coherence_margin", observation_id="port_x", maximum=-1.0)
    assert run(_arms(None).model_copy(update={"targets": (margin,)})).targets[0].status.value == "PASS"
    open_margin = run(_arms(_train(delay)).model_copy(update={"targets": (margin,)})).targets[0]
    assert open_margin.status.value == "UNKNOWN" and open_margin.value is None and "pulse train" in open_margin.message
    issue = next(i for i in revived.issues if i.code.value == "COHERENCE_UNDEFINED" and "pulse train" in i.message)
    assert "period length" in issue.message and issue.numerical_context["period_length_mm"] == pytest.approx(delay, rel=1e-12)
    far = next(i for i in run(_arms(_train(3.0 * delay))).interference if i.observation_id == "port_x")
    assert revival_coherence(delay, 0.0, LENGTH_MM, (3.0 * delay,)) < REVIVAL_NEGLIGIBLE
    assert far.coherence_factor == pytest.approx(cw.coherence_factor, rel=1e-12) and "COHERENCE_UNDEFINED" not in far.issue_codes
    assert far.coherence_margin == pytest.approx(cw.coherence_margin, rel=1e-12)


def test_a_coherent_power_near_a_revival_is_not_stated():
    """The coherent sum of the two arms on a port: a number for continuous light and far from a revival, none (UNKNOWN,
    ``COHERENCE_UNDEFINED``) where the train's next pulse overlaps."""
    delay = _delay()

    def port(pulse):
        obs = next(o for o in run(_arms(pulse)).observations if o.observation_id == "port_x")
        (total,) = obs.superpositions
        return total

    assert port(None).total_power_W is not None and port(_train(3.0 * delay)).total_power_W is not None
    revived = port(_train(delay))
    assert revived.status.value == "UNKNOWN" and revived.total_power_W is None and "COHERENCE_UNDEFINED" in revived.issue_codes


def test_the_exposure_counts_a_pulse_train_coherent_again_at_its_period():
    """On the second splitter the two arms meet: with continuous light their degree of coherence anywhere on the surface is
    tiny (the delay is about 5 coherence lengths); a train whose period length is the delay makes them fully coherent (the
    next pulse overlaps): the exposure's upper bound is then the in-phase maximum, never the incoherent sum."""
    delay = _delay()

    def splitter(pulse):
        result = run(_arms(pulse, rating=1e12))
        return next(s for s in result.exposures.surfaces if s.surface_id == "BS2/coating")

    cw = splitter(None)
    revived = splitter(_train(delay))
    far = splitter(_train(3.0 * delay))
    assert revived.coherent_groups and max(g.largest_coherence_factor for g in revived.coherent_groups) == 1.0
    assert all(g.largest_coherence_factor < 0.5 for g in cw.coherent_groups)
    # (the traced arms alone set the surface values: their in-phase maximum with the revival, about their sum without)
    assert revived.peak_irradiance_W_per_cm2 > 1.9 * cw.peak_irradiance_W_per_cm2
    assert far.peak_irradiance_W_per_cm2 == pytest.approx(cw.peak_irradiance_W_per_cm2, rel=1e-9)
