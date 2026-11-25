"""The run's per-pair work grows with the pairs that can matter, not with every pair of beams.

* ``validation/superposition.py``: a plane's coherent sum forms cross terms only within one coherence group; the result
  is the same as forming every pair (a reference that loops over every pair is kept here), and a plane of N mutually
  incoherent beams evaluates no pair at all.
* ``local/validity.py``: where a model-scope finding starts on a path, the observations and devices it reaches are found
  by walking that path's branches, with the same result as testing every path of the run (reference below), on random
  branch trees.
"""

from __future__ import annotations

import functools
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from os_optics.builders import laser, npbs, observation_plane
from os_optics.contracts import ProjectSnapshot
from os_optics.contracts.optics import InterfacePhases
from os_optics.errors import Status
from os_optics.local.validity import SCOPE_DISTANCE_TOLERANCE_MM, _Collector
from os_optics.math.exact_linalg import cvdot
from os_optics.math.vectors import correctly_rounded_sum
from os_optics.validation import superposition as sp

from helpers import CFG, run

X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
Q = math.sqrt(0.5)


def _reference_superpose(members, plane, cfg, sources=None):
    """The coherent sum over every pair of ``members`` (the loop before pairs were grouped): (total, interference, codes)."""
    u, v, normal = np.array(plane.u_axis), np.array(plane.v_axis), np.array(plane.normal)
    incoherent = float(correctly_rounded_sum(sp.received_power_W(m) for m in members))
    interference = below = above = open_terms = 0.0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if not sp.same_direction(a, b, normal):
                continue
            gamma, delta, open_phase, code, _assumed = sp._bounded_gamma(a, b, cfg, sources)
            if code is not None:
                return None, None, code
            if gamma == 0.0 and delta == 0.0:
                continue
            na = a.beam.gaussian.medium_index
            integral = sp.cross_integral(a, b, u, v, na, received=True)
            overlap = cvdot(sp._polarization(a), sp._polarization(b))
            flux = 0.5 * (sp._cos(a, normal) + sp._cos(b, normal))
            assert not open_phase
            cross = overlap * integral
            term = (2.0 * gamma * cross.real * flux if not isinstance(gamma, complex)
                    else 2.0 * (gamma.real * cross.real + gamma.imag * cross.imag) * flux)
            interference += term
            assert delta == 0.0
    return incoherent + interference, interference, None


def _stated(component):
    s = component.surfaces[0]
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2.0,), reflection_p_rad=(3.0 * math.pi / 2.0,),
                             transmission_s_rad=(0.0,), transmission_p_rad=(0.0,))
    return component.model_copy(update={"surfaces": (s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})}),)})


def _mixed_groups_project():
    """Three independent 633 nm lasers side by side on one plane; each of the first two is split by its own beam splitter
    and recombined on the plane by a second one, so the plane holds same-group pairs (two arms of one laser) interleaved
    with beams of other groups."""
    components, y = [], 0.0
    for k in range(3):
        y = 4.0 * k
        components.append(laser(f"L{k}", wavelength_nm=633.0, waist_radius_mm=0.3, position=(-50.0, y, 0.0), direction=X,
                                coherence_length_mm=1000.0))
        if k < 2:
            components.append(_stated(npbs(f"BS{k}", reflectance=0.5, position=(0.0, y, 0.0), normal=(-Q, Q, 0.0))))
            components.append(_stated(npbs(f"BR{k}", reflectance=0.5, position=(0.0, y + 0.002, 0.0), normal=(Q, -Q, 0.0))))
    plane = observation_plane("P", position=(80.0, 0.0, 0.0), normal=X, radius_mm=40.0)
    return ProjectSnapshot(components=tuple(components), observations=(plane,))


@functools.lru_cache(maxsize=1)
def _mixed_groups_arrivals():
    """The plane's arrivals of one run of ``_mixed_groups_project`` (shared by the tests below).

    Each recombining splitter BR lies 0.002 mm behind its BS, so every beam ends on one coating with the other one beside
    its chief ray. The Gaussian-tail check beside the chief ray (``Tracer._tail_surfaces``) tests a plane coating by its own
    outline against the leg's start and end planes: the other coating lies wholly beyond the end (or behind the start), so
    no arrival's power is out of model scope."""
    (obs,) = run(_mixed_groups_project()).observations
    assert obs.arrivals and not any(a.power_out_of_scope_at for a in obs.arrivals)
    return tuple(obs.arrivals), obs.plane


def test_grouped_superposition_equals_every_pair():
    arrivals, plane = _mixed_groups_arrivals()
    groups = {a.coherence_group for a in arrivals}
    assert len(arrivals) >= 4 and len(groups) == 3
    members = sorted(arrivals, key=lambda a: a.beam.path_id)
    # (interleave the groups so a pair loop restricted to one group skips beams between its members)
    members = members[::2] + members[1::2]
    got = sp.superpose(members, plane, CFG)
    total, interference, code = _reference_superpose(members, plane, CFG)
    assert code is None and got.status is Status.PASS
    assert got.total_power_W == total and got.interference_power_W == interference  # the same terms in the same order
    assert interference != 0.0  # (the arms of one laser interfere)


def test_a_plane_of_independent_beams_evaluates_no_pair(monkeypatch):
    arrivals, plane = _mixed_groups_arrivals()
    independent = [a.model_copy(update={"coherence_group": f"g{k}"}) for k, a in enumerate(arrivals)]
    calls = []
    original = sp._bounded_gamma
    monkeypatch.setattr(sp, "_bounded_gamma", lambda *args, **kw: calls.append(1) or original(*args, **kw))
    got = sp.superpose(independent, plane, CFG)
    assert calls == [] and got.interference_power_W == 0.0
    assert got.total_power_W == float(correctly_rounded_sum(sp.received_power_W(a) for a in independent))


def _reference_resolve(collector: _Collector, path_id: str, minimum_length: float):
    """The scope context of a location by testing every path of the run (the rule before the branch walk)."""
    def descended(pid):
        while pid in collector.paths:
            if pid == path_id:
                return True
            if collector.start_length(collector.paths[pid]) < minimum_length - SCOPE_DISTANCE_TOLERANCE_MM:
                return False
            pid = collector.paths[pid].parent_path_id
        return False
    context = {o.observation_id for o in collector.result.observations
               if any(descended(a.beam.path_id) and a.beam.geometric_length_mm > minimum_length + SCOPE_DISTANCE_TOLERANCE_MM
                      for a in o.arrivals)}

    def scattered_after(line) -> bool:  # (the light a surface scattered out of a descendant behind the point)
        if not descended(line.parent_path_id):
            return False
        if line.parent_path_id != path_id:
            return True
        path = collector.paths[path_id]
        distance = collector.start_length(path)
        for segment in path.segments:
            distance += segment.length_mm or 0.0
            if segment.end_surface_id == line.scatter_surface_id and distance >= minimum_length - SCOPE_DISTANCE_TOLERANCE_MM:
                return True
        return False

    lines = {(o.observation_id, line.source_id, line.spectral_index) for o in collector.result.observations for line in o.sampled_light
             if line.scatter_surface_id is not None and scattered_after(line)}
    context |= {oid for oid, _, _ in lines}
    devices = set()
    for path in collector.result.paths:
        if not descended(path.path_id):
            continue
        distance = collector.start_length(path)
        for segment in path.segments:
            distance += segment.length_mm or 0.0
            owner = collector.owners.get(segment.end_surface_id or "")
            if owner is not None and distance >= minimum_length - SCOPE_DISTANCE_TOLERANCE_MM:
                devices.add(owner[0].instance_id)
    return context, devices, lines


def _random_run(rng: random.Random, paths: int):
    """A run of ``paths`` paths in a random branch tree (a branch starts at a point along its parent, with its parent's
    source and wavelength), each with a few segments ending on random surfaces, and observation arrivals (first crossing)
    at random lengths along random paths, and sampled lines of light a random surface scattered out of a random path (and an
    emitter's line, which no surface scattered)."""
    surfaces = [f"S{k}" for k in range(8)]
    built, starts = [], {}
    for k in range(paths):
        parent = None if k == 0 or rng.random() < 0.3 else built[rng.randrange(len(built))]
        start = 0.0 if parent is None else starts[parent.path_id] + rng.uniform(0.0, parent_length(parent))
        segments = [SimpleNamespace(length_mm=rng.choice([rng.uniform(0.0, 30.0), None, 0.0]),
                                    end_surface_id=rng.choice(surfaces + [None])) for _ in range(rng.randint(1, 5))]
        beam = (f"s{k % 2}", (532.0, 1064.0)[k % 3 % 2]) if parent is None else (parent.source_id, parent.wavelength_nm)
        path = SimpleNamespace(path_id=f"p{k}", parent_path_id=None if parent is None else parent.path_id, segments=segments,
                               geometric_path_length_mm=start + sum(s.length_mm or 0.0 for s in segments),
                               source_id=beam[0], wavelength_nm=beam[1])
        starts[path.path_id] = start
        built.append(path)
    observations = []
    for o in range(4):
        arrivals = [SimpleNamespace(beam=SimpleNamespace(path_id=rng.choice(built).path_id, geometric_length_mm=rng.uniform(0.0, 120.0)),
                                    crossing_index=0)
                    for _ in range(rng.randint(0, 6))]
        sampled = [SimpleNamespace(source_id=f"s{k % 2}", spectral_index=k % 3, scatter_surface_id=rng.choice(surfaces),
                                   parent_path_id=rng.choice(built).path_id) for k in range(rng.randint(0, 3))]
        sampled.append(SimpleNamespace(source_id="EMITTER", spectral_index=0, scatter_surface_id=None, parent_path_id=None))
        observations.append(SimpleNamespace(observation_id=f"o{o}", arrivals=arrivals, sampled_light=sampled))
    owners = {s: (SimpleNamespace(instance_id=f"E{s}"), None) for s in surfaces[:6]}
    return SimpleNamespace(paths=built, observations=observations), owners


def parent_length(path) -> float:
    return sum(s.length_mm or 0.0 for s in path.segments)


@pytest.mark.parametrize("seed", range(6))
def test_scope_context_by_branch_walk_equals_every_path(seed):
    rng = random.Random(seed)
    result, owners = _random_run(rng, 60)
    collector = _Collector(result, owners)
    for path in result.paths:
        start = collector.start_length(path)
        for minimum in (start, start + rng.uniform(0.0, 40.0), start + 1e-4, rng.uniform(0.0, 150.0)):
            collector.after(path.path_id, minimum)
            collector._resolve()
            assert (collector.context, collector.context_devices, collector.context_lines) == _reference_resolve(
                collector, path.path_id, minimum)
    collector.after("not-a-path", 0.0)
    collector._resolve()
    assert collector.context == set() and collector.context_devices == set() and collector.context_lines == set()
