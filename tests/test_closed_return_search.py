"""The closed-return screen ends, states its answer and stays within its work and storage limits.

The screen of the return-light analysis

* examines trails shortest first and stops at the first closed return it cannot rule out (``POTENTIAL_RETURN``); holds one
  frame per step of the trail being extended (``ClosedReturnSearch.frames_peak`` <= ``frames_limit``: one per directed leg
  between the screened surfaces), never one copy of a trail per branch;
* counts its work against ``max_return_analysis_work``, the run's limit: the wavelengths share it in the order the run
  reads them; a screen that reaches it is ``INCOMPLETE`` and supplies no closed-return bound;
* records, per wavelength, its answer, stage, termination, work and storage in ``outcome.execution.budgets.return_analysis``.

Its answer is checked against an independent reference: an unbounded depth-first trail enumeration (``_reference``), on
scenes where it ends -- the same existence answer, and the cycle found is a closed trail the narrowed cone follows. All
scenes are synthetic (rings, folds and stacks built here or in the other return-analysis tests).
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.validation import closed_returns as closed_module
from os_optics.validation import resonance
from os_optics.validation.resonance import (EITHER, REFLECTING, _cone, _critical, _incidence, _other, _overlap,
                                            _reflected, _sides)
from os_optics.validation.return_work import WorkCounter

from helpers import run as run_snapshot
from test_closed_returns import faint_ring
from test_return_analysis_budget import three_mirrors
from test_return_pairs import LAM, _coat, _constant, _scene


def test_wavelengths_share_the_run_limit_in_the_order_they_are_read():
    """The run's limit, shared in reading order (``resonance.ReturnAnalysis``): with room for every wavelength each finishes
    as it does alone -- the same answer and work --; one unit less and the last read stops (INCOMPLETE), the others
    unchanged; the total never passes the limit."""
    wavelengths = [LAM, LAM + 10.0, LAM + 20.0]
    order = [wavelengths[2], wavelengths[0], wavelengths[1]]
    scene, library = _scene(three_mirrors())
    alone = {w: resonance.ReturnAnalysis(scene, library, [w])[w] for w in wavelengths}
    need = sum(g.work.steps for g in alone.values())
    for limit in (need, need - 1, alone[order[0]].work.steps):
        analysis = resonance.ReturnAnalysis(scene, library, wavelengths, work_limit=limit)
        read = [analysis[w] for w in order]
        assert sum(g.work.steps for g in read) <= limit
        before = 0
        for g in read:
            assert g.work.before == before and g.work.limit == limit - before
            before += g.work.steps
        for g in read:
            if g.screened:
                assert g.work.steps == alone[g.wavelength_nm].work.steps
                assert g.closed_return_screen == alone[g.wavelength_nm].closed_return_screen == "COMPLETED_NO_RETURN"
        assert read[-1].screened == (limit == need)
        assert read[0].screened


def _ring4() -> ProjectSnapshot:
    return faint_ring(corners=4)


def _folded() -> ProjectSnapshot:
    source = B.laser("S", wavelength_nm=LAM, power_W=1, waist_radius_mm=.5, position=(-100, 0, 0), direction=(1, 0, 0))
    first = B.window("W1", thickness_mm=3, clear_radius_mm=12.7, material="N-BK7", position=(0, 0, 0))
    a = _coat(B.mirror_for_reflection("M1", position=(100, 0, 0), d_in=(1, 0, 0), d_out=(0, 0, 1), clear_radius_mm=12.7),
              _constant(.995))
    b = _coat(B.mirror_for_reflection("M2", position=(100, 0, 100), d_in=(0, 0, 1), d_out=(1, 0, 0), clear_radius_mm=12.7),
              _constant(.995))
    last = B.window("W2", thickness_mm=3, clear_radius_mm=12.7, material="N-BK7", position=(150, 0, 100))
    return ProjectSnapshot(components=(source, first, a, b, last))


SCENES = {"ring3": faint_ring, "ring4": _ring4, "three_mirrors": three_mirrors, "folded": _folded}


def _reference(sc) -> tuple[int, ...] | None:
    """An unbounded closed-return trail search: depth first over every trail, one copy of the trail per branch, no work
    limit -- the independent reference of the screen's existence answer."""
    candidates = [i for i, kind in enumerate(sc.kinds) if kind in REFLECTING and (sc.faces[i].plane or kind != "refract")]
    if len(candidates) < 3:
        return None

    def reflectance(b, incoming):
        low, high = _incidence(incoming, sc.faces[b])
        r = sc.coefficients(b, incoming.side, low, high)[0]
        if sc.kinds[b] == "refract":
            for side in _sides(incoming.side):
                r1, r2 = sc.side_range(b, side), sc.side_range(b, _other(side))
                n1, n2 = (None, None) if r1 is None or r2 is None else (r1[1], r2[0])
                critical = None if n1 is None or n2 is None else _critical(n1, n2)
                if n1 is None or n2 is None or (critical is not None and high >= critical):
                    r = 1.0
        return r

    edges = {(a, b): [] for a in candidates for b in candidates if a != b}
    for (a, b), targets in edges.items():
        incoming = sc.leg(a, b)
        if reflectance(b, incoming) == 0.0:
            continue
        reflected = _cone(_reflected(incoming.u, sc.faces[b].axis), incoming.reach + 2.0 * sc.faces[b].cone, incoming.side)
        for c in candidates:
            toward = sc.toward(b, c) if c != b else None
            if c == b or (incoming.side != EITHER and toward != EITHER and incoming.side != toward):
                continue
            if _overlap(reflected, sc.leg(b, c)):
                targets.append((b, c))

    def advance(current, b, c):
        image = _cone(_reflected(current.u, sc.faces[b].axis), current.reach + 2 * sc.faces[b].cone, current.side)
        geometric = sc.leg(b, c)
        if not _overlap(image, geometric):
            return None
        chosen = image if image.reach < geometric.reach else geometric
        return _cone(chosen.u, chosen.reach, geometric.side)

    def possible(path):
        current = sc.leg(path[-1], path[0])
        for _ in range(2 * len(path)):
            for k, b in enumerate(path):
                current = advance(current, b, path[(k + 1) % len(path)])
                if current is None:
                    return False
        return True

    members = candidates  # (every candidate: a closed trail lies in one strongly connected group anyway)
    for start in members:
        for second in members:
            if second == start:
                continue
            stack = [((start, second), sc.leg(start, second))]
            while stack:
                path, incoming = stack.pop()
                visited = set(zip(path, path[1:]))
                inter = len({sc.surfaces[i].owner_id for i in path}) >= 2
                for _, nxt in edges[path[-2:]]:
                    outgoing = advance(incoming, path[-1], nxt)
                    if outgoing is None:
                        continue
                    if path[-1] == start and nxt == second and len(set(path)) >= 3 and inter and possible(path[:-1]):
                        return path[:-1]
                    if (path[-1], nxt) not in visited:
                        stack.append(((*path, nxt), outgoing))
    return None


def _screen(project: ProjectSnapshot, limit: int | None = None):
    scene, library = _scene(project)
    faces = [resonance._face(i, s) for i, s in enumerate(scene.surfaces)]
    sc = resonance._Scene(scene, library, LAM, faces, None, work=WorkCounter(limit))
    return sc, closed_module.closed_returns(sc)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_screen_answers_as_the_unbounded_reference_does(name):
    sc, (cycles, series) = _screen(SCENES[name]())
    reference = _reference(_screen(SCENES[name]())[0])
    assert bool(cycles) == (reference is not None), name
    search = sc.search
    assert 0 <= search.frames_peak <= search.frames_limit
    if cycles and series is None:
        assert search.termination == "RETURN_FOUND"
        ids = [s.surface_id for s in sc.surfaces]
        path = [ids.index(s) for s in cycles[0]]
        # (the cycle found is a closed trail the narrowed cone follows, of three or more surfaces of two or more elements,
        # and no longer than the reference's)
        assert len(set(path)) >= 3 and len({sc.surfaces[i].owner_id for i in path}) >= 2
        assert len(path) <= len(reference)
    elif not cycles:
        assert search.termination == "COMPLETED"


def test_rings_and_folds_stay_detected_with_every_budget_that_finishes():
    for project in (faint_ring(), _ring4(), _folded()):
        _, (cycles, _) = _screen(project)
        need = _screen(project)[0].work.steps
        for limit in (need, need + 1, 10 * need):
            sc, (again, _) = _screen(project, limit)
            assert again == cycles and sc.search.termination == "RETURN_FOUND"
        for limit in (1, need // 2, need - 1):
            sc, (none, series) = _screen(project, limit)
            # (stopped: no cycle, no series -- INCOMPLETE, never "no return")
            assert none == () and series is None and sc.work.record().exhausted
            assert sc.search.termination == "BUDGET_EXHAUSTED"


def test_the_analysis_states_its_screen_in_three_ways():
    complete = resonance.ReturnAnalysis(*_scene(three_mirrors()), [LAM])[LAM]
    assert complete.closed_return_screen == "COMPLETED_NO_RETURN" and math.isfinite(complete.crossings("B/face"))
    found = resonance.ReturnAnalysis(*_scene(faint_ring()), [LAM])[LAM]
    assert found.closed_return_screen == "POTENTIAL_RETURN" and found.crossings("B/face") == math.inf
    stopped = resonance.ReturnAnalysis(*_scene(three_mirrors()), [LAM], work_limit=complete.work.steps - 1)[LAM]
    assert stopped.closed_return_screen == "INCOMPLETE" and stopped.crossings("B/face") == math.inf
    assert stopped.search.termination == "BUDGET_EXHAUSTED"
    cut = resonance.ReturnAnalysis(*_scene(three_mirrors()), [LAM], work_limit=1)[LAM]
    assert cut.closed_return_screen == "INCOMPLETE" and cut.search.termination == "NOT_STARTED"


# ------------------------------------------------------------------------------------------------ what the texts say


def test_a_target_left_undecided_by_a_closed_path_names_that_path():
    """A target on a plane beside a three-mirror ring (no pair whose returns are unbounded) names the closed path
    the screen found, never a pair of surfaces facing each other that does not exist."""
    side = B.observation_plane("side", position=(50, -40, 0), normal=(0, 1, 0), radius_mm=10)
    targets = (Target(target_id="side-power", observation_id="side", metric="power_W", maximum=1e-3),
               Target(target_id="side-coh", observation_id="side", metric="coherent_power_W", maximum=1e-3))
    project = faint_ring().model_copy(update={"observations": (side,), "targets": targets})
    gains = resonance.ReturnAnalysis(*_scene(project), [LAM])[LAM]
    assert gains.closed_return_screen == "POTENTIAL_RETURN" and not [p for p in gains.pairs if p.returns == math.inf]
    path = " -> ".join(gains.closed_cycles[0])
    result = run_snapshot(project)
    messages = [t.message for t in result.targets if t.target_id in ("side-power", "side-coh")]
    assert len(messages) == 2
    # (nothing bounds the left-out light's returns there: more than the negligible level may arrive, the targets undecided)
    assert all(t.assessment.value == "INDETERMINATE" for t in result.targets if t.target_id in ("side-power", "side-coh"))
    for message in messages:
        assert f"a closed path through {path} may circulate it" in message
        assert "two surfaces facing each other" not in message


def test_a_parallel_stack_is_recorded_as_a_stack_with_its_series():
    """A stack of parallel planes is not searched for trails; its record says so (``PARALLEL_STACK``), names the
    stack's surfaces as a group (``parallel_stack``, not a path) and the series that bounds them where one applies."""
    from os_optics.validation.pipeline import return_analysis_records  # noqa: PLC0415
    import test_closed_return_stacks as stacks  # noqa: PLC0415

    for project, bounded in ((stacks.windows(3), True), (stacks.window_before_mirror(), True), (stacks.coupled_cavity(), False)):
        analysis = resonance.ReturnAnalysis(*_scene(project), [LAM])
        analysis[LAM]
        (record,) = return_analysis_records(analysis)
        assert record.closed_return_screen == "POTENTIAL_RETURN" and record.termination == "PARALLEL_STACK"
        assert record.closed_return_kinds == ("parallel_stack",) * len(record.closed_return_surfaces)
        assert (record.closed_series_bound is not None) == bounded
