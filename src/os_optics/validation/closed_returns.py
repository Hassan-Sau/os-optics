"""Screen specular closed returns that the two-surface exposure bound does not cover.

Vertices are *directed legs*, not mirrors. A transition (a,b)->(b,c) exists
only if reflection of the aperture-to-aperture cone at b can meet c. Merely
having three reflectors therefore does not make an open folded path a cavity.
Strongly connected components spanning three surfaces include rings and
coupled pairs. The screen is conservative: overlapping cones establish a
possible return, not a solved cavity mode or a circulating power.

Plane interfaces participate in inter-element loops and their coefficients
include total internal reflection. Detected nonparallel inter-element loops
never receive the finite open-path bound.

A stack of parallel planes in which every reflecting surface is one of the
planes gets an all-orders series that includes its closed returns: the
sufficient ``1 / (1 - A)`` where ``A = sum m_s r_s < 1``, else the alternating
series (``_alternating_series``) where it converges and no thin lens can turn
the light. A stack neither series bounds has its bound withheld, like a ring,
and so has a stack of parallel planes closed by a curved mirror. A stack is
not searched for trails: its record names its groups of surfaces (not a path)
with termination ``PARALLEL_STACK`` and the series where one bounds them.

Closed paths inside one optical solid are not searched here (groups of one owner are skipped). A plane-faced solid's
internal folds (a prism's or a cube's, total internal reflection there included) are bounded by
``validation/solid_folds.py``: their certified closed walks add to the open paths' count (``ReturnGains.loops``), and a fold
solid without a certified sum joins this screen's cycles with no series (``resonance._gains_at``: the screen's answer is
then ``POTENTIAL_RETURN``, never ``COMPLETED_NO_RETURN``; a screen that reached its work limit stays ``INCOMPLETE``). Not
screened (their surfaces keep the pair and open-path bounds): solids with a crystal body, non-convex or curved faces, the
bodies of lenses and other curved refracting surfaces, and parallel planes combined with curved refracting surfaces. This
is not a general refractive trapping solver.

Search. Between the surfaces of separate elements the screen looks for one closed trail of directed legs (no directed
leg twice; a surface may be met again, as a fold mirror is on a folded return) that the narrowed cone can follow around
twice (``possible``). It examines trails shortest first (iterative deepening: every trail of 4 surfaces, then 5, ...)
and stops at the first that closes: the shortest closed return the screen finds, named in the issues. Its storage is one
frame per step of the trail being extended (``ClosedReturnSearch.frames_limit``: at most one per directed leg between
the screened surfaces), never one copy of a trail per branch.

Work budget. The number of trails can grow exponentially with the surfaces of a large scene. The screen counts one unit
per cone test (``validation/return_work.py``: each test of a reflected cone against a leg -- a candidate transition, a
step of a trail or of its closure check); a screen that reaches its limit stops with no cycle and no series: whether a
closed return exists is then not known (``ReturnGains.closed_return_screen`` ``INCOMPLETE``), which bounds nothing
(``ReturnGains.crossings``: inf), never the open-path count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .return_work import (BUDGET_EXHAUSTED, CLOSED_RETURN_STAGE, PARALLEL_STACK, RETURN_FOUND, SEARCH_COMPLETED,
                          ClosedReturnSearch, OutOfReturnWork, WorkCounter)

if TYPE_CHECKING:
    from .resonance import _Scene


def closed_returns(sc: _Scene) -> tuple[tuple[tuple[str, ...], ...], float | None]:
    """The closed returns of the scene's reflecting surfaces at one wavelength (module docstring): the groups or cycles
    found, and the all-orders series where one bounds them (``None``: none does). ``sc.work``: the analysis' work budget
    (``None``: none); a screen that reaches it returns no cycle and no series (``sc.work.record().exhausted``). What its
    trail search did is left in ``sc.search`` (``ClosedReturnSearch``)."""
    counter = sc.work if sc.work is not None else WorkCounter()
    sc.stage = CLOSED_RETURN_STAGE
    search = _Search()
    try:
        found = _closed_returns(sc, counter, search)
    except OutOfReturnWork:
        search.termination = BUDGET_EXHAUSTED
        found = ((), None)
    sc.search = ClosedReturnSearch(search.depth, search.frames_peak, search.frames_limit, search.termination,
                                   search.candidates if search.termination != BUDGET_EXHAUSTED else (),
                                   search.cycle_gain if search.termination == RETURN_FOUND else None)
    return found


class _Search:
    """The trail search's record while it runs (frozen into ``ClosedReturnSearch``)."""

    def __init__(self) -> None:
        self.depth = self.frames_peak = self.frames_limit = 0
        self.termination = SEARCH_COMPLETED
        self.candidates: tuple[tuple[str, ...], ...] = ()
        self.cycle_gain: float | None = None


def _closed_returns(sc: _Scene, counter: WorkCounter, search: _Search) -> tuple[tuple[tuple[str, ...], ...], float | None]:
    # Local import keeps resonance's geometry as the single implementation.
    from .resonance import (
        EITHER, REFLECTING, _HALF_PI, _cone, _critical, _incidence,
        _other, _overlap, _reflected, _sides,
    )

    candidates = [i for i, kind in enumerate(sc.kinds) if kind in REFLECTING
                  and (sc.faces[i].plane or kind != "refract")]
    if len(candidates) < 3:
        return (), None

    def reflectance(b, incoming):
        low, high = _incidence(incoming, sc.faces[b])
        r = sc.coefficients(b, incoming.side, low, high)[0]
        # The pair calculation deliberately excludes TIR from its geometric
        # series. The cycle screen must include that reflected branch.
        if sc.kinds[b] == "refract":
            for side in _sides(incoming.side):
                # (a crystal's side: the smallest critical angle over its indices, its largest index into the other's least)
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
        face = sc.faces[b]
        reflected = _cone(_reflected(incoming.u, face.axis),
                          incoming.reach + 2.0 * face.cone, incoming.side)
        for c in candidates:
            if c == b:
                continue
            toward = sc.toward(b, c)
            if incoming.side != EITHER and toward != EITHER and incoming.side != toward:
                continue
            counter.spend(1, CLOSED_RETURN_STAGE)
            if _overlap(reflected, sc.leg(b, c)):
                targets.append(c)

    # Iterative Kosaraju: no recursion depth depends on customer scene size.
    seen, order = set(), []
    for start in edges:
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(edges[start]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(node)
                stack.pop()
                continue
            child = (node[1], child)
            if child not in seen:
                seen.add(child)
                stack.append((child, iter(edges[child])))
    reverse = {node: [] for node in edges}
    for node, children in edges.items():
        for child in children:
            reverse[(node[1], child)].append(node)
    seen.clear()
    loops = []
    for start in reversed(order):
        if start in seen:
            continue
        seen.add(start)
        stack, surfaces = [start], set()
        while stack:
            node = stack.pop()
            surfaces.update(node)
            for child in reverse[node]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        if len(surfaces) >= 3:
            loops.append(tuple(sc.surfaces[i].surface_id for i in sorted(surfaces)))
    groups = tuple(sorted(set(loops)))
    if not groups:
        return (), None

    # A full all-orders amplitude series is justified without geometrical
    # pruning where each transmission is straight (or all interfaces parallel):
    # each plane is met at most once between reflections and transmission has
    # norm <= 1. If A=sum r_s<1, sum A**j=1/(1-A) includes every return sequence.
    # This is intentionally a sufficient bound, never a convergence heuristic.
    parallel = len(set(sc.classes.values())) <= 1
    total = 0.0
    for i in candidates:
        meetings = 1 if sc.met_once(i) else len(sc.turning) + 1
        if parallel and i in sc.classes:
            momenta = sc.momenta(i).get(i)
            if momenta is None:
                continue
            total += meetings * max(_plane_bound(sc, i, side, momenta) for side in _sides(EITHER))
        else:
            # Without a common conserved momentum only the full angular range
            # is justified for the all-orders series.
            total += meetings * sc.coefficients(i, EITHER, 0.0, _HALF_PI)[0]
    plane_reflectors = all(sc.faces[i].plane for i, kind in enumerate(sc.kinds) if kind in REFLECTING)
    if parallel and plane_reflectors:
        # (no trail search: the groups named are the stack's surfaces, bounded by the series where one applies)
        search.termination = PARALLEL_STACK
        if total < 1.0:
            return groups, 1.0 / (1.0 - total)
        # A stack of parallel planes whose reflections the sufficient series
        # above does not bound (A >= 1: strong coatings, a mirror behind a
        # window, three or more bare windows): the pair and open-path counts do
        # not cover its chained returns. It gets the alternating series where
        # that converges; otherwise its bound is withheld (``None``).
        return groups, _alternating_series(sc, [i for i in candidates if i in sc.classes])
    if parallel:
        # The inter-element ring screen below needs nonparallel reflection
        # directions. Parallel planes closed by a curved mirror (a linear
        # cavity with a window or a plate inside) chain their returns through
        # the mirror, which no series here bounds: withheld. With curved
        # refracting surfaces (lens trains with flat faces) they retain the pair
        # and open-path treatment; independent aperture cones cannot decide
        # their correlated multi-pass transverse walk.
        if not any(kind == "refract" and not sc.faces[i].plane for i, kind in enumerate(sc.kinds)):
            search.termination = PARALLEL_STACK
            return groups, None
        return (), None

    # A graph of independently widened legs can invent a turn at a nearby
    # aperture (a window's two bounding balls overlap). Carry the narrower
    # incoming cone around each directed-leg loop repeatedly before declaring it
    # possible. A plane reflection preserves its width, so a narrow leg cannot
    # silently become an arbitrary direction at the next overlapping aperture.
    def advance(current, b, c):
        counter.spend(1, CLOSED_RETURN_STAGE)
        return carried(current, b, c)

    def carried(current, b, c):
        face = sc.faces[b]
        image = _cone(_reflected(current.u, face.axis),
                      current.reach + 2 * face.cone, current.side)
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

    # An optical solid's internal faces share aperture edges. Independent
    # aperture cones lose the correlation between entry position and
    # direction there and cannot establish internal trapping. Those bodies
    # retain the existing treatment; this screen covers returns between
    # separate optical elements, not an internal prism/cube mode solver.
    screened = []
    for group in groups:
        members = [i for i in candidates if sc.surfaces[i].surface_id in group]
        if len({sc.surfaces[i].owner_id for i in members}) >= 2:
            screened.append(members)
    # (a trail never repeats a directed leg: it holds at most one step per directed leg between its group's surfaces)
    most = max((len(m) * (len(m) - 1) + 1 for m in screened), default=0)
    search.frames_limit = most
    search.candidates = tuple(tuple(sc.surfaces[i].surface_id for i in members) for members in screened)
    found = _shortest_trail(sc, screened, edges, advance, possible, most, search)
    if found is None:
        return (), None

    def round_gain(path):
        """The amplitude gain bound of one round of the closed trail: the product over its steps of the reflectance bound
        of each surface over the cone that arrives at it, the cones carried as ``possible`` carries them on its first
        round (the widest it accepts: later rounds only narrow them, so their product is no larger). One pass over the
        trail the search already carried its cones around: it is not counted as search work, so the work of a screen and
        the limit at which it stops are what they are without it."""
        current = sc.leg(path[-1], path[0])
        gain = 1.0
        for k, b in enumerate(path):
            gain *= reflectance(b, current)
            current = carried(current, b, path[(k + 1) % len(path)])
            if current is None:  # (``possible`` carried it around: not reached)
                return None
        return gain

    search.cycle_gain = round_gain(found)
    search.termination = RETURN_FOUND
    return (tuple(sc.surfaces[i].surface_id for i in found),), None


def _shortest_trail(sc: _Scene, screened: list[list[int]], edges: dict[tuple[int, int], list[int]], advance, possible,
                    most: int, search: _Search) -> tuple[int, ...] | None:
    """The first closed trail of directed legs, shortest first, whose narrowed cone the closure check carries around it
    (``possible``) through surfaces of two or more elements, three or more surfaces: ``None`` when every trail of the
    screened groups was examined and none closes.

    Iterative deepening: every trail of at most ``depth`` surfaces (its return to the first counted), ``depth`` = 4, 5, ...
    until a closed return is found or no trail was cut at ``depth`` (every trail examined). A trail starts on a directed
    leg ``(start, second)`` and closes when it would take that leg again; it never repeats a directed leg, but it may meet
    a surface again (a folded return passes a fold mirror twice, a ring repeats none). The search holds one frame per step
    of the trail being extended -- its leg's cone and the position among the leg's transitions --, so its storage is at
    most ``most`` frames (the directed legs between a group's surfaces, plus one), whatever the number of trails; each
    frame costs one work unit (``advance``) before it is held, and the work budget bounds the trails examined."""
    for depth in range(4, most + 2):
        search.depth = depth
        cut = False
        for members in screened:
            member_set = set(members)
            for start in members:
                for second in members:
                    if second == start:
                        continue
                    found, cut_here = _trails(sc, start, second, member_set, edges, advance, possible, depth, search)
                    if found is not None:
                        return found
                    cut = cut or cut_here
        if not cut:
            return None
    return None


def _trails(sc: _Scene, start: int, second: int, member_set: set[int], edges: dict[tuple[int, int], list[int]], advance,
            possible, depth: int, search: _Search) -> tuple[tuple[int, ...] | None, bool]:
    """The trails of at most ``depth`` surfaces that begin with the leg ``(start, second)`` (``_shortest_trail``): the first
    that closes, and whether a trail was cut at ``depth``. Depth-first with one frame per step (no copy of the trail per
    branch): the path, its legs and its surfaces' and owners' counts change by one step at a time."""
    owner = [s.owner_id for s in sc.surfaces]
    path = [start, second]
    legs = {(start, second)}
    surfaces = {start: 1, second: 1} if start != second else {start: 2}
    owners: dict[str, int] = {}
    for i in path:
        owners[owner[i]] = owners.get(owner[i], 0) + 1
    frames = [(sc.leg(start, second), iter(edges[(start, second)]))]
    search.frames_peak = max(search.frames_peak, len(frames))
    cut = False
    while frames:
        incoming, children = frames[-1]
        here = path[-1]
        nxt = next(children, None)
        if nxt is None:
            frames.pop()
            gone = path.pop()
            if path:
                legs.discard((path[-1], gone))
            surfaces[gone] -= 1
            if not surfaces[gone]:
                del surfaces[gone]
            owners[owner[gone]] -= 1
            if not owners[owner[gone]]:
                del owners[owner[gone]]
            continue
        # Carry the same conservative cone as the final closure check
        # while extending the walk. Widening it independently at every
        # aperture would enumerate enormous numbers of impossible trails
        # inside cube splitters before checking their geometry.
        outgoing = advance(incoming, here, nxt)
        if outgoing is None:
            continue
        if here == start and nxt == second and len(surfaces) >= 3 and len(owners) >= 2 and possible(path[:-1]):
            return tuple(path[:-1]), cut
        # A folded return revisits a fold mirror on its other
        # leg. Keep it; only repeating the same directed leg
        # would add a redundant subcycle to this search.
        if (here, nxt) in legs or nxt not in member_set:
            continue
        if len(path) >= depth:
            cut = True
            continue
        path.append(nxt)
        legs.add((here, nxt))
        surfaces[nxt] = surfaces.get(nxt, 0) + 1
        owners[owner[nxt]] = owners.get(owner[nxt], 0) + 1
        frames.append((outgoing, iter(edges[(here, nxt)])))
        if len(frames) > search.frames_peak:
            search.frames_peak = len(frames)
    return None, cut


def _plane_bound(sc: _Scene, i: int, side: int, momenta: tuple[float, float]) -> float:
    """The amplitude reflectance bound of plane ``i`` met from ``side`` by light of the class's momenta there: its
    coefficient over their angles of incidence; 1 where a refracting plane may reflect it totally -- a momentum along the
    planes above the index of its other side (light that entered the stack from that side has at most that momentum and
    meets the plane at most at the critical angle, where its coefficient holds) -- or an index is not known."""
    from .resonance import _angles, _other, _plane_indices

    n, n_least = _plane_indices(sc, i, side)
    low = _angles([momenta[0]], n)[0]
    high = _angles([momenta[1]], n_least)[0]
    bound = sc.coefficients(i, side, low, high)[0]
    if sc.kinds[i] == "refract":
        other = _plane_indices(sc, i, _other(side))[1]
        if n is None or other is None or not momenta[1] <= other:
            bound = 1.0
    return bound


#: Rounds of the iterations of ``_alternating_series`` (each a pass over the stack's transitions).
ALTERNATING_ROUNDS = 512
#: The relative margin the series' certificate keeps over its iterate (far above the rounding of its sums).
ALTERNATING_MARGIN = 1e-9


def _alternating_series(sc: _Scene, members: list[int]) -> float | None:
    """The all-orders amplitude series of the left-out light in a stack of parallel planes, where every surface that
    redirects light is one of them (``None``: it does not converge, or a thin lens may turn the light).

    Between two reflections light crosses parallel planes only (refraction there keeps the sign of its direction along
    their normal), so its depth along the normal changes monotonically: after a reflection at plane ``s`` it meets next
    only planes strictly ahead of ``s`` in its new direction, each at most once, and the next reflection reverses that
    direction. A state is (the plane that reflected it, its direction); the transition to (``t``, the opposite direction)
    carries ``t``'s amplitude reflectance bound for light arriving from that side (``_plane_bound``), transmissions at most
    1. The left-out light starts as a piece of amplitude 1 moving either way: its first reflections are at planes ahead,
    so the start vector ``w0`` holds every plane's bound from one side, the larger total of the two directions counting.
    Its total amplitude over every reflection sequence, closed ones included, is ``1 + sum_k |W^k w0|`` (``W`` the
    transitions, all entries non-negative). The sum is certified, not estimated: ``q`` with ``W q <= lam q`` (``lam < 1``,
    shifted power iteration), the partial sum ``u`` after ``ALTERNATING_ROUNDS`` rounds, and ``u2 = (1 + margin) u + 2 eps
    q`` with ``eps`` the largest residual of ``u`` over ``(1 - lam) q`` (at least the margin times the largest entry of
    ``u``); then ``w0 + W u2 <= u2`` holds row by row (checked in the computed values), so every partial sum stays below
    ``u2`` and the series is at most ``sum u2``. A stack whose
    returns do not converge (two strong reflectors facing each other through the others, a coupled pair of cavities) has
    no such certificate: ``None``. Plain sums and products in a fixed order: the same bits in every runtime."""
    from .resonance import BENDING, NEGATIVE, POSITIVE, _dot

    if any(kind in BENDING for kind in sc.kinds):
        return None
    axis = sc.faces[members[0]].axis
    depth = [_dot(sc.faces[i].centre, axis) for i in members]
    # (per plane, the bound for light moving along +axis and along -axis: light moving with a plane's normal meets it from
    # its negative side)
    weight: list[tuple[float, float]] = []
    for i in members:
        momenta = sc.momenta(i).get(i)
        if momenta is None:
            weight.append((0.0, 0.0))  # (no light of the stack meets it)
            continue
        along = _dot(sc.faces[i].axis, axis) > 0.0
        up = _plane_bound(sc, i, NEGATIVE if along else POSITIVE, momenta)
        down = _plane_bound(sc, i, POSITIVE if along else NEGATIVE, momenta)
        weight.append((up, down))
    count = len(members)
    # state 2 k: reflected by plane k, now moving along +axis; 2 k + 1: moving along -axis
    out: list[list[tuple[int, float]]] = []
    for k in range(2 * count):
        s, up = k // 2, k % 2 == 0
        row = []
        for t in range(count):
            if (depth[t] > depth[s]) if up else (depth[t] < depth[s]):
                w = weight[t][0 if up else 1]
                if w > 0.0:
                    row.append((2 * t + (1 if up else 0), w))
        out.append(row)
    states = 2 * count

    def apply(x: list[float]) -> list[float]:
        y = [0.0] * states
        for k in range(states):
            a = x[k]
            if a != 0.0:
                for target, w in out[k]:
                    y[target] += w * a
        return y

    q = [1.0] * states
    for _ in range(ALTERNATING_ROUNDS):
        moved = apply(q)
        q = [a + b for a, b in zip(q, moved)]
        top = max(q)
        q = [a / top for a in q]
    wq = apply(q)
    lam = 0.0
    for a, b in zip(wq, q):
        if a > 0.0:
            if not b > 0.0:
                return None  # (no certificate where the iteration lost a state that light reaches)
            lam = max(lam, a / b)
    if not lam < 1.0:
        return None
    best = 0.0
    for up in (True, False):
        w0 = [0.0] * states
        for t in range(count):
            w0[2 * t + (1 if up else 0)] = weight[t][0 if up else 1]
        u = list(w0)
        for _ in range(ALTERNATING_ROUNDS):
            u = [a + b for a, b in zip(w0, apply(u))]
        residual = [a + b - c for a, b, c in zip(w0, apply(u), u)]
        # (at least the margin of the largest partial sum: a converged iterate leaves no residual, and the check needs room
        # above the rounding of its sums)
        eps = ALTERNATING_MARGIN * max(u)
        for r, b in zip(residual, q):
            if r > 0.0:
                if not b > 0.0:
                    return None
                eps = max(eps, r / ((1.0 - lam) * b))
        u2 = [(1.0 + ALTERNATING_MARGIN) * a + 2.0 * eps * b for a, b in zip(u, q)]
        if any(a + b > c for a, b, c in zip(w0, apply(u2), u2)):
            return None
        total = 0.0
        for a in u2:  # (plain additions in order: ``sum`` is compensated on some Pythons only)
            total += a
        best = max(best, total)
    return 1.0 + best
