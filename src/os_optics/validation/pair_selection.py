"""Which beam pairs on an observation plane the two-beam analysis lists (``ValidationResult.interference``).

Two beams of the same wavelength from mutually incoherent sources (different coherence groups) form no lasting fringe:
their irradiances add, and the analysis of such a pair states only that (``INCOHERENT_SOURCES``). Listing every such
pair grows with the square of the beams on a plane -- 195 000 pairs and 300 MB of run files for 625 independent sources
-- so they are counted per plane instead (``IncoherentPairs`` and one ``INCOHERENT_SOURCES`` issue), and the pairs
listed and analysed one by one are:

* every pair of the same coherence group (the pairs that can interfere), as before;
* every pair whose beams arrive in media of different index (its analysis reports ``MEDIUM_MISMATCH``);
* every pair a target on a pair metric of the plane selects (``target_selects_pair``: the selection the target's
  evaluation applies), so such a target reads the same pair analysis as before -- a fringe visibility of 0 for two
  independent sources, for example.

The listed pairs keep the order of every pair of the plane's beams, ``(i, j)`` with ``i < j`` in the plane's arrival
order, and they alone count against ``max_interference_pairs``. Nothing here changes a pair's analysis.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..config import NumericalConfig

#: A pair-metric target's selection: (beam a, beam b) -> whether the target selects that pair.
Selector = Callable[[Any, Any], bool]


def target_selects_pair(t: Any, source_a: str, source_b: str, wavelength_nm: float, path_a: str, crossing_a: int,
                        path_b: str, crossing_b: int, cfg: NumericalConfig) -> bool:
    """Whether the pair-metric target ``t`` selects the pair of beams with these sources, paths and crossings (beam
    ``a``'s wavelength) on its observation plane: its ``source_id`` / ``source_id_b`` (one named: both beams of that
    source; two: one beam of each), its ``wavelength_nm``, and its ``path_id_a`` / ``path_id_b`` and ``crossing_index_a``
    / ``crossing_index_b`` parameters (both paths named: exactly those two paths, at those crossings when both are given;
    one named: a pair containing it)."""
    if t.source_id is not None or t.source_id_b is not None:
        want = {t.source_id, t.source_id_b} - {None}
        if len(want) == 1:
            if not (source_a in want and source_b in want):
                return False
        elif {source_a, source_b} != want:
            return False
    if t.wavelength_nm is not None and not abs(wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm:
        return False
    pa, pb = t.parameters.get("path_id_a"), t.parameters.get("path_id_b")
    if pa is not None and pb is not None:
        if {path_a, path_b} != {pa, pb}:
            return False
        ca, cb = t.parameters.get("crossing_index_a"), t.parameters.get("crossing_index_b")
        if ca is not None and cb is not None and {(path_a, crossing_a), (path_b, crossing_b)} != {(pa, int(ca)), (pb, int(cb))}:
            return False
    elif pa is not None or pb is not None:
        if (pa if pa is not None else pb) not in (path_a, path_b):
            return False
    return True


def _selects(t: Any, a: Any, b: Any, cfg: NumericalConfig) -> bool:
    return target_selects_pair(t, a.source_id, b.source_id, a.wavelength_nm, a.path_id, a.crossing_index, b.path_id,
                               b.crossing_index, cfg)


def _named(t: Any) -> bool:
    """The target names a source or a path: the pairs it selects involve only the beams so named."""
    return (t.source_id is not None or t.source_id_b is not None
            or t.parameters.get("path_id_a") is not None or t.parameters.get("path_id_b") is not None)


def _candidates(t: Any, recs: Sequence[Any]) -> tuple[list[int], bool]:
    """The beams a naming target's pairs involve, and whether both beams of a pair must be among them (sources named)
    or one (a path named)."""
    if t.source_id is not None or t.source_id_b is not None:
        want = {t.source_id, t.source_id_b} - {None}
        return [k for k, r in enumerate(recs) if r.source_id in want], True
    paths = {t.parameters.get("path_id_a"), t.parameters.get("path_id_b")} - {None}
    return [k for k, r in enumerate(recs) if r.path_id in paths], False


def same_wavelength_pair_count(wavelengths: Sequence[float], cfg: NumericalConfig) -> int:
    """The number of pairs ``i < j`` with ``|w_i - w_j| <= interference_wavelength_tol_nm`` (exact integers: the beams of
    one wavelength value pair among themselves, two values within the tolerance pair across)."""
    counts: dict[float, int] = {}
    for w in wavelengths:
        counts[w] = counts.get(w, 0) + 1
    values = sorted(counts)
    total = 0
    for k, v in enumerate(values):
        n = counts[v]
        total += n * (n - 1) // 2
        for w in values[k + 1:]:
            if abs(v - w) <= cfg.interference_wavelength_tol_nm:
                total += n * counts[w]
    return total


def plane_pairs(recs: Sequence[Any], targets: Sequence[Any], cfg: NumericalConfig) -> Iterator[tuple[Any, Any]]:
    """The pairs of the plane's beams ``recs`` (in arrival order) the analysis lists (module docstring), lazily in
    ``(i, j)`` order. ``targets``: the pair-metric targets on this plane."""
    tol = cfg.interference_wavelength_tol_nm
    n = len(recs)

    def same(i: int, j: int) -> bool:
        return abs(recs[i].wavelength_nm - recs[j].wavelength_nm) <= tol

    indices = [r.n for r in recs]
    mixed_media = bool(recs) and max(indices) - min(indices) > cfg.energy_conservation_tol
    if mixed_media or any(not _named(t) for t in targets):
        # (beams in media of different index, or a target that may select any pair: every pair is tested)
        def listed(i: int, j: int) -> bool:
            a, b = recs[i], recs[j]
            return (a.coherence_group == b.coherence_group or abs(a.n - b.n) > cfg.energy_conservation_tol
                    or any(_selects(t, a, b, cfg) for t in targets))

        return ((recs[i], recs[j]) for i in range(n) for j in range(i + 1, n) if same(i, j) and listed(i, j))
    members: dict[str, list[int]] = {}
    position: list[int] = []  # (each beam's place in its group's list)
    for k, r in enumerate(recs):
        group = members.setdefault(r.coherence_group, [])
        position.append(len(group))
        group.append(k)
    extra: dict[int, set[int]] = {}  # (pairs of different groups a target selects: i -> its partners j > i)
    for t in targets:
        chosen, both = _candidates(t, recs)
        others = chosen if both else range(n)
        for i in chosen:
            for j in others:
                lo, hi = (i, j) if i < j else (j, i)
                if lo != hi and recs[lo].coherence_group != recs[hi].coherence_group and _selects(t, recs[lo], recs[hi], cfg):
                    extra.setdefault(lo, set()).add(hi)

    def walk() -> Iterator[tuple[Any, Any]]:
        for i in range(n):
            after = members[recs[i].coherence_group][position[i] + 1:]
            for j in heapq.merge(after, sorted(extra.get(i, ()))):
                if same(i, j):
                    yield recs[i], recs[j]

    return walk()


__all__ = ["plane_pairs", "same_wavelength_pair_count", "target_selects_pair"]
