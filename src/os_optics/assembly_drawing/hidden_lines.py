"""Hidden-line classification of posed triangle meshes in one orthographic view (pure Python, deterministic).

Which edges are drawn (per part, per view):

* **feature edges** -- welded edges whose two faces meet at more than 30 deg (box edges, cylinder cap rims, steps);
* **boundary / non-manifold edges** -- edges with one face or more than two (where two shells share an edge);
* **silhouette edges** -- smooth edges between a face turned towards the viewer and one turned away (the outline of
  a faceted cylinder seen from the side).

Edges that project onto the same paper segment within one part are drawn once (the nearest).

How visible they are: every triangle of every part is projected into the view once, with its depth plane and whether a
view ray enters (faces the viewer) or leaves the solid there, and bucketed in a uniform 2-D grid. A point of an edge at
paper position (u, v) and depth d is classified by casting the view ray through (u, v) against the triangles of its
grid cell (edge functions with a fixed tie rule, as in the assembly checker's column casts, so a ray through a shared
edge or vertex is counted once):

* DROP -- the point lies strictly inside ANOTHER shell of the same part (the winding number of that part's other shells
  is positive there and no other-shell surface passes through the point): an internal edge of overlapping mesh shells,
  not a surface edge of the part;
* HIDDEN -- some surface lies in front of it (depth < d - ``DEPTH_EPS_MM``);
* VISIBLE -- otherwise.

An edge is sampled at a fixed paper spacing and every change of state is located by bisection, so visible and hidden
runs split where the occluding outline crosses the edge (to a small fraction of the sample spacing).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .geometry import FEATURE_COS, PosedMesh, View, dot, norm
from ..assembly.transforms import Vec3

V2 = tuple[float, float]

VISIBLE, HIDDEN, DROP = 0, 1, 2
ENTER, LEAVE = 1, -1
DEPTH_EPS_MM = 1e-3  # depth agreement of a point and the surface it lies on (float32 meshes: ~1e-5 mm)
GRID_CELLS = 64  # buckets along the longer view extent
MIN_CELL_MM = 0.5
SAMPLE_STEP_PAPER_MM = 0.9  # edge sampling on paper; state changes are then bisected
REFINE_STEPS = 7
DEDUP_DECIMALS = 5
MIN_PROJECTED_MM = 1e-9


def _edge_fn(pu: float, pv: float, qu: float, qv: float, x: float, y: float) -> float:
    """Edge function of directed edge P->Q at (x, y), bit-antisymmetric under P <-> Q."""

    if (pu, pv) <= (qu, qv):
        return (qu - pu) * (y - pv) - (qv - pv) * (x - pu)
    return -((pu - qu) * (y - qv) - (pv - qv) * (x - qu))


def _owns_tie(pu: float, pv: float, qu: float, qv: float) -> bool:
    du, dv = qu - pu, qv - pv
    return dv < 0.0 or (dv == 0.0 and du > 0.0)


@dataclass(frozen=True)
class _Tri:
    corners: tuple[float, float, float, float, float, float]  # counter-clockwise (u, v) x 3
    u0: float
    v0: float
    d0: float
    gu: float
    gv: float
    delta: int
    mesh: int
    shell: int
    ties: tuple[bool, bool, bool]

    def contains(self, x: float, y: float) -> bool:
        au, av, bu, bv, cu, cv = self.corners
        e = _edge_fn(au, av, bu, bv, x, y)
        if e < 0.0 or (e == 0.0 and not self.ties[0]):
            return False
        e = _edge_fn(bu, bv, cu, cv, x, y)
        if e < 0.0 or (e == 0.0 and not self.ties[1]):
            return False
        e = _edge_fn(cu, cv, au, av, x, y)
        return not (e < 0.0 or (e == 0.0 and not self.ties[2]))

    def depth_at(self, x: float, y: float) -> float:
        return self.d0 + self.gu * (x - self.u0) + self.gv * (y - self.v0)


@dataclass(frozen=True)
class Segment3:
    """A drawable edge of one part: world endpoints, its shell."""

    a: Vec3
    b: Vec3
    shell: int


class ViewScene:
    """All posed meshes of one view, projected and bucketed for ray queries."""

    def __init__(self, view: View, meshes: Sequence[PosedMesh]):
        self.view = view
        self.meshes = tuple(meshes)
        self.glass = tuple(m.glass for m in self.meshes)
        tris: list[_Tri] = []
        f = view.forward
        for mi, mesh in enumerate(self.meshes):
            uvd = [(dot(p, view.right), dot(p, view.up), dot(p, f)) for p in mesh.vertices]
            for ti, (ia, ib, ic) in enumerate(mesh.triangles):
                a, b, c = uvd[ia], uvd[ib], uvd[ic]
                d1u, d1v, d1d = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                d2u, d2v, d2d = c[0] - a[0], c[1] - a[1], c[2] - a[2]
                area = d1u * d2v - d1v * d2u
                if area == 0.0:
                    continue  # edge-on: never crossed by a view ray
                gu = (d1d * d2v - d2d * d1v) / area
                gv = (d2d * d1u - d1d * d2u) / area
                pts = (a, b, c) if area > 0.0 else (a, c, b)
                corners = (pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1])
                ties = (_owns_tie(corners[0], corners[1], corners[2], corners[3]),
                        _owns_tie(corners[2], corners[3], corners[4], corners[5]),
                        _owns_tie(corners[4], corners[5], corners[0], corners[1]))
                # (u, v, depth) is left-handed: counter-clockwise on paper means the face looks at the viewer.
                tris.append(_Tri(corners, a[0], a[1], a[2], gu, gv, ENTER if area > 0.0 else LEAVE, mi, mesh.shell_of[ti], ties))
        self.tris = tuple(tris)
        us = [c for t in tris for c in t.corners[0::2]] or [0.0]
        vs = [c for t in tris for c in t.corners[1::2]] or [0.0]
        self.u_min, self.v_min = min(us), min(vs)
        extent = max(max(us) - self.u_min, max(vs) - self.v_min)
        self.cell = max(extent / GRID_CELLS, MIN_CELL_MM)
        cells: dict[tuple[int, int], list[int]] = {}
        for index, t in enumerate(tris):
            cu = t.corners[0::2]
            cv = t.corners[1::2]
            i0, i1 = self._index(min(cu), self.u_min), self._index(max(cu), self.u_min)
            j0, j1 = self._index(min(cv), self.v_min), self._index(max(cv), self.v_min)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cells.setdefault((i, j), []).append(index)
        self.cells = {key: tuple(value) for key, value in cells.items()}

    def _index(self, x: float, origin: float) -> int:
        return math.floor((x - origin) / self.cell)

    # ------------------------------------------------------------------ ray queries

    def hits(self, u: float, v: float) -> list[tuple[float, int, int, int]]:
        """(depth, ENTER/LEAVE, mesh index, shell) of every surface the view ray through (u, v) crosses."""

        bucket = self.cells.get((self._index(u, self.u_min), self._index(v, self.v_min)), ())
        out = []
        for index in bucket:
            t = self.tris[index]
            if t.contains(u, v):
                out.append((t.depth_at(u, v), t.delta, t.mesh, t.shell))
        return out

    def state(self, u: float, v: float, d: float, mesh: int, shell: int, *, through_glass: bool = False) -> int:
        hidden = False
        winding = 0
        on_other = False
        for depth, delta, m, s in self.hits(u, v):
            if through_glass and self.glass[m]:
                continue
            front = depth < d - DEPTH_EPS_MM
            if front:
                hidden = True
            if m == mesh and s != shell:
                if front:
                    winding += delta
                elif depth <= d + DEPTH_EPS_MM:
                    on_other = True
        if mesh >= 0 and winding > 0 and not on_other:
            return DROP
        return HIDDEN if hidden else VISIBLE

    def first_mesh(self, u: float, v: float) -> tuple[int, float] | None:
        """The mesh whose surface the view ray through (u, v) meets first, and that depth."""

        best: tuple[float, int] | None = None
        for depth, _, m, _ in self.hits(u, v):
            if best is None or (depth, m) < best:
                best = (depth, m)
        return None if best is None else (best[1], best[0])

    def meshes_at(self, u: float, v: float) -> set[int]:
        return {m for _, _, m, _ in self.hits(u, v)}

    # ------------------------------------------------------------------ edges

    def drawable_edges(self, mesh_index: int) -> list[Segment3]:
        """Feature, boundary and silhouette edges of one mesh in this view (duplicates on paper drawn once)."""

        mesh = self.meshes[mesh_index]
        f = self.view.forward
        best: dict[tuple[tuple[float, float], tuple[float, float]], tuple[float, Segment3]] = {}
        order: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for (i, j), faces in mesh.edges:
            if len(faces) == 2:
                n1, n2 = mesh.normals[faces[0]], mesh.normals[faces[1]]
                crease = dot(n1, n2) < FEATURE_COS * norm(n1) * norm(n2)
                if not crease and (dot(n1, f) < 0.0) == (dot(n2, f) < 0.0):
                    continue
            a, b = mesh.vertices[i], mesh.vertices[j]
            pa, pb = self.view.uv(a), self.view.uv(b)
            if abs(pb[0] - pa[0]) + abs(pb[1] - pa[1]) < MIN_PROJECTED_MM:
                continue
            ka = (round(pa[0], DEDUP_DECIMALS), round(pa[1], DEDUP_DECIMALS))
            kb = (round(pb[0], DEDUP_DECIMALS), round(pb[1], DEDUP_DECIMALS))
            key = (ka, kb) if ka <= kb else (kb, ka)
            depth = self.view.depth(a) + self.view.depth(b)
            if key not in best:
                order.append(key)
                best[key] = (depth, Segment3(a, b, mesh.shell_of[faces[0]]))
            elif depth < best[key][0]:
                best[key] = (depth, Segment3(a, b, mesh.shell_of[faces[0]]))
        return [best[key][1] for key in order]

    def classify(self, a: Vec3, b: Vec3, mesh: int, shell: int, paper_scale: float, *,
                 through_glass: bool = False) -> list[tuple[int, list[V2]]]:
        """Visible / hidden / dropped runs of the world segment a-b (view uv coordinates)."""

        view = self.view
        pa, pb = view.uv(a), view.uv(b)
        da, db = view.depth(a), view.depth(b)
        du, dv, dd = pb[0] - pa[0], pb[1] - pa[1], db - da
        length = math.sqrt(du * du + dv * dv) * paper_scale
        n = max(1, math.ceil(length / SAMPLE_STEP_PAPER_MM))

        def at(t: float) -> int:
            return self.state(pa[0] + du * t, pa[1] + dv * t, da + dd * t, mesh, shell, through_glass=through_glass)

        samples = [(k + 0.5) / n for k in range(n)]
        states = [at(t) for t in samples]
        cuts = [0.0]
        run_states = [states[0]]
        for k in range(1, n):
            if states[k] == states[k - 1]:
                continue
            lo, hi = samples[k - 1], samples[k]
            for _ in range(REFINE_STEPS):
                mid = (lo + hi) / 2.0
                if at(mid) == states[k - 1]:
                    lo = mid
                else:
                    hi = mid
            cuts.append((lo + hi) / 2.0)
            run_states.append(states[k])
        cuts.append(1.0)
        runs = []
        for index, st in enumerate(run_states):
            t0, t1 = cuts[index], cuts[index + 1]
            runs.append((st, [(pa[0] + du * t0, pa[1] + dv * t0), (pa[0] + du * t1, pa[1] + dv * t1)]))
        return runs


def chain(segments: Sequence[list[V2]], decimals: int = DEDUP_DECIMALS) -> list[list[V2]]:
    """Join segments sharing endpoints into polylines (deterministic greedy order) and drop collinear interior points."""

    def key(p: V2) -> tuple[float, float]:
        return (round(p[0], decimals), round(p[1], decimals))

    remaining = [list(s) for s in segments if len(s) >= 2]
    by_start: dict[tuple[float, float], list[int]] = {}
    by_end: dict[tuple[float, float], list[int]] = {}
    for index, seg in enumerate(remaining):
        by_start.setdefault(key(seg[0]), []).append(index)
        by_end.setdefault(key(seg[-1]), []).append(index)
    used = [False] * len(remaining)
    out: list[list[V2]] = []
    for index, seg in enumerate(remaining):
        if used[index]:
            continue
        used[index] = True
        line = list(seg)
        extended = True
        while extended:
            extended = False
            tail = key(line[-1])
            for other in by_start.get(tail, ()):
                if not used[other]:
                    used[other] = True
                    line.extend(remaining[other][1:])
                    extended = True
                    break
            else:
                for other in by_end.get(tail, ()):
                    if not used[other]:
                        used[other] = True
                        line.extend(list(reversed(remaining[other]))[1:])
                        extended = True
                        break
        out.append(simplify(line))
    return out


COLLINEAR_TOLERANCE = 1e-9


def simplify(pts: list[V2]) -> list[V2]:
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        abx, aby, bcx, bcy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
        cr = abx * bcy - aby * bcx
        scale_ab = math.sqrt(abx * abx + aby * aby)
        scale_bc = math.sqrt(bcx * bcx + bcy * bcy)
        if abs(cr) <= COLLINEAR_TOLERANCE * scale_ab * scale_bc + COLLINEAR_TOLERANCE * COLLINEAR_TOLERANCE and abx * bcx + aby * bcy > 0:
            continue
        out.append(b)
    out.append(pts[-1])
    return out
