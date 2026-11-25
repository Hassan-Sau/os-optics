"""R1 of the ring-return census on convex mesh solids (FACET_SOLID; review BND-1): a corner cube, a penta, a rhomboid, a dove and a cube given as
closed triangle meshes go through the internal-fold analysis (``validation/solid_folds.py``) like a plane-faced solid of
surfaces, and the analysis never misses a closed walk the light can make inside them.

Before, the convexity test measured each polygon patch by the bounding disc about one of its corners: every mesh solid
read as non-convex and kept the open-path and pair reading only (its faces met beyond the critical angle read at their
coating's value, the screen ``COMPLETED_NO_RETURN``, a faint corner-cube cavity a finite exposure upper bound).

The reference is independent of the engine's geometry and optics: the solid is rebuilt from its own mesh (triangles grouped
by plane; inside a convex solid the next face a ray meets is the nearest plane ahead it crosses outward), rays enter from the
source's beam by vector Snell, and every face reflects the bare Fresnel amplitude max(|r_s|, |r_p|) -- 1 beyond the critical
angle --, a silvered face 1. Per face, the largest sum over a ray's later reflections there of their amplitudes (and powers)
after any reflection at, or entry through, that face must never exceed the certified ``G - 1`` (``G2 - 1``); a closed walk
through three or more faces must never be reported as no return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from facet_helpers import EXACT, convex_solid, extruded, glass, part, request, silvered, solid
from helpers import run
from test_return_pairs import LAM, _constant, _rating, _scene, _targets

from os_optics import builders as B
from os_optics.contracts import EngineeringComponentSnapshot, MeshFacetGroup, MeshSurface, ProjectSnapshot
from os_optics.contracts.optics import RefractInteraction
from os_optics.validation import resonance as R
from os_optics.validation import solid_folds as F

WAVELENGTH = 633.0
WAIST = 0.1


def _unit(v) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def _laser(at, direction, back: float = 30.0):
    d = _unit(direction)
    return B.laser("S", wavelength_nm=WAVELENGTH, power_W=1e-3, waist_radius_mm=WAIST, position=tuple(np.array(at, float) - back * d),
                   direction=tuple(d))


def _rot_y(v, deg: float):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


# ------------------------------------------------------------------------------------------------ the mesh solids


def corner_cube(tilt=(0.0, 0.0), entry=(9.0, 7.0, 4.0), index: float = 1.5):
    vertices, triangles, _ = convex_solid([(0, 0, 0), (20, 0, 0), (0, 20, 0), (0, 0, 20)], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    d = -_unit((1.0, 1.0, 1.0)) + np.array([tilt[0], tilt[1], -tilt[0] - tilt[1]])
    project = request(_laser(entry, d, back=10.0), part("P", solid("cc", vertices, triangles)), materials=(glass(index),)).project
    return project, (vertices, triangles, (), index)


def penta(tilt_deg: float = 0.0, oop_deg: float = 0.0):
    s, b = 10.0, 4.0
    c, sn = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    a = (s - b / math.sqrt(2.0)) / (c - sn)
    corner_a = (s, 0.0)
    corner_b = (corner_a[0] + a * sn, corner_a[1] + a * c)
    corner_c = (corner_b[0] - b / math.sqrt(2.0), corner_b[1] + b / math.sqrt(2.0))
    vertices, triangles, faces = extruded([(0.0, 0.0), corner_a, corner_b, corner_c, (0.0, s)], 8.0)
    mirrors = tuple(faces[3] + faces[5])
    t, p = math.radians(tilt_deg), math.radians(oop_deg)
    d = (math.sin(t), math.sin(p), math.cos(t) * math.cos(p))
    project = request(_laser((3.0, 0.5, 0.0), d), part("P", solid("penta", vertices, triangles, groups=(silvered(mirrors),)))).project
    return project, (vertices, triangles, mirrors, 1.5)


def _prism(poly_xz, half_y: float, entry, direction, index: float = 1.5):
    vertices, triangles, _ = extruded(poly_xz, half_y)
    project = request(_laser(entry, direction), part("P", solid("prism", vertices, triangles)), materials=(glass(index),)).project
    return project, (vertices, triangles, (), index)


def rhomboid(tilt_deg: float = 0.0):
    h, length = 10.0, 30.0
    return _prism([(0.0, 0.0), (h, 0.0), (h + length, length), (length, length)], 8.0, (h / 2.0, 0.0, 0.0),
                  _rot_y((0.0, 0.0, 1.0), tilt_deg))


def dove(tilt_deg: float = 0.0):
    length, h = 40.0, 8.0
    return _prism([(0.0, 0.0), (length, 0.0), (length - h, h), (h, h)], 6.0, (h / 2.0 - 1.0, 0.0, h / 2.0 + 1.0),
                  _rot_y((1.0, 0.0, 0.0), tilt_deg))


def low_index_cube(direction):
    """A 12 mm cube of index 1.05 (critical angle 72 deg) lit across its corner: every face is met well below the critical
    angle, its faces are squares (two triangles each, one polygon patch), and its returns are lossy."""
    corners = [(x, y, z) for x in (0.0, 12.0) for y in (0.0, 12.0) for z in (0.0, 12.0)]
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    vertices, triangles, _ = convex_solid(corners, faces)
    project = request(_laser((0.0, 5.0, 6.0), direction, back=20.0), part("P", solid("cube", vertices, triangles)),
                      materials=(glass(1.05),)).project
    return project, (vertices, triangles, (), 1.05)


# ------------------------------------------------------------------------------------------------ the reference


class _Convex:
    """A convex solid rebuilt from its mesh: outward unit normals and offsets of its face planes (triangles grouped by plane),
    which are silvered, the body's index (air outside)."""

    def __init__(self, vertices, triangles, mirrors, index: float) -> None:
        pts = np.array(vertices, dtype=float)
        centroid = pts.mean(axis=0)
        normals: list[np.ndarray] = []
        offsets: list[float] = []
        face_of: list[int] = []
        for tri in triangles:
            a, b, c = pts[list(tri)]
            n = np.cross(b - a, c - a)
            n /= np.linalg.norm(n)
            if float(np.dot(n, a - centroid)) < 0.0:
                n = -n
            o = float(np.dot(n, a))
            for k, (m, q) in enumerate(zip(normals, offsets)):
                if float(np.dot(m, n)) > 1.0 - 1e-12 and abs(q - o) < 1e-9:
                    face_of.append(k)
                    break
            else:
                normals.append(n)
                offsets.append(o)
                face_of.append(len(normals) - 1)
        self.normals = np.array(normals)
        self.offsets = np.array(offsets)
        self.mirror = [False] * len(normals)
        for t in mirrors:
            self.mirror[face_of[t]] = True
        self.index = index

    def inside(self, p, tol: float = 1e-9) -> bool:
        return bool(np.all(self.normals @ p - self.offsets <= tol))

    def amplitude(self, k: int, d) -> float:
        if self.mirror[k]:
            return 1.0
        c = abs(float(np.dot(self.normals[k], d)))
        s2 = self.index * self.index * (1.0 - c * c)
        if s2 >= 1.0:
            return 1.0  # (total internal reflection)
        ct = math.sqrt(1.0 - s2)
        rs = (self.index * c - ct) / (self.index * c + ct)
        rp = (c - self.index * ct) / (c + self.index * ct)
        return max(abs(rs), abs(rp))

    def enter(self, p, d):
        """(face, point, inside direction) where a ray from outside enters, or ``None``."""
        den = self.normals @ d
        best = None
        for k in range(len(self.offsets)):
            if den[k] < -1e-12:
                t = (self.offsets[k] - float(np.dot(self.normals[k], p))) / den[k]
                if t > 0.0 and (best is None or t > best[0]):
                    best = (t, k)
        if best is None:
            return None
        t, k = best
        q = p + t * d
        if not self.inside(q, 1e-7) or self.mirror[k]:
            return None
        n = self.normals[k]
        cos_i = -float(np.dot(d, n))
        eta = 1.0 / self.index
        root = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
        inside = eta * d + (eta * cos_i - math.sqrt(root)) * n
        return k, q, inside / np.linalg.norm(inside)

    def walk(self, k: int, q, d, depth: int = 400, cut: float = 1e-10):
        """The faces met in order and each reflection's amplitude, from the entry (face ``k``, amplitude 1)."""
        faces, amps = [k], [1.0]
        total = 1.0
        p = q
        for _ in range(depth):
            den = self.normals @ d
            t = np.where(den > 1e-15, (self.offsets - self.normals @ p) / np.where(den > 1e-15, den, 1.0), np.inf)
            t[faces[-1]] = np.inf
            nxt = int(np.argmin(t))
            if not math.isfinite(float(t[nxt])):
                break
            p = p + float(t[nxt]) * d
            r = self.amplitude(nxt, d)
            total *= r
            if total < cut:
                break
            faces.append(nxt)
            amps.append(r)
            n = self.normals[nxt]
            d = d - 2.0 * float(np.dot(d, n)) * n
        return faces, amps


def _reference(body: _Convex, scene, rays: int, rng) -> tuple[dict[int, float], dict[int, float], float]:
    """Per face: the largest later-return amplitude sum and power sum over every event of rays from the source's beam (within
    2 waist radii and 3 far-field half angles of its axis), and the largest weight of a closed walk through three or more
    faces."""
    beam = scene.sources[0]
    origin, axis = np.array(beam.position, float), _unit(beam.direction)
    e1 = _unit(np.cross(axis, [0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [0.0, 1.0, 0.0]))
    e2 = np.cross(axis, e1)
    spread = WAVELENGTH * 1e-6 / (math.pi * WAIST)
    amp: dict[int, float] = {}
    power: dict[int, float] = {}
    loop = 0.0
    for _ in range(rays):
        r, phi = 2.0 * WAIST * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
        a, psi = 3.0 * spread * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
        p = origin + r * (math.cos(phi) * e1 + math.sin(phi) * e2)
        d = _unit(axis + a * (math.cos(psi) * e1 + math.sin(psi) * e2))
        entered = body.enter(p, d)
        if entered is None:
            continue
        faces, amps = body.walk(*entered)
        cum = np.cumprod(amps)
        for i, f in enumerate(faces):
            later = [j for j in range(i + 1, len(faces)) if faces[j] == f]
            amp[f] = max(amp.get(f, 0.0), sum(cum[j] / cum[i] for j in later))
            power[f] = max(power.get(f, 0.0), sum((cum[j] / cum[i]) ** 2 for j in later))
            for j in later:
                if len(set(faces[i + 1:j]) - {f}) >= 2:
                    loop = max(loop, float(cum[j] / cum[i]))
    return amp, power, loop


def _surface_ids(body: _Convex, scene) -> dict[int, str]:
    """The engine's surface id of each reference face (the patch in its plane)."""
    ids = {}
    for s in scene.surfaces:
        if s.owner_id != "P":
            continue
        n = np.array(s.normal, float)
        for k, (m, o) in enumerate(zip(body.normals, body.offsets)):
            if abs(float(np.dot(n, m))) > 1.0 - 1e-9 and abs(float(np.dot(m, np.array(s.origin, float))) - o) < 1e-6:
                ids[k] = s.surface_id
    return ids


def _check(made, rays: int, rng) -> dict:
    project, (vertices, triangles, mirrors, index) = made
    scene, library = _scene(project)
    sc = R._Scene(scene, library, WAVELENGTH, [R._face(i, s) for i, s in enumerate(scene.surfaces)])
    analysed = [solid.owner_id for solid in F._solids(sc)]
    gains = R.return_gains(scene, library, [WAVELENGTH])[WAVELENGTH]
    folds = [fold for fold in gains.solid_folds if fold.owner_id == "P"]
    body = _Convex(vertices, triangles, mirrors, index)
    amp, power, loop = _reference(body, scene, rays, rng)
    ids = _surface_ids(body, scene)
    out = {"analysed": analysed == ["P"], "screen": gains.closed_return_screen, "fold": bool(folds), "loop": loop,
           "bounded": bool(folds) and folds[0].bounded, "violations": [], "checks": 0, "faces": len(ids), "ratio": 0.0}
    # (a closed walk through three faces is counted by a fold: certified (its sums in the counts, the screen may complete with
    # no return) or not (a potential return, never COMPLETED_NO_RETURN))
    if loop > 0.0 and (not folds or (not folds[0].bounded and gains.closed_return_screen == "COMPLETED_NO_RETURN")):
        out["violations"].append(("closed walk through three faces not counted", loop))
    if folds and folds[0].bounded:
        fold = folds[0]
        bound = {sid: (g - 1.0, g2 - 1.0) for sid, g, g2 in zip(fold.surfaces, fold.loop, fold.loop_power)}
        for k, sid in ids.items():
            ga, gp = bound[sid]
            out["checks"] += 2
            if ga > 0.0:
                out["ratio"] = max(out["ratio"], amp.get(k, 0.0) / ga)
            if amp.get(k, 0.0) > ga * (1.0 + 1e-9) + 1e-15:
                out["violations"].append((sid, "amplitude", amp.get(k, 0.0), ga))
            if power.get(k, 0.0) > gp * (1.0 + 1e-9) + 1e-15:
                out["violations"].append((sid, "power", power.get(k, 0.0), gp))
    return out


# ------------------------------------------------------------------------------------------------ the tests


def _l_prism():
    """An L-shaped prism (non-convex): the L in (x, z) -- [0, 10] x [0, 4] and [0, 4] x [0, 10] -- extruded over y in [-5, 5],
    its triangles turned outward (the caps fanned from the corner (0, 0), which sees the whole L)."""
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]  # counter-clockwise in (x, z)
    n = len(poly)
    vertices = tuple((x, -5.0, z) for x, z in poly) + tuple((x, 5.0, z) for x, z in poly)
    triangles = []
    for k in range(1, n - 1):
        triangles.append((0, k, k + 1))  # bottom (normal -y)
        triangles.append((n, n + k + 1, n + k))  # top (normal +y)
    for i in range(n):
        j = (i + 1) % n
        triangles += [(i, n + i, n + j), (i, n + j, j)]  # side of edge i (outward)
    return vertices, tuple(triangles)


def test_a_polygon_patch_is_measured_by_its_corners():
    """Every convex mesh solid passes the convexity test (exact: a polygon's corners on the body's side of each face plane);
    a non-convex one (an L-shaped prism) still does not."""
    for made in (corner_cube(), penta(), rhomboid(), dove(), low_index_cube((1.0, 0.8, 0.7))):
        project = made[0]
        scene, library = _scene(project)
        sc = R._Scene(scene, library, WAVELENGTH, [R._face(i, s) for i, s in enumerate(scene.surfaces)])
        assert [s.owner_id for s in F._solids(sc)] == ["P"]
    vertices, triangles = _l_prism()
    project = request(_laser((2.0, 0.0, -5.0), (0.0, 0.0, 1.0)), part("P", solid("L", vertices, triangles))).project
    scene, library = _scene(project)
    sc = R._Scene(scene, library, WAVELENGTH, [R._face(i, s) for i, s in enumerate(scene.surfaces)])
    assert F._solids(sc) == []


def test_the_cells_kept_on_a_polygon_are_every_cell_that_meets_it():
    """``_meets_outline`` against dense sampling of each cell: a cell with a sampled point inside the polygon is kept."""
    rng = np.random.default_rng(5)
    outline = np.array([(-3.0, -2.0), (4.0, -1.0), (2.0, 3.5), (-1.0, 1.0), (-4.0, 2.5)])
    xs, ys = rng.uniform(-6.0, 6.0, 400), rng.uniform(-5.0, 5.0, 400)
    h = 0.4
    kept = F._meets_outline(outline, xs, ys, math.sqrt(2.0) * h)
    grid = np.linspace(-h, h, 9)
    for x, y, k in zip(xs, ys, kept):
        px, py = (x + grid[:, None] + 0.0 * grid[None, :]).ravel(), (y + 0.0 * grid[:, None] + grid[None, :]).ravel()
        inside = np.zeros(px.shape, dtype=bool)
        for i in range(len(outline)):
            (ax, ay), (bx, by) = outline[i], outline[(i + 1) % len(outline)]
            crosses = (ay > py) != (by > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                inside ^= crosses & (px < ax + (py - ay) * (bx - ax) / (by - ay))
        if inside.any():
            assert k, (x, y)


def test_a_faint_corner_cube_cavity_has_no_exposure_bound():
    """The faint ring inside one convex mesh solid: a corner cube whose entry face reflects 1 - 1e-12 from both sides
    and whose AR-coated back faces are met beyond the critical angle. Its round trip loses 1e-12: no finite exposure bound
    holds, so the screen names the solid's faces (a potential return) and every face's upper bound is withheld (it read
    0.0214 W/cm2, the screen ``COMPLETED_NO_RETURN``)."""
    vertices, triangles, faces = convex_solid([(0, 0, 0), (20, 0, 0), (0, 20, 0), (0, 0, 20)], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    surface = MeshSurface(surface_id="cc", vertices_mm=vertices, triangles=triangles, approximation=EXACT, closed=True,
                          medium_positive="air", medium_negative="GLASS15", interaction=RefractInteraction(coating=_constant(0.0025)),
                          facet_groups=(MeshFacetGroup(triangles=tuple(faces[3]),
                                                       interaction=RefractInteraction(coating=_constant(1.0 - 1e-12))),))
    element = EngineeringComponentSnapshot(instance_id="P", role="prism", surfaces=(surface,), power_ratings=(_rating(100.0),))
    d = -_unit((1.0, 1.0, 1.0))
    laser = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=tuple(np.array([9.0, 7.0, 4.0]) - 30.0 * d),
                    direction=tuple(d))
    project = ProjectSnapshot(components=(laser, element), materials=(glass(1.5),), targets=_targets("P"))
    result = run(project)
    faces_p = [s for s in result.exposures.surfaces if s.surface_id.startswith("P/")]
    assert len(faces_p) == 4
    for s in faces_p:
        assert all(r.exposure_upper_bound is None for r in s.ratings), s.surface_id
    (record,) = result.outcome.execution.budgets.return_analysis
    assert record.closed_return_screen == "POTENTIAL_RETURN"
    assert ("P/cc.t0", "P/cc.t1", "P/cc.t2", "P/cc.t3") in record.closed_return_surfaces
    assert "solid_folds" in record.closed_return_kinds
    assert all(t.assessment.value == "INDETERMINATE" for t in result.targets)


def _configurations():
    rng = np.random.default_rng(363)
    made = []
    for _ in range(3):
        made.append(("corner_cube", corner_cube(tuple(rng.uniform(-0.08, 0.08, 2)), tuple(np.array([9.0, 7.0, 4.0]) + rng.uniform(-1.0, 1.0, 3)))))
        made.append(("penta", penta(float(rng.uniform(-4.0, 4.0)), float(rng.uniform(-3.0, 3.0)))))
        made.append(("rhomboid", rhomboid(float(rng.uniform(-4.0, 4.0)))))
        made.append(("dove", dove(float(rng.uniform(-4.0, 4.0)))))
        made.append(("low_index_cube", low_index_cube((1.0, float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.6, 1.0))))))
    return made


def test_the_mesh_solids_never_hide_a_closed_walk_against_independent_ray_trees():
    """Corner cube, penta, rhomboid, dove (index 1.5, faces met beyond the critical angle) and a low-index cube, each at three
    random tilts: every one is analysed; every closed walk through three or more faces the reference traces is counted (a
    fold, never ``COMPLETED_NO_RETURN``); every certified sum holds every traced walk. The glass solids' returns are not
    certified (their states keep one aperture part per face for every visit, so light advancing along lossless faces closes a
    lossless walk of states): a potential return. The low-index cube is bounded."""
    rng = np.random.default_rng(518)
    violations, checks = [], 0
    for label, made in _configurations():
        out = _check(made, 160, rng)
        assert out["analysed"], label
        checks += out["checks"] + 1
        violations += [(label, v) for v in out["violations"]]
        if label == "low_index_cube":
            assert out["bounded"] and out["checks"] == 12 and out["screen"] == "COMPLETED_NO_RETURN", (label, out)
            assert out["ratio"] > 0.0, (label, out)  # (the reference sees returns at a face: the check is not vacuous)
        elif out["loop"] > 0.0:
            assert out["screen"] == "POTENTIAL_RETURN", (label, out)
    assert violations == [], violations
    assert checks >= 15 + 3 * 12
