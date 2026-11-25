"""Internal folds and total internal reflection inside one plane-faced solid (``validation/solid_folds.py``).

The bound is checked against an independent reference: exact ray trees inside the catalog solids (plane intersections,
the trace's apertures, branching into reflection at the exact angle and transmission), whose returns to a face must
never exceed the certified ``G_s - 1``; and against the analytic fold of a penta prism. Scenes without a fold
solid keep every factor bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_catalog3_physics as T
from catalog3_products import catalog3_document
from local_helpers import project as document
from test_return_pairs import _coat, _constant, _scene

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot
from os_optics.local.lowering import lower_project
from os_optics.validation import resonance as R
from os_optics.validation.solid_folds import solid_folds

LAMBDA = 633.0


def _quat(axis, angle):
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def _mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _solid(kind: str, tilt_y: float = 0.0, tilt_z: float = 0.0, shift=(0.0, 0.0, 0.0)):
    sku = {"right_angle": T.RIGHT_ANGLE, "penta": T.PENTA, "cube": T.CUBE, "pbs": T.PBS_CUBE, "dove": T.DOVE}[kind]
    role = "splitter" if kind in ("cube", "pbs") else "prism"
    q = _mul(_quat((0, 0, 1), tilt_z), _mul(_quat((0, 1, 0), tilt_y), T.ALONG_X))
    point = tuple(a + b for a, b in zip(T._in_point(sku), shift))
    src = T._source_at(point, quaternion=q, wavelength_nm=LAMBDA, power_w=1.0)
    return lower_project(document([src, T._element("P", role, sku)], []), catalog3_document()).request.project


def _analysis(project):
    scene, library = _scene(project)
    faces = [R._face(i, s) for i, s in enumerate(scene.surfaces)]
    sc = R._Scene(scene, library, LAMBDA, faces)
    return scene, library, solid_folds(sc), R.return_gains(scene, library, [LAMBDA])[LAMBDA]


# ------------------------------------------------------------------------------------------------ the reference


class _Body:
    """The solid as plane faces with the trace's apertures, its body index, and each face's reflectance at an exact angle."""

    def __init__(self, scene, library, owner: str) -> None:
        surfaces = [s for s in scene.surfaces if s.owner_id == owner and s.interaction.kind in R.REFLECTING]
        centre = np.mean([np.array(s.origin, float) for s in surfaces], axis=0)
        body = next((s.medium_pos for s in surfaces if s.medium_pos == s.medium_neg), None)
        self.faces = []
        for s in surfaces:
            n = np.array(s.normal, float)
            inner_pos = float(np.dot(centre - np.array(s.origin, float), n)) > 0
            if body is None:
                body = s.medium_pos if inner_pos else s.medium_neg
            outer = s.medium_neg if inner_pos else s.medium_pos
            self.faces.append({"s": s, "o": np.array(s.origin, float), "n": n, "inner_pos": inner_pos,
                               "internal": s.medium_pos == s.medium_neg, "n_out": R._index(library, outer, LAMBDA)})
        self.n_in = R._index(library, body, LAMBDA)

    @staticmethod
    def inside(f, p) -> bool:
        s = f["s"]
        ap = s.aperture
        d = p - f["o"]
        x = float(np.dot(d, s.u_axis)) - ap.offset_u
        y = float(np.dot(d, s.v_axis)) - ap.offset_v
        return x * x + y * y <= ap.p1 * ap.p1 if ap.shape == "circle" else abs(x) <= ap.p1 and abs(y) <= ap.p2

    def reflectance(self, f, d) -> float:
        inter = f["s"].interaction
        c = abs(float(np.dot(d, f["n"])))
        aoi = math.acos(min(1.0, c))
        if inter.kind == "refract":
            if self.n_in > f["n_out"] and math.sin(aoi) * self.n_in >= f["n_out"]:
                return 1.0  # (total internal reflection)
            if inter.coating is None:
                return R._fresnel(self.n_in, f["n_out"], c)
        if inter.kind == "mirror" and inter.reflective_side != "both":
            if (inter.reflective_side == "positive") != (float(np.dot(d, f["n"])) < 0):
                return 0.0
        if inter.coating is None:
            return 1.0
        # (the record's stated value at the exact angle, an air-equivalent record at the air-equivalent angle -- grazing where
        # none exists --, as the return factors read it; a use outside the record withholds that face's bound, not this value)
        declared = inter.coating.declared_validity
        deg = math.degrees(aoi)
        if declared is not None and declared.aoi_reference == "AIR_EQUIVALENT" and abs(self.n_in - 1.0) > 1e-3:
            s_air = self.n_in * math.sin(aoi)
            deg = 90.0 if s_air >= 1.0 else math.degrees(math.asin(s_air))
        r, _ = R._coating(inter.coating, LAMBDA, deg, deg)
        return math.sqrt(r)

    def hit(self, p, d, skip):
        best = None
        for k, f in enumerate(self.faces):
            den = float(np.dot(d, f["n"]))
            if k == skip or abs(den) < 1e-15:
                continue
            t = float(np.dot(f["o"] - p, f["n"])) / den
            if t > 1e-9 and (best is None or t < best[0]):
                best = (t, k)
        return None if best is None else (best[1], p + best[0] * d)

    def returns(self, p, d, at: int, leaving: int, amp: float, depth: int = 0) -> float:
        """The walks from ``p`` along ``d`` that reflect at face ``at`` (every return, each counted): their amplitudes summed."""
        if amp < 1e-10 or depth > 30:
            return 0.0
        found = self.hit(p, d, leaving)
        if found is None:
            return 0.0
        k, q = found
        f = self.faces[k]
        if not self.inside(f, q):
            return 0.0  # (beyond the aperture: the light leaves the model)
        total = 0.0
        r = self.reflectance(f, d)
        if r > 0.0:
            if k == at:
                total += amp * r
            total += self.returns(q, d - 2.0 * np.dot(d, f["n"]) * f["n"], at, k, amp * r, depth + 1)
        if f["internal"]:
            total += self.returns(q, d, at, k, amp, depth + 1)
        return total

    def events(self, p, d, leaving: int, amp: float, out: list, depth: int = 0) -> None:
        if amp < 1e-6 or depth > 8:
            return
        found = self.hit(p, d, leaving)
        if found is None:
            return
        k, q = found
        f = self.faces[k]
        if not self.inside(f, q):
            return
        r = self.reflectance(f, d)
        if r > 0.0:
            d2 = d - 2.0 * np.dot(d, f["n"]) * f["n"]
            out.append((k, q, d2))
            self.events(q, d2, k, amp * r, out, depth + 1)
        if f["internal"]:
            self.events(q, d, k, amp, out, depth + 1)


def _sampled_returns(scene, library, grid: int = 4) -> dict[str, float]:
    """Per face, the largest sum of the closed walks back to it over every reflection event of rays entering on a grid of the entrance face
    within the source beam's cone (the reference)."""
    body = _Body(scene, library, "P")
    src = scene.sources[0]
    d0 = np.array(src.direction, float)
    spread = R.RETURN_ANGLE_SIGMAS * R._divergence(scene, library, LAMBDA)
    e1 = np.cross(d0, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d0, e1)
    directions = [d0] + [math.cos(spread) * d0 + math.sin(spread) * (math.cos(a) * e1 + math.sin(a) * e2) for a in (0.0, 2.1, 4.2)]
    entry = None
    for k, f in enumerate(body.faces):
        den = float(np.dot(d0, f["n"]))
        if f["internal"] or abs(den) < 1e-12 or (den < 0) == f["inner_pos"]:
            continue
        t = float(np.dot(f["o"] - np.array(src.position, float), f["n"])) / den
        if t > 0 and body.inside(f, np.array(src.position, float) + t * d0) and (entry is None or t < entry[0]):
            entry = (t, k)
    assert entry is not None
    k_in = entry[1]
    f_in = body.faces[k_in]
    s_in = f_in["s"]
    half = s_in.aperture.p1 if s_in.aperture.shape == "circle" else max(s_in.aperture.p1, s_in.aperture.p2)
    starts = []
    inward = f_in["n"] if f_in["inner_pos"] else -f_in["n"]
    for gx in np.linspace(-half, half, grid):
        for gy in np.linspace(-half, half, grid):
            p = f_in["o"] + gx * np.array(s_in.u_axis) + gy * np.array(s_in.v_axis)
            if not body.inside(f_in, p):
                continue
            for d in directions:
                refracted = R._refracted(tuple(d), tuple(inward), f_in["n_out"], body.n_in)
                if refracted is not None:
                    starts.append((k_in, p, np.array(refracted)))
                    body.events(p, np.array(refracted), k_in, 1.0, starts)
    worst: dict[str, float] = {}
    for k, p, d in starts:
        sid = body.faces[k]["s"].surface_id
        worst[sid] = max(worst.get(sid, 0.0), body.returns(p, d, k, k, 1.0))
    return worst


@pytest.mark.parametrize(("kind", "tilt"), [("cube", (0.0, 0.0)), ("pbs", (0.0, 0.0)), ("penta", (0.0, 0.0)),
                                            ("right_angle", (0.0, 0.0)), ("cube", (0.04, -0.03)), ("pbs", (-0.05, 0.02)),
                                            ("penta", (0.03, 0.04)), ("right_angle", (0.05, 0.055))])
def test_the_certified_closed_walks_hold_every_traced_internal_walk(kind, tilt):
    scene, library, folds, _ = _analysis(_solid(kind, *tilt))
    (fold,) = folds
    assert fold.fold
    sampled = _sampled_returns(scene, library)
    for sid, g in zip(fold.surfaces, fold.loop):
        assert math.isfinite(g), sid  # (these solids' walks are lossy: a certified sum exists)
        bound = g - 1.0
        assert sampled.get(sid, 0.0) <= bound * (1.0 + 1e-9) + 1e-15, (sid, sampled.get(sid), bound)


def test_the_penta_prisms_fold_is_its_analytic_round_trip():
    """Entrance face, both mirrors, exit face, both mirrors back: r_in r_out r_m^4 (0.0707^2 x 0.995^4 = 0.0049005)."""
    scene, library, (fold,), gains = _analysis(_solid("penta"))
    loop = dict(zip(fold.surfaces, fold.loop))
    analytic = 0.005 * 0.99 * 0.99
    for face in ("P/S1", "P/S4"):
        closed = loop[face] - 1.0  # (the fold's runs: L / (1 - L))
        assert analytic / (1.0 - analytic) <= closed <= analytic / (1.0 - analytic) * (1.0 + 1e-6)
    assert not fold.total_reflection
    assert gains.loops and all(math.isfinite(g) for row in gains.loops for g in row)


def test_a_cube_splitters_internal_fold_raises_its_faces_bounds_a_little():
    """The catalog cubes: the fold (S1, diagonal, S4, diagonal, S1: 0.0025 in amplitude) and the parallel faces' returns;
    the alternations with a paired face are left to the pairs."""
    _, _, (fold,), gains = _analysis(_solid("cube"))
    loop = dict(zip(fold.surfaces, fold.loop))
    assert 0.0075 <= loop["P/S1"] - 1.0 <= 0.0077
    plain = R.ReturnGains(gains.wavelength_nm, gains.by_surface, gains.pairs, gains.reflectors, gains.reflectance, gains.legs,
                          gains.once, gains.per_leg, gains.reflecting)
    for sid in ("P/S1", "P/S3", "P/S5"):
        assert plain.crossing_field(sid) < gains.crossing_field(sid) <= 1.01 * plain.crossing_field(sid)
        assert plain.crossings(sid) <= gains.crossings(sid) <= 1.01 * plain.crossings(sid)


def test_a_tilted_beam_neither_enters_beside_the_cube_nor_widens_its_returns():
    """A beam tilted 3 deg passes beside the cube's top face (its 8-radius ball overlaps that face's plane 100 mm away):
    it enters through the entrance face only, the steps keep each face's part of the aperture, and the sums equal the
    untilted cube's (no spurious grazing step from the diagonal's corner)."""
    from os_optics.validation import solid_folds as F  # noqa: PLC0415

    _, _, (straight,), _ = _analysis(_solid("cube"))
    project = _solid("cube", -0.06, -0.025, (-1.07, 0.08, 1.27))
    scene, library = _scene(project)
    sc = R._Scene(scene, library, LAMBDA, [R._face(i, s) for i, s in enumerate(scene.surfaces)])
    (solid,) = F._solids(sc)
    entering = {sc.surfaces[a].surface_id: F._entering(sc, solid, a) for a in solid.inner}
    assert [sid for sid, cones in entering.items() if cones] == ["P/S1"]
    (tilted,) = solid_folds(sc)
    assert not tilted.total_reflection
    for g, h in zip(tilted.loop, straight.loop):
        assert g == pytest.approx(h, rel=1e-9)


def test_a_step_outside_an_air_equivalent_record_withholds_that_face_only():
    """A beam tilted 11 deg into the cube meets its AR faces from the glass beyond the record's 0-10 deg air-equivalent: those
    steps read the record's stated value (the sums stay those of the untilted cube) and the faces met so lose their exposure
    upper bound (``ReturnGains.withheld``, the use among ``refused_coatings``), as a refused first run of left-out light does."""
    _, _, (straight,), _ = _analysis(_solid("cube"))
    _, _, (tilted,), gains = _analysis(_solid("cube", 0.2, 0.1))
    assert tilted.refused and all(use.refused is not None for use in tilted.refused)
    for g, h in zip(tilted.loop, straight.loop):
        assert g == pytest.approx(h, rel=1e-9)
    faces = {use.surface_id for use in tilted.refused}
    assert faces <= gains.withheld and {use.surface_id for use in gains.refused_coatings} >= faces
    assert all(gains.exposure_factor(sid) == math.inf for sid in faces)
    assert all(math.isfinite(gains.crossing_field(sid)) for sid in tilted.surfaces)


def test_total_internal_reflection_in_a_right_angle_prism_counts_one():
    _, _, (fold,), gains = _analysis(_solid("right_angle"))
    assert fold.total_reflection
    step = dict(zip(fold.surfaces, fold.step))
    assert max(step.values()) == 1.0
    assert all(r <= 1.0 for r, _ in gains.reflecting)


def _windows_and_lenses():
    laser = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    windows = [B.window(f"W{i}", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(30.0 * i, 0.0, 0.0)) for i in (1, 2)]
    lenses = [_coat(B.singlet_lens(f"L{i}", r1_mm=101.32, r2_mm=-101.32, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7",
                                   position=(100.0 + 60.0 * i, 0.0, 0.0), axis=(1.0, 0.0, 0.0)), _constant(0.0025)) for i in (1, 2)]
    return ProjectSnapshot(components=(laser, *windows, *lenses)), 1064.0


def test_a_scene_without_a_fold_solid_keeps_its_factors_bit_for_bit():
    """Windows, coated lenses and a dove prism (each face met once between returns of one pair): no solid is analysed, no
    loop is added."""
    for project, lam in (_windows_and_lenses(), (_solid("dove"), LAMBDA)):
        scene, library = _scene(project)
        gains = R.return_gains(scene, library, [lam])[lam]
        assert gains.loops == () and gains.solid_folds == ()


def test_an_unbounded_fold_is_a_closed_return_without_a_series():
    """A fold solid whose returns have no certificate names its faces as a closed return (no series): no surface the
    left-out light reaches keeps an exposure upper bound."""
    from os_optics.validation import solid_folds as F  # noqa: PLC0415

    project = _solid("cube")
    scene, library = _scene(project)
    original = F._loop
    try:
        F._loop = lambda *args, **kwargs: (math.inf, math.inf)  # (as a lossless return gives)
        gains = R.return_gains(scene, library, [LAMBDA])[LAMBDA]
    finally:
        F._loop = original
    assert ("P/S1", "P/S2", "P/S3", "P/S4", "P/S5") in gains.closed_cycles
    assert gains.closed_series is None
    assert gains.crossing_field("P/S1") == math.inf
    # why: the sum of a closed walk has no certificate (the analysis' limit, not a statement that the light circulates);
    # the certified solid of the same scene names no reason
    assert [fold.uncertified for fold in gains.solid_folds if not fold.bounded] == ["SUM_NOT_CERTIFIED"]
    certified = R.return_gains(scene, library, [LAMBDA])[LAMBDA]
    assert all(fold.uncertified is None and fold.bounded for fold in certified.solid_folds)
    # beside the withheld count: the open paths' count without that solid's loops (factor 1: no trail was searched) --
    # at least the count of the scene's open paths, finite, and in no proven field
    assert gains.nominal_returns_factor() == 1.0 and certified.nominal_returns_factor() is None
    assert R.LEAST_CROSSINGS <= gains.nominal_crossing_field("P/S1") < math.inf
    assert gains.nominal_crossing_field("P/S1") <= certified.crossing_field("P/S1")  # (the certified loops only add)


def test_a_fold_solid_under_the_three_way_closed_return_screen():
    """The solid analysis beside the closed-return screen's three states: a fold solid without a certified sum makes
    a finished screen a potential return, named by its faces in the run's record (never ``COMPLETED_NO_RETURN``); a screen
    that reaches its work limit stays ``INCOMPLETE`` -- no cycle, no surface named, no bound from the open paths --
    whether or not the solid's walks are certified."""
    from os_optics.validation import solid_folds as F  # noqa: PLC0415
    from os_optics.validation.pipeline import return_analysis_records  # noqa: PLC0415

    scene, library = _scene(_solid("cube"))
    certified = R.ReturnAnalysis(scene, library, [LAMBDA])
    bounded = certified[LAMBDA]
    assert bounded.solid_folds and all(fold.bounded for fold in bounded.solid_folds)
    assert bounded.closed_return_screen == "COMPLETED_NO_RETURN" and math.isfinite(bounded.crossing_field("P/S1"))
    need = bounded.work.steps
    original = F._loop
    try:
        F._loop = lambda *args, **kwargs: (math.inf, math.inf)  # (as a lossless return gives)
        unbounded = R.ReturnAnalysis(scene, library, [LAMBDA])
        held = unbounded[LAMBDA]
        # (one unit short of what this analysis needs: its fold analysis, which the work limit counts, does less here)
        stopped_held = R.ReturnAnalysis(scene, library, [LAMBDA], work_limit=held.work.steps - 1)
        stopped_held[LAMBDA]
    finally:
        F._loop = original
    assert held.closed_return_screen == "POTENTIAL_RETURN" and held.search.termination == "COMPLETED"
    assert held.crossings("P/S1") == held.crossing_field("P/S1") == math.inf
    (record,) = return_analysis_records(unbounded)
    assert record.closed_return_screen == "POTENTIAL_RETURN" and ("P/S1", "P/S2", "P/S3", "P/S4", "P/S5") in record.closed_return_surfaces
    # (named as the solid's faces, not as a path through them, in the record and in the cause a bound's text gives)
    assert record.closed_return_kinds == ("solid_folds",) * len(record.closed_return_surfaces)
    # (and the cause says why the fold analysis has no certified sum for this solid: here a closed walk's sum)
    assert held.return_causes("P/S1") == ("the internal folds of P (P/S1, P/S2, P/S3, P/S4, P/S5) may hold it (the fold analysis "
                                         "has no certified sum for this solid: SUM_NOT_CERTIFIED)",)
    stopped_bounded = R.ReturnAnalysis(scene, library, [LAMBDA], work_limit=need - 1)
    for analysis in (stopped_held, stopped_bounded):
        stopped = analysis[LAMBDA]
        assert stopped.work.stage == "closed_returns" and stopped.closed_return_screen == "INCOMPLETE"
        assert stopped.closed_cycles == () and stopped.search.termination == "BUDGET_EXHAUSTED"
        assert stopped.crossings("P/S1") == stopped.crossing_field("P/S1") == math.inf
        (record,) = return_analysis_records(analysis)
        assert record.closed_return_screen == "INCOMPLETE" and record.closed_return_surfaces == ()


def test_the_fold_analysis_counts_against_the_run_work_limit():
    """The fold-solid analysis spends the return-light analysis' work units (``validation/return_work.py``, stage
    ``solid_folds``): its count is in the record, and a limit it reaches leaves the closed returns not known --
    ``INCOMPLETE``, ended in ``solid_folds``, the trail search not started, every bound that reads the open paths withheld --
    while the pairs it follows stay as they are."""
    from os_optics.validation import solid_folds as F  # noqa: PLC0415
    from os_optics.validation.pipeline import return_analysis_records  # noqa: PLC0415

    scene, library = _scene(_solid("cube"))
    full = R.ReturnAnalysis(scene, library, [LAMBDA])
    done = full[LAMBDA]
    (record,) = return_analysis_records(full)
    assert done.work.fold_steps > 0 and record.solid_fold_work == done.work.fold_steps <= record.work_done
    entered = []
    original = F.solid_folds

    def spy(sc, *args, **kwargs):
        entered.append(sc.work.steps)
        return original(sc, *args, **kwargs)

    F.solid_folds = spy
    try:
        R.ReturnAnalysis(scene, library, [LAMBDA])[LAMBDA]
    finally:
        F.solid_folds = original
    limit = entered[0] + done.work.fold_steps // 2
    cut = R.ReturnAnalysis(scene, library, [LAMBDA], work_limit=limit)
    stopped = cut[LAMBDA]
    assert stopped.work.stage == "solid_folds" and stopped.closed_return_screen == "INCOMPLETE"
    assert stopped.search.termination == "NOT_STARTED" and stopped.closed_cycles == () and stopped.solid_folds == ()
    assert stopped.crossings("P/S1") == stopped.crossing_field("P/S1") == math.inf
    assert stopped.pairs == done.pairs and stopped.factor("P/S1") == done.factor("P/S1")
    (record,) = return_analysis_records(cut)
    assert (record.ended_in, record.termination, record.closed_return_screen) == ("solid_folds", "NOT_STARTED", "INCOMPLETE")
    assert entered[0] < record.work_done <= limit and record.solid_fold_work == record.work_done - entered[0]
