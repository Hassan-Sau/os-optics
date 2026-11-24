"""The leg-end pre-filter changes no result (performance).

At the end of every straight leg the tracer checks each observation plane the chief ray does not cross for the part of
the beam that reaches it first (``Tracer._leg_end``). The check skips a plane that no live pupil ray reaches first and
whose Gaussian share is certainly below the line's negligible level (``decisions.beyond_reach``: a plane the beam's
``K``-radius tube misses, ``K`` from that level), and the per-ray ends are found once per leg. ``_prefilter_leg_ends =
False`` runs the full check for every plane, as before. Every ``_leg_end`` call is audited here: the planes it records
(``PartialCrossing``, bit for bit), the planes it marks and the beam's power scope are the same with and without the
pre-filter; and whole validation results (every field but the runtime) are identical, for the fixture requests, the
crossing scenes of the intersecting-plane tests and seeded random scenes (lenses, mirrors, windows, dumps, tilted, parallel,
absorbing and small-area planes). The scalar ``erfc`` / ``exp`` paths have the bits of the array ones.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, builders as B, validate
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.math.exact_erf import ERFC_SERIES_LIMIT, erfc, erfc_float
from os_optics.math.exact_trig import exp, exp_float
from os_optics.optics.rays import decisions
from os_optics.optics.rays.tracer import Tracer

REQUESTS = sorted((Path(__file__).parent / "fixtures" / "requests").glob("*.json"))
Q = math.sqrt(0.5)
X = (1.0, 0.0, 0.0)


def _audit(monkeypatch) -> dict[str, list]:
    """Audit every ``_leg_end``: run it without the pre-filter, put the packet back, run it with it, and compare what each
    run recorded and marked. Returns the mismatches and the number of planes the pre-filter skipped."""
    record: dict[str, list] = {"mismatches": [], "skipped": [0], "calls": [0]}
    leg_end, beyond = Tracer._leg_end, decisions.beyond_reach

    def counted(index, beam, leg, cand, cfg):
        far = beyond(index, beam, leg, cand, cfg)
        record["skipped"][0] += int(far.sum())
        return far

    def audited(self, pk, times, crossed, end, ends_at, end_si=None):
        record["calls"][0] += 1
        held, n = (pk.leg_partial, pk.power_scope), len(self.out.partial_crossings)
        self._prefilter_leg_ends = False
        try:
            leg_end(self, pk, times, crossed, end, ends_at, end_si)
        finally:
            del self._prefilter_leg_ends
        full = (list(self.out.partial_crossings[n:]), pk.leg_partial, pk.power_scope)
        del self.out.partial_crossings[n:]
        pk.leg_partial, pk.power_scope = held
        leg_end(self, pk, times, crossed, end, ends_at, end_si)
        fast = (list(self.out.partial_crossings[n:]), pk.leg_partial, pk.power_scope)
        if fast != full:
            record["mismatches"].append((pk.path_id, ends_at, full, fast))

    monkeypatch.setattr(decisions, "beyond_reach", counted)
    monkeypatch.setattr(Tracer, "_leg_end", audited)
    return record


def _result(request: ValidationRequest) -> dict:
    data = json.loads(validate(request).model_dump_json())
    data.get("manifest", {}).pop("runtime", None)
    return data


def _request(project: ProjectSnapshot) -> ValidationRequest:
    return ValidationRequest(project=project, configuration=EngineConfiguration(run_layout=False, run_control=False,
                                                                                numerical=NumericalConfig()))


def _unit(v) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)  # type: ignore[return-value]


def _laser(**kw):
    kw.setdefault("waist_radius_mm", 0.5)
    return B.laser("L", wavelength_nm=780.0, power_W=1e-3, position=(-10.0, 0.0, 0.0), direction=X, **kw)


def _plane(oid, position, normal, **kw):
    kw.setdefault("radius_mm", 50.0)
    return B.observation_plane(oid, position=position, normal=normal, **kw)


def _crossing_scenes() -> dict[str, ProjectSnapshot]:
    """The geometries of the intersecting-plane tests and the review probes: planes through one point, ties, partial crossings before a
    tilted detector, an absorber, a fold mirror and a lens, tails beyond the pupil rays, small areas, planes along a
    diverging beam."""
    diag = (Q, Q, 0.0)
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99)
    lens = B.thin_lens("F", position=(0.0, 0.0, 0.0), focal_length_mm=200.0, clear_radius_mm=10.0)
    detector = _plane("D", (0.0, 0.0, 0.0), X, absorbing=True)
    tmax = (Target(target_id="tmax", metric="power_W", observation_id="T", maximum=1e-9, unit="W"),)
    scenes = {
        "issue_b_a": ProjectSnapshot(components=(_laser(),), observations=(_plane("B", (0, 0, 0), diag), _plane("A", (0, 0, 0), X))),
        "detector_tie": ProjectSnapshot(components=(_laser(),), observations=(_plane("D", (0, 0, 0), diag, absorbing=True),
                                                                            _plane("T", (0, 0, 0), X), _plane("behind", (50.0, 0, 0), X))),
        "fold_monitor": ProjectSnapshot(components=(_laser(), fold), observations=(_plane("T", (0.3, 0.3, 0.0), _unit((1, 1, 0))),
                                                                                  _plane("D", (0.0, 60.0, 0.0), (0, 1.0, 0)))),
        "lens_monitor": ProjectSnapshot(components=(_laser(), lens), observations=(_plane("T", (0.3, 0.0, 0.0), diag),
                                                                                  _plane("D", (60.0, 0.0, 0.0), X))),
        "fold_absorbing": ProjectSnapshot(components=(_laser(), fold), observations=(_plane("T", (0.1, 0.0, 0.0), X, absorbing=True),
                                                                                    _plane("E", (0.0, 50.0, 0.0), (0, 1.0, 0)))),
    }
    for c in (0.001, 0.1, 0.3, 0.6, 0.9, 2.0):
        scenes[f"tilted_before_detector_{c}"] = ProjectSnapshot(components=(_laser(),), observations=(_plane("T", (c, 0, 0), diag), detector),
                                                                targets=tmax)
        scenes[f"absorbing_before_detector_{c}"] = ProjectSnapshot(components=(_laser(),), observations=(
            _plane("T", (c, 0, 0), diag, absorbing=True), detector), targets=tmax)
    for centre, radius in (((-0.15, 0.35, 0.0), 0.03), ((-0.2, 0.4, 0.1), 0.03), ((-1.0, 5.0, 0.0), 0.05)):
        scenes[f"small_area_{centre}"] = ProjectSnapshot(components=(_laser(),), observations=(
            _plane("T", centre, diag, radius_mm=radius), detector), targets=tmax)
    narrow = _laser(waist_radius_mm=0.01)
    for x_centre, area in ((40.0, {"radius_mm": 2.0}), (140.0, {"radius_mm": 2.0}), (40.0, {"radius_mm": None})):
        scenes[f"along_{x_centre}_{area}"] = ProjectSnapshot(components=(narrow,), observations=(
            _plane("T", (x_centre, 2.0, 0.0), (0, 1.0, 0), **area), _plane("D", (90.0, 0.0, 0.0), X, absorbing=True)))
    return scenes


def _random_scene(seed: int) -> ProjectSnapshot:
    rng = np.random.default_rng(seed)

    def r(lo, hi):
        return float(rng.uniform(lo, hi))

    sources = [B.laser(f"L{k}", wavelength_nm=float(rng.choice([532.0, 780.0, 1064.0])), power_W=1e-3,
                       waist_radius_mm=r(0.02, 1.5), waist_position_z_mm=r(-30.0, 30.0),
                       position=(-10.0, r(-3, 3), r(-3, 3)), direction=_unit((1.0, r(-0.05, 0.05), r(-0.05, 0.05))))
               for k in range(int(rng.integers(1, 3)))]
    parts = []
    for k in range(int(rng.integers(1, 5))):
        x = 15.0 + 25.0 * k + r(-3, 3)
        kind = int(rng.integers(0, 6))
        if kind == 0:
            parts.append(B.mirror(f"C{k}", position=(x, r(-1, 1), 0.0), normal=_unit((-1.0, r(-1.2, 1.2), r(-0.3, 0.3))),
                                  clear_radius_mm=r(1.0, 12.7), reflectance=0.99))
        elif kind == 1:
            parts.append(B.thin_lens(f"C{k}", position=(x, 0.0, 0.0), focal_length_mm=float(rng.choice([-50.0, 30.0, 100.0])),
                                     clear_radius_mm=r(2.0, 12.7)))
        elif kind == 2:
            parts.append(B.window(f"C{k}", thickness_mm=r(1.0, 6.0), clear_radius_mm=r(3.0, 12.7), material="N-BK7",
                                  position=(x, 0.0, 0.0), axis=_unit((1.0, r(-0.2, 0.2), 0.0))))
        elif kind == 3:
            parts.append(B.singlet_lens(f"C{k}", r1_mm=float(rng.choice([25.0, 51.5, 100.0])), r2_mm=float(rng.choice([-25.0, -100.0, 1e9])),
                                        thickness_mm=r(2.0, 5.0), clear_radius_mm=r(5.0, 12.0), material="N-BK7", position=(x, 0.0, 0.0)))
        elif kind == 4:
            parts.append(B.spherical_mirror(f"C{k}", position=(x, 0.0, 0.0), normal=_unit((-1.0, r(-0.4, 0.4), 0.0)),
                                            radius_of_curvature_mm=float(rng.choice([-100.0, 50.0, 200.0])), clear_radius_mm=r(3.0, 12.7)))
        else:
            parts.append(B.beam_dump(f"C{k}", position=(x, r(-0.5, 0.5), 0.0), axis=_unit((1.0, r(-1, 1), 0.0)), radius_mm=r(0.5, 12.7)))
    axes = [X, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (Q, Q, 0.0), (Q, 0.0, Q)]
    planes = []
    for k in range(int(rng.integers(2, 9))):
        normal = axes[int(rng.integers(0, len(axes)))] if rng.random() < 0.5 else _unit(tuple(rng.normal(size=3)))
        area_kind = int(rng.integers(0, 3))
        area = {"radius_mm": None} if area_kind == 0 else (
            {"radius_mm": r(0.01, 10.0)} if area_kind == 1 else {"half_width_mm": r(0.05, 5.0), "half_height_mm": r(0.05, 5.0)})
        up = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
        planes.append(B.observation_plane(f"P{k}", position=(r(-5.0, 110.0), r(-4, 4), r(-4, 4)), normal=normal, up=up,
                                          absorbing=bool(rng.random() < 0.3), **area))
    return ProjectSnapshot(components=(*sources, *parts), observations=tuple(planes))


SCENES = {**_crossing_scenes(), **{f"random_{seed}": _random_scene(seed) for seed in range(40)}}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_prefilter_skips_only_planes_the_full_check_leaves_out(name: str, monkeypatch):
    record = _audit(monkeypatch)
    fast = _result(_request(SCENES[name]))
    assert record["mismatches"] == []
    monkeypatch.setattr(Tracer, "_prefilter_leg_ends", False)
    assert _result(_request(SCENES[name])) == fast


@pytest.mark.parametrize("path", REQUESTS, ids=[p.stem for p in REQUESTS])
def test_fixture_requests_are_identical_with_and_without_the_prefilter(path: Path, monkeypatch):
    request = ValidationRequest.model_validate_json(path.read_text(encoding="utf-8"))
    record = _audit(monkeypatch)
    fast = _result(request)
    assert record["mismatches"] == []
    monkeypatch.setattr(Tracer, "_prefilter_leg_ends", False)
    assert _result(request) == fast


def test_the_prefilter_skips_planes_in_the_audited_scenes(monkeypatch):
    """(The audit is not vacuous: the pre-filter does skip planes, in a scene of many planes along a folded path.)"""
    record = _audit(monkeypatch)
    comps = [_laser(), *(B.mirror(f"M{i}", position=(50.0 + 50.0 * i, 20.0 * (i % 2), 0.0), normal=(-1.0, 1.0 if i % 2 == 0 else -1.0, 0.0))
                         for i in range(4))]
    planes = tuple(_plane(f"P{k}", (5.0 + 4.0 * k, 0.5 * (k % 3), 0.0), _unit((1.0, 0.05 * (k % 5), 0.0)), radius_mm=5.0) for k in range(40))
    validate(_request(ProjectSnapshot(components=tuple(comps), observations=planes)))
    assert record["mismatches"] == [] and record["calls"][0] > 0 and record["skipped"][0] > 0


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def test_the_scalar_erfc_and_exp_have_the_bits_of_the_array_ones():
    rng = np.random.default_rng(7)
    xs = np.concatenate([rng.uniform(0.0, 45.0, 4000), rng.uniform(0.0, 2.0, 2000), np.exp(rng.uniform(-40.0, 4.0, 2000)),
                         np.nextafter(ERFC_SERIES_LIMIT, [0.0, 10.0]), [0.0, -0.0, -1.0, 1.5, 40.0, 41.0, 1e300, 5e-324, np.inf]])
    assert [_bits(erfc_float(float(x))) for x in xs] == [_bits(float(erfc(np.array([x]))[0])) for x in xs]
    es = np.concatenate([rng.uniform(-800.0, 720.0, 4000), rng.uniform(-1.0, 1.0, 2000), -np.exp(rng.uniform(-40.0, 7.0, 2000)),
                         [0.0, -0.0, 709.7, 709.8, -745.2, -745.1, -1600.0, 0.35, -0.35, 0.5 * math.log(2.0)]])
    assert [_bits(exp_float(float(x))) for x in es] == [_bits(float(exp(np.array([x]))[0])) for x in es]
