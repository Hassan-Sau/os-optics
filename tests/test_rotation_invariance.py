"""A design turned rigidly as a whole gives the same validation.

Each bench runs as given and turned about oblique axes through the design origin: once exactly (every position and
orientation turned in binary64) and once written back with decimal positions (rounded to 1e-9 mm, as a layout tool
stores them), which leaves an exactly normal surface met at 1e-16 .. 1e-10 rad. The run status, every issue (code and
status), every target (status, issue codes) must be identical, and the physical results (target values, arrival powers,
Gaussian beam radii, Stokes parameters, phases) equal within a tight tolerance. Before the fix, a chief ray within
rounding of normal incidence took its plane of incidence (s / p basis, Gaussian tangential / sagittal axes) from the
rounding noise of ``d x n``: POLARIZATION_BASIS_DRIFT warnings and a spurious GAUSSIAN_TWISTED_ASTIGMATISM on the turned
acylinder (``math.vectors.NORMAL_INCIDENCE_SIN``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from os_optics.local import load_public_catalog, run_optical_project

from catalog_store_helpers import REAL_CATALOG, pinned_to_seed

BENCHES = json.loads((Path(__file__).resolve().parent / "fixtures" / "rotation" / "rotation_benches.json").read_text(encoding="utf-8"))

#: (axis, angle in deg, decimal digits of the written positions or None for the exact turn)
TURNS = {
    "axis_1_2_3_by_23_exact": ((1.0, 2.0, 3.0), 23.0, None),
    "axis_1_1_1_by_120_exact": ((1.0, 1.0, 1.0), 120.0, None),
    "axis_-3_1_2_by_57_rounded": ((-3.0, 1.0, 2.0), 57.0, 9),
    "axis_0.7_-0.5_0.2_by_171_rounded": ((0.7, -0.5, 0.2), 171.0, 9),
}


def _quaternion(axis, angle_deg):
    norm = math.sqrt(sum(c * c for c in axis))
    half = math.radians(angle_deg) / 2.0
    return (math.cos(half), *(math.sin(half) * c / norm for c in axis))


def _turned(document: dict, axis, angle_deg: float, digits: int | None) -> dict:
    from local_helpers import quat_multiply

    w, x, y, z = turn = _quaternion(axis, angle_deg)
    rot = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
           [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
           [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    out = json.loads(json.dumps(document))
    poses = [c["nominal_pose"] for c in out["components"]] + [o["pose"] for o in out["observations"] if not o.get("device_instance_id")]
    for pose in poses:
        p = pose["position_mm"]
        moved = [sum(rot[i][j] * p[j] for j in range(3)) for i in range(3)]
        pose["position_mm"] = [round(v, digits) + 0.0 if digits is not None else v for v in moved]
        q = quat_multiply(turn, pose["quaternion_wxyz"])
        size = math.sqrt(sum(c * c for c in q))
        pose["quaternion_wxyz"] = [c / size for c in q]
    return out


def _benches() -> dict[str, dict]:
    from test_packaging_and_pyodide import _coating_and_polarization_project, _coherent_elliptical_case, _rated_oblique_rotary_case

    return {"oblique_window_rotary_nd": _rated_oblique_rotary_case(), "coatings_phases_polarization": _coating_and_polarization_project(),
            "coherent_elliptical_interference": _coherent_elliptical_case(), **pinned_to_seed(BENCHES["projects"])}


def _outcome(document: dict, catalog) -> dict:
    run = run_optical_project(document, catalog=catalog)
    result = run.result
    arrivals = []
    for observation in result.observations:
        for arrival in observation.arrivals:
            beam = arrival.beam
            g = beam.gaussian
            pol = arrival.polarization_state
            arrivals.append({
                "where": (observation.observation_id, beam.path_id, arrival.crossing_index),
                "power_W": beam.power_W,
                "radii_mm": None if g is None or not g.valid else (g.beam_radius_x_mm, g.beam_radius_y_mm),
                "stokes": None if pol is None else (pol.s1, pol.s2, pol.s3, pol.ellipticity_deg),
                "phase_rad": beam.phase_rad,
            })
    return {
        "status": (run.status.value, result.status.value),
        "issues": sorted((i.code.value if hasattr(i.code, "value") else i.code, i.status.value) for i in result.issues),
        "targets": [(t.target_id, t.status.value, tuple(sorted(t.issue_codes))) for t in result.targets],
        "values": [t.value for t in result.targets],
        "observations": [(o.observation_id, o.status.value, len(o.arrivals)) for o in result.observations],
        "arrivals": arrivals,
    }


def _close(a, b, rel: float, abs_tol: float) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_close(x, y, rel, abs_tol) for x, y in zip(a, b))
    return abs(a - b) <= abs_tol + rel * max(abs(a), abs(b))


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


@pytest.fixture(scope="module")
def references(catalog):
    return {name: (document, _outcome(document, catalog)) for name, document in _benches().items()}


@pytest.mark.parametrize("turn", sorted(TURNS))
@pytest.mark.parametrize("bench", sorted(_benches()))
def test_a_design_turned_as_a_whole_gives_the_same_validation(bench, turn, references, catalog):
    document, want = references[bench]
    axis, angle, digits = TURNS[turn]
    got = _outcome(_turned(document, axis, angle, digits), catalog)
    for key in ("status", "issues", "targets", "observations"):
        assert got[key] == want[key], (bench, turn, key)
    # rounded positions move the design by up to 5e-10 mm: lengths and offsets follow at that level; phases by k0 x 1e-9 mm
    abs_tol, phase_tol = (1e-12, 2e-6) if digits is None else (1e-8, 1e-4)
    assert _close(got["values"], want["values"], 1e-9, abs_tol), (bench, turn, got["values"], want["values"])
    assert [a["where"] for a in got["arrivals"]] == [a["where"] for a in want["arrivals"]]
    for g, w in zip(got["arrivals"], want["arrivals"]):
        for key in ("power_W", "radii_mm", "stokes"):
            assert _close(g[key], w[key], 1e-9, abs_tol), (bench, turn, g["where"], key, g[key], w[key])
        if w["phase_rad"] is not None:
            difference = abs((g["phase_rad"] - w["phase_rad"] + math.pi) % (2.0 * math.pi) - math.pi)
            assert difference <= phase_tol, (bench, turn, g["where"], g["phase_rad"], w["phase_rad"])


def test_the_s_axis_is_the_rays_own_e1_within_the_normal_incidence_bound_and_d_x_n_beyond_it():
    import numpy as np

    from os_optics.math.vectors import NORMAL_INCIDENCE_SIN, dot, incidence_s_axis

    e1 = np.array([[0.0, 1.0, 0.0]])
    n = np.array([[-1.0, 0.0, 0.0]])
    for tilt in (0.0, 1e-16, 1e-10, 0.9 * NORMAL_INCIDENCE_SIN):  # a ray along +X tilted toward +Z
        d = np.array([[math.sqrt(1.0 - tilt * tilt), 0.0, tilt]])
        s, resolved = incidence_s_axis(d, n, e1, 1e-12)
        assert not resolved[0] and abs(float(dot(s[0], d[0]))) <= 1e-16 and float(dot(s[0], e1[0])) == pytest.approx(1.0, abs=1e-15)
    d = np.array([[math.sqrt(1.0 - 1e-10), 0.0, 1e-5]])
    s, resolved = incidence_s_axis(d, n, e1, 1e-12)
    assert resolved[0] and s[0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)  # d x n normalized
