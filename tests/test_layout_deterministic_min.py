"""Regression: the minimum-clearance pair must be deterministic when several pairs are tolerance-equivalent.

The pairwise loop orders candidates by ``box_dist`` (the bounding-box gap), a floating lower bound of the exact footprint distance
that can differ in the last ulp across platforms. When several pairs share a tolerance-equal minimum clearance, the winner must be
chosen lexicographically on the (sorted) instance-id pair, independent of that floating ordering and of the early-exit pruning.
"""

from __future__ import annotations

import pytest

from os_optics.contracts import EngineeringComponentSnapshot, EngineeringEnvelopeSnapshot, Pose3D, ProjectSnapshot

from helpers import run


def _sq(instance_id, x, *, size=10.0):
    return EngineeringComponentSnapshot(
        instance_id=instance_id,
        pose=Pose3D.from_yaw_deg((x, 0.0, 0.0), 0.0),
        envelope=EngineeringEnvelopeSnapshot(
            footprint_mm=((-size / 2, -size / 2), (size / 2, -size / 2), (size / 2, size / 2), (-size / 2, size / 2)),
            z_range_mm=(0.0, 20.0),
        ),
    )


def layout(*comps):
    return run(ProjectSnapshot(components=tuple(comps)), layout=True).layout


def test_min_clearance_pair_is_lexicographic_among_tolerance_equal_minima():
    """Three pairs with clearances 10, 10+1e-7, 10+2e-7 mm (all within layout_geometry_tol_mm = 1e-6), spaced so no cross pair is
    closer. The clearance order is the reverse of the id order: ("E","F") is closest but lexicographically largest, ("A","B") is
    furthest but lexicographically smallest. The loop therefore sees ("E","F") first; the old code keeps it, the new code breaks the
    tolerance tie lexicographically and selects ("A","B")."""
    lay = layout(
        _sq("A", 0.0),
        _sq("B", 20.0 + 2e-7),
        _sq("C", 100.0),
        _sq("D", 120.0 + 1e-7),
        _sq("E", 200.0),
        _sq("F", 220.0),
    )
    assert lay.min_clearance_pair == ("A", "B")
    # the reported clearance is the chosen pair's actual distance, not the absolute (tolerance-equivalent) minimum
    assert lay.min_clearance_mm == pytest.approx(10.0 + 2e-7, abs=1e-9)


def test_min_clearance_pair_is_emission_order_independent():
    forward = layout(_sq("A", 0.0), _sq("B", 20.0 + 2e-7), _sq("C", 100.0), _sq("D", 120.0 + 1e-7), _sq("E", 200.0), _sq("F", 220.0))
    reverse = layout(_sq("F", 220.0), _sq("E", 200.0), _sq("D", 120.0 + 1e-7), _sq("C", 100.0), _sq("B", 20.0 + 2e-7), _sq("A", 0.0))
    assert forward.min_clearance_pair == reverse.min_clearance_pair == ("A", "B")
    assert forward.min_clearance_mm == reverse.min_clearance_mm
