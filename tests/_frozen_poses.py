"""Frozen canonical pose constants used by the fixture factories and golden-test helpers.

The engine reconstructs pose quaternions from axis/normal directions via ``matrix_to_quat`` then
``quat_normalize`` (both libm-sensitive: the last bit can differ across Windows / Linux / Pyodide). To make the
committed ``tests/fixtures/requests/*.json`` byte-exact and platform-independent, the fixture factories pin the
reconstructed poses to these explicit constants (verified on Windows CPython 3.14.7).
"""

from __future__ import annotations

from os_optics.contracts import Pose3D

# normal (-1/√2, 1/√2, 0): 45° mirror / non-polarizing / polarizing splitter.
Q_45_NEG = (0.2705980500730985, 0.2705980500730985, -0.6532814824381882, -0.6532814824381883)
# normal (1/√2, -1/√2, 0): MZI "MB" mirror.
Q_45_POS = (0.6532814824381883, 0.6532814824381882, 0.2705980500730985, 0.2705980500730985)
# normal (0, -1, 0): observation plane.
Q_NEG_Y = (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)
# normal (0, 1, 0): observation plane.
Q_POS_Y = (0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
# normal (1/√2, 1/√2, 0): 45° absorbing detector plane (trace regression fixtures).
Q_45_DIAG = (0.2705980500730985, 0.2705980500730985, 0.6532814824381882, 0.6532814824381883)


def pin_pose(component_or_observation, q, *, position=None):
    """Return a copy whose pose quaternion is pinned to ``q`` (``position`` preserved unless overridden)."""
    return component_or_observation.model_copy(update={
        "pose": Pose3D(
            position_mm=position if position is not None else component_or_observation.pose.position_mm,
            quaternion_wxyz=q,
        )
    })
