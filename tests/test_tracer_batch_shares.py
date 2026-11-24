"""The shares a stopping surface takes before an observation plane keep their bits.

``lower_share._meeting_share`` (``certain_meeting_share`` / ``possible_meeting_share``) and ``tracer._sag_region_share``
weigh the boxes of every profile cell and chief-ray stretch at once (``boxes``: cells as columns, stretches as rows,
``tracer._box_terms`` computed once for every linear function) and weigh the certain cells with the axis shares of every
grid edge computed once (``_below`` / ``_mass``). Their results reach ``power_taken_before_lower_W`` /
``power_taken_before_upper_W``, which results report and hash, and no frozen snapshot depends on a share's last bits. The
values below are ``float.hex`` of the shares the cell-by-cell implementation before that rewrite computed for the same
calls: the three skimmed bowls of ``test_start_and_certain_shares.py`` (sphere, rotational and cylindrical sag, through the
trace) and surfaces of every kind placed on the sphere bowl's leg (plane with a circle and with a rectangle, rotational
sag with a conic, cylindrical sag, asphere, axicon, sphere of either sign of radius), each certain from either side and
from both, and possible.

A share is a sum of the masses of the cells the boxes decide: the pins hold every decision of these calls and the mass
arithmetic bit for bit. A last-bit change of a box's range moves a share only where it flips a decision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

import os_optics.optics.rays.lower_share as lower_share
import os_optics.optics.rays.tracer as tracer
from helpers import run
from os_optics.scene.model import ApertureG

import test_start_and_certain_shares as bowls

DELTA, TARGET_MIN = 0.36, 0.995e-3


def _hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _recorded_bowl(kind: str) -> tuple[list[tuple[str, str | None]], list[tuple[Any, ...]]]:
    """The share calls the trace of a skimmed bowl makes: (function, float.hex of its result) and the arguments of each
    ``_meeting_share`` call."""

    meeting, region = lower_share._meeting_share, tracer._sag_region_share
    values: list[tuple[str, str | None]] = []
    meeting_args: list[tuple[Any, ...]] = []

    def recorded_meeting(*args: Any) -> float | None:
        meeting_args.append(args)
        value = meeting(*args)
        values.append(("_meeting_share", _hex(value)))
        return value

    def recorded_region(*args: Any) -> float | None:
        value = region(*args)
        values.append(("_sag_region_share", _hex(value)))
        return value

    lower_share._meeting_share, tracer._sag_region_share = recorded_meeting, recorded_region
    try:
        run(bowls._bowl(kind, DELTA, TARGET_MIN))
    finally:
        lower_share._meeting_share, tracer._sag_region_share = meeting, region
    return values, meeting_args


def _unit(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / math.sqrt(float(np.sum(v * v)))


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, helper))
    return u, np.cross(axis, u)


def _surfaces(surf: Any, leg: Any) -> dict[str, Any]:
    """Surfaces of every kind across the leg's beam (radius about 0.3 mm), part of it on their support."""

    point = leg.start + 30.0 * leg.d
    tilted = _unit(leg.d + np.array([0.0, 0.3, 0.6]))
    u, v = _frame(tilted)
    flat = replace(surf, geometry="plane", origin=point + np.array([0.0, 0.08, 0.05]), normal=tilted, u_axis=u, v_axis=v,
                   aperture=ApertureG("circle", 0.25, 0.0, 0.05, -0.03), mount_margin=0.0, outline=None, curvature=0.0,
                   conic=0.0, coefficients=(), cylindrical=False, linear=0.0, freeform=None, center=np.zeros(3), radius=0.0,
                   cap_axis=np.zeros(3), aperture_radius=0.0)
    rectangle = replace(flat, aperture=ApertureG("rectangle", 0.3, 0.12, 0.0, 0.04), mount_margin=0.05)
    against = _unit(-leg.d + np.array([0.0, -0.05, 0.2]))
    ua, va = _frame(against)
    sag = replace(flat, geometry="sag", origin=point + np.array([4.0, -0.1, 0.12]), normal=against, u_axis=ua, v_axis=va,
                  curvature=1.0 / 20.0, conic=-1.0, aperture=ApertureG("rectangle", 0.35, 0.2, 0.05, 0.0))
    cylinder = replace(sag, curvature=-1.0 / 15.0, conic=0.0, cylindrical=True, aperture=ApertureG("circle", 0.3, 0.0, 0.0, 0.1))
    asphere = replace(sag, curvature=1.0 / 30.0, conic=0.0, coefficients=(1e-5, -1e-8), aperture=ApertureG("circle", 0.28, 0.0, -0.04, 0.0))
    axicon = replace(sag, curvature=0.0, conic=0.0, linear=0.05, aperture=ApertureG("circle", 0.4, 0.0, 0.1, 0.1))
    radius = 25.0
    vertex = point + np.array([-6.0, 3.0, -0.1])  # (the rim of the cap crosses the beam)
    sphere = replace(surf, geometry="sphere", center=vertex - radius * against, radius=radius, cap_axis=against, origin=vertex,
                     normal=against, u_axis=ua, v_axis=va, aperture=ApertureG("circle", 2.9), aperture_radius=2.9, mount_margin=0.1)
    negative = replace(sphere, center=vertex + radius * against, radius=-radius, cap_axis=-against)
    return {"plane circle": flat, "plane rectangle": rectangle, "conic sag": sag, "cylindrical sag": cylinder,
            "asphere": asphere, "axicon": axicon, "sphere": sphere, "negative sphere": negative}


def share_values() -> dict[str, str | None]:
    """Every pinned share, by name (float.hex)."""

    values: dict[str, str | None] = {}
    leg_args: tuple[Any, ...] | None = None
    for kind in ("sphere", "rotational", "cylinder"):
        recorded, meeting_args = _recorded_bowl(kind)
        for index, (name, value) in enumerate(recorded):
            values[f"bowl {kind} #{index} {name}"] = value
        if kind == "sphere":
            leg_args = meeting_args[0]
    assert leg_args is not None
    surf, leg, k_sigma, _side, cfg, _possible = leg_args
    calls: dict[str, Callable[[], float | None]] = {}
    for label, s in _surfaces(surf, leg).items():
        for side in (1, -1, lower_share.EITHER_SIDE):
            calls[f"{label} certain side {side}"] = lambda s=s, side=side: lower_share._meeting_share(s, leg, k_sigma, side, cfg, False)
        calls[f"{label} possible"] = lambda s=s: lower_share._meeting_share(s, leg, k_sigma, lower_share.EITHER_SIDE, cfg, True)
        if s.geometry == "sag":
            calls[f"{label} sag region"] = lambda s=s: tracer._sag_region_share(s, leg, k_sigma, cfg)
    for label, call in calls.items():
        values[label] = _hex(call())
    return values


#: float.hex of each share, as the implementation before the batched rewrite computed it (recorded with the Engine of
#: 1f4e3d71e).
EXPECTED: dict[str, str | None] = {
    "bowl sphere #0 _meeting_share": "0x1.7b8653d76ead6p-8",
    "bowl rotational #0 _sag_region_share": "0x1.6cc1e6f1683d1p-7",
    "bowl rotational #1 _meeting_share": "0x1.9e4dc734f2c0bp-8",
    "bowl cylinder #0 _sag_region_share": "0x1.6cc1e6f11aebcp-7",
    "bowl cylinder #1 _meeting_share": "0x1.9e4dc2cc8c448p-8",
    "plane circle certain side 1": "0x0.0p+0",
    "plane circle certain side -1": "0x1.14a0ec5d467c9p-1",
    "plane circle certain side 0": "0x1.14a0ec5d467c9p-1",
    "plane circle possible": "0x1.81a3053bc861cp-1",
    "plane rectangle certain side 1": "0x0.0p+0",
    "plane rectangle certain side -1": "0x1.e1b1e7eadea82p-2",
    "plane rectangle certain side 0": "0x1.e1b1e7eadea82p-2",
    "plane rectangle possible": "0x1.5eb1e4a9bacb1p-1",
    "conic sag certain side 1": "0x1.5bdebf9d3b236p-1",
    "conic sag certain side -1": "0x0.0p+0",
    "conic sag certain side 0": "0x1.5bdebf9d3b236p-1",
    "conic sag possible": "0x1.784189bd70410p-1",
    "conic sag sag region": "0x1.7210641085184p-1",
    "cylindrical sag certain side 1": "0x1.0a8432ca36544p-1",
    "cylindrical sag certain side -1": "0x0.0p+0",
    "cylindrical sag certain side 0": "0x1.0a8432ca36544p-1",
    "cylindrical sag possible": "0x1.30fd65071990ep-1",
    "cylindrical sag sag region": "0x1.2a9e5802beb29p-1",
    "asphere certain side 1": "0x1.1f54c03090c77p-1",
    "asphere certain side -1": "0x0.0p+0",
    "asphere certain side 0": "0x1.1f54c03090c77p-1",
    "asphere possible": "0x1.44580d57f5913p-1",
    "asphere sag region": "0x1.3e8faf4697c40p-1",
    "axicon certain side 1": "0x1.a7eff4ad3a08cp-1",
    "axicon certain side -1": "0x0.0p+0",
    "axicon certain side 0": "0x1.a7eff4ad3a08cp-1",
    "axicon possible": "0x1.bf7330b678430p-1",
    "axicon sag region": "0x1.b9d704395437bp-1",
    "sphere certain side 1": "0x1.45f435c697d90p-2",
    "sphere certain side -1": "0x0.0p+0",
    "sphere certain side 0": "0x1.45f435c697d90p-2",
    "sphere possible": "0x1.0000000000000p+0",
    "negative sphere certain side 1": "0x1.8e6b7877c7ea1p-2",
    "negative sphere certain side -1": "0x0.0p+0",
    "negative sphere certain side 0": "0x1.8e6b7877c7ea1p-2",
    "negative sphere possible": "0x1.0000000000000p+0",
}


def test_the_shares_keep_the_bits_of_the_cell_by_cell_implementation() -> None:
    values = share_values()
    assert values == EXPECTED


def test_the_pinned_shares_reach_the_rewritten_branches() -> None:
    """The pins are not vacuous: most shares lie strictly between 0 and 1, from each kind of surface."""

    interior = {name for name, value in EXPECTED.items() if value is not None and 0.0 < float.fromhex(value) < 1.0}
    for label in ("plane circle", "plane rectangle", "conic sag", "cylindrical sag", "asphere", "axicon", "sphere",
                  "negative sphere"):
        assert any(name.startswith(label) and "certain" in name for name in interior), label
    assert any("sag region" in name for name in interior)
    assert any(name.startswith("bowl") for name in interior)
