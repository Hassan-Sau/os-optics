"""The linear optical network export and neutral-density attenuators: a thin ND plane states powers only (no optical
path through its substrate, no reflection phase), so every path that crosses or is reflected by one is refused with
``PHASE_UNDEFINED_ATTENUATOR`` -- the same rule as an ideal thin lens -- while the same layout without the attenuator
exports. Fixture data: the attenuators are declared in the test Project.
"""

from __future__ import annotations

import math

from os_optics.local import json_api

from test_neutral_density import _document, nd_model

FIXED = {"kind": "FIXED", "optical_density": "1"}


def _codes(reply: dict) -> set[str]:
    assert reply["ok"] is False
    return {p["code"] for p in reply["error"]["problems"]}


def test_a_path_through_an_attenuator_is_refused():
    reply = json_api.linear_optical_network(_document(nd_model(FIXED)))
    assert _codes(reply) == {"PHASE_UNDEFINED_ATTENUATOR"}
    (problem,) = reply["error"]["problems"]
    assert "crosses the attenuator at ND/attenuator" in problem["message"] and problem["location"] == "observation:det"


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def test_a_reflected_attenuator_branch_is_refused_too():
    c = math.sqrt(0.5)
    document = _document(nd_model(FIXED, reflectance="0.05"))
    (nd,) = [comp for comp in document["components"] if comp["instance_id"] == "ND"]
    turn = [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]  # 45 deg about design Z: the face normal along (1, 1, 0)/sqrt 2
    nd["nominal_pose"]["quaternion_wxyz"] = _qmul(turn, nd["nominal_pose"]["quaternion_wxyz"])
    # the reflection leaves along -Y; a plane facing it there (local +Z along -Y)
    document["observations"].append({"observation_id": "back", "pose": {"position_mm": [0.0, -50.0, 0.0], "quaternion_wxyz": [c, c, 0.0, 0.0]},
                                     "aperture": {"shape": "circle", "radius_mm": 20.0}})
    reply = json_api.linear_optical_network(document, {"observation_ids": ["back"]})
    assert _codes(reply) == {"PHASE_UNDEFINED_ATTENUATOR"}
    (problem,) = reply["error"]["problems"]
    assert "is reflected by the attenuator at ND/attenuator" in problem["message"]


def test_the_same_layout_without_the_attenuator_exports():
    document = _document(nd_model(FIXED))
    document["components"] = [c for c in document["components"] if c["instance_id"] != "ND"]
    document.pop("optical_models")
    reply = json_api.linear_optical_network(document)
    assert reply["ok"] is True
    assert any("neutral-density attenuator" in item for item in reply["network"]["conventions"]["not_modelled"])
