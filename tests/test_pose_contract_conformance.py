"""The Engine ``Pose3D`` and the shared ``PoseV1`` the Engine ships (``os_optics._vendor``) are one wire contract: the same
JSON schema (descriptions included), the same accepted and refused payloads and the same wire form. The shared-contracts
suite holds the same check against the source package; this one runs in every Engine environment and cannot skip."""

from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics.contracts.geometry import Pose3D


def _structural_schema(model: type) -> dict:
    schema = copy.deepcopy(model.model_json_schema())
    schema.pop("title", None)
    return schema


def _accepts(model: type, payload: dict) -> bool:
    try:
        model.model_validate(payload)
    except ValidationError:
        return False
    return True


def test_pose_json_schema_is_structurally_equal_to_the_shared_pose() -> None:
    assert _structural_schema(Pose3D) == _structural_schema(PoseV1)


def test_pose_and_quaternion_descriptions_are_the_shared_ones() -> None:
    engine, shared = Pose3D.model_json_schema(), PoseV1.model_json_schema()
    assert engine["description"] == shared["description"]
    assert engine["properties"]["quaternion_wxyz"]["description"] == shared["properties"]["quaternion_wxyz"]["description"]
    assert "q / |q|" in engine["description"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"position_mm": [1.0, -2.5, 3.0]},
        {"quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
        {"quaternion_wxyz": [2.0, 0.0, 0.0, 0.0]},
        {"quaternion_wxyz": [0.0, 0.0, 0.0, 0.0]},
        {"position_mm": [1.0, 2.0]},
        {"quaternion_wxyz": [1.0, 0.0, 0.0]},
        {"unknown": 1},
        {"position_mm": [math.nan, 0.0, 0.0]},
        {"position_mm": [math.inf, 0.0, 0.0]},
        {"quaternion_wxyz": [1.0, 0.0, 0.0, -math.inf]},
    ],
)
def test_pose_accept_reject_corpus_matches_the_shared_pose(payload: dict) -> None:
    assert _accepts(Pose3D, payload) == _accepts(PoseV1, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"position_mm": [0.1, 0.2, 0.3], "quaternion_wxyz": [2.0, 0.0, 0.0, 0.0]},
        {"quaternion_wxyz": [-1.0, 0.0, 0.0, 0.0]},
        {"quaternion_wxyz": [0.0, 0.0, 0.70711, 0.70711]},
    ],
)
def test_pose_wire_form_is_the_shared_pose_wire_form(payload: dict) -> None:
    shared = PoseV1.model_validate(payload)
    engine = Pose3D.model_validate(shared.model_dump(mode="json"))
    assert engine.model_dump(mode="json") == shared.model_dump(mode="json")
