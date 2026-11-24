"""Beam path ids stay bounded however deep the beam tree grows: a cavity of two native ideal plate
splitters used to fail the whole run (OPTICAL_VIEW_BINDING_FAILED: path_id longer than the scene's 128 characters).

The tracer keeps the branch part of an id at most ``BRANCH_SUFFIX_MAX_LENGTH`` characters (``optics/rays/path_ids.py``) and the
scene contract's ``BeamPathId`` admits the resulting ``PATH_ID_MAX_LENGTH``."""

from __future__ import annotations

import json
import re

import pytest

from os_optics._vendor.shared_contracts.optical_view.scene import BEAM_PATH_ID_MAX_LENGTH, BeamPathV1
from os_optics.local import json_api
from os_optics.local.catalog import load_project
from os_optics.local.lowering import lower_project
from os_optics.native_library import build_native_model
from os_optics.native_library.project import add_to_project
from os_optics.optics.rays.path_ids import (
    BRANCH_SUFFIX_MAX_LENGTH,
    BRANCH_SUFFIX_PATTERN,
    PATH_ID_MAX_LENGTH,
    child_path_id,
    root_path_id,
)
from os_optics.validation.pipeline import CHIEF_RAYS, run_physics

#: Quaternion turning the component +Z (the beam axis of sources and splitters) onto design +X.
ALONG_X = [0.5, 0.5, 0.5, 0.5]


def native_splitter_cavity(reflectance: str = "0.7") -> dict:
    """A laser through two native ideal plate splitters facing each other 10 mm apart (a low-finesse Fabry-Perot), and an
    observation plane behind them: the light bounces between the plates until its power falls below the power floor."""
    project = {"schema": "opensci.project", "project_schema_version": "2.0", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
        "nominal_pose": {"position_mm": [-200.0, 0.0, 0.0], "quaternion_wxyz": ALONG_X},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 780.0, "power_w": 0.001, "waist_radius_mm": 1.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "out", "pose": {"position_mm": [100.0, 0.0, 0.0], "quaternion_wxyz": ALONG_X},
                          "absorbing": False, "medium": "air"}]}
    transmittance = str(round(1.0 - float(reflectance), 12))
    build = build_native_model("splitter.plate", {"diameter_mm": "12.7", "reflectance": reflectance, "transmittance": transmittance,
                                                  "valid_band_nm": ["700", "900"]})
    project = add_to_project(project, build, instance_id="BS1", position_mm=(-100.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)
    return add_to_project(project, build, instance_id="BS2", position_mm=(-90.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


def test_child_ids_are_bounded_unique_and_independent_of_the_line_index():
    ids = {root_path_id("S", 0)}
    current = root_path_id("S", 0)
    for depth in range(400):
        child = child_path_id(current, root_path_id("S", 0), depth % 3)
        assert len(child) - len("S#0") <= BRANCH_SUFFIX_MAX_LENGTH
        assert re.fullmatch(r"S#0" + BRANCH_SUFFIX_PATTERN, child)
        assert child not in ids
        ids.add(child)
        # the same branch of line 7 has the same suffix (the network export relabels probe ids by the line index)
        assert child_path_id(current.replace("S#0", "S#7", 1), root_path_id("S", 7), depth % 3) == "S#7" + child[3:]
        current = child
    # siblings differ, and a compacted id never equals an uncompacted one (its suffix starts with '~')
    assert child_path_id(current, "S#0", 0) != child_path_id(current, "S#0", 1)
    assert any("~" in i for i in ids) and all(i.count("~") <= 1 for i in ids)
    # the longest id: a 128-character source id, an index and a full branch part fit the scene contract's limit
    assert PATH_ID_MAX_LENGTH == BEAM_PATH_ID_MAX_LENGTH
    assert 128 + len("#999999") + BRANCH_SUFFIX_MAX_LENGTH <= PATH_ID_MAX_LENGTH


def test_shallow_trees_keep_the_readable_ids():
    """Below the bound nothing changes: '<source>#<line>.<k>...' as before (stored results keep their ids)."""
    root = root_path_id("L0", 2)
    assert child_path_id(root, root, 0) == "L0#2.0"
    assert child_path_id("L0#2.0.13", root, 4) == "L0#2.0.13.4"


@pytest.mark.parametrize("reflectance", ["0.7", "0.9"])
def test_a_native_splitter_cavity_runs_and_its_network_is_refused_with_a_structured_code(reflectance):
    """R3 #1: the Fabry-Perot of two native plate splitters (R = 0.7 / 0.9) used to end OPTICAL_VIEW_BINDING_FAILED; it now
    runs, its scene holds every path, and the network export refuses with a network code (no binding failure)."""
    document = native_splitter_cavity(reflectance)
    reply = json_api.run_project(document)
    assert reply["ok"] is True, reply.get("error")
    scene = reply["optical_scene"]
    ids = [p["path_id"] for p in scene["beam_paths"]]
    assert max(len(i) for i in ids) <= PATH_ID_MAX_LENGTH and any("~" in i for i in ids)  # compacted deep branches
    for path in scene["beam_paths"]:
        BeamPathV1.model_validate(path)
        assert path.get("parent_path_id") is None or path["parent_path_id"] in ids
    network = json_api.linear_optical_network(document)
    assert network["ok"] is False and network["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    assert {p["code"] for p in network["error"]["problems"]} <= {"PARTIAL_MODE_OVERLAP", "CAVITY_NOT_SUPPORTED", "TRACE_INCOMPLETE"}


def test_the_cavity_series_of_the_traced_paths_is_the_textbook_one():
    """Independent of the ids: the k-th transmitted beam behind the cavity carries T^2 R^(2k) of the input (lossless plates,
    no interference in the per-path ledger)."""
    lowered = lower_project(load_project(native_splitter_cavity("0.7")), None)
    outcome = run_physics(lowered.request.project, lowered.request.configuration.numerical, plan=CHIEF_RAYS)
    powers = sorted((a.power_W for a in outcome.trace.arrivals if a.observation_id == "out"), reverse=True)
    r, t = 0.7, 0.3
    for k, power in enumerate(powers[:20]):
        assert power == pytest.approx(1e-3 * t * t * r ** (2 * k), rel=1e-9)
    assert json.dumps(sorted(p.path_id for p in outcome.trace.paths)) == json.dumps(sorted({p.path_id for p in outcome.trace.paths}))
