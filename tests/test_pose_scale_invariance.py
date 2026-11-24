"""A stated quaternion is taken by its direction wherever a rotation is computed from it.

A Project accepts a component, observation or placement quaternion within 1e-6 of unit length and stores it as given.
Every result computed from such a pose -- composition placements and transforms, attached observation planes, the Optical
View scene, the trace -- must be the result of its unit quaternion: scaling the stated quaternions by 1 +/- 8e-7 moves no
coordinate by more than rel 1e-12 + 1e-12 mm, changes no record, state or code of the result (its numbers agree within
the cross-runtime tolerance), and a Project the contract accepts never fails to run. ``q / |q|`` of a scaled quaternion may differ from the unit one in the last bit; an angle of
incidence is therefore taken from ``|n x d|`` near normal incidence and reads 0 below the 1e-7 deg a refracted direction
resolves (``math.exact_trig.incidence_deg``), where a cosine would read such a direction as 0 or 8.5e-7 deg by rounding. The plane of incidence of a beam that met a surface just off normal incidence turns with those bits: what follows it (its transport basis, Jones and Stokes components, Gaussian axes) is resolved to ``TRANSPORT_BASIS_RESOLUTION``. The result hash is equal wherever the hashed values (9
significant digits) are; a value that sits at a rounding boundary of those digits (a power of 2.523975625e-08 W, a target's
error of 5e-5 mm taken from a 2 mm value) rounds by its last bit, so there the hash follows that bit while the values agree
within rel 1e-12 + 1e-12. The references are the same operations on the unit
quaternions (the Project hash, the Engine input hash and the scene hash legitimately differ: they carry the stated
quaternion)."""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from catalog_store_helpers import REAL_CATALOG, REPO_ROOT
from composition_helpers import ok, screen
from local_helpers import UNIT_INDEX_ENVIRONMENT, catalog_document, with_lossless_bulk
from test_composition import REBIND, fold_mod, system_with_laser, two_fold_system

from os_optics import composition as comp
from os_optics.composition import rotation as rot
from os_optics.local import json_api as J
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics import ValidationRequest, validate
from os_optics.config import DEFAULT_NUMERICAL_CONFIG
from os_optics.errors.status import TerminationReason
from os_optics.local import EXAMPLE_KINDS, example_project_of_kind, run_optical_project
from os_optics.manifest.canonical import quantize
from os_optics.math.exact_trig import acos_deg, incidence_deg
from os_optics.math.phase import reported_phase
from os_optics.math.vectors import TRANSPORT_BASIS_RESOLUTION
from os_optics.optics.interference.coherence import delay_key

#: In-tolerance scales of a stated quaternion (beyond 1e-6 the Project refuses it).
SCALES = (1.0 - 8e-7, 1.0 + 8e-7)
QUARTER_TURN_Z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
_G = (0.95, 0.12, -0.21, 0.18)
GENERAL = tuple(c / math.sqrt(sum(v * v for v in _G)) for c in _G)
#: A half turn about the y = z diagonal.
HALF_YZ = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


def _scaled(q, s):
    return [c * s for c in q]


def _close(a, b, rel=1e-12, floor=1e-12):
    a, b = list(a), list(b)
    return len(a) == len(b) and all(abs(x - y) <= rel * max(abs(x), abs(y)) + floor for x, y in zip(a, b, strict=True))


def _poses(project: dict) -> dict:
    out = {("COMPONENT", c["instance_id"]): c["nominal_pose"] for c in project["components"]}
    out.update({("OBSERVATION", o["observation_id"]): o["pose"] for o in project["observations"]})
    return out


def _unit(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def _assert_same_poses(scaled: dict, reference: dict, keys=None) -> None:
    a, b = _poses(scaled), _poses(reference)
    for key in keys or b:
        assert _close(a[key]["position_mm"], b[key]["position_mm"]), (key, a[key], b[key])
        assert _close(_unit(a[key]["quaternion_wxyz"]), _unit(b[key]["quaternion_wxyz"])), (key, a[key], b[key])


# ------------------------------------------------------------------------------------------------ rotation primitive


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("q", [QUARTER_TURN_Z, GENERAL, (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))])
def test_rotate_takes_a_stated_quaternion_by_its_direction(q, s):
    v = (100.0, 20.0, -5.0)
    assert _close(rot.rotate(_scaled(q, s), v), rot.rotate(q, v))
    # a scaled quarter turn still moves points by the exact signed permutation
    assert (rot.signed_permutation(_scaled(q, s)) is None) == (rot.signed_permutation(q) is None)
    assert rot.unit((0.5, 0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5, 0.5) and rot.unit((-2.0, 0.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------------ composition


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("placement_q", [QUARTER_TURN_Z, GENERAL])
def test_an_import_placement_is_taken_by_its_direction(placement_q, s):
    def imported(q):
        placement = {"position_mm": [10.0, -20.0, 3.0], "quaternion_wxyz": list(q)}
        return ok(J.workspace_import(system_with_laser(), fold_mod(), {"instance_key": "k", "placement": placement, "fixtures": REBIND}))

    reference, scaled = imported(placement_q), imported(_scaled(placement_q, s))
    _assert_same_poses(scaled["workspace"]["project"], reference["workspace"]["project"])
    for a, b in zip(scaled["result"]["interfaces"], reference["result"]["interfaces"], strict=True):
        assert _close(a["point_mm"], b["point_mm"]) and _close(a["direction"], b["direction"])


def _with_scaled_device(s: float, plane_s: float) -> dict:
    """``two_fold_system`` with a plane attached to ``fa_M1`` at an offset, ``fa_M1`` and the plane quaternions scaled."""

    ws = two_fold_system()
    ws["project"]["observations"].append(screen("CAM", (1.5, -2.0, 3.0), GENERAL, device="fa_M1"))
    for component in ws["project"]["components"]:
        if component["instance_id"] == "fa_M1":
            component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], s)
    for observation in ws["project"]["observations"]:
        if observation["observation_id"] == "CAM":
            observation["pose"]["quaternion_wxyz"] = _scaled(observation["pose"]["quaternion_wxyz"], plane_s)
    return comp.load_workspace(ws).payload()


REQUESTS = {
    "turn_in_device_frame": {"selector": {"ids": ["fa_M1"]}, "rotation": {"axis": [0.2, 0.3, 1.0], "angle_deg": 37.0},
                             "translation_mm": [0.0, 0.0, 2.0], "frame": {"component": "fa_M1"}},
    "attached_plane_alone": {"selector": {"ids": ["CAM"]}, "rotation": {"axis": [1.0, -1.0, 0.5], "angle_deg": 11.0},
                             "translation_mm": [1.0, 0.0, 0.0]},
    "about_the_selection_centroid": {"selector": {"ids": ["fa_M1", "CAM", "fb_M1"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 23.0},
                                     "pivot": "selection_centroid"},
}


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_a_transform_of_a_device_and_its_attached_plane_is_taken_by_the_stated_directions(name, s):
    request = REQUESTS[name]
    reference = ok(J.workspace_transform(_with_scaled_device(1.0, 1.0), request))
    scaled = ok(J.workspace_transform(_with_scaled_device(s, s), request))
    _assert_same_poses(scaled["workspace"]["project"], reference["workspace"]["project"])
    if "pivot_mm" in reference["result"]:
        assert _close(scaled["result"]["pivot_mm"], reference["result"]["pivot_mm"])
    # the attached plane, wherever its device is, sits where the unit rotations put it
    for result in (scaled, reference):
        project = ProjectDocumentV2.model_validate(result["workspace"]["project"])
        cam = next(o for o in project.observations if o.observation_id == "CAM")
        assert abs(math.sqrt(sum(c * c for c in project.observation_design_pose(cam).quaternion_wxyz)) - 1.0) <= 1e-15


# ------------------------------------------------------------------------------------------------ run and Optical View


def _dichroic(parent_s: float, plane_s: float) -> dict:
    """The two-colour dichroic example with its reflected detector plane moved off the device origin and tilted, and the
    detector's and the plane's stated quaternions scaled."""

    document = example_project_of_kind("two_color_dichroic").model_dump(mode="json", by_alias=True, exclude_none=True)
    for component in document["components"]:
        if component["instance_id"] == "DET_R":
            component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], parent_s)
    (plane,) = [o for o in document["observations"] if o.get("device_instance_id") == "DET_R"]
    plane["pose"] = {"position_mm": [1.5, -2.0, 3.0], "quaternion_wxyz": _scaled(GENERAL, plane_s)}
    return document


def _run_facts(document: dict) -> dict:
    run = run_optical_project(ProjectDocumentV2.model_validate(document))
    facts = {"result_hash": run.local_validation.result_hash.digest, "planes": {}, "arrivals": {}}
    for observation in run.result.observations:
        plane = observation.plane
        facts["arrivals"][observation.observation_id] = [list(a.uv_mm) for a in observation.arrivals]
        if plane is not None:
            facts["planes"][observation.observation_id] = (list(plane.origin_mm), list(plane.normal), list(plane.u_axis))
    facts["scene"] = {p.observation_id: (list(p.pose.position_mm), list(p.pose.quaternion_wxyz)) for p in run.optical_view.observation_planes}
    return facts


@pytest.mark.parametrize("s", SCALES)
def test_a_project_with_in_tolerance_device_and_plane_quaternions_runs_as_its_unit_rotations(s):
    reference = _run_facts(_dichroic(1.0, 1.0))
    scaled = _run_facts(_dichroic(s, s))  # each within 1e-6, their product not: this run used to fail scene binding
    assert reference["arrivals"]["reflected_detector"], "the moved plane still receives the reflected beam"
    assert scaled["result_hash"] == reference["result_hash"]
    assert scaled["arrivals"].keys() == reference["arrivals"].keys()
    for observation_id, uvs in reference["arrivals"].items():
        assert all(_close(a, b) for a, b in zip(scaled["arrivals"][observation_id], uvs, strict=True))
    for observation_id, (origin, normal, u_axis) in reference["planes"].items():
        got = scaled["planes"][observation_id]
        assert _close(got[0], origin) and _close(got[1], normal) and _close(got[2], u_axis)
    for observation_id, (position, q) in reference["scene"].items():
        got_position, got_q = scaled["scene"][observation_id]
        assert _close(got_position, position) and _close(got_q, q)
        assert abs(math.sqrt(sum(c * c for c in got_q)) - 1.0) <= 1e-15
    # the attached plane sits where the unit rotations of its device and its own pose put it
    project = ProjectDocumentV2.model_validate(_dichroic(1.0, 1.0))
    (plane,) = [o for o in project.observations if o.device_instance_id == "DET_R"]
    assert _close(reference["scene"]["reflected_detector"][0], project.observation_design_pose(plane).position_mm, rel=0, floor=1e-6)


def _every_pose_scaled(kind: str, s: float) -> dict:
    document = example_project_of_kind(kind).model_dump(mode="json", by_alias=True, exclude_none=True)
    for component in document["components"]:
        component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], s)
    for observation in document["observations"]:
        if "pose" in observation:
            observation["pose"]["quaternion_wxyz"] = _scaled(observation["pose"]["quaternion_wxyz"], s)
    return document


def _leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


@cache
def _example_result(kind: str, s: float) -> tuple[dict, str]:
    run = run_optical_project(ProjectDocumentV2.model_validate(_every_pose_scaled(kind, s)))
    return dict(_leaves(run.result.model_dump(mode="json"))), run.local_validation.result_hash.digest


#: The manifest hashes of the stated Project: they carry the stated quaternions.
_STATED_HASHES = re.compile(r"\.manifest\.(project_hash|input_hash|design_projection_hash|result_hash)")


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_every_example_with_every_pose_scaled_reports_its_unit_result(kind, s):
    """Every component and observation quaternion of a bundled example scaled by ``s``: every record, state and code of the
    result is the unit run's (an undetermined pass's ``group_delay_unknown_at`` record names the same surfaces as normal or
    not), every number agrees within the Engine's cross-runtime tolerance (rel 1e-9 + 1e-12: a curvature radius of a nearly
    collimated beam carries the last bits of the poses magnified), and the result hash is equal unless a hashed value sits
    at a rounding boundary of its 9 digits, where the two values differ by last-bit noise only (rel 1e-12 + 1e-12)."""

    _assert_unit_result(*_example_result(kind, s), *_example_result(kind, 1.0))


#: A coordinate leaf of a result (a point or offset in mm: ``position_mm``, ``uv_mm``, a segment's ``start_mm`` ...).
_COORDINATE = re.compile(r".*_mm\[\d\]")
#: The trace's own hash: compared with the result hash (``_assert_unit_result``).
_TRACE_HASH = re.compile(r"\.manifest\.trace_hash")
#: A leaf that follows a beam's plane of incidence: a component of its transport basis, of its Jones vector in that basis,
#: of its Gaussian axes, its Stokes s1 / s2 -- resolved to ``TRANSPORT_BASIS_RESOLUTION`` (a beam that met a surface near
#: normal incidence).
_TRANSPORT_BASIS = re.compile(r".*\.(basis\.e[12]|polarization\.e[xy]|gaussian\.e[12])\[\d\]|.*\.polarization_state\.s[12]")


def _assert_unit_result(scaled: dict, scaled_hash: str | None, reference: dict, reference_hash: str | None, *,
                        coordinate_floor_mm: float = 0.0, basis_floor: float = 0.0) -> None:
    """``coordinate_floor_mm``: the absolute floor of a coordinate where it is above 1e-12 mm (a system whose extent puts
    the last bits of its coordinates above 1e-12 mm, ``_extent_floor_mm``); ``basis_floor``: the absolute floor of a leaf
    that follows a beam's plane of incidence, given only for a design whose beams meet a surface within
    ``NORMAL_INCIDENCE_SIN`` of normal incidence (``TRANSPORT_BASIS_RESOLUTION``); every other number keeps the
    cross-runtime tolerance. Hashes ``None``: the result and trace hashes are not compared, and a value on one of their
    9-digit rounding boundaries is held to the cross-runtime tolerance like any other (not to the 1e-12 that explains a
    moved hash)."""

    assert scaled.keys() == reference.keys()
    numerical = DEFAULT_NUMERICAL_CONFIG
    rounded_apart = []
    for path, want in reference.items():
        if _STATED_HASHES.fullmatch(path) or (reference_hash is None and _TRACE_HASH.fullmatch(path)):
            continue
        got = scaled[path]
        floor = numerical.cross_runtime_abs_tol
        if _COORDINATE.fullmatch(path):
            floor = max(floor, coordinate_floor_mm)
        if basis_floor and _TRANSPORT_BASIS.fullmatch(path):
            floor = max(floor, basis_floor)
        if isinstance(want, float) and isinstance(got, float):
            assert _close([got], [want], numerical.cross_runtime_rel_tol, floor), (path, got, want)
        else:
            assert got == want, (path, got, want)
        if quantize(got, numerical.hash_float_sig_digits, numerical.hash_float_abs_floor) != quantize(
                want, numerical.hash_float_sig_digits, numerical.hash_float_abs_floor):
            rounded_apart.append((path, got, want, max(1e-12, floor)))
    if reference_hash is None:
        return
    if rounded_apart:
        assert all(_close([got], [want], 1e-12, floor) for _path, got, want, floor in rounded_apart), rounded_apart
    else:
        assert scaled_hash == reference_hash


def _extent_floor_mm(reference: dict) -> float:
    """128 units in the last place of the largest coordinate of a result: a coordinate that crossed several imported
    placements and surfaces carries the rounding of each, of the order of the system's extent times the unit roundoff per
    operation (a coordinate that is 0 in exact arithmetic reads a few 1e-12 mm on a 480 mm system)."""

    return 128 * 2.0 ** -52 * max(abs(v) for p, v in reference.items() if isinstance(v, float) and _COORDINATE.fullmatch(p))


def test_a_pass_record_reads_a_signed_zero_hit_point_as_zero():
    """A hit point of +-1e-17 mm on either axis is the same point to the record's 1e-6 mm: one record, no "-0.000000"."""

    def key(at):
        return delay_key("OBJ/lens", "thin_lens", True, 0.0, at=at)

    assert key((-1e-17, 0.0)) == key((1e-17, 0.0)) == key((0.0, -1e-17)) == key((0.0, 0.0))
    assert key((-1e-17, -1e-17)) == "OBJ/lens:thin_lens:+@0.000000(0.000000,0.000000)"
    assert key((-4e-7, 6e-7)) == "OBJ/lens:thin_lens:+@0.000000(0.000000,0.000001)"


#: Stored designs run with the real catalog: (file, whether it is a workspace holding the Project, targets added to it).
STORED_DESIGNS = {
    "mzi_cascade_x8": (Path(__file__).parent / "fixtures" / "composition" / "mzi_cascade_x8.project.json", False, (
        # (a balanced pair on a cell's dump port: interface phase pi, optical path difference 0 up to last-bit noise)
        {"target_id": "T_SEAM_PHASE", "metric": "phase_difference_rad", "observation_id": "s2_DUMP",
         "parameters": [{"key": "path_id_a", "value": "LASER#0.0.1"}, {"key": "path_id_b", "value": "LASER#0.0.2.0.0"}],
         "target_value": 3.141593, "tolerance_abs": 0.01},
        # (the same pair named in the other order)
        {"target_id": "T_SEAM_REVERSED", "metric": "phase_difference_rad", "observation_id": "s2_DUMP",
         "parameters": [{"key": "path_id_a", "value": "LASER#0.0.2.0.0"}, {"key": "path_id_b", "value": "LASER#0.0.1"}],
         "target_value": 3.141593, "tolerance_abs": 0.01})),
    # (a refracting bench: lenses on the axis, an untraced hit's angle of incidence 0 up to the noise of refracted
    # directions; a counter-propagating pair of orthogonal polarizations)
    "confocal_bench": (REPO_ROOT / "docs" / "engine" / "native_demo" / "confocal_bench.project.json", False, ()),
    # (a dark port: the coherent sum on SCREEN_Y cancels, its target T_OUTY_COHERENT_POWER)
    "module_B": (Path(__file__).parent / "fixtures" / "designs" / "dark_port_module.workspace.json", True, ()),
    # (a fold prism and a lens met a few microradians from normal incidence: |d x n| 4.4e-6 and 1.3e-5)
    "prism_fold": (Path(__file__).parent / "fixtures" / "designs" / "prism_fold.project.json", False, ()),
    # (a two-mirror cavity: a beam reflected exactly back on itself at normal incidence, bounce after bounce)
    "two_mirror_cavity": (Path(__file__).parent / "fixtures" / "cavity" / "two_mirror_cavity_probe.json", False, ()),
    # (a source polarized at 45 degrees: its Jones components equal up to rounding, of opposite signs)
    "plate_blocked": (Path(__file__).parent / "fixtures" / "designs" / "plate_blocked.project.json", False, ()),
    # (a Mach-Zehnder interferometer: its two arms carry the same power)
    "v_mz": (Path(__file__).parent / "fixtures" / "designs" / "mach_zehnder.project.json", False, ()),
}
#: Stored designs turned as a whole before their poses are scaled: (the design, the turn), so that every component of every
#: direction and normal carries rounding, not only the axis-aligned ones.
TURNED_DESIGNS = {"prism_fold_general": ("prism_fold", GENERAL), "confocal_bench_general": ("confocal_bench", GENERAL),
                  "mzi_cascade_x8_general": ("mzi_cascade_x8", GENERAL), "v_mz_general": ("v_mz", GENERAL),
                  "two_mirror_cavity_general": ("two_mirror_cavity", GENERAL), "plate_blocked_general": ("plate_blocked", GENERAL),
                  "plate_blocked_half_yz": ("plate_blocked", HALF_YZ)}


def _turned(document: dict, q) -> dict:
    """``document`` turned rigidly by the unit quaternion ``q`` about the origin: every component pose and every plane pose
    that is not attached to a device (an attached one is stated in its device's frame)."""

    for pose in [c["nominal_pose"] for c in document["components"]] + [
            o["pose"] for o in document["observations"] if "pose" in o and o.get("device_instance_id") is None]:
        pose["position_mm"] = list(rot.rotate(q, pose["position_mm"]))
        pose["quaternion_wxyz"] = list(rot.multiply(q, pose["quaternion_wxyz"]))
    return document


@cache
def _clear_catalog() -> dict:
    return with_lossless_bulk(json.loads(REAL_CATALOG.read_bytes()))


UNIT_INDEX_SETTINGS = tuple((f"environment.{key}", value) for key, value in UNIT_INDEX_ENVIRONMENT.items())


@cache
def _stored_result(name: str, s: float, settings: tuple[tuple[str, float | str], ...] = (), *,
                   clear_glass: bool = False) -> tuple[dict, str]:
    """Run a stored design with scaled quaternions and optional settings (``section.field`` or ``numerical_settings``).
    Tests that need numeric light powers or ghost branches explicitly assume zero absorption for otherwise unspecified
    glass in their in-memory fixture. The published optics and customer Project files retain their original loss data.
    """

    base, turn = TURNED_DESIGNS.get(name, (name, None))
    path, workspace, targets = STORED_DESIGNS[base]
    document = json.loads(path.read_text(encoding="utf-8"))
    document = document["project"] if workspace else document
    if clear_glass:
        for material in document.get("model_materials", []):
            if material.get("bulk_loss") is None and material.get("absorption_coefficient_per_mm") is None:
                material["absorption_coefficient_per_mm"] = "0"
                material["basis"] += "; zero absorption assumed for this pose-scale test"
    document["targets"] = [*document.get("targets", []), *targets]
    for key, value in settings:
        section, _dot, field = key.rpartition(".")
        document[section or "numerical_settings"] = {**document.get(section or "numerical_settings", {}), field: value}
    if turn is not None:
        document = _turned(document, turn)
    for component in document["components"]:
        component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], s)
    for observation in document["observations"]:
        if "pose" in observation:
            observation["pose"]["quaternion_wxyz"] = _scaled(observation["pose"]["quaternion_wxyz"], s)
    catalog = _clear_catalog() if clear_glass else REAL_CATALOG
    run = run_optical_project(document, catalog=catalog)
    return dict(_leaves(run.result.model_dump(mode="json"))), run.local_validation.result_hash.digest


@pytest.mark.parametrize("s", SCALES)
def test_eight_imported_interferometer_cells_with_every_pose_scaled_report_their_unit_result(s):
    """``mzi_cascade_x8`` (eight Mach-Zehnder cells imported into one Project, 480 mm across) with every quaternion scaled
    by ``s``: every record, state and code is the unit run's and every number agrees within the cross-runtime tolerance, a
    coordinate within 128 units in the last place of the system's extent (``_extent_floor_mm``: 1.4e-11 mm)."""

    reference, reference_hash = _stored_result("mzi_cascade_x8", 1.0)
    _assert_unit_result(*_stored_result("mzi_cascade_x8", s), reference, reference_hash,
                        coordinate_floor_mm=_extent_floor_mm(reference))


@pytest.mark.parametrize("s", SCALES)
def test_a_balanced_pair_on_the_phase_seam_reports_one_sign_under_a_scaled_pose(s):
    """A balanced pair whose interface phase is pi sits on the seam of the wrapped phase: its optical path difference is 0
    up to last-bit noise of either sign. Its phase difference reads +3.141593 in the unit and every scaled run (the
    reported interval is (-pi, pi]), and a phase target at 3.141593 on it keeps error 0 whichever order it names the pair
    in (the negation of +3.141593 is -3.141593, which reads +3.141593 again)."""

    for scale in (1.0, s):
        result, _hash = _stored_result("mzi_cascade_x8", scale)
        seam = [p for p, v in result.items() if p.endswith(".phase_difference_rad") and v is not None and abs(abs(v) - math.pi) < 1e-6]
        assert len(seam) > 100, len(seam)
        assert {result[p] for p in seam} == {3.141593}, scale
        for target_id in ("T_SEAM_PHASE", "T_SEAM_REVERSED"):
            (index,) = [p.removesuffix(".target_id") for p, v in result.items() if p.endswith(".target_id") and v == target_id]
            assert (result[f"{index}.value"], result[f"{index}.error"], result[f"{index}.message"]) == (3.141593, 0.0, ""), (scale, target_id)


def test_a_phase_target_that_names_its_pair_in_the_other_order_reads_the_negated_record():
    """Off the seam, a phase_difference_rad target whose path_id_a is the record's path_id_b reads the exact negation of the
    record's reported value. The two-beam phase experiment uses an explicitly lossless copy of the splitter glass."""

    catalog = with_lossless_bulk(catalog_document())
    document = example_project_of_kind("two_beam_interference", catalog=catalog).model_dump(mode="json", by_alias=True, exclude_none=True)
    (record,) = [r for r in run_optical_project(document, catalog=catalog).result.interference
                 if r.observation_id == "crossing_plane" and r.phase_difference_rad not in (None, 0.0)]
    assert abs(abs(record.phase_difference_rad) - math.pi) > 0.1
    document["targets"] = [
        {"target_id": target_id, "metric": "phase_difference_rad", "observation_id": "crossing_plane",
         "parameters": [{"key": "path_id_a", "value": a}, {"key": "path_id_b", "value": b}],
         "target_value": 0.0, "tolerance_abs": 10.0}
        for target_id, a, b in (("T_AB", record.path_id_a, record.path_id_b), ("T_BA", record.path_id_b, record.path_id_a))]
    targets = {t.target_id: t.value for t in run_optical_project(document, catalog=catalog).result.targets}
    assert targets == {"T_AB": record.phase_difference_rad, "T_BA": -record.phase_difference_rad}


@pytest.mark.parametrize("s", SCALES)
def test_a_refracting_bench_with_every_pose_scaled_reports_its_unit_result(s):
    """The native confocal bench with every quaternion scaled by ``s``: its on-axis hits through the lenses read an angle of
    incidence of 0 (the refracted directions carry up to about 4e-9 deg of rounding, below the 1e-7 deg resolution), its
    orthogonally polarized pair a phase difference on the stated convention, and every record, state, code and number is
    the unit run's. No angle of incidence lies near the resolution, where the rounding could still decide it."""

    reference, reference_hash = _stored_result("confocal_bench", 1.0)
    scaled, scaled_hash = _stored_result("confocal_bench", s)
    _assert_unit_result(scaled, scaled_hash, reference, reference_hash)
    for result in (reference, scaled):
        angles = [v for p, v in result.items() if p.endswith(".aoi_deg") and isinstance(v, float)]
        assert angles and any(v == 0.0 for v in angles)
        assert not [v for v in angles if 0.0 < v < 1e-5], sorted(v for v in angles if 0.0 < v < 1e-5)


@pytest.mark.parametrize("s", SCALES)
def test_a_dark_port_reports_a_coherent_power_of_zero_under_a_scaled_pose(s):
    """The coherent sum on a dark port cancels to a residue of a few units in the last place of the beams' powers, of
    either sign (1.1e-19 W in the unit run, -3.3e-19 W with every quaternion scaled by 1 - 8e-7): within 1e-12 of the
    powers it adds it reads 0 W, never a negative power, so the coherent_power_W target, its message and the interval it
    states are the unit run's."""

    reference, reference_hash = _stored_result("module_B", 1.0)
    scaled, scaled_hash = _stored_result("module_B", s)
    _assert_unit_result(scaled, scaled_hash, reference, reference_hash)
    for result in (reference, scaled):
        (index,) = [p.removesuffix(".target_id") for p, v in result.items() if p.endswith(".target_id") and v == "T_OUTY_COHERENT_POWER"]
        assert result[f"{index}.value"] == 0.0
        assert "is 0 W" in result[f"{index}.message"] and "between 0 W and 0.000995 W" in result[f"{index}.message"]
        sums = [p.removesuffix(".total_power_W") for p in result if ".superpositions[" in p and p.endswith(".total_power_W")]
        assert sums and all(result[f"{q}.total_power_W"] is None or result[f"{q}.total_power_W"] >= 0.0 for q in sums)
        for q in sums:
            if result[f"{q}.total_power_W"] == 0.0:
                assert result[f"{q}.interference_power_W"] == -result[f"{q}.incoherent_power_W"]


#: The floor of the basis-following leaves of the fold-prism designs, whose beams meet a surface |d x n| 4.4e-6 from normal
#: incidence: their Jones components move by 4.5e-12 under a scaled pose, outside the cross-runtime 1e-12. The stated
#: resolution is open with the lead (fix3-r8 question 1, accept TRANSPORT_BASIS_RESOLUTION or not); only these designs take
#: it, every other design keeps the cross-runtime tolerance on those leaves.
PRISM_FOLD_BASIS_FLOOR = TRANSPORT_BASIS_RESOLUTION


def _part(leaves: dict, prefix: str) -> dict:
    return {path: value for path, value in leaves.items() if path.startswith(prefix)}


@pytest.mark.parametrize("s", SCALES)
def test_a_beam_a_few_microradians_from_normal_incidence_reports_its_unit_result(s):
    """A fold prism's beam meets the lens behind it a few microradians from normal incidence (|d x n| 4.4e-6 and 1.3e-5),
    where the direction of d x n turns with the last bits of the poses: its transport basis, its Jones vector in that
    basis, its Gaussian axes and its Stokes s1 / s2 read 3e-11 apart, within ``TRANSPORT_BASIS_RESOLUTION``; every other
    number of the result is the unit run's within the cross-runtime tolerance."""

    reference, reference_hash = _stored_result("prism_fold", 1.0)
    scaled, scaled_hash = _stored_result("prism_fold", s)
    _assert_unit_result(scaled, scaled_hash, reference, reference_hash, basis_floor=PRISM_FOLD_BASIS_FLOOR)
    moved = [p for p in reference if _TRANSPORT_BASIS.fullmatch(p) and isinstance(reference[p], float)
             and abs(scaled[p] - reference[p]) > DEFAULT_NUMERICAL_CONFIG.cross_runtime_abs_tol]
    assert all(abs(scaled[p] - reference[p]) <= TRANSPORT_BASIS_RESOLUTION for p in moved), moved


@pytest.mark.parametrize("s", SCALES)
def test_a_design_turned_about_a_general_axis_near_normal_and_grazing_incidence_reports_its_unit_result(s):
    """The fold prism design turned as a whole about a general axis, so that every component of every direction and normal
    carries rounding: every record, state, code, work count and number of the result is the unit run's -- the basis-following
    leaves within ``PRISM_FOLD_BASIS_FLOOR``. The hash is not compared (D3-A1 does not require it): the angles of incidence
    of its ghost beams a few hundredths of a degree from normal follow the poses' last bits by about 4e-11 relative, within
    the cross-runtime tolerance but beyond the 1e-12 at which a value on a 9-digit rounding boundary may move. Among them the exposures of its ghost beams that graze the prism's faces (|cos(AOI)| down to 5e-9): a footprint is
    stated only where |cos(AOI)| is at least ``FOOTPRINT_COS_RESOLUTION`` (its long axis w / cos(AOI) would otherwise follow
    the rounding of the beam's direction), a beam passing beside a face it grazes has no known size there, and a rating's
    largest beam diameter is stated, or not, in every run alike."""

    from os_optics.validation.exposures import FOOTPRINT_COS_RESOLUTION

    reference, _hash = _stored_result("prism_fold_general", 1.0, clear_glass=True)
    scaled, _hash = _stored_result("prism_fold_general", s, clear_glass=True)
    _assert_unit_result(scaled, None, reference, None, basis_floor=PRISM_FOLD_BASIS_FLOOR)
    grazing = math.degrees(math.acos(FOOTPRINT_COS_RESOLUTION))
    hits = {p.removesuffix(".aoi_deg") for p, v in reference.items()
            if re.fullmatch(r"\.exposures\.surfaces\[\d+\]\.(hits|untraced_hits|beside_hits)\[\d+\]\.aoi_deg", p) and v > grazing}
    assert hits
    for hit in hits:
        assert reference.get(f"{hit}.footprint_semi_axes_mm") is None and f"{hit}.footprint_semi_axes_mm[0]" not in reference, hit
        if ".beside_hits[" in hit:
            assert reference[f"{hit}.beam_radius_x_mm"] is None and reference[f"{hit}.status"] == "UNKNOWN", hit


#: The work of the fold prism's return screen at unit quaternions in the reference's unit-index medium.
PRISM_FOLD_GENERAL_RETURN_WORK = 558992


def _return_analysis(result: dict) -> tuple[dict, str, list[str]]:
    """The return analysis's records (work counts, terminations), the run's termination and its issue codes."""

    return ({p: v for p, v in result.items() if p.startswith(".outcome.execution.budgets.return_analysis[")},
            result[".outcome.execution.termination"], sorted({v for p, v in result.items() if re.fullmatch(r"\.issues\[\d+\]\.code", p)}))


@pytest.mark.parametrize("s", SCALES)
def test_the_return_analysis_of_a_design_turned_about_a_general_axis_does_the_unit_runs_work(s):
    """The fold prism design turned about a general axis: the iterations of the solid folds' first-return sums stop at a
    stated relative resolution (``CERTIFY_RESOLUTION``), not at the bit where their sums repeat, so the return analysis's
    work count (562346 / 562395 / 562496 when they stopped at that bit), its solid-fold work and its termination are the
    unit run's under a scaled pose; a budget just below that work stops every run the same way."""

    from os_optics.validation.solid_folds import CERTIFY_RESOLUTION, MARGIN

    assert 2.0 ** -52 < CERTIFY_RESOLUTION < MARGIN
    first = ".outcome.execution.budgets.return_analysis[0]."
    settings = UNIT_INDEX_SETTINGS
    reference = _return_analysis(_stored_result("prism_fold_general", 1.0, settings, clear_glass=True)[0])
    assert reference[0][first + "termination"] == "RETURN_FOUND"
    assert reference[0][first + "work_done"] == PRISM_FOLD_GENERAL_RETURN_WORK
    assert _return_analysis(_stored_result("prism_fold_general", s, settings, clear_glass=True)[0]) == reference

    budget = (*settings, ("compute_budget.max_return_analysis_work", PRISM_FOLD_GENERAL_RETURN_WORK - 7))
    stopped = _return_analysis(_stored_result("prism_fold_general", 1.0, budget, clear_glass=True)[0])
    assert stopped[0][first + "termination"] == "BUDGET_EXHAUSTED"
    assert _return_analysis(_stored_result("prism_fold_general", s, budget, clear_glass=True)[0]) == stopped


#: Turned designs whose beams of equal power (the halves of a 50:50 split, symmetric ghosts) carry powers a few units in the
#: last place apart that follow the poses' last bits, and the budget each reaches: the probes following its untraced light
#: (``confocal_bench``, whose ghosts also run nearly parallel to an observation plane with an active area), its primary
#: trace's paths (``mzi_cascade_x8``), none (``v_mz``).
EQUAL_POWER_DESIGNS = {"confocal_bench_general": "untraced_light", "mzi_cascade_x8_general": "primary", "v_mz_general": None}


def _listed(result: dict, pattern: str) -> list:
    return [value for path, value in result.items() if re.fullmatch(pattern, path)]


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("name", sorted(EQUAL_POWER_DESIGNS))
def test_beams_of_equal_power_are_traced_in_one_order_under_a_scaled_pose(name, s):
    """A design turned about a general axis, with every quaternion scaled by ``s``: beams whose powers are equal in exact
    arithmetic are ordered on the run's power grid (``scheduler.order_power``), so the trace queue and the probes follow
    them in queue order in every run, and a budget that runs out leaves the same beams untraced. Every record, issue (its
    numerical context and message), followed-light entry, ledger entry and their order, traced path and affected path is
    the unit run's, and every number agrees within the cross-runtime tolerance. A chief ray nearly parallel to an
    observation plane with an active area (10^14 mm to its crossing) does not cross it off the area beyond the scene's
    reach, whatever the last bits of its direction."""

    # The confocal ghost probes depend on known power after traversing glass.
    clear_glass = name == "confocal_bench_general"
    # Preserve the original unit-index medium in the MZI queue test; the test is about ties from equal splitting.
    settings = UNIT_INDEX_SETTINGS if name == "mzi_cascade_x8_general" else ()
    reference, _hash = _stored_result(name, 1.0, settings, clear_glass=clear_glass)
    scaled, _hash = _stored_result(name, s, settings, clear_glass=clear_glass)
    phase = EQUAL_POWER_DESIGNS[name]
    exhausted = _listed(reference, rf"\.outcome\.execution\.budgets\.exhausted_by_phase\.{phase}\[\d+\]")
    assert bool(exhausted) == (phase is not None), exhausted
    for pattern in (r"\.paths\[\d+\]\.path_id", r"\.outcome\.execution\.budgets\.affected_path_ids\[\d+\]",
                    r"\.power_budget\.entries\[\d+\]\.(path_id|source_id|kind|power_W)",
                    r"\.observations\[\d+\]\.followed_light\[\d+\]\.(path_id|surface_id|kind|reach|power_W)",
                    r"\.issues\[\d+\]\.(code|message)", r"\.issues\[\d+\]\.numerical_context\..*"):
        want, got = _listed(reference, pattern), _listed(scaled, pattern)
        assert want or "affected" in pattern or "followed" in pattern, pattern
        assert len(got) == len(want) and all(
            _close([g], [w], DEFAULT_NUMERICAL_CONFIG.cross_runtime_rel_tol, DEFAULT_NUMERICAL_CONFIG.cross_runtime_abs_tol)
            if isinstance(w, float) and isinstance(g, float) else g == w for g, w in zip(got, want)), pattern
    _assert_unit_result(scaled, None, reference, None, coordinate_floor_mm=_extent_floor_mm(reference))


_CAVITY_FOOTPRINT = re.compile(r"\.exposures\.surfaces\[\d+\]\.hits\[\d+\]\.footprint_semi_axes_mm\[\d\]")


@pytest.mark.parametrize("s", (1.0, *SCALES))
def test_a_turned_cavity_keeps_its_beams_footprint_bounce_after_bounce(s):
    """A two-mirror cavity turned about a general axis, its quaternions scaled by ``s``: the beam is reflected exactly back
    on itself at every bounce, where its Gaussian axes are turned by pi about themselves. Its footprint on the mirror reads
    the unturned design's 0.500584 mm at every hit (the turn's axis unit only to rounding grew it to 5.4 mm by the 24th
    hit), the run ends UNKNOWN on its interaction limit alone, and the scaled runs report the unit run's result."""

    plain, _hash = _stored_result("two_mirror_cavity", 1.0)
    turned, _hash = _stored_result("two_mirror_cavity_general", s)
    want = [v for p, v in plain.items() if _CAVITY_FOOTPRINT.fullmatch(p)]
    got = [v for p, v in turned.items() if _CAVITY_FOOTPRINT.fullmatch(p)]
    assert len(got) == len(want) > 40 and abs(want[0] - 0.500584) < 1e-6
    assert _close(got, want, DEFAULT_NUMERICAL_CONFIG.cross_runtime_rel_tol, 0.0), max(abs(g - w) for g, w in zip(got, want))
    assert turned[".status"] == "UNKNOWN"
    assert sorted({v for p, v in turned.items() if re.fullmatch(r"\.issues\[\d+\]\.code", p)}) == ["MAX_INTERACTIONS"]
    if s != 1.0:
        _assert_unit_result(turned, None, _stored_result("two_mirror_cavity_general", 1.0)[0], None)


@pytest.mark.parametrize("s", SCALES)
def test_a_beam_polarized_at_45_degrees_reports_one_phase_under_a_scaled_pose(s):
    """A source polarized at 45 degrees, its design turned about a general axis and every quaternion scaled by ``s``: the
    beam's Jones components are equal up to the rounding of their transport and of opposite signs, so a phase referred
    to whichever is larger by its last bits jumps by pi. The phase is referred to Ex unless |Ey|^2 exceeds |Ex|^2 by more
    than ``DOMINANT_COMPONENT_REL_TOL``: it reads the unit run's and the unturned design's value, and so does every other
    record and number of the result."""

    from os_optics.validation.observations import DOMINANT_COMPONENT_REL_TOL

    reference, _hash = _stored_result("plate_blocked_general", 1.0, clear_glass=True)
    scaled, _hash = _stored_result("plate_blocked_general", s, clear_glass=True)
    plain, _hash = _stored_result("plate_blocked", 1.0, clear_glass=True)
    beam = ".observations[0].arrivals[0].beam"
    ex, ey = (complex(*(reference[f"{beam}.polarization.{c}[{k}]"] for k in (0, 1))) for c in ("ex", "ey"))
    assert abs(abs(ex) ** 2 / abs(ey) ** 2 - 1.0) < DOMINANT_COMPONENT_REL_TOL and (ex / ey).real < 0.0
    phase = reference[f"{beam}.phase_rad"]
    assert phase is not None and scaled[f"{beam}.phase_rad"] == phase == plain[f"{beam}.phase_rad"]
    assert phase == reported_phase((_k0_opl(reference, beam) + math.atan2(ex.imag, ex.real) + math.pi) % (2.0 * math.pi) - math.pi)
    _assert_unit_result(scaled, None, reference, None)


def _k0_opl(result: dict, beam: str) -> float:
    return 2.0 * math.pi / (result[f"{beam}.wavelength_nm"] * 1e-6) * result[f"{beam}.optical_path_length_mm"]


def test_the_power_grid_puts_a_cascade_of_equal_splits_on_whole_steps():
    """The run's power grid is a binary fraction of its launched power, so the powers of a cascade of 50:50 splits (the
    launched power times 2^-n) are whole steps and a few units in the last place either side read the same step; on a
    decimal grid (1e-12 of the launched power) 2^-13 of it would sit on a midpoint, where those bits decide the step."""

    from os_optics.optics.rays.scheduler import order_power, order_quantum

    launched = 1e-3
    quantum = order_quantum(DEFAULT_NUMERICAL_CONFIG.power_ledger_fraction_resolution, launched)
    assert quantum == launched * 2.0 ** -40
    for n in range(1, 40):
        exact = launched * 2.0 ** -n
        steps = {order_power(exact * (1.0 + k * 2.0 ** -52), quantum) for k in (-8, -1, 0, 1, 8)}
        assert steps == {2.0 ** (40 - n)}, (n, steps)
    decimal = launched * DEFAULT_NUMERICAL_CONFIG.power_ledger_fraction_resolution
    assert abs(launched * 2.0 ** -13 / decimal % 1.0 - 0.5) < 1e-6
    assert order_power(launched, 0.0) == launched


#: A caller-stated Gaussian angle-of-incidence limit: the Gaussian state then records the angle of each focusing step it
#: checked (a ``gaussian_max_aoi_deg`` validity entry with the observed angle).
GAUSSIAN_AOI_STATED = (("gaussian_max_aoi_deg", 70.0),)


def _gaussian_aoi_checks(result: dict) -> list[float]:
    return [result[p.removesuffix(".parameter") + ".observed"] for p, v in result.items()
            if v == "aoi_deg" and re.search(r"\.gaussian\.value_basis\..*\.validity\[\d+\]\.parameter$", p)]


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("name", ["confocal_bench"])
def test_a_stated_gaussian_angle_limit_reads_normal_incidence_as_zero_under_a_scaled_pose(name, s):
    """With ``gaussian_max_aoi_deg`` stated, the Gaussian state records the angle of incidence of each focusing step it
    checked -- curved lens surfaces met on their axis (``confocal_bench``). That angle is the incidence of the beam on the surface's normal (``incidence_deg``), so a beam at
    normal incidence reads 0 in the unit and the scaled run alike, not the 1e-6 deg its cosine's rounding would give, and
    the result, its hash included, is the unit run's."""

    reference, reference_hash = _stored_result(name, 1.0, GAUSSIAN_AOI_STATED)
    scaled, scaled_hash = _stored_result(name, s, GAUSSIAN_AOI_STATED)
    _assert_unit_result(scaled, scaled_hash, reference, reference_hash)
    for result in (reference, scaled):
        checks = _gaussian_aoi_checks(result)
        assert checks and set(checks) == {0.0}, checks


def test_the_share_beside_a_support_is_resolved_upward_to_the_ledger_resolution():
    """The share beyond a support, ``1 - inside``, is resolved upward to ``power_ledger_fraction_resolution``: two fractions
    a few units in the last place apart give one share, never below ``1 - inside`` (it bounds the light left out from above),
    and a whole beam inside gives none."""

    from types import SimpleNamespace

    from os_optics.optics.rays.tracer import Tracer

    tracer = SimpleNamespace(cfg=DEFAULT_NUMERICAL_CONFIG)
    quantum = DEFAULT_NUMERICAL_CONFIG.power_ledger_fraction_resolution
    for inside in (0.9999999931137297, 0.9999992469161214, 0.9999999997784799, 0.75):
        near = float(np.nextafter(np.nextafter(inside, 2.0), 2.0))
        share = Tracer._beyond_share(tracer, inside)
        assert share == Tracer._beyond_share(tracer, near) and 1.0 - inside <= share < 1.0 - inside + quantum, inside
    assert Tracer._beyond_share(tracer, 1.0) == 0.0


def test_an_orthogonal_pair_reports_its_phase_difference_on_the_stated_convention_whatever_its_overlap_noise():
    """Two beams polarized along x and along y up to a 1e-12 component of either sign or phase: their polarization overlap
    is rounding noise, whose argument is the direction of that noise, not a phase of the fields. The phase difference is
    k0 (OPL_a - OPL_b) wrapped (here 0) in every case, an ESTIMATE whose entry names the overlap below
    energy_conservation_tol; the overlap itself is reported as computed."""

    from os_optics.config import NumericalConfig
    from os_optics.optics.interference import analyse_pair
    from test_interferometer_open_phase import _record

    config = NumericalConfig()
    a = _record("a", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 100.0)
    pairs = []
    for noise in (1e-12, -1e-12, 1e-12j, -1e-12j, 0.0):
        b = _record("b", 1e-3, (0.0, 0.0), 1.0, 90.0, (), 100.0, 100.0)
        b.jones = np.array([noise + 0j, 1.0 + 0j])
        pair, issues = analyse_pair(a, b, config)
        assert "ORTHOGONAL_POLARIZATION" in pair.issue_codes and pair.polarization_overlap <= 1.1e-12
        pairs.append(pair)
    assert {pair.phase_difference_rad for pair in pairs} == {0.0}
    for pair in pairs:
        entry = pair.value_basis["phase_difference_rad"]
        assert entry.kind.value == "ESTIMATE" and not entry.assumed and not entry.omitted
        (check,) = entry.validity
        assert (check.parameter, check.minimum, check.within, check.setting, check.code) == (
            "polarization_overlap", config.energy_conservation_tol, False, "energy_conservation_tol", "ORTHOGONAL_POLARIZATION")
        assert check.observed == pair.polarization_overlap
    # (a pair whose overlap is above the limit keeps the overlap's phase, with no entry: pi for opposite fields)
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 180.0, (), 100.0, 100.0)
    pair, _issues = analyse_pair(a, b, config)
    assert pair.phase_difference_rad == 3.141593 and "phase_difference_rad" not in (pair.value_basis or {})


def test_an_angle_of_incidence_next_to_normal_is_taken_from_the_sine():
    """One unit in the last place of a cosine next to 1 is 8.5e-7 deg: near normal incidence the angle is taken from
    ``|n x d|`` -- a tilt of 4e-9 rad, whose cosine rounds to 1, reads 2.3e-7 deg -- and an angle below the 1e-7 deg a
    refracted direction resolves reads 0 (a direction a last bit off the normal, and a tilt of 1e-9 rad, 5.7e-8 deg), in
    the scalar, one-row and many-row paths alike; away from normal incidence the angle is ``acos_deg``'s, bit for bit."""

    n = np.array([0.0, 0.0, 1.0])
    assert float(acos_deg(np.nextafter(1.0, 0.0))) > 5e-7  # (the cosine's floor the record threshold used to sit below)
    off = np.array([2.0 ** -53, 0.0, 1.0])
    tilt = np.array([4e-9, 0.0, 1.0])
    below = np.array([1e-9, 0.0, 1.0])
    rows = np.array([off, tilt, [0.3, 0.1, 0.9486832980505138], below])
    cos_rows = np.minimum(np.abs(rows @ n), 1.0)
    assert cos_rows[1] == 1.0 and float(acos_deg(cos_rows[1])) == 0.0
    many = incidence_deg(cos_rows, rows, n)
    for k in range(4):
        single = float(incidence_deg(float(cos_rows[k]), rows[k], n))
        assert float(incidence_deg(cos_rows[k:k + 1], rows[k:k + 1], n)[0]) == single == float(many[k])
    assert float(many[0]) == 0.0 and float(many[3]) == 0.0
    assert abs(float(many[1]) - math.degrees(4e-9)) <= 1e-12 * math.degrees(4e-9)
    assert float(many[2]) == float(acos_deg(cos_rows[2]))
    tilted = np.array([math.sin(math.radians(0.5)), 0.0, math.cos(math.radians(0.5))])
    assert abs(float(incidence_deg(float(tilted @ n), tilted, n)) - 0.5) <= 1e-12


# ------------------------------------------------------------------------------------------------ return analysis

_MZI_WINDOW = Path(__file__).parent / "fixtures" / "requests" / "mzi_window.json"
#: The return analysis's work on ``mzi_window`` with unit quaternions.
MZI_WINDOW_RETURN_WORK = 104202


def _every_request_pose_scaled(document, s: float):
    if isinstance(document, dict):
        return {key: [c * s for c in value] if key == "quaternion_wxyz" else _every_request_pose_scaled(value, s)
                for key, value in document.items()}
    if isinstance(document, list):
        return [_every_request_pose_scaled(value, s) for value in document]
    return document


@cache
def _mzi_window_return(s: float, budget: int | None = None):
    document = _every_request_pose_scaled(json.loads(_MZI_WINDOW.read_text(encoding="utf-8")), s)
    if budget is not None:
        document["configuration"]["numerical"]["max_return_analysis_work"] = budget
    result = validate(ValidationRequest.model_validate(document))
    execution = result.outcome.execution
    return ([r.model_dump(mode="json") for r in execution.budgets.return_analysis], execution.termination,
            sorted({i.code for i in result.issues}))


@pytest.mark.parametrize("s", SCALES)
def test_coplanar_faces_are_screened_for_returns_the_same_under_a_scaled_pose(s):
    """The window and splitter faces of ``mzi_window`` are coplanar pairs: under in-tolerance scaling ``q / |q|`` puts them
    exactly level or a few 1e-14 mm apart. A side or overlap test on surface extents resolves lengths to
    ``SIDE_ABS_TOL_MM + SIDE_REL_TOL`` times the coordinates compared (``EITHER`` inside it), so the return analysis --
    its records and its work count -- is the unit run's."""

    from os_optics.validation.resonance import SIDE_ABS_TOL_MM, SIDE_REL_TOL

    assert (SIDE_ABS_TOL_MM, SIDE_REL_TOL) == (1e-9, 1e-12)
    reference = _mzi_window_return(1.0)
    assert [r["work_done"] for r in reference[0]] == [MZI_WINDOW_RETURN_WORK]
    assert _mzi_window_return(s) == reference


@pytest.mark.parametrize("s", SCALES)
def test_a_return_analysis_budget_next_to_the_work_ends_the_same_under_a_scaled_pose(s):
    """A budget between the work counts the unit and a scaled pose used to reach (104192 and 104202) stops both the same way."""

    reference = _mzi_window_return(1.0, MZI_WINDOW_RETURN_WORK - 7)
    assert [(r["termination"], r["closed_return_screen"]) for r in reference[0]] == [("BUDGET_EXHAUSTED", "INCOMPLETE")]
    assert reference[1] == TerminationReason.COMPLETED and "RETURN_ANALYSIS_BUDGET" in reference[2]
    assert _mzi_window_return(s, MZI_WINDOW_RETURN_WORK - 7) == reference


# ------------------------------------------------------------------------------------------------ local assembly validation


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("q", [QUARTER_TURN_Z, GENERAL])
def test_an_assembly_pose_within_the_norm_tolerance_is_taken_by_its_direction(q, s):
    from os_optics.assembly.transforms import Rigid

    reference, scaled = Rigid.from_pose((10.0, -4.0, 2.5), q), Rigid.from_pose((10.0, -4.0, 2.5), _scaled(q, s))
    for v in ((100.0, 0.0, 0.0), (0.0, 50.0, -20.0), (12.5, -3.0, 40.0)):
        assert _close(scaled.apply(v), reference.apply(v))
    for index in range(3):
        assert abs(math.sqrt(sum(c * c for c in scaled.axis(index))) - 1.0) <= 1e-15


def test_the_frozen_quaternion_tolerance_says_who_refuses_and_that_the_engine_takes_q_over_its_length():
    """``quaternion_norm_tol`` is a refusal bound of the shared contract and the Engine input builder: the Engine takes any
    non-zero quaternion as ``q / |q|`` (no computation uses an in-tolerance quaternion's components unchanged), and a zero
    one is refused by the request contract with the code a caller receives (``INVALID_INPUT``)."""

    from os_optics.config import NumericalConfig
    from os_optics.math.rotation import quat_normalize

    text = NumericalConfig.model_fields["quaternion_norm_tol"].description
    assert "refused" in text and "POSE_INVALID" in text and "q / |q|" in text and "INVALID_INPUT" in text
    assert "used unchanged" not in text and "ZERO_LENGTH_VECTOR" not in text
    q = _scaled(GENERAL, 1.0 - 8e-7)
    assert list(quat_normalize(q)) != q  # (an in-tolerance quaternion is divided by its length, not used as given)
    assert abs(math.sqrt(sum(c * c for c in quat_normalize(q))) - 1.0) <= 1e-15


_APERTURE_CLIPPING = Path(__file__).parent / "fixtures" / "requests" / "aperture_clipping.json"


def _first_pose_scaled(scale: float | None) -> dict:
    """``aperture_clipping`` (its first component's quaternion is (0.5, 0.5, 0.5, 0.5)) with that quaternion times
    ``scale`` (``None``: zero)."""

    document = json.loads(_APERTURE_CLIPPING.read_text(encoding="utf-8"))
    pose = document["project"]["components"][0]["pose"]
    assert pose["quaternion_wxyz"] == [0.5, 0.5, 0.5, 0.5]
    pose["quaternion_wxyz"] = [0.0] * 4 if scale is None else [c * scale for c in pose["quaternion_wxyz"]]
    return document


def test_a_zero_quaternion_is_refused_by_the_request_contract_and_any_other_length_is_taken_by_its_direction():
    """Through the public JSON path: a zero quaternion is ``INVALID_INPUT`` at its field (the request contract); one whose
    squared components overflow (1e308) or underflow (1e-170) states the rotation of ``q / |q|`` like any other non-zero
    length -- the run is the unit quaternion's (its result hash; only the stated input differs)."""

    from os_optics.math.rotation import quat_normalize
    from os_optics.validation.pipeline import validate_json

    refused = json.loads(validate_json(json.dumps(_first_pose_scaled(None))))
    assert refused["status"] == "FAIL"
    assert [i["code"] for i in refused["issues"]] == ["INVALID_INPUT"]
    assert "project.components.0.pose.quaternion_wxyz" in refused["issues"][0]["message"]
    unit = json.loads(validate_json(json.dumps(_first_pose_scaled(1.0))))
    for scale in (1e308, 1e-170):
        scaled = json.loads(validate_json(json.dumps(_first_pose_scaled(scale))))
        assert scaled["status"] == unit["status"] and [i["code"] for i in scaled["issues"]] == [i["code"] for i in unit["issues"]], scale
        assert scaled["manifest"]["result_hash"] == unit["manifest"]["result_hash"], scale
        assert list(quat_normalize([0.5 * scale] * 4)) == [0.5] * 4
    # (the arithmetic of a quaternion of ordinary length is unchanged: q / |q| as before)
    q = [3.0 * c for c in GENERAL]
    length = math.sqrt(sum(c * c for c in q))
    assert list(quat_normalize(q)) == [c / length for c in q]


#: Turns that put the rounding of the poses on a polarizer's blocked share (``_ledger_run``).
LEDGER_TURNS = {"quarter_z": QUARTER_TURN_Z, "half_yz": HALF_YZ}
#: A quarter turn about the beam axis (+x): the polarization example's polarizer crossed with its light.
_CROSS = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)


@cache
def _ledger_run(design: str, turn: str, s: float) -> tuple[list, list, dict]:
    """The power budget of ``design`` -- the ``polarization`` example (its polarizer along the light), ``crossed`` (that
    polarizer turned a quarter turn about the beam: across the light) or the stored ``confocal_bench`` -- turned by
    ``LEDGER_TURNS[turn]`` with every quaternion scaled by ``s``: its entries (kind, path, surface, source, line), its
    elements' sinks, and every other leaf of the result."""

    if design == "confocal_bench":
        document = json.loads(STORED_DESIGNS["confocal_bench"][0].read_text(encoding="utf-8"))
    else:
        document = example_project_of_kind("polarization").model_dump(mode="json", by_alias=True, exclude_none=True)
        if design == "crossed":
            (polarizer,) = [c for c in document["components"] if c["instance_id"] == "POL"]
            polarizer["nominal_pose"]["quaternion_wxyz"] = list(rot.multiply(_CROSS, polarizer["nominal_pose"]["quaternion_wxyz"]))
    document = _turned(document, LEDGER_TURNS[turn])
    for component in document["components"]:
        component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], s)
    for observation in document["observations"]:
        if "pose" in observation:
            observation["pose"]["quaternion_wxyz"] = _scaled(observation["pose"]["quaternion_wxyz"], s)
    result = run_optical_project(document, catalog=REAL_CATALOG if design == "confocal_bench" else None).result.model_dump(mode="json")
    budget = result["power_budget"]
    entries = [(e["kind"], e.get("path_id"), e.get("surface_id"), e.get("source_id"), e.get("spectral_index")) for e in budget["entries"]]
    sinks = [(e["element_id"], sorted(e["sinks"])) for e in budget["by_element"]]
    return entries, sinks, dict(_leaves(result))


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("turn", sorted(LEDGER_TURNS))
@pytest.mark.parametrize("design", ["polarization", "crossed", "confocal_bench"])
def test_a_polarizer_books_no_ledger_entry_for_the_rounding_of_its_blocked_share_under_a_scaled_pose(design, turn, s):
    """A polarizer along its light (the ``polarization`` example, the confocal bench's analyser) blocks a share of rounding
    noise (1e-19 W of 1e-3 W), and one across its light (``crossed``) passes one: neither is a loss. Turned about the test
    axes with every quaternion scaled by ``s``, the power budget lists the unit run's entries in its order, each power within
    the cross-runtime tolerance, and the unit run's sinks per element."""

    entries, sinks, leaves = _ledger_run(design, turn, s)
    unit_entries, unit_sinks, unit_leaves = _ledger_run(design, turn, 1.0)
    assert entries == unit_entries
    assert sinks == unit_sinks
    if design == "crossed":
        assert ("ELEMENT_TRANSMISSION_LOSS", "SRC#0", "POL/element", "SRC", 0) not in entries
        assert ("POLARIZER_REJECTION", "SRC#0", "POL/element", "SRC", 0) in entries
    numerical = DEFAULT_NUMERICAL_CONFIG
    powers = [leaves[f".power_budget.entries[{i}].power_W"] for i in range(len(entries))]
    unit_powers = [unit_leaves[f".power_budget.entries[{i}].power_W"] for i in range(len(entries))]
    assert _close(powers, unit_powers, numerical.cross_runtime_rel_tol, numerical.cross_runtime_abs_tol), (powers, unit_powers)
    assert min(powers) > numerical.energy_conservation_tol * 1e-3  # (no entry of rounding noise: the sources launch 1 mW)


# ------------------------------------------------------------------------------------------------ polarization major axis

#: Turns of a bundled example: about the test axes (``QUARTER_TURN_Z`` puts the ``polarization`` example's major axis at 45
#: degrees in the design frame).
EXAMPLE_TURNS = {"quarter_z": QUARTER_TURN_Z, "half_yz": HALF_YZ}


@cache
def _turned_example_result(kind: str, turn: str, s: float) -> tuple[dict, str]:
    """The result leaves and result hash of the bundled example ``kind`` turned by ``EXAMPLE_TURNS[turn]`` with every
    component and observation quaternion scaled by ``s``."""

    document = _turned(example_project_of_kind(kind).model_dump(mode="json", by_alias=True, exclude_none=True), EXAMPLE_TURNS[turn])
    for component in document["components"]:
        component["nominal_pose"]["quaternion_wxyz"] = _scaled(component["nominal_pose"]["quaternion_wxyz"], s)
    for observation in document["observations"]:
        if "pose" in observation:
            observation["pose"]["quaternion_wxyz"] = _scaled(observation["pose"]["quaternion_wxyz"], s)
    run = run_optical_project(ProjectDocumentV2.model_validate(document))
    return dict(_leaves(run.result.model_dump(mode="json"))), run.local_validation.result_hash.digest


def _major_axes(leaves: dict) -> dict:
    return {p: v for p, v in leaves.items() if ".polarization_state.major_axis[" in p}


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("turn", sorted(EXAMPLE_TURNS))
def test_the_polarization_example_turned_about_a_test_axis_reports_its_unit_major_axis(turn, s):
    """The ``polarization`` example turned about the test axes with every quaternion scaled by ``s``: a major axis at 45
    degrees in the design frame has two components equal up to last-bit noise, and its sign is set by the first component
    within ``MAJOR_AXIS_TIE_REL_TOL`` of the largest, so every major axis -- and every other record and number of the
    result -- is the unit run's."""

    reference, reference_hash = _turned_example_result("polarization", turn, 1.0)
    assert _major_axes(reference)
    _assert_unit_result(*_turned_example_result("polarization", turn, s), reference, reference_hash,
                        coordinate_floor_mm=_extent_floor_mm(reference))


@pytest.mark.parametrize("s", SCALES)
def test_a_source_polarized_at_45_degrees_turned_about_a_test_axis_reports_its_unit_major_axis(s):
    """``plate_blocked`` (a source polarized at 45 degrees) turned about the y = z diagonal with every quaternion scaled by
    ``s``: its major axes, and every other record and number of the result, are the unit run's."""

    reference, reference_hash = _stored_result("plate_blocked_half_yz", 1.0)
    assert _major_axes(reference)
    _assert_unit_result(*_stored_result("plate_blocked_half_yz", s), reference, reference_hash,
                        coordinate_floor_mm=_extent_floor_mm(reference))


@pytest.mark.parametrize("ulps", [-1, 0, 1])
@pytest.mark.parametrize("signs", [(1.0, -1.0), (-1.0, 1.0)])
def test_a_major_axis_at_45_degrees_takes_its_sign_from_the_first_tied_component(signs, ulps):
    """A linear Jones vector at 45 degrees whose two components differ in magnitude by up to one unit in the last place,
    either way: the major axis is the one whose x component is positive (the first within ``MAJOR_AXIS_TIE_REL_TOL`` of the
    largest magnitude), whichever component the last bit makes larger."""

    from os_optics.validation.observations import MAJOR_AXIS_TIE_REL_TOL, jones_summary

    assert MAJOR_AXIS_TIE_REL_TOL == 1e-9
    a = b = math.sqrt(0.5)
    for _ in range(abs(ulps)):
        b = math.nextafter(b, math.inf if ulps > 0 else 0.0)
    e1, e2 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    summary = jones_summary(complex(signs[0] * a), complex(signs[1] * b), e1, e2, e1, e2)
    major = summary.major_axis
    assert major[0] > 0.0 and major[1] < 0.0 and major[2] == 0.0, major
    assert _close(major, (math.sqrt(0.5), -math.sqrt(0.5), 0.0), rel=0.0, floor=4e-16), major
    assert summary.azimuth_in_plane_deg == pytest.approx(135.0, abs=1e-12)
