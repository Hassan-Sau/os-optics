"""Adversarial geometry tests of the local assembly layout / gross-collision checker (``os_optics.assembly``).

Every perturbed package is re-signed through the canonical ``build_assembly_preview_package`` (``assembly_helpers``),
so each test isolates the CHECKER's semantics: the package always verifies, only the geometry / poses are wrong.

Sections: reference acceptance; determinism; verification before geometry; off-board; collisions; expected
engagement; broken support; missing / bad geometry; rotations; beam axis; semantic holes found by review (now fixed).
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from assembly_helpers import (
    BOARD_REF,
    FIXTURE_ZIP,
    IDENTITY_Q,
    REFERENCE_PARTS,
    STACK,
    SynthMate,
    SynthPart,
    apply,
    axis_angle,
    board,
    box,
    check_status,
    combine,
    cylinder,
    inverted,
    issue_keys,
    issues,
    mate,
    metric,
    move_world,
    mutate,
    normalized,
    part,
    part_status,
    pose_dict,
    pose_of,
    q_conj,
    reference_files,
    reference_package,
    resync_mate,
    rotate,
    rotation_about,
    set_geometry,
    standing,
    statuses,
    synthetic_package,
    translate_parts,
    without_last_triangles,
)
from os_optics._vendor.shared_contracts.assembly_preview import (
    ASSEMBLY_CHECK_ORDER,
    AssemblyIssueV1,
    json_file_bytes,
    package_archive_bytes,
)
from os_optics.assembly import (
    DEFAULT_TOLERANCES,
    AssemblyLocalValidationV1,
    AssemblyPreviewPackageError,
    AssemblyPreviewPackageV1,
    load_assembly_preview,
    result_file_bytes,
    validate_assembly_layout,
)
from os_optics.assembly import checks as checks_module
from os_optics.assembly import overlap as overlap_module
from os_optics.assembly.mesh import read_part_mesh

ENGINE_ROOT = Path(__file__).resolve().parents[1]
OPTIC, MOUNT, POST, HOLDER, BOARD = REFERENCE_PARTS
CONTEXT_CODES = {"CUSTOMER_SUPPLIED_COMPONENT"}  # present in every reference-derived result
GEOMETRIC_CHECKS = ("SUPPORT_ATTACHMENT", "BOARD_CONTAINMENT", "BOARD_FIXATION", "EXPECTED_ENGAGEMENT", "PART_COLLISION",
                    "BEAM_AXIS_CLEARANCE")


def problems(result) -> list[tuple[str, tuple[str, ...], str | None]]:
    """Issues other than the two context issues every reference-derived result carries."""

    return [k for k in issue_keys(result) if k[0] not in CONTEXT_CODES]


def non_pass(result) -> dict[str, str]:
    return {k: v for k, v in statuses(result).items() if v not in ("PASS", "NOT_APPLICABLE")}


@pytest.fixture(scope="module")
def ref_package() -> AssemblyPreviewPackageV1:
    return reference_package()


@pytest.fixture(scope="module")
def ref_result(ref_package):
    return validate_assembly_layout(ref_package)


# =============================================================================================== 1. reference


def test_reference_statuses_are_exact(ref_result):
    assert ref_result.status.value == "PASS"
    assert non_pass(ref_result) == {}
    assert [c.check for c in ref_result.checks] == list(ASSEMBLY_CHECK_ORDER)
    assert all(c.status.value == "PASS" for c in ref_result.checks)
    assert issue_keys(ref_result) == [("CUSTOMER_SUPPLIED_COMPONENT", (), None)]
    assert ref_result.summary.interference.value == "NO_INTERFERENCE"
    assert {p.part_ref: p.status.value for p in ref_result.parts} == {ref: "PASS" for ref in REFERENCE_PARTS}


def test_reference_chain_is_really_evaluated_not_skipped(ref_package, ref_result):
    """optic -> mount -> post -> holder/base -> board: every contact evaluated, every other pair evaluated."""

    evaluated = {c.check.value: c.evaluated for c in ref_result.checks}
    assert evaluated["PACKAGE_INTEGRITY"] == 7  # manifest + scene + 5 GLBs
    assert evaluated["PART_GEOMETRY"] == 5
    assert evaluated["SUPPORT_STRUCTURE"] == 4
    assert evaluated["SUPPORT_ATTACHMENT"] == 8  # 4 mates + 4 support edges
    assert evaluated["BOARD_CONTAINMENT"] == 4  # every part but the board
    assert evaluated["BOARD_FIXATION"] == 1
    assert evaluated["EXPECTED_ENGAGEMENT"] == 4  # the four support contacts
    assert evaluated["PART_COLLISION"] == math.comb(5, 2) - 4  # every pair that is not a support contact
    segments = sum(len(axis.points_mm) - 1 for axis in ref_package.scene.beam_reference.beam_axes)
    assert segments >= 1 and evaluated["BEAM_AXIS_CLEARANCE"] == segments  # every axis segment


def _seat_bore_radius(package, seat_datum: str) -> float:
    """Radius of the mount's proxy bore: the mount vertex nearest to the seat axis (the seat datum's +Z)."""

    mount = package.scene.part(MOUNT)
    seat = next(d for d in mount.datums if d.datum_key == seat_datum)
    origin, axis = seat.pose_in_part.position_mm, rotate(seat.pose_in_part.quaternion_wxyz, (0.0, 0.0, 1.0))
    radii = []
    for v in read_part_mesh(package.files[mount.geometry.asset_path]).vertices:
        d = [v[k] - origin[k] for k in range(3)]
        along = sum(d[k] * axis[k] for k in range(3))
        radii.append(math.sqrt(max(0.0, sum(c * c for c in d) - along * along)))
    return min(radii)


def test_reference_lens_seat_passes_only_because_of_its_engagement_zone(ref_package, ref_result):
    """The lens really overlaps its mount: the proxy bore is narrower than the lens, so the lens rim sits in the bore wall
    (~270 mm^3). The OD25.4 seat zone accepts that; a seat designation narrower than the bore exposes all of it."""

    assert check_status(ref_result, "EXPECTED_ENGAGEMENT") == "PASS"
    (seat,) = [m for m in ref_package.scene.mates if m.mate_ref == "mate-1"]
    assert (seat.part_a_ref, seat.part_b_ref, seat.interface_designation) == (OPTIC, MOUNT, "OD25.4-SEAT")
    lens = ref_package.scene.part(OPTIC).geometry
    radius, thickness = lens.bounds_max_mm[0], lens.bounds_max_mm[2] - lens.bounds_min_mm[2]
    bore = _seat_bore_radius(ref_package, seat.datum_b)
    assert 20.0 / 2.0 + DEFAULT_TOLERANCES.engagement_radial_allowance_mm < bore < radius  # premise: OD20 ends inside the bore
    result = validate_assembly_layout(mutate(lambda s, g: mate(s, "mate-1").update(interface_designation="OD20-SEAT")))
    assert problems(result) == [("ENGAGEMENT_EXCEEDS_INTERFACE", (OPTIC, MOUNT), "mate-1")]
    found = issues(result, "ENGAGEMENT_EXCEEDS_INTERFACE")[0]
    ring = 0.5 * 32 * math.sin(2.0 * math.pi / 32) * (radius ** 2 - bore ** 2) * thickness  # the 32-gon lens rim in the wall
    assert metric(found, "OVERLAP_VOLUME_MM3") == pytest.approx(ring, rel=0.05)
    assert metric(found, "ENGAGED_VOLUME_MM3") == 0.0  # none of it within the narrow zone
    assert 0.0 < metric(found, "PENETRATION_MM") <= thickness
    assert result.status.value == "FAIL" and result.summary.interference.value == "INTERFERENCE"


def test_reference_customer_light_source_is_excluded_explicitly(ref_package, ref_result):
    src = next(a for a in ref_package.scene.beam_reference.design_anchors if a.instance_id == "SRC")
    assert src.customer_supplied and src.position_mm[0] < ref_package.scene.breadboard.outline_min_xy_mm[0]  # off-board
    excluded = issues(ref_result, "CUSTOMER_SUPPLIED_COMPONENT")
    assert len(excluded) == 1 and excluded[0].design_instance_id == "SRC" and excluded[0].part_refs == ()
    assert excluded[0].severity.value == "INFO" and excluded[0].check.value == "BOARD_CONTAINMENT"
    assert check_status(ref_result, "BOARD_CONTAINMENT") == "PASS"


def test_reference_summary_and_identities(ref_result):
    summary = ref_result.summary
    assert summary.geometry_basis == "ASSEMBLY_PREVIEW_PART_GEOMETRY"
    assert summary.geometry_fidelity.value == "NOMINAL_SHAPE"
    assert summary.interference.value == "NO_INTERFERENCE"
    assert summary.statement.startswith("Layout and collision check of the resolved assembly")
    parts = {p.part_ref: p for p in ref_result.parts}
    assert parts[OPTIC].opensci_sku.startswith("OS-OPT-")
    assert all(parts[ref].opensci_sku.startswith("OS-MECH-") for ref in (MOUNT, POST, HOLDER, BOARD))
    assert [parts[ref].commercial_line_ref for ref in REFERENCE_PARTS] == ["B01", "B02", "B03", "B04", "B05"]
    assert set(ref_result.package.model_dump(exclude_none=True)) <= {
        "package_content_hash", "package_revision", "project", "resolution", "local_validation_hash", "commercial_resolution_id"}


def test_result_texts_state_what_was_checked(ref_result):
    text = result_file_bytes(ref_result).decode().lower()
    for phrase in ("proxy", "gross", "not_verified", "not verified", "manufacturing", "engineering review", "engineering_review",
                   "approximate", "resolver", "preview_realized", "not checked"):
        assert phrase not in text, phrase


def test_reference_result_round_trips_through_its_own_json(ref_result):
    data = result_file_bytes(ref_result)
    again = AssemblyLocalValidationV1.model_validate_json(data)
    assert again == ref_result and result_file_bytes(again) == data
    assert json.loads(data)["result_hash"] == ref_result.result_hash
    tampered = json.loads(data)
    tampered["status"] = "WARNING"
    with pytest.raises(ValueError):
        AssemblyLocalValidationV1.model_validate(tampered)


# ============================================================================================ 2. determinism


def test_every_source_form_gives_identical_bytes(tmp_path, ref_package, ref_result):
    expected = result_file_bytes(ref_result)
    files = reference_files()
    directory = tmp_path / "pkg"
    for path, data in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(data)
    sources = [FIXTURE_ZIP, str(FIXTURE_ZIP), FIXTURE_ZIP.read_bytes(), bytearray(FIXTURE_ZIP.read_bytes()), files, directory, ref_package]
    for source in sources:
        assert result_file_bytes(validate_assembly_layout(source)) == expected, type(source).__name__
    assert validate_assembly_layout(ref_package).result_hash == ref_result.result_hash


@pytest.mark.parametrize("seed", ["0", "4242"])
def test_fresh_interpreter_with_other_hash_seed_gives_same_hash(seed, ref_result):
    code = ("import sys\nfrom os_optics.assembly import result_file_bytes, validate_assembly_layout\n"
            "r = validate_assembly_layout(sys.argv[1])\nsys.stdout.write(r.result_hash + ' ' + str(len(result_file_bytes(r))))\n")
    out = subprocess.run([sys.executable, "-c", code, str(FIXTURE_ZIP)], capture_output=True, text=True, check=True, cwd=ENGINE_ROOT,
                         env={**os.environ, "PYTHONHASHSEED": seed}, timeout=120)
    assert out.stdout.split() == [ref_result.result_hash, str(len(result_file_bytes(ref_result)))]


def test_scene_list_order_does_not_change_the_verdict(ref_result):
    """Reversing parts / mates / settings / notes in scene.json changes the package hash, not one check, issue or part."""

    def shuffle(scene, _geometry):
        for key in ("parts", "mates", "settings", "notes", "dof_implementations"):
            scene[key] = list(reversed(scene[key]))

    shuffled = validate_assembly_layout(mutate(shuffle))
    assert shuffled.package.package_content_hash != ref_result.package.package_content_hash
    for key in ("status", "checks", "issues", "parts", "summary"):
        assert shuffled.payload()[key] == ref_result.payload()[key], key


def test_issue_ordering_is_the_contract_sort_key():
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT], (0.0, 0.0, -60.0))))
    assert len(result.issues) >= 6
    assert list(result.issues) == sorted(result.issues, key=AssemblyIssueV1.sort_key)
    assert [c.check for c in result.checks] == list(ASSEMBLY_CHECK_ORDER)


# ====================================================================== 3. verification before any geometry


@pytest.fixture
def no_geometry(monkeypatch):
    def boom(self):
        raise AssertionError("the checker ran on a package that did not verify")

    monkeypatch.setattr(checks_module.AssemblyChecker, "run", boom)


def _flip_byte(data: bytes, index: int) -> bytes:
    return data[:index] + bytes([data[index] ^ 0x01]) + data[index + 1:]


def _tampered_scene(files: dict[str, bytes]) -> dict[str, bytes]:
    scene = json.loads(files["scene.json"])
    part(scene, OPTIC)["pose_in_world"]["position_mm"][2] += 40.0
    return {**files, "scene.json": json_file_bytes(scene)}  # canonical form, but not re-signed


@pytest.mark.parametrize("damage", ["flip-glb-byte", "drop-glb", "tamper-scene", "extra-file", "drop-manifest"])
def test_corrupted_package_is_refused_before_any_geometry(no_geometry, damage):
    files = reference_files()
    if damage == "flip-glb-byte":
        files["geometry/b03-post.glb"] = _flip_byte(files["geometry/b03-post.glb"], len(files["geometry/b03-post.glb"]) - 9)
    elif damage == "drop-glb":
        del files["geometry/b02-optic-mount.glb"]
    elif damage == "tamper-scene":
        files = _tampered_scene(files)
    elif damage == "extra-file":
        files["geometry/b06-extra.glb"] = files["geometry/b03-post.glb"]
    else:
        del files["manifest.json"]
    with pytest.raises(AssemblyPreviewPackageError) as caught:
        validate_assembly_layout(files)
    assert caught.value.problems


def test_wrong_expected_content_hash_is_refused(ref_package):
    assert load_assembly_preview(FIXTURE_ZIP, expected_content_hash=ref_package.content_hash).content_hash == ref_package.content_hash
    with pytest.raises(AssemblyPreviewPackageError, match="package content hash"):
        load_assembly_preview(FIXTURE_ZIP, expected_content_hash="0" * 64)


def test_garbage_archive_is_refused(no_geometry):
    for junk in (b"", b"PK\x03\x04 not really a zip", FIXTURE_ZIP.read_bytes()[:-40]):
        with pytest.raises(AssemblyPreviewPackageError):
            validate_assembly_layout(junk)
    with pytest.raises(AssemblyPreviewPackageError):
        validate_assembly_layout(FIXTURE_ZIP.parent / "no-such-package.zip")


def test_a_flipped_byte_is_refused_in_every_byte_form(no_geometry, ref_package, tmp_path):
    """Bytes are verified where they enter: a file map, archive bytes and a package directory with one flipped GLB byte
    (or a scene changed without re-signing) are refused before any geometry runs."""

    files = dict(ref_package.files)
    files["geometry/b01-optic.glb"] = _flip_byte(files["geometry/b01-optic.glb"], 40)
    directory = tmp_path / "pkg"
    for path, data in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(data)
    for source in (files, package_archive_bytes(files), directory, _tampered_scene(dict(ref_package.files))):
        with pytest.raises(AssemblyPreviewPackageError):
            validate_assembly_layout(source)


def test_a_verified_package_object_is_used_as_it_is(ref_package, ref_result, monkeypatch):
    """The verifier's own output is not verified again: with the package verifier disabled, a verified package still
    checks to the same result, and its manifest still answers the expected hashes."""

    from os_optics import assembly as assembly_module

    def never(*_args, **_kwargs):
        raise AssertionError("a verified package was verified again")

    monkeypatch.setattr(assembly_module, "load_assembly_preview_package", never)
    assert result_file_bytes(validate_assembly_layout(ref_package)) == result_file_bytes(ref_result)
    assert load_assembly_preview(ref_package, expected_content_hash=ref_package.content_hash) is ref_package
    with pytest.raises(AssemblyPreviewPackageError, match="package content hash"):
        load_assembly_preview(ref_package, expected_content_hash="0" * 64)
    with pytest.raises(AssemblyPreviewPackageError, match="project hash"):
        load_assembly_preview(ref_package, expected_project_hash="0" * 64)


def test_a_package_object_is_trusted_as_constructed(ref_package):
    """The constructor is not a verification path: a package object built by hand in the caller's own process is
    checked as it states (here its scene moves the stack off the board), which is why only the loader builds one."""

    off_board = mutate(lambda s, g: translate_parts(s, STACK, (400.0, 0.0, 0.0)))
    chimera = AssemblyPreviewPackageV1(files=ref_package.files, manifest=ref_package.manifest, scene=off_board.scene)
    assert issues(validate_assembly_layout(chimera), "PART_OUTSIDE_BREADBOARD")


# ============================================================================================ 4A. off-board


def test_whole_stack_off_board_with_consistent_mate():
    def edit(scene, _geometry):
        translate_parts(scene, STACK, (400.0, 0.0, 0.0))
        resync_mate(scene, "mate-4")

    result = validate_assembly_layout(mutate(edit))
    assert result.status.value == "FAIL"
    outside = issues(result, "PART_OUTSIDE_BREADBOARD")
    assert sorted(i.part_refs[0] for i in outside) == sorted(STACK)
    assert all(metric(i, "OUTSIDE_DISTANCE_MM") > 200.0 for i in outside)
    assert check_status(result, "BOARD_CONTAINMENT") == "FAIL"
    assert [i.part_refs for i in issues(result, "FIXATION_HOLE_NOT_UNDER_PART")] == [(HOLDER,)]
    assert check_status(result, "BOARD_FIXATION") == "FAIL"
    # The mates close exactly; the holder base still stands beside the board, i.e. on nothing -> floating.
    assert not issues(result, "MATE_SEPARATED") and not issues(result, "MATE_MISALIGNED")
    assert [(i.part_refs, i.mate_ref) for i in issues(result, "SUPPORT_GEOMETRY_SEPARATED")] == [((HOLDER, BOARD), "mate-4")]
    assert non_pass(result) == {"SUPPORT_ATTACHMENT": "FAIL", "BOARD_CONTAINMENT": "FAIL", "BOARD_FIXATION": "FAIL"}
    assert result.summary.interference.value == "NO_INTERFERENCE"
    assert all(part_status(result, ref) == "FAIL" for ref in STACK)
    assert issues(result, "CUSTOMER_SUPPLIED_COMPONENT")  # SRC stays excluded, not failed


def test_whole_stack_off_board_without_adjusting_the_mate_also_separates_it():
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, STACK, (400.0, 0.0, 0.0))))
    separated = issues(result, "MATE_SEPARATED")
    assert [(i.part_refs, i.mate_ref) for i in separated] == [((HOLDER, BOARD), "mate-4")]
    assert metric(separated[0], "POSITION_ERROR_MM") == pytest.approx(400.0)
    assert len(issues(result, "PART_OUTSIDE_BREADBOARD")) == 4 and result.status.value == "FAIL"


def _far_y(package) -> dict[str, float]:
    """How far in +Y each part standing on / carried by the board reaches (world, from its decoded mesh)."""

    return {ref: _world_box(package, ref)[1][1] for ref in STACK}


def _board_depth(depth: float):
    """An edit moving the board outline's +Y edge to ``depth`` (the X extent kept)."""

    return lambda s, g: s["breadboard"].update(outline_max_xy_mm=[s["breadboard"]["outline_max_xy_mm"][0], depth])


def test_carried_part_overhang_is_a_warning_not_a_failure(ref_package):
    """The mount (carried) reaches furthest in +Y: an outline 1.75 mm short of it leaves only the mount overhanging, while
    the holder base (standing on the board) stays inside."""

    far = _far_y(ref_package)
    edge = far[MOUNT] - 1.75
    assert max(far[ref] for ref in STACK if ref != MOUNT) < edge - DEFAULT_TOLERANCES.board_outline_tolerance_mm
    result = validate_assembly_layout(mutate(_board_depth(edge)))
    assert problems(result) == [("PART_OVERHANGS_BREADBOARD", (MOUNT,), None)]
    assert metric(issues(result, "PART_OVERHANGS_BREADBOARD")[0], "OUTSIDE_DISTANCE_MM") == pytest.approx(1.75, abs=1e-5)
    assert result.status.value == "WARNING" and check_status(result, "BOARD_CONTAINMENT") == "WARNING"


def test_board_standing_part_over_the_edge_fails_while_carried_parts_only_overhang(ref_package):
    """An outline 1 mm short of the holder base: the base (on the board) is outside; the optic and mount it carries, which
    reach further, only overhang; the post (within the base's footprint) stays inside."""

    far = _far_y(ref_package)
    edge = far[HOLDER] - 1.0
    tolerance = DEFAULT_TOLERANCES.board_outline_tolerance_mm
    assert far[POST] < edge and far[OPTIC] > edge + tolerance and far[MOUNT] > edge + tolerance
    result = validate_assembly_layout(mutate(_board_depth(edge)))
    assert problems(result) == [("PART_OUTSIDE_BREADBOARD", (HOLDER,), None), ("PART_OVERHANGS_BREADBOARD", (OPTIC,), None),
                                ("PART_OVERHANGS_BREADBOARD", (MOUNT,), None)]
    assert result.status.value == "FAIL"


def test_outline_tolerance_boundary(ref_package):
    """0.4 mm beyond the outline is within board_outline_tolerance_mm (0.5); 0.6 mm is not."""

    far = _far_y(ref_package)
    assert DEFAULT_TOLERANCES.board_outline_tolerance_mm == 0.5
    assert max(far[ref] for ref in STACK if ref != MOUNT) < far[MOUNT] - 0.6 - 0.5
    within = validate_assembly_layout(mutate(_board_depth(far[MOUNT] - 0.4)))
    beyond = validate_assembly_layout(mutate(_board_depth(far[MOUNT] - 0.6)))
    assert check_status(within, "BOARD_CONTAINMENT") == "PASS"
    assert problems(beyond) == [("PART_OVERHANGS_BREADBOARD", (MOUNT,), None)]


# ========================================================================================= 4B. collisions


def test_holder_sunk_into_board_is_an_engagement_failure_of_a_planar_contact():
    def sink(depth):
        def edit(scene, _geometry):
            translate_parts(scene, STACK, (0.0, 0.0, -depth))
            resync_mate(scene, "mate-4")
        return edit

    deep = validate_assembly_layout(mutate(sink(5.0)))
    assert problems(deep) == [("ENGAGEMENT_EXCEEDS_INTERFACE", (HOLDER, BOARD), "mate-4")]
    found = issues(deep, "ENGAGEMENT_EXCEEDS_INTERFACE")[0]
    assert metric(found, "PENETRATION_MM") == pytest.approx(5.0, abs=0.05)
    assert metric(found, "OVERLAP_VOLUME_MM3") > 1000.0
    assert "ENGAGED_VOLUME_MM3" not in {m.name.value for m in found.metrics}  # STANDS_ON has no engagement zone
    assert deep.status.value == "FAIL" and deep.summary.interference.value == "INTERFERENCE"

    shallow = validate_assembly_layout(mutate(sink(0.5)))
    assert problems(shallow) == [("ENGAGEMENT_MINOR_OVERLAP", (HOLDER, BOARD), "mate-4")]
    assert shallow.status.value == "WARNING" and shallow.summary.interference.value == "NO_INTERFERENCE"


def test_lens_driven_into_the_post_holder_is_gross_interference():
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT], (0.0, 0.0, -60.0))))
    assert result.status.value == "FAIL" and result.summary.interference.value == "INTERFERENCE"
    gross = {i.part_refs: i for i in issues(result, "PART_INTERFERENCE")}
    assert (OPTIC, HOLDER) in gross and (OPTIC, POST) in gross
    lens_holder = gross[(OPTIC, HOLDER)]
    assert lens_holder.severity.value == "ERROR" and lens_holder.check.value == "PART_COLLISION"
    assert 100.0 < metric(lens_holder, "OVERLAP_VOLUME_MM3") < math.pi * 12.7 ** 2 * 6.8  # at most the whole lens
    assert 1.0 <= metric(lens_holder, "PENETRATION_MM") <= 6.8 + 1e-6  # at most the lens thickness
    # moving a carried part breaks its own mate as well
    assert [(i.part_refs, i.mate_ref) for i in issues(result, "MATE_SEPARATED")] == [((MOUNT, POST), "mate-2")]
    assert part_status(result, OPTIC) == "FAIL" and part_status(result, HOLDER) == "FAIL"


def test_extra_block_on_the_board_colliding_with_the_holder_is_isolated():
    """A new part standing on the board with an exactly closing mate: the only problem is the collision itself."""

    def edit(scene, geometry):
        holder_pose = pose_of(part(scene, HOLDER)["pose_in_world"])
        block = json.loads(json.dumps(part(scene, HOLDER)))
        block.update(part_ref="b06-block", commercial_line_ref="B06", assembly_order=6,
                     identity={"opensci_sku": "OS-MECH-TEST-BLOCK", "identity_kind": "COMMERCIAL_SKU", "display_name": "Test block"},
                     support={"parent_ref": BOARD, "mate_ref": "mate-5", "relation": "STANDS_ON"},
                     datums=[{"datum_key": "bottom", "pose_in_part": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}],
                     # 20 mm from the holder's post axis, in the holder's frame: into its post tube and over its fork base
                     pose_in_world=pose_dict(apply(holder_pose, (20.0, 0.0, 0.0)), holder_pose[1]))
        block["geometry"]["asset_path"] = "geometry/b06-block.glb"
        scene["parts"].append(block)
        set_geometry(scene, geometry, "b06-block", box((-10.0, -5.0, 0.0), (10.0, 5.0, 60.0)))
        scene["mates"].append({"mate_ref": "mate-5", "part_a_ref": "b06-block", "datum_a": "bottom", "part_b_ref": BOARD,
                               "datum_b": "board_top", "b_in_a": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}})
        resync_mate(scene, "mate-5")

    result = validate_assembly_layout(mutate(edit))
    gross = issues(result, "PART_INTERFERENCE")
    assert gross and all("b06-block" in i.part_refs for i in gross)
    assert {i.part_refs for i in gross} <= {("b04-post-holder-base", "b06-block"), ("b03-post", "b06-block")}
    assert non_pass(result) == {"PART_COLLISION": "FAIL"}
    assert part_status(result, OPTIC) == "PASS" and part_status(result, "b06-block") == "FAIL"


# ===================================================================================== 4C. expected engagement


def _holder_and_post(*, post_radius=6.35, post_z=28.0, designation="OD12.7-POST", closing_mate=True, stud=None, mount=False):
    """Solid post holder (bore modelled SOLID, top at 75) on the board, a 75 mm post inserted from 75 down to ``post_z``;
    optionally an M4 stud of length ``stud`` on the post and a solid mount MOUNTED_ON the post top."""

    holder = SynthPart("p1-holder", box((-15.0, -15.0, 0.0), (15.0, 15.0, 75.0)), (100.0, 100.0, 0.0), category="POST_HOLDER",
                       parent=BOARD_REF, mate_ref="m1", relation="STANDS_ON",
                       datums={"origin": ((0.0, 0.0, 0.0), IDENTITY_Q), "bore_top": ((0.0, 0.0, 75.0), IDENTITY_Q)})
    post_mesh = cylinder(post_radius, 0.0, 75.0)
    if stud:
        post_mesh = combine(post_mesh, cylinder(2.0, 75.0, 75.0 + stud))
    post = SynthPart("p2-post", post_mesh, (100.0, 100.0, post_z), category="POST", parent="p1-holder", mate_ref="m2",
                     relation="INSERTED_INTO", datums={"bottom": ((0.0, 0.0, 0.0), IDENTITY_Q), "top": ((0.0, 0.0, 75.0), IDENTITY_Q)})
    parts = [board(), holder, post]
    mates = [SynthMate("m1", "p1-holder", "origin", BOARD_REF, "top"),
             SynthMate("m2", "p2-post", "bottom", "p1-holder", "bore_top", designation=designation,
                       b_in_a=None if closing_mate else ((0.0, 0.0, 47.0), IDENTITY_Q))]
    if mount:
        parts.append(SynthPart("p3-mount", box((-10.0, -10.0, 0.0), (10.0, 10.0, 30.0)), (100.0, 100.0, post_z + 75.0),
                               category="OPTIC_MOUNT", parent="p2-post", mate_ref="m3", relation="MOUNTED_ON",
                               datums={"support": ((0.0, 0.0, 0.0), IDENTITY_Q)}))
        mates.append(SynthMate("m3", "p3-mount", "support", "p2-post", "top", designation="M4x0.7"))
    return synthetic_package(parts, mates)


def _skip_one_column(monkeypatch, pair: tuple[str, str]) -> None:
    """Make the winding of one sample column of ``pair``'s overlap measure not close (the first column in which a part
    is crossed), as an unclosed winding does: that column is left out of every sum."""

    measure, solid = checks_module.AssemblyChecker._measure, overlap_module._solid

    def patched_measure(self, a, b, frame, zone=None):  # noqa: ANN001, ANN202
        if {a, b} != set(pair):
            return measure(self, a, b, frame, zone)
        calls = {"n": 0}

        def one_bad_column(tris, x, y):  # noqa: ANN001, ANN202
            intervals, ok = solid(tris, x, y)
            calls["n"] += 1
            return intervals, ok and calls["n"] != 1

        monkeypatch.setattr(overlap_module, "_solid", one_bad_column)
        try:
            return measure(self, a, b, frame, zone)
        finally:
            monkeypatch.setattr(overlap_module, "_solid", solid)

    monkeypatch.setattr(checks_module.AssemblyChecker, "_measure", patched_measure)


def test_a_collision_proven_by_the_evaluated_columns_stands_with_a_skipped_column(monkeypatch):
    """A skipped column can only add overlap: a deep core already at the collision volume proves the collision, which
    is reported with its metrics and the column counts (the volume within one column's share of the full measure)."""

    full = issues(validate_assembly_layout(_holder_and_post(post_z=-20.0)), "PART_INTERFERENCE")
    assert [i.part_refs for i in full] == [(BOARD_REF, "p2-post")]
    _skip_one_column(monkeypatch, (BOARD_REF, "p2-post"))
    result = validate_assembly_layout(_holder_and_post(post_z=-20.0))
    (found,) = issues(result, "PART_INTERFERENCE")
    assert not issues(result, "PAIR_NOT_EVALUATED") and result.status.value == "FAIL"
    assert metric(found, "COLUMNS_SKIPPED") == 1.0 and metric(found, "COLUMNS_CAST") > 1000.0
    assert not any(m.name.value.startswith("COLUMNS_") for m in full[0].metrics)  # a complete measure states no counts
    volume, reference = metric(found, "OVERLAP_VOLUME_MM3"), metric(full[0], "OVERLAP_VOLUME_MM3")
    assert 0.95 * reference <= volume <= reference
    assert metric(found, "DEEP_OVERLAP_VOLUME_MM3") >= DEFAULT_TOLERANCES.collision_volume_mm3


def test_a_clear_pair_with_a_skipped_column_is_not_evaluated_but_keeps_its_measure(monkeypatch, ref_result):
    """No clear result can be claimed from a partial measure: the board contact of the reference holder stays
    ENGAGEMENT_NOT_EVALUATED (UNKNOWN), now with the measured overlap and the column counts."""

    pair = ("b04-post-holder-base", "b05-breadboard")
    _skip_one_column(monkeypatch, pair)
    result = validate_assembly_layout(reference_package())
    (found,) = issues(result, "ENGAGEMENT_NOT_EVALUATED")
    assert found.part_refs == pair and result.status.value == "UNKNOWN"
    names = [m.name.value for m in found.metrics]
    assert names == ["COLUMNS_CAST", "COLUMNS_SKIPPED", "DEEP_OVERLAP_VOLUME_MM3", "OVERLAP_VOLUME_MM3", "PENETRATION_MM"]
    assert metric(found, "COLUMNS_SKIPPED") == 1.0 and metric(found, "COLUMNS_CAST") > 1000.0
    assert metric(found, "PENETRATION_MM") < DEFAULT_TOLERANCES.overlap_noise_mm  # contact on the evaluated columns
    # every other check of the reference is unchanged
    assert [i.code.value for i in result.issues if i.code.value != "ENGAGEMENT_NOT_EVALUATED"] == [
        i.code.value for i in ref_result.issues]


def test_designation_families_are_one_table():
    from os_optics.assembly.interfaces import DESIGNATION_FAMILIES, nominal_diameter_mm

    assert len(DESIGNATION_FAMILIES) == 3
    assert nominal_diameter_mm("M6x1.0-GRID25.0") == 6.0 and nominal_diameter_mm("OD12.7-POST") == 12.7
    assert nominal_diameter_mm("SM1") == pytest.approx(1.035 * 25.4) and nominal_diameter_mm("UNC-QX") is None
    assert nominal_diameter_mm(None) is None


def test_post_in_a_solid_modelled_bore_is_accepted_engagement():
    result = validate_assembly_layout(_holder_and_post())
    assert non_pass(result) == {} and issue_keys(result) == []
    assert check_status(result, "EXPECTED_ENGAGEMENT") == "PASS"
    # ...and the same overlap with the zone switched off is gross: the pass is the zone's doing, not a skipped pair
    no_zone = validate_assembly_layout(_holder_and_post(designation="QX9"))
    found = issues(no_zone, "ENGAGEMENT_NOT_BOUNDED")
    assert [(i.part_refs, i.mate_ref) for i in found] == [(("p1-holder", "p2-post"), "m2")]
    analytic = 47.0 * 0.5 * 32 * 6.35 ** 2 * math.sin(2.0 * math.pi / 32)  # the 32-gon post over the inserted 47 mm
    assert metric(found[0], "OVERLAP_VOLUME_MM3") == pytest.approx(analytic, rel=0.02)
    assert check_status(no_zone, "EXPECTED_ENGAGEMENT") == "UNKNOWN" and no_zone.status.value == "UNKNOWN"
    assert no_zone.summary.interference.value == "NOT_DETERMINED"


def test_unknown_designation_on_the_reference_lens_seat_is_not_bounded(ref_package, ref_result):
    result = validate_assembly_layout(mutate(lambda s, g: mate(s, "mate-1").update(interface_designation="UNC-QX")))
    assert problems(result) == [("ENGAGEMENT_NOT_BOUNDED", (OPTIC, MOUNT), "mate-1")]
    assert result.status.value == "UNKNOWN" and result.summary.interference.value == "NOT_DETERMINED"
    assert part_status(result, OPTIC) == "UNKNOWN" and part_status(result, MOUNT) == "UNKNOWN"
    # a face-to-face contact leaves nothing to bound: the post (no stud modelled) meets the mount's underside flush
    post = ref_package.scene.part(POST)
    assert post.geometry.bounds_max_mm[2] == next(d.pose_in_part.position_mm[2] for d in post.datums if d.datum_key == "top")
    flush = validate_assembly_layout(mutate(lambda s, g: mate(s, "mate-2").update(interface_designation="UNC-QX")))
    for key in ("status", "checks", "issues", "parts", "summary"):
        assert flush.payload()[key] == ref_result.payload()[key], key


def test_post_too_fat_for_its_outer_diameter_designation_fails():
    result = validate_assembly_layout(_holder_and_post(post_radius=8.0))
    assert issue_keys(result) == [
        ("ENGAGEMENT_EXCEEDS_INTERFACE", ("p1-holder", "p2-post"), "m2")]
    found = issues(result, "ENGAGEMENT_EXCEEDS_INTERFACE")[0]
    assert metric(found, "ENGAGED_VOLUME_MM3") > metric(found, "OVERLAP_VOLUME_MM3") > 1000.0
    assert 1.0 < metric(found, "PENETRATION_MM") < 16.0
    assert result.summary.interference.value == "INTERFERENCE"
    # within the radial allowance (6.35 + 0.5): accepted
    assert check_status(validate_assembly_layout(_holder_and_post(post_radius=6.8)), "EXPECTED_ENGAGEMENT") == "PASS"


def test_post_pushed_through_the_holder_hits_the_board():
    """INSERTED_INTO's zone follows the mate datums, so a deeper (consistently mated) insertion is bounded by the
    board underneath: the post bottom at z = -20 interpenetrates the board, an unexpected pair."""

    result = validate_assembly_layout(_holder_and_post(post_z=-20.0))
    assert issue_keys(result) == [
        ("PART_INTERFERENCE", (BOARD_REF, "p2-post"), None)]
    assert check_status(result, "EXPECTED_ENGAGEMENT") == "PASS"
    # inserted 1.5 mm deeper than the mate declares: a misaligned mate, not an engagement failure
    shifted = validate_assembly_layout(_holder_and_post(post_z=26.5, closing_mate=False))
    assert issue_keys(shifted) == [("MATE_MISALIGNED", ("p1-holder", "p2-post"), "m2")]
    assert metric(issues(shifted, "MATE_MISALIGNED")[0], "POSITION_ERROR_MM") == pytest.approx(1.5)


@pytest.mark.parametrize("stud, verdict", [(4.0, "PASS"), (12.0, "PASS"), (14.0, "FAIL"), (20.0, "FAIL")])
def test_m4_stud_may_reach_three_diameters(stud, verdict):
    """Zone reach 3 x 4 mm + 0.5 mm: a 12 mm stud is engagement, 14 / 20 mm poke out of the zone into the mount."""

    result = validate_assembly_layout(_holder_and_post(stud=stud, mount=True))
    assert check_status(result, "EXPECTED_ENGAGEMENT") == verdict
    if verdict == "FAIL":
        found = issues(result, "ENGAGEMENT_EXCEEDS_INTERFACE")
        assert [(i.part_refs, i.mate_ref) for i in found] == [(("p2-post", "p3-mount"), "m3")]
        residual = (stud - 12.5) * 0.5 * 32 * 2.0 ** 2 * math.sin(2.0 * math.pi / 32)
        assert metric(found[0], "OVERLAP_VOLUME_MM3") == pytest.approx(residual, rel=0.05)
        assert metric(found[0], "ENGAGED_VOLUME_MM3") == pytest.approx(12.5 / (stud - 12.5) * residual, rel=0.05)
    else:
        assert non_pass(result) == {}


def test_optic_in_a_solid_modelled_seat_is_accepted_and_obstructs_nothing_optical():
    """Mount body modelled solid (no bore): the lens is SEATED_IN inside the SM1 zone -> engagement passes; the solid
    mount now sits on the beam axis -> BEAM_AXIS_OBSTRUCTED, a WARNING only."""

    solid = mutate(lambda s, g: set_geometry(s, g, MOUNT, box((-5.1, -15.25, 0.0), (5.1, 15.25, 37.25))))
    result = validate_assembly_layout(solid)
    assert problems(result) == [("BEAM_AXIS_OBSTRUCTED", (MOUNT,), None)]
    assert metric(issues(result, "BEAM_AXIS_OBSTRUCTED")[0], "AXIS_LENGTH_IN_PART_MM") == pytest.approx(10.2, abs=1e-4)
    assert result.status.value == "WARNING" and check_status(result, "BEAM_AXIS_CLEARANCE") == "WARNING"
    assert check_status(result, "EXPECTED_ENGAGEMENT") == "PASS"


# ======================================================================================== 4D. broken support


def test_post_lifted_out_of_its_holder_floats_and_separates(ref_package):
    """Lifted by its insertion depth + 3 mm, the post bottom hangs 3 mm above the holder's bore top."""

    depth = float(next(s for s in ref_package.scene.settings if s.part_ref == POST and s.kind.value == "INSERTION_DEPTH").value)
    lift = depth + 3.0
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT, POST], (0.0, 0.0, lift))))
    separated = issues(result, "MATE_SEPARATED")
    assert [(i.part_refs, i.mate_ref) for i in separated] == [((POST, HOLDER), "mate-3")]
    assert metric(separated[0], "POSITION_ERROR_MM") == pytest.approx(lift)
    floating = issues(result, "SUPPORT_GEOMETRY_SEPARATED")
    assert [(i.part_refs, i.mate_ref) for i in floating] == [((POST, HOLDER), "mate-3")]
    assert metric(floating[0], "GAP_MM") == pytest.approx(3.0, abs=1e-4)  # post bottom vs holder top
    assert check_status(result, "SUPPORT_ATTACHMENT") == "FAIL" and result.status.value == "FAIL"
    # lifted by less than its insertion depth it is still held by the bore: the mate separates, the geometry does not
    held = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT, POST], (0.0, 0.0, depth - 3.0))))
    assert [(i.part_refs, i.mate_ref) for i in issues(held, "MATE_SEPARATED")] == [((POST, HOLDER), "mate-3")]
    assert not issues(held, "SUPPORT_GEOMETRY_SEPARATED")


def test_mount_lifted_50mm_floats_and_separates():
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT], (0.0, 0.0, 50.0))))
    assert problems(result) == [("MATE_SEPARATED", (MOUNT, POST), "mate-2"), ("SUPPORT_GEOMETRY_SEPARATED", (MOUNT, POST), "mate-2")]
    assert result.status.value == "FAIL"


def test_small_lift_is_only_a_misaligned_mate():
    result = validate_assembly_layout(mutate(lambda s, g: translate_parts(s, [OPTIC, MOUNT, POST], (0.0, 0.0, 0.2))))
    assert problems(result) == [("MATE_MISALIGNED", (POST, HOLDER), "mate-3")]
    assert metric(issues(result, "MATE_MISALIGNED")[0], "POSITION_ERROR_MM") == pytest.approx(0.2)
    assert result.status.value == "WARNING"


def test_small_mate_rotation_is_misaligned_and_large_rotation_separates():
    def twist(degrees):
        def edit(scene, _geometry):
            mate(scene, "mate-1")["b_in_a"]["quaternion_wxyz"] = list(axis_angle((1.0, 0.0, 0.0), degrees))
        return edit

    small = validate_assembly_layout(mutate(twist(1.0)))
    assert problems(small) == [("MATE_MISALIGNED", (OPTIC, MOUNT), "mate-1")]
    assert metric(issues(small, "MATE_MISALIGNED")[0], "ANGLE_ERROR_DEG") == pytest.approx(1.0, abs=1e-6)
    large = validate_assembly_layout(mutate(twist(5.0)))
    assert problems(large) == [("MATE_SEPARATED", (OPTIC, MOUNT), "mate-1")]


# ================================================================================ 4E. missing / bad geometry


@functools.cache
def _holder_box():
    """A closed box over the holder's scene bounds (broken on purpose by the tests below)."""

    bounds = reference_package().scene.part(HOLDER).geometry
    return box(tuple(bounds.bounds_min_mm), tuple(bounds.bounds_max_mm))


def _holder_unplaceable(result, code):
    assert [k for k in problems(result) if k[0].startswith("GEOMETRY_")] == [(code, (HOLDER,), None)]
    expected_not_evaluated = [
        ("ATTACHMENT_NOT_EVALUATED", (POST, HOLDER), "mate-3"), ("ATTACHMENT_NOT_EVALUATED", (HOLDER, BOARD), "mate-4"),
        ("CONTAINMENT_NOT_EVALUATED", (HOLDER,), None), ("FIXATION_NOT_EVALUATED", (HOLDER,), None),
        ("ENGAGEMENT_NOT_EVALUATED", (POST, HOLDER), "mate-3"), ("ENGAGEMENT_NOT_EVALUATED", (HOLDER, BOARD), "mate-4"),
        ("PAIR_NOT_EVALUATED", (OPTIC, HOLDER), None), ("PAIR_NOT_EVALUATED", (MOUNT, HOLDER), None),
        ("BEAM_AXIS_NOT_EVALUATED", (HOLDER,), None)]
    assert [k for k in problems(result) if not k[0].startswith("GEOMETRY_")] == expected_not_evaluated
    assert non_pass(result) == {**{c: "UNKNOWN" for c in ("PART_GEOMETRY", *GEOMETRIC_CHECKS)}}
    assert result.status.value == "UNKNOWN" and result.summary.interference.value == "NOT_DETERMINED"
    assert part_status(result, HOLDER) == "UNKNOWN" and part_status(result, OPTIC) == "UNKNOWN"


def _holder_not_solid(result, code):
    """An open or inward-facing holder mesh: its GEOMETRY issue (UNKNOWN) stays, the volume-based checks of it (its two
    engagements) are not evaluated, and every check that needs only its vertices is measured as for a closed holder."""

    closed = validate_assembly_layout(mutate(lambda s, g: set_geometry(s, g, HOLDER, _holder_box())))
    assert closed.status.value == "PASS"
    assert problems(result) == [(code, (HOLDER,), None), ("ENGAGEMENT_NOT_EVALUATED", (POST, HOLDER), "mate-3"),
                                ("ENGAGEMENT_NOT_EVALUATED", (HOLDER, BOARD), "mate-4")]
    assert non_pass(result) == {"PART_GEOMETRY": "UNKNOWN", "EXPECTED_ENGAGEMENT": "UNKNOWN"}
    evaluated = {c.check.value: c.evaluated for c in result.checks}
    reference = {c.check.value: c.evaluated for c in closed.checks}
    for check in ("SUPPORT_ATTACHMENT", "BOARD_CONTAINMENT", "BOARD_FIXATION", "PART_COLLISION", "BEAM_AXIS_CLEARANCE"):
        assert evaluated[check] == reference[check], check
    assert evaluated["PART_GEOMETRY"] == reference["PART_GEOMETRY"] - 1 and evaluated["EXPECTED_ENGAGEMENT"] == reference["EXPECTED_ENGAGEMENT"] - 2
    assert result.status.value == "UNKNOWN" and result.summary.interference.value == "NOT_DETERMINED"
    assert part_status(result, HOLDER) == "UNKNOWN" and part_status(result, OPTIC) == "PASS"


def test_open_mesh_is_not_closed_and_only_its_volume_checks_are_unknown():
    result = validate_assembly_layout(mutate(lambda s, g: set_geometry(s, g, HOLDER, without_last_triangles(_holder_box(), 2))))
    _holder_not_solid(result, "GEOMETRY_NOT_CLOSED")


def test_inward_facing_mesh_is_degenerate():
    result = validate_assembly_layout(mutate(lambda s, g: set_geometry(s, g, HOLDER, inverted(_holder_box()))))
    _holder_not_solid(result, "GEOMETRY_DEGENERATE")


@pytest.mark.parametrize("mesh", ["open", "inward"])
def test_a_mesh_that_is_not_solid_is_measured_by_its_vertices(mesh):
    """Metamorphic: the holder moved off the board is outside by the same distance whether its mesh is closed or not
    (containment uses vertices only); a pair whose boxes meet it stays not evaluated."""

    shape = (lambda: without_last_triangles(_holder_box(), 2)) if mesh == "open" else (lambda: inverted(_holder_box()))

    def moved(make):
        def edit(scene, geometry):
            set_geometry(scene, geometry, HOLDER, make())
            translate_parts(scene, [HOLDER], (400.0, 0.0, 0.0))
        return validate_assembly_layout(mutate(edit))

    closed, not_solid = moved(_holder_box), moved(shape)
    (a,) = [i for i in issues(closed, "PART_OUTSIDE_BREADBOARD") if i.part_refs == (HOLDER,)]
    (b,) = [i for i in issues(not_solid, "PART_OUTSIDE_BREADBOARD") if i.part_refs == (HOLDER,)]
    assert metric(a, "OUTSIDE_DISTANCE_MM") == metric(b, "OUTSIDE_DISTANCE_MM") > 300.0
    assert not issues(not_solid, "CONTAINMENT_NOT_EVALUATED") and not issues(not_solid, "FIXATION_NOT_EVALUATED")
    assert [i.part_refs for i in issues(not_solid, "FIXATION_HOLE_NOT_UNDER_PART")] == [
        i.part_refs for i in issues(closed, "FIXATION_HOLE_NOT_UNDER_PART")]

    # a part whose box meets the not-solid holder: the pair needs a volume, so it is not evaluated
    def overlapping(scene, geometry):
        set_geometry(scene, geometry, HOLDER, shape())
        holder_at = part(scene, HOLDER)["pose_in_world"]["position_mm"]
        optic_at = part(scene, OPTIC)["pose_in_world"]["position_mm"]
        translate_parts(scene, [OPTIC], tuple(h - o for h, o in zip(holder_at, optic_at)))

    crossing = validate_assembly_layout(mutate(overlapping))
    assert tuple(sorted((OPTIC, HOLDER))) in [i.part_refs for i in issues(crossing, "PAIR_NOT_EVALUATED")]


def test_the_drawing_still_needs_closed_shells():
    from assembly_helpers import glb_bytes

    from os_optics.assembly.mesh import MeshError, MeshErrorCode, read_part_mesh as read

    data = glb_bytes(HOLDER, without_last_triangles(_holder_box(), 2))
    with pytest.raises(MeshError) as caught:
        read(data)
    assert caught.value.code is MeshErrorCode.NOT_CLOSED
    mesh = read(data, require_solid=False)
    assert mesh.defect is MeshErrorCode.NOT_CLOSED and mesh.volume_mm3 is None and not mesh.solid
    assert read(glb_bytes(HOLDER, _holder_box())).solid


def test_unreadable_mesh_indices_are_unreadable():
    def edit(scene, geometry):
        set_geometry(scene, geometry, HOLDER, _holder_box())
        data = bytearray(geometry["geometry/b04-post-holder-base.glb"])
        data[-4:] = (1000).to_bytes(4, "little")  # last index -> far beyond the 8 vertices (JSON untouched)
        geometry["geometry/b04-post-holder-base.glb"] = bytes(data)

    _holder_unplaceable(validate_assembly_layout(mutate(edit)), "GEOMETRY_UNREADABLE")


def test_open_geometry_with_a_real_failure_elsewhere_is_fail():
    def edit(scene, geometry):
        set_geometry(scene, geometry, HOLDER, without_last_triangles(_holder_box(), 2))
        translate_parts(scene, [OPTIC, MOUNT], (0.0, 0.0, 50.0))

    result = validate_assembly_layout(mutate(edit))
    assert result.status.value == "FAIL" and check_status(result, "PART_GEOMETRY") == "UNKNOWN"


def test_non_unit_part_quaternion_fails_transforms_and_unplaces_the_part():
    result = validate_assembly_layout(mutate(lambda s, g: part(s, MOUNT)["pose_in_world"].update(quaternion_wxyz=[2.0, 0.0, 0.0, 0.0])))
    bad = issues(result, "NON_UNIT_QUATERNION")
    assert [(i.part_refs, i.mate_ref) for i in bad] == [((MOUNT,), None)] and metric(bad[0], "QUATERNION_NORM") == 2.0
    assert check_status(result, "SCENE_TRANSFORMS") == "FAIL" and result.status.value == "FAIL"
    assert [k for k in problems(result) if k[0] != "NON_UNIT_QUATERNION"] == [
        ("ATTACHMENT_NOT_EVALUATED", (OPTIC, MOUNT), "mate-1"), ("ATTACHMENT_NOT_EVALUATED", (MOUNT, POST), "mate-2"),
        ("CONTAINMENT_NOT_EVALUATED", (MOUNT,), None),
        ("ENGAGEMENT_NOT_EVALUATED", (OPTIC, MOUNT), "mate-1"), ("ENGAGEMENT_NOT_EVALUATED", (MOUNT, POST), "mate-2"),
        ("PAIR_NOT_EVALUATED", (MOUNT, HOLDER), None), ("PAIR_NOT_EVALUATED", (MOUNT, BOARD), None),
        ("BEAM_AXIS_NOT_EVALUATED", (MOUNT,), None)]
    for check in ("SUPPORT_ATTACHMENT", "BOARD_CONTAINMENT", "EXPECTED_ENGAGEMENT", "PART_COLLISION", "BEAM_AXIS_CLEARANCE"):
        assert check_status(result, check) == "UNKNOWN", check
    assert result.summary.interference.value == "NOT_DETERMINED"


def test_non_unit_mate_and_datum_quaternions():
    mate_bad = validate_assembly_layout(mutate(lambda s, g: mate(s, "mate-3")["b_in_a"].update(quaternion_wxyz=[0.0, 0.0, 0.0, 0.5])))
    assert ("NON_UNIT_QUATERNION", (POST, HOLDER), "mate-3") in problems(mate_bad)
    assert ("ATTACHMENT_NOT_EVALUATED", (POST, HOLDER), "mate-3") in problems(mate_bad)

    def datum(scene, _geometry):
        part(scene, POST)["datums"][1]["pose_in_part"]["quaternion_wxyz"] = [1.0, 0.0, 0.0, 1.0]  # 'top', used by mate-2

    datum_bad = validate_assembly_layout(mutate(datum))
    assert ("NON_UNIT_QUATERNION", (POST,), None) in problems(datum_bad)
    assert ("ATTACHMENT_NOT_EVALUATED", (MOUNT, POST), "mate-2") in problems(datum_bad)
    assert ("ENGAGEMENT_NOT_EVALUATED", (MOUNT, POST), "mate-2") in problems(datum_bad)
    assert not any(k[0] == "CONTAINMENT_NOT_EVALUATED" for k in problems(datum_bad))  # the post itself stays placed
    # within quaternion_norm_tolerance (1e-6): accepted as unit
    near = validate_assembly_layout(mutate(lambda s, g: part(s, MOUNT)["pose_in_world"].update(quaternion_wxyz=[1.0 + 5e-7, 0.0, 0.0, 0.0])))
    assert check_status(near, "SCENE_TRANSFORMS") == "PASS"


def test_triangle_budget_is_a_work_limit_and_unknown(ref_package, ref_result):
    """The work limit reached: the model covers the assembly, the check was not run (UNKNOWN, not out of scope); the
    limit is a caller setting, recorded in the result, and a larger one runs the checks."""

    from os_optics.assembly import tolerances_with

    tolerances = tolerances_with({"max_total_triangles": 100})
    result = validate_assembly_layout(ref_package, tolerances=tolerances)
    budget = issues(result, "GEOMETRY_BUDGET_EXCEEDED")
    total = sum(p.geometry.triangle_count for p in ref_package.scene.parts)
    assert total > 100 and len(budget) == 1 and metric(budget[0], "TRIANGLE_COUNT") == float(total)
    assert check_status(result, "PART_GEOMETRY") == "UNKNOWN" and result.status.value == "UNKNOWN"
    for check in GEOMETRIC_CHECKS:
        assert check_status(result, check) == "UNKNOWN", check
    assert not any(c.status.value == "PASS" for c in result.checks if c.check.value in GEOMETRIC_CHECKS)
    assert result.summary.interference.value == "NOT_DETERMINED" and result.tolerances.max_total_triangles == 100
    assert {c.check.value: c.evaluated for c in result.checks}["PART_GEOMETRY"] == 0
    raised = validate_assembly_layout(ref_package, tolerances=tolerances_with({"max_total_triangles": 1_000_000}))
    assert raised.tolerances.max_total_triangles == 1_000_000
    for key in ("status", "checks", "issues", "parts", "summary"):
        assert raised.payload()[key] == ref_result.payload()[key], key


def test_caller_tolerances_are_validated_by_the_model():
    from os_optics.assembly import TOLERANCE_NAMES, tolerance_argument, tolerances_with

    assert tolerances_with(None) == DEFAULT_TOLERANCES and tolerances_with({}) == DEFAULT_TOLERANCES
    assert set(TOLERANCE_NAMES) == set(type(DEFAULT_TOLERANCES).model_fields)
    assert tolerance_argument("max_total_triangles=250000") == ("max_total_triangles", 250_000)
    assert tolerance_argument("overlap_noise_mm=0.2") == ("overlap_noise_mm", 0.2)
    for bad in ({"no_such": 1.0}, {"overlap_noise_mm": 5.0}, {"max_total_triangles": 0}, {"buried_fraction": True},
                {"overlap_noise_mm": "0.2"}):
        with pytest.raises(ValueError):
            tolerances_with(bad)
    for text in ("x=1", "max_total_triangles=1.5", "overlap_noise_mm"):
        with pytest.raises(ValueError):
            tolerance_argument(text)


# =========================================================================================== 4F. rotations


def _key_metrics(result, checks):
    return [(i.code.value, i.part_refs, i.mate_ref, {m.name.value: m.value for m in i.metrics})
            for i in result.issues if i.check.value in checks]


def _assert_same_measurements(a, b, checks):
    ka, kb = _key_metrics(a, checks), _key_metrics(b, checks)
    assert [k[:3] for k in ka] == [k[:3] for k in kb]
    for (_, _, _, ma), (_, _, _, mb) in zip(ka, kb):
        assert ma.keys() == mb.keys()
        for name in ma:
            # the deep-core volume counts lattice points, which move against the geometry under a rigid motion
            rel = 1e-2 if name == "DEEP_OVERLAP_VOLUME_MM3" else 1e-4
            assert mb[name] == pytest.approx(ma[name], rel=rel, abs=1e-4), name


def _quarter_turn_about_the_board_centre(scene, _geometry):
    """Turn everything placed in the world by +90 deg about the board centre's vertical, and the board's world-XY record
    (outline and hole grid) with it: the turned outline is its own AABB; the grid's X now counts the old Y holes
    backwards. Every fixation hole is renumbered to the grid hole it is carried to."""

    board = scene["breadboard"]
    (x0, y0), (x1, y1) = board["outline_min_xy_mm"], board["outline_max_xy_mm"]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    turn = rotation_about((cx, cy, 0.0), axis_angle((0.0, 0.0, 1.0), 90.0))
    move_world(scene, turn)
    grid = board["hole_grid"]
    (ox, oy), (px, py), (nx, ny) = grid["origin_xy_mm"], grid["pitch_mm"], grid["count"]
    board["outline_min_xy_mm"] = [cx - (y1 - cy), cy + (x0 - cx)]
    board["outline_max_xy_mm"] = [cx - (y0 - cy), cy + (x1 - cx)]
    # the new hole (0, 0) is the old hole (0, ny - 1), turned: (x, y) -> (cx - (y - cy), cy + (x - cx))
    grid.update(origin_xy_mm=[cx - (oy + (ny - 1) * py - cy), cy + (ox - cx)], pitch_mm=[py, px], count=[ny, nx])
    for fixation in scene["fixations"]:
        fixation["board_holes"] = [[ny - 1 - j, i] for i, j in fixation["board_holes"]]


def test_whole_reference_rotated_90deg_about_the_board_centre_is_unchanged(ref_package, ref_result):
    """A rigid quarter turn of the whole assembly, breadboard included, about the board centre changes no check, issue,
    part status or measurement: the fixation hole under the holder is still the hole the holder stands over."""

    turned = mutate(_quarter_turn_about_the_board_centre)
    ((i, j),) = ref_package.scene.fixations[0].board_holes
    grid = ref_package.scene.breadboard.hole_grid
    before = (grid.origin_xy_mm[0] + i * grid.pitch_mm[0], grid.origin_xy_mm[1] + j * grid.pitch_mm[1])
    new_grid = turned.scene.breadboard.hole_grid
    ((ti, tj),) = turned.scene.fixations[0].board_holes
    after = (new_grid.origin_xy_mm[0] + ti * new_grid.pitch_mm[0], new_grid.origin_xy_mm[1] + tj * new_grid.pitch_mm[1])
    board = ref_package.scene.breadboard
    centre = ((board.outline_min_xy_mm[0] + board.outline_max_xy_mm[0]) / 2.0, (board.outline_min_xy_mm[1] + board.outline_max_xy_mm[1]) / 2.0)
    expected = apply(rotation_about((*centre, 0.0), axis_angle((0.0, 0.0, 1.0), 90.0)), (*before, 0.0))
    assert after == pytest.approx(expected[:2], abs=1e-9)  # the renumbered hole is where the turn carried the old one
    rotated = validate_assembly_layout(turned)
    for key in ("status", "checks", "issues", "parts", "summary"):
        assert rotated.payload()[key] == ref_result.payload()[key], key


def _sink(scene, _geometry):
    translate_parts(scene, STACK, (0.0, 0.0, -5.0))
    resync_mate(scene, "mate-4")


def _lens_down(scene, _geometry):
    translate_parts(scene, [OPTIC, MOUNT], (0.0, 0.0, -60.0))


ARBITRARY = rotation_about((150.0, 150.0, 60.0), normalized((0.9, 0.3, -0.2, 0.25)))


@pytest.mark.parametrize("edit", [_sink, _lens_down], ids=["holder-sunk", "lens-into-holder"])
def test_rigid_motion_about_an_arbitrary_axis_keeps_contact_measurements(edit):
    """Collision / engagement / attachment are measured in part and datum frames: a rigid motion of the whole
    world (board included, about a non-axis quaternion) must not change their statuses or metrics."""

    plain = validate_assembly_layout(mutate(edit))

    def moved(scene, geometry):
        edit(scene, geometry)
        move_world(scene, ARBITRARY)

    turned = validate_assembly_layout(mutate(moved))
    checks = ("SUPPORT_ATTACHMENT", "EXPECTED_ENGAGEMENT", "PART_COLLISION")
    for check in (*checks, "BEAM_AXIS_CLEARANCE"):
        assert check_status(turned, check) == check_status(plain, check), check
    _assert_same_measurements(plain, turned, checks)
    assert turned.summary.interference == plain.summary.interference


def _bars(separation: float):
    """Two 100 x 4 x 10 bars rotated 45 deg about Z, parallel, centres ``separation`` apart across their width."""

    q45 = axis_angle((0.0, 0.0, 1.0), 45.0)
    across = rotate(q45, (0.0, 1.0, 0.0))
    bar = box((-50.0, -2.0, 0.0), (50.0, 2.0, 10.0))
    a, ma = standing("p1-bar", bar, (200.0, 200.0, 0.0), q45, mate_ref="m1")
    b, mb = standing("p2-bar", bar, (200.0 + separation * across[0], 200.0 + separation * across[1], 0.0), q45, mate_ref="m2")
    return synthetic_package([board(), a, b], [ma, mb])


def _world_box(package, ref):
    """World AABB of a part, from its GLB decoded exactly as the checker decodes it."""

    p = package.scene.part(ref)
    mesh = read_part_mesh(package.files[p.geometry.asset_path])
    pose = (p.pose_in_world.position_mm, p.pose_in_world.quaternion_wxyz)
    pts = [apply(pose, v) for v in mesh.vertices]
    return [min(q[k] for q in pts) for k in range(3)], [max(q[k] for q in pts) for k in range(3)]


@pytest.mark.parametrize("separation, verdict, penetration", [(10.0, None, None), (6.0, None, None), (4.1, None, None),
                                                              (3.5, "MINOR_OVERLAP", 0.5), (2.0, "PART_INTERFERENCE", 2.0)])
def test_rotated_bars_collide_only_when_the_rotated_solids_do(separation, verdict, penetration):
    package = _bars(separation)
    (alo, ahi), (blo, bhi) = _world_box(package, "p1-bar"), _world_box(package, "p2-bar")
    assert all(alo[k] < bhi[k] and blo[k] < ahi[k] for k in range(3))  # world AABBs overlap in every case
    result = validate_assembly_layout(package)
    found = [i for i in result.issues if i.check.value == "PART_COLLISION"]
    if verdict is None:
        assert found == [] and check_status(result, "PART_COLLISION") == "PASS"
        assert {c.check.value: c.evaluated for c in result.checks}["PART_COLLISION"] == 1
    else:
        assert [(i.code.value, i.part_refs) for i in found] == [(verdict, ("p1-bar", "p2-bar"))]
        overlap = 4.0 - separation
        assert metric(found[0], "PENETRATION_MM") == pytest.approx(penetration, abs=1e-3)
        assert metric(found[0], "OVERLAP_VOLUME_MM3") == pytest.approx(overlap * 100.0 * 10.0, rel=1e-3)


Q_ARBITRARY = normalized((0.9, 0.3, -0.2, 0.25))


def _rod_scene(q):
    """A 100 x 2 x 2 rod whose local (-50, 0, 0) end sits at the centre of a 40 mm cube near the +Y board edge.

    With the correct (active, w-first) rotation the rod rises to +x/+y/+z and pokes over the y = 270 outline; with
    the conjugate it would stay inside the outline."""

    centre = (150.0, 250.0, 20.0)
    d = rotate(q, (1.0, 0.0, 0.0))
    position = tuple(centre[k] + 50.0 * d[k] for k in range(3))
    cube, m1 = standing("p2-block", box((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0)), (150.0, 250.0, 0.0), mate_ref="m1")
    rod, m2 = standing("p1-rod", box((-50.0, -1.0, -1.0), (50.0, 1.0, 1.0)), position, q, mate_ref="m2", category="OTHER_SUPPORT")
    corners = [apply((position, q), v) for v in itertools.product((-50.0, 50.0), (-1.0, 1.0), (-1.0, 1.0))]
    return synthetic_package([board((300.0, 270.0)), cube, rod], [m1, m2], outline_max=(300.0, 270.0)), d, corners


def test_arbitrary_quaternion_is_applied_in_world_space():
    package, d, corners = _rod_scene(Q_ARBITRARY)
    excursion = max(c[1] for c in corners) - 270.0
    _, _, conjugate_corners = _rod_scene(q_conj(Q_ARBITRARY))
    assert excursion > 5.0 and max(c[1] for c in conjugate_corners) < 270.0  # the test discriminates the convention
    result = validate_assembly_layout(package)
    outside = issues(result, "PART_OUTSIDE_BREADBOARD")
    assert [i.part_refs for i in outside] == [("p1-rod",)]
    assert metric(outside[0], "OUTSIDE_DISTANCE_MM") == pytest.approx(excursion, abs=1e-3)
    gross = issues(result, "PART_INTERFERENCE")
    assert [i.part_refs for i in gross] == [("p1-rod", "p2-block")]
    inside_length = 20.0 / max(abs(c) for c in d)  # the rod leaves the cube through its +x face
    assert metric(gross[0], "OVERLAP_VOLUME_MM3") == pytest.approx(4.0 * inside_length, rel=5e-3)
    assert metric(gross[0], "PENETRATION_MM") == pytest.approx(2.0, abs=1e-3)


# ============================================================================================== 5. beam axis


def test_reference_beam_axis_is_clear(ref_result):
    assert check_status(ref_result, "BEAM_AXIS_CLEARANCE") == "PASS"


def test_beam_axis_through_a_coaxial_cylinder_and_a_diagonal_box():
    """Exact centre-line lengths: a cylinder whose fan-cap centres lie ON the axis (tie-rule stress) and a diagonal
    crossing of a box."""

    tube, m1 = standing("p1-cyl", cylinder(5.0, -10.0, 10.0), (200.0, 100.0, 50.0), axis_angle((0.0, 1.0, 0.0), 90.0), mate_ref="m1")
    coaxial = validate_assembly_layout(synthetic_package([board(), tube], [m1], beam_axes=[[(0.0, 100.0, 50.0), (150.0, 100.0, 50.0),
                                                                                          (400.0, 100.0, 50.0)]]))
    blocked = issues(coaxial, "BEAM_AXIS_OBSTRUCTED")
    assert [i.part_refs for i in blocked] == [("p1-cyl",)]
    assert metric(blocked[0], "AXIS_LENGTH_IN_PART_MM") == pytest.approx(20.0, abs=1e-5)
    assert check_status(coaxial, "BEAM_AXIS_CLEARANCE") == "WARNING"

    block, m2 = standing("p1-box", box((-10.0, -10.0, 0.0), (10.0, 10.0, 20.0)), (200.0, 200.0, 0.0), mate_ref="m1")
    diagonal = validate_assembly_layout(synthetic_package([board(), block], [m2], beam_axes=[[(150.0, 150.0, 10.0), (250.0, 250.0, 10.0)]]))
    found = issues(diagonal, "BEAM_AXIS_OBSTRUCTED")
    assert metric(found[0], "AXIS_LENGTH_IN_PART_MM") == pytest.approx(20.0 * math.sqrt(2.0), abs=1e-5)
    assert diagonal.status.value == "WARNING"


def test_optical_elements_on_the_axis_are_not_obstructions(ref_package):
    """The lens is crossed by the beam by design; only mechanical parts obstruct it."""

    board = ref_package.scene.breadboard
    x0, x1 = board.outline_min_xy_mm[0], board.outline_max_xy_mm[0]
    (lens_anchor,) = [a for a in ref_package.scene.beam_reference.design_anchors if a.realized_by_part_ref == OPTIC]
    _, y, z = lens_anchor.position_mm
    assert x0 < lens_anchor.position_mm[0] < x1
    # across the board; an axis that does not name its elements (its optics are those on its polyline)
    moved = mutate(lambda s, g: s["beam_reference"]["beam_axes"][0].update(points_mm=[[x0, y, z], [x1, y, z]], through=None))
    result = validate_assembly_layout(moved)
    assert check_status(result, "BEAM_AXIS_CLEARANCE") == "PASS"
    # between the holder's top and the mount's underside the same kind of line crosses only the post, through its axis
    holder_top, mount_bottom = _world_box(ref_package, HOLDER)[1][2], _world_box(ref_package, MOUNT)[0][2]
    assert holder_top < mount_bottom
    post = ref_package.scene.part(POST)
    _, post_y, _ = post.pose_in_world.position_mm
    zp = (holder_top + mount_bottom) / 2.0
    through_post = mutate(lambda s, g: s["beam_reference"]["beam_axes"][0].update(points_mm=[[x0, post_y, zp], [x1, post_y, zp]],
                                                                                    through=None))
    blocked = issues(validate_assembly_layout(through_post), "BEAM_AXIS_OBSTRUCTED")
    assert [i.part_refs for i in blocked] == [(POST,)]
    diameter = post.geometry.bounds_max_mm[0] - post.geometry.bounds_min_mm[0]
    assert metric(blocked[0], "AXIS_LENGTH_IN_PART_MM") == pytest.approx(diameter, abs=0.05)


def test_no_beam_axes_is_not_applicable():
    a, m1 = standing("p1-a", box((-5.0, -5.0, 0.0), (5.0, 5.0, 5.0)), (100.0, 100.0, 0.0), mate_ref="m1")
    result = validate_assembly_layout(synthetic_package([board(), a], [m1]))
    assert check_status(result, "BEAM_AXIS_CLEARANCE") == "NOT_APPLICABLE" and check_status(result, "BOARD_FIXATION") == "NOT_APPLICABLE"
    assert {c.check.value: c.evaluated for c in result.checks}["BEAM_AXIS_CLEARANCE"] == 0


# ================================================================== semantic holes found by review (fixed)


def _poisoned_board_part(value: float):
    cube = box((-10.0, -10.0, 0.0), (10.0, 10.0, 20.0))
    bad = cube.with_vertex((10.0, 10.0, 20.0), (value, 10.0, 20.0))
    a, m1 = standing("p1-a", bad, (100.0, 100.0, 0.0), mate_ref="m1")
    b, m2 = standing("p2-b", box((-10.0, -10.0, 0.0), (10.0, 10.0, 20.0)), (300.0, 300.0, 0.0), mate_ref="m2")
    return synthetic_package([board(), a, b], [m1, m2])


def test_nan_vertex_is_unreadable_geometry():
    result = validate_assembly_layout(_poisoned_board_part(float("nan")))
    assert ("GEOMETRY_UNREADABLE", ("p1-a",), None) in issue_keys(result)
    assert result.status.value == "UNKNOWN"


def test_infinite_vertex_is_unreadable_geometry():
    result = validate_assembly_layout(_poisoned_board_part(float("inf")))
    assert ("GEOMETRY_UNREADABLE", ("p1-a",), None) in issue_keys(result)


def _table_with_pin(overlap: float):
    """Part A: a 90 x 90 table on 4 legs (outside block B) with its top at z = 40 and a 3 mm pin under the top driven
    20 mm into B. B is 60 x 60 and ``40 + overlap`` tall, so the table top also overlaps B's top by ``overlap``."""

    legs = [box((sx * 38.0 - 3.5, sy * 38.0 - 3.5, 0.0), (sx * 38.0 + 3.5, sy * 38.0 + 3.5, 40.0)) for sx in (-1, 1) for sy in (-1, 1)]
    table = combine(*legs, box((-45.0, -45.0, 40.0), (45.0, 45.0, 45.0)), cylinder(1.5, 20.0, 41.0))
    a, ma = standing("p1-table", table, (100.0, 100.0, 0.0), mate_ref="m1")
    block, mb = standing("p2-block", box((-30.0, -30.0, 0.0), (30.0, 30.0, 40.0 + overlap)), (100.0, 100.0, 0.0), mate_ref="m2")
    return synthetic_package([board(), block, a], [mb, ma])


def test_pin_driven_into_a_block_is_gross_when_the_faces_only_touch():
    result = validate_assembly_layout(_table_with_pin(0.0))
    assert [(i.code.value, i.part_refs) for i in result.issues if i.check.value == "PART_COLLISION"] == [
        ("PART_INTERFERENCE", ("p1-table", "p2-block"))]


def test_deep_pin_is_not_masked_by_a_broad_shallow_overlap():
    result = validate_assembly_layout(_table_with_pin(0.3))
    assert [(i.code.value, i.part_refs) for i in result.issues if i.check.value == "PART_COLLISION"] == [
        ("PART_INTERFERENCE", ("p1-table", "p2-block"))]


def test_thin_part_entirely_inside_another_is_gross():
    block, m1 = standing("p1-block", box((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0)), (100.0, 100.0, 0.0), mate_ref="m1")
    window, m2 = standing("p2-window", box((-0.4, -12.5, 0.0), (0.4, 12.5, 25.0)), (100.0, 100.0, 0.0), mate_ref="m2")
    result = validate_assembly_layout(synthetic_package([board(), block, window], [m1, m2]))
    assert [(i.code.value, i.part_refs) for i in result.issues if i.check.value == "PART_COLLISION"] == [
        ("PART_INTERFERENCE", ("p1-block", "p2-window"))]


# ============================================================================== several beams on one breadboard (v0.13)


def test_the_multi_chain_inputs_pass_with_every_beam_clear():
    from drawing_helpers import MULTI_CHAIN, multi_chain_preview

    from os_optics.assembly import ASSEMBLY_CHECKER_VERSION

    for name in MULTI_CHAIN:
        result = validate_assembly_layout(multi_chain_preview(name))
        assert result.status.value == "PASS" and result.checker.checker_version == ASSEMBLY_CHECKER_VERSION == "2.2.0"
        assert not [i for i in result.issues if i.code.value.startswith("BEAM_AXIS")], name


def test_a_beam_through_another_chains_optic_is_an_obstruction():
    """An optic is crossed by design only by the beams it realizes a design anchor on (checker 1.1.0 on): a beam axis that
    runs through another chain's lens, off that lens's anchor, is an obstruction like a mechanical part (v0.12 exempted
    every optic from every beam, which held for one chain only)."""

    from drawing_helpers import edited_preview, multi_chain_preview

    base = multi_chain_preview("two_lenses")

    def stray_beam(scene: dict) -> None:
        lens = next(p for p in scene["parts"] if p["part_ref"] == "b05-optic")
        # through the middle of the lens body (half its thickness along its axis), not along its vertex face
        axis = rotate(lens["pose_in_world"]["quaternion_wxyz"], (0.0, 0.0, 1.0))
        mid = (lens["geometry"]["bounds_min_mm"][2] + lens["geometry"]["bounds_max_mm"][2]) / 2.0
        x, y, z = (p + mid * a for p, a in zip(lens["pose_in_world"]["position_mm"], axis, strict=True))
        axis = dict(scene["beam_reference"]["beam_axes"][0])
        axis.update(path_key="stray", points_mm=[[x, y - 60.0, z + 6.0], [x, y + 20.0, z + 6.0]])
        scene["beam_reference"]["beam_axes"].append(axis)

    result = validate_assembly_layout(edited_preview(stray_beam, base))
    obstructed = {ref for i in result.issues if i.code.value == "BEAM_AXIS_OBSTRUCTED" for ref in i.part_refs}
    assert "b05-optic" in obstructed
    clean = validate_assembly_layout(base)  # each lens on its own beam: crossed by design
    assert not [i for i in clean.issues if i.code.value == "BEAM_AXIS_OBSTRUCTED"]
