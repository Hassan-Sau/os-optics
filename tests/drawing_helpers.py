"""Test helpers for the local Detailed Assembly Drawing (``os_optics.assembly_drawing``).

* the reference pair: the reference Assembly Preview Package (one optic on one support chain) bound to a commercial
  resolution and its Assembly Detail Package (``tests/fixtures/assembly_drawing``; the tests read their bytes, they never
  hard-code their hashes);
* ``edited_preview(edit)``: apply ``edit(scene_payload)`` to a copy of the reference scene and re-sign the package the way
  the service's projector signs it (canonical file bytes, SHA-256, osc1 content hash), with the reference binding;
* ``rebound_detail(preview)``: the reference dimensions rebound to another preview package with the canonical
  ``build_assembly_detail`` (binding copied from that package's manifest), i.e. what the server would issue for it;
* ``sunk_post`` / ``raised_optic_stack``: the FAIL perturbation (the post sunk to z = -12 mm)
  and a consistent pose change of the post and everything it carries;
* ``withheld_dofs``: the DOF delivery cases the reference no longer has (a fixed DOF, a nominal travel), and ``enlarged_board``: the reference on a larger breadboard (free board corners in PLAN);
* the v0.13 multi-chain pairs (``MULTI_CHAIN``: 2 and 4 support chains on one breadboard, from the generalized
  commercial resolution) and ``replicated_chains(package, rows)``: the 4-chain package with its chains repeated in
  offset rows (8, 12, ... chains), re-signed like ``edited_preview``, and ``projected_detail(package, reference)``:
  its detail projected with the server's fixed rule table (what ``project_assembly_detail`` issues), for scaling tests;
* SVG parsing (``svg_tree``, ``groups``, ``metadata``) and the plain-text forbidden vocabulary with its scans
  (``forbidden_hits``; ``encoded_vocabulary_hits`` also finds hashed and regex-shape encodings) -- tests only: the
  engine wheel carries no supplier vocabulary in any form.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from assembly_helpers import box, mate, part, rebuild, resync_mate, set_geometry

from os_optics._vendor.shared_contracts.assembly_detail import (
    MEASURE_TOLERANCE_MM,
    AssemblyDetailPackageV1,
    DimensionBasisV1,
    DimensionKindV1,
    DimensionV1,
    MeasureAxisV1,
    PointKindV1,
    PointRefV1,
    SettingRefV1,
    build_assembly_detail,
    detail_file_bytes,
    point_in_world,
)
from os_optics._vendor.shared_contracts.assembly_preview import (
    AssemblyPreviewPackageV1,
    AssemblyNoteCodeV1,
    note_message,
    load_assembly_preview_package,
    read_package_archive,
)
from os_optics._vendor.shared_contracts.optical_design.dof import DofAxisV1, DofControlIntentV1

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "assembly_drawing"
PREVIEW_ZIP = FIXTURES / "reference.assembly-preview.zip"
DETAIL_JSON = FIXTURES / "reference.assembly-detail.json"
#: The customer-supplied pair: the reference with a customer-supplied polarizer P1 (its catalog model named in the detail).
CUSTOMER_PREVIEW_ZIP = FIXTURES / "customer_supplied.assembly-preview.zip"
CUSTOMER_DETAIL_JSON = FIXTURES / "customer_supplied.assembly-detail.json"
#: v0.13 multi-chain pairs: name -> (preview archive, detail document, number of support chains).
MULTI_CHAIN = {
    "two_lenses": (FIXTURES / "two_lenses.assembly-preview.zip", FIXTURES / "two_lenses.assembly-detail.json", 2),
    "four_lenses": (FIXTURES / "four_lenses.assembly-preview.zip", FIXTURES / "four_lenses.assembly-detail.json", 4),
}
SVG_NS = "{http://www.w3.org/2000/svg}"

#: Stand-ins for the supplier / pipeline vocabulary no drawing output may contain: synthetic vendor and part names, and the
#: demo identity namespace. (The scan of the outputs for the real private identities runs where those are kept.)
FORBIDDEN_TOKENS = ("ACME-OPTICS", "XQ-412B", "ZK-75A", "QB-3", "OS-DEMO")
SUNK_POST_Z_MM = -12.0
RAISE_MM = 10.0


def preview_files() -> dict[str, bytes]:
    return read_package_archive(PREVIEW_ZIP.read_bytes())


def preview() -> AssemblyPreviewPackageV1:
    return load_assembly_preview_package(preview_files())


def detail_bytes() -> bytes:
    return DETAIL_JSON.read_bytes()


def detail_doc() -> dict:
    return json.loads(detail_bytes())


def edited_preview(edit: Callable[[dict], None], base: AssemblyPreviewPackageV1 | None = None) -> AssemblyPreviewPackageV1:
    base = base or preview()
    scene = copy.deepcopy(base.scene.payload())
    edit(scene)
    geometry = {p: bytes(d) for p, d in base.files.items() if p.startswith("geometry/")}
    return rebuild(scene, geometry, like=base)


def rebound_detail(package: AssemblyPreviewPackageV1, *, models: tuple = (), release=None,  # noqa: ANN001
                   line_names: dict | None = None) -> bytes:
    """The reference dimensions, bound to ``package`` by the canonical builder (raises if they do not hold there);
    ``models``: the catalog models of customer-supplied components, ``release``: another release reference,
    ``line_names``: the Simplified Chinese line names (default: the reference's, of the lines ``package`` carries)."""

    reference = AssemblyDetailPackageV1.model_validate_json(detail_bytes())
    if line_names is None:
        lines = {p.commercial_line_ref for p in package.scene.parts if p.commercial_line_ref is not None}
        line_names = {k: v for k, v in reference.line_display_names_zh_cn.items() if k in lines}
    built = build_assembly_detail(preview=package, release=release or reference.release, dimensions=reference.dimensions,
                                  customer_supplied_models=tuple(models), line_display_names_zh_cn=line_names)
    return detail_file_bytes(built)


def set_setting(scene: dict, part_ref: str, kind: str, value: str) -> None:
    next(s for s in scene["settings"] if s["part_ref"] == part_ref and s["kind"] == kind)["value"] = value


def setting_value(scene: dict, part_ref: str, kind: str) -> str:
    return next(s for s in scene["settings"] if s["part_ref"] == part_ref and s["kind"] == kind)["value"]


def sunk_post(scene: dict) -> None:
    """The collision-check FAIL perturbation: the post sunk to z = SUNK_POST_Z_MM, through its holder into the breadboard (mates left
    as they were); the insertion-depth setting follows the pose so that a rebound detail still verifies."""

    post = part(scene, "b03-post")
    x, y, z = post["pose_in_world"]["position_mm"]
    post["pose_in_world"]["position_mm"] = [x, y, SUNK_POST_Z_MM]
    depth = float(setting_value(scene, "b03-post", "INSERTION_DEPTH"))
    set_setting(scene, "b03-post", "INSERTION_DEPTH", f"{depth + z - SUNK_POST_Z_MM:.1f}")


def raised_optic_stack(scene: dict) -> None:
    """A consistent pose change: the post pulled RAISE_MM out of its holder with everything it carries (mount, optic),
    the insertion-depth setting, the post/holder mate, the optical axis height, the beam axes and design anchors with it."""

    for ref in ("b03-post", "b02-optic-mount", "b01-optic"):
        pose = part(scene, ref)["pose_in_world"]
        pose["position_mm"][2] += RAISE_MM
    depth = float(setting_value(scene, "b03-post", "INSERTION_DEPTH")) - RAISE_MM
    set_setting(scene, "b03-post", "INSERTION_DEPTH", f"{depth:.1f}")
    resync_mate(scene, "mate-3")
    assert mate(scene, "mate-3")["b_in_a"]["position_mm"][2] == depth
    beam = scene["beam_reference"]
    beam["optical_axis_height_mm"] += RAISE_MM
    beam["design_frame_in_world"]["position_mm"][2] += RAISE_MM
    for anchor in beam["design_anchors"]:
        anchor["position_mm"][2] += RAISE_MM
    for axis in beam["beam_axes"]:
        for point in axis["points_mm"]:
            point[2] += RAISE_MM
    scene["bounds"]["max_mm"][2] += RAISE_MM


#: DOF implementation fields a DOF without a mechanism does not carry (scene contract).
_DELIVERY_FIELDS = ("implemented_by_part_ref", "moved_part_ref", "travel_min", "travel_max", "travel_basis", "adjustment")


def withheld_dofs(scene: dict) -> None:
    """The DOF delivery cases the reference itself no longer has: its first DOF left without a mechanism (NOT_DELIVERED)
    and the travel of its second one the nominal travel of its mechanism, with the note the scene contract requires."""

    first, second = scene["dof_implementations"][:2]
    assert second.get("travel_basis") == "RESOLVED", "the second reference DOF must have a resolved travel"
    for key in _DELIVERY_FIELDS:
        first.pop(key, None)
    first.update(delivered_control="NONE", status="NOT_DELIVERED", mechanism="NONE")
    second["travel_basis"] = "NOMINAL"
    message = note_message(AssemblyNoteCodeV1.DOF_FIXED, axis=DofAxisV1(first["axis"]), instance=first["design_instance_id"],
                           requested=DofControlIntentV1(first["requested_control"]))
    scene["notes"].append({"code": "DOF_FIXED", "design_instance_id": first["design_instance_id"], "dof_axis": first["axis"],
                           "message": message, "severity": "WARNING"})
    order = list(AssemblyNoteCodeV1)
    scene["notes"].sort(key=lambda n: (order.index(AssemblyNoteCodeV1(n["code"])), n.get("part_ref", ""),
                                       n.get("design_instance_id", ""), n.get("dof_axis", "")))


def enlarged_board(base: AssemblyPreviewPackageV1 | None = None, *, grow_mm: float = 150.0) -> AssemblyPreviewPackageV1:
    """``base`` (the reference) on a breadboard ``grow_mm`` longer in +X and +Y: the hole grid keeps its origin and pitch
    and grows with it, so every world pose, fixation hole and dimension stays; the PLAN view gets free board corners.
    The breadboard proxy becomes a plain slab of the new outline (hole symbols are drawn from the grid)."""

    base = base or preview()
    scene = copy.deepcopy(base.scene.payload())
    geometry = {p: bytes(d) for p, d in base.files.items() if p.startswith("geometry/")}
    board = scene["breadboard"]
    grid = board["hole_grid"]
    (x0, y0), (x1, y1) = board["outline_min_xy_mm"], board["outline_max_xy_mm"]
    width, height = x1 + grow_mm - x0, y1 + grow_mm - y0
    board["outline_max_xy_mm"] = [x1 + grow_mm, y1 + grow_mm]
    grid["count"] = [grid["count"][k] + round(grow_mm / grid["pitch_mm"][k]) for k in (0, 1)]
    board_part = part(scene, board["part_ref"])
    assert board_part["pose_in_world"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    px, py, _ = board_part["pose_in_world"]["position_mm"]
    board_part["identity"].update(opensci_sku=f"OS-MECH-BREADBOARD-{width:.0f}X{height:.0f}-M6G25",
                                  display_name=f"Breadboard {width:.0f} mm x {height:.0f} mm, M6 holes on a 25 mm grid")
    set_geometry(scene, geometry, board["part_ref"], box((x0 - px, y0 - py, 0.0), (x1 + grow_mm - px, y1 + grow_mm - py,
                                                                                     board["thickness_mm"])))
    for k in (0, 1):
        scene["bounds"]["max_mm"][k] = max(scene["bounds"]["max_mm"][k], board["outline_max_xy_mm"][k])
    return rebuild(scene, geometry, like=base)


# ------------------------------------------------------------------------------------------------ SVG


def svg_tree(data: bytes) -> ET.Element:
    return ET.fromstring(data)


def groups(root: ET.Element, cls: str, **attrs: str) -> list[ET.Element]:
    out = []
    for el in root.iter(f"{SVG_NS}g"):
        if cls in (el.get("class") or "").split() and all(el.get(f"data-{k.replace('_', '-')}") == v for k, v in attrs.items()):
            out.append(el)
    return out


def texts(el: ET.Element) -> list[str]:
    return [t.text or "" for t in el.iter(f"{SVG_NS}text")]


def metadata(root: ET.Element) -> dict:
    node = next(el for el in root.iter(f"{SVG_NS}metadata") if el.get("id") == "opensci-drawing-metadata")
    return json.loads(node.text or "{}")


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def path_bounds(el: ET.Element) -> tuple[float, float, float, float]:
    """Paper bounding box of every path under ``el``."""

    xs, ys = [], []
    for path in el.iter(f"{SVG_NS}path"):
        nums = [float(n) for n in _NUM.findall(path.get("d", ""))]
        xs += nums[0::2]
        ys += nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def flat(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def forbidden_hits(text: str) -> list[str]:
    """The repository leak-scan rule: case-insensitive, separator-blind for tokens of 5+ alphanumerics."""

    lowered, flattened = text.lower(), flat(text)
    return [t for t in FORBIDDEN_TOKENS if t.lower() in lowered or (len(flat(t)) >= 5 and flat(t) in flattened)]


def _hashed_forms(token: str) -> set[str]:
    lowered = token.lower()
    stripped = flat(token)
    return {hashlib.new(algorithm, form.encode()).hexdigest()
            for form in {token, lowered, token.upper(), stripped, stripped.upper()} for algorithm in ("sha256", "sha1", "md5")}


def _shape_forms(form: str) -> set[str]:
    """Character-class shapes of ``form`` as a regex would spell them (run-length and one-class-per-character)."""

    classes = [("[a-z]" if c.isalpha() else "[0-9]" if c.isdigit() else re.escape(c)) for c in form]
    runs: list[list] = []
    for cls in classes:
        if runs and runs[-1][0] == cls:
            runs[-1][1] += 1
        else:
            runs.append([cls, 1])
    run_form = "".join(cls if n == 1 else f"{cls}{{{n}}}" for cls, n in runs)
    return {run_form, "".join(classes), run_form.replace(r"\-", "-")}


def encoded_vocabulary_hits(text: str) -> list[str]:
    """The forbidden tokens ``text`` carries in plain (the leak-scan rule), hashed (SHA-256 / SHA-1 / MD5 of the
    original, lower-case, upper-case and separator-stripped forms) or regex-shape form; plus ``HEX64`` if it holds
    any 64-hex literal at all (source code has no reason to)."""

    lowered = text.lower()
    hits = [f"plain:{t}" for t in forbidden_hits(text)]
    for token in FORBIDDEN_TOKENS:
        if any(digest in lowered for digest in _hashed_forms(token)):
            hits.append(f"hashed:{token}")
        if any(shape in text for form in (token.lower(), flat(token)) for shape in _shape_forms(form)):
            hits.append(f"shape:{token}")
    if re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", lowered):
        hits.append("HEX64")
    return hits


# ------------------------------------------------------------------------------------------------ multi-chain


def multi_chain_preview(name: str) -> AssemblyPreviewPackageV1:
    return load_assembly_preview_package(read_package_archive(MULTI_CHAIN[name][0].read_bytes()))


def _renamed_glb(glb: bytes, part_ref: str) -> bytes:
    """The same GLB with its node and mesh named ``part_ref`` (the package contract requires the part's own ref)."""

    magic, version, _ = struct.unpack_from("<4sII", glb, 0)
    length, kind = struct.unpack_from("<II", glb, 12)
    document = json.loads(glb[20:20 + length])
    document["nodes"][0]["name"] = document["meshes"][0]["name"] = part_ref
    chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)
    rest = glb[20 + length:]
    return struct.pack("<4sII", magic, version, 20 + len(chunk) + len(rest)) + struct.pack("<II", len(chunk), kind) + chunk + rest


def _same(a, b) -> bool:  # noqa: ANN001
    return all(abs(a[k] - b[k]) < 1e-9 for k in range(3))


def replicated_chains(base: AssemblyPreviewPackageV1, rows: int, *, shift: tuple[float, float] = (50.0, 150.0)) -> AssemblyPreviewPackageV1:
    """``base`` (the 4-chain package: chains in a row along X, beams along +Y from customer sources off the board) with
    its support chains repeated in ``rows`` rows, row r moved by (r % 2 * shift x, r * shift y) so that no beam runs
    through another chain; the breadboard grows to hold them. Every chain gets its own parts, BOM lines, mates, settings,
    fixation, DOFs, notes, design anchors (optic and source) and beam axis."""

    scene = copy.deepcopy(base.scene.payload())
    geometry = {p: bytes(d) for p, d in base.files.items() if p.startswith("geometry/")}
    board_ref = scene["breadboard"]["part_ref"]
    grid = scene["breadboard"]["hole_grid"]
    pitch = grid["pitch_mm"]
    originals = sorted((p for p in scene["parts"] if p["part_ref"] != board_ref), key=lambda p: p["assembly_order"])
    beam = scene["beam_reference"]
    anchors, axes = list(beam["design_anchors"]), list(beam["beam_axes"])
    mates, settings, fixations = list(scene["mates"]), list(scene["settings"]), list(scene["fixations"])
    dofs, notes = list(scene["dof_implementations"]), list(scene["notes"])
    line = max(int(p["commercial_line_ref"][1:]) for p in scene["parts"]) + 1
    order = max(p["assembly_order"] for p in scene["parts"]) + 1
    mate_no = max(int(m["mate_ref"].split("-")[1]) for m in mates) + 1
    lens_no = src_no = beam_no = len([a for a in anchors if a["customer_supplied"]]) + 1
    for r in range(1, rows):
        dx, dy = (r % 2) * shift[0], r * shift[1]
        rename: dict[str, str] = {}
        ids: dict[str, str] = {}
        mate_names: dict[str, str] = {}
        for p in originals:
            rename[p["part_ref"]] = f"b{line:02d}-{p['part_ref'].split('-', 1)[1]}"
            line += 1
        for m in mates:
            if m["part_a_ref"] in rename or m["part_b_ref"] in rename:
                mate_names[m["mate_ref"]] = f"mate-{mate_no}"
                mate_no += 1
        for anchor in anchors:
            if anchor.get("realized_by_part_ref") in rename:
                ids[anchor["instance_id"]] = f"L{lens_no}"
                lens_no += 1
        for axis in axes:  # the source feeding each cloned optic, and their beam
            lens = next((a for a in anchors if a["instance_id"] in ids and any(_same(a["position_mm"], q) for q in axis["points_mm"])), None)
            source = next((a for a in anchors if a["customer_supplied"] and _same(a["position_mm"], axis["points_mm"][0])), None)
            if lens is not None and source is not None:
                ids[source["instance_id"]] = f"SRC{src_no}"
                src_no += 1
                points = [[q[0] + dx, q[1] + dy, q[2]] for q in (source["position_mm"], lens["position_mm"])]
                beam["beam_axes"].append({**copy.deepcopy(axis), "path_key": f"beam-{beam_no}", "points_mm": points})
                beam_no += 1
        for p in originals:
            new = copy.deepcopy(p)
            ref = rename[p["part_ref"]]
            new.update(part_ref=ref, commercial_line_ref="B" + ref[1:3], assembly_order=order)
            order += 1
            x, y, z = p["pose_in_world"]["position_mm"]
            new["pose_in_world"]["position_mm"] = [x + dx, y + dy, z]
            new["geometry"]["asset_path"] = f"geometry/{ref}.glb"
            if new.get("design_binding"):
                new["design_binding"]["instance_id"] = ids[new["design_binding"]["instance_id"]]
            if new.get("support"):
                new["support"].update(parent_ref=rename.get(new["support"]["parent_ref"], new["support"]["parent_ref"]),
                                      mate_ref=mate_names[new["support"]["mate_ref"]])
            geometry[new["geometry"]["asset_path"]] = _renamed_glb(geometry[p["geometry"]["asset_path"]], ref)
            scene["parts"].append(new)
        for m in mates:
            if m["mate_ref"] in mate_names:
                scene["mates"].append({**copy.deepcopy(m), "mate_ref": mate_names[m["mate_ref"]],
                                       "part_a_ref": rename.get(m["part_a_ref"], m["part_a_ref"]),
                                       "part_b_ref": rename.get(m["part_b_ref"], m["part_b_ref"])})
        scene["settings"] += [{**copy.deepcopy(s), "part_ref": rename[s["part_ref"]]} for s in settings if s["part_ref"] in rename]
        scene["fixations"] += [{**copy.deepcopy(f), "part_ref": rename[f["part_ref"]],
                                "board_holes": [[i + round(dx / pitch[0]), j + round(dy / pitch[1])] for i, j in f["board_holes"]]}
                               for f in fixations if f["part_ref"] in rename]
        for d in dofs:
            if d["design_instance_id"] in ids:
                new = copy.deepcopy(d)
                new["design_instance_id"] = ids[d["design_instance_id"]]
                for key in ("implemented_by_part_ref", "moved_part_ref", "optic_part_ref"):
                    if new.get(key):
                        new[key] = rename[new[key]]
                scene["dof_implementations"].append(new)
        for d in notes:
            if d.get("part_ref") in rename or d.get("design_instance_id") in ids:
                new = copy.deepcopy(d)
                if "part_ref" in new:
                    new["part_ref"] = rename[new["part_ref"]]
                if "design_instance_id" in new:
                    new["design_instance_id"] = ids[new["design_instance_id"]]
                    new["message"] = new["message"].replace(d["design_instance_id"], new["design_instance_id"])
                scene["notes"].append(new)
        for anchor in anchors:
            if anchor["instance_id"] in ids:
                new = copy.deepcopy(anchor)
                new["instance_id"] = ids[anchor["instance_id"]]
                x, y, z = anchor["position_mm"]
                new["position_mm"] = [x + dx, y + dy, z]
                if new.get("realized_by_part_ref"):
                    new["realized_by_part_ref"] = rename[new["realized_by_part_ref"]]
                beam["design_anchors"].append(new)
    width = scene["breadboard"]["outline_max_xy_mm"][0]
    height = rows * shift[1]
    scene["breadboard"]["outline_max_xy_mm"] = [width, height]
    grid["count"] = [grid["count"][0], round((height - 2.0 * grid["origin_xy_mm"][1]) / pitch[1]) + 1]
    board = part(scene, board_ref)
    board["identity"].update(opensci_sku=f"OS-MECH-BREADBOARD-{width:.0f}X{height:.0f}-M6G25",
                             display_name=f"Breadboard {width:.0f} mm x {height:.0f} mm, M6 holes on a 25 mm grid")
    set_geometry(scene, geometry, board_ref, box((0.0, 0.0, 0.0), (width, height, scene["breadboard"]["thickness_mm"])))
    scene["bounds"]["max_mm"][1] = max(scene["bounds"]["max_mm"][1], height)
    for m in scene["mates"]:
        if m["mate_ref"] not in {x["mate_ref"] for x in mates}:
            resync_mate(scene, m["mate_ref"])
    return rebuild(scene, geometry, like=base)


def projected_detail(package: AssemblyPreviewPackageV1, reference: bytes) -> bytes:
    """The Assembly Detail Package the server's fixed rule table (``project_assembly_detail``) issues for ``package``,
    under the release of the ``reference`` detail document."""

    scene = package.scene
    ref_doc = AssemblyDetailPackageV1.model_validate_json(reference)
    board = scene.breadboard
    corner = PointRefV1(kind=PointKindV1.BOARD_CORNER, part_ref=board.part_ref)
    top = PointRefV1(kind=PointKindV1.PART_DATUM, part_ref=board.part_ref, datum_key="board_top")
    plan = (MeasureAxisV1.WORLD_X, MeasureAxisV1.WORLD_Y)
    axes = (MeasureAxisV1.WORLD_X, MeasureAxisV1.WORLD_Y, MeasureAxisV1.WORLD_Z)
    specs: list = []
    for p in sorted(scene.parts, key=lambda p: (p.assembly_order, p.part_ref)):
        origin = PointRefV1(kind=PointKindV1.PART_ORIGIN, part_ref=p.part_ref)
        for fixation in (f for f in scene.fixations if f.part_ref == p.part_ref):
            for i, j in fixation.board_holes:
                hole = PointRefV1(kind=PointKindV1.BOARD_HOLE, part_ref=board.part_ref, hole=(i, j))
                specs += [(DimensionKindV1.FIXATION_HOLE_POSITION, p.part_ref, corner, hole, a, DimensionBasisV1.RESOLVED, None) for a in plan]
                here, there = point_in_world(origin, scene), point_in_world(hole, scene)
                specs += [(DimensionKindV1.FIXATION_OFFSET, p.part_ref, origin, hole, a, DimensionBasisV1.REFERENCE, None)
                          for k, a in enumerate(plan) if abs(there[k] - here[k]) > MEASURE_TOLERANCE_MM]
        if p.support is not None and p.support.relation.value == "INSERTED_INTO":
            setting = next((s for s in scene.settings if s.part_ref == p.part_ref and s.kind.value == "INSERTION_DEPTH"), None)
            m = next(m for m in scene.mates if m.mate_ref == p.support.mate_ref)
            own, parents = (m.datum_a, m.datum_b) if m.part_a_ref == p.part_ref else (m.datum_b, m.datum_a)
            if setting is not None:
                specs.append((DimensionKindV1.INSERTION_DEPTH, p.part_ref,
                              PointRefV1(kind=PointKindV1.PART_DATUM, part_ref=p.part_ref, datum_key=own),
                              PointRefV1(kind=PointKindV1.PART_DATUM, part_ref=p.support.parent_ref, datum_key=parents),
                              MeasureAxisV1.WORLD_Z, DimensionBasisV1(setting.basis.value), SettingRefV1(part_ref=p.part_ref, kind=setting.kind)))
        if p.category.value == "OPTICAL_ELEMENT":
            specs.append((DimensionKindV1.OPTICAL_AXIS_HEIGHT, p.part_ref, top, origin, MeasureAxisV1.WORLD_Z, DimensionBasisV1.RESOLVED, None))
            specs += [(DimensionKindV1.OPTIC_BOARD_POSITION, p.part_ref, corner, origin, a, DimensionBasisV1.RESOLVED, None) for a in plan]
            for anchor in sorted(scene.beam_reference.design_anchors, key=lambda a: a.instance_id):
                if anchor.customer_supplied:
                    src = PointRefV1(kind=PointKindV1.DESIGN_ANCHOR, instance_id=anchor.instance_id)
                    a, b = point_in_world(src, scene), point_in_world(origin, scene)
                    deltas = [abs(b[k] - a[k]) for k in range(3)]
                    specs.append((DimensionKindV1.EXTERNAL_SOURCE_STANDOFF, p.part_ref, src, origin, axes[deltas.index(max(deltas))],
                                  DimensionBasisV1.CUSTOMER_DESIGN, None))
    dimensions = tuple(DimensionV1(dim_ref=f"dim-{n}", kind=kind, subject_part_ref=subject, from_point=frm, to_point=to, axis=axis,
                                   basis=basis, setting=setting)
                       for n, (kind, subject, frm, to, axis, basis, setting) in enumerate(specs, start=1))
    return detail_file_bytes(build_assembly_detail(preview=package, release=ref_doc.release, dimensions=dimensions))
