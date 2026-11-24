"""The paid Detailed Assembly Drawing as a customer deliverable.

* one role vocabulary (the drawing manifest's roles are the artifact store's), the message catalog with the same keys
  and named_fields in English and Simplified Chinese, a zh-CN sheet with Chinese PDF text (predefined CID font, no
  embedded font file) that still passes the output allow-list;
* the Order Number and the customer in the title block, one check statement;
* the optical spacing table and the alignment procedure derived from the package facts;
* customer-supplied components located from the board corner, with an outside-the-breadboard statement;
* labels: coincident beam axes share one label, and no two view labels overlap;
* the assembly view's beam colours.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest
from drawing_helpers import DETAIL_JSON, PREVIEW_ZIP, detail_doc, edited_preview, groups, metadata, rebound_detail, svg_tree, texts

from os_optics._vendor.shared_contracts.assembly_detail import (
    LEGACY_DRAWING_ROLE_ALIASES,
    DetailReleaseRefV1,
    verify_drawing_outputs,
)
from os_optics._vendor.shared_contracts.client_assembly_artifact import ROLE_RULES
from os_optics.assembly_drawing import AssemblyDrawingError, DrawingErrorCode, DrawingStatusV1, generate_assembly_drawing
import drawing_text_check as privacy
from os_optics.assembly_drawing import i18n
from os_optics.assembly_drawing.text import mm, text_width, wrap
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.optical_layout import messages as layout_messages

SVG, PDF, RESULT, MANIFEST = "assembly_drawing.svg", "assembly_drawing.pdf", "drawing_result.json", "drawing_manifest.json"
CJK = re.compile(r"[一-鿿]")


@pytest.fixture(scope="module")
def en():
    return generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON)


@pytest.fixture(scope="module")
def zh():
    return generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, locale="zh-CN")


# ============================================================================== roles


def test_the_manifest_roles_are_the_artifact_store_roles(en):
    store = {role.value: rule.filename for role, rule in ROLE_RULES.items()}
    assert {f.role.value: f.path for f in en.manifest.files} == {
        "DRAWING_PDF": PDF, "DRAWING_RESULT": RESULT, "DRAWING_SVG": SVG}
    assert all(store[f.role.value] == f.path for f in en.manifest.files)
    assert LEGACY_DRAWING_ROLE_ALIASES == {"ASSEMBLY_DRAWING_SVG": "DRAWING_SVG", "ASSEMBLY_DRAWING_PDF": "DRAWING_PDF"}
    assert b"ASSEMBLY_DRAWING_PDF" not in en.files[MANIFEST]


def test_artifacts_download_help_lists_the_roles(capsys):
    with pytest.raises(SystemExit):
        cli_main(["artifacts", "download", "--help"])
    out = capsys.readouterr().out
    for role in ("DRAWING_PDF", "DRAWING_SVG", "DRAWING_RESULT", "DRAWING_MANIFEST", "VALIDATION_RESULT", "ASSEMBLY_DRAWING_PDF"):
        assert role in out, role
    with pytest.raises(SystemExit):
        cli_main(["assembly", "drawing", "--help"])
    assert "--locale" in capsys.readouterr().out


# ============================================================================== locale


@pytest.mark.parametrize("catalog", [i18n.CATALOG, layout_messages.CATALOG], ids=["assembly", "layout"])
def test_both_locales_have_the_same_keys_and_placeholders(catalog):
    en_messages, zh_messages = catalog.messages["en"], catalog.messages["zh-CN"]
    assert set(en_messages) == set(zh_messages)
    for key, template in en_messages.items():
        assert i18n.named_fields(template) == i18n.named_fields(zh_messages[key]), key
        assert template.strip() and zh_messages[key].strip(), key


def test_every_dof_mechanism_status_and_role_has_catalog_words():
    from os_optics._vendor.shared_contracts.assembly_preview.scene import AssemblyNoteCodeV1, DofMechanismV1

    for prefix, values in (("mechanism", DofMechanismV1), ("note_code", AssemblyNoteCodeV1)):
        for value in values:
            assert i18n.CATALOG.has(f"{prefix}.{value.value}"), (prefix, value)


def test_a_zh_cn_drawing_is_written_in_chinese_and_verifies(zh, en):
    assert zh.result.locale == "zh-CN" == zh.manifest.locale and en.result.locale == "en"
    assert verify_drawing_outputs(zh.files) == ()
    root = svg_tree(zh.files[SVG])
    assert root.get("{http://www.w3.org/XML/1998/namespace}lang") == "zh-CN"
    assert "Microsoft YaHei" in root.get("font-family") and "Noto Sans CJK SC" in root.get("font-family")
    (banner,) = groups(root, "notice-banner")
    assert texts(banner)[0] == "详细装配图"
    assert metadata(root)["locale"] == "zh-CN"
    steps = zh.result.assembly_steps
    assert steps[0].text.startswith("放置面包板 B05") and all(CJK.search(s.text) for s in steps)
    # the same facts as the English sheet: identities and numbers do not change with the language
    assert [s.action for s in steps] == [s.action for s in en.result.assembly_steps]
    assert zh.result.dimensions == en.result.dimensions and zh.result.optical_spacing == en.result.optical_spacing


def test_the_chinese_pdf_uses_the_predefined_cid_font_and_carries_the_text(zh, en):
    pdf = zh.files[PDF]
    for needle in (b"/Subtype /Type0", b"/BaseFont /STSong-Light", b"/Encoding /UniGB-UCS2-H", b"/Ordering (GB1)", b"/DW 1000"):
        assert needle in pdf, needle
    assert b"FontFile" not in pdf  # no embedded font program
    decoded = [text for _, text in privacy.pdf_strings(PDF, pdf) if CJK.search(text)]
    assert "详细装配图" in decoded
    assert any("放置面包板" in t for t in decoded)
    # an English sheet needs no Chinese font
    assert b"STSong-Light" not in en.files[PDF] and b"/F3" not in en.files[PDF]
    # every xref offset points at its object
    xref = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    count = int(pdf[xref:].split(b"\n")[1].split()[1])
    for number, entry in enumerate(pdf[xref:].split(b"\n")[3:3 + count - 1], start=1):
        assert pdf[int(entry.split()[0]):].startswith(f"{number} 0 obj".encode()), number


def test_wide_characters_are_one_em_and_chinese_wraps_between_characters():
    assert text_width("详细装配图", 2.0) == pytest.approx(10.0)
    assert text_width("A", 1.0) == pytest.approx(0.667)
    lines = wrap("将 B04（OS-MECH-POST）立于 B05 上；通过槽孔固定在孔 (5, 4)。", 20.0, 2.0)
    assert len(lines) > 1 and all(text_width(line, 2.0) <= 20.0 + 2.0 for line in lines)
    assert not any(line.startswith(("，", "。", "；", "）")) for line in lines)


def test_an_unknown_locale_is_refused():
    with pytest.raises(AssemblyDrawingError) as caught:
        generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, locale="fr")
    assert caught.value.code is DrawingErrorCode.INPUT_INVALID
    reply = json_api.generate_assembly_drawing(PREVIEW_ZIP.read_bytes(), DETAIL_JSON.read_bytes(), {"locale": "fr"})
    assert reply["ok"] is False and reply["error"]["code"] == "INPUT_INVALID"


def test_the_json_api_and_the_cli_take_the_locale(tmp_path, capsys):
    reply = json_api.generate_assembly_drawing(PREVIEW_ZIP.read_bytes(), DETAIL_JSON.read_bytes(), {"locale": "zh-CN"})
    assert reply["ok"] and reply["locale"] == "zh-CN" and reply["result"]["locale"] == "zh-CN"
    assert "drawing_locales" in json_api.capabilities() and json_api.capabilities()["drawing_locales"] == ["en", "zh-CN"]
    out = tmp_path / "zh"
    assert cli_main(["assembly", "draw", str(PREVIEW_ZIP), "--detail", str(DETAIL_JSON), "--out", str(out), "--locale", "zh-CN"]) == 0
    assert '"locale": "zh-CN"' in capsys.readouterr().out
    assert "详细装配图" in (out / SVG).read_text(encoding="utf-8")


def test_the_output_allow_list_still_checks_chinese_text(monkeypatch):
    """A Chinese character that is in no template and no input is reported by the build-time gate, like an unknown
    ASCII token."""

    from drawing_helpers import detail_bytes, preview

    from os_optics._vendor.shared_contracts.assembly_detail import load_assembly_detail

    monkeypatch.setitem(i18n.ZH, "title", i18n.ZH["title"] + "鑫")
    with privacy.unchecked():
        drawing = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, locale="zh-CN")
    generator = drawing.result.generator.model_dump(mode="json").values()
    problems = privacy.drawing_problems(drawing.files, load_assembly_detail(detail_bytes(), preview()), drawing.validation,
                                        (str(v) for v in generator))
    assert any(p.startswith(f"{PDF}: text string") for p in problems)
    assert privacy.tokens("光学 AB12") == {"光", "学", "ab12"}


# ============================================================================== title block


def confirmed(customer: str = "北京光学实验室有限公司") -> bytes:
    from drawing_helpers import preview

    return rebound_detail(preview(), release=DetailReleaseRefV1(basis="CONFIRMED_ORDER", order_number="OS-20261006-0007",
                                                                customer_display_name=customer))


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_the_title_block_names_the_order_number_and_the_customer(locale):
    drawing = generate_assembly_drawing(PREVIEW_ZIP, confirmed(), locale=locale)
    root = svg_tree(drawing.files[SVG])
    (order,) = groups(root, "order-number")
    (customer,) = groups(root, "customer")
    assert texts(order)[-1] == "OS-20261006-0007"
    assert "".join(texts(customer)[1:]) == "北京光学实验室有限公司"
    assert drawing.result.binding.release.order_number == "OS-20261006-0007"
    assert metadata(root)["order_number"] == "OS-20261006-0007"
    # the source identities are data (SVG metadata, drawing result), not title-block cells
    assert not groups(root, "references")
    assert drawing.result.binding.commercial_resolution_id not in " ".join(texts(groups(root, "title-block")[0]))
    assert metadata(root)["commercial_resolution_id"] == drawing.result.binding.commercial_resolution_id
    if locale == "zh-CN":
        decoded = [text for _, text in privacy.pdf_strings(PDF, drawing.files[PDF])]
        assert "北京光学实验室有限公司" in decoded


def test_a_long_customer_name_is_fitted_into_its_cell():
    name = "Institute of Optics and Photonics, Department of Applied Laser Physics, University Laboratory 12"
    drawing = generate_assembly_drawing(PREVIEW_ZIP, confirmed(name))
    (customer,) = groups(svg_tree(drawing.files[SVG]), "customer")
    assert " ".join(texts(customer)[1:]) == name


# ============================================================================== spacing and alignment


def test_the_optical_spacing_follows_the_design_beam_path(en):
    rows = [(r.path_key, r.from_instance_id, r.to_instance_id, r.distance_mm) for r in en.result.optical_spacing]
    # measured to the lens's optical centre (the middle of its 3.6 mm on its axis, the point the alignment
    # names and its board position refers to), not to its first vertex. Physical-air focusing moves the image plane
    # +0.080 mm; the reference scene's design-anchor distance, rather than an index-one focal pin, is the authority.
    assert rows == [("beam-1", "SRC", "L1", "201.8"), ("beam-1", "L1", "focal_plane", "100.3")]
    from drawing_helpers import preview

    anchors = {a.instance_id: a.position_mm for a in preview().scene.beam_reference.design_anchors}
    lens_vertex = next(p.pose_in_world.position_mm for p in preview().scene.parts if p.design_binding is not None and
                       p.design_binding.instance_id == "L1")
    # The lens's optical centre is half its 3.6 mm thickness past its first vertex along +X.
    assert float(rows[1][3]) == pytest.approx(round(anchors["focal_plane"][0] - (lens_vertex[0] + 1.8), 1))
    root = svg_tree(en.files[SVG])
    table = " ".join(texts(groups(root, "optical-spacing")[0]))
    assert "beam-1 (780 nm):" in table and "SRC -> L1 (B01): 201.8" in table


def test_the_alignment_procedure_is_derived_from_the_package_facts(en):
    from drawing_helpers import preview

    scene = preview().scene
    source = next(a for a in scene.beam_reference.design_anchors if a.instance_id == "SRC")
    lens = next(a for a in scene.beam_reference.design_anchors if a.instance_id == "L1")
    items = texts(groups(svg_tree(en.files[SVG]), "alignment")[0])
    text = " ".join(items)
    assert (f"Bring the input beam of SRC onto the design axis at 125 mm above the board top: it must pass through R1 x "
            f"{mm(source.position_mm[0])} y {mm(source.position_mm[1])} z {mm(source.position_mm[2])}") in text
    assert f"R2 x {mm(lens.position_mm[0])} y {mm(lens.position_mm[1])}" in text and "along world +X" in text
    assert "Place the optics in beam order along beam-1: L1 (B01)." in text
    # the base slot moves L1 along the beam: it is the adjustment that sets the source -> L1 spacing
    assert "Tz of L1 (B01) (base slot of B04, travel -12.5 to +12.5 mm) moves it along the beam" in text
    assert "sets the spacing SRC -> L1 (B01) (201.8 mm)" in text
    # the spacing is checked between the points it is measured to, which the procedure names
    assert "between the optic centres (a lens, window, filter, waveplate or polarizer: the middle of its thickness" in text
    for invented in ("collimat", "focus", "interferomet", "autocollimator", "shear"):
        assert invented not in text.lower(), invented


# ============================================================================== customer-supplied components


def test_customer_equipment_outside_the_breadboard_is_stated_with_its_height(en):
    root = svg_tree(en.files[SVG])
    (row,) = groups(root, "customer-supplied-row")
    assert row.get("data-instance-id") == "SRC" and row.get("data-outside-board") == "true"
    text = " ".join(texts(row))
    assert "OUTSIDE THE BREADBOARD: 120.8 mm beyond its -X edge; it needs its own support at 125 mm above the board top" in text
    assert "design position x -120.8 y 87.5 z 125" in text and "mm from the board" not in text  # no negative distance


# ============================================================================== labels


def _label_boxes(root, view):
    boxes = []
    for kind in ("beam-label", "customer-supplied-label"):
        for group in groups(root, kind, view=view):
            for t in group.iter("{http://www.w3.org/2000/svg}text"):
                size, x, y = float(t.get("font-size")), float(t.get("x")), float(t.get("y"))
                width = text_width(t.text, size, t.get("font-weight") == "bold")
                x0 = {"start": x, "middle": x - width / 2, "end": x - width}[t.get("text-anchor", "start")]
                boxes.append((x0, y - 0.75 * size, x0 + width, y + 0.25 * size))
    return boxes


def two_colour_beams(scene: dict) -> None:
    """The reference beam split into a 532 nm and a 1064 nm axis along the same line (a two-colour design)."""

    (axis,) = scene["beam_reference"]["beam_axes"]
    axis["wavelength_nm"] = 532.0
    scene["beam_reference"]["beam_axes"].append({**axis, "path_key": "beam-2", "wavelength_nm": 1064.0})


def test_coincident_beam_axes_share_one_label_and_labels_do_not_overlap():
    package = edited_preview(two_colour_beams)
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    root = svg_tree(drawing.files[SVG])
    for view in ("FRONT", "PLAN"):
        labels = groups(root, "beam-label", view=view)
        assert [texts(g) for g in labels] == [["DESIGN BEAM AXES 532, 1064 nm"]], view
        assert labels[0].get("data-path-keys") == "beam-1 beam-2"
        boxes = _label_boxes(root, view)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (view, a, b)


def test_the_pictorial_says_board_in_part_only_when_it_is_cropped(en):
    from drawing_helpers import enlarged_board
    from os_optics.assembly_drawing.model import drawing_facts
    from os_optics.assembly_drawing.sheet import SheetBuilder
    from os_optics.assembly_drawing.steps import assembly_steps
    from os_optics._vendor.shared_contracts.assembly_detail import load_assembly_detail

    for package, detail in ((None, None), (enlarged_board(), None)):
        if package is None:
            from drawing_helpers import detail_bytes, preview
            package, detail = preview(), detail_bytes()
        else:
            detail = rebound_detail(package)
        facts = drawing_facts(load_assembly_detail(detail, package))
        builder = SheetBuilder(facts, drawing_status=DrawingStatusV1.DETAILED_ASSEMBLY_DRAWING, validation_status="PASS", validation_issues=[], checker_version="1",
                               steps=assembly_steps(facts), generator_version="1", validation_interference="NO_INTERFERENCE")
        root = svg_tree(__import__("os_optics.assembly_drawing.svg", fromlist=["render_svg"]).render_svg(builder.build(), "t", "d", {}))
        board = facts.board
        (lo, hi) = builder.pictorial_crop
        cropped = (lo[0], lo[1], hi[0], hi[1]) != (board.x0, board.y0, board.x1, board.y1)
        label = " ".join(texts(groups(root, "view", view="PICTORIAL")[0]))
        assert ("breadboard shown in part" in label) == cropped
        assert "axonometric" in label


# ============================================================================== beam colours


def test_the_default_beam_colour_differs_from_every_wavelength_colour():
    from os_optics.optical_view.colors import wavelength_display_rgb
    from os_optics.viewer.assembly import DEFAULT_BEAM_RGB

    for nm in range(200, 2001):
        assert wavelength_display_rgb(float(nm)) != DEFAULT_BEAM_RGB, nm


def test_no_internal_language_on_the_sheet(en, zh):
    for drawing in (en, zh):
        text = " ".join(texts(svg_tree(drawing.files[SVG])))
        for internal in ("SUPPORT CHAIN", "FIXATIONS in", "zero-based", "height closure", "intra chain envelope", "projector",
                         "b01-optic", "b05-breadboard", "not a BOM line", "not BOM lines", "RESOLUTION CHECKS PASSED"):
            assert internal.lower() not in text.lower(), internal
        assert not re.search(r"\d\.\d{2,}\b", text.replace(drawing.result.binding.commercial_resolution_id, "")), "0.1 mm"
    assert detail_doc()["schema_version"] == "3.1"


# ============================================================================== delivery checks

CJK_SPACE_CJK = re.compile(r"[一-鿿（）；，。] +[一-鿿（）；，。]")
LATIN_CJK = re.compile(r"[A-Za-z0-9][一-鿿]|[一-鿿][A-Za-z0-9]")
LINE_NAMES = {"B01": "平凸透镜 D25.4 mm f 100 mm", "B02": "XY 平移镜架 D25", "B03": "光学支杆 D12.7 长 75 mm",
              "B04": "支杆座（带底座）75 mm", "B05": "面包板 150 mm x 150 mm，M6 孔距 25 mm"}


def test_the_zh_parts_legend_names_the_lines_in_chinese():
    from drawing_helpers import preview

    detail = rebound_detail(preview(), line_names=LINE_NAMES)
    zh = svg_tree(generate_assembly_drawing(PREVIEW_ZIP, detail, locale="zh-CN").files[SVG])
    rows = {g.get("data-line-ref"): "".join(texts(g)) for g in groups(zh, "legend-row")}
    assert all("".join(name.split()) in "".join(rows[line].split()) for line, name in LINE_NAMES.items())
    assert "Plano-convex" not in " ".join(rows.values())
    en = svg_tree(generate_assembly_drawing(PREVIEW_ZIP, detail).files[SVG])
    assert "Plano-convex" in " ".join(" ".join(texts(g)) for g in groups(en, "legend-row"))


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_equipment_outside_the_board_has_no_negative_edge_distance(locale):
    drawing = generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, locale=locale)
    root = svg_tree(drawing.files[SVG])
    texts_ = " ".join(texts(groups(root, "customer-supplied-table")[0])) + " ".join(s.text for s in drawing.result.assembly_steps)
    assert not re.search(r"-\d+(\.\d+)? mm from|距 ?-?[XY]? ?边 ?-\d|-X 边 -|-Y 边 -", texts_)
    assert ("120.8 mm beyond" in texts_) if locale == "en" else ("120.8 mm" in texts_ and "超出" in texts_)


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_a_drawing_on_the_customers_table_never_says_breadboard(locale):
    from test_assembly_drawing_customer_table import customer_table

    package = edited_preview(customer_table)
    drawing = generate_assembly_drawing(package, rebound_detail(package), locale=locale)
    text = " ".join(texts(svg_tree(drawing.files[SVG]))) + " ".join(s.text for s in drawing.result.assembly_steps)
    for word in ("breadboard", "board top", "board +X", "on one breadboard", "TABLE,", "(TABLE", "面包板", "TABLE，"):
        assert word not in text, word
    assert ("table top" in text) if locale == "en" else ("平台台面" in text)


def test_the_chinese_sheet_follows_the_house_spacing(tmp_path):
    from drawing_helpers import CUSTOMER_PREVIEW_ZIP, CUSTOMER_DETAIL_JSON
    from os_optics.assembly_drawing.i18n import tidy_zh

    assert tidy_zh("将 SRC放在 您的光学平台 上，以 平面反射镜 D25 建模） 立于") == "将 SRC 放在您的光学平台上，以平面反射镜 D25 建模）立于"
    drawing = generate_assembly_drawing(CUSTOMER_PREVIEW_ZIP, CUSTOMER_DETAIL_JSON, locale="zh-CN")
    sheet = list(texts(svg_tree(drawing.files[SVG]))) + [s.text for s in drawing.result.assembly_steps]
    assert [t for t in sheet if CJK_SPACE_CJK.search(t)] == []
    assert [t for t in sheet if LATIN_CJK.search(t)] == []


def customer_camera(scene: dict) -> None:
    """The customer's own camera (a detector) at the end of the reference beam, 30 mm past its last point."""

    (axis,) = scene["beam_reference"]["beam_axes"]
    end = list(axis["points_mm"][-1])
    position = [end[0] + 30.0, end[1], end[2]]
    axis["points_mm"].append(position)
    if axis.get("through") is not None:
        axis["through"].append("CAM")
    source = next(a for a in scene["beam_reference"]["design_anchors"] if a.get("customer_supplied"))
    scene["beam_reference"]["design_anchors"].append(
        {"instance_id": "CAM", "anchor_kind": "COMPONENT", "position_mm": position, "customer_supplied": True,
         "customer_supply": {"reason": "NOT_IN_CATALOG", "optical_role": "detector",
                             "orientation_wxyz": list(source["customer_supply"]["orientation_wxyz"])}})
    scene["beam_reference"]["design_anchors"].sort(key=lambda a: (a["anchor_kind"], a["instance_id"]))


def test_the_beam_order_lists_optics_and_names_the_customers_equipment_apart():
    package = edited_preview(customer_camera)
    drawing = generate_assembly_drawing(package, rebound_detail(package))
    items = texts(groups(svg_tree(drawing.files[SVG]), "alignment")[0])
    text = " ".join(items)
    assert "Place the optics in beam order along beam-1: L1 (B01)." in text and "L1 (B01), CAM" not in text
    assert "Set up your own detector CAM on your own support where the beam reaches it after focal_plane" in text


def test_an_order_opensci_assembles_says_so_above_the_steps():
    from drawing_helpers import preview

    release = DetailReleaseRefV1(basis="CONFIRMED_ORDER", order_number="OS-20261006-0009", delivery_mode="OPENSCI_ASSEMBLY")
    drawing = generate_assembly_drawing(PREVIEW_ZIP, rebound_detail(preview(), release=release))
    root = svg_tree(drawing.files[SVG])
    (note,) = groups(root, "opensci-assembly")
    assert "OpenSci assembles and aligns this system before delivery" in " ".join(texts(note))
    assert texts(groups(root, "alignment")[0])[2].startswith("OpenSci assembles the system and aligns its optics")
    assert drawing.result.binding.release.delivery_mode == "OPENSCI_ASSEMBLY"
    assert not groups(svg_tree(generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON).files[SVG]), "opensci-assembly")


def test_the_drawing_upload_names_its_locale():
    from os_optics.client_artifacts import drawing_artifact_upload

    for locale in ("en", "zh-CN"):
        upload = drawing_artifact_upload(generate_assembly_drawing(PREVIEW_ZIP, DETAIL_JSON, locale=locale))
        assert upload.manifest.locale == locale and upload.manifest_payload()["locale"] == locale
