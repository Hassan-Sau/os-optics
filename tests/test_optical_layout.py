"""The Optical Layout Drawing of an optics-only purchase (``os_optics.optical_layout``).

A synthetic folded layout (``layout_helpers``): the customer's own light source (the datum), two OpenSci lenses, an
OpenSci fold mirror, the customer's own window modelled as an OpenSci product, and an observation plane. Checked:
the outputs verify and are deterministic; the element table, spacing table, notes and title block state the package's
facts in both languages; nothing outside the package and the generator's templates is printed; the CLI, the JSON API
and the commercial client (against the commerce stub) render it.
"""

from __future__ import annotations

import json
import re

import pytest
from drawing_helpers import FORBIDDEN_TOKENS, SVG_NS, groups, svg_tree, texts
from layout_helpers import CUSTOMER, ORDER_NUMBER, RESOLUTION_ID, layout_document, layout_package

from os_optics._vendor.shared_contracts.optical_layout import seal_layout
from os_optics._vendor.shared_contracts.optical_layout.drawing import (
    LAYOUT_DRAWING_FILES,
    LAYOUT_MANIFEST_PATH,
    LAYOUT_PDF_PATH,
    LAYOUT_SVG_PATH,
    OpticalLayoutDrawingManifestV1,
)
import drawing_text_check as drawing_privacy
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.optical_layout import (
    OpticalLayoutError,
    OpticalLayoutErrorCode,
    generate_optical_layout,
    load_layout_package,
    verify_optical_layout_drawing,
)
from os_optics.optical_layout import messages
from os_optics.optical_layout.angles import atan2_deg, azimuth_elevation

CJK = re.compile(r"[一-鿿]")


@pytest.fixture(scope="module")
def en():
    return generate_optical_layout(layout_package(), locale="en")


@pytest.fixture(scope="module")
def zh():
    return generate_optical_layout(layout_package(), locale="zh-CN")


def table_rows(root) -> dict[str, str]:
    return {g.get("data-instance-id"): " ".join(texts(g)) for g in groups(root, "element-row")}


def test_the_outputs_verify_and_bind_the_package(en):
    package = layout_package()
    assert set(en.files) == set(LAYOUT_DRAWING_FILES)
    assert verify_optical_layout_drawing(en.files) == ()
    manifest = OpticalLayoutDrawingManifestV1.model_validate_json(en.files[LAYOUT_MANIFEST_PATH])
    assert manifest == en.manifest and manifest.locale == "en" and manifest.sheet.value == "A3_LANDSCAPE_420x297_MM"
    assert manifest.layout_content_hash == package.layout_content_hash and manifest.binding == package.binding
    assert manifest.generator.generator_id == "opensci.engine.optical_layout"
    assert en.files[LAYOUT_PDF_PATH].startswith(b"%PDF-1.4") and b"<svg" in en.files[LAYOUT_SVG_PATH][:400]
    node = next(el for el in svg_tree(en.files[LAYOUT_SVG_PATH]).iter(f"{SVG_NS}metadata"))
    assert node.get("id") == "opensci-optical-layout-metadata"
    meta = json.loads(node.text)
    assert meta["layout_content_hash"] == package.layout_content_hash and meta["order_number"] == ORDER_NUMBER


def test_generation_is_deterministic_for_every_source_form(en, tmp_path):
    document = json.loads(json.dumps(layout_package().model_dump(mode="json", by_alias=True)))
    path = tmp_path / "optical-layout.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    for source in (document, json.dumps(document).encode(), path):
        assert dict(generate_optical_layout(source).files) == dict(en.files)


def test_the_views_draw_every_element_beam_and_the_datum(en):
    root = svg_tree(en.files[LAYOUT_SVG_PATH])
    views = [g.get("data-view") for g in groups(root, "view")]
    assert views == ["PLAN", "ELEVATION_X"]
    for view in views:
        drawn = {g.get("data-instance-id"): g for g in groups(root, "element", view=view)}
        assert set(drawn) == {"SRC", "L1", "M1", "W1", "L2"}
        assert drawn["SRC"].get("data-outline") == "UNKNOWN" and drawn["L1"].get("data-outline") == "OUTLINE"
        assert drawn["W1"].get("data-supplied-by") == "CUSTOMER"
        (beam,) = groups(root, "beam-path", view=view)
        assert beam.get("data-path-key") == "beam-1" and beam.get("data-wavelength-nm") == "780"
        (label,) = groups(root, "beam-label", view=view)
        assert texts(label) == ["beam-1 780 nm"]
        assert groups(root, "datum", view=view)[0].get("data-instance-id") == "SRC"
    plan_labels = {g.get("data-instance-id"): texts(g) for g in groups(root, "element-label", view="PLAN")}
    assert plan_labels["SRC"] == ["SRC", "outline unknown", "DATUM SRC"]


def test_the_element_table_names_each_kind_of_element(en):
    rows = table_rows(svg_tree(en.files[LAYOUT_SVG_PATH]))
    assert rows["L1"].startswith("L1 Plano-convex lens D25.4 mm f 100 mm") and "OS-OPT-LENS-PCX-D25P4-F100 (BOM B01)" in rows["L1"]
    assert " 100 0 0 0 / 0 Tz manual" in rows["L1"]
    assert "135 / 0" in rows["M1"] and "Rx manual, Ry motorized" in rows["M1"]  # the fold mirror's normal, 0.1 deg
    # the customer's own window: never presented as the OpenSci product, only as the model the design used
    assert "your own window (modelled as Plane window D25.4 mm t 5 mm OS-OPT-WIN-PLANE-D25P4-T5)" in rows["W1"]
    assert "not delivered by OpenSci" in rows["W1"] and "BOM" not in rows["W1"]
    assert "your own light source (custom component laser-780)" in rows["SRC"]
    assert " 200 150 5 " in rows["L2"]  # its relative height is kept


def test_spacing_notes_and_title_block(en):
    root = svg_tree(en.files[LAYOUT_SVG_PATH])
    spacing = [(g.get("data-from"), g.get("data-to"), g.get("data-distance-mm")) for g in groups(root, "spacing-row")]
    assert spacing == [("SRC", "L1", "100"), ("L1", "M1", "100"), ("M1", "W1", "80"), ("W1", "L2", "70.2"),
                       ("L2", "focal_plane", "50")]
    notes = " ".join(texts(groups(root, "notes")[0]))
    assert "Positions are relative to SRC" in notes and "No height above a table or breadboard is implied" in notes
    assert "OpenSci supplies the optical elements on the Order only" in notes
    assert "Components supplied by you (SRC, W1) are not delivered by OpenSci" in notes
    assert "azimuth in the X-Y plane from +X, counter-clockwise seen from +Z" in notes
    (order,) = groups(root, "order-number")
    (customer,) = groups(root, "customer")
    assert texts(order)[-1] == ORDER_NUMBER and texts(customer)[-1] == CUSTOMER
    block = " ".join(texts(groups(root, "title-block")[0]))
    assert "OPTICAL LAYOUT DRAWING" in block and "optics-only purchase" in block


def test_a_declared_optic_is_drawn_with_the_geometry_its_project_declares():
    """A customer-supplied user-defined optic whose Project declares its optical model carries that model's outline: the
    views draw it as an outline (not the "outline unknown" symbol), its row names it as the customer's own component,
    and the outline note covers declared geometry as well as catalog products, in both locales."""

    document = layout_document()
    window = next(e for e in document["elements"] if e["instance_id"] == "W1")
    for key in ("opensci_sku", "display_name", "display_name_zh_cn"):
        window.pop(key)
    window.update(supply_reason="NOT_IN_CATALOG", custom_component_key="own-window",
                  outline={"shape": "ROUND", "diameter_mm": 25.4, "z_min_mm": 0.0, "z_max_mm": 4.0})
    package = seal_layout(document)
    sheets = {locale: generate_optical_layout(package, locale=locale) for locale in ("en", "zh-CN")}
    root = svg_tree(sheets["en"].files[LAYOUT_SVG_PATH])
    for view in ("PLAN", "ELEVATION_X"):
        drawn = {g.get("data-instance-id"): g for g in groups(root, "element", view=view)}
        assert drawn["W1"].get("data-outline") == "OUTLINE" and drawn["W1"].get("data-supplied-by") == "CUSTOMER"
    plan_labels = {g.get("data-instance-id"): texts(g) for g in groups(root, "element-label", view="PLAN")}
    assert "outline unknown" not in plan_labels["W1"]
    row = table_rows(root)["W1"]
    assert "own-window" in row and "BOM" not in row and "OS-OPT" not in row
    notes = " ".join(texts(groups(root, "notes")[0]))
    assert "or the geometry your Project declares for your own optic" in notes
    zh_notes = " ".join(texts(groups(svg_tree(sheets["zh-CN"].files[LAYOUT_SVG_PATH]), "notes")[0]))
    assert "声明的几何外形" in zh_notes


def test_a_layout_before_the_order_says_so_and_keeps_the_resolution_as_data():
    drawing = generate_optical_layout(layout_package(ordered=False))
    root = svg_tree(drawing.files[LAYOUT_SVG_PATH])
    assert not groups(root, "order-number")
    (state,) = groups(root, "order-state")
    assert texts(state) == ["ORDER NUMBER", "Not ordered"]
    block = " ".join(texts(groups(root, "title-block")[0]))
    assert RESOLUTION_ID not in block and "REFERENCES" not in block and "GENERATOR" not in block
    assert drawing.manifest.binding.commercial_resolution_id == RESOLUTION_ID
    # the source identities stay in the SVG metadata
    binding = layout_package(ordered=False).binding
    meta = json.loads(next(el for el in root.iter(f"{SVG_NS}metadata")).text)
    assert (meta["commercial_resolution_id"], meta["project_hash"]) == (RESOLUTION_ID, binding.project_hash)
    assert (meta["local_validation_hash"], meta["public_catalog_version"]) == (binding.local_validation_hash,
                                                                            binding.public_catalog_version)


def test_the_zh_cn_sheet(zh, en):
    root = svg_tree(zh.files[LAYOUT_SVG_PATH])
    assert zh.manifest.locale == "zh-CN" and root.get("{http://www.w3.org/XML/1998/namespace}lang") == "zh-CN"
    assert texts(groups(root, "notice-banner")[0])[0] == "光学布局图"
    rows = {k: "".join(v.split()) for k, v in table_rows(root).items()}  # wrapped lines joined, spaces aside
    assert "平凸透镜D25.4mmf100mmOS-OPT-LENS-PCX-D25P4-F100（清单行B01）" in rows["L1"]
    assert "您自备的窗口片（设计中以平面窗口片D25.4mm厚5mmOS-OPT-WIN-PLANE-D25P4-T5建模）" in rows["W1"]
    assert b"/BaseFont /STSong-Light" in zh.files[LAYOUT_PDF_PATH]
    decoded = [t for _, t in drawing_privacy.pdf_strings(LAYOUT_PDF_PATH, zh.files[LAYOUT_PDF_PATH]) if CJK.search(t)]
    assert "光学布局图" in decoded and CUSTOMER in decoded
    # the same facts in both languages
    assert [g.get("data-distance-mm") for g in groups(root, "spacing-row")] == [
        g.get("data-distance-mm") for g in groups(svg_tree(en.files[LAYOUT_SVG_PATH]), "spacing-row")]


def test_both_locales_share_keys_and_placeholders():
    from os_optics.assembly_drawing.i18n import named_fields

    assert set(messages.EN) == set(messages.ZH)
    for key in messages.EN:
        assert named_fields(messages.EN[key]) == named_fields(messages.ZH[key]), key


def test_every_output_token_is_a_package_field_a_literal_or_a_number(en, zh, monkeypatch):
    for drawing in (en, zh):
        for name, data in drawing.files.items():
            text = data.decode("latin-1" if name == LAYOUT_PDF_PATH else "utf-8")
            assert not [t for t in FORBIDDEN_TOKENS if t.lower() in text.lower()], name
        assert drawing_privacy.layout_problems(drawing) == []  # the build-time gate (also run on every layout by conftest)
    monkeypatch.setitem(messages.EN, "title", messages.EN["title"] + " zqxjv")
    with drawing_privacy.unchecked():
        injected = generate_optical_layout(layout_package())
    problems = drawing_privacy.layout_problems(injected)
    assert any(p.startswith(f"{LAYOUT_SVG_PATH}: <title>") for p in problems)
    assert all("zqxjv" not in p for p in problems)
    assert not hasattr(OpticalLayoutErrorCode, "UNEXPECTED_TEXT_IN_OUTPUT")


def test_a_package_that_does_not_verify_is_refused():
    document = layout_package().model_dump(mode="json", by_alias=True)
    document["elements"][1]["position_mm"][0] += 1.0  # content no longer matches its hash
    for source, code in ((document, OpticalLayoutErrorCode.PACKAGE_INVALID),
                         ({**document, "schema_version": "2.0"}, OpticalLayoutErrorCode.UNSUPPORTED_VERSION),
                         (b"not json", OpticalLayoutErrorCode.PACKAGE_INVALID), (42, OpticalLayoutErrorCode.INPUT_INVALID)):
        with pytest.raises(OpticalLayoutError) as caught:
            load_layout_package(source)
        assert caught.value.code is code
    with pytest.raises(OpticalLayoutError) as caught:
        generate_optical_layout(layout_package(), locale="fr")
    assert caught.value.code is OpticalLayoutErrorCode.INPUT_INVALID


def test_a_layout_along_y_gets_the_y_z_elevation():
    document = layout_document()
    for e in document["elements"]:
        x, y, z = e["position_mm"]
        e["position_mm"] = [x * 0.1, y * 3.0, z]
    for path in document["beam_paths"]:
        path["points_mm"] = [[x * 0.1, y * 3.0, z] for x, y, z in path["points_mm"]]
        pts = path["points_mm"]
        path["segment_lengths_mm"] = [round(sum((b[k] - a[k]) ** 2 for k in range(3)) ** 0.5, 3) for a, b in zip(pts, pts[1:])]
    for obs in document["observations"]:
        x, y, z = obs["position_mm"]
        obs["position_mm"] = [x * 0.1, y * 3.0, z]
    drawing = generate_optical_layout(seal_layout(document))
    assert [g.get("data-view") for g in groups(svg_tree(drawing.files[LAYOUT_SVG_PATH]), "view")] == ["PLAN", "ELEVATION_Y"]


def test_azimuth_and_elevation_are_deterministic_and_exact_on_the_axes():
    assert atan2_deg(0.0, 1.0) == 0.0 and atan2_deg(1.0, 0.0) == 90.0 and atan2_deg(0.0, -1.0) == 180.0
    assert atan2_deg(-1.0, 0.0) == -90.0 and atan2_deg(1.0, 1.0) == pytest.approx(45.0, abs=1e-12)
    import math

    for k in range(-179, 181, 7):
        a = math.radians(k)
        assert atan2_deg(math.sin(a), math.cos(a)) == pytest.approx(k, abs=1e-9)
    assert azimuth_elevation((0.0, 0.0, 1.0)) == (0.0, 90.0)
    assert azimuth_elevation((-0.7071068, 0.7071068, 0.0))[0] == pytest.approx(135.0, abs=1e-6)


def test_the_cli_draws_a_package_file(tmp_path, capsys):
    path = tmp_path / "optical-layout.json"
    path.write_text(json.dumps(layout_package().model_dump(mode="json", by_alias=True)), encoding="utf-8")
    out = tmp_path / "layout"
    assert cli_main(["layout", "draw", str(path), "--out", str(out), "--locale", "zh-CN"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "WRITTEN" and doc["locale"] == "zh-CN" and doc["order_number"] == ORDER_NUMBER
    assert sorted(p.name for p in out.iterdir()) == sorted(LAYOUT_DRAWING_FILES)
    assert verify_optical_layout_drawing(out) == ()
    (out / "notes.txt").write_bytes(b"mine")
    assert cli_main(["layout", "draw", str(path), "--out", str(out)]) == 0  # regenerated: its own files are replaced
    capsys.readouterr()
    assert cli_main(["layout", "draw", str(path), "--out", str(out), "--overwrite"]) == 0  # accepted, changes nothing
    capsys.readouterr()
    assert (out / "notes.txt").read_bytes() == b"mine" and verify_optical_layout_drawing(out) == ()
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli_main(["layout", "draw", str(bad), "--out", str(tmp_path / "x")]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNSUPPORTED_VERSION"


def test_the_json_api_renders_a_package():
    document = layout_package().model_dump(mode="json", by_alias=True)
    reply = json_api.generate_optical_layout(document, "zh-CN")
    assert reply["ok"] and reply["locale"] == "zh-CN" and set(reply["files"]) == set(LAYOUT_DRAWING_FILES)
    assert reply["files"] == dict(generate_optical_layout(document, locale="zh-CN").files)
    assert json_api.generate_optical_layout(document, "fr")["error"]["code"] == "INPUT_INVALID"
    assert json_api.generate_optical_layout({"schema": "x"})["error"]["code"] == "UNSUPPORTED_VERSION"


def test_the_client_fetches_the_layout_of_an_order_and_of_a_resolution(tmp_path, capsys, monkeypatch):
    from commerce_stub import AGENT_SCOPES, StubServer, StubState, credential_for

    from os_optics.client import OpenSciApiError, OpenSciClient

    state = StubState(resolution_id=RESOLUTION_ID)
    token = state.add_token("agent", AGENT_SCOPES)
    document = layout_package().model_dump(mode="json", by_alias=True)
    order_id = "00000000-0000-4000-8000-00000000c0de"
    state.optical_layouts[f"orders/{order_id}"] = document
    state.optical_layouts[f"resolutions/{RESOLUTION_ID}"] = layout_package(ordered=False).model_dump(mode="json", by_alias=True)
    with StubServer(state) as server:
        client = OpenSciClient(server.url, token)
        assert client.order_optical_layout(order_id) == document
        assert client.resolution_optical_layout(RESOLUTION_ID)["binding"].get("order_number") is None
        with pytest.raises(OpenSciApiError) as missing:
            client.order_optical_layout("00000000-0000-4000-8000-000000000000")
        assert missing.value.code == "OPTICAL_LAYOUT_NOT_AVAILABLE"
        token_file = tmp_path / "token"
        token_file.write_text(credential_for("agent"), encoding="utf-8")
        common = ["--api-url", server.url, "--token-file", str(token_file)]
        out = tmp_path / "order-layout"
        code = cli_main(["order", "layout", order_id, "--out", str(out), "--locale", "zh-CN", *common])
        printed = capsys.readouterr().out
        assert code == 0, printed
        doc = json.loads(printed)
        assert doc["order_id"] == order_id and doc["order_number"] == ORDER_NUMBER and doc["locale"] == "zh-CN"
        assert verify_optical_layout_drawing(out) == ()
        # the fetched package is saved next to the drawing: 'layout draw' renders exactly the same files from it
        saved = out / "optical-layout.json"
        assert doc["package_file"] == str(saved) and load_layout_package(saved) == layout_package()
        again = tmp_path / "again"
        assert cli_main(["layout", "draw", str(saved), "--out", str(again), "--locale", "zh-CN"]) == 0
        capsys.readouterr()
        assert all((again / n).read_bytes() == (out / n).read_bytes() for n in LAYOUT_DRAWING_FILES)
        assert cli_main(["order", "layout", order_id, "--out", str(out), "--locale", "zh-CN", *common]) == 0  # regenerated in place
        capsys.readouterr()
        assert all((again / n).read_bytes() == (out / n).read_bytes() for n in LAYOUT_DRAWING_FILES)
        assert cli_main(["resolution", "layout", RESOLUTION_ID, "--out", str(tmp_path / "res-layout"), *common]) == 0
        assert json.loads(capsys.readouterr().out)["resolution_id"] == RESOLUTION_ID


def test_the_resolution_layout_help_explains_the_fetch(capsys):
    with pytest.raises(SystemExit):
        cli_main(["resolution", "layout", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "saves it as optical-layout.json in --out" in text and "'draw PACKAGE_JSON'" not in text
