"""Order options: the engine client and command line against the local Commercial API stand-in.

* ``resolve RUN_DIR`` takes the independent options ``--mechanics`` / ``--platform`` / ``--assembly`` (and ``--mounting
  FILE`` for an optical-system design); today's ``--optics-only`` / ``--no-breadboard`` / ``--assembly-service`` are the
  same options as presets and send exactly the bodies they always sent; combinations that cannot be supplied are
  refused before anything is sent, with the service's code names; there is still no beam-height option;
* ``resolve-parts PARTS.json`` prices a part list (idempotent; ``--mechanics`` replaces the file's choice);
* ``products mechanical`` lists the mechanical part numbers; ``resolution|order mounting-sheet`` read the mounting
  sheet (JSON, or text in English / Simplified Chinese);
* the recovery of the new refusals (``PART_NOT_ORDERABLE``, ``MOUNTING_INPUT_REQUIRED``, ``MOUNTING_NOT_AVAILABLE``,
  ``ORDER_OPTION_ON_HOLD``) is kept or derived, with a hint that says what to do.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from commerce_stub import StubServer
from os_optics.cli import main
from os_optics.client import (
    LocalInputError,
    OpenSciApiError,
    OpenSciClient,
    Recovery,
    canonical_json_bytes,
    design_order_constraints,
    idempotency_key_for,
    part_list_document,
)
from os_optics.client.errors import derive_recovery, error_from_response
from os_optics.mounting_sheet import render_mounting_sheet
from test_commerce_client import AGENT, RESOLUTION_ID, new_state

ORDER_ID = "00000000-0000-4000-8000-000000000001"
PART_LIST = {"lines": [{"line_id": "L1", "opensci_sku": "OS-OPT-TEST", "quantity": 2},
                       {"line_id": "M1", "opensci_sku": "OS-MECH-BB-300", "quantity": 1}],
             "mechanical_supply": "DERIVED",
             "mounting": {"L1": {"beam_height_mm": 125, "retained_dofs": [], "incidence_deg": 0}}}
#: Real mounting sheets the OpenSci service answered (surfaces review, a local instance of the integrated branch):
#: a part list with derived mounts (lens x2, mirror with Rx/Ry, plate splitter; read for the resolution) and an
#: optical-system design with mounts per optic (read for its Order).
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "order_options"
SHEET = json.loads((FIXTURES / "mounting_sheet.part_list_derived.resolution.json").read_text(encoding="utf-8"))
DESIGN_SHEET = json.loads((FIXTURES / "mounting_sheet.optical_system.order.json").read_text(encoding="utf-8"))


@pytest.fixture()
def stub():
    with StubServer(new_state()) as server:
        yield server


@pytest.fixture()
def env(stub, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    for name in ("OPENSCI_TOKEN", "OPENSCI_TOKEN_FILE", "OPENSCI_API_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("OPENSCI_API_URL", stub.url)
    monkeypatch.setenv("OPENSCI_TOKEN", AGENT)
    return home


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from os_optics.local import example_project, run_optical_project

    directory = tmp_path_factory.mktemp("options") / "run"
    run_optical_project(example_project(), output_dir=directory)
    return directory


def run_cli(capsys, *argv: str) -> tuple[int, dict, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    try:
        doc = json.loads(captured.out) if captured.out.strip() else {}
    except ValueError:
        doc = {}
    return code, doc, captured.out + captured.err


def sent(stub) -> dict:
    return json.loads(stub.state.requests[-1]["body"])


# ------------------------------------------------------------------------------------------ design orders


def test_the_option_flags_send_the_preset_bodies(env, capsys, stub, run_dir):
    """Each combination of the independent options is sent as the preset keys (byte-identical bodies)."""

    cases = [
        ((), None),
        (("--mechanics", "derived", "--platform", "breadboard", "--assembly", "customer"), None),
        (("--mechanics", "none"), {"purchase_scope": "OPTICS_ONLY"}),
        (("--mechanics", "none", "--platform", "none", "--assembly", "customer"), {"purchase_scope": "OPTICS_ONLY"}),
        (("--platform", "none"), {"include_breadboard": False}),
        (("--mechanics", "derived", "--platform", "none"), {"include_breadboard": False}),
        (("--assembly", "opensci"), {"assembly_service": True}),
        (("--optics-only", "--mechanics", "none"), {"purchase_scope": "OPTICS_ONLY"}),
        (("--no-breadboard", "--platform", "none", "--mechanics", "derived"), {"include_breadboard": False}),
        (("--assembly-service", "--platform", "breadboard"), {"assembly_service": True}),
    ]
    for argv, constraints in cases:
        code, doc, out = run_cli(capsys, "resolve", str(run_dir), *argv)
        assert code == 0, (argv, out)
        body = sent(stub)
        assert body.get("customer_constraints") == constraints, argv
    # the preset flags and the options give the same request bytes
    bodies = {}
    for argv in (("--optics-only",), ("--mechanics", "none")):
        run_cli(capsys, "resolve", str(run_dir), *argv)
        bodies[argv] = stub.state.requests[-1]["body"]
    assert len(set(bodies.values())) == 1


@pytest.mark.parametrize(("argv", "code", "options"), [
    (("--mechanics", "none", "--platform", "breadboard"), "PURCHASE_OPTIONS_CONFLICT",
     ["--mechanics none", "--platform breadboard"]),
    (("--mechanics", "none", "--assembly", "opensci"), "PURCHASE_OPTIONS_CONFLICT",
     ["--mechanics none", "--assembly opensci"]),
    (("--optics-only", "--assembly", "opensci"), "PURCHASE_OPTIONS_CONFLICT", ["--optics-only", "--assembly opensci"]),
    (("--platform", "none", "--assembly", "opensci"), "ASSEMBLY_SERVICE_REQUIRES_BREADBOARD",
     ["--platform none", "--assembly opensci"]),
    (("--no-breadboard", "--assembly", "opensci"), "ASSEMBLY_SERVICE_REQUIRES_BREADBOARD",
     ["--no-breadboard", "--assembly opensci"]),
    (("--optics-only", "--mechanics", "derived"), "PURCHASE_OPTIONS_CONFLICT", ["--optics-only", "--mechanics derived"]),
    (("--no-breadboard", "--platform", "breadboard"), "PURCHASE_OPTIONS_CONFLICT",
     ["--no-breadboard", "--platform breadboard"]),
    (("--assembly-service", "--assembly", "customer"), "PURCHASE_OPTIONS_CONFLICT",
     ["--assembly-service", "--assembly customer"]),
    # three options: every flag given for a conflicting field, in the rule's order (two presets cannot be given
    # together: the parser refuses them)
    (("--mechanics", "none", "--platform", "breadboard", "--assembly", "opensci"), "PURCHASE_OPTIONS_CONFLICT",
     ["--mechanics none", "--platform breadboard", "--assembly opensci"]),
    (("--assembly-service", "--platform", "none"), "ASSEMBLY_SERVICE_REQUIRES_BREADBOARD",
     ["--platform none", "--assembly-service"]),
    # a board option beside a refused combination: the refusal names the conflicting options only (board options without
    # OpenSci mechanics are left out by the service, not refused)
    (("--mechanics", "none", "--platform", "breadboard", "--max-board-size-mm", "300", "300"), "PURCHASE_OPTIONS_CONFLICT",
     ["--mechanics none", "--platform breadboard"]),
])
def test_conflicting_options_are_refused_before_anything_is_sent(env, capsys, stub, run_dir, argv, code, options):
    status, doc, _ = run_cli(capsys, "resolve", str(run_dir), *argv)
    error = doc["error"]
    assert status == 64 and (error["code"], error["recovery"], error["family"]) == (code, "FIX_REQUEST", "LOCAL_INPUT")
    assert error["details"]["conflicting_options"] == options
    assert stub.state.requests == []


@pytest.mark.parametrize("argv", [("--mechanics", "none", "--max-board-size-mm", "300", "300"),
                                  ("--optics-only", "--require-all-retained-dofs")])
def test_options_that_do_not_apply_to_optics_only_are_sent_and_left_out_by_the_service(env, capsys, stub, run_dir, argv):
    """A board-size limit and the retained-DOF requirement have no effect on an optics-only purchase: the client sends
    them, the service leaves them out and lists them in options_not_applied."""

    code, _, out = run_cli(capsys, "resolve", str(run_dir), *argv)
    assert code == 0, out
    assert sent(stub)["customer_constraints"]["purchase_scope"] == "OPTICS_ONLY"


def test_the_option_rule_is_the_clients_and_the_command_line_names_its_refusal_by_the_flags(env, capsys, stub, run_dir, monkeypatch):
    """Which options go together is stated once, in ``client.design_order_constraints``: the command line sends its
    refusal with the same code and recovery, the conflicting fields named by the flags given and the client's sentence
    after them. A rule the client adds is refused the same way without a second copy in the command line."""
    from os_optics import cli_commerce
    from os_optics.client import LocalInputError, design_order_constraints

    with pytest.raises(LocalInputError) as rule:
        design_order_constraints(mechanical_supply="NONE", platform="BREADBOARD")
    assert rule.value.details == {"conflicting_options": ["mechanical_supply", "platform"]}
    status, doc, _ = run_cli(capsys, "resolve", str(run_dir), "--optics-only", "--platform", "breadboard")
    error = doc["error"]
    assert status == 64 and (error["code"], error["recovery"]) == (rule.value.code, rule.value.recovery.value)
    assert error["details"]["conflicting_options"] == ["--optics-only", "--platform breadboard"]
    assert error["message"] == f"--optics-only, --platform breadboard cannot be combined: {rule.value.message}"
    assert "_design_options" in cli_commerce.__dict__ and "design_order_constraints" not in cli_commerce._design_options.__code__.co_names

    def stricter(**options):  # (a rule only the client states)
        if options["assembly"] == "OPENSCI":
            raise LocalInputError("PURCHASE_OPTIONS_CONFLICT", "assembly OPENSCI is not offered", details={"conflicting_options": ["assembly"]},
                                  recovery=rule.value.recovery)
        return design_order_constraints(**options)

    import os_optics.client as client_package

    monkeypatch.setattr(client_package, "design_order_constraints", stricter)
    status, doc, _ = run_cli(capsys, "resolve", str(run_dir), "--assembly-service")
    assert status == 64 and doc["error"]["details"]["conflicting_options"] == ["--assembly-service"]
    assert doc["error"]["message"] == "--assembly-service cannot be combined: assembly OPENSCI is not offered"
    assert stub.state.requests == []


def test_an_optical_system_design_takes_its_beam_heights_from_a_mounting_file(env, capsys, stub, run_dir, tmp_path):
    mounting = {"L1": {"beam_height_mm": 125}, "M1": {"beam_height_mm": 125, "incidence_deg": 45}}
    path = tmp_path / "mounting.json"
    path.write_text(json.dumps(mounting), encoding="utf-8")
    for argv in (("--mounting", str(path)), ("--mechanics", "derived", "--platform", "none", "--mounting", str(path))):
        code, _, out = run_cli(capsys, "resolve", str(run_dir), *argv)
        assert code == 0, out
        assert sent(stub)["customer_constraints"] == {"include_breadboard": False, "mounting": mounting}
    count = len(stub.state.requests)
    for argv in (("--platform", "breadboard"), ("--optics-only",), ("--assembly", "opensci"), ("--assembly-service",)):
        code, doc, _ = run_cli(capsys, "resolve", str(run_dir), "--mounting", str(path), *argv)
        assert code == 64 and doc["error"]["code"] == "PURCHASE_OPTIONS_CONFLICT"
        assert doc["error"]["details"]["conflicting_options"][0] == "--mounting"
    for content in ("[]", "{}", '{"L1": 125}', "not json"):
        path.write_text(content, encoding="utf-8")
        code, _, out = run_cli(capsys, "resolve", str(run_dir), "--mounting", str(path))
        assert code == 64 and "--mounting" in out
    assert len(stub.state.requests) == count


def test_there_is_still_no_beam_height_option(capsys, run_dir):
    for command in (["resolve", str(run_dir)], ["resolve-parts", "parts.json"]):
        with pytest.raises(SystemExit) as refused:
            main([*command, "--beam-height-mm", "100"])
        assert refused.value.code == 64
    capsys.readouterr()


def test_design_order_constraints_maps_the_options_onto_the_presets():
    assert design_order_constraints() == {}
    assert design_order_constraints(mechanical_supply="DERIVED", platform="BREADBOARD", assembly="CUSTOMER") == {}
    assert design_order_constraints(mechanical_supply="NONE", platform="NONE") == {"purchase_scope": "OPTICS_ONLY"}
    assert design_order_constraints(platform="NONE") == {"include_breadboard": False}
    assert design_order_constraints(assembly="OPENSCI") == {"assembly_service": True}
    with pytest.raises(LocalInputError) as refused:
        design_order_constraints(platform="NONE", assembly="OPENSCI")
    assert refused.value.code == "ASSEMBLY_SERVICE_REQUIRES_BREADBOARD"
    with pytest.raises(ValueError):
        design_order_constraints(platform="INTEGRATED_BASE")


# ------------------------------------------------------------------------------------------ part lists


def test_a_part_list_is_priced_idempotently(env, capsys, stub, tmp_path):
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps(PART_LIST), encoding="utf-8")
    code, doc, out = run_cli(capsys, "resolve-parts", str(parts))
    assert code == 0, out
    assert doc["replayed"] is False and doc["resolution"]["order_options"]["basis"] == "PART_LIST"
    request = stub.state.requests[-1]
    body = json.loads(request["body"])
    assert request["path"] == "/api/v1/part-list-submissions"
    assert body == {"schema": "opensci.commercial.part_list_submission", "schema_version": "1", **PART_LIST}
    key = next(v for k, v in request["headers"].items() if k.lower() == "idempotency-key")
    assert key == idempotency_key_for("part-list-submit", canonical_json_bytes(body))
    code, again, _ = run_cli(capsys, "resolve-parts", str(parts))
    assert code == 0 and again["replayed"] is True and stub.state.part_list_submissions == 1
    # --mechanics none prices the listed parts alone: the mounting entries are left out
    code, _, _ = run_cli(capsys, "resolve-parts", str(parts), "--mechanics", "none")
    assert code == 0 and sent(stub) == {"schema": "opensci.commercial.part_list_submission", "schema_version": "1",
                                        "lines": PART_LIST["lines"], "mechanical_supply": "NONE"}


def test_part_list_refusals_name_the_lines(env, capsys, stub, tmp_path):
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"lines": [{"line_id": "X1", "opensci_sku": "OS-MECH-UNKNOWN", "quantity": 1}]}),
                     encoding="utf-8")
    code, doc, _ = run_cli(capsys, "resolve-parts", str(parts))
    error = doc["error"]
    assert code == 67 and (error["code"], error["recovery"]) == ("PART_NOT_ORDERABLE", "LIST_MECHANICAL_PRODUCTS")
    assert error["details"] == {"line_ids": ["X1"]} and "os-optics products mechanical" in error["hint"]
    assert sent(stub)["mechanical_supply"] == "NONE"  # the default when the file names none
    parts.write_text(json.dumps({"lines": PART_LIST["lines"], "mechanical_supply": "DERIVED"}), encoding="utf-8")
    code, doc, _ = run_cli(capsys, "resolve-parts", str(parts))
    assert code == 67 and doc["error"]["code"] == "MOUNTING_INPUT_REQUIRED" and doc["error"]["details"] == {"line_ids": ["L1"]}
    assert "beam_height_mm" in doc["error"]["hint"]
    count = len(stub.state.requests)
    for content, expected in (("[]", "PART_LIST_INVALID"), ('{"lines": [], "extra": 1}', "PART_LIST_INVALID")):
        parts.write_text(content, encoding="utf-8")
        code, doc, _ = run_cli(capsys, "resolve-parts", str(parts))
        assert code == 65 and doc["error"]["code"] == expected and doc["error"]["recovery"] == "FIX_REQUEST"
    parts.write_text("{", encoding="utf-8")
    code, _, out = run_cli(capsys, "resolve-parts", str(parts))
    assert code == 64 and "is not JSON" in out
    assert len(stub.state.requests) == count


def test_a_part_list_states_the_other_order_options(env, capsys, stub, tmp_path):
    """The part list may state optics_source / platform / assembly as the service takes them: sent as given, so an
    on-hold value reaches the service's ORDER_OPTION_ON_HOLD."""

    parts = tmp_path / "parts.json"
    stated = {**PART_LIST, "optics_source": "CATALOG", "platform": "NONE", "assembly": "CUSTOMER"}
    parts.write_text(json.dumps(stated), encoding="utf-8")
    code, _, out = run_cli(capsys, "resolve-parts", str(parts))
    assert code == 0, out
    assert sent(stub) == {"schema": "opensci.commercial.part_list_submission", "schema_version": "1", **stated}
    for field, value in (("optics_source", "CUSTOM"), ("platform", "INTEGRATED_BASE")):
        parts.write_text(json.dumps({**PART_LIST, field: value}), encoding="utf-8")
        code, doc, _ = run_cli(capsys, "resolve-parts", str(parts))
        error = doc["error"]
        assert code == 67 and (error["code"], error["recovery"]) == ("ORDER_OPTION_ON_HOLD", "FIX_REQUEST")
        assert sent(stub)[field] == value and "catalog optics" in error["hint"]


def test_part_list_document_keeps_the_given_schema_and_checks_the_override():
    body = part_list_document({**PART_LIST, "schema": "opensci.commercial.part_list_submission", "schema_version": "1"})
    assert body["mounting"] == PART_LIST["mounting"]
    assert part_list_document(PART_LIST, mechanical_supply="DERIVED")["mechanical_supply"] == "DERIVED"
    with pytest.raises(ValueError):
        part_list_document(PART_LIST, mechanical_supply="ALL")


# ------------------------------------------------------------------------------------------ products and sheets


def test_the_mechanical_products_are_listed(env, capsys, stub):
    code, doc, _ = run_cli(capsys, "products", "mechanical", "--json")
    assert code == 0 and [p["kind"] for p in doc["products"]] == ["BREADBOARD", "MOUNT", "POST_HOLDER", "POST"]
    assert stub.state.requests[-1]["path"] == "/api/v1/mechanical-products" and stub.state.requests[-1]["query"] == ""
    code, doc, _ = run_cli(capsys, "products", "mechanical", "--kind", "MOUNT", "--json")
    assert code == 0 and [p["opensci_sku"] for p in doc["products"]] == ["OS-MECH-KM-1"]
    assert stub.state.requests[-1]["query"] == "kind=MOUNT"
    code, _, out = run_cli(capsys, "products", "mechanical")
    assert code == 0 and "OS-MECH-KM-1  MOUNT  Kinematic mount 25.4 mm  centre_height_mm=15" in out
    assert "adjustment_axes=Rx/Ry" in out and "4 products" in out
    code, _, out = run_cli(capsys, "products", "mechanical", "--kind", "POST", "--locale", "zh-CN")
    assert code == 0 and "立杆 40 mm" in out and "diameter_mm=12.7" in out
    with pytest.raises(SystemExit):
        main(["products", "mechanical", "--kind", "LENS"])
    capsys.readouterr()


def _section(text: str, heading: str) -> list[str]:
    """The bullet lines under ``heading`` in every unit of a rendered sheet."""

    out, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == heading:
            inside = True
        elif inside and stripped.startswith("- "):
            out.append(stripped[2:])
        elif inside:
            inside = False
    return out


def test_the_mounting_sheet_is_read_as_json_or_text(env, capsys, stub, tmp_path):
    stub.state.mounting_sheets[f"resolutions/{RESOLUTION_ID}"] = SHEET
    stub.state.mounting_sheets[f"orders/{ORDER_ID}"] = DESIGN_SHEET
    code, doc, _ = run_cli(capsys, "resolution", "mounting-sheet", RESOLUTION_ID)
    assert code == 0 and doc == SHEET
    code, _, out = run_cli(capsys, "resolution", "mounting-sheet", RESOLUTION_ID, "--locale", "en")
    assert code == 0
    for phrase in ("Mounting sheet\n==============\n", "1. Optic: OS-OPT-LENS-PCX-D25P4-F100-T3P6-R51P5-NBK7-AR400-1100-"
                   "W587P6 (Plano-convex lens, D25.4 mm, f = 100 mm, N-BK7, AR 400-1100 nm)  [line L1]\n   Sets: 2 sets",
                   "Mount: OS-MECH-LENS-MOUNT-FIXED-D25 (Fixed lens mount for 25.4 mm optics) × 1 (2 in total)",
                   "Post holder: OS-MECH-POST-HOLDER-FORK-L75", "Angle of incidence: 0° (the default for this kind of "
                   "optic; state incidence_deg to change it)", "   Adjustments: none (fixed)",
                   "Holder 75 mm + post 75 mm + mount axis height 22 mm − insertion 47 mm = beam height 125 mm.",
                   "Rx: the mount's adjusters, -4 / +4 deg.", "Sets: 1 set\n"):
        assert phrase in out, phrase
    # only PASS checks are made; NOT_RUN ones are listed as not checked; the assumed values are stated
    made, not_made = _section(out, "Checks made:"), _section(out, "Not checked:")
    assumed_range = ("The post holder's insertion range is an assumed value, so the insertion depth is not checked "
                     "against a measured range.")
    assert "The post insertion is within the holder's insertion range." not in made + not_made
    assert assumed_range not in made
    assert "The part outlines rest on accepted facts." in made  # PASS for the lens and the splitter
    assert not_made.count(assumed_range) == 3
    assert not_made.count("Some part outlines are assumed values, so the clearance between the parts is checked against "
                          "assumed outlines.") == 1  # NOT_RUN for the mirror
    assert len(made) == 5 + 4 + 5 and len(not_made) == 2 + 3 + 2
    insertion, outline = ("The post holder's insertion range is an assumed value.",
                          "The mount's outline or axis height is an assumed value.")
    assert _section(out, "Assumed values:") == [insertion, insertion, outline, insertion]  # lens, mirror, splitter
    assert "MOUNT:" not in out and "POST_HOLDER" not in out and "TIP_TILT_MOUNT" not in out and "total:" not in out

    target = tmp_path / "sheet.txt"
    code, doc, _ = run_cli(capsys, "order", "mounting-sheet", ORDER_ID, "--locale", "zh-CN", "--out", str(target))
    assert code == 0 and doc == {"out": str(target), "format": "text", "locale": "zh-CN", "units": 1, "order_id": ORDER_ID}
    assert stub.state.requests[-1]["path"] == f"/api/v1/orders/{ORDER_ID}/mounting-sheet"
    text = target.read_text(encoding="utf-8")
    for phrase in ("安装说明\n========\n", "订单号: OS-ORD-20261014-000004", "[元件 L1]\n", "套数: 1 套", "镜架: OS-MECH-",
                   "支杆座: OS-MECH-", "Ty：在支杆座中上下滑动支杆，-10 / +15 mm。", "Tx：这组光机不提供该调节，装配时固定。",
                   "未做的检查:", "假定值:", "使用条件:", "沿光束方向的位置调节要求支杆座的槽沿光束方向固定。", "入射角: 0° (取自您的设计)",
                   "且在设计的入射角下不遮挡光束。"):
        assert phrase in text, phrase
    assert _section(text, "使用条件:") == ["沿光束方向的位置调节要求支杆座的槽沿光束方向固定。"]
    assumed_range = "支杆座允许的插入深度范围是假定值，因此插入深度未按实测范围检查。"
    assert assumed_range in _section(text, "未做的检查:") and assumed_range not in _section(text, "已做的检查:")
    assert "支杆插入深度在支杆座允许的范围内。" not in text
    for raw in ("MOUNT:", "POST_HOLDER", "BASE_SLOT_TRAVEL", "POST_HEIGHT_CLAMP", "Checks made", "Assumed values"):
        assert raw not in text, raw
    code, doc, _ = run_cli(capsys, "order", "mounting-sheet", ORDER_ID, "--out", str(target))
    assert code == 0 and doc["format"] == "json" and json.loads(target.read_text(encoding="utf-8")) == DESIGN_SHEET  # replaced
    code, doc, _ = run_cli(capsys, "order", "mounting-sheet", ORDER_ID, "--out", str(target), "--overwrite")
    assert code == 0 and json.loads(target.read_text(encoding="utf-8")) == DESIGN_SHEET  # the flag is accepted
    assert sorted(p.name for p in target.parent.iterdir() if p.name.endswith(".partial")) == []
    stub.state.mounting_sheets.clear()
    code, doc, _ = run_cli(capsys, "order", "mounting-sheet", ORDER_ID)
    assert code == 67 and doc["error"]["code"] == "MOUNTING_SHEET_NOT_PART_OF_PURCHASE"
    assert "this purchase does not have one" in doc["error"]["hint"]


def test_the_renderer_takes_what_the_service_adds():
    """Tolerant of the service's later fields: a NOT_RUN check whose text states why, a ``reason`` field, a failed
    check, Chinese axis strings without per-axis text, unknown roles and fields."""

    sheet = copy.deepcopy(SHEET)
    unit = sheet["units"][1]
    unit["checks"][4] = {"code": "ENGAGEMENT_RANGE", "status": "NOT_RUN", "detail": {"x": 1},
                         "text": "The post holder's insertion range is an assumed value, so the insertion depth is not "
                                 "checked against a measured range.",
                         "text_zh_cn": "支杆座允许的插入深度范围是假定值，因此插入深度未按实测范围检查。"}
    unit["checks"].append({"code": "NEW_CHECK", "status": "NOT_RUN", "text": "A new check.", "text_zh_cn": "新检查。",
                           "reason": "No data.", "reason_zh_cn": "没有数据。"})
    unit["checks"].append({"code": "OTHER", "status": "FAIL", "text": "Another check.", "text_zh_cn": "另一项检查。"})
    unit["adjustments"] = []
    unit["adjustment_axes_zh_cn"] = ["Rx（镜架调节旋钮，-4 / +4 deg）"]
    unit["chain"].append({"role": "SPACER_RING", "opensci_sku": "OS-MECH-X", "display_name": "Ring", "quantity": 1})
    unit["future_field"] = {"anything": True}
    en, zh = render_mounting_sheet(sheet, "en"), render_mounting_sheet(sheet, "zh-CN")
    assert "The post holder's insertion range is an assumed value, so the insertion depth is not checked against a " \
           "measured range." in _section(en, "Not checked:")
    assert "A new check. (No data.)" in _section(en, "Not checked:") and "新检查。 (没有数据。)" in _section(zh, "未做的检查:")
    assert "Another check. (status: did not pass)" in _section(en, "Not checked:")
    assert "另一项检查。 (状态: 未通过)" in _section(zh, "未做的检查:")
    assert _section(zh, "调节:") == ["Rx（镜架调节旋钮，-4 / +4 deg）"]
    assert "Rx (TIP_TILT_MOUNT, -4 / +4 deg)" in _section(en, "Adjustments:")
    assert "Spacer ring: OS-MECH-X (Ring) × 1" in en


def test_the_text_of_a_sheet_falls_back_between_languages():
    sparse = {"units": [{"optic": {"opensci_sku": "OS-OPT-TEST", "display_name": "Lens"}, "adjustment_axes": [],
                         "steps": [{"text": {"en": "Insert", "zh-CN": "装入"}}, "Tighten"]}]}
    zh = render_mounting_sheet(sparse, "zh-CN")
    assert "OS-OPT-TEST (Lens)" in zh and "调节: 无（固定）" in zh and "1. 装入" in zh and "2. Tighten" in zh
    with pytest.raises(ValueError):
        render_mounting_sheet(sparse, "fr")


def test_the_client_methods_use_the_documented_routes(stub):
    client = OpenSciClient(stub.url, AGENT, allow_insecure_http=True)
    assert client.mechanical_products(kind="BREADBOARD")["products"][0]["opensci_sku"] == "OS-MECH-BB-300"
    resolution = client.submit_part_list(PART_LIST)
    assert resolution["resolution_id"] == RESOLUTION_ID and client.submit_part_list(PART_LIST)["replayed"] is True
    with pytest.raises(ValueError):
        client.mechanical_products(kind="LENS")
    with pytest.raises(OpenSciApiError) as refused:
        client.resolution_mounting_sheet(RESOLUTION_ID)
    assert refused.value.http_status == 409


# ------------------------------------------------------------------------------------------ recovery


@pytest.mark.parametrize(("code", "recovery", "phrase"), [
    ("PART_NOT_ORDERABLE", Recovery.LIST_MECHANICAL_PRODUCTS, "os-optics products mechanical"),
    ("MOUNTING_INPUT_REQUIRED", Recovery.FIX_REQUEST, "--mounting FILE"),
    ("MOUNTING_NOT_AVAILABLE", Recovery.FIX_REQUEST, "reachable range"),
    ("ORDER_OPTION_ON_HOLD", Recovery.FIX_REQUEST, "catalog optics"),
    ("DESIGN_INTENT_HAS_NO_LAB_ASSEMBLY", Recovery.CHOOSE_OPTICS_ONLY, "--platform none --mounting FILE"),
    ("ASSEMBLY_NOT_PART_OF_PURCHASE", Recovery.USE_OPTICAL_LAYOUT, "resolution layout"),
])
def test_the_new_refusals_have_their_recovery_and_hint(code, recovery, phrase):
    assert derive_recovery(422, code, {}) is recovery
    for stated in (recovery.value, None):  # stated by the service, or derived when it states none
        error = {"code": code, "family": "REJECTED", "message": code.lower(), "details": {"line_ids": ["L1"]}}
        if stated:
            error["recovery"] = stated
        parsed = error_from_response(422, json.dumps({"error": error, "request_id": "req-1"}).encode(), None)
        assert parsed.recovery is recovery and phrase in parsed.hint, code


COMPONENTS_SHEET = {"delivery_mode": "COMPONENTS", "deliverables": ["MOUNTING_SHEET"]}


@pytest.mark.parametrize(("code", "status", "details", "recovery", "present", "absent"), [
    # products delivered as components with mounts per optic: the mounting sheet, never the optical layout
    ("ASSEMBLY_NOT_PART_OF_PURCHASE", 409, COMPONENTS_SHEET, "FIX_REQUEST", "resolution mounting-sheet", "resolution layout"),
    ("ASSEMBLY_NOT_PART_OF_PURCHASE", 409, {"delivery_mode": "COMPONENTS"}, "FIX_REQUEST", "mounting-sheet",
     "resolution layout"),
    ("OPTICAL_LAYOUT_NOT_PART_OF_PURCHASE", 409, COMPONENTS_SHEET, "FIX_REQUEST", "mounting-sheet", "includes OpenSci"),
    # a part list without mechanics: nothing besides its lines
    ("OPTICAL_LAYOUT_NOT_PART_OF_PURCHASE", 409, {"delivery_mode": "COMPONENTS", "deliverables": []}, "FIX_REQUEST",
     "no deliverable besides its lines", "mechanics"),
    ("MOUNTING_SHEET_NOT_PART_OF_PURCHASE", 409, {"deliverable": "MOUNTING_SHEET", "deliverables": []}, "FIX_REQUEST",
     "no deliverable besides its lines", "mounting-sheet"),
    # a design bought without mechanics: its optical layout
    ("ASSEMBLY_NOT_PART_OF_PURCHASE", 409, {}, "USE_OPTICAL_LAYOUT", "resolution layout", "mounting-sheet"),
    ("MOUNTING_SHEET_NOT_PART_OF_PURCHASE", 409, {"deliverables": ["OPTICAL_LAYOUT_DRAWING"]}, "FIX_REQUEST",
     "Optical Layout Drawing", "assembly preview"),
    # a design with mechanics on a laboratory layout: its assembly preview and drawing
    ("OPTICAL_LAYOUT_NOT_PART_OF_PURCHASE", 409, {}, "FIX_REQUEST", "laboratory layout", "mounting-sheet"),
    ("MOUNTING_SHEET_NOT_PART_OF_PURCHASE", 409, {"deliverables": ["ASSEMBLY_PREVIEW", "DETAILED_ASSEMBLY_DRAWING"]},
     "FIX_REQUEST", "assembly drawing RESOLUTION_ID", "layout"),
])
def test_a_refused_deliverable_is_answered_by_the_kind_of_purchase(code, status, details, recovery, present, absent):
    """Review m2 / m3: the hint names what this kind of purchase delivers instead; a purchase delivered as components
    never gets USE_OPTICAL_LAYOUT (its optical layout is refused too), even when an older service states it."""

    states = ("FIX_REQUEST", "USE_OPTICAL_LAYOUT", None) if code == "ASSEMBLY_NOT_PART_OF_PURCHASE" else ("FIX_REQUEST",)
    for stated in states:
        error = {"code": code, "family": "CONFLICT", "message": code.lower(), "details": details}
        if stated:
            error["recovery"] = stated
        if stated == "FIX_REQUEST" and recovery == "USE_OPTICAL_LAYOUT":
            continue  # the service states FIX_REQUEST only for components
        parsed = error_from_response(status, json.dumps({"error": error, "request_id": "req-1"}).encode(), None)
        assert parsed.recovery.value == recovery, (stated, details)
        assert present in parsed.hint and absent not in parsed.hint, (parsed.hint, details)


def test_an_empty_fact_list_reads_none():
    from os_optics.cli_commerce import _fact_text

    assert (_fact_text([]), _fact_text(["Rx", "Ry"]), _fact_text(25.0)) == ("none", "Rx/Ry", "25")


# ------------------------------------------------------------------------------------------ re-verification fixes


def test_the_parts_of_each_set_are_counted_per_set():
    """A chain line's quantity is the line total: two sets of one mount each are 'x 1 (2 in total)', never 'x 2'."""

    sheet = {"units": [{"line_id": "L1", "quantity": 2, "chain": [
        {"role": "MOUNT", "opensci_sku": "OS-MECH-M", "display_name": "Mount A", "display_name_zh_cn": "镜架 A",
         "quantity": 2},
        {"role": "POST", "opensci_sku": "OS-MECH-P", "display_name": "Post B", "quantity": 3}]},
        {"line_id": "L2", "quantity": 1, "chain": [{"role": "MOUNT", "opensci_sku": "OS-MECH-M", "quantity": 1}]}]}
    en, zh = render_mounting_sheet(sheet, "en"), render_mounting_sheet(sheet, "zh-CN")
    assert _section(en, "Parts of each set:") == ["Mount: OS-MECH-M (Mount A) × 1 (2 in total)",
                                                  "Post: OS-MECH-P (Post B) 3 in total", "Mount: OS-MECH-M × 1"]
    assert _section(zh, "每套零件:")[0] == "镜架: OS-MECH-M (镜架 A) × 1（共 2 件）"
    assert "× 2" not in en and "× 2" not in zh


def test_an_option_on_hold_never_suggests_a_platform_a_part_list_refuses():
    error = {"code": "ORDER_OPTION_ON_HOLD", "family": "REJECTED", "message": "on hold",
             "details": {"option": "platform", "value": "INTEGRATED_BASE"}}
    parsed = error_from_response(422, json.dumps({"error": error, "request_id": "req-1"}).encode(), None)
    assert "BREADBOARD" not in parsed.hint and "catalog optics" in parsed.hint and "as a line" in parsed.hint


@pytest.mark.parametrize("code", ["OPTICAL_LAYOUT_NOT_PART_OF_PURCHASE", "MOUNTING_SHEET_NOT_PART_OF_PURCHASE",
                                  "ASSEMBLY_NOT_PART_OF_PURCHASE"])
def test_a_purchase_without_deliverables_is_told_so_without_a_dead_end(code):
    error = {"code": code, "family": "CONFLICT", "message": code.lower(), "recovery": "FIX_REQUEST",
             "details": {"delivery_mode": "COMPONENTS", "deliverables": []}}
    parsed = error_from_response(409, json.dumps({"error": error, "request_id": "req-1"}).encode(), None)
    assert parsed.hint == "This purchase has no deliverable besides its lines: the products ship as listed."
    sheet = {**error, "details": {"delivery_mode": "COMPONENTS", "deliverables": ["MOUNTING_SHEET"]}}
    parsed = error_from_response(409, json.dumps({"error": sheet, "request_id": "req-1"}).encode(), None)
    assert parsed.hint.endswith("Use that instead of the refused request.")
