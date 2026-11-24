"""Shell and JSON-API surfaces of the local assembly check: ``os-optics assembly validate|verify``,
``os-optics schema assembly-validation`` and ``os_optics.local.json_api.validate_assembly_package``."""

from __future__ import annotations

import json

import pytest

from assembly_helpers import (
    FIXTURE_ZIP,
    STACK,
    box,
    mutate,
    reference_files,
    reference_package,
    set_geometry,
    translate_parts,
    without_last_triangles,
)
from os_optics._vendor.shared_contracts.assembly_preview import package_archive_bytes
from os_optics.assembly import result_file_bytes, validate_assembly_layout
from os_optics.cli import main
from os_optics.local import json_api


@pytest.fixture(scope="module")
def reference_bytes() -> bytes:
    return result_file_bytes(validate_assembly_layout(FIXTURE_ZIP))


@pytest.fixture(scope="module")
def fail_archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("fail") / "off-board.assembly-preview.zip"
    path.write_bytes(mutate(lambda s, g: translate_parts(s, STACK, (400.0, 0.0, 0.0))).archive_bytes())
    return path


@pytest.fixture(scope="module")
def invalid_archive(tmp_path_factory):
    files = reference_files()
    glb = files["geometry/b03-post.glb"]
    files["geometry/b03-post.glb"] = glb[:-9] + bytes([glb[-9] ^ 0x01]) + glb[-8:]
    path = tmp_path_factory.mktemp("invalid") / "corrupt.assembly-preview.zip"
    path.write_bytes(package_archive_bytes(files))
    return path


def _stdout_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# ------------------------------------------------------------------------------------------------- validate


def test_validate_with_out_writes_the_exact_result_file(tmp_path, capsys, reference_bytes):
    out = tmp_path / "assembly-validation.json"
    assert main(["assembly", "validate", str(FIXTURE_ZIP), "--out", str(out)]) == 0
    assert out.read_bytes() == reference_bytes
    summary = _stdout_json(capsys)
    result = json.loads(reference_bytes)
    assert summary["status"] == "PASS" and summary["result_hash"] == result["result_hash"]
    assert summary["interference"] == "NO_INTERFERENCE"
    assert summary["package_content_hash"] == result["package"]["package_content_hash"]
    assert summary["issues"] == 1 and summary["out"] == str(out)
    assert list(summary["checks"]) == [c["check"] for c in result["checks"]]


def test_validate_without_out_prints_the_full_result(capsys, reference_bytes):
    assert main(["assembly", "validate", str(FIXTURE_ZIP)]) == 0
    assert capsys.readouterr().out.encode("utf-8") == reference_bytes


def test_validate_a_package_directory(tmp_path, capsys, reference_bytes):
    for path, data in reference_files().items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(data)
    assert main(["assembly", "validate", str(tmp_path)]) == 0
    assert capsys.readouterr().out.encode("utf-8") == reference_bytes


def test_validate_exit_code_is_the_status(tmp_path, capsys, fail_archive):
    assert main(["assembly", "validate", str(fail_archive)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    holder_box = box((-12.5, -20.4, 0.0), (50.0, 12.5, 75.0))
    unknown = tmp_path / "open-mesh.assembly-preview.zip"
    unknown.write_bytes(mutate(lambda s, g: set_geometry(s, g, "b04-post-holder-base", without_last_triangles(holder_box))).archive_bytes())
    assert main(["assembly", "validate", str(unknown)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "UNKNOWN"


def test_validate_an_invalid_package_is_a_structured_error(tmp_path, capsys, invalid_archive):
    out = tmp_path / "never-written.json"
    assert main(["assembly", "validate", str(invalid_archive), "--out", str(out)]) == 65
    doc = _stdout_json(capsys)
    assert doc["status"] == "ERROR" and doc["error"]["code"] == "ASSEMBLY_PACKAGE_INVALID"
    assert any("b03-post.glb" in p for p in doc["error"]["problems"])
    assert not out.exists()
    assert main(["assembly", "validate", str(tmp_path / "missing.zip")]) == 65
    assert _stdout_json(capsys)["error"]["code"] == "ASSEMBLY_PACKAGE_INVALID"


def test_validate_unwritable_out_is_a_usage_error(tmp_path, capsys):
    assert main(["assembly", "validate", str(FIXTURE_ZIP), "--out", str(tmp_path)]) == 64  # a directory
    assert "cannot write" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------- verify


def test_verify_valid_and_invalid(capsys, invalid_archive, fail_archive):
    assert main(["assembly", "verify", str(FIXTURE_ZIP)]) == 0
    doc = _stdout_json(capsys)
    assert doc["status"] == "VERIFIED" and doc["problems"] == []
    assert doc["package_content_hash"] == reference_package().content_hash
    assert main(["assembly", "verify", str(fail_archive)]) == 0  # verification says nothing about the layout
    capsys.readouterr()
    assert main(["assembly", "verify", str(invalid_archive)]) == 66
    doc = _stdout_json(capsys)
    assert doc["status"] == "INVALID" and doc["problems"]


def test_schema_assembly_validation(capsys):
    assert main(["schema", "assembly-validation"]) == 0
    schema = _stdout_json(capsys)
    assert {"checks", "issues", "parts", "summary", "result_hash", "status"} <= set(schema["properties"])


# ------------------------------------------------------------------------------------------------ json api


def test_json_api_accepts_archive_bytes_and_file_maps(reference_bytes):
    for source in (FIXTURE_ZIP.read_bytes(), memoryview(FIXTURE_ZIP.read_bytes()), reference_files()):
        doc = json_api.validate_assembly_package(source)
        assert doc["ok"] is True and doc["status"] == "PASS", type(source).__name__
        assert doc["result_file"] == reference_bytes
        assert doc["result"] == json.loads(reference_bytes)
        json.dumps({k: v for k, v in doc.items() if k != "result_file"})  # the rest is plain JSON


def test_json_api_reports_failures_as_results_not_errors(fail_archive):
    doc = json_api.validate_assembly_package(fail_archive.read_bytes())
    assert doc["ok"] is True and doc["status"] == "FAIL"
    assert any(i["code"] == "PART_OUTSIDE_BREADBOARD" for i in doc["result"]["issues"])


@pytest.mark.parametrize("garbage", [b"", b"not a zip at all", 42, None, ["manifest.json"], {"manifest.json": "text, not bytes"}],
                         ids=["empty", "junk-bytes", "int", "none", "list", "str-values"])
def test_json_api_garbage_is_package_invalid(garbage):
    doc = json_api.validate_assembly_package(garbage)
    assert doc["ok"] is False and doc["error"]["code"] == "ASSEMBLY_PACKAGE_INVALID"
    assert doc["error"]["problems"] and all(p["code"] == "ASSEMBLY_PACKAGE_INVALID" for p in doc["error"]["problems"])


def test_json_api_invalid_package_lists_problems(invalid_archive):
    doc = json_api.validate_assembly_package(invalid_archive.read_bytes())
    assert doc["ok"] is False and any("b03-post.glb" in p["message"] for p in doc["error"]["problems"])
    assert json_api.json_schema("assembly-validation")["ok"] is True


# ---- caller tolerances: the CLI's --tolerance NAME=VALUE and the JSON API's options.tolerances ----------------------------


def test_validate_takes_caller_tolerances_and_records_them(capsys, reference_bytes):
    """A threshold set by the caller is recorded in the result and changes nothing else for the reference; the work
    limit below the package's size makes the check UNKNOWN (exit 2), never out of model scope."""

    assert main(["assembly", "validate", str(FIXTURE_ZIP), "--tolerance", "board_outline_tolerance_mm=0.6"]) == 0
    result = json.loads(capsys.readouterr().out)
    reference = json.loads(reference_bytes)
    assert result["tolerances"] == {**reference["tolerances"], "board_outline_tolerance_mm": 0.6}
    for key in ("status", "checks", "issues", "parts", "summary"):
        assert result[key] == reference[key], key
    assert main(["assembly", "validate", str(FIXTURE_ZIP), "--tolerance", "max_total_triangles=100"]) == 2
    limited = json.loads(capsys.readouterr().out)
    assert limited["status"] == "UNKNOWN" and limited["tolerances"]["max_total_triangles"] == 100
    assert [i["code"] for i in limited["issues"] if i["check"] == "PART_GEOMETRY"] == ["GEOMETRY_BUDGET_EXCEEDED"]


@pytest.mark.parametrize("argument", ["no_such_tolerance=1", "max_total_triangles=1.5", "overlap_noise_mm", "overlap_noise_mm=x",
                                      "overlap_noise_mm=5"])
def test_a_bad_tolerance_is_a_usage_error(capsys, argument):
    """An unknown name, a value of the wrong kind and a value that breaks the model's ordering rule (contact noise above
    the collision penetration) are usage errors; nothing is checked."""

    with pytest.raises(SystemExit) as caught:
        code = main(["assembly", "validate", str(FIXTURE_ZIP), "--tolerance", argument])
        raise SystemExit(code)
    assert caught.value.code in (2, 64)  # argparse's own usage error for a malformed value, 64 for a refused one
    assert "--tolerance" in capsys.readouterr().err


def test_json_api_takes_caller_tolerances(reference_bytes):
    files = reference_files()
    reply = json_api.validate_assembly_package(files, {"tolerances": {"max_total_triangles": 100}})
    assert reply["ok"] is True and reply["status"] == "UNKNOWN" and reply["result"]["tolerances"]["max_total_triangles"] == 100
    same = json_api.validate_assembly_package(files, {"tolerances": {}})
    assert same["result_file"] == reference_bytes
    for options in ({"tolerances": {"no_such_tolerance": 1}}, {"tolerances": {"overlap_noise_mm": 5.0}}, {"other": 1}, ["tolerances"]):
        refused = json_api.validate_assembly_package(files, options)
        assert refused["ok"] is False and refused["error"]["code"] == "INPUT_INVALID", options
