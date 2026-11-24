"""Deterministic input and state errors end in structured refusals, never in a traceback, a wrong exit code or a false
fact (the Engine CLI / json_api items)."""

from __future__ import annotations

import contextlib
import io
import json
import shutil

import pytest

from os_optics.cli import USAGE_ERROR
from os_optics.cli import main as cli_main


def _known_loss_catalog():
    from local_helpers import catalog_document, with_lossless_bulk

    return with_lossless_bulk(catalog_document())


def _known_loss_fixture(path):
    file = path / "synthetic-catalog.json"
    file.write_text(json.dumps(_known_loss_catalog()), encoding="utf-8")
    return file


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli_main(list(argv))
        except SystemExit as exc:
            code = exc.code
    return int(code), out.getvalue(), err.getvalue()


# ---- A4: usage errors exit 64 (2 is UNKNOWN) -------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    (),
    ("--no-such-flag",),
    ("project", "example", "--kind", "nope"),
    ("export", "arrivals", "RUN", "--format", "xlsx"),
    ("schema", "nope"),
    ("resolve", "RUN", "--token", "oat_" + "a" * 32 + "_" + "b" * 64, "--no-such-flag"),
], ids=lambda argv: " ".join(argv)[:40] or "no-args")
def test_usage_errors_exit_with_the_documented_usage_code(argv: tuple[str, ...]) -> None:
    code, out, err = _cli(*argv)
    assert code == USAGE_ERROR == 64
    assert out == ""
    assert "error:" in err and "usage:" in err
    assert "a" * 32 not in err  # a token typed on the command line is never repeated


# ---- A10: CSV exports neutralize formula-looking text cells -----------------------------------------------------------


def test_csv_cells_that_read_as_formulas_are_neutralized_and_numbers_are_unchanged() -> None:
    from os_optics.results.export import _cell

    for text in ("=2+3", "+1", "-x", "@SUM(1+1)", "\tA", "\rB"):
        cell = _cell(text)
        assert cell.strip('"').startswith("'" + text[0]), (text, cell)
    assert _cell(["=a", "b"]).startswith("'=a")
    assert _cell(-1.5) == "-1.5" and _cell(-3) == "-3" and _cell([-1.0, 2.0]) == "-1.0;2.0"
    assert _cell("L1") == "L1" and _cell("a,b") == '"a,b"' and _cell(None) == "" and _cell(True) == "true"


def test_a_run_whose_ids_look_like_formulas_exports_them_neutralized() -> None:
    from local_helpers import PCX_F100, element, plane, project, run, source

    from os_optics.results import load_run

    document = project([source("@SUM(1+1)"), element("=2+3", "lens", PCX_F100)], [plane("+obs", (101.5, 0.0, 0.0))])
    csv = load_run(run(document), verify=False).export("arrivals", "csv").decode("utf-8")
    rows = csv.splitlines()[1:]
    assert rows
    for row in rows:
        for cell in row.split(","):
            assert not cell.strip('"').startswith(("=", "+", "@")), row
    assert "'@SUM(1+1)" in csv and "'+obs" in csv


# ---- A3: a damaged run directory is INCONSISTENT (exit 66), never a crash ---------------------------------------------

DEEP_ARRAY = b"[" * 100_000 + b"]" * 100_000


def _manifest_edit(change):
    def apply(directory):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_bytes())
        change(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")
    return apply


def _write(name: str, data: bytes):
    def apply(directory):
        (directory / name).write_bytes(data)
    return apply


TAMPERING = {
    "files_ints": _manifest_edit(lambda m: m.__setitem__("files", [1, 2])),
    "files_string": _manifest_edit(lambda m: m.__setitem__("files", "x")),
    "files_object": _manifest_edit(lambda m: m.__setitem__("files", {})),
    "ray_data_string": _manifest_edit(lambda m: m.__setitem__("ray_data", "x")),
    "optical_view_list": _manifest_edit(lambda m: m.__setitem__("optical_view", [])),
    "status_list": _manifest_edit(lambda m: m.__setitem__("status", [])),
    "project_string": _manifest_edit(lambda m: m.__setitem__("project", "x")),
    "manifest_deep": _write("manifest.json", DEEP_ARRAY),
    "ray_data_deep": _write("ray_data.json", DEEP_ARRAY),
    "project_deep": _write("project.json", b'{"a":' * 100_000 + b"1" + b"}" * 100_000),
}


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    from local_helpers import PCX_F100, element, plane, project, run, source

    directory = tmp_path_factory.mktemp("good") / "run"
    document = project([source(), element("L1", "lens", PCX_F100)], [plane("focal_plane", (101.5, 0.0, 0.0))])
    run(document, output_dir=directory)
    return directory


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_a_damaged_run_directory_is_inconsistent_and_never_crashes(case: str, good_run, tmp_path) -> None:
    from os_optics.local import json_api

    directory = tmp_path / case
    shutil.copytree(good_run, directory)
    TAMPERING[case](directory)

    code, out, err = _cli("project", "verify", str(directory))
    assert code == 66, (out, err)
    report = json.loads(out)
    assert report["status"] == "INCONSISTENT" and report["problems"] and "Traceback" not in err

    code, out, err = _cli("results", "describe", str(directory))
    assert "Traceback" not in err
    assert code in (0, 64) and isinstance(json.loads(out), dict)

    files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    submission = json_api.commercial_submission(files)
    assert submission["ok"] is False and submission["error"]["code"] == "RUN_DIRECTORY_INCONSISTENT"
    for query in ({"kind": "describe"}, {"kind": "describe", "verify": True}, {"kind": "monitors"}, {"kind": "paths"}):
        answer = json_api.run_results(files, query)
        assert isinstance(answer, dict) and "ok" in answer


# ---- A2: malformed JSON text is a structured refusal ------------------------------------------------------------------

DEEP_TEXT = "[" * 100_000 + "]" * 100_000
BIGINT_TEXT = '{"schema": "opensci.project", "n": ' + "9" * 5000 + "}"
MALFORMED = {"deep": DEEP_TEXT.encode(), "bigint": BIGINT_TEXT.encode(), "latin1": '{"a": "caf\xe9"}'.encode("latin-1")}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_project_files_are_refused_with_exit_65(case: str, tmp_path) -> None:
    path = tmp_path / f"{case}.json"
    path.write_bytes(MALFORMED[case])
    for argv in (("project", "run", str(path)), ("project", "update-catalog", str(path), "--out", str(tmp_path / "o.json"))):
        code, out, err = _cli(*argv)
        assert code == 65, (argv, out, err[-300:])
        assert json.loads(out)["problems"][0]["code"] == "PROJECT_INVALID" and "Traceback" not in err


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_validate_answers_malformed_requests_with_a_structured_fail(case: str, tmp_path) -> None:
    path = tmp_path / f"{case}.json"
    path.write_bytes(MALFORMED[case])
    code, out, err = _cli("validate", str(path))
    assert code == 1 and "Traceback" not in err
    result = json.loads(out)
    assert result["status"] == "FAIL" and result["issues"][0]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("argument", [DEEP_TEXT, BIGINT_TEXT, MALFORMED["latin1"], "abc\x00def"],
                         ids=["deep", "bigint", "latin1", "nul-path"])
def test_json_api_project_functions_answer_malformed_input_with_an_error_envelope(argument) -> None:
    from os_optics.local import json_api

    for call in (json_api.check_project, json_api.run_project, json_api.update_project_catalog):
        answer = call(argument)
        assert answer["ok"] is False, call.__name__
        assert answer["error"]["problems"][0]["code"] == "PROJECT_INVALID", (call.__name__, answer)


# ---- A1: numbers of an extreme magnitude are a structured refusal -----------------------------------------------------


def _focus(**change):
    from local_helpers import PCX_F100, element, plane, project, source

    document = project([source(), element("L1", "lens", PCX_F100)], [plane("focal_plane", (101.5, 0.0, 0.0))],
                       [{"target_id": "spot", "metric": "beam_radius_mm", "observation_id": "focal_plane", "maximum": 0.05,
                         "unit": "mm"}])
    for key, value in change.items():
        if key == "observation_x":
            document["observations"][0]["pose"]["position_mm"][0] = value
        elif key == "aperture":
            document["observations"][0]["aperture"]["radius_mm"] = value
        elif key == "waist":
            document["components"][0]["source_model"]["waist_radius_mm"] = value
        elif key == "lens_y":
            document["components"][1]["nominal_pose"]["position_mm"][1] = value
        elif key == "shift":
            for component in document["components"]:
                component["nominal_pose"]["position_mm"][0] += value
            document["observations"][0]["pose"]["position_mm"][0] += value
    return document


EXTREME = {"observation_1e300": {"observation_x": 1e300}, "observation_2e154": {"observation_x": 2e154},
           "waist_1e-300": {"waist": 1e-300}, "waist_1e-200": {"waist": 1e-200}, "aperture_1e300": {"aperture": 1e300},
           "lens_y_1e200": {"lens_y": 1e200}, "shift_all_1e300": {"shift": 1e300}}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("case", sorted(EXTREME))
def test_extreme_numbers_are_refused_with_exit_65_and_never_a_traceback(case: str, tmp_path) -> None:
    from local_helpers import FIXTURE_CATALOG

    from os_optics.local import json_api

    path = tmp_path / "project.json"
    path.write_text(json.dumps(_focus(**EXTREME[case])), encoding="utf-8")
    code, out, err = _cli("project", "run", str(path), "--catalog", str(FIXTURE_CATALOG), "--out-dir", str(tmp_path / "run"))
    assert code == 65, (out, err[-400:])
    problems = json.loads(out)["problems"]
    assert problems[0]["code"] in ("ENGINE_RUN_INCOMPLETE", "OPTICAL_VIEW_BINDING_FAILED") and "Traceback" not in err
    assert "NUMERICAL_FAILURE" in problems[0]["message"] or "too large" in problems[0]["message"]
    assert not (tmp_path / "run" / "manifest.json").exists()
    answer = json_api.run_project(_focus(**EXTREME[case]))
    assert answer["ok"] is False


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_validate_reports_extreme_numbers_as_a_numerical_failure() -> None:
    from local_helpers import load_test_catalog

    from os_optics.local import lower_project, load_project
    from os_optics.validation import validate

    request = lower_project(load_project(_focus(observation_x=1e300)), load_test_catalog()).request
    result = validate(request)
    assert result.status.value == "FAIL"
    assert [issue.code.value for issue in result.issues] == ["NUMERICAL_FAILURE"]


# ---- B1: identifiers with a control character are refused locally (the service answers 422) -------------------------


@pytest.mark.parametrize("text", ["L\x00x", "\x00", "L\x01x", "L\tx", "L\x1bx", "L\x7fx", "L\x85x", "L\x9f"],
                         ids=["nul", "nul-only", "soh", "tab", "esc", "del", "nel", "c1-last"])
def test_an_identifier_with_a_control_character_is_refused_with_exit_65(text: str, tmp_path) -> None:
    from local_helpers import FIXTURE_CATALOG

    from os_optics.local import json_api

    document = _focus()
    document["components"][1]["instance_id"] = text
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    code, out, err = _cli("project", "run", str(path), "--catalog", str(FIXTURE_CATALOG), "--out-dir", str(tmp_path / "run"))
    assert code == 65, (out, err[-400:])
    problems = json.loads(out)["problems"]
    assert problems[0]["code"] == "PROJECT_INVALID" and "Traceback" not in err
    assert not (tmp_path / "run" / "manifest.json").exists()
    answer = json_api.check_project(document)
    assert answer["ok"] is False and answer["error"]["problems"][0]["code"] == "PROJECT_INVALID"


def test_engine_identifiers_refuse_control_characters_and_keep_the_rest() -> None:
    from pydantic import TypeAdapter, ValidationError

    from os_optics._vendor.shared_contracts.common.identifiers import FrameRef, OpenSciId
    from os_optics.contracts.base import Id

    for adapter in (TypeAdapter(Id), TypeAdapter(OpenSciId)):
        for good in ("L1", "lens 1", "透镜", "L\xa0x"):
            adapter.validate_python(good)
        for bad in ("L\x00x", "L\nx", "L\x7f", "L\x85", " L", "a/b"):
            with pytest.raises(ValidationError):
                adapter.validate_python(bad)
    TypeAdapter(FrameRef).validate_python("L1/optical_axis")
    with pytest.raises(ValidationError):
        TypeAdapter(FrameRef).validate_python("L1/optical\x00axis")


# ---- A5: an attempt that does not complete leaves --out-dir as it found it; the kept run states whose run it is --------


def _run_files(out) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(out.iterdir()) if path.is_file()}


def _assert_earlier_run_kept(out, before: dict[str, bytes], project_hash: str) -> None:
    """After an attempt that did not complete, ``out`` is as it was: every file of the earlier run and the file the
    Engine does not write, byte for byte. The run verifies and describes itself as the run of the earlier Project
    (``project_hash``), which is what tells it from a result of the attempt."""

    assert _run_files(out) == before, sorted(path.name for path in out.iterdir())
    code, report, _ = _cli("project", "verify", str(out))
    assert code == 0 and json.loads(report) == {"run_dir": str(out), "status": "CONSISTENT", "problems": []}
    code, described, _ = _cli("results", "describe", str(out))
    assert code == 0 and json.loads(described)["run"]["project_hash"] == project_hash
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_refused_or_crashed_rerun_leaves_the_previous_run_as_it_was(tmp_path, monkeypatch) -> None:
    from local_helpers import FIXTURE_CATALOG

    from os_optics.local import load_project, run_freshness
    from os_optics.local import run as run_module

    out = tmp_path / "run"
    good, unknown, extreme = tmp_path / "good.json", tmp_path / "unknown.json", tmp_path / "extreme.json"
    good.write_text(json.dumps(_focus()), encoding="utf-8")
    document = _focus()
    document["components"][1]["product"]["optical_sku"] = "OS-OPT-TEST-NOPE"
    unknown.write_text(json.dumps(document), encoding="utf-8")
    extreme.write_text(json.dumps(_focus(observation_x=1e300)), encoding="utf-8")
    catalog = ("--catalog", str(FIXTURE_CATALOG))
    project_hash = load_project(_focus()).content_hash()

    assert _cli("project", "run", str(good), *catalog, "--out-dir", str(out))[0] == 0
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    before = _run_files(out)
    for refused in (unknown, extreme):
        assert _cli("project", "run", str(refused), *catalog, "--out-dir", str(out))[0] == 65
        _assert_earlier_run_kept(out, before, project_hash)
    # the kept run is not the refused Project's: another Project hash, and HISTORICAL for the edited design
    edited = run_freshness(json.loads(extreme.read_text(encoding="utf-8")), out, catalog=FIXTURE_CATALOG)
    assert edited["freshness"] != "CURRENT" and edited["run"]["project_hash"] == project_hash != edited["current"]["project_hash"]

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(run_module, "traced_evaluation", crash)
    # (a crash inside the command is an internal error: named, with its type, exit 70 -- not the 1 of an unmet target)
    code, printed, _ = _cli("project", "run", str(good), *catalog, "--out-dir", str(out))
    crashed = json.loads(printed)
    assert code == 70 and (crashed["status"], crashed["error"]["code"], crashed["error"]["type"]) == ("ERROR", "INTERNAL_ERROR", "RuntimeError")
    assert "simulated crash" in crashed["error"]["message"]
    _assert_earlier_run_kept(out, before, project_hash)
    with pytest.raises(RuntimeError):
        run_module.run_optical_project(_focus(), catalog=FIXTURE_CATALOG, output_dir=out)
    monkeypatch.undo()
    _assert_earlier_run_kept(out, before, project_hash)


def test_an_attempt_that_never_runs_leaves_the_previous_run_as_it_was(tmp_path) -> None:
    """The customer-Agent workflow: one run directory reused for every attempt. An attempt that ends before it runs --
    a missing or unreadable Project or catalog file, a Project that is not JSON, arguments refused as a usage error --
    exits non-zero and writes nothing: the directory still holds the earlier run, whole, which verifies, describes and
    submits as the run of the earlier Project. The attempt's exit code and reply state that it did not run."""

    from os_optics.client.api import submission_document
    from os_optics.local import load_project

    out = tmp_path / "run"
    good, garbled, folder = tmp_path / "good.json", tmp_path / "garbled.json", tmp_path / "a-folder.json"
    good.write_text(json.dumps(_focus()), encoding="utf-8")
    garbled.write_text("{\"schema_version\": ", encoding="utf-8")
    folder.mkdir()
    catalog = ("--catalog", str(_known_loss_fixture(tmp_path)))
    project_hash = load_project(_focus()).content_hash()
    attempts = {
        "missing Project file": (("project", "run", str(tmp_path / "missing.json"), *catalog, "--out-dir", str(out)), USAGE_ERROR),
        "missing catalog file": (("project", "run", str(good), "--catalog", str(tmp_path / "no-catalog.json"), "--out-dir", str(out)),
                                 USAGE_ERROR),
        "unreadable Project path": (("project", "run", str(folder), *catalog, "--out-dir", str(out)), USAGE_ERROR),
        "Project that is not JSON": (("project", "run", str(garbled), *catalog, "--out-dir", str(out)), 65),
        "refused argument": (("project", "run", str(good), *catalog, "--out-dir", str(out), "--pupil-rays", "SOME"), USAGE_ERROR),
        "refused argument, --out-dir=": (("project", "run", str(good), *catalog, f"--out-dir={out}", "--no-such-flag"), USAGE_ERROR),
    }
    assert _cli("project", "run", str(good), *catalog, "--out-dir", str(out))[0] == 0
    assert json.loads(_cli("results", "describe", str(out))[1])["status"]["simulation_status"] == "PASS"
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    before = _run_files(out)
    submitted = submission_document(out)
    for name, (argv, expected) in attempts.items():
        code, _, err = _cli(*argv)
        assert code == expected, (name, code, err)
        assert "Traceback" not in err, name
        _assert_earlier_run_kept(out, before, project_hash)
        assert submission_document(out) == submitted, name

    # a refused command line that names no run directory, or is not 'project run', touches nothing either
    assert _cli("project", "run", "--no-such-flag")[0] == USAGE_ERROR
    assert _cli("project", "verify", str(out), "--no-such-flag")[0] == USAGE_ERROR
    assert _cli("results", "describe", str(out), "--out-dir", str(out))[0] == USAGE_ERROR
    _assert_earlier_run_kept(out, before, project_hash)


def test_a_json_api_rerun_that_does_not_complete_leaves_the_earlier_run_in_its_output_dir(tmp_path, monkeypatch) -> None:
    from local_helpers import FIXTURE_CATALOG

    from os_optics.local import json_api, load_project

    out, good = tmp_path / "run", tmp_path / "good.json"
    good.write_text(json.dumps(_focus()), encoding="utf-8")
    assert _cli("project", "run", str(good), "--catalog", str(FIXTURE_CATALOG), "--out-dir", str(out))[0] == 0
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    before = _run_files(out)

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash outside the Engine's refusals")

    monkeypatch.setattr(json_api, "run_optical_project", crash)
    with pytest.raises(RuntimeError):
        json_api.run_project(_focus(), output_dir=str(out))
    monkeypatch.undo()
    _assert_earlier_run_kept(out, before, load_project(_focus()).content_hash())
    refused = json_api.run_project({"schema": "opensci.project"}, output_dir=str(out))
    assert refused["ok"] is False
    _assert_earlier_run_kept(out, before, load_project(_focus()).content_hash())


# ---- A9: the run summary keeps every field of an Engine issue ---------------------------------------------------------


def test_the_run_summary_issues_carry_location_numbers_and_suggested_action(tmp_path) -> None:
    from local_helpers import ALONG_X, FIXTURE_CATALOG, FLAT_MIRROR, element, plane, project, source

    from os_optics.local import json_api

    document = project([source(), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), ALONG_X)], [plane("p", (50.0, 0.0, 0.0))])
    path = tmp_path / "project.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    code, out, _ = _cli("project", "run", str(path), "--catalog", str(FIXTURE_CATALOG), "--out-dir", str(tmp_path / "run"))
    assert code == 0  # WARNING: the back side blocks the beam and the plane without targets is dark
    summary = json.loads(out)
    stored = json.loads((tmp_path / "run" / "simulation_result.json").read_bytes())["issues"]
    assert summary["issues"] == stored
    back = next(i for i in summary["issues"] if i["code"] == "BLOCKED_BY_BACK_SIDE")
    assert back["location"] == "surface:M1/S1" and back["suggested_action"] and back["numerical_context"]
    answer = json_api.run_project(document)
    assert [(i["code"], i["location"], i["suggested_action"], i["numerical_context"]) for i in answer["summary"]["issues"]] == [
        (i["code"], i["location"], i["suggested_action"], i["numerical_context"]) for i in stored]


# ---- A6: a source at or below the power floor is named, not "check the beam path" ------------------------------------


@pytest.mark.parametrize("power_w", [1e-15, 1e-16])
def test_a_source_below_the_power_floor_is_named_instead_of_not_reached(power_w: float) -> None:
    from local_helpers import PCX_F100, element, plane, project, run, source

    document = project([source(power_w=power_w), element("L1", "lens", PCX_F100)], [plane("focal_plane", (101.5, 0.0, 0.0))],
                       [{"target_id": "spot", "metric": "beam_radius_mm", "observation_id": "focal_plane", "maximum": 0.05,
                         "unit": "mm"}])
    result = run(document).result
    codes = [issue.code.value for issue in result.issues]
    assert "OBSERVATION_NOT_REACHED" not in codes and codes == ["ZERO_POWER"]
    floor = result.issues[0]
    assert floor.location == "observation:focal_plane"
    context = floor.numerical_context
    assert (context["power_floor_W"], context["source_power_W"], context["setting"]) == (1e-15, power_w, "power_epsilon_W")
    assert floor.suggested_action == f"set numerical_settings.power_epsilon_W below the source power (for example {power_w * 1e-9:.0e} W)"


def test_a_project_without_a_power_floor_traces_a_source_far_below_the_default_floor() -> None:
    """``numerical_settings.power_epsilon_W`` 0 sets no absolute floor: a 1e-30 W source is traced, and since the trace is
    linear in power its plane reads the 1 mW run's power scaled by 1e-27 (the independent reference). A Project that omits
    the setting keeps the default floor and its bytes."""
    from local_helpers import PCX_F100, element, plane, project, run, source

    def lens_project(power_w: float) -> dict:
        return project([source(power_w=power_w), element("L1", "lens", PCX_F100)], [plane("focal_plane", (101.5, 0.0, 0.0))])

    catalog = _known_loss_catalog()
    bright = run(lens_project(1e-3), catalog=catalog).result
    unfloored = run({**lens_project(1e-30), "numerical_settings": {"power_epsilon_W": 0}}, catalog=catalog).result
    assert unfloored.manifest.numerical_tolerances["power_epsilon_W"] == 0
    assert "ZERO_POWER" not in [issue.code.value for issue in unfloored.issues]
    (plane_result,) = unfloored.observations
    assert plane_result.total_power_W > 0.0
    assert plane_result.total_power_W == pytest.approx(bright.observations[0].total_power_W * 1e-27, rel=1e-9, abs=0.0)
    assert "numerical_settings" not in lens_project(1e-30)


def test_a_plane_missed_for_another_reason_is_still_not_reached() -> None:
    from local_helpers import PCX_F100, element, plane, project, run, source

    document = project([source(), element("L1", "lens", PCX_F100)], [plane("aside", (101.5, 500.0, 0.0))])
    # Preserve the single geometric refusal: the test fixture lens explicitly assumes zero bulk loss.
    assert [issue.code.value for issue in run(document, catalog=_known_loss_catalog()).result.issues] == ["OBSERVATION_NOT_REACHED"]


# ---- A8: the Project's quaternion tolerance is the Engine's -----------------------------------------------------------


def test_the_project_quaternion_tolerance_is_the_engines_and_six_decimal_quaternions_run() -> None:
    from local_helpers import FLAT_MIRROR, element, plane, project, run, source

    from os_optics._vendor.shared_contracts.optical_design.component import UNIT_QUATERNION_TOLERANCE
    from os_optics.config import NumericalConfig
    from os_optics.local import LocalRunError, load_project

    assert UNIT_QUATERNION_TOLERANCE == NumericalConfig().quaternion_norm_tol
    q = (0.707107, 0.0, 0.707107, 0.0)
    document = project([source(), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), q)], [plane("p", (50.0, 0.0, 0.0))])
    assert run(document).result.status is not None
    with pytest.raises(LocalRunError) as refused:
        load_project(project([source(), element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))], []))
    assert "within 1e-06" in str(refused.value) and "normalized: [0.5, 0.5, 0.5, 0.5]" in str(refused.value)


# ---- A11: a store whose active release is in an earlier catalog schema -----------------------------------------------

OLD_VERSION = "2026-10-04.1"


def _retired_store(root, monkeypatch):
    """A store as an os-optics 0.4.x left it: the active release is a schema 2.1 catalog this Engine cannot read
    (its content fails today's contract with thousands of errors), next to a current release."""

    import hashlib

    from catalog_store_helpers import ENV_CATALOG_DIR, ENV_CHECK_INTERVAL, seed_store
    from local_helpers import FIXTURE_CATALOG

    seed_store(root, FIXTURE_CATALOG, active=FIXTURE_CATALOG)
    old = json.loads(FIXTURE_CATALOG.read_bytes())
    old.update(catalog_version=OLD_VERSION, schema_version="2.1")
    for product in old["products"]:  # the earlier format: today's contract refuses every product
        product["legacy_field"] = "x" * 50
    data = json.dumps(old).encode("utf-8")
    (root / "releases" / f"{OLD_VERSION}.json").write_bytes(data)
    state = json.loads((root / "state.json").read_bytes())
    current = state["active"]
    state["releases"][OLD_VERSION] = dict(state["releases"][current], sha256=hashlib.sha256(data).hexdigest(),
                                          size_bytes=len(data), schema_version="2.1")
    state["active"] = OLD_VERSION
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv(ENV_CATALOG_DIR, str(root))
    monkeypatch.setenv(ENV_CHECK_INTERVAL, "0")
    return current


def test_doctor_status_and_use_on_a_store_whose_active_release_is_in_an_earlier_schema(tmp_path, monkeypatch) -> None:
    current = _retired_store(tmp_path / "store", monkeypatch)

    code, out, _ = _cli("doctor")
    # (the catalog is its own readiness fact: the interpreter can run a Project, exit 0; the catalog work cannot start)
    assert code == 0 and len(out.encode("utf-8")) < 16_000
    report = json.loads(out)
    assert (report["ready"]["local_design"], report["ready"]["catalog"]) == (True, False)
    assert report["catalog"]["status"] == "EARLIER_FORMAT" and "earlier format" in report["catalog"]["error"]
    step = next(s for s in report["next_steps"] if s["check"] == "catalog")
    assert step["commands"][0] == "os-optics catalog sync"

    code, out, _ = _cli("catalog", "status")
    status = json.loads(out)
    assert code != 0 and status["available"] is False
    assert [p["code"] for p in status["problems"]] == ["CATALOG_NOT_AVAILABLE"] and "catalog sync" in status["problems"][0]["message"]

    code, out, _ = _cli("catalog", "use", OLD_VERSION)
    assert code != 0 and json.loads(out)["error"]["code"] == "CATALOG_NOT_AVAILABLE"

    # after the move to a current release, the kept earlier release no longer fails a deep verification
    assert _cli("catalog", "use", current)[0] == 0
    code, out, _ = _cli("catalog", "verify", "--deep")
    verified = json.loads(out)
    assert code == 0 and verified["ok"] is True
    assert "earlier format" in next(r for r in verified["releases"] if r["catalog_version"] == OLD_VERSION)["note"]
    assert _cli("catalog", "status")[0] == 0 and _cli("doctor")[0] in (0, 1)


def test_json_api_catalog_status_does_not_call_an_earlier_schema_release_available(tmp_path, monkeypatch) -> None:
    from os_optics.local import json_api

    _retired_store(tmp_path / "store", monkeypatch)
    answer = json_api.catalog_status()
    assert answer["active"]["catalog_version"] == OLD_VERSION and answer["available"] is False


# ---- A7: a two-beam target of two wavelengths is NOT_APPLICABLE, not "not reached" -----------------------------------


def test_a_pair_target_of_two_wavelengths_names_the_mismatch_instead_of_not_reached() -> None:
    from local_helpers import PCX_F100, element, plane, project, run, source

    document = project(
        [source("SRC_488", (-200.0, 0.0, 0.0), wavelength_nm=488.0), source("SRC_638", (-200.0, 1.0, 0.0), wavelength_nm=638.0),
         element("L1", "lens", PCX_F100)],
        [plane("focal_plane", (101.5, 0.0, 0.0))],
        [{"target_id": "angle", "metric": "beam_crossing_angle_deg", "observation_id": "focal_plane", "source_id": "SRC_488",
          "source_id_b": "SRC_638", "maximum": 5.0, "unit": "deg"},
         {"target_id": "p488", "metric": "power_W", "observation_id": "focal_plane", "source_id": "SRC_488", "minimum": 0.0,
          "unit": "W"}])
    result = run(document).result
    angle = next(t for t in result.targets if t.target_id == "angle")
    assert angle.status.value == "NOT_APPLICABLE", angle
    assert "WAVELENGTH_MISMATCH" in angle.issue_codes and "488 nm" in angle.message and "638 nm" in angle.message
    assert next(t for t in result.targets if t.target_id == "p488").status.value == "PASS"
    from os_optics.validation.descriptors import metric_descriptors
    crossing = next(d for d in metric_descriptors() if d.name == "beam_crossing_angle_deg")
    assert "same wavelength" in crossing.validity
