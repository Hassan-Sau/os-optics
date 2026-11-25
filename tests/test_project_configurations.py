"""Projects with configurations: a 35 / 70 / 105 mm three-group zoom of ideal thin lenses runs in one invocation,
each configuration as its own Project, written side by side with a comparison; a budget-limited configuration reports its
own termination and leaves the others complete; there is no overall verdict.

Independent references (computed here, never with the runtime's own optics): the paraxial ABCD of thin lenses in slope form
(``s' = s - h / f``, exact at every angle for an ideal thin lens) gives each configuration's focal length and the chief-ray
height of the 1 deg field beam on the image plane; the Gaussian q-parameter through the same matrices gives the axial beam
radius there."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

from os_optics.cli import main
from os_optics.cli_exit import STATUS_CONTRACT_ENV
from os_optics.local import (
    LocalRunError,
    ProjectDocumentV2,
    configurations_run_files,
    json_api,
    lower_project,
    run_configured_project,
    run_freshness,
    run_optical_project,
    verify_run_artifacts,
)
from os_optics.local.artifacts import run_artifact_files
from os_optics.local.configurations import (
    COMPARISON_FILE,
    CONFIGURATIONS_RUN_SCHEMA,
    configurations_input_hash,
)
from os_optics.local.submission import CONFIGURATIONS_NOT_SUBMITTABLE, ConfigurationsNotSubmittableError, submission_document
from os_optics.results import ResultsError, load_run

ZOOM = Path(__file__).resolve().parent / "fixtures" / "configurations" / "zoom_35_70_105.json"
FOCAL = {"G1": 100.0, "G2": -30.0, "G3": 50.0}
NOMINAL_EFL = {"wide": 35.0, "mid": 70.0, "tele": 105.0}
SOURCE_X, IMAGE_X = -50.0, 200.0
WAVELENGTH_MM = 633.0e-6
WAIST_MM = 1.0
FIELD_DEG = 1.0


@pytest.fixture(autouse=True)
def _default_contract(monkeypatch):
    monkeypatch.delenv(STATUS_CONTRACT_ENV, raising=False)


def zoom() -> dict:
    return json.loads(ZOOM.read_text(encoding="utf-8"))


def _positions(document: dict, configuration_id: str) -> dict[str, float]:
    poses = {c["instance_id"]: c["nominal_pose"]["position_mm"][0] for c in document["components"]}
    configuration = next(c for c in document["configurations"] if c["configuration_id"] == configuration_id)
    poses.update({o["instance_id"]: o["nominal_pose"]["position_mm"][0] for o in configuration["component_poses"]})
    return poses


def _mul(a, b):
    return ((a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
            (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]))


def _system(document: dict, configuration_id: str, start: float, end: float):
    """The ABCD matrix (y, slope) from x = start to x = end through the three groups."""

    x = _positions(document, configuration_id)
    m, at = ((1.0, 0.0), (0.0, 1.0)), start
    for group in ("G1", "G2", "G3"):
        m = _mul(((1.0, x[group] - at), (0.0, 1.0)), m)
        m = _mul(((1.0, 0.0), (-1.0 / FOCAL[group], 1.0)), m)
        at = x[group]
    return _mul(((1.0, end - at), (0.0, 1.0)), m)


def reference(document: dict, configuration_id: str) -> dict[str, float]:
    x = _positions(document, configuration_id)
    (a, _b), (c, _d) = _system(document, configuration_id, x["G1"], x["G3"])
    (_a2, b2), _row = _system(document, configuration_id, SOURCE_X, IMAGE_X)
    in_air_mm = WAVELENGTH_MM / air_index(633.0)  # (the Project operates in air at 20 C and 101325 Pa: the beam's wavelength there)
    q = complex(0.0, math.pi * WAIST_MM * WAIST_MM / in_air_mm)  # the waist at the source
    (qa, qb), (qc, qd) = _system(document, configuration_id, SOURCE_X, IMAGE_X)
    q = (qa * q + qb) / (qc * q + qd)
    return {"efl_mm": -1.0 / c, "bfl_mm": -a / c, "field_height_mm": b2 * math.tan(math.radians(FIELD_DEG)),
            "axis_radius_mm": math.sqrt(in_air_mm / math.pi * abs(q) ** 2 / q.imag)}


def _arrival(run, source_id: str):
    (observation,) = run.result.observations
    return next(a for a in observation.arrivals if a.beam.source_id == source_id)


def test_the_zoom_runs_in_one_invocation_with_each_configuration_separate(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["project", "run", str(ZOOM), "--out-dir", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["exit_contract"] == "requirements" and summary["execution"] == {
        "complete": True, "completed": ["wide", "mid", "tele"], "not_completed": []}
    assert "status" not in summary  # no overall verdict
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == CONFIGURATIONS_RUN_SCHEMA
    assert [e["directory"] for e in manifest["configurations"]] == ["configurations/wide", "configurations/mid", "configurations/tele"]
    assert manifest["engine_input_hash"] == configurations_input_hash(
        [(e["configuration_id"], e["engine_input_hash"]) for e in manifest["configurations"]])
    assert len({e["engine_input_hash"] for e in manifest["configurations"]}) == 3
    assert verify_run_artifacts(out) == ()
    assert main(["project", "verify", str(out)]) == 0
    capsys.readouterr()
    document = zoom()
    for configuration_id, efl in NOMINAL_EFL.items():
        expected = reference(document, configuration_id)
        results = load_run(out / "configurations" / configuration_id, verify=True)  # an ordinary run directory: every reader opens it
        assert results.integrity_problems == ()
        targets = {t["target_id"]: t for t in results.targets()["targets"]}
        assert set(targets) == {"axis_power", "axis_spot", f"image_height_{configuration_id}"}  # only its own height target
        assert all(t["status"] == "PASS" for t in targets.values())
        assert targets[f"image_height_{configuration_id}"]["value"] == pytest.approx(expected["field_height_mm"], rel=1e-9)
        assert expected["efl_mm"] == pytest.approx(efl, abs=2e-4)
        assert expected["field_height_mm"] == pytest.approx(efl * math.tan(math.radians(FIELD_DEG)), abs=1e-5)
        assert targets["axis_spot"]["value"] == pytest.approx(expected["axis_radius_mm"], rel=1e-6)


def test_the_comparison_sets_the_configurations_side_by_side(tmp_path, capsys):
    out = tmp_path / "run"
    run = run_configured_project(ZOOM.read_bytes(), output_dir=out)
    document = zoom()
    comparison = run.comparison
    assert comparison.analysis == "CONFIGURATIONS" and comparison.execution.outcome.value == "COMPLETED"
    assert comparison.identity.physical.engine_input_hash == run.engine_input_hash
    rows = comparison.arrays["configurations"]
    assert [r["configuration_id"] for r in rows] == ["wide", "mid", "tele"] and [r["label"] for r in rows] == ["35 mm", "70 mm", "105 mm"]
    assert all(r["execution"]["outcome"] == "COMPLETED" and r["requirements"]["unmet"] == [] for r in rows)
    for cid in NOMINAL_EFL:
        expected = reference(document, cid)
        assert comparison.quantity(f"{cid}|image|FIELD#0|0|position_u_mm").value == pytest.approx(expected["field_height_mm"], rel=1e-9)
        radius = comparison.quantity(f"{cid}|image|AXIS#0|0|beam_radius_x_mm")
        assert radius.value == pytest.approx(expected["axis_radius_mm"], rel=1e-6) and radius.scope.configuration_id == cid
        assert comparison.quantity(f"{cid}|image|power_W").state.value in ("KNOWN", "BOUNDED")
    heights = next(t for t in comparison.arrays["targets"] if t["target_id"] == "image_height_mid")
    assert [c["applies"] for c in heights["configurations"]] == [False, True, False]
    assert heights["configurations"][1]["assessment"] == "MET"
    # the paraxial rows: effective and back focal length per configuration against the slope-form ABCD
    assert main(["results", "configurations", str(out), "--paraxial", "G1,G2,G3", "--wavelength-nm", "633"]) == 0
    with_paraxial = json.loads(capsys.readouterr().out)
    values = {q["quantity_id"]: q["value"] for q in with_paraxial["quantities"]}
    for cid, efl in NOMINAL_EFL.items():
        expected = reference(document, cid)
        assert values[f"{cid}|paraxial|efl_mm"] == pytest.approx(expected["efl_mm"], rel=1e-12)
        assert values[f"{cid}|paraxial|bfl_mm"] == pytest.approx(expected["bfl_mm"], rel=1e-12)
    # the side-by-side table: one row per quantity, one column per configuration
    assert main(["results", "configurations", str(out), "--table"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["configurations"] == ["wide", "mid", "tele"]
    row = next(r for r in table["rows"] if r["name"] == "position_u_mm" and r["source_id"] == "FIELD")
    assert list(row["values"]) == ["wide", "mid", "tele"] and row["crossing_index"] == 0
    assert row["values"]["tele"]["value"] / row["values"]["wide"]["value"] == pytest.approx(3.0, rel=1e-4)


def test_each_configuration_is_exactly_the_run_of_its_own_project(tmp_path):
    run = run_configured_project(ZOOM.read_bytes())
    files = configurations_run_files(run)
    project = ProjectDocumentV2.model_validate(zoom())
    for configuration_id in NOMINAL_EFL:
        solo = run_artifact_files(run_optical_project(project.configured(configuration_id)))
        prefix = f"configurations/{configuration_id}/"
        mine = {name[len(prefix):]: data for name, data in files.items() if name.startswith(prefix)}
        assert mine == solo, configuration_id
    # the Engine input of a configuration is that of the same design written as an ordinary Project
    plain = zoom()
    del plain["configurations"]
    plain["targets"] = [t for t in plain["targets"] if t.get("configuration_ids") in (None, ["mid"])]
    for target in plain["targets"]:
        target.pop("configuration_ids", None)
    assert lower_project(plain).engine_input_hash == run.configuration("mid").engine_input_hash


def _budget_limited() -> dict:
    document = zoom()
    tele = next(c for c in document["configurations"] if c["configuration_id"] == "tele")
    tele["compute_budget"] = {"max_interactions_per_path": 2}  # the beams stop after G1 and G2
    return document


def test_a_budget_limited_configuration_reports_its_termination_beside_completed_ones(tmp_path, capsys):
    out, full = tmp_path / "limited", tmp_path / "full"
    path = tmp_path / "limited.json"
    path.write_text(json.dumps(_budget_limited()), encoding="utf-8")
    codes = {}
    for contract in ("requirements", "execution", "aggregate"):
        codes[contract] = main(["project", "run", str(path), "--out-dir", str(out), "--status-contract", contract])
        summary = json.loads(capsys.readouterr().out)
    assert codes == {"requirements": 3, "execution": 1, "aggregate": 2}
    assert summary["execution"]["complete"] is False and summary["execution"]["completed"] == ["wide", "mid"]
    assert summary["execution"]["not_completed"] == [{"configuration_id": "tele", "outcome": "PARTIAL", "termination": "BUDGET_EXHAUSTED"}]
    tele = next(c for c in summary["configurations"] if c["configuration_id"] == "tele")
    assert tele["execution"]["budgets"]["exhausted"] == ["max_interactions_per_path"]
    assert verify_run_artifacts(out) == ()
    run_configured_project(ZOOM.read_bytes(), output_dir=full)
    for cid in ("wide", "mid"):  # the completed configurations are exactly those of the run without the limit
        for name in ("simulation_result.json", "local_validation.json", "manifest.json"):
            assert (out / "configurations" / cid / name).read_bytes() == (full / "configurations" / cid / name).read_bytes()
    comparison = json.loads((out / COMPARISON_FILE).read_text(encoding="utf-8"))
    assert comparison["execution"]["outcome"] == "PARTIAL" and comparison["execution"]["termination"] == "BUDGET_EXHAUSTED"
    assert "tele (PARTIAL, BUDGET_EXHAUSTED)" in comparison["execution"]["reason"]
    rows = {r["configuration_id"]: r for r in comparison["arrays"]["configurations"]}
    assert rows["wide"]["execution"]["outcome"] == rows["mid"]["execution"]["outcome"] == "COMPLETED"
    assert rows["tele"]["requirements"]["indeterminate"] and rows["tele"]["validation_status"] != "PASS"
    assert main(["results", "configurations", str(out)]) == 2  # the comparison of a partial run
    capsys.readouterr()


def test_exit_codes_combine_the_configurations_without_an_overall_pass(tmp_path, capsys):
    document = zoom()
    spot = next(t for t in document["targets"] if t["target_id"] == "axis_spot")
    spot["maximum"] = 0.018  # met at 35 and 70 mm (7.1 / 14.1 um), not at 105 mm (21.2 um)
    path = tmp_path / "zoom.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    codes = {}
    for contract in ("requirements", "execution", "aggregate"):
        codes[contract] = main(["project", "run", str(path), "--status-contract", contract])
        summary = json.loads(capsys.readouterr().out)
    assert codes == {"requirements": 1, "execution": 0, "aggregate": 1}
    statuses = {c["configuration_id"]: c["summary"]["status"] for c in summary["configurations"]}
    assert statuses == {"wide": "PASS", "mid": "PASS", "tele": "FAIL"}
    assert main(["project", "run", str(path), "--configuration", "wide", "--configuration", "mid"]) == 0
    assert [c["configuration_id"] for c in json.loads(capsys.readouterr().out)["configurations"]] == ["wide", "mid"]


def test_one_configuration_and_freshness_per_configuration(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["project", "run", str(ZOOM), "--configuration", "mid", "--out-dir", str(out)]) == 0
    capsys.readouterr()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [e["configuration_id"] for e in manifest["configurations"]] == ["mid"]
    assert verify_run_artifacts(out) == ()
    partial = run_freshness(ZOOM.read_bytes(), out)
    assert partial["freshness"] == "HISTORICAL" and partial["reasons"] == ["CONFIGURATIONS_CHANGED"]
    assert partial["not_run"] == ["wide", "tele"] and partial["configurations"][0]["freshness"] == "CURRENT"
    # the configuration that was run is the Project's: its physics is current, and the coverage names what is missing
    assert partial["physics"] == {"freshness": "CURRENT", "reasons": []} and partial["requirements"] == {"freshness": "CURRENT"}
    assert partial["coverage"] == {"declared": ["wide", "mid", "tele"], "run": ["mid"], "not_run": ["wide", "tele"],
                                   "no_longer_declared": [], "complete": False}
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    whole = run_freshness(ZOOM.read_bytes(), out)
    assert whole["freshness"] == "CURRENT" and whole["physics"]["freshness"] == "CURRENT" and whole["coverage"]["complete"] is True
    # a zoom position added to the Project: the three that were run keep their physics
    more = zoom()
    more["configurations"].append({**copy.deepcopy(more["configurations"][0]), "configuration_id": "wider", "label": "30 mm"})
    grown = run_freshness(more, out)
    assert (grown["freshness"], grown["reasons"]) == ("HISTORICAL", ["CONFIGURATIONS_CHANGED"])
    assert grown["physics"] == {"freshness": "CURRENT", "reasons": []}
    assert grown["coverage"]["not_run"] == ["wider"] and grown["coverage"]["complete"] is False
    fewer = zoom()
    fewer["configurations"] = fewer["configurations"][:2]
    fewer["targets"] = [t for t in fewer["targets"] if "tele" not in (t.get("configuration_ids") or [])]
    shrunk = run_freshness(fewer, out)
    assert shrunk["coverage"]["no_longer_declared"] == ["tele"] and shrunk["coverage"]["run"] == ["wide", "mid"]
    assert main(["results", "freshness", str(out), "--project", str(ZOOM)]) == 0
    capsys.readouterr()
    moved = zoom()
    moved["configurations"][2]["component_poses"][0]["nominal_pose"]["position_mm"][0] = 60.0  # tele's G2
    retargeted = zoom()
    next(t for t in retargeted["targets"] if t["target_id"] == "image_height_wide")["tolerance_abs"] = 0.002
    physics = run_freshness(moved, out)
    assert physics["freshness"] == "HISTORICAL" and physics["reasons"] == ["PHYSICS_CHANGED"]
    assert physics["physics"] == {"freshness": "HISTORICAL", "reasons": ["PHYSICS_CHANGED"]}
    assert [r["physics"]["freshness"] for r in physics["configurations"]] == ["CURRENT", "CURRENT", "HISTORICAL"]
    assert [(r["configuration_id"], r["freshness"]) for r in physics["configurations"]] == [
        ("wide", "CURRENT"), ("mid", "CURRENT"), ("tele", "HISTORICAL")]
    targets = run_freshness(retargeted, out)
    assert [(r["configuration_id"], r["reasons"]) for r in targets["configurations"]] == [
        ("wide", ["TARGETS_CHANGED"]), ("mid", []), ("tele", [])]
    # a target edit leaves every configuration's physics the run's; only wide's requirements need assessing again
    assert targets["physics"] == {"freshness": "CURRENT", "reasons": []} and targets["requirements"] == {"freshness": "HISTORICAL"}
    assert [r["requirements"]["freshness"] for r in targets["configurations"]] == ["HISTORICAL", "CURRENT", "CURRENT"]
    # an ordinary run is not a run of the Project's configurations
    plain = run_optical_project(ProjectDocumentV2.model_validate(zoom()).configured("mid"), output_dir=tmp_path / "plain")
    assert run_freshness(ZOOM.read_bytes(), plain.artifacts.directory)["reasons"] == ["CONFIGURATIONS_CHANGED"]


def test_projects_with_configurations_are_refused_where_one_state_is_evaluated(tmp_path, capsys):
    for call in (lambda: lower_project(zoom()), lambda: run_optical_project(zoom())):
        with pytest.raises(LocalRunError) as caught:
            call()
        assert caught.value.codes == ("CONFIGURATION_NOT_SELECTED",) and "wide, mid, tele" in str(caught.value)
    refused = json_api.paraxial_analysis(zoom(), {"components": ["G1", "G2", "G3"], "wavelength_nm": 633})
    assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "CONFIGURATION_NOT_SELECTED"
    with pytest.raises(LocalRunError) as caught:
        run_configured_project(zoom(), configuration_ids=["zoom"])
    assert caught.value.codes == ("CONFIGURATION_UNKNOWN",)
    plain = zoom()
    del plain["configurations"]
    for target in plain["targets"]:
        target.pop("configuration_ids", None)
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(plain), encoding="utf-8")
    assert main(["project", "run", str(path), "--configuration", "wide"]) == 65
    assert json.loads(capsys.readouterr().out)["problems"][0]["code"] == "CONFIGURATION_UNKNOWN"
    assert main(["project", "run", str(ZOOM), "-o", str(tmp_path / "record.json")]) == 64
    capsys.readouterr()


def test_one_configuration_is_analysed_and_evaluated_as_its_own_project(tmp_path, capsys):
    """Paraxial analysis, cavity analysis, network export and evaluation of a Project with configurations take the
    configuration by its id and return what they return for ``configured(id)``; no configuration is chosen for the caller."""

    from os_optics.local import evaluate_project, load_project  # noqa: PLC0415
    from os_optics.local.configurations import configured_project, lower_configurations  # noqa: PLC0415
    from os_optics.local.network import export_linear_optical_network  # noqa: PLC0415
    from os_optics.network import NetworkExportError  # noqa: PLC0415

    tele = load_project(zoom()).configured("tele")
    assert configured_project(zoom(), "tele") == tele and configured_project(tele) == tele
    assert configured_project(zoom()).configurations  # (no id: the Project as it is; lowering it refuses)
    for bad in ("zoom", 3):
        with pytest.raises(LocalRunError) as caught:
            configured_project(zoom(), bad)
        assert caught.value.codes == ("CONFIGURATION_UNKNOWN",)
    with pytest.raises(LocalRunError) as caught:
        configured_project(tele, "tele")  # (the configured Project declares none)
    assert caught.value.codes == ("CONFIGURATION_UNKNOWN",)

    # paraxial: quantity by quantity the analysis of configured("tele"), with the configuration stated
    request = {"components": ["G1", "G2", "G3"], "wavelength_nm": 633}
    named = json_api.paraxial_analysis(zoom(), {**request, "configuration_id": "tele"})
    own = json_api.paraxial_analysis(tele.payload(), request)
    assert named["ok"] and own["ok"] and "configuration_id" not in own
    assert named.pop("configuration_id") == "tele" and named == own
    wide = json_api.paraxial_analysis(zoom(), {**request, "configuration_id": "wide"})
    assert wide["system"]["efl_mm"] != named["system"]["efl_mm"]  # (another zoom position, another focal length)
    unknown = json_api.paraxial_analysis(zoom(), {**request, "configuration_id": "zoom"})
    assert unknown["ok"] is False and unknown["error"]["problems"][0]["code"] == "CONFIGURATION_UNKNOWN"

    # evaluation: the Engine input is the configuration's (the one a run of the configurations records for it)
    run_hashes = {item.configuration_id: item.lowered.engine_input_hash for item in lower_configurations(zoom())}
    evaluation = evaluate_project(zoom(), configuration_id="tele")
    assert evaluation.engine_input_hash == run_hashes["tele"] == evaluate_project(tele).engine_input_hash
    reply = json_api.evaluate_project(zoom(), configuration_id="tele")
    assert reply["ok"] and reply["monitors"] == json_api.evaluate_project(tele.payload())["monitors"]
    assert main(["project", "evaluate", str(ZOOM), "--configuration", "tele", "--progress", "none"]) == 0
    assert json.loads(capsys.readouterr().out)["monitors"] == reply["monitors"]
    assert main(["project", "evaluate", str(ZOOM), "--configuration", "zoom", "--progress", "none"]) == 65
    assert json.loads(capsys.readouterr().out)["error"]["problems"][0]["code"] == "CONFIGURATION_UNKNOWN"

    # network export and the paraxial command: the same document or refusal as for configured("tele")
    def network(project, **kwargs):
        try:
            return export_linear_optical_network(project, **kwargs).contract_hash
        except NetworkExportError as refused:
            return [p.code for p in refused.problems]

    assert network(zoom(), configuration_id="tele") == network(tele)
    assert main(["project", "paraxial", str(ZOOM), "--components", "G1", "G2", "G3", "--wavelength-nm", "633", "--configuration", "tele"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("configuration_id") == "tele" and printed == {k: v for k, v in own.items() if k != "ok"}


def test_a_cavity_analysis_names_its_configuration_in_its_settings_and_quantities():
    cavity = json.loads((Path(__file__).resolve().parent / "fixtures" / "cavity" / "two_mirror_cavity_probe.json").read_text(encoding="utf-8"))
    mirrors = [c["instance_id"] for c in cavity["components"] if c["optical_role"] == "mirror"]
    moved = copy.deepcopy(next(c for c in cavity["components"] if c["instance_id"] == mirrors[-1])["nominal_pose"])
    moved["position_mm"] = [moved["position_mm"][0] + 5.0, *moved["position_mm"][1:]]
    document = {**cavity, "design_intent": "OPTICAL_SYSTEM", "configurations": [
        {"configuration_id": "short", "component_poses": []},
        {"configuration_id": "long", "component_poses": [{"instance_id": mirrors[-1], "nominal_pose": moved}]}]}
    from os_optics.local import load_project  # noqa: PLC0415

    request = {"components": mirrors, "wavelength_nm": 1064}
    refused = json_api.cavity_analysis(document, request)
    assert refused["ok"] is False and refused["error"]["problems"][0]["code"] == "CONFIGURATION_NOT_SELECTED"
    named = json_api.cavity_analysis(document, {**request, "configuration_id": "long"})
    own = json_api.cavity_analysis(load_project(document).configured("long").payload(), request)
    assert named["ok"] and own["ok"], (named, own)
    assert named["settings"] == {**own["settings"], "configuration_id": "long"}
    assert named["identity"]["physical"]["engine_input_hash"] == own["identity"]["physical"]["engine_input_hash"]
    assert named["identity"]["physical"]["settings_hash"] != own["identity"]["physical"]["settings_hash"]
    assert [(q["quantity_id"], q["state"], q.get("value")) for q in named["quantities"]] == [
        (q["quantity_id"], q["state"], q.get("value")) for q in own["quantities"]]
    assert {q["scope"]["configuration_id"] for q in named["quantities"]} == {"long"}
    assert {q["scope"]["configuration_id"] for q in own["quantities"]} == {None}
    short = json_api.cavity_analysis(document, {**request, "configuration_id": "short"})
    length = {q["quantity_id"]: q.get("value") for q in short["quantities"]}
    longer = {q["quantity_id"]: q.get("value") for q in named["quantities"]}
    assert length != longer  # (another mirror spacing, another cavity)


def _groups_overlap(document: dict, *configuration_ids: str) -> dict:
    """``document`` with G2 moved onto G1 in the named configurations: there the two groups occupy the same place, so that
    configuration's Project cannot be lowered (the optical-solid check refuses it); the others are unchanged."""

    g1 = next(c for c in document["components"] if c["instance_id"] == "G1")
    for configuration in document["configurations"]:
        if configuration["configuration_id"] in configuration_ids:
            next(p for p in configuration["component_poses"] if p["instance_id"] == "G2")["nominal_pose"] = copy.deepcopy(g1["nominal_pose"])
    return document


def test_a_configuration_that_cannot_be_lowered_is_kept_as_failed_and_the_others_run(tmp_path, capsys):
    """A zoom position whose groups overlap ('mid' only): the run writes the two valid configurations complete, keeps
    'mid' as FAILED / INVALID_INPUT with its problems and no Engine input, and is not complete (exit 3)."""

    from os_optics.local.configurations import configurations_freshness, run_reader, verify_configurations_run  # noqa: PLC0415

    document = _groups_overlap(zoom(), "mid")
    out = tmp_path / "run"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 3
    capsys.readouterr()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["configuration_id"]: e for e in manifest["configurations"]}
    mid = entries["mid"]
    assert mid["engine_input_hash"] is None and mid["manifest_sha256"] is None and mid["result_hash"] is None
    assert (mid["execution"]["outcome"], mid["execution"]["termination"], mid["execution"]["complete"]) == ("FAILED", "INVALID_INPUT", False)
    assert mid["problems"] and all(p["location"].startswith("configuration mid") for p in mid["problems"])
    assert {p["code"] for p in mid["problems"]} <= {"OPTICAL_BODIES_OVERLAP", "OPTICAL_BODIES_TOUCH"}  # the solid check's
    assert manifest["execution"] == {"complete": False, "completed": ["wide", "tele"], "not_completed": [
        {"configuration_id": "mid", "outcome": "FAILED", "termination": "INVALID_INPUT"}]}
    assert not (out / "configurations" / "mid").exists()
    # the valid configurations are the runs of the Project without the problem, hash for hash
    good = run_configured_project(ZOOM.read_bytes())
    for cid in ("wide", "tele"):
        assert (out / "configurations" / cid / "manifest.json").is_file()
        assert entries[cid]["engine_input_hash"] == good.configuration(cid).engine_input_hash
        assert entries[cid]["result_hash"] == good.configuration(cid).run.local_validation.result_hash.digest
    read = run_reader(str(out))
    assert verify_configurations_run(read) == ()
    # the run's input hash covers the configuration without an Engine input: the same for the same Project, another than
    # the hash of the run in which it lowers
    assert manifest["engine_input_hash"] == configurations_input_hash([(cid, entries[cid]["engine_input_hash"]) for cid in ("wide", "mid", "tele")])
    again = run_configured_project(document)
    assert again.engine_input_hash == manifest["engine_input_hash"] != good.engine_input_hash
    assert again.complete is False and again.configuration("mid").run is None and again.configuration("mid").engine_input_hash is None
    # freshness names it as not run; the check lists it with its problems and the others as ok
    freshness = configurations_freshness(document, read)
    assert freshness["not_run"] == ["mid"] and freshness["freshness"] == "HISTORICAL"
    reply = json_api.check_project(document)
    assert reply["ok"] is True and reply["not_lowered"] == ["mid"] and reply["engine_input_hash"] == manifest["engine_input_hash"]
    assert [(row["configuration_id"], row["ok"]) for row in reply["configurations"]] == [("wide", True), ("mid", False), ("tele", True)]
    assert reply["configurations"][1]["problems"] == mid["problems"] and reply["configurations"][1]["engine_input_hash"] is None
    assert json_api.check_project(zoom())["not_lowered"] == []


def test_a_call_that_runs_nothing_leaves_an_earlier_run_as_it_was(tmp_path, capsys):
    """``run_configured_project`` refusals found before anything is traced -- a configuration the Project does not declare,
    a selection of which no configuration can be lowered -- leave the directory of an earlier run as it was, and so does
    the command line with ``--out-dir``."""

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    before = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    with pytest.raises(LocalRunError) as caught:
        run_configured_project(zoom(), configuration_ids=["zoom"], output_dir=out)
    assert caught.value.codes == ("CONFIGURATION_UNKNOWN",)
    with pytest.raises(LocalRunError) as caught:
        run_configured_project(_groups_overlap(zoom(), "wide", "mid", "tele"), output_dir=out)
    assert {p.location.split()[1] for p in caught.value.problems} == {"wide", "mid", "tele"}
    with pytest.raises(LocalRunError) as caught:  # one configuration selected, and it cannot be lowered: nothing runs either
        run_configured_project(_groups_overlap(zoom(), "mid"), configuration_ids=["mid"], output_dir=out)
    assert all(p.location.startswith("configuration mid") for p in caught.value.problems)
    assert {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()} == before
    # from the command line the three are refused with exit 65, into the same directory
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_groups_overlap(zoom(), "wide", "mid", "tele")), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 65
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert {p["location"].split()[1] for p in problems} == {"wide", "mid", "tele"}
    # the refused run wrote nothing: the earlier run is as it was and verifies; it is the earlier Project's, not this one's
    assert {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()} == before
    assert verify_run_artifacts(out) == ()
    assert json.loads((out / "manifest.json").read_bytes())["project"]["project_hash"]["digest"] != \
        ProjectDocumentV2.model_validate(json.loads(path.read_bytes())).content_hash()


def test_a_configuration_whose_engine_run_fails_is_recorded_and_the_others_written(tmp_path, monkeypatch):
    from os_optics.local import configurations as module
    from os_optics.local.errors import LocalRunErrorCode, fail

    real = module.session_run  # (each configuration's run: run_optical_project through the run's one session)

    def failing(session, project, **kwargs):
        if project.components[3].nominal_pose.position_mm[0] == _positions(zoom(), "tele")["G2"]:
            raise fail(LocalRunErrorCode.ENGINE_RUN_INCOMPLETE, "the Engine could not evaluate the Project (NUMERICAL_FAILURE): test")
        return real(session, project, **kwargs)

    monkeypatch.setattr(module, "session_run", failing)
    out = tmp_path / "run"
    run = run_configured_project(ZOOM.read_bytes(), output_dir=out)
    tele = run.configuration("tele")
    assert tele.run is None and tele.execution["outcome"] == "FAILED" and tele.execution["termination"] == "NUMERICAL_FAILURE"
    assert tele.problems[0].location == "configuration tele"
    assert sorted(p.name for p in (out / "configurations").iterdir()) == ["mid", "wide"]
    assert verify_run_artifacts(out) == ()
    assert run.comparison.execution.outcome.value == "PARTIAL" and run.comparison.execution.termination.value == "NUMERICAL_FAILURE"
    from os_optics.cli_exit import configurations_exit_code

    assert [configurations_exit_code(c, run) for c in ("requirements", "execution", "aggregate")] == [3, 1, 2]


def test_readers_open_a_configuration_and_name_the_configurations_at_the_top(tmp_path):
    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    with pytest.raises(ResultsError) as caught:
        load_run(out)
    assert caught.value.code == "CONFIGURATIONS_RUN" and "configurations/wide" in caught.value.message
    assert load_run(out / "configurations" / "wide", verify=True).describe()["integrity"]["consistent"] is True
    reply = json_api.run_project(zoom(), output_dir=str(tmp_path / "api"))
    assert reply["ok"] is True and reply["verify"] == [] and "status" not in reply
    assert [c["status"] for c in reply["configurations"]] == ["PASS", "PASS", "PASS"]
    assert reply["comparison"]["analysis"] == "CONFIGURATIONS"
    files = reply["files"]
    assert "configurations/tele/simulation_result.json" in files and "comparison.json" in files
    targets = json_api.run_results(files, {"kind": "targets", "configuration": "tele"})
    assert targets["ok"] is True and {t["target_id"] for t in targets["result"]["targets"]} >= {"image_height_tele"}
    top = json_api.run_results(files, {"kind": "targets"})
    assert top["ok"] is False and top["error"]["code"] == "CONFIGURATIONS_RUN"
    compared = json_api.configurations_comparison(files, {"components": ["G1", "G2", "G3"], "wavelength_nm": 633})
    assert compared["ok"] is True and any(q["quantity_id"] == "mid|paraxial|efl_mm" for q in compared["quantities"])
    assert json_api.configurations_comparison(run_artifact_files(run_optical_project(
        ProjectDocumentV2.model_validate(zoom()).configured("wide"))))["error"]["code"] == "CONFIGURATIONS_RUN_INVALID"
    assert json_api.run_freshness(zoom(), files)["freshness"] == "CURRENT"
    check = json_api.check_project(zoom())
    assert check["ok"] is True and check["engine_input_hash"] == reply["engine_input_hash"]
    assert [c["engine_input_hash"] for c in check["configurations"]] == [c["engine_input_hash"] for c in reply["configurations"]]


def test_tampered_runs_do_not_verify(tmp_path):
    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    comparison = out / COMPARISON_FILE
    comparison.write_bytes(comparison.read_bytes().replace(b'"35 mm"', b'"36 mm"'))
    assert any("comparison.json: size or SHA-256" in p for p in verify_run_artifacts(out))
    # comparison.json is the Engine's name in the run directory: the next run replaces the changed file
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    assert verify_run_artifacts(out) == ()
    record = out / "configurations" / "mid" / "local_validation.json"
    record.write_bytes(record.read_bytes() + b" ")
    assert any(p.startswith("configurations/mid/") for p in verify_run_artifacts(out))


def test_a_run_of_configurations_is_not_submitted(tmp_path):
    out = tmp_path / "run"
    run = run_configured_project(ZOOM.read_bytes(), output_dir=out)
    for given in (out, configurations_run_files(run), run):
        with pytest.raises(ConfigurationsNotSubmittableError) as caught:
            submission_document(given)
        assert caught.value.code == CONFIGURATIONS_NOT_SUBMITTABLE and "configurations/<id>" in caught.value.problems[0]
    refused = json_api.commercial_submission(configurations_run_files(run))
    assert refused["error"]["code"] == CONFIGURATIONS_NOT_SUBMITTABLE
    # one configuration's run directory is the run of an ordinary Project
    document = submission_document(out / "configurations" / "mid")
    assert "configurations" not in document["project"]
    assert document["project_hash"] == ProjectDocumentV2.model_validate(zoom()).configured("mid").content_hash()


def test_a_project_without_configurations_runs_as_before():
    plain = copy.deepcopy(zoom())
    del plain["configurations"]
    for target in plain["targets"]:
        target.pop("configuration_ids", None)
    run = run_optical_project(plain)
    files = run_artifact_files(run)
    assert json.loads(files["manifest.json"])["schema"] == "opensci.local_run"
    assert "configurations" not in json.loads(files["project.json"])


def test_a_laboratory_assembly_whose_configurations_move_fixed_dofs_loads_and_check_project_names_each_move():
    """The zoom as a laboratory assembly (no DOF retained): every configuration moves G2 and G3 along their fixed Tz. The
    Project loads and is checked; each move is a CONFIGURATION_MOVES_FIXED_DOF advisory of its configuration; as an
    optical system there is none."""

    lab = {k: v for k, v in zoom().items() if k != "design_intent"}
    checked = json_api.check_project(lab)
    assert checked["ok"], checked
    moves = [w for w in checked["warnings"] if w["code"] == "CONFIGURATION_MOVES_FIXED_DOF"]
    nominal = {c["instance_id"]: c["nominal_pose"]["position_mm"] for c in lab["components"]}
    expected = {(c["configuration_id"], p["instance_id"]) for c in lab["configurations"] for p in c["component_poses"]
                if p["nominal_pose"]["position_mm"] != nominal[p["instance_id"]]}  # (the mid position is the nominal one)
    assert {i for _, i in expected} == {"G2", "G3"}
    assert {(w["details"]["configuration_id"], w["details"]["instance_id"]) for w in moves} == expected
    assert all(w["location"] == f"configuration {w['configuration_id']} component {w['details']['instance_id']}" for w in moves)
    assert all("along its fixed Tz" in w["details"]["motions"][0] for w in moves)
    system = json_api.check_project(zoom())
    assert not [w for w in system["warnings"] if w["code"] == "CONFIGURATION_MOVES_FIXED_DOF"]
    assert [r["engine_input_hash"] for r in checked["configurations"]] != [] and len(checked["configurations"]) == len(system["configurations"])


def _tree(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("own_name, own_bytes", [("simulation_result.json", b'{"mine": true}'),
                                                 ("manifest.json", b'{"my": "index of this folder"}')])
def test_a_configuration_that_fails_never_removes_a_file_of_the_persons_own_in_its_directory(tmp_path, capsys, own_name, own_bytes):
    """'mid' cannot be lowered, so the run removes what an earlier run held under configurations/mid. A file there at a
    run-file name that no Engine manifest lists -- the person's own result, or a manifest.json that is no Engine manifest
    -- is not the Engine's: the run names it and stops before anything is written (exit 64 from the command line, a
    RUN_DIRECTORY_OCCUPIED reply from the JSON API), and every file stays as it was. Files at other names are never
    touched either way."""

    out = tmp_path / "run"
    mine = out / "configurations" / "mid"
    mine.mkdir(parents=True)
    (mine / own_name).write_bytes(own_bytes)
    (mine / "notes.txt").write_bytes(b"my notes")
    before = _tree(out)
    document = _groups_overlap(zoom(), "mid")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 64
    assert f"{own_name}" in capsys.readouterr().err
    assert _tree(out) == before
    reply = json_api.run_project(document, output_dir=str(out))
    assert reply["ok"] is False and reply["error"]["problems"][0]["code"] == "RUN_DIRECTORY_OCCUPIED"
    assert own_name in reply["error"]["problems"][0]["message"]
    assert _tree(out) == before


def test_a_configuration_that_fails_removes_the_earlier_engine_run_in_its_directory(tmp_path, capsys):
    """The positive side: an earlier run of the Project wrote configurations/mid (named by the directory's manifest, its
    files listed by mid's own manifest). When 'mid' now fails, that earlier run of it goes -- it is the Engine's -- while a
    file of the person's at another name stays."""

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    assert (out / "configurations" / "mid" / "manifest.json").is_file()
    (out / "configurations" / "mid" / "notes.txt").write_bytes(b"my notes")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_groups_overlap(zoom(), "mid")), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 3
    capsys.readouterr()
    assert sorted(p.name for p in (out / "configurations" / "mid").iterdir()) == ["notes.txt"]
    assert (out / "configurations" / "wide" / "manifest.json").is_file()


class _Killed(BaseException):
    """The process ends here (no handler, no ``finally`` of the commit's own cleanup runs as a kill would not run it)."""


@pytest.mark.parametrize("killed_after", ["configurations/mid/simulation_result.json", "configurations/mid/manifest.json",
                                          "configurations/wide/manifest.json", "manifest.json"])
def test_the_same_failing_run_started_again_after_a_kill_during_its_commit_completes(tmp_path, capsys, monkeypatch, killed_after):
    """An earlier run wrote configurations/mid; now 'mid' fails and the commit is killed right after it deletes
    ``killed_after`` (the commit's temporary files are left, as a kill leaves them). The same run started again completes
    (exit 3, the module's promise) and removes what the Engine wrote under configurations/mid, while the person's notes
    there stay: no kill leaves an Engine file in a configuration directory whose own manifest is gone."""

    from os_optics.output_files import StagedFiles  # noqa: PLC0415

    out = tmp_path / "run"
    run_configured_project(ZOOM.read_bytes(), output_dir=out)
    (out / "configurations" / "mid" / "notes.txt").write_bytes(b"my notes")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_groups_overlap(zoom(), "mid")), encoding="utf-8")
    target = (out / killed_after).resolve()
    unlink = Path.unlink

    def killing_unlink(self, missing_ok=False):
        unlink(self, missing_ok=missing_ok)
        if self.resolve() == target:
            raise _Killed

    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", killing_unlink)
        m.setattr(StagedFiles, "discard", lambda self: None)
        with pytest.raises(_Killed):
            main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"])
    assert not (out / killed_after).exists()
    capsys.readouterr()
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 3, capsys.readouterr().err
    assert sorted(p.name for p in (out / "configurations" / "mid").iterdir()) == ["notes.txt"]
    assert (out / "configurations" / "wide" / "manifest.json").is_file() and (out / "manifest.json").is_file()


@pytest.mark.parametrize("killed_after", ["configurations/mid/simulation_result.json", "configurations/mid/ray_data.json",
                                          "configurations/mid/manifest.json"])
def test_a_run_that_drops_a_configuration_killed_during_its_commit_leaves_nothing_a_later_run_refuses(tmp_path, capsys, monkeypatch,
                                                                                                      killed_after):
    """An earlier run wrote all three configurations; the Project without 'mid' is run there and its commit is killed right
    after it deletes ``killed_after``. The same run started again completes (exit 0) and leaves no run file under
    configurations/mid, and the Project with 'mid' started again completes too (exit 0): no Engine file is left there that
    no manifest lists."""

    from os_optics.output_files import StagedFiles  # noqa: PLC0415

    out = tmp_path / "run"
    full = tmp_path / "zoom.json"
    full.write_bytes(ZOOM.read_bytes())
    assert main(["project", "run", str(full), "--out-dir", str(out), "--progress", "none"]) == 0
    document = zoom()
    document["configurations"] = [c for c in document["configurations"] if c["configuration_id"] != "mid"]
    document["targets"] = [t for t in document["targets"] if "mid" not in (t.get("configuration_ids") or ())]
    path = tmp_path / "without_mid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    target = (out / killed_after).resolve()
    unlink = Path.unlink

    def killing_unlink(self, missing_ok=False):
        unlink(self, missing_ok=missing_ok)
        if self.resolve() == target:
            raise _Killed

    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", killing_unlink)
        m.setattr(StagedFiles, "discard", lambda self: None)
        with pytest.raises(_Killed):
            main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"])
    capsys.readouterr()
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 0, capsys.readouterr().err
    mid = out / "configurations" / "mid"
    assert not mid.exists() or list(mid.iterdir()) == []
    assert main(["project", "run", str(full), "--out-dir", str(out), "--progress", "none"]) == 0, capsys.readouterr().err
    assert (mid / "manifest.json").is_file()


def test_after_a_completed_failing_run_a_result_of_the_persons_own_in_that_configurations_directory_is_refused(tmp_path, capsys):
    """'mid' failed in the completed run there (the directory's manifest names it with no run). A simulation_result.json the
    person then places under configurations/mid is listed by no Engine manifest: the same run started again refuses it
    (exit 64) and every file stays."""

    out = tmp_path / "run"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_groups_overlap(zoom(), "mid")), encoding="utf-8")
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 3
    mid = out / "configurations" / "mid"
    mid.mkdir(parents=True, exist_ok=True)
    (mid / "simulation_result.json").write_bytes(b'{"mine": true}')
    before = _tree(out)
    capsys.readouterr()
    assert main(["project", "run", str(path), "--out-dir", str(out), "--progress", "none"]) == 64
    assert "simulation_result.json" in capsys.readouterr().err
    assert _tree(out) == before
