"""Model applicability of a tabulated thin lens used outside its validated domain.

The Engine checks a ``TABULATED_THIN_LENS``'s domain itself (``MODEL_WAVELENGTH_OUTSIDE_BAND``, ``MODEL_FIELD_ANGLE_EXCEEDED``,
``MODEL_BEAM_SIZE_EXCEEDED``, ``MODEL_POLARIZATION_NOT_VALIDATED``: run issues of status ``OUT_OF_MODEL_SCOPE``). A target whose
light crossed the lens outside that domain must not be decided: its raw diagnostic status (computed with the nearest table
row) is kept, its assessment is ``INDETERMINATE`` with the code in ``applicability_issue_codes``, exactly as for the other
model-scope issues (``local/validity.py``). Targets the out-of-domain light does not reach -- a plane before the lens, a
plane only a beam beside the lens reaches -- keep their decided assessment.
"""

from __future__ import annotations

import json
import math

import pytest

from os_optics.cli import main
from os_optics.local import json_api, run_optical_project
from os_optics.local.artifacts import verify_run_artifacts
from os_optics.results import load_run

from declared_helpers import declared_component, thin_lens_model, with_models
from local_helpers import plane, project, source
from test_tabulated_thin_lens import SOURCE_X, WAIST_MM, _aim, laser, ttl_model, ttl_project

# the reproduction's kind of target, a maximum waist radius behind the lens (1.8 used 0.01 mm; this fixture's 0.5 mm waist
# focuses to ~0.07 mm, so the bound is 1 mm, met at every wavelength and angle used here: only applicability decides)
WAIST_MAX_MM = 1.0
SUPPORTED = [{"metric": "waist_radius_x_mm", "rel_error": "0.01", "basis": "held-out wavelengths"},
             {"metric": "waist_distance_x_mm", "rel_error": "0.005", "basis": "held-out wavelengths"},
             {"metric": "transmittance", "abs_error": "0.15", "basis": "held-out wavelengths"}]


def _model(**fields) -> dict:
    return ttl_model(supported_metrics=SUPPORTED, **{"max_field_angle_deg": "2", **fields})


def _waist(target_id: str = "spot", observation_id: str = "P", **extra) -> dict:
    return {"target_id": target_id, "observation_id": observation_id, "metric": "waist_radius_x_mm", "maximum": WAIST_MAX_MM, **extra}


def _tilted(angle_deg: float, wavelength_nm: float = 532.0) -> dict:
    """A 532 nm beam aimed at the lens centre from 200 mm before it, ``angle_deg`` from the lens axis."""

    angle = math.radians(angle_deg)
    return laser(((wavelength_nm, 0.001),), position=(SOURCE_X, 0.0, -SOURCE_X * math.tan(angle)), quaternion=_aim(angle))


def _tilted_project(angle_deg: float, **kwargs) -> dict:
    """The lens Project lit at ``angle_deg``; its plane is wide enough to receive the tilted beam behind the lens."""

    document = ttl_project(src=_tilted(angle_deg), model=_model(), targets=[_waist()], **kwargs)
    document["observations"][0]["aperture"]["radius_mm"] = 100.0
    return document


def beside_and_before_project() -> dict:
    """450 nm through the lens to plane P (target ``spot``), crossing plane Q 100 mm before the lens (``before``); a 532 nm
    beam passing beside the lens (beyond its 12.7 mm edge) to the small plane R the 450 nm beam misses (``beside``)."""

    document = ttl_project(((450.0, 0.001),), model=_model(), targets=[
        _waist(source_id="SRC"),
        {"target_id": "before", "observation_id": "Q", "metric": "beam_radius_mm", "maximum": 5.0},
        {"target_id": "beside", "observation_id": "R", "metric": "beam_radius_mm", "source_id": "SIDE", "maximum": 5.0}])
    document["observations"] += [plane("Q", (-100.0, 0.0, 0.0), radius_mm=10.0), plane("R", (190.0, 0.0, 15.0), radius_mm=2.0)]
    document["components"].append(source("SIDE", position=(SOURCE_X, 0.0, 15.0), wavelength_nm=532.0, waist_mm=WAIST_MM))
    return document


def _rows(run) -> dict[str, dict]:
    return {row["target_id"]: row for row in run.summary()["targets"]}


def _assert_undecided(run, target_id: str, code: str) -> None:
    row = _rows(run)[target_id]
    assert row["status"] == "PASS" and row["value"] is not None  # the raw diagnostic result is unchanged
    assert row["assessment"] == "INDETERMINATE"
    assert row["applicability_issue_codes"] == [code] and code in row["issue_codes"]
    requirements = run.local_validation.simulation.outcome.requirements
    assert target_id in requirements.indeterminate and target_id not in requirements.met
    (issue,) = [i for i in run.model_scope_issues if i.code == code]
    assert (issue.instance_id, issue.surface_id) == ("ML", "lens") and target_id in issue.affected_target_ids
    assert "ML" in issue.affected_element_ids
    # the Engine's own issue is unchanged and still says the same
    assert any(i.code.value == code and i.location == "surface:ML/lens" for i in run.result.issues)


def _assert_met(run, target_id: str) -> None:
    row = _rows(run)[target_id]
    assert (row["status"], row["assessment"], row["applicability_issue_codes"]) == ("PASS", "MET", [])


# ------------------------------------------------------------------------------------------------ the reproduction


def test_a_wavelength_outside_the_table_leaves_the_spot_behind_the_lens_undecided(tmp_path, capsys) -> None:
    run = run_optical_project(ttl_project(((450.0, 0.001),), model=_model(), targets=[_waist()]), output_dir=tmp_path / "run")
    assert run.status.value == "OUT_OF_MODEL_SCOPE" and run.result.status.value == "OUT_OF_MODEL_SCOPE"
    _assert_undecided(run, "spot", "MODEL_WAVELENGTH_OUTSIDE_BAND")
    (issue,) = run.model_scope_issues
    assert issue.affected_observation_ids == ("P",) and issue.affected_target_ids == ("spot",)
    assert issue.message.startswith("450.0 nm light meets the tabulated thin lens ML/lens")
    # the run directory, the results reader and the CLI say the same, and the run directory verifies
    assert verify_run_artifacts(tmp_path / "run") == ()
    targets = load_run(tmp_path / "run").targets()
    (row,) = targets["targets"]
    assert (row["status"], row["assessment"], row["applicability_issue_codes"]) == ("PASS", "INDETERMINATE", ["MODEL_WAVELENGTH_OUTSIDE_BAND"])
    assert targets["outcome"]["requirements"]["indeterminate"] == ["spot"] and targets["outcome"]["requirements"]["met"] == []
    assert targets["in_model_scope"] is False
    assert {(f["source"], f["code"]) for f in targets["out_of_model_issues"]} == {
        ("ENGINE_ISSUE", "MODEL_WAVELENGTH_OUTSIDE_BAND"), ("MODEL_SCOPE", "MODEL_WAVELENGTH_OUTSIDE_BAND")}
    assert main(["results", "targets", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["targets"][0]["assessment"] == "INDETERMINATE"


def test_the_requirements_status_contract_reports_the_undecided_target(tmp_path, capsys) -> None:
    for wavelength, expected in ((450.0, 2), (532.0, 0)):
        document = tmp_path / f"p{wavelength:g}.json"
        document.write_text(json.dumps(ttl_project(((wavelength, 0.001),), model=_model(), targets=[_waist()])), encoding="utf-8")
        assert main(["project", "run", str(document), "--status-contract", "requirements"]) == expected
        assert main(["project", "run", str(document)]) == expected  # (the default contract)
        assert main(["project", "run", str(document), "--status-contract", "aggregate"]) == (3 if wavelength == 450.0 else 0)
        capsys.readouterr()


@pytest.mark.parametrize("angle_deg", [5.0, 8.0, 20.0])
def test_a_field_angle_beyond_the_validated_one_leaves_the_spot_undecided(angle_deg: float) -> None:
    run = run_optical_project(_tilted_project(angle_deg))
    assert run.status.value == "OUT_OF_MODEL_SCOPE"
    _assert_undecided(run, "spot", "MODEL_FIELD_ANGLE_EXCEEDED")


def test_beam_size_and_polarization_outside_the_domain_leave_the_spot_undecided() -> None:
    big = run_optical_project(ttl_project(((532.0, 0.001),), model=_model(max_beam_radius_mm="0.3"), targets=[_waist()]))
    _assert_undecided(big, "spot", "MODEL_BEAM_SIZE_EXCEEDED")
    crossed = laser(((532.0, 0.001),), polarization_deg=90.0)
    other = run_optical_project(ttl_project(src=crossed, model=_model(validated_polarizations=["LINEAR_X"]), targets=[_waist()]))
    _assert_undecided(other, "spot", "MODEL_POLARIZATION_NOT_VALIDATED")


# ------------------------------------------------------------------------------------------------ controls


def test_inside_the_domain_the_spot_is_met() -> None:
    run = run_optical_project(ttl_project(((532.0, 0.001),), model=_model(), targets=[_waist()]))
    assert run.status.value in ("PASS", "WARNING") and run.model_scope_issues == ()
    _assert_met(run, "spot")
    within = run_optical_project(_tilted_project(1.5))
    assert within.model_scope_issues == ()
    _assert_met(within, "spot")


def test_targets_the_out_of_domain_light_does_not_reach_stay_decided() -> None:
    """A plane the 450 nm beam crosses before the lens, and a plane only a 532 nm beam passing beside the lens reaches, keep
    their assessment; the spot behind the lens does not."""

    run = run_optical_project(beside_and_before_project())
    _assert_undecided(run, "spot", "MODEL_WAVELENGTH_OUTSIDE_BAND")
    _assert_met(run, "before")
    _assert_met(run, "beside")
    (issue,) = run.model_scope_issues
    assert issue.affected_observation_ids == ("P",) and issue.affected_target_ids == ("spot",)
    requirements = run.local_validation.simulation.outcome.requirements
    assert set(requirements.met) == {"before", "beside"} and requirements.indeterminate == ("spot",)


def test_a_field_angle_reached_only_on_a_later_pass_applies_from_that_pass() -> None:
    """``_tabulated_crossing``: the field-angle issue starts at the crossing the Engine judged beyond the validated angle."""

    from types import SimpleNamespace

    from os_optics.local.validity import _Collector, _tabulated_crossing

    segment = lambda surface, event, length, aoi=None: SimpleNamespace(end_surface_id=surface, end_event=event, length_mm=length,
                                                                       angle_of_incidence_deg=aoi)
    path = SimpleNamespace(path_id="S#0", geometric_path_length_mm=60.0,
                           segments=[segment("ML/lens", "thin_lens", 10.0, 0.5), segment("M/mirror", "reflect", 20.0, 3.0),
                                     segment("ML/lens", "thin_lens", 30.0, 6.0)])
    issue = lambda code, **ctx: SimpleNamespace(code=SimpleNamespace(value=code), numerical_context=ctx)
    found = _Collector(SimpleNamespace(paths=[], observations=[]), {})
    assert _tabulated_crossing(found, path, "ML/lens", issue("MODEL_FIELD_ANGLE_EXCEEDED", max_field_angle_deg=2.0)) == 60.0
    assert _tabulated_crossing(found, path, "ML/lens", issue("MODEL_WAVELENGTH_OUTSIDE_BAND")) == 10.0
    assert _tabulated_crossing(found, path, "other/lens", issue("MODEL_WAVELENGTH_OUTSIDE_BAND")) == 0.0  # not found: from the start


def test_an_ideal_thin_lens_outside_its_band_has_the_same_dependency_specific_applicability() -> None:
    """Another model type with a model-scope issue (a declared ``IDEAL_THIN_LENS``, checked by ``validity.py`` against its
    band): the target behind it is undecided, a plane before it is not."""

    targets = [{"target_id": "after", "observation_id": "P", "metric": "beam_radius_mm", "maximum": 5.0},
               {"target_id": "before", "observation_id": "Q", "metric": "beam_radius_mm", "maximum": 5.0}]
    document = project([source(wavelength_nm=2500.0, waist_mm=WAIST_MM), declared_component("TL", "lens", "thin", position=(0.0, 0.0, 0.0))],
                       [plane("P", (90.0, 0.0, 0.0), radius_mm=20.0), plane("Q", (-100.0, 0.0, 0.0), radius_mm=10.0)], targets)
    run = run_optical_project(with_models(document, [thin_lens_model()]))
    (issue,) = run.model_scope_issues
    assert (issue.instance_id, issue.code) == ("TL", "MODEL_WAVELENGTH_OUTSIDE_BAND") and issue.affected_target_ids == ("after",)
    rows = _rows(run)
    assert (rows["after"]["assessment"], rows["after"]["applicability_issue_codes"]) == ("INDETERMINATE", ["MODEL_WAVELENGTH_OUTSIDE_BAND"])
    assert (rows["before"]["assessment"], rows["before"]["applicability_issue_codes"]) == ("MET", [])


def test_the_browser_json_api_reports_the_same_applicability() -> None:
    """The json_api (the Designer's runtime, same code in Pyodide) gives the same assessment as the native run."""

    document = ttl_project(((450.0, 0.001),), model=_model(), targets=[_waist()])
    reply = json_api.run_project(document)
    assert reply["ok"] is True
    (row,) = reply["summary"]["targets"]
    assert (row["assessment"], row["applicability_issue_codes"]) == ("INDETERMINATE", ["MODEL_WAVELENGTH_OUTSIDE_BAND"])
    targets = json_api.run_results(reply["files"], {"kind": "targets"})["result"]
    assert targets["targets"][0]["assessment"] == "INDETERMINATE" and targets["in_model_scope"] is False


def test_network_export_refusal_is_unchanged_by_the_applicability() -> None:
    """The export still refuses a tabulated thin lens for its own reasons; the domain issues are not added to that refusal."""

    from os_optics.local import export_linear_optical_network
    from os_optics.network.errors import NetworkExportError

    with pytest.raises(NetworkExportError) as refused:
        export_linear_optical_network(ttl_project(((450.0, 0.001),), model=_model()))
    assert not any(p.message.startswith("MODEL_WAVELENGTH_OUTSIDE_BAND: ") for p in refused.value.problems)


def test_an_engine_issue_without_a_listed_path_applies_to_every_path_through_the_lens() -> None:
    """``_check_tabulated_domains``: an issue whose path the run does not list is not dropped; it applies from the first
    crossing of the lens on every path that crosses it (planes before the lens stay decided)."""

    from os_optics.local.lowering import lower_project
    from os_optics.local.catalog import load_project
    from os_optics.local.validity import element_scope_issues

    document = beside_and_before_project()
    run = run_optical_project(document)
    issues = [i.model_copy(update={"numerical_context": {**i.numerical_context, "path_id": "NOT-A-PATH"}})
              if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND" else i for i in run.result.issues]
    result = run.result.model_copy(update={"issues": issues})
    lowered = lower_project(load_project(document), None)
    (issue,) = element_scope_issues(result, lowered.elements, tabulated_domains=True)
    assert issue.code == "MODEL_WAVELENGTH_OUTSIDE_BAND" and issue.affected_observation_ids == ("P",)
    assert issue.affected_target_ids == ("spot",)


def test_a_table_given_at_its_rows_in_one_quantity_names_only_that_quantity() -> None:
    """Linear focal length, transmittance at the rows only: between rows only the transmittance is the nearest row's."""

    mixed = run_optical_project(ttl_project(((510.0, 0.001),), model=_model(transmittance_interpolation="POINTS_ONLY"), targets=[_waist()]))
    (issue,) = [i for i in mixed.result.issues if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    assert "the nearest row's transmittance was used." in issue.message and "focal length" not in issue.message
    _assert_undecided(mixed, "spot", "MODEL_WAVELENGTH_OUTSIDE_BAND")
    below = run_optical_project(ttl_project(((450.0, 0.001),), model=_model(), targets=[_waist()]))
    assert "the nearest row's focal length and transmittance were used." in below.model_scope_issues[0].message


def test_the_domain_sentences_end_with_the_next_action() -> None:
    angle = run_optical_project(_tilted_project(8.0)).model_scope_issues[0].message
    assert angle.endswith("validated up to 2 deg. Keep the field angle within that limit, or use a response model validated for it.")
    size = run_optical_project(ttl_project(((532.0, 0.001),), model=_model(max_beam_radius_mm="0.3"))).model_scope_issues[0].message
    assert size.endswith("Use a smaller beam at the lens, or a response model validated for that beam size.")
    crossed = laser(((532.0, 0.001),), polarization_deg=90.0)
    other = run_optical_project(ttl_project(src=crossed, model=_model(validated_polarizations=["LINEAR_X"]))).model_scope_issues[0].message
    assert other.endswith("the polarization its response was validated for. Use light polarized that way, or a response model "
                          "validated for the arriving polarization.")


# ------------------------------------------------------------------------------------------------ polarization components


def _unpolarized_through_a_lens_validated_for_x() -> dict:
    """Unpolarized 532 nm light through the lens validated for LINEAR_X only (its degree of polarization is a supported
    metric): the component along Y meets the lens outside its validated polarizations. Three degree-of-polarization
    targets on P name the one route of the light by component 0's path id, by component 1's, and by its source."""

    by = lambda target_id, **selector: {"target_id": target_id, "observation_id": "P", "metric": "degree_of_polarization",
                                         "maximum": 0.5, **selector}
    src = laser(((532.0, 0.001),))
    src["source_model"]["polarization"] = {"kind": "UNPOLARIZED"}
    supported = [*SUPPORTED, {"metric": "degree_of_polarization", "abs_error": "0.01", "basis": "held-out wavelengths"}]
    return ttl_project(src=src, model=ttl_model(supported_metrics=supported, validated_polarizations=["LINEAR_X"]), targets=[
        by("dop_c0", parameters=[{"key": "path_id", "value": "SRC#0"}]),
        by("dop_c1", parameters=[{"key": "path_id", "value": "SRC#0:p1"}]),
        by("dop_src", source_id="SRC")])


def test_a_polarization_target_reads_both_components_of_its_route_whichever_path_id_names_it(tmp_path) -> None:
    """The degree of polarization of a route is the light of both components (``mixture_metric``), so the finding on the
    component outside the lens's validated polarization marks every target reading that route: by component 0's path id,
    by component 1's, by the source -- on the native run, its exported run directory and the json_api evaluation."""

    code = "MODEL_POLARIZATION_NOT_VALIDATED"
    names = {"dop_c0", "dop_c1", "dop_src"}
    document = _unpolarized_through_a_lens_validated_for_x()
    run = run_optical_project(document, output_dir=tmp_path / "run")
    (issue,) = [i for i in run.model_scope_issues if i.code == code]
    assert set(issue.affected_target_ids) == names
    reply = json_api.evaluate_project(document, ["readings", "requirements"])
    assert reply["ok"], reply
    (listed,) = [i for i in reply["model_scope_issues"] if i["code"] == code]
    assert set(listed["affected_target_ids"]) == names
    surfaces = {"run_optical_project": _rows(run),
                "exported run": {t["target_id"]: t for t in load_run(tmp_path / "run").targets()["targets"]},
                "evaluate_project": {t["target_id"]: t for t in reply["targets"]}}
    values = set()
    for surface, rows in surfaces.items():
        for name in names:
            row = rows[name]
            assert row["assessment"] == "INDETERMINATE" and code in row["issue_codes"], (surface, name)
            assert row.get("applicability_issue_codes", [code]) == [code], (surface, name)  # (the reply states issue_codes)
            basis = row["value_basis"]["value"]
            assert basis["kind"] == "ESTIMATE" and code in {c["code"] for c in basis["validity"]}, (surface, name)
            values.add(row["value"])
    assert len(values) == 1  # (one route, one value)
    assert run.local_validation.simulation.outcome.requirements.indeterminate == tuple(sorted(names))
    assert reply["requirements"]["indeterminate"] == sorted(names) and reply["requirements"]["met"] == []
