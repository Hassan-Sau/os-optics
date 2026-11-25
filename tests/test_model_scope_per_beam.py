"""A model-scope finding marks the targets that read the light it concerns, beam by beam.

A published model used outside its validity (here an ideal thin lens published for 400-1100 nm, lit at 1200 nm) leaves
the numbers of the light that met it outside the model's scope. A target that selects other light on the same plane -- a
source, a wavelength, a beam pair -- keeps its own assessment: one wavelength outside a lens's band does not invalidate
another wavelength's spot. A target that selects no beam reads every beam on the plane and is marked when any of them is.

The fixture ``fixtures/scope/thin_lens_band.project.json``: a 532 nm Gaussian source ``SRC``, an ideal thin lens published
for 400-1100 nm, planes ``early`` (x = 50 mm) and ``image`` (x = 100 mm).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from os_optics.errors.status import QuantityState
from os_optics.local import EvaluationSession, json_api
from os_optics.local.json_api import _evaluation_reply

HERE = Path(__file__).resolve().parent
OUTSIDE = "MODEL_WAVELENGTH_OUTSIDE_BAND"
OUTPUTS = ("readings", "requirements")
IMAGING = {"launch_grid": 16, "diffraction": False, "convergence": False}


def lens(wavelength: float = 1200) -> dict:
    document = json.loads((HERE / "fixtures" / "scope" / "thin_lens_band.project.json").read_text(encoding="utf-8"))
    document["components"][0]["source_model"]["wavelength_nm"] = wavelength
    return document


def add_source(document: dict, name: str, wavelength: float, y: float = 0.0) -> None:
    source = copy.deepcopy(document["components"][0])
    source["instance_id"] = name
    source["custom_part"]["custom_part_key"] = name.lower()
    source["source_model"]["wavelength_nm"] = wavelength
    source["nominal_pose"]["position_mm"][1] = y
    document["components"].append(source)


def spot(target_id: str, observation_id: str = "early", maximum: float = 1.0, **selector) -> dict:
    return {"target_id": target_id, "metric": "spot_rms_mm", "observation_id": observation_id, "maximum": maximum,
            "unit": "mm", **selector}


def assessments(reply: dict) -> dict[str, str]:
    assert reply["ok"], reply
    return {t["target_id"]: t["assessment"] for t in reply["targets"]}


def mixed(source_wavelength: float = 1200) -> dict:
    """SRC at ``source_wavelength`` and GOOD at 532 nm through the same lens onto the same planes, a plane ``before`` the
    lens, and targets selecting each beam, the pair, and no beam."""

    document = lens(source_wavelength)
    add_source(document, "GOOD", 532, y=1.0)
    before = copy.deepcopy(document["observations"][0])
    before["observation_id"] = "before"
    before["pose"]["position_mm"][0] = -50
    document["observations"].append(before)
    document["targets"] = [
        spot("bad", source_id="SRC"),
        spot("good_by_source", source_id="GOOD"),
        spot("good_by_wavelength", wavelength_nm=532),
        spot("bad_before_lens", observation_id="before", source_id="SRC"),
        {"target_id": "pair", "metric": "centroid_separation_mm", "observation_id": "early", "source_id": "SRC",
         "source_id_b": "GOOD", "wavelength_nm": source_wavelength, "wavelength_nm_b": 532, "maximum": 100.0, "unit": "mm"},
        {"target_id": "any_beam", "metric": "power_W", "observation_id": "early", "minimum": 0.0, "unit": "W"},
    ]
    return document


def parameters(**values) -> list[dict]:
    """A target's parameters as the Project states them."""

    return [{"key": key, "value": value} for key, value in values.items()]


def affected(reply: dict) -> set[str]:
    return {t for issue in reply["model_scope_issues"] for t in issue["affected_target_ids"]}


def test_the_out_of_band_beam_and_a_pair_that_reads_it_are_marked():
    reply = json_api.evaluate_project(mixed(), list(OUTPUTS))
    got = assessments(reply)
    assert got["bad"] == "INDETERMINATE"
    assert got["pair"] == "INDETERMINATE"  # one operand is the out-of-band beam
    assert {"bad", "pair"} <= affected(reply)
    issue = next(i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE)
    assert issue["instance_id"] == "LENS" and set(issue["affected_observation_ids"]) == {"early", "image"}


def test_a_plane_before_the_out_of_band_optic_stays_decided():
    reply = json_api.evaluate_project(mixed(), list(OUTPUTS))
    assert assessments(reply)["bad_before_lens"] == "MET"
    assert "bad_before_lens" not in affected(reply)


@pytest.mark.parametrize("target_id", ["good_by_source", "good_by_wavelength"])
def test_a_beam_in_band_on_the_same_plane_keeps_its_assessment(target_id):
    reply = json_api.evaluate_project(mixed(), list(OUTPUTS))
    assert assessments(reply)[target_id] == "MET"
    assert target_id not in affected(reply)
    target = next(t for t in reply["targets"] if t["target_id"] == target_id)
    assert OUTSIDE not in target["issue_codes"]
    # (the independent reference: the same design with SRC moved into the band, 1000 nm, assesses the same target the
    # same way, on the same number)
    control = json_api.evaluate_project(mixed(1000), list(OUTPUTS))
    assert assessments(control)[target_id] == "MET" and control["model_scope_issues"] == []
    assert target["value"] == next(t for t in control["targets"] if t["target_id"] == target_id)["value"]


def test_a_target_that_selects_no_beam_reads_every_beam_on_the_plane():
    reply = json_api.evaluate_project(mixed(), list(OUTPUTS))
    assert assessments(reply)["any_beam"] == "INDETERMINATE"
    assert "any_beam" in affected(reply)


def test_a_planes_monitor_states_the_value_basis_of_the_targets_a_finding_names(tmp_path, capsys):
    """RunResults.monitor(plane)["targets"] states each target's value basis as targets() does: the out-of-band beam, the
    pair that reads it and the target that selects no beam carry the finding's ESTIMATE -- through the session's results,
    the JSON results query and the CLI on the exported run; an in-band beam's target carries none."""

    from os_optics.cli import main
    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = mixed()
    run_optical_project(document, output_dir=tmp_path / "run")
    assert main(["results", "monitor", str(tmp_path / "run"), "early"]) == 0
    from_cli = json.loads(capsys.readouterr().out)["targets"]
    run = json_api.run_project(document)
    assert run["ok"], run
    query = json_api.run_results(run["files"], {"kind": "monitor", "monitor": "early"})
    assert query["ok"], query
    session = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    expected = {t["target_id"]: t for t in session.targets()["targets"]}
    for surface, rows in (("results().monitor()", session.monitor("early")["targets"]), ("results query", query["result"]["targets"]),
                          ("CLI results monitor", from_cli), ("load_run().monitor()", load_run(tmp_path / "run").monitor("early")["targets"])):
        rows = {t["target_id"]: t for t in rows}
        for target_id in ("bad", "pair", "any_beam"):
            basis = rows[target_id]["value_basis"]["value"]
            assert rows[target_id]["assessment"] == "INDETERMINATE" and basis["kind"] == "ESTIMATE", (surface, target_id)
            assert [c["code"] for c in basis["validity"]] == [OUTSIDE], (surface, target_id)
            assert rows[target_id]["value_basis"] == expected[target_id]["value_basis"], (surface, target_id)
        for target_id in ("good_by_source", "good_by_wavelength"):
            assert rows[target_id].get("value_basis") == expected[target_id].get("value_basis"), (surface, target_id)
            assert OUTSIDE not in json.dumps(rows[target_id].get("value_basis")), (surface, target_id)


def test_the_monitors_export_states_the_quantity_state_monitors_states(tmp_path):
    """The planes table (``table('monitors')`` and its csv / json / jsonl exports) states each plane's quantity state as
    ``monitors()`` and ``monitor()`` do: the planes behind the out-of-band lens OUT_OF_MODEL_SCOPE, the plane before it
    KNOWN -- of a loaded run and of a session's results alike."""

    import csv
    import io

    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = mixed()
    run_optical_project(document, output_dir=tmp_path / "run")
    expected = {"early": "OUT_OF_MODEL_SCOPE", "image": "OUT_OF_MODEL_SCOPE", "before": "KNOWN"}
    for name, results in (("load_run", load_run(tmp_path / "run")),
                          ("EvaluationSession", EvaluationSession().evaluate(document, outputs=OUTPUTS).results())):
        assert {m["observation_id"]: m["quantity_state"] for m in results.monitors()} == expected, name
        assert {plane: results.monitor(plane)["quantity_state"] for plane in expected} == expected, name
        rows, _summary = results.table("monitors")
        assert {r["observation_id"]: r["quantity_state"] for r in rows} == expected, name
        from_csv = list(csv.DictReader(io.StringIO(results.export("monitors", "csv").decode("utf-8"))))
        assert {r["observation_id"]: r["quantity_state"] for r in from_csv} == expected, name
        from_json = json.loads(results.export("monitors", "json"))
        assert {r["observation_id"]: r["quantity_state"] for r in from_json["rows"]} == expected, name
        from_jsonl = [json.loads(line) for line in results.export("monitors", "jsonl").decode("utf-8").splitlines() if line]
        assert {r["observation_id"]: r["quantity_state"] for r in from_jsonl} == expected, name


def test_a_plane_a_finding_names_carries_its_check_on_its_power_and_on_the_light_it_reaches(tmp_path):
    """A plane the results mark OUT_OF_MODEL_SCOPE because of a model-scope finding states why beside its total_power_W: an
    ESTIMATE with the finding's check, the entry of the target that reads the same number (any_beam) -- on every surface
    that carries the plane's power. Of the beams and lines on it, only the light the finding reaches (SRC#0, 1200 nm)
    carries the check; GOOD#0 and the 532 nm line stay the model's value, and the plane before the lens has no entry."""

    import csv
    import io

    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = mixed()
    run_optical_project(document, output_dir=tmp_path / "run")
    session = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    run = json_api.run_project(document)
    assert reply["ok"] and run["ok"], (reply, run)
    (any_beam,) = [t["value_basis"]["value"] for t in reply["targets"] if t["target_id"] == "any_beam"]
    assert [c["code"] for c in any_beam["validity"]] == [OUTSIDE]

    def cell(value):
        return (json.loads(value) if value else None) if isinstance(value, str) else value

    planes: dict[str, dict[str, dict]] = {}
    for name, results in (("EvaluationSession", session), ("load_run", load_run(tmp_path / "run"))):
        planes[f"{name}.monitors()"] = {m["observation_id"]: m for m in results.monitors()}
        planes[f"{name}.monitor()"] = {p: results.monitor(p) for p in ("early", "image", "before")}
        planes[f"{name}.table"] = {r["observation_id"]: {"value_basis": r["value_basis"]} for r in results.table("monitors")[0]}
        planes[f"{name}.csv"] = {r["observation_id"]: {"value_basis": cell(r["value_basis"]) or None}
                                 for r in csv.DictReader(io.StringIO(results.export("monitors", "csv").decode("utf-8")))}
        planes[f"{name}.json"] = {r["observation_id"]: r for r in json.loads(results.export("monitors", "json"))["rows"]}
    planes["evaluate_project"] = {m["observation_id"]: m for m in reply["monitors"]}
    planes["run_project summary"] = {m["observation_id"]: m for m in run["summary"]["observations"]}
    for surface, rows in planes.items():
        for plane in ("early", "image"):
            entry = (rows[plane].get("value_basis") or {}).get("total_power_W")
            assert entry == any_beam, (surface, plane, entry)
        assert not (rows["before"].get("value_basis") or {}).get("total_power_W"), surface

    # the light the finding reaches: SRC#0's arrival, the 1200 nm line; not GOOD#0, not the 532 nm line
    def arrival_entries(rows, key):
        return {a["path_id"] if "path_id" in a else a["beam"]["path_id"]: (a.get("value_basis") or {}).get(key) for a in rows}

    for plane in ("early", "image"):
        for surface, rows, key in (("monitors()", session.monitors()[["early", "image"].index(plane)]["arrivals"], "power_W"),
                                   ("monitor()", session.monitor(plane)["arrivals"], "beam.power_W"),
                                   ("arrivals table", [r for r in session.table("arrivals")[0] if r["observation_id"] == plane], "power_W"),
                                   ("evaluate_project", planes["evaluate_project"][plane]["arrivals"], "power_W"),
                                   ("run_project summary", planes["run_project summary"][plane]["arrivals"], "power_W")):
            got = arrival_entries(rows, key)
            assert got["SRC#0"] is not None and [c["code"] for c in got["SRC#0"]["validity"]] == [OUTSIDE], (surface, plane)
            assert got["SRC#0"]["kind"] == "ESTIMATE", (surface, plane)
            assert got["GOOD#0"] is None, (surface, plane)
        lines = {s["wavelength_nm"]: (s.get("value_basis") or {}).get("power_W") for s in session.monitor(plane)["power_by_wavelength"]}
        assert [c["code"] for c in lines[1200.0]["validity"]] == [OUTSIDE] and lines[532.0] is None, plane
    before = session.monitor("before")
    assert not any(a.get("value_basis") for a in before["arrivals"]) and not any(s.get("value_basis") for s in before["power_by_wavelength"])


def test_an_arrival_a_finding_reaches_carries_its_check_on_every_number_of_a_kind_the_finding_touches(tmp_path):
    """The thin-lens finding touches every kind of quantity: the out-of-band beam SRC#0's Gaussian radii, its position on
    the plane and its spot size carry the same ESTIMATE as the targets that read them (beam_radius_x_mm, position_u_mm,
    spot_rms_mm on SRC) -- on monitor(), the arrivals table and its CSV export, the evaluate_project reply and the run
    summary; GOOD#0 carries none."""

    import csv
    import io

    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = mixed()
    document["targets"] = [
        {"target_id": "radius", "metric": "beam_radius_x_mm", "observation_id": "early", "source_id": "SRC", "maximum": 100.0, "unit": "mm"},
        {"target_id": "spot", "metric": "spot_rms_mm", "observation_id": "early", "source_id": "SRC", "maximum": 100.0, "unit": "mm"},
        {"target_id": "u", "metric": "position_u_mm", "observation_id": "early", "source_id": "SRC", "maximum": 100.0, "unit": "mm"}]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    run = json_api.run_project(document)
    assert reply["ok"] and run["ok"], (reply, run)
    entries = {t["target_id"]: t["value_basis"]["value"] for t in reply["targets"]}
    assert all(e["kind"] == "ESTIMATE" and [c["code"] for c in e["validity"]] == [OUTSIDE] for e in entries.values()), entries
    entry = entries["radius"]
    run_optical_project(document, output_dir=tmp_path / "run")
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    stored = load_run(tmp_path / "run")

    def by_path(rows, path_key=lambda r: r["path_id"]):
        return {path_key(r): r for r in rows}

    monitor = by_path(results.monitor("early")["arrivals"], lambda a: a["beam"]["path_id"])
    assert monitor["SRC#0"]["beam"]["gaussian"]["value_basis"]["*"]["validity"] == entry["validity"]
    for field in ("uv_mm", "pupil_rms_radius_mm", "beam.position_mm"):
        assert monitor["SRC#0"]["value_basis"][field] == entry, field
    assert not monitor["GOOD#0"].get("value_basis") and not monitor["GOOD#0"]["beam"]["gaussian"].get("value_basis")
    for surface, rows in (("arrivals table", results.table("arrivals", monitor="early")[0]),
                          ("load_run arrivals csv", [{**r, "value_basis": json.loads(r["value_basis"]) if r["value_basis"] else None}
                                                     for r in csv.DictReader(io.StringIO(stored.export("arrivals", "csv").decode("utf-8")))
                                                     if r["observation_id"] == "early"])):
        rows = by_path(rows)
        for column in ("beam_radius_x_mm", "u_mm", "v_mm", "pupil_rms_radius_mm", "position_x_mm"):
            assert rows["SRC#0"]["value_basis"][column]["kind"] == "ESTIMATE", (surface, column)
            assert rows["SRC#0"]["value_basis"][column]["validity"] == entry["validity"], (surface, column)
        assert not rows["GOOD#0"]["value_basis"], surface
    for surface, planes in (("evaluate_project", reply["monitors"]), ("run summary", run["summary"]["observations"])):
        (early,) = [m for m in planes if m["observation_id"] == "early"]
        arrivals = by_path(early["arrivals"])
        for key in ("beam_radius_x_mm", "position_mm"):
            assert arrivals["SRC#0"]["value_basis"][key]["kind"] == "ESTIMATE", (surface, key)
        assert not arrivals["GOOD#0"].get("value_basis"), surface


def test_a_coefficient_finding_on_an_arrival_marks_its_power_and_polarization_and_leaves_its_geometry():
    """A coating finding (COATING_WAVELENGTH_OUTSIDE_BANDS) touches the power and the polarization of the light behind it,
    not where it goes: on the arrival it lists, the power and the Stokes parameters carry its check; the Gaussian state, the
    position and the direction on the plane do not, as a target on them is not marked."""

    from os_optics.results.basis import scoped_observation

    document = lens(532)
    result = EvaluationSession().evaluate(document, outputs=OUTPUTS).results().result
    (early,) = [o for o in result.observations if o.observation_id == "early"]
    (arrival,) = early.arrivals
    check = {"parameter": "wavelength_nm", "unit": "nm", "minimum": 600.0, "maximum": 1100.0, "observed": 532.0, "within": False,
             "at": ["LENS"], "code": "COATING_WAVELENGTH_OUTSIDE_BANDS"}
    finding = {"code": "COATING_WAVELENGTH_OUTSIDE_BANDS", "instance_id": "LENS", "affected_observation_ids": ["early"],
               "affected_arrivals": [{"observation_id": "early", "path_id": arrival.beam.path_id, "crossing_index": 0}],
               "validity": [check]}
    scoped = scoped_observation(early, [finding]).arrivals[0]
    bases = scoped.value_basis or {}
    for field in ("beam.power_W", "polarization_state.s1", "polarization_state.s3", "pupil_rms_radius_mm"):
        assert bases[field].kind.value == "ESTIMATE" and bases[field].validity[0].code == "COATING_WAVELENGTH_OUTSIDE_BANDS", field
    for field in ("uv_mm", "direction_uvn", "beam.position_mm", "beam.direction", "aoi_deg"):
        assert field not in bases, field
    assert scoped.beam.gaussian.value_basis == arrival.beam.gaussian.value_basis
    thin_lens = scoped_observation(early, [{**finding, "code": OUTSIDE, "validity": [{**check, "code": OUTSIDE}]}]).arrivals[0]
    assert {"uv_mm", "beam.position_mm", "polarization_state.s1"} <= set(thin_lens.value_basis)
    assert thin_lens.beam.gaussian.value_basis["*"].kind.value == "ESTIMATE"


def test_a_coherent_sum_of_light_a_finding_reaches_carries_its_check_and_one_of_in_band_light_keeps_its_own(tmp_path):
    """Two 1200 nm beams (SRC, SRC2: outside the lens's band) and two 532 nm beams (GOOD, ALSO: inside it) on one plane. The
    coherent sum of the 1200 nm pair -- the number the coherent_power_W target at 1200 nm reads -- carries the target's
    ESTIMATE on monitor(), superpositions(), the superpositions table and its CSV export, and the evaluate_project reply,
    and so does their interference record; the sum and the record of the 532 nm pair keep their own entries."""

    import csv
    import io

    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = lens(1200)
    add_source(document, "SRC2", 1200, y=0.5)
    add_source(document, "GOOD", 532, y=1.0)
    add_source(document, "ALSO", 532, y=1.5)
    document["targets"] = [{"target_id": "coh", "metric": "coherent_power_W", "observation_id": "early", "wavelength_nm": 1200,
                            "minimum": 0.0, "unit": "W"},
                           {"target_id": "coh532", "metric": "coherent_power_W", "observation_id": "early", "wavelength_nm": 532,
                            "minimum": 0.0, "unit": "W"},
                           {"target_id": "vis", "metric": "fringe_visibility", "observation_id": "early", "source_id": "SRC",
                            "source_id_b": "SRC2", "minimum": 0.0, "unit": "1"},
                           {"target_id": "vis532", "metric": "fringe_visibility", "observation_id": "early", "source_id": "GOOD",
                            "source_id_b": "ALSO", "minimum": 0.0, "unit": "1"}]
    reply = json_api.evaluate_project(document, [*OUTPUTS, "beam_pairs"])
    assert reply["ok"], reply
    entries = {t["target_id"]: t.get("value_basis") for t in reply["targets"]}
    entry = entries["coh"]["value"]
    assert entry["kind"] == "ESTIMATE" and [c["observed"] for c in entry["validity"]] == [1200.0]
    assert not entries["coh532"] or OUTSIDE not in json.dumps(entries["coh532"])
    run_optical_project(document, output_dir=tmp_path / "run")
    results = EvaluationSession().evaluate(document, outputs=(*OUTPUTS, "beam_pairs")).results()
    stored = load_run(tmp_path / "run")

    def sums(rows):
        out = {r["wavelength_nm"]: ((r.get("value_basis") or {}) if not isinstance(r.get("value_basis"), str)
                                    else json.loads(r["value_basis"] or "{}")) for r in rows}
        return {float(w): v or {} for w, v in out.items()}

    (early,) = [m for m in reply["monitors"] if m["observation_id"] == "early"]
    surfaces = {
        "monitor()": sums(results.monitor("early")["superpositions"]),
        "superpositions()": sums(results.superpositions("early")),
        "table": sums(results.table("superpositions", monitor="early")[0]),
        "load_run csv": sums([r for r in csv.DictReader(io.StringIO(stored.export("superpositions", "csv").decode("utf-8")))
                              if r["observation_id"] == "early"]),
        "evaluate_project": sums(early["superpositions"]),
    }
    for surface, by_wavelength in surfaces.items():
        out_of_band = by_wavelength[1200.0]
        for field in ("total_power_W", "incoherent_power_W"):
            assert out_of_band[field]["kind"] == "ESTIMATE", (surface, field)
            assert [c["code"] for c in out_of_band[field]["validity"]] == [OUTSIDE], (surface, field)
        assert OUTSIDE not in json.dumps(by_wavelength[532.0]), surface
    marked = {frozenset((i["path_id_a"], i["path_id_b"])): OUTSIDE in json.dumps(i.get("value_basis") or {})
              for i in results.monitor("early")["interference"]}
    assert marked == {frozenset(("SRC#0", "SRC2#0")): True, frozenset(("GOOD#0", "ALSO#0")): False}, marked


_DATASHEET = {"kind": "DATASHEET", "reference": "photodiode datasheet rev B"}
_DETECTOR = {"detector": {
    "quantum_efficiency": {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 1300.0, "qe": 0.8}], "source": _DATASHEET},
    "electrical": {"bandwidth_Hz": 1e4, "bandwidth_definition": "FIRST_ORDER_3DB", "dark_current_A": 1e-12,
                   "load_resistance_ohm": 1e8, "temperature_K": 295.0, "source": _DATASHEET}}}
#: The detector's numbers that do not read the plane's light.
_UNLIT = ("detector_bandwidth_Hz", "noise_equivalent_bandwidth_Hz", "dark_noise_density_A_per_rtHz", "dark_noise_current_A")


def test_a_detector_on_light_a_finding_reaches_carries_its_check_on_every_number_that_reads_that_light(tmp_path):
    """An absorbing detector on ``early``, behind the lens lit outside its band, and a transparent one on ``before``, ahead
    of it. The detector_snr target on ``early`` is an ESTIMATE with the wavelength check; the same number in the detector's
    reading -- ``detection()``, ``monitor()["detection"]``, the JSON results ``detection`` query, ``load_run().detection()`` and the
    CLI -- carries the same check in ``value_basis["value"]`` on every quantity that reads the plane's light (detected
    power, photocurrent, shot and total noise, SNR, NEP), not on the bandwidths or the dark noise. ``power_state`` stays the
    plane's. The detector ahead of the lens reads light no finding reaches and stays unmarked."""

    from os_optics.cli import main
    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = lens(1200)
    before = copy.deepcopy(document["observations"][0])
    before["observation_id"] = "before"
    before["pose"]["position_mm"][0] = -50
    document["observations"].append(before)
    for plane in document["observations"]:
        if plane["observation_id"] in ("early", "before"):
            plane["absorbing"] = plane["observation_id"] == "early"
            plane["detection"] = copy.deepcopy(_DETECTOR)
    document["targets"] = [{"target_id": "snr", "metric": "detector_snr", "observation_id": "early", "minimum": 1.0, "unit": "1"},
                           {"target_id": "snr_before", "metric": "detector_snr", "observation_id": "before", "minimum": 1.0,
                            "unit": "1"}]
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    targets = {t["target_id"]: t for t in results.targets()["targets"]}
    entry = targets["snr"]["value_basis"]["value"]
    assert targets["snr"]["assessment"] == "INDETERMINATE" and entry["kind"] == "ESTIMATE"
    assert [(c["code"], c["observed"]) for c in entry["validity"]] == [(OUTSIDE, 1200.0)]
    assert targets["snr_before"]["assessment"] == "MET" and "value_basis" not in targets["snr_before"]
    run_optical_project(document, output_dir=tmp_path / "run")
    files = {p.name: p.read_bytes() for p in (tmp_path / "run").iterdir() if p.is_file()}
    stored = load_run(tmp_path / "run")

    def readings(plane: str) -> dict[str, list[dict]]:
        out = {"detection()": results.detection(plane)[0]["quantities"],
               "monitor()": results.monitor(plane)["detection"]["quantities"],
               "json results": json_api.run_results(files, {"kind": "detection", "monitor": plane})["result"][0]["quantities"],
               "load_run": stored.detection(plane)[0]["quantities"]}
        assert main(["results", "detection", str(tmp_path / "run"), "--monitor", plane]) == 0
        return out

    for surface, quantities in readings("early").items():
        by_name = {q["name"]: q for q in quantities}
        assert by_name["detector_snr"]["value"] == targets["snr"]["value"], surface
        for name in ("detected_power_W", "photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr",
                     "noise_equivalent_power_W_per_rtHz"):
            got = by_name[name]["value_basis"]["value"]
            assert got["kind"] == "ESTIMATE" and got["validity"] == entry["validity"], (surface, name)
        for name in _UNLIT:
            assert "value_basis" not in by_name[name], (surface, name)
    engine = next(o for o in results.result.observations if o.observation_id == "early").detection
    assert results.detection("early")[0]["power_state"] == engine.power_state.value
    for surface, quantities in readings("before").items():
        assert OUTSIDE not in json.dumps(quantities), surface


def test_a_polarization_mixture_of_light_a_finding_reaches_carries_its_check_and_keeps_its_values():
    """A partially polarized 1200 nm source (degree of polarization 0.5) through the lens published for 400-1100 nm, a
    degree_of_polarization target on ``early`` and a plane ``before`` the lens. The mixture of the route on ``early`` keeps
    its Stokes values, its degree of polarization 0.5 and its quantity_state, and carries the target's ESTIMATE on each of
    them (``monitor()["polarization_mixtures"]``); the mixture on ``before``, whose light met no lens yet, carries none."""

    document = lens(1200)
    document["components"][0]["source_model"]["polarization"] = {
        "kind": "PARTIAL", "degree_of_polarization": 0.5, "polarized": {"kind": "LINEAR", "angle_deg": 0.0}}
    before = copy.deepcopy(document["observations"][0])
    before["observation_id"] = "before"
    before["pose"]["position_mm"][0] = -50
    document["observations"].append(before)
    document["targets"] = [{"target_id": "dop", "metric": "degree_of_polarization", "observation_id": "early", "minimum": 0.0,
                            "unit": "1"}]
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    (target,) = results.targets()["targets"]
    entry = target["value_basis"]["value"]
    assert target["assessment"] == "INDETERMINATE" and entry["kind"] == "ESTIMATE"
    (mixture,) = results.monitor("early")["polarization_mixtures"]
    assert mixture["quantity_state"] == "KNOWN" and mixture["degree_of_polarization"] == target["value"] == 0.5
    assert mixture["stokes_W"][0] > 0.0
    for field in ("stokes_W", "degree_of_polarization", "polarization_state.s1", "polarization_state.degree_of_polarization"):
        got = mixture["value_basis"][field]
        assert got["kind"] == "ESTIMATE" and got["validity"] == entry["validity"], field
    (unmarked,) = results.monitor("before")["polarization_mixtures"]
    assert "value_basis" not in unmarked and unmarked["degree_of_polarization"] == 0.5


def _focal_request(wavelength: float) -> dict:
    from test_focal_field import request

    body = request(0.5, wavelength=wavelength, sampling={"theta_nodes": 48, "phi_nodes": 48, "grid_cells": 64})
    body.pop("polarization")
    body["illumination"] = {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}
    return body


def _pupil_request(wavelength: float) -> dict:
    return {"wavelength_nm": wavelength, "focal_length_mm": 100.0,
            "sampling": {"pixel_count": [128, 128], "pitch_mm": [0.05, 0.05], "fft_size": [1024, 1024]},
            "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.3, 0.3]}}


_CAMERA = {"pixel_pitch_mm": [0.01, 0.01], "pixel_count": [8, 8], "fill_factor": 1.0, "exposure_s": 0.001,
           "quantum_efficiency": {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 1300.0, "qe": 0.8}],
                                  "source": _DATASHEET}}
#: The numbers of a focal field that read the light's power.
_FOCAL_POWERS = ("incident_power_W", "pupil_power_W", "transmitted_power_W", "focal_plane_flux_W")


def _checks(entry: dict | None) -> list:
    return [(c["code"], c["observed"]) for c in (entry or {}).get("validity", ())]


def test_a_handoff_of_a_beam_a_finding_reaches_carries_its_check_into_every_field_map_and_reading_built_from_it(tmp_path):
    """One 1200 nm beam through the lens published for 400-1100 nm. Its handoff keeps the beam a TEM00 field (its numbers
    and ``tem00_field_applicable`` stay), lists the finding in ``validity.model_scope``, extends the statement, and carries
    the finding's ESTIMATE on its power, chief ray and Jones vector (``value_basis``) and its Gaussian state
    (``gaussian.value_basis["*"]``) -- on ``handoff()``, the JSON ``handoff`` query and the handoff export. The focal field
    built from it reads its powers OUT_OF_MODEL_SCOPE with that ESTIMATE, the pupil field and the irradiance map carry it
    on their maps and quantities (``model_scope``), and a camera reading the map on the numbers read from the light, not
    on its photon energy or dark noise."""

    from os_optics.local import run_optical_project

    document = lens(1200)
    results = EvaluationSession().evaluate(document, outputs=("complete",)).results()
    run_optical_project(document, output_dir=tmp_path / "run")
    files = {p.name: p.read_bytes() for p in (tmp_path / "run").iterdir() if p.is_file()}
    want = [(OUTSIDE, 1200.0)]
    for surface, doc in (("handoff()", results.handoff("early")),
                         ("json", json_api.run_results(files, {"kind": "handoff", "monitor": "early"})["result"])):
        # model_scope and value_basis are fields of handoff schema 1.2
        assert (doc["schema"], doc["schema_version"]) == ("opensci.transverse_handoff", "1.2"), surface
        (beam,) = doc["beams"]
        validity = beam["validity"]
        assert (validity["beam_parameters"], validity["tem00_field_applicable"]) == ("TEM00", True), surface
        assert [(f["code"], f["instance_id"]) for f in validity["model_scope"]] == [(OUTSIDE, "LENS")], surface
        assert OUTSIDE in validity["statement"] and beam["power_W"] is not None, surface
        for key in ("power_W", "chief_ray.uv_mm", "chief_ray.direction_uvn", "polarization.jones"):
            assert beam["value_basis"][key]["kind"] == "ESTIMATE" and _checks(beam["value_basis"][key]) == want, (surface, key)
        assert _checks(beam["gaussian"]["value_basis"]["*"]) == want, surface
    (row,) = results.table("handoff", monitor="early")[0]
    assert row["model_scope"] == [f"{OUTSIDE}@LENS"] and _checks(row["value_basis"]["power_W"]) == want
    focal = {q.quantity_id: q for q in results.focal_field("early", _focal_request(1200.0)).quantities}
    for name in _FOCAL_POWERS:
        q = focal[name]
        assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and q.value is not None, name
        assert [(c.code, c.observed) for c in q.value_basis.validity] == want, name
    pupil = results.pupil_field("early", _pupil_request(1200.0))
    assert [f["code"] for f in pupil["model_scope"]] == [OUTSIDE] and _checks(pupil["quantities"]["pupil_power_W"]["value_basis"]) == want
    assert _checks(pupil["value_basis"]["incoherent_W_per_mm2"]) == want and _checks(pupil["quantities"]["focal_irradiance"]["value_basis"]) == want
    assert all("value_basis" not in pupil["quantities"][name] for name in ("mask_phase", "device_efficiency"))
    irradiance = results.irradiance_map("early", wavelength_nm=1200.0)
    assert irradiance["status"] == "OK" and [f["code"] for f in irradiance["model_scope"]] == [OUTSIDE]
    assert _checks(irradiance["value_basis"]["incoherent_W_per_mm2"]) == want
    camera = results.detector_response("early", _CAMERA, wavelength_nm=1200.0)
    assert _checks(camera["quantities"]["pixel_power_W"]["value_basis"]) == want
    assert _checks(camera["quantities"]["signal_electrons"]["value_basis"]) == want
    assert "value_basis" not in camera["quantities"]["photon_energy_J"] and "value_basis" not in camera["quantities"]["dark_electrons"]


def test_a_handoff_of_an_in_band_beam_beside_one_a_finding_reaches_stays_unmarked():
    """``mixed()``: GOOD at 532 nm and SRC at 1200 nm through the lens published for 400-1100 nm onto ``early``. GOOD's handoff
    beam, its irradiance map, the camera reading it and the focal field built from it carry no model-scope mark; SRC's do."""

    results = EvaluationSession().evaluate(mixed(), outputs=("complete",)).results()
    beams = {b["path_id"]: b for b in results.handoff("early")["beams"]}
    assert "model_scope" not in beams["GOOD#0"]["validity"] and "value_basis" not in beams["GOOD#0"]
    assert "value_basis" not in (beams["GOOD#0"]["gaussian"] or {})
    assert [f["code"] for f in beams["SRC#0"]["validity"]["model_scope"]] == [OUTSIDE]
    irradiance = results.irradiance_map("early", wavelength_nm=532.0)
    assert "model_scope" not in irradiance and OUTSIDE not in json.dumps(irradiance.get("value_basis") or {})
    camera = results.detector_response("early", _CAMERA, wavelength_nm=532.0)
    assert OUTSIDE not in json.dumps(camera)
    focal = results.focal_field("early", _focal_request(532.0))
    assert all(q.value_basis is None or OUTSIDE not in json.dumps(q.value_basis.model_dump(mode="json")) for q in focal.quantities)
    assert results.irradiance_map("early", wavelength_nm=1200.0)["model_scope"]


def test_the_complete_run_names_the_same_targets_as_the_evaluation():
    document = mixed()
    run = json_api.run_project(document)
    assert run["ok"], run
    evaluation = json_api.evaluate_project(document, list(OUTPUTS))
    assert [(i["code"], i["affected_target_ids"]) for i in run["summary"]["model_scope_issues"]] == \
        [(i["code"], i["affected_target_ids"]) for i in evaluation["model_scope_issues"]]


def test_widening_the_declared_band_clears_the_finding_in_the_same_session():
    document = lens(1200)
    document["targets"] = [spot("spot")]
    session = EvaluationSession()
    before = _evaluation_reply(session.evaluate(document, outputs=OUTPUTS))
    assert assessments(before) == {"spot": "INDETERMINATE"}
    assert {i["code"] for i in before["model_scope_issues"]} == {OUTSIDE}
    widened = copy.deepcopy(document)
    widened["optical_models"][0]["model"]["valid_band_nm"]["max_nm"] = "1500"
    after = _evaluation_reply(session.evaluate(widened, outputs=OUTPUTS))
    assert assessments(after) == {"spot": "MET"} and after["model_scope_issues"] == []
    fresh = json_api.evaluate_project(widened, list(OUTPUTS))
    assert after["targets"] == fresh["targets"] and after["identity"] == fresh["identity"]
    again = _evaluation_reply(session.evaluate(document, outputs=OUTPUTS))  # narrowed again: the finding is back
    assert assessments(again) == {"spot": "INDETERMINATE"}


def _three_beams() -> dict:
    """SRC (1200 nm, outside the band), GOOD and ALSO (532 nm, inside it) through the lens onto the same planes."""

    document = lens(1200)
    add_source(document, "GOOD", 532, y=1.0)
    add_source(document, "ALSO", 532, y=-1.0)
    return document


def _balance(target_id: str, first: str, second: str) -> dict:
    return {"target_id": target_id, "metric": "beam_crossing_angle_deg", "observation_id": "early", "maximum": 90.0,
            "unit": "deg", "parameters": parameters(path_id_a=first, path_id_b=second, crossing_index_a=0, crossing_index_b=0)}


def test_a_pair_target_is_marked_by_its_own_two_beams_only():
    document = _three_beams()
    document["targets"] = [_balance("in_band_pair", "GOOD#0", "ALSO#0"), _balance("missing_pair", "SRC#0", "missing#0"),
                           spot("out_of_band", source_id="SRC")]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    assert {"out_of_band"} == affected(reply)
    got = assessments(reply)
    assert got["out_of_band"] == "INDETERMINATE" and got["in_band_pair"] == "MET"


def test_a_pair_that_selects_the_out_of_band_beam_is_marked():
    document = _three_beams()
    add_source(document, "BAD", 1200, y=2.0)
    document["targets"] = [_balance("pair", "SRC#0", "BAD#0")]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    assert affected(reply) == {"pair"} and assessments(reply)["pair"] == "INDETERMINATE"


def test_a_sampled_line_outside_the_band_marks_only_the_targets_that_select_it():
    """A 532 nm Gaussian beam and a 1200 nm sampled emitter through a lens published for 400-1100 nm."""

    from test_sampled_sources import _component, _intensity

    document = lens(532)
    sampled = _component(_intensity([[1.0]], 0.1, rays=16, wavelength_nm=1200), "BAD")
    sampled["nominal_pose"]["position_mm"][0] = -100
    document["components"].append(sampled)
    document["targets"] = [spot("beam_spot", source_id="SRC"),
                           {"target_id": "beam_power", "metric": "power_W", "observation_id": "early", "source_id": "SRC",
                            "minimum": 0.0, "unit": "W"},
                           {"target_id": "sampled_power", "metric": "power_W", "observation_id": "early", "source_id": "BAD",
                            "minimum": 0.0, "unit": "W"},
                           {"target_id": "line532", "metric": "power_W", "observation_id": "early", "wavelength_nm": 532,
                            "minimum": 0.0, "unit": "W"}]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    assert OUTSIDE in {i["code"] for i in reply["model_scope_issues"]}
    assert affected(reply) == {"sampled_power"}
    got = assessments(reply)
    assert got["beam_spot"] == "MET" and got["beam_power"] == "MET" and got["sampled_power"] == "INDETERMINATE"
    (line532,) = [t for t in reply["targets"] if t["target_id"] == "line532"]
    assert line532["assessment"] == "MET" and not line532.get("value_basis")
    # the finding lists no arrival (sampled light, named by its planes): it marks the 1200 nm line the sampled light is on
    # and the plane's total, not the 532 nm line only the Gaussian beam reaches -- whose number line532 reads unmarked
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    for surface, plane in (("monitor()", results.monitor("early")),
                           ("monitors()", [m for m in results.monitors() if m["observation_id"] == "early"][0])):
        lines = {s["wavelength_nm"]: (s.get("value_basis") or {}).get("power_W") for s in plane["power_by_wavelength"]}
        assert lines[532.0] is None, (surface, lines[532.0])
        assert lines[1200.0]["kind"] == "ESTIMATE" and [c["observed"] for c in lines[1200.0]["validity"]] == [1200.0], surface
        total = plane["value_basis"]["total_power_W"]
        assert total["kind"] == "ESTIMATE" and [c["observed"] for c in total["validity"]] == [1200.0], surface
    (early,) = [m for m in reply["monitors"] if m["observation_id"] == "early"]
    assert [c["observed"] for c in early["value_basis"]["total_power_W"]["validity"]] == [1200.0]


def _sampled_through_lens(*emitters: tuple[str, float, float]) -> dict:
    """The 532 nm Gaussian ``SRC`` through the lens published for 400-1100 nm, and per ``(name, wavelength, y)`` a sampled
    emitter 100 mm before the lens, ``y`` mm off its axis (the lens is 25.4 mm across: 40 mm off, its light misses it)."""

    from test_sampled_sources import _component, _intensity

    document = lens(532)
    for name, wavelength, y in emitters:
        sampled = _component(_intensity([[1.0]], 0.1, rays=16, wavelength_nm=wavelength), name)
        sampled["nominal_pose"]["position_mm"][0] = -100
        sampled["nominal_pose"]["position_mm"][1] = y
        document["components"].append(sampled)
    return document


def test_a_finding_on_sampled_light_lists_the_lines_it_reaches():
    """The finding on the 1200 nm sampled emitter lists that line on each plane its rays reach (``affected_sampled_lines``,
    also per check in ``validity_reach``); the 532 nm Gaussian beam is in no finding."""

    reply = json_api.evaluate_project(_sampled_through_lens(("BAD", 1200, 0.0)), list(OUTPUTS))
    (finding,) = [i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE]
    lines = finding["affected_sampled_lines"]
    assert {(r["source_id"], r["spectral_index"]) for r in lines} == {("BAD", 0)}
    assert {r["observation_id"] for r in lines} == set(finding["affected_observation_ids"]) >= {"early"}
    assert not finding.get("affected_arrivals")
    assert [r["affected_sampled_lines"] for r in finding["validity_reach"]] == [lines]


#: The numbers of a sampled line's row that read the light carried to the plane (its emitted_W is the source's own).
_SAMPLED_READ = ("power_W", "collection_fraction", "collection_fraction_half_samples", "collection_fraction_difference",
                 "collection_fraction_sigma")


def _row_basis(row: dict) -> dict:
    basis = row.get("value_basis")
    return json.loads(basis) if isinstance(basis, str) and basis else (basis or {})


def test_a_sampled_lines_rows_carry_the_check_of_their_own_line_only(tmp_path):
    """Two 1200 nm sampled emitters, BAD through the lens published for 400-1100 nm and MISS 40 mm off its axis (its light
    passes beside the lens), and the 532 nm Gaussian SRC. The finding lists BAD's line only: BAD's row on ``early`` carries
    the ESTIMATE with the 1200 nm wavelength check on its power and collection fractions (not on emitted_W) on every
    projection of the sampled light -- ``monitor()``, ``monitors()``, ``sampled_light()``, the ``sampled-light`` table and
    its CSV export, the run summary of ``evaluate_project`` -- and its crossings' powers in the ``sampled-crossings`` rows;
    MISS's row, at the same wavelength, stays bare, as does the 532 nm line. The plane's 1200 nm line, which BAD's light is
    part of, carries the check."""

    import csv
    import io

    from os_optics.local import run_optical_project
    from os_optics.results import load_run

    document = _sampled_through_lens(("BAD", 1200, 0.0), ("MISS", 1200, 40.0))
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    (finding,) = [i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE]
    assert {r["source_id"] for r in finding["affected_sampled_lines"]} == {"BAD"}
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    run_optical_project(document, output_dir=tmp_path / "run")
    stored = load_run(tmp_path / "run")
    (summary,) = [m for m in reply["monitors"] if m["observation_id"] == "early"]

    def early(rows):
        return {r["source_id"]: _row_basis(r) for r in rows if r.get("observation_id", "early") == "early"}

    surfaces = {
        "monitor()": early(results.monitor("early")["sampled_light"]),
        "monitors()": early(next(m for m in results.monitors() if m["observation_id"] == "early")["sampled_light"]),
        "sampled_light()": early(results.sampled_light("early")),
        "table": early(results.table("sampled-light", monitor="early")[0]),
        "load_run csv": early(csv.DictReader(io.StringIO(stored.export("sampled-light", "csv").decode("utf-8")))),
        "evaluate_project": early(summary["sampled_light"]),
    }
    for surface, rows in surfaces.items():
        assert set(rows) == {"BAD", "MISS"}, surface
        assert rows["MISS"] == {}, (surface, rows["MISS"])
        assert set(rows["BAD"]) == set(_SAMPLED_READ), (surface, sorted(rows["BAD"]))
        for field in _SAMPLED_READ:
            entry = rows["BAD"][field]
            assert entry["kind"] == "ESTIMATE" and [(c["code"], c["observed"]) for c in entry["validity"]] == [(OUTSIDE, 1200.0)], \
                (surface, field)
    crossings = results.table("sampled-crossings", monitor="early")[0]
    assert {r["source_id"] for r in crossings} == {"BAD", "MISS"}
    for row in crossings:
        basis = _row_basis(row)
        assert (set(basis) == {"power_W"}) if row["source_id"] == "BAD" else basis == {}, row
    lines = {s["wavelength_nm"]: (s.get("value_basis") or {}).get("power_W") for s in results.monitor("early")["power_by_wavelength"]}
    assert OUTSIDE not in json.dumps(lines[532.0]) and [c["observed"] for c in lines[1200.0]["validity"]] == [1200.0]


def test_a_sampled_line_beside_a_gaussian_beam_of_its_wavelength_through_the_lens_carries_the_check():
    """A 1200 nm Gaussian beam and a 1200 nm sampled emitter both cross the lens: the finding lists the beam's arrivals and
    the emitter's line, and the emitter's row carries the check (a rule that took sampled light only where a finding lists
    no arrival on the plane left it bare)."""

    document = _sampled_through_lens(("BAD", 1200, 0.0))
    document["components"][0]["source_model"]["wavelength_nm"] = 1200
    results = EvaluationSession().evaluate(document, outputs=OUTPUTS).results()
    (row,) = [r for r in results.sampled_light("early") if r["source_id"] == "BAD"]
    assert [c["observed"] for c in row["value_basis"]["power_W"]["validity"]] == [1200.0]
    arrivals = results.monitor("early")["arrivals"]
    assert arrivals and all(OUTSIDE in json.dumps(a["value_basis"]["beam.power_W"]) for a in arrivals)


def test_a_repeated_crossing_is_marked_from_the_crossing_behind_the_finding():
    from declared_helpers import declared_component, sequence, surface
    from test_paraxial_mirrors_compounds import HR

    document = lens(1200)
    document["observations"] = document["observations"][:1]
    document["observations"][0]["pose"]["position_mm"][0] = -50  # crossed on the way in and again behind the mirror
    coating = copy.deepcopy(HR)
    coating["coating_id"] = "SCOPE-MIRROR"
    coating["bands"][0]["max_nm"] = "2000"
    document["model_coatings"] = [coating]
    document["optical_models"].append({"model_id": "mirror", "model": sequence(
        [surface(1, "0", traced="REFLECTED", coating_id=coating["coating_id"])], [],
        band={"min_nm": "400", "max_nm": "2000", "status": "SPECIFIED"})})
    document["components"].append(declared_component("M", "mirror", "mirror", (40, 0, 0), quaternion=(0.5, 0.5, -0.5, -0.5)))

    def crossing(target_id: str, index: int) -> dict:
        return {"target_id": target_id, "metric": "beam_radius_mm", "observation_id": "early", "maximum": 100.0, "unit": "mm",
                "parameters": parameters(path_id="SRC#0", crossing_index=index)}

    document["targets"] = [crossing("first", 0), crossing("second", 1)]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    assert affected(reply) == {"second"}
    got = assessments(reply)
    assert got["first"] == "MET" and got["second"] == "INDETERMINATE"


def test_a_finding_lists_the_arrivals_it_reaches_and_the_check_it_states():
    document = mixed()
    run = json_api.run_project(document)
    assert run["ok"], run
    (issue,) = [i for i in run["summary"]["model_scope_issues"] if i["code"] == OUTSIDE]
    # the out-of-band beam behind the lens on both planes; not GOOD's beam, not the plane before the lens
    assert issue["affected_arrivals"] == [{"observation_id": "early", "path_id": "SRC#0", "crossing_index": 0},
                                          {"observation_id": "image", "path_id": "SRC#0", "crossing_index": 0}]
    assert issue["validity"] == [{"parameter": "wavelength_nm", "unit": "nm", "minimum": 400.0, "maximum": 1100.0, "observed": 1200.0,
                                  "within": False, "at": ["LENS"], "code": OUTSIDE}]
    # (the record holds the applicability; the arrivals and the checks stay with the run's own listing)
    (recorded,) = [i for i in run["summary"]["local_validation"]["simulation"]["model_scope_issues"] if i["code"] == OUTSIDE]
    assert "affected_arrivals" not in recorded and "validity" not in recorded
    evaluated = json_api.evaluate_project(document, list(OUTPUTS))
    assert [i for i in evaluated["model_scope_issues"] if i["code"] == OUTSIDE] == [issue]


def test_the_value_of_a_target_a_finding_names_carries_its_check():
    document = mixed()
    check = {"parameter": "wavelength_nm", "unit": "nm", "minimum": 400.0, "maximum": 1100.0, "observed": 1200.0, "within": False,
             "at": ["LENS"], "code": OUTSIDE}
    session = EvaluationSession()
    evaluation = session.evaluate(document, outputs=OUTPUTS)
    reply = _evaluation_reply(evaluation)
    run = json_api.run_project(document)
    rows = {"evaluate_project": {t["target_id"]: t for t in reply["targets"]},
            "run_project": {t["target_id"]: t for t in run["summary"]["targets"]},
            "results().targets()": {t["target_id"]: t for t in evaluation.results().targets()["targets"]},
            "results().monitor()": {t["target_id"]: t for o in document["observations"]
                                    for t in evaluation.results().monitor(o["observation_id"])["targets"]}}
    for surface, targets in rows.items():
        bad = targets["bad"]
        assert bad["assessment"] == "INDETERMINATE" and bad["value"] is not None, surface
        assert bad["value_basis"] == {"value": {"kind": "ESTIMATE", "validity": [check]}}, surface
        # (a target on in-band light on the same plane is the model's value; so is one before the lens)
        for kept in ("good_by_source", "good_by_wavelength", "bad_before_lens"):
            assert "value_basis" not in targets[kept], (surface, kept)
        assert targets["pair"]["value_basis"]["value"]["kind"] == "ESTIMATE", surface


def test_a_rating_utilization_target_reads_every_beam_on_its_element_whatever_source_it_states():
    """A power_rating_utilization target's value is the element's utilization by all the light on it (the exposure of
    every beam on its rated surfaces), so a finding on any beam that meets the element marks it, also one that states the
    in-band source: SRC at 1110 nm (outside the lens's 400-1100 nm band) and GOOD at 1095 nm, both on the rated lens."""

    document = lens(1110)
    add_source(document, "GOOD", 1095, y=1.0)
    document["optical_models"][0]["power_ratings"] = [{
        "regime": "CW", "value": "1", "unit": "W/cm2", "wavelength_nm": "1100", "value_definition": "PEAK", "status": "DECLARED",
        "basis": "test fixture rating, not product data", "beam_diameter_mm": "1", "beam_diameter_definition": "1/E2",
        "polarization_condition": "linear"}]
    utilization = {"metric": "power_rating_utilization", "device_instance_id": "LENS", "maximum": 1.0}
    document["targets"] = [{"target_id": "u_any", **utilization}, {"target_id": "u_good", "source_id": "GOOD", **utilization}]
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    assert reply["ok"], reply
    (issue,) = [i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE]
    assert set(issue["affected_target_ids"]) == {"u_any", "u_good"}
    rows = {t["target_id"]: t for t in reply["targets"]}
    # (the same value: the stated source does not narrow what the element's utilization reads)
    assert rows["u_good"]["value"] == rows["u_any"]["value"] is not None
    for target_id in ("u_any", "u_good"):
        row = rows[target_id]
        assert row["assessment"] != "MET" and OUTSIDE in row["issue_codes"], row
        assert row["value_basis"]["value"]["kind"] == "ESTIMATE", row


def test_the_documents_state_what_a_target_reads_on_its_element_and_of_a_polarization_route():
    """The results reference and element_scope_issues docstring state the rule pinned by the
    headless tests above: element utilization reads every beam whatever its source or wavelength,
    and a polarization metric reads both components of a route."""

    from os_optics.local.validity import element_scope_issues

    repo = Path(__file__).resolve().parents[1]
    results = " ".join((repo / "docs/engine/RESULTS_AND_EXPORT.md").read_text(encoding="utf-8").split())
    element = ("on the element it names, the beams of its source and wavelength, and for `power_rating_utilization` every beam "
               "on the element, whatever source or wavelength the target states (the element's exposure by all of them)")
    route = ("a polarization metric (degree of polarization, Stokes, azimuth, ellipticity) whose path selects one "
             "polarization component reads both components of that route (same source line, crossing and lineage)")
    assert element in results
    assert route in results
    assert "the beams of its source and wavelength on the element it names" not in results
    assert "the beams of its source and wavelength on the element" not in results
    # (another wavelength leaves a target as it is only by what the target's selection reads on its plane)
    assert ("On its plane, a beam its selection does not read, such as a beam of another wavelength outside a lens's band, "
            "leaves its assessment as it is.") in results
    assert "a beam of another wavelength on the same plane, outside a lens's band, leaves" not in results
    doc = " ".join((element_scope_issues.__doc__ or "").split())
    assert ("on the element it names, the beams of its source and wavelength, and for an element metric (``ELEMENT_METRICS``: "
            "``power_rating_utilization``) every beam on the element, whatever source or wavelength the target states") in doc
    assert "both components of its route" in doc
    assert "on the element it names, a beam of its source and wavelength --" not in doc


def test_a_targets_value_basis_states_the_checks_of_the_light_it_reads():
    """SRC at 1200 nm and FAR at 1300 nm, both outside the lens's 400-1100 nm band: one finding lists a check per line. A
    target's basis keeps the check of the line it reads -- near (SRC) 1200 nm, far (FAR) 1300 nm -- and a pair of the two,
    or a target that selects no beam, both: on every surface that states the row (evaluate_project, run_project,
    RunResults.targets())."""

    document = lens(1200)
    add_source(document, "FAR", 1300, y=1.0)
    document["targets"] = [
        spot("near", source_id="SRC"),
        spot("far", source_id="FAR"),
        {"target_id": "pair", "metric": "centroid_separation_mm", "observation_id": "early", "source_id": "SRC",
         "source_id_b": "FAR", "wavelength_nm": 1200, "wavelength_nm_b": 1300, "maximum": 100.0, "unit": "mm"},
        {"target_id": "any_beam", "metric": "power_W", "observation_id": "early", "minimum": 0.0, "unit": "W"},
    ]

    def check(observed: float) -> dict:
        return {"parameter": "wavelength_nm", "unit": "nm", "minimum": 400.0, "maximum": 1100.0, "observed": observed,
                "within": False, "at": ["LENS"], "code": OUTSIDE}

    expected = {"near": [check(1200.0)], "far": [check(1300.0)], "pair": [check(1200.0), check(1300.0)],
                "any_beam": [check(1200.0), check(1300.0)]}
    session = EvaluationSession()
    evaluation = session.evaluate(document, outputs=OUTPUTS)
    reply = _evaluation_reply(evaluation)
    (issue,) = [i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE]
    assert issue["validity"] == [check(1200.0), check(1300.0)] and set(issue["affected_target_ids"]) == set(expected)
    run = json_api.run_project(document)
    assert run["ok"], run
    rows = {"evaluate_project": {t["target_id"]: t for t in reply["targets"]},
            "run_project": {t["target_id"]: t for t in run["summary"]["targets"]},
            "results().targets()": {t["target_id"]: t for t in evaluation.results().targets()["targets"]}}
    for surface, targets in rows.items():
        for target_id, checks in expected.items():
            assert targets[target_id]["value_basis"]["value"] == {"kind": "ESTIMATE", "validity": checks}, (surface, target_id)


# ------------------------------------------------------------------------------------------------ a check per beam


def _two_beams_on_one_mirror() -> dict:
    """Sources A (780 nm, along +X) and B (780 nm, 10 deg off it) on one flat mirror whose coating is published for 0-45
    deg: A meets it at 50 deg and goes on to plane screenA, B at 60 deg and goes on to screenB."""

    import math

    from local_helpers import FLAT_MIRROR, element, plane, project, source
    from os_optics.optical_view import geometry as g
    from test_local_validity import _level_quaternion

    theta = math.radians(50.0)
    normal = (-math.cos(theta), math.sin(theta), 0.0)

    def reflect(d):
        dot = sum(a * b for a, b in zip(d, normal, strict=True))
        return tuple(a - 2.0 * dot * b for a, b in zip(d, normal, strict=True))

    d_a, d_b = (1.0, 0.0, 0.0), (math.cos(math.radians(10.0)), math.sin(math.radians(10.0)), 0.0)
    r_a, r_b = reflect(d_a), reflect(d_b)
    return project(
        [source("A", position=g.scale(d_a, -100.0), quaternion=_level_quaternion(d_a)),
         source("B", position=g.scale(d_b, -100.0), quaternion=_level_quaternion(d_b)),
         element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), _level_quaternion(normal))],
        [plane("screenA", g.scale(r_a, 100.0), _level_quaternion(r_a)), plane("screenB", g.scale(r_b, 100.0), _level_quaternion(r_b))],
        [{"target_id": "pa", "metric": "power_W", "observation_id": "screenA", "source_id": "A", "minimum": 0.0, "unit": "W"},
         {"target_id": "pb", "metric": "power_W", "observation_id": "screenB", "source_id": "B", "minimum": 0.0, "unit": "W"},
         {"target_id": "pa_any", "metric": "power_W", "observation_id": "screenA", "minimum": 0.0, "unit": "W"},
         {"target_id": "spot_a", "metric": "spot_rms_mm", "observation_id": "screenA", "maximum": 100.0, "unit": "mm"}])


def _rows_on_every_surface(document: dict, outputs=OUTPUTS) -> dict[str, dict[str, dict]]:
    """A target's row as each surface states it: evaluate_project, a prepared handle's reply, the run_project summary,
    RunResults.targets() of the session's evaluation and its RunResults.monitor() of every plane the design declares."""

    from os_optics.local.prepared import PREPARED_EVALUATIONS

    evaluation = EvaluationSession().evaluate(document, outputs=outputs)
    reply = json_api.evaluate_project(document, list(outputs))
    prepared = json_api.prepare_evaluation(document)
    try:
        through_handle = json_api.evaluate_prepared(prepared["handle"], list(outputs))
    finally:
        PREPARED_EVALUATIONS.clear()
    run = json_api.run_project(document)
    assert reply["ok"] and through_handle["ok"] and run["ok"], (reply, through_handle, run)
    return {"evaluate_project": {t["target_id"]: t for t in reply["targets"]},
            "evaluate_prepared": {t["target_id"]: t for t in through_handle["targets"]},
            "run_project": {t["target_id"]: t for t in run["summary"]["targets"]},
            "results().targets()": {t["target_id"]: t for t in evaluation.results().targets()["targets"]},
            "results().monitor()": {t["target_id"]: t for o in document["observations"]
                                    for t in evaluation.results().monitor(o["observation_id"])["targets"]}}


def _observed(row: dict, code: str) -> list:
    return [c["observed"] for c in row["value_basis"]["value"]["validity"] if c["code"] == code]


def test_a_targets_value_basis_states_the_angles_its_own_beam_met():
    """One finding, COATING_AOI_OUTSIDE_RANGE on M1, states a check per angle (50 and 60 deg). A target's value states the
    angle of the beam it reads: pa (A on screenA) 50 deg only, pb (B on screenB) 60 deg only, and a target on screenA that
    selects no beam reads A's light alone there -- on every surface that states the row."""

    code = "COATING_AOI_OUTSIDE_RANGE"
    document = _two_beams_on_one_mirror()
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    (issue,) = [i for i in reply["model_scope_issues"] if i["code"] == code]
    assert [c["observed"] for c in issue["validity"]] == pytest.approx([50.0, 60.0], abs=1e-6)
    assert {a["observation_id"] for a in issue["affected_arrivals"]} == {"screenA", "screenB"}
    # (per check: the targets that read the light met at that angle, and its arrivals; the record does not hold it)
    reach = issue["validity_reach"]
    assert [sorted(r["affected_target_ids"]) for r in reach] == [["pa", "pa_any", "spot_a"], ["pb"]]
    assert [{a["observation_id"] for a in r["affected_arrivals"]} for r in reach] == [{"screenA"}, {"screenB"}]
    for surface, rows in _rows_on_every_surface(document).items():
        assert _observed(rows["pa"], code) == pytest.approx([50.0], abs=1e-6), surface
        assert _observed(rows["pa_any"], code) == pytest.approx([50.0], abs=1e-6), surface
        assert _observed(rows["pb"], code) == pytest.approx([60.0], abs=1e-6), surface
        assert rows["pa"]["value_basis"]["value"]["validity"] == rows["pa_any"]["value_basis"]["value"]["validity"], surface


def test_an_imaging_field_states_the_angle_its_own_beam_met():
    """The same layout imaged: the field of A on screenA carries the 50 deg check only, the field of B on screenB the 60 deg
    check only (a coating's angle touches what reads power: the geometric MTF; the cone and the spot stay the trace's)."""

    code = "COATING_AOI_OUTSIDE_RANGE"
    results = EvaluationSession().evaluate(_two_beams_on_one_mirror(), outputs=OUTPUTS).results()
    for plane, field, angle in (("screenA", "A#0", 50.0), ("screenB", "B#0", 60.0)):
        image = results.imaging(plane, IMAGING)
        marked = [q for q in image.quantities if q.value_basis is not None and any(c.code == code for c in q.value_basis.validity)]
        assert {q.quantity_id for q in marked} == {f"{field}.geometric_mtf_tangential", f"{field}.geometric_mtf_sagittal"}, plane
        for q in marked:
            assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and q.reason == code
            assert [c.observed for c in q.value_basis.validity if c.code == code] == [pytest.approx(angle, abs=1e-9)], q.quantity_id


def test_a_targets_value_basis_states_the_intercepted_rays_of_its_own_beam():
    """Two 3 mm beams pass beside a window's edge at different distances (0.01 mm and 0.6 mm): one BEAM_PARTIALLY_INTERCEPTED
    finding states a check per beam (its count of intercepted pupil rays). A target selecting one source states that
    beam's count; one that selects no beam both."""

    from local_helpers import WINDOW, element, plane, product_document, project, source

    code = "BEAM_PARTIALLY_INTERCEPTED"
    edge = float(product_document(WINDOW)["simulation_models"][0]["edge_radius_mm"])
    document = project(
        [source("NEAR", position=(-200.0, 0.0, 0.0), waist_mm=3.0), source("FAR", position=(-200.0, -0.6, 0.0), waist_mm=3.0),
         element("W", "window", WINDOW, (0.0, edge + 0.01, 0.0))],
        [plane("D", (100.0, 0.0, 0.0), radius_mm=25.0)],
        [{"target_id": "near", "metric": "power_W", "observation_id": "D", "source_id": "NEAR", "minimum": 0.0, "unit": "W"},
         {"target_id": "far", "metric": "power_W", "observation_id": "D", "source_id": "FAR", "minimum": 0.0, "unit": "W"},
         {"target_id": "both", "metric": "power_W", "observation_id": "D", "minimum": 0.0, "unit": "W"}])
    reply = json_api.evaluate_project(document, list(OUTPUTS))
    (issue,) = [i for i in reply["model_scope_issues"] if i["code"] == code]
    counts = [c["observed"] for c in issue["validity"]]
    assert len(counts) == 2 and counts[0] < counts[1], counts  # (the farther beam loses fewer rays)
    for surface, rows in _rows_on_every_surface(document).items():
        assert _observed(rows["far"], code) == [counts[0]], surface
        assert _observed(rows["near"], code) == [counts[1]], surface
        assert _observed(rows["both"], code) == counts, surface


# ------------------------------------------------------------------------------------------------ imaging fields


def _crossed_twice(wavelength: float) -> dict:
    """The lens design of ``lens(wavelength)`` with its plane ``early`` moved before the lens (x = -50 mm) and a mirror
    behind the lens that sends the beam back across it: SRC crosses ``early`` first on its way in (crossing 0, before the
    lens), then behind the lens on its way back (crossing 1)."""

    from declared_helpers import declared_component, sequence, surface
    from test_paraxial_mirrors_compounds import HR

    document = lens(wavelength)
    document["observations"] = document["observations"][:1]
    document["observations"][0]["pose"]["position_mm"][0] = -50
    coating = copy.deepcopy(HR)
    coating["coating_id"] = "SCOPE-MIRROR"
    coating["bands"][0]["max_nm"] = "2000"
    document["model_coatings"] = [coating]
    document["optical_models"].append({"model_id": "mirror", "model": sequence(
        [surface(1, "0", traced="REFLECTED", coating_id=coating["coating_id"])], [],
        band={"min_nm": "400", "max_nm": "2000", "status": "SPECIFIED"})})
    document["components"].append(declared_component("M", "mirror", "mirror", (40, 0, 0), quaternion=(0.5, 0.5, -0.5, -0.5)))
    document["targets"] = []
    return document


def _marked(quantity) -> bool:
    return quantity.value_basis is not None and any(c.code == OUTSIDE for c in quantity.value_basis.validity)

def test_imaging_at_the_crossing_before_the_out_of_band_lens_is_the_models_own():
    """The field on ``early`` at crossing 0 reads light that has not met the lens yet: it is not marked, and its numbers and
    states are those of the same design with the source inside the lens's band. At crossing 1 (behind the lens), and with
    no crossing asked (the last one), the field is marked."""

    scoped = EvaluationSession().evaluate(_crossed_twice(1200), outputs=OUTPUTS).results()
    (issue,) = [i for i in scoped.targets()["model_scope_issues"] if i["code"] == OUTSIDE]
    assert issue["affected_arrivals"] == [{"observation_id": "early", "path_id": "SRC#0", "crossing_index": 1}]
    inside = EvaluationSession().evaluate(_crossed_twice(1000), outputs=OUTPUTS).results()
    first = scoped.imaging("early", {**IMAGING, "crossing_index": 0})
    reference = inside.imaging("early", {**IMAGING, "crossing_index": 0})
    assert not any(_marked(q) for q in first.quantities)
    for name in ("image_space_na", "geometric_spot_rms_radius"):
        got, want = first.quantity(f"SRC#0.{name}"), reference.quantity(f"SRC#0.{name}")
        assert (got.state, got.reason, got.value, got.value_basis) == (want.state, want.reason, want.value, want.value_basis), name
    spot = first.quantity("SRC#0.geometric_spot_rms_radius")
    assert (spot.state, spot.reason) == (QuantityState.UNKNOWN, "PUPIL_NOT_BOUNDED") and spot.value is not None
    for request in ({**IMAGING, "crossing_index": 1}, IMAGING):
        behind = scoped.imaging("early", request).quantity("SRC#0.geometric_spot_rms_radius")
        assert behind.state is QuantityState.OUT_OF_MODEL_SCOPE and behind.reason == OUTSIDE and _marked(behind), request


def test_imaging_on_the_strongest_route_is_not_marked_by_a_weaker_arm_that_met_the_lens():
    """The two-beam interferometer with an ideal thin lens on its weaker (transmitted) arm, between M1 and the plane: the
    lens is published for 400-700 nm and the source is at 780 nm, so the finding reaches the plane on that arm only. The
    imaging field analyses the line's strongest route (the reflected arm): it is not marked, and is the field of the same
    design with the lens published for 400-1100 nm."""

    from declared_helpers import declared_component, thin_lens_model
    from local_helpers import ALONG_Y
    from os_optics.local import example_project_of_kind

    def design(band_max_nm: str) -> dict:
        document = example_project_of_kind("two_beam_interference").payload()
        model = thin_lens_model("arm-lens")
        model["model"]["valid_band_nm"]["max_nm"] = band_max_nm
        document["optical_models"] = [model]
        document["components"].append(declared_component("ARM_LENS", "lens", "arm-lens", (100.0, 50.0, 0.0), ALONG_Y))
        return document

    scoped = EvaluationSession().evaluate(design("700"), outputs=OUTPUTS).results()
    (issue,) = [i for i in scoped.targets()["model_scope_issues"] if i["code"] == OUTSIDE]
    assert issue["instance_id"] == "ARM_LENS"
    assert issue["affected_arrivals"] == [{"observation_id": "crossing_plane", "path_id": "SRC#0", "crossing_index": 0}]
    arrivals = {a.beam.path_id: a for o in scoped.result.observations if o.observation_id == "crossing_plane" for a in o.arrivals}
    transmitted, reflected = arrivals["SRC#0"], arrivals["SRC#0.0"]
    # The catalog splitter glass has unstated bulk loss. Even its unattenuated transmitted arm is weaker than the reflection.
    assert transmitted.beam.power_W is None and transmitted.power_upper_bound_W is not None
    assert reflected.beam.power_W is not None and reflected.beam.power_W > transmitted.power_upper_bound_W
    image = scoped.imaging("crossing_plane", IMAGING)
    assert not any(_marked(q) for q in image.quantities)
    reference = EvaluationSession().evaluate(design("1100"), outputs=OUTPUTS).results().imaging("crossing_plane", IMAGING)
    assert image.quantities == reference.quantities


def _sampled_fields(*emitters: tuple[str, float, float]) -> dict:
    """The lens published for 400-1100 nm with its Gaussian source taken out, and per ``(name, wavelength, y)`` a 64 x 64
    Gaussian ``SAMPLED_FIELD`` (0.04 mm pitch) 20 mm before the lens, ``y`` mm off its axis (40 mm off: its light passes
    beside the lens, which is 25.4 mm across)."""

    from test_sampled_sources import _component, _field, _gaussian_rows

    document = lens()
    document["components"] = [c for c in document["components"] if c["optical_role"] != "source"]
    amplitude = _gaussian_rows(64, 0.04, 0.6, amplitude=True)
    phase = [[0.0] * 64 for _ in range(64)]
    for name, wavelength, y in emitters:
        sampled = _component(_field(amplitude, phase, 0.04, wavelength_nm=wavelength), name)
        sampled["custom_part"]["custom_part_key"] = name.lower()
        sampled["nominal_pose"]["position_mm"][0] = -20
        sampled["nominal_pose"]["position_mm"][1] = y
        document["components"].append(sampled)
    return document


#: A field request without the padding refinement (the plane values once).
FIELD = {"refinement": False}


def _field_marks(doc) -> dict[str, list]:
    """Each quantity of a field document with a number: its state and the (code, observed) of its value_basis checks."""

    return {q.quantity_id: (q.state, [(c.code, c.observed) for c in (q.value_basis.validity if q.value_basis else ())])
            for q in doc.quantities if q.value is not None or q.values_ref is not None}


def test_a_sampled_field_a_finding_reaches_carries_its_check_on_every_number_of_its_field(tmp_path):
    """A 1200 nm sampled field through the lens published for 400-1100 nm: the finding lists its line, and every number of
    the field on ``early`` (its rays' power, the propagated plane values, the contrast) is ``OUT_OF_MODEL_SCOPE`` with the
    finding's ESTIMATE, its number kept -- on ``RunResults.field``, the JSON ``field`` query and the command line; the field
    file ``--field-out`` writes carries the finding, and a pupil field built from that file states it in its
    ``model_scope``. The same design at 1000 nm reads every number ``KNOWN`` with no basis, and the values are those of a
    field."""

    import hashlib

    from os_optics.cli import main
    from os_optics.local import run_optical_project
    from os_optics.results.field import field_npz

    document = _sampled_fields(("S", 1200, 0.0))
    scoped = EvaluationSession().evaluate(document, outputs=("complete",)).results()
    (finding,) = [i for i in scoped.targets()["model_scope_issues"] if i["code"] == OUTSIDE]
    assert {(r["observation_id"], r["source_id"]) for r in finding["affected_sampled_lines"]} >= {("early", "S")}
    doc = scoped.field("early", FIELD)
    marked = _field_marks(doc)
    assert "S.geometric_power_W" in marked and "S.power_window_W" in marked and "S.contrast" in marked
    assert marked == {q: (QuantityState.OUT_OF_MODEL_SCOPE, [(OUTSIDE, 1200.0)]) for q in marked}
    assert [(f["code"], f["instance_id"]) for f in doc.arrays["model_scope"]] == [(OUTSIDE, "LENS")]
    inside = EvaluationSession().evaluate(_sampled_fields(("S", 1000, 0.0)), outputs=("complete",)).results()
    reference = inside.field("early", FIELD)
    assert _field_marks(reference) == {q: (QuantityState.KNOWN, []) for q in marked}
    assert all(q.value_basis is None for q in reference.quantities) and "model_scope" not in reference.arrays
    # the JSON results query and the command line read the same document
    run_optical_project(document, output_dir=tmp_path / "run")
    files = {p.name: p.read_bytes() for p in (tmp_path / "run").iterdir() if p.is_file()}
    queried = json_api.run_results(files, {"kind": "field", "monitor": "early", "request": FIELD})["result"]
    assert {q["quantity_id"]: (q["state"], _checks(q.get("value_basis"))) for q in queried["quantities"] if q.get("value") is not None}         == {q: ("OUT_OF_MODEL_SCOPE", [(OUTSIDE, 1200.0)]) for q in marked}
    out = tmp_path / "field.npz"
    assert main(["results", "field", str(tmp_path / "run"), "--monitor", "early", "--field-out", str(out)]) == 0
    # the field file carries the finding into a pupil field lit by it
    lit = scoped.field("early", {**FIELD, "include_field": True})
    data = field_npz(lit)
    assert out.read_bytes() == data
    count = len(lit.arrays["field_u_mm"])
    pitch = lit.arrays["field_u_mm"][1] - lit.arrays["field_u_mm"][0]
    request = {"wavelength_nm": 1200.0, "focal_length_mm": 100.0, "sampling": {"pixel_count": [count, len(lit.arrays["field_v_mm"])],
                                                                                "pitch_mm": [pitch, pitch]},
               "illumination": {"kind": "FIELD", "file_name": "f.npz", "sha256": hashlib.sha256(data).hexdigest()},
               "output": {"half_extent_mm": [0.3, 0.3]}}
    from os_optics.results.pupil_field import pupil_field

    pupil = pupil_field(request, {"f.npz": data})
    assert pupil["status"] == "OK", pupil
    assert [f["code"] for f in pupil["model_scope"]] == [OUTSIDE]
    assert _checks(pupil["quantities"]["pupil_power_W"]["value_basis"]) == [(OUTSIDE, 1200.0)]
    assert _checks(pupil["value_basis"]["coherent_W_per_mm2"]) == [(OUTSIDE, 1200.0)]
    plain = field_npz(reference if "field_re" in reference.arrays else inside.field("early", {**FIELD, "include_field": True}))
    unmarked = pupil_field({**request, "wavelength_nm": 1000.0, "illumination": {**request["illumination"], "sha256": hashlib.sha256(plain).hexdigest()}},
                           {"f.npz": plain})
    assert unmarked["status"] == "OK" and "model_scope" not in unmarked and OUTSIDE not in json.dumps(unmarked)


def test_a_sampled_field_beside_the_lens_keeps_its_field_unmarked():
    """Two 1200 nm sampled fields, S through the lens published for 400-1100 nm and MISS 40 mm off its axis: the finding
    lists S's line only, and MISS's field on ``early`` reads every number ``KNOWN`` with no basis while S's is marked."""

    results = EvaluationSession().evaluate(_sampled_fields(("S", 1200, 0.0), ("MISS", 1200, 40.0)), outputs=("complete",)).results()
    (finding,) = [i for i in results.targets()["model_scope_issues"] if i["code"] == OUTSIDE]
    assert {r["source_id"] for r in finding["affected_sampled_lines"]} == {"S"}
    miss = results.field("early", {**FIELD, "source_id": "MISS"})
    states = _field_marks(miss)
    assert states and states == {q: (QuantityState.KNOWN, []) for q in states}
    assert "model_scope" not in miss.arrays
    through = _field_marks(results.field("early", {**FIELD, "source_id": "S"}))
    assert through and through == {q: (QuantityState.OUT_OF_MODEL_SCOPE, [(OUTSIDE, 1200.0)]) for q in through}


def test_a_field_file_whose_model_scope_is_not_a_list_of_findings_is_refused():
    import hashlib

    import numpy as np

    from os_optics.results.export import npz_bytes
    from os_optics.results.pupil_field import pupil_field

    field = np.ones((8, 8))
    for scope in (["not json"], ['{"code": "X"}'], ['[{"code": "X", "instance_id": "L", "touches": ["COLOUR"], "validity": []}]']):
        data = npz_bytes({"field_re": field, "field_im": 0.0 * field, "model_scope_json": np.array(scope)})
        request = {"wavelength_nm": 1000.0, "focal_length_mm": 100.0, "sampling": {"pixel_count": [8, 8], "pitch_mm": [0.05, 0.05]},
                   "illumination": {"kind": "FIELD", "file_name": "f.npz", "sha256": hashlib.sha256(data).hexdigest()},
                   "output": {"half_extent_mm": [0.3, 0.3]}}
        reply = pupil_field(request, {"f.npz": data})
        assert (reply["status"], reply["reason"]) == ("UNAVAILABLE", "MASK_FILE_INVALID") and "model_scope_json" in reply["message"], scope



def _scattering_plate(wavelength: float, plate_x: float = 70.0, side=(55.0, 10.0, 0.0), band: dict | None = None, *,
                      scatters=((1, "REFLECT"),), side_radius: float = 4.0, back=None) -> dict:
    """``lens(wavelength)`` with a declared PLATE at ``plate_x`` whose front face scatters 5 % LAMBERTIAN back (REFLECT), and a
    plane ``side`` (radius ``side_radius`` mm) off the axis that only that scattered light reaches; power targets on
    ``early`` and ``side``. ``band``: the plate's published band (default 400-1100 nm). ``scatters``: the scattering faces
    instead, (surface index, side) each. ``back``: the position of one more plane ``back`` (radius 4 mm) with a power
    target ``back_power``."""

    from declared_helpers import declared_component, sequence, surface

    document = lens(wavelength)
    glass = {"material_id": "CLEAR", "display_name": "Clear glass", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.5"},
             "valid_band_nm": {"min_nm": "300", "max_nm": "2000", "status": "SPECIFIED"},
             "absorption_coefficient_per_mm": "0", "basis": "Ideal lossless plate for isolating surface scatter"}
    sides = dict(scatters)
    faces = [surface(index, z, clear="9", **({"scatter": {"bsdf": "LAMBERTIAN", "fraction": "0.05", "side": sides[index],
                                                          "sample_count": 2048}} if index in sides else {}))
             for index, z in ((1, "0"), (2, "5"))]
    plate = sequence(faces, ["CLEAR"], band=band)
    document["optical_models"].append({"model_id": "plate", "model": plate})
    document["model_materials"] = [glass]
    document["components"].append(declared_component("PLATE", "window", "plate", (plate_x, 0.0, 0.0)))
    plane = copy.deepcopy(document["observations"][0])
    plane["observation_id"] = "side"
    plane["pose"]["position_mm"] = list(side)
    plane["aperture"]["radius_mm"] = side_radius
    document["observations"] = [document["observations"][0], plane]
    document["targets"] = [{"target_id": "side_power", "metric": "power_W", "observation_id": "side", "minimum": 0.0, "unit": "W"},
                           {"target_id": "early_power", "metric": "power_W", "observation_id": "early", "minimum": 0.0, "unit": "W"}]
    if back is not None:
        extra = copy.deepcopy(plane)
        extra["observation_id"] = "back"
        extra["pose"]["position_mm"] = list(back)
        extra["aperture"]["radius_mm"] = 4
        document["observations"].append(extra)
        document["targets"].append({"target_id": "back_power", "metric": "power_W", "observation_id": "back", "minimum": 0.0, "unit": "W"})
    return document


def _at(basis: dict | None) -> list:
    return sorted(at for check in ((basis or {}).get("validity") or ()) for at in check["at"])


def test_light_a_surface_scatters_out_of_a_beam_a_finding_reaches_carries_its_check():
    """1200 nm through the lens published for 400-1100 nm onto PLATE (also published for 400-1100 nm), whose front face
    scatters 5 % back; ``side`` is lit only by that scattered light (no arrival). Both findings list the scattered line on
    ``side`` and ``side_power``: the target reads INDETERMINATE with an ESTIMATE naming LENS and PLATE and the plane reads
    OUT_OF_MODEL_SCOPE. At 1000 nm the same light leaves the target MET, the plane BOUNDED and no finding."""

    reply = json_api.evaluate_project(_scattering_plate(1200), list(OUTPUTS))
    assert reply["ok"], reply
    findings = {i["instance_id"]: i for i in reply["model_scope_issues"] if i["code"] == OUTSIDE}
    assert set(findings) == {"LENS", "PLATE"}
    for finding in findings.values():
        assert "side" in finding["affected_observation_ids"] and "side_power" in finding["affected_target_ids"]
        assert {"observation_id": "side", "source_id": "SRC", "spectral_index": 0} in finding["affected_sampled_lines"]
    target = next(t for t in reply["targets"] if t["target_id"] == "side_power")
    assert target["assessment"] == "INDETERMINATE"
    assert target["value_basis"]["value"]["kind"] == "ESTIMATE" and _at(target["value_basis"]["value"]) == ["LENS", "PLATE"]
    monitors = {m["observation_id"]: m for m in reply["monitors"]}
    assert monitors["side"]["quantity_state"] == "OUT_OF_MODEL_SCOPE"
    assert not monitors["side"]["arrivals"]
    (row,) = monitors["side"]["sampled_light"]
    assert row["source_id"] == "SRC" and _at(row["value_basis"]["power_W"]) == ["LENS", "PLATE"]

    control = json_api.evaluate_project(_scattering_plate(1000), list(OUTPUTS))
    assert control["ok"], control
    assert control["model_scope_issues"] == []
    assert assessments(control)["side_power"] == "MET"
    assert {m["observation_id"]: m["quantity_state"] for m in control["monitors"]}["side"] == "BOUNDED"
    side = next(t for t in control["targets"] if t["target_id"] == "side_power")
    assert side["value"] == target["value"] and side["value_basis"]["value"]["kind"] == "LOWER_BOUND"


def test_light_scattered_before_the_point_a_finding_starts_at_keeps_its_assessment():
    """PLATE (published for 300-2000 nm) stands before the lens at x = -50 and scatters back onto ``side`` at (-45, 10, 0):
    at 1200 nm the lens's finding starts behind it, so it does not list the scattered line, and ``side_power`` stays MET."""

    reply = json_api.evaluate_project(_scattering_plate(1200, plate_x=-50.0, side=(-55.0, 10.0, 0.0),
                                                        band={"min_nm": "300", "max_nm": "2000", "status": "SPECIFIED"}), list(OUTPUTS))
    assert reply["ok"], reply
    (finding,) = reply["model_scope_issues"]
    assert finding["instance_id"] == "LENS"
    assert all(line["observation_id"] != "side" for line in finding.get("affected_sampled_lines") or ())
    monitors = {m["observation_id"]: m for m in reply["monitors"]}
    assert monitors["side"]["sampled_light"] and monitors["side"]["sampled_light"][0]["power_W"] > 0.0
    assert assessments(reply)["side_power"] == "MET"
    assert monitors["side"]["quantity_state"] != "OUT_OF_MODEL_SCOPE"


#: PLATE (published for 300-2000 nm) before the lens at x = -50, its back face scattering 5 % forward (TRANSMIT) through the
#: lens onto ``side`` (radius 2 mm, off the axis behind the lens).
WIDE_BAND = {"min_nm": "300", "max_nm": "2000", "status": "SPECIFIED"}
FORWARD = {"plate_x": -50.0, "side": (50.0, 8.0, 0.0), "band": WIDE_BAND, "scatters": ((2, "TRANSMIT"),), "side_radius": 2.0}


def _complete_reply(document: dict) -> dict:
    """A session's complete evaluation of ``document`` read like a reply: its findings, targets and monitors, and the
    irradiance map of ``side``."""

    results = EvaluationSession().evaluate(document, outputs=("complete",)).results()
    targets = results.targets()
    return {"ok": True, "model_scope_issues": targets["model_scope_issues"], "targets": targets["targets"],
            "monitors": results.monitors(), "map": results.irradiance_map("side", 1200.0)}


@pytest.mark.parametrize("surface", ["evaluate_project", "session"])
def test_light_a_surface_scatters_through_an_element_used_outside_its_band_carries_its_finding(surface):
    """1200 nm: the light PLATE scatters forward crosses the lens published for 400-1100 nm on its way to ``side``. The
    lens's finding lists that line on ``side`` and ``side_power``: the target reads INDETERMINATE with an ESTIMATE naming
    LENS, the plane OUT_OF_MODEL_SCOPE and its sampled-light row the same check -- through the JSON API and a session's
    complete evaluation alike. At 1000 nm: no finding, MET, BOUNDED."""

    document = _scattering_plate(1200, **FORWARD)
    reply = json_api.evaluate_project(document, list(OUTPUTS)) if surface == "evaluate_project" else _complete_reply(document)
    assert reply["ok"], reply
    (finding,) = [i for i in reply["model_scope_issues"] if i["instance_id"] == "LENS"]
    assert finding["code"] == OUTSIDE
    assert {"observation_id": "side", "source_id": "SRC", "spectral_index": 0} in finding["affected_sampled_lines"]
    assert "side" in finding["affected_observation_ids"] and "side_power" in finding["affected_target_ids"]
    target = next(t for t in reply["targets"] if t["target_id"] == "side_power")
    assert target["assessment"] == "INDETERMINATE"
    assert target["value_basis"]["value"]["kind"] == "ESTIMATE" and "LENS" in _at(target["value_basis"]["value"])
    side = next(m for m in reply["monitors"] if m["observation_id"] == "side")
    assert side["quantity_state"] == "OUT_OF_MODEL_SCOPE" and not side["arrivals"]
    (row,) = side["sampled_light"]
    assert row["power_W"] > 0.0 and "LENS" in _at(row["value_basis"]["power_W"])
    if surface == "session":
        assert "LENS" in json.dumps(reply["map"].get("model_scope")), reply["map"].get("model_scope")

    control = json_api.evaluate_project(_scattering_plate(1000, **FORWARD), list(OUTPUTS))
    assert control["ok"], control
    assert control["model_scope_issues"] == []
    assert assessments(control)["side_power"] == "MET"
    assert {m["observation_id"]: m["quantity_state"] for m in control["monitors"]}["side"] == "BOUNDED"


def test_a_finding_lists_only_the_planes_of_the_scattered_light_whose_rays_met_its_element():
    """PLATE's two faces scatter the same source line: the back face forward through the lens onto ``side``, the front face
    back onto ``back`` before the lens. Only the forward light met the lens: the lens's finding lists the line on ``side``
    (and ``side_power``), not on ``back``, whose target stays MET and whose plane keeps its state."""

    reply = json_api.evaluate_project(_scattering_plate(1200, **{**FORWARD, "scatters": ((1, "REFLECT"), (2, "TRANSMIT"))},
                                                        back=(-55.0, 10.0, 0.0)), list(OUTPUTS))
    assert reply["ok"], reply
    (finding,) = reply["model_scope_issues"]
    assert finding["instance_id"] == "LENS"
    planes = {line["observation_id"] for line in finding["affected_sampled_lines"]}
    assert "side" in planes and "back" not in planes
    assert "back" not in finding["affected_observation_ids"] and "back_power" not in finding["affected_target_ids"]
    monitors = {m["observation_id"]: m for m in reply["monitors"]}
    assert sum(row["power_W"] for row in monitors["back"]["sampled_light"]) > 0.0
    assert assessments(reply)["back_power"] == "MET" and assessments(reply)["side_power"] == "INDETERMINATE"
    assert monitors["back"]["quantity_state"] != "OUT_OF_MODEL_SCOPE" and monitors["side"]["quantity_state"] == "OUT_OF_MODEL_SCOPE"
