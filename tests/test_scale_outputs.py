"""What a run of many beams writes: independent-source pairs counted, ray data recording level, streamed files.

* Pairs of beams from mutually incoherent sources are counted per plane (``ValidationResult.incoherent_pairs``, one
  ``INCOHERENT_SOURCES`` issue), not listed and analysed one by one; every pair of one coherence group, every pair a
  pair-metric target selects and every pair in media of different index is listed as before
  (``validation/pair_selection.py``), and only the listed pairs count against ``max_interference_pairs``.
* ``ray_data="summary"`` writes the chief rays only; the trace, the results and every hash are those of a full run, and a
  query that needs pupil rays refuses the run (``PUPIL_RAYS_NOT_RECORDED``).
* The run files are serialized and hashed piece by piece (``manifest.streaming``) with exactly the bytes and digests of
  the whole-document functions.
"""

from __future__ import annotations

import itertools
import json
import random
from types import SimpleNamespace

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.local import run_optical_project
from os_optics.local.artifacts import (MANIFEST_FILE, RAY_DATA_FILE, run_artifact_files, verify_run_artifacts,
                                            write_run_artifacts)
from os_optics.manifest import hash_quantized
from os_optics.manifest.streaming import dict_parts, hash_quantized_parts, json_bytes, model_parts
from os_optics.optical_view import canonical_json_bytes
from os_optics.results import ResultsError, load_run
from os_optics.validation.pair_selection import plane_pairs, same_wavelength_pair_count, target_selects_pair
from os_optics.validation.ray_paths import ray_paths_document

from composition_helpers import laser, project_of, screen
from helpers import run

LAM = 780.0


def lasers(count: int, *, extra: tuple = (), targets: tuple = ()) -> ProjectSnapshot:
    """``count`` independent lasers of one wavelength side by side, all onto one plane (each its own coherence group)."""
    sources = tuple(B.laser(f"S{i}", wavelength_nm=LAM, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, 2.0 * i, 0.0),
                            direction=(1.0, 0.0, 0.0)) for i in range(count))
    plane = B.observation_plane("P", position=(100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0)
    return ProjectSnapshot(components=(*sources, *extra), observations=(plane,), targets=targets)


def _incoherent_issues(result):
    return [i for i in result.issues if i.code.value == "INCOHERENT_SOURCES"]


def test_pairs_of_independent_sources_are_counted_not_listed():
    result = run(lasers(5))
    assert result.interference == ()
    assert [(p.observation_id, p.pair_count, p.coherence_group_count) for p in result.incoherent_pairs] == [("P", 10, 5)]
    (issue,) = _incoherent_issues(result)
    assert issue.status.value == "NOT_APPLICABLE" and issue.location == "observation:P"
    assert issue.numerical_context["pair_count"] == 10 and issue.numerical_context["coherence_group_count"] == 5
    assert result.summary.issue_counts["INCOHERENT_SOURCES"] == 1
    # (the plane's power adds the beams' powers, whatever the pairs' listing)
    assert result.observations[0].total_power_W == pytest.approx(5e-3)
    assert result.outcome.execution.outcome.value == "COMPLETED"


def test_only_listed_pairs_count_against_the_pair_cap():
    result = run(lasers(6), numerical=NumericalConfig(max_interference_pairs=1))
    assert not any(i.code.value == "MAX_BEAM_PAIRS" for i in result.issues)
    assert result.outcome.execution.complete and result.incoherent_pairs[0].pair_count == 15


def test_a_pair_target_on_two_independent_sources_reads_its_pair_analysis():
    targets = (Target(target_id="vis", observation_id="P", metric="fringe_visibility", source_id="S1", source_id_b="S3",
                      maximum=0.5),)
    result = run(lasers(4, targets=targets))
    (pair,) = result.interference
    assert {pair.source_id_a, pair.source_id_b} == {"S1", "S3"} and "INCOHERENT_SOURCES" in pair.issue_codes
    target = result.targets[0]
    assert target.value == 0.0 and "INCOHERENT_SOURCES" in target.issue_codes
    assert result.incoherent_pairs[0].pair_count == 5  # (the other five pairs of the six)


def test_a_pair_target_without_a_selector_lists_every_pair():
    targets = (Target(target_id="vis", observation_id="P", metric="fringe_visibility", maximum=0.5),)
    result = run(lasers(3, targets=targets))
    assert len(result.interference) == 3 and result.incoherent_pairs == ()
    assert result.targets[0].value is None and "INVALID_INPUT" in result.targets[0].issue_codes  # (three pairs match)


def _rec(k, group, wavelength, n=1.0, source=None, path=None):
    return SimpleNamespace(coherence_group=group, wavelength_nm=wavelength, n=n, source_id=source or f"s{group}",
                           path_id=path or f"p{k}", crossing_index=0)


def _brute(recs, targets, cfg):
    tol = cfg.interference_wavelength_tol_nm
    out = []
    for i, j in itertools.combinations(range(len(recs)), 2):
        a, b = recs[i], recs[j]
        if abs(a.wavelength_nm - b.wavelength_nm) > tol:
            continue
        if a.coherence_group == b.coherence_group or abs(a.n - b.n) > cfg.energy_conservation_tol or any(
                target_selects_pair(t, a.source_id, b.source_id, a.wavelength_nm, a.path_id, a.crossing_index, b.path_id,
                                    b.crossing_index, cfg) for t in targets):
            out.append((i, j))
    return out


def _target(**kw):
    base = dict(source_id=None, source_id_b=None, wavelength_nm=None, parameters={})
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize("seed", range(6))
def test_plane_pairs_are_exactly_the_defined_pairs_in_order(seed):
    rng = random.Random(seed)
    cfg = NumericalConfig()
    tol = cfg.interference_wavelength_tol_nm
    recs = [_rec(k, rng.choice("abcd"), rng.choice([LAM, LAM + 0.5 * tol, LAM + 3 * tol, 532.0]),
                 n=1.0 if seed % 3 else rng.choice([1.0, 1.5])) for k in range(18)]
    for k, r in enumerate(recs):
        r.source_id = f"s{r.coherence_group}{k % 2}"
    choices = [(), (_target(source_id="sa0", source_id_b="sb1"),), (_target(parameters={"path_id_a": "p3"}),),
               (_target(source_id="sc0"),), (_target(),), (_target(wavelength_nm=532.0),)]
    targets = choices[seed % len(choices)]
    index = {id(r): k for k, r in enumerate(recs)}
    found = [(index[id(a)], index[id(b)]) for a, b in plane_pairs(recs, targets, cfg)]
    assert found == _brute(recs, targets, cfg)
    total = sum(abs(a.wavelength_nm - b.wavelength_nm) <= tol for a, b in itertools.combinations(recs, 2))
    assert same_wavelength_pair_count([r.wavelength_nm for r in recs], cfg) == total


def _canonical_run(tmp_path, ray_data="full"):
    project = project_of([laser("A", (-100.0, 0.0, 0.0)), laser("B", (-100.0, 3.0, 0.0))], [screen("P", (100.0, 0.0, 0.0))])
    return run_optical_project(project, ray_data=ray_data)


def test_streamed_result_and_ray_data_have_the_whole_document_bytes_and_hashes(tmp_path):
    run_ = _canonical_run(tmp_path)
    result = run_.result
    assert json_bytes(model_parts(result)) == canonical_json_bytes(result.model_dump(mode="json"))
    body = result.model_copy(update={"manifest": None}).model_dump(mode="json")
    cfg = run_.request.configuration.numerical
    digest = hash_quantized(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor)
    assert hash_quantized_parts(model_parts(result, update={"manifest": None}), cfg.hash_float_sig_digits,
                                cfg.hash_float_abs_floor) == digest == result.manifest.result_hash
    assert hash_quantized_parts(dict_parts(body), cfg.hash_float_sig_digits, cfg.hash_float_abs_floor) == digest
    files = run_artifact_files(run_)
    assert files[RAY_DATA_FILE] == canonical_json_bytes(ray_paths_document(run_.ray_paths))
    written = write_run_artifacts(run_, tmp_path / "run")
    assert {name: (tmp_path / "run" / name).read_bytes() for name in files} == files
    assert written.manifest == json.loads(files[MANIFEST_FILE])
    assert verify_run_artifacts(tmp_path / "run") == ()
    assert json_bytes({}) == b"{}\n" and json_bytes({"a": dict_parts({"x": []})["x"]}) == canonical_json_bytes({"a": []})


def test_summary_ray_data_keeps_the_run_and_states_what_it_left_out(tmp_path):
    full = _canonical_run(tmp_path)
    summary = _canonical_run(tmp_path, ray_data="summary")
    assert summary.result == full.result and summary.local_validation == full.local_validation
    files = write_run_artifacts(summary, tmp_path / "summary")
    assert verify_run_artifacts(tmp_path / "summary") == ()
    document = json.loads((tmp_path / "summary" / RAY_DATA_FILE).read_bytes())
    assert document["recording"] == "summary" and document["trace_hash"] == full.result.manifest.trace_hash
    assert all("rays" not in p["pupil"] and p["pupil"]["ray_count"] > 0 and p["chief"]["points_mm"] for p in document["paths"])
    manifest = files.manifest
    assert manifest["ray_data"]["recording"] == "summary"
    assert "recording" not in json.loads(run_artifact_files(full)[MANIFEST_FILE])["ray_data"]
    sizes = {entry["role"]: entry["bytes"] for entry in manifest["files"]}
    full_sizes = {entry["role"]: entry["bytes"] for entry in json.loads(run_artifact_files(full)[MANIFEST_FILE])["files"]}
    assert sizes["ray_data"] < full_sizes["ray_data"] / 10
    assert {k: v for k, v in sizes.items() if k != "ray_data"} == {k: v for k, v in full_sizes.items() if k != "ray_data"}
    loaded = load_run(tmp_path / "summary")
    assert loaded.describe()["available"]["ray_data_recording"] == "summary"
    assert loaded.describe()["available"]["wavefront"] is False
    with pytest.raises(ResultsError) as refused:
        loaded.rays()
    assert refused.value.code == "PUPIL_RAYS_NOT_RECORDED"
    front = loaded.wavefront("P", full.result.paths[0].path_id)
    assert front["status"] == "UNAVAILABLE" and front["reason"] == "PUPIL_RAYS_NOT_RECORDED"
    # (what the chief rays alone answer still works)
    assert [row["path_id"] for row in loaded.paths()] == [row["path_id"] for row in load_run(full).paths()]


def test_a_ray_data_level_is_full_or_summary():
    with pytest.raises(ValueError):
        run_optical_project(project_of([laser("A")], [screen("P", (100.0, 0.0, 0.0))]), ray_data="none")


def test_the_command_line_records_a_summary(tmp_path, capsys):
    from os_optics.cli import main

    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps(project_of([laser("A")], [screen("P", (100.0, 0.0, 0.0))])), encoding="utf-8")
    main(["project", "run", str(project_path), "--out-dir", str(tmp_path / "run"), "--ray-data", "summary"])
    capsys.readouterr()
    manifest = json.loads((tmp_path / "run" / MANIFEST_FILE).read_bytes())
    assert manifest["ray_data"]["recording"] == "summary"
    assert verify_run_artifacts(tmp_path / "run") == ()


def test_the_json_api_takes_the_recording_level_and_refuses_another():
    from os_optics.local import json_api

    document = project_of([laser("A")], [screen("P", (100.0, 0.0, 0.0))])
    reply = json_api.run_project(document, ray_data="summary")
    assert reply["ok"] and json.loads(reply["files"][RAY_DATA_FILE])["recording"] == "summary"
    refused = json_api.run_project(document, ray_data="pupil")
    assert refused["ok"] is False and "ray_data" in json.dumps(refused["error"])
