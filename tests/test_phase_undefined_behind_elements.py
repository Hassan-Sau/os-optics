"""No phase and no optical path length behind an element whose model states no optical path: an ideal or tabulated thin lens, a
Faraday rotator, a diffraction grating (an acousto-optic deflector is one). ``ArrivalResult.ray_opl_defined`` is false there, and every
reader of the arrival says the same: the result, ``monitor``, the arrivals table and export, the handoff, ``json_api.run_results``, the
CLI and a target. Glass, which states its optical path, keeps both. The difference of two beams' optical paths is stated only where the
passes they do not share state theirs. This module checks the retained headless result and JSON transports.
"""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from os_optics.builders import laser, mirror_for_reflection, npbs, observation_plane, thin_lens
from os_optics.config import NumericalConfig
from os_optics.contracts import ArrivalResult, ProjectSnapshot, Target
from os_optics.contracts.optics import InterfacePhases
from os_optics.local import json_api, run_optical_project
from os_optics.results import load_run
from os_optics.validation.superposition import irradiance

from helpers import run as run_snapshot
from local_helpers import ALONG_X, plane, project, source
from test_native_library import ALL_CLASSES

SNAPSHOTS = Path(__file__).parent / "fixtures" / "snapshots"
SOURCE_X = -60.0
ELEMENT_X = 0.0
PLANE_X = 40.0
PHASE_TARGET = {"target_id": "PHASE", "observation_id": "OUT", "metric": "phase_rad", "unit": "rad", "minimum": -10.0}

# class, its wavelength; each in the band the fixture parameters state
UNDEFINED = {
    "lens.ideal_thin": 633.0,
    "lens.tabulated_thin": 532.0,
    "rotator.faraday": 1060.0,
    "grating.plane.transmission": 633.0,
    "deflector.acousto_optic": 633.0,
}
GRATING_ZERO_ORDER = {**ALL_CLASSES["grating.plane"], "mode": "TRANSMISSION", "orders": [0]}


def _built(name: str, model_id: str = "m") -> dict:
    class_id, parameters = ("grating.plane", GRATING_ZERO_ORDER) if name == "grating.plane.transmission" else (name, ALL_CLASSES[name])
    built = json_api.build_native_model(class_id, {**parameters, "model_id": model_id})
    assert built["ok"], built
    return built


def _component(built: dict, instance_id: str, x: float) -> dict:
    return {**built["component_template"], "instance_id": instance_id, "nominal_pose": {"position_mm": [x, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}


def _document(wavelength_nm: float, *elements: tuple[dict, str, float], targets=()) -> dict:
    document = project([source("SRC", (SOURCE_X, 0.0, 0.0), wavelength_nm=wavelength_nm, waist_mm=0.5), *(_component(b, i, x) for b, i, x in elements)],
                       [plane("OUT", (PLANE_X, 0.0, 0.0), radius_mm=50.0)], list(targets))
    document["design_intent"] = "OPTICAL_SYSTEM"
    document["optical_models"] = [b["optical_model"] for b, _, _ in elements]
    materials = {m["material_id"]: m for b, _, _ in elements for m in b.get("model_materials") or ()}
    coatings = {c["coating_id"]: c for b, _, _ in elements for c in b.get("model_coatings") or ()}
    if materials:
        document["model_materials"] = list(materials.values())
    if coatings:
        document["model_coatings"] = list(coatings.values())
    return document


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _cli(*args: str) -> str:
    done = subprocess.run([sys.executable, "-m", "os_optics", *args], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _phases_on_every_consumer(run, directory: Path) -> dict[str, list]:
    """``phase_rad`` of the arrivals on plane OUT, as each consumer reports them (an empty CSV cell is ``None``)."""

    run.write_artifacts(directory)
    results = load_run(directory)
    files = _files(directory)
    answered = json_api.run_results(files, {"kind": "monitor", "monitor": "OUT"})
    assert answered["ok"], answered
    exported = json_api.run_results(files, {"kind": "export", "export_kind": "arrivals", "format": "jsonl"})
    assert exported["ok"], exported
    return {
        "result": [a.beam.phase_rad for a in run.result.observations[0].arrivals],
        "arrivals": [row["phase_rad"] for row in results.arrivals("OUT")],
        "monitor": [a["beam"]["phase_rad"] for a in results.monitor("OUT")["arrivals"]],
        "handoff": [beam["phase_rad"] for beam in results.handoff("OUT")["beams"]],
        "json_api monitor": [a["beam"]["phase_rad"] for a in answered["result"]["arrivals"]],
        "json_api export": [json.loads(text)["phase_rad"] for text in exported["result"]["data"].splitlines() if text.strip()],
        "cli export": [row["phase_rad"] or None for row in csv.DictReader(io.StringIO(_cli("export", "arrivals", str(directory), "--format", "csv")))],
        "cli monitor": [a["beam"]["phase_rad"] for a in json.loads(_cli("results", "monitor", str(directory), "OUT"))["arrivals"]],
    }


@pytest.mark.parametrize("name", list(UNDEFINED))
def test_every_consumer_reads_no_phase(name, tmp_path):
    run = run_optical_project(_document(UNDEFINED[name], (_built(name), "E", ELEMENT_X)), catalog=None)
    arrivals = run.result.observations[0].arrivals
    assert arrivals and all(a.ray_opl_defined is False for a in arrivals)
    phases = _phases_on_every_consumer(run, tmp_path)
    assert phases == dict.fromkeys(phases, [None] * len(arrivals)), phases


@pytest.mark.parametrize("name", [name for name in UNDEFINED if name != "deflector.acousto_optic"])  # (several orders: a target needs one beam)
def test_a_phase_target_has_no_value(name):
    run = run_optical_project(_document(UNDEFINED[name], (_built(name), "E", ELEMENT_X), targets=[PHASE_TARGET]), catalog=None)
    (target,) = run.result.targets
    assert target.value is None
    if name == "lens.tabulated_thin":  # (a metric the table does not validate is not evaluated, whatever it is)
        assert target.status.value == "OUT_OF_MODEL_SCOPE" and "METRIC_NOT_VALIDATED_FOR_ELEMENT" in target.issue_codes
    else:
        assert target.status.value == "UNKNOWN" and "PHASE_UNDEFINED" in target.issue_codes


def test_glass_behind_the_element_does_not_bring_the_phase_back(tmp_path):
    glass = _built("lens.singlet", "glass")
    run = run_optical_project(_document(UNDEFINED["rotator.faraday"], (_built("rotator.faraday"), "E", ELEMENT_X), (glass, "GLASS", 20.0)), catalog=None)
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.ray_opl_defined is False and arrival.beam.phase_rad is None


def test_glass_alone_keeps_its_phase_on_every_consumer(tmp_path):
    """The control: an element that states its optical path. The handoff is the arrival's phase; the rest read the same number."""

    run = run_optical_project(_document(633.0, (_built("lens.singlet", "glass"), "E", ELEMENT_X), targets=[PHASE_TARGET]), catalog=None)
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.ray_opl_defined is True and isinstance(arrival.beam.phase_rad, float)
    phases = _phases_on_every_consumer(run, tmp_path)
    assert {float(value) for values in phases.values() for value in values} == {arrival.beam.phase_rad}, phases
    (target,) = run.result.targets
    assert target.value == arrival.beam.phase_rad and target.status.value == "PASS"


@pytest.mark.parametrize("path", sorted(SNAPSHOTS.glob("*.json")), ids=lambda path: path.stem)
def test_recorded_scenes_hold_no_phase_where_no_optical_path_is_stated(path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    stated = [(o["observation_id"], a["beam"]["path_id"]) for o in snapshot.get("observations", ()) for a in o.get("arrivals", ())
              if a.get("ray_opl_defined") is False and (a["beam"].get("phase_rad") is not None or a["beam"].get("optical_path_length_mm") is not None)]
    assert stated == []
    for o in snapshot.get("observations", ()):
        for a in o.get("arrivals", ()):
            assert (a.get("relative_path_mm") is None) == (a.get("ray_opl_defined") is not False), (path.stem, a["beam"]["path_id"])


# ------------------------------------------------------------------------------------------------- the optical path length


OPL_TARGET = {"target_id": "PATH", "observation_id": "OUT", "metric": "optical_path_length_mm", "unit": "mm", "minimum": 0.0}


def _paths_on_every_consumer(run, directory: Path) -> dict[str, list]:
    """The optical path length of everything that reaches plane OUT, as each consumer reports it (an empty CSV cell is ``None``)."""

    run.write_artifacts(directory)
    results = load_run(directory)
    files = _files(directory)
    answered = json_api.run_results(files, {"kind": "monitor", "monitor": "OUT"})
    assert answered["ok"], answered
    exported = json_api.run_results(files, {"kind": "export", "export_kind": "arrivals", "format": "jsonl"})
    assert exported["ok"], exported
    return {
        "result": [a.beam.optical_path_length_mm for a in run.result.observations[0].arrivals],
        "arrivals": [row["optical_path_length_mm"] for row in results.arrivals("OUT")],
        "monitor": [a["beam"]["optical_path_length_mm"] for a in results.monitor("OUT")["arrivals"]],
        "handoff": [beam["optical_path_length_mm"] for beam in results.handoff("OUT")["beams"]],
        "json_api monitor": [a["beam"]["optical_path_length_mm"] for a in answered["result"]["arrivals"]],
        "json_api export": [json.loads(text)["optical_path_length_mm"] for text in exported["result"]["data"].splitlines() if text.strip()],
        "cli export": [row["optical_path_length_mm"] or None for row in csv.DictReader(io.StringIO(_cli("export", "arrivals", str(directory), "--format", "csv")))],
        "cli monitor": [a["beam"]["optical_path_length_mm"] for a in json.loads(_cli("results", "monitor", str(directory), "OUT"))["arrivals"]],
    }


@pytest.mark.parametrize("name", list(UNDEFINED))
def test_every_consumer_reads_no_optical_path_length(name, tmp_path):
    run = run_optical_project(_document(UNDEFINED[name], (_built(name), "E", ELEMENT_X)), catalog=None)
    arrivals = run.result.observations[0].arrivals
    lengths = _paths_on_every_consumer(run, tmp_path)
    assert lengths == dict.fromkeys(lengths, [None] * len(arrivals)), lengths
    results = load_run(tmp_path)
    # no other name for it in a reader
    assert all("relative_path_mm" not in a for a in results.monitor("OUT")["arrivals"])
    assert all("relative_path_mm" not in row for row in results.arrivals("OUT"))
    # the path tree: a path behind the element states none, in the library and the CLI
    behind = {a.beam.path_id for a in arrivals}
    for rows in (results.paths(), json.loads(_cli("results", "paths", str(tmp_path)))):
        for row in rows:
            if row["path_id"] in behind:
                assert row["optical_path_length_mm"] is None, row
    # every recorded step: the flag and the value agree, in the table and in the run's file
    rays = results.rays()
    assert rays and all((row["opl_mm"] is None) == (not row["opl_defined"]) for row in rays)
    assert any(row["opl_mm"] is None for row in rays)
    for path in json.loads((tmp_path / "ray_data.json").read_text(encoding="utf-8"))["paths"]:
        assert [value is not None for value in path["chief"]["opl_mm"]] == path["opl_defined"], path["path_id"]


@pytest.mark.parametrize("name", [name for name in UNDEFINED if name != "deflector.acousto_optic"])
def test_an_optical_path_length_target_has_no_value_and_does_not_pass(name):
    run = run_optical_project(_document(UNDEFINED[name], (_built(name), "E", ELEMENT_X), targets=[OPL_TARGET]), catalog=None)
    (target,) = run.result.targets
    assert target.value is None
    if name == "lens.tabulated_thin":
        assert target.status.value == "OUT_OF_MODEL_SCOPE" and "METRIC_NOT_VALIDATED_FOR_ELEMENT" in target.issue_codes
    else:
        assert target.status.value == "UNKNOWN" and "PHASE_UNDEFINED" in target.issue_codes
        assert "no optical path" in target.message


def test_glass_alone_keeps_its_optical_path_length_on_every_consumer(tmp_path):
    """The control. Glass reads longer than the geometric path."""

    run = run_optical_project(_document(633.0, (_built("lens.singlet", "glass"), "E", ELEMENT_X), targets=[OPL_TARGET]), catalog=None)
    (arrival,) = run.result.observations[0].arrivals
    assert arrival.ray_opl_defined is True and arrival.relative_path_mm is None
    assert arrival.beam.optical_path_length_mm > arrival.beam.geometric_length_mm
    lengths = _paths_on_every_consumer(run, tmp_path)
    assert {float(value) for values in lengths.values() for value in values} == {arrival.beam.optical_path_length_mm}, lengths
    (target,) = run.result.targets
    assert target.value == arrival.beam.optical_path_length_mm and target.status.value == "PASS"
    assert all(row["optical_path_length_mm"] is not None for row in load_run(tmp_path).paths())
    assert all(row["opl_mm"] is not None and row["opl_defined"] for row in load_run(tmp_path).rays())


# A Mach-Zehnder with a thin lens: before the first splitter both arms cross it, so it cancels; in one arm it does not.
X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
ARM_MM = 100.0
ROOT_HALF = math.sqrt(0.5)
FOCAL_MM = 5000.0


def _mach_zehnder(lens_x: float, target=None) -> ProjectSnapshot:
    phases = InterfacePhases(reflection_s_rad=(math.pi / 2.0,), reflection_p_rad=(3.0 * math.pi / 2.0,), transmission_s_rad=(0.0,), transmission_p_rad=(0.0,))

    def splitter(name, position):
        component = npbs(name, reflectance=0.5, position=position, normal=(-ROOT_HALF, ROOT_HALF, 0.0))
        surface = component.surfaces[0]
        return component.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": surface.interaction.model_copy(update={"phases": phases})}),)})

    parts = (laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(-50.0, 0.0, 0.0), direction=X, coherence_length_mm=1000.0),
             thin_lens("T", focal_length_mm=FOCAL_MM, clear_radius_mm=10.0, position=(lens_x, 0.0, 0.0), axis=X),
             splitter("BS1", (0.0, 0.0, 0.0)), mirror_for_reflection("M1", position=(ARM_MM, 0.0, 0.0), d_in=X, d_out=Y),
             mirror_for_reflection("M2", position=(0.0, ARM_MM, 0.0), d_in=Y, d_out=X), splitter("BS2", (ARM_MM, ARM_MM, 0.0)))
    planes = (observation_plane("OUT", position=(ARM_MM + 60.0, ARM_MM, 0.0), normal=X), observation_plane("OUT_Y", position=(ARM_MM, ARM_MM + 60.0, 0.0), normal=Y))
    return ProjectSnapshot(components=parts, observations=planes, targets=() if target is None else (target,))


def _opd_target(metric: str = "optical_path_difference_mm") -> Target:
    return Target(target_id="OPD", metric=metric, observation_id="OUT", minimum=-1000.0, maximum=1000.0)


def test_a_lens_both_beams_cross_cancels_in_their_difference_and_their_lengths_stay_unstated():
    result = run_snapshot(_mach_zehnder(-20.0, target=_opd_target()))
    (pair,) = [p for p in result.interference if p.observation_id == "OUT"]
    assert pair.optical_path_length_mm_a is None and pair.optical_path_length_mm_b is None
    assert pair.optical_path_difference_mm == pytest.approx(0.0, abs=1e-9)
    (target,) = result.targets
    assert target.status.value == "PASS" and target.value == pytest.approx(0.0, abs=1e-9)


def test_the_coherent_sum_behind_a_shared_lens_still_conserves_power_and_reads_back_from_its_json():
    """The sum needs the two beams' difference, so the result keeps it (``relative_path_mm``), under its own name."""

    result = run_snapshot(_mach_zehnder(-20.0))
    observed = {o.observation_id: o for o in result.observations}
    assert sum(observed[name].superpositions[0].total_power_W for name in ("OUT", "OUT_Y")) == pytest.approx(1e-3, rel=1e-9)
    out = observed["OUT"]
    assert all(a.beam.optical_path_length_mm is None and a.relative_path_mm is not None for a in out.arrivals)
    points = np.array([[0.0, 0.0], [0.2, -0.1]])
    direct = irradiance(list(out.arrivals), out.plane, points, NumericalConfig())
    reread = irradiance([ArrivalResult.model_validate(a.model_dump(mode="json")) for a in out.arrivals], out.plane, points, NumericalConfig())
    assert np.array_equal(direct, reread)


@pytest.mark.parametrize("metric", ["optical_path_difference_mm", "optical_path_difference_abs_mm"])
def test_a_lens_in_one_arm_leaves_the_optical_path_difference_unstated(metric):
    # on arm 1, between the first splitter and its mirror
    result = run_snapshot(_mach_zehnder(50.0, target=_opd_target(metric)))
    (pair,) = [p for p in result.interference if p.observation_id == "OUT"]
    assert pair.optical_path_difference_mm is None and "PHASE_UNDEFINED" in pair.issue_codes
    assert {pair.optical_path_length_mm_a is None, pair.optical_path_length_mm_b is None} == {True, False}
    (target,) = result.targets
    assert target.value is None and target.status.value == "UNKNOWN" and "PHASE_UNDEFINED" in target.issue_codes


def test_mutually_incoherent_beams_with_a_lens_on_one_read_no_difference_and_no_extra_issue():
    parts = (laser("A", wavelength_nm=633.0, waist_radius_mm=0.5, position=(-50.0, 0.0, 0.0), direction=X),
             laser("B", wavelength_nm=633.0, waist_radius_mm=0.5, position=(-50.0, 30.0, 0.0), direction=X),
             thin_lens("T", focal_length_mm=FOCAL_MM, clear_radius_mm=5.0, position=(0.0, 30.0, 0.0), axis=X))
    snapshot = ProjectSnapshot(components=parts, observations=(observation_plane("OUT", position=(60.0, 0.0, 0.0), normal=X, radius_mm=100.0),), targets=(_opd_target(),))
    result = run_snapshot(snapshot)
    (pair,) = result.interference
    assert pair.optical_path_difference_mm is None and "INCOHERENT_SOURCES" in pair.issue_codes and "PHASE_UNDEFINED" not in pair.issue_codes
    (target,) = result.targets
    assert target.value is None and target.status.value == "UNKNOWN" and "PHASE_UNDEFINED" in target.issue_codes and "no optical path" in target.message


def test_only_a_pass_that_states_no_optical_path_opens_the_difference():
    """A coating's open phase is no open path: n x length is still defined on both arms."""

    from os_optics.optics.interference.coherence import path_open

    lens = "T:thin_lens:+@0.000000(0.000000,0.000000)"
    assert path_open(("M:coating_reflection:+@45.000000",), ()) is False
    assert path_open((lens,), ()) is True
    assert path_open(("F:faraday_rotator:+@0.000000",), ()) is True
    assert path_open(("G:grating_order_+1:+@10.000000(0.000000,0.000000)",), ()) is True
    assert path_open((lens,), (lens,)) is False
