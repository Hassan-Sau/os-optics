"""Source coherence in the local Project workflow: the Project source declares ``linewidth_nm`` or
``coherence_length_mm``; the lowering passes it to the Engine. The two arms of the interferometer example cross different
coated catalog optics (the splitter's coating in reflection and in transmission, two coated fold mirrors), whose group
delays the catalog does not state (no phase spectrum): the arms' relative envelope delay is not known, so their degree of
coherence is not either -- the visibility and the coherence margin have no value. The example's targets are those the
design decides on test-only declared copies of the catalog optics with alpha=0 assumed: the fringe period, the intensity
balance, the polarization overlap and the coherence length, the last of which needs the declared coherence. Without
one it is missing for that reason, with an actionable message; any other target decides the run (``COHERENCE_UNDEFINED`` is informational)."""

from __future__ import annotations

import json
import math

import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

from os_optics.local import (
    COHERENCE_METRICS,
    LocalValidationStatusV1,
    example_project_of_kind,
    json_api,
    lower_project,
    target_metrics,
)
from os_optics.scene.lower import coherence_length_from_linewidth_mm

from local_helpers import catalog_document
from test_bounded_coating_coherence import _ideal_run

LINEWIDTH_NM = 0.001
WAVELENGTH_NM = 780.0


@pytest.fixture(scope="module")
def catalog():
    return catalog_document()


@pytest.fixture(scope="module")
def interferometer(catalog) -> dict:
    return example_project_of_kind("two_beam_interference", catalog).payload()


def _source_model(document: dict) -> dict:
    return document["components"][0]["source_model"]


def _with_source(document: dict, **changes) -> dict:
    document = json.loads(json.dumps(document))
    model = _source_model(document)
    for key, value in changes.items():
        if value is None:
            model.pop(key, None)
        else:
            model[key] = value
    return document


def _engine_source(document: dict, catalog):
    (component,) = [c for c in lower_project(document, catalog).request.project.components if c.source is not None]
    return component.source


def test_the_lowering_passes_the_declared_coherence_to_the_engine(interferometer, catalog):
    assert _source_model(interferometer)["linewidth_nm"] == LINEWIDTH_NM
    spec = _engine_source(interferometer, catalog)
    assert (spec.linewidth_nm, spec.coherence_length_mm, spec.coherence_group) == (LINEWIDTH_NM, None, None)
    by_length = _with_source(interferometer, linewidth_nm=None, coherence_length_mm=250.0)
    spec = _engine_source(by_length, catalog)
    assert (spec.linewidth_nm, spec.coherence_length_mm) == (None, 250.0)
    undeclared = _engine_source(_with_source(interferometer, linewidth_nm=None), catalog)
    assert (undeclared.linewidth_nm, undeclared.coherence_length_mm) == (None, None)


def test_the_interferometer_example_decides_its_targets_its_coherence_not_known_through_its_coated_optics(interferometer, catalog):
    run = _ideal_run(interferometer, catalog)
    assert run.status is LocalValidationStatusV1.PASS, run.summary()["targets"]
    assert run.model_scope_issues == ()
    (pair,) = run.result.interference
    assert pair.source_id_a == pair.source_id_b == "SRC"
    # (the beams cross at 90 deg in the example's air, 20 C and 101325 Pa: lambda / (n_air sqrt 2))
    assert pair.fringe_period_nm == pytest.approx(WAVELENGTH_NM / (air_index(WAVELENGTH_NM) * math.sqrt(2.0)), rel=1e-6)
    assert pair.coherence_length_mm == pytest.approx(coherence_length_from_linewidth_mm(WAVELENGTH_NM, LINEWIDTH_NM), rel=1e-12)
    # the arms' passes of undetermined group delay differ: the group path difference and the coherence metrics are stated at
    # no added delay at those passes, with the entry naming them
    arrivals = {a.beam.path_id: a for o in run.result.observations for a in o.arrivals if a.crossing_index == pair.crossing_index_a}
    a, b = arrivals[pair.path_id_a], arrivals[pair.path_id_b]
    assert a.group_delay_unknown_at and b.group_delay_unknown_at
    assert sorted(a.group_delay_unknown_at) != sorted(b.group_delay_unknown_at)
    assert all(entry.split(":")[0].split("/")[0] in ("BS1", "M1", "M2") for entry in (*a.group_delay_unknown_at, *b.group_delay_unknown_at))
    assert pair.coherence_factor == pytest.approx(math.exp(-((pair.group_path_difference_mm / pair.coherence_length_mm) ** 2)), rel=1e-9)
    entry = pair.value_basis["coherence_factor"]
    assert entry.kind.value == "ESTIMATE" and entry.assumed[0].input.value == "UNSTATED_GROUP_DELAY_ZERO" and entry.assumed[0].at
    assert pair.value_basis["fringe_visibility"] == pair.value_basis["coherence_margin"] == entry
    assert pair.spatial_overlap_ratio > 0.999 and pair.polarization_overlap == pytest.approx(1.0)
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    outcomes = {o.target_id: o for o in run.local_validation.simulation.target_outcomes}
    assert set(outcomes) == {"fringe_period", "intensity_balance", "polarization_overlap", "coherence_length"}
    assert all(o.status.value == "PASS" for o in outcomes.values())
    assert outcomes["coherence_length"].value == pytest.approx(pair.coherence_length_mm, rel=1e-12)


def test_a_declared_coherence_length_gives_the_same_coherence_as_the_equivalent_linewidth(interferometer, catalog):
    by_linewidth = _ideal_run(interferometer, catalog)
    length = coherence_length_from_linewidth_mm(WAVELENGTH_NM, LINEWIDTH_NM)
    by_length = _ideal_run(_with_source(interferometer, linewidth_nm=None, coherence_length_mm=length), catalog)
    a, b = by_linewidth.result.interference[0], by_length.result.interference[0]
    assert b.coherence_length_mm == length and a.coherence_length_mm == pytest.approx(length, rel=1e-12)
    assert (b.coherence_factor, b.coherence_margin, b.fringe_visibility) == (a.coherence_factor, a.coherence_margin, a.fringe_visibility)
    assert by_length.status is by_linewidth.status


def test_without_declared_coherence_a_coherence_target_is_unknown_with_a_recovery(interferometer, catalog):
    run = _ideal_run(_with_source(interferometer, linewidth_nm=None), catalog)
    assert run.status is LocalValidationStatusV1.UNKNOWN
    targets = {t.target_id: t for t in run.result.targets}
    for target_id in ("fringe_period", "intensity_balance", "polarization_overlap"):  # (they do not depend on coherence)
        assert targets[target_id].status.value == "PASS"
    target = targets["coherence_length"]
    assert target.status.value == "UNKNOWN" and target.value is None
    assert "declare the source's coherence_length_mm or linewidth_nm" in target.message
    assert "COHERENCE_UNDEFINED" in target.issue_codes
    # (the line is taken as monochromatic: the pair states its degree of coherence on that stand-in, and on no added delay
    # at the arms' passes of unstated group delay; no run issue withholds anything, and a coherence length has no value)
    (pair,) = run.result.interference
    entry = pair.value_basis["coherence_factor"]
    assert entry.kind.value == "ESTIMATE" and {a.input.value for a in entry.assumed} >= {"FULL_COHERENCE", "UNSTATED_GROUP_DELAY_ZERO"}
    assert pair.coherence_length_mm is None
    assert not [i for i in run.result.issues if i.code.value == "COHERENCE_UNDEFINED"]
    record = run.local_validation
    assert record.validation_status is LocalValidationStatusV1.UNKNOWN
    assert "COHERENCE_UNDEFINED" not in {c.code for c in record.simulation.issue_counts}


def test_without_declared_coherence_targets_that_do_not_need_it_decide_the_run(interferometer, catalog):
    document = _with_source(interferometer, linewidth_nm=None)
    document["targets"] = [t for t in document["targets"] if t["metric"] not in COHERENCE_METRICS]
    assert [t["target_id"] for t in document["targets"]] == ["fringe_period", "intensity_balance", "polarization_overlap"]
    run = _ideal_run(document, catalog)
    assert run.status is LocalValidationStatusV1.PASS and run.engine_status.value == "PASS"
    # (the visibility is stated on the monochromatic stand-in, its entry naming it: no target reads it here)
    pair = run.result.interference[0]
    assert pair.fringe_visibility is not None
    assert "FULL_COHERENCE" in {a.input.value for a in pair.value_basis["fringe_visibility"].assumed}
    assert "COHERENCE_UNDEFINED" not in run.result.summary.issue_counts


def test_two_source_components_stay_mutually_incoherent(interferometer, catalog):
    """A second laser with its own linewidth is a different source: its beams never interfere with the first one's."""
    document = json.loads(json.dumps(interferometer))
    second = json.loads(json.dumps(document["components"][0]))
    second["instance_id"] = "SRC_B"
    plane_x, plane_y, _ = interferometer["observations"][0]["pose"]["position_mm"]
    second["nominal_pose"]["position_mm"] = [plane_x - 50.0, plane_y, 0.0]  # along +X onto the crossing plane
    document["components"].append(second)
    document["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "crossing_plane", "source_id": "SRC", "minimum": 0.0, "unit": "W"}]
    run = _ideal_run(document, catalog)
    # (their pairs are counted on the plane, not analysed: incoherent_pairs; the first laser's own pairs stay analysed)
    assert not any({p.source_id_a, p.source_id_b} == {"SRC", "SRC_B"} for p in run.result.interference)
    own = [p for p in run.result.interference if p.source_id_a == p.source_id_b == "SRC"]
    beams = [a.beam.source_id for o in run.result.observations if o.observation_id == "crossing_plane" for a in o.arrivals]
    (counted,) = run.result.incoherent_pairs
    assert own and counted.pair_count == beams.count("SRC") * beams.count("SRC_B") > 0


def test_coherence_is_declared_once_and_positive(interferometer):
    both = _with_source(interferometer, coherence_length_mm=100.0)
    reply = json_api.check_project(both)
    assert reply["ok"] is False and "linewidth_nm or coherence_length_mm, not both" in json.dumps(reply["error"])
    for bad in ({"linewidth_nm": 0.0}, {"linewidth_nm": None, "coherence_length_mm": -1.0}):
        assert json_api.check_project(_with_source(interferometer, **bad))["ok"] is False


def test_the_capabilities_name_the_coherence_metrics():
    caps = json_api.capabilities()
    assert caps["coherence_metrics"] == sorted(COHERENCE_METRICS)
    assert set(COHERENCE_METRICS) <= set(target_metrics()) == set(caps["target_metrics"])
    assert "two_beam_interference" in caps["example_kinds"]
