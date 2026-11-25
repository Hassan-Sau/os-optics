"""Reference designs built by catalog query (never a SKU): their focus / afocal / imaging targets stay
strictly decidable, while absent public substrate loss or coating phase is reported as UNKNOWN, not as a false PASS."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys

import pytest

from os_optics.local import (
    EXAMPLE_KINDS,
    LocalRunError,
    LocalValidationStatusV1,
    ProjectDocumentV2,
    example_project_of_kind,
    json_api,
    load_public_catalog,
    lower_project,
    run_optical_project,
)
from os_optics.local import example as example_module

from local_helpers import TEST_CATALOG_VERSION, catalog_document

KIND_ELEMENTS = {
    "focus": ["source", "lens"],
    "folded": ["source", "mirror", "lens"],
    "beam_expander": ["source", "lens", "lens"],
    "relay_4f": ["source", "lens", "lens"],
    "folded_expander": ["source", "lens", "mirror", "lens"],
    "dichroic_routing": ["source", "source", "splitter"],
    "polarization": ["source", "waveplate", "polarizer"],
    "two_beam_interference": ["source", "splitter", "mirror", "mirror"],
    "two_color_dichroic": ["source", "splitter", "detector", "detector"],
    "dichroic_combiner": ["source", "source", "splitter", "detector"],
}


@pytest.fixture(scope="module")
def bundled_runs():
    return {kind: run_optical_project(example_project_of_kind(kind)) for kind in EXAMPLE_KINDS}


def test_the_kinds():
    assert EXAMPLE_KINDS == tuple(KIND_ELEMENTS)


def _example_outcomes_hold(kind, run) -> None:
    """Check each geometric target strictly; do not invent bulk transmission or coating phase for unknown power."""
    outcomes = run.local_validation.simulation.target_outcomes
    assert outcomes and len(outcomes) == len(run.project.targets), run.summary()["targets"]
    by_id = {t.target_id: t for t in run.result.targets}
    for outcome in outcomes:
        target = by_id[outcome.target_id]
        assert outcome.status == target.status, outcome.target_id
        if outcome.status is LocalValidationStatusV1.UNKNOWN:
            assert outcome.value is None and target.assessment.value == "INDETERMINATE"
            assert set(target.issue_codes) & {"BULK_LOSS_NOT_ESTABLISHED", "PHASE_UNDEFINED"}, target
        else:
            assert outcome.status is LocalValidationStatusV1.PASS, target
    assert all(t.status.value == "PASS" for t in run.result.targets if t.metric in
               ("waist_distance_x_mm", "waist_radius_x_mm", "beam_radius_mm")), run.summary()["targets"]
    bulk_unknown = "BULK_LOSS_NOT_ESTABLISHED" in {i.code.value for i in run.result.issues}
    unknown_target = any(o.status is LocalValidationStatusV1.UNKNOWN for o in outcomes)
    expected = LocalValidationStatusV1.UNKNOWN if bulk_unknown or unknown_target else LocalValidationStatusV1.PASS
    assert run.status is expected and run.engine_status.value == expected.value, run.summary()["targets"]
    if bulk_unknown:
        assert any(a.beam.power_W is None and a.power_upper_bound_W is not None
                   for observation in run.result.observations for a in observation.arrivals)


@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_every_example_passes_on_the_bundled_release(kind, bundled_runs):
    run = bundled_runs[kind]
    _example_outcomes_hold(kind, run)
    assert run.model_scope_issues == ()
    assert [c.optical_role.value for c in run.project.components] == KIND_ELEMENTS[kind]
    catalog = load_public_catalog()
    assert {c.product.catalog_version for c in run.project.components if c.product is not None} == {catalog.catalog_version}


@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_every_example_is_built_from_any_catalog_with_suitable_products(kind):
    document = catalog_document()
    project = example_project_of_kind(kind, document)
    run = run_optical_project(project, catalog=document)
    _example_outcomes_hold(kind, run)
    assert run.model_scope_issues == ()
    assert {c.product.catalog_version for c in project.components if c.product is not None} == {TEST_CATALOG_VERSION}


def test_examples_never_name_a_sku_or_a_catalog_release():
    source = inspect.getsource(example_module)
    assert "OS-OPT-" not in source
    assert not re.search(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\.[0-9]+\b", source)


def test_examples_are_deterministic_and_exact_decimal_positions():
    for kind in EXAMPLE_KINDS:
        first, second = example_project_of_kind(kind), example_project_of_kind(kind)
        assert first.content_hash() == second.content_hash()
        assert lower_project(first).engine_input_hash == lower_project(second).engine_input_hash
        for component in first.components:
            for value in component.nominal_pose.position_mm:
                assert round(value, 3) == value  # positions are micrometre decimals


def test_the_focus_example_places_its_screen_at_the_predicted_waist(bundled_runs):
    target = next(t for t in bundled_runs["focus"].result.targets if t.target_id == "focus_position")
    assert abs(target.value) < 0.001  # only the micrometre rounding of the screen position remains


def test_the_expanders_and_relay_hold_their_first_order_design(bundled_runs):
    for kind in ("beam_expander", "folded_expander"):
        (target,) = bundled_runs[kind].result.targets
        assert target.value == pytest.approx(target.target_value, rel=1e-4)
    relay = {t.target_id: t.value for t in bundled_runs["relay_4f"].result.targets}
    assert abs(relay["image_position"]) < 0.002 and relay["image_waist"] == pytest.approx(0.1, rel=1e-6)
    relay_skus = [c.product.optical_sku for c in bundled_runs["relay_4f"].project.components if c.product is not None]
    assert len(relay_skus) == 2 and relay_skus[0] == relay_skus[1]  # one product used twice


@pytest.mark.parametrize("bundled", [True, False])
def test_the_dichroic_combiner_is_the_time_reversed_routing(bundled, bundled_runs):
    """The combiner uses the routing example's dichroic and lines; the transmitted laser meets the dichroic from
    behind its coated face. Both lasers end centred on the one detector with the powers of the routing legs."""

    if bundled:
        routing, combiner = bundled_runs["dichroic_routing"], bundled_runs["dichroic_combiner"]
    else:
        document = catalog_document()
        routing, combiner = (run_optical_project(example_project_of_kind(kind, document), catalog=document)
                             for kind in ("dichroic_routing", "dichroic_combiner"))
    _example_outcomes_hold("dichroic_routing", routing)
    _example_outcomes_hold("dichroic_combiner", combiner)
    assert combiner.model_scope_issues == ()

    def product(run):
        return next(c.product.optical_sku for c in run.project.components if c.product is not None)

    def lines(run):
        return {c.instance_id: c.source_model.wavelength_nm for c in run.project.components if c.source_model is not None}

    assert product(combiner) == product(routing) and lines(combiner) == lines(routing)
    routed = {t.target_id: t.value for t in routing.result.targets}
    combined = {t.target_id: t.value for t in combiner.result.targets}
    assert combined.keys() == routed.keys()
    for target_id in routed:
        assert combined[target_id] == (pytest.approx(routed[target_id], rel=1e-12) if routed[target_id] is not None else None)
    # For the transmitted glass path neither orientation establishes bulk loss; compare only the public upper bounds.
    routed_arrivals = {source_id: max((a for o in routing.result.observations for a in o.arrivals if a.beam.source_id == source_id),
                                       key=lambda a: a.beam.power_W if a.beam.power_W is not None else a.power_upper_bound_W)
                       for source_id in ("SRC_R", "SRC_T")}
    combined_arrivals = {a.beam.source_id: a for o in combiner.result.observations for a in o.arrivals}
    for source_id in ("SRC_R", "SRC_T"):
        source_routed = routed_arrivals[source_id]
        source_combined = combined_arrivals[source_id]
        if source_routed.beam.power_W is None:
            assert source_combined.beam.power_W is None
            assert source_combined.power_upper_bound_W == pytest.approx(source_routed.power_upper_bound_W, rel=1e-12)
        else:
            assert source_combined.beam.power_W == pytest.approx(source_routed.beam.power_W, rel=1e-12)
    (observation,) = combiner.result.observations
    assert sorted(a.beam.source_id for a in observation.arrivals) == ["SRC_R", "SRC_T"]
    assert all(abs(v) <= 0.001 for a in observation.arrivals for v in a.uv_mm)  # co-linear up to the micrometre rounding
    transmitted = next(p for p in combiner.result.paths if p.source_id == "SRC_T" and p.parent_path_id is None)
    pose = next(c.nominal_pose for c in combiner.project.components if c.instance_id == "DM1")
    w, x, y, z = pose.quaternion_wxyz
    normal = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))  # the dichroic's local +Z (coated face normal)
    first = transmitted.segments[0]
    assert sum((first.end_mm[k] - first.start_mm[k]) * normal[k] for k in range(3)) > 0.0  # it arrives from behind


def test_a_catalog_without_suitable_products_gives_example_unavailable():
    document = catalog_document() | {"catalog_version": "2026-01-01.3"}
    document["products"] = [p for p in document["products"] if p["category"] != "WAVEPLATE"]
    document["categories_offered"] = [c for c in document["categories_offered"] if c != "WAVEPLATE"]
    with pytest.raises(LocalRunError) as exc:
        example_project_of_kind("polarization", document)
    assert [c.value for c in exc.value.codes] == ["EXAMPLE_UNAVAILABLE"] and "2026-01-01.3" in str(exc.value)
    with pytest.raises(LocalRunError) as exc:
        example_project_of_kind("kaleidoscope")
    assert [c.value for c in exc.value.codes] == ["EXAMPLE_UNAVAILABLE"] and "beam_expander" in str(exc.value)


def test_examples_through_the_json_api_and_the_cli():
    for kind in EXAMPLE_KINDS:
        reply = json_api.example_project_document(kind=kind)
        assert reply["ok"] is True and reply["kind"] == kind
        assert reply["project"] == example_project_of_kind(kind).payload()
    assert json_api.example_project_document()["kind"] == "focus"
    unknown = json_api.example_project_document(kind="kaleidoscope")
    assert unknown["ok"] is False and [p["code"] for p in unknown["error"]["problems"]] == ["EXAMPLE_UNAVAILABLE"]
    proc = subprocess.run([sys.executable, "-m", "os_optics", "project", "example", "--kind", "relay_4f"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert ProjectDocumentV2.model_validate(json.loads(proc.stdout)).payload() == example_project_of_kind("relay_4f").payload()
    clash = subprocess.run([sys.executable, "-m", "os_optics", "project", "example", "--folded", "--kind", "focus"], capture_output=True,
                           text=True, timeout=120)
    assert clash.returncode == 64
