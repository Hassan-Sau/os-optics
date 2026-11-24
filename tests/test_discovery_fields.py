"""Discovery fields: metric descriptors name their result fields, native classes state their availability and
catalog equivalents, 'os-optics schema' lists the linear optical network, and results identity carries the
engine_input_hash name the network provenance uses."""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout

import pytest

from os_optics._vendor.shared_contracts.design_catalog import OpticalCategory, OpticalForm
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.local.json_api import SUMMARY_SPECIFICATIONS
from os_optics.native_library import native_components
from os_optics.results import columns, load_run
from os_optics.validation.descriptors import metric_descriptors

from local_helpers import PCX_F50, catalog_document, element, plane, project, run, source


def _resolve(document, path: str):
    """A 'monitor:' result field path (``a[].b.c[0].d[name].e``) in a monitor() document: True when the field exists.
    ``[]`` is every entry of a list, ``[0]`` its first entry, ``[name]`` the entry whose ``name`` is that."""

    values = [document]
    for part in path.split("."):
        name, index = re.fullmatch(r"([A-Za-z0-9_]+)(\[\]|\[\d\]|\[[A-Za-z_][A-Za-z0-9_]*\])?", part).groups()
        found = []
        for value in values:
            if not isinstance(value, dict) or name not in value:
                return False
            item = value[name]
            if index == "[]":
                found.extend(item)
            elif index is not None and index[1:-1].isdigit():
                found.append(item[int(index[1:-1])])
            elif index is not None:
                found.extend(entry for entry in item if isinstance(entry, dict) and entry.get("name") == index[1:-1])
            else:
                found.append(item)
        values = found
    return bool(values)


def _detection_monitor() -> dict:
    """monitor() of an absorbing plane whose detection states the detector's electrical response and a lock-in, lit by a
    laser chopped at 1 kHz."""

    char = {"kind": "DATASHEET", "reference": "photodiode datasheet rev B"}
    qe = {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 900.0, "qe": 0.8}], "source": char}
    detector = {**plane("D", (40.0, 0.0, 0.0), radius_mm=20.0), "absorbing": True, "detection": {
        "detector": {"quantum_efficiency": qe, "electrical": {
            "bandwidth_Hz": 10000.0, "bandwidth_definition": "FIRST_ORDER_3DB", "dark_current_A": 1e-12, "load_resistance_ohm": 1e8,
            "temperature_K": 295.0, "source": char}},
        "lock_in": {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": 0.5, "source_ids": ["SRC"]},
                    "time_constant_s": 0.1, "filter_order": 2}}}
    doc = project([source(position=(-100.0, 0.0, 0.0), waist_mm=0.5, power_w=1e-6)], [detector])  # (all its power is known)
    return load_run(run(doc)).monitor("D")


def test_every_result_field_of_a_metric_exists_in_the_results():
    doc = project([source(position=(-100.0, 0.0, 0.0), waist_mm=0.5), element("L1", "lens", PCX_F50, (0.0, 0.0, 0.0))],
                  [plane("screen", (40.0, 0.0, 0.0), radius_mm=20.0)])
    monitor = load_run(run(doc)).monitor("screen")
    detection = _detection_monitor()
    named = 0
    for descriptor in metric_descriptors():
        for field in descriptor.result_fields:
            named += 1
            if field.startswith("monitor:"):
                path = field[len("monitor:"):]
                if path.startswith("interference[]") or path.startswith("superpositions[]"):
                    continue  # needs two beams on the plane; checked against the contract below
                if path.startswith("polarization_mixtures[]"):
                    continue  # needs a partially polarized source (test_partial_polarization.py resolves it)
                if path.startswith("detection."):
                    # needs a plane whose detection states the electrical response; absent elsewhere
                    assert not _resolve(monitor, path), (descriptor.name, field)
                    assert _resolve(detection, path), (descriptor.name, field)
                    continue
                assert _resolve(monitor, path), (descriptor.name, field)
            else:
                kind, column = field.split(".", 1)
                assert column in {c["name"] for c in columns(kind)}, (descriptor.name, field)
    assert named > 60
    by_name = {d.name: d for d in metric_descriptors()}
    assert by_name["divergence_half_angle_x_rad"].result_fields[0] == "arrivals.far_field_half_angle_x_rad"
    from os_optics.contracts.results import InterferenceResult

    for descriptor in by_name.values():
        for field in descriptor.result_fields:
            if field.startswith("monitor:interference[]."):
                assert field.split(".", 1)[1] in InterferenceResult.model_fields, field
    assert json_api.metric_descriptors()["descriptors"][0]["result_fields"] is not None
    exported = json.loads(load_run(run(doc)).export("metrics", "json"))
    assert "result_fields" in exported["rows"][0]


def test_native_classes_state_availability_and_catalog_equivalents():
    classes = {c["class_id"]: c for c in native_components()}
    forms = {f.value for f in OpticalForm}
    categories = {c.value for c in OpticalCategory}
    for class_id, described in classes.items():
        assert described["availability"] == ["LOCAL_SIMULATION", "CUSTOMER_SUPPLIED"], class_id
        assert "ORDERABLE" not in json.dumps(described["availability"])
        for equivalent in described["catalog_equivalents"]:
            assert equivalent["category"] in categories and set(equivalent["forms"]) <= forms, class_id
            assert set(equivalent["match"]) <= set(SUMMARY_SPECIFICATIONS), class_id
            assert equivalent["query"].startswith(f"os-optics catalog list --category {equivalent['category']} --form ")
    assert classes["lens.achromat"]["catalog_equivalents"][0]["forms"] == ["ACHROMATIC_DOUBLET"]
    assert classes["aperture.circular"]["catalog_equivalents"] == []  # no catalog product of this kind
    assert json_api.native_components()["classes"] == native_components()


def test_a_catalog_equivalent_query_finds_comparable_products():
    from os_optics.local import install_catalog, clear_installed_catalogs

    install_catalog(json.dumps(catalog_document()).encode())
    try:
        equivalent = next(c for c in native_components() if c["class_id"] == "lens.singlet")["catalog_equivalents"][0]
        found = json_api.catalog_products(category=equivalent["category"], form="PLANO_CONVEX_LENS", wavelength_nm=780, summary=True)
        assert found["ok"] and found["total"] >= 2
        assert all(set(equivalent["match"]) & set(row) for row in found["products"])
    finally:
        clear_installed_catalogs()


def test_schema_command_lists_the_linear_optical_network():
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["schema", "linear-optical-network"]) == 0
    schema = json.loads(out.getvalue())
    assert schema == json.loads(json.dumps(json_api.json_schema("linear-optical-network")["schema"], sort_keys=True))
    assert "group_delay_ps" in schema["$defs"]["NetworkPathV1"]["properties"]
    with redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as info:
        cli_main(["schema", "--help"])
    assert info.value.code == 0
