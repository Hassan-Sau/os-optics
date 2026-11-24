"""Linear optical network export of Projects: the project adapter (``lower_project``, the same optics a run
simulates), ``json_api.linear_optical_network`` envelopes, its JSON Schema and the ``os-optics project network`` command."""

from __future__ import annotations

import json

import numpy as np
import pytest

from os_optics.cli import main as cli_main
from os_optics.contracts.network import LinearOpticalNetworkV1
from os_optics.local import example_project_of_kind, json_api, lower_project, run_optical_project
from os_optics.network import contract_hash_of

from generate_fixtures import REQ_DIR
from local_helpers import catalog_document


def _network_catalog():
    """Synthetic catalog optics for numeric matrix tests explicitly assume zero bulk absorption."""
    fixture = catalog_document()
    fixture["schema_version"] = "3.6"  # Required for material bulk-loss evidence.
    for material in fixture["materials"]:
        material["bulk_loss"] = {"kind": "INTERNAL_TRANSMITTANCE",
                                 "wavelength_nm": [material["valid_band_nm"]["min_nm"], material["valid_band_nm"]["max_nm"]],
                                 "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
                                 "basis": "Assumed lossless synthetic test fixture."}
    return fixture


def _matrix(network: dict) -> np.ndarray:
    return np.array([[complex(*z) for z in row] for row in network["transfer_matrix"]["entries"]])


def test_a_project_exports_the_optics_its_run_simulates():
    project = example_project_of_kind("two_beam_interference")
    reply = json_api.linear_optical_network(project.payload(), catalog=_network_catalog())
    assert reply["ok"] is True
    network = reply["network"]
    LinearOpticalNetworkV1.model_validate(network)
    assert reply["contract_hash"] == network["contract_hash"] == contract_hash_of(network)
    provenance = network["provenance"]
    lowered = lower_project(project, catalog=_network_catalog())
    assert provenance["project_hash"] == project.content_hash()
    assert provenance["engine_input_hash"] == lowered.engine_input_hash
    assert provenance["catalog_version"] == lowered.catalog_version
    assert provenance["catalog_sha256"] is None  # The test-only catalog is supplied explicitly, not a stored pinned release.
    # the two arms cross at 90 deg on the plane: two separate output beams, each with the arm's share of the light
    assert network["transfer_matrix"]["rows"] == ["out:crossing_plane:0:H", "out:crossing_plane:0:V", "out:crossing_plane:1:H",
                                                  "out:crossing_plane:1:V"]
    run = run_optical_project(project, catalog=_network_catalog())
    arrivals = [a for o in run.result.observations if o.observation_id == "crossing_plane" for a in o.arrivals]
    source = run.result.power_budget.lines[0].input_power_W
    t = _matrix(network)
    # the source is s-polarized (V): the power fraction of each beam equals the run's beam power / source power
    beam_power = sorted(float(np.sum(np.abs(t[2 * k:2 * k + 2, 1]) ** 2)) for k in range(2))
    assert beam_power == pytest.approx(sorted(a.beam.power_W / source for a in arrivals), abs=1e-8)


def test_mixed_wavelength_projects_export_one_line_at_a_time():
    document = example_project_of_kind("dichroic_routing").payload()
    refused = json_api.linear_optical_network(document)
    assert refused["ok"] is False and refused["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    assert [p["code"] for p in refused["error"]["problems"]] == ["MIXED_WAVELENGTHS"]
    wavelengths = sorted({c["source_model"]["wavelength_nm"] for c in document["components"] if c.get("source_model")})
    for wavelength in wavelengths:
        reply = json_api.linear_optical_network(document, {"wavelength_nm": float(wavelength)}, catalog=_network_catalog())
        assert reply["ok"] is True and reply["network"]["wavelength_nm"] == float(wavelength)
        assert len(reply["network"]["input_modes"]) == 2  # only the source of that line


def test_a_project_outside_a_published_model_is_refused():
    """A wavelength 1 nm beyond the lens model's band: the run would be OUT_OF_MODEL_SCOPE, so the network is refused."""
    from decimal import Decimal

    from os_optics.local import load_public_catalog

    document = example_project_of_kind("focus").payload()
    lens = load_public_catalog().get(document["components"][1]["product"]["optical_sku"])
    document["components"][0]["source_model"]["wavelength_nm"] = float(Decimal(lens.simulation_models[0].valid_band_nm.max_nm) + 1)
    reply = json_api.linear_optical_network(document)
    assert reply["ok"] is False and reply["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    problems = reply["error"]["problems"]
    assert {p["code"] for p in problems} == {"OUT_OF_MODEL_SCOPE"}
    assert any(p["message"].startswith("MODEL_WAVELENGTH_OUTSIDE_BAND") and p["location"] == "component L1" for p in problems)


def test_envelopes_for_invalid_input_options_and_engine_requests():
    invalid = json_api.linear_optical_network({"schema": "opensci.project"})
    assert invalid["ok"] is False and invalid["error"]["code"] == "LOCAL_RUN_ERROR"
    document = example_project_of_kind("focus").payload()
    bad = json_api.linear_optical_network(document, {"mode_overlap_tolerance": 2.0})
    assert bad["error"]["code"] == "NETWORK_NOT_EXPORTABLE" and bad["error"]["problems"][0]["code"] == "OPTIONS_INVALID"
    request = json.loads((REQ_DIR / "hwp_pbs.json").read_text(encoding="utf-8"))
    reply = json_api.linear_optical_network(request)
    assert reply["ok"] is True and reply["network"]["provenance"]["project_hash"] is None
    assert json_api.linear_optical_network(json.dumps(request))["contract_hash"] == reply["contract_hash"]  # JSON text too


def test_capabilities_and_json_schema_name_the_contract():
    assert json_api.capabilities()["linear_optical_network"] == "opensci-linear-optical-network/v1"
    schema = json_api.json_schema("linear-optical-network")
    assert schema["ok"] is True and schema["schema"]["title"] == "LinearOpticalNetworkV1"
    assert "schema" in schema["schema"]["properties"] and "contract_hash" in schema["schema"]["required"]


def test_cli_project_network(tmp_path, capsys):
    fixture = tmp_path / "synthetic-catalog.json"
    fixture.write_text(json.dumps(_network_catalog()), encoding="utf-8")
    path = tmp_path / "project.json"
    path.write_text(json.dumps(example_project_of_kind("polarization").payload()), encoding="utf-8")
    out = tmp_path / "network.json"
    assert cli_main(["project", "network", str(path), "--catalog", str(fixture), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert summary["status"] == "EXPORTED" and summary["contract_hash"] == written["contract_hash"] == contract_hash_of(written)
    assert summary["input_modes"] == 2 and summary["output_modes"] == 2
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps(example_project_of_kind("dichroic_routing").payload()), encoding="utf-8")
    assert cli_main(["project", "network", str(mixed), "--catalog", str(fixture)]) == 65
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "ERROR" and refused["error"]["problems"][0]["code"] == "MIXED_WAVELENGTHS"
    assert cli_main(["project", "network", str(tmp_path / "missing.json")]) == 64
    capsys.readouterr()
    request = REQ_DIR / "mzi_window.json"
    assert cli_main(["project", "network", str(request), "--observation", "port_x"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["transfer_matrix"]["rows"] == ["out:port_x:0:H", "out:port_x:0:V"]


def test_a_project_without_catalog_components_needs_and_names_no_catalog(tmp_path, monkeypatch):
    """A Project of declared optics only exports with no catalog on the device, and its provenance names none -- as its
    run does (no catalog is loaded, fetched or bound)."""

    monkeypatch.setenv("OPENSCI_CATALOG_DIR", str(tmp_path / "empty-store"))
    monkeypatch.setenv("OPENSCI_CATALOG_CHECK_INTERVAL", "0")
    window = json_api.build_native_model("plate.window", {"model_id": "w", "material": {"constant_index": "1.5", "valid_band_nm": ["600", "700"], "absorption_coefficient_per_mm": "0"},
                                                          "thickness_mm": "1", "diameter_mm": "25.4"})
    along_x = [0.5, 0.5, 0.5, 0.5]
    project = {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
               "components": [{"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "l", "description": "Laser"},
                               "nominal_pose": {"position_mm": [-50, 0, 0], "quaternion_wxyz": along_x},
                               "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 1.0,
                                                "waist_position_mm": 0, "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}}},
                              {**window["component_template"], "instance_id": "W", "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": along_x}}],
               "observations": [{"observation_id": "OUT", "pose": {"position_mm": [50, 0, 0], "quaternion_wxyz": along_x},
                                 "aperture": {"shape": "circle", "radius_mm": 10}, "absorbing": True, "medium": "air"}],
               "optical_models": [window["optical_model"]], "model_materials": window["model_materials"]}
    run = json_api.run_project(project)
    assert run["ok"] and run["summary"]["catalog_version"] is None
    exported = json_api.linear_optical_network(project)
    assert exported["ok"], exported
    provenance = exported["network"]["provenance"]
    assert provenance.get("catalog_version") is None and provenance.get("catalog_sha256") is None


def retarder_two_source_project(*sources: tuple[str, float, float]) -> dict:
    """A native half-wave retarder characterized for 600-700 nm on the axis, a Gaussian source per ``(id, wavelength_nm,
    y_mm)`` 50 mm before it (all through the plate), and one absorbing output plane behind it."""
    plate = json_api.build_native_model("waveplate.retarder", {"retardance_waves": "0.5", "design_wavelength_nm": "633", "diameter_mm": "25.4",
                                                                "valid_band_nm": ["600", "700"]})
    assert plate["ok"], plate
    along_x = [0.5, 0.5, 0.5, 0.5]
    components = [{"instance_id": sid, "optical_role": "source", "custom_part": {"custom_part_key": sid.lower(), "description": "Laser"},
                   "nominal_pose": {"position_mm": [-50, y, 0], "quaternion_wxyz": along_x},
                   "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": wavelength, "power_w": 0.001, "waist_radius_mm": 1.0,
                                    "waist_position_mm": 0, "m2": 1, "polarization": {"kind": "LINEAR", "angle_deg": 0}}}
                  for sid, wavelength, y in sources]
    components.append({**plate["component_template"], "instance_id": "WP", "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": along_x}})
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": components,
            "observations": [{"observation_id": "OUT", "pose": {"position_mm": [50, 0, 0], "quaternion_wxyz": along_x},
                              "aperture": {"shape": "circle", "radius_mm": 20}, "absorbing": True, "medium": "air"}],
            "optical_models": [plate["optical_model"]]}


def _scope_problems(reply: dict) -> list[tuple[str, str, str]]:
    assert reply["ok"] is False and reply["error"]["code"] == "NETWORK_NOT_EXPORTABLE", reply
    return [(p["code"], p["message"].split(":")[0], p.get("location")) for p in reply["error"]["problems"]]


def test_a_source_the_export_does_not_select_cannot_refuse_it():
    """A native retarder valid for 600-700 nm; source GOOD at 633 nm and OTHER at 800 nm. Exporting only GOOD at 633 nm
    describes light inside the retarder's band: it exports whether or not OTHER is in the Project (the published-model check
    used to run the whole Project, so OTHER's 800 nm refused it). OTHER itself (selected, or the 800 nm line) is still
    refused with the retarder's band violation -- the same as a run of the Project, which is OUT_OF_MODEL_SCOPE."""
    selection = {"source_ids": ["GOOD"], "wavelength_nm": 633}
    alone = json_api.linear_optical_network(retarder_two_source_project(("GOOD", 633, 0)), selection)
    assert alone["ok"] is True, alone
    both_project = retarder_two_source_project(("GOOD", 633, 0), ("OTHER", 800, 3))
    both = json_api.linear_optical_network(both_project, selection)
    assert both["ok"] is True, both
    assert [m["source_id"] for m in both["network"]["input_modes"]] == ["GOOD", "GOOD"]
    assert both["network"]["transfer_matrix"]["entries"] == alone["network"]["transfer_matrix"]["entries"]
    # the line at 633 nm selects GOOD alone too (OTHER has no line there: no input mode, no check)
    assert json_api.linear_optical_network(both_project, {"wavelength_nm": 633})["ok"] is True
    out_of_band = ("OUT_OF_MODEL_SCOPE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "component WP")
    assert _scope_problems(json_api.linear_optical_network(both_project, {"source_ids": ["OTHER"], "wavelength_nm": 800})) == [out_of_band]
    assert _scope_problems(json_api.linear_optical_network(both_project, {"wavelength_nm": 800})) == [out_of_band]
    # the selection is resolved first: an invalid one is refused as such, not by the light it does not export
    assert [p[0] for p in _scope_problems(json_api.linear_optical_network(both_project))] == ["MIXED_WAVELENGTHS"]
    assert [p[0] for p in _scope_problems(json_api.linear_optical_network(both_project, {"source_ids": ["NOPE"]}))] == ["OPTIONS_INVALID"]
    run = json_api.run_project(both_project)
    assert run["ok"] and run["status"] == "OUT_OF_MODEL_SCOPE"


def test_a_published_model_violation_refuses_only_an_export_whose_outputs_receive_that_light():
    """Source GOOD (633 nm) through a retarder characterized for 600-700 nm onto OUT; source SIDE (633 nm) 40 mm aside
    through a retarder characterized for 700-800 nm onto its own plane SIDE_OUT. Exporting OUT alone describes light inside
    every band it crosses: exported with the violation listed; with SIDE_OUT the export is refused naming that plane."""
    document = retarder_two_source_project(("GOOD", 633, 0), ("SIDE", 633, 40))
    side = json_api.build_native_model("waveplate.retarder", {"retardance_waves": "0.5", "design_wavelength_nm": "750", "diameter_mm": "25.4",
                                                              "valid_band_nm": ["700", "800"], "model_id": "side_plate"})
    assert side["ok"], side
    along_x = [0.5, 0.5, 0.5, 0.5]
    document["components"].append({**side["component_template"], "instance_id": "WP2",
                                   "nominal_pose": {"position_mm": [0, 40, 0], "quaternion_wxyz": along_x}})
    document["optical_models"].append(side["optical_model"])
    document["observations"][0]["aperture"]["radius_mm"] = 10
    document["observations"].append({"observation_id": "SIDE_OUT", "pose": {"position_mm": [50, 40, 0], "quaternion_wxyz": along_x},
                                     "aperture": {"shape": "circle", "radius_mm": 10}, "absorbing": True, "medium": "air"})
    refused = json_api.linear_optical_network(document)
    assert _scope_problems(refused) == [("OUT_OF_MODEL_SCOPE", "MODEL_WAVELENGTH_OUTSIDE_BAND", "component WP2")]
    assert refused["error"]["problems"][0]["observation_id"] == "SIDE_OUT"
    exported = json_api.linear_optical_network(document, {"observation_ids": ["OUT"]})
    assert exported["ok"] is True, exported
    assert [(w["code"], w["location"]) for w in exported["network"]["validity"]["warnings"]] == [
        ("MODEL_WAVELENGTH_OUTSIDE_BAND", "component WP2")]
    # the matrix is that of the layout without the side retarder (SIDE's column stays empty on OUT either way)
    without = retarder_two_source_project(("GOOD", 633, 0), ("SIDE", 633, 40))
    without["observations"][0]["aperture"]["radius_mm"] = 10
    reference = json_api.linear_optical_network(without, {"observation_ids": ["OUT"]})
    assert exported["network"]["transfer_matrix"] == reference["network"]["transfer_matrix"]


def test_the_cli_exports_a_selection_beside_an_out_of_band_source(tmp_path, capsys):
    path = tmp_path / "project.json"
    path.write_text(json.dumps(retarder_two_source_project(("GOOD", 633, 0), ("OTHER", 800, 3))), encoding="utf-8")
    assert cli_main(["project", "network", str(path), "--source", "GOOD", "--wavelength-nm", "633", "--out", str(tmp_path / "n.json")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "EXPORTED"
    assert cli_main(["project", "network", str(path), "--source", "OTHER", "--wavelength-nm", "800"]) == 65
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "ERROR" and [p["code"] for p in refused["error"]["problems"]] == ["OUT_OF_MODEL_SCOPE"]
