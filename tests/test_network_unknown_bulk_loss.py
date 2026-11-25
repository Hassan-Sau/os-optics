"""The network v1 numeric transfer matrix cannot stand for an upper-envelope field.

An unstated bulk absorption leaves phase and path geometry available, but not a
single measured amplitude or transmission. Only a test-only explicit zero-loss
assumption permits the same window to have a numeric network.
"""

from __future__ import annotations

import pytest

from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target, ValidationRequest
from os_optics.contracts.materials import ConstantCoating, ConstantIndexMaterial, SellmeierMaterial
from os_optics.local import example_project_of_kind, json_api
from os_optics.local.network import export_request_network
from os_optics.network import NetworkErrorCode, NetworkExportError
from os_optics.optics.materials.library import BUILTIN_MATERIALS
from os_optics.validation.pipeline import validate


def _window_project(material: str, *materials: ConstantIndexMaterial | SellmeierMaterial) -> ProjectSnapshot:
    # Ideal coated faces isolate bulk absorption from untraced Fresnel returns.
    window = B.window("W", thickness_mm=2, clear_radius_mm=12.7, material=material,
                      position=(0, 0, 20))
    ar = ConstantCoating(reflectance_s=0, reflectance_p=0, transmittance_s=1, transmittance_p=1)
    window = window.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(
        update={"coating": ar})}) for s in window.surfaces)})
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=632.8, waist_radius_mm=0.5, position=(-50, 0, 20),
                            power_W=0.001), window),
        observations=(B.observation_plane("out", position=(50, 0, 20), normal=(1, 0, 0),
                                          radius_mm=10).model_copy(update={"absorbing": True}),),
        materials=materials,
        targets=(Target(target_id="minimum", metric="power_W", observation_id="out", minimum=.0009, unit="W"),
                 Target(target_id="maximum", metric="power_W", observation_id="out", maximum=.0009, unit="W")),
    )


def test_unknown_synthetic_bulk_loss_refuses_a_numeric_matrix_but_preserves_geometry():
    unstated = ConstantIndexMaterial(material_id="test-glass", valid_wavelength_nm=(500, 800), refractive_index=1.5)
    assumed = ConstantIndexMaterial(material_id="test-glass", valid_wavelength_nm=(500, 800), refractive_index=1.5,
                                    absorption_coefficient_per_mm=0.0)
    assert "absorption_coefficient_per_mm" not in unstated.model_fields_set
    assert "absorption_coefficient_per_mm" in assumed.model_fields_set
    unknown_request = ValidationRequest(project=_window_project("test-glass", unstated))
    assumed_request = ValidationRequest(project=_window_project("test-glass", assumed))

    uncertain_result = validate(unknown_request)
    assumed_result = validate(assumed_request)
    unknown, known = uncertain_result.observations[0], assumed_result.observations[0]
    assert unknown.total_power_W is None
    assert unknown.power_upper_bound_W == pytest.approx(known.total_power_W)
    assert unknown.quantity_state.value == "BOUNDED"
    assert [t.status.value for t in uncertain_result.targets] == ["UNKNOWN", "UNKNOWN"]
    assert all(t.value is None for t in uncertain_result.targets)
    assert [t.status.value for t in assumed_result.targets] == ["PASS", "FAIL"]
    assert unknown.arrivals[0].beam.geometric_length_mm == pytest.approx(known.arrivals[0].beam.geometric_length_mm)
    assert unknown.arrivals[0].beam.optical_path_length_mm == pytest.approx(known.arrivals[0].beam.optical_path_length_mm)

    with pytest.raises(NetworkExportError) as exc:
        export_request_network(unknown_request)
    assert len(exc.value.problems) == 1
    problem = exc.value.problems[0]
    assert problem.code is NetworkErrorCode.OUT_OF_MODEL_SCOPE
    assert problem.location == "material:test-glass"
    assert "BULK_LOSS_NOT_ESTABLISHED" in problem.message
    response = json_api.linear_optical_network(unknown_request.model_dump(mode="json"))
    assert response["ok"] is False
    assert response["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    assert response["error"]["problems"] == [problem.as_dict()]
    assert "network" not in response  # a hypothetical zero-loss upper envelope is not a transfer matrix

    network = export_request_network(assumed_request)
    assert network.classical_check.status == "PASS"
    assert network.transfer_matrix.column_transmission[0] == pytest.approx(known.total_power_W / .001, abs=1e-8)
    assert json_api.linear_optical_network(assumed_request.model_dump(mode="json"))["ok"] is True


def test_catalog_n_bk7_without_bulk_loss_cannot_be_exported_as_known_transmission():
    # The historical *built-in* N-BK7 is an ideal-lossless tutorial medium; a
    # catalog copy (separate material id) must not inherit that assumption.
    glass = next(m for m in BUILTIN_MATERIALS if m.material_id == "N-BK7").model_copy(
        update={"material_id": "catalog:N-BK7"})
    assert "absorption_coefficient_per_mm" not in glass.model_fields_set
    request = ValidationRequest(project=_window_project("catalog:N-BK7", glass))
    observed = validate(request).observations[0]
    assert observed.total_power_W is None
    assert observed.power_upper_bound_W is not None and observed.power_upper_bound_W > 0
    response = json_api.linear_optical_network(request.model_dump(mode="json"))
    assert response["ok"] is False and "network" not in response
    assert response["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    assert {p["code"] for p in response["error"]["problems"]} == {"OUT_OF_MODEL_SCOPE"}
    assert any(p["location"] == "material:catalog:N-BK7" and "BULK_LOSS_NOT_ESTABLISHED" in p["message"]
               for p in response["error"]["problems"])


def test_real_catalog_n_bk7_project_returns_structured_refusal_not_pyodide_type_error():
    # This goes through the Project/catalog lowering path of the browser JSON API,
    # not the Engine's historically ideal-lossless built-in N-BK7 tutorial medium.
    document = example_project_of_kind("focus").payload()
    response = json_api.linear_optical_network(document)
    assert response["ok"] is False and "network" not in response
    assert response["error"]["code"] == "NETWORK_NOT_EXPORTABLE"
    assert len(response["error"]["problems"]) == 1  # no partial numeric matrix
    (problem,) = response["error"]["problems"]
    assert problem["code"] == "OUT_OF_MODEL_SCOPE" and problem["location"] == "material:catalog:N-BK7"
    assert "BULK_LOSS_NOT_ESTABLISHED" in problem["message"] and "780" in problem["message"]
