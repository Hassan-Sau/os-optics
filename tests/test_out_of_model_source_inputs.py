"""Inputs outside the Engine's static model are refused with the boundary and the route: time-domain dynamics of a Project
(vibration, timing noise, time series, feedback) and a continuous spectral density on an extended emitter. A sampled source
written under another name (COMPLEX_FIELD, ARBITRARY_INTENSITY, SPECKLE_FIELD) is answered with the kind's name and fields;
the kinds themselves (SAMPLED_FIELD, SAMPLED_INTENSITY, SPECKLE) are traced (tests/test_sampled_sources.py)."""

from __future__ import annotations

import copy

import pytest

from os_optics.local import run_optical_project
from os_optics.local.catalog import SOURCE_MODEL_REFUSALS, TIME_DOMAIN_FIELDS
from os_optics.local.errors import LocalRunError
from os_optics.interop import entry_ids

from local_helpers import plane, project, source

BASE = project([source()], [plane("OUT", (100.0, 0.0, 0.0))])


def _refusal(document: dict) -> dict:
    with pytest.raises(LocalRunError) as caught:
        run_optical_project(document, catalog=None)
    (problem,) = caught.value.problems
    return problem.as_dict()


@pytest.mark.parametrize("field", TIME_DOMAIN_FIELDS)
def test_time_domain_inputs_name_the_static_model(field):
    document = copy.deepcopy(BASE)
    document[field] = {"amplitude_nm": "10"}
    problem = _refusal(document)
    assert problem["code"] == "PROJECT_INVALID"
    assert problem["message"].startswith(f"{field}: time-domain dynamics")
    assert "static approximation" in problem["message"] and "unknown field" not in problem["message"]


@pytest.mark.parametrize(("kind", "name"), [("COMPLEX_FIELD", "SAMPLED_FIELD"), ("ARBITRARY_INTENSITY", "SAMPLED_INTENSITY"),
                                            ("SPECKLE_FIELD", "SPECKLE")])
def test_other_names_of_the_sampled_kinds_name_the_kind(kind, name):
    assert set(SOURCE_MODEL_REFUSALS) == {"COMPLEX_FIELD", "ARBITRARY_INTENSITY", "SPECKLE_FIELD"}
    document = copy.deepcopy(BASE)
    document["components"][0]["source_model"] = {"model_kind": kind, "wavelength_nm": 633.0}
    problem = _refusal(document)
    assert problem["code"] == "PROJECT_INVALID"
    assert f"is source_model kind {name}:" in problem["message"] and "outside the Engine's model" not in problem["message"]


@pytest.mark.parametrize("kind", ["SAMPLED_FIELD", "SAMPLED_INTENSITY", "SPECKLE"])
def test_the_sampled_kinds_are_read_as_their_models(kind):
    """An incomplete sampled source is refused for its own missing fields, not as an unknown or out-of-model kind."""
    document = copy.deepcopy(BASE)
    document["components"][0]["source_model"] = {"model_kind": kind, "wavelength_nm": 633.0}
    message = _refusal(document)["message"]
    assert "components[0].source_model." in message and "required field missing" in message
    assert "expected tags" not in message and "outside the Engine's model" not in message


def test_the_named_routes_exist():
    assert {"scalar-diffraction-imaging", "scattering-media", "multiphysics-stray-light"} <= set(entry_ids())


def test_another_unknown_kind_still_reads_as_an_unknown_tag():
    document = copy.deepcopy(BASE)
    document["components"][0]["source_model"] = {"model_kind": "LAMP", "wavelength_nm": 633.0}
    assert "does not match any of the expected tags" in _refusal(document)["message"]


def test_a_continuous_spectral_density_on_an_extended_emitter_names_its_discrete_form():
    """(A Gaussian-beam source states a continuous spectrum: tests/test_continuous_spectrum.py.)"""
    document = copy.deepcopy(BASE)
    document["components"][0]["source_model"] = {
        "model_kind": "EXTENDED_EMITTER", "wavelength_nm": 600.0, "power_w": 0.001, "angular_law": "LAMBERTIAN",
        "emitting_surface": {"shape": "DISC", "diameter_mm": 2.0}, "sampling": {"sample_count": 64},
        "spectral_density": {"shape": {"kind": "TOP_HAT", "center_nm": 600.0, "fwhm_nm": 200.0}, "line_count": 8}}
    message = _refusal(document)["message"]
    assert "source_model" in message and "spectral_density: an extended emitter's spectrum is spectral_lines" in message
    assert "GAUSSIAN_BEAM source states a continuous spectrum as spectral_density" in message
