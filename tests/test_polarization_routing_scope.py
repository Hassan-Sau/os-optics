"""Polarization routing marks where its share is not backed by the element's characterized data (review findings N2, N3):
a coating met beyond its own angles or outside its stated ``valid_aoi_deg``, a tabulated thin lens outside its wavelengths
or validated polarizations. The share itself stays the one the run traces with; each noted case is one the run reports
out of model scope, checked here against the run's own issues.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot
from os_optics.contracts.materials import ConstantCoating
from os_optics.local import example as ex
from os_optics.local import json_api, lower_project, run_optical_project
from os_optics.optics.polarization.routing import (
    COATING_AOI_OUTSIDE_RANGE,
    MODEL_POLARIZATION_NOT_VALIDATED,
    MODEL_WAVELENGTH_OUTSIDE_BAND,
    ROUTE_PASS,
    PolarizationRouter,
)

from declared_helpers import declared_component, with_models

POWER_W = ex.SOURCE_POWER_W
X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
INV = 1.0 / math.sqrt(2.0)


def _issue_codes(document) -> set[str]:
    """The run's Engine issues and its model-scope issues (the published / declared data's validity)."""

    run = run_optical_project(document)
    return {i.code.value for i in run.result.issues} | {i.code for i in run.model_scope_issues}


# ------------------------------------------------------------------------------------------------ coating angles


def _fold(valid_max_deg: str) -> dict:
    """A 45 deg fold mirror whose declared coating (R = 0.9; fixture data) is characterized for 0..valid_max_deg."""

    built = json_api.build_native_model("mirror.flat", {"reflectance": "0.9", "valid_band_nm": ["400", "1100"], "diameter_mm": "25.4"})
    assert built["ok"], built
    mirror, (coating,) = built["optical_model"], built["model_coatings"]
    coating = {**coating, "valid_aoi_deg": {"min_deg": "0", "max_deg": valid_max_deg, "status": "SPECIFIED"}}
    document = {"schema": "opensci.project", "project_schema_version": "2.0", "targets": [],
                "components": [ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=780.0, polarization_deg=30.0),
                               declared_component("M", "mirror", mirror["model_id"], quaternion=ex.FOLD_X_TO_Y)],
                "observations": [ex._plane("OUT", [0.0, 100.0, 0.0], ex.BEAM_ALONG_Y)]}
    return with_models(document, [mirror], None, [coating])


@pytest.mark.parametrize("valid_max_deg, outside", [("50", False), ("10", True)])
def test_a_coating_met_outside_its_stated_angles_is_noted_as_the_run_reports_it(valid_max_deg, outside):
    document = _fold(valid_max_deg)
    lowered = lower_project(document, None)
    plain = PolarizationRouter(lowered.request.project)
    checked = PolarizationRouter.for_lowered(lowered)
    passages = [r.step("M", ROUTE_PASS, 780.0, (0.0, 0.0, 0.0), X, Y, r.source_state("SRC")) for r in (plain, checked)]
    assert passages[0].notes == () and passages[0].resolved
    assert passages[1].notes == ((COATING_AOI_OUTSIDE_RANGE,) if outside else ())
    assert passages[1].share == passages[0].share  # the share the run traces with, noted or not
    assert (COATING_AOI_OUTSIDE_RANGE in _issue_codes(document)) is outside


def test_a_low_level_coating_beyond_its_own_angles_is_noted():
    """N2 at Engine-request level: a constant coating stated up to 10 deg, met at 45 deg. The router holds it at 10 deg (as
    before) and now says so."""

    mirror = B.mirror_for_reflection("M", position=(0.0, 0.0, 0.0), d_in=X, d_out=Y, reflectance=0.9)
    surface = mirror.surfaces[0]
    limited = ConstantCoating(reflectance_s=0.9, reflectance_p=0.9, transmittance_s=0.1, transmittance_p=0.1, max_aoi_deg=10.0)
    mirror = mirror.model_copy(update={"surfaces": (surface.model_copy(update={"interaction": surface.interaction.model_copy(
        update={"coating": limited})}),)})
    project = ProjectSnapshot(components=(B.laser("L", wavelength_nm=780.0, position=(-50.0, 0.0, 0.0), direction=X,
                                                  polarization=JonesVector.linear(0.0)), mirror),
                              observations=(B.observation_plane("OUT", position=(0.0, 60.0, 0.0), normal=Y),))
    router = PolarizationRouter(project)
    passage = router.step("M", ROUTE_PASS, 780.0, (0.0, 0.0, 0.0), X, Y, router.source_state("L"))
    assert passage is not None and passage.notes == (COATING_AOI_OUTSIDE_RANGE,) and not passage.resolved


# ------------------------------------------------------------------------------------------------ tabulated thin lens


def _ttl(polarization_deg: float, wavelength_nm: float = 780.0, validated: list[str] | None = None) -> dict:
    evidence = {"method": "fixture response", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "0" * 64,
                "fit_model": "fixture", "validation_summary": "fixture", "date": "2026-10-15"}
    model = {"model_kind": "TABULATED_THIN_LENS", "model_version": "1", "status": "DERIVED", "evidence": evidence,
             "rows": [{"wavelength_nm": "760", "focal_length_mm": "103", "transmittance": "0.85"},
                      {"wavelength_nm": "800", "focal_length_mm": "98", "transmittance": "0.9"}],
             "focal_interpolation": "LINEAR_IN_OPTICAL_POWER", "transmittance_interpolation": "LINEAR", "clear_radius_mm": "10",
             "edge_radius_mm": "12.7", "max_field_angle_deg": "5", "supported_metrics": [{"metric": "transmittance", "abs_error": "0.05", "basis": "fixture"}]}
    if validated is not None:
        model["validated_polarizations"] = validated
    document = {"schema": "opensci.project", "project_schema_version": "2.0", "targets": [],
                "components": [ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=wavelength_nm, polarization_deg=polarization_deg),
                               declared_component("ML", "lens", "ml", (0.0, 0.0, 0.0), ex.BEAM_ALONG_X)],
                "observations": [ex._plane("OUT", [50.0, 0.0, 0.0], ex.BEAM_ALONG_X)]}
    return with_models(document, [{"model_id": "ml", "model": model}])


def _lens_step(document, wavelength_nm: float, *, known: bool = True):
    router = PolarizationRouter(lower_project(document, None).request.project)
    return router.step("ML", ROUTE_PASS, wavelength_nm, (0.0, 0.0, 0.0), X, X, router.source_state("SRC") if known else None)


def test_a_tabulated_thin_lens_outside_its_wavelengths_is_noted():
    inside, outside = _lens_step(_ttl(0.0), 780.0), _lens_step(_ttl(0.0, 850.0), 850.0)
    assert inside.notes == () and outside.notes == (MODEL_WAVELENGTH_OUTSIDE_BAND,)
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in _issue_codes(_ttl(0.0, 850.0))


@pytest.mark.parametrize("polarization_deg, validated", [(0.0, True), (90.0, False), (45.0, False)])
def test_a_tabulated_thin_lens_reached_by_another_polarization_is_noted(polarization_deg, validated):
    """The lens is validated for light linear along its local X; the source's 0 deg polarization is along that axis (both
    are design +Y here), 90 deg is across it, 45 deg is half of each. The run reports the same cases."""

    document = _ttl(polarization_deg, validated=["LINEAR_X"])
    passage = _lens_step(document, 780.0)
    assert passage.notes == (() if validated else (MODEL_POLARIZATION_NOT_VALIDATED,))
    assert ("MODEL_POLARIZATION_NOT_VALIDATED" in _issue_codes(document)) is not validated
    unknown = _lens_step(document, 780.0, known=False)
    assert unknown.notes == (MODEL_POLARIZATION_NOT_VALIDATED,)  # a polarization that is not known is not a validated one


# ------------------------------------------------------------------------------------------------ a plate met back first


PLATE_SKU = "OS-OPT-BS-PLATE-D12P7-R50T50-700-1100-W0P5-AOI45-T3-UVFS-CA11P43"


@pytest.mark.parametrize("flipped", [False, True], ids=["coated-face-first", "back-face-first"])
def test_a_plate_splitter_met_back_face_first_routes_its_transmission(real_catalog_store, flipped):
    """A wedged plate splitter turned so that its back face (tilted by the wedge) meets the line first: the walk starts
    before that face's crossing, so the routed transmission is the traced one (it read 1.0, the back face alone, before).
    Its reflected arm meets the back face twice and is not followed (None: the caller's fallback, marked unresolved)."""

    from os_optics.local import load_public_catalog

    catalog = load_public_catalog()
    plate = catalog.get(PLATE_SKU)
    turn = ex._quat_multiply(ex.FOLD_X_TO_Y, (0.0, 1.0, 0.0, 0.0)) if flipped else ex.FOLD_X_TO_Y
    project = ex._project([ex._source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=1064.0), ex._element("BS", "splitter", plate, catalog, [0.0, 0.0, 0.0], turn)],
                          [ex._plane("T", [60.0, 0.0, 0.0], ex.BEAM_ALONG_X), ex._plane("R", [0.0, 60.0, 0.0], ex.BEAM_ALONG_Y)], []).payload()
    run = run_optical_project(project)
    # This is a real public plate with unstated bulk loss: its lossless routed share is an
    # upper bound, not an invented measured power. Geometry and surface traversal remain exact.
    traced = {}
    for observation in run.result.observations:
        assert observation.observation_id in {"T", "R"}
        for arrival in observation.arrivals:
            direction = arrival.beam.direction
            assert direction[2] == pytest.approx(0.0, abs=1e-12)
            if observation.observation_id == "T":
                # The real plate is wedged: Snell gives a small *nonzero* transmitted deflection.
                assert direction[0] > 0.999 and 0.006 < abs(direction[1]) < 0.008
            else:
                assert direction[1] > 0.999
            if arrival.beam.power_W is None:
                assert observation.quantity_state.value == "UNKNOWN"
                assert arrival.power_upper_bound_W is not None
        traced[observation.observation_id] = sum(
            arrival.beam.power_W if arrival.beam.power_W is not None else arrival.power_upper_bound_W
            for arrival in observation.arrivals) / POWER_W
    router = PolarizationRouter.for_lowered(lower_project(project))
    state = router.source_state("SRC")
    through = router.step("BS", ROUTE_PASS, 1064.0, (0.0, 0.0, 0.0), X, X, state)
    assert through.surfaces == (("BS/S2", "BS/S1") if flipped else ("BS/S1", "BS/S2"))
    assert through.share == pytest.approx(traced["T"], rel=1e-3) and through.resolved
    reflected = router.step("BS", "SPLIT_REFLECT", 1064.0, (0.0, 0.0, 0.0), X, Y, state)
    assert (reflected is None) is flipped
