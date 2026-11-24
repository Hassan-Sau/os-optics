"""An imaging field reads the light of one source's spectral line at the plane: where an element's published model does
not cover that light (``local/validity.py``), every number of the field is the model's number outside its validity and
says so -- state ``OUT_OF_MODEL_SCOPE``, the number kept, a ``value_basis`` of kind ``ESTIMATE`` whose validity check names
the element, the fact out of range and the published limits. The numbers themselves do not move: the band is the
model's characterization, not its physics, so the same design with a band that covers the light is the reference.

Another field on the same plane whose light the model covers keeps its quantities as they are.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from os_optics.errors.status import QuantityState, ValueKind
from os_optics.local import EvaluationSession
from os_optics.results import load_run

from local_helpers import run
from test_imaging import _hyperbolic, _lens_project

HERE = Path(__file__).resolve().parent
OUTSIDE = "MODEL_WAVELENGTH_OUTSIDE_BAND"
SETTINGS = {"psf_samples": 64}
DESIGN_NM = 587.5618


def _singlet(band: tuple[str, str] | None = None) -> dict:
    """The F/8 plano-hyperbolic singlet of ``test_imaging`` (its member published for 300-2500 nm), with ``band`` instead."""

    document = _lens_project(_hyperbolic(8.0, 40.0, DESIGN_NM), waist_mm=1.0)
    if band is not None:
        member = next(m for m in document["optical_models"] if m["model_id"] == "lens-L1")
        member["model"]["valid_band_nm"] = {"min_nm": band[0], "max_nm": band[1], "status": "SPECIFIED"}
    return document


def _valued(quantity) -> bool:
    return quantity.value is not None or quantity.values_ref is not None


def _outside_check(quantity):
    checks = [c for c in quantity.value_basis.validity if c.code == OUTSIDE]
    assert len(checks) == 1, quantity.value_basis
    return checks[0]


def test_every_number_of_a_field_outside_the_band_says_so_and_keeps_its_value():
    reference = load_run(run(_singlet())).imaging("image", SETTINGS)
    scoped_results = load_run(run(_singlet(("600", "700"))))
    assert {i["code"] for i in scoped_results.targets()["model_scope_issues"]} == {OUTSIDE}
    scoped = scoped_results.imaging("image", SETTINGS)
    assert scoped.execution.complete and scoped.execution.outcome.value == "COMPLETED"
    assert scoped.arrays == reference.arrays  # (the band is metadata: the curves are the model's, unchanged)
    valued = 0
    for before, after in zip(reference.quantities, scoped.quantities, strict=True):
        assert after.quantity_id == before.quantity_id
        if not _valued(before):
            assert after == before
            continue
        valued += 1
        assert (after.value, after.values_ref, after.convergence, after.unit) == (before.value, before.values_ref, before.convergence, before.unit)
        assert after.state is QuantityState.OUT_OF_MODEL_SCOPE, after.quantity_id
        assert after.reason == (before.reason if before.state is QuantityState.OUT_OF_MODEL_SCOPE else OUTSIDE)
        assert after.value_basis.kind is ValueKind.ESTIMATE
        check = _outside_check(after)
        assert (check.parameter, check.unit, check.minimum, check.maximum, check.within) == ("wavelength_nm", "nm", 600.0, 700.0, False)
        assert check.observed == pytest.approx(DESIGN_NM, abs=1e-9)
        assert check.at and check.at[0].startswith("LENS")
        # (what qualified the number before still does)
        for kept in (before.value_basis.validity if before.value_basis is not None else ()):
            assert kept in after.value_basis.validity
        assert not after.decides(0.0) if before.value is not None else True
    # (the geometric, wavefront and diffraction quantities of the field, the geometric MTF out of its regime already)
    assert valued == 14


def test_a_field_inside_the_band_on_the_same_plane_keeps_its_quantities():
    def two_lines(band):
        document = _singlet(band)
        red = copy.deepcopy(document["components"][0])
        red["instance_id"] = "RED"
        red["source_model"]["wavelength_nm"] = 650.0
        document["components"].append(red)
        return document

    reference = load_run(run(two_lines(None))).imaging("image", SETTINGS)
    scoped = load_run(run(two_lines(("550", "620")))).imaging("image", SETTINGS)  # 587.56 nm inside, 650 nm outside
    before = {q.quantity_id: q for q in reference.quantities}
    marked = {q.quantity_id for q in scoped.quantities if q.state is QuantityState.OUT_OF_MODEL_SCOPE
              and q.value_basis is not None and any(c.code == OUTSIDE for c in q.value_basis.validity)}
    assert marked and all(qid.startswith("RED#") for qid in marked)
    for q in scoped.quantities:
        if q.quantity_id.startswith("SRC#"):
            assert q == before[q.quantity_id]
        elif _valued(q):
            assert q.quantity_id in marked and _outside_check(q).observed == pytest.approx(650.0, abs=1e-9)


def test_the_out_of_band_thin_lens_spot_is_marked_in_an_evaluation_and_in_its_run():
    document = json.loads((HERE / "fixtures" / "scope" / "thin_lens_band.project.json").read_text(encoding="utf-8"))
    document["components"][0]["source_model"]["wavelength_nm"] = 1200  # the lens is published for 400-1100 nm
    evaluated = EvaluationSession().evaluate(document, outputs=("readings",)).results().imaging("image", {"launch_grid": 32})
    spot = evaluated.quantity("SRC#0.geometric_spot_rms_radius")
    assert spot.state is QuantityState.OUT_OF_MODEL_SCOPE and spot.reason == OUTSIDE and spot.value is not None
    check = _outside_check(spot)
    assert (check.minimum, check.maximum, check.observed, check.within, check.at) == (400.0, 1100.0, 1200.0, False, ("LENS",))
    complete = load_run(run(document)).imaging("image", {"launch_grid": 32})
    assert complete.quantity("SRC#0.geometric_spot_rms_radius") == spot
    document["components"][0]["source_model"]["wavelength_nm"] = 1000  # inside the band: no finding qualifies the field
    inside = EvaluationSession().evaluate(document, outputs=("readings",)).results().imaging("image", {"launch_grid": 32})
    kept = inside.quantity("SRC#0.geometric_spot_rms_radius")  # (the stand-in pupil of an unbounded system qualifies it)
    assert kept.state is QuantityState.UNKNOWN and not any(c.code == OUTSIDE for c in kept.value_basis.validity)


def test_a_coating_outside_its_band_qualifies_what_reads_power_and_leaves_the_geometry():
    from local_helpers import catalog_document

    coating = copy.deepcopy(next(c for c in catalog_document()["coatings"] if c["kind"] == "ANTI_REFLECTION"))
    coating["coating_id"] = "SCOPE-AR"
    low, high = copy.deepcopy(coating["bands"][0]), copy.deepcopy(coating["bands"][0])
    low["max_nm"], high["min_nm"] = "580", "600"  # the design line, 587.56 nm, falls between the coating's two bands
    coating["bands"] = [low, high]
    document = _singlet(("400", "1100"))  # (the element's own band covers the line and lies within the coating's)
    document["model_coatings"] = [coating]
    member = next(m for m in document["optical_models"] if m["model_id"] == "lens-L1")
    member["model"]["surfaces"][0]["coating_id"] = "SCOPE-AR"
    results = load_run(run(document))
    assert {i["code"] for i in results.targets()["model_scope_issues"]} == {"COATING_WAVELENGTH_OUTSIDE_BANDS"}
    scoped = results.imaging("image", SETTINGS)
    geometry = {"image_space_na", "geometric_spot_rms_radius", "opd_rms_waves", "opd_pv_waves"}
    for q in scoped.quantities:
        name = q.quantity_id.rsplit(".", 1)[-1]
        marked = q.value_basis is not None and any(c.code == "COATING_WAVELENGTH_OUTSIDE_BANDS" for c in q.value_basis.validity)
        if name in geometry:
            assert q.state is QuantityState.KNOWN and not marked, name  # (no coating coefficient moves a ray)
        elif _valued(q):
            assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and marked, name
