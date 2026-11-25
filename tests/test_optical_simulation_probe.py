"""The local-simulation coverage probe (``scripts/optical_simulation_probe.py``) builds valid one-element designs.

The probe's counts feed the "locally simulatable" stage of the v0.14 coverage report, so a probe whose Project the
local runtime refuses would silently report zero coverage. One product of every (category, form) of the repository's
seed catalog release is probed here with the probe's own code, and the checkpoint / resume file is exercised on a small
catalog slice. The committed probe file is release QA data of that catalog release (not bound to an Engine artifact).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from os_optics.local import load_public_catalog, run_optical_project

from catalog_store_helpers import REAL_CATALOG

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "optical_simulation_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("optical_simulation_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()
CATALOG = load_public_catalog(REAL_CATALOG)
ONE_PER_FORM = {}
for _product in CATALOG.products:
    ONE_PER_FORM.setdefault((_product.category.value, _product.form.value), _product)


@pytest.mark.parametrize("key", sorted(ONE_PER_FORM), ids=lambda k: f"{k[0]}-{k[1]}")
def test_the_probe_design_of_every_form_runs_in_scope(key):
    product = ONE_PER_FORM[key]
    wavelengths = PROBE._wavelengths(product, CATALOG)
    expected = {"status": "IN_SCOPE", "run_status": "PASS", "wavelengths": len(wavelengths)}
    # This seed release does not publish bulk loss for the transmitting glasses in these forms.
    # Their rays still exercise the design, but power is bounded, never asserted to be delivered.
    if key[0] in {"BEAMSPLITTER", "LENS", "PRISM", "WINDOW"}:
        expected.update(run_status="UNKNOWN", power_evidence="UNKNOWN_BULK_LOSS", codes=["BULK_LOSS_NOT_ESTABLISHED"])
    if key[1] == "AXICON":  # a Gaussian probe beam behind a cone is outside the Gaussian model by definition
        expected.update(run_status="OUT_OF_MODEL_SCOPE", documented_scope=[PROBE.AXICON_GAUSSIAN_SCOPE])
    assert PROBE.probe_one(product, CATALOG) == expected


def test_a_seed_lens_with_unstated_bulk_loss_has_a_ray_but_no_known_power():
    product = ONE_PER_FORM[("LENS", "PLANO_CONVEX_LENS")]
    run = run_optical_project(PROBE.probe_project(product, CATALOG), catalog=CATALOG)
    (observation,) = run.result.observations
    assert observation.arrivals and observation.total_power_W is None
    assert observation.power_lower_bound_W is None  # No separately established positive lower bound.
    assert observation.power_upper_bound_W is None  # Transparent plane may be recrossed: no finite total bound.
    for arrival in observation.arrivals:
        assert arrival.beam.power_W is None and arrival.pupil_rays_arrived > 0
        assert arrival.power_upper_bound_W is not None and 0 < arrival.power_upper_bound_W <= 1e-3
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in run.result.issues)


AXICON = next(p for p in CATALOG.products if p.form.value == "AXICON")


def test_an_axicon_counts_in_scope_only_through_its_documented_gaussian_scope():
    """The axicon's run is OUT_OF_MODEL_SCOPE for the Gaussian beam behind its cone and for nothing else; its rays met
    the element and reached the observation plane geometrically; lacking catalog bulk-loss data, its
    power remains bounded rather than being called measured or zero."""

    run = run_optical_project(PROBE.probe_project(AXICON, CATALOG), catalog=CATALOG)
    failing = {(i.code.value, i.status.value) for i in run.result.issues if i.status.value not in ("PASS", "WARNING", "NOT_APPLICABLE")}
    assert failing == {(PROBE.AXICON_GAUSSIAN_SCOPE, "OUT_OF_MODEL_SCOPE")}
    assert run.result.observations[0].arrivals
    assert all(a.beam.power_W is None and a.power_upper_bound_W is not None and 0 < a.power_upper_bound_W <= 1e-3
               and a.pupil_rays_arrived > 0 for o in run.result.observations for a in o.arrivals)
    assert "BULK_LOSS_NOT_ESTABLISHED" in {i.code.value for i in run.result.issues}


def test_an_axicon_whose_light_misses_the_plane_stays_out_of_scope(monkeypatch):
    document = PROBE.probe_project(AXICON, CATALOG)
    document["observations"][0]["pose"]["position_mm"] = [20.0, 0.0, 500.0]  # far above the beam: no arrival
    monkeypatch.setattr(PROBE, "probe_project", lambda *_: document)
    entry = PROBE.probe_one(AXICON, CATALOG)
    assert entry["status"] == "OUT_OF_SCOPE" and PROBE.AXICON_GAUSSIAN_SCOPE in entry["codes"]


def test_the_gaussian_scope_of_an_axicon_is_not_accepted_for_another_form(monkeypatch):
    lens = ONE_PER_FORM[("LENS", "PLANO_CONVEX_LENS")]
    run = run_optical_project(PROBE.probe_project(AXICON, CATALOG), catalog=CATALOG)
    assert not PROBE._axicon_scope_only(lens, run, run.status.value, [], [])
    assert PROBE._axicon_scope_only(AXICON, run, run.status.value, [], [])
    assert not PROBE._axicon_scope_only(AXICON, run, run.status.value, ["SOME_SCOPE_ISSUE"], [])


def test_edge_filters_and_dichroics_are_probed_in_their_high_and_low_transmission_bands():
    for form in ("LONGPASS_FILTER", "SHORTPASS_FILTER", "DICHROIC_MIRROR"):
        product = next(p for key, p in sorted(ONE_PER_FORM.items()) if key[1] == form)
        coating = next(CATALOG.coating(s.coating_id) for s in product.simulation_models[0].surfaces if s.coating_id)
        transmittances = {coating.band_at(Decimal(str(w))).transmittance for w in PROBE._wavelengths(product, CATALOG)}
        assert len(transmittances) > 1, form


NORMAL_INCIDENCE_REFLECTORS = [p for p in CATALOG.products if p.frame.is_reflector and PROBE._probe_aoi(p, CATALOG) == 0]


def test_the_seed_release_has_reflectors_the_probe_turns_to_normal_incidence():
    # a failure, not a skip: the regression below must keep exercising a real 0 deg product
    assert NORMAL_INCIDENCE_REFLECTORS


@pytest.mark.parametrize("product", NORMAL_INCIDENCE_REFLECTORS, ids=lambda p: p.optical_sku)
def test_a_normal_incidence_reflector_is_probed_with_its_plane_behind_the_source(product):
    """A 0 deg mirror reflects the probe beam straight back onto its source. The plane for that beam must not lie on the
    incoming beam (one beam path would cross it twice, which the Optical Simulation View refuses): it lies
    behind the source, and the product is in scope."""

    document = PROBE.probe_project(product, CATALOG)
    (plane,) = document["observations"]
    assert plane["pose"]["position_mm"] == [-(PROBE.SOURCE_DISTANCE_MM + PROBE.OBSERVATION_DISTANCE_MM), 0.0, 0.0]
    assert PROBE.probe_one(product, CATALOG) == {"status": "IN_SCOPE", "run_status": "PASS", "wavelengths": len(PROBE._wavelengths(product, CATALOG))}


def test_no_probe_plane_lies_on_the_incoming_beam():
    """For every reflector of the release the incoming beam (x from -SOURCE_DISTANCE_MM to 0 on the X axis) stays on the
    near side of each observation plane, so no plane is crossed by both the incoming and the outgoing beam."""

    for product in (p for p in CATALOG.products if p.frame.is_reflector):
        for plane in PROBE.probe_project(product, CATALOG)["observations"]:
            position = plane["pose"]["position_mm"]
            distance = sum(c * c for c in position) ** 0.5
            normal = [c / distance for c in position]  # the plane faces along its outgoing beam from the element at the origin
            reach = max(0.0, -normal[0]) * PROBE.SOURCE_DISTANCE_MM  # furthest the incoming beam gets along that beam
            assert reach < distance - 1.0, (product.optical_sku, plane["observation_id"])


# One face-normal splitter per form whose model is a surface set (a wedged plate or a wedged sampler): its transmitted beam
# leaves deviated by a fraction of a degree, toward the reflected beam's side for some orientations.
WEDGED_SPLITTERS = list({p.form: p for p in reversed(CATALOG.products)
                         if p.category.value == "BEAMSPLITTER" and p.frame.is_reflector
                         and p.simulation_models[0].model_kind == "SURFACE_SET"}.values())


def test_the_seed_release_has_wedged_face_normal_splitters():
    assert WEDGED_SPLITTERS  # a failure, not a skip: the regression below must keep exercising a real wedged splitter


@pytest.mark.parametrize("product", WEDGED_SPLITTERS, ids=lambda p: p.form.value)
def test_each_probe_plane_counts_only_the_beam_the_element_sends_toward_it(product):
    """A wedged splitter's transmitted beam is deviated: the reflected beam's plane, were it unbounded, would catch the
    transmitted beam metres away at grazing incidence (a second, unrelated arrival; GAUSSIAN_PLANE_AXIAL_SPREAD). Each
    probe plane has an active area centred on its own beam, so P1 counts the reflected branch only and P2 the transmitted
    beam only; the catalog's unstated substrate loss keeps the *power*, not those routes, UNKNOWN."""

    run = run_optical_project(PROBE.probe_project(product, CATALOG), catalog=CATALOG)
    arrivals = {o.observation_id: {a.beam.path_id for a in o.arrivals} for o in run.result.observations}
    main = {p.path_id for p in run.result.paths if p.parent_path_id is None}
    reflected = {p.path_id for p in run.result.paths if p.parent_path_id in main}
    assert arrivals == {"P1": reflected, "P2": main} and main and reflected
    assert run.status.value == "UNKNOWN"
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in run.result.issues)
    bounded = [a for o in run.result.observations for a in o.arrivals if a.beam.power_W is None]
    assert bounded and all(a.power_upper_bound_W is not None and 0 < a.power_upper_bound_W <= 1e-3 for a in bounded)
    assert all(0 <= a.beam.power_W <= 1e-3 for o in run.result.observations for a in o.arrivals if a.beam.power_W is not None)


def test_a_design_that_misses_the_element_is_not_counted_as_simulatable(monkeypatch):
    product = ONE_PER_FORM[("LENS", "PLANO_CONVEX_LENS")]
    document = PROBE.probe_project(product, CATALOG)
    lens = next(c for c in document["components"] if c["instance_id"] == "E1")
    lens["nominal_pose"]["position_mm"] = [0.0, 0.0, 200.0]  # far above the beam
    assert run_optical_project(document, catalog=CATALOG).status.value in PROBE.IN_SCOPE_STATUSES
    monkeypatch.setattr(PROBE, "probe_project", lambda *_: document)
    assert PROBE.probe_one(product, CATALOG)["status"] == "NOT_EXERCISED"


def test_the_probe_checkpoints_and_resumes(tmp_path, capsys):
    document = json.loads(REAL_CATALOG.read_bytes())
    document["products"] = sorted(document["products"], key=lambda p: p["optical_sku"])[:3]
    document["catalog_version"] = "2026-01-01.9"  # a modified catalog is its own release
    document["categories_offered"] = sorted({p["category"] for p in document["products"]})
    used = json.dumps(document["products"])  # a catalog lists exactly the materials and coatings its products use
    document["materials"] = [m for m in document["materials"] if f'"{m["material_id"]}"' in used]
    document["coatings"] = [c for c in document["coatings"] if f'"{c["coating_id"]}"' in used]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "probe.json"
    assert PROBE.main(["--out", str(out), "--catalog", str(catalog_path)]) == 0
    first = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(first["optics"]) == [p["optical_sku"] for p in document["products"]]
    assert all(entry["status"] == "IN_SCOPE" for entry in first["optics"].values())
    capsys.readouterr()
    assert PROBE.main(["--out", str(out), "--catalog", str(catalog_path)]) == 0
    assert "resuming: 3 of 3 already probed" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8")) == first
