"""The Optical Simulation View draws a path as a main beam only when it is the designed light.

A path is PRIMARY until it becomes a weak branch at a split (``SECONDARY_BRANCH_POWER_SHARE``) or meets an element that
rejects its line (``BLOCKED_POWER_SHARE`` of the element's best share: a filter's stop band; a neutral density filter
never blocks). A path that reaches a plane whose target asks for it (``designed_arrivals``) has its weak stretches
lifted, never a blocked one: a target naming a source and a line asks for that light; a target leaving them open sums
what arrives and asks for the beams carrying a material share of it; a power target asks for light only when it
requires more than zero (a leak bound written as ``minimum: 0`` or ``target_value: 0`` asks for none). A designed weak
pick-off stays PRIMARY, and so does a weak beam whose target bounds a non-power metric. The snapshot ``legend.md``
prints each path's display role.

On ``main`` @ ``8afcbc95`` any path reaching a plane with a target for its source was PRIMARY: the shipped
``two_color_dichroic`` example drew both dichroic leaks as main beams, a leak-bound target promoted the very leak it
bounds, and a beam blocked by a filter stayed a main beam.
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from os_optics.local import example_project_of_kind, load_public_catalog, run_optical_project
from os_optics.local.catalog import load_project
from os_optics.local.example import BEAM_ALONG_X, BEAM_ALONG_Y, FOLD_X_TO_Y, MIRROR_DOFS, _element, _plane, _project, _source
from os_optics.viewer import optical_view_scene, render_snapshots

from catalog_store_helpers import REAL_CATALOG, pinned_to_seed

IR, NIR = 1064.0, 780.0


@pytest.fixture(scope="module")
def catalog():
    return load_public_catalog(REAL_CATALOG)


def _roles(project) -> dict[str, tuple[list[str], str]]:
    run = run_optical_project(project, catalog=REAL_CATALOG)
    assert run.model_scope_issues == ()
    if run.status.value == "UNKNOWN":
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
    else:
        assert run.status.value == "PASS", run.status
    return {p.path_id: ([r.value for r in p.segment_roles], p.display_role.value) for p in run.optical_view.beam_paths}


def test_the_shipped_two_color_example_draws_its_leaks_as_weak_branches(tmp_path):
    project = example_project_of_kind("two_color_dichroic", REAL_CATALOG)
    run_optical_project(project, catalog=REAL_CATALOG, output_dir=tmp_path / "run")
    scene = optical_view_scene(tmp_path / "run")
    roles = {b.beam_id: b.facts["display_role"] for b in scene.beams}
    # SRC#0: the reflected line's 5 % leak through the dichroic; SRC#1.0: the transmitted line's 10 % reflection
    assert roles == {"SRC#0": "SECONDARY", "SRC#0.0": "PRIMARY", "SRC#1": "PRIMARY", "SRC#1.0": "SECONDARY"}
    render_snapshots(tmp_path / "run", tmp_path / "shots", views=("top",), width=640, height=480)
    legend = (tmp_path / "shots" / "legend.md").read_text(encoding="utf-8")
    beams = legend.split("## Beams", 1)[1].strip().splitlines()
    assert beams == ["- `SRC#0` 1064 nm: SECONDARY", "- `SRC#0.0` 1064 nm: PRIMARY", "- `SRC#1` 532 nm: PRIMARY",
                     "- `SRC#1.0` 532 nm: SECONDARY"]


def test_a_target_that_only_bounds_a_leak_does_not_promote_it():
    project = example_project_of_kind("dichroic_routing", REAL_CATALOG).payload()
    bound = {"target_id": "leak_bound", "metric": "power_W", "observation_id": "transmitted_leg", "source_id": "SRC_R",
             "maximum": 1e-4, "unit": "W"}
    with_bound = _roles(_project(project["components"], project["observations"], [*project["targets"], bound]))
    without = _roles(_project(project["components"], project["observations"], project["targets"]))
    assert with_bound == without
    assert with_bound["SRC_R#0"] == (["PRIMARY", "SECONDARY"], "SECONDARY")  # the 1064 nm leak through the dichroic


def _blocking_filter(catalog):
    """The first ORDERABLE short-pass filter whose coatings all cover 1064 nm and pass less than 1e-3 of it."""

    for product in sorted(catalog.products, key=lambda p: p.optical_sku):
        if product.form.value != "SHORTPASS_FILTER" or product.commercial_readiness.value != "ORDERABLE" or len(product.simulation_models) != 1:
            continue
        coated = [s for s in getattr(product.simulation_models[0], "surfaces", ()) if s.coating_id]
        bands = [catalog.coating(s.coating_id).band_at(Decimal(repr(IR))) for s in coated]
        if bands and all(b is not None for b in bands) and Decimal(bands[0].transmittance) < Decimal("1e-3"):
            return product
    raise AssertionError("no blocking short-pass filter for 1064 nm")


def test_light_beyond_a_blocking_filter_is_no_main_beam(catalog):
    components = [_source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=IR),
                  _element("F1", "filter", _blocking_filter(catalog), catalog, [0.0, 0.0, 0.0], BEAM_ALONG_X)]
    observations = [_plane("screen", [100.0, 0.0, 0.0], BEAM_ALONG_X)]
    leak = {"target_id": "leak", "metric": "power_W", "observation_id": "screen", "source_id": "SRC", "maximum": 1e-6, "unit": "W"}
    for targets in ([leak], [{**leak, "source_id": None}]):
        roles = _roles(_project(components, observations, targets))
        (segments, display), = [v for k, v in roles.items() if k.startswith("SRC#0")][:1]
        assert segments[0] == "PRIMARY" and segments[-1] == "SECONDARY" and display == "SECONDARY"


def _plate(catalog, reflectance: str):
    for product in sorted(catalog.products, key=lambda p: p.optical_sku):
        if product.form.value != "PLATE_BEAMSPLITTER" or product.commercial_readiness.value != "ORDERABLE" or len(product.simulation_models) != 1:
            continue
        split = [s for s in product.simulation_models[0].surfaces if s.traced.value == "TRANSMITTED_AND_REFLECTED" and s.coating_id]
        if len(split) == 1:
            bands = [catalog.coating(split[0].coating_id).band_at(Decimal(repr(w))) for w in (IR, NIR)]
            if all(b is not None and Decimal(b.reflectance) == Decimal(reflectance) for b in bands):
                return product
    raise AssertionError(f"no plate beamsplitter reflecting {reflectance}")


def _pick_off(catalog, targets, lines=((IR, 0.001), (NIR, 0.001))):
    """A laser along +X onto a 10 / 90 plate beamsplitter: ``out_T`` on the transmitted arm, ``out_R`` on the 10 % arm."""

    return _project([_source("SRC", [-100.0, 0.0, 0.0], spectral_lines=list(lines)) if len(lines) > 1
                     else _source("SRC", [-100.0, 0.0, 0.0], wavelength_nm=lines[0][0]),
                     _element("BS", "splitter", _plate(catalog, "0.1"), catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS)],
                    [_plane("out_T", [100.0, 0.0, 0.0], BEAM_ALONG_X), _plane("out_R", [0.0, 100.0, 0.0], BEAM_ALONG_Y)], targets)


def _reflected(roles, spectral_index: int) -> str:
    """The display role of the plate's reflected branch of one line (the branch that starts at the coated face)."""

    branches = {k: v for k, v in roles.items() if k.startswith(f"SRC#{spectral_index}.")}
    return min(branches.items())[1][1]


def test_a_designed_weak_pick_off_stays_a_main_beam_for_the_line_it_asks_for(catalog):
    pick = {"target_id": "pick", "metric": "power_W", "observation_id": "out_R", "source_id": "SRC", "wavelength_nm": IR,
            "minimum": 5e-5, "unit": "W"}
    main = {"target_id": "main", "metric": "power_W", "observation_id": "out_T", "source_id": "SRC", "minimum": 5e-4, "unit": "W"}
    roles = _roles(_pick_off(catalog, [main, pick]))
    assert _reflected(roles, 0) == "PRIMARY"  # 1064 nm: the designed pick-off
    assert _reflected(roles, 1) == "SECONDARY"  # 780 nm on the same arm: nobody asked for it
    assert roles["SRC#0"][1] == roles["SRC#1"][1] == "PRIMARY"


@pytest.mark.parametrize(("metric", "bound", "role"), [("beam_radius_mm", 10.0, "PRIMARY"), ("power_W", 1e-3, "SECONDARY")])
def test_an_upper_bound_promotes_only_on_a_metric_that_is_no_amount_of_light(catalog, metric, bound, role):
    target = {"target_id": "bound", "metric": metric, "observation_id": "out_R", "source_id": "SRC", "maximum": bound,
              "unit": "mm" if metric.endswith("_mm") else "W"}
    roles = _roles(_pick_off(catalog, [target], lines=((IR, 0.001),)))
    assert _reflected(roles, 0) == role


def test_a_target_without_a_source_asks_for_the_main_beams_of_its_plane_not_the_leaks():
    project = example_project_of_kind("dichroic_routing", REAL_CATALOG).payload()
    targets = copy.deepcopy(project["targets"])
    for target in targets:
        target["source_id"] = None
    roles = _roles(_project(project["components"], project["observations"], targets))
    # the leaks reach the targeted legs but nothing asks for them: the power share decides
    assert {k: v[1] for k, v in roles.items()} == {"SRC_R#0": "SECONDARY", "SRC_R#0.0": "PRIMARY", "SRC_T#0": "PRIMARY",
                                                    "SRC_T#0.0": "SECONDARY"}


def test_a_target_naming_only_a_line_asks_for_that_line_on_its_plane(catalog):
    pick = {"target_id": "pick", "metric": "power_W", "observation_id": "out_R", "wavelength_nm": IR, "minimum": 5e-5, "unit": "W"}
    roles = _roles(_pick_off(catalog, [pick]))
    assert (_reflected(roles, 0), _reflected(roles, 1)) == ("PRIMARY", "SECONDARY")


# ---------------------------------------------------------------------- review designs
#
# ``fixtures/beam_role_review`` holds the reviewer's reproduction designs (re-pinned to the current seed when read): the 532 / 1064 nm
# separator with a leak bound written as ``minimum: 0`` (a9_leak_min0), as ``target_value: 0`` within a tolerance
# (a9_leak_tv0) and a position target on the blocked IR leak (a9_leak_pos); a two-colour Mach-Zehnder with a filter in
# each arm (a10_mz_filters); a three-line laser on two dichroics with targets naming no source or line
# (a5_multiline_nosrc); a 10 % pick-off of two lasers whose target names no source (a4_pick_nosrc_multi); a neutral
# density OD4 filter before a camera with a saturation bound, no target, and the absorptive kind (a3_nd4_*).

REVIEW_DESIGNS = Path(__file__).parent / "fixtures" / "beam_role_review"
#: path id -> (segment roles as P/S letters, display role); before the beam-role fix the leaks were PRIMARY
#: (D1 / D2 / D3), the designed pick-off SECONDARY (D5) and the ND-attenuated beam SECONDARY (D4).
REVIEW_ROLES = {
    "a9_leak_min0": {"SRC_IR#0": ("PSSSSS", "SECONDARY"), "SRC_IR#0.0": ("PPPPPP", "PRIMARY"),
                     "SRC_G#0": ("PPPPPP", "PRIMARY"), "SRC_G#0.0": ("SSSSSS", "SECONDARY")},
    "a10_mz_filters": {"S532#0": ("PPPPPP", "PRIMARY"), "S532#0.0": ("PPSSS", "SECONDARY"), "S532#0.1": ("PP", "PRIMARY"),
                       "S633#0": ("PPPSSS", "SECONDARY"), "S633#0.0": ("PPPPP", "PRIMARY"), "S633#0.1": ("SS", "SECONDARY")},
    "a5_multiline_nosrc": {"SRC#0": ("PSS", "SECONDARY"), "SRC#0.0": ("PP", "PRIMARY"), "SRC#0.1": ("SS", "SECONDARY"),
                           "SRC#1": ("PPS", "SECONDARY"), "SRC#1.0": ("SS", "SECONDARY"), "SRC#1.1": ("PP", "PRIMARY"),
                           "SRC#2": ("PPP", "PRIMARY"), "SRC#2.0": ("SS", "SECONDARY"), "SRC#2.1": ("SS", "SECONDARY")},
    # (the main beams end past MAIN: their chief rays no longer cross the PICK plane 14 m on, off its area, ce3133ada)
    "a4_pick_nosrc_multi": {"SRC_G#0": ("PPP", "PRIMARY"), "SRC_G#0.0": ("P", "PRIMARY"),
                            "SRC_R#0": ("PPP", "PRIMARY"), "SRC_R#0.0": ("P", "PRIMARY")},
    "a3_nd4_max": {"SRC#0": ("PP", "PRIMARY")},
}
REVIEW_ROLES["a9_leak_tv0"] = REVIEW_ROLES["a9_leak_pos"] = REVIEW_ROLES["a9_leak_min0"]
REVIEW_ROLES["a3_nd4_none"] = REVIEW_ROLES["a3_nd4abs_max"] = REVIEW_ROLES["a3_nd4_max"]


@pytest.mark.parametrize("design", sorted(REVIEW_ROLES))
def test_review_designs_draw_only_the_designed_light_as_main_beams(design):
    run = run_optical_project(load_project(pinned_to_seed(json.loads((REVIEW_DESIGNS / f"{design}.json").read_bytes()))), catalog=REAL_CATALOG)
    assert run.model_scope_issues == ()
    if run.status.value == "UNKNOWN":
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
    else:
        assert run.status.value == "PASS", run.status
    got = {p.path_id: ("".join(r.value[0] for r in p.segment_roles), p.display_role.value)
           for p in run.optical_view.beam_paths if p.segment_roles}
    assert got == REVIEW_ROLES[design]
    if design == "a10_mz_filters":
        # The 633 nm line passes unknown-loss glass before the short-pass filter.
        # Missing bulk transmission is not evidence that the blocked leak has zero power,
        # nor a reason to lose its SECONDARY display role established by the filter coating.
        filter_sku = next(component.product.optical_sku for component in run.project.components if component.instance_id == "F_T")
        catalog = load_public_catalog(REAL_CATALOG)
        public_filter = catalog.get(filter_sku)
        coated_surface = next(surface for surface in public_filter.simulation_models[0].surfaces if surface.coating_id)
        coating = catalog.coating(coated_surface.coating_id)
        assert Decimal(coating.band_at(Decimal("633")).transmittance) <= Decimal("1e-4")
        blocked = next(path for path in run.result.paths if path.path_id == "S633#0")
        assert any(segment.end_surface_id == "F_T/S1" and segment.power_start_W is None for segment in blocked.segments)
        arrivals = {a.beam.path_id: a for observation in run.result.observations for a in observation.arrivals}
        main, leak = arrivals["S633#0.0"], arrivals["S633#0.1"]
        assert main.beam.power_W is leak.beam.power_W is None
        assert 0 < leak.power_upper_bound_W < 1e-4 * main.power_upper_bound_W
