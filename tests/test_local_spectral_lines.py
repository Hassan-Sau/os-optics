"""Multi-line Project sources (``GaussianBeamSourceV1.spectral_lines``): the lowering, per-line results and
power, per-line dispersion and per-line coherence.

References are independent of the production code path: line powers from the document, the paraxial focus of each line
from the test's own Sellmeier indices and ABCD matrices (``local_helpers``), the coherence length of each line from the
closed form ``L_c = (2 sqrt(ln 2) / pi) lambda^2 / d_lambda`` written out here.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot, Target
from os_optics.local import LocalValidationStatusV1, example_project_of_kind, lower_project, run_optical_project

from _frozen_poses import Q_45_NEG, pin_pose
from catalog_store_helpers import REAL_CATALOG
from declared_helpers import declared_project
from helpers import run as run_engine, stated_splitter
from local_helpers import (
    DICHROIC,
    FOLD_X_TO_Y,
    PCX_F100,
    arrivals,
    catalog_document,
    element,
    lens_matrix,
    plane,
    project,
    q_through,
    rayleigh_range,
    run,
    source,
    thickness,
)
from test_golden_interference import _two_beam_geometry

LINES = ((486.1327, 0.002), (656.2725, 0.001))  # (wavelength nm, power W): the F and C lines, author's order


def _ideal_fixture_project(document: dict, catalog: dict) -> dict:
    """Declare test-only copies of catalog parts with stated zero bulk loss.

    Public catalog materials do not accept bulk-loss fields: the release itself
    remains untouched and must not be taken to establish lossless transmission.
    """
    declared = declared_project(document, catalog)
    for material in declared["model_materials"]:
        material["absorption_coefficient_per_mm"] = "0"
    return declared


def multi_line_source(instance_id: str = "SRC", lines=LINES, **kwargs) -> dict:
    document = source(instance_id, **kwargs)
    model = document["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": w, "power_w": p} for w, p in lines]
    return document


def _engine_source(document: dict):
    (component,) = [c for c in lower_project(document, catalog_document()).request.project.components
                    if c.source is not None and c.instance_id == "SRC"]
    return component.source


# ------------------------------------------------------------------------------------------------ lowering


def test_spectral_lines_lower_to_an_engine_spectrum_weighted_by_line_power():
    document = project([multi_line_source(), source("SRC_633", wavelength_nm=633.0)], [plane("screen", (100.0, 0.0, 0.0))])
    spec = _engine_source(document)
    assert spec.wavelength_nm is None
    assert [(line.wavelength_nm, line.weight) for line in spec.spectrum] == list(LINES)
    assert spec.power_W == pytest.approx(sum(p for _, p in LINES), rel=1e-15)
    lowered = lower_project(document, catalog_document())
    assert lowered.source_wavelengths_nm == (486.1327, 633.0, 656.2725)  # every line of every source, ascending


def test_a_single_line_source_lowers_exactly_as_before():
    spec = _engine_source(project([source()], [plane("screen", (100.0, 0.0, 0.0))]))
    assert (spec.wavelength_nm, spec.spectrum, spec.power_W) == (780.0, (), 0.001)


# ------------------------------------------------------------------------------------------------ per-line results


def test_every_line_arrives_as_its_own_beam_with_its_own_power():
    result = run(project([multi_line_source()], [plane("screen", (100.0, 0.0, 0.0))])).result
    beams = sorted(arrivals(result, "screen"), key=lambda a: a.beam.spectral_index)
    assert [(a.beam.spectral_index, a.beam.wavelength_nm) for a in beams] == [(0, LINES[0][0]), (1, LINES[1][0])]
    for arrival, (_, power) in zip(beams, LINES):
        assert arrival.beam.source_id == "SRC"
        assert arrival.beam.power_W == pytest.approx(power, rel=1e-12)
    (screen,) = result.observations
    assert [(s.wavelength_nm, s.power_W) for s in screen.power_by_wavelength] == pytest.approx(sorted(LINES))
    budget = {(line.source_id, line.spectral_index): line for line in result.power_budget.lines}
    assert set(budget) == {("SRC", 0), ("SRC", 1)}
    for index, (wavelength, power) in enumerate(LINES):
        line = budget[("SRC", index)]
        assert (line.wavelength_nm, line.conserved) == (wavelength, True)
        assert line.input_power_W == pytest.approx(power, rel=1e-12)
        assert line.input_power_W == pytest.approx(sum(line.sinks.values()) + line.residual_W, rel=1e-12)
    assert result.power_budget.total_input_W == pytest.approx(sum(p for _, p in LINES), rel=1e-12) and result.power_budget.conserved


def test_each_line_focuses_where_its_own_dispersion_puts_it():
    """F and C lines of ONE source through a N-BK7 plano-convex lens: each line's waist lies where q propagation through
    the prescription at that line's Sellmeier index puts it -- the same as two single-line sources."""

    distance, screen_x = 200.0, 100.0
    lens = element("L1", "lens", PCX_F100)
    combined = run(project([multi_line_source(position=(-distance, 0.0, 0.0)), lens], [plane("screen", (screen_x, 0.0, 0.0))]))
    separate = run(project([source("F", (-distance, 0.0, 0.0), wavelength_nm=LINES[0][0], power_w=LINES[0][1]),
                            source("C", (-distance, 0.0, 0.0), wavelength_nm=LINES[1][0], power_w=LINES[1][1]), lens],
                           [plane("screen", (screen_x, 0.0, 0.0))]))
    assert combined.model_scope_issues == ()
    t = thickness(PCX_F100)
    by_line = {a.beam.wavelength_nm: a for a in arrivals(combined.result, "screen")}
    for name, (wavelength, _) in zip(("F", "C"), LINES):
        predicted = t - q_through(lens_matrix(PCX_F100, wavelength), complex(distance, rayleigh_range(1.0, wavelength))).real
        engine = screen_x - by_line[wavelength].beam.gaussian.waist_distance_x_mm
        assert engine == pytest.approx(predicted, abs=1e-6)
        (alone,) = arrivals(separate.result, "screen", name)
        assert by_line[wavelength].beam.gaussian.waist_distance_x_mm == pytest.approx(alone.beam.gaussian.waist_distance_x_mm, abs=1e-12)
        assert by_line[wavelength].beam.power_W == pytest.approx(alone.beam.power_W, rel=1e-12)
    assert by_line[LINES[1][0]].beam.gaussian.waist_distance_x_mm < by_line[LINES[0][0]].beam.gaussian.waist_distance_x_mm  # C focuses later


def test_targets_select_a_line_by_wavelength():
    document = project([multi_line_source()], [plane("screen", (100.0, 0.0, 0.0))], [
        {"target_id": "f_power", "metric": "power_W", "observation_id": "screen", "wavelength_nm": LINES[0][0],
         "target_value": LINES[0][1], "tolerance_rel": 1e-9, "unit": "W"},
        {"target_id": "c_radius", "metric": "beam_radius_mm", "observation_id": "screen", "source_id": "SRC",
         "wavelength_nm": LINES[1][0], "maximum": 2.0, "unit": "mm"},
        {"target_id": "all_power", "metric": "power_W", "observation_id": "screen", "source_id": "SRC",
         "target_value": sum(p for _, p in LINES), "tolerance_rel": 1e-9, "unit": "W"},
        {"target_id": "ambiguous_radius", "metric": "beam_radius_mm", "observation_id": "screen", "source_id": "SRC", "maximum": 2.0, "unit": "mm"},
    ])
    targets = {t.target_id: t for t in run(document).result.targets}
    assert targets["f_power"].status.value == "PASS" and targets["f_power"].value == pytest.approx(LINES[0][1], rel=1e-12)
    assert targets["c_radius"].status.value == "PASS"
    assert targets["all_power"].status.value == "PASS"
    assert targets["ambiguous_radius"].status.value == "UNKNOWN" and "wavelength_nm" in targets["ambiguous_radius"].message


def test_a_dichroic_routes_each_line_of_one_source():
    """The test catalog's short-pass dichroic (reflects below its edge, transmits above) splits a 532 + 780 nm source."""

    lines = ((532.0, 0.001), (780.0, 0.002))
    document = project(
        [multi_line_source(lines=lines, position=(-100.0, 0.0, 0.0)), element("DM", "splitter", DICHROIC, quaternion=FOLD_X_TO_Y)],
        [plane("reflected", (0.0, 100.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))), plane("transmitted", (100.0, 0.0, 0.0))],
    )
    result = run(_ideal_fixture_project(document, catalog_document()))
    assert result.model_scope_issues == ()
    reflected = {a.beam.wavelength_nm: a.beam.power_W for a in arrivals(result.result, "reflected")}
    transmitted = {a.beam.wavelength_nm: a.beam.power_W for a in arrivals(result.result, "transmitted")}
    assert reflected[532.0] > 0.5 * 0.001 and transmitted[780.0] > 0.5 * 0.002
    assert reflected[780.0] < transmitted[780.0] and transmitted[532.0] < reflected[532.0]
    for line in result.result.power_budget.lines:
        assert line.conserved


# ------------------------------------------------------------------------------------------------ per-line coherence


def _analytic_coherence_length_mm(wavelength_nm: float, linewidth_nm: float) -> float:
    return 2.0 * math.sqrt(math.log(2.0)) / math.pi * wavelength_nm * wavelength_nm / linewidth_nm * 1e-6


def _two_line_interferometer(opd_mm: float, **coherence) -> ProjectSnapshot:
    """The 405 nm golden two-beam geometry (ideal 50/50 splitter with its phases stated, and mirrors, wavelength
    independent) fed by one source of two lines, 405 and 810 nm."""

    d1, d2, ma, mb, s_pt = _two_beam_geometry(opd_mm=opd_mm)
    n_sub = -(d1 + d2) / np.linalg.norm(d1 + d2)
    return ProjectSnapshot(
        components=(
            B.laser("L", spectrum=((405.0, 1.0), (810.0, 1.0)), waist_radius_mm=1.0, position=(-50, 0, 20), direction=(1, 0, 0),
                    power_W=2e-3, polarization=JonesVector.linear(90.0), **coherence),
            pin_pose(stated_splitter(B.npbs("BS", position=(0, 0, 20), normal=(-1, 1, 0), reflectance=0.5)), Q_45_NEG),
            B.mirror_for_reflection("MA", position=tuple(ma), d_in=(1, 0, 0), d_out=tuple(d1)),
            B.mirror_for_reflection("MB", position=tuple(mb), d_in=(0, 1, 0), d_out=tuple(d2)),
        ),
        observations=(B.observation_plane("sub", position=tuple(s_pt), normal=tuple(n_sub), radius_mm=10.0),),
        targets=(Target(target_id="vis_810", metric="fringe_visibility", observation_id="sub", wavelength_nm=810.0, minimum=0.0),),
    )


@pytest.mark.golden
@pytest.mark.parametrize("opd", [0.0, 5.0, 15.0])
def test_each_line_has_the_coherence_length_of_its_own_wavelength(opd):
    linewidth = 0.01
    result = run_engine(_two_line_interferometer(opd, linewidth_nm=linewidth))
    same = {p.wavelength_nm: p for p in result.interference}
    # two arms x two lines: one pair per line; cross-line pairs (different frequencies) never interfere and are not analysed
    assert set(same) == {405.0, 810.0} and len(result.interference) == 2
    for wavelength, pair in same.items():
        lc = _analytic_coherence_length_mm(wavelength, linewidth)
        assert pair.coherence_length_mm == pytest.approx(lc, rel=1e-12)
        assert pair.optical_path_difference_mm == pytest.approx(opd, abs=1e-9)
        assert pair.coherence_factor == pytest.approx(math.exp(-((opd / lc) ** 2)), rel=1e-9)
        # the two arms' Gaussian envelopes overlap to ~1e-7 at 810 nm (longer divergence): visibility ~ the coherence factor
        assert pair.fringe_visibility == pytest.approx(math.exp(-((opd / lc) ** 2)), rel=1e-6)
    assert same[810.0].coherence_length_mm == pytest.approx(4.0 * same[405.0].coherence_length_mm, rel=1e-12)
    assert same[810.0].fringe_period_nm == pytest.approx(2.0 * same[405.0].fringe_period_nm, rel=1e-9)
    (target,) = result.targets
    assert target.value == pytest.approx(same[810.0].fringe_visibility, rel=1e-12)
    if opd:
        assert same[810.0].fringe_visibility > same[405.0].fringe_visibility  # the longer line stays coherent longer


@pytest.mark.golden
def test_a_declared_coherence_length_holds_for_every_line():
    result = run_engine(_two_line_interferometer(5.0, coherence_length_mm=20.0))
    pairs = [p for p in result.interference if "WAVELENGTH_MISMATCH" not in p.issue_codes]
    assert sorted(p.wavelength_nm for p in pairs) == [405.0, 810.0]
    for pair in pairs:
        assert pair.coherence_length_mm == 20.0
        assert pair.coherence_factor == pytest.approx(math.exp(-((5.0 / 20.0) ** 2)), rel=1e-9)


def test_a_project_linewidth_gives_each_line_its_own_coherence_length():
    """A Project source's linewidth reaches the Engine scene as one coherence length per line, from each line's own
    wavelength (the interference of each line is checked by the golden cases above)."""

    from os_optics.config import DEFAULT_NUMERICAL_CONFIG
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.scene.lower import lower_scene

    document = copy.deepcopy(project([multi_line_source()], [plane("screen", (100.0, 0.0, 0.0))]))
    document["components"][0]["source_model"]["linewidth_nm"] = 0.002
    request = lower_project(document, catalog_document()).request
    scene, _ = lower_scene(request.project, DEFAULT_NUMERICAL_CONFIG, MaterialLibrary(request.project.materials, DEFAULT_NUMERICAL_CONFIG))
    (src,) = scene.sources
    assert src.coherence_lengths_mm == pytest.approx(tuple(_analytic_coherence_length_mm(w, 0.002) for w, _ in LINES), rel=1e-12)


# ------------------------------------------------------------------------------------------------ the reference design


@pytest.mark.parametrize("catalog", ["fixture", "seed"])
def test_the_two_color_example_routes_each_line_to_its_detector(catalog):
    document = catalog_document() if catalog == "fixture" else REAL_CATALOG
    example = example_project_of_kind("two_color_dichroic", document)
    project_ = _ideal_fixture_project(example.payload(), document) if catalog == "fixture" else example
    run_ = run_optical_project(project_, catalog=document)
    assert run_.status is LocalValidationStatusV1.PASS and run_.model_scope_issues == ()
    (src,) = [c for c in run_.project.components if c.optical_role.value == "source"]
    lines = src.source_model.emission_lines
    assert len(lines) == 2 and [line.power_w for line in lines] == [0.001, 0.002]
    targets = {t.target_id: t for t in run_.result.targets}
    assert targets["reflected_line_power"].value > 0.5 * lines[0].power_w
    assert targets["transmitted_line_power"].value > 0.5 * lines[1].power_w
    for observation in run_.result.observations:
        assert sorted(s.wavelength_nm for s in observation.power_by_wavelength) == sorted(line.wavelength_nm for line in lines)
    assert all(line.conserved for line in run_.result.power_budget.lines)
    assert {line.spectral_index for line in run_.result.power_budget.lines} == {0, 1}
