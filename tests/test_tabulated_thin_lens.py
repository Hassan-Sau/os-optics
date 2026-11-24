"""TABULATED_THIN_LENS: a Project-declared thin focusing element whose focal length and transmittance are tabulated per
wavelength (an external response model).

Each traced line is focused with its own focal length (checked against an independent Gaussian ABCD computation in this
file), carries its own transmittance (the rest is booked ``UNMODELLED_RESPONSE``), and the element's validity domain
(wavelengths, field angle, beam size, polarization, validated metrics) is enforced by the Engine as ``OUT_OF_MODEL_SCOPE``.

Fixture data: the rows, errors and evidence below are made up for the test (shaped like a metalens fit).
"""

from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

from os_optics.contracts.optics import TabulatedThinLensInteraction
from os_optics.local import export_linear_optical_network, run_optical_project
from os_optics.local.catalog import load_project
from os_optics.local.compile import ProjectResolver, compile_declared, paraxial_matrix
from os_optics.optics.surfaces.tabulated_lens import ThinLensTable, response_at
from os_optics.network.errors import NetworkExportError
from os_optics.results import load_run

from declared_helpers import declared_component, with_models
from local_helpers import plane, project, source
from test_local_spectral_lines import multi_line_source

ROWS = ((490.0, 216.9, 0.70), (532.0, 199.3, 0.95), (580.0, 183.3, 0.80))
LENS_X = 0.0  # the element plane (global x)
SOURCE_X = -200.0
PLANE_X = 190.0
WAIST_MM = 0.5
EVIDENCE = {"method": "fixture: RCWA library + local phase approximation + angular spectrum", "tools": [{"name": "fixture-rcwa", "version": "0.1.2"}],
            "source_data_sha256": "a" * 64, "configuration_sha256": "b" * 64, "fit_model": "f = c0 + c1 (532 / lambda)",
            "validation_summary": "focus within 0.2 % on held-out wavelengths", "date": "2026-10-15"}


def ttl_model(**fields) -> dict:
    model = {"model_kind": "TABULATED_THIN_LENS", "model_version": "1",
             "rows": [{"wavelength_nm": f"{w:g}", "focal_length_mm": f"{f:g}", "transmittance": f"{t:g}"} for w, f, t in ROWS],
             "focal_interpolation": "LINEAR_IN_OPTICAL_POWER", "transmittance_interpolation": "LINEAR", "clear_radius_mm": "10",
             "edge_radius_mm": "12.7", "max_field_angle_deg": "5",
             "supported_metrics": [{"metric": "waist_distance_x_mm", "rel_error": "0.005", "basis": "held-out wavelengths"},
                                   {"metric": "waist_distance_y_mm", "rel_error": "0.005", "basis": "held-out wavelengths"},
                                   {"metric": "transmittance", "abs_error": "0.15", "basis": "held-out wavelengths"}],
             "evidence": copy.deepcopy(EVIDENCE), "status": "DERIVED"}
    model.update(fields)
    return {"model_id": "ml", "model": model}


def laser(lines, **kwargs) -> dict:
    """A source of one line, or of several (``spectral_lines``)."""

    kwargs.setdefault("position", (SOURCE_X, 0.0, 0.0))
    kwargs.setdefault("waist_mm", WAIST_MM)
    if len(lines) == 1:
        (wavelength, power), = lines
        return source(wavelength_nm=wavelength, power_w=power, **kwargs)
    return multi_line_source(lines=lines, **kwargs)


def ttl_project(lines=((500.0, 0.001), (560.0, 0.002)), *, model=None, targets=None, src=None) -> dict:
    laser_ = src or laser(lines)
    document = project([laser_, declared_component("ML", "lens", "ml", position=(LENS_X, 0.0, 0.0))],
                       [plane("P", (PLANE_X, 0.0, 0.0), radius_mm=20.0)], targets)
    return with_models(document, [model or ttl_model()])


def focal(wavelength_nm: float) -> float:
    """Optical power linear in wavelength between the rows (independent of the Engine)."""

    for (w0, f0, _), (w1, f1, _) in zip(ROWS, ROWS[1:]):
        if w0 <= wavelength_nm <= w1:
            s = (wavelength_nm - w0) / (w1 - w0)
            return 1.0 / (1.0 / f0 + (1.0 / f1 - 1.0 / f0) * s)
    raise AssertionError("outside the table")


def transmittance(wavelength_nm: float) -> float:
    for (w0, _, t0), (w1, _, t1) in zip(ROWS, ROWS[1:]):
        if w0 <= wavelength_nm <= w1:
            return t0 + (t1 - t0) * (wavelength_nm - w0) / (w1 - w0)
    raise AssertionError("outside the table")


def waist_after_lens(wavelength_nm: float) -> tuple[float, float]:
    """(waist position behind the lens, waist radius): Gaussian ABCD, q = z + i zR, thin lens 1/q' = 1/q - 1/f."""

    lam = wavelength_nm * 1e-6
    q = complex(LENS_X - SOURCE_X, math.pi * WAIST_MM**2 / lam)  # waist at the source
    q2 = 1.0 / (1.0 / q - 1.0 / focal(wavelength_nm))
    return -q2.real, math.sqrt(lam * q2.imag / math.pi)


# ------------------------------------------------------------------------------------------------ physics


def test_each_line_is_focused_with_its_own_focal_length() -> None:
    lines = ((500.0, 0.001), (532.0, 0.002), (560.0, 0.0015))
    run = run_optical_project(ttl_project(lines))
    arrivals = {a.beam.wavelength_nm: a for a in run.result.observations[0].arrivals}
    assert sorted(arrivals) == [500.0, 532.0, 560.0]
    positions = []
    for wavelength, power in lines:
        g = arrivals[wavelength].beam.gaussian
        expected_z, expected_w0 = waist_after_lens(wavelength)
        # waist_distance = signed distance waist -> plane, so the waist lies at plane - waist_distance
        assert PLANE_X - g.waist_distance_x_mm == pytest.approx(expected_z, rel=1e-9)
        assert PLANE_X - g.waist_distance_y_mm == pytest.approx(expected_z, rel=1e-9)
        assert g.waist_radius_x_mm == pytest.approx(expected_w0, rel=1e-9)
        assert arrivals[wavelength].beam.power_W == pytest.approx(power * transmittance(wavelength), rel=1e-12)
        positions.append(expected_z)
    assert positions[0] > positions[1] > positions[2]  # a diffractive-like lens: shorter focal length at longer wavelength
    assert run.status.value == "PASS" or run.status.value == "WARNING"
    assert not [i for i in run.result.issues if i.status.value == "OUT_OF_MODEL_SCOPE"]


def test_unmodelled_response_is_its_own_power_ledger_sink() -> None:
    lines = ((500.0, 0.001), (560.0, 0.002))
    run = run_optical_project(ttl_project(lines))
    budget = run.result.power_budget
    expected = sum(p * (1.0 - transmittance(w)) for w, p in lines)
    assert budget.total_sinks["UNMODELLED_RESPONSE"] == pytest.approx(expected, rel=1e-12)
    entries = [e for e in budget.entries if e.kind == "UNMODELLED_RESPONSE"]
    assert {e.surface_id for e in entries} == {"ML/lens"} and {e.element_id for e in entries} == {"ML"}
    assert budget.conserved
    # the light the response does not deliver may still reach the plane: a summed power there is a lower bound, which meets
    # a minimum of 0 W, but a maximum just above the traced power is not decided
    traced = 0.001 * transmittance(500.0)
    targets = [{"target_id": "p", "observation_id": "P", "metric": "power_W", "wavelength_nm": 500.0, "minimum": 0.0},
               {"target_id": "q", "observation_id": "P", "metric": "power_W", "wavelength_nm": 500.0, "maximum": 1.001 * traced}]
    target, upper = run_optical_project(ttl_project(lines, targets=targets)).result.targets
    assert target.status.value == "PASS" and "UNMODELLED_RESPONSE" in target.message
    assert target.value == pytest.approx(traced, rel=1e-12)
    assert upper.status.value == "UNKNOWN" and upper.value == pytest.approx(traced, rel=1e-12)


@pytest.mark.parametrize("transmittance_text, left_out", [("0.999999999999", True), ("0.999", False)])
def test_a_response_share_at_or_below_the_negligible_level_is_left_out_inside_the_bound(transmittance_text, left_out) -> None:
    # one 1 W line at 532 nm through a lens that delivers T of it: the undelivered 1 - T is UNMODELLED_RESPONSE. At 1e-12 of
    # the line (below the negligible level 1e-9) it is left out as negligible: the plane is complete and KNOWN, and its upper
    # bound covers the traced power plus that share. At 1e-3 it keeps an unknown fate and the plane reads a lower bound.
    t = float(transmittance_text)
    rows = [{"wavelength_nm": f"{w:g}", "focal_length_mm": f"{f:g}", "transmittance": transmittance_text} for w, f, _ in ROWS]
    result = run_optical_project(ttl_project(((532.0, 1.0),), model=ttl_model(rows=rows))).result
    (line,) = result.power_budget.lines
    assert line.sinks["UNMODELLED_RESPONSE"] == pytest.approx(1.0 - t, rel=1e-3)
    (plane_p,) = result.observations
    assert plane_p.total_power_W == pytest.approx(t, rel=1e-12)
    if left_out:
        assert plane_p.quantity_state == "KNOWN" and plane_p.power_complete is True and plane_p.untraced_power_W == 0.0
        assert plane_p.power_upper_bound_W >= plane_p.total_power_W + line.sinks["UNMODELLED_RESPONSE"]
        assert result.exposures.trace_complete is True and result.exposures.unknown_fate_power_W == 0.0
        assert "value_basis" not in plane_p.model_dump(exclude_none=True)
    else:
        assert plane_p.quantity_state == "BOUNDED" and plane_p.power_complete is False
        assert plane_p.untraced_power_W == pytest.approx(1.0 - t, rel=1e-12) and plane_p.power_upper_bound_W is None
        entry = plane_p.value_basis["total_power_W"]
        assert entry.kind.value == "LOWER_BOUND" and [o.at for o in entry.omitted] == [("surface:ML/lens",)]
        assert result.exposures.trace_complete is False


def test_paraxial_matrix_uses_the_focal_length_of_the_wavelength() -> None:
    document = load_project(ttl_project())
    element = compile_declared("ML", document.optical_models[0], ProjectResolver(document))
    assert paraxial_matrix(element, 510.0)[1][0] == pytest.approx(-1.0 / focal(510.0), rel=1e-15)
    with pytest.raises(ValueError, match="outside its response table"):
        paraxial_matrix(element, 450.0)


# ------------------------------------------------------------------------------------------------ validity domain


def _codes(run) -> set[str]:
    return {i.code.value for i in run.result.issues if i.status.value == "OUT_OF_MODEL_SCOPE"}


def test_wavelength_outside_the_rows_span_is_out_of_model_scope() -> None:
    run = run_optical_project(ttl_project(((450.0, 0.001), (500.0, 0.001))))
    assert run.status.value == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in run.result.issues if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    assert issue.location == "surface:ML/lens" and "450.0 nm" in issue.message and "490.0-580.0 nm" in issue.message


def test_points_only_rows_and_a_single_row_match_within_half_a_percent() -> None:
    points = ttl_model(focal_interpolation="POINTS_ONLY", transmittance_interpolation="POINTS_ONLY")
    inside = run_optical_project(ttl_project(((532.0, 0.001), (581.0, 0.001)), model=points))  # 581 within 0.5 % of 580
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" not in _codes(inside)
    between = run_optical_project(ttl_project(((510.0, 0.001),), model=points))
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in _codes(between)
    single = ttl_model(rows=[{"wavelength_nm": "532", "focal_length_mm": "199.3", "transmittance": "0.95"}])
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" not in _codes(run_optical_project(ttl_project(((534.0, 0.001),), model=single)))
    assert "MODEL_WAVELENGTH_OUTSIDE_BAND" in _codes(run_optical_project(ttl_project(((535.0, 0.001),), model=single)))


def test_field_angle_beyond_the_validated_one_is_out_of_model_scope() -> None:
    # the source 200 mm before the lens, 26.25 mm off axis and aimed at the lens centre: 7.47 deg from the axis
    off = 26.25
    angle = math.atan2(off, -SOURCE_X)
    tilted = laser(((532.0, 0.001),), position=(SOURCE_X, 0.0, off), quaternion=_aim(angle))
    run = run_optical_project(ttl_project(src=tilted))
    (issue,) = [i for i in run.result.issues if i.code.value == "MODEL_FIELD_ANGLE_EXCEEDED"]
    assert issue.numerical_context["field_angle_deg"] == pytest.approx(math.degrees(angle), abs=1e-9)
    on_axis = run_optical_project(ttl_project(((532.0, 0.001),)))
    assert "MODEL_FIELD_ANGLE_EXCEEDED" not in _codes(on_axis)


def _aim(angle_rad: float) -> tuple[float, float, float, float]:
    """The source pose ALONG_X turned about global +Y by ``angle_rad`` (its beam tilted from +X toward -Z)."""

    from local_helpers import ALONG_X, quat_multiply

    turn = (math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0)
    return quat_multiply(turn, ALONG_X)


def test_beam_larger_than_validated_is_out_of_model_scope() -> None:
    small = ttl_model(max_beam_radius_mm="2")
    assert "MODEL_BEAM_SIZE_EXCEEDED" not in _codes(run_optical_project(ttl_project(((532.0, 0.001),), model=small)))
    tight = ttl_model(max_beam_radius_mm="0.3")
    run = run_optical_project(ttl_project(((532.0, 0.001),), model=tight))
    (issue,) = [i for i in run.result.issues if i.code.value == "MODEL_BEAM_SIZE_EXCEEDED"]
    lam = 532e-6
    z_r = math.pi * WAIST_MM**2 / lam
    expected = WAIST_MM * math.sqrt(1.0 + (200.0 / z_r) ** 2)
    assert issue.numerical_context["beam_radius_mm"] == pytest.approx(expected, rel=1e-9)


def test_polarization_outside_the_validated_states_is_out_of_model_scope() -> None:
    # element frame: ALONG_X maps local X to global Y, local Y to global Z; the source's 0 deg polarization lies along its
    # local X, i.e. global Y (the element's local X), 90 deg along global Z (the element's local Y)
    model = ttl_model(validated_polarizations=["LINEAR_X"])
    along_x = laser(((532.0, 0.001),), polarization_deg=0.0)
    along_y = laser(((532.0, 0.001),), polarization_deg=90.0)
    diagonal = laser(((532.0, 0.001),), polarization_deg=45.0)
    assert "MODEL_POLARIZATION_NOT_VALIDATED" not in _codes(run_optical_project(ttl_project(src=along_x, model=model)))
    assert "MODEL_POLARIZATION_NOT_VALIDATED" in _codes(run_optical_project(ttl_project(src=along_y, model=model)))
    run = run_optical_project(ttl_project(src=diagonal, model=model))
    (issue,) = [i for i in run.result.issues if i.code.value == "MODEL_POLARIZATION_NOT_VALIDATED"]
    assert issue.numerical_context["orthogonal_power_fraction"] == pytest.approx(0.5, abs=1e-12)
    both = ttl_model(validated_polarizations=["LINEAR_X", "LINEAR_Y"])
    assert "MODEL_POLARIZATION_NOT_VALIDATED" not in _codes(run_optical_project(ttl_project(src=along_y, model=both)))
    anything = ttl_model(validated_polarizations=["ANY"])
    assert "MODEL_POLARIZATION_NOT_VALIDATED" not in _codes(run_optical_project(ttl_project(src=diagonal, model=anything)))


def test_metric_level_scope() -> None:
    targets = [{"target_id": "waist", "observation_id": "P", "metric": "waist_distance_x_mm", "wavelength_nm": 532.0, "minimum": -1000.0},
               {"target_id": "radius", "observation_id": "P", "metric": "beam_radius_mm", "wavelength_nm": 532.0, "maximum": 5.0},
               {"target_id": "power", "observation_id": "P", "metric": "power_W", "wavelength_nm": 532.0, "minimum": 0.0},
               {"target_id": "where", "observation_id": "P", "metric": "position_u_mm", "wavelength_nm": 532.0, "minimum": -1.0}]
    run = run_optical_project(ttl_project(((532.0, 0.001),), targets=targets))
    results = {t.target_id: t for t in run.result.targets}
    assert results["waist"].status.value == "PASS"
    assert results["power"].status.value == "PASS"  # covered by 'transmittance'; the unmodelled response may only add to it
    # a metric the element's record does not validate keeps the number of the ideal thin-lens step, as an estimate that is
    # not graded: the Gaussian radius on the plane from the independent ABCD computation above, the chief ray on the axis
    waist_x, waist_radius = waist_after_lens(532.0)
    rayleigh = math.pi * waist_radius**2 / (532.0 * 1e-6)
    expected = {"radius": waist_radius * math.sqrt(1.0 + ((PLANE_X - waist_x) / rayleigh) ** 2), "where": 0.0}
    for name in ("radius", "where"):
        result = results[name]
        assert result.status.value == "OUT_OF_MODEL_SCOPE" and result.assessment.value == "INDETERMINATE"
        assert result.value == pytest.approx(expected[name], rel=1e-6, abs=1e-9)
        assert "METRIC_NOT_VALIDATED_FOR_ELEMENT" in result.issue_codes and "ML/lens" in result.message
        assert "TARGET_NOT_MET" not in result.issue_codes
        entry = result.value_basis["value"]
        assert entry.kind.value == "ESTIMATE"
        (assumed,) = entry.assumed
        assert (assumed.input.value, assumed.at, assumed.code) == ("RESPONSE_MODEL_UNVALIDATED", ("ML/lens",), "METRIC_NOT_VALIDATED_FOR_ELEMENT")
    # the radius misses no bound here; one that does is not graded either (5 mm maximum against a 0.07 mm radius above)
    missed = run_optical_project(ttl_project(((532.0, 0.001),), targets=[{**targets[1], "maximum": 0.01}])).result.targets[0]
    assert (missed.status.value, missed.assessment.value) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE")
    assert missed.value == pytest.approx(expected["radius"], rel=1e-6)
    # the Local Validation record states the value beside the undecided assessment
    from os_optics.local.record import scientific_projection

    outcomes = {t.target_id: t for t in scientific_projection(run.result, run.model_scope_issues).target_outcomes}
    assert outcomes["radius"].assessment == "INDETERMINATE" and outcomes["radius"].value == results["radius"].value
    no_power = ttl_model(supported_metrics=[{"metric": "waist_distance_x_mm", "rel_error": "0.005", "basis": "fixture"}])
    run = run_optical_project(ttl_project(((532.0, 0.001),), targets=targets[2:3], model=no_power))
    assert run.result.targets[0].status.value == "OUT_OF_MODEL_SCOPE"
    assert run.result.targets[0].value == pytest.approx(0.001 * transmittance(532.0), rel=1e-6)


def test_light_that_misses_the_element_keeps_its_metrics() -> None:
    # a second source passing beside the lens (beyond its 12.7 mm edge) reaches the plane without crossing it
    beside = source("SIDE", position=(SOURCE_X, 0.0, 15.0), wavelength_nm=532.0, waist_mm=WAIST_MM)
    document = ttl_project(((532.0, 0.001),), targets=[
        {"target_id": "side", "observation_id": "P", "metric": "beam_radius_mm", "source_id": "SIDE", "maximum": 5.0}])
    document["components"].append(beside)
    run = run_optical_project(document)
    assert run.result.targets[0].status.value == "PASS"


def test_a_pair_target_reads_the_scope_of_its_own_two_beams() -> None:
    """Three beams of 532 nm reach the plane: one through the lens, two beside it (beyond its 12.7 mm edge, 30 mm apart).
    A pair or geometric pair target that names two paths reads the lineage of those two beams: the pair beside the lens is
    graded (their centres are 30 mm apart: the sources' own offsets), a pair with one arm through the lens is out of scope
    with its value."""

    def side(name: str, z: float) -> dict:
        return source(name, position=(SOURCE_X, 0.0, z), wavelength_nm=532.0, waist_mm=WAIST_MM)

    def paths(a: str, b: str) -> list:
        return [{"key": "path_id_a", "value": a}, {"key": "path_id_b", "value": b}]

    document = ttl_project(((532.0, 0.001),))
    document["components"] += [side("UP", 15.0), side("DOWN", -15.0)]
    document["targets"] = [
        {"target_id": "beside", "observation_id": "P", "metric": "centroid_separation_mm", "maximum": 40.0, "parameters": paths("UP#0", "DOWN#0")},
        {"target_id": "through", "observation_id": "P", "metric": "centroid_separation_mm", "maximum": 40.0, "parameters": paths("SRC#0", "UP#0")},
        {"target_id": "angle_beside", "observation_id": "P", "metric": "beam_crossing_angle_deg", "maximum": 1.0, "parameters": paths("UP#0", "DOWN#0")},
        {"target_id": "angle_through", "observation_id": "P", "metric": "beam_crossing_angle_deg", "maximum": 1.0, "parameters": paths("SRC#0", "UP#0")},
    ]
    results = {t.target_id: t for t in run_optical_project(document).result.targets}
    assert (results["beside"].status.value, results["beside"].value_basis) == ("PASS", None)
    assert results["beside"].value == pytest.approx(30.0, abs=1e-9)
    assert (results["angle_beside"].status.value, results["angle_beside"].value_basis) == ("PASS", None)
    assert results["angle_beside"].value == pytest.approx(0.0, abs=1e-9)
    for name, value in (("through", 15.0), ("angle_through", 0.0)):
        result = results[name]
        assert (result.status.value, result.assessment.value) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE"), name
        assert "METRIC_NOT_VALIDATED_FOR_ELEMENT" in result.issue_codes
        assert result.value == pytest.approx(value, abs=1e-9)
        assert result.value_basis["value"].assumed[0].at == ("ML/lens",)


# ------------------------------------------------------------------------------------------------ refusals and records


def test_network_export_and_wavefront_are_refused_as_for_an_ideal_lens(tmp_path) -> None:
    document = ttl_project(((532.0, 0.001),))
    with pytest.raises(NetworkExportError) as refused:
        export_linear_optical_network(document)
    text = " ".join(f"{p.code} {p.message}" for p in refused.value.problems)
    assert "PHASE_UNDEFINED_IDEAL_LENS" in text and "tabulated thin lens" in text
    run = run_optical_project(document, output_dir=tmp_path / "run")
    results = load_run(tmp_path / "run")
    path = run.result.observations[0].arrivals[0].beam.path_id
    answer = results.wavefront("P", path)
    assert "RAY_OPL_UNDEFINED" in str(answer)


def test_evidence_and_domain_are_carried_into_the_run_summary_and_view() -> None:
    run = run_optical_project(ttl_project(((532.0, 0.001),)))
    (row,) = [r for r in run.summary()["element_models"] if r["instance_id"] == "ML"]
    assert row["model_kind"] == "TABULATED_THIN_LENS" and row["fidelity"] == "IDEAL_THIN_ELEMENT"
    response = row["response"]
    assert response["evidence"] == EVIDENCE and response["status"] == "DERIVED" and response["data_kind"] == "EVIDENCED"
    assert response["engine_metrics"] == ["coherent_power_W", "power_W", "waist_distance_x_mm", "waist_distance_y_mm"]
    assert response["domain"]["wavelength_nm"] == ["490", "532", "580"]
    (element,) = [e for e in run.optical_view.elements if e.element_id == "ML"]
    assert element.geometry.model_kind == "TABULATED_THIN_LENS" and element.geometry.surfaces[0].focal_length_mm is None


def test_a_table_without_stated_limits_metrics_or_evidence_is_an_assumed_response_with_the_same_numbers() -> None:
    """The same table with and without the optional statements, used outside the stated twin's field angle and asked for
    a metric the stated twin does not list: plane power, beam and focus are equal to the last bit; the stated twin flags
    the angle and the metric, the unstated one has no limit to check; the summary says which is which."""

    targets = [{"target_id": "radius", "observation_id": "P", "metric": "beam_radius_mm", "wavelength_nm": 532.0, "maximum": 50.0},
               {"target_id": "power", "observation_id": "P", "metric": "power_W", "wavelength_nm": 532.0, "minimum": 0.0}]
    bare = ttl_model()
    for field in ("max_field_angle_deg", "supported_metrics", "evidence"):
        del bare["model"][field]

    def issue_codes(run) -> set[str]:
        return _codes(run) | {code for target in run.result.targets for code in target.issue_codes}

    def same_light(stated, unstated):
        (a,), (b,) = stated.result.observations[0].arrivals, unstated.result.observations[0].arrivals
        assert b.beam.power_W == a.beam.power_W and unstated.result.observations[0].total_power_W == stated.result.observations[0].total_power_W
        assert b.model_dump() == a.model_dump()  # position, direction, Gaussian state, polarization: every field
        assert b.beam.power_W == pytest.approx(0.001 * transmittance(532.0), rel=1e-12)  # the table's own value
        return b

    # a field angle of 1 deg, beyond a stated 0.5 deg
    off = -SOURCE_X * math.tan(math.radians(1.0))
    tilted = laser(((532.0, 0.001),), position=(SOURCE_X, 0.0, off), quaternion=_aim(math.atan2(off, -SOURCE_X)))
    stated = run_optical_project(ttl_project(src=tilted, model=ttl_model(max_field_angle_deg="0.5")))
    unstated = run_optical_project(ttl_project(src=tilted, model=bare))
    assert "MODEL_FIELD_ANGLE_EXCEEDED" in issue_codes(stated) and "MODEL_FIELD_ANGLE_EXCEEDED" not in issue_codes(unstated)
    same_light(stated, unstated)
    # a metric the stated list does not hold
    stated = run_optical_project(ttl_project(((532.0, 0.001),), targets=targets))
    unstated = run_optical_project(ttl_project(((532.0, 0.001),), targets=targets, model=bare))
    assert "METRIC_NOT_VALIDATED_FOR_ELEMENT" in issue_codes(stated) and "METRIC_NOT_VALIDATED_FOR_ELEMENT" not in issue_codes(unstated)
    assert not unstated.result.issues
    beam = same_light(stated, unstated).beam
    results = {t.target_id: t for t in unstated.result.targets}
    assert results["radius"].status.value == "PASS" and results["power"].status.value == "PASS"
    assert results["radius"].value == pytest.approx(max(beam.gaussian.beam_radius_x_mm, beam.gaussian.beam_radius_y_mm), rel=1e-12)
    expected_z, expected_w0 = waist_after_lens(532.0)  # the independent Gaussian ABCD of this file
    assert PLANE_X - beam.gaussian.waist_distance_x_mm == pytest.approx(expected_z, rel=1e-9)
    assert unstated.status.value == "PASS"
    (row,) = [r for r in unstated.summary()["element_models"] if r["instance_id"] == "ML"]
    response = row["response"]
    assert response["data_kind"] == "ASSUMED"
    assert (response["evidence"], response["supported_metrics"], response["engine_metrics"]) == (None, None, None)
    assert response["domain"]["max_field_angle_deg"] is None and response["domain"]["wavelength_nm"] == ["490", "532", "580"]


# ------------------------------------------------------------------------------------------------ low-level contract


def test_engine_interaction_contract_and_interpolation() -> None:
    interaction = TabulatedThinLensInteraction(wavelength_nm=(500.0, 600.0), focal_length_mm=(100.0, 200.0), transmittance=(0.5, 1.0),
                                               focal_interpolation="linear_in_optical_power", transmittance_interpolation="linear")
    table = ThinLensTable(interaction.wavelength_nm, interaction.focal_length_mm, interaction.transmittance,
                          interaction.focal_interpolation, interaction.transmittance_interpolation, interaction.point_tolerance)
    f, t, inside = response_at(table, 550.0)
    assert inside and f == 1.0 / (0.01 + (0.005 - 0.01) * 0.5) and t == 0.75
    assert response_at(table, 500.0) == (100.0, 0.5, True) and response_at(table, 600.0) == (200.0, 1.0, True)
    assert response_at(table, 650.0) == (200.0, 1.0, False) and response_at(table, 450.0) == (100.0, 0.5, False)
    with pytest.raises(ValidationError, match="one sign"):
        TabulatedThinLensInteraction(wavelength_nm=(500.0, 600.0), focal_length_mm=(100.0, -200.0), transmittance=(0.5, 1.0),
                                     focal_interpolation="linear_in_optical_power", transmittance_interpolation="linear")
    with pytest.raises(ValidationError, match="strictly increasing"):
        TabulatedThinLensInteraction(wavelength_nm=(600.0, 500.0), focal_length_mm=(100.0, 200.0), transmittance=(0.5, 1.0),
                                     focal_interpolation="points_only", transmittance_interpolation="points_only")
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        TabulatedThinLensInteraction(wavelength_nm=(500.0,), focal_length_mm=(100.0,), transmittance=(0.0,),
                                     focal_interpolation="points_only", transmittance_interpolation="points_only")


@pytest.mark.parametrize("wavelength_nm", [510.0, 515.0, 520.0])
def test_a_beam_exactly_at_the_validated_radius_is_in_scope_at_every_wavelength(wavelength_nm: float) -> None:
    """A 30 um waist at the element and max_beam_radius_mm "0.03": the radius computed at the element is 0.03 mm to the
    last bits (0.030000000000000002 at 515 nm), which is the validated radius, not beyond it."""
    at_lens = laser(((wavelength_nm, 0.001),), position=(-1.0, 0.0, 0.0), waist_mm=0.03, waist_position_mm=1.0)
    exact = run_optical_project(ttl_project(src=at_lens, model=ttl_model(max_beam_radius_mm="0.03")))
    assert "MODEL_BEAM_SIZE_EXCEEDED" not in _codes(exact)
    beyond = run_optical_project(ttl_project(src=at_lens, model=ttl_model(max_beam_radius_mm="0.02999")))
    assert "MODEL_BEAM_SIZE_EXCEEDED" in _codes(beyond)


def test_a_chief_ray_exactly_at_the_validated_field_angle_is_in_scope() -> None:
    off = 26.25
    angle = math.degrees(math.atan2(off, -SOURCE_X))
    tilted = laser(((532.0, 0.001),), position=(SOURCE_X, 0.0, off), quaternion=_aim(math.radians(angle)))
    assert "MODEL_FIELD_ANGLE_EXCEEDED" not in _codes(run_optical_project(ttl_project(src=tilted, model=ttl_model(max_field_angle_deg=f"{angle:.15g}"))))
    assert "MODEL_FIELD_ANGLE_EXCEEDED" in _codes(run_optical_project(ttl_project(src=tilted, model=ttl_model(max_field_angle_deg=f"{angle * (1 - 1e-6):.15g}"))))
