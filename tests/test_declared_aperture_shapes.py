"""Project-declared aperture shapes, holes and lens arrays, compiled and traced, against independent references.

* a shaped aperture-stop opening (``ApertureStopModelV1.opening_shape``): an ellipse, a D shape and a triangle pass the
  Gaussian power over the shape (SciPy quadrature of the beam over the shape written out here);
* a hole through a surface-sequence window (``SurfaceHoleV1``): a beam through the hole crosses no surface (no Fresnel
  loss: its full power), the same beam on the window loses the two bare Fresnel reflections;
* a lens array (``CompoundMemberV1.array``): every cell is a member of the unit at the lattice offset; beams centred on
  three cells of an array of ideal thin lenses focus on the pitch grid at the focal plane, and one cell passes the beam
  like the single lens.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import integrate, special

from os_optics.local import lower_project, run_optical_project

from declared_helpers import compound_model, cylinder, declared_component, nbk7, sequence, stop_model, surface, thin_lens_model, with_models
from local_helpers import ALONG_X, arrivals, project, rayleigh_range, source

from os_optics._vendor.shared_contracts.optical_design.optical_model import ApertureStopModelV1, CompoundMemberV1, MemberArrayV1

WAVELENGTH_NM = 632.8
P_W = 0.001


def _beam_radius(waist_mm: float, distance_mm: float) -> float:
    return waist_mm * math.sqrt(1.0 + (distance_mm / rayleigh_range(waist_mm, WAVELENGTH_NM)) ** 2)


def _stop_run(stop: dict, *, waist_mm: float = 1.0):
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=waist_mm),
                        declared_component("A", "aperture", stop["model_id"])],
                       [{"observation_id": "after", "pose": {"position_mm": [20.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    return run_optical_project(with_models(document, [stop]))


def _chord_fraction(w: float, x_lo: float, x_hi: float, y_half) -> float:
    """``(2 / (pi w^2)) int int exp(-2 (x^2 + y^2) / w^2)`` over ``x_lo <= x <= x_hi``, ``|y| <= y_half(x)`` (the inner
    integral in closed form)."""

    def inner(x: float) -> float:
        return w * math.sqrt(math.pi / 2.0) * special.erf(math.sqrt(2.0) * y_half(x) / w) * math.exp(-2.0 * x * x / (w * w))

    value, _ = integrate.quad(inner, x_lo, x_hi, epsabs=1e-15, epsrel=1e-13, limit=200)
    return value * 2.0 / (math.pi * w * w)


def test_an_elliptical_opening_passes_the_power_over_the_ellipse():
    run = _stop_run(stop_model(opening_half_widths_mm=["0.6", "1.5"], opening_shape={"kind": "ELLIPSE"}, edge_radius_mm="12.7"))
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius(1.0, 50.0)
    expected = _chord_fraction(w, -0.6, 0.6, lambda x: 1.5 * math.sqrt(max(1.0 - (x / 0.6) ** 2, 0.0)))
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-8)


@pytest.mark.parametrize(("flat", "side"), [("0.3", "-Y"), ("0.05", "+X")])
def test_a_d_shaped_opening_passes_the_power_over_the_d(flat, side):
    run = _stop_run(stop_model(opening_radius_mm="1", opening_shape={"kind": "D_SHAPE", "flat_mm": flat, "flat_side": side},
                               edge_radius_mm="12.7"))
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius(1.0, 50.0)
    expected = _chord_fraction(w, -1.0, float(flat), lambda x: math.sqrt(max(1.0 - x * x, 0.0)))  # (a round beam: any side)
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-8)


def test_a_triangular_opening_passes_the_power_over_the_triangle():
    corners = (("-0.9", "-0.6"), ("1", "-0.5"), ("0", "1.1"))
    run = _stop_run(stop_model(opening_radius_mm="1.2", opening_shape={"kind": "POLYGON", "vertices_mm": [list(c) for c in corners]},
                               edge_radius_mm="12.7"))
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius(1.0, 50.0)
    (x0, y0), (x1, y1), (x2, y2) = [(float(x), float(y)) for x, y in corners]

    def beam_at(t: float, q: float) -> float:
        x = x0 + q * (x1 - x0) + t * (x2 - x0)
        y = y0 + q * (y1 - y0) + t * (y2 - y0)
        return math.exp(-2.0 * (x * x + y * y) / (w * w))

    integral, _ = integrate.dblquad(beam_at, 0.0, 1.0, 0.0, lambda q: 1.0 - q, epsabs=1e-14, epsrel=1e-12)
    expected = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * integral * 2.0 / (math.pi * w * w)
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-8)


def test_a_shaped_opening_states_its_shape_in_the_view():
    from os_optics.optical_view.builder import declared_body
    from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2

    stop = stop_model(opening_half_widths_mm=["0.6", "1.5"], opening_shape={"kind": "ELLIPSE"}, edge_radius_mm="12.7")
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM), declared_component("A", "aperture", "stop")], [])
    lowered = lower_project(with_models(document, [stop]))
    (element,) = lowered.elements
    body = declared_body(element, ProjectDocumentV2.model_validate(with_models(document, [stop])))
    (plane,) = body.surfaces
    ring = np.array(plane.clear_outline_mm)
    assert ring.shape[0] >= 16 and np.allclose((ring[:, 0] / 0.6) ** 2 + (ring[:, 1] / 1.5) ** 2, 1.0, atol=1e-8)


def test_shaped_openings_are_checked_against_their_circle_or_rectangle():
    with pytest.raises(ValidationError, match="ELLIPSE is stated by half widths"):
        ApertureStopModelV1.model_validate({"opening_radius_mm": "1", "opening_shape": {"kind": "ELLIPSE"}, "edge_radius_mm": "5",
                                            "status": "SPECIFIED"})
    with pytest.raises(ValidationError, match="outside the stated rectangle"):
        ApertureStopModelV1.model_validate({"opening_half_widths_mm": ["1", "1"], "edge_radius_mm": "5", "status": "SPECIFIED",
                                            "opening_shape": {"kind": "POLYGON", "vertices_mm": [["0", "0"], ["2", "0"], ["0", "1"]]}})
    with pytest.raises(ValidationError, match="no central obstruction"):
        ApertureStopModelV1.model_validate({"opening_radius_mm": "1", "obstruction_radius_mm": "0.2", "edge_radius_mm": "5", "status": "SPECIFIED",
                                            "opening_shape": {"kind": "D_SHAPE", "flat_mm": "0.5", "flat_side": "+X"}})
    with pytest.raises(ValidationError, match="counter-clockwise"):
        ApertureStopModelV1.model_validate({"opening_radius_mm": "1", "edge_radius_mm": "5", "status": "SPECIFIED",
                                            "opening_shape": {"kind": "POLYGON", "vertices_mm": [["0", "0"], ["0", "0.5"], ["0.5", "0"]]}})


# --------------------------------------------------------------------------------------------------- holes


def _window(holes: list[dict] | None) -> dict:
    extra = {"holes": holes} if holes is not None else {}
    surfaces = [surface(1, "0", clear="9", **extra), surface(2, "3", clear="9", **extra)]
    return {"model_id": "window", "model": sequence(surfaces, ["N-BK7"])}


def _window_run(holes: list[dict] | None, beam_y_mm: float):
    document = project([source(position=(-50.0, beam_y_mm, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.3),
                        declared_component("W", "window", "window")],
                       [{"observation_id": "after", "pose": {"position_mm": [30.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    return run_optical_project(with_models(document, [_window(holes)], [nbk7()]))


def test_a_beam_through_a_hole_crosses_no_surface():
    """A 2 mm hole 4 mm off the axis through both faces of a window: a 0.3 mm beam through it keeps its power, the same
    beam beside the hole loses the two bare Fresnel reflections ((1 - R)^2 at normal incidence, R = ((n - 1) / (n + 1))^2)."""

    hole = [{"center_mm": ["4", "0"], "radius_mm": "2"}]  # local X = design +Y (declared_component's frame)
    through = _window_run(hole, 4.0)
    beside = _window_run(hole, -4.0)
    (a,) = arrivals(through.result, "after")
    (b,) = arrivals(beside.result, "after")
    b1, b2, b3 = 1.03961212, 0.231792344, 1.01046945
    c1, c2, c3 = 0.00600069867, 0.0200179144, 103.560653
    lam2 = (WAVELENGTH_NM / 1000.0) ** 2
    n = math.sqrt(1.0 + b1 * lam2 / (lam2 - c1) + b2 * lam2 / (lam2 - c2) + b3 * lam2 / (lam2 - c3))
    r = ((n - 1.0) / (n + 1.0)) ** 2
    assert a.beam.power_W == pytest.approx(P_W, rel=1e-12)
    assert b.beam.power_W == pytest.approx(P_W * (1.0 - r) ** 2, rel=1e-9)
    # the hole is bored through the glass: the beam in it is in air, not inside the element's body
    assert [i.code for i in through.model_scope_issues] == [] and through.summary()["status"] == "PASS"


def test_a_hole_must_lie_inside_the_clear_aperture():
    with pytest.raises(ValueError, match="does not lie inside"):
        run_optical_project(with_models(project([declared_component("W", "window", "window")], []),
                                        [_window([{"center_mm": ["8.5", "0"], "radius_mm": "1"}])], [nbk7()]))


# --------------------------------------------------------------------------------------------------- arrays


PITCH_MM, FOCAL_MM = 2.5, 50.0


def _array_unit(layout: str = "RECTANGULAR", counts=(3, 1), pitch=(str(PITCH_MM), str(PITCH_MM))) -> dict:
    unit = compound_model("mla", [("L", "lenslet", (0, 0, 0), (1, 0, 0, 0))], cylinder("20", "-1", "1"))
    unit["model"]["members"][0]["array"] = {"layout": layout, "pitch_mm": list(pitch), "counts": list(counts)}
    return unit


def _array_project(beams_y: list[float], unit: dict) -> dict:
    sources = [source(f"S{i}", position=(-50.0, y, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.3) for i, y in enumerate(beams_y)]
    document = project([*sources, declared_component("MLA", "lens", unit["model_id"])],
                       [{"observation_id": "focus", "pose": {"position_mm": [FOCAL_MM, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    return with_models(document, [thin_lens_model("lenslet", f"{FOCAL_MM:g}", clear="1", edge="1.2"), unit])


def test_an_array_member_is_one_member_per_cell_at_its_lattice_offset():
    lowered = lower_project(_array_project([0.0], _array_unit("HEXAGONAL", counts=(3, 2), pitch=("2", "1.732"))))
    (element,) = lowered.elements
    assert [m.member_id for m in element.members] == ["L_0_0", "L_0_1", "L_0_2", "L_1_0", "L_1_1", "L_1_2"]
    positions = [tuple(m.member_pose.position_mm) for m in element.members]
    # rows 0.866 below / above the centre, alternately shifted by -/+ pitch / 4
    assert positions == pytest.approx([(-2.5, -0.866, 0.0), (-0.5, -0.866, 0.0), (1.5, -0.866, 0.0),
                                       (-1.5, 0.866, 0.0), (0.5, 0.866, 0.0), (2.5, 0.866, 0.0)])
    (component,) = [c for c in lowered.request.project.components if c.instance_id == "MLA"]
    assert [s.surface_id for s in component.surfaces][:2] == ["L_0_0-lens", "L_0_1-lens"]


def test_each_lenslet_focuses_its_beam_on_the_pitch_grid():
    """Parallel beams centred on three lenslets cross the focal plane at the lenslets' centres (an ideal thin lens sends
    a beam along its axis through its own focus): the spots lie on the pitch grid."""

    beams = [-PITCH_MM, 0.0, PITCH_MM]
    run = run_optical_project(_array_project(beams, _array_unit()))
    for i, y in enumerate(beams):
        (arrival,) = arrivals(run.result, "focus", f"S{i}")
        assert arrival.beam.position_mm[1] == pytest.approx(y, abs=1e-9)
        assert arrival.beam.power_W == pytest.approx(P_W, rel=1e-9)


def test_one_cell_passes_a_beam_like_the_single_lens():
    single = compound_model("one", [("L", "lenslet", (0, 0, 0), (1, 0, 0, 0))], cylinder("20", "-1", "1"))
    beam = [0.15]
    (in_array,) = arrivals(run_optical_project(_array_project(beam, _array_unit())).result, "focus")
    (alone,) = arrivals(run_optical_project(_array_project(beam, single)).result, "focus")
    assert in_array.beam.power_W == pytest.approx(alone.beam.power_W, rel=1e-12)
    assert in_array.beam.position_mm == pytest.approx(alone.beam.position_mm, abs=1e-12)
    assert in_array.beam.gaussian.beam_radius_x_mm == pytest.approx(alone.beam.gaussian.beam_radius_x_mm, rel=1e-12)


def test_array_contract_limits():
    member = {"member_id": "L", "model_id": "m", "pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}}
    with pytest.raises(ValidationError, match="at most 256 cells"):
        CompoundMemberV1.model_validate({**member, "array": {"layout": "RECTANGULAR", "pitch_mm": ["1", "1"], "counts": [17, 16]}})
    with pytest.raises(ValidationError, match="at least two cells"):
        MemberArrayV1.model_validate({"layout": "RECTANGULAR", "pitch_mm": ["1", "1"], "counts": [1, 1]})
    with pytest.raises(ValidationError, match="at most 8 characters"):
        CompoundMemberV1.model_validate({**copy.deepcopy(member), "member_id": "LENSLETS1",
                                         "array": {"layout": "RECTANGULAR", "pitch_mm": ["1", "1"], "counts": [2, 1]}})
