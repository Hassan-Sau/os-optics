"""Project-declared optical models traced by the Engine, against independent references computed here with plain
``math`` / NumPy from the declared numbers -- never with the runtime's own compile, paraxial or transform helpers:

* an air-spaced multi-element objective (one ``SURFACE_SEQUENCE``: a cemented doublet, an air gap, a plano-convex lens):
  effective / back focal distance from an independent y-nu trace, and the Gaussian focus from q propagation;
* the same objective as a ``COMPOUND`` unit of two members: the flattened Engine surfaces are the members' surfaces moved
  by the member poses (checked with an independent quaternion rotation), and it traces as the single model does;
* aperture stops: the power a centred circular stop, a slit and an off-axis annulus pass of a Gaussian beam (closed forms
  and a direct numerical integral), light beyond the plate passing by;
* an ideal thin lens: a ray parallel to the axis crosses it at the focal length.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.local import LocalValidationStatusV1, lower_project, run_optical_project

from declared_helpers import (
    FLINT_INDEX,
    OBJECTIVE_MEDIA,
    OBJECTIVE_RADII,
    OBJECTIVE_VERTICES,
    compound_model,
    cylinder,
    declared_component,
    doublet_model,
    flint,
    nbk7,
    objective_model,
    plano_convex_model,
    stop_model,
    thin_lens_model,
    with_models,
)
from local_helpers import ALONG_X, arrivals, plane, project, q_through, rayleigh_range, source

WAVELENGTH_NM = 632.8
W0_MM = 1.0
#: Schott N-BK7 Sellmeier coefficients (B_i, C_i in um^2).
NBK7_B = (1.03961212, 0.231792344, 1.01046945)
NBK7_C = (0.00600069867, 0.0200179144, 103.560653)


def _n(material: str, wavelength_nm: float = WAVELENGTH_NM) -> float:
    """Independent refractive index: the N-BK7 Sellmeier formula written out here, the flint's constant index."""

    if material == "AIR":
        return 1.0
    if material == "FLINT":
        return float(FLINT_INDEX)
    b, c = NBK7_B, NBK7_C
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def _y_nu(radii, vertices, media, wavelength_nm: float = WAVELENGTH_NM) -> tuple[float, float]:
    """(EFL, BFD) by tracing a marginal ray (y, n u) = (1, 0) through the surfaces: refraction n'u' = nu - y (n' - n) / R,
    transfer y' = y + t (n u) / n."""

    media = ["AIR", *media, "AIR"]
    y, nu = 1.0, 0.0
    for k, (radius, z) in enumerate(zip(radii, vertices)):
        if k:
            y += (float(z) - float(vertices[k - 1])) * nu / _n(media[k], wavelength_nm)
        if radius is not None:
            nu -= y * (_n(media[k + 1], wavelength_nm) - _n(media[k], wavelength_nm)) / float(radius)
    return -1.0 / nu, -y / nu


def _matrix(radii, vertices, media, wavelength_nm: float = WAVELENGTH_NM) -> np.ndarray:
    media = ["AIR", *media, "AIR"]
    matrix = np.eye(2)
    for k, (radius, z) in enumerate(zip(radii, vertices)):
        if k:
            matrix = np.array([[1.0, (float(z) - float(vertices[k - 1])) / _n(media[k], wavelength_nm)], [0.0, 1.0]]) @ matrix
        if radius is not None:
            matrix = np.array([[1.0, 0.0], [-(_n(media[k + 1], wavelength_nm) - _n(media[k], wavelength_nm)) / float(radius), 1.0]]) @ matrix
    return matrix


def _objective_project(components: list[dict], models: list[dict], screen_x: float, **fields) -> dict:
    document = project([source(position=(-200.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=W0_MM), *components],
                       [plane("focus", (screen_x, 0.0, 0.0), radius_mm=20.0)])
    return with_models(document, models, [nbk7(), flint()], **fields)


def _expected_focus() -> float:
    """Distance of the Gaussian waist behind the objective's last vertex (q propagation through the independent matrix)."""

    q_out = q_through(_matrix(OBJECTIVE_RADII, OBJECTIVE_VERTICES, OBJECTIVE_MEDIA), complex(200.0, rayleigh_range(W0_MM, WAVELENGTH_NM)))
    return -q_out.real


def _assert_objective_auxiliary_limit(run):
    # All main light reaches the focus; only the independent faint-light search
    # reaches the unchanged default step budget: an auxiliary analysis, recorded per phase, with a notice.
    assert run.status is LocalValidationStatusV1.PASS and run.model_scope_issues == ()
    execution = run.result.outcome.execution
    assert execution.outcome.value == "COMPLETED" and execution.complete
    assert execution.budgets.exhausted_by_phase == {"untraced_light": ("max_trace_steps",)}
    (budget,) = [i for i in run.result.issues if i.code.value == "MAX_PATHS"]
    assert budget.status.value == "NOT_APPLICABLE" and budget.numerical_context["phase"] == "untraced_light"
    assert execution.budgets.untraced_power_W is None
    # (each lit plane: the traced value is the lower end of its power, the upper end open; the entry names the budget)
    assert all(o.quantity_state.value == "BOUNDED" and not o.power_complete and o.power_upper_bound_W is None
               for o in run.result.observations)
    for o in run.result.observations:
        entry = o.value_basis["total_power_W"]
        assert entry.kind.value == "LOWER_BOUND" and ("BUDGET", "MAX_PATHS") in {(part.part.value, part.code) for part in entry.omitted}


# --------------------------------------------------------------------------------------- air-spaced objective


def test_an_air_spaced_objective_is_one_model_with_its_independent_focal_lengths():
    from os_optics.local import paraxial_matrix

    document = _objective_project([declared_component("OBJ", "lens", "objective")], [objective_model()], 100.0)
    lowered = lower_project(document)
    (element,) = lowered.elements
    assert [type(s).__name__ for s in element.surfaces] == ["SphereSurface"] * 4 + ["PlaneSurface"]
    assert [m.material_id for m in lowered.request.project.materials] == ["air", "project:FLINT", "project:N-BK7"]  # (air: the environment's)
    matrix = paraxial_matrix(element, WAVELENGTH_NM)
    efl, bfd = _y_nu(OBJECTIVE_RADII, OBJECTIVE_VERTICES, OBJECTIVE_MEDIA)
    assert -1.0 / matrix[1][0] == pytest.approx(efl, rel=1e-12)
    assert -matrix[0][0] / matrix[1][0] == pytest.approx(bfd, rel=1e-12)
    assert 30.0 < bfd < efl  # a positive objective whose back focus lies inside its focal length


def test_the_objective_focuses_where_independent_q_propagation_puts_the_waist():
    screen_x = float(OBJECTIVE_VERTICES[-1]) + _expected_focus()
    run = run_optical_project(_objective_project([declared_component("OBJ", "lens", "objective")], [objective_model()], screen_x))
    _assert_objective_auxiliary_limit(run)
    (arrival,) = arrivals(run.result, "focus")
    assert arrival.beam.gaussian.waist_distance_x_mm == pytest.approx(0.0, abs=1e-6)
    # every uncoated interface transmits 1 - ((n2 - n1) / (n2 + n1))^2 of the power at the chief ray's normal incidence
    media = ["AIR", *OBJECTIVE_MEDIA, "AIR"]
    transmitted = math.prod(1.0 - ((_n(b) - _n(a)) / (_n(b) + _n(a))) ** 2 for a, b in zip(media, media[1:]))
    assert arrival.beam.power_W == pytest.approx(0.001 * transmitted, rel=1e-12)


# ------------------------------------------------------------------------------------------- compound unit


def _rotate(q, v) -> np.ndarray:
    """Independent active rotation of ``v`` by the unit quaternion ``q`` (w, x, y, z): q v q*."""

    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


UNIT_ENVELOPE = cylinder("22", "-1", "30")


def _unit(members, model_id: str = "unit") -> dict:
    return compound_model(model_id, members, UNIT_ENVELOPE,
                          ports=[{"port_id": "IN", "role": "INPUT", "point_mm": ["0", "0", "-1"], "direction": ["0", "0", "1"], "status": "SPECIFIED"}])


def test_a_compound_unit_is_one_engine_component_with_every_member_surface_placed():
    turned = (0.0, 1.0, 0.0, 0.0)  # 180 deg about X: the plano-convex lens's plane side first
    document = _objective_project([declared_component("OBJ", "lens", "unit")],
                                  [doublet_model(), plano_convex_model(), _unit([("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)),
                                                                                 ("L2", "pcx", (0, 0, 25.0), turned)])], 100.0)
    lowered = lower_project(document)
    (component,) = [c for c in lowered.request.project.components if c.instance_id == "OBJ"]
    assert [s.surface_id for s in component.surfaces] == ["L1-S1", "L1-S2", "L1-S3", "L2-S1", "L2-S2"]
    (element,) = lowered.elements
    assert [m.member_id for m in element.members] == ["L1", "L2"]
    # every unit-frame surface is its member-frame surface moved by the member pose (independent rotation)
    for member, position, q in ((element.members[0], (0, 0, 0), (1, 0, 0, 0)), (element.members[1], (0, 0, 25.0), turned)):
        for local, placed in zip(member.surfaces, (s for s in component.surfaces if s.surface_id.startswith(member.member_id + "-"))):
            if local.geometry == "sphere":
                assert np.allclose(placed.center_mm, _rotate(q, local.center_mm) + position, atol=1e-12)
                assert np.allclose(placed.cap_axis, _rotate(q, local.cap_axis), atol=1e-15)
                assert placed.radius_mm == local.radius_mm and placed.medium_positive == local.medium_positive
            else:
                assert np.allclose(placed.origin_mm, _rotate(q, local.origin_mm) + position, atol=1e-12)
                assert np.allclose(placed.normal, _rotate(q, local.normal), atol=1e-15)
    # the turned plano-convex lens: its plane first, at z = 25 - 4.5, its sphere's centre on the other side of its vertex
    plane_surface = component.surfaces[4]
    assert plane_surface.origin_mm == pytest.approx((0.0, 0.0, 20.5)) and plane_surface.normal == pytest.approx((0.0, 0.0, -1.0))


def test_a_unit_of_the_objective_members_traces_as_the_one_model():
    """The doublet and the plano-convex lens as two members (identity rotations, the lens 20.5 mm behind) are the very
    surfaces of the single objective model: the same trace, to rounding."""

    screen_x = float(OBJECTIVE_VERTICES[-1]) + _expected_focus()
    single = run_optical_project(_objective_project([declared_component("OBJ", "lens", "objective")], [objective_model()], screen_x))
    unit_doc = _objective_project([declared_component("OBJ", "lens", "unit")],
                                  [doublet_model(), plano_convex_model(), _unit([("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)),
                                                                                 ("L2", "pcx", (0, 0, 20.5), (1, 0, 0, 0))])], screen_x)
    unit = run_optical_project(unit_doc)
    a, b = arrivals(single.result, "focus")[0], arrivals(unit.result, "focus")[0]
    assert b.beam.power_W == pytest.approx(a.beam.power_W, rel=1e-12)
    assert b.beam.gaussian.waist_distance_x_mm == pytest.approx(a.beam.gaussian.waist_distance_x_mm, abs=1e-9)
    assert b.beam.gaussian.beam_radius_x_mm == pytest.approx(a.beam.gaussian.beam_radius_x_mm, rel=1e-9)
    _assert_objective_auxiliary_limit(single)
    _assert_objective_auxiliary_limit(unit)
    rows = unit.summary()["element_models"]
    assert rows[0]["optical_model_id"] == "unit" and [m["member_id"] for m in rows[0]["members"]] == ["L1", "L2"]


def test_a_member_issue_names_its_member():
    narrow = doublet_model()
    narrow["model"]["valid_band_nm"] = {"min_nm": "700", "max_nm": "900", "status": "SPECIFIED"}
    document = _objective_project([declared_component("OBJ", "lens", "unit")],
                                  [narrow, plano_convex_model(), _unit([("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)),
                                                                        ("L2", "pcx", (0, 0, 20.5), (1, 0, 0, 0))])], 100.0)
    run = run_optical_project(document)
    assert run.status is LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    (issue,) = run.model_scope_issues
    assert (issue.instance_id, issue.member_id, issue.code) == ("OBJ", "L1", "MODEL_WAVELENGTH_OUTSIDE_BAND")
    assert "OBJ member L1 (doublet)" in issue.message and "700-900 nm" in issue.message
    assert issue.as_dict()["member_id"] == "L1"


# --------------------------------------------------------------------------------------------- aperture stops


def _stop_run(stop: dict, *, beam_y_mm: float = 0.0, waist_mm: float = 1.0):
    document = project([source(position=(-50.0, beam_y_mm, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=waist_mm),
                        declared_component("A", "aperture", stop["model_id"])],
                       [{"observation_id": "after", "pose": {"position_mm": [20.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    return run_optical_project(with_models(document, [stop]))


def _beam_radius_at_stop(waist_mm: float) -> float:
    zr = rayleigh_range(waist_mm, WAVELENGTH_NM)
    return waist_mm * math.sqrt(1.0 + (50.0 / zr) ** 2)


def test_a_circular_stop_passes_the_gaussian_fraction_inside_its_opening():
    """P/P0 = 1 - exp(-2 a^2 / w^2) for a centred circular opening of radius a and a beam of 1/e^2 radius w there."""

    radius = 0.8
    run = _stop_run(stop_model(opening_radius_mm="0.8", edge_radius_mm="12.7"))
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius_at_stop(1.0)
    assert arrival.beam.power_W == pytest.approx(0.001 * (1.0 - math.exp(-2.0 * radius**2 / w**2)), rel=1e-6)


def test_a_slit_passes_the_erf_product():
    """A centred rectangular opening of half widths (a, b): P/P0 = erf(sqrt(2) a / w) erf(sqrt(2) b / w)."""

    run = _stop_run(stop_model(opening_half_widths_mm=["0.1", "3"], edge_half_widths_mm=["10", "10"]))
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius_at_stop(1.0)
    expected = math.erf(math.sqrt(2.0) * 0.1 / w) * math.erf(math.sqrt(2.0) * 3.0 / w)
    assert arrival.beam.power_W == pytest.approx(0.001 * expected, rel=1e-6)


def test_an_annular_opening_passes_the_integral_over_the_annulus():
    """A beam whose chief ray passes the annulus between a 0.5 mm central obstruction and a 2 mm opening: the transmitted
    fraction is the Gaussian integral over the annulus (a direct polar quadrature here)."""

    offset = 1.2  # the chief ray crosses the annulus 1.2 mm off the axis (design +Y = stop local +X)
    run = _stop_run(stop_model(opening_radius_mm="2", obstruction_radius_mm="0.5", edge_radius_mm="12.7"), beam_y_mm=offset, waist_mm=0.4)
    (arrival,) = arrivals(run.result, "after")
    w = _beam_radius_at_stop(0.4)
    radii = np.linspace(0.5, 2.0, 3001)
    angles = np.linspace(0.0, 2.0 * math.pi, 4001)
    r, a = np.meshgrid(radii, angles, indexing="ij")
    x, y = r * np.cos(a), r * np.sin(a)
    density = 2.0 / (math.pi * w * w) * np.exp(-2.0 * ((x - offset) ** 2 + y**2) / (w * w)) * r
    fraction = np.trapezoid(np.trapezoid(density, angles, axis=1), radii)
    assert arrival.beam.power_W == pytest.approx(0.001 * fraction, rel=1e-4)


def test_light_beyond_the_plate_passes_by_and_light_on_it_is_blocked():
    plate = stop_model(opening_radius_mm="1", edge_radius_mm="3")
    beyond = _stop_run(plate, beam_y_mm=5.0, waist_mm=0.2)
    assert arrivals(beyond.result, "after")[0].beam.power_W == pytest.approx(0.001, rel=1e-12)
    blocked = _stop_run(plate, beam_y_mm=2.0, waist_mm=0.2)
    assert arrivals(blocked.result, "after") == []


def test_a_decentred_opening_is_centred_where_it_is_declared():
    """A 0.8 mm opening centred 1.5 mm off axis passes a beam aimed through its centre like the centred stop does."""

    run = _stop_run(stop_model(opening_radius_mm="0.8", opening_center_mm=["1.5", "0"], edge_radius_mm="12.7"), beam_y_mm=1.5)
    w = _beam_radius_at_stop(1.0)
    assert arrivals(run.result, "after")[0].beam.power_W == pytest.approx(0.001 * (1.0 - math.exp(-2.0 * 0.64 / w**2)), rel=1e-6)


# ------------------------------------------------------------------------------------------ ideal thin lens


@pytest.mark.parametrize("focal_length", [80.0, 125.0])
def test_a_ray_parallel_to_the_axis_crosses_it_at_the_focal_length(focal_length):
    document = project([source(position=(-50.0, 2.0, 0.0), wavelength_nm=WAVELENGTH_NM, waist_mm=0.3),
                        declared_component("T", "lens", "thin")],
                       [{"observation_id": "f", "pose": {"position_mm": [focal_length, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    run = run_optical_project(with_models(document, [thin_lens_model(focal_length_mm=f"{focal_length:g}")]))
    (arrival,) = arrivals(run.result, "f")
    assert np.allclose(arrival.uv_mm, (0.0, 0.0), atol=1e-12)
    assert run.summary()["element_models"][0]["fidelity"] == "IDEAL_THIN_ELEMENT"


def test_a_thin_lens_outside_its_band_is_out_of_model_scope():
    lens = thin_lens_model()
    lens["model"]["valid_band_nm"] = {"min_nm": "1000", "max_nm": "1100", "status": "SPECIFIED"}
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=WAVELENGTH_NM), declared_component("T", "lens", "thin")],
                       [plane("f", (100.0, 0.0, 0.0))])
    run = run_optical_project(with_models(document, [lens]))
    assert [(i.instance_id, i.code) for i in run.model_scope_issues] == [("T", "MODEL_WAVELENGTH_OUTSIDE_BAND")]
