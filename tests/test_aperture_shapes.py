"""Aperture regions beyond the circle and the rectangle: ellipse, D shape, simple polygon (triangles included) and holes
through an element (``optics/apertures.py``, ``optics/gaussian/aperture_regions.py``).

References are independent of the Engine's integrals:

* the Gaussian power on a region is checked against the closed forms that exist (a polygon equal to a rectangle against
  the rectangle's product of two ``erf`` intervals, an ellipse with equal semi-axes against the disc, a round beam on a
  disc of radius ``a`` about its centre: ``1 - exp(-2 a^2 / w^2)``) and otherwise against SciPy's adaptive quadrature
  of the beam over the region in the region's own coordinates (``scipy.integrate.quad`` on the closed-form chord
  integral, or ``dblquad`` on a triangle);
* ray membership against the shapes' defining inequalities;
* end to end: a triangular stop transmits the triangle's Gaussian power; a mirror with a hole lets a beam through the hole
  pass behind it with its power, and a centred beam on it reflects the power outside the hole.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import integrate, special

from os_optics import builders as B
from os_optics.contracts import (
    AbsorberInteraction,
    ApertureStopInteraction,
    CircularAperture,
    DShapedAperture,
    EllipticalAperture,
    EngineeringComponentSnapshot,
    MirrorInteraction,
    PlaneSurface,
    PolygonAperture,
    ProjectSnapshot,
    RectangularAperture,
)
from os_optics.config import NumericalConfig
from os_optics.optics import apertures
from os_optics.optics.gaussian.aperture_regions import region_integral
from os_optics.optics.gaussian.truncation import _circle_integral, _rectangle_integral, gauss_legendre_rule
from os_optics.scene.lower import lower_aperture
from os_optics.scene.model import ApertureG

from helpers import one_arrival, rayleigh_range, run

CFG = NumericalConfig()
RULE = gauss_legendre_rule(CFG.aperture_quad_sector_nodes)
X = (1.0, 0.0, 0.0)


def _s_matrix(wx: float, wy: float, angle: float) -> tuple[np.ndarray, float]:
    c, s = math.cos(angle), math.sin(angle)
    bx, by = np.array([c, s]), np.array([-s, c])
    return np.outer(bx, bx) / (wx * wx) + np.outer(by, by) / (wy * wy), 1.0 / (wx * wy) ** 2


BEAMS = [(1.0, 1.0, 0.0), (0.5, 2.0, 0.3), (0.05, 3.0, 1.1), (2.0, 2.5, 0.7)]
CENTRES = [(0.0, 0.0), (0.3, -0.2), (3.0, 1.0), (-5.0, 2.0)]


# ------------------------------------------------------------------------------------------- Gaussian power


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("centre", CENTRES)
def test_a_polygon_equal_to_a_rectangle_has_the_rectangle_power(beam, centre):
    s, det = _s_matrix(*beam)
    hu, hv = 1.3, 0.7
    rectangle = _rectangle_integral(s, det, centre, hu, hv, CFG, RULE)
    polygon = ApertureG("polygon", offset_u=centre[0], offset_v=centre[1], vertices=((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv)))
    assert region_integral(polygon, s, det, (0.0, 0.0), CFG, RULE) == pytest.approx(rectangle, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("centre", CENTRES)
def test_an_ellipse_with_equal_semi_axes_is_the_disc(beam, centre):
    s, det = _s_matrix(*beam)
    disc = _circle_integral(s, det, centre, 1.1, CFG, RULE)
    ellipse = ApertureG("ellipse", 1.1, 1.1, centre[0], centre[1], axis=(math.cos(0.4), math.sin(0.4)))
    assert region_integral(ellipse, s, det, (0.0, 0.0), CFG, RULE) == pytest.approx(disc, rel=1e-12, abs=1e-14)


def _round_beam_chord_reference(w: float, x_lo: float, x_hi: float, y_half) -> float:
    """``int_{x_lo}^{x_hi} int_{-y(x)}^{y(x)} exp(-2 (x^2 + y^2) / w^2) dy dx`` (the inner integral in closed form)."""
    root = math.sqrt(2.0) / w

    def inner(x: float) -> float:
        half = y_half(x)
        return w * math.sqrt(math.pi / 2.0) * special.erf(root * half) * math.exp(-2.0 * x * x / (w * w))

    value, _ = integrate.quad(inner, x_lo, x_hi, epsabs=1e-15, epsrel=1e-13, limit=200)
    return value


@pytest.mark.parametrize("w", [0.4, 1.0, 3.0])
@pytest.mark.parametrize("flat", [0.6, 0.0, -0.5])
def test_a_d_shape_against_the_chord_integral(w, flat):
    """A round beam centred on a D shape (radius 1, the flat at x = flat): the integral over x of the closed-form chord."""
    r = 1.0
    reference = _round_beam_chord_reference(w, -r, flat, lambda x: math.sqrt(max(r * r - x * x, 0.0)))
    s, det = _s_matrix(w, w, 0.0)
    region = ApertureG("d_shape", r, flat, 0.0, 0.0, axis=(1.0, 0.0))
    assert region_integral(region, s, det, (0.0, 0.0), CFG, RULE) == pytest.approx(reference, rel=1e-10, abs=1e-14)
    # turned: the flat's normal along +v, the beam centre off the region's centre
    turned = ApertureG("d_shape", r, flat, 0.0, 0.0, axis=(0.0, 1.0))
    assert region_integral(turned, s, det, (0.0, 0.0), CFG, RULE) == pytest.approx(reference, rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("beam", [(1.0, 1.0, 0.0), (0.4, 1.5, 0.6)])
@pytest.mark.parametrize("chief", [(0.0, 0.0), (0.4, 0.2), (2.5, -1.0)])
def test_a_triangle_against_two_dimensional_quadrature(beam, chief):
    s, det = _s_matrix(*beam)
    corners = ((-1.0, -0.5), (1.5, -0.2), (0.1, 1.2))
    region = ApertureG("polygon", offset_u=0.2, offset_v=-0.1, vertices=corners)
    (x0, y0), (x1, y1), (x2, y2) = [(x + 0.2 - chief[0], y - 0.1 - chief[1]) for x, y in corners]

    def beam_at(t: float, q: float) -> float:  # barycentric (q along the first edge, t across it), times the Jacobian
        x = x0 + q * (x1 - x0) + t * (x2 - x0)
        y = y0 + q * (y1 - y0) + t * (y2 - y0)
        return math.exp(-2.0 * (s[0, 0] * x * x + 2.0 * s[0, 1] * x * y + s[1, 1] * y * y))

    jacobian = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    reference, _ = integrate.dblquad(beam_at, 0.0, 1.0, 0.0, lambda q: 1.0 - q, epsabs=1e-14, epsrel=1e-12)
    assert region_integral(region, s, det, chief, CFG, RULE) == pytest.approx(jacobian * reference, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("hole_mm", [0.2, 1.0])
def test_a_hole_takes_out_the_power_through_it(hole_mm):
    """A round beam centred on a disc of radius ``a``: ``1 - exp(-2 a^2 / w^2)`` passes through it."""
    from os_optics.optics.gaussian.state import initial_state
    from os_optics.optics.gaussian.truncation import transmission_fraction

    w = 0.8
    gstate = initial_state(wavelength_nm=633.0, n=1.0, w0x=w, w0y=w, z_waist=0.0, m2x=1.0, m2y=1.0, bx=np.array([1.0, 0.0, 0.0]),
                           by=np.array([0.0, 1.0, 0.0]), cfg=CFG)
    hole = ApertureG("circle", hole_mm)
    holed = ApertureG("circle", 50.0, holes=(hole,))
    fraction = transmission_fraction(gstate, holed, (0.0, 0.0), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0, CFG, RULE)
    assert fraction == pytest.approx(math.exp(-2.0 * hole_mm * hole_mm / (w * w)), rel=1e-12)


# ----------------------------------------------------------------------------------------------- membership


def test_membership_follows_each_shapes_inequalities():
    rng = np.random.default_rng(7)
    u, v = rng.uniform(-2.0, 2.0, 4000), rng.uniform(-2.0, 2.0, 4000)
    c, s = math.cos(0.5), math.sin(0.5)
    ellipse = ApertureG("ellipse", 1.5, 0.6, 0.1, -0.2, axis=(c, s))
    x, y = c * (u - 0.1) + s * (v + 0.2), c * (v + 0.2) - s * (u - 0.1)
    assert (apertures.inside(ellipse, u, v, 0.0) == ((x / 1.5) ** 2 + (y / 0.6) ** 2 <= 1.0)).all()
    d_shape = ApertureG("d_shape", 1.2, 0.4, 0.0, 0.0, axis=(0.0, -1.0))
    assert (apertures.inside(d_shape, u, v, 0.0) == ((u * u + v * v <= 1.44) & (-v <= 0.4))).all()
    triangle = ApertureG("polygon", vertices=((-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)))
    in_triangle = (v >= -1.0) & (v <= 1.0 - 2.0 * np.abs(u)) & (v <= 1.0)
    assert (apertures.inside(triangle, u, v, 0.0) == in_triangle).all()
    holed = ApertureG("circle", 1.8, holes=(ApertureG("circle", 0.5, offset_u=0.6), triangle))
    expected = (u * u + v * v <= 1.8 * 1.8) & ~(((u - 0.6) ** 2 + v * v < 0.25) | in_triangle)
    inner = apertures.inside(holed, u, v, 0.0)
    near_edge = np.abs(np.sqrt((u - 0.6) ** 2 + v * v) - 0.5) < 1e-12
    assert (inner == expected)[~near_edge].all()


def test_a_non_convex_polygon_is_held_by_the_even_odd_rule():
    notch = ApertureG("polygon", vertices=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)))
    assert apertures.inside(notch, np.array([0.0, 0.0, 0.5]), np.array([-0.5, 0.5, 0.3]), 0.0).tolist() == [True, False, True]
    assert not apertures.convex(notch) and apertures.convex(ApertureG("polygon", vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))


def test_extents_and_distances():
    ellipse = ApertureG("ellipse", 2.0, 1.0, 0.5, 0.0, axis=(0.0, 1.0))  # first axis along v
    assert apertures.bounds(ellipse) == pytest.approx((-0.5, 1.5, -2.0, 2.0))
    assert apertures.spread(ellipse, 0.0, 1.0) == pytest.approx(2.0) and apertures.reach(ellipse) == 2.0
    d_shape = ApertureG("d_shape", 1.0, 0.5, 0.0, 0.0, axis=(1.0, 0.0))
    assert apertures.spread(d_shape, 1.0, 0.0) == pytest.approx(0.5) and apertures.spread(d_shape, -1.0, 0.0) == pytest.approx(1.0)
    assert apertures.spread(d_shape, 1.0, 1.0) == pytest.approx(0.5 + math.sqrt(0.75))
    square = ApertureG("polygon", vertices=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    assert apertures.miss_distance(square, 3.0, 0.0) == pytest.approx(2.0) and apertures.room(square, 0.5, 0.0) == pytest.approx(0.5)
    grown = [value for corner in apertures.dilate(square, 0.5).vertices for value in corner]
    assert grown == pytest.approx([-1.5, -1.5, 1.5, -1.5, 1.5, 1.5, -1.5, 1.5])
    holed = ApertureG("circle", 3.0, holes=(ApertureG("circle", 1.0),))
    assert apertures.miss_distance(holed, 0.25, 0.0) == pytest.approx(0.75)  # in the hole: outside the element by 0.75
    assert apertures.area(holed) == pytest.approx(8.0 * math.pi)


# ----------------------------------------------------------------------------------------------- contracts


def test_polygon_contract_refuses_crossing_clockwise_and_repeated_corners():
    with pytest.raises(ValidationError, match="edges do not cross"):
        PolygonAperture(vertices_mm=((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValidationError, match="counter-clockwise"):
        PolygonAperture(vertices_mm=((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    with pytest.raises(ValidationError, match="distinct"):
        PolygonAperture(vertices_mm=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    with pytest.raises(ValidationError):
        DShapedAperture(radius_mm=1.0, flat_distance_mm=1.0)
    PolygonAperture(vertices_mm=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))  # a triangle


def test_contract_apertures_lower_to_their_regions():
    ellipse = lower_aperture(EllipticalAperture(semi_axes_mm=(2.0, 1.0), rotation_deg=90.0, center_offset_mm=(0.5, 0.0)))
    assert (ellipse.shape, ellipse.p1, ellipse.p2, ellipse.axis) == ("ellipse", 2.0, 1.0, (0.0, 1.0))
    d_shape = lower_aperture(DShapedAperture(radius_mm=3.0, flat_distance_mm=1.0, flat_angle_deg=180.0))
    assert (d_shape.shape, d_shape.p2, d_shape.axis) == ("d_shape", 1.0, (-1.0, 0.0))
    assert lower_aperture(PolygonAperture(vertices_mm=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))).vertices == ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def test_a_hole_must_lie_inside_the_clear_aperture_and_apart():
    def plate(holes):
        surface = PlaneSurface(surface_id="m", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=5.0),
                               holes=holes, interaction=AbsorberInteraction())
        laser = B.laser("SRC", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=X)
        return ProjectSnapshot(components=(laser, EngineeringComponentSnapshot(instance_id="P", role="aperture", pose=B._pose((10.0, 0.0, 0.0), X),
                                                                               surfaces=(surface,))))
    def refusal(holes) -> str:
        result = run(plate(holes))
        (issue,) = [i for i in result.issues if i.code.value == "DEGENERATE_SURFACE"]
        return issue.message

    assert "does not lie inside" in refusal((CircularAperture(radius_mm=1.0, center_offset_mm=(4.5, 0.0)),))
    assert "not apart" in refusal((CircularAperture(radius_mm=1.0, center_offset_mm=(1.0, 0.0)), CircularAperture(radius_mm=1.0, center_offset_mm=(-0.5, 0.0))))
    run(plate((CircularAperture(radius_mm=1.0, center_offset_mm=(1.5, 0.0)), CircularAperture(radius_mm=1.0, center_offset_mm=(-1.5, 0.0)))))
    with pytest.raises(ValidationError, match="obstruction"):
        PlaneSurface(surface_id="m", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=5.0),
                     holes=(CircularAperture(radius_mm=1.0, obstruction_radius_mm=0.5),), interaction=AbsorberInteraction())


def test_a_surface_without_holes_serializes_as_before():
    surface = PlaneSurface(surface_id="m", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=RectangularAperture(half_width_mm=1.0, half_height_mm=2.0),
                           interaction=AbsorberInteraction())
    assert "holes" not in surface.model_dump(mode="json")


# ----------------------------------------------------------------------------------------------- end to end


P_W, LAM, W0 = 1e-3, 633.0, 0.8
STOP_X, OUT_X = 10.0, 30.0


def _w(z: float) -> float:
    return W0 * math.sqrt(1.0 + (z / rayleigh_range(W0, LAM)) ** 2)


def _component(instance_id: str, x: float, surface: PlaneSurface, normal=X) -> EngineeringComponentSnapshot:
    return EngineeringComponentSnapshot(instance_id=instance_id, role="aperture", pose=B._pose((x, 0.0, 0.0), normal), surfaces=(surface,))


def test_a_triangular_stop_transmits_the_triangles_gaussian_power():
    corners = ((-1.0, -0.8), (1.2, -0.6), (0.0, 1.0))
    stop = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=PolygonAperture(vertices_mm=corners),
                        aperture_u_axis=(1.0, 0.0, 0.0), mount_margin_mm=10.0, interaction=ApertureStopInteraction())
    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W0, position=(0.0, 0.0, 0.0), direction=X)
    out = B.observation_plane("OUT", position=(OUT_X, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser, _component("STOP", STOP_X, stop)), observations=(out,)))
    w = _w(STOP_X)
    (x0, y0), (x1, y1), (x2, y2) = corners

    def beam_at(t: float, q: float) -> float:
        x = x0 + q * (x1 - x0) + t * (x2 - x0)
        y = y0 + q * (y1 - y0) + t * (y2 - y0)
        return math.exp(-2.0 * (x * x + y * y) / (w * w))

    jacobian = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    integral, _ = integrate.dblquad(beam_at, 0.0, 1.0, 0.0, lambda q: 1.0 - q, epsabs=1e-14, epsrel=1e-12)
    expected = jacobian * integral * 2.0 / (math.pi * w * w)
    arrival = one_arrival(result, "OUT")
    assert arrival.beam.power_W == pytest.approx(P_W * expected, rel=1e-8)
    (line,) = result.power_budget.lines
    assert line.conserved


def _holed_mirror(hole_offset: float) -> PlaneSurface:
    """A 45-degree flat mirror (normal in the x-y plane) with a round hole of radius 2 mm, decentred along u."""
    return PlaneSurface(surface_id="m", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=12.0),
                        aperture_u_axis=(1.0, 0.0, 0.0), holes=(CircularAperture(radius_mm=2.0, center_offset_mm=(hole_offset, 0.0)),),
                        mount_margin_mm=1.0, interaction=MirrorInteraction())


def test_light_through_the_hole_of_a_mirror_passes_behind_it():
    """The beam's chief ray goes through the hole: the mirror is absent there and the beam reaches the plane behind it with
    its power (the beam, 0.8 mm, fits the 2 mm hole: its share on the rim is far below the ledger's resolution)."""
    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=W0, position=(0.0, 0.0, 0.0), direction=X)
    mirror = _component("M", STOP_X, _holed_mirror(0.0), normal=(-math.sqrt(0.5), math.sqrt(0.5), 0.0))
    behind = B.observation_plane("BEHIND", position=(OUT_X, 0.0, 0.0), normal=X)
    result = run(ProjectSnapshot(components=(laser, mirror), observations=(behind,)))
    arrival = one_arrival(result, "BEHIND")
    assert arrival.beam.power_W == pytest.approx(P_W, rel=1e-9)
    assert [p.termination for p in result.paths] == ["ESCAPED"]


def test_a_mirror_with_a_hole_reflects_the_power_outside_the_hole():
    """A beam whose chief ray meets a mirror at normal incidence 3 mm from the centre of a 2 mm hole: the reflected power
    is the incident power less the beam's share on the hole (the chord integral over the hole), relative to the same
    mirror without the hole; the share through the hole goes on beside the element."""
    laser = B.laser("SRC", wavelength_nm=LAM, power_W=P_W, waist_radius_mm=2.0 * W0, position=(0.0, 0.0, 0.0), direction=X)
    w = 2.0 * W0 * math.sqrt(1.0 + (STOP_X / rayleigh_range(2.0 * W0, LAM)) ** 2)
    back = B.observation_plane("BACK", position=(5.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))

    def reflected(surface: PlaneSurface) -> float:
        result = run(ProjectSnapshot(components=(laser, _component("M", STOP_X, surface, normal=(-1.0, 0.0, 0.0))), observations=(back,)))
        returning = [a for a in result.observations[0].arrivals if a.beam.direction[0] < 0.0]
        assert len(returning) == 1
        return float(returning[0].beam.power_W)

    holed = _holed_mirror(3.0)
    plain = holed.model_copy(update={"holes": None})
    hole_share = _round_beam_chord_reference(w, 1.0, 5.0, lambda x: math.sqrt(max(4.0 - (x - 3.0) ** 2, 0.0))) * 2.0 / (math.pi * w * w)
    assert 1e-3 < hole_share < 0.5
    assert reflected(holed) / reflected(plain) == pytest.approx(1.0 - hole_share, rel=1e-9)
