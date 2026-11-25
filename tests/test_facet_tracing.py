"""Triangle-facet surfaces traced facet by facet (``contracts.optics.MeshSurface``, ``optics/surfaces/facets.py``), against
independent references: Snell's law and the minimum deviation of a triangular prism, the retroreflection of a solid corner
cube (exit direction -input, transverse offset inverted about the apex, Fresnel transmittance at entry and exit, total
internal reflection at the three back faces), the 90-degree deviation of a silvered penta prism whatever the input tilt, and
a block of facets that traces exactly as the same window stated by two analytic planes. A beam divided by a facet edge, a
tessellated surface and the domain refusals state what the trace does not follow."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from os_optics import validate, validate_json
from os_optics import builders as B
from os_optics.contracts import FacetApproximation, MeshSurface
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.contracts.optics import MirrorInteraction, PlaneSurface, RectangularAperture, RefractInteraction
from os_optics.contracts.polarization import JonesVector
from os_optics.optics.surfaces.facets import planar_patches

from facet_helpers import (EXACT, GLASS, N_GLASS, convex_solid, extruded, fresnel_t, part, request, silvered, snell, solid,
                           unit)

# The synthetic prism/window here explicitly assumes no bulk absorption; only Fresnel loss is tested.
IDEAL_GLASS = ConstantIndexMaterial(material_id=GLASS, refractive_index=N_GLASS, valid_wavelength_nm=(400.0, 1100.0),
                                    absorption_coefficient_per_mm=0.0)
WL = 633.0


def _laser(at, direction, *, back: float = 40.0, waist: float = 0.1, polarization=None):
    d = unit(direction)
    start = np.array(at, dtype=float) - back * d
    return B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=waist, position=tuple(start), direction=tuple(d),
                   polarization=polarization)


def _plane(oid: str, at, normal):
    return B.observation_plane(oid, position=tuple(float(x) for x in at), normal=tuple(float(x) for x in unit(normal)))


def _arrival(result, oid: str):
    (obs,) = [o for o in result.observations if o.observation_id == oid]
    assert len(obs.arrivals) == 1, [a.beam.path_id for a in obs.arrivals]
    return obs.arrivals[0]


def _closed(result) -> None:
    budget = result.power_budget
    assert budget is not None and budget.conserved and abs(budget.residual_W) <= budget.tolerance_W


# ------------------------------------------------------------------------------------------------ decomposition


def test_coplanar_facets_merge_into_one_outline_and_a_ring_face_stays_triangles():
    cube_v, cube_t, faces = convex_solid(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    patches = planar_patches(cube_v, cube_t)
    assert [p.triangles for p in patches] == [tuple(f) for f in faces]
    for p in patches:  # every outline: the face's 4 corners, counter-clockwise from outside, unit square
        xs = [u for u, _ in p.polygon]
        ys = [v for _, v in p.polygon]
        assert len(p.polygon) == 4 and max(xs) - min(xs) == pytest.approx(1.0, abs=1e-15) and max(ys) - min(ys) == pytest.approx(1.0, abs=1e-15)
    # a square plate with a square hole (a frame) fanned into 8 coplanar triangles: one patch around a hole is no simple
    # polygon, so it is traced triangle by triangle
    outer = [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)]
    inner = [(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)]
    frame = tuple((a, b, c) for a, b, c in ((0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)))
    sheet = planar_patches(tuple(map(tuple, outer + inner)), frame)
    assert [p.triangles for p in sheet] == [(k,) for k in range(8)]


# ------------------------------------------------------------------------------------------------ triangular prism


def _equilateral(side: float = 20.0, half_y: float = 10.0):
    h = side * math.sqrt(3.0) / 2.0
    return extruded([(-side / 2.0, 0.0), (side / 2.0, 0.0), (0.0, h)], half_y), h


@pytest.mark.parametrize("incidence_deg", [None, 55.0, 40.0])
def test_prism_deviation_follows_snell_and_its_minimum(incidence_deg):
    (vertices, triangles, _faces), h = _equilateral()
    apex = math.radians(60.0)
    d_min = 2.0 * math.asin(N_GLASS * math.sin(apex / 2.0)) - apex  # minimum deviation (closed form)
    theta = (d_min + apex) / 2.0 if incidence_deg is None else math.radians(incidence_deg)
    inward = np.array([math.cos(-math.pi / 6.0), 0.0, math.sin(-math.pi / 6.0)])  # into the left face
    entry_normal = -inward
    exit_normal = np.array([math.cos(math.pi / 6.0), 0.0, math.sin(math.pi / 6.0)])
    d_in = np.array([math.cos(-math.pi / 6.0 + theta), 0.0, math.sin(-math.pi / 6.0 + theta)])
    hit = np.array([-5.0, 0.0, h / 2.0])
    # the reference: Snell's law at the two faces (vector form), and the scalar deviation theta1 + theta2' - A
    d_glass = snell(d_in, entry_normal, 1.0, N_GLASS)
    d_out = snell(d_glass, exit_normal, N_GLASS, 1.0)
    sin_t = math.sin(theta) / N_GLASS
    theta2 = math.asin(N_GLASS * math.sin(apex - math.asin(sin_t)))
    deviation = theta + theta2 - apex
    assert math.acos(float(np.dot(d_in, d_out))) == pytest.approx(deviation, abs=1e-12)
    if incidence_deg is None:
        assert deviation == pytest.approx(d_min, abs=1e-12) and abs(d_glass[2]) < 1e-15  # parallel to the base inside
    result = validate(request(_laser(hit, d_in), part("P", solid("prism", vertices, triangles)),
                              observations=[_plane("out", hit + 60.0 * d_out, d_out)], materials=(IDEAL_GLASS,)))
    arrival = _arrival(result, "out")
    assert np.allclose(arrival.beam.direction, d_out, atol=1e-12, rtol=0)
    assert [step.split(":")[1] for step in arrival.lineage] == ["refract", "refract"]
    assert all(step.startswith("P/prism.t") for step in arrival.lineage)
    # the source's x axis is the prism's y axis (normal to the plane of incidence): s light; Fresnel at both faces
    t_entry = fresnel_t(math.cos(theta), 1.0, N_GLASS)[0]
    t_exit = fresnel_t(abs(float(np.dot(d_glass, exit_normal))), N_GLASS, 1.0)[0]
    assert arrival.beam.power_W == pytest.approx(1e-3 * t_entry * t_exit, rel=1e-12)
    assert arrival.beam.gaussian is not None and arrival.beam.gaussian.valid
    assert not [i for i in result.issues if i.code.value.startswith("FACET")]
    _closed(result)


# ------------------------------------------------------------------------------------------------ corner cube


def test_a_solid_corner_cube_retroreflects_with_fresnel_at_entry_and_exit():
    side = 20.0
    vertices, triangles, _faces = convex_solid([(0, 0, 0), (side, 0, 0), (0, side, 0), (0, 0, side)],
                                               [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    d_in = -unit((1.0, 1.0, 1.0))  # normal to the front face x + y + z = side
    entry = np.array([9.0, 7.0, 4.0])  # on the front face, off its centre: the chief ray meets each back face once
    axis_point = np.full(3, side / 3.0)  # the front face's centre: the apex projected along the axis
    result = validate(request(_laser(entry, d_in, back=10.0), part("CC", solid("cube_corner", vertices, triangles)),
                              observations=[_plane("back", axis_point + 30.0 * unit((1, 1, 1)), (1, 1, 1))], materials=(IDEAL_GLASS,)))
    arrival = _arrival(result, "back")
    d_out = np.array(arrival.beam.direction)
    assert np.allclose(d_out, -d_in, atol=1e-12, rtol=0)  # exit direction = -input
    events = [step.split(":")[1] for step in arrival.lineage]
    assert events == ["refract", "tir", "tir", "tir", "refract"]
    assert len({step.split(":")[0] for step in arrival.lineage[1:4]}) == 3  # the three back faces, each once
    # the transverse offset from the axis through the apex is inverted (a point reflection through the axis)
    axis = unit((1, 1, 1))
    position = np.array(arrival.beam.position_mm)
    offset_in = entry - axis_point - float(np.dot(entry - axis_point, axis)) * axis
    offset_out = position - axis_point - float(np.dot(position - axis_point, axis)) * axis
    assert np.allclose(offset_out, -offset_in, atol=1e-11, rtol=0)
    # power: Fresnel transmittance at normal incidence in and out (4 n / (n + 1)^2 each), total internal reflection lossless
    t = 4.0 * N_GLASS / (N_GLASS + 1.0) ** 2
    assert arrival.beam.power_W == pytest.approx(1e-3 * t * t, rel=1e-12)
    assert arrival.beam.gaussian is not None and arrival.beam.gaussian.valid
    _closed(result)


# ------------------------------------------------------------------------------------------------ penta prism


def _penta(s: float = 10.0, b: float = 4.0, half_y: float = 8.0):
    """A penta prism in (x, z): entrance face z = 0 (x in 0 .. s), exit face x = 0 (z in 0 .. s), interior angles 90 and 4 x
    112.5 degrees. Faces: 0, 1 the caps; 2 + i the side from corner i (2 entrance, 3 A-B, 4 B-C, 5 C-D, 6 exit)."""
    c, sn = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    a = (s - b / math.sqrt(2.0)) / (c - sn)
    A = (s, 0.0)
    Bp = (A[0] + a * sn, A[1] + a * c)  # heading 67.5 degrees
    C = (Bp[0] - b / math.sqrt(2.0), Bp[1] + b / math.sqrt(2.0))  # heading 135 degrees
    D = (0.0, s)
    assert abs(C[0] - a * c - D[0]) < 1e-12 and abs(C[1] - a * sn - D[1]) < 1e-12  # closes on D (heading 202.5 degrees)
    return extruded([(0.0, 0.0), A, Bp, C, D], half_y)


@pytest.mark.parametrize("tilt_deg, out_of_plane_deg", [(0.0, 0.0), (2.5, 0.0), (-3.0, 0.0), (1.5, 2.0)])
def test_a_penta_prism_deviates_by_90_degrees_whatever_the_input_tilt(tilt_deg, out_of_plane_deg):
    vertices, triangles, faces = _penta()
    mirrors = silvered(faces[3] + faces[5])
    t, p = math.radians(tilt_deg), math.radians(out_of_plane_deg)
    d_in = unit((math.sin(t), math.sin(p), math.cos(t) * math.cos(p)))
    entry = np.array([3.0, 0.5, 0.0])
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # 90 degrees about y: +z -> -x
    expected = rotation @ d_in
    result = validate(request(_laser(entry, d_in), part("PP", solid("penta", vertices, triangles, groups=(mirrors,))),
                              observations=[_plane("out", np.array([-40.0, 0.0, 5.0]), (-1, 0, 0))]))
    arrival = _arrival(result, "out")
    assert np.allclose(arrival.beam.direction, expected, atol=1e-12, rtol=0)
    if out_of_plane_deg == 0.0:  # in the principal plane: a 90-degree deviation (out of it: the same rotation about y)
        assert math.degrees(math.acos(float(np.dot(arrival.beam.direction, d_in)))) == pytest.approx(90.0, abs=1e-10)
    assert [step.split(":")[1] for step in arrival.lineage] == ["refract", "reflect", "reflect", "refract"]
    _closed(result)


# ------------------------------------------------------------------------------------------------ window


def _block(half: float, thick: float):
    return convex_solid(
        [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0),
         (-half, -half, thick), (half, -half, thick), (half, half, thick), (-half, half, thick)],
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def test_a_block_of_facets_traces_exactly_as_the_analytic_window():
    half, thick = 10.0, 6.0
    vertices, triangles, _faces = _block(half, thick)
    ap = RectangularAperture(half_width_mm=half, half_height_mm=half)
    planes = (PlaneSurface(surface_id="front", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, -1.0), aperture=ap,
                           medium_positive="air", medium_negative=GLASS, interaction=RefractInteraction()),
              PlaneSurface(surface_id="back", origin_mm=(0.0, 0.0, thick), normal=(0.0, 0.0, 1.0), aperture=ap,
                           medium_positive="air", medium_negative=GLASS, interaction=RefractInteraction()))
    d_in = unit((math.sin(math.radians(12.0)), math.sin(math.radians(5.0)), 1.0))
    entry = np.array([1.0, -2.0, 0.0])
    pol = JonesVector(ex=(0.6, 0.0), ey=(0.0, 0.8))
    obs = [_plane("out", (0.0, 0.0, 40.0), (0, 0, 1))]
    faceted = validate(request(_laser(entry, d_in, polarization=pol), part("W", solid("block", vertices, triangles)),
                               observations=obs, materials=(IDEAL_GLASS,)))
    analytic = validate(request(_laser(entry, d_in, polarization=pol), part("W", *planes),
                                observations=obs, materials=(IDEAL_GLASS,)))
    a, b = _arrival(faceted, "out"), _arrival(analytic, "out")
    assert np.allclose(a.beam.position_mm, b.beam.position_mm, atol=1e-12, rtol=0)
    assert np.allclose(a.beam.direction, b.beam.direction, atol=1e-15, rtol=0)
    assert a.beam.power_W == pytest.approx(b.beam.power_W, rel=1e-14)
    assert a.beam.optical_path_length_mm == pytest.approx(b.beam.optical_path_length_mm, rel=1e-14)
    assert a.beam.polarization.ex == pytest.approx(b.beam.polarization.ex, abs=1e-14)
    assert a.beam.polarization.ey == pytest.approx(b.beam.polarization.ey, abs=1e-14)
    ga, gb = a.beam.gaussian, b.beam.gaussian
    assert ga.valid and gb.valid
    for name in ("beam_radius_x_mm", "beam_radius_y_mm", "waist_distance_x_mm", "waist_distance_y_mm"):
        assert getattr(ga, name) == pytest.approx(getattr(gb, name), rel=1e-12), name
    _closed(faceted)


# ------------------------------------------------------------------------------------------------ facet edges


@pytest.mark.parametrize("offset_radii", [1.2, 0.5, 0.0])
def test_a_beam_divided_by_a_facet_edge_follows_one_facet_and_states_the_share_beyond(offset_radii):
    """A beam falling on the ridge of a prism (the edge between its two upper faces) at ``offset_radii`` beam radii from it:
    the beam goes on through the face its chief ray meets; the share beyond the ridge -- the Gaussian share across a straight
    edge, ``erfc(sqrt(2) d / w) / 2`` -- is not followed and booked with an unknown fate."""
    (vertices, triangles, _faces), h = _equilateral()
    w = 0.3
    x = -offset_radii * w - (1e-9 if offset_radii == 0.0 else 0.0)  # (on the ridge: the chief ray takes the left face)
    hit = np.array([x, 0.0, h - abs(x) * math.tan(math.radians(60.0))])
    source = B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=w, position=tuple(hit + np.array([0.0, 0.0, 30.0])),
                     direction=(0.0, 0.0, -1.0), waist_position_z_mm=30.0)
    result = validate(request(source, part("P", solid("prism", vertices, triangles))))
    share = 0.5 * math.erfc(math.sqrt(2.0) * offset_radii * w / w)
    edge = [i for i in result.issues if i.code.value == "FACET_EDGE_DIVIDES_BEAM"]
    flows = [e for e in result.power_budget.entries if e.kind == "TERMINATED_OUT_OF_MODEL" and e.detail == "facet edge"]
    (first,) = [e for e in flows if e.path_id == "S#0"][:1]
    assert first.power_W == pytest.approx(1e-3 * share, rel=1e-6)
    (issue,) = [i for i in edge if i.numerical_context.get("path_id") == "S#0" and "P/prism.t" in (i.location or "")][:1]
    assert issue.numerical_context["loss_fraction"] == pytest.approx(share, rel=1e-6)
    assert issue.status.value == ("OUT_OF_MODEL_SCOPE" if share > 0.1 else "WARNING")
    (path,) = [p for p in result.paths if p.path_id == "S#0"]
    assert path.segments[0].end_surface_id.startswith("P/prism.t")  # the chief ray meets one face (no gap at the ridge)
    _closed(result)


# ------------------------------------------------------------------------------------------------ tessellated surface


def _ray_polygon_exit(o2, d2, polygon):
    """Nearest crossing (t, edge index) of a 2D ray with a convex polygon's edges (an independent reference)."""
    best = (math.inf, -1)
    n = len(polygon)
    for i in range(n):
        a, b = np.array(polygon[i]), np.array(polygon[(i + 1) % n])
        e = b - a
        den = d2[0] * (-e[1]) - d2[1] * (-e[0])
        if abs(den) < 1e-15:
            continue
        rhs = a - o2
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / den
        s = (d2[0] * rhs[1] - d2[1] * rhs[0]) / den
        if t > 1e-9 and -1e-12 <= s <= 1 + 1e-12 and t < best[0]:
            best = (t, i)
    return best


def test_a_tessellated_rod_traces_its_facets_and_states_no_gaussian_beam_behind_them():
    radius, sides = 5.0, 48
    polygon = [(radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    chord = radius * (1.0 - math.cos(math.pi / sides))
    vertices, triangles, _faces = extruded(polygon, 6.0)
    approx = FacetApproximation(kind="TESSELLATED_FROM_CAD", max_chord_deviation_mm=chord, source="a round rod tessellated")
    rod = solid("rod", vertices, triangles, approximation=approx)
    # aim at the centre of a lower facet (centre angle -pi/2 + 5 pi / sides): the ray along +z through x = apothem cos(phi)
    k = sides * 3 // 4 + 2
    phi = math.pi * (2 * k + 1) / sides
    apothem = radius * math.cos(math.pi / sides)
    entry = np.array([apothem * math.cos(phi), 0.0, apothem * math.sin(phi)])
    d_in = np.array([0.0, 0.0, 1.0])
    # the reference: Snell at the entry facet (its outward normal along phi), then the exit facet found by a 2D intersection
    n_entry = np.array([math.cos(phi), 0.0, math.sin(phi)])
    d_glass = snell(d_in, n_entry, 1.0, N_GLASS)
    t, edge = _ray_polygon_exit(np.array([entry[0], entry[2]]), np.array([d_glass[0], d_glass[2]]), polygon)
    phi_out = math.pi * (2 * edge + 1) / sides
    d_out = snell(d_glass, np.array([math.cos(phi_out), 0.0, math.sin(phi_out)]), N_GLASS, 1.0)
    exit_point = entry + t * d_glass
    result = validate(request(_laser(entry, d_in, waist=0.03), part("R", rod),
                              observations=[_plane("out", exit_point + 30.0 * d_out, d_out)]))
    arrival = _arrival(result, "out")
    assert np.allclose(arrival.beam.direction, d_out, atol=1e-12, rtol=0)
    assert np.allclose(arrival.beam.position_mm, exit_point + 30.0 * d_out, atol=1e-9, rtol=0)
    assert arrival.beam.gaussian is not None and not arrival.beam.gaussian.valid
    assert arrival.beam.gaussian.reason == "GAUSSIAN_UNDEFINED_AFTER_FACETS"
    (issue,) = [i for i in result.issues if i.code.value == "GAUSSIAN_UNDEFINED_AFTER_FACETS"]
    assert f"chord deviation {chord:g} mm" in issue.message
    assert arrival.first_order is None and "TESSELLATED_FROM_CAD" in (arrival.first_order_issue or "")
    _closed(result)


# ------------------------------------------------------------------------------------------------ domain


def test_an_open_sheet_between_two_media_is_refused_and_an_open_mirror_sheet_is_traced():
    sheet_v = ((-5.0, -5.0, 0.0), (5.0, -5.0, 0.0), (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0))
    sheet_t = ((0, 1, 2), (0, 2, 3))
    open_glass = MeshSurface(surface_id="face", vertices_mm=sheet_v, triangles=sheet_t, approximation=EXACT,
                             medium_positive="air", medium_negative=GLASS, interaction=RefractInteraction())
    result = validate(request(_laser((0.0, 0.0, 0.0), (0, 0, -1)), part("F", open_glass)))
    assert result.status.value == "OUT_OF_MODEL_SCOPE" and result.outcome.execution.termination == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in result.issues if i.code.value == "FACET_SURFACE_NOT_TRACED"]
    assert "open sheet" in issue.message and issue.location == "component:F/surface:face"
    # a hollow roof mirror: two open facets at 90 degrees reflect a beam in their principal plane back, reversed
    roof_v = ((-5.0, -5.0, 5.0), (0.0, -5.0, 0.0), (5.0, -5.0, 5.0), (-5.0, 5.0, 5.0), (0.0, 5.0, 0.0), (5.0, 5.0, 5.0))
    roof_t = ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))  # counter-clockwise seen from above (+z: the reflecting side)
    roof = MeshSurface(surface_id="roof", vertices_mm=roof_v, triangles=roof_t, approximation=EXACT, interaction=MirrorInteraction())
    d_in = unit((0.2, 0.1, -1.0))
    traced = validate(request(_laser((-1.7, 0.4, 1.7), d_in, back=10.0), part("RM", roof),
                              observations=[_plane("up", (0.0, 0.0, 60.0), (0, 0, 1))]))
    arrival = _arrival(traced, "up")
    expected = np.array([-d_in[0], d_in[1], -d_in[2]])  # x and z reversed, y kept (reflections at x+z and x-z planes)
    assert np.allclose(arrival.beam.direction, expected, atol=1e-12, rtol=0)
    assert [step.split(":")[1] for step in arrival.lineage] == ["reflect", "reflect"]
    _closed(traced)


def test_more_patches_than_the_trace_takes_are_refused():
    sides = 520  # a 520-sided rod: 520 side faces + 2 caps
    polygon = [(10 * math.cos(2 * math.pi * k / sides), 10 * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices, triangles, _ = extruded(polygon, 1.0)
    # the side faces of the rod are pairs of triangles; tilt alternate vertices out of plane so no two triangles share a plane
    bumped = list(vertices)
    for i in range(sides, 2 * sides, 2):
        x, y, z = bumped[i]
        bumped[i] = (x * 1.001, y, z * 1.001)
    approx = FacetApproximation(kind="TESSELLATED_FROM_CAD", max_chord_deviation_mm=0.01)
    try:
        mesh = solid("many", tuple(bumped), triangles, approximation=approx)
    except Exception:  # pragma: no cover - the bumped rod stays a valid closed mesh
        raise
    result = validate(request(_laser((30.0, 0.0, 0.0), (1, 0, 0)), part("M", mesh)))
    assert result.status.value == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in result.issues if i.code.value == "FACET_SURFACE_NOT_TRACED"]
    assert issue.numerical_context["patch_count"] > issue.numerical_context["limit"] == 1024
    assert issue.numerical_context["setting"] == "max_mesh_patches"
    assert f"compute_budget.max_mesh_patches to at least {issue.numerical_context['patch_count']}" in issue.suggested_action


def _rod(sides: int):
    """A regular ``sides``-gon rod of radius 10 mm along y, 2 mm long: ``sides`` side faces and two caps."""
    polygon = [(10 * math.cos(2 * math.pi * k / sides), 10 * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices, triangles, _ = extruded(polygon, 1.0)
    return vertices, triangles


def _with_patch_budget(req, budget: int):
    from os_optics.config import NumericalConfig

    configuration = req.configuration.model_copy(update={"numerical": NumericalConfig(max_mesh_patches=budget)})
    return req.model_copy(update={"configuration": configuration})


def test_the_patch_budget_of_the_run_decides_which_facet_surface_is_traced():
    """``max_mesh_patches`` (the run's compute budget) is the limit the lowering reads: a surface with one patch more is
    refused naming the setting and the count; at the budget it is traced, and the beam through its two caps carries the
    Fresnel transmittance of a plate at normal incidence (closed form)."""
    vertices, triangles = _rod(20)
    patches = len(planar_patches(vertices, triangles))
    assert patches == 22
    laser = B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=0.1, position=(0.0, -40.0, 0.0), direction=(0.0, 1.0, 0.0))
    base = request(laser, part("R", solid("rod", vertices, triangles)),
                   observations=[_plane("out", (0.0, 30.0, 0.0), (0, 1, 0))], materials=(IDEAL_GLASS,))
    refused = validate(_with_patch_budget(base, patches - 1))
    assert refused.status.value == "OUT_OF_MODEL_SCOPE"
    (issue,) = [i for i in refused.issues if i.code.value == "FACET_SURFACE_NOT_TRACED"]
    assert issue.numerical_context == {"patch_count": 22, "limit": 21, "setting": "max_mesh_patches"}
    assert "more than the 21 the trace takes on one surface (compute_budget.max_mesh_patches)" in issue.message
    assert "raise compute_budget.max_mesh_patches to at least 22" in issue.suggested_action
    for budget in (patches, None):  # (at the budget, and at the default 1024)
        traced = validate(base if budget is None else _with_patch_budget(base, budget))
        assert not [i for i in traced.issues if i.code.value == "FACET_SURFACE_NOT_TRACED"]
        through = fresnel_t(1.0, 1.0, N_GLASS)[0] * fresnel_t(1.0, N_GLASS, 1.0)[0]  # 4n/(n+1)^2, twice
        assert through == pytest.approx((4 * N_GLASS / (N_GLASS + 1) ** 2) ** 2, rel=1e-15)
        assert _arrival(traced, "out").beam.power_W == pytest.approx(1e-3 * through, rel=1e-12)
        _closed(traced)


def test_a_surface_above_the_default_patch_budget_is_lowered_at_a_raised_one():
    """Above the default: a 1023-sided rod (1023 side faces and two caps: 1025 patches) is refused at the default 1024, and
    lowered whole -- one traced plane per patch -- at its own count."""
    from os_optics.config import NumericalConfig
    from os_optics.errors import ModelScopeRefusal
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.scene.lower import lower_scene

    vertices, triangles = _rod(1023)
    patches = len(planar_patches(vertices, triangles))
    assert patches == 1025
    project = request(part("R", solid("rod", vertices, triangles))).project
    cfg = NumericalConfig()
    assert cfg.max_mesh_patches == 1024
    with pytest.raises(ModelScopeRefusal) as refused:
        lower_scene(project, cfg, MaterialLibrary(project.materials, cfg))
    assert refused.value.context["limit"] == 1024 and refused.value.context["patch_count"] == patches
    cfg = NumericalConfig(max_mesh_patches=patches)
    scene, _poses = lower_scene(project, cfg, MaterialLibrary(project.materials, cfg))
    assert len(scene.surfaces) == patches


def test_the_json_route_traces_a_facet_surface_and_reports_its_patches():
    vertices, triangles, faces = convex_solid([(0, 0, 0), (20, 0, 0), (0, 20, 0), (0, 0, 20)],
                                              [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    req = request(_laser((9.0, 7.0, 4.0), -unit((1, 1, 1)), back=10.0), part("CC", solid("cube_corner", vertices, triangles)))
    document = json.loads(req.model_dump_json())
    assert "facet_groups" not in document["project"]["components"][1]["surfaces"][0]  # (omitted when absent: hashes unchanged)
    out = json.loads(validate_json(json.dumps(document)))
    assert out["outcome"]["execution"]["termination"] == "COMPLETED"
    (path,) = out["paths"]
    assert [s["end_surface_id"] for s in path["segments"]][:5] == [
        "CC/cube_corner.t3", "CC/cube_corner.t0", "CC/cube_corner.t1", "CC/cube_corner.t2", "CC/cube_corner.t3"]  # front, z, y, x, front
    assert [s["end_event"] for s in path["segments"]][:5] == ["refract", "tir", "tir", "tir", "refract"]
