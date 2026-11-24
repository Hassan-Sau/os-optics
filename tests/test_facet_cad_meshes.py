"""Facet surfaces as CAD exports give them (``optics/surfaces/facets.py``): coordinates rounded to a number of decimals or
stored in single precision (float32), round faces of many corners, and solids of many faces. References: a flat face of
rounded coordinates is one planar patch whose plane lies within the coordinates' rounding of every corner, and a beam
through it transmits the Fresnel share of two flat faces, ``(4 n / (n + 1)^2)^2``, as through the unrounded part; a round
face of more than 64 corners is one patch; neighbouring patches share their outline corners (no gap at an edge); the
return-light screen of a solid of many faces costs work in proportion to the faces it considers and its notices name a
bounded number of pairs."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest

from os_optics import validate
from os_optics import builders as B
from os_optics.contracts import FacetApproximation
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.contracts.optics import CircularAperture, PlaneSurface, RefractInteraction
from os_optics.optics.surfaces.facets import (COPLANAR_FLOOR_MM, COPLANAR_MAX_FRACTION, COPLANAR_QUANTA, FITTED_FACE_NOTICE_FRACTION,
                                              MAX_FACET_OUTLINE_CORNERS, coordinate_quantum, coplanar_tolerance, mesh_extent,
                                              outline_problem, planar_patches)
from os_optics._vendor.shared_contracts.common.polygons import polygon_problem
from os_optics.validation.exposures import NAMED_RETURNS

from facet_helpers import GLASS, N_GLASS, convex_solid, extruded, part, request, solid, unit

# These CAD-geometry references assume a test-only ideal glass with zero bulk absorption.
IDEAL_GLASS = ConstantIndexMaterial(material_id=GLASS, refractive_index=N_GLASS, valid_wavelength_nm=(400.0, 1100.0),
                                    absorption_coefficient_per_mm=0.0)
WL = 633.0
T_FACE = 4.0 * N_GLASS / (N_GLASS + 1.0) ** 2  # Fresnel power transmittance at normal incidence, either way
CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def _laser(at, direction, *, back: float = 30.0, waist: float = 0.3):
    d = unit(direction)
    return B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=waist, position=tuple(np.array(at, dtype=float) - back * d),
                   direction=tuple(d))


def _plane(oid: str, at, normal):
    return B.observation_plane(oid, position=tuple(float(x) for x in at), normal=tuple(float(x) for x in unit(normal)))


def _rotation(axis, angle: float) -> np.ndarray:
    a = unit(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _rotated_cube(rounding):
    """A 10 mm cube centred on the origin, turned 17 deg about (1, 2, 3), its corners rounded to ``rounding`` decimals,
    stored as float32 (``"float32"``) or kept (``None``)."""
    corners = np.array([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)],
                       dtype=float) - 5.0
    points = (_rotation((1.0, 2.0, 3.0), math.radians(17.0)) @ corners.T).T
    if rounding == "float32":
        points = points.astype(np.float32).astype(float)
    elif rounding is not None:
        points = np.round(points, rounding)
    return points, convex_solid([tuple(p) for p in points], CUBE_FACES)


def _corner_3d(patch, k: int) -> np.ndarray:
    u = np.array(patch.u_axis)
    v = np.cross(np.array(patch.normal), u)
    pu, pv = patch.polygon[k]
    return np.array(patch.origin) + pu * u + pv * v


def _shared_corners_agree(patches) -> float:
    """The largest distance between the 3D positions one outline vertex has in the patches whose outlines hold it."""
    seen: dict[int, list[np.ndarray]] = {}
    for patch in patches:
        for k, vertex in enumerate(patch.loop):
            seen.setdefault(vertex, []).append(_corner_3d(patch, k))
    return max(float(np.linalg.norm(p - q)) for points in seen.values() for p, q in itertools.combinations(points, 2))


# ------------------------------------------------------------------------------------------------ tolerance from the data


def test_the_coordinate_rounding_is_read_from_the_coordinates():
    assert coordinate_quantum([(0.25, 1.0, -3.5)]) == 0.005  # (2 decimals; float32 numbers too, of spacing 2**-22 at 3.5)
    assert coordinate_quantum([(1.23456789, 0.0, 2.0)]) == pytest.approx(5e-9, rel=1e-12)
    assert coordinate_quantum([(1.5e-07, 0.0, 1.0)]) == pytest.approx(5e-9, rel=1e-12)
    # single precision: half the float32 spacing at the largest coordinate (4.x: 2**-21 / 2)
    stored = [tuple(float(np.float32(x)) for x in (4.123456789, -1.987654321, 0.333333333))]
    assert coordinate_quantum(stored) == 2.0 ** -22
    # full double precision: no rounding to speak of, the floor holds
    full = [(math.sqrt(2.0), math.pi, math.e)]
    assert coordinate_quantum(full) < 1e-15 and coplanar_tolerance(full + [(0.0, 0.0, 0.0)]) == COPLANAR_FLOOR_MM
    # integer corners (exact): the tolerance is capped by the part's size
    cube = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)]
    assert coplanar_tolerance(cube) == pytest.approx(COPLANAR_MAX_FRACTION * math.sqrt(300.0), rel=1e-12)
    assert coplanar_tolerance([(1.12345678, 0.0, 0.0), (10.0, 10.0, 10.0)]) == pytest.approx(COPLANAR_QUANTA * 5e-9, rel=1e-12)


@pytest.mark.parametrize("rounding", [12, 9, 8, 6, 5, 4, 3, "float32", None])
def test_a_flat_face_of_rounded_coordinates_is_one_patch_and_its_plane_is_within_the_rounding(rounding):
    points, (vertices, triangles, faces) = _rotated_cube(rounding)
    patches = planar_patches(vertices, triangles)
    assert [p.triangles for p in patches] == [tuple(f) for f in faces]
    tol = coplanar_tolerance(vertices)
    q = coordinate_quantum(vertices)
    for patch in patches:
        # every corner of the face lies within the stated deviation of the patch's plane, itself within the tolerance and
        # within twice the rounding bound sqrt(3) q (or the floor)
        n = np.array(patch.normal)
        heights = [float(np.dot(n, np.array(vertices[i]) - np.array(patch.origin))) for k in patch.triangles for i in triangles[k]]
        assert max(abs(h) for h in heights) <= patch.deviation_mm + 1e-12
        assert patch.deviation_mm <= tol and patch.deviation_mm <= max(2.0 * math.sqrt(3.0) * q, COPLANAR_FLOOR_MM)
        assert len(patch.polygon) == 4
    # neighbouring faces share their outline corners: no gap along an edge or at a corner (exact data: each patch places a
    # corner on its own plane, within the floor of it)
    assert _shared_corners_agree(patches) <= 2.0 * COPLANAR_FLOOR_MM


def _kinked_cube(rounding, lift: float):
    """The turned cube with its corner 6 lifted by ``lift`` mm along the cube's own z (the top face bent along its diagonal
    0-6 of the fan (4, 5, 6), (4, 6, 7)), rounded."""
    corners = np.array([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)], dtype=float) - 5.0
    corners[6, 2] += lift
    points = (_rotation((1.0, 2.0, 3.0), math.radians(17.0)) @ corners.T).T
    points = np.round(points, rounding)
    triangles = []
    for face in CUBE_FACES:
        normal = np.cross(corners[face[1]] - corners[face[0]], corners[face[2]] - corners[face[0]])
        f = face if float(np.dot(normal, corners[face].mean(axis=0))) > 0.0 else face[::-1]
        triangles += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
    return tuple(tuple(float(c) for c in p) for p in points), tuple(triangles)


def test_a_face_fitted_farther_than_its_stated_fraction_of_the_size_is_stated_in_a_notice():
    """A 7e-4 mm bend of the top face (about 1e-4 rad) written to 3 decimals lies within the coordinates' rounding: the two
    facets are traced as one flat face, and a FACET_FACE_FITTED notice (NOT_APPLICABLE) names the face and its deviation
    (above FITTED_FACE_NOTICE_FRACTION of the size). Written to 6 decimals the bend is resolved: two faces, no notice; a flat
    face rounded to 4 decimals deviates by far less than the fraction: no notice."""
    def lowered(vertices, triangles):
        laser = B.laser("S", wavelength_nm=WL, power_W=1e-3, waist_radius_mm=0.2, position=(0.3, 0.2, -40.0), direction=(0.0, 0.0, 1.0))
        result = validate(request(laser, part("P", solid("body", vertices, triangles))))
        return planar_patches(vertices, triangles), [i for i in result.issues if i.code.value == "FACET_FACE_FITTED"]

    vertices, triangles = _kinked_cube(3, 7e-4)
    patches, notices = lowered(vertices, triangles)
    assert len(patches) == 6
    (notice,) = notices
    extent = mesh_extent(vertices)
    worst = max(p.deviation_mm for p in patches)
    (bent,) = [p for p in patches if p.deviation_mm > FITTED_FACE_NOTICE_FRACTION * extent]
    assert notice.status.value == "NOT_APPLICABLE" and notice.location == "component:P/surface:body"
    assert notice.numerical_context["faces"] == [f"P/body.t{bent.first}"] and notice.numerical_context["face_count"] == 1
    assert notice.numerical_context["max_face_deviation_mm"] == worst == bent.deviation_mm
    assert notice.numerical_context["part_size_mm"] == extent
    # the stated deviation is the farthest corner of the face from its traced plane (computed here), and the four corners do
    # not lie in one plane (the bend is in the data: the lifted corner is off the plane of the other three)
    n, o = np.array(bent.normal), np.array(bent.origin)
    corners = sorted({i for k in bent.triangles for i in triangles[k]})
    heights = [float(np.dot(n, np.array(vertices[i]) - o)) for i in corners]
    assert max(abs(h) for h in heights) == pytest.approx(bent.deviation_mm, abs=1e-12)
    a, b, c, d = (np.array(vertices[i]) for i in corners)
    assert abs(float(np.dot(unit(np.cross(b - a, c - a)), d - a))) > 1e-4
    vertices, triangles = _kinked_cube(6, 7e-4)
    patches, notices = lowered(vertices, triangles)
    assert len(patches) == 7 and not notices
    points, (vertices, triangles, _faces) = _rotated_cube(4)
    patches, notices = lowered(vertices, triangles)
    assert len(patches) == 6 and not notices
    assert max(p.deviation_mm for p in patches) < FITTED_FACE_NOTICE_FRACTION * mesh_extent(vertices)


@pytest.mark.parametrize("rounding", [8, 6, 4, "float32"])
def test_a_beam_through_rounded_faces_transmits_as_through_two_flat_faces(rounding):
    points, (vertices, triangles, _faces) = _rotated_cube(rounding)
    face = points[[0, 1, 5, 4]]
    centre = face.mean(axis=0)
    outward = unit(np.cross(face[1] - face[0], face[3] - face[0]))
    if float(np.dot(outward, centre)) < 0.0:
        outward = -outward
    result = validate(request(_laser(centre, -outward), part("C", solid("c", vertices, triangles)),
                              observations=[_plane("o", centre - 40.0 * outward, outward)], materials=(IDEAL_GLASS,)))
    (arrival,) = result.observations[0].arrivals
    assert arrival.beam.power_W == pytest.approx(1e-3 * T_FACE ** 2, rel=1e-12)
    assert arrival.beam.gaussian is not None and arrival.beam.gaussian.valid
    assert not [i for i in result.issues if i.code.value == "FACET_EDGE_DIVIDES_BEAM"]
    assert result.power_budget.conserved


@pytest.mark.parametrize("rounding", [6, "float32"])
def test_rays_through_the_edges_and_corners_of_rounded_faces_enter_and_leave_once(rounding):
    points, (vertices, triangles, _faces) = _rotated_cube(rounding)
    mesh = solid("c", vertices, triangles)
    edges = [(a, b) for a, b in itertools.combinations(range(8), 2) if abs(np.linalg.norm(points[a] - points[b]) - 10.0) < 1e-2]
    targets = [points[i] for i in range(8)] + [0.5 * (points[a] + points[b]) for a, b in edges]
    rng = np.random.default_rng(3)
    for target in targets:
        d = unit(rng.normal(size=3) * 1.5 - target)  # toward a point inside, near the centre
        result = validate(request(_laser(target, d, waist=0.05), part("C", mesh), observations=[_plane("o", target + 60.0 * d, d)]))
        events = [s.end_event for p in result.paths for s in p.segments if (s.end_surface_id or "").startswith("C/")]
        assert events.count("refract") == 2, (target, events)
        assert not [i for i in result.issues if i.code.value == "MEDIUM_MISMATCH"]
        assert result.power_budget.conserved


def test_the_facets_of_a_tessellated_rod_stay_separate_patches_under_rounding():
    radius, sides = 5.0, 96
    polygon = [(round(radius * math.cos(2 * math.pi * k / sides), 6), round(radius * math.sin(2 * math.pi * k / sides), 6))
               for k in range(sides)]
    vertices, triangles, faces = extruded(polygon, 3.0)
    patches = planar_patches(vertices, triangles)
    # 2 caps + one patch per side face (the side faces differ by 3.75 deg: far beyond the 1e-6 mm rounding)
    assert len(patches) == sides + 2
    assert sorted(p.triangles for p in patches) == sorted(tuple(f) for f in faces)


# ------------------------------------------------------------------------------------------------ round faces


def _rod(sides: int, radius: float = 5.0, hz: float = 3.0):
    """A rod along z whose caps are fans of ``sides - 2`` triangles from corner 0 (as CAD exports a round face)."""
    corners = [(radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
    vertices = [(x, y, -hz) for x, y in corners] + [(x, y, hz) for x, y in corners]
    triangles = [(0, k + 1, k) for k in range(1, sides - 1)] + [(sides, sides + k, sides + k + 1) for k in range(1, sides - 1)]
    for i in range(sides):
        j = (i + 1) % sides
        triangles += [(i, j, sides + j), (i, sides + j, sides + i)]
    approx = FacetApproximation(kind="TESSELLATED_FROM_CAD", max_chord_deviation_mm=radius * (1.0 - math.cos(math.pi / sides)))
    return solid("rod", tuple(vertices), tuple(triangles), approximation=approx)


@pytest.mark.parametrize("sides", [64, 65, 96, 360])
def test_a_round_cap_of_many_corners_is_one_patch_and_passes_a_beam_whole(sides):
    rod = _rod(sides)
    patches = planar_patches(rod.vertices_mm, rod.triangles)
    caps = [p for p in patches if abs(p.normal[2]) > 0.999]
    assert len(patches) == sides + 2 and len(caps) == 2 and all(len(c.polygon) == sides for c in caps)
    # an axial beam off the axis, inside one cap: through both caps (no observation: the ledger alone)
    result = validate(request(_laser((0.3, 0.2, 3.0), (0, 0, -1), back=10.0), part("R", rod), materials=(IDEAL_GLASS,)))
    escaped = [e for e in result.power_budget.entries if e.kind == "ESCAPED"]
    assert sum(e.power_W for e in escaped) == pytest.approx(1e-3 * T_FACE ** 2, rel=1e-12)
    assert not [i for i in result.issues if i.code.value == "FACET_EDGE_DIVIDES_BEAM"]
    assert result.power_budget.conserved


def test_a_facet_outline_is_checked_as_a_polygon_beyond_64_corners_and_up_to_its_limit():
    def ring(n, r=5.0):
        return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    # the same verdicts as the aperture contract's check up to 64 corners
    star = [(math.cos(4 * math.pi * k / 5), math.sin(4 * math.pi * k / 5)) for k in range(5)]  # (self-crossing)
    cases = [ring(3), ring(64), list(reversed(ring(12))), star, [(0, 0), (2, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (0, 0)]]
    for points in cases:
        assert (outline_problem(points) is None) == (polygon_problem(points) is None), points
    # beyond 64 corners: simple rings pass, a crossing is found, more than the limit is refused
    assert outline_problem(ring(65)) is None and outline_problem(ring(MAX_FACET_OUTLINE_CORNERS)) is None
    crossed = ring(400)
    crossed[10], crossed[200] = crossed[200], crossed[10]
    assert "meet" in outline_problem(crossed)
    assert "at most" in outline_problem(ring(MAX_FACET_OUTLINE_CORNERS + 1))
    start = time.perf_counter()
    outline_problem(ring(MAX_FACET_OUTLINE_CORNERS))
    assert time.perf_counter() - start < 2.0  # (the sweep compares neighbouring edges only: about 10 ms)


# ------------------------------------------------------------------------------------------------ facet edge, no Gaussian


def test_a_beam_without_a_gaussian_state_on_a_ridge_books_the_pupil_share_beyond_the_facet_edge():
    """A pinhole truncates the beam (no Gaussian state behind it); its rest falls on the ridge of a prism. The share of the
    pupil samples that meet the other face is booked beyond the patch's edge (unknown fate), on the patch the chief ray
    meets; mirrored offsets give the mirrored patch and the same share."""
    side, half_y = 20.0, 10.0
    h = side * math.sqrt(3.0) / 2.0
    vertices, triangles, _faces = extruded([(-side / 2, 0.0), (side / 2, 0.0), (0.0, h)], half_y)
    prism = solid("prism", vertices, triangles)
    found = {}
    for x in (-0.05, 0.05):
        pin = PlaneSurface(surface_id="pin", origin_mm=(x, 0.0, h + 10.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=0.15), medium_positive="air", medium_negative="air",
                           interaction=RefractInteraction())
        result = validate(request(_laser((x, 0.0, h + 10.0), (0, 0, -1), back=10.0), part("PH", pin), part("P", prism),
                                  observations=[_plane("o", (0.0, 0.0, -60.0), (0, 0, 1))]))
        (issue,) = [i for i in result.issues if i.code.value == "FACET_EDGE_DIVIDES_BEAM"]
        ctx = issue.numerical_context
        cut, live = ctx["pupil_rays_cut"], ctx["pupil_rays_live"]
        assert 0 < cut < live and ctx["loss_fraction"] == pytest.approx(cut / live, rel=1e-15)
        assert issue.status.value == "OUT_OF_MODEL_SCOPE" and "no valid Gaussian state" in issue.message
        entries = result.power_budget.entries
        truncated = sum(e.power_W for e in entries if e.kind == "APERTURE_TRUNCATION")
        (edge,) = [e for e in entries if e.kind == "TERMINATED_OUT_OF_MODEL" and e.detail == "facet edge"]
        assert edge.power_W == pytest.approx((1e-3 - truncated) * cut / live, rel=1e-9)
        assert result.power_budget.conserved
        found[x] = (issue.location, cut, live)
    assert found[-0.05][1:] == found[0.05][1:] and found[-0.05][0] != found[0.05][0]


# ------------------------------------------------------------------------------------------------ many faces: notices


def test_a_withheld_bound_notice_names_a_bounded_number_of_pairs_and_counts_them_all():
    vertices, triangles, _faces = convex_solid([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10),
                                                (0, 10, 10)], CUBE_FACES)
    result = validate(request(_laser((5.0, 5.0, 10.0), (0, 0, -1), back=10.0), part("C", solid("c", vertices, triangles)),
                              observations=[_plane("o", (5.0, 5.0, -30.0), (0, 0, 1))], materials=(IDEAL_GLASS,)))
    notices = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    (notice,) = [i for i in notices if i.numerical_context["cause"] == "RETURN_PAIR_GAIN_BOUND_1"]
    # (the cube is a convex plane-faced solid: its internal folds are analysed, and its lossless total internal reflections
    # leave them without a certified sum -- a closed return through its faces, named as one solid's)
    (folded,) = [i for i in notices if i.numerical_context["cause"] == "CLOSED_RETURN_NOT_BOUNDED"]
    assert "the faces C/c.t0, C/c.t2, C/c.t4, C/c.t6, C/c.t8, C/c.t10 of one solid" in folded.message
    ctx = notice.numerical_context
    # six faces facing each other inside the glass (total internal reflection: gain bound 1): 15 pairs, 12 named
    assert ctx["return_pair_count"] == 15 and len(ctx["return_pair_surfaces"]) == NAMED_RETURNS
    assert notice.message.count(" and C/c.t") == NAMED_RETURNS and "15 pairs of surfaces (among them" in notice.message
