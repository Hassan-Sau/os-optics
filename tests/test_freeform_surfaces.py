"""Two-variable sag surfaces: Zernike, XY polynomial, biconic, toroid.

Independent references, none of them the Engine's own code:

* ``freeform_reference.py``: a separately coded ray trace in 60-digit decimal arithmetic from the published polar formulas, the
  documented comparison case ``FS-1`` (docs/PHYSICS_SCOPE.md), tolerances fixed before running: 1e-9 mm at every hit and at
  the plane, 1e-11 per direction component;
* the published tables typed out here (Noll 1976 Z4 .. Z11; the Fringe polynomials), exact orthonormality over the unit disc in
  rational arithmetic;
* limits with closed forms: a biconic of equal sections is a conic, an XY ``x^2 + y^2`` a paraboloid, a toroid of equal radii a
  sphere, a toroid without rotation a cylinder; the line foci of an astigmatism term from the thin-interface power.

Fixture data: every physical value below is made up for the test.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from freeform_reference import biconic_surface, toroid_surface, trace_through, xy_surface, zernike_surface
from os_optics.config import NumericalConfig
from os_optics.contracts.optics import BiconicSag, CircularAperture, RefractInteraction, SagSurface, ZernikeSag, ZernikeSagTerm
from os_optics.local import json_api, run_optical_project
from os_optics.local.catalog import load_project
from os_optics.local.compile import ParaxialSurfaceNotSupported, ProjectResolver, compile_declared, paraxial_matrix
from os_optics.optical_view import surface_mesh as sm
from os_optics.optical_view.optic_mesh import draw_body
from os_optics.optical_view.section import SectionElement, optical_section
from os_optics.optics.surfaces import freeform as ff
from os_optics.optics.surfaces.geometry import intersect
from os_optics.optics.surfaces.sag import local_normals, principal_curvatures, sag_bounds
from os_optics.prescription import PrescriptionImportError, import_prescription
from os_optics._vendor.shared_contracts.common import sag_shapes as ss
from os_optics._vendor.shared_contracts.design_catalog import PrescriptionSurfaceV1

from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, sequence, surface, with_models
from local_helpers import ALONG_X, plane, project, quat_multiply, source

WAVELENGTH_NM = 780.0
#: Tolerances of comparison case FS-1 (fixed before running; docs/PHYSICS_SCOPE.md).
FS1_POSITION_TOLERANCE_MM = 1e-9
FS1_DIRECTION_TOLERANCE = 1e-11
THICKNESS = 5.0
PLANE_AT = 80.0


def nbk7_index(wavelength_nm: float = WAVELENGTH_NM) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    b = [float(v) for v in NBK7_SELLMEIER["b"]]
    c = [float(v) for v in NBK7_SELLMEIER["c_um2"]]
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def noll(radius: str, **values: str) -> dict:
    return {"kind": "ZERNIKE", "indexing": "NOLL", "normalization": "RMS_ORTHONORMAL", "normalization_radius_mm": radius,
            "terms": [{"index": int(name[1:]), "value_mm": value} for name, value in values.items()]}


def lens_model(rear: dict, front: dict | None = None, *, clear: str = "9", edge: str = "10", thickness: str = "5") -> dict:
    """A singlet: ``front`` at z = 0 (default: a plane), the surface under test ``rear`` at z = thickness, N-BK7 between."""

    return {"model_id": "ff", "model": sequence([surface(1, "0", clear=clear, **(front or {})), surface(2, thickness, clear=clear, **rear)], ["N-BK7"], edge=edge)}


def lens_project(model: dict, *, height_mm: float = 0.0, lateral_mm: float = 0.0, waist_mm: float = 0.5, targets=None) -> dict:
    # the element's local Z is the global X axis, local X the global Y, local Y the global Z (local_helpers.ALONG_X)
    document = project([source(position=(-200.0, lateral_mm, height_mm), waist_mm=waist_mm, wavelength_nm=WAVELENGTH_NM),
                        declared_component("FF", "lens", "ff")], [plane("P", (PLANE_AT, 0.0, 0.0), radius_mm=60.0)], targets)
    return with_models(document, [model], [nbk7()])


def chief_after(run, surface_id: str = "FF/S2") -> tuple[np.ndarray, np.ndarray]:
    """(hit point, direction) of the chief ray where it leaves ``surface_id``, global."""

    path = run.result.paths[0]
    for k, segment in enumerate(path.segments):
        if segment.end_surface_id == surface_id:
            leaving = path.segments[k + 1]
            d = np.subtract(leaving.end_mm, leaving.start_mm)
            return np.array(leaving.start_mm), d / np.linalg.norm(d)
    raise AssertionError(f"the chief ray does not reach {surface_id}")


def local_of(point: np.ndarray) -> tuple[float, float, float]:
    """Global (X, Y, Z) as the element's (x, y, z): x = Y, y = Z, z = X."""

    return float(point[1]), float(point[2]), float(point[0])


# ---------------------------------------------------------------------------------------------- the published tables


def noll_published(j: int, rho: float, t: float) -> float:
    """R. J. Noll, J. Opt. Soc. Am. 66, 207 (1976), Table I, typed from the paper's formulas."""

    table = {
        4: math.sqrt(3) * (2 * rho ** 2 - 1), 5: math.sqrt(6) * rho ** 2 * math.sin(2 * t), 6: math.sqrt(6) * rho ** 2 * math.cos(2 * t),
        7: math.sqrt(8) * (3 * rho ** 3 - 2 * rho) * math.sin(t), 8: math.sqrt(8) * (3 * rho ** 3 - 2 * rho) * math.cos(t),
        9: math.sqrt(8) * rho ** 3 * math.sin(3 * t), 10: math.sqrt(8) * rho ** 3 * math.cos(3 * t),
        11: math.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1),
    }
    return table[j]


def fringe_published(j: int, rho: float, t: float) -> float:
    """The 37-term Fringe polynomials (peak 1), typed from the standard table."""

    r, c, s = rho, math.cos, math.sin
    table = {
        4: 2 * r ** 2 - 1, 5: r ** 2 * c(2 * t), 6: r ** 2 * s(2 * t), 7: (3 * r ** 2 - 2) * r * c(t), 8: (3 * r ** 2 - 2) * r * s(t),
        9: 6 * r ** 4 - 6 * r ** 2 + 1, 10: r ** 3 * c(3 * t), 11: r ** 3 * s(3 * t), 12: (4 * r ** 2 - 3) * r ** 2 * c(2 * t),
        13: (4 * r ** 2 - 3) * r ** 2 * s(2 * t), 14: (10 * r ** 4 - 12 * r ** 2 + 3) * r * c(t), 15: (10 * r ** 4 - 12 * r ** 2 + 3) * r * s(t),
        16: 20 * r ** 6 - 30 * r ** 4 + 12 * r ** 2 - 1, 17: r ** 4 * c(4 * t), 18: r ** 4 * s(4 * t), 19: (5 * r ** 2 - 4) * r ** 3 * c(3 * t),
        20: (5 * r ** 2 - 4) * r ** 3 * s(3 * t), 21: (15 * r ** 4 - 20 * r ** 2 + 6) * r ** 2 * c(2 * t),
        22: (15 * r ** 4 - 20 * r ** 2 + 6) * r ** 2 * s(2 * t), 23: (35 * r ** 6 - 60 * r ** 4 + 30 * r ** 2 - 4) * r * c(t),
        24: (35 * r ** 6 - 60 * r ** 4 + 30 * r ** 2 - 4) * r * s(t), 25: 70 * r ** 8 - 140 * r ** 6 + 90 * r ** 4 - 20 * r ** 2 + 1,
        26: r ** 5 * c(5 * t), 27: r ** 5 * s(5 * t), 28: (6 * r ** 2 - 5) * r ** 4 * c(4 * t), 29: (6 * r ** 2 - 5) * r ** 4 * s(4 * t),
        30: (21 * r ** 4 - 30 * r ** 2 + 10) * r ** 3 * c(3 * t), 31: (21 * r ** 4 - 30 * r ** 2 + 10) * r ** 3 * s(3 * t),
        32: (56 * r ** 6 - 105 * r ** 4 + 60 * r ** 2 - 10) * r ** 2 * c(2 * t), 33: (56 * r ** 6 - 105 * r ** 4 + 60 * r ** 2 - 10) * r ** 2 * s(2 * t),
        34: (126 * r ** 8 - 280 * r ** 6 + 210 * r ** 4 - 60 * r ** 2 + 5) * r * c(t), 35: (126 * r ** 8 - 280 * r ** 6 + 210 * r ** 4 - 60 * r ** 2 + 5) * r * s(t),
        36: 252 * r ** 10 - 630 * r ** 8 + 560 * r ** 6 - 210 * r ** 4 + 30 * r ** 2 - 1,
        37: 924 * r ** 12 - 2772 * r ** 10 + 3150 * r ** 8 - 1680 * r ** 6 + 420 * r ** 4 - 42 * r ** 2 + 1,
    }
    return table[j]


SAMPLE_POINTS = [(rho, t) for rho in (0.0, 0.3, 0.71, 1.0) for t in (0.0, 0.4, 1.3, 2.9, 4.4, 5.9)]


def evaluate(indexing: str, normalization: str, j: int, rho: float, t: float, radius: float = 4.0) -> float:
    shape = type("Shape", (), {"kind": "ZERNIKE", "indexing": indexing, "normalization": normalization, "normalization_radius_mm": radius,
                               "terms": [type("Term", (), {"index": j, "value_mm": 1.0})()]})()
    return ss.sag_value(ss.sag_form(shape), radius * rho * math.cos(t), radius * rho * math.sin(t))[0]


@pytest.mark.parametrize("j", range(4, 12))
def test_noll_z4_to_z11_match_the_published_table(j: int) -> None:
    for rho, t in SAMPLE_POINTS:
        assert evaluate("NOLL", "RMS_ORTHONORMAL", j, rho, t) == pytest.approx(noll_published(j, rho, t), abs=1e-13)


@pytest.mark.parametrize("j", range(4, 38))
def test_fringe_terms_match_the_published_table(j: int) -> None:
    for rho, t in SAMPLE_POINTS:
        assert evaluate("FRINGE", "PEAK_UNIT", j, rho, t) == pytest.approx(fringe_published(j, rho, t), abs=2e-11)


def test_fringe_and_noll_describe_the_same_polynomials_up_to_their_normalization() -> None:
    # Fringe 4 = Noll 4 / sqrt(3), Fringe 9 = Noll 11 / sqrt(5), Fringe 5 = Noll 6 / sqrt(6), Fringe 6 = Noll 5 / sqrt(6)
    for fringe, noll_index, factor in ((4, 4, 3), (9, 11, 5), (5, 6, 6), (6, 5, 6), (7, 8, 8), (8, 7, 8), (10, 10, 8), (11, 9, 8)):
        for rho, t in SAMPLE_POINTS:
            assert evaluate("FRINGE", "PEAK_UNIT", fringe, rho, t) * math.sqrt(factor) == pytest.approx(
                evaluate("NOLL", "RMS_ORTHONORMAL", noll_index, rho, t), abs=1e-13)


def _disc_moment(i: int, j: int) -> Fraction:
    """``(1 / pi) integral x^i y^j dA`` over the unit disc, exactly."""

    if i % 2 or j % 2:
        return Fraction(0)

    def half_gamma(k: int) -> Fraction:  # Gamma(k + 1/2) / sqrt(pi)
        return Fraction(math.factorial(2 * k), 4 ** k * math.factorial(k))

    m = (i + j) // 2
    return 2 * half_gamma(i // 2) * half_gamma(j // 2) / ((i + j + 2) * math.factorial(m))


def test_noll_terms_are_orthonormal_over_the_unit_disc_exactly() -> None:
    """With the Cartesian expansion and exact moments: ``(1 / pi) integral Z_i Z_j dA = delta_ij`` once ``N^2`` is applied."""

    terms = [ss.noll_order(j) for j in range(1, 37)]
    for a, (n1, m1) in enumerate(terms):
        for n2, m2 in terms[a:]:
            total = Fraction(0)
            for i1, j1, c1 in ss.zernike_monomials(n1, m1):
                for i2, j2, c2 in ss.zernike_monomials(n2, m2):
                    total += c1 * c2 * _disc_moment(i1 + i2, j1 + j2)
            expected = Fraction(1) if (n1, m1) == (n2, m2) else Fraction(0)
            assert total * ss.zernike_norm_squared("RMS_ORTHONORMAL", n1, m1) == expected


def test_the_highest_radial_order_stays_accurate_in_binary64() -> None:
    """Noll 211 .. 231 (radial order 20, the largest monomial coefficients of the expansion, about 2e6): the Cartesian form
    against the factorial series in floats, to 1e-8 of a unit coefficient (fixed before running)."""

    for j in range(211, 232):
        n, m = ss.noll_order(j)
        assert n == 20
        size = abs(m)
        for rho in (0.2, 0.5, 0.9, 1.0):
            radial = sum((-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + size) // 2 - s) * math.factorial((n - size) // 2 - s))
                         * rho ** (n - 2 * s) for s in range((n - size) // 2 + 1))
            t = 0.3
            angular = 1.0 if m == 0 else (math.cos(size * t) if m > 0 else math.sin(size * t))
            expected = math.sqrt(ss.zernike_norm_squared("RMS_ORTHONORMAL", n, m)) * radial * angular
            assert evaluate("NOLL", "RMS_ORTHONORMAL", j, rho, t) == pytest.approx(expected, abs=1e-8)


# ------------------------------------------------------------------------------------------------- the three twins


def test_float_array_and_decimal_kernels_agree() -> None:
    shape = PrescriptionSurfaceV1(surface_id="S1", vertex_z_mm="0", sag=noll("5", Z4="0.02", Z6="-0.01", Z7="0.003", Z11="0.0004", Z22="0.0001"),
                                  clear_radius_mm="5", traced="TRANSMITTED", status="SPECIFIED").sag
    form = ss.sag_form(shape)
    xs = np.array([0.0, 1.0, -2.5, 3.1, 4.0])
    ys = np.array([0.0, -2.0, 1.25, 3.0, -2.2])
    z_array, _ = ss.sag_value(form, xs, ys, ff.ARRAY_OPS)
    with localcontext() as context:
        context.prec = 40
        decimal_form = ss.sag_form(shape, ss.DECIMAL_OPS)
        for k, (x, y) in enumerate(zip(xs, ys)):
            z_float, _ = ss.sag_value(form, float(x), float(y))
            z_decimal, _ = ss.sag_value(decimal_form, Decimal(float(x)), Decimal(float(y)), ss.DECIMAL_OPS)
            assert z_float == float(z_array[k])
            assert float(z_decimal) == pytest.approx(z_float, abs=1e-15)


# ------------------------------------------------------------------------------------- FS-1: the independent reference


FS1_RAYS = ((0.0, 0.0), (0.5, 1.0), (-3.0, 2.0), (4.0, -4.0), (-5.5, -6.0), (7.0, 3.5), (0.0, 8.5))
#: Tilted rays: (x, y) at the source plane and the tilt about the global Z axis (the element's local y): (sin, cos of half).
FS1_TILT_HALF = (0.01, math.sqrt(1.0 - 0.01 * 0.01))
FS1_TILTED_RAYS = ((-4.0, 0.0), (-6.5, 3.0), (-1.5, -5.0))


def _assert_fs1(model: dict, reference_surfaces: list, rays=FS1_RAYS, *, plane_at: float = PLANE_AT) -> None:
    sin_half, cos_half = FS1_TILT_HALF
    sin_t, cos_t = 2.0 * sin_half * cos_half, cos_half * cos_half - sin_half * sin_half
    for lateral, height, tilted in [*((a, b, False) for a, b in rays), *((a, b, True) for a, b in FS1_TILTED_RAYS)]:
        document = lens_project(model, height_mm=height, lateral_mm=lateral)
        direction = ("0", "0", "1")
        if tilted:  # turned about the global Z axis: the ray climbs toward +Y (the element's +x)
            document["components"][0]["nominal_pose"]["quaternion_wxyz"] = list(quat_multiply((cos_half, 0.0, 0.0, sin_half), ALONG_X))
            direction = (repr(sin_t), "0", repr(cos_t))
        run = run_optical_project(document)
        assert run.model_scope_issues == ()
        hit, d = chief_after(run)
        reference = trace_through(reference_surfaces, (str(lateral), str(height), "-200"), direction)
        rx, ry, rz = (float(v) for v in reference["hits"][-1])
        dx, dy, dz = (float(v) for v in reference["directions"][-1])
        x, y, z = local_of(hit)
        lx, ly, lz = local_of(d)
        position = max(abs(x - rx), abs(y - ry), abs(z - rz))
        turned = max(abs(lx - dx), abs(ly - dy), abs(lz - dz))
        assert position <= FS1_POSITION_TOLERANCE_MM, (lateral, height, position)
        assert turned <= FS1_DIRECTION_TOLERANCE, (lateral, height, turned)
        # where the chief ray meets the observation plane
        ex, ey, ez = local_of(np.array(run.result.observations[0].arrivals[0].beam.position_mm))
        s = (plane_at - rz) / dz
        assert abs(ez - plane_at) < 1e-9
        assert max(abs(ex - (rx + s * dx)), abs(ey - (ry + s * dy))) <= FS1_POSITION_TOLERANCE_MM


def _front_and(sag, index: float) -> list:
    return [(lambda x, y: Decimal(0), Decimal(0), Decimal(repr(index))), (sag, Decimal(THICKNESS), Decimal(1))]


def test_fs1_zernike_lens_matches_the_decimal_reference() -> None:
    terms = {"Z4": "0.02", "Z5": "0.008", "Z6": "-0.015", "Z7": "0.004", "Z8": "-0.006", "Z9": "0.002", "Z11": "0.001", "Z22": "-0.0003"}
    model = lens_model({"radius": "-80", "conic": "-0.5", "sag": noll("9.5", **terms)})
    sag = zernike_surface("-80", "-0.5", "9.5", {int(k[1:]): v for k, v in terms.items()})
    _assert_fs1(model, _front_and(sag, nbk7_index()))


def test_fs1_xy_polynomial_lens_matches_the_decimal_reference() -> None:
    terms = {(2, 0): "0.15", (0, 2): "-0.1", (1, 1): "0.06", (3, 0): "0.01", (1, 2): "-0.02", (4, 0): "0.004", (2, 3): "0.002"}
    shape = {"kind": "XY_POLYNOMIAL", "normalization_radius_mm": "9",
             "terms": [{"x_power": i, "y_power": j, "value_mm": v} for (i, j), v in terms.items()]}
    model = lens_model({"radius": "-120", "sag": shape})
    _assert_fs1(model, _front_and(xy_surface("-120", None, "9", terms), nbk7_index()))


def test_fs1_biconic_and_toroid_lenses_match_the_decimal_reference() -> None:
    biconic = {"kind": "BICONIC", "radius_x_mm": "-60", "radius_y_mm": "-110", "conic_x": "-0.7", "conic_y": "0.4"}
    _assert_fs1(lens_model({"sag": biconic}), _front_and(biconic_surface("-60", "-110", "-0.7", "0.4"), nbk7_index()))
    toroid = {"kind": "TOROIDAL", "radius_y_mm": "-90", "conic": "-0.3", "y_terms": ["0.0002"], "radius_of_rotation_mm": "-70"}
    _assert_fs1(lens_model({"sag": toroid}), _front_and(toroid_surface("-90", "-0.3", "0.0002", "-70"), nbk7_index()))


def test_the_reference_notices_a_wrong_term() -> None:
    """The comparison has teeth: one coefficient off by 1e-8 mm in the reference moves the exit direction beyond the tolerance."""

    model = lens_model({"radius": "-80", "sag": noll("9.5", Z8="0.006")})
    run = run_optical_project(lens_project(model, height_mm=8.0, lateral_mm=2.0))
    _, d = chief_after(run)
    wrong = trace_through(_front_and(zernike_surface("-80", None, "9.5", {8: "0.00600001"}), nbk7_index()), ("2", "8", "-200"), ("0", "0", "1"))
    assert max(abs(local_of(d)[k] - float(wrong["directions"][-1][k])) for k in range(3)) > FS1_DIRECTION_TOLERANCE


# ------------------------------------------------------------------------------------------------------- limits


def _rays_on(model: dict, rays=FS1_RAYS) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for lateral, height in rays:
        out.append(chief_after(run_optical_project(lens_project(model, height_mm=height, lateral_mm=lateral))))
    return out


def _assert_same_rays(a: dict, b: dict, *, tolerance: float = 1e-12) -> None:
    for (hit_a, d_a), (hit_b, d_b) in zip(_rays_on(a), _rays_on(b), strict=True):
        assert np.max(np.abs(hit_a - hit_b)) <= tolerance
        assert np.max(np.abs(d_a - d_b)) <= tolerance


def test_a_biconic_of_equal_sections_traces_as_the_conic() -> None:
    biconic = lens_model({"sag": {"kind": "BICONIC", "radius_x_mm": "-70", "radius_y_mm": "-70", "conic_x": "-0.8", "conic_y": "-0.8"}})
    conic = lens_model({"radius": "-70", "conic": "-0.8"})
    _assert_same_rays(biconic, conic)


def test_an_xy_x2_plus_y2_traces_as_the_paraboloid() -> None:
    # z = v (x^2 + y^2) / R^2 = c (x^2 + y^2) / 2: R = 9, v = -0.5: c = -1 / 81 * ... radius = -R^2 / (2 |v|) = -81 mm
    xy = lens_model({"sag": {"kind": "XY_POLYNOMIAL", "normalization_radius_mm": "9",
                             "terms": [{"x_power": 2, "y_power": 0, "value_mm": "-0.5"}, {"x_power": 0, "y_power": 2, "value_mm": "-0.5"}]}})
    paraboloid = lens_model({"radius": "-81", "conic": "-1"})
    _assert_same_rays(xy, paraboloid)


def test_a_toroid_of_equal_radii_is_the_sphere_and_without_rotation_the_cylinder() -> None:
    toroid = lens_model({"sag": {"kind": "TOROIDAL", "radius_y_mm": "-60", "radius_of_rotation_mm": "-60"}})
    # a posed sag surface with no conic: the same sphere as a conic profile k = 0 (compile lowers a plain sphere differently)
    sphere = lens_model({"radius": "-60"})
    _assert_same_rays(toroid, sphere, tolerance=2e-11)
    extruded = lens_model({"sag": {"kind": "TOROIDAL", "radius_y_mm": "-60", "conic": "-0.2"}})
    cylinder = lens_model({"radius": "-60", "conic": "-0.2", "cylinder_axis": "X"})
    _assert_same_rays(extruded, cylinder)


def test_astigmatism_term_gives_the_line_foci_of_its_section_powers() -> None:
    """Independent derivation. Rear surface of a plano lens: ``z = c (x^2 + y^2) / 2 + sqrt(6) v (x^2 - y^2) / R^2 + ...``, so the
    section curvatures at the vertex are ``c_x = c + 2 sqrt(6) v / R^2``, ``c_y = c - 2 sqrt(6) v / R^2`` and, from inside glass
    of index n into air, the powers ``phi = (1 - n) c``. A collimated beam comes to line foci ``1 / phi_x`` and ``1 / phi_y``
    behind the vertex, separated by ``1 / phi_y - 1 / phi_x``; tolerance 1e-4 mm at ray height 0.02 mm (the aberration is
    of order ``h^2 / f``, 3e-6 mm)."""

    n = nbk7_index()
    c, v, radius = -1.0 / 60.0, 0.01, 9.5
    delta = 2.0 * math.sqrt(6.0) * v / (radius * radius)
    f_x, f_y = 1.0 / ((1.0 - n) * (c + delta)), 1.0 / ((1.0 - n) * (c - delta))
    model = lens_model({"radius": "-60", "sag": noll("9.5", Z6="0.01")}, thickness="5")
    h = 0.02
    for lateral, height, focus in ((h, 0.0, f_x), (0.0, h, f_y)):
        hit, d = chief_after(run_optical_project(lens_project(model, height_mm=height, lateral_mm=lateral)))
        x, y, z = local_of(hit)
        lx, ly, lz = local_of(d)
        across = x if lateral else y
        slope = lx if lateral else ly
        assert (z - THICKNESS) - across * lz / slope == pytest.approx(focus, abs=1e-4)
    assert abs(f_y - f_x) > 3.0  # the separation is well above the tolerance
    # and the Gaussian beam model sees the same two sections: q through the section's reduced ABCD, by hand
    waist = 0.3
    run = run_optical_project(lens_project(model, waist_mm=waist))
    gaussian = run.result.observations[0].arrivals[0].beam.gaussian
    assert gaussian.valid
    q_in = complex(0.0, math.pi * waist * waist / (WAVELENGTH_NM * 1e-6))
    for q_engine, focal in ((gaussian.q_x_mm, f_x), (gaussian.q_y_mm, f_y)):
        q = q_in + 200.0 + THICKNESS / n  # the air to the plane front, the glass in its reduced length
        q = q / (1.0 - q / focal)  # the rear surface as a thin lens of the section's power
        q = q + (PLANE_AT - THICKNESS)
        assert complex(*q_engine) == pytest.approx(q, abs=1e-6)


# ---------------------------------------------------------------------------------------- derivatives and the frame


def _surface(shape, *, c: float = 0.0, k: float = 0.0, coefficients=()) -> object:
    from os_optics.scene.lower import lower_scene  # noqa: F401
    from os_optics.scene.lower import _lower_sag
    from os_optics.math.pose import Pose
    from os_optics.scene.model import InteractionG

    surf = SagSurface(surface_id="S", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), curvature_per_mm=c, conic=k,
                      aspheric_coefficients=coefficients, aperture=CircularAperture(radius_mm=6.0), interaction=RefractInteraction(), sag_shape=shape)
    return _lower_sag("E", surf, Pose(np.eye(3), np.zeros(3)), NumericalConfig(), InteractionG("refract"), "here")


SHAPES = {
    "zernike": (dict(indexing="NOLL", normalization="RMS_ORTHONORMAL", normalization_radius_mm=6.0,
                     terms=(ZernikeSagTerm(index=4, value_mm=0.02), ZernikeSagTerm(index=5, value_mm=-0.01), ZernikeSagTerm(index=8, value_mm=0.005),
                            ZernikeSagTerm(index=11, value_mm=-0.003))), ZernikeSag, {"c": 1.0 / 50.0, "k": -0.4, "coefficients": (1e-5, -2e-7)}),
    "biconic": (dict(radius_x_mm=40.0, radius_y_mm=-70.0, conic_x=-0.6, conic_y=0.5), BiconicSag, {}),
}


def _shapes():
    from os_optics.contracts.optics import ToroidalSag, XyPolynomialSag, XySagTerm

    out = {name: (cls(**fields), extra) for name, (fields, cls, extra) in SHAPES.items()}
    out["xy"] = (XyPolynomialSag(normalization_radius_mm=6.0, terms=(XySagTerm(x_power=2, y_power=1, value_mm=0.01), XySagTerm(x_power=1, y_power=1, value_mm=-0.02),
                                                                     XySagTerm(x_power=4, y_power=0, value_mm=0.004))), {})
    out["toroid"] = (ToroidalSag(radius_y_mm=-45.0, conic=-0.3, y_terms=(2e-4, -3e-6), radius_of_rotation_mm=30.0), {})
    out["extruded"] = (ToroidalSag(radius_y_mm=45.0, conic=0.2), {})
    return out


POINTS = [(0.0, 0.0), (1.0, -2.0), (-3.5, 1.5), (2.0, 4.0), (-1.0, -5.0), (4.5, 2.5)]


@pytest.mark.parametrize("name", ["zernike", "biconic", "xy", "toroid", "extruded"])
def test_normals_and_curvatures_match_finite_differences(name: str) -> None:
    shape, extra = _shapes()[name]
    surf = _surface(shape, **extra)
    h = 1e-4

    def z(a: float, b: float) -> float:
        return float(ff.sag_at(surf, np.array([a]), np.array([b]))[0])

    for a, b in POINTS:
        z_a = (z(a + h, b) - z(a - h, b)) / (2 * h)
        z_b = (z(a, b + h) - z(a, b - h)) / (2 * h)
        z_aa = (z(a + h, b) - 2 * z(a, b) + z(a - h, b)) / (h * h)
        z_bb = (z(a, b + h) - 2 * z(a, b) + z(a, b - h)) / (h * h)
        z_ab = (z(a + h, b + h) - z(a + h, b - h) - z(a - h, b + h) + z(a - h, b - h)) / (4 * h * h)
        normal = local_normals(surf, np.array([a]), np.array([b]))[0]
        w = math.sqrt(1.0 + z_a * z_a + z_b * z_b)
        assert normal == pytest.approx(np.array([-z_a / w, -z_b / w, 1.0 / w]), abs=1e-8)
        # the shape operator from the finite-difference forms, with numpy's eigenvalues as the reference
        first = np.array([[1.0 + z_a * z_a, z_a * z_b], [z_a * z_b, 1.0 + z_b * z_b]])
        second = np.array([[z_aa, z_ab], [z_ab, z_bb]]) / w
        eigenvalues = np.sort(np.linalg.eigvals(np.linalg.solve(first, second)).real)
        point = np.array([a, b, z(a, b)])
        geo = principal_curvatures(surf, point)
        assert sorted((geo.k1, geo.k2)) == pytest.approx(list(eigenvalues), abs=2e-6)
        assert geo.normal == pytest.approx(normal, abs=1e-13)
        # the directions: unit, tangent, orthogonal; along dir1 the normal curvature is k1
        assert float(geo.dir1 @ geo.normal) == pytest.approx(0.0, abs=1e-13) and float(geo.dir2 @ geo.dir1) == pytest.approx(0.0, abs=1e-13)
        du, dv = geo.dir1[0], geo.dir1[1]
        curvature = (second[0, 0] * du * du + 2 * second[0, 1] * du * dv + second[1, 1] * dv * dv)  # II(t, t) with |t| = 1
        assert curvature == pytest.approx(geo.k1, abs=2e-6)


@pytest.mark.parametrize("name", ["zernike", "biconic", "xy", "toroid", "extruded"])
def test_bounds_hold_over_a_dense_grid(name: str) -> None:
    shape, extra = _shapes()[name]
    surf = _surface(shape, **extra)
    grid = np.linspace(-6.0, 6.0, 41)
    a, b = np.meshgrid(grid, grid)
    keep = a * a + b * b <= 36.0
    a, b = a[keep], b[keep]
    z = ff.sag_at(surf, a, b)
    lo, hi = sag_bounds(surf, 6.0)
    assert np.all(np.isfinite(z)) and lo - 1e-12 <= float(z.min()) and float(z.max()) <= hi + 1e-12
    h = 1e-3
    bound = ff.hessian_bound(surf, 0.0, 6.0)
    for x, y in zip(a[::25], b[::25]):
        hessian = np.empty((2, 2))
        f = lambda p, q: float(ff.sag_at(surf, np.array([p]), np.array([q]))[0])  # noqa: E731
        hessian[0, 0] = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h)
        hessian[1, 1] = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h * h)
        hessian[0, 1] = hessian[1, 0] = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)
        assert float(np.linalg.norm(hessian, 2)) <= bound * (1.0 + 1e-6) + 1e-6


# ------------------------------------------------------------------------------------------ the solver and its evidence


def _intersect_one(surf, origin, direction, cfg=None):
    return intersect(surf, np.array([origin], dtype=float), np.array([direction], dtype=float), cfg or NumericalConfig())


def test_a_grazing_ray_finds_both_crossings_of_a_bump() -> None:
    from os_optics.contracts.optics import XyPolynomialSag, XySagTerm

    # z = -0.5 (x / 6)^2 + ...: a dome; a ray at z slightly below its top, along x, crosses it twice, both between two samples
    shape = XyPolynomialSag(normalization_radius_mm=6.0, terms=(XySagTerm(x_power=2, y_power=0, value_mm=-0.5), XySagTerm(x_power=0, y_power=2, value_mm=-0.5)))
    surf = _surface(shape)
    height = -1e-4  # the crossings at x = +- 6 sqrt(2e-4) = +- 0.085 mm, 0.17 mm apart: a sample step of the 12 mm chord is 0.19 mm
    hits = _intersect_one(surf, (-20.0, 0.0, height), (1.0, 0.0, 0.0))
    assert math.isfinite(float(hits.t[0]))
    assert float(hits.u[0]) == pytest.approx(-6.0 * math.sqrt(2e-4), abs=1e-9)


def test_a_converged_run_states_its_evidence_and_a_plain_run_has_none() -> None:
    run = run_optical_project(lens_project(lens_model({"radius": "-80", "sag": noll("9.5", Z4="0.02", Z6="0.01")})))
    (profile,) = run.result.surface_profiles
    assert profile.surface_id == "FF/S2" and profile.kind == "ZERNIKE" and profile.indexing == "NOLL" and profile.normalization == "RMS_ORTHONORMAL"
    assert profile.normalization_radius_mm == 9.5 and profile.units == "mm"
    assert profile.vertex_sag_mm == pytest.approx(-0.02 * math.sqrt(3.0), abs=1e-15)
    assert profile.solved_rays > 0 and 0 < profile.max_iterations < 64 and profile.non_converged_rays == 0 and profile.max_residual_mm < 1e-10
    plain = run_optical_project(lens_project(lens_model({"radius": "-80", "conic": "-0.5"})))
    assert plain.result.surface_profiles == () and "surface_profiles" not in plain.result.model_dump(mode="json")


def test_a_ray_the_solver_does_not_converge_on_is_reported_not_taken_for_a_miss() -> None:
    from os_optics.contracts import ValidationRequest
    from os_optics.local.lowering import lower_project
    from os_optics.validation.pipeline import validate

    loaded = load_project(lens_project(lens_model({"radius": "-80", "sag": noll("9.5", Z4="0.02", Z6="0.01")})))
    request = lower_project(loaded).request
    starved = request.model_copy(update={"configuration": request.configuration.model_copy(update={"numerical": NumericalConfig(sag_max_iterations=1, sag_intersection_tol_mm=1e-30)})})
    result = validate(starved)
    (profile,) = result.surface_profiles
    assert profile.non_converged_rays > 0
    issue = next(i for i in result.issues if i.code.value == "SAG_INTERSECTION_NOT_CONVERGED")
    assert issue.status.value == "UNKNOWN" and issue.numerical_context["non_converged_rays"] == profile.non_converged_rays
    assert result.status.value != "PASS"
    assert isinstance(ValidationRequest.model_validate(starved.model_dump(mode="json")), ValidationRequest)


def test_the_result_is_deterministic() -> None:
    document = lens_project(lens_model({"radius": "-80", "sag": noll("9.5", Z4="0.02", Z6="0.01", Z8="0.003")}))
    first, second = run_optical_project(document), run_optical_project(copy.deepcopy(document))
    assert first.result.manifest.result_hash == second.result.manifest.result_hash
    assert first.result.manifest.trace_hash == second.result.manifest.trace_hash


# ---------------------------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("fields, code", [
    ({"radius": "-80", "cylinder_axis": "X", "sag": noll("9.5", Z4="0.02")}, "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"),
    ({"linear_sag_coefficient": "0.1", "sag": noll("9.5", Z4="0.02")}, "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"),
    ({"radius": "-80", "sag": {"kind": "BICONIC", "radius_x_mm": "-60", "radius_y_mm": "-70"}}, "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"),
    ({"sag": noll("9.5", Z2="0.02")}, "SAG_TERM_NOT_SUPPORTED"),
    ({"sag": noll("9.5", Z232="0.02")}, "SAG_TERM_NOT_SUPPORTED"),
    ({"sag": {**noll("9.5", Z4="0.02"), "normalization": "PEAK_UNIT"}}, "SAG_NORMALIZATION_MISMATCH"),
    ({"sag": {**noll("9.5", Z4="0.02"), "indexing": "FRINGE"}}, "SAG_NORMALIZATION_MISMATCH"),
    ({"sag": {"kind": "XY_POLYNOMIAL", "normalization_radius_mm": "9", "terms": [{"x_power": 1, "y_power": 0, "value_mm": "0.1"}]}}, "SAG_TERM_NOT_SUPPORTED"),
    ({"sag": {"kind": "XY_POLYNOMIAL", "normalization_radius_mm": "9", "terms": [{"x_power": 21, "y_power": 0, "value_mm": "0.1"}]}}, "SAG_TERM_NOT_SUPPORTED"),
    ({"sag": {"kind": "BICONIC", "radius_x_mm": "8", "radius_y_mm": "-70"}}, "SAG_OUTSIDE_DOMAIN"),
    ({"sag": {"kind": "TOROIDAL", "radius_y_mm": "-70", "radius_of_rotation_mm": "4"}}, "SAG_OUTSIDE_DOMAIN"),
])
def test_unsupported_combinations_fail_with_a_reason_code(fields: dict, code: str) -> None:
    surface_fields = {key: value for key, value in fields.items()}
    with pytest.raises(ValueError, match=code):
        PrescriptionSurfaceV1(surface_id="S2", vertex_z_mm="5", clear_radius_mm="9", traced="TRANSMITTED", status="SPECIFIED",
                              **{("radius_mm" if key == "radius" else key): value for key, value in surface_fields.items()})


@pytest.mark.parametrize("fields, code", [
    ({"cylindrical": True, "curvature_per_mm": 0.01}, "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"),
    ({"linear_sag_coefficient": 0.1}, "SAG_SHAPE_COMBINATION_NOT_SUPPORTED"),
])
def test_the_engine_refuses_the_same_combinations_with_the_same_codes(fields: dict, code: str) -> None:
    from os_optics.errors import EngineInputError
    from os_optics.math.pose import Pose
    from os_optics.scene.lower import _lower_sag
    from os_optics.scene.model import InteractionG

    shape = ZernikeSag(indexing="NOLL", normalization="RMS_ORTHONORMAL", normalization_radius_mm=6.0, terms=(ZernikeSagTerm(index=4, value_mm=0.01),))
    surf = SagSurface(surface_id="S", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), **{"curvature_per_mm": 0.0, **fields},
                      aperture=CircularAperture(radius_mm=6.0), interaction=RefractInteraction(), sag_shape=shape)
    with pytest.raises(EngineInputError) as caught:
        _lower_sag("E", surf, Pose(np.eye(3), np.zeros(3)), NumericalConfig(), InteractionG("refract"), "here")
    assert caught.value.code.value == code


def _compiled(model: dict):
    document = lens_project(model)
    loaded = load_project(document)
    return compile_declared("FF", loaded.optical_models[0], ProjectResolver(loaded))


def test_paraxial_refuses_zernike_and_xy_with_its_code_and_gives_section_powers_of_a_biconic() -> None:
    element = _compiled(lens_model({"radius": "-80", "sag": noll("9.5", Z4="0.02")}))
    with pytest.raises(ParaxialSurfaceNotSupported, match="PARAXIAL_SURFACE_NOT_SUPPORTED"):
        paraxial_matrix(element, WAVELENGTH_NM)
    with pytest.raises(ParaxialSurfaceNotSupported):
        paraxial_matrix(element, WAVELENGTH_NM, axis="X")
    biconic = _compiled(lens_model({"sag": {"kind": "BICONIC", "radius_x_mm": "-60", "radius_y_mm": "-110"}}))
    with pytest.raises(ValueError, match="section axis"):
        paraxial_matrix(biconic, WAVELENGTH_NM)
    n = nbk7_index()
    for axis, radius in (("X", -60.0), ("Y", -110.0)):
        (_, _), (c, _) = paraxial_matrix(biconic, WAVELENGTH_NM, axis=axis)
        assert c == pytest.approx(-(1.0 - n) / radius, rel=1e-12)
    toroid = _compiled(lens_model({"sag": {"kind": "TOROIDAL", "radius_y_mm": "-110", "y_terms": ["0.0002"], "radius_of_rotation_mm": "-60"}}))
    (_, _), (c_x, _) = paraxial_matrix(toroid, WAVELENGTH_NM, axis="X")
    (_, _), (c_y, _) = paraxial_matrix(toroid, WAVELENGTH_NM, axis="Y")
    assert c_x == pytest.approx(-(1.0 - n) / -60.0, rel=1e-12)
    assert c_y == pytest.approx(-(1.0 - n) * (-1.0 / 110.0 + 2.0 * 0.0002), rel=1e-12)


def test_the_paraxial_analysis_states_the_reason_code() -> None:
    document = lens_project(lens_model({"radius": "-80", "sag": noll("9.5", Z4="0.02")}))
    reply = json_api.paraxial_analysis(document, {"components": ["FF"], "wavelength_nm": WAVELENGTH_NM})
    assert reply["ok"] is False
    assert "PARAXIAL_SURFACE_NOT_SUPPORTED" in str(reply)


# ------------------------------------------------------------------------------------------- meshes, section, solids


FREEFORM_LENS = {"radius": "-60", "sag": noll("9.5", Z6="0.12", Z8="0.05", Z9="-0.04")}


def _geometry(model: dict):
    run = run_optical_project(lens_project(model))
    (element,) = [e for e in run.optical_view.elements if e.element_id == "FF"]
    return element.geometry


def test_the_mesh_lies_on_the_two_variable_profile_and_is_not_rotationally_symmetric() -> None:
    geom = _geometry(lens_model(FREEFORM_LENS))
    target = geom.surfaces[1]
    assert target.sag is not None and target.is_plane is False and target.is_spherical is False
    drawing = draw_body(geom)
    meshes = drawing.surfaces[1].meshes
    profile = sm.Profile.of(target)
    assert not profile.rotational and not profile.is_flat
    assert sm.max_deviation(meshes, profile, sm.Placement.coaxial(THICKNESS)) <= sm.CHORD_TOLERANCE_MM
    assert drawing.surfaces[1].facts["max_deviation_mm"] <= sm.CHORD_TOLERANCE_MM
    worst = 0.0
    for material, mesh in meshes:
        for x, y, z in mesh.positions:
            worst = max(worst, abs(z - THICKNESS - profile.sag(x, y)))
    assert worst < 1e-12  # every vertex on the profile
    assert abs(profile.sag(9.0, 0.0) - profile.sag(0.0, 9.0)) > 0.1  # visibly not a surface of revolution
    # analytic normals against the mesh's surface slope
    hit = profile.sag_normal(5.0, 2.0)
    h = 1e-6
    gx = (profile.sag(5.0 + h, 2.0) - profile.sag(5.0 - h, 2.0)) / (2 * h)
    gy = (profile.sag(5.0, 2.0 + h) - profile.sag(5.0, 2.0 - h)) / (2 * h)
    w = math.sqrt(1.0 + gx * gx + gy * gy)
    assert hit[1:] == pytest.approx((-gx / w, -gy / w, 1.0 / w), abs=1e-8)


def test_the_mesh_kernel_is_the_engines_profile() -> None:
    from os_optics.contracts.optics import XyPolynomialSag, XySagTerm

    shape = XyPolynomialSag(normalization_radius_mm=6.0, terms=(XySagTerm(x_power=2, y_power=1, value_mm=0.01), XySagTerm(x_power=3, y_power=0, value_mm=-0.02)))
    surf = _surface(shape, c=1.0 / 50.0, k=-0.4, coefficients=(1e-5,))
    profile = sm.Profile(curvature=1.0 / 50.0, conic=-0.4, coefficients=(1e-5,), form=ss.sag_form(shape))
    for a, b in POINTS:
        z = float(ff.sag_at(surf, np.array([a]), np.array([b]))[0])
        assert profile.sag(a, b) == pytest.approx(z, abs=1e-15)
        normal = local_normals(surf, np.array([a]), np.array([b]))[0]
        assert profile.normal(a, b) == pytest.approx(tuple(normal), abs=1e-15)


def test_a_rectangular_element_meshes_on_the_grid_within_tolerance() -> None:
    model = lens_model(FREEFORM_LENS)
    for surface_doc in model["model"]["surfaces"]:
        del surface_doc["clear_radius_mm"]
        surface_doc["clear_half_widths_mm"] = ["6", "4"]
    del model["model"]["edge_radius_mm"]
    model["model"]["edge_half_widths_mm"] = ["7", "5"]
    geom = _geometry(model)
    drawing = draw_body(geom)
    profile = sm.Profile.of(geom.surfaces[1])
    assert "chord_tolerance_met" not in drawing.surfaces[1].facts
    assert sm.max_deviation(drawing.surfaces[1].meshes, profile, sm.Placement.coaxial(THICKNESS)) <= sm.CHORD_TOLERANCE_MM


def test_a_biconic_mirror_focuses_each_section_at_half_its_radius() -> None:
    """A concave mirror (its centres of curvature on the +Z side, the light arriving from there). A ray parallel to the axis on a
    parabolic section of radius R reflects through the axis R / 2 from the vertex, exactly, for ``k = -1``."""

    coating = {"coating_id": "METAL", "kind": "METALLIC_REFLECTOR", "display_name": "metal", "aoi_reference": "AIR_EQUIVALENT",
               "bands": [{"min_nm": "400", "max_nm": "1100", "reflectance": "0.9", "transmittance": "0", "status": "DESIGN_BASIS"}],
               "valid_aoi_deg": {"min_deg": "0", "max_deg": "60", "status": "SPECIFIED"}, "basis": "Synthetic test coating."}
    shape = {"kind": "BICONIC", "radius_x_mm": "120", "radius_y_mm": "200", "conic_x": "-1", "conic_y": "-1"}
    model = {"model_id": "ff", "model": sequence([surface(1, "0", traced="REFLECTED", coating_id="METAL", sag=shape)], [])}
    backward = quat_multiply((0.0, 0.0, 0.0, 1.0), ALONG_X)  # travelling along global -X, toward the mirror's front
    for lateral, height, focus in ((3.0, 0.0, 60.0), (0.0, 3.0, 100.0)):
        document = project([source(position=(300.0, lateral, height), waist_mm=0.2, wavelength_nm=WAVELENGTH_NM, quaternion=backward),
                            declared_component("FF", "mirror", "ff")], [plane("P", (150.0, 0.0, 0.0), radius_mm=200.0)])
        run = run_optical_project(with_models(document, [model], [], [coating]))
        hit, d = chief_after(run, "FF/S1")
        off, slope = (hit[1], d[1]) if lateral else (hit[2], d[2])
        assert hit[0] - off / slope * d[0] == pytest.approx(focus, abs=1e-9)
        assert run.result.surface_profiles[0].kind == "BICONIC"


def test_a_posed_freeform_surface_traces_in_its_own_frame() -> None:
    """A SURFACE_SET mirror turned by 90 degrees about Z in the element: its (u, v) frame carries the shape."""

    coating = {"coating_id": "METAL", "kind": "METALLIC_REFLECTOR", "display_name": "metal", "aoi_reference": "AIR_EQUIVALENT",
               "bands": [{"min_nm": "400", "max_nm": "1100", "reflectance": "0.9", "transmittance": "0", "status": "DESIGN_BASIS"}],
               "valid_aoi_deg": {"min_deg": "0", "max_deg": "60", "status": "SPECIFIED"}, "basis": "Synthetic test coating."}
    shape = {"kind": "BICONIC", "radius_x_mm": "120", "radius_y_mm": "200", "conic_x": "-1", "conic_y": "-1"}
    posed = {"surface_id": "S1", "origin_mm": ["0", "0", "0"], "normal": ["0", "0", "1"], "u_axis": ["0", "1", "0"], "sag": shape, "clear_radius_mm": "9",
             "medium_front": "AIR", "medium_back": "AIR", "coating_id": "METAL", "traced": "REFLECTED", "status": "SPECIFIED"}
    model = {"model_id": "ff", "frame": {"origin": "REFLECTING_FACE_CENTER", "nominal_propagation": "REFLECTED_ABOUT_+Z", "reference_axis": "FACE_NORMAL", "symmetry": "NONE"},
             "design_geometry": {"kind": "CYLINDER", "diameter_mm": "20", "z_min_mm": "-1", "z_max_mm": "1", "status": "NOMINAL"},
             "model": {"model_kind": "SURFACE_SET", "model_version": "1", "fidelity": "SURFACE_PRESCRIPTION", "surfaces": [posed],
                       "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}, "approximations": ["IDEAL_SURFACE_FIGURE", "NOMINAL_PRESCRIPTION"], "status": "SPECIFIED"}}
    backward = quat_multiply((0.0, 0.0, 0.0, 1.0), ALONG_X)
    # u = local Y: the element's local y section has the 120 mm radius, its local x section (v = n x u = -x) the 200 mm one
    for lateral, height, focus in ((3.0, 0.0, 100.0), (0.0, 3.0, 60.0)):
        document = project([source(position=(300.0, lateral, height), waist_mm=0.2, wavelength_nm=WAVELENGTH_NM, quaternion=backward),
                            declared_component("FF", "mirror", "ff")], [plane("P", (150.0, 0.0, 0.0), radius_mm=200.0)])
        run = run_optical_project(with_models(document, [model], [], [coating]))
        hit, d = chief_after(run, "FF/S1")
        off, slope = (hit[1], d[1]) if lateral else (hit[2], d[2])
        assert hit[0] - off / slope * d[0] == pytest.approx(focus, abs=1e-9)


@pytest.mark.parametrize("name, rear", [
    ("zernike", FREEFORM_LENS),
    ("xy", {"radius": "-60", "sag": {"kind": "XY_POLYNOMIAL", "normalization_radius_mm": "9",
                                     "terms": [{"x_power": 2, "y_power": 0, "value_mm": "0.2"}, {"x_power": 0, "y_power": 2, "value_mm": "-0.2"},
                                               {"x_power": 2, "y_power": 1, "value_mm": "0.05"}]}}),
    ("biconic", {"sag": {"kind": "BICONIC", "radius_x_mm": "-30", "radius_y_mm": "-120", "conic_x": "-0.5"}}),
    ("toroid", {"sag": {"kind": "TOROIDAL", "radius_y_mm": "-120", "radius_of_rotation_mm": "-30"}}),
])
def test_rect_and_round_elements_mesh_within_tolerance_and_section_differs_between_planes(name: str, rear: dict) -> None:
    geom = _geometry(lens_model(rear))
    drawing = draw_body(geom)
    assert "chord_tolerance_met" not in drawing.surfaces[1].facts
    placed = [SectionElement("FF", "FF", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), geom)]
    xz = optical_section(placed, "FF", "XZ")
    yz = optical_section(placed, "FF", "YZ")

    def curve(section) -> list:
        (element,) = [e for e in section["elements"] if e.get("element_id") == "FF"]
        return element["surfaces"][1]["segments"][0]["points"]

    assert curve(xz) != curve(yz)
    profile = sm.Profile.of(geom.surfaces[1])
    for t, z in curve(xz):
        assert z == pytest.approx(THICKNESS + profile.sag(t, 0.0), abs=1e-5)
    for t, z in curve(yz):
        assert z == pytest.approx(THICKNESS + profile.sag(0.0, t), abs=1e-5)


def test_the_derived_body_contains_a_peak_between_the_sample_directions() -> None:
    # Z9 peaks at 30 degrees, between the sampled directions; the drawn surface and the true rim stay inside the derived body
    rear = {"radius": "-60", "sag": noll("14", Z9="0.25", Z6="0.7")}
    model = {"model_id": "ff", "model": sequence([surface(1, "0", clear="12", radius="90"), surface(2, "9", clear="12", **rear)], ["N-BK7"], edge="14")}
    geom = _geometry(model)
    profile = sm.Profile.of(geom.surfaces[1])
    peak = max(profile.sag(14 * math.cos(k * 0.001), 14 * math.sin(k * 0.001)) for k in range(6284))
    assert 9.0 + peak <= geom.z_max_mm
    assert geom.z_max_mm - (9.0 + peak) < 0.1  # and does not swell beyond the sampling estimate
    drawn = max(p[2] for _, mesh in draw_body(geom).meshes() for p in mesh.positions)
    assert drawn <= geom.z_max_mm


def test_oblique_astigmatism_is_the_same_gaussian_beam_turned_by_45_degrees() -> None:
    """Z5 is Z6 turned by 45 degrees: a round beam comes out with the same two q values. An elliptical beam meets its principal
    axes turned from its own: general astigmatism, which the Gaussian model reports instead of tracing."""

    z5 = run_optical_project(lens_project(lens_model({"radius": "-60", "sag": noll("9.5", Z5="0.01")}), waist_mm=0.3)).result
    z6 = run_optical_project(lens_project(lens_model({"radius": "-60", "sag": noll("9.5", Z6="0.01")}), waist_mm=0.3)).result
    g5, g6 = z5.observations[0].arrivals[0].beam.gaussian, z6.observations[0].arrivals[0].beam.gaussian
    assert g5.valid and g6.valid
    for a, b in zip(sorted((tuple(g5.q_x_mm), tuple(g5.q_y_mm))), sorted((tuple(g6.q_x_mm), tuple(g6.q_y_mm)))):
        assert a == pytest.approx(b, abs=1e-9)
    # the beam made astigmatic along X and Y by a biconic front, then a surface whose principal axes are at 45 degrees
    crossed = lens_model({"radius": "-60", "sag": noll("9.5", Z5="0.01")}, {"sag": {"kind": "BICONIC", "radius_x_mm": "200", "radius_y_mm": "400"}})
    run = run_optical_project(lens_project(crossed, waist_mm=0.3))
    assert run.result.observations[0].arrivals[0].beam.gaussian.reason == "GAUSSIAN_TWISTED_ASTIGMATISM"


def test_the_design_body_contains_the_freeform_surface_and_the_solid_check_passes() -> None:
    document = lens_project(lens_model(FREEFORM_LENS))
    report = json_api.inspect_optical_solids(document)
    assert report["ok"] is True, report
    text = str(report)
    assert "SURFACES_INTERSECT" not in text and "SAG_DOMAIN_INSIDE_OUTLINE" not in text


def test_the_solid_check_finds_a_shape_that_cuts_through_the_front_face() -> None:
    # Z6 of 2 mm against a plate 2 mm thick: the rear dips below the front off the axis in the x section
    rough = lens_model({"sag": noll("10", Z6="0.9")}, thickness="2", clear="1", edge="10")
    report = json_api.inspect_optical_solids(lens_project(rough))
    assert "SURFACES_INTERSECT" in str(report)


def test_the_glb_and_viewer_facts_carry_the_shape() -> None:
    from os_optics.optical_view.glb import _surface_extras

    geom = _geometry(lens_model(FREEFORM_LENS))
    drawing = draw_body(geom)
    extras = _surface_extras(type("E", (), {"element_id": "FF"})(), drawing.surfaces[1], drawing.tolerance_mm)
    assert extras["sag"]["kind"] == "ZERNIKE" and extras["sag"]["terms"][0] == {"index": 6, "value_mm": "0.12"}
    text = ss.sag_shape_text(extras["sag"])
    assert text.startswith("Zernike NOLL RMS_ORTHONORMAL, R 9.5 mm: Z6 0.12")


# ------------------------------------------------------------------------------------------------ prescription import


CSV_HEADER = "surface,type,radius,thickness,material,semi_diameter,conic,norm_radius,sag_terms,radius_x,conic_x,radius_of_rotation\n"


def _table(rear: str) -> str:
    return ("#! format = csv-surface-table/1\n#! units = mm\n#! wavelength_min_nm = 500\n#! wavelength_max_nm = 900\n" + CSV_HEADER
            + "S1,,inf,5,N-BK7,9,,,,,,\n" + rear + "\n"
            + "[materials]\nname,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm\n"
            + f"N-BK7,SELLMEIER,{','.join(NBK7_SELLMEIER['b'])},{','.join(NBK7_SELLMEIER['c_um2'])},400,1100\n")


@pytest.mark.parametrize("row, kind", [
    ("S2,ZERNIKE,-80,,,9,,9.5,Z4=0.02; Z6=-0.01,,,", "ZERNIKE"),
    ("S2,FZERNSAG,-80,,,9,,9.5,Z4=0.02; Z5=-0.01,,,", "ZERNIKE"),
    ("S2,XY_POLY,-80,,,9,,9,X2Y0=0.1; X0Y2=-0.1; X2Y1=0.02,,,", "XY_POLYNOMIAL"),
    ("S2,BICONIC,-110,,,9,0.4,,,-60,-0.7,", "BICONIC"),
    ("S2,TOROIDAL,-90,,,9,-0.3,,Y2=0.0002,,,-70", "TOROIDAL"),
])
def test_prescription_import_reads_every_shape(row: str, kind: str) -> None:
    from os_optics.prescription.csv_table import read_csv_surface_table

    prescription = read_csv_surface_table(_table(row))
    (front, rear) = prescription.surfaces
    assert rear.sag.kind == kind and front.sag is None
    imported = import_prescription(_table(row), model_id="imported", glass_source="inline")
    assert kind in str(imported.optical_models)


def test_prescription_import_values_are_the_native_models() -> None:
    from os_optics.prescription.csv_table import read_csv_surface_table

    rx = read_csv_surface_table(_table("S2,BICONIC,-110,,,9,0.4,,,-60,-0.7,"))
    shape = rx.surfaces[1].sag
    assert (shape.radius_x_mm, shape.radius_y_mm, shape.conic_x, shape.conic_y) == (Decimal("-60"), Decimal("-110"), Decimal("-0.7"), Decimal("0.4"))
    assert rx.surfaces[1].radius_mm is None and rx.surfaces[1].conic is None
    rx = read_csv_surface_table(_table("S2,TOROIDAL,-90,,,9,-0.3,,Y2=0.0002; Y4=-0.000001,,,-70"))
    assert rx.surfaces[1].sag.y_terms == (Decimal("0.0002"), Decimal("-0.000001")) and rx.surfaces[1].sag.radius_of_rotation_mm == Decimal("-70")


def test_prescription_import_converts_units() -> None:
    from os_optics.prescription.csv_table import read_csv_surface_table

    table = _table("S2,ZERNIKE,-8,,,0.9,,0.95,Z4=0.002,,,").replace("#! units = mm", "#! units = cm")
    rx = read_csv_surface_table(table)
    assert rx.surfaces[1].sag.normalization_radius_mm == Decimal("9.50") and rx.surfaces[1].sag.terms == ((4, Decimal("0.020")),)


@pytest.mark.parametrize("row, words", [
    ("S2,ZERNIKE,-80,,,9,,9.5,Z2=0.02,,,", "SAG_TERM_NOT_SUPPORTED"),
    ("S2,ZERNIKE,-80,,,9,,,Z4=0.02,,,", "norm_radius"),
    ("S2,ZERNIKE,-80,,,9,,9.5,,,,", "sag_terms"),
    ("S2,ZERNIKE,-80,,,9,,9.5,Q4=0.02,,,", "sag_terms"),
    ("S2,,-80,,,9,,9.5,Z4=0.02,,,", "belongs to a surface type"),
    ("S2,EXTENDED_POLY,-80,,,9,,,,,,", "a profile other than"),
])
def test_prescription_import_refuses_what_it_cannot_represent(row: str, words: str) -> None:
    with pytest.raises(PrescriptionImportError, match=words):
        import_prescription(_table(row), model_id="imported", glass_source="inline")


def test_prescription_import_takes_a_zernike_surface_beyond_its_normalization_radius() -> None:
    imported = import_prescription(_table("S2,ZERNIKE,-80,,,9,,8,Z4=0.02,,,"), model_id="imported", glass_source="inline")
    (lens,) = [m for m in imported.optical_models if m.model.model_kind == "SURFACE_SEQUENCE"]
    assert lens.model.surfaces[1].sag.normalization_radius_mm == "8" and lens.model.surfaces[1].clear_radius_mm == "9"


def _beyond(normalization: str, value: Decimal, rear_vertex: Decimal) -> dict:
    def plain(d: Decimal) -> str:
        return format(d.normalize(), "f")

    return lens_project(lens_model({"radius": "-80", "sag": noll(normalization, Z4=plain(value))}, thickness=plain(rear_vertex)), height_mm=6.0)


def test_a_zernike_surface_beyond_its_normalization_radius_is_the_stated_polynomial_there() -> None:
    """Noll Z4 = sqrt(3) (2 rho^2 - 1) is a parabola: c sqrt(3) (2 r^2 / R^2 - 1). Normalized at R = 8 mm on a 9 mm clear
    aperture (rho up to 1.125), it is the same surface as c' = c (9.5 / 8)^2 normalized at 9.5 mm, the vertex moved by
    sqrt(3) (c' - c) for the constant term. Both give the same ray (closed form, no reference to the Engine's polynomial
    code beyond the input), and the reach beyond the radius is stated on the surface's evidence only."""

    with localcontext() as ctx:
        ctx.prec = 40
        c = Decimal("0.02")
        c_wide = c * (Decimal("9.5") / Decimal(8)) ** 2
        shift = Decimal(3).sqrt() * (c_wide - c)
    beyond = run_optical_project(_beyond("8", c, Decimal(5)))
    wide = run_optical_project(_beyond("9.5", c_wide, Decimal(5) + shift))
    (profile,) = beyond.result.surface_profiles
    assert profile.normalization_radius_mm == 8.0 and profile.aperture_reach_mm == pytest.approx(9.0, abs=1e-12)
    (inside,) = wide.result.surface_profiles
    assert inside.aperture_reach_mm is None and "aperture_reach_mm" not in inside.model_dump(mode="json")
    hit, direction = chief_after(beyond)
    hit_wide, direction_wide = chief_after(wide)
    assert np.allclose(hit, hit_wide, rtol=0, atol=1e-9) and np.allclose(direction, direction_wide, rtol=0, atol=1e-9)
    # the ray met the surface at 6 mm from its axis: the sag there is the closed form of the stated terms
    a, b, z = local_of(hit)
    r2 = a * a + b * b
    expected = float(Decimal(5)) - 80.0 + math.sqrt(6400.0 - r2) + 0.02 * math.sqrt(3.0) * (2.0 * r2 / 64.0 - 1.0)
    assert z == pytest.approx(expected, abs=1e-9)
    assert not [i for i in beyond.result.issues if "NORMALIZATION" in i.code.value]
