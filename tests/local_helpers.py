"""Shared helpers of the local-workflow tests: a small committed test catalog, Project document builders, and independent
first-order optics (Sellmeier indices and paraxial ABCD matrices computed here with plain Python / NumPy from the
catalog's published numbers, never with the runtime's own compile / paraxial helpers).

The test catalog (``tests/fixtures/catalog/test_optical_catalog.v2.json``, release ``2026-01-01.1``) is a frozen copy of
the synthetic schema-2.1 development catalog: plano-convex / plano-concave N-BK7 lenses, a flat and a concave mirror, a
plate beamsplitter, a short-wave-reflecting dichroic, an N-BK7 window, a zero-order half-wave plate and a linear
polarizer. Tests pass it explicitly (``catalog=catalog_document()``), so they do not depend on whichever catalog release
the package bundles.
"""

from __future__ import annotations

import copy
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np

from os_optics.local import load_public_catalog, run_optical_project

FIXTURE_CATALOG = Path(__file__).resolve().parent / "fixtures" / "catalog" / "test_optical_catalog.v2.json"
TEST_CATALOG_VERSION = "2026-01-01.1"
_DOCUMENT = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8"))

PCX_F100 = "OS-OPT-TEST-LENS-PCX-F100"
PCX_F50 = "OS-OPT-TEST-LENS-PCX-F50"
PCV_FN50 = "OS-OPT-TEST-LENS-PCV-FN50"
FLAT_MIRROR = "OS-OPT-TEST-MIR-FLAT"
CONCAVE_MIRROR = "OS-OPT-TEST-MIR-CONCAVE-R200"
PLATE_BS = "OS-OPT-TEST-BS-PLATE"
DICHROIC = "OS-OPT-TEST-FLT-DICHROIC-SP600"
WINDOW = "OS-OPT-TEST-WIN-NBK7-T3"
HWP = "OS-OPT-TEST-WP-HWP-633"
POLARIZER = "OS-OPT-TEST-POL-LIN"

SQRT_HALF = math.sqrt(0.5)
# Local +Z -> design +X / -X / +Y (local +Y -> design +Z, level).
ALONG_X = (0.5, 0.5, 0.5, 0.5)
AGAINST_X = (0.5, 0.5, -0.5, -0.5)
ALONG_Y = (0.0, 0.0, SQRT_HALF, SQRT_HALF)
# Face normal -> normalize(-X + Y): folds a +X beam into +Y at 45 deg.
FOLD_A = math.sqrt((2.0 - math.sqrt(2.0)) / 8.0)
FOLD_B = math.sqrt((2.0 + math.sqrt(2.0)) / 8.0)
FOLD_X_TO_Y = (FOLD_A, FOLD_A, -FOLD_B, -FOLD_B)
POWER_W = 0.001
NM = 1e-6


def catalog_document() -> dict:
    """A fresh copy of the test catalog document (tests may edit it)."""

    return copy.deepcopy(_DOCUMENT)


#: The group-delay bounds (fs) of the test catalog's coating classes in ``bounded_catalog_document`` (the catalog classes'
#: published values, the catalog's coating class bounds).
GROUP_DELAY_BOUNDS_FS = {"DIELECTRIC_HIGH_REFLECTOR": "1000", "BEAMSPLITTER": "1000", "ANTI_REFLECTION": "50"}


def bounded_catalog_document(*, unbounded_kinds: tuple[str, ...] = ()) -> dict:
    """The test catalog as a schema-3.2 document whose high-reflector, splitter and anti-reflection coatings publish a
    group-delay bound (``GROUP_DELAY_BOUNDS_FS``; not the kinds in ``unbounded_kinds``)."""

    document = catalog_document()
    document["schema_version"] = "3.2"
    for coating in document["coatings"]:
        if coating["kind"] in GROUP_DELAY_BOUNDS_FS and coating["kind"] not in unbounded_kinds:
            coating["group_delay_bound_fs"] = GROUP_DELAY_BOUNDS_FS[coating["kind"]]
    return document


def with_lossless_bulk(catalog: dict) -> dict:
    """Copy a catalog for tests that prescribe unit transmission where the glass loss is unstated."""
    document = copy.deepcopy(catalog)
    document["schema_version"] = "3.6"
    for material in document["materials"]:
        if material.get("bulk_loss") is not None:
            continue
        band = material["valid_band_nm"]
        material["bulk_loss"] = {
            "kind": "INTERNAL_TRANSMITTANCE", "wavelength_nm": [band["min_nm"], band["max_nm"]],
            "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
            "basis": "Assumed lossless synthetic test fixture.",
        }
    return document


def load_test_catalog():
    return load_public_catalog(catalog_document())


def product_document(sku: str, document: dict | None = None) -> dict:
    return next(p for p in (document or _DOCUMENT)["products"] if p["optical_sku"] == sku)


def quat_multiply(a, b) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def turn_about_z(cos_half: float, sin_half: float) -> tuple[float, float, float, float]:
    """The rotation by ``theta`` about local +Z, given ``cos(theta / 2)`` and ``sin(theta / 2)``."""

    return (cos_half, 0.0, 0.0, sin_half)


def source(instance_id: str = "SRC", position=(-200.0, 0.0, 0.0), *, wavelength_nm: float = 780.0, waist_mm: float = 1.0,
           quaternion=ALONG_X, polarization_deg: float = 0.0, waist_position_mm: float = 0.0, power_w: float = POWER_W) -> dict:
    return {
        "instance_id": instance_id,
        "optical_role": "source",
        "custom_part": {"custom_part_key": "test-laser", "description": "Test laser."},
        "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)},
        "source_model": {
            "model_kind": "GAUSSIAN_BEAM", "wavelength_nm": wavelength_nm, "power_w": power_w, "waist_radius_mm": waist_mm,
            "waist_position_mm": waist_position_mm, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": polarization_deg},
        },
    }


def element(instance_id: str, role: str, sku: str, position=(0.0, 0.0, 0.0), quaternion=ALONG_X, *, catalog_version: str | None = TEST_CATALOG_VERSION,
            retained=()) -> dict:
    product = {"optical_sku": sku}
    if catalog_version is not None:
        product["catalog_version"] = catalog_version
    return {
        "instance_id": instance_id, "optical_role": role, "product": product,
        "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)},
        "retained_dofs": [{"axis": axis, "control": control} for axis, control in retained],
    }


def plane(observation_id: str, position, quaternion=ALONG_X, *, radius_mm: float = 12.5) -> dict:
    return {"observation_id": observation_id, "pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)},
            "aperture": {"shape": "circle", "radius_mm": radius_mm}}


#: The propagation environment of the test Projects: vacuum (zero pressure). The independent optics below (and the tests'
#: closed-form expectations) take the medium outside the elements to have index exactly 1; a native Project that states no
#: environment operates in air at 20 C and 101325 Pa (n ~ 1.00027), which the material-physics tests exercise explicitly.
UNIT_INDEX_ENVIRONMENT = {"pressure_pa": "0"}


def project(components: list[dict], observations: list[dict], targets: list[dict] | None = None, *,
            environment: dict | None = UNIT_INDEX_ENVIRONMENT) -> dict:
    """A Project document; ``environment``: its propagation environment (default vacuum, ``UNIT_INDEX_ENVIRONMENT``; ``None``:
    none stated, i.e. air at 20 C and 101325 Pa)."""

    document = {"schema": "opensci.project", "project_schema_version": "2.0", "components": components, "observations": observations,
                "targets": targets or []}
    if environment is not None:
        document["environment"] = dict(environment)
    return document


def run(document: dict, **kwargs):
    kwargs.setdefault("catalog", catalog_document())
    return run_optical_project(document, **kwargs)


def arrivals(result, observation_id: str, source_id: str | None = None):
    obs = next(o for o in result.observations if o.observation_id == observation_id)
    return [a for a in obs.arrivals if source_id is None or a.beam.source_id == source_id]


def scope_codes(run_) -> list[tuple[str, str, str | None]]:
    return [(i.instance_id, i.code, i.surface_id) for i in run_.model_scope_issues]


# ------------------------------------------------------------------------------------------------ independent optics


def sellmeier_index(material_id: str, wavelength_nm: float, document: dict | None = None) -> float:
    material = next(m for m in (document or _DOCUMENT)["materials"] if m["material_id"] == material_id)
    b = [float(v) for v in material["dispersion"]["b"]]
    c = [float(v) for v in material["dispersion"]["c_um2"]]
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def lens_matrix(sku: str, wavelength_nm: float, *, turned: bool = False, document: dict | None = None) -> np.ndarray:
    """Reduced paraxial ABCD of a SURFACE_SEQUENCE lens from its entrance vertex to its exit vertex (air outside);
    ``turned``: light along local -Z (the surfaces in reverse order, radii negated)."""

    model = product_document(sku, document)["simulation_models"][0]
    surfaces = model["surfaces"]
    media = ["AIR", *model["media"], "AIR"]
    steps = []
    for i, surface in enumerate(surfaces):
        radius = float(surface["radius_mm"]) if "radius_mm" in surface else math.inf
        steps.append((float(surface["vertex_z_mm"]), radius, media[i], media[i + 1]))
    if turned:
        steps = [(-z, -r, after, before) for z, r, before, after in reversed(steps)]

    def n(material: str) -> float:
        return 1.0 if material == "AIR" else sellmeier_index(material, wavelength_nm, document)

    matrix = np.eye(2)
    for k, (z, radius, before, after) in enumerate(steps):
        if k:
            gap = z - steps[k - 1][0]
            matrix = np.array([[1.0, gap / n(before)], [0.0, 1.0]]) @ matrix
        if math.isfinite(radius):
            matrix = np.array([[1.0, 0.0], [-(n(after) - n(before)) / radius, 1.0]]) @ matrix
    return matrix


def free(length: float) -> np.ndarray:
    return np.array([[1.0, length], [0.0, 1.0]])


def q_through(matrix: np.ndarray, q: complex) -> complex:
    return (matrix[0, 0] * q + matrix[0, 1]) / (matrix[1, 0] * q + matrix[1, 1])


def rayleigh_range(waist_mm: float, wavelength_nm: float) -> float:
    return math.pi * waist_mm**2 / (wavelength_nm * NM)


def waist_radius(q: complex, wavelength_nm: float) -> float:
    """Waist radius of the beam whose q (in air) is given: ``w0 = sqrt(lambda z_R / pi)``."""

    return math.sqrt(wavelength_nm * NM * q.imag / math.pi)


def thickness(sku: str) -> float:
    surfaces = product_document(sku)["simulation_models"][0]["surfaces"]
    return float(Decimal(surfaces[-1]["vertex_z_mm"]) - Decimal(surfaces[0]["vertex_z_mm"]))
