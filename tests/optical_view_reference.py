"""Shared fixtures of the Optical Simulation View tests (scene schema 2.0).

Designs are assembled from the repository's seed public optical catalog release (``REAL_CATALOG``) **by query** (category, form, wavelength and published
specifications), never by SKU, so the same tests hold for any catalog release: the smallest ``optical_sku`` that fits
the query is taken. The optical elements are ordinary catalog products lowered by the real local runtime
(``os_optics.local.lower_project``, which compiles every product with ``os_optics.local.compile``); the Engine
result and ray paths come from one ``validate_with_ray_paths`` execution.

* ``reference_project``: a user-defined collimated 633 nm Gaussian source (1 mm waist radius, 1 mW, explicit
  ``source_model``) on the design +X axis, the first converging catalog lens valid at 633 nm, a target plane ("focal",
  a power target) at the lens's specified focal length and a screen beyond it. Illustrative beam line only: no optical
  height, mount, post, holder, base or breadboard exists at this stage.
* ``folded_project``: a multi-element design at 633 nm: two converging lenses (a telescope), a flat fold mirror at 45 deg,
  a window, a plate beamsplitter at 45 deg (both branches), a thin filter that transmits 633 nm on the transmitted
  branch, and a screen on each branch. A plate splitter shifts (a thick plate) and deviates (a wedged plate) its
  transmitted beam, so the filter and the screen are centred on the transmitted chief ray the Engine traces through the
  chosen splitter;
* ``polarization_project``: a waveplate and a linear polarizer on a collimated beam, at the first common laser line
  (``POLARIZATION_WAVELENGTHS_NM``) at which the catalog publishes both.

``rigidly_moved`` re-poses a whole design (every component and free observation plane) by one rigid transform. Import
the fixtures into a test module with ``from optical_view_reference import catalog, project, engine_request, engine_run,
engine_result, ray_paths, scene``. Quaternions use exact literals or ``math.sqrt`` only (no libm trig).
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from os_optics import trace_ray_paths, validate_with_ray_paths
from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics._vendor.shared_contracts.design_catalog import OpticalCategory, OpticalForm, PublicOpticalCatalogV2, PublicOpticalProductV2
from os_optics._vendor.shared_contracts.optical_design.design_content import CircularApertureV1, ObservationV1, TargetV1
from os_optics._vendor.shared_contracts.optical_design import (
    OpticalComponentInstanceV1,
    OpticalProductRefV1,
    OpticalRoleV1,
    ProjectDocumentV2,
)
from os_optics.local import load_public_catalog, lower_project

from catalog_store_helpers import REAL_CATALOG
from os_optics.optical_view import build_optical_simulation_scene
from os_optics.optical_view import geometry as g

DESIGN_WAVELENGTH_NM = "633"
SOURCE_X_MM = -100.0
SOURCE_WAIST_MM = 1.0

# Exact literals (= sqrt(0.5) correctly rounded), never libm trig.
SQRT_HALF = 0.7071067811865476
BEAM_ALONG_X = (SQRT_HALF, 0.0, SQRT_HALF, 0.0)  # local +Z -> design +X, local +X -> design -Z
BEAM_ALONG_Y = (SQRT_HALF, -SQRT_HALF, 0.0, 0.0)  # local +Z -> design +Y, local +X -> design +X


def frame_quaternion(z_axis: g.Vec3, y_axis: g.Vec3) -> tuple[float, float, float, float]:
    """The unit quaternion turning local +Z / +Y onto the given orthonormal design directions (sqrt only)."""
    z, y = g.unit(z_axis), g.unit(y_axis)
    return g.matrix_to_quat((g.cross(y, z), y, z))


# A face-normal element folding a beam that travels along +X onto +Y: face normal (-1, 1, 0)/sqrt(2), local +Y up.
FOLD_X_TO_Y = frame_quaternion((-1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
# A 45 deg splitter on a beam travelling along +Y: face normal (1, -1, 0)/sqrt(2); reflects to +X, transmits along +Y.
SPLIT_Y_TO_PLUS_X = frame_quaternion((1.0, -1.0, 0.0), (0.0, 0.0, 1.0))
FACING_PLUS_X = frame_quaternion((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def public_catalog() -> PublicOpticalCatalogV2:
    """The repository's seed release: the multi-element designs need its filters, splitters and waveplates."""
    return load_public_catalog(REAL_CATALOG)


def spec(product: PublicOpticalProductV2, name: str) -> Decimal | None:
    value = getattr(product.specifications, name)
    return None if value is None else Decimal(value.value)


def find_product(catalog: PublicOpticalCatalogV2, category: OpticalCategory, *, form: OpticalForm | None = None, predicate=None,
                 exclude: tuple[str, ...] = (), wavelength_nm: str = DESIGN_WAVELENGTH_NM) -> PublicOpticalProductV2 | None:
    """The smallest-SKU product of ``category`` (and ``form``) with a model valid at ``wavelength_nm`` that satisfies
    ``predicate`` (catalog order is by SKU)."""
    for product in catalog.select(category=category, form=form, wavelength_nm=wavelength_nm):
        if product.optical_sku not in exclude and (predicate is None or predicate(product)):
            return product
    return None


def converging_lens(catalog: PublicOpticalCatalogV2, *, exclude: tuple[str, ...] = (), max_focal_mm: str | None = None) -> PublicOpticalProductV2:
    def ok(product: PublicOpticalProductV2) -> bool:
        f = spec(product, "focal_length_mm")
        return f is not None and f > 0 and (max_focal_mm is None or f <= Decimal(max_focal_mm))

    product = find_product(catalog, OpticalCategory.LENS, predicate=ok, exclude=exclude)
    if product is None:
        pytest.skip(f"the seed catalog has no converging lens at {DESIGN_WAVELENGTH_NM} nm")
    return product


def required(product: PublicOpticalProductV2 | None, what: str) -> PublicOpticalProductV2:
    if product is None:
        pytest.skip(f"the seed catalog has no {what} at {DESIGN_WAVELENGTH_NM} nm")
    return product


def _source(x_mm: float = SOURCE_X_MM, wavelength_nm: str = DESIGN_WAVELENGTH_NM) -> OpticalComponentInstanceV1:
    return OpticalComponentInstanceV1.model_validate({
        "instance_id": "SRC",
        "optical_role": "source",
        "custom_part": {"custom_part_key": "user-collimated-laser", "description": "User-supplied collimated laser source."},
        "nominal_pose": {"position_mm": (x_mm, 0.0, 0.0), "quaternion_wxyz": BEAM_ALONG_X},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": float(wavelength_nm), "power_w": 0.001,
                         "waist_radius_mm": SOURCE_WAIST_MM, "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}},
    })


def _part(instance_id: str, role: OpticalRoleV1, product: PublicOpticalProductV2, position, q) -> OpticalComponentInstanceV1:
    return OpticalComponentInstanceV1(instance_id=instance_id, optical_role=role, product=OpticalProductRefV1(optical_sku=product.optical_sku),
                                      nominal_pose=PoseV1(position_mm=position, quaternion_wxyz=q))


def _plane(observation_id: str, position, q, radius: float) -> ObservationV1:
    return ObservationV1(observation_id=observation_id, pose=PoseV1(position_mm=position, quaternion_wxyz=q),
                         aperture=CircularApertureV1(radius_mm=radius))


def reference_lens(catalog: PublicOpticalCatalogV2) -> PublicOpticalProductV2:
    return converging_lens(catalog, max_focal_mm="200")


def focal_plane_x_mm(catalog: PublicOpticalCatalogV2) -> float:
    """The target plane: the lens's specified focal length behind its first vertex (a display position, not a focus claim)."""
    return float(spec(reference_lens(catalog), "focal_length_mm"))


def reference_project(catalog: PublicOpticalCatalogV2 | None = None) -> ProjectDocumentV2:
    catalog = catalog or public_catalog()
    lens = reference_lens(catalog)
    focal_x = focal_plane_x_mm(catalog)
    return ProjectDocumentV2.model_validate(ProjectDocumentV2(
        schema="opensci.project",
        project_schema_version="2.0",
        components=[_source(), _part("L1", OpticalRoleV1.LENS, lens, (0.0, 0.0, 0.0), BEAM_ALONG_X)],
        observations=[_plane("focal", (focal_x, 0.0, 0.0), BEAM_ALONG_X, 12.5), _plane("screen", (2 * focal_x + 50.0, 0.0, 0.0), BEAM_ALONG_X, 25.0)],
        targets=[TargetV1(target_id="power_at_focal", metric="power_W", observation_id="focal", minimum=1e-9, unit="W")],
    ).payload())


# A filter on the transmitted branch passes at least this fraction at the design wavelength on each of its coated faces.
FILTER_MIN_TRANSMITTANCE = Decimal("0.5")


def transmits_design_wavelength(catalog: PublicOpticalCatalogV2, product: PublicOpticalProductV2) -> bool:
    """Every coated surface of the product's first model valid at ``DESIGN_WAVELENGTH_NM`` transmits at least
    ``FILTER_MIN_TRANSMITTANCE`` there (a filter whose spectrum covers the line but blocks it is not a transmissive part)."""
    wavelength = Decimal(DESIGN_WAVELENGTH_NM)
    model = next(m for m in product.simulation_models if m.valid_band_nm.contains(wavelength))
    for surface in getattr(model, "surfaces", ()) or ():
        if surface.coating_id:
            band = catalog.coating(surface.coating_id).band_at(wavelength)
            if band is None or Decimal(band.transmittance) < FILTER_MIN_TRANSMITTANCE:
                return False
    return True


def folded_parts(catalog: PublicOpticalCatalogV2) -> dict[str, PublicOpticalProductV2]:
    """The catalog products of ``folded_project`` (every category it needs, by query, at ``DESIGN_WAVELENGTH_NM``)."""
    first = converging_lens(catalog, max_focal_mm="200")
    second = converging_lens(catalog, exclude=(first.optical_sku,))
    return {
        "L1": first,
        "L2": second,
        "M1": required(find_product(catalog, OpticalCategory.MIRROR, form=OpticalForm.FLAT_MIRROR), "flat mirror"),
        "W1": required(find_product(catalog, OpticalCategory.WINDOW), "window"),
        "BS": required(find_product(catalog, OpticalCategory.BEAMSPLITTER, form=OpticalForm.PLATE_BEAMSPLITTER), "plate beamsplitter"),
        "F1": required(find_product(catalog, OpticalCategory.FILTER,
                                    predicate=lambda p: not p.frame.is_reflector and transmits_design_wavelength(catalog, p)),
                       "filter transmitting the design wavelength"),
    }


FOLDED_FILTER_Y_MM = 120.0
FOLDED_SCREEN_Y_MM = 200.0
# Layout positions computed from a traced ray are rounded to this many decimals (mm) in the Project document.
LAYOUT_DECIMALS = 6


def folded_project(catalog: PublicOpticalCatalogV2 | None = None) -> ProjectDocumentV2:
    """SRC -> L1 -> L2 (a Keplerian telescope) -> M1 folds +X onto +Y -> W1 -> BS (reflects to +X, transmits along +Y)
    -> F1 -> screen; the reflected branch (+X) reaches screen_r. No plane lies across another branch. F1 and the screen
    are centred on the splitter's transmitted chief ray (``_transmitted_x_mm``)."""
    catalog = catalog or public_catalog()
    parts = folded_parts(catalog)
    f1, f2 = float(spec(parts["L1"], "focal_length_mm")), float(spec(parts["L2"], "focal_length_mm"))
    l2_x = f1 + f2
    fold_x = l2_x + 60.0
    components = [
        _source(),
        _part("L1", OpticalRoleV1.LENS, parts["L1"], (0.0, 0.0, 0.0), BEAM_ALONG_X),
        _part("L2", OpticalRoleV1.LENS, parts["L2"], (l2_x, 0.0, 0.0), BEAM_ALONG_X),
        _part("M1", OpticalRoleV1.MIRROR, parts["M1"], (fold_x, 0.0, 0.0), FOLD_X_TO_Y),
        _part("W1", OpticalRoleV1.WINDOW, parts["W1"], (fold_x, 40.0, 0.0), BEAM_ALONG_Y),
        _part("BS", OpticalRoleV1.SPLITTER, parts["BS"], (fold_x, 80.0, 0.0), SPLIT_Y_TO_PLUS_X),
    ]
    filter_x, screen_x = _transmitted_x_mm(components, catalog, fold_x, (FOLDED_FILTER_Y_MM, FOLDED_SCREEN_Y_MM))
    return ProjectDocumentV2.model_validate(ProjectDocumentV2(
        schema="opensci.project",
        project_schema_version="2.0",
        components=[*components, _part("F1", OpticalRoleV1.FILTER, parts["F1"], (filter_x, FOLDED_FILTER_Y_MM, 0.0), BEAM_ALONG_Y)],
        observations=[
            _plane("screen", (screen_x, FOLDED_SCREEN_Y_MM, 0.0), BEAM_ALONG_Y, 25.0),
            _plane("screen_r", (fold_x + 60.0, 80.0, 0.0), FACING_PLUS_X, 25.0),
        ],
        targets=[TargetV1(target_id="power_on_screen", metric="power_W", observation_id="screen", minimum=1e-12, unit="W")],
    ).payload())


def _transmitted_x_mm(components, catalog: PublicOpticalCatalogV2, fold_x: float, ys: tuple[float, ...]) -> tuple[float, ...]:
    """Design x where the splitter's transmitted chief ray (main path, after its last ``BS`` surface) crosses each design
    y in ``ys``: one Engine trace of the design up to the splitter, with a screen beyond them."""
    layout = ProjectDocumentV2.model_validate(ProjectDocumentV2(
        schema="opensci.project", project_schema_version="2.0", components=components,
        observations=[_plane("screen", (fold_x, 2 * max(ys), 0.0), BEAM_ALONG_Y, 100.0)],
    ).payload())
    (main,) = [p for p in trace_ray_paths(lowered_request(layout, catalog)).paths if p.parent_path_id is None]
    last = max(k for k, surface in enumerate(main.chief_surface_ids) if (surface or "").startswith("BS/"))
    (x0, y0, _), (x1, y1, _) = main.chief_points_mm[last], main.chief_points_mm[last + 1]
    return tuple(round(x0 + (y - y0) * (x1 - x0) / (y1 - y0), LAYOUT_DECIMALS) for y in ys)


# Common laser lines, in the order a polarization design is tried (the first where a waveplate and a polarizer exist).
POLARIZATION_WAVELENGTHS_NM = ("633", "532", "780", "1064", "1310", "1550", "405", "850")


def polarization_parts(catalog: PublicOpticalCatalogV2) -> tuple[str, dict[str, PublicOpticalProductV2]]:
    """``(wavelength, {"WP": waveplate, "POL": linear polarizer})`` at the first common laser line with both."""
    for wavelength in POLARIZATION_WAVELENGTHS_NM:
        waveplate = find_product(catalog, OpticalCategory.WAVEPLATE, wavelength_nm=wavelength)
        polarizer = find_product(catalog, OpticalCategory.POLARIZER, wavelength_nm=wavelength)
        if waveplate is not None and polarizer is not None:
            return wavelength, {"WP": waveplate, "POL": polarizer}
    pytest.skip(f"the seed catalog has no waveplate and polarizer at a common laser line {POLARIZATION_WAVELENGTHS_NM}")


def polarization_project(catalog: PublicOpticalCatalogV2 | None = None) -> ProjectDocumentV2:
    """SRC -> WP -> POL -> screen along design +X; both elements' local +X lies along design -Z (``BEAM_ALONG_X``)."""
    catalog = catalog or public_catalog()
    wavelength, parts = polarization_parts(catalog)
    return ProjectDocumentV2.model_validate(ProjectDocumentV2(
        schema="opensci.project",
        project_schema_version="2.0",
        components=[
            _source(wavelength_nm=wavelength),
            _part("WP", OpticalRoleV1.WAVEPLATE, parts["WP"], (0.0, 0.0, 0.0), BEAM_ALONG_X),
            _part("POL", OpticalRoleV1.POLARIZER, parts["POL"], (30.0, 0.0, 0.0), BEAM_ALONG_X),
        ],
        observations=[_plane("screen", (80.0, 0.0, 0.0), BEAM_ALONG_X, 25.0)],
        targets=[TargetV1(target_id="power_on_screen", metric="power_W", observation_id="screen", minimum=1e-12, unit="W")],
    ).payload())


def lowered_request(project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2):
    """The real local-runtime Engine request of ``project``."""
    return lower_project(project, catalog).request


def build_scene(project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2, **kwargs):
    """Scene of one Engine run of ``project`` (lowered by the real runtime) and that run's request."""
    request = lowered_request(project, catalog)
    return build_optical_simulation_scene(project=project, catalog=catalog, engine_request=request, **kwargs), request


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _rotate(q, v):
    w, x, y, z = q
    p = _qmul(_qmul(q, (0.0, *v)), (w, -x, -y, -z))
    return p[1:]


def _moved(pose: PoseV1, position, q) -> PoseV1:
    p = _rotate(q, pose.position_mm)
    return PoseV1(position_mm=(p[0] + position[0], p[1] + position[1], p[2] + position[2]),
                  quaternion_wxyz=_qmul(q, pose.quaternion_wxyz))


def rigidly_moved(project: ProjectDocumentV2, position, q) -> ProjectDocumentV2:
    """The same design, re-posed by ``x -> R(q) x + position`` (device-attached planes follow their component)."""
    components = [c.model_copy(update={"nominal_pose": _moved(c.nominal_pose, position, q)}) for c in project.components]
    observations = [o if o.device_instance_id is not None else o.model_copy(update={"pose": _moved(o.pose, position, q)})
                    for o in project.observations]
    return ProjectDocumentV2.model_validate(
        project.model_copy(update={"components": tuple(components), "observations": tuple(observations)}).payload()
    )


def to_local(element, point) -> g.Vec3:
    """Global point -> element frame (inverse of x_global = R(q) x_local + p)."""
    w, x, y, z = element.pose.quaternion_wxyz
    return g.rotate((w, -x, -y, -z), g.sub(point, element.pose.position_mm))


def unit_quaternion(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


@pytest.fixture(scope="module")
def catalog():
    return public_catalog()


@pytest.fixture(scope="module")
def project(catalog):
    return reference_project(catalog)


@pytest.fixture(scope="module")
def engine_request(project, catalog):
    return lowered_request(project, catalog)


@pytest.fixture(scope="module")
def engine_run(engine_request):
    return validate_with_ray_paths(engine_request)


@pytest.fixture(scope="module")
def engine_result(engine_run):
    return engine_run[0]


@pytest.fixture(scope="module")
def ray_paths(engine_run):
    return engine_run[1]


@pytest.fixture(scope="module")
def scene(project, catalog, engine_request, engine_result, ray_paths):
    return build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                          ray_paths=ray_paths)


def element_primitives(gltf: dict, binary: bytes, element_id: str) -> dict[str, list[tuple[float, float, float]]]:
    """Element-frame vertex positions by material of an element drawn in a GLB: the element node's own mesh (its body)
    and its optical-surface child nodes (``surface:<element_id>/<surface_id>``)."""

    from os_optics.optical_view import read_accessor

    (node,) = [n for n in gltf["nodes"] if n.get("name") == f"element:{element_id}"]
    owners = [node] + [gltf["nodes"][c] for c in node.get("children", ())]
    out: dict[str, list[tuple[float, float, float]]] = {}
    for owner in owners:
        if "mesh" not in owner:
            continue
        for prim in gltf["meshes"][owner["mesh"]]["primitives"]:
            out.setdefault(gltf["materials"][prim["material"]]["name"], []).extend(read_accessor(gltf, binary, prim["attributes"]["POSITION"]))
    return out
