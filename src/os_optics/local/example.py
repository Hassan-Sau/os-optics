"""Ready-to-run reference designs as ``ProjectDocumentV2`` documents (starting points to copy and edit).

Every example picks its products by QUERY from the public catalog (default: the active catalog) -- never by a
hard-coded SKU or catalog version -- so it survives a catalog regeneration: among the products of the wanted category
and form whose single published simulation model is valid at the example's wavelength, whose coatings are published
for that wavelength and the example's angle of incidence and whose clear aperture is at least
``MIN_CLEAR_APERTURE_MM``, it takes the one whose specified focal length is closest to the wanted value (lenses), else
the first by ``optical_sku`` order. Element positions follow from the chosen products' published prescriptions by
paraxial first-order optics (``compile.paraxial_matrix``, Gaussian ``q`` propagation) and are rounded to the
micrometre, so each design is in focus / afocal / imaging by construction. A catalog without a suitable product raises
``LocalRunError`` ``EXAMPLE_UNAVAILABLE``.

Kinds (``EXAMPLE_KINDS``; ``example_project_of_kind(kind)``, ``os-optics project example --kind KIND``):

* ``focus`` -- ``example_project()``: a user-defined collimated 780 nm Gaussian source (1 mW, 1.0 mm 1/e^2 waist radius
  at the emitter, M^2 1, linear polarization along local +X) 200 mm before a plano-convex lens (focal length closest to
  100 mm, curved side toward the source); the beam runs along design +X. The lens retains Tx / Ty (MANUAL) and Tz
  (NUMERICAL). One observation sits at the predicted Gaussian waist behind the lens, with two targets: the waist within
  0.1 mm of that plane and a focal spot radius of at most 0.05 mm.
* ``folded`` -- ``example_folded_project()``: the same source, lens and targets with the beam folded by a flat mirror
  (source -> 45 deg fold mirror -> lens -> focal plane; the source-to-lens path is 200 mm, as in ``focus``).
* ``beam_expander``: a Keplerian telescope: a plano-convex lens of focal length closest to 50 mm, then one of about
  twice its focal length turned around (curved side toward the collimated output), spaced so the pair is afocal; the
  1 mm beam leaves expanded by the ratio of the two effective focal lengths (target: its waist radius within 1 %).
* ``relay_4f``: a 4f relay of two identical plano-convex lenses (focal length closest to 100 mm): a 0.1 mm waist at the
  front focal plane of the first lens is imaged onto the back focal plane of the second at unit magnification (targets:
  the waist on the image plane within 0.1 mm, its radius within 2 %). The first lens is turned around so both curved
  sides face the collimated space between the lenses.
* ``folded_expander``: the beam expander folded by a 45 deg flat mirror between the lenses (mixed reflective /
  transmissive path).
* ``dichroic_routing``: two co-located sources of different wavelengths (chosen from common laser lines inside a
  dichroic mirror's reflect and transmit bands) meet a dichroic mirror at 45 deg: one wavelength is reflected into
  design +Y, the other transmitted along +X (targets: at least half of each source's power on its own leg).
* ``two_color_dichroic``: ONE source emitting two spectral lines (``spectral_lines``: a line the dichroic mirror reflects
  at 1 mW and one it transmits at 2 mW, chosen like ``dichroic_routing``'s) meets the dichroic mirror at 45 deg; each
  line ends on the user's own detector (a ``custom_part`` ``detector`` whose absorbing observation plane is its sensitive
  face) on its leg. Targets select the line by ``wavelength_nm``: at least half of each line's power on its own
  detector.
* ``dichroic_combiner``: the time reversal of ``dichroic_routing``: the same two lines, from two lasers, are combined
  onto one beam by the same dichroic mirror at 45 deg -- the reflected line meets its coated face, the transmitted line
  its back face -- and both end on the user's own detector (targets: at least half of each laser's power on it).
* ``polarization``: a source linearly polarized along its local +X (at a half-wave plate's design wavelength) passes a
  half-wave plate whose fast axis is turned 22.5 deg about the beam, which turns the polarization by 45 deg, then a
  linear polarizer whose axis is turned 45 deg (target: the transmitted power equals the source power times the two
  published transmittances, within 1e-6).
* ``two_beam_interference``: one laser (780 nm, s-polarized, with a declared ``linewidth_nm`` of 0.001 nm, i.e. a
  coherence length of about 0.32 m) split into two arms by a plate beamsplitter at 45 deg (the one whose splitting
  coating is closest to 50 / 50): the transmitted arm is folded from +X into +Y and the reflected arm from +Y into +X by
  two 45 deg flat mirrors, so the arms cross at 90 deg on an observation plane whose normal bisects them. The mirror of
  the transmitted arm sits on that arm's axis behind the plate's lateral shift (Snell's law with the substrate's
  published dispersion), so both beams are centred on the plane. Targets: the fringe period ``wavelength / (n_air sqrt(2))``
  within 0.1 %, an intensity balance and a polarization overlap of at least 0.9 each (the two factors of the fringes'
  visibility the arms set), and a coherence length of at least 100 mm, thirty times the arms' path difference -- the
  last needs the source coherence and is not computed without it. The arms meet the catalog's splitter and mirror
  coatings differently: where those coatings state neither a phase nor a group-delay bound, the arms' relative group
  delay is not determined, so the visibility and the coherence margin are stated at no added delay as an estimate that
  decides no target, and they are not targets here; with published bounds they are stated within them.

Every design holds optical elements, sources, observation planes and targets.

Orientation: the design frame has +Z up, and every optical axis here is horizontal (in the design XY plane). Each pose
turns the element's local frame so that local +Z is its optical axis / emission direction / face normal and local +Y is
design +Z (up), so local +X is horizontal and ``Tx`` / ``Ty`` / ``Rx`` / ``Ry`` mean the same thing on every element:
horizontal / vertical shift and tilt about the horizontal / vertical axis. The quaternions are exact literals or built
from them with square roots, products and sums only (never libm trig):

* local +Z -> design +X, local +Y -> design +Z: ``(0.5, 0.5, 0.5, 0.5)``; turned around (local +Z -> design -X):
  ``(0.5, 0.5, -0.5, -0.5)``;
* local +Z -> design +Y, local +Y -> design +Z: ``(0, 0, sqrt(0.5), sqrt(0.5))``; turned around (local +Z -> design -Y):
  ``(sqrt(0.5), sqrt(0.5), 0, 0)``;
* local +Z -> normalize(-X + Y) (a 45 deg fold of a +X beam into +Y), local +Y -> design +Z:
  ``(a, a, -b, -b)`` with ``a = sqrt((2 - sqrt(2)) / 8)`` and ``b = sqrt((2 + sqrt(2)) / 8)``; local +Z ->
  normalize(X - Y) (a 45 deg fold of a +Y beam into +X): ``(b, b, a, a)``; local +Z -> normalize(X + Y):
  ``(a, a, b, b)``;
* a turn by ``theta`` about an element's own optical axis (a waveplate / polarizer axis): the element quaternion times
  ``(cos(theta / 2), 0, 0, sin(theta / 2))``, whose half-angle cosines and sines of 22.5 deg multiples are square-root
  expressions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal

from .._vendor.shared_contracts.design_catalog import (
    OpticalCategory,
    OpticalForm,
    PolarizationElementModelV1,
    PublicOpticalCatalogV2,
    PublicOpticalProductV2,
    SurfaceSequenceModelV1,
    TracedBranches,
)
from .._vendor.shared_contracts.optical_design import ProjectDocumentV2
from .catalog import load_public_catalog
from ..config import DEFAULT_NUMERICAL_CONFIG
from ..optics.materials.index import build_evaluator
from .compile import CompiledElement, compile_element, engine_material_id, paraxial_matrix, surrounding_index
from .errors import LocalRunErrorCode, fail

EXAMPLE_KINDS = ("focus", "folded", "beam_expander", "relay_4f", "folded_expander", "dichroic_routing", "polarization",
                 "two_beam_interference", "two_color_dichroic", "dichroic_combiner")
SQRT_HALF = math.sqrt(0.5)
# Local +Z -> design +X / -X / +Y / -Y, local +Y -> design +Z.
BEAM_ALONG_X = (0.5, 0.5, 0.5, 0.5)
BEAM_AGAINST_X = (0.5, 0.5, -0.5, -0.5)
BEAM_ALONG_Y = (0.0, 0.0, SQRT_HALF, SQRT_HALF)
BEAM_AGAINST_Y = (SQRT_HALF, SQRT_HALF, 0.0, 0.0)
# Mirror face normal (local +Z) -> normalize(-X + Y), local +Y -> design +Z: folds a +X beam into +Y at 45 deg incidence.
FOLD_MIRROR_A = math.sqrt((2.0 - math.sqrt(2.0)) / 8.0)
FOLD_MIRROR_B = math.sqrt((2.0 + math.sqrt(2.0)) / 8.0)
FOLD_X_TO_Y = (FOLD_MIRROR_A, FOLD_MIRROR_A, -FOLD_MIRROR_B, -FOLD_MIRROR_B)
# Face normal -> normalize(X - Y): folds a +Y beam into +X at 45 deg incidence.
FOLD_Y_TO_X = (FOLD_MIRROR_B, FOLD_MIRROR_B, FOLD_MIRROR_A, FOLD_MIRROR_A)
# Local +Z -> normalize(X + Y), local +Y -> design +Z: a plane whose normal bisects a +X and a +Y beam.
BISECT_X_Y = (FOLD_MIRROR_A, FOLD_MIRROR_A, FOLD_MIRROR_B, FOLD_MIRROR_B)
# Half-angle cosines / sines of turns about the optical axis: cos / sin of 11.25 deg (a 22.5 deg turn) and 22.5 deg (45 deg).
COS_22_5 = math.sqrt(2.0 + math.sqrt(2.0)) / 2.0
SIN_22_5 = math.sqrt(2.0 - math.sqrt(2.0)) / 2.0
COS_11_25 = math.sqrt((1.0 + COS_22_5) / 2.0)
SIN_11_25 = math.sqrt((1.0 - COS_22_5) / 2.0)
TURN_22_5_DEG = (COS_11_25, 0.0, 0.0, SIN_11_25)
TURN_45_DEG = (COS_22_5, 0.0, 0.0, SIN_22_5)

REFERENCE_WAVELENGTH_NM = 780.0
SOURCE_POWER_W = 0.001
SOURCE_WAIST_RADIUS_MM = 1.0
RELAY_WAIST_RADIUS_MM = 0.1
SCREEN_RADIUS_MM = 12.5
FOCUS_TOLERANCE_MM = 0.1
MAX_SPOT_RADIUS_MM = 0.05
EXPANDER_RADIUS_TOLERANCE_REL = 0.01
RELAY_RADIUS_TOLERANCE_REL = 0.02
POLARIZATION_POWER_TOLERANCE_REL = 1e-6
ROUTED_POWER_FRACTION = 0.5
REFLECTED_LINE_POWER_W = SOURCE_POWER_W  # the two lines of the two-color example
TRANSMITTED_LINE_POWER_W = SOURCE_POWER_W * 2.0
MIN_CLEAR_APERTURE_MM = Decimal(10)
FOCUS_LENS_FOCAL_LENGTH_MM = Decimal(100)
EXPANDER_FIRST_FOCAL_LENGTH_MM = Decimal(50)
EXPANDER_MIN_RATIO = Decimal("1.5")
EXPANDER_RATIO = Decimal(2)
SOURCE_TO_LENS_MM = 200.0
FOLD_LEG_MM = 150.0  # source -> fold mirror in the folded design (then 50 mm to the lens)
OUTPUT_PLANE_DISTANCE_MM = 100.0
ELEMENT_SPACING_MM = 50.0
FOLD_AOI_DEG = Decimal(45)
NORMAL_AOI_DEG = Decimal(0)
MICROMETRE_DECIMALS = 3
NM_TO_MM = 1e-6
INTERFERENCE_ARM_MM = 100.0  # beamsplitter -> each fold mirror -> the crossing plane
INTERFERENCE_LINEWIDTH_NM = 0.001  # a single-longitudinal-mode laser: coherence length ~0.32 m at 780 nm
S_POLARIZATION_DEG = 90.0  # source local +Y = design +Z: perpendicular to the horizontal plane of the two arms
SPLIT_RATIO = Decimal("0.5")
MIN_INTENSITY_BALANCE = 0.9
MIN_POLARIZATION_OVERLAP = 0.9
MIN_COHERENCE_LENGTH_MM = 100.0  # about thirty times the arms' path difference (the plate's glass in the transmitted arm)
FRINGE_PERIOD_TOLERANCE_REL = 1e-3
# Common laser lines tried, in this order, as the wavelengths of the dichroic routing example.
LASER_LINES_NM = (532, 633, 780, 1064, 405, 488, 561, 850, 980, 1550, 450, 515, 594, 660, 705, 730, 808, 905, 940, 1030, 1310)


def _mm(value: float) -> float:
    """A design coordinate rounded to the micrometre (correctly rounded decimal; no platform dependence)."""

    return round(value, MICROMETRE_DECIMALS) + 0.0


def _quat_multiply(a: tuple[float, ...], b: tuple[float, ...]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


# ------------------------------------------------------------------------------------------------ queries


def _model(product: PublicOpticalProductV2):
    return product.simulation_models[0] if len(product.simulation_models) == 1 else None


def _covers(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, wavelength_nm: Decimal, aoi_deg: Decimal) -> bool:
    """One published model, valid at ``wavelength_nm``, every coating published there and at ``aoi_deg``, a clear
    aperture of at least ``MIN_CLEAR_APERTURE_MM``."""

    model = _model(product)
    if model is None or not model.valid_band_nm.contains(wavelength_nm):
        return False
    if min(product.specifications.clear_aperture_extents_mm()) < MIN_CLEAR_APERTURE_MM:
        return False
    if isinstance(model, PolarizationElementModelV1):
        return model.valid_aoi_deg.contains(aoi_deg)
    for surface in model.surfaces:
        coating = catalog.coating(surface.coating_id) if surface.coating_id is not None else None
        if coating is not None and (coating.band_at(wavelength_nm) is None or not coating.valid_aoi_deg.contains(aoi_deg)):
            return False
    return True


def _candidates(catalog: PublicOpticalCatalogV2, category: OpticalCategory, form: OpticalForm, wavelength_nm: Decimal, aoi_deg: Decimal,
                accept: Callable[[PublicOpticalProductV2], bool] | None = None) -> list[PublicOpticalProductV2]:
    return [p for p in catalog.products
            if p.category is category and p.form is form and _covers(p, catalog, wavelength_nm, aoi_deg) and (accept is None or accept(p))]


def _unavailable(what: str, catalog: PublicOpticalCatalogV2):
    return fail(LocalRunErrorCode.EXAMPLE_UNAVAILABLE, f"public catalog {catalog.catalog_version} has no {what} for this example design")


def _focal_length(product: PublicOpticalProductV2) -> Decimal:
    return Decimal(product.specifications.focal_length_mm.value)  # type: ignore[union-attr]


def _lens(catalog: PublicOpticalCatalogV2, wavelength_nm: Decimal, focal_length_mm: Decimal, minimum_mm: Decimal = Decimal(0)) -> PublicOpticalProductV2:
    """The plano-convex lens whose specified focal length is closest to ``focal_length_mm`` (ties: ``optical_sku``)."""

    lenses = _candidates(catalog, OpticalCategory.LENS, OpticalForm.PLANO_CONVEX_LENS, wavelength_nm, NORMAL_AOI_DEG,
                         lambda p: p.specifications.focal_length_mm is not None and _focal_length(p) >= minimum_mm
                         and isinstance(_model(p), SurfaceSequenceModelV1))
    if not lenses:
        raise _unavailable(f"plano-convex lens (focal length >= {minimum_mm} mm) valid at {wavelength_nm} nm", catalog)
    return min(lenses, key=lambda p: (abs(_focal_length(p) - focal_length_mm), p.optical_sku))


def _first(catalog: PublicOpticalCatalogV2, category: OpticalCategory, form: OpticalForm, wavelength_nm: Decimal, aoi_deg: Decimal) -> PublicOpticalProductV2:
    found = _candidates(catalog, category, form, wavelength_nm, aoi_deg)
    if not found:
        raise _unavailable(f"{form.value} valid at {wavelength_nm} nm and {aoi_deg} deg incidence", catalog)
    return found[0]


# ------------------------------------------------------------------------------------------------ paraxial helpers


def _compiled(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2) -> CompiledElement:
    return compile_element("example", product, catalog)


def _thickness(product: PublicOpticalProductV2) -> float:
    model = _model(product)
    assert isinstance(model, SurfaceSequenceModelV1)
    return float(Decimal(model.surfaces[-1].vertex_z_mm) - Decimal(model.surfaces[0].vertex_z_mm))


def _matrix(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, wavelength_nm: float, *, reversed_: bool = False):
    (a, b), (c, d) = paraxial_matrix(_compiled(product, catalog), wavelength_nm)
    return ((d, b), (c, a)) if reversed_ else ((a, b), (c, d))


def _air_index(wavelength_nm: float) -> float:
    """The index of the air an example operates in (it states no environment: air at 20 C and 101325 Pa), at a vacuum
    wavelength: the paraxial matrices are relative to it (``compile.paraxial_matrix``), and a beam's wavelength in it is
    ``lambda / n_air``."""

    return surrounding_index(None, wavelength_nm)


def _rayleigh_range_mm(waist_radius_mm: float, wavelength_nm: float) -> float:
    """The Rayleigh range of a waist in the examples' air (``pi w0^2 n_air / lambda``, ``lambda`` the vacuum wavelength)."""

    return math.pi * waist_radius_mm * waist_radius_mm / (wavelength_nm / _air_index(wavelength_nm) * NM_TO_MM)


def _waist_after(matrix, q_in: complex) -> float:
    """Distance from the exit vertex to the output Gaussian waist (``q`` in air, reduced ABCD)."""

    (a, b), (c, d) = matrix
    return -((a * q_in + b) / (c * q_in + d)).real


def _front_focal_distance(matrix) -> float:
    return -matrix[1][1] / matrix[1][0]


def _back_focal_distance(matrix) -> float:
    return -matrix[0][0] / matrix[1][0]


def _afocal_gap(first, second) -> float:
    """Vertex-to-vertex gap that makes ``second * gap * first`` afocal (system C = 0)."""

    (a1, _), (c1, _) = first
    (_, _), (c2, d2) = second
    return -(c2 * a1 + d2 * c1) / (c2 * c1)


def _efl(matrix) -> float:
    return -1.0 / matrix[1][0]


# ------------------------------------------------------------------------------------------------ documents


def _pose(position_mm: list[float], quaternion_wxyz) -> dict:
    return {"position_mm": [_mm(v) for v in position_mm], "quaternion_wxyz": list(quaternion_wxyz)}


def _source(instance_id: str, position_mm: list[float], *, wavelength_nm: float = REFERENCE_WAVELENGTH_NM,
            waist_radius_mm: float = SOURCE_WAIST_RADIUS_MM, polarization_deg: float = 0.0, linewidth_nm: float | None = None,
            spectral_lines: list[tuple[float, float]] | None = None, quaternion_wxyz=BEAM_ALONG_X) -> dict:
    coherence = {} if linewidth_nm is None else {"linewidth_nm": linewidth_nm}
    if spectral_lines is None:
        emission: dict = {"wavelength_nm": wavelength_nm, "power_w": SOURCE_POWER_W}
    else:
        emission = {"spectral_lines": [{"wavelength_nm": line, "power_w": power} for line, power in spectral_lines]}
    return {
        "instance_id": instance_id,
        "optical_role": "source",
        "custom_part": {"custom_part_key": "user-laser", "description": "User-supplied laser source."},
        "nominal_pose": _pose(position_mm, quaternion_wxyz),
        "source_model": {
            "model_kind": "GAUSSIAN_BEAM",
            **emission,
            "waist_radius_mm": waist_radius_mm,
            "waist_position_mm": 0.0,
            "m2": 1.0,
            "polarization": {"kind": "LINEAR", "angle_deg": polarization_deg},
            **coherence,
        },
    }


def _element(instance_id: str, role: str, product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, position_mm: list[float],
             quaternion_wxyz, retained: tuple[tuple[str, str], ...] = ()) -> dict:
    return {
        "instance_id": instance_id,
        "optical_role": role,
        "product": {"optical_sku": product.optical_sku, "catalog_version": catalog.catalog_version},
        "nominal_pose": _pose(position_mm, quaternion_wxyz),
        "retained_dofs": [{"axis": axis, "control": control} for axis, control in retained],
    }


LENS_DOFS = (("Tx", "MANUAL"), ("Ty", "MANUAL"), ("Tz", "NUMERICAL"))
MIRROR_DOFS = (("Rx", "MANUAL"), ("Ry", "MANUAL"))
ROTATION_DOFS = (("Rz", "MANUAL"),)


def _plane(observation_id: str, position_mm: list[float], quaternion_wxyz) -> dict:
    return {"observation_id": observation_id, "pose": _pose(position_mm, quaternion_wxyz), "aperture": {"shape": "circle", "radius_mm": SCREEN_RADIUS_MM}}


def _focus_targets(observation_id: str) -> list[dict]:
    return [
        {"target_id": "focus_position", "metric": "waist_distance_x_mm", "observation_id": observation_id,
         "target_value": 0.0, "tolerance_abs": FOCUS_TOLERANCE_MM, "unit": "mm"},
        {"target_id": "spot_radius", "metric": "beam_radius_mm", "observation_id": observation_id, "maximum": MAX_SPOT_RADIUS_MM, "unit": "mm"},
    ]


def _project(components: list[dict], observations: list[dict], targets: list[dict]) -> ProjectDocumentV2:
    return ProjectDocumentV2.model_validate({"schema": "opensci.project", "project_schema_version": "2.0", "components": components,
                                             "observations": observations, "targets": targets})


def _catalog(catalog) -> PublicOpticalCatalogV2:
    return load_public_catalog(catalog)


def _focus_distance(lens: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, source_to_lens_mm: float) -> float:
    """First lens vertex -> predicted Gaussian waist of the collimated reference source behind the lens."""

    q_in = complex(source_to_lens_mm, _rayleigh_range_mm(SOURCE_WAIST_RADIUS_MM, REFERENCE_WAVELENGTH_NM))
    return _thickness(lens) + _waist_after(_matrix(lens, catalog, REFERENCE_WAVELENGTH_NM), q_in)


# ------------------------------------------------------------------------------------------------ examples


def example_project(catalog=None) -> ProjectDocumentV2:
    """``focus``: source -> plano-convex lens -> focal plane along design +X (module docstring).
    ``example_project().payload()`` is its JSON document."""

    catalog = _catalog(catalog)
    lens = _lens(catalog, Decimal(repr(REFERENCE_WAVELENGTH_NM)), FOCUS_LENS_FOCAL_LENGTH_MM)
    focus_x = _focus_distance(lens, catalog, SOURCE_TO_LENS_MM)
    return _project(
        [_source("SRC", [-SOURCE_TO_LENS_MM, 0.0, 0.0]), _element("L1", "lens", lens, catalog, [0.0, 0.0, 0.0], BEAM_ALONG_X, LENS_DOFS)],
        [_plane("focal_plane", [focus_x, 0.0, 0.0], BEAM_ALONG_X)],
        _focus_targets("focal_plane"),
    )


def example_folded_project(catalog=None) -> ProjectDocumentV2:
    """``folded``: source -> 45 deg fold mirror -> lens -> focal plane, all in the horizontal plane.

    * ``SRC``: the ``focus`` source at design x = -150 mm, emitting along design +X.
    * ``M1``: the first flat mirror (by ``optical_sku``) whose published model and coating cover 780 nm at 45 deg, at the
      origin. Its face normal (local +Z) is normalize(-X + Y), so the beam meets it at 45 deg incidence and is reflected
      from design +X into design +Y. It retains Rx and Ry (MANUAL): tip and tilt to steer the folded beam.
    * ``L1``: the ``focus`` lens at design y = 50 mm on the folded leg, local +Z along design +Y; the source-to-lens path is
      200 mm, as in ``focus``.
    * ``focal_plane``: the observation at the predicted waist on the folded leg, with the ``focus`` targets.
    """

    catalog = _catalog(catalog)
    wavelength = Decimal(repr(REFERENCE_WAVELENGTH_NM))
    lens = _lens(catalog, wavelength, FOCUS_LENS_FOCAL_LENGTH_MM)
    mirror = _first(catalog, OpticalCategory.MIRROR, OpticalForm.FLAT_MIRROR, wavelength, FOLD_AOI_DEG)
    lens_y = SOURCE_TO_LENS_MM - FOLD_LEG_MM
    focus_y = lens_y + _focus_distance(lens, catalog, SOURCE_TO_LENS_MM)
    return _project(
        [_source("SRC", [-FOLD_LEG_MM, 0.0, 0.0]), _element("M1", "mirror", mirror, catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _element("L1", "lens", lens, catalog, [0.0, lens_y, 0.0], BEAM_ALONG_Y, LENS_DOFS)],
        [_plane("focal_plane", [0.0, focus_y, 0.0], BEAM_ALONG_Y)],
        _focus_targets("focal_plane"),
    )


def _expander_lenses(catalog: PublicOpticalCatalogV2):
    wavelength = Decimal(repr(REFERENCE_WAVELENGTH_NM))
    first = _lens(catalog, wavelength, EXPANDER_FIRST_FOCAL_LENGTH_MM)
    second = _lens(catalog, wavelength, _focal_length(first) * EXPANDER_RATIO, _focal_length(first) * EXPANDER_MIN_RATIO)
    m1 = _matrix(first, catalog, REFERENCE_WAVELENGTH_NM)
    m2 = _matrix(second, catalog, REFERENCE_WAVELENGTH_NM, reversed_=True)
    gap = _mm(_afocal_gap(m1, m2))
    magnification = abs(_efl(m2) / _efl(m1))
    return first, second, gap, magnification


def _expander_target(observation_id: str, magnification: float) -> list[dict]:
    return [{"target_id": "expanded_waist", "metric": "waist_radius_x_mm", "observation_id": observation_id,
             "target_value": SOURCE_WAIST_RADIUS_MM * magnification, "tolerance_rel": EXPANDER_RADIUS_TOLERANCE_REL, "unit": "mm"}]


def example_beam_expander_project(catalog=None) -> ProjectDocumentV2:
    """``beam_expander``: source -> L1 (f ~ 50 mm) -> afocal gap -> L2 (f ~ 2 f1, turned around) -> output plane."""

    catalog = _catalog(catalog)
    first, second, gap, magnification = _expander_lenses(catalog)
    l2_entrance = _thickness(first) + gap
    l2_origin = l2_entrance + _thickness(second)  # turned around: its first vertex (the curved side) is downstream
    return _project(
        [_source("SRC", [-SOURCE_TO_LENS_MM, 0.0, 0.0]),
         _element("L1", "lens", first, catalog, [0.0, 0.0, 0.0], BEAM_ALONG_X, LENS_DOFS),
         _element("L2", "lens", second, catalog, [l2_origin, 0.0, 0.0], BEAM_AGAINST_X, LENS_DOFS)],
        [_plane("output", [l2_origin + OUTPUT_PLANE_DISTANCE_MM, 0.0, 0.0], BEAM_ALONG_X)],
        _expander_target("output", magnification),
    )


def example_folded_expander_project(catalog=None) -> ProjectDocumentV2:
    """``folded_expander``: the beam expander with a 45 deg flat fold mirror halfway between L1 and its focus."""

    catalog = _catalog(catalog)
    wavelength = Decimal(repr(REFERENCE_WAVELENGTH_NM))
    first, second, gap, magnification = _expander_lenses(catalog)
    mirror = _first(catalog, OpticalCategory.MIRROR, OpticalForm.FLAT_MIRROR, wavelength, FOLD_AOI_DEG)
    to_mirror = _mm(_back_focal_distance(_matrix(first, catalog, REFERENCE_WAVELENGTH_NM)) / 2.0)
    mirror_x = _thickness(first) + to_mirror
    l2_entrance_y = gap - to_mirror
    l2_origin_y = l2_entrance_y + _thickness(second)
    return _project(
        [_source("SRC", [-SOURCE_TO_LENS_MM, 0.0, 0.0]),
         _element("L1", "lens", first, catalog, [0.0, 0.0, 0.0], BEAM_ALONG_X, LENS_DOFS),
         _element("M1", "mirror", mirror, catalog, [mirror_x, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _element("L2", "lens", second, catalog, [mirror_x, l2_origin_y, 0.0], BEAM_AGAINST_Y, LENS_DOFS)],
        [_plane("output", [mirror_x, l2_origin_y + OUTPUT_PLANE_DISTANCE_MM, 0.0], BEAM_ALONG_Y)],
        _expander_target("output", magnification),
    )


def example_relay_4f_project(catalog=None) -> ProjectDocumentV2:
    """``relay_4f``: a 0.1 mm waist at the front focal plane of L1 (turned around) is imaged by L1 + L2 (the same product)
    onto the back focal plane of L2 at unit magnification."""

    catalog = _catalog(catalog)
    lens = _lens(catalog, Decimal(repr(REFERENCE_WAVELENGTH_NM)), FOCUS_LENS_FOCAL_LENGTH_MM)
    forward = _matrix(lens, catalog, REFERENCE_WAVELENGTH_NM)
    turned = _matrix(lens, catalog, REFERENCE_WAVELENGTH_NM, reversed_=True)
    thickness = _thickness(lens)
    source_x = -_mm(_front_focal_distance(turned))
    l1_origin = thickness  # turned around: entrance (plane side) at x = 0
    l2_entrance = _mm(thickness + _back_focal_distance(turned) + _front_focal_distance(forward))
    image_x = _mm(l2_entrance + thickness + _back_focal_distance(forward))
    targets = [
        {"target_id": "image_position", "metric": "waist_distance_x_mm", "observation_id": "image_plane",
         "target_value": 0.0, "tolerance_abs": FOCUS_TOLERANCE_MM, "unit": "mm"},
        {"target_id": "image_waist", "metric": "waist_radius_x_mm", "observation_id": "image_plane",
         "target_value": RELAY_WAIST_RADIUS_MM, "tolerance_rel": RELAY_RADIUS_TOLERANCE_REL, "unit": "mm"},
    ]
    return _project(
        [_source("SRC", [source_x, 0.0, 0.0], waist_radius_mm=RELAY_WAIST_RADIUS_MM),
         _element("L1", "lens", lens, catalog, [l1_origin, 0.0, 0.0], BEAM_AGAINST_X, LENS_DOFS),
         _element("L2", "lens", lens, catalog, [l2_entrance, 0.0, 0.0], BEAM_ALONG_X, LENS_DOFS)],
        [_plane("image_plane", [image_x, 0.0, 0.0], BEAM_ALONG_X)],
        targets,
    )


def _splitting_coating(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2):
    model = _model(product)
    if not isinstance(model, SurfaceSequenceModelV1):
        return None
    split = [s for s in model.surfaces if s.traced is TracedBranches.TRANSMITTED_AND_REFLECTED and s.coating_id is not None]
    return catalog.coating(split[0].coating_id) if len(split) == 1 else None  # type: ignore[arg-type]


def _routing_line(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, reflect: bool) -> Decimal | None:
    coating = _splitting_coating(product, catalog)
    if coating is None:
        return None
    for line in LASER_LINES_NM:
        wavelength = Decimal(line)
        band = coating.band_at(wavelength)
        if band is None or not _covers(product, catalog, wavelength, FOLD_AOI_DEG):
            continue
        fraction = Decimal(band.reflectance if reflect else band.transmittance)
        if fraction > Decimal(repr(ROUTED_POWER_FRACTION)):
            return wavelength
    return None


def _routing_dichroic(catalog: PublicOpticalCatalogV2) -> tuple[PublicOpticalProductV2, Decimal, Decimal]:
    """The first dichroic mirror (by ``optical_sku``) with a common laser line in a reflect band and one in a transmit band,
    and those two lines."""

    for product in catalog.products:
        if product.category is not OpticalCategory.FILTER or product.form is not OpticalForm.DICHROIC_MIRROR:
            continue
        reflected, transmitted = _routing_line(product, catalog, True), _routing_line(product, catalog, False)
        if reflected is not None and transmitted is not None and min(product.specifications.clear_aperture_extents_mm()) >= MIN_CLEAR_APERTURE_MM:
            return product, reflected, transmitted
    raise _unavailable("dichroic mirror with a reflect band and a transmit band at common laser lines", catalog)


def example_dichroic_routing_project(catalog=None) -> ProjectDocumentV2:
    """``dichroic_routing``: ``SRC_R`` (a laser line the dichroic reflects) and ``SRC_T`` (one it transmits), co-located at
    design x = -100 mm and both emitting along +X, meet the first suitable dichroic mirror (by ``optical_sku``) at the
    origin at 45 deg. Observations: ``reflected_leg`` 100 mm along +Y and ``transmitted_leg`` 100 mm along +X."""

    catalog = _catalog(catalog)
    product, reflected, transmitted = _routing_dichroic(catalog)
    distance = ELEMENT_SPACING_MM * 2.0
    targets = [
        {"target_id": "reflected_power", "metric": "power_W", "observation_id": "reflected_leg", "source_id": "SRC_R",
         "minimum": SOURCE_POWER_W * ROUTED_POWER_FRACTION, "unit": "W"},
        {"target_id": "transmitted_power", "metric": "power_W", "observation_id": "transmitted_leg", "source_id": "SRC_T",
         "minimum": SOURCE_POWER_W * ROUTED_POWER_FRACTION, "unit": "W"},
    ]
    return _project(
        [_source("SRC_R", [-distance, 0.0, 0.0], wavelength_nm=float(reflected)),
         _source("SRC_T", [-distance, 0.0, 0.0], wavelength_nm=float(transmitted)),
         _element("DM1", "splitter", product, catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS)],
        [_plane("reflected_leg", [0.0, distance, 0.0], BEAM_ALONG_Y), _plane("transmitted_leg", [distance, 0.0, 0.0], BEAM_ALONG_X)],
        targets,
    )


def _detector(instance_id: str, position_mm: list[float], quaternion_wxyz) -> dict:
    """The user's own detector (a ``custom_part``, no optical surfaces): its absorbing observation plane is its face."""

    return {
        "instance_id": instance_id,
        "optical_role": "detector",
        "custom_part": {"custom_part_key": "user-detector", "description": "User-supplied detector."},
        "nominal_pose": _pose(position_mm, quaternion_wxyz),
    }


def _detector_face(observation_id: str, detector_id: str) -> dict:
    """The sensitive face of ``detector_id``: an absorbing plane at the detector's own origin, normal along its local +Z."""

    return {"observation_id": observation_id, "device_instance_id": detector_id, "pose": _pose([0.0, 0.0, 0.0], (1.0, 0.0, 0.0, 0.0)),
            "aperture": {"shape": "circle", "radius_mm": SCREEN_RADIUS_MM}, "absorbing": True}


def example_two_color_dichroic_project(catalog=None) -> ProjectDocumentV2:
    """``two_color_dichroic``: ``SRC`` at design x = -100 mm emits two spectral lines along +X -- the line the dichroic
    reflects (``REFLECTED_LINE_POWER_W``) and the one it transmits (``TRANSMITTED_LINE_POWER_W``), the lines of
    ``dichroic_routing`` -- onto that example's dichroic mirror at the origin at 45 deg. ``DET_R`` 100 mm along +Y and
    ``DET_T`` 100 mm along +X are the user's own detectors facing the beam; their absorbing faces are the observations
    ``reflected_detector`` and ``transmitted_detector``. The targets select each line by ``wavelength_nm``."""

    catalog = _catalog(catalog)
    product, reflected, transmitted = _routing_dichroic(catalog)
    distance = ELEMENT_SPACING_MM * 2.0
    targets = [
        {"target_id": "reflected_line_power", "metric": "power_W", "observation_id": "reflected_detector", "source_id": "SRC",
         "wavelength_nm": float(reflected), "minimum": REFLECTED_LINE_POWER_W * ROUTED_POWER_FRACTION, "unit": "W"},
        {"target_id": "transmitted_line_power", "metric": "power_W", "observation_id": "transmitted_detector", "source_id": "SRC",
         "wavelength_nm": float(transmitted), "minimum": TRANSMITTED_LINE_POWER_W * ROUTED_POWER_FRACTION, "unit": "W"},
    ]
    return _project(
        [_source("SRC", [-distance, 0.0, 0.0], spectral_lines=[(float(reflected), REFLECTED_LINE_POWER_W),
                                                                (float(transmitted), TRANSMITTED_LINE_POWER_W)]),
         _element("DM1", "splitter", product, catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _detector("DET_R", [0.0, distance, 0.0], BEAM_ALONG_Y),
         _detector("DET_T", [distance, 0.0, 0.0], BEAM_ALONG_X)],
        [_detector_face("reflected_detector", "DET_R"), _detector_face("transmitted_detector", "DET_T")],
        targets,
    )


def _substrate_shift(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, wavelength_nm: float) -> float:
    """The lateral shift of a beam crossing the product's substrate at 45 deg (0 when no substrate is modelled)."""

    model = _model(product)
    return _plate_lateral_shift(product, catalog, wavelength_nm) if isinstance(model, SurfaceSequenceModelV1) and len(model.media) == 1 else 0.0


def example_dichroic_combiner_project(catalog=None) -> ProjectDocumentV2:
    """``dichroic_combiner``: the time reversal of ``dichroic_routing`` -- two lasers combined onto one beam by the same
    dichroic mirror at the origin at 45 deg. ``SRC_R`` (the line the dichroic reflects) at design y = +100 mm emits
    along -Y onto the coated face; ``SRC_T`` (the line it transmits) at x = +100 mm emits along -X onto the back face,
    set off in y by the substrate's lateral shift so the transmitted beam leaves on the reflected beam's axis. Both
    leave along -X onto the user's own detector ``DET`` 100 mm away, whose absorbing face is the observation
    ``combined_detector``. Targets: at least half of each laser's power on the detector."""

    catalog = _catalog(catalog)
    product, reflected, transmitted = _routing_dichroic(catalog)
    distance = ELEMENT_SPACING_MM * 2.0
    targets = [
        {"target_id": f"{name}_power", "metric": "power_W", "observation_id": "combined_detector", "source_id": source_id,
         "minimum": SOURCE_POWER_W * ROUTED_POWER_FRACTION, "unit": "W"}
        for name, source_id in (("reflected", "SRC_R"), ("transmitted", "SRC_T"))
    ]
    return _project(
        [_source("SRC_R", [0.0, distance, 0.0], wavelength_nm=float(reflected), quaternion_wxyz=BEAM_AGAINST_Y),
         _source("SRC_T", [distance, -_substrate_shift(product, catalog, float(transmitted)), 0.0], wavelength_nm=float(transmitted),
                 quaternion_wxyz=BEAM_AGAINST_X),
         _element("DM1", "splitter", product, catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _detector("DET", [-distance, 0.0, 0.0], BEAM_AGAINST_X)],
        [_detector_face("combined_detector", "DET")],
        targets,
    )


def example_polarization_project(catalog=None) -> ProjectDocumentV2:
    """``polarization``: source (linear, local +X) -> half-wave plate turned 22.5 deg -> linear polarizer turned 45 deg ->
    detector plane, along design +X at the half-wave plate's design wavelength."""

    catalog = _catalog(catalog)
    plates = [p for p in catalog.products if p.category is OpticalCategory.WAVEPLATE and p.form is OpticalForm.HALF_WAVE_PLATE
              and isinstance(_model(p), PolarizationElementModelV1)]
    for plate in plates:
        wavelength = Decimal(_model(plate).design_wavelength_nm)  # type: ignore[union-attr, arg-type]
        polarizers = _candidates(catalog, OpticalCategory.POLARIZER, OpticalForm.LINEAR_POLARIZER, wavelength, NORMAL_AOI_DEG)
        if _covers(plate, catalog, wavelength, NORMAL_AOI_DEG) and polarizers:
            polarizer = polarizers[0]
            break
    else:
        raise _unavailable("half-wave plate and linear polarizer valid at the plate's design wavelength", catalog)
    power = SOURCE_POWER_W * float(Decimal(_model(plate).transmittance.value) * Decimal(_model(polarizer).transmittance.value))  # type: ignore[union-attr]
    return _project(
        [_source("SRC", [-ELEMENT_SPACING_MM * 2.0, 0.0, 0.0], wavelength_nm=float(wavelength)),
         _element("HWP", "waveplate", plate, catalog, [0.0, 0.0, 0.0], _quat_multiply(BEAM_ALONG_X, TURN_22_5_DEG), ROTATION_DOFS),
         _element("POL", "polarizer", polarizer, catalog, [ELEMENT_SPACING_MM, 0.0, 0.0], _quat_multiply(BEAM_ALONG_X, TURN_45_DEG), ROTATION_DOFS)],
        [_plane("detector", [ELEMENT_SPACING_MM * 2.0, 0.0, 0.0], BEAM_ALONG_X)],
        [{"target_id": "transmitted_power", "metric": "power_W", "observation_id": "detector", "target_value": power,
          "tolerance_rel": POLARIZATION_POWER_TOLERANCE_REL, "unit": "W"}],
    )


def _plate_lateral_shift(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, wavelength_nm: float) -> float:
    """Lateral displacement of a beam transmitted through a plane-parallel plate at 45 deg incidence:
    ``t sin(theta) (1 - cos(theta) / sqrt(n^2 - sin^2(theta)))`` with the substrate index at ``wavelength_nm``."""

    model = _model(product)
    assert isinstance(model, SurfaceSequenceModelV1) and len(model.media) == 1
    material_id = engine_material_id(model.media[0])
    material = next(m for m in _compiled(product, catalog).materials if m.material_id == material_id)
    index, issue = build_evaluator(material, DEFAULT_NUMERICAL_CONFIG).index(wavelength_nm)
    if index is None:
        raise _unavailable(f"plate beamsplitter substrate index at {wavelength_nm} nm ({issue.message if issue else 'no data'})", catalog)
    index = index / _air_index(wavelength_nm)  # (relative to the air the beam meets the plate from, at 45 deg in that air)
    return _thickness(product) * SQRT_HALF * (1.0 - SQRT_HALF / math.sqrt(index * index - 0.5))


def _split_ratio_error(product: PublicOpticalProductV2, catalog: PublicOpticalCatalogV2, wavelength_nm: Decimal) -> Decimal | None:
    model = _model(product)
    if not isinstance(model, SurfaceSequenceModelV1) or len(model.media) != 1:
        return None
    coating = _splitting_coating(product, catalog)
    band = coating.band_at(wavelength_nm) if coating is not None else None
    return None if band is None else abs(Decimal(band.reflectance) - SPLIT_RATIO)


def example_two_beam_interference_project(catalog=None) -> ProjectDocumentV2:
    """``two_beam_interference``: SRC (780 nm, s-polarized, ``linewidth_nm`` 0.001) -> BS1 (plate beamsplitter at the
    origin, 45 deg) -> transmitted arm: M1 folds +X into +Y; reflected arm: M2 at (0, 100, 0) folds +Y into +X -> the
    arms cross at 90 deg on ``crossing_plane`` at (100, 100, 0), whose normal bisects them (module docstring)."""

    catalog = _catalog(catalog)
    wavelength = Decimal(repr(REFERENCE_WAVELENGTH_NM))
    ranked = []
    for product in _candidates(catalog, OpticalCategory.BEAMSPLITTER, OpticalForm.PLATE_BEAMSPLITTER, wavelength, FOLD_AOI_DEG):
        error = _split_ratio_error(product, catalog, wavelength)
        if error is not None:
            ranked.append((error, product.optical_sku, product))
    if not ranked:
        raise _unavailable(f"plate beamsplitter with a splitting coating at {wavelength} nm and 45 deg incidence", catalog)
    splitter = min(ranked, key=lambda entry: entry[:2])[2]
    mirror = _first(catalog, OpticalCategory.MIRROR, OpticalForm.FLAT_MIRROR, wavelength, FOLD_AOI_DEG)
    arm = INTERFERENCE_ARM_MM
    shift = _mm(_plate_lateral_shift(splitter, catalog, REFERENCE_WAVELENGTH_NM))  # the transmitted arm runs at y = -shift
    targets = [
        {"target_id": "fringe_period", "metric": "fringe_period_nm", "observation_id": "crossing_plane",
         "target_value": REFERENCE_WAVELENGTH_NM / (_air_index(REFERENCE_WAVELENGTH_NM) * math.sqrt(2.0)), "tolerance_rel": FRINGE_PERIOD_TOLERANCE_REL, "unit": "nm"},
        {"target_id": "intensity_balance", "metric": "intensity_balance", "observation_id": "crossing_plane",
         "minimum": MIN_INTENSITY_BALANCE, "unit": "1"},
        {"target_id": "polarization_overlap", "metric": "polarization_overlap", "observation_id": "crossing_plane",
         "minimum": MIN_POLARIZATION_OVERLAP, "unit": "1"},
        {"target_id": "coherence_length", "metric": "coherence_length_mm", "observation_id": "crossing_plane",
         "minimum": MIN_COHERENCE_LENGTH_MM, "unit": "mm"},
    ]
    return _project(
        [_source("SRC", [-arm, 0.0, 0.0], polarization_deg=S_POLARIZATION_DEG, linewidth_nm=INTERFERENCE_LINEWIDTH_NM),
         _element("BS1", "splitter", splitter, catalog, [0.0, 0.0, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _element("M1", "mirror", mirror, catalog, [arm, -shift, 0.0], FOLD_X_TO_Y, MIRROR_DOFS),
         _element("M2", "mirror", mirror, catalog, [0.0, arm, 0.0], FOLD_Y_TO_X, MIRROR_DOFS)],
        [_plane("crossing_plane", [arm, arm, 0.0], BISECT_X_Y)],
        targets,
    )


EXAMPLE_BUILDERS = {
    "focus": example_project,
    "folded": example_folded_project,
    "beam_expander": example_beam_expander_project,
    "relay_4f": example_relay_4f_project,
    "folded_expander": example_folded_expander_project,
    "dichroic_routing": example_dichroic_routing_project,
    "dichroic_combiner": example_dichroic_combiner_project,
    "polarization": example_polarization_project,
    "two_beam_interference": example_two_beam_interference_project,
    "two_color_dichroic": example_two_color_dichroic_project,
}


def example_project_of_kind(kind: str, catalog=None) -> ProjectDocumentV2:
    """The reference design ``kind`` (``EXAMPLE_KINDS``) built from ``catalog`` (default: the active catalog)."""

    builder = EXAMPLE_BUILDERS.get(kind)
    if builder is None:
        raise fail(LocalRunErrorCode.EXAMPLE_UNAVAILABLE, f"unknown example kind {kind!r}; known: {', '.join(EXAMPLE_KINDS)}")
    return builder(catalog)
